#pragma once

#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::exl3 {

enum class Exl3ReadyNumericalStage : std::uint8_t {
    HostPreparation,
    TargetProjection,
    PrivateDraftProjection,
    Verification,
    GreedyDecision,
};

enum class Exl3ReadyArithmetic : std::uint8_t {
    OrdinaryFp16B8GreedyText,
};

enum class Exl3ReadyModality : std::uint8_t {
    Text,
};

enum class Exl3ReadyWorkBlock : std::uint8_t {
    None,
    StaleAcquisition,
    BlockedDependency,
    MissingPhysicalLane,
    MissingResource,
    Cancelled,
};

struct Exl3ReadyWorkResources {
    std::uint64_t logical_host_bytes=0;
    std::uint64_t request_local_device_bytes=0;
    std::uint32_t physical_lane_slots=1;
    bool requires_preallocated_context=true;

    friend bool operator==(const Exl3ReadyWorkResources&,
        const Exl3ReadyWorkResources&) noexcept=default;
};

// Optional supplied scheduling information, never a resource entitlement or a
// measured-memory claim. Unknown bytes remain null. A positive known demand may
// identify an immutable history owner and a finite interval during which an
// equivalent feasible ordering may avoid duplicating its active transfer.
struct Exl3ReadyCopyDemand {
    static constexpr auto maximum_fairness_deferral=std::chrono::seconds(1);
    std::shared_ptr<const void> history_owner;
    std::optional<std::uint64_t> transfer_bytes;
    std::chrono::microseconds maximum_deferral{};

    bool valid() const noexcept {
        if(!transfer_bytes)return maximum_deferral==std::chrono::microseconds::zero();
        if(!*transfer_bytes)return maximum_deferral==std::chrono::microseconds::zero();
        return history_owner && maximum_deferral>std::chrono::microseconds::zero() &&
            maximum_deferral<=maximum_fairness_deferral;
    }
    bool actionable() const noexcept {
        return valid() && transfer_bytes && *transfer_bytes>0;
    }
    bool same_history(const Exl3ReadyCopyDemand& other) const noexcept {
        return history_owner && other.history_owner &&
            !history_owner.owner_before(other.history_owner) &&
            !other.history_owner.owner_before(history_owner);
    }
    bool same_as(const Exl3ReadyCopyDemand& other) const noexcept {
        const bool same_owner=(!history_owner && !other.history_owner) || same_history(other);
        return same_owner && transfer_bytes==other.transfer_bytes &&
            maximum_deferral==other.maximum_deferral;
    }
};

// A compatible signature is categorical and owner based. It contains no
// pointers into a Request and does not infer compatibility from row count,
// prompt length, or an optimistic numerical alternative.
struct Exl3ReadyExecutionSignature {
    std::shared_ptr<const void> model_owner;
    int device=-1;
    Exl3ReadyNumericalStage stage=Exl3ReadyNumericalStage::HostPreparation;
    Exl3ReadyArithmetic arithmetic=Exl3ReadyArithmetic::OrdinaryFp16B8GreedyText;
    Exl3ReadyModality modality=Exl3ReadyModality::Text;

    static bool same_owner(const std::shared_ptr<const void>& left,
        const std::shared_ptr<const void>& right) noexcept {
        return left && right && !left.owner_before(right) && !right.owner_before(left);
    }
    bool compatible_with(const Exl3ReadyExecutionSignature& other) const noexcept {
        return same_owner(model_owner,other.model_owner) && device==other.device &&
            stage==other.stage && arithmetic==other.arithmetic && modality==other.modality;
    }
};

// Current queue/acquisition facts are deliberately separate from the immutable
// descriptor. Toggling cancellation or losing a lane therefore cannot silently
// rewrite the work's numerical/resource identity.
struct Exl3ReadyWorkState {
    std::uint64_t observed_generation=0;
    bool dependency_ready=false;
    bool physical_lane_available=false;
    bool required_resources_available=false;
    bool cancelled=false;
};

struct Exl3ReadyWorkAssessment {
    bool logically_admitted=true;
    bool physically_executable=false;
    Exl3ReadyWorkBlock blocked=Exl3ReadyWorkBlock::BlockedDependency;
};

inline Exl3ReadyWorkAssessment exl3_assess_ready_work(
    std::uint64_t generation,const Exl3ReadyWorkState& state) noexcept {
    if(state.cancelled)return {true,false,Exl3ReadyWorkBlock::Cancelled};
    if(state.observed_generation!=generation)
        return {true,false,Exl3ReadyWorkBlock::StaleAcquisition};
    if(!state.dependency_ready)
        return {true,false,Exl3ReadyWorkBlock::BlockedDependency};
    if(!state.physical_lane_available)
        return {true,false,Exl3ReadyWorkBlock::MissingPhysicalLane};
    if(!state.required_resources_available)
        return {true,false,Exl3ReadyWorkBlock::MissingResource};
    return {true,true,Exl3ReadyWorkBlock::None};
}

struct Exl3ReadyWorkObservation {
    std::uint64_t generation=0;
    std::uint64_t input_fingerprint=0;
    std::size_t input_tokens=0;
    std::chrono::steady_clock::time_point submitted{};
    Exl3ReadyWorkResources resources;
    Exl3ReadyExecutionSignature signature;
    Exl3ReadyCopyDemand copy_demand;

    Exl3ReadyWorkAssessment assess(const Exl3ReadyWorkState& state) const noexcept {
        return exl3_assess_ready_work(generation,state);
    }
};

struct Exl3ReadyLocalityCost {
    std::optional<std::uint64_t> root_attachment_microseconds;
    std::optional<std::uint64_t> history_restore_microseconds;

    std::optional<std::uint64_t> total_microseconds() const noexcept {
        if(!root_attachment_microseconds || !history_restore_microseconds ||
            *root_attachment_microseconds>std::numeric_limits<std::uint64_t>::max()-
                *history_restore_microseconds)return std::nullopt;
        return *root_attachment_microseconds+*history_restore_microseconds;
    }
};

struct Exl3ReadyLaneLocality {
    std::size_t lane=0;
    bool physical_resources_feasible=false;
    bool identity_compatible=false;
    std::size_t reusable_tokens=0;
    Exl3ReadyLocalityCost cost;
};

// Hard resource/identity gates always run before optional locality estimates.
// If any feasible lane is unmeasured, no estimate wins by treating unknown as
// zero or infinity: exact longest-prefix and stable lane order are retained.
inline std::optional<std::size_t> exl3_select_ready_lane(
    std::span<const Exl3ReadyLaneLocality> candidates) noexcept {
    std::optional<std::size_t> fallback;
    bool all_costs_known=true;
    for(std::size_t index=0;index<candidates.size();++index) {
        const auto& candidate=candidates[index];
        if(!candidate.physical_resources_feasible || !candidate.identity_compatible)continue;
        if(!fallback || candidate.reusable_tokens>candidates[*fallback].reusable_tokens)
            fallback=index;
        if(!candidate.cost.total_microseconds())all_costs_known=false;
    }
    if(!fallback || !all_costs_known)return fallback?std::optional<std::size_t>{
        candidates[*fallback].lane}:std::nullopt;
    auto selected=*fallback;
    auto selected_cost=*candidates[selected].cost.total_microseconds();
    for(std::size_t index=0;index<candidates.size();++index) {
        const auto& candidate=candidates[index];
        if(!candidate.physical_resources_feasible || !candidate.identity_compatible)continue;
        const auto cost=*candidate.cost.total_microseconds();
        if(cost<selected_cost || (cost==selected_cost &&
            candidate.reusable_tokens>candidates[selected].reusable_tokens)) {
            selected=index;selected_cost=cost;
        }
    }
    return candidates[selected].lane;
}

struct Exl3ReadyRequestCandidate {
    std::size_t queue_index=0;
    std::chrono::steady_clock::time_point submitted{};
    bool physically_eligible=false;
    bool cancelled=false;
};

struct Exl3ReadyCopyCandidate {
    Exl3ReadyRequestCandidate request;
    Exl3ReadyCopyDemand demand;
    bool duplicates_active_history=false;
};

inline bool exl3_ready_request_precedes(const Exl3ReadyRequestCandidate& candidate,
    const Exl3ReadyRequestCandidate& selected) noexcept {
    return candidate.submitted<selected.submitted ||
        (candidate.submitted==selected.submitted && candidate.queue_index<selected.queue_index);
}

// Oldest eligible submission wins; queue position is the stable tie-breaker.
// Ineligible and cancelled work cannot block a feasible peer, while repeated
// newer arrivals can never overtake an older request once it is eligible.
inline std::optional<std::size_t> exl3_select_fair_ready_request(
    std::span<const Exl3ReadyRequestCandidate> candidates) noexcept {
    const Exl3ReadyRequestCandidate* selected=nullptr;
    for(const auto& candidate:candidates) {
        if(!candidate.physically_eligible || candidate.cancelled)continue;
        if(!selected || exl3_ready_request_precedes(candidate,*selected))
            selected=&candidate;
    }
    return selected?std::optional<std::size_t>{selected->queue_index}:std::nullopt;
}

// T244 fairness remains authoritative. Copy pressure is consulted only when
// every feasible candidate has a supplied value and the oldest candidate is a
// known duplicate of an active history transfer. It may then choose the oldest
// known nonduplicate alternative until the oldest request's finite bound.
inline std::optional<std::size_t> exl3_select_copy_aware_ready_request(
    std::span<const Exl3ReadyCopyCandidate> candidates,
    std::chrono::steady_clock::time_point now) noexcept {
    const Exl3ReadyCopyCandidate* oldest=nullptr;
    const Exl3ReadyCopyCandidate* alternative=nullptr;
    bool all_known=true;
    for(const auto& candidate:candidates) {
        if(!candidate.request.physically_eligible || candidate.request.cancelled)continue;
        if(!candidate.demand.transfer_bytes)all_known=false;
        if(!oldest || exl3_ready_request_precedes(candidate.request,oldest->request))
            oldest=&candidate;
        if(!candidate.duplicates_active_history &&
           (!alternative || exl3_ready_request_precedes(candidate.request,alternative->request)))
            alternative=&candidate;
    }
    if(!oldest || !all_known || !oldest->duplicates_active_history ||
       !oldest->demand.actionable() || !alternative)return oldest?
        std::optional<std::size_t>{oldest->request.queue_index}:std::nullopt;
    const auto age=now<=oldest->request.submitted?std::chrono::microseconds::zero():
        std::chrono::duration_cast<std::chrono::microseconds>(now-oldest->request.submitted);
    return age>=oldest->demand.maximum_deferral?
        std::optional<std::size_t>{oldest->request.queue_index}:
        std::optional<std::size_t>{alternative->request.queue_index};
}

enum class Exl3ReadyDecisionReason : std::uint8_t {
    OldestFeasible,
    ExactLocality,
    CopyNonduplicate,
    UnknownCopyFallback,
    CopyFairnessOverride,
};

// Optional bounded provenance for an actual queue-to-active decision. It logs
// descriptor identity and aggregate policy outcomes only: never prompt tokens,
// text, pointers, transfer-byte values or elapsed numerical time.
struct Exl3ReadyDecisionRecord {
    std::uint64_t authority_revision=0;
    std::uint64_t generation=0;
    std::uint64_t input_fingerprint=0;
    std::uint32_t queue_size=0;
    std::uint32_t eligible=0;
    std::uint32_t eligible_deferred=0;
    std::uint32_t cancelled_rejected=0;
    std::uint32_t physical_rejected=0;
    std::uint32_t identity_rejected=0;
    std::uint32_t affinity_deferred=0;
    std::uint8_t lane=0;
    bool affinity_enabled=false;
    bool locality_costs_complete=false;
    Exl3ReadyDecisionReason reason=Exl3ReadyDecisionReason::OldestFeasible;

    bool count_conserved() const noexcept {
        return eligible && eligible_deferred==eligible-1 &&
            std::uint64_t(eligible)+cancelled_rejected+physical_rejected+
                identity_rejected+affinity_deferred==queue_size;
    }
};

class Exl3ReadyDecisionTrace {
public:
    static constexpr std::size_t capacity=64;
    struct Snapshot {
        std::array<Exl3ReadyDecisionRecord,capacity> records{};
        std::size_t count=0;
        std::uint64_t total=0;
    };

    void append(Exl3ReadyDecisionRecord record) noexcept {
        records_[next_]=record;
        next_=(next_+1)%capacity;
        if(total_!=std::numeric_limits<std::uint64_t>::max())++total_;
        if(count_<capacity)++count_;
    }
    Snapshot snapshot() const noexcept {
        Snapshot result;result.count=count_;result.total=total_;
        const auto start=count_==capacity?next_:0;
        for(std::size_t index=0;index<count_;++index)
            result.records[index]=records_[(start+index)%capacity];
        return result;
    }
    void reset() noexcept {records_={};next_=0;count_=0;total_=0;}

private:
    std::array<Exl3ReadyDecisionRecord,capacity> records_{};
    std::size_t next_=0,count_=0;
    std::uint64_t total_=0;
};

// Allocation-free snapshot authority used under the existing Engine mutex.
// A lane may have one newest issued ticket; a later assessment or any policy/
// queue revision invalidates an earlier ticket. Consumption rechecks the exact
// descriptor generation and all mutable readiness facts before the caller may
// leave the mutex for numerical work.
class Exl3ReadyWorkSnapshotAuthority {
public:
    static constexpr std::size_t maximum_lanes=2;
    struct Ticket {
        Ticket(const Ticket&)=delete;
        Ticket& operator=(const Ticket&)=delete;
        Ticket(Ticket&& other) noexcept
            :observation(std::move(other.observation)),assessment(other.assessment),
             revision(other.revision),nonce(other.nonce),lane(other.lane) {
            other.nonce=0;
        }
        Ticket& operator=(Ticket&&)=delete;
        Exl3ReadyWorkObservation observation;
        Exl3ReadyWorkAssessment assessment;
        std::uint64_t revision=0,nonce=0;
        std::size_t lane=0;
    private:
        Ticket(Exl3ReadyWorkObservation value,Exl3ReadyWorkAssessment state,
            std::uint64_t source_revision,std::uint64_t source_nonce,std::size_t source_lane)
            :observation(std::move(value)),assessment(state),revision(source_revision),
             nonce(source_nonce),lane(source_lane) {}
        friend class Exl3ReadyWorkSnapshotAuthority;
    };
    struct Grant {
        Exl3ReadyWorkObservation observation;
        std::uint64_t revision=0,nonce=0;
        std::size_t lane=0;
    };

    std::uint64_t revision() const noexcept {return revision_;}
    void revise() noexcept {
        if(revision_==std::numeric_limits<std::uint64_t>::max())exhausted_=true;
        else ++revision_;
    }
    Ticket snapshot(Exl3ReadyWorkObservation observation,
        const Exl3ReadyWorkState& state,std::size_t lane) {
        if(exhausted_ || next_nonce_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("EXL3 ready-work snapshot authority exhausted");
        if(lane>=maximum_lanes)
            throw std::invalid_argument("EXL3 ready-work snapshot lane");
        const auto nonce=next_nonce_++;
        issued_[lane]=nonce;
        const auto assessment=exl3_assess_ready_work(observation.generation,state);
        return Ticket(std::move(observation),assessment,revision_,nonce,lane);
    }
    std::optional<Grant> consume(Ticket& ticket,
        const Exl3ReadyWorkObservation& current,const Exl3ReadyWorkState& state) noexcept {
        if(!ticket.nonce || ticket.lane>=maximum_lanes ||
            issued_[ticket.lane]!=ticket.nonce)return std::nullopt;
        issued_[ticket.lane]=0;
        const bool current_ready=current.assess(state).physically_executable;
        const bool same_signature=ticket.observation.signature.compatible_with(current.signature);
        const bool same_copy_demand=ticket.observation.copy_demand.same_as(current.copy_demand);
        if(exhausted_ || ticket.revision!=revision_ || !ticket.assessment.physically_executable ||
            !current_ready || ticket.observation.generation!=current.generation ||
            ticket.observation.input_fingerprint!=current.input_fingerprint ||
            ticket.observation.input_tokens!=current.input_tokens ||
            ticket.observation.submitted!=current.submitted ||
            ticket.observation.resources!=current.resources || !same_signature || !same_copy_demand) {
            ticket.nonce=0;return std::nullopt;
        }
        Grant result{std::move(ticket.observation),ticket.revision,ticket.nonce,ticket.lane};
        ticket.nonce=0;return result;
    }

private:
    std::uint64_t revision_=1,next_nonce_=1;
    std::array<std::uint64_t,maximum_lanes> issued_{};
    bool exhausted_=false;
};

// Immutable once published to the request queue. It owns the exact converted
// input tokens used by affinity and execution, replacing the former span into
// mutable Request storage. Logical admission is represented by publication of
// this object; assess() describes whether a current physical snapshot may run
// the represented stage.
class Exl3ReadyWorkDescriptor {
public:
    using Clock=std::chrono::steady_clock;

    Exl3ReadyWorkDescriptor(std::uint64_t generation,Clock::time_point submitted,
        std::vector<std::int64_t> input_tokens,std::uint64_t input_fingerprint,
        Exl3ReadyWorkResources resources,Exl3ReadyExecutionSignature signature,
        Exl3ReadyCopyDemand copy_demand={})
        :generation_(generation),submitted_(submitted),input_tokens_(std::move(input_tokens)),
         input_fingerprint_(input_fingerprint),resources_(resources),signature_(std::move(signature)),
         copy_demand_(std::move(copy_demand)) {
        if(!generation_ || !input_fingerprint_ ||
            !resources_.logical_host_bytes || resources_.physical_lane_slots!=1 ||
            !signature_.model_owner || signature_.device<0 || !copy_demand_.valid())
            throw std::invalid_argument("EXL3 ready-work descriptor extent");
    }

    std::uint64_t generation() const noexcept {return generation_;}
    Clock::time_point submitted() const noexcept {return submitted_;}
    std::span<const std::int64_t> input_tokens() const noexcept {return input_tokens_;}
    std::uint64_t input_fingerprint() const noexcept {return input_fingerprint_;}
    const Exl3ReadyWorkResources& resources() const noexcept {return resources_;}
    const Exl3ReadyExecutionSignature& signature() const noexcept {return signature_;}
    const Exl3ReadyCopyDemand& copy_demand() const noexcept {return copy_demand_;}
    std::chrono::microseconds age(Clock::time_point now) const noexcept {
        if(now<=submitted_)return std::chrono::microseconds::zero();
        return std::chrono::duration_cast<std::chrono::microseconds>(now-submitted_);
    }
    bool compatible_with(const Exl3ReadyWorkDescriptor& other) const noexcept {
        return signature_.compatible_with(other.signature_);
    }
    Exl3ReadyWorkAssessment assess(const Exl3ReadyWorkState& state) const noexcept {
        return exl3_assess_ready_work(generation_,state);
    }
    Exl3ReadyWorkObservation observation() const {
        return {generation_,input_fingerprint_,input_tokens_.size(),submitted_,resources_,signature_,copy_demand_};
    }

private:
    const std::uint64_t generation_;
    const Clock::time_point submitted_;
    const std::vector<std::int64_t> input_tokens_;
    const std::uint64_t input_fingerprint_;
    const Exl3ReadyWorkResources resources_;
    const Exl3ReadyExecutionSignature signature_;
    const Exl3ReadyCopyDemand copy_demand_;
};

} // namespace ninfer::exl3
