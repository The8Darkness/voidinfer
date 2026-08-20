#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {

enum class LogicalLaneState : std::uint8_t {
    Free,
    Active,
    Resident,
};

struct LogicalLaneSnapshot {
    LogicalLaneState state = LogicalLaneState::Free;
    AdmissionResources resources;
};

[[nodiscard]] constexpr bool resources_fit(AdmissionResources value,
                                           AdmissionResources capacity) noexcept {
    return value.active_lanes <= capacity.active_lanes &&
           value.main_kv_pages <= capacity.main_kv_pages &&
           value.backend_kv_pages <= capacity.backend_kv_pages;
}

[[nodiscard]] constexpr bool resident_within_active(AdmissionResources resident,
                                                    AdmissionResources active) noexcept {
    return resident.active_lanes == 0 && resident.main_kv_pages <= active.main_kv_pages &&
           resident.backend_kv_pages <= active.backend_kv_pages;
}

class ResourceLedger {
public:
    ResourceLedger(AdmissionResources capacity, std::uint32_t lane_count)
        : capacity_(capacity), lane_count_(lane_count) {
        if (lane_count == 0 || lane_count > kMaximumConcurrency ||
            capacity.active_lanes != lane_count) {
            throw std::invalid_argument("resource ledger capacity is invalid");
        }
    }

    [[nodiscard]] AdmissionResources capacity() const noexcept { return capacity_; }

    [[nodiscard]] AdmissionResources used() const noexcept {
        AdmissionResources out;
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            out = out + lanes_[lane].resources;
        }
        return out;
    }

    [[nodiscard]] const LogicalLaneSnapshot& lane(LaneId id) const noexcept {
        static const LogicalLaneSnapshot invalid;
        return id.value < lane_count_ ? lanes_[id.value] : invalid;
    }

    [[nodiscard]] bool adopt_active(LaneId id, AdmissionResources resources) noexcept {
        if (id.value >= lane_count_ || lanes_[id.value].state != LogicalLaneState::Free ||
            resources.active_lanes != 1 || !fits_replacing(id, {}, resources)) {
            return false;
        }
        lanes_[id.value] = {LogicalLaneState::Active, resources};
        return true;
    }

    [[nodiscard]] bool release_active(LaneId id, AdmissionResources released) noexcept {
        if (id.value >= lane_count_ || lanes_[id.value].state != LogicalLaneState::Active ||
            lanes_[id.value].resources != released) {
            return false;
        }
        lanes_[id.value] = {};
        return true;
    }

    [[nodiscard]] bool retain(LaneId id, AdmissionResources active,
                              AdmissionResources resident) noexcept {
        if (id.value >= lane_count_ || lanes_[id.value].state != LogicalLaneState::Active ||
            lanes_[id.value].resources != active || !resident_within_active(resident, active) ||
            !fits_replacing(id, active, resident)) {
            return false;
        }
        lanes_[id.value] = {LogicalLaneState::Resident, resident};
        return true;
    }

    [[nodiscard]] bool release_resident(LaneId id, AdmissionResources released) noexcept {
        if (id.value >= lane_count_ || lanes_[id.value].state != LogicalLaneState::Resident ||
            lanes_[id.value].resources != released) {
            return false;
        }
        lanes_[id.value] = {};
        return true;
    }

    void clear() noexcept {
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) { lanes_[lane] = {}; }
    }

private:
    [[nodiscard]] bool fits_replacing(LaneId id, AdmissionResources old_resources,
                                      AdmissionResources replacement) const noexcept {
        const AdmissionResources current = used();
        if (old_resources.active_lanes > current.active_lanes ||
            old_resources.main_kv_pages > current.main_kv_pages ||
            old_resources.backend_kv_pages > current.backend_kv_pages) {
            return false;
        }
        return resources_fit(
            AdmissionResources{
                .active_lanes =
                    current.active_lanes - old_resources.active_lanes + replacement.active_lanes,
                .main_kv_pages =
                    current.main_kv_pages - old_resources.main_kv_pages + replacement.main_kv_pages,
                .backend_kv_pages = current.backend_kv_pages - old_resources.backend_kv_pages +
                                    replacement.backend_kv_pages,
            },
            capacity_);
    }

    AdmissionResources capacity_;
    std::uint32_t lane_count_ = 0;
    std::array<LogicalLaneSnapshot, kMaximumConcurrency> lanes_{};
};

struct ResourceCandidateDescriptor {
    LaneId destination;
    AdmissionResources active_resources;
    AdmissionResources source_resources;
    std::uint32_t reused_prompt_tokens = 0;
};

struct ResidentResourceDescriptor {
    LaneId lane;
    AdmissionResources resources;
};

struct ResourceCandidateSelection {
    bool found                  = false;
    std::size_t candidate_index = 0;
    std::array<LaneId, kMaximumConcurrency> evictions{};
    std::size_t eviction_count = 0;
};

namespace detail {

[[nodiscard]] inline bool candidate_fits(AdmissionResources used,
                                         const ResourceCandidateDescriptor& candidate,
                                         std::span<const ResidentResourceDescriptor> evictions,
                                         AdmissionResources capacity) noexcept {
    if (candidate.source_resources.active_lanes > used.active_lanes ||
        candidate.source_resources.main_kv_pages > used.main_kv_pages ||
        candidate.source_resources.backend_kv_pages > used.backend_kv_pages) {
        return false;
    }
    std::uint64_t active  = used.active_lanes - candidate.source_resources.active_lanes;
    std::uint64_t main    = used.main_kv_pages - candidate.source_resources.main_kv_pages;
    std::uint64_t backend = used.backend_kv_pages - candidate.source_resources.backend_kv_pages;
    for (const ResidentResourceDescriptor& eviction : evictions) {
        if (eviction.resources.active_lanes > active || eviction.resources.main_kv_pages > main ||
            eviction.resources.backend_kv_pages > backend) {
            return false;
        }
        active -= eviction.resources.active_lanes;
        main -= eviction.resources.main_kv_pages;
        backend -= eviction.resources.backend_kv_pages;
    }
    active += candidate.active_resources.active_lanes;
    main += candidate.active_resources.main_kv_pages;
    backend += candidate.active_resources.backend_kv_pages;
    return active <= capacity.active_lanes && main <= capacity.main_kv_pages &&
           backend <= capacity.backend_kv_pages;
}

[[nodiscard]] inline bool better_candidate(const ResourceCandidateDescriptor& candidate,
                                           const ResourceCandidateDescriptor& selected) noexcept {
    return candidate.reused_prompt_tokens > selected.reused_prompt_tokens ||
           (candidate.reused_prompt_tokens == selected.reused_prompt_tokens &&
            candidate.destination < selected.destination);
}

} // namespace detail

[[nodiscard]] inline ResourceCandidateSelection
select_resource_candidate(std::span<const ResourceCandidateDescriptor> candidates,
                          std::span<const ResidentResourceDescriptor> residents,
                          AdmissionResources used, AdmissionResources capacity) noexcept {
    ResourceCandidateSelection selected;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (!detail::candidate_fits(used, candidates[index], {}, capacity)) { continue; }
        if (!selected.found ||
            detail::better_candidate(candidates[index], candidates[selected.candidate_index])) {
            selected                 = {};
            selected.found           = true;
            selected.candidate_index = index;
        }
    }
    if (selected.found) { return selected; }

    for (std::size_t index = 0; index < candidates.size(); ++index) {
        std::array<ResidentResourceDescriptor, kMaximumConcurrency> prefix{};
        std::size_t prefix_size = 0;
        bool fits               = false;
        for (const ResidentResourceDescriptor& resident : residents) {
            if (resident.lane == candidates[index].destination) { continue; }
            prefix[prefix_size++] = resident;
            if (detail::candidate_fits(
                    used, candidates[index],
                    std::span<const ResidentResourceDescriptor>(prefix.data(), prefix_size),
                    capacity)) {
                fits = true;
                break;
            }
        }
        if (!fits) { continue; }
        if (!selected.found ||
            detail::better_candidate(candidates[index], candidates[selected.candidate_index])) {
            selected                 = {};
            selected.found           = true;
            selected.candidate_index = index;
            selected.eviction_count  = prefix_size;
            for (std::size_t i = 0; i < prefix_size; ++i) {
                selected.evictions[i] = prefix[i].lane;
            }
        }
    }
    return selected;
}

template <class Package>
class ResourceManager {
public:
    using Program            = typename Package::Program;
    using PreparedPrompt     = typename Package::PreparedPrompt;
    using RequestBasePlan    = typename Package::RequestBasePlan;
    using AdmissionPlan      = typename Package::AdmissionPlan;
    using SequenceHandle     = typename Package::SequenceHandle;
    using ContinuationHandle = typename Package::ContinuationHandle;
    using StartResult        = typename Package::StartResult;
    using FinishResult       = typename Package::FinishResult;
    using AbortResult        = typename Package::AbortResult;

    class Choice {
    public:
        Choice(Choice&&) noexcept        = default;
        Choice& operator=(Choice&&)      = delete;
        Choice(const Choice&)            = delete;
        Choice& operator=(const Choice&) = delete;

        [[nodiscard]] const RequestPlanSummary& summary() const noexcept {
            return plan_->summary();
        }

        [[nodiscard]] LaneId destination() const noexcept { return destination_; }

    private:
        Choice(LaneId destination, AdmissionPlan&& plan) : destination_(destination) {
            plan_.emplace(std::move(plan));
        }

        LaneId destination_{};
        std::optional<AdmissionPlan> plan_;
        std::uint64_t source_id_       = 0;
        std::uint64_t source_revision_ = 0;
        std::array<std::uint64_t, kMaximumConcurrency> eviction_ids_{};
        std::array<std::uint64_t, kMaximumConcurrency> eviction_revisions_{};
        std::array<LaneId, kMaximumConcurrency> evictions_{};
        std::size_t eviction_count_ = 0;

        friend class ResourceManager;
    };

    ResourceManager(AdmissionResources capacity, std::uint32_t lane_count)
        : ledger_(capacity, lane_count), lane_count_(lane_count) {}

    [[nodiscard]] std::optional<Choice> inspect(Program& program, const PreparedPrompt& prompt,
                                                const RequestBasePlan& base) {
        std::array<std::optional<AdmissionPlan>, kMaximumConcurrency> plans{};
        std::array<ResourceCandidateDescriptor, kMaximumConcurrency> candidates{};
        std::array<std::uint32_t, kMaximumConcurrency> candidate_lanes{};
        std::size_t candidate_count = 0;
        std::array<ResidentResourceDescriptor, kMaximumConcurrency> residents{};
        std::size_t resident_count = 0;

        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            const LaneId id{lane};
            const LogicalLaneSnapshot& logical = ledger_.lane(id);
            if (logical.state == LogicalLaneState::Resident) {
                residents[resident_count++] = {id, logical.resources};
            }
            if (logical.state == LogicalLaneState::Active) { continue; }
            const ContinuationHandle* source =
                logical.state == LogicalLaneState::Resident ? &*catalog_[lane].handle : nullptr;
            plans[candidate_count].emplace(program.inspect_admission(prompt, base, id, source));
            const RequestPlanSummary& summary = plans[candidate_count]->summary();
            candidates[candidate_count]       = ResourceCandidateDescriptor{
                      .destination          = id,
                      .active_resources     = summary.admission,
                      .source_resources     = logical.state == LogicalLaneState::Resident
                                                  ? logical.resources
                                                  : AdmissionResources{},
                      .reused_prompt_tokens = summary.reusable_prompt_tokens,
            };
            candidate_lanes[candidate_count] = lane;
            ++candidate_count;
        }

        const ResourceCandidateSelection selected = select_resource_candidate(
            std::span<const ResourceCandidateDescriptor>(candidates.data(), candidate_count),
            std::span<const ResidentResourceDescriptor>(residents.data(), resident_count),
            ledger_.used(), ledger_.capacity());
        if (!selected.found) { return std::nullopt; }

        const std::uint32_t lane = candidate_lanes[selected.candidate_index];
        Choice choice(LaneId{lane}, std::move(*plans[selected.candidate_index]));
        if (ledger_.lane(LaneId{lane}).state == LogicalLaneState::Resident) {
            choice.source_id_       = catalog_[lane].id;
            choice.source_revision_ = catalog_[lane].revision;
        }
        choice.eviction_count_ = selected.eviction_count;
        for (std::size_t i = 0; i < selected.eviction_count; ++i) {
            const std::uint32_t eviction_lane = selected.evictions[i].value;
            choice.evictions_[i]              = selected.evictions[i];
            choice.eviction_ids_[i]           = catalog_[eviction_lane].id;
            choice.eviction_revisions_[i]     = catalog_[eviction_lane].revision;
        }
        return std::optional<Choice>(std::move(choice));
    }

    [[nodiscard]] StartResult start(Program& program, Choice&& choice, PreparedPrompt&& prompt) {
        validate_choice(choice);
        const AdmissionResources expected = choice.summary().admission;
        for (std::size_t i = 0; i < choice.eviction_count_; ++i) {
            const LaneId lane                 = choice.evictions_[i];
            CatalogEntry& entry               = catalog_[lane.value];
            const AdmissionResources resident = ledger_.lane(lane).resources;
            auto released = program.release_continuation(std::move(*entry.handle));
            entry.handle.reset();
            if (released.status != ConsumeStatus::Consumed ||
                released.released_resources != resident ||
                !ledger_.release_resident(lane, resident)) {
                throw std::logic_error("continuation eviction violated the resource ledger");
            }
            clear_catalog_entry(entry);
        }

        const LaneId destination = choice.destination_;
        std::optional<ContinuationHandle> source;
        CatalogEntry& destination_entry = catalog_[destination.value];
        if (choice.source_id_ != 0) {
            const AdmissionResources resident = ledger_.lane(destination).resources;
            source.emplace(std::move(*destination_entry.handle));
            destination_entry.handle.reset();
            if (!ledger_.release_resident(destination, resident)) {
                throw std::logic_error("continuation claim violated the resource ledger");
            }
            clear_catalog_entry(destination_entry);
        }

        StartResult result =
            program.start_request(std::move(*choice.plan_), std::move(prompt), std::move(source));
        choice.plan_.reset();
        if (result.active_resources != expected ||
            !ledger_.adopt_active(destination, result.active_resources)) {
            const auto cleanup = program.abort(result.sequence);
            (void)cleanup;
            throw std::logic_error("Runtime start violated the selected resource entitlement");
        }
        return result;
    }

    [[nodiscard]] FinishResult finish(Program& program, LaneId lane, SequenceHandle sequence,
                                      RetentionDecision decision) {
        const AdmissionResources active = require_active(lane);
        FinishResult result             = program.finish(sequence, decision);
        if (result.status != ConsumeStatus::Consumed || result.released_resources != active) {
            throw std::logic_error("Runtime finish did not consume the active entitlement");
        }
        if (decision == RetentionDecision::RetainResident) {
            if (!result.continuation ||
                !resident_within_active(result.resident_resources, active) ||
                !ledger_.retain(lane, active, result.resident_resources)) {
                if (result.continuation) {
                    (void)program.release_continuation(std::move(*result.continuation));
                    result.continuation.reset();
                }
                throw std::logic_error("Runtime continuation violated the resident entitlement");
            }
            CatalogEntry& entry = catalog_[lane.value];
            entry.id            = next_continuation_id_++;
            ++entry.revision;
            entry.handle.emplace(std::move(*result.continuation));
            result.continuation.reset();
        } else {
            if (result.continuation || result.resident_resources != AdmissionResources{} ||
                !ledger_.release_active(lane, active)) {
                throw std::logic_error("Runtime release finish violated the resource ledger");
            }
            ++catalog_[lane.value].revision;
        }
        return result;
    }

    [[nodiscard]] AbortResult abort(Program& program, LaneId lane, SequenceHandle sequence) {
        const AdmissionResources active = require_active(lane);
        AbortResult result              = program.abort(sequence);
        if (result.status != ConsumeStatus::Consumed || result.released_resources != active ||
            !ledger_.release_active(lane, active)) {
            throw std::logic_error("Runtime abort violated the resource ledger");
        }
        ++catalog_[lane.value].revision;
        return result;
    }

    void apply_commit(std::span<const LaneId> lanes, const typename Package::CommitResult& result) {
        if (lanes.size() != result.row_count) {
            throw std::logic_error("commit result membership is not row aligned");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const AdmissionResources active = require_active(lanes[row]);
            if (result.rows[row].disposition == CommitDisposition::CancelledReleased &&
                result.rows[row].released_resources != active) {
                throw std::logic_error("cancelled commit did not release its active entitlement");
            }
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (result.rows[row].disposition != CommitDisposition::CancelledReleased) { continue; }
            const AdmissionResources active = ledger_.lane(lanes[row]).resources;
            if (!ledger_.release_active(lanes[row], active)) {
                throw std::logic_error("cancelled commit ledger transition failed");
            }
            ++catalog_[lanes[row].value].revision;
        }
    }

    void apply_discard(std::span<const LaneId> lanes,
                       const typename Package::DiscardResult& result) {
        if (lanes.size() != result.row_count || result.status != ConsumeStatus::Consumed) {
            throw std::logic_error("pending discard did not consume its membership");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (result.released_resources[row] != require_active(lanes[row])) {
                throw std::logic_error("pending discard release acknowledgement is invalid");
            }
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const AdmissionResources active = ledger_.lane(lanes[row]).resources;
            (void)ledger_.release_active(lanes[row], active);
            ++catalog_[lanes[row].value].revision;
        }
    }

    void release_failed_commit(std::span<const LaneId> lanes) noexcept {
        for (const LaneId lane : lanes) {
            if (lane.value >= lane_count_ || ledger_.lane(lane).state != LogicalLaneState::Active) {
                continue;
            }
            const AdmissionResources active = ledger_.lane(lane).resources;
            (void)ledger_.release_active(lane, active);
            ++catalog_[lane.value].revision;
        }
    }

    [[nodiscard]] const ResourceLedger& ledger() const noexcept { return ledger_; }

    void clear_after_program_cleanup() noexcept {
        for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
            catalog_[lane].handle.reset();
            clear_catalog_entry(catalog_[lane]);
        }
        ledger_.clear();
    }

private:
    struct CatalogEntry {
        std::uint64_t id       = 0;
        std::uint64_t revision = 1;
        std::optional<ContinuationHandle> handle;
    };

    static void clear_catalog_entry(CatalogEntry& entry) noexcept {
        entry.id = 0;
        ++entry.revision;
    }

    [[nodiscard]] AdmissionResources require_active(LaneId lane) const {
        if (lane.value >= lane_count_ || ledger_.lane(lane).state != LogicalLaneState::Active) {
            throw std::logic_error("resource lane is not active");
        }
        return ledger_.lane(lane).resources;
    }

    void validate_choice(const Choice& choice) const {
        if (!choice.plan_ || choice.destination_.value >= lane_count_) {
            throw std::logic_error("admission choice is empty");
        }
        const CatalogEntry& source = catalog_[choice.destination_.value];
        if (choice.source_id_ == 0) {
            if (ledger_.lane(choice.destination_).state != LogicalLaneState::Free) {
                throw std::logic_error("admission destination changed after inspection");
            }
        } else if (ledger_.lane(choice.destination_).state != LogicalLaneState::Resident ||
                   source.id != choice.source_id_ || source.revision != choice.source_revision_ ||
                   !source.handle) {
            throw std::logic_error("admission source changed after inspection");
        }
        for (std::size_t i = 0; i < choice.eviction_count_; ++i) {
            const CatalogEntry& entry = catalog_[choice.evictions_[i].value];
            if (ledger_.lane(choice.evictions_[i]).state != LogicalLaneState::Resident ||
                entry.id != choice.eviction_ids_[i] ||
                entry.revision != choice.eviction_revisions_[i] || !entry.handle) {
                throw std::logic_error("admission eviction changed after inspection");
            }
        }
    }

    ResourceLedger ledger_;
    std::uint32_t lane_count_ = 0;
    std::array<CatalogEntry, kMaximumConcurrency> catalog_{};
    std::uint64_t next_continuation_id_ = 1;
};

} // namespace ninfer::runtime
