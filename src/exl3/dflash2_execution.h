#pragma once

#include "exl3/vericache_serving_coordinator.h"
#include "exl3/suffix_proposer.h"
#include "exl3/verifier_horizon_policy.h"
#include "exl3/retirement_state.h"
#include "exl3/request_completion_event_pool.h"
#include "exl3/execution_dependency_graph.h"
#include "exl3/bounded_shared_owner.h"
#include "exl3/pending_payload_extent.h"
#include <new>
#include <atomic>
#include <functional>

namespace ninfer::exl3 {

// A physical C1 execution resource, never a scheduler. The caller obtains a
// coordinator lease first. It owns the context and fixed staging across turns;
// the resident drafter is exclusively borrowed while a request is acquired.
// Separate draft.create_execution() resources may share immutable weights;
// sharing one mutable draft object between lanes remains unsupported. This
// default remains stream0. An explicit caller-owned stream must outlive this
// resource; C2 admission/overlap require their own qualification.
class Exl3Dflash2Execution {
    struct FailedRetirement {
        std::shared_ptr<const void> stream;
        std::shared_ptr<Exl3Dflash2DraftModel> draft;
        std::shared_ptr<Exl3TextContext> context;
        std::array<std::uint16_t*,5> staging{};
        std::array<std::int64_t,8> proposal_block{};
        bool proposal_active=false;
        std::shared_ptr<const Exl3VeriCacheRequest> proposal_root;
        std::array<cudaEvent_t,Exl3RequestCompletionEventPool::capacity> completion_events{};
        Exl3RequestCompletionEventPool completion_pool;
        Exl3ExecutionDependencyGraph dependency_graph;
        Exl3RetirementResult retirement;
        std::uint64_t acquisition=0,execution=0,round=0;
        std::uint64_t context_device_bytes=0,staging_device_bytes=0;
        std::unique_lock<std::mutex> draft_lock;
        std::array<std::shared_ptr<const void>,4> request_owners;
        FailedRetirement* next=nullptr;
    };
    inline static std::atomic<FailedRetirement*> failed_retirements_{nullptr};
    inline static std::atomic<std::uint64_t> failed_retirement_count_{0};
    inline static std::atomic<std::uint64_t> completion_authority_counter_{0};
    static std::uint64_t acquire_completion_authority_generation() {
        auto prior=completion_authority_counter_.load(std::memory_order_acquire);
        for(;;) {
            if(prior==std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("DFlash completion authority generation exhausted");
            if(completion_authority_counter_.compare_exchange_weak(prior,prior+1,
                std::memory_order_acq_rel,std::memory_order_acquire))return prior+1;
        }
    }
    struct ProposalScope {
        bool& active;
        explicit ProposalScope(bool& value):active(value) {
            if(active)throw std::logic_error("draft proposal already active");
            active=true;
        }
        ~ProposalScope(){active=false;}
        ProposalScope(const ProposalScope&)=delete;
        ProposalScope& operator=(const ProposalScope&)=delete;
    };
public:
    using Coordinator = Exl3VeriCacheServingCoordinator;
    using GreedyPacketBatchFinish=std::function<Exl3GreedyPacket(
        Exl3TextContext&,Exl3PendingGreedyPacket&&,std::uint64_t,std::uint64_t,
        cudaStream_t)>;
    struct Pending {
        std::uint64_t epoch = 0, round = 0;
        Coordinator::Lease parent;
        std::shared_ptr<const Exl3VeriCacheRequest> root;
        Exl3OuterReferenceResult verification;
        unsigned proposed_rows=0; // zero for control publication
        std::uint64_t completion_generation=0;
        Exl3VerifierHorizonPolicy::Costs costs;
        std::uint64_t policy_revision=0;
        std::optional<std::array<Exl3VerifierHorizonPolicy::Observation,2>> conditional_observations;
        std::optional<std::uint16_t> repaired_draft_mask;
        std::uint16_t accepted_draft_mask() const noexcept {
            if(repaired_draft_mask)return *repaired_draft_mask;
            std::uint16_t mask=0;
            const auto mark=[&](std::size_t accepted,unsigned offset) {
                for(unsigned row=1;row<8 && row<accepted;++row)
                    mask|=static_cast<std::uint16_t>(1u<<(offset+row));
            };
            if(conditional_observations) {
                mark((*conditional_observations)[0].accepted,0);
                mark((*conditional_observations)[1].accepted,8);
            } else mark(verification.accepted,0);
            return mask;
        }
        std::uint64_t accepted_draft_rows() const noexcept {
            auto mask=accepted_draft_mask();std::uint64_t count=0;
            while(mask){count+=mask&1u;mask>>=1;}return count;
        }
        std::optional<std::size_t> payload_extent(std::size_t wrapper=0) const noexcept {
            std::array<std::size_t,5> capacities{};
            for(std::size_t i=0;i<5;++i)capacities[i]=verification.committed_taps[i].capacity();
            const auto segments=verification.committed_tap_segments.capacity();
            if(segments>(std::numeric_limits<std::size_t>::max()-wrapper)/
                    sizeof(Exl3CommittedTapSegmentMetadata))return std::nullopt;
            wrapper+=segments*sizeof(Exl3CommittedTapSegmentMetadata);
            return pending_payload_extent(verification.committed_tokens.capacity(),capacities,wrapper);
        }
    };
    static bool attach_conditional_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner)return false;
        const auto* pending=static_cast<const Pending*>(owner.get());
        const auto bytes=pending->payload_extent();
        if(!bytes)return false;
        return attach_bounded_split_retirement_credit<Pending>(owner,std::move(credit),*bytes);
    }
    static std::shared_ptr<Pending> copy_pending_reserved(const Pending& source,
        const Exl3TextContext::SnapshotMetadataReservation& reserve) {
        if(!reserve)throw std::invalid_argument("Pending copy requires metadata reservation");
        const auto ceiling=source.payload_extent(bounded_shared_allocation_bytes<Pending>());
        if(!ceiling)throw std::overflow_error("Pending copy metadata ceiling");
        auto credit=reserve(*ceiling); // Before control block or vector copy.
        if(credit.bytes()!=*ceiling)throw std::invalid_argument("Pending copy reservation extent");
        if(source.payload_extent(bounded_shared_allocation_bytes<Pending>())!=ceiling)
            throw std::invalid_argument("Pending copy capacity changed during reservation");
        auto result=make_bounded_shared<Pending>(source);
        const auto actual=result->payload_extent(bounded_shared_allocation_bytes<Pending>());
        if(!actual || *actual>*ceiling)throw std::overflow_error("Pending copy exceeded reserved capacity");
        if(*actual<*ceiling) {
            auto excess=credit.split(*ceiling-*actual);
            if(!excess)throw std::logic_error("Pending copy excess credit split");
        }
        if(!attach_conditional_retirement_credit(result,std::move(credit)))
            throw std::logic_error("Pending copy credit attachment");
        return result;
    }
    struct ConditionalBlock;
    template<class Factory> std::shared_ptr<Pending> make_pending_owned(
        std::size_t rows,std::size_t peak_rows,Factory&& factory,bool capture_taps=true) {
        if(!rows || rows>(capture_taps?16u:128u) || peak_rows<rows || peak_rows>(capture_taps?32u:128u))
            throw std::invalid_argument("owned Pending row/peak extent");
        const auto& reserve=context_->request_metadata_reservation();
        if(!reserve)throw std::logic_error("owned Pending requires request metadata authority");
        std::array<std::size_t,5> tap_capacity{};tap_capacity.fill(capture_taps?peak_rows*5120:0);
        const auto segment_bytes=capture_taps?peak_rows*sizeof(Exl3CommittedTapSegmentMetadata):0;
        const auto peak=pending_payload_extent(peak_rows,tap_capacity,
            bounded_shared_allocation_bytes<Pending>()+segment_bytes);
        auto credit=reserve(*peak); // Existing old Pending remains charged separately.
        if(credit.bytes()!=*peak)throw std::invalid_argument("owned Pending peak credit extent");
        bool entered=false;
        try {
            entered=true;
            auto value=std::forward<Factory>(factory)();
            auto result=make_bounded_shared<Pending>(std::move(value));
            const auto actual=result->payload_extent(bounded_shared_allocation_bytes<Pending>());
            if(!actual || *actual>*peak)throw std::overflow_error(
                "owned Pending retained capacity exceeded rows="+std::to_string(rows)+
                " peak_rows="+std::to_string(peak_rows)+
                " peak_bytes="+std::to_string(*peak)+
                " actual_bytes="+(actual?std::to_string(*actual):std::string("overflow"))+
                " token_capacity="+std::to_string(result->verification.committed_tokens.capacity())+
                " tap0_capacity="+std::to_string(result->verification.committed_taps[0].capacity())+
                " segment_capacity="+std::to_string(result->verification.committed_tap_segments.capacity()));
            if(*actual<*peak) {
                auto temporary=credit.split(*peak-*actual);
                if(!temporary)throw std::logic_error("owned Pending peak credit split");
            }
            if(!attach_conditional_retirement_credit(result,std::move(credit)))
                throw std::logic_error("owned Pending credit attachment");
            return result;
        }catch(...){if(entered)cleanup_after_failure(true);throw;}
    }
    std::shared_ptr<Pending> verify_owned(std::span<const std::int64_t> tokens) {
        require_ready();
        if(tokens.empty() || tokens.size()>8)throw std::invalid_argument("owned verification extent");
        if(staged_b8_verifier_enabled_ && tokens.size()==8)
            return make_pending_owned(8,8,[&]{return verify_staged_b8(tokens);});
        return make_pending_owned(tokens.size(),tokens.size(),[&]{return verify(tokens);});
    }
    std::shared_ptr<Pending> verify_conditional_second_owned(const Pending& first,const ConditionalBlock& second) {
        require_current_pending(first);
        return make_pending_owned(16,24,[&]{return verify_conditional_second(first,second);});
    }
    std::shared_ptr<Pending> repair_verified_prefix_owned(const Pending& source,std::size_t rows) {
        require_current_pending(source);
        if(!rows || rows>16 || rows>source.verification.committed_tokens.size())
            throw std::invalid_argument("owned prefix repair extent");
        return make_pending_owned(rows,rows>8?2*rows:rows,[&]{return repair_verified_prefix(source,rows);});
    }
    std::shared_ptr<Pending> apply_control_owned(std::span<const std::int64_t> tokens) {
        require_ready();
        if(tokens.empty() || tokens.size()>128)throw std::invalid_argument("owned control extent");
        return make_pending_owned(tokens.size(),tokens.size(),[&]{return apply_control(tokens);},false);
    }
    struct Stats {
        std::uint64_t acquisitions=0, proposal_calls=0, proposed_rows=0,draft_local_topk_calls=0;
        std::uint64_t draft_dense_kmajor_launches=0;
        // Completed scoped reset operations, even if later draft restore fails.
        std::uint64_t acquired_payload_preservations=0,acquired_full_resets=0;
        // Cancellation observed only at synchronized acquisition boundaries;
        // no partially restored target/draft state becomes active.
        std::uint64_t acquired_attachment_cancellations=0;
        // Exact immutable draft-ring witness transfers versus required physical
        // restores/reconstructions at acquisition.
        std::uint64_t acquired_draft_ring_preservations=0,acquired_draft_ring_restores=0;
        // Successful neural B8 attempts, including attempts later aborted.
        // Physical rows include the seed input; returned/discarded counts refer
        // only to the seven predicted suffix rows, never accepted output.
        std::uint64_t neural_input_rows=0,neural_returned_suffix_rows=0,neural_discarded_suffix_rows=0;
        std::uint64_t suffix_calls=0,suffix_rows=0,suffix_misses=0;
        std::uint64_t suffix_published_rounds=0,suffix_accepted_rows=0,suffix_repair_rows=0;
        std::uint64_t suffix_published_skipped_neural_blocks=0;
        std::uint64_t suffix_policy_neural_fallbacks=0;
        std::uint64_t suffix_limit_installations=0;
        std::uint64_t suffix_metadata_bytes=0;
        double reset_ms=0, restore_ms=0, proposal_ms=0, verification_ms=0;
        std::uint64_t completed_native_verifier_calls=0;
        std::uint64_t fast_mia_parity_w1_settlements=0;
        std::uint64_t repair_checkpoint_captured_bytes=0;
        std::uint64_t repair_checkpoint_restores=0;
        std::uint64_t repair_checkpoint_reconstructed_rows=0;
        std::uint64_t repair_checkpoint_fallback_rows=0;
        std::uint64_t device_seed_handoffs=0,device_seed_host_fallbacks=0;
        std::uint64_t committed_tap_device_bytes=0;
        std::uint64_t committed_tap_host_export_rows_avoided=0;
        std::uint64_t cost_menu_updates=0;
        std::uint64_t horizon_decisions=0,full_budget_horizon_switches=0;
        std::array<std::uint64_t,9> supplied_horizon_decisions{};
        std::uint64_t conditional_second_calls=0;
        std::uint64_t conditional_retention_factories=0;
        // Default-off B8 target staging. The pinned draft still executes one
        // physical B8; target rows 5..8 run only after rows 1..4 all match.
        std::uint64_t staged_b8_verifier_calls=0;
        std::uint64_t staged_b8_first_half_exits=0;
        std::uint64_t staged_b8_second_half_calls=0;
        std::uint64_t staged_b8_skipped_verification_rows=0;
        // Selected nondefault-stream draft->upload->attention->export chains.
        std::uint64_t dependency_graph_begins=0;
        std::uint64_t dependency_graph_submissions=0;
        std::uint64_t dependency_graph_completions=0;
        std::uint64_t dependency_graph_cancellations=0;
        std::uint64_t dependency_graph_failures=0;
    };
    // Explicit experimental dependency result. The first target verification
    // has completed but neither block is published by this operation.
    struct ConditionalBlock {
        std::uint64_t epoch=0,predecessor_round=0,acquisition=0;
        std::shared_ptr<const Exl3VeriCacheRequest> conditioning_root;
        int position=0;
        std::vector<std::int64_t> tokens;
        Exl3VerifierHorizonPolicy::Costs costs,predecessor_costs;
        std::optional<double> maximum_publication_ms,estimated_complete_ms;
    };

    // Context and continuation have independent credits. This transaction owns
    // only the lane's staging and represented retirement/suffix metadata.
    static std::shared_ptr<Exl3Dflash2Execution> create_reserved(
        Coordinator& authority,std::shared_ptr<Exl3TextContext> context,
        std::shared_ptr<Exl3Dflash2DraftModel> draft,std::string compatibility,
        cudaStream_t stream,std::shared_ptr<const void> stream_owner,
        unsigned startup_fault_for_test=0) {
        if(startup_fault_for_test>5)throw std::invalid_argument("lane startup fault stage1..5");
        (void)preserve_acquired_root_option(); // Reject before lane reservation/allocation.
        if(!context || context->continuation_bytes_required(8)!=0)
            throw std::invalid_argument("reserved lane requires prepared continuation");
        using Inventory=Exl3ResourceInventory;
        using Domain=Inventory::Domain;
        const auto* suffix=std::getenv("NINFER_EXL3_SUFFIX_PROPOSALS");
        if(suffix && std::string_view(suffix)!="0" && std::string_view(suffix)!="1")
            throw std::invalid_argument("suffix proposals must be0 or1");
        const auto* fast_mia_parity_w1=std::getenv("NINFER_EXL3_FAST_MIA_PARITY_W1");
        if(fast_mia_parity_w1 && std::string_view(fast_mia_parity_w1)!="0" &&
           std::string_view(fast_mia_parity_w1)!="1")
            throw std::invalid_argument("FAST_MIA_PARITY W+1 must be 0 or 1");
        const bool suffix_enabled=suffix && std::string_view(suffix)=="1";
        (void)suffix_key_tokens_option(suffix_enabled);
        Inventory::Requirement requirement;
        requirement.configuration=0x4C414E4553544152;
        requirement.add(Domain::device,5,16ULL*5120*2);
        requirement.add(Domain::event_count,1,
            Exl3RequestCompletionEventPool::capacity);
        requirement.add(Domain::host_metadata,1,sizeof(FailedRetirement));
        requirement.add(Domain::host_metadata,1,sizeof(Exl3Dflash2Execution));
        if(suffix_enabled)
            requirement.add(Domain::host_metadata,1,sizeof(Exl3SuffixProposer));
        std::shared_ptr<Exl3Dflash2Execution> prepared;
        authority.allocate_startup_resources(requirement,[&](auto configuration) {
            if(configuration!=requirement.configuration)
                throw std::logic_error("lane startup reservation identity");
            if(startup_fault_for_test==1)throw std::runtime_error("injected lane startup preconstruction failure");
            prepared=std::make_shared<Exl3Dflash2Execution>(
                context,draft,compatibility,stream,stream_owner,true);
            if(startup_fault_for_test==5)prepared->fail_destructor_drain_for_test();
            if(startup_fault_for_test==2)throw std::runtime_error("injected lane startup precommit failure");
            Inventory actual;
            actual.add({prepared,0,Domain::device,
                prepared->construction_added_device_bytes()-
                    (startup_fault_for_test==3 || startup_fault_for_test==5?1:0)});
            actual.add({prepared,1,Domain::host_metadata,
                sizeof(Exl3Dflash2Execution)+sizeof(FailedRetirement)+prepared->stats_.suffix_metadata_bytes-(startup_fault_for_test==4?1:0)});
            actual.add({prepared,2,Domain::event_count,
                Exl3RequestCompletionEventPool::capacity});
            return actual;
        },[&]() noexcept {
            const auto lanes_before=quarantined_execution_count();
            const auto contexts_before=Exl3TextContext::retirement_quarantine_witness();
            prepared.reset();
            if(quarantined_execution_count()!=lanes_before ||
               Exl3TextContext::retirement_quarantine_witness()!=contexts_before)
                authority.seal_failed_startup_retirement();
        });
        return prepared;
    }

    // Retained Engine form. A retained lane also retains its mutable draft
    // resource and stream backing; shared weights alone did not provide this.
    Exl3Dflash2Execution(std::unique_ptr<Exl3TextContext> context,
        std::shared_ptr<Exl3Dflash2DraftModel> draft,std::string compatibility,
        cudaStream_t stream,std::shared_ptr<const void> stream_owner)
        :Exl3Dflash2Execution(std::shared_ptr<Exl3TextContext>(std::move(context)),
            std::move(draft),std::move(compatibility),stream,std::move(stream_owner)) {}

    // A coordinator may retain the same context solely for allocation lifetime.
    // Execution remains serialized by this lane's existing lease/epoch protocol.
    Exl3Dflash2Execution(std::shared_ptr<Exl3TextContext> context,
        std::shared_ptr<Exl3Dflash2DraftModel> draft,std::string compatibility,
        cudaStream_t stream,std::shared_ptr<const void> stream_owner,
        bool reserved_startup=false)
        :Exl3Dflash2Execution(std::move(context),require_retained_draft(draft,stream,stream_owner),
            std::move(compatibility),stream,0,reserved_startup) {
        if(stream)context_->retain_execution_stream_owner(stream,stream_owner);
        retained_draft_=std::move(draft);retained_stream_=std::move(stream_owner);
    }

    Exl3Dflash2Execution(std::unique_ptr<Exl3TextContext> context,
        Exl3Dflash2DraftModel& draft, std::string compatibility,cudaStream_t stream=nullptr)
        :Exl3Dflash2Execution(std::shared_ptr<Exl3TextContext>(std::move(context)),draft,
            std::move(compatibility),stream) {}

    Exl3Dflash2Execution(std::shared_ptr<Exl3TextContext> context,
        Exl3Dflash2DraftModel& draft,std::string compatibility,cudaStream_t stream=nullptr,
        unsigned allocation_fault_for_test=0,bool reserved_startup=false)
        : context_(std::move(context)), draft_(draft), compatibility_(std::move(compatibility)),stream_(stream) {
        require_retirement_admission();
        if(allocation_fault_for_test>6)throw std::invalid_argument("lane allocation fault stage1..6");
        if (!context_ || context_->oscar_enabled() || compatibility_.empty() ||
            draft_.draft_layers()!=5 || draft_.block_capacity()!=8)
            throw std::invalid_argument("DFlash execution requires exact context and pinned five-layer B8 draft");
        const auto* suffix=std::getenv("NINFER_EXL3_SUFFIX_PROPOSALS");
        if(suffix && std::string_view(suffix)!="0" && std::string_view(suffix)!="1")
            throw std::invalid_argument("suffix proposals must be0 or1");
        const bool suffix_enabled=suffix && std::string_view(suffix)=="1";
        const unsigned suffix_key_tokens=suffix_key_tokens_option(suffix_enabled);
        const auto* horizon=std::getenv("NINFER_EXL3_BOUNDED_VERIFIER_HORIZON");
        if(horizon && std::string_view(horizon)!="0" && std::string_view(horizon)!="1")
            throw std::invalid_argument("bounded verifier horizon must be0 or1");
        horizon_enabled_=horizon && std::string_view(horizon)=="1";
        const auto* native_horizon=std::getenv("NINFER_EXL3_NATIVE_VERIFIER_HORIZON");
        if(native_horizon && std::string_view(native_horizon)!="0" &&
            std::string_view(native_horizon)!="1")
            throw std::invalid_argument("native verifier horizon must be0 or1");
        native_horizon_enabled_=native_horizon && std::string_view(native_horizon)=="1";
        if(native_horizon_enabled_ && !horizon_enabled_)
            throw std::invalid_argument("native verifier horizon requires bounded verifier horizon");
        const auto* staged_b8=std::getenv("NINFER_EXL3_STAGED_B8_VERIFIER");
        if(staged_b8 && std::string_view(staged_b8)!="0" &&
            std::string_view(staged_b8)!="1")
            throw std::invalid_argument("staged B8 verifier must be0 or1");
        staged_b8_verifier_enabled_=staged_b8 && std::string_view(staged_b8)=="1";
        if(staged_b8_verifier_enabled_ && native_horizon_enabled_)
            throw std::invalid_argument("staged B8 verifier is incompatible with native verifier horizon");
        const auto* staged_split=std::getenv("NINFER_EXL3_STAGED_B8_SPLIT_ROWS");
        if(staged_split && !staged_b8_verifier_enabled_)
            throw std::invalid_argument("staged B8 split requires staged verifier");
        if(staged_split) {
            if(std::string_view(staged_split)=="4")staged_b8_split_rows_=4;
            else if(std::string_view(staged_split)=="6")staged_b8_split_rows_=6;
            else throw std::invalid_argument("staged B8 split must be4 or6");
        }
        const auto* repair=std::getenv("NINFER_EXL3_REPAIR_CHECKPOINT");
        if(repair && std::string_view(repair)!="0" && std::string_view(repair)!="1")
            throw std::invalid_argument("repair checkpoint must be0 or1");
        const bool repair_requested=repair && std::string_view(repair)=="1";
        repair_checkpoint_enabled_=repair_requested &&
            (!reserved_startup || context_->transaction_prepared());
        const auto* device_seed=std::getenv("NINFER_EXL3_DEVICE_SEED_HANDOFF");
        if(device_seed && std::string_view(device_seed)!="0" &&
            std::string_view(device_seed)!="1")
            throw std::invalid_argument("device seed handoff must be0 or1");
        device_seed_handoff_enabled_=device_seed && std::string_view(device_seed)=="1";
        if(device_seed_handoff_enabled_ && !exl3_device_greedy_enabled())
            throw std::invalid_argument("device seed handoff requires device greedy");
        preserve_acquired_root_=preserve_acquired_root_option();
        context_entry_device_bytes_=context_->persistent_bytes();
        auto planned_added=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            context_->continuation_bytes_required(8),5ULL*16*5120*2);
        if(repair_checkpoint_enabled_) {
            const auto transaction_required=context_->transaction_bytes_required();
            if(transaction_required)
                planned_added=Exl3LinearWorkspaceRequirements::append_owned_bytes(
                    planned_added,transaction_required);
        }
        if(suffix_enabled) {
            suffix_=std::make_unique<Exl3SuffixProposer>(suffix_key_tokens);
            stats_.suffix_metadata_bytes=sizeof(Exl3SuffixProposer);
        }
        context_->bind_request_compatibility(compatibility_);
        try {
            unsigned allocation_stage=0;
            for(auto& p:staging_) {
                check(cudaMalloc(reinterpret_cast<void**>(&p),16ULL*5120*2));
                if(++allocation_stage==allocation_fault_for_test)throw std::runtime_error("injected lane staging allocation failure");
            }
            for(auto& event:completion_events_)
                check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
            std::array<std::uintptr_t,Exl3RequestCompletionEventPool::capacity> event_ids{};
            for(std::size_t i=0;i<event_ids.size();++i)
                event_ids[i]=reinterpret_cast<std::uintptr_t>(completion_events_[i]);
            completion_pool_.configure(event_ids,completion_authority_generation_);
            if(allocation_fault_for_test==6)throw std::runtime_error("injected lane event allocation failure");
        context_->prepare_continuation(8);
        if(repair_checkpoint_enabled_ && !context_->transaction_prepared())
            context_->prepare_transaction();
        if(context_->persistent_bytes()<context_entry_device_bytes_)
            throw std::logic_error("lane continuation shrank startup context inventory");
        construction_added_device_bytes_=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            context_->persistent_bytes()-context_entry_device_bytes_,5ULL*16*5120*2);
        if(construction_added_device_bytes_!=planned_added)
            throw std::logic_error("lane construction allocation requirement mismatch");
        } catch(...) {
            for(auto& event:completion_events_)if(event) {
                (void)cudaEventDestroy(event);event=nullptr;
            }
            free_staging();throw;
        }
    }
    ~Exl3Dflash2Execution() {
        // The object cannot be recycled after any failed fence. Destruction
        // drains work before its address-stable buffers are released. Keep this
        // device-wide fallback: a poisoned lane may no longer prove a complete
        // stream set. Ordinary Engine completion uses release() and its exact
        // event/retirement certificate instead.
        const auto drain_error=retired_after_drain_?cudaSuccess:
            destructor_drain_failure_for_test_?cudaErrorUnknown:cudaDeviceSynchronize();
        cudaError_t cleanup_error=cudaSuccess;
        if(drain_error==cudaSuccess) {
            completion_pool_.retire_all_after_device_drain();
            for(auto& event:completion_events_)if(event) {
                const auto error=cudaEventDestroy(event);
                if(error!=cudaSuccess){cleanup_error=error;break;}
                event=nullptr;
            }
        }
        if(drain_error!=cudaSuccess || cleanup_error!=cudaSuccess) {
            // Preallocated node: no allocation or fallible cleanup after an
            // uncertain drain. Preserve every owner this lane actually holds.
            auto* retained=failed_retirement_.release();
            retained->context_device_bytes=context_->persistent_bytes();
            retained->staging_device_bytes=5ULL*16*5120*2;
            retained->stream=std::move(retained_stream_);retained->draft=std::move(retained_draft_);
            retained->context=std::move(context_);retained->staging=staging_;
            retained->completion_events=completion_events_;
            retained->completion_pool=std::move(completion_pool_);
            retained->dependency_graph=std::move(dependency_graph_);
            retained->retirement.phase=Exl3RetirementPhase::quarantined;
            retained->retirement.prior_execution_failure=poisoned_;
            if(drain_error!=cudaSuccess)
                retained->retirement.first_drain_error=static_cast<int>(drain_error);
            else
                retained->retirement.first_cleanup_error=static_cast<int>(cleanup_error);
            retained->acquisition=lease_.acquisition;retained->execution=epoch_;retained->round=round_;
            retained->draft_lock=std::move(draft_owner_);
            retained->request_owners={lease_.root,
                conditional_witness_?conditional_witness_->conditioning_root:nullptr,
                conditional_predecessor_,conditional_retention_charge_.owner};
            auto* head=failed_retirements_.load();
            do{retained->next=head;}while(!failed_retirements_.compare_exchange_weak(head,retained));
            ++failed_retirement_count_;return;
        }
        free_staging();
    }
    Exl3Dflash2Execution(const Exl3Dflash2Execution&)=delete;
    Exl3Dflash2Execution& operator=(const Exl3Dflash2Execution&)=delete;
    static std::uint64_t quarantined_execution_count() noexcept {return failed_retirement_count_.load();}
    struct QuarantineSnapshot {
        Exl3RetirementResult retirement;
        Exl3FinalUseWitness::Snapshot final_use;
        std::uint64_t acquisition=0,execution=0,round=0;
        std::uint64_t completion_authority_generation=0;
        std::uint64_t completion_retirement_generation=0;
        std::uint64_t context_device_bytes=0,staging_device_bytes=0;
        std::uintptr_t proposal_host_address=0;
        std::size_t proposal_host_bytes=0;
        std::weak_ptr<const void> request_owner;
        bool draft_lock_retained=false;
    };
    static std::optional<QuarantineSnapshot> latest_quarantine_for_test() noexcept {
        const auto* retained=failed_retirements_.load();
        if(!retained)return std::nullopt;
        const auto completion=retained->completion_pool.latest_snapshot();
        return QuarantineSnapshot{retained->retirement,
            {completion.acquisition,completion.execution,
                completion.final_use_generation,completion.event,
                completion.phase==Exl3RequestCompletionEventPool::Phase::ready,
                completion.first_error},
            retained->acquisition,retained->execution,retained->round,
            completion.authority_generation,completion.retirement_generation,
            retained->context_device_bytes,retained->staging_device_bytes,
            reinterpret_cast<std::uintptr_t>(retained->proposal_block.data()),sizeof(retained->proposal_block),
            retained->request_owners[0],retained->draft_lock.owns_lock()};
    }
    std::uintptr_t proposal_host_address_for_test() const noexcept {
        return reinterpret_cast<std::uintptr_t>(failed_retirement_->proposal_block.data());
    }
    static constexpr std::size_t retirement_metadata_bytes() noexcept {return sizeof(FailedRetirement);}
    void fail_destructor_drain_for_test() {
        if(active_ || retired_after_drain_)throw std::logic_error("destructor fault requires idle unretired lane");
        destructor_drain_failure_for_test_=true;
    }

    using AttachmentCancellation=std::function<bool(unsigned)>;
    using AttachmentRestoreProgress=std::function<void(int)>;
    bool acquire(const Coordinator::Lease& lease,const Coordinator* authority=nullptr,
        const AttachmentCancellation& cancelled={},
        const AttachmentRestoreProgress& restore_progress={}) {
        require_retirement_admission();
        if(epoch_==UINT64_MAX)throw std::overflow_error("DFlash execution epoch exhausted");
        if(active_ || poisoned_ || retired_after_drain_ || !lease.root || !lease.root->compact_draft() ||
            lease.root->state()->model_identity()!=context_->model_identity() ||
            lease.root->state()->position()>=context_->max_context())
            throw std::invalid_argument("DFlash execution acquire ownership/identity/capacity");
        // The authority is borrowed only for this call, never retained by lane
        // teardown. Callers without coordinator proof keep the full reset path.
        if(authority && !authority->compute_leases_current(std::span<const Coordinator::Lease>(&lease,1)))
            throw std::invalid_argument("DFlash execution acquire stale coordinator lease");
        std::unique_lock<std::mutex> draft_owner(draft_.execution_mutex_,std::try_to_lock);
        if(!draft_owner.owns_lock()) throw std::logic_error("DFlash physical draft resource occupied");
        // Phase0 owns no lane state. Later phases are synchronized points after
        // reset, target restore, and complete target+draft attachment.
        if(cancelled && cancelled(0)) {
            ++stats_.acquired_attachment_cancellations;
            return false;
        }
        const auto start=Clock::now();
        // Both target and draft restore can enqueue before acquisition commits.
        // Retain their request source and attempted epoch even when restore fails.
        lease_=lease;
        ++epoch_;
        draft_owner_=std::move(draft_owner);
        const auto abandon_cancelled_attachment=[&] {
            retire_active_completion();
            lease_={};
            if(draft_owner_.owns_lock())draft_owner_.unlock();
            ++stats_.acquired_attachment_cancellations;
            return false;
        };
        try {
            const bool preservation_candidate=preserve_acquired_root_ && authority &&
                !lease.root->prepared_identity();
            const auto reset=preservation_candidate?
                context_->reset_for_request_preserving(compatibility_,*lease.root->state(),context_->request_generation()):
                context_->reset_for_request(compatibility_);
            if(reset.exact_payload_preserved)++stats_.acquired_payload_preservations;
            else ++stats_.acquired_full_resets;
            stats_.reset_ms+=elapsed(start);
            if(cancelled && cancelled(1))return abandon_cancelled_attachment();
            const auto restore=Clock::now();
            if(!reset.exact_payload_preserved) {
                if(restore_progress)
                    context_->observe_next_exact_restore_layer_for_test(restore_progress);
                context_->restore_exact_host_state(*lease.root->state(),stream_);
            }
            if(cancelled && cancelled(2))return abandon_cancelled_attachment();
            if(!preservation_candidate ||
               !lease.root->rebind_draft_scope_if_resident(draft_,lease.acquisition,epoch_))
                draft_.bind_ring_scope(lease.acquisition,epoch_);
            if(lease.root->restore_draft_if_needed(draft_,staging_,stream_))
                ++stats_.acquired_draft_ring_restores;
            else ++stats_.acquired_draft_ring_preservations;
            fence(lease.acquisition,epoch_);
            stats_.restore_ms+=elapsed(restore);
            if(cancelled && cancelled(3))return abandon_cancelled_attachment();
            if(suffix_) {
                const auto position=lease.root->state()->position();
                const auto history=lease.root->token_suffix(std::max(0,position-static_cast<int>(Exl3SuffixProposer::capacity)));
                suffix_->reset();suffix_->append(history);
            }
        } catch(...) {poisoned_=true;throw;}
        horizon_policy_.reset(); // never transfer learned request history to a new lease
        cost_menu_update_round_=0;cost_menu_hold_rounds_=1;
        last_full_budget_horizon_=0;
        conditional_second_round_=0;
        conditional_publication_constraint_={};
        conditional_retention_failure_for_test_=false;
        conditional_verified_failure_for_test_=false;
        tap_copy_failure_for_test_=0;
        proposal_rows_=0;
        fast_mia_parity_pending_.reset();
        active_=true;pending_=false;++stats_.acquisitions;
        return true;
    }
    std::vector<std::int64_t> propose(int maximum_rows) {
        if(failed_retirement_ && failed_retirement_->proposal_active)
            throw std::logic_error("draft proposal already active");
        require_ready();
        proposal_rows_=0;
        int rows=std::min({maximum_rows,8,context_->max_context()-context_->position()});
        if(rows<1) throw std::invalid_argument("DFlash proposal capacity");
        if(horizon_enabled_) {
            const bool full_budget=rows==8;
            rows=static_cast<int>(horizon_policy_.horizon(rows));
            if(horizon_policy_.has_cost_menu())++stats_.supplied_horizon_decisions[rows];
            ++stats_.horizon_decisions;
            if(full_budget) {
                if(last_full_budget_horizon_ && last_full_budget_horizon_!=static_cast<unsigned>(rows))
                    ++stats_.full_budget_horizon_switches;
                last_full_budget_horizon_=static_cast<unsigned>(rows);
            }
        }
        // Ring contains exactly [0, position). The seed is L2's next-token
        // argmax; draft positions start at position and include that seed.
        // Five FP16 post-layer taps are from L2; neither OSCAR nor tentative
        // draft state supplies the conditioning. Target embedding is BF16,
        // with the pinned target H6 head used by the existing draft adapter.
        const bool suffix_eligible=suffix_ && suffix_->eligible();
        const bool device_seed_route=
            device_seed_handoff_enabled_ && exl3_device_greedy_enabled();
        const auto deferred_seed=fast_mia_parity_pending_;
        const bool have_deferred_seed=deferred_seed.has_value();
        std::int64_t seed=-1;
        if(!have_deferred_seed && (!device_seed_route || rows==1 || suffix_eligible)) {
            if(greedy_packet_batch_finish_) {
                auto pending=context_->submit_greedy_packet(
                    false,lease_.acquisition,epoch_,stream_,true);
                const auto packet=greedy_packet_batch_finish_(*context_,
                    std::move(pending),lease_.acquisition,epoch_,stream_);
                seed=packet.decisions[0].token;
            } else seed=exl3_branch_greedy(*context_,stream_);
            if(device_seed_handoff_enabled_)++stats_.device_seed_host_fallbacks;
        }
        if(have_deferred_seed) seed=*deferred_seed;
        if(rows==1) {
            std::vector<std::int64_t> result{seed};
            remember_proposal(result,0,0);
            if(have_deferred_seed) fast_mia_parity_pending_.reset();
            return result;
        }
        if(suffix_ && !suffix_eligible)++stats_.suffix_policy_neural_fallbacks;
        if(suffix_eligible) {
            const auto proposal=suffix_->propose(seed,rows);
            if(proposal.rows>1) {
                ++stats_.suffix_calls;stats_.suffix_rows+=proposal.rows-1;
                std::vector<std::int64_t> result{proposal.tokens.begin(),proposal.tokens.begin()+proposal.rows};
                remember_proposal(result,0,0);
                if(have_deferred_seed) fast_mia_parity_pending_.reset();
                return result;
            }
            ++stats_.suffix_misses;
        }
        const auto start=Clock::now();
        try {
            // Fixed private proposal input; returned proposals retain separate
            // owning storage and never borrow this next-call scratch.
            if(!lease_.root->draft_lineage_ready(draft_,lease_.acquisition,epoch_,context_->position()))
                throw std::invalid_argument("DFlash proposal lacks completed root-bound ring conditioning");
            auto& proposal_block_=failed_retirement_->proposal_block;
            // The inherited policy used a logical short horizon but still ran
            // the pinned draft at physical B8 and discarded its suffix.  The
            // draft implementation already accepts every exact block extent
            // in [2,8].  Keep B8 as the default and let this research option
            // remove the unused neural rows selected by the same policy.
            const int neural_rows=native_horizon_enabled_?rows:8;
            const auto neural_block=std::span<const std::int64_t>(
                proposal_block_.data(),static_cast<std::size_t>(neural_rows));
            const auto local_topk_before=draft_.local_topk_calls();
            const auto dense_kmajor_before=draft_.dense_kmajor_launches();
            auto proposals=[&] {
                ProposalScope proposal_scope(failed_retirement_->proposal_active);
                failed_retirement_->proposal_root=lease_.root;
                std::fill(proposal_block_.begin(),proposal_block_.end(),248070);
                if(device_seed_route && !have_deferred_seed) {
                    auto pending_seed=context_->submit_greedy_packet(
                        false,lease_.acquisition,epoch_,stream_);
                    const auto device_seed=context_->device_greedy_seed(
                        pending_seed,lease_.acquisition,epoch_);
                    auto value=draft_.propose_cached_device_seed(
                        neural_block,device_seed,context_->position(),
                        context_->target_embedding(),context_->target_lm_head_weights(),
                        context_->target_lm_head_metadata(),248070,stream_);
                    const auto packet=context_->finish_greedy_packet(
                        std::move(pending_seed),lease_.acquisition,epoch_);
                    seed=packet.decisions[0].token;
                    proposal_block_[0]=seed;
                    ++stats_.device_seed_handoffs;
                    return value;
                }
                proposal_block_[0]=seed;
                return draft_.propose_cached_view(
                    neural_block,context_->position(),context_->target_embedding(),
                    context_->target_lm_head_weights(),context_->target_lm_head_metadata(),
                    248070,stream_);
            }();
            ++stats_.proposal_calls;stats_.proposed_rows+=proposals.size();
            stats_.draft_local_topk_calls+=draft_.local_topk_calls()-local_topk_before;
            stats_.draft_dense_kmajor_launches+=
                draft_.dense_kmajor_launches()-dense_kmajor_before;
            const auto proposal_ms=elapsed(start);stats_.proposal_ms+=proposal_ms;
            if(proposals.size()!=static_cast<std::size_t>(neural_rows-1))
                throw std::runtime_error("DFlash neural proposal extent");
            stats_.neural_input_rows+=static_cast<unsigned>(neural_rows);
            stats_.neural_returned_suffix_rows+=static_cast<unsigned>(rows-1);
            stats_.neural_discarded_suffix_rows+=static_cast<unsigned>(neural_rows-rows);
            proposals.insert(proposals.begin(),seed);proposals.resize(rows);
            remember_proposal(proposals,static_cast<unsigned>(neural_rows),proposal_ms);
            if(have_deferred_seed) fast_mia_parity_pending_.reset();
            return proposals;
        } catch(...) {cleanup_after_failure(false);throw;}
    }
    Pending verify(std::span<const std::int64_t> tokens,
        std::span<const std::int64_t> terminal={}) {
        require_ready();
        if(round_==UINT64_MAX)throw std::overflow_error("DFlash round exhausted");
        if(tokens.empty() || tokens.size()>8 ||
            tokens.size()>static_cast<std::size_t>(context_->max_context()-context_->position()))
            throw std::invalid_argument("DFlash verification extent");
        const int verification_position=context_->position();
        const auto start=Clock::now();
        try {
            const auto tap_binding=lease_.root->committed_tap_binding(
                context_,lease_.acquisition,epoch_);
            auto [root,result]=lease_.root->verify_compact(*context_,draft_,staging_,tokens,terminal,stream_,
                repair_checkpoint_enabled_,tap_binding,
                std::exchange(tap_copy_failure_for_test_,0u));
            bind_target_dependencies(root);
            observe_tap_before_fence();fence();bind_tap_readiness(result);
            const auto verification_ms=elapsed(start);
            stats_.verification_ms+=verification_ms;pending_=true;
            stats_.completed_native_verifier_calls+=result.native_invocations;
            stats_.fast_mia_parity_w1_settlements+=result.fast_mia_parity_w1_bonus_rows;
            stats_.repair_checkpoint_captured_bytes+=result.checkpoint_captured_bytes;
            stats_.repair_checkpoint_restores+=result.checkpoint_restores;
            stats_.repair_checkpoint_reconstructed_rows+=result.checkpoint_reconstructed_rows;
            stats_.repair_checkpoint_fallback_rows+=result.checkpoint_fallback_rows;
            stats_.committed_tap_device_bytes+=result.committed_tap_d2d_bytes;
            stats_.committed_tap_host_export_rows_avoided+=
                result.committed_tap_host_export_rows_avoided;
            Exl3VerifierHorizonPolicy::Costs costs;
            // Host wall interval includes compact verification, any repair and
            // completion fence. Separate repair/complete costs remain unknown.
            costs.verification_ms=verification_ms;
            if(proposal_rows_==tokens.size() && proposal_epoch_==epoch_ &&
                proposal_round_==round_ && proposal_position_==verification_position &&
                proposal_policy_revision_==horizon_policy_.counters().published_rounds &&
                std::equal(tokens.begin(),tokens.end(),proposal_tokens_.begin())) {
                costs.neural_ms=proposal_costs_.neural_ms;
                costs.neural_rows=proposal_costs_.neural_rows;
            }
            proposal_rows_=0;
            return Pending{epoch_,++round_,lease_,std::move(root),std::move(result),static_cast<unsigned>(tokens.size()),active_completion_generation_,costs,
                horizon_policy_.counters().published_rounds};
        } catch(...) {fail_dependency_graph();cleanup_after_failure(false);throw;}
    }
    // Verify an already-produced physical B8 in two authoritative stages. B4
    // remains the historical default; B6 is an explicit experiment which
    // trades fewer skippable rows for materially better target batch geometry.
    // This is not the rejected fixed-horizon route: draft work remains one B8,
    // and a fully accepted first half consumes the existing suffix without a
    // second proposal.  An early mismatch returns the first exact child and
    // removes the unexecuted tail rows; no tentative suffix is published.
    Pending verify_staged_b8(std::span<const std::int64_t> tokens) {
        require_ready();
        if(tokens.size()!=8 || round_>UINT64_MAX-2)
            throw std::invalid_argument("staged B8 verifier extent");
        const int verification_position=context_->position();
        const bool proposal_matches=proposal_rows_==8 && proposal_epoch_==epoch_ &&
            proposal_round_==round_ && proposal_position_==verification_position &&
            proposal_policy_revision_==horizon_policy_.counters().published_rounds &&
            std::equal(tokens.begin(),tokens.end(),proposal_tokens_.begin());
        auto inherited_costs=proposal_matches?proposal_costs_:
            Exl3VerifierHorizonPolicy::Costs{};
        const auto split=static_cast<std::size_t>(staged_b8_split_rows_);
        const auto tail=tokens.size()-split;
        auto first=verify(tokens.first(split));
        ++stats_.staged_b8_verifier_calls;
        inherited_costs.verification_ms=first.costs.verification_ms;
        first.proposed_rows=8;
        first.costs=inherited_costs;
        if(first.verification.rejected || first.verification.stopped ||
           first.verification.accepted!=split ||
           first.verification.committed_tokens.size()!=split) {
            ++stats_.staged_b8_first_half_exits;
            stats_.staged_b8_skipped_verification_rows+=tail;
            return first;
        }

        const auto& a=first.verification;
        std::vector<std::int64_t> combined_tokens;
        combined_tokens.reserve(8);
        combined_tokens.insert(combined_tokens.end(),a.committed_tokens.begin(),
                               a.committed_tokens.end());
        std::array<std::vector<std::uint16_t>,5> combined_taps;
        for(std::size_t plane=0;plane<combined_taps.size();++plane) {
            if(a.committed_taps[plane].size()!=split*5120ULL)
                throw std::runtime_error("staged B8 first tap extent");
            combined_taps[plane].reserve(8ULL*5120);
            combined_taps[plane].insert(combined_taps[plane].end(),
                a.committed_taps[plane].begin(),a.committed_taps[plane].end());
        }
        std::vector<Exl3CommittedTapSegmentMetadata> combined_tap_segments;
        combined_tap_segments.reserve(8);
        combined_tap_segments.insert(combined_tap_segments.end(),
            a.committed_tap_segments.begin(),a.committed_tap_segments.end());

        const auto start=Clock::now();
        try {
            const auto tap_binding=first.root->committed_tap_binding(
                context_,lease_.acquisition,epoch_);
            auto [root,v]=first.root->verify_compact(*context_,draft_,staging_,
                tokens.subspan(split),{},stream_,repair_checkpoint_enabled_,tap_binding,
                std::exchange(tap_copy_failure_for_test_,0u));
            observe_tap_before_fence();fence();bind_tap_readiness(v);
            const auto second_ms=elapsed(start);
            stats_.verification_ms+=second_ms;
            stats_.completed_native_verifier_calls+=v.native_invocations;
            stats_.fast_mia_parity_w1_settlements+=v.fast_mia_parity_w1_bonus_rows;
            stats_.repair_checkpoint_captured_bytes+=v.checkpoint_captured_bytes;
            stats_.repair_checkpoint_restores+=v.checkpoint_restores;
            stats_.repair_checkpoint_reconstructed_rows+=v.checkpoint_reconstructed_rows;
            stats_.repair_checkpoint_fallback_rows+=v.checkpoint_fallback_rows;
            stats_.committed_tap_device_bytes+=v.committed_tap_d2d_bytes;
            ++stats_.staged_b8_second_half_calls;
            if(v.committed_tokens.empty() || v.committed_tokens.size()>tail)
                throw std::runtime_error("staged B8 second committed extent");
            for(const auto& tap:v.committed_taps)
                if(tap.size()!=v.committed_tokens.size()*5120)
                    throw std::runtime_error("staged B8 second tap extent");

            root=root->compose_verified_child(
                *lease_.root,*first.root,context_->request_metadata_reservation());
            combined_tokens.insert(combined_tokens.end(),v.committed_tokens.begin(),
                                   v.committed_tokens.end());
            v.committed_tokens.swap(combined_tokens);
            for(std::size_t plane=0;plane<v.committed_taps.size();++plane) {
                combined_taps[plane].insert(combined_taps[plane].end(),
                    v.committed_taps[plane].begin(),v.committed_taps[plane].end());
                v.committed_taps[plane].swap(combined_taps[plane]);
            }
            for(auto segment:v.committed_tap_segments) {
                segment.destination_first+=static_cast<int>(split);
                combined_tap_segments.push_back(segment);
            }
            v.committed_tap_segments.swap(combined_tap_segments);
            v.accepted+=a.accepted;v.executed_rows+=a.executed_rows;
            v.verification_rows+=a.verification_rows;v.replay_rows+=a.replay_rows;
            v.native_invocations+=a.native_invocations;v.root_restores+=a.root_restores;
            v.checkpoint_captured_bytes+=a.checkpoint_captured_bytes;
            v.checkpoint_restores+=a.checkpoint_restores;
            v.checkpoint_reconstructed_rows+=a.checkpoint_reconstructed_rows;
            v.checkpoint_fallback_rows+=a.checkpoint_fallback_rows;
            v.committed_tap_d2d_bytes+=a.committed_tap_d2d_bytes;
            v.discarded_tentative_tap_segments+=a.discarded_tentative_tap_segments;
            for(std::size_t i=0;i<v.native_batch_hist.size();++i)
                v.native_batch_hist[i]+=a.native_batch_hist[i];
            inherited_costs.verification_ms=
                first.costs.verification_ms.value_or(0.0)+second_ms;
            return Pending{epoch_,++round_,lease_,std::move(root),std::move(v),8,
                active_completion_generation_,inherited_costs,
                horizon_policy_.counters().published_rounds};
        } catch(...) {cleanup_after_failure(true);throw;}
    }
    ConditionalBlock propose_conditional_second(Pending& predecessor,int remaining_output_rows_at_parent,
        Coordinator* retention_authority=nullptr,std::optional<double> estimated_complete_ms=std::nullopt) {
        if(failed_retirement_ && failed_retirement_->proposal_active)
            throw std::logic_error("draft proposal already active");
        require_current_pending(predecessor);
        if(!conditional_publication_constraint_.allows(estimated_complete_ms))
            throw std::invalid_argument("conditional publication estimate absent or exceeds request limit");
        const auto& v=predecessor.verification;
        if(remaining_output_rows_at_parent<16 || epoch_==UINT64_MAX || round_==UINT64_MAX || conditional_second_round_==round_ || predecessor.proposed_rows!=8 || v.accepted!=8 ||
            v.rejected || v.stopped || v.committed_tokens.size()!=8 ||
            !predecessor.root || !predecessor.root->is_child_of(*lease_.root) ||
            v.committed_state!=predecessor.root->state() ||
            predecessor.root->state()->position()!=lease_.root->state()->position()+8 ||
            context_->position()!=predecessor.root->state()->position() ||
            !context_->exact_host_state_resident(*predecessor.root->state()) ||
            context_->max_context()-context_->position()<8 ||
            !predecessor.root->draft_lineage_ready(draft_,lease_.acquisition,epoch_,context_->position()))
            throw std::invalid_argument("conditional B8 requires complete accepted predecessor conditioning and capacity");
        for(const auto& tap:v.committed_taps)if(tap.size()!=8ULL*5120)
            throw std::invalid_argument("conditional predecessor tap extent before retention");
        if(predecessor.root->token_suffix(lease_.root->state()->position())!=v.committed_tokens)
            throw std::invalid_argument("conditional predecessor committed token lineage");
        ConditionalBlock result;
        result.epoch=epoch_;result.predecessor_round=round_;result.acquisition=lease_.acquisition;
        result.conditioning_root=predecessor.root;result.position=context_->position();
        result.predecessor_costs=predecessor.costs; // includes intervening target work, never free
        result.maximum_publication_ms=conditional_publication_constraint_.maximum_ms;
        result.estimated_complete_ms=estimated_complete_ms;
        // Prepare retained tap/token metadata before any dependent numerical
        // submission. Allocation failure leaves the first Pending publishable.
        if(conditional_retention_failure_for_test_) {
            conditional_retention_failure_for_test_=false;
            throw std::bad_alloc();
        }
        std::shared_ptr<Pending> retained_predecessor;
        result.tokens.resize(8);
        ConditionalBlock retained_witness=result;
        if(retention_authority) {
            if(conditional_retention_charge_.units ||
                (conditional_retention_authority_ && conditional_retention_authority_!=retention_authority))
                throw std::logic_error("conditional retention requires collected same-authority storage");
            using Inventory=Exl3ResourceInventory;
            const auto payload_extent=predecessor.payload_extent(
                bounded_shared_allocation_bytes<Pending>());
            if(!payload_extent)throw std::overflow_error("conditional Pending metadata ceiling");
            const std::uint64_t payload=*payload_extent;
            Inventory::Requirement required;required.configuration=0x434F4E444238;
            required.add(Inventory::Domain::host_metadata,1,payload);
            retention_authority->allocate_runtime_resources(lease_,required,[&](std::uint64_t configuration) {
                if(configuration!=required.configuration)throw std::logic_error("conditional retention configuration");
                ++stats_.conditional_retention_factories;
                retained_predecessor=make_bounded_shared<Pending>(predecessor);
                // Vector copy construction may shrink capacity to size. Restore
                // the predecessor capacities so the exact authority reservation
                // includes every retained tap-segment descriptor as well as the
                // token/tap payloads it guards.
                retained_predecessor->verification.committed_tokens.reserve(
                    predecessor.verification.committed_tokens.capacity());
                for(std::size_t i=0;i<retained_predecessor->verification.committed_taps.size();++i)
                    retained_predecessor->verification.committed_taps[i].reserve(
                        predecessor.verification.committed_taps[i].capacity());
                retained_predecessor->verification.committed_tap_segments.reserve(
                    predecessor.verification.committed_tap_segments.capacity());
                const auto actual=retained_predecessor->payload_extent(bounded_shared_allocation_bytes<Pending>());
                if(!actual || *actual!=payload)
                    throw std::invalid_argument("conditional Pending metadata extent");
                Inventory inventory;inventory.add({retained_predecessor,0,Inventory::Domain::host_metadata,*actual,{},
                    &attach_conditional_retirement_credit});
                return inventory;
            },[&]() noexcept {retained_predecessor.reset();});
            conditional_retention_authority_=retention_authority;
            conditional_retention_charge_={retained_predecessor,0,Inventory::Domain::host_metadata,payload,{},
                &attach_conditional_retirement_credit};
        } else if(context_->request_metadata_reservation())
            retained_predecessor=copy_pending_reserved(predecessor,context_->request_metadata_reservation());
        else retained_predecessor=std::make_shared<Pending>(predecessor);
        try {
            const bool device_seed_route=
                device_seed_handoff_enabled_ && exl3_device_greedy_enabled();
            std::int64_t seed=-1;
            if(!device_seed_route) {
                if(greedy_packet_batch_finish_) {
                    auto pending=context_->submit_greedy_packet(
                        false,lease_.acquisition,epoch_,stream_,true);
                    const auto packet=greedy_packet_batch_finish_(*context_,
                        std::move(pending),lease_.acquisition,epoch_,stream_);
                    seed=packet.decisions[0].token;
                } else seed=exl3_branch_greedy(*context_,stream_);
                if(device_seed_handoff_enabled_)++stats_.device_seed_host_fallbacks;
            }
            auto& proposal_block_=failed_retirement_->proposal_block;
            const auto start=Clock::now();
            auto predicted=[&] {
                ProposalScope proposal_scope(failed_retirement_->proposal_active);
                failed_retirement_->proposal_root=predecessor.root;
                std::fill(proposal_block_.begin(),proposal_block_.end(),248070);
                if(device_seed_route) {
                    auto pending_seed=context_->submit_greedy_packet(
                        false,lease_.acquisition,epoch_,stream_);
                    const auto device_seed=context_->device_greedy_seed(
                        pending_seed,lease_.acquisition,epoch_);
                    auto value=draft_.propose_cached_device_seed(
                        proposal_block_,device_seed,result.position,
                        context_->target_embedding(),context_->target_lm_head_weights(),
                        context_->target_lm_head_metadata(),248070,stream_);
                    const auto packet=context_->finish_greedy_packet(
                        std::move(pending_seed),lease_.acquisition,epoch_);
                    seed=packet.decisions[0].token;proposal_block_[0]=seed;
                    ++stats_.device_seed_handoffs;
                    return value;
                }
                proposal_block_[0]=seed;
                return draft_.propose_cached_view(
                    proposal_block_,result.position,context_->target_embedding(),
                    context_->target_lm_head_weights(),context_->target_lm_head_metadata(),
                    248070,stream_);
            }();
            const auto neural_ms=elapsed(start);
            ++stats_.proposal_calls;stats_.proposed_rows+=predicted.size();stats_.proposal_ms+=neural_ms;
            if(predicted.size()!=7)throw std::runtime_error("conditional B8 proposal extent");
            stats_.neural_input_rows+=8;stats_.neural_returned_suffix_rows+=7;
            ++stats_.conditional_second_calls;
            result.tokens.front()=seed;
            std::copy(predicted.begin(),predicted.end(),result.tokens.begin()+1);
            // Extend final-use ownership past the dependent draft before the
            // first block can publish/release. Older Pending copies are stale.
            fence();predecessor.completion_generation=active_completion_generation_;
            bind_tap_readiness(predecessor.verification);
            conditional_second_round_=round_;
            result.costs.neural_rows=8;result.costs.neural_ms=neural_ms;
            std::copy(result.tokens.begin(),result.tokens.end(),retained_witness.tokens.begin());
            retained_witness.costs=result.costs;
            retained_predecessor->completion_generation=predecessor.completion_generation;
            retained_predecessor->verification.committed_tap_ready_generation=
                predecessor.verification.committed_tap_ready_generation;
            retained_predecessor->verification.committed_tap_ready_event=
                predecessor.verification.committed_tap_ready_event;
            conditional_witness_=std::move(retained_witness);
            conditional_predecessor_=std::move(retained_predecessor);
            return result;
        } catch(...) {cleanup_after_failure(true);throw;}
    }
    std::optional<ConditionalBlock> try_propose_conditional_second(Pending& predecessor,
        int remaining_output_rows_at_parent,Coordinator& authority) {
        const auto calls=stats_.proposal_calls;
        try {return propose_conditional_second(predecessor,remaining_output_rows_at_parent,&authority);}
        catch(const Exl3ResourceReservationExhausted&) {
            // Only a refused reservation before dependent work is recoverable.
            // A poisoned/stale lane or installed credit must follow retirement.
            require_current_pending(predecessor);
            if(stats_.proposal_calls!=calls || conditional_retention_charge_.units)throw;
            return std::nullopt;
        }
    }
    Pending verify_conditional_second(const Pending& first,const ConditionalBlock& second,
        std::span<const std::int64_t> terminal={}) {
        require_current_pending(first);
        if(!conditional_witness_ || !conditional_predecessor_ || first.root!=conditional_predecessor_->root ||
            round_==UINT64_MAX || epoch_==UINT64_MAX ||
            second.epoch!=epoch_ || second.acquisition!=lease_.acquisition ||
            second.predecessor_round!=round_ || second.conditioning_root!=first.root ||
            second.position!=context_->position() || second.tokens!=conditional_witness_->tokens ||
            second.conditioning_root!=conditional_witness_->conditioning_root ||
            second.position!=conditional_witness_->position ||
            second.maximum_publication_ms!=conditional_witness_->maximum_publication_ms ||
            second.estimated_complete_ms!=conditional_witness_->estimated_complete_ms ||
            !first.root->draft_lineage_ready(draft_,lease_.acquisition,epoch_,context_->position()))
            throw std::invalid_argument("conditional B8 stale or changed dependency");
        // Costs come from the private witness, not the caller-owned descriptor.
        auto costs=conditional_witness_->costs;
        const auto& first_result=conditional_predecessor_->verification;
        // A changed stop policy cannot retrospectively authorize rows beyond
        // an earlier terminal in the retained first block.
        for(auto token:terminal) {
            if(token<0 || token>=248320 ||
                std::find(first_result.committed_tokens.begin(),first_result.committed_tokens.end(),token)!=first_result.committed_tokens.end())
                throw std::invalid_argument("conditional B8 terminal crosses predecessor boundary");
        }
        std::vector<std::int64_t> combined_tokens;
        combined_tokens.reserve(16);
        combined_tokens.insert(combined_tokens.end(),first_result.committed_tokens.begin(),first_result.committed_tokens.end());
        std::array<std::vector<std::uint16_t>,5> combined_taps;
        std::vector<Exl3CommittedTapSegmentMetadata> combined_tap_segments;
        combined_tap_segments.reserve(16);
        combined_tap_segments.insert(combined_tap_segments.end(),
            first_result.committed_tap_segments.begin(),first_result.committed_tap_segments.end());
        for(std::size_t i=0;i<combined_taps.size();++i) {
            if(first_result.committed_taps[i].size()!=8ULL*5120)
                throw std::invalid_argument("conditional first tap extent");
            combined_taps[i].reserve(16ULL*5120);
            combined_taps[i].insert(combined_taps[i].end(),first_result.committed_taps[i].begin(),first_result.committed_taps[i].end());
        }
        const auto start=Clock::now();
        try {
            const auto tap_binding=first.root->committed_tap_binding(
                context_,lease_.acquisition,epoch_);
            auto [root,v]=first.root->verify_compact(*context_,draft_,staging_,second.tokens,terminal,stream_,
                repair_checkpoint_enabled_,tap_binding,
                std::exchange(tap_copy_failure_for_test_,0u));
            observe_tap_before_fence();fence();bind_tap_readiness(v);
            costs.verification_ms=elapsed(start);stats_.verification_ms+=*costs.verification_ms;
            stats_.completed_native_verifier_calls+=v.native_invocations;
            stats_.fast_mia_parity_w1_settlements+=v.fast_mia_parity_w1_bonus_rows;
            stats_.repair_checkpoint_captured_bytes+=v.checkpoint_captured_bytes;
            stats_.repair_checkpoint_restores+=v.checkpoint_restores;
            stats_.repair_checkpoint_reconstructed_rows+=v.checkpoint_reconstructed_rows;
            stats_.repair_checkpoint_fallback_rows+=v.checkpoint_fallback_rows;
            stats_.committed_tap_device_bytes+=v.committed_tap_d2d_bytes;
            if(conditional_verified_failure_for_test_) {
                conditional_verified_failure_for_test_=false;
                throw std::runtime_error("injected conditional failure after verified completion");
            }
            if(v.committed_tokens.empty() || v.committed_tokens.size()>8)
                throw std::runtime_error("conditional second committed extent");
            for(const auto& tap:v.committed_taps)if(tap.size()!=v.committed_tokens.size()*5120)
                throw std::runtime_error("conditional second tap extent");
            const auto observation=[](unsigned rows,const Exl3OuterReferenceResult& value,
                const Exl3VerifierHorizonPolicy::Costs& work) {
                using Source=Exl3VerifierHorizonPolicy::ProposalSource;
                const auto source=!work.neural_rows?Source::unknown:
                    *work.neural_rows?Source::neural_b8:rows==1?Source::seed_only:Source::suffix;
                return Exl3VerifierHorizonPolicy::Observation{rows,static_cast<unsigned>(value.accepted),
                    static_cast<unsigned>(value.replay_rows),static_cast<unsigned>(value.native_invocations),
                    value.rejected,value.stopped,work,source};
            };
            const std::array observations{observation(8,first_result,conditional_predecessor_->costs),observation(8,v,costs)};
            root=root->compose_verified_child(*lease_.root,*first.root,context_->request_metadata_reservation());
            combined_tokens.insert(combined_tokens.end(),v.committed_tokens.begin(),v.committed_tokens.end());
            v.committed_tokens.swap(combined_tokens);
            for(std::size_t i=0;i<v.committed_taps.size();++i) {
                combined_taps[i].insert(combined_taps[i].end(),v.committed_taps[i].begin(),v.committed_taps[i].end());
                v.committed_taps[i].swap(combined_taps[i]);
            }
            for(auto segment:v.committed_tap_segments) {
                segment.destination_first+=static_cast<int>(first_result.committed_tokens.size());
                combined_tap_segments.push_back(segment);
            }
            v.committed_tap_segments.swap(combined_tap_segments);
            v.accepted+=first_result.accepted;v.executed_rows+=first_result.executed_rows;
            v.verification_rows+=first_result.verification_rows;v.replay_rows+=first_result.replay_rows;
            v.native_invocations+=first_result.native_invocations;v.root_restores+=first_result.root_restores;
            v.checkpoint_captured_bytes+=first_result.checkpoint_captured_bytes;
            v.checkpoint_restores+=first_result.checkpoint_restores;
            v.checkpoint_reconstructed_rows+=first_result.checkpoint_reconstructed_rows;
            v.checkpoint_fallback_rows+=first_result.checkpoint_fallback_rows;
            v.committed_tap_d2d_bytes+=first_result.committed_tap_d2d_bytes;
            v.discarded_tentative_tap_segments+=first_result.discarded_tentative_tap_segments;
            for(std::size_t i=0;i<v.native_batch_hist.size();++i)v.native_batch_hist[i]+=first_result.native_batch_hist[i];
            conditional_witness_.reset();
            conditional_predecessor_.reset();
            Pending result{epoch_,++round_,lease_,std::move(root),std::move(v),16,active_completion_generation_,{},
                horizon_policy_.counters().published_rounds};
            result.conditional_observations=observations;
            return result;
        }catch(...){cleanup_after_failure(true);throw;}
    }
    // Repair only a prefix of an already verified pending result. No output is
    // published between the independent verifier calls at the B8 boundary.
    Pending repair_verified_prefix(const Pending& source,std::size_t rows) {
        require_current_pending(source);
        const auto& committed=source.verification.committed_tokens;
        if(!rows || rows>16 || rows>committed.size() || round_>UINT64_MAX-2)
            throw std::invalid_argument("verified prefix repair extent");
        std::array<std::int64_t,16> retained{};
        const auto accepted_mask=static_cast<std::uint16_t>(source.accepted_draft_mask() & ((1u<<rows)-1u));
        std::copy_n(committed.begin(),rows,retained.begin());
        std::vector<std::int64_t> combined_tokens;
        std::array<std::vector<std::uint16_t>,5> combined_taps;
        std::vector<Exl3CommittedTapSegmentMetadata> combined_tap_segments;
        if(rows>8) {
            // Allocate join storage while the original Pending is still valid.
            // Failure here must not invalidate it or submit replay work.
            if(std::exchange(repair_join_failure_for_test_,false))throw std::bad_alloc();
            combined_tokens.reserve(rows);
            for(auto& tap:combined_taps)tap.reserve(rows*5120);
            combined_tap_segments.reserve(rows);
        }
        abort();
        auto first=verify(std::span(retained).first(std::min<std::size_t>(rows,8)));
        // Replaying known committed tokens must reproduce the complete prefix.
        try {
            if(first.verification.committed_tokens.size()!=std::min<std::size_t>(rows,8) ||
               !std::equal(first.verification.committed_tokens.begin(),first.verification.committed_tokens.end(),retained.begin()))
                throw std::runtime_error("verified prefix first replay diverged");
            first.proposed_rows=0; // Repair never trains proposal selection.
            first.repaired_draft_mask=accepted_mask;
            if(rows<=8)return first;
            const auto start=Clock::now();
            const auto tap_binding=first.root->committed_tap_binding(
                context_,lease_.acquisition,epoch_);
            auto [root,v]=first.root->verify_compact(*context_,draft_,staging_,
                std::span(retained).subspan(8,rows-8),{},stream_,repair_checkpoint_enabled_,
                tap_binding,std::exchange(tap_copy_failure_for_test_,0u));
            observe_tap_before_fence();fence();bind_tap_readiness(v);
            stats_.verification_ms+=elapsed(start);
            stats_.completed_native_verifier_calls+=v.native_invocations;
            stats_.fast_mia_parity_w1_settlements+=v.fast_mia_parity_w1_bonus_rows;
            stats_.repair_checkpoint_captured_bytes+=v.checkpoint_captured_bytes;
            stats_.repair_checkpoint_restores+=v.checkpoint_restores;
            stats_.repair_checkpoint_reconstructed_rows+=v.checkpoint_reconstructed_rows;
            stats_.repair_checkpoint_fallback_rows+=v.checkpoint_fallback_rows;
            stats_.committed_tap_device_bytes+=v.committed_tap_d2d_bytes;
            if(v.committed_tokens.size()!=rows-8 ||
               !std::equal(v.committed_tokens.begin(),v.committed_tokens.end(),retained.begin()+8))
                throw std::runtime_error("verified prefix second replay diverged");
            root=root->compose_verified_child(*lease_.root,*first.root,context_->request_metadata_reservation());
            const auto& a=first.verification;
            combined_tokens.insert(combined_tokens.end(),a.committed_tokens.begin(),a.committed_tokens.end());
            combined_tokens.insert(combined_tokens.end(),v.committed_tokens.begin(),v.committed_tokens.end());
            v.committed_tokens.swap(combined_tokens);
            for(std::size_t i=0;i<v.committed_taps.size();++i) {
                if(a.committed_taps[i].size()!=8ULL*5120 || v.committed_taps[i].size()!=(rows-8)*5120)
                    throw std::runtime_error("verified prefix replay tap extent");
                combined_taps[i].insert(combined_taps[i].end(),a.committed_taps[i].begin(),a.committed_taps[i].end());
                combined_taps[i].insert(combined_taps[i].end(),v.committed_taps[i].begin(),v.committed_taps[i].end());
                v.committed_taps[i].swap(combined_taps[i]);
            }
            combined_tap_segments.insert(combined_tap_segments.end(),
                a.committed_tap_segments.begin(),a.committed_tap_segments.end());
            for(auto segment:v.committed_tap_segments) {
                segment.destination_first+=8;
                combined_tap_segments.push_back(segment);
            }
            v.committed_tap_segments.swap(combined_tap_segments);
            v.accepted+=a.accepted;v.executed_rows+=a.executed_rows;
            v.verification_rows+=a.verification_rows;v.replay_rows+=a.replay_rows;
            v.native_invocations+=a.native_invocations;v.root_restores+=a.root_restores;
            v.checkpoint_captured_bytes+=a.checkpoint_captured_bytes;
            v.checkpoint_restores+=a.checkpoint_restores;
            v.checkpoint_reconstructed_rows+=a.checkpoint_reconstructed_rows;
            v.checkpoint_fallback_rows+=a.checkpoint_fallback_rows;
            v.committed_tap_d2d_bytes+=a.committed_tap_d2d_bytes;
            v.discarded_tentative_tap_segments+=a.discarded_tentative_tap_segments;
            for(std::size_t i=0;i<v.native_batch_hist.size();++i)v.native_batch_hist[i]+=a.native_batch_hist[i];
            Pending result{epoch_,++round_,lease_,std::move(root),std::move(v),0,active_completion_generation_};
            result.repaired_draft_mask=accepted_mask;return result;
        }catch(...){cleanup_after_failure(true);throw;}
    }
    Coordinator::WindowPublication publish(Coordinator& coordinator,const Pending& result) {
        require_current_pending(result);
        // Completion was synchronized before Pending escaped. Coordinator
        // establishes resident + VirtualLocked ownership before exposing tokens.
        // Validate policy input before publication, then commit its fixed value
        // only after the resident/causal transaction succeeds. Aborted proposals
        // and control tokens never train the policy.
        std::optional<Exl3VerifierHorizonPolicy> next_policy;
        if(horizon_enabled_ && result.proposed_rows) {
            next_policy=horizon_policy_;
            if(result.conditional_observations) {
                for(const auto& observation:*result.conditional_observations)next_policy->observe(observation);
            } else {
            const auto& v=result.verification;
            using Source=Exl3VerifierHorizonPolicy::ProposalSource;
            const auto source=!result.costs.neural_rows?Source::unknown:
                *result.costs.neural_rows?Source::neural_b8:
                result.proposed_rows==1?Source::seed_only:Source::suffix;
            next_policy->observe({result.proposed_rows,static_cast<unsigned>(v.accepted),
                static_cast<unsigned>(v.replay_rows),static_cast<unsigned>(v.native_invocations),v.rejected,v.stopped,result.costs,source});
            }
        }
        auto publication=coordinator.publish_window(lease_,result.root,result.verification.committed_tokens);
        fast_mia_parity_pending_=result.verification.fast_mia_parity_pending;
        if(suffix_) suffix_->append(result.verification.committed_tokens);
        if(next_policy)horizon_policy_=*next_policy;
        if(result.conditional_observations)for(const auto& observation:*result.conditional_observations) {
            using Source=Exl3VerifierHorizonPolicy::ProposalSource;
            if(suffix_ && observation.source==Source::neural_b8)suffix_->record_neural_publication();
            if(observation.source==Source::suffix) {
                const unsigned accepted=observation.accepted?observation.accepted-1:0;
                if(suffix_)suffix_->record_published_outcome({observation.proposed-1,accepted,observation.replay});
                ++stats_.suffix_published_rounds;stats_.suffix_accepted_rows+=accepted;
                stats_.suffix_repair_rows+=observation.replay;++stats_.suffix_published_skipped_neural_blocks;
            }
        }
        if(suffix_ && result.costs.neural_rows && *result.costs.neural_rows)
            suffix_->record_neural_publication();
        // A lookup hit is only an attempt. Publish outcome evidence after the
        // coordinator commits, independently of whether horizon learning is on.
        if(result.proposed_rows>1 && result.costs.neural_rows==0) {
            if(suffix_)suffix_->record_published_outcome({result.proposed_rows-1,
                static_cast<unsigned>(result.verification.accepted>0?result.verification.accepted-1:0),
                static_cast<unsigned>(result.verification.replay_rows)});
            ++stats_.suffix_published_rounds;
            stats_.suffix_accepted_rows+=result.verification.accepted>0?result.verification.accepted-1:0;
            stats_.suffix_repair_rows+=result.verification.replay_rows;
            ++stats_.suffix_published_skipped_neural_blocks;
        }
        conditional_witness_.reset();
        conditional_predecessor_.reset();
        lease_=publication.lease;pending_=false;
        if(dependency_ticket_) {
            if(!dependency_graph_.retire(*dependency_ticket_))
                throw std::logic_error("DFlash published dependency graph not complete");
            dependency_ticket_.reset();
        }
        return publication;
    }
    Pending apply_control(std::span<const std::int64_t> tokens) {
        require_ready();
        if(round_==UINT64_MAX)throw std::overflow_error("DFlash round exhausted");
        if(tokens.empty() || tokens.size()>128)throw std::invalid_argument("DFlash control extent");
        Exl3OuterReferenceResult result;
        result.committed_tokens.assign(tokens.begin(),tokens.end());
        proposal_rows_=0;
        try {
            // Control suffix preparation retains its canonical stream0 path.
            // Nonblocking request work must finish before switching streams;
            // append_prompt_compact fences state/ring export before returning.
            // This is an explicit serialization boundary, not control overlap.
            if(stream_)fence();
            auto root=lease_.root->append_control_compact(*context_,draft_,staging_,tokens);
            fence();pending_=true;
            result.committed_state=root->state();
            return Pending{epoch_,++round_,lease_,std::move(root),std::move(result),0,active_completion_generation_};
        }catch(...){cleanup_after_failure(false);throw;}
    }
    void abort() {
        require_no_active_proposal();
        if(epoch_==UINT64_MAX)throw std::overflow_error("DFlash execution epoch exhausted");
        if(!active_ || poisoned_) throw std::logic_error("DFlash abort inactive/poisoned lane");
        // Rebind before restoration so the completed witness belongs to the
        // post-abort epoch. A failed restoration poisons the lane in rollback.
        cancel_dependency_graph();
        ++epoch_;draft_.bind_ring_scope(lease_.acquisition,epoch_);
        rollback();pending_=false;
    }
    Coordinator::Retirement release() {
        require_no_active_proposal();
        if(epoch_==UINT64_MAX || retirement_generation_==UINT64_MAX)
            throw std::overflow_error("DFlash execution retirement epoch exhausted");
        if(!active_ || pending_ || poisoned_ || conditional_retention_charge_.units) throw std::logic_error("DFlash release state or uncollected conditional retention");
        const auto retired_lease=lease_;
        const auto retired_execution=epoch_;
        std::uint64_t retired_final_use=0;
        try {
            fence();retired_final_use=active_completion_generation_;
            retire_active_completion();
        }catch(...){poisoned_=true;throw;}
        draft_.bind_ring_scope(0,epoch_+1);
        failed_retirement_->proposal_root.reset();
        proposal_rows_=0;
        lease_={};active_=false;++epoch_;draft_owner_.unlock();
        return Coordinator::Retirement(retired_lease.ticket,
            retired_lease.acquisition,retired_execution,retired_final_use,
            ++retirement_generation_);
    }
    const Coordinator::Lease& lease() const {return lease_;}
    bool execution_active() const noexcept {return active_ && !poisoned_ && !retired_after_drain_;}
    void set_greedy_packet_batch_finish(GreedyPacketBatchFinish finish) {
        if(active_ || pending_ || proposal_rows_ || retired_after_drain_)
            throw std::logic_error("greedy packet batch installation requires idle lane");
        greedy_packet_batch_finish_=std::move(finish);
    }
    std::uint64_t execution_epoch() const noexcept {return epoch_;}
    Exl3ExecutionDependencyGraph::Snapshot dependency_graph_for_test() const noexcept {
        return dependency_graph_.snapshot();
    }
    bool shared_proposal_active_for_test() const noexcept {
        return failed_retirement_ && failed_retirement_->proposal_active;
    }
    std::shared_ptr<const Exl3VeriCacheRequest> shared_proposal_root() const noexcept {
        return shared_proposal_active_for_test()?failed_retirement_->proposal_root:nullptr;
    }
    bool matches_shared_proposal(const Exl3DraftPrivateSegment& segment,std::int64_t seed) const noexcept {
        return failed_retirement_ && failed_retirement_->proposal_active &&
            failed_retirement_->proposal_root &&
            segment.positions[0]==failed_retirement_->proposal_root->state()->position() &&
            segment.acquisition==lease_.acquisition && segment.execution==epoch_ &&
            segment.matches_tokens(failed_retirement_->proposal_block) &&
            seed==failed_retirement_->proposal_block[0];
    }
    void set_conditional_publication_limit(double maximum_ms,std::uint64_t epoch,std::uint64_t acquisition) {
        require_ready();
        if(proposal_rows_ || epoch!=epoch_ || acquisition!=lease_.acquisition ||
            conditional_publication_constraint_.maximum_ms)
            throw std::invalid_argument("conditional publication constraint request scope");
        Exl3VerifierHorizonPolicy::PublicationConstraint constraint{maximum_ms};
        (void)constraint.allows(std::nullopt); // validate without manufacturing an estimate
        conditional_publication_constraint_=constraint;
    }
    void fail_next_repair_join_for_test(const Pending& pending) {
        require_current_pending(pending);repair_join_failure_for_test_=true;
    }
    void fail_next_tap_copy_for_test(unsigned one_based_copy) {
        require_ready();
        if(!one_based_copy || one_based_copy>80 || tap_copy_failure_for_test_)
            throw std::invalid_argument("tap copy fault requires one unarmed copy1..80");
        tap_copy_failure_for_test_=one_based_copy;
    }
    void invalidate_tap_readiness_for_test(Pending& pending) const {
        require_current_pending(pending);
        if(pending.verification.committed_tap_segments.empty())
            throw std::invalid_argument("tap readiness fixture requires committed taps");
        pending.verification.committed_tap_ready_generation=0;
    }
    void observe_next_tap_fence_for_test(std::function<void()> observer) {
        require_ready();
        if(!observer || tap_fence_observer_for_test_)
            throw std::invalid_argument("tap fence observer requires one idle callback");
        tap_fence_observer_for_test_=std::move(observer);
    }
    bool tap_waiting_for_fence_for_test() const noexcept {
        return tap_waiting_for_fence_.load(std::memory_order_acquire);
    }
    bool has_conditional_retention() const noexcept {return conditional_retention_charge_.units!=0;}
    bool collect_conditional_retention(Coordinator& authority) {
        require_ready();
        if(proposal_rows_ || conditional_predecessor_ || &authority!=conditional_retention_authority_)
            throw std::logic_error("conditional retention collection scope");
        if(!conditional_retention_charge_.units)return true;
        if(conditional_retention_charge_.owner) {
            if(!authority.retire_runtime_metadata_to_lifetime(lease_,conditional_retention_charge_))return false;
            conditional_retired_owner_=conditional_retention_charge_.owner;
            conditional_retention_charge_.owner.reset();
        }
        if(!conditional_retired_owner_.expired())return false;
        conditional_retired_owner_.reset();conditional_retention_charge_={};return true;
    }
    void fail_next_conditional_retention_for_test() {
        require_ready();conditional_retention_failure_for_test_=true;
    }
    void fail_next_conditional_verified_completion_for_test() {
        require_ready();conditional_verified_failure_for_test_=true;
    }
    void substitute_conditional_tokens_for_test(const Pending& first,ConditionalBlock& second,
        std::span<const std::int64_t> tokens) {
        require_current_pending(first);
        if(!conditional_witness_ || second.epoch!=epoch_ || second.predecessor_round!=round_ ||
            second.acquisition!=lease_.acquisition || second.conditioning_root!=first.root ||
            second.conditioning_root!=conditional_witness_->conditioning_root ||
            second.tokens!=conditional_witness_->tokens || second.tokens.size()!=8 || tokens.size()!=8)
            throw std::invalid_argument("conditional test substitution scope/extent");
        for(auto token:tokens)if(token<0 || token>=248320)
            throw std::invalid_argument("conditional test substitution token");
        std::copy(tokens.begin(),tokens.end(),second.tokens.begin());
        std::copy(tokens.begin(),tokens.end(),conditional_witness_->tokens.begin());
        // Physical attempt counters remain, but these tokens are no longer
        // attributed to that neural proposal in publication policy evidence.
        second.costs={};conditional_witness_->costs={};
    }
    void set_suffix_selection_limits(Exl3SuffixProposer::SelectionLimits limits,
        std::uint64_t expected_epoch,std::uint64_t expected_acquisition) {
        require_ready();
        if(!suffix_ || proposal_rows_ || expected_epoch!=epoch_ || expected_acquisition!=lease_.acquisition)
            throw std::invalid_argument("suffix selection requires matching idle request");
        suffix_->set_selection_limits(limits);
        ++stats_.suffix_limit_installations;
    }
    // Supplied request-local estimates only; no autonomous calibration. Never
    // replace the decision basis underneath a returned proposal or Pending.
    void set_horizon_cost_menu(Exl3VerifierHorizonPolicy::CostMenu menu,
        std::uint64_t expected_epoch,std::uint64_t expected_acquisition,
        std::uint64_t minimum_published_rounds=1) {
        require_ready();
        if(!horizon_enabled_ || proposal_rows_ || expected_epoch!=epoch_ ||
            expected_acquisition!=lease_.acquisition)
            throw std::invalid_argument("cost menu requires matching idle request decision boundary");
        const auto published=horizon_policy_.counters().published_rounds;
        if(!minimum_published_rounds || (horizon_policy_.has_cost_menu() &&
            (published<cost_menu_update_round_ || published-cost_menu_update_round_<cost_menu_hold_rounds_)))
            throw std::invalid_argument("cost menu update before publication hold interval");
        horizon_policy_.set_cost_menu(menu);
        cost_menu_update_round_=published;cost_menu_hold_rounds_=minimum_published_rounds;
        ++stats_.cost_menu_updates;
    }
    // A shared consumer could have written this lane's scratch. Suppress all
    // rollback submission until Engine's retained global retirement boundary.
    void poison_shared_completion() noexcept {poisoned_=true;}
    void fail_next_rollback_for_test() noexcept {rollback_failure_for_test_=true;}
    std::weak_ptr<const void> conditional_predecessor_owner_for_test() const noexcept {return conditional_predecessor_;}
    int first_completion_error() const noexcept {return completion_pool_.first_error();}
    // Engine has joined every producer and successfully drained the device.
    // This is a destruction certificate, not permission to recycle a request.
    void retire_after_device_drain() noexcept {retired_after_drain_=true;}
    const Stats& stats() const {return stats_;}
    const Exl3VerifierHorizonPolicy& verifier_horizon_policy() const noexcept {return horizon_policy_;}
    bool verifier_horizon_enabled() const noexcept {return horizon_enabled_;}
    bool repair_checkpoint_enabled() const noexcept {return repair_checkpoint_enabled_;}
    bool suffix_proposals_enabled() const noexcept {return suffix_!=nullptr;}
    Exl3TextContext& context() {return *context_;}
    std::shared_ptr<Exl3TextContext> context_owner_for_device_round() const noexcept {
        return context_;
    }
    std::size_t persistent_bytes() const {return context_->persistent_bytes()+5ULL*16*5120*2;}
    // Construction snapshot only: excludes the context supplied by the caller.
    // Later dynamic context growth is inventoried by its own transition.
    std::size_t construction_added_device_bytes() const noexcept {return construction_added_device_bytes_;}
    std::size_t context_entry_device_bytes() const noexcept {return context_entry_device_bytes_;}
private:
    static unsigned suffix_key_tokens_option(bool suffix_enabled) {
        const auto* value=std::getenv("NINFER_EXL3_SUFFIX_KEY_TOKENS");
        if(!value)return 4;
        if(!suffix_enabled)
            throw std::invalid_argument("suffix key tokens require suffix proposals");
        if(std::string_view(value)=="2")return 2;
        if(std::string_view(value)=="3")return 3;
        if(std::string_view(value)=="4")return 4;
        throw std::invalid_argument("suffix key tokens must be2,3,or4");
    }
    static bool preserve_acquired_root_option() {
        const auto* value=std::getenv("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT");
        if(value && std::string_view(value)!="0" && std::string_view(value)!="1")
            throw std::invalid_argument("preserve acquired root must be0 or1");
        return value && std::string_view(value)=="1";
    }
    static void require_retirement_admission() {
        if(failed_retirement_count_.load())throw std::runtime_error("unresolved DFlash execution retirement");
        if(Exl3TextContext::host_kv_quarantined_contexts())
            throw std::runtime_error("unresolved HostKV retirement blocks DFlash execution");
        if(Exl3TextContext::generic_quarantined_allocations())
            throw std::runtime_error("unresolved generic cleanup blocks DFlash execution");
    }
    static Exl3Dflash2DraftModel& require_retained_draft(
        const std::shared_ptr<Exl3Dflash2DraftModel>& draft,cudaStream_t stream,
        const std::shared_ptr<const void>& stream_owner) {
        if(!draft || (stream && !stream_owner))throw std::invalid_argument("retained DFlash draft/stream owner missing");
        return *draft;
    }
    using Clock=std::chrono::steady_clock;
    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    }
    static void check(cudaError_t error) {if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
    void observe_tap_before_fence() {
        auto observer=std::move(tap_fence_observer_for_test_);
        tap_fence_observer_for_test_={};
        if(!observer)return;
        tap_waiting_for_fence_.store(true,std::memory_order_release);
        try {observer();}
        catch(...) {tap_waiting_for_fence_.store(false,std::memory_order_release);throw;}
        tap_waiting_for_fence_.store(false,std::memory_order_release);
    }
    void bind_tap_readiness(Exl3OuterReferenceResult& result) const noexcept {
        if(result.committed_tap_segments.empty())return;
        result.committed_tap_ready_generation=active_completion_generation_;
        result.committed_tap_ready_event=completion_pool_.event(
            active_completion_generation_);
    }
    void retire_active_completion() {
        if(!active_completion_generation_)return;
        if(!completion_pool_.retire(active_completion_generation_))
            throw std::logic_error("DFlash completion event not ready for retirement");
        active_completion_generation_=0;
    }
    void fence(std::uint64_t acquisition=0,std::uint64_t execution=0) {
        retire_active_completion();
        if(retirement_generation_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("DFlash completion retirement generation exhausted");
        const auto ticket=completion_pool_.acquire(
            acquisition?acquisition:lease_.acquisition,
            execution?execution:epoch_,retirement_generation_+1,lease_.root);
        if(!ticket)throw std::runtime_error("DFlash completion event pool exhausted");
        active_completion_generation_=ticket->generation;
        if(!completion_pool_.submit(*ticket))
            throw std::logic_error("DFlash completion event submission refused");
        const auto dependency_before=dependency_graph_.snapshot();
        if(dependency_before.phase==Exl3ExecutionDependencyGraph::Phase::building) {
            if(!dependency_ticket_)throw std::logic_error(
                "DFlash dependency graph ticket missing");
            try {
                dependency_graph_.submit_selected_slice(*dependency_ticket_,ticket->event);
                ++stats_.dependency_graph_submissions;
            } catch(...) {
                dependency_graph_.fail(*dependency_ticket_,-1);
                ++stats_.dependency_graph_failures;
                throw;
            }
        }
        auto error=cudaEventRecord(reinterpret_cast<cudaEvent_t>(ticket->event),stream_);
        if(error==cudaSuccess)
            error=cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(ticket->event));
        const auto notified=completion_pool_.notify(
            *ticket,ticket->event,static_cast<int>(error));
        if(dependency_graph_.snapshot().phase==
                Exl3ExecutionDependencyGraph::Phase::submitted) {
            if(error==cudaSuccess && notified) {
                if(!dependency_graph_.complete(*dependency_ticket_,ticket->event,0))
                    throw std::logic_error("DFlash dependency completion refused");
                ++stats_.dependency_graph_completions;
            } else {
                dependency_graph_.complete(*dependency_ticket_,ticket->event,
                    error==cudaSuccess?-1:static_cast<int>(error));
                ++stats_.dependency_graph_failures;
            }
        }
        if(error==cudaSuccess && !notified) {
            poisoned_=true;
            throw std::logic_error("DFlash completion event notification refused");
        }
        if(error!=cudaSuccess)poisoned_=true;
        check(error);
    }
    std::uint64_t conditional_second_round_=0;
    Exl3VerifierHorizonPolicy::PublicationConstraint conditional_publication_constraint_;
    bool conditional_retention_failure_for_test_=false;
    bool repair_join_failure_for_test_=false;
    bool conditional_verified_failure_for_test_=false;
    unsigned tap_copy_failure_for_test_=0;
    std::function<void()> tap_fence_observer_for_test_;
    std::atomic<bool> tap_waiting_for_fence_{false};
    std::optional<ConditionalBlock> conditional_witness_;
    // Finalized before installation. Keep charged token/tap payload immutable
    // throughout second verification and retirement; only local construction
    // may update the predecessor completion generation.
    std::shared_ptr<const Pending> conditional_predecessor_;
    Coordinator* conditional_retention_authority_=nullptr; // identity only; never dereferenced during teardown
    Exl3ResourceInventory::Allocation conditional_retention_charge_;
    std::weak_ptr<const void> conditional_retired_owner_;
    void require_current_pending(const Pending& result) const {
        require_no_active_proposal();
        require_retirement_admission();
        if(!active_ || !pending_ || poisoned_ || result.epoch!=epoch_ || result.round!=round_ ||
            (result.proposed_rows && result.policy_revision!=horizon_policy_.counters().published_rounds) ||
            result.parent.root!=lease_.root || result.parent.ticket.request_id!=lease_.ticket.request_id ||
            result.parent.ticket.generation!=lease_.ticket.generation ||
            result.parent.acquisition!=lease_.acquisition ||
            !completion_pool_.matches(completion_authority_generation_,
                lease_.acquisition,epoch_,retirement_generation_+1,
                result.completion_generation) ||
            (!result.verification.committed_tap_segments.empty() &&
             (result.verification.committed_tap_ready_generation!=result.completion_generation ||
              result.verification.committed_tap_ready_event!=
                completion_pool_.event(result.completion_generation))))
            throw std::invalid_argument("DFlash stale completion");
    }
    void require_ready() const {
        require_no_active_proposal();
        require_retirement_admission();
        if(!active_ || pending_ || poisoned_ || retired_after_drain_)throw std::logic_error("DFlash lane not ready");
    }
    void require_no_active_proposal() const {
        if(failed_retirement_ && failed_retirement_->proposal_active)
            throw std::logic_error("draft proposal already active");
    }
    void rollback() {
        conditional_retention_failure_for_test_=false;
        conditional_verified_failure_for_test_=false;
        proposal_rows_=0;
        fast_mia_parity_pending_.reset();
        if(poisoned_)return; // preserve the original failure; no new GPU work on uncertain owners
        try {
            if(std::exchange(rollback_failure_for_test_,false))
                throw std::runtime_error("injected DFlash rollback completion failure");
            context_->restore_exact_host_state(*lease_.root->state(),stream_);
            lease_.root->restore_draft(draft_,staging_,stream_);fence();
            retire_dependency_after_stream_drain();
            // Conditional roots/verification storage can be touched by work
            // preceding rollback. Keep them through the rollback's final fence,
            // including any failure that poisons this lane for retirement.
            conditional_witness_.reset();conditional_predecessor_.reset();
            repair_join_failure_for_test_=false;
        }
        catch(...) {poisoned_=true;throw;}
    }
    // Called only while handling an earlier operation failure. Cleanup may
    // poison ownership, but must not replace the exception that caused it.
    void cleanup_after_failure(bool advance_epoch) noexcept {
        try {
            if(advance_epoch) {if(!poisoned_)abort();}
            else rollback();
        } catch(...) {poisoned_=true;}
    }
    void free_staging() {for(auto& p:staging_)if(p){cudaFree(p);p=nullptr;}}
    // Declared before context so stream/draft backing survives context cleanup.
    std::shared_ptr<const void> retained_stream_;
    std::unique_ptr<FailedRetirement> failed_retirement_=std::make_unique<FailedRetirement>();
    bool destructor_drain_failure_for_test_=false;
    bool rollback_failure_for_test_=false;
    bool repair_checkpoint_enabled_=false;
    bool device_seed_handoff_enabled_=false;
    GreedyPacketBatchFinish greedy_packet_batch_finish_;
    std::shared_ptr<Exl3Dflash2DraftModel> retained_draft_;
    std::shared_ptr<Exl3TextContext> context_;
    std::size_t context_entry_device_bytes_=0,construction_added_device_bytes_=0;
    void begin_dependency_graph() {
        if(!stream_)return; // Selected slice requires an explicit retained stream.
        if(dependency_graph_.snapshot().phase!=
                Exl3ExecutionDependencyGraph::Phase::idle)
            throw std::logic_error("DFlash prior dependency graph not retired");
        dependency_ticket_=dependency_graph_.begin(lease_.acquisition,epoch_,
            reinterpret_cast<std::uintptr_t>(stream_),retained_stream_);
        ++stats_.dependency_graph_begins;
        dependency_graph_.add_node(*dependency_ticket_,
            Exl3ExecutionDependencyGraph::Stage::draft,retained_draft_);
    }
    void bind_target_dependencies(
        const std::shared_ptr<const Exl3VeriCacheRequest>& exported) {
        if(!dependency_ticket_)return;
        using Stage=Exl3ExecutionDependencyGraph::Stage;
        dependency_graph_.add_node(*dependency_ticket_,Stage::upload,context_);
        dependency_graph_.add_node(*dependency_ticket_,Stage::attention,context_);
        dependency_graph_.add_node(*dependency_ticket_,Stage::export_state,exported);
        dependency_graph_.add_edge(*dependency_ticket_,Stage::draft,Stage::upload);
        dependency_graph_.add_edge(*dependency_ticket_,Stage::upload,Stage::attention);
        dependency_graph_.add_edge(*dependency_ticket_,Stage::attention,Stage::export_state);
    }
    void fail_dependency_graph() noexcept {
        if(dependency_ticket_ && dependency_graph_.fail(*dependency_ticket_,-1))
            ++stats_.dependency_graph_failures;
    }
    void cancel_dependency_graph() noexcept {
        if(dependency_ticket_ && dependency_graph_.cancel(*dependency_ticket_))
            ++stats_.dependency_graph_cancellations;
    }
    void retire_dependency_after_stream_drain() noexcept {
        if(dependency_ticket_ &&
           dependency_graph_.retire_after_stream_drain(*dependency_ticket_))
            dependency_ticket_.reset();
    }
    void remember_proposal(std::span<const std::int64_t> tokens,unsigned neural_rows,double neural_ms) {
        std::copy(tokens.begin(),tokens.end(),proposal_tokens_.begin());
        proposal_rows_=static_cast<unsigned>(tokens.size());
        proposal_epoch_=epoch_;proposal_round_=round_;proposal_position_=context_->position();
        proposal_policy_revision_=horizon_policy_.counters().published_rounds;
        proposal_costs_={};proposal_costs_.neural_rows=neural_rows;proposal_costs_.neural_ms=neural_ms;
        begin_dependency_graph();
    }
    std::array<std::int64_t,8> proposal_tokens_{};
    unsigned proposal_rows_=0;
    std::uint64_t proposal_epoch_=0,proposal_round_=0,proposal_policy_revision_=0;
    int proposal_position_=0;
    Exl3VerifierHorizonPolicy::Costs proposal_costs_;
    Exl3Dflash2DraftModel& draft_;
    std::string compatibility_;
    cudaStream_t stream_=nullptr;
    std::array<std::uint16_t*,5> staging_{};
    std::unique_ptr<Exl3SuffixProposer> suffix_;
    std::array<cudaEvent_t,Exl3RequestCompletionEventPool::capacity> completion_events_{};
    std::uint64_t completion_authority_generation_=
        acquire_completion_authority_generation();
    Exl3RequestCompletionEventPool completion_pool_;
    std::uint64_t active_completion_generation_=0;
    Exl3ExecutionDependencyGraph dependency_graph_;
    std::optional<Exl3ExecutionDependencyGraph::Ticket> dependency_ticket_;
    Coordinator::Lease lease_;
    std::uint64_t epoch_=0,round_=0,retirement_generation_=0;
    std::optional<std::int64_t> fast_mia_parity_pending_;
    bool active_=false,pending_=false,poisoned_=false;
    bool retired_after_drain_=false;
    bool horizon_enabled_=false;
    bool native_horizon_enabled_=false;
    bool staged_b8_verifier_enabled_=false;
    unsigned staged_b8_split_rows_=4;
    bool preserve_acquired_root_=false;
    Exl3VerifierHorizonPolicy horizon_policy_;
    std::uint64_t cost_menu_update_round_=0,cost_menu_hold_rounds_=1;
    unsigned last_full_budget_horizon_=0;
    Stats stats_;
    std::unique_lock<std::mutex> draft_owner_;
};
} // namespace ninfer::exl3
