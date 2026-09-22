#pragma once
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "exl3/retirement_state.h"
#include "exl3/verifier_horizon_policy.h"
#include "exl3/suffix_proposer.h"
#include "exl3/vericache_request.h"
#include "exl3/resource_availability.h"
#include "exl3/ready_work_descriptor.h"
#include <ninfer/targets/qwen3_6/frontend.h>

namespace ninfer::exl3 {
class Exl3VeriCacheRequest;
class Exl3PackedCostPolicy;
struct Exl3DevicePageWeakComponents;
// Concrete pinned EXL3 package/core behind the existing Engine facade. It does
// not alias a .ninfer package or use that package's numerical Program.
class Exl3EngineCore {
    struct Impl;
    struct Request;
public:
    using ReadyCopyDemandProvider=std::function<Exl3ReadyCopyDemand(
        std::span<const TokenId>)>;
    class Submission {
    public:
        Submission(Submission&&) noexcept;
        Submission& operator=(Submission&&) noexcept;
        ~Submission();
        GenerationPollResult poll(OutputSink*,const CancellationView&);
        GenerationResult wait(OutputSink*,const CancellationView&);
        void set_delivery_byte_limit_for_test(std::size_t);
        std::function<void()> cancellation_callback_for_test() const;
        std::weak_ptr<const void> result_owner_for_test() const;
        std::uint64_t result_reserved_bytes_for_test() const;
        std::uint64_t result_fixed_bytes_for_test() const;
        std::uint64_t option_storage_bytes_for_test() const;
        std::uint64_t prompt_storage_bytes_for_test() const;
        std::uint64_t initial_session_bytes_for_test() const;
        std::uint64_t observed_session_bytes_for_test() const;
        std::uint64_t outstanding_delivery_batches_for_test() const;
        Exl3ReadyWorkObservation ready_work_for_test() const;
    private:
        explicit Submission(std::shared_ptr<Request>);
        std::shared_ptr<Request> request_;
        friend class Exl3EngineCore;
    };
    explicit Exl3EngineCore(const EngineOptions&,unsigned shared_allocation_fault_for_test=0,
        unsigned draft_clone_fault_for_test=0,unsigned first_draft_fault_for_test=0,
        unsigned context_startup_fault_for_test=0,
        Exl3DeviceAvailability::Provider device_availability_provider_for_test={});
    // Fault123/124 fails the first represented-prefix shared control allocation
    // with clean/failed storage cleanup. Requires the represented-prefix route.
    // Fault125 fails shared-page container publication after bounded allocation;
    // requires the shared-pages route and preserves normal startup unwind.
    // Startup fault1..11 targets lane0,12..22 lane1: context1..4,
    // continuation1..3, lane1..4. Requires a reserved-context Engine route.
    // Fault23..25 targets stream0 and26..28 stream1: before construction,
    // before commit, actual metadata mismatch. Existing codes remain stable.
    // Fault31 leaves first-stream metadata one byte short after private drafts.
    // Fault41..50 fails after the corresponding C1/C2 private tap allocation.
    // Fault51..53: tap precommit, actual device mismatch, actual metadata mismatch.
    // Fault61..63: cancellation guard preconstruction, precommit, metadata mismatch.
    // Fault64: one-byte-short cancellation guard metadata credit, before allocator.
    // Fault71..73: request queue preconstruction, precommit, actual extent mismatch.
    // Fault74: initial host limit one byte short of queue credit, before allocator.
    // Fault75..78: coordinator metadata preconstruction, entry allocation,
    // actual extent mismatch, and initial credit refusal before allocator.
    ~Exl3EngineCore();
    // Read-only latch/capacity preflight; performs no device or model setup.
    static void require_startup_retirement_available(const EngineOptions& options);
    // Owning shutdown only; caller serializes close against frontend/submission.
    // Failure retains the complete Engine bundle and forbids process reload.
    Exl3RetirementResult close() noexcept;
    void fail_retirement_drain_for_test(int error);
    void fail_next_admission_budget_for_test();
    void fail_next_output_extent_for_test();
    void fail_next_window_publication_for_test();
    // Fail after record(1), token vector(2), reasoning(3), or content(4) allocation;
    // stage5 fails Request construction after the result reservation commits.
    // Stage6 fails shared_ptr control-block allocation after Request construction.
    // Stage7 fails after host output-session construction and capacity validation.
    // Stage8 fails after reserved input-token conversion before session construction.
    // Stage9 fails after reserved immutable control-token conversion.
    void fail_next_result_allocation_for_test(unsigned stage=4);
    std::weak_ptr<const void> cancellation_owner_for_test() const;
    std::size_t request_queue_capacity_for_test() const;
    static std::size_t cancellation_owner_blocks_for_test() noexcept;
    static std::size_t execution_stream_owner_bytes_for_test() noexcept;
    static std::size_t execution_stream_owner_blocks_for_test() noexcept;
    static std::size_t shared_projection_owner_blocks_for_test() noexcept;
    std::uint64_t shared_projection_metadata_bytes_for_test() const;
    std::weak_ptr<const void> shared_projection_owner_for_test() const noexcept;
    std::optional<std::uint64_t> shared_projection_retirement_credit_for_test() const noexcept;
    static int execution_stream_retirement_error_for_test() noexcept;
    static std::size_t execution_stream_retirement_slots_for_test() noexcept;
    static std::size_t execution_stream_quarantined_handles_for_test() noexcept;
    void fail_execution_stream_destroy_for_test(unsigned lane,int error);
    static std::size_t result_owner_blocks_for_test() noexcept;
    std::uint64_t retained_host_metadata_for_test() const;
    std::vector<Exl3ResourceInventory::Attribution> resource_attribution_for_test() const;
    // Idle-only monotonic tightening of the real ceiling, including retained
    // cache, result and weak-owner credits. Returns usage at installation.
    std::uint64_t tighten_metadata_headroom_for_test(std::uint64_t headroom);
    std::shared_ptr<const void> retain_host_kv_workspace_for_test();
    std::uint64_t request_queue_metadata_bytes_for_test() const;
    void fail_next_shared_q_completion_for_test(bool before_producer_drain=false,std::function<void()> observer={},
        bool after_first_scatter=false,unsigned family=15);
    void invalidate_next_shared_offer_for_test();
    void split_shared_contracts_for_test(bool enabled);
    void set_shared_preclaim_fault_for_test(unsigned fault);
    void expire_next_shared_pair_for_test();
    void claim_next_shared_pair_at_age_for_test(std::chrono::microseconds);
    std::uint64_t expired_shared_pair_refusals_for_test();
    std::array<std::weak_ptr<const void>,8> shared_claim_owners_for_test();
    void set_shared_rendezvous_timeout_for_test(std::chrono::microseconds timeout);
    void set_greedy_packet_batch_timeout_for_test(std::chrono::microseconds timeout);
    void observe_next_shared_wait_for_test(std::function<void()> observer);
    void set_shared_cost_policy_for_test(const Exl3PackedCostPolicy&);
    // Supplied microsecond estimates only; existing sharing opt-ins and exact
    // dispatch restrictions remain required. Empty policy restores fallback.
    void set_shared_cost_policy(const Exl3PackedCostPolicy&);
    // Explicit supplied estimates; mutation requires an idle Engine and the
    // existing bounded-horizon opt-in. Each acquisition gets a fresh policy.
    void set_verifier_cost_menu(std::optional<Exl3VerifierHorizonPolicy::CostMenu>);
    void set_suffix_selection_limits(std::optional<Exl3SuffixProposer::SelectionLimits>);
    // Optional modeled/measured transfer demand only. The provider runs before
    // queue publication and must return immutable bounded hints. Empty restores
    // age-only scheduling. Mutation requires an idle live Engine.
    void set_ready_copy_demand_provider(ReadyCopyDemandProvider);
    // Optional fixed-capacity scheduling provenance. Enabling/resetting requires
    // an idle live Engine; snapshots contain descriptor hashes and aggregate
    // decisions only, never prompt content. Disabled is the production default.
    void enable_ready_decision_trace_for_test(bool);
    Exl3ReadyDecisionTrace::Snapshot ready_decision_trace_for_test() const;
    // Explicit, one-shot supplied retention policy. Both operations require an
    // idle live core; generations bind the observation to its exact cache entry.
    std::vector<Exl3VeriCachePrefixIndex::RetentionMetadata> prefix_retention_metadata();
    Exl3VeriCachePrefixIndex::PolicyTrim trim_prefix_retention(
        std::uint64_t available_physical_bytes,
        std::span<const Exl3VeriCachePrefixIndex::RetentionDecision>);
    Exl3VeriCachePrefixIndex::Admission admit_prefix_input_with_retention(
        std::shared_ptr<const Exl3VeriCacheRequest> root,std::span<const std::int64_t> input,
        std::uint64_t available_physical_bytes,
        std::span<const Exl3VeriCachePrefixIndex::RetentionDecision>);
    void set_registered_upload_for_test(bool enabled);
    // Idle C1 diagnostic holds the real cache external-reader guard across
    // requests. It prevents eviction, without forcing registration misses.
    void hold_registration_reader_for_test(bool hold);
    void set_host_kv_routes_for_test(bool registered_upload,bool shared_pages);
    // Idle-only diagnostic: present a foreign model owner to the page ancestry
    // gate. Numerical execution and private fallback retain the real model.
    void use_foreign_page_ancestor_for_test(bool enabled);
    void set_attention_staging_for_test(bool enabled);
    void refuse_attention_read_pairs_for_test(bool enabled);
    // C1 fixture: 1 copy, 2 producer record, 3 producer wait,
    // 4 consumer record, 5 consumer wait. One shot.
    void fail_attention_staging_for_test(unsigned stage);
    void fail_reconstruction_retirement_for_test();
    // Fresh idle C1, pinned export enabled; 1/2 post-registration throw, 3/4
    // event-create failure; odd modes fail cleanup, even modes clean normally.
    void fail_recurrent_constructor_for_test(unsigned fault);
    // 1/2: preparation slab/borrower; 3/4: active-lease slab/borrower.
    void fail_recurrent_commit_for_test(unsigned fault);
    // One-shot cancellation after the recurrent slab factory returns but
    // before its inventory can publish: 1 preparation, 2 active lease.
    void cancel_recurrent_growth_after_factory_for_test(unsigned phase);
    void cancel_next_active_snapshot_for_test();
    // Idle C1 conditional Engine; 0 clears, 1 before dependent proposal,
    // 2 after proposal, 3 after verification. One-shot request cancellation.
    void cancel_conditional_stage_for_test(unsigned stage);
    unsigned conditional_cancel_hit_for_test() const;
    // One-shot safe-boundary cancellation during lane attachment: 1 after
    // target reset, 2 after target restore, 3 after target+draft restore/fence.
    void cancel_next_attachment_for_test(unsigned phase);
    unsigned attachment_cancel_hit_for_test() const;
    void cancel_next_attachment_during_restore_for_test(unsigned completed_layer);
    unsigned attachment_restore_cancel_hit_for_test() const;
    // 1/2 synthetic preparation/active refusal; 3 tightens the real ceiling
    // at the next lease-authorized nested reservation, after result admission.
    void fail_request_metadata_for_test(unsigned phase);
    // Idle C1 copied-tap route: 1 after payload reservation, 2 after tracking
    // the third plane. Both are one-shot, after the actual lease gate.
    void fail_request_host_payload_for_test(unsigned phase);
    unsigned request_host_payload_fault_hit_for_test() const;
    void fail_next_draft_export_completion_for_test();
    std::weak_ptr<const void> uncertain_draft_source_owner_for_test() const;
    unsigned request_metadata_fault_hit_for_test() const;
    unsigned recurrent_commit_fault_hit_for_test() const;
    unsigned recurrent_growth_cancel_hit_for_test() const;
    std::uint64_t recurrent_event_create_failures_for_test() const;
    void fail_context_linear_retirement_for_test(unsigned slot,bool partial=false);
    void fail_attention_buffer_retirement_for_test(unsigned owned_index);
    void fail_gdn_buffer_retirement_for_test(unsigned owned_index);
    std::uint64_t fail_layer_buffer_retirement_for_test(bool gdn,unsigned layer,unsigned slot);
    void fail_continuation_allocation_retirement_for_test(unsigned slot,unsigned fault=1);
    // 1/2/3 fill copy/record/wait; 4/5/6 D2D copy/record/wait;
    // 7/8 direct attention final-use record/wait. One shot.
    // Callback runs once on the consuming worker, outside Engine locks, before
    // the injected provider is returned. Install only while idle.
    void fail_next_shared_page_for_test(unsigned stage,std::function<void()> before_failure={});
    void observe_next_shared_page_fill_for_test(std::function<void()> observer);
    // Idle shared-page Engine; stages1..4 follow actual storage/fill/View construction.
    void fail_next_page_constructor_for_test(unsigned stage,bool fail_cleanup=false);
    Exl3DevicePageWeakComponents shared_page_weak_components_for_test();
    void observe_terminal_roots_for_test(std::function<void(std::shared_ptr<const Exl3VeriCacheRequest>)> observer);
    // After lane acquisition, before model/control decode. Idle install;
    // C2 callers must synchronize observer-owned state.
    void observe_acquired_roots_for_test(std::function<void(std::shared_ptr<const Exl3VeriCacheRequest>)> observer);
    // Worker-owned request, before device selection or execution. Idle install.
    void observe_request_start_for_test(std::function<void()> observer);
    // Called after the FIFO head is assigned and outside the Engine lock.
    // `affine` means exact resident-root affinity selected this idle lane.
    void observe_lane_assignment_for_test(
        std::function<void(std::size_t lane,bool affine)> observer);
    // Idle C2 seam: invalidate a retained affinity generation without changing
    // the context, for deterministic stale-witness source coverage.
    void stale_lane_affinity_for_test(std::size_t lane);
    // Concurrent preparation only; zero denotes a joined waiter's poll, positive
    // positions denote private prefill progress. Callback may run on both lanes
    // without Engine/cache locks; fixture owns synchronization. Idle install.
    void observe_prefix_preparation_for_test(std::function<void(std::size_t,int)> observer);
    // Invoked after the immutable host prefix plan completes and before that
    // request accepts it. peer_numerical is an observation, not a grant.
    void observe_host_preparation_for_test(
        std::function<void(std::size_t,std::uint64_t,bool)> observer);
    void fail_next_prefix_preparation_admission_for_test();
    targets::qwen3_6::Frontend& frontend();
    ModelSamplingDefaults sampling_defaults() const;
    LoadSummary load_summary() const;
    MemorySummary memory_summary() const;
    RuntimeStats runtime_stats() const;
    void reset_memory_peaks() noexcept;
    Submission submit(targets::qwen3_6::PreparedPrompt,PromptSummary,double,
        runtime::ResolvedRequestOptions,OutputConsumerMode,std::chrono::steady_clock::time_point);
private:
    std::unique_ptr<Impl> impl_;
};
}
