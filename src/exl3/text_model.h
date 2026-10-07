#pragma once
#include "exl3/gaming_optimizations.h"
#include "exl3/target_q_continuation.h"
#include "exl3/exact_kv_extent.h"
#include "exl3/resource_inventory.h"
#include "exl3/attention_history_coverage.h"
#include "exl3/device_page_attention.h"
#include "exl3/attention_stage_command.h"
#include "exl3/attention_stage_history.h"

#include "exl3/oscar_runtime.h"

#include "exl3/full_attention_layer.h"
#include "exl3/gdn_layer.h"
#include "exl3/safetensors.h"
#include "exl3/recurrent_export_pool.h"
#include "exl3/greedy_packet.h"
#include "exl3/mtp_runner.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <functional>

namespace ninfer::exl3 {

class Exl3TextContext;
class Exl3VeriCacheServingCoordinator;
struct Exl3GreedyPacketTransfer;

struct Exl3CommittedTapSegment {
    std::shared_ptr<const void> owner;
    std::shared_ptr<const void> model_identity;
    // Immutable request-root revision that authorized this target continuation.
    // This is distinct from the mutable physical context generation below.
    std::shared_ptr<const void> root_revision;
    std::array<const std::uint16_t*,5> planes{};
    const std::atomic<std::uint64_t>* current_generation=nullptr;
    std::uint64_t generation=0,acquisition=0,execution=0;
    int root_position=0,source_position=0,source_first=0,rows=0;
    int logical_first=0,destination_first=0;
    int attempt_first=0,attempt_rows=0;
    int source_partition_first=0,source_partition_rows=0;
    bool repair=false,correction=false;
    bool current() const noexcept {
        return owner && model_identity && root_revision && current_generation && generation &&
            current_generation->load(std::memory_order_acquire)==generation;
    }
    void validate() const;
};

struct Exl3CommittedTapBinding {
    std::shared_ptr<const void> context_owner;
    std::shared_ptr<const void> root_revision;
    int root_position=-1;
    std::uint64_t acquisition=0,execution=0;
    explicit operator bool() const noexcept {
        return context_owner && root_revision && root_position>=0 && acquisition && execution;
    }
};

struct Exl3DeviceGreedySeed {
    std::shared_ptr<const void> owner;
    std::shared_ptr<const void> model_owner;
    std::shared_ptr<const void> model_identity;
    const Exl3GreedyRow* device_row=nullptr;
    const std::uint16_t* target_embedding_bf16=nullptr;
    Exl3CudaLinearWeights target_head{};
    Exl3CudaLinearMetadata target_head_metadata{};
    cudaEvent_t producer_ready=nullptr,consumer_done=nullptr;
    std::atomic<bool>* consumer_claimed=nullptr;
    std::atomic<bool>* consumer_recorded=nullptr;
    cudaStream_t* consumer_stream=nullptr;
    std::uint64_t acquisition=0,execution=0,generation=0,serial=0;
    int position=0;
};

// Deferred bounded-row source for an Engine-owned transport gather. The source
// retains its private device result until the batch consumer records final use.
struct Exl3GreedyBatchSource {
    std::shared_ptr<const void> owner;
    // Strong physical dependencies for the whole pending interval. The
    // transfer owns its device/registered-host storage and events; model_owner
    // keeps projection metadata alive, and stream_owner (when nondefault)
    // owns the exact producer stream handle.
    std::shared_ptr<const void> model_owner;
    std::shared_ptr<const void> model_identity;
    std::shared_ptr<const void> stream_owner;
    cudaStream_t producer_stream=nullptr;
    const Exl3GreedyRow* device_row=nullptr;
    cudaEvent_t producer_ready=nullptr,consumer_done=nullptr;
    std::atomic<bool>* consumer_claimed=nullptr;
    std::atomic<bool>* consumer_recorded=nullptr;
    cudaStream_t* consumer_stream=nullptr;
    std::uint64_t acquisition=0,execution=0,generation=0,serial=0;
    std::uint64_t final_use_generation=0;
    std::uintptr_t final_use_event=0;
    std::size_t device_bytes=0,registered_host_bytes=0,metadata_bytes=0;
    int position=0,rows=0;
    bool owns_pending_execution() const noexcept {
        return owner && model_owner && model_identity && device_row &&
            producer_ready && consumer_done && consumer_claimed &&
            consumer_recorded && consumer_stream && acquisition && execution &&
            generation && serial && final_use_generation &&
            final_use_event==reinterpret_cast<std::uintptr_t>(consumer_done) &&
            device_bytes && registered_host_bytes && metadata_bytes &&
            (!producer_stream || stream_owner);
    }
};

class Exl3PendingGreedyPacket {
public:
    Exl3PendingGreedyPacket() = default;
    Exl3PendingGreedyPacket(Exl3PendingGreedyPacket&&) noexcept = default;
    Exl3PendingGreedyPacket& operator=(Exl3PendingGreedyPacket&&) noexcept = default;
    Exl3PendingGreedyPacket(const Exl3PendingGreedyPacket&) = delete;
    Exl3PendingGreedyPacket& operator=(const Exl3PendingGreedyPacket&) = delete;
    bool valid() const noexcept { return owner_ != nullptr; }
private:
    friend class Exl3TextContext;
    std::shared_ptr<Exl3GreedyPacketTransfer> owner_;
    std::uint64_t acquisition_=0,execution_=0,generation_=0,serial_=0;
    int position_=0,rows_=0;
    bool continuation_=false;
};
class Exl3TurboAngleWarmPages;

enum Exl3ContinuationGraphRouteBits : std::uint32_t {
    Exl3ContinuationGraphRouteFixedB8 = 1u << 0,
    Exl3ContinuationGraphRouteChronologicalOscar = 1u << 1,
    Exl3ContinuationGraphRouteGdnQkvzConcurrent = 1u << 2,
    Exl3ContinuationGraphRouteVariableRows = 1u << 3,
};

// Allocation-free cohort admission snapshot.  This is deliberately a
// read-only description of one independently owned graph/context, not a
// cross-context execution capability.
struct Exl3ContinuationGraphAdmission {
    int capacity = 0;
    bool active = false;
    std::uint64_t route_generation = 0;
    std::uint64_t capture_count = 0;
    std::uint64_t compatible_reuse_count = 0;
    int oscar_split_class = 0;
    int position = 0;
    int remaining_position_extent = 0;
    bool position_ready = false;
    bool ready = false;
    std::uintptr_t graph_origin_stream_identity = 0;
    std::uint32_t qualified_route_bits = 0;
};

// Phase counters are intentionally independent. Replay submission is host
// launch work, while final-use observation follows a caller-owned fence; they
// are not additive elapsed-time components. CUDA driver graph bytes are not
// observable from the supported source contract and therefore remain unknown
// instead of being reported as zero.
struct Exl3ContinuationGraphStats {
    std::uint64_t preparation_attempts = 0;
    std::uint64_t preparation_successes = 0;
    std::uint64_t compatible_active_hits = 0;
    std::uint64_t compatible_reactivations = 0;
    std::uint64_t eager_fallbacks = 0;
    std::uint64_t replay_submissions = 0;
    std::uint64_t final_use_observations = 0;
    std::uint64_t narrow_final_use_drains = 0;
    std::uint64_t device_final_use_drains = 0;
    std::uint64_t destruction_attempts = 0;
    std::uint64_t destruction_successes = 0;
    std::uint64_t destruction_failures = 0;
    std::uint64_t live_definitions = 0;
    std::uint64_t live_executables = 0;
    // B4/B6/B8 entries may bind the same model/context allocations. Keep
    // their extents separate; callers must not add them as physical memory.
    std::array<std::uint64_t,3> bound_resource_owners_by_shape{};
    std::array<std::uint64_t,3> bound_external_bytes_by_shape{};
    std::array<Exl3ResourceInventory::Totals,3> retained_units_by_shape{};
    std::uint64_t quarantined_entries = 0;
    Exl3ResourceInventory::Totals reserved_peak_units{};
    bool driver_memory_bytes_known = false;
    std::uint64_t driver_memory_bytes = 0;
};

struct Exl3ExactStorageStats {
    std::uint64_t logical_context_tokens = 0, longest_context = 0;
    std::uint64_t unique_images = 0, unique_kv_pages = 0, materialized_kv_token_rows = 0;
    std::uint64_t logical_kv_bytes = 0, materialized_kv_bytes = 0, allocated_kv_bytes = 0;
    std::uint64_t unique_other_payload_bytes = 0;
};

struct Exl3ExactAllocationDomainStats {
    std::uint64_t kv = 0;
    std::uint64_t recurrent = 0;
    std::uint64_t convolution = 0;
    std::uint64_t state_taps = 0;
    std::uint64_t logits = 0;
    std::uint64_t embedding = 0;
};

struct Exl3HostKVStats {
    bool enabled = false;
    std::uint64_t layer_workspace_bytes = 0, h2d_bytes = 0, d2h_bytes = 0;
    // transfer_calls counts logical page-plane moves. copy_submissions counts
    // runtime API submissions, which may carry multiple independent moves.
    std::uint64_t transfer_calls = 0, copy_submissions = 0, completed_rows = 0;
    // Default-off authoritative snapshot export path. Logical K/V page-plane
    // copies are unchanged; only their per-copy completion waits are collapsed
    // into one fence for a bounded decode-sized batch.
    std::uint64_t exact_kv_export_batched_calls=0;
    std::uint64_t exact_kv_export_batched_ranges=0;
    std::uint64_t exact_kv_export_batched_bytes=0;
    std::uint64_t exact_kv_export_batched_fences=0;
    std::uint64_t exact_kv_export_saved_fences=0;
    std::uint64_t pinned_staging_bytes = 0;
    std::uint64_t shared_prefix_hit_bytes=0,shared_prefix_busy_fallbacks=0;
    std::uint64_t registered_upload_bytes=0,registered_upload_fallbacks=0;
    std::uint64_t registered_upload_failures=0;
    std::uint64_t registered_upload_peak_pending=0;
    std::uint64_t shared_page_copy_bytes=0,shared_page_fallbacks=0,shared_page_failures=0;
    std::uint64_t shared_page_attention_bytes=0;
    std::uint64_t query_pair_launch_attempts=0,query_pair_row_attempts=0;
    std::uint64_t query_pair_requested_row_attempts=0;
    std::uint64_t gqa_six_query_pair_score_launch_attempts=0;
    std::uint64_t gqa_six_query_pair_score_row_attempts=0;
    std::uint64_t gqa_six_score_k_tile64_launch_attempts=0;
    std::uint64_t gqa_six_score_k_tile64_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_value_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_value_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_row_attempts=0;
    std::uint64_t gqa_six_softmax_six_values_single_load_launch_attempts=0;
    std::uint64_t gqa_six_softmax_six_values_single_load_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_key_pair_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_key_pair_row_attempts=0;
    // Default-off stable HostKV transaction-checkpoint copy graph. Captures
    // and replays count complete checkpoint-copy batches; capture time is
    // one-time host wall and remains separate from request replay work.
    std::uint64_t transaction_checkpoint_graph_captures=0;
    std::uint64_t transaction_checkpoint_graph_replays=0;
    double transaction_checkpoint_graph_capture_ms=0.0;
    // Default-off exact HostKV verifier route. Counts GDN layer checkpoints
    // whose root image is stored in the existing state-before trace, allowing
    // the immediately following verifier forward to skip its duplicate copy.
    std::uint64_t transaction_recurrent_trace_alias_layers=0;
    std::uint64_t transaction_recurrent_trace_copy_bytes_saved=0;
    std::uint64_t attention_stage_upload_bytes=0,attention_stage_consumed_bytes=0;
    std::uint64_t attention_stage_direct_bytes=0;
    std::uint64_t attention_stage_banks=0;
    // Subsets of existing context persistent bytes, never extra inventory charges.
    std::uint64_t exact_attention_score_bytes=0,numeric_attention_scratch_bytes=0;
    std::uint64_t coalesced_attention_layers=0,coalesced_attention_bytes_saved=0;
    std::uint64_t shared_page_attention_groups=0,shared_page_attention_pages=0,shared_page_attention_max_pages=0;
    std::uint64_t pinned_slot_waits = 0, pinned_deferred_drains = 0;
    std::uint64_t pinned_scatter_bytes = 0, pinned_max_pending_bytes = 0;
    std::uint64_t pinned_plane_tail_deferrals = 0;
    std::uint64_t pinned_publication_drains = 0;
    // Default-off cold-prefill experiment: gather registered-page candidates
    // into the existing bounded pinned slots. Logical page-plane work remains
    // unchanged; these counters prove the selected route executed.
    std::uint64_t prefill_pinned_batch_plane_calls = 0;
    std::uint64_t prefill_pinned_batch_page_planes = 0;
    std::uint64_t prefill_pinned_batch_bytes = 0;
    // Default-off small-M route: one disjoint K/V slice per full-attention
    // bank, followed by a single ordered scatter at publication.
    std::uint64_t banked_d2h_forwards = 0, banked_d2h_planes = 0;
    std::uint64_t banked_d2h_rows = 0, banked_d2h_bytes = 0;
    std::uint64_t banked_d2h_drains = 0;
    // Opt-in T24 attribution. Times are host steady-clock nanoseconds and are
    // collected only when NINFER_EXL3_HOST_KV_CPU_PROFILE=1.
    std::uint64_t page_extension_calls = 0, page_prefix_refs = 0;
    std::uint64_t page_clone_bytes = 0, page_extension_cpu_ns = 0;
    std::uint64_t page_unique_tail_reuses = 0, page_unique_tail_reuse_bytes = 0;
    std::uint64_t page_prefix_ref_cpu_ns = 0, page_clone_cpu_ns = 0;
    std::uint64_t page_payload_prepare_cpu_ns = 0;
    std::uint64_t pinned_wait_cpu_ns = 0, pinned_scatter_cpu_ns = 0;
    std::uint64_t pinned_gather_cpu_ns = 0;
    std::uint64_t device_prefix_bytes = 0, device_prefix_fallbacks = 0;
    std::uint64_t device_prefix_hit_bytes = 0, device_prefix_fill_bytes = 0;
    std::uint64_t device_prefix_segmented_bytes = 0;
    // Default-off W12/W15 route: publish the exact just-computed HostKV rows
    // into a 16K private represented prefix. Subsequent forwards can consume
    // even the incomplete 64-row tail without another host upload.
    std::uint64_t device_prefix_forward_publish_forwards = 0;
    std::uint64_t device_prefix_partial_hit_bytes = 0;
    std::uint64_t device_prefix_forward_publish_bytes = 0;
    std::uint64_t pinned_pending_scan_cpu_ns = 0;
    std::uint64_t batch_metadata_bytes = 0,batch_metadata_reuses = 0;
    std::uint64_t batch_metadata_reuse_drains = 0;
};

// Immutable in-process L2 image produced only by ordinary full-FP16 attention.
// Shares the model's lifetime contract with contexts. Restoring into a different
// context of the same resident model explicitly forks this exact prefix.
// No OSCAR expansion/cast can construct an authoritative image.
class Exl3ExactHostState : public std::enable_shared_from_this<Exl3ExactHostState> {
    struct ConstructionKey {};
public:
    explicit Exl3ExactHostState(ConstructionKey) {}
    Exl3ExactHostState(const Exl3ExactHostState&)=delete;
    Exl3ExactHostState& operator=(const Exl3ExactHostState&)=delete;
    Exl3ExactHostState(Exl3ExactHostState&&)=delete;
    Exl3ExactHostState& operator=(Exl3ExactHostState&&)=delete;
    int position() const noexcept { return position_; }
    int rope_offset() const noexcept {return rope_offset_;}
    // Opaque identity of this resident numerical reference, not a file hash.
    std::shared_ptr<const void> model_identity() const noexcept {return model_identity_;}
    std::size_t payload_bytes() const noexcept;
    // Per-image metadata, excluding page objects and shared-pointer control blocks.
    // Slab record may also be retained by the export pool; not globally additive.
    std::uint64_t owner_metadata_bytes() const;
    // Snapshot block/control storage and page-table allocation only. Shared
    // page objects, payload planes and recurrent slab are separate owners.
    std::uint64_t snapshot_metadata_bytes() const;
    static bool attach_snapshot_metadata_credit(const std::shared_ptr<const void>&,
        RetainedDescriptorLedger::Ticket) noexcept;
    static bool snapshot_metadata_credit_belongs_to(const std::shared_ptr<const void>&,
        const RetainedDescriptorLedger&) noexcept;
    // Registration owned by this image's recurrent slab; excludes external KV
    // registration leases and may overlap the export pool's same slab owner.
    std::size_t recurrent_registered_bytes() const noexcept;
    bool same_payload(const Exl3ExactHostState& other) const;
    // Diagnostic only: compares represented values without model identity.
    bool same_represented_payload_for_test(const Exl3ExactHostState& other) const;
    // Stable within one binary/platform and covers every field compared by
    // same_represented_payload_for_test. Qualification scripts use it to join
    // exact state produced by separate model processes; it is not an identity
    // or persistence format.
    std::uint64_t represented_payload_hash_for_test() const noexcept;
    // Diagnostic only: names the first unequal represented field and element.
    std::string represented_first_difference_for_test(
        const Exl3ExactHostState& other) const;
    // Diagnostic-only value copy for comparisons across Engine lifetimes.
    // No model, registration, slab or source-page owner survives in this copy.
    std::shared_ptr<const Exl3ExactHostState> detached_payload_for_test() const;
    // Diagnostic semantic view; the callback cannot mutate retained pages.
    void visit_kv_for_test(const std::function<void(int layer, int first, int rows,
        std::span<const std::uint16_t> k, std::span<const std::uint16_t> v)>& visitor) const;
    // Restore admission contract: an exact image must describe a contiguous,
    // non-truncated native history before any physical context is mutated.
    bool native_extent_valid(int maximum_position) const noexcept;
    bool is_prefix_of(const Exl3ExactHostState& other) const;
    // Immutable full-page owner at this exact frontier. A partial tail cannot
    // authorize a 64-row shared destination, even if its allocation is larger.
    std::shared_ptr<const Exl3ExactKVPage> shared_kv_page(int first) const noexcept;
    template<class Visitor> void visit_page_metadata_owners(Visitor&& visitor) const {
        for(const auto& page:kv_pages_)visitor(page);
    }
    std::size_t kv_export_bytes() const noexcept {return kv_export_bytes_;}
    static Exl3ExactStorageStats storage_stats(
        std::span<const std::shared_ptr<const Exl3ExactHostState>> requests);
    static void exercise_recurrent_storage_accounting_for_test();
    static Exl3ExactAllocationDomainStats allocation_domain_stats(
        std::span<const std::shared_ptr<const Exl3ExactHostState>> requests,
        bool include_unused_capacity=false);
    // Read-only tensor allocation inventory, deduplicated by proven ownership.
    // Caller retains the images and serializes their lifetime during visitation.
    // Capacity may include untouched tail reserve. Select false for actual payload.
    // Neither inventory claims physical residency; metadata is separate.
    static void visit_host_allocations(
        std::span<const std::shared_ptr<const Exl3ExactHostState>> requests,
        const std::function<void(const void*,std::size_t)>& visitor,bool include_unused_capacity=true);
private:
    static std::shared_ptr<Exl3ExactHostState> create();
    std::shared_ptr<const int> model_identity_;
    int position_ = 0, device_position_ = 0;
    int tap_rows_ = 0, embedding_rows_ = 0, last_rows_ = 0;
    int rope_offset_ = 0;
    std::vector<std::shared_ptr<const Exl3ExactKVPage>> kv_pages_;
    std::size_t kv_export_bytes_ = 0;
    std::uint64_t residency_id_ = 0;
    std::array<std::vector<float>, 48> recurrent_;
    std::shared_ptr<const Exl3RecurrentSlab> recurrent_slab_;
    std::span<const float> recurrent_plane(std::size_t i) const noexcept {
        return recurrent_slab_?recurrent_slab_->plane(i):std::span<const float>(recurrent_[i]);
    }
    std::array<std::vector<std::uint16_t>, 48> convolution_;
    std::vector<std::uint16_t> logits_, embedding_;
    std::array<std::vector<std::uint16_t>, 5> taps_;
    // L0 OSCAR INT2 history parked with the state: per full-attention layer,
    // the codes ([rows][4][64] K then V) and FP16 meta ([rows][4][4] K then V)
    // of the encoded rows [0, l0_rows_), so a resume skips re-encoding.
    int l0_rows_ = 0;
    bool l2_fp8_ = false;   // kv_pages_ hold FP8 rows (l0_l2_fp8.cuh)
    std::array<std::vector<std::uint8_t>, 16> l0_codes_;
    std::array<std::vector<std::uint16_t>, 16> l0_meta_;
    friend class Exl3TextContext;
};

class Exl3TurboAngleWarmPages {
public:
    std::size_t payload_bytes() const noexcept;
    int first_position() const noexcept { return first_; }
    int rows() const noexcept { return rows_; }
    bool full_history() const noexcept {
        return source_ && first_ == 0 && rows_ == source_->position();
    }
private:
    Exl3TurboAngleWarmPages() = default;
    std::shared_ptr<const Exl3ExactHostState> source_;
    int first_ = 0, rows_ = 0;
    std::array<std::vector<std::uint8_t>,16> k_, v_;
    friend class Exl3TextContext;
};

struct Exl3LoadOptions {
    bool disable_dual_artifact_loading = false;
    std::filesystem::path verified_dual_manifest;
};

struct Exl3ModelLoadStats {
    bool parallel_staged = false;
    std::array<std::uint64_t, 2> source_bytes{};
    std::uint64_t audited_tensors = 0;
    std::uint64_t audited_bytes = 0;
    double staging_prepare_ms = 0;
    double staging_release_ms = 0;
    double payload_read_ms = 0;
    double upload_ms = 0;
    double staged_transfer_ms = 0;
    std::uint64_t staging_host_bytes = 0;
    // Successful canonical linear loads consuming four retained descriptors.
    // Counts removed duplicate header searches, not bytes saved or elapsed time.
    std::uint64_t linear_descriptor_reuses = 0;
};

struct Exl3TextDecodeAttribution {
    double total_microseconds = 0.0;
    double token_id_h2d_microseconds = 0.0;
    double embedding_microseconds = 0.0;
    double layer_stack_microseconds = 0.0;
    double final_norm_microseconds = 0.0;
    double lm_head_microseconds = 0.0;
    double other_microseconds = 0.0;
    std::array<double, 64> layer_microseconds{};
};

struct Exl3RequestResetStats {
    std::uint64_t generation = 0;
    double prior_work_sync_ms = 0.0;
    double reset_complete_ms = 0.0;
    std::size_t persistent_bytes = 0;
    bool exact_payload_preserved = false;
};

// Test/qualification-only export of the exact tensors at the OSCAR calibration boundary.
// The vectors are raw F16 bit units in contiguous [tokens, heads, head_dim] order.  This is
// deliberately a host-copy API; it is not part of the production decode path.
struct Exl3FullAttentionQKVHost {
    int layer = -1;
    int rows = 0;
    std::vector<std::uint16_t> q_rope;
    std::vector<std::uint16_t> k_rope;
    std::vector<std::uint16_t> v_projection;
};

// Borrowed existing traces, valid only during the eager callback. No copying or
// allocation is performed by the model. Graph capture is forbidden with a hook.
struct Exl3LayerObservation {
    int layer = -1, position = 0, rows = 0;
    const std::uint16_t* output = nullptr;
    Exl3FullAttentionLayerTrace attention{};
    Exl3GdnLayerTrace gdn{};
    const std::uint16_t* k_cache = nullptr;
    const std::uint16_t* v_cache = nullptr;
    cudaStream_t stream = nullptr;
};
using Exl3LayerObserver = void (*)(const Exl3LayerObservation&, void*);

class Exl3TextModel {
public:
    static std::unique_ptr<Exl3TextModel> load(const std::filesystem::path& model_directory,
                                               int max_context = 256,
                                               const Exl3LoadOptions& options = {});
    ~Exl3TextModel();

    Exl3TextModel(const Exl3TextModel&) = delete;
    Exl3TextModel& operator=(const Exl3TextModel&) = delete;

    // Engine shared-page mode suppresses redundant private prefix allocation;
    // authoritative HostKV and its pinned staging remain available as fallback.
    std::unique_ptr<Exl3TextContext> create_context(bool capture_taps = true,bool allocate_device_prefix = true,bool defer_reconstruction = false) const;
    // Reserved startup path excludes optional private prefix and defers exact
    // reconstruction to its existing separate coordinator transaction.
    std::shared_ptr<Exl3TextContext> create_context_reserved(
        Exl3VeriCacheServingCoordinator&,bool capture_taps=true,
        unsigned startup_fault_for_test=0,bool enable_qualified_media=false,
        bool allow_ordinary_graphs=true) const;
    int layer_count() const noexcept { return 64; }
    int full_attention_layer_count() const noexcept { return 16; }
    int gdn_layer_count() const noexcept { return 48; }
    int max_context() const noexcept { return max_context_; }
    std::size_t model_bytes() const noexcept { return model_bytes_; }
    // Device-allocation records and retained pointer-vector capacity only.
    // Excludes manifests, allocator overhead and opaque driver storage.
    std::uint64_t allocation_owner_metadata_bytes() const;
    const Exl3ModelLoadStats& load_stats() const noexcept;
    // Aliases immutable decoded headers while retaining the uploaded model owner.
    std::shared_ptr<const void> metadata_owner() const noexcept;

private:
    struct ContextConstruction {
        std::unique_ptr<Exl3TextContext> exclusive;
        std::shared_ptr<Exl3TextContext> shared;
    };
    ContextConstruction create_context_impl(bool capture_taps,bool allocate_device_prefix,
        bool defer_reconstruction,Exl3VeriCacheServingCoordinator* authority,unsigned startup_fault=0,
        const std::function<void(const Exl3ResourceInventory::Requirement&)>& extend={},
        Exl3ResourceInventory* actual_result=nullptr,bool enable_qualified_media=false,
        bool allow_ordinary_graphs=true) const;
    struct Impl;
    explicit Exl3TextModel(std::unique_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_; // immutable uploaded backing also retained by contexts
    int max_context_ = 0;
    std::size_t model_bytes_ = 0;

    friend class Exl3TextContext;
};

class Exl3TextContext {
public:
    ~Exl3TextContext();
    static void reserve_reconstruction_lanes(Exl3VeriCacheServingCoordinator&,
        std::span<Exl3TextContext*> contexts,bool allow_budget_fallback = false);
    static std::size_t reconstruction_quarantined_contexts() noexcept;
    static std::size_t host_kv_quarantined_contexts() noexcept;
    static std::uint64_t generic_quarantined_allocations() noexcept;
    // Physical fixed control blocks, including weak-only survivors; nonadditive
    // with context/reconstruction metadata inventories.
    static std::uint64_t live_shared_control_metadata_bytes() noexcept;
    // Change witness only: HostKV contexts, reconstruction contexts/backings,
    // prefix bytes, generic allocations. Mixed units must never be summed.
    static std::array<std::uint64_t,8> retirement_quarantine_witness() noexcept;
    struct GenericRetirementSnapshot {
        std::uintptr_t pointer=0;std::uint64_t bytes=0;int device=-1,error=0;
        std::size_t metadata_bytes=0;
        std::uint64_t metadata_credit_bytes=0,device_credit_bytes=0;
    };
    static GenericRetirementSnapshot generic_retirement_snapshot_for_test() noexcept;
    static void exercise_generic_upload_for_test(std::span<const std::byte> source,unsigned fault=0);
    static void exercise_generic_lifetime_credits_for_test();
    static void exercise_exact_page_extension_for_test();
    static void exercise_generic_constructor_credits_for_test(unsigned fault);
    static void exercise_shared_constructor_credits_for_test(unsigned fault);
    void fail_host_kv_retirement_for_test(bool device_mismatch=false);
    void fail_host_kv_device_query_for_test();
    void fail_host_kv_compute_retirement_for_test();
    // Bind before execution. The owner must keep this nondefault stream alive;
    // default-stream calls remain permitted. Binding cannot be replaced.
    void retain_execution_stream_owner(cudaStream_t stream,std::shared_ptr<const void> owner);
    void fail_next_host_kv_copy_for_test();
    void fail_next_host_kv_restore_for_test();
    // Invoked after each completely restored target layer during the next
    // exact restore. Test observers may request cancellation but must not throw.
    void observe_next_exact_restore_layer_for_test(std::function<void(int)> observer);
    void fail_host_kv_prefill_boundary_for_test(unsigned stage);
    std::weak_ptr<const void> host_kv_workspace_owner_for_test() const;
    static std::array<std::uint64_t,3> shared_kv_credits_for_test(const std::shared_ptr<const void>& owner) noexcept;
    // Only a joined borrower returned by the HostKV owner diagnostic is valid.
    static bool fail_shared_kv_retirement_for_test(const std::shared_ptr<const void>& owner,unsigned fault) noexcept;
    std::weak_ptr<const Exl3ExactKVPage> host_kv_pending_page_owner_for_test() const noexcept;
    static std::weak_ptr<const void> quarantined_host_kv_workspace_for_test() noexcept;
    struct HostKVRetirementSnapshot {
        bool retained=false;
        int drain_error=0;
        std::size_t device_bytes=0,pinned_bytes=0;
        std::size_t inline_metadata_bytes=0;
        std::size_t pending_slots=0,scatter_pieces=0;
        std::size_t failed_final_use_slots=0;
        // Exact descriptor arrays; excludes allocation records and page metadata.
        std::size_t transfer_descriptor_metadata_bytes=0;
        std::uint64_t allocation_owner_metadata_bytes=0;
        // Deduplicated within this context; excludes page plane payloads.
        std::uint64_t page_owner_metadata_bytes=0;
        std::uint64_t page_payload_capacity_bytes=0;
        // Restore-image payload overlaps page payload above; do not add them.
        std::uint64_t restore_source_payload_capacity_bytes=0;
        std::size_t restore_source_object_bytes=0;
        std::uint64_t restore_source_owner_metadata_bytes=0;
        std::size_t restore_source_registered_bytes=0;
        bool forward_stream_retained=false;
        bool copy_stream_retained=false;
        std::uintptr_t forward_stream_identity=0;
        int forward_device=-1;
        bool batch_descriptors_pending=false;
        // Destination/source/byte-count tables, observed without retaining owners.
        std::array<std::uintptr_t,3> batch_table_addresses{};
        std::array<std::size_t,3> batch_table_sizes{},batch_table_capacities{};
    };
    static HostKVRetirementSnapshot host_kv_retirement_snapshot();
    static std::size_t reconstruction_quarantined_allocations() noexcept;
    static std::size_t reconstruction_backing_metadata_bytes() noexcept;
    bool reconstruction_retirement_uncertain() const noexcept;
    void fail_reconstruction_retirement_for_test();
    void fail_reconstruction_allocation_for_test(bool fail_prior_cleanup = false);
    void fail_reconstruction_cleanup_for_test();

    Exl3TextContext(const Exl3TextContext&) = delete;
    Exl3TextContext& operator=(const Exl3TextContext&) = delete;

    void reset(cudaStream_t stream = nullptr);
    // Engine-owned eager execution boundary; installation requires serialized
    // access to this context. The executor must retain/drain all queued users.
    void set_target_q_executor(Exl3TargetQExecutor executor,bool kv=false,bool output=false,bool gateup=false,bool down=false,bool head=false);
    using RegisteredKVUploader=std::function<std::uint64_t(const Exl3ExactKVExtent&,void*,int,cudaStream_t,cudaEvent_t,int)>;
    using RegisteredKVCompletion=std::function<void(int,std::uint64_t)>;
    // Startup installation. Zero means staged fallback; a generation retains
    // owners until completion(slot,generation), before that event is reused.
    void set_registered_kv_uploader(RegisteredKVUploader uploader,RegisteredKVCompletion completion);
    using SharedPageCopier=std::function<bool(const Exl3ExactKVExtent&,void*,int,cudaStream_t,cudaEvent_t)>;
    void set_shared_page_copier(SharedPageCopier copier);
    using SharedPageAttention=std::function<Exl3SharedAttentionPrefix(std::shared_ptr<const Exl3ExactKVPage>,int,int,cudaStream_t,cudaEvent_t)>;
    enum class SharedPageAttentionPhase { before_compute,final_use };
    using SharedPageAttentionCompletion=std::function<void(std::span<const Exl3SharedAttentionPrefix>,cudaStream_t,SharedPageAttentionPhase)>;
    void set_shared_page_attention(SharedPageAttention acquire,SharedPageAttentionCompletion complete);
    void set_attention_staging(std::shared_ptr<Exl3AttentionStageStorage> storage,
        std::shared_ptr<Exl3AttentionStage> stages,std::shared_ptr<Exl3AttentionStageHistory> history);
    bool attention_staging_uncertain() const noexcept;
    void set_attention_registration_cache(std::shared_ptr<Exl3KVRegistrationCache> cache);
    void set_attention_staging_enabled_for_test(bool enabled);
    void fail_attention_staging_for_test(unsigned stage);
    // Startup-only sharing of an already allocated redundant cache. Both
    // contexts keep private mutable tails and authoritative HostKV roots.
    // Legacy private owners require an explicit shared inventory destination.
    // Device and metadata records are prepared before either context changes.
    std::shared_ptr<const void> share_device_prefix_with(Exl3TextContext& other,
        Exl3ResourceInventory& shared_resources);
    bool prepare_device_prefix_reserved(Exl3VeriCacheServingCoordinator&,int capacity,bool segmented=false,
        unsigned startup_fault_for_test=0);
    std::shared_ptr<const void> share_device_prefix_with_empty(Exl3TextContext& other);
    void retire_device_prefix_after_drain() noexcept;
    std::weak_ptr<const void> device_prefix_owner_for_test() const noexcept;
    void fail_device_prefix_cleanup_for_test();
    // Explicit serving-lane contract. Bind once while the context is pristine;
    // subsequent completed-request reuse must supply the same full constructor
    // and serving compatibility identity. This synchronous boundary drains all
    // prior device/copy work before clearing request-owned state. Any drain or
    // reset failure throws before the context can be admitted again; callers
    // must discard it and construct a fresh compatible context.
    void bind_request_compatibility(std::string contract);
    Exl3RequestResetStats reset_for_request(std::string_view contract);
    Exl3RequestResetStats reset_for_request_preserving(std::string_view contract,
        const Exl3ExactHostState& root,std::uint64_t expected_generation);
    std::uint64_t request_generation() const noexcept;
    // One-way test boundary; never restores an exhausted generation to service.
    void exhaust_request_generation_for_test();
    void fail_next_request_reset_completion_for_test();
    // Explicit research seam; rows1..8, FP32 features [rows,5120], positions
    // row-major [rows,3]. Full V6 numerical/media serving remains unqualified.
    // Invalid input is rejected before mutation; runtime failure requires restore/reset.
    void append_media_embeddings_numeric(std::span<const float> embeddings,
        std::span<const std::int32_t> positions_xyz,cudaStream_t stream=nullptr);
    int rope_offset() const noexcept;
    GoptSubmissions gaming_submissions() const noexcept;
    Exl3HostKVStats host_kv_stats() const noexcept;
    struct DeviceTransactionCheckpointGraphStats {
        std::uint64_t captures=0,replays=0;
        double capture_ms=0.0;
    };
    DeviceTransactionCheckpointGraphStats
        device_transaction_checkpoint_graph_stats() const noexcept;
    struct HostKVGdnSegmentGraphStats {
        std::uint64_t captures=0,replays=0;
        double capture_ms=0.0;
        std::uint64_t launch_cpu_ns=0;
    };
    HostKVGdnSegmentGraphStats host_kv_gdn_segment_graph_stats() const noexcept;
    std::uint64_t fast_same_weights_fp16kv_gdn_decode_conv_calls() const noexcept;
    std::uint64_t fast_same_weights_fp16kv_gdn_m1_gate_up_pair_submissions() const noexcept;
    struct HostKVFullLayerGraphStats {
        std::uint64_t captures=0,replays=0;
        std::uint64_t six_softmax_triple_captures=0;
        std::uint64_t k6_stream_reduction_captures=0;
        std::uint64_t extended_stream_reduction_captures=0;
        std::uint64_t target_down_k6_async_a_captures=0;
        std::uint64_t target_k6_small_m_async_a_captures=0;
        std::uint64_t target_k7_small_m_async_a_captures=0;
        double capture_ms=0.0;
        std::uint64_t launch_cpu_ns=0;
    };
    HostKVFullLayerGraphStats host_kv_full_layer_graph_stats() const noexcept;
    // Separately labeled default-off ordinary device-KV full-attention-layer
    // graph candidate.  This is not part of the HostKV/reference statistics.
    struct OrdinaryFullLayerGraphStats {
        std::uint64_t captures=0,replays=0;
        double capture_ms=0.0;
    };
    OrdinaryFullLayerGraphStats ordinary_full_layer_graph_stats() const noexcept;
    // Process-wide ordinary device-KV graph capture/replay counts for linked
    // qualification of Engine-owned contexts.
    struct OrdinaryGraphProcessStats {
        std::uint64_t gdn_segment_captures=0,gdn_segment_replays=0;
        std::uint64_t full_layer_captures=0,full_layer_replays=0;
        std::uint64_t mlp_tail_captures=0,mlp_tail_replays=0;
    };
    static OrdinaryGraphProcessStats ordinary_graph_process_stats_for_test() noexcept;
    struct HostKVMlpTailGraphStats {
        std::uint64_t captures=0,replays=0;
        double capture_ms=0.0;
    };
    HostKVMlpTailGraphStats host_kv_mlp_tail_graph_stats() const noexcept;
    Exl3RecurrentExportStats recurrent_export_stats() const;
    void set_recurrent_growth_admission(Exl3RecurrentExportPool::GrowthAdmission admission);
    using SnapshotMetadataReservation=std::function<RetainedDescriptorLedger::Ticket(std::uint64_t)>;
    void set_snapshot_metadata_reservation(SnapshotMetadataReservation reservation);
    void set_request_metadata_reservation(SnapshotMetadataReservation reservation);
    const SnapshotMetadataReservation& request_metadata_reservation() const noexcept;
    using RequestHostPayloadReservation=std::function<RetainedHostAllocationLedger::Ticket(std::uint64_t)>;
    void set_request_host_payload_reservation(RequestHostPayloadReservation reservation);
    const RequestHostPayloadReservation& request_host_payload_reservation() const noexcept;
    using RequestHostPayloadObserver=std::function<void(const std::shared_ptr<const void>&,std::size_t,const void*,std::size_t)>;
    void set_request_host_payload_observer(RequestHostPayloadObserver observer);
    const RequestHostPayloadObserver& request_host_payload_observer() const noexcept;
    void set_recurrent_borrow_admission(Exl3RecurrentExportPool::BorrowAdmission admission);
    void fail_recurrent_constructor_for_test(unsigned fault);
    void fail_export_copy_for_test(unsigned submission);
    Exl3ReconstructGemmStats numeric_prefill_projection_stats() const noexcept;
    std::uint64_t fast_wide_prefill_gemm_calls() const noexcept;
    std::uint64_t fast_wide_prefill_gemm_rows() const noexcept;
    std::uint64_t exact_attention_gqa_six_softmax_fused_scalar_values_calls() const noexcept;
    std::uint64_t exact_attention_gqa_six_softmax_fused_scalar_values_rows() const noexcept;
    std::uint64_t exact_attention_gqa_six_decode_fused_calls() const noexcept;
    // Strictly opt-in same-weight FP16-KV prefill telemetry.  The candidate
    // owns no alternate numerical authority: wide forwards use the existing
    // exact/reference layer path whenever their shape is not admitted by the
    // candidate workspace.
    struct FastSameWeightsFp16KvPrefillStats {
        bool enabled = false;
        std::uint64_t wide_prefill_forwards = 0;
        std::uint64_t wide_prefill_rows = 0;
        std::uint64_t numeric_dispatch_calls = 0;
        std::uint64_t numeric_dispatch_rows = 0;
    };
    FastSameWeightsFp16KvPrefillStats
    fast_same_weights_fp16kv_prefill_stats() const noexcept;
    Exl3ReconstructedExactStats reconstructed_exact_stats() const noexcept;
    std::uint64_t paired_transform_submissions() const noexcept;
    struct HeadWorkStats {
        // Eager host submission chains, not GPU completion or graph replay.
        std::uint64_t submitted_rows=0;
        std::uint64_t omitted_rows=0;
    };
    HeadWorkStats head_work_stats() const noexcept;
    std::uint64_t fused_gate_up_submissions() const noexcept;
    std::uint64_t fused_residual_norm_submissions() const noexcept;
    bool oscar_only_context() const noexcept;
    std::size_t cache_staging_bytes() const noexcept;
    int max_context() const noexcept;
    std::shared_ptr<const void> model_identity() const noexcept;
    std::shared_ptr<const void> metadata_owner() const noexcept;
    // Blocking L2 transfer. Ordinary eager path only; captured taps required.
    // All history KV stays FP16, recurrent state FP32 and conv BF16. Exported
    // host bytes are live DRAM, with no disk/pagefile backing managed by engine.
    // share_prefix=false independently downloads actual GPU KV for auditing.
    std::shared_ptr<const Exl3ExactHostState> export_exact_host_state(
        cudaStream_t stream = nullptr, bool share_prefix = true,
        bool fail_after_recurrent_plan_for_test = false) const;
    // Prevalidates identity before mutation. Failed HostKV lineage requires
    // retirement; neither reset nor restore grants recovery authority.
    // Drops old suffix visibility and restores logits/taps for exact continuation.
    void restore_exact_host_state(const Exl3ExactHostState& state,
                                  cudaStream_t stream = nullptr);
    // Opaque physical-state witness. False after any forward/reset and for an
    // equal-payload but different immutable image; true only after this context
    // successfully restored or exported this exact state identity.
    bool exact_host_state_resident(const Exl3ExactHostState& state) const noexcept;
    // Returns true only when a restore was submitted. A resident hit still
    // passes retirement/failed-lineage gates; callers own acquisition identity.
    bool restore_exact_host_state_if_needed(const Exl3ExactHostState& state,cudaStream_t stream=0);
    // Bounded warm donor: at most64 recent rows, generated from this exact L2 image.
    static std::shared_ptr<const Exl3TurboAngleWarmPages> make_turboangle_warm_pages(
        std::shared_ptr<const Exl3ExactHostState> source, int rows);
    // Independent L1 root: every admitted attention row is encoded in immutable
    // TurboAngle DRAM pages. Recurrent and convolution state remain in the trusted
    // L2 root and are copied into the private L1 execution context on restore.
    static std::shared_ptr<const Exl3TurboAngleWarmPages> make_turboangle_l1_pages(
        std::shared_ptr<const Exl3ExactHostState> source);
    static std::shared_ptr<const Exl3TurboAngleWarmPages> extend_turboangle_l1_pages(
        std::shared_ptr<const Exl3ExactHostState> source,
        std::shared_ptr<const Exl3TurboAngleWarmPages> prefix);
    // Rebuild original OSCAR L0 directly from authoritative L2 KV/checkpoint.
    // Optional L1 recent rows replace only tentative L0 input, never L2 bytes.
    void restore_oscar_host_state(const Exl3ExactHostState& state,
        const Exl3TurboAngleWarmPages* warm = nullptr, cudaStream_t stream = nullptr);
    // Replace only a newly authoritative 1..8-row delta after caller rollback
    // to old_root. Unchanged OSCAR prefix storage stays live; GDN/convolution
    // and observable outputs are replaced from new_root.
    void rebase_oscar_host_state_delta(const Exl3ExactHostState& old_root,
        const Exl3ExactHostState& new_root,const Exl3TurboAngleWarmPages& pages,
        cudaStream_t stream = nullptr);
    // The default/reference entry remains limited to the established initial
    // 16-row path.  The same-weight candidate is admitted only through the
    // existing explicit wide-prefill API; this entry is not widened implicitly.
    void prefill(std::span<const std::int64_t> token_ids,
                 cudaStream_t stream = nullptr);
    void decode(std::int64_t token_id, cudaStream_t stream = nullptr);
    // Eager canonical-OSCAR ingestion after a nonempty prefix, 1..8 teacher rows.
    // Preserves continuation/M1 layer topology and publishes only final-row
    // logits. No continuation scratch or tap capture is required. With taps
    // enabled all input rows are captured for ordered draft-ring ingestion.
    // Rejects active transactions/graphs; like prefill/decode, runtime failure
    // after enqueue is non-atomic and requires the caller to reset the context.
    void append_prefill(std::span<const std::int64_t> token_ids,
                        cudaStream_t stream = nullptr);
    // Separate opt-in ingestion preserving M1 layer mathematics. Construction
    // latches capacity16,32 with staged reduction,64 with PREFILL_WIDE64=1,
    // or128 with PREFILL_WIDE128=1 and shared accumulation;256 additionally
    // requires PREFILL_WIDE256=1 and shared input transforms.
    //512/1024 additionally require their exact flags and shared layer scratch.
    // Requires a context constructed with NINFER_EXL3_WIDE_PREFILL=1.
    void append_prefill_wide(std::span<const std::int64_t> token_ids,
                             cudaStream_t stream = nullptr);
    // Experimental ordinary device-KV suffix schedule: process each layer's
    // causal 1024-row chunks while its reconstructed projections are resident.
    // The entire suffix is freshly computed; only final-chunk taps/logits are
    // published, as with sequential append_prefill_wide calls.
    // Optional borrowed device tail receives five post-layer FP16 planes in
    // tap order {5,19,33,47,61}. Its exact absolute interval ends at the
    // completed prefix. The caller owns the arena through this synchronous
    // stream completion and must drain the stream before releasing it on error.
    struct RetainedTapTail {
        std::uint16_t* device = nullptr;
        std::size_t bytes = 0;
        int first_abs = 0;
        int rows = 0;
    };
    void append_prefill_layer_major(std::span<const std::int64_t> token_ids,
                                    cudaStream_t stream = nullptr,
                                    const RetainedTapTail* retained_taps = nullptr);
    // Fresh-prompt layer-major policy (NINFER_EXL3_LAYER_MAJOR_FROM_ZERO, default
    // 1, process latched; 0 = initial16 control): a reset context admits
    // layer-major prefill at position 0, so the
    // whole prompt runs as 1024-row layer-major chunks instead of an ordered
    // initial16 forward followed by the layer-major suffix. Numerical policy:
    // the first 16 rows use the chunk projection route.
    static bool layer_major_from_zero();
    // Rows a fresh layer-major prompt executes before the layer-major suffix.
    static int layer_major_initial_rows() { return layer_major_from_zero() ? 0 : 16; }

    // Qualification/reference path for a bounded eager target continuation.
    // Setup allocates fixed scratch once. Execution and host getters are ordered
    // only by their supplied stream; callers using multiple streams establish
    // dependencies externally. Device output is borrowed until reset, rollback,
    // or another forward supersedes it. A failure after work is enqueued requires
    // caller-owned transaction rollback; the ordinary call is not atomic. Layer
    // projections use qualified small-M kernels where admitted and otherwise
    // preserve per-row M1 arithmetic. Capacity is 2..8, or explicit ordinary
    // exact-host16; M16 shares packed loads and retains every row's outputs.
    void prepare_continuation(int capacity);
    void prepare_continuation_reserved(Exl3VeriCacheServingCoordinator&,int capacity,unsigned startup_fault_for_test=0);
    std::size_t continuation_bytes_required(int capacity) const;
    // Fixed represented owners only; workspace internals and graph growth are separate.
    static std::size_t continuation_owner_metadata_bytes() noexcept;
    static std::size_t continuation_owner_blocks_for_test() noexcept;
    static void exercise_continuation_owner_metadata_for_test();
    static void exercise_graph_retirement_poison_for_test();
    static void exercise_context_graph_retirement_for_test(unsigned failure);
    static void exercise_graph_invalidation_for_test();
    static std::size_t fixed_owner_metadata_bytes() noexcept;
    static std::size_t allocation_record_metadata_bytes() noexcept;
    void continue_rows(std::span<const std::int64_t> token_ids,
                       cudaStream_t stream = nullptr);
    // Ordinary FP16 continuation: keep its final consumed row as the canonical
    // single-row host checkpoint payload after callers inspect all-row outputs.
    // Invalidates borrowed continuation logits; no state arithmetic or replay.
    void finish_exact_continuation(cudaStream_t stream = nullptr);
    // Explicit authoritative wide prompt ingestion, host-KV eager lane only.
    // Uses existing qualified wide projections and bounded query score tiles.
    void append_exact_prefill_wide(std::span<const std::int64_t> token_ids,
                                  cudaStream_t stream = nullptr);
    // Explicit unqualified actual-row tail route; not selected by the normal
    // request partition menu. Preserves real row extents without padding tokens.
    void append_exact_prefill_tail(std::span<const std::int64_t> token_ids,
                                  cudaStream_t stream = nullptr);
    // Test seam: poison only unused hidden rows up to the next32-row boundary.
    void poison_prefill_tail_hidden_for_test(int rows,int byte,cudaStream_t stream = nullptr);
    void finish_exact_prefill(cudaStream_t stream = nullptr);
    int continuation_capacity() const noexcept;
    int continuation_rows() const noexcept;
    std::size_t continuation_bytes() const noexcept;
    std::string continuation_head_dispatch() const;
    const std::uint16_t* continuation_logits_device() const noexcept;
    // Default-off greedy readback only. Full logits and immutable root payloads
    // remain available. Caller exclusively owns this context and the stream.
    void prepare_greedy_packet(); // prepare before request/resource admission
    static std::size_t greedy_packet_bytes_required() noexcept;
    static std::size_t greedy_transfer_pool_registered_bytes_required() noexcept;
    static std::size_t greedy_transfer_pool_metadata_bytes_required() noexcept;
    static constexpr std::size_t greedy_transfer_pool_capacity() noexcept {return 2;}
    std::size_t greedy_transfer_pool_available_for_test() const noexcept;
    static std::uintptr_t greedy_transfer_identity_for_test(
        const Exl3PendingGreedyPacket&) noexcept;
    Exl3GreedyPacket greedy_packet(bool continuation, cudaStream_t stream = nullptr);
    Exl3PendingGreedyPacket submit_greedy_packet(
        bool continuation,std::uint64_t acquisition,std::uint64_t execution,
        cudaStream_t stream = nullptr,bool defer_host_readback=false);
    Exl3DeviceGreedySeed device_greedy_seed(
        const Exl3PendingGreedyPacket& pending,std::uint64_t acquisition,
        std::uint64_t execution) const;
    Exl3GreedyPacket finish_greedy_packet(
        Exl3PendingGreedyPacket&& pending,std::uint64_t acquisition,
        std::uint64_t execution);
    Exl3GreedyBatchSource greedy_batch_source(
        const Exl3PendingGreedyPacket& pending,std::uint64_t acquisition,
        std::uint64_t execution) const;
    Exl3GreedyPacket finish_batched_greedy_packet(
        Exl3PendingGreedyPacket&& pending,std::span<const Exl3GreedyRow> rows,
        std::uint64_t acquisition,std::uint64_t execution);
    static void fail_pending_greedy_packet_completion_for_test(
        Exl3PendingGreedyPacket& pending);
    static std::uint64_t quarantined_greedy_packet_transfers() noexcept;
    // Qualification-only actual-kernel seam for selector/tie/finite fixtures.
    // Input is row-major represented FP16 with stride >= logical vocabulary.
    Exl3GreedyPacket greedy_packet_from_scores_for_test(
        std::span<const std::uint16_t> scores,int rows,int vocabulary,int stride,
        cudaStream_t stream = nullptr);
    struct GreedyReadbackStats {
        std::uint64_t full_score_bytes=0,packet_bytes=0,packet_rows=0,packet_calls=0;
    };
    GreedyReadbackStats greedy_readback_stats() const noexcept;
    std::vector<std::uint16_t> continuation_logits_bits_host(
        cudaStream_t stream = nullptr) const;
    std::vector<float> continuation_logits_host(cudaStream_t stream = nullptr) const;
    // Experimental fixed-B8 transaction compute graph. Capture/replay covers
    // only layer traversal, taps, final norm and H6 head. Admission, uploads,
    // exact host decision, repair and publication remain eager caller work.
    bool capture_continuation_graph(cudaStream_t stream = nullptr);
    // Strict experimental companion for B4/B6 variants. Capacity remains B8;
    // each independently owned executable captures one exact row extent.
    bool capture_continuation_graph_rows(int rows,
                                         cudaStream_t stream = nullptr);
    void continue_rows_graph(std::span<const std::int64_t> token_ids,
                             cudaStream_t stream = nullptr);
    bool continuation_graph_active() const noexcept;
    Exl3ContinuationGraphAdmission continuation_graph_admission() const noexcept;
    Exl3ContinuationGraphStats continuation_graph_stats() const noexcept;
    std::string continuation_graph_status() const;
    std::string continuation_graph_rows_status(int rows) const;
    void discard_continuation_graph_for_test();
    void invalidate_projection_graph_binding_for_test();
    bool capture_decode_graph(cudaStream_t stream = nullptr);
    void decode_graph(std::int64_t token_id, cudaStream_t stream = nullptr);
    bool graph_active() const noexcept;
    std::string graph_status() const;
    Exl3TextDecodeAttribution profile_decode(std::int64_t token_id,
                                             cudaStream_t stream = nullptr);

    int position() const noexcept { return position_; }
    const std::uint16_t* logits_device() const noexcept;
    std::vector<float> logits_host(cudaStream_t stream = nullptr) const;
    std::vector<float> embedding_host(cudaStream_t stream = nullptr) const;
    std::vector<std::uint16_t> embedding_bits_host_for_test(
        cudaStream_t stream = nullptr) const;
    std::vector<float> hidden_host(int layer, cudaStream_t stream = nullptr) const;
    std::vector<float> gdn_state_host(int layer, cudaStream_t stream = nullptr) const;
    std::vector<std::uint16_t> gdn_physical_conv_host(
        int layer, cudaStream_t stream = nullptr) const;
    std::shared_ptr<const void> gdn_coefficient_owner_for_test(int layer) const;
    const void* gdn_recurrent_state_identity_for_test(int layer) const;
    const void* gdn_convolution_state_identity_for_test(int layer) const;
    // Read-only bounded row slice of the latest native B1..B8/B16 GDN
    // continuation history. The returned alias owner keeps this context's
    // storage alive; overwrite/reset/restore makes `current()` false.
    Exl3GdnContinuationHistoryView gdn_continuation_history(
        int layer,int first_row,int rows) const;
    // Qualification-only deterministic layer-0 GDN witnesses. Inputs are
    // exactly eight token ids embedded into context-owned hidden storage;
    // returned values are raw F16 output bits indexed by context identity.
    std::vector<std::uint16_t> gdn_layer0_serial_for_test(
        std::span<const std::int64_t> token_ids,
        cudaStream_t stream = nullptr);
    std::array<std::vector<std::uint16_t>, 2>
    gdn_layer0_pair_staged_serial_for_test(
        Exl3TextContext& peer,
        std::span<const std::int64_t> token_ids,
        std::span<const std::int64_t> peer_token_ids,
        bool peer_first = false, cudaStream_t stream = nullptr,
        int fail_after_published_lane = -1,
        Exl3GdnStageOracleTelemetry* telemetry = nullptr);
    std::vector<std::byte> oscar_live_state_host_for_test(
        cudaStream_t stream = nullptr) const;
    Exl3OscarGraphStateObservation oscar_graph_state_host_for_test(
        cudaStream_t stream = nullptr) const;
    std::vector<std::uint16_t> full_attention_k_host(int layer, int position,
                                                      cudaStream_t stream = nullptr) const;
    // E4C1: attach resident OSCAR to the 16 full-attention layers using
    // environment configuration (NINFER_OSCAR_EXL3=1 plus asset paths).
    // Returns false when OSCAR is not requested. Throws fail-closed on bad
    // assets. Ordinary F16 attention stays the path until this succeeds.
    bool try_enable_oscar_from_environment();
    bool oscar_enabled() const noexcept;
    const Exl3OscarTelemetry& oscar_telemetry() const noexcept;
    // E4C1 graph orchestration: fix the split class before capture (16/32/64,
    // 0 selects eager), upload host mirrors to device state, then capture.
    void oscar_set_graph_class(int splits);
    void oscar_sync_device_state(cudaStream_t stream = nullptr);
    void oscar_routing_counts(std::uint64_t& oscar_full, std::uint64_t& ordinary_full) const;
    // Prepare allocates the fixed eager target checkpoint once. Begin, rollback,
    // and commit allocate and synchronize nothing; order them with target work
    // on one stream or establish dependencies externally. One transaction may
    // be active. Canonical OSCAR and captured taps are required. QKV diagnostic
    // scratch is excluded and remains invalid after rollback until a fresh
    // forward completes. Dispatch telemetry and H2D counters describe executed
    // work and are cumulative rather than transaction state.
    void prepare_transaction();
    void prepare_transaction_reserved(Exl3VeriCacheServingCoordinator&,
                                      unsigned startup_fault_for_test = 0);
    std::size_t transaction_bytes_required() const;
    static std::size_t transaction_owner_metadata_bytes() noexcept;
    void begin_transaction(cudaStream_t stream = nullptr);
    // Retain a verified prefix from the one immediate successful continuation
    // after begin_transaction(). Partial retention reconstructs target state
    // asynchronously on the bound stream without model replay or allocation.
    // The transaction stays active for caller publication followed by commit,
    // or rollback if that later publication fails.
    void retain_transaction_prefix(int retained_rows,
                                   cudaStream_t stream = nullptr);
    // The next continue_rows() treats its last `siblings` rows as sibling
    // leaves (sibling_rows.cuh); one-shot.
    void set_verifier_siblings(int siblings);
    // Copies an accepted sibling leaf row of the last continuation into its
    // chain slot (all per-row traces, K/V rows, taps, embedding and logits)
    // before retain_transaction_prefix(destination + 1).
    void promote_sibling_row(int source_row,int destination_row,cudaStream_t stream);
    // Qualification-only deterministic failure injection. Uses the identical
    // retention implementation and throws after repairing fail_after_model_layer
    // (0..63), before logits, position, or row metadata publication.
    void retain_transaction_prefix_for_test(int retained_rows,
                                            int fail_after_model_layer,
                                            cudaStream_t stream = nullptr);
    void rollback_transaction(cudaStream_t stream = nullptr);
    void commit_transaction();
    bool transaction_prepared() const noexcept;
    bool transaction_active() const noexcept;
    std::size_t transaction_bytes() const noexcept;
    // Default-off diagnostic. The environment opt-in is latched when this
    // context is constructed; explicit preparation creates the fixed event
    // pool outside inference. The caller synchronizes before finish.
    bool target_projection_timing_enabled() const noexcept;
    void prepare_target_projection_timing(cudaStream_t stream = nullptr);
    void begin_target_projection_timing_round(int round,
                                              cudaStream_t stream = nullptr);
    void set_target_projection_timing_phase(Exl3TargetProjectionPhase phase);
    std::vector<Exl3TargetProjectionTimingRecord>
        finish_target_projection_timing_round_after_synchronize();
    // Qualification-only observation of represented F16 inputs to one bounded
    // selected K6 or K7 MUL1 family during continuation execution. Weight
    // pointers remain valid for the model lifetime; input is borrowed only for
    // the callback and any stream-ordered copy must finish before scratch reuse.
    // Installation/execution is serialized with this context. The observer may
    // copy but must not mutate buffers, reenter the context, or run with graph
    // capture or target projection timing. The selector is explicit and
    // defaults to the previously qualified gate/up family. No observer means
    // no CUDA resources.
    void set_target_projection_observer_for_test(
        Exl3TargetProjectionObserver observer, void* user = nullptr,
        cudaStream_t stream = nullptr,
        Exl3TargetProjectionObserverSelection selection =
            Exl3TargetProjectionObserverSelection::gate_up_k6);
    void set_layer_observer_for_test(Exl3LayerObserver observer, void* user = nullptr,
                                     cudaStream_t stream = nullptr);
    Exl3FullAttentionQKVHost full_attention_qkv_host(
        int layer, cudaStream_t stream = nullptr) const;
    // E5A2 draft-tap export (copies only; target numerics unchanged). Copies the
    // LAST captured tap row for one of the five retained tap layers (5/19/33/47/61),
    // device-to-device. Requires capture_taps and a completed forward.
    void copy_tap_row_to_device(int layer, std::uint16_t* dst,
                                  cudaStream_t stream = nullptr) const;
    int captured_tap_rows() const noexcept;
    // Blocking authoritative tap export before continuation canonicalization.
    // Ordinary eager FP16 only. Rows belong to the most recent forward.
    std::array<std::vector<std::uint16_t>,5> exact_tap_rows_host(
        cudaStream_t stream = nullptr) const;
    int captured_embedding_rows() const noexcept;
    int last_forward_rows() const noexcept;
    int device_position_host(cudaStream_t stream = nullptr) const;
    // E5A2 range copy of captured tap rows (device-to-device, copies only; target numerics
    // unchanged). first_row/rows are relative to the tap buffer of the LAST completed
    // forward (prefill rows or the most recent decode row).
    void copy_tap_rows_to_device(int layer, int first_row, std::uint16_t* dst,
                                 int rows, cudaStream_t stream = nullptr) const;
    Exl3CommittedTapSegment committed_tap_segment(
        const Exl3CommittedTapBinding& binding,int source_first,int rows,
        int logical_first,int destination_first,int attempt_first,int attempt_rows,
        int source_partition_first,int source_partition_rows,
        bool repair=false,bool correction=false) const;
    // E5A2 target-side single-token embedding lookup (e.g. draft mask token 248070),
    // device-side. Copies only; target numerics unchanged.
    void embed_token_to_device(std::int64_t token_id, std::uint16_t* dst,
                                  cudaStream_t stream = nullptr) const;
    void embed_tokens_to_device(const std::int64_t* ids, std::uint16_t* dst,
                                int rows, cudaStream_t stream = nullptr) const;
    // E5A2 read-only handles to the authoritative target H6 LM head and the
    // BF16 embedding table used by the native DFlash2 draft adapter.
    const Exl3CudaLinearWeights& target_lm_head_weights() const noexcept;
    const Exl3CudaLinearMetadata& target_lm_head_metadata() const noexcept;
    const std::uint16_t* target_embedding() const noexcept;
    // Explicit, default-off capture for a native MTP bridge.  Preparation
    // must happen on a pristine context, before the first target forward.
    // Once prepared, ordinary eager prefill/decode forwards copy their exact
    // final-normalized hidden rows into a separate position-indexed device
    // allocation.  The capture never changes target authority or target head
    // selection; callers still obtain independent handoff copies below.
    void prepare_native_mtp_hidden_capture(int capacity);
    bool native_mtp_hidden_capture_prepared() const noexcept;
    int native_mtp_hidden_capture_capacity() const noexcept;
    int native_mtp_hidden_capture_rows() const noexcept;
    int native_mtp_hidden_capture_first_position() const noexcept;
    std::uint64_t native_mtp_hidden_capture_generation() const noexcept;
    // Copies the exact final-normalized target hidden row into an owned device
    // allocation for a native MTP bridge. The source final-norm/ping-pong
    // buffers are never borrowed by the caller and may be reused immediately
    // after the stream-ordered copy is enqueued.
    Exl3MtpTargetHiddenHandoff target_hidden_handoff(
        int row = 0, cudaStream_t stream = nullptr) const;
    // Bounded batch copy of exact final-normalized target rows.  When the
    // explicit native-MTP capture is prepared and covers the requested range,
    // first_row/rows address that position-indexed capture; otherwise they
    // retain the legacy most-recent-forward meaning.  All returned row views
    // alias one separately owned allocation, so the transient hidden
    // ping-pong/capture buffers may be reused after the producer stream reaches
    // the MTP consumer.  Rows retain absolute positions and one generation.
    std::vector<Exl3MtpTargetHiddenHandoff> target_hidden_handoffs(
        int first_row, int rows, cudaStream_t stream = nullptr) const;
    // Explicit default-off native MTP materialization.  This validates and
    // uploads the closed 39-tensor manifest, binds the resident target
    // embedding/H6 head, and allocates a separate bounded MTP KV frontier.
    // It never changes target authority or enables MTP for ordinary requests.
    std::shared_ptr<Exl3MtpPrefixState> create_native_mtp_prefix_state(
        std::uint32_t kv_capacity = 4'096) const;
    std::size_t persistent_bytes() const noexcept { return persistent_bytes_; }
    // Represented context allocation records and fixed layer/workspace/staging
    // objects. Excludes separately reserved reconstruction/prefix owners, graph
    // driver storage, manifests and opaque allocator/control-block overhead.
    std::uint64_t allocation_owner_metadata_bytes() const;
    std::uint64_t base_linear_owner_metadata_bytes() const noexcept;
    std::uint64_t private_generic_owner_metadata_bytes() const noexcept;
    std::uint64_t shared_generic_owner_metadata_bytes() const noexcept;
    std::uint64_t layer_linear_owner_metadata_bytes() const noexcept;
    // Joined test caller: 0=head,1=auxiliary,2=continuation; then 13 slots/layer.
    bool fail_linear_retirement_for_test(unsigned slot,bool partial=false) noexcept;
    bool fail_attention_buffer_retirement_for_test(unsigned owned_index) noexcept;
    bool fail_gdn_buffer_retirement_for_test(unsigned owned_index) noexcept;
    std::uint64_t fail_layer_buffer_retirement_for_test(bool gdn,unsigned layer,unsigned slot) noexcept;
    bool fail_continuation_allocation_retirement_for_test(unsigned slot,unsigned fault=1) noexcept;
    std::uint64_t stream_reduction_calls(bool extended=false) const noexcept;
    std::uint64_t k6_gateup_warpgroup_async_calls() const noexcept;
    std::uint64_t k6_gateup_n32_pair_cta_calls() const noexcept;
    std::uint64_t k6_fast_decode_calls() const noexcept;
    std::uint64_t k6_fast_decode_rows() const noexcept;
    std::uint64_t k6_rowpair_n64_calls() const noexcept;
    std::uint64_t k6_rowpair_n64_rows() const noexcept;
    std::uint64_t k6_down_rowpair_calls() const noexcept;
    std::uint64_t k6_down_rowpair_rows() const noexcept;
    std::uint64_t shape4_n64_calls() const noexcept;
    std::uint64_t shape4_n64_rows() const noexcept;
    std::uint64_t reduce_shfl_min_barrier_calls() const noexcept;
    std::uint64_t reduce_shfl_min_barrier_rows() const noexcept;
    std::uint64_t k6_gateup_n32_pair_cta_rows() const noexcept;
    std::uint64_t k7_tiles64_exact_splits_calls() const noexcept;
    std::uint64_t target_k8_kv_prefill_async_a_calls() const noexcept;
    std::uint64_t target_k8_kv_prefill_async_a_rows() const noexcept;
    std::uint64_t target_down_k6_async_a_calls() const noexcept;
    std::uint64_t target_gateup_k6_n16_calls() const noexcept;
    std::uint64_t target_k6_small_m_async_a_calls() const noexcept;
    std::uint64_t target_k7_small_m_async_a_calls() const noexcept;
    std::uint64_t target_k5_small_m_batch_calls() const noexcept;
    std::uint64_t native_k6_critical_path_calls() const noexcept;
    std::uint64_t native_k6_register_pipeline_calls() const noexcept;
    std::uint64_t eager_mlp_gateup_concurrent_calls() const noexcept;
    std::uint64_t prefill_qkv_concurrent_calls() const noexcept;
    std::uint64_t prefill_qkv_concurrent_rows() const noexcept;
    std::uint64_t prefill_projection_graph_captures() const noexcept;
    std::uint64_t prefill_projection_graph_replays() const noexcept;
    std::uint64_t prefill_projection_graph_binding_fallbacks() const noexcept;
    std::uint64_t prefill_projection_chain_graph_captures() const noexcept;
    std::uint64_t prefill_projection_chain_graph_replays() const noexcept;
    std::uint64_t prefill_projection_chain_graph_fallbacks() const noexcept;
    std::uint64_t target_prefill_gate_up_pair_attempts() const noexcept;
    std::uint64_t target_prefill_gate_up_pair_calls() const noexcept;
    std::uint64_t target_prefill_gate_up_pair_rows() const noexcept;
    std::uint64_t gqa_six_softmax_triple_v_tile_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_v_tile_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_full_cta_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_full_cta_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_threads128_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_threads128_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_score_tile_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_score_tile_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_key_pair_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_key_pair_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_warp_score_broadcast_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_warp_score_broadcast_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_six_values_single_load_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_six_values_single_load_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_scalar_dim_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_scalar_dim_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_two_query_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_two_query_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_fused_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_triple_fused_row_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_tile512_launch_attempts() const noexcept;
    std::uint64_t gqa_six_softmax_tile512_row_attempts() const noexcept;
    std::size_t last_decode_mallocs() const noexcept { return 0; }
    std::size_t last_decode_frees() const noexcept { return 0; }
    std::size_t last_decode_h2d() const noexcept { return last_decode_h2d_; }
    std::size_t last_decode_d2h() const noexcept { return 0; }
    std::size_t last_decode_syncs() const noexcept { return 0; }

private:
    static std::shared_ptr<const Exl3TurboAngleWarmPages> make_turboangle_pages(
        std::shared_ptr<const Exl3ExactHostState> source, int rows);
    struct Impl;
    explicit Exl3TextContext(std::unique_ptr<Impl> impl);
    void reset_impl(cudaStream_t stream,bool preserve_exact_payload);
    Exl3RequestResetStats reset_for_request_impl(std::string_view contract,
        const Exl3ExactHostState* root,std::uint64_t expected_generation);
    void restore_host_state_impl(const Exl3ExactHostState& state,
        const Exl3TurboAngleWarmPages* warm, bool oscar, cudaStream_t stream);
    void append_prefill_impl(std::span<const std::int64_t> token_ids,
                             cudaStream_t stream, bool wide, bool exact = false);
    void canonicalize_exact_rows(cudaStream_t stream);
    void prepare_transaction_impl(Exl3VeriCacheServingCoordinator*,
                                  unsigned startup_fault_for_test);
    void retain_transaction_prefix_impl(int retained_rows,
                                        int fail_after_model_layer,
                                        cudaStream_t stream);
    std::unique_ptr<Impl> impl_;
    int position_ = 0;
    std::uint16_t* logits_ = nullptr;
    std::size_t persistent_bytes_ = 0;
    std::size_t last_decode_h2d_ = 0;

    void prepare_continuation_impl(int capacity,Exl3VeriCacheServingCoordinator*,unsigned startup_fault=0);
    friend class Exl3TextModel;
};

} // namespace ninfer::exl3
