#pragma once
// E4C1 native EXL3 OSCAR runtime adapter (eager-first).
// Owns calibrated rotations + per-layer device-resident mixed cache and
// dispatches the qualified fused split-KV decode. Ordinary EXL3 attention
// stays the control path; OSCAR is opt-in per context.
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer::exl3 {

inline constexpr int kExl3OscarQHeads = 24;
inline constexpr int kExl3OscarKVHeads = 4;
inline constexpr int kExl3OscarHeadDim = 256;
inline constexpr int kExl3OscarPrefixTokens = 64;
inline constexpr int kExl3OscarRecentTokens = 256;
inline constexpr int kExl3OscarFullLayers[16] = {3,  7,  11, 15, 19, 23, 27, 31,
                                                35, 39, 43, 47, 51, 55, 59, 63};

// Immutable calibrated rotations, one 256x256 FP32 bank per full-attention
// layer, row-vector convention matching the qualified OSCAR reference.
struct Exl3OscarRotations {
    std::string asset_identity;
    std::string manifest_sha256;
    std::array<std::vector<float>, 16> r_k;
    std::array<std::vector<float>, 16> r_v;
};

// Stage 18 loader: fail-closed. Verifies asset identity, manifest digest,
// layer map, geometry, sizes, finite values, and the EXL3 compatibility
// manifest attestation. Never weakens to a wildcard model match.
Exl3OscarRotations exl3_oscar_load_rotations(const std::filesystem::path& asset_dir,
                                             const std::filesystem::path& compat_manifest);

// Production telemetry proving routing (Stage 20).
struct Exl3OscarTelemetry {
    std::array<std::uint64_t, 64> oscar_dispatches{};
    std::uint64_t ordinary_dispatches = 0;
    std::uint64_t gdn_oscar_dispatches = 0;
    std::uint64_t append_calls = 0;
    // Eager continuation-cohort admission is separately observable from the
    // graph-only route. A latch miss is distinct from an adaptive split
    // boundary fallback so tests can prove that the default remains inert.
    std::uint64_t continuation_cohort_eager_attempts = 0;
    std::uint64_t continuation_cohort_eager_dispatches = 0;
    std::uint64_t continuation_cohort_eager_latch_misses = 0;
    std::uint64_t continuation_cohort_eager_boundary_fallbacks = 0;
    std::uint64_t continuation_cohort_eager_malformed = 0;
    std::uint64_t aging_events = 0;
    int last_split_class = 0;
    std::uint64_t resident_cache_bytes = 0;
    std::uint64_t workspace_bytes = 0;
};

// Test-only graph-state observation.  canonical_bytes is populated only when
// every device extent/count and host mirror cross-check is valid.  Invalid
// observations retain diagnostics but never return candidate cache bytes.
struct Exl3OscarGraphStateObservation {
    bool valid = false;
    std::string failure;
    std::array<std::array<int, 5>, 16> device_counts{};
    std::size_t observed_layers = 0;
    std::vector<std::byte> canonical_bytes;
};

// Test-only observation of the actual graph append plan kernel.  The wrapper
// does not allocate or touch an OSCAR cache; only the nine defined plan words
// and five signed count words are returned.
struct Exl3OscarPlanBoundaryObservation {
    std::array<std::uint32_t, 9> plan{};
    std::array<std::int32_t, 5> counts{};
};

Exl3OscarPlanBoundaryObservation oscar_plan_boundary_observe_for_test(
    int old_position);

// Caller-owned storage for one eager OSCAR context transaction. cache_device
// points to checkpoint_device_bytes() bytes of device memory disjoint from all
// context-owned cache/scratch. Descriptor and payload remain immutable while the
// snapshot is usable, and the originating context must remain alive; owner is
// not a lifetime token. A saved checkpoint is bound to its context because
// historical INT2 rows are append-only and intentionally are not copied.
struct Exl3OscarCheckpoint {
    void* cache_device = nullptr;
    std::size_t cache_capacity_bytes = 0;
    std::array<std::uint32_t, 16> contexts{};
    std::array<std::uint32_t, 16> recent_heads{};
    const void* owner = nullptr;
    std::uint64_t generation = 0;
    bool valid = false;
};

// Device-resident mixed cache for one full-attention layer. Cursor math is an
// exact port of the qualified D4.4/D4.8B append driver: encode aged rows
// before publishing new recent rows sharing the same ring slot.
class Exl3OscarLayerCache {
public:
    Exl3OscarLayerCache(int model_layer, int bank_index, int max_context);
    ~Exl3OscarLayerCache();
    Exl3OscarLayerCache(const Exl3OscarLayerCache&) = delete;
    Exl3OscarLayerCache& operator=(const Exl3OscarLayerCache&) = delete;

    // Rotated FP32 K/V in tokens-major [tokens, kv_heads, head_dim] order (head_dim fastest), logical range
    // [logical_start, logical_start + token_count). Must extend context.
    // Returns the number of tokens encoded to INT2 history by this append.
    std::uint32_t append(const float* rotated_k, const float* rotated_v, int token_count,
                         std::uint32_t logical_start, cudaStream_t stream);

    void prefix_extent(std::int32_t& prefix, std::int32_t& historical, std::int32_t& recent,
                       std::int32_t& recent_ring_head) const;
    std::uint32_t context() const noexcept { return context_; }
    int model_layer() const noexcept { return model_layer_; }

    struct View {
        std::uint16_t* prefix_k = nullptr;
        std::uint16_t* prefix_v = nullptr;
        std::uint8_t* hist_k = nullptr;
        std::uint8_t* hist_v = nullptr;
        float* hist_k_meta = nullptr;
        float* hist_v_meta = nullptr;
        std::uint16_t* recent_k = nullptr;
        std::uint16_t* recent_v = nullptr;
        float* workspace = nullptr;
        const float* r_k = nullptr;
        const float* r_v = nullptr;
        const float* r_v_inv = nullptr;
    };
    // Takes ownership of device rotation layouts (FWD RK, FWD RV, INV RV).
    void set_device_rotations(void* rk_fwd, void* rv_fwd, void* rv_inv) noexcept;
    // Graph-safe single-token append/decode. All extents derive on-device
    // from the live position word; launch geometry is fixed.
    void ensure_device_state();
    void append_device(const float* rotated_k, const float* rotated_v,
                       const int* position_device, int row_offset,
                       cudaStream_t stream);
    const int* device_counts() const noexcept;
    void reset_device(cudaStream_t stream);
    void sync_device_state(std::uint32_t context, std::uint32_t head, cudaStream_t stream);
    std::uint32_t host_head() const noexcept { return recent_head_; }
    void note_host_replay(int rows = 1) noexcept;
    View view() const noexcept;
    void reset() noexcept;
    std::uint64_t cache_bytes() const noexcept;
    std::uint64_t workspace_bytes() const noexcept;

private:
    friend class Exl3OscarContext;
    static constexpr std::size_t kCheckpointPrefixBytes =
        static_cast<std::size_t>(kExl3OscarPrefixTokens) * kExl3OscarKVHeads *
        kExl3OscarHeadDim * sizeof(std::uint16_t);
    static constexpr std::size_t kCheckpointRecentBytes =
        static_cast<std::size_t>(kExl3OscarRecentTokens) * kExl3OscarKVHeads *
        kExl3OscarHeadDim * sizeof(std::uint16_t);
    static constexpr std::size_t kCheckpointBytes =
        2 * kCheckpointPrefixBytes + 2 * kCheckpointRecentBytes;
    void save_checkpoint(void* destination, cudaStream_t stream) const;
    void restore_checkpoint(const void* source, std::uint32_t context,
                            std::uint32_t recent_head, cudaStream_t stream);

    int model_layer_ = -1;
    int max_context_ = 0;
    std::uint32_t context_ = 0;
    std::uint32_t recent_head_ = 0;
    void* prefix_k_ = nullptr;
    void* prefix_v_ = nullptr;
    void* hist_k_ = nullptr;
    void* hist_v_ = nullptr;
    void* hist_k_meta_ = nullptr;
    void* hist_v_meta_ = nullptr;
    void* recent_k_ = nullptr;
    void* recent_v_ = nullptr;
    void* workspace_ = nullptr;
    void* dev_r_k_ = nullptr;
    void* dev_r_v_ = nullptr;
    void* dev_r_inv_ = nullptr;
    void* dev_plan_ = nullptr;  // uint32[16]: append plan (device-owned)
    void* dev_counts_ = nullptr;  // int[5]: prefix,hist,recent,head,query (device)
};

// Per-context OSCAR state: 16 layer caches + shared rotation/staging.
// Staging is shared because layers execute sequentially.
class Exl3OscarContext {
public:
    static std::unique_ptr<Exl3OscarContext> create(const Exl3OscarRotations& rotations,
                                                    int max_context, Exl3OscarTelemetry* telemetry);
    ~Exl3OscarContext();
    Exl3OscarContext(const Exl3OscarContext&) = delete;
    Exl3OscarContext& operator=(const Exl3OscarContext&) = delete;

    static bool is_full_attention_layer(int model_layer) noexcept;
    static int bank_index(int model_layer);

    // Append EXL3 F16 taps for [logical_start, logical_start + rows).
    // q_rope/k_rope/v_projection are DEVICE buffers holding [rows, heads, 256]
    // F16 bit units.
    void append_layer(int model_layer, const std::uint16_t* q_rope_f16,
                      const std::uint16_t* k_rope_f16, const std::uint16_t* v_f16, int rows,
                      std::uint32_t logical_start, cudaStream_t stream);
    // Cache restore needs no query or target forward. Same original rotations
    // and append/aging arithmetic as append_layer.
    void append_kv_layer(int model_layer, const std::uint16_t* k_rope_f16,
                         const std::uint16_t* v_f16, int rows,
                         std::uint32_t logical_start, cudaStream_t stream);

    // Single-query OSCAR decode. q_row_f16 is a DEVICE buffer with [24, 256]
    // F16 of the query token (must be the last appended token); attn_out_f16
    // is a DEVICE buffer receiving the recovered [24, 256] F16 attention
    // output (pre-gate, O-proj input).
    // split_count 0 selects the inherited adaptive policy.
    void decode_layer(int model_layer, const std::uint16_t* q_row_f16,
                      std::uint16_t* attn_out_f16, std::uint32_t query_token,
                      int split_count, cudaStream_t stream);

    // Multi-row prefill: appends K/V for [logical_start, +rows), then answers
    // every query with the fused batch path (query-major [rows, 24, 256]).
    // attn_out_f16 is a DEVICE [rows, 24, 256] F16 buffer.
    void prefill_layer(int model_layer, const std::uint16_t* q_rope_f16,
                       const std::uint16_t* k_rope_f16, const std::uint16_t* v_f16,
                       int rows, std::uint32_t logical_start, std::uint16_t* attn_out_f16,
                       cudaStream_t stream);

    // Wide-prefill only: batch independent rotations, but retain each original
    // single-token cache append/aging and split-decode operation in order.
    bool prefill_batch_rotations_enabled() const noexcept { return batch_rotations_; }
    void prefill_chronological_layer(int model_layer, const std::uint16_t* q,
                                     const std::uint16_t* k, const std::uint16_t* v,
                                     int rows, std::uint32_t logical_start,
                                     std::uint16_t* output, cudaStream_t stream);

    void record_ordinary_dispatch() noexcept;
    // Graph-safe single-token decode for capture/replay. All extents are
    // device-driven; launch geometry is fixed by graph_class() (16/32/64).
    // Requires set_graph_class(class) before capture; 0 selects eager.
    void set_graph_class(int splits) noexcept { graph_class_ = splits; }
    int graph_class() const noexcept { return graph_class_; }
    void forward_decode_device(int model_layer, const std::uint16_t* q_row_f16,
                                 const std::uint16_t* k_row_f16,
                                 const std::uint16_t* v_row_f16,
                                 std::uint16_t* attn_out_f16,
                                 const int* position_device, cudaStream_t stream);
    void forward_continuation_device(int model_layer,
        const std::uint16_t* q_rows_f16,const std::uint16_t* k_rows_f16,
        const std::uint16_t* v_rows_f16,std::uint16_t* attn_out_f16,
        int rows,const int* position_device,int split_class,cudaStream_t stream);
    // Default-off chronological cohort. The environment decision is latched at
    // construction. Graph capture still supplies its fixed split class; eager
    // admission uses the guarded helper below and falls back at boundaries.
    void forward_continuation_device_cohort_b8(
        int model_layer,const std::uint16_t* q_rows_f16,
        const std::uint16_t* k_rows_f16,const std::uint16_t* v_rows_f16,
        std::uint16_t* attn_out_f16,const int* position_device,
        int split_class,cudaStream_t stream);
    void forward_continuation_device_cohort(
        int model_layer,const std::uint16_t* q_rows_f16,
        const std::uint16_t* k_rows_f16,const std::uint16_t* v_rows_f16,
        std::uint16_t* attn_out_f16,int rows,const int* position_device,
        int split_class,cudaStream_t stream);
    // Eager-only adapter for target verifier rows. Returns true only when the
    // exact cohort was launched; false leaves the caller responsible for the
    // existing chronological row loop. The host logical start is valid here
    // because this path is excluded during graph capture.
    bool try_forward_continuation_device_cohort_eager(
        int model_layer,const std::uint16_t* q_rows_f16,
        const std::uint16_t* k_rows_f16,const std::uint16_t* v_rows_f16,
        std::uint16_t* attn_out_f16,int rows,int logical_start,
        const int* position_device,cudaStream_t stream);
    bool continuation_cohort_b8_enabled() const noexcept {
        return continuation_cohort_b8_;
    }
    // Host mirror advance after each graph replay (no D2H): the host knows
    // exactly one token was appended by every full-attention layer.
    void note_graph_replay(int rows = 1) noexcept;
    void note_graph_execution(int rows = 1,int split_class = 0) noexcept;
    // Pre-capture guarantee: plan/counts storage exists. Device extents
    // derive from the live position word, so no upload is needed.
    void sync_device_state(cudaStream_t stream);
    // Capture-safe setup for graph kernels (call outside capture).
    void ensure_device_attributes();
    // Save/restore are eager-only, asynchronous on stream, allocate no memory,
    // and copy prefix64 + recent256 K/V for all 16 layers. The caller owns and
    // retains checkpoint storage until the stream has completed the operation.
    // Order cache use and the copy on one stream, or establish the dependency
    // externally. Each save invalidates older snapshots; reset invalidates all.
    // Repeated restore of the latest snapshot is valid while only appends occur.
    // Restore reads but never modifies the caller's descriptor or host metadata.
    std::size_t checkpoint_device_bytes() const noexcept;
    void save_checkpoint(Exl3OscarCheckpoint& checkpoint, cudaStream_t stream);
    void restore_checkpoint(const Exl3OscarCheckpoint& checkpoint, cudaStream_t stream);
    // Qualification-only read of live logical cache state. Per layer: six u32
    // fields (model layer/context/prefix/history/recent/head), prefix K/V BF16,
    // historical K codes/metadata then V codes/metadata, recent K/V BF16 in
    // logical ring order. Excludes unused capacity, plans and telemetry.
    // Allocates host memory and synchronizes; serialize with all context work.
    // Never saves a checkpoint or changes cache generations. Eager-only.
    std::vector<std::byte> live_state_host_for_test(cudaStream_t stream = nullptr) const;
    Exl3OscarGraphStateObservation graph_live_state_host_for_test(
        cudaStream_t stream = nullptr) const;
    // Operator-probe export for one device-positioned graph layer. Extents
    // come from that layer's device counts, so no host cursor publication is
    // required before comparing sequential and cohort candidates.
    std::vector<std::byte> graph_layer_state_host_for_test(
        int model_layer,cudaStream_t stream = nullptr) const;
    void reset() noexcept;
    const Exl3OscarTelemetry& telemetry() const noexcept { return *telemetry_; }

private:
    Exl3OscarContext(const Exl3OscarRotations& rotations, int max_context,
                     Exl3OscarTelemetry* telemetry);
    void decode_rotated_layer(int model_layer, const float* rotated_q,
                              float* av, std::uint32_t query_token,
                              int split_count, cudaStream_t stream);
    bool batch_rotations_ = false;
    bool prefill_parallel_merge_ = false;
    bool prefill_parallel_scores_ = false;
    bool prefill_coalesced_rotations_ = false;
    bool decode_coalesced_rotations_ = false;
    bool decode_fused_kv_rotations_ = false;
    bool continuation_cohort_b8_ = false;
    bool prefill_query_parallel_ = false;
    bool prefill_packed_byte4_ = false; // Exact-1 construction latch; eager query-parallel prefill only.
    std::shared_ptr<void> prefill_old_recent_;
    std::shared_ptr<void> prefill_query_workspace_;
    std::array<std::shared_ptr<void>,16> coalesced_r_k_;
    int batch_staging_rows_ = 64;
    void* staging_rk_ = nullptr;  // FP32 tokens-major [chunk, 4, 256] rotated K
    void* staging_rv_ = nullptr;  // FP32 tokens-major [chunk, 4, 256] rotated V
    void* staging_rq_ = nullptr;  // FP32 [24, 256] rotated decode Q
    void* staging_bq_ = nullptr;  // FP32 [batch_staging_rows_, 24, 256] rotated prefill Q
    void* staging_bav_ = nullptr;  // FP32 [batch_staging_rows_, 24, 256] fused batch AV
    void* staging_av_ = nullptr;  // FP32 [24, 256] query-major fused AV
    int chunk_rows_ = 256;
    int max_context_ = 0;
    int graph_class_ = 0;
    std::uint64_t checkpoint_generation_ = 0;
    Exl3OscarTelemetry* telemetry_ = nullptr;
    std::array<std::unique_ptr<Exl3OscarLayerCache>, 16> layers_;
};

}  // namespace ninfer::exl3
