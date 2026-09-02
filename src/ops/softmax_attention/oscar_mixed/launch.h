#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Complete mixed-tier diagnostic contract. Rows are supplied in logical order within
// each tier; the logical attention order is prefix, historical, recent.
//   BF16 rows: [tokens, 4, 256] as raw BF16 bits
//   INT2 rows: [tokens, 4, 64] bytes and [tokens, 4, 4] FP32 metadata
struct OscarMixedKernelResources {
    int score_registers = 0;
    int softmax_registers = 0;
    int av_registers = 0;
    int fused_registers = 0;
    std::size_t score_static_shared_bytes = 0;
    std::size_t softmax_static_shared_bytes = 0;
    std::size_t av_static_shared_bytes = 0;
    std::size_t fused_dynamic_shared_bytes = 0;
    int score_max_threads = 0;
    int softmax_max_threads = 0;
    int av_max_threads = 0;
    int fused_max_threads = 0;
};

// D4.6's fixed production configuration is retained as a named baseline. D4.7B uses the
// adaptive sentinel below in the production caller and selects one of the supported values at
// launch time. An explicit supported value remains useful to the direct CUDA qualification and
// benchmark harnesses; it does not change cache representation or attention arithmetic.
inline constexpr int kOscarMixedFusedDecodeSplits = 4;
inline constexpr int kOscarMixedFusedDecodeAdaptiveSplits = 0;
inline constexpr int kOscarMixedFusedDecodeMaxSplits = 64;

[[nodiscard]] constexpr std::size_t kOscarMixedFusedDecodeWorkspaceBytesForSplits(
    int split_count) noexcept {
    return static_cast<std::size_t>(split_count * 4 * 6 * 256 + split_count * 4 * 6 * 2) *
           sizeof(float);
}

// Allocated once per resident cache and reused by all decode calls. This is intentionally sized
// for the largest production policy value so adaptive dispatch never allocates in the hot loop.
inline constexpr std::size_t kOscarMixedFusedDecodeWorkspaceBytes =
    kOscarMixedFusedDecodeWorkspaceBytesForSplits(kOscarMixedFusedDecodeMaxSplits);

inline constexpr std::size_t kOscarMixedFusedDecodeD46WorkspaceBytes =
    kOscarMixedFusedDecodeWorkspaceBytesForSplits(kOscarMixedFusedDecodeSplits);

// Return the production split count for a visible logical sequence length. The implementation is
// intentionally small and deterministic; thresholds are qualified by the D4.7B report.
int oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(std::int32_t visible_tokens);

// Persistent device-side cache view used by the D4.4 resident path.  Historical rows remain
// packed INT2 plus FP32 metadata; prefix/recent rows remain BF16.  The view is intentionally
// layout-only and carries no ownership.
struct OscarInt2G128ResidentCacheView {
    std::uint16_t* prefix_k_bf16 = nullptr;
    std::uint16_t* prefix_v_bf16 = nullptr;
    std::uint8_t* historical_k_packed = nullptr;
    std::uint8_t* historical_v_packed = nullptr;
    float* historical_k_metadata = nullptr;
    float* historical_v_metadata = nullptr;
    std::uint16_t* recent_k_bf16 = nullptr;
    std::uint16_t* recent_v_bf16 = nullptr;
    std::int32_t max_context = 0;
};

void oscar_int2_g128_mixed_attention_launch(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens, float attention_scale,
    float* scores, float* softmax, float* output, cudaStream_t stream);

// Same qualified mixed reader, with a physical ring offset for the BF16 recent window.  The
// legacy API above is preserved as the ring-zero D4.2b/D4.3 control path.
void oscar_int2_g128_mixed_attention_launch_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, float attention_scale, float* scores, float* softmax,
    float* output, cudaStream_t stream);

// Batched causal prefill variant for the D4.4 resident cache.  Queries are contiguous logical
// columns starting at query_start; each query still sees only keys <= query_start + query_index.
// Scores/softmax use query-major rows with workspace_stride elements per Q/KV row.  The scratch
// buffer is reused by successive batches and is not a decoded-cache representation.
void oscar_int2_g128_mixed_attention_launch_batch_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, std::int32_t query_start, std::int32_t query_count,
    std::int32_t workspace_stride, float attention_scale, float* scores, float* softmax,
    float* output, cudaStream_t stream);

// Fused resident reader. One block owns one KV head and a small query tile, then walks the
// logical prefix/history/recent sequence once per 32-token K/V tile. K and V are decoded into a
// reused shared-memory slab, consumed by an online stable softmax + AV update, and discarded
// before the next tile. A one-query call uses the decode-specialized 64-row variant.
// query_start is the absolute logical position of q_rotated column zero; query_count columns are
// contiguous.  The compressed resident cache remains the only historical representation.
void oscar_int2_g128_mixed_attention_launch_fused_batch_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, std::int32_t query_start, std::int32_t query_count,
    float attention_scale, float* output, cudaStream_t stream);

// Split-KV decode variant. Each split independently computes online-softmax partials over a
// disjoint logical history interval; a fixed-size merge combines those partials. Passing zero for
// split_count selects the production context-adaptive policy. Passing a supported nonzero value
// is reserved for qualification/benchmark control. The workspace is constant-sized and contains
// no history-proportional score/probability tensor.
void oscar_int2_g128_mixed_attention_launch_fused_decode_split_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, std::int32_t query_token, std::int32_t split_count,
    float attention_scale, float* partial_workspace, float* output, cudaStream_t stream);

// Device append primitives for the D4.4 cache.  Input K/V are the actual rotated FP32 runtime
// tensors in [head_dim, kv_heads, tokens] order.  The write primitive publishes only new
// prefix/recent rows; the encode primitive publishes new historical rows and rows aging out of
// the old recent ring.  Both use the official BF16-rounding -> clipped asymmetric G128 equations.
void oscar_int2_g128_cache_write_bf16_launch(
    const float* rotated_k, const float* rotated_v, std::int32_t token_count,
    std::uint32_t logical_start, std::uint32_t final_recent_begin,
    std::uint32_t final_recent_head, const OscarInt2G128ResidentCacheView& cache,
    cudaStream_t stream);

void oscar_int2_g128_cache_encode_launch(
    const float* rotated_k, const float* rotated_v, std::int32_t token_count,
    std::uint32_t logical_start, std::uint32_t old_context, std::uint32_t old_recent_begin,
    std::uint32_t old_recent_head, std::uint32_t final_recent_begin,
    std::uint32_t new_historical_tokens, std::uint32_t old_aging_tokens,
    const OscarInt2G128ResidentCacheView& cache, cudaStream_t stream);

OscarMixedKernelResources oscar_int2_g128_mixed_kernel_resources();

} // namespace ninfer::ops::detail
