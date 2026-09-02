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
    std::size_t score_static_shared_bytes = 0;
    std::size_t softmax_static_shared_bytes = 0;
    std::size_t av_static_shared_bytes = 0;
    int score_max_threads = 0;
    int softmax_max_threads = 0;
    int av_max_threads = 0;
};

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
