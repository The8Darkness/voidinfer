#pragma once

// Correctness-first NVFP4 packed-KV attention. This is intentionally a separate kernel family
// from the tensor-core BF16/INT8/FP8 implementations: NVFP4 codes are decoded in the warp and
// the result is verified by the caller's authoritative target path before it is treated as an
// exact continuation. It is an opt-in draft-cache implementation, not a claim that lossy KV is
// mathematically lossless.

#include "ops/kv_cache/nvfp4_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, bool Masked>
__launch_bounds__(32) __global__ void causal_attention_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride,
    std::int32_t logical_pages, std::int32_t width, float attention_scale,
    __nv_bfloat16* __restrict__ out) {
    constexpr int D             = kKVCacheNvfp4HeadDim;
    constexpr int Threads       = 32;
    constexpr float Log2E       = 1.4426950408889634074F;
    constexpr unsigned FullMask = 0xffffffffU;

    const int q_head = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int lane   = static_cast<int>(threadIdx.x);
    if (q_head >= Geometry::QHeads || token >= width) return;

    const int flat_column = batch * width + token;
    const int valid_tokens = [&]() {
        if constexpr (Masked) {
            const int valid = valid_columns[batch];
            return valid <= 0 ? 0 : min(valid, width);
        }
        return width;
    }();
    const auto write_zero = [&]() {
        for (int d = lane; d < D; d += Threads) {
            out[causal_q_index<Geometry>(q_head, d, flat_column)] = __float2bfloat16(0.0F);
        }
    };
    if (token >= valid_tokens) {
        write_zero();
        return;
    }

    const int q_abs = positions[flat_column];
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if (q_abs < 0 || q_abs >= logical_pages * kPagedKVPageSize || table_row < 0) {
        write_zero();
        return;
    }
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const int kv_head = q_head / Geometry::GroupSize;
    const std::int64_t q_base = static_cast<std::int64_t>(D) * Geometry::QHeads * flat_column;
    const std::int64_t out_base = q_base;

    float q_values[D / Threads];
#pragma unroll
    for (int r = 0; r < D / Threads; ++r) {
        q_values[r] = __bfloat162float(q[q_base + static_cast<std::int64_t>(D) * q_head +
                                           lane + Threads * r]);
    }

    float numerator[D / Threads] = {};
    float running_m                 = -CUDART_INF_F;
    float running_l                 = 0.0F;
    for (int key = 0; key <= q_abs; ++key) {
        const int physical_page = block_table[key >> kPagedKVPageShift];
        const int page_offset   = key & kPagedKVPageMask;
        const std::uint8_t* k_codes =
            cache_k + paged_kv_page_head_offset<kKVCacheNvfp4CodeExtent, Geometry::KVHeads>(
                          physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4CodeExtent;
        const std::uint8_t* v_codes =
            cache_v + paged_kv_page_head_offset<kKVCacheNvfp4CodeExtent, Geometry::KVHeads>(
                          physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4CodeExtent;
        const std::uint8_t* k_scales =
            cache_k_scale + paged_kv_page_head_offset<kKVCacheNvfp4ScaleExtent, Geometry::KVHeads>(
                                physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4ScaleExtent;
        const std::uint8_t* v_scales =
            cache_v_scale + paged_kv_page_head_offset<kKVCacheNvfp4ScaleExtent, Geometry::KVHeads>(
                                physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4ScaleExtent;

        float dot = 0.0F;
        float v_values[D / Threads];
#pragma unroll
        for (int r = 0; r < D / Threads; ++r) {
            const int d = lane + Threads * r;
            const float k_value = kv_cache_nvfp4_decode_value(k_codes, k_scales, d);
            v_values[r]            = kv_cache_nvfp4_decode_value(v_codes, v_scales, d);
            dot                    = __fmaf_rn(q_values[r], k_value, dot);
        }
        const float score = warp_sum(dot, FullMask) * attention_scale;
        const float new_m = fmaxf(running_m, score);
        const float alpha = running_m == -CUDART_INF_F
                                ? 0.0F
                                : exp2_approx((running_m - new_m) * Log2E);
        const float probability = exp2_approx((score - new_m) * Log2E);
#pragma unroll
        for (int r = 0; r < D / Threads; ++r) {
            numerator[r] = __fmaf_rn(numerator[r], alpha, probability * v_values[r]);
        }
        running_l = running_l * alpha + probability;
        running_m = new_m;
    }

    const float inverse_l = running_l > 0.0F ? __frcp_rn(running_l) : 0.0F;
#pragma unroll
    for (int r = 0; r < D / Threads; ++r) {
        out[out_base + static_cast<std::int64_t>(D) * q_head + lane + Threads * r] =
            __float2bfloat16(numerator[r] * inverse_l);
    }
}

} // namespace ninfer::ops
