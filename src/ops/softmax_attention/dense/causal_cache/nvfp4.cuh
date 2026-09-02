#pragma once

// Correctness-first NVFP4 packed-KV attention. This is intentionally a separate kernel family
// from the tensor-core BF16/INT8/FP8 implementations: NVFP4 codes are decoded in the warp and
// the result is verified by the caller's authoritative target path before it is treated as an
// exact continuation. The pair-decode route is the serving default; NINFER_NVFP4_PAIR=0 restores
// the scalar fallback and NINFER_NVFP4_XQA=1/2 selects the experimental grouped routes.

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

// Warp-local packed-row path.  The scalar fallback above is deliberately simple, but for D=256
// it decodes each E2M1 byte once per nibble and reloads each of the sixteen E4M3 scales in every
// lane.  This layout assigns one packed byte to each lane, broadcasts the sixteen scales from
// four warp leaders, and keeps the same per-query online-softmax state and output contract.
template <typename Geometry, bool Masked>
__launch_bounds__(32) __global__ void causal_attention_nvfp4_pair_kernel(
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
    const std::int64_t q_base = static_cast<std::int64_t>(D) * Geometry::QHeads * flat_column;
    const std::int64_t out_base = q_base;
    const __nv_bfloat16* q_row = q + q_base + static_cast<std::int64_t>(D) * q_head;
    __nv_bfloat16* out_row = out + out_base + static_cast<std::int64_t>(D) * q_head;

    // The four pairs cover dimensions [0,64), [64,128), [128,192), and [192,256).  Keeping the
    // pair order also makes the output writes coalesced while avoiding duplicate packed-byte loads.
    const int d0 = lane * 2;
    const int d1 = d0 + 64;
    const int d2 = d0 + 128;
    const int d3 = d0 + 192;
    float q_values[8];
    q_values[0] = __bfloat162float(q_row[d0]);
    q_values[1] = __bfloat162float(q_row[d0 + 1]);
    q_values[2] = __bfloat162float(q_row[d1]);
    q_values[3] = __bfloat162float(q_row[d1 + 1]);
    q_values[4] = __bfloat162float(q_row[d2]);
    q_values[5] = __bfloat162float(q_row[d2 + 1]);
    q_values[6] = __bfloat162float(q_row[d3]);
    q_values[7] = __bfloat162float(q_row[d3 + 1]);

    float numerator[8] = {};
    float running_m    = -CUDART_INF_F;
    float running_l    = 0.0F;
    for (int key = 0; key <= q_abs; ++key) {
        const int physical_page = block_table[key >> kPagedKVPageShift];
        const int page_offset   = key & kPagedKVPageMask;
        const std::uint8_t* k_codes =
            cache_k + paged_kv_page_head_offset<kKVCacheNvfp4CodeExtent, Geometry::KVHeads>(
                          physical_page, q_head / Geometry::GroupSize) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4CodeExtent;
        const std::uint8_t* v_codes =
            cache_v + paged_kv_page_head_offset<kKVCacheNvfp4CodeExtent, Geometry::KVHeads>(
                          physical_page, q_head / Geometry::GroupSize) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4CodeExtent;
        const std::uint8_t* k_scales =
            cache_k_scale +
            paged_kv_page_head_offset<kKVCacheNvfp4ScaleExtent, Geometry::KVHeads>(
                physical_page, q_head / Geometry::GroupSize) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4ScaleExtent;
        const std::uint8_t* v_scales =
            cache_v_scale +
            paged_kv_page_head_offset<kKVCacheNvfp4ScaleExtent, Geometry::KVHeads>(
                physical_page, q_head / Geometry::GroupSize) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4ScaleExtent;

        float key_scale0 = 0.0F;
        float key_scale1 = 0.0F;
        float key_scale2 = 0.0F;
        float key_scale3 = 0.0F;
        float value_scale0 = 0.0F;
        float value_scale1 = 0.0F;
        float value_scale2 = 0.0F;
        float value_scale3 = 0.0F;
        if ((lane & 7) == 0) {
            const int group = lane >> 3;
            key_scale0     = kv_cache_nvfp4_decode_scale(k_scales[group]);
            key_scale1     = kv_cache_nvfp4_decode_scale(k_scales[group + 4]);
            key_scale2     = kv_cache_nvfp4_decode_scale(k_scales[group + 8]);
            key_scale3     = kv_cache_nvfp4_decode_scale(k_scales[group + 12]);
            value_scale0   = kv_cache_nvfp4_decode_scale(v_scales[group]);
            value_scale1   = kv_cache_nvfp4_decode_scale(v_scales[group + 4]);
            value_scale2   = kv_cache_nvfp4_decode_scale(v_scales[group + 8]);
            value_scale3   = kv_cache_nvfp4_decode_scale(v_scales[group + 12]);
        }
        const int scale_lane = (lane >> 3) * 8;
        key_scale0           = __shfl_sync(FullMask, key_scale0, scale_lane);
        key_scale1           = __shfl_sync(FullMask, key_scale1, scale_lane);
        key_scale2           = __shfl_sync(FullMask, key_scale2, scale_lane);
        key_scale3           = __shfl_sync(FullMask, key_scale3, scale_lane);
        value_scale0         = __shfl_sync(FullMask, value_scale0, scale_lane);
        value_scale1         = __shfl_sync(FullMask, value_scale1, scale_lane);
        value_scale2         = __shfl_sync(FullMask, value_scale2, scale_lane);
        value_scale3         = __shfl_sync(FullMask, value_scale3, scale_lane);

        const float2 key_pair0   = kv_cache_nvfp4_decode_pair(k_codes[lane]);
        const float2 key_pair1   = kv_cache_nvfp4_decode_pair(k_codes[lane + 32]);
        const float2 key_pair2   = kv_cache_nvfp4_decode_pair(k_codes[lane + 64]);
        const float2 key_pair3   = kv_cache_nvfp4_decode_pair(k_codes[lane + 96]);
        const float2 value_pair0 = kv_cache_nvfp4_decode_pair(v_codes[lane]);
        const float2 value_pair1 = kv_cache_nvfp4_decode_pair(v_codes[lane + 32]);
        const float2 value_pair2 = kv_cache_nvfp4_decode_pair(v_codes[lane + 64]);
        const float2 value_pair3 = kv_cache_nvfp4_decode_pair(v_codes[lane + 96]);

        float dot = 0.0F;
        dot       = __fmaf_rn(q_values[0], key_pair0.x * key_scale0, dot);
        dot       = __fmaf_rn(q_values[1], key_pair0.y * key_scale0, dot);
        dot       = __fmaf_rn(q_values[2], key_pair1.x * key_scale1, dot);
        dot       = __fmaf_rn(q_values[3], key_pair1.y * key_scale1, dot);
        dot       = __fmaf_rn(q_values[4], key_pair2.x * key_scale2, dot);
        dot       = __fmaf_rn(q_values[5], key_pair2.y * key_scale2, dot);
        dot       = __fmaf_rn(q_values[6], key_pair3.x * key_scale3, dot);
        dot       = __fmaf_rn(q_values[7], key_pair3.y * key_scale3, dot);
        const float score = warp_sum(dot, FullMask) * attention_scale;
        const float new_m = fmaxf(running_m, score);
        const float alpha = running_m == -CUDART_INF_F
                                ? 0.0F
                                : exp2_approx((running_m - new_m) * Log2E);
        const float probability = exp2_approx((score - new_m) * Log2E);
        numerator[0] = __fmaf_rn(numerator[0], alpha, probability * value_pair0.x * value_scale0);
        numerator[1] = __fmaf_rn(numerator[1], alpha, probability * value_pair0.y * value_scale0);
        numerator[2] = __fmaf_rn(numerator[2], alpha, probability * value_pair1.x * value_scale1);
        numerator[3] = __fmaf_rn(numerator[3], alpha, probability * value_pair1.y * value_scale1);
        numerator[4] = __fmaf_rn(numerator[4], alpha, probability * value_pair2.x * value_scale2);
        numerator[5] = __fmaf_rn(numerator[5], alpha, probability * value_pair2.y * value_scale2);
        numerator[6] = __fmaf_rn(numerator[6], alpha, probability * value_pair3.x * value_scale3);
        numerator[7] = __fmaf_rn(numerator[7], alpha, probability * value_pair3.y * value_scale3);
        running_l    = running_l * alpha + probability;
        running_m    = new_m;
    }

    const float inverse_l = running_l > 0.0F ? __frcp_rn(running_l) : 0.0F;
    out_row[d0]     = __float2bfloat16(numerator[0] * inverse_l);
    out_row[d0 + 1] = __float2bfloat16(numerator[1] * inverse_l);
    out_row[d1]     = __float2bfloat16(numerator[2] * inverse_l);
    out_row[d1 + 1] = __float2bfloat16(numerator[3] * inverse_l);
    out_row[d2]     = __float2bfloat16(numerator[4] * inverse_l);
    out_row[d2 + 1] = __float2bfloat16(numerator[5] * inverse_l);
    out_row[d3]     = __float2bfloat16(numerator[6] * inverse_l);
    out_row[d3 + 1] = __float2bfloat16(numerator[7] * inverse_l);
}

// Experimental XQA path for packed NVFP4 KV. One CTA owns a KV head and all query heads that
// share it. The CTA cooperatively decodes each packed K/V row once into shared memory, then every
// query-head warp consumes that row. The exact target cache remains authoritative; this route is
// only a low-bit execution optimization and is enabled by the host-side experiment flag.
template <typename Geometry, bool Masked, int Split = 1>
__launch_bounds__((Geometry::GroupSize / Split) * 32, 2)
    __global__ void causal_attention_nvfp4_xqa_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride,
    std::int32_t logical_pages, std::int32_t width, float attention_scale,
    __nv_bfloat16* __restrict__ out) {
    constexpr int D              = kKVCacheNvfp4HeadDim;
    constexpr int WarpThreads    = 32;
    constexpr int WarpsPerCTA    = Geometry::GroupSize / Split;
    constexpr int BlockThreads   = WarpsPerCTA * WarpThreads;
    constexpr float Log2E        = 1.4426950408889634074F;
    constexpr unsigned FullMask  = 0xffffffffU;
    static_assert(Split > 0 && Geometry::GroupSize % Split == 0);

    const int cta     = static_cast<int>(blockIdx.x);
    const int kv_head = cta / Split;
    const int split   = cta - kv_head * Split;
    const int token   = static_cast<int>(blockIdx.y);
    const int batch   = static_cast<int>(blockIdx.z);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid / WarpThreads;
    const int lane    = tid & (WarpThreads - 1);
    if (kv_head >= Geometry::KVHeads || token >= width || warp >= WarpsPerCTA) return;

    const int q_head       = kv_head * Geometry::GroupSize + split * WarpsPerCTA + warp;
    const int flat_column  = batch * width + token;
    const int valid_tokens = [&]() {
        if constexpr (Masked) {
            const int valid = valid_columns[batch];
            return valid <= 0 ? 0 : min(valid, width);
        }
        return width;
    }();
    const auto write_zero = [&]() {
        for (int d = lane; d < D; d += WarpThreads) {
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
    const std::int64_t q_base = static_cast<std::int64_t>(D) * Geometry::QHeads * flat_column;
    const std::int64_t out_base = q_base;

    float q_values[D / WarpThreads];
#pragma unroll
    for (int r = 0; r < D / WarpThreads; ++r) {
        q_values[r] = __bfloat162float(q[q_base + static_cast<std::int64_t>(D) * q_head +
                                           lane + WarpThreads * r]);
    }

    // Ping-pong the decoded row so the next row can be staged while slower query warps are
    // finishing the current row.  This removes the second barrier that would otherwise be
    // needed before overwriting a single shared tile.
    __shared__ float shared_key[2][D];
    __shared__ float shared_value[2][D];
    float numerator[D / WarpThreads] = {};
    float running_m                  = -CUDART_INF_F;
    float running_l                  = 0.0F;
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
            cache_k_scale +
            paged_kv_page_head_offset<kKVCacheNvfp4ScaleExtent, Geometry::KVHeads>(
                physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4ScaleExtent;
        const std::uint8_t* v_scales =
            cache_v_scale +
            paged_kv_page_head_offset<kKVCacheNvfp4ScaleExtent, Geometry::KVHeads>(
                physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * kKVCacheNvfp4ScaleExtent;

        const int buffer = key & 1;
        for (int d = tid; d < D; d += BlockThreads) {
            shared_key[buffer][d]   = kv_cache_nvfp4_decode_value(k_codes, k_scales, d);
            shared_value[buffer][d] = kv_cache_nvfp4_decode_value(v_codes, v_scales, d);
        }
        __syncthreads();

        float dot = 0.0F;
#pragma unroll
        for (int r = 0; r < D / WarpThreads; ++r) {
            dot = __fmaf_rn(q_values[r], shared_key[buffer][lane + WarpThreads * r], dot);
        }
        const float score = warp_sum(dot, FullMask) * attention_scale;
        const float new_m = fmaxf(running_m, score);
        const float alpha = running_m == -CUDART_INF_F
                                ? 0.0F
                                : exp2_approx((running_m - new_m) * Log2E);
        const float probability = exp2_approx((score - new_m) * Log2E);
#pragma unroll
        for (int r = 0; r < D / WarpThreads; ++r) {
            numerator[r] = __fmaf_rn(numerator[r], alpha,
                                     probability * shared_value[buffer][lane + WarpThreads * r]);
        }
        running_l = running_l * alpha + probability;
        running_m = new_m;
    }

    const float inverse_l = running_l > 0.0F ? __frcp_rn(running_l) : 0.0F;
#pragma unroll
    for (int r = 0; r < D / WarpThreads; ++r) {
        out[out_base + static_cast<std::int64_t>(D) * q_head + lane + WarpThreads * r] =
            __float2bfloat16(numerator[r] * inverse_l);
    }
}

} // namespace ninfer::ops
