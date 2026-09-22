#pragma once

// Direct paged OSCAR-Q2 attention for the Qwen3.8 full-attention KV cache.  The cache stores the
// normalized H256-rotated K/V row as affine packed symbols plus a BF16 (scale, zero) pair.  The
// warp reconstructs each row directly from the page store, so an FP16 KV copy is not materialized
// on the device just to consume L0.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/oscar_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, int Bits, bool Masked, bool TransposedQ2 = false>
__launch_bounds__(32) __global__ void causal_attention_oscar_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v,
    const __nv_bfloat16* __restrict__ cache_k_scale,
    const __nv_bfloat16* __restrict__ cache_v_scale, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride,
    std::int32_t logical_pages, std::int32_t width, float attention_scale,
    __nv_bfloat16* __restrict__ out) {
    static_assert(Bits == 2 || Bits == 4);
    static_assert(!TransposedQ2 || Bits == 2);
    constexpr int D             = kPagedKVCacheOscarHeadDim;
    constexpr int CodeExtent    = cyclic_oscar_code_extent<Bits>(D);
    constexpr int ScaleExtent   = kPagedKVCacheOscarScaleExtent;
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

    const int q_abs     = positions[flat_column];
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if (q_abs < 0 || q_abs >= logical_pages * kPagedKVPageSize || table_row < 0) {
        write_zero();
        return;
    }
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const int kv_head = q_head / Geometry::GroupSize;
    const std::int64_t q_base = static_cast<std::int64_t>(D) * Geometry::QHeads * flat_column;

    float q_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        q_values[r] = __bfloat162float(q[q_base + static_cast<std::int64_t>(D) * q_head +
                                           lane + Threads * r]);
    }
    normalized_hadamard_d256_inplace(q_values, lane);

    float numerator[8] = {};
    float running_m    = -CUDART_INF_F;
    float running_l    = 0.0F;
    for (int key = 0; key <= q_abs; ++key) {
        const int physical_page = block_table[key >> kPagedKVPageShift];
        const int page_offset   = key & kPagedKVPageMask;
        const std::uint8_t* k_codes =
            cache_k + paged_kv_page_head_offset<CodeExtent, Geometry::KVHeads>(physical_page,
                                                                                 kv_head) +
            static_cast<std::int64_t>(page_offset) * CodeExtent;
        const std::uint8_t* v_codes =
            cache_v + paged_kv_page_head_offset<CodeExtent, Geometry::KVHeads>(physical_page,
                                                                                 kv_head) +
            static_cast<std::int64_t>(page_offset) * CodeExtent;
        const __nv_bfloat16* k_metadata =
            cache_k_scale +
            paged_kv_page_head_offset<ScaleExtent, Geometry::KVHeads>(physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * ScaleExtent;
        const __nv_bfloat16* v_metadata =
            cache_v_scale +
            paged_kv_page_head_offset<ScaleExtent, Geometry::KVHeads>(physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * ScaleExtent;

        float k_scale            = 0.0F;
        float k_zero             = 0.0F;
        float v_scale            = 0.0F;
        float v_zero             = 0.0F;
        if (lane == 0) {
            k_scale = __bfloat162float(k_metadata[0]);
            k_zero  = __bfloat162float(k_metadata[1]);
            v_scale = __bfloat162float(v_metadata[0]);
            v_zero  = __bfloat162float(v_metadata[1]);
        }
        k_scale = __shfl_sync(FullMask, k_scale, 0);
        k_zero  = __shfl_sync(FullMask, k_zero, 0);
        v_scale = __shfl_sync(FullMask, v_scale, 0);
        v_zero  = __shfl_sync(FullMask, v_zero, 0);

        float dot = 0.0F;
        float v_values[8];
        if constexpr (TransposedQ2) {
            // The Q2 writer stores dimensions lane+32*r in bytes lane (r<4) and lane+32
            // (r>=4). This keeps the decoder in the same natural H256 distribution as Q and
            // removes the four-lane byte broadcasts required by the legacy contiguous stream.
            const int packed_k0 = static_cast<int>(k_codes[lane]);
            const int packed_k1 = static_cast<int>(k_codes[lane + 32]);
            const int packed_v0 = static_cast<int>(v_codes[lane]);
            const int packed_v1 = static_cast<int>(v_codes[lane + 32]);
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int shift      = r << 1;
                const float k_value0 =
                    fmaf(static_cast<float>((packed_k0 >> shift) & 3), k_scale, k_zero);
                const float k_value1 =
                    fmaf(static_cast<float>((packed_k1 >> shift) & 3), k_scale, k_zero);
                v_values[r] =
                    fmaf(static_cast<float>((packed_v0 >> shift) & 3), v_scale, v_zero);
                v_values[r + 4] =
                    fmaf(static_cast<float>((packed_v1 >> shift) & 3), v_scale, v_zero);
                dot            = __fmaf_rn(q_values[r], k_value0, dot);
                dot            = __fmaf_rn(q_values[r + 4], k_value1, dot);
            }
        } else {
            // Compatibility decoder for the original contiguous Q2 page layout. Four adjacent
            // lanes share each byte, so the leader fetches and broadcasts it.
            const int quartet_leader = lane & ~3;
            const int symbol_shift   = (lane & 3) << 1;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int code_byte = (lane >> 2) + (r << 3);
                const int packed_k = __shfl_sync(
                    FullMask, (lane & 3) == 0 ? static_cast<int>(k_codes[code_byte]) : 0,
                    quartet_leader);
                const int packed_v = __shfl_sync(
                    FullMask, (lane & 3) == 0 ? static_cast<int>(v_codes[code_byte]) : 0,
                    quartet_leader);
                const float k_value =
                    fmaf(static_cast<float>((packed_k >> symbol_shift) & 3), k_scale, k_zero);
                v_values[r] =
                    fmaf(static_cast<float>((packed_v >> symbol_shift) & 3), v_scale, v_zero);
                dot = __fmaf_rn(q_values[r], k_value, dot);
            }
        }
        const float score = warp_sum(dot, FullMask) * attention_scale;
        const float new_m = fmaxf(running_m, score);
        const float alpha = running_m == -CUDART_INF_F
                                ? 0.0F
                                : exp2_approx((running_m - new_m) * Log2E);
        const float probability = exp2_approx((score - new_m) * Log2E);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            numerator[r] = __fmaf_rn(numerator[r], alpha, probability * v_values[r]);
        }
        running_l = running_l * alpha + probability;
        running_m = new_m;
    }

    const float inverse_l = running_l > 0.0F ? __frcp_rn(running_l) : 0.0F;
#pragma unroll
    for (float& value : numerator) { value *= inverse_l; }
    // H256 is normalized and self-inverse; return the weighted V sum to model coordinates.
    normalized_hadamard_d256_inplace(numerator, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        out[q_base + static_cast<std::int64_t>(D) * q_head + lane + Threads * r] =
            __float2bfloat16(numerator[r]);
    }
}

// Q2 row decoder specialized for the packed byte layout.  Each lane owns four
// symbols from byte lane and four symbols from byte lane+32, which covers the
// complete D=256 row with two coalesced byte loads per lane.  The query,
// numerator and inverse H256 transform use the matching packed-row layout;
// only the final stores map back to the model's natural dimension order.
template <typename Geometry, bool Masked>
__launch_bounds__(32) __global__ void causal_attention_oscar_q2_packed_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v,
    const __nv_bfloat16* __restrict__ cache_k_scale,
    const __nv_bfloat16* __restrict__ cache_v_scale, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride,
    std::int32_t logical_pages, std::int32_t width, float attention_scale,
    __nv_bfloat16* __restrict__ out) {
    constexpr int D             = kPagedKVCacheOscarHeadDim;
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
        for (int d = lane; d < D; d += 32) {
            out[causal_q_index<Geometry>(q_head, d, flat_column)] = __float2bfloat16(0.0F);
        }
    };
    if (token >= valid_tokens) {
        write_zero();
        return;
    }

    const int q_abs     = positions[flat_column];
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if (q_abs < 0 || q_abs >= logical_pages * kPagedKVPageSize || table_row < 0) {
        write_zero();
        return;
    }
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const int kv_head = q_head / Geometry::GroupSize;
    const std::int64_t q_base = static_cast<std::int64_t>(D) * Geometry::QHeads * flat_column;
    const __nv_bfloat16* q_row = q + q_base + static_cast<std::int64_t>(D) * q_head;
    __nv_bfloat16* out_row = out + q_base + static_cast<std::int64_t>(D) * q_head;

    float q_values[8];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        q_values[item]     = __bfloat162float(q_row[lane * 4 + item]);
        q_values[item + 4] = __bfloat162float(q_row[128 + lane * 4 + item]);
    }
    normalized_hadamard_d256_packed_inplace(q_values, lane);

    float numerator[8] = {};
    float running_m    = -CUDART_INF_F;
    float running_l    = 0.0F;
    for (int key = 0; key <= q_abs; ++key) {
        const int physical_page = block_table[key >> kPagedKVPageShift];
        const int page_offset   = key & kPagedKVPageMask;
        const std::uint8_t* k_codes =
            cache_k + paged_kv_page_head_offset<64, Geometry::KVHeads>(physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * 64;
        const std::uint8_t* v_codes =
            cache_v + paged_kv_page_head_offset<64, Geometry::KVHeads>(physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * 64;
        const __nv_bfloat16* k_metadata =
            cache_k_scale +
            paged_kv_page_head_offset<kPagedKVCacheOscarScaleExtent, Geometry::KVHeads>(
                physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * kPagedKVCacheOscarScaleExtent;
        const __nv_bfloat16* v_metadata =
            cache_v_scale +
            paged_kv_page_head_offset<kPagedKVCacheOscarScaleExtent, Geometry::KVHeads>(
                physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * kPagedKVCacheOscarScaleExtent;

        float k_scale = 0.0F;
        float k_zero  = 0.0F;
        float v_scale = 0.0F;
        float v_zero  = 0.0F;
        if (lane == 0) {
            k_scale = __bfloat162float(k_metadata[0]);
            k_zero  = __bfloat162float(k_metadata[1]);
            v_scale = __bfloat162float(v_metadata[0]);
            v_zero  = __bfloat162float(v_metadata[1]);
        }
        k_scale = __shfl_sync(FullMask, k_scale, 0);
        k_zero  = __shfl_sync(FullMask, k_zero, 0);
        v_scale = __shfl_sync(FullMask, v_scale, 0);
        v_zero  = __shfl_sync(FullMask, v_zero, 0);

        const int packed_k0 = static_cast<int>(k_codes[lane]);
        const int packed_k1 = static_cast<int>(k_codes[lane + 32]);
        const int packed_v0 = static_cast<int>(v_codes[lane]);
        const int packed_v1 = static_cast<int>(v_codes[lane + 32]);
        float dot            = 0.0F;
        float v_values[8];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int shift       = item << 1;
            const float k_value0  = fmaf(static_cast<float>((packed_k0 >> shift) & 3), k_scale,
                                        k_zero);
            const float k_value1  = fmaf(static_cast<float>((packed_k1 >> shift) & 3), k_scale,
                                        k_zero);
            v_values[item]        = fmaf(static_cast<float>((packed_v0 >> shift) & 3), v_scale,
                                         v_zero);
            v_values[item + 4]    = fmaf(static_cast<float>((packed_v1 >> shift) & 3), v_scale,
                                         v_zero);
            dot                   = __fmaf_rn(q_values[item], k_value0, dot);
            dot                   = __fmaf_rn(q_values[item + 4], k_value1, dot);
        }
        const float score = warp_sum(dot, FullMask) * attention_scale;
        const float new_m = fmaxf(running_m, score);
        const float alpha = running_m == -CUDART_INF_F
                                ? 0.0F
                                : exp2_approx((running_m - new_m) * Log2E);
        const float probability = exp2_approx((score - new_m) * Log2E);
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            numerator[item] = __fmaf_rn(numerator[item], alpha, probability * v_values[item]);
        }
        running_l = running_l * alpha + probability;
        running_m = new_m;
    }

    const float inverse_l = running_l > 0.0F ? __frcp_rn(running_l) : 0.0F;
#pragma unroll
    for (float& value : numerator) { value *= inverse_l; }
    normalized_hadamard_d256_packed_inplace(numerator, lane);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        out_row[lane * 4 + item]          = __float2bfloat16(numerator[item]);
        out_row[128 + lane * 4 + item]    = __float2bfloat16(numerator[item + 4]);
    }
}

// XQA-style OSCAR path. A CTA owns one KV head and all query heads that share it, so each packed
// Q2 K/V row is reconstructed once and reused by the grouped-query warps. The old one-warp path
// decoded the same row once per query head; that is especially expensive for the D=256 target
// cache and was the main reason the first strict-Q2 implementation fell below the BF16 baseline.
// Q2-specific XQA producer. Two warps load the 64-byte K/V row and expand
// four symbols per thread into natural D=256 shared rows. The generic XQA
// fallback assigns one thread to each natural dimension, re-reading every Q2
// byte four times before paying the same CTA barrier.
template <typename Geometry, bool Masked, int Split = 1>
__launch_bounds__((Geometry::GroupSize / Split) * 32, 2)
    __global__ void causal_attention_oscar_q2_xqa_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v,
    const __nv_bfloat16* __restrict__ cache_k_scale,
    const __nv_bfloat16* __restrict__ cache_v_scale, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride,
    std::int32_t logical_pages, std::int32_t width, float attention_scale,
    __nv_bfloat16* __restrict__ out) {
    static_assert(Split > 0 && Geometry::GroupSize % Split == 0);
    constexpr int D            = kPagedKVCacheOscarHeadDim;
    constexpr int CodeExtent   = kPagedKVCacheOscarCodeExtent;
    constexpr int ScaleExtent  = kPagedKVCacheOscarScaleExtent;
    constexpr int WarpThreads  = 32;
    constexpr int WarpsPerCTA  = Geometry::GroupSize / Split;
    constexpr int BlockThreads = WarpsPerCTA * WarpThreads;
    constexpr float Log2E      = 1.4426950408889634074F;
    constexpr unsigned FullMask = 0xffffffffU;

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

    const int q_abs     = positions[flat_column];
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if (q_abs < 0 || q_abs >= logical_pages * kPagedKVPageSize || table_row < 0) {
        write_zero();
        return;
    }
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const std::int64_t q_base = static_cast<std::int64_t>(D) * Geometry::QHeads * flat_column;

    float q_values[D / WarpThreads];
#pragma unroll
    for (int r = 0; r < D / WarpThreads; ++r) {
        q_values[r] = __bfloat162float(q[q_base + static_cast<std::int64_t>(D) * q_head +
                                           lane + WarpThreads * r]);
    }
    normalized_hadamard_d256_inplace(q_values, lane);

    __shared__ float shared_key[D];
    __shared__ float shared_value[D];
    __shared__ float shared_metadata[4];
    float numerator[D / WarpThreads] = {};
    float running_m                  = -CUDART_INF_F;
    float running_l                  = 0.0F;

    for (int key = 0; key <= q_abs; ++key) {
        const int physical_page = block_table[key >> kPagedKVPageShift];
        const int page_offset   = key & kPagedKVPageMask;
        const std::uint8_t* k_codes =
            cache_k + paged_kv_page_head_offset<CodeExtent, Geometry::KVHeads>(physical_page,
                                                                                  kv_head) +
            static_cast<std::int64_t>(page_offset) * CodeExtent;
        const std::uint8_t* v_codes =
            cache_v + paged_kv_page_head_offset<CodeExtent, Geometry::KVHeads>(physical_page,
                                                                                  kv_head) +
            static_cast<std::int64_t>(page_offset) * CodeExtent;
        const __nv_bfloat16* k_metadata =
            cache_k_scale +
            paged_kv_page_head_offset<ScaleExtent, Geometry::KVHeads>(physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * ScaleExtent;
        const __nv_bfloat16* v_metadata =
            cache_v_scale +
            paged_kv_page_head_offset<ScaleExtent, Geometry::KVHeads>(physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * ScaleExtent;

        if (tid == 0) {
            shared_metadata[0] = __bfloat162float(k_metadata[0]);
            shared_metadata[1] = __bfloat162float(k_metadata[1]);
            shared_metadata[2] = __bfloat162float(v_metadata[0]);
            shared_metadata[3] = __bfloat162float(v_metadata[1]);
        }
        __syncthreads();

        if (tid < CodeExtent) {
            const int packed_k = static_cast<int>(k_codes[tid]);
            const int packed_v = static_cast<int>(v_codes[tid]);
            const int d0        = tid * 4;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int shift = item << 1;
                shared_key[d0 + item] =
                    fmaf(static_cast<float>((packed_k >> shift) & 3), shared_metadata[0],
                         shared_metadata[1]);
                shared_value[d0 + item] =
                    fmaf(static_cast<float>((packed_v >> shift) & 3), shared_metadata[2],
                         shared_metadata[3]);
            }
        }
        __syncthreads();

        float dot = 0.0F;
#pragma unroll
        for (int r = 0; r < D / WarpThreads; ++r) {
            dot = __fmaf_rn(q_values[r], shared_key[lane + WarpThreads * r], dot);
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
                                     probability * shared_value[lane + WarpThreads * r]);
        }
        running_l = running_l * alpha + probability;
        running_m = new_m;
    }

    const float inverse_l = running_l > 0.0F ? __frcp_rn(running_l) : 0.0F;
#pragma unroll
    for (float& value : numerator) { value *= inverse_l; }
    normalized_hadamard_d256_inplace(numerator, lane);
#pragma unroll
    for (int r = 0; r < D / WarpThreads; ++r) {
        out[q_base + static_cast<std::int64_t>(D) * q_head + lane + WarpThreads * r] =
            __float2bfloat16(numerator[r]);
    }
}

template <typename Geometry, int Bits, bool Masked, int Split = 1>
__launch_bounds__((Geometry::GroupSize / Split) * 32, 2)
    __global__ void causal_attention_oscar_xqa_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v,
    const __nv_bfloat16* __restrict__ cache_k_scale,
    const __nv_bfloat16* __restrict__ cache_v_scale, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ block_tables, std::int32_t table_stride,
    std::int32_t logical_pages, std::int32_t width, float attention_scale,
    __nv_bfloat16* __restrict__ out) {
    static_assert(Bits == 2 || Bits == 4);
    static_assert(Split > 0 && Geometry::GroupSize % Split == 0);
    constexpr int D            = kPagedKVCacheOscarHeadDim;
    constexpr int WarpThreads  = 32;
    constexpr int WarpsPerCTA  = Geometry::GroupSize / Split;
    constexpr int BlockThreads = WarpsPerCTA * WarpThreads;
    constexpr int CodeExtent   = cyclic_oscar_code_extent<Bits>(D);
    constexpr int ScaleExtent  = kPagedKVCacheOscarScaleExtent;
    constexpr float Log2E      = 1.4426950408889634074F;
    constexpr unsigned FullMask = 0xffffffffU;

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

    const int q_abs     = positions[flat_column];
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    if (q_abs < 0 || q_abs >= logical_pages * kPagedKVPageSize || table_row < 0) {
        write_zero();
        return;
    }
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const std::int64_t q_base = static_cast<std::int64_t>(D) * Geometry::QHeads * flat_column;

    float q_values[D / WarpThreads];
#pragma unroll
    for (int r = 0; r < D / WarpThreads; ++r) {
        q_values[r] = __bfloat162float(q[q_base + static_cast<std::int64_t>(D) * q_head +
                                           lane + WarpThreads * r]);
    }
    normalized_hadamard_d256_inplace(q_values, lane);

    __shared__ float shared_key[2][D];
    __shared__ float shared_value[2][D];
    __shared__ float shared_k_metadata[2][ScaleExtent];
    __shared__ float shared_v_metadata[2][ScaleExtent];
    float numerator[D / WarpThreads] = {};
    float running_m                  = -CUDART_INF_F;
    float running_l                  = 0.0F;
    for (int key = 0; key <= q_abs; ++key) {
        const int physical_page = block_table[key >> kPagedKVPageShift];
        const int page_offset   = key & kPagedKVPageMask;
        const std::uint8_t* k_codes =
            cache_k + paged_kv_page_head_offset<CodeExtent, Geometry::KVHeads>(physical_page,
                                                                                  kv_head) +
            static_cast<std::int64_t>(page_offset) * CodeExtent;
        const std::uint8_t* v_codes =
            cache_v + paged_kv_page_head_offset<CodeExtent, Geometry::KVHeads>(physical_page,
                                                                                  kv_head) +
            static_cast<std::int64_t>(page_offset) * CodeExtent;
        const __nv_bfloat16* k_metadata =
            cache_k_scale +
            paged_kv_page_head_offset<ScaleExtent, Geometry::KVHeads>(physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * ScaleExtent;
        const __nv_bfloat16* v_metadata =
            cache_v_scale +
            paged_kv_page_head_offset<ScaleExtent, Geometry::KVHeads>(physical_page, kv_head) +
            static_cast<std::int64_t>(page_offset) * ScaleExtent;

        const int buffer = key & 1;
        if (tid < ScaleExtent) {
            shared_k_metadata[buffer][tid] = __bfloat162float(k_metadata[tid]);
            shared_v_metadata[buffer][tid] = __bfloat162float(v_metadata[tid]);
        }
        __syncthreads();
        for (int d = tid; d < D; d += BlockThreads) {
            shared_key[buffer][d] =
                fmaf(static_cast<float>(cyclic_oscar_unpack_code<Bits>(k_codes, d)),
                     shared_k_metadata[buffer][0], shared_k_metadata[buffer][1]);
            shared_value[buffer][d] =
                fmaf(static_cast<float>(cyclic_oscar_unpack_code<Bits>(v_codes, d)),
                     shared_v_metadata[buffer][0], shared_v_metadata[buffer][1]);
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
        numerator[r] *= inverse_l;
    }
    normalized_hadamard_d256_inplace(numerator, lane);
#pragma unroll
    for (int r = 0; r < D / WarpThreads; ++r) {
        out[q_base + static_cast<std::int64_t>(D) * q_head + lane + WarpThreads * r] =
            __float2bfloat16(numerator[r]);
    }
}

} // namespace ninfer::ops
