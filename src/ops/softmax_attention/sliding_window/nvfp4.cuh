#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

template <bool Packed, bool PairLayout = false>
__device__ __forceinline__ void sliding_window_nvfp4_accumulate(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const __nv_bfloat16* key_raw, const __nv_bfloat16* value_raw,
    const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const std::uint8_t* key_scales, const std::uint8_t* value_scales) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr float Log2E       = 1.4426950408889634074F;

    if constexpr (PairLayout) {
        const int d0 = lane * 2;
        const int d1 = d0 + 64;
        float key_values[4];
        float value_values[4];
        if constexpr (Packed) {
            float key_scale0   = 0.0F;
            float key_scale1   = 0.0F;
            float value_scale0 = 0.0F;
            float value_scale1 = 0.0F;
            if ((lane & 7) == 0) {
                const int group = lane >> 3;
                key_scale0     = kv_cache_nvfp4_decode_scale(key_scales[group]);
                key_scale1     = kv_cache_nvfp4_decode_scale(key_scales[group + 4]);
                value_scale0   = kv_cache_nvfp4_decode_scale(value_scales[group]);
                value_scale1   = kv_cache_nvfp4_decode_scale(value_scales[group + 4]);
            }
            const int scale_lane = (lane >> 3) * 8;
            key_scale0 = __shfl_sync(FullMask, key_scale0, scale_lane);
            key_scale1 = __shfl_sync(FullMask, key_scale1, scale_lane);
            value_scale0 = __shfl_sync(FullMask, value_scale0, scale_lane);
            value_scale1 = __shfl_sync(FullMask, value_scale1, scale_lane);

            const float2 key_pair0   = kv_cache_nvfp4_decode_pair(key_codes[lane]);
            const float2 key_pair1   = kv_cache_nvfp4_decode_pair(key_codes[lane + 32]);
            const float2 value_pair0 = kv_cache_nvfp4_decode_pair(value_codes[lane]);
            const float2 value_pair1 = kv_cache_nvfp4_decode_pair(value_codes[lane + 32]);
            key_values[0] = key_pair0.x * key_scale0;
            key_values[1] = key_pair0.y * key_scale0;
            key_values[2] = key_pair1.x * key_scale1;
            key_values[3] = key_pair1.y * key_scale1;
            value_values[0] = value_pair0.x * value_scale0;
            value_values[1] = value_pair0.y * value_scale0;
            value_values[2] = value_pair1.x * value_scale1;
            value_values[3] = value_pair1.y * value_scale1;
        } else {
            key_values[0] = __bfloat162float(key_raw[d0]);
            key_values[1] = __bfloat162float(key_raw[d0 + 1]);
            key_values[2] = __bfloat162float(key_raw[d1]);
            key_values[3] = __bfloat162float(key_raw[d1 + 1]);
            value_values[0] = __bfloat162float(value_raw[d0]);
            value_values[1] = __bfloat162float(value_raw[d0 + 1]);
            value_values[2] = __bfloat162float(value_raw[d1]);
            value_values[3] = __bfloat162float(value_raw[d1 + 1]);
        }
        float dot = 0.0F;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            dot = fmaf(q_values[item], key_values[item], dot);
        }
        dot = warp_sum<32>(dot, FullMask) * scale;
        const float next_max = fmaxf(running_max, dot);
        const float old_weight = running_max == -CUDART_INF_F
                                      ? 0.0F
                                      : exp2_approx((running_max - next_max) * Log2E);
        const float new_weight = exp2_approx((dot - next_max) * Log2E);
        normalizer             = normalizer * old_weight + new_weight;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            accumulator[item] = accumulator[item] * old_weight + value_values[item] * new_weight;
        }
        running_max = next_max;
        return;
    }

    float dot = 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * 32;
        const float key_value = [&]() {
            if constexpr (Packed) {
                return kv_cache_nvfp4_decode_value(key_codes, key_scales, d);
            }
            return __bfloat162float(key_raw[d]);
        }();
        dot = fmaf(q_values[item], key_value, dot);
    }
    dot = warp_sum<32>(dot, FullMask) * scale;
    const float next_max = fmaxf(running_max, dot);
    const float old_weight = running_max == -CUDART_INF_F
                                  ? 0.0F
                                  : exp2_approx((running_max - next_max) * Log2E);
    const float new_weight = exp2_approx((dot - next_max) * Log2E);
    normalizer             = normalizer * old_weight + new_weight;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * 32;
        const float value = [&]() {
            if constexpr (Packed) {
                return kv_cache_nvfp4_decode_value(value_codes, value_scales, d);
            }
            return __bfloat162float(value_raw[d]);
        }();
        accumulator[item] = accumulator[item] * old_weight + value * new_weight;
    }
    running_max = next_max;
}

// Pair-decode while retaining the scalar fallback's dimension-to-lane order.  The packed bytes
// are decoded by their owning lane and then transposed with warp shuffles before the dot/value
// update.  This costs a few shuffles, but keeps greedy decisions stable across the optimized and
// fallback routes instead of turning a harmless floating-point reordering into a draft mismatch.
template <bool Packed>
__device__ __forceinline__ void sliding_window_nvfp4_accumulate_pair_stable(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const __nv_bfloat16* key_raw, const __nv_bfloat16* value_raw,
    const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const std::uint8_t* key_scales, const std::uint8_t* value_scales) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr float Log2E       = 1.4426950408889634074F;
    const int d0                = lane * 2;
    const int d1                = d0 + 64;
    float key_values[8];
    float value_values[8];
    if constexpr (Packed) {
        float key_scale0   = 0.0F;
        float key_scale1   = 0.0F;
        float value_scale0 = 0.0F;
        float value_scale1 = 0.0F;
        if ((lane & 7) == 0) {
            const int group = lane >> 3;
            key_scale0     = kv_cache_nvfp4_decode_scale(key_scales[group]);
            key_scale1     = kv_cache_nvfp4_decode_scale(key_scales[group + 4]);
            value_scale0   = kv_cache_nvfp4_decode_scale(value_scales[group]);
            value_scale1   = kv_cache_nvfp4_decode_scale(value_scales[group + 4]);
        }
        const int scale_lane = (lane >> 3) * 8;
        key_scale0           = __shfl_sync(FullMask, key_scale0, scale_lane);
        key_scale1           = __shfl_sync(FullMask, key_scale1, scale_lane);
        value_scale0         = __shfl_sync(FullMask, value_scale0, scale_lane);
        value_scale1         = __shfl_sync(FullMask, value_scale1, scale_lane);
        const float2 key_pair0   = kv_cache_nvfp4_decode_pair(key_codes[lane]);
        const float2 key_pair1   = kv_cache_nvfp4_decode_pair(key_codes[lane + 32]);
        const float2 value_pair0 = kv_cache_nvfp4_decode_pair(value_codes[lane]);
        const float2 value_pair1 = kv_cache_nvfp4_decode_pair(value_codes[lane + 32]);
        key_values[0] = key_pair0.x * key_scale0;
        key_values[1] = key_pair0.y * key_scale0;
        key_values[2] = key_pair1.x * key_scale1;
        key_values[3] = key_pair1.y * key_scale1;
        value_values[0] = value_pair0.x * value_scale0;
        value_values[1] = value_pair0.y * value_scale0;
        value_values[2] = value_pair1.x * value_scale1;
        value_values[3] = value_pair1.y * value_scale1;
    } else {
        key_values[0] = __bfloat162float(key_raw[d0]);
        key_values[1] = __bfloat162float(key_raw[d0 + 1]);
        key_values[2] = __bfloat162float(key_raw[d1]);
        key_values[3] = __bfloat162float(key_raw[d1 + 1]);
        value_values[0] = __bfloat162float(value_raw[d0]);
        value_values[1] = __bfloat162float(value_raw[d0 + 1]);
        value_values[2] = __bfloat162float(value_raw[d1]);
        value_values[3] = __bfloat162float(value_raw[d1 + 1]);
    }

    // The raw fallback path has the same two-segment pair ownership as the packed path.  For each
    // original scalar lane/item, select the owner lane and its corresponding pair component.
    float dot = 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d       = lane + item * 32;
        const int owner   = (d & 63) >> 1;
        const int segment = d >> 6;
        const float key_component0 =
            __shfl_sync(FullMask, key_values[segment * 2], owner);
        const float key_component1 =
            __shfl_sync(FullMask, key_values[segment * 2 + 1], owner);
        const float key_value = (d & 1) == 0 ? key_component0 : key_component1;
        dot                   = fmaf(q_values[item], key_value, dot);
    }
    dot = warp_sum<32>(dot, FullMask) * scale;
    const float next_max = fmaxf(running_max, dot);
    const float old_weight = running_max == -CUDART_INF_F
                                  ? 0.0F
                                  : exp2_approx((running_max - next_max) * Log2E);
    const float new_weight = exp2_approx((dot - next_max) * Log2E);
    normalizer             = normalizer * old_weight + new_weight;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d       = lane + item * 32;
        const int owner   = (d & 63) >> 1;
        const int segment = d >> 6;
        const float value_component0 =
            __shfl_sync(FullMask, value_values[segment * 2], owner);
        const float value_component1 =
            __shfl_sync(FullMask, value_values[segment * 2 + 1], owner);
        const float value = (d & 1) == 0 ? value_component0 : value_component1;
        accumulator[item] = accumulator[item] * old_weight + value * new_weight;
    }
    running_max = next_max;
}

template <bool Packed, bool PairLayout, bool StablePair>
__device__ __forceinline__ void sliding_window_nvfp4_accumulate_dispatch(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const __nv_bfloat16* key_raw, const __nv_bfloat16* value_raw,
    const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const std::uint8_t* key_scales, const std::uint8_t* value_scales) {
    if constexpr (StablePair) {
        sliding_window_nvfp4_accumulate_pair_stable<Packed>(
            lane, q_values, scale, normalizer, running_max, accumulator, key_raw, value_raw,
            key_codes, value_codes, key_scales, value_scales);
    } else {
        sliding_window_nvfp4_accumulate<Packed, PairLayout>(
            lane, q_values, scale, normalizer, running_max, accumulator, key_raw, value_raw,
            key_codes, value_codes, key_scales, value_scales);
    }
}

template <bool Packed>
__device__ __forceinline__ void sliding_window_nvfp4_load_shared(
    int lane, const __nv_bfloat16* key_raw, const __nv_bfloat16* value_raw,
    const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const std::uint8_t* key_scales, const std::uint8_t* value_scales, float* shared_key,
    float* shared_value) {
    const int d0 = lane * 2;
    const int d1 = d0 + 64;
    if constexpr (Packed) {
        constexpr unsigned FullMask = 0xffffffffU;
        const int group = lane >> 3;
        float key_scale0   = 0.0F;
        float key_scale1   = 0.0F;
        float value_scale0 = 0.0F;
        float value_scale1 = 0.0F;
        if (lane < 4) {
            key_scale0   = kv_cache_nvfp4_decode_scale(key_scales[lane]);
            value_scale0 = kv_cache_nvfp4_decode_scale(value_scales[lane]);
        } else if (lane < 8) {
            key_scale1   = kv_cache_nvfp4_decode_scale(key_scales[lane]);
            value_scale1 = kv_cache_nvfp4_decode_scale(value_scales[lane]);
        }
        key_scale0   = __shfl_sync(FullMask, key_scale0, group);
        key_scale1   = __shfl_sync(FullMask, key_scale1, group + 4);
        value_scale0 = __shfl_sync(FullMask, value_scale0, group);
        value_scale1 = __shfl_sync(FullMask, value_scale1, group + 4);

        const float2 key_pair0   = kv_cache_nvfp4_decode_pair(key_codes[lane]);
        const float2 key_pair1   = kv_cache_nvfp4_decode_pair(key_codes[lane + 32]);
        const float2 value_pair0 = kv_cache_nvfp4_decode_pair(value_codes[lane]);
        const float2 value_pair1 = kv_cache_nvfp4_decode_pair(value_codes[lane + 32]);
        shared_key[d0]       = key_pair0.x * key_scale0;
        shared_key[d0 + 1]   = key_pair0.y * key_scale0;
        shared_key[d1]       = key_pair1.x * key_scale1;
        shared_key[d1 + 1]   = key_pair1.y * key_scale1;
        shared_value[d0]     = value_pair0.x * value_scale0;
        shared_value[d0 + 1] = value_pair0.y * value_scale0;
        shared_value[d1]     = value_pair1.x * value_scale1;
        shared_value[d1 + 1] = value_pair1.y * value_scale1;
    } else {
        shared_key[d0]       = __bfloat162float(key_raw[d0]);
        shared_key[d0 + 1]   = __bfloat162float(key_raw[d0 + 1]);
        shared_key[d1]       = __bfloat162float(key_raw[d1]);
        shared_key[d1 + 1]   = __bfloat162float(key_raw[d1 + 1]);
        shared_value[d0]     = __bfloat162float(value_raw[d0]);
        shared_value[d0 + 1] = __bfloat162float(value_raw[d0 + 1]);
        shared_value[d1]     = __bfloat162float(value_raw[d1]);
        shared_value[d1 + 1] = __bfloat162float(value_raw[d1 + 1]);
    }
}

__device__ __forceinline__ void sliding_window_nvfp4_accumulate_shared(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const float* shared_key, const float* shared_value) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr float Log2E        = 1.4426950408889634074F;
    float dot                     = 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        dot = fmaf(q_values[item], shared_key[lane + item * 32], dot);
    }
    dot = warp_sum<32>(dot, FullMask) * scale;
    const float next_max = fmaxf(running_max, dot);
    const float old_weight = running_max == -CUDART_INF_F
                                  ? 0.0F
                                  : exp2_approx((running_max - next_max) * Log2E);
    const float new_weight = exp2_approx((dot - next_max) * Log2E);
    normalizer = normalizer * old_weight + new_weight;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        accumulator[item] = accumulator[item] * old_weight +
                            shared_value[lane + item * 32] * new_weight;
    }
    running_max = next_max;
}

// One block owns the four query heads mapped to one KV head.  The compressed K/V row is decoded
// once by warp 0 and consumed by all four query warps.  This removes the four-way KV byte/decode
// duplication of the one-warp-per-query fallback while retaining one independent online-softmax
// state per query head.
template <bool HasProtected>
__launch_bounds__(128, 2) __global__ void sliding_window_attention_nvfp4_grouped_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const std::uint8_t* __restrict__ context_k, const std::uint8_t* __restrict__ context_v,
    const std::uint8_t* __restrict__ context_k_scale,
    const std::uint8_t* __restrict__ context_v_scale,
    const __nv_bfloat16* __restrict__ protected_k,
    const __nv_bfloat16* __restrict__ protected_v, int padded_context, int max_context,
    int window, int protected_capacity, int protected_anchor_capacity,
    int protected_padded_capacity, int tokens, float scale, __nv_bfloat16* __restrict__ out) {
    constexpr int D          = kCyclicKVCacheNvfp4HeadDim;
    constexpr int QHeads     = 32;
    constexpr int KVHeads    = kCyclicKVCacheNvfp4KVHeads;
    constexpr int CodeExtent = kCyclicKVCacheNvfp4CodeExtent;
    constexpr int ScaleExtent = kCyclicKVCacheNvfp4ScaleExtent;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int group      = static_cast<int>(blockIdx.x) % KVHeads;
    const int token      = static_cast<int>(blockIdx.x) / KVHeads;
    const int q_head     = group * 4 + warp;
    const int batch      = static_cast<int>(blockIdx.y);
    if (token >= tokens || warp >= 4) return;

    const std::int64_t q_batch_stride = static_cast<std::int64_t>(D) * QHeads * tokens;
    const std::int64_t kv_batch_stride = static_cast<std::int64_t>(D) * KVHeads * tokens;
    const __nv_bfloat16* q_row = q + q_batch_stride * batch +
                                 static_cast<std::int64_t>(D) * (q_head + QHeads * token);
    __nv_bfloat16* out_row = out + q_batch_stride * batch +
                             static_cast<std::int64_t>(D) * (q_head + QHeads * token);

    const int length = positions[static_cast<std::int64_t>(tokens) * batch];
    const int valid  = valid_columns[batch];
    if (token >= valid || valid < 0 || valid > tokens || length < 0 || length > max_context) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            out_row[lane + item * 32] = __float2bfloat16(0.0F);
        }
        return;
    }

    float q_values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        q_values[item] = __bfloat162float(q_row[lane + item * 32]);
    }

    const int context_count = min(length, window - 1);
    const int context_start = length - context_count;
    const std::int64_t code_lane_stride =
        static_cast<std::int64_t>(CodeExtent) * padded_context * KVHeads;
    const std::int64_t scale_lane_stride =
        static_cast<std::int64_t>(ScaleExtent) * padded_context * KVHeads;
    const int state_lane = lanes[batch];
    const std::uint8_t* lane_k = context_k + code_lane_stride * state_lane;
    const std::uint8_t* lane_v = context_v + code_lane_stride * state_lane;
    const std::uint8_t* lane_k_scale = context_k_scale + scale_lane_stride * state_lane;
    const std::uint8_t* lane_v_scale = context_v_scale + scale_lane_stride * state_lane;
    const __nv_bfloat16* lane_protected_k = nullptr;
    const __nv_bfloat16* lane_protected_v = nullptr;
    if constexpr (HasProtected) {
        const std::int64_t protected_lane_stride =
            static_cast<std::int64_t>(D) * protected_padded_capacity * KVHeads;
        lane_protected_k = protected_k + protected_lane_stride * state_lane;
        lane_protected_v = protected_v + protected_lane_stride * state_lane;
    }
    const __nv_bfloat16* query_k_batch = query_k + kv_batch_stride * batch;
    const __nv_bfloat16* query_v_batch = query_v + kv_batch_stride * batch;

    __shared__ float shared_key[D];
    __shared__ float shared_value[D];
    __shared__ float shared_query_key[16 * D];
    __shared__ float shared_query_value[16 * D];
    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float normalizer      = 0.0F;
    float running_max     = -CUDART_INF_F;
    if constexpr (HasProtected) {
        const int anchor_end   = min(length, protected_anchor_capacity);
        const int recent_begin = protected_capacity == 0 ? length : max(0, length - protected_capacity);
        for (int context_position = context_start;
             context_position < min(anchor_end, length); ++context_position) {
            const std::int64_t protected_offset =
                static_cast<std::int64_t>(D) *
                (context_position + static_cast<std::int64_t>(protected_padded_capacity) * group);
            if (warp == 0) {
                sliding_window_nvfp4_load_shared<false>(
                    lane, lane_protected_k + protected_offset, lane_protected_v + protected_offset,
                    nullptr, nullptr, nullptr, nullptr, shared_key, shared_value);
            }
            __syncthreads();
            sliding_window_nvfp4_accumulate_shared(
                lane, q_values, scale, normalizer, running_max, accumulator, shared_key,
                shared_value);
        }
        const int packed_begin = max(context_start, anchor_end);
        const int packed_end   = min(length, recent_begin);
        for (int context_position = packed_begin; context_position < packed_end;
             ++context_position) {
            const int slot = context_position & (window - 1);
            const std::int64_t slot_offset =
                static_cast<std::int64_t>(CodeExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * group);
            const std::int64_t scale_offset =
                static_cast<std::int64_t>(ScaleExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * group);
            if (warp == 0) {
                sliding_window_nvfp4_load_shared<true>(
                    lane, nullptr, nullptr, lane_k + slot_offset, lane_v + slot_offset,
                    lane_k_scale + scale_offset, lane_v_scale + scale_offset, shared_key,
                    shared_value);
            }
            __syncthreads();
            sliding_window_nvfp4_accumulate_shared(
                lane, q_values, scale, normalizer, running_max, accumulator, shared_key,
                shared_value);
        }
        if (protected_capacity != 0) {
            const int recent_begin_in_window = max(context_start, max(recent_begin, anchor_end));
            for (int context_position = recent_begin_in_window; context_position < length;
                 ++context_position) {
                const int protected_slot =
                    protected_anchor_capacity + (context_position & (protected_capacity - 1));
                const std::int64_t protected_offset =
                    static_cast<std::int64_t>(D) *
                    (protected_slot + static_cast<std::int64_t>(protected_padded_capacity) * group);
                if (warp == 0) {
                    sliding_window_nvfp4_load_shared<false>(
                        lane, lane_protected_k + protected_offset,
                        lane_protected_v + protected_offset, nullptr, nullptr, nullptr, nullptr,
                        shared_key, shared_value);
                }
                __syncthreads();
                sliding_window_nvfp4_accumulate_shared(
                    lane, q_values, scale, normalizer, running_max, accumulator, shared_key,
                    shared_value);
            }
        }
    } else {
        for (int key = 0; key < context_count; ++key) {
            const int context_position = context_start + key;
            const int slot             = context_position & (window - 1);
            const std::int64_t slot_offset =
                static_cast<std::int64_t>(CodeExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * group);
            const std::int64_t scale_offset =
                static_cast<std::int64_t>(ScaleExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * group);
            if (warp == 0) {
                sliding_window_nvfp4_load_shared<true>(
                    lane, nullptr, nullptr, lane_k + slot_offset, lane_v + slot_offset,
                    lane_k_scale + scale_offset, lane_v_scale + scale_offset, shared_key,
                    shared_value);
            }
            __syncthreads();
            sliding_window_nvfp4_accumulate_shared(
                lane, q_values, scale, normalizer, running_max, accumulator, shared_key,
                shared_value);
        }
    }
    // Query K/V rows are common to the four query heads in this KV-head group.  Stage the small
    // BF16 query block once so the four online-softmax warps do not repeat the same global reads.
    for (int query_token = 0; query_token < valid; ++query_token) {
        const __nv_bfloat16* query_key_row =
            query_k_batch + static_cast<std::int64_t>(D) * (group + KVHeads * query_token);
        const __nv_bfloat16* query_value_row =
            query_v_batch + static_cast<std::int64_t>(D) * (group + KVHeads * query_token);
        shared_query_key[query_token * D + threadIdx.x] =
            __bfloat162float(query_key_row[threadIdx.x]);
        shared_query_value[query_token * D + threadIdx.x] =
            __bfloat162float(query_value_row[threadIdx.x]);
    }
    __syncthreads();
    for (int query_token = 0; query_token < valid; ++query_token) {
        sliding_window_nvfp4_accumulate_shared(
            lane, q_values, scale, normalizer, running_max, accumulator,
            shared_query_key + query_token * D, shared_query_value + query_token * D);
    }

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const float value = normalizer > 0.0F ? accumulator[item] * __frcp_rn(normalizer) : 0.0F;
        out_row[lane + item * 32] = __float2bfloat16(value);
    }
}

// Correctness-first NVFP4 DFlash2 local attention.  One warp owns one query head/token row;
// this keeps the packed-cache path independent of the BF16 tensor-core staging contract.  The
// verifier still consumes the exact target cache, so this route is only an approximate proposer.
template <bool HasProtected, bool PairLayout = false, bool StablePair = false>
__launch_bounds__(32, 4) __global__ void sliding_window_attention_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const std::uint8_t* __restrict__ context_k, const std::uint8_t* __restrict__ context_v,
    const std::uint8_t* __restrict__ context_k_scale,
    const std::uint8_t* __restrict__ context_v_scale,
    const __nv_bfloat16* __restrict__ protected_k,
    const __nv_bfloat16* __restrict__ protected_v, int padded_context, int max_context,
    int window, int protected_capacity, int protected_anchor_capacity,
    int protected_padded_capacity, int tokens, float scale, __nv_bfloat16* __restrict__ out) {
    constexpr int D              = kCyclicKVCacheNvfp4HeadDim;
    constexpr int QHeads         = 32;
    constexpr int KVHeads        = kCyclicKVCacheNvfp4KVHeads;
    constexpr int CodeExtent     = kCyclicKVCacheNvfp4CodeExtent;
    constexpr int ScaleExtent    = kCyclicKVCacheNvfp4ScaleExtent;

    const int lane       = static_cast<int>(threadIdx.x);
    const int row        = static_cast<int>(blockIdx.x);
    const int token      = row / QHeads;
    const int q_head     = row - token * QHeads;
    const int batch      = static_cast<int>(blockIdx.y);
    if (lane >= 32 || token >= tokens) return;

    const std::int64_t q_batch_stride = static_cast<std::int64_t>(D) * QHeads * tokens;
    const std::int64_t kv_batch_stride = static_cast<std::int64_t>(D) * KVHeads * tokens;
    const __nv_bfloat16* q_row = q + q_batch_stride * batch +
                                 static_cast<std::int64_t>(D) * (q_head + QHeads * token);
    __nv_bfloat16* out_row = out + q_batch_stride * batch +
                             static_cast<std::int64_t>(D) * (q_head + QHeads * token);

    const int length = positions[static_cast<std::int64_t>(tokens) * batch];
    const int valid  = valid_columns[batch];
    if (token >= valid || valid < 0 || valid > tokens || length < 0 || length > max_context) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            out_row[lane + item * 32] = __float2bfloat16(0.0F);
        }
        return;
    }

    float q_values[4];
    if constexpr (PairLayout && !StablePair) {
        const int d0 = lane * 2;
        const int d1 = d0 + 64;
        q_values[0] = __bfloat162float(q_row[d0]);
        q_values[1] = __bfloat162float(q_row[d0 + 1]);
        q_values[2] = __bfloat162float(q_row[d1]);
        q_values[3] = __bfloat162float(q_row[d1 + 1]);
    } else {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            q_values[item] = __bfloat162float(q_row[lane + item * 32]);
        }
    }

    const int context_count = min(length, window - 1);
    const int context_start = length - context_count;
    const int kv_head       = q_head >> 2;
    const int state_lane    = lanes[batch];
    const std::int64_t code_lane_stride =
        static_cast<std::int64_t>(CodeExtent) * padded_context * KVHeads;
    const std::int64_t scale_lane_stride =
        static_cast<std::int64_t>(ScaleExtent) * padded_context * KVHeads;
    const std::uint8_t* lane_k = context_k + code_lane_stride * state_lane;
    const std::uint8_t* lane_v = context_v + code_lane_stride * state_lane;
    const std::uint8_t* lane_k_scale = context_k_scale + scale_lane_stride * state_lane;
    const std::uint8_t* lane_v_scale = context_v_scale + scale_lane_stride * state_lane;
    const __nv_bfloat16* lane_protected_k = nullptr;
    const __nv_bfloat16* lane_protected_v = nullptr;
    if constexpr (HasProtected) {
        const std::int64_t protected_lane_stride =
            static_cast<std::int64_t>(D) * protected_padded_capacity * KVHeads;
        lane_protected_k = protected_k + protected_lane_stride * state_lane;
        lane_protected_v = protected_v + protected_lane_stride * state_lane;
    }
    const __nv_bfloat16* query_k_batch = query_k + kv_batch_stride * batch;
    const __nv_bfloat16* query_v_batch = query_v + kv_batch_stride * batch;

    float accumulator[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float normalizer      = 0.0F;
    float running_max     = -CUDART_INF_F;
    if constexpr (HasProtected) {
        // The sidecar has two disjoint regions: fixed prefix anchors followed by the recent
        // cyclic ring.  Partition the active window explicitly so an anchor that is still inside
        // the window is never accidentally consumed from its quantized overwrite slot.
        const int anchor_end  = min(length, protected_anchor_capacity);
        const int recent_begin = protected_capacity == 0 ? length : max(0, length - protected_capacity);
        for (int context_position = context_start;
             context_position < min(anchor_end, length); ++context_position) {
            const std::int64_t protected_offset =
                static_cast<std::int64_t>(D) *
                (context_position +
                 static_cast<std::int64_t>(protected_padded_capacity) * kv_head);
            sliding_window_nvfp4_accumulate_dispatch<false, PairLayout, StablePair>(
                lane, q_values, scale, normalizer, running_max, accumulator,
                lane_protected_k + protected_offset, lane_protected_v + protected_offset, nullptr,
                nullptr, nullptr, nullptr);
        }
        const int packed_begin = max(context_start, anchor_end);
        const int packed_end   = min(length, recent_begin);
        for (int context_position = packed_begin; context_position < packed_end;
             ++context_position) {
            const int slot = context_position & (window - 1);
            const std::int64_t slot_offset =
                static_cast<std::int64_t>(CodeExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * kv_head);
            const std::int64_t scale_offset =
                static_cast<std::int64_t>(ScaleExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * kv_head);
            sliding_window_nvfp4_accumulate_dispatch<true, PairLayout, StablePair>(
                lane, q_values, scale, normalizer, running_max, accumulator, nullptr, nullptr,
                lane_k + slot_offset, lane_v + slot_offset, lane_k_scale + scale_offset,
                lane_v_scale + scale_offset);
        }
        if (protected_capacity != 0) {
            const int recent_begin_in_window = max(context_start, max(recent_begin, anchor_end));
            for (int context_position = recent_begin_in_window; context_position < length;
                 ++context_position) {
                const int protected_slot =
                    protected_anchor_capacity + (context_position & (protected_capacity - 1));
                const std::int64_t protected_offset =
                    static_cast<std::int64_t>(D) *
                    (protected_slot +
                     static_cast<std::int64_t>(protected_padded_capacity) * kv_head);
                sliding_window_nvfp4_accumulate_dispatch<false, PairLayout, StablePair>(
                    lane, q_values, scale, normalizer, running_max, accumulator,
                    lane_protected_k + protected_offset, lane_protected_v + protected_offset,
                    nullptr, nullptr, nullptr, nullptr);
            }
        }
    } else {
        for (int key = 0; key < context_count; ++key) {
            const int context_position = context_start + key;
            const int slot             = context_position & (window - 1);
            const std::int64_t slot_offset =
                static_cast<std::int64_t>(CodeExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * kv_head);
            const std::int64_t scale_offset =
                static_cast<std::int64_t>(ScaleExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * kv_head);
            sliding_window_nvfp4_accumulate_dispatch<true, PairLayout, StablePair>(
                lane, q_values, scale, normalizer, running_max, accumulator, nullptr, nullptr,
                lane_k + slot_offset, lane_v + slot_offset, lane_k_scale + scale_offset,
                lane_v_scale + scale_offset);
        }
    }
    for (int query_token = 0; query_token < valid; ++query_token) {
        const __nv_bfloat16* query_key_row =
            query_k_batch + static_cast<std::int64_t>(D) * (kv_head + KVHeads * query_token);
        const __nv_bfloat16* query_value_row =
            query_v_batch + static_cast<std::int64_t>(D) * (kv_head + KVHeads * query_token);
        sliding_window_nvfp4_accumulate_dispatch<false, PairLayout, StablePair>(
            lane, q_values, scale, normalizer, running_max, accumulator, query_key_row,
            query_value_row, nullptr, nullptr, nullptr, nullptr);
    }

    const float inverse_normalizer = normalizer > 0.0F ? __frcp_rn(normalizer) : 0.0F;
    if constexpr (PairLayout && !StablePair) {
        const int d0 = lane * 2;
        const int d1 = d0 + 64;
        out_row[d0]     = __float2bfloat16(accumulator[0] * inverse_normalizer);
        out_row[d0 + 1] = __float2bfloat16(accumulator[1] * inverse_normalizer);
        out_row[d1]     = __float2bfloat16(accumulator[2] * inverse_normalizer);
        out_row[d1 + 1] = __float2bfloat16(accumulator[3] * inverse_normalizer);
    } else {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            out_row[lane + item * 32] = __float2bfloat16(accumulator[item] * inverse_normalizer);
        }
    }
}

} // namespace ninfer::ops
