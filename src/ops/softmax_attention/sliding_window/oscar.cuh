#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/oscar_codec.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ void sliding_window_oscar_accumulate_values(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const float (&key_values)[4], const float (&value_values)[4]) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr float Log2E       = 1.4426950408889634074F;
    float dot                   = 0.0F;
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
        accumulator[item] = fmaf(value_values[item], new_weight, accumulator[item] * old_weight);
    }
    running_max = next_max;
    (void)lane;
}

template <int Bits>
__device__ __forceinline__ void sliding_window_oscar_accumulate_packed_pair(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const __nv_bfloat16* key_metadata, const __nv_bfloat16* value_metadata) {
    float key_values[4];
    float value_values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d        = lane + item * 32;
        key_values[item]   = cyclic_oscar_decode_value<Bits>(key_codes, key_metadata, d);
        value_values[item] = cyclic_oscar_decode_value<Bits>(value_codes, value_metadata, d);
    }
    sliding_window_oscar_accumulate_values(lane, q_values, scale, normalizer, running_max,
                                           accumulator, key_values, value_values);
}

// Q2-specific packed-row decoder. Four adjacent natural dimensions share a
// byte, so the packed H128 lane layout lets every lane decode one K and one V
// byte without the four-way broadcasts used by the generic decoder.
__device__ __forceinline__ void sliding_window_oscar_accumulate_q2_packed(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const __nv_bfloat16* key_metadata, const __nv_bfloat16* value_metadata) {
    // These four values are uniform for the row. On SM120 the read-only path
    // services a warp-uniform address as a broadcast, so direct loads avoid
    // four explicit synchronization instructions per context row.
    const float key_scale   = __bfloat162float(key_metadata[0]);
    const float key_zero    = __bfloat162float(key_metadata[1]);
    const float value_scale = __bfloat162float(value_metadata[0]);
    const float value_zero  = __bfloat162float(value_metadata[1]);

    const int packed_key   = static_cast<int>(key_codes[lane]);
    const int packed_value = static_cast<int>(value_codes[lane]);
    float key_values[4];
    float value_values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int shift = item << 1;
        key_values[item] = fmaf(static_cast<float>((packed_key >> shift) & 3), key_scale,
                                key_zero);
        value_values[item] = fmaf(static_cast<float>((packed_value >> shift) & 3), value_scale,
                                  value_zero);
    }
    sliding_window_oscar_accumulate_values(lane, q_values, scale, normalizer, running_max,
                                           accumulator, key_values, value_values);
}

// Q2 decoder for the DFlash lane-local layout. Each lane owns the four Q2
// symbols for dimensions lane + {0,32,64,96}, so a row needs one K byte and
// one V byte per lane and no code-byte shuffles. The query/output tensors keep
// their original natural H128 layout; only the compressed code stream is
// transposed at append/restore time.
__device__ __forceinline__ void sliding_window_oscar_accumulate_q2_natural(
    int lane, const float (&q_values)[4], float q_sum, float scale, float& normalizer,
    float& running_max,
    float (&accumulator)[4], const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const __nv_bfloat16* key_metadata, const __nv_bfloat16* value_metadata) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr float Log2E       = 1.4426950408889634074F;
    // These addresses are warp-uniform. On SM120 the read-only path broadcasts
    // a uniform load, matching the Q4 decoder without four explicit shuffles.
    const float key_scale   = __bfloat162float(key_metadata[0]);
    const float key_zero    = __bfloat162float(key_metadata[1]);
    const float value_scale = __bfloat162float(value_metadata[0]);
    const float value_zero  = __bfloat162float(value_metadata[1]);

    // K and V have the same lane-local layout. Pack the two byte loads into a
    // single register so the four symbols are decoded without warp traffic.
    const int packed_key   = static_cast<int>(key_codes[lane]);
    const int packed_value = static_cast<int>(value_codes[lane]);
    const std::uint8_t key_code0   = static_cast<std::uint8_t>(packed_key & 3);
    const std::uint8_t key_code1   = static_cast<std::uint8_t>((packed_key >> 2) & 3);
    const std::uint8_t key_code2   = static_cast<std::uint8_t>((packed_key >> 4) & 3);
    const std::uint8_t key_code3   = static_cast<std::uint8_t>((packed_key >> 6) & 3);
    const std::uint8_t value_code0 = static_cast<std::uint8_t>(packed_value & 3);
    const std::uint8_t value_code1 = static_cast<std::uint8_t>((packed_value >> 2) & 3);
    const std::uint8_t value_code2 = static_cast<std::uint8_t>((packed_value >> 4) & 3);
    const std::uint8_t value_code3 = static_cast<std::uint8_t>((packed_value >> 6) & 3);

    // For affine OSCAR K, factor the row transform out of the dot product. This
    // avoids four key dequantization FMAs and keeps only the four integer-code
    // products plus two row-level affine operations. The value side is still
    // reconstructed after the softmax weight, so output accumulation semantics
    // remain unchanged.
    float code_dot = 0.0F;
    code_dot = fmaf(q_values[0], static_cast<float>(key_code0), code_dot);
    code_dot = fmaf(q_values[1], static_cast<float>(key_code1), code_dot);
    code_dot = fmaf(q_values[2], static_cast<float>(key_code2), code_dot);
    code_dot = fmaf(q_values[3], static_cast<float>(key_code3), code_dot);
    const float dot_local = fmaf(code_dot, key_scale, q_sum * key_zero);
    const float dot        = warp_sum<32>(dot_local, FullMask) * scale;
    const float next_max   = fmaxf(running_max, dot);
    const float old_weight = running_max == -CUDART_INF_F
                                 ? 0.0F
                                 : exp2_approx((running_max - next_max) * Log2E);
    const float new_weight = exp2_approx((dot - next_max) * Log2E);
    normalizer             = normalizer * old_weight + new_weight;
    const float value0     = fmaf(static_cast<float>(value_code0), value_scale, value_zero);
    const float value1     = fmaf(static_cast<float>(value_code1), value_scale, value_zero);
    const float value2     = fmaf(static_cast<float>(value_code2), value_scale, value_zero);
    const float value3     = fmaf(static_cast<float>(value_code3), value_scale, value_zero);
    accumulator[0] = fmaf(value0, new_weight, accumulator[0] * old_weight);
    accumulator[1] = fmaf(value1, new_weight, accumulator[1] * old_weight);
    accumulator[2] = fmaf(value2, new_weight, accumulator[2] * old_weight);
    accumulator[3] = fmaf(value3, new_weight, accumulator[3] * old_weight);
    running_max    = next_max;
    (void)lane;
}

__device__ __forceinline__ void sliding_window_oscar_accumulate_raw(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const __nv_bfloat16* key, const __nv_bfloat16* value) {
    float key_values[4];
    float value_values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d        = lane + item * 32;
        key_values[item]   = __bfloat162float(key[d]);
        value_values[item] = __bfloat162float(value[d]);
    }
    sliding_window_oscar_accumulate_values(lane, q_values, scale, normalizer, running_max,
                                           accumulator, key_values, value_values);
}

template <bool Rotate>
__device__ __forceinline__ void sliding_window_oscar_accumulate_packed_raw(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const __nv_bfloat16* key, const __nv_bfloat16* value) {
    float key_values[4];
    float value_values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d        = lane * 4 + item;
        key_values[item]   = __bfloat162float(key[d]);
        value_values[item] = __bfloat162float(value[d]);
    }
    if constexpr (Rotate) {
        normalized_hadamard_d128_packed_inplace(key_values, lane);
        normalized_hadamard_d128_packed_inplace(value_values, lane);
    }
    sliding_window_oscar_accumulate_values(lane, q_values, scale, normalizer, running_max,
                                           accumulator, key_values, value_values);
}

__device__ __forceinline__ void sliding_window_oscar_load_q2_shared(
    int lane, const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const __nv_bfloat16* key_metadata, const __nv_bfloat16* value_metadata, float* shared_key,
    float* shared_value) {
    constexpr unsigned FullMask = 0xffffffffU;
    float key_scale             = 0.0F;
    float key_zero              = 0.0F;
    float value_scale           = 0.0F;
    float value_zero            = 0.0F;
    if (lane == 0) {
        key_scale   = __bfloat162float(key_metadata[0]);
        key_zero    = __bfloat162float(key_metadata[1]);
        value_scale = __bfloat162float(value_metadata[0]);
        value_zero  = __bfloat162float(value_metadata[1]);
    }
    key_scale   = __shfl_sync(FullMask, key_scale, 0);
    key_zero    = __shfl_sync(FullMask, key_zero, 0);
    value_scale = __shfl_sync(FullMask, value_scale, 0);
    value_zero  = __shfl_sync(FullMask, value_zero, 0);

    const int packed_key   = static_cast<int>(key_codes[lane]);
    const int packed_value = static_cast<int>(value_codes[lane]);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int shift = item << 1;
        shared_key[lane * 4 + item] =
            fmaf(static_cast<float>((packed_key >> shift) & 3), key_scale, key_zero);
        shared_value[lane * 4 + item] =
            fmaf(static_cast<float>((packed_value >> shift) & 3), value_scale, value_zero);
    }
}

__device__ __forceinline__ void sliding_window_oscar_accumulate_shared_packed(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const float* shared_key, const float* shared_value) {
    float key_values[4];
    float value_values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        key_values[item]   = shared_key[lane * 4 + item];
        value_values[item] = shared_value[lane * 4 + item];
    }
    sliding_window_oscar_accumulate_values(lane, q_values, scale, normalizer, running_max,
                                           accumulator, key_values, value_values);
}

// Grouped Q2 DFlash attention. One CTA owns the four query heads sharing a
// KV head, and warp zero decodes each compressed row once into shared memory.
// The row is kept in the packed H128 layout so the other three warps consume
// it without a dimension transpose or duplicate Q2 byte loads.
template <bool HasProtected>
__launch_bounds__(128, 2) __global__ void sliding_window_attention_oscar_q2_grouped_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const std::uint8_t* __restrict__ context_k, const std::uint8_t* __restrict__ context_v,
    const __nv_bfloat16* __restrict__ context_k_scale,
    const __nv_bfloat16* __restrict__ context_v_scale,
    const __nv_bfloat16* __restrict__ protected_k,
    const __nv_bfloat16* __restrict__ protected_v, int padded_context, int max_context, int window,
    int protected_capacity, int protected_anchor_capacity, int protected_padded_capacity,
    int tokens, float scale, __nv_bfloat16* __restrict__ out) {
    constexpr int D       = kCyclicKVCacheOscarHeadDim;
    constexpr int QHeads  = 32;
    constexpr int KVHeads = kCyclicKVCacheOscarKVHeads;

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int group = static_cast<int>(blockIdx.x) % KVHeads;
    const int token = static_cast<int>(blockIdx.x) / KVHeads;
    const int batch = static_cast<int>(blockIdx.y);
    const int q_head = group * 4 + warp;
    if (token >= tokens || warp >= 4) return;

    const std::int64_t q_batch_stride  = static_cast<std::int64_t>(D) * QHeads * tokens;
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
        q_values[item] = __bfloat162float(q_row[lane * 4 + item]);
    }
    normalized_hadamard_d128_packed_inplace(q_values, lane);

    const int context_count = min(length, window - 1);
    const int context_start = length - context_count;
    const int state_lane    = lanes[batch];
    const std::int64_t code_lane_stride =
        static_cast<std::int64_t>(cyclic_oscar_code_extent<2>(D)) * padded_context * KVHeads;
    const std::int64_t scale_lane_stride =
        static_cast<std::int64_t>(kCyclicKVCacheOscarScaleExtent) * padded_context * KVHeads;
    const std::uint8_t* lane_k = context_k + code_lane_stride * state_lane;
    const std::uint8_t* lane_v = context_v + code_lane_stride * state_lane;
    const __nv_bfloat16* lane_k_scale = context_k_scale + scale_lane_stride * state_lane;
    const __nv_bfloat16* lane_v_scale = context_v_scale + scale_lane_stride * state_lane;
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
    float running_max    = -CUDART_INF_F;

    if constexpr (HasProtected) {
        const int anchor_end    = min(length, protected_anchor_capacity);
        const int recent_begin  = protected_capacity == 0 ? length : max(0, length - protected_capacity);
        for (int context_position = context_start;
             context_position < min(anchor_end, length); ++context_position) {
            const std::int64_t protected_offset =
                static_cast<std::int64_t>(D) *
                (context_position + static_cast<std::int64_t>(protected_padded_capacity) * group);
            if (warp == 0) {
#pragma unroll
                for (int item = 0; item < 4; ++item) {
                    shared_key[lane * 4 + item] =
                        __bfloat162float(lane_protected_k[protected_offset + lane * 4 + item]);
                    shared_value[lane * 4 + item] =
                        __bfloat162float(lane_protected_v[protected_offset + lane * 4 + item]);
                }
            }
            __syncthreads();
            sliding_window_oscar_accumulate_shared_packed(
                lane, q_values, scale, normalizer, running_max, accumulator, shared_key,
                shared_value);
        }
        const int packed_begin = max(context_start, anchor_end);
        const int packed_end   = min(length, recent_begin);
        for (int context_position = packed_begin; context_position < packed_end;
             ++context_position) {
            const int slot = context_position & (window - 1);
            const std::int64_t slot_offset =
                static_cast<std::int64_t>(cyclic_oscar_code_extent<2>(D)) *
                (slot + static_cast<std::int64_t>(padded_context) * group);
            const std::int64_t scale_offset =
                static_cast<std::int64_t>(kCyclicKVCacheOscarScaleExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * group);
            if (warp == 0) {
                sliding_window_oscar_load_q2_shared(
                    lane, lane_k + slot_offset, lane_v + slot_offset, lane_k_scale + scale_offset,
                    lane_v_scale + scale_offset, shared_key, shared_value);
            }
            __syncthreads();
            sliding_window_oscar_accumulate_shared_packed(
                lane, q_values, scale, normalizer, running_max, accumulator, shared_key,
                shared_value);
        }
        if constexpr (true) {
            if (protected_capacity != 0) {
                const int recent_begin_in_window =
                    max(context_start, max(recent_begin, anchor_end));
                for (int context_position = recent_begin_in_window; context_position < length;
                     ++context_position) {
                    const int protected_slot =
                        protected_anchor_capacity + (context_position & (protected_capacity - 1));
                    const std::int64_t protected_offset =
                        static_cast<std::int64_t>(D) *
                        (protected_slot + static_cast<std::int64_t>(protected_padded_capacity) *
                                         group);
                    if (warp == 0) {
#pragma unroll
                        for (int item = 0; item < 4; ++item) {
                            shared_key[lane * 4 + item] =
                                __bfloat162float(lane_protected_k[protected_offset + lane * 4 + item]);
                            shared_value[lane * 4 + item] =
                                __bfloat162float(lane_protected_v[protected_offset + lane * 4 + item]);
                        }
                    }
                    __syncthreads();
                    sliding_window_oscar_accumulate_shared_packed(
                        lane, q_values, scale, normalizer, running_max, accumulator, shared_key,
                        shared_value);
                }
            }
        }
    } else {
        for (int key = 0; key < context_count; ++key) {
            const int context_position = context_start + key;
            const int slot             = context_position & (window - 1);
            const std::int64_t slot_offset =
                static_cast<std::int64_t>(cyclic_oscar_code_extent<2>(D)) *
                (slot + static_cast<std::int64_t>(padded_context) * group);
            const std::int64_t scale_offset =
                static_cast<std::int64_t>(kCyclicKVCacheOscarScaleExtent) *
                (slot + static_cast<std::int64_t>(padded_context) * group);
            if (warp == 0) {
                sliding_window_oscar_load_q2_shared(
                    lane, lane_k + slot_offset, lane_v + slot_offset, lane_k_scale + scale_offset,
                    lane_v_scale + scale_offset, shared_key, shared_value);
            }
            __syncthreads();
            sliding_window_oscar_accumulate_shared_packed(
                lane, q_values, scale, normalizer, running_max, accumulator, shared_key,
                shared_value);
        }
    }

    // The current query K/V block is shared by the four query heads. Warp zero
    // performs the model-to-H128 transform once per proposed token.
    for (int query_token = 0; query_token < valid; ++query_token) {
        if (warp == 0) {
            const __nv_bfloat16* query_key_row =
                query_k_batch + static_cast<std::int64_t>(D) * (group + KVHeads * query_token);
            const __nv_bfloat16* query_value_row =
                query_v_batch + static_cast<std::int64_t>(D) * (group + KVHeads * query_token);
            float key_values[4];
            float value_values[4];
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                key_values[item]   = __bfloat162float(query_key_row[lane * 4 + item]);
                value_values[item] = __bfloat162float(query_value_row[lane * 4 + item]);
            }
            normalized_hadamard_d128_packed_inplace(key_values, lane);
            normalized_hadamard_d128_packed_inplace(value_values, lane);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                shared_query_key[query_token * D + lane * 4 + item]   = key_values[item];
                shared_query_value[query_token * D + lane * 4 + item] = value_values[item];
            }
        }
    }
    __syncthreads();
    for (int query_token = 0; query_token < valid; ++query_token) {
        sliding_window_oscar_accumulate_shared_packed(
            lane, q_values, scale, normalizer, running_max, accumulator,
            shared_query_key + query_token * D, shared_query_value + query_token * D);
    }

    const float inverse_normalizer = normalizer > 0.0F ? __frcp_rn(normalizer) : 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) { accumulator[item] *= inverse_normalizer; }
    normalized_hadamard_d128_packed_inplace(accumulator, lane);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        out_row[lane * 4 + item] = __float2bfloat16(accumulator[item]);
    }
}

template <int Bits>
__launch_bounds__(32, 4) __global__ void sliding_window_attention_oscar_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const std::uint8_t* __restrict__ context_k, const std::uint8_t* __restrict__ context_v,
    const __nv_bfloat16* __restrict__ context_k_scale,
    const __nv_bfloat16* __restrict__ context_v_scale,
    const __nv_bfloat16* __restrict__ protected_k,
    const __nv_bfloat16* __restrict__ protected_v, int padded_context, int max_context, int window,
    int protected_capacity, int protected_anchor_capacity, int protected_padded_capacity,
    int tokens, float scale, __nv_bfloat16* __restrict__ out) {
    constexpr int D          = kCyclicKVCacheOscarHeadDim;
    constexpr int QHeads     = 32;
    constexpr int KVHeads    = kCyclicKVCacheOscarKVHeads;
    constexpr int CodeExtent = cyclic_oscar_code_extent<Bits>(D);
    constexpr int ScaleExtent = kCyclicKVCacheOscarScaleExtent;

    const int lane  = static_cast<int>(threadIdx.x);
    const int row   = static_cast<int>(blockIdx.x);
    const int token = row / QHeads;
    const int q_head = row - token * QHeads;
    const int batch  = static_cast<int>(blockIdx.y);
    if (lane >= 32 || token >= tokens) return;

    const std::int64_t q_batch_stride  = static_cast<std::int64_t>(D) * QHeads * tokens;
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
    normalized_hadamard_d128_inplace(q_values, lane);
    float q_sum = 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) { q_sum += q_values[item]; }

    const int context_count = min(length, window - 1);
    const int context_start = length - context_count;
    const int kv_head        = q_head >> 2;
    const int state_lane     = lanes[batch];
    const std::int64_t code_lane_stride =
        static_cast<std::int64_t>(CodeExtent) * padded_context * KVHeads;
    const std::int64_t scale_lane_stride =
        static_cast<std::int64_t>(ScaleExtent) * padded_context * KVHeads;
    const std::uint8_t* lane_k = context_k + code_lane_stride * state_lane;
    const std::uint8_t* lane_v = context_v + code_lane_stride * state_lane;
    const __nv_bfloat16* lane_k_scale = context_k_scale + scale_lane_stride * state_lane;
    const __nv_bfloat16* lane_v_scale = context_v_scale + scale_lane_stride * state_lane;
    const __nv_bfloat16* lane_protected_k = nullptr;
    const __nv_bfloat16* lane_protected_v = nullptr;
    if (protected_capacity != 0 || protected_anchor_capacity != 0) {
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
    if (protected_capacity != 0 || protected_anchor_capacity != 0) {
        const int anchor_end   = min(length, protected_anchor_capacity);
        const int recent_begin = protected_capacity == 0 ? length : max(0, length - protected_capacity);
        for (int context_position = context_start;
             context_position < min(anchor_end, length); ++context_position) {
            const std::int64_t protected_offset =
                static_cast<std::int64_t>(D) *
                (context_position + static_cast<std::int64_t>(protected_padded_capacity) * kv_head);
            sliding_window_oscar_accumulate_raw(
                lane, q_values, scale, normalizer, running_max, accumulator,
                lane_protected_k + protected_offset, lane_protected_v + protected_offset);
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
            if constexpr (Bits == 2) {
                sliding_window_oscar_accumulate_q2_natural(
                    lane, q_values, q_sum, scale, normalizer, running_max, accumulator,
                    lane_k + slot_offset, lane_v + slot_offset, lane_k_scale + scale_offset,
                    lane_v_scale + scale_offset);
            } else {
                sliding_window_oscar_accumulate_packed_pair<Bits>(
                    lane, q_values, scale, normalizer, running_max, accumulator,
                    lane_k + slot_offset, lane_v + slot_offset, lane_k_scale + scale_offset,
                    lane_v_scale + scale_offset);
            }
        }
        if (protected_capacity != 0) {
            const int recent_begin_in_window = max(context_start, max(recent_begin, anchor_end));
            for (int context_position = recent_begin_in_window; context_position < length;
                 ++context_position) {
                const int protected_slot =
                    protected_anchor_capacity + (context_position & (protected_capacity - 1));
                const std::int64_t protected_offset =
                    static_cast<std::int64_t>(D) *
                    (protected_slot + static_cast<std::int64_t>(protected_padded_capacity) * kv_head);
                sliding_window_oscar_accumulate_raw(
                    lane, q_values, scale, normalizer, running_max, accumulator,
                    lane_protected_k + protected_offset, lane_protected_v + protected_offset);
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
            if constexpr (Bits == 2) {
                sliding_window_oscar_accumulate_q2_natural(
                    lane, q_values, q_sum, scale, normalizer, running_max, accumulator,
                    lane_k + slot_offset, lane_v + slot_offset, lane_k_scale + scale_offset,
                    lane_v_scale + scale_offset);
            } else {
                sliding_window_oscar_accumulate_packed_pair<Bits>(
                    lane, q_values, scale, normalizer, running_max, accumulator,
                    lane_k + slot_offset, lane_v + slot_offset, lane_k_scale + scale_offset,
                    lane_v_scale + scale_offset);
            }
        }
    }

    for (int query_token = 0; query_token < valid; ++query_token) {
        const __nv_bfloat16* query_key_row =
            query_k_batch + static_cast<std::int64_t>(D) * (kv_head + KVHeads * query_token);
        const __nv_bfloat16* query_value_row =
            query_v_batch + static_cast<std::int64_t>(D) * (kv_head + KVHeads * query_token);
        float key_values[4];
        float value_values[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d        = lane + item * 32;
            key_values[item]   = __bfloat162float(query_key_row[d]);
            value_values[item] = __bfloat162float(query_value_row[d]);
        }
        normalized_hadamard_d128_inplace(key_values, lane);
        normalized_hadamard_d128_inplace(value_values, lane);
        sliding_window_oscar_accumulate_values(lane, q_values, scale, normalizer, running_max,
                                                accumulator, key_values, value_values);
    }

    const float inverse_normalizer = normalizer > 0.0F ? __frcp_rn(normalizer) : 0.0F;
#pragma unroll
    for (int item = 0; item < 4; ++item) { accumulator[item] *= inverse_normalizer; }
    normalized_hadamard_d128_inplace(accumulator, lane);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        out_row[lane + item * 32] = __float2bfloat16(accumulator[item]);
    }
}

} // namespace ninfer::ops
