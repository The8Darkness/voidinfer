#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

template <bool Packed>
__device__ __forceinline__ void sliding_window_nvfp4_accumulate(
    int lane, const float (&q_values)[4], float scale, float& normalizer, float& running_max,
    float (&accumulator)[4], const __nv_bfloat16* key_raw, const __nv_bfloat16* value_raw,
    const std::uint8_t* key_codes, const std::uint8_t* value_codes,
    const std::uint8_t* key_scales, const std::uint8_t* value_scales) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr float Log2E        = 1.4426950408889634074F;
    float dot                    = 0.0F;
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

// Correctness-first NVFP4 DFlash2 local attention.  One warp owns one query head/token row;
// this keeps the packed-cache path independent of the BF16 tensor-core staging contract.  The
// verifier still consumes the exact target cache, so this route is only an approximate proposer.
template <bool HasProtected>
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
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        q_values[item] = __bfloat162float(q_row[lane + item * 32]);
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
            sliding_window_nvfp4_accumulate<false>(
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
            sliding_window_nvfp4_accumulate<true>(
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
                sliding_window_nvfp4_accumulate<false>(
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
            sliding_window_nvfp4_accumulate<true>(
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
        sliding_window_nvfp4_accumulate<false>(
            lane, q_values, scale, normalizer, running_max, accumulator, query_key_row,
            query_value_row, nullptr, nullptr, nullptr, nullptr);
    }

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const float value = normalizer > 0.0F ? accumulator[item] * __frcp_rn(normalizer) : 0.0F;
        out_row[lane + item * 32] = __float2bfloat16(value);
    }
}

} // namespace ninfer::ops
