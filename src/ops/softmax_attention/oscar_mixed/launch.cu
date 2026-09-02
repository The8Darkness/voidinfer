#include "ops/softmax_attention/oscar_mixed/launch.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kGqa = 6;
constexpr int kHeadDim = 256;
constexpr int kGroups = 2;
constexpr int kGroupSize = 128;
constexpr int kCodeBytes = 64;
constexpr int kPrefixTokens = 64;
constexpr int kRecentTokens = 256;
constexpr int kInt2Levels = 3;
constexpr int kInt2MetadataItems = 4;
constexpr int kThreads = 256;
constexpr int kTokenTile = 256;
static_assert(kGroupSize == 128 && kGroups == 2 && kCodeBytes == 64,
              "OscarInt2G128 mixed layout changed");

__device__ __forceinline__ float decode_int2(const std::uint8_t* packed,
                                              const float* metadata, int dimension) {
    const int byte = dimension & 63;
    const int shift = (dimension >> 6) << 1;
    const std::uint8_t symbol = static_cast<std::uint8_t>((packed[byte] >> shift) & 3U);
    const int group = dimension >> 7;
    return (static_cast<float>(symbol) - metadata[group * 2 + 1]) * metadata[group * 2];
}

__device__ __forceinline__ float decode_bf16(std::uint16_t bits) {
    return __int_as_float(static_cast<int>(static_cast<std::uint32_t>(bits) << 16U));
}

__device__ __forceinline__ float load_key(
    int logical, int kv_head, int dimension, const std::uint16_t* prefix_k, int prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    int historical_tokens, const std::uint16_t* recent_k, int recent_tokens,
    int recent_ring_head) {
    if (logical < prefix_tokens) {
        const std::size_t row = (static_cast<std::size_t>(logical) * kKVHeads + kv_head) * kHeadDim;
        return decode_bf16(prefix_k[row + dimension]);
    }
    const int historical_logical = logical - prefix_tokens;
    if (historical_logical < historical_tokens) {
        const std::size_t row = static_cast<std::size_t>(historical_logical) * kKVHeads + kv_head;
        return decode_int2(historical_k_packed + row * kCodeBytes,
                           historical_k_metadata + row * (kGroups * 2), dimension);
    }
    const int recent_logical = historical_logical - historical_tokens;
    if (recent_logical < recent_tokens) {
        const int physical_recent = (recent_ring_head + recent_logical) & (kTokenTile - 1);
        const std::size_t row =
            (static_cast<std::size_t>(physical_recent) * kKVHeads + kv_head) * kHeadDim;
        return decode_bf16(recent_k[row + dimension]);
    }
    return 0.0F;
}

__device__ __forceinline__ float load_value(
    int logical, int kv_head, int dimension, const std::uint16_t* prefix_v, int prefix_tokens,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    int historical_tokens, const std::uint16_t* recent_v, int recent_tokens,
    int recent_ring_head) {
    if (logical < prefix_tokens) {
        const std::size_t row = (static_cast<std::size_t>(logical) * kKVHeads + kv_head) * kHeadDim;
        return decode_bf16(prefix_v[row + dimension]);
    }
    const int historical_logical = logical - prefix_tokens;
    if (historical_logical < historical_tokens) {
        const std::size_t row = static_cast<std::size_t>(historical_logical) * kKVHeads + kv_head;
        return decode_int2(historical_v_packed + row * kCodeBytes,
                           historical_v_metadata + row * (kGroups * 2), dimension);
    }
    const int recent_logical = historical_logical - historical_tokens;
    if (recent_logical < recent_tokens) {
        const int physical_recent = (recent_ring_head + recent_logical) & (kTokenTile - 1);
        const std::size_t row =
            (static_cast<std::size_t>(physical_recent) * kKVHeads + kv_head) * kHeadDim;
        return decode_bf16(recent_v[row + dimension]);
    }
    return 0.0F;
}

// One block owns one KV head and one logical 256-token tile. K is decoded at the
// point of use and each decoded scalar feeds all six Q heads sharing the KV head.
__global__ void oscar_mixed_score_kernel(
    const float* __restrict__ q_rotated, const std::uint16_t* __restrict__ prefix_k,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_k_packed,
    const float* __restrict__ historical_k_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_k, int recent_tokens, int recent_ring_head,
    int total_tokens, float attention_scale, float* __restrict__ scores) {
    __shared__ float q_shared[kGqa * kHeadDim];
    const int kv_head = static_cast<int>(blockIdx.x);
    const int logical = static_cast<int>(blockIdx.y) * kTokenTile + static_cast<int>(threadIdx.x);
    for (int index = static_cast<int>(threadIdx.x); index < kGqa * kHeadDim; index += kThreads) {
        const int query_head = kv_head * kGqa + index / kHeadDim;
        q_shared[index] = q_rotated[query_head * kHeadDim + index % kHeadDim];
    }
    __syncthreads();
    if (logical >= total_tokens) return;

    float accum[kGqa] = {};
    for (int dimension = 0; dimension < kHeadDim; ++dimension) {
        const float key = load_key(logical, kv_head, dimension, prefix_k, prefix_tokens,
                                   historical_k_packed, historical_k_metadata, historical_tokens,
                                   recent_k, recent_tokens, recent_ring_head);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            accum[query_group] += q_shared[query_group * kHeadDim + dimension] * key;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        scores[query_head * total_tokens + logical] = accum[query_group] * attention_scale;
    }
}

__device__ __forceinline__ float reduce_max(float value) {
    __shared__ float values[kThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return values[0];
}

__device__ __forceinline__ float reduce_sum(float value) {
    __shared__ float values[kThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    return values[0];
}

__global__ void oscar_mixed_softmax_kernel(const float* __restrict__ scores, int total_tokens,
                                           float* __restrict__ softmax) {
    const int query_head = static_cast<int>(blockIdx.x);
    const float* row = scores + query_head * total_tokens;
    float maximum = -3.402823466e+38F;
    for (int logical = static_cast<int>(threadIdx.x); logical < total_tokens; logical += kThreads) {
        maximum = fmaxf(maximum, row[logical]);
    }
    maximum = reduce_max(maximum);
    float sum = 0.0F;
    for (int logical = static_cast<int>(threadIdx.x); logical < total_tokens; logical += kThreads) {
        const float probability = expf(row[logical] - maximum);
        softmax[query_head * total_tokens + logical] = probability;
        sum += probability;
    }
    sum = reduce_sum(sum);
    for (int logical = static_cast<int>(threadIdx.x); logical < total_tokens; logical += kThreads) {
        softmax[query_head * total_tokens + logical] /= sum;
    }
}

// One block owns one KV head and one output dimension per thread. V is decoded
// directly into six FP32 AV accumulators; there is no decoded-V allocation.
__global__ void oscar_mixed_av_kernel(
    const float* __restrict__ softmax, const std::uint16_t* __restrict__ prefix_v,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_v_packed,
    const float* __restrict__ historical_v_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_v, int recent_tokens, int recent_ring_head,
    int total_tokens, float* __restrict__ output) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    if (dimension >= kHeadDim) return;

    float accum[kGqa] = {};
    for (int logical = 0; logical < total_tokens; ++logical) {
        const float value = load_value(logical, kv_head, dimension, prefix_v, prefix_tokens,
                                       historical_v_packed, historical_v_metadata, historical_tokens,
                                       recent_v, recent_tokens, recent_ring_head);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            const int query_head = kv_head * kGqa + query_group;
            accum[query_group] += softmax[query_head * total_tokens + logical] * value;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        output[query_head * kHeadDim + dimension] = accum[query_group];
    }
}

// The batched prefill kernels intentionally retain the D4.4 scalar-query operation order.  A
// query is an independent grid-z slice, so batching changes launch/setup overhead without
// changing the mixed-cache decode, softmax, GQA, or AV arithmetic for any query column.
__global__ void oscar_mixed_score_batch_kernel(
    const float* __restrict__ q_rotated, const std::uint16_t* __restrict__ prefix_k,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_k_packed,
    const float* __restrict__ historical_k_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_k, int recent_tokens, int recent_ring_head,
    int total_tokens, int query_start, int workspace_stride, float attention_scale,
    float* __restrict__ scores) {
    __shared__ float q_shared[kGqa * kHeadDim];
    const int kv_head = static_cast<int>(blockIdx.x);
    const int query_index = static_cast<int>(blockIdx.z);
    const int query_token = query_start + query_index;
    const int logical = static_cast<int>(blockIdx.y) * kTokenTile + static_cast<int>(threadIdx.x);
    for (int index = static_cast<int>(threadIdx.x); index < kGqa * kHeadDim; index += kThreads) {
        const int query_head = kv_head * kGqa + index / kHeadDim;
        q_shared[index] = q_rotated[(static_cast<std::size_t>(query_index) * kQHeads +
                                     query_head) * kHeadDim + index % kHeadDim];
    }
    __syncthreads();
    if (logical >= total_tokens || logical > query_token) return;

    float accum[kGqa] = {};
    for (int dimension = 0; dimension < kHeadDim; ++dimension) {
        const float key = load_key(logical, kv_head, dimension, prefix_k, prefix_tokens,
                                   historical_k_packed, historical_k_metadata, historical_tokens,
                                   recent_k, recent_tokens, recent_ring_head);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            accum[query_group] += q_shared[query_group * kHeadDim + dimension] * key;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        const std::size_t row = (static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                                static_cast<std::size_t>(workspace_stride);
        scores[row + logical] = accum[query_group] * attention_scale;
    }
}

__global__ void oscar_mixed_softmax_batch_kernel(const float* __restrict__ scores,
                                                 int total_tokens, int query_start,
                                                 int query_count, int workspace_stride,
                                                 float* __restrict__ softmax) {
    const int flat = static_cast<int>(blockIdx.x);
    const int query_index = flat / kQHeads;
    const int query_head = flat % kQHeads;
    if (query_index >= query_count) return;
    const int visible_tokens = min(total_tokens, query_start + query_index + 1);
    const float* row = scores +
                       (static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                           static_cast<std::size_t>(workspace_stride);
    float maximum = -3.402823466e+38F;
    for (int logical = static_cast<int>(threadIdx.x); logical < visible_tokens;
         logical += kThreads) {
        maximum = fmaxf(maximum, row[logical]);
    }
    maximum = reduce_max(maximum);
    float sum = 0.0F;
    for (int logical = static_cast<int>(threadIdx.x); logical < visible_tokens;
         logical += kThreads) {
        const float probability = expf(row[logical] - maximum);
        softmax[(static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                    static_cast<std::size_t>(workspace_stride) + logical] = probability;
        sum += probability;
    }
    sum = reduce_sum(sum);
    for (int logical = static_cast<int>(threadIdx.x); logical < visible_tokens;
         logical += kThreads) {
        softmax[(static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                    static_cast<std::size_t>(workspace_stride) + logical] /= sum;
    }
}

__global__ void oscar_mixed_av_batch_kernel(
    const float* __restrict__ softmax, const std::uint16_t* __restrict__ prefix_v,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_v_packed,
    const float* __restrict__ historical_v_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_v, int recent_tokens, int recent_ring_head,
    int total_tokens, int query_start, int query_count, int workspace_stride,
    float* __restrict__ output) {
    const int flat = static_cast<int>(blockIdx.x);
    const int query_index = flat / kKVHeads;
    const int kv_head = flat % kKVHeads;
    const int dimension = static_cast<int>(threadIdx.x);
    if (query_index >= query_count || dimension >= kHeadDim) return;
    const int visible_tokens = min(total_tokens, query_start + query_index + 1);

    float accum[kGqa] = {};
    for (int logical = 0; logical < visible_tokens; ++logical) {
        const float value = load_value(logical, kv_head, dimension, prefix_v, prefix_tokens,
                                       historical_v_packed, historical_v_metadata, historical_tokens,
                                       recent_v, recent_tokens, recent_ring_head);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            const int query_head = kv_head * kGqa + query_group;
            const std::size_t probability =
                (static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                    static_cast<std::size_t>(workspace_stride) + logical;
            accum[query_group] += softmax[probability] * value;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        output[(static_cast<std::size_t>(query_index) * kQHeads + query_head) * kHeadDim +
               dimension] = accum[query_group];
    }
}

__device__ __forceinline__ std::uint16_t oscar_float_to_bf16(float value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(__float_as_uint(value));
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

__device__ __forceinline__ float oscar_bf16_to_float(std::uint16_t bits) {
    return __int_as_float(static_cast<int>(static_cast<std::uint32_t>(bits) << 16U));
}

// The official host reference first rounds a newly produced FP32 row to BF16 while it is in the
// protected/recent tier, then quantizes that BF16 value when it ages.  The resident path keeps
// that contract by applying the same round-to-nearest-even conversion before either storing or
// encoding a value.
__device__ __forceinline__ std::uint16_t resident_input_bf16(
    const float* rotated, std::uint32_t input_token, int kv_head, int dimension,
    int token_count) {
    const std::size_t index = static_cast<std::size_t>(dimension) +
                              static_cast<std::size_t>(kHeadDim) *
                                  (static_cast<std::size_t>(kv_head) +
                                   static_cast<std::size_t>(kKVHeads) * input_token);
    (void)token_count;
    return oscar_float_to_bf16(rotated[index]);
}

__device__ __forceinline__ std::uint16_t resident_recent_bf16(
    const std::uint16_t* recent, std::uint32_t logical_token, int kv_head, int dimension,
    std::uint32_t old_recent_begin, std::uint32_t old_recent_head) {
    const std::uint32_t physical =
        (old_recent_head + logical_token - old_recent_begin) & (kTokenTile - 1);
    const std::size_t index =
        (static_cast<std::size_t>(physical) * kKVHeads + kv_head) * kHeadDim + dimension;
    return recent[index];
}

// This is a deliberately simple GPU encoder used only at the cache-publish boundary.  It uses a
// 256-element bitonic sort to reproduce the official sorted-absolute percentile clip, then the
// exact two independent g128 asymmetric groups and quartered byte layout.  The historical
// attention kernel still decodes the resulting bytes directly; no decoded historical buffer is
// created.
__device__ void encode_resident_row(
    const float* input, const std::uint16_t* old_recent, bool from_recent,
    std::uint32_t logical_token, std::uint32_t logical_start, int token_count, int kv_head,
    std::uint32_t old_recent_begin, std::uint32_t old_recent_head, int clip_index,
    std::uint8_t* packed_output, float* metadata_output, float* abs_values,
    float* clipped_values, std::uint8_t* symbols) {
    const int dimension = static_cast<int>(threadIdx.x);
    const std::uint32_t input_token = logical_token - logical_start;
    const std::uint16_t bits = from_recent
                                   ? resident_recent_bf16(old_recent, logical_token, kv_head,
                                                          dimension, old_recent_begin,
                                                          old_recent_head)
                                   : resident_input_bf16(input, input_token, kv_head, dimension,
                                                         token_count);
    const float value = oscar_bf16_to_float(bits);
    abs_values[dimension] = fabsf(value);
    clipped_values[dimension] = value;
    __syncthreads();

    for (int size = 2; size <= kHeadDim; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int partner = dimension ^ stride;
            if (partner > dimension) {
                const bool ascending = (dimension & size) == 0;
                const float left = abs_values[dimension];
                const float right = abs_values[partner];
                if ((ascending && left > right) || (!ascending && left < right)) {
                    abs_values[dimension] = right;
                    abs_values[partner] = left;
                }
            }
            __syncthreads();
        }
    }
    const float threshold = abs_values[clip_index];
    const float clipped = fminf(fmaxf(value, -threshold), threshold);
    clipped_values[dimension] = clipped;
    __syncthreads();

    if (dimension == 0) {
        for (int group = 0; group < kGroups; ++group) {
            const int begin = group * kGroupSize;
            float minimum = clipped_values[begin];
            float maximum = minimum;
            for (int offset = 1; offset < kGroupSize; ++offset) {
                const float current = clipped_values[begin + offset];
                minimum = fminf(minimum, current);
                maximum = fmaxf(maximum, current);
            }
            const float range = fmaxf(maximum - minimum, 1.0e-8F);
            metadata_output[group * 2] = range / static_cast<float>(kInt2Levels);
            metadata_output[group * 2 + 1] = -minimum / metadata_output[group * 2];
        }
    }
    __syncthreads();
    float normalized = clipped_values[dimension] /
                           metadata_output[(dimension >> 7) * 2] +
                       metadata_output[(dimension >> 7) * 2 + 1] + 0.5F;
    int code = static_cast<int>(floorf(normalized));
    code = max(0, min(kInt2Levels, code));
    symbols[dimension] = static_cast<std::uint8_t>(code);
    __syncthreads();
    if (dimension < kCodeBytes) {
        packed_output[dimension] = static_cast<std::uint8_t>(
            symbols[dimension] | (symbols[dimension + 64] << 2U) |
            (symbols[dimension + 128] << 4U) | (symbols[dimension + 192] << 6U));
    }
    __syncthreads();
}

__global__ void oscar_resident_write_bf16_kernel(
    const float* __restrict__ rotated_k, const float* __restrict__ rotated_v, int token_count,
    std::uint32_t logical_start, std::uint32_t final_recent_begin,
    std::uint32_t final_recent_head, std::uint16_t* __restrict__ prefix_k,
    std::uint16_t* __restrict__ prefix_v, std::uint16_t* __restrict__ recent_k,
    std::uint16_t* __restrict__ recent_v) {
    const std::uint32_t input_token = static_cast<std::uint32_t>(blockIdx.x);
    if (input_token >= static_cast<std::uint32_t>(token_count)) return;
    const std::uint32_t logical = logical_start + input_token;
    if (logical >= final_recent_begin && logical < final_recent_begin + kRecentTokens) {
        const std::uint32_t physical = (final_recent_head + logical - final_recent_begin) &
                                       (kTokenTile - 1);
        for (int flat = static_cast<int>(threadIdx.x); flat < kKVHeads * kHeadDim;
             flat += kThreads) {
            const int head = flat / kHeadDim;
            const int dimension = flat % kHeadDim;
            const std::size_t source = static_cast<std::size_t>(dimension) +
                                       static_cast<std::size_t>(kHeadDim) *
                                           (static_cast<std::size_t>(head) +
                                            static_cast<std::size_t>(kKVHeads) * input_token);
            const std::size_t destination =
                (static_cast<std::size_t>(physical) * kKVHeads + head) * kHeadDim + dimension;
            recent_k[destination] = oscar_float_to_bf16(rotated_k[source]);
            recent_v[destination] = oscar_float_to_bf16(rotated_v[source]);
        }
    } else if (logical < kPrefixTokens) {
        for (int flat = static_cast<int>(threadIdx.x); flat < kKVHeads * kHeadDim;
             flat += kThreads) {
            const int head = flat / kHeadDim;
            const int dimension = flat % kHeadDim;
            const std::size_t source = static_cast<std::size_t>(dimension) +
                                       static_cast<std::size_t>(kHeadDim) *
                                           (static_cast<std::size_t>(head) +
                                            static_cast<std::size_t>(kKVHeads) * input_token);
            const std::size_t destination =
                (static_cast<std::size_t>(logical) * kKVHeads + head) * kHeadDim + dimension;
            prefix_k[destination] = oscar_float_to_bf16(rotated_k[source]);
            prefix_v[destination] = oscar_float_to_bf16(rotated_v[source]);
        }
    }
}

__global__ void oscar_resident_encode_kernel(
    const float* __restrict__ rotated_k, const float* __restrict__ rotated_v,
    int token_count, std::uint32_t logical_start, std::uint32_t old_context,
    std::uint32_t old_recent_begin, std::uint32_t old_recent_head,
    std::uint32_t final_recent_begin, std::uint32_t new_historical_tokens,
    std::uint32_t old_aging_tokens, std::uint16_t* __restrict__ old_recent_k,
    std::uint16_t* __restrict__ old_recent_v, std::uint8_t* __restrict__ historical_k,
    std::uint8_t* __restrict__ historical_v, float* __restrict__ historical_k_metadata,
    float* __restrict__ historical_v_metadata) {
    const std::uint32_t work = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t total_work = (new_historical_tokens + old_aging_tokens) * kKVHeads;
    if (work >= total_work) return;
    const std::uint32_t row = work / kKVHeads;
    const int head = static_cast<int>(work % kKVHeads);
    const bool from_recent = row >= new_historical_tokens;
    const std::uint32_t logical = from_recent
                                      ? old_recent_begin + row - new_historical_tokens
                                      : logical_start + row;
    if (logical < kPrefixTokens || logical >= final_recent_begin ||
        logical >= old_context + (from_recent ? 0U : static_cast<std::uint32_t>(token_count)) ||
        logical < logical_start && !from_recent) {
        return;
    }
    const std::size_t output_row =
        (static_cast<std::size_t>(logical - kPrefixTokens) * kKVHeads) + head;
    extern __shared__ std::uint8_t shared_bytes[];
    float* abs_values = reinterpret_cast<float*>(shared_bytes);
    float* clipped_values = abs_values + kHeadDim;
    std::uint8_t* symbols = reinterpret_cast<std::uint8_t*>(clipped_values + kHeadDim);
    std::uint8_t* k_packed = symbols + kHeadDim;
    float* k_metadata = reinterpret_cast<float*>(k_packed + kCodeBytes);
    std::uint8_t* v_packed = reinterpret_cast<std::uint8_t*>(k_metadata + kInt2MetadataItems);
    float* v_metadata = reinterpret_cast<float*>(v_packed + kCodeBytes);

    const float* k_input = rotated_k;
    const float* v_input = rotated_v;
    float* k_scratch = abs_values;
    float* v_scratch = abs_values;
    encode_resident_row(k_input, old_recent_k, from_recent, logical, logical_start, token_count,
                        head, old_recent_begin, old_recent_head, 245, k_packed, k_metadata,
                        k_scratch, clipped_values, symbols);
    encode_resident_row(v_input, old_recent_v, from_recent, logical, logical_start, token_count,
                        head, old_recent_begin, old_recent_head, 235, v_packed, v_metadata,
                        v_scratch, clipped_values, symbols);
    if (threadIdx.x == 0) {
        const std::size_t payload_offset = output_row * kCodeBytes;
        const std::size_t metadata_offset = output_row * kInt2MetadataItems;
        for (int byte = 0; byte < kCodeBytes; ++byte) {
            historical_k[payload_offset + byte] = k_packed[byte];
            historical_v[payload_offset + byte] = v_packed[byte];
        }
        for (int item = 0; item < kInt2MetadataItems; ++item) {
            historical_k_metadata[metadata_offset + item] = k_metadata[item];
            historical_v_metadata[metadata_offset + item] = v_metadata[item];
        }
    }
}

} // namespace

void oscar_int2_g128_mixed_attention_launch(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens, float attention_scale,
    float* scores, float* softmax, float* output, cudaStream_t stream) {
    oscar_int2_g128_mixed_attention_launch_ring(
        q_rotated, prefix_k_bf16, prefix_v_bf16, prefix_tokens, historical_k_packed,
        historical_k_metadata, historical_v_packed, historical_v_metadata, historical_tokens,
        recent_k_bf16, recent_v_bf16, recent_tokens, 0, attention_scale, scores, softmax, output,
        stream);
}

void oscar_int2_g128_mixed_attention_launch_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, float attention_scale, float* scores, float* softmax,
    float* output, cudaStream_t stream) {
    if (prefix_tokens < 0 || prefix_tokens > 64 || historical_tokens < 0 ||
        recent_tokens < 0 || recent_tokens > kRecentTokens || recent_ring_head < 0 ||
        recent_ring_head >= kRecentTokens) {
        throw std::invalid_argument("OSCAR mixed attention tier count is invalid");
    }
    const int total_tokens = prefix_tokens + historical_tokens + recent_tokens;
    if (total_tokens <= 0) throw std::invalid_argument("OSCAR mixed attention is empty");
    const dim3 score_grid(kKVHeads,
                          static_cast<unsigned>((total_tokens + kTokenTile - 1) / kTokenTile), 1u);
    oscar_mixed_score_kernel<<<score_grid, kThreads, 0, stream>>>(
        q_rotated, prefix_k_bf16, prefix_tokens, historical_k_packed, historical_k_metadata,
        historical_tokens, recent_k_bf16, recent_tokens, recent_ring_head, total_tokens,
        attention_scale, scores);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_softmax_kernel<<<kQHeads, kThreads, 0, stream>>>(scores, total_tokens, softmax);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_av_kernel<<<kKVHeads, kThreads, 0, stream>>>(
        softmax, prefix_v_bf16, prefix_tokens, historical_v_packed, historical_v_metadata,
        historical_tokens, recent_v_bf16, recent_tokens, recent_ring_head, total_tokens, output);
    CUDA_CHECK(cudaGetLastError());
}

void oscar_int2_g128_mixed_attention_launch_batch_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, std::int32_t query_start, std::int32_t query_count,
    std::int32_t workspace_stride, float attention_scale, float* scores, float* softmax,
    float* output, cudaStream_t stream) {
    if (prefix_tokens < 0 || prefix_tokens > kPrefixTokens || historical_tokens < 0 ||
        recent_tokens < 0 || recent_tokens > kRecentTokens || recent_ring_head < 0 ||
        recent_ring_head >= kRecentTokens || query_start < 0 || query_count <= 0 ||
        query_count > kRecentTokens / 4 || workspace_stride <= 0) {
        throw std::invalid_argument("OSCAR mixed batched attention arguments are invalid");
    }
    const int total_tokens = prefix_tokens + historical_tokens + recent_tokens;
    if (total_tokens <= 0 || workspace_stride < total_tokens ||
        query_start + query_count > total_tokens) {
        throw std::invalid_argument("OSCAR mixed batched attention extent is invalid");
    }
    const dim3 score_grid(
        kKVHeads, static_cast<unsigned>((total_tokens + kTokenTile - 1) / kTokenTile),
        static_cast<unsigned>(query_count));
    oscar_mixed_score_batch_kernel<<<score_grid, kThreads, 0, stream>>>(
        q_rotated, prefix_k_bf16, prefix_tokens, historical_k_packed, historical_k_metadata,
        historical_tokens, recent_k_bf16, recent_tokens, recent_ring_head, total_tokens,
        query_start, workspace_stride, attention_scale, scores);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_softmax_batch_kernel<<<static_cast<unsigned>(query_count * kQHeads), kThreads, 0,
                                       stream>>>(scores, total_tokens, query_start, query_count,
                                                workspace_stride, softmax);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_av_batch_kernel<<<static_cast<unsigned>(query_count * kKVHeads), kThreads, 0,
                                  stream>>>(
        softmax, prefix_v_bf16, prefix_tokens, historical_v_packed, historical_v_metadata,
        historical_tokens, recent_v_bf16, recent_tokens, recent_ring_head, total_tokens,
        query_start, query_count, workspace_stride, output);
    CUDA_CHECK(cudaGetLastError());
}

void oscar_int2_g128_cache_write_bf16_launch(
    const float* rotated_k, const float* rotated_v, std::int32_t token_count,
    std::uint32_t logical_start, std::uint32_t final_recent_begin,
    std::uint32_t final_recent_head, const OscarInt2G128ResidentCacheView& cache,
    cudaStream_t stream) {
    if (rotated_k == nullptr || rotated_v == nullptr || token_count <= 0 ||
        final_recent_head >= kRecentTokens || cache.prefix_k_bf16 == nullptr ||
        cache.prefix_v_bf16 == nullptr || cache.recent_k_bf16 == nullptr ||
        cache.recent_v_bf16 == nullptr) {
        throw std::invalid_argument("OSCAR resident BF16 append arguments are invalid");
    }
    oscar_resident_write_bf16_kernel<<<static_cast<unsigned>(token_count), kThreads, 0, stream>>>(
        rotated_k, rotated_v, token_count, logical_start, final_recent_begin, final_recent_head,
        cache.prefix_k_bf16, cache.prefix_v_bf16, cache.recent_k_bf16, cache.recent_v_bf16);
    CUDA_CHECK(cudaGetLastError());
}

void oscar_int2_g128_cache_encode_launch(
    const float* rotated_k, const float* rotated_v, std::int32_t token_count,
    std::uint32_t logical_start, std::uint32_t old_context, std::uint32_t old_recent_begin,
    std::uint32_t old_recent_head, std::uint32_t final_recent_begin,
    std::uint32_t new_historical_tokens, std::uint32_t old_aging_tokens,
    const OscarInt2G128ResidentCacheView& cache, cudaStream_t stream) {
    if (rotated_k == nullptr || rotated_v == nullptr || token_count <= 0 ||
        old_recent_head >= kRecentTokens || cache.historical_k_packed == nullptr ||
        cache.historical_v_packed == nullptr || cache.historical_k_metadata == nullptr ||
        cache.historical_v_metadata == nullptr || cache.max_context <= 0) {
        throw std::invalid_argument("OSCAR resident INT2 append arguments are invalid");
    }
    const std::uint32_t work_items = (new_historical_tokens + old_aging_tokens) * kKVHeads;
    if (work_items == 0) return;
    constexpr std::size_t kSharedBytes =
        2ULL * kHeadDim * sizeof(float) + kHeadDim * sizeof(std::uint8_t) +
        kCodeBytes * 2ULL * sizeof(std::uint8_t) +
        kInt2MetadataItems * 2ULL * sizeof(float);
    oscar_resident_encode_kernel<<<static_cast<unsigned>(work_items), kThreads, kSharedBytes,
                                   stream>>>(
        rotated_k, rotated_v, token_count, logical_start, old_context, old_recent_begin,
        old_recent_head, final_recent_begin, new_historical_tokens, old_aging_tokens,
        cache.recent_k_bf16, cache.recent_v_bf16, cache.historical_k_packed,
        cache.historical_v_packed, cache.historical_k_metadata, cache.historical_v_metadata);
    CUDA_CHECK(cudaGetLastError());
}

OscarMixedKernelResources oscar_int2_g128_mixed_kernel_resources() {
    OscarMixedKernelResources result;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_mixed_score_kernel)));
    result.score_registers = attributes.numRegs;
    result.score_static_shared_bytes = attributes.sharedSizeBytes;
    result.score_max_threads = attributes.maxThreadsPerBlock;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_mixed_softmax_kernel)));
    result.softmax_registers = attributes.numRegs;
    result.softmax_static_shared_bytes = attributes.sharedSizeBytes;
    result.softmax_max_threads = attributes.maxThreadsPerBlock;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_mixed_av_kernel)));
    result.av_registers = attributes.numRegs;
    result.av_static_shared_bytes = attributes.sharedSizeBytes;
    result.av_max_threads = attributes.maxThreadsPerBlock;
    return result;
}

} // namespace ninfer::ops::detail
