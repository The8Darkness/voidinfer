#include "ops/softmax_attention/oscar_history/launch.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kGqa = 6;
constexpr int kHeadDim = 256;
constexpr int kGroups = 2;
constexpr int kGroupSize = 128;
constexpr int kCodeBytes = 64;
constexpr int kLevels = 3;
constexpr int kThreads = 256;
constexpr int kTokenTile = 256;
static_assert(kGroupSize == 128 && kLevels == 3 && kCodeBytes == 64,
              "OscarInt2G128 historical layout changed");

__device__ __forceinline__ float decode_symbol(const std::uint8_t* packed,
                                                const float* metadata, int dimension) {
    const int byte = dimension & 63;
    const int shift = (dimension >> 6) << 1;
    const std::uint8_t symbol = static_cast<std::uint8_t>((packed[byte] >> shift) & 3U);
    const int group = dimension >> 7;
    const float scale = metadata[group * 2];
    const float zero = metadata[group * 2 + 1];
    return (static_cast<float>(symbol) - zero) * scale;
}

// One block owns one KV head and one 256-token tile.  Each thread handles a
// token and computes all six GQA query-head scores.  K is decoded once per
// dimension per thread and is consumed immediately; no decoded K buffer exists.
__global__ void oscar_history_score_kernel(const float* __restrict__ q_rotated,
                                           const std::uint8_t* __restrict__ k_packed,
                                           const float* __restrict__ k_metadata,
                                           std::int32_t history, float attention_scale,
                                           float* __restrict__ scores) {
    __shared__ float q_shared[kGqa * kHeadDim];
    const int kv_head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y) * kTokenTile + static_cast<int>(threadIdx.x);

    for (int index = static_cast<int>(threadIdx.x); index < kGqa * kHeadDim; index += kThreads) {
        const int query_head = kv_head * kGqa + index / kHeadDim;
        q_shared[index] = q_rotated[query_head * kHeadDim + index % kHeadDim];
    }
    __syncthreads();
    if (token >= history) return;

    const std::size_t row = (static_cast<std::size_t>(token) * kKVHeads + kv_head);
    const std::uint8_t* packed = k_packed + row * kCodeBytes;
    const float* metadata = k_metadata + row * (kGroups * 2);
    float accum[kGqa] = {};
    for (int dimension = 0; dimension < kHeadDim; ++dimension) {
        const float key = decode_symbol(packed, metadata, dimension);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            accum[query_group] +=
                q_shared[query_group * kHeadDim + dimension] * key;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        scores[query_head * history + token] = accum[query_group] * attention_scale;
    }
}

__device__ __forceinline__ float block_reduce_max(float value) {
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

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float values[kThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    return values[0];
}

// One block computes one causal row.  The reductions are stable and entirely
// FP32; no approximating or fused low-precision softmax is used in this gate.
__global__ void oscar_history_softmax_kernel(const float* __restrict__ scores,
                                              std::int32_t history,
                                              float* __restrict__ softmax) {
    const int query_head = static_cast<int>(blockIdx.x);
    const float* row = scores + query_head * history;
    float maximum = -3.402823466e+38F;
    for (int token = static_cast<int>(threadIdx.x); token < history; token += kThreads) {
        maximum = fmaxf(maximum, row[token]);
    }
    maximum = block_reduce_max(maximum);

    float sum = 0.0F;
    for (int token = static_cast<int>(threadIdx.x); token < history; token += kThreads) {
        const float probability = expf(row[token] - maximum);
        softmax[query_head * history + token] = probability;
        sum += probability;
    }
    sum = block_reduce_sum(sum);
    for (int token = static_cast<int>(threadIdx.x); token < history; token += kThreads) {
        softmax[query_head * history + token] /= sum;
    }
}

// One block owns one KV head.  Each thread handles one output dimension and
// consumes each V code once for all six GQA query heads.  V is decoded directly
// into six FP32 AV accumulators and is never materialized as a decoded buffer.
__global__ void oscar_history_av_kernel(const float* __restrict__ softmax,
                                        const std::uint8_t* __restrict__ v_packed,
                                        const float* __restrict__ v_metadata,
                                        std::int32_t history, float* __restrict__ output) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    if (dimension >= kHeadDim) return;

    float accum[kGqa] = {};
    for (int token = 0; token < history; ++token) {
        const std::size_t row = (static_cast<std::size_t>(token) * kKVHeads + kv_head);
        const float value = decode_symbol(v_packed + row * kCodeBytes,
                                          v_metadata + row * (kGroups * 2), dimension);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            const int query_head = kv_head * kGqa + query_group;
            accum[query_group] += softmax[query_head * history + token] * value;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        output[query_head * kHeadDim + dimension] = accum[query_group];
    }
}

} // namespace

void oscar_int2_g128_history_attention_launch(
    const float* q_rotated, const std::uint8_t* k_packed, const float* k_metadata,
    const std::uint8_t* v_packed, const float* v_metadata, std::int32_t history,
    float attention_scale, float* scores, float* softmax, float* output, cudaStream_t stream) {
    if (history <= 0) return;
    const dim3 score_grid(kKVHeads,
                          static_cast<unsigned>((history + kTokenTile - 1) / kTokenTile), 1u);
    oscar_history_score_kernel<<<score_grid, kThreads, 0, stream>>>(
        q_rotated, k_packed, k_metadata, history, attention_scale, scores);
    CUDA_CHECK(cudaGetLastError());

    oscar_history_softmax_kernel<<<kQHeads, kThreads, 0, stream>>>(scores, history, softmax);
    CUDA_CHECK(cudaGetLastError());

    oscar_history_av_kernel<<<kKVHeads, kThreads, 0, stream>>>(
        softmax, v_packed, v_metadata, history, output);
    CUDA_CHECK(cudaGetLastError());
}

OscarHistoryKernelResources oscar_int2_g128_history_kernel_resources() {
    OscarHistoryKernelResources result;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_history_score_kernel)));
    result.score_registers = attributes.numRegs;
    result.score_static_shared_bytes = attributes.sharedSizeBytes;
    result.score_max_threads = attributes.maxThreadsPerBlock;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_history_softmax_kernel)));
    result.softmax_registers = attributes.numRegs;
    result.softmax_static_shared_bytes = attributes.sharedSizeBytes;
    result.softmax_max_threads = attributes.maxThreadsPerBlock;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_history_av_kernel)));
    result.av_registers = attributes.numRegs;
    result.av_static_shared_bytes = attributes.sharedSizeBytes;
    result.av_max_threads = attributes.maxThreadsPerBlock;
    return result;
}

} // namespace ninfer::ops::detail
