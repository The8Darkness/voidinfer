#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops::detail {

struct Fp8IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Fp8ContiguousOutput {
    static constexpr bool kNeedsBlockReduce = false;

    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + parent_row;
        store_vec(destination, values);
    }
};

// Greedy-only epilogue for the Qwen3.8 vocabulary head. The MMA mainloop still computes the
// exact same FP32 dot products and rounds each value to BF16 before comparing it, matching the
// public linear+argmax path. Each row CTA reduces its sixteen vocabulary rows in shared memory;
// this avoids materializing [248320,T] BF16 logits and then rereading them for argmax.
struct Fp8ArgmaxOutput {
    static constexpr bool kNeedsBlockReduce = true;
    static constexpr std::int32_t kRowsPerCta = 16;

    std::int32_t valid_rows;
    float* shared_values;
    __nv_bfloat16* partial_values;
    std::int32_t* partial_indices;
    std::int32_t partial_rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const std::int32_t local_row = parent_row & (kRowsPerCta - 1);
        shared_values[token * kRowsPerCta + local_row] =
            __bfloat162float(__float2bfloat16_rn(value));
    }

    __device__ __forceinline__ void finish(std::int32_t row0, std::int32_t active_tokens) const {
        const int lane = threadIdx.x & 31;
        for (std::int32_t token = lane; token < active_tokens; token += 32) {
            const std::int32_t first_row = row0;
            if (first_row >= valid_rows) {
                partial_values[token * partial_rows + blockIdx.x] =
                    __float2bfloat16_rn(-CUDART_INF_F);
                partial_indices[token * partial_rows + blockIdx.x] = INT_MAX;
                continue;
            }

            float best_value        = shared_values[token * kRowsPerCta];
            std::int32_t best_index = first_row;
#pragma unroll
            for (std::int32_t local_row = 1; local_row < kRowsPerCta; ++local_row) {
                const std::int32_t index = row0 + local_row;
                if (index >= valid_rows) { break; }
                const float value = shared_values[token * kRowsPerCta + local_row];
                if (value > best_value || (value == best_value && index < best_index)) {
                    best_value = value;
                    best_index = index;
                }
            }
            partial_values[token * partial_rows + blockIdx.x] =
                __float2bfloat16_rn(best_value);
            partial_indices[token * partial_rows + blockIdx.x] = best_index;
        }
    }
};

} // namespace ninfer::ops::detail
