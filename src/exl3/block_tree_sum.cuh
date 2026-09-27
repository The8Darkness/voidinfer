#pragma once

#include <cuda_runtime.h>

namespace ninfer::exl3 {

// Replays the shared-memory pairwise tree
//     for(stride=B/2; stride>=1; stride>>=1) s[l]+=s[l+stride] (l<stride)
// with the identical FP32 operands and order: strides B/2..32 in warp-0
// registers, strides 16..1 as shfl_down. Two block barriers instead of
// log2(B)+1. `shared` needs B floats; every thread receives s[0].
template <int B>
__device__ __forceinline__ float block_tree_sum_exact(float sum, float* shared, int lane) {
    static_assert(B == 256 || B == 512, "block tree width");
    shared[lane] = sum;
    __syncthreads();
    if (lane < 32) {
        float v[B / 32];
        #pragma unroll
        for (int k = 0; k < B / 32; ++k) v[k] = shared[lane + 32 * k];
        #pragma unroll
        for (int s = B / 64; s >= 1; s >>= 1) {
            #pragma unroll
            for (int k = 0; k < s; ++k) v[k] += v[k + s];
        }
        float value = v[0];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

} // namespace ninfer::exl3
