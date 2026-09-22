#pragma once
#include "exl3/greedy_packet.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace ninfer::exl3 {
// Read represented FP16 scores, never pre-cast head accumulators. No FP
// arithmetic reduction: only comparisons, with the lowest index winning ties.
// Each row scans its complete logical vocabulary; stride padding is ignored.
static __global__ void exl3_greedy_packet_kernel(const std::uint16_t* scores,
    int vocabulary, int stride, std::uint64_t serial, Exl3GreedyRow* output) {
    __shared__ float maxima[256];
    __shared__ int indices[256];
    __shared__ unsigned invalid[256];
    const int t=threadIdx.x;
    float best=-CUDART_INF_F;
    int index=vocabulary;
    unsigned bad=0;
    for(int i=t;i<vocabulary;i+=256) {
        const float value=__half2float(__ushort_as_half(scores[static_cast<std::size_t>(blockIdx.x)*stride+i]));
        if(!isfinite(value)) {bad=1;continue;}
        if(value>best || (value==best && i<index)) {best=value;index=i;}
    }
    maxima[t]=best;indices[t]=index;invalid[t]=bad;
    __syncthreads();
    for(int width=128;width; width/=2) {
        if(t<width) {
            const float other=maxima[t+width];
            const int oi=indices[t+width];
            if(other>maxima[t] || (other==maxima[t] && oi<indices[t])) {
                maxima[t]=other;indices[t]=oi;
            }
            invalid[t]|=invalid[t+width];
        }
        __syncthreads();
    }
    if(t==0) {
        output[blockIdx.x].serial=serial;
        output[blockIdx.x].token=indices[0];
        output[blockIdx.x].nonfinite=invalid[0];
    }
}
} // namespace ninfer::exl3
