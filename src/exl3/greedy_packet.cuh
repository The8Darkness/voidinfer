#pragma once
#include "exl3/greedy_packet.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace ninfer::exl3 {
// Read represented FP16 scores, never pre-cast head accumulators. No FP
// arithmetic reduction: only comparisons, with the lowest index winning ties.
// Each row scans its complete logical vocabulary; stride padding is ignored.
template<bool Warp=false>
static __global__ void exl3_greedy_packet_kernel(const std::uint16_t* scores,
    int vocabulary, int stride, std::uint64_t serial, Exl3GreedyRow* output) {
    __shared__ float maxima[Warp?8:256];
    __shared__ int indices[Warp?8:256];
    __shared__ unsigned invalid[Warp?8:256];
    const int t=threadIdx.x;
    float best=-CUDART_INF_F;
    int index=vocabulary;
    unsigned bad=0;
    for(int i=t;i<vocabulary;i+=256) {
        const float value=__half2float(__ushort_as_half(scores[static_cast<std::size_t>(blockIdx.x)*stride+i]));
        if(!isfinite(value)) {bad=1;continue;}
        if(value>best || (value==best && i<index)) {best=value;index=i;}
    }
    if constexpr(Warp) {
        constexpr unsigned mask=0xffffffffu;
        for(int offset=16;offset;offset>>=1) {
            const float other=__shfl_down_sync(mask,best,offset);
            const int oi=__shfl_down_sync(mask,index,offset);
            bad|=__shfl_down_sync(mask,bad,offset);
            if(other>best || (other==best && oi<index)) {best=other;index=oi;}
        }
        if((t&31)==0) {maxima[t>>5]=best;indices[t>>5]=index;invalid[t>>5]=bad;}
        __syncthreads();
        if(t<32) {
            best=t<8?maxima[t]:-CUDART_INF_F;
            index=t<8?indices[t]:vocabulary;
            bad=t<8?invalid[t]:0;
            for(int offset=16;offset;offset>>=1) {
                const float other=__shfl_down_sync(mask,best,offset);
                const int oi=__shfl_down_sync(mask,index,offset);
                bad|=__shfl_down_sync(mask,bad,offset);
                if(other>best || (other==best && oi<index)) {best=other;index=oi;}
            }
            if(t==0) {
                output[blockIdx.x].serial=serial;
                output[blockIdx.x].token=index;
                output[blockIdx.x].nonfinite=bad;
            }
        }
    } else {
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
}

inline void exl3_launch_greedy_packet(bool warp,int rows,cudaStream_t stream,
    const std::uint16_t* scores,int vocabulary,int stride,std::uint64_t serial,
    Exl3GreedyRow* output) {
    if(warp)exl3_greedy_packet_kernel<true><<<rows,256,0,stream>>>(scores,vocabulary,stride,serial,output);
    else exl3_greedy_packet_kernel<false><<<rows,256,0,stream>>>(scores,vocabulary,stride,serial,output);
}
} // namespace ninfer::exl3
