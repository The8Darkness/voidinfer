#pragma once
// Coupled sampled token selection for the coherent-device route.
//
// Semantics are the registered EXL3 sampler's (include/voidinfer/ops/sampling.h,
// with the presence penalty but no frequency penalty): logits of tokens already
// generated before the chosen index drop by presence_penalty; candidates are
// the top min(top_k, 20) penalized logits (top_k <= 0
// or >= 20 keeps 20) ordered by value with the lower token id winning ties;
// weights exp(logit/T - max); min_p drops the suffix below min_p * max weight;
// top_p keeps the shortest prefix whose cumulative weight reaches top_p of the
// pre-truncation weight; at least the best candidate stays. Temperature 0 (a
// penalized request without sampling) selects the best penalized logit.
//
// The draw is Gumbel-max over that support: token = argmax_j logit_j/T +
// G(seed, key, id_j), G a counter-based Gumbel(0,1) of the request seed, the
// absolute index `key` of the token being chosen and the token id. That is an
// exact draw from the renormalized support, and every selector of a request
// (decode rounds, VeriCache verification) uses the same noise for the same
// (key, token): wherever two of them see equal supports and logits they pick
// the same token, so the verifier's argmax check verifies samples exactly.
#include "exl3/greedy_packet.h"
#include "exl3/sampled_select.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <span>

namespace voidinfer::exl3 {


constexpr int kSampledCandidates=20;
// Round rows: row r selects index first + depth[r]. With inputs > 0, input[]
// are the rows' input tokens ([seed, chain drafts, sibling leaves]); row r's
// drafted history past the committed output is input[0..depth[r]) and input[r].
struct Exl3SampledKeys { int first; int depth[16]; int inputs; int input[16]; };
constexpr int kSampledThreads=256;

__device__ __forceinline__ std::uint64_t exl3_sampled_mix(std::uint64_t x) {
    x+=0x9E3779B97F4A7C15ull;
    x=(x^(x>>30))*0xBF58476D1CE4E5B9ull;
    x=(x^(x>>27))*0x94D049BB133111EBull;
    return x^(x>>31);
}
// Gumbel(0,1) of (seed, key, token): a pure function (graph-replay safe).
__device__ __forceinline__ float exl3_sampled_gumbel(std::uint64_t seed,int key,int token) {
    std::uint64_t h=exl3_sampled_mix(seed^(static_cast<std::uint64_t>(static_cast<unsigned>(key))*
        0xD1B54A32D192ED03ull));
    h=exl3_sampled_mix(h^(static_cast<std::uint64_t>(static_cast<unsigned>(token))*0x2545F4914F6CDD1Dull));
    const float u=(static_cast<float>(h>>40)+0.5f)*(1.0f/16777216.0f);   // (0,1)
    return -logf(-logf(u));
}
__device__ __forceinline__ bool exl3_sampled_better(float v,int i,float bv,int bi) {
    return v>bv || (v==bv && i<bi);
}
// Presence penalty of `token` for the row choosing absolute index `key`: the
// token was generated before it (committed output before `key`, or one of the
// row's `path` drafted tokens).
__device__ __forceinline__ float exl3_presence(const Exl3DeviceSampling& cfg,int key,
    const int* path,int path_count,int token) {
    if(cfg.presence_penalty==0.0f) return 0.0f;
    bool seen=cfg.first_occurrence && cfg.first_occurrence[token]<key;
    for(int i=0;i<path_count && !seen;++i) seen=path[i]==token;
    return seen?cfg.presence_penalty:0.0f;
}

// One CTA (kSampledThreads) per row. On return (all threads) the support
// cand_val/cand_idx[0..support) and the selected candidate are in shared
// memory. `bad` reports any non-finite logit of the row.
struct Exl3SampledRow {
    float cand_val[kSampledCandidates];
    int cand_idx[kSampledCandidates];
    float value[kSampledCandidates];     // logit/T + Gumbel of each support entry
    int support,selected;
    unsigned bad;
};
__device__ __forceinline__ int exl3_sampled_cap(const Exl3DeviceSampling& cfg,int vocabulary) {
    int cap=kSampledCandidates;
    if(cfg.top_k>0 && cfg.top_k<cap) cap=cfg.top_k;
    if(vocabulary<cap) cap=vocabulary;
    return cap;
}
// Truncation and coupled draw over the sorted candidates cand_val/cand_idx[0..cap)
// (one thread): exactly as sampling_normalize_support (ops/kernel/sampling_device.cuh);
// temperature 0 selects the best (penalized) candidate.
__device__ inline void exl3_sampled_draw(const Exl3DeviceSampling& cfg,int key,int cap,Exl3SampledRow& out) {
    if(!(cfg.temperature>0.0f)) {
        out.value[0]=out.cand_val[0];
        out.support=1;out.selected=0;
        return;
    }
    const float inv_temp=1.0f/cfg.temperature;
    const float m=out.cand_val[0]*inv_temp;
    float prob[kSampledCandidates];
    float sum=0.0f;
    for(int j=0;j<cap;++j) {prob[j]=__expf(out.cand_val[j]*inv_temp-m);sum+=prob[j];}
    const float min_p_thresh=cfg.min_p>0.0f?cfg.min_p*prob[0]:-1.0f;
    const bool top_p_active=cfg.top_p<1.0f;
    const float top_p_target=cfg.top_p*sum;
    float cum=0.0f;
    int support=0;
    for(int j=0;j<cap;++j) {
        if(min_p_thresh>=0.0f && prob[j]<min_p_thresh) break;
        cum+=prob[j];support=j+1;
        if(top_p_active && cum>=top_p_target) break;
    }
    if(support<1) support=1;
    int best=0;
    for(int j=0;j<support;++j) {
        out.value[j]=out.cand_val[j]*inv_temp+exl3_sampled_gumbel(cfg.seed,key,out.cand_idx[j]);
        if(out.value[j]>out.value[best]) best=j;
    }
    out.support=support;out.selected=best;
}
// Candidates and support use the penalized logits (logit - presence penalty);
// temperature 0 selects their argmax. `begin`/`end` limit the scan to one
// vocabulary slice (exl3_sampled_slice_kernel): `draw` false then leaves only
// the slice's sorted top-cap candidates (and `bad`) in `out`.
__device__ inline void exl3_sampled_select_row(const std::uint16_t* row,int vocabulary,
    const Exl3DeviceSampling& cfg,int key,const int* path,int path_count,Exl3SampledRow& out,
    float (*merge_val)[kSampledCandidates],int (*merge_idx)[kSampledCandidates],
    int begin=0,int end=-1,bool draw=true) {
    const int tid=static_cast<int>(threadIdx.x),lane=tid&31,warp=tid>>5;
    const int cap=exl3_sampled_cap(cfg,vocabulary);
    if(end<0) end=vocabulary;
    // Per-thread sorted top-cap (local memory), then a per-warp tournament.
    float val[kSampledCandidates];
    int idx[kSampledCandidates];
    #pragma unroll
    for(int j=0;j<kSampledCandidates;++j) {val[j]=-CUDART_INF_F;idx[j]=0x7fffffff;}
    unsigned bad=0;
    // A penalty only moves a logit by at most |presence_penalty|: tokens that
    // cannot enter the list even then skip the occurrence lookup.
    const float slack=cfg.presence_penalty<0.0f?-cfg.presence_penalty:0.0f;
    for(int v=begin+tid;v<end;v+=kSampledThreads) {
        float x=__half2float(__ushort_as_half(row[v]));
        if(!isfinite(x)) {bad=1;continue;}
        if(!exl3_sampled_better(x+slack,v,val[cap-1],idx[cap-1])) continue;
        x-=exl3_presence(cfg,key,path,path_count,v);
        if(!exl3_sampled_better(x,v,val[cap-1],idx[cap-1])) continue;
        int pos=cap-1;
        while(pos>0 && exl3_sampled_better(x,v,val[pos-1],idx[pos-1])) {
            val[pos]=val[pos-1];idx[pos]=idx[pos-1];--pos;
        }
        val[pos]=x;idx[pos]=v;
    }
    bad=__any_sync(0xffffffffu,bad)?1u:0u;
    // Warp merge: cap rounds of a warp argmax over each lane's head.
    int head=0;
    for(int r=0;r<cap;++r) {
        float bv=head<cap?val[head]:-CUDART_INF_F;
        int bi=head<cap?idx[head]:0x7fffffff;
        int owner=lane;
        for(int o=16;o;o>>=1) {
            const float ov=__shfl_xor_sync(0xffffffffu,bv,o);
            const int oi=__shfl_xor_sync(0xffffffffu,bi,o);
            const int oo=__shfl_xor_sync(0xffffffffu,owner,o);
            if(exl3_sampled_better(ov,oi,bv,bi)) {bv=ov;bi=oi;owner=oo;}
        }
        if(lane==0) {merge_val[warp][r]=bv;merge_idx[warp][r]=bi;}
        if(lane==owner) ++head;
    }
    __shared__ unsigned bad_s;
    if(tid==0) bad_s=0;
    __syncthreads();
    if(lane==0 && bad) atomicOr(&bad_s,1u);
    // Block merge of the per-warp lists by warp 0 (lanes 0..7 hold a list head).
    if(warp==0) {
        constexpr int kWarps=kSampledThreads/32;
        int h=0;
        for(int r=0;r<cap;++r) {
            float bv=lane<kWarps && h<cap?merge_val[lane][h]:-CUDART_INF_F;
            int bi=lane<kWarps && h<cap?merge_idx[lane][h]:0x7fffffff;
            int owner=lane;
            for(int o=16;o;o>>=1) {
                const float ov=__shfl_xor_sync(0xffffffffu,bv,o);
                const int oi=__shfl_xor_sync(0xffffffffu,bi,o);
                const int oo=__shfl_xor_sync(0xffffffffu,owner,o);
                if(exl3_sampled_better(ov,oi,bv,bi)) {bv=ov;bi=oi;owner=oo;}
            }
            if(lane==0) {out.cand_val[r]=bv;out.cand_idx[r]=bi;}
            if(lane==owner) ++h;
        }
    }
    __syncthreads();
    if(tid==0) {
        if(draw) exl3_sampled_draw(cfg,key,cap,out);
        out.bad=bad_s;
    }
    __syncthreads();
}

// Decode-round packet rows (at most 16): row r selects the token at absolute
// index first + depth[r] (verification-tree rows carry their logical depth;
// depth[r] = r for a plain chain). Each row's vocabulary is scanned by
// kSampledSlices CTAs (one CTA per row took ~0.6 ms); the merge of the slices'
// sorted top-cap lists is the row's top-cap list, then one thread draws.
constexpr int kSampledSlices=32;
struct Exl3SampledSliceScratch {
    float* value;     // [16][kSampledSlices][kSampledCandidates]
    int* index;
    unsigned* bad;    // [16][kSampledSlices]
    static constexpr std::size_t kRows=16;
    static constexpr std::size_t bytes() {
        return kRows*kSampledSlices*kSampledCandidates*(sizeof(float)+sizeof(int))+kRows*kSampledSlices*sizeof(unsigned);
    }
    static Exl3SampledSliceScratch carve(void* storage) {
        auto* base=static_cast<unsigned char*>(storage);
        Exl3SampledSliceScratch s;
        s.value=reinterpret_cast<float*>(base);
        s.index=reinterpret_cast<int*>(base+kRows*kSampledSlices*kSampledCandidates*sizeof(float));
        s.bad=reinterpret_cast<unsigned*>(base+kRows*kSampledSlices*kSampledCandidates*(sizeof(float)+sizeof(int)));
        return s;
    }
};
__device__ __forceinline__ int exl3_sampled_row_path(const Exl3SampledKeys& keys,int r,int (&path)[17]) {
    int count=0;
    if(keys.inputs>r) {
        for(int j=0;j<keys.depth[r] && j<keys.inputs;++j) path[count++]=keys.input[j];
        path[count++]=keys.input[r];
    }
    return count;
}
// grid (kSampledSlices, rows): the slice's sorted top-cap candidates.
static __global__ void __launch_bounds__(kSampledThreads) exl3_sampled_slice_kernel(
    const std::uint16_t* scores,int vocabulary,int stride,Exl3DeviceSampling cfg,Exl3SampledKeys keys,
    Exl3SampledSliceScratch scratch) {
    __shared__ Exl3SampledRow row_s;
    __shared__ float merge_val[kSampledThreads/32][kSampledCandidates];
    __shared__ int merge_idx[kSampledThreads/32][kSampledCandidates];
    const int slice=static_cast<int>(blockIdx.x),r=static_cast<int>(blockIdx.y);
    int path[17];
    const int path_count=exl3_sampled_row_path(keys,r,path);
    const int span=(vocabulary+kSampledSlices-1)/kSampledSlices;
    const int begin=slice*span,end=min(vocabulary,begin+span);
    exl3_sampled_select_row(scores+static_cast<std::size_t>(r)*stride,vocabulary,cfg,
        keys.first+keys.depth[r],path,path_count,row_s,merge_val,merge_idx,begin,end,false);
    const int cap=exl3_sampled_cap(cfg,vocabulary);
    const std::size_t at=(static_cast<std::size_t>(r)*kSampledSlices+slice)*kSampledCandidates;
    if(static_cast<int>(threadIdx.x)<cap) {
        scratch.value[at+threadIdx.x]=row_s.cand_val[threadIdx.x];
        scratch.index[at+threadIdx.x]=row_s.cand_idx[threadIdx.x];
    }
    if(threadIdx.x==0) scratch.bad[static_cast<std::size_t>(r)*kSampledSlices+slice]=row_s.bad;
}
// grid (rows), one warp: merges the slices' lists (cap rounds of a warp
// argmax over the 32 list heads), then draws.
static __global__ void __launch_bounds__(32) exl3_sampled_merge_kernel(
    int vocabulary,Exl3DeviceSampling cfg,Exl3SampledKeys keys,Exl3SampledSliceScratch scratch,
    std::uint64_t serial,Exl3GreedyRow* output) {
    __shared__ Exl3SampledRow row_s;
    const int r=static_cast<int>(blockIdx.x),lane=static_cast<int>(threadIdx.x);
    const int cap=exl3_sampled_cap(cfg,vocabulary);
    const std::size_t at=(static_cast<std::size_t>(r)*kSampledSlices+lane)*kSampledCandidates;
    int head=0;
    for(int k=0;k<cap;++k) {
        float bv=head<cap?scratch.value[at+head]:-CUDART_INF_F;
        int bi=head<cap?scratch.index[at+head]:0x7fffffff;
        int owner=lane;
        for(int o=16;o;o>>=1) {
            const float ov=__shfl_xor_sync(0xffffffffu,bv,o);
            const int oi=__shfl_xor_sync(0xffffffffu,bi,o);
            const int oo=__shfl_xor_sync(0xffffffffu,owner,o);
            if(exl3_sampled_better(ov,oi,bv,bi)) {bv=ov;bi=oi;owner=oo;}
        }
        if(lane==0) {row_s.cand_val[k]=bv;row_s.cand_idx[k]=bi;}
        if(lane==owner) ++head;
    }
    const unsigned bad=__any_sync(0xffffffffu,scratch.bad[static_cast<std::size_t>(r)*kSampledSlices+lane]!=0u)?1u:0u;
    __syncwarp();
    if(lane==0) {
        exl3_sampled_draw(cfg,keys.first+keys.depth[r],cap,row_s);
        output[r].serial=serial;
        output[r].token=row_s.cand_idx[row_s.selected];
        output[r].nonfinite=bad;
    }
}

// Support size of candidates sorted by value (descending), as in
// exl3_sampled_select_row.
__device__ inline int exl3_sampled_support(const float* value,int cap,const Exl3DeviceSampling& cfg) {
    const float inv_temp=1.0f/cfg.temperature;
    const float m=value[0]*inv_temp;
    float prob[kSampledCandidates];
    float sum=0.0f;
    for(int j=0;j<cap;++j) {prob[j]=__expf(value[j]*inv_temp-m);sum+=prob[j];}
    const float min_p_thresh=cfg.min_p>0.0f?cfg.min_p*prob[0]:-1.0f;
    float cum=0.0f;
    int support=0;
    for(int j=0;j<cap;++j) {
        if(min_p_thresh>=0.0f && prob[j]<min_p_thresh) break;
        cum+=prob[j];support=j+1;
        if(cfg.top_p<1.0f && cum>=cfg.top_p*sum) break;
    }
    return support<1?1:support;
}

// VeriCache rows: the selected token and the gap, in logit units (T times the
// perturbed-value gap), by which next[r] trails it. A next[r] outside the
// support has an infinite gap unless raising its logit by `tolerance` brings
// it into the support of the exact sampler: then its gap is measured against
// the best other token of that support.
static __global__ void __launch_bounds__(kSampledThreads) exl3_sampled_score_kernel(
    const std::uint16_t* scores,int vocabulary,Exl3DeviceSampling cfg,int first_key,float tolerance,
    const std::int64_t* next,std::int64_t* chosen,float* gap) {
    __shared__ Exl3SampledRow row_s;
    __shared__ float merge_val[kSampledThreads/32][kSampledCandidates];
    __shared__ int merge_idx[kSampledThreads/32][kSampledCandidates];
    const int r=static_cast<int>(blockIdx.x);
    const int key=first_key+r;
    exl3_sampled_select_row(scores+static_cast<std::size_t>(r)*vocabulary,vocabulary,cfg,key,
        nullptr,0,row_s,merge_val,merge_idx);
    if(threadIdx.x==0) {
        const int token=row_s.cand_idx[row_s.selected];
        chosen[r]=token;
        const std::int64_t want=next[r];
        float g=0.0f;
        if(want>=0 && want!=token && !(cfg.temperature>0.0f)) {
            // Penalized argmax: next[r]'s exact penalized logit gap.
            const float own=__half2float(__ushort_as_half(
                scores[static_cast<std::size_t>(r)*vocabulary+static_cast<std::size_t>(want)]))-
                exl3_presence(cfg,key,nullptr,0,static_cast<int>(want));
            g=fmaxf(0.0f,row_s.cand_val[row_s.selected]-own);
        } else if(want>=0 && want!=token) {
            g=CUDART_INF_F;
            int cap=kSampledCandidates;
            if(cfg.top_k>0 && cfg.top_k<cap) cap=cfg.top_k;
            int rank=-1;
            for(int j=0;j<cap;++j) if(row_s.cand_idx[j]==want) {rank=j;break;}
            if(rank>=0 && rank<row_s.support) {
                g=(row_s.value[row_s.selected]-row_s.value[rank])*cfg.temperature;
            } else if(rank>=0 && tolerance>0.0f) {
                // Raise next[r] by the tolerance and re-rank it.
                float value[kSampledCandidates];
                int index[kSampledCandidates];
                const float boosted=row_s.cand_val[rank]+tolerance;
                int at=0;
                for(int j=0;j<cap;++j) {
                    if(j==rank) continue;
                    value[at]=row_s.cand_val[j];index[at]=row_s.cand_idx[j];++at;
                }
                int pos=at;
                while(pos>0 && boosted>value[pos-1]) {value[pos]=value[pos-1];index[pos]=index[pos-1];--pos;}
                value[pos]=boosted;index[pos]=static_cast<int>(want);
                const int support=exl3_sampled_support(value,cap,cfg);
                if(pos<support) {
                    const float inv_temp=1.0f/cfg.temperature;
                    float best=-CUDART_INF_F;
                    for(int j=0;j<support;++j) if(j!=pos)
                        best=fmaxf(best,value[j]*inv_temp+exl3_sampled_gumbel(cfg.seed,key,index[j]));
                    const float own=row_s.cand_val[rank]*inv_temp+exl3_sampled_gumbel(cfg.seed,key,static_cast<int>(want));
                    g=fmaxf(0.0f,(best-own)*cfg.temperature);
                }
            }
        }
        gap[r]=g;
    }
}

inline void exl3_launch_sampled_packet(int rows,cudaStream_t stream,const std::uint16_t* scores,
    int vocabulary,int stride,const Exl3DeviceSampling& cfg,int first_key,
    std::span<const int> depths,std::span<const int> inputs,std::uint64_t serial,Exl3GreedyRow* output,
    Exl3SampledSliceScratch scratch) {
    Exl3SampledKeys keys{};
    keys.first=first_key;
    const bool tree=static_cast<int>(depths.size())==rows;
    for(int r=0;r<rows && r<16;++r) keys.depth[r]=tree?depths[static_cast<std::size_t>(r)]:r;
    if(static_cast<int>(inputs.size())==rows) {
        keys.inputs=rows<16?rows:16;
        for(int r=0;r<keys.inputs;++r) keys.input[r]=inputs[static_cast<std::size_t>(r)];
    }
    exl3_sampled_slice_kernel<<<dim3(kSampledSlices,rows),kSampledThreads,0,stream>>>(scores,vocabulary,stride,
        cfg,keys,scratch);
    exl3_sampled_merge_kernel<<<rows,32,0,stream>>>(vocabulary,cfg,keys,scratch,serial,output);
}

} // namespace voidinfer::exl3
