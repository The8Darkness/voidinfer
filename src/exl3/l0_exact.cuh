#pragma once
// L0 OSCAR exact history (VeriCache verifier and lossless ingestion). Included
// by l0_oscar_attention.cuh after l0_hot.cuh; shares its tile layout and MMA
// helpers.
//
// Attention of up to `rows` query rows over the history [kSink, history_end)
// read directly from the authoritative FP16 L2 planes ([row][4][256], mapped
// host memory), in the original basis. CTA = (group of 8 query rows, KV head,
// history split), 12 warps as in l0_hot_cta; 32-key tiles stream through a
// cp.async double buffer. Row groups are the fastest grid dimension, so the
// groups reading one split's keys run side by side and share them in L2.
// Writes the split's normalized numerator out[split][row][qh][256] and
// (max, denominator) stats[split][row][qh][2], natural-log max.
constexpr std::size_t l0_exact_smem_bytes() {
    return (static_cast<std::size_t>(kL0Queries)*kL0HotStride+4*kL0HotTile*kHeadDim)*sizeof(half)+
        static_cast<std::size_t>(kL0Queries)*kL0HotTile*sizeof(float);
}
__global__ void __launch_bounds__(kL0HistoryThreads,1) l0_exact_history_kernel(
    const std::uint16_t* q_bits,const std::uint16_t* k_src,const std::uint16_t* v_src,
    int src_first,float* out,float* stats,int rows,int key_begin,int key_end,int position,
    const int* position_device,int capacity,int splits) {
    constexpr int H=kFastFusedFlashHeads;
    constexpr float kLog2Scale=0.0625f*1.4426950408889634f;
    constexpr int kThreads=kL0HistoryThreads;
    extern __shared__ __align__(16) unsigned char l0_exact_smem[];
    half* qs=reinterpret_cast<half*>(l0_exact_smem);
    half* kv_base=qs+kL0Queries*kL0HotStride;   // [buffer][K|V][32][256]
    float* scores=reinterpret_cast<float*>(kv_base+4*kL0HotTile*kHeadDim);   // [48][32]
    const int split=static_cast<int>(blockIdx.z),kv=static_cast<int>(blockIdx.y);
    const int row0=static_cast<int>(blockIdx.x)*(kL0Queries/H);
    // key_end < 0: the live history end of the (device) position.
    const int history_end=key_end>=0?key_end:
        min(l0::history_end(position_device?*position_device:position),capacity);
    const int tid=static_cast<int>(threadIdx.x),warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
    const int mtile=warp%3,vq=warp/3;
    const int keys=history_end>key_begin?history_end-key_begin:0;
    const int span=((keys+splits-1)/splits+kL0HotTile-1)/kL0HotTile*kL0HotTile;
    const int first=key_begin+split*span,last=min(first+span,history_end);
    const int tiles=last>first?(last-first+kL0HotTile-1)/kL0HotTile:0;
    const auto fetch=[&](int tile) {
        if(tile<tiles) {
            half* kd=kv_base+(tile&1)*2*kL0HotTile*kHeadDim;
            half* vd=kd+kL0HotTile*kHeadDim;
            const int key0=first+tile*kL0HotTile;
            for(int i=tid;i<2*kL0HotTile*(kHeadDim/8);i+=kThreads) {
                const int kind=i/(kL0HotTile*(kHeadDim/8)),k=(i/(kHeadDim/8))%kL0HotTile,c=i%(kHeadDim/8);
                const int key=key0+k;
                const bool valid=key<last;
                const std::size_t at=(static_cast<std::size_t>((valid?key:first)-src_first)*kKVHeads+kv)*kHeadDim+c*8;
                l0_hot_cp16((kind?vd:kd)+l0_hot_at(k,c*8),(kind?v_src:k_src)+at,valid);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };
    for(int i=tid;i<kL0Queries*(kHeadDim/8);i+=kThreads) {
        const int v=i/(kHeadDim/8),c=i%(kHeadDim/8),row=row0+v/H;
        l0_hot_cp16(qs+v*kL0HotStride+c*8,q_bits+
            (static_cast<std::size_t>(row<rows?row:0)*kQHeads+kv*H+v%H)*kHeadDim+c*8,row<rows);
    }
    fetch(0);
    const int v0=mtile*16+g,v1=v0+8;
    float mx[2]={-INFINITY,-INFINITY},den[2]={0.f,0.f};
    float acc[8][4];
    #pragma unroll
    for(int n=0;n<8;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.f;
    for(int tile=0;tile<tiles;++tile) {
        fetch(tile+1);
        asm volatile("cp.async.wait_group 1;\n");
        __syncthreads();
        const half* ks=kv_base+(tile&1)*2*kL0HotTile*kHeadDim;
        const half* vs=ks+kL0HotTile*kHeadDim;
        const int key0=first+tile*kL0HotTile;
        // QK once per (m-tile, 8-key quarter): warp vq scores keys 8 vq..8 vq+7;
        // the tile's scores meet in shared memory for every value quarter.
        {
            // FP16-accumulated 64-dim partials (twice the FP32-accumulate rate),
            // each folded into FP32 as in the FA2 prefill kernel.
            float c[4]={0.f,0.f,0.f,0.f};
            #pragma unroll
            for(int group=0;group<kHeadDim;group+=64) {
                unsigned h[2]={0u,0u};
                #pragma unroll
                for(int kk=group;kk<group+64;kk+=32) {
                    unsigned a0[4],a1[4],b[4];
                    reg_attn_ldmatrix_x4(a0,qs+(mtile*16+(lane&15))*kL0HotStride+kk+(lane>>4)*8);
                    reg_attn_ldmatrix_x4(a1,qs+(mtile*16+(lane&15))*kL0HotStride+kk+16+(lane>>4)*8);
                    reg_attn_ldmatrix_x4(b,ks+l0_hot_at(vq*8+(lane&7),kk+(lane>>3)*8));
                    mma_h(h,a0,b[0],b[1]);
                    mma_h(h,a1,b[2],b[3]);
                }
                const float2 lo=__half22float2(*reinterpret_cast<const __half2*>(&h[0]));
                const float2 hi=__half22float2(*reinterpret_cast<const __half2*>(&h[1]));
                c[0]+=lo.x; c[1]+=lo.y; c[2]+=hi.x; c[3]+=hi.y;
            }
            const int key=vq*8+2*t;
            *reinterpret_cast<float2*>(scores+(mtile*16+g)*kL0HotTile+key)=make_float2(c[0],c[1]);
            *reinterpret_cast<float2*>(scores+(mtile*16+g+8)*kL0HotTile+key)=make_float2(c[2],c[3]);
        }
        __syncthreads();
        float s[kL0HotTile/8][4];
        #pragma unroll
        for(int n=0;n<kL0HotTile/8;++n) {
            const float2 lo=*reinterpret_cast<const float2*>(scores+v0*kL0HotTile+8*n+2*t);
            const float2 hi=*reinterpret_cast<const float2*>(scores+v1*kL0HotTile+8*n+2*t);
            s[n][0]=lo.x; s[n][1]=lo.y; s[n][2]=hi.x; s[n][3]=hi.y;
        }
        float tmax[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(int n=0;n<kL0HotTile/8;++n)
            #pragma unroll
            for(int e=0;e<4;++e) {
                const int r=e>>1,key=8*n+2*t+(e&1);
                const float x=key0+key<last?s[n][e]*kLog2Scale:-INFINITY;
                s[n][e]=x;
                tmax[r]=fmaxf(tmax[r],x);
            }
        #pragma unroll
        for(int r=0;r<2;++r) {
            tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffU,tmax[r],1));
            tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffU,tmax[r],2));
            const float next=fmaxf(mx[r],tmax[r]);
            const float alpha=next==-INFINITY||mx[r]==-INFINITY?0.f:exp2f(mx[r]-next);
            mx[r]=next;
            den[r]*=alpha;
            #pragma unroll
            for(int n=0;n<8;++n) { acc[n][2*r]*=alpha; acc[n][2*r+1]*=alpha; }
        }
        float rs[2]={0.f,0.f};
        #pragma unroll
        for(int n=0;n<kL0HotTile/8;++n)
            #pragma unroll
            for(int e=0;e<4;++e) {
                const int r=e>>1;
                const float p=mx[r]==-INFINITY?0.f:exp2f(s[n][e]-mx[r]);
                s[n][e]=p;
                rs[r]+=p;
            }
        #pragma unroll
        for(int r=0;r<2;++r) {
            rs[r]+=__shfl_xor_sync(0xffffffffU,rs[r],1);
            rs[r]+=__shfl_xor_sync(0xffffffffU,rs[r],2);
            den[r]+=rs[r];
        }
        // PV: the tile's FP16-accumulated partial is folded into FP32.
        unsigned pv[8][2];
        #pragma unroll
        for(int n=0;n<8;++n) pv[n][0]=pv[n][1]=0u;
        #pragma unroll
        for(int kk=0;kk<kL0HotTile/16;++kk) {
            const auto pack=[](float lo,float hi) {
                const __half2 h=__floats2half2_rn(lo,hi);
                return *reinterpret_cast<const unsigned*>(&h);
            };
            const unsigned a[4]={pack(s[2*kk][0],s[2*kk][1]),pack(s[2*kk][2],s[2*kk][3]),
                                 pack(s[2*kk+1][0],s[2*kk+1][1]),pack(s[2*kk+1][2],s[2*kk+1][3])};
            #pragma unroll
            for(int pair=0;pair<4;++pair) {
                unsigned b[4];
                reg_attn_ldmatrix_x4_trans(b,vs+l0_hot_at(kk*16+(lane&7)+((lane>>3)&1)*8,
                    64*vq+pair*16+(lane>>4)*8));
                mma_h(pv[2*pair],a,b[0],b[1]);
                mma_h(pv[2*pair+1],a,b[2],b[3]);
            }
        }
        #pragma unroll
        for(int n=0;n<8;++n) {
            const float2 lo=__half22float2(*reinterpret_cast<const __half2*>(&pv[n][0]));
            const float2 hi=__half22float2(*reinterpret_cast<const __half2*>(&pv[n][1]));
            acc[n][0]+=lo.x; acc[n][1]+=lo.y; acc[n][2]+=hi.x; acc[n][3]+=hi.y;
        }
        __syncthreads();   // the buffer is refilled two tiles on
    }
    asm volatile("cp.async.wait_all;\n");
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int v=r?v1:v0,row=row0+v/H,head=v%H;
        if(row>=rows) continue;
        const std::size_t at=(static_cast<std::size_t>(split)*rows+row)*kQHeads+kv*H+head;
        const float inv=den[r]>0.f?1.f/den[r]:0.f;
        float* dst=out+at*kHeadDim+64*vq+2*t;
        #pragma unroll
        for(int n=0;n<8;++n)
            *reinterpret_cast<float2*>(dst+8*n)=make_float2(acc[n][2*r]*inv,acc[n][2*r+1]*inv);
        if(t==0 && vq==0) {
            stats[2*at]=mx[r]==-INFINITY?-INFINITY:mx[r]*0.6931471805599453f;
            stats[2*at+1]=den[r];
        }
    }
}

// Merges the `splits` exact-history partials of one (row, query head) into a
// normalized original-basis numerator `numer` [rows][24][256]. The merged
// (max, denominator) goes to the prefill split-stats layout (`split_stats`,
// slot `slot`) and/or the verifier fused-flash history slot (`history_slot`,
// whose rotated numerator `rotated` is zeroed: l0_rotate_kernel adds `numer`).
__global__ void __launch_bounds__(256) l0_exact_merge_kernel(const float* out,const float* stats,
    int splits,int rows,float* numer,float* split_stats,int slot,float* history_slot,__half* rotated,
    bool accumulate=false) {
    constexpr int H=kFastFusedFlashHeads;
    const int vector=static_cast<int>(blockIdx.x),d=static_cast<int>(threadIdx.x);
    const int row=vector/kQHeads,qh=vector%kQHeads;
    const std::size_t split_vectors=static_cast<std::size_t>(rows)*kQHeads;
    // accumulate: (numer, split_stats slot) already hold earlier pieces' merge.
    const std::size_t prior=(static_cast<std::size_t>(slot)*rows*kQHeads+vector)*2;
    const float prior_m=accumulate?split_stats[prior]:-INFINITY;
    const float prior_den=accumulate?split_stats[prior+1]:0.f;
    float m=prior_den>0.f?prior_m:-INFINITY;
    for(int s=0;s<splits;++s) {
        const float* st=stats+2*(s*split_vectors+vector);
        if(st[1]>0.f) m=fmaxf(m,st[0]);
    }
    float den=0.f,num=0.f;
    if(prior_den>0.f) {
        den=prior_den*expf(prior_m-m);
        num=den*numer[static_cast<std::size_t>(vector)*kHeadDim+d];
    }
    if(m>-INFINITY)
        for(int s=0;s<splits;++s) {
            const float* st=stats+2*(s*split_vectors+vector);
            if(st[1]<=0.f) continue;
            const float w=st[1]*expf(st[0]-m);
            den+=w;
            num=fmaf(w,out[(s*split_vectors+vector)*kHeadDim+d],num);
        }
    numer[static_cast<std::size_t>(vector)*kHeadDim+d]=den>0.f?num/den:0.f;
    if(rotated) rotated[static_cast<std::size_t>(vector)*kHeadDim+d]=__float2half_rn(0.f);
    if(d==0) {
        if(split_stats) {
            const std::size_t dst=(static_cast<std::size_t>(slot)*rows*kQHeads+vector)*2;
            split_stats[dst]=m;
            split_stats[dst+1]=den;
        }
        if(history_slot) {
            float* slot_ptr=history_slot+(static_cast<std::size_t>(row)*kKVHeads+qh/H)*kFastFusedFlashStride;
            slot_ptr[kFastFusedFlashValues+qh%H]=m;
            slot_ptr[kFastFusedFlashValues+H+qh%H]=den;
        }
    }
}

inline void l0_launch_exact_history(const void* q,const std::uint16_t* k_src,const std::uint16_t* v_src,
    int src_first,float* out,float* stats,int rows,int key_begin,int key_end,int position,
    const int* position_device,int capacity,int splits,cudaStream_t stream) {
    static const bool configured=[] {
        cuda_check(cudaFuncSetAttribute(l0_exact_history_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(l0_exact_smem_bytes())),"configure L0 exact history shared memory");
        return true;
    }();
    (void)configured;
    l0_exact_history_kernel<<<dim3((rows+kL0Queries/kFastFusedFlashHeads-1)/(kL0Queries/kFastFusedFlashHeads),
        kKVHeads,splits),kL0HistoryThreads,l0_exact_smem_bytes(),stream>>>(
        static_cast<const std::uint16_t*>(q),k_src,v_src,src_first,out,stats,rows,key_begin,key_end,
        position,position_device,capacity,splits);
}

// Exact history of a prefill / verifier chunk of more than 8 rows into the
// prefill split slot `slot`: the L2 rows [kSink, history_end) are copied in
// 8K-row pieces into double-buffered device staging on a copy stream (row
// groups would otherwise each read them over PCIe), so the copy of piece i+1
// overlaps the attention of piece i; each piece runs through the FA2 prefill
// kernel (four key splits) and merges into the slot. The pass is bound by the
// FA2 compute, not the copies (a deeper cross-layer prefetch ring measured no
// gain at 64K-128K).
constexpr int kL0ExactPieceRows=8192;
inline void l0_exact_history_staged(const void* q,const std::uint16_t* k_src,const std::uint16_t* v_src,
    float* out,float* stats,float* numer,float* split_stats,int slot,int rows,int history_end,
    cudaStream_t stream) {
    struct Stage {
        std::uint16_t* k[2]{}; std::uint16_t* v[2]{};
        cudaStream_t copy=nullptr;
        cudaEvent_t fork=nullptr,ready[2]{},consumed[2]{};
    };
    static const Stage stage=[] {
        Stage s;
        const std::size_t bytes=static_cast<std::size_t>(kL0ExactPieceRows)*kKVHeads*kHeadDim*2;
        for(int b=0;b<2;++b) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&s.k[b]),bytes),"L0 exact staging");
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&s.v[b]),bytes),"L0 exact staging");
            cuda_check(cudaEventCreateWithFlags(&s.ready[b],cudaEventDisableTiming),"L0 exact staging event");
            cuda_check(cudaEventCreateWithFlags(&s.consumed[b],cudaEventDisableTiming),"L0 exact staging event");
        }
        cuda_check(cudaEventCreateWithFlags(&s.fork,cudaEventDisableTiming),"L0 exact staging event");
        cuda_check(cudaStreamCreateWithFlags(&s.copy,cudaStreamNonBlocking),"L0 exact staging stream");
        return s;
    }();
    constexpr std::size_t row=static_cast<std::size_t>(kKVHeads)*kHeadDim;
    // Copies read L2 rows written by earlier work on `stream`.
    cuda_check(cudaEventRecord(stage.fork,stream),"L0 exact staging fork");
    cuda_check(cudaStreamWaitEvent(stage.copy,stage.fork,0),"L0 exact staging fork");
    const int pieces=history_end>l0::kSink?(history_end-l0::kSink+kL0ExactPieceRows-1)/kL0ExactPieceRows:1;
    const auto submit=[&](int piece) {
        const int b=piece&1,first=l0::kSink+piece*kL0ExactPieceRows;
        const int last=std::min(history_end,first+kL0ExactPieceRows);
        cuda_check(cudaStreamWaitEvent(stage.copy,stage.consumed[b],0),"L0 exact staging reuse");
        if(last>first) {
            cuda_check(cudaMemcpyAsync(stage.k[b],k_src+first*row,(last-first)*row*2,cudaMemcpyDefault,stage.copy),"L0 exact staging copy");
            cuda_check(cudaMemcpyAsync(stage.v[b],v_src+first*row,(last-first)*row*2,cudaMemcpyDefault,stage.copy),"L0 exact staging copy");
        }
        cuda_check(cudaEventRecord(stage.ready[b],stage.copy),"L0 exact staging ready");
    };
    submit(0);
    for(int piece=0;piece<pieces;++piece) {
        if(piece+1<pieces) submit(piece+1);
        const int b=piece&1,first=l0::kSink+piece*kL0ExactPieceRows;
        const int last=std::max(first,std::min(history_end,first+kL0ExactPieceRows));
        cuda_check(cudaStreamWaitEvent(stream,stage.ready[b],0),"L0 exact staging wait");
        // The staged piece is a plain linear cache for the FA2 prefill kernel:
        // with base = capacity = piece rows every staged key is visible.
        const int piece_rows=last-first;
        if(piece_rows>0)
            launch_fa2_prefill_variant<4>(static_cast<const std::uint16_t*>(q),stage.k[b],stage.v[b],
                nullptr,rows,piece_rows,piece_rows,out,stats,stream,0,false);
        l0_exact_merge_kernel<<<rows*kQHeads,kHeadDim,0,stream>>>(out,stats,piece_rows>0?4:0,rows,numer,
            split_stats,slot,nullptr,nullptr,piece>0);
        cuda_check(cudaEventRecord(stage.consumed[b],stream),"L0 exact staging consumed");
    }
}