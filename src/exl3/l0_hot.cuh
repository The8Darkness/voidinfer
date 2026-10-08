#pragma once
// L0 OSCAR exact hot rows (l0_oscar.cuh HotView). Included by
// l0_oscar_attention.cuh ahead of the verifier history kernel; uses the
// verifier MMA helpers and the fused-flash slot format.
//
// Per layer and KV head, `slots` history rows are held as exact FP16 copies
// of their L2 rows (original basis, post-RoPE). Each verifier round:
//   l0_history_mma_kernel  every CTA first runs exact attention over its hot
//                          tiles (l0_hot_cta), then its INT2 segment, which
//                          skips hot rows and emits candidates
//   l0_history_merge_kernel / l0_rotate_kernel
//                          merge the INT2 partials in the rotated basis and the
//                          hot partials in the original basis, un-rotate the
//                          former and add the latter; four extra merge CTAs
//                          (l0_hot_select) rank the candidates against the
//                          stalest slots; the next encode_kernel copies the
//                          won rows in from L2
// Hot rows are a cache of committed history (< history_end), so they never
// take part in rollback; a context restart or restore clears them.

constexpr int kL0HotTile=32;
constexpr int kL0HotStride=kHeadDim+8;   // query rows (padded)
constexpr std::size_t l0_hot_smem_bytes() {
    return (static_cast<std::size_t>(kL0Queries)*kL0HotStride+2*kL0HotTile*kHeadDim)*sizeof(half);
}
// K/V tile rows are unpadded with their 16-byte chunks XOR-swizzled by
// (row & 7): ldmatrix row groups stay bank-conflict free.
__device__ __forceinline__ int l0_hot_at(int row,int col) {
    return row*kHeadDim+((((col>>3)^(row&7)))<<3)+(col&7);
}

// Hot tiles of one history CTA (384 threads = 12 warps of (m-tile of 16 query
// vectors, 64-dim value quarter)), run before its INT2 segment: 32-slot tiles
// cta, cta + ctas, ... of KV head kv with an online softmax over FP16 MMAs,
// one partial into slot `slot_index`. Spreading the tiles over every history
// CTA keeps the exact pass off the critical path at any context length.
// `ref_s` holds the per-vector candidate reference (log2 LSE + log2 tau).
// 16-byte cp.async global -> shared (zero-filled when !valid).
__device__ __forceinline__ void l0_hot_cp16(void* dst,const void* src,bool valid) {
    const unsigned d=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(d),"l"(src),"r"(valid?16:0));
}
__host__ __device__ __forceinline__ int l0_hot_tiles(int slots) { return slots/kL0HotTile; }
__device__ __forceinline__ void l0_hot_cta(unsigned char* smem,const std::uint16_t* q_bits,
    const l0::HotView& hot,float* workspace,
    int rows,int slot_stride,int slot_index,int cta,int ctas,int kv,int base,float log2_tau,
    float lambda,const float* ref_s) {
    constexpr int H=kFastFusedFlashHeads;
    constexpr float kLog2Scale=0.0625f*1.4426950408889634f;
    constexpr int kThreads=kL0HistoryThreads;
    __shared__ int row_s[kL0HotTile];
    __shared__ unsigned hit_s[kL0HotTile];
    half* qs=reinterpret_cast<half*>(smem);
    half* ks=qs+kL0Queries*kL0HotStride;
    half* vs=ks+kL0HotTile*kHeadDim;
    const int tid=static_cast<int>(threadIdx.x),warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
    const int mtile=warp%3,vq=warp/3;
    const int tiles=l0_hot_tiles(hot.slots);
    if(cta>=tiles) return;
    const int v0=mtile*16+g,v1=v0+8;
    const bool live[2]={v0/H<rows,v1/H<rows};
    float mx[2]={-INFINITY,-INFINITY},den[2]={0.f,0.f};
    float acc[8][4];
    #pragma unroll
    for(int n=0;n<8;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.f;
    bool first_tile=true;
    for(int tile_index=cta;tile_index<tiles;tile_index+=ctas) {
        const int tile=tile_index*kL0HotTile;
        if(!first_tile) __syncthreads();   // previous tile consumed
        // One load phase, all copies in flight: the query vectors (first tile)
        // and the slot tile.
        if(first_tile)
            for(int i=tid;i<kL0Queries*(kHeadDim/8);i+=kThreads) {
                const int v=i/(kHeadDim/8),c=i%(kHeadDim/8),row=v/H;
                l0_hot_cp16(qs+v*kL0HotStride+c*8,q_bits+
                    (static_cast<std::size_t>(row<rows?row:0)*kQHeads+kv*H+v%H)*kHeadDim+c*8,row<rows);
            }
        first_tile=false;
        for(int i=tid;i<2*kL0HotTile*(kHeadDim/8);i+=kThreads) {
            const int kind=i/(kL0HotTile*(kHeadDim/8)),k=(i/(kHeadDim/8))%kL0HotTile,c=i%(kHeadDim/8);
            const std::size_t at=(static_cast<std::size_t>(kv)*hot.slots+tile+k)*kHeadDim+c*8;
            l0_hot_cp16((kind?vs:ks)+l0_hot_at(k,c*8),(kind?hot.v:hot.k)+at,true);
        }
        if(tid<kL0HotTile) { row_s[tid]=hot.row[kv*hot.slots+tile+tid]; hit_s[tid]=0u; }
        asm volatile("cp.async.wait_all;\n");
        __syncthreads();
        // QK: s[n][e] rows v0 (e < 2) / v1, key 8 n + 2 t + (e & 1).
        float s[kL0HotTile/8][4];
        #pragma unroll
        for(int n=0;n<kL0HotTile/8;++n) s[n][0]=s[n][1]=s[n][2]=s[n][3]=0.f;
        #pragma unroll 4
        for(int kk=0;kk<kHeadDim;kk+=16) {
            unsigned a[4];
            reg_attn_ldmatrix_x4(a,qs+(mtile*16+(lane&15))*kL0HotStride+kk+(lane>>4)*8);
            #pragma unroll
            for(int pair=0;pair<kL0HotTile/16;++pair) {
                const int m=lane>>3;
                unsigned b[4];
                reg_attn_ldmatrix_x4(b,ks+l0_hot_at(pair*16+(m>>1)*8+(lane&7),kk+(m&1)*8));
                reg_attn_mma(s[2*pair],a,b[0],b[1]);
                reg_attn_mma(s[2*pair+1],a,b[2],b[3]);
            }
        }
        float tmax[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(int n=0;n<kL0HotTile/8;++n)
            #pragma unroll
            for(int e=0;e<4;++e) {
                const int r=e>>1,key=8*n+2*t+(e&1);
                const float x=row_s[key]>=0?s[n][e]*kLog2Scale:-INFINITY;
                s[n][e]=x;
                tmax[r]=fmaxf(tmax[r],x);
                if(vq==0 && live[r] && x-ref_s[r?v1:v0]>0.f)
                    atomicMax(&hit_s[key],__float_as_uint(x-ref_s[r?v1:v0]+1.0f));
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
        // PV over this warp's 64 value dims: acc[n] dims 64 vq + 8 n + 2 t + (e & 1).
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
                reg_attn_mma(acc[2*pair],a,b[0],b[1]);
                reg_attn_mma(acc[2*pair+1],a,b[2],b[3]);
            }
        }
        __syncthreads();
        if(tid<kL0HotTile && hit_s[tid]) {
            const float prio=(__uint_as_float(hit_s[tid])-1.0f+log2_tau)*0.6931471805599453f+
                lambda*static_cast<float>(base);
            float& cell=hot.prio[kv*hot.slots+tile+tid];
            cell=fmaxf(cell,prio);
        }
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int v=r?v1:v0,row=v/H,head=v%H;
        if(row>=rows) continue;
        float* slot=workspace+((static_cast<std::size_t>(row)*kKVHeads+kv)*slot_stride+slot_index)*
            kFastFusedFlashStride;
        #pragma unroll
        for(int n=0;n<8;++n)
            *reinterpret_cast<float2*>(slot+head*kHeadDim+64*vq+8*n+2*t)=make_float2(acc[n][2*r],acc[n][2*r+1]);
        if(t==0 && vq==0) {
            slot[kFastFusedFlashValues+head]=mx[r]==-INFINITY?-INFINITY:mx[r]*0.6931471805599453f;
            slot[kFastFusedFlashValues+H+head]=den[r];
        }
    }
    __syncthreads();   // the caller reuses the dynamic shared memory
}

// Per KV head (one 256-thread CTA): rank this round's candidates best-first
// and the slots stalest-first (256-bin histograms of the decayed priority,
// empty slots stalest), pair them rank by rank while the candidate is better,
// and publish the won (slot, row) pairs; the next round's encode_kernel fills
// the won slots from the FP16 L2 planes. Runs as extra CTAs of the history
// merge, after every reader of this round's hot set, so the next round's INT2
// segments and hot tiles see one consistent set.
constexpr int kL0HotSelectThreads=256;
__device__ __forceinline__ int l0_hot_bin(float eff) {
    return min(254,max(0,static_cast<int>(-eff*8.0f)));
}
// Exclusive prefix sum of a[0..255] into out (ascending, or descending bins when reverse).
__device__ __forceinline__ void l0_hot_scan(const int* a,int* out,bool reverse,int* warp_sums) {
    const int tid=static_cast<int>(threadIdx.x);
    const int b=reverse?255-tid:tid,x=a[b];
    int incl=x;
    for(int o=1;o<32;o<<=1) { const int y=__shfl_up_sync(0xffffffffU,incl,o); if((tid&31)>=o) incl+=y; }
    if((tid&31)==31) warp_sums[tid>>5]=incl;
    __syncthreads();
    int before=0;
    for(int w=0;w<(tid>>5);++w) before+=warp_sums[w];
    out[b]=before+incl-x;
}
__device__ __forceinline__ void l0_hot_select(const l0::HotView& hot,int kv,int base,int capacity,
    int insert,float lambda) {
    constexpr int kT=kL0HotSelectThreads;
    constexpr int kCand=l0::kHotCandidates/kT,kSlots=4096/kT;
    __shared__ int chist[256],ccum[256],cfill[256],shist[256],scum[256],sfill[256],sums[2][8];
    __shared__ int pick_row[l0::kHotMaxInsert],victim[l0::kHotMaxInsert];
    __shared__ float pick_eff[l0::kHotMaxInsert],veff[l0::kHotMaxInsert];
    __shared__ int won_slot[l0::kHotMaxInsert],won_row[l0::kHotMaxInsert];
    __shared__ int valid,won;
    constexpr float kLn2=0.6931471805599453f;
    const int tid=static_cast<int>(threadIdx.x);
    const int n=min(hot.count[kv],l0::kHotCandidates);
    if(tid==0) hot.won[kv]=0;
    if(n==0) return;
    const int end=min(l0::history_end(base),capacity);
    const float now=lambda*static_cast<float>(base);
    chist[tid]=0; cfill[tid]=0; shist[tid]=0; sfill[tid]=0;
    if(tid==0) { valid=0; won=0; }
    __syncthreads();
    if(tid==0) hot.count[kv]=0;
    // Candidate effective priority = ln p (relative to the reference LSE).
    int crow[kCand],cbin[kCand];
    float ceff[kCand];
    #pragma unroll
    for(int q=0;q<kCand;++q) {
        const int i=tid+q*kT;
        cbin[q]=-1; crow[q]=-1; ceff[q]=0.f;
        if(i<n) {
            crow[q]=hot.cand_row[kv*l0::kHotCandidates+i];
            if(crow[q]>=l0::kSink && crow[q]<end) {
                ceff[q]=hot.cand_val[kv*l0::kHotCandidates+i]*kLn2;
                cbin[q]=l0_hot_bin(ceff[q]);
                atomicAdd(&chist[cbin[q]],1);
                atomicAdd(&valid,1);
            }
        }
    }
    // Slot effective priority, decayed to now; empty slots go first.
    int sbin[kSlots];
    float seff[kSlots];
    #pragma unroll
    for(int q=0;q<kSlots;++q) {
        const int s=tid+q*kT;
        sbin[q]=-1; seff[q]=0.f;
        if(s<hot.slots) {
            const int r=hot.row[kv*hot.slots+s];
            seff[q]=r<0?-INFINITY:hot.prio[kv*hot.slots+s]-now;
            sbin[q]=r<0?255:l0_hot_bin(seff[q]);
            atomicAdd(&shist[sbin[q]],1);
        }
    }
    __syncthreads();
    const int take=min(valid,insert);
    if(take==0) return;
    l0_hot_scan(chist,ccum,false,sums[0]);   // best candidates first
    l0_hot_scan(shist,scum,true,sums[1]);    // stalest slots first
    __syncthreads();
    #pragma unroll
    for(int q=0;q<kCand;++q)
        if(cbin[q]>=0) {
            const int rank=ccum[cbin[q]]+atomicAdd(&cfill[cbin[q]],1);
            if(rank<take) { pick_row[rank]=crow[q]; pick_eff[rank]=ceff[q]; }
        }
    #pragma unroll
    for(int q=0;q<kSlots;++q)
        if(sbin[q]>=0) {
            const int rank=scum[sbin[q]]+atomicAdd(&sfill[sbin[q]],1);
            if(rank<take) { victim[rank]=tid+q*kT; veff[rank]=seff[q]; }
        }
    __syncthreads();
    if(tid<take && pick_eff[tid]>veff[tid]) {
        const int s=victim[tid],cell=kv*hot.slots+s,row=pick_row[tid];
        const int old=hot.row[cell];
        if(old>=0) atomicAnd(hot.bits+kv*hot.words+(old>>5),~(1u<<(old&31)));
        atomicOr(hot.bits+kv*hot.words+(row>>5),1u<<(row&31));
        hot.row[cell]=row;
        hot.prio[cell]=pick_eff[tid]+now;
        const int w=atomicAdd(&won,1);
        won_slot[w]=s; won_row[w]=row;
    }
    __syncthreads();
    if(tid<won) {
        hot.won[l0::kHotWonSlots+kv*l0::kHotMaxInsert+tid]=won_slot[tid];
        hot.won[l0::kHotWonRows+kv*l0::kHotMaxInsert+tid]=won_row[tid];
    }
    if(tid==0) hot.won[kv]=won;
}
