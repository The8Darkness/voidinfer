#pragma once
// L0 OSCAR history attention (see l0_oscar.cuh). Included by
// full_attention_layer.cu after the verifier MMA helpers; shares their
// fragment layout, online softmax and fused-flash slot format.

namespace l0 = voidinfer::exl3::l0_oscar;

// Verifier history attention over INT4 or INT2 codes (template Bits,
// l0_oscar_format.h), consumed straight from the codes (no FP16 tiles):
//   QK  s8 MMA (m16n8k32) of the int8-quantized rotated queries (per 128-dim
//       group scale) against the raw codes, masked in place (INT4: one code
//       word = 8 dims per lane and k-step, nibble masks; INT2: 2-bit masks);
//       the per-key K scale and zero are applied to the int32 sums of each
//       group.
//   PV  u8 MMA of P (scaled by the per-key V scale, quantized per row)
//       against the V codes read in place (c 16^(n & 1) / c 4^(n & 3)); the
//       V zero term is carried as a per-row bias.
// CTA = (history segment, KV head), 12 warps = (m-tile of 16 query vectors,
// 128-dim value group, key stream). Raw codes and metadata stream through a
// cp.async ring of 64-key stages shared by all warps; each stream takes one
// 16-key block of every 32 keys. The two streams merge in shared memory.
constexpr int kL0HistoryStreams=1;          // partial slots per segment
constexpr int kL0KeyStreams=2;              // in-CTA key streams
constexpr int kL0HistoryThreads=192*kL0KeyStreams;
constexpr int kL0Queries=48;
constexpr int kL0StageKeys=64;
// Per code width: raw key bytes (K codes, V codes, K meta, V meta, FP16) and
// the cp.async ring depth (INT4 272-byte keys: 4 x 17 KiB stages + ~26 KiB
// static; INT2 144-byte keys: 5 stages).
template<int Bits> struct L0HistoryLayout {
    static constexpr int code_bytes=voidinfer::exl3::l0_oscar_format::CodeFormat<Bits>::code_bytes;
    static constexpr int stages=Bits==4?4:5;
    static constexpr int raw_key_bytes=2*code_bytes+2*l0::kMetaFloats*2;
    static constexpr std::size_t stage_bytes=static_cast<std::size_t>(kL0StageKeys)*raw_key_bytes;
};
constexpr std::size_t kL0MergeBytes=static_cast<std::size_t>(6)*32*70*sizeof(float);
#include "exl3/l0_hot.cuh"
#include "exl3/l0_exact.cuh"
template<int Bits> constexpr std::size_t l0_history_smem_bytes() {
    using L=L0HistoryLayout<Bits>;
    constexpr std::size_t a=L::stages*L::stage_bytes>kL0MergeBytes?L::stages*L::stage_bytes:kL0MergeBytes;
    return a>l0_hot_smem_bytes()?a:l0_hot_smem_bytes();
}

__device__ __forceinline__ void l0_cp16(void* dst,const void* src,bool valid) {
    const unsigned d=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(d),"l"(src),"r"(valid?16:0));
}
// Same copies to a 32-bit shared-window address.
__device__ __forceinline__ void l0_cp16(unsigned d,const void* src,bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(d),"l"(src),"r"(valid?16:0));
}
__device__ __forceinline__ void l0_cp8(unsigned d,const void* src,bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"::"r"(d),"l"(src),"r"(valid?8:0));
}
__device__ __forceinline__ void l0_cp4(unsigned d,const void* src,bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"::"r"(d),"l"(src),"r"(valid?4:0));
}
// 2^x on the MUFU (flushes subnormal results to zero; 2^-inf = 0).
__device__ __forceinline__ float l0_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;":"=f"(y):"f"(x));
    return y;
}
__device__ __forceinline__ void l0_cp4(void* dst,const void* src,bool valid) {
    const unsigned d=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"::"r"(d),"l"(src),"r"(valid?4:0));
}
__device__ __forceinline__ void l0_cp8(void* dst,const void* src,bool valid) {
    const unsigned d=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"::"r"(d),"l"(src),"r"(valid?8:0));
}
// FP16 (scale0, zero0, scale1, zero1) of one key and KV head.
__device__ __forceinline__ float4 l0_meta4(const void* p) {
    const uint2 raw=*reinterpret_cast<const uint2*>(p);
    const float2 a=__half22float2(*reinterpret_cast<const __half2*>(&raw.x));
    const float2 b=__half22float2(*reinterpret_cast<const __half2*>(&raw.y));
    return make_float4(a.x,a.y,b.x,b.y);
}
__device__ __forceinline__ void l0_imma(int (&c)[4],const unsigned (&a)[4],unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// Undoes the in-place code scale of PV n-tile n (see the PV loop): INT4
// 16^(n & 1), INT2 4^(n & 3).
template<int Bits> __device__ __forceinline__ constexpr float l0_code_scale(int n) {
    if constexpr(Bits==4) return (n&1)?0.0625f:1.0f;
    else return 1.0f/static_cast<float>(1<<(2*(n&3)));
}
__device__ __forceinline__ void l0_immau(int (&c)[4],const unsigned (&a)[4],unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}


// Integer MMA accumulators start at the bits of 1.5 * 2^23: while |sum| <
// 2^22 the int32 result read as a float is 12582912 + sum exactly, so one
// FADD replaces a quarter-rate I2F.
constexpr int kL0IntMagic=0x4B400000;
__device__ __forceinline__ float l0_int_value(int bits) {
    return __int_as_float(bits)-12582912.0f;
}
// kHot: each CTA first runs its exact hot tiles (l0_hot_cta) over the
// original-basis queries `q_orig`; rows marked hot are then skipped and keys
// whose probability relative to the previous round's history LSE exceeds tau
// are emitted once per (key, KV head) as hot candidates.
template<int kKeys,bool kHot,int Bits>
__global__ void __launch_bounds__(kL0HistoryThreads,1) l0_history_mma_kernel(
    const std::uint16_t* q_rot_bits,const float* q_mu,const std::uint8_t* k_codes,
    const std::uint8_t* v_codes,const __half* k_meta,const __half* v_meta,
    float* workspace,int rows,int position,int capacity,int segments,
    const int* position_device,int slot_stride,l0::HotView hot,float log2_tau,
    const std::uint16_t* q_orig,float lambda) {
    constexpr int H=kFastFusedFlashHeads;
    constexpr float kLog2Scale=0.0625f*1.4426950408889634f;
    using L=L0HistoryLayout<Bits>;
    constexpr int kL0Stages=L::stages,kL0RawKeyBytes=L::raw_key_bytes,kCodeBytes=L::code_bytes;
    constexpr std::size_t kL0StageBytes=L::stage_bytes;
    const auto* q_rot=reinterpret_cast<const __half*>(q_rot_bits);
    extern __shared__ __align__(16) unsigned char l0_smem[];
    __shared__ __align__(16) signed char q8[kL0Queries][kHeadDim];
    // INT4: [m-tile][k-step][lane] one uint4; INT2: [m-tile][k-step pair][lane] two uint4.
    __shared__ __align__(16) unsigned qfrag[3][Bits==4?8:4][32][Bits==4?4:8];
    __shared__ float qinfo[kL0Queries][5];   // qs0, qs1, qsum0, qsum1, mu (log2 units)
    // Per stage key: bits of 1 + log2 p - log2 tau of its best vector (0 = none), double-buffered.
    __shared__ unsigned hot_emit[2][kL0StageKeys];
    // Per stage: the hot-row bit word of each 32-key block (streamed with the codes).
    __shared__ unsigned hot_ring[kHot?kL0Stages:1][kL0KeyStreams];
    __shared__ float hot_ref_s[kL0Queries];
    const int segment=static_cast<int>(blockIdx.x);
    const int kv_head=static_cast<int>(blockIdx.y);
    const int tid=static_cast<int>(threadIdx.x);
    const int warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
    const int stream=warp/6,vgroup=(warp/3)&1,mtile=warp%3;
    const int base=position_device?*position_device:position;
    if constexpr(kHot) {
        if(tid<kL0Queries) hot_ref_s[tid]=hot.ref[kv_head*H+tid%H]+log2_tau;
        l0_hot_cta(l0_smem,q_orig,hot,workspace,rows,slot_stride,
            segments*kL0HistoryStreams+segment,segment,segments,kv_head,base,log2_tau,lambda,hot_ref_s);
    }
    const int end=min(l0::history_end(base),capacity);
    const l0::HistorySplit split=l0::history_split(end,segments,kHot?l0_hot_tiles(hot.slots)%segments:0);
    const int first=split.first(segment);
    if(first>=end) return;
    const int segment_end=min(first+split.length(segment),end);
    const int stages=(segment_end-first+kL0StageKeys-1)/kL0StageKeys;
    // The two key streams run decoupled pipelines: stream h's 192 threads
    // fetch its 32 keys of every stage and sync on named barrier 7 + h.
    // Chunks in plane order (K codes, V codes in 16 B chunks, then K and V
    // meta in 8 B): each thread's plane is warp-uniform and its key and part
    // are shifts of its chunk index.
    constexpr int kStreamThreads=kL0HistoryThreads/kL0KeyStreams;
    constexpr int kStreamKeys=kL0StageKeys/kL0KeyStreams;
    constexpr int kCodeChunks=kCodeBytes/16;
    constexpr int kCodeSpan=kStreamKeys*kCodeChunks;
    constexpr int kStreamChunks=2*kCodeSpan+2*kStreamKeys;
    const int stid=tid-kStreamThreads*stream;
    const unsigned ring_s=static_cast<unsigned>(__cvta_generic_to_shared(l0_smem))+kStreamKeys*stream*kL0RawKeyBytes;
    const unsigned hot_ring_s=static_cast<unsigned>(__cvta_generic_to_shared(&hot_ring[0][stream]));
    const auto fetch=[&](int stage) {
        if(stage<stages) {
            const unsigned dst=ring_s+(stage%kL0Stages)*static_cast<unsigned>(kL0StageBytes);
            const int key0=first+stage*kL0StageKeys+kStreamKeys*stream;
            // The chunk layout is re-derived from %tid.x per stage: cheaper than
            // the spill reloads of hoisted per-thread descriptors.
            unsigned tid_now;
            asm volatile("mov.u32 %0, %%tid.x;":"=r"(tid_now));
            const int stid_now=static_cast<int>(tid_now)%kStreamThreads;
            #pragma unroll
            for(int j=0;j<(kStreamChunks+kStreamThreads-1)/kStreamThreads;++j) {
                const int c=stid_now+j*kStreamThreads;
                if(c>=kStreamChunks) break;
                const bool code=c<2*kCodeSpan;
                const int m=code?c:c-2*kCodeSpan;
                const bool second=code?m>=kCodeSpan:m>=kStreamKeys;
                const int key_off=code?(m-(second?kCodeSpan:0))/kCodeChunks:m-(second?kStreamKeys:0);
                const int key=key0+key_off;
                const bool valid=key<segment_end;
                const std::size_t cell=static_cast<std::size_t>(valid?key:first)*kKVHeads+kv_head;
                const unsigned d=dst+key_off*kL0RawKeyBytes;
                if(code) {
                    const int part=m%kCodeChunks;
                    l0_cp16(d+(second?kCodeBytes:0)+part*16,(second?v_codes:k_codes)+cell*kCodeBytes+part*16,valid);
                } else {
                    l0_cp8(d+2*kCodeBytes+(second?8:0),(second?v_meta:k_meta)+cell*l0::kMetaFloats,valid);
                }
            }
            if constexpr(kHot) {
                if(stid==0)
                    l0_cp4(hot_ring_s+(stage%kL0Stages)*kL0KeyStreams*4,hot.bits+kv_head*hot.words+(key0<segment_end?key0>>5:0),
                        key0<segment_end);
            }
        }
        asm volatile("cp.async.commit_group;\n");
    };
    #pragma unroll
    for(int s=0;s<kL0Stages-1;++s) fetch(s);
    // Quantize the 48 query vectors per 128-dim group: two threads per
    // (vector, group), all loads in flight before the reductions.
    if(tid<4*kL0Queries) {
        const int unit=tid>>1,part=tid&1,v=unit>>1,group=unit&1,row=v/H,head=v%H;
        const int dim0=group*128+part*64;
        uint4 raw[8];
        #pragma unroll
        for(int i=0;i<8;++i) raw[i]=make_uint4(0,0,0,0);
        if(row<rows) {
            const uint4* src=reinterpret_cast<const uint4*>(q_rot+
                (static_cast<std::size_t>(row)*kQHeads+kv_head*H+head)*kHeadDim+dim0);
            #pragma unroll
            for(int i=0;i<8;++i) raw[i]=src[i];
        }
        float amax=0.f,sum=0.f;
        #pragma unroll
        for(int i=0;i<8;++i) {
            const __half2* h2=reinterpret_cast<const __half2*>(&raw[i]);
            #pragma unroll
            for(int k=0;k<4;++k) {
                const float2 f=__half22float2(h2[k]);
                amax=fmaxf(amax,fmaxf(fabsf(f.x),fabsf(f.y)));sum+=f.x+f.y;
            }
        }
        amax=fmaxf(amax,__shfl_xor_sync(0xffffffffU,amax,1));
        sum+=__shfl_xor_sync(0xffffffffU,sum,1);
        const float inv=amax>0.f?127.f/amax:0.f;
        #pragma unroll
        for(int i=0;i<8;++i) {
            const __half2* h2=reinterpret_cast<const __half2*>(&raw[i]);
            unsigned packed[2];
            #pragma unroll
            for(int k=0;k<2;++k) {
                const float2 f0=__half22float2(h2[2*k]),f1=__half22float2(h2[2*k+1]);
                packed[k]=(__float2int_rn(f0.x*inv)&255)|((__float2int_rn(f0.y*inv)&255)<<8)|
                    ((__float2int_rn(f1.x*inv)&255)<<16)|((__float2int_rn(f1.y*inv)&255)<<24);
            }
            *reinterpret_cast<uint2*>(&q8[v][dim0+8*i])=make_uint2(packed[0],packed[1]);
        }
        if(part==0) {
            qinfo[v][group]=amax/127.f*kLog2Scale;
            qinfo[v][2+group]=sum*kLog2Scale;
            if(group==0) qinfo[v][4]=row<rows?q_mu[row*kQHeads+kv_head*H+head]*kLog2Scale:0.f;
        }
    }
    if constexpr(kHot) {
        if(tid<2*kL0StageKeys) hot_emit[tid/kL0StageKeys][tid%kL0StageKeys]=0u;
    }
    __syncthreads();
    // A fragments. INT4: k-step s uses code word W = 8 (s >> 1) + 2t + (s & 1)
    // of each key; byte i of (w >> 4j) & 0x0F0F0F0F is dim 8 W + 2 i + j (k rows
    // 4t + i for j = 0, 16 + 4t + i for j = 1). INT2: k-step pair m uses code
    // word W = 8 (m >> 1) + 2t + (m & 1); byte i of (w >> 2j) & 0x03030303 is
    // dim 16 W + 4 i + j.
    // Appends this stream's flagged keys of stage `stage` and clears them.
    const auto hot_flush=[&](int stage) {
        if constexpr(kHot) {
            if(stage>=0 && stid<kStreamKeys) {
                unsigned& cell=hot_emit[stage&1][kStreamKeys*stream+stid];
                if(cell) {
                    const int slot=atomicAdd(hot.count+kv_head,1);
                    if(slot<l0::kHotCandidates) {
                        hot.cand_row[kv_head*l0::kHotCandidates+slot]=first+stage*kL0StageKeys+kStreamKeys*stream+stid;
                        hot.cand_val[kv_head*l0::kHotCandidates+slot]=__uint_as_float(cell)-1.0f+log2_tau;
                    }
                    cell=0u;
                }
            }
        }
    };
    if constexpr(Bits==4) {
        for(int e=tid;e<3*8*32;e+=kL0HistoryThreads) {
            const int mt=e/256,s=(e/32)%8,ln=e%32,gg=ln>>2,tt=ln&3;
            const int W=8*(s>>1)+2*tt+(s&1);
            const signed char* r0=q8[mt*16+gg];
            const signed char* r1=q8[mt*16+gg+8];
            unsigned a0=0,a1=0,a2=0,a3=0;
            #pragma unroll
            for(int i=0;i<4;++i) {
                const int d=8*W+2*i;
                a0|=static_cast<unsigned>(static_cast<unsigned char>(r0[d]))<<(8*i);
                a1|=static_cast<unsigned>(static_cast<unsigned char>(r1[d]))<<(8*i);
                a2|=static_cast<unsigned>(static_cast<unsigned char>(r0[d+1]))<<(8*i);
                a3|=static_cast<unsigned>(static_cast<unsigned char>(r1[d+1]))<<(8*i);
            }
            *reinterpret_cast<uint4*>(&qfrag[mt][s][ln][0])=make_uint4(a0,a1,a2,a3);
        }
    } else {
        for(int e=tid;e<3*4*32;e+=kL0HistoryThreads) {
            const int mt=e/128,m=(e/32)%4,ln=e%32,gg=ln>>2,tt=ln&3;
            const int W=8*(m>>1)+2*tt+(m&1);
            const signed char* r0=q8[mt*16+gg];
            const signed char* r1=q8[mt*16+gg+8];
            unsigned out[8];
            #pragma unroll
            for(int sb=0;sb<2;++sb) {
                unsigned a0=0,a1=0,a2=0,a3=0;
                #pragma unroll
                for(int i=0;i<4;++i) {
                    const int d=16*W+4*i+2*sb;
                    a0|=static_cast<unsigned>(static_cast<unsigned char>(r0[d]))<<(8*i);
                    a1|=static_cast<unsigned>(static_cast<unsigned char>(r1[d]))<<(8*i);
                    a2|=static_cast<unsigned>(static_cast<unsigned char>(r0[d+1]))<<(8*i);
                    a3|=static_cast<unsigned>(static_cast<unsigned char>(r1[d+1]))<<(8*i);
                }
                out[4*sb]=a0;out[4*sb+1]=a1;out[4*sb+2]=a2;out[4*sb+3]=a3;
            }
            *reinterpret_cast<uint4*>(&qfrag[mt][m][ln][0])=make_uint4(out[0],out[1],out[2],out[3]);
            *reinterpret_cast<uint4*>(&qfrag[mt][m][ln][4])=make_uint4(out[4],out[5],out[6],out[7]);
        }
    }
    const int v0=mtile*16+g,v1=v0+8;
    // acc[n][e]: rows v0 (e < 2) / v1, dims 128 vgroup + 16 (2t + (e & 1)) + n in
    // units of 4^-(n & 3) (codes read in place); bias[r] is the V zero term.
    float acc[16][4];
    #pragma unroll
    for(int n=0;n<16;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.0f;
    float mx[2]={-INFINITY,-INFINITY},den[2]={0.f,0.f},bias[2]={0.f,0.f};
    for(int stage=0;stage<stages;++stage) {
        asm volatile("cp.async.wait_group %0;\n"::"n"(kL0Stages-2));
        asm volatile("bar.sync %0, %1;\n"::"r"(7+stream),"n"(kStreamThreads):"memory");
        fetch(stage+kL0Stages-1);
        hot_flush(stage-1);
        const unsigned char* raw=l0_smem+(stage%kL0Stages)*kL0StageBytes;
        float qi[2][5];
        #pragma unroll
        for(int i=0;i<5;++i) { qi[0][i]=qinfo[v0][i]; qi[1][i]=qinfo[v1][i]; }
        // One 32-key block per stream and stage.
        const int key_off=32*stream;
        const int kb=first+stage*kL0StageKeys+key_off;
        const unsigned char* bk=raw+key_off*kL0RawKeyBytes;
        // QK: int32 sums per group. Score n-tile j column c is key
        // 4 (c >> 1) + 2 (j & 1) + (c & 1) + 16 (j >> 1), so this lane's
        // scores are keys 4t..4t+3 and 16+4t..16+4t+3 (the PV A-fragment keys).
        // The two value-group warps of an (m-tile, stream) pair split them:
        // vgroup h scores n-tiles j = 2h + jj (keys 16h..16h+15) and the pair
        // swaps its halves through shared memory (q8, dead after the query
        // fragments are built), so every score is computed once.
        int s[2][2][4];
        #pragma unroll
        for(int h=0;h<2;++h)
            #pragma unroll
            for(int jj=0;jj<2;++jj) s[h][jj][0]=s[h][jj][1]=s[h][jj][2]=s[h][jj][3]=kL0IntMagic;
        if constexpr(Bits==4) {
            // Key jj's code words 8k + 2t, 8k + 2t + 1, loaded per k-step pair.
            const uint2* kw=reinterpret_cast<const uint2*>(bk+(4*(g>>1)+(g&1)+16*vgroup)*kL0RawKeyBytes)+t;
            #pragma unroll
            for(int k=0;k<4;++k) {
                const uint2 kc[2]={kw[4*k],kw[4*k+2*kL0RawKeyBytes/8]};
                #pragma unroll
                for(int half=0;half<2;++half) {
                    const int st=2*k+half;
                    const uint4 f=*reinterpret_cast<const uint4*>(&qfrag[mtile][st][lane][0]);
                    const unsigned a[4]={f.x,f.y,f.z,f.w};
                    #pragma unroll
                    for(int jj=0;jj<2;++jj) {
                        const unsigned w=half?kc[jj].y:kc[jj].x;
                        l0_imma(s[k>>1][jj],a,w&0x0F0F0F0FU,(w>>4)&0x0F0F0F0FU);
                    }
                }
            }
        } else {
            uint2 kc[2][2];
            #pragma unroll
            for(int jj=0;jj<2;++jj) {
                const int key=4*(g>>1)+2*jj+(g&1)+16*vgroup;
                const uint2* w=reinterpret_cast<const uint2*>(bk+key*kL0RawKeyBytes);
                kc[jj][0]=w[t];kc[jj][1]=w[4+t];
            }
            #pragma unroll
            for(int m=0;m<4;++m) {
                const uint4 f0=*reinterpret_cast<const uint4*>(&qfrag[mtile][m][lane][0]);
                const uint4 f1=*reinterpret_cast<const uint4*>(&qfrag[mtile][m][lane][4]);
                const unsigned a[2][4]={{f0.x,f0.y,f0.z,f0.w},{f1.x,f1.y,f1.z,f1.w}};
                #pragma unroll
                for(int jj=0;jj<2;++jj) {
                    const unsigned w=(m&1)?kc[jj][m>>1].y:kc[jj][m>>1].x;
                    #pragma unroll
                    for(int sb=0;sb<2;++sb)
                        l0_imma(s[m>>1][jj],a[sb],(w>>(4*sb))&0x03030303U,(w>>(4*sb+2))&0x03030303U);
                }
            }
        }
        // This lane's keys: 4t + (i & 3) + 16 (i >> 2), i = 4 (j >> 1) + 2 (j & 1) + (e & 1);
        // own scores mine[r][i & 3] are i >> 2 == vgroup.
        float mine[2][4];
        // The 32-key block is one hot-bit word (blocks are 32-aligned).
        const unsigned hot_word=kHot?hot_ring[stage%kL0Stages][stream]:0u;
        #pragma unroll
        for(int jj=0;jj<2;++jj)
            #pragma unroll
            for(int e=0;e<4;++e) {
                const int r=e>>1,ii=2*jj+(e&1),key=4*t+ii+16*vgroup;
                const float4 k=l0_meta4(bk+key*kL0RawKeyBytes+2*kCodeBytes);
                const float v=qi[r][0]*k.x*l0_int_value(s[0][jj][e])+qi[r][1]*k.z*l0_int_value(s[1][jj][e])-
                    qi[r][2]*k.y*k.x-qi[r][3]*k.w*k.z+qi[r][4];
                mine[r][ii]=kb+key<segment_end && !((hot_word>>key)&1u)?v:-INFINITY;
            }
        if constexpr(kHot) {
            // Candidates are rare: one test per lane before the per-key atomics.
            const float ref[2]={hot_ref_s[v0],hot_ref_s[v1]};
            const float best=fmaxf(fmaxf(fmaxf(mine[0][0],mine[0][1])-ref[0],fmaxf(mine[0][2],mine[0][3])-ref[0]),
                                   fmaxf(fmaxf(mine[1][0],mine[1][1])-ref[1],fmaxf(mine[1][2],mine[1][3])-ref[1]));
            if(best>0.f) {
                #pragma unroll
                for(int r=0;r<2;++r)
                    #pragma unroll
                    for(int ii=0;ii<4;++ii) {
                        const float rel=mine[r][ii]-ref[r];
                        if(rel>0.f)
                            atomicMax(&hot_emit[stage&1][key_off+4*t+ii+16*vgroup],__float_as_uint(rel+1.0f));
                    }
            }
        }
        float sc[2][8];
        {
            float4* xchg=reinterpret_cast<float4*>(&q8[0][0]);
            xchg[2*(warp*32+lane)]=make_float4(mine[0][0],mine[0][1],mine[0][2],mine[0][3]);
            xchg[2*(warp*32+lane)+1]=make_float4(mine[1][0],mine[1][1],mine[1][2],mine[1][3]);
            asm volatile("bar.sync %0, 64;\n"::"r"(1+3*stream+mtile):"memory");
            const int partner=warp+(vgroup?-3:3);
            const float4 o0=xchg[2*(partner*32+lane)],o1=xchg[2*(partner*32+lane)+1];
            const float other[2][4]={{o0.x,o0.y,o0.z,o0.w},{o1.x,o1.y,o1.z,o1.w}};
            #pragma unroll
            for(int r=0;r<2;++r)
                #pragma unroll
                for(int ii=0;ii<4;++ii) {
                    sc[r][ii]=vgroup?other[r][ii]:mine[r][ii];
                    sc[r][4+ii]=vgroup?mine[r][ii]:other[r][ii];
                }
        }
        float tmax[2];
        #pragma unroll
        for(int r=0;r<2;++r) {
            tmax[r]=sc[r][0];
            #pragma unroll
            for(int i=1;i<8;++i) tmax[r]=fmaxf(tmax[r],sc[r][i]);
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
            tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffU,tmax[r],1));
            tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffU,tmax[r],2));
        }
        // Lazy rescale: the running max moves only when a score exceeds it by
        // more than 2^kSlack.
        constexpr float kSlack=8.0f;
        const bool grow0=tmax[0]>mx[0]+kSlack,grow1=tmax[1]>mx[1]+kSlack;
        if(__any_sync(0xffffffffU,grow0||grow1)) {
            float alpha[2];
            #pragma unroll
            for(int r=0;r<2;++r) {
                const float next=(r?grow1:grow0)?tmax[r]:mx[r];
                alpha[r]=mx[r]==-INFINITY?(next==-INFINITY?1.0f:0.0f):l0_ex2(mx[r]-next);
                mx[r]=next;
                den[r]*=alpha[r];bias[r]*=alpha[r];
            }
            #pragma unroll
            for(int n=0;n<16;++n) {
                acc[n][0]*=alpha[0];acc[n][1]*=alpha[0];
                acc[n][2]*=alpha[1];acc[n][3]*=alpha[1];
            }
        }
        // P scaled by this group's V scale, quantized to u8 per row.
        // A row without a live key yet keeps mx = -inf (and only -inf scores).
        float pv[2][8],pmax[2]={0.f,0.f};
        const float base_mx[2]={mx[0]==-INFINITY?0.f:mx[0],mx[1]==-INFINITY?0.f:mx[1]};
        #pragma unroll
        for(int r=0;r<2;++r)
            #pragma unroll
            for(int i=0;i<8;++i) {
                const int key=4*t+(i&3)+16*(i>>2);
                const float p=l0_ex2(sc[r][i]-base_mx[r]);
                const float2 vm=__half22float2(*reinterpret_cast<const __half2*>(
                    bk+key*kL0RawKeyBytes+2*kCodeBytes+8+4*vgroup));
                den[r]+=p;
                bias[r]=fmaf(-p*vm.y,vm.x,bias[r]);
                pv[r][i]=p*vm.x;
                pmax[r]=fmaxf(pmax[r],pv[r][i]);
            }
        float qscale[2];
        unsigned pa[4];   // a0 row v0 keys 4t.., a1 row v1, a2 row v0 keys 16+4t.., a3 row v1
        #pragma unroll
        for(int r=0;r<2;++r) {
            float m=pmax[r];
            m=fmaxf(m,__shfl_xor_sync(0xffffffffU,m,1)); m=fmaxf(m,__shfl_xor_sync(0xffffffffU,m,2));
            const float inv=m>0.f?__fdividef(255.f,m):0.f;
            qscale[r]=m*(1.f/255.f);
            #pragma unroll
            for(int half_index=0;half_index<2;++half_index) {
                // round(pv inv) in the low byte of pv inv + 1.5 2^23 (pv inv <= 255).
                unsigned b[4];
                #pragma unroll
                for(int i=0;i<4;++i) b[i]=__float_as_uint(fmaf(pv[r][4*half_index+i],inv,12582912.0f));
                pa[2*half_index+r]=__byte_perm(__byte_perm(b[0],b[1],0x0040),__byte_perm(b[2],b[3],0x0040),0x5410);
            }
        }
        // PV: B column g of n-tile n is dim 128 vgroup + 16 g + n. INT4: word
        // 16 vgroup + 2 g + (n >> 3), byte (n & 7) >> 1, nibble n & 1. INT2: word
        // 8 vgroup + g, byte n >> 2, bits 2 (n & 3). b0 keys 4t..4t+3, b1 keys
        // 16+4t..16+4t+3.
        constexpr int kWordsPerColumn=Bits==4?2:1;
        unsigned u[kWordsPerColumn][2][4];   // [word][quad][byte]: that byte of the 4 keys
        #pragma unroll
        for(int half_word=0;half_word<kWordsPerColumn;++half_word)
            #pragma unroll
            for(int quad=0;quad<2;++quad) {
                const unsigned char* vb=bk+(16*quad+4*t)*kL0RawKeyBytes+kCodeBytes+
                    4*(Bits==4?16*vgroup+2*g+half_word:8*vgroup+g);
                const unsigned w0=*reinterpret_cast<const unsigned*>(vb);
                const unsigned w1=*reinterpret_cast<const unsigned*>(vb+kL0RawKeyBytes);
                const unsigned w2=*reinterpret_cast<const unsigned*>(vb+2*kL0RawKeyBytes);
                const unsigned w3=*reinterpret_cast<const unsigned*>(vb+3*kL0RawKeyBytes);
                const unsigned t0=__byte_perm(w0,w1,0x5140),t1=__byte_perm(w2,w3,0x5140);
                const unsigned t2=__byte_perm(w0,w1,0x7362),t3=__byte_perm(w2,w3,0x7362);
                u[half_word][quad][0]=__byte_perm(t0,t1,0x5410);u[half_word][quad][1]=__byte_perm(t0,t1,0x7632);
                u[half_word][quad][2]=__byte_perm(t2,t3,0x5410);u[half_word][quad][3]=__byte_perm(t2,t3,0x7632);
            }
        #pragma unroll
        for(int n=0;n<16;++n) {
            int c[4]={kL0IntMagic,kL0IntMagic,kL0IntMagic,kL0IntMagic};
            if constexpr(Bits==4) {
                const unsigned mask=0x0F0F0F0FU<<(4*(n&1));
                l0_immau(c,pa,u[n>>3][0][(n&7)>>1]&mask,u[n>>3][1][(n&7)>>1]&mask);
            } else {
                const unsigned mask=0x03030303U<<(2*(n&3));
                l0_immau(c,pa,u[0][0][n>>2]&mask,u[0][1][n>>2]&mask);
            }
            acc[n][0]=fmaf(qscale[0],l0_int_value(c[0]),acc[n][0]);
            acc[n][1]=fmaf(qscale[0],l0_int_value(c[1]),acc[n][1]);
            acc[n][2]=fmaf(qscale[1],l0_int_value(c[2]),acc[n][2]);
            acc[n][3]=fmaf(qscale[1],l0_int_value(c[3]),acc[n][3]);
        }
    }
    asm volatile("cp.async.wait_group 0;\n");
    #pragma unroll
    for(int r=0;r<2;++r)
        #pragma unroll
        for(int o=1;o<=2;o<<=1) {
            den[r]+=__shfl_xor_sync(0xffffffffU,den[r],o);
            bias[r]+=__shfl_xor_sync(0xffffffffU,bias[r],o);
        }
    // Merge stream 1 into stream 0 through shared memory (the ring is idle).
    __syncthreads();
    hot_flush(stages-1);
    float* xfer=reinterpret_cast<float*>(l0_smem)+static_cast<std::size_t>(warp%6)*32*70;
    if(stream==1) {
        #pragma unroll
        for(int n=0;n<16;++n)
            #pragma unroll
            for(int e=0;e<4;++e) xfer[(4*n+e)*32+lane]=acc[n][e];
        xfer[64*32+lane]=mx[0];xfer[65*32+lane]=mx[1];
        xfer[66*32+lane]=den[0];xfer[67*32+lane]=den[1];
        xfer[68*32+lane]=bias[0];xfer[69*32+lane]=bias[1];
    }
    __syncthreads();
    if(stream==1) return;
    {
        const float om[2]={xfer[64*32+lane],xfer[65*32+lane]};
        const float od[2]={xfer[66*32+lane],xfer[67*32+lane]};
        const float ob[2]={xfer[68*32+lane],xfer[69*32+lane]};
        float a_self[2],a_other[2];
        #pragma unroll
        for(int r=0;r<2;++r) {
            const float m=fmaxf(mx[r],om[r]);
            a_self[r]=mx[r]==-INFINITY?0.f:exp2f(mx[r]-m);
            a_other[r]=om[r]==-INFINITY?0.f:exp2f(om[r]-m);
            mx[r]=m;
            den[r]=den[r]*a_self[r]+od[r]*a_other[r];
            bias[r]=bias[r]*a_self[r]+ob[r]*a_other[r];
        }
        #pragma unroll
        for(int n=0;n<16;++n)
            #pragma unroll
            for(int e=0;e<4;++e)
                acc[n][e]=acc[n][e]*a_self[e>>1]+xfer[(4*n+e)*32+lane]*a_other[e>>1];
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int v=r?v1:v0,row=v/H,head=v%H;
        if(row>=rows) continue;
        float* slot=workspace+
            ((static_cast<std::size_t>(row)*kKVHeads+kv_head)*slot_stride+segment)*kFastFusedFlashStride;
        #pragma unroll
        for(int c=0;c<2;++c) {
            float* dst=slot+head*kHeadDim+128*vgroup+16*(2*t+c);
            #pragma unroll
            for(int n=0;n<16;n+=4)
                *reinterpret_cast<float4*>(dst+n)=make_float4(
                    acc[n][2*r+c]*l0_code_scale<Bits>(n)+bias[r],acc[n+1][2*r+c]*l0_code_scale<Bits>(n+1)+bias[r],
                    acc[n+2][2*r+c]*l0_code_scale<Bits>(n+2)+bias[r],acc[n+3][2*r+c]*l0_code_scale<Bits>(n+3)+bias[r]);
        }
        if(t==0&&vgroup==0) {
            slot[kFastFusedFlashValues+head]=mx[r]*0.6931471805599453f;
            slot[kFastFusedFlashValues+H+head]=den[r];
        }
    }
}

// Merges the live history segments of one (row, query head) in the rotated
// value basis: normalized FP16 numerator into `rotated` (un-rotated and
// re-scaled by l0_rotate_kernel), max and denominator into the slot. The
// `hot_chunks` exact hot partials follow the segments; their normalized
// original-basis numerator goes to `hot_out` (added by l0_rotate_kernel) and
// row 0 publishes its history LSE (log2) as the next round's candidate reference.
__global__ void __launch_bounds__(256) l0_history_merge_kernel(const float* workspace,
    float* history_slot,__half* rotated,int rows,int segments,int keys,int position,
    int capacity,const int* position_device,int slot_stride,int hot_chunks,float* hot_ref,
    float* hot_out,l0::HotView hot,int hot_insert,float hot_lambda) {
    constexpr int H=kFastFusedFlashHeads;
    // CTAs past the (row, query head) merges select the next round's hot rows.
    if(static_cast<int>(blockIdx.x)>=rows*kQHeads) {
        l0_hot_select(hot,static_cast<int>(blockIdx.x)-rows*kQHeads,
            position_device?*position_device:position,capacity,hot_insert,hot_lambda);
        return;
    }
    const int row=blockIdx.x/kQHeads,qh=blockIdx.x%kQHeads,kv=qh/H,head=qh%H,j=threadIdx.x;
    if(row>=rows) return;
    const int base=position_device?*position_device:position;
    const int end=min(l0::history_end(base),capacity);
    (void)keys;
    const l0::HistorySplit split=l0::history_split(end,segments,hot_chunks?l0_hot_tiles(hot.slots)%segments:0);
    const int live=l0::history_live(split,end,segments)*kL0HistoryStreams;
    const float* slots=workspace+(static_cast<std::size_t>(row)*kKVHeads+kv)*slot_stride*
        kFastFusedFlashStride;
    const int total=live+hot_chunks;
    const auto slot_index=[&](int s) { return s<live?s:segments*kL0HistoryStreams+(s-live); };
    float gmax=-INFINITY;
    for(int s=0;s<total;++s) gmax=fmaxf(gmax,slots[slot_index(s)*kFastFusedFlashStride+kFastFusedFlashValues+head]);
    float den=0.0f,num=0.0f,hot_num=0.0f;
    if(gmax>-INFINITY)
        for(int s=0;s<total;++s) {
            const float* slot=slots+slot_index(s)*kFastFusedFlashStride;
            const float sc=expf(slot[kFastFusedFlashValues+head]-gmax);
            den+=slot[kFastFusedFlashValues+H+head]*sc;
            (s<live?num:hot_num)+=slot[head*kHeadDim+j]*sc;
        }
    const std::size_t at=(static_cast<std::size_t>(row)*kQHeads+qh)*kHeadDim+j;
    rotated[at]=__float2half_rn(den>0.0f?num/den:0.0f);
    if(hot_out) hot_out[at]=den>0.0f?hot_num/den:0.0f;
    if(j==0) {
        float* out=history_slot+(static_cast<std::size_t>(row)*kKVHeads+kv)*kFastFusedFlashStride;
        out[kFastFusedFlashValues+head]=gmax;
        out[kFastFusedFlashValues+H+head]=den;
        if(hot_ref && row==0 && den>0.0f) hot_ref[qh]=(gmax+logf(den))*1.4426950408889634f;
    }
}

// L0 OSCAR prefill history: FA2-shaped (64 query rows x 2 query heads per
// CTA, 32-key tiles) over the INT4 / INT2 history [kSink, history_end) of a prefill
// chunk whose rows all follow it, entirely on s8/u8 MMAs over the raw codes:
//   QK  int8 rotated queries (per 128-dim group, quantized in registers)
//       against the K codes, as in l0_history_mma_kernel. Score columns are
//       permuted so this lane's scores are keys 4t..4t+3 and 16+4t..16+4t+3,
//       the A-fragment keys of the PV MMA.
//   PV  P scaled by the per-key V scale (per 128-dim group), quantized to u8
//       per (row, tile), against the V codes read in place (c 16^j / c 4^j), with the
//       V zero term as a per-row bias; each n-tile's int32 tile sum is folded
//       into the FP32 accumulator.
// Writes the normalized rotated-basis output and (max, denominator).
constexpr int kL0PrefillSplits=4;
constexpr int kL0PrefillThreads=256;
// Key rows of l0_prefill_row_stride words (32 / 16 code words + 4 pad);
// dynamic shared memory: K and V raw tiles (double-buffered), meta, query
// fragments.
template<int Bits> __host__ __device__ constexpr int l0_prefill_row_stride() {
    return voidinfer::exl3::l0_oscar_format::CodeFormat<Bits>::code_bytes/4+4;
}
template<int Bits> constexpr std::size_t l0_prefill_smem_bytes() {
    return 2*2*static_cast<std::size_t>(fa2_prefill::BN)*l0_prefill_row_stride<Bits>()*sizeof(unsigned)+
        2*2*static_cast<std::size_t>(fa2_prefill::BN)*sizeof(uint2)+8*8*32*sizeof(uint4);
}
template<int Bits>
__global__ void __launch_bounds__(256,1) l0_history_prefill_kernel(
    const std::uint16_t* q_rot_bits,const float* q_mu,const std::uint8_t* k_codes,
    const std::uint8_t* v_codes,const __half* k_meta,const __half* v_meta,
    float* out,float* stats,int rows,int history_end){
  // blockIdx.z = head group * kL0PrefillSplits + split over the history tiles.
  using namespace fa2_prefill;
  constexpr float kLog2Scale=0.0625f*1.4426950408889634f;
  constexpr int kCodeBytes=voidinfer::exl3::l0_oscar_format::CodeFormat<Bits>::code_bytes;
  constexpr int kChunks=kCodeBytes/16;   // 16-byte code chunks per key (8 INT4, 4 INT2)
  const auto* q=reinterpret_cast<const __half*>(q_rot_bits);
  // Two raw tile stages (cp.async a tile ahead): codes and FP16 meta.
  // Key rows of kRowStride words with the 16-byte chunks XOR-swizzled by
  // (key >> 2) & (kChunks - 1): the QK code loads and the PV byte transposes
  // are bank-conflict free.
  constexpr int kRowStride=l0_prefill_row_stride<Bits>();
  const auto word_at=[](int key,int word) {
      return key*kRowStride+(((word>>2)^((key>>2)&(kChunks-1)))<<2)+(word&3);
  };
  extern __shared__ __align__(16) unsigned char l0_prefill_smem[];
  using RawTile=unsigned[BN*kRowStride];
  using MetaTile=uint2[BN];
  RawTile* kraw_s=reinterpret_cast<RawTile*>(l0_prefill_smem);
  RawTile* vraw_s=kraw_s+2;
  MetaTile* kmeta_s=reinterpret_cast<MetaTile*>(vraw_s+2);
  MetaTile* vmeta_s=kmeta_s+2;
  // Per-warp int8 query fragments ([k-step] as uint4, lane-consecutive),
  // read back each tile instead of holding 32 registers.
  using QaWarp=uint4[8][32];
  QaWarp* qa_s=reinterpret_cast<QaWarp*>(vmeta_s+2);
  const int query_base=blockIdx.x*BM;
  const int kv_head=blockIdx.y;
  const int head_group=blockIdx.z/kL0PrefillSplits,split=blockIdx.z%kL0PrefillSplits;
  const int history_tiles=(history_end-l0::kSink)/BN;
  const int split_tiles=(history_tiles+kL0PrefillSplits-1)/kL0PrefillSplits;
  const int split_begin=l0::kSink+split*split_tiles*BN;
  const int split_end=min(history_end,split_begin+split_tiles*BN);
  out+=static_cast<std::size_t>(split)*rows*kQHeads*kHeadDim;
  stats+=static_cast<std::size_t>(split)*rows*kQHeads*2;
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  const int head_local=warp>>2, m_base=(warp&3)*16;
  const int g=lane>>2, t=lane&3;
  const int row0=m_base+g, row1=row0+8;
  const int q_head=kv_head*(kQHeads/kKVHeads)+head_group*2+head_local;
  const int active_rows=min(BM,rows-query_base);
  if(active_rows<=0) return;
  const bool l0r=row0<active_rows, l1r=row1<active_rows;
  // Int8 A fragments. INT4: k-step s reads code word W = 8 (s >> 1) + 2t +
  // (s & 1) of each key; byte i of (w >> 4j) & 0x0F0F0F0F is dim 8 W + 2 i + j;
  // this lane holds rows g, g + 8 at the dims of words {8k + 2t, 8k + 2t + 1}.
  // INT2: k-step pair m reads code word W = 8 (m >> 1) + 2t + (m & 1); byte i
  // of (w >> 2j) & 0x03030303 is dim 16 W + 4 i + j; this lane holds the dims
  // of words {2t, 2t+1, 8+2t, 9+2t}.

  float qs[2][2],qsum[2][2],qmu[2];
  if constexpr(Bits==4) {
    float x[2][8][8];
    #pragma unroll
    for(int r=0;r<2;++r) {
      const bool live=r?l1r:l0r;
      const __half* src=q+(static_cast<std::size_t>(query_base+(live?(r?row1:row0):0))*kQHeads+q_head)*kHeadDim;
      #pragma unroll
      for(int st=0;st<8;++st) {
        const int W=8*(st>>1)+2*t+(st&1);
        uint4 raw=make_uint4(0,0,0,0);
        if(live) raw=*reinterpret_cast<const uint4*>(src+8*W);
        const __half* h=reinterpret_cast<const __half*>(&raw);
        #pragma unroll
        for(int d=0;d<8;++d) x[r][st][d]=__half2float(h[d]);
      }
      qmu[r]=live?q_mu[(query_base+(r?row1:row0))*kQHeads+q_head]*kLog2Scale:0.f;
    }
    #pragma unroll
    for(int r=0;r<2;++r)
      #pragma unroll
      for(int grp=0;grp<2;++grp) {
        float amax=0.f,sum=0.f;
        #pragma unroll
        for(int st=4*grp;st<4*grp+4;++st)
          #pragma unroll
          for(int d=0;d<8;++d) { amax=fmaxf(amax,fabsf(x[r][st][d])); sum+=x[r][st][d]; }
        amax=fmaxf(amax,__shfl_xor_sync(0xffffffffU,amax,1)); amax=fmaxf(amax,__shfl_xor_sync(0xffffffffU,amax,2));
        sum+=__shfl_xor_sync(0xffffffffU,sum,1); sum+=__shfl_xor_sync(0xffffffffU,sum,2);
        const float inv=amax>0.f?127.f/amax:0.f;
        #pragma unroll
        for(int st=4*grp;st<4*grp+4;++st)
          #pragma unroll
          for(int d=0;d<8;++d) x[r][st][d]=static_cast<float>(__float2int_rn(x[r][st][d]*inv));
        qs[r][grp]=amax/127.f*kLog2Scale; qsum[r][grp]=sum*kLog2Scale;
      }
    #pragma unroll
    for(int st=0;st<8;++st) {
      unsigned w[4]={0u,0u,0u,0u};
      #pragma unroll
      for(int i=0;i<4;++i) {
        w[0]|=(static_cast<unsigned>(static_cast<int>(x[0][st][2*i]))&255u)<<(8*i);
        w[1]|=(static_cast<unsigned>(static_cast<int>(x[1][st][2*i]))&255u)<<(8*i);
        w[2]|=(static_cast<unsigned>(static_cast<int>(x[0][st][2*i+1]))&255u)<<(8*i);
        w[3]|=(static_cast<unsigned>(static_cast<int>(x[1][st][2*i+1]))&255u)<<(8*i);
      }
      qa_s[warp][st][lane]=make_uint4(w[0],w[1],w[2],w[3]);
    }
  } else {
    float x[2][4][16];
    #pragma unroll
    for(int r=0;r<2;++r) {
      const bool live=r?l1r:l0r;
      const __half* src=q+(static_cast<std::size_t>(query_base+(live?(r?row1:row0):0))*kQHeads+q_head)*kHeadDim;
      #pragma unroll
      for(int m=0;m<4;++m) {
        const int W=8*(m>>1)+2*t+(m&1);
        uint4 raw[2]={make_uint4(0,0,0,0),make_uint4(0,0,0,0)};
        if(live) { raw[0]=*reinterpret_cast<const uint4*>(src+16*W); raw[1]=*reinterpret_cast<const uint4*>(src+16*W+8); }
        const __half* h=reinterpret_cast<const __half*>(raw);
        #pragma unroll
        for(int d=0;d<16;++d) x[r][m][d]=__half2float(h[d]);
      }
      qmu[r]=live?q_mu[(query_base+(r?row1:row0))*kQHeads+q_head]*kLog2Scale:0.f;
    }
    #pragma unroll
    for(int r=0;r<2;++r)
      #pragma unroll
      for(int grp=0;grp<2;++grp) {
        float amax=0.f,sum=0.f;
        #pragma unroll
        for(int mm=0;mm<2;++mm)
          #pragma unroll
          for(int d=0;d<16;++d) { amax=fmaxf(amax,fabsf(x[r][2*grp+mm][d])); sum+=x[r][2*grp+mm][d]; }
        amax=fmaxf(amax,__shfl_xor_sync(0xffffffffU,amax,1)); amax=fmaxf(amax,__shfl_xor_sync(0xffffffffU,amax,2));
        sum+=__shfl_xor_sync(0xffffffffU,sum,1); sum+=__shfl_xor_sync(0xffffffffU,sum,2);
        const float inv=amax>0.f?127.f/amax:0.f;
        #pragma unroll
        for(int mm=0;mm<2;++mm)
          #pragma unroll
          for(int d=0;d<16;++d) x[r][2*grp+mm][d]=static_cast<float>(__float2int_rn(x[r][2*grp+mm][d]*inv));
        qs[r][grp]=amax/127.f*kLog2Scale; qsum[r][grp]=sum*kLog2Scale;
      }
    #pragma unroll
    for(int m=0;m<4;++m)
      #pragma unroll
      for(int sb=0;sb<2;++sb) {
        unsigned w[4]={0u,0u,0u,0u};
        #pragma unroll
        for(int i=0;i<4;++i) {
          const int d=4*i+2*sb;
          w[0]|=(static_cast<unsigned>(static_cast<int>(x[0][m][d]))&255u)<<(8*i);
          w[1]|=(static_cast<unsigned>(static_cast<int>(x[1][m][d]))&255u)<<(8*i);
          w[2]|=(static_cast<unsigned>(static_cast<int>(x[0][m][d+1]))&255u)<<(8*i);
          w[3]|=(static_cast<unsigned>(static_cast<int>(x[1][m][d+1]))&255u)<<(8*i);
        }
        qa_s[warp][2*m+sb][lane]=make_uint4(w[0],w[1],w[2],w[3]);
      }
  }
  // acc[16 h + n][e]: rows g (e < 2) / g + 8, dims 128 h + 16 (2t + (e & 1)) + n,
  // in code-scale units (l0_code_scale, codes read in place); bias[r][h] is the
  // V zero term.
  float acc[32][4];
  #pragma unroll
  for(int n=0;n<32;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.f;
  float run_max[2]={-3.402823466e+38F,-3.402823466e+38F}, den[2]={0.f,0.f},bias[2][2]={{0.f,0.f},{0.f,0.f}};
  const bool warp_live=m_base<active_rows;
  // Raw tile: K codes, V codes (16 B chunks), K and V meta (32 x 8 B each).
  const auto fetch=[&](int tile_first,int buf){
    for(int c=tid;c<BN*(2*kChunks+2);c+=256) {
      const int part=c<BN*2*kChunks?c/(BN*kChunks):2+(c-BN*2*kChunks)/BN;   // 0 K, 1 V, 2 K meta, 3 V meta
      const int idx=c<BN*2*kChunks?c%(BN*kChunks):(c-BN*2*kChunks)%BN;
      const int key=tile_first+(part<2?idx/kChunks:idx);
      const bool valid=key<split_end;
      const std::size_t cell=static_cast<std::size_t>(valid?key:split_begin)*kKVHeads+kv_head;
      if(part==0) l0_cp16(&kraw_s[buf][word_at(idx/kChunks,(idx%kChunks)*4)],k_codes+cell*kCodeBytes+(idx%kChunks)*16,valid);
      else if(part==1) l0_cp16(&vraw_s[buf][word_at(idx/kChunks,(idx%kChunks)*4)],v_codes+cell*kCodeBytes+(idx%kChunks)*16,valid);
      else if(part==2) l0_cp8(&kmeta_s[buf][idx],k_meta+cell*l0::kMetaFloats,valid);
      else l0_cp8(&vmeta_s[buf][idx],v_meta+cell*l0::kMetaFloats,valid);
    }
    asm volatile("cp.async.commit_group;\n");
  };
  int buf=0;
  if(split_begin<split_end) fetch(split_begin,0);
  for(int first=split_begin;first<split_end;first+=BN,buf^=1){
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    if(first+BN<split_end) fetch(first+BN,buf^1);
    if(!warp_live) continue;
    const unsigned* kraw=kraw_s[buf];
    const unsigned* vraw=vraw_s[buf];
    // QK. Score n-tile j column c is key 4 (c >> 1) + 2 (j & 1) + (c & 1) + 16 (j >> 1).
    int si[2][BN/8][4];
    #pragma unroll
    for(int h=0;h<2;++h)
      #pragma unroll
      for(int j=0;j<BN/8;++j) si[h][j][0]=si[h][j][1]=si[h][j][2]=si[h][j][3]=kL0IntMagic;
    if constexpr(Bits==4) {
      uint2 kc[BN/8][4];
      #pragma unroll
      for(int j=0;j<BN/8;++j) {
        const int key=4*(g>>1)+2*(j&1)+(g&1)+16*(j>>1);
        #pragma unroll
        for(int k=0;k<4;++k) kc[j][k]=*reinterpret_cast<const uint2*>(kraw+word_at(key,8*k+2*t));
      }
      #pragma unroll
      for(int st=0;st<8;++st) {
        const uint4 f=qa_s[warp][st][lane];
        const unsigned qa[4]={f.x,f.y,f.z,f.w};
        #pragma unroll
        for(int j=0;j<BN/8;++j) {
          const unsigned w=(st&1)?kc[j][st>>1].y:kc[j][st>>1].x;
          l0_imma(si[st>>2][j],qa,w&0x0F0F0F0FU,(w>>4)&0x0F0F0F0FU);
        }
      }
    } else {
      uint2 kc[BN/8][2];
      #pragma unroll
      for(int j=0;j<BN/8;++j) {
        const int key=4*(g>>1)+2*(j&1)+(g&1)+16*(j>>1);
        kc[j][0]=*reinterpret_cast<const uint2*>(kraw+word_at(key,2*t));
        kc[j][1]=*reinterpret_cast<const uint2*>(kraw+word_at(key,8+2*t));
      }
      #pragma unroll
      for(int m=0;m<4;++m)
        #pragma unroll
        for(int sb=0;sb<2;++sb) {
          const uint4 f=qa_s[warp][2*m+sb][lane];
          const unsigned qa[4]={f.x,f.y,f.z,f.w};
          #pragma unroll
          for(int j=0;j<BN/8;++j) {
            const unsigned w=(m&1)?kc[j][m>>1].y:kc[j][m>>1].x;
            l0_imma(si[m>>1][j],qa,(w>>(4*sb))&0x03030303U,(w>>(4*sb+2))&0x03030303U);
          }
        }
    }
    // This lane's keys: 4t + (i & 3) + 16 (i >> 2). Per-key factors (shared
    // by both rows): K scale per group, K zero term -z s per group; same for V.
    float kf[8][4],vf[8][4];
    #pragma unroll
    for(int i=0;i<8;++i) {
      const int key=4*t+(i&3)+16*(i>>2);
      const float4 k=l0_meta4(&kmeta_s[buf][key]);
      const float4 v=l0_meta4(&vmeta_s[buf][key]);
      kf[i][0]=k.x;kf[i][1]=k.z;kf[i][2]=-k.y*k.x;kf[i][3]=-k.w*k.z;
      vf[i][0]=v.x;vf[i][1]=v.z;vf[i][2]=-v.y*v.x;vf[i][3]=-v.w*v.z;
    }
    const bool whole=first+BN<=split_end;
    float s[2][8];
    float tmax[2]={-3.402823466e+38F,-3.402823466e+38F};
    #pragma unroll
    for(int j=0;j<BN/8;++j)
      #pragma unroll
      for(int e=0;e<4;++e){
        const int r=e>>1,i=4*(j>>1)+2*(j&1)+(e&1),key=4*t+(i&3)+16*(i>>2);
        const float base=fmaf(qsum[r][0],kf[i][2],fmaf(qsum[r][1],kf[i][3],qmu[r]));
        const float v=fmaf(qs[r][0]*kf[i][0],l0_int_value(si[0][j][e]),
                           fmaf(qs[r][1]*kf[i][1],l0_int_value(si[1][j][e]),base));
        const bool live=(r?l1r:l0r)&&(whole||first+key<split_end);
        s[r][i]=live?v:-3.402823466e+38F; tmax[r]=fmaxf(tmax[r],s[r][i]);
      }
    #pragma unroll
    for(int r=0;r<2;++r){ tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffu,tmax[r],1)); tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffu,tmax[r],2)); }
    float scale[2];
    #pragma unroll
    for(int r=0;r<2;++r){ const float nm=fmaxf(run_max[r],tmax[r]); scale[r]=(nm==run_max[r])?1.f:exp2f(run_max[r]-nm); run_max[r]=nm; }
    // P (masked scores give exp2(-huge) = 0), P V-scaled per group.
    float pv[2][2][8];
    float pmax[2][2]={{0.f,0.f},{0.f,0.f}};
    float rs[2]={0.f,0.f},tile_bias[2][2]={{0.f,0.f},{0.f,0.f}};
    #pragma unroll
    for(int r=0;r<2;++r)
      #pragma unroll
      for(int i=0;i<8;++i) {
        const float p=exp2f(s[r][i]-run_max[r]);
        rs[r]+=p;
        tile_bias[r][0]=fmaf(p,vf[i][2],tile_bias[r][0]);
        tile_bias[r][1]=fmaf(p,vf[i][3],tile_bias[r][1]);
        pv[r][0][i]=p*vf[i][0]; pv[r][1][i]=p*vf[i][1];
        pmax[r][0]=fmaxf(pmax[r][0],pv[r][0][i]); pmax[r][1]=fmaxf(pmax[r][1],pv[r][1][i]);
      }
    float qscale[2][2];
    unsigned pa[2][4];   // [h]: a0 row g keys 4t.., a1 row g+8, a2 row g keys 16+4t.., a3 row g+8
    #pragma unroll
    for(int r=0;r<2;++r)
      #pragma unroll
      for(int h=0;h<2;++h) {
        float mx=pmax[r][h];
        mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,1)); mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,2));
        const float inv=mx>0.f?255.f/mx:0.f;
        qscale[r][h]=mx*(1.f/255.f);
        #pragma unroll
        for(int half_index=0;half_index<2;++half_index) {
          // Round to u8 with the 1.5 * 2^23 magic: the low byte of the sum's bits.
          unsigned b[4];
          #pragma unroll
          for(int i=0;i<4;++i) b[i]=__float_as_uint(fmaf(pv[r][h][4*half_index+i],inv,12582912.0f));
          pa[h][2*half_index+r]=__byte_perm(__byte_perm(b[0],b[1],0x0040),__byte_perm(b[2],b[3],0x0040),0x5410);
        }
      }
    #pragma unroll
    for(int r=0;r<2;++r){
      rs[r]+=__shfl_xor_sync(0xffffffffu,rs[r],1); rs[r]+=__shfl_xor_sync(0xffffffffu,rs[r],2);
      den[r]=den[r]*scale[r]+rs[r];
    }
    if(scale[0]!=1.f||scale[1]!=1.f) {
      #pragma unroll
      for(int n=0;n<32;++n){ acc[n][0]*=scale[0]; acc[n][1]*=scale[0]; acc[n][2]*=scale[1]; acc[n][3]*=scale[1]; }
      #pragma unroll
      for(int h=0;h<2;++h){ bias[0][h]*=scale[0]; bias[1][h]*=scale[1]; }
    }
    #pragma unroll
    for(int h=0;h<2;++h){ bias[0][h]+=tile_bias[0][h]; bias[1][h]+=tile_bias[1][h]; }
    // PV. B column g of n-tile 16 h + n is dim 128 h + 16 g + n. INT4: word
    // 16 h + 2 g + (n >> 3), byte (n & 7) >> 1, nibble n & 1. INT2: word 8 h + g,
    // byte n >> 2, bits 2 (n & 3). b0 keys 4t..4t+3, b1 keys 16+4t..16+4t+3.
    constexpr int kWordsPerColumn=Bits==4?2:1;
    #pragma unroll
    for(int h=0;h<2;++h) {
      unsigned u[kWordsPerColumn][2][4];
      #pragma unroll
      for(int half_word=0;half_word<kWordsPerColumn;++half_word)
        #pragma unroll
        for(int quad=0;quad<2;++quad) {
          const int k0=16*quad+4*t,word=Bits==4?16*h+2*g+half_word:8*h+g;
          const unsigned w0=vraw[word_at(k0,word)],w1=vraw[word_at(k0+1,word)];
          const unsigned w2=vraw[word_at(k0+2,word)],w3=vraw[word_at(k0+3,word)];
          const unsigned t0=__byte_perm(w0,w1,0x5140),t1=__byte_perm(w2,w3,0x5140);
          const unsigned t2=__byte_perm(w0,w1,0x7362),t3=__byte_perm(w2,w3,0x7362);
          u[half_word][quad][0]=__byte_perm(t0,t1,0x5410);u[half_word][quad][1]=__byte_perm(t0,t1,0x7632);
          u[half_word][quad][2]=__byte_perm(t2,t3,0x5410);u[half_word][quad][3]=__byte_perm(t2,t3,0x7632);
        }
      const unsigned a[4]={pa[h][0],pa[h][1],pa[h][2],pa[h][3]};
      #pragma unroll
      for(int n=0;n<16;++n) {
        int c[4]={kL0IntMagic,kL0IntMagic,kL0IntMagic,kL0IntMagic};
        if constexpr(Bits==4) {
          const unsigned mask=0x0F0F0F0FU<<(4*(n&1));
          l0_immau(c,a,u[n>>3][0][(n&7)>>1]&mask,u[n>>3][1][(n&7)>>1]&mask);
        } else {
          const unsigned mask=0x03030303U<<(2*(n&3));
          l0_immau(c,a,u[0][0][n>>2]&mask,u[0][1][n>>2]&mask);
        }
        acc[16*h+n][0]=fmaf(qscale[0][h],l0_int_value(c[0]),acc[16*h+n][0]);
        acc[16*h+n][1]=fmaf(qscale[0][h],l0_int_value(c[1]),acc[16*h+n][1]);
        acc[16*h+n][2]=fmaf(qscale[1][h],l0_int_value(c[2]),acc[16*h+n][2]);
        acc[16*h+n][3]=fmaf(qscale[1][h],l0_int_value(c[3]),acc[16*h+n][3]);
      }
    }
  }
  #pragma unroll
  for(int r=0;r<2;++r)
    #pragma unroll
    for(int h=0;h<2;++h){ bias[r][h]+=__shfl_xor_sync(0xffffffffu,bias[r][h],1); bias[r][h]+=__shfl_xor_sync(0xffffffffu,bias[r][h],2); }
  #pragma unroll
  for(int r=0;r<2;++r){
    const int row=r==0?row0:row1;
    if(row>=active_rows) continue;
    const std::size_t row_base=(static_cast<std::size_t>(query_base+row)*kQHeads+q_head)*kHeadDim;
    const float inv=den[r]>0.f?1.f/den[r]:0.f;
    #pragma unroll
    for(int h=0;h<2;++h)
      #pragma unroll
      for(int c=0;c<2;++c) {
        float* dst=out+row_base+128*h+16*(2*t+c);
        #pragma unroll
        for(int n=0;n<16;n+=4)
          *reinterpret_cast<float4*>(dst+n)=make_float4(
            (acc[16*h+n][2*r+c]*l0_code_scale<Bits>(n)+bias[r][h])*inv,
            (acc[16*h+n+1][2*r+c]*l0_code_scale<Bits>(n+1)+bias[r][h])*inv,
            (acc[16*h+n+2][2*r+c]*l0_code_scale<Bits>(n+2)+bias[r][h])*inv,
            (acc[16*h+n+3][2*r+c]*l0_code_scale<Bits>(n+3)+bias[r][h])*inv);
      }
    if(t==0){
      const std::size_t stat=(static_cast<std::size_t>(query_base+row)*kQHeads+q_head)*2;
      stats[stat]=run_max[r]*0.6931471805599453f; stats[stat+1]=den[r];
    }
  }
}

// Y[v] = X[v] M for the (row, head) vectors of one KV head, v = row * 6 + head
// at [row][kv * 6 + head][256]: the rotations q R_k and o' R_v^T as FP16
// tensor-core GEMMs. CTA = 64 vectors x Cols output columns (blockIdx.y =
// column block, blockIdx.z = KV head), 4 warps of 16 vectors; `mu` (column
// block 0) also forms q . mu. Slot output writes the fused-flash history slot
// [row][kv][head][256] scaled by the slot denominator (rotated inputs are
// normalized numerators).
constexpr int kL0RotVectors=64,kL0RotThreads=128;
constexpr int kL0RotAStride=kHeadDim+8;
template<int Cols> constexpr std::size_t l0_rot_smem() {
    return (static_cast<std::size_t>(kL0RotVectors)*kL0RotAStride+kHeadDim*(Cols+8))*sizeof(half);
}
__device__ __forceinline__ void l0_rot_store(__half* y,float a,float b) {
    *reinterpret_cast<__half2*>(y)=__floats2half2_rn(a,b);
}
__device__ __forceinline__ void l0_rot_store(float* y,float a,float b) {
    *reinterpret_cast<float2*>(y)=make_float2(a,b);
}
template<typename Out,int Cols,bool Slot>
__global__ void __launch_bounds__(kL0RotThreads) l0_rotate_kernel(const __half* x,
    const __half* m_bank,Out* y,int rows,const float* mu,float* q_mu,const float* add) {
    constexpr int H=kFastFusedFlashHeads;
    constexpr int BStride=Cols+8;
    extern __shared__ __align__(16) unsigned char l0_rot_smem_raw[];
    half* xs=reinterpret_cast<half*>(l0_rot_smem_raw);
    half* ms=xs+kL0RotVectors*kL0RotAStride;
    const int kv=static_cast<int>(blockIdx.z),cb=static_cast<int>(blockIdx.y);
    const int v0=static_cast<int>(blockIdx.x)*kL0RotVectors,total=rows*H;
    const int tid=static_cast<int>(threadIdx.x),warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
    const auto vector_at=[&](int v){
        return (static_cast<std::size_t>(v/H)*kQHeads+kv*H+v%H)*kHeadDim;
    };
    const __half* m=m_bank+static_cast<std::size_t>(kv)*kHeadDim*kHeadDim+cb*Cols;
    for(int i=tid;i<kHeadDim*(Cols/8);i+=kL0RotThreads) {
        const int r=i/(Cols/8),c=(i%(Cols/8))*8;
        *reinterpret_cast<uint4*>(ms+r*BStride+c)=
            *reinterpret_cast<const uint4*>(m+static_cast<std::size_t>(r)*kHeadDim+c);
    }
    for(int i=tid;i<kL0RotVectors*(kHeadDim/8);i+=kL0RotThreads) {
        const int vv=i/(kHeadDim/8),c=(i%(kHeadDim/8))*8,v=v0+vv;
        uint4 value=make_uint4(0,0,0,0);
        if(v<total) value=*reinterpret_cast<const uint4*>(x+vector_at(v)+c);
        *reinterpret_cast<uint4*>(xs+vv*kL0RotAStride+c)=value;
    }
    __syncthreads();
    if(v0+warp*16<total) {
        float acc[Cols/8][4];
        #pragma unroll
        for(int n=0;n<Cols/8;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.0f;
        #pragma unroll 4
        for(int kk=0;kk<kHeadDim;kk+=16) {
            unsigned a[4];
            reg_attn_ldmatrix_x4(a,xs+(warp*16+(lane&15))*kL0RotAStride+kk+(lane>>4)*8);
            #pragma unroll
            for(int pair=0;pair<Cols/16;++pair) {
                unsigned b[4];
                reg_attn_ldmatrix_x4_trans(b,ms+(kk+(lane&7)+((lane>>3)&1)*8)*BStride+
                    pair*16+(lane>>4)*8);
                reg_attn_mma(acc[2*pair],a,b[0],b[1]);
                reg_attn_mma(acc[2*pair+1],a,b[2],b[3]);
            }
        }
        #pragma unroll
        for(int r=0;r<2;++r) {
            const int v=v0+warp*16+g+8*r;
            if(v>=total) continue;
            Out* dst;
            float scale=1.0f;
            if constexpr(Slot) {
                float* slot=reinterpret_cast<float*>(y)+
                    (static_cast<std::size_t>(v/H)*kKVHeads+kv)*kFastFusedFlashStride;
                scale=slot[kFastFusedFlashValues+H+v%H];
                dst=reinterpret_cast<Out*>(slot+(v%H)*kHeadDim+cb*Cols+2*t);
            } else {
                dst=y+vector_at(v)+cb*Cols+2*t;
            }
            if(add) {
                const float* extra=add+vector_at(v)+cb*Cols+2*t;
                #pragma unroll
                for(int n=0;n<Cols/8;++n)
                    l0_rot_store(dst+8*n,(acc[n][2*r]+extra[8*n])*scale,(acc[n][2*r+1]+extra[8*n+1])*scale);
            } else {
                #pragma unroll
                for(int n=0;n<Cols/8;++n) l0_rot_store(dst+8*n,acc[n][2*r]*scale,acc[n][2*r+1]*scale);
            }
        }
    }
    if(mu && cb==0) {
        const float* mk=mu+static_cast<std::size_t>(kv)*kHeadDim;
        for(int i=0;i<16;++i) {
            const int v=v0+warp*16+i;
            if(v>=total) break;
            float s=0.0f;
            for(int d=lane;d<kHeadDim;d+=32) s=fmaf(__half2float(xs[(warp*16+i)*kL0RotAStride+d]),mk[d],s);
            for(int o=16;o>0;o>>=1) s+=__shfl_xor_sync(0xffffffffU,s,o);
            if(lane==0) q_mu[(v/H)*kQHeads+kv*H+v%H]=s;
        }
    }
}

template<typename Out,int Cols,bool Slot>
void l0_launch_rotate_cols(const void* x,const __half* m_bank,Out* y,int rows,const float* mu,
    float* q_mu,cudaStream_t stream,const float* add) {
    static const bool configured=[] {
        cuda_check(cudaFuncSetAttribute(l0_rotate_kernel<Out,Cols,Slot>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(l0_rot_smem<Cols>())),
            "configure L0 rotation shared memory");
        return true;
    }();
    (void)configured;
    l0_rotate_kernel<Out,Cols,Slot><<<dim3((rows*kFastFusedFlashHeads+kL0RotVectors-1)/kL0RotVectors,
        kHeadDim/Cols,kKVHeads),kL0RotThreads,l0_rot_smem<Cols>(),stream>>>(
        reinterpret_cast<const __half*>(x),m_bank,y,rows,mu,q_mu,add);
}
// Few vectors (verifier rows): 16-column blocks spread R over 64 CTAs. `add`
// (Slot only): original-basis normalized numerator added before re-scaling.
template<typename Out,bool Slot=false>
void l0_launch_rotate(const void* x,const __half* m_bank,Out* y,int rows,const float* mu,
    float* q_mu,cudaStream_t stream,const float* add=nullptr) {
    if(rows<=16) l0_launch_rotate_cols<Out,16,Slot>(x,m_bank,y,rows,mu,q_mu,stream,add);
    else l0_launch_rotate_cols<Out,64,Slot>(x,m_bank,y,rows,mu,q_mu,stream,add);
}

// Merges the kL0PrefillSplits normalized history partials of one (row, query
// head) into FP16 rotated numerators (normalized) and writes the merged
// (max, denominator) as prefill split `slot`.
__global__ void __launch_bounds__(256) l0_history_split_merge_kernel(const float* rotated,
    const float* rotated_stats,__half* numer,float* split_stats,int rows,int slot) {
    const int vector=static_cast<int>(blockIdx.x),d=static_cast<int>(threadIdx.x);
    const std::size_t split_elements=static_cast<std::size_t>(rows)*kQHeads*kHeadDim;
    const std::size_t split_stat_stride=static_cast<std::size_t>(rows)*kQHeads*2;
    const std::size_t stat=static_cast<std::size_t>(vector)*2;
    float m=-3.402823466e+38F;
    #pragma unroll
    for(int s=0;s<kL0PrefillSplits;++s)
        if(rotated_stats[s*split_stat_stride+stat+1]>0.f) m=fmaxf(m,rotated_stats[s*split_stat_stride+stat]);
    float w[kL0PrefillSplits],den=0.f;
    #pragma unroll
    for(int s=0;s<kL0PrefillSplits;++s) {
        const float dn=rotated_stats[s*split_stat_stride+stat+1];
        w[s]=dn>0.f?dn*expf(rotated_stats[s*split_stat_stride+stat]-m):0.f;
        den+=w[s];
    }
    const float inv=den>0.f?1.f/den:0.f;
    float v=0.f;
    #pragma unroll
    for(int s=0;s<kL0PrefillSplits;++s)
        v=fmaf(w[s]*inv,rotated[s*split_elements+static_cast<std::size_t>(vector)*kHeadDim+d],v);
    numer[static_cast<std::size_t>(vector)*kHeadDim+d]=__float2half_rn(v);
    if(d==0) {
        const std::size_t dst=(static_cast<std::size_t>(slot)*rows*kQHeads+vector)*2;
        split_stats[dst]=m;
        split_stats[dst+1]=den;
    }
}
