#pragma once
// L0 OSCAR history attention (see l0_oscar.cuh). Included by
// full_attention_layer.cu after the verifier MMA helpers; shares their
// fragment layout, online softmax and fused-flash slot format.

namespace l0 = ninfer::exl3::l0_oscar;

// One decode slot: 8 consecutive dims of one key, K and V. Raw codes and
// metadata are fetched into registers a tile ahead of their decode.
struct L0RawSlot {
    unsigned k_bits=0,v_bits=0;
    float k_scale=0.f,k_zero=0.f,v_scale=0.f,v_zero=0.f;
};
__device__ __forceinline__ L0RawSlot l0_load_slot(const std::uint8_t* k_codes,
    const std::uint8_t* v_codes,const float* k_meta,const float* v_meta,int key,int kv_head,
    int dim,bool valid) {
    L0RawSlot r;
    if(!valid) return r;
    const std::size_t cell=static_cast<std::size_t>(key)*kKVHeads+kv_head;
    const int group=dim>>7;
    const float2 km=*reinterpret_cast<const float2*>(k_meta+cell*l0::kMetaFloats+2*group);
    const float2 vm=*reinterpret_cast<const float2*>(v_meta+cell*l0::kMetaFloats+2*group);
    r.k_scale=km.x;r.k_zero=km.y;r.v_scale=vm.x;r.v_zero=vm.y;
    r.k_bits=*reinterpret_cast<const std::uint16_t*>(k_codes+cell*l0::kCodeBytes+(dim>>2));
    r.v_bits=*reinterpret_cast<const std::uint16_t*>(v_codes+cell*l0::kCodeBytes+(dim>>2));
    return r;
}
__device__ __forceinline__ void l0_store_slot(const L0RawSlot& r,half* k_dst,half* v_dst) {
    // x^ = (code - zero) * scale = code * scale + (-zero * scale): one FMA per value.
    const float kb=-r.k_zero*r.k_scale,vb=-r.v_zero*r.v_scale;
    half k8[8],v8[8];
    #pragma unroll
    for(int e=0;e<8;++e) {
        k8[e]=__float2half_rn(fmaf(static_cast<float>((r.k_bits>>(2*e))&3u),r.k_scale,kb));
        v8[e]=__float2half_rn(fmaf(static_cast<float>((r.v_bits>>(2*e))&3u),r.v_scale,vb));
    }
    *reinterpret_cast<uint4*>(k_dst)=*reinterpret_cast<const uint4*>(k8);
    *reinterpret_cast<uint4*>(v_dst)=*reinterpret_cast<const uint4*>(v8);
}

// Verifier history attention over INT2 codes, consumed straight from the
// codes (no FP16 tiles):
//   QK  s8 MMA (m16n8k32) of the int8-quantized rotated queries (per 128-dim
//       group scale) against the raw codes; the per-key K scale and zero are
//       applied to the int32 sums of each group.
//   PV  FP16 MMA whose B operand is the code itself read in place as an FP16
//       subnormal (c 4^e 2^-24); the per-key V scale is folded into P and
//       the V zero term is carried as a per-row bias.
// CTA = (history segment, KV head), 12 warps = (m-tile of 16 query vectors,
// 128-dim value group, key stream). Raw codes and metadata stream through a
// cp.async ring of 64-key stages shared by all warps; each stream takes one
// 16-key block of every 32 keys. The two streams merge in shared memory.
constexpr int kL0HistoryStreams=1;          // partial slots per segment
constexpr int kL0KeyStreams=2;              // in-CTA key streams
constexpr int kL0HistoryThreads=192*kL0KeyStreams;
constexpr int kL0Queries=48;
constexpr int kL0StageKeys=64;
constexpr int kL0Stages=5;
constexpr int kL0RawKeyBytes=2*l0::kCodeBytes+2*l0::kMetaFloats*4;   // K codes, V codes, K meta, V meta
constexpr std::size_t kL0StageBytes=static_cast<std::size_t>(kL0StageKeys)*kL0RawKeyBytes;
constexpr std::size_t kL0MergeBytes=static_cast<std::size_t>(6)*32*70*sizeof(float);
constexpr std::size_t l0_history_smem_bytes() {
    return kL0Stages*kL0StageBytes>kL0MergeBytes?kL0Stages*kL0StageBytes:kL0MergeBytes;
}

__device__ __forceinline__ void l0_cp16(void* dst,const void* src,bool valid) {
    const unsigned d=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(d),"l"(src),"r"(valid?16:0));
}
__device__ __forceinline__ void l0_imma(int (&c)[4],const unsigned (&a)[4],unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
// Undoes the subnormal code scale of PV n-tile n (see the PV loop).
__device__ __forceinline__ constexpr float l0_code_scale(int n) {
    return 16777216.0f/static_cast<float>(1<<(2*((n&7)<5?(n&7):(n&7)-5)));
}

template<int kKeys>
__global__ void __launch_bounds__(kL0HistoryThreads,1) l0_history_mma_kernel(
    const std::uint16_t* q_rot_bits,const float* q_mu,const std::uint8_t* k_codes,
    const std::uint8_t* v_codes,const float* k_meta,const float* v_meta,
    float* workspace,int rows,int position,int capacity,int segments,
    const int* position_device) {
    constexpr int H=kFastFusedFlashHeads;
    constexpr float kLog2Scale=0.0625f*1.4426950408889634f;
    const auto* q_rot=reinterpret_cast<const __half*>(q_rot_bits);
    extern __shared__ __align__(16) unsigned char l0_smem[];
    __shared__ __align__(16) signed char q8[kL0Queries][kHeadDim];
    __shared__ __align__(16) unsigned qfrag[3][4][32][8];
    __shared__ float qinfo[kL0Queries][5];   // qs0, qs1, qsum0, qsum1, mu (log2 units)
    const int segment=static_cast<int>(blockIdx.x);
    const int kv_head=static_cast<int>(blockIdx.y);
    const int tid=static_cast<int>(threadIdx.x);
    const int warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
    const int stream=warp/6,vgroup=(warp/3)&1,mtile=warp%3;
    const int base=position_device?*position_device:position;
    const int end=min(l0::history_end(base),capacity);
    const int span=l0::history_span(end,segments);
    const int first=l0::kSink+segment*span;
    if(first>=end) return;
    const int segment_end=min(first+span,end);
    const int stages=(segment_end-first+kL0StageKeys-1)/kL0StageKeys;
    constexpr int kChunksPerKey=kL0RawKeyBytes/16;   // 10
    const auto fetch=[&](int stage) {
        if(stage<stages) {
            unsigned char* dst=l0_smem+(stage%kL0Stages)*kL0StageBytes;
            const int key0=first+stage*kL0StageKeys;
            for(int c=tid;c<kL0StageKeys*kChunksPerKey;c+=kL0HistoryThreads) {
                const int key_off=c/kChunksPerKey,part=c%kChunksPerKey,key=key0+key_off;
                const bool valid=key<segment_end;
                const std::size_t cell=static_cast<std::size_t>(valid?key:first)*kKVHeads+kv_head;
                const void* src=part<4?static_cast<const void*>(k_codes+cell*l0::kCodeBytes+part*16):
                    part<8?static_cast<const void*>(v_codes+cell*l0::kCodeBytes+(part-4)*16):
                    part==8?static_cast<const void*>(k_meta+cell*l0::kMetaFloats):
                            static_cast<const void*>(v_meta+cell*l0::kMetaFloats);
                l0_cp16(dst+key_off*kL0RawKeyBytes+part*16,src,valid);
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
    __syncthreads();
    // A fragments: k-step pair m uses code word W = 8 (m >> 1) + 2t + (m & 1)
    // of each key; byte i of (w >> 2j) & 0x03030303 is dim 16 W + 4 i + j.
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
    const int v0=mtile*16+g,v1=v0+8;
    float acc[16][4];
    #pragma unroll
    for(int n=0;n<16;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.0f;
    float mx[2]={-INFINITY,-INFINITY},den[2]={0.f,0.f},bias[2]={0.f,0.f};
    for(int stage=0;stage<stages;++stage) {
        asm volatile("cp.async.wait_group %0;\n"::"n"(kL0Stages-2));
        __syncthreads();
        fetch(stage+kL0Stages-1);
        const unsigned char* raw=l0_smem+(stage%kL0Stages)*kL0StageBytes;
        float qi[2][5];
        #pragma unroll
        for(int i=0;i<5;++i) { qi[0][i]=qinfo[v0][i]; qi[1][i]=qinfo[v1][i]; }
        #pragma unroll
        for(int blk=0;blk<kL0StageKeys/(16*kL0KeyStreams);++blk) {
            const int key_off=16*(kL0KeyStreams*blk+stream);
            const int kb=first+stage*kL0StageKeys+key_off;
            const unsigned char* bk=raw+key_off*kL0RawKeyBytes;
            // QK: int32 sums per group.
            int s[2][2][4];
            #pragma unroll
            for(int h=0;h<2;++h)
                #pragma unroll
                for(int j=0;j<2;++j) s[h][j][0]=s[h][j][1]=s[h][j][2]=s[h][j][3]=0;
            uint2 kc[2][2];
            #pragma unroll
            for(int j=0;j<2;++j) {
                const uint2* w=reinterpret_cast<const uint2*>(bk+(8*j+g)*kL0RawKeyBytes);
                kc[j][0]=w[t];kc[j][1]=w[4+t];
            }
            #pragma unroll
            for(int m=0;m<4;++m) {
                const uint4 f0=*reinterpret_cast<const uint4*>(&qfrag[mtile][m][lane][0]);
                const uint4 f1=*reinterpret_cast<const uint4*>(&qfrag[mtile][m][lane][4]);
                const unsigned a[2][4]={{f0.x,f0.y,f0.z,f0.w},{f1.x,f1.y,f1.z,f1.w}};
                #pragma unroll
                for(int j=0;j<2;++j) {
                    const unsigned w=(m&1)?kc[j][m>>1].y:kc[j][m>>1].x;
                    #pragma unroll
                    for(int sb=0;sb<2;++sb)
                        l0_imma(s[m>>1][j],a[sb],(w>>(4*sb))&0x03030303U,(w>>(4*sb+2))&0x03030303U);
                }
            }
            // Keys of this lane's score columns: 2t + (i & 1) + 8 (i >> 1).
            float4 km[4];float2 vm[4];unsigned vw[4];
            #pragma unroll
            for(int i=0;i<4;++i) {
                const unsigned char* r=bk+(2*t+(i&1)+8*(i>>1))*kL0RawKeyBytes;
                km[i]=*reinterpret_cast<const float4*>(r+2*l0::kCodeBytes);
                vm[i]=*reinterpret_cast<const float2*>(r+2*l0::kCodeBytes+16+8*vgroup);
                vw[i]=*reinterpret_cast<const unsigned*>(r+l0::kCodeBytes+4*(8*vgroup+g));
            }
            float sc[2][4];
            float tmax[2]={-INFINITY,-INFINITY};
            #pragma unroll
            for(int j=0;j<2;++j)
                #pragma unroll
                for(int e=0;e<4;++e) {
                    const int r=e>>1,i=2*j+(e&1);
                    const float4 k=km[i];
                    const float v=qi[r][0]*k.x*static_cast<float>(s[0][j][e])+qi[r][1]*k.z*static_cast<float>(s[1][j][e])-
                        qi[r][2]*k.y*k.x-qi[r][3]*k.w*k.z+qi[r][4];
                    sc[j][e]=kb+8*j+2*t+(e&1)<segment_end?v:-INFINITY;
                    tmax[r]=fmaxf(tmax[r],sc[j][e]);
                }
            #pragma unroll
            for(int r=0;r<2;++r) {
                tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffU,tmax[r],1));
                tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffU,tmax[r],2));
            }
            // Lazy rescale: the running max moves only when a score exceeds it
            // by more than 2^kSlack, so P stays <= 2^kSlack (FP16-safe after the
            // V scale).
            constexpr float kSlack=8.0f;
            const bool grow0=tmax[0]>mx[0]+kSlack,grow1=tmax[1]>mx[1]+kSlack;
            if(__any_sync(0xffffffffU,grow0||grow1)) {
                float alpha[2];
                #pragma unroll
                for(int r=0;r<2;++r) {
                    const float next=(r?grow1:grow0)?tmax[r]:mx[r];
                    alpha[r]=mx[r]==-INFINITY?(next==-INFINITY?1.0f:0.0f):exp2f(mx[r]-next);
                    mx[r]=next;
                    den[r]*=alpha[r];bias[r]*=alpha[r];
                }
                #pragma unroll
                for(int n=0;n<16;++n) {
                    acc[n][0]*=alpha[0];acc[n][1]*=alpha[0];
                    acc[n][2]*=alpha[1];acc[n][3]*=alpha[1];
                }
            }
            unsigned pa[4];
            #pragma unroll
            for(int j=0;j<2;++j) {
                float p[4];
                #pragma unroll
                for(int e=0;e<4;++e) {
                    const int r=e>>1,i=2*j+(e&1);
                    p[e]=sc[j][e]==-INFINITY?0.f:exp2f(sc[j][e]-mx[r]);
                    den[r]+=p[e];
                    bias[r]-=p[e]*vm[i].y*vm[i].x;
                }
                pa[2*j]=reg_attn_pack(__float2half_rn(p[0]*vm[2*j].x),__float2half_rn(p[1]*vm[2*j+1].x));
                pa[2*j+1]=reg_attn_pack(__float2half_rn(p[2]*vm[2*j].x),__float2half_rn(p[3]*vm[2*j+1].x));
            }
            // PV: n-tile n covers dims 128 vgroup + 16 c + n (column c = lane g
            // of B). Codes stay in place: n < 5 reads c 4^n 2^-24, n >= 5 reads
            // after a 10-bit shift as c 4^(n-5) 2^-24 (FP16 subnormals).
            const unsigned z0lo=__byte_perm(vw[0],vw[1],0x5410),z0hi=__byte_perm(vw[0],vw[1],0x7632);
            const unsigned z1lo=__byte_perm(vw[2],vw[3],0x5410),z1hi=__byte_perm(vw[2],vw[3],0x7632);
            #pragma unroll
            for(int n=0;n<16;++n) {
                const unsigned z0=n<8?z0lo:z0hi,z1=n<8?z1lo:z1hi;
                const int sh=(n&7)<5?0:10,e=(n&7)<5?(n&7):(n&7)-5;
                const unsigned mask=0x00030003U<<(2*e);
                reg_attn_mma(acc[n],pa,(z0>>sh)&mask,(z1>>sh)&mask);
            }
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
    const int slots=segments*kL0HistoryStreams;
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int v=r?v1:v0,row=v/H,head=v%H;
        if(row>=rows) continue;
        float* slot=workspace+
            ((static_cast<std::size_t>(row)*kKVHeads+kv_head)*slots+segment)*kFastFusedFlashStride;
        #pragma unroll
        for(int c=0;c<2;++c) {
            float* dst=slot+head*kHeadDim+128*vgroup+16*(2*t+c);
            #pragma unroll
            for(int n=0;n<16;n+=4)
                *reinterpret_cast<float4*>(dst+n)=make_float4(
                    acc[n][2*r+c]*l0_code_scale(n)+bias[r],acc[n+1][2*r+c]*l0_code_scale(n+1)+bias[r],
                    acc[n+2][2*r+c]*l0_code_scale(n+2)+bias[r],acc[n+3][2*r+c]*l0_code_scale(n+3)+bias[r]);
        }
        if(t==0&&vgroup==0) {
            slot[kFastFusedFlashValues+head]=mx[r]*0.6931471805599453f;
            slot[kFastFusedFlashValues+H+head]=den[r];
        }
    }
}

// Merges the live history segments of one (row, query head) in the rotated
// value basis: numerator into `rotated`, max and denominator into the slot.
__global__ void __launch_bounds__(256) l0_history_merge_kernel(const float* workspace,
    float* history_slot,float* rotated,int rows,int segments,int keys,int position,
    int capacity,const int* position_device) {
    constexpr int H=kFastFusedFlashHeads;
    const int row=blockIdx.x/kQHeads,qh=blockIdx.x%kQHeads,kv=qh/H,head=qh%H,j=threadIdx.x;
    if(row>=rows) return;
    const int base=position_device?*position_device:position;
    const int end=min(l0::history_end(base),capacity);
    (void)keys;
    const int span=l0::history_span(end,segments);
    // Two partial slots (key streams) per history segment.
    const int live=(end>l0::kSink?min(segments,(end-l0::kSink+span-1)/span):0)*kL0HistoryStreams;
    const float* slots=workspace+(static_cast<std::size_t>(row)*kKVHeads+kv)*segments*
        kL0HistoryStreams*kFastFusedFlashStride;
    float gmax=-INFINITY;
    for(int s=0;s<live;++s) gmax=fmaxf(gmax,slots[s*kFastFusedFlashStride+kFastFusedFlashValues+head]);
    float den=0.0f,num=0.0f;
    if(gmax>-INFINITY)
        for(int s=0;s<live;++s) {
            const float* slot=slots+s*kFastFusedFlashStride;
            const float sc=expf(slot[kFastFusedFlashValues+head]-gmax);
            den+=slot[kFastFusedFlashValues+H+head]*sc;
            num+=slot[head*kHeadDim+j]*sc;
        }
    rotated[(static_cast<std::size_t>(row)*kQHeads+qh)*kHeadDim+j]=num;
    if(j==0) {
        float* out=history_slot+(static_cast<std::size_t>(row)*kKVHeads+kv)*kFastFusedFlashStride;
        out[kFastFusedFlashValues+head]=gmax;
        out[kFastFusedFlashValues+H+head]=den;
    }
}

// Un-rotates the merged history numerators, o = o' R_v^T: one CTA per (row,
// KV head, 32-column block), 192 threads = 6 heads x 32 columns.
__global__ void __launch_bounds__(192) l0_history_unrotate_kernel(const float* rotated,
    float* history_slot,int rows,const float* rvt) {
    constexpr int H=kFastFusedFlashHeads;
    __shared__ float numer[H][kHeadDim];
    const int row=blockIdx.x/32,kv=(blockIdx.x/8)%4,block=blockIdx.x%8;
    const int t=threadIdx.x,head=t/32,j=block*32+t%32;
    if(row>=rows) return;
    const float* src=rotated+(static_cast<std::size_t>(row)*kQHeads+kv*H)*kHeadDim;
    for(int i=t;i<H*kHeadDim;i+=blockDim.x) numer[i/kHeadDim][i%kHeadDim]=src[i];
    __syncthreads();
    const float* r=rvt+static_cast<std::size_t>(kv)*kHeadDim*kHeadDim;
    float o=0.0f;
    for(int i=0;i<kHeadDim;++i) o=fmaf(numer[head][i],r[i*kHeadDim+j],o);
    history_slot[(static_cast<std::size_t>(row)*kKVHeads+kv)*kFastFusedFlashStride+head*kHeadDim+j]=o;
}

// L0 OSCAR prefill history: FA2-shaped (64 query rows x 2 query heads per
// CTA, 32-key tiles) over the INT2 history [kSink, history_end) of a prefill
// chunk whose rows all follow it, entirely on s8/u8 MMAs over the raw codes:
//   QK  int8 rotated queries (per 128-dim group, quantized in registers)
//       against the K codes, as in l0_history_mma_kernel. Score columns are
//       permuted so this lane's scores are keys 4t..4t+3 and 16+4t..16+4t+3,
//       the A-fragment keys of the PV MMA.
//   PV  P scaled by the per-key V scale (per 128-dim group), quantized to u8
//       per (row, tile), against the V codes read in place (c 4^j), with the
//       V zero term as a per-row bias; each n-tile's int32 tile sum is folded
//       into the FP32 accumulator.
// Writes the normalized rotated-basis output and (max, denominator).
constexpr int kL0PrefillSplits=4;
constexpr int kL0PrefillThreads=256;
__device__ __forceinline__ void l0_immau(int (&c)[4],const unsigned (&a)[4],unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
__global__ void __launch_bounds__(256,1) l0_history_prefill_kernel(
    const std::uint16_t* q_rot_bits,const float* q_mu,const std::uint8_t* k_codes,
    const std::uint8_t* v_codes,const float* k_meta,const float* v_meta,
    float* out,float* stats,int rows,int history_end){
  // blockIdx.z = head group * kL0PrefillSplits + split over the history tiles.
  using namespace fa2_prefill;
  constexpr float kLog2Scale=0.0625f*1.4426950408889634f;
  constexpr int kWords=l0::kCodeBytes/4;   // 16 code words per key
  const auto* q=reinterpret_cast<const __half*>(q_rot_bits);
  __shared__ __align__(16) unsigned kraw[BN*kWords];
  __shared__ __align__(16) unsigned vraw[BN*kWords];
  __shared__ __align__(16) float4 kmeta[BN];
  __shared__ __align__(16) float4 vmeta[BN];
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
  // Int8 A fragments: k-step pair m reads code word W = 8 (m >> 1) + 2t + (m & 1)
  // of each key; byte i of (w >> 2j) & 0x03030303 is dim 16 W + 4 i + j. This
  // lane holds rows g, g + 8 at the dims of words {2t, 2t+1, 8+2t, 9+2t}.
  unsigned qa[4][2][4];
  float qs[2][2],qsum[2][2],qmu[2];
  {
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
        qa[m][sb][0]=w[0];qa[m][sb][1]=w[1];qa[m][sb][2]=w[2];qa[m][sb][3]=w[3];
      }
  }
  // acc[16 h + n][e]: rows g (e < 2) / g + 8, dims 128 h + 16 (2t + (e & 1)) + n,
  // in units of 4^-(n & 3) (codes read in place); bias[r][h] is the V zero term.
  float acc[32][4];
  #pragma unroll
  for(int n=0;n<32;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.f;
  float run_max[2]={-3.402823466e+38F,-3.402823466e+38F}, den[2]={0.f,0.f},bias[2][2]={{0.f,0.f},{0.f,0.f}};
  const bool warp_live=m_base<active_rows;
  // Raw tile: K codes, V codes (128 x 16 B each), K and V meta (32 x 16 B each).
  uint4 stage[2]={make_uint4(0,0,0,0),make_uint4(0,0,0,0)};
  const auto fetch=[&](int tile_first){
    #pragma unroll
    for(int i=0;i<2;++i) {
      const int c=tid+256*i;
      if(c>=BN*10) break;
      const int part=c<BN*8?c/(BN*4):2+(c-BN*8)/BN;     // 0 K, 1 V, 2 K meta, 3 V meta
      const int idx=c<BN*8?c%(BN*4):(c-BN*8)%BN;
      const int key=tile_first+(part<2?idx/4:idx);
      uint4 value=make_uint4(0,0,0,0);
      if(key<split_end) {
        const std::size_t cell=static_cast<std::size_t>(key)*kKVHeads+kv_head;
        value=part==0?*reinterpret_cast<const uint4*>(k_codes+cell*l0::kCodeBytes+(idx%4)*16):
              part==1?*reinterpret_cast<const uint4*>(v_codes+cell*l0::kCodeBytes+(idx%4)*16):
              part==2?*reinterpret_cast<const uint4*>(k_meta+cell*l0::kMetaFloats):
                      *reinterpret_cast<const uint4*>(v_meta+cell*l0::kMetaFloats);
      }
      stage[i]=value;
    }
  };
  const auto publish=[&]{
    #pragma unroll
    for(int i=0;i<2;++i) {
      const int c=tid+256*i;
      if(c>=BN*10) break;
      if(c<BN*4) reinterpret_cast<uint4*>(kraw)[c]=stage[i];
      else if(c<BN*8) reinterpret_cast<uint4*>(vraw)[c-BN*4]=stage[i];
      else if(c<BN*9) reinterpret_cast<uint4*>(kmeta)[c-BN*8]=stage[i];
      else reinterpret_cast<uint4*>(vmeta)[c-BN*9]=stage[i];
    }
  };
  if(split_begin<split_end) fetch(split_begin);
  for(int first=split_begin;first<split_end;first+=BN){
    __syncthreads();
    publish();
    __syncthreads();
    if(first+BN<split_end) fetch(first+BN);
    if(!warp_live) continue;
    // QK. Score n-tile j column c is key 4 (c >> 1) + 2 (j & 1) + (c & 1) + 16 (j >> 1).
    int si[2][BN/8][4];
    #pragma unroll
    for(int h=0;h<2;++h)
      #pragma unroll
      for(int j=0;j<BN/8;++j) si[h][j][0]=si[h][j][1]=si[h][j][2]=si[h][j][3]=0;
    #pragma unroll
    for(int j=0;j<BN/8;++j) {
      const int key=4*(g>>1)+2*(j&1)+(g&1)+16*(j>>1);
      const uint2* kw=reinterpret_cast<const uint2*>(kraw+key*kWords);
      const uint2 lo=kw[t],hi=kw[4+t];
      #pragma unroll
      for(int m=0;m<4;++m) {
        const unsigned w=m==0?lo.x:m==1?lo.y:m==2?hi.x:hi.y;
        #pragma unroll
        for(int sb=0;sb<2;++sb)
          l0_imma(si[m>>1][j],qa[m][sb],(w>>(4*sb))&0x03030303U,(w>>(4*sb+2))&0x03030303U);
      }
    }
    // This lane's keys: kk = 4t + i (i < 4) and 16 + 4t + (i - 4); s[e][i] row e.
    float s[2][8];
    float tmax[2]={-3.402823466e+38F,-3.402823466e+38F};
    #pragma unroll
    for(int j=0;j<BN/8;++j)
      #pragma unroll
      for(int e=0;e<4;++e){
        const int r=e>>1,i=4*(j>>1)+2*(j&1)+(e&1),key=4*t+(i&3)+16*(i>>2);
        const float4 k=kmeta[key];
        const bool live=(r?l1r:l0r)&&first+key<split_end;
        const float v=qs[r][0]*k.x*static_cast<float>(si[0][j][e])+qs[r][1]*k.z*static_cast<float>(si[1][j][e])-
            qsum[r][0]*k.y*k.x-qsum[r][1]*k.w*k.z+qmu[r];
        s[r][i]=live?v:-3.402823466e+38F; tmax[r]=fmaxf(tmax[r],s[r][i]);
      }
    #pragma unroll
    for(int r=0;r<2;++r){ tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffu,tmax[r],1)); tmax[r]=fmaxf(tmax[r],__shfl_xor_sync(0xffffffffu,tmax[r],2)); }
    float scale[2];
    #pragma unroll
    for(int r=0;r<2;++r){ const float nm=fmaxf(run_max[r],tmax[r]); scale[r]=(nm==run_max[r])?1.f:exp2f(run_max[r]-nm); run_max[r]=nm; }
    // P, P V-scaled per group, and its u8 quantization per (row, group).
    float pv[2][2][8];
    float pmax[2][2]={{0.f,0.f},{0.f,0.f}};
    float rs[2]={0.f,0.f},tile_bias[2][2]={{0.f,0.f},{0.f,0.f}};
    #pragma unroll
    for(int r=0;r<2;++r)
      #pragma unroll
      for(int i=0;i<8;++i) {
        const float x=s[r][i];
        const float p=x==-3.402823466e+38F?0.f:exp2f(x-run_max[r]);
        const float4 vm=vmeta[4*t+(i&3)+16*(i>>2)];
        rs[r]+=p;
        tile_bias[r][0]=fmaf(-p*vm.y,vm.x,tile_bias[r][0]);
        tile_bias[r][1]=fmaf(-p*vm.w,vm.z,tile_bias[r][1]);
        pv[r][0][i]=p*vm.x; pv[r][1][i]=p*vm.z;
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
          unsigned w=0u;
          #pragma unroll
          for(int i=0;i<4;++i) w|=static_cast<unsigned>(__float2uint_rn(pv[r][h][4*half_index+i]*inv))<<(8*i);
          pa[h][2*half_index+r]=w;
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
    // PV. B column g of n-tile 16 h + n is dim 128 h + 16 g + n (word 8 h + g,
    // byte n >> 2, bits 2 (n & 3)); b0 keys 4t..4t+3, b1 keys 16+4t..16+4t+3.
    #pragma unroll
    for(int h=0;h<2;++h) {
      unsigned u[2][4];
      #pragma unroll
      for(int quad=0;quad<2;++quad) {
        const unsigned* base=vraw+(16*quad+4*t)*kWords+8*h+g;
        const unsigned w0=base[0],w1=base[kWords],w2=base[2*kWords],w3=base[3*kWords];
        const unsigned t0=__byte_perm(w0,w1,0x5140),t1=__byte_perm(w2,w3,0x5140);
        const unsigned t2=__byte_perm(w0,w1,0x7362),t3=__byte_perm(w2,w3,0x7362);
        u[quad][0]=__byte_perm(t0,t1,0x5410);u[quad][1]=__byte_perm(t0,t1,0x7632);
        u[quad][2]=__byte_perm(t2,t3,0x5410);u[quad][3]=__byte_perm(t2,t3,0x7632);
      }
      const unsigned a[4]={pa[h][0],pa[h][1],pa[h][2],pa[h][3]};
      #pragma unroll
      for(int n=0;n<16;++n) {
        const unsigned mask=0x03030303U<<(2*(n&3));
        int c[4]={0,0,0,0};
        l0_immau(c,a,u[0][n>>2]&mask,u[1][n>>2]&mask);
        acc[16*h+n][0]=fmaf(qscale[0][h],static_cast<float>(c[0]),acc[16*h+n][0]);
        acc[16*h+n][1]=fmaf(qscale[0][h],static_cast<float>(c[1]),acc[16*h+n][1]);
        acc[16*h+n][2]=fmaf(qscale[1][h],static_cast<float>(c[2]),acc[16*h+n][2]);
        acc[16*h+n][3]=fmaf(qscale[1][h],static_cast<float>(c[3]),acc[16*h+n][3]);
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
            (acc[16*h+n][2*r+c]+bias[r][h])*inv,
            (acc[16*h+n+1][2*r+c]*0.25f+bias[r][h])*inv,
            (acc[16*h+n+2][2*r+c]*0.0625f+bias[r][h])*inv,
            (acc[16*h+n+3][2*r+c]*0.015625f+bias[r][h])*inv);
      }
    if(t==0){
      const std::size_t stat=(static_cast<std::size_t>(query_base+row)*kQHeads+q_head)*2;
      stats[stat]=run_max[r]*0.6931471805599453f; stats[stat+1]=den[r];
    }
  }
}

// Y[v] = X[v] M for the (row, head) vectors of one KV head, v = row * 6 + head
// at [row][kv * 6 + head][256]: the rotations q R_k and o' R_v^T as FP16
// tensor-core GEMMs. CTA = 64 vectors x 64 output columns (blockIdx.y = column
// block, blockIdx.z = KV head), 4 warps of 16 vectors; `mu` (column block 0)
// also forms q . mu.
constexpr int kL0RotVectors=64,kL0RotCols=64,kL0RotThreads=128;
constexpr int kL0RotAStride=kHeadDim+8,kL0RotBStride=kL0RotCols+8;
constexpr std::size_t kL0RotSmem=
    (static_cast<std::size_t>(kL0RotVectors)*kL0RotAStride+kHeadDim*kL0RotBStride)*sizeof(half);
__device__ __forceinline__ void l0_rot_store(__half* y,float a,float b) {
    *reinterpret_cast<__half2*>(y)=__floats2half2_rn(a,b);
}
__device__ __forceinline__ void l0_rot_store(float* y,float a,float b) {
    *reinterpret_cast<float2*>(y)=make_float2(a,b);
}
template<typename Out>
__global__ void __launch_bounds__(kL0RotThreads) l0_rotate_kernel(const __half* x,
    const __half* m_bank,Out* y,int rows,const float* mu,float* q_mu) {
    constexpr int H=kFastFusedFlashHeads;
    extern __shared__ __align__(16) unsigned char l0_rot_smem[];
    half* xs=reinterpret_cast<half*>(l0_rot_smem);
    half* ms=xs+kL0RotVectors*kL0RotAStride;
    const int kv=static_cast<int>(blockIdx.z),cb=static_cast<int>(blockIdx.y);
    const int v0=static_cast<int>(blockIdx.x)*kL0RotVectors,total=rows*H;
    const int tid=static_cast<int>(threadIdx.x),warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
    const auto vector_at=[&](int v){
        return (static_cast<std::size_t>(v/H)*kQHeads+kv*H+v%H)*kHeadDim;
    };
    for(int i=tid;i<kL0RotVectors*(kHeadDim/8);i+=kL0RotThreads) {
        const int vv=i/(kHeadDim/8),c=(i%(kHeadDim/8))*8,v=v0+vv;
        uint4 value=make_uint4(0,0,0,0);
        if(v<total) value=*reinterpret_cast<const uint4*>(x+vector_at(v)+c);
        *reinterpret_cast<uint4*>(xs+vv*kL0RotAStride+c)=value;
    }
    const __half* m=m_bank+static_cast<std::size_t>(kv)*kHeadDim*kHeadDim+cb*kL0RotCols;
    for(int i=tid;i<kHeadDim*(kL0RotCols/8);i+=kL0RotThreads) {
        const int r=i/(kL0RotCols/8),c=(i%(kL0RotCols/8))*8;
        *reinterpret_cast<uint4*>(ms+r*kL0RotBStride+c)=
            *reinterpret_cast<const uint4*>(m+static_cast<std::size_t>(r)*kHeadDim+c);
    }
    __syncthreads();
    float acc[kL0RotCols/8][4];
    #pragma unroll
    for(int n=0;n<kL0RotCols/8;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.0f;
    #pragma unroll 4
    for(int kk=0;kk<kHeadDim;kk+=16) {
        unsigned a[4];
        reg_attn_ldmatrix_x4(a,xs+(warp*16+(lane&15))*kL0RotAStride+kk+(lane>>4)*8);
        #pragma unroll
        for(int pair=0;pair<kL0RotCols/16;++pair) {
            unsigned b[4];
            reg_attn_ldmatrix_x4_trans(b,ms+(kk+(lane&7)+((lane>>3)&1)*8)*kL0RotBStride+
                pair*16+(lane>>4)*8);
            reg_attn_mma(acc[2*pair],a,b[0],b[1]);
            reg_attn_mma(acc[2*pair+1],a,b[2],b[3]);
        }
    }
    #pragma unroll
    for(int r=0;r<2;++r) {
        const int v=v0+warp*16+g+8*r;
        if(v>=total) continue;
        Out* dst=y+vector_at(v)+cb*kL0RotCols+2*t;
        #pragma unroll
        for(int n=0;n<kL0RotCols/8;++n) l0_rot_store(dst+8*n,acc[n][2*r],acc[n][2*r+1]);
    }
    if(mu && cb==0) {
        const float* mk=mu+static_cast<std::size_t>(kv)*kHeadDim;
        for(int i=0;i<16;++i) {
            const int v=v0+warp*16+i;
            float s=0.0f;
            for(int d=lane;d<kHeadDim;d+=32) s=fmaf(__half2float(xs[(warp*16+i)*kL0RotAStride+d]),mk[d],s);
            for(int o=16;o>0;o>>=1) s+=__shfl_xor_sync(0xffffffffU,s,o);
            if(lane==0 && v<total) q_mu[(v/H)*kQHeads+kv*H+v%H]=s;
        }
    }
}

template<typename Out>
void l0_launch_rotate(const void* x,const __half* m_bank,Out* y,int rows,const float* mu,
    float* q_mu,cudaStream_t stream) {
    static const bool configured=[] {
        cuda_check(cudaFuncSetAttribute(l0_rotate_kernel<Out>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(kL0RotSmem)),
            "configure L0 rotation shared memory");
        return true;
    }();
    (void)configured;
    l0_rotate_kernel<Out><<<dim3((rows*kFastFusedFlashHeads+kL0RotVectors-1)/kL0RotVectors,
        kHeadDim/kL0RotCols,kKVHeads),kL0RotThreads,kL0RotSmem,stream>>>(
        reinterpret_cast<const __half*>(x),m_bank,y,rows,mu,q_mu);
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
