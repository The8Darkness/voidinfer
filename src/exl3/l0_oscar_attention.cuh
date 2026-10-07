#pragma once
// L0 OSCAR history attention (see l0_oscar.cuh). Included by
// full_attention_layer.cu after the verifier MMA helpers; shares their
// fragment layout, online softmax and fused-flash slot format.

namespace l0 = ninfer::exl3::l0_oscar;

// One CTA per (history segment, KV head) for up to eight query rows, as
// attention_verify_flash_mma_kernel, over keys [kSink + segment * kKeys, ...)
// below history_end(base). Tiles are decoded from INT2 codes into the rotated
// (and, for K, centered) basis; queries are the rotated q R_k and every score
// gains the row's q . mu. Slots hold rotated-basis value numerators.
template<int kKeys>
__global__ void __launch_bounds__(96) l0_history_mma_kernel(
    const std::uint16_t* q_rot_bits,const float* q_mu,const std::uint8_t* k_codes,
    const std::uint8_t* v_codes,const float* k_meta,const float* v_meta,
    float* workspace,int rows,int position,int capacity,int segments,
    const int* position_device) {
    constexpr int H=kFastFusedFlashHeads;
    constexpr int kVectors=kHeadDim/8;
    const auto* q_rot=reinterpret_cast<const __half*>(q_rot_bits);
    __shared__ __align__(16) half k_s[kVerifyMmaChunk*kVerifyMmaStride];
    __shared__ __align__(16) half v_s[kVerifyMmaChunk*kVerifyMmaStride];
    const int segment=static_cast<int>(blockIdx.x);
    const int kv_head=static_cast<int>(blockIdx.y);
    const int tid=static_cast<int>(threadIdx.x);
    const int warp=tid>>5,lane=tid&31,g=lane>>2,t=lane&3;
    const int base=position_device?*position_device:position;
    const int end=min(l0::history_end(base),capacity);
    const int first=l0::kSink+segment*kKeys;
    if(first>=end) return;
    const int segment_end=min(first+kKeys,end);
    const int pair0=warp*16+g,pair1=pair0+8;
    const int row0=pair0/H,head0=pair0%H,row1=pair1/H,head1=pair1%H;
    const bool live0=row0<rows,live1=row1<rows;
    const float mu0=live0?q_mu[row0*kQHeads+kv_head*H+head0]:0.0f;
    const float mu1=live1?q_mu[row1*kQHeads+kv_head*H+head1]:0.0f;
    const auto* q0=reinterpret_cast<const unsigned*>(q_rot+
        (static_cast<std::size_t>(live0?row0:0)*kQHeads+kv_head*H+head0)*kHeadDim+2*t);
    const auto* q1=reinterpret_cast<const unsigned*>(q_rot+
        (static_cast<std::size_t>(live1?row1:0)*kQHeads+kv_head*H+head1)*kHeadDim+2*t);
    float acc[kHeadDim/8][4];
    #pragma unroll
    for(int n=0;n<kHeadDim/8;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.0f;
    float running_max[2]={-INFINITY,-INFINITY};
    float denominator[2]={0.0f,0.0f};
    for(int chunk_first=first;chunk_first<segment_end;chunk_first+=kVerifyMmaChunk) {
        __syncthreads();
        for(int index=tid;index<kVerifyMmaChunk*kVectors;index+=blockDim.x) {
            const int key_offset=index/kVectors;
            const int dim=(index%kVectors)*8;
            const int key=chunk_first+key_offset;
            half kv8[8],vv8[8];
            if(key<segment_end) {
                const std::size_t cell=static_cast<std::size_t>(key)*kKVHeads+kv_head;
                const int group=dim>>7;
                const float ks=k_meta[cell*l0::kMetaFloats+2*group];
                const float kz=k_meta[cell*l0::kMetaFloats+2*group+1];
                const float vs=v_meta[cell*l0::kMetaFloats+2*group];
                const float vz=v_meta[cell*l0::kMetaFloats+2*group+1];
                const unsigned kb=*reinterpret_cast<const std::uint16_t*>(k_codes+cell*l0::kCodeBytes+(dim>>2));
                const unsigned vb=*reinterpret_cast<const std::uint16_t*>(v_codes+cell*l0::kCodeBytes+(dim>>2));
                #pragma unroll
                for(int e=0;e<8;++e) {
                    kv8[e]=__float2half_rn((static_cast<float>((kb>>(2*e))&3u)-kz)*ks);
                    vv8[e]=__float2half_rn((static_cast<float>((vb>>(2*e))&3u)-vz)*vs);
                }
            } else {
                #pragma unroll
                for(int e=0;e<8;++e) kv8[e]=vv8[e]=__float2half_rn(0.0f);
            }
            *reinterpret_cast<uint4*>(k_s+key_offset*kVerifyMmaStride+dim)=*reinterpret_cast<const uint4*>(kv8);
            *reinterpret_cast<uint4*>(v_s+key_offset*kVerifyMmaStride+dim)=*reinterpret_cast<const uint4*>(vv8);
        }
        __syncthreads();
        float s[4][4];
        #pragma unroll
        for(int j=0;j<4;++j) s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.0f;
        #pragma unroll
        for(int kk=0;kk<kHeadDim;kk+=16) {
            const unsigned a[4]={live0?q0[kk/2]:0u,live1?q1[kk/2]:0u,
                                 live0?q0[kk/2+4]:0u,live1?q1[kk/2+4]:0u};
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
                unsigned b[4];
                reg_attn_ldmatrix_x4(b,k_s+(pair*16+(lane&7)+((lane>>4)<<3))*
                    kVerifyMmaStride+kk+((lane>>3)&1)*8);
                reg_attn_mma(s[2*pair],a,b[0],b[1]);
                reg_attn_mma(s[2*pair+1],a,b[2],b[3]);
            }
        }
        float tile_max[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(int j=0;j<4;++j)
            #pragma unroll
            for(int e=0;e<4;++e) {
                const int key=chunk_first+8*j+2*t+(e&1);
                const bool valid=key<segment_end&&(e<2?live0:live1);
                s[j][e]=valid?(s[j][e]+(e<2?mu0:mu1))*0.0625f:-INFINITY;
                tile_max[e>>1]=fmaxf(tile_max[e>>1],s[j][e]);
            }
        #pragma unroll
        for(int i=0;i<2;++i) {
            tile_max[i]=fmaxf(tile_max[i],__shfl_xor_sync(0xffffffffU,tile_max[i],1));
            tile_max[i]=fmaxf(tile_max[i],__shfl_xor_sync(0xffffffffU,tile_max[i],2));
        }
        float scale[2];
        #pragma unroll
        for(int i=0;i<2;++i) {
            const float next=fmaxf(running_max[i],tile_max[i]);
            scale[i]=next==-INFINITY?1.0f:expf(running_max[i]-next);
            running_max[i]=next;
        }
        half p[4][4];
        float sums[2]={0.0f,0.0f};
        #pragma unroll
        for(int j=0;j<4;++j)
            #pragma unroll
            for(int e=0;e<4;++e) {
                const float m=running_max[e>>1];
                p[j][e]=__float2half_rn(s[j][e]==-INFINITY?0.0f:expf(s[j][e]-m));
                sums[e>>1]+=__half2float(p[j][e]);
            }
        #pragma unroll
        for(int i=0;i<2;++i) {
            sums[i]+=__shfl_xor_sync(0xffffffffU,sums[i],1);
            sums[i]+=__shfl_xor_sync(0xffffffffU,sums[i],2);
            denominator[i]=denominator[i]*scale[i]+sums[i];
        }
        #pragma unroll
        for(int n=0;n<kHeadDim/8;++n) {
            acc[n][0]*=scale[0];acc[n][1]*=scale[0];
            acc[n][2]*=scale[1];acc[n][3]*=scale[1];
        }
        #pragma unroll
        for(int kk=0;kk<2;++kk) {
            const unsigned a[4]={reg_attn_pack(p[2*kk][0],p[2*kk][1]),
                                 reg_attn_pack(p[2*kk][2],p[2*kk][3]),
                                 reg_attn_pack(p[2*kk+1][0],p[2*kk+1][1]),
                                 reg_attn_pack(p[2*kk+1][2],p[2*kk+1][3])};
            #pragma unroll
            for(int pair=0;pair<kHeadDim/16;++pair) {
                unsigned b[4];
                reg_attn_ldmatrix_x4_trans(b,v_s+(16*kk+(lane&7)+((lane>>3)&1)*8)*
                    kVerifyMmaStride+pair*16+(lane>>4)*8);
                reg_attn_mma(acc[2*pair],a,b[0],b[1]);
                reg_attn_mma(acc[2*pair+1],a,b[2],b[3]);
            }
        }
    }
    #pragma unroll
    for(int i=0;i<2;++i) {
        if(!(i==0?live0:live1)) continue;
        const int row=i==0?row0:row1,head=i==0?head0:head1;
        float* slot=workspace+
            ((static_cast<std::size_t>(row)*kKVHeads+kv_head)*segments+segment)*kFastFusedFlashStride;
        #pragma unroll
        for(int n=0;n<kHeadDim/8;++n) {
            const int dim=8*n+2*t;
            slot[head*kHeadDim+dim]=acc[n][2*i];
            slot[head*kHeadDim+dim+1]=acc[n][2*i+1];
        }
        if(t==0) {
            slot[kFastFusedFlashValues+head]=running_max[i];
            slot[kFastFusedFlashValues+H+head]=denominator[i];
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
    const int live=end>l0::kSink?min(segments,(end-l0::kSink+keys-1)/keys):0;
    const float* slots=workspace+(static_cast<std::size_t>(row)*kKVHeads+kv)*segments*kFastFusedFlashStride;
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
