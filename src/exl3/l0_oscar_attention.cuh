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

// L0 OSCAR prefill history: FA2-shaped (64 query rows x 2 query heads per
// CTA, 32-key tiles) over the INT2 history [kSink, history_end) of a prefill
// chunk whose rows all follow it. Rotated queries with the q . mu constant;
// writes the normalized rotated-basis output and (max, denominator).
__global__ void __launch_bounds__(256,1) l0_history_prefill_kernel(
    const std::uint16_t* q_rot_bits,const float* q_mu,const std::uint8_t* k_codes,
    const std::uint8_t* v_codes,const float* k_meta,const float* v_meta,
    float* out,float* stats,int rows,int history_end){
  using namespace fa2_prefill;
  const auto* q=reinterpret_cast<const std::uint16_t*>(q_rot_bits);
  __shared__ __align__(16) half ks[BN*kStride];
  __shared__ __align__(16) half vs[BN*kStride];
  const int query_base=blockIdx.x*BM;
  const int kv_head=blockIdx.y;
  const int head_group=blockIdx.z;
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  const int head_local=warp>>2, m_base=(warp&3)*16;
  const int g=lane>>2, t=lane&3;
  const int row0=m_base+g, row1=row0+8;
  const int q_head=kv_head*(kQHeads/kKVHeads)+head_group*2+head_local;
  const int active_rows=min(BM,rows-query_base);
  if(active_rows<=0) return;
  const bool l0r=row0<active_rows, l1r=row1<active_rows;
  const float mu0=l0r?q_mu[(query_base+row0)*kQHeads+q_head]:0.f;
  const float mu1=l1r?q_mu[(query_base+row1)*kQHeads+q_head]:0.f;
  unsigned qa[kHeadDim/16][4];
  {
    const unsigned* q0=reinterpret_cast<const unsigned*>(q+(static_cast<std::size_t>(query_base+(l0r?row0:0))*kQHeads+q_head)*kHeadDim+2*t);
    const unsigned* q1=reinterpret_cast<const unsigned*>(q+(static_cast<std::size_t>(query_base+(l1r?row1:0))*kQHeads+q_head)*kHeadDim+2*t);
    #pragma unroll
    for(int kk=0;kk<kHeadDim/16;++kk){
      qa[kk][0]=l0r?q0[kk*8]:0u; qa[kk][1]=l1r?q1[kk*8]:0u;
      qa[kk][2]=l0r?q0[kk*8+4]:0u; qa[kk][3]=l1r?q1[kk*8+4]:0u;
    }
  }
  float acc[kHeadDim/8][4];
  #pragma unroll
  for(int n=0;n<kHeadDim/8;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.f;
  float run_max[2]={-3.402823466e+38F,-3.402823466e+38F}, den[2]={0.f,0.f};
  const bool warp_live=m_base<active_rows;
  for(int first=l0::kSink;first<history_end;first+=BN){
    __syncthreads();
    #pragma unroll
    for(int i=0;i<(BN*kHeadDim/8)/256;++i){
      const int idx=tid+i*256; const int key_off=idx/(kHeadDim/8); const int dim=(idx%(kHeadDim/8))*8;
      const int key=first+key_off;
      const std::size_t cell=static_cast<std::size_t>(key)*kKVHeads+kv_head;
      const int group=dim>>7;
      const float kscale=k_meta[cell*l0::kMetaFloats+2*group], kzero=k_meta[cell*l0::kMetaFloats+2*group+1];
      const float vscale=v_meta[cell*l0::kMetaFloats+2*group], vzero=v_meta[cell*l0::kMetaFloats+2*group+1];
      const unsigned kb=*reinterpret_cast<const std::uint16_t*>(k_codes+cell*l0::kCodeBytes+(dim>>2));
      const unsigned vb=*reinterpret_cast<const std::uint16_t*>(v_codes+cell*l0::kCodeBytes+(dim>>2));
      half k8[8],v8[8];
      #pragma unroll
      for(int e=0;e<8;++e){
        k8[e]=__float2half_rn((static_cast<float>((kb>>(2*e))&3u)-kzero)*kscale);
        v8[e]=__float2half_rn((static_cast<float>((vb>>(2*e))&3u)-vzero)*vscale);
      }
      *reinterpret_cast<uint4*>(ks+key_off*kStride+dim)=*reinterpret_cast<const uint4*>(k8);
      *reinterpret_cast<uint4*>(vs+key_off*kStride+dim)=*reinterpret_cast<const uint4*>(v8);
    }
    __syncthreads();
    if(!warp_live) continue;
    float s[BN/8][4];
    #pragma unroll
    for(int j=0;j<BN/8;++j) s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.f;
    #pragma unroll
    for(int kk=0;kk<kHeadDim/16;++kk){
      #pragma unroll
      for(int pair=0;pair<BN/16;++pair){
        unsigned b[4];
        reg_attn_ldmatrix_x4(b,ks+(pair*16+(lane&7)+((lane>>4)<<3))*kStride+kk*16+((lane>>3)&1)*8);
        reg_attn_mma(s[2*pair],qa[kk],b[0],b[1]);
        reg_attn_mma(s[2*pair+1],qa[kk],b[2],b[3]);
      }
    }
    float tmax[2]={-3.402823466e+38F,-3.402823466e+38F};
    #pragma unroll
    for(int j=0;j<BN/8;++j)
      #pragma unroll
      for(int e=0;e<4;++e){
        const bool live=e<2?l0r:l1r;
        const float sc=live?(s[j][e]+(e<2?mu0:mu1))*0.0625f:-3.402823466e+38F;
        s[j][e]=sc; tmax[e>>1]=fmaxf(tmax[e>>1],sc);
      }
    #pragma unroll
    for(int i=0;i<2;++i){ tmax[i]=fmaxf(tmax[i],__shfl_xor_sync(0xffffffffu,tmax[i],1)); tmax[i]=fmaxf(tmax[i],__shfl_xor_sync(0xffffffffu,tmax[i],2)); }
    float scale[2];
    #pragma unroll
    for(int i=0;i<2;++i){ const float nm=fmaxf(run_max[i],tmax[i]); scale[i]=(nm==run_max[i])?1.f:exp2f((run_max[i]-nm)*1.4426950408889634f); run_max[i]=nm; }
    unsigned pp[BN/8][2]; float rs[2]={0.f,0.f};
    #pragma unroll
    for(int j=0;j<BN/8;++j){
      half p[4];
      #pragma unroll
      for(int e=0;e<4;++e){
        const float x=s[j][e];
        p[e]=__float2half(x==-3.402823466e+38F?0.f:exp2f((x-run_max[e>>1])*1.4426950408889634f));
        rs[e>>1]+=__half2float(p[e]);
      }
      pp[j][0]=reg_attn_pack(p[0],p[1]); pp[j][1]=reg_attn_pack(p[2],p[3]);
    }
    #pragma unroll
    for(int i=0;i<2;++i){ rs[i]+=__shfl_xor_sync(0xffffffffu,rs[i],1); rs[i]+=__shfl_xor_sync(0xffffffffu,rs[i],2); den[i]=den[i]*scale[i]+rs[i]; }
    #pragma unroll
    for(int n=0;n<kHeadDim/8;++n){ acc[n][0]*=scale[0]; acc[n][1]*=scale[0]; acc[n][2]*=scale[1]; acc[n][3]*=scale[1]; }
    #pragma unroll
    for(int kk=0;kk<BN/16;++kk){
      const unsigned a[4]={pp[2*kk][0],pp[2*kk][1],pp[2*kk+1][0],pp[2*kk+1][1]};
      #pragma unroll
      for(int pair=0;pair<kHeadDim/16;++pair){
        unsigned b[4];
        reg_attn_ldmatrix_x4_trans(b,vs+(16*kk+(lane&7)+((lane>>3)&1)*8)*kStride+pair*16+(lane>>4)*8);
        reg_attn_mma(acc[2*pair],a,b[0],b[1]); reg_attn_mma(acc[2*pair+1],a,b[2],b[3]);
      }
    }
  }
  #pragma unroll
  for(int i=0;i<2;++i){
    const int row=i==0?row0:row1;
    if(row>=active_rows) continue;
    const std::size_t row_base=(static_cast<std::size_t>(query_base+row)*kQHeads+q_head)*kHeadDim;
    const float inv=den[i]>0.f?1.f/den[i]:0.f;
    #pragma unroll
    for(int n=0;n<kHeadDim/8;++n){
      const int dim=8*n+2*t;
      *reinterpret_cast<float2*>(out+row_base+dim)=make_float2(acc[n][2*i]*inv,acc[n][2*i+1]*inv);
    }
    if(t==0){
      const std::size_t stat=(static_cast<std::size_t>(query_base+row)*kQHeads+q_head)*2;
      stats[stat]=run_max[i]; stats[stat+1]=den[i];
    }
  }
}

// Un-rotates normalized prefill history outputs o = o' R_v^T into split slot
// `slot` of the FA2 split buffers: one CTA per (row, KV head, 32-column block).
__global__ void __launch_bounds__(192) l0_history_unrotate_rows_kernel(const float* rotated,
    const float* rotated_stats,float* split_output,float* split_stats,int rows,int slot,
    const float* rvt) {
    constexpr int H=kFastFusedFlashHeads;
    __shared__ float numer[H][kHeadDim];
    const int row=blockIdx.x/32,kv=(blockIdx.x/8)%4,block=blockIdx.x%8;
    const int t=threadIdx.x,head=t/32,j=block*32+t%32;
    if(row>=rows) return;
    const std::size_t at=(static_cast<std::size_t>(row)*kQHeads+kv*H)*kHeadDim;
    for(int i=t;i<H*kHeadDim;i+=blockDim.x) numer[i/kHeadDim][i%kHeadDim]=rotated[at+i];
    __syncthreads();
    const float* r=rvt+static_cast<std::size_t>(kv)*kHeadDim*kHeadDim;
    float o=0.0f;
    for(int i=0;i<kHeadDim;++i) o=fmaf(numer[head][i],r[i*kHeadDim+j],o);
    const std::size_t elements=static_cast<std::size_t>(rows)*kQHeads*kHeadDim;
    split_output[static_cast<std::size_t>(slot)*elements+at+head*kHeadDim+j]=o;
    if(block==0 && t%32==0) {
        const std::size_t stat=(static_cast<std::size_t>(row)*kQHeads+kv*H+head)*2;
        const std::size_t dst=(static_cast<std::size_t>(slot)*rows*kQHeads+row*kQHeads+kv*H+head)*2;
        split_stats[dst]=rotated_stats[stat];
        split_stats[dst+1]=rotated_stats[stat+1];
    }
}
