#include "exl3/gaming_operator_fixture.h"
#include "exl3/gdn_layer.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <mma.h>
#include <vector>
using namespace ninfer::exl3;
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void require(bool x,const char* s){if(!x)throw std::runtime_error(s);}
template<class T> struct Buffer {
    T* p=nullptr;std::size_t n;
    explicit Buffer(std::size_t size):n(size){ck(cudaMalloc(reinterpret_cast<void**>(&p),(n+16)*sizeof(T)));ck(cudaMemset(p,0xa5,(n+16)*sizeof(T)));}
    ~Buffer(){cudaFree(p);}
    Buffer(const Buffer&)=delete;
    void set(const std::vector<T>& v){require(v.size()==n,"upload extent");ck(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get()const{std::vector<T> v(n);ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
    void guard()const{std::vector<unsigned char> b(16*sizeof(T));ck(cudaMemcpy(b.data(),p+n,b.size(),cudaMemcpyDeviceToHost));require(std::all_of(b.begin(),b.end(),[](auto x){return x==0xa5;}),"tail guard overwritten");}
};
using H=Buffer<std::uint16_t>;
static std::vector<std::uint16_t> bits(std::size_t n,bool bf=false){
    std::vector<std::uint16_t> v(n);
    const std::uint16_t base=bf?0x3d00:0x2800;
    for(std::size_t i=0;i<n;++i)v[i]=base+static_cast<std::uint16_t>((i*17)%128)+(i%3==0?0x8000:0);
    return v;
}
static double half_value(std::uint16_t v){
    const int e=(v>>10)&31,m=v&1023;
    const double x=e?std::ldexp(double(1024+m),e-25):std::ldexp(double(m),-24);
    return (v&0x8000)?-x:x;
}
template<class T> static void equal(const Buffer<T>& a,const Buffer<T>& b,const char* label){
    const auto av=a.get(),bv=b.get();require(av.size()==bv.size() && std::memcmp(av.data(),bv.data(),av.size()*sizeof(T))==0,label);a.guard();b.guard();
}
// Capture is a separate execution of the same borrowed, stable allocations.
template<class F> static void captured(F f){
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));f(stream);
    ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    for(int i=0;i<2;++i){ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));}
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
}
static void recurrence(int rows){
    const std::size_t qn=rows*2048,vn=rows*6144,cn=rows*48,sn=48*128*128;
    H q(qn),k(qn),v(vn),oa(vn),ob(vn);q.set(bits(qn,true));k.set(bits(qn,true));v.set(bits(vn,true));
    Buffer<float> g(cn),beta(cn),sa(sn),sb(sn),nq(qn),nk(qn),alpha(cn);
    g.set(std::vector<float>(cn,-0.5f));beta.set(std::vector<float>(cn,0.5f));
    std::vector<float> initial(sn);for(std::size_t i=0;i<sn;++i)initial[i]=float(int(i%17)-8)/256;
    sa.set(initial);sb.set(initial);
    Exl3GdnStageFusionFixtureView view{q.p,k.p,v.p,g.p,beta.p,sa.p,sb.p,oa.p,ob.p,nq.p,nk.p,alpha.p,
        qn*2,vn*2,cn*4,sn*4,qn*4,rows};
    exl3_gdn_stage_fusion_fixture(view,nullptr,true);
    equal(sa,sb,"GOPT-001 FP32 recurrence");equal(oa,ob,"GOPT-001 BF16 rows");
    // Both paths start from the same evolved state on every replay.
    captured([&](cudaStream_t stream){exl3_gdn_stage_fusion_fixture(view,stream,true);});
    equal(sa,sb,"GOPT-001 captured recurrence");equal(oa,ob,"GOPT-001 captured outputs");
    // Independent closed-form oracle: Q=K=0,g=0 => state identity and zero output.
    q.set(std::vector<std::uint16_t>(qn,0));k.set(std::vector<std::uint16_t>(qn,0));g.set(std::vector<float>(cn,0));
    sa.set(initial);sb.set(initial);exl3_gdn_stage_fusion_fixture(view,nullptr,true);
    require(sa.get()==initial&&sb.get()==initial,"GOPT-001 analytic state oracle");
    require(oa.get()==std::vector<std::uint16_t>(vn,0)&&ob.get()==oa.get(),"GOPT-001 analytic output oracle");
}
static void staging(int rows,int which){
    const int nq=rows*10240,n=rows*6144;
    H input(nq),weight(10240*4),sa(10240*4),sb(10240*4),ca(nq),cb(nq),
      qa(rows*2048),qb(rows*2048),ka(rows*2048),kb(rows*2048),va(n),vb(n),pa(nq),pb(nq),ta(10240*3),tb(10240*3);
    input.set(bits(nq));weight.set(bits(weight.n,true));sa.set(bits(sa.n,true));sb.set(bits(sb.n,true));
    gopt_gdn_conv_fixture(false,input.p,weight.p,sa.p,ca.p,qa.p,ka.p,va.p,pa.p,ta.p,rows);
    gopt_gdn_conv_fixture(true,input.p,weight.p,sb.p,cb.p,qb.p,kb.p,vb.p,pb.p,tb.p,rows);
    if(which==2) {
    equal(sa,sb,"GOPT-002 physical state");equal(ta,tb,"GOPT-002 trace");equal(ca,cb,"GOPT-002 retained input");
    equal(qa,qb,"GOPT-002 Q");equal(ka,kb,"GOPT-002 K");equal(va,vb,"GOPT-002 V");equal(pa,pb,"GOPT-002 packed trace");
    const auto state=sb.get(),trace=tb.get();
    for(int c=0;c<10240;++c)for(int j=0;j<3;++j)require(trace[c*3+j]==state[c*4+j],"GOPT-002 trace oracle");
    return;
    }
    H norm(n),trace0(n),trace1(n),projection0(n),projection1(n);norm.set(bits(n,true));
    gopt_gdn_pack_fixture(false,va.p,norm.p,trace0.p,projection0.p,rows);
    gopt_gdn_pack_fixture(true,va.p,norm.p,trace1.p,projection1.p,rows);
    equal(trace0,trace1,"GOPT-003 trace");equal(projection0,projection1,"GOPT-003 cast");
    require(trace1.get()==va.get(),"GOPT-003 trace identity");
}
static void norms(int rows,int which){
    const int n=rows*5120;
    H a(n),b(n),w(5120),ra(n),rb(n),na(n),nb(n);a.set(bits(n));b.set(bits(n));w.set(bits(5120));
    if(which==4) {
    exl3_gdn_residual_norm(a.p,b.p,w.p,ra.p,na.p,rows,false,nullptr);
    exl3_gdn_residual_norm(a.p,b.p,w.p,rb.p,nb.p,rows,true,nullptr);
    equal(ra,rb,"GOPT-004 FP16 residual");equal(na,nb,"GOPT-004 norm");
    return;
    }
    a.set(bits(n,true));b.set(bits(n,true));
    gopt_draft_norm_fixture(false,a.p,b.p,w.p,ra.p,na.p,rows);
    captured([&](cudaStream_t stream){gopt_draft_norm_fixture(true,a.p,b.p,w.p,rb.p,nb.p,rows,stream);});
    equal(ra,rb,"GOPT-008 BF16 residual");equal(na,nb,"GOPT-008 norm");
    // Independent zero-input oracle, including the normalization epsilon.
    a.set(std::vector<std::uint16_t>(n,0));b.set(std::vector<std::uint16_t>(n,0));
    gopt_draft_norm_fixture(true,a.p,b.p,w.p,rb.p,nb.p,rows);
    const auto zero_norm = nb.get();
    require(std::all_of(zero_norm.begin(),zero_norm.end(),[](std::uint16_t value){
        return (value & 0x7fffU)==0;
    }),"GOPT-008 zero oracle");
}
static void rotary(int rows,int heads,int which){
    const int n=rows*heads*128;
    H input(n),w(128),na(n),nb(n),oa(n),ob(n);input.set(bits(n));w.set(bits(128));
    Buffer<int> positions(rows);std::vector<int> pos(rows);
    for(int i=0;i<rows;++i)pos[i]=i%3==0?0:2047+i*4096;positions.set(pos);
    if(which==5) {
    gopt_draft_rope_fixture(false,input.p,oa.p,positions.p,rows,heads);
    gopt_draft_rope_fixture(true,input.p,ob.p,positions.p,rows,heads);equal(oa,ob,"GOPT-005 RoPE pair");
    return;
    }
    gopt_draft_head_fixture(false,false,input.p,w.p,na.p,oa.p,positions.p,rows,heads);
    for(bool pair:{false,true}) {
        captured([&](cudaStream_t stream){gopt_draft_head_fixture(true,pair,input.p,w.p,nb.p,ob.p,positions.p,rows,heads,stream);});
        equal(na,nb,"GOPT-006 represented norm");equal(oa,ob,"GOPT-006 rotated output");
    }
}
static void ring_norm(int rows,int first){
    constexpr int heads=8,dim=128,slots=2048;
    const int n=rows*heads*dim;
    H k(n),v(n),weight(dim),norm(n),rotated(n),ka(slots*heads*dim),kb(slots*heads*dim),
      va(slots*heads*dim),vb(slots*heads*dim);
    k.set(bits(n));v.set(bits(n,true));weight.set(bits(dim));
    Buffer<int> positions(rows);std::vector<int> pos(rows);
    for(int i=0;i<rows;++i)pos[i]=first+i;
    positions.set(pos);
    auto launch=[&](bool fused,cudaStream_t stream){
        gopt_draft_ring_norm_fixture(fused,k.p,v.p,weight.p,positions.p,
            norm.p,rotated.p,fused?kb.p:ka.p,fused?vb.p:va.p,rows,first,stream);
    };
    launch(false,nullptr);launch(true,nullptr);
    equal(ka,kb,"GOPT-012 represented ring K");
    equal(va,vb,"GOPT-012 represented ring V");
    const auto ring_v=vb.get(),input_v=v.get();
    for(int r=0;r<rows;++r)for(int e=0;e<heads*dim;++e)
        require(ring_v[((first+r)&(slots-1))*heads*dim+e]==input_v[r*heads*dim+e],
            "GOPT-012 independent V slot oracle");
    if(rows<=8){
        captured([&](cudaStream_t stream){launch(true,stream);});
        equal(ka,kb,"GOPT-012 graph ring K");
        equal(va,vb,"GOPT-012 graph ring V");
    }
}
template<bool RowMajor>
__global__ static void pv_layout_probe(const std::uint16_t* probabilities,
    const std::uint16_t* values,float* output,int key_base,int value_base) {
    __shared__ half p[16][32];
    __shared__ half v[32][256];
    for(int i=threadIdx.x;i<16*32;i+=blockDim.x)
        p[i/32][i%32]=__ushort_as_half(probabilities[i]);
    for(int i=threadIdx.x;i<32*256;i+=blockDim.x)
        v[i/256][i%256]=__ushort_as_half(values[i]);
    __syncthreads();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,
        nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> c;
    nvcuda::wmma::fill_fragment(c,0.0f);
    nvcuda::wmma::load_matrix_sync(a,&p[0][key_base],32);
    if constexpr(RowMajor) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,
            nvcuda::wmma::row_major> b;
        nvcuda::wmma::load_matrix_sync(b,&v[key_base][value_base],256);
        nvcuda::wmma::mma_sync(c,a,b,c);
    } else {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,
            nvcuda::wmma::col_major> b;
        nvcuda::wmma::load_matrix_sync(b,&v[key_base][value_base],256);
        nvcuda::wmma::mma_sync(c,a,b,c);
    }
    nvcuda::wmma::store_matrix_sync(output,c,16,nvcuda::wmma::mem_row_major);
}
__global__ static void qk_layout_probe(const std::uint16_t* queries,
    const std::uint16_t* keys,float* output,int key_base) {
    __shared__ half q[16][256],k[32][256];
    for(int i=threadIdx.x;i<16*256;i+=blockDim.x)
        q[i/256][i%256]=__ushort_as_half(queries[i]);
    for(int i=threadIdx.x;i<32*256;i+=blockDim.x)
        k[i/256][i%256]=__ushort_as_half(keys[i]);
    __syncthreads();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,
        nvcuda::wmma::row_major> a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,
        nvcuda::wmma::col_major> b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> c;
    nvcuda::wmma::fill_fragment(c,0.0f);
    for(int d=0;d<256;d+=16){
        nvcuda::wmma::load_matrix_sync(a,&q[0][d],256);
        nvcuda::wmma::load_matrix_sync(b,&k[key_base][d],256);
        nvcuda::wmma::mma_sync(c,a,b,c);
    }
    nvcuda::wmma::store_matrix_sync(output,c,16,nvcuda::wmma::mem_row_major);
}
static void qk_layout(int key_base){
    std::vector<std::uint16_t> qh(16*256),kh(32*256);
    for(int r=0;r<16;++r)for(int d=0;d<256;++d)
        qh[r*256+d]=__half_as_ushort(__float2half_rn(float((r*3+d*7)%17-8)/16.0f));
    for(int k=0;k<32;++k)for(int d=0;d<256;++d)
        kh[k*256+d]=__half_as_ushort(__float2half_rn(float((k*11+d*5)%23-11)/16.0f));
    H q(qh.size()),k(kh.size());q.set(qh);k.set(kh);
    Buffer<float> result(256);
    qk_layout_probe<<<1,32>>>(q.p,k.p,result.p,key_base);ck(cudaGetLastError());
    const auto actual=result.get();result.guard();
    for(int r=0;r<16;++r)for(int col=0;col<16;++col){
        double oracle=0.0;
        for(int d=0;d<256;++d)
            oracle+=half_value(qh[r*256+d])*half_value(kh[(key_base+col)*256+d]);
        require(std::abs(actual[r*16+col]-oracle)<0.0001,
            "WMMA QK intentional col-major transpose differs from FP32 oracle");
    }
}
static void pv_layout(int key_base,int value_base,int active_keys){
    std::vector<std::uint16_t> ph(16*32),vh(32*256);
    for(int r=0;r<16;++r)for(int k=0;k<32;++k)
        ph[r*32+k]=__half_as_ushort(__float2half_rn(
            k<active_keys?float(((r+2)*(k+3))%13)/16.0f:0.0f));
    for(int k=0;k<32;++k)for(int d=0;d<256;++d)
        vh[k*256+d]=__half_as_ushort(__float2half_rn(
            k<active_keys?float((k*7+d*3)%19)/32.0f:0.0f));
    H p(ph.size()),v(vh.size());p.set(ph);v.set(vh);
    Buffer<float> wrong(256),correct(256);
    pv_layout_probe<false><<<1,32>>>(p.p,v.p,wrong.p,key_base,value_base);
    pv_layout_probe<true><<<1,32>>>(p.p,v.p,correct.p,key_base,value_base);
    ck(cudaGetLastError());
    const auto bad=wrong.get(),good=correct.get();
    int mismatches=0;
    for(int r=0;r<16;++r)for(int d=0;d<16;++d){
        float oracle=0.0f;
        for(int k=0;k<16;++k)
            oracle+=float(half_value(ph[r*32+key_base+k])*
                half_value(vh[(key_base+k)*256+value_base+d]));
        const int index=r*16+d;
        require(std::abs(good[index]-oracle)<0.0001f,
            "WMMA P×V row-major layout differs from independent FP32 product");
        mismatches+=std::abs(bad[index]-oracle)>0.01f;
    }
    require(mismatches>64,"WMMA P×V col-major probe did not expose transpose");
    wrong.guard();correct.guard();
}
static void wmma_attention_reference(bool rows32,int rows,int position,
    int query_offset,bool device_position,bool zero_scores,bool padded=false,
    bool split2=false,bool split4=false,bool rows64=false,
    bool register_owned=false,bool shared_heads=false,bool keys64=false){
    constexpr int qheads=24,kvheads=4,dim=256,capacity=80;
    const int base=position+query_offset;
    require(base+rows<=capacity,"WMMA reference capacity");
    std::vector<std::uint16_t> qh(rows*qheads*dim),kh(capacity*kvheads*dim),
        vh(capacity*kvheads*dim);
    for(int r=0;r<rows;++r)for(int h=0;h<qheads;++h)for(int d=0;d<dim;++d)
        qh[(r*qheads+h)*dim+d]=__half_as_ushort(__float2half_rn(
            float((r*3+h*5+d*7)%17-8)/16.0f));
    for(int k=0;k<capacity;++k)for(int h=0;h<kvheads;++h)for(int d=0;d<dim;++d){
        const int index=(k*kvheads+h)*dim+d;
        kh[index]=__half_as_ushort(__float2half_rn(
            float((k*11+h*3+d*5)%23-11)/16.0f));
        vh[index]=__half_as_ushort(__float2half_rn(
            float((k*7+h*11+d*3)%29-14)/32.0f));
    }
    if(zero_scores){std::fill(qh.begin(),qh.end(),0);std::fill(kh.begin(),kh.end(),0);}
    H q(qh.size()),k(kh.size()),v(vh.size()),out(qh.size());
    Buffer<float> partial(static_cast<std::size_t>(4)*qh.size());
    Buffer<float> stats(static_cast<std::size_t>(4)*rows*qheads*2);
    q.set(qh);k.set(kh);v.set(vh);
    Buffer<int> live(1);live.set(std::vector<int>{position});
    if(rows64)
        fast_wmma64_attention_fixture(q.p,k.p,v.p,out.p,
            partial.p,stats.p,rows,device_position?0:position,capacity,
            device_position?live.p:nullptr,query_offset,nullptr,padded,split2,
            register_owned,shared_heads,keys64);
    else if(split4)
        fast_wmma_split4_attention_fixture(q.p,k.p,v.p,out.p,
            partial.p,stats.p,rows,device_position?0:position,capacity,
            device_position?live.p:nullptr,query_offset,nullptr,padded);
    else if(split2)
        fast_wmma_split2_attention_fixture(q.p,k.p,v.p,out.p,
            partial.p,stats.p,rows,device_position?0:position,capacity,
            device_position?live.p:nullptr,query_offset,nullptr,padded);
    else
        fast_wmma_attention_fixture(rows32,q.p,k.p,v.p,out.p,rows,
            device_position?0:position,capacity,device_position?live.p:nullptr,
            query_offset,nullptr,padded);
    const auto actual=out.get();out.guard();partial.guard();stats.guard();
    double max_error=0.0;int failures=0;
    for(int r:{0,rows/2,rows-1})for(int h:{0,5,6,17,23}){
        const int kv=h/6;
        const int count=base+r+1;
        std::vector<double> scores(count);
        double maximum=-1.0e30;
        for(int key=0;key<count;++key){
            double dot=0.0;
            for(int d=0;d<dim;++d)
                dot+=half_value(qh[(r*qheads+h)*dim+d])*
                    half_value(kh[(key*kvheads+kv)*dim+d]);
            scores[key]=dot*0.0625;
            maximum=std::max(maximum,scores[key]);
        }
        double denominator=0.0;
        for(double& score:scores){score=std::exp(score-maximum);denominator+=score;}
        for(int d:{0,17,31,32,63,96,127,128,191,224,255}){
            double numerator=0.0;
            for(int key=0;key<count;++key)
                numerator+=scores[key]*half_value(vh[(key*kvheads+kv)*dim+d]);
            const double expected=numerator/denominator;
            const double got=half_value(actual[(r*qheads+h)*dim+d]);
            max_error=std::max(max_error,std::abs(got-expected));
            if(!std::isfinite(got) || std::abs(got-expected)>=0.02){
                if(failures<8)std::cerr<<"WMMA mismatch rows32="<<rows32<<
                    " r="<<r<<" h="<<h<<" d="<<d<<" got="<<got<<
                    " expected="<<expected<<" count="<<count<<'\n';
                ++failures;
            }
        }
    }
    std::cout<<"WMMA "<<(rows64?64:(rows32?32:16))<<(split2?" split2":"")<<
        (split4?" split4":"")<<" rows="<<rows<<
        " base="<<base<<" zero_scores="<<zero_scores<<
        " register_owned="<<register_owned<<
        " shared_heads="<<shared_heads<<
        " keys="<<(keys64?64:32)<<
        " max_error="<<max_error<<" failures="<<failures<<'\n';
    require(failures==0,"repaired WMMA QK/causal softmax/PV differs from independent FP32 reference");
}
// Bitwise differential: the register-resident WMMA32 twin against the shared
// WMMA32 reference for split 1/2/4, every CTA head grouping, ragged row blocks,
// unaligned frontiers and multi-tile causal scans.
static void wmma32_register_bitwise(int rows,int position,int split,int heads,bool m64=false){
    constexpr int qheads=24,kvheads=4,dim=256;
    const int capacity=position+rows+7;
    std::vector<std::uint16_t> qh(static_cast<std::size_t>(rows)*qheads*dim),
        kh(static_cast<std::size_t>(capacity)*kvheads*dim),vh(kh.size());
    std::uint32_t state=0x9e3779b9U^static_cast<std::uint32_t>(rows*131+position*7+split);
    auto next=[&](float scale){
        state^=state<<13;state^=state>>17;state^=state<<5;
        const float u=float(state>>8)/float(1u<<24)*2.0f-1.0f;
        return __half_as_ushort(__float2half_rn(u*scale));
    };
    for(auto& x:qh)x=next(2.0f);
    for(auto& x:kh)x=next(2.0f);
    for(auto& x:vh)x=next(1.0f);
    H q(qh.size()),k(kh.size()),v(vh.size()),reference(qh.size()),candidate(qh.size());
    Buffer<float> reference_partial(static_cast<std::size_t>(4)*qh.size()),
        candidate_partial(static_cast<std::size_t>(4)*qh.size());
    Buffer<float> reference_stats(static_cast<std::size_t>(4)*rows*qheads*2),
        candidate_stats(static_cast<std::size_t>(4)*rows*qheads*2);
    q.set(qh);k.set(kh);v.set(vh);
    if(m64)
        fast_wmma64_attention_fixture(q.p,k.p,v.p,reference.p,reference_partial.p,
            reference_stats.p,rows,position,capacity,nullptr,0,nullptr,false,split==2);
    else if(split==1)
        fast_wmma_attention_fixture(true,q.p,k.p,v.p,reference.p,rows,position,
            capacity,nullptr,0,nullptr,false);
    else if(split==2)
        fast_wmma_split2_attention_fixture(q.p,k.p,v.p,reference.p,reference_partial.p,
            reference_stats.p,rows,position,capacity,nullptr,0,nullptr,false);
    else
        fast_wmma_split4_attention_fixture(q.p,k.p,v.p,reference.p,reference_partial.p,
            reference_stats.p,rows,position,capacity,nullptr,0,nullptr,false);
    fast_wmma32_register_attention_fixture(q.p,k.p,v.p,candidate.p,candidate_partial.p,
        candidate_stats.p,rows,position,capacity,nullptr,0,split,heads,nullptr,m64?64:32);
    ck(cudaDeviceSynchronize());
    equal(reference,candidate,"register WMMA32 output differs from reference");
    if(split>1){
        equal(reference_partial,candidate_partial,"register WMMA32 split partials differ");
        equal(reference_stats,candidate_stats,"register WMMA32 split stats differ");
    }
    std::cout<<"WMMA32_REGISTER_BITWISE rows="<<rows<<" position="<<position<<
        " split="<<split<<" heads="<<heads<<" m64="<<m64<<" equal=1\n";
}
// Verify-path attention numerics: the scalar fused flash kernel and the
// tensor-core candidate against one independent FP64 oracle (causal GQA-6,
// head dim 256, 1/16 score scale) over realistic activation magnitudes.
static void verify_flash_mma_oracle(int rows,int position,double* worst_candidate,
                                    double* worst_reference,int keys=256){
    constexpr int qheads=24,kvheads=4,dim=256;
    const int capacity=position+rows+64;
    const int segments=(position+rows+255)/256;
    std::vector<std::uint16_t> qh(static_cast<std::size_t>(rows)*qheads*dim),
        kh(static_cast<std::size_t>(capacity)*kvheads*dim),vh(kh.size());
    std::uint32_t state=0x2545f491U^static_cast<std::uint32_t>(rows*977+position);
    auto uniform=[&](){state^=state<<13;state^=state>>17;state^=state<<5;
        return float(state>>8)/float(1u<<24)*2.0f-1.0f;};
    for(auto& x:qh)x=__half_as_ushort(__float2half_rn(uniform()*3.0f));
    for(auto& x:kh)x=__half_as_ushort(__float2half_rn(uniform()*3.0f));
    for(auto& x:vh)x=__half_as_ushort(__float2half_rn(uniform()*2.0f));
    H q(qh.size()),k(kh.size()),v(vh.size()),reference(qh.size()),candidate(qh.size());
    Buffer<float> workspace(static_cast<std::size_t>(rows)*kvheads*
        ((position+rows+63)/64)*(6*dim+12));
    q.set(qh);k.set(kh);v.set(vh);
    fast_fused_attention_fixture(q.p,k.p,v.p,workspace.p,reference.p,position,capacity,
        segments,nullptr,nullptr,rows);
    fast_verify_flash_mma_fixture(q.p,k.p,v.p,workspace.p,candidate.p,position,capacity,
        segments,rows,nullptr,keys);
    ck(cudaDeviceSynchronize());
    const auto ref_out=reference.get(),cand_out=candidate.get();
    double max_ref=0,max_cand=0;
    for(int r=0;r<rows;++r)for(int h=0;h<qheads;++h){
        const int kv=h/6,count=position+r+1;
        std::vector<double> scores(count);double maximum=-1e300;
        for(int key=0;key<count;++key){double dot=0;
            for(int d=0;d<dim;++d)dot+=half_value(qh[(r*qheads+h)*dim+d])*
                half_value(kh[(key*kvheads+kv)*dim+d]);
            scores[key]=dot/16.0;maximum=std::max(maximum,scores[key]);}
        double denominator=0;for(double& x:scores){x=std::exp(x-maximum);denominator+=x;}
        for(int d=0;d<dim;++d){double numerator=0;
            for(int key=0;key<count;++key)numerator+=scores[key]*
                half_value(vh[(key*kvheads+kv)*dim+d]);
            const double expected=numerator/denominator;
            const std::size_t index=(static_cast<std::size_t>(r)*qheads+h)*dim+d;
            max_ref=std::max(max_ref,std::abs(half_value(ref_out[index])-expected));
            max_cand=std::max(max_cand,std::abs(half_value(cand_out[index])-expected));
            require(std::isfinite(half_value(cand_out[index])),"verify MMA non-finite output");
        }
    }
    *worst_candidate=std::max(*worst_candidate,max_cand);
    *worst_reference=std::max(*worst_reference,max_ref);
    std::cout<<"VERIFY_FLASH_MMA keys="<<keys<<" rows="<<rows<<" position="<<position<<
        " max_abs_error_candidate="<<max_cand<<" max_abs_error_scalar="<<max_ref<<'\n';
}
static void fused_graph_coverage(){
    constexpr int capacity=16640,heads=24,kvheads=4,dim=256;
    constexpr int captured_segments=(capacity+255)/256;
    H q(heads*dim),k(capacity*kvheads*dim),v(capacity*kvheads*dim),
        eager(heads*dim),replay(heads*dim);
    Buffer<float> workspace(kvheads*65*(6*dim+12));Buffer<int> live(1);
    q.set(std::vector<std::uint16_t>(q.n,
        __half_as_ushort(__float2half_rn(1.0f))));
    ck(cudaMemset(k.p,0,k.n*sizeof(std::uint16_t)));
    ck(cudaMemset(v.p,0,v.n*sizeof(std::uint16_t)));
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;
    ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    fast_fused_attention_fixture(q.p,k.p,v.p,workspace.p,replay.p,4223,
        capacity,captured_segments,live.p,stream);
    ck(cudaStreamEndCapture(stream,&graph));
    ck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    std::vector<std::uint16_t> sentinel(dim,
        __half_as_ushort(__float2half_rn(1.0f)));
    std::vector<std::uint16_t> value(dim,
        __half_as_ushort(__float2half_rn(7.0f)));
    int previous=-1,misses=0;
    for(int position:{4223,4224,4351,4352,4607,4608,16383,16384}){
        if(previous>=0){
            ck(cudaMemset(k.p+static_cast<std::size_t>(previous)*kvheads*dim,
                0,dim*sizeof(std::uint16_t)));
            ck(cudaMemset(v.p+static_cast<std::size_t>(previous)*kvheads*dim,
                0,dim*sizeof(std::uint16_t)));
        }
        ck(cudaMemcpy(k.p+static_cast<std::size_t>(position)*kvheads*dim,
            sentinel.data(),dim*sizeof(std::uint16_t),cudaMemcpyHostToDevice));
        ck(cudaMemcpy(v.p+static_cast<std::size_t>(position)*kvheads*dim,
            value.data(),dim*sizeof(std::uint16_t),cudaMemcpyHostToDevice));
        live.set(std::vector<int>{position});
        const int eager_segments=(position+1+255)/256;
        fast_fused_attention_fixture(q.p,k.p,v.p,workspace.p,eager.p,position,
            capacity,eager_segments,live.p,stream);
        ck(cudaStreamSynchronize(stream));
        const double expected=7.0*std::exp(16.0)/
            (std::exp(16.0)+position);
        const double eager_value=half_value(eager.get()[0]);
        std::cout<<"FUSED_EAGER position="<<position<<" expected="<<
            expected<<" got="<<eager_value<<'\n';
        require(std::abs(eager_value-expected)<0.02,
            "fused eager attention misses one-hot final key");
        ck(cudaGraphLaunch(exec,stream));ck(cudaStreamSynchronize(stream));
        const double replay_value=half_value(replay.get()[0]);
        const bool covered=std::abs(replay_value-eager_value)<0.02;
        std::cout<<"FUSED_GRID position="<<position<<" captured="<<
            captured_segments<<" eager="<<eager_segments<<" eager_value="<<
            eager_value<<" replay_value="<<replay_value<<
            " covered="<<covered<<'\n';
        if(!covered)++misses;
        previous=position;
    }
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
    ck(cudaStreamDestroy(stream));
    workspace.guard();eager.guard();replay.guard();
    require(misses==0,"fused captured replay omitted live one-hot key");
}
static void fused_multirow_coverage(){
    constexpr int capacity=16640,heads=24,kvheads=4,dim=256,rows_max=8;
    constexpr int segments_max=(capacity+255)/256;
    H q(rows_max*heads*dim),k(capacity*kvheads*dim),
        v(capacity*kvheads*dim),multi(rows_max*heads*dim),
        scalar(rows_max*heads*dim);
    Buffer<float> workspace(rows_max*kvheads*segments_max*(6*dim+12));
    q.set(std::vector<std::uint16_t>(q.n,
        __half_as_ushort(__float2half_rn(1.0f))));
    std::vector<std::uint16_t> key(kvheads*dim,
        __half_as_ushort(__float2half_rn(1.0f)));
    for(int position:{4094,4223,16380})for(int rows:{2,4,8}){
        ck(cudaMemset(k.p,0,k.n*sizeof(std::uint16_t)));
        ck(cudaMemset(v.p,0,v.n*sizeof(std::uint16_t)));
        for(int row=0;row<rows;++row){
            const auto offset=static_cast<std::size_t>(position+row)*kvheads*dim;
            std::vector<std::uint16_t> value(kvheads*dim,
                __half_as_ushort(__float2half_rn(float(row+1))));
            ck(cudaMemcpy(k.p+offset,key.data(),key.size()*sizeof(std::uint16_t),
                cudaMemcpyHostToDevice));
            ck(cudaMemcpy(v.p+offset,value.data(),value.size()*sizeof(std::uint16_t),
                cudaMemcpyHostToDevice));
        }
        const int segments=(position+rows+255)/256;
        fast_fused_attention_fixture(q.p,k.p,v.p,workspace.p,multi.p,
            position,capacity,segments,nullptr,nullptr,rows);
        for(int row=0;row<rows;++row)
            fast_fused_attention_fixture(q.p+row*heads*dim,k.p,v.p,
                workspace.p,scalar.p+row*heads*dim,position+row,capacity,
                (position+row+256)/256,nullptr);
        ck(cudaDeviceSynchronize());
        const auto batched=multi.get(),reference=scalar.get();
        for(int row=0;row<rows;++row){
            const auto first=static_cast<std::size_t>(row)*heads*dim;
            require(std::memcmp(batched.data()+first,reference.data()+first,
                heads*dim*sizeof(std::uint16_t))==0,
                "fused B2..B8 differs from scalar fused rows");
            const double e16=std::exp(16.0);
            const double weighted=double((row+1)*(row+2))/2.0;
            const double expected=e16*weighted/
                (double(position)+(row+1)*e16);
            require(std::abs(half_value(batched[first])-expected)<0.02,
                "fused B2..B8 causal one-hot oracle");
        }
        std::cout<<"FUSED_MULTIROW rows="<<rows<<" position="<<position
                 <<" scalar_equal=1 causal_oracle=1\n";
    }
    q.guard();k.guard();v.guard();multi.guard();scalar.guard();workspace.guard();
}
static void convolution(int rows){
    const int n=rows*5120;
    H input(n),dynamic(rows*1280),base(4*5120),residual(n),ca(n),cb(n),oa(n),ob(n);
    input.set(bits(n));dynamic.set(bits(dynamic.n));base.set(bits(base.n));residual.set(bits(n,true));
    gopt_draft_conv_fixture(false,input.p,dynamic.p,base.p,ca.p,residual.p,oa.p,rows);
    gopt_draft_conv_fixture(true,input.p,dynamic.p,base.p,cb.p,residual.p,ob.p,rows);
    equal(ca,cb,"GOPT-007 conv BF16");equal(oa,ob,"GOPT-007 residual");
    // Actual second finish call aliases residual destination, never convolution input.
    ob.set(residual.get());gopt_draft_conv_fixture(true,input.p,dynamic.p,base.p,cb.p,ob.p,ob.p,rows);
    equal(oa,ob,"GOPT-007 in-place residual");
}
static void dense(int rows,int k,int n){
    auto ih=bits(rows*k),wh=bits(n*k);H input(ih.size()),weight(wh.size()),a(rows*n),b(rows*n);input.set(ih);
    for(bool major:{false,true}){
        auto arranged=wh;if(major)for(int j=0;j<n;++j)for(int i=0;i<k;++i)arranged[i*n+j]=wh[j*k+i];
        weight.set(arranged);gopt_draft_dense_fixture(false,major,input.p,weight.p,a.p,rows,k,n);
        gopt_draft_dense_fixture(true,major,input.p,weight.p,b.p,rows,k,n);equal(a,b,"GOPT-010 dense exact");
        const auto output=b.get();
        // Independent FP64 dot from represented binary16, no production reduction tree.
        for(int r=0;r<rows;++r)for(int j=0;j<n;++j){
            double expected=0;for(int i=0;i<k;++i)expected+=half_value(ih[r*k+i])*half_value(wh[j*k+i]);
            require(std::abs(half_value(output[r*n+j])-expected)<=0.002*std::abs(expected)+0.00002,"GOPT-010 FP64 oracle");
        }
    }
}
static void control_pair(int rows){
    constexpr int hidden=5120, heads=48;
    H input(rows*hidden),aw(heads*hidden),bw(heads*hidden);
    const auto ih=bits(input.n),ah=bits(aw.n),bh=bits(bw.n,true);
    input.set(ih);aw.set(ah);bw.set(bh);
    Buffer<float> al(heads),bias(heads),a0(rows*heads),a1(rows*heads),
        b0(rows*heads),b1(rows*heads),beta0(rows*heads),beta1(rows*heads),
        g0(rows*heads),g1(rows*heads);
    al.set(std::vector<float>(heads,-0.5f));
    bias.set(std::vector<float>(heads,0.125f));
    auto launch=[&](bool paired,cudaStream_t stream){
        gopt_gdn_control_row_pair_fixture(paired,input.p,aw.p,bw.p,al.p,bias.p,
            paired?a1.p:a0.p,paired?b1.p:b0.p,
            paired?beta1.p:beta0.p,paired?g1.p:g0.p,rows,stream);
    };
    launch(false,nullptr);launch(true,nullptr);
    equal(a0,a1,"GOPT-011 control A exact");equal(b0,b1,"GOPT-011 control B exact");
    equal(beta0,beta1,"GOPT-011 beta exact");equal(g0,g1,"GOPT-011 gate exact");
    captured([&](cudaStream_t stream){launch(true,stream);});
    equal(a0,a1,"GOPT-011 graph A exact");equal(b0,b1,"GOPT-011 graph B exact");
    const auto av=a1.get(),bv=b1.get();
    for(int r=0;r<rows;++r)for(int h:{0,17,47}){
        double expected_a=0,expected_b=0;
        for(int d=0;d<hidden;++d){
            expected_a+=half_value(ih[r*hidden+d])*half_value(ah[h*hidden+d]);
            expected_b+=half_value(ih[r*hidden+d])*half_value(bh[h*hidden+d]);
        }
        const int index=r*heads+h;
        require(std::abs(av[index]-expected_a)<1e-5+1e-5*std::abs(expected_a),"GOPT-011 A FP64 oracle");
        require(std::abs(bv[index]-expected_b)<1e-5+1e-5*std::abs(expected_b),"GOPT-011 B FP64 oracle");
    }
}
int main(int argc,char** argv){try{
    require(argc<=2,"pass optional candidate number 1..14 (9 uses greedy-device executable)");
    const int requested=argc==2?std::atoi(argv[1]):0;
    require(argc==1 || (requested>=1&&requested<=16&&requested!=9),"operator candidate id");
    for(int which=1;which<=16;++which) {
    if(which==9 || (requested && which!=requested))continue;
    if(which==1)for(int rows:{2,3,7,8})recurrence(rows);
    for(int rows:{1,2,3,7,8,9,16}){
        if(which==2||which==3)staging(rows,which);
        if(which==4||which==8)norms(rows,which);
        if(which==5||which==6){rotary(rows,8,which);rotary(rows,32,which);}
        if(which==7)convolution(rows);
        if(which==10)dense(rows,129,257);
    }
    if(which==10){dense(7,5120,256);dense(8,5120,1280);}
    if(which==11)for(int rows:{1,2,3,7,8,9,16})control_pair(rows);
    if(which==12)for(int rows:{1,2,7,8,16})for(int first:{0,2046,4095})ring_norm(rows,first);
    if(which==13){
        qk_layout(0);qk_layout(16);
        pv_layout(0,0,16);pv_layout(16,32,27);pv_layout(16,192,29);
        wmma_attention_reference(false,17,3,5,false,true);
        wmma_attention_reference(true,33,30,3,true,true);
        wmma_attention_reference(false,17,3,5,false,false);
        wmma_attention_reference(true,33,30,3,true,false);
        wmma_attention_reference(true,33,30,3,true,false,true);
        wmma_attention_reference(true,17,3,5,false,true,true);
        wmma_attention_reference(true,33,30,3,true,false,true,true);
        wmma_attention_reference(true,17,3,5,false,true,false,true);
        wmma_attention_reference(true,33,30,3,true,true,true,true);
        wmma_attention_reference(true,33,30,3,true,false,true,false,true);
        wmma_attention_reference(true,17,3,5,false,true,false,false,true);
        wmma_attention_reference(true,64,8,3,true,false,false,false,false,true);
        wmma_attention_reference(true,64,8,3,true,false,true,true,false,true);
        wmma_attention_reference(true,64,8,3,true,true,true,true,false,true);
        wmma_attention_reference(true,65,8,3,true,false,true,true,false,true);
        wmma_attention_reference(true,64,8,3,true,false,false,false,
            false,true,true);
        wmma_attention_reference(true,64,8,3,true,false,true,true,
            false,true,true);
        wmma_attention_reference(true,64,8,3,true,true,true,true,
            false,true,true);
        wmma_attention_reference(true,65,8,3,true,false,true,true,
            false,true,true);
        wmma_attention_reference(true,64,8,3,true,false,false,false,
            false,true,true,true);
        wmma_attention_reference(true,64,8,3,true,false,true,true,
            false,true,true,true);
        wmma_attention_reference(true,64,8,3,true,true,true,true,
            false,true,true,true);
        wmma_attention_reference(true,65,8,3,true,false,true,true,
            false,true,true,true);
        wmma_attention_reference(true,64,8,3,true,false,false,false,
            false,true,true,false,true);
        wmma_attention_reference(true,64,8,3,true,false,true,true,
            false,true,true,false,true);
        wmma_attention_reference(true,64,8,3,true,true,true,true,
            false,true,true,false,true);
        wmma_attention_reference(true,65,8,3,true,false,true,true,
            false,true,true,false,true);
    }
    if(which==14){fused_graph_coverage();fused_multirow_coverage();}
    if(which==16){
        double worst_candidate=0,worst_reference=0;
        for(auto shape:std::vector<std::pair<int,int>>{{1,0},{8,0},{8,31},{3,255},{8,256},
                {8,1000},{5,2047},{8,4095},{8,8190}})
            for(int keys:{256,128,64})
                verify_flash_mma_oracle(shape.first,shape.second,&worst_candidate,
                    &worst_reference,keys);
        std::cout<<"VERIFY_FLASH_MMA worst candidate="<<worst_candidate<<
            " scalar="<<worst_reference<<'\n';
        require(worst_candidate<=2.0*worst_reference+2e-3,
            "verify MMA attention error exceeds the scalar route envelope");
    }
    if(which==15){
        for(int heads:{1,2,3})for(int split:{1,2,4})
            for(auto shape:std::vector<std::pair<int,int>>{{1,0},{17,5},{32,0},{33,30},
                    {64,1000},{100,2047},{1024,0},{1024,3072},{256,4001}})
                wmma32_register_bitwise(shape.first,shape.second,split,heads);
        for(int heads:{2,3})for(int split:{1,2})
            for(auto shape:std::vector<std::pair<int,int>>{{64,0},{128,33},{1024,0},
                    {1024,3072},{512,8000}})
                wmma32_register_bitwise(shape.first,shape.second,split,heads,true);
    }
    }
    std::cout<<"PASS GOPT operator differentials, bounded oracles, tails and graph replay\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
