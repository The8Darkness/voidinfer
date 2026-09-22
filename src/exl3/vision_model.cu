#include "exl3/vision_model.h"
#include "exl3/encoded_media_cache.h"
#include "exl3/linear_cuda.h"
#include "exl3/resource_inventory.h"
#include "exl3/safetensors.h"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <nlohmann/json.hpp>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace ninfer::exl3 {
namespace {
constexpr int H=1152, D=72, Heads=16, Up=4352, Merge=4608, Out=5120, Patch=1536;
void require(bool ok,const std::string& what){if(!ok)throw std::runtime_error("V6: "+what);}
void check(cudaError_t value,const char* what){require(value==cudaSuccess,std::string(what)+": "+cudaGetErrorString(value));}
void blas(cublasStatus_t value,const char* what){require(value==CUBLAS_STATUS_SUCCESS,what);}
struct Buffer {
    void* data=nullptr; std::size_t bytes=0;
    explicit Buffer(std::size_t size):bytes(size){check(cudaMalloc(&data,size),"allocate");}
    ~Buffer(){if(data)cudaFree(data);}
    Buffer(const Buffer&)=delete;
    template<class T>T* as()const{return static_cast<T*>(data);}
};
struct Linear {Exl3CudaLinearWeights weights;Exl3CudaLinearMetadata metadata;const half* bias=nullptr;};
struct Norm {const half *weight=nullptr,*bias=nullptr;};
struct Layer {Linear q,k,v,o,up,down;Norm n1,n2;};

__global__ void add_bias_half(half* x,const half* bias,int count,int width,bool gelu){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    half value=__hadd(x[i],bias[i%width]);
    if(gelu){float v=__half2float(value);value=__float2half_rn(0.5f*v*(1+tanhf(0.7978845608028654f*(v+0.044715f*v*v*v))));}
    x[i]=value;
}
__global__ void add_bias_float(float* x,const half* bias,int count,int width){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)x[i]=__fadd_rn(x[i],__half2float(bias[i%width]));
}
__global__ void add_residual(float* x,const half* delta,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)x[i]=__fadd_rn(x[i],__half2float(delta[i]));
}
__global__ void position_embed(float* x,const half* table,const int* indices,const float* weights,int count){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    int row=i/H, c=i%H;
    // Separate framework tensor products/sums each round to FP16. _rn prevents
    // nvcc from contracting a product and its following addition into half FMA.
    half value=__hmul_rn(table[indices[row*4]*H+c],__float2half_rn(weights[row*4]));
    for(int j=1;j<4;++j)value=__hadd_rn(value,__hmul_rn(table[indices[row*4+j]*H+c],__float2half_rn(weights[row*4+j])));
    x[i]=__fadd_rn(x[i],__half2float(value));
}
__global__ void layer_norm(const float* x,half* y,const half* weight,const half* bias){
    __shared__ float scratch[256];
    int row=blockIdx.x,t=threadIdx.x;float sum=0;
    for(int c=t;c<H;c+=256)sum+=x[row*H+c];
    scratch[t]=sum;__syncthreads();
    for(int n=128;n;n/=2){if(t<n)scratch[t]+=scratch[t+n];__syncthreads();}
    float mean=scratch[0]/H;sum=0;
    for(int c=t;c<H;c+=256){float v=x[row*H+c]-mean;sum=fmaf(v,v,sum);}
    __syncthreads();scratch[t]=sum;__syncthreads();
    for(int n=128;n;n/=2){if(t<n)scratch[t]+=scratch[t+n];__syncthreads();}
    float inv=rsqrtf(scratch[0]/H+1e-6f);
    for(int c=t;c<H;c+=256)y[row*H+c]=__float2half_rn(
        __fadd_rn(__fmul_rn((x[row*H+c]-mean)*inv,__half2float(weight[c])),__half2float(bias[c])));
}
__global__ void rope2d(half* q,half* k,const int* positions,int rows){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*Heads*(D/2))return;
    int pair=i%(D/2),head=(i/(D/2))%Heads,row=i/(Heads*(D/2));
    float angle=positions[(pair/18)*rows+row]*powf(10000.0f,-float(pair%18)/18.0f);
    float s=sinf(angle),c=cosf(angle);int a=row*H+head*D+pair,b=a+D/2;
    float qa=__half2float(q[a]),qb=__half2float(q[b]),ka=__half2float(k[a]),kb=__half2float(k[b]);
    q[a]=__float2half_rn(__fsub_rn(qa*c,qb*s));q[b]=__float2half_rn(__fadd_rn(qb*c,qa*s));
    k[a]=__float2half_rn(__fsub_rn(ka*c,kb*s));k[b]=__float2half_rn(__fadd_rn(kb*c,ka*s));
}
// Full noncausal attention inside each temporal segment. No N*N device arena;
// bounded one-row softmax scratch. Explicit numerical reference-style execution.
__global__ void attention(const half* q,const half* k,const half* v,half* out,int segment){
    extern __shared__ float scores[];
    __shared__ float reduction[128];
    int row=blockIdx.x,head=blockIdx.y,t=threadIdx.x,first=(row/segment)*segment;
    float maximum=-CUDART_INF_F;
    for(int j=t;j<segment;j+=128){float dot=0;
        for(int c=0;c<D;++c)dot=fmaf(__half2float(q[row*H+head*D+c]),__half2float(k[(first+j)*H+head*D+c]),dot);
        scores[j]=dot*0.1178511301977579f;maximum=fmaxf(maximum,scores[j]);}
    reduction[t]=maximum;__syncthreads();
    for(int n=64;n;n/=2){if(t<n)reduction[t]=fmaxf(reduction[t],reduction[t+n]);__syncthreads();}
    maximum=reduction[0];float sum=0;
    for(int j=t;j<segment;j+=128){scores[j]=expf(scores[j]-maximum);sum+=scores[j];}
    __syncthreads();reduction[t]=sum;__syncthreads();
    for(int n=64;n;n/=2){if(t<n)reduction[t]+=reduction[t+n];__syncthreads();}
    if(t<D){float value=0;for(int j=0;j<segment;++j)value=fmaf(scores[j]/reduction[0],__half2float(v[(first+j)*H+head*D+t]),value);
        out[row*H+head*D+t]=__float2half_rn(value);}
}
}

struct Exl3VisionModel::Impl {
    int device=0;std::size_t bytes=0;IndexedSafetensors files;
    std::vector<std::unique_ptr<Buffer>> allocations;
    const half *patch_weight=nullptr,*patch_bias=nullptr,*positions=nullptr;
    std::array<Layer,27> layers;Norm merger_norm;Linear merger_up,merger_down;
    TensorInfo info(const std::string& name){for(const auto& shard:files.shards)if(auto p=shard.header.find(name))return *p;throw std::runtime_error("Missing V6 tensor "+name);}
    const void* tensor(const std::string& name,const std::string& dtype,const std::vector<std::uint64_t>& shape){
        auto meta=info(name);require(meta.dtype==dtype&&meta.shape==shape,"tensor extent/dtype "+name);
        for(const auto& shard:files.shards)if(shard.header.find(name)){
            auto payload=read_tensor(shard.path,shard.header,name);auto buffer=std::make_unique<Buffer>(payload.bytes().size());
            check(cudaMemcpy(buffer->data,payload.bytes().data(),payload.bytes().size(),cudaMemcpyHostToDevice),"upload immutable vision tensor");
            const void* result=buffer->data;bytes+=buffer->bytes;allocations.push_back(std::move(buffer));return result;}
        throw std::runtime_error("V6 tensor lookup");
    }
    const half* half_tensor(const std::string& name,std::vector<std::uint64_t> shape){return static_cast<const half*>(tensor(name,"F16",shape));}
    Norm norm(const std::string& name){return {half_tensor(name+".weight",{H}),half_tensor(name+".bias",{H})};}
    Linear linear(const std::string& name,int in,int out){
        Linear l;l.metadata={in,out,6,false,true,false};
        l.weights={static_cast<const std::uint16_t*>(tensor(name+".trellis","I16",{std::uint64_t(in/16),std::uint64_t(out/16),96})),
            reinterpret_cast<const std::uint16_t*>(half_tensor(name+".suh",{std::uint64_t(in)})),
            reinterpret_cast<const std::uint16_t*>(half_tensor(name+".svh",{std::uint64_t(out)})),
            static_cast<const std::int32_t*>(tensor(name+".mul1","I32",{}))};
        l.bias=half_tensor(name+".bias",{std::uint64_t(out)});return l;
    }
    explicit Impl(const std::filesystem::path& directory){
        check(cudaGetDevice(&device),"model device");
        std::ifstream config(directory/"config.json");auto json=nlohmann::json::parse(config);const auto& v=json.at("vision_config");
        require(v.at("depth")==27&&v.at("hidden_size")==H&&v.at("num_heads")==Heads&&v.at("intermediate_size")==4304&&
            v.at("out_hidden_size")==Out&&v.at("patch_size")==16&&v.at("temporal_patch_size")==2&&v.at("spatial_merge_size")==2&&
            v.at("num_position_embeddings")==2304&&v.at("hidden_act")=="gelu_pytorch_tanh"&&v.at("deepstack_visual_indexes").empty(),"pinned vision config");
        files=inspect_indexed_directory(directory);
        patch_weight=half_tensor("model.visual.patch_embed.proj.weight",{H,3,2,16,16});patch_bias=half_tensor("model.visual.patch_embed.proj.bias",{H});
        positions=half_tensor("model.visual.pos_embed.weight",{2304,H});
        for(int i=0;i<27;++i){auto& l=layers[i];std::string b="model.visual.blocks."+std::to_string(i);
            l.n1=norm(b+".norm1");l.n2=norm(b+".norm2");
            l.q=linear(b+".attn.q_proj",H,H);l.k=linear(b+".attn.k_proj",H,H);l.v=linear(b+".attn.v_proj",H,H);l.o=linear(b+".attn.proj",H,H);
            l.up=linear(b+".mlp.linear_fc1",H,Up);l.down=linear(b+".mlp.linear_fc2",Up,H);}
        merger_norm=norm("model.visual.merger.norm");merger_up=linear("model.visual.merger.linear_fc1",Merge,Merge);merger_down=linear("model.visual.merger.linear_fc2",Merge,Out);
    }
};

struct Exl3VisionContext::Impl {
    std::shared_ptr<Exl3VisionModel::Impl> model;int capacity;bool poisoned=false;std::recursive_mutex mutex;
    mutable std::mutex encoded_cache_mutex;
    std::vector<std::shared_ptr<Exl3EncodedMediaEntry>> encoded_cache;
    Exl3EncodedMediaCacheStats encoded_cache_stats;
    Exl3EncodedMediaCacheLimits encoded_cache_limits;
    RetainedHostAllocationLedger encoded_output_ledger,replay_payload_ledger;
    cudaStream_t stream=nullptr;cublasHandle_t handle=nullptr;
    std::vector<std::unique_ptr<Buffer>> allocations;std::size_t bytes=0;
    half *patches=nullptr,*normalized=nullptr,*q=nullptr,*k=nullptr,*v=nullptr,*temp=nullptr,*up=nullptr;
    float *residual=nullptr,*merged=nullptr,*position_weights=nullptr;int *position_ids=nullptr,*table_indices=nullptr;
    std::unique_ptr<Exl3CudaReconstructGemmWorkspace> hidden,mlp,merge_up,merge_down;
    void* alloc(std::size_t size){auto p=std::make_unique<Buffer>(size);auto ptr=p->data;bytes+=size;allocations.push_back(std::move(p));return ptr;}
    explicit Impl(std::shared_ptr<Exl3VisionModel::Impl> m,int maximum,
        Exl3EncodedMediaCacheLimits limits):model(std::move(m)),capacity(maximum),
        encoded_cache_limits(limits){
        require(capacity>=4&&capacity<=1024&&capacity%4==0,"bounded patch capacity 4..1024, multiple of4");
        require(encoded_cache_limits.max_entries &&
            encoded_cache_limits.max_encoded_host_bytes &&
            encoded_cache_limits.max_replay_host_bytes,"encoded cache limits");
        int device;check(cudaGetDevice(&device),"context device");require(device==model->device,"context/model device mismatch");
        try{check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");blas(cublasCreate(&handle),"patch BLAS handle");blas(cublasSetStream(handle,stream),"patch stream");
            patches=static_cast<half*>(alloc(std::size_t(capacity)*Patch*2));
            normalized=static_cast<half*>(alloc(std::size_t(capacity)*H*2));
            q=static_cast<half*>(alloc(std::size_t(capacity)*H*2));k=static_cast<half*>(alloc(std::size_t(capacity)*H*2));v=static_cast<half*>(alloc(std::size_t(capacity)*H*2));
            temp=static_cast<half*>(alloc(std::size_t(capacity)*H*2));up=static_cast<half*>(alloc(std::size_t(capacity)*Up*2));
            residual=static_cast<float*>(alloc(std::size_t(capacity)*H*4));merged=static_cast<float*>(alloc(std::size_t(capacity/4)*Out*4));
            position_ids=static_cast<int*>(alloc(std::size_t(capacity)*2*4));table_indices=static_cast<int*>(alloc(std::size_t(capacity)*4*4));position_weights=static_cast<float*>(alloc(std::size_t(capacity)*4*4));
            hidden=std::make_unique<Exl3CudaReconstructGemmWorkspace>(H,H,capacity);
            mlp=std::make_unique<Exl3CudaReconstructGemmWorkspace>(H,Up,capacity,true);
            merge_up=std::make_unique<Exl3CudaReconstructGemmWorkspace>(Merge,Merge,capacity/4);
            merge_down=std::make_unique<Exl3CudaReconstructGemmWorkspace>(Merge,Out,capacity/4);
            bytes+=hidden->workspace_bytes()+mlp->workspace_bytes()+merge_up->workspace_bytes()+merge_down->workspace_bytes();
        }catch(...){if(handle)cublasDestroy(handle);if(stream)cudaStreamDestroy(stream);throw;}
    }
    ~Impl(){if(stream)cudaStreamSynchronize(stream);if(handle)cublasDestroy(handle);if(stream)cudaStreamDestroy(stream);}
    void projection(const Linear& l,Exl3CudaReconstructGemmWorkspace& workspace,const half* x,half* y,int rows,bool gelu=false){
        workspace.forward_v6_numeric(l.weights,l.metadata,reinterpret_cast<const std::uint16_t*>(x),reinterpret_cast<std::uint16_t*>(y),nullptr,rows,stream);
        int count=rows*l.metadata.out_features;add_bias_half<<<(count+255)/256,256,0,stream>>>(y,l.bias,count,l.metadata.out_features,gelu);
    }
};

Exl3VisionModel::Exl3VisionModel(const std::filesystem::path& directory):impl_(std::make_shared<Impl>(directory)){}
Exl3VisionModel::~Exl3VisionModel()=default;
std::size_t Exl3VisionModel::device_bytes()const noexcept{return impl_->bytes;}
Exl3VisionContext::Exl3VisionContext(const Exl3VisionModel& model,int capacity,
    Exl3EncodedMediaCacheLimits limits):impl_(std::make_unique<Impl>(model.impl_,capacity,limits)){}
Exl3VisionContext::~Exl3VisionContext()=default;
std::size_t Exl3VisionContext::device_bytes()const noexcept{return impl_->bytes;}
Exl3VisionOutput Exl3VisionContext::encode_numeric_candidate(
    const targets::qwen3_6::PreparedMediaPayload& payload,const targets::qwen3_6::VisionItemControl& control,
    const std::function<bool()>& cancelled,const std::function<void(int,std::span<const float>)>& observer){
    auto& c=*impl_;std::unique_lock lock(c.mutex);require(!c.poisoned,"poisoned context");
    int device;check(cudaGetDevice(&device),"encode device");require(device==c.model->device,"encode device mismatch");
    require(payload.storage==VisionPatchStorage::Float16,"requires FP16 patches");
    require(payload.preprocess.valid(control.modality==targets::qwen3_6::PromptModality::Video),
        "prepared media preprocessing identity is missing or mismatched");
    require(control.patch_count>0&&control.patch_count<=std::size_t(c.capacity)&&control.patch_count%4==0,"patch capacity/extent");
    int rows=static_cast<int>(control.patch_count),count=rows*H;
    require(payload.span().size()==std::size_t(rows)*Patch,"payload extent");
    require(control.grid.temporal>0&&control.grid.height>0&&control.grid.width>0&&control.grid.height%2==0&&control.grid.width%2==0&&
        std::int64_t(control.grid.temporal)*control.grid.height*control.grid.width==rows,"grid extent");
    require(control.segment_length==control.grid.height*control.grid.width&&control.segment_count==control.grid.temporal&&control.merged_count==std::size_t(rows/4),"segment/merger extent");
    require(control.position_ids.size()==std::size_t(rows)*2&&control.position_table_indices.size()==std::size_t(rows)*4&&control.position_table_weights.size()==std::size_t(rows)*4,"position control extent");
    for(int i:control.position_table_indices)require(i>=0&&i<2304,"position table index");
    for(int i:control.position_ids)require(i>=0&&i<=1024,"position id");
    for(float w:control.position_table_weights)require(w>=0&&w<=1,"interpolation weight");
    auto start=std::chrono::steady_clock::now();auto stop=[&]{if(cancelled&&cancelled())throw std::runtime_error("V6 cancelled");};
    if(control.segment_count>1){
        // Row-count-dependent GEMM reductions amplified across27 blocks in the
        // joint-frame numeric screen. Match the actual independent image extent
        // for every temporal segment; never let another frame change dispatch.
        stop();const int n=control.segment_length;
        targets::qwen3_6::PreparedMediaPayload frame;frame.storage=VisionPatchStorage::Float16;
        frame.preprocess=payload.preprocess;
        frame.patch_elements=std::size_t(n)*Patch;frame.patches=std::make_unique<std::uint16_t[]>(frame.patch_elements);
        Exl3VisionOutput result;result.embeddings.reserve(std::size_t(rows/4)*Out);
        result.stats={c.model->bytes,c.bytes,std::size_t(rows),std::size_t(rows/4),0,0,frame.patch_elements*2};
        for(int t=0;t<control.segment_count;++t){
            stop();std::copy_n(payload.span().data()+std::size_t(t)*frame.patch_elements,frame.patch_elements,frame.patches.get());
            auto fc=control;fc.grid.temporal=1;fc.patch_count=n;fc.merged_count=n/4;fc.segment_count=1;fc.patch_begin=0;
            fc.position_ids.clear();for(int axis=0;axis<2;++axis)fc.position_ids.insert(fc.position_ids.end(),control.position_ids.begin()+axis*rows+t*n,control.position_ids.begin()+axis*rows+(t+1)*n);
            fc.position_table_indices.assign(control.position_table_indices.begin()+t*n*4,control.position_table_indices.begin()+(t+1)*n*4);
            fc.position_table_weights.assign(control.position_table_weights.begin()+t*n*4,control.position_table_weights.begin()+(t+1)*n*4);
            auto encoded=encode_numeric_candidate(frame,fc,cancelled,observer);
            result.stats.packed_projections+=encoded.stats.packed_projections;
            result.embeddings.insert(result.embeddings.end(),encoded.embeddings.begin(),encoded.embeddings.end());
        }
        result.stats.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return result;
    }
    auto observe=[&](int layer){if(observer){std::vector<float> data(count);check(cudaMemcpyAsync(data.data(),c.residual,count*4,cudaMemcpyDeviceToHost,c.stream),"trace copy");check(cudaStreamSynchronize(c.stream),"trace fence");observer(layer,data);}};
    const char* audit_env=std::getenv("NINFER_V6_OPERATOR_AUDIT");
    const bool operator_audit=observer&&audit_env&&std::string(audit_env)=="1";
    auto observe_half=[&](int code,const half* source,int elements){if(operator_audit){
        std::vector<half> raw(elements);std::vector<float> data(elements);
        check(cudaMemcpyAsync(raw.data(),source,elements*2,cudaMemcpyDeviceToHost,c.stream),"operator trace copy");check(cudaStreamSynchronize(c.stream),"operator trace fence");
        for(int j=0;j<elements;++j)data[j]=__half2float(raw[j]);observer(code,data);}};
    try{stop();
        check(cudaMemcpyAsync(c.patches,payload.span().data(),payload.span().size_bytes(),cudaMemcpyHostToDevice,c.stream),"patch copy");
        check(cudaMemcpyAsync(c.position_ids,control.position_ids.data(),rows*2*4,cudaMemcpyHostToDevice,c.stream),"position copy");
        check(cudaMemcpyAsync(c.table_indices,control.position_table_indices.data(),rows*4*4,cudaMemcpyHostToDevice,c.stream),"table index copy");
        check(cudaMemcpyAsync(c.position_weights,control.position_table_weights.data(),rows*4*4,cudaMemcpyHostToDevice,c.stream),"table weight copy");
        float alpha=1,beta=0;blas(cublasGemmEx(c.handle,CUBLAS_OP_T,CUBLAS_OP_N,H,rows,Patch,&alpha,c.model->patch_weight,CUDA_R_16F,Patch,
            c.patches,CUDA_R_16F,Patch,&beta,c.residual,CUDA_R_32F,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"patch projection");
        add_bias_float<<<(count+255)/256,256,0,c.stream>>>(c.residual,c.model->patch_bias,count,H);
        observe(-2);
        position_embed<<<(count+255)/256,256,0,c.stream>>>(c.residual,c.model->positions,c.table_indices,c.position_weights,count);observe(-1);
        for(int i=0;i<27;++i){stop();const auto& l=c.model->layers[i];
            layer_norm<<<rows,256,0,c.stream>>>(c.residual,c.normalized,l.n1.weight,l.n1.bias);
            const int trace=Exl3VisionBoundaryCode::block_operator(i,0);const bool probe=i==0||i>=24;
            if(probe)observe_half(trace,c.normalized,count);
            c.projection(l.q,*c.hidden,c.normalized,c.q,rows);c.projection(l.k,*c.hidden,c.normalized,c.k,rows);c.projection(l.v,*c.hidden,c.normalized,c.v,rows);
            if(probe){observe_half(trace+1,c.q,count);observe_half(trace+2,c.k,count);observe_half(trace+3,c.v,count);}
            rope2d<<<(rows*Heads*36+255)/256,256,0,c.stream>>>(c.q,c.k,c.position_ids,rows);
            if(probe){observe_half(trace+4,c.q,count);observe_half(trace+5,c.k,count);}
            attention<<<dim3(rows,Heads),128,control.segment_length*4,c.stream>>>(c.q,c.k,c.v,c.temp,control.segment_length);
            if(probe)observe_half(trace+6,c.temp,count);
            c.projection(l.o,*c.hidden,c.temp,c.q,rows);add_residual<<<(count+255)/256,256,0,c.stream>>>(c.residual,c.q,count);
            if(probe)observe_half(trace+7,c.q,count);
            layer_norm<<<rows,256,0,c.stream>>>(c.residual,c.normalized,l.n2.weight,l.n2.bias);
            if(probe)observe_half(trace+8,c.normalized,count);
            c.projection(l.up,*c.mlp,c.normalized,c.up,rows,true);if(probe)observe_half(trace+9,c.up,rows*Up);
            c.projection(l.down,*c.mlp,c.up,c.temp,rows);if(probe)observe_half(trace+10,c.temp,count);
            add_residual<<<(count+255)/256,256,0,c.stream>>>(c.residual,c.temp,count);check(cudaGetLastError(),"vision block");observe(i);
        }
        stop();layer_norm<<<rows,256,0,c.stream>>>(c.residual,c.normalized,c.model->merger_norm.weight,c.model->merger_norm.bias);
        observe_half(Exl3VisionBoundaryCode::merger_normalized,c.normalized,rows*H);
        c.projection(c.model->merger_up,*c.merge_up,c.normalized,c.up,rows/4,true);
        observe_half(Exl3VisionBoundaryCode::merger_up,c.up,(rows/4)*Merge);
        c.merge_down->forward_v6_numeric(c.model->merger_down.weights,c.model->merger_down.metadata,reinterpret_cast<std::uint16_t*>(c.up),nullptr,c.merged,rows/4,c.stream);
        int final_count=(rows/4)*Out;add_bias_float<<<(final_count+255)/256,256,0,c.stream>>>(c.merged,c.model->merger_down.bias,final_count,Out);
        Exl3VisionOutput result;result.embeddings.resize(final_count);
        check(cudaMemcpyAsync(result.embeddings.data(),c.merged,final_count*4,cudaMemcpyDeviceToHost,c.stream),"merged output copy");
        check(cudaStreamSynchronize(c.stream),"complete media fence");stop();
        if(operator_audit)observer(Exl3VisionBoundaryCode::merger_output,result.embeddings);
        result.stats={c.model->bytes,c.bytes,std::size_t(rows),std::size_t(rows/4),27*6+2,
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()};return result;
    }catch(...){if(cudaStreamSynchronize(c.stream)!=cudaSuccess)c.poisoned=true;throw;}
}

std::shared_ptr<const Exl3EncodedMediaResult> Exl3VisionContext::encode_prepared_cached(
    std::shared_ptr<const targets::qwen3_6::PreparedMediaPayload> payload,
    const targets::qwen3_6::VisionItem& item,
    const targets::qwen3_6::VisionItemControl& control,
    const std::function<bool()>& cancelled,
    const Exl3EncodedMediaRetentionReserve& reserve) {
    auto& c=*impl_;
    bool cancellation_counted=false;
    const auto cancellation_requested=[&] {
        const bool requested=cancelled && cancelled();
        if(requested && !cancellation_counted) {
            std::lock_guard lock(c.encoded_cache_mutex);
            ++c.encoded_cache_stats.cancelled_consumers;
            cancellation_counted=true;
        }
        return requested;
    };
    const auto check_cancel=[&] {
        if(cancellation_requested())
            throw std::runtime_error("encoded media cache consumer cancelled");
    };
    check_cancel();
    std::shared_ptr<const void> encoder(c.model,c.model.get());
    std::shared_ptr<Exl3EncodedMediaEntry> entry;
    bool producer=false;
    {
        std::lock_guard lock(c.encoded_cache_mutex);
        for(const auto& candidate:c.encoded_cache)
            if(candidate->matches(payload,item,encoder,control)) {
                entry=candidate;++c.encoded_cache_stats.hits;break;
        }
        if(!entry) {
            if(payload->patch_elements>std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t) ||
               control.merged_count>std::numeric_limits<std::size_t>::max()/(5120ULL*sizeof(float)))
                throw std::overflow_error("encoded cache retention extent overflow");
            const std::uint64_t replay_bytes=sizeof(targets::qwen3_6::PreparedMediaPayload)+
                payload->patch_elements*sizeof(std::uint16_t);
            const std::uint64_t output_bytes=control.merged_count*5120ULL*sizeof(float);
            const auto fits=[&] {
                return c.encoded_cache.size()<c.encoded_cache_limits.max_entries &&
                    c.replay_payload_ledger.bytes()<=c.encoded_cache_limits.max_replay_host_bytes &&
                    replay_bytes<=c.encoded_cache_limits.max_replay_host_bytes-c.replay_payload_ledger.bytes() &&
                    c.encoded_output_ledger.bytes()<=c.encoded_cache_limits.max_encoded_host_bytes &&
                    output_bytes<=c.encoded_cache_limits.max_encoded_host_bytes-c.encoded_output_ledger.bytes();
            };
            while(!fits()) {
                const auto victim=std::find_if(c.encoded_cache.begin(),c.encoded_cache.end(),
                    [](const auto& candidate) {
                        return candidate->completion()==Exl3EncodedMediaEntry::Completion::ready;
                    });
                if(victim==c.encoded_cache.end()) {
                    ++c.encoded_cache_stats.quota_refusals;
                    throw Exl3ResourceReservationExhausted{};
                }
                c.encoded_cache.erase(victim);++c.encoded_cache_stats.evictions;
            }
            auto replay_credit=c.replay_payload_ledger.acquire(replay_bytes);
            auto output_credit=c.encoded_output_ledger.acquire(output_bytes);
            auto external=reserve?reserve(replay_bytes,output_bytes):
                Exl3EncodedMediaRetentionCredits{};
            entry=std::make_shared<Exl3EncodedMediaEntry>(
                payload,item,encoder,control,std::move(replay_credit),
                std::move(output_credit),std::move(external.replay),
                std::move(external.output));
            c.encoded_cache.push_back(entry);producer=true;
            ++c.encoded_cache_stats.misses;
            c.encoded_cache_stats.entries=c.encoded_cache.size();
            ++c.encoded_cache_stats.inflight_producers;
            c.encoded_cache_stats.inflight_scratch_device_bytes=c.bytes;
        }
    }
    if(producer) {
        try {
            auto encoded=encode_numeric_candidate(*payload,control,cancellation_requested);
            check_cancel();
            entry->publish(std::move(encoded.embeddings));
            std::lock_guard lock(c.encoded_cache_mutex);
            --c.encoded_cache_stats.inflight_producers;
            if(!c.encoded_cache_stats.inflight_producers)
                c.encoded_cache_stats.inflight_scratch_device_bytes=0;
        } catch(...) {
            entry->fail();
            std::lock_guard lock(c.encoded_cache_mutex);
            ++c.encoded_cache_stats.failed_producers;
            --c.encoded_cache_stats.inflight_producers;
            if(!c.encoded_cache_stats.inflight_producers)
                c.encoded_cache_stats.inflight_scratch_device_bytes=0;
            const auto found=std::find(c.encoded_cache.begin(),c.encoded_cache.end(),entry);
            if(found!=c.encoded_cache.end())c.encoded_cache.erase(found);
            c.encoded_cache_stats.entries=c.encoded_cache.size();
            throw;
        }
    }
    check_cancel();return entry->wait_ready(cancellation_requested);
}

Exl3EncodedMediaCacheStats Exl3VisionContext::encoded_media_cache_stats() const {
    auto& c=*impl_;std::lock_guard lock(c.encoded_cache_mutex);
    auto result=c.encoded_cache_stats;result.entries=c.encoded_cache.size();
    result.retained_encoded_host_bytes=c.encoded_output_ledger.bytes();
    result.retained_replay_host_bytes=c.replay_payload_ledger.bytes();
    result.encoder_scratch_device_bytes=c.bytes;return result;
}

void Exl3VisionContext::set_encoded_media_cache_limits(Exl3EncodedMediaCacheLimits limits) {
    if(!limits.max_entries || !limits.max_encoded_host_bytes || !limits.max_replay_host_bytes)
        throw std::invalid_argument("encoded cache limits");
    auto& c=*impl_;std::lock_guard lock(c.encoded_cache_mutex);
    if(c.encoded_cache_stats.inflight_producers)
        throw std::logic_error("encoded cache policy change while producer active");
    while((c.encoded_cache.size()>limits.max_entries ||
           c.encoded_output_ledger.bytes()>limits.max_encoded_host_bytes ||
           c.replay_payload_ledger.bytes()>limits.max_replay_host_bytes) &&
          !c.encoded_cache.empty()) {
        const auto victim=std::find_if(c.encoded_cache.begin(),c.encoded_cache.end(),
            [](const auto& candidate) {
                return candidate->completion()==Exl3EncodedMediaEntry::Completion::ready;
            });
        if(victim==c.encoded_cache.end())break;
        c.encoded_cache.erase(victim);++c.encoded_cache_stats.evictions;
    }
    if(c.encoded_cache.size()>limits.max_entries ||
       c.encoded_output_ledger.bytes()>limits.max_encoded_host_bytes ||
       c.replay_payload_ledger.bytes()>limits.max_replay_host_bytes) {
        ++c.encoded_cache_stats.quota_refusals;
        throw Exl3ResourceReservationExhausted{};
    }
    c.encoded_cache_limits=limits;c.encoded_cache_stats.entries=c.encoded_cache.size();
}
} // namespace ninfer::exl3
