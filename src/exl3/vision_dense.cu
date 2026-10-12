#include "exl3/vision_dense.h"
#include "exl3/safetensors.h"
#include "core/tensor.h"
#include "voidinfer/ops/add_bias.h"
#include "voidinfer/ops/gelu.h"
#include "voidinfer/ops/layer_norm.h"
#include "voidinfer/ops/residual_add.h"
#include "voidinfer/ops/rope.h"
#include "voidinfer/ops/softmax_attention.h"
#include "voidinfer/ops/vision_pos_embed.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace voidinfer::exl3 {
namespace {
constexpr int H=1152,Heads=16,D=72,Inter=4304,Patch=1536,Merge=4*H,Out=5120,Positions=2304,Depth=27;
constexpr float NormEps=1e-6F,RopeTheta=10000.0F,Scale=0.11785113019775792F;

void require(bool ok,const std::string& what){if(!ok)throw std::runtime_error("Vision: "+what);}
void check(cudaError_t value,const char* what){
    require(value==cudaSuccess,std::string(what)+": "+cudaGetErrorString(value));
}
void blas(cublasStatus_t value,const char* what){require(value==CUBLAS_STATUS_SUCCESS,what);}

__global__ void bf16_to_half_kernel(const __nv_bfloat16* in,__half* out,std::size_t count){
    const std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)out[i]=__float2half_rn(__bfloat162float(in[i]));
}

// Streamed units: the patch stem, one per block, and the merger in two parts
// (the largest unit bounds each of the two device slots).
struct Slice {std::size_t offset=0,elements=0;};
struct Unit {std::size_t offset=0,bytes=0;};
struct BlockSlices {Slice norm1_w,norm1_b,qkv_w,qkv_b,proj_w,proj_b,norm2_w,norm2_b,fc1_w,fc1_b,fc2_w,fc2_b;};
} // namespace

struct Exl3DenseVision::Impl {
    void* host=nullptr;std::size_t host_bytes=0;
    std::vector<Unit> units;
    Slice patch_w,patch_b,pos;
    std::array<BlockSlices,Depth> blocks{};
    Slice merger_norm_w,merger_norm_b,merger_fc1_w,merger_fc1_b,merger_fc2_w,merger_fc2_b;
    std::size_t slot_bytes=0;
    cudaStream_t copy_stream=nullptr;
    std::array<cudaEvent_t,2> loaded{},consumed{};
    cublasHandle_t handle=nullptr;
    std::mutex mutex;

    explicit Impl(const std::filesystem::path& directory) {
        {
            std::ifstream config(directory/"config.json");
            require(config.good(),"missing config.json in "+directory.string());
            const auto json=nlohmann::json::parse(config);const auto& v=json.at("vision_config");
            require(v.at("depth")==Depth && v.at("hidden_size")==H && v.at("num_heads")==Heads &&
                v.at("intermediate_size")==Inter && v.at("out_hidden_size")==Out &&
                v.at("patch_size")==16 && v.at("temporal_patch_size")==2 &&
                v.at("spatial_merge_size")==2 && v.at("num_position_embeddings")==Positions &&
                v.at("hidden_act")=="gelu_pytorch_tanh" && v.at("deepstack_visual_indexes").empty(),
                "unsupported vision_config");
        }
        // Only the shards named for `model.visual.*` by the index weight map are
        // opened (one shard in the BF16 checkpoint); a single model.safetensors
        // also works.
        struct Shard {std::filesystem::path path;SafetensorsHeader header;};
        std::vector<Shard> shards;
        {
            std::vector<std::string> names;
            std::ifstream index(directory/"model.safetensors.index.json");
            if(index.good()) {
                const auto json=nlohmann::json::parse(index);
                for(const auto& [name,file]:json.at("weight_map").items())
                    if(name.rfind("model.visual.",0)==0) {
                        const auto shard=file.get<std::string>();
                        if(std::find(names.begin(),names.end(),shard)==names.end())names.push_back(shard);
                    }
            } else names.push_back("model.safetensors");
            for(const auto& name:names)shards.push_back({directory/name,inspect_file(directory/name)});
        }
        struct Wanted {std::string name;std::vector<std::uint64_t> shape;Slice* slice;};
        std::vector<std::vector<Wanted>> plan;
        const auto vec=[](std::initializer_list<std::uint64_t> v){return std::vector<std::uint64_t>(v);};
        const std::string root="model.visual.";
        plan.push_back({{root+"patch_embed.proj.weight",vec({H,3,2,16,16}),&patch_w},
                        {root+"patch_embed.proj.bias",vec({H}),&patch_b},
                        {root+"pos_embed.weight",vec({Positions,H}),&pos}});
        for(int i=0;i<Depth;++i) {
            const auto b=root+"blocks."+std::to_string(i)+".";auto& s=blocks[i];
            plan.push_back({{b+"norm1.weight",vec({H}),&s.norm1_w},{b+"norm1.bias",vec({H}),&s.norm1_b},
                {b+"attn.qkv.weight",vec({3*H,H}),&s.qkv_w},{b+"attn.qkv.bias",vec({3*H}),&s.qkv_b},
                {b+"attn.proj.weight",vec({H,H}),&s.proj_w},{b+"attn.proj.bias",vec({H}),&s.proj_b},
                {b+"norm2.weight",vec({H}),&s.norm2_w},{b+"norm2.bias",vec({H}),&s.norm2_b},
                {b+"mlp.linear_fc1.weight",vec({Inter,H}),&s.fc1_w},{b+"mlp.linear_fc1.bias",vec({Inter}),&s.fc1_b},
                {b+"mlp.linear_fc2.weight",vec({H,Inter}),&s.fc2_w},{b+"mlp.linear_fc2.bias",vec({H}),&s.fc2_b}});
        }
        plan.push_back({{root+"merger.norm.weight",vec({H}),&merger_norm_w},
                        {root+"merger.norm.bias",vec({H}),&merger_norm_b},
                        {root+"merger.linear_fc1.weight",vec({Merge,Merge}),&merger_fc1_w},
                        {root+"merger.linear_fc1.bias",vec({Merge}),&merger_fc1_b}});
        plan.push_back({{root+"merger.linear_fc2.weight",vec({Out,Merge}),&merger_fc2_w},
                        {root+"merger.linear_fc2.bias",vec({Out}),&merger_fc2_b}});
        // Unit-relative offsets (256-byte aligned), then one pinned allocation.
        const auto align=[](std::size_t value){return (value+255)&~std::size_t(255);};
        const auto locate=[&](const std::string& name)->const Shard& {
            for(const auto& shard:shards)if(shard.header.find(name))return shard;
            throw std::runtime_error("Vision: missing tensor "+name);
        };
        std::size_t total=0;
        for(auto& group:plan) {
            Unit unit{total,0};
            for(auto& w:group) {
                const auto* info=locate(w.name).header.find(w.name);
                require(info->dtype=="BF16" && info->shape==w.shape,"tensor dtype/extent "+w.name);
                w.slice->offset=unit.bytes;w.slice->elements=info->bytes()/2;
                unit.bytes=align(unit.bytes+info->bytes());
            }
            units.push_back(unit);total+=unit.bytes;slot_bytes=std::max(slot_bytes,unit.bytes);
        }
        check(cudaHostAlloc(&host,total,cudaHostAllocDefault),"pin vision weights");
        host_bytes=total;
        try {
            for(std::size_t u=0;u<plan.size();++u)
                for(const auto& w:plan[u]) {
                    const auto& shard=locate(w.name);
                    const auto payload=read_tensor(shard.path,shard.header,w.name);
                    std::memcpy(static_cast<std::byte*>(host)+units[u].offset+w.slice->offset,
                        payload.bytes().data(),payload.bytes().size());
                }
            check(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking),"vision copy stream");
            for(auto* events:{&loaded,&consumed})
                for(auto& event:*events)check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"vision event");
            blas(cublasCreate(&handle),"vision cuBLAS handle");
        } catch(...) {release();throw;}
    }
    void release() noexcept {
        if(handle)cublasDestroy(handle);
        for(auto* events:{&loaded,&consumed})for(auto& event:*events)if(event)cudaEventDestroy(event);
        if(copy_stream)cudaStreamDestroy(copy_stream);
        if(host)cudaFreeHost(host);
        handle=nullptr;copy_stream=nullptr;host=nullptr;
    }
    ~Impl(){release();}

    // y[N,T] = W[N,K] x[K,T] (+ bias), BF16 in/out, FP32 accumulation.
    void linear(const __nv_bfloat16* w,const __nv_bfloat16* x,__nv_bfloat16* y,int n,int k,int t,
                cudaStream_t stream) {
        const float one=1.0F,zero=0.0F;
        blas(cublasSetStream(handle,stream),"vision cuBLAS stream");
        blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,t,k,&one,w,CUDA_R_16BF,k,x,CUDA_R_16BF,k,
            &zero,y,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT),"vision projection");
    }
};

Exl3DenseVision::Exl3DenseVision(const std::filesystem::path& directory)
    :impl_(std::make_unique<Impl>(directory)) {}
Exl3DenseVision::~Exl3DenseVision()=default;
std::size_t Exl3DenseVision::host_bytes() const noexcept {return impl_->host_bytes;}

Exl3DenseVisionStats Exl3DenseVision::encode(const targets::qwen3_6::PreparedMediaPayload& payload,
    const targets::qwen3_6::VisionItemControl& control,std::uint16_t* output,cudaStream_t stream) {
    auto& m=*impl_;
    const std::lock_guard lock(m.mutex);
    const auto start=std::chrono::steady_clock::now();
    using voidinfer::VisionPatchStorage;
    require(payload.storage==VisionPatchStorage::BFloat16,"BF16 patches required");
    const auto rows64=control.patch_count;
    require(rows64>0 && rows64%4==0 && rows64<=131072 && control.merged_count==rows64/4,"patch extent");
    require(payload.patch_elements==rows64*Patch,"payload extent");
    require(control.segment_length>0 && rows64%static_cast<std::size_t>(control.segment_length)==0,
        "segment extent");
    require(control.position_ids.size()==rows64*2 && control.position_table_indices.size()==rows64*4 &&
        control.position_table_weights.size()==rows64*4,"position control extent");
    const int rows=static_cast<int>(rows64),merged=rows/4;
    // Activations: residual x, normalized h, attention output a, and one wide
    // region shared by the patches, QKV, the MLP up projection and the merger.
    const auto bf=[](std::size_t elements){return (elements*sizeof(__nv_bfloat16)+255)&~std::size_t(255);};
    const std::size_t wide=std::max({static_cast<std::size_t>(Patch),static_cast<std::size_t>(3*H),
        static_cast<std::size_t>(Inter)})*rows64;
    const std::size_t x_bytes=bf(rows64*H),wide_bytes=bf(wide),merged_bytes=bf(static_cast<std::size_t>(merged)*Out);
    const std::size_t int_bytes=(rows64*(2+4)*sizeof(int)+255)&~std::size_t(255);
    const std::size_t scratch=2*m.slot_bytes+3*x_bytes+wide_bytes+merged_bytes+int_bytes+
        ((rows64*4*sizeof(float)+255)&~std::size_t(255));
    std::byte* base=nullptr;
    check(cudaMallocAsync(reinterpret_cast<void**>(&base),scratch,stream),"allocate vision scratch");
    struct Free {std::byte* p;cudaStream_t s;~Free(){if(p)(void)cudaFreeAsync(p,s);}} free_scratch{base,stream};
    std::byte* cursor=base;
    const auto take=[&](std::size_t bytes){auto* p=cursor;cursor+=bytes;return p;};
    std::array<std::byte*,2> slot{take(m.slot_bytes),take(m.slot_bytes)};
    auto* x=reinterpret_cast<__nv_bfloat16*>(take(x_bytes));
    auto* h=reinterpret_cast<__nv_bfloat16*>(take(x_bytes));
    auto* a=reinterpret_cast<__nv_bfloat16*>(take(x_bytes));
    auto* w=reinterpret_cast<__nv_bfloat16*>(take(wide_bytes));
    auto* out_bf16=reinterpret_cast<__nv_bfloat16*>(take(merged_bytes));
    auto* position_ids=reinterpret_cast<int*>(take(int_bytes));
    auto* table_indices=position_ids+rows64*2;
    auto* table_weights=reinterpret_cast<float*>(cursor);

    // Weight streaming: unit u lands in slot u%2 after the compute stream has
    // released the slot's previous unit.
    const auto unit_count=m.units.size();
    const auto issue=[&](std::size_t u) {
        const auto s=u%2;
        check(cudaStreamWaitEvent(m.copy_stream,m.consumed[s],0),"vision slot reuse");
        check(cudaMemcpyAsync(slot[s],static_cast<const std::byte*>(m.host)+m.units[u].offset,
            m.units[u].bytes,cudaMemcpyHostToDevice,m.copy_stream),"stream vision weights");
        check(cudaEventRecord(m.loaded[s],m.copy_stream),"vision weights loaded");
    };
    // Slots start free once the scratch allocation is ordered on `stream`.
    for(auto& event:m.consumed)check(cudaEventRecord(event,stream),"vision slot free");
    issue(0);issue(1);
    std::size_t next=2;
    const auto acquire=[&](std::size_t u){
        check(cudaStreamWaitEvent(stream,m.loaded[u%2],0),"vision weights ready");
        return slot[u%2];
    };
    const auto release=[&](std::size_t u){
        check(cudaEventRecord(m.consumed[u%2],stream),"vision slot consumed");
        if(next<unit_count)issue(next++);
    };
    const auto weight=[](std::byte* s,const Slice& slice){
        return reinterpret_cast<__nv_bfloat16*>(s+slice.offset);
    };
    const auto tensor=[](const void* p,DType type,std::initializer_list<std::int32_t> shape){
        return Tensor(const_cast<void*>(p),type,shape);
    };

    check(cudaMemcpyAsync(w,payload.span().data(),payload.span().size_bytes(),cudaMemcpyHostToDevice,stream),
        "upload vision patches");
    check(cudaMemcpyAsync(position_ids,control.position_ids.data(),rows64*2*sizeof(int),
        cudaMemcpyHostToDevice,stream),"upload vision positions");
    check(cudaMemcpyAsync(table_indices,control.position_table_indices.data(),rows64*4*sizeof(int),
        cudaMemcpyHostToDevice,stream),"upload vision table indices");
    check(cudaMemcpyAsync(table_weights,control.position_table_weights.data(),rows64*4*sizeof(float),
        cudaMemcpyHostToDevice,stream),"upload vision table weights");

    Tensor tx=tensor(x,DType::BF16,{H,rows}),th=tensor(h,DType::BF16,{H,rows}),ta=tensor(a,DType::BF16,{H,rows});
    {
        auto* s=acquire(0);
        m.linear(weight(s,m.patch_w),w,x,H,Patch,rows,stream);
        ops::add_bias(tensor(weight(s,m.patch_b),DType::BF16,{H}),tx,stream);
        ops::vision_pos_embed_add(tensor(weight(s,m.pos),DType::BF16,{H,Positions}),
            tensor(table_indices,DType::I32,{4,rows}),tensor(table_weights,DType::FP32,{4,rows}),tx,stream);
        release(0);
    }
    const Tensor positions=tensor(position_ids,DType::I32,{rows,2});
    for(int i=0;i<Depth;++i) {
        const std::size_t u=1+static_cast<std::size_t>(i);
        auto* s=acquire(u);const auto& b=m.blocks[i];
        ops::layer_norm(tx,tensor(weight(s,b.norm1_w),DType::BF16,{H}),tensor(weight(s,b.norm1_b),DType::BF16,{H}),
            NormEps,th,stream);
        m.linear(weight(s,b.qkv_w),h,w,3*H,H,rows,stream);
        Tensor qkv=tensor(w,DType::BF16,{3*H,rows});
        ops::add_bias(tensor(weight(s,b.qkv_b),DType::BF16,{3*H}),qkv,stream);
        const std::size_t plane=static_cast<std::size_t>(H)*sizeof(__nv_bfloat16);
        Tensor q(w,DType::BF16,{D,Heads,rows});
        Tensor k(reinterpret_cast<std::byte*>(w)+plane,DType::BF16,{D,Heads,rows});
        Tensor v(reinterpret_cast<std::byte*>(w)+2*plane,DType::BF16,{D,Heads,rows});
        q.nb[2]=qkv.nb[1];k.nb[2]=qkv.nb[1];v.nb[2]=qkv.nb[1];
        ops::rope(positions,D,RopeTheta,q,k,stream);
        Tensor heads=ta.view({D,Heads,rows});
        ops::packed_softmax_attention(q,k,v,{D,Heads,Heads},Scale,control.segment_length,heads,stream);
        m.linear(weight(s,b.proj_w),a,h,H,H,rows,stream);
        ops::add_bias(tensor(weight(s,b.proj_b),DType::BF16,{H}),th,stream);
        ops::residual_add(th,tx,stream);
        ops::layer_norm(tx,tensor(weight(s,b.norm2_w),DType::BF16,{H}),tensor(weight(s,b.norm2_b),DType::BF16,{H}),
            NormEps,th,stream);
        m.linear(weight(s,b.fc1_w),h,w,Inter,H,rows,stream);
        Tensor up=tensor(w,DType::BF16,{Inter,rows});
        ops::add_bias(tensor(weight(s,b.fc1_b),DType::BF16,{Inter}),up,stream);
        ops::gelu(up,ops::GeluMode::Tanh,stream);
        m.linear(weight(s,b.fc2_w),w,a,H,Inter,rows,stream);
        ops::add_bias(tensor(weight(s,b.fc2_b),DType::BF16,{H}),ta,stream);
        ops::residual_add(ta,tx,stream);
        release(u);
    }
    {
        const std::size_t u=1+Depth;
        auto* s=acquire(u);
        ops::layer_norm(tx,tensor(weight(s,m.merger_norm_w),DType::BF16,{H}),
            tensor(weight(s,m.merger_norm_b),DType::BF16,{H}),NormEps,th,stream);
        // Four consecutive patches form one merged token ([4*H, V] view).
        m.linear(weight(s,m.merger_fc1_w),h,w,Merge,Merge,merged,stream);
        Tensor hidden=tensor(w,DType::BF16,{Merge,merged});
        ops::add_bias(tensor(weight(s,m.merger_fc1_b),DType::BF16,{Merge}),hidden,stream);
        ops::gelu(hidden,ops::GeluMode::Exact,stream);
        release(u);
    }
    {
        const std::size_t u=2+Depth;
        auto* s=acquire(u);
        m.linear(weight(s,m.merger_fc2_w),w,out_bf16,Out,Merge,merged,stream);
        Tensor out=tensor(out_bf16,DType::BF16,{Out,merged});
        ops::add_bias(tensor(weight(s,m.merger_fc2_b),DType::BF16,{Out}),out,stream);
        release(u);
    }
    const auto count=static_cast<std::size_t>(merged)*Out;
    bf16_to_half_kernel<<<static_cast<unsigned>((count+255)/256),256,0,stream>>>(
        out_bf16,reinterpret_cast<__half*>(output),count);
    check(cudaGetLastError(),"convert vision output");
    check(cudaStreamSynchronize(stream),"vision encode fence");
    Exl3DenseVisionStats stats;
    stats.patches=rows64;stats.merged_tokens=static_cast<std::size_t>(merged);stats.scratch_device_bytes=scratch;
    stats.encode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return stats;
}

void Exl3DenseVision::warm_up(cudaStream_t stream) {
    constexpr int side=32,rows=side*side;
    targets::qwen3_6::PreparedMediaPayload payload;
    payload.storage=voidinfer::VisionPatchStorage::BFloat16;
    payload.patch_elements=static_cast<std::size_t>(rows)*Patch;
    payload.patches=std::make_unique<std::uint16_t[]>(payload.patch_elements);
    std::fill_n(payload.patches.get(),payload.patch_elements,std::uint16_t{0x3f00});
    targets::qwen3_6::VisionItemControl control;
    control.grid={1,side,side};control.patch_count=rows;control.merged_count=rows/4;
    control.segment_length=rows;control.segment_count=1;
    control.position_ids.assign(static_cast<std::size_t>(rows)*2,0);
    for(int i=0;i<rows;++i){control.position_ids[i]=i/side;control.position_ids[rows+i]=i%side;}
    control.position_table_indices.assign(static_cast<std::size_t>(rows)*4,0);
    control.position_table_weights.assign(static_cast<std::size_t>(rows)*4,0.25F);
    std::uint16_t* output=nullptr;
    check(cudaMallocAsync(reinterpret_cast<void**>(&output),
        static_cast<std::size_t>(rows/4)*Out*sizeof(std::uint16_t),stream),"vision warm-up output");
    struct Free {std::uint16_t* p;cudaStream_t s;~Free(){(void)cudaFreeAsync(p,s);(void)cudaStreamSynchronize(s);}} free_output{output,stream};
    (void)encode(payload,control,output,stream);
}

} // namespace voidinfer::exl3
