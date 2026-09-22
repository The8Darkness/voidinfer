#include "exl3/vision_model.h"
#include "exl3/exl3_frontend_resources.h"
#include "exl3/safetensors.h"
#include <cuda_fp16.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <set>
using namespace ninfer;
namespace v=targets::qwen3_6;
void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
float f16(std::uint16_t x){half value;std::memcpy(&value,&x,2);return __half2float(value);}
float rounded(float x){return __half2float(__float2half_rn(x));}
std::vector<std::uint16_t> tensor(const exl3::IndexedSafetensors& files,const char* name){
    for(const auto& shard:files.shards)if(shard.header.find(name)){auto p=exl3::read_tensor(shard.path,shard.header,name);auto data=p.typed<std::uint16_t>("F16");return {data.begin(),data.end()};}
    throw std::runtime_error(name);
}
int main(int argc,char** argv){try{
    if(argc!=4)throw std::invalid_argument("target-dir patch-image.png output-dir");
    std::filesystem::path output(argv[3]);std::filesystem::create_directories(output);
    auto frontend=v::make_frontend(exl3::load_pinned_frontend_resources(argv[1]),{.vision_patch_storage=VisionPatchStorage::Float16,.vision_enabled=true,.max_context=4096});
    std::ifstream image(argv[2],std::ios::binary);OwnedMedia media;media.bytes.assign(std::istreambuf_iterator<char>(image),{});media.media_type="image/png";
    PromptInput prompt;prompt.options.enable_thinking=false;prompt.messages.push_back({ChatRole::User,{{MessagePartKind::Text,"Describe this image."},{MessagePartKind::Media,"",media}}});
    auto prepared=frontend.prepare(prompt);auto& data=v::PreparedPromptAccess::view(prepared);
    auto controls=v::build_vision_control(data,v::plan_vision_control(data),0);require(controls.items.size()==1,"one control");
    const auto& control=controls.items[0];const auto& payload=*data.media_payloads[0];require(control.patch_count==256,"fixture patches");
    std::ofstream(output/"control.json")<<nlohmann::json({{"rows",control.patch_count},{"position_ids",control.position_ids},{"table_indices",control.position_table_indices},{"table_weights",control.position_table_weights}}).dump();
    {std::ofstream patches_file(output/"patches-fp16.bin",std::ios::binary);patches_file.write(reinterpret_cast<const char*>(payload.span().data()),payload.span().size_bytes());}
    auto files=exl3::inspect_indexed_directory(argv[1]);auto weights=tensor(files,"model.visual.patch_embed.proj.weight"),bias=tensor(files,"model.visual.patch_embed.proj.bias"),positions=tensor(files,"model.visual.pos_embed.weight");
    auto model=std::make_unique<exl3::Exl3VisionModel>(argv[1]);exl3::Exl3VisionContext context(*model,512);
    std::vector<float> patch_result,projection_result;std::vector<nlohmann::json> layers;
    std::set<int> held_boundaries;
    auto observe=[&](int layer,std::span<const float> values){double squared=0;float peak=0;for(float x:values){require(std::isfinite(x),"nonfinite layer");squared+=double(x)*x;peak=std::max(peak,std::abs(x));}
        held_boundaries.insert(layer);
        if(layer==exl3::Exl3VisionBoundaryCode::merger_normalized)require(values.size()==control.patch_count*1152,"merger normalized held extent");
        if(layer==exl3::Exl3VisionBoundaryCode::merger_up)require(values.size()==control.merged_count*4608,"merger up held extent");
        if(layer==exl3::Exl3VisionBoundaryCode::merger_output)require(values.size()==control.merged_count*5120,"merger output held extent");
        layers.push_back({{"layer",layer},{"rms",std::sqrt(squared/values.size())},{"max_abs",peak}});if(layer==-1)patch_result.assign(values.begin(),values.end());if(layer==-2)projection_result.assign(values.begin(),values.end());
        std::ofstream trace(output/("layer-"+std::to_string(layer)+".f32"),std::ios::binary);trace.write(reinterpret_cast<const char*>(values.data()),values.size_bytes());};
    auto first=context.encode_numeric_candidate(payload,control,{},observe);
    if(const auto* audit=std::getenv("NINFER_V6_OPERATOR_AUDIT");audit&&std::string_view(audit)=="1") {
        for(int boundary=0;boundary<=10;++boundary)
            require(held_boundaries.contains(exl3::Exl3VisionBoundaryCode::block_operator(26,boundary)),
                "block26 held boundary missing");
        require(held_boundaries.contains(exl3::Exl3VisionBoundaryCode::block_residual(26)) &&
                held_boundaries.contains(exl3::Exl3VisionBoundaryCode::merger_normalized) &&
                held_boundaries.contains(exl3::Exl3VisionBoundaryCode::merger_up) &&
                held_boundaries.contains(exl3::Exl3VisionBoundaryCode::merger_output),
            "late V6/merger discriminating boundary missing");
    }
    // Distinct FP64 dot product on actual prepared inputs and all1152 outputs
    // at8 spatial locations. Interpolation retains the declared half cast order.
    double error2=0,reference2=0,max_error=0,projection_error2=0,projection_ref2=0,projection_max=0;std::size_t samples=0;
    for(int row:{0,1,2,3,63,127,191,255})for(int c=0;c<1152;++c){double dot=0;
        for(int k=0;k<1536;++k)dot+=double(f16(payload.span()[row*1536+k]))*f16(weights[c*1536+k]);
        double pr=float(float(dot)+f16(bias[c])),pe=projection_result[row*1152+c]-pr;projection_error2+=pe*pe;projection_ref2+=pr*pr;projection_max=std::max(projection_max,std::abs(pe));
        float emb=rounded(f16(positions[control.position_table_indices[row*4]*1152+c])*rounded(control.position_table_weights[row*4]));
        for(int j=1;j<4;++j)emb=rounded(emb+rounded(f16(positions[control.position_table_indices[row*4+j]*1152+c])*rounded(control.position_table_weights[row*4+j])));
        double ref=double(float(float(dot)+f16(bias[c]))+emb),err=patch_result[row*1152+c]-ref;error2+=err*err;reference2+=ref*ref;max_error=std::max(max_error,std::abs(err));++samples;
    }
    double relative=std::sqrt(error2/reference2);
    nlohmann::json patch_gate={{"relative_rmse",relative},{"max_abs_error",max_error},{"projection_relative_rmse",std::sqrt(projection_error2/projection_ref2)},{"projection_max_abs_error",projection_max},{"samples",samples},{"layers",layers}};
    std::ofstream(output/"patch-gate.json")<<patch_gate.dump(2);std::cout<<patch_gate.dump(2)<<std::endl;
    require(relative<1e-5&&max_error<0.005,"patch FP64 operator gate");
    auto second=context.encode_numeric_candidate(payload,control);require(first.embeddings==second.embeddings,"repeat not bitwise deterministic");
    int checks=0;bool cancelled=false;try{context.encode_numeric_candidate(payload,control,[&]{return ++checks==4;});}catch(const std::runtime_error& e){cancelled=std::string(e.what())=="V6 cancelled";}
    require(cancelled,"mid-tower cancellation");auto after=context.encode_numeric_candidate(payload,control);require(after.embeddings==first.embeddings,"cancel polluted context");
    v::PreparedMediaPayload changed;changed.storage=VisionPatchStorage::Float16;
    changed.preprocess=payload.preprocess;changed.patch_elements=payload.patch_elements;changed.patches=std::make_unique<std::uint16_t[]>(changed.patch_elements);
    std::copy(payload.span().begin(),payload.span().end(),changed.patches.get());for(std::size_t i=0;i<changed.patch_elements;++i)changed.patches[i]^=0x8000;
    auto different=context.encode_numeric_candidate(changed,control);double difference2=0;
    for(std::size_t i=0;i<first.embeddings.size();++i){require(std::isfinite(first.embeddings[i])&&std::isfinite(different.embeddings[i]),"nonfinite merger");double d=first.embeddings[i]-different.embeddings[i];difference2+=d*d;}
    require(difference2>1,"changed patches ignored");
    // Two temporal segments with distinct pixels must reproduce their separate
    // image embeddings. Frame2 must never influence frame1's noncausal attention.
    v::PreparedMediaPayload video;video.storage=VisionPatchStorage::Float16;
    video.preprocess=payload.preprocess;video.preprocess.video_fps=2.0;
    video.preprocess.video_min_frames=4;video.preprocess.video_max_frames=768;
    video.patch_elements=payload.patch_elements*2;video.patches=std::make_unique<std::uint16_t[]>(video.patch_elements);
    std::copy(payload.span().begin(),payload.span().end(),video.patches.get());std::copy(changed.span().begin(),changed.span().end(),video.patches.get()+payload.patch_elements);
    auto vc=control;vc.modality=v::PromptModality::Video;vc.grid.temporal=2;vc.patch_count*=2;vc.merged_count*=2;vc.segment_count=2;
    vc.position_ids.clear();for(int axis=0;axis<2;++axis)for(int frame=0;frame<2;++frame)vc.position_ids.insert(vc.position_ids.end(),control.position_ids.begin()+axis*256,control.position_ids.begin()+(axis+1)*256);
    vc.position_table_indices.insert(vc.position_table_indices.end(),control.position_table_indices.begin(),control.position_table_indices.end());vc.position_table_weights.insert(vc.position_table_weights.end(),control.position_table_weights.begin(),control.position_table_weights.end());
    auto frames=context.encode_numeric_candidate(video,vc);double video_error2=0,video_ref2=0;float video_max=0;
    for(std::size_t i=0;i<frames.embeddings.size();++i){float ref=i<first.embeddings.size()?first.embeddings[i]:different.embeddings[i-first.embeddings.size()];float e=frames.embeddings[i]-ref;require(std::isfinite(e),"video nonfinite");video_error2+=double(e)*e;video_ref2+=double(ref)*ref;video_max=std::max(video_max,std::abs(e));}
    double video_relative=std::sqrt(video_error2/video_ref2);
    nlohmann::json video_gate={{"relative_rmse",video_relative},{"max_abs_error",video_max},{"repeat_exact",true},{"cancel_recycle_exact",true},{"changed_embedding_l2",std::sqrt(difference2)}};
    std::ofstream(output/"video-gate.json")<<video_gate.dump(2);std::cout<<video_gate.dump(2)<<std::endl;
    require(video_relative<0.002&&video_max<0.25,"separate video segment numeric gate");
    // Context retains immutable weights after the public model object retires.
    model.reset();auto survived=context.encode_numeric_candidate(payload,control);require(survived.embeddings==first.embeddings,"model survivor ownership");
    changed.storage=VisionPatchStorage::BFloat16;bool refused=false;try{context.encode_numeric_candidate(changed,control);}catch(const std::exception&){refused=true;}require(refused,"BF16 silently accepted");
    std::ofstream binary(output/"merged-fp32.bin",std::ios::binary);binary.write(reinterpret_cast<const char*>(first.embeddings.data()),first.embeddings.size()*4);
    nlohmann::json result={{"status","PASS_V6_NUMERIC_BRINGUP"},{"full_model_quality","UNMEASURED"},{"engine_media_enabled",false},{"model_device_bytes",first.stats.model_device_bytes},{"context_device_bytes",first.stats.context_device_bytes},
        {"patches",first.stats.patches},{"merged_tokens",first.stats.merged_tokens},{"packed_projections",first.stats.packed_projections},{"cold_trace_wall_ms",first.stats.wall_ms},{"warm_wall_ms",second.stats.wall_ms},
        {"patch_fp64_samples",samples},{"patch_relative_rmse",relative},{"patch_max_abs_error",max_error},{"changed_embedding_l2",std::sqrt(difference2)},{"video_relative_rmse",video_relative},{"video_max_abs_error",video_max},
        {"repeat_exact",true},{"cancel_recycle_exact",true},{"model_retirement_exact",true},{"bf16_refused",true},{"layers",layers}};
    std::ofstream(output/"result.json")<<result.dump(2);std::cout<<result.dump(2)<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
