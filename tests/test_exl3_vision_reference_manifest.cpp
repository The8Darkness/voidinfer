#include "exl3/vision_reference_manifest.h"
#include <iostream>

using namespace ninfer::exl3;
template<class F> bool refuses(F&& f){try{f();return false;}catch(const std::exception&){return true;}}
void need(bool value,const char* label){if(!value)throw std::runtime_error(label);}
int main(){try{
    const std::string a(64,'a'),b(64,'b'),c(64,'c'),d(64,'d'),e(64,'e');
    std::vector<Exl3VisionReferenceArtifact> artifacts;
    const auto add=[&](int code,std::size_t rows,std::size_t columns) {
        const auto name=std::to_string(code)+".f32";
        artifacts.push_back({code,rows,columns,rows*columns,Exl3VisionReferenceScalar::fp32,
            std::filesystem::path("native")/name,std::filesystem::path("reference")/name,d,e});
    };
    for(int boundary=0;boundary<=10;++boundary)add(
        Exl3VisionBoundaryCode::block_operator(26,boundary),256,boundary==9?4352:1152);
    add(Exl3VisionBoundaryCode::block_residual(26),256,1152);
    add(Exl3VisionBoundaryCode::merger_normalized,256,1152);
    add(Exl3VisionBoundaryCode::merger_up,64,4608);
    add(Exl3VisionBoundaryCode::merger_output,64,5120);
    Exl3VisionReferenceManifest manifest{"SC_6.00bpw_H6_V6",a,b,c,
        "pinned-independent-adapter",256,64,std::move(artifacts)};
    Exl3VisionReferenceAdapter adapter(manifest);
    need(adapter.require_boundary(Exl3VisionBoundaryCode::merger_output,64,5120,
        Exl3VisionReferenceScalar::fp32).stored_bytes()==64ULL*5120*4,
        "pinned reference output contract");
    auto wrong_extent=manifest;wrong_extent.artifacts[1].stored_elements--;
    need(refuses([&]{Exl3VisionReferenceAdapter invalid(wrong_extent);}),
        "logical/stored mismatch accepted");
    auto same_path=manifest;same_path.artifacts[0].reference_file=same_path.artifacts[0].native_file;
    need(refuses([&]{Exl3VisionReferenceAdapter invalid(same_path);}),
        "native/reference artifact alias accepted");
    auto wrong_hash=manifest;wrong_hash.artifacts[0].reference_sha256="unresolved";
    need(refuses([&]{Exl3VisionReferenceAdapter invalid(wrong_hash);}),
        "artifact hash mismatch accepted");
    need(refuses([&]{(void)adapter.require_boundary(
            Exl3VisionBoundaryCode::merger_output,256,1280,Exl3VisionReferenceScalar::fp32);}),
        "equal stored element count hid wrong logical shape");
    need(refuses([&]{(void)adapter.require_boundary(9999,1,1,
            Exl3VisionReferenceScalar::fp32);}),"missing reference boundary accepted");
    std::cout<<"PASS bounded independent V6 reference manifest\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
