#include "exl3/exl3_frontend_resources.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using namespace ninfer;
int main(int argc,char** argv){try{
    if(argc!=3)throw std::invalid_argument("target-directory fixtures.json");
    auto resources=exl3::load_pinned_frontend_resources(argv[1]);
    auto frontend=targets::qwen3_6::make_frontend(resources,{.vision_enabled=false,.max_context=4096});
    std::ifstream input(argv[2]);const auto fixtures=nlohmann::ordered_json::parse(input);
    for(const auto& fixture:fixtures){
        PromptInput prompt;const auto& options=fixture.at("options");
        prompt.options.enable_thinking=options.at("enable_thinking");prompt.options.preserve_thinking=options.at("preserve_thinking");
        prompt.options.add_generation_prompt=options.at("add_generation_prompt");
        if(options.contains("reasoning_effort")){const auto effort=options.at("reasoning_effort").get<std::string>();prompt.options.reasoning_effort=effort=="low"?ReasoningEffort::Low:effort=="medium"?ReasoningEffort::Medium:ReasoningEffort::XHigh;}
        if(options.contains("tools"))for(const auto& tool:options.at("tools"))prompt.options.tool_jsons.push_back(tool.dump());
        for(const auto& row:fixture.at("messages")){
            ChatMessage message;const auto role=row.at("role").get<std::string>();
            message.role=role=="system"?ChatRole::System:role=="assistant"?ChatRole::Assistant:role=="tool"?ChatRole::Tool:ChatRole::User;
            message.parts.push_back({MessagePartKind::Text,row.at("content").get<std::string>()});
            message.reasoning_content=row.value("reasoning_content","");message.tool_call_id=row.value("tool_call_id","");
            if(row.contains("tool_calls"))for(const auto& call:row.at("tool_calls"))message.tool_calls.push_back({call.at("id").get<std::string>(),call.at("function").at("name").get<std::string>(),call.at("function").at("arguments").dump()});
            prompt.messages.push_back(std::move(message));
        }
        auto prepared=frontend.prepare(prompt);const auto& actual=targets::qwen3_6::PreparedPromptAccess::view(prepared).token_ids;
        auto expected=fixture.at("ids").get<std::vector<TokenId>>();
        if(actual!=expected){std::size_t i=0;while(i<std::min(actual.size(),expected.size()) && actual[i]==expected[i])++i;throw std::runtime_error(fixture.at("name").get<std::string>()+" token mismatch at "+std::to_string(i)+" actual="+std::to_string(actual.size())+" expected="+std::to_string(expected.size()));}
        if(frontend.count_tokens(prompt)!=expected.size())throw std::runtime_error("counted token mismatch");
        std::cout<<fixture.at("name").get<std::string>()<<" tokens="<<actual.size()<<" EXACT\n";
    }
    const auto directory=std::filesystem::path(argv[2]).parent_path();
    if(std::filesystem::exists(directory/"patch-image.png")) {
        std::ifstream image(directory/"patch-image.png",std::ios::binary);
        OwnedMedia media;media.bytes.assign(std::istreambuf_iterator<char>(image),{});media.media_type="image/png";
        for(auto storage:{VisionPatchStorage::BFloat16,VisionPatchStorage::Float16}){
            auto visual=targets::qwen3_6::make_frontend(resources,{.vision_patch_storage=storage,.vision_enabled=true,.max_context=4096});
            PromptInput prompt;prompt.options.enable_thinking=false;
            prompt.messages.push_back({ChatRole::User,{{MessagePartKind::Text,"Describe this image."},{MessagePartKind::Media,"",media}}});
            auto prepared=visual.prepare(prompt);const auto& data=targets::qwen3_6::PreparedPromptAccess::view(prepared);
            if(data.media_payloads.size()!=1 || data.vision_items[0].patch_count!=256)throw std::runtime_error("V6 patch geometry");
            const auto& payload=data.media_payloads[0];
            std::ifstream reference(directory/(storage==VisionPatchStorage::Float16?"patch-fp16.bin":"patch-bf16.bin"),std::ios::binary);
            std::vector<char> expected(std::istreambuf_iterator<char>(reference),{});
            if(payload->storage!=storage || expected.size()!=payload->span().size_bytes() || std::memcmp(expected.data(),payload->span().data(),expected.size()))throw std::runtime_error("V6 patch dtype/order mismatch");
            auto reused=visual.prepare(prompt);const auto& again=targets::qwen3_6::PreparedPromptAccess::view(reused);
            if(again.media_payloads[0]!=payload || again.token_ids!=data.token_ids || again.positions!=data.positions)throw std::runtime_error("V6 media cache identity mismatch");
            if(storage==VisionPatchStorage::Float16){if(const auto* destination=std::getenv("NINFER_TEST_MEDIA_PLAN_OUT")){
                if(std::filesystem::exists(destination))throw std::runtime_error("preserve prepared media plan");
                nlohmann::ordered_json items=nlohmann::ordered_json::array();
                for(const auto& item:data.vision_items){nlohmann::ordered_json spans=nlohmann::ordered_json::array();for(auto span:item.token_spans)spans.push_back({{"begin",span.begin},{"count",span.count}});
                    items.push_back({{"modality",static_cast<int>(item.modality)},{"grid",{item.grid.temporal,item.grid.height,item.grid.width}},{"content_digest",item.content_digest},{"token_spans",spans},{"timestamps",item.timestamps}});}
                std::ofstream(destination)<<nlohmann::ordered_json{{"token_ids",data.token_ids},{"token_types",data.token_types},{"positions_axis_major",data.positions},{"rope_delta",data.rope_delta},{"vision_items",items},{"patch_storage","fp16"},{"source_prompt","Describe this image."}}.dump(2);
            }}
            std::cout<<"patch_storage="<<(storage==VisionPatchStorage::Float16?"fp16":"bf16")<<" bytes="<<expected.size()<<" pixels_all_256=1 cache_identity=1 EXACT\n";
        }
    }
    std::cout<<"PASS_PINNED_FRONTEND_REFERENCE\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
