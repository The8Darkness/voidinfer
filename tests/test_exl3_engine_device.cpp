#include <ninfer/engine.h>
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "exl3/exl3_frontend_resources.h"
#include "exl3/dflash2_draft.h"
#include "exl3/fast_device_round.h"
#include "exl3/branch_reference.h"
#include "exl3/linear_cuda.h"
#include "exl3/text_model.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

static void verify_engine_linked_graph_launch() {
    DeviceContext device(0);
    DeviceArena storage(sizeof(std::uint32_t));
    CUDA_CHECK(cudaMemsetAsync(storage.base(),0x13,sizeof(std::uint32_t),
        device.stream));
    DecodeGraphDefinition definition;
    definition.capture(device.stream,[&] {
        CUDA_CHECK(cudaMemsetAsync(storage.base(),0x5a,
            sizeof(std::uint32_t),device.stream));
    });
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    executable.upload(device.stream);
    device.synchronize();
    std::uint32_t observed=0;
    CUDA_CHECK(cudaMemcpy(&observed,storage.base(),sizeof(observed),
        cudaMemcpyDeviceToHost));
    if(observed!=0x13131313U)
        throw std::runtime_error("Engine-linked graph upload executed work");
    executable.launch(device.stream);
    device.synchronize();
    CUDA_CHECK(cudaMemcpy(&observed,storage.base(),sizeof(observed),
        cudaMemcpyDeviceToHost));
    if(observed!=0x5a5a5a5aU)
        throw std::runtime_error("Engine-linked graph launch did not execute work");
    std::cout<<"COHERENT_DEVICE_GRAPH_LAUNCH_PASS\n";
}

static void direct_target_only_probe(const char* path,const char* target,
    int max_context) {
    std::ifstream input(path);
    std::vector<std::int64_t> ids;
    std::int64_t id=0;
    while(input>>id)ids.push_back(id);
    if(ids.size()<2048 || ids.size()>static_cast<std::size_t>(max_context-128))
        throw std::runtime_error("direct target-only probe prompt extent");
    auto model=ninfer::exl3::Exl3TextModel::load(target,max_context);
    auto context=model->create_context(true);
    const auto initial=static_cast<std::size_t>(
        ninfer::exl3::Exl3TextContext::layer_major_initial_rows());
    if(initial)context->prefill(std::span<const std::int64_t>(ids).first(initial));
    context->append_prefill_layer_major(
        std::span<const std::int64_t>(ids).subspan(initial));
    if(cudaStreamSynchronize(nullptr)!=cudaSuccess)
        throw std::runtime_error("direct target-only prefill completion");
    const auto root=context->export_exact_host_state();
    const auto first=ninfer::exl3::exl3_branch_greedy(*context);
    context->decode(first);
    const auto second=ninfer::exl3::exl3_branch_greedy(*context);
    std::cout<<"COHERENT_DEVICE_DIRECT_TARGET_ONLY hash="
             <<root->represented_payload_hash_for_test()
             <<" pair="<<first<<','<<second<<'\n';
}

static void direct_binary_probe(const char* path,const char* target,
    const char* draft,int max_context,const char* label) {
    std::ifstream input(path);
    std::vector<std::int64_t> ids;
    std::int64_t id=0;
    while(input>>id)ids.push_back(id);
    if(ids.size()<2048 || ids.size()>static_cast<std::size_t>(max_context-128))
        throw std::runtime_error("direct Engine-binary probe prompt extent");
    auto direct_target=ninfer::exl3::Exl3TextModel::load(target,max_context);
    auto direct_draft=ninfer::exl3::Exl3Dflash2DraftModel::load(draft);
    std::shared_ptr<ninfer::exl3::Exl3TextContext> context(
        direct_target->create_context(true,false,true));
    context->prepare_continuation(8);
    context->prepare_transaction();
    context->bind_request_compatibility("direct-engine-binary-probe");
    context->reset_for_request("direct-engine-binary-probe");
    struct Stage {
        std::array<std::uint16_t*,5> planes{};
        ~Stage(){for(auto* plane:planes)if(plane)cudaFree(plane);}
    } stage;
    for(auto& plane:stage.planes)
        if(cudaMalloc(reinterpret_cast<void**>(&plane),16ULL*5120*2)!=
            cudaSuccess)
            throw std::runtime_error("direct Engine-binary staging allocation");
    ninfer::exl3::Exl3FastDeviceRound round(context,*direct_draft,
        stage.planes,1,context->request_generation());
    round.begin_fresh(ids,true);
    const auto direct_root=context->export_exact_host_state();
    const auto direct_first=ninfer::exl3::exl3_branch_greedy(*context);
    context->decode(direct_first);
    const auto direct_second=ninfer::exl3::exl3_branch_greedy(*context);
    const auto logits=context->logits_host();
    const auto host_second=static_cast<std::int64_t>(
        std::max_element(logits.begin(),logits.end())-logits.begin());
    std::cout<<label<<" hash="<<direct_root->represented_payload_hash_for_test()
             <<" pair="<<direct_first<<','<<direct_second
             <<" host_second="<<host_second<<'\n';
}

int main() {
    try {
        const auto* target=std::getenv("NINFER_EXL3_TARGET_PATH");
        const auto* draft=std::getenv("NINFER_EXL3_DFLASH2_PATH");
        const auto* prefix_text=std::getenv("NINFER_TEST_ENGINE_DEVICE_PREFIX");
        if(!target || !draft || !prefix_text)return 77;
        const int prefix=std::stoi(prefix_text);
        const char* l0_flag=std::getenv("NINFER_EXL3_L0_OSCAR");
        const bool l0_long=l0_flag && std::string_view(l0_flag)=="1" && prefix>16384 && prefix<=260000;
        if(prefix!=4096 && prefix!=16384 && !l0_long)
            throw std::invalid_argument("device Engine witness prefix must be 4K or 16K (L0 OSCAR: up to 260000)");
        EngineOptions options;
        options.exl3_package=PinnedExl3PackageOptions{target,draft,{}};
        options.exl3_package->round_implementation=
            Exl3RoundImplementation::CoherentDevice;
        int long_outputs=128;
        if(const auto* output_option=std::getenv(
               "NINFER_TEST_ENGINE_DEVICE_OUTPUT_TOKENS")) {
            long_outputs=std::stoi(output_option);
            if(long_outputs<128 || long_outputs>4096)
                throw std::invalid_argument(
                    "device Engine long output budget must be 128..4096");
        }
        options.max_context=prefix+long_outputs+128;
        options.kv_capacity=KvCapacityPolicy::explicit_capacity(options.max_context);
        options.max_concurrency=1;
        options.enable_vision=false;
        options.use_cuda_graph=false;
        options.speculative={SpeculativeBackend::DFlash2,7,ProposalHead::Full};
        options.context_cache.enabled=true;
        options.context_cache.max_shared_prefixes=1;
        // L0 OSCAR long witnesses: one terminal exact host snapshot of the full context.
        if(l0_long)options.context_cache.host_kv_capacity_bytes=std::max<std::size_t>(
            options.context_cache.host_kv_capacity_bytes,
            static_cast<std::size_t>(options.max_context)*65536);
        options.max_pending_requests=2;
        verify_engine_linked_graph_launch();
        if(const auto* path=std::getenv("NINFER_TEST_ENGINE_DEVICE_PRE_MODEL_IDS")) {
            direct_target_only_probe(path,target,options.max_context);
            direct_binary_probe(path,target,draft,options.max_context,
                "COHERENT_DEVICE_DIRECT_BINARY_PRE");
        }
        // Dispatches recorded while the Engine constructs its context include
        // ordinary graph capture; replay bypasses host-side dispatch counters.
        using Linear=ninfer::exl3::Exl3CudaLinearWorkspace;
        const auto dispatch_snapshot=[] {
            std::array<std::uint64_t,7> value{Linear::process_coherent_down_k7_calls_for_test(),
                Linear::process_coherent_o_k7_calls_for_test()};
            for(int operation=0;operation<5;++operation)
                value[2+operation]=Linear::coherent_wide_k6_calls_for_test(operation);
            return value;
        };
        auto construction_dispatch=dispatch_snapshot();
        auto engine=std::make_unique<Engine>(options);
        {
            const auto after=dispatch_snapshot();
            for(std::size_t index=0;index<after.size();++index)
                construction_dispatch[index]=after[index]-construction_dispatch[index];
        }
        std::cout<<"COHERENT_DEVICE_ENGINE_CONSTRUCTION_DISPATCH down_k7="<<construction_dispatch[0]
                 <<" o_k7="<<construction_dispatch[1]<<" wide_q="<<construction_dispatch[2]
                 <<" wide_qkv="<<construction_dispatch[3]<<" wide_z="<<construction_dispatch[4]
                 <<" wide_o="<<construction_dispatch[5]<<" wide_gate_up="<<construction_dispatch[6]<<'\n';
        std::size_t free_bytes=0,total_bytes=0;
        if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess)
            throw std::runtime_error("device Engine witness memory query failed");
        std::cerr<<"COHERENT_DEVICE_ENGINE_MEMORY free_mib="
                 <<free_bytes/(1024*1024)
                 <<" runtime_reservation_mib="
                 <<engine->memory_summary().runtime_reservation_bytes/(1024*1024)
                 <<" reconstruction_mib="
                 <<engine->runtime_stats().reconstruction_device_bytes/(1024*1024)
                 <<" repair_fallback_lanes="
                 <<engine->runtime_stats().repair_checkpoint_reservation_fallback_lanes
                 <<'\n';
        PromptInput prompt;
        prompt.options.enable_thinking=false;
        std::string code;
        const std::string line="def accumulate(values):\n    return sum(v * 3 for v in values)\n";
        code.reserve(static_cast<std::size_t>(prefix)*14);
        while(code.size()<static_cast<std::size_t>(prefix)*2)code+=line;
        prompt.messages.push_back({ChatRole::User,{{MessagePartKind::Text,code}}});
        auto count=engine->count_tokens(prompt);
        while(count<static_cast<std::uint32_t>(prefix)) {
            code+=line;
            prompt.messages.back().parts.front().text=code;
            count=engine->count_tokens(prompt);
        }
        if(count>static_cast<std::uint32_t>(prefix+128))
            throw std::runtime_error("device Engine witness exceeded prompt headroom");
        RequestOptions request;
        request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=16;
        request.execution.allow_prefix_reuse=false;
        if(const auto* failure=std::getenv("NINFER_TEST_ENGINE_DEVICE_FAILURE");
           failure && std::string_view(failure)=="1") {
            bool reached=false;
            try {(void)engine->generate(engine->prepare(prompt),request);}
            catch(const std::exception& error) {
                reached=std::string_view(error.what()).find(
                    "injected layer-major partial-layer failure")!=
                    std::string_view::npos;
            }
#ifdef _WIN32
            _putenv_s("NINFER_EXL3_TEST_LAYER_MAJOR_FAIL_AFTER_LAYER","");
#else
            unsetenv("NINFER_EXL3_TEST_LAYER_MAJOR_FAIL_AFTER_LAYER");
#endif
            if(!reached)throw std::runtime_error(
                "device Engine partial prefill failure was not observed");
            bool sealed=false;
            try {(void)engine->generate(engine->prepare(prompt),request);}
            catch(const RequestError& error) {
                sealed=error.kind()==RequestErrorKind::Unavailable;
            }
            if(!sealed)throw std::runtime_error(
                "device Engine unexpectedly reused a failed physical context");
            engine.reset();
            engine=std::make_unique<Engine>(options);
            std::cout<<"COHERENT_DEVICE_ENGINE_PREFILL_FAILURE_RECOVERED\n";
        }
        if(const auto* path=std::getenv("NINFER_TEST_ENGINE_DEVICE_FIRST_IDS_OUT")) {
            auto frontend=ninfer::targets::qwen3_6::make_frontend(
                ninfer::exl3::load_pinned_frontend_resources(target),
                {.vision_enabled=false,.max_context=options.max_context});
            const auto encoded=frontend.prepare(prompt);
            const auto& ids=ninfer::targets::qwen3_6::PreparedPromptAccess::
                view(encoded).token_ids;
            if(ids.size()!=count)
                throw std::runtime_error("first Engine prompt ID extent changed");
            std::ofstream out(path,std::ios::binary);
            for(const auto id:ids)out<<id<<'\n';
            if(!out)throw std::runtime_error("first Engine prompt ID export failed");
        }
        const auto first=engine->generate(engine->prepare(prompt),request);
        if(const auto* path=std::getenv("NINFER_TEST_ENGINE_DEVICE_FIRST_TOKENS_OUT")) {
            std::ofstream out(path,std::ios::binary);
            for(const auto id:first.generated_token_ids)out<<id<<'\n';
            if(!out)throw std::runtime_error("first Engine token ID export failed");
        }
        if(first.generated_token_ids.empty() || first.generated_token_ids.size()>16 ||
           !first.speculative.enabled ||
           first.speculative.backend!=SpeculativeBackend::DFlash2 ||
           first.reused_prompt_tokens!=0 ||
           first.prefix_reuse_path!=PrefixReusePath::Root ||
           !first.token_accounting.conserves_result_tokens(
               first.generated_token_ids.size()) ||
           first.speculative.proposed_rows!=
               first.speculative.drafted_tokens+first.speculative.rounds ||
           first.speculative.committed_model_rows!=
               first.speculative.externally_visible_model_rows+
               first.speculative.hidden_terminal_rows ||
           first.timings.prefill_seconds<=0 || first.timings.decode_seconds<=0)
            throw std::runtime_error("device Engine output/accounting contract failed");
        // Warm-only mode serves matched timing arms: the first full-prefix
        // request above still checks the output contract and warms every
        // kernel at the measured shape; the repeat and ring-edge gates run in
        // the dedicated gate invocation instead of every timing arm.
        const bool warm_only=[] {
            const auto* option=std::getenv("NINFER_TEST_ENGINE_DEVICE_WARM_ONLY");
            return option && std::string_view(option)=="1";
        }();
        if(warm_only)
            std::cout<<"COHERENT_DEVICE_ENGINE_WARM_ONLY prompt="<<count
                     <<" first_output="<<first.generated_token_ids.size()<<'\n';
        if(!warm_only) {
        const auto again=engine->generate(engine->prepare(prompt),request);
        if(again.generated_token_ids!=first.generated_token_ids ||
           again.reused_prompt_tokens!=0 ||
           !again.token_accounting.conserves_result_tokens(
               again.generated_token_ids.size()))
            throw std::runtime_error("device Engine sequential request isolation failed");
        }
        if(prefix==4096 && !warm_only) {
            auto edge_prompt=prompt;
            const auto edge_count_for=[&](int repeats) {
                auto& text=edge_prompt.messages.back().parts.front().text;
                text.clear();
                text.reserve(static_cast<std::size_t>(repeats)*3);
                for(int i=0;i<repeats;++i)text+="hi ";
                return engine->count_tokens(edge_prompt);
            };
            int lower=0,upper=2048;
            while(edge_count_for(upper)<2048) {
                lower=upper;
                upper*=2;
            }
            while(lower+1<upper) {
                const int middle=lower+(upper-lower)/2;
                if(edge_count_for(middle)<2048)lower=middle;
                else upper=middle;
            }
            const auto edge_count=edge_count_for(upper);
            if(edge_count>2063)
                throw std::runtime_error("device Engine ring edge fixture overshot");
            auto edge_request=request;
            edge_request.execution.requested_output_tokens=4;
            const auto edge_first=engine->generate(engine->prepare(edge_prompt),edge_request);
            const auto edge_again=engine->generate(engine->prepare(edge_prompt),edge_request);
            if(edge_first.generated_token_ids!=edge_again.generated_token_ids ||
               edge_first.timings.prefill_seconds<=0 ||
               edge_first.reused_prompt_tokens!=0 ||
               !edge_first.token_accounting.conserves_result_tokens(
                   edge_first.generated_token_ids.size()))
                throw std::runtime_error("device Engine 2K ring edge contract failed");
            std::cout<<"COHERENT_DEVICE_ENGINE_RING_EDGE prompt="<<edge_count
                     <<" output="<<edge_first.generated_token_ids.size()
                     <<" repeats_equal=1\n";
        }
        if(const auto* long_option=std::getenv("NINFER_TEST_ENGINE_DEVICE_LONG");
           long_option && std::string_view(long_option)=="1") {
            const auto* fixture_option=std::getenv("NINFER_TEST_ENGINE_DEVICE_FIXTURE");
            if(!fixture_option ||
               (std::string_view(fixture_option)!="code" &&
                std::string_view(fixture_option)!="prose"))
                throw std::invalid_argument("device Engine long fixture must be code or prose");
            const bool prose=std::string_view(fixture_option)=="prose";
            const auto* source_path=std::getenv("NINFER_TEST_ENGINE_DEVICE_SOURCE");
            if(!source_path)throw std::invalid_argument(
                "device Engine long witness needs an explicit source file");
            std::ifstream source_file(source_path,std::ios::binary);
            if(!source_file)throw std::runtime_error(
                "device Engine long source file unavailable");
            std::string source(std::istreambuf_iterator<char>{source_file},
                std::istreambuf_iterator<char>{});
            if(source.empty())throw std::runtime_error(
                "device Engine long source file is empty");
            if(prose) {
                const auto* second_path=
                    std::getenv("NINFER_TEST_ENGINE_DEVICE_SOURCE_2");
                if(!second_path)throw std::invalid_argument(
                    "device Engine prose witness needs a second source file");
                std::ifstream second_file(second_path,std::ios::binary);
                if(!second_file)throw std::runtime_error(
                    "device Engine second prose source unavailable");
                source.append("\n\n");
                source.append(std::istreambuf_iterator<char>{second_file},
                    std::istreambuf_iterator<char>{});
            }
            auto long_prompt=prompt;
            // NINFER_TEST_ENGINE_DEVICE_MESSAGES: an explicit system/user chat
            // (system text, 0x1E, user text) replaces the source-sized prompt.
            if(const auto* messages_path=std::getenv("NINFER_TEST_ENGINE_DEVICE_MESSAGES")) {
                std::ifstream messages_file(messages_path,std::ios::binary);
                std::string messages(std::istreambuf_iterator<char>{messages_file},
                    std::istreambuf_iterator<char>{});
                const auto split=messages.find('');
                if(!messages_file.good() && messages.empty())
                    throw std::runtime_error("device Engine messages file unavailable");
                if(split==std::string::npos)
                    throw std::runtime_error("device Engine messages file lacks a separator");
                long_prompt.messages.clear();
                if(split>0)long_prompt.messages.push_back({ChatRole::System,
                    {{MessagePartKind::Text,messages.substr(0,split)}}});
                long_prompt.messages.push_back({ChatRole::User,
                    {{MessagePartKind::Text,messages.substr(split+1)}}});
                source.clear();
            }
            const std::string instruction=long_outputs>=512?
                (prose?
                    "\n\nWrite a detailed, numbered implementation guide for the serving "
                    "design above. Cover at least forty distinct mechanisms, with a "
                    "full paragraph for each. Continue consecutively through all "
                    "forty items before concluding. Begin item one now.\n":
                    "\n\nWrite a detailed, numbered code review of the C++ source above. "
                    "Cover at least forty distinct functions, state transitions, or "
                    "failure paths, with a full paragraph for each. Continue "
                    "consecutively through all forty items before concluding. "
                    "Begin item one now.\n"):
                (prose?
                    "\n\nExplain the serving design, state transitions, and failure "
                    "recovery described above in twenty numbered sections. Each "
                    "section should discuss a specific mechanism in several "
                    "sentences. Begin section one now.\n":
                    "\n\nExplain the architecture and request lifecycle of the C++ code above "
                    "in twenty numbered sections. Each section should discuss a "
                    "specific function, state transition, or failure path in detail. "
                    "Begin section one now.\n");
            const bool explicit_messages=source.empty();
            std::vector<std::size_t> line_ends;
            for(std::size_t pos=0;pos<source.size();++pos)
                if(source[pos]=='\n')line_ends.push_back(pos+1);
            if(line_ends.empty() || line_ends.back()!=source.size())
                line_ends.push_back(source.size());
            const auto count_for=[&](std::size_t line) {
                long_prompt.messages.back().parts.front().text=
                    source.substr(0,line_ends[line])+instruction;
                return engine->count_tokens(long_prompt);
            };
            std::size_t low=0,high=line_ends.size()-1;
            if(!explicit_messages) {
            if(count_for(high)<static_cast<std::uint32_t>(prefix))
                throw std::runtime_error("device Engine long source lacks prompt tokens");
            while(low<high) {
                const auto middle=low+(high-low)/2;
                if(count_for(middle)<static_cast<std::uint32_t>(prefix))low=middle+1;
                else high=middle;
            }
            }
            const auto long_count=explicit_messages?engine->count_tokens(long_prompt):count_for(low);
            if(long_count>static_cast<std::uint32_t>(prefix+128))
                throw std::runtime_error("device Engine long prompt exceeded headroom");
            if(const auto* path=std::getenv("NINFER_TEST_ENGINE_DEVICE_PROMPT_OUT")) {
                std::ofstream out(path,std::ios::binary);
                out<<long_prompt.messages.back().parts.front().text;
                if(!out)throw std::runtime_error(
                    "device Engine HTTP prompt export failed");
            }
            request.execution.requested_output_tokens=long_outputs;
            auto long_prepared=engine->prepare(long_prompt);
            if(const auto* path=std::getenv("NINFER_TEST_ENGINE_DEVICE_IDS_OUT")) {
                auto frontend=ninfer::targets::qwen3_6::make_frontend(
                    ninfer::exl3::load_pinned_frontend_resources(target),
                    {.vision_enabled=false,.max_context=options.max_context});
                const auto encoded=frontend.prepare(long_prompt);
                const auto& ids=ninfer::targets::qwen3_6::PreparedPromptAccess::
                    view(encoded).token_ids;
                if(ids.size()!=long_count)
                    throw std::runtime_error("device Engine HTTP token export count changed");
                std::ofstream out(path,std::ios::binary);
                for(const auto id:ids)out<<id<<'\n';
                if(!out)throw std::runtime_error(
                    "device Engine HTTP token export failed");
            }
            const bool coherent_o_k7=[] {
                const auto* option=std::getenv("NINFER_EXL3_COHERENT_O_K7");
                return option && std::string_view(option)=="1";
            }();
            const auto o_k7_calls_before=ninfer::exl3::Exl3CudaLinearWorkspace::
                process_coherent_o_k7_calls_for_test();
            const auto o_k7_rows_before=ninfer::exl3::Exl3CudaLinearWorkspace::
                process_coherent_o_k7_rows_for_test();
            const bool coherent_down_k7=[] {
                const auto* option=std::getenv("NINFER_EXL3_COHERENT_DOWN_K7");
                return option && std::string_view(option)=="1";
            }();
            const auto down_k7_calls_before=ninfer::exl3::Exl3CudaLinearWorkspace::
                process_coherent_down_k7_calls_for_test();
            const auto down_k7_rows_before=ninfer::exl3::Exl3CudaLinearWorkspace::
                process_coherent_down_k7_rows_for_test();
            const bool coherent_wide_k6=[] {
                const auto* option=std::getenv("NINFER_EXL3_TARGET_COHERENT_WIDE_K6");
                return option && std::string_view(option)=="1";
            }();
            std::uint64_t wide_calls_before[5]{},wide_rows_before[5]{};
            for(int operation=0;operation<5;++operation) {
                wide_calls_before[operation]=
                    ninfer::exl3::Exl3CudaLinearWorkspace::
                        coherent_wide_k6_calls_for_test(operation);
                wide_rows_before[operation]=
                    ninfer::exl3::Exl3CudaLinearWorkspace::
                        coherent_wide_k6_rows_for_test(operation);
            }
            const bool coherent_ordinary_graphs=[] {
                const auto* option=std::getenv("NINFER_EXL3_COHERENT_ORDINARY_GRAPHS");
                return !option || std::string_view(option)=="1";
            }();
            const auto graphs_before=ninfer::exl3::Exl3TextContext::
                ordinary_graph_process_stats_for_test();
            const auto long_result=engine->generate(std::move(long_prepared),request);
            const auto graphs_after=ninfer::exl3::Exl3TextContext::
                ordinary_graph_process_stats_for_test();
            const auto gdn_replays=graphs_after.gdn_segment_replays-graphs_before.gdn_segment_replays;
            const auto full_replays=graphs_after.full_layer_replays-graphs_before.full_layer_replays;
            const auto mlp_replays=graphs_after.mlp_tail_replays-graphs_before.mlp_tail_replays;
            std::cout<<"COHERENT_DEVICE_ENGINE_GRAPHS policy="<<coherent_ordinary_graphs
                     <<" gdn_segment_captures="<<graphs_after.gdn_segment_captures
                     <<" full_layer_captures="<<graphs_after.full_layer_captures
                     <<" mlp_tail_captures="<<graphs_after.mlp_tail_captures
                     <<" gdn_segment_replays="<<gdn_replays
                     <<" full_layer_replays="<<full_replays
                     <<" mlp_tail_replays="<<mlp_replays<<'\n';
            if(!coherent_ordinary_graphs && (gdn_replays || full_replays || mlp_replays))
                throw std::runtime_error("device Engine ordinary graph guard leaked replays");
            // A kernel reached only through replayed graphs has no host-side
            // request delta; it must then have been dispatched during capture.
            const bool graphs_replayed=gdn_replays || full_replays || mlp_replays;
            const auto captured_only=[&](std::uint64_t request_calls,std::size_t index) {
                return graphs_replayed && request_calls==0 && construction_dispatch[index]>0;
            };
            const auto down_k7_calls=ninfer::exl3::Exl3CudaLinearWorkspace::
                process_coherent_down_k7_calls_for_test()-down_k7_calls_before;
            const auto down_k7_rows=ninfer::exl3::Exl3CudaLinearWorkspace::
                process_coherent_down_k7_rows_for_test()-down_k7_rows_before;
            std::cout<<"COHERENT_DEVICE_ENGINE_DOWN_K7 enabled="<<coherent_down_k7
                     <<" calls="<<down_k7_calls<<" rows="<<down_k7_rows<<'\n';
            if((coherent_down_k7 && !captured_only(down_k7_calls,0) && (down_k7_calls==0 ||
                    down_k7_rows<static_cast<std::uint64_t>(long_outputs))) ||
               (!coherent_down_k7 && (down_k7_calls!=0 || down_k7_rows!=0)))
                throw std::runtime_error("device Engine K7 down dispatch mismatch");
            const auto o_k7_calls=ninfer::exl3::Exl3CudaLinearWorkspace::
                process_coherent_o_k7_calls_for_test()-o_k7_calls_before;
            const auto o_k7_rows=ninfer::exl3::Exl3CudaLinearWorkspace::
                process_coherent_o_k7_rows_for_test()-o_k7_rows_before;
            std::cout<<"COHERENT_DEVICE_ENGINE_O_K7 enabled="<<coherent_o_k7
                     <<" calls="<<o_k7_calls<<" rows="<<o_k7_rows<<'\n';
            if((coherent_o_k7 && !captured_only(o_k7_calls,1) && (o_k7_calls==0 ||
                    o_k7_rows<static_cast<std::uint64_t>(long_outputs))) ||
               (!coherent_o_k7 && (o_k7_calls!=0 || o_k7_rows!=0)))
                throw std::runtime_error("device Engine K7 O dispatch mismatch");
            constexpr const char* wide_names[5]={"q","qkv","z","o","gate_up"};
            for(int operation=0;operation<5;++operation) {
                const auto calls=ninfer::exl3::Exl3CudaLinearWorkspace::
                    coherent_wide_k6_calls_for_test(operation)-
                    wide_calls_before[operation];
                const auto rows=ninfer::exl3::Exl3CudaLinearWorkspace::
                    coherent_wide_k6_rows_for_test(operation)-
                    wide_rows_before[operation];
                std::cout<<"COHERENT_DEVICE_ENGINE_WIDE_K6 op="
                         <<wide_names[operation]<<" enabled="<<coherent_wide_k6
                         <<" calls="<<calls<<" rows="<<rows<<'\n';
                if(coherent_wide_k6 ?
                        !captured_only(calls,2+static_cast<std::size_t>(operation)) &&
                            (calls==0 || rows<calls) :
                        calls!=0 || rows!=0)
                    throw std::runtime_error(
                        "device Engine coherent wide K6 dispatch mismatch");
            }
            if(const auto* path=std::getenv("NINFER_TEST_ENGINE_DEVICE_TOKEN_IDS_OUT")) {
                std::ofstream out(path,std::ios::binary);
                for(const auto id:long_result.generated_token_ids)out<<id<<'\n';
                if(!out)throw std::runtime_error(
                    "device Engine output token export failed");
            }
            if(const auto* path=std::getenv("NINFER_TEST_ENGINE_DEVICE_OUTPUT_OUT")) {
                std::ofstream out(path,std::ios::binary);
                out<<long_result.content;
                if(!out)throw std::runtime_error(
                    "device Engine HTTP output export failed");
            }
            std::cout<<"COHERENT_DEVICE_ENGINE_LONG fixture="<<fixture_option
                     <<" prompt="<<long_count
                     <<" output="<<long_result.generated_token_ids.size()
                     <<" visible="<<long_result.token_accounting.visible_model_tokens
                     <<" rounds="<<long_result.speculative.rounds
                     <<" prefill_seconds="<<long_result.timings.prefill_seconds
                     <<" decode_seconds="<<long_result.timings.decode_seconds
                     <<" finish="<<static_cast<int>(long_result.finish_reason)<<'\n';
            if(long_result.generated_token_ids.size()!=
               static_cast<std::size_t>(long_outputs))
                throw std::runtime_error(
                    "device Engine long output ended before requested budget");
            std::array<std::size_t,9> accepted_hist{};
            for(const auto accepted:long_result.speculative.accepted_prefix_per_round)
                if(accepted<accepted_hist.size())++accepted_hist[accepted];
            std::cout<<"COHERENT_DEVICE_ENGINE_LONG_ROUNDS proposed="
                     <<long_result.speculative.proposed_rows
                     <<" verified="<<long_result.speculative.verified_rows
                     <<" replayed="<<long_result.speculative.replayed_rows
                     <<" accepted_hist=";
            for(const auto value:accepted_hist)std::cout<<value<<',';
            std::cout<<" first_proposal=";
            for(const auto token:long_result.speculative.first_proposed_tokens)
                std::cout<<token<<',';
            std::cout<<'\n';
            if(long_result.token_accounting.visible_model_tokens<64 ||
               !long_result.token_accounting.conserves_result_tokens(
                   long_result.generated_token_ids.size()))
                throw std::runtime_error(
                    "device Engine long output witness too short or inconsistent: "+
                    long_result.content.substr(0,160));
            if(const auto* cancel=std::getenv("NINFER_TEST_ENGINE_DEVICE_CANCEL");
               cancel && std::string_view(cancel)=="1") {
                struct CancellingSink final:OutputSink {
                    bool published=false;
                    std::string text;
                    void publish(OutputDelta delta) override {
                        if(delta.channel==OutputChannel::Content && !delta.text.empty()) {
                            text+=delta.text;
                            published=true;
                        }
                    }
                } sink;
                auto active=engine->submit(engine->prepare(long_prompt),request,
                    OutputConsumerMode::Streaming);
                const auto partial=active.wait(&sink,
                    CancellationView([&]{return sink.published;}));
                if(partial.finish_reason!=FinishReason::Cancelled ||
                   partial.generated_token_ids.empty() ||
                   partial.generated_token_ids.size()>=
                       long_result.generated_token_ids.size() ||
                   sink.text.empty())
                    throw std::runtime_error(
                        "device Engine streaming cancellation did not stop active output");
                const auto recovered=engine->generate(engine->prepare(long_prompt),request);
                if(recovered.generated_token_ids!=long_result.generated_token_ids)
                    throw std::runtime_error(
                        "device Engine cancellation contaminated subsequent output");
                std::cout<<"COHERENT_DEVICE_ENGINE_CANCEL_PASS partial="
                         <<partial.generated_token_ids.size()
                         <<" restored="<<recovered.generated_token_ids.size()<<'\n';
            }
        }
        std::cout<<"COHERENT_DEVICE_ENGINE_PASS prefix_target="<<prefix
                 <<" prompt="<<count
                 <<" first_output="<<first.generated_token_ids.size()
                 <<" rounds="<<first.speculative.rounds
                 <<" proposed="<<first.speculative.proposed_rows
                 <<" verified="<<first.speculative.verified_rows
                 <<" prefill_seconds="<<first.timings.prefill_seconds
                 <<" decode_seconds="<<first.timings.decode_seconds
                 <<" repeats_equal="<<(warm_only?"skipped":"1")<<'\n';
        if(const auto* path=std::getenv("NINFER_TEST_ENGINE_DEVICE_FIRST_IDS_OUT")) {
            engine.reset();
            direct_binary_probe(path,target,draft,options.max_context,
                "COHERENT_DEVICE_DIRECT_BINARY_POST");
        }
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<"COHERENT_DEVICE_ENGINE_FAIL "<<error.what()<<'\n';
        return 1;
    }
}
