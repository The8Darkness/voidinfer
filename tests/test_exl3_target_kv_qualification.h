#pragma once
#include "test_exl3_target_kv_oracle.h"

struct TargetKvQualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit = Admission::target_continuation_kv;
    static constexpr int output = 1024;
    static constexpr std::size_t guard = 128, extent = 8u * output;
    std::filesystem::path directory;
    std::ofstream csv;
    std::vector<std::string> layers;
    explicit TargetKvQualification(const std::filesystem::path& path):directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),"K/V small-M qualification output must be new");
        std::filesystem::create_directories(path);
        csv.open(path/"operators.csv");csv << "layer,K,rows,elements,exact,dispatch,canaries,repeat,finite\n";
    }
    template<class T> void save(const std::string& name,const std::vector<T>& data) {
        std::ofstream f(directory/name,std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
        require(f.good(),"K/V small-M raw evidence write");
    }
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        if ((std::string(x.operation)=="k" || std::string(x.operation)=="v") && (x.metadata.K==7 || x.metadata.K==8))
            static_cast<TargetKvQualification*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        const auto stream_mode=env("NINFER_TEST_KV_K7_STREAM_REDUCTION");
        require(stream_mode.empty() || stream_mode=="1","K7 KV stream fixture option");
        const bool stream_test=stream_mode=="1";
        if(stream_test)require(env("NINFER_EXL3_EXTENDED_STREAM_REDUCTION").empty() ||
            env("NINFER_EXL3_EXTENDED_STREAM_REDUCTION")=="0","K7 KV fixture isolates new route from extended stream option");
        struct RestoreStream {
            std::string value;
            ~RestoreStream(){_putenv_s("NINFER_EXL3_KV_K7_STREAM_REDUCTION",value.c_str());}
        } restore_stream{env("NINFER_EXL3_KV_K7_STREAM_REDUCTION")};
        if(stream_test)_putenv_s("NINFER_EXL3_KV_K7_STREAM_REDUCTION","0");
        require(x.rows==8 && (x.metadata.K==7 || x.metadata.K==8) && x.metadata.in_features==5120 && x.metadata.out_features==output && (std::string(x.operation)=="k" || std::string(x.operation)=="v"),"K/V small-M capture extent");
        const auto key=std::to_string(x.layer)+"-"+x.operation;
        require(std::find(layers.begin(),layers.end(),key)==layers.end(),"duplicate K/V small-M capture");
        require(x.layer>=0 && x.layer<64,"K/V small-M target owner layer");
        layers.push_back(key);
        cuda_check(cudaStreamSynchronize(x.stream),"K/V small-M input ready");
        std::vector<std::uint16_t> input(8u*5120);
        cuda_check(cudaMemcpy(input.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"K/V small-M input copy");
        require(std::all_of(input.begin(),input.end(),[](std::uint16_t v){return (v&0x7c00)!=0x7c00;}),"K/V small-M finite represented inputs");
        _putenv_s("NINFER_EXL3_TARGET_KV_SMALL_M","0");
        Exl3CudaLinearWorkspace baseline(5120,output,8,false,false,false,false,false,false,false,{}, {},false,true);
        _putenv_s("NINFER_EXL3_TARGET_KV_SMALL_M","1");
        Exl3CudaLinearWorkspace parent(5120,output,8,false,false,false,false,false,false,false,{}, {},false,true);
        if(stream_test)_putenv_s("NINFER_EXL3_KV_K7_STREAM_REDUCTION","1");
        Exl3CudaLinearWorkspace candidate(5120,output,8,false,false,false,false,false,false,false,{}, {},false,true);
        Exl3CudaLinearWorkspace unowned(5120,output,8);
        Exl3CudaLinearWorkspace short_owner(5120,output,4,false,false,false,false,false,false,false,{}, {},false,true);
        _putenv_s("NINFER_EXL3_TARGET_KV_SMALL_M","0");
        require(candidate.target_kv_small_m_candidate(x.metadata,8,admit) && !baseline.target_kv_small_m_candidate(x.metadata,8,admit),"K/V small-M latched enable");
        require(!unowned.target_kv_small_m_candidate(x.metadata,8,admit) && !candidate.target_kv_small_m_candidate(x.metadata,8,Admission::ordinary) && !short_owner.target_kv_small_m_candidate(x.metadata,8,admit),"K/V small-M ownership/capacity");
        for(int m=0;m<=17;++m)require(candidate.target_kv_small_m_candidate(x.metadata,m,admit)==(m>=2 && m<=8),"K/V small-M width admission");
        if(stream_test) {
            for(int m=0;m<=17;++m)
                require((std::string(candidate.dispatch_name(x.metadata,m,admit))=="target_extended_stream_reduction")==
                    (x.metadata.K==7 && m>=2 && m<=8),"K7 KV stream width menu");
            for(const auto* workspace:{&baseline,&parent,&unowned,&short_owner})
                require(std::string(workspace->dispatch_name(x.metadata,8,admit))!="target_extended_stream_reduction",
                    "K7 KV stream bypassed ownership/capacity or disabled option");
            require(std::string(candidate.dispatch_name(x.metadata,8,Admission::ordinary))!="target_extended_stream_reduction",
                "K7 KV stream admitted ordinary projection");
        }
        for(int bad=0;bad<6;++bad) {
            auto md=x.metadata;
            if(bad==0)md.K=6;if(bad==1)md.mcg=true;if(bad==2)md.mul1=false;
            if(bad==3)md.has_bias=true;if(bad==4)md.in_features=4096;if(bad==5)md.out_features=5120;
            require(!candidate.target_kv_small_m_candidate(md,8,admit),"K/V small-M metadata admission");
            if(stream_test)require(std::string(candidate.dispatch_name(md,8,admit))!="target_extended_stream_reduction",
                "K7 KV stream admitted incompatible metadata");
        }
        if(layers.size()==1)for(const char* disabled:{"","0","2","true","01"}) {
            _putenv_s("NINFER_EXL3_TARGET_KV_SMALL_M",disabled);
            Exl3CudaLinearWorkspace off(5120,output,8,false,false,false,false,false,false,false,{}, {},false,true);
            require(!off.target_kv_small_m_candidate(x.metadata,8,admit),"K/V small-M exact flag1 required");
        }
        _putenv_s("NINFER_EXL3_TARGET_KV_SMALL_M","0");
        if(stream_test && layers.size()==1) {
            for(const char* malformed:{"2","true","01"," 1","1 "}) {
                _putenv_s("NINFER_EXL3_KV_K7_STREAM_REDUCTION",malformed);
                bool refused=false;
                try{Exl3CudaLinearWorkspace invalid(5120,output,8,false,false,false,false,false,false,false,{}, {},false,true);}
                catch(const std::invalid_argument&){refused=true;}
                require(refused,"malformed K7 KV stream option admitted");
            }
            _putenv_s("NINFER_EXL3_KV_K7_STREAM_REDUCTION","0");
        }
        require(std::string(candidate.dispatch_name(x.metadata,8,admit))==
            (stream_test && x.metadata.K==7?"target_extended_stream_reduction":"target_kv_small_m_mma_split") &&
            std::string(candidate.dispatch_name(x.metadata,1,admit))=="generic_mma_split","K/V small-M dispatch");
        DeviceBuffer a((extent+2*guard)*2),b((extent+2*guard)*2);
        auto* pa=static_cast<std::uint16_t*>(a.get())+guard;
        auto* pb=static_cast<std::uint16_t*>(b.get())+guard;
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555),ref(initial),got(initial),repeat(initial);
        const auto serial=[&](int n){
            if(stream_test)parent.forward(x.weights,x.metadata,x.input,pa,n,x.stream,admit);
            else for(int row=0;row<n;++row)baseline.forward(x.weights,x.metadata,x.input+row*5120,pa+row*output,1,x.stream);
        };
        for(int n=1;n<=8;++n) {
            cuda_check(cudaMemcpy(a.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"K/V small-M reference guards");
            cuda_check(cudaMemcpy(b.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"K/V small-M candidate guards");
            const auto stream_before=candidate.stream_reduction_calls(true);
            serial(n);candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);
            if(stream_test)require(candidate.stream_reduction_calls(true)-stream_before==
                (x.metadata.K==7 && n>=2?1u:0u),"K7 KV stream dispatch count or K8/M1 exclusion");
            cuda_check(cudaStreamSynchronize(x.stream),"K/V small-M exact ready");
            cuda_check(cudaMemcpy(ref.data(),a.get(),ref.size()*2,cudaMemcpyDeviceToHost),"K/V small-M reference copy");
            cuda_check(cudaMemcpy(got.data(),b.get(),got.size()*2,cudaMemcpyDeviceToHost),"K/V small-M candidate copy");
            const auto tag=key+"-m"+std::to_string(n);
            if(n==8 || ref!=got){save(tag+"-reference.bin",ref);save(tag+"-candidate.bin",got);}
            require(ref==got,"K/V small-M exact output mismatch "+tag);
            for(std::size_t i=0;i<got.size();++i) {
                if(i<guard || i>=guard+n*output)require(got[i]==0x3555,"K/V small-M output tail/canary");
                else require((got[i]&0x7c00)!=0x7c00,"K/V small-M nonfinite output");
            }
            candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"K/V small-M repeat ready");
            cuda_check(cudaMemcpy(repeat.data(),b.get(),repeat.size()*2,cudaMemcpyDeviceToHost),"K/V small-M repeat copy");
            require(repeat==got,"K/V small-M repeat mismatch");
            csv << key << ',' << x.metadata.K << ',' << n << ',' << n*output << ",1," << candidate.dispatch_name(x.metadata,n,admit) << ",1,1,1\n";csv.flush();require(csv.good(),"K/V small-M csv write");
        }
        save(key+"-input.bin",input);
        if(stream_test && x.metadata.K==7) {
            struct CaptureResources {
                cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;
                ~CaptureResources(){if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
            } capture_resources;
            cuda_check(cudaStreamCreateWithFlags(&capture_resources.stream,cudaStreamNonBlocking),"K7 KV refusal stream");
            auto* scratch=const_cast<std::uint16_t*>(candidate.transformed_device());
            cuda_check(cudaMemsetAsync(scratch,0x5a,input.size()*2,capture_resources.stream),"K7 KV capture scratch sentinel");
            cuda_check(cudaMemsetAsync(b.get(),0x5a,initial.size()*2,capture_resources.stream),"K7 KV capture output sentinel");
            cuda_check(cudaStreamSynchronize(capture_resources.stream),"K7 KV capture sentinels ready");
            const auto submissions=candidate.stream_reduction_calls(true);
            cuda_check(cudaStreamBeginCapture(capture_resources.stream,cudaStreamCaptureModeThreadLocal),"K7 KV begin refused capture");
            bool refused=false;
            try{candidate.forward(x.weights,x.metadata,x.input,pb,8,capture_resources.stream,admit);}
            catch(const std::invalid_argument&){refused=true;}
            cuda_check(cudaStreamEndCapture(capture_resources.stream,&capture_resources.graph),"K7 KV end refused capture");
            require(refused && candidate.stream_reduction_calls(true)==submissions,"K7 KV capture refusal submitted stream work");
            std::size_t nodes=0;
            cuda_check(cudaGraphGetNodes(capture_resources.graph,nullptr,&nodes),"K7 KV refused graph nodes");
            require(nodes==0,"K7 KV refusal captured input transform or numerical work");
            std::vector<std::uint16_t> scratch_after(input.size());
            cuda_check(cudaMemcpy(scratch_after.data(),scratch,scratch_after.size()*2,cudaMemcpyDeviceToHost),"K7 KV refused scratch readback");
            cuda_check(cudaMemcpy(got.data(),b.get(),got.size()*2,cudaMemcpyDeviceToHost),"K7 KV refused output readback");
            require(std::all_of(scratch_after.begin(),scratch_after.end(),[](auto v){return v==0x5a5a;}) &&
                std::all_of(got.begin(),got.end(),[](auto v){return v==0x5a5a;}),"K7 KV capture refusal mutated buffers");
        }
        std::vector<std::uint16_t> after(input.size());
        cuda_check(cudaMemcpy(after.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"K/V small-M immutable input");require(after==input,"K/V small-M input mutation");
        // Reuse both workspaces with a changed input while keeping observed input immutable.
        auto changed=input;for(auto& bits:changed)bits^=0x8000;
        DeviceBuffer changed_device(changed.size()*2);
        cuda_check(cudaMemcpy(changed_device.get(),changed.data(),changed.size()*2,cudaMemcpyHostToDevice),"K/V small-M changed input");
        auto* changed_ptr=static_cast<const std::uint16_t*>(changed_device.get());
        cuda_check(cudaMemcpy(a.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"K/V small-M reuse reference guards");
        cuda_check(cudaMemcpy(b.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"K/V small-M reuse candidate guards");
        for(int row=0;row<3;++row)baseline.forward(x.weights,x.metadata,changed_ptr+row*5120,pa+row*output,1,x.stream);
        candidate.forward(x.weights,x.metadata,changed_ptr,pb,3,x.stream,admit);
        cuda_check(cudaStreamSynchronize(x.stream),"K/V small-M reuse ready");
        cuda_check(cudaMemcpy(ref.data(),a.get(),ref.size()*2,cudaMemcpyDeviceToHost),"K/V small-M reused reference");
        cuda_check(cudaMemcpy(repeat.data(),b.get(),repeat.size()*2,cudaMemcpyDeviceToHost),"K/V small-M reused candidate");
        if(ref!=repeat){save(key+"-reuse-reference.bin",ref);save(key+"-reuse-candidate.bin",repeat);}
        require(ref==repeat,"K/V small-M changed input M8-to-M3 workspace reuse");
        for(std::size_t i=0;i<repeat.size();++i){if(i<guard || i>=guard+3*output)require(repeat[i]==0x3555,"K/V small-M reused inactive tail/canary");else require((repeat[i]&0x7c00)!=0x7c00,"K/V small-M reused finite output");}
        const auto oracle=target_kv_check_oracle_groups(x.weights,x.metadata,input,got,guard,{0,1,2,3,4,5,6,7});
        std::ofstream f(directory/(key+"-oracle.txt"));
        f << "selected_complete_H128_groups=0,1,2,3,4,5,6,7 values=" << oracle.values << " max_ratio=" << oracle.max_ratio << " within_bound=" << oracle.within_bound << '\n';
        require(f.good() && oracle.within_bound && oracle.values==8192,"K/V small-M selected-column FP64 oracle gate");

    }
};

void run_target_kv_qualification(Exl3TextModel& target, Exl3Dflash2DraftModel& draft) {
    require(env("NINFER_EXL3_TARGET_KV_SMALL_M")=="0","K/V small-M capture requires construction flag0");
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));require(ids.size()>=512,"K/V small-M prompt extent");
    std::vector<std::int64_t> prompt(ids.begin(),ids.begin()+512);
    TapStage stage;
    for(int tap=0;tap<kTapCount;++tap){
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16u*kHidden*2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    auto ctx=target.create_context(true);require(ctx->try_enable_oscar_from_environment(),"K/V small-M OSCAR required");
    draft.reset();ingest_prefix(*ctx,prompt,CommitSink{&draft,&stage});
    require(ctx->position()==512 && draft.ring_count()==512,"K/V small-M capture prefix");
    const auto pending=sample_target(*ctx);
    std::vector<std::int64_t> block(8,kMaskToken);block[0]=pending;
    const auto proposals=draft.propose_cached(block,512,ctx->target_embedding(),ctx->target_lm_head_weights(),ctx->target_lm_head_metadata(),kMaskToken);
    require(proposals.size()==7,"K/V small-M real proposals");
    block.assign(1,pending);block.insert(block.end(),proposals.begin(),proposals.end());
    TargetKvQualification check(env("NINFER_E5A4_OUT"));
    std::ofstream request(check.directory/"request.txt");request << "pending=" << pending << " proposals=";for(auto token:proposals)request << token << ',';request << '\n';request.flush();require(request.good(),"K/V small-M request witness write");
    ctx->prepare_transaction();ctx->prepare_continuation(8);const auto ring=draft.ring_digest();
    ctx->set_target_projection_observer_for_test(TargetKvQualification::callback,&check,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    ctx->begin_transaction();ctx->continue_rows(block);ctx->set_target_projection_observer_for_test(nullptr,nullptr);ctx->rollback_transaction();
    cuda_check(cudaDeviceSynchronize(),"K/V small-M sweep ready");
    require(check.layers.size()==32 && ctx->position()==512 && ctx->device_position_host()==511 && sample_target(*ctx)==pending && !ctx->transaction_active(),"K/V small-M full layer coverage/rollback");
    require(draft.ring_digest()==ring && draft.ring_count()==512 && draft.ring_base_abs()==0,"K/V small-M ring unchanged");
    if(env("NINFER_TEST_KV_K7_STREAM_REDUCTION")=="1") {
        struct RestoreContextFlags {
            std::string kv,stream;
            ~RestoreContextFlags(){_putenv_s("NINFER_EXL3_TARGET_KV_SMALL_M",kv.c_str());
                _putenv_s("NINFER_EXL3_KV_K7_STREAM_REDUCTION",stream.c_str());}
        } restore{env("NINFER_EXL3_TARGET_KV_SMALL_M"),env("NINFER_EXL3_KV_K7_STREAM_REDUCTION")};
        const auto snapshot=[](Exl3TextContext& context) {
            auto bytes=context.oscar_live_state_host_for_test();
            const auto append=[&](const auto& values) {
                const auto* begin=reinterpret_cast<const std::byte*>(values.data());
                bytes.insert(bytes.end(),begin,begin+values.size()*sizeof(values[0]));
            };
            for(int layer=0;layer<64;++layer)if(layer%4!=3) {
                append(context.gdn_state_host(layer));append(context.gdn_physical_conv_host(layer));
            }
            return bytes;
        };
        std::vector<std::byte> expected;
        std::vector<std::byte> expected_root;
        std::ofstream context_checks(check.directory/"stream_context.csv");
        context_checks << "candidate,state_bytes,submissions,root_equal,outputs_equal,rollback_equal,replay_equal\n";
        require(context_checks.good(),"K7 KV context receipt creation");
        std::vector<std::uint16_t> expected_logits;
        std::array<std::vector<std::uint16_t>,5> expected_taps;
        for(bool candidate:{false,true}) {
            _putenv_s("NINFER_EXL3_TARGET_KV_SMALL_M","1");
            _putenv_s("NINFER_EXL3_KV_K7_STREAM_REDUCTION",candidate?"1":"0");
            auto comparison=target.create_context(true);
            require(comparison->try_enable_oscar_from_environment(),"K7 KV context OSCAR required");
            ingest_prefix(*comparison,prompt,CommitSink{});
            comparison->prepare_transaction();comparison->prepare_continuation(8);
            const auto root=snapshot(*comparison);
            if(!candidate)expected_root=root;
            else require(root==expected_root,"K7 KV candidate began from different logical state");
            const auto root_position=comparison->position();
            const auto root_device_position=comparison->device_position_host();
            const auto before=comparison->stream_reduction_calls(true);
            comparison->begin_transaction();comparison->continue_rows(block);
            const auto state=snapshot(*comparison);
            std::vector<std::uint16_t> logits(248320);
            cuda_check(cudaMemcpy(logits.data(),comparison->logits_device(),logits.size()*2,cudaMemcpyDeviceToHost),"K7 KV context logits");
            const auto taps=comparison->exact_tap_rows_host();
            if(!candidate) {
                require(comparison->stream_reduction_calls(true)==before,"K7 KV parent used candidate route");
                expected=state;expected_logits=logits;expected_taps=taps;
            } else {
                require(comparison->stream_reduction_calls(true)>before,"K7 KV context missed new route");
                require(state==expected && logits==expected_logits && taps==expected_taps,
                    "K7 KV context state/logits/taps differ from cooperative parent");
            }
            comparison->rollback_transaction();
            require(snapshot(*comparison)==root && comparison->position()==root_position &&
                comparison->device_position_host()==root_device_position,"K7 KV context rollback changed root");
            comparison->begin_transaction();comparison->continue_rows(block);
            require(snapshot(*comparison)==state,"K7 KV replay changed logical state");
            std::vector<std::uint16_t> replay_logits(logits.size());
            cuda_check(cudaMemcpy(replay_logits.data(),comparison->logits_device(),replay_logits.size()*2,cudaMemcpyDeviceToHost),
                "K7 KV replay logits");
            require(replay_logits==logits && comparison->exact_tap_rows_host()==taps,"K7 KV replay changed outputs");
            comparison->rollback_transaction();
            require(snapshot(*comparison)==root && comparison->position()==root_position &&
                comparison->device_position_host()==root_device_position,"K7 KV replay rollback changed root");
            context_checks << candidate << ',' << state.size() << ','
                << comparison->stream_reduction_calls(true)-before << ",1,1,1,1\n";
            context_checks.flush();require(context_checks.good(),"K7 KV context receipt write");
        }
        require(draft.ring_digest()==ring,"K7 KV context comparison changed private draft ring");
    }
    std::ofstream f(check.directory/"result.txt");f << "PASS operators=32 widths=1..8 cases=256 selected_oracle_values_per_layer=8192 groups=0,1,2,3,4,5,6,7\n";
    require(f.good(),"K/V small-M result write");std::cout << "TARGET_KV_SMALL_M PASS operators=32 cases=256\n";
}
