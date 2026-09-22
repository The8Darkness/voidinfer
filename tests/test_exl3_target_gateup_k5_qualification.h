#pragma once
#include "test_exl3_k5_oracle.h"

struct TargetGateupK5Qualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit = Admission::target_continuation_gate_up;
    static constexpr int output = 17408;
    static constexpr std::size_t guard = 128, extent = 8u * output;
    std::filesystem::path directory;
    std::ofstream csv;
    std::vector<std::string> layers;
    explicit TargetGateupK5Qualification(const std::filesystem::path& path):directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),"gate/up K5 qualification output must be new");
        std::filesystem::create_directories(path);
        csv.open(path/"operators.csv");csv << "layer,rows,elements,exact,dispatch,canaries,repeat,finite\n";
    }
    template<class T> void save(const std::string& name,const std::vector<T>& data) {
        std::ofstream f(directory/name,std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
        require(f.good(),"gate/up K5 raw evidence write");
    }
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        if ((std::string(x.operation)=="gate" || std::string(x.operation)=="up") && x.metadata.K==5)
            static_cast<TargetGateupK5Qualification*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        require(x.rows==8 && x.metadata.K==5 && x.metadata.in_features==5120 && x.metadata.out_features==output && (std::string(x.operation)=="gate" || std::string(x.operation)=="up"),"gate/up K5 capture extent");
        const auto key=std::to_string(x.layer)+"-"+x.operation;
        require(std::find(layers.begin(),layers.end(),key)==layers.end(),"duplicate gate/up K5 capture");
        require(x.layer>=0 && x.layer<64,"gate/up K5 target owner layer");
        layers.push_back(key);
        cuda_check(cudaStreamSynchronize(x.stream),"gate/up K5 input ready");
        std::vector<std::uint16_t> input(8u*5120);
        cuda_check(cudaMemcpy(input.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"gate/up K5 input copy");
        require(std::all_of(input.begin(),input.end(),[](std::uint16_t v){return (v&0x7c00)!=0x7c00;}),"gate/up K5 finite represented inputs");
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M","0");
        Exl3CudaLinearWorkspace baseline(5120,output,8,false,true);
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M","1");
        Exl3CudaLinearWorkspace candidate(5120,output,8,false,true);
        Exl3CudaLinearWorkspace unowned(5120,output,8);
        Exl3CudaLinearWorkspace short_owner(5120,output,4,false,true);
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M","0");
        require(candidate.target_gateup_k5_small_m_candidate(x.metadata,8,admit) && !baseline.target_gateup_k5_small_m_candidate(x.metadata,8,admit),"gate/up K5 latched enable");
        require(!unowned.target_gateup_k5_small_m_candidate(x.metadata,8,admit) && !candidate.target_gateup_k5_small_m_candidate(x.metadata,8,Admission::ordinary) && !short_owner.target_gateup_k5_small_m_candidate(x.metadata,8,admit),"gate/up K5 ownership/capacity");
        for(int m=0;m<=17;++m)require(candidate.target_gateup_k5_small_m_candidate(x.metadata,m,admit)==(m>=2 && m<=8),"gate/up K5 width admission");
        for(int bad=0;bad<6;++bad) {
            auto md=x.metadata;
            if(bad==0)md.K=6;if(bad==1)md.mcg=true;if(bad==2)md.mul1=false;
            if(bad==3)md.has_bias=true;if(bad==4)md.in_features=4096;if(bad==5)md.out_features=5120;
            require(!candidate.target_gateup_k5_small_m_candidate(md,8,admit),"gate/up K5 metadata admission");
        }
        if(layers.size()==1)for(const char* disabled:{"","0","2","true","01"}) {
            _putenv_s("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M",disabled);
            Exl3CudaLinearWorkspace off(5120,output,8,false,true);
            require(!off.target_gateup_k5_small_m_candidate(x.metadata,8,admit),"gate/up K5 exact flag1 required");
        }
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M","0");
        require(std::string(candidate.dispatch_name(x.metadata,8,admit))=="m1_shape4_split" && std::string(candidate.dispatch_name(x.metadata,1,admit))=="m1_shape4_split","gate/up K5 dispatch");
        DeviceBuffer a((extent+2*guard)*2),b((extent+2*guard)*2);
        auto* pa=static_cast<std::uint16_t*>(a.get())+guard;
        auto* pb=static_cast<std::uint16_t*>(b.get())+guard;
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555),ref(initial),got(initial),repeat(initial);
        const auto serial=[&](int n){for(int row=0;row<n;++row)baseline.forward(x.weights,x.metadata,x.input+row*5120,pa+row*output,1,x.stream);};
        for(int n=1;n<=8;++n) {
            cuda_check(cudaMemcpy(a.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"gate/up K5 reference guards");
            cuda_check(cudaMemcpy(b.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"gate/up K5 candidate guards");
            serial(n);candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"gate/up K5 exact ready");
            cuda_check(cudaMemcpy(ref.data(),a.get(),ref.size()*2,cudaMemcpyDeviceToHost),"gate/up K5 reference copy");
            cuda_check(cudaMemcpy(got.data(),b.get(),got.size()*2,cudaMemcpyDeviceToHost),"gate/up K5 candidate copy");
            const auto tag=key+"-m"+std::to_string(n);
            if(n==8 || ref!=got){save(tag+"-reference.bin",ref);save(tag+"-candidate.bin",got);}
            require(ref==got,"gate/up K5 exact output mismatch "+tag);
            for(std::size_t i=0;i<got.size();++i) {
                if(i<guard || i>=guard+n*output)require(got[i]==0x3555,"gate/up K5 output tail/canary");
                else require((got[i]&0x7c00)!=0x7c00,"gate/up K5 nonfinite output");
            }
            candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"gate/up K5 repeat ready");
            cuda_check(cudaMemcpy(repeat.data(),b.get(),repeat.size()*2,cudaMemcpyDeviceToHost),"gate/up K5 repeat copy");
            require(repeat==got,"gate/up K5 repeat mismatch");
            csv << key << ',' << n << ',' << n*output << ",1," << candidate.dispatch_name(x.metadata,n,admit) << ",1,1,1\n";csv.flush();require(csv.good(),"gate/up K5 csv write");
        }
        save(key+"-input.bin",input);
        std::vector<std::uint16_t> after(input.size());
        cuda_check(cudaMemcpy(after.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"gate/up K5 immutable input");require(after==input,"gate/up K5 input mutation");
        // Reuse both workspaces with a changed input while keeping observed input immutable.
        auto changed=input;for(auto& bits:changed)bits^=0x8000;
        DeviceBuffer changed_device(changed.size()*2);
        cuda_check(cudaMemcpy(changed_device.get(),changed.data(),changed.size()*2,cudaMemcpyHostToDevice),"gate/up K5 changed input");
        auto* changed_ptr=static_cast<const std::uint16_t*>(changed_device.get());
        cuda_check(cudaMemcpy(a.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"gate/up K5 reuse reference guards");
        cuda_check(cudaMemcpy(b.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"gate/up K5 reuse candidate guards");
        for(int row=0;row<3;++row)baseline.forward(x.weights,x.metadata,changed_ptr+row*5120,pa+row*output,1,x.stream);
        candidate.forward(x.weights,x.metadata,changed_ptr,pb,3,x.stream,admit);
        cuda_check(cudaStreamSynchronize(x.stream),"gate/up K5 reuse ready");
        cuda_check(cudaMemcpy(ref.data(),a.get(),ref.size()*2,cudaMemcpyDeviceToHost),"gate/up K5 reused reference");
        cuda_check(cudaMemcpy(repeat.data(),b.get(),repeat.size()*2,cudaMemcpyDeviceToHost),"gate/up K5 reused candidate");
        if(ref!=repeat){save(key+"-reuse-reference.bin",ref);save(key+"-reuse-candidate.bin",repeat);}
        require(ref==repeat,"gate/up K5 changed input M8-to-M3 workspace reuse");
        for(std::size_t i=0;i<repeat.size();++i){if(i<guard || i>=guard+3*output)require(repeat[i]==0x3555,"gate/up K5 reused inactive tail/canary");else require((repeat[i]&0x7c00)!=0x7c00,"gate/up K5 reused finite output");}
        const auto oracle=k5_check_oracle_groups(x.weights,x.metadata,input,got,guard,{0,67,135});
        std::ofstream f(directory/(key+"-oracle.txt"));
        f << "selected_complete_H128_groups=0,67,135 values=" << oracle.values << " max_ratio=" << oracle.max_ratio << " within_bound=" << oracle.within_bound << '\n';
        require(f.good() && oracle.within_bound && oracle.values==3072,"gate/up K5 selected-column FP64 oracle gate");

    }
};

void run_target_gateup_k5_qualification(Exl3TextModel& target, Exl3Dflash2DraftModel& draft) {
    require(env("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M")=="0","gate/up K5 capture requires construction flag0");
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));require(ids.size()>=512,"gate/up K5 prompt extent");
    std::vector<std::int64_t> prompt(ids.begin(),ids.begin()+512);
    TapStage stage;
    for(int tap=0;tap<kTapCount;++tap){
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16u*kHidden*2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    auto ctx=target.create_context(true);require(ctx->try_enable_oscar_from_environment(),"gate/up K5 OSCAR required");
    draft.reset();ingest_prefix(*ctx,prompt,CommitSink{&draft,&stage});
    require(ctx->position()==512 && draft.ring_count()==512,"gate/up K5 capture prefix");
    const auto pending=sample_target(*ctx);
    std::vector<std::int64_t> block(8,kMaskToken);block[0]=pending;
    const auto proposals=draft.propose_cached(block,512,ctx->target_embedding(),ctx->target_lm_head_weights(),ctx->target_lm_head_metadata(),kMaskToken);
    require(proposals.size()==7,"gate/up K5 real proposals");
    block.assign(1,pending);block.insert(block.end(),proposals.begin(),proposals.end());
    TargetGateupK5Qualification check(env("NINFER_E5A4_OUT"));
    std::ofstream request(check.directory/"request.txt");request << "pending=" << pending << " proposals=";for(auto token:proposals)request << token << ',';request << '\n';request.flush();require(request.good(),"gate/up K5 request witness write");
    ctx->prepare_transaction();ctx->prepare_continuation(8);const auto ring=draft.ring_digest();
    ctx->set_target_projection_observer_for_test(TargetGateupK5Qualification::callback,&check,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    ctx->begin_transaction();ctx->continue_rows(block);ctx->set_target_projection_observer_for_test(nullptr,nullptr);ctx->rollback_transaction();
    cuda_check(cudaDeviceSynchronize(),"gate/up K5 sweep ready");
    require(check.layers.size()==30 && ctx->position()==512 && ctx->device_position_host()==511 && sample_target(*ctx)==pending && !ctx->transaction_active(),"gate/up K5 full layer coverage/rollback");
    require(draft.ring_digest()==ring && draft.ring_count()==512 && draft.ring_base_abs()==0,"gate/up K5 ring unchanged");
    std::ofstream f(check.directory/"result.txt");f << "PASS operators=30 widths=1..8 cases=240 selected_oracle_values_per_layer=3072 groups=0,67,135\n";
    require(f.good(),"gate/up K5 result write");std::cout << "TARGET_GATEUP_K5 PASS operators=30 cases=240\n";
}

