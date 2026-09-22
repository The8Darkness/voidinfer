#pragma once

struct Initial16Qualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit = Admission::target_initial16;
    static constexpr std::size_t guard = 128;
    std::uint64_t total_elements = 0;
    std::filesystem::path directory;
    std::ofstream csv;
    int count = 0;
    std::vector<std::string> seen;

    explicit Initial16Qualification(const std::filesystem::path& path)
        : directory(path) {
        require(!path.empty() && !std::filesystem::exists(path), "initial16 ordered output must be new");
        std::filesystem::create_directories(path);
        csv.open(path / "operators.csv");
        csv << "layer,operation,input,output,K,elements,exact,baseline_us,candidate_us\n";
    }
    template<class T> void save(const std::string& name, const std::vector<T>& data) {
        std::ofstream out(directory / name, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
        require(out.good(), "initial16 ordered raw output failed");
    }
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x, void* user) {
        static_cast<Initial16Qualification*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        require(x.rows == 16 && x.metadata.K >= 5 && x.metadata.K <= 8, "initial16 ordered observation extent");
        const int in = x.metadata.in_features, out = x.metadata.out_features;
        const std::size_t extent = 16u * out;
        const std::string tag = std::to_string(x.layer) + "-" + x.operation;
        require(std::find(seen.begin(), seen.end(), tag) == seen.end(), "duplicate prefill projection");
        seen.push_back(tag);
        cuda_check(cudaStreamSynchronize(x.stream), "prefill input ready");
        std::vector<std::uint16_t> input(16u * in);
        cuda_check(cudaMemcpy(input.data(), x.input, input.size()*2, cudaMemcpyDeviceToHost), "prefill input copy");
        _putenv_s("NINFER_EXL3_TARGET_INITIAL16_ORDERED", "0");
        Exl3CudaLinearWorkspace baseline(in, out, 16, false, false, false, false, false, true);
        _putenv_s("NINFER_EXL3_TARGET_INITIAL16_ORDERED", "1");
        Exl3CudaLinearWorkspace candidate(in, out, 16, false, false, false, false, false, true);
        Exl3CudaLinearWorkspace unowned(in, out, 16);
        _putenv_s("NINFER_EXL3_TARGET_INITIAL16_ORDERED", "0");
        require(std::string(baseline.dispatch_name(x.metadata,16,admit)) == "generic_tile", "baseline dispatch");
        require(std::string(candidate.dispatch_name(x.metadata,16,admit)) == "target_initial16_ordered", "candidate dispatch");
        require(!unowned.target_initial16_candidate(x.metadata,16,admit) &&
                !candidate.target_initial16_candidate(x.metadata,16,Admission::ordinary), "ownership admission");
        _putenv_s("NINFER_EXL3_TARGET_INITIAL16_ORDERED", "1");
        Exl3CudaLinearWorkspace short_owner(in,out,15,false,false,false,false,false,true);
        _putenv_s("NINFER_EXL3_TARGET_INITIAL16_ORDERED", "0");
        require(!short_owner.target_initial16_candidate(x.metadata,16,admit),"workspace extent admission");
        for(int bad=0;bad<6;++bad) {
            auto metadata=x.metadata;
            if(bad==0) metadata.K=4;
            if(bad==1) metadata.mcg=true;
            if(bad==2) metadata.mul1=false;
            if(bad==3) metadata.has_bias=true;
            if(bad==4) metadata.in_features=in+128;
            if(bad==5) metadata.out_features=out+128;
            require(!candidate.target_initial16_candidate(metadata,16,admit),"metadata admission");
        }
        for(int m=1;m<=17;++m) require(candidate.target_initial16_candidate(x.metadata,m,admit)==(m==16), "M16 boundary");
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555), old(initial), now(initial), repeat(initial);
        DeviceBuffer a(initial.size()*2), b(initial.size()*2);
        cuda_check(cudaMemcpy(a.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"baseline guards");
        cuda_check(cudaMemcpy(b.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"candidate guards");
        auto* pa=static_cast<std::uint16_t*>(a.get())+guard;
        auto* pb=static_cast<std::uint16_t*>(b.get())+guard;
        baseline.forward(x.weights,x.metadata,x.input,pa,16,x.stream,admit);
        candidate.forward(x.weights,x.metadata,x.input,pb,16,x.stream,admit);
        cuda_check(cudaStreamSynchronize(x.stream),"prefill compare ready");
        cuda_check(cudaMemcpy(old.data(),a.get(),old.size()*2,cudaMemcpyDeviceToHost),"baseline copy");
        cuda_check(cudaMemcpy(now.data(),b.get(),now.size()*2,cudaMemcpyDeviceToHost),"candidate copy");
        save(tag+"-input.bin",input);save(tag+"-baseline.bin",old);save(tag+"-candidate.bin",now);
        require(old==now,"initial16 ordered exact mismatch: "+tag);
        for(std::size_t i=0;i<guard;++i) require(now[i]==0x3555 && now[guard+extent+i]==0x3555,"prefill canary");
        for(std::size_t i=guard;i<guard+extent;++i) require((now[i]&0x7c00)!=0x7c00,"nonfinite prefill output");
        candidate.forward(x.weights,x.metadata,x.input,pb,16,x.stream,admit);
        cuda_check(cudaStreamSynchronize(x.stream),"prefill repeat ready");
        cuda_check(cudaMemcpy(repeat.data(),b.get(),repeat.size()*2,cudaMemcpyDeviceToHost),"repeat copy");
        require(repeat==now,"prefill repeat mismatch");
        std::vector<std::uint16_t> after(input.size());
        cuda_check(cudaMemcpy(after.data(),x.input,after.size()*2,cudaMemcpyDeviceToHost),"input immutability");
        require(after==input,"prefill input mutated");
        cudaEvent_t begin{},end{};
        cuda_check(cudaEventCreate(&begin),"create prefill timing begin");
        cuda_check(cudaEventCreate(&end),"create prefill timing end");
        const auto time=[&](Exl3CudaLinearWorkspace& w,std::uint16_t* output) {
            cuda_check(cudaEventRecord(begin,x.stream),"record prefill begin");
            for(int rep=0;rep<3;++rep) w.forward(x.weights,x.metadata,x.input,output,16,x.stream,admit);
            cuda_check(cudaEventRecord(end,x.stream),"record prefill end");
            cuda_check(cudaEventSynchronize(end),"sync prefill end");
            float ms=0;cuda_check(cudaEventElapsedTime(&ms,begin,end),"prefill elapsed");return ms*1000.0/3;
        };
        const double before=time(baseline,pa),after_us=time(candidate,pb);
        cudaEventDestroy(begin);cudaEventDestroy(end);
        csv << x.layer << ',' << x.operation << ',' << in << ',' << out << ',' << x.metadata.K << ',' << extent << ",1," << before << ',' << after_us << '\n';
        csv.flush();require(csv.good(),"prefill csv write");++count;total_elements+=extent;
    }
};

void run_initial16_qualification(Exl3TextModel& target) {
    require(env("NINFER_EXL3_TARGET_INITIAL16_ORDERED")=="0", "prefill capture needs flag0 baseline");
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));require(ids.size()>=16,"prefill prompt extent");ids.resize(16);
    auto ctx=target.create_context(true);
    require(ctx->try_enable_oscar_from_environment(),"prefill OSCAR required");
    Initial16Qualification check(env("NINFER_E5A4_OUT"));
    ctx->set_target_projection_observer_for_test(Initial16Qualification::callback,&check,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::initial16_generic);
    ctx->prefill(ids);
    cuda_check(cudaDeviceSynchronize(),"prefill qualification complete");
    ctx->set_target_projection_observer_for_test(nullptr,nullptr);
    require(check.count==276 && ctx->position()==16,"initial16 complete276 operator coverage");
    // Independently constructed flag1 context must preserve the complete native
    // state, all captured tap rows and logits, then later teacher decodes.
    _putenv_s("NINFER_EXL3_TARGET_INITIAL16_ORDERED","1");
    auto candidate=target.create_context(true);
    _putenv_s("NINFER_EXL3_TARGET_INITIAL16_ORDERED","0");
    require(candidate->try_enable_oscar_from_environment(),"candidate prefill OSCAR");
    candidate->prefill(ids);
    const auto compare=[&] {
        cuda_check(cudaDeviceSynchronize(),"prefill state ready");
        require_target_continue_state_equal(*candidate,*ctx,"prefill M16 state");
        for(int layer=0;layer<64;++layer) if((layer+1)%4!=0)
            require(candidate->gdn_physical_conv_host(layer)==ctx->gdn_physical_conv_host(layer),"prefill conv state");
        require(candidate->oscar_live_state_host_for_test()==ctx->oscar_live_state_host_for_test(),"prefill OSCAR state");
    };
    compare();
    auto teacher=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(teacher.size()>=520,"prefill teacher extent");
    for(int i=16;i<520;++i) {
        ctx->decode(teacher[i]);candidate->decode(teacher[i]);
        if(i==16 || i==511 || i==519) compare();
    }
    std::ofstream state(check.directory / "state.txt");
    state << "PASS positions=16,17,512,520 logits,taps,GDN,physical_conv,OSCAR\n";
    require(state.good(),"prefill state evidence write");
    std::cout << "INITIAL16_ORDERED PASS operators=276 elements=" << check.total_elements << "\n";
}
