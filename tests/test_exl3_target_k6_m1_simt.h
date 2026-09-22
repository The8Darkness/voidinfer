#pragma once

struct TargetK6M1SimtQualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    std::filesystem::path directory;
    std::ofstream csv;
    std::vector<std::string> shapes;

    explicit TargetK6M1SimtQualification(const std::filesystem::path& path)
        : directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),
                "target K6 M1 SIMT output must be new");
        std::filesystem::create_directories(path);
        csv.open(path / "operators.csv");
        csv << "operator,in_features,out_features,elements,exact,guards,dispatch,calls\n";
    }

    static Admission admission(const char* operation) noexcept {
        const std::string op(operation ? operation : "");
        if (op == "q") return Admission::target_continuation_q;
        if (op == "qkv") return Admission::target_continuation_qkv;
        if (op == "z") return Admission::target_continuation_z;
        if (op == "k" || op == "v") return Admission::target_continuation_kv;
        if (op == "o") return Admission::target_continuation_o;
        return Admission::ordinary;
    }

    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,
                         void* user) {
        auto* self=static_cast<TargetK6M1SimtQualification*>(user);
        const auto admit=admission(x.operation);
        if(x.metadata.K!=6 || admit==Admission::ordinary)return;
        const std::string key=std::string(x.operation)+"-"+
            std::to_string(x.metadata.in_features)+"x"+
            std::to_string(x.metadata.out_features);
        if(std::find(self->shapes.begin(),self->shapes.end(),key)!=
           self->shapes.end())return;
        self->check(x,key,admit);
        self->shapes.push_back(key);
    }

    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x,
               const std::string& key,Admission admit) {
        require(x.rows==8 && x.metadata.mul1 && !x.metadata.mcg &&
                    !x.metadata.has_bias,
                "target K6 M1 SIMT capture contract");
        _putenv_s("NINFER_EXL3_TARGET_K6_M1_SIMT","0");
        Exl3CudaLinearWorkspace baseline(
            x.metadata.in_features,x.metadata.out_features,8,
            false,true,true,true,true,false,false,{}, {},true,true,true);
        _putenv_s("NINFER_EXL3_TARGET_K6_M1_SIMT","1");
        Exl3CudaLinearWorkspace candidate(
            x.metadata.in_features,x.metadata.out_features,8,
            false,true,true,true,true,false,false,{}, {},true,true,true);
        _putenv_s("NINFER_EXL3_TARGET_K6_M1_SIMT","0");
        require(candidate.target_k6_m1_simt_candidate(x.metadata,1,admit) &&
                    !candidate.target_k6_m1_simt_candidate(x.metadata,2,admit) &&
                    !candidate.target_k6_m1_simt_candidate(
                        x.metadata,1,Admission::ordinary),
                "target K6 M1 SIMT width/owner admission");
        require(std::string(candidate.dispatch_name(x.metadata,1,admit))==
                    "target_k6_m1_simt",
                "target K6 M1 SIMT dispatch identity");

        constexpr std::size_t guard=128;
        const std::size_t extent=x.metadata.out_features;
        DeviceBuffer reference_device((extent+2*guard)*sizeof(std::uint16_t));
        DeviceBuffer candidate_device((extent+2*guard)*sizeof(std::uint16_t));
        auto* reference_output=
            static_cast<std::uint16_t*>(reference_device.get())+guard;
        auto* candidate_output=
            static_cast<std::uint16_t*>(candidate_device.get())+guard;
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555);
        std::vector<std::uint16_t> reference(initial),actual(initial);
        cuda_check(cudaMemcpy(reference_device.get(),initial.data(),
                              initial.size()*sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target K6 M1 SIMT reference guards");
        cuda_check(cudaMemcpy(candidate_device.get(),initial.data(),
                              initial.size()*sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target K6 M1 SIMT candidate guards");
        baseline.forward(x.weights,x.metadata,x.input,reference_output,1,x.stream,
                         admit);
        candidate.forward(x.weights,x.metadata,x.input,candidate_output,1,x.stream,
                          admit);
        cuda_check(cudaStreamSynchronize(x.stream),
                   "target K6 M1 SIMT exact ready");
        cuda_check(cudaMemcpy(reference.data(),reference_device.get(),
                              reference.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "target K6 M1 SIMT reference copy");
        cuda_check(cudaMemcpy(actual.data(),candidate_device.get(),
                              actual.size()*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "target K6 M1 SIMT candidate copy");
        require(reference==actual,"target K6 M1 SIMT exact mismatch "+key);
        for(std::size_t i=0;i<actual.size();++i)
            if(i<guard || i>=guard+extent)
                require(actual[i]==0x3555,
                        "target K6 M1 SIMT inactive output/canary");
        require(candidate.target_k6_m1_simt_calls()==1,
                "target K6 M1 SIMT dispatch count");
        csv << x.operation << ',' << x.metadata.in_features << ','
            << x.metadata.out_features << ',' << extent
            << ",1,1," << candidate.dispatch_name(x.metadata,1,admit)
            << ',' << candidate.target_k6_m1_simt_calls() << '\n';
        csv.flush();
        require(csv.good(),"target K6 M1 SIMT result write");
    }
};

void run_target_k6_m1_simt_qualification(Exl3TextModel& target,
                                         Exl3Dflash2DraftModel& draft) {
    require(env("NINFER_EXL3_TARGET_K6_M1_SIMT")=="0",
            "target K6 M1 SIMT capture requires construction flag0");
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size()>=512,"target K6 M1 SIMT prompt extent");
    std::vector<std::int64_t> prompt(ids.begin(),ids.begin()+512);
    TapStage stage;
    for(int tap=0;tap<kTapCount;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16u*kHidden*2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        stage.bulk_ptrs.push_back(
            static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(
            static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    auto context=target.create_context(true);
    draft.reset();
    context->prefill(std::span<const std::int64_t>(prompt.data(),16));
    commit_current_rows(*context,draft,stage,16,0);
    int cursor=16;
    while(cursor<static_cast<int>(prompt.size())) {
        const int remaining=static_cast<int>(prompt.size())-cursor;
        const int count=remaining>=128?128:16;
        context->append_exact_prefill_wide(
            std::span<const std::int64_t>(prompt.data()+cursor,count));
        commit_current_rows(*context,draft,stage,count,cursor);
        cursor+=count;
    }
    const auto pending=sample_target(*context);
    std::vector<std::int64_t> block(8,kMaskToken);
    block[0]=pending;
    const auto proposals=draft.propose_cached(
        block,512,context->target_embedding(),context->target_lm_head_weights(),
        context->target_lm_head_metadata(),kMaskToken);
    require(proposals.size()==7,"target K6 M1 SIMT proposal extent");
    block.assign(1,pending);
    block.insert(block.end(),proposals.begin(),proposals.end());

    TargetK6M1SimtQualification check(env("NINFER_E5A4_OUT"));
    context->prepare_transaction();
    context->prepare_continuation(8);
    context->set_target_projection_observer_for_test(
        TargetK6M1SimtQualification::callback,&check,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->begin_transaction();
    context->continue_rows(block);
    context->set_target_projection_observer_for_test(nullptr,nullptr);
    context->rollback_transaction();
    cuda_check(cudaDeviceSynchronize(),"target K6 M1 SIMT sweep ready");
    require(check.shapes.size()>=3,
            "target K6 M1 SIMT real caller shape coverage");
    std::ofstream result(check.directory/"result.txt");
    result << "PASS shapes=" << check.shapes.size()
           << " rows=1 bit_exact=1 guards=1 dispatch_counted=1\n";
    require(result.good(),"target K6 M1 SIMT summary write");
    std::cout << "TARGET_K6_M1_SIMT PASS shapes=" << check.shapes.size()
              << "\n";
}
