#pragma once

struct T71BAccumulatedMetric {
    double error_squared=0.0;
    double reference_squared=0.0;
    double max_abs=0.0;
    std::uint64_t values=0;

    void add(const std::vector<float>& actual,const std::vector<float>& reference) {
        require(actual.size()==reference.size(),"T71B metric extent");
        for(std::size_t i=0;i<actual.size();++i) {
            require(std::isfinite(actual[i])&&std::isfinite(reference[i]),
                "T71B nonfinite logits");
            const double difference=static_cast<double>(actual[i])-reference[i];
            error_squared+=difference*difference;
            reference_squared+=static_cast<double>(reference[i])*reference[i];
            max_abs=std::max(max_abs,std::abs(difference));++values;
        }
    }
    double relative_l2() const {
        return reference_squared>0.0?std::sqrt(error_squared/reference_squared):0.0;
    }
};

void run_numeric_attention_splitk_route_t71b(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    require(target.max_context()==4352&&code.size()>=4096&&prose.size()>=4096,
        "T71B route fixture/context extent");
    const std::filesystem::path directory=env("NINFER_T71B_OUT");
    require(!directory.empty()&&!std::filesystem::exists(directory),
        "T71B output directory must be new");
    std::filesystem::create_directories(directory);
    std::ofstream quality(directory/"quality.csv");
    require(quality.good(),"T71B quality evidence creation");
    quality<<"fixture,steps,logit_values,logit_max_abs,logit_rel_l2,kv_values,"
        "kv_max_abs,kv_rel_l2,recurrent_values,recurrent_max_abs,recurrent_rel_l2,"
        "tokens_exact,control_persistent_bytes,candidate_persistent_bytes,"
        "score_plane_bytes,partial_workspace_bytes,position_equal\n";

    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC","0");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC","0");
    constexpr int prefix=4096,steps=128;
    const std::size_t score_plane_bytes=static_cast<std::size_t>(16)*24*4352*4;
    const std::size_t partial_workspace_bytes=
        ninfer::exl3::exl3_numeric_attention_splitk_workspace_bytes(16,4352);
    const std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>
        fixtures{{{"code",&code},{"prose",&prose}}};
    for(const auto& [name,source]:fixtures) {
        _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","0");
        auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        const auto prompt=std::span<const std::int64_t>(source->data(),prefix);
        t69b_ingest(*control,prompt);t69b_ingest(*candidate,prompt);
        require(candidate->persistent_bytes()+score_plane_bytes==
                control->persistent_bytes()+partial_workspace_bytes,
            "T71B bounded workspace ownership");

        T71BAccumulatedMetric logits;
        std::vector<std::int64_t> control_tokens,candidate_tokens;
        for(int step=0;step<steps;++step) {
            const auto control_logits=control->logits_host();
            const auto candidate_logits=candidate->logits_host();
            logits.add(candidate_logits,control_logits);
            const auto control_token=static_cast<std::int64_t>(
                std::max_element(control_logits.begin(),control_logits.end())-
                control_logits.begin());
            const auto candidate_token=static_cast<std::int64_t>(
                std::max_element(candidate_logits.begin(),candidate_logits.end())-
                candidate_logits.begin());
            control_tokens.push_back(control_token);candidate_tokens.push_back(candidate_token);
            require(control_token==candidate_token,
                std::string("T71B token divergence fixture=")+name+
                " step="+std::to_string(step));
            control->decode(control_token);candidate->decode(candidate_token);
        }
        cuda_check(cudaDeviceSynchronize(),"T71B route synchronize");
        const auto control_state=control->export_exact_host_state();
        const auto candidate_state=candidate->export_exact_host_state();
        const auto kv=t69b_exact_kv_metric(candidate_state,control_state);
        T71BAccumulatedMetric recurrent;
        for(int layer=0;layer<64;++layer) if((layer+1)%4!=0)
            recurrent.add(candidate->gdn_state_host(layer),control->gdn_state_host(layer));
        const bool tokens_exact=control_tokens==candidate_tokens;
        const bool position_equal=control->position()==candidate->position()&&
            control_state->position()==candidate_state->position();
        quality<<name<<','<<steps<<','<<logits.values<<','<<logits.max_abs<<','
            <<logits.relative_l2()<<','<<kv.values<<','<<kv.max_abs<<','
            <<kv.relative_l2<<','<<recurrent.values<<','<<recurrent.max_abs<<','
            <<recurrent.relative_l2()<<','<<(tokens_exact?1:0)<<','
            <<control->persistent_bytes()<<','<<candidate->persistent_bytes()<<','
            <<score_plane_bytes<<','<<partial_workspace_bytes<<','
            <<(position_equal?1:0)<<'\n';quality.flush();
        require(logits.relative_l2()<=0.005&&logits.max_abs<=0.25&&
                kv.relative_l2<=0.005&&recurrent.relative_l2()<=0.005&&
                recurrent.max_abs<=0.25&&tokens_exact&&position_equal,
            std::string("T71B frozen 4K quality gate ")+name);
        std::cout<<"T71B_ROUTE_QUALITY fixture="<<name<<" prefix="<<prefix
            <<" steps="<<steps<<" logits_max_abs="<<logits.max_abs
            <<" logits_rel_l2="<<logits.relative_l2()<<" kv_max_abs="<<kv.max_abs
            <<" kv_rel_l2="<<kv.relative_l2<<" recurrent_max_abs="
            <<recurrent.max_abs<<" recurrent_rel_l2="<<recurrent.relative_l2()
            <<" tokens_exact=1 control_persistent_bytes="<<control->persistent_bytes()
            <<" candidate_persistent_bytes="<<candidate->persistent_bytes()
            <<" score_plane_bytes="<<score_plane_bytes
            <<" partial_workspace_bytes="<<partial_workspace_bytes<<"\n";
    }

    // Unset and explicit zero remain the bit-exact qualified route.
    _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","");
    auto unset=target.create_context(true);unset->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","0");
    auto zero=target.create_context(true);zero->prepare_continuation(8);
    const auto short_prompt=std::span<const std::int64_t>(code.data(),96);
    t69b_ingest(*unset,short_prompt);t69b_ingest(*zero,short_prompt);
    for(int row=0;row<8;++row) {
        const auto token=sample_target(*unset);
        require(token==sample_target(*zero),"T71B zero default token contract");
        unset->decode(token);zero->decode(token);
    }
    require(unset->export_exact_host_state()->same_payload(*zero->export_exact_host_state()),
        "T71B zero default state contract");

    // Reset after a partially populated candidate request and prove clean reuse.
    _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","1");
    auto reused=target.create_context(true);reused->prepare_continuation(8);
    const std::string contract="t71b;numeric-attention;exact-host;single-user";
    reused->bind_request_compatibility(contract);
    t69b_ingest(*reused,std::span<const std::int64_t>(code.data(),1040));
    const auto persistent=reused->persistent_bytes();
    const auto reset=reused->reset_for_request(contract);
    require(reused->position()==0&&reused->persistent_bytes()==persistent&&
            reset.generation==1,"T71B cancellation reset ownership");
    t69b_ingest(*reused,std::span<const std::int64_t>(prose.data(),96));
    auto fresh=target.create_context(true);fresh->prepare_continuation(8);
    t69b_ingest(*fresh,std::span<const std::int64_t>(prose.data(),96));
    for(int row=0;row<8;++row) {
        const auto token=sample_target(*reused);
        require(token==sample_target(*fresh),"T71B reset deterministic token");
        reused->decode(token);fresh->decode(token);
    }
    require(reused->export_exact_host_state()->same_payload(*fresh->export_exact_host_state()),
        "T71B reset contaminated candidate state");
    std::cout<<"T71B_ROUTE_GATE PASS fixtures=2 prefix=4096 steps=128 "
        "tokens_exact=1 numeric_gates=1 default_zero_exact=1 cancellation=1 "
        "production_changed=0 lane=NUMERIC_CANDIDATE\n";
}
