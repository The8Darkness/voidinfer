#pragma once

void run_gdn_pair_vector_io_qualification(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(target.max_context()==4352,"GDN pair vector-I/O context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_PREFILL_GDN_RESIDENT_PAIR_COLUMNS","1");
    _putenv_s("NINFER_EXL3_PREFILL_GDN_RESIDENT_QUAD_COLUMNS","0");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "GDN pair vector-I/O finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    int cases=0;
    for(const auto& fixture:std::array<
        std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        require(fixture.second->size()>=512,"GDN pair vector-I/O fixture extent");
        _putenv_s("NINFER_EXL3_PREFILL_GDN_RESIDENT_PAIR_VECTOR_IO","0");
        auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_PREFILL_GDN_RESIDENT_PAIR_VECTOR_IO","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        const auto control_coefficients=
            control->gdn_coefficient_owner_for_test(5);
        const auto candidate_coefficients=
            candidate->gdn_coefficient_owner_for_test(5);
        require(control_coefficients && candidate_coefficients &&
                    control_coefficients.get()==candidate_coefficients.get(),
                "GDN immutable coefficients were duplicated across contexts");
        require(control->gdn_recurrent_state_identity_for_test(5)!=
                    candidate->gdn_recurrent_state_identity_for_test(5) &&
                control->gdn_convolution_state_identity_for_test(5)!=
                    candidate->gdn_convolution_state_identity_for_test(5),
                "GDN mutable state was shared across contexts");
        for(const int prefix:std::array<int,3>{96,321,512}) {
            const std::vector<std::int64_t> input(
                fixture.second->begin(),fixture.second->begin()+prefix);
            struct Result {
                std::vector<std::int64_t> tokens;
                std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
            };
            const auto run=[&](Exl3TextContext& context) {
                const auto request=Request::initialize(context,input,1024);
                std::vector<std::int64_t> tokens;
                for(int row=0;row<8;++row) {
                    const auto token=greedy(context);
                    tokens.push_back(token);
                    context.decode(token);
                }
                return Result{std::move(tokens),context.export_exact_host_state()};
            };
            const auto baseline=run(*control),changed=run(*candidate);
            require(baseline.tokens==changed.tokens&&
                baseline.state->same_payload(*changed.state),
                "GDN pair vector-I/O token/state fixture="+
                std::string(fixture.first)+" prefix="+std::to_string(prefix));
            std::cout<<"GDN_PAIR_VECTOR_IO_CASE fixture="<<fixture.first
                <<" prefix="<<prefix<<" decode_rows=8 exact_tokens_state=1"
                <<std::endl;
            ++cases;
        }
    }
    std::cout<<"GDN_PAIR_VECTOR_IO PASS cases="<<cases
        <<" fixtures=2 prefixes=96,321,512 decode_rows=8 reset_between_prefixes=1"
        <<" exact_tokens_state=1 control=pair_scalar_io candidate=pair_vector_io"
        <<std::endl;
}

void run_gdn_pair_vector_io_route_screen(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(target.max_context()==4352,"GDN pair vector-I/O route context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_PREFILL_GDN_RESIDENT_PAIR_COLUMNS","1");
    _putenv_s("NINFER_EXL3_PREFILL_GDN_RESIDENT_QUAD_COLUMNS","0");
    const int measured_pairs=env_int("NINFER_T25_MEASURED_PAIRS",4);
    require(measured_pairs>=1&&measured_pairs<=4,
        "GDN pair vector-I/O route measured pair extent");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "GDN pair vector-I/O route finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    struct Result {
        double prefill_ms=0;
        double route_ms=0;
        std::vector<std::int64_t> tokens;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
    };
    constexpr int prefix=4096;
    for(const auto& fixture:std::array<
        std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        require(fixture.second->size()>=prefix,
            "GDN pair vector-I/O route fixture extent");
        const std::vector<std::int64_t> input(
            fixture.second->begin(),fixture.second->begin()+prefix);
        _putenv_s("NINFER_EXL3_PREFILL_GDN_RESIDENT_PAIR_VECTOR_IO","0");
        auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_PREFILL_GDN_RESIDENT_PAIR_VECTOR_IO","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        const auto run=[&](Exl3TextContext& context) {
            const auto route_started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-route_started).count();
            std::vector<std::int64_t> tokens;
            for(int row=0;row<8;++row) {
                const auto token=greedy(context);
                tokens.push_back(token);
                context.decode(token);
            }
            const double route_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-route_started).count();
            return Result{prefill_ms,route_ms,std::move(tokens),
                context.export_exact_host_state()};
        };
        const auto warm_control=run(*control),warm_candidate=run(*candidate);
        require(warm_control.tokens==warm_candidate.tokens&&
            warm_control.state->same_payload(*warm_candidate.state),
            "GDN pair vector-I/O route warmup token/state");
        std::cout<<"GDN_PAIR_VECTOR_IO_ROUTE_WARMUP fixture="<<fixture.first
            <<" prefix="<<prefix
            <<" control_prefill_ms="<<warm_control.prefill_ms
            <<" candidate_prefill_ms="<<warm_candidate.prefill_ms
            <<" control_route_ms="<<warm_control.route_ms
            <<" candidate_route_ms="<<warm_candidate.route_ms<<std::endl;
        for(int rep=0;rep<measured_pairs;++rep) {
            const std::array<bool,2> order=(rep&1)?
                std::array{true,false}:std::array{false,true};
            Result first,second;
            for(int ordinal=0;ordinal<2;++ordinal) {
                const bool changed=order[ordinal];
                const auto result=run(changed?*candidate:*control);
                if(ordinal==0) first=result;else second=result;
                std::cout<<"GDN_PAIR_VECTOR_IO_ROUTE_CASE fixture="
                    <<fixture.first<<" prefix="<<prefix<<" rep="<<rep
                    <<" order="<<ordinal<<" arm="
                    <<(changed?"vector":"scalar")
                    <<" prefill_ms="<<result.prefill_ms
                    <<" route_ms="<<result.route_ms
                    <<" prefill_tps="<<prefix*1000/result.prefill_ms
                    <<std::endl;
            }
            require(first.tokens==second.tokens&&
                first.state->same_payload(*second.state),
                "GDN pair vector-I/O route measured token/state");
        }
    }
    std::cout<<"GDN_PAIR_VECTOR_IO_ROUTE_SCREEN PASS fixtures=2 prefix=4096"
        <<" warmup_pairs=1 measured_pairs="<<measured_pairs
        <<" decode_rows=8 exact_tokens_state=1"
        <<" control=pair_scalar_io candidate=pair_vector_io"<<std::endl;
}
