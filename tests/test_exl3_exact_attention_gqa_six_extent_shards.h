#pragma once

void run_exact_attention_gqa_six_extent_shards_screen(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(target.max_context()==4352,
        "GQA-six extent-shards screen context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE","");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_SOFTMAX_STAGED","");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SCORES","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_PACKED_TRIPLES","0");
    const int measured_pairs=env_int("NINFER_T23_MEASURED_PAIRS",4);
    const bool candidate_uses_default=env_int("NINFER_T23_CANDIDATE_DEFAULT",0)==1;
    require(measured_pairs>=1&&measured_pairs<=4,
        "GQA-six extent-shards measured pair extent");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "GQA-six extent-shards finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    struct Result {
        double prefill_ms=0;
        double route_ms=0;
        std::vector<std::int64_t> tokens;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
    };
    for(const int prefix:std::array<int,2>{512,4096}) {
        for(const auto& fixture:std::array<
            std::pair<const char*,const std::vector<std::int64_t>*>,2>{
                std::pair{"code",&code},std::pair{"prose",&prose}}) {
            require(fixture.second->size()>=prefix,
                "GQA-six extent-shards fixture extent");
            const std::vector<std::int64_t> input(
                fixture.second->begin(),fixture.second->begin()+prefix);
            _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_EXTENT_SHARDS","0");
            auto control=target.create_context(true);control->prepare_continuation(8);
            _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_EXTENT_SHARDS",
                candidate_uses_default?"":"1");
            auto candidate=target.create_context(true);candidate->prepare_continuation(8);
            const auto run=[&](Exl3TextContext& context) {
                const auto route_started=std::chrono::steady_clock::now();
                const auto prefill_started=route_started;
                const auto request=Request::initialize(context,input,1024);
                const double prefill_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-prefill_started).count();
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
                "GQA-six extent-shards warmup token/state");
            std::cout<<"EXACT_ATTENTION_GQA_SIX_EXTENT_SHARDS_WARMUP fixture="
                <<fixture.first<<" prefix="<<prefix
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
                    std::cout<<"EXACT_ATTENTION_GQA_SIX_EXTENT_SHARDS_CASE fixture="
                        <<fixture.first<<" prefix="<<prefix<<" rep="<<rep
                        <<" order="<<ordinal<<" arm="
                        <<(changed?"extent":"fixed6")
                        <<" prefill_ms="<<result.prefill_ms
                        <<" route_ms="<<result.route_ms
                        <<" prefill_tps="<<prefix*1000/result.prefill_ms
                        <<std::endl;
                }
                require(first.tokens==second.tokens&&
                    first.state->same_payload(*second.state),
                    "GQA-six extent-shards measured token/state");
            }
        }
    }
    std::cout<<"EXACT_ATTENTION_GQA_SIX_EXTENT_SHARDS_SCREEN PASS fixtures=2"
        <<" prefixes=512,4096 warmup_pairs=1 measured_pairs="<<measured_pairs
        <<" decode_rows=8"
        <<" exact_tokens_state=1 control=fixed6 candidate=public_extent_shards"
        <<" candidate_setting="<<(candidate_uses_default?"omitted":"explicit1")
        <<std::endl;
}
