#pragma once

void run_exact_attention_gqa_triple_t3(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose,
    const std::vector<std::int64_t>& structured,const std::vector<std::int64_t>& heldout) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;constexpr int prefix=4096;
    require(target.max_context()==4352,"exact attention GQA-triple T3 context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");_putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");_putenv_s("NINFER_EXL3_EXACT_ATTENTION_Q_SHARED","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_K_HALF2","1");_putenv_s("NINFER_EXL3_EXACT_ATTENTION_V_HALF2","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_PAIR","1");
    const auto greedy=[](Exl3TextContext& context){const auto logits=context.logits_host();require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),[](float value){return !std::isfinite(value);}),"exact attention GQA-triple T3 finite logits");return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());};
    const auto median=[](std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];};
    const std::array<std::pair<const char*,const std::vector<std::int64_t>*>,4> fixtures={std::pair{"code",&code},std::pair{"prose",&prose},std::pair{"structured",&structured},std::pair{"heldout_mixed",&heldout}};
    for(const auto& fixture:fixtures) {
        require(fixture.second->size()>=prefix,"exact attention GQA-triple T3 fixture extent");
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE","0");auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE","1");auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        struct Result{double prefill_ms=0;std::vector<std::int64_t> tokens;std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;};
        const auto run=[&](Exl3TextContext& context){const auto started=std::chrono::steady_clock::now();const auto request=Request::initialize(context,input,1024);const double prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();std::vector<std::int64_t> tokens;for(int row=0;row<8;++row){const auto token=greedy(context);tokens.push_back(token);context.decode(token);}return Result{prefill_ms,std::move(tokens),context.export_exact_host_state()};};
        std::vector<double> control_ms,candidate_ms;
        for(int rep=0;rep<3;++rep) {
            const std::array<bool,2> order=rep==1?std::array{true,false}:std::array{false,true};Result first,second;
            for(int ordinal=0;ordinal<2;++ordinal) {
                const bool changed=order[ordinal];const auto result=run(changed?*candidate:*control);(changed?candidate_ms:control_ms).push_back(result.prefill_ms);if(ordinal==0)first=result;else second=result;
                std::cout<<"EXACT_ATTENTION_GQA_TRIPLE_CASE fixture="<<fixture.first<<" rep="<<rep<<" order="<<ordinal<<" arm="<<(changed?"gqa_triple":"gqa_pair")<<" prefill_ms="<<result.prefill_ms<<" tps="<<prefix*1000/result.prefill_ms<<std::endl;
            }
            require(first.tokens==second.tokens&&first.state->same_payload(*second.state),"exact attention GQA-triple T3 token/state");
        }
        const double baseline=median(control_ms),changed=median(candidate_ms);
        std::cout<<"EXACT_ATTENTION_GQA_TRIPLE_T3 fixture="<<fixture.first<<" baseline_median_ms="<<baseline<<" gqa_triple_median_ms="<<changed<<" wall_reduction_percent="<<(baseline-changed)*100/baseline<<" gqa_triple_tps="<<prefix*1000/changed<<std::endl;
    }
    std::cout<<"EXACT_ATTENTION_GQA_TRIPLE_T3 PASS fixtures=4 pairs=3 prefix=4096 decode_rows=8 exact_tokens_state=1"<<std::endl;
}
