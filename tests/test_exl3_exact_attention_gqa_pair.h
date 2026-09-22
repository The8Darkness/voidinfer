#pragma once

void run_exact_attention_gqa_pair_screen(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    const int prefix=target.max_context()>=4352?4096:96;
    require(code.size()>=prefix&&prose.size()>=prefix,
        "exact attention GQA-pair fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_Q_SHARED","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_K_HALF2","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_V_HALF2","1");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "exact attention GQA-pair finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_PAIR","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_PAIR","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
        std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        const auto run=[&](Exl3TextContext& context) {
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            std::vector<std::int64_t> tokens;
            for(int row=0;row<8;++row) {const auto token=greedy(context);tokens.push_back(token);context.decode(token);}
            return std::tuple{prefill_ms,tokens,context.export_exact_host_state()};
        };
        const auto baseline=run(*control),changed=run(*candidate);
        require(std::get<1>(baseline)==std::get<1>(changed)&&
            std::get<2>(baseline)->same_payload(*std::get<2>(changed)),
            "exact attention GQA-pair token/state");
        std::cout<<"EXACT_ATTENTION_GQA_PAIR fixture="<<fixture.first<<" prefix="<<prefix
            <<" decode_rows=8 control_prefill_ms="<<std::get<0>(baseline)
            <<" candidate_prefill_ms="<<std::get<0>(changed)
            <<" control_tps="<<prefix*1000/std::get<0>(baseline)
            <<" candidate_tps="<<prefix*1000/std::get<0>(changed)<<std::endl;
    }
    std::cout<<"EXACT_ATTENTION_GQA_PAIR PASS fixtures=2 prefix="<<prefix
        <<" decode_rows=8 exact_tokens_state=1"<<std::endl;
}
