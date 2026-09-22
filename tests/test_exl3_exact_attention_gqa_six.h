#pragma once

void run_exact_attention_gqa_six_screen(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose,
    bool score_only=false,bool candidate_default=false,bool values_sharded=false) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    const int prefix=target.max_context()>=4352?4096:96;
    require(code.size()>=prefix&&prose.size()>=prefix,"exact attention GQA-six fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");_putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE","");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_SOFTMAX_STAGED","");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SCORES","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    if(values_sharded) {
        _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SCORES","1");
        _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","1");
    } else if(score_only&&candidate_default)
        _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SCORES","");
    else
        _putenv_s(score_only?"NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SCORES":"NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context){const auto logits=context.logits_host();require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),[](float value){return !std::isfinite(value);}),"exact attention GQA-six finite logits");return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());};
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},std::pair{"prose",&prose}}){
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        const auto run=[&](Exl3TextContext& context){const auto started=std::chrono::steady_clock::now();const auto request=Request::initialize(context,input,1024);const double prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();std::vector<std::int64_t> tokens;for(int row=0;row<8;++row){const auto token=greedy(context);tokens.push_back(token);context.decode(token);}return std::tuple{prefill_ms,tokens,context.export_exact_host_state()};};
        const auto baseline=run(*control),changed=run(*candidate);require(std::get<1>(baseline)==std::get<1>(changed)&&std::get<2>(baseline)->same_payload(*std::get<2>(changed)),"exact attention GQA-six token/state");
        const char* label=values_sharded?"EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED":
            (score_only?"EXACT_ATTENTION_GQA_SIX_SCORES":"EXACT_ATTENTION_GQA_SIX");
        std::cout<<label<<" fixture="<<fixture.first<<" prefix="<<prefix<<" decode_rows=8 control_prefill_ms="<<std::get<0>(baseline)<<" candidate_prefill_ms="<<std::get<0>(changed)<<" control_tps="<<prefix*1000/std::get<0>(baseline)<<" candidate_tps="<<prefix*1000/std::get<0>(changed)<<std::endl;
    }
    const char* label=values_sharded?"EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED":
        (score_only?"EXACT_ATTENTION_GQA_SIX_SCORES":"EXACT_ATTENTION_GQA_SIX");
    const char* candidate_name=values_sharded?"gqa_six_scores_sharded_values":
        (score_only?"gqa_six_scores_triple_values":"gqa_six_staged");
    std::cout<<label<<" PASS fixtures=2 prefix="<<prefix<<" decode_rows=8 exact_tokens_state=1 control=gqa_triple_staged candidate="<<candidate_name<<" default_candidate="<<(candidate_default?1:0)<<std::endl;
}
