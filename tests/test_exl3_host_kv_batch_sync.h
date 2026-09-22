#pragma once

void run_exact_host_kv_batch_sync_screen(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()==4352 && code.size()>=prefix && prose.size()>=prefix,
        "host KV batch-sync fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_SYNC","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_SYNC","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"host KV batch-sync finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        const auto run=[&](Exl3TextContext& context) {
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            const auto token=greedy(context);context.decode(token);
            const auto state=context.export_exact_host_state();
            return std::tuple{prefill_ms,token,state,context.host_kv_stats()};
        };
        const auto baseline=run(*control);
        const auto changed=run(*candidate);
        require(std::get<1>(baseline)==std::get<1>(changed) &&
            std::get<2>(baseline)->same_payload(*std::get<2>(changed)),
            "host KV batch-sync token/state");
        const auto a=std::get<3>(baseline),b=std::get<3>(changed);
        require(a.h2d_bytes==b.h2d_bytes && a.d2h_bytes==b.d2h_bytes &&
            a.transfer_calls==b.transfer_calls && a.completed_rows==b.completed_rows &&
            a.completed_rows%(prefix+1)==0,"host KV batch-sync transfer accounting");
        std::cout << "EXACT_HOST_KV_BATCH_SYNC fixture=" << fixture.first
            << " prefix=4096 control_ms=" << std::get<0>(baseline)
            << " candidate_ms=" << std::get<0>(changed)
            << " control_tps=" << prefix*1000/std::get<0>(baseline)
            << " candidate_tps=" << prefix*1000/std::get<0>(changed)
            << " h2d_bytes=" << a.h2d_bytes << " d2h_bytes=" << a.d2h_bytes
            << " transfer_calls=" << a.transfer_calls << '\n';
    }
    std::cout << "EXACT_HOST_KV_BATCH_SYNC PASS fixtures=2 prefix=4096"
        << " exact_state=1 transfer_accounting=1 final_barrier=1\n";
}
