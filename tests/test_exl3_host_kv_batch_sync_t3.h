#pragma once

void run_exact_host_kv_batch_sync_t3(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose,
    const std::vector<std::int64_t>& structured,const std::vector<std::int64_t>& heldout) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()==4352,"host KV batch-sync T3 context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"host KV batch-sync T3 finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const auto median=[](std::vector<double> values) {
        std::sort(values.begin(),values.end());return values[values.size()/2];
    };
    const std::array<std::pair<const char*,const std::vector<std::int64_t>*>,4> fixtures={
        std::pair{"code",&code},std::pair{"prose",&prose},
        std::pair{"structured",&structured},std::pair{"heldout_mixed",&heldout}};
    for(const auto& fixture:fixtures) {
        require(fixture.second->size()>=prefix,"host KV batch-sync T3 fixture extent");
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_SYNC","0");
        auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_SYNC","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        struct Result {
            double prefill_ms=0;
            std::int64_t token=0;
            std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
            ninfer::exl3::Exl3HostKVStats delta;
        };
        const auto run=[&](Exl3TextContext& context) {
            const auto before=context.host_kv_stats();
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            const auto token=greedy(context);context.decode(token);
            const auto state=context.export_exact_host_state();
            const auto after=context.host_kv_stats();
            ninfer::exl3::Exl3HostKVStats delta=after;
            delta.h2d_bytes-=before.h2d_bytes;delta.d2h_bytes-=before.d2h_bytes;
            delta.transfer_calls-=before.transfer_calls;delta.completed_rows-=before.completed_rows;
            return Result{prefill_ms,token,state,delta};
        };
        std::vector<double> control_ms,candidate_ms;
        for(int rep=0;rep<3;++rep) {
            const std::array<bool,2> order=rep==1?std::array{true,false}:std::array{false,true};
            Result first,second;
            for(int ordinal=0;ordinal<2;++ordinal) {
                const bool changed=order[ordinal];
                const auto result=run(changed?*candidate:*control);
                (changed?candidate_ms:control_ms).push_back(result.prefill_ms);
                if(ordinal==0) first=result; else second=result;
                std::cout << "EXACT_HOST_KV_BATCH_SYNC_CASE fixture=" << fixture.first
                    << " rep=" << rep << " order=" << ordinal
                    << " arm=" << (changed?"final_barrier":"per_page")
                    << " prefill_ms=" << result.prefill_ms
                    << " tps=" << prefix*1000/result.prefill_ms << std::endl;
            }
            require(first.token==second.token && first.state->same_payload(*second.state),
                "host KV batch-sync T3 token/state");
            require(first.delta.h2d_bytes==second.delta.h2d_bytes &&
                first.delta.d2h_bytes==second.delta.d2h_bytes &&
                first.delta.transfer_calls==second.delta.transfer_calls &&
                first.delta.completed_rows==second.delta.completed_rows &&
                first.delta.completed_rows==prefix+1,
                "host KV batch-sync T3 transfer accounting");
        }
        const double baseline=median(control_ms),changed=median(candidate_ms);
        require(changed<baseline,"host KV batch-sync T3 median must improve");
        std::cout << "EXACT_HOST_KV_BATCH_SYNC_T3 fixture=" << fixture.first
            << " per_page_median_ms=" << baseline
            << " final_barrier_median_ms=" << changed
            << " wall_reduction_percent=" << (baseline-changed)*100/baseline
            << " final_barrier_tps=" << prefix*1000/changed << std::endl;
    }
    std::cout << "EXACT_HOST_KV_BATCH_SYNC_T3 PASS fixtures=4 pairs=3 prefix=4096"
        << " exact_state=1 transfer_accounting=1 all_medians_positive=1" << std::endl;
}
