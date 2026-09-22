#pragma once

void run_exact_host_kv_batch_copy_screen(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose,bool metadata=false) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    const int prefix=target.max_context()>=4352?4096:96;
    const int forced_rows=metadata?64:0;
    require(code.size()>=prefix && prose.size()>=prefix,
        "host KV batch-copy fixture/context extent");
    require(target.max_context()>=prefix+1+forced_rows,
        "HostKV repeated metadata page-boundary fixture lacks context capacity");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_SYNC","1");
    if(metadata) {
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_CHUNKS","0");
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","0");
    }
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_COPY","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_COPY","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"host KV batch-copy finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        const auto run=[&](Exl3TextContext& context) {
            const auto before=context.host_kv_stats();
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            const auto token=greedy(context);context.decode(token);
            // Both routes consume the same frozen continuation. Cross at least
            // one complete page boundary without growing descriptor storage.
            for(int row=0;row<forced_rows;++row)context.decode(input[row]);
            const auto state=context.export_exact_host_state();
            const auto after=context.host_kv_stats();
            require(after.batch_metadata_bytes==before.batch_metadata_bytes,
                "HostKV page-boundary reuse grew prepared descriptor storage");
            ninfer::exl3::Exl3HostKVStats delta=after;
            delta.h2d_bytes-=before.h2d_bytes;delta.d2h_bytes-=before.d2h_bytes;
            delta.transfer_calls-=before.transfer_calls;
            delta.copy_submissions-=before.copy_submissions;
            delta.completed_rows-=before.completed_rows;
            delta.batch_metadata_reuses-=before.batch_metadata_reuses;
            delta.batch_metadata_reuse_drains-=before.batch_metadata_reuse_drains;
            return std::tuple{prefill_ms,token,state,delta};
        };
        const auto baseline=run(*control);
        const auto changed=run(*candidate);
        require(std::get<1>(baseline)==std::get<1>(changed) &&
            std::get<2>(baseline)->same_payload(*std::get<2>(changed)),
            "host KV batch-copy token/state");
        const auto a=std::get<3>(baseline),b=std::get<3>(changed);
        if(metadata) require(b.batch_metadata_bytes>0 && b.batch_metadata_reuses>1,
            "prepared HostKV metadata was not reused");
        if(metadata) require(a.batch_metadata_reuse_drains==0 && b.batch_metadata_reuse_drains>0 &&
            b.batch_metadata_reuse_drains<=b.batch_metadata_reuses,
            "HostKV descriptor reuse did not fence pending batch submissions");
        require(a.h2d_bytes==b.h2d_bytes && a.d2h_bytes==b.d2h_bytes &&
            a.transfer_calls==b.transfer_calls && a.completed_rows==b.completed_rows &&
            a.completed_rows==prefix+1+forced_rows,"host KV batch-copy logical accounting");
        require(std::get<2>(baseline)->position()==prefix+1+forced_rows &&
            std::get<2>(changed)->position()==prefix+1+forced_rows,
            "HostKV page-boundary fixture did not consume every frozen row");
        std::cout << "EXACT_HOST_KV_BATCH_COPY fixture=" << fixture.first
            << " prefix=" << prefix
            << " forced_rows=" << forced_rows
            << " control_ms=" << std::get<0>(baseline)
            << " candidate_ms=" << std::get<0>(changed)
            << " control_tps=" << prefix*1000/std::get<0>(baseline)
            << " candidate_tps=" << prefix*1000/std::get<0>(changed)
            << " logical_transfers=" << a.transfer_calls
            << " control_submissions=" << a.copy_submissions
            << " candidate_submissions=" << b.copy_submissions << std::endl;
        require(a.copy_submissions==a.transfer_calls && b.copy_submissions<a.copy_submissions,
            "host KV batch-copy submission reduction");
    }
    std::cout << "EXACT_HOST_KV_BATCH_COPY PASS fixtures=2 prefix=" << prefix
        << " exact_state=1 logical_accounting=1 submission_reduction=1" << std::endl;
}
