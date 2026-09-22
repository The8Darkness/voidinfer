#pragma once

inline void configure_exact_host_kv_pinned_d2h() {
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_SYNC","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BATCH_COPY","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_CHUNKS","1");
}

void run_exact_host_kv_pinned_d2h_screen(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,bool candidate_default=false) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    const int prefix=target.max_context()>=4352?4096:96;
    require(code.size()>=prefix&&prose.size()>=prefix,
        "host KV pinned D2H fixture/context extent");
    configure_exact_host_kv_pinned_d2h();
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    if(candidate_default)_putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","");
    else _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "host KV pinned D2H finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    for(const auto& fixture:std::array<std::pair<const char*,
        const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},
        std::pair{"prose",&prose}}) {
        const std::vector<std::int64_t> input(
            fixture.second->begin(),fixture.second->begin()+prefix);
        const auto run=[&](Exl3TextContext& context) {
            const auto before=context.host_kv_stats();
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            std::vector<std::int64_t> tokens;
            for(int row=0;row<8;++row) {
                const auto token=greedy(context);tokens.push_back(token);context.decode(token);
            }
            const auto after=context.host_kv_stats();auto delta=after;
            delta.h2d_bytes-=before.h2d_bytes;delta.d2h_bytes-=before.d2h_bytes;
            delta.transfer_calls-=before.transfer_calls;
            delta.copy_submissions-=before.copy_submissions;
            delta.completed_rows-=before.completed_rows;
            return std::tuple{prefill_ms,tokens,context.export_exact_host_state(),delta};
        };
        const auto baseline=run(*control),changed=run(*candidate);
        require(std::get<1>(baseline)==std::get<1>(changed) &&
            std::get<2>(baseline)->same_payload(*std::get<2>(changed)),
            "host KV pinned D2H token/state");
        const auto a=std::get<3>(baseline),b=std::get<3>(changed);
        require(a.h2d_bytes==b.h2d_bytes&&a.d2h_bytes==b.d2h_bytes&&
            a.transfer_calls==b.transfer_calls&&a.completed_rows==b.completed_rows&&
            a.pinned_staging_bytes==b.pinned_staging_bytes&&
            b.pinned_staging_bytes==2*(1U<<20),
            "host KV pinned D2H logical/fixed-resource accounting");
        std::cout<<"EXACT_HOST_KV_PINNED_D2H fixture="<<fixture.first
            <<" prefix="<<prefix<<" decode_rows=8 control_prefill_ms="
            <<std::get<0>(baseline)<<" candidate_prefill_ms="<<std::get<0>(changed)
            <<" control_submissions="<<a.copy_submissions
            <<" candidate_submissions="<<b.copy_submissions
            <<" pinned_bytes="<<b.pinned_staging_bytes<<std::endl;
    }
    std::cout<<"EXACT_HOST_KV_PINNED_D2H PASS fixtures=2 prefix="<<prefix
        <<" decode_rows=8 exact_tokens_state=1 logical_accounting=1"
        <<" event_scatter_before_publication=1 bounded_pinned_bytes="<<2*(1U<<20)
        <<" default_candidate="<<(candidate_default?1:0)
        <<std::endl;
}

void run_exact_host_kv_pinned_d2h_t3(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::vector<std::int64_t>& structured,
    const std::vector<std::int64_t>& heldout,
    bool deferred_scatter=false) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()==4352,"host KV pinned D2H T3 context extent");
    configure_exact_host_kv_pinned_d2h();
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "host KV pinned D2H T3 finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const auto median=[](std::vector<double> values) {
        std::sort(values.begin(),values.end());return values[values.size()/2];
    };
    const std::array<std::pair<const char*,const std::vector<std::int64_t>*>,4>
        fixtures={std::pair{"code",&code},std::pair{"prose",&prose},
            std::pair{"structured",&structured},std::pair{"heldout_mixed",&heldout}};
    for(const auto& fixture:fixtures) {
        require(fixture.second->size()>=prefix,"host KV pinned D2H T3 fixture extent");
        const std::vector<std::int64_t> input(
            fixture.second->begin(),fixture.second->begin()+prefix);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H",
            deferred_scatter?"1":"0");
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER","0");
        auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","1");
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER",
            deferred_scatter?"1":"0");
        auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        struct Result {
            double prefill_ms=0;
            std::vector<std::int64_t> tokens;
            std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
            ninfer::exl3::Exl3HostKVStats delta;
        };
        const auto run=[&](Exl3TextContext& context) {
            const auto before=context.host_kv_stats();
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            std::vector<std::int64_t> tokens;
            for(int row=0;row<8;++row) {
                const auto token=greedy(context);tokens.push_back(token);context.decode(token);
            }
            const auto after=context.host_kv_stats();auto delta=after;
            delta.h2d_bytes-=before.h2d_bytes;delta.d2h_bytes-=before.d2h_bytes;
            delta.transfer_calls-=before.transfer_calls;
            delta.copy_submissions-=before.copy_submissions;
            delta.completed_rows-=before.completed_rows;
            delta.pinned_slot_waits-=before.pinned_slot_waits;
            delta.pinned_deferred_drains-=before.pinned_deferred_drains;
            delta.pinned_scatter_bytes-=before.pinned_scatter_bytes;
            delta.pinned_plane_tail_deferrals-=before.pinned_plane_tail_deferrals;
            delta.pinned_publication_drains-=before.pinned_publication_drains;
            return Result{prefill_ms,std::move(tokens),context.export_exact_host_state(),delta};
        };
        std::vector<double> control_ms,candidate_ms;
        for(int rep=0;rep<3;++rep) {
            const std::array<bool,2> order=rep==1?std::array{true,false}:
                std::array{false,true};
            Result first,second,immediate_result,deferred_result;
            for(int ordinal=0;ordinal<2;++ordinal) {
                const bool changed=order[ordinal];
                const auto result=run(changed?*candidate:*control);
                (changed?candidate_ms:control_ms).push_back(result.prefill_ms);
                if(changed) deferred_result=result;else immediate_result=result;
                if(ordinal==0)first=result;else second=result;
                std::cout<<"EXACT_HOST_KV_PINNED_D2H_CASE fixture="<<fixture.first
                    <<" rep="<<rep<<" order="<<ordinal<<" arm="
                    <<(deferred_scatter
                        ?(changed?"deferred_scatter":"immediate_scatter")
                        :(changed?"pinned_d2h":"batch_d2h"))
                    <<" prefill_ms="<<result.prefill_ms
                    <<" tps="<<prefix*1000/result.prefill_ms
                    <<" submissions="<<result.delta.copy_submissions<<std::endl;
            }
            require(first.tokens==second.tokens&&first.state->same_payload(*second.state),
                "host KV pinned D2H T3 token/state");
            require(first.delta.h2d_bytes==second.delta.h2d_bytes&&
                first.delta.d2h_bytes==second.delta.d2h_bytes&&
                first.delta.transfer_calls==second.delta.transfer_calls&&
                first.delta.completed_rows==second.delta.completed_rows&&
                first.delta.pinned_staging_bytes==second.delta.pinned_staging_bytes,
                "host KV pinned D2H T3 logical/fixed-resource accounting");
            if(deferred_scatter) require(
                immediate_result.delta.copy_submissions==
                    deferred_result.delta.copy_submissions&&
                deferred_result.delta.pinned_scatter_bytes==
                    deferred_result.delta.d2h_bytes&&
                immediate_result.delta.pinned_plane_tail_deferrals==0&&
                deferred_result.delta.pinned_plane_tail_deferrals==
                    32*deferred_result.delta.pinned_publication_drains&&
                deferred_result.delta.pinned_max_pending_bytes<=2*(1U<<20),
                "deferred exact host KV scatter T3 dependency/accounting");
        }
        const double baseline=median(control_ms),changed=median(candidate_ms);
        std::cout<<(deferred_scatter?"EXACT_HOST_KV_DEFERRED_SCATTER_T3 fixture=":
                "EXACT_HOST_KV_PINNED_D2H_T3 fixture=")<<fixture.first
            <<(deferred_scatter?" immediate_median_ms=":" batch_median_ms=")
            <<baseline<<(deferred_scatter?" deferred_median_ms=":" pinned_median_ms=")<<changed
            <<" wall_reduction_percent="<<(baseline-changed)*100/baseline
            <<(deferred_scatter?" deferred_tps=":" pinned_tps=")
            <<prefix*1000/changed<<std::endl;
    }
    std::cout<<(deferred_scatter?"EXACT_HOST_KV_DEFERRED_SCATTER_T3":
            "EXACT_HOST_KV_PINNED_D2H_T3")
        <<" PASS fixtures=4 pairs=3 prefix=4096"
        <<" decode_rows=8 exact_tokens_state=1 logical_accounting=1"
        <<" event_scatter_before_publication=1 bounded_pinned_bytes="<<2*(1U<<20)
        <<std::endl;
}

void run_exact_host_kv_deferred_scatter_screen(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()==4352 && code.size()>=prefix && prose.size()>=prefix,
        "deferred exact host KV scatter fixture/context extent");
    configure_exact_host_kv_pinned_d2h();
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "deferred exact host KV scatter finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    struct Result {
        std::vector<std::int64_t> tokens;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
        ninfer::exl3::Exl3HostKVStats delta;
    };
    const auto run=[&](Exl3TextContext& context,
                       const std::vector<std::int64_t>& input) {
        const auto before=context.host_kv_stats();
        const auto request=Request::initialize(context,input,1024);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=greedy(context);tokens.push_back(token);context.decode(token);
        }
        const auto after=context.host_kv_stats();auto delta=after;
        delta.h2d_bytes-=before.h2d_bytes;delta.d2h_bytes-=before.d2h_bytes;
        delta.transfer_calls-=before.transfer_calls;
        delta.copy_submissions-=before.copy_submissions;
        delta.completed_rows-=before.completed_rows;
        delta.pinned_slot_waits-=before.pinned_slot_waits;
        delta.pinned_deferred_drains-=before.pinned_deferred_drains;
        delta.pinned_scatter_bytes-=before.pinned_scatter_bytes;
        delta.pinned_plane_tail_deferrals-=before.pinned_plane_tail_deferrals;
        delta.pinned_publication_drains-=before.pinned_publication_drains;
        delta.page_extension_calls-=before.page_extension_calls;
        delta.page_prefix_refs-=before.page_prefix_refs;
        delta.page_clone_bytes-=before.page_clone_bytes;
        delta.page_extension_cpu_ns-=before.page_extension_cpu_ns;
        delta.page_unique_tail_reuses-=before.page_unique_tail_reuses;
        delta.page_unique_tail_reuse_bytes-=before.page_unique_tail_reuse_bytes;
        delta.page_prefix_ref_cpu_ns-=before.page_prefix_ref_cpu_ns;
        delta.page_clone_cpu_ns-=before.page_clone_cpu_ns;
        delta.page_payload_prepare_cpu_ns-=before.page_payload_prepare_cpu_ns;
        delta.pinned_wait_cpu_ns-=before.pinned_wait_cpu_ns;
        delta.pinned_scatter_cpu_ns-=before.pinned_scatter_cpu_ns;
        delta.pinned_pending_scan_cpu_ns-=before.pinned_pending_scan_cpu_ns;
        return Result{std::move(tokens),context.export_exact_host_state(),delta};
    };
    for(const auto& fixture:std::array<std::pair<const char*,
        const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},
        std::pair{"prose",&prose}}) {
        const std::vector<std::int64_t> input(
            fixture.second->begin(),fixture.second->begin()+prefix);
        const auto baseline=run(*control,input),changed=run(*candidate,input);
        require(baseline.tokens==changed.tokens &&
                baseline.state->same_payload(*changed.state),
            "deferred exact host KV scatter token/state mismatch");
        const auto& a=baseline.delta;const auto& b=changed.delta;
        require(a.h2d_bytes==b.h2d_bytes && a.d2h_bytes==b.d2h_bytes &&
                a.transfer_calls==b.transfer_calls &&
                a.copy_submissions==b.copy_submissions &&
                a.completed_rows==b.completed_rows &&
                a.pinned_staging_bytes==b.pinned_staging_bytes &&
                b.pinned_staging_bytes==2*(1U<<20) &&
                b.pinned_scatter_bytes==b.d2h_bytes &&
                a.pinned_plane_tail_deferrals==0 &&
                b.pinned_plane_tail_deferrals>0 &&
                b.pinned_publication_drains>0 &&
                b.pinned_plane_tail_deferrals==
                    32*b.pinned_publication_drains &&
                b.pinned_max_pending_bytes<=2*(1U<<20),
            "deferred exact host KV scatter ownership/accounting mismatch");
        std::cout<<"EXACT_HOST_KV_DEFERRED_SCATTER fixture="<<fixture.first
            <<" prefix="<<prefix<<" decode_rows=8"
            <<" deferrals="<<b.pinned_plane_tail_deferrals
            <<" drains="<<b.pinned_deferred_drains
            <<" publication_drains="<<b.pinned_publication_drains
            <<" slot_waits="<<b.pinned_slot_waits
            <<" scatter_bytes="<<b.pinned_scatter_bytes
            <<" max_pending_bytes="<<b.pinned_max_pending_bytes
            <<" page_extension_calls="<<b.page_extension_calls
            <<" page_prefix_refs="<<b.page_prefix_refs
            <<" page_clone_bytes="<<b.page_clone_bytes
            <<" page_extension_cpu_ns="<<b.page_extension_cpu_ns
            <<" page_unique_tail_reuses="<<b.page_unique_tail_reuses
            <<" page_unique_tail_reuse_bytes="<<b.page_unique_tail_reuse_bytes
            <<" page_prefix_ref_cpu_ns="<<b.page_prefix_ref_cpu_ns
            <<" page_clone_cpu_ns="<<b.page_clone_cpu_ns
            <<" page_payload_prepare_cpu_ns="<<b.page_payload_prepare_cpu_ns
            <<" pinned_wait_cpu_ns="<<b.pinned_wait_cpu_ns
            <<" pinned_scatter_cpu_ns="<<b.pinned_scatter_cpu_ns
            <<" pinned_pending_scan_cpu_ns="<<b.pinned_pending_scan_cpu_ns
            <<std::endl;
    }
    for(const char* malformed:{"2","true","01"}) {
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER",malformed);
        bool rejected=false;
        try {auto invalid=target.create_context(true);}
        catch(const std::runtime_error&) {rejected=true;}
        require(rejected,"deferred exact host KV scatter accepted malformed flag");
    }
    std::cout<<"EXACT_HOST_KV_DEFERRED_SCATTER PASS fixtures=2 prefix=4096"
        <<" decode_rows=8 exact_tokens_state=1 bounded_pinned_bytes="<<2*(1U<<20)
        <<" slot_reuse_drain=1 publication_drain=1 malformed=1"<<std::endl;
}

void run_exact_host_kv_banked_d2h_screen(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096,rows=8;
    constexpr std::uint64_t banked_bytes=16ULL*2*rows*1024*2;
    require(target.max_context()==4352 &&
            code.size()>=prefix && prose.size()>=prefix,
        "banked D2H fixture/context extent");
    configure_exact_host_kv_pinned_d2h();
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER","0");
    const auto retirement_before=Exl3TextContext::retirement_quarantine_witness();
    {
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BANKED_D2H","0");
        auto control=target.create_context(true);control->prepare_continuation(rows);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BANKED_D2H","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(rows);
        struct Result {
            std::vector<std::uint16_t> logits;
            std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
            ninfer::exl3::Exl3HostKVStats delta;
        };
        const auto run=[&](Exl3TextContext& context,
                           const std::vector<std::int64_t>& source) {
            const auto request=Request::initialize(context,
                std::span<const std::int64_t>(source.data(),prefix),1024);
            const auto before=context.host_kv_stats();
            context.continue_rows(std::span<const std::int64_t>(
                source.data()+prefix-rows,rows));
            auto logits=context.continuation_logits_bits_host();
            context.finish_exact_continuation();
            auto state=context.export_exact_host_state();
            const auto after=context.host_kv_stats();auto delta=after;
            delta.h2d_bytes-=before.h2d_bytes;delta.d2h_bytes-=before.d2h_bytes;
            delta.transfer_calls-=before.transfer_calls;
            delta.copy_submissions-=before.copy_submissions;
            delta.completed_rows-=before.completed_rows;
            delta.pinned_slot_waits-=before.pinned_slot_waits;
            delta.pinned_scatter_bytes-=before.pinned_scatter_bytes;
            delta.banked_d2h_forwards-=before.banked_d2h_forwards;
            delta.banked_d2h_planes-=before.banked_d2h_planes;
            delta.banked_d2h_rows-=before.banked_d2h_rows;
            delta.banked_d2h_bytes-=before.banked_d2h_bytes;
            delta.banked_d2h_drains-=before.banked_d2h_drains;
            return Result{std::move(logits),std::move(state),delta};
        };
        for(const auto& fixture:std::array<std::pair<const char*,
            const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},
            std::pair{"prose",&prose}}) {
            const auto baseline=run(*control,*fixture.second);
            const auto changed=run(*candidate,*fixture.second);
            require(baseline.logits==changed.logits &&
                    baseline.state->same_payload(*changed.state),
                "banked D2H continuation logits/state mismatch");
            const auto& a=baseline.delta;const auto& b=changed.delta;
            require(a.h2d_bytes==b.h2d_bytes && a.d2h_bytes==b.d2h_bytes &&
                    a.transfer_calls==b.transfer_calls &&
                    a.copy_submissions==b.copy_submissions &&
                    a.completed_rows==b.completed_rows &&
                    a.banked_d2h_forwards==0 && a.banked_d2h_planes==0 &&
                    a.banked_d2h_rows==0 && a.banked_d2h_bytes==0 &&
                    a.banked_d2h_drains==0 &&
                    b.banked_d2h_forwards==1 && b.banked_d2h_planes==32 &&
                    b.banked_d2h_rows==rows && b.banked_d2h_bytes==banked_bytes &&
                    b.banked_d2h_drains==1 &&
                    b.pinned_scatter_bytes==b.d2h_bytes &&
                    a.pinned_staging_bytes==2*(1U<<20) &&
                    b.pinned_staging_bytes==2*(1U<<20)+(512U<<10),
                "banked D2H route/resource accounting mismatch");
            std::cout<<"EXACT_HOST_KV_BANKED_D2H fixture="<<fixture.first
                <<" prefix="<<prefix<<" rows="<<rows
                <<" control_waits="<<a.pinned_slot_waits
                <<" candidate_waits="<<b.pinned_slot_waits
                <<" planes="<<b.banked_d2h_planes
                <<" drains="<<b.banked_d2h_drains
                <<" banked_bytes="<<b.banked_d2h_bytes
                <<" pinned_bytes="<<b.pinned_staging_bytes<<std::endl;
        }
    }
    require(Exl3TextContext::retirement_quarantine_witness()==retirement_before,
        "banked D2H normal destruction changed retirement quarantine");
    for(const char* malformed:{"2","true","01"}) {
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BANKED_D2H",malformed);
        bool rejected=false;
        try {auto invalid=target.create_context(true);}
        catch(const std::runtime_error&) {rejected=true;}
        require(rejected,"banked D2H accepted malformed flag");
    }
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BANKED_D2H","0");
    std::cout<<"EXACT_HOST_KV_BANKED_D2H PASS fixtures=2 prefix="<<prefix
        <<" rows="<<rows<<" exact_logits_state=1 route_counters=1"
        <<" bounded_extra_pinned_bytes="<<(512U<<10)
        <<" normal_retirement=1 malformed=1"<<std::endl;
}

void run_exact_host_kv_unique_tail_reuse_screen(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()==4352 && code.size()>=prefix && prose.size()>=prefix,
        "unique-tail reuse fixture/context extent");
    configure_exact_host_kv_pinned_d2h();
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER","1");
    _putenv_s("NINFER_EXL3_HOST_KV_CPU_PROFILE","1");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "unique-tail reuse finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    struct Result {
        std::vector<std::int64_t> tokens;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>> retained;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
        ninfer::exl3::Exl3HostKVStats delta;
    };
    const auto run=[&](Exl3TextContext& context,
                       const std::vector<std::int64_t>& input,bool retain_each) {
        const auto before=context.host_kv_stats();
        const auto request=Request::initialize(context,input,1024);
        Result result;
        for(int row=0;row<8;++row) {
            const auto token=greedy(context);result.tokens.push_back(token);context.decode(token);
            if(retain_each) result.retained.push_back(context.export_exact_host_state());
        }
        result.state=context.export_exact_host_state();
        const auto after=context.host_kv_stats();result.delta=after;
        result.delta.h2d_bytes-=before.h2d_bytes;result.delta.d2h_bytes-=before.d2h_bytes;
        result.delta.transfer_calls-=before.transfer_calls;
        result.delta.copy_submissions-=before.copy_submissions;
        result.delta.completed_rows-=before.completed_rows;
        result.delta.pinned_slot_waits-=before.pinned_slot_waits;
        result.delta.pinned_deferred_drains-=before.pinned_deferred_drains;
        result.delta.pinned_scatter_bytes-=before.pinned_scatter_bytes;
        result.delta.pinned_plane_tail_deferrals-=before.pinned_plane_tail_deferrals;
        result.delta.pinned_publication_drains-=before.pinned_publication_drains;
        result.delta.page_extension_calls-=before.page_extension_calls;
        result.delta.page_prefix_refs-=before.page_prefix_refs;
        result.delta.page_clone_bytes-=before.page_clone_bytes;
        result.delta.page_extension_cpu_ns-=before.page_extension_cpu_ns;
        result.delta.page_unique_tail_reuses-=before.page_unique_tail_reuses;
        result.delta.page_unique_tail_reuse_bytes-=before.page_unique_tail_reuse_bytes;
        result.delta.page_prefix_ref_cpu_ns-=before.page_prefix_ref_cpu_ns;
        result.delta.page_clone_cpu_ns-=before.page_clone_cpu_ns;
        result.delta.page_payload_prepare_cpu_ns-=before.page_payload_prepare_cpu_ns;
        result.delta.pinned_wait_cpu_ns-=before.pinned_wait_cpu_ns;
        result.delta.pinned_scatter_cpu_ns-=before.pinned_scatter_cpu_ns;
        result.delta.pinned_pending_scan_cpu_ns-=before.pinned_pending_scan_cpu_ns;
        return result;
    };
    for(const auto& fixture:std::array<std::pair<const char*,
        const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},
        std::pair{"prose",&prose}}) for(bool retain_each:{false,true}) {
        const std::vector<std::int64_t> input(
            fixture.second->begin(),fixture.second->begin()+prefix);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_UNIQUE_TAIL_REUSE","0");
        auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_UNIQUE_TAIL_REUSE","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        const auto baseline=run(*control,input,retain_each);
        const auto changed=run(*candidate,input,retain_each);
        require(baseline.tokens==changed.tokens &&
                baseline.state->same_payload(*changed.state) &&
                baseline.retained.size()==changed.retained.size(),
            "unique-tail reuse token/final-state mismatch");
        for(std::size_t i=0;i<baseline.retained.size();++i)
            require(baseline.retained[i]->same_payload(*changed.retained[i]),
                "unique-tail reuse retained snapshot changed");
        const auto& a=baseline.delta;const auto& b=changed.delta;
        std::cout<<"EXACT_HOST_KV_UNIQUE_TAIL_ACCOUNTING fixture="<<fixture.first
            <<" retain_each="<<(retain_each?1:0)
            <<" control_h2d="<<a.h2d_bytes<<" candidate_h2d="<<b.h2d_bytes
            <<" control_d2h="<<a.d2h_bytes<<" candidate_d2h="<<b.d2h_bytes
            <<" control_calls="<<a.transfer_calls<<" candidate_calls="<<b.transfer_calls
            <<" control_submissions="<<a.copy_submissions<<" candidate_submissions="<<b.copy_submissions
            <<" control_rows="<<a.completed_rows<<" candidate_rows="<<b.completed_rows
            <<" control_extensions="<<a.page_extension_calls<<" candidate_extensions="<<b.page_extension_calls
            <<" control_refs="<<a.page_prefix_refs<<" candidate_refs="<<b.page_prefix_refs
            <<" control_reuses="<<a.page_unique_tail_reuses<<" candidate_reuses="<<b.page_unique_tail_reuses
            <<std::endl;
        require(a.h2d_bytes==b.h2d_bytes && a.d2h_bytes==b.d2h_bytes &&
                a.transfer_calls==b.transfer_calls &&
                a.copy_submissions==b.copy_submissions &&
                a.completed_rows==b.completed_rows &&
                a.page_extension_calls==b.page_extension_calls &&
                a.page_prefix_refs==b.page_prefix_refs &&
                a.page_unique_tail_reuses==0,
            "unique-tail reuse route accounting mismatch");
        if(retain_each) require(b.page_unique_tail_reuses>0 &&
                b.page_clone_bytes>0 &&
                b.page_clone_bytes+b.page_unique_tail_reuse_bytes==a.page_clone_bytes,
            "unique-tail reuse bypassed retained snapshot copy-on-write");
        else require(b.page_unique_tail_reuses>0 &&
                b.page_clone_bytes<a.page_clone_bytes &&
                b.page_unique_tail_reuse_bytes==
                    a.page_clone_bytes-b.page_clone_bytes,
            "unique-tail reuse did not remove private tail clones");
        std::cout<<"EXACT_HOST_KV_UNIQUE_TAIL_REUSE fixture="<<fixture.first
            <<" retain_each="<<(retain_each?1:0)
            <<" control_clone_bytes="<<a.page_clone_bytes
            <<" candidate_clone_bytes="<<b.page_clone_bytes
            <<" tail_reuses="<<b.page_unique_tail_reuses
            <<" avoided_clone_bytes="<<b.page_unique_tail_reuse_bytes
            <<" control_page_cpu_ns="<<a.page_extension_cpu_ns
            <<" candidate_page_cpu_ns="<<b.page_extension_cpu_ns<<std::endl;
    }
    for(const char* malformed:{"2","true","01"}) {
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_UNIQUE_TAIL_REUSE",malformed);
        bool rejected=false;
        try {auto invalid=target.create_context(true);}
        catch(const std::runtime_error&) {rejected=true;}
        require(rejected,"unique-tail reuse accepted malformed flag");
    }
    std::cout<<"EXACT_HOST_KV_UNIQUE_TAIL_REUSE PASS fixtures=2 retention_arms=2"
        <<" prefix=4096 decode_rows=8 exact_tokens_state_snapshots=1"
        <<" exclusive_ownership_only=1 malformed=1"<<std::endl;
}

void run_exact_host_kv_unique_tail_reuse_t1(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()==4352 && code.size()>=prefix && prose.size()>=prefix,
        "unique-tail T1 fixture/context extent");
    configure_exact_host_kv_pinned_d2h();
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER","1");
    _putenv_s("NINFER_EXL3_HOST_KV_CPU_PROFILE","0");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "unique-tail T1 finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const auto median=[](std::vector<double> values) {
        std::sort(values.begin(),values.end());return values[values.size()/2];
    };
    struct Result {
        double wall_ms=0;
        std::vector<std::int64_t> tokens;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
    };
    for(const auto& fixture:std::array<std::pair<const char*,
        const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},
        std::pair{"prose",&prose}}) {
        const std::vector<std::int64_t> input(
            fixture.second->begin(),fixture.second->begin()+prefix);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_UNIQUE_TAIL_REUSE","0");
        auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV_UNIQUE_TAIL_REUSE","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        const auto run=[&](Exl3TextContext& context) {
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            Result result;
            for(int row=0;row<8;++row) {
                const auto token=greedy(context);result.tokens.push_back(token);context.decode(token);
            }
            result.wall_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            result.state=context.export_exact_host_state();
            return result;
        };
        std::vector<double> control_ms,candidate_ms;
        for(int rep=0;rep<3;++rep) {
            const std::array<bool,2> order=rep==1?std::array{true,false}:
                std::array{false,true};
            Result first,second;
            for(int ordinal=0;ordinal<2;++ordinal) {
                const bool changed=order[ordinal];
                auto result=run(changed?*candidate:*control);
                (changed?candidate_ms:control_ms).push_back(result.wall_ms);
                if(ordinal==0) first=std::move(result);else second=std::move(result);
                std::cout<<"EXACT_HOST_KV_UNIQUE_TAIL_T1_CASE fixture="<<fixture.first
                    <<" rep="<<rep<<" order="<<ordinal
                    <<" arm="<<(changed?"unique_tail":"cow_clone")
                    <<" wall_ms="<<(ordinal==0?first.wall_ms:second.wall_ms)<<std::endl;
            }
            require(first.tokens==second.tokens && first.state->same_payload(*second.state),
                "unique-tail T1 token/state mismatch");
        }
        const double baseline=median(control_ms),changed=median(candidate_ms);
        std::cout<<"EXACT_HOST_KV_UNIQUE_TAIL_T1 fixture="<<fixture.first
            <<" clone_median_ms="<<baseline<<" reuse_median_ms="<<changed
            <<" wall_reduction_percent="<<(baseline-changed)*100/baseline<<std::endl;
    }
    std::cout<<"EXACT_HOST_KV_UNIQUE_TAIL_T1 PASS fixtures=2 pairs=3"
        <<" prefix=4096 decode_rows=8 exact_tokens_state=1 uninstrumented=1"<<std::endl;
}
