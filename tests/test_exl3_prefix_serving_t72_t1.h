#pragma once

void run_prefix_serving_t72_t1(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Cache=ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity=ninfer::exl3::Exl3VeriCacheServingIdentity;
    constexpr std::size_t prompt_rows=4096,prefix_rows=3968,tail_rows=128;
    require(target.max_context()>=static_cast<int>(prompt_rows) &&
        code.size()>=prompt_rows && prose.size()>=tail_rows,
        "T72 T1 4K fixture extent");
    const auto output=std::filesystem::path(env("NINFER_T72_OUT"));
    require(!output.empty() && !std::filesystem::exists(output),"T72 T1 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib=std::stoull(env("NINFER_T72_RESERVE_GIB"));
    const auto cache_mib=std::stoull(env("NINFER_T72_CACHE_MIB"));
    const std::uint64_t reserve=reserve_gib*(1ULL<<30),cache_budget=cache_mib*(1ULL<<20);
    require(reserve_gib>=4 && cache_mib>=512,"T72 T1 memory policy extent");
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    require(GlobalMemoryStatusEx(&memory)!=0 && memory.ullTotalPhys>reserve &&
        memory.ullAvailPhys>reserve,"T72 T1 physical memory reserve");
    const auto available=[] {
        MEMORYSTATUSEX state{};state.dwLength=sizeof(state);
        require(GlobalMemoryStatusEx(&state)!=0,"T72 T1 physical memory query");return state.ullAvailPhys;
    };
    const Identity identity{env("NINFER_T72_NAMESPACE"),env("NINFER_T72_ARTIFACT_ID"),
        env("NINFER_T72_TOKENIZER_ID"),env("NINFER_T72_CONFIGURATION_ID"),
        env("NINFER_T72_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    Cache cache(Cache::Policy{4,64,cache_budget,reserve},identity);
    ninfer::exl3::Exl3HostResidentSet residency(memory.ullTotalPhys-reserve,reserve);
    auto cached_context=target.create_context(true);cached_context->prepare_continuation(8);
    auto fresh_context=target.create_context(true);fresh_context->prepare_continuation(8);

    std::vector<std::int64_t> common(code.begin(),code.begin()+prefix_rows);
    std::vector<std::int64_t> code_prompt(code.begin(),code.begin()+prompt_rows);
    auto prose_prompt=common;prose_prompt.insert(prose_prompt.end(),prose.begin(),prose.begin()+tail_rows);
    require(code_prompt.size()==prompt_rows && prose_prompt.size()==prompt_rows &&
        !std::equal(code_prompt.begin()+prefix_rows,code_prompt.end(),
            prose_prompt.begin()+prefix_rows),"T72 T1 distinct real tails");

    struct LockResult {ninfer::exl3::Exl3HostResidentSet::Stats update;Exl3HostResidency observed;};
    const auto pin=[&](std::shared_ptr<const Request> active={}) {
        auto roots=cache.roots();if(active) roots.push_back(std::move(active));
        ninfer::exl3::Exl3HostResidentSet::Snapshot snapshot;
        for(const auto& root:roots) snapshot.owners.push_back(root);
        Request::visit_host_allocations(roots,[&](const void* data,std::size_t bytes){snapshot.add(data,bytes);},false);
        const auto update=residency.replace(std::move(snapshot));
        Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(roots,[&](const void* data,std::size_t bytes){probe.add(data,bytes);},false);
        const auto observed=probe.measure();
        require(observed.resident_tensor_bytes==observed.allocated_union_bytes &&
            observed.locked_tensor_bytes==observed.allocated_union_bytes,
            "T72 T1 authoritative host payload is not resident and locked");
        return LockResult{update,observed};
    };

    const auto fill_start=std::chrono::steady_clock::now();
    auto fill=cache.prepare(*cached_context,code_prompt,prefix_rows,available(),1024);
    cuda_check(cudaDeviceSynchronize(),"T72 T1 cache fill completion");
    const auto fill_lock=pin(fill.request);const auto fill_token=sample_target(*cached_context);
    const double fill_ttft_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-fill_start).count();
    require(!fill.metrics.cache_hit && fill.metrics.admitted &&
        fill.metrics.executed_prompt_tokens==prompt_rows && fill_token>=0,
        "T72 T1 initial cache fill");
    fill.request.reset();pin();

    std::ofstream requests(output/"requests.csv");
    require(requests.good(),"T72 T1 request evidence open");
    requests<<"fixture,rep,order,arm,prompt_tokens,reused_prompt_tokens,executed_prompt_tokens,"
        "lookup_ms,prefill_ms,attach_ms,residency_update_ms,ttft_ms,first_token,"
        "resident_payload_bytes,locked_payload_bytes,new_locked_bytes,unlocked_bytes,exact\n";
    struct Arm {
        std::shared_ptr<const Request> request;
        Cache::Metrics metrics;
        LockResult lock;
        double ttft_ms=0;
        std::int64_t token=-1;
    };
    std::array<std::vector<double>,2> fresh_ttft,cached_ttft;
    const std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2> fixtures={
        std::pair{"code_tail",&code_prompt},std::pair{"prose_tail",&prose_prompt}};
    for(std::size_t fixture=0;fixture<fixtures.size();++fixture) for(int rep=0;rep<3;++rep) {
        Arm fresh_arm,cached_arm;
        const auto run=[&](bool cached) {
            auto& arm=cached?cached_arm:fresh_arm;
            const auto started=std::chrono::steady_clock::now();
            if(cached) {
                auto result=cache.prepare(*cached_context,*fixtures[fixture].second,
                    prefix_rows,available(),1024);
                arm.request=std::move(result.request);arm.metrics=result.metrics;
            } else {
                const auto prefill=std::chrono::steady_clock::now();
                arm.request=Request::initialize(*fresh_context,*fixtures[fixture].second,1024);
                arm.metrics.prompt_tokens=prompt_rows;arm.metrics.executed_prompt_tokens=prompt_rows;
                arm.metrics.prefill_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-prefill).count();
            }
            cuda_check(cudaDeviceSynchronize(),"T72 T1 request completion");
            arm.lock=pin(arm.request);
            arm.token=sample_target(cached?*cached_context:*fresh_context);
            arm.ttft_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            (cached?cached_ttft[fixture]:fresh_ttft[fixture]).push_back(arm.ttft_ms);
            const int order=(rep==1)?(cached?0:1):(cached?1:0);
            requests<<fixtures[fixture].first<<','<<rep<<','<<order<<','<<(cached?"cached":"fresh")
                <<','<<arm.metrics.prompt_tokens<<','<<arm.metrics.reused_prompt_tokens
                <<','<<arm.metrics.executed_prompt_tokens<<','<<arm.metrics.lookup_ms
                <<','<<arm.metrics.prefill_ms<<','<<arm.metrics.attach_ms
                <<','<<arm.lock.update.elapsed_ms<<','<<arm.ttft_ms<<','<<arm.token
                <<','<<arm.lock.observed.resident_tensor_bytes
                <<','<<arm.lock.observed.locked_tensor_bytes
                <<','<<arm.lock.update.new_locked_bytes<<','<<arm.lock.update.unlocked_bytes<<",1\n"<<std::flush;
        };
        if(rep==1) {run(true);run(false);} else {run(false);run(true);}
        require(cached_arm.metrics.cache_hit && cached_arm.metrics.reused_prompt_tokens==prefix_rows &&
            cached_arm.metrics.executed_prompt_tokens==tail_rows &&
            fresh_arm.metrics.reused_prompt_tokens==0 && fresh_arm.metrics.executed_prompt_tokens==prompt_rows &&
            cached_arm.token==fresh_arm.token &&
            cached_arm.request->token_suffix()==fresh_arm.request->token_suffix() &&
            cached_arm.request->same_taps(*fresh_arm.request) &&
            cached_arm.request->state()->same_payload(*fresh_arm.request->state()),
            "T72 T1 cached route differs from fresh exact authority");
        pin();fresh_arm.request.reset();cached_arm.request.reset();
    }
    requests.close();require(requests.good(),"T72 T1 request evidence flush");

    bool cancelled=false;
    try {
        auto cancelled_prompt=prose_prompt;cancelled_prompt.back()=(cancelled_prompt.back()+29)%kVocab;
        cache.prepare(*cached_context,cancelled_prompt,prefix_rows,available(),32,
            [&](int position){if(position>static_cast<int>(prefix_rows))
                throw std::runtime_error("intentional T72 T1 cancellation");});
    } catch(const std::runtime_error&) {cancelled=true;}
    auto after_cancel=cache.prepare(*cached_context,code_prompt,prefix_rows,available(),1024);
    require(cancelled && after_cancel.metrics.cache_hit &&
        after_cancel.metrics.reused_prompt_tokens==prefix_rows,
        "T72 T1 cancellation contaminated cached root");
    after_cancel.request.reset();const auto before_pressure=cache.storage_stats();
    require(before_pressure.entries==1 && before_pressure.accounted_bytes<=cache_budget,
        "T72 T1 cache byte budget");
    std::size_t pressure_evicted=0;const auto pressure=cache.trim(reserve,&pressure_evicted);
    const auto pressure_unlock=pin();
    require(pressure.entries==0 && pressure_evicted==1 && pressure_unlock.update.unlocked_bytes>0,
        "T72 T1 pressure eviction/residency release");
    auto changed_identity=identity;changed_identity.configuration_identity+="-reload";
    cache.reset_for_model_reload(changed_identity);
    require(cache.storage_stats().entries==0,"T72 T1 reload retained publication");
    residency.close();

    const auto median=[](std::vector<double> values) {
        std::sort(values.begin(),values.end());return values[values.size()/2];
    };
    std::ofstream summary(output/"summary.csv");
    require(summary.good(),"T72 T1 summary open");
    summary<<"fixture,fresh_ttft_median_ms,cached_ttft_median_ms,ttft_reduction_percent,"
        "avoided_input_tokens,executed_input_tokens,pairs_positive\n";
    for(std::size_t fixture=0;fixture<fixtures.size();++fixture) {
        const double fresh=median(fresh_ttft[fixture]),cached=median(cached_ttft[fixture]);
        const int positive=static_cast<int>(std::inner_product(fresh_ttft[fixture].begin(),
            fresh_ttft[fixture].end(),cached_ttft[fixture].begin(),0,std::plus<int>(),
            [](double a,double b){return a>b?1:0;}));
        summary<<fixtures[fixture].first<<','<<fresh<<','<<cached<<','
            <<(fresh-cached)*100.0/fresh<<','<<prefix_rows<<','<<tail_rows<<','<<positive<<"/3\n";
        std::cout<<"T72_PREFIX_SERVING_T1 fixture="<<fixtures[fixture].first
            <<" fresh_ttft_median_ms="<<fresh<<" cached_ttft_median_ms="<<cached
            <<" ttft_reduction_percent="<<(fresh-cached)*100.0/fresh
            <<" avoided_input_tokens="<<prefix_rows<<" executed_input_tokens="<<tail_rows
            <<" pairs_positive="<<positive<<"/3"<<std::endl;
    }
    summary.close();require(summary.good(),"T72 T1 summary flush");
    std::ofstream memory_out(output/"memory.csv");
    memory_out<<"phase,cache_payload_bytes,cache_identity_bytes,cache_accounted_bytes,"
        "resident_payload_bytes,locked_payload_bytes,resident_page_bytes,new_locked_bytes,"
        "unlocked_bytes,host_available_bytes,physical_reserve_bytes\n"
        <<"filled,"<<fill.metrics.cache_storage.payload_allocated_bytes<<','
        <<fill.metrics.cache_storage.identity_allocated_bytes<<','
        <<fill.metrics.cache_storage.accounted_bytes<<','
        <<fill_lock.observed.resident_tensor_bytes<<','<<fill_lock.observed.locked_tensor_bytes<<','
        <<fill_lock.update.page_bytes<<','<<fill_lock.update.new_locked_bytes<<','
        <<fill_lock.update.unlocked_bytes<<','<<available()<<','<<reserve<<"\n"
        <<"pressure_cleared,0,0,0,0,0,0,0,"<<pressure_unlock.update.unlocked_bytes
        <<','<<available()<<','<<reserve<<"\n";
    memory_out.close();require(memory_out.good(),"T72 T1 memory evidence flush");
    std::cout<<"T72_PREFIX_SERVING_T1 PASS fixtures=2 pairs=3 prefix=3968 tail=128"
        <<" exact_tokens_taps_state=1 cancellation=rollback reload=cleared"
        <<" resident_locked=1 cache_payload_bytes="<<fill.metrics.cache_storage.payload_allocated_bytes
        <<" cache_identity_bytes="<<fill.metrics.cache_storage.identity_allocated_bytes
        <<" cache_budget_bytes="<<cache_budget<<" physical_reserve_bytes="<<reserve
        <<" fill_ttft_ms="<<fill_ttft_ms<<" fill_token="<<fill_token
        <<" output="<<output.string()<<std::endl;
}
