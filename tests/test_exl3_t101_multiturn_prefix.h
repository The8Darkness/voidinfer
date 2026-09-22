#pragma once

void run_t101_multiturn_prefix(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose, const std::filesystem::path& output) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Cache=ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity=ninfer::exl3::Exl3VeriCacheServingIdentity;
    constexpr std::size_t initial_rows=3136,tail_rows=128;
    constexpr int response_rows=8;
    require(target.max_context()>=3600 && code.size()>=3800 && prose.size()>=3800,
        "T101 fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
        "T101 output must be new");
    std::filesystem::create_directories(output);
    const std::uint64_t reserve=std::stoull(env("NINFER_T101_RESERVE_GIB"))*(1ULL<<30);
    const std::uint64_t budget=std::stoull(env("NINFER_T101_CACHE_MIB"))*(1ULL<<20);
    require(reserve>=(4ULL<<30) && budget>=(512ULL<<20),"T101 memory policy extent");
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    require(GlobalMemoryStatusEx(&memory)!=0 && memory.ullAvailPhys>reserve,
        "T101 initial physical reserve");
    const auto available=[] {
        MEMORYSTATUSEX state{};state.dwLength=sizeof(state);
        require(GlobalMemoryStatusEx(&state)!=0,"T101 physical memory query");
        return state.ullAvailPhys;
    };
    Identity identity{env("NINFER_T101_NAMESPACE"),env("NINFER_T101_ARTIFACT_ID"),
        env("NINFER_T101_TOKENIZER_ID"),env("NINFER_T101_CONFIGURATION_ID"),
        env("NINFER_T101_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    Cache cache(Cache::Policy{1,64,budget,reserve},identity);
    ninfer::exl3::Exl3HostResidentSet residency(memory.ullTotalPhys-reserve,reserve);
    auto cached_context=target.create_context(true);cached_context->prepare_continuation(8);
    auto fresh_context=target.create_context(true);fresh_context->prepare_continuation(8);

    struct Pin {ninfer::exl3::Exl3HostResidentSet::Stats update;Exl3HostResidency observed;};
    const auto pin=[&](std::shared_ptr<const Request> active={}) {
        auto roots=cache.roots();if(active) roots.push_back(std::move(active));
        ninfer::exl3::Exl3HostResidentSet::Snapshot snapshot;
        for(const auto& root:roots) snapshot.owners.push_back(root);
        Request::visit_host_allocations(roots,[&](const void* p,std::size_t n){snapshot.add(p,n);},false);
        auto update=residency.replace(std::move(snapshot));Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(roots,[&](const void* p,std::size_t n){probe.add(p,n);},false);
        auto observed=probe.measure();
        require(observed.resident_tensor_bytes==observed.allocated_union_bytes &&
            observed.locked_tensor_bytes==observed.allocated_union_bytes,
            "T101 authoritative payload not resident and locked");
        return Pin{update,observed};
    };
    const auto complete=[&](std::shared_ptr<const Request> root,Exl3TextContext& context) {
        std::vector<std::int64_t> output_tokens;
        for(int row=0;row<response_rows;++row) {
            const auto pending=sample_target(context);
            const std::array<std::int64_t,1> tentative{pending};
            auto [next,verified]=root->verify(context,tentative);
            require(verified.committed_tokens.size()==1 &&
                verified.committed_tokens.front()==pending && verified.accepted==1 &&
                !verified.rejected && !verified.stopped && verified.verification_rows==1 &&
                verified.executed_rows==1 && verified.replay_rows==0 &&
                verified.native_invocations==1 && verified.root_restores==0,
                "T101 authoritative response publication");
            output_tokens.push_back(pending);root=std::move(next);
        }
        return std::pair{std::move(root),std::move(output_tokens)};
    };

    std::ofstream rows(output/"turns.csv");
    require(rows.good(),"T101 turn evidence open");
    rows<<"fixture,rep,turn,order,arm,prompt_tokens,reused_tokens,executed_tokens,"
        "prefill_ms,attach_ms,ttft_ms,first_token,promotion_ms,promotion_evicted_prior,"
        "resident_bytes,locked_bytes,exact\n";
    std::uint64_t publications=0,promotions=0,avoided=0,executed=0;
    bool cancellation_checked=false,reload_checked=false;
    const std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2> fixtures{{
        {"code",&code},{"prose",&prose}}};
    for(std::size_t fixture=0;fixture<fixtures.size();++fixture) for(int rep=0;rep<2;++rep) {
        cache.reset_for_model_reload(identity);pin();
        const auto& base_source=*fixtures[fixture].second;
        const auto& other=fixture?code:prose;
        std::vector<std::int64_t> initial(base_source.begin(),base_source.begin()+initial_rows);
        auto active=Request::initialize(*cached_context,initial,128);
        auto [completed_initial,initial_output]=complete(active,*cached_context);
        publications+=initial_output.size();
        auto admission=cache.admit_completed_authority(completed_initial,available());
        require(admission.admitted && admission.storage.entries==1,
            "T101 initial completed-root admission");
        const auto initial_index_identity_bytes=admission.storage.identity_allocated_bytes;
        require(cache.roots().front()==completed_initial &&
            completed_initial->token_count()==completed_initial->token_suffix().size(),
            "completed admission lost immutable history authority");
        ++promotions;pin();active=std::move(completed_initial);

        for(int turn=2;turn<=3;++turn) {
            if(turn==2 && !cancellation_checked) {
                auto cancelled_prompt=active->token_suffix();
                cancelled_prompt.insert(cancelled_prompt.end(),other.begin(),other.begin()+tail_rows);
                bool cancelled=false;
                try {cache.prepare_concurrent(*cached_context,cancelled_prompt,
                    active->token_suffix().size(),available(),128,
                    [&](int position){if(position>active->state()->position())
                        throw std::runtime_error("intentional T101 cancellation");});}
                catch(const std::runtime_error&) {cancelled=true;}
                require(cancelled && cache.roots().size()==1 &&
                    cache.roots().front()->token_suffix()==active->token_suffix(),
                    "T101 cancellation changed completed prefix");
                cancellation_checked=true;
            }
            const auto prior_tokens=active->token_suffix();
            std::vector<std::int64_t> prompt=prior_tokens;
            const std::size_t offset=turn==2?0:256;
            prompt.insert(prompt.end(),other.begin()+offset,other.begin()+offset+tail_rows);
            struct Arm {std::shared_ptr<const Request> root;Cache::Metrics metrics;
                double ttft_ms=0;std::int64_t first=-1;};
            Arm cached,fresh;
            const auto run_cached=[&] {
                const auto started=std::chrono::steady_clock::now();
                auto result=cache.prepare_concurrent(*cached_context,prompt,prior_tokens.size(),
                    available(),128);cached.root=std::move(result.request);cached.metrics=result.metrics;
                cuda_check(cudaDeviceSynchronize(),"T101 cached prepare completion");
                cached.first=sample_target(*cached_context);
                cached.ttft_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-started).count();
            };
            const auto run_fresh=[&] {
                const auto started=std::chrono::steady_clock::now();
                fresh.root=Request::initialize(*fresh_context,prompt,128);
                cuda_check(cudaDeviceSynchronize(),"T101 fresh prepare completion");
                fresh.first=sample_target(*fresh_context);
                fresh.ttft_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-started).count();
                fresh.metrics.prompt_tokens=prompt.size();fresh.metrics.executed_prompt_tokens=prompt.size();
            };
            const int order=(rep+turn+static_cast<int>(fixture))&1;
            if(order==0){run_fresh();run_cached();}else{run_cached();run_fresh();}
            require(cached.metrics.cache_hit && cached.metrics.reused_prompt_tokens==prior_tokens.size() &&
                cached.metrics.executed_prompt_tokens==tail_rows && cached.first==fresh.first &&
                cached.root->token_suffix()==fresh.root->token_suffix() &&
                cached.root->same_taps(*fresh.root) && cached.root->state()->same_payload(*fresh.root->state()),
                "T101 growing prompt differs from fresh authority");
            auto live=pin(cached.root);
            auto [cached_done,cached_output]=complete(cached.root,*cached_context);
            auto [fresh_done,fresh_output]=complete(fresh.root,*fresh_context);
            require(cached_output==fresh_output && cached_done->token_suffix()==fresh_done->token_suffix() &&
                cached_done->same_taps(*fresh_done) && cached_done->state()->same_payload(*fresh_done->state()),
                "T101 completed turn differs from fresh authority");
            publications+=cached_output.size();avoided+=cached.metrics.reused_prompt_tokens;
            executed+=cached.metrics.executed_prompt_tokens;
            const auto promotion_start=std::chrono::steady_clock::now();
            admission=cache.admit_completed_authority(cached_done,available());
            const double promotion_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-promotion_start).count();
            require(admission.admitted && !admission.replaced && admission.evicted==1 &&
                admission.storage.entries==1,
                "T101 growing completed-root FIFO turnover");
            require(admission.storage.identity_allocated_bytes==initial_index_identity_bytes &&
                cache.roots().front()==cached_done && cached_done->same_tokens(*fresh_done) &&
                cached_done->token_fingerprint()==fresh_done->token_fingerprint(),
                "growing turn rebuilt duplicate index keys or changed history identity");
            ++promotions;pin();
            rows<<fixtures[fixture].first<<','<<rep<<','<<turn<<','<<order<<",fresh,"
                <<prompt.size()<<",0,"<<prompt.size()<<','<<fresh.metrics.prefill_ms<<",0,"
                <<fresh.ttft_ms<<','<<fresh.first<<",0,0,"<<live.observed.resident_tensor_bytes
                <<','<<live.observed.locked_tensor_bytes<<",1\n";
            rows<<fixtures[fixture].first<<','<<rep<<','<<turn<<','<<order<<",cached,"
                <<prompt.size()<<','<<cached.metrics.reused_prompt_tokens<<','
                <<cached.metrics.executed_prompt_tokens<<','<<cached.metrics.prefill_ms<<','
                <<cached.metrics.attach_ms<<','<<cached.ttft_ms<<','<<cached.first<<','
                <<promotion_ms<<','<<(admission.evicted==1?1:0)<<','
                <<live.observed.resident_tensor_bytes<<','<<live.observed.locked_tensor_bytes<<",1\n";
            active=std::move(cached_done);
        }
    }
    rows.close();require(rows.good(),"T101 turn evidence flush");

    auto changed=identity;changed.configuration_identity+="-reload";
    cache.reset_for_model_reload(changed);pin();
    require(cache.storage_stats().entries==0,"T101 reload retained completed root");
    auto reload_prompt=std::vector<std::int64_t>(code.begin(),code.begin()+256);
    auto reloaded=cache.prepare(*cached_context,reload_prompt,reload_prompt.size(),available(),128);
    require(!reloaded.metrics.cache_hit && reloaded.metrics.executed_prompt_tokens==reload_prompt.size() &&
        reloaded.metrics.admitted,"T101 reload did not refill cold");
    reload_checked=true;cache.reset_for_model_reload(changed);pin();residency.close();
    std::cout<<"T101_MULTITURN_PREFIX PASS pairs=8 response_publications="<<publications
        <<" promotions="<<promotions<<" avoided_tokens="<<avoided
        <<" executed_tail_tokens="<<executed
        <<" cancellation="<<(cancellation_checked?"rollback":"fail")
        <<" reload="<<(reload_checked?"cleared_refilled":"fail")
        <<" output="<<output.string()<<std::endl;
}
