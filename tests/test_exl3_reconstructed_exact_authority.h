#pragma once

// Campaign draft: actual Request initialization/verification and resident publication.
// Fixed output count is NOT a completed chat turn; no completed-authority admission.
// Supersedes code-r1 timing only: INVALID_TIMING_OBSERVER_RETENTION. That earlier
// cohort remains functional evidence, not eligible performance evidence.
void run_reconstructed_exact_authority(Exl3TextModel& target,
                                       const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Cache=ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity=ninfer::exl3::Exl3VeriCacheServingIdentity;
    using Registry=ninfer::exl3::Exl3HostResidentSet;
    using Clock=std::chrono::steady_clock;
    using Root=std::shared_ptr<const Request>;
    const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
    const std::filesystem::path directory=env("NINFER_RECON_EXACT_AUTHORITY_OUT");
    require(!directory.empty() && !std::filesystem::exists(directory),"exact authority new output directory");
    std::filesystem::create_directories(directory);
    std::ofstream checks(directory/"correctness.csv"),runs(directory/"runs.csv"),gaps(directory/"releases.csv");
    require(checks.good() && runs.good() && gaps.good(),"exact authority evidence creation");
    checks << "case,boundary,pass,exports,calls,rows,workspace_bytes,persistent_bytes\n";
    runs << "pair,order,arm,construction_ms,prefill_ms,verify_ms,registry_ms,fresh_local_ttft_ms,first_release_ms,teardown_ms,wall_ms,exports,publications,calls,rows,workspace_bytes,persistent_bytes,registry_final_page_bytes,registry_peak_page_bytes,private_observer_roots,private_observer_union_bytes,pass\n";
    gaps << "pair,arm,release,committed,elapsed_ms,gap_ms\n";
    const auto gate=[&](const std::string& name,int boundary,bool pass,int exports=0,
                        std::uint64_t calls=0,std::uint64_t rows=0,std::size_t workspace=0,std::size_t persistent=0) {
        checks << name << ',' << boundary << ',' << pass << ',' << exports << ',' << calls << ',' << rows << ',' << workspace << ',' << persistent << '\n';
        checks.flush();require(checks.good(),"exact authority evidence flush");require(pass,"exact authority gate: "+name);
    };
    require(target.max_context()==4352 && source.size()>=4096,"exact authority fixture extent");
    for(const char* key:{"NINFER_EXL3_WIDE_PREFILL","NINFER_EXL3_PREFILL_STAGED_REDUCTION",
            "NINFER_EXL3_PREFILL_WIDE1024","NINFER_EXL3_PREFILL_DIRECT_PARTIALS","NINFER_EXL3_PREFILL_DIRECT_ASYNC_A"})
        require(env(key)=="1",std::string("exact authority direct flag ")+key);
    const auto configure=[](bool candidate) {
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");_putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
        _putenv_s("NINFER_EXL3_NATIVE_CONTINUATION16","0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC","0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC","0");
        _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7",candidate?"1":"0");
    };
    const auto create=[&](bool candidate) {configure(candidate);auto ctx=target.create_context(true);ctx->prepare_continuation(8);return ctx;};
    const auto memory=[] {MEMORYSTATUSEX m{};m.dwLength=sizeof(m);require(GlobalMemoryStatusEx(&m)!=0,"exact authority memory query");return m;};
    // Launcher should set these explicitly; bounded standalone defaults are 8 GiB/1024 MiB.
    const auto reserve_setting=env("NINFER_RECON_EXACT_AUTHORITY_RESERVE_GIB");
    const auto cache_setting=env("NINFER_RECON_EXACT_AUTHORITY_CACHE_MIB");
    const std::uint64_t reserve=std::stoull(reserve_setting.empty()?"8":reserve_setting)*(1ULL<<30);
    const std::uint64_t cache_budget=std::stoull(cache_setting.empty()?"1024":cache_setting)*(1ULL<<20);
    require(reserve>=(8ULL<<30) && cache_budget>0,"exact authority memory policy");
    const auto physical=memory();require(physical.ullAvailPhys>reserve,"exact authority physical reserve");
    Registry registry(physical.ullTotalPhys-reserve,reserve);
    std::vector<Root> canonical;
    std::uint64_t registry_pages=0,registry_peak=0;
    const auto live_roots=[&](const std::vector<Root>& active) {
        std::vector<Root> roots;if(!canonical.empty()) roots.push_back(canonical.front());
        roots.insert(roots.end(),active.begin(),active.end());return roots;
    };
    const auto pin=[&](const std::vector<Root>& active) {
        auto roots=live_roots(active);
        Registry::Snapshot snapshot;for(const auto& root:roots) snapshot.owners.push_back(root);
        Request::visit_host_allocations(roots,[&](const void* p,std::size_t n){snapshot.add(p,n);},false);
        const auto update=registry.replace(std::move(snapshot));
        registry_pages=update.page_bytes;registry_peak=std::max(registry_peak,registry_pages);
    };
    const auto probe=[&](const std::vector<Root>& active,const char* phase) {
        Exl3HostResidencyProbe observation;const auto roots=live_roots(active);
        Request::visit_host_allocations(roots,[&](const void* p,std::size_t n){observation.add(p,n);},false);
        const auto measured=observation.measure();
        gate(phase,0,measured.resident_tensor_bytes==measured.allocated_union_bytes &&
            measured.locked_tensor_bytes==measured.allocated_union_bytes);
    };
    const auto same=[](const Root& a,const Root& b) {
        return a && b && a->token_suffix()==b->token_suffix() && a->same_taps(*b) && a->state()->same_payload(*b->state());
    };
    const auto bytes=[](const Root& root) {
        std::vector<std::uint8_t> result;
        const std::array<Root,1> roots{root};
        Request::visit_host_allocations(roots,[&](const void* p,std::size_t n) {
            const auto* b=static_cast<const std::uint8_t*>(p);result.insert(result.end(),b,b+n);
        },false);return result;
    };
    const auto prompt=std::span<const std::int64_t>(source.data(),4096);
    std::vector<std::int64_t> trajectory;
    std::uint64_t eligible_calls=0,eligible_rows=0;std::size_t parent_persistent=0;
    struct Observed {std::uint64_t calls=0,rows=0;} observed;
    {
        auto ctx=create(false);
        ctx->set_target_projection_observer_for_test(
            [](const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
                if((x.rows!=256 && x.rows!=512 && x.rows!=1024) || x.metadata.K!=7 ||
                    x.metadata.in_features!=17408 || x.metadata.out_features!=5120 ||
                    x.metadata.mcg || !x.metadata.mul1 || x.metadata.has_bias ||
                    !x.operation || std::string(x.operation)!="down") return;
                auto& o=*static_cast<Observed*>(user);++o.calls;o.rows+=x.rows;
            },&observed,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
        auto current=Request::initialize(*ctx,prompt,1024);
        ctx->set_target_projection_observer_for_test(nullptr);
        canonical.push_back(current);pin({current});parent_persistent=ctx->persistent_bytes();
        eligible_calls=observed.calls;eligible_rows=observed.rows;
        for(int i=0;i<128;++i) {
            const auto token=sample_target(*ctx);const std::array<std::int64_t,1> proposal{token};
            auto [next,result]=current->verify(*ctx,proposal);
            pin({current,next}); // Residency precedes even the canonical trace record.
            if(i==0) probe({current,next},"before_first_correctness_publication");
            gate("canonical_M1",i+1,result.committed_tokens==std::vector<std::int64_t>{token} &&
                result.accepted==1 && !result.rejected && result.replay_rows==0,1);
            trajectory.push_back(token);current=std::move(next);
            pin({current}); // Retire previous publication; private snapshots stay CPU-only.
            if((i+1)%8==0) canonical.push_back(current);
        }
        probe({current},"after_final_canonical_publication");pin({});const auto s=ctx->reconstructed_exact_stats();
        gate("canonical_initialize",4096,eligible_calls>0 && s.calls==0 && s.rows==0 && s.workspace_bytes==0,
            1,s.calls,s.rows,s.workspace_bytes,parent_persistent);
    }
    const auto frozen_prompt=bytes(canonical.front());
    const auto frozen_mid=bytes(canonical[8]);
    struct Run {
        std::vector<Root> roots;std::vector<std::int64_t> output;std::vector<double> release;
        double construction=0,prefill=0,verify=0,registry_ms=0,teardown=0,wall=0;
        std::uint64_t calls=0,rows=0;std::size_t workspace=0,persistent=0;
        std::uint64_t registry_final=0,registry_max=0;
        bool protocol=true;
    };
    const auto execute=[&](bool candidate,bool correctness=false) {
        Run result;result.roots.reserve(correctness?17:2);result.output.reserve(128);result.release.reserve(16);
        cuda_check(cudaDeviceSynchronize(),"exact authority timed start");const auto begin=Clock::now();
        registry_peak=registry_pages;
        auto ctx=create(candidate);const auto constructed=Clock::now();
        auto current=Request::initialize(*ctx,prompt,1024);
        const auto prefilled=Clock::now();result.construction=ms(begin,constructed);result.prefill=ms(constructed,prefilled);
        result.roots.push_back(current);
        auto registration=Clock::now();pin({current});result.registry_ms+=ms(registration,Clock::now());
        for(int window=0;window<16;++window) {
            const auto started=Clock::now();
            auto [next,verified]=current->verify(*ctx,std::span<const std::int64_t>(trajectory).subspan(window*8,8));
            result.verify+=ms(started,Clock::now());
            result.protocol=result.protocol && verified.accepted==8 && !verified.rejected && !verified.stopped &&
                verified.verification_rows==8 && verified.replay_rows==0 && verified.executed_rows==8 && verified.native_invocations==1;
            if(correctness || window==15) result.roots.push_back(next);
            registration=Clock::now();pin({current,next});
            result.registry_ms+=ms(registration,Clock::now());
            if(correctness && window==0) probe({current,next},"before_first_paired_correctness_publication");
            // Actual local token publication follows completed immutable registration.
            result.output.insert(result.output.end(),verified.committed_tokens.begin(),verified.committed_tokens.end());
            result.release.push_back(ms(begin,Clock::now()));current=std::move(next);
            registration=Clock::now();pin({current});result.registry_ms+=ms(registration,Clock::now());
        }
        if(correctness) probe({current},"after_final_paired_correctness_publication");
        result.registry_final=registry_pages;result.registry_max=registry_peak;
        const auto s=ctx->reconstructed_exact_stats();result.calls=s.calls;result.rows=s.rows;
        result.workspace=s.workspace_bytes;result.persistent=ctx->persistent_bytes();
        const auto teardown=Clock::now();ctx.reset();cuda_check(cudaDeviceSynchronize(),"exact authority teardown");
        // Correctness retains all boundaries; timing retains only initial/final.
        // Intermediate owners retire on each handoff, including registry unlock
        // and CPU destruction in this interval. Initial/final oracle observations
        // are retained symmetrically for post-timing comparison.
        pin({});const auto ended=Clock::now();result.teardown=ms(teardown,ended);result.wall=ms(begin,ended);
        return result;
    };
    const auto compare=[&](const Run& r,bool candidate,const std::string& phase) {
        bool exact=r.roots.size()==canonical.size() && r.output==trajectory && r.protocol;
        for(std::size_t i=0;i<r.roots.size() && i<canonical.size();++i) {
            const bool equal=same(r.roots[i],canonical[i]);exact=exact && equal;
            gate(phase,4096+static_cast<int>(8*i),equal,0);
        }
        const bool stats=candidate?(r.calls==eligible_calls && r.rows==eligible_rows && r.workspace==178257920 &&
            r.persistent==parent_persistent+178257920):(r.calls==0 && r.rows==0 && r.workspace==0 && r.persistent==parent_persistent);
        gate(phase+"_stats",4224,exact && stats && bytes(canonical.front())==frozen_prompt && bytes(canonical[8])==frozen_mid,
            17,r.calls,r.rows,r.workspace,r.persistent);return exact && stats;
    };
    // Numerical/state checks complete before matched timings; observers are absent.
    for(bool candidate:{false,true}) {const auto result=execute(candidate,true);compare(result,candidate,candidate?"candidate":"parent8");}

    // Cancellation and cache reload use a declared INPUT prefix frontier only.
    // No arbitrary generated suffix is admitted as a completed conversational turn.
    {
        auto ctx=create(true);
        const Identity identity{env("NINFER_RECON_EXACT_AUTHORITY_NAMESPACE"),
            env("NINFER_RECON_EXACT_AUTHORITY_ARTIFACT_ID"),env("NINFER_RECON_EXACT_AUTHORITY_TOKENIZER_ID"),
            env("NINFER_RECON_EXACT_AUTHORITY_CONFIGURATION_ID"),env("NINFER_RECON_EXACT_AUTHORITY_MODALITY_ID")};
        Cache cache(Cache::Policy{1,64,cache_budget,reserve},identity);
        int reached=0;bool cancelled=false;std::shared_ptr<const Request> partial;
        try {partial=Request::initialize(*ctx,prompt,1024,[&](int position) {
            reached=position;if(position>=1040) throw std::runtime_error("intentional exact authority cancellation");
        });} catch(const std::runtime_error& e) {cancelled=std::string(e.what())=="intentional exact authority cancellation";if(!cancelled) throw;}
        gate("cancel_initialize_no_admission_or_publication",reached,cancelled && reached==1040 && !partial && cache.roots().empty(),0);
        ctx->restore_exact_host_state(*canonical.front()->state());
        auto proposal=std::vector<std::int64_t>(trajectory.begin(),trajectory.begin()+8);proposal[3]=(proposal[3]+1)%248320;
        auto [repaired,decision]=canonical.front()->verify(*ctx,proposal);pin({repaired});
        // Independent M1 authoritative Request repair oracle, same context reused after restore.
        auto expected=canonical.front();ctx->restore_exact_host_state(*expected->state());
        for(int i=0;i<4;++i) {const std::array<std::int64_t,1> one{trajectory[i]};expected=expected->verify(*ctx,one).first;}
        gate("cancel_oldroot_repair",4100,decision.rejected && decision.accepted==3 &&
            decision.committed_tokens==std::vector<std::int64_t>(trajectory.begin(),trajectory.begin()+4) && same(repaired,expected) &&
            bytes(canonical.front())==frozen_prompt,5);
        const auto admitted=cache.prepare(*ctx,prompt,4096,memory().ullAvailPhys,1024);pin({admitted.request});
        gate("declared_input_frontier_admission",4096,admitted.metrics.admitted && same(admitted.request,canonical.front()),1);
        const auto hit=cache.prepare(*ctx,prompt,4096,memory().ullAvailPhys,1024);pin({hit.request});
        gate("declared_input_frontier_hit",4096,hit.metrics.cache_hit && hit.metrics.reused_prompt_tokens==4096 && same(hit.request,canonical.front()),0);
        auto changed=identity;changed.configuration_identity+=";reload-test";cache.reset_for_model_reload(changed);
        gate("reload_invalidates_input_frontier",4096,cache.roots().empty(),0);
        const auto reloaded=cache.prepare(*ctx,prompt,4096,memory().ullAvailPhys,1024);pin({reloaded.request});
        gate("reload_recomputes_input_frontier",4096,!reloaded.metrics.cache_hit && reloaded.metrics.admitted && same(reloaded.request,canonical.front()),1);
        pin({});
    }
    // Retire private intermediate canonical observers before the timed cohort.
    // The frozen prompt byte copy remains only an explicit CPU oracle; it owns
    // no live Request pages and cannot extend a publication allocation lifetime.
    canonical=std::vector<Root>{canonical.front(),canonical.back()};
    pin({});
    for(int pair=0;pair<6;++pair) for(int order=0;order<2;++order) {
        const bool candidate=pair%2?order==0:order==1;const auto result=execute(candidate);
        // Expensive comparisons and reports are outside all run timestamps.
        bool exact=result.roots.size()==canonical.size() && result.output==trajectory && result.protocol;
        for(std::size_t i=0;i<result.roots.size() && i<canonical.size();++i) exact=exact && same(result.roots[i],canonical[i]);
        exact=exact && (candidate?(result.calls==eligible_calls && result.rows==eligible_rows && result.workspace==178257920 && result.persistent==parent_persistent+178257920):
            (result.calls==0 && result.rows==0 && result.workspace==0 && result.persistent==parent_persistent));
        // Read-only union accounting after timing; no state payload copies here.
        auto observers=canonical;observers.insert(observers.end(),result.roots.begin(),result.roots.end());
        Exl3HostResidencyProbe observer_inventory;
        Request::visit_host_allocations(observers,[&](const void* p,std::size_t n){observer_inventory.add(p,n);},false);
        const auto observer_bytes=observer_inventory.measure().allocated_union_bytes;
        runs << pair << ',' << (pair%2?"BA":"AB") << ',' << (candidate?"B":"A") << ',' << result.construction << ',' << result.prefill << ','
            << result.verify << ',' << result.registry_ms << ',' << result.release.front() << ',' << result.release.front() << ',' << result.teardown << ',' << result.wall
            << ",17,16," << result.calls << ',' << result.rows << ',' << result.workspace << ',' << result.persistent << ','
            << result.registry_final << ',' << result.registry_max << ',' << observers.size() << ',' << observer_bytes << ',' << exact << '\n';
        for(std::size_t i=0;i<result.release.size();++i) gaps << pair << ',' << (candidate?"B":"A") << ',' << i+1 << ",8," << result.release[i]
            << ',' << result.release[i]-(i?result.release[i-1]:0) << '\n';
        runs.flush();gaps.flush();require(runs.good() && gaps.good(),"exact authority timed evidence flush");
        require(exact,"exact authority timed zero-bit gate");
    }
    gate("old_prompt_immutable_final",4096,bytes(canonical.front())==frozen_prompt);
    registry.close();
    std::cout << "RECONSTRUCTED_EXACT_AUTHORITY PASS prompt=4096 useful_tokens=128 windows=16 pairs=6 exports_per_run=17 timed_observer_roots=4 intermediate_retirement=in_wall target_L2=1 completed_turn_frontend=PENDING no_HTTP=1\n";
}
