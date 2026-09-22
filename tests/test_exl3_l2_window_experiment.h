#pragma once

void run_l2_window_experiment(Exl3TextModel& target,Exl3Dflash2DraftModel& draft) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Clock=std::chrono::steady_clock;
    const std::filesystem::path fixtures=env("NINFER_COMPACT_PERF_DIR"),output=env("NINFER_COMPACT_PERF_OUTPUT");
    require(!fixtures.empty() && !output.empty() && target.max_context()>=4224 &&
        !std::filesystem::exists(output/"progress.jsonl"),"L2 window evidence/context extent");
    std::ofstream progress(output/"progress.jsonl");require(progress.good(),"L2 window progress file");
    const auto stop=load_ids((fixtures/"stop.ids").string());
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");_putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    auto inner=target.create_context(true);require(inner->try_enable_oscar_from_environment(),"L2 window OSCAR");
    inner->prepare_transaction();inner->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");_putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    TapStage stage;std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16ULL*kHidden*2));stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        staging[tap]=static_cast<std::uint16_t*>(stage.bulk.back()->get());stage.bulk_ptrs.push_back(staging[tap]);
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    struct Result {
        std::shared_ptr<const Request> state;
        std::vector<std::int64_t> tokens;
        std::vector<double> publication_ms,tentative_wait_ms;
        std::vector<std::size_t> release_sizes;
        std::array<std::uint64_t,9> l2_batch_hist{};
        std::uint64_t l0_attempted=0,l0_accepted=0,l0_replay=0,l0_reconstruction=0,l0_boundary_discarded=0;
        std::uint64_t l2_verification=0,l2_replay=0,l2_native=0,l2_restores=0,l2_windows=0,l2_rejections=0,partial_flushes=0,mandatory_flushes=0;
        std::uint64_t comparison_positions=0,downstream_discarded=0,attempted_candidates=0;
        std::size_t peak_pending_tokens=0,peak_pending_bytes=0;
        double wall_ms=0,first_release_ms=0,l2_ms=0,rebase_ms=0,sync_ms=0;
        std::uint64_t kv_h2d=0,kv_d2h=0;
        bool stopped=false,fault_injected=false;
    };
    const auto elapsed=[](auto begin){return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();};
    const auto run=[&](std::shared_ptr<const Request> request,int limit,int window,int corrupt_at=-1,int mandatory_flush=0) {
        Result result;result.tokens.reserve(limit);const auto start=Clock::now();
        int candidate_ordinal=0;bool fault_done=false;
        while(result.tokens.size()<static_cast<std::size_t>(limit) && !result.stopped) {
            const auto rebase_begin=Clock::now();request->restore_inner(*inner,draft,staging);result.rebase_ms+=elapsed(rebase_begin);
            std::vector<std::int64_t> pending;std::vector<double> ready_ms;
            const std::size_t boundary=std::min<std::size_t>(window,static_cast<std::size_t>(limit)-result.tokens.size());
            if(boundary<static_cast<std::size_t>(window)) ++result.partial_flushes;
            while(pending.size()<boundary) {
                const auto seed=sample_target(*inner);std::vector<std::int64_t> block(8,kMaskToken);block[0]=seed;
                const auto proposals=draft.propose_cached(block,inner->position(),inner->target_embedding(),
                    inner->target_lm_head_weights(),inner->target_lm_head_metadata(),kMaskToken);
                const auto local=verify_pending_round_transactional(*inner,draft,stage,seed,proposals,inner->position(),
                    nullptr,-1,0x7e00u,TransactionRepairMode::retain_prefix);
                std::vector<std::int64_t> produced{seed};produced.insert(produced.end(),local.emitted.begin(),local.emitted.end());
                result.l0_attempted+=local.attempted_rows;result.l0_accepted+=local.accepted;
                result.l0_replay+=local.replay_rows;result.l0_reconstruction+=local.state_reconstruction_rows;
                const auto keep=std::min(produced.size(),boundary-pending.size());result.l0_boundary_discarded+=produced.size()-keep;
                for(std::size_t i=0;i<keep;++i,++candidate_ordinal) {
                    auto token=produced[i];
                    if(!fault_done && candidate_ordinal==corrupt_at) {token=(token+1)%248320;fault_done=true;result.fault_injected=true;}
                    pending.push_back(token);ready_ms.push_back(elapsed(start));
                    if(ninfer::exl3::exl3_terminal_token(token,stop)) break;
                }
                result.peak_pending_tokens=std::max(result.peak_pending_tokens,pending.size());
                result.peak_pending_bytes=std::max(result.peak_pending_bytes,
                    pending.capacity()*sizeof(std::int64_t)+ready_ms.capacity()*sizeof(double));
                if(std::any_of(pending.begin(),pending.end(),[&](auto token){return ninfer::exl3::exl3_terminal_token(token,stop);})) break;
                if(mandatory_flush>0 && pending.size()>=static_cast<std::size_t>(mandatory_flush)) {++result.mandatory_flushes;break;}
            }
            ++result.l2_windows;result.attempted_candidates+=pending.size();
            const auto before=exact->host_kv_stats();const auto l2_begin=Clock::now();
            auto [updated,verified]=request->verify_compact(*exact,draft,staging,pending,stop);
            result.l2_ms+=elapsed(l2_begin);const auto after=exact->host_kv_stats();
            result.kv_h2d+=after.h2d_bytes-before.h2d_bytes;result.kv_d2h+=after.d2h_bytes-before.d2h_bytes;
            result.l2_verification+=verified.verification_rows;result.l2_replay+=verified.replay_rows;
            result.l2_native+=verified.native_invocations;result.l2_restores+=verified.root_restores;
            for(int i=1;i<=8;++i) result.l2_batch_hist[i]+=verified.native_batch_hist[i];
            result.l2_rejections+=verified.rejected;result.comparison_positions+=verified.accepted+(verified.rejected?1:0);
            if(verified.rejected) result.downstream_discarded+=pending.size()-verified.accepted;
            const auto sync_begin=Clock::now();cuda_check(cudaDeviceSynchronize(),"L2 window authoritative publication");result.sync_ms+=elapsed(sync_begin);
            const double released_at=elapsed(start);std::size_t visible_release=0;
            for(std::size_t i=0;i<verified.committed_tokens.size();++i) {
                const auto token=verified.committed_tokens[i];
                if(!ninfer::exl3::exl3_terminal_token(token,stop)) {
                    result.tokens.push_back(token);result.publication_ms.push_back(released_at);++visible_release;
                }
                if(i<verified.accepted && i<ready_ms.size()) result.tentative_wait_ms.push_back(released_at-ready_ms[i]);
            }
            if(visible_release) {result.release_sizes.push_back(visible_release);if(result.first_release_ms==0) result.first_release_ms=released_at;}
            request=std::move(updated);result.stopped=verified.stopped;
        }
        cuda_check(cudaDeviceSynchronize(),"L2 window whole decode complete");result.wall_ms=elapsed(start);result.state=std::move(request);
        require(corrupt_at<0 || result.fault_injected,"requested L2 window fault was not reached");return result;
    };
    const auto equal=[&](const Result& a,const Result& b,const char* label) {
        require(a.tokens==b.tokens && a.stopped==b.stopped &&
            a.state->state()->same_payload(*b.state->state()) && a.state->token_suffix()==b.state->token_suffix(),label);
        a.state->restore_draft(draft,staging);const auto ring=draft.export_host_ring(nullptr,false);
        b.state->restore_draft(draft,staging);require(ring->same_payload(*draft.export_host_ring(nullptr,false)),label);
    };
    const auto quantile=[](std::vector<double> values,double q) {
        if(values.empty()) return -1.0;std::sort(values.begin(),values.end());
        return values[static_cast<std::size_t>(std::ceil(q*values.size()))-1];
    };
    const auto emit=[&](const std::string& fixture,const char* mode,int run_id,const Result& r,double prefill_ms,double setup_ms,double uncached_first) {
        std::vector<double> gaps;for(std::size_t i=1;i<r.publication_ms.size();++i) gaps.push_back(r.publication_ms[i]-r.publication_ms[i-1]);
        const auto zeros=std::count_if(gaps.begin(),gaps.end(),[](double x){return x==0;});
        const auto max_gap=gaps.empty()?-1.0:*std::max_element(gaps.begin(),gaps.end());
        progress << std::setprecision(10) << "{\"fixture\":\"" << fixture << "\",\"mode\":\"" << mode
            << "\",\"run\":" << run_id << ",\"visible\":" << r.tokens.size() << ",\"wall_ms\":" << r.wall_ms
            << ",\"decode_tps\":" << r.tokens.size()*1000/r.wall_ms << ",\"prefill_ms\":" << prefill_ms
            << ",\"setup_ms\":" << setup_ms << ",\"uncached_ttft_ms\":" << uncached_first
            << ",\"cached_first_release_ms\":" << r.first_release_ms << ",\"l2_windows\":" << r.l2_windows
            << ",\"l2_verification_rows\":" << r.l2_verification << ",\"l2_replay_rows\":" << r.l2_replay
            << ",\"l2_native_invocations\":" << r.l2_native << ",\"l2_root_restores\":" << r.l2_restores
            << ",\"l2_rejections\":" << r.l2_rejections << ",\"comparison_positions\":" << r.comparison_positions
            << ",\"downstream_discarded\":" << r.downstream_discarded << ",\"l0_attempted_rows\":" << r.l0_attempted
            << ",\"l0_accepted\":" << r.l0_accepted << ",\"l0_replay_rows\":" << r.l0_replay
            << ",\"l0_reconstruction_rows\":" << r.l0_reconstruction << ",\"l0_boundary_discarded\":" << r.l0_boundary_discarded
            << ",\"candidate_attempts\":" << r.attempted_candidates << ",\"h2d_bytes\":" << r.kv_h2d << ",\"d2h_bytes\":" << r.kv_d2h
            << ",\"l2_ms\":" << r.l2_ms << ",\"rebase_ms\":" << r.rebase_ms << ",\"sync_ms\":" << r.sync_ms
            << ",\"release_events\":" << r.release_sizes.size() << ",\"gap_p50_ms\":" << quantile(gaps,.50)
            << ",\"gap_p95_ms\":" << quantile(gaps,.95) << ",\"gap_p99_ms\":" << quantile(gaps,.99)
            << ",\"max_silent_gap_ms\":" << max_gap << ",\"zero_gap_fraction\":" << (gaps.empty()?0.0:static_cast<double>(zeros)/gaps.size())
            << ",\"wait_p50_ms\":" << quantile(r.tentative_wait_ms,.50) << ",\"wait_p95_ms\":" << quantile(r.tentative_wait_ms,.95)
            << ",\"peak_pending_tokens\":" << r.peak_pending_tokens << ",\"peak_pending_bytes\":" << r.peak_pending_bytes
            << ",\"batch1\":" << r.l2_batch_hist[1] << ",\"batch8\":" << r.l2_batch_hist[8] << "}\n" << std::flush;
        std::cout << std::setprecision(10) << "L2_WINDOW_RUN fixture=" << fixture << " mode=" << mode << " run=" << run_id
            << " visible=" << r.tokens.size() << " tps=" << r.tokens.size()*1000/r.wall_ms << " wall_ms=" << r.wall_ms
            << " uncached_ttft_ms=" << uncached_first << " cached_first_release_ms=" << r.first_release_ms
            << " l2_windows=" << r.l2_windows << " l2_rows=" << r.l2_verification+r.l2_replay
            << " l2_native=" << r.l2_native << " l2_restores=" << r.l2_restores << " release_events=" << r.release_sizes.size()
            << " gap_p50_ms=" << quantile(gaps,.50) << " gap_p95_ms=" << quantile(gaps,.95) << " max_gap_ms=" << max_gap
            << " zero_gap_fraction=" << (gaps.empty()?0.0:static_cast<double>(zeros)/gaps.size()) << std::endl;
    };

    int pairs=0;
    if(env("NINFER_L2_WINDOW_GATES_ONLY")!="1") for(int fixture_index=0;fixture_index<2;++fixture_index) {
        const std::string fixture=fixture_index?"prose":"code";const auto ids=load_ids((fixtures/(fixture+".ids")).string());
        require(ids.size()==4096,"L2 window prompt extent");
        std::array<Result,2> results;std::array<double,2> prefill{},setup{},uncached{};
        const std::array<int,2> order=fixture_index?std::array<int,2>{1,0}:std::array<int,2>{0,1};
        for(int arm:order) {
            const auto admission=Clock::now(),prefill_begin=Clock::now();const auto root=Request::initialize(*exact,ids,1024);
            prefill[arm]=elapsed(prefill_begin);const auto setup_begin=Clock::now();const auto compact=root->compact_draft(draft,staging);setup[arm]=elapsed(setup_begin);
            results[arm]=run(compact,128,arm?64:8);uncached[arm]=prefill[arm]+setup[arm]+results[arm].first_release_ms;
            emit(fixture,arm?"WINDOW_ONLY":"BASELINE",0,results[arm],prefill[arm],setup[arm],uncached[arm]);
        }
        equal(results[0],results[1],"L2 window matched output/state");++pairs;
    }

    const auto ids=load_ids((fixtures/"code.ids").string());const auto root=Request::initialize(*exact,ids,1024);
    const auto compact=root->compact_draft(draft,staging);const auto partial_a=run(compact,80,8),partial_b=run(compact,80,64);
    equal(partial_a,partial_b,"L2 partial final window state");require(partial_b.partial_flushes>=1,"L2 partial final window absent");
    const auto fault_early=run(compact,80,64,1),fault_late=run(compact,80,64,62);
    equal(partial_a,fault_early,"L2 early repair state");equal(partial_a,fault_late,"L2 late repair state");
    require(fault_early.l2_rejections>=1 && fault_late.l2_rejections>=1 &&
        fault_early.l2_replay>0 && fault_late.l2_replay>0,"L2 forced repair accounting");
    const auto tool_flush=run(compact,80,64,-1,17);equal(partial_a,tool_flush,"L2 mocked boundary state");
    require(tool_flush.mandatory_flushes>=1,"L2 mocked boundary did not flush early");
    // Cancellation gate: produce private L0 work but invoke no L2 verification/publication.
    compact->restore_inner(*inner,draft,staging);std::vector<std::int64_t> cancelled_pending;
    while(cancelled_pending.size()<16) {
        const auto seed=sample_target(*inner);std::vector<std::int64_t> block(8,kMaskToken);block[0]=seed;
        const auto proposals=draft.propose_cached(block,inner->position(),inner->target_embedding(),
            inner->target_lm_head_weights(),inner->target_lm_head_metadata(),kMaskToken);
        const auto local=verify_pending_round_transactional(*inner,draft,stage,seed,proposals,inner->position(),
            nullptr,-1,0x7e00u,TransactionRepairMode::retain_prefix);
        cancelled_pending.push_back(seed);cancelled_pending.insert(cancelled_pending.end(),local.emitted.begin(),local.emitted.end());
    }
    require(cancelled_pending.size()>=16 && compact->state()->same_payload(*root->state()) &&
        compact->token_suffix()==root->token_suffix(),"L2 cancellation root ownership");
    std::cout << "L2_WINDOW_GATES PASS full_windows=1 partial_final=1 early_repair=1 late_repair=1 mocked_boundary=1 immutable_cancel_root=1\n";
    std::cout << "L2_WINDOW_EXPERIMENT PASS matched_pairs=" << pairs
        << " modes=BASELINE,WINDOW_ONLY cascade=NOT_IMPLEMENTED turboangle_forward=ABSENT serving_integration=HARNESS_ONLY\n";
}
