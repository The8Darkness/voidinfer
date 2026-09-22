#pragma once

void run_compact_draft_performance(Exl3TextModel& target,Exl3Dflash2DraftModel& draft) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    const std::filesystem::path fixtures=env("NINFER_COMPACT_PERF_DIR"),output=env("NINFER_COMPACT_PERF_OUTPUT");
    require(!fixtures.empty() && !output.empty() && target.max_context()>=4224 &&
        !std::filesystem::exists(output/"progress.jsonl"),"compact perf evidence/context extent");
    std::ofstream progress(output/"progress.jsonl");require(progress.good(),"compact perf progress file");
    const auto terminal=load_ids((fixtures/"stop.ids").string());
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");_putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    auto inner=target.create_context(true);require(inner->try_enable_oscar_from_environment(),"compact perf OSCAR");
    inner->prepare_transaction();inner->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");_putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    TapStage stage;std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16ULL*kHidden*2));stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        staging[tap]=static_cast<std::uint16_t*>(stage.bulk.back()->get());stage.bulk_ptrs.push_back(staging[tap]);
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    if(env("NINFER_COMPACT_PROFILE")=="1") {
        const auto ids=load_ids((fixtures/"code.ids").string());
        require(ids.size()==4096,"compact profile prompt extent");
        const auto now=[] {return std::chrono::steady_clock::now();};
        const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
        const auto prefill_begin=now();
        const auto full=Request::initialize(*exact,ids,1024);
        const auto prefill_end=now();
        const auto compact_begin=now();
        auto request=full->compact_draft(draft,staging);
        const auto compact_end=now();
        double target_restore_ms=0,draft_restore_ms=0,seed_ms=0,draft_proposal_ms=0;
        double inner_verify_ms=0,outer_verify_ms=0,post_restore_ms=0,publication_sync_ms=0;
        double total_ms=0,first_release_ms=0;
        std::uint64_t committed=0,visible=0,inner_accepted=0,inner_attempted=0;
        std::uint64_t inner_reconstruction=0,outer_executed=0,outer_matched=0,outer_rejections=0;
        std::uint64_t kv_h2d=0,kv_d2h=0;
        const auto total_begin=now();
        while(visible<32) {
            auto begin=now();inner->restore_oscar_host_state(*request->state());auto end=now();target_restore_ms+=ms(begin,end);
            begin=now();request->restore_draft(draft,staging);end=now();draft_restore_ms+=ms(begin,end);
            begin=now();const auto seed=sample_target(*inner);end=now();seed_ms+=ms(begin,end);
            std::vector<std::int64_t> block(8,kMaskToken);block[0]=seed;
            begin=now();const auto proposals=draft.propose_cached(block,inner->position(),inner->target_embedding(),
                inner->target_lm_head_weights(),inner->target_lm_head_metadata(),kMaskToken);end=now();draft_proposal_ms+=ms(begin,end);
            begin=now();const auto local=verify_pending_round_transactional(*inner,draft,stage,seed,proposals,inner->position(),
                nullptr,-1,0x7e00u,TransactionRepairMode::retain_prefix);end=now();inner_verify_ms+=ms(begin,end);
            std::vector<std::int64_t> tentative{seed};tentative.insert(tentative.end(),local.emitted.begin(),local.emitted.end());
            tentative.resize(std::min<std::size_t>({tentative.size(),8,32-visible}));
            const auto transfers=exact->host_kv_stats();
            begin=now();auto [updated,verified]=request->verify_compact(*exact,draft,staging,tentative,terminal);end=now();outer_verify_ms+=ms(begin,end);
            const auto transferred=exact->host_kv_stats();
            kv_h2d+=transferred.h2d_bytes-transfers.h2d_bytes;kv_d2h+=transferred.d2h_bytes-transfers.d2h_bytes;
            if(env("NINFER_COMPACT_DEFER_POST_RESTORE")!="1") {
                begin=now();updated->restore_inner(*inner,draft,staging);end=now();post_restore_ms+=ms(begin,end);
            }
            begin=now();cuda_check(cudaDeviceSynchronize(),"compact profile publication");end=now();publication_sync_ms+=ms(begin,end);
            request=std::move(updated);committed+=verified.committed_tokens.size();
            for(auto token:verified.committed_tokens) if(!ninfer::exl3::exl3_terminal_token(token,terminal)) ++visible;
            inner_accepted+=local.accepted;inner_attempted+=local.attempted_rows;inner_reconstruction+=local.state_reconstruction_rows;
            outer_executed+=verified.executed_rows;outer_matched+=verified.accepted;outer_rejections+=verified.rejected;
            if(first_release_ms==0) first_release_ms=ms(total_begin,now());
            require(!verified.stopped && visible<=32,"compact profile unexpected terminal/extent");
        }
        total_ms=ms(total_begin,now());
        progress << "{\"event\":\"compact_profile\",\"visible\":" << visible << ",\"committed\":" << committed
            << ",\"prefill_ms\":" << ms(prefill_begin,prefill_end) << ",\"compact_setup_ms\":" << ms(compact_begin,compact_end)
            << ",\"target_restore_ms\":" << target_restore_ms << ",\"draft_restore_ms\":" << draft_restore_ms
            << ",\"seed_ms\":" << seed_ms << ",\"draft_proposal_ms\":" << draft_proposal_ms
            << ",\"inner_verify_ms\":" << inner_verify_ms << ",\"outer_verify_ms\":" << outer_verify_ms
            << ",\"post_restore_ms\":" << post_restore_ms << ",\"publication_sync_ms\":" << publication_sync_ms
            << ",\"total_ms\":" << total_ms << "}\n" << std::flush;
        std::cout << std::setprecision(10) << "COMPACT_PROFILE PASS prompt=4096 visible=" << visible << " committed=" << committed
            << " decode_tps=" << visible*1000/total_ms << " prefill_ms=" << ms(prefill_begin,prefill_end)
            << " compact_setup_ms=" << ms(compact_begin,compact_end) << " first_release_ms=" << first_release_ms
            << " target_restore_ms=" << target_restore_ms << " draft_restore_ms=" << draft_restore_ms
            << " seed_ms=" << seed_ms << " draft_proposal_ms=" << draft_proposal_ms << " inner_verify_ms=" << inner_verify_ms
            << " outer_verify_ms=" << outer_verify_ms << " post_restore_ms=" << post_restore_ms
            << " publication_sync_ms=" << publication_sync_ms << " total_ms=" << total_ms
            << " authoritative_layer_kv_h2d=" << kv_h2d << " authoritative_layer_kv_d2h=" << kv_d2h
            << " inner_accepted=" << inner_accepted << " inner_attempted_rows=" << inner_attempted
            << " inner_reconstruction_rows=" << inner_reconstruction << " outer_executed_rows=" << outer_executed
            << " outer_matched=" << outer_matched << " outer_rejections=" << outer_rejections
            << " queue_wait_ms=unavailable repair_ms=unavailable publication_ms=sync_only" << std::endl;
        return;
    }
    struct Result {
        std::shared_ptr<const Request> state;
        std::vector<std::int64_t> tokens;
        std::array<std::uint64_t,8> work{};
        double elapsed_ms=0,first_release_ms=0;
        bool stopped=false;
    };
    const auto run=[&](std::shared_ptr<const Request> request,int limit,bool one_round,bool restore_after=true) {
        Result result;result.tokens.reserve(limit);
        const auto start=std::chrono::steady_clock::now();
        do {
            request->restore_inner(*inner,draft,staging);
            const auto seed=sample_target(*inner);std::vector<std::int64_t> block(8,kMaskToken);block[0]=seed;
            const auto proposals=draft.propose_cached(block,inner->position(),inner->target_embedding(),
                inner->target_lm_head_weights(),inner->target_lm_head_metadata(),kMaskToken);
            const auto local=verify_pending_round_transactional(*inner,draft,stage,seed,proposals,inner->position(),
                nullptr,-1,0x7e00u,TransactionRepairMode::retain_prefix);
            std::vector<std::int64_t> tentative{seed};tentative.insert(tentative.end(),local.emitted.begin(),local.emitted.end());
            const auto produced=tentative.size();
            tentative.resize(std::min<std::size_t>({tentative.size(),8,static_cast<std::size_t>(limit)-result.tokens.size()}));
            auto [updated,verified]=request->compact_draft()?request->verify_compact(*exact,draft,staging,tentative,terminal):
                request->verify(*exact,tentative,terminal);
            if(restore_after) updated->restore_inner(*inner,draft,staging);
            cuda_check(cudaDeviceSynchronize(),"compact perf authoritative release");
            result.tokens.insert(result.tokens.end(),verified.committed_tokens.begin(),verified.committed_tokens.end());
            result.work[0]+=local.accepted;result.work[1]+=local.attempted_rows;result.work[2]+=local.replay_rows;
            result.work[3]+=local.state_reconstruction_rows;result.work[4]+=verified.accepted;result.work[5]+=verified.executed_rows;
            result.work[6]+=verified.rejected;result.work[7]+=produced-tentative.size();
            request=std::move(updated);result.stopped=verified.stopped;
            if(result.first_release_ms==0) result.first_release_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(one_round) break;
        } while(result.tokens.size()<static_cast<std::size_t>(limit) && !result.stopped);
        cuda_check(cudaDeviceSynchronize(),"compact perf whole decode complete");
        result.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        result.state=std::move(request);return result;
    };
    if(env("NINFER_T77_DIRECT_TAP_PERF")=="1") {
        constexpr int repeats=3,limit=128;
        int pairs=0,positive=0;
        std::cout << std::setprecision(10);
        for(const std::string fixture:{"code","prose"}) {
            const auto ids=load_ids((fixtures/(fixture+".ids")).string());
            require(ids.size()==4096,"T77 prompt extent");
            const auto root=Request::initialize(*exact,ids,1024);
            const auto compact=root->compact_draft(draft,staging);
            for(int arm=0;arm<2;++arm) {
                _putenv_s("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING",arm?"1":"0");
                (void)run(compact,8,true);
            }
            for(int repeat=0;repeat<repeats;++repeat) {
                std::array<Result,2> results;
                for(int order=0;order<2;++order) {
                    const int arm=(repeat&1)?1-order:order;
                    _putenv_s("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING",arm?"1":"0");
                    results[arm]=run(compact,limit,false);
                    const auto path=output/(fixture+"-pair-"+std::to_string(repeat)+
                        "-arm-"+std::to_string(arm)+".ids");
                    require(!std::filesystem::exists(path),"preserve T77 output");
                    std::ofstream file(path);for(auto token:results[arm].tokens) file<<token<<'\n';
                    file.close();require(file.good(),"T77 output file");
                }
                _putenv_s("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING","0");
                const auto& control=results[0];const auto& candidate=results[1];
                require(control.tokens.size()==limit && !control.stopped &&
                    control.tokens==candidate.tokens && control.stopped==candidate.stopped &&
                    control.work==candidate.work &&
                    control.state->state()->same_payload(*candidate.state->state()) &&
                    control.state->token_suffix()==candidate.state->token_suffix(),
                    "T77 output/state/work oracle failed");
                control.state->restore_draft(draft,staging);
                const auto ring=draft.export_host_ring(nullptr,false);
                candidate.state->restore_draft(draft,staging);
                require(ring->same_payload(*draft.export_host_ring(nullptr,false)),
                    "T77 projected draft ring differs");
                const double gain=(control.elapsed_ms-candidate.elapsed_ms)*100.0/control.elapsed_ms;
                positive+=gain>0; ++pairs;
                progress << "{\"fixture\":\"" << fixture << "\",\"pair\":" << repeat
                    << ",\"control_ms\":" << control.elapsed_ms << ",\"candidate_ms\":"
                    << candidate.elapsed_ms << ",\"gain_pct\":" << gain
                    << ",\"removed_tap_payload_bytes\":6553600,\"exact\":true}\n" << std::flush;
                std::cout << "T77_WORKSPACE_PAIR fixture=" << fixture << " pair=" << repeat
                    << " order=" << ((repeat&1)?"BA":"AB") << " visible_tokens=128"
                    << " control_ms=" << control.elapsed_ms
                    << " candidate_ms=" << candidate.elapsed_ms << " gain_pct=" << gain
                    << " control_tps=" << limit*1000/control.elapsed_ms
                    << " candidate_tps=" << limit*1000/candidate.elapsed_ms
                    << " removed_tap_payload_bytes=6553600 work=exact state=exact ring=exact\n";
            }
        }
        _putenv_s("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING","0");
        std::cout << "T77_WORKSPACE_REUSE PASS pairs=" << pairs << " positive=" << positive
            << " fixtures=2 tokens_per_arm=128 transient_tap_payload_removed_per_arm=6553600"
               " explicit_opt_in=1\n";
        return;
    }
    if(env("NINFER_COMPACT_RESTORE_PERF")=="1") {
        const int repeats=std::stoi(env("NINFER_COMPACT_RESTORE_REPEATS"));
        const int limit=std::stoi(env("NINFER_COMPACT_RESTORE_TOKENS"));
        require(repeats>=1 && repeats<=3 && (limit==64 || limit==128),"compact restore performance extent");
        int pairs=0;
        std::cout << std::setprecision(10);
        for(const std::string fixture:{"code","prose"}) {
            const auto ids=load_ids((fixtures/(fixture+".ids")).string());require(ids.size()==4096,"compact restore prompt extent");
            const auto prefill_start=std::chrono::steady_clock::now();
            const auto root=Request::initialize(*exact,ids,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prefill_start).count();
            const auto compact_start=std::chrono::steady_clock::now();
            const auto compact=root->compact_draft(draft,staging);
            const double compact_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compact_start).count();
            for(int arm=0;arm<2;++arm) (void)run(compact,8,true,arm==0);
            for(int repeat=0;repeat<repeats;++repeat) {
                std::array<Result,2> results;
                for(int order=0;order<2;++order) {
                    const int arm=(repeat&1)?1-order:order;
                    results[arm]=run(compact,limit,false,arm==0);
                    const auto path=output/(fixture+"-pair-"+std::to_string(repeat)+"-arm-"+std::to_string(arm)+".ids");
                    require(!std::filesystem::exists(path),"preserve compact restore output");
                    std::ofstream file(path);for(auto token:results[arm].tokens) file<<token<<'\n';file.close();require(file.good(),"compact restore output file");
                    progress << "{\"fixture\":\"" << fixture << "\",\"pair\":" << repeat << ",\"defer_restore\":" << arm
                        << ",\"tokens\":" << results[arm].tokens.size() << ",\"wall_ms\":" << results[arm].elapsed_ms << "}\n" << std::flush;
                }
                const auto& reference=results[0];const auto& candidate=results[1];
                require(reference.tokens.size()==static_cast<std::size_t>(limit) && !reference.stopped &&
                    reference.tokens==candidate.tokens && reference.stopped==candidate.stopped && reference.work==candidate.work &&
                    reference.state->state()->same_payload(*candidate.state->state()) &&
                    reference.state->token_suffix()==candidate.state->token_suffix(),"compact deferred restore output/state/work oracle failed");
                reference.state->restore_draft(draft,staging);const auto ring=draft.export_host_ring(nullptr,false);
                candidate.state->restore_draft(draft,staging);require(ring->same_payload(*draft.export_host_ring(nullptr,false)),
                    "compact deferred restore final draft state differs");
                std::cout << "COMPACT_RESTORE_PAIR fixture=" << fixture << " pair=" << repeat << " visible_tokens=" << limit
                    << " reference_ms=" << reference.elapsed_ms << " candidate_ms=" << candidate.elapsed_ms
                    << " reference_tps=" << limit*1000/reference.elapsed_ms << " candidate_tps=" << limit*1000/candidate.elapsed_ms
                    << " reference_first_release_ms=" << reference.first_release_ms << " candidate_first_release_ms=" << candidate.first_release_ms
                    << " inner_accepted=" << reference.work[0] << " inner_rows=" << reference.work[1]
                    << " outer_rows=" << reference.work[5] << " outer_rejections=" << reference.work[6]
                    << " prefill_ms=" << prefill_ms << " compact_setup_ms=" << compact_ms << " work=exact state=exact" << std::endl;
                ++pairs;
            }
        }
        std::cout << "COMPACT_RESTORE_PERF PASS pairs=" << pairs << " fixtures=2 tokens_per_arm=" << limit
            << " candidate=defer_physical_inner_restore_until_next_lease explicit_opt_in=1\n";
        return;
    }
    int pairs=0;
    std::cout << std::setprecision(10);
    for(const std::string fixture:{"code","prose","structured"}) {
        const auto ids=load_ids((fixtures/(fixture+".ids")).string());require(ids.size()==4096,"compact perf prompt extent");
        const auto prefill_start=std::chrono::steady_clock::now();
        const auto root=Request::initialize(*exact,ids,1024);
        const double prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prefill_start).count();
        const auto compact_start=std::chrono::steady_clock::now();
        const std::array<std::shared_ptr<const Request>,2> roots{root,root->compact_draft(draft,staging)};
        const double compact_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-compact_start).count();
        std::cout << "COMPACT_PERF_SETUP fixture=" << fixture << " prompt=4096 prefill_ms=" << prefill_ms
            << " prefill_tps=" << 4096*1000/prefill_ms << " compact_setup_ms=" << compact_ms << '\n';
        for(int arm=0;arm<2;++arm) {
            const auto warm=run(roots[arm],8,true);
            std::cout << "COMPACT_PERF_WARM fixture=" << fixture << " compact=" << arm << " tokens=" << warm.tokens.size()
                << " wall_ms=" << warm.elapsed_ms << '\n';
        }
        for(int repeat=0;repeat<3;++repeat) {
            std::array<Result,2> results;
            for(int order=0;order<2;++order) {
                const int arm=(repeat&1)?1-order:order;
                const auto path=output/(fixture+"-pair-"+std::to_string(repeat)+"-arm-"+std::to_string(arm)+".ids");
                require(!std::filesystem::exists(path),"preserve compact perf output");
                results[arm]=run(roots[arm],128,false);
                std::ofstream file(path);for(auto token:results[arm].tokens) file<<token<<'\n';file.close();require(file.good(),"compact perf output file");
                progress << "{\"fixture\":\"" << fixture << "\",\"pair\":" << repeat << ",\"compact\":" << arm
                    << ",\"tokens\":" << results[arm].tokens.size() << ",\"wall_ms\":" << results[arm].elapsed_ms << "}\n" << std::flush;
            }
            const auto& a=results[0];const auto& b=results[1];
            require(a.tokens.size()==128 && !a.stopped && a.tokens==b.tokens && a.stopped==b.stopped && a.work==b.work &&
                a.state->state()->same_payload(*b.state->state()) && a.state->token_suffix()==b.state->token_suffix(),
                "compact perf long output/state/work oracle failed");
            a.state->restore_draft(draft,staging);const auto reference=draft.export_host_ring(nullptr,false);
            b.state->restore_draft(draft,staging);require(reference->same_payload(*draft.export_host_ring(nullptr,false)),"compact perf final draft state differs");
            std::cout << "COMPACT_PERF_PAIR fixture=" << fixture << " pair=" << repeat << " visible_tokens=128 tap_ms=" << a.elapsed_ms
                << " compact_ms=" << b.elapsed_ms << " tap_tps=" << 128*1000/a.elapsed_ms << " compact_tps=" << 128*1000/b.elapsed_ms
                << " tap_first_release_ms=" << a.first_release_ms << " compact_first_release_ms=" << b.first_release_ms
                << " inner_accepted=" << a.work[0] << " inner_rows=" << a.work[1] << " outer_rows=" << a.work[5]
                << " outer_rejections=" << a.work[6] << " work=exact state=exact" << std::endl;
            ++pairs;
        }
    }
    std::cout << "COMPACT_PERF PASS pairs=" << pairs << " fixtures=3 tokens_per_arm=128 timing=continuous_uninstrumented_wall\n";
}
