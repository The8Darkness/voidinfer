#pragma once

// Draft: requires root's native16 ordinary-exact admission and 17-bin outer histogram.
// No acceptance-survival, serving, publication or cancellation claim.
void run_native16_qualification(Exl3TextModel& target,
                                const std::vector<std::int64_t>& source) {
    using State=ninfer::exl3::Exl3ExactHostState;
    using Clock=std::chrono::steady_clock;
    const std::filesystem::path directory=env("NINFER_NATIVE16_OUT");
    require(!directory.empty() && !std::filesystem::exists(directory),"native16 new evidence directory");
    std::filesystem::create_directories(directory);
    std::ofstream checks(directory/"correctness.csv"),pairs(directory/"pairs.csv"),projections(directory/"projections.csv");
    require(checks.good() && pairs.good() && projections.good(),"native16 evidence creation");
    projections << "prefix,arm,layer,operation,rows,K,in_features,out_features,topology,calls,gpu_us\n";
    checks << "prefix,case,width,ordinal,pass,verification_rows,replay_rows,executed_rows,committed_rows,native_invocations,root_restores,exports,h2d_bytes,copy_submissions,persistent_bytes,histogram\n";
    pairs << "prefix,pair,order,arm,restore_ms,compute_ms,export_ms,wall_ms,exports,h2d_bytes,copy_submissions,persistent_bytes,state_exact\n";
    require(target.max_context()==4352 && source.size()>=4096,"native16 source/context extent");
    const auto configure=[](bool native) {
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
        _putenv_s("NINFER_EXL3_NATIVE_CONTINUATION16",native?"1":"0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7","0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC","0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC","0");
        _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","0");
    };
    const auto create=[&](bool native) {
        configure(native);auto ctx=target.create_context(true);ctx->prepare_continuation(native?16:8);return ctx;
    };
    const auto frozen_bytes=[](const std::shared_ptr<const State>& state) {
        std::vector<std::uint8_t> bytes;bytes.reserve(state->payload_bytes());
        const std::array<std::shared_ptr<const State>,1> states{state};
        State::visit_host_allocations(states,[&](const void* p,std::size_t n) {
            const auto* b=static_cast<const std::uint8_t*>(p);bytes.insert(bytes.end(),b,b+n);
        },false);return bytes;
    };
    const auto gate=[&](int prefix,const std::string& name,int width,int ordinal,bool pass,
                        std::size_t memory=0,int exports=0) {
        checks << prefix << ',' << name << ',' << width << ',' << ordinal << ',' << pass
               << ",0,0,0,0,0,0," << exports << ",0,0," << memory << ",none\n";
        checks.flush();require(checks.good(),"native16 evidence flush");
        require(pass,"native16 zero-bit gate: "+name);
    };
    struct PayloadRows {
        std::vector<std::uint16_t> logits,embedding;
        std::array<std::vector<std::uint16_t>,5> taps;
    };
    const auto append=[](auto& dst,const auto& src) {dst.insert(dst.end(),src.begin(),src.end());};
    const auto capture=[&](Exl3TextContext& ctx,int rows,PayloadRows& out) {
        if(rows==1) append(out.logits,target_continue_device_bits(ctx.logits_device(),kVocab,"native16 M1 logits"));
        else append(out.logits,ctx.continuation_logits_bits_host());
        append(out.embedding,ctx.embedding_bits_host_for_test());
        const auto taps=ctx.exact_tap_rows_host();
        for(int t=0;t<5;++t) append(out.taps[t],taps[t]);
    };
    const auto row_equal=[](const PayloadRows& a,const PayloadRows& b) {
        return a.logits==b.logits && a.embedding==b.embedding && a.taps==b.taps;
    };
    _putenv_s("NINFER_EXL3_TARGET_PROJECTION_TIMING","0");
    for(bool native:{false,true}) {
        configure(native);auto ctx=target.create_context(true);
        bool rejected=false;
        try {ctx->prepare_continuation(native?9:16);} catch(const std::exception&) {rejected=true;}
        gate(0,native?"capacity9_rejected":"unopted16_rejected",native?9:16,0,
            rejected && ctx->continuation_capacity()==0);
        ctx->prepare_continuation(native?16:8);
        gate(0,"valid_capacity_after_rejection",native?16:8,0,ctx->continuation_capacity()==(native?16:8));
    }
    // Pure decision tests separate ordinal policy from repeated real token IDs.
    std::vector<std::int64_t> unique(16);
    for(int i=0;i<16;++i) unique[i]=1000+i;
    for(int stop:{8,9,16}) for(bool mismatch:{false,true}) {
        auto tentative=unique;if(mismatch) tentative[stop-1]=2000+stop;
        const std::array<std::int64_t,1> terminal{unique[stop-1]};
        const auto decision=ninfer::exl3::decide_exl3_outer_prefix(unique,tentative,terminal);
        gate(0,mismatch?"synthetic_mismatch_terminal":"synthetic_terminal",16,stop,
            decision.stopped && decision.rejected==mismatch &&
            decision.accepted==static_cast<std::size_t>(stop-(mismatch?1:0)) &&
            decision.committed_tokens==std::vector<std::int64_t>(unique.begin(),unique.begin()+stop));
    }
    for(int prefix:{16,4096}) {
        std::shared_ptr<const State> root,final;
        std::vector<std::int64_t> greedy;PayloadRows serial_rows;
        {
            auto ctx=create(false);
            if(prefix==16) ctx->prefill(std::span<const std::int64_t>(source.data(),16));
            else t69b_ingest(*ctx,std::span<const std::int64_t>(source.data(),4096));
            root=ctx->export_exact_host_state();
            for(int row=0;row<64;++row) {
                const auto token=sample_target(*ctx);greedy.push_back(token);ctx->decode(token);
                if(row<16) capture(*ctx,1,serial_rows);
                if(row==15) final=ctx->export_exact_host_state();
            }
        }
        const auto immutable=frozen_bytes(root);
        gate(prefix,"serial_fixture",16,0,root->position()==prefix && final->position()==prefix+16,0,2);
        // One M16 and two M8 executions consume the identical represented inputs.
        for(bool native:{false,true}) {
            auto ctx=create(native);ctx->restore_exact_host_state(*root);PayloadRows rows;
            const int block=native?16:8;
            ninfer::exl3::Exl3GdnContinuationHistoryView previous_history;
            for(int first=0;first<16;first+=block) {
                ctx->continue_rows(std::span<const std::int64_t>(greedy).subspan(first,block));
                if(previous_history.storage_owner)
                    require(!previous_history.current(),
                            "native continuation reused stale GDN history");
                const auto history=ctx->gdn_continuation_history(0,0,block);
                const auto final_row=ctx->gdn_continuation_history(0,block-1,1);
                require(history.current()&&final_row.current()&&
                            history.base_position==prefix+first &&
                            history.source_rows==block && final_row.rows==1 &&
                            final_row.first_row==block-1,
                        "native continuation GDN history boundary");
                previous_history=history;
                capture(*ctx,block,rows);ctx->finish_exact_continuation();
            }
            const auto state=ctx->export_exact_host_state();
            gate(prefix,native?"M16_payload":"M8x2_payload",16,0,
                row_equal(rows,serial_rows) && state->same_payload(*final),ctx->persistent_bytes(),1);
        }
        ninfer::exl3::Exl3GdnContinuationHistoryView survivor;
        {
            auto ctx=create(true);ctx->restore_exact_host_state(*root);
            ctx->continue_rows(std::span<const std::int64_t>(greedy).first(16));
            survivor=ctx->gdn_continuation_history(0,7,1);
        }
        gate(prefix,"M16_history_survivor",16,8,
            survivor.current()&&survivor.storage_owner.use_count()==1);
        survivor={};
        struct Case { std::string name;int width,reject,terminal; };
        std::vector<Case> cases{{"accept",16,0,0},{"partial9",9,0,0},{"partial15",15,0,0}};
        for(int r=1;r<=16;++r) cases.push_back({"reject"+std::to_string(r),16,r,0});
        for(int t:{8,9,16}) cases.push_back({"terminal"+std::to_string(t),16,0,t});
        cases.push_back({"repeat_reject9",16,9,0});cases.push_back({"repeat_accept",16,0,0});
        for(const auto& test:cases) {
            std::vector<std::int64_t> tentative(greedy.begin(),greedy.begin()+test.width),terminal;
            if(test.reject) tentative[test.reject-1]=(tentative[test.reject-1]+1)%248320;
            int actual_terminal=0;
            std::string case_name=test.name;
            if(test.terminal) {
                terminal.push_back(greedy[test.terminal-1]);
                const int earliest=static_cast<int>(std::find(greedy.begin(),greedy.end(),terminal[0])-greedy.begin())+1;
                actual_terminal=earliest;
                case_name+="_requested"+std::to_string(test.terminal)+"_actual"+std::to_string(earliest);
                if(earliest!=test.terminal) case_name+="_requested_boundary_UNMEASURED";
            }
            std::shared_ptr<const State> expected;
            std::vector<std::int64_t> committed;
            std::array<std::vector<std::uint16_t>,5> expected_taps;
            std::size_t accepted=0;bool rejected=false,stopped=false;
            {
                auto oracle=create(false);oracle->restore_exact_host_state(*root);
                for(int row=0;row<test.width;++row) {
                    const auto token=sample_target(*oracle);
                    if(token==tentative[row]) ++accepted;else rejected=true;
                    committed.push_back(token);oracle->decode(token);
                    const auto t=oracle->exact_tap_rows_host();
                    for(int j=0;j<5;++j) append(expected_taps[j],t[j]);
                    stopped=!terminal.empty() && token==terminal[0];
                    if(rejected || stopped) break;
                }
                expected=oracle->export_exact_host_state();
            }
            auto ctx=create(true);
            const bool repeated=test.name.rfind("repeat_",0)==0;
            if(repeated) {
                const auto previous=ninfer::exl3::verify_exl3_outer_windowed_reference(
                    *ctx,*root,tentative,true,terminal);
                gate(prefix,"repeat_first_payload",test.width,test.reject,
                    previous.committed_state->same_payload(*expected),ctx->persistent_bytes(),1);
            }
            const auto before=ctx->host_kv_stats();
            const auto result=ninfer::exl3::verify_exl3_outer_windowed_reference(
                *ctx,*root,tentative,true,terminal);
            const auto after=ctx->host_kv_stats();
            const bool repair=rejected || committed.size()<tentative.size();
            const std::size_t expected_replay=repair?committed.size():0;
            const std::size_t expected_calls=repair?2:1;
            bool hist=result.native_batch_hist.size()==17;
            for(std::size_t m=0;m<result.native_batch_hist.size();++m) {
                const std::size_t count=(m==tentative.size()?1:0)+(repair && m==committed.size()?1:0);
                hist=hist && result.native_batch_hist[m]==count;
            }
            const bool pass=result.committed_tokens==committed && result.accepted==accepted &&
                result.rejected==rejected && result.stopped==stopped && result.committed_state &&
                result.committed_state->same_payload(*expected) && result.committed_taps==expected_taps &&
                result.verification_rows==tentative.size() && result.replay_rows==expected_replay &&
                result.executed_rows==tentative.size()+expected_replay &&
                result.native_invocations==expected_calls && result.root_restores==(repair?2:1) && hist &&
                frozen_bytes(root)==immutable;
            checks << prefix << ',' << case_name << ',' << test.width << ',' << (test.reject?test.reject:actual_terminal)
                << ',' << pass << ',' << result.verification_rows << ',' << result.replay_rows << ','
                << result.executed_rows << ',' << committed.size() << ',' << result.native_invocations << ','
                << result.root_restores << ",2," << after.h2d_bytes-before.h2d_bytes << ','
                << after.copy_submissions-before.copy_submissions << ',' << ctx->persistent_bytes() << ',';
            for(std::size_t m=0;m<result.native_batch_hist.size();++m) checks << (m?";":"") << m << ':' << result.native_batch_hist[m];
            checks << '\n';checks.flush();require(checks.good(),"native16 outer evidence flush");
            require(pass,"native16 outer zero-bit/state/accounting gate: "+test.name);
        }
        // Real regression: terminal at an exact native chunk boundary must stop
        // the larger logical window, without executing its second native chunk.
        for(int width:{8,16}) {
            int chosen=-1;
            for(int start=0;start+2*width<=static_cast<int>(greedy.size());++start) {
                const auto terminal=greedy[start+width-1];
                if(std::find(greedy.begin()+start,greedy.begin()+start+width-1,terminal)==
                        greedy.begin()+start+width-1) {chosen=start;break;}
            }
            if(chosen<0) {
                checks << prefix << ",real_terminal_chunk_boundary_UNMEASURED," << width
                    << ",0,,0,0,0,0,0,0,0,0,0,0,none\n";
                checks.flush();require(checks.good(),"native16 unmeasured terminal evidence");
                continue;
            }
            std::shared_ptr<const State> boundary_root,expected;
            std::array<std::vector<std::uint16_t>,5> expected_taps;
            {
                auto serial=create(false);serial->restore_exact_host_state(*root);
                for(int i=0;i<chosen;++i) serial->decode(greedy[i]);
                boundary_root=serial->export_exact_host_state();
                for(int i=0;i<width;++i) {
                    serial->decode(greedy[chosen+i]);const auto t=serial->exact_tap_rows_host();
                    for(int layer=0;layer<5;++layer) append(expected_taps[layer],t[layer]);
                }
                expected=serial->export_exact_host_state();
            }
            const auto boundary_bytes=frozen_bytes(boundary_root);
            auto ctx=create(width==16);const auto before=ctx->host_kv_stats();
            const auto tentative=std::span<const std::int64_t>(greedy).subspan(chosen,2*width);
            const std::array<std::int64_t,1> terminal{greedy[chosen+width-1]};
            const auto result=ninfer::exl3::verify_exl3_outer_windowed_reference(
                *ctx,*boundary_root,tentative,true,terminal);
            const auto after=ctx->host_kv_stats();
            bool histogram=result.native_batch_hist.size()==17;
            for(std::size_t m=0;m<result.native_batch_hist.size();++m)
                histogram=histogram && result.native_batch_hist[m]==(m==static_cast<std::size_t>(width)?1:0);
            const bool pass=result.stopped && !result.rejected && result.accepted==static_cast<std::size_t>(width) &&
                result.committed_tokens==std::vector<std::int64_t>(greedy.begin()+chosen,greedy.begin()+chosen+width) &&
                result.committed_taps==expected_taps && result.committed_state && result.committed_state->same_payload(*expected) &&
                result.verification_rows==static_cast<std::size_t>(width) && result.executed_rows==static_cast<std::size_t>(width) &&
                result.replay_rows==0 && result.native_invocations==1 && result.root_restores==1 && histogram &&
                frozen_bytes(boundary_root)==boundary_bytes && frozen_bytes(root)==immutable;
            checks << prefix << ",real_terminal_chunk_boundary_start" << chosen << ',' << 2*width << ',' << width << ',' << pass
                << ',' << result.verification_rows << ',' << result.replay_rows << ',' << result.executed_rows << ','
                << result.committed_tokens.size() << ',' << result.native_invocations << ',' << result.root_restores << ",3,"
                << after.h2d_bytes-before.h2d_bytes << ',' << after.copy_submissions-before.copy_submissions << ',' << ctx->persistent_bytes() << ',';
            for(std::size_t m=0;m<result.native_batch_hist.size();++m) checks << (m?";":"") << m << ':' << result.native_batch_hist[m];
            checks << '\n';checks.flush();require(checks.good(),"native16 terminal boundary evidence");
            require(pass,"native16 real terminal at native chunk boundary continued into suffix");
        }
        // Attribute actual projection dispatch once, separately from unprofiled timing.
        for(bool native:{false,true}) {
            _putenv_s("NINFER_EXL3_TARGET_PROJECTION_TIMING","1");auto ctx=create(native);
            _putenv_s("NINFER_EXL3_TARGET_PROJECTION_TIMING","0");
            ctx->restore_exact_host_state(*root);ctx->prepare_target_projection_timing();
            cuda_check(cudaDeviceSynchronize(),"native16 projection profile start");
            ctx->begin_target_projection_timing_round(0);
            const int block=native?16:8;
            for(int first=0;first<16;first+=block) {
                ctx->continue_rows(std::span<const std::int64_t>(greedy).subspan(first,block));ctx->finish_exact_continuation();
            }
            cuda_check(cudaDeviceSynchronize(),"native16 projection profile end");
            const auto records=ctx->finish_target_projection_timing_round_after_synchronize();
            for(const auto& record:records)
                projections << prefix << ',' << (native?"M16":"M8x2") << ',' << record.layer << ','
                    << target_projection_operator_name(record.operation) << ',' << record.rows << ',' << record.K << ','
                    << record.in_features << ',' << record.out_features << ',' << target_projection_topology_name(record.topology)
                    << ',' << record.calls << ',' << record.microseconds << '\n';
            projections.flush();require(projections.good(),"native16 projection evidence flush");
            gate(prefix,"profile_payload",16,0,ctx->export_exact_host_state()->same_payload(*final),ctx->persistent_bytes(),1);
        }
        // Frozen zero-bit qualification has passed before any timing at this prefix.
        // Context construction is outside the interval; restore+compute+one export is inside.
        for(int pair=0;pair<6;++pair) for(int order=0;order<2;++order) {
            const bool native=pair%2?order==0:order==1;auto ctx=create(native);
            const auto before=ctx->host_kv_stats();
            cuda_check(cudaDeviceSynchronize(),"native16 timing start");const auto a=Clock::now();
            ctx->restore_exact_host_state(*root);cuda_check(cudaDeviceSynchronize(),"native16 timing restore");const auto b=Clock::now();
            const int block=native?16:8;
            for(int first=0;first<16;first+=block) {
                ctx->continue_rows(std::span<const std::int64_t>(greedy).subspan(first,block));ctx->finish_exact_continuation();
            }
            cuda_check(cudaDeviceSynchronize(),"native16 timing compute");const auto c=Clock::now();
            auto state=ctx->export_exact_host_state();cuda_check(cudaDeviceSynchronize(),"native16 timing export");const auto d=Clock::now();
            const auto ms=[](auto start,auto end) {return std::chrono::duration<double,std::milli>(end-start).count();};
            const auto after=ctx->host_kv_stats();const bool exact=state->same_payload(*final) && frozen_bytes(root)==immutable;
            pairs << prefix << ',' << pair << ',' << (pair%2?"BA":"AB") << ',' << (native?"M16":"M8x2")
                << ',' << ms(a,b) << ',' << ms(b,c) << ',' << ms(c,d) << ',' << ms(a,d) << ",1,"
                << after.h2d_bytes-before.h2d_bytes << ',' << after.copy_submissions-before.copy_submissions
                << ',' << ctx->persistent_bytes() << ',' << exact << '\n';
            pairs.flush();require(pairs.good(),"native16 timing evidence flush");require(exact,"native16 timed exact gate");
        }
    }
    std::cout << "NATIVE16 PASS prefixes=16,4096 rejection_positions=1..16 terminal_positions=8,9,16 partial_widths=9,15 pairs_per_prefix=6 target_L2_only=1\n";
}
