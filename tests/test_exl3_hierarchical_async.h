#pragma once

void run_hierarchical_async_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using ninfer::exl3::Exl3HierarchyFrontiers;
    using ninfer::exl3::Exl3HierarchyRunAhead;
    using ninfer::exl3::Exl3OuterReferenceResult;
    using ninfer::exl3::Exl3TextContext;
    using ninfer::exl3::Exl3TurboAngleL1Context;
    constexpr int block=4;
    const bool t2=target.max_context()==4352;
    const int prefix=t2?4096:96,budget=t2?32:16;
    require(code.size()>=static_cast<std::size_t>(prefix) &&
        prose.size()>=static_cast<std::size_t>(prefix) &&
        (target.max_context()==256 || t2),
        "hierarchical async fixture/context extent");
    run_hierarchical_frontier_logic();
    if(!t2) {
        _putenv_s("NINFER_EXL3_WIDE_PREFILL","0");
        _putenv_s("NINFER_EXL3_PREFILL_STAGED_REDUCTION","0");
        _putenv_s("NINFER_EXL3_SHARED_ACCUM","0");
        _putenv_s("NINFER_EXL3_SHARED_TRANSFORM","0");
        _putenv_s("NINFER_EXL3_SHARED_LAYER_SCRATCH","0");
        _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB","0");
    }
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    _putenv_s("NINFER_EXL3_WIDE_PREFILL","0");
    _putenv_s("NINFER_EXL3_PREFILL_STAGED_REDUCTION","0");
    _putenv_s("NINFER_EXL3_SHARED_ACCUM","0");
    _putenv_s("NINFER_EXL3_SHARED_TRANSFORM","0");
    _putenv_s("NINFER_EXL3_SHARED_LAYER_SCRATCH","0");
    _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB","0");
    auto l0=target.create_context(true),l1_native=target.create_context(true);
    require(l0->try_enable_oscar_from_environment() && l1_native->try_enable_oscar_from_environment(),
        "hierarchical async OSCAR contexts");

    struct OwnedStream {
        cudaStream_t value=nullptr;
        explicit OwnedStream(const char* label) {
            cuda_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),label);
        }
        ~OwnedStream() {if(value) cudaStreamDestroy(value);}
        OwnedStream(const OwnedStream&)=delete;
        OwnedStream& operator=(const OwnedStream&)=delete;
    } exact_stream("hierarchical async exact stream"),l0_stream("hierarchical async L0 stream");

    const auto greedy=[](Exl3TextContext& context,cudaStream_t stream=nullptr) {
        const auto logits=context.logits_host(stream);
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"hierarchical async finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    struct Metrics {
        std::vector<std::int64_t> tokens;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
        std::size_t l0_rows=0,l1_rows=0,l2_rows=0,l2_replay=0,l2_rejections=0,events=0;
        std::size_t submissions=0,adopted=0,discarded=0,max_in_flight=0;
        double wall_ms=0,setup_construct_ms=0,setup_restore_ms=0,l2_ms=0;
        double run_ahead_ms=0,overlap_span_ms=0,overlap_saved_ms=0;
        std::vector<double> release_us;
        std::vector<int> bursts;
    };
    struct Buffered {
        Exl3HierarchyRunAhead identity;
        std::vector<std::int64_t> dependency_tokens;
        std::vector<std::int64_t> tokens;
        bool valid=false;
    };
    struct TimedL2 {Exl3OuterReferenceResult result;double wall_ms=0;};

    std::unique_ptr<Exl3TurboAngleL1Context> l1;
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const auto& source=*fixture.second;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> root;
        if(t2) {
            const std::vector<std::int64_t> input(source.begin(),source.begin()+prefix);
            root=ninfer::exl3::Exl3VeriCacheRequest::initialize(*exact,input,1024)->state();
        } else {
            exact->reset();exact->prefill(std::span<const std::int64_t>(source.data(),16));
            for(int row=16;row<prefix;++row) exact->decode(source[row]);
            root=exact->export_exact_host_state();
        }
        std::vector<std::int64_t> reference;reference.reserve(budget);
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> reference8,reference_final;
        exact->restore_exact_host_state(*root);
        for(int row=0;row<budget;++row) {
            const auto token=greedy(*exact);exact->decode(token);reference.push_back(token);
            if(row==7) reference8=exact->export_exact_host_state();
        }
        reference_final=exact->export_exact_host_state();
        if(!l1) l1=std::make_unique<Exl3TurboAngleL1Context>(std::move(l1_native),root);
        else l1->rebase(root);

        const auto run=[&](bool asynchronous,int output_budget,bool force_l2) {
            Metrics metrics;metrics.tokens.reserve(output_budget);
            auto current=root;
            l0->restore_oscar_host_state(*current,nullptr,l0_stream.value);
            const auto setup_before=l1->stats();
            l1->rebase(current);
            const auto setup_after=l1->stats();
            metrics.setup_construct_ms=setup_after.construction_ms-setup_before.construction_ms;
            metrics.setup_restore_ms=setup_after.restore_ms-setup_before.restore_ms;
            Exl3HierarchyFrontiers frontiers(asynchronous?4:3,prefix);
            Buffered buffered;
            int round=0;
            const auto started=std::chrono::steady_clock::now();
            while(metrics.tokens.size()<static_cast<std::size_t>(output_budget)) {
                const int rows=std::min<int>(block,output_budget-static_cast<int>(metrics.tokens.size()));
                std::vector<std::int64_t> candidates;
                ninfer::exl3::Exl3HierarchyCompletion l0_completion;
                if(buffered.valid) {
                    require(static_cast<int>(buffered.tokens.size())==rows &&
                        buffered.identity.first==prefix+static_cast<int>(metrics.tokens.size()),
                        "hierarchical async buffered adoption extent");
                    l0_completion=frontiers.begin_l0(rows);
                    require(l0_completion.first==buffered.identity.first,
                        "hierarchical async adopted frontier identity");
                    candidates=std::move(buffered.tokens);buffered.valid=false;++metrics.adopted;
                } else {
                    l0_completion=frontiers.begin_l0(rows);
                    candidates.reserve(rows);
                    for(int row=0;row<rows;++row) {
                        const auto token=greedy(*l0,l0_stream.value);
                        candidates.push_back(token);l0->decode(token,l0_stream.value);++metrics.l0_rows;
                    }
                }
                require(frontiers.current_completion(l0_completion),
                    "hierarchical async current L0 completion");
                const auto screened_decision=rows==1?l1->screen_one(candidates[0]):l1->screen(candidates);
                metrics.l1_rows+=screened_decision.executed_rows;
                auto screened=screened_decision.authorized_tokens;
                if(screened_decision.rejected) {
                    l0->restore_oscar_host_state(*current,nullptr,l0_stream.value);
                    for(auto token:screened) l0->decode(token,l0_stream.value);
                }
                frontiers.complete_l1(l0_completion,static_cast<int>(screened.size()),screened_decision.rejected);
                if(force_l2 && round==0) {
                    screened.back()=(screened.back()+1)%248320;
                    l0->restore_oscar_host_state(*current,nullptr,l0_stream.value);
                    for(auto token:screened) l0->decode(token,l0_stream.value);
                }
                const auto l2_completion=frontiers.begin_l2();

                TimedL2 verified;
                Buffered produced;
                if(asynchronous) {
                    const int predicted_remaining=output_budget-
                        static_cast<int>(metrics.tokens.size())-static_cast<int>(screened.size());
                    const int next_rows=std::min(block,std::max(0,predicted_remaining));
                    if(next_rows>0) {
                        produced.identity=frontiers.begin_run_ahead(l2_completion,next_rows);
                        produced.dependency_tokens=screened;
                    }
                    std::promise<void> launched_promise;
                    auto launched=launched_promise.get_future();
                    auto root_owner=current;
                    auto owned_candidates=screened;
                    double ahead_ms=0;
                    const auto overlap_start=std::chrono::steady_clock::now();
                    auto pending=std::async(std::launch::async,
                        [&,root_owner=std::move(root_owner),owned_candidates=std::move(owned_candidates),
                         signal=std::move(launched_promise)]() mutable {
                            signal.set_value();
                            const auto begin=std::chrono::steady_clock::now();
                            TimedL2 timed;
                            timed.result=owned_candidates.size()==1
                                ? ninfer::exl3::verify_exl3_outer_reference(
                                    *exact,*root_owner,owned_candidates,false,{},exact_stream.value,true)
                                : ninfer::exl3::verify_exl3_outer_batched_reference(
                                    *exact,*root_owner,owned_candidates,false,{},exact_stream.value,true);
                            timed.wall_ms=std::chrono::duration<double,std::milli>(
                                std::chrono::steady_clock::now()-begin).count();
                            return timed;
                        });
                    launched.wait();++metrics.submissions;metrics.max_in_flight=1;
                    if(next_rows>0) {
                        const auto ahead_start=std::chrono::steady_clock::now();
                        produced.tokens.reserve(next_rows);
                        for(int row=0;row<next_rows;++row) {
                            const auto token=greedy(*l0,l0_stream.value);
                            produced.tokens.push_back(token);l0->decode(token,l0_stream.value);++metrics.l0_rows;
                        }
                        ahead_ms=std::chrono::duration<double,std::milli>(
                            std::chrono::steady_clock::now()-ahead_start).count();
                        metrics.run_ahead_ms+=ahead_ms;
                    }
                    verified=pending.get();
                    const double span=std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-overlap_start).count();
                    metrics.overlap_span_ms+=span;metrics.l2_ms+=verified.wall_ms;
                    metrics.overlap_saved_ms+=std::max(0.0,verified.wall_ms+ahead_ms-span);
                } else {
                    const auto begin=std::chrono::steady_clock::now();
                    verified.result=screened.size()==1
                        ? ninfer::exl3::verify_exl3_outer_reference(*exact,*current,screened,false,{},nullptr,true)
                        : ninfer::exl3::verify_exl3_outer_batched_reference(*exact,*current,screened,false,{},nullptr,true);
                    verified.wall_ms=std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-begin).count();
                    metrics.l2_ms+=verified.wall_ms;
                }
                auto& result=verified.result;
                metrics.l2_rows+=result.verification_rows;metrics.l2_replay+=result.replay_rows;
                metrics.l2_rejections+=result.rejected;
                const bool adopt=asynchronous && !produced.tokens.empty() &&
                    frontiers.can_adopt_run_ahead(produced.identity,static_cast<int>(result.committed_tokens.size())) &&
                    produced.dependency_tokens==result.committed_tokens;
                frontiers.complete_l2(l2_completion,static_cast<int>(result.committed_tokens.size()));
                const auto publication=frontiers.publish_authorized();
                require(publication.first==prefix+static_cast<int>(metrics.tokens.size()) &&
                    publication.rows==static_cast<int>(result.committed_tokens.size()),
                    "hierarchical async rolling L2 publication");
                metrics.tokens.insert(metrics.tokens.end(),result.committed_tokens.begin(),result.committed_tokens.end());
                metrics.release_us.push_back(std::chrono::duration<double,std::micro>(
                    std::chrono::steady_clock::now()-started).count());
                metrics.bursts.push_back(static_cast<int>(result.committed_tokens.size()));
                ++metrics.events;current=result.committed_state;frontiers.rebase();
                l1->rebase_delta(current);
                if(adopt) {produced.valid=true;buffered=std::move(produced);}
                else {
                    if(!produced.tokens.empty()) ++metrics.discarded;
                    l0->restore_oscar_host_state(*current,nullptr,l0_stream.value);
                }
                ++round;
            }
            cuda_check(cudaStreamSynchronize(exact_stream.value),"hierarchical async exact completion");
            cuda_check(cudaStreamSynchronize(l0_stream.value),"hierarchical async L0 completion");
            metrics.wall_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            metrics.state=current;
            require(!buffered.valid && frontiers.published()==prefix+output_budget &&
                frontiers.published()==frontiers.authoritative() &&
                frontiers.authoritative()==frontiers.screened() &&
                frontiers.screened()==frontiers.drafted(),"hierarchical async settled P/A/S/D");
            return metrics;
        };

        const auto run_w=[&](int output_budget) {
            Metrics metrics;metrics.tokens.reserve(output_budget);
            auto current=root;
            l0->restore_oscar_host_state(*current,nullptr,l0_stream.value);
            Exl3HierarchyFrontiers frontiers(5,prefix);
            const auto started=std::chrono::steady_clock::now();
            while(metrics.tokens.size()<static_cast<std::size_t>(output_budget)) {
                const int rows=std::min<int>(block,output_budget-static_cast<int>(metrics.tokens.size()));
                const auto l0_completion=frontiers.begin_l0(rows);
                std::vector<std::int64_t> candidates;candidates.reserve(rows);
                for(int row=0;row<rows;++row) {
                    const auto token=greedy(*l0,l0_stream.value);
                    candidates.push_back(token);l0->decode(token,l0_stream.value);++metrics.l0_rows;
                }
                frontiers.complete_l1(l0_completion,rows,false);
                const auto l2_completion=frontiers.begin_l2();
                const auto begin=std::chrono::steady_clock::now();
                auto result=rows==1
                    ? ninfer::exl3::verify_exl3_outer_reference(*exact,*current,candidates,false,{},nullptr,true)
                    : ninfer::exl3::verify_exl3_outer_batched_reference(*exact,*current,candidates,false,{},nullptr,true);
                metrics.l2_ms+=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-begin).count();
                metrics.l2_rows+=result.verification_rows;metrics.l2_replay+=result.replay_rows;
                metrics.l2_rejections+=result.rejected;
                frontiers.complete_l2(l2_completion,static_cast<int>(result.committed_tokens.size()));
                const auto publication=frontiers.publish_authorized();
                require(publication.first==prefix+static_cast<int>(metrics.tokens.size()),
                    "hierarchical async W publication");
                metrics.tokens.insert(metrics.tokens.end(),result.committed_tokens.begin(),result.committed_tokens.end());
                metrics.release_us.push_back(std::chrono::duration<double,std::micro>(
                    std::chrono::steady_clock::now()-started).count());
                metrics.bursts.push_back(static_cast<int>(result.committed_tokens.size()));
                ++metrics.events;current=result.committed_state;frontiers.rebase();
                l0->restore_oscar_host_state(*current,nullptr,l0_stream.value);
            }
            cuda_check(cudaStreamSynchronize(l0_stream.value),"hierarchical async W completion");
            metrics.wall_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            metrics.state=current;
            return metrics;
        };

        const auto w=run_w(budget);
        const auto sync=run(false,budget,false);
        const auto asynchronous=run(true,budget,false);
        Metrics repair;
        if(!t2) repair=run(true,8,true);
        require(w.tokens==reference && sync.tokens==reference && asynchronous.tokens==reference &&
            w.state->same_payload(*reference_final) && sync.state->same_payload(*reference_final) &&
            asynchronous.state->same_payload(*reference_final),
            "hierarchical async W/C_sync/C_async output/state");
        if(!t2) require(std::equal(repair.tokens.begin(),repair.tokens.end(),reference.begin()) &&
                repair.state->same_payload(*reference8) && repair.l2_rejections>=1 && repair.discarded>=1,
                "hierarchical async rejection invalidation/repair");
        const auto print=[&](const char* mode,const Metrics& m) {
            std::cout << "HIERARCHICAL_ASYNC fixture=" << fixture.first << " mode=" << mode
                << " prefix=" << prefix
                << " tokens=" << m.tokens.size() << " wall_ms=" << m.wall_ms
                << " published_tps=" << m.tokens.size()*1000/m.wall_ms
                << " setup_construct_ms=" << m.setup_construct_ms
                << " setup_restore_ms=" << m.setup_restore_ms
                << " events=" << m.events << " l0_rows=" << m.l0_rows << " l1_rows=" << m.l1_rows
                << " l2_rows=" << m.l2_rows << " l2_replay=" << m.l2_replay
                << " l2_rejections=" << m.l2_rejections << " submissions=" << m.submissions
                << " adopted=" << m.adopted << " discarded=" << m.discarded
                << " max_in_flight=" << m.max_in_flight << " l2_ms=" << m.l2_ms
                << " run_ahead_ms=" << m.run_ahead_ms << " overlap_span_ms=" << m.overlap_span_ms
                << " overlap_saved_ms=" << m.overlap_saved_ms << " release_events=" << m.release_us.size()
                << " release_us=";
            for(std::size_t i=0;i<m.release_us.size();++i) std::cout << (i?",":"") << m.release_us[i];
            std::cout << " bursts=";
            for(std::size_t i=0;i<m.bursts.size();++i) std::cout << (i?",":"") << m.bursts[i];
            std::cout << '\n';
        };
        print("W",w);print("C_sync",sync);print("C_async",asynchronous);
        if(!t2) print("C_async_l2_fault",repair);
    }
    std::cout << "HIERARCHICAL_ASYNC PASS fixtures=2 prefix=" << prefix
        << " block=4 budget=" << budget << " max_in_flight=1"
        << " owned_root_buffer=1 stale_cancel=1 rejection_discard=1 exact_state=1\n";
}
