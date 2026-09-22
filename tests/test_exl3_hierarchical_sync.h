#pragma once

void run_hierarchical_frontier_logic() {
    using ninfer::exl3::Exl3HierarchyFrontiers;
    Exl3HierarchyFrontiers frontiers(7,10);
    const auto l0=frontiers.begin_l0(4);
    frontiers.complete_l1(l0,4,false);
    const auto l2=frontiers.begin_l2();
    frontiers.complete_l2(l2,2);
    const auto publication=frontiers.publish_authorized();
    require(publication.first==10 && publication.rows==2 && frontiers.published()==12 &&
        frontiers.authoritative()==12 && frontiers.screened()==12 && frontiers.drafted()==12,
        "hierarchy rolling publication frontier");
    frontiers.rebase();
    bool stale=false;
    try {frontiers.complete_l1(l0,1,true);} catch(const std::logic_error&) {stale=true;}
    require(stale && !frontiers.current_completion(l0),"hierarchy stale epoch completion");
    bool unpublished=false;
    try {(void)frontiers.publish_authorized();} catch(const std::logic_error&) {unpublished=true;}
    require(unpublished,"hierarchy unauthorized publication");
    Exl3HierarchyFrontiers ahead(8,20);
    const auto ahead_l0=ahead.begin_l0(4);ahead.complete_l1(ahead_l0,4,false);
    const auto ahead_l2=ahead.begin_l2();
    const auto buffered=ahead.begin_run_ahead(ahead_l2,4);
    require(ahead.can_adopt_run_ahead(buffered,4) &&
        !ahead.can_adopt_run_ahead(buffered,3),"hierarchy run-ahead dependency gate");
    ahead.complete_l2(ahead_l2,4);(void)ahead.publish_authorized();ahead.rebase();
    require(!ahead.can_adopt_run_ahead(buffered,4),"hierarchy stale run-ahead rejection");
    const auto adopted=ahead.begin_l0(buffered.rows);
    require(adopted.first==buffered.first,"hierarchy run-ahead adoption position");

    Exl3HierarchyFrontiers wide(10,40);
    for(int block=0;block<4;++block) {
        const auto completion=wide.begin_l0(8);
        require(completion.first==40+block*8,
            "hierarchy multi-round L0 absolute position");
        wide.complete_l1(completion,8,false);
    }
    const auto wide_l2=wide.begin_l2();
    require(wide_l2.first==40 && wide_l2.rows==32,
        "hierarchy 32-row L2 dependency range");
    wide.complete_l2(wide_l2,32);
    const auto wide_publication=wide.publish_authorized();
    require(wide_publication.first==40 && wide_publication.rows==32 &&
        wide.published()==72,"hierarchy 32-row publication range");

    Exl3HierarchyFrontiers cancelled(9,30);
    const auto cancel_l0=cancelled.begin_l0(4);cancelled.complete_l1(cancel_l0,4,false);
    const auto cancel_l2=cancelled.begin_l2();
    const auto cancelled_work=cancelled.begin_run_ahead(cancel_l2,4);
    cancelled.cancel_tentative();
    require(!cancelled.current_completion(cancel_l2) &&
        !cancelled.can_adopt_run_ahead(cancelled_work,4) &&
        cancelled.published()==30 && cancelled.authoritative()==30 &&
        cancelled.screened()==30 && cancelled.drafted()==30,
        "hierarchy cancellation invalidates tentative work");
    std::cout << "HIERARCHICAL_FRONTIERS PASS stale_epoch=1 unauthorized_publication=1"
        << " run_ahead_dependency=1 cancellation=1\n";
}

void run_hierarchical_sync_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using ninfer::exl3::Exl3HierarchyFrontiers;
    using ninfer::exl3::Exl3OuterReferenceResult;
    using ninfer::exl3::Exl3TextContext;
    using ninfer::exl3::Exl3TurboAngleL1Context;
    constexpr int prefix=96,budget=16,block=4;
    require(code.size()>=prefix && prose.size()>=prefix && target.max_context()==256,
        "hierarchical sync fixture/context extent");
    run_hierarchical_frontier_logic();
    _putenv_s("NINFER_EXL3_WIDE_PREFILL","0");
    _putenv_s("NINFER_EXL3_PREFILL_STAGED_REDUCTION","0");
    _putenv_s("NINFER_EXL3_SHARED_ACCUM","0");
    _putenv_s("NINFER_EXL3_SHARED_TRANSFORM","0");
    _putenv_s("NINFER_EXL3_SHARED_LAYER_SCRATCH","0");
    _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    auto l0=target.create_context(true),l1_native=target.create_context(true);
    require(l0->try_enable_oscar_from_environment() && l1_native->try_enable_oscar_from_environment(),
        "hierarchical sync OSCAR contexts");
    auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"hierarchical sync finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    struct Metrics {
        std::vector<std::int64_t> tokens;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
        std::size_t l0_rows=0,l0_discarded=0,l1_rows=0,l1_replay=0,l1_rejections=0;
        std::size_t l2_rows=0,l2_replay=0,l2_calls=0,l2_restores=0,l2_rejections=0,events=0;
        double wall_ms=0;
    };
    std::unique_ptr<Exl3TurboAngleL1Context> l1;
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const auto& source=*fixture.second;
        exact->reset();exact->prefill(std::span<const std::int64_t>(source.data(),16));
        for(int row=16;row<prefix;++row) exact->decode(source[row]);
        const auto root=exact->export_exact_host_state();
        std::vector<std::int64_t> reference;reference.reserve(budget);
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> reference8,reference16;
        exact->restore_exact_host_state(*root);
        for(int row=0;row<budget;++row) {
            const auto token=greedy(*exact);exact->decode(token);reference.push_back(token);
            if(row==7) reference8=exact->export_exact_host_state();
        }
        reference16=exact->export_exact_host_state();
        if(!l1) l1=std::make_unique<Exl3TurboAngleL1Context>(std::move(l1_native),root);
        else l1->rebase(root);
        const auto run=[&](bool use_l1,int output_budget,bool force_l0,bool force_l2) {
            Metrics metrics;metrics.tokens.reserve(output_budget);
            auto current=root;
            l0->restore_oscar_host_state(*current);
            if(use_l1) l1->rebase(current);
            Exl3HierarchyFrontiers frontiers(use_l1?2:1,prefix);
            int round=0;
            const auto started=std::chrono::steady_clock::now();
            while(metrics.tokens.size()<static_cast<std::size_t>(output_budget)) {
                const int rows=std::min<int>(block,output_budget-static_cast<int>(metrics.tokens.size()));
                const auto l0_completion=frontiers.begin_l0(rows);
                std::vector<std::int64_t> candidates;candidates.reserve(rows);
                for(int row=0;row<rows;++row) {
                    auto token=greedy(*l0);
                    if(force_l0 && round==0 && row==1) token=(token+1)%248320;
                    candidates.push_back(token);l0->decode(token);++metrics.l0_rows;
                }
                std::vector<std::int64_t> screened=candidates;
                bool l1_rejected=false;
                if(use_l1) {
                    const auto decision=rows==1?l1->screen_one(candidates[0]):l1->screen(candidates);
                    screened=decision.authorized_tokens;l1_rejected=decision.rejected;
                    metrics.l1_rows+=decision.executed_rows;metrics.l1_replay+=decision.replay_rows;
                    metrics.l1_rejections+=decision.rejected;
                    metrics.l0_discarded+=candidates.size()-decision.accepted;
                    if(decision.rejected) {
                        l0->restore_oscar_host_state(*current);
                        for(auto token:screened) l0->decode(token);
                    }
                }
                frontiers.complete_l1(l0_completion,static_cast<int>(screened.size()),l1_rejected);
                if(force_l2 && round==0) screened.back()=(screened.back()+1)%248320;
                const auto l2_completion=frontiers.begin_l2();
                Exl3OuterReferenceResult verified=screened.size()==1
                    ? ninfer::exl3::verify_exl3_outer_reference(*exact,*current,screened,false,{},nullptr,true)
                    : ninfer::exl3::verify_exl3_outer_batched_reference(*exact,*current,screened,false,{},nullptr,true);
                metrics.l2_rows+=verified.verification_rows;metrics.l2_replay+=verified.replay_rows;
                metrics.l2_calls+=verified.native_invocations;metrics.l2_restores+=verified.root_restores;
                metrics.l2_rejections+=verified.rejected;
                frontiers.complete_l2(l2_completion,static_cast<int>(verified.committed_tokens.size()));
                const auto publication=frontiers.publish_authorized();
                require(publication.first==prefix+static_cast<int>(metrics.tokens.size()) &&
                    publication.rows==static_cast<int>(verified.committed_tokens.size()),
                    "hierarchical rolling L2 publication range");
                metrics.tokens.insert(metrics.tokens.end(),verified.committed_tokens.begin(),verified.committed_tokens.end());
                ++metrics.events;current=verified.committed_state;
                frontiers.rebase();
                l0->restore_oscar_host_state(*current);
                if(use_l1) l1->rebase_delta(current);
                ++round;
            }
            cuda_check(cudaDeviceSynchronize(),"hierarchical sync completion");
            metrics.wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            metrics.state=current;
            require(frontiers.published()==prefix+output_budget &&
                frontiers.published()==frontiers.authoritative() &&
                frontiers.authoritative()==frontiers.screened() &&
                frontiers.screened()==frontiers.drafted(),"hierarchical settled P/A/S/D");
            return metrics;
        };
        const auto w=run(false,budget,false,false);
        const auto sync=run(true,budget,false,false);
        require(w.tokens==reference && sync.tokens==reference && w.state->same_payload(*reference16) &&
            sync.state->same_payload(*reference16),"hierarchical W/C_sync exact output/state");
        const auto l0_fault=run(true,8,true,false);
        const auto l2_fault=run(true,8,false,true);
        require(std::equal(l0_fault.tokens.begin(),l0_fault.tokens.end(),reference.begin()) &&
            std::equal(l2_fault.tokens.begin(),l2_fault.tokens.end(),reference.begin()) &&
            l0_fault.state->same_payload(*reference8) && l2_fault.state->same_payload(*reference8) &&
            l0_fault.l1_rejections>=1 && l2_fault.l2_rejections>=1,
            "hierarchical forced L0/L1 and L1/L2 repair");
        const auto print=[&](const char* mode,const Metrics& m) {
            std::cout << "HIERARCHICAL_SYNC fixture=" << fixture.first << " mode=" << mode
                << " tokens=" << m.tokens.size() << " wall_ms=" << m.wall_ms
                << " published_tps=" << m.tokens.size()*1000/m.wall_ms
                << " events=" << m.events << " l0_rows=" << m.l0_rows
                << " l0_discarded=" << m.l0_discarded << " l1_rows=" << m.l1_rows
                << " l1_replay=" << m.l1_replay << " l1_rejections=" << m.l1_rejections
                << " l2_rows=" << m.l2_rows << " l2_replay=" << m.l2_replay
                << " l2_calls=" << m.l2_calls << " l2_restores=" << m.l2_restores
                << " l2_rejections=" << m.l2_rejections << '\n';
        };
        print("W",w);print("C_sync",sync);print("C_sync_l0_fault",l0_fault);print("C_sync_l2_fault",l2_fault);
    }
    std::cout << "HIERARCHICAL_SYNC PASS fixtures=2 block=4 budget=16 rolling_publication=1"
        << " l0_l1_repair=1 l1_l2_repair=1 stale_epoch=1 exact_state=1\n";
}
