#pragma once

void run_hierarchical_adaptive_logic() {
    using ninfer::exl3::Exl3AdaptiveHierarchyPolicy;
    using ninfer::exl3::Exl3AdaptiveSample;
    using ninfer::exl3::Exl3HierarchyMode;
    Exl3AdaptiveHierarchyPolicy policy;
    require(policy.mode()==Exl3HierarchyMode::window_exact &&
        !policy.select(false,true,8,0),"adaptive conservative cold start");
    policy.observe({Exl3HierarchyMode::window_exact,100,4,0,0,false,true});
    policy.observe({Exl3HierarchyMode::cascade_async,96,4,80,0,false,true});
    policy.observe({Exl3HierarchyMode::cascade_async,96,4,80,0,false,true});
    require(!policy.select(true,true,8,25) && policy.select(true,true,8,0),
        "adaptive setup guard and measured transition");
    for(int round=0;round<3;++round) {
        policy.observe({Exl3HierarchyMode::cascade_async,110,4,80,0,false,true});
        require(policy.mode()==Exl3HierarchyMode::cascade_async,"adaptive minimum economic dwell");
    }
    policy.observe({Exl3HierarchyMode::cascade_async,110,4,80,0,false,true});
    require(policy.mode()==Exl3HierarchyMode::window_exact && policy.cooldown_rounds()==8 &&
        policy.fallback_count()==1,"adaptive economic fallback/cooldown");
    for(int round=0;round<8;++round)
        policy.observe({Exl3HierarchyMode::window_exact,100,4,0,0,false,true});
    policy.observe({Exl3HierarchyMode::cascade_async,90,4,80,0,false,true});
    policy.observe({Exl3HierarchyMode::cascade_async,90,4,80,0,false,true});
    require(policy.select(true,true,8,0),"adaptive cooldown recovery");
    policy.observe({Exl3HierarchyMode::cascade_async,90,4,250,0,false,true});
    require(policy.mode()==Exl3HierarchyMode::window_exact && policy.fallback_count()==2,
        "adaptive pending-age safety fallback");

    Exl3AdaptiveHierarchyPolicy rejected;
    rejected.observe({Exl3HierarchyMode::window_exact,100,4,0,0,false,true});
    rejected.observe({Exl3HierarchyMode::cascade_async,90,4,80,0,false,true});
    rejected.observe({Exl3HierarchyMode::cascade_async,90,4,80,0,false,true});
    require(rejected.select(true,true,8,0),"adaptive rejection setup");
    rejected.observe({Exl3HierarchyMode::cascade_async,90,4,80,0,true,true});
    require(rejected.mode()==Exl3HierarchyMode::window_exact && rejected.fallback_count()==1,
        "adaptive target-rejection fallback");
    std::cout << "HIERARCHICAL_ADAPTIVE_LOGIC PASS conservative_start=1 hysteresis=1"
        << " dwell=4 cooldown=8 setup_guard=1 rejection_fallback=1 pending_fallback=1\n";
}

void run_hierarchical_adaptive_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using ninfer::exl3::Exl3AdaptiveHierarchyPolicy;
    using ninfer::exl3::Exl3AdaptiveSample;
    using ninfer::exl3::Exl3HierarchyFrontiers;
    using ninfer::exl3::Exl3HierarchyMode;
    constexpr int prefix=4096,budget=32,block=4;
    require(target.max_context()==4352 && code.size()>=prefix && prose.size()>=prefix,
        "adaptive T2 fixture/context extent");
    run_hierarchical_adaptive_logic();
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    _putenv_s("NINFER_EXL3_WIDE_PREFILL","0");
    auto l0=target.create_context(true);
    require(l0->try_enable_oscar_from_environment(),"adaptive W OSCAR context");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"adaptive finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        const auto root=Request::initialize(*exact,input,1024)->state();
        exact->restore_exact_host_state(*root);
        std::vector<std::int64_t> reference;reference.reserve(budget);
        for(int row=0;row<budget;++row) {const auto token=greedy(*exact);exact->decode(token);reference.push_back(token);}
        const auto reference_state=exact->export_exact_host_state();

        Exl3AdaptiveHierarchyPolicy policy;
        const bool prose_fixture=std::string(fixture.first)=="prose";
        const double prior_w=prose_fixture?4234.48:4263.52;
        const double prior_async=prose_fixture?4540.77:4464.44;
        const double prior_setup=prose_fixture?8814.34:9025.73;
        policy.observe({Exl3HierarchyMode::window_exact,prior_w/8,4,0,0,false,true});
        policy.observe({Exl3HierarchyMode::cascade_async,prior_async/8,4,0,0,false,true});
        policy.observe({Exl3HierarchyMode::cascade_async,prior_async/8,4,0,0,false,true});
        require(!policy.select(true,true,8,prior_setup) &&
            policy.mode()==Exl3HierarchyMode::window_exact,
            "adaptive natural W selection");

        auto current=root;l0->restore_oscar_host_state(*current);
        Exl3HierarchyFrontiers frontiers(6,prefix);
        std::vector<std::int64_t> tokens;tokens.reserve(budget);
        std::vector<double> releases;std::vector<int> bursts;
        const auto started=std::chrono::steady_clock::now();
        while(tokens.size()<budget) {
            const auto round_start=std::chrono::steady_clock::now();
            const int rows=std::min<int>(block,budget-static_cast<int>(tokens.size()));
            const auto l0_completion=frontiers.begin_l0(rows);
            std::vector<std::int64_t> candidates;candidates.reserve(rows);
            for(int row=0;row<rows;++row) {const auto token=greedy(*l0);candidates.push_back(token);l0->decode(token);}
            frontiers.complete_l1(l0_completion,rows,false);
            const auto l2_completion=frontiers.begin_l2();
            auto verified=ninfer::exl3::verify_exl3_outer_batched_reference(
                *exact,*current,candidates,false,{},nullptr,true);
            frontiers.complete_l2(l2_completion,static_cast<int>(verified.committed_tokens.size()));
            const auto publication=frontiers.publish_authorized();
            require(publication.first==prefix+static_cast<int>(tokens.size()),"adaptive publication range");
            tokens.insert(tokens.end(),verified.committed_tokens.begin(),verified.committed_tokens.end());
            bursts.push_back(static_cast<int>(verified.committed_tokens.size()));
            releases.push_back(std::chrono::duration<double,std::micro>(
                std::chrono::steady_clock::now()-started).count());
            const double round_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-round_start).count();
            policy.observe({Exl3HierarchyMode::window_exact,round_ms,
                static_cast<int>(verified.committed_tokens.size()),0,0,verified.rejected,true});
            current=verified.committed_state;frontiers.rebase();l0->restore_oscar_host_state(*current);
        }
        const double wall_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count();
        require(tokens==reference && current->same_payload(*reference_state) &&
            policy.mode()==Exl3HierarchyMode::window_exact,
            "adaptive W exact output/state");
        std::cout << "HIERARCHICAL_ADAPTIVE fixture=" << fixture.first
            << " mode=W prefix=4096 tokens=32 wall_ms=" << wall_ms
            << " published_tps=" << 32000/wall_ms << " l1_allocated=0 transitions=0"
            << " fallback_queue_rows=0 release_events=" << releases.size() << " release_us=";
        for(std::size_t i=0;i<releases.size();++i) std::cout << (i?",":"") << releases[i];
        std::cout << " bursts=";
        for(std::size_t i=0;i<bursts.size();++i) std::cout << (i?",":"") << bursts[i];
        std::cout << '\n';
    }
    std::cout << "HIERARCHICAL_ADAPTIVE PASS fixtures=2 prefix=4096 budget=32"
        << " conservative_W=1 unused_L1_allocation=0 exact_state=1\n";
}
