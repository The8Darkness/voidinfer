#pragma once

void run_exact_resident_root_t3(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose,
    const std::vector<std::int64_t>& structured,const std::vector<std::int64_t>& heldout) {
    constexpr int prefix=4096,budget=128,block=4;
    require(target.max_context()==4352,"resident-root T3 context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto source=target.create_context(true),forced=target.create_context(true),resident=target.create_context(true);
    source->prepare_continuation(8);forced->prepare_continuation(8);resident->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"resident-root T3 finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const auto median=[](std::vector<double> values) {
        std::sort(values.begin(),values.end());return values[values.size()/2];
    };
    const std::array<std::pair<const char*,const std::vector<std::int64_t>*>,4> fixtures={
        std::pair{"code",&code},std::pair{"prose",&prose},
        std::pair{"structured",&structured},std::pair{"heldout_mixed",&heldout}};
    for(const auto& fixture:fixtures) {
        require(fixture.second->size()>=prefix,"resident-root T3 fixture extent");
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        const auto root=ninfer::exl3::Exl3VeriCacheRequest::initialize(*source,input,1024)->state();
        source->restore_exact_host_state(*root);
        std::vector<std::int64_t> reference;reference.reserve(budget);
        for(int row=0;row<budget;++row) {const auto token=greedy(*source);source->decode(token);reference.push_back(token);}
        const auto reference_state=source->export_exact_host_state();
        const auto run=[&](Exl3TextContext& context,bool reuse) {
            std::vector<std::int64_t> tokens;tokens.reserve(budget);
            std::size_t restores=0;
            auto current=root;context.restore_exact_host_state(*current);
            const auto started=std::chrono::steady_clock::now();
            while(tokens.size()<budget) {
                std::array<std::int64_t,block> tentative{};
                std::copy_n(reference.begin()+tokens.size(),block,tentative.begin());
                const auto verified=ninfer::exl3::verify_exl3_outer_batched_reference(
                    context,*current,tentative,false,{},nullptr,reuse);
                restores+=verified.root_restores;
                tokens.insert(tokens.end(),verified.committed_tokens.begin(),verified.committed_tokens.end());
                current=verified.committed_state;
            }
            const double wall_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            require(tokens==reference && current->same_payload(*reference_state),
                "resident-root T3 exact output/state");
            require(restores==(reuse?0u:static_cast<std::size_t>(budget/block)),
                "resident-root T3 restore accounting");
            return std::pair{wall_ms,restores};
        };
        std::vector<double> control,candidate;
        for(int rep=0;rep<3;++rep) {
            const std::array<bool,2> order=rep==1?std::array{true,false}:std::array{false,true};
            for(int ordinal=0;ordinal<2;++ordinal) {
                const bool reuse=order[ordinal];
                const auto result=run(reuse?*resident:*forced,reuse);
                (reuse?candidate:control).push_back(result.first);
                std::cout << "EXACT_RESIDENT_ROOT_CASE fixture=" << fixture.first
                    << " rep=" << rep << " order=" << ordinal
                    << " arm=" << (reuse?"resident":"forced")
                    << " wall_ms=" << result.first << " restores=" << result.second
                    << " tps=" << budget*1000/result.first << std::endl;
            }
        }
        const double control_median=median(control),candidate_median=median(candidate);
        require(candidate_median<control_median,"resident-root T3 median must improve");
        std::cout << "EXACT_RESIDENT_ROOT_T3 fixture=" << fixture.first
            << " forced_median_ms=" << control_median
            << " resident_median_ms=" << candidate_median
            << " wall_reduction_percent=" << (control_median-candidate_median)*100/control_median
            << " resident_tps=" << budget*1000/candidate_median << std::endl;
    }
    std::cout << "EXACT_RESIDENT_ROOT_T3 PASS fixtures=4 pairs=3 prefix=4096 budget=128"
        << " exact_state=1 all_medians_positive=1" << std::endl;
}
