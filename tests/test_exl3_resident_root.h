#pragma once

void run_exact_resident_root_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    constexpr int block=4;
    const bool t2=target.max_context()==4352;
    const int prefix=t2?4096:96,budget=t2?32:16;
    require((target.max_context()==256 || t2) &&
        code.size()>=static_cast<std::size_t>(prefix) &&
        prose.size()>=static_cast<std::size_t>(prefix),
        "resident-root fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto source=target.create_context(true),forced=target.create_context(true),resident=target.create_context(true);
    source->prepare_continuation(8);forced->prepare_continuation(8);resident->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"resident-root finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    std::array<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>,2> roots;
    for(int fixture=0;fixture<2;++fixture) {
        const auto& input=fixture?prose:code;
        if(t2) {
            const std::vector<std::int64_t> prefix_tokens(input.begin(),input.begin()+prefix);
            roots[fixture]=ninfer::exl3::Exl3VeriCacheRequest::initialize(
                *source,prefix_tokens,1024)->state();
        } else {
            source->reset();source->prefill(std::span<const std::int64_t>(input.data(),16));
            for(int row=16;row<prefix;++row) source->decode(input[row]);
            roots[fixture]=source->export_exact_host_state();
        }
    }
    require(!roots[0]->same_payload(*roots[1]),"resident-root fixtures must diverge");

    for(int fixture=0;fixture<2;++fixture) {
        const auto& root=roots[fixture];
        std::vector<std::int64_t> reference;reference.reserve(budget);
        source->restore_exact_host_state(*root);
        for(int row=0;row<budget;++row) {const auto token=greedy(*source);source->decode(token);reference.push_back(token);}
        const auto reference_state=source->export_exact_host_state();
        const auto run=[&](Exl3TextContext& context,bool reuse) {
            std::vector<std::int64_t> tokens;tokens.reserve(budget);
            std::size_t restores=0;
            auto current=root;
            context.restore_exact_host_state(*current);
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
                "resident-root exact output/state");
            return std::pair{wall_ms,restores};
        };
        const auto control=run(*forced,false);
        const auto candidate=run(*resident,true);
        require(control.second==static_cast<std::size_t>(budget/block) && candidate.second==0,
            "resident-root restore accounting");
        std::cout << "EXACT_RESIDENT_ROOT fixture=" << (fixture?"prose":"code")
            << " prefix=" << prefix << " tokens=" << budget << " forced_ms=" << control.first
            << " resident_ms=" << candidate.first << " forced_restores=" << control.second
            << " resident_restores=" << candidate.second
            << " resident_tps=" << budget*1000/candidate.first << '\n';
    }

    resident->restore_exact_host_state(*roots[0]);
    require(resident->exact_host_state_resident(*roots[0]) &&
        !resident->exact_host_state_resident(*roots[1]),"resident-root identity discrimination");
    const auto token=greedy(*resident);resident->decode(token);
    require(!resident->exact_host_state_resident(*roots[0]),"resident-root forward invalidation");
    const std::array<std::int64_t,4> prose_tentative={1,2,3,4};
    const auto switched=ninfer::exl3::verify_exl3_outer_batched_reference(
        *resident,*roots[1],prose_tentative,false,{},nullptr,true);
    require(switched.root_restores>=1,"resident-root request switch must restore");
    std::cout << "EXACT_RESIDENT_ROOT PASS fixtures=2 prefix=" << prefix
        << " budget=" << budget << " same_position_divergence=1"
        << " forward_invalidation=1 request_switch_restores=" << switched.root_restores
        << " exact_state=1\n";
}
