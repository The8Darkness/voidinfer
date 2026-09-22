#pragma once

void run_hierarchical_4k_profile(Exl3TextModel& target) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using ninfer::exl3::Exl3TextContext;
    using ninfer::exl3::Exl3TurboAngleL1Context;
    using ninfer::exl3::Exl3TurboAngleL1Stats;
    constexpr int block=4;
    require(target.max_context()==4352,"hierarchical 4K profile context extent");
    auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"hierarchical 4K finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
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
        "hierarchical 4K OSCAR contexts");
    std::unique_ptr<Exl3TurboAngleL1Context> l1;
    for(const std::string fixture:{"code","prose"}) {
        const auto ids=load_ids((std::filesystem::path(env("NINFER_COMPACT_PERF_DIR"))/(fixture+".ids")).string());
        require(ids.size()==4096,"hierarchical 4K fixture extent");
        const auto prefill_start=std::chrono::steady_clock::now();
        const auto request=Request::initialize(*exact,ids,1024);
        const auto root=request->state();
        const double prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prefill_start).count();
        std::array<std::int64_t,block> reference{};
        std::array<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>,block> reference_states;
        exact->restore_exact_host_state(*root);
        for(int row=0;row<block;++row) {
            reference[row]=greedy(*exact);exact->decode(reference[row]);
            reference_states[row]=exact->export_exact_host_state();
        }

        l0->restore_oscar_host_state(*root);
        const auto w_start=std::chrono::steady_clock::now();
        std::array<std::int64_t,block> w_candidates{};
        for(auto& token:w_candidates) {token=greedy(*l0);l0->decode(token);}
        const auto w_proposed=std::chrono::steady_clock::now();
        const auto w_verified=ninfer::exl3::verify_exl3_outer_batched_reference(
            *exact,*root,w_candidates,false,{},nullptr,true);
        const auto w_end=std::chrono::steady_clock::now();
        require(std::equal(w_verified.committed_tokens.begin(),w_verified.committed_tokens.end(),reference.begin()) &&
            w_verified.committed_state->same_payload(*reference_states[w_verified.committed_tokens.size()-1]),
            "hierarchical 4K W output/state");

        const Exl3TurboAngleL1Stats before=l1?l1->stats():Exl3TurboAngleL1Stats{};
        if(!l1) l1=std::make_unique<Exl3TurboAngleL1Context>(std::move(l1_native),root);
        else l1->rebase(root);
        const auto ready=l1->stats();
        const double setup_construct_ms=ready.construction_ms-before.construction_ms;
        const double setup_restore_ms=ready.restore_ms-before.restore_ms;
        l0->restore_oscar_host_state(*root);
        const auto c_start=std::chrono::steady_clock::now();
        std::array<std::int64_t,block> c_candidates{};
        for(auto& token:c_candidates) {token=greedy(*l0);l0->decode(token);}
        const auto c_proposed=std::chrono::steady_clock::now();
        const auto screened=l1->screen(c_candidates);
        const auto c_screened=std::chrono::steady_clock::now();
        const auto c_verified=screened.authorized_tokens.size()==1
            ? ninfer::exl3::verify_exl3_outer_reference(
                *exact,*root,screened.authorized_tokens,false,{},nullptr,true)
            : ninfer::exl3::verify_exl3_outer_batched_reference(
                *exact,*root,screened.authorized_tokens,false,{},nullptr,true);
        const auto c_released=std::chrono::steady_clock::now();
        require(std::equal(c_verified.committed_tokens.begin(),c_verified.committed_tokens.end(),reference.begin()) &&
            c_verified.committed_state->same_payload(*reference_states[c_verified.committed_tokens.size()-1]),
            "hierarchical 4K C_sync output/state");
        const auto pre_rebase=l1->stats();
        l1->rebase_delta(c_verified.committed_state);
        const auto post_rebase=l1->stats();
        const auto c_ready=std::chrono::steady_clock::now();
        const auto ms=[](auto begin,auto end){return std::chrono::duration<double,std::milli>(end-begin).count();};
        std::cout << "HIERARCHICAL_4K fixture=" << fixture << " prefix=4096 block=4"
            << " prefill_ms=" << prefill_ms
            << " w_proposal_ms=" << ms(w_start,w_proposed) << " w_l2_ms=" << ms(w_proposed,w_end)
            << " w_release_ms=" << ms(w_start,w_end)
            << " c_setup_construct_ms=" << setup_construct_ms << " c_setup_restore_ms=" << setup_restore_ms
            << " c_proposal_ms=" << ms(c_start,c_proposed) << " c_l1_ms=" << ms(c_proposed,c_screened)
            << " c_l2_ms=" << ms(c_screened,c_released)
            << " c_release_excluding_setup_ms=" << ms(c_start,c_released)
            << " c_release_including_setup_ms=" << setup_construct_ms+setup_restore_ms+ms(c_start,c_released)
            << " c_post_commit_construct_ms=" << post_rebase.construction_ms-pre_rebase.construction_ms
            << " c_post_commit_restore_ms=" << post_rebase.restore_ms-pre_rebase.restore_ms
            << " c_cycle_ready_ms=" << setup_construct_ms+setup_restore_ms+ms(c_start,c_ready)
            << " l1_accepted=" << screened.accepted << " l1_rejected=" << screened.rejected
            << " l2_rows=" << c_verified.verification_rows << " l2_replay=" << c_verified.replay_rows
            << " page_bytes=" << l1->page_bytes() << " context_bytes=" << l1->context_bytes() << '\n';
    }
    std::cout << "HIERARCHICAL_4K PASS fixtures=2 block=4 exact_output_state=1\n";
}
