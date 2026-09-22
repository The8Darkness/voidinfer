#pragma once

void run_turboangle_l1_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using ninfer::exl3::Exl3TextContext;
    using ninfer::exl3::Exl3TurboAngleL1Context;
    constexpr int prefix=96,block=4;
    require(code.size()>=prefix+16 && prose.size()>=prefix+16 && target.max_context()==256,
        "TurboAngle L1 fixture/context extent");
    _putenv_s("NINFER_EXL3_WIDE_PREFILL","0");
    _putenv_s("NINFER_EXL3_PREFILL_STAGED_REDUCTION","0");
    _putenv_s("NINFER_EXL3_SHARED_ACCUM","0");
    _putenv_s("NINFER_EXL3_SHARED_TRANSFORM","0");
    _putenv_s("NINFER_EXL3_SHARED_LAYER_SCRATCH","0");
    _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    auto l0=target.create_context(true);
    auto l1_native=target.create_context(true);
    require(l0->try_enable_oscar_from_environment() && l1_native->try_enable_oscar_from_environment(),
        "TurboAngle L1 OSCAR contexts");
    auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"TurboAngle L1 finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    std::unique_ptr<Exl3TurboAngleL1Context> l1;
    std::size_t cases=0,total_agreement=0,total_quality_rows=0;
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const auto& source=*fixture.second;
        exact->reset();
        exact->prefill(std::span<const std::int64_t>(source.data(),16));
        for(int row=16;row<prefix;++row) exact->decode(source[row]);
        const auto root=exact->export_exact_host_state();
        const auto pages=Exl3TextContext::make_turboangle_l1_pages(root);
        require(pages->full_history() && pages->first_position()==0 && pages->rows()==prefix &&
            pages->payload_bytes()==static_cast<std::size_t>(prefix)*4*16*(248+168),
            "TurboAngle L1 full-history DRAM coverage");
        l0->restore_oscar_host_state(*root,pages.get());
        std::array<std::int64_t,block> scalar_path{};
        for(auto& token:scalar_path) {token=greedy(*l0);l0->decode(token);}
        const auto scalar_next=greedy(*l0);
        if(!l1) l1=std::make_unique<Exl3TurboAngleL1Context>(std::move(l1_native),root);
        else l1->rebase(root);
        const auto chunk=l1->screen(scalar_path);
        require(!chunk.rejected && chunk.accepted==block && chunk.authorized_tokens.size()==block &&
            std::equal(chunk.authorized_tokens.begin(),chunk.authorized_tokens.end(),scalar_path.begin()) &&
            l1->greedy()==scalar_next,"TurboAngle L1 chunked versus scalar continuation");
        ++cases;

        // The candidate block is fixed. Mutating a distinct L0 through real
        // attention/GDN/convolution forwards cannot change L1 output or state.
        l1->rebase(root);
        const auto before=l1->screen(scalar_path);
        const auto before_next=l1->greedy();
        l1->rebase(root);
        l0->restore_oscar_host_state(*root,pages.get());
        const auto wrong=(greedy(*l0)+1)%248320;
        l0->decode(wrong);
        l0->decode((wrong+17)%248320);
        const auto after=l1->screen(scalar_path);
        require(before.authorized_tokens==after.authorized_tokens && before.accepted==after.accepted &&
            before.rejected==after.rejected && before_next==l1->greedy(),
            "TurboAngle L1 changed after private L0 state perturbation");
        ++cases;

        // Inject a real early mismatch. L1 must discard the dependent native
        // chunk, replay only the matching prefix and advance its own correction.
        l1->rebase(root);
        auto bad=scalar_path;bad[1]=(bad[1]+1)%248320;
        const auto rejected=l1->screen(bad);
        require(rejected.rejected && rejected.accepted==1 && rejected.authorized_tokens.size()==2 &&
            rejected.authorized_tokens[0]==scalar_path[0] && rejected.authorized_tokens[1]==scalar_path[1] &&
            rejected.replay_rows==2 && l1->position()==prefix+2,
            "TurboAngle L1 early corruption detection/repair");
        l0->restore_oscar_host_state(*root,pages.get());
        for(auto token:rejected.authorized_tokens) l0->decode(token);
        require(l1->greedy()==greedy(*l0),"TurboAngle L1 correction continuation");
        const auto checkpoint=l1->checkpoint();
        const auto checkpoint_next=l1->greedy();
        l1->advance(checkpoint_next);
        l1->restore(checkpoint);
        require(l1->greedy()==checkpoint_next && l1->position()==prefix+2,
            "TurboAngle L1 short checkpoint replay");
        ++cases;

        // Rebase only from a newer authoritative L2 image and compare with a
        // fresh scalar TurboAngle reconstruction of that same trusted root.
        exact->restore_exact_host_state(*root);
        for(int row=0;row<2;++row) {const auto token=greedy(*exact);exact->decode(token);}
        const auto newer=exact->export_exact_host_state();
        l1->rebase(newer);
        const auto newer_pages=Exl3TextContext::make_turboangle_l1_pages(newer);
        l0->restore_oscar_host_state(*newer,newer_pages.get());
        require(l1->history_rows()==newer->position() && l1->greedy()==greedy(*l0),
            "TurboAngle L1 authoritative rebase");
        ++cases;

        // Quality is measured on one common teacher-forced L2 prefix; it is
        // not a claim that approximate L1 state equals FP16 L2 numerically.
        exact->restore_exact_host_state(*root);l1->rebase(root);
        std::size_t agreement=0;
        for(int row=0;row<8;++row) {
            const auto l2_token=greedy(*exact),l1_token=l1->greedy();
            agreement+=l1_token==l2_token;
            exact->decode(l2_token);l1->advance(l2_token);
        }
        total_agreement+=agreement;total_quality_rows+=8;
        const auto& stats=l1->stats();
        std::cout << "TURBOANGLE_L1_FIXTURE fixture=" << fixture.first
            << " prefix=" << prefix << " full_history_rows=" << l1->history_rows()
            << " page_bytes=" << l1->page_bytes() << " context_bytes=" << l1->context_bytes()
            << " agreement=" << agreement << "/8 construction_ms=" << stats.construction_ms
            << " restore_ms=" << stats.restore_ms << " execution_ms=" << stats.execution_ms
            << " executed_rows=" << stats.executed_rows << " replay_rows=" << stats.replay_rows << '\n';
    }
    const auto& stats=l1->stats();
    std::cout << "TURBOANGLE_L1 PASS fixtures=2 cases=" << cases
        << " full_history=1 independent_l0=1 early_rejection=1 checkpoint_replay=1 rebase=1"
        << " agreement=" << total_agreement << '/' << total_quality_rows
        << " constructions=" << stats.construction_count << " restores=" << stats.restore_count << '\n';
}
