#pragma once

void run_outer_reference_qualification(Exl3TextModel& target,
    Exl3Dflash2DraftModel& draft, const std::vector<std::int64_t>& source) {
    require(target.max_context() == 1024 && source.size() >= 700, "outerref fixture extent");
    auto inner = target.create_context(true);
    auto exact = target.create_context(true);
    require(inner->try_enable_oscar_from_environment(), "outerref OSCAR enable");
    inner->prepare_transaction(); inner->prepare_continuation(8);
    exact->prepare_continuation(8);
    TapStage stage;
    for (int tap = 0; tap < kTapCount; ++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16ULL * kHidden * 2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden * 2));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    std::array<std::vector<std::int64_t>, 2> histories;
    std::array<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>, 2> l2;
    const auto exact_ingest = [&](const std::vector<std::int64_t>& ids) {
        exact->reset(); exact->prefill(std::span<const std::int64_t>(ids.data(), 16));
        for (std::size_t i = 16; i < ids.size(); ++i) exact->decode(ids[i]);
    };
    const auto rebuild_inner = [&](const std::vector<std::int64_t>& ids) {
        inner->reset(); draft.reset();
        ingest_prefix(*inner, ids, CommitSink{&draft, &stage});
        require(inner->position() == ids.size() &&
                draft.ring_base_abs() + draft.ring_count() == ids.size(), "outerref repaired ownership");
    };
    for (int request = 0; request < 2; ++request) {
        histories[request].assign(source.begin() + request * 350, source.begin() + request * 350 + 321);
        exact_ingest(histories[request]);
        l2[request] = exact->export_exact_host_state();
    }
    // T186: host tap vectors are optional only for callers that do not publish
    // or compose a compact request. Authoritative state and tokens are never
    // optional. Exercise scalar, native-B8 and windowed-B9 with the same root
    // both with and without host diagnostics, and require a device consumer to
    // present its strong source binding before the context is touched.
    const auto artifact_route=[&](std::span<const std::int64_t> tentative,bool taps,
                                  const ninfer::exl3::Exl3CommittedTapConsumer* consumer=nullptr) {
        if(tentative.size()==1)
            return verify_exl3_outer_reference(*exact,*l2[0],tentative,taps,{},nullptr,false,nullptr,consumer);
        if(tentative.size()<=8)
            return verify_exl3_outer_batched_reference(*exact,*l2[0],tentative,taps,{},nullptr,false,nullptr,consumer);
        return verify_exl3_outer_windowed_reference(*exact,*l2[0],tentative,taps,{},nullptr,false,nullptr,consumer);
    };
    for(int rows:{1,8,9}) {
        std::vector<std::int64_t> tentative(source.begin()+321,source.begin()+321+rows);
        const auto without=artifact_route(tentative,false);
        const auto with=artifact_route(tentative,true);
        require(without.committed_state && with.committed_state &&
                without.committed_tokens==with.committed_tokens &&
                without.committed_state->same_payload(*with.committed_state),
                "outerref optional diagnostics changed mandatory result");
        require(std::all_of(without.committed_taps.begin(),without.committed_taps.end(),
                    [](const auto& tap){return tap.empty();}) &&
                without.committed_tap_segments.empty(),
                "outerref omitted diagnostic materialized");
        require(!with.committed_tokens.empty() &&
                std::all_of(with.committed_taps.begin(),with.committed_taps.end(),
                    [&](const auto& tap){return tap.size()==with.committed_tokens.size()*kHidden;}) &&
                !with.committed_tap_segments.empty(),
                "outerref requested diagnostic missing");
        exact->restore_exact_host_state(*l2[0]);
        const auto before=exact->export_exact_host_state();
        ninfer::exl3::Exl3CommittedTapConsumer missing_binding=[](
            const ninfer::exl3::Exl3CommittedTapSegment&,cudaStream_t){};
        bool refused=false;
        try{(void)artifact_route(tentative,false,&missing_binding);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused && exact->export_exact_host_state()->same_payload(*before),
                "outerref unbound required consumer mutated state");
    }
    std::size_t inner_accepted = 0, outer_accepted = 0, outer_proposed = 0, committed = 0;
    int rejects = 0;
    double complete_ms = 0;
    for (int round = 0; round < 3; ++round) {
        for (int request = 0; request < 2; ++request) {
            const auto start = std::chrono::steady_clock::now();
            rebuild_inner(histories[request]);
            const auto seed = sample_target(*inner); // still tentative at outer boundary
            std::vector<std::int64_t> block(8, kMaskToken); block[0] = seed;
            const auto proposals = draft.propose_cached(block, inner->position(), inner->target_embedding(),
                inner->target_lm_head_weights(), inner->target_lm_head_metadata(), kMaskToken);
            const auto local = verify_pending_round_transactional(*inner, draft, stage, seed,
                proposals, inner->position(), nullptr, -1, 0x7e00u, TransactionRepairMode::retain_prefix);
            inner_accepted += local.accepted;
            std::vector<std::int64_t> tentative{seed};
            tentative.insert(tentative.end(), local.emitted.begin(), local.emitted.end());
            // Deterministic adversarial outer rejection; separately identified,
            // never counted as a natural DFlash2 conditional correctness result.
            if (round == 1) {
                exact->restore_exact_host_state(*l2[request]);
                tentative[0] = (sample_target(*exact) + 1) % kVocab;
            }
            outer_proposed += tentative.size();
            const auto result = ninfer::exl3::verify_exl3_outer_reference(*exact, *l2[request], tentative);
            outer_accepted += result.accepted;
            rejects += result.rejected;
            committed += result.committed_tokens.size();
            auto updated = histories[request];
            updated.insert(updated.end(), result.committed_tokens.begin(), result.committed_tokens.end());
            // Rebuild all approximate upstream and draft state, including pending
            // token consumption, before publishing this request's next checkpoint.
            rebuild_inner(updated);
            l2[request] = result.committed_state;
            histories[request] = std::move(updated);
            cuda_check(cudaDeviceSynchronize(), "outerref complete round");
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            complete_ms += ms;
            // Independent full serial oracle is excluded from complete-round
            // timer, but every restore/draft/decision/repair/export is included.
            exact_ingest(histories[request]);
            require(l2[request]->same_payload(*exact->export_exact_host_state()),
                    "outerref authoritatively committed full state differs from serial");
            require(!inner->transaction_active(), "outerref leaked inner transaction");
            std::cout << "OUTER_REFERENCE_ROUND request=" << request << " round=" << round
                      << " injected=" << (round == 1) << " inner_accepted=" << local.accepted
                      << " outer_accepted=" << result.accepted << " proposed=" << tentative.size()
                      << " committed=" << result.committed_tokens.size() << " complete_ms=" << ms << '\n';
        }
    }
    require(rejects >= 2, "outerref adversarial rejection absent");
    std::cout << "OUTER_REFERENCE PASS rounds=6 inner_proposed=42 inner_accepted=" << inner_accepted
              << " outer_proposed=" << outer_proposed << " outer_accepted=" << outer_accepted
              << " committed=" << committed << " rejected_rounds=" << rejects
              << " complete_ms=" << complete_ms << " committed_tps=" << committed * 1000 / complete_ms
              << " identity=vericache_exact_fp16_eager l1=absent_comparison_baseline\n";
}
