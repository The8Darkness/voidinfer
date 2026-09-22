#pragma once

void run_hierarchical_l2_windows_t70_t1(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Frontiers = ninfer::exl3::Exl3HierarchyFrontiers;
    using L1 = ninfer::exl3::Exl3TurboAngleL1Context;
    using Clock = std::chrono::steady_clock;
    constexpr int prefix = 4096, budget = 128, block = 4;
    require(target.max_context() == 4352 && code.size() >= prefix &&
            prose.size() >= prefix, "T70 T1 fixture/context extent");
    const std::filesystem::path output = env("NINFER_T70_T1_OUT");
    require(!output.empty() && !std::filesystem::exists(output),
            "T70 T1 output directory must be new");
    std::filesystem::create_directories(output);
    std::ofstream pairs(output / "pairs.csv");
    require(pairs.good(), "T70 T1 pair evidence creation");
    pairs << "fixture,candidate_window,pair,order,arm,wall_ms,tps,setup_ms,"
             "first_release_ms,gap_p95_ms,l0_rows,l1_rows,l1_replay,"
             "l1_rejections,l2_windows,l2_verification,l2_replay,l2_native,"
             "l2_restores,l2_rejections,publications,downstream_discarded,"
             "h2d_bytes,d2h_bytes,l2_ms,sync_ms,peak_pending,exact\n";

    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL", "1");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    auto exact = target.create_context(true);
    exact->prepare_continuation(8);
    const auto exact_persistent = exact->persistent_bytes();

    _putenv_s("NINFER_EXL3_WIDE_PREFILL", "0");
    _putenv_s("NINFER_EXL3_PREFILL_STAGED_REDUCTION", "0");
    _putenv_s("NINFER_EXL3_SHARED_ACCUM", "0");
    _putenv_s("NINFER_EXL3_SHARED_TRANSFORM", "0");
    _putenv_s("NINFER_EXL3_SHARED_LAYER_SCRATCH", "0");
    _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "1");
    auto l0 = target.create_context(true);
    auto l1_native = target.create_context(true);
    require(l0->try_enable_oscar_from_environment() &&
            l1_native->try_enable_oscar_from_environment(),
            "T70 T1 OSCAR L0/L1 contexts");
    l0->prepare_continuation(8);
    const auto l0_persistent = l0->persistent_bytes();

    TapStage stage;
    std::array<std::uint16_t*, 5> staging{};
    for (int tap = 0; tap < 5; ++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16ULL * kHidden * 2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden * 2));
        staging[tap] = static_cast<std::uint16_t*>(stage.bulk.back()->get());
        stage.bulk_ptrs.push_back(staging[tap]);
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    const auto greedy = [](Exl3TextContext& context) {
        const auto logits = context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(), logits.end(),
            [](float value) { return !std::isfinite(value); }),
            "T70 T1 finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    };
    const auto elapsed_ms = [](Clock::time_point start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    const auto p95 = [](std::vector<double> values) {
        require(!values.empty(), "T70 T1 p95 extent");
        std::sort(values.begin(), values.end());
        return values[static_cast<std::size_t>(std::ceil(0.95 * values.size())) - 1];
    };
    struct Result {
        std::shared_ptr<const Request> request;
        std::vector<std::int64_t> tokens;
        std::vector<double> release_ms;
        double wall_ms = 0, setup_ms = 0, first_release_ms = 0;
        double gap_p95_ms = 0, l2_ms = 0, sync_ms = 0;
        std::uint64_t l0_rows = 0, l1_rows = 0, l1_replay = 0;
        std::uint64_t l1_rejections = 0, l2_windows = 0;
        std::uint64_t l2_verification = 0, l2_replay = 0, l2_native = 0;
        std::uint64_t l2_restores = 0, l2_rejections = 0;
        std::uint64_t publications = 0, downstream_discarded = 0;
        std::uint64_t h2d_bytes = 0, d2h_bytes = 0;
        std::size_t peak_pending = 0, l1_context_bytes = 0;
    };

    std::unique_ptr<L1> l1;
    std::uint64_t request_id = 1000;
    const auto run = [&](std::shared_ptr<const Request> initial, int window) {
        require(window == 8 || window == 16 || window == 32,
                "T70 T1 window selection");
        Result result;
        result.request = std::move(initial);
        auto current = result.request->state();
        const auto before = exact->host_kv_stats();
        const auto started = Clock::now();
        l0->restore_oscar_host_state(*current);
        if (!l1) l1 = std::make_unique<L1>(std::move(l1_native), current);
        else l1->rebase(current);
        result.l1_context_bytes = l1->context_bytes();
        result.setup_ms = elapsed_ms(started);
        Frontiers frontiers(++request_id, current->position());
        while (result.tokens.size() < budget) {
            const std::size_t boundary = std::min<std::size_t>(
                window, budget - result.tokens.size());
            std::vector<std::int64_t> pending;
            pending.reserve(boundary);
            while (pending.size() < boundary) {
                const int rows = static_cast<int>(std::min<std::size_t>(
                    block, boundary - pending.size()));
                const auto completion = frontiers.begin_l0(rows);
                std::vector<std::int64_t> candidates;
                candidates.reserve(rows);
                for (int row = 0; row < rows; ++row) {
                    const auto token = greedy(*l0);
                    candidates.push_back(token);
                    l0->decode(token);
                    ++result.l0_rows;
                }
                const auto screened = rows == 1
                    ? l1->screen_one(candidates[0]) : l1->screen(candidates);
                result.l1_rows += screened.executed_rows;
                result.l1_replay += screened.replay_rows;
                result.l1_rejections += screened.rejected;
                frontiers.complete_l1(completion,
                    static_cast<int>(screened.authorized_tokens.size()),
                    screened.rejected);
                if (screened.rejected) {
                    l0->restore_oscar_host_state(*current);
                    for (auto token : pending) l0->decode(token);
                    for (auto token : screened.authorized_tokens) l0->decode(token);
                }
                pending.insert(pending.end(), screened.authorized_tokens.begin(),
                               screened.authorized_tokens.end());
                result.peak_pending = std::max(result.peak_pending, pending.size());
            }
            const auto l2_completion = frontiers.begin_l2();
            require(l2_completion.rows == static_cast<int>(pending.size()),
                    "T70 T1 complete L2 dependency");
            const auto l2_started = Clock::now();
            auto [updated, verified] = result.request->verify_compact(
                *exact, draft, staging, pending);
            result.l2_ms += elapsed_ms(l2_started);
            ++result.l2_windows;
            result.l2_verification += verified.verification_rows;
            result.l2_replay += verified.replay_rows;
            result.l2_native += verified.native_invocations;
            result.l2_restores += verified.root_restores;
            result.l2_rejections += verified.rejected;
            if (verified.rejected)
                result.downstream_discarded += pending.size() - verified.accepted;
            const auto sync_started = Clock::now();
            cuda_check(cudaDeviceSynchronize(), "T70 T1 authoritative publication");
            result.sync_ms += elapsed_ms(sync_started);
            frontiers.complete_l2(l2_completion,
                static_cast<int>(verified.committed_tokens.size()));
            const auto publication = frontiers.publish_authorized();
            require(publication.first == current->position() &&
                    publication.rows ==
                        static_cast<int>(verified.committed_tokens.size()),
                    "T70 T1 publication range");
            const double released = elapsed_ms(started);
            if (result.release_ms.empty()) result.first_release_ms = released;
            result.release_ms.insert(result.release_ms.end(),
                verified.committed_tokens.size(), released);
            result.tokens.insert(result.tokens.end(),
                verified.committed_tokens.begin(), verified.committed_tokens.end());
            ++result.publications;
            result.request = std::move(updated);
            current = result.request->state();
            l1->rebase_delta(current);
            l0->restore_oscar_host_state(*current);
            frontiers.rebase();
        }
        cuda_check(cudaDeviceSynchronize(), "T70 T1 complete request");
        result.wall_ms = elapsed_ms(started);
        const auto after = exact->host_kv_stats();
        result.h2d_bytes = after.h2d_bytes - before.h2d_bytes;
        result.d2h_bytes = after.d2h_bytes - before.d2h_bytes;
        std::vector<double> gaps;
        for (std::size_t row = 1; row < result.release_ms.size(); ++row)
            gaps.push_back(result.release_ms[row] - result.release_ms[row - 1]);
        result.gap_p95_ms = p95(std::move(gaps));
        require(result.tokens.size() == budget &&
                result.peak_pending <= static_cast<std::size_t>(window) &&
                result.l1_context_bytes == l1->context_bytes() &&
                exact_persistent == exact->persistent_bytes() &&
                l0_persistent == l0->persistent_bytes(),
                "T70 T1 output/ownership gate");
        return result;
    };
    const auto equal = [&](const Result& actual, const Result& reference,
                           const char* label) {
        require(actual.tokens == reference.tokens &&
                actual.request->state()->same_payload(*reference.request->state()) &&
                actual.request->token_suffix() == reference.request->token_suffix(),
                label);
        actual.request->restore_draft(draft, staging);
        const auto ring = draft.export_host_ring(nullptr, false);
        reference.request->restore_draft(draft, staging);
        require(ring->same_payload(*draft.export_host_ring(nullptr, false)), label);
    };
    const auto emit = [&](const char* fixture, int candidate_window, int pair,
                          const char* order, const char* arm,
                          const Result& result) {
        pairs << std::setprecision(12) << fixture << ',' << candidate_window
            << ',' << pair << ',' << order << ',' << arm << ','
            << result.wall_ms << ',' << budget * 1000.0 / result.wall_ms << ','
            << result.setup_ms << ',' << result.first_release_ms << ','
            << result.gap_p95_ms << ',' << result.l0_rows << ','
            << result.l1_rows << ',' << result.l1_replay << ','
            << result.l1_rejections << ',' << result.l2_windows << ','
            << result.l2_verification << ',' << result.l2_replay << ','
            << result.l2_native << ',' << result.l2_restores << ','
            << result.l2_rejections << ',' << result.publications << ','
            << result.downstream_discarded << ',' << result.h2d_bytes << ','
            << result.d2h_bytes << ',' << result.l2_ms << ',' << result.sync_ms
            << ',' << result.peak_pending << ",1\n";
        pairs.flush();
    };

    for (const auto& [name, source] :
         std::array<std::pair<const char*, const std::vector<std::int64_t>*>, 2>{
             std::pair{"code", &code}, std::pair{"prose", &prose}}) {
        _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
        const auto root = Request::initialize(*exact,
            std::span<const std::int64_t>(source->data(), prefix), 1024);
        const auto compact = root->compact_draft(draft, staging);
        std::vector<std::int64_t> reference;
        exact->restore_exact_host_state(*root->state());
        for (int row = 0; row < budget; ++row) {
            const auto token = greedy(*exact);
            reference.push_back(token);
            exact->decode(token);
        }
        const auto reference_state = exact->export_exact_host_state();
        const auto warm8 = run(compact, 8);
        const auto warm16 = run(compact, 16);
        const auto warm32 = run(compact, 32);
        equal(warm16, warm8, "T70 T1 warm W16 exactness");
        equal(warm32, warm8, "T70 T1 warm W32 exactness");
        require(warm8.tokens == reference &&
                warm8.request->state()->same_payload(*reference_state),
                "T70 T1 serial oracle warm gate");
        for (int candidate_window : {16, 32}) {
            for (int pair = 0; pair < 3; ++pair) {
                Result baseline, candidate;
                const bool candidate_first = pair == 1;
                if (candidate_first) {
                    candidate = run(compact, candidate_window);
                    baseline = run(compact, 8);
                } else {
                    baseline = run(compact, 8);
                    candidate = run(compact, candidate_window);
                }
                equal(candidate, baseline, "T70 T1 paired exactness");
                require(candidate.tokens == reference &&
                        candidate.request->state()->same_payload(*reference_state),
                        "T70 T1 paired serial oracle");
                const char* order = candidate_first ? "BA" : "AB";
                emit(name, candidate_window, pair, order, "baseline", baseline);
                emit(name, candidate_window, pair, order, "candidate", candidate);
                std::cout << "T70_T1_PAIR fixture=" << name
                    << " window=" << candidate_window << " pair=" << pair
                    << " order=" << order
                    << " baseline_tps=" << budget * 1000.0 / baseline.wall_ms
                    << " candidate_tps=" << budget * 1000.0 / candidate.wall_ms
                    << " gain_pct="
                    << (baseline.wall_ms - candidate.wall_ms) * 100.0 /
                           baseline.wall_ms
                    << " exact=1\n";
            }
        }
    }
    std::cout << "T70_HIERARCHICAL_L2_WINDOWS_T1 PASS fixtures=2 prefix=4096 "
                 "budget=128 pairs_per_fixture_size=3 windows=8,16,32 "
                 "exact_state_history_taps_ring=1 production_changed=0"
              << std::endl;
}
