#pragma once

void run_hierarchical_l2_windows_t70(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Frontiers = ninfer::exl3::Exl3HierarchyFrontiers;
    using L1 = ninfer::exl3::Exl3TurboAngleL1Context;
    constexpr int prefix = 96, budget = 64, block = 4;
    require(target.max_context() == 256 && code.size() >= prefix &&
            prose.size() >= prefix, "T70 fixture/context extent");
    run_hierarchical_frontier_logic();

    _putenv_s("NINFER_EXL3_WIDE_PREFILL", "0");
    _putenv_s("NINFER_EXL3_PREFILL_STAGED_REDUCTION", "0");
    _putenv_s("NINFER_EXL3_SHARED_ACCUM", "0");
    _putenv_s("NINFER_EXL3_SHARED_TRANSFORM", "0");
    _putenv_s("NINFER_EXL3_SHARED_LAYER_SCRATCH", "0");
    _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB", "0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL", "1");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    auto exact = target.create_context(true);
    exact->prepare_continuation(8);
    const auto exact_persistent = exact->persistent_bytes();

    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "1");
    auto l0 = target.create_context(true);
    auto l1_native = target.create_context(true);
    require(l0->try_enable_oscar_from_environment() &&
            l1_native->try_enable_oscar_from_environment(),
            "T70 OSCAR L0/L1 contexts");
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
            "T70 finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
    };
    struct Result {
        std::shared_ptr<const Request> request;
        std::vector<std::int64_t> tokens;
        std::array<std::uint64_t, 9> native_batch_hist{};
        std::uint64_t l0_rows = 0, l1_rows = 0, l1_replay = 0;
        std::uint64_t l1_rejections = 0, l2_windows = 0;
        std::uint64_t l2_verification = 0, l2_replay = 0, l2_native = 0;
        std::uint64_t l2_restores = 0, l2_rejections = 0;
        std::uint64_t publications = 0, downstream_discarded = 0;
        std::size_t peak_pending = 0, l1_context_bytes = 0;
        bool stopped = false, l0_fault = false, l2_fault = false;
        bool cancelled = false;
    };

    std::unique_ptr<L1> l1;
    std::uint64_t request_id = 100;
    const auto run = [&](std::shared_ptr<const Request> initial, int limit,
                         int window, std::span<const std::int64_t> reference,
                         int force_l0_at = -1, int force_l2_at = -1,
                         std::span<const std::int64_t> terminal = {},
                         int cancel_after = 0) {
        require(window == 8 || window == 16 || window == 32,
                "T70 window selection");
        Result result;
        result.request = std::move(initial);
        auto current = result.request->state();
        l0->restore_oscar_host_state(*current);
        if (!l1) l1 = std::make_unique<L1>(std::move(l1_native), current);
        else l1->rebase(current);
        result.l1_context_bytes = l1->context_bytes();
        Frontiers frontiers(++request_id, current->position());
        int attempted_ordinal = 0;
        while (result.tokens.size() < static_cast<std::size_t>(limit) &&
               !result.stopped) {
            const std::size_t boundary = std::min<std::size_t>(
                window, static_cast<std::size_t>(limit) - result.tokens.size());
            std::vector<std::int64_t> pending;
            pending.reserve(boundary);
            while (pending.size() < boundary) {
                const int rows = static_cast<int>(std::min<std::size_t>(
                    block, boundary - pending.size()));
                const auto l0_completion = frontiers.begin_l0(rows);
                std::vector<std::int64_t> candidates;
                candidates.reserve(rows);
                for (int row = 0; row < rows; ++row, ++attempted_ordinal) {
                    const auto token = greedy(*l0);
                    candidates.push_back(token);
                    l0->decode(token);
                    ++result.l0_rows;
                }
                if (!result.l0_fault && force_l0_at >= attempted_ordinal - rows &&
                    force_l0_at < attempted_ordinal) {
                    const int local = force_l0_at - (attempted_ordinal - rows);
                    require(local == 0, "T70 forced L0 fault must begin a block");
                    candidates[0] = (l1->greedy() + 1) % 248320;
                    result.l0_fault = true;
                }
                const auto screened = rows == 1
                    ? l1->screen_one(candidates[0]) : l1->screen(candidates);
                result.l1_rows += screened.executed_rows;
                result.l1_replay += screened.replay_rows;
                result.l1_rejections += screened.rejected;
                frontiers.complete_l1(l0_completion,
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
                if (cancel_after > 0 &&
                    pending.size() >= static_cast<std::size_t>(cancel_after)) {
                    frontiers.cancel_tentative();
                    l1->rebase(current);
                    l0->restore_oscar_host_state(*current);
                    result.cancelled = true;
                    require(frontiers.published() == current->position() &&
                            frontiers.authoritative() == current->position() &&
                            frontiers.screened() == current->position() &&
                            frontiers.drafted() == current->position(),
                            "T70 cancellation settled frontiers");
                    return result;
                }
                if (std::any_of(pending.begin(), pending.end(), [&](auto token) {
                        return ninfer::exl3::exl3_terminal_token(token, terminal);
                    })) break;
            }
            require(!pending.empty() && pending.size() <=
                    static_cast<std::size_t>(window), "T70 private lead bound");
            if (!result.l2_fault && force_l2_at >= 0) {
                require(force_l2_at < static_cast<int>(pending.size()) &&
                        result.tokens.size() + pending.size() <= reference.size(),
                        "T70 forced L2 fault extent");
                for (std::size_t row = 0; row < pending.size(); ++row)
                    pending[row] = reference[result.tokens.size() + row];
                pending[force_l2_at] = (pending[force_l2_at] + 1) % 248320;
                result.l2_fault = true;
            }
            const auto l2_completion = frontiers.begin_l2();
            require(l2_completion.first == current->position() &&
                    l2_completion.rows == static_cast<int>(pending.size()),
                    "T70 L2 dependency covers complete screened window");
            auto [updated, verified] = result.request->verify_compact(
                *exact, draft, staging, pending, terminal);
            ++result.l2_windows;
            result.l2_verification += verified.verification_rows;
            result.l2_replay += verified.replay_rows;
            result.l2_native += verified.native_invocations;
            result.l2_restores += verified.root_restores;
            result.l2_rejections += verified.rejected;
            for (int rows = 1; rows <= 8; ++rows)
                result.native_batch_hist[rows] += verified.native_batch_hist[rows];
            if (verified.rejected)
                result.downstream_discarded += pending.size() - verified.accepted;
            frontiers.complete_l2(l2_completion,
                static_cast<int>(verified.committed_tokens.size()));
            const auto publication = frontiers.publish_authorized();
            require(publication.first == current->position() &&
                    publication.rows ==
                        static_cast<int>(verified.committed_tokens.size()) &&
                    publication.root_version == l2_completion.root_version &&
                    publication.epoch == l2_completion.epoch,
                    "T70 current L2 publication identity/range");
            result.tokens.insert(result.tokens.end(),
                verified.committed_tokens.begin(), verified.committed_tokens.end());
            ++result.publications;
            result.request = std::move(updated);
            current = result.request->state();
            l1->rebase_delta(current);
            l0->restore_oscar_host_state(*current);
            result.stopped = verified.stopped;
            frontiers.rebase();
            require(frontiers.published() == current->position() &&
                    frontiers.published() == frontiers.authoritative() &&
                    frontiers.authoritative() == frontiers.screened() &&
                    frontiers.screened() == frontiers.drafted(),
                    "T70 settled publication frontiers");
        }
        require(result.peak_pending <= static_cast<std::size_t>(window) &&
                result.l1_context_bytes == l1->context_bytes() &&
                exact_persistent == exact->persistent_bytes() &&
                l0_persistent == l0->persistent_bytes(),
                "T70 bounded persistent ownership");
        const auto native_total = std::accumulate(result.native_batch_hist.begin(),
            result.native_batch_hist.end(), std::uint64_t{0});
        require(native_total == result.l2_native &&
                result.l2_verification + result.l2_replay >= result.tokens.size(),
                "T70 native work accounting");
        return result;
    };

    const auto equal = [&](const Result& actual, const Result& reference,
                           const char* label) {
        require(actual.tokens == reference.tokens &&
                actual.stopped == reference.stopped &&
                actual.request->state()->same_payload(*reference.request->state()) &&
                actual.request->token_suffix() == reference.request->token_suffix(),
                label);
        actual.request->restore_draft(draft, staging);
        const auto ring = draft.export_host_ring(nullptr, false);
        reference.request->restore_draft(draft, staging);
        require(ring->same_payload(*draft.export_host_ring(nullptr, false)), label);
    };
    const auto print = [](const char* fixture, int window, const Result& result) {
        std::cout << "T70_L2_WINDOW fixture=" << fixture << " window=" << window
            << " committed=" << result.tokens.size()
            << " publications=" << result.publications
            << " peak_pending=" << result.peak_pending
            << " l0_rows=" << result.l0_rows
            << " l1_rows=" << result.l1_rows
            << " l1_replay=" << result.l1_replay
            << " l1_rejections=" << result.l1_rejections
            << " l2_windows=" << result.l2_windows
            << " l2_verification=" << result.l2_verification
            << " l2_replay=" << result.l2_replay
            << " l2_native=" << result.l2_native
            << " l2_restores=" << result.l2_restores
            << " l2_rejections=" << result.l2_rejections
            << " downstream_discarded=" << result.downstream_discarded
            << " exact_state=1\n";
    };

    for (const auto& [name, source] :
         std::array<std::pair<const char*, const std::vector<std::int64_t>*>, 2>{
             std::pair{"code", &code}, std::pair{"prose", &prose}}) {
        _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
        const auto root = Request::initialize(*exact,
            std::span<const std::int64_t>(source->data(), prefix), 8);
        const auto compact = root->compact_draft(draft, staging);
        std::vector<std::int64_t> reference;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> reference50;
        exact->restore_exact_host_state(*root->state());
        for (int row = 0; row < budget; ++row) {
            const auto token = greedy(*exact);
            exact->decode(token);
            reference.push_back(token);
            if (row == 49) reference50 = exact->export_exact_host_state();
        }
        const auto reference64 = exact->export_exact_host_state();

        const auto w8 = run(compact, budget, 8, reference);
        const auto w16 = run(compact, budget, 16, reference);
        const auto w32 = run(compact, budget, 32, reference);
        require(w8.tokens == reference && w16.tokens == reference &&
                w32.tokens == reference &&
                w8.request->state()->same_payload(*reference64) &&
                w16.request->state()->same_payload(*reference64) &&
                w32.request->state()->same_payload(*reference64),
                "T70 serial oracle output/state");
        equal(w16, w8, "T70 W16 complete request equivalence");
        equal(w32, w8, "T70 W32 complete request equivalence");
        require(w16.peak_pending == 16 && w32.peak_pending == 32 &&
                w16.l1_rows > w16.l2_windows &&
                w32.l1_rows > w32.l2_windows,
                "T70 genuine multi-round L1 windows");
        print(name, 8, w8);
        print(name, 16, w16);
        print(name, 32, w32);

        if (std::string_view(name) == "code") {
            const auto l0_fault = run(compact, budget, 16, reference, 4);
            equal(l0_fault, w8, "T70 forced L0/L1 repair");
            require(l0_fault.l0_fault && l0_fault.l1_rejections > 0 &&
                    l0_fault.l1_replay > 0, "T70 forced L1 rollback accounting");
            const auto early = run(compact, budget, 16, reference, -1, 1);
            const auto late = run(compact, budget, 32, reference, -1, 30);
            equal(early, w8, "T70 early L2 repair");
            equal(late, w8, "T70 late L2 repair");
            require(early.l2_fault && late.l2_fault &&
                    early.l2_rejections > 0 && late.l2_rejections > 0 &&
                    early.l2_replay > 0 && late.l2_replay > 0 &&
                    early.downstream_discarded > 0 &&
                    late.downstream_discarded > 0,
                    "T70 forced L2 repair/discard accounting");
            const auto partial = run(compact, 50, 32, reference);
            require(partial.tokens == std::vector<std::int64_t>(
                        reference.begin(), reference.begin() + 50) &&
                    partial.request->state()->same_payload(*reference50) &&
                    partial.peak_pending == 32,
                    "T70 partial final window exactness");
            std::size_t terminal_index = 9;
            while (terminal_index < 32 &&
                   std::find(reference.begin(), reference.begin() + terminal_index,
                             reference[terminal_index]) !=
                       reference.begin() + terminal_index)
                ++terminal_index;
            require(terminal_index < 32, "T70 unique terminal fixture token");
            exact->restore_exact_host_state(*root->state());
            for(std::size_t row=0;row<=terminal_index;++row)
                exact->decode(reference[row]);
            const auto reference_terminal=exact->export_exact_host_state();
            const std::array<std::int64_t, 1> terminal{
                reference[terminal_index]};
            const auto stopped = run(compact, 32, 32, reference, -1, -1,
                                     terminal);
            std::cout << "T70_TERMINAL_GATE index=" << terminal_index
                << " stopped=" << stopped.stopped
                << " committed=" << stopped.tokens.size() << '\n';
            require(stopped.stopped &&
                    stopped.tokens.size() == terminal_index + 1 &&
                    std::equal(stopped.tokens.begin(), stopped.tokens.end(),
                               reference.begin()) &&
                    stopped.request->state()->same_payload(*reference_terminal),
                    "T70 terminal exposure flush exactness");
            const auto cancelled = run(compact, budget, 16, reference,
                                       -1, -1, {}, 12);
            require(cancelled.cancelled && cancelled.tokens.empty() &&
                    cancelled.publications == 0 && cancelled.l2_windows == 0 &&
                    cancelled.request.get() == compact.get() &&
                    cancelled.request->state()->same_payload(*root->state()),
                    "T70 cancellation leaves immutable L2 root");
        }
    }
    std::cout << "T70_HIERARCHICAL_L2_WINDOWS PASS fixtures=2 prefix=96 "
                 "budget=64 windows=8,16,32 multi_l0_l1=1 exact_state=1 "
                 "l0_l1_repair=1 early_l2_repair=1 late_l2_repair=1 "
                 "partial=1 terminal=1 cancellation=1 stale_epoch=1 "
                 "production_changed=0"
              << std::endl;
}
