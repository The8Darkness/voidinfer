#pragma once

struct T69BMetric {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
    std::uint64_t values = 0;
};

T69BMetric t69b_float_metric(const std::vector<float>& actual,
                             const std::vector<float>& reference) {
    require(actual.size() == reference.size(), "T69B float metric extent");
    double error_sq = 0.0, norm_sq = 0.0, max_abs = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(reference[i]),
                "T69B nonfinite float state");
        const double difference = static_cast<double>(actual[i]) - reference[i];
        error_sq += difference * difference;
        norm_sq += static_cast<double>(reference[i]) * reference[i];
        max_abs = std::max(max_abs, std::abs(difference));
    }
    return {max_abs, norm_sq > 0.0 ? std::sqrt(error_sq / norm_sq) : 0.0,
            static_cast<std::uint64_t>(actual.size())};
}

T69BMetric t69b_exact_kv_metric(
    const std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>& actual,
    const std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>& reference) {
    require(actual && reference && actual->position() == reference->position(),
            "T69B KV state extent");
    struct Segment {
        const std::uint16_t* data = nullptr;
        std::size_t values = 0;
    };
    const auto collect = [](const auto& state) {
        std::vector<Segment> result;
        const std::array<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>, 1>
            states{state};
        ninfer::exl3::Exl3ExactHostState::visit_host_allocations(
            states, [&](const void* data, std::size_t bytes) {
                require(bytes % sizeof(std::uint16_t) == 0,
                        "T69B KV allocation alignment");
                result.push_back({static_cast<const std::uint16_t*>(data),
                                  bytes / sizeof(std::uint16_t)});
            }, false);
        return result;
    };
    const auto actual_segments = collect(actual);
    const auto reference_segments = collect(reference);
    const std::uint64_t expected_values =
        static_cast<std::uint64_t>(actual->position()) * 16 * 2 * 1024;
    std::size_t ai = 0, ri = 0, ao = 0, ro = 0;
    std::uint64_t values = 0;
    double error_sq = 0.0, norm_sq = 0.0, max_abs = 0.0;
    while (values < expected_values) {
        require(ai < actual_segments.size() && ri < reference_segments.size(),
                "T69B KV allocation inventory short");
        const auto available = std::min(actual_segments[ai].values - ao,
                                        reference_segments[ri].values - ro);
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
            available, expected_values - values));
        require(count > 0, "T69B KV allocation inventory progress");
        for (std::size_t i = 0; i < count; ++i) {
            const double got = half_to_float(actual_segments[ai].data[ao + i]);
            const double want = half_to_float(reference_segments[ri].data[ro + i]);
            require(std::isfinite(got) && std::isfinite(want),
                    "T69B nonfinite KV state");
            const double difference = got - want;
            error_sq += difference * difference;
            norm_sq += want * want;
            max_abs = std::max(max_abs, std::abs(difference));
        }
        values += count;
        ao += count;
        ro += count;
        if (ao == actual_segments[ai].values) { ++ai; ao = 0; }
        if (ro == reference_segments[ri].values) { ++ri; ro = 0; }
    }
    return {max_abs, norm_sq > 0.0 ? std::sqrt(error_sq / norm_sq) : 0.0,
            values};
}
struct T69BObservedEligible {
    std::uint64_t k6_calls = 0;
    std::uint64_t k7_calls = 0;
    std::uint64_t rows = 0;

    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& observation,
        void* user) {
        auto& self = *static_cast<T69BObservedEligible*>(user);
        if (observation.rows != 1024 || observation.metadata.mcg ||
            !observation.metadata.mul1 || observation.metadata.has_bias)
            return;
        if (observation.metadata.K == 6 &&
            observation.metadata.in_features == 5120 &&
            observation.metadata.out_features == 17408 &&
            (observation.operation == "gate" || observation.operation == "up")) {
            ++self.k6_calls; self.rows += 1024;
        }
        if (observation.metadata.K == 7 &&
            observation.metadata.in_features == 17408 &&
            observation.metadata.out_features == 5120 &&
            observation.operation == "down") {
            ++self.k7_calls; self.rows += 1024;
        }
    }
};

struct T69BRun {
    double construction_ms = 0.0;
    double prompt_ms = 0.0;
    double output_ms = 0.0;
    std::size_t persistent_bytes = 0;
    std::vector<float> logits;
    std::array<std::vector<float>, 64> recurrent;
    std::vector<std::int64_t> tokens;
    std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
    ninfer::exl3::Exl3ReconstructGemmStats candidate{};
    T69BObservedEligible observed{};
};

void t69b_ingest(Exl3TextContext& context,
                  std::span<const std::int64_t> prompt) {
    require(prompt.size() >= 16 && (prompt.size() - 16) % 8 == 0,
            "T69B prompt decomposition");
    context.prefill(prompt.first(16));
    std::size_t first = 16;
    while (prompt.size() - first >= 1024) {
        context.append_exact_prefill_wide(prompt.subspan(first, 1024));
        context.finish_exact_prefill();
        first += 1024;
    }
    while (first < prompt.size()) {
        const std::size_t rows = std::min<std::size_t>(8, prompt.size() - first);
        context.continue_rows(prompt.subspan(first, rows));
        context.finish_exact_continuation();
        first += rows;
    }
    require(context.position() == static_cast<int>(prompt.size()),
            "T69B prompt position");
}

T69BRun t69b_run(Exl3TextModel& target,
                 std::span<const std::int64_t> prompt,
                 bool candidate, bool preserve_state,
                 bool k7_only = false) {
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC",
              candidate && !k7_only ? "1" : "0");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC",
              candidate && k7_only ? "1" : "0");
    const auto construction_start = std::chrono::steady_clock::now();
    auto context = target.create_context(true);
    context->prepare_continuation(8);
    const auto construction_end = std::chrono::steady_clock::now();
    T69BObservedEligible observed;
    context->set_target_projection_observer_for_test(
        T69BObservedEligible::callback, &observed, nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    const auto prompt_start = std::chrono::steady_clock::now();
    t69b_ingest(*context, prompt);
    cuda_check(cudaDeviceSynchronize(), "T69B prompt complete");
    const auto prompt_end = std::chrono::steady_clock::now();
    context->set_target_projection_observer_for_test(nullptr);

    T69BRun result;
    result.construction_ms = std::chrono::duration<double, std::milli>(
        construction_end - construction_start).count();
    result.prompt_ms = std::chrono::duration<double, std::milli>(
        prompt_end - prompt_start).count();
    result.persistent_bytes = context->persistent_bytes();
    result.candidate = context->numeric_prefill_projection_stats();
    result.observed = observed;
    if (candidate) {
        const std::uint64_t expected_k6 = k7_only ? 0 : observed.k6_calls;
        const std::uint64_t expected_rows =
            (expected_k6 + observed.k7_calls) * 1024;
        require(result.candidate.workspace_bytes ==
                    (k7_only ? 251658240u : 301989888u) &&
                result.candidate.k6_calls == expected_k6 &&
                result.candidate.k7_calls == observed.k7_calls &&
                result.candidate.rows == expected_rows &&
                result.candidate.calls == expected_k6 + observed.k7_calls &&
                (k7_only || result.candidate.k6_calls > 0) &&
                result.candidate.k7_calls > 0,
                "T69B executed projection accounting");
    } else {
        require(result.candidate.workspace_bytes == 0 &&
                result.candidate.calls == 0,
                "T69B control candidate isolation");
    }
    if (preserve_state) {
        result.logits = context->logits_host();
        for (int layer = 0; layer < 64; ++layer)
            if ((layer + 1) % 4 != 0)
                result.recurrent[layer] = context->gdn_state_host(layer);
        result.state = context->export_exact_host_state();
    }
    const auto output_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 8; ++i) {
        const auto token = sample_target(*context);
        result.tokens.push_back(token);
        context->decode(token);
    }
    cuda_check(cudaDeviceSynchronize(), "T69B output complete");
    result.output_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - output_start).count();
    return result;
}

void run_projection_reconstruct_route_t69b(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    require(target.max_context() == 4352 && code.size() >= 4096 &&
            prose.size() >= 4096, "T69B fixture/context extent");
    const std::filesystem::path directory = env("NINFER_T69B_OUT");
    require(!directory.empty() && !std::filesystem::exists(directory),
            "T69B output directory must be new");
    std::filesystem::create_directories(directory);
    std::ofstream correctness(directory / "correctness.csv");
    std::ofstream pairs(directory / "pairs.csv");
    require(correctness.good() && pairs.good(), "T69B evidence creation");
    correctness << "fixture,logits_values,logits_max_abs,logits_rel_l2,kv_values,"
                   "kv_max_abs,kv_rel_l2,recurrent_values,recurrent_max_abs,"
                   "recurrent_rel_l2,tokens_exact,control_persistent_bytes,"
                   "candidate_persistent_bytes,workspace_bytes,calls,k6_calls,"
                   "k7_calls,executed_rows\n";
    pairs << "fixture,pair,order,control_construct_ms,candidate_construct_ms,"
             "control_prompt_ms,candidate_prompt_ms,prompt_gain_pct,"
             "control_output_ms,candidate_output_ms,output_rate_gain_pct,"
             "tokens_exact\n";

    const std::array<std::pair<const char*, const std::vector<std::int64_t>*>, 2>
        fixtures{{{"code", &code}, {"prose", &prose}}};
    for (const auto& [name, source] : fixtures) {
        const auto prompt = std::span<const std::int64_t>(source->data(), 4096);
        const auto control = t69b_run(target, prompt, false, true);
        const auto candidate = t69b_run(target, prompt, true, true);
        const auto logits = t69b_float_metric(candidate.logits, control.logits);
        const auto kv = t69b_exact_kv_metric(candidate.state, control.state);
        T69BMetric recurrent{};
        double recurrent_error_sq = 0.0, recurrent_norm_sq = 0.0;
        for (int layer = 0; layer < 64; ++layer) {
            if ((layer + 1) % 4 == 0) continue;
            const auto metric = t69b_float_metric(candidate.recurrent[layer],
                                                   control.recurrent[layer]);
            recurrent.max_abs = std::max(recurrent.max_abs, metric.max_abs);
            for (std::size_t i = 0; i < candidate.recurrent[layer].size(); ++i) {
                const double difference =
                    static_cast<double>(candidate.recurrent[layer][i]) -
                    control.recurrent[layer][i];
                recurrent_error_sq += difference * difference;
                recurrent_norm_sq +=
                    static_cast<double>(control.recurrent[layer][i]) *
                    control.recurrent[layer][i];
            }
            recurrent.values += metric.values;
        }
        recurrent.relative_l2 = recurrent_norm_sq > 0.0
            ? std::sqrt(recurrent_error_sq / recurrent_norm_sq) : 0.0;
        const bool tokens_exact = candidate.tokens == control.tokens;
        require(logits.relative_l2 <= 0.005 && logits.max_abs <= 0.25 &&
                kv.relative_l2 <= 0.005 && kv.max_abs <= 0.25 &&
                recurrent.relative_l2 <= 0.005 && recurrent.max_abs <= 0.25 &&
                tokens_exact,
                std::string("T69B frozen 4K numeric gate ") + name);
        require(candidate.persistent_bytes - control.persistent_bytes == 301989888,
                "T69B one-workspace persistent delta");
        correctness << name << ',' << logits.values << ',' << logits.max_abs << ','
                    << logits.relative_l2 << ',' << kv.values << ',' << kv.max_abs
                    << ',' << kv.relative_l2 << ',' << recurrent.values << ','
                    << recurrent.max_abs << ',' << recurrent.relative_l2 << ','
                    << (tokens_exact ? 1 : 0) << ',' << control.persistent_bytes
                    << ',' << candidate.persistent_bytes << ','
                    << candidate.candidate.workspace_bytes << ','
                    << candidate.candidate.calls << ','
                    << candidate.candidate.k6_calls << ','
                    << candidate.candidate.k7_calls << ','
                    << candidate.candidate.rows << '\n';

        for (int pair = 0; pair < 4; ++pair) {
            T69BRun timed_control, timed_candidate;
            if ((pair & 1) == 0) {
                timed_control = t69b_run(target, prompt, false, false);
                timed_candidate = t69b_run(target, prompt, true, false);
            } else {
                timed_candidate = t69b_run(target, prompt, true, false);
                timed_control = t69b_run(target, prompt, false, false);
            }
            const bool pair_tokens = timed_control.tokens == timed_candidate.tokens;
            require(pair_tokens, "T69B timed token identity");
            pairs << name << ',' << pair << ','
                  << ((pair & 1) == 0 ? "AB" : "BA") << ','
                  << timed_control.construction_ms << ','
                  << timed_candidate.construction_ms << ','
                  << timed_control.prompt_ms << ',' << timed_candidate.prompt_ms
                  << ',' << (timed_control.prompt_ms - timed_candidate.prompt_ms) *
                                  100.0 / timed_control.prompt_ms
                  << ',' << timed_control.output_ms << ','
                  << timed_candidate.output_ms << ','
                  << (timed_control.output_ms - timed_candidate.output_ms) *
                         100.0 / timed_control.output_ms
                  << ",1\n";
            pairs.flush();
        }
        correctness.flush();
    }

    // Cancel a completed partial prompt, explicitly reset the candidate-owned
    // context, and prove that an unrelated noncandidate-width request is clean.
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC", "1");
    auto cancelled = target.create_context(true);
    cancelled->prepare_continuation(8);
    const std::string contract = "t69b;target-only;exact-host;wide1024";
    cancelled->bind_request_compatibility(contract);
    cancelled->prefill(std::span<const std::int64_t>(code.data(), 16));
    cancelled->append_exact_prefill_wide(
        std::span<const std::int64_t>(code.data() + 16, 1024));
    cancelled->finish_exact_prefill();
    const auto persistent = cancelled->persistent_bytes();
    const auto reset = cancelled->reset_for_request(contract);
    require(cancelled->position() == 0 &&
            cancelled->persistent_bytes() == persistent && reset.generation == 1,
            "T69B cancellation reset ownership");
    t69b_ingest(*cancelled,
        std::span<const std::int64_t>(prose.data(), 96));
    std::vector<std::int64_t> reused_tokens;
    for (int i = 0; i < 8; ++i) {
        const auto token = sample_target(*cancelled);
        reused_tokens.push_back(token); cancelled->decode(token);
    }
    const auto fresh = t69b_run(target,
        std::span<const std::int64_t>(prose.data(), 96), false, false);
    require(reused_tokens == fresh.tokens,
            "T69B cancellation contaminated unrelated request");

    std::cout << "T69B_RECONSTRUCT_ROUTE PASS fixtures=2 prefix=4096 pairs=8 "
                 "lane=NUMERIC_CANDIDATE workspace_bytes=301989888 "
                 "kv_logits_recurrent_gates=1 tokens_exact=1 cancellation=1 "
                 "production_changed=0"
              << std::endl;
}

void run_projection_reconstruct_route_t69c(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    require(target.max_context() == 4352 && code.size() >= 4096 &&
            prose.size() >= 4096, "T69C fixture/context extent");
    const std::filesystem::path directory = env("NINFER_T69C_OUT");
    require(!directory.empty() && !std::filesystem::exists(directory),
            "T69C output directory must be new");
    std::filesystem::create_directories(directory);
    std::ofstream correctness(directory / "correctness.csv");
    std::ofstream pairs(directory / "pairs.csv");
    require(correctness.good() && pairs.good(), "T69C evidence creation");
    correctness << "fixture,logits_values,logits_max_abs,logits_rel_l2,kv_values,"
                   "kv_max_abs,kv_rel_l2,recurrent_values,recurrent_max_abs,"
                   "recurrent_rel_l2,tokens_exact,control_persistent_bytes,"
                   "candidate_persistent_bytes,workspace_bytes,calls,k6_calls,"
                   "k7_calls,executed_rows,numeric_pass,persistent_delta_pass\n";
    pairs << "fixture,pair,order,control_construct_ms,candidate_construct_ms,"
             "control_prompt_ms,candidate_prompt_ms,prompt_gain_pct,"
             "control_output_ms,candidate_output_ms,output_rate_gain_pct,"
             "tokens_exact\n";

    const std::array<std::pair<const char*, const std::vector<std::int64_t>*>, 2>
        fixtures{{{"code", &code}, {"prose", &prose}}};
    for (const auto& [name, source] : fixtures) {
        const auto prompt = std::span<const std::int64_t>(source->data(), 4096);
        const auto control = t69b_run(target, prompt, false, true, true);
        const auto candidate = t69b_run(target, prompt, true, true, true);
        const auto logits = t69b_float_metric(candidate.logits, control.logits);
        const auto kv = t69b_exact_kv_metric(candidate.state, control.state);
        T69BMetric recurrent{};
        double recurrent_error_sq = 0.0, recurrent_norm_sq = 0.0;
        for (int layer = 0; layer < 64; ++layer) {
            if ((layer + 1) % 4 == 0) continue;
            const auto metric = t69b_float_metric(candidate.recurrent[layer],
                                                   control.recurrent[layer]);
            recurrent.max_abs = std::max(recurrent.max_abs, metric.max_abs);
            for (std::size_t i = 0; i < candidate.recurrent[layer].size(); ++i) {
                const double difference =
                    static_cast<double>(candidate.recurrent[layer][i]) -
                    control.recurrent[layer][i];
                recurrent_error_sq += difference * difference;
                recurrent_norm_sq +=
                    static_cast<double>(control.recurrent[layer][i]) *
                    control.recurrent[layer][i];
            }
            recurrent.values += metric.values;
        }
        recurrent.relative_l2 = recurrent_norm_sq > 0.0
            ? std::sqrt(recurrent_error_sq / recurrent_norm_sq) : 0.0;
        const bool tokens_exact = candidate.tokens == control.tokens;
        const bool numeric_pass =
            logits.relative_l2 <= 0.005 && logits.max_abs <= 0.25 &&
            kv.relative_l2 <= 0.005 && kv.max_abs <= 0.25 &&
            recurrent.relative_l2 <= 0.005 && recurrent.max_abs <= 0.25 &&
            tokens_exact;
        const bool persistent_delta_pass =
            candidate.persistent_bytes >= control.persistent_bytes &&
            candidate.persistent_bytes - control.persistent_bytes == 251658240;
        correctness << name << ',' << logits.values << ',' << logits.max_abs << ','
                    << logits.relative_l2 << ',' << kv.values << ',' << kv.max_abs
                    << ',' << kv.relative_l2 << ',' << recurrent.values << ','
                    << recurrent.max_abs << ',' << recurrent.relative_l2 << ','
                    << (tokens_exact ? 1 : 0) << ',' << control.persistent_bytes
                    << ',' << candidate.persistent_bytes << ','
                    << candidate.candidate.workspace_bytes << ','
                    << candidate.candidate.calls << ','
                    << candidate.candidate.k6_calls << ','
                    << candidate.candidate.k7_calls << ','
                    << candidate.candidate.rows << ','
                    << (numeric_pass ? 1 : 0) << ','
                    << (persistent_delta_pass ? 1 : 0) << '\n';
        correctness.flush();
        require(numeric_pass,
                std::string("T69C frozen 4K numeric gate ") + name);
        require(persistent_delta_pass,
                "T69C one-K7-workspace persistent delta");

        for (int pair = 0; pair < 4; ++pair) {
            T69BRun timed_control, timed_candidate;
            if ((pair & 1) == 0) {
                timed_control = t69b_run(target, prompt, false, false, true);
                timed_candidate = t69b_run(target, prompt, true, false, true);
            } else {
                timed_candidate = t69b_run(target, prompt, true, false, true);
                timed_control = t69b_run(target, prompt, false, false, true);
            }
            const bool pair_tokens = timed_control.tokens == timed_candidate.tokens;
            require(pair_tokens, "T69C timed token identity");
            pairs << name << ',' << pair << ','
                  << ((pair & 1) == 0 ? "AB" : "BA") << ','
                  << timed_control.construction_ms << ','
                  << timed_candidate.construction_ms << ','
                  << timed_control.prompt_ms << ',' << timed_candidate.prompt_ms
                  << ',' << (timed_control.prompt_ms - timed_candidate.prompt_ms) *
                                  100.0 / timed_control.prompt_ms
                  << ',' << timed_control.output_ms << ','
                  << timed_candidate.output_ms << ','
                  << (timed_control.output_ms - timed_candidate.output_ms) *
                         100.0 / timed_control.output_ms
                  << ",1\n";
            pairs.flush();
        }
    }

    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC", "0");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC", "1");
    auto cancelled = target.create_context(true);
    cancelled->prepare_continuation(8);
    const std::string contract = "t69c;target-only;exact-host;wide1024;k7-only";
    cancelled->bind_request_compatibility(contract);
    cancelled->prefill(std::span<const std::int64_t>(code.data(), 16));
    cancelled->append_exact_prefill_wide(
        std::span<const std::int64_t>(code.data() + 16, 1024));
    cancelled->finish_exact_prefill();
    const auto persistent = cancelled->persistent_bytes();
    const auto reset = cancelled->reset_for_request(contract);
    require(cancelled->position() == 0 &&
            cancelled->persistent_bytes() == persistent && reset.generation == 1,
            "T69C cancellation reset ownership");
    t69b_ingest(*cancelled,
        std::span<const std::int64_t>(prose.data(), 96));
    std::vector<std::int64_t> reused_tokens;
    for (int i = 0; i < 8; ++i) {
        const auto token = sample_target(*cancelled);
        reused_tokens.push_back(token);
        cancelled->decode(token);
    }
    const auto fresh = t69b_run(target,
        std::span<const std::int64_t>(prose.data(), 96), false, false, true);
    require(reused_tokens == fresh.tokens,
            "T69C cancellation contaminated unrelated request");

    std::cout << "T69C_K7_RECONSTRUCT_ROUTE PASS fixtures=2 prefix=4096 pairs=8 "
                 "lane=NUMERIC_CANDIDATE workspace_bytes=251658240 "
                 "kv_logits_recurrent_gates=1 tokens_exact=1 cancellation=1 "
                 "production_changed=0"
              << std::endl;
}
