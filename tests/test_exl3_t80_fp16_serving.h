#pragma once
#include "test_exl3_host_residency.h"

struct T80Fp16ServingArm {
    std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> request;
    ninfer::exl3::Exl3VeriCacheServingPrefixCache::Metrics preparation;
    std::vector<std::int64_t> published;
    std::vector<double> publication_ms;
    double preparation_ms = 0.0;
    double output_ms = 0.0;
    double p95_interval_ms = 0.0;
    double residency_update_ms = 0.0;
    std::uint64_t verification_rows = 0;
    std::uint64_t executed_rows = 0;
    std::uint64_t replay_rows = 0;
    std::uint64_t committed_rows = 0;
    std::uint64_t native_invocations = 0;
    std::uint64_t root_restores = 0;
    std::size_t ready_free_bytes = 0;
    std::size_t end_free_bytes = 0;
    Exl3HostResidency ready_residency;
    Exl3HostResidency end_residency;
};

double t80_p95(std::vector<double> values) {
    require(!values.empty(), "T80 p95 input");
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

void run_t80_fp16_serving(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    constexpr std::size_t kPrompt = 4096;
    constexpr std::size_t kPrefix = 3968;
    constexpr std::size_t kTail = 128;
    constexpr int kOutputs = 64;
    require(target.max_context() >= 4352 && code.size() >= kPrompt &&
                prose.size() >= kTail,
            "T80 serving fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T80 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib = std::stoull(env("NINFER_T80_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T80_CACHE_MIB"));
    require(reserve_gib >= 4 && cache_mib >= 512,
            "T80 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullTotalPhys > reserve &&
                memory.ullAvailPhys > reserve,
            "T80 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0,
                "T80 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T80_NAMESPACE"), env("NINFER_T80_ARTIFACT_ID"),
        env("NINFER_T80_TOKENIZER_ID"), env("NINFER_T80_CONFIGURATION_ID"),
        env("NINFER_T80_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{4, 64, cache_budget, reserve}, identity);
    ninfer::exl3::Exl3HostResidentSet residency(
        memory.ullTotalPhys - reserve, reserve);
    auto cached_context = target.create_context(true);
    auto fresh_context = target.create_context(true);
    cached_context->prepare_continuation(8);
    fresh_context->prepare_continuation(8);

    std::vector<std::int64_t> common(code.begin(), code.begin() + kPrefix);
    std::array<std::vector<std::int64_t>, 2> prompts{
        std::vector<std::int64_t>(code.begin(), code.begin() + kPrompt),
        common};
    prompts[1].insert(prompts[1].end(), prose.begin(), prose.begin() + kTail);
    require(!std::equal(prompts[0].begin() + kPrefix, prompts[0].end(),
                        prompts[1].begin() + kPrefix),
            "T80 distinct tails");

    struct PinResult {
        ninfer::exl3::Exl3HostResidentSet::Stats update;
        Exl3HostResidency observed;
    };
    const auto pin = [&](std::shared_ptr<const Request> active,
                         bool observe) {
        auto roots = cache.roots();
        if (active) roots.push_back(std::move(active));
        ninfer::exl3::Exl3HostResidentSet::Snapshot snapshot;
        for (const auto& root : roots) snapshot.owners.push_back(root);
        Request::visit_host_allocations(roots,
            [&](const void* data, std::size_t bytes) { snapshot.add(data, bytes); },
            false);
        const auto update = residency.replace(std::move(snapshot));
        Exl3HostResidency observed;
        if (observe) {
            Exl3HostResidencyProbe probe;
            Request::visit_host_allocations(roots,
                [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
                false);
            observed = probe.measure();
            require(observed.resident_tensor_bytes ==
                        observed.allocated_union_bytes &&
                        observed.locked_tensor_bytes ==
                            observed.allocated_union_bytes,
                    "T80 authoritative payload not resident and locked");
        }
        return PinResult{update, observed};
    };

    const auto fill_started = std::chrono::steady_clock::now();
    auto fill = cache.prepare(*cached_context, common, kPrefix,
                              available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T80 cache fill completion");
    const double fill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - fill_started).count();
    require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                fill.metrics.executed_prompt_tokens == kPrefix,
            "T80 cache fill gate");
    fill.request.reset();
    const auto fill_lock = pin({}, true);

    std::ofstream arms(output / "arms.csv");
    std::ofstream publications(output / "publications.csv");
    std::ofstream cancellation(output / "cancellation.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(arms.good() && publications.good() && cancellation.good() &&
                memory_out.good(),
            "T80 evidence open");
    arms << "fixture,rep,order,arm,preparation_ms,prompt_tokens,"
            "reused_prompt_tokens,executed_prompt_tokens,output_ms,"
            "authoritative_output_s,p95_interval_ms,residency_update_ms,"
            "verification_rows,executed_rows,replay_rows,committed_rows,"
            "native_invocations,root_restores,ready_free_bytes,end_free_bytes,"
            "ready_resident_bytes,ready_locked_bytes,end_resident_bytes,"
            "end_locked_bytes,exact\n";
    publications << "fixture,rep,arm,ordinal,token_id,publication_ms\n";
    cancellation << "fixture,cancel_after,canceled_committed_rows,"
                    "subsequent_cache_hit,full_stream_exact,final_state_exact,"
                    "cache_entries,pass\n";
    memory_out << "phase,cache_payload_bytes,cache_identity_bytes,"
                  "cache_accounted_bytes,resident_payload_bytes,"
                  "locked_payload_bytes,resident_page_bytes,new_locked_bytes,"
                  "unlocked_bytes,physical_reserve_bytes,device_free_bytes\n";
    std::size_t fill_free = 0, total_device = 0;
    cuda_check(cudaMemGetInfo(&fill_free, &total_device), "T80 fill device memory");
    memory_out << "filled," << fill.metrics.cache_storage.payload_allocated_bytes
        << ',' << fill.metrics.cache_storage.identity_allocated_bytes << ','
        << fill.metrics.cache_storage.accounted_bytes << ','
        << fill_lock.observed.resident_tensor_bytes << ','
        << fill_lock.observed.locked_tensor_bytes << ',' << fill_lock.update.page_bytes
        << ',' << fill_lock.update.new_locked_bytes << ','
        << fill_lock.update.unlocked_bytes << ',' << reserve << ',' << fill_free
        << '\n';

    const auto run_arm = [&](int fixture, bool cached, int outputs) {
        T80Fp16ServingArm result;
        auto& context = cached ? *cached_context : *fresh_context;
        const auto preparation_started = std::chrono::steady_clock::now();
        if (cached) {
            auto prepared = cache.prepare(context, prompts[fixture], kPrefix,
                                          available(), 1024);
            result.request = std::move(prepared.request);
            result.preparation = prepared.metrics;
        } else {
            result.request = Request::initialize(context, prompts[fixture], 1024);
            result.preparation.prompt_tokens = kPrompt;
            result.preparation.executed_prompt_tokens = kPrompt;
        }
        cuda_check(cudaDeviceSynchronize(), "T80 request preparation completion");
        result.preparation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preparation_started).count();
        const auto ready_pin = pin(result.request, true);
        result.ready_residency = ready_pin.observed;
        cuda_check(cudaMemGetInfo(&result.ready_free_bytes, &total_device),
                   "T80 ready device memory");
        require(result.ready_free_bytes >= (1ULL << 30),
                "T80 ready device reserve");
        auto pending = sample_target(context);
        const auto output_started = std::chrono::steady_clock::now();
        for (int ordinal = 0; ordinal < outputs; ++ordinal) {
            const std::array<std::int64_t, 1> tentative{pending};
            auto [updated, verified] = result.request->verify(context, tentative);
            require(verified.committed_tokens.size() == 1 &&
                        verified.committed_tokens[0] == pending &&
                        verified.accepted == 1 && !verified.rejected &&
                        !verified.stopped,
                    "T80 width-one authoritative decision");
            result.verification_rows += verified.verification_rows;
            result.executed_rows += verified.executed_rows;
            result.replay_rows += verified.replay_rows;
            result.committed_rows += verified.committed_tokens.size();
            result.native_invocations += verified.native_invocations;
            result.root_restores += verified.root_restores;
            result.request = std::move(updated);
            const auto lock_update = pin(result.request, false);
            result.residency_update_ms += lock_update.update.elapsed_ms;
            result.published.push_back(pending);
            result.publication_ms.push_back(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - output_started).count());
            if (ordinal + 1 < outputs) pending = sample_target(context);
        }
        result.output_ms = result.publication_ms.back();
        std::vector<double> intervals;
        intervals.reserve(result.publication_ms.size());
        double previous = 0.0;
        for (const double stamp : result.publication_ms) {
            intervals.push_back(stamp - previous);
            previous = stamp;
        }
        result.p95_interval_ms = t80_p95(std::move(intervals));
        const auto end_pin = pin(result.request, true);
        result.end_residency = end_pin.observed;
        cuda_check(cudaMemGetInfo(&result.end_free_bytes, &total_device),
                   "T80 end device memory");
        require(result.verification_rows == static_cast<std::uint64_t>(outputs) &&
                    result.executed_rows == static_cast<std::uint64_t>(outputs) &&
                    result.committed_rows == static_cast<std::uint64_t>(outputs) &&
                    result.replay_rows == 0 && result.native_invocations ==
                        static_cast<std::uint64_t>(outputs) &&
                    result.root_restores == 0 &&
                    result.end_free_bytes >= (1ULL << 30),
                "T80 authoritative row/replay/memory gate");
        return result;
    };

    std::array<std::vector<double>, 2> fresh_prepare, cached_prepare;
    std::array<std::vector<double>, 2> fresh_rate, cached_rate;
    std::array<T80Fp16ServingArm, 2> cancellation_controls;
    const char* fixture_names[2]{"code_tail", "prose_tail"};
    for (int fixture = 0; fixture < 2; ++fixture) {
        for (int rep = 0; rep < 3; ++rep) {
            T80Fp16ServingArm fresh, cached;
            if (rep == 1) {
                cached = run_arm(fixture, true, kOutputs);
                fresh = run_arm(fixture, false, kOutputs);
            } else {
                fresh = run_arm(fixture, false, kOutputs);
                cached = run_arm(fixture, true, kOutputs);
            }
            require(fresh.published == cached.published &&
                        fresh.request->token_suffix() ==
                            cached.request->token_suffix() &&
                        fresh.request->same_taps(*cached.request) &&
                        fresh.request->state()->same_payload(
                            *cached.request->state()),
                    "T80 fresh/cached authoritative stream mismatch");
            fresh_prepare[fixture].push_back(fresh.preparation_ms);
            cached_prepare[fixture].push_back(cached.preparation_ms);
            fresh_rate[fixture].push_back(
                kOutputs * 1000.0 / fresh.output_ms);
            cached_rate[fixture].push_back(
                kOutputs * 1000.0 / cached.output_ms);
            const auto write_arm = [&](const char* name,
                                       const T80Fp16ServingArm& arm) {
                arms << fixture_names[fixture] << ',' << rep << ','
                    << (rep == 1 ? "cached-fresh" : "fresh-cached") << ','
                    << name << ',' << arm.preparation_ms << ','
                    << arm.preparation.prompt_tokens << ','
                    << arm.preparation.reused_prompt_tokens << ','
                    << arm.preparation.executed_prompt_tokens << ','
                    << arm.output_ms << ',' << kOutputs * 1000.0 / arm.output_ms
                    << ',' << arm.p95_interval_ms << ','
                    << arm.residency_update_ms << ',' << arm.verification_rows
                    << ',' << arm.executed_rows << ',' << arm.replay_rows << ','
                    << arm.committed_rows << ',' << arm.native_invocations << ','
                    << arm.root_restores << ',' << arm.ready_free_bytes << ','
                    << arm.end_free_bytes << ','
                    << arm.ready_residency.resident_tensor_bytes << ','
                    << arm.ready_residency.locked_tensor_bytes << ','
                    << arm.end_residency.resident_tensor_bytes << ','
                    << arm.end_residency.locked_tensor_bytes << ",1\n";
                for (std::size_t ordinal = 0;
                     ordinal < arm.published.size(); ++ordinal)
                    publications << fixture_names[fixture] << ',' << rep << ','
                        << name << ',' << ordinal << ',' << arm.published[ordinal]
                        << ',' << arm.publication_ms[ordinal] << '\n';
            };
            write_arm("fresh", fresh);
            write_arm("cached", cached);
            if (rep == 2) cancellation_controls[fixture] = std::move(cached);
        }
    }

    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    for (int fixture = 0; fixture < 2; ++fixture) {
        int preparation_positive = 0;
        std::vector<double> rate_gains;
        for (int rep = 0; rep < 3; ++rep) {
            if (cached_prepare[fixture][rep] < fresh_prepare[fixture][rep])
                ++preparation_positive;
            const double gain = (cached_rate[fixture][rep] /
                fresh_rate[fixture][rep] - 1.0) * 100.0;
            rate_gains.push_back(gain);
            require(gain >= -5.0, "T80 individual cached output-rate guard");
        }
        require(preparation_positive == 3 &&
                    median(cached_prepare[fixture]) < median(fresh_prepare[fixture]) &&
                    median(rate_gains) >= -2.0,
                "T80 preparation/output-rate gate");
    }

    for (int fixture = 0; fixture < 2; ++fixture) {
        auto partial = run_arm(fixture, true, 16);
        require(partial.committed_rows == 16 && partial.published.size() == 16,
                "T80 cancellation boundary");
        partial.request.reset();
        const auto full = run_arm(fixture, true, kOutputs);
        const bool full_stream_exact =
            full.published == cancellation_controls[fixture].published;
        const bool final_state_exact =
            full.request->token_suffix() ==
                cancellation_controls[fixture].request->token_suffix() &&
            full.request->same_taps(*cancellation_controls[fixture].request) &&
            full.request->state()->same_payload(
                *cancellation_controls[fixture].request->state());
        const auto storage = cache.storage_stats();
        const bool pass = full.preparation.cache_hit && full_stream_exact &&
            final_state_exact && storage.entries == 1 &&
            storage.accounted_bytes <= cache_budget;
        cancellation << fixture_names[fixture] << ",16,16,"
            << full.preparation.cache_hit << ',' << full_stream_exact << ','
            << final_state_exact << ',' << storage.entries << ',' << pass << '\n';
        require(pass, "T80 cancellation/cache integrity gate");
    }

    const auto final_pin = pin({}, true);
    const auto final_storage = cache.storage_stats();
    std::size_t final_free = 0;
    cuda_check(cudaMemGetInfo(&final_free, &total_device),
               "T80 final device memory");
    require(final_storage.entries == 1 &&
                final_storage.accounted_bytes <= cache_budget &&
                final_free >= (1ULL << 30),
            "T80 final cache/memory gate");
    memory_out << "final_cache," << final_storage.payload_allocated_bytes << ','
        << final_storage.identity_allocated_bytes << ','
        << final_storage.accounted_bytes << ','
        << final_pin.observed.resident_tensor_bytes << ','
        << final_pin.observed.locked_tensor_bytes << ','
        << final_pin.update.page_bytes << ',' << final_pin.update.new_locked_bytes
        << ',' << final_pin.update.unlocked_bytes << ',' << reserve << ','
        << final_free << '\n';
    arms.flush(); publications.flush(); cancellation.flush(); memory_out.flush();
    require(arms.good() && publications.good() && cancellation.good() &&
                memory_out.good(),
            "T80 evidence flush");
    residency.close();
    std::cout << "T80_FP16_SERVING PASS fixtures=2 pairs=3 outputs=64"
              << " exact=1 replay_rows=0 root_restores=0 cancellation=2"
              << " resident_locked=1 draft_loaded=0 cache_fill_ms=" << fill_ms
              << " output=" << output.string() << std::endl;
}
