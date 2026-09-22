#pragma once
#include "test_exl3_host_residency.h"
#include <atomic>
#include <thread>

struct T82Fp16LaneResult {
    std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> request;
    std::vector<std::int64_t> published;
    std::vector<double> publication_ms;
    std::uint64_t verification_rows = 0;
    std::uint64_t executed_rows = 0;
    std::uint64_t replay_rows = 0;
    std::uint64_t committed_rows = 0;
    std::uint64_t native_invocations = 0;
    std::uint64_t root_restores = 0;
    double p95_interval_ms = 0.0;
};

struct T82Fp16ArmResult {
    std::array<T82Fp16LaneResult, 2> lanes;
    double output_ms = 0.0;
    double residency_update_ms = 0.0;
    std::size_t ready_free_bytes = 0;
    std::size_t end_free_bytes = 0;
    std::size_t ready_owned_bytes = 0;
    std::size_t end_owned_bytes = 0;
    Exl3HostResidency ready_residency;
    Exl3HostResidency end_residency;
};

void run_t82_fp16_c2_serving(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    constexpr std::size_t kPrompt = 4096;
    constexpr std::size_t kPrefix = 3968;
    constexpr std::size_t kTail = 128;
    constexpr int kOutputs = 32;
    constexpr int kWarmups = 2;
    constexpr int kMeasured = 4;
    require(target.max_context() >= 4352 && code.size() >= kPrompt &&
                prose.size() >= kTail,
            "T82 serving fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T82 output must be new");
    std::filesystem::create_directories(output);

    const auto reserve_gib = std::stoull(env("NINFER_T82_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T82_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024,
            "T82 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullTotalPhys > reserve &&
                memory.ullAvailPhys > reserve,
            "T82 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0,
                "T82 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T82_NAMESPACE"), env("NINFER_T82_ARTIFACT_ID"),
        env("NINFER_T82_TOKENIZER_ID"), env("NINFER_T82_CONFIGURATION_ID"),
        env("NINFER_T82_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{4, 64, cache_budget, reserve}, identity);
    ninfer::exl3::Exl3HostResidentSet residency(
        memory.ullTotalPhys - reserve, reserve);
    auto context_a = target.create_context(true);
    auto context_b = target.create_context(true);
    context_a->prepare_continuation(8);
    context_b->prepare_continuation(8);
    Exl3TextContext* contexts[2]{context_a.get(), context_b.get()};
    TargetGraphC2Stream streams[2];

    std::vector<std::int64_t> common(code.begin(), code.begin() + kPrefix);
    std::array<std::vector<std::int64_t>, 2> prompts{
        std::vector<std::int64_t>(code.begin(), code.begin() + kPrompt), common};
    prompts[1].insert(prompts[1].end(), prose.begin(), prose.begin() + kTail);
    require(!std::equal(prompts[0].begin() + kPrefix, prompts[0].end(),
                        prompts[1].begin() + kPrefix),
            "T82 distinct tails");

    struct PinResult {
        ninfer::exl3::Exl3HostResidentSet::Stats update;
        Exl3HostResidency observed;
    };
    const auto pin = [&](const std::array<std::shared_ptr<const Request>, 2>& active,
                         bool observe) {
        auto roots = cache.roots();
        for (const auto& root : active) if (root) roots.push_back(root);
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
                    observed.locked_tensor_bytes == observed.allocated_union_bytes,
                    "T82 authoritative payload not resident and locked");
        }
        return PinResult{update, observed};
    };
    const auto owned_bytes = [&] {
        std::size_t bytes = 0;
        for (auto* context : contexts)
            bytes += context->persistent_bytes() + context->continuation_bytes();
        return bytes;
    };

    auto fill = cache.prepare(*context_a, common, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T82 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                fill.metrics.executed_prompt_tokens == kPrefix,
            "T82 cache fill gate");
    fill.request.reset();
    std::array<std::shared_ptr<const Request>, 2> initial;
    std::array<Cache::Metrics, 2> preparation;
    std::array<double, 2> preparation_ms{};
    for (int lane = 0; lane < 2; ++lane) {
        const auto started = std::chrono::steady_clock::now();
        auto prepared = cache.prepare(*contexts[lane], prompts[lane], kPrefix,
                                      available(), 1024);
        cuda_check(cudaDeviceSynchronize(), "T82 request preparation completion");
        preparation_ms[lane] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        preparation[lane] = prepared.metrics;
        initial[lane] = std::move(prepared.request);
        require(preparation[lane].cache_hit &&
                    preparation[lane].reused_prompt_tokens == kPrefix &&
                    preparation[lane].executed_prompt_tokens == kTail,
                "T82 cached preparation accounting");
    }
    const auto initial_lock = pin(initial, true);

    std::ofstream arms(output / "arms.csv");
    std::ofstream publications(output / "publications.csv");
    std::ofstream cancellation(output / "cancellation.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(arms.good() && publications.good() && cancellation.good() &&
                memory_out.good(),
            "T82 evidence open");
    arms << "rep,warmup,order,arm,lane,preparation_ms,reused_prompt_tokens,"
            "executed_prompt_tokens,output_ms,aggregate_output_s,"
            "lane_output_s,p95_interval_ms,final_release_ms,"
            "residency_update_ms,verification_rows,executed_rows,replay_rows,"
            "committed_rows,native_invocations,root_restores,ready_free_bytes,"
            "end_free_bytes,ready_owned_bytes,end_owned_bytes,"
            "ready_resident_bytes,ready_locked_bytes,end_resident_bytes,"
            "end_locked_bytes,exact\n";
    publications << "rep,warmup,arm,lane,ordinal,token_id,publication_ms\n";
    cancellation << "survivor,canceled,verified_rows_discarded,"
                    "survivor_publications,canceled_publications,"
                    "survivor_exact,canceled_root_exact,transactions_closed,pass\n";
    memory_out << "phase,cache_payload_bytes,cache_identity_bytes,"
                  "cache_accounted_bytes,resident_payload_bytes,"
                  "locked_payload_bytes,resident_page_bytes,new_locked_bytes,"
                  "unlocked_bytes,physical_reserve_bytes,device_free_bytes\n";
    std::size_t total_device = 0, initial_free = 0;
    cuda_check(cudaMemGetInfo(&initial_free, &total_device),
               "T82 initial device memory");
    memory_out << "active_c2," << cache.storage_stats().payload_allocated_bytes
        << ',' << cache.storage_stats().identity_allocated_bytes << ','
        << cache.storage_stats().accounted_bytes << ','
        << initial_lock.observed.resident_tensor_bytes << ','
        << initial_lock.observed.locked_tensor_bytes << ','
        << initial_lock.update.page_bytes << ',' << initial_lock.update.new_locked_bytes
        << ',' << initial_lock.update.unlocked_bytes << ',' << reserve << ','
        << initial_free << '\n';

    const auto run_arm = [&](bool concurrent, bool b_first) {
        T82Fp16ArmResult result;
        std::array<std::shared_ptr<const Request>, 2> current = initial;
        std::array<std::int64_t, 2> pending{};
        for (int lane = 0; lane < 2; ++lane) {
            contexts[lane]->restore_exact_host_state(
                *initial[lane]->state(), streams[lane].value);
            cuda_check(cudaStreamSynchronize(streams[lane].value),
                       "T82 initial root restore");
            pending[lane] = sample_target(*contexts[lane], streams[lane].value);
        }
        const auto ready_pin = pin(current, true);
        result.ready_residency = ready_pin.observed;
        result.ready_owned_bytes = owned_bytes();
        cuda_check(cudaMemGetInfo(&result.ready_free_bytes, &total_device),
                   "T82 ready device memory");
        require(result.ready_free_bytes >= (1ULL << 30),
                "T82 ready device reserve");
        const auto output_started = std::chrono::steady_clock::now();
        const auto verify_lane = [&](int lane) {
            const std::array<std::int64_t, 1> tentative{pending[lane]};
            auto [updated, verified] = current[lane]->verify(
                *contexts[lane], tentative, {}, streams[lane].value);
            require(verified.committed_tokens.size() == 1 &&
                        verified.committed_tokens[0] == pending[lane] &&
                        verified.accepted == 1 && !verified.rejected &&
                        !verified.stopped,
                    "T82 width-one authoritative decision");
            current[lane] = std::move(updated);
            auto& trace = result.lanes[lane];
            trace.verification_rows += verified.verification_rows;
            trace.executed_rows += verified.executed_rows;
            trace.replay_rows += verified.replay_rows;
            trace.committed_rows += verified.committed_tokens.size();
            trace.native_invocations += verified.native_invocations;
            trace.root_restores += verified.root_restores;
        };
        for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
            if (concurrent) {
                std::array<std::exception_ptr, 2> errors;
                std::atomic<int> ready{0};
                std::atomic<bool> go{false};
                std::thread workers[2];
                for (int lane = 0; lane < 2; ++lane) {
                    workers[lane] = std::thread([&, lane] {
                        ready.fetch_add(1, std::memory_order_release);
                        while (!go.load(std::memory_order_acquire))
                            std::this_thread::yield();
                        try { verify_lane(lane); }
                        catch (...) { errors[lane] = std::current_exception(); }
                    });
                }
                while (ready.load(std::memory_order_acquire) != 2)
                    std::this_thread::yield();
                go.store(true, std::memory_order_release);
                for (auto& worker : workers) worker.join();
                for (const auto& error : errors) if (error) std::rethrow_exception(error);
                const auto update = pin(current, false);
                result.residency_update_ms += update.update.elapsed_ms;
                const double stamp = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - output_started).count();
                for (int lane = 0; lane < 2; ++lane) {
                    result.lanes[lane].published.push_back(pending[lane]);
                    result.lanes[lane].publication_ms.push_back(stamp);
                }
                if (ordinal + 1 < kOutputs) {
                    std::array<std::exception_ptr, 2> sample_errors;
                    std::thread sample_workers[2];
                    for (int lane = 0; lane < 2; ++lane)
                        sample_workers[lane] = std::thread([&, lane] {
                            try { pending[lane] = sample_target(
                                *contexts[lane], streams[lane].value); }
                            catch (...) { sample_errors[lane] = std::current_exception(); }
                        });
                    for (auto& worker : sample_workers) worker.join();
                    for (const auto& error : sample_errors)
                        if (error) std::rethrow_exception(error);
                }
            } else {
                const int order[2]{b_first ? 1 : 0, b_first ? 0 : 1};
                for (const int lane : order) {
                    verify_lane(lane);
                    const auto update = pin(current, false);
                    result.residency_update_ms += update.update.elapsed_ms;
                    result.lanes[lane].published.push_back(pending[lane]);
                    result.lanes[lane].publication_ms.push_back(
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - output_started).count());
                    if (ordinal + 1 < kOutputs)
                        pending[lane] = sample_target(
                            *contexts[lane], streams[lane].value);
                }
            }
        }
        result.output_ms = std::max(result.lanes[0].publication_ms.back(),
                                    result.lanes[1].publication_ms.back());
        for (int lane = 0; lane < 2; ++lane) {
            auto& trace = result.lanes[lane];
            trace.request = current[lane];
            std::vector<double> intervals;
            double previous = 0.0;
            for (const double stamp : trace.publication_ms) {
                intervals.push_back(stamp - previous);
                previous = stamp;
            }
            trace.p95_interval_ms = t80_p95(std::move(intervals));
            require(trace.verification_rows == kOutputs &&
                        trace.executed_rows == kOutputs &&
                        trace.committed_rows == kOutputs &&
                        trace.native_invocations == kOutputs &&
                        trace.replay_rows == 0 && trace.root_restores == 0,
                    "T82 authoritative work accounting");
        }
        const auto end_pin = pin(current, true);
        result.end_residency = end_pin.observed;
        result.end_owned_bytes = owned_bytes();
        cuda_check(cudaMemGetInfo(&result.end_free_bytes, &total_device),
                   "T82 end device memory");
        require(result.ready_owned_bytes == result.end_owned_bytes &&
                    result.end_free_bytes >= (1ULL << 30),
                "T82 logical/device memory gate");
        return result;
    };

    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const auto middle = values.size() / 2;
        return values.size() & 1 ? values[middle] :
            (values[middle - 1] + values[middle]) / 2.0;
    };
    std::vector<double> gains;
    std::vector<double> c1_rates, c2_rates;
    std::vector<double> c1_p95, c2_p95;
    double min_free = static_cast<double>(initial_free);
    for (int rep = 0; rep < kWarmups + kMeasured; ++rep) {
        T82Fp16ArmResult c1, c2;
        const bool c2_first = (rep & 1) != 0;
        for (int arm = 0; arm < 2; ++arm) {
            const bool concurrent = c2_first ? arm == 0 : arm == 1;
            auto result = run_arm(concurrent, (rep & 1) != 0);
            if (concurrent) c2 = std::move(result);
            else c1 = std::move(result);
        }
        for (int lane = 0; lane < 2; ++lane) {
            require(c1.lanes[lane].published == c2.lanes[lane].published &&
                        c1.lanes[lane].request->token_suffix() ==
                            c2.lanes[lane].request->token_suffix() &&
                        c1.lanes[lane].request->same_taps(
                            *c2.lanes[lane].request) &&
                        c1.lanes[lane].request->state()->same_payload(
                            *c2.lanes[lane].request->state()),
                    "T82 C1/C2 authoritative stream mismatch");
        }
        const auto write_arm = [&](const char* name,
                                   const T82Fp16ArmResult& result) {
            const double aggregate_rate = 2.0 * kOutputs * 1000.0 /
                result.output_ms;
            for (int lane = 0; lane < 2; ++lane) {
                const auto& trace = result.lanes[lane];
                arms << rep << ',' << (rep < kWarmups) << ','
                    << (c2_first ? "C2C1" : "C1C2") << ',' << name << ','
                    << (lane ? 'B' : 'A') << ',' << preparation_ms[lane] << ','
                    << preparation[lane].reused_prompt_tokens << ','
                    << preparation[lane].executed_prompt_tokens << ','
                    << result.output_ms << ',' << aggregate_rate << ','
                    << kOutputs * 1000.0 / trace.publication_ms.back() << ','
                    << trace.p95_interval_ms << ','
                    << trace.publication_ms.back() << ','
                    << result.residency_update_ms << ','
                    << trace.verification_rows << ',' << trace.executed_rows
                    << ',' << trace.replay_rows << ',' << trace.committed_rows
                    << ',' << trace.native_invocations << ','
                    << trace.root_restores << ',' << result.ready_free_bytes
                    << ',' << result.end_free_bytes << ','
                    << result.ready_owned_bytes << ',' << result.end_owned_bytes
                    << ',' << result.ready_residency.resident_tensor_bytes << ','
                    << result.ready_residency.locked_tensor_bytes << ','
                    << result.end_residency.resident_tensor_bytes << ','
                    << result.end_residency.locked_tensor_bytes << ",1\n";
                for (int ordinal = 0; ordinal < kOutputs; ++ordinal)
                    publications << rep << ',' << (rep < kWarmups) << ','
                        << name << ',' << (lane ? 'B' : 'A') << ',' << ordinal
                        << ',' << trace.published[ordinal] << ','
                        << trace.publication_ms[ordinal] << '\n';
            }
        };
        write_arm("C1", c1);
        write_arm("C2", c2);
        min_free = std::min({min_free, static_cast<double>(c1.ready_free_bytes),
            static_cast<double>(c1.end_free_bytes),
            static_cast<double>(c2.ready_free_bytes),
            static_cast<double>(c2.end_free_bytes)});
        if (rep >= kWarmups) {
            const double c1_rate = 2.0 * kOutputs * 1000.0 / c1.output_ms;
            const double c2_rate = 2.0 * kOutputs * 1000.0 / c2.output_ms;
            const double gain = (c2_rate / c1_rate - 1.0) * 100.0;
            require(gain >= -2.0, "T82 individual C2 performance guard");
            const double release_low = std::min(
                c2.lanes[0].publication_ms.back(),
                c2.lanes[1].publication_ms.back());
            const double release_high = std::max(
                c2.lanes[0].publication_ms.back(),
                c2.lanes[1].publication_ms.back());
            require(release_high / release_low <= 1.05,
                    "T82 C2 final-release fairness");
            gains.push_back(gain);
            c1_rates.push_back(c1_rate);
            c2_rates.push_back(c2_rate);
            c1_p95.push_back(std::max(c1.lanes[0].p95_interval_ms,
                                      c1.lanes[1].p95_interval_ms));
            c2_p95.push_back(std::max(c2.lanes[0].p95_interval_ms,
                                      c2.lanes[1].p95_interval_ms));
        }
    }
    const double median_gain = median(gains);
    require(median_gain > 0.0, "T82 median C2 performance gate");

    // Cancellation after both width-one verifies but before publication.
    contexts[0]->restore_exact_host_state(*initial[0]->state(), streams[0].value);
    auto control_pending = sample_target(*contexts[0], streams[0].value);
    const std::array<std::int64_t, 1> control_token{control_pending};
    auto [control_root, control_verified] = initial[0]->verify(
        *contexts[0], control_token, {}, streams[0].value);
    require(control_verified.committed_tokens.size() == 1,
            "T82 cancellation control");
    for (int lane = 0; lane < 2; ++lane) {
        contexts[lane]->restore_exact_host_state(
            *initial[lane]->state(), streams[lane].value);
        cuda_check(cudaStreamSynchronize(streams[lane].value),
                   "T82 cancellation initial restore");
    }
    std::array<std::int64_t, 2> cancel_pending{
        sample_target(*contexts[0], streams[0].value),
        sample_target(*contexts[1], streams[1].value)};
    std::array<std::shared_ptr<const Request>, 2> verified_roots;
    std::array<ninfer::exl3::Exl3OuterReferenceResult, 2> verified_results;
    std::array<std::exception_ptr, 2> errors;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::thread workers[2];
    for (int lane = 0; lane < 2; ++lane)
        workers[lane] = std::thread([&, lane] {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            try {
                const std::array<std::int64_t, 1> token{cancel_pending[lane]};
                auto result = initial[lane]->verify(
                    *contexts[lane], token, {}, streams[lane].value);
                verified_roots[lane] = std::move(result.first);
                verified_results[lane] = std::move(result.second);
            } catch (...) { errors[lane] = std::current_exception(); }
        });
    while (ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    contexts[1]->restore_exact_host_state(*initial[1]->state(), streams[1].value);
    cuda_check(cudaStreamSynchronize(streams[1].value),
               "T82 canceled lane restore");
    const std::array<std::shared_ptr<const Request>, 2> published_roots{
        verified_roots[0], initial[1]};
    (void)pin(published_roots, true);
    const bool survivor_exact = verified_roots[0]->token_suffix() ==
            control_root->token_suffix() &&
        verified_roots[0]->same_taps(*control_root) &&
        verified_roots[0]->state()->same_payload(*control_root->state());
    const bool canceled_exact = contexts[1]->exact_host_state_resident(
        *initial[1]->state());
    const bool cancellation_pass = survivor_exact && canceled_exact &&
        verified_results[1].verification_rows == 1;
    cancellation << "A,B," << verified_results[1].verification_rows
        << ",1,0," << survivor_exact << ',' << canceled_exact << ",1,"
        << cancellation_pass << '\n';
    require(cancellation_pass, "T82 cancellation gate");

    const auto final_lock = pin(initial, true);
    const auto storage = cache.storage_stats();
    std::size_t final_free = 0;
    cuda_check(cudaMemGetInfo(&final_free, &total_device),
               "T82 final device memory");
    require(storage.entries == 1 && storage.accounted_bytes <= cache_budget &&
                final_free >= (1ULL << 30),
            "T82 final cache/memory gate");
    memory_out << "final_c2," << storage.payload_allocated_bytes << ','
        << storage.identity_allocated_bytes << ',' << storage.accounted_bytes
        << ',' << final_lock.observed.resident_tensor_bytes << ','
        << final_lock.observed.locked_tensor_bytes << ','
        << final_lock.update.page_bytes << ',' << final_lock.update.new_locked_bytes
        << ',' << final_lock.update.unlocked_bytes << ',' << reserve << ','
        << final_free << '\n';
    arms.flush(); publications.flush(); cancellation.flush(); memory_out.flush();
    require(arms.good() && publications.good() && cancellation.good() &&
                memory_out.good(),
            "T82 evidence flush");
    residency.close();
    std::cout << "T82_FP16_C2_SERVING PASS pairs=4 warmups=2 outputs_per_lane=32"
              << " c1_rate_median=" << median(c1_rates)
              << " c2_rate_median=" << median(c2_rates)
              << " gain_median_percent=" << median_gain
              << " c1_p95_median_ms=" << median(c1_p95)
              << " c2_p95_median_ms=" << median(c2_p95)
              << " min_free_bytes=" << static_cast<std::uint64_t>(min_free)
              << " exact=1 cancellation=1 resident_locked=1 draft_loaded=0"
              << " output=" << output.string() << std::endl;
}
