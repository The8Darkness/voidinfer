#pragma once
#include "test_exl3_host_residency.h"
#include <atomic>
#include <thread>

struct T83Fp16RequestTrace {
    std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> request;
    std::vector<std::int64_t> published;
    std::vector<double> publication_ms;
    double queue_wait_ms = 0.0;
    double restore_ms = 0.0;
    double p95_interval_ms = 0.0;
    std::uint64_t verification_rows = 0;
    std::uint64_t executed_rows = 0;
    std::uint64_t replay_rows = 0;
    std::uint64_t committed_rows = 0;
    std::uint64_t native_invocations = 0;
    std::uint64_t root_restores = 0;
};

struct T83Fp16ArmTrace {
    std::array<T83Fp16RequestTrace, 8> requests;
    double makespan_ms = 0.0;
    double residency_update_ms = 0.0;
    std::size_t ready_free_bytes = 0;
    std::size_t end_free_bytes = 0;
    std::size_t ready_owned_bytes = 0;
    std::size_t end_owned_bytes = 0;
    Exl3HostResidency ready_residency;
    Exl3HostResidency end_residency;
};

void run_t83_fp16_c8_serving(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    constexpr std::size_t kPrompt = 4096;
    constexpr std::size_t kPrefix = 3968;
    constexpr std::size_t kTail = 128;
    constexpr int kOutputs = 8;
    constexpr int kWarmups = 2;
    constexpr int kMeasured = 4;
    require(target.max_context() >= 4352 && code.size() >= kPrompt &&
                prose.size() >= 4 * kTail,
            "T83 serving fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T83 output must be new");
    std::filesystem::create_directories(output);

    const auto reserve_gib = std::stoull(env("NINFER_T83_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T83_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024,
            "T83 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullTotalPhys > reserve &&
                memory.ullAvailPhys > reserve,
            "T83 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0,
                "T83 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T83_NAMESPACE"), env("NINFER_T83_ARTIFACT_ID"),
        env("NINFER_T83_TOKENIZER_ID"), env("NINFER_T83_CONFIGURATION_ID"),
        env("NINFER_T83_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{16, 64, cache_budget, reserve}, identity);
    ninfer::exl3::Exl3HostResidentSet residency(
        memory.ullTotalPhys - reserve, reserve);
    auto context_a = target.create_context(true);
    auto context_b = target.create_context(true);
    context_a->prepare_continuation(8);
    context_b->prepare_continuation(8);
    Exl3TextContext* contexts[2]{context_a.get(), context_b.get()};
    TargetGraphC2Stream streams[2];

    std::vector<std::int64_t> common(code.begin(), code.begin() + kPrefix);
    std::array<std::vector<std::int64_t>, 8> prompts;
    for (auto& prompt : prompts) prompt = common;
    for (int tail = 0; tail < 4; ++tail) {
        const auto code_end = code.begin() + kPrompt - tail * kTail;
        prompts[tail].insert(prompts[tail].end(), code_end - kTail, code_end);
        const auto prose_begin = prose.begin() + tail * kTail;
        prompts[tail + 4].insert(prompts[tail + 4].end(), prose_begin,
                                 prose_begin + kTail);
    }
    for (int left = 0; left < 8; ++left) {
        require(prompts[left].size() == kPrompt, "T83 prompt extent");
        for (int right = left + 1; right < 8; ++right)
            require(!std::equal(prompts[left].begin() + kPrefix,
                                prompts[left].end(),
                                prompts[right].begin() + kPrefix),
                    "T83 tails must be pairwise distinct");
    }

    struct PinResult {
        ninfer::exl3::Exl3HostResidentSet::Stats update;
        Exl3HostResidency observed;
    };
    const auto pin = [&](const std::array<std::shared_ptr<const Request>, 8>& active,
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
                    "T83 authoritative payload not resident and locked");
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
    cuda_check(cudaDeviceSynchronize(), "T83 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                fill.metrics.executed_prompt_tokens == kPrefix,
            "T83 cache fill gate");
    fill.request.reset();
    std::array<std::shared_ptr<const Request>, 8> initial;
    std::array<Cache::Metrics, 8> preparation;
    std::array<double, 8> preparation_ms{};
    for (int request = 0; request < 8; ++request) {
        auto* context = contexts[request & 1];
        const auto started = std::chrono::steady_clock::now();
        auto prepared = cache.prepare(*context, prompts[request], kPrefix,
                                      available(), 1024);
        cuda_check(cudaDeviceSynchronize(), "T83 request preparation completion");
        preparation_ms[request] = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        preparation[request] = prepared.metrics;
        initial[request] = std::move(prepared.request);
        require(preparation[request].cache_hit &&
                    preparation[request].reused_prompt_tokens == kPrefix &&
                    preparation[request].executed_prompt_tokens == kTail,
                "T83 cached preparation accounting");
    }
    const auto initial_lock = pin(initial, true);
    require(initial_lock.observed.resident_tensor_bytes <
                memory.ullTotalPhys - reserve,
            "T83 C8 preflight host reserve");

    std::ofstream batches(output / "batches.csv");
    std::ofstream requests_out(output / "requests.csv");
    std::ofstream preparation_out(output / "preparation.csv");
    std::ofstream cancellation(output / "cancellation.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(batches.good() && requests_out.good() && preparation_out.good() &&
                cancellation.good() && memory_out.good(),
            "T83 evidence open");
    preparation_out << "request,preparation_ms,reused_prompt_tokens,"
                       "executed_prompt_tokens,cache_hit\n";
    for (int request = 0; request < 8; ++request)
        preparation_out << request << ',' << preparation_ms[request] << ','
            << preparation[request].reused_prompt_tokens << ','
            << preparation[request].executed_prompt_tokens << ",1\n";
    batches << "rep,warmup,order,arm,schedule,logical_requests,physical_limit,"
               "queue_high_water,makespan_ms,aggregate_output_s,"
               "residency_update_ms,ready_free_bytes,end_free_bytes,"
               "ready_owned_bytes,end_owned_bytes,ready_resident_bytes,"
               "ready_locked_bytes,end_resident_bytes,end_locked_bytes,exact\n";
    requests_out << "rep,warmup,arm,request,wave,first_wave,queue_wait_ms,"
                    "restore_ms,first_release_ms,final_release_ms,"
                    "p95_interval_ms,verification_rows,executed_rows,"
                    "replay_rows,committed_rows,native_invocations,root_restores,"
                    "published_tokens,exact\n";
    cancellation << "case,survivor,canceled,verified_rows_discarded,"
                    "survivor_publications,canceled_publications,survivor_exact,"
                    "canceled_root_exact,pass\n";
    memory_out << "phase,cache_payload_bytes,cache_identity_bytes,"
                  "cache_accounted_bytes,resident_payload_bytes,"
                  "locked_payload_bytes,resident_page_bytes,new_locked_bytes,"
                  "unlocked_bytes,physical_reserve_bytes,device_free_bytes\n";
    std::size_t total_device = 0, initial_free = 0;
    cuda_check(cudaMemGetInfo(&initial_free, &total_device),
               "T83 initial device memory");
    memory_out << "active_c8," << cache.storage_stats().payload_allocated_bytes
        << ',' << cache.storage_stats().identity_allocated_bytes << ','
        << cache.storage_stats().accounted_bytes << ','
        << initial_lock.observed.resident_tensor_bytes << ','
        << initial_lock.observed.locked_tensor_bytes << ','
        << initial_lock.update.page_bytes << ',' << initial_lock.update.new_locked_bytes
        << ',' << initial_lock.update.unlocked_bytes << ',' << reserve << ','
        << initial_free << '\n';

    const auto run_arm = [&](bool concurrent, int rep) {
        T83Fp16ArmTrace result;
        std::array<std::shared_ptr<const Request>, 8> current = initial;
        const auto ready_pin = pin(current, true);
        result.ready_residency = ready_pin.observed;
        result.ready_owned_bytes = owned_bytes();
        cuda_check(cudaMemGetInfo(&result.ready_free_bytes, &total_device),
                   "T83 ready device memory");
        require(result.ready_free_bytes >= (1ULL << 30),
                "T83 ready device reserve");
        std::array<int, 8> schedule{};
        const int first = (2 * rep) & 7;
        for (int index = 0; index < 8; ++index)
            schedule[index] = (first + index) & 7;
        const auto batch_started = std::chrono::steady_clock::now();

        const auto restore_request = [&](int request, int physical) {
            const auto started = std::chrono::steady_clock::now();
            contexts[physical]->restore_exact_host_state(
                *current[request]->state(), streams[physical].value);
            cuda_check(cudaStreamSynchronize(streams[physical].value),
                       "T83 request root restore");
            result.requests[request].restore_ms +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
        };
        const auto verify_request = [&](int request, int physical,
                                        std::int64_t pending) {
            const std::array<std::int64_t, 1> tentative{pending};
            auto [updated, verified] = current[request]->verify(
                *contexts[physical], tentative, {}, streams[physical].value);
            require(verified.committed_tokens.size() == 1 &&
                        verified.committed_tokens[0] == pending &&
                        verified.accepted == 1 && !verified.rejected &&
                        !verified.stopped,
                    "T83 width-one authoritative decision");
            current[request] = std::move(updated);
            auto& trace = result.requests[request];
            trace.verification_rows += verified.verification_rows;
            trace.executed_rows += verified.executed_rows;
            trace.replay_rows += verified.replay_rows;
            trace.committed_rows += verified.committed_tokens.size();
            trace.native_invocations += verified.native_invocations;
            trace.root_restores += verified.root_restores;
        };

        if (concurrent) {
            for (int wave = 0; wave < 4; ++wave) {
                const int request_ids[2]{schedule[2 * wave], schedule[2 * wave + 1]};
                const double queue_wait = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - batch_started).count();
                for (const int request : request_ids) {
                    result.requests[request].queue_wait_ms = queue_wait;
                }
                std::array<std::exception_ptr, 2> restore_errors;
                std::thread restore_workers[2];
                for (int physical = 0; physical < 2; ++physical)
                    restore_workers[physical] = std::thread([&, physical] {
                        try { restore_request(request_ids[physical], physical); }
                        catch (...) { restore_errors[physical] = std::current_exception(); }
                    });
                for (auto& worker : restore_workers) worker.join();
                for (const auto& error : restore_errors)
                    if (error) std::rethrow_exception(error);
                std::array<std::int64_t, 2> pending{
                    sample_target(*contexts[0], streams[0].value),
                    sample_target(*contexts[1], streams[1].value)};
                for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
                    std::array<std::exception_ptr, 2> errors;
                    std::atomic<int> ready{0};
                    std::atomic<bool> go{false};
                    std::thread workers[2];
                    for (int physical = 0; physical < 2; ++physical)
                        workers[physical] = std::thread([&, physical] {
                            ready.fetch_add(1, std::memory_order_release);
                            while (!go.load(std::memory_order_acquire))
                                std::this_thread::yield();
                            try { verify_request(request_ids[physical], physical,
                                                 pending[physical]); }
                            catch (...) { errors[physical] = std::current_exception(); }
                        });
                    while (ready.load(std::memory_order_acquire) != 2)
                        std::this_thread::yield();
                    go.store(true, std::memory_order_release);
                    for (auto& worker : workers) worker.join();
                    for (const auto& error : errors)
                        if (error) std::rethrow_exception(error);
                    const auto update = pin(current, false);
                    result.residency_update_ms += update.update.elapsed_ms;
                    const double stamp = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - batch_started).count();
                    for (int physical = 0; physical < 2; ++physical) {
                        auto& trace = result.requests[request_ids[physical]];
                        trace.published.push_back(pending[physical]);
                        trace.publication_ms.push_back(stamp);
                    }
                    if (ordinal + 1 < kOutputs) {
                        std::array<std::exception_ptr, 2> sample_errors;
                        std::thread sample_workers[2];
                        for (int physical = 0; physical < 2; ++physical)
                            sample_workers[physical] = std::thread([&, physical] {
                                try { pending[physical] = sample_target(
                                    *contexts[physical], streams[physical].value); }
                                catch (...) { sample_errors[physical] =
                                    std::current_exception(); }
                            });
                        for (auto& worker : sample_workers) worker.join();
                        for (const auto& error : sample_errors)
                            if (error) std::rethrow_exception(error);
                    }
                }
            }
        } else {
            for (int order_index = 0; order_index < 8; ++order_index) {
                const int request = schedule[order_index];
                auto& trace = result.requests[request];
                trace.queue_wait_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - batch_started).count();
                restore_request(request, 0);
                auto pending = sample_target(*contexts[0], streams[0].value);
                for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
                    verify_request(request, 0, pending);
                    const auto update = pin(current, false);
                    result.residency_update_ms += update.update.elapsed_ms;
                    trace.published.push_back(pending);
                    trace.publication_ms.push_back(
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - batch_started).count());
                    if (ordinal + 1 < kOutputs)
                        pending = sample_target(*contexts[0], streams[0].value);
                }
            }
        }
        result.makespan_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - batch_started).count();
        for (int request = 0; request < 8; ++request) {
            auto& trace = result.requests[request];
            trace.request = current[request];
            std::vector<double> intervals;
            double previous = trace.queue_wait_ms;
            for (const double stamp : trace.publication_ms) {
                intervals.push_back(stamp - previous);
                previous = stamp;
            }
            trace.p95_interval_ms = t80_p95(std::move(intervals));
            require(trace.published.size() == kOutputs &&
                        trace.verification_rows == kOutputs &&
                        trace.executed_rows == kOutputs &&
                        trace.committed_rows == kOutputs &&
                        trace.native_invocations == kOutputs &&
                        trace.replay_rows == 0 && trace.root_restores == 0,
                    "T83 request work accounting");
        }
        const auto end_pin = pin(current, true);
        result.end_residency = end_pin.observed;
        result.end_owned_bytes = owned_bytes();
        cuda_check(cudaMemGetInfo(&result.end_free_bytes, &total_device),
                   "T83 end device memory");
        require(result.ready_owned_bytes == result.end_owned_bytes &&
                    result.end_free_bytes >= (1ULL << 30),
                "T83 logical/device memory gate");
        return result;
    };

    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const auto middle = values.size() / 2;
        return values.size() & 1 ? values[middle] :
            (values[middle - 1] + values[middle]) / 2.0;
    };
    std::vector<double> gains, c1_rates, c2_rates;
    std::array<int, 8> first_wave_counts{};
    for (int rep = 0; rep < kWarmups + kMeasured; ++rep) {
        T83Fp16ArmTrace c1, c2;
        const bool c2_first = (rep & 1) != 0;
        for (int arm = 0; arm < 2; ++arm) {
            const bool concurrent = c2_first ? arm == 0 : arm == 1;
            auto trace = run_arm(concurrent, rep);
            if (concurrent) c2 = std::move(trace);
            else c1 = std::move(trace);
        }
        for (int request = 0; request < 8; ++request)
            require(c1.requests[request].published ==
                        c2.requests[request].published &&
                    c1.requests[request].request->token_suffix() ==
                        c2.requests[request].request->token_suffix() &&
                    c1.requests[request].request->same_taps(
                        *c2.requests[request].request) &&
                    c1.requests[request].request->state()->same_payload(
                        *c2.requests[request].request->state()),
                    "T83 C1/C2 authoritative request mismatch");
        const auto schedule_text = [&] {
            std::ostringstream out;
            for (int index = 0; index < 8; ++index) {
                if (index) out << '|';
                out << (((2 * rep) + index) & 7);
            }
            return out.str();
        }();
        const auto write_arm = [&](const char* name,
                                   const T83Fp16ArmTrace& trace) {
            const double rate = 8.0 * kOutputs * 1000.0 / trace.makespan_ms;
            batches << rep << ',' << (rep < kWarmups) << ','
                << (c2_first ? "C2C1" : "C1C2") << ',' << name << ','
                << schedule_text << ",8," << (name[1] == '2' ? 2 : 1)
                << ',' << (name[1] == '2' ? 6 : 7) << ',' << trace.makespan_ms
                << ',' << rate << ',' << trace.residency_update_ms << ','
                << trace.ready_free_bytes << ',' << trace.end_free_bytes << ','
                << trace.ready_owned_bytes << ',' << trace.end_owned_bytes << ','
                << trace.ready_residency.resident_tensor_bytes << ','
                << trace.ready_residency.locked_tensor_bytes << ','
                << trace.end_residency.resident_tensor_bytes << ','
                << trace.end_residency.locked_tensor_bytes << ",1\n";
            for (int request = 0; request < 8; ++request) {
                const auto& row = trace.requests[request];
                const int relative = (request - ((2 * rep) & 7) + 8) & 7;
                const int wave = name[1] == '2' ? relative / 2 : relative;
                requests_out << rep << ',' << (rep < kWarmups) << ',' << name
                    << ',' << request << ',' << wave << ',' << (wave == 0)
                    << ',' << row.queue_wait_ms << ',' << row.restore_ms << ','
                    << row.publication_ms.front() << ','
                    << row.publication_ms.back() << ',' << row.p95_interval_ms
                    << ',' << row.verification_rows << ',' << row.executed_rows
                    << ',' << row.replay_rows << ',' << row.committed_rows << ','
                    << row.native_invocations << ',' << row.root_restores << ','
                    << target_graph_c2_verifier_ids(row.published) << ",1\n";
            }
        };
        write_arm("C1", c1);
        write_arm("C2", c2);
        if (rep >= kWarmups) {
            const int first = (2 * rep) & 7;
            ++first_wave_counts[first];
            ++first_wave_counts[(first + 1) & 7];
            const double c1_rate = 8.0 * kOutputs * 1000.0 / c1.makespan_ms;
            const double c2_rate = 8.0 * kOutputs * 1000.0 / c2.makespan_ms;
            const double gain = (c2_rate / c1_rate - 1.0) * 100.0;
            require(gain >= -2.0, "T83 individual C2 performance guard");
            gains.push_back(gain);
            c1_rates.push_back(c1_rate);
            c2_rates.push_back(c2_rate);
        }
    }
    require(first_wave_counts == std::array<int, 8>{1,1,1,1,1,1,1,1},
            "T83 measured first-wave fairness");
    const double median_gain = median(gains);
    require(median_gain > 0.0, "T83 median C2 performance gate");

    // Queued request 7 is canceled before restore or verification.
    const auto queued_state = initial[7]->state();
    const auto queued_tokens = initial[7]->token_suffix();
    const bool queued_exact = initial[7]->state() == queued_state &&
        initial[7]->token_suffix() == queued_tokens;
    cancellation << "queued,NONE,7,0,0,0,1," << queued_exact << ','
        << queued_exact << '\n';
    require(queued_exact, "T83 queued cancellation gate");

    // Pair cancellation after verification and before resident publication.
    contexts[0]->restore_exact_host_state(*initial[0]->state(), streams[0].value);
    const auto control_pending = sample_target(*contexts[0], streams[0].value);
    const std::array<std::int64_t,1> control_token{control_pending};
    auto [control_root, control_verified] = initial[0]->verify(
        *contexts[0], control_token, {}, streams[0].value);
    require(control_verified.committed_tokens.size() == 1,
            "T83 cancellation control");
    for (int physical = 0; physical < 2; ++physical) {
        contexts[physical]->restore_exact_host_state(
            *initial[physical]->state(), streams[physical].value);
        cuda_check(cudaStreamSynchronize(streams[physical].value),
                   "T83 cancellation restore");
    }
    std::array<std::int64_t,2> pending{
        sample_target(*contexts[0], streams[0].value),
        sample_target(*contexts[1], streams[1].value)};
    std::array<std::shared_ptr<const Request>,2> updated;
    std::array<ninfer::exl3::Exl3OuterReferenceResult,2> verified;
    std::array<std::exception_ptr,2> errors;
    std::thread workers[2];
    for (int physical = 0; physical < 2; ++physical)
        workers[physical] = std::thread([&, physical] {
            try {
                const std::array<std::int64_t,1> token{pending[physical]};
                auto result = initial[physical]->verify(
                    *contexts[physical], token, {}, streams[physical].value);
                updated[physical] = std::move(result.first);
                verified[physical] = std::move(result.second);
            } catch (...) { errors[physical] = std::current_exception(); }
        });
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    contexts[1]->restore_exact_host_state(*initial[1]->state(), streams[1].value);
    cuda_check(cudaStreamSynchronize(streams[1].value),
               "T83 canceled root restore");
    auto published_roots = initial;
    published_roots[0] = updated[0];
    (void)pin(published_roots, true);
    const bool survivor_exact = updated[0]->token_suffix() ==
            control_root->token_suffix() && updated[0]->same_taps(*control_root) &&
        updated[0]->state()->same_payload(*control_root->state());
    const bool canceled_exact = contexts[1]->exact_host_state_resident(
        *initial[1]->state());
    const bool cancel_pass = survivor_exact && canceled_exact &&
        verified[1].verification_rows == 1;
    cancellation << "prepublication,0,1," << verified[1].verification_rows
        << ",1,0," << survivor_exact << ',' << canceled_exact << ','
        << cancel_pass << '\n';
    require(cancel_pass, "T83 prepublication cancellation gate");

    const auto final_lock = pin(initial, true);
    const auto storage = cache.storage_stats();
    std::size_t final_free = 0;
    cuda_check(cudaMemGetInfo(&final_free, &total_device),
               "T83 final device memory");
    require(final_lock.observed.resident_tensor_bytes ==
                initial_lock.observed.resident_tensor_bytes &&
            final_lock.observed.locked_tensor_bytes ==
                initial_lock.observed.locked_tensor_bytes &&
            storage.entries == 1 && storage.accounted_bytes <= cache_budget &&
            final_free >= (1ULL << 30),
            "T83 final memory gate");
    memory_out << "final_c8," << storage.payload_allocated_bytes << ','
        << storage.identity_allocated_bytes << ',' << storage.accounted_bytes
        << ',' << final_lock.observed.resident_tensor_bytes << ','
        << final_lock.observed.locked_tensor_bytes << ','
        << final_lock.update.page_bytes << ',' << final_lock.update.new_locked_bytes
        << ',' << final_lock.update.unlocked_bytes << ',' << reserve << ','
        << final_free << '\n';
    batches.flush(); requests_out.flush(); preparation_out.flush();
    cancellation.flush(); memory_out.flush();
    require(batches.good() && requests_out.good() && preparation_out.good() &&
                cancellation.good() && memory_out.good(),
            "T83 evidence flush");
    residency.close();
    std::cout << "T83_FP16_C8_SERVING PASS logical_requests=8 physical_limit=2"
              << " outputs_per_request=8 measured_pairs=4"
              << " c1_rate_median=" << median(c1_rates)
              << " c2_rate_median=" << median(c2_rates)
              << " gain_median_percent=" << median_gain
              << " active_resident_locked_bytes="
              << initial_lock.observed.resident_tensor_bytes
              << " exact=1 cancellation=2 draft_loaded=0"
              << " output=" << output.string() << std::endl;
}
