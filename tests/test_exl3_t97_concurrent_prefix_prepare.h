#pragma once

struct T97PrepareTrace {
    std::array<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>,8>
        roots{};
    std::array<std::int64_t,8> first_tokens{};
    std::array<double,8> request_ms{};
    std::array<double,8> ready_ms{};
    std::array<int,8> workers{};
    double wall_ms = 0.0;
    std::uint64_t reused_rows = 0;
    std::uint64_t executed_rows = 0;
    std::uint64_t resident_payload_bytes = 0;
    std::uint64_t locked_payload_bytes = 0;
    std::uint64_t locked_page_bytes = 0;
    std::uint64_t device_free_bytes = 0;
};

void run_t97_concurrent_prefix_prepare(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t kPrefix = 3840;
    constexpr std::array<int,8> kTails{40,56,72,88,104,120,136,152};
    constexpr int kPairs = 4;
    require(target.max_context() >= 4352 && code.size() >= 4096 &&
                prose.size() >= 2048,
            "T97 serving fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T97 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib = std::stoull(env("NINFER_T97_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T97_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024,
            "T97 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 &&
                memory.ullTotalPhys > reserve && memory.ullAvailPhys > reserve,
            "T97 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX value{};
        value.dwLength = sizeof(value);
        require(GlobalMemoryStatusEx(&value) != 0,
                "T97 physical memory query");
        return value.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T97_NAMESPACE"), env("NINFER_T97_ARTIFACT_ID"),
        env("NINFER_T97_TOKENIZER_ID"), env("NINFER_T97_CONFIGURATION_ID"),
        env("NINFER_T97_MODALITY_ID")};
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
    std::array<std::vector<std::int64_t>,8> prompts{};
    for (int request = 0; request < 8; ++request) {
        prompts[request] = common;
        if ((request & 1) == 0) {
            const auto begin = code.begin() + kPrefix + 24 * (request / 2);
            prompts[request].insert(prompts[request].end(), begin,
                                    begin + kTails[request]);
        } else {
            const auto begin = prose.begin() + 192 * (request / 2);
            prompts[request].insert(prompts[request].end(), begin,
                                    begin + kTails[request]);
        }
    }
    auto fill = cache.prepare(*context_a, common, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T97 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                fill.metrics.executed_prompt_tokens == kPrefix,
            "T97 cache fill gate");
    fill.request.reset();

    std::ofstream arms_out(output / "arms.csv");
    std::ofstream requests_out(output / "requests.csv");
    std::ofstream lifecycle(output / "lifecycle.csv");
    require(arms_out.good() && requests_out.good() && lifecycle.good(),
            "T97 evidence open");
    arms_out << "pair,order,arm,physical_contexts,wall_ms,tail_rows_per_s,"
        "reused_rows,executed_rows,p95_ready_ms,resident_payload_bytes,"
        "locked_payload_bytes,locked_page_bytes,device_free_bytes,exact\n";
    requests_out << "pair,arm,request,tail_rows,worker,request_ms,ready_ms,"
        "reused_rows,executed_rows,first_token,exact\n";
    lifecycle << "case,cache_entries_before,cache_entries_after,"
        "accounted_bytes_before,accounted_bytes_after,rejected,recovered,pass\n";

    const auto pin = [&](const std::array<std::shared_ptr<const Request>,8>& roots) {
        auto all = cache.roots();
        for (const auto& root : roots) all.push_back(root);
        ninfer::exl3::Exl3HostResidentSet::Snapshot snapshot;
        for (const auto& root : all) snapshot.owners.push_back(root);
        Request::visit_host_allocations(all,
            [&](const void* data, std::size_t bytes) { snapshot.add(data, bytes); },
            false);
        const auto update = residency.replace(std::move(snapshot));
        Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(all,
            [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
            false);
        return std::pair{update, probe.measure()};
    };
    const auto run_arm = [&](bool concurrent) {
        T97PrepareTrace result;
        const auto batch_started = Clock::now();
        std::atomic<int> next{0};
        std::array<std::exception_ptr,2> errors{};
        const auto worker = [&](int physical) {
            try {
                while (true) {
                    const int request = next.fetch_add(1);
                    if (request >= 8) break;
                    const auto started = Clock::now();
                    auto prepared = cache.prepare_concurrent(
                        *contexts[physical], prompts[request], kPrefix,
                        available(), 1024);
                    cuda_check(cudaStreamSynchronize(streams[physical].value),
                               "T97 concurrent preparation completion");
                    result.request_ms[request] =
                        std::chrono::duration<double,std::milli>(
                            Clock::now() - started).count();
                    result.ready_ms[request] =
                        std::chrono::duration<double,std::milli>(
                            Clock::now() - batch_started).count();
                    result.workers[request] = physical;
                    require(prepared.metrics.cache_hit &&
                                prepared.metrics.reused_prompt_tokens == kPrefix &&
                                prepared.metrics.executed_prompt_tokens ==
                                    static_cast<std::size_t>(kTails[request]),
                            "T97 concurrent cached accounting");
                    result.roots[request] = std::move(prepared.request);
                    result.first_tokens[request] = sample_target(
                        *contexts[physical], streams[physical].value);
                }
            } catch (...) { errors[physical] = std::current_exception(); }
        };
        if (concurrent) {
            std::thread workers[2]{std::thread(worker, 0),
                                   std::thread(worker, 1)};
            for (auto& thread : workers) thread.join();
        } else {
            // Same two contexts and request order, but no overlapping work.
            for (int request = 0; request < 8; ++request) {
                const int physical = request & 1;
                const auto started = Clock::now();
                auto prepared = cache.prepare(
                    *contexts[physical], prompts[request], kPrefix,
                    available(), 1024);
                cuda_check(cudaStreamSynchronize(streams[physical].value),
                           "T97 serialized preparation completion");
                result.request_ms[request] =
                    std::chrono::duration<double,std::milli>(
                        Clock::now() - started).count();
                result.ready_ms[request] =
                    std::chrono::duration<double,std::milli>(
                        Clock::now() - batch_started).count();
                result.workers[request] = physical;
                require(prepared.metrics.cache_hit &&
                            prepared.metrics.reused_prompt_tokens == kPrefix &&
                            prepared.metrics.executed_prompt_tokens ==
                                static_cast<std::size_t>(kTails[request]),
                        "T97 serialized cached accounting");
                result.roots[request] = std::move(prepared.request);
                result.first_tokens[request] = sample_target(
                    *contexts[physical], streams[physical].value);
            }
        }
        for (const auto& error : errors)
            if (error) std::rethrow_exception(error);
        result.reused_rows = 8 * kPrefix;
        result.executed_rows = 768;
        result.wall_ms = std::chrono::duration<double,std::milli>(
            Clock::now() - batch_started).count();
        require(result.reused_rows == 8 * kPrefix &&
                    result.executed_rows == 768,
                "T97 aggregate prompt accounting");
        const auto [update, observed] = pin(result.roots);
        result.resident_payload_bytes = observed.resident_tensor_bytes;
        result.locked_payload_bytes = observed.locked_tensor_bytes;
        result.locked_page_bytes = update.page_bytes;
        std::size_t total_bytes = 0;
        cuda_check(cudaMemGetInfo(&result.device_free_bytes, &total_bytes),
                   "T97 device memory query");
        require(observed.resident_tensor_bytes == observed.allocated_union_bytes &&
                    observed.locked_tensor_bytes == observed.allocated_union_bytes &&
                    result.device_free_bytes >= (1ULL << 30),
                "T97 resident and device memory gate");
        return result;
    };

    std::array<std::shared_ptr<const Request>,8> reference_roots{};
    std::array<std::int64_t,8> reference_tokens{};
    bool reference_ready = false;
    for (int pair = 0; pair < kPairs; ++pair) {
        const bool concurrent_first = (pair & 1) != 0;
        std::array<T97PrepareTrace,2> result{};
        for (int ordinal = 0; ordinal < 2; ++ordinal) {
            const bool concurrent = concurrent_first ? ordinal == 0 : ordinal == 1;
            const int slot = concurrent ? 1 : 0;
            result[slot] = run_arm(concurrent);
            bool exact = true;
            if (!reference_ready) {
                reference_roots = result[slot].roots;
                reference_tokens = result[slot].first_tokens;
                reference_ready = true;
            } else {
                for (int request = 0; request < 8; ++request)
                    exact = exact &&
                        result[slot].first_tokens[request] ==
                            reference_tokens[request] &&
                        result[slot].roots[request]->token_suffix() ==
                            reference_roots[request]->token_suffix() &&
                        result[slot].roots[request]->same_taps(
                            *reference_roots[request]) &&
                        result[slot].roots[request]->state()->same_payload(
                            *reference_roots[request]->state());
            }
            require(exact, "T97 cross-arm exact preparation");
            auto ready = result[slot].ready_ms;
            std::sort(ready.begin(), ready.end());
            const double p95_ready = ready.back();
            arms_out << pair << ',' << (concurrent_first ? "CS" : "SC")
                << ',' << (concurrent ? "CONCURRENT" : "SERIALIZED") << ','
                << (concurrent ? 2 : 1) << ',' << result[slot].wall_ms << ','
                << 768000.0 / result[slot].wall_ms << ','
                << result[slot].reused_rows << ','
                << result[slot].executed_rows << ',' << p95_ready << ','
                << result[slot].resident_payload_bytes << ','
                << result[slot].locked_payload_bytes << ','
                << result[slot].locked_page_bytes << ','
                << result[slot].device_free_bytes << ",1\n";
            for (int request = 0; request < 8; ++request)
                requests_out << pair << ','
                    << (concurrent ? "CONCURRENT" : "SERIALIZED") << ','
                    << request << ',' << kTails[request] << ','
                    << result[slot].workers[request] << ','
                    << result[slot].request_ms[request] << ','
                    << result[slot].ready_ms[request] << ',' << kPrefix << ','
                    << kTails[request] << ','
                    << result[slot].first_tokens[request] << ",1\n";
        }
        for (int request = 0; request < 8; ++request)
            require(result[0].first_tokens[request] ==
                        result[1].first_tokens[request] &&
                    result[0].roots[request]->token_suffix() ==
                        result[1].roots[request]->token_suffix() &&
                    result[0].roots[request]->same_taps(
                        *result[1].roots[request]) &&
                    result[0].roots[request]->state()->same_payload(
                        *result[1].roots[request]->state()),
                "T97 pair exact equality");
    }
    residency.close();

    const auto before_cancel = cache.storage_stats();
    bool cancellation_rejected = false;
    try {
        (void)cache.prepare_concurrent(*context_a, prompts[7], kPrefix,
            available(), 32, [&](int) {
                throw std::runtime_error("intentional T97 cancellation");
            });
    } catch (const std::runtime_error&) { cancellation_rejected = true; }
    const auto after_cancel = cache.storage_stats();
    auto recovered = cache.prepare_concurrent(
        *context_a, prompts[0], kPrefix, available(), 1024);
    const bool cancel_recovered = recovered.metrics.cache_hit &&
        recovered.metrics.reused_prompt_tokens == kPrefix;
    recovered.request.reset();
    const bool cancel_unchanged = before_cancel.entries == after_cancel.entries &&
        before_cancel.accounted_bytes == after_cancel.accounted_bytes;
    lifecycle << "cancellation," << before_cancel.entries << ','
        << after_cancel.entries << ',' << before_cancel.accounted_bytes << ','
        << after_cancel.accounted_bytes << ',' << cancellation_rejected << ','
        << cancel_recovered << ','
        << (cancellation_rejected && cancel_recovered && cancel_unchanged)
        << '\n';
    require(cancellation_rejected && cancel_recovered && cancel_unchanged,
            "T97 cancellation rollback");

    const auto before_reload = cache.storage_stats();
    std::atomic<bool> progress_entered{false};
    std::atomic<bool> reload_complete{false};
    bool epoch_rejected = false;
    std::shared_ptr<const Request> stale_result;
    std::exception_ptr reload_error;
    std::thread in_flight([&] {
        try {
            auto prepared = cache.prepare_concurrent(
                *context_a, prompts[7], kPrefix, available(), 32,
                [&](int) {
                    if (!progress_entered.exchange(true)) {
                        while (!reload_complete.load(std::memory_order_acquire))
                            std::this_thread::yield();
                    }
                });
            stale_result = std::move(prepared.request);
        } catch (const std::runtime_error&) { epoch_rejected = true; }
        catch (...) { reload_error = std::current_exception(); }
    });
    while (!progress_entered.load(std::memory_order_acquire))
        std::this_thread::yield();
    auto changed = identity;
    changed.configuration_identity += "-t97-reload";
    cache.reset_for_model_reload(changed);
    reload_complete.store(true, std::memory_order_release);
    in_flight.join();
    if (reload_error) std::rethrow_exception(reload_error);
    const auto after_reload = cache.storage_stats();
    const bool reload_cleared = after_reload.entries == 0 &&
                                after_reload.accounted_bytes == 0;
    cache.reset_for_model_reload(identity);
    auto refill = cache.prepare(*context_b, common, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T97 reload recovery fill");
    const bool reload_recovered = !refill.metrics.cache_hit &&
                                  refill.metrics.admitted;
    refill.request.reset();
    lifecycle << "reload_epoch," << before_reload.entries << ','
        << after_reload.entries << ',' << before_reload.accounted_bytes << ','
        << after_reload.accounted_bytes << ',' << epoch_rejected << ','
        << reload_recovered << ','
        << (epoch_rejected && !stale_result && reload_cleared && reload_recovered)
        << '\n';
    require(epoch_rejected && !stale_result && reload_cleared &&
                reload_recovered,
            "T97 reload epoch gate");
    const auto final_storage = cache.storage_stats();
    require(final_storage.entries == 1 &&
                final_storage.accounted_bytes <= cache_budget,
            "T97 final cache budget");
    arms_out.flush();
    requests_out.flush();
    lifecycle.flush();
    require(arms_out.good() && requests_out.good() && lifecycle.good(),
            "T97 evidence flush");
    std::cout << "T97_CONCURRENT_PREFIX_PREPARE PASS pairs=4 arms=8"
        " logical_requests=8 physical_contexts=2 exact=1 lifecycle=2 output="
        << output.string() << std::endl;
}
