#pragma once

struct T98ArmTrace {
    double arrival_turnover_ms = 0.0;
    double total_wall_ms = 0.0;
    double preparation_ms = 0.0;
    double execute_publish_ms = 0.0;
    double resident_registry_ms = 0.0;
    std::uint64_t resident_updates = 0;
    std::uint64_t reused_rows = 0;
    std::uint64_t prompt_executed_rows = 0;
    std::uint64_t verification_rows = 0;
    std::uint64_t executed_rows = 0;
    std::uint64_t replay_rows = 0;
    std::uint64_t committed_rows = 0;
    std::array<double,8> replacement_ttft_ms{};
    std::array<std::int64_t,16> tokens{};
    std::array<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>,16>
        roots{};
    std::uint64_t resident_payload_bytes = 0;
    std::uint64_t locked_payload_bytes = 0;
    std::uint64_t locked_page_bytes = 0;
    std::uint64_t device_free_bytes = 0;
};

void run_t98_pipelined_turnover(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    using Coordinator = ninfer::exl3::Exl3VeriCacheServingCoordinator;
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t kPrefix = 3840;
    constexpr std::array<int,8> kInitialTails{32,48,64,80,96,112,128,144};
    constexpr std::array<int,8> kReplacementTails{40,56,72,88,104,120,136,152};
    constexpr int kPairs = 4;
    require(target.max_context() >= 4352 && code.size() >= 4096 &&
                prose.size() >= 2048,
            "T98 serving fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T98 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib = std::stoull(env("NINFER_T98_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T98_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024,
            "T98 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 &&
                memory.ullTotalPhys > reserve && memory.ullAvailPhys > reserve,
            "T98 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX value{};
        value.dwLength = sizeof(value);
        require(GlobalMemoryStatusEx(&value) != 0,
                "T98 physical memory query");
        return value.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T98_NAMESPACE"), env("NINFER_T98_ARTIFACT_ID"),
        env("NINFER_T98_TOKENIZER_ID"), env("NINFER_T98_CONFIGURATION_ID"),
        env("NINFER_T98_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{16, 64, cache_budget, reserve}, identity);
    auto context_a = target.create_context(true);
    auto context_b = target.create_context(true);
    context_a->prepare_continuation(8);
    context_b->prepare_continuation(8);
    Exl3TextContext* contexts[2]{context_a.get(), context_b.get()};
    TargetGraphC2Stream streams[2];
    std::vector<std::int64_t> common(code.begin(), code.begin() + kPrefix);
    const auto make_prompt = [&](int request, bool replacement) {
        std::vector<std::int64_t> prompt = common;
        const int rows = replacement ? kReplacementTails[request] :
                                       kInitialTails[request];
        if ((request & 1) == 0) {
            const int offset = replacement ? 24 * (request / 2) :
                                             32 * (request / 2);
            const auto begin = code.begin() + kPrefix + offset;
            prompt.insert(prompt.end(), begin, begin + rows);
        } else {
            const int offset = (replacement ? 1024 : 0) +
                               192 * (request / 2);
            const auto begin = prose.begin() + offset;
            prompt.insert(prompt.end(), begin, begin + rows);
        }
        return prompt;
    };
    std::array<std::vector<std::int64_t>,8> replacement_prompts{};
    for (int request = 0; request < 8; ++request)
        replacement_prompts[request] = make_prompt(request, true);
    auto fill = cache.prepare(*context_a, common, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T98 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                fill.metrics.executed_prompt_tokens == kPrefix,
            "T98 cache fill gate");
    fill.request.reset();
    std::array<std::shared_ptr<const Request>,8> initial{};
    for (int request = 0; request < 8; ++request) {
        auto prepared = cache.prepare(*contexts[request & 1],
            make_prompt(request, false), kPrefix, available(), 1024);
        cuda_check(cudaDeviceSynchronize(),
                   "T98 initial preparation completion");
        require(prepared.metrics.cache_hit &&
                    prepared.metrics.reused_prompt_tokens == kPrefix &&
                    prepared.metrics.executed_prompt_tokens ==
                        static_cast<std::size_t>(kInitialTails[request]),
                "T98 initial preparation gate");
        initial[request] = std::move(prepared.request);
    }

    std::ofstream arms_out(output / "arms.csv");
    std::ofstream rows_out(output / "rows.csv");
    std::ofstream lifecycle(output / "lifecycle.csv");
    require(arms_out.good() && rows_out.good() && lifecycle.good(),
            "T98 evidence open");
    arms_out << "pair,order,arm,pipelined,arrival_turnover_ms,total_wall_ms,"
        "authoritative_output_s,preparation_ms,execute_publish_ms,"
        "resident_registry_ms,resident_updates,reused_rows,prompt_executed_rows,"
        "verification_rows,executed_rows,replay_rows,committed_rows,"
        "replacement_ttft_p95_ms,resident_payload_bytes,locked_payload_bytes,"
        "locked_page_bytes,device_free_bytes,exact\n";
    rows_out << "pair,arm,request_kind,request,tail_rows,token,"
        "replacement_ttft_ms,verification_rows,executed_rows,replay_rows,"
        "committed_rows,exact\n";
    lifecycle << "case,queued,active,admitted,publications,cancellations,"
        "completions,rejected,pass\n";

    const auto run_arm = [&](bool pipelined) {
        T98ArmTrace result;
        Coordinator coordinator(cache, Coordinator::Policy{
            8, 2, memory.ullTotalPhys - reserve, reserve});
        for (const auto& root : initial) (void)coordinator.admit(root);
        std::array<Coordinator::Lease,2> active{};
        for (int lane = 0; lane < 2; ++lane) {
            const auto lease = coordinator.acquire();
            require(lease.has_value(), "T98 initial acquire");
            active[lane] = *lease;
        }
        std::array<std::shared_ptr<const Request>,8> replacement_roots{};
        std::array<Clock::time_point,8> arrivals{};
        const auto execute_one = [&](int lane, int output_index) {
            const auto started = Clock::now();
            contexts[lane]->restore_exact_host_state(
                *active[lane].root->state(), streams[lane].value);
            cuda_check(cudaStreamSynchronize(streams[lane].value),
                       "T98 request restore");
            const auto pending = sample_target(
                *contexts[lane], streams[lane].value);
            const std::array<std::int64_t,1> token{pending};
            auto [updated, verified] = active[lane].root->verify(
                *contexts[lane], token, {}, streams[lane].value);
            require(verified.committed_tokens.size() == 1 &&
                        verified.committed_tokens.front() == pending &&
                        verified.accepted == 1 && !verified.rejected &&
                        !verified.stopped && verified.verification_rows == 1 &&
                        verified.executed_rows == 1 &&
                        verified.replay_rows == 0 &&
                        verified.native_invocations == 1 &&
                        verified.root_restores == 0,
                    "T98 exact width-one publication");
            const auto published = coordinator.publish(
                active[lane], updated, pending);
            active[lane] = published.lease;
            result.tokens[output_index] = pending;
            result.roots[output_index] = published.lease.root;
            result.verification_rows += verified.verification_rows;
            result.executed_rows += verified.executed_rows;
            result.replay_rows += verified.replay_rows;
            result.committed_rows += verified.committed_tokens.size();
            result.execute_publish_ms +=
                std::chrono::duration<double,std::milli>(
                    Clock::now() - started).count();
        };
        const auto batch_started = Clock::now();
        const auto before_registry = coordinator.stats();
        for (int cycle = 0; cycle < 8; ++cycle) {
            const int execute_lane = cycle & 1;
            const int prepare_lane = 1 - execute_lane;
            arrivals[cycle] = Clock::now();
            Cache::Metrics preparation{};
            double preparation_ms = 0.0;
            std::exception_ptr execute_error, prepare_error;
            if (pipelined) {
                std::thread execution([&] {
                    try { execute_one(execute_lane, cycle); }
                    catch (...) { execute_error = std::current_exception(); }
                });
                std::thread arrival([&] {
                    try {
                        const auto started = Clock::now();
                        auto prepared = cache.prepare_concurrent(
                            *contexts[prepare_lane], replacement_prompts[cycle],
                            kPrefix, available(), 1024);
                        cuda_check(cudaDeviceSynchronize(),
                                   "T98 pipelined preparation completion");
                        preparation_ms =
                            std::chrono::duration<double,std::milli>(
                                Clock::now() - started).count();
                        preparation = prepared.metrics;
                        replacement_roots[cycle] = std::move(prepared.request);
                    } catch (...) { prepare_error = std::current_exception(); }
                });
                execution.join();
                arrival.join();
                if (execute_error) std::rethrow_exception(execute_error);
                if (prepare_error) std::rethrow_exception(prepare_error);
                const auto transition = coordinator.complete_and_admit(
                    active[execute_lane], replacement_roots[cycle]);
                require(transition.admitted.request_id ==
                            static_cast<std::uint64_t>(9 + cycle),
                        "T98 atomic replacement identity");
            } else {
                execute_one(execute_lane, cycle);
                coordinator.complete(active[execute_lane]);
                const auto started = Clock::now();
                auto prepared = cache.prepare(
                    *contexts[prepare_lane], replacement_prompts[cycle],
                    kPrefix, available(), 1024);
                cuda_check(cudaDeviceSynchronize(),
                           "T98 sequential preparation completion");
                preparation_ms = std::chrono::duration<double,std::milli>(
                    Clock::now() - started).count();
                preparation = prepared.metrics;
                replacement_roots[cycle] = std::move(prepared.request);
                const auto ticket = coordinator.admit(replacement_roots[cycle]);
                require(ticket.request_id == static_cast<std::uint64_t>(9 + cycle),
                        "T98 sequential replacement identity");
            }
            require(preparation.cache_hit &&
                        preparation.reused_prompt_tokens == kPrefix &&
                        preparation.executed_prompt_tokens ==
                            static_cast<std::size_t>(kReplacementTails[cycle]),
                    "T98 replacement preparation accounting");
            result.preparation_ms += preparation_ms;
            result.reused_rows += preparation.reused_prompt_tokens;
            result.prompt_executed_rows += preparation.executed_prompt_tokens;
            const auto next = coordinator.acquire();
            require(next.has_value(), "T98 turnover acquire");
            active[execute_lane] = *next;
        }
        result.arrival_turnover_ms = std::chrono::duration<double,std::milli>(
            Clock::now() - batch_started).count();
        const auto after_registry = coordinator.stats();
        result.resident_registry_ms = after_registry.resident_registry_ms -
                                      before_registry.resident_registry_ms;
        result.resident_updates = after_registry.resident_updates -
                                  before_registry.resident_updates;
        require(after_registry.active == 2 && after_registry.queued == 6 &&
                    after_registry.admitted == 8 &&
                    result.resident_updates == static_cast<std::uint64_t>(
                        pipelined ? 16 : 24),
                "T98 turnover lifecycle and update count");

        auto resident_roots = cache.roots();
        for (const auto& root : replacement_roots)
            resident_roots.push_back(root);
        Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(resident_roots,
            [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
            false);
        const auto observed = probe.measure();
        result.resident_payload_bytes = observed.resident_tensor_bytes;
        result.locked_payload_bytes = observed.locked_tensor_bytes;
        result.locked_page_bytes = after_registry.locked_page_bytes;
        std::size_t total_bytes = 0;
        cuda_check(cudaMemGetInfo(&result.device_free_bytes, &total_bytes),
                   "T98 device memory query");
        require(observed.resident_tensor_bytes == observed.allocated_union_bytes &&
                    observed.locked_tensor_bytes == observed.allocated_union_bytes &&
                    result.device_free_bytes >= (1ULL << 30),
                "T98 resident and device memory gate");

        for (int cycle = 0; cycle < 8; ++cycle) {
            const int lane = cycle & 1;
            const int replacement = static_cast<int>(
                active[lane].ticket.request_id - 9);
            require(replacement >= 0 && replacement < 8,
                    "T98 replacement FIFO identity");
            execute_one(lane, 8 + replacement);
            result.replacement_ttft_ms[replacement] =
                std::chrono::duration<double,std::milli>(
                    Clock::now() - arrivals[replacement]).count();
            coordinator.complete(active[lane]);
            if (cycle < 6) {
                const auto next = coordinator.acquire();
                require(next.has_value(), "T98 replacement drain acquire");
                active[lane] = *next;
            }
        }
        result.total_wall_ms = std::chrono::duration<double,std::milli>(
            Clock::now() - batch_started).count();
        require(result.reused_rows == 30720 &&
                    result.prompt_executed_rows == 768 &&
                    result.verification_rows == 16 &&
                    result.executed_rows == 16 && result.replay_rows == 0 &&
                    result.committed_rows == 16 &&
                    coordinator.stats().active == 0 &&
                    coordinator.stats().queued == 0 &&
                    coordinator.stats().admitted == 0 &&
                    coordinator.stats().publications == 16 &&
                    coordinator.stats().completions == 16,
                "T98 complete work accounting");
        coordinator.close();
        return result;
    };

    std::array<std::shared_ptr<const Request>,16> reference_roots{};
    std::array<std::int64_t,16> reference_tokens{};
    bool reference_ready = false;
    for (int pair = 0; pair < kPairs; ++pair) {
        const bool pipeline_first = (pair & 1) != 0;
        std::array<T98ArmTrace,2> result{};
        for (int ordinal = 0; ordinal < 2; ++ordinal) {
            const bool pipelined = pipeline_first ? ordinal == 0 : ordinal == 1;
            const int slot = pipelined ? 1 : 0;
            result[slot] = run_arm(pipelined);
            bool exact = true;
            if (!reference_ready) {
                reference_roots = result[slot].roots;
                reference_tokens = result[slot].tokens;
                reference_ready = true;
            } else {
                for (int request = 0; request < 16; ++request)
                    exact = exact && result[slot].tokens[request] ==
                        reference_tokens[request] &&
                        result[slot].roots[request]->token_suffix() ==
                            reference_roots[request]->token_suffix() &&
                        result[slot].roots[request]->same_taps(
                            *reference_roots[request]) &&
                        result[slot].roots[request]->state()->same_payload(
                            *reference_roots[request]->state());
            }
            require(exact, "T98 cross-arm exact publication state");
            auto ttft = result[slot].replacement_ttft_ms;
            std::sort(ttft.begin(), ttft.end());
            arms_out << pair << ',' << (pipeline_first ? "PS" : "SP") << ','
                << (pipelined ? "PIPELINED" : "SEQUENTIAL") << ','
                << pipelined << ',' << result[slot].arrival_turnover_ms << ','
                << result[slot].total_wall_ms << ','
                << 16000.0 / result[slot].total_wall_ms << ','
                << result[slot].preparation_ms << ','
                << result[slot].execute_publish_ms << ','
                << result[slot].resident_registry_ms << ','
                << result[slot].resident_updates << ','
                << result[slot].reused_rows << ','
                << result[slot].prompt_executed_rows << ','
                << result[slot].verification_rows << ','
                << result[slot].executed_rows << ','
                << result[slot].replay_rows << ','
                << result[slot].committed_rows << ',' << ttft.back() << ','
                << result[slot].resident_payload_bytes << ','
                << result[slot].locked_payload_bytes << ','
                << result[slot].locked_page_bytes << ','
                << result[slot].device_free_bytes << ",1\n";
            for (int request = 0; request < 16; ++request) {
                const bool replacement = request >= 8;
                const int logical = replacement ? request - 8 : request;
                rows_out << pair << ','
                    << (pipelined ? "PIPELINED" : "SEQUENTIAL") << ','
                    << (replacement ? "replacement" : "initial") << ','
                    << logical << ','
                    << (replacement ? kReplacementTails[logical] :
                                      kInitialTails[logical]) << ','
                    << result[slot].tokens[request] << ','
                    << (replacement ?
                        result[slot].replacement_ttft_ms[logical] : 0.0)
                    << ",1,1,0,1,1\n";
            }
        }
        for (int request = 0; request < 16; ++request)
            require(result[0].tokens[request] == result[1].tokens[request] &&
                        result[0].roots[request]->token_suffix() ==
                            result[1].roots[request]->token_suffix() &&
                        result[0].roots[request]->same_taps(
                            *result[1].roots[request]) &&
                        result[0].roots[request]->state()->same_payload(
                            *result[1].roots[request]->state()),
                    "T98 matched pair exact equality");
    }

    Coordinator cancellation(cache, Coordinator::Policy{
        8, 2, memory.ullTotalPhys - reserve, reserve});
    const auto first_ticket = cancellation.admit(initial[0]);
    const auto second_ticket = cancellation.admit(initial[1]);
    const auto first = cancellation.acquire();
    const auto second = cancellation.acquire();
    require(first.has_value() && second.has_value(),
            "T98 cancellation physical acquire");
    const auto before_unadmitted = cancellation.stats();
    auto unadmitted = cache.prepare_concurrent(
        *context_b, replacement_prompts[0], kPrefix, available(), 1024);
    const auto after_unadmitted = cancellation.stats();
    const bool unadmitted_unchanged =
        before_unadmitted.queued == after_unadmitted.queued &&
        before_unadmitted.active == after_unadmitted.active &&
        before_unadmitted.admitted == after_unadmitted.admitted &&
        before_unadmitted.resident_updates == after_unadmitted.resident_updates;
    unadmitted.request.reset();
    lifecycle << "prepared_not_admitted," << after_unadmitted.queued << ','
        << after_unadmitted.active << ',' << after_unadmitted.admitted << ','
        << after_unadmitted.publications << ',' << after_unadmitted.cancellations
        << ',' << after_unadmitted.completions << ",1,"
        << unadmitted_unchanged << '\n';
    require(unadmitted_unchanged, "T98 unadmitted cancellation");
    contexts[0]->restore_exact_host_state(
        *first->root->state(), streams[0].value);
    const auto pending = sample_target(*contexts[0], streams[0].value);
    const std::array<std::int64_t,1> token{pending};
    auto [late_root, verified] = first->root->verify(
        *contexts[0], token, {}, streams[0].value);
    require(verified.verification_rows == 1,
            "T98 active cancellation worker completion");
    cancellation.cancel(first_ticket, true);
    bool stale_rejected = false;
    try { (void)cancellation.publish(*first, late_root, pending); }
    catch (const std::invalid_argument&) { stale_rejected = true; }
    cancellation.complete(*second);
    const auto canceled = cancellation.stats();
    lifecycle << "active_after_numerical," << canceled.queued << ','
        << canceled.active << ',' << canceled.admitted << ','
        << canceled.publications << ',' << canceled.cancellations << ','
        << canceled.completions << ',' << stale_rejected << ','
        << stale_rejected << '\n';
    require(stale_rejected && canceled.active == 0 &&
                canceled.admitted == 0 && canceled.cancellations == 1 &&
                canceled.completions == 1,
            "T98 active cancellation stale publication");
    cancellation.close();
    const auto storage = cache.storage_stats();
    require(storage.entries == 1 && storage.accounted_bytes <= cache_budget,
            "T98 final cache gate");
    arms_out.flush();
    rows_out.flush();
    lifecycle.flush();
    require(arms_out.good() && rows_out.good() && lifecycle.good(),
            "T98 evidence flush");
    std::cout << "T98_PIPELINED_TURNOVER PASS pairs=4 arms=8 publications=128"
        " logical_capacity=8 physical_contexts=2 exact=1 lifecycle=2 output="
        << output.string() << std::endl;
}
