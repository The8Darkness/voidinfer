#pragma once

struct T95ArmTrace {
    double turnover_wall_ms = 0.0;
    double resident_registry_ms = 0.0;
    std::uint64_t resident_updates = 0;
    std::uint64_t new_locked_bytes = 0;
    std::uint64_t unlocked_bytes = 0;
    std::uint64_t resident_payload_bytes = 0;
    std::uint64_t locked_payload_bytes = 0;
    std::uint64_t locked_page_bytes = 0;
    std::uint64_t device_free_bytes = 0;
    std::uint64_t verification_rows = 0;
    std::uint64_t executed_rows = 0;
    std::uint64_t replay_rows = 0;
    std::uint64_t committed_rows = 0;
    std::array<std::int64_t,8> tokens{};
    std::array<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>,8>
        roots{};
};

void run_t95_atomic_turnover(
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
            "T95 serving fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T95 output must be new");
    std::filesystem::create_directories(output);

    const auto reserve_gib = std::stoull(env("NINFER_T95_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T95_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024,
            "T95 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 &&
                memory.ullTotalPhys > reserve && memory.ullAvailPhys > reserve,
            "T95 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX value{};
        value.dwLength = sizeof(value);
        require(GlobalMemoryStatusEx(&value) != 0,
                "T95 physical memory query");
        return value.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T95_NAMESPACE"), env("NINFER_T95_ARTIFACT_ID"),
        env("NINFER_T95_TOKENIZER_ID"), env("NINFER_T95_CONFIGURATION_ID"),
        env("NINFER_T95_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{32, 64, cache_budget, reserve}, identity);
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

    auto fill = cache.prepare(*context_a, common, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T95 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                fill.metrics.executed_prompt_tokens == kPrefix,
            "T95 cache fill gate");
    fill.request.reset();
    std::array<std::shared_ptr<const Request>,8> initial{};
    std::array<std::shared_ptr<const Request>,8> replacement{};
    for (int request = 0; request < 8; ++request) {
        auto first = cache.prepare(*contexts[request & 1],
            make_prompt(request, false), kPrefix, available(), 1024);
        cuda_check(cudaDeviceSynchronize(),
                   "T95 initial preparation completion");
        require(first.metrics.cache_hit &&
                    first.metrics.reused_prompt_tokens == kPrefix &&
                    first.metrics.executed_prompt_tokens ==
                        static_cast<std::size_t>(kInitialTails[request]),
                "T95 initial preparation gate");
        initial[request] = std::move(first.request);
        auto second = cache.prepare(*contexts[request & 1],
            make_prompt(request, true), kPrefix, available(), 1024);
        cuda_check(cudaDeviceSynchronize(),
                   "T95 replacement preparation completion");
        require(second.metrics.cache_hit &&
                    second.metrics.reused_prompt_tokens == kPrefix &&
                    second.metrics.executed_prompt_tokens ==
                        static_cast<std::size_t>(kReplacementTails[request]),
                "T95 replacement preparation gate");
        replacement[request] = std::move(second.request);
    }

    std::ofstream arms_out(output / "arms.csv");
    std::ofstream rows_out(output / "rows.csv");
    std::ofstream failures(output / "failure_atomicity.csv");
    require(arms_out.good() && rows_out.good() && failures.good(),
            "T95 evidence open");
    arms_out << "pair,order,arm,atomic,turnover_wall_ms,resident_registry_ms,"
        "resident_updates,new_locked_bytes,unlocked_bytes,verification_rows,"
        "executed_rows,replay_rows,committed_rows,resident_payload_bytes,"
        "locked_payload_bytes,locked_page_bytes,device_free_bytes,exact\n";
    rows_out << "pair,arm,replacement,tail_rows,request_id,token,"
        "verification_rows,executed_rows,replay_rows,committed_rows,exact\n";
    failures << "pair,arm,case,queued_before,active_before,admitted_before,"
        "updates_before,queued_after,active_after,admitted_after,updates_after,"
        "rejected,unchanged\n";

    std::array<std::shared_ptr<const Request>,8> reference_roots{};
    std::array<std::int64_t,8> reference_tokens{};
    bool reference_ready = false;
    const auto run_arm = [&](bool atomic, int pair) {
        T95ArmTrace result;
        Coordinator coordinator(cache, Coordinator::Policy{
            8, 2, memory.ullTotalPhys - reserve, reserve});
        std::array<Coordinator::Ticket,8> initial_tickets{};
        for (int request = 0; request < 8; ++request)
            initial_tickets[request] = coordinator.admit(initial[request]);
        std::array<Coordinator::Lease,2> active{};
        for (int lane = 0; lane < 2; ++lane) {
            const auto lease = coordinator.acquire();
            require(lease.has_value(), "T95 initial acquire");
            active[lane] = *lease;
        }
        require(!coordinator.acquire().has_value() &&
                    coordinator.stats().active == 2,
                "T95 physical C2 boundary");
        const auto before_turnover = coordinator.stats();
        for (int cycle = 0; cycle < 8; ++cycle) {
            const int lane = cycle & 1;
            const auto retired = active[lane];
            if (cycle == 0) {
                const auto before = coordinator.stats();
                bool rejected = false;
                try {
                    (void)coordinator.complete_and_admit(retired, {});
                } catch (const std::invalid_argument&) { rejected = true; }
                const auto after = coordinator.stats();
                const bool unchanged = before.queued == after.queued &&
                    before.active == after.active &&
                    before.admitted == after.admitted &&
                    before.resident_updates == after.resident_updates;
                failures << pair << ',' << (atomic ? "ATOMIC" : "SEQUENTIAL")
                    << ",null_root," << before.queued << ',' << before.active
                    << ',' << before.admitted << ',' << before.resident_updates
                    << ',' << after.queued << ',' << after.active << ','
                    << after.admitted << ',' << after.resident_updates << ','
                    << rejected << ',' << unchanged << '\n';
                require(rejected && unchanged,
                        "T95 null turnover atomicity");
            }
            Coordinator::Ticket admitted{};
            const auto started = Clock::now();
            if (atomic) {
                const auto transition = coordinator.complete_and_admit(
                    retired, replacement[cycle]);
                require(transition.completed.request_id ==
                            retired.ticket.request_id &&
                            transition.completed.generation ==
                                retired.ticket.generation,
                        "T95 completed ticket identity");
                admitted = transition.admitted;
            } else {
                coordinator.complete(retired);
                admitted = coordinator.admit(replacement[cycle]);
            }
            result.turnover_wall_ms +=
                std::chrono::duration<double,std::milli>(
                    Clock::now() - started).count();
            require(admitted.request_id == static_cast<std::uint64_t>(9 + cycle)
                        && admitted.generation == 1,
                    "T95 replacement ticket identity");
            if (cycle == 0) {
                const auto before = coordinator.stats();
                bool rejected = false;
                try {
                    (void)coordinator.complete_and_admit(
                        retired, replacement[cycle]);
                } catch (const std::invalid_argument&) { rejected = true; }
                const auto after = coordinator.stats();
                const bool unchanged = before.queued == after.queued &&
                    before.active == after.active &&
                    before.admitted == after.admitted &&
                    before.resident_updates == after.resident_updates;
                failures << pair << ',' << (atomic ? "ATOMIC" : "SEQUENTIAL")
                    << ",stale_lease," << before.queued << ',' << before.active
                    << ',' << before.admitted << ',' << before.resident_updates
                    << ',' << after.queued << ',' << after.active << ','
                    << after.admitted << ',' << after.resident_updates << ','
                    << rejected << ',' << unchanged << '\n';
                require(rejected && unchanged,
                        "T95 stale turnover atomicity");
            }
            const auto next = coordinator.acquire();
            require(next.has_value(), "T95 turnover FIFO acquire");
            const std::uint64_t expected = cycle < 6 ?
                static_cast<std::uint64_t>(3 + cycle) :
                static_cast<std::uint64_t>(9 + cycle - 6);
            require(next->ticket.request_id == expected,
                    "T95 turnover FIFO identity");
            active[lane] = *next;
        }
        const auto after_turnover = coordinator.stats();
        result.resident_registry_ms = after_turnover.resident_registry_ms -
                                      before_turnover.resident_registry_ms;
        result.resident_updates = after_turnover.resident_updates -
                                  before_turnover.resident_updates;
        result.new_locked_bytes = after_turnover.resident_new_locked_bytes -
                                  before_turnover.resident_new_locked_bytes;
        result.unlocked_bytes = after_turnover.resident_unlocked_bytes -
                                before_turnover.resident_unlocked_bytes;
        require(result.resident_updates == static_cast<std::uint64_t>(
                    atomic ? 8 : 16) && after_turnover.active == 2 &&
                    after_turnover.queued == 6 &&
                    after_turnover.admitted == 8,
                "T95 resident update cardinality");

        auto resident_roots = cache.roots();
        for (const auto& root : replacement) resident_roots.push_back(root);
        Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(resident_roots,
            [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
            false);
        const auto observed = probe.measure();
        result.resident_payload_bytes = observed.resident_tensor_bytes;
        result.locked_payload_bytes = observed.locked_tensor_bytes;
        result.locked_page_bytes = after_turnover.locked_page_bytes;
        std::size_t total_bytes = 0;
        cuda_check(cudaMemGetInfo(&result.device_free_bytes, &total_bytes),
                   "T95 device memory query");
        require(result.resident_payload_bytes ==
                    observed.allocated_union_bytes &&
                    result.locked_payload_bytes ==
                    observed.allocated_union_bytes &&
                    result.device_free_bytes >= (1ULL << 30),
                "T95 resident and device memory gate");

        for (int cycle = 0; cycle < 8; ++cycle) {
            const int lane = cycle & 1;
            const int request = static_cast<int>(
                active[lane].ticket.request_id - 9);
            require(request >= 0 && request < 8,
                    "T95 replacement drain identity");
            contexts[lane]->restore_exact_host_state(
                *active[lane].root->state(), streams[lane].value);
            cuda_check(cudaStreamSynchronize(streams[lane].value),
                       "T95 replacement restore");
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
                    "T95 replacement publication work");
            const auto published = coordinator.publish(
                active[lane], updated, pending);
            result.tokens[request] = pending;
            result.roots[request] = published.lease.root;
            result.verification_rows += verified.verification_rows;
            result.executed_rows += verified.executed_rows;
            result.replay_rows += verified.replay_rows;
            result.committed_rows += verified.committed_tokens.size();
            coordinator.complete(published.lease);
            if (cycle < 6) {
                const auto next = coordinator.acquire();
                require(next.has_value(), "T95 replacement drain acquire");
                active[lane] = *next;
            }
        }
        require(coordinator.stats().active == 0 &&
                    coordinator.stats().queued == 0 &&
                    coordinator.stats().admitted == 0 &&
                    coordinator.stats().completions == 16,
                "T95 completed lifecycle");
        const auto queued = coordinator.admit(initial[0]);
        coordinator.cancel(queued);
        const auto active_ticket = coordinator.admit(initial[1]);
        const auto cancel_lease = coordinator.acquire();
        require(cancel_lease.has_value() &&
                    cancel_lease->ticket.request_id == active_ticket.request_id,
                "T95 cancellation acquire");
        coordinator.cancel(cancel_lease->ticket, true);
        require(coordinator.stats().cancellations == 2 &&
                    coordinator.stats().admitted == 0,
                "T95 cancellation lifecycle");
        coordinator.close();
        require(coordinator.stats().closed,
                "T95 empty coordinator close");
        return result;
    };

    for (int pair = 0; pair < kPairs; ++pair) {
        const bool atomic_first = (pair & 1) != 0;
        std::array<T95ArmTrace,2> result{};
        for (int ordinal = 0; ordinal < 2; ++ordinal) {
            const bool atomic = atomic_first ? ordinal == 0 : ordinal == 1;
            auto trace = run_arm(atomic, pair);
            const int slot = atomic ? 1 : 0;
            result[slot] = std::move(trace);
            bool exact = true;
            if (!reference_ready) {
                reference_tokens = result[slot].tokens;
                reference_roots = result[slot].roots;
                reference_ready = true;
            } else {
                for (int request = 0; request < 8; ++request)
                    exact = exact &&
                        result[slot].tokens[request] ==
                            reference_tokens[request] &&
                        result[slot].roots[request]->token_suffix() ==
                            reference_roots[request]->token_suffix() &&
                        result[slot].roots[request]->same_taps(
                            *reference_roots[request]) &&
                        result[slot].roots[request]->state()->same_payload(
                            *reference_roots[request]->state());
            }
            require(exact, "T95 matched authoritative state");
            arms_out << pair << ',' << (atomic_first ? "AS" : "SA") << ','
                << (atomic ? "ATOMIC" : "SEQUENTIAL") << ',' << atomic << ','
                << result[slot].turnover_wall_ms << ','
                << result[slot].resident_registry_ms << ','
                << result[slot].resident_updates << ','
                << result[slot].new_locked_bytes << ','
                << result[slot].unlocked_bytes << ','
                << result[slot].verification_rows << ','
                << result[slot].executed_rows << ','
                << result[slot].replay_rows << ','
                << result[slot].committed_rows << ','
                << result[slot].resident_payload_bytes << ','
                << result[slot].locked_payload_bytes << ','
                << result[slot].locked_page_bytes << ','
                << result[slot].device_free_bytes << ",1\n";
            for (int request = 0; request < 8; ++request)
                rows_out << pair << ','
                    << (atomic ? "ATOMIC" : "SEQUENTIAL") << ',' << request
                    << ',' << kReplacementTails[request] << ','
                    << (9 + request) << ',' << result[slot].tokens[request]
                    << ",1,1,0,1,1\n";
        }
        for (int request = 0; request < 8; ++request)
            require(result[0].tokens[request] == result[1].tokens[request] &&
                        result[0].roots[request]->token_suffix() ==
                            result[1].roots[request]->token_suffix() &&
                        result[0].roots[request]->same_taps(
                            *result[1].roots[request]) &&
                        result[0].roots[request]->state()->same_payload(
                            *result[1].roots[request]->state()),
                    "T95 pair exact equality");
    }
    const auto storage = cache.storage_stats();
    require(storage.entries == 1 && storage.accounted_bytes <= cache_budget,
            "T95 final cache gate");
    arms_out.flush();
    rows_out.flush();
    failures.flush();
    require(arms_out.good() && rows_out.good() && failures.good(),
            "T95 evidence flush");
    std::cout << "T95_ATOMIC_TURNOVER PASS pairs=4 arms=8 replacements=64"
        " logical_capacity=8 physical_capacity=2 exact=1 output="
        << output.string() << std::endl;
}
