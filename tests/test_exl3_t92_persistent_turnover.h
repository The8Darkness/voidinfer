#pragma once

#include "exl3/vericache_serving_coordinator.h"
#include "test_exl3_host_residency.h"

#include <array>
#include <chrono>
#include <fstream>

void run_t92_persistent_turnover(
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
    require(target.max_context() >= 4352 && code.size() >= 4096 &&
                prose.size() >= 2048,
            "T92 serving fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T92 output must be new");
    std::filesystem::create_directories(output);

    const auto reserve_gib = std::stoull(env("NINFER_T92_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T92_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024,
            "T92 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 &&
                memory.ullTotalPhys > reserve && memory.ullAvailPhys > reserve,
            "T92 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX value{};
        value.dwLength = sizeof(value);
        require(GlobalMemoryStatusEx(&value) != 0,
                "T92 physical memory query");
        return value.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T92_NAMESPACE"), env("NINFER_T92_ARTIFACT_ID"),
        env("NINFER_T92_TOKENIZER_ID"), env("NINFER_T92_CONFIGURATION_ID"),
        env("NINFER_T92_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{16, 64, cache_budget, reserve}, identity);
    Coordinator coordinator(cache, Coordinator::Policy{
        8, 2, memory.ullTotalPhys - reserve, reserve});
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
        require(prompt.size() == kPrefix + static_cast<std::size_t>(rows),
                "T92 prompt extent");
        return prompt;
    };

    auto fill = cache.prepare(*context_a, common, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T92 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                fill.metrics.executed_prompt_tokens == kPrefix,
            "T92 cache fill gate");
    fill.request.reset();

    std::array<std::shared_ptr<const Request>,8> initial{};
    std::vector<std::pair<std::uint64_t,std::shared_ptr<const Request>>> live;
    std::array<Coordinator::Ticket,8> initial_tickets{};
    for (int request = 0; request < 8; ++request) {
        const auto prompt = make_prompt(request, false);
        auto prepared = cache.prepare(*contexts[request & 1], prompt, kPrefix,
                                      available(), 1024);
        cuda_check(cudaDeviceSynchronize(),
                   "T92 initial request preparation completion");
        require(prepared.metrics.cache_hit &&
                    prepared.metrics.reused_prompt_tokens == kPrefix &&
                    prepared.metrics.executed_prompt_tokens ==
                        static_cast<std::size_t>(kInitialTails[request]),
                "T92 initial cached request preparation");
        initial[request] = std::move(prepared.request);
        initial_tickets[request] = coordinator.admit(initial[request]);
        live.emplace_back(initial_tickets[request].request_id, initial[request]);
    }
    require(coordinator.stats().admitted == 8 &&
                coordinator.stats().queued == 8,
            "T92 initial logical C8 admission");

    struct ReplacementTrace {
        int tail_rows = 0;
        std::uint64_t request_id = 0;
        std::uint64_t admitted_generation = 0;
        std::uint64_t published_generation = 0;
        Cache::Metrics preparation{};
        double preparation_ms = 0.0;
        double completion_resident_ms = 0.0;
        double admission_resident_ms = 0.0;
        double turnover_wall_ms = 0.0;
        double acquire_wait_ms = 0.0;
        double resident_ttft_ms = 0.0;
        double cached_ttft_ms = 0.0;
        double restore_ms = 0.0;
        double publication_ms = 0.0;
        std::uint64_t completion_new_locked_bytes = 0;
        std::uint64_t completion_unlocked_bytes = 0;
        std::uint64_t admission_new_locked_bytes = 0;
        std::uint64_t admission_unlocked_bytes = 0;
        std::uint64_t verification_rows = 0;
        std::uint64_t executed_rows = 0;
        std::uint64_t replay_rows = 0;
        std::uint64_t committed_rows = 0;
        std::uint64_t native_invocations = 0;
        std::uint64_t root_restores = 0;
        std::int64_t token = -1;
        Clock::time_point arrival{};
        Clock::time_point admitted{};
        std::shared_ptr<const Request> parent;
        std::shared_ptr<const Request> published;
    };
    std::array<ReplacementTrace,8> traces{};
    const auto replacement_index = [&](std::uint64_t id) {
        for (int index = 0; index < 8; ++index)
            if (traces[index].request_id == id) return index;
        return -1;
    };
    const auto replace_live = [&](std::uint64_t id,
                                  std::shared_ptr<const Request> root) {
        for (auto& entry : live)
            if (entry.first == id) {
                entry.second = std::move(root);
                return;
            }
        throw std::invalid_argument("T92 live request missing");
    };
    const auto erase_live = [&](std::uint64_t id) {
        const auto found = std::find_if(live.begin(), live.end(),
            [&](const auto& entry) { return entry.first == id; });
        require(found != live.end(), "T92 live erase identity");
        live.erase(found);
    };

    std::ofstream turnover(output / "turnover.csv");
    std::ofstream lifecycle(output / "lifecycle.csv");
    std::ofstream memory_out(output / "memory.csv");
    std::ofstream correctness(output / "correctness.csv");
    require(turnover.good() && lifecycle.good() && memory_out.good() &&
                correctness.good(),
            "T92 evidence open");
    turnover << "replacement,tail_rows,request_id,admitted_generation,"
        "published_generation,reused_rows,executed_prompt_rows,preparation_ms,"
        "completion_resident_ms,admission_resident_ms,resident_sum_ms,"
        "turnover_wall_ms,resident_share_percent,acquire_wait_ms,"
        "resident_ttft_ms,cached_prefix_ttft_ms,restore_ms,publication_ms,"
        "completion_new_locked_bytes,completion_unlocked_bytes,"
        "admission_new_locked_bytes,admission_unlocked_bytes,verification_rows,"
        "executed_rows,replay_rows,committed_rows,native_invocations,"
        "root_restores,token,exact\n";
    lifecycle << "phase,queued,active,admitted,publications,cancellations,"
        "completions,resident_updates,resident_registry_ms,locked_page_bytes,"
        "high_queued,high_active,high_admitted,pass\n";
    memory_out << "phase,cache_entries,cache_payload_bytes,cache_identity_bytes,"
        "cache_accounted_bytes,resident_payload_bytes,locked_payload_bytes,"
        "resident_page_bytes,physical_reserve_bytes,device_free_bytes\n";
    correctness << "case,replacement,request_id,token_match,suffix_match,"
        "taps_match,state_match,pass\n";
    const auto write_lifecycle = [&](const char* phase, bool pass) {
        const auto stats = coordinator.stats();
        lifecycle << phase << ',' << stats.queued << ',' << stats.active << ','
            << stats.admitted << ',' << stats.publications << ','
            << stats.cancellations << ',' << stats.completions << ','
            << stats.resident_updates << ',' << stats.resident_registry_ms << ','
            << stats.locked_page_bytes << ',' << coordinator.high_queued() << ','
            << coordinator.high_active() << ',' << coordinator.high_admitted()
            << ',' << pass << '\n';
    };
    const auto observe = [&](const char* phase) {
        auto roots = cache.roots();
        for (const auto& entry : live) roots.push_back(entry.second);
        Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(roots,
            [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
            false);
        const auto observed = probe.measure();
        require(observed.resident_tensor_bytes == observed.allocated_union_bytes &&
                    observed.locked_tensor_bytes == observed.allocated_union_bytes,
                "T92 authoritative payload not resident and locked");
        std::size_t free_bytes = 0, total_bytes = 0;
        cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
                   "T92 device memory query");
        require(free_bytes >= (1ULL << 30), "T92 device memory gate");
        const auto storage = cache.storage_stats();
        require(storage.accounted_bytes <= cache_budget,
                "T92 cache byte budget");
        memory_out << phase << ',' << storage.entries << ','
            << storage.payload_allocated_bytes << ','
            << storage.identity_allocated_bytes << ',' << storage.accounted_bytes
            << ',' << observed.resident_tensor_bytes << ','
            << observed.locked_tensor_bytes << ','
            << coordinator.stats().locked_page_bytes << ',' << reserve << ','
            << free_bytes << '\n';
    };

    std::array<Coordinator::Lease,2> active{};
    for (int lane = 0; lane < 2; ++lane) {
        const auto lease = coordinator.acquire();
        require(lease.has_value(), "T92 initial physical C2 acquire");
        active[lane] = *lease;
    }
    require(!coordinator.acquire().has_value() &&
                coordinator.stats().active == 2 &&
                coordinator.high_active() == 2,
            "T92 physical C2 boundary");
    write_lifecycle("initial_c8_c2", true);
    observe("initial_c8_c2");

    bool stale_completion_rejected = false;
    const auto execute_one = [&](int lane) {
        auto& lease = active[lane];
        const auto restore_started = Clock::now();
        contexts[lane]->restore_exact_host_state(
            *lease.root->state(), streams[lane].value);
        cuda_check(cudaStreamSynchronize(streams[lane].value),
                   "T92 request restore");
        const double restore_ms = std::chrono::duration<double,std::milli>(
            Clock::now() - restore_started).count();
        const auto pending = sample_target(*contexts[lane], streams[lane].value);
        const std::array<std::int64_t,1> token{pending};
        const auto publication_started = Clock::now();
        auto [updated, verified] = lease.root->verify(
            *contexts[lane], token, {}, streams[lane].value);
        require(verified.committed_tokens.size() == 1 &&
                    verified.committed_tokens.front() == pending &&
                    verified.accepted == 1 && !verified.rejected &&
                    !verified.stopped && verified.verification_rows == 1 &&
                    verified.executed_rows == 1 && verified.replay_rows == 0 &&
                    verified.native_invocations == 1 &&
                    verified.root_restores == 0,
                "T92 exact width-one publication");
        const auto published = coordinator.publish(lease, updated, pending);
        const double publication_ms = std::chrono::duration<double,std::milli>(
            Clock::now() - publication_started).count();
        replace_live(lease.ticket.request_id, published.lease.root);
        const int replacement = replacement_index(lease.ticket.request_id);
        if (replacement >= 0) {
            auto& trace = traces[replacement];
            trace.restore_ms = restore_ms;
            trace.publication_ms = publication_ms;
            trace.resident_ttft_ms = std::chrono::duration<double,std::milli>(
                Clock::now() - trace.admitted).count();
            trace.cached_ttft_ms = std::chrono::duration<double,std::milli>(
                Clock::now() - trace.arrival).count();
            trace.published_generation = published.lease.ticket.generation;
            trace.verification_rows = verified.verification_rows;
            trace.executed_rows = verified.executed_rows;
            trace.replay_rows = verified.replay_rows;
            trace.committed_rows = verified.committed_tokens.size();
            trace.native_invocations = verified.native_invocations;
            trace.root_restores = verified.root_restores;
            trace.token = pending;
            trace.published = published.lease.root;
        }
        lease = published.lease;
    };

    for (int cycle = 0; cycle < 8; ++cycle) {
        const int lane = cycle & 1;
        execute_one(lane);
        const auto retired = active[lane];
        const auto turnover_started = Clock::now();
        const auto before_complete = coordinator.stats();
        coordinator.complete(retired);
        erase_live(retired.ticket.request_id);
        const auto after_complete = coordinator.stats();
        if (cycle == 0) {
            try { coordinator.complete(retired); }
            catch (const std::invalid_argument&) {
                stale_completion_rejected = true;
            }
        }

        auto& trace = traces[cycle];
        trace.tail_rows = kReplacementTails[cycle];
        trace.arrival = Clock::now();
        const auto prompt = make_prompt(cycle, true);
        const auto preparation_started = Clock::now();
        auto prepared = cache.prepare(*contexts[lane], prompt, kPrefix,
                                      available(), 1024);
        cuda_check(cudaDeviceSynchronize(),
                   "T92 replacement preparation completion");
        trace.preparation_ms = std::chrono::duration<double,std::milli>(
            Clock::now() - preparation_started).count();
        trace.preparation = prepared.metrics;
        require(prepared.metrics.cache_hit &&
                    prepared.metrics.reused_prompt_tokens == kPrefix &&
                    prepared.metrics.executed_prompt_tokens ==
                        static_cast<std::size_t>(trace.tail_rows),
                "T92 replacement cached request preparation");
        trace.parent = prepared.request;
        const auto before_admit = coordinator.stats();
        const auto ticket = coordinator.admit(std::move(prepared.request));
        trace.admitted = Clock::now();
        const auto after_admit = coordinator.stats();
        trace.request_id = ticket.request_id;
        trace.admitted_generation = ticket.generation;
        trace.completion_resident_ms = after_complete.resident_registry_ms -
                                       before_complete.resident_registry_ms;
        trace.admission_resident_ms = after_admit.resident_registry_ms -
                                      before_admit.resident_registry_ms;
        trace.completion_new_locked_bytes =
            after_complete.resident_new_locked_bytes -
            before_complete.resident_new_locked_bytes;
        trace.completion_unlocked_bytes =
            after_complete.resident_unlocked_bytes -
            before_complete.resident_unlocked_bytes;
        trace.admission_new_locked_bytes =
            after_admit.resident_new_locked_bytes -
            before_admit.resident_new_locked_bytes;
        trace.admission_unlocked_bytes =
            after_admit.resident_unlocked_bytes -
            before_admit.resident_unlocked_bytes;
        trace.turnover_wall_ms = std::chrono::duration<double,std::milli>(
            Clock::now() - turnover_started).count();
        live.emplace_back(ticket.request_id, trace.parent);

        const auto next = coordinator.acquire();
        require(next.has_value(), "T92 replacement physical acquire");
        active[lane] = *next;
        const int acquired_replacement = replacement_index(
            next->ticket.request_id);
        if (acquired_replacement >= 0)
            traces[acquired_replacement].acquire_wait_ms =
                std::chrono::duration<double,std::milli>(
                    Clock::now() - traces[acquired_replacement].admitted)
                    .count();
    }
    require(stale_completion_rejected && coordinator.stats().admitted == 8 &&
                coordinator.stats().active == 2 &&
                coordinator.stats().queued == 6,
            "T92 steady C8/C2 turnover state");
    write_lifecycle("after_eight_replacements", true);
    observe("after_eight_replacements");

    for (int cycle = 0; cycle < 8; ++cycle) {
        const int lane = cycle & 1;
        const int found = replacement_index(active[lane].ticket.request_id);
        require(found >= 0,
                "T92 replacement FIFO drain identity");
        if (traces[found].acquire_wait_ms == 0.0)
            traces[found].acquire_wait_ms =
                std::chrono::duration<double,std::milli>(
                    Clock::now() - traces[found].admitted).count();
        execute_one(lane);
        const auto retired = active[lane];
        coordinator.complete(retired);
        erase_live(retired.ticket.request_id);
        if (cycle + 2 < 8) {
            const auto next = coordinator.acquire();
            require(next.has_value(), "T92 replacement FIFO continuation");
            active[lane] = *next;
            const int next_found = replacement_index(next->ticket.request_id);
            require(next_found >= 0,
                    "T92 replacement acquire identity");
            traces[next_found].acquire_wait_ms =
                std::chrono::duration<double,std::milli>(
                    Clock::now() - traces[next_found].admitted).count();
        }
    }
    require(coordinator.stats().admitted == 0 &&
                coordinator.stats().active == 0 &&
                coordinator.stats().queued == 0 &&
                coordinator.stats().completions == 16,
            "T92 empty after FIFO drain");

    for (int replacement = 0; replacement < 8; ++replacement) {
        auto& trace = traces[replacement];
        const int lane = replacement & 1;
        contexts[lane]->restore_exact_host_state(
            *trace.parent->state(), streams[lane].value);
        cuda_check(cudaStreamSynchronize(streams[lane].value),
                   "T92 oracle parent restore");
        const auto oracle_pending = sample_target(
            *contexts[lane], streams[lane].value);
        const std::array<std::int64_t,1> oracle_token{oracle_pending};
        auto [oracle, verified] = trace.parent->verify(
            *contexts[lane], oracle_token, {}, streams[lane].value);
        const bool token_match = trace.token == oracle_pending;
        const bool suffix_match = trace.published->token_suffix() ==
                                  oracle->token_suffix();
        const bool taps_match = trace.published->same_taps(*oracle);
        const bool state_match = trace.published->state()->same_payload(
            *oracle->state());
        const bool exact = token_match && suffix_match && taps_match &&
                           state_match && verified.verification_rows == 1;
        correctness << "serial_oracle," << replacement << ','
            << trace.request_id << ',' << token_match << ',' << suffix_match
            << ',' << taps_match << ',' << state_match << ',' << exact << '\n';
        require(exact, "T92 replacement serial oracle");
        const double resident_sum = trace.completion_resident_ms +
                                    trace.admission_resident_ms;
        const double resident_share = trace.turnover_wall_ms > 0.0 ?
            100.0 * resident_sum / trace.turnover_wall_ms : 0.0;
        turnover << replacement << ',' << trace.tail_rows << ','
            << trace.request_id << ',' << trace.admitted_generation << ','
            << trace.published_generation << ','
            << trace.preparation.reused_prompt_tokens << ','
            << trace.preparation.executed_prompt_tokens << ','
            << trace.preparation_ms << ',' << trace.completion_resident_ms << ','
            << trace.admission_resident_ms << ',' << resident_sum << ','
            << trace.turnover_wall_ms << ',' << resident_share << ','
            << trace.acquire_wait_ms << ',' << trace.resident_ttft_ms << ','
            << trace.cached_ttft_ms << ',' << trace.restore_ms << ','
            << trace.publication_ms << ','
            << trace.completion_new_locked_bytes << ','
            << trace.completion_unlocked_bytes << ','
            << trace.admission_new_locked_bytes << ','
            << trace.admission_unlocked_bytes << ','
            << trace.verification_rows << ',' << trace.executed_rows << ','
            << trace.replay_rows << ',' << trace.committed_rows << ','
            << trace.native_invocations << ',' << trace.root_restores << ','
            << trace.token << ",1\n";
        trace.parent.reset();
        trace.published.reset();
    }

    const auto queued_ticket = coordinator.admit(initial[0]);
    coordinator.cancel(queued_ticket);
    const auto active_ticket = coordinator.admit(initial[1]);
    const auto canceled = coordinator.acquire();
    require(canceled.has_value() && canceled->ticket.request_id ==
                active_ticket.request_id,
            "T92 active cancellation acquire");
    contexts[0]->restore_exact_host_state(
        *canceled->root->state(), streams[0].value);
    const auto canceled_pending = sample_target(*contexts[0], streams[0].value);
    const std::array<std::int64_t,1> canceled_token{canceled_pending};
    auto [late_root, late_verified] = canceled->root->verify(
        *contexts[0], canceled_token, {}, streams[0].value);
    require(late_verified.verification_rows == 1,
            "T92 active cancellation completed worker");
    coordinator.cancel(canceled->ticket, true);
    bool canceled_publish_rejected = false;
    try { (void)coordinator.publish(*canceled, late_root, canceled_pending); }
    catch (const std::invalid_argument&) { canceled_publish_rejected = true; }
    require(canceled_publish_rejected && coordinator.stats().admitted == 0 &&
                coordinator.stats().active == 0 &&
                coordinator.stats().cancellations == 2,
            "T92 cancellation and stale publication gate");
    correctness << "queued_cancel,-1," << queued_ticket.request_id
        << ",1,1,1,1,1\n";
    correctness << "active_cancel,-1," << active_ticket.request_id
        << ",1,1,1,1," << canceled_publish_rejected << '\n';
    write_lifecycle("cancellation_complete", true);

    coordinator.close();
    require(coordinator.stats().closed, "T92 empty close");
    const auto storage = cache.storage_stats();
    require(storage.entries == 1 && storage.accounted_bytes <= cache_budget,
            "T92 final cache memory gate");
    std::size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
               "T92 final device memory");
    require(free_bytes >= (1ULL << 30), "T92 final device memory gate");
    turnover.flush();
    lifecycle.flush();
    memory_out.flush();
    correctness.flush();
    require(turnover.good() && lifecycle.good() && memory_out.good() &&
                correctness.good(),
            "T92 evidence flush");
    std::cout << "T92_PERSISTENT_TURNOVER PASS logical_capacity=8"
        " physical_capacity=2 replacements=8 publications=16 exact=1"
        " cancellations=2 output=" << output.string() << std::endl;
}
