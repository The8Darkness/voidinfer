#pragma once
#include "test_exl3_host_residency.h"
#include <atomic>
#include <thread>

void run_t89_resident_profile(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    using Coordinator = ninfer::exl3::Exl3VeriCacheServingCoordinator;
    constexpr std::size_t kPrompt = 4096;
    constexpr std::size_t kPrefix = 3968;
    constexpr std::size_t kTail = 128;
    constexpr int kOutputs = 8;
    require(target.max_context() >= 4352 && code.size() >= kPrompt &&
                prose.size() >= 4 * kTail,
            "T89 fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T89 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib = std::stoull(env("NINFER_T89_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T89_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024, "T89 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullAvailPhys > reserve,
            "T89 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0, "T89 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T89_NAMESPACE"), env("NINFER_T89_ARTIFACT_ID"),
        env("NINFER_T89_TOKENIZER_ID"), env("NINFER_T89_CONFIGURATION_ID"),
        env("NINFER_T89_MODALITY_ID")};
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
    std::array<std::vector<std::int64_t>,8> prompts;
    for (auto& prompt : prompts) prompt = common;
    for (int tail = 0; tail < 4; ++tail) {
        const auto code_end = code.begin() + kPrompt - tail * kTail;
        prompts[tail].insert(prompts[tail].end(), code_end - kTail, code_end);
        const auto prose_begin = prose.begin() + tail * kTail;
        prompts[tail + 4].insert(prompts[tail + 4].end(), prose_begin,
                                 prose_begin + kTail);
    }
    auto fill = cache.prepare(*context_a, common, kPrefix, available(), 1024);
    cuda_check(cudaDeviceSynchronize(), "T89 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted, "T89 cache fill");
    fill.request.reset();
    std::array<std::shared_ptr<const Request>,8> roots;
    for (int request = 0; request < 8; ++request) {
        auto prepared = cache.prepare(*contexts[request & 1], prompts[request],
                                      kPrefix, available(), 1024);
        cuda_check(cudaDeviceSynchronize(), "T89 request preparation completion");
        require(prepared.metrics.cache_hit &&
                    prepared.metrics.reused_prompt_tokens == kPrefix &&
                    prepared.metrics.executed_prompt_tokens == kTail,
                "T89 preparation accounting");
        roots[request] = std::move(prepared.request);
    }

    std::array<std::shared_ptr<const Request>,2> reference;
    std::array<std::vector<std::int64_t>,2> reference_tokens;
    for (int lane = 0; lane < 2; ++lane) {
        contexts[lane]->restore_exact_host_state(*roots[lane]->state(),
                                                 streams[lane].value);
        auto current = roots[lane];
        auto pending = sample_target(*contexts[lane], streams[lane].value);
        for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
            const std::array<std::int64_t,1> token{pending};
            auto [updated, verified] = current->verify(
                *contexts[lane], token, {}, streams[lane].value);
            require(verified.verification_rows == 1 &&
                        verified.committed_tokens.size() == 1 &&
                        verified.committed_tokens[0] == pending,
                    "T89 serial reference");
            reference_tokens[lane].push_back(pending);
            current = std::move(updated);
            if (ordinal + 1 < kOutputs)
                pending = sample_target(*contexts[lane], streams[lane].value);
        }
        reference[lane] = std::move(current);
    }

    Coordinator coordinator(cache, Coordinator::Policy{
        8, 2, memory.ullTotalPhys - reserve, reserve});
    const auto before = coordinator.stats();
    std::array<Coordinator::Ticket,8> tickets;
    for (int request = 0; request < 8; ++request)
        tickets[request] = coordinator.admit(roots[request]);
    const auto admitted = coordinator.stats();
    auto first = coordinator.acquire();
    auto second = coordinator.acquire();
    require(first && second && first->ticket.request_id == tickets[0].request_id &&
                second->ticket.request_id == tickets[1].request_id,
            "T89 FIFO acquisition");
    std::array<Coordinator::Lease,2> leases{*first, *second};
    for (int lane = 0; lane < 2; ++lane)
        contexts[lane]->restore_exact_host_state(*leases[lane].root->state(),
                                                 streams[lane].value);
    std::array<std::int64_t,2> pending{
        sample_target(*contexts[0], streams[0].value),
        sample_target(*contexts[1], streams[1].value)};
    std::array<std::vector<std::int64_t>,2> published_tokens;
    for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
        std::array<std::shared_ptr<const Request>,2> updated;
        std::array<ninfer::exl3::Exl3OuterReferenceResult,2> verified;
        std::array<std::exception_ptr,2> errors;
        std::thread workers[2];
        for (int lane = 0; lane < 2; ++lane)
            workers[lane] = std::thread([&, lane] {
                try {
                    const std::array<std::int64_t,1> token{pending[lane]};
                    auto result = leases[lane].root->verify(
                        *contexts[lane], token, {}, streams[lane].value);
                    updated[lane] = std::move(result.first);
                    verified[lane] = std::move(result.second);
                } catch (...) { errors[lane] = std::current_exception(); }
            });
        for (auto& worker : workers) worker.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        const auto published = coordinator.publish_batch(leases, updated, pending);
        for (int lane = 0; lane < 2; ++lane) {
            require(verified[lane].verification_rows == 1 &&
                        verified[lane].executed_rows == 1 &&
                        verified[lane].replay_rows == 0 &&
                        verified[lane].root_restores == 0,
                    "T89 publication work");
            leases[lane] = published[lane].lease;
            roots[lane] = published[lane].lease.root;
            published_tokens[lane].push_back(pending[lane]);
        }
        if (ordinal + 1 < kOutputs)
            for (int lane = 0; lane < 2; ++lane)
                pending[lane] = sample_target(*contexts[lane], streams[lane].value);
    }
    const auto published = coordinator.stats();
    for (int lane = 0; lane < 2; ++lane)
        require(published_tokens[lane] == reference_tokens[lane] &&
                    roots[lane]->token_suffix() == reference[lane]->token_suffix() &&
                    roots[lane]->same_taps(*reference[lane]) &&
                    roots[lane]->state()->same_payload(*reference[lane]->state()),
                "T89 publication oracle");

    auto observed_roots = cache.roots();
    observed_roots.insert(observed_roots.end(), roots.begin(), roots.end());
    Exl3HostResidencyProbe probe;
    Request::visit_host_allocations(observed_roots,
        [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
        false);
    const auto resident = probe.measure();
    require(resident.resident_tensor_bytes == resident.allocated_union_bytes &&
                resident.locked_tensor_bytes == resident.allocated_union_bytes,
            "T89 active union residency");
    coordinator.complete(leases[0]);
    coordinator.complete(leases[1]);
    for (int wave = 1; wave < 4; ++wave) {
        auto left = coordinator.acquire();
        auto right = coordinator.acquire();
        require(left && right, "T89 completion acquisition");
        coordinator.complete(*left);
        coordinator.complete(*right);
    }
    const auto completed = coordinator.stats();
    require(completed.queued == 0 && completed.active == 0 &&
                completed.admitted == 0 && completed.publications == 16 &&
                completed.completions == 8 && completed.cancellations == 0,
            "T89 completion state");
    std::size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "T89 device memory");
    require(free_bytes >= (1ULL << 30), "T89 device reserve");

    std::ofstream phases(output / "phases.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(phases.good() && memory_out.good(), "T89 evidence open");
    phases << "phase,updates,inventory_ms,registry_ms,instrumented_ms,"
              "new_locked_bytes,unlocked_bytes,locked_page_bytes,queued,active,"
              "admitted,publications,completions,cancellations\n";
    const auto write_delta = [&](const char* phase, const Coordinator::Stats& a,
                                 const Coordinator::Stats& b) {
        phases << phase << ',' << b.resident_updates - a.resident_updates << ','
            << b.resident_inventory_ms - a.resident_inventory_ms << ','
            << b.resident_registry_ms - a.resident_registry_ms << ','
            << (b.resident_inventory_ms - a.resident_inventory_ms) +
                   (b.resident_registry_ms - a.resident_registry_ms) << ','
            << b.resident_new_locked_bytes - a.resident_new_locked_bytes << ','
            << b.resident_unlocked_bytes - a.resident_unlocked_bytes << ','
            << b.locked_page_bytes << ',' << b.queued << ',' << b.active << ','
            << b.admitted << ',' << b.publications << ',' << b.completions << ','
            << b.cancellations << '\n';
    };
    write_delta("admission", before, admitted);
    write_delta("publication", admitted, published);
    write_delta("completion", published, completed);
    const auto storage = cache.storage_stats();
    memory_out << "resident_payload_bytes,locked_payload_bytes,locked_page_bytes,"
                  "cache_payload_bytes,cache_identity_bytes,device_free_bytes,"
                  "physical_reserve_bytes\n";
    memory_out << resident.resident_tensor_bytes << ','
        << resident.locked_tensor_bytes << ',' << published.locked_page_bytes
        << ',' << storage.payload_allocated_bytes << ','
        << storage.identity_allocated_bytes << ',' << free_bytes << ','
        << reserve << '\n';
    phases.flush(); memory_out.flush();
    require(phases.good() && memory_out.good(), "T89 evidence flush");
    coordinator.close();
    require(coordinator.stats().closed && coordinator.stats().locked_page_bytes == 0,
            "T89 close");
    const double publication_total =
        (published.resident_inventory_ms - admitted.resident_inventory_ms) +
        (published.resident_registry_ms - admitted.resident_registry_ms);
    const double inventory_share = 100.0 *
        (published.resident_inventory_ms - admitted.resident_inventory_ms) /
        publication_total;
    std::cout << "T89_RESIDENT_PROFILE PASS admission_updates=8"
              << " publication_updates=8 completion_updates=8"
              << " publication_instrumented_ms=" << publication_total
              << " publication_inventory_share_pct=" << inventory_share
              << " resident_locked_bytes=" << resident.resident_tensor_bytes
              << " exact=1 completions=8 cancellations=0"
              << " output=" << output.string() << std::endl;
}
