#pragma once
#include "test_exl3_host_residency.h"

void run_t87_batch_admission(
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
    constexpr int kWarmups = 2;
    constexpr int kMeasured = 6;
    require(target.max_context() >= 4352 && code.size() >= kPrompt &&
                prose.size() >= 4 * kTail,
            "T87 fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T87 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib = std::stoull(env("NINFER_T87_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T87_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024, "T87 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullAvailPhys > reserve,
            "T87 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0, "T87 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T87_NAMESPACE"), env("NINFER_T87_ARTIFACT_ID"),
        env("NINFER_T87_TOKENIZER_ID"), env("NINFER_T87_CONFIGURATION_ID"),
        env("NINFER_T87_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{16, 64, cache_budget, reserve}, identity);
    auto context_a = target.create_context(true);
    auto context_b = target.create_context(true);
    context_a->prepare_continuation(8);
    context_b->prepare_continuation(8);
    Exl3TextContext* contexts[2]{context_a.get(), context_b.get()};
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
    cuda_check(cudaDeviceSynchronize(), "T87 cache fill completion");
    require(!fill.metrics.cache_hit && fill.metrics.admitted,
            "T87 cache fill gate");
    fill.request.reset();
    std::array<std::shared_ptr<const Request>,8> roots;
    for (int request = 0; request < 8; ++request) {
        auto prepared = cache.prepare(*contexts[request & 1], prompts[request],
                                      kPrefix, available(), 1024);
        cuda_check(cudaDeviceSynchronize(), "T87 request preparation completion");
        require(prepared.metrics.cache_hit &&
                    prepared.metrics.reused_prompt_tokens == kPrefix &&
                    prepared.metrics.executed_prompt_tokens == kTail,
                "T87 request preparation accounting");
        roots[request] = std::move(prepared.request);
    }
    const auto observe = [&] {
        auto observed_roots = cache.roots();
        observed_roots.insert(observed_roots.end(), roots.begin(), roots.end());
        Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(observed_roots,
            [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
            false);
        return probe.measure();
    };

    std::ofstream arms(output / "arms.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(arms.good() && memory_out.good(), "T87 evidence open");
    arms << "rep,warmup,order,arm,admission_ms,gain_pct,tickets,queued,active,"
            "admitted,high_queued,high_active,high_admitted,publications,"
            "cancellations,completions,resident_payload_bytes,locked_payload_bytes,"
            "locked_page_bytes,device_free_bytes,invalid_batches_rejected,"
            "fifo,completed_stale_rejected,closed_empty,pass\n";
    memory_out << "cache_payload_bytes,cache_identity_bytes,cache_accounted_bytes,"
                  "physical_reserve_bytes\n";
    const auto storage = cache.storage_stats();
    memory_out << storage.payload_allocated_bytes << ','
        << storage.identity_allocated_bytes << ',' << storage.accounted_bytes
        << ',' << reserve << '\n';

    struct Trace {
        double admission_ms = 0.0;
        std::vector<Coordinator::Ticket> tickets;
        Coordinator::Stats admitted;
        Exl3HostResidency residency;
        std::size_t device_free = 0;
        bool invalid_rejected = false;
        bool fifo = false;
        bool stale_rejected = false;
        bool closed_empty = false;
    };
    std::size_t total_device = 0;
    const auto run_arm = [&](bool batch) {
        Trace trace;
        Coordinator coordinator(cache, Coordinator::Policy{
            8, 2, memory.ullTotalPhys - reserve, reserve});
        if (batch) {
            bool empty = false, null_root = false, overflow = false;
            try { (void)coordinator.admit_batch({}); }
            catch (const std::invalid_argument&) { empty = true; }
            auto invalid = roots;
            invalid[3].reset();
            try { (void)coordinator.admit_batch(invalid); }
            catch (const std::invalid_argument&) { null_root = true; }
            std::array<std::shared_ptr<const Request>,9> too_many;
            for (int index = 0; index < 9; ++index) too_many[index] = roots[index & 7];
            try { (void)coordinator.admit_batch(too_many); }
            catch (const std::runtime_error&) { overflow = true; }
            const auto clean = coordinator.stats();
            trace.invalid_rejected = empty && null_root && overflow &&
                clean.queued == 0 && clean.active == 0 && clean.admitted == 0 &&
                clean.publications == 0 && clean.cancellations == 0 &&
                clean.completions == 0 && clean.locked_page_bytes == 0;
        } else {
            trace.invalid_rejected = true;
        }
        const auto started = std::chrono::steady_clock::now();
        if (batch) trace.tickets = coordinator.admit_batch(roots);
        else for (const auto& root : roots) trace.tickets.push_back(
            coordinator.admit(root));
        trace.admission_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        trace.admitted = coordinator.stats();
        trace.residency = observe();
        cuda_check(cudaMemGetInfo(&trace.device_free, &total_device),
                   "T87 device memory");
        require(trace.tickets.size() == 8 && trace.invalid_rejected &&
                    trace.admitted.queued == 8 && trace.admitted.active == 0 &&
                    trace.admitted.admitted == 8 &&
                    coordinator.high_queued() == 8 &&
                    coordinator.high_admitted() == 8 &&
                    trace.residency.resident_tensor_bytes ==
                        trace.residency.allocated_union_bytes &&
                    trace.residency.locked_tensor_bytes ==
                        trace.residency.allocated_union_bytes &&
                    trace.device_free >= (1ULL << 30),
                "T87 admitted contract");
        for (int index = 0; index < 8; ++index)
            require(trace.tickets[index].request_id ==
                        static_cast<std::uint64_t>(index + 1) &&
                    trace.tickets[index].generation == 1,
                    "T87 ticket identity");
        Coordinator::Lease first_completed;
        trace.fifo = true;
        for (int wave = 0; wave < 4; ++wave) {
            const auto first = coordinator.acquire();
            const auto second = coordinator.acquire();
            trace.fifo = trace.fifo && first && second &&
                first->ticket.request_id == static_cast<std::uint64_t>(2 * wave + 1) &&
                second->ticket.request_id == static_cast<std::uint64_t>(2 * wave + 2);
            if (wave == 0) first_completed = *first;
            coordinator.complete(*first);
            coordinator.complete(*second);
        }
        try { coordinator.complete(first_completed); }
        catch (const std::invalid_argument&) { trace.stale_rejected = true; }
        const auto finished = coordinator.stats();
        require(trace.fifo && trace.stale_rejected && finished.queued == 0 &&
                    finished.active == 0 && finished.admitted == 0 &&
                    finished.completions == 8 && finished.cancellations == 0 &&
                    coordinator.high_active() == 2,
                "T87 completion contract");
        coordinator.close();
        const auto closed = coordinator.stats();
        trace.closed_empty = closed.closed && closed.queued == 0 &&
            closed.active == 0 && closed.admitted == 0 &&
            closed.locked_page_bytes == 0;
        require(trace.closed_empty, "T87 close contract");
        return trace;
    };

    std::vector<double> sequential_times, batch_times, gains;
    for (int rep = 0; rep < kWarmups + kMeasured; ++rep) {
        Trace sequential, batch;
        const bool batch_first = (rep & 1) != 0;
        for (int arm = 0; arm < 2; ++arm) {
            const bool use_batch = batch_first ? arm == 0 : arm == 1;
            auto trace = run_arm(use_batch);
            if (use_batch) batch = std::move(trace);
            else sequential = std::move(trace);
        }
        require(sequential.tickets.size() == batch.tickets.size() &&
                    sequential.residency.allocated_union_bytes ==
                        batch.residency.allocated_union_bytes &&
                    sequential.residency.resident_tensor_bytes ==
                        batch.residency.resident_tensor_bytes &&
                    sequential.residency.locked_tensor_bytes ==
                        batch.residency.locked_tensor_bytes,
                "T87 arm equivalence");
        const double gain = 100.0 *
            (1.0 - batch.admission_ms / sequential.admission_ms);
        const auto write = [&](const char* name, const Trace& trace) {
            std::ostringstream tickets;
            for (std::size_t index = 0; index < trace.tickets.size(); ++index) {
                if (index) tickets << '|';
                tickets << trace.tickets[index].request_id << ':'
                        << trace.tickets[index].generation;
            }
            arms << rep << ',' << (rep < kWarmups) << ','
                << (batch_first ? "BATCH_SEQUENTIAL" : "SEQUENTIAL_BATCH")
                << ',' << name << ',' << trace.admission_ms << ',' << gain << ','
                << tickets.str() << ',' << trace.admitted.queued << ','
                << trace.admitted.active << ',' << trace.admitted.admitted << ','
                << 8 << ',' << 2 << ',' << 8 << ','
                << trace.admitted.publications << ','
                << trace.admitted.cancellations << ','
                << trace.admitted.completions << ','
                << trace.residency.resident_tensor_bytes << ','
                << trace.residency.locked_tensor_bytes << ','
                << trace.admitted.locked_page_bytes << ',' << trace.device_free
                << ',' << trace.invalid_rejected << ',' << trace.fifo << ','
                << trace.stale_rejected << ',' << trace.closed_empty << ",1\n";
        };
        write("SEQUENTIAL", sequential);
        write("BATCH", batch);
        if (rep >= kWarmups) {
            sequential_times.push_back(sequential.admission_ms);
            batch_times.push_back(batch.admission_ms);
            gains.push_back(gain);
        }
    }
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const auto middle = values.size() / 2;
        return values.size() & 1 ? values[middle] :
            (values[middle - 1] + values[middle]) / 2.0;
    };
    arms.flush(); memory_out.flush();
    require(arms.good() && memory_out.good(), "T87 evidence flush");
    std::cout << "T87_BATCH_ADMISSION PASS measured_pairs=6"
              << " sequential_median_ms=" << median(sequential_times)
              << " batch_median_ms=" << median(batch_times)
              << " gain_median_percent=" << median(gains)
              << " exact_lifecycle=1 resident_locked=1"
              << " output=" << output.string() << std::endl;
}
