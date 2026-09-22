#pragma once
#include "test_exl3_host_residency.h"

void run_t81_cached_prefix_c4_admission(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft_a,
    const std::filesystem::path& draft_path,
    const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    constexpr std::size_t kPromptRows = 4096;
    constexpr std::size_t kPrefixRows = 3968;
    constexpr std::size_t kTailRows = 128;
    constexpr int kWarmups = 2;
    constexpr int kMeasured = 4;
    require(target.max_context() >= 4352 && code.size() >= kPromptRows &&
                prose.size() >= 2 * kTailRows,
            "T81 requires maxctx4352 and code/prose fixtures");
    require(!output.empty() && !std::filesystem::exists(output),
            "T81 output must be new");
    std::filesystem::create_directories(output);

    const auto reserve_gib = std::stoull(env("NINFER_T81_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T81_CACHE_MIB"));
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    require(reserve_gib >= 8 && cache_mib >= 1024,
            "T81 memory policy extent");
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullTotalPhys > reserve &&
                memory.ullAvailPhys > reserve,
            "T81 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0,
                "T81 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T81_NAMESPACE"), env("NINFER_T81_ARTIFACT_ID"),
        env("NINFER_T81_TOKENIZER_ID"), env("NINFER_T81_CONFIGURATION_ID"),
        env("NINFER_T81_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{8, 64, cache_budget, reserve}, identity);
    ninfer::exl3::Exl3HostResidentSet residency(
        memory.ullTotalPhys - reserve, reserve);

    std::vector<std::int64_t> common(code.begin(), code.begin() + kPrefixRows);
    std::array<std::vector<std::int64_t>, 4> prompts;
    for (auto& prompt : prompts) prompt = common;
    prompts[0].insert(prompts[0].end(), code.begin() + kPrefixRows,
                      code.begin() + kPromptRows);
    prompts[1].insert(prompts[1].end(), prose.begin(),
                      prose.begin() + kTailRows);
    prompts[2].insert(prompts[2].end(), code.begin() + kPrefixRows - kTailRows,
                      code.begin() + kPrefixRows);
    prompts[3].insert(prompts[3].end(), prose.begin() + kTailRows,
                      prose.begin() + 2 * kTailRows);
    for (const auto& prompt : prompts)
        require(prompt.size() == kPromptRows, "T81 prompt extent");
    for (int left = 0; left < 4; ++left)
        for (int right = left + 1; right < 4; ++right)
            require(!std::equal(prompts[left].begin() + kPrefixRows,
                                prompts[left].end(),
                                prompts[right].begin() + kPrefixRows),
                    "T81 tails must be pairwise distinct");

    struct LockResult {
        ninfer::exl3::Exl3HostResidentSet::Stats update;
        Exl3HostResidency observed;
    };
    const auto pin = [&](const std::array<std::shared_ptr<const Request>, 4>* active = nullptr) {
        auto roots = cache.roots();
        if (active)
            for (const auto& root : *active) roots.push_back(root);
        ninfer::exl3::Exl3HostResidentSet::Snapshot snapshot;
        for (const auto& root : roots) snapshot.owners.push_back(root);
        Request::visit_host_allocations(roots,
            [&](const void* data, std::size_t bytes) { snapshot.add(data, bytes); },
            false);
        const auto update = residency.replace(std::move(snapshot));
        Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(roots,
            [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
            false);
        const auto observed = probe.measure();
        require(observed.resident_tensor_bytes == observed.allocated_union_bytes &&
                    observed.locked_tensor_bytes == observed.allocated_union_bytes,
                "T81 authoritative host payload is not resident and locked");
        return LockResult{update, observed};
    };

    std::ofstream preparation(output / "preparation.csv");
    std::ofstream batches(output / "batches.csv");
    std::ofstream requests(output / "requests.csv");
    std::ofstream cancellation(output / "cancellation.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(preparation.good() && batches.good() && requests.good() &&
                cancellation.good() && memory_out.good(),
            "T81 evidence open");
    preparation << "request,tail,preparation_ms,reused_prompt_tokens,"
                   "executed_prompt_tokens,cache_hit,first_token\n";
    batches << "rep,warmup,arm_order,arm,schedule,logical_requests,"
               "physical_limit,queue_high_water,attach_ms,verifier_ms,"
               "full_read_ms,makespan_ms,retained_rows,verifier_rows_per_s,"
               "full_rows_per_s,batch_rows_per_s,min_free_bytes,"
               "ready_owned_bytes,end_owned_bytes,exact\n";
    requests << "rep,warmup,arm,request,wave,physical_lane,first_wave,"
                "queue_wait_ms,release_ms,proposals,accepted,retained_rows,"
                "emitted,pending,position,ring_base,ring_count,publications,"
                "generation,route_bits,exact\n";
    cancellation << "case,survivor,control_ms,candidate_ms,canceled_publications,"
                    "canceled_graph_launched,transactions_closed,target_exact,"
                    "ring_exact,pass\n";
    memory_out << "phase,cache_payload_bytes,cache_identity_bytes,"
                  "cache_accounted_bytes,resident_payload_bytes,"
                  "locked_payload_bytes,resident_page_bytes,new_locked_bytes,"
                  "unlocked_bytes,host_available_bytes,physical_reserve_bytes\n";

    std::array<std::shared_ptr<const Request>, 4> cached_roots;
    Cache::Metrics fill_metrics;
    LockResult fill_lock;
    {
        auto context = target.create_context(true);
        context->prepare_continuation(8);
        auto fill = cache.prepare(*context, common, kPrefixRows,
                                  available(), 1024);
        cuda_check(cudaDeviceSynchronize(), "T81 prefix fill completion");
        fill_metrics = fill.metrics;
        require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                    fill.metrics.reused_prompt_tokens == 0 &&
                    fill.metrics.executed_prompt_tokens == kPrefixRows,
                "T81 prefix fill/admission");
        fill.request.reset();
        fill_lock = pin();
        static constexpr const char* names[4]{
            "code_terminal", "prose_0", "code_prior", "prose_1"};
        for (int index = 0; index < 4; ++index) {
            const auto started = std::chrono::steady_clock::now();
            auto prepared = cache.prepare(*context, prompts[index],
                                          kPrefixRows, available(), 1024);
            cuda_check(cudaDeviceSynchronize(),
                       "T81 cached request preparation completion");
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            require(prepared.metrics.cache_hit &&
                        prepared.metrics.reused_prompt_tokens == kPrefixRows &&
                        prepared.metrics.executed_prompt_tokens == kTailRows,
                    "T81 cached request preparation accounting");
            cached_roots[index] = std::move(prepared.request);
            preparation << index << ',' << names[index] << ',' << elapsed_ms
                << ',' << prepared.metrics.reused_prompt_tokens << ','
                << prepared.metrics.executed_prompt_tokens << ",1,"
                << sample_target(*context) << '\n';
        }
    }

    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "0");
    auto draft_b = Exl3Dflash2DraftModel::load(draft_path);
    TargetGraphC2VerifierLane lane_a(target, draft_a, prompts[0]);
    TargetGraphC2VerifierLane lane_b(target, *draft_b, prompts[1]);
    const auto staging_a = t78_cached_prefix_c2_staging(lane_a);
    const auto staging_b = t78_cached_prefix_c2_staging(lane_b);
    std::array<std::shared_ptr<const Request>, 4> compact_roots;
    compact_roots[0] = cached_roots[0]->compact_draft(draft_a, staging_a);
    compact_roots[1] = cached_roots[1]->compact_draft(*draft_b, staging_b);
    compact_roots[2] = cached_roots[2]->compact_draft(draft_a, staging_a);
    compact_roots[3] = cached_roots[3]->compact_draft(*draft_b, staging_b);
    const auto queued_ring_digest = draft_b->ring_digest();
    const auto queued_state = compact_roots[3]->state();
    const auto queued_tokens = compact_roots[3]->token_suffix();
    cached_roots = {};
    const auto active_lock = pin(&compact_roots);
    memory_out << "filled," << fill_metrics.cache_storage.payload_allocated_bytes
        << ',' << fill_metrics.cache_storage.identity_allocated_bytes << ','
        << fill_metrics.cache_storage.accounted_bytes << ','
        << fill_lock.observed.resident_tensor_bytes << ','
        << fill_lock.observed.locked_tensor_bytes << ',' << fill_lock.update.page_bytes
        << ',' << fill_lock.update.new_locked_bytes << ','
        << fill_lock.update.unlocked_bytes << ',' << available() << ',' << reserve
        << "\nactive_c4," << cache.storage_stats().payload_allocated_bytes << ','
        << cache.storage_stats().identity_allocated_bytes << ','
        << cache.storage_stats().accounted_bytes << ','
        << active_lock.observed.resident_tensor_bytes << ','
        << active_lock.observed.locked_tensor_bytes << ','
        << active_lock.update.page_bytes << ',' << active_lock.update.new_locked_bytes
        << ',' << active_lock.update.unlocked_bytes << ',' << available() << ','
        << reserve << '\n';

    struct LogicalTrace {
        TargetGraphC2VerifierLaneTrace lane;
        int wave = 0;
        int physical_lane = 0;
        double queue_wait_ms = 0.0;
        double release_ms = 0.0;
    };
    struct ArmTrace {
        std::array<LogicalTrace, 4> logical;
        double attach_ms = 0.0;
        double verifier_ms = 0.0;
        double full_read_ms = 0.0;
        double makespan_ms = 0.0;
        std::uint64_t retained_rows = 0;
        std::size_t min_free_bytes = std::numeric_limits<std::size_t>::max();
        std::size_t ready_owned_bytes = 0;
        std::size_t end_owned_bytes = 0;
    };
    static constexpr int schedules[4][4]{
        {0, 1, 2, 3}, {2, 3, 0, 1},
        {0, 3, 2, 1}, {2, 1, 0, 3}};
    const auto run_arm = [&](bool overlap, int rep) {
        ArmTrace result;
        const auto batch_started = std::chrono::steady_clock::now();
        const int* schedule = schedules[rep & 3];
        for (int wave = 0; wave < 2; ++wave) {
            const int logical_a = schedule[2 * wave];
            const int logical_b = schedule[2 * wave + 1];
            require((logical_a & 1) == 0 && (logical_b & 1) == 1,
                    "T81 draft identity schedule");
            const std::array<std::shared_ptr<const Request>, 2> roots{
                compact_roots[logical_a], compact_roots[logical_b]};
            const auto attach = t78_cached_prefix_c2_attach(
                lane_a, lane_b, roots);
            result.attach_ms += attach.elapsed_ms;
            const auto execute_started = std::chrono::steady_clock::now();
            const double queue_wait_ms = std::chrono::duration<double, std::milli>(
                execute_started - batch_started).count();
            const bool b_first = ((rep + wave) & 1) != 0;
            auto trace = target_graph_c2_verifier_run_arm(
                lane_a, lane_b, overlap, b_first, true, true);
            result.verifier_ms += trace.complete_verifier_pair_ms;
            result.full_read_ms += trace.full_read_pair_ms;
            result.min_free_bytes = std::min({result.min_free_bytes,
                trace.ready_free_bytes, trace.end_free_bytes});
            if (!result.ready_owned_bytes)
                result.ready_owned_bytes = trace.ready_owned_bytes;
            require(result.ready_owned_bytes == trace.ready_owned_bytes &&
                        trace.ready_owned_bytes == trace.end_owned_bytes,
                    "T81 physical owned-byte drift");
            result.end_owned_bytes = trace.end_owned_bytes;
            const int logical_ids[2]{logical_a, logical_b};
            for (int physical = 0; physical < 2; ++physical) {
                auto& destination = result.logical[logical_ids[physical]];
                destination.lane = std::move(trace.lanes[physical]);
                destination.wave = wave;
                destination.physical_lane = physical;
                destination.queue_wait_ms = queue_wait_ms;
                destination.release_ms = queue_wait_ms +
                    destination.lane.full_read_latency_ms;
                result.retained_rows += static_cast<std::uint64_t>(
                    destination.lane.retained_rows);
            }
        }
        result.makespan_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - batch_started).count();
        return result;
    };
    const auto schedule_name = [](int rep) {
        const int* values = schedules[rep & 3];
        std::ostringstream out;
        for (int index = 0; index < 4; ++index) {
            if (index) out << '|';
            out << values[index];
        }
        return out.str();
    };
    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };

    struct PairTrace { ArmTrace c1, c2; };
    std::vector<PairTrace> measured;
    std::array<int, 4> first_wave_counts{};
    std::array<std::array<int, 4>, 2> publication_counts{};
    std::vector<double> full_gains, makespan_gains;
    for (int rep = 0; rep < kWarmups + kMeasured; ++rep) {
        PairTrace pair;
        const bool c2_first = (rep & 1) != 0;
        for (int arm_index = 0; arm_index < 2; ++arm_index) {
            const bool overlap = c2_first ? arm_index == 0 : arm_index == 1;
            auto trace = run_arm(overlap, rep);
            if (overlap) pair.c2 = std::move(trace);
            else pair.c1 = std::move(trace);
        }
        for (int logical = 0; logical < 4; ++logical)
            target_graph_c2_verifier_require_lane_equal(
                pair.c1.logical[logical].lane, pair.c2.logical[logical].lane,
                "T81 matched rep " + std::to_string(rep) + " request " +
                    std::to_string(logical));
        const auto write_arm = [&](const char* name, const ArmTrace& arm,
                                   int arm_slot) {
            batches << rep << ',' << (rep < kWarmups) << ','
                << (c2_first ? "C2C1" : "C1C2") << ',' << name << ','
                << schedule_name(rep) << ",4,2,2," << arm.attach_ms << ','
                << arm.verifier_ms << ',' << arm.full_read_ms << ','
                << arm.makespan_ms << ',' << arm.retained_rows << ','
                << arm.retained_rows * 1000.0 / arm.verifier_ms << ','
                << arm.retained_rows * 1000.0 / arm.full_read_ms << ','
                << arm.retained_rows * 1000.0 / arm.makespan_ms << ','
                << arm.min_free_bytes << ',' << arm.ready_owned_bytes << ','
                << arm.end_owned_bytes << ",1\n";
            for (int logical = 0; logical < 4; ++logical) {
                const auto& row = arm.logical[logical];
                requests << rep << ',' << (rep < kWarmups) << ',' << name
                    << ',' << logical << ',' << row.wave << ','
                    << (row.physical_lane ? 'B' : 'A') << ','
                    << (row.wave == 0) << ',' << row.queue_wait_ms << ','
                    << row.release_ms << ','
                    << target_graph_c2_verifier_ids(row.lane.proposals) << ','
                    << row.lane.accepted << ',' << row.lane.retained_rows << ','
                    << target_graph_c2_verifier_ids(row.lane.emitted) << ','
                    << row.lane.pending << ',' << row.lane.position << ','
                    << row.lane.ring_base << ',' << row.lane.ring_count << ','
                    << row.lane.publication_count << ',' << row.lane.generation
                    << ',' << row.lane.route_bits << ",1\n";
                require(row.lane.publication_count == 1,
                        "T81 request publication count");
                if (rep >= kWarmups)
                    ++publication_counts[arm_slot][logical];
            }
        };
        write_arm("C1", pair.c1, 0);
        write_arm("C2_BACKPRESSURE", pair.c2, 1);
        require(pair.c1.min_free_bytes >= (1ULL << 30) &&
                    pair.c2.min_free_bytes >= (1ULL << 30),
                "T81 device reserve");
        if (rep >= kWarmups) {
            const int* schedule = schedules[rep & 3];
            ++first_wave_counts[schedule[0]];
            ++first_wave_counts[schedule[1]];
            const double full_gain = (pair.c1.full_read_ms - pair.c2.full_read_ms) *
                100.0 / pair.c1.full_read_ms;
            const double makespan_gain = (pair.c1.makespan_ms - pair.c2.makespan_ms) *
                100.0 / pair.c1.makespan_ms;
            require(full_gain >= -2.0 && makespan_gain >= -2.0,
                    "T81 individual performance guard");
            full_gains.push_back(full_gain);
            makespan_gains.push_back(makespan_gain);
            measured.push_back(std::move(pair));
        }
    }
    require(first_wave_counts == std::array<int, 4>{2, 2, 2, 2},
            "T81 rotating first-wave fairness");
    for (const auto& arm : publication_counts)
        require(arm == std::array<int, 4>{4, 4, 4, 4},
                "T81 measured publication fairness");
    const double full_median_gain = median(full_gains);
    const double makespan_median_gain = median(makespan_gains);
    require(full_median_gain > 0.0 && makespan_median_gain > 0.0,
            "T81 median performance gate");

    // Queued cancellation: request 3 is removed before assignment. Its
    // immutable target root and identity-bound projected ring are then restored
    // only for audit; no graph is captured or launched and no token publishes.
    require(compact_roots[3]->state() == queued_state &&
                compact_roots[3]->token_suffix() == queued_tokens,
            "T81 queued cancellation immutable target root");
    lane_b.target->reset(lane_b.stream.value);
    draft_b->reset(lane_b.stream.value);
    cuda_check(cudaStreamSynchronize(lane_b.stream.value),
               "T81 queued cancellation audit reset");
    compact_roots[3]->restore_inner(*lane_b.target, *draft_b, staging_b,
                                    nullptr, lane_b.stream.value);
    cuda_check(cudaStreamSynchronize(lane_b.stream.value),
               "T81 queued cancellation audit restore");
    const bool queued_target_exact = lane_b.target->position() ==
        compact_roots[3]->state()->position();
    const bool queued_ring_exact = draft_b->ring_digest(lane_b.stream.value) ==
        queued_ring_digest;
    const bool queued_pass = queued_target_exact && queued_ring_exact &&
        !lane_b.target->transaction_active();
    cancellation << "queued,NONE,0,0,0,0," <<
        !lane_b.target->transaction_active() << ',' << queued_target_exact << ','
        << queued_ring_exact << ',' << queued_pass << '\n';
    require(queued_pass, "T81 queued cancellation gate");

    const std::array<std::shared_ptr<const Request>, 2> cancel_roots{
        compact_roots[0], compact_roots[1]};
    t78_cached_prefix_c2_attach(lane_a, lane_b, cancel_roots);
    const auto control = target_graph_c2_verifier_run_cancellation_arm(
        lane_a, lane_b, 0, false, true);
    t78_cached_prefix_c2_attach(lane_a, lane_b, cancel_roots);
    const auto candidate = target_graph_c2_verifier_run_cancellation_arm(
        lane_a, lane_b, 0, true, true);
    target_graph_c2_verifier_require_lane_equal(
        control.survivor, candidate.survivor,
        "T81 prelaunch cancellation survivor exact");
    target_graph_c2_require_snapshot_equal(
        control.canceled_root, candidate.canceled_root, false,
        "T81 prelaunch cancellation target root exact");
    const bool canceled_ring_exact =
        control.canceled_ring_base == candidate.canceled_ring_base &&
        control.canceled_ring_count == candidate.canceled_ring_count &&
        control.canceled_ring_digest == candidate.canceled_ring_digest;
    const bool cancel_pass = candidate.canceled_publications == 0 &&
        !candidate.canceled_graph_launched && candidate.transactions_closed &&
        candidate.end_owned_bytes == candidate.ready_owned_bytes &&
        canceled_ring_exact;
    cancellation << "prelaunch,A," << control.settle_ms << ','
        << candidate.settle_ms << ',' << candidate.canceled_publications << ','
        << candidate.canceled_graph_launched << ','
        << candidate.transactions_closed << ",1," << canceled_ring_exact << ','
        << cancel_pass << '\n';
    require(cancel_pass, "T81 prelaunch cancellation gate");

    const auto final_lock = pin(&compact_roots);
    require(final_lock.observed.resident_tensor_bytes ==
                active_lock.observed.resident_tensor_bytes &&
                final_lock.observed.locked_tensor_bytes ==
                active_lock.observed.locked_tensor_bytes,
            "T81 active C4 resident/locked drift");
    memory_out << "final_c4," << cache.storage_stats().payload_allocated_bytes
        << ',' << cache.storage_stats().identity_allocated_bytes << ','
        << cache.storage_stats().accounted_bytes << ','
        << final_lock.observed.resident_tensor_bytes << ','
        << final_lock.observed.locked_tensor_bytes << ','
        << final_lock.update.page_bytes << ',' << final_lock.update.new_locked_bytes
        << ',' << final_lock.update.unlocked_bytes << ',' << available() << ','
        << reserve << '\n';
    preparation.flush();
    batches.flush();
    requests.flush();
    cancellation.flush();
    memory_out.flush();
    require(preparation.good() && batches.good() && requests.good() &&
                cancellation.good() && memory_out.good(),
            "T81 evidence flush");
    residency.close();
    std::cout << "T81_CACHED_PREFIX_C4_ADMISSION PASS logical_requests=4"
              << " physical_limit=2 queue_high_water=2 measured_batches=4"
              << " full_median_gain_percent=" << full_median_gain
              << " makespan_median_gain_percent=" << makespan_median_gain
              << " exact=1 cancellation=2 resident_locked=1"
              << " cache_payload_bytes="
              << cache.storage_stats().payload_allocated_bytes
              << " output=" << output.string() << std::endl;
}
