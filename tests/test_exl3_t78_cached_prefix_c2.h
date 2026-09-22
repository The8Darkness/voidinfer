#pragma once
#include "test_exl3_host_residency.h"

struct T78CachedPrefixC2Attach {
    double elapsed_ms = 0.0;
    std::array<std::uint64_t, 2> generation{};
};

std::array<std::uint16_t*, 5> t78_cached_prefix_c2_staging(
    const TargetGraphC2VerifierLane& lane) {
    require(lane.stage.bulk_ptrs.size() == 5, "T78 staging pointer extent");
    return {lane.stage.bulk_ptrs[0], lane.stage.bulk_ptrs[1],
            lane.stage.bulk_ptrs[2], lane.stage.bulk_ptrs[3],
            lane.stage.bulk_ptrs[4]};
}

T78CachedPrefixC2Attach t78_cached_prefix_c2_attach(
    TargetGraphC2VerifierLane& a, TargetGraphC2VerifierLane& b,
    const std::array<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>, 2>& roots) {
    const auto started = std::chrono::steady_clock::now();
    TargetGraphC2VerifierLane* lanes[2]{&a, &b};
    for (int index = 0; index < 2; ++index) {
        auto& lane = *lanes[index];
        require(roots[index] && roots[index]->compact_draft(),
                "T78 compact cached root missing");
        lane.target->reset(lane.stream.value);
        lane.draft->reset(lane.stream.value);
        cuda_check(cudaStreamSynchronize(lane.stream.value),
                   "T78 cached bridge reset");
        const auto staging = t78_cached_prefix_c2_staging(lane);
        roots[index]->restore_inner(*lane.target, *lane.draft,
            staging, nullptr, lane.stream.value);
        cuda_check(cudaStreamSynchronize(lane.stream.value),
                   "T78 cached bridge restore");
        lane.pending = sample_target(*lane.target, lane.stream.value);
        lane.abs_pos = roots[index]->state()->position();
        lane.publication_count = 0;
        require(lane.abs_pos == static_cast<int>(lane.prefix.size()) &&
                    lane.target->position() == lane.abs_pos &&
                    lane.draft->ring_base_abs() + lane.draft->ring_count() ==
                        lane.abs_pos,
                "T78 cached bridge target/draft tail");
        require(lane.target->capture_continuation_graph(lane.stream.value),
                "T78 cached bridge graph capture: " +
                    lane.target->continuation_graph_status());
        cuda_check(cudaStreamSynchronize(lane.stream.value),
                   "T78 cached bridge graph capture complete");
        const auto admission = lane.target->continuation_graph_admission();
        target_graph_c2_verifier_require_admission(
            admission, lane.stream.value, "T78 cached bridge idle", false);
    }
    T78CachedPrefixC2Attach result;
    result.elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    result.generation = {
        a.target->continuation_graph_admission().route_generation,
        b.target->continuation_graph_admission().route_generation};
    return result;
}

void run_t78_cached_prefix_c2(
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
                prose.size() >= kTailRows,
            "T78 requires maxctx4352 and code/prose fixtures");
    require(!output.empty() && !std::filesystem::exists(output),
            "T78 output must be new");
    std::filesystem::create_directories(output);

    const auto reserve_gib = std::stoull(env("NINFER_T78_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T78_CACHE_MIB"));
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    require(reserve_gib >= 4 && cache_mib >= 512,
            "T78 memory policy extent");
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullTotalPhys > reserve &&
                memory.ullAvailPhys > reserve,
            "T78 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0,
                "T78 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T78_NAMESPACE"), env("NINFER_T78_ARTIFACT_ID"),
        env("NINFER_T78_TOKENIZER_ID"), env("NINFER_T78_CONFIGURATION_ID"),
        env("NINFER_T78_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{4, 64, cache_budget, reserve}, identity);
    ninfer::exl3::Exl3HostResidentSet residency(
        memory.ullTotalPhys - reserve, reserve);

    std::vector<std::int64_t> common(code.begin(), code.begin() + kPrefixRows);
    std::array<std::vector<std::int64_t>, 2> prompts{
        std::vector<std::int64_t>(code.begin(), code.begin() + kPromptRows),
        common};
    prompts[1].insert(prompts[1].end(), prose.begin(), prose.begin() + kTailRows);
    require(prompts[0].size() == kPromptRows &&
                prompts[1].size() == kPromptRows &&
                !std::equal(prompts[0].begin() + kPrefixRows, prompts[0].end(),
                            prompts[1].begin() + kPrefixRows),
            "T78 distinct tails");

    struct LockResult {
        ninfer::exl3::Exl3HostResidentSet::Stats update;
        Exl3HostResidency observed;
    };
    const auto pin = [&](const std::array<std::shared_ptr<const Request>, 2>* active = nullptr) {
        auto roots = cache.roots();
        if (active) for (const auto& root : *active) roots.push_back(root);
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
                "T78 authoritative host payload is not resident and locked");
        return LockResult{update, observed};
    };

    std::ofstream preparation(output / "preparation.csv");
    std::ofstream rounds(output / "rounds.csv");
    std::ofstream cancellation(output / "cancellation.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(preparation.good() && rounds.good() && cancellation.good() &&
                memory_out.good(),
            "T78 evidence open");
    preparation << "rep,request_order,arm,pair_ready_ms,reused_prompt_tokens,"
                   "executed_prompt_tokens,first_token_a,first_token_b,exact\n";
    rounds << "rep,warmup,request_order,arm_order,arm,attach_ms,retained_rows,"
              "verifier_ms,full_read_ms,verifier_rows_per_s,full_rows_per_s,"
              "ready_free_bytes,end_free_bytes,ready_owned_bytes,end_owned_bytes,"
              "generation_a,generation_b,exact\n";
    cancellation << "survivor,control_ms,c2_ms,ratio,canceled_publications,"
                    "transactions_closed,logical_growth_bytes,pass\n";
    memory_out << "phase,cache_payload_bytes,cache_identity_bytes,"
                  "cache_accounted_bytes,resident_payload_bytes,"
                  "locked_payload_bytes,resident_page_bytes,new_locked_bytes,"
                  "unlocked_bytes,host_available_bytes,physical_reserve_bytes\n";

    std::array<std::shared_ptr<const Request>, 2> cached_roots;
    std::vector<double> fresh_pair_ms, cached_pair_ms;
    Cache::Metrics fill_metrics;
    LockResult fill_lock;
    double fill_ms = 0.0;
    {
        auto cached_context = target.create_context(true);
        auto fresh_context = target.create_context(true);
        cached_context->prepare_continuation(8);
        fresh_context->prepare_continuation(8);
        const auto fill_started = std::chrono::steady_clock::now();
        auto fill = cache.prepare(*cached_context, common, kPrefixRows,
                                  available(), 1024);
        cuda_check(cudaDeviceSynchronize(), "T78 prefix fill completion");
        fill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fill_started).count();
        fill_metrics = fill.metrics;
        require(!fill.metrics.cache_hit && fill.metrics.admitted &&
                    fill.metrics.reused_prompt_tokens == 0 &&
                    fill.metrics.executed_prompt_tokens == kPrefixRows,
                "T78 prefix fill/admission");
        fill.request.reset();
        fill_lock = pin();

        struct PreparedPair {
            std::array<std::shared_ptr<const Request>, 2> roots;
            std::array<Cache::Metrics, 2> metrics;
            std::array<std::int64_t, 2> first_tokens{-1, -1};
            double elapsed_ms = 0.0;
        };
        for (int rep = 0; rep < 3; ++rep) {
            PreparedPair fresh, cached;
            const bool b_first = (rep & 1) != 0;
            const int order[2]{b_first ? 1 : 0, b_first ? 0 : 1};
            const auto run = [&](bool use_cache, PreparedPair& pair) {
                auto& context = use_cache ? *cached_context : *fresh_context;
                const auto started = std::chrono::steady_clock::now();
                for (const int index : order) {
                    if (use_cache) {
                        auto prepared = cache.prepare(context, prompts[index],
                            kPrefixRows, available(), 1024);
                        pair.roots[index] = std::move(prepared.request);
                        pair.metrics[index] = prepared.metrics;
                    } else {
                        pair.roots[index] = Request::initialize(
                            context, prompts[index], 1024);
                        pair.metrics[index].prompt_tokens = kPromptRows;
                        pair.metrics[index].executed_prompt_tokens = kPromptRows;
                    }
                    pair.first_tokens[index] = sample_target(context);
                }
                cuda_check(cudaDeviceSynchronize(),
                           "T78 paired preparation completion");
                pair.elapsed_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
            };
            if (rep == 1) {
                run(true, cached);
                run(false, fresh);
            } else {
                run(false, fresh);
                run(true, cached);
            }
            for (int index = 0; index < 2; ++index) {
                require(cached.metrics[index].cache_hit &&
                            cached.metrics[index].reused_prompt_tokens == kPrefixRows &&
                            cached.metrics[index].executed_prompt_tokens == kTailRows &&
                            fresh.metrics[index].reused_prompt_tokens == 0 &&
                            fresh.metrics[index].executed_prompt_tokens == kPromptRows &&
                            cached.first_tokens[index] == fresh.first_tokens[index] &&
                            cached.roots[index]->token_suffix() ==
                                fresh.roots[index]->token_suffix() &&
                            cached.roots[index]->same_taps(*fresh.roots[index]) &&
                            cached.roots[index]->state()->same_payload(
                                *fresh.roots[index]->state()),
                        "T78 cached preparation differs from fresh authority");
            }
            fresh_pair_ms.push_back(fresh.elapsed_ms);
            cached_pair_ms.push_back(cached.elapsed_ms);
            const auto write_pair = [&](const char* arm, const PreparedPair& pair) {
                preparation << rep << ',' << (b_first ? "BA" : "AB") << ','
                    << arm << ',' << pair.elapsed_ms << ','
                    << (pair.metrics[0].reused_prompt_tokens +
                        pair.metrics[1].reused_prompt_tokens) << ','
                    << (pair.metrics[0].executed_prompt_tokens +
                        pair.metrics[1].executed_prompt_tokens) << ','
                    << pair.first_tokens[0] << ',' << pair.first_tokens[1]
                    << ",1\n";
            };
            write_pair("fresh", fresh);
            write_pair("cached", cached);
            if (rep == 2) cached_roots = std::move(cached.roots);
        }
    }

    const auto median = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    const double fresh_median = median(fresh_pair_ms);
    const double cached_median = median(cached_pair_ms);
    const int preparation_positive = static_cast<int>(std::inner_product(
        fresh_pair_ms.begin(), fresh_pair_ms.end(), cached_pair_ms.begin(), 0,
        std::plus<int>(), [](double fresh, double cached) {
            return cached < fresh ? 1 : 0;
        }));
    require(preparation_positive == 3 && cached_median < fresh_median,
            "T78 cached pair-ready preparation performance gate");

    // Context storage modes are selected at construction. The temporary
    // authoritative-host contexts above are gone; the graph lanes below own
    // OSCAR storage and receive the immutable exact state through the existing
    // restore_oscar_host_state bridge.
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "0");
    auto draft_b = Exl3Dflash2DraftModel::load(draft_path);
    TargetGraphC2VerifierLane lane_a(target, draft_a, prompts[0]);
    TargetGraphC2VerifierLane lane_b(target, *draft_b, prompts[1]);
    const auto staging_a = t78_cached_prefix_c2_staging(lane_a);
    const auto staging_b = t78_cached_prefix_c2_staging(lane_b);
    std::array<std::shared_ptr<const Request>, 2> compact_roots{
        cached_roots[0]->compact_draft(draft_a, staging_a),
        cached_roots[1]->compact_draft(*draft_b, staging_b)};
    cached_roots = {};
    const auto active_lock = pin(&compact_roots);
    memory_out << "filled," << fill_metrics.cache_storage.payload_allocated_bytes
        << ',' << fill_metrics.cache_storage.identity_allocated_bytes << ','
        << fill_metrics.cache_storage.accounted_bytes << ','
        << fill_lock.observed.resident_tensor_bytes << ','
        << fill_lock.observed.locked_tensor_bytes << ',' << fill_lock.update.page_bytes
        << ',' << fill_lock.update.new_locked_bytes << ','
        << fill_lock.update.unlocked_bytes << ',' << available() << ',' << reserve
        << "\nactive_bridge," << cache.storage_stats().payload_allocated_bytes
        << ',' << cache.storage_stats().identity_allocated_bytes << ','
        << cache.storage_stats().accounted_bytes << ','
        << active_lock.observed.resident_tensor_bytes << ','
        << active_lock.observed.locked_tensor_bytes << ','
        << active_lock.update.page_bytes << ',' << active_lock.update.new_locked_bytes
        << ',' << active_lock.update.unlocked_bytes << ',' << available() << ','
        << reserve << '\n';

    struct PairTrace {
        TargetGraphC2VerifierArmTrace c1, c2;
        double c1_attach_ms = 0.0, c2_attach_ms = 0.0;
    };
    std::vector<PairTrace> measured;
    for (int rep = 0; rep < kWarmups + kMeasured; ++rep) {
        PairTrace pair;
        const bool b_first = (rep & 1) != 0;
        const bool c2_first = (rep & 1) != 0;
        for (int arm_index = 0; arm_index < 2; ++arm_index) {
            const bool overlap = c2_first ? arm_index == 0 : arm_index == 1;
            const auto attach = t78_cached_prefix_c2_attach(
                lane_a, lane_b, compact_roots);
            auto trace = target_graph_c2_verifier_run_arm(
                lane_a, lane_b, overlap, b_first, true, true);
            if (overlap) {
                pair.c2 = std::move(trace);
                pair.c2_attach_ms = attach.elapsed_ms;
            } else {
                pair.c1 = std::move(trace);
                pair.c1_attach_ms = attach.elapsed_ms;
            }
        }
        for (int lane = 0; lane < 2; ++lane)
            target_graph_c2_verifier_require_lane_equal(
                pair.c1.lanes[static_cast<std::size_t>(lane)],
                pair.c2.lanes[static_cast<std::size_t>(lane)],
                "T78 cached bridge matched rep " + std::to_string(rep) +
                    " lane " + std::to_string(lane));
        const auto write_arm = [&](const char* name,
                                   const TargetGraphC2VerifierArmTrace& trace,
                                   double attach_ms) {
            const auto retained = trace.lanes[0].retained_rows +
                trace.lanes[1].retained_rows;
            rounds << rep << ',' << (rep < kWarmups) << ','
                << (b_first ? "BA" : "AB") << ','
                << (c2_first ? "C2C1" : "C1C2") << ',' << name << ','
                << attach_ms << ',' << retained << ','
                << trace.complete_verifier_pair_ms << ',' << trace.full_read_pair_ms
                << ',' << retained * 1000.0 / trace.complete_verifier_pair_ms
                << ',' << retained * 1000.0 / trace.full_read_pair_ms << ','
                << trace.ready_free_bytes << ',' << trace.end_free_bytes << ','
                << trace.ready_owned_bytes << ',' << trace.end_owned_bytes << ','
                << trace.lanes[0].generation << ',' << trace.lanes[1].generation
                << ",1\n";
        };
        write_arm("C1", pair.c1, pair.c1_attach_ms);
        write_arm("C2", pair.c2, pair.c2_attach_ms);
        require(pair.c1.ready_owned_bytes == pair.c1.end_owned_bytes &&
                    pair.c2.ready_owned_bytes == pair.c2.end_owned_bytes &&
                    std::min({pair.c1.ready_free_bytes, pair.c1.end_free_bytes,
                              pair.c2.ready_free_bytes, pair.c2.end_free_bytes}) >=
                        (1ULL << 30),
                "T78 bridge memory gate");
        if (rep >= kWarmups) measured.push_back(std::move(pair));
    }

    std::vector<double> verifier_gains, full_gains;
    for (const auto& pair : measured) {
        const double verifier_gain =
            (pair.c1.complete_verifier_pair_ms - pair.c2.complete_verifier_pair_ms) *
            100.0 / pair.c1.complete_verifier_pair_ms;
        const double full_gain =
            (pair.c1.full_read_pair_ms - pair.c2.full_read_pair_ms) * 100.0 /
            pair.c1.full_read_pair_ms;
        verifier_gains.push_back(verifier_gain);
        full_gains.push_back(full_gain);
        require(verifier_gain >= -2.0 && full_gain >= -2.0,
                "T78 individual C2 performance guard");
    }
    const double verifier_median_gain = median(verifier_gains);
    const double full_median_gain = median(full_gains);
    require(verifier_median_gain > 0.0 && full_median_gain > 0.0,
            "T78 median C2 performance gate");

    for (int survivor = 0; survivor < 2; ++survivor) {
        t78_cached_prefix_c2_attach(lane_a, lane_b, compact_roots);
        const auto control = target_graph_c2_verifier_run_cancellation_arm(
            lane_a, lane_b, survivor, false, true);
        t78_cached_prefix_c2_attach(lane_a, lane_b, compact_roots);
        const auto candidate = target_graph_c2_verifier_run_cancellation_arm(
            lane_a, lane_b, survivor, true, true);
        target_graph_c2_verifier_require_lane_equal(
            control.survivor, candidate.survivor,
            "T78 cached cancellation survivor exact");
        target_graph_c2_require_snapshot_equal(
            control.canceled_root, candidate.canceled_root, false,
            "T78 cached cancellation peer exact");
        require(control.canceled_ring_base == candidate.canceled_ring_base &&
                    control.canceled_ring_count == candidate.canceled_ring_count &&
                    control.canceled_ring_digest == candidate.canceled_ring_digest,
                "T78 cached cancellation draft root exact");
        const double ratio = candidate.settle_ms / control.settle_ms;
        const auto growth = candidate.end_owned_bytes - candidate.ready_owned_bytes;
        const bool pass = ratio <= 1.25 && growth == 0 &&
            candidate.canceled_publications == 0 &&
            !candidate.canceled_graph_launched && candidate.transactions_closed;
        cancellation << (survivor ? 'B' : 'A') << ',' << control.settle_ms
            << ',' << candidate.settle_ms << ',' << ratio << ','
            << candidate.canceled_publications << ','
            << candidate.transactions_closed << ',' << growth << ',' << pass
            << '\n';
        require(pass, "T78 cached cancellation gate");
    }

    const auto final_lock = pin(&compact_roots);
    require(final_lock.observed.resident_tensor_bytes ==
                active_lock.observed.resident_tensor_bytes &&
                final_lock.observed.locked_tensor_bytes ==
                active_lock.observed.locked_tensor_bytes,
            "T78 active bridge resident/locked drift");
    memory_out << "final_bridge," << cache.storage_stats().payload_allocated_bytes
        << ',' << cache.storage_stats().identity_allocated_bytes << ','
        << cache.storage_stats().accounted_bytes << ','
        << final_lock.observed.resident_tensor_bytes << ','
        << final_lock.observed.locked_tensor_bytes << ','
        << final_lock.update.page_bytes << ',' << final_lock.update.new_locked_bytes
        << ',' << final_lock.update.unlocked_bytes << ',' << available() << ','
        << reserve << '\n';
    preparation.flush();
    rounds.flush();
    cancellation.flush();
    memory_out.flush();
    require(preparation.good() && rounds.good() && cancellation.good() &&
                memory_out.good(),
            "T78 evidence flush");
    residency.close();
    std::cout << "T78_CACHED_PREFIX_C2 PASS preparation_pairs=3"
              << " fresh_pair_median_ms=" << fresh_median
              << " cached_pair_median_ms=" << cached_median
              << " pair_reduction_percent="
              << (fresh_median - cached_median) * 100.0 / fresh_median
              << " avoided_input_tokens=7936 executed_input_tokens=256"
              << " c2_verifier_median_gain_percent=" << verifier_median_gain
              << " c2_full_median_gain_percent=" << full_median_gain
              << " exact=1 cancellation=2 resident_locked=1"
              << " cache_fill_ms=" << fill_ms
              << " output=" << output.string() << std::endl;
}
