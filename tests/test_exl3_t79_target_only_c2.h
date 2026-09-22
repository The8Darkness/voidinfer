#pragma once
#include "test_exl3_host_residency.h"

struct T79TargetOnlySnapshot {
    int position = 0;
    int device_position = 0;
    std::vector<std::uint16_t> logits;
    std::array<std::vector<float>, 64> recurrent;
    std::array<std::vector<std::uint16_t>, 64> convolution;
    std::vector<std::byte> oscar;
    std::array<std::array<int, 5>, 16> oscar_counts{};
};

T79TargetOnlySnapshot t79_target_only_snapshot(
    Exl3TextContext& context, cudaStream_t stream) {
    cuda_check(cudaStreamSynchronize(stream), "T79 snapshot stream fence");
    T79TargetOnlySnapshot result;
    result.position = context.position();
    result.device_position = context.device_position_host(stream);
    result.logits = transaction_round_device_bits(
        context.logits_device(), kVocab, "T79 target-only logits");
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        result.recurrent[static_cast<std::size_t>(layer)] =
            context.gdn_state_host(layer, stream);
        result.convolution[static_cast<std::size_t>(layer)] =
            context.gdn_physical_conv_host(layer, stream);
    }
    const auto oscar = context.oscar_graph_state_host_for_test(stream);
    require(oscar.valid && oscar.observed_layers == oscar.device_counts.size(),
            "T79 graph OSCAR state observer: " + oscar.failure);
    result.oscar = oscar.canonical_bytes;
    result.oscar_counts = oscar.device_counts;
    return result;
}

void t79_require_target_only_snapshot_equal(
    const T79TargetOnlySnapshot& left, const T79TargetOnlySnapshot& right,
    const std::string& label) {
    require(left.position == right.position &&
                left.device_position == right.device_position &&
                left.logits == right.logits && left.convolution == right.convolution &&
                left.oscar == right.oscar &&
                left.oscar_counts == right.oscar_counts,
            label + " target-only persistent state mismatch");
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        require(target_graph_c2_float_bits_equal(
                    left.recurrent[static_cast<std::size_t>(layer)],
                    right.recurrent[static_cast<std::size_t>(layer)]),
                label + " recurrent mismatch layer=" + std::to_string(layer));
    }
}

struct T79TargetOnlyLane {
    std::unique_ptr<Exl3TextContext> context;
    TargetGraphC2Stream stream;
    std::vector<std::int64_t> prefix;
    std::vector<std::int64_t> tokens;
    std::vector<double> publication_ms;
    T79TargetOnlySnapshot snapshot;

    T79TargetOnlyLane(Exl3TextModel& model,
        std::vector<std::int64_t> request_prefix)
        : context(model.create_context(false)), prefix(std::move(request_prefix)) {
        require(!prefix.empty() && context->try_enable_oscar_from_environment(),
                "T79 target-only OSCAR lane construction");
    }
};

double t79_percentile95(std::vector<double> values) {
    require(!values.empty(), "T79 p95 input");
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
    return values[std::min(index, values.size() - 1)];
}

void t79_attach_target_only_lane(T79TargetOnlyLane& lane) {
    lane.context->reset(lane.stream.value);
    cuda_check(cudaStreamSynchronize(lane.stream.value),
               "T79 target-only reset");
    ingest_prefix(*lane.context, lane.prefix, CommitSink{});
    cuda_check(cudaStreamSynchronize(lane.stream.value),
               "T79 target-only fresh prefix completion");
    lane.tokens.clear();
    lane.publication_ms.clear();
    lane.tokens.push_back(sample_target(*lane.context, lane.stream.value));
    lane.context->oscar_set_graph_class(32);
    lane.context->oscar_sync_device_state(lane.stream.value);
    cuda_check(cudaStreamSynchronize(lane.stream.value),
               "T79 target-only graph state sync");
    require(lane.context->capture_decode_graph(lane.stream.value),
            "T79 target-only graph capture: " + lane.context->graph_status());
    cuda_check(cudaStreamSynchronize(lane.stream.value),
               "T79 target-only graph capture sync");
    require(lane.context->position() == static_cast<int>(lane.prefix.size()) &&
                lane.context->graph_active(),
            "T79 target-only prepared state");
}

struct T79TargetOnlyArm {
    std::array<std::vector<std::int64_t>, 2> tokens;
    std::array<T79TargetOnlySnapshot, 2> snapshots;
    std::array<double, 2> lane_ms{};
    std::array<double, 2> mean_interval_ms{};
    std::array<double, 2> p95_interval_ms{};
    double pair_ms = 0.0;
    std::size_t ready_free_bytes = 0;
    std::size_t end_free_bytes = 0;
    std::size_t ready_owned_bytes = 0;
    std::size_t end_owned_bytes = 0;
};

T79TargetOnlyArm t79_run_target_only_arm(
    T79TargetOnlyLane& a, T79TargetOnlyLane& b,
    bool overlap, bool b_first, int decodes = 64) {
    require(decodes > 0, "T79 target-only decode count");
    T79TargetOnlyLane* lanes[2]{&a, &b};
    for (auto* lane : lanes) t79_attach_target_only_lane(*lane);
    T79TargetOnlyArm result;
    cuda_check(cudaDeviceSynchronize(), "T79 target-only ready fence");
    std::size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&result.ready_free_bytes, &total_bytes),
               "T79 target-only ready memory");
    result.ready_owned_bytes = a.context->persistent_bytes() +
        b.context->persistent_bytes();
    require(result.ready_free_bytes >= (1ULL << 30),
            "T79 target-only 1 GiB ready reserve");
    const int order[2]{b_first ? 1 : 0, b_first ? 0 : 1};
    const auto pair_started = std::chrono::steady_clock::now();
    std::array<std::chrono::steady_clock::time_point, 2> lane_started{};
    if (!overlap) {
        for (const int index : order) {
            auto& lane = *lanes[index];
            lane_started[index] = std::chrono::steady_clock::now();
            for (int step = 0; step < decodes; ++step) {
                lane.context->decode_graph(lane.tokens.back(), lane.stream.value);
                lane.tokens.push_back(sample_target(*lane.context, lane.stream.value));
                lane.publication_ms.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - lane_started[index]).count());
            }
        }
    } else {
        lane_started[0] = lane_started[1] = std::chrono::steady_clock::now();
        for (int step = 0; step < decodes; ++step) {
            for (const int index : order) {
                auto& lane = *lanes[index];
                lane.context->decode_graph(lane.tokens.back(), lane.stream.value);
            }
            for (const int index : order) {
                auto& lane = *lanes[index];
                lane.tokens.push_back(sample_target(*lane.context, lane.stream.value));
                lane.publication_ms.push_back(std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - lane_started[index]).count());
            }
        }
    }
    result.pair_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pair_started).count();
    for (int index = 0; index < 2; ++index) {
        auto& lane = *lanes[index];
        result.lane_ms[index] = lane.publication_ms.back();
        std::vector<double> intervals;
        intervals.reserve(lane.publication_ms.size());
        double previous = 0.0;
        for (const double stamp : lane.publication_ms) {
            intervals.push_back(stamp - previous);
            previous = stamp;
        }
        result.mean_interval_ms[index] = std::accumulate(
            intervals.begin(), intervals.end(), 0.0) / intervals.size();
        result.p95_interval_ms[index] = t79_percentile95(std::move(intervals));
        require(lane.tokens.size() == static_cast<std::size_t>(decodes + 1) &&
                    lane.context->position() ==
                        static_cast<int>(lane.prefix.size()) + decodes,
                "T79 target-only publication/position count");
        result.tokens[index] = lane.tokens;
        result.snapshots[index] = t79_target_only_snapshot(
            *lane.context, lane.stream.value);
    }
    result.end_owned_bytes = a.context->persistent_bytes() +
        b.context->persistent_bytes();
    cuda_check(cudaDeviceSynchronize(), "T79 target-only end fence");
    cuda_check(cudaMemGetInfo(&result.end_free_bytes, &total_bytes),
               "T79 target-only end memory");
    require(result.ready_owned_bytes == result.end_owned_bytes &&
                result.end_free_bytes >= (1ULL << 30),
            "T79 target-only storage/reserve gate");
    return result;
}

void run_t79_target_only_c2(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    constexpr int kPrompt = 4096;
    constexpr int kDecodes = 64;
    constexpr int kWarmups = 2;
    constexpr int kMeasured = 4;
    require(target.max_context() >= 4352 && code.size() >= kPrompt &&
                prose.size() >= kPrompt,
            "T79 target-only fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T79 output must be new");
    std::filesystem::create_directories(output);
    std::array<std::vector<std::int64_t>, 2> prompts{
        std::vector<std::int64_t>(code.begin(), code.begin() + kPrompt),
        std::vector<std::int64_t>(prose.begin(), prose.begin() + kPrompt)};

    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "0");
    T79TargetOnlyLane lane_a(target, prompts[0]);
    T79TargetOnlyLane lane_b(target, prompts[1]);
    std::ofstream arms(output / "arms.csv");
    std::ofstream tokens(output / "tokens.csv");
    std::ofstream cancellation(output / "cancellation.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(arms.good() && tokens.good() && cancellation.good() &&
                memory_out.good(),
            "T79 evidence open");
    arms << "rep,warmup,request_order,arm_order,arm,pair_ms,aggregate_output_s,"
            "lane_a_ms,lane_b_ms,lane_a_output_s,lane_b_output_s,"
            "lane_latency_ratio,mean_interval_a_ms,mean_interval_b_ms,"
            "p95_interval_a_ms,p95_interval_b_ms,ready_free_bytes,end_free_bytes,"
            "ready_owned_bytes,end_owned_bytes,exact\n";
    tokens << "rep,arm,lane,ordinal,token_id\n";
    cancellation << "survivor,cancel_after,canceled_publications,"
                    "canceled_state_stable,survivor_exact,ready_owned_bytes,"
                    "end_owned_bytes,min_free_bytes,pass\n";
    memory_out << "authoritative_host_root_bytes,draft_loaded,"
                  "target_contexts,device_reserve_bytes\n0,0,2,1073741824\n";

    struct Pair { T79TargetOnlyArm c1, c2; };
    std::vector<Pair> measured;
    std::size_t warm_end_free = 0;
    for (int rep = 0; rep < kWarmups + kMeasured; ++rep) {
        Pair pair;
        const bool b_first = (rep & 1) != 0;
        const bool c2_first = (rep & 1) != 0;
        for (int arm_index = 0; arm_index < 2; ++arm_index) {
            const bool overlap = c2_first ? arm_index == 0 : arm_index == 1;
            auto trace = t79_run_target_only_arm(
                lane_a, lane_b, overlap, b_first, kDecodes);
            (overlap ? pair.c2 : pair.c1) = std::move(trace);
        }
        for (int lane = 0; lane < 2; ++lane) {
            require(pair.c1.tokens[lane] == pair.c2.tokens[lane],
                    "T79 C1/C2 token mismatch");
            t79_require_target_only_snapshot_equal(
                pair.c1.snapshots[lane], pair.c2.snapshots[lane],
                "T79 matched rep " + std::to_string(rep) +
                    " lane " + std::to_string(lane));
        }
        const auto write = [&](const char* name, const T79TargetOnlyArm& trace) {
            const double low = std::min(trace.lane_ms[0], trace.lane_ms[1]);
            const double high = std::max(trace.lane_ms[0], trace.lane_ms[1]);
            arms << rep << ',' << (rep < kWarmups) << ','
                << (b_first ? "BA" : "AB") << ','
                << (c2_first ? "C2C1" : "C1C2") << ',' << name << ','
                << trace.pair_ms << ',' << 2 * kDecodes * 1000.0 / trace.pair_ms
                << ',' << trace.lane_ms[0] << ',' << trace.lane_ms[1] << ','
                << kDecodes * 1000.0 / trace.lane_ms[0] << ','
                << kDecodes * 1000.0 / trace.lane_ms[1] << ',' << high / low
                << ',' << trace.mean_interval_ms[0] << ','
                << trace.mean_interval_ms[1] << ',' << trace.p95_interval_ms[0]
                << ',' << trace.p95_interval_ms[1] << ','
                << trace.ready_free_bytes << ',' << trace.end_free_bytes << ','
                << trace.ready_owned_bytes << ',' << trace.end_owned_bytes
                << ",1\n";
            for (int lane = 0; lane < 2; ++lane)
                for (std::size_t ordinal = 0;
                     ordinal < trace.tokens[lane].size(); ++ordinal)
                    tokens << rep << ',' << name << ',' << (lane ? 'B' : 'A')
                        << ',' << ordinal << ',' << trace.tokens[lane][ordinal]
                        << '\n';
        };
        write("C1", pair.c1);
        write("C2", pair.c2);
        const double c2_ratio = std::max(pair.c2.lane_ms[0], pair.c2.lane_ms[1]) /
            std::min(pair.c2.lane_ms[0], pair.c2.lane_ms[1]);
        require(c2_ratio <= 1.15, "T79 C2 lane fairness gate");
        const auto current_end_free = std::min(
            pair.c1.end_free_bytes, pair.c2.end_free_bytes);
        if (rep == kWarmups - 1) warm_end_free = current_end_free;
        if (rep >= kWarmups) {
            require(current_end_free + (1ULL << 20) >= warm_end_free,
                    "T79 post-warmup physical memory drift");
            measured.push_back(std::move(pair));
        }
    }
    std::vector<double> aggregate_gains;
    for (const auto& pair : measured) {
        const double gain = (pair.c1.pair_ms / pair.c2.pair_ms - 1.0) * 100.0;
        aggregate_gains.push_back(gain);
        require(gain >= -2.0, "T79 individual aggregate performance guard");
    }
    const auto sorted_gains = [&] {
        auto result = aggregate_gains;
        std::sort(result.begin(), result.end());
        return result;
    }();
    const double median_gain =
        (sorted_gains[1] + sorted_gains[2]) / 2.0;
    require(median_gain > 0.0, "T79 aggregate C2 median performance gate");

    for (int survivor_index = 0; survivor_index < 2; ++survivor_index) {
        T79TargetOnlyLane* lanes[2]{&lane_a, &lane_b};
        t79_attach_target_only_lane(lane_a);
        t79_attach_target_only_lane(lane_b);
        auto& control = *lanes[survivor_index];
        for (int step = 0; step < kDecodes; ++step) {
            control.context->decode_graph(control.tokens.back(), control.stream.value);
            control.tokens.push_back(sample_target(*control.context, control.stream.value));
        }
        const auto control_tokens = control.tokens;
        const auto control_state = t79_target_only_snapshot(
            *control.context, control.stream.value);

        t79_attach_target_only_lane(lane_a);
        t79_attach_target_only_lane(lane_b);
        auto& survivor = *lanes[survivor_index];
        auto& canceled = *lanes[1 - survivor_index];
        for (int step = 0; step < 16; ++step) {
            lane_a.context->decode_graph(lane_a.tokens.back(), lane_a.stream.value);
            lane_b.context->decode_graph(lane_b.tokens.back(), lane_b.stream.value);
            lane_a.tokens.push_back(sample_target(*lane_a.context, lane_a.stream.value));
            lane_b.tokens.push_back(sample_target(*lane_b.context, lane_b.stream.value));
        }
        const auto canceled_tokens = canceled.tokens;
        const auto canceled_state = t79_target_only_snapshot(
            *canceled.context, canceled.stream.value);
        const auto ready_owned = lane_a.context->persistent_bytes() +
            lane_b.context->persistent_bytes();
        std::size_t ready_free = 0, total_bytes = 0;
        cuda_check(cudaMemGetInfo(&ready_free, &total_bytes),
                   "T79 cancellation ready memory");
        for (int step = 16; step < kDecodes; ++step) {
            survivor.context->decode_graph(
                survivor.tokens.back(), survivor.stream.value);
            survivor.tokens.push_back(sample_target(
                *survivor.context, survivor.stream.value));
        }
        const auto canceled_after = t79_target_only_snapshot(
            *canceled.context, canceled.stream.value);
        const auto survivor_state = t79_target_only_snapshot(
            *survivor.context, survivor.stream.value);
        const auto end_owned = lane_a.context->persistent_bytes() +
            lane_b.context->persistent_bytes();
        std::size_t end_free = 0;
        cuda_check(cudaMemGetInfo(&end_free, &total_bytes),
                   "T79 cancellation end memory");
        const bool canceled_stable = canceled.tokens == canceled_tokens;
        t79_require_target_only_snapshot_equal(
            canceled_after, canceled_state, "T79 canceled lane stable");
        const bool survivor_exact = survivor.tokens == control_tokens;
        require(survivor_exact, "T79 cancellation survivor token mismatch");
        t79_require_target_only_snapshot_equal(
            survivor_state, control_state, "T79 cancellation survivor exact");
        const bool pass = canceled_stable && survivor_exact &&
            canceled.tokens.size() == 17 && ready_owned == end_owned &&
            std::min(ready_free, end_free) >= (1ULL << 30);
        cancellation << (survivor_index ? 'B' : 'A') << ",16,16,"
            << canceled_stable << ',' << survivor_exact << ',' << ready_owned
            << ',' << end_owned << ',' << std::min(ready_free, end_free) << ','
            << pass << '\n';
        require(pass, "T79 target-only cancellation gate");
    }
    arms.flush(); tokens.flush(); cancellation.flush(); memory_out.flush();
    require(arms.good() && tokens.good() && cancellation.good() &&
                memory_out.good(),
            "T79 evidence flush");
    std::cout << "T79_TARGET_ONLY_C2 PASS warmups=2 measured_pairs=4"
              << " outputs_per_lane=64 aggregate_median_gain_percent="
              << median_gain << " exact=1 cancellation=2 host_root_bytes=0"
              << " target_only=1 draft_loaded=0 output=" << output.string()
              << std::endl;
}
