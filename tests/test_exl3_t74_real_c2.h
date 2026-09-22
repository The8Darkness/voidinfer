#pragma once

void run_t74_real_c2(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft_a,
    const std::filesystem::path& draft_path,
    const std::vector<std::int64_t>& source,
    const std::filesystem::path& output) {
    const std::string arm_order = env("NINFER_T74_ARM_ORDER");
    require(arm_order == "C1C2" || arm_order == "C2C1",
            "T74 arm order must be C1C2 or C2C1");
    require(target.max_context() >= 4352 && source.size() >= 4113,
            "T74 requires maxctx4352 and a 4113-token source");
    require(!output.empty() && !std::filesystem::exists(output),
            "T74 output must be new");
    std::filesystem::create_directories(output);
    std::ofstream rounds(output / "rounds.csv");
    std::ofstream summary(output / "summary.csv");
    std::ofstream cancellation(output / "cancellation.csv");
    require(rounds.good() && summary.good() && cancellation.good(),
            "T74 output creation");
    rounds << "arm_order,arm,round,warmup,request_order,lane,proposals,accepted,"
              "retained_rows,emitted,pending,position,ring_base,ring_count,"
              "publication_count,generation,route_bits,proposal_ms,verifier_ms,"
              "full_read_ms,pair_verifier_ms,pair_full_read_ms,ready_free_bytes,"
              "end_free_bytes,ready_owned_bytes,end_owned_bytes\n";
    summary << "arm_order,arm,rounds,warmups,measured_rounds,retained_rows,"
               "verifier_ms,full_read_ms,verifier_rows_per_s,full_rows_per_s,"
               "lane_a_rows_per_s,lane_b_rows_per_s,max_lane_latency_ratio,"
               "ready_owned_bytes_min,ready_owned_bytes_max,end_owned_bytes_min,"
               "end_owned_bytes_max,min_free_bytes,generation_a,generation_b\n";
    cancellation << "arm_order,survivor,control_ms,c2_ms,ratio,"
                    "control_publications,c2_publications,canceled_publications,"
                    "canceled_graph_launched,transactions_closed,logical_growth_bytes,pass\n";

    auto draft_b = Exl3Dflash2DraftModel::load(draft_path);
    TargetGraphC2VerifierLane a(
        target, draft_a, target_graph_c2_shifted_prefix(source, 4096, 0));
    TargetGraphC2VerifierLane b(
        target, *draft_b, target_graph_c2_shifted_prefix(source, 4096, 17));
    struct Arm {
        std::string name;
        std::vector<TargetGraphC2VerifierArmTrace> rounds;
    };
    Arm c1{"C1", {}}, c2{"C2", {}};
    Arm* arms[2] = {
        arm_order == "C1C2" ? &c1 : &c2,
        arm_order == "C1C2" ? &c2 : &c1
    };
    constexpr int kRounds = 12;
    constexpr int kWarmups = 4;
    for (Arm* arm : arms) {
        const bool overlap = arm->name == "C2";
        for (int round = 0; round < kRounds; ++round) {
            const bool b_first = (round & 1) != 0;
            arm->rounds.push_back(target_graph_c2_verifier_run_arm(
                a, b, overlap, b_first, false, round != 0));
        }
    }

    const auto write_arm = [&](const Arm& arm) {
        std::uint64_t retained = 0;
        double verifier_ms = 0.0, full_ms = 0.0;
        std::array<std::uint64_t, 2> lane_rows{};
        std::array<double, 2> lane_ms{};
        double max_lane_ratio = 0.0;
        std::size_t ready_owned_min = std::numeric_limits<std::size_t>::max();
        std::size_t ready_owned_max = 0;
        std::size_t end_owned_min = std::numeric_limits<std::size_t>::max();
        std::size_t end_owned_max = 0;
        std::size_t min_free = std::numeric_limits<std::size_t>::max();
        for (int round = 0; round < kRounds; ++round) {
            const auto& trace = arm.rounds[static_cast<std::size_t>(round)];
            require(trace.ready_owned_bytes == trace.end_owned_bytes,
                    "T74 candidate-owned byte growth");
            ready_owned_min = std::min(ready_owned_min, trace.ready_owned_bytes);
            ready_owned_max = std::max(ready_owned_max, trace.ready_owned_bytes);
            end_owned_min = std::min(end_owned_min, trace.end_owned_bytes);
            end_owned_max = std::max(end_owned_max, trace.end_owned_bytes);
            min_free = std::min({min_free, trace.ready_free_bytes,
                                 trace.end_free_bytes});
            for (int lane = 0; lane < 2; ++lane) {
                const auto& row = trace.lanes[static_cast<std::size_t>(lane)];
                rounds << arm_order << ',' << arm.name << ',' << round << ','
                       << (round < kWarmups) << ',' << ((round & 1) ? "BA" : "AB")
                       << ',' << (lane ? 'B' : 'A') << ','
                       << target_graph_c2_verifier_ids(row.proposals) << ','
                       << row.accepted << ',' << row.retained_rows << ','
                       << target_graph_c2_verifier_ids(row.emitted) << ','
                       << row.pending << ',' << row.position << ',' << row.ring_base
                       << ',' << row.ring_count << ',' << row.publication_count
                       << ',' << row.generation << ',' << row.route_bits << ','
                       << row.proposal_ms << ',' << row.verifier_latency_ms << ','
                       << row.full_read_latency_ms << ','
                       << trace.complete_verifier_pair_ms << ','
                       << trace.full_read_pair_ms << ',' << trace.ready_free_bytes
                       << ',' << trace.end_free_bytes << ',' << trace.ready_owned_bytes
                       << ',' << trace.end_owned_bytes << '\n';
                if (round >= kWarmups) {
                    retained += static_cast<std::uint64_t>(row.retained_rows);
                    lane_rows[static_cast<std::size_t>(lane)] +=
                        static_cast<std::uint64_t>(row.retained_rows);
                    lane_ms[static_cast<std::size_t>(lane)] += row.full_read_latency_ms;
                }
            }
            if (round >= kWarmups) {
                verifier_ms += trace.complete_verifier_pair_ms;
                full_ms += trace.full_read_pair_ms;
                const double low = std::min(trace.lanes[0].verifier_latency_ms,
                                            trace.lanes[1].verifier_latency_ms);
                const double high = std::max(trace.lanes[0].verifier_latency_ms,
                                             trace.lanes[1].verifier_latency_ms);
                max_lane_ratio = std::max(max_lane_ratio, high / low);
            }
        }
        require(min_free >= (1ULL << 30), "T74 existing 1 GiB device reserve");
        const auto generation_a = arm.rounds.front().lanes[0].generation;
        const auto generation_b = arm.rounds.front().lanes[1].generation;
        for (const auto& trace : arm.rounds) {
            require(trace.lanes[0].generation == generation_a &&
                        trace.lanes[1].generation == generation_b,
                    "T74 graph recaptured inside persistent arm");
        }
        const double verifier_rate = retained * 1000.0 / verifier_ms;
        const double full_rate = retained * 1000.0 / full_ms;
        const double lane_a_rate = lane_rows[0] * 1000.0 / lane_ms[0];
        const double lane_b_rate = lane_rows[1] * 1000.0 / lane_ms[1];
        summary << arm_order << ',' << arm.name << ',' << kRounds << ','
                << kWarmups << ',' << (kRounds - kWarmups) << ',' << retained
                << ',' << verifier_ms << ',' << full_ms << ',' << verifier_rate
                << ',' << full_rate << ',' << lane_a_rate << ',' << lane_b_rate
                << ',' << max_lane_ratio << ',' << ready_owned_min << ','
                << ready_owned_max << ',' << end_owned_min << ',' << end_owned_max
                << ',' << min_free << ',' << generation_a << ',' << generation_b
                << '\n';
    };

    for (int round = 0; round < kRounds; ++round) {
        for (int lane = 0; lane < 2; ++lane) {
            target_graph_c2_verifier_require_lane_equal(
                c1.rounds[static_cast<std::size_t>(round)].lanes[static_cast<std::size_t>(lane)],
                c2.rounds[static_cast<std::size_t>(round)].lanes[static_cast<std::size_t>(lane)],
                "T74 persistent matched round " + std::to_string(round) +
                    " lane " + std::to_string(lane));
        }
    }
    write_arm(c1);
    write_arm(c2);

    for (int survivor = 0; survivor < 2; ++survivor) {
        const auto control = target_graph_c2_verifier_run_cancellation_arm(
            a, b, survivor, false);
        const auto candidate = target_graph_c2_verifier_run_cancellation_arm(
            a, b, survivor, true);
        target_graph_c2_verifier_require_lane_equal(
            control.survivor, candidate.survivor,
            "T74 cancellation survivor exact");
        target_graph_c2_require_snapshot_equal(
            control.canceled_root, candidate.canceled_root, false,
            "T74 cancellation peer root exact");
        const double ratio = candidate.settle_ms / control.settle_ms;
        const auto growth = candidate.end_owned_bytes - candidate.ready_owned_bytes;
        const bool pass = ratio <= 1.25 && growth == 0 &&
            candidate.canceled_publications == 0 &&
            !candidate.canceled_graph_launched && candidate.transactions_closed;
        cancellation << arm_order << ',' << (survivor ? 'B' : 'A') << ','
                     << control.settle_ms << ',' << candidate.settle_ms << ','
                     << ratio << ',' << control.survivor.publication_count << ','
                     << candidate.survivor.publication_count << ','
                     << candidate.canceled_publications << ','
                     << candidate.canceled_graph_launched << ','
                     << candidate.transactions_closed << ',' << growth << ','
                     << pass << '\n';
        require(pass, "T74 cancellation/fairness gate");
    }
    rounds.flush(); summary.flush(); cancellation.flush();
    require(rounds.good() && summary.good() && cancellation.good(),
            "T74 output write");
    std::cout << "T74_REAL_C2 PASS arm_order=" << arm_order
              << " rounds=12 warmups=4 measured=8 exact=1 cancellation=2"
                 " logical_growth=0 reserve=1GiB\n";
}
