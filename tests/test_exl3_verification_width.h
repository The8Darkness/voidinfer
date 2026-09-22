#pragma once

// Diagnostic sweep only. Every width starts from the same restored prefix and
// consumes the same teacher-token prefix. No drafter/acceptance effect is timed.
void run_verification_width_profile(Exl3TextModel& target,
                                    const std::vector<std::int64_t>& ids,
                                    const std::string& output) {
    require(target.max_context() == 4608 && ids.size() >= 4108 && !output.empty(),
            "verifywidth needs maxctx4608,4108 prompt IDs and output");
    require(env("NINFER_OSCAR_EXL3") == "1" &&
                env("NINFER_EXL3_TARGET_PROJECTION_TIMING") == "1",
            "verifywidth requires OSCAR and projection diagnostics");
    std::ofstream out(output), projections(output + ".projections.csv");
    require(out.good() && projections.good(), "verifywidth output open");
    out << "ctx,W,repeat,diagnostic,forward_us,transaction_roundtrip_us,correctness,route\n";
    projections << "ctx,W,layer,operator,rows,K,in_features,out_features,topology,calls,microseconds\n";
    const auto sync = [] { cuda_check(cudaStreamSynchronize(nullptr), "verifywidth sync"); };
    const auto state_equal = [&](Exl3TextContext& a, Exl3TextContext& b) {
        require_target_continue_state_equal(a, b, "verifywidth state", false);
        for (int layer = 0; layer < 64; ++layer) {
            if ((layer + 1) % 4 == 0) continue;
            require(a.gdn_physical_conv_host(layer) == b.gdn_physical_conv_host(layer),
                    "verifywidth physical convolution mismatch");
        }
        require(a.oscar_live_state_host_for_test() == b.oscar_live_state_host_for_test(),
                "verifywidth live OSCAR mismatch");
    };
    int cases = 0;
    for (const int ctx : {512, 4096}) {
        auto reference = target.create_context(true);
        auto candidate = target.create_context(true);
        require(reference->try_enable_oscar_from_environment() &&
                    candidate->try_enable_oscar_from_environment(), "verifywidth OSCAR");
        reference->prepare_transaction();
        candidate->prepare_transaction();
        candidate->prepare_continuation(8);
        candidate->prepare_target_projection_timing();
        const std::vector<std::int64_t> prefix(ids.begin(), ids.begin() + ctx);
        ingest_prefix(*reference, prefix, CommitSink{});
        ingest_prefix(*candidate, prefix, CommitSink{});
        state_equal(*candidate, *reference);
        for (const int width : {1, 2, 3, 4, 5, 6, 8}) {
            const std::span<const std::int64_t> teacher(ids.data() + ctx, width);
            const auto forward = [&] {
                if (width == 1) candidate->decode(teacher.front());
                else candidate->continue_rows(teacher);
            };
            // Untimed exact all-row oracle and persistent-state check.
            reference->begin_transaction();
            candidate->begin_transaction();
            std::vector<std::uint16_t> expected_logits;
            std::array<std::vector<std::uint16_t>, 5> expected_taps;
            for (const auto token : teacher) {
                reference->decode(token);
                const auto logits = target_continue_device_bits(reference->logits_device(),
                    kVocab, "verifywidth reference logits");
                expected_logits.insert(expected_logits.end(), logits.begin(), logits.end());
                for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
                    const auto values = target_continue_tap_bits(*reference, kTapLayers[tap], 1);
                    expected_taps[tap].insert(expected_taps[tap].end(), values.begin(), values.end());
                }
            }
            forward(); sync();
            const auto logits = width == 1
                ? target_continue_device_bits(candidate->logits_device(), kVocab, "verifywidth logits")
                : candidate->continuation_logits_bits_host();
            require_target_continue_exact(logits, expected_logits, kVocab, "verifywidth logits");
            for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap)
                require_target_continue_exact(target_continue_tap_bits(*candidate, kTapLayers[tap], width),
                    expected_taps[tap], kHidden, "verifywidth taps");
            state_equal(*candidate, *reference);
            // Check one following M1, outside the measured work.
            candidate->decode(ids[ctx + width]);
            reference->decode(ids[ctx + width]);
            sync(); state_equal(*candidate, *reference);
            candidate->rollback_transaction();
            reference->rollback_transaction();
            sync(); state_equal(*candidate, *reference);

            // Two warmups, five uninstrumented forwards, one separate diagnostic.
            for (int repeat = -2; repeat < 6; ++repeat) {
                const bool diagnostic = repeat == 5;
                if (diagnostic) candidate->begin_target_projection_timing_round(width);
                sync();
                const auto t0 = std::chrono::steady_clock::now();
                candidate->begin_transaction(); sync();
                const auto f0 = std::chrono::steady_clock::now();
                forward(); sync();
                const auto f1 = std::chrono::steady_clock::now();
                candidate->rollback_transaction(); sync();
                const auto t1 = std::chrono::steady_clock::now();
                if (repeat >= 0) {
                    out << ctx << ',' << width << ',' << repeat << ',' << diagnostic << ','
                        << std::fixed << std::setprecision(3)
                        << std::chrono::duration<double, std::micro>(f1-f0).count() << ','
                        << std::chrono::duration<double, std::micro>(t1-t0).count()
                        << ",PASS," << (width == 1 ? "ordinary_m1" : "bounded_continuation") << '\n';
                }
                if (diagnostic) {
                    const auto records = candidate->finish_target_projection_timing_round_after_synchronize();
                    require(records.size() == 401, "verifywidth projection inventory");
                    for (const auto& r : records)
                        projections << ctx << ',' << width << ',' << r.layer << ','
                            << ninfer::exl3::target_projection_operator_name(r.operation) << ','
                            << r.rows << ',' << r.K << ',' << r.in_features << ',' << r.out_features << ','
                            << ninfer::exl3::target_projection_topology_name(r.topology) << ','
                            << r.calls << ',' << r.microseconds << '\n';
                }
            }
            state_equal(*candidate, *reference);
            ++cases;
            out.flush(); projections.flush();
            std::cout << "VERIFYWIDTH_CASE PASS ctx=" << ctx << " W=" << width << std::endl;
        }
    }
    std::cout << "VERIFYWIDTH PASS cases=" << cases
              << " repeats=5 warmups=2 diagnostic=1 W12=unsupported W16=unsupported"
                 " scope=forward_and_transaction_roundtrip_not_full_caller\n";
}
