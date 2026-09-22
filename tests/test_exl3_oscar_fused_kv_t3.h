#pragma once

void run_oscar_fused_kv_t3(Exl3TextModel& target,
                            const std::vector<std::int64_t>& code,
                            const std::vector<std::int64_t>& prose,
                            const std::vector<std::int64_t>& structured,
                            const std::vector<std::int64_t>& heldout) {
    require(target.max_context() >= 4224 && env("NINFER_OSCAR_EXL3") == "1" &&
                env("NINFER_EXL3_PREFILL_BATCH_ROTATIONS") == "1",
            "fused KV T3 prerequisites");
    constexpr int prefix = 4096, decode_rows = 128;
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    const std::array fixtures{
        std::pair{"code", &code}, std::pair{"prose", &prose},
        std::pair{"structured", &structured}, std::pair{"heldout", &heldout}};
    for (const auto& fixture : fixtures) {
        require(fixture.second->size() >= prefix, "fused KV T3 fixture extent");
        std::vector<double> control_prefill, candidate_prefill;
        std::vector<double> control_decode, candidate_decode;
        for (int rep = 0; rep < 3; ++rep) {
            _putenv_s("NINFER_OSCAR_DECODE_FUSED_KV_ROTATIONS", "0");
            auto control = target.create_context(true);
            require(control->try_enable_oscar_from_environment(), "fused KV T3 control OSCAR");
            _putenv_s("NINFER_OSCAR_DECODE_FUSED_KV_ROTATIONS", "1");
            auto candidate = target.create_context(true);
            require(candidate->try_enable_oscar_from_environment(), "fused KV T3 candidate OSCAR");
            _putenv_s("NINFER_OSCAR_DECODE_FUSED_KV_ROTATIONS", "0");
            struct Timing { double prefill_ms = 0.0, decode_ms = 0.0; };
            const auto run = [&](Exl3TextContext& context) {
                const std::span<const std::int64_t> input(fixture.second->data(), prefix);
                auto started = std::chrono::steady_clock::now();
                context.prefill(input.first(16));
                for (int consumed = 16; consumed < prefix;) {
                    const int rows = std::min(1024, prefix - consumed);
                    context.append_prefill_wide(input.subspan(
                        static_cast<std::size_t>(consumed), static_cast<std::size_t>(rows)));
                    consumed += rows;
                }
                cuda_check(cudaDeviceSynchronize(), "fused KV T3 prefill ready");
                const double prefill_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                started = std::chrono::steady_clock::now();
                for (int row = 0; row < decode_rows; ++row)
                    context.decode((*fixture.second)[static_cast<std::size_t>(row)]);
                cuda_check(cudaDeviceSynchronize(), "fused KV T3 decode ready");
                return Timing{prefill_ms, std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count()};
            };
            const bool candidate_first = rep == 1;
            Timing old_timing, new_timing;
            if (candidate_first) { new_timing = run(*candidate); old_timing = run(*control); }
            else { old_timing = run(*control); new_timing = run(*candidate); }
            require_target_continue_state_equal(*candidate, *control,
                                                "fused KV T3 final state", false);
            for (int layer = 0; layer < 64; ++layer) if ((layer + 1) % 4 != 0)
                require(candidate->gdn_physical_conv_host(layer) ==
                            control->gdn_physical_conv_host(layer),
                        "fused KV T3 physical convolution");
            require(candidate->oscar_live_state_host_for_test() ==
                        control->oscar_live_state_host_for_test(),
                    "fused KV T3 OSCAR state");
            control_prefill.push_back(old_timing.prefill_ms);
            candidate_prefill.push_back(new_timing.prefill_ms);
            control_decode.push_back(old_timing.decode_ms);
            candidate_decode.push_back(new_timing.decode_ms);
            std::cout << "OSCAR_FUSED_KV_T3_CASE fixture=" << fixture.first
                      << " rep=" << rep << " order=" << (candidate_first ? "BA" : "AB")
                      << " control_prefill_ms=" << old_timing.prefill_ms
                      << " candidate_prefill_ms=" << new_timing.prefill_ms
                      << " control_decode_ms=" << old_timing.decode_ms
                      << " candidate_decode_ms=" << new_timing.decode_ms << std::endl;
        }
        const double old_prefill = median(control_prefill), new_prefill = median(candidate_prefill);
        const double old_decode = median(control_decode), new_decode = median(candidate_decode);
        std::cout << "OSCAR_FUSED_KV_T3 fixture=" << fixture.first
                  << " control_prefill_median_ms=" << old_prefill
                  << " candidate_prefill_median_ms=" << new_prefill
                  << " prefill_wall_reduction_percent=" << (old_prefill-new_prefill)*100.0/old_prefill
                  << " candidate_prefill_tps=" << prefix*1000.0/new_prefill
                  << " control_decode_median_ms=" << old_decode
                  << " candidate_decode_median_ms=" << new_decode
                  << " decode_wall_reduction_percent=" << (old_decode-new_decode)*100.0/old_decode
                  << " candidate_decode_tps=" << decode_rows*1000.0/new_decode << std::endl;
    }
    std::cout << "OSCAR_FUSED_KV_T3 PASS fixtures=4 pairs=3 prefix=4096 decode_rows=128"
              << " exact_logits_gdn_physical_oscar=1 orders=AB_BA_AB" << std::endl;
}
