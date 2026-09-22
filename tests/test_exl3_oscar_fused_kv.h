#pragma once

void run_oscar_fused_kv_screen(Exl3TextModel& target,
                               const std::vector<std::int64_t>& code,
                               const std::vector<std::int64_t>& prose) {
    require(env("NINFER_OSCAR_EXL3") == "1", "fused KV screen requires OSCAR");
    require(env("NINFER_EXL3_PREFILL_BATCH_ROTATIONS") == "1",
            "fused KV screen requires batch rotations");
    const int prefix = target.max_context() >= 4352 ? 4096 : 96;
    constexpr int decode_rows = 32;
    require(target.max_context() >= prefix + decode_rows &&
                code.size() >= static_cast<std::size_t>(prefix) &&
                prose.size() >= static_cast<std::size_t>(prefix),
            "fused KV fixture/context extent");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");

    for (const auto& fixture : std::array<
             std::pair<const char*, const std::vector<std::int64_t>*>, 2>{
             std::pair{"code", &code}, std::pair{"prose", &prose}}) {
        _putenv_s("NINFER_OSCAR_DECODE_FUSED_KV_ROTATIONS", "0");
        auto control = target.create_context(true);
        require(control->try_enable_oscar_from_environment(), "control OSCAR enable");
        _putenv_s("NINFER_OSCAR_DECODE_FUSED_KV_ROTATIONS", "1");
        auto candidate = target.create_context(true);
        require(candidate->try_enable_oscar_from_environment(), "candidate OSCAR enable");
        // Prove the switch is construction-latched: subsequent environment
        // changes cannot alter either live context.
        _putenv_s("NINFER_OSCAR_DECODE_FUSED_KV_ROTATIONS", "0");

        const std::span<const std::int64_t> input(fixture.second->data(), prefix);
        const auto prefill = [&](Exl3TextContext& context) {
            const auto started = std::chrono::steady_clock::now();
            context.prefill(input.first(16));
            for (int consumed = 16; consumed < prefix;) {
                const int rows = std::min(1024, prefix - consumed);
                context.append_prefill_wide(input.subspan(
                    static_cast<std::size_t>(consumed), static_cast<std::size_t>(rows)));
                consumed += rows;
            }
            cuda_check(cudaDeviceSynchronize(), "fused KV prefill ready");
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        };
        const double control_prefill_ms = prefill(*control);
        const double candidate_prefill_ms = prefill(*candidate);
        require_target_continue_state_equal(*candidate, *control,
                                            "fused KV post-prefill state", false);
        for (int layer = 0; layer < 64; ++layer) if ((layer + 1) % 4 != 0)
            require(candidate->gdn_physical_conv_host(layer) ==
                        control->gdn_physical_conv_host(layer),
                    "fused KV post-prefill physical convolution");
        require(candidate->oscar_live_state_host_for_test() ==
                    control->oscar_live_state_host_for_test(),
                "fused KV post-prefill OSCAR state");

        const auto decode = [&](Exl3TextContext& context) {
            const auto started = std::chrono::steady_clock::now();
            for (int row = 0; row < decode_rows; ++row)
                context.decode((*fixture.second)[static_cast<std::size_t>(row)]);
            cuda_check(cudaDeviceSynchronize(), "fused KV decode ready");
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        };
        const double control_decode_ms = decode(*control);
        const double candidate_decode_ms = decode(*candidate);
        require_target_continue_state_equal(*candidate, *control,
                                            "fused KV post-decode state", false);
        for (int layer = 0; layer < 64; ++layer) if ((layer + 1) % 4 != 0)
            require(candidate->gdn_physical_conv_host(layer) ==
                        control->gdn_physical_conv_host(layer),
                    "fused KV post-decode physical convolution");
        require(candidate->oscar_live_state_host_for_test() ==
                    control->oscar_live_state_host_for_test(),
                "fused KV post-decode OSCAR state");
        std::cout << "OSCAR_FUSED_KV fixture=" << fixture.first << " prefix=" << prefix
                  << " decode_rows=" << decode_rows
                  << " control_prefill_ms=" << control_prefill_ms
                  << " candidate_prefill_ms=" << candidate_prefill_ms
                  << " control_decode_ms=" << control_decode_ms
                  << " candidate_decode_ms=" << candidate_decode_ms << std::endl;
    }
    std::cout << "OSCAR_FUSED_KV PASS fixtures=2 prefix=" << prefix
              << " decode_rows=" << decode_rows
              << " exact_logits_gdn_physical_oscar=1 control=two_launches"
              << " candidate=one_disjoint_cta_grid" << std::endl;
}
