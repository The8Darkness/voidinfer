#pragma once

struct GdnStageOracleState {
    std::vector<float> recurrent;
    std::vector<std::uint16_t> convolution;
    int host_position = -1;
    int device_position = -1;
    std::int64_t pending = -1;
};

GdnStageOracleState gdn_stage_oracle_state(Exl3TextContext& context,
                                            bool include_pending = true) {
    GdnStageOracleState state;
    state.recurrent = context.gdn_state_host(0);
    state.convolution = context.gdn_physical_conv_host(0);
    state.host_position = context.position();
    state.device_position = context.device_position_host();
    if (include_pending) state.pending = sample_target(context);
    return state;
}

void require_gdn_stage_oracle_state(const GdnStageOracleState& actual,
                                    const GdnStageOracleState& expected,
                                    const std::string& label) {
    require(actual.recurrent.size() == expected.recurrent.size() &&
                std::memcmp(actual.recurrent.data(), expected.recurrent.data(),
                            actual.recurrent.size() * sizeof(float)) == 0,
            label + " recurrent state mismatch");
    require(actual.convolution == expected.convolution,
            label + " physical convolution state mismatch");
    require(actual.host_position == expected.host_position,
            label + " host position mismatch");
    require(actual.device_position == expected.device_position,
            label + " device position mismatch");
    require(actual.pending == expected.pending,
            label + " pending token mismatch");
}

void require_gdn_stage_oracle_output(const std::vector<std::uint16_t>& actual,
                                     const std::vector<std::uint16_t>& expected,
                                     const std::string& label) {
    require(actual.size() == 8ULL * kHidden && actual.size() == expected.size(),
            label + " output extent mismatch");
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end()) {
        const std::size_t index = static_cast<std::size_t>(mismatch.first - actual.begin());
        throw std::runtime_error(label + " output mismatch row=" +
            std::to_string(index / kHidden) + " feature=" +
            std::to_string(index % kHidden) + " actual=" +
            std::to_string(*mismatch.first) + " expected=" +
            std::to_string(expected[index]));
    }
}

void require_gdn_stage_oracle_commit_rejected(Exl3TextContext& context,
                                              const std::string& label) {
    std::string failure;
    try {
        context.commit_transaction();
    } catch (const std::exception& error) {
        failure = error.what();
    }
    require(failure.find("requires rollback") != std::string::npos,
            label + " did not reject commit: " + failure);
    require(context.transaction_active(),
            label + " commit rejection consumed the rollback snapshot");
}

void require_gdn_stage_oracle_telemetry(
    const ninfer::exl3::Exl3GdnStageOracleTelemetry& telemetry,
    const std::string& label) {
    require(telemetry.staged_pairs == 1 && telemetry.projection_phases == 6 &&
                telemetry.serial_b8_projection_calls == 12 &&
                telemetry.fallback_calls == 0 && telemetry.published_lanes == 2,
            label + " topology telemetry mismatch staged=" +
                std::to_string(telemetry.staged_pairs) + " phases=" +
                std::to_string(telemetry.projection_phases) + " serial_b8=" +
                std::to_string(telemetry.serial_b8_projection_calls) + " fallback=" +
                std::to_string(telemetry.fallback_calls) + " published=" +
                std::to_string(telemetry.published_lanes));
}

void run_gdn_stage_oracle(Exl3TextModel& target,
                          const std::vector<std::int64_t>& source) {
    constexpr std::size_t kPeerOffset = 512;
    constexpr int kFirstPosition = 63;
    constexpr int kSecondPosition = 319;
    constexpr int kRows = 8;
    require(target.max_context() >= kSecondPosition + kRows,
            "gdnstageoracle target context is too small");
    require(source.size() >= kPeerOffset + kSecondPosition + kRows,
            "gdnstageoracle source fixture is too short");
    require(env("NINFER_OSCAR_EXL3") == "1",
            "gdnstageoracle requires canonical OSCAR for rollback snapshots");

    const auto make_slice = [&](std::size_t offset, int rows) {
        std::vector<std::int64_t> values(
            source.begin() + static_cast<std::ptrdiff_t>(offset),
            source.begin() + static_cast<std::ptrdiff_t>(offset + rows));
        require(std::all_of(values.begin(), values.end(), [](std::int64_t token) {
                    return token >= 0 && token < kVocab && token != kMaskToken;
                }), "gdnstageoracle source slice contains an invalid token");
        return values;
    };
    const auto first_prefix = make_slice(0, kFirstPosition);
    const auto first_ids = make_slice(kFirstPosition, kRows);
    const auto second_prefix = make_slice(kPeerOffset, kSecondPosition);
    const auto second_ids = make_slice(kPeerOffset + kSecondPosition, kRows);
    require(first_prefix != second_prefix && first_ids != second_ids,
            "gdnstageoracle requires distinct context fixtures");

    auto first = target.create_context(true);
    auto second = target.create_context(true);
    require(first->try_enable_oscar_from_environment() &&
                second->try_enable_oscar_from_environment(),
            "gdnstageoracle OSCAR attachment failed");
    ingest_prefix(*first, first_prefix, CommitSink{});
    ingest_prefix(*second, second_prefix, CommitSink{});
    require(first->position() == kFirstPosition &&
                second->position() == kSecondPosition,
            "gdnstageoracle divergent prefix positions are wrong");
    first->prepare_transaction();
    second->prepare_transaction();
    const auto first_base = gdn_stage_oracle_state(*first);
    const auto second_base = gdn_stage_oracle_state(*second);

    const auto rollback_and_require_base = [&](const std::string& label) {
        require(first->transaction_active() && second->transaction_active(),
                label + " lost active rollback snapshots");
        first->rollback_transaction();
        second->rollback_transaction();
        require(!first->transaction_active() && !second->transaction_active(),
                label + " rollback remained active");
        require_gdn_stage_oracle_state(gdn_stage_oracle_state(*first), first_base,
                                       label + " first rollback");
        require_gdn_stage_oracle_state(gdn_stage_oracle_state(*second), second_base,
                                       label + " second rollback");
    };

    first->begin_transaction();
    second->begin_transaction();
    const auto first_reference = first->gdn_layer0_serial_for_test(first_ids);
    const auto second_reference = second->gdn_layer0_serial_for_test(second_ids);
    require_gdn_stage_oracle_commit_rejected(*first,
                                              "gdnstageoracle serial first");
    require_gdn_stage_oracle_commit_rejected(*second,
                                              "gdnstageoracle serial second");
    const auto first_reference_state = gdn_stage_oracle_state(*first, false);
    const auto second_reference_state = gdn_stage_oracle_state(*second, false);
    require(first_reference_state.host_position == first_base.host_position &&
                first_reference_state.device_position == first_base.device_position &&
                second_reference_state.host_position == second_base.host_position &&
                second_reference_state.device_position == second_base.device_position,
            "gdnstageoracle serial layer execution changed context publication");
    rollback_and_require_base("gdnstageoracle serial reference");

    std::array<std::vector<std::uint16_t>, 2> first_ab_outputs;
    const auto run_success = [&](bool peer_first, const std::string& label,
                                 bool save_repeat) {
        first->begin_transaction();
        second->begin_transaction();
        ninfer::exl3::Exl3GdnStageOracleTelemetry telemetry{};
        const auto outputs = first->gdn_layer0_pair_staged_serial_for_test(
            *second, first_ids, second_ids, peer_first, nullptr, -1, &telemetry);
        require_gdn_stage_oracle_commit_rejected(*first, label + " first");
        require_gdn_stage_oracle_commit_rejected(*second, label + " second");
        require_gdn_stage_oracle_telemetry(telemetry, label);
        require_gdn_stage_oracle_output(outputs[0], first_reference,
                                        label + " first");
        require_gdn_stage_oracle_output(outputs[1], second_reference,
                                        label + " second");
        require_gdn_stage_oracle_state(gdn_stage_oracle_state(*first, false),
                                       first_reference_state, label + " first");
        require_gdn_stage_oracle_state(gdn_stage_oracle_state(*second, false),
                                       second_reference_state, label + " second");
        if (save_repeat) first_ab_outputs = outputs;
        rollback_and_require_base(label);
        return outputs;
    };

    run_success(false, "gdnstageoracle AB", true);
    run_success(true, "gdnstageoracle BA", false);
    const auto repeated = run_success(false, "gdnstageoracle AB repeat", false);
    require_gdn_stage_oracle_output(repeated[0], first_ab_outputs[0],
                                    "gdnstageoracle repeat first");
    require_gdn_stage_oracle_output(repeated[1], first_ab_outputs[1],
                                    "gdnstageoracle repeat second");

    int failure_cases = 0;
    for (const bool peer_first : {false, true}) {
        for (const int fail_after_lane : {0, 1}) {
            first->begin_transaction();
            second->begin_transaction();
            ninfer::exl3::Exl3GdnStageOracleTelemetry telemetry{};
            std::string failure;
            try {
                (void)first->gdn_layer0_pair_staged_serial_for_test(
                    *second, first_ids, second_ids, peer_first, nullptr,
                    fail_after_lane, &telemetry);
            } catch (const std::exception& error) {
                failure = error.what();
            }
            const std::string label = std::string("gdnstageoracle failure ") +
                (peer_first ? "BA" : "AB") + " lane=" +
                std::to_string(fail_after_lane);
            // Even an unexpected exception must leave the reusable base restored
            // before the harness reports the incorrect failure classification.
            rollback_and_require_base(label);
            require(failure.find("injected staged GDN lane publication failure") !=
                        std::string::npos,
                    "gdnstageoracle expected publication failure was absent: " + failure);
            require(telemetry.staged_pairs == 1 &&
                        telemetry.projection_phases == 2 &&
                        telemetry.serial_b8_projection_calls == 4 &&
                        telemetry.fallback_calls == 0 &&
                        telemetry.published_lanes ==
                            static_cast<std::uint32_t>(fail_after_lane + 1),
                    "gdnstageoracle failure telemetry mismatch");
            run_success(peer_first, label + " reuse", false);
            ++failure_cases;
        }
    }

    std::cout << "GDN_STAGE_ORACLE PASS positions=63,319 rows=8 orders=AB,BA"
              << " repeats=1 failure_cases=" << failure_cases
              << " staged_pairs=1 projection_phases=6 serial_b8_calls=12"
              << " fallback_calls=0 output_bits=" << 2 * kRows * kHidden
              << " recurrent=FP32 physical_conv_slots=4\n";
}
