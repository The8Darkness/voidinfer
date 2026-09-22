#pragma once

struct TransactionRoundResult {
    std::vector<std::int64_t> emitted;
    std::int64_t pending = -1;
    int accepted = 0;
    int rejection_index = -1;
    int attempted_rows = 0;
    int replay_rows = 0;
    int retained_rows = 0;
    int state_reconstruction_rows = 0;
};

enum class TransactionRepairMode {
    rollback_replay,
    retain_prefix,
};

void validate_transaction_logits(const std::vector<std::uint16_t>& logits) {
    require(!logits.empty() && logits.size() % kVocab == 0,
            "transaction verifier logits have an invalid shape");
    for (std::size_t index = 0; index < logits.size(); ++index) {
        if (!std::isfinite(half_to_float(logits[index]))) {
            throw std::runtime_error(
                "transaction verifier nonfinite logit row=" +
                std::to_string(index / kVocab) + " token=" +
                std::to_string(index % kVocab));
        }
    }
}

std::int64_t transaction_row_argmax(const std::vector<std::uint16_t>& logits,
                                    int row) {
    const std::size_t base = static_cast<std::size_t>(row) * kVocab;
    require(base + kVocab <= logits.size(), "transaction verifier logits row missing");
    int best = 0;
    float best_value = half_to_float(logits[base]);
    for (int token = 1; token < kVocab; ++token) {
        const float value = half_to_float(logits[base + static_cast<std::size_t>(token)]);
        if (value > best_value) { best = token; best_value = value; }
    }
    return best;
}

// Test/harness-owned transaction verifier. Setup is caller-owned and must be
// completed before timed rounds. Target, staging and ring operations are ordered
// on one stream. No speculative tap reaches the ring before the decision.
// rollback_replay preserves the established reference repair; retain_prefix
// reconstructs target state without executing retained rows through the model.
// A failure after ring publication is fail-closed but is not jointly atomic:
// the target-only transaction cannot roll back the draft ring.
TransactionRoundResult verify_pending_round_transactional(
    Exl3TextContext& target,
    Exl3Dflash2DraftModel& draft,
    TapStage& staging,
    std::int64_t pending_anchor,
    const std::vector<std::int64_t>& proposals,
    int abs_pos,
    cudaStream_t stream = nullptr,
    int injected_nonfinite_index = -1,
    std::uint16_t injected_nonfinite_bits = 0x7e00u,
    TransactionRepairMode repair_mode = TransactionRepairMode::rollback_replay,
    int injected_retain_failure_layer = -1,
    HandoffProfile* handoff_profile = nullptr,
    int profile_round = -1) {
    require(pending_anchor >= 0 && pending_anchor < kVocab && !proposals.empty(),
            "transaction verifier requires a valid anchor and proposals");
    require(proposals.size() + 1 <= 8,
            "transaction verifier exceeds bounded continuation capacity");
    for (const auto token : proposals)
        require(token >= 0 && token < kVocab,
                "transaction verifier proposal outside vocabulary");
    require(target.transaction_prepared() && !target.transaction_active() &&
                target.continuation_capacity() >= static_cast<int>(proposals.size() + 1),
            "transaction verifier setup is incomplete");
    require(target.position() == abs_pos &&
                draft.ring_base_abs() + draft.ring_count() == abs_pos,
            "transaction verifier target/ring tail mismatch");
    require(target.oscar_enabled() && !target.graph_active(),
            "transaction verifier requires eager canonical OSCAR");
    require(injected_retain_failure_layer >= -1 &&
                injected_retain_failure_layer < 64,
            "transaction verifier retain failure layer is outside 0..63");
    require(injected_retain_failure_layer < 0 ||
                repair_mode == TransactionRepairMode::retain_prefix,
            "transaction verifier retain failure injection requires retain mode");
    require(staging.bulk_ptrs.size() == kTapCount &&
                staging.row_ptrs.size() == kTapCount,
            "transaction verifier staging count mismatch");
    for (int tap = 0; tap < kTapCount; ++tap) {
        require(staging.bulk_ptrs[static_cast<std::size_t>(tap)] != nullptr &&
                    staging.row_ptrs[static_cast<std::size_t>(tap)] != nullptr,
                "transaction verifier staging contains a null pointer");
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status),
               "query transaction verifier stream capture");
    require(capture_status == cudaStreamCaptureStatusNone,
            "transaction verifier requires an eager stream");

    std::vector<std::int64_t> block;
    block.reserve(proposals.size() + 1);
    block.push_back(pending_anchor);
    block.insert(block.end(), proposals.begin(), proposals.end());

    TransactionRoundResult result;
    result.attempted_rows = static_cast<int>(block.size());
    bool publication_started = false;
    try {
        HandoffProfile::Handle target_verify;
        if (handoff_profile)
            target_verify = handoff_profile->begin(
                profile_round, "verify", "target_verify_total", stream);
        HandoffProfile::Handle target_begin;
        if (handoff_profile)
            target_begin = handoff_profile->begin(
                profile_round, "verify", "target_begin", stream);
        target.set_target_projection_timing_phase(
            Exl3TargetProjectionPhase::attempt);
        target.begin_transaction(stream);
        if (handoff_profile) handoff_profile->end(target_begin, stream);
        HandoffProfile::Handle target_forward;
        if (handoff_profile)
            target_forward = handoff_profile->begin(
                profile_round, "verify", "target_forward", stream);
        if (target.continuation_graph_active())
            target.continue_rows_graph(block, stream);
        else
            target.continue_rows(block, stream);
        if (handoff_profile) handoff_profile->end(target_forward, stream);
        HandoffProfile::Handle download;
        if (handoff_profile)
            download = handoff_profile->begin(
                profile_round, "verify", "logit_download_validation", stream);
        auto logits = target.continuation_logits_bits_host(stream);
        require(logits.size() == block.size() * static_cast<std::size_t>(kVocab),
                "transaction verifier continuation logits row count mismatch");
        if (injected_nonfinite_index >= 0) {
            require(static_cast<std::size_t>(injected_nonfinite_index) < logits.size(),
                    "transaction verifier injected logit index is outside output");
            logits[static_cast<std::size_t>(injected_nonfinite_index)] = injected_nonfinite_bits;
        }
        validate_transaction_logits(logits);
        if (handoff_profile) handoff_profile->end(download, stream);
        HandoffProfile::Handle decision;
        if (handoff_profile)
            decision = handoff_profile->begin(
                profile_round, "verify", "decision", stream);
        int accepted = 0;
        for (; accepted < static_cast<int>(proposals.size()); ++accepted) {
            if (proposals[static_cast<std::size_t>(accepted)] !=
                transaction_row_argmax(logits, accepted)) break;
        }
        result.accepted = accepted;
        result.rejection_index =
            accepted == static_cast<int>(proposals.size()) ? -1 : accepted;
        result.retained_rows = accepted + 1;
        result.pending = transaction_row_argmax(logits, accepted);
        result.emitted.assign(proposals.begin(), proposals.begin() + accepted);
        result.emitted.push_back(result.pending);
        if (handoff_profile) handoff_profile->end(decision, stream);
        if (handoff_profile) handoff_profile->end(target_verify, stream);

        HandoffProfile::Handle accept_commit;
        if (handoff_profile)
            accept_commit = handoff_profile->begin(
                profile_round, "accept_commit", "accept_commit_total", stream);
        HandoffProfile::Handle repair;
        if (handoff_profile)
            repair = handoff_profile->begin(
                profile_round, "verify", "retained_state_repair", stream);

        if (repair_mode == TransactionRepairMode::retain_prefix) {
            if (injected_retain_failure_layer >= 0) {
                target.retain_transaction_prefix_for_test(
                    result.retained_rows, injected_retain_failure_layer, stream);
            } else {
                target.retain_transaction_prefix(result.retained_rows, stream);
            }
            result.state_reconstruction_rows =
                result.rejection_index < 0 ? 0 : result.retained_rows;
            if (handoff_profile) handoff_profile->end(repair, stream);
            HandoffProfile::Handle tap_sync;
            if (handoff_profile)
                tap_sync = handoff_profile->begin(
                    profile_round, "verify", "tap_stage_precommit_sync", stream);
            for (int tap = 0; tap < kTapCount; ++tap) {
                target.copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(tap)], 0,
                    staging.bulk_ptrs[static_cast<std::size_t>(tap)],
                    result.retained_rows, stream);
            }
            // Keep the original transaction checkpoint active until retained
            // state and every staged tap are known complete.  A failure before
            // this boundary is recoverable by the catch block below.
            cuda_check(cudaStreamSynchronize(stream),
                       "synchronize retained-prefix transaction verifier");
            if (handoff_profile) handoff_profile->end(tap_sync, stream);
            HandoffProfile::Handle target_commit;
            if (handoff_profile)
                target_commit = handoff_profile->begin(
                    profile_round, "verify", "target_commit", stream);
            target.commit_transaction();
            if (handoff_profile) handoff_profile->end(target_commit, stream);
        } else if (result.rejection_index < 0) {
            if (handoff_profile) handoff_profile->end(repair, stream);
            HandoffProfile::Handle tap_sync;
            if (handoff_profile)
                tap_sync = handoff_profile->begin(
                    profile_round, "verify", "tap_stage_precommit_sync", stream);
            for (int tap = 0; tap < kTapCount; ++tap) {
                target.copy_tap_rows_to_device(kTapLayers[static_cast<std::size_t>(tap)], 0,
                    staging.bulk_ptrs[static_cast<std::size_t>(tap)],
                    result.attempted_rows, stream);
            }
            if (handoff_profile) handoff_profile->end(tap_sync, stream);
            HandoffProfile::Handle target_commit;
            if (handoff_profile)
                target_commit = handoff_profile->begin(
                    profile_round, "verify", "target_commit", stream);
            target.commit_transaction();
            if (handoff_profile) handoff_profile->end(target_commit, stream);
        } else {
            target.rollback_transaction(stream);
            result.replay_rows = result.retained_rows;
            target.set_target_projection_timing_phase(
                Exl3TargetProjectionPhase::replay);
            target.begin_transaction(stream);
            for (int row = 0; row < result.retained_rows; ++row) {
                target.decode(block[static_cast<std::size_t>(row)], stream);
                for (int tap = 0; tap < kTapCount; ++tap) {
                    target.copy_tap_row_to_device(
                        kTapLayers[static_cast<std::size_t>(tap)],
                        staging.bulk_ptrs[static_cast<std::size_t>(tap)] +
                            static_cast<std::size_t>(row) * kHidden,
                        stream);
                }
            }
            if (handoff_profile) handoff_profile->end(repair, stream);
            HandoffProfile::Handle target_commit;
            if (handoff_profile)
                target_commit = handoff_profile->begin(
                    profile_round, "verify", "target_commit", stream);
            target.commit_transaction();
            if (handoff_profile) handoff_profile->end(target_commit, stream);
        }

        // Set this before the call because ring publication may enqueue or
        // mutate a prefix and then report failure. There is no joint rollback.
        publication_started = true;
        HandoffProfile::Handle ring_commit;
        if (handoff_profile)
            ring_commit = handoff_profile->begin(
                profile_round, "verify", "draft_ring_commit", stream);
        draft.commit_target_block(
            const_cast<const std::uint16_t**>(staging.bulk_ptrs.data()),
            result.retained_rows, abs_pos, stream);
        if (handoff_profile) handoff_profile->end(ring_commit, stream);
        HandoffProfile::Handle final_sync;
        if (handoff_profile)
            final_sync = handoff_profile->begin(
                profile_round, "verify", "final_sync", stream);
        cuda_check(cudaStreamSynchronize(stream),
                   "synchronize transaction verifier round");
        if (handoff_profile) handoff_profile->end(final_sync, stream);
        if (handoff_profile) handoff_profile->end(accept_commit, stream);
        require(target.position() == abs_pos + result.retained_rows &&
                    draft.ring_base_abs() + draft.ring_count() ==
                        abs_pos + result.retained_rows,
                "transaction verifier exit tail mismatch");
        return result;
    } catch (...) {
        const std::exception_ptr failure = std::current_exception();
        if (!publication_started && target.transaction_active()) {
            target.rollback_transaction(stream);
            cuda_check(cudaStreamSynchronize(stream),
                       "synchronize transaction verifier failure rollback");
        }
        std::rethrow_exception(failure);
    }
}

TapStage make_transaction_round_stage() {
    TapStage stage;
    for (int tap = 0; tap < kTapCount; ++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(16) * kHidden * sizeof(std::uint16_t)));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t)));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    return stage;
}

std::vector<std::uint16_t> transaction_round_device_bits(
    const std::uint16_t* source, std::size_t count, const char* label) {
    std::vector<std::uint16_t> result(count);
    cuda_check(cudaMemcpy(result.data(), source, count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), label);
    return result;
}

std::vector<std::uint16_t> transaction_round_last_tap_bits(
    Exl3TextContext& context, int layer) {
    DeviceBuffer staging(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    context.copy_tap_row_to_device(layer,
        static_cast<std::uint16_t*>(staging.get()));
    return transaction_round_device_bits(
        static_cast<const std::uint16_t*>(staging.get()), kHidden,
        "transactionqual download final tap");
}

std::vector<std::uint16_t> transaction_round_tap_rows_bits(
    Exl3TextContext& context,int layer,int rows,cudaStream_t stream) {
    DeviceBuffer staging(static_cast<std::size_t>(rows)*kHidden*sizeof(std::uint16_t));
    context.copy_tap_rows_to_device(layer,0,
        static_cast<std::uint16_t*>(staging.get()),rows,stream);
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows)*kHidden);
    cuda_check(cudaMemcpyAsync(result.data(),staging.get(),
        result.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost,stream),
        "transactionqual download tap rows");
    cuda_check(cudaStreamSynchronize(stream),
        "transactionqual synchronize tap rows");
    return result;
}

void require_transaction_round_state_equal(Exl3TextContext& actual,
                                           Exl3TextContext& expected,
                                           const std::string& label) {
    require(actual.position() == expected.position() &&
                actual.device_position_host() == expected.device_position_host(),
            label + " position mismatch");
    require(transaction_round_device_bits(actual.logits_device(), kVocab,
                "transactionqual download actual logits") ==
            transaction_round_device_bits(expected.logits_device(), kVocab,
                "transactionqual download expected logits"),
            label + " logits mismatch");
    for (const int layer : kTapLayers) {
        require(transaction_round_last_tap_bits(actual, layer) ==
                    transaction_round_last_tap_bits(expected, layer),
                label + " final tap mismatch at layer " + std::to_string(layer));
    }
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        const auto a = actual.gdn_state_host(layer);
        const auto b = expected.gdn_state_host(layer);
        require(a.size() == b.size() &&
                    std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                label + " GDN state mismatch at layer " + std::to_string(layer));
        require(actual.gdn_physical_conv_host(layer) ==
                    expected.gdn_physical_conv_host(layer),
                label + " GDN physical convolution mismatch at layer " +
                    std::to_string(layer));
    }
    require(actual.oscar_live_state_host_for_test() ==
                expected.oscar_live_state_host_for_test(),
            label + " OSCAR live state mismatch");
}

void run_transaction_round_qualification(Exl3TextModel& target,
                                         Exl3Dflash2DraftModel& draft,
                                         const std::vector<std::int64_t>& source_ids) {
    require(target.max_context() >= 4096, "transactionqual requires max context at least 4096");
    require(env("NINFER_OSCAR_EXL3") == "1", "transactionqual requires canonical OSCAR");
    require(source_ids.size() >= 32, "transactionqual prompt fixture is too short");
    std::vector<std::uint16_t> injected_logits(
        static_cast<std::size_t>(2) * kVocab, 0);
    validate_transaction_logits(injected_logits);
    auto require_nonfinite_rejected = [&](std::size_t index, std::uint16_t bits,
                                          const char* label) {
        injected_logits[index] = bits;
        bool rejected = false;
        try {
            validate_transaction_logits(injected_logits);
        } catch (const std::exception&) {
            rejected = true;
        }
        injected_logits[index] = 0;
        require(rejected, std::string("transactionqual accepted ") + label);
    };
    require_nonfinite_rejected(0, 0x7e00u, "first NaN logit");
    require_nonfinite_rejected(injected_logits.size() / 2, 0x7c00u,
                               "middle positive-infinity logit");
    require_nonfinite_rejected(injected_logits.size() - 1, 0xfc00u,
                               "last negative-infinity logit");
    auto extended_prefix = [&](int rows) {
        std::vector<std::int64_t> prefix(static_cast<std::size_t>(rows));
        for (int row = 0; row < rows; ++row)
            prefix[static_cast<std::size_t>(row)] =
                source_ids[static_cast<std::size_t>(row) % source_ids.size()];
        return prefix;
    };

    int cases = 0;
    bool graph_failure_injected = false;
    const bool graph_flag_enabled =
        env("NINFER_E5A4_CONTINUATION_GRAPH_B8") == "1";
    const bool cohort_flag_enabled =
        env("NINFER_OSCAR_CONTINUATION_COHORT_B8") == "1";
    const bool graph_cohort_enabled = graph_flag_enabled && cohort_flag_enabled;
    const bool gdn_qkvz_flag_enabled =
        env("NINFER_EXL3_CONTINUATION_GRAPH_GDN_QKVZ_CONCURRENT") == "1";
    const bool graph_gdn_qkvz_enabled =
        graph_flag_enabled && gdn_qkvz_flag_enabled;
    auto run_case = [&](const std::vector<std::int64_t>& prefix, int block_len,
                        int forced_rejection, const std::string& label) {
        const int proposal_count = block_len - 1;
        require(forced_rejection >= -1 && forced_rejection < proposal_count,
                label + " invalid forced rejection");
        auto test = target.create_context(true);
        auto reference = target.create_context(true);
        require(test->try_enable_oscar_from_environment() &&
                    reference->try_enable_oscar_from_environment(),
                label + " OSCAR enable failed");
        test->prepare_transaction();
        test->prepare_continuation(block_len);
        reference->prepare_transaction();
        reference->prepare_continuation(block_len);
        const auto eager_telemetry_before = test->oscar_telemetry();
        draft.reset();
        TapStage stage = make_transaction_round_stage();
        TapStage replay_stage = make_transaction_round_stage();
        ingest_prefix(*test, prefix, CommitSink{&draft, &stage});
        ingest_prefix(*reference, prefix, CommitSink{});
        struct GraphStream {
            cudaStream_t value = nullptr;
            ~GraphStream() { if (value) cudaStreamDestroy(value); }
        } graph_stream;
        const bool graph_case = block_len == 8 && graph_flag_enabled;
        if (graph_case) {
            cuda_check(cudaStreamCreateWithFlags(&graph_stream.value,cudaStreamNonBlocking),
                       "transactionqual create continuation graph stream");
            cuda_check(cudaDeviceSynchronize(),
                       "transactionqual pre-capture synchronization");
            require(test->capture_continuation_graph(graph_stream.value),
                    label + " continuation graph capture failed: " +
                        test->continuation_graph_status());
            const std::string expected_gdn_qkvz = graph_gdn_qkvz_enabled
                ? "sibling GDN QKV/Z projections; "
                  "gdn_qkvz_private_workspace_bytes=1064960; gdn_qkvz_layers=48"
                : "sequential GDN QKV/Z";
            require(test->continuation_graph_status().find(expected_gdn_qkvz) !=
                        std::string::npos,
                    label + " continuation graph GDN QKV/Z route/resource marker mismatch");
            std::cout << "TRANSACTIONQUAL_GDN_QKVZ_ROUTE PASS label=" << label
                      << " status=" << test->continuation_graph_status() << std::endl;
            cuda_check(cudaStreamSynchronize(graph_stream.value),
                       "transactionqual continuation graph capture sync");
        }
        cudaStream_t test_stream = graph_stream.value;
        std::int64_t pending = sample_target(*test);
        require(pending == sample_target(*reference), label + " initial seed mismatch");
        if (graph_case) {
            std::vector<std::int64_t> exact_block(8);
            exact_block[0] = pending;
            for (int row = 1; row < 8; ++row)
                exact_block[static_cast<std::size_t>(row)] =
                    source_ids[static_cast<std::size_t>(row) % source_ids.size()];
            const int exact_base = test->position();
            const auto telemetry_before = test->oscar_telemetry();
            std::uint64_t route_before=0,ordinary_before=0;
            test->oscar_routing_counts(route_before,ordinary_before);
            test->begin_transaction(test_stream);
            reference->begin_transaction();
            test->continue_rows_graph(exact_block,test_stream);
            reference->continue_rows(exact_block);
            require(test->position() == exact_base + 8 &&
                        test->device_position_host(test_stream) == exact_base + 7 &&
                        reference->position() == exact_base + 8 &&
                        reference->device_position_host() == exact_base + 7,
                    label + " graph T0 position mismatch");
            require(test->continuation_logits_bits_host(test_stream) ==
                        reference->continuation_logits_bits_host(),
                    label + " graph T0 all-row logits mismatch");
            for (const int layer : kTapLayers) {
                require(transaction_round_tap_rows_bits(*test,layer,8,test_stream) ==
                            transaction_round_tap_rows_bits(*reference,layer,8,nullptr),
                        label + " graph T0 all-row taps mismatch at layer " +
                            std::to_string(layer));
            }
            require(test->embedding_bits_host_for_test(test_stream) ==
                        reference->embedding_bits_host_for_test(),
                    label + " graph T0 all-row embedding mismatch");
            const auto telemetry_after = test->oscar_telemetry();
            for (int layer=3;layer<64;layer+=4)
                require(telemetry_after.oscar_dispatches[layer] ==
                            telemetry_before.oscar_dispatches[layer]+8,
                        label + " graph T0 OSCAR dispatch telemetry mismatch");
            const auto recent_begin=[](int context) {
                return context<=64?context:std::max(64,context-256);
            };
            require(telemetry_after.append_calls ==
                        telemetry_before.append_calls+16ULL*8ULL &&
                    telemetry_after.aging_events == telemetry_before.aging_events+
                        16ULL*static_cast<std::uint64_t>(
                            recent_begin(exact_base+8)-recent_begin(exact_base)) &&
                    telemetry_after.last_split_class ==
                        (exact_base+8<=512?16:(exact_base+8<=8192?32:64)),
                    label + " graph T0 append/aging/split telemetry mismatch");
            std::uint64_t route_after=0,ordinary_after=0;
            test->oscar_routing_counts(route_after,ordinary_after);
            require(route_after == route_before+16ULL*8ULL &&
                        ordinary_after == ordinary_before,
                    label + " graph T0 full-attention route telemetry mismatch");
            for (int layer = 3; layer < 64; layer += 4) {
                const auto actual_qkv = test->full_attention_qkv_host(layer,test_stream);
                const auto expected_qkv = reference->full_attention_qkv_host(layer);
                require(actual_qkv.rows == 8 && expected_qkv.rows == 8 &&
                            actual_qkv.q_rope == expected_qkv.q_rope &&
                            actual_qkv.k_rope == expected_qkv.k_rope &&
                            actual_qkv.v_projection == expected_qkv.v_projection,
                        label + " graph T0 QKV mismatch at layer " +
                            std::to_string(layer));
            }
            require_transaction_round_state_equal(
                *test,*reference,label + " graph T0 live state");
            test->rollback_transaction(test_stream);
            reference->rollback_transaction();
            cuda_check(cudaStreamSynchronize(test_stream),
                       "transactionqual graph T0 rollback sync");
            require_transaction_round_state_equal(
                *test,*reference,label + " graph T0 rollback");
        }
        std::vector<std::int64_t> emitted{pending};
        std::vector<std::int64_t> processed;
        long long attempted_total = 0, replay_total = 0, retained_total = 0;

        for (int round_index = 0; round_index < 2; ++round_index) {
            reference->begin_transaction();
            reference->decode(pending);
            std::vector<std::int64_t> future;
            future.reserve(static_cast<std::size_t>(proposal_count + 1));
            for (int index = 0; index <= proposal_count; ++index) {
                future.push_back(sample_target(*reference));
                if (index < proposal_count) reference->decode(future.back());
            }
            reference->rollback_transaction();
            std::vector<std::int64_t> proposals(
                future.begin(), future.begin() + proposal_count);
            if (forced_rejection >= 0)
                proposals[static_cast<std::size_t>(forced_rejection)] =
                    (proposals[static_cast<std::size_t>(forced_rejection)] + 1) % kVocab;

            const int abs_pos = test->position();
            if (round_index == 0 &&
                (cases == 0 || (graph_case && !graph_failure_injected))) {
                const auto ring_before_failure = draft.ring_digest();
                const int ring_count_before_failure = draft.ring_count();
                const long long ring_base_before_failure = draft.ring_base_abs();
                bool rejected = false;
                try {
                    (void)verify_pending_round_transactional(
                        *test, draft, stage, pending, proposals, abs_pos,
                        test_stream, 0, 0x7e00u,
                        graph_case ? TransactionRepairMode::retain_prefix :
                                     TransactionRepairMode::rollback_replay);
                } catch (const std::exception& error) {
                    rejected = std::string(error.what()).find(
                        "nonfinite logit row=0 token=0") != std::string::npos;
                }
                require(rejected && !test->transaction_active(),
                        label + " injected post-forward failure did not rollback");
                require_transaction_round_state_equal(
                    *test, *reference, label + " injected failure rollback");
                require(draft.ring_digest() == ring_before_failure &&
                            draft.ring_count() == ring_count_before_failure &&
                            draft.ring_base_abs() == ring_base_before_failure,
                        label + " injected failure changed ring state");
                if (graph_case) graph_failure_injected = true;
            }
            const TransactionRoundResult result = verify_pending_round_transactional(
                *test, draft, stage, pending, proposals, abs_pos,test_stream,-1,0x7e00u,
                graph_case ? TransactionRepairMode::retain_prefix :
                             TransactionRepairMode::rollback_replay);
            const int accepted = forced_rejection < 0 ? proposal_count : forced_rejection;
            require(result.accepted == accepted &&
                        result.rejection_index == forced_rejection &&
                        result.attempted_rows == block_len &&
                        result.retained_rows == accepted + 1 &&
                        result.replay_rows ==
                            (graph_case || forced_rejection < 0 ? 0 : accepted + 1) &&
                        result.state_reconstruction_rows ==
                            (graph_case && forced_rejection >= 0 ? accepted + 1 : 0),
                    label + " transactional outcome/accounting mismatch");

            reference->decode(pending);
            processed.push_back(pending);
            for (int index = 0; index < accepted; ++index) {
                reference->decode(future[static_cast<std::size_t>(index)]);
                processed.push_back(future[static_cast<std::size_t>(index)]);
            }
            std::vector<std::int64_t> expected_emitted(
                future.begin(), future.begin() + accepted + 1);
            require(result.emitted == expected_emitted &&
                        result.pending == future[static_cast<std::size_t>(accepted)],
                    label + " emitted/pending mismatch");
            emitted.insert(emitted.end(), result.emitted.begin(), result.emitted.end());
            pending = result.pending;
            attempted_total += result.attempted_rows;
            replay_total += result.replay_rows;
            retained_total += result.retained_rows;
            require(processed.size() + 1 == emitted.size() &&
                        std::equal(processed.begin(), processed.end(), emitted.begin()),
                    label + " round pending ledger mismatch");
            require_transaction_round_state_equal(*test, *reference,
                label + " round " + std::to_string(round_index));
            const long long end = static_cast<long long>(prefix.size() + processed.size());
            const int count = static_cast<int>(std::min<long long>(
                end, Exl3Dflash2DraftModel::ring_keep()));
            require(draft.ring_count() == count && draft.ring_base_abs() == end - count,
                    label + " ring boundary mismatch");
        }
        require(processed.size() + 1 == emitted.size() &&
                    std::equal(processed.begin(), processed.end(), emitted.begin()),
                label + " pending ledger mismatch");
        // This digest covers only prefix ingest plus the two transactional
        // rounds. The target-only continuations below intentionally do not
        // publish taps to the draft ring.
        const auto live_digest = draft.ring_digest();
        const long long live_end = static_cast<long long>(prefix.size() + processed.size());
        const int live_count = static_cast<int>(std::min<long long>(
            live_end, Exl3Dflash2DraftModel::ring_keep()));
        std::uint64_t oscar_full = 0, ordinary_full = 0;
        test->oscar_routing_counts(oscar_full, ordinary_full);
        const auto route_telemetry = test->oscar_telemetry();
        require(oscar_full > 0 && ordinary_full == 0 &&
                    route_telemetry.gdn_oscar_dispatches == 0,
                label + " escaped canonical OSCAR routing");
        if (!graph_flag_enabled && block_len > 1) {
            require(route_telemetry.continuation_cohort_eager_attempts >
                        eager_telemetry_before.continuation_cohort_eager_attempts,
                    label + " did not exercise eager OSCAR cohort admission");
            if (cohort_flag_enabled) {
                require(route_telemetry.continuation_cohort_eager_dispatches >
                            eager_telemetry_before.continuation_cohort_eager_dispatches &&
                            route_telemetry.continuation_cohort_eager_latch_misses ==
                                eager_telemetry_before.continuation_cohort_eager_latch_misses,
                        label + " did not dispatch the enabled eager OSCAR cohort");
            } else {
                require(route_telemetry.continuation_cohort_eager_dispatches ==
                            eager_telemetry_before.continuation_cohort_eager_dispatches &&
                            route_telemetry.continuation_cohort_eager_latch_misses >
                                eager_telemetry_before.continuation_cohort_eager_latch_misses,
                        label + " default-off eager OSCAR cohort was not inert");
            }
        }

        for (int continuation = 0; continuation < 4; ++continuation) {
            test->decode(pending);
            reference->decode(pending);
            require_transaction_round_state_equal(*test, *reference,
                label + " continuation " + std::to_string(continuation));
            pending = sample_target(*reference);
            require(pending == sample_target(*test), label + " continuation greedy mismatch");
        }
        test.reset();
        reference.reset();

        draft.reset();
        auto replay = target.create_context(true);
        require(replay->try_enable_oscar_from_environment(), label + " replay OSCAR failed");
        ingest_prefix(*replay, prefix, CommitSink{&draft, &replay_stage});
        for (const auto token : processed) {
            const int position = replay->position();
            replay->decode(token);
            commit_latest_row(*replay, draft, replay_stage, position);
        }
        require(draft.ring_digest() == live_digest && draft.ring_count() == live_count &&
                    draft.ring_base_abs() == live_end - live_count,
                label + " independently rebuilt ring mismatch");
        ++cases;
        std::cout << "TRANSACTIONQUAL_CASE PASS label=" << label
                  << " ctx=" << prefix.size() << " B=" << block_len
                  << " forced_rejection=" << forced_rejection
                  << " rounds=2 continuations=4 attempted=" << attempted_total
                  << " replay=" << replay_total << " retained=" << retained_total
                  << " ring_count=" << live_count
                  << " ring_base=" << live_end - live_count << std::endl;
    };

    auto run_graph_boundary_rejection = [&] {
        auto test = target.create_context(true);
        auto reference = target.create_context(true);
        require(test->try_enable_oscar_from_environment() &&
                    reference->try_enable_oscar_from_environment(),
                "ctx505-B8-boundary OSCAR enable failed");
        test->prepare_transaction();
        test->prepare_continuation(8);
        const auto prefix = extended_prefix(505);
        ingest_prefix(*test,prefix,CommitSink{});
        ingest_prefix(*reference,prefix,CommitSink{});
        cuda_check(cudaDeviceSynchronize(),
                   "transactionqual synchronize split-boundary setup");
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),
                   "transactionqual create split-boundary stream");
        const bool captured = test->capture_continuation_graph(stream);
        cuda_check(cudaStreamSynchronize(stream),
                   "transactionqual synchronize split-boundary rejection");
        cuda_check(cudaStreamDestroy(stream),
                   "transactionqual destroy split-boundary stream");
        const std::string expected_reason = graph_flag_enabled ? "crosses" : "disabled";
        require(!captured && !test->continuation_graph_active() &&
                    test->continuation_graph_status().find(expected_reason) !=
                        std::string::npos,
                "ctx505-B8-boundary graph admission did not reject");
        require_transaction_round_state_equal(
            *test,*reference,"ctx505-B8-boundary nonmutation");
        std::cout << "TRANSACTIONQUAL_GRAPH_BOUNDARY PASS ctx=505 B=8"
                  << std::endl;
    };

    const auto prefix32 = extended_prefix(32);
    for (int block_len : {2, 4}) {
        for (int rejection = 0; rejection < block_len - 1; ++rejection)
            run_case(prefix32, block_len, rejection,
                "ctx32-B" + std::to_string(block_len) + "-reject" +
                    std::to_string(rejection));
        run_case(prefix32, block_len, -1,
                 "ctx32-B" + std::to_string(block_len) + "-full");
    }
    for (const int rejection : {0, 3, 6, -1})
        run_case(prefix32, 8, rejection,
            rejection < 0 ? "ctx32-B8-full" :
                "ctx32-B8-reject" + std::to_string(rejection));
    const auto prefix512 = extended_prefix(512);
    for (const int rejection : {0, 6, -1})
        run_case(prefix512, 8, rejection,
            rejection < 0 ? "ctx512-B8-full" :
                "ctx512-B8-reject" + std::to_string(rejection));
    run_graph_boundary_rejection();
    run_case(extended_prefix(4096),8,3,"ctx4096-B8-reject3");
    run_case(extended_prefix(2047), 2, -1, "ctx2047-B2-full-wrap");
    std::cout << "TRANSACTIONQUAL PASS cases=" << cases
              << " protocol=pending_anchor_v1 semantic_continuations=4"
              << " target_only_continuations=4 ring_digest_rounds=2"
              << " nonfinite_patterns=3 post_forward_rollback=1"
              << " gdn_states=48 physical_conv_states=48 oscar_live_state=exact"
              << " graph_post_forward_rollback=" << (graph_failure_injected ? 1 : 0)
              << " oscar_b8_route=" << (graph_cohort_enabled ?
                    "chronological_cohort" :
                    (graph_flag_enabled ? "sequential_graph" :
                        (cohort_flag_enabled ? "chronological_cohort_eager" :
                            "sequential_eager")))
              << " gdn_qkvz_b8_route=" << (graph_gdn_qkvz_enabled ?
                    "sibling_concurrent" :
                    (graph_flag_enabled ? "sequential_graph" : "sequential_eager"))
              << std::endl;
}
