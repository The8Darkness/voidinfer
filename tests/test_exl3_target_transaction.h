#pragma once

void run_target_transaction(Exl3TextModel& target) {
    const std::string prompt_path = env("NINFER_E5A4_PROMPT_FILE");
    require(!prompt_path.empty(), "targettxn requires NINFER_E5A4_PROMPT_FILE");
    const auto ids = load_ids(prompt_path);
    constexpr int bounded_prefix_rows = 16;
    constexpr int history_rows = 512;
    constexpr int continuation_rows = 8;
    require(ids.size() >= static_cast<std::size_t>(history_rows + continuation_rows),
            "targettxn prompt fixture is too short");
    require(target.max_context() == 1024, "targettxn requires max context 1024");

    auto transaction = target.create_context(true);
    auto reference = target.create_context(true);
    require(transaction->try_enable_oscar_from_environment(),
            "targettxn requires canonical OSCAR");
    require(reference->try_enable_oscar_from_environment(),
            "targettxn reference requires canonical OSCAR");
    transaction->prepare_transaction();
    require(transaction->transaction_prepared(), "targettxn setup did not prepare");
    constexpr std::size_t expected_checkpoint_bytes =
        48ULL * (3145728ULL + 81920ULL) + 20ULL * 1024 * 1024 +
        248320ULL * 2 + 5ULL * 16 * 5120 * 2 + 16ULL * 5120 * 2 + sizeof(int);
    require(transaction->transaction_bytes() == expected_checkpoint_bytes,
            "targettxn checkpoint byte accounting mismatch");

    const std::vector<std::int64_t> prefix(ids.begin(), ids.begin() + bounded_prefix_rows);
    transaction->prefill(prefix);
    reference->prefill(prefix);
    cuda_check(cudaDeviceSynchronize(), "targettxn prefill sync");

    struct Observable {
        int position = 0;
        int device_position = 0;
        int tap_rows = 0;
        int embedding_rows = 0;
        int last_rows = 0;
        std::vector<float> logits;
        std::vector<float> embedding;
        std::array<std::vector<float>, 5> taps;
        std::array<std::vector<float>, 64> gdn;
    };
    auto capture = [](Exl3TextContext& context, bool include_gdn) {
        Observable result;
        result.position = context.position();
        result.device_position = context.device_position_host();
        result.tap_rows = context.captured_tap_rows();
        result.embedding_rows = context.captured_embedding_rows();
        result.last_rows = context.last_forward_rows();
        result.logits = context.logits_host();
        result.embedding = context.embedding_host();
        for (std::size_t i = 0; i < kTapLayers.size(); ++i) {
            result.taps[i] = context.hidden_host(kTapLayers[i]);
        }
        if (include_gdn) {
            for (int layer = 0; layer < 64; ++layer) {
                if (layer >= 3 && (layer - 3) % 4 == 0) continue;
                result.gdn[layer] = context.gdn_state_host(layer);
            }
        }
        return result;
    };
    auto require_equal = [](const Observable& actual, const Observable& expected,
                            bool include_gdn, const std::string& label) {
        require(actual.position == expected.position &&
                    actual.device_position == expected.device_position &&
                    actual.tap_rows == expected.tap_rows &&
                    actual.embedding_rows == expected.embedding_rows &&
                    actual.last_rows == expected.last_rows,
                label + " metadata mismatch");
        require(actual.logits == expected.logits, label + " logits mismatch");
        require(actual.embedding == expected.embedding, label + " embedding mismatch");
        require(actual.taps == expected.taps, label + " tap mismatch");
        if (include_gdn) {
            for (int layer = 0; layer < 64; ++layer) {
                if (layer >= 3 && (layer - 3) % 4 == 0) continue;
                require(actual.gdn[layer].size() == expected.gdn[layer].size(),
                        label + " GDN state size mismatch at layer " + std::to_string(layer));
                require(std::memcmp(actual.gdn[layer].data(), expected.gdn[layer].data(),
                                    actual.gdn[layer].size() * sizeof(float)) == 0,
                        label + " GDN state byte mismatch at layer " + std::to_string(layer));
            }
        }
    };

    const Observable bounded_baseline = capture(*transaction, true);
    require_equal(capture(*reference, true), bounded_baseline, true,
                  "targettxn paired bounded prefill");
    require(bounded_baseline.position == bounded_prefix_rows &&
                bounded_baseline.device_position == 0 &&
                bounded_baseline.tap_rows == bounded_prefix_rows &&
                bounded_baseline.embedding_rows == bounded_prefix_rows &&
                bounded_baseline.last_rows == bounded_prefix_rows,
            "targettxn prefill metadata is unexpected");

    // Preserve and restore the maximum bounded tap/embedding payload once.
    transaction->begin_transaction();
    transaction->decode(ids[bounded_prefix_rows]);
    transaction->rollback_transaction();
    require_equal(capture(*transaction, true), bounded_baseline, true,
                  "targettxn bounded 16-row snapshot");
    bool stale_qkv_rejected = false;
    try { transaction->full_attention_qkv_host(3); }
    catch (const std::runtime_error&) { stale_qkv_rejected = true; }
    require(stale_qkv_rejected, "targettxn exposed discarded QKV scratch after rollback");

    // Reach live OSCAR history aging before exercising the main transaction
    // paths. Serial ingestion preserves the canonical eager cache chronology.
    for (int row = bounded_prefix_rows; row < history_rows; ++row) {
        transaction->decode(ids[row]);
        reference->decode(ids[row]);
        if (row == bounded_prefix_rows) {
            const auto qkv = transaction->full_attention_qkv_host(3);
            require(qkv.rows == 1 && !qkv.q_rope.empty() && !qkv.k_rope.empty() &&
                        !qkv.v_projection.empty(),
                    "targettxn fresh forward did not revalidate QKV trace");
        }
    }
    const Observable baseline = capture(*transaction, true);
    require_equal(capture(*reference, true), baseline, true,
                  "targettxn paired history 512");
    require(baseline.position == history_rows && baseline.device_position == history_rows - 1 &&
                baseline.tap_rows == 1 && baseline.embedding_rows == 1 &&
                baseline.last_rows == 1,
            "targettxn history metadata is unexpected");

    for (int discarded : {1, 4, 8}) {
        transaction->begin_transaction();
        require(transaction->transaction_active(), "targettxn begin did not activate");
        if (discarded == 1) {
            bool nested_rejected = false;
            try { transaction->begin_transaction(); }
            catch (const std::runtime_error&) { nested_rejected = true; }
            require(nested_rejected, "targettxn allowed a nested begin");
            bool graph_rejected = false;
            try { transaction->oscar_set_graph_class(16); }
            catch (const std::runtime_error&) { graph_rejected = true; }
            require(graph_rejected, "targettxn allowed graph mode while active");
            bool reattach_rejected = false;
            try { transaction->try_enable_oscar_from_environment(); }
            catch (const std::runtime_error&) { reattach_rejected = true; }
            require(reattach_rejected, "targettxn allowed OSCAR replacement while active");
        }
        for (int row = 0; row < discarded; ++row) {
            transaction->decode(ids[history_rows + row]);
        }
        transaction->rollback_transaction();
        require(!transaction->transaction_active(), "targettxn rollback stayed active");
        require_equal(capture(*transaction, true), baseline, true,
                      "targettxn discarded " + std::to_string(discarded));
    }

    // Several accepted eager continuations must match the untouched reference.
    for (int row = 0; row < 3; ++row) {
        const auto token = ids[history_rows + row];
        transaction->decode(token);
        reference->decode(token);
        require(transaction->logits_host() == reference->logits_host(),
                "targettxn continuation logits mismatch at row " + std::to_string(row));
        require(transaction->hidden_host(61) == reference->hidden_host(61),
                "targettxn continuation tap mismatch at row " + std::to_string(row));
    }
    require_equal(capture(*transaction, true), capture(*reference, true), true,
                  "targettxn continuation state");

    // Commit keeps the executed state and closes the transaction.
    transaction->begin_transaction();
    for (int row = 3; row < 5; ++row) transaction->decode(ids[history_rows + row]);
    transaction->commit_transaction();
    require(!transaction->transaction_active(), "targettxn commit stayed active");
    for (int row = 3; row < 5; ++row) reference->decode(ids[history_rows + row]);
    require_equal(capture(*transaction, true), capture(*reference, true), true,
                  "targettxn committed state");
    transaction->decode(ids[history_rows + 5]);
    reference->decode(ids[history_rows + 5]);
    require_equal(capture(*transaction, true), capture(*reference, true), true,
                  "targettxn post-commit continuation");

    // Reset invalidates an active image; the prepared storage can then create a
    // fresh transaction after a new completed forward.
    transaction->begin_transaction();
    transaction->reset();
    require(!transaction->transaction_active(), "targettxn reset stayed active");
    bool stale_rollback_rejected = false;
    try { transaction->rollback_transaction(); }
    catch (const std::runtime_error&) { stale_rollback_rejected = true; }
    require(stale_rollback_rejected, "targettxn reset did not invalidate rollback");
    transaction->prefill(prefix);
    transaction->begin_transaction();
    transaction->decode(ids[bounded_prefix_rows]);
    transaction->rollback_transaction();
    require_equal(capture(*transaction, true), bounded_baseline, true,
                  "targettxn recreated snapshot");

    std::uint64_t oscar_full = 0, ordinary_full = 0;
    transaction->oscar_routing_counts(oscar_full, ordinary_full);
    require(oscar_full > 0 && ordinary_full == 0 &&
                transaction->oscar_telemetry().gdn_oscar_dispatches == 0,
            "targettxn routing invariant failed");
    std::cout << "TARGETTXN PASS bytes=" << transaction->transaction_bytes()
              << " history=512 discarded={1,4,8} continuations=6 oscar_full=" << oscar_full
              << " ordinary_full=" << ordinary_full << " gdn_oscar=0\n";
}
