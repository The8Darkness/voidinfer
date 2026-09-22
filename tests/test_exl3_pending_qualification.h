#pragma once

struct PendingObservable {
    int position = 0;
    int device_position = 0;
    std::vector<float> logits;
    std::array<std::vector<float>, 5> taps;
    std::array<std::vector<float>, 64> gdn;
};

PendingObservable capture_pending_observable(Exl3TextContext& context) {
    PendingObservable value;
    value.position = context.position();
    value.device_position = context.device_position_host();
    value.logits = context.logits_host();
    for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap)
        value.taps[tap] = context.hidden_host(kTapLayers[tap]);
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        value.gdn[static_cast<std::size_t>(layer)] = context.gdn_state_host(layer);
    }
    return value;
}

void require_pending_observable_equal(const PendingObservable& actual,
                                      const PendingObservable& expected,
                                      const std::string& label) {
    require(actual.position == expected.position &&
                actual.device_position == expected.device_position,
            label + " position mismatch");
    require(actual.logits == expected.logits, label + " logits mismatch");
    require(actual.taps == expected.taps, label + " taps mismatch");
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        const auto& a = actual.gdn[static_cast<std::size_t>(layer)];
        const auto& b = expected.gdn[static_cast<std::size_t>(layer)];
        require(a.size() == b.size() &&
                    std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                label + " GDN state mismatch at layer " + std::to_string(layer));
    }
}

TapStage make_pending_tap_stage() {
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

void run_pending_qualification(Exl3TextModel& target,
                               Exl3Dflash2DraftModel& draft,
                               const std::vector<std::int64_t>& source_ids) {
    require(target.max_context() >= 4096, "pendingqual requires max context at least 4096");
    require(env("NINFER_OSCAR_EXL3") == "1", "pendingqual requires canonical OSCAR");
    require(source_ids.size() >= 32, "pendingqual prompt fixture is too short");

    auto extended_prefix = [&](int rows) {
        std::vector<std::int64_t> prefix;
        prefix.reserve(static_cast<std::size_t>(rows));
        for (int row = 0; row < rows; ++row)
            prefix.push_back(source_ids[static_cast<std::size_t>(row) % source_ids.size()]);
        return prefix;
    };

    auto run_case = [&](const std::vector<std::int64_t>& prefix, int block_len,
                        int forced_rejection, const std::string& label) {
        const int proposal_count = block_len - 1;
        require(forced_rejection >= -1 && forced_rejection < proposal_count,
                label + " forced outcome outside block");
        auto test = target.create_context(true);
        auto reference = target.create_context(true);
        require(test->try_enable_oscar_from_environment(), label + " test OSCAR enable failed");
        require(reference->try_enable_oscar_from_environment(),
                label + " reference OSCAR enable failed");
        reference->prepare_transaction();
        draft.reset();
        TapStage test_stage = make_pending_tap_stage();
        TapStage replay_stage = make_pending_tap_stage();
        ingest_prefix(*test, prefix, CommitSink{&draft, &test_stage});
        ingest_prefix(*reference, prefix, CommitSink{});

        std::int64_t pending = sample_target(*test);
        require(pending == sample_target(*reference), label + " initial seed mismatch");
        std::vector<std::int64_t> emitted{pending};
        std::vector<std::int64_t> expected_emitted{pending};
        std::vector<std::int64_t> processed;
        require(test->position() == static_cast<int>(prefix.size()) &&
                    reference->position() == static_cast<int>(prefix.size()) &&
                    draft.ring_base_abs() + draft.ring_count() ==
                        static_cast<long long>(prefix.size()),
                label + " seed-only pending state mismatch");

        for (int round_index = 0; round_index < 2; ++round_index) {
            // Derive the exact target continuation under a transaction, then
            // roll it back. This independent fixture path never calls the shared
            // pending verifier.
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
            if (forced_rejection >= 0) {
                auto& rejected = proposals[static_cast<std::size_t>(forced_rejection)];
                rejected = (rejected + 1) % kVocab;
            }
            int abs_pos = test->position();
            const PendingRoundResult result = verify_pending_round(
                pending, proposals,
                [&]() { return sample_target(*test); },
                [&](std::int64_t token) {
                    test->decode(token);
                    commit_latest_row(*test, draft, test_stage, abs_pos);
                    processed.push_back(token);
                    ++abs_pos;
                });
            const int expected_accepted =
                forced_rejection < 0 ? proposal_count : forced_rejection;
            require(result.accepted == expected_accepted &&
                        result.rejection_index == forced_rejection &&
                        result.processed == expected_accepted + 1,
                    label + " helper outcome mismatch in round " +
                        std::to_string(round_index));
            for (int index = 0; index <= expected_accepted; ++index)
                expected_emitted.push_back(future[static_cast<std::size_t>(index)]);
            emitted.insert(emitted.end(), result.emitted.begin(), result.emitted.end());
            require(emitted == expected_emitted && result.pending == expected_emitted.back(),
                    label + " emitted ledger/pending mismatch");
            require(processed.size() + 1 == emitted.size() &&
                        std::equal(processed.begin(), processed.end(), emitted.begin()),
                    label + " processed ledger is not emitted[0:-1]");

            // Clean serial replay advances only the anchor and accepted matches.
            reference->decode(pending);
            for (int index = 0; index < expected_accepted; ++index)
                reference->decode(future[static_cast<std::size_t>(index)]);
            pending = result.pending;
            require(sample_target(*reference) == pending,
                    label + " clean reference pending mismatch");
            require_pending_observable_equal(
                capture_pending_observable(*test), capture_pending_observable(*reference),
                label + " round " + std::to_string(round_index));
            const long long end = static_cast<long long>(prefix.size() + processed.size());
            const int expected_count = static_cast<int>(
                std::min<long long>(end, Exl3Dflash2DraftModel::ring_keep()));
            require(test->position() == end && draft.ring_count() == expected_count &&
                        draft.ring_base_abs() == end - expected_count,
                    label + " ring/cache chronology mismatch");
        }

        // Four untouched serial continuations expose any lost/double-decoded
        // pending token and semantically cover recurrent/conv continuation state.
        for (int continuation = 0; continuation < 4; ++continuation) {
            const int abs_pos = test->position();
            test->decode(pending);
            commit_latest_row(*test, draft, test_stage, abs_pos);
            reference->decode(pending);
            processed.push_back(pending);
            const auto test_next = sample_target(*test);
            const auto reference_next = sample_target(*reference);
            require(test_next == reference_next, label + " continuation token mismatch");
            pending = test_next;
            emitted.push_back(pending);
            expected_emitted.push_back(pending);
            require_pending_observable_equal(
                capture_pending_observable(*test), capture_pending_observable(*reference),
                label + " continuation " + std::to_string(continuation));
        }
        require(emitted == expected_emitted && processed.size() + 1 == emitted.size() &&
                    std::equal(processed.begin(), processed.end(), emitted.begin()),
                label + " final pending ledger mismatch");

        const auto live_digest = draft.ring_digest();
        const long long live_end = static_cast<long long>(prefix.size() + processed.size());
        const int live_count = static_cast<int>(
            std::min<long long>(live_end, Exl3Dflash2DraftModel::ring_keep()));
        require(draft.ring_count() == live_count &&
                    draft.ring_base_abs() == live_end - live_count,
                label + " final ring boundary mismatch");
        std::uint64_t test_oscar = 0, test_ordinary = 0;
        test->oscar_routing_counts(test_oscar, test_ordinary);
        require(test_oscar > 0 && test_ordinary == 0,
                label + " escaped canonical OSCAR routing");
        test.reset();
        reference.reset();

        // Independently rebuild ring contents from real target taps without the
        // shared verifier and compare the complete logical K/V digest.
        draft.reset();
        auto replay = target.create_context(true);
        require(replay->try_enable_oscar_from_environment(), label + " replay OSCAR enable failed");
        ingest_prefix(*replay, prefix, CommitSink{&draft, &replay_stage});
        for (std::size_t index = 0; index < processed.size(); ++index) {
            const int abs_pos = replay->position();
            replay->decode(processed[index]);
            commit_latest_row(*replay, draft, replay_stage, abs_pos);
        }
        require(draft.ring_digest() == live_digest && draft.ring_count() == live_count &&
                    draft.ring_base_abs() == live_end - live_count,
                label + " independently replayed ring mismatch");
        std::cout << "PENDINGQUAL_CASE_PASS label=" << label
                  << " ctx=" << prefix.size() << " B=" << block_len
                  << " forced_rejection=" << forced_rejection
                  << " rounds=2 continuations=4 emitted=" << emitted.size()
                  << " processed=" << processed.size()
                  << " ring_count=" << live_count
                  << " ring_base=" << live_end - live_count << '\n';
    };

    const auto prefix32 = extended_prefix(32);
    for (int block_len : {2, 4, 8}) {
        for (int rejection = 0; rejection < block_len - 1; ++rejection)
            run_case(prefix32, block_len, rejection,
                     "ctx32-B" + std::to_string(block_len) + "-reject" +
                         std::to_string(rejection));
        run_case(prefix32, block_len, -1,
                 "ctx32-B" + std::to_string(block_len) + "-full");
    }
    const auto prefix512 = extended_prefix(512);
    for (int outcome : {0, 6, -1})
        run_case(prefix512, 8, outcome,
                 outcome < 0 ? "ctx512-B8-full" :
                     "ctx512-B8-reject" + std::to_string(outcome));

    // Live real-tap crossing of the logical 2047-row keep boundary. Repetition
    // of the provided canonical token fixture is explicit and deterministic.
    run_case(extended_prefix(2047), 2, -1, "ctx2047-B2-full-wrap");
    std::cout << "PENDINGQUAL_PASS protocol=pending_anchor_v1"
              << " raw_oscar_or_conv_byte_compare=unavailable"
              << " semantic_continuations=4 gdn_states=48\n";
}
