#pragma once

void run_transaction_retention_odd_qualification(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& source_ids) {
    require(target.max_context() == 4096 && source_ids.size() >= 4096,
            "transactionretainoddqual requires maxctx4096 and real prompt IDs");
    require(env("NINFER_OSCAR_EXL3") == "1",
            "transactionretainoddqual requires canonical OSCAR");

    int cases = 0;
    auto run_case = [&](int prefix_rows, int block, int rejection) {
        require((block == 3 || block == 5 || block == 6) &&
                    (rejection == -1 || (rejection >= 0 && rejection <= block - 2)),
                "transactionretainoddqual invalid matrix case");
        const std::string label = "transactionretainoddqual ctx=" +
            std::to_string(prefix_rows) + " B=" + std::to_string(block) +
            " rejection=" + std::to_string(rejection);
        const std::vector<std::int64_t> prefix(source_ids.begin(),
                                               source_ids.begin() + prefix_rows);
        auto candidate = target.create_context(true);
        auto reference = target.create_context(true);
        require(candidate->try_enable_oscar_from_environment() &&
                    reference->try_enable_oscar_from_environment(),
                label + " OSCAR enable");
        candidate->prepare_transaction();
        candidate->prepare_continuation(block);
        reference->prepare_transaction();
        draft.reset();
        auto staging = make_transaction_round_stage();
        ingest_prefix(*candidate, prefix, CommitSink{&draft, &staging});
        ingest_prefix(*reference, prefix, CommitSink{});
        require(prefix_retention_snapshot(*candidate) == prefix_retention_snapshot(*reference),
                label + " base state mismatch");

        auto pending = sample_target(*candidate);
        require(pending == sample_target(*reference), label + " initial pending mismatch");
        std::vector<std::int64_t> emitted{pending}, processed;
        std::array<std::array<std::uint64_t, 5>, 2> round_digests{};
        std::array<std::size_t, 2> processed_cuts{};
        int attempted_total = 0, replay_total = 0, reconstruction_total = 0;
        for (int round = 0; round < 2; ++round) {
            reference->begin_transaction();
            reference->decode(pending);
            std::vector<std::int64_t> future;
            for (int row = 0; row < block; ++row) {
                future.push_back(sample_target(*reference));
                if (row + 1 < block) reference->decode(future.back());
            }
            reference->rollback_transaction();

            std::vector<std::int64_t> proposals(future.begin(), future.end() - 1);
            if (rejection >= 0)
                proposals[static_cast<std::size_t>(rejection)] =
                    (proposals[static_cast<std::size_t>(rejection)] + 1) % kVocab;
            const int position = candidate->position();
            const auto result = verify_pending_round_transactional(
                *candidate, draft, staging, pending, proposals, position,
                nullptr, -1, 0x7e00u, TransactionRepairMode::retain_prefix);
            const int accepted = rejection < 0 ? block - 1 : rejection;
            const int retained = accepted + 1;
            require(result.accepted == accepted && result.rejection_index == rejection &&
                        result.attempted_rows == block && result.retained_rows == retained &&
                        result.replay_rows == 0 &&
                        result.state_reconstruction_rows == (rejection < 0 ? 0 : retained) &&
                        !candidate->transaction_active(),
                    label + " result/accounting mismatch");
            std::vector<std::int64_t> expected_emitted(future.begin(),
                                                       future.begin() + retained);
            require(result.emitted == expected_emitted && result.pending == future[accepted],
                    label + " emitted/pending mismatch");

            std::vector<std::uint16_t> expected_logits, expected_embedding;
            std::array<std::vector<std::uint16_t>, 5> expected_taps;
            for (int row = 0; row < retained; ++row) {
                const auto token = row == 0 ? pending : future[static_cast<std::size_t>(row - 1)];
                reference->decode(token);
                processed.push_back(token);
                const auto logits = target_continue_device_bits(
                    reference->logits_device(), kVocab, label + " reference logits");
                expected_logits.insert(expected_logits.end(), logits.begin(), logits.end());
                const auto embedding = reference->embedding_bits_host_for_test();
                expected_embedding.insert(expected_embedding.end(), embedding.begin(), embedding.end());
                for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
                    const auto values = target_continue_tap_bits(*reference, kTapLayers[tap], 1);
                    expected_taps[tap].insert(expected_taps[tap].end(), values.begin(), values.end());
                }
            }
            const auto actual = prefix_retention_snapshot(*candidate);
            const auto expected = prefix_retention_snapshot(*reference);
            require_prefix_retention_semantic(actual, expected, label + " round state");
            require_transaction_round_state_equal(*candidate, *reference,
                                                   label + " final tap/state");
            require(actual.embedding == expected_embedding && actual.taps == expected_taps &&
                        actual.continuation_logits == expected_logits,
                    label + " retained all-row payload mismatch");
            for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
                const auto staged = transaction_round_device_bits(
                    staging.bulk_ptrs[tap], static_cast<std::size_t>(retained) * kHidden,
                    "transactionretainoddqual staged taps");
                require(staged == expected_taps[tap],
                        label + " published staging includes rejected/wrong taps");
            }

            emitted.insert(emitted.end(), result.emitted.begin(), result.emitted.end());
            require(processed.size() + 1 == emitted.size() &&
                        std::equal(processed.begin(), processed.end(), emitted.begin()),
                    label + " processed/emitted ledger mismatch");
            pending = result.pending;
            attempted_total += result.attempted_rows;
            replay_total += result.replay_rows;
            reconstruction_total += result.state_reconstruction_rows;
            const auto end = static_cast<long long>(prefix_rows + processed.size());
            const auto count = std::min<long long>(end, Exl3Dflash2DraftModel::ring_keep());
            require(draft.ring_count() == count && draft.ring_base_abs() == end - count,
                    label + " ring cursor mismatch");
            round_digests[round] = draft.ring_digest();
            processed_cuts[round] = processed.size();
        }

        std::uint64_t oscar = 0, ordinary = 0;
        candidate->oscar_routing_counts(oscar, ordinary);
        require(oscar > 0 && ordinary == 0 &&
                    candidate->oscar_telemetry().gdn_oscar_dispatches == 0,
                label + " escaped canonical OSCAR");
        for (int step = 0; step < 4; ++step) {
            candidate->decode(pending);
            reference->decode(pending);
            require(prefix_retention_snapshot(*candidate) == prefix_retention_snapshot(*reference),
                    label + " following M1 state/payload mismatch");
            pending = sample_target(*reference);
            require(pending == sample_target(*candidate), label + " following greedy mismatch");
        }

        candidate.reset();
        reference.reset();
        draft.reset();
        auto rebuild = target.create_context(true);
        require(rebuild->try_enable_oscar_from_environment(), label + " rebuild OSCAR");
        auto rebuild_staging = make_transaction_round_stage();
        ingest_prefix(*rebuild, prefix, CommitSink{&draft, &rebuild_staging});
        int cut = 0;
        for (std::size_t index = 0; index < processed.size(); ++index) {
            const int position = rebuild->position();
            rebuild->decode(processed[index]);
            commit_latest_row(*rebuild, draft, rebuild_staging, position);
            if (cut < 2 && index + 1 == processed_cuts[cut]) {
                const auto end = static_cast<long long>(prefix_rows + index + 1);
                const auto count = std::min<long long>(end, Exl3Dflash2DraftModel::ring_keep());
                require(draft.ring_digest() == round_digests[cut] &&
                            draft.ring_count() == count && draft.ring_base_abs() == end - count,
                        label + " independently rebuilt ring mismatch");
                ++cut;
            }
        }
        require(cut == 2, label + " missing ring comparison");
        ++cases;
        std::cout << "TRANSACTIONRETAINODD_CASE PASS ctx=" << prefix_rows
                  << " B=" << block << " rejection=" << rejection
                  << " route=retain rounds=2 continuation=4 attempted=" << attempted_total
                  << " replay=" << replay_total << " reconstruction=" << reconstruction_total
                  << " retained=" << processed.size() << " ring_count=" << draft.ring_count()
                  << " ring_base=" << draft.ring_base_abs()
                  << " ring_digests=2 state=exact payload=exact" << std::endl;
    };

    for (const int block : {3, 5, 6}) {
        for (int rejection = 0; rejection <= block - 2; ++rejection)
            run_case(32, block, rejection);
        run_case(32, block, -1);
    }
    run_case(63, 3, 0);  run_case(63, 3, -1);
    run_case(319, 5, 2); run_case(319, 5, -1);
    run_case(575, 6, 4); run_case(575, 6, -1);
    for (const int block : {3, 5, 6}) {
        run_case(512, block, block - 2);
        run_case(512, block, -1);
    }
    for (const int block : {3, 5, 6}) {
        run_case(2047, block, 0);
        run_case(2047, block, -1);
    }
    require(cases == 32, "transactionretainoddqual matrix count mismatch");
    std::cout << "TRANSACTIONRETAINODD PASS cases=32 retain=32 default=0 failures=0 nonfinite=0"
              << " rounds=2 continuation=4 ring_digests=2 gdn=48 conv_slots=4 oscar_live=exact"
              << " payload=exact model_replay_separate=1 guards=inherited" << std::endl;
}
