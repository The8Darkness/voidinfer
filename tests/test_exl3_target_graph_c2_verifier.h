#pragma once

// Test-only complete two-request verifier.  Both arms execute the same captured
// fixed-B8 graphs.  C1 completely finalizes the first request before submitting
// the second; C2 submits both and then finalizes them. Proposal calls remain serialized on one host
// thread and every request owns its target context, stream, tap stage and draft
// ring.

struct TargetGraphC2VerifierLaneTrace {
    std::vector<std::int64_t> proposals;
    std::vector<std::int64_t> emitted;
    std::int64_t pending = -1;
    int accepted = 0;
    int rejection_index = -1;
    int retained_rows = 0;
    int position = 0;
    long long ring_base = 0;
    int ring_count = 0;
    std::array<std::uint64_t, 5> ring_digest{};
    int publication_count = 0;
    std::uint64_t generation = 0;
    std::uint32_t route_bits = 0;
    bool position_ready = false;
    bool launch_ready = false;
    double proposal_ms = 0.0;
    double verifier_latency_ms = 0.0;
    double full_read_latency_ms = 0.0;
    TargetGraphC2Snapshot attempt;
    TargetGraphC2Snapshot target;
};

struct TargetGraphC2VerifierArmTrace {
    std::array<TargetGraphC2VerifierLaneTrace, 2> lanes;
    double setup_ms = 0.0;
    double capture_ms = 0.0;
    double complete_verifier_pair_ms = 0.0;
    double full_read_pair_ms = 0.0;
    std::size_t ready_free_bytes = 0;
    std::size_t end_free_bytes = 0;
    std::size_t ready_owned_bytes = 0;
    std::size_t end_owned_bytes = 0;
};

struct TargetGraphC2VerifierLane {
    std::unique_ptr<Exl3TextContext> target;
    Exl3Dflash2DraftModel* draft = nullptr;
    TargetGraphC2Stream stream;
    TapStage stage;
    std::vector<std::int64_t> prefix;
    std::int64_t pending = -1;
    int abs_pos = 0;
    int publication_count = 0;

    TargetGraphC2VerifierLane(Exl3TextModel& model,
                              Exl3Dflash2DraftModel& draft_model,
                              std::vector<std::int64_t> request_prefix)
        : target(model.create_context(true)), draft(&draft_model),
          stage(make_transaction_round_stage()), prefix(std::move(request_prefix)) {
        require(target->try_enable_oscar_from_environment(),
                "target-graph-c2 verifier OSCAR attachment");
        target->prepare_transaction();
        target->prepare_continuation(8);
    }
};

std::size_t target_graph_c2_verifier_owned_bytes(
    const TargetGraphC2VerifierLane& a,
    const TargetGraphC2VerifierLane& b) {
    return a.target->persistent_bytes() + a.target->transaction_bytes() +
        a.target->continuation_bytes() + b.target->persistent_bytes() +
        b.target->transaction_bytes() + b.target->continuation_bytes() +
        a.draft->weight_bytes() + a.draft->kv_bytes() +
        a.draft->scratch_bytes() + a.draft->ring_bytes() +
        b.draft->weight_bytes() + b.draft->kv_bytes() +
        b.draft->scratch_bytes() + b.draft->ring_bytes() +
        2ULL * kTapCount * 17ULL * kHidden * sizeof(std::uint16_t);
}

std::string target_graph_c2_verifier_ids(const std::vector<std::int64_t>& values) {
    std::ostringstream out;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) out << '|';
        out << values[index];
    }
    return out.str();
}

void target_graph_c2_verifier_require_lane_equal(
    const TargetGraphC2VerifierLaneTrace& left,
    const TargetGraphC2VerifierLaneTrace& right, const std::string& label) {
    require(left.proposals == right.proposals && left.emitted == right.emitted &&
                left.pending == right.pending && left.accepted == right.accepted &&
                left.rejection_index == right.rejection_index &&
                left.retained_rows == right.retained_rows &&
                left.position == right.position && left.ring_base == right.ring_base &&
                left.ring_count == right.ring_count &&
                left.ring_digest == right.ring_digest &&
                left.publication_count == right.publication_count &&
                left.route_bits == right.route_bits && left.position_ready &&
                right.position_ready && left.launch_ready && right.launch_ready,
            label + " proposal/decision/ring/admission mismatch");
    target_graph_c2_require_snapshot_equal(
        left.attempt, right.attempt, true, label + " exhaustive attempted state");
    target_graph_c2_require_snapshot_equal(
        left.target, right.target, false, label + " exhaustive target state");
}

void target_graph_c2_verifier_require_admission(
    const ninfer::exl3::Exl3ContinuationGraphAdmission& admission,
    cudaStream_t stream, const std::string& label, bool ready) {
    constexpr std::uint32_t required =
        ninfer::exl3::Exl3ContinuationGraphRouteFixedB8 |
        ninfer::exl3::Exl3ContinuationGraphRouteChronologicalOscar |
        ninfer::exl3::Exl3ContinuationGraphRouteGdnQkvzConcurrent;
    require(admission.active && admission.capacity == 8 &&
                admission.route_generation > 0 && admission.position_ready &&
                admission.graph_origin_stream_identity ==
                    reinterpret_cast<std::uintptr_t>(stream) &&
                (admission.qualified_route_bits & required) == required &&
                admission.ready == ready,
            label + " typed graph admission mismatch");
}

TargetGraphC2VerifierArmTrace target_graph_c2_verifier_run_arm(
    TargetGraphC2VerifierLane& a, TargetGraphC2VerifierLane& b,
    bool overlap, bool b_first, bool exhaustive_attempt_audit,
    bool reuse_live_request = false) {
    TargetGraphC2VerifierArmTrace result;
    TargetGraphC2VerifierLane* lanes[2]{&a, &b};
    if (!reuse_live_request) {
        const auto setup_started = std::chrono::steady_clock::now();
        for (auto* lane : lanes) {
            lane->target->reset(lane->stream.value);
            lane->draft->reset(lane->stream.value);
            cuda_check(cudaStreamSynchronize(lane->stream.value),
                       "target-graph-c2 verifier reset");
            // Populate each private draft ring through the qualified target-tap
            // prefill path. A last-row-only seed does not represent the drafter's
            // context and can produce invalid head inputs.
            ingest_prefix(*lane->target, lane->prefix,
                          CommitSink{lane->draft, &lane->stage});
            cuda_check(cudaStreamSynchronize(nullptr),
                       "target-graph-c2 verifier seed private ring");
            lane->pending = sample_target(*lane->target, lane->stream.value);
            lane->abs_pos = static_cast<int>(lane->prefix.size());
            lane->publication_count = 0;
            require(lane->draft->ring_base_abs() + lane->draft->ring_count() ==
                        lane->abs_pos,
                    "target-graph-c2 verifier initial target/ring tail");
        }
        result.setup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - setup_started).count();
        const auto capture_started = std::chrono::steady_clock::now();
        for (auto* lane : lanes) {
            require(lane->target->capture_continuation_graph(lane->stream.value),
                    "target-graph-c2 verifier graph capture: " +
                        lane->target->continuation_graph_status());
            cuda_check(cudaStreamSynchronize(lane->stream.value),
                       "target-graph-c2 verifier graph capture complete");
            const auto first = lane->target->continuation_graph_admission();
            target_graph_c2_verifier_require_admission(
                first, lane->stream.value, "target-graph-c2 verifier idle", false);
            require(lane->target->capture_continuation_graph(lane->stream.value) &&
                        lane->target->continuation_graph_admission().route_generation ==
                            first.route_generation,
                    "target-graph-c2 verifier same-origin generation reuse");
        }
        result.capture_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - capture_started).count();
    } else {
        for (auto* lane : lanes) {
            require(lane->target->position() == lane->abs_pos &&
                        lane->draft->ring_base_abs() + lane->draft->ring_count() ==
                            lane->abs_pos &&
                        !lane->target->transaction_active(),
                    "target-graph-c2 verifier live request state");
            target_graph_c2_verifier_require_admission(
                lane->target->continuation_graph_admission(), lane->stream.value,
                "target-graph-c2 verifier persistent idle", false);
        }
    }
    cuda_check(cudaDeviceSynchronize(), "target-graph-c2 verifier ready fence");
    std::size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&result.ready_free_bytes, &total_bytes),
               "target-graph-c2 verifier ready memory");
    result.ready_owned_bytes = target_graph_c2_verifier_owned_bytes(a, b);
    require(result.ready_free_bytes >= (1ULL << 30),
            "target-graph-c2 verifier requires at least 1 GiB free after ready state");

    // Export both private publication roots outside production timing.  This is
    // the only restoration authority after a wrapping draft append; rewind is
    // deliberately not used because evicted slots and host-ring failure state
    // are not recoverable from a count alone.
    std::array<std::shared_ptr<const ninfer::exl3::Exl3DraftHostRing>, 2> ring_roots;
    std::array<long long, 2> root_bases{};
    std::array<int, 2> root_counts{};
    std::array<std::array<std::uint64_t, 5>, 2> root_digests{};
    for (int lane_index = 0; lane_index < 2; ++lane_index) {
        auto& lane = *lanes[lane_index];
        root_bases[lane_index] = lane.draft->ring_base_abs();
        root_counts[lane_index] = lane.draft->ring_count();
        root_digests[lane_index] = lane.draft->ring_digest(lane.stream.value);
        ring_roots[lane_index] = lane.draft->export_host_ring(lane.stream.value);
        require(ring_roots[lane_index] &&
                    root_bases[lane_index] + root_counts[lane_index] == lane.abs_pos,
                "target-graph-c2 verifier private publication root");
    }

    std::array<std::vector<std::int64_t>, 2> blocks;
    const int order[2]{b_first ? 1 : 0, b_first ? 0 : 1};
    const auto full_started = std::chrono::steady_clock::now();
    for (const int lane_index : order) {
        auto& lane = *lanes[lane_index];
        blocks[lane_index].assign(8, kMaskToken);
        blocks[lane_index][0] = lane.pending;
        const auto proposal_started = std::chrono::steady_clock::now();
        result.lanes[lane_index].proposals = lane.draft->propose_cached(
            blocks[lane_index], lane.abs_pos, lane.target->target_embedding(),
            lane.target->target_lm_head_weights(),
            lane.target->target_lm_head_metadata(), kMaskToken,
            lane.stream.value);
        result.lanes[lane_index].proposal_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - proposal_started).count();
        require(result.lanes[lane_index].proposals.size() == 7,
                "target-graph-c2 verifier real proposal count");
        std::copy(result.lanes[lane_index].proposals.begin(),
                  result.lanes[lane_index].proposals.end(),
                  blocks[lane_index].begin() + 1);
    }

    const auto verifier_started = std::chrono::steady_clock::now();
    const auto idle_a = a.target->continuation_graph_admission();
    const auto idle_b = b.target->continuation_graph_admission();
    target_graph_c2_verifier_require_admission(
        idle_a, a.stream.value, "target-graph-c2 verifier round idle A", false);
    target_graph_c2_verifier_require_admission(
        idle_b, b.stream.value, "target-graph-c2 verifier round idle B", false);
    require(idle_a.oscar_split_class == idle_b.oscar_split_class,
            "target-graph-c2 verifier round heterogeneous idle split");
    for (auto* lane : lanes) lane->target->begin_transaction(lane->stream.value);
    for (auto* lane : lanes) {
        cuda_check(cudaStreamSynchronize(lane->stream.value),
                   "target-graph-c2 verifier begin transaction");
        target_graph_c2_verifier_require_admission(
            lane->target->continuation_graph_admission(), lane->stream.value,
            "target-graph-c2 verifier launch", true);
    }
    require(a.target->continuation_graph_admission().oscar_split_class ==
                b.target->continuation_graph_admission().oscar_split_class,
            "target-graph-c2 verifier heterogeneous cohort split");
    const auto launch = [&](int lane_index) {
        auto& lane = *lanes[lane_index];
        lane.target->continue_rows_graph(blocks[lane_index], lane.stream.value);
    };
    double audit_ms = 0.0;
    const auto finish_lane = [&](int lane_index) {
        auto& lane = *lanes[lane_index];
        auto& trace = result.lanes[lane_index];
        cuda_check(cudaStreamSynchronize(lane.stream.value),
                   overlap ? "target-graph-c2 verifier C2 lane join"
                           : "target-graph-c2 verifier C1 lane join");
        if (exhaustive_attempt_audit) {
            const auto audit_started = std::chrono::steady_clock::now();
            trace.attempt = target_graph_c2_snapshot(
                *lane.target, lane.stream.value, true);
            audit_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - audit_started).count();
        }
        const auto logits = lane.target->continuation_logits_bits_host(lane.stream.value);
        validate_transaction_logits(logits);
        while (trace.accepted < 7 &&
               trace.proposals[static_cast<std::size_t>(trace.accepted)] ==
                   transaction_row_argmax(logits, trace.accepted))
            ++trace.accepted;
        trace.rejection_index = trace.accepted == 7 ? -1 : trace.accepted;
        trace.retained_rows = trace.accepted + 1;
        trace.pending = transaction_row_argmax(logits, trace.accepted);
        trace.emitted.assign(trace.proposals.begin(),
                             trace.proposals.begin() + trace.accepted);
        trace.emitted.push_back(trace.pending);
        const int expected_count = std::min(
            Exl3Dflash2DraftModel::ring_keep(),
            root_counts[lane_index] + trace.retained_rows);
        const long long expected_base = lane.abs_pos + trace.retained_rows - expected_count;
        require(root_bases[lane_index] + root_counts[lane_index] == lane.abs_pos &&
                    (root_counts[lane_index] + trace.retained_rows <=
                         Exl3Dflash2DraftModel::ring_keep() ||
                     root_counts[lane_index] == Exl3Dflash2DraftModel::ring_keep()),
                "target-graph-c2 verifier private ring append precondition");
        try {
            lane.target->retain_transaction_prefix(trace.retained_rows,
                                                    lane.stream.value);
            for (int tap = 0; tap < kTapCount; ++tap) {
                lane.target->copy_tap_rows_to_device(
                    kTapLayers[static_cast<std::size_t>(tap)], 0,
                    lane.stage.bulk_ptrs[static_cast<std::size_t>(tap)],
                    trace.retained_rows, lane.stream.value);
            }
            cuda_check(cudaStreamSynchronize(lane.stream.value),
                       "target-graph-c2 verifier retained tap staging");
            lane.draft->commit_target_block(
                const_cast<const std::uint16_t**>(lane.stage.bulk_ptrs.data()),
                trace.retained_rows, lane.abs_pos, lane.stream.value);
            cuda_check(cudaStreamSynchronize(lane.stream.value),
                       "target-graph-c2 verifier private ring commit");
            require(lane.draft->ring_base_abs() == expected_base &&
                        lane.draft->ring_count() == expected_count &&
                        lane.draft->ring_base_abs() + lane.draft->ring_count() ==
                            lane.abs_pos + trace.retained_rows,
                    "target-graph-c2 verifier private ring did not advance");
            lane.target->commit_transaction();
        } catch (...) {
            const auto failure = std::current_exception();
            // A restore failure is lane/process poison and intentionally
            // supersedes the original exception: no successful rollback is
            // claimed unless the saved root is bit-exact again.
            lane.draft->restore_host_ring(ring_roots[lane_index], lane.stream.value);
            cuda_check(cudaStreamSynchronize(lane.stream.value),
                       "target-graph-c2 verifier restore private ring root");
            require(lane.draft->ring_base_abs() == root_bases[lane_index] &&
                        lane.draft->ring_count() == root_counts[lane_index] &&
                        lane.draft->ring_digest(lane.stream.value) ==
                            root_digests[lane_index],
                    "target-graph-c2 verifier restored ring root mismatch");
            if (lane.target->transaction_active()) {
                lane.target->rollback_transaction(lane.stream.value);
                cuda_check(cudaStreamSynchronize(lane.stream.value),
                           "target-graph-c2 verifier failure target rollback");
            }
            std::rethrow_exception(failure);
        }
        ++lane.publication_count;
        lane.abs_pos += trace.retained_rows;
        lane.pending = trace.pending;
        trace.publication_count = lane.publication_count;
        trace.position = lane.target->position();
        trace.ring_base = lane.draft->ring_base_abs();
        trace.ring_count = lane.draft->ring_count();
        const auto admission = lane.target->continuation_graph_admission();
        trace.generation = admission.route_generation;
        trace.route_bits = admission.qualified_route_bits;
        trace.position_ready = admission.position_ready;
        trace.launch_ready = true;
        require(!lane.target->transaction_active() && admission.active &&
                    !admission.ready && admission.route_generation > 0,
                "target-graph-c2 verifier transaction close/graph reuse");
        trace.verifier_latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - verifier_started).count() - audit_ms;
        trace.full_read_latency_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - full_started).count() - audit_ms;
    };
    if (overlap) {
        launch(order[0]);
        launch(order[1]);
        finish_lane(order[0]);
        finish_lane(order[1]);
    } else {
        launch(order[0]);
        finish_lane(order[0]);
        launch(order[1]);
        finish_lane(order[1]);
    }
    result.complete_verifier_pair_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - verifier_started).count() - audit_ms;
    result.full_read_pair_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - full_started).count() - audit_ms;
    for (int lane_index = 0; lane_index < 2; ++lane_index) {
        const auto& lane = *lanes[lane_index];
        require(result.lanes[lane_index].position == lane.abs_pos &&
                    result.lanes[lane_index].ring_base +
                            result.lanes[lane_index].ring_count == lane.abs_pos,
                "target-graph-c2 verifier published target/ring tail mismatch");
        result.lanes[lane_index].ring_digest =
            lane.draft->ring_digest(lane.stream.value);
        result.lanes[lane_index].target = target_graph_c2_snapshot(
            *lane.target, lane.stream.value, false);
    }
    cuda_check(cudaDeviceSynchronize(), "target-graph-c2 verifier end fence");
    cuda_check(cudaMemGetInfo(&result.end_free_bytes, &total_bytes),
               "target-graph-c2 verifier end memory");
    result.end_owned_bytes = target_graph_c2_verifier_owned_bytes(a, b);
    require(result.end_owned_bytes == result.ready_owned_bytes,
            "target-graph-c2 verifier logical owned-byte growth");
    return result;
}

struct TargetGraphC2VerifierCancellationTrace {
    TargetGraphC2VerifierLaneTrace survivor;
    TargetGraphC2Snapshot canceled_root;
    std::array<std::uint64_t, 5> canceled_ring_digest{};
    long long canceled_ring_base = 0;
    int canceled_ring_count = 0;
    int canceled_publications = 0;
    bool canceled_graph_launched = false;
    bool transactions_closed = false;
    double settle_ms = 0.0;
    std::size_t ready_free_bytes = 0;
    std::size_t end_free_bytes = 0;
    std::size_t ready_owned_bytes = 0;
    std::size_t end_owned_bytes = 0;
};

TargetGraphC2VerifierCancellationTrace
target_graph_c2_verifier_run_cancellation_arm(
    TargetGraphC2VerifierLane& a, TargetGraphC2VerifierLane& b,
    int survivor_index, bool after_one_submission,
    bool reuse_live_request = false) {
    require(survivor_index == 0 || survivor_index == 1,
            "target-graph-c2 verifier cancellation survivor index");
    TargetGraphC2VerifierCancellationTrace result;
    TargetGraphC2VerifierLane* lanes[2]{&a, &b};
    const int canceled_index = 1 - survivor_index;
    if (!reuse_live_request) {
        for (auto* lane : lanes) {
            lane->target->reset(lane->stream.value);
            lane->draft->reset(lane->stream.value);
            cuda_check(cudaStreamSynchronize(lane->stream.value),
                       "target-graph-c2 verifier cancellation reset");
            ingest_prefix(*lane->target, lane->prefix,
                          CommitSink{lane->draft, &lane->stage});
            cuda_check(cudaStreamSynchronize(nullptr),
                       "target-graph-c2 verifier cancellation private ring seed");
            lane->pending = sample_target(*lane->target, lane->stream.value);
            lane->abs_pos = static_cast<int>(lane->prefix.size());
            lane->publication_count = 0;
            require(lane->target->capture_continuation_graph(lane->stream.value),
                    "target-graph-c2 verifier cancellation capture: " +
                        lane->target->continuation_graph_status());
            cuda_check(cudaStreamSynchronize(lane->stream.value),
                       "target-graph-c2 verifier cancellation capture complete");
            target_graph_c2_verifier_require_admission(
                lane->target->continuation_graph_admission(), lane->stream.value,
                "target-graph-c2 verifier cancellation idle", false);
        }
    } else {
        for (auto* lane : lanes) {
            require(lane->target->position() == lane->abs_pos &&
                        lane->draft->ring_base_abs() + lane->draft->ring_count() ==
                            lane->abs_pos &&
                        !lane->target->transaction_active(),
                    "target-graph-c2 verifier cancellation live request state");
            target_graph_c2_verifier_require_admission(
                lane->target->continuation_graph_admission(), lane->stream.value,
                "target-graph-c2 verifier cancellation persistent idle", false);
        }
    }
    std::size_t total_bytes = 0;

    std::array<std::vector<std::int64_t>, 2> blocks;
    std::array<std::vector<std::int64_t>, 2> proposals;
    for (int lane_index = 0; lane_index < 2; ++lane_index) {
        auto& lane = *lanes[lane_index];
        blocks[lane_index].assign(8, kMaskToken);
        blocks[lane_index][0] = lane.pending;
        proposals[lane_index] = lane.draft->propose_cached(
            blocks[lane_index], lane.abs_pos, lane.target->target_embedding(),
            lane.target->target_lm_head_weights(),
            lane.target->target_lm_head_metadata(), kMaskToken,
            lane.stream.value);
        require(proposals[lane_index].size() == 7,
                "target-graph-c2 verifier cancellation proposal count");
        std::copy(proposals[lane_index].begin(), proposals[lane_index].end(),
                  blocks[lane_index].begin() + 1);
    }
    auto& survivor = *lanes[survivor_index];
    auto& canceled = *lanes[canceled_index];
    const auto canceled_target_root = target_graph_c2_snapshot(
        *canceled.target, canceled.stream.value, false);
    const auto canceled_ring_root = canceled.draft->export_host_ring(
        canceled.stream.value);
    const auto canceled_digest_root = canceled.draft->ring_digest(
        canceled.stream.value);
    const long long canceled_base_root = canceled.draft->ring_base_abs();
    const int canceled_count_root = canceled.draft->ring_count();
    const auto survivor_ring_root = survivor.draft->export_host_ring(
        survivor.stream.value);
    const auto survivor_digest_root = survivor.draft->ring_digest(
        survivor.stream.value);
    const long long survivor_base_root = survivor.draft->ring_base_abs();
    const int survivor_count_root = survivor.draft->ring_count();
    require(canceled_ring_root && survivor_ring_root &&
                canceled_base_root + canceled_count_root == canceled.abs_pos &&
                survivor_base_root + survivor_count_root == survivor.abs_pos,
            "target-graph-c2 verifier cancellation ring roots");
    // Proposals plus exact target/ring roots are part of the fully ready
    // verifier state. Sample physical/logical baselines only after their lazy
    // graph/library/diagnostic initialization, before either transaction opens.
    cuda_check(cudaDeviceSynchronize(),
               "target-graph-c2 verifier cancellation ready fence");
    cuda_check(cudaMemGetInfo(&result.ready_free_bytes, &total_bytes),
               "target-graph-c2 verifier cancellation ready memory");
    result.ready_owned_bytes = target_graph_c2_verifier_owned_bytes(a, b);
    require(result.ready_free_bytes >= (1ULL << 30),
            "target-graph-c2 verifier cancellation requires 1 GiB reserve");

    survivor.target->begin_transaction(survivor.stream.value);
    if (after_one_submission)
        canceled.target->begin_transaction(canceled.stream.value);
    cuda_check(cudaStreamSynchronize(survivor.stream.value),
               "target-graph-c2 verifier cancellation survivor ready");
    target_graph_c2_verifier_require_admission(
        survivor.target->continuation_graph_admission(), survivor.stream.value,
        "target-graph-c2 verifier cancellation survivor launch", true);
    if (after_one_submission) {
        cuda_check(cudaStreamSynchronize(canceled.stream.value),
                   "target-graph-c2 verifier cancellation peer ready");
        target_graph_c2_verifier_require_admission(
            canceled.target->continuation_graph_admission(), canceled.stream.value,
            "target-graph-c2 verifier cancellation peer launch", true);
    } else {
        require(!canceled.target->transaction_active(),
                "target-graph-c2 verifier single-survivor peer transaction");
    }

    const auto started = std::chrono::steady_clock::now();
    survivor.target->continue_rows_graph(blocks[survivor_index],
                                         survivor.stream.value);
    if (after_one_submission) {
        canceled.target->rollback_transaction(canceled.stream.value);
        cuda_check(cudaStreamSynchronize(canceled.stream.value),
                   "target-graph-c2 verifier canceled peer settle");
    }
    cuda_check(cudaStreamSynchronize(survivor.stream.value),
               "target-graph-c2 verifier survivor join");
    const auto audit_started = std::chrono::steady_clock::now();
    result.survivor.attempt = target_graph_c2_snapshot(
        *survivor.target, survivor.stream.value, true);
    const double audit_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - audit_started).count();
    result.survivor.proposals = proposals[survivor_index];
    const auto logits = survivor.target->continuation_logits_bits_host(
        survivor.stream.value);
    validate_transaction_logits(logits);
    while (result.survivor.accepted < 7 &&
           result.survivor.proposals[
               static_cast<std::size_t>(result.survivor.accepted)] ==
               transaction_row_argmax(logits, result.survivor.accepted))
        ++result.survivor.accepted;
    result.survivor.rejection_index =
        result.survivor.accepted == 7 ? -1 : result.survivor.accepted;
    result.survivor.retained_rows = result.survivor.accepted + 1;
    result.survivor.pending = transaction_row_argmax(
        logits, result.survivor.accepted);
    result.survivor.emitted.assign(
        result.survivor.proposals.begin(),
        result.survivor.proposals.begin() + result.survivor.accepted);
    result.survivor.emitted.push_back(result.survivor.pending);
    try {
        survivor.target->retain_transaction_prefix(
            result.survivor.retained_rows, survivor.stream.value);
        for (int tap = 0; tap < kTapCount; ++tap) {
            survivor.target->copy_tap_rows_to_device(
                kTapLayers[static_cast<std::size_t>(tap)], 0,
                survivor.stage.bulk_ptrs[static_cast<std::size_t>(tap)],
                result.survivor.retained_rows, survivor.stream.value);
        }
        cuda_check(cudaStreamSynchronize(survivor.stream.value),
                   "target-graph-c2 verifier cancellation retained staging");
        survivor.draft->commit_target_block(
            const_cast<const std::uint16_t**>(survivor.stage.bulk_ptrs.data()),
            result.survivor.retained_rows, survivor.abs_pos,
            survivor.stream.value);
        cuda_check(cudaStreamSynchronize(survivor.stream.value),
                   "target-graph-c2 verifier cancellation ring publication");
        survivor.target->commit_transaction();
    } catch (...) {
        const auto failure = std::current_exception();
        survivor.draft->restore_host_ring(
            survivor_ring_root, survivor.stream.value);
        cuda_check(cudaStreamSynchronize(survivor.stream.value),
                   "target-graph-c2 verifier cancellation ring restore");
        require(survivor.draft->ring_base_abs() == survivor_base_root &&
                    survivor.draft->ring_count() == survivor_count_root &&
                    survivor.draft->ring_digest(survivor.stream.value) ==
                        survivor_digest_root,
                "target-graph-c2 verifier cancellation restored ring mismatch");
        if (survivor.target->transaction_active()) {
            survivor.target->rollback_transaction(survivor.stream.value);
            cuda_check(cudaStreamSynchronize(survivor.stream.value),
                       "target-graph-c2 verifier cancellation target rollback");
        }
        std::rethrow_exception(failure);
    }
    ++survivor.publication_count;
    survivor.abs_pos += result.survivor.retained_rows;
    survivor.pending = result.survivor.pending;
    require(survivor.target->position() == survivor.abs_pos &&
                survivor.draft->ring_base_abs() + survivor.draft->ring_count() ==
                    survivor.abs_pos,
            "target-graph-c2 verifier cancellation survivor tail");
    result.settle_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count() - audit_ms;
    result.survivor.publication_count = survivor.publication_count;
    result.survivor.position = survivor.target->position();
    result.survivor.ring_base = survivor.draft->ring_base_abs();
    result.survivor.ring_count = survivor.draft->ring_count();
    result.survivor.ring_digest = survivor.draft->ring_digest(
        survivor.stream.value);
    const auto survivor_admission = survivor.target->continuation_graph_admission();
    result.survivor.generation = survivor_admission.route_generation;
    result.survivor.route_bits = survivor_admission.qualified_route_bits;
    result.survivor.position_ready = survivor_admission.position_ready;
    result.survivor.launch_ready = true;
    result.survivor.target = target_graph_c2_snapshot(
        *survivor.target, survivor.stream.value, false);
    result.canceled_root = target_graph_c2_snapshot(
        *canceled.target, canceled.stream.value, false);
    result.canceled_ring_digest = canceled.draft->ring_digest(
        canceled.stream.value);
    result.canceled_ring_base = canceled.draft->ring_base_abs();
    result.canceled_ring_count = canceled.draft->ring_count();
    result.canceled_publications = canceled.publication_count;
    result.transactions_closed = !survivor.target->transaction_active() &&
        !canceled.target->transaction_active();
    result.end_owned_bytes = target_graph_c2_verifier_owned_bytes(a, b);
    cuda_check(cudaDeviceSynchronize(),
               "target-graph-c2 verifier cancellation end fence");
    cuda_check(cudaMemGetInfo(&result.end_free_bytes, &total_bytes),
               "target-graph-c2 verifier cancellation end memory");
    require(result.transactions_closed && survivor.publication_count == 1 &&
                canceled.publication_count == 0 &&
                canceled.target->continuation_rows() == 0 &&
                canceled.target->continuation_graph_admission().active &&
                !canceled.target->continuation_graph_admission().ready &&
                canceled.draft->ring_base_abs() == canceled_base_root &&
                canceled.draft->ring_count() == canceled_count_root &&
                result.canceled_ring_digest == canceled_digest_root,
            "target-graph-c2 verifier cancellation publication/closure");
    target_graph_c2_require_snapshot_equal(
        result.canceled_root, canceled_target_root, false,
        "target-graph-c2 verifier canceled target root");
    require(result.end_owned_bytes == result.ready_owned_bytes,
            "target-graph-c2 verifier cancellation logical growth");
    return result;
}

void target_graph_c2_verifier_cancellation_cases(
    TargetGraphC2VerifierLane& a, TargetGraphC2VerifierLane& b,
    const std::string& phase, const std::string& fixture,
    std::ofstream& cancellation) {
    for (int survivor = 0; survivor < 2; ++survivor) {
        // Establish post-warmup physical stability for both control shapes.
        // Each discarded arm still executes all internal state, publication,
        // transaction-closure and reserve assertions; only its timing/free
        // deltas are excluded from the measured cancellation ledger.
        const auto warm_control = target_graph_c2_verifier_run_cancellation_arm(
            a, b, survivor, false);
        const auto warm_candidate = target_graph_c2_verifier_run_cancellation_arm(
            a, b, survivor, true);
        const std::size_t warm_end_free =
            std::min(warm_control.end_free_bytes, warm_candidate.end_free_bytes);
        const auto control = target_graph_c2_verifier_run_cancellation_arm(
            a, b, survivor, false);
        const auto candidate = target_graph_c2_verifier_run_cancellation_arm(
            a, b, survivor, true);
        target_graph_c2_verifier_require_lane_equal(
            control.survivor, candidate.survivor,
            "target-graph-c2 verifier cancellation survivor exact");
        target_graph_c2_require_snapshot_equal(
            control.canceled_root, candidate.canceled_root, false,
            "target-graph-c2 verifier cancellation peer root exact");
        require(control.canceled_ring_base == candidate.canceled_ring_base &&
                    control.canceled_ring_count == candidate.canceled_ring_count &&
                    control.canceled_ring_digest == candidate.canceled_ring_digest &&
                    control.canceled_publications == 0 &&
                    candidate.canceled_publications == 0 &&
                    !control.canceled_graph_launched &&
                    !candidate.canceled_graph_launched &&
                    control.transactions_closed && candidate.transactions_closed,
                "target-graph-c2 verifier matched cancellation state");
        const double ratio = candidate.settle_ms / control.settle_ms;
        const std::size_t control_drift =
            control.end_free_bytes < control.ready_free_bytes
                ? control.ready_free_bytes - control.end_free_bytes : 0;
        const std::size_t candidate_drift =
            candidate.end_free_bytes < candidate.ready_free_bytes
                ? candidate.ready_free_bytes - candidate.end_free_bytes : 0;
        const std::size_t physical_drift =
            std::max(control_drift, candidate_drift);
        const std::size_t current_end_free =
            std::min(control.end_free_bytes, candidate.end_free_bytes);
        const bool physical_stable =
            current_end_free + (1ULL << 20) >= warm_end_free;
        const bool gate = ratio <= 1.25 && physical_stable;
        cancellation << phase << ',' << fixture << ','
                     << (survivor == 0 ? 'A' : 'B') << ','
                     << control.settle_ms << ',' << candidate.settle_ms << ','
                     << ratio << ',' << control.survivor.publication_count << ','
                     << candidate.survivor.publication_count << ','
                     << candidate.canceled_publications << ','
                     << candidate.canceled_graph_launched << ','
                     << control.transactions_closed << ','
                     << candidate.transactions_closed << ','
                     << std::min(control.ready_free_bytes,
                                 candidate.ready_free_bytes) << ','
                     << std::min(control.end_free_bytes,
                                 candidate.end_free_bytes) << ','
                     << warm_end_free << ','
                     << physical_drift << ','
                     << (candidate.end_owned_bytes - candidate.ready_owned_bytes)
                     << ',' << gate << '\n';
        cancellation.flush();
        require(ratio <= 1.25,
                "target-graph-c2 verifier cancellation survivor ratio");
        require(physical_stable,
                "target-graph-c2 verifier cancellation cumulative physical drift");
    }
}

void target_graph_c2_verifier_split_sentinels(
    TargetGraphC2VerifierLane& a, TargetGraphC2VerifierLane& b,
    const std::vector<std::int64_t>& source, std::ofstream& transitions) {
    TargetGraphC2VerifierLane* lanes[2]{&a, &b};
    std::uint64_t split16_generation[2]{};
    for (const int base : {504, 505, 512}) {
        for (int index = 0; index < 2; ++index) {
            auto& lane = *lanes[index];
            lane.target->reset(lane.stream.value);
            cuda_check(cudaStreamSynchronize(lane.stream.value),
                       "target-graph-c2 verifier sentinel reset");
            const auto prefix = target_graph_c2_shifted_prefix(
                source, base, index == 0 ? 0 : 17);
            ingest_prefix(*lane.target, prefix, CommitSink{});
        }
        if (base == 505) {
            const int position_a = a.target->position();
            const int position_b = b.target->position();
            require(!a.target->capture_continuation_graph(a.stream.value) &&
                        !b.target->capture_continuation_graph(b.stream.value) &&
                        !a.target->transaction_active() &&
                        !b.target->transaction_active() &&
                        a.target->position() == position_a &&
                        b.target->position() == position_b &&
                        !a.target->continuation_graph_admission().active &&
                        !b.target->continuation_graph_admission().active,
                    "target-graph-c2 verifier base505 atomic no-launch fallback");
            transitions << "505,none,0,0,1\n";
            continue;
        }
        for (int index = 0; index < 2; ++index) {
            auto& lane = *lanes[index];
            require(lane.target->capture_continuation_graph(lane.stream.value),
                    "target-graph-c2 verifier sentinel capture");
            cuda_check(cudaStreamSynchronize(lane.stream.value),
                       "target-graph-c2 verifier sentinel capture complete");
            const auto admission = lane.target->continuation_graph_admission();
            require(admission.oscar_split_class == (base == 504 ? 16 : 32) &&
                        admission.position_ready,
                    "target-graph-c2 verifier sentinel split class");
            if (base == 504) split16_generation[index] = admission.route_generation;
            else require(admission.route_generation > split16_generation[index],
                         "target-graph-c2 verifier split recapture generation");
        }
        transitions << base << ',' << (base == 504 ? 16 : 32) << ','
                    << a.target->continuation_graph_admission().route_generation << ','
                    << b.target->continuation_graph_admission().route_generation
                    << ",0\n";
    }
    transitions.flush();
}

double target_graph_c2_verifier_gain(double c1, double c2) {
    return (c1 - c2) * 100.0 / c1;
}

void run_target_graph_c2_verifier_t1(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft_a,
    const std::filesystem::path& draft_path,
    const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    const std::string phase = env("NINFER_EXL3_TARGET_GRAPH_C2_T1_PHASE");
    require(phase == "preflight" || phase == "screen",
            "targetgraphc2verifiert1 phase must be preflight or screen");
    require(!output.empty() && !std::filesystem::exists(output),
            "targetgraphc2verifiert1 output must be new");
    require(target.max_context() >= (phase == "screen" ? 4112 : 528),
            "targetgraphc2verifiert1 context extent");
    std::filesystem::create_directories(output);
    std::ofstream rounds(output / "rounds.csv");
    std::ofstream pairs(output / "pairs.csv");
    std::ofstream transitions(output / "transitions.csv");
    std::ofstream cancellation(output / "cancellation.csv");
    std::ofstream summary(output / "summary.csv");
    require(rounds.good() && pairs.good() && transitions.good() &&
                cancellation.good() && summary.good(),
            "targetgraphc2verifiert1 output creation");
    rounds << "phase,fixture,pair,warmup,arm,arm_order,request_order,lane,"
              "proposal_ids,accepted,rejection_index,retained_rows,emitted_ids,pending,"
              "position,ring_base,ring_count,publication_count,generation,route_bits,"
              "position_ready,launch_ready,proposal_ms,verifier_latency_ms,"
              "full_read_latency_ms,setup_ms,"
              "capture_ms,complete_verifier_pair_ms,full_read_pair_ms,ready_free_bytes,"
              "end_free_bytes,ready_owned_bytes,end_owned_bytes\n";
    pairs << "phase,fixture,pair,warmup,arm_order,request_order,c1_complete_ms,"
             "c2_complete_ms,complete_gain_pct,c1_full_read_ms,c2_full_read_ms,"
             "full_read_gain_pct,max_request_latency_ratio,physical_drift_bytes,exact\n";
    transitions << "base,split_class,generation_a,generation_b,fallback\n";
    cancellation << "phase,fixture,survivor,control_settle_ms,c2_cancel_settle_ms,"
                    "settle_ratio,control_survivor_publications,"
                    "c2_survivor_publications,canceled_publications,"
                    "canceled_graph_launched,control_transactions_closed,"
                    "c2_transactions_closed,ready_free_bytes,end_free_bytes,"
                    "warm_end_free_bytes,physical_drift_bytes,logical_growth_bytes,pass\n";
    summary << "phase,fixture,pairs,warmups,measured_pairs,"
               "complete_median_gain_pct,complete_worst_gain_pct,"
               "full_read_median_gain_pct,full_read_worst_gain_pct,"
               "max_request_latency_ratio,max_physical_drift_bytes,"
               "cancellation_cases,cancellation_gate,logical_growth_bytes,gate\n";

    auto draft_b = Exl3Dflash2DraftModel::load(draft_path);
    const auto run_fixture = [&](const std::vector<std::int64_t>& source,
                                 const std::string& fixture, int prefix_rows,
                                 int pair_count, int warmups) {
        TargetGraphC2VerifierLane a(
            target, draft_a, target_graph_c2_shifted_prefix(source, prefix_rows, 0));
        TargetGraphC2VerifierLane b(
            target, *draft_b, target_graph_c2_shifted_prefix(source, prefix_rows, 17));
        if (phase == "preflight" && fixture == "code")
            target_graph_c2_verifier_split_sentinels(a, b, source, transitions);
        target_graph_c2_verifier_cancellation_cases(
            a, b, phase, fixture, cancellation);
        std::vector<double> complete_gains, full_gains;
        double max_latency_ratio = 0.0;
        std::size_t warm_free = 0;
        std::size_t max_physical_drift = 0;
        for (int pair = 0; pair < pair_count; ++pair) {
            const bool warmup = pair < warmups;
            const bool c2_first = phase == "screen" &&
                ((!warmup && pair >= warmups + 2) || (warmup && pair == 1));
            const bool b_first = phase == "screen" &&
                ((warmup && pair == 1) || (!warmup && ((pair - warmups) & 1)));
            TargetGraphC2VerifierArmTrace c1, c2;
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const bool overlap = c2_first ? ordinal == 0 : ordinal == 1;
                auto arm = target_graph_c2_verifier_run_arm(
                    a, b, overlap, b_first, phase == "preflight");
                if (overlap) c2 = std::move(arm); else c1 = std::move(arm);
            }
            for (int lane = 0; lane < 2; ++lane) {
                target_graph_c2_verifier_require_lane_equal(
                    c1.lanes[lane], c2.lanes[lane],
                    "target-graph-c2 verifier matched lane " +
                        std::to_string(lane));
                require((c2_first
                            ? c1.lanes[lane].generation > c2.lanes[lane].generation
                            : c2.lanes[lane].generation > c1.lanes[lane].generation),
                        "target-graph-c2 verifier reset recapture generation sentinel");
            }
            const double latency_ratio = std::max(
                c2.lanes[0].verifier_latency_ms /
                    c1.lanes[0].verifier_latency_ms,
                c2.lanes[1].verifier_latency_ms /
                    c1.lanes[1].verifier_latency_ms);
            if (!warmup)
                max_latency_ratio = std::max(max_latency_ratio, latency_ratio);
            const std::size_t c1_drift = c1.end_free_bytes < c1.ready_free_bytes
                ? c1.ready_free_bytes - c1.end_free_bytes : 0;
            const std::size_t c2_drift = c2.end_free_bytes < c2.ready_free_bytes
                ? c2.ready_free_bytes - c2.end_free_bytes : 0;
            const std::size_t drift = std::max(c1_drift, c2_drift);
            if (!warmup) max_physical_drift = std::max(max_physical_drift, drift);
            const auto current_free = std::min(c1.end_free_bytes, c2.end_free_bytes);
            if (warmup) warm_free = current_free;
            const char* arm_order = c2_first ? "C2C1" : "C1C2";
            const char* request_order = b_first ? "BA" : "AB";
            pairs << phase << ',' << fixture << ',' << pair << ',' << warmup << ','
                  << arm_order << ',' << request_order << ','
                  << c1.complete_verifier_pair_ms << ','
                  << c2.complete_verifier_pair_ms << ','
                  << target_graph_c2_verifier_gain(c1.complete_verifier_pair_ms,
                                                   c2.complete_verifier_pair_ms) << ','
                  << c1.full_read_pair_ms << ',' << c2.full_read_pair_ms << ','
                  << target_graph_c2_verifier_gain(c1.full_read_pair_ms,
                                                   c2.full_read_pair_ms) << ','
                  << latency_ratio << ',' << drift << ",1\n";
            const TargetGraphC2VerifierArmTrace* arms[2]{&c1, &c2};
            for (int arm_index = 0; arm_index < 2; ++arm_index) {
                const auto& arm = *arms[arm_index];
                for (int lane = 0; lane < 2; ++lane) {
                    const auto& trace = arm.lanes[lane];
                    rounds << phase << ',' << fixture << ',' << pair << ',' << warmup
                           << ',' << (arm_index ? "C2" : "C1") << ',' << arm_order
                           << ',' << request_order << ',' << (lane ? 'B' : 'A') << ','
                           << target_graph_c2_verifier_ids(trace.proposals) << ','
                           << trace.accepted << ',' << trace.rejection_index << ','
                           << trace.retained_rows << ','
                           << target_graph_c2_verifier_ids(trace.emitted) << ','
                           << trace.pending << ',' << trace.position << ','
                           << trace.ring_base << ',' << trace.ring_count << ','
                           << trace.publication_count << ',' << trace.generation << ','
                           << trace.route_bits << ',' << trace.position_ready << ','
                           << trace.launch_ready << ',' << trace.proposal_ms << ','
                           << trace.verifier_latency_ms << ','
                           << trace.full_read_latency_ms << ',' << arm.setup_ms << ','
                           << arm.capture_ms << ',' << arm.complete_verifier_pair_ms << ','
                           << arm.full_read_pair_ms << ',' << arm.ready_free_bytes << ','
                           << arm.end_free_bytes << ',' << arm.ready_owned_bytes << ','
                           << arm.end_owned_bytes << '\n';
                }
            }
            if (!warmup) {
                complete_gains.push_back(target_graph_c2_verifier_gain(
                    c1.complete_verifier_pair_ms, c2.complete_verifier_pair_ms));
                full_gains.push_back(target_graph_c2_verifier_gain(
                    c1.full_read_pair_ms, c2.full_read_pair_ms));
                require(warmups == 0 ||
                            (warm_free != 0 &&
                             current_free + (1ULL << 20) >= warm_free),
                        "target-graph-c2 verifier physical drift exceeds 1 MiB");
            }
        }
        if (phase == "screen") {
            require(complete_gains.size() == 4 && full_gains.size() == 4,
                    "target-graph-c2 verifier measured pair count");
            require(target_graph_c2_standard_median(complete_gains) >= 10.0 &&
                        *std::min_element(complete_gains.begin(), complete_gains.end()) >= 0.0 &&
                        *std::min_element(full_gains.begin(), full_gains.end()) >= 0.0 &&
                        max_latency_ratio <= 1.5,
                    "target-graph-c2 verifier T1 performance gate");
        }
        require(!complete_gains.empty() && !full_gains.empty(),
                "target-graph-c2 verifier summary measured rows");
        summary << phase << ',' << fixture << ',' << pair_count << ',' << warmups
                << ',' << complete_gains.size() << ','
                << target_graph_c2_standard_median(complete_gains) << ','
                << *std::min_element(complete_gains.begin(), complete_gains.end())
                << ',' << target_graph_c2_standard_median(full_gains) << ','
                << *std::min_element(full_gains.begin(), full_gains.end()) << ','
                << max_latency_ratio << ',' << max_physical_drift
                << ",2,1,0,1\n";
        summary.flush();
    };

    if (phase == "preflight") {
        run_fixture(code, "code", 512, 1, 0);
        run_fixture(prose, "prose", 512, 1, 0);
    } else {
        run_fixture(code, "code", 4096, 6, 2);
        run_fixture(prose, "prose", 4096, 6, 2);
    }
    rounds.flush();
    pairs.flush();
    std::cout << "TARGET_GRAPH_C2_VERIFIER_T1 PASS phase=" << phase
              << " fixtures=2 cancellation_cases=4 c1_graph=1 c2_graph=1"
                 " real_draft=1 private_rings=2\n";
}
