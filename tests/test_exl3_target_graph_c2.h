#pragma once

// Test-only C2 overlap screen for two independently captured fixed-B8 target
// graphs.  There is deliberately no cross-context engine API, shared workspace,
// host worker, or projection rendezvous here.  The caller owns two ordinary
// contexts and two origin streams and waits for both before observing state.

struct TargetGraphC2Stream {
    cudaStream_t value = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;

    TargetGraphC2Stream() {
        cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking),
                   "target-graph-c2 create origin stream");
        cuda_check(cudaEventCreate(&begin), "target-graph-c2 create begin event");
        cuda_check(cudaEventCreate(&end), "target-graph-c2 create end event");
    }
    ~TargetGraphC2Stream() {
        if (end) cudaEventDestroy(end);
        if (begin) cudaEventDestroy(begin);
        if (value) cudaStreamDestroy(value);
    }
    TargetGraphC2Stream(const TargetGraphC2Stream&) = delete;
    TargetGraphC2Stream& operator=(const TargetGraphC2Stream&) = delete;
};

struct TargetGraphC2Snapshot {
    int position = 0;
    int device_position = 0;
    int tap_rows = 0;
    int embedding_rows = 0;
    int last_rows = 0;
    std::vector<std::uint16_t> final_logits;
    std::vector<std::uint16_t> continuation_logits;
    std::array<std::vector<std::uint16_t>, kTapCount> taps;
    std::vector<std::uint16_t> embedding;
    std::array<ninfer::exl3::Exl3FullAttentionQKVHost, 16> qkv;
    std::array<std::vector<float>, 64> gdn_recurrent;
    std::array<std::vector<std::uint16_t>, 64> gdn_conv;
    std::vector<std::byte> oscar;
};

bool target_graph_c2_float_bits_equal(const std::vector<float>& left,
                                      const std::vector<float>& right) {
    return left.size() == right.size() &&
        (left.empty() || std::memcmp(left.data(), right.data(),
                                    left.size() * sizeof(float)) == 0);
}

TargetGraphC2Snapshot target_graph_c2_snapshot(Exl3TextContext& context,
                                               cudaStream_t stream,
                                               bool continuation) {
    TargetGraphC2Snapshot result;
    result.position = context.position();
    result.device_position = context.device_position_host(stream);
    result.tap_rows = context.captured_tap_rows();
    result.embedding_rows = context.captured_embedding_rows();
    result.last_rows = context.last_forward_rows();
    result.final_logits = transaction_round_device_bits(
        context.logits_device(), kVocab, "target-graph-c2 download final logits");
    if (continuation) {
        require(context.continuation_rows() == 8,
                "target-graph-c2 continuation snapshot row count");
        result.continuation_logits = context.continuation_logits_bits_host(stream);
    }
    require(result.tap_rows > 0 && result.embedding_rows > 0 && result.last_rows > 0,
            "target-graph-c2 snapshot row metadata");
    for (int tap = 0; tap < kTapCount; ++tap) {
        result.taps[static_cast<std::size_t>(tap)] =
            transaction_round_tap_rows_bits(
                context, kTapLayers[static_cast<std::size_t>(tap)],
                result.tap_rows, stream);
    }
    result.embedding = context.embedding_bits_host_for_test(stream);
    require(result.embedding.size() ==
                static_cast<std::size_t>(result.embedding_rows) * kHidden,
            "target-graph-c2 embedding snapshot extent");
    int full = 0;
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) {
            if (continuation)
                result.qkv[static_cast<std::size_t>(full)] =
                    context.full_attention_qkv_host(layer, stream);
            ++full;
        } else {
            result.gdn_recurrent[static_cast<std::size_t>(layer)] =
                context.gdn_state_host(layer, stream);
            result.gdn_conv[static_cast<std::size_t>(layer)] =
                context.gdn_physical_conv_host(layer, stream);
        }
    }
    require(full == 16, "target-graph-c2 full-attention layer count");
    result.oscar = context.oscar_live_state_host_for_test(stream);
    return result;
}

void target_graph_c2_require_snapshot_equal(const TargetGraphC2Snapshot& actual,
                                            const TargetGraphC2Snapshot& expected,
                                            bool continuation,
                                            const std::string& label) {
    require(actual.position == expected.position &&
                actual.device_position == expected.device_position,
            label + " position mismatch");
    require(actual.tap_rows == expected.tap_rows &&
                actual.embedding_rows == expected.embedding_rows &&
                actual.last_rows == expected.last_rows,
            label + " captured-row metadata mismatch");
    require(actual.final_logits == expected.final_logits,
            label + " final logits mismatch");
    if (continuation)
        require(actual.continuation_logits == expected.continuation_logits,
                label + " all-row continuation logits mismatch");
    require(actual.taps == expected.taps, label + " all-row taps mismatch");
    require(actual.embedding == expected.embedding, label + " embedding mismatch");
    if (continuation) {
        for (std::size_t index = 0; index < actual.qkv.size(); ++index) {
            const auto& left = actual.qkv[index];
            const auto& right = expected.qkv[index];
            require(left.layer == right.layer && left.rows == right.rows &&
                        left.q_rope == right.q_rope &&
                        left.k_rope == right.k_rope &&
                        left.v_projection == right.v_projection,
                    label + " full-attention QKV mismatch index=" +
                        std::to_string(index));
        }
    }
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        require(target_graph_c2_float_bits_equal(
                    actual.gdn_recurrent[static_cast<std::size_t>(layer)],
                    expected.gdn_recurrent[static_cast<std::size_t>(layer)]),
                label + " GDN recurrent mismatch layer=" + std::to_string(layer));
        require(actual.gdn_conv[static_cast<std::size_t>(layer)] ==
                    expected.gdn_conv[static_cast<std::size_t>(layer)],
                label + " physical convolution mismatch layer=" +
                    std::to_string(layer));
    }
    require(actual.oscar == expected.oscar, label + " OSCAR live-state mismatch");
}

std::vector<std::int64_t> target_graph_c2_shifted_prefix(
    const std::vector<std::int64_t>& source, int rows, std::size_t shift) {
    require(!source.empty() && rows >= 32,
            "target-graph-c2 source/prefix extent");
    std::vector<std::int64_t> result(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row)
        result[static_cast<std::size_t>(row)] =
            source[(static_cast<std::size_t>(row) + shift) % source.size()];
    return result;
}

std::array<std::int64_t, 8> target_graph_c2_block(
    Exl3TextContext& context, const std::vector<std::int64_t>& source,
    std::size_t shift, cudaStream_t stream) {
    std::array<std::int64_t, 8> block{};
    block[0] = sample_target(context, stream);
    for (std::size_t row = 1; row < block.size(); ++row)
        block[row] = source[(shift + row) % source.size()];
    return block;
}

struct TargetGraphC2ArmTiming {
    double wall_ms = 0.0;
    double event_a_ms = 0.0;
    double event_b_ms = 0.0;
};

TargetGraphC2ArmTiming target_graph_c2_timed_arm(
    Exl3TextContext& a, Exl3TextContext& b,
    TargetGraphC2Stream& stream_a, TargetGraphC2Stream& stream_b,
    const std::array<std::int64_t, 8>& block_a,
    const std::array<std::int64_t, 8>& block_b,
    bool overlap, bool b_first) {
    a.begin_transaction(stream_a.value);
    b.begin_transaction(stream_b.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 transaction A ready");
    cuda_check(cudaStreamSynchronize(stream_b.value),
               "target-graph-c2 transaction B ready");
    const auto launch = [](Exl3TextContext& context, TargetGraphC2Stream& stream,
                           const std::array<std::int64_t, 8>& block) {
        cuda_check(cudaEventRecord(stream.begin, stream.value),
                   "target-graph-c2 record begin");
        context.continue_rows_graph(block, stream.value);
        cuda_check(cudaEventRecord(stream.end, stream.value),
                   "target-graph-c2 record end");
    };
    const auto started = std::chrono::steady_clock::now();
    if (overlap) {
        if (b_first) {
            launch(b, stream_b, block_b);
            launch(a, stream_a, block_a);
        } else {
            launch(a, stream_a, block_a);
            launch(b, stream_b, block_b);
        }
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 overlap A complete");
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 overlap B complete");
    } else if (b_first) {
        launch(b, stream_b, block_b);
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 sequential B complete");
        launch(a, stream_a, block_a);
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 sequential A complete");
    } else {
        launch(a, stream_a, block_a);
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 sequential A complete");
        launch(b, stream_b, block_b);
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 sequential B complete");
    }
    TargetGraphC2ArmTiming result;
    result.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    float event_a_ms = 0.0F, event_b_ms = 0.0F;
    cuda_check(cudaEventElapsedTime(&event_a_ms, stream_a.begin, stream_a.end),
               "target-graph-c2 elapsed A");
    cuda_check(cudaEventElapsedTime(&event_b_ms, stream_b.begin, stream_b.end),
               "target-graph-c2 elapsed B");
    result.event_a_ms = event_a_ms;
    result.event_b_ms = event_b_ms;
    return result;
}

void target_graph_c2_rollback_pair(Exl3TextContext& a, Exl3TextContext& b,
                                   TargetGraphC2Stream& stream_a,
                                   TargetGraphC2Stream& stream_b) {
    a.rollback_transaction(stream_a.value);
    b.rollback_transaction(stream_b.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 rollback A complete");
    cuda_check(cudaStreamSynchronize(stream_b.value),
               "target-graph-c2 rollback B complete");
}

struct TargetGraphC2FixtureResult {
    std::string fixture;
    double median_gain_pct = 0.0;
    double worst_gain_pct = 0.0;
    double max_event_ratio = 0.0;
    bool gate = false;
};

double target_graph_c2_standard_median(std::vector<double> values) {
    require(!values.empty(), "target-graph-c2 median of empty values");
    std::sort(values.begin(), values.end());
    const std::size_t upper = values.size() / 2;
    return (values.size() & 1U) != 0U
        ? values[upper]
        : 0.5 * (values[upper - 1] + values[upper]);
}

TargetGraphC2FixtureResult run_target_graph_c2_fixture(
    Exl3TextModel& target, const std::vector<std::int64_t>& source,
    const std::string& fixture, std::ofstream& exactness,
    std::ofstream& timing) {
    constexpr int prefix_rows = 4096;
    const auto prefix_a = target_graph_c2_shifted_prefix(source, prefix_rows, 0);
    const auto prefix_b = target_graph_c2_shifted_prefix(source, prefix_rows, 17);
    require(prefix_a != prefix_b, fixture + " target-graph-c2 prefixes are not distinct");
    auto a = target.create_context(true);
    auto b = target.create_context(true);
    require(a->try_enable_oscar_from_environment() &&
                b->try_enable_oscar_from_environment(),
            fixture + " target-graph-c2 OSCAR enable failed");
    a->prepare_transaction();
    b->prepare_transaction();
    a->prepare_continuation(8);
    b->prepare_continuation(8);
    ingest_prefix(*a, prefix_a, CommitSink{});
    ingest_prefix(*b, prefix_b, CommitSink{});

    TargetGraphC2Stream stream_a;
    TargetGraphC2Stream stream_b;
    require(stream_a.value != stream_b.value,
            fixture + " target-graph-c2 origin streams are not distinct");
    const auto block_a = target_graph_c2_block(*a, source, 31, stream_a.value);
    const auto block_b = target_graph_c2_block(*b, source, 79, stream_b.value);
    const auto root_a = target_graph_c2_snapshot(*a, stream_a.value, false);
    const auto root_b = target_graph_c2_snapshot(*b, stream_b.value, false);

    const auto eager_reference = [](Exl3TextContext& context,
                                    TargetGraphC2Stream& stream,
                                    const std::array<std::int64_t, 8>& block) {
        context.begin_transaction(stream.value);
        context.continue_rows(block, stream.value);
        cuda_check(cudaStreamSynchronize(stream.value),
                   "target-graph-c2 eager reference complete");
        auto result = target_graph_c2_snapshot(context, stream.value, true);
        context.rollback_transaction(stream.value);
        cuda_check(cudaStreamSynchronize(stream.value),
                   "target-graph-c2 eager reference rollback complete");
        return result;
    };
    const auto eager_a = eager_reference(*a, stream_a, block_a);
    const auto eager_b = eager_reference(*b, stream_b, block_b);
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*a, stream_a.value, false), root_a, false,
        fixture + " eager A rollback");
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*b, stream_b.value, false), root_b, false,
        fixture + " eager B rollback");

    cuda_check(cudaDeviceSynchronize(), "target-graph-c2 pre-capture synchronize");
    require(a->capture_continuation_graph(stream_a.value),
            fixture + " target-graph-c2 capture A failed: " +
                a->continuation_graph_status());
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 capture A complete");
    require(b->capture_continuation_graph(stream_b.value),
            fixture + " target-graph-c2 capture B failed: " +
                b->continuation_graph_status());
    cuda_check(cudaStreamSynchronize(stream_b.value),
               "target-graph-c2 capture B complete");
    constexpr std::uint32_t required_routes =
        ninfer::exl3::Exl3ContinuationGraphRouteFixedB8 |
        ninfer::exl3::Exl3ContinuationGraphRouteChronologicalOscar |
        ninfer::exl3::Exl3ContinuationGraphRouteGdnQkvzConcurrent;
    const auto admission_a = a->continuation_graph_admission();
    const auto admission_b = b->continuation_graph_admission();
    require(admission_a.active && admission_b.active &&
                admission_a.position_ready && admission_b.position_ready &&
                admission_a.capacity == 8 && admission_b.capacity == 8 &&
                admission_a.oscar_split_class == admission_b.oscar_split_class &&
                admission_a.graph_origin_stream_identity ==
                    reinterpret_cast<std::uintptr_t>(stream_a.value) &&
                admission_b.graph_origin_stream_identity ==
                    reinterpret_cast<std::uintptr_t>(stream_b.value) &&
                admission_a.route_generation > 0 &&
                admission_b.route_generation > 0 &&
                (admission_a.qualified_route_bits & required_routes) == required_routes &&
                (admission_b.qualified_route_bits & required_routes) == required_routes &&
                !admission_a.ready && !admission_b.ready,
            fixture + " target-graph-c2 typed admission mismatch");
    require(a->capture_continuation_graph(stream_a.value) &&
                b->capture_continuation_graph(stream_b.value) &&
                a->continuation_graph_admission().route_generation ==
                    admission_a.route_generation &&
                b->continuation_graph_admission().route_generation ==
                    admission_b.route_generation,
            fixture + " target-graph-c2 same-class reuse changed generation");

    for (int exact_case = 0; exact_case < 3; ++exact_case) {
        const bool b_first = exact_case == 1;
        a->begin_transaction(stream_a.value);
        b->begin_transaction(stream_b.value);
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 exact transaction A ready");
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 exact transaction B ready");
        require(a->continuation_graph_admission().ready &&
                    b->continuation_graph_admission().ready,
                fixture + " target-graph-c2 transaction admission not ready");
        if (b_first) {
            b->continue_rows_graph(block_b, stream_b.value);
            a->continue_rows_graph(block_a, stream_a.value);
        } else {
            a->continue_rows_graph(block_a, stream_a.value);
            b->continue_rows_graph(block_b, stream_b.value);
        }
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 exact A complete");
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 exact B complete");
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*a, stream_a.value, true), eager_a, true,
            fixture + " graph A exact case=" + std::to_string(exact_case));
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*b, stream_b.value, true), eager_b, true,
            fixture + " graph B exact case=" + std::to_string(exact_case));
        target_graph_c2_rollback_pair(*a, *b, stream_a, stream_b);
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*a, stream_a.value, false), root_a, false,
            fixture + " graph A rollback case=" + std::to_string(exact_case));
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*b, stream_b.value, false), root_b, false,
            fixture + " graph B rollback case=" + std::to_string(exact_case));
        exactness << fixture << ',' << exact_case << ','
                  << (b_first ? "BA" : "AB") << ",PASS,PASS,PASS\n";
        exactness.flush();
    }
    for(auto* context:{a.get(),b.get()}) {
        const auto graph=context->continuation_graph_stats();
        const auto graph_domain=static_cast<unsigned>(
            ninfer::exl3::Exl3ResourceInventory::Domain::graph_count);
        require(graph.preparation_attempts==2 &&
                    graph.preparation_successes==1 &&
                    graph.compatible_active_hits==1 &&
                    graph.compatible_reactivations==0 &&
                    graph.eager_fallbacks==0 &&
                    graph.replay_submissions==3 &&
                    graph.final_use_observations==3 &&
                    graph.destruction_attempts==0 &&
                    graph.live_definitions==1 && graph.live_executables==1 &&
                    graph.bound_resource_owners_by_shape[2]>0 &&
                    graph.bound_external_bytes_by_shape[2]>0 &&
                    graph.retained_units_by_shape[2][graph_domain]==2 &&
                    graph.reserved_peak_units[graph_domain]==0 &&
                    !graph.driver_memory_bytes_known &&
                    graph.driver_memory_bytes==0,
                fixture + " target-graph-c2 phase accounting mismatch");
    }

    auto run_pair = [&](const std::string& phase, int pair) {
        const bool b_first = (pair & 1) != 0;
        // R2 independently crosses arm order with launch order. Consecutive
        // four-pair blocks cover SO/AB, SO/BA, OS/AB, OS/BA exactly once.
        const bool overlap_first = ((pair / 2) & 1) != 0;
        TargetGraphC2ArmTiming sequential;
        TargetGraphC2ArmTiming overlap;
        if (overlap_first) {
            overlap = target_graph_c2_timed_arm(
                *a, *b, stream_a, stream_b, block_a, block_b, true, b_first);
            target_graph_c2_rollback_pair(*a, *b, stream_a, stream_b);
            sequential = target_graph_c2_timed_arm(
                *a, *b, stream_a, stream_b, block_a, block_b, false, b_first);
            target_graph_c2_rollback_pair(*a, *b, stream_a, stream_b);
        } else {
            sequential = target_graph_c2_timed_arm(
                *a, *b, stream_a, stream_b, block_a, block_b, false, b_first);
            target_graph_c2_rollback_pair(*a, *b, stream_a, stream_b);
            overlap = target_graph_c2_timed_arm(
                *a, *b, stream_a, stream_b, block_a, block_b, true, b_first);
            target_graph_c2_rollback_pair(*a, *b, stream_a, stream_b);
        }
        const double gain = 100.0 *
            (sequential.wall_ms - overlap.wall_ms) / sequential.wall_ms;
        require(sequential.event_a_ms > 0.0 && sequential.event_b_ms > 0.0,
                fixture + " target-graph-c2 zero sequential event duration");
        const double ratio_a = overlap.event_a_ms / sequential.event_a_ms;
        const double ratio_b = overlap.event_b_ms / sequential.event_b_ms;
        timing << phase << ',' << fixture << ',' << pair << ','
               << (overlap_first ? "OS" : "SO") << ','
               << (b_first ? "BA" : "AB") << ','
               << std::setprecision(12) << sequential.wall_ms << ','
               << overlap.wall_ms << ',' << sequential.event_a_ms << ','
               << sequential.event_b_ms << ',' << overlap.event_a_ms << ','
               << overlap.event_b_ms << ',' << gain << ',' << ratio_a << ','
               << ratio_b << '\n';
        timing.flush();
        return std::array<double, 3>{gain, ratio_a, ratio_b};
    };
    for (int warmup = 0; warmup < 2; ++warmup)
        (void)run_pair("warmup", warmup);
    std::vector<double> gains;
    double max_event_ratio = 0.0;
    for (int pair = 0; pair < 8; ++pair) {
        const auto values = run_pair("measured", pair);
        gains.push_back(values[0]);
        max_event_ratio = std::max(max_event_ratio,
                                   std::max(values[1], values[2]));
    }
    TargetGraphC2FixtureResult result;
    result.fixture = fixture;
    result.median_gain_pct = target_graph_c2_standard_median(gains);
    result.worst_gain_pct = *std::min_element(gains.begin(), gains.end());
    result.max_event_ratio = max_event_ratio;
    result.gate = result.median_gain_pct >= 10.0 &&
                  result.worst_gain_pct >= 0.0 &&
                  result.max_event_ratio <= 1.5;
    return result;
}

TargetGraphC2Snapshot target_graph_c2_lifecycle_authority(
    Exl3TextContext& context, TargetGraphC2Stream& stream,
    const std::array<std::int64_t, 8>& block, int rows) {
    require(rows >= 1 && rows <= 8, "target-graph-c2 lifecycle authority extent");
    context.begin_transaction(stream.value);
    if (rows == 1)
        context.decode(block[0], stream.value);
    else
        context.continue_rows(
            std::span<const std::int64_t>(block.data(), static_cast<std::size_t>(rows)),
            stream.value);
    cuda_check(cudaStreamSynchronize(stream.value),
               "target-graph-c2 lifecycle authority complete");
    const auto result = target_graph_c2_snapshot(context, stream.value, false);
    context.rollback_transaction(stream.value);
    cuda_check(cudaStreamSynchronize(stream.value),
               "target-graph-c2 lifecycle authority rollback");
    return result;
}

std::array<std::int64_t, 8> target_graph_c2_lifecycle_greedy_block(
    Exl3TextContext& context, TargetGraphC2Stream& stream) {
    std::array<std::int64_t, 8> block{};
    context.begin_transaction(stream.value);
    for (auto& token : block) {
        token = sample_target(context, stream.value);
        context.decode(token, stream.value);
    }
    cuda_check(cudaStreamSynchronize(stream.value),
               "target-graph-c2 lifecycle greedy block complete");
    context.rollback_transaction(stream.value);
    cuda_check(cudaStreamSynchronize(stream.value),
               "target-graph-c2 lifecycle greedy block rollback");
    return block;
}

void run_target_graph_c2_lifecycle_fixture(
    Exl3TextModel& target, const std::vector<std::int64_t>& source,
    const std::string& fixture, std::ofstream& lifecycle) {
    constexpr int prefix_rows = 512;
    constexpr std::uint32_t required_routes =
        ninfer::exl3::Exl3ContinuationGraphRouteFixedB8 |
        ninfer::exl3::Exl3ContinuationGraphRouteChronologicalOscar |
        ninfer::exl3::Exl3ContinuationGraphRouteGdnQkvzConcurrent;
    require(target.max_context() >= prefix_rows + 96,
            "target-graph-c2 lifecycle context extent");

    const auto prefix_a = target_graph_c2_shifted_prefix(source, prefix_rows, 0);
    const auto prefix_b = target_graph_c2_shifted_prefix(source, prefix_rows, 17);
    require(prefix_a != prefix_b, "target-graph-c2 lifecycle prefixes must differ");
    auto a = target.create_context(true);
    auto b = target.create_context(true);
    const std::string reuse_contract="target-graph-c2;maxctx="+
        std::to_string(target.max_context())+
        ";capture_taps=1;oscar=1;transaction=1;continuation=8;stream=A";
    a->bind_request_compatibility(reuse_contract);
    require(a->try_enable_oscar_from_environment() &&
                b->try_enable_oscar_from_environment(),
            "target-graph-c2 lifecycle OSCAR attachment");
    a->prepare_transaction();
    b->prepare_transaction();
    a->prepare_continuation(8);
    b->prepare_continuation(8);
    ingest_prefix(*a, prefix_a, CommitSink{});
    ingest_prefix(*b, prefix_b, CommitSink{});
    TargetGraphC2Stream stream_a;
    TargetGraphC2Stream stream_b;

    const auto require_idle_admission = [&](Exl3TextContext& context,
                                             TargetGraphC2Stream& stream,
                                             const std::string& label) {
        const auto admission = context.continuation_graph_admission();
        require(admission.active && admission.position_ready && !admission.ready &&
                    admission.capacity == 8 && admission.route_generation > 0 &&
                    admission.graph_origin_stream_identity ==
                        reinterpret_cast<std::uintptr_t>(stream.value) &&
                    (admission.qualified_route_bits & required_routes) == required_routes,
                label + " idle admission mismatch");
        return admission;
    };
    const auto capture_pair = [&] {
        require(a->capture_continuation_graph(stream_a.value),
                "target-graph-c2 lifecycle capture A");
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 lifecycle capture A complete");
        require(b->capture_continuation_graph(stream_b.value),
                "target-graph-c2 lifecycle capture B");
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 lifecycle capture B complete");
        const auto admission_a = require_idle_admission(*a, stream_a, "lifecycle A");
        const auto admission_b = require_idle_admission(*b, stream_b, "lifecycle B");
        require(admission_a.oscar_split_class == admission_b.oscar_split_class,
                "target-graph-c2 lifecycle heterogeneous split class");
    };
    const auto begin_pair = [&] {
        a->begin_transaction(stream_a.value);
        try {
            b->begin_transaction(stream_b.value);
        } catch (...) {
            cuda_check(cudaStreamSynchronize(stream_a.value),
                       "target-graph-c2 lifecycle failed pair begin A fence");
            a->rollback_transaction(stream_a.value);
            cuda_check(cudaStreamSynchronize(stream_a.value),
                       "target-graph-c2 lifecycle failed pair begin A rollback");
            throw;
        }
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 lifecycle transaction A ready");
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 lifecycle transaction B ready");
        require(a->continuation_graph_admission().ready &&
                    b->continuation_graph_admission().ready,
                "target-graph-c2 lifecycle pair not launch-ready");
    };
    const auto launch_pair = [&](const std::array<std::int64_t, 8>& block_a,
                                  const std::array<std::int64_t, 8>& block_b,
                                  bool b_first) {
        if (b_first) {
            b->continue_rows_graph(block_b, stream_b.value);
            a->continue_rows_graph(block_a, stream_a.value);
        } else {
            a->continue_rows_graph(block_a, stream_a.value);
            b->continue_rows_graph(block_b, stream_b.value);
        }
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 lifecycle graph A complete");
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 lifecycle graph B complete");
    };
    const auto require_decision = [&](Exl3TextContext& context,
                                       TargetGraphC2Stream& stream,
                                       const std::array<std::int64_t, 8>& block,
                                       int expected_accepted,
                                       int expected_rejection,
                                       const std::string& label) {
        const auto logits = context.continuation_logits_bits_host(stream.value);
        int accepted = 0;
        while (accepted < 7 &&
               block[static_cast<std::size_t>(accepted + 1)] ==
                   transaction_row_argmax(logits, accepted))
            ++accepted;
        const int rejection = accepted == 7 ? -1 : accepted;
        require(accepted == expected_accepted && rejection == expected_rejection,
                label + " decision mismatch");
    };
    const auto require_roots = [&](const TargetGraphC2Snapshot& root_a,
                                    const TargetGraphC2Snapshot& root_b,
                                    const std::string& label) {
        require(!a->transaction_active() && !b->transaction_active() &&
                    a->continuation_rows() == 0 && b->continuation_rows() == 0,
                label + " stale transaction/publication");
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*a, stream_a.value, false), root_a, false,
            label + " root A");
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*b, stream_b.value, false), root_b, false,
            label + " root B");
    };
    const auto write_case = [&](const std::string& name, const char* lane,
                                 const char* order, const char* outcome,
                                 int accepted, int rejection_index, int retained_rows,
                                 int publication_count, int published_rows) {
        lifecycle << fixture << ',' << name << ',' << lane << ',' << order << ',' << outcome
                  << ',' << accepted << ',' << rejection_index << ',' << retained_rows
                  << ',' << publication_count << ',' << published_rows
                  << ",1,1,1,1,1,1,1\n";
        lifecycle.flush();
    };

    const auto initial_a = a->continuation_graph_admission();
    const auto initial_b = b->continuation_graph_admission();
    require(!initial_a.active && !initial_b.active &&
                initial_a.route_generation == 0 && initial_b.route_generation == 0 &&
                initial_a.graph_origin_stream_identity == 0 &&
                initial_b.graph_origin_stream_identity == 0,
            "target-graph-c2 lifecycle initial admission not inactive");
    capture_pair();
    const auto generation_a = a->continuation_graph_admission().route_generation;
    const auto generation_b = b->continuation_graph_admission().route_generation;
    capture_pair();
    require(a->continuation_graph_admission().route_generation == generation_a &&
                b->continuation_graph_admission().route_generation == generation_b,
            "target-graph-c2 lifecycle same-origin reuse changed generation");
    {
        TargetGraphC2Stream alternate_stream_a;
        require(a->capture_continuation_graph(alternate_stream_a.value),
                "target-graph-c2 lifecycle alternate-origin recapture A");
        cuda_check(cudaStreamSynchronize(alternate_stream_a.value),
                   "target-graph-c2 lifecycle alternate-origin recapture complete");
        const auto alternate = a->continuation_graph_admission();
        require(alternate.route_generation > generation_a &&
                    alternate.graph_origin_stream_identity ==
                        reinterpret_cast<std::uintptr_t>(alternate_stream_a.value),
                "target-graph-c2 lifecycle alternate origin was reused stale");
        require(a->capture_continuation_graph(stream_a.value),
                "target-graph-c2 lifecycle restore origin recapture A");
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 lifecycle restore origin recapture complete");
        require(a->continuation_graph_admission().route_generation >
                    alternate.route_generation,
                "target-graph-c2 lifecycle restored origin did not recapture");
    }

    auto block_a = target_graph_c2_lifecycle_greedy_block(*a, stream_a);
    auto block_b = target_graph_c2_lifecycle_greedy_block(*b, stream_b);
    auto eager_full_a = [&] {
        a->begin_transaction(stream_a.value);
        a->continue_rows(block_a, stream_a.value);
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 lifecycle eager full A");
        const auto result = target_graph_c2_snapshot(*a, stream_a.value, true);
        a->rollback_transaction(stream_a.value);
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 lifecycle eager full A rollback");
        return result;
    }();
    auto eager_full_b = [&] {
        b->begin_transaction(stream_b.value);
        b->continue_rows(block_b, stream_b.value);
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 lifecycle eager full B");
        const auto result = target_graph_c2_snapshot(*b, stream_b.value, true);
        b->rollback_transaction(stream_b.value);
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 lifecycle eager full B rollback");
        return result;
    }();
    auto root_a = target_graph_c2_snapshot(*a, stream_a.value, false);
    auto root_b = target_graph_c2_snapshot(*b, stream_b.value, false);
    for (int exact_case = 0; exact_case < 3; ++exact_case) {
        const bool b_first = exact_case == 1;
        begin_pair();
        launch_pair(block_a, block_b, b_first);
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*a, stream_a.value, true), eager_full_a, true,
            "target-graph-c2 lifecycle exact A");
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*b, stream_b.value, true), eager_full_b, true,
            "target-graph-c2 lifecycle exact B");
        require_decision(*a, stream_a, block_a, 7, -1,
                         "target-graph-c2 lifecycle exact decision A");
        require_decision(*b, stream_b, block_b, 7, -1,
                         "target-graph-c2 lifecycle exact decision B");
        target_graph_c2_rollback_pair(*a, *b, stream_a, stream_b);
        require_roots(root_a, root_b, "target-graph-c2 lifecycle exact rollback");
        const char* case_name = exact_case == 0 ? "exact_ab" :
            exact_case == 1 ? "exact_ba" : "exact_repeat";
        write_case(case_name, "A", b_first ? "BA" : "AB",
                   "rollback", 7, -1, 0, 0, 0);
        write_case(case_name, "B", b_first ? "BA" : "AB",
                   "rollback", 7, -1, 0, 0, 0);
    }

    // Cancellation before cohort launch: the surviving lane executes as C1;
    // the canceled peer never opens a transaction.
    a->begin_transaction(stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle prelaunch survivor ready");
    require(a->continuation_graph_admission().ready && !b->transaction_active(),
            "target-graph-c2 lifecycle prelaunch cancellation admission");
    a->continue_rows_graph(block_a, stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle prelaunch survivor complete");
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*a, stream_a.value, true), eager_full_a, true,
        "target-graph-c2 lifecycle prelaunch survivor authority");
    require_decision(*a, stream_a, block_a, 7, -1,
                     "target-graph-c2 lifecycle prelaunch survivor");
    a->rollback_transaction(stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle prelaunch survivor rollback");
    require_roots(root_a, root_b, "target-graph-c2 lifecycle cancel before launch");
    write_case("cancel_before", "A", "A", "c1_rollback", 7, -1, 0, 0, 0);
    write_case("cancel_before", "B", "A", "cancel", 0, -1, 0, 0, 0);

    // Cancellation after A submission is cooperative: join submitted work,
    // let A independently retain/commit, and rollback unpublished B.
    begin_pair();
    a->continue_rows_graph(block_a, stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle cancel-after A join");
    cuda_check(cudaStreamSynchronize(stream_b.value),
               "target-graph-c2 lifecycle cancel-after B checkpoint join");
    require_decision(*a, stream_a, block_a, 7, -1,
                     "target-graph-c2 lifecycle cancel-after survivor");
    a->retain_transaction_prefix(8, stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle cancel-after survivor retain");
    a->commit_transaction();
    b->rollback_transaction(stream_b.value);
    cuda_check(cudaStreamSynchronize(stream_b.value),
               "target-graph-c2 lifecycle cancel-after peer rollback");
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*a, stream_a.value, false), eager_full_a, false,
        "target-graph-c2 lifecycle cancel-after survivor authority");
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*b, stream_b.value, false), root_b, false,
        "target-graph-c2 lifecycle cancel-after peer root");
    require(!a->transaction_active() && !b->transaction_active(),
            "target-graph-c2 lifecycle cancel-after transaction remained active");
    write_case("cancel_after_a", "A", "A", "commit", 7, -1, 8, 1, 8);
    write_case("cancel_after_a", "B", "A", "cancel", 0, -1, 0, 0, 0);

    capture_pair();
    block_a = target_graph_c2_lifecycle_greedy_block(*a, stream_a);
    block_b = target_graph_c2_lifecycle_greedy_block(*b, stream_b);
    eager_full_a = target_graph_c2_lifecycle_authority(*a, stream_a, block_a, 8);
    eager_full_b = target_graph_c2_lifecycle_authority(*b, stream_b, block_b, 8);
    root_a = target_graph_c2_snapshot(*a, stream_a.value, false);
    root_b = target_graph_c2_snapshot(*b, stream_b.value, false);

    for (const bool after_both : {false, true}) {
        begin_pair();
        std::string failure;
        try {
            a->continue_rows_graph(block_a, stream_a.value);
            if (after_both) b->continue_rows_graph(block_b, stream_b.value);
            throw std::runtime_error(after_both
                ? "injected caller failure after both submissions"
                : "injected caller failure after A submission");
        } catch (const std::exception& error) {
            failure = error.what();
        }
        require(failure.find("injected caller failure") != std::string::npos,
                "target-graph-c2 lifecycle injected caller failure absent");
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 lifecycle injected A join");
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 lifecycle injected B join");
        require_decision(*a, stream_a, block_a, 7, -1,
                         "target-graph-c2 lifecycle injected decision A");
        if (after_both)
            require_decision(*b, stream_b, block_b, 7, -1,
                             "target-graph-c2 lifecycle injected decision B");
        target_graph_c2_rollback_pair(*a, *b, stream_a, stream_b);
        require_roots(root_a, root_b, after_both
            ? "target-graph-c2 lifecycle failure after both"
            : "target-graph-c2 lifecycle failure after A");
        const char* case_name = after_both ? "failure_after_both" : "failure_after_a";
        write_case(case_name, "A", after_both ? "AB" : "A",
                   "exception", 7, -1, 0, 0, 0);
        write_case(case_name, "B", after_both ? "AB" : "A",
                   "exception", after_both ? 7 : 0, -1, 0, 0, 0);
    }

    // Full acceptance publishes only after both independent commits.
    begin_pair();
    launch_pair(block_a, block_b, false);
    require_decision(*a, stream_a, block_a, 7, -1,
                     "target-graph-c2 lifecycle full accept A");
    require_decision(*b, stream_b, block_b, 7, -1,
                     "target-graph-c2 lifecycle full accept B");
    a->retain_transaction_prefix(8, stream_a.value);
    b->retain_transaction_prefix(8, stream_b.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle full retain A");
    cuda_check(cudaStreamSynchronize(stream_b.value),
               "target-graph-c2 lifecycle full retain B");
    a->commit_transaction();
    b->commit_transaction();
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*a, stream_a.value, false), eager_full_a, false,
        "target-graph-c2 lifecycle full commit A");
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*b, stream_b.value, false), eager_full_b, false,
        "target-graph-c2 lifecycle full commit B");
    write_case("full_accept", "A", "AB", "commit", 7, -1, 8, 1, 8);
    write_case("full_accept", "B", "AB", "commit", 7, -1, 8, 1, 8);

    // One lane's deterministic retained-prefix reconstruction failure cannot
    // revoke the peer's independently retained and committed authority.
    capture_pair();
    block_a = target_graph_c2_lifecycle_greedy_block(*a, stream_a);
    block_b = target_graph_c2_lifecycle_greedy_block(*b, stream_b);
    block_a[4] = (block_a[4] + 1) % kVocab;
    if (block_a[4] == kMaskToken) block_a[4] = (block_a[4] + 1) % kVocab;
    block_b[4] = (block_b[4] + 1) % kVocab;
    if (block_b[4] == kMaskToken) block_b[4] = (block_b[4] + 1) % kVocab;
    const auto retain_failure_authority_b = target_graph_c2_lifecycle_authority(
        *b, stream_b, block_b, 4);
    const auto retain_failure_root_a = target_graph_c2_snapshot(
        *a, stream_a.value, false);
    begin_pair();
    launch_pair(block_a, block_b, false);
    require_decision(*a, stream_a, block_a, 3, 3,
                     "target-graph-c2 lifecycle retain-failure A");
    require_decision(*b, stream_b, block_b, 3, 3,
                     "target-graph-c2 lifecycle retain-failure B");
    std::string retain_failure;
    try {
        a->retain_transaction_prefix_for_test(4, 31, stream_a.value);
    } catch (const std::exception& error) {
        retain_failure = error.what();
    }
    require(retain_failure.find("injected retained-prefix failure") !=
                std::string::npos,
            "target-graph-c2 lifecycle retained-prefix failure absent");
    b->retain_transaction_prefix(4, stream_b.value);
    cuda_check(cudaStreamSynchronize(stream_b.value),
               "target-graph-c2 lifecycle retain-failure peer retain");
    b->commit_transaction();
    a->rollback_transaction(stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle retain-failure rollback");
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*a, stream_a.value, false),
        retain_failure_root_a, false,
        "target-graph-c2 lifecycle retain-failure root A");
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*b, stream_b.value, false),
        retain_failure_authority_b, false,
        "target-graph-c2 lifecycle retain-failure authority B");
    require(!a->transaction_active() && !b->transaction_active(),
            "target-graph-c2 lifecycle retain-failure transaction remained active");
    write_case("retain_failure", "A", "AB", "rollback", 3, 3, 4, 0, 0);
    write_case("retain_failure", "B", "AB", "commit", 3, 3, 4, 1, 4);

    const std::array<int, 3> accepted_counts{0, 3, 7};
    const std::array<int, 3> rejection_indices{0, 3, -1};
    const std::array<int, 3> retained_counts{1, 4, 8};
    for (std::size_t case_index = 0; case_index < retained_counts.size(); ++case_index) {
        capture_pair();
        block_a = target_graph_c2_lifecycle_greedy_block(*a, stream_a);
        block_b = target_graph_c2_lifecycle_greedy_block(*b, stream_b);
        const int retained = retained_counts[case_index];
        if (rejection_indices[case_index] >= 0) {
            const auto force_mismatch = [](std::int64_t token) {
                auto replacement = (token + 1) % kVocab;
                if (replacement == kMaskToken) replacement = (replacement + 1) % kVocab;
                return replacement;
            };
            block_a[static_cast<std::size_t>(retained)] =
                force_mismatch(block_a[static_cast<std::size_t>(retained)]);
            block_b[static_cast<std::size_t>(retained)] =
                force_mismatch(block_b[static_cast<std::size_t>(retained)]);
        }
        const auto authority_a = target_graph_c2_lifecycle_authority(
            *a, stream_a, block_a, retained);
        const auto authority_b = target_graph_c2_lifecycle_authority(
            *b, stream_b, block_b, retained);
        begin_pair();
        launch_pair(block_a, block_b, (case_index & 1U) != 0U);
        require_decision(*a, stream_a, block_a, accepted_counts[case_index],
                         rejection_indices[case_index],
                         "target-graph-c2 lifecycle retained decision A");
        require_decision(*b, stream_b, block_b, accepted_counts[case_index],
                         rejection_indices[case_index],
                         "target-graph-c2 lifecycle retained decision B");
        a->retain_transaction_prefix(retained, stream_a.value);
        cuda_check(cudaStreamSynchronize(stream_a.value),
                   "target-graph-c2 lifecycle retained lane complete");
        a->commit_transaction();
        b->rollback_transaction(stream_b.value);
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 lifecycle replay lane rollback");
        if (retained == 1)
            b->decode(block_b[0], stream_b.value);
        else
            b->continue_rows(std::span<const std::int64_t>(
                block_b.data(), static_cast<std::size_t>(retained)), stream_b.value);
        cuda_check(cudaStreamSynchronize(stream_b.value),
                   "target-graph-c2 lifecycle replay lane complete");
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*a, stream_a.value, false), authority_a, false,
            "target-graph-c2 lifecycle retained authority A");
        target_graph_c2_require_snapshot_equal(
            target_graph_c2_snapshot(*b, stream_b.value, false), authority_b, false,
            "target-graph-c2 lifecycle replay authority B");
        require(!a->transaction_active() && !b->transaction_active() &&
                    a->continuation_graph_active() && b->continuation_graph_active(),
                "target-graph-c2 lifecycle independent completion state");
        const std::string case_name = "retain_" + std::to_string(retained);
        const char* outcome = rejection_indices[case_index] < 0 ? "full" : "reject";
        write_case(case_name, "A", (case_index & 1U) != 0U ? "BA" : "AB",
                   outcome, accepted_counts[case_index], rejection_indices[case_index],
                   retained, 1, retained);
        write_case(case_name, "B", (case_index & 1U) != 0U ? "BA" : "AB",
                   "rollback_replay", accepted_counts[case_index],
                   rejection_indices[case_index], retained, 1, retained);
    }

    // A mutation-free capture preflight failure invalidates only admission,
    // preserves the monotonic counter, and requires a real recapture.
    capture_pair();
    const auto before_failure = a->continuation_graph_admission().route_generation;
    a->begin_transaction(stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle failure-boundary transaction");
    require(!a->capture_continuation_graph(stream_a.value),
            "target-graph-c2 lifecycle capture failure boundary did not fail");
    const auto failed = a->continuation_graph_admission();
    require(!failed.active && !failed.ready && failed.route_generation == 0 &&
                failed.graph_origin_stream_identity == 0 &&
                failed.qualified_route_bits == 0,
            "target-graph-c2 lifecycle failed capture remained admitted");
    a->rollback_transaction(stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle failure-boundary rollback");
    require(a->capture_continuation_graph(stream_a.value),
            "target-graph-c2 lifecycle recapture after failure");
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle recapture after failure complete");
    require(a->continuation_graph_admission().route_generation > before_failure,
            "target-graph-c2 lifecycle generation did not advance after recapture");

    // Compatible request reset revokes launch admission while retaining the
    // immutable graph key and object. An unrelated same-shape prompt then
    // reactivates without capture or generation change.
    const auto before_reset = a->continuation_graph_admission().route_generation;
    const auto captures_before_reset=a->continuation_graph_admission().capture_count;
    const auto reuses_before_reset=
        a->continuation_graph_admission().compatible_reuse_count;
    const auto request_reset=a->reset_for_request(reuse_contract);
    const auto reset = a->continuation_graph_admission();
    require(reset.capacity == 8 && !reset.active && !reset.ready &&
                reset.route_generation == before_reset &&
                reset.graph_origin_stream_identity ==
                    reinterpret_cast<std::uintptr_t>(stream_a.value) &&
                reset.capture_count==captures_before_reset &&
                reset.compatible_reuse_count==reuses_before_reset &&
                request_reset.generation==1,
            "target-graph-c2 lifecycle reset admission mismatch");
    const auto prefix_after_reset=
        target_graph_c2_shifted_prefix(source,prefix_rows,31);
    require(prefix_after_reset!=prefix_a,
        "target-graph-c2 lifecycle reset prompt must differ");
    ingest_prefix(*a, prefix_after_reset, CommitSink{});
    require(a->capture_continuation_graph(stream_a.value),
            "target-graph-c2 lifecycle reuse after reset");
    cuda_check(cudaStreamSynchronize(stream_a.value),
               "target-graph-c2 lifecycle reuse after reset complete");
    const auto reused_after_reset=a->continuation_graph_admission();
    require(reused_after_reset.route_generation==before_reset &&
                reused_after_reset.capture_count==captures_before_reset &&
                reused_after_reset.compatible_reuse_count==reuses_before_reset+1 &&
                reused_after_reset.graph_origin_stream_identity==
                    reinterpret_cast<std::uintptr_t>(stream_a.value) &&
                reused_after_reset.qualified_route_bits==required_routes &&
                a->continuation_graph_status().find("reused after context reset")!=
                    std::string::npos,
            "target-graph-c2 lifecycle reset rebuilt compatible graph");

    // A stale projection binding cannot take the compatible active-graph path.
    // The seam revokes identity admission but retains all owners for retirement.
    const auto before_binding_change=a->continuation_graph_admission();
    a->invalidate_projection_graph_binding_for_test();
    require(!a->continuation_graph_admission().ready,
        "stale projection binding exposed ready graph");
    const auto binding_root=target_graph_c2_snapshot(*a,stream_a.value,false);
    const auto binding_position=a->position();
    a->begin_transaction(stream_a.value);
    require(!a->continuation_graph_admission().ready,
        "fresh transaction admitted stale projection graph");
    bool binding_rejected=false;
    try {a->continue_rows_graph(block_a,stream_a.value);}
    catch(const std::exception&) {binding_rejected=true;}
    require(binding_rejected && a->position()==binding_position,
        "stale projection graph reached numerical replay");
    a->rollback_transaction(stream_a.value);
    cuda_check(cudaStreamSynchronize(stream_a.value),"stale projection graph rollback");
    target_graph_c2_require_snapshot_equal(
        target_graph_c2_snapshot(*a,stream_a.value,false),binding_root,false,
        "stale projection binding changed authority");
    require(a->capture_continuation_graph(stream_a.value),
        "projection binding invalidation failed to recapture");
    const auto after_binding_change=a->continuation_graph_admission();
    require(after_binding_change.route_generation>before_binding_change.route_generation &&
        after_binding_change.capture_count==before_binding_change.capture_count+1 &&
        after_binding_change.compatible_reuse_count==before_binding_change.compatible_reuse_count,
        "stale projection binding reused cached graph");

    // Mutate one actual dispatch-time option only while both lanes are idle.
    // No changed-arithmetic graph is replayed in this ownership fixture.
    {
        struct RestoreProjectionOption {
            std::string value;
            RestoreProjectionOption():value(std::getenv("NINFER_EXL3_GENERIC_SPLITS")?
                std::getenv("NINFER_EXL3_GENERIC_SPLITS"):""){}
            ~RestoreProjectionOption(){_putenv_s("NINFER_EXL3_GENERIC_SPLITS",value.c_str());}
        } restore;
        const auto previous=a->continuation_graph_admission();
        require(_putenv_s("NINFER_EXL3_GENERIC_SPLITS",restore.value=="1"?"2":"1")==0,
            "graph lifecycle option mutation failed");
        a->begin_transaction(stream_a.value);
        require(!a->continuation_graph_admission().ready,
            "changed projection options retained ready admission");
        bool refused=false;
        try {a->continue_rows_graph(block_a,stream_a.value);}
        catch(const std::exception&) {refused=true;}
        require(refused,"changed projection options reached graph replay");
        a->rollback_transaction(stream_a.value);
        require(a->capture_continuation_graph(stream_a.value),"changed projection option recapture failed");
        const auto changed=a->continuation_graph_admission();
        require(changed.capture_count==previous.capture_count+1 &&
            changed.route_generation>previous.route_generation,
            "changed projection options reused old capture");
        require(_putenv_s("NINFER_EXL3_GENERIC_SPLITS",restore.value.c_str())==0,
            "graph lifecycle option restoration failed");
        require(a->capture_continuation_graph(stream_a.value),"restored projection option recapture failed");
        require(a->continuation_graph_admission().capture_count==changed.capture_count+1,
            "restored arithmetic reused changed capture");
    }

    // Final real replay proves both contexts reusable after lifecycle edges.
    capture_pair();
    block_a = target_graph_c2_lifecycle_greedy_block(*a, stream_a);
    block_b = target_graph_c2_lifecycle_greedy_block(*b, stream_b);
    const auto final_root_a = target_graph_c2_snapshot(*a, stream_a.value, false);
    const auto final_root_b = target_graph_c2_snapshot(*b, stream_b.value, false);
    begin_pair();
    launch_pair(block_a, block_b, false);
    require_decision(*a, stream_a, block_a, 7, -1,
                     "target-graph-c2 lifecycle final reuse decision A");
    require_decision(*b, stream_b, block_b, 7, -1,
                     "target-graph-c2 lifecycle final reuse decision B");
    target_graph_c2_rollback_pair(*a, *b, stream_a, stream_b);
    require_roots(final_root_a, final_root_b, "target-graph-c2 lifecycle final reuse");
    write_case("final_reuse", "A", "AB", "rollback", 7, -1, 0, 0, 0);
    write_case("final_reuse", "B", "AB", "rollback", 7, -1, 0, 0, 0);
    std::cout << "TARGET_GRAPH_C2_LIFECYCLE_FIXTURE PASS fixture=" << fixture
              << " exact_orders=3 full_accept=1 retain_cases=3 retain_failures=1"
              << " cancellation_cases=2 failure_cases=2 capture_failure=1"
              << " reset_graph_reuse=1 reuse=1\n";
}

void run_target_graph_c2_lifecycle(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    require(!output.empty() && !std::filesystem::exists(output),
            "target-graph-c2 lifecycle output must be new");
    std::filesystem::create_directories(output);
    std::ofstream lifecycle(output / "lifecycle.csv");
    require(lifecycle.good(), "target-graph-c2 lifecycle output creation failed");
    lifecycle << "fixture,case,lane,order,outcome,accepted,rejection_index,"
                 "retained_rows,publication_count,published_rows,authority,root,"
                 "peer,rollback,reuse,txn_closed,graph_active\n";
    run_target_graph_c2_lifecycle_fixture(target, code, "code", lifecycle);
    run_target_graph_c2_lifecycle_fixture(target, prose, "prose", lifecycle);
    lifecycle.flush();
    std::cout << "TARGET_GRAPH_C2_LIFECYCLE PASS fixtures=2 exact_orders=6 "
              << "full_accept=2 retain_cases=6 retain_failures=2"
              << " cancellation_cases=4 failure_cases=4 capture_failures=2"
              << " reset_graph_reuses=2 reuse=2\n";
}

void run_target_graph_reset_reuse_performance(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    constexpr int prefix_rows=512;
    require(target.max_context()>=prefix_rows+8 && code.size()>=prefix_rows &&
        prose.size()>=prefix_rows,
        "target graph reset reuse performance extent");
    require(env("NINFER_OSCAR_EXL3")=="1" &&
        env("NINFER_E5A4_CONTINUATION_GRAPH_B8")=="1" &&
        env("NINFER_OSCAR_CONTINUATION_COHORT_B8")=="1" &&
        env("NINFER_EXL3_CONTINUATION_GRAPH_GDN_QKVZ_CONCURRENT")=="1",
        "target graph reset reuse performance qualified route");
    auto context=target.create_context(true);
    const std::string contract="target-graph-reset-reuse;maxctx="+
        std::to_string(target.max_context())+
        ";capture_taps=1;oscar=1;transaction=1;continuation=8;stream=stable";
    context->bind_request_compatibility(contract);
    require(context->try_enable_oscar_from_environment(),
        "target graph reset reuse OSCAR attachment");
    context->prepare_transaction();context->prepare_continuation(8);
    TargetGraphC2Stream stream;
    const auto warm=target_graph_c2_shifted_prefix(code,prefix_rows,3);
    ingest_prefix(*context,warm,CommitSink{});
    require(context->capture_continuation_graph(stream.value),
        "target graph reset reuse warm capture");
    cuda_check(cudaStreamSynchronize(stream.value),
        "target graph reset reuse warm capture complete");
    const auto persistent=context->persistent_bytes();
    struct Outcome {
        double total_ms=0,capture_ms=0;
        TargetGraphC2Snapshot snapshot;
        std::uint64_t generation=0,captures=0,reuses=0;
    };
    const auto run=[&](const std::vector<std::int64_t>& source,bool reuse,int shift) {
        const auto begin=std::chrono::steady_clock::now();
        context->reset_for_request(contract);
        if(!reuse) context->discard_continuation_graph_for_test();
        const auto prefix=target_graph_c2_shifted_prefix(source,prefix_rows,shift);
        ingest_prefix(*context,prefix,CommitSink{});
        const auto capture_begin=std::chrono::steady_clock::now();
        require(context->capture_continuation_graph(stream.value),
            "target graph reset reuse capture/reactivate");
        cuda_check(cudaStreamSynchronize(stream.value),
            "target graph reset reuse setup complete");
        const auto capture_end=std::chrono::steady_clock::now();
        const auto block=target_graph_c2_block(*context,source,shift+37,stream.value);
        context->begin_transaction(stream.value);
        cuda_check(cudaStreamSynchronize(stream.value),
            "target graph reset reuse transaction ready");
        context->continue_rows_graph(block,stream.value);
        cuda_check(cudaStreamSynchronize(stream.value),
            "target graph reset reuse graph complete");
        const auto complete=std::chrono::steady_clock::now();
        const auto admission=context->continuation_graph_admission();
        Outcome result;
        result.total_ms=std::chrono::duration<double,std::milli>(complete-begin).count();
        result.capture_ms=std::chrono::duration<double,std::milli>(
            capture_end-capture_begin).count();
        result.snapshot=target_graph_c2_snapshot(*context,stream.value,true);
        result.generation=admission.route_generation;
        result.captures=admission.capture_count;
        result.reuses=admission.compatible_reuse_count;
        context->rollback_transaction(stream.value);
        cuda_check(cudaStreamSynchronize(stream.value),
            "target graph reset reuse rollback");
        require(context->persistent_bytes()==persistent,
            "target graph reset reuse persistent allocation drift");
        return result;
    };
    std::vector<double> all_gains;
    for(const auto& fixture:std::array<
        std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        std::vector<double> gains,reuse_capture_ms,recapture_ms;
        for(int rep=0;rep<3;++rep) {
            Outcome reused,control;
            const bool control_first=rep==1;
            if(control_first) {
                control=run(*fixture.second,false,rep*11+1);
                reused=run(*fixture.second,true,rep*11+1);
            } else {
                reused=run(*fixture.second,true,rep*11+1);
                control=run(*fixture.second,false,rep*11+1);
            }
            target_graph_c2_require_snapshot_equal(reused.snapshot,control.snapshot,
                true,std::string("target graph reset reuse exactness ")+fixture.first);
            require(reused.reuses>0 && control.captures>0,
                "target graph reset reuse counters");
            const double gain=(control.total_ms-reused.total_ms)*100.0/control.total_ms;
            gains.push_back(gain);all_gains.push_back(gain);
            reuse_capture_ms.push_back(reused.capture_ms);
            recapture_ms.push_back(control.capture_ms);
            std::cout<<"TARGET_GRAPH_RESET_REUSE_PAIR fixture="<<fixture.first
                <<" rep="<<rep<<" order="<<(control_first?"CR":"RC")
                <<" reused_total_ms="<<reused.total_ms
                <<" control_total_ms="<<control.total_ms
                <<" gain_percent="<<gain
                <<" reuse_setup_ms="<<reused.capture_ms
                <<" recapture_setup_ms="<<control.capture_ms
                <<" reused_generation="<<reused.generation
                <<" control_generation="<<control.generation<<std::endl;
        }
        std::cout<<"TARGET_GRAPH_RESET_REUSE_FIXTURE fixture="<<fixture.first
            <<" median_gain_percent="<<target_graph_c2_standard_median(gains)
            <<" median_reuse_setup_ms="
            <<target_graph_c2_standard_median(reuse_capture_ms)
            <<" median_recapture_setup_ms="
            <<target_graph_c2_standard_median(recapture_ms)<<std::endl;
    }
    std::cout<<"TARGET_GRAPH_RESET_REUSE PASS fixtures=2 pairs=6 exact_state=1"
        <<" median_gain_percent="<<target_graph_c2_standard_median(all_gains)
        <<" positive_pairs="<<std::count_if(all_gains.begin(),all_gains.end(),
            [](double value){return value>0;})
        <<" persistent_bytes="<<persistent<<std::endl;
}

void run_target_graph_c2_screen(Exl3TextModel& target,
                                const std::vector<std::int64_t>& code,
                                const std::vector<std::int64_t>& prose,
                                const std::filesystem::path& output) {
    require(target.max_context() >= 4352,
            "target-graph-c2 requires max context at least 4352");
    require(env("NINFER_OSCAR_EXL3") == "1" &&
                env("NINFER_E5A4_CONTINUATION_GRAPH_B8") == "1" &&
                env("NINFER_OSCAR_CONTINUATION_COHORT_B8") == "1" &&
                env("NINFER_EXL3_CONTINUATION_GRAPH_GDN_QKVZ_CONCURRENT") == "1",
            "target-graph-c2 requires qualified B8 OSCAR GDN-QKVZ graph flags");
    require(env("NINFER_EXL3_TARGET_PROJECTION_TIMING").empty() &&
                env("NINFER_EXL3_TARGET_PROJECTION_TIMING_OUT").empty(),
            "target-graph-c2 rejects projection timing instrumentation");
    require(!output.empty() && !std::filesystem::exists(output),
            "preserve target-graph-c2 output directory");
    std::filesystem::create_directories(output);
    std::ofstream exactness(output / "exactness.csv");
    std::ofstream timing(output / "timing.csv");
    std::ofstream summary(output / "summary.csv");
    require(exactness.good() && timing.good() && summary.good(),
            "target-graph-c2 output creation failed");
    exactness << "fixture,case,launch_order,live_a,live_b,rollback\n";
    timing << "phase,fixture,pair,arm_order,launch_order,sequential_wall_ms,"
              "overlap_wall_ms,sequential_event_a_ms,sequential_event_b_ms,"
              "overlap_event_a_ms,overlap_event_b_ms,gain_pct,event_ratio_a,"
              "event_ratio_b\n";
    summary << "fixture,pairs,median_gain_pct,worst_gain_pct,max_event_ratio,gate\n";
    const auto code_result = run_target_graph_c2_fixture(
        target, code, "code", exactness, timing);
    const auto prose_result = run_target_graph_c2_fixture(
        target, prose, "prose", exactness, timing);
    bool gate = true;
    for (const auto& result : {code_result, prose_result}) {
        summary << result.fixture << ",8," << std::setprecision(12)
                << result.median_gain_pct << ',' << result.worst_gain_pct << ','
                << result.max_event_ratio << ',' << (result.gate ? 1 : 0) << '\n';
        gate = gate && result.gate;
    }
    summary.flush();
    require(gate, "target-graph-c2 timing gate failed");
    std::cout << "TARGET_GRAPH_C2_SCREEN_DONE exact=1 timing_gate=1 "
              << "code_median_gain_pct=" << code_result.median_gain_pct << ' '
              << "prose_median_gain_pct=" << prose_result.median_gain_pct << '\n';
}
