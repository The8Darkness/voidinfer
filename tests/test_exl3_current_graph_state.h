#pragma once

// Focused current graph lifecycle probe.  This deliberately stops without a
// PASS marker when a graph-class context cannot export its OSCAR live state;
// the public test API documents that export as eager-only, so silently
// dropping OSCAR state would make the differential unsound.

struct GraphCurrentDiagnostics {
    int last_forward_rows = 0;
    int captured_tap_rows = 0;
    int captured_embedding_rows = 0;
    int continuation_rows = 0;
    std::size_t last_decode_h2d = 0;
    bool graph_active = false;
    std::string graph_status;
    std::uint64_t routed_oscar = 0;
    std::uint64_t routed_ordinary = 0;
    ninfer::exl3::Exl3OscarTelemetry telemetry{};
};

struct GraphCurrentSnapshot {
    int position = 0;
    int device_position = 0;
    std::vector<std::uint16_t> logits;
    std::array<std::vector<std::uint16_t>, kTapCount> taps;
    std::vector<std::uint16_t> embedding;
    std::vector<std::byte> gdn;
    std::vector<std::byte> convolution;
    std::vector<std::byte> oscar;
    std::string oscar_export_error;
    GraphCurrentDiagnostics diagnostics;
};

std::uint64_t graph_current_hash(const void* data, std::size_t bytes) {
    const auto* raw = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= raw[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

template <class T>
std::uint64_t graph_current_hash(const std::vector<T>& values) {
    return graph_current_hash(values.data(), values.size() * sizeof(T));
}

GraphCurrentSnapshot graph_current_snapshot(Exl3TextContext& context,
                                            const std::string& label) {
    cuda_check(cudaDeviceSynchronize(), (label + " ordering").c_str());
    GraphCurrentSnapshot result;
    result.position = context.position();
    result.device_position = context.device_position_host();
    result.diagnostics.last_forward_rows = context.last_forward_rows();
    result.diagnostics.captured_tap_rows = context.captured_tap_rows();
    result.diagnostics.captured_embedding_rows = context.captured_embedding_rows();
    result.diagnostics.continuation_rows = context.continuation_rows();
    result.diagnostics.last_decode_h2d = context.last_decode_h2d();
    result.diagnostics.graph_active = context.graph_active();
    result.diagnostics.graph_status = context.graph_status();
    context.oscar_routing_counts(result.diagnostics.routed_oscar,
                                 result.diagnostics.routed_ordinary);
    result.diagnostics.telemetry = context.oscar_telemetry();
    if (result.position > 0 && context.logits_device() != nullptr)
        result.logits = target_continue_device_bits(
            context.logits_device(), kVocab, label + " logits");
    if (result.diagnostics.captured_tap_rows > 0) {
        for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap)
            result.taps[tap] = target_continue_tap_bits(
                context, kTapLayers[tap], result.diagnostics.captured_tap_rows);
    }
    if (result.diagnostics.captured_embedding_rows > 0)
        result.embedding = context.embedding_bits_host_for_test();
    result.gdn.reserve(48ULL * 3145728ULL);
    result.convolution.reserve(48ULL * 81920ULL * sizeof(std::uint16_t));
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        const auto state = context.gdn_state_host(layer);
        const auto* state_bytes = reinterpret_cast<const std::byte*>(state.data());
        result.gdn.insert(result.gdn.end(), state_bytes,
                          state_bytes + state.size() * sizeof(float));
        const auto convolution = context.gdn_physical_conv_host(layer);
        const auto* convolution_bytes =
            reinterpret_cast<const std::byte*>(convolution.data());
        result.convolution.insert(result.convolution.end(), convolution_bytes,
                                  convolution_bytes +
                                      convolution.size() * sizeof(std::uint16_t));
    }
    try {
        result.oscar = context.oscar_live_state_host_for_test();
    } catch (const std::exception& error) {
        result.oscar_export_error = error.what();
    }
    return result;
}

void graph_current_write_diagnostics(std::ostream& out, const std::string& label,
                                     const GraphCurrentDiagnostics& d) {
    out << label << " last_forward_rows=" << d.last_forward_rows
        << " captured_tap_rows=" << d.captured_tap_rows
        << " captured_embedding_rows=" << d.captured_embedding_rows
        << " continuation_rows=" << d.continuation_rows
        << " last_decode_h2d=" << d.last_decode_h2d
        << " graph_active=" << (d.graph_active ? 1 : 0)
        << " graph_status=" << d.graph_status
        << " routed_oscar=" << d.routed_oscar
        << " routed_ordinary=" << d.routed_ordinary
        << " ordinary_dispatches=" << d.telemetry.ordinary_dispatches
        << " gdn_oscar_dispatches=" << d.telemetry.gdn_oscar_dispatches
        << " append_calls=" << d.telemetry.append_calls
        << " aging_events=" << d.telemetry.aging_events
        << " last_split_class=" << d.telemetry.last_split_class
        << " resident_cache_bytes=" << d.telemetry.resident_cache_bytes
        << " workspace_bytes=" << d.telemetry.workspace_bytes
        << " oscar_dispatches=";
    for (std::size_t i = 0; i < d.telemetry.oscar_dispatches.size(); ++i) {
        if (i) out << ',';
        out << d.telemetry.oscar_dispatches[i];
    }
    out << '\n';
}

void graph_current_write_snapshot(std::ostream& out, const std::string& label,
                                  const GraphCurrentSnapshot& s) {
    out << label << " position=" << s.position
        << " device_position=" << s.device_position
        << " logits_bytes=" << s.logits.size() * sizeof(std::uint16_t)
        << " logits_hash=" << graph_current_hash(s.logits)
        << " embedding_bytes=" << s.embedding.size() * sizeof(std::uint16_t)
        << " embedding_hash=" << graph_current_hash(s.embedding)
        << " gdn_bytes=" << s.gdn.size()
        << " gdn_hash=" << graph_current_hash(s.gdn)
        << " convolution_bytes=" << s.convolution.size()
        << " convolution_hash=" << graph_current_hash(s.convolution)
        << " oscar_bytes=" << s.oscar.size()
        << " oscar_hash=" << graph_current_hash(s.oscar)
        << " oscar_exportable=" << (s.oscar_export_error.empty() ? 1 : 0)
        << " oscar_export_error=" << (s.oscar_export_error.empty() ? "none" : s.oscar_export_error)
        << '\n';
    for (std::size_t i = 0; i < s.taps.size(); ++i)
        out << label << " tap=" << i << " bytes="
            << s.taps[i].size() * sizeof(std::uint16_t)
            << " hash=" << graph_current_hash(s.taps[i]) << '\n';
    graph_current_write_diagnostics(out, label + " diagnostics", s.diagnostics);
}

void graph_current_require_observed(const GraphCurrentSnapshot& snapshot,
                                    const std::string& label) {
    require(snapshot.oscar_export_error.empty(),
            label + " exact OSCAR live-state export unavailable: " +
                snapshot.oscar_export_error);
}

void graph_current_require_model_equal(const GraphCurrentSnapshot& actual,
                                       const GraphCurrentSnapshot& expected,
                                       const std::string& label) {
    graph_current_require_observed(actual, label + " actual");
    graph_current_require_observed(expected, label + " expected");
    require(actual.position == expected.position &&
                actual.device_position == expected.device_position,
            label + " position mismatch");
    require(actual.logits == expected.logits, label + " logits mismatch");
    require(actual.taps == expected.taps, label + " tap mismatch");
    require(actual.embedding == expected.embedding, label + " embedding mismatch");
    require(actual.gdn == expected.gdn, label + " GDN mismatch");
    require(actual.convolution == expected.convolution,
            label + " physical convolution mismatch");
    require(actual.oscar == expected.oscar, label + " OSCAR live state mismatch");
}

void graph_current_require_capture_diagnostics(const GraphCurrentSnapshot& before,
                                               const GraphCurrentSnapshot& after,
                                               const std::string& label) {
    // Graph identity/status is the intentional capture transition.  All
    // other exposed metadata must remain unchanged until its contract is
    // explicitly defined; report every value before failing closed.
    require(after.diagnostics.graph_active,
            label + " did not publish graph_active after capture");
    require(before.diagnostics.last_forward_rows == after.diagnostics.last_forward_rows &&
                before.diagnostics.captured_tap_rows == after.diagnostics.captured_tap_rows &&
                before.diagnostics.captured_embedding_rows == after.diagnostics.captured_embedding_rows &&
                before.diagnostics.continuation_rows == after.diagnostics.continuation_rows &&
                before.diagnostics.last_decode_h2d == after.diagnostics.last_decode_h2d &&
                before.diagnostics.routed_oscar == after.diagnostics.routed_oscar &&
                before.diagnostics.routed_ordinary == after.diagnostics.routed_ordinary &&
                before.diagnostics.telemetry.oscar_dispatches == after.diagnostics.telemetry.oscar_dispatches &&
                before.diagnostics.telemetry.ordinary_dispatches == after.diagnostics.telemetry.ordinary_dispatches &&
                before.diagnostics.telemetry.gdn_oscar_dispatches == after.diagnostics.telemetry.gdn_oscar_dispatches &&
                before.diagnostics.telemetry.append_calls == after.diagnostics.telemetry.append_calls &&
                before.diagnostics.telemetry.aging_events == after.diagnostics.telemetry.aging_events &&
                before.diagnostics.telemetry.last_split_class == after.diagnostics.telemetry.last_split_class,
            label + " unexpected capture-side diagnostic delta");
}

void graph_current_ingest(Exl3TextContext& context,
                          const std::vector<std::int64_t>& ids, int count,
                          cudaStream_t stream) {
    require(count >= 16 && static_cast<std::size_t>(count) <= ids.size(),
            "graphcurrentstate prefix extent");
    context.prefill(std::span<const std::int64_t>(ids.data(), 16), stream);
    for (int offset = 16; offset < count;) {
        const int rows = std::min(128, count - offset);
        context.append_prefill_wide(
            std::span<const std::int64_t>(ids.data() + offset, rows), stream);
        offset += rows;
    }
    cuda_check(cudaStreamSynchronize(stream), "graphcurrentstate prefix sync");
}

struct GraphCurrentStreamGuard {
    cudaStream_t stream = nullptr;
    GraphCurrentStreamGuard() {
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "graphcurrentstate create nonblocking stream");
    }
    ~GraphCurrentStreamGuard() { if (stream) cudaStreamDestroy(stream); }
    GraphCurrentStreamGuard(const GraphCurrentStreamGuard&) = delete;
};

struct GraphReplaySnapshot {
    int position = 0;
    int device_position = 0;
    std::vector<std::uint16_t> logits;
    std::vector<std::byte> gdn;
    std::vector<std::byte> convolution;
    std::vector<std::byte> oscar;
    std::array<std::array<int, 5>, 16> counts{};
};

GraphReplaySnapshot graph_replay_snapshot(Exl3TextContext& context,
                                           cudaStream_t stream,
                                           const std::string& label) {
    const auto base = graph_current_snapshot(context, label + " observable");
    const auto observation = context.oscar_graph_state_host_for_test(stream);
    require(observation.valid,
            label + " graph OSCAR observer failed: " + observation.failure);
    require(observation.observed_layers == observation.device_counts.size(),
            label + " graph OSCAR observer layer count");
    GraphReplaySnapshot result;
    result.position = base.position;
    result.device_position = base.device_position;
    result.logits = base.logits;
    result.gdn = base.gdn;
    result.convolution = base.convolution;
    result.oscar = observation.canonical_bytes;
    result.counts = observation.device_counts;
    return result;
}

GraphReplaySnapshot graph_replay_eager_snapshot(Exl3TextContext& context,
                                                const std::string& label) {
    const auto base = graph_current_snapshot(context, label + " eager");
    graph_current_require_observed(base, label + " eager");
    GraphReplaySnapshot result;
    result.position = base.position;
    result.device_position = base.device_position;
    result.logits = base.logits;
    result.gdn = base.gdn;
    result.convolution = base.convolution;
    result.oscar = base.oscar;
    return result;
}

void graph_replay_write_snapshot(std::ostream& out, const std::string& label,
                                 const GraphReplaySnapshot& snapshot) {
    out << label << " position=" << snapshot.position
        << " device_position=" << snapshot.device_position
        << " logits_hash=" << graph_current_hash(snapshot.logits)
        << " gdn_hash=" << graph_current_hash(snapshot.gdn)
        << " convolution_hash=" << graph_current_hash(snapshot.convolution)
        << " oscar_bytes=" << snapshot.oscar.size()
        << " oscar_hash=" << graph_current_hash(snapshot.oscar) << '\n';
    for (std::size_t bank = 0; bank < snapshot.counts.size(); ++bank)
        out << label << " bank=" << bank
            << " counts=" << snapshot.counts[bank][0] << ','
            << snapshot.counts[bank][1] << ','
            << snapshot.counts[bank][2] << ','
            << snapshot.counts[bank][3] << ','
            << snapshot.counts[bank][4] << '\n';
}

void graph_replay_require_equal(const GraphReplaySnapshot& actual,
                                const GraphReplaySnapshot& expected,
                                const std::string& label) {
    require(actual.position == expected.position &&
                actual.device_position == expected.device_position,
            label + " position mismatch");
    require(actual.logits == expected.logits, label + " logits mismatch");
    require(actual.gdn == expected.gdn, label + " GDN mismatch");
    require(actual.convolution == expected.convolution,
            label + " physical convolution mismatch");
    require(actual.oscar == expected.oscar, label + " canonical OSCAR mismatch");
}

void graph_replay_require_counts(const GraphReplaySnapshot& snapshot,
                                 const std::string& label) {
    const int context = snapshot.position;
    const int prefix = std::min(context, 64);
    const int begin = context <= 64 ? context : std::max(64, context - 256);
    const std::array<int, 5> expected = {
        prefix, begin - prefix, context - begin, begin & 255, context - 1};
    for (const auto& counts : snapshot.counts)
        require(counts == expected, label + " device count extent mismatch");
}

void graph_current_enable_oscar(Exl3TextContext& context);
void graph_current_require_environment(int max_context);

void run_graph_current_replay(Exl3TextModel& target,
                              const std::vector<std::int64_t>& ids,
                              const std::string& out_path, int max_context) {
    require(!std::filesystem::exists(out_path),
            "graphcurrentreplay output must be a new file");
    std::ofstream report(out_path);
    require(report.good(), "graphcurrentreplay cannot open output");
    report << "mode=graphcurrentreplay\n"
           << "qualification_scope=exact logits,GDN,physical_conv,positions,graph_observer_OSCAR; capture_taps=false\n"
           << "unqualified_scope=taps,embedding_traces,full_attention_V_image,qkv_trace_valid,graph_class_getter\n"
           << "prefix=320 chunks=16+128+128+48 graph_class=16 replay_steps=4 stream=nonblocking_named\n"
           << "capture_metadata=reported_only; no counter-equality claim\n";
    std::string phase = "environment";
    try {
        graph_current_require_environment(max_context);
        require(ids.size() >= 324,
                "graphcurrentreplay prompt requires at least 324 fixed IDs");
        GraphCurrentStreamGuard stream_guard;
        const auto stream = stream_guard.stream;
        auto eager = target.create_context(false);
        auto graph = target.create_context(false);
        graph_current_enable_oscar(*eager);
        graph_current_enable_oscar(*graph);

        phase = "prefix320";
        graph_current_ingest(*eager, ids, 320, stream);
        graph_current_ingest(*graph, ids, 320, stream);
        const auto eager_prefix = graph_replay_eager_snapshot(*eager, "prefix320 eager");
        const auto graph_prefix = graph_replay_eager_snapshot(*graph, "prefix320 graph");
        graph_replay_write_snapshot(report, "prefix320 eager", eager_prefix);
        graph_replay_write_snapshot(report, "prefix320 graph", graph_prefix);
        graph_replay_require_equal(graph_prefix, eager_prefix,
                                   "prefix320 eager-vs-graph");

        phase = "capture320";
        graph->oscar_set_graph_class(16);
        graph->oscar_sync_device_state(stream);
        cuda_check(cudaStreamSynchronize(stream),
                   "graphcurrentreplay graph-class setup sync");
        const auto before_capture = graph_current_snapshot(*graph,
                                                            "capture320 before");
        graph_current_write_snapshot(report, "capture320 before", before_capture);
        require(graph->capture_decode_graph(stream),
                "graphcurrentreplay capture failed: " + graph->graph_status());
        cuda_check(cudaStreamSynchronize(stream), "graphcurrentreplay capture sync");
        const auto after_capture = graph_current_snapshot(*graph,
                                                           "capture320 after");
        graph_current_write_snapshot(report, "capture320 after", after_capture);

        phase = "replay320";
        for (int step = 0; step < 4; ++step) {
            const auto token = ids[static_cast<std::size_t>(320 + step)];
            eager->decode(token, stream);
            graph->decode_graph(token, stream);
            cuda_check(cudaStreamSynchronize(stream),
                       "graphcurrentreplay replay sync");
            const auto eager_state = graph_replay_eager_snapshot(
                *eager, "replay step=" + std::to_string(step));
            const auto graph_state = graph_replay_snapshot(
                *graph, stream, "replay step=" + std::to_string(step));
            graph_replay_write_snapshot(report,
                                        "replay step=" + std::to_string(step) + " eager",
                                        eager_state);
            graph_replay_write_snapshot(report,
                                        "replay step=" + std::to_string(step) + " graph",
                                        graph_state);
            graph_replay_require_counts(graph_state,
                                        "replay step=" + std::to_string(step));
            graph_replay_require_equal(graph_state, eager_state,
                                       "replay step=" + std::to_string(step));
        }
        report << "reportStatus=PASS\n";
        report.flush();
        std::cout << "GRAPH_CURRENT_REPLAY PASS\n";
    } catch (const std::exception& error) {
        report << "reportStatus=FAIL\n"
               << "failure_phase=" << phase << "\n"
               << "failure=" << error.what() << "\n"
               << "failure_action=stop_at_first_observer_or_differential_mismatch\n";
        report.flush();
        throw;
    }
}

void run_graph_boundary_replay(Exl3TextModel& target,
                               const std::vector<std::int64_t>& ids,
                               const std::string& out_path, int max_context) {
    require(!std::filesystem::exists(out_path),
            "graphboundaryreplay output must be a new file");
    std::ofstream report(out_path);
    require(report.good(), "graphboundaryreplay cannot open output");
    report << "mode=graphboundaryreplay\n"
           << "qualification_scope=exact positions,logits,GDN,physical_conv,graph_observer_OSCAR; capture_taps=false\n"
           << "unqualified_scope=taps,embedding_traces,full_attention_V_image,qkv_trace_valid,graph_class_getter\n"
           << "prefix=256 chunks=16+128+112 graph_class=16 replay_steps=68 final_position=324 stream=nonblocking_named\n"
           << "boundary_steps=old319_to_new320,old320_to_new321\n";
    std::string phase = "environment";
    try {
        graph_current_require_environment(max_context);
        require(ids.size() >= 324,
                "graphboundaryreplay prompt requires at least 324 fixed IDs");
        GraphCurrentStreamGuard stream_guard;
        const auto stream = stream_guard.stream;
        auto eager = target.create_context(false);
        auto graph = target.create_context(false);
        graph_current_enable_oscar(*eager);
        graph_current_enable_oscar(*graph);

        phase = "prefix256";
        graph_current_ingest(*eager, ids, 256, stream);
        graph_current_ingest(*graph, ids, 256, stream);
        const auto eager_prefix = graph_replay_eager_snapshot(*eager, "prefix256 eager");
        const auto graph_prefix = graph_replay_eager_snapshot(*graph, "prefix256 graph");
        graph_replay_write_snapshot(report, "prefix256 eager", eager_prefix);
        graph_replay_write_snapshot(report, "prefix256 graph", graph_prefix);
        graph_replay_require_equal(graph_prefix, eager_prefix,
                                   "prefix256 eager-vs-graph");

        phase = "capture256";
        graph->oscar_set_graph_class(16);
        graph->oscar_sync_device_state(stream);
        cuda_check(cudaStreamSynchronize(stream),
                   "graphboundaryreplay graph-class setup sync");
        require(graph->capture_decode_graph(stream),
                "graphboundaryreplay capture failed: " + graph->graph_status());
        cuda_check(cudaStreamSynchronize(stream), "graphboundaryreplay capture sync");
        const auto eager_capture = graph_replay_eager_snapshot(
            *eager, "capture256 eager");
        graph_replay_write_snapshot(report, "capture256 eager", eager_capture);

        phase = "replay256_to_324";
        for (int step = 0; step < 68; ++step) {
            const auto token = ids[static_cast<std::size_t>(256 + step)];
            eager->decode(token, stream);
            graph->decode_graph(token, stream);
            cuda_check(cudaStreamSynchronize(stream),
                       "graphboundaryreplay replay sync");
            const auto eager_state = graph_replay_eager_snapshot(
                *eager, "replay step=" + std::to_string(step));
            const auto graph_state = graph_replay_snapshot(
                *graph, stream, "replay step=" + std::to_string(step));
            graph_replay_write_snapshot(report,
                                        "replay step=" + std::to_string(step) + " eager",
                                        eager_state);
            graph_replay_write_snapshot(report,
                                        "replay step=" + std::to_string(step) + " graph",
                                        graph_state);
            graph_replay_require_counts(graph_state,
                                        "replay step=" + std::to_string(step));
            graph_replay_require_equal(graph_state, eager_state,
                                       "replay step=" + std::to_string(step));
        }
        require(eager->position() == 324 && graph->position() == 324,
                "graphboundaryreplay final position mismatch");
        report << "reportStatus=PASS\n";
        report.flush();
        std::cout << "GRAPH_BOUNDARY_REPLAY PASS\n";
    } catch (const std::exception& error) {
        report << "reportStatus=FAIL\n"
               << "failure_phase=" << phase << "\n"
               << "failure=" << error.what() << "\n"
               << "failure_action=stop_at_first_state_or_differential_mismatch\n";
        report.flush();
        throw;
    }
}

void graph_current_enable_oscar(Exl3TextContext& context) {
    require(context.try_enable_oscar_from_environment(),
            "graphcurrentstate canonical OSCAR attachment");
}

void graph_current_require_environment(int max_context) {
    require(max_context == 1024, "graphcurrentstate requires maxctx1024");
    require(env("NINFER_OSCAR_EXL3") == "1",
            "graphcurrentstate requires canonical OSCAR");
    require(env("NINFER_EXL3_PREFILL_ROWPAIR_K6") == "0",
            "graphcurrentstate requires rowpair0");
    require(env("NINFER_EXL3_WIDE_PREFILL") == "1" &&
                env("NINFER_EXL3_PREFILL_WIDE64") == "1" &&
                env("NINFER_EXL3_PREFILL_WIDE128") == "1" &&
                (env("NINFER_EXL3_SHARED_LAYER_SCRATCH").empty() ||
                 env("NINFER_EXL3_SHARED_LAYER_SCRATCH") == "1"),
            "graphcurrentstate requires the P17 wide128 overlay");
    require(env("NINFER_EXL3_PREFILL_WIDE256") != "1" &&
                env("NINFER_EXL3_PREFILL_WIDE512") != "1" &&
                env("NINFER_EXL3_PREFILL_WIDE1024") != "1",
            "graphcurrentstate requires final prefill capacity128");
}

void run_graph_current_state(Exl3TextModel& target,
                             const std::vector<std::int64_t>& ids,
                             const std::string& out_path, int max_context) {
    require(!std::filesystem::exists(out_path),
            "graphcurrentstate output must be a new file");
    std::ofstream report(out_path);
    require(report.good(), "graphcurrentstate cannot open output");
    report << "mode=graphcurrentstate\n"
           << "qualification_scope=exact exported logits,taps,embedding,GDN,physical_conv,OSCAR_live_state,positions; full_attention_V_not_exported\n"
           << "unobservable_contract_fields=qkv_trace_valid,OSCAR_graph_class,full_attention_V_image\n"
           << "capture_diagnostic_policy=graph_active_and_status_transition_allowed; all_other_exposed_metadata_must_be_unchanged\n"
           << "prefix=320 chunks=16+128+128+48 capture_only_first_probe=1 stream=nonblocking_named\n"
           << "deferred_until_capture_scope_passes=transition511_reset_reingest_replay\n";
    std::string phase = "environment";
    try {
        graph_current_require_environment(max_context);
        require(ids.size() >= 324,
                "graphcurrentstate prompt requires at least 324 fixed IDs");
        GraphCurrentStreamGuard stream_guard;
        const auto stream = stream_guard.stream;
        auto eager = target.create_context(true);
        auto graph = target.create_context(true);
        graph_current_enable_oscar(*eager);
        graph_current_enable_oscar(*graph);

        phase = "prefix320";
        graph_current_ingest(*eager, ids, 320, stream);
        graph_current_ingest(*graph, ids, 320, stream);
        auto eager_prefix = graph_current_snapshot(*eager, "prefix320 eager");
        auto graph_prefix = graph_current_snapshot(*graph, "prefix320 graph");
        graph_current_write_snapshot(report, "prefix320 eager", eager_prefix);
        graph_current_write_snapshot(report, "prefix320 graph", graph_prefix);
        graph_current_require_model_equal(graph_prefix, eager_prefix,
                                          "prefix320 eager-vs-graph");

        phase = "capture320";
        graph->oscar_set_graph_class(16);
        graph->oscar_sync_device_state(stream);
        cuda_check(cudaStreamSynchronize(stream),
                   "graphcurrentstate graph-class setup sync");
        auto before_capture = graph_current_snapshot(*graph, "capture320 before");
        graph_current_write_snapshot(report, "capture320 before", before_capture);
        require(graph->capture_decode_graph(stream),
                "graphcurrentstate capture failed: " + graph->graph_status());
        cuda_check(cudaStreamSynchronize(stream), "graphcurrentstate capture sync");
        auto after_capture = graph_current_snapshot(*graph, "capture320 after");
        graph_current_write_snapshot(report, "capture320 after", after_capture);
        graph_current_require_model_equal(after_capture, before_capture,
                                          "capture320 before-vs-after model state");
        graph_current_require_capture_diagnostics(before_capture, after_capture,
                                                  "capture320");

        report << "reportStatus=PASS\n";
        report.flush();
        std::cout << "GRAPH_CURRENT_STATE PASS\n";
    } catch (const std::exception& error) {
        report << "reportStatus=FAIL\n"
               << "failure_phase=" << phase << "\n"
               << "failure=" << error.what() << "\n"
               << "failure_action=stop_at_first_mismatch_or_unobservable_contract_field\n";
        report.flush();
        throw;
    }
}
