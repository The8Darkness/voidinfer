#pragma once

struct ProjectionRouteRecord {
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    int layer = -1;
    std::string operation;
    std::vector<std::uint16_t> guarded_input;
    std::unique_ptr<DeviceBuffer> device_input;
};

struct ProjectionRouteWorkspace {
    int in_features = 0;
    int out_features = 0;
    std::string operation;
    std::unique_ptr<Exl3CudaLinearWorkspace> workspace;
};

ninfer::exl3::Exl3CudaLinearAdmission projection_route_serial_admission(
    const std::string& operation) {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    if (operation == "gate" || operation == "up")
        return Admission::target_continuation_gate_up;
    if (operation == "down") return Admission::target_continuation_down;
    if (operation == "o") return Admission::target_continuation_o;
    if (operation == "z") return Admission::target_continuation_z;
    if (operation == "qkv") return Admission::target_continuation_qkv;
    if (operation == "k" || operation == "v")
        return Admission::target_continuation_kv;
    if (operation == "q") return Admission::target_continuation_q;
    throw std::runtime_error("projection route matrix unknown operation: " + operation);
}

Exl3CudaLinearWorkspace& projection_route_workspace(
    std::vector<ProjectionRouteWorkspace>& workspaces,
    const ProjectionRouteRecord& record) {
    for (auto& entry : workspaces)
        if (entry.in_features == record.metadata.in_features &&
            entry.out_features == record.metadata.out_features &&
            entry.operation == record.operation) return *entry.workspace;
    const bool gate_up = record.operation == "gate" || record.operation == "up";
    const bool down = record.operation == "down";
    const bool output = record.operation == "o";
    const bool z = record.operation == "z";
    const bool qkv = record.operation == "qkv";
    const bool kv = record.operation == "k" || record.operation == "v";
    const bool q = record.operation == "q";
    ProjectionRouteWorkspace entry;
    entry.in_features = record.metadata.in_features;
    entry.out_features = record.metadata.out_features;
    entry.operation = record.operation;
    entry.workspace = std::make_unique<Exl3CudaLinearWorkspace>(
        entry.in_features, entry.out_features, 32, false, gate_up, down,
        output, z, true, false, ninfer::exl3::Exl3CudaAccumulationView{},
        ninfer::exl3::Exl3CudaTransformView{}, qkv, kv, q);
    workspaces.push_back(std::move(entry));
    return *workspaces.back().workspace;
}

std::vector<TargetK6CapturedProjection> capture_projection_route_context(
    Exl3TextModel& target, const std::vector<std::int64_t>& source,
    std::size_t begin) {
    constexpr int prefix_rows = 512;
    require(begin + prefix_rows + 7 <= source.size(),
            "projection route source slice is incomplete");
    std::vector<std::int64_t> prefix(
        source.begin() + static_cast<std::ptrdiff_t>(begin),
        source.begin() + static_cast<std::ptrdiff_t>(begin + prefix_rows));
    for (const auto token : prefix)
        require(token >= 0 && token < kVocab && token != kMaskToken,
                "projection route prefix token is invalid");
    auto context = target.create_context(true);
    require(context->try_enable_oscar_from_environment(),
            "projection route OSCAR attachment failed");
    ingest_prefix(*context, prefix, CommitSink{});
    const auto pending = sample_target(*context);
    std::vector<std::int64_t> block{pending};
    for (int row = 0; row < 7; ++row) {
        const auto token = source[begin + prefix_rows + static_cast<std::size_t>(row)];
        require(token >= 0 && token < kVocab && token != kMaskToken,
                "projection route continuation token is invalid");
        block.push_back(token);
    }
    context->prepare_transaction();
    context->prepare_continuation(8);
    std::vector<TargetK6CapturedProjection> captured;
    captured.reserve(400);
    context->set_target_projection_observer_for_test(
        capture_target_k6_projection, &captured, nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    struct ResetObserver {
        Exl3TextContext* context = nullptr;
        ~ResetObserver() {
            if (context) context->set_target_projection_observer_for_test(nullptr);
        }
    } reset{context.get()};
    context->begin_transaction();
    context->continue_rows(block);
    context->set_target_projection_observer_for_test(nullptr);
    reset.context = nullptr;
    context->rollback_transaction();
    cuda_check(cudaStreamSynchronize(nullptr),
               "synchronize projection route capture rollback");
    require(!context->transaction_active() && context->position() == prefix_rows &&
                sample_target(*context) == pending,
            "projection route capture did not restore target state");
    return captured;
}

std::vector<std::string> projection_route_expected_inventory() {
    std::vector<std::string> expected;
    expected.reserve(400);
    for (int layer = 0; layer < 64; ++layer) {
        const std::vector<std::string> operations = layer % 4 == 3
            ? std::vector<std::string>{"q", "k", "v", "o", "gate", "up", "down"}
            : std::vector<std::string>{"qkv", "z", "o", "gate", "up", "down"};
        for (const auto& operation : operations)
            expected.push_back(std::to_string(layer) + ":" + operation);
    }
    return expected;
}

void validate_projection_route_captures(
    const std::vector<std::vector<TargetK6CapturedProjection>>& captures) {
    require(captures.size() == 4, "projection route matrix requires four captures");
    const auto expected = projection_route_expected_inventory();
    for (std::size_t context = 0; context < captures.size(); ++context) {
        require(captures[context].size() == expected.size(),
                "projection route canonical projection count mismatch");
        std::set<std::string> seen;
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const auto& item = captures[context][index];
            const std::string key = std::to_string(item.layer) + ":" + item.operation;
            require(key == expected[index] && seen.insert(key).second &&
                        item.metadata.in_features > 0 && item.metadata.out_features > 0 &&
                        item.metadata.K >= 5 && item.metadata.K <= 8 &&
                        item.input.size() == 8u * static_cast<std::size_t>(item.metadata.in_features) &&
                        item.weights.trellis && item.weights.suh && item.weights.svh,
                    "projection route inventory/shape provenance mismatch");
            require(std::all_of(item.input.begin(), item.input.end(), [](std::uint16_t bits) {
                        return std::isfinite(half_to_float(bits));
                    }), "projection route captured nonfinite input");
            if (context == 0) continue;
            const auto& first = captures[0][index];
            require(item.layer == first.layer && item.operation == first.operation &&
                        item.m1_dispatch == first.m1_dispatch &&
                        item.metadata.in_features == first.metadata.in_features &&
                        item.metadata.out_features == first.metadata.out_features &&
                        item.metadata.K == first.metadata.K &&
                        item.metadata.mcg == first.metadata.mcg &&
                        item.metadata.mul1 == first.metadata.mul1 &&
                        item.metadata.has_bias == first.metadata.has_bias &&
                        item.weights.trellis == first.weights.trellis &&
                        item.weights.suh == first.weights.suh &&
                        item.weights.svh == first.weights.svh &&
                        item.weights.mul1 == first.weights.mul1,
                    "projection route aligned weight/metadata provenance mismatch");
        }
    }
}

std::vector<ProjectionRouteRecord> make_projection_route_records(
    const std::vector<std::vector<TargetK6CapturedProjection>>& captures) {
    constexpr std::uint16_t input_canary = 0x3555u;
    std::vector<ProjectionRouteRecord> records;
    records.reserve(captures[0].size());
    for (std::size_t index = 0; index < captures[0].size(); ++index) {
        const auto& first = captures[0][index];
        ProjectionRouteRecord record;
        record.weights = first.weights;
        record.metadata = first.metadata;
        record.layer = first.layer;
        record.operation = first.operation;
        record.guarded_input.push_back(input_canary);
        for (const auto& capture : captures)
            record.guarded_input.insert(record.guarded_input.end(),
                capture[index].input.begin(), capture[index].input.end());
        record.guarded_input.push_back(input_canary);
        record.device_input = std::make_unique<DeviceBuffer>(
            record.guarded_input.size() * sizeof(std::uint16_t));
        cuda_check(cudaMemcpy(record.device_input->get(), record.guarded_input.data(),
                              record.guarded_input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "upload projection route guarded input");
        records.push_back(std::move(record));
    }
    return records;
}

void launch_projection_route_record(
    const ProjectionRouteRecord& record, Exl3CudaLinearWorkspace& workspace,
    int contexts, bool candidate, std::uint16_t* output) {
    const auto* input = static_cast<const std::uint16_t*>(record.device_input->get()) + 1;
    const auto serial_admission = projection_route_serial_admission(record.operation);
    const std::size_t input_stride = 8u * static_cast<std::size_t>(record.metadata.in_features);
    const std::size_t output_stride = 8u * static_cast<std::size_t>(record.metadata.out_features);
    const std::string serial_dispatch =
        workspace.dispatch_name(record.metadata, 8, serial_admission);
    if (!candidate) {
        for (int context = 0; context < contexts; ++context)
            workspace.forward(record.weights, record.metadata,
                              input + static_cast<std::size_t>(context) * input_stride,
                              output + static_cast<std::size_t>(context) * output_stride,
                              8, nullptr, serial_admission);
    } else if (serial_dispatch == "generic_tile") {
        for (int context = 0; context < contexts; context += 2)
            workspace.forward(record.weights, record.metadata,
                              input + static_cast<std::size_t>(context) * input_stride,
                              output + static_cast<std::size_t>(context) * output_stride,
                              16, nullptr,
                              ninfer::exl3::Exl3CudaLinearAdmission::target_initial16);
    } else {
        workspace.forward(record.weights, record.metadata, input, output,
                          contexts * 8, nullptr,
                          ninfer::exl3::Exl3CudaLinearAdmission::target_wide_prefill);
    }
}

std::string projection_route_candidate_dispatch(
    const ProjectionRouteRecord& record, Exl3CudaLinearWorkspace& workspace,
    int contexts, int& calls) {
    const auto serial_admission = projection_route_serial_admission(record.operation);
    const std::string serial = workspace.dispatch_name(record.metadata, 8, serial_admission);
    if (serial == "generic_tile") {
        calls = contexts / 2;
        return workspace.dispatch_name(
            record.metadata, 16,
            ninfer::exl3::Exl3CudaLinearAdmission::target_initial16);
    }
    calls = 1;
    return workspace.dispatch_name(
        record.metadata, contexts * 8,
        ninfer::exl3::Exl3CudaLinearAdmission::target_wide_prefill);
}

double time_projection_route_sequence(
    const std::vector<ProjectionRouteRecord>& records,
    std::vector<ProjectionRouteWorkspace>& workspaces,
    int contexts, bool candidate, std::uint16_t* output) {
    ProjectionOrderedC2Event begin, end;
    cuda_check(cudaEventRecord(begin.get(), nullptr),
               "record projection route timing begin");
    for (const auto& record : records)
        launch_projection_route_record(record,
            projection_route_workspace(workspaces, record), contexts, candidate, output);
    cuda_check(cudaEventRecord(end.get(), nullptr),
               "record projection route timing end");
    cuda_check(cudaEventSynchronize(end.get()),
               "synchronize projection route timing end");
    float milliseconds = 0.0F;
    cuda_check(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()),
               "read projection route elapsed time");
    require(milliseconds > 0.0F, "projection route timing must be positive");
    return milliseconds;
}

void run_projection_route_matrix_screen(
    Exl3TextModel& target, const std::vector<std::int64_t>& source,
    const std::filesystem::path& output_directory) {
    constexpr std::uint16_t output_canary = 0x5a5au;
    require(target.max_context() >= 520 && source.size() >= 2055,
            "projection route matrix requires maxctx520 and 2055 source tokens");
    require(env("NINFER_EXL3_TARGET_INITIAL16_ORDERED") == "1",
            "projection route matrix requires explicit initial16 ordered admission");
    require(!output_directory.empty() && !std::filesystem::exists(output_directory) &&
                std::filesystem::create_directories(output_directory),
            "projection route matrix requires a new output directory");
    for (int left = 0; left < 4; ++left)
        for (int right = left + 1; right < 4; ++right)
            require(!std::equal(source.begin() + left * 512,
                                source.begin() + (left + 1) * 512,
                                source.begin() + right * 512),
                    "projection route matrix requires four distinct prefixes");

    std::vector<std::vector<TargetK6CapturedProjection>> captures;
    for (int context = 0; context < 4; ++context)
        captures.push_back(capture_projection_route_context(
            target, source, static_cast<std::size_t>(context) * 512u));
    validate_projection_route_captures(captures);
    auto records = make_projection_route_records(captures);
    captures.clear();
    captures.shrink_to_fit();

    std::size_t max_output_values = 0;
    for (const auto& record : records)
        max_output_values = std::max(max_output_values,
            32u * static_cast<std::size_t>(record.metadata.out_features));
    std::vector<std::uint16_t> guarded_output(max_output_values + 2, output_canary);
    DeviceBuffer serial_buffer(guarded_output.size() * sizeof(std::uint16_t));
    DeviceBuffer candidate_buffer(guarded_output.size() * sizeof(std::uint16_t));
    auto* serial = static_cast<std::uint16_t*>(serial_buffer.get()) + 1;
    auto* candidate = static_cast<std::uint16_t*>(candidate_buffer.get()) + 1;
    std::vector<ProjectionRouteWorkspace> workspaces;
    std::ofstream exact(output_directory / "exactness.csv");
    require(exact.good(), "cannot create projection route exactness matrix");
    exact.imbue(std::locale::classic());
    exact << "contexts,index,layer,operation,K,in_features,out_features,serial_dispatch,"
             "candidate_dispatch,candidate_calls,values,mismatches,first_mismatch,"
             "input_canary,serial_canary,candidate_canary,route_ok\n";
    std::uint64_t total_mismatches = 0;
    std::uint64_t total_canary_failures = 0;
    std::uint64_t total_route_failures = 0;
    for (const int contexts : {2, 4}) {
        for (std::size_t index = 0; index < records.size(); ++index) {
            const auto& record = records[index];
            auto& workspace = projection_route_workspace(workspaces, record);
            cuda_check(cudaMemcpy(serial_buffer.get(), guarded_output.data(),
                                  guarded_output.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "initialize projection route serial canaries");
            cuda_check(cudaMemcpy(candidate_buffer.get(), guarded_output.data(),
                                  guarded_output.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "initialize projection route candidate canaries");
            launch_projection_route_record(record, workspace, contexts, false, serial);
            launch_projection_route_record(record, workspace, contexts, true, candidate);
            cuda_check(cudaDeviceSynchronize(),
                       "synchronize projection route exactness record");
            const std::size_t values = static_cast<std::size_t>(contexts) * 8u *
                static_cast<std::size_t>(record.metadata.out_features);
            std::vector<std::uint16_t> serial_host(values + 2), candidate_host(values + 2);
            cuda_check(cudaMemcpy(serial_host.data(), serial_buffer.get(),
                                  serial_host.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "download projection route serial output");
            cuda_check(cudaMemcpy(candidate_host.data(), candidate_buffer.get(),
                                  candidate_host.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "download projection route candidate output");
            std::uint64_t mismatches = 0;
            std::int64_t first_mismatch = -1;
            for (std::size_t value = 0; value < values; ++value)
                if (serial_host[value + 1] != candidate_host[value + 1]) {
                    if (first_mismatch < 0) first_mismatch = static_cast<std::int64_t>(value);
                    ++mismatches;
                }
            total_mismatches += mismatches;
            std::array<std::uint16_t, 2> input_guards{};
            const auto* input_base = static_cast<const std::uint16_t*>(record.device_input->get());
            cuda_check(cudaMemcpy(&input_guards[0], input_base, sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "download projection route input prefix canary");
            cuda_check(cudaMemcpy(&input_guards[1],
                                  input_base + record.guarded_input.size() - 1,
                                  sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                       "download projection route input suffix canary");
            const bool input_ok = input_guards[0] == 0x3555u && input_guards[1] == 0x3555u;
            const bool serial_ok = serial_host.front() == output_canary &&
                                   serial_host.back() == output_canary;
            const bool candidate_ok = candidate_host.front() == output_canary &&
                                      candidate_host.back() == output_canary;
            int candidate_calls = 0;
            const auto serial_dispatch = workspace.dispatch_name(
                record.metadata, 8, projection_route_serial_admission(record.operation));
            const auto candidate_dispatch = projection_route_candidate_dispatch(
                record, workspace, contexts, candidate_calls);
            const bool route_ok = serial_dispatch == "generic_tile"
                ? candidate_dispatch == "target_initial16_ordered"
                : candidate_dispatch.rfind("target_wide_prefill", 0) == 0;
            exact << contexts << ',' << index << ',' << record.layer << ','
                  << record.operation << ',' << record.metadata.K << ','
                  << record.metadata.in_features << ',' << record.metadata.out_features << ','
                  << serial_dispatch << ',' << candidate_dispatch << ',' << candidate_calls << ','
                  << values << ',' << mismatches << ',' << first_mismatch << ','
                  << input_ok << ',' << serial_ok << ',' << candidate_ok << ','
                  << route_ok << '\n';
            total_canary_failures += static_cast<std::uint64_t>(!input_ok) +
                static_cast<std::uint64_t>(!serial_ok) +
                static_cast<std::uint64_t>(!candidate_ok);
            total_route_failures += static_cast<std::uint64_t>(!route_ok);
        }
    }
    exact.flush();
    require(exact.good(), "projection route exactness matrix write failed");
    exact.close();
    require(total_mismatches == 0 && total_canary_failures == 0 &&
                total_route_failures == 0,
            "projection route exactness matrix contains mismatch/canary/route failures");

    time_projection_route_sequence(records, workspaces, 2, false, serial);
    time_projection_route_sequence(records, workspaces, 2, true, candidate);
    time_projection_route_sequence(records, workspaces, 4, false, serial);
    time_projection_route_sequence(records, workspaces, 4, true, candidate);
    std::ofstream performance(output_directory / "performance.csv");
    require(performance.good(), "cannot create projection route performance CSV");
    performance.imbue(std::locale::classic());
    performance << "row_kind,contexts,round,order,records,serial_ms,candidate_ms,"
                   "gain_pct,gate_pct,gate_pass\n";
    performance << std::fixed << std::setprecision(9);
    bool all_gates = true;
    for (const auto [contexts, gate] :
         {std::pair<int, double>{2, 25.0}, std::pair<int, double>{4, 35.0}}) {
        std::vector<double> gains;
        for (int round = 0; round < 5; ++round) {
            const bool serial_first = round % 2 == 0;
            double serial_ms = 0.0, candidate_ms = 0.0;
            if (serial_first) {
                serial_ms = time_projection_route_sequence(
                    records, workspaces, contexts, false, serial);
                candidate_ms = time_projection_route_sequence(
                    records, workspaces, contexts, true, candidate);
            } else {
                candidate_ms = time_projection_route_sequence(
                    records, workspaces, contexts, true, candidate);
                serial_ms = time_projection_route_sequence(
                    records, workspaces, contexts, false, serial);
            }
            const double gain = 100.0 * (serial_ms - candidate_ms) / serial_ms;
            gains.push_back(gain);
            performance << "round," << contexts << ',' << round << ','
                        << (serial_first ? "serial_candidate" : "candidate_serial") << ','
                        << records.size() << ',' << serial_ms << ',' << candidate_ms << ','
                        << gain << ',' << gate << ',' << (gain >= gate ? 1 : 0) << '\n';
        }
        const double median_gain = median(gains);
        performance << "aggregate," << contexts << ",-1,median," << records.size()
                    << ",0,0," << median_gain << ',' << gate << ','
                    << (median_gain >= gate ? 1 : 0) << '\n';
        std::cout << "PROJECTION_ROUTE_MATRIX contexts=" << contexts
                  << " records=" << records.size()
                  << " median_gain_pct=" << median_gain
                  << " gate_pct=" << gate
                  << " gate_pass=" << (median_gain >= gate) << std::endl;
        all_gates = all_gates && median_gain >= gate;
    }
    performance.flush();
    require(performance.good(), "projection route performance CSV write failed");
    require(all_gates, "projection route matrix performance gate failed");
    std::cout << "PROJECTION_ROUTE_MATRIX_SCREEN_DONE exact=1 gates=1" << std::endl;
}
