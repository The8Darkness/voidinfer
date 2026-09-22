#pragma once

// Test-only C2 screen for the exact ordered-M16 reuse hypothesis. It captures
// only the real layer-0/down K5 input from each request and never batches state.

void capture_projection_ordered_c2_down(
    const ninfer::exl3::Exl3TargetProjectionObservation& observation,
    void* user) {
    if (observation.layer != 0 || observation.operation == nullptr ||
        std::string(observation.operation) != "down") return;
    capture_target_k6_projection(
        observation, static_cast<std::vector<TargetK6CapturedProjection>*>(user));
}

TargetK6CapturedProjection capture_projection_ordered_c2_context(
    Exl3TextModel& target, const std::vector<std::int64_t>& source,
    std::size_t begin) {
    constexpr int prefix_rows = 512;
    require(begin + prefix_rows + 7 <= source.size(),
            "ordered C2 source slice is incomplete");
    std::vector<std::int64_t> prefix(
        source.begin() + static_cast<std::ptrdiff_t>(begin),
        source.begin() + static_cast<std::ptrdiff_t>(begin + prefix_rows));
    for (const auto token : prefix)
        require(token >= 0 && token < kVocab && token != kMaskToken,
                "ordered C2 prefix token is invalid");
    auto context = target.create_context(true);
    require(context->try_enable_oscar_from_environment(),
            "ordered C2 OSCAR attachment failed");
    ingest_prefix(*context, prefix, CommitSink{});
    const auto pending = sample_target(*context);
    std::vector<std::int64_t> block{pending};
    for (int row = 0; row < 7; ++row) {
        const auto token = source[begin + prefix_rows + static_cast<std::size_t>(row)];
        require(token >= 0 && token < kVocab && token != kMaskToken,
                "ordered C2 continuation token is invalid");
        block.push_back(token);
    }
    context->prepare_transaction();
    context->prepare_continuation(8);
    std::vector<TargetK6CapturedProjection> captured;
    context->set_target_projection_observer_for_test(
        capture_projection_ordered_c2_down, &captured, nullptr,
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
               "synchronize ordered C2 capture rollback");
    require(!context->transaction_active() && context->position() == prefix_rows &&
                sample_target(*context) == pending,
            "ordered C2 capture did not restore target state");
    require(captured.size() == 1,
            "ordered C2 capture expected one layer-0/down record");
    return std::move(captured.front());
}

class ProjectionOrderedC2Event {
public:
    ProjectionOrderedC2Event() {
        cuda_check(cudaEventCreate(&event_), "create ordered C2 timing event");
    }
    ~ProjectionOrderedC2Event() { if (event_) cudaEventDestroy(event_); }
    cudaEvent_t get() const noexcept { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};

double time_projection_ordered_c2(
    Exl3CudaLinearWorkspace& workspace,
    const TargetK6CapturedProjection& record,
    const std::uint16_t* input, std::uint16_t* output, bool candidate) {
    ProjectionOrderedC2Event begin, end;
    cuda_check(cudaEventRecord(begin.get(), nullptr),
               "record ordered C2 timing begin");
    if (candidate) {
        workspace.forward(record.weights, record.metadata, input, output, 16, nullptr,
                          ninfer::exl3::Exl3CudaLinearAdmission::target_initial16);
    } else {
        constexpr std::size_t input_stride = 8u * 17408u;
        constexpr std::size_t output_stride = 8u * 5120u;
        workspace.forward(record.weights, record.metadata, input, output, 8, nullptr,
                          ninfer::exl3::Exl3CudaLinearAdmission::target_continuation_down);
        workspace.forward(record.weights, record.metadata, input + input_stride,
                          output + output_stride, 8, nullptr,
                          ninfer::exl3::Exl3CudaLinearAdmission::target_continuation_down);
    }
    cuda_check(cudaEventRecord(end.get(), nullptr),
               "record ordered C2 timing end");
    cuda_check(cudaEventSynchronize(end.get()),
               "synchronize ordered C2 timing end");
    float milliseconds = 0.0F;
    cuda_check(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()),
               "read ordered C2 elapsed time");
    require(milliseconds > 0.0F, "ordered C2 timing must be positive");
    return milliseconds;
}

void run_projection_ordered_c2_screen(
    Exl3TextModel& target, const std::vector<std::int64_t>& source,
    const std::string& output_path) {
    constexpr std::size_t input_values = 16u * 17408u;
    constexpr std::size_t output_values = 16u * 5120u;
    constexpr std::uint16_t input_canary = 0x3555u;
    constexpr std::uint16_t output_canary = 0x5a5au;
    require(target.max_context() >= 520 && source.size() >= 1031,
            "ordered C2 screen requires maxctx520 and 1031 source tokens");
    require(!std::equal(source.begin(), source.begin() + 512,
                        source.begin() + 512),
            "ordered C2 screen requires two distinct request prefixes");
    require(env("NINFER_EXL3_TARGET_INITIAL16_ORDERED") == "1",
            "ordered C2 screen requires explicit initial16 ordered admission");
    require(!output_path.empty() && !std::filesystem::exists(output_path),
            "ordered C2 screen requires a new output path");

    std::array<TargetK6CapturedProjection, 2> captures{
        capture_projection_ordered_c2_context(target, source, 0),
        capture_projection_ordered_c2_context(target, source, 512)};
    for (const auto& item : captures) {
        require(item.layer == 0 && item.operation == "down" &&
                    item.metadata.in_features == 17408 &&
                    item.metadata.out_features == 5120 && item.metadata.K == 5 &&
                    item.metadata.mul1 && !item.metadata.mcg &&
                    !item.metadata.has_bias && item.input.size() == 8u * 17408u &&
                    item.weights.trellis && item.weights.suh && item.weights.svh &&
                    item.weights.mul1,
                "ordered C2 layer-0/down K5 provenance mismatch");
        require(std::all_of(item.input.begin(), item.input.end(), [](std::uint16_t bits) {
                    return std::isfinite(half_to_float(bits));
                }), "ordered C2 captured nonfinite input");
    }
    require(captures[0].weights.trellis == captures[1].weights.trellis &&
                captures[0].weights.suh == captures[1].weights.suh &&
                captures[0].weights.svh == captures[1].weights.svh &&
                captures[0].weights.mul1 == captures[1].weights.mul1 &&
                captures[0].m1_dispatch == captures[1].m1_dispatch &&
                captures[0].input != captures[1].input,
            "ordered C2 weight/input provenance mismatch");

    std::vector<std::uint16_t> guarded_input(input_values + 2, input_canary);
    std::copy(captures[0].input.begin(), captures[0].input.end(),
              guarded_input.begin() + 1);
    std::copy(captures[1].input.begin(), captures[1].input.end(),
              guarded_input.begin() + 1 + captures[0].input.size());
    DeviceBuffer device_input(guarded_input.size() * sizeof(std::uint16_t));
    DeviceBuffer serial_output((output_values + 2) * sizeof(std::uint16_t));
    DeviceBuffer candidate_output((output_values + 2) * sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(device_input.get(), guarded_input.data(),
                          guarded_input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "upload ordered C2 guarded input");
    std::vector<std::uint16_t> guarded_output(output_values + 2, output_canary);
    cuda_check(cudaMemcpy(serial_output.get(), guarded_output.data(),
                          guarded_output.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "initialize ordered C2 serial canaries");
    cuda_check(cudaMemcpy(candidate_output.get(), guarded_output.data(),
                          guarded_output.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "initialize ordered C2 candidate canaries");

    Exl3CudaLinearWorkspace workspace(
        17408, 5120, 16, false, false, true, false, false, true);
    const auto serial_admission =
        ninfer::exl3::Exl3CudaLinearAdmission::target_continuation_down;
    const auto candidate_admission =
        ninfer::exl3::Exl3CudaLinearAdmission::target_initial16;
    const std::string serial_dispatch =
        workspace.dispatch_name(captures[0].metadata, 8, serial_admission);
    const std::string candidate_dispatch =
        workspace.dispatch_name(captures[0].metadata, 16, candidate_admission);
    require(serial_dispatch == "generic_tile" &&
                candidate_dispatch == "target_initial16_ordered",
            "ordered C2 dispatch contract mismatch");

    auto* input = static_cast<std::uint16_t*>(device_input.get()) + 1;
    auto* serial = static_cast<std::uint16_t*>(serial_output.get()) + 1;
    auto* candidate = static_cast<std::uint16_t*>(candidate_output.get()) + 1;
    workspace.forward(captures[0].weights, captures[0].metadata, input, serial,
                      8, nullptr, serial_admission);
    workspace.forward(captures[0].weights, captures[0].metadata,
                      input + 8u * 17408u, serial + 8u * 5120u,
                      8, nullptr, serial_admission);
    workspace.forward(captures[0].weights, captures[0].metadata, input, candidate,
                      16, nullptr, candidate_admission);
    cuda_check(cudaDeviceSynchronize(), "synchronize ordered C2 exactness");

    std::vector<std::uint16_t> input_check(guarded_input.size());
    std::vector<std::uint16_t> serial_check(guarded_output.size());
    std::vector<std::uint16_t> candidate_check(guarded_output.size());
    cuda_check(cudaMemcpy(input_check.data(), device_input.get(),
                          input_check.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "download ordered C2 guarded input");
    cuda_check(cudaMemcpy(serial_check.data(), serial_output.get(),
                          serial_check.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "download ordered C2 serial output");
    cuda_check(cudaMemcpy(candidate_check.data(), candidate_output.get(),
                          candidate_check.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "download ordered C2 candidate output");
    require(input_check == guarded_input &&
                serial_check.front() == output_canary && serial_check.back() == output_canary &&
                candidate_check.front() == output_canary && candidate_check.back() == output_canary,
            "ordered C2 input/output canary mismatch");
    require(std::equal(serial_check.begin() + 1, serial_check.end() - 1,
                       candidate_check.begin() + 1),
            "ordered C2 output bits mismatch");

    workspace.forward(captures[0].weights, captures[0].metadata, input, candidate,
                      16, nullptr, candidate_admission);
    cuda_check(cudaDeviceSynchronize(),
               "synchronize ordered C2 deterministic repeat");
    std::vector<std::uint16_t> candidate_repeat(guarded_output.size());
    cuda_check(cudaMemcpy(candidate_repeat.data(), candidate_output.get(),
                          candidate_repeat.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download ordered C2 deterministic repeat");
    require(candidate_repeat == candidate_check &&
                candidate_repeat.front() == output_canary &&
                candidate_repeat.back() == output_canary,
            "ordered C2 candidate repeat/canary mismatch");

    time_projection_ordered_c2(workspace, captures[0], input, serial, false);
    time_projection_ordered_c2(workspace, captures[0], input, candidate, true);
    struct Round { int index; std::string order; double serial_ms; double candidate_ms; double gain; };
    std::vector<Round> rounds;
    std::vector<double> gains;
    for (int round = 0; round < 7; ++round) {
        Round result{round, round % 2 == 0 ? "serial_candidate" : "candidate_serial", 0, 0, 0};
        if (round % 2 == 0) {
            result.serial_ms = time_projection_ordered_c2(
                workspace, captures[0], input, serial, false);
            result.candidate_ms = time_projection_ordered_c2(
                workspace, captures[0], input, candidate, true);
        } else {
            result.candidate_ms = time_projection_ordered_c2(
                workspace, captures[0], input, candidate, true);
            result.serial_ms = time_projection_ordered_c2(
                workspace, captures[0], input, serial, false);
        }
        result.gain = 100.0 * (result.serial_ms - result.candidate_ms) /
            result.serial_ms;
        gains.push_back(result.gain);
        rounds.push_back(std::move(result));
    }
    const double median_gain = median(gains);
    std::ofstream output(output_path);
    require(output.good(), "cannot create ordered C2 output");
    output.imbue(std::locale::classic());
    output << "row_kind,round,order,contexts,rows,layer,operation,K,in_features,"
              "out_features,serial_dispatch,candidate_dispatch,exact_values,"
              "serial_ms,candidate_ms,gain_pct,gate_pct,gate_pass\n";
    output << std::fixed << std::setprecision(9);
    for (const auto& round : rounds)
        output << "round," << round.index << ',' << round.order
               << ",2,16,0,down,5,17408,5120," << serial_dispatch << ','
               << candidate_dispatch << ',' << output_values << ','
               << round.serial_ms << ',' << round.candidate_ms << ','
               << round.gain << ",25," << (round.gain >= 25.0 ? 1 : 0) << '\n';
    output << "aggregate,-1,median,2,16,0,down,5,17408,5120,"
           << serial_dispatch << ',' << candidate_dispatch << ',' << output_values
           << ",0,0," << median_gain << ",25,"
           << (median_gain >= 25.0 ? 1 : 0) << '\n';
    output.flush();
    require(output.good(), "ordered C2 output write failed");
    std::cout << "PROJECTION_ORDERED_C2 exact_values=" << output_values
              << " median_gain_pct=" << median_gain
              << " gate_pct=25 gate_pass=" << (median_gain >= 25.0) << std::endl;
    require(median_gain >= 25.0, "ordered C2 performance gate failed");
    std::cout << "PROJECTION_ORDERED_C2_SCREEN_DONE exact=1 gate=1" << std::endl;
}
