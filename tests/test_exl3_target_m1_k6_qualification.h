#pragma once

// T05: exact real-model proof for the target-owned K6/N32 M1 async-A route.
// The target model used for capture is constructed with the candidate disabled;
// borrowed weights remain valid while the local control/candidate workspaces run.


constexpr std::initializer_list<const char*> kTargetM1K6EnvironmentOptions{
    "NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A",
    "NINFER_EXL3_TARGET_M1_K6_N16",
    "NINFER_EXL3_GENERIC_SPLITS",
    "NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV"};
void capture_target_m1_k6_projection(
    const ninfer::exl3::Exl3TargetProjectionObservation& observation,
    void* user) {
    auto& captured = *static_cast<std::vector<TargetK6CapturedProjection>*>(user);
    require(observation.input != nullptr && observation.operation != nullptr &&
                observation.m1_dispatch != nullptr && observation.rows >= 2,
            "target M1 K6 capture received an incomplete observation");
    TargetK6CapturedProjection item;
    item.weights = observation.weights;
    item.metadata = observation.metadata;
    item.layer = observation.layer;
    item.operation = observation.operation;
    item.m1_dispatch = observation.m1_dispatch;
    // The production continuation executes each row through the M1 path. One
    // represented row per owner is sufficient for the direct differential;
    // copying only that row keeps the proof bounded and unambiguous.
    item.input.resize(static_cast<std::size_t>(observation.metadata.in_features));
    cuda_check(cudaMemcpyAsync(item.input.data(), observation.input,
                               item.input.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, observation.stream),
               "capture target M1 K6 represented input");
    cuda_check(cudaStreamSynchronize(observation.stream),
               "synchronize target M1 K6 represented input capture");
    captured.push_back(std::move(item));
}

std::vector<TargetK6CapturedProjection> run_target_m1_k6_capture(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt, bool oscar) {
    require(prompt.size() == 512, "target M1 K6 capture requires ctx512");
    require(target.max_context() >= 514,
            "target M1 K6 capture requires context capacity at least 514");
    require(oscar, "target M1 K6 capture requires canonical OSCAR");

    TapStage stage;
    for (int tap = 0; tap < kTapCount; ++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(16) * kHidden * sizeof(std::uint16_t)));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t)));
        stage.bulk_ptrs.push_back(
            static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(
            static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }

    auto context = target.create_context(true);
    require(context->try_enable_oscar_from_environment(),
            "target M1 K6 capture could not enable OSCAR");
    draft.reset();
    ingest_prefix(*context, prompt, CommitSink{&draft, &stage});
    require(context->position() == 512 && context->device_position_host() == 511 &&
                draft.ring_count() == 512 && draft.ring_base_abs() == 0,
            "target M1 K6 capture prefix state mismatch");

    const std::int64_t pending = sample_target(*context);
    const auto ring_before = draft.ring_digest();
    context->prepare_transaction();
    context->prepare_continuation(2);
    std::vector<TargetK6CapturedProjection> captured;
    captured.reserve(229);
    context->set_target_projection_observer_for_test(
        capture_target_m1_k6_projection, &captured, nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::m1_k6_n32);
    struct ObserverReset {
        Exl3TextContext* context = nullptr;
        ~ObserverReset() {
            if (context) context->set_target_projection_observer_for_test(nullptr);
        }
    } observer_reset{context.get()};
    context->begin_transaction();
    context->continue_rows(std::vector<std::int64_t>{pending, 198});
    context->set_target_projection_observer_for_test(nullptr);
    observer_reset.context = nullptr;
    context->rollback_transaction();
    cuda_check(cudaStreamSynchronize(nullptr),
               "synchronize target M1 K6 capture rollback");
    require(!context->transaction_active() && context->position() == 512 &&
                context->device_position_host() == 511 &&
                sample_target(*context) == pending,
            "target M1 K6 capture did not restore target state");
    require(draft.ring_digest() == ring_before && draft.ring_count() == 512 &&
                draft.ring_base_abs() == 0,
            "target M1 K6 capture changed the draft ring");
    require(captured.size() == 229,
            "target M1 K6 capture canonical record count mismatch");

    std::array<int, 64> layer_records{};
    for (const auto& item : captured) {
        require(item.layer >= 0 && item.layer < 64 && item.metadata.K == 6 &&
                    item.metadata.mul1 && !item.metadata.mcg &&
                    !item.metadata.has_bias && item.input.size() ==
                        static_cast<std::size_t>(item.metadata.in_features) &&
                    item.metadata.out_features % (32 * 16) == 0 &&
                    item.m1_dispatch == "generic_mma_split" &&
                    item.weights.trellis && item.weights.suh && item.weights.svh &&
                    item.weights.mul1,
                "target M1 K6 capture shape/metadata mismatch");
        require(item.operation == "q" || item.operation == "k" ||
                    item.operation == "v" || item.operation == "o" ||
                    item.operation == "gate" || item.operation == "up" ||
                    item.operation == "qkv" || item.operation == "z",
                "target M1 K6 capture observed unsupported operator");
        require(std::all_of(item.input.begin(), item.input.end(),
                            [](std::uint16_t bits) {
                                return std::isfinite(half_to_float(bits));
                            }) &&
                    std::any_of(item.input.begin(), item.input.end(),
                                [](std::uint16_t bits) {
                                    return (bits & 0x7fffU) != 0U;
                                }),
                "target M1 K6 capture input invalid");
        ++layer_records[static_cast<std::size_t>(item.layer)];
    }
    require(std::all_of(layer_records.begin(), layer_records.end(),
                        [](int count) { return count > 0; }),
            "target M1 K6 capture missed a target layer");
    std::cout << "TARGETM1K6CAPTURE PASS records=" << captured.size()
              << " ctx=512 transaction_rows=2 captured_rows=1 pending=" << pending
              << " rollback=1 target_restored=1 ring_unchanged=1" << std::endl;
    return captured;
}

std::unique_ptr<Exl3CudaLinearWorkspace> make_target_m1_k6_workspace(
    const TargetK6CapturedProjection& item, bool target_owner) {
    const bool gateup = target_owner &&
        (item.operation == "gate" || item.operation == "up");
    const bool down = target_owner && item.operation == "down";
    const bool o = target_owner && item.operation == "o";
    const bool z = target_owner && item.operation == "z";
    const bool qkv = target_owner && item.operation == "qkv";
    const bool kv = target_owner &&
        (item.operation == "k" || item.operation == "v");
    const bool q = target_owner && item.operation == "q";
    return std::make_unique<Exl3CudaLinearWorkspace>(
        item.metadata.in_features, item.metadata.out_features, 1, false,
        gateup, down, o, z, false, false,
        ninfer::exl3::Exl3CudaAccumulationView{},
        ninfer::exl3::Exl3CudaTransformView{}, qkv, kv, q);
}

void run_target_m1_k6_qualification(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt, bool oscar, std::ostream& out) {
    require(env("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A") == "0",
            "target M1 K6 capture requires explicit candidate flag0");
    ninfer::test::ScopedEnvironmentRestore restore_environment(kTargetM1K6EnvironmentOptions);
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "0");
    auto captured = run_target_m1_k6_capture(target, draft, prompt, oscar);

    const auto& representative = captured.front();
    for (const char* malformed : {"2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A", malformed);
        bool rejected = false;
        try {
            auto invalid = make_target_m1_k6_workspace(representative, true);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "target M1 K6 async-A accepted malformed opt-in");
    }
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A", "1");
    for (const char* malformed : {"2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", malformed);
        bool rejected = false;
        try {
            auto invalid = make_target_m1_k6_workspace(representative, true);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "target M1 K6 N16 accepted malformed opt-in");
    }
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "0");
    auto n16_latched_off = make_target_m1_k6_workspace(representative, true);
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "1");
    require(std::string(n16_latched_off->dispatch_name(
                representative.metadata, 1)) ==
                "target_m1_k6_n32_async_a",
            "target M1 K6 N16 workspace did not latch flag0");
    auto n16_enabled = make_target_m1_k6_workspace(representative, true);
    require(std::string(n16_enabled->dispatch_name(
                representative.metadata, 1)) ==
                "target_m1_k6_n16_async_a_stream_reduction",
            "target M1 K6 N16 owned dispatch missing");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "0");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A", "0");
    auto latched_off = make_target_m1_k6_workspace(representative, true);
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A", "1");
    require(std::string(latched_off->dispatch_name(
                representative.metadata, 1)) == "generic_mma_split",
            "target M1 K6 workspace did not latch flag0");
    auto generic_owner = make_target_m1_k6_workspace(representative, false);
    require(std::string(generic_owner->dispatch_name(
                representative.metadata, 1)) == "generic_mma_split",
            "target M1 K6 route escaped target ownership");

    constexpr std::size_t guard = 128;
    constexpr std::uint16_t sentinel = 0x3555;
    out << "capture,layer,operator,in_features,out_features,K,control_dispatch,"
           "candidate_dispatch,n16_dispatch,repeat_equal,guards_ok,input_unchanged,finite,"
           "bit_mismatches,n16_bit_mismatches,changed_input,changed_bit_mismatches,"
           "changed_n16_bit_mismatches,changed_output\n";
    const std::string timing_path = env("NINFER_E5A4_TARGET_M1_K6_TIMING_OUT");
    require(!timing_path.empty(), "target M1 K6 qualification needs timing output");
    std::ofstream timing(timing_path, std::ios::trunc);
    require(timing.good(), "cannot create target M1 K6 timing CSV");
    timing << "capture,layer,operator,in_features,out_features,round,order,"
              "repetitions,control_us,candidate_us,gain_percent\n";

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    cuda_check(cudaEventCreate(&begin), "create target M1 K6 timing begin");
    cuda_check(cudaEventCreate(&end), "create target M1 K6 timing end");
    std::size_t checked_values = 0;
    std::size_t n16_checked_values = 0;
    std::size_t timing_records = 0;
    double control_total = 0.0;
    double candidate_total = 0.0;
    try {
        for (std::size_t capture = 0; capture < captured.size(); ++capture) {
            const auto& item = captured[capture];
            const std::size_t output_values =
                static_cast<std::size_t>(item.metadata.out_features);
            const std::size_t guarded_values = output_values + 2u * guard;
            DeviceBuffer input(item.input.size() * sizeof(std::uint16_t));
            DeviceBuffer output(guarded_values * sizeof(std::uint16_t));
            DeviceBuffer reference(guarded_values * sizeof(std::uint16_t));
            auto* input_device = static_cast<std::uint16_t*>(input.get());
            auto* output_base = static_cast<std::uint16_t*>(output.get());
            auto* reference_base = static_cast<std::uint16_t*>(reference.get());
            auto* output_device = output_base + guard;
            auto* reference_device = reference_base + guard;
            cuda_check(cudaMemcpy(input_device, item.input.data(),
                                  item.input.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target M1 K6 input upload");
            _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A", "0");
            auto control = make_target_m1_k6_workspace(item, true);
            _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A", "1");
            _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "0");
            auto candidate = make_target_m1_k6_workspace(item, true);
            _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "1");
            auto n16 = make_target_m1_k6_workspace(item, true);
            _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "0");
            require(std::string(control->dispatch_name(item.metadata, 1)) ==
                        "generic_mma_split" &&
                    std::string(candidate->dispatch_name(item.metadata, 1)) ==
                        "target_m1_k6_n32_async_a" &&
                    std::string(n16->dispatch_name(item.metadata, 1)) ==
                        "target_m1_k6_n16_async_a_stream_reduction",
                    "target M1 K6 dispatch ownership mismatch");

            std::vector<std::uint16_t> canary(guarded_values, sentinel);
            std::vector<std::uint16_t> actual(guarded_values);
            std::vector<std::uint16_t> repeated(guarded_values);
            std::vector<std::uint16_t> n16_actual(guarded_values);
            std::vector<std::uint16_t> expected(guarded_values);
            std::vector<std::uint16_t> input_after(item.input.size());
            cuda_check(cudaMemcpy(reference_base, canary.data(),
                                  guarded_values * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target M1 K6 reference canary upload");
            control->forward(item.weights, item.metadata, input_device,
                             reference_device, 1);
            cuda_check(cudaMemcpy(expected.data(), reference_base,
                                  guarded_values * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target M1 K6 reference download");
            cuda_check(cudaMemcpy(output_base, canary.data(),
                                  guarded_values * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target M1 K6 candidate canary upload");
            candidate->forward(item.weights, item.metadata, input_device,
                               output_device, 1);
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  guarded_values * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target M1 K6 candidate download");
            cuda_check(cudaMemcpy(output_base, canary.data(),
                                  guarded_values * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target M1 K6 repeat canary upload");
            candidate->forward(item.weights, item.metadata, input_device,
                               output_device, 1);
            cuda_check(cudaMemcpy(repeated.data(), output_base,
                                  guarded_values * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target M1 K6 repeat download");
            cuda_check(cudaMemcpy(input_after.data(), input_device,
                                  item.input.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target M1 K6 input immutability download");
            cuda_check(cudaMemcpy(output_base, canary.data(),
                                  guarded_values * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target M1 K6 N16 canary upload");
            n16->forward(item.weights, item.metadata, input_device,
                         output_device, 1);
            cuda_check(cudaMemcpy(n16_actual.data(), output_base,
                                  guarded_values * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target M1 K6 N16 output download");
            const bool guards_ok =
                std::all_of(actual.begin(), actual.begin() + guard,
                            [](std::uint16_t v) { return v == sentinel; }) &&
                std::all_of(actual.begin() + guard + output_values, actual.end(),
                            [](std::uint16_t v) { return v == sentinel; });
            const bool finite = std::all_of(
                actual.begin() + guard, actual.begin() + guard + output_values,
                [](std::uint16_t bits) {
                    return std::isfinite(half_to_float(bits));
                });
            std::size_t mismatches = 0;
            std::size_t n16_mismatches = 0;
            for (std::size_t i = 0; i < output_values; ++i)
            {
                mismatches += actual[guard + i] != expected[guard + i];
                n16_mismatches += n16_actual[guard + i] != expected[guard + i];
            }
            require(actual == repeated && guards_ok && finite && mismatches == 0 &&
                        n16_mismatches == 0 && n16_actual.front() == sentinel &&
                        n16_actual.back() == sentinel && input_after == item.input,
                    "target M1 K6 exact differential failed");

            auto changed = item.input;
            const auto changed_position = std::find_if(
                changed.begin(), changed.end(),
                [](std::uint16_t bits) { return (bits & 0x7fffU) != 0U; });
            require(changed_position != changed.end(),
                    "target M1 K6 has no nonzero changed-input lane");
            *changed_position ^= 0x8000U;
            cuda_check(cudaMemcpy(input_device, changed.data(),
                                  changed.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target M1 K6 changed input upload");
            control->forward(item.weights, item.metadata, input_device,
                             reference_device, 1);
            candidate->forward(item.weights, item.metadata, input_device,
                               output_device, 1);
            cuda_check(cudaMemcpy(expected.data() + guard, reference_device,
                                  output_values * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target M1 K6 changed reference download");
            cuda_check(cudaMemcpy(repeated.data() + guard, output_device,
                                  output_values * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target M1 K6 changed candidate download");
            n16->forward(item.weights, item.metadata, input_device,
                         output_device, 1);
            cuda_check(cudaMemcpy(n16_actual.data() + guard, output_device,
                                  output_values * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target M1 K6 changed N16 download");
            std::size_t changed_mismatches = 0;
            std::size_t changed_n16_mismatches = 0;
            bool changed_output = false;
            for (std::size_t i = 0; i < output_values; ++i) {
                changed_mismatches +=
                    expected[guard + i] != repeated[guard + i];
                changed_n16_mismatches +=
                    expected[guard + i] != n16_actual[guard + i];
                changed_output = changed_output ||
                    repeated[guard + i] != actual[guard + i];
            }
            require(changed_mismatches == 0 && changed_n16_mismatches == 0 &&
                        changed_output,
                    "target M1 K6 changed-input reuse failed");
            cuda_check(cudaMemcpy(input_device, item.input.data(),
                                  item.input.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target M1 K6 timing input restore");

            for (int warmup = 0; warmup < 2; ++warmup) {
                control->forward(item.weights, item.metadata, input_device,
                                 reference_device, 1);
                candidate->forward(item.weights, item.metadata, input_device,
                                   output_device, 1);
            }
            constexpr int repetitions = 16;
            auto measure = [&](Exl3CudaLinearWorkspace& workspace,
                               std::uint16_t* destination) {
                cuda_check(cudaEventRecord(begin),
                           "record target M1 K6 timing begin");
                for (int repetition = 0; repetition < repetitions; ++repetition)
                    workspace.forward(item.weights, item.metadata, input_device,
                                      destination, 1);
                cuda_check(cudaEventRecord(end),
                           "record target M1 K6 timing end");
                cuda_check(cudaEventSynchronize(end),
                           "synchronize target M1 K6 timing end");
                float milliseconds = 0.0F;
                cuda_check(cudaEventElapsedTime(&milliseconds, begin, end),
                           "resolve target M1 K6 timing");
                return static_cast<double>(milliseconds) * 1000.0 /
                    repetitions;
            };
            for (int round = 0; round < 8; ++round) {
                double control_us = 0.0;
                double candidate_us = 0.0;
                const bool ab = (round % 2) == 0;
                if (ab) {
                    control_us = measure(*control, reference_device);
                    candidate_us = measure(*candidate, output_device);
                } else {
                    candidate_us = measure(*candidate, output_device);
                    control_us = measure(*control, reference_device);
                }
                const double gain = 100.0 *
                    (control_us - candidate_us) / control_us;
                timing << capture << ',' << item.layer << ',' << item.operation << ','
                       << item.metadata.in_features << ','
                       << item.metadata.out_features << ',' << round << ','
                       << (ab ? "AB" : "BA") << ',' << repetitions << ','
                       << control_us << ',' << candidate_us << ',' << gain << '\n';
                control_total += control_us;
                candidate_total += candidate_us;
                ++timing_records;
            }
            checked_values += output_values;
            n16_checked_values += output_values;
            out << capture << ',' << item.layer << ',' << item.operation << ','
                << item.metadata.in_features << ',' << item.metadata.out_features
                << ',' << item.metadata.K << ','
                << control->dispatch_name(item.metadata, 1) << ','
                << candidate->dispatch_name(item.metadata, 1)
                << ',' << n16->dispatch_name(item.metadata, 1)
                << ",1," << guards_ok << ',' << (input_after == item.input)
                << ',' << finite << ',' << mismatches << ',' << n16_mismatches
                << ",1," << changed_mismatches << ',' << changed_n16_mismatches
                << ',' << changed_output << '\n';
        }
    } catch (...) {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        throw;
    }
    cuda_check(cudaEventDestroy(begin), "destroy target M1 K6 timing begin");
    cuda_check(cudaEventDestroy(end), "destroy target M1 K6 timing end");
    out.flush();
    timing.flush();
    require(out.good() && timing.good(),
            "target M1 K6 qualification output write failed");
    const double aggregate_gain = 100.0 *
        (control_total - candidate_total) / control_total;
    std::cout << "TARGETM1K6QUAL PASS records=" << captured.size()
              << " checked_values=" << checked_values
              << " n16_checked_values=" << n16_checked_values
              << " timing_records=" << timing_records
              << " control_us=" << control_total
              << " candidate_us=" << candidate_total
              << " aggregate_gain_percent=" << aggregate_gain
              << " exact=1 guards=1 repeat=1 changed_input=1 malformed=1"
              << std::endl;
}

void run_fast_same_weights_int8_qualification(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt, bool oscar, std::ostream& out) {
    require(env("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV") == "0",
            "FAST_SAME_WEIGHTS INT8 qualification requires explicit flag0 capture");
    ninfer::test::ScopedEnvironmentRestore restore_environment(kTargetM1K6EnvironmentOptions);
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "0");
    auto captured = run_target_m1_k6_capture(target, draft, prompt, oscar);

    const auto& representative = captured.front();
    for (const char* malformed : {"2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV", malformed);
        bool rejected = false;
        try { auto invalid = make_target_m1_k6_workspace(representative, true); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "FAST_SAME_WEIGHTS INT8 accepted malformed opt-in");
    }

    constexpr std::size_t guard = 128;
    constexpr std::uint16_t sentinel = 0x3555;
    out << "capture,layer,operator,in_features,out_features,finite,guards_ok,"
           "repeat_equal,cosine,nrmse,max_abs\n";
    long double dot = 0.0L;
    long double reference_sq = 0.0L;
    long double candidate_sq = 0.0L;
    long double error_sq = 0.0L;
    double max_abs = 0.0;
    std::size_t checked_values = 0;
    for (std::size_t capture = 0; capture < captured.size(); ++capture) {
        const auto& item = captured[capture];
        const std::size_t output_values =
            static_cast<std::size_t>(item.metadata.out_features);
        const std::size_t guarded_values = output_values + 2u * guard;
        DeviceBuffer input(item.input.size() * sizeof(std::uint16_t));
        DeviceBuffer reference(guarded_values * sizeof(std::uint16_t));
        DeviceBuffer candidate(guarded_values * sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* reference_base = static_cast<std::uint16_t*>(reference.get());
        auto* candidate_base = static_cast<std::uint16_t*>(candidate.get());
        auto* reference_device = reference_base + guard;
        auto* candidate_device = candidate_base + guard;
        const std::vector<std::uint16_t> canary(guarded_values, sentinel);
        std::vector<std::uint16_t> expected(guarded_values);
        std::vector<std::uint16_t> actual(guarded_values);
        std::vector<std::uint16_t> repeated(guarded_values);
        cuda_check(cudaMemcpy(input_device, item.input.data(),
                              item.input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "FAST_SAME_WEIGHTS INT8 input upload");
        cuda_check(cudaMemcpy(reference_base, canary.data(),
                              guarded_values * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "FAST_SAME_WEIGHTS INT8 reference canary upload");
        cuda_check(cudaMemcpy(candidate_base, canary.data(),
                              guarded_values * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "FAST_SAME_WEIGHTS INT8 candidate canary upload");
        _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16", "1");
        _putenv_s("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV", "0");
        auto control = make_target_m1_k6_workspace(item, true);
        _putenv_s("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV", "1");
        auto int8 = make_target_m1_k6_workspace(item, true);
        control->forward(item.weights, item.metadata, input_device,
                         reference_device, 1);
        int8->forward(item.weights, item.metadata, input_device,
                      candidate_device, 1);
        cuda_check(cudaMemcpy(expected.data(), reference_base,
                              guarded_values * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "FAST_SAME_WEIGHTS INT8 reference download");
        cuda_check(cudaMemcpy(actual.data(), candidate_base,
                              guarded_values * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "FAST_SAME_WEIGHTS INT8 candidate download");
        cuda_check(cudaMemcpy(candidate_base, canary.data(),
                              guarded_values * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "FAST_SAME_WEIGHTS INT8 repeat canary upload");
        int8->forward(item.weights, item.metadata, input_device,
                      candidate_device, 1);
        cuda_check(cudaMemcpy(repeated.data(), candidate_base,
                              guarded_values * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "FAST_SAME_WEIGHTS INT8 repeat download");
        const bool guards_ok =
            std::all_of(actual.begin(), actual.begin() + guard,
                        [](std::uint16_t value) { return value == sentinel; }) &&
            std::all_of(actual.begin() + guard + output_values, actual.end(),
                        [](std::uint16_t value) { return value == sentinel; });
        const bool finite = std::all_of(
            actual.begin() + guard, actual.begin() + guard + output_values,
            [](std::uint16_t bits) { return std::isfinite(half_to_float(bits)); });
        long double local_dot = 0.0L;
        long double local_reference_sq = 0.0L;
        long double local_candidate_sq = 0.0L;
        long double local_error_sq = 0.0L;
        double local_max_abs = 0.0;
        for (std::size_t i = 0; i < output_values; ++i) {
            const double reference_value = half_to_float(expected[guard + i]);
            const double candidate_value = half_to_float(actual[guard + i]);
            const double error = candidate_value - reference_value;
            local_dot += reference_value * candidate_value;
            local_reference_sq += reference_value * reference_value;
            local_candidate_sq += candidate_value * candidate_value;
            local_error_sq += error * error;
            local_max_abs = std::max(local_max_abs, std::abs(error));
        }
        const double local_cosine = static_cast<double>(local_dot /
            std::sqrt(local_reference_sq * local_candidate_sq));
        const double local_nrmse = std::sqrt(static_cast<double>(
            local_error_sq / local_reference_sq));
        require(finite && guards_ok && actual == repeated &&
                    local_cosine >= 0.9990 && local_nrmse <= 0.03 &&
                    local_max_abs <= 0.50,
                "FAST_SAME_WEIGHTS INT8 operator numerical gate failed");
        out << capture << ',' << item.layer << ',' << item.operation << ','
            << item.metadata.in_features << ',' << item.metadata.out_features
            << ',' << finite << ',' << guards_ok << ",1," << local_cosine
            << ',' << local_nrmse << ',' << local_max_abs << '\n';
        dot += local_dot;
        reference_sq += local_reference_sq;
        candidate_sq += local_candidate_sq;
        error_sq += local_error_sq;
        max_abs = std::max(max_abs, local_max_abs);
        checked_values += output_values;
    }
    out.flush();
    require(out.good(), "FAST_SAME_WEIGHTS INT8 qualification output write failed");
    const double cosine = static_cast<double>(dot /
        std::sqrt(reference_sq * candidate_sq));
    const double nrmse = std::sqrt(static_cast<double>(error_sq / reference_sq));
    require(cosine >= 0.9990 && nrmse <= 0.03 && max_abs <= 0.50,
            "FAST_SAME_WEIGHTS INT8 aggregate numerical gate failed");
    std::cout << "FASTSAMEWEIGHTSINT8QUAL PASS records=" << captured.size()
              << " checked_values=" << checked_values
              << " cosine=" << cosine << " nrmse=" << nrmse
              << " max_abs=" << max_abs
              << " thresholds=0.9990/0.03/0.50 repeat=1 guards=1 malformed=1"
              << std::endl;
}

void run_target_m1_predecoded_discriminator(
    Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,bool oscar) {
    require(env("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A")=="0",
            "predecoded M1 discriminator requires explicit N32 flag0 capture");
    ninfer::test::ScopedEnvironmentRestore restore_environment(kTargetM1K6EnvironmentOptions);
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS","");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16","0");
    auto captured=run_target_m1_k6_capture(target,draft,prompt,oscar);
    const auto selected=std::find_if(captured.begin(),captured.end(),
        [](const TargetK6CapturedProjection& item) {
            return item.operation=="qkv"&&item.metadata.in_features==5120&&
                item.metadata.out_features==10240;
        });
    require(selected!=captured.end(),"predecoded M1 discriminator missed QKV owner");
    const auto& item=*selected;
    const std::size_t decoded_bytes=static_cast<std::size_t>(
        item.metadata.in_features)*item.metadata.out_features*sizeof(std::uint16_t);
    constexpr std::size_t guard=128;
    constexpr std::uint16_t sentinel=0x3555;
    const std::size_t output_values=static_cast<std::size_t>(
        item.metadata.out_features);
    const std::size_t guarded_values=output_values+2*guard;
    DeviceBuffer input(item.input.size()*sizeof(std::uint16_t));
    DeviceBuffer reference(guarded_values*sizeof(std::uint16_t));
    DeviceBuffer candidate(guarded_values*sizeof(std::uint16_t));
    DeviceBuffer decoded(decoded_bytes);
    auto* input_device=static_cast<std::uint16_t*>(input.get());
    auto* reference_base=static_cast<std::uint16_t*>(reference.get());
    auto* candidate_base=static_cast<std::uint16_t*>(candidate.get());
    auto* decoded_device=static_cast<std::uint16_t*>(decoded.get());
    cuda_check(cudaMemcpy(input_device,item.input.data(),
        item.input.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
        "predecoded M1 input upload");
    std::vector<std::uint16_t> canary(guarded_values,sentinel);
    cuda_check(cudaMemcpy(reference_base,canary.data(),
        guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
        "predecoded M1 reference canary upload");
    cuda_check(cudaMemcpy(candidate_base,canary.data(),
        guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
        "predecoded M1 candidate canary upload");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A","1");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16","1");
    auto workspace=make_target_m1_k6_workspace(item,true);
    require(std::string(workspace->dispatch_name(item.metadata,1))==
        "target_m1_k6_n16_async_a_stream_reduction",
        "predecoded M1 discriminator selected dispatch mismatch");
    cudaEvent_t begin=nullptr,end=nullptr;
    cuda_check(cudaEventCreate(&begin),"create predecoded M1 begin event");
    cuda_check(cudaEventCreate(&end),"create predecoded M1 end event");
    auto cleanup=[&]() {
        if(begin)cudaEventDestroy(begin);
        if(end)cudaEventDestroy(end);
    };
    try {
        cuda_check(cudaEventRecord(begin),"record predecoded M1 setup begin");
        ninfer::exl3::exl3_reconstruct_native_weights_for_test(
            item.weights,item.metadata,decoded_device,decoded_bytes);
        cuda_check(cudaEventRecord(end),"record predecoded M1 setup end");
        cuda_check(cudaEventSynchronize(end),"synchronize predecoded M1 setup");
        float setup_ms=0.0F;
        cuda_check(cudaEventElapsedTime(&setup_ms,begin,end),
            "resolve predecoded M1 setup");
        workspace->forward(item.weights,item.metadata,input_device,
            reference_base+guard,1);
        workspace->forward_predecoded_m1_k6_for_test(
            item.weights,item.metadata,input_device,candidate_base+guard,1,
            decoded_device,decoded_bytes);
        std::vector<std::uint16_t> expected(guarded_values),actual(guarded_values),
            repeated(guarded_values),input_after(item.input.size());
        cuda_check(cudaMemcpy(expected.data(),reference_base,
            guarded_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download predecoded M1 reference");
        cuda_check(cudaMemcpy(actual.data(),candidate_base,
            guarded_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download predecoded M1 candidate");
        cuda_check(cudaMemcpy(candidate_base,canary.data(),
            guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
            "reset predecoded M1 repeat canary");
        workspace->forward_predecoded_m1_k6_for_test(
            item.weights,item.metadata,input_device,candidate_base+guard,1,
            decoded_device,decoded_bytes);
        cuda_check(cudaMemcpy(repeated.data(),candidate_base,
            guarded_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download predecoded M1 repeat");
        cuda_check(cudaMemcpy(input_after.data(),input_device,
            item.input.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download predecoded M1 input");
        std::size_t bit_mismatches=0;
        for(std::size_t i=0;i<output_values;++i)
            if(actual[guard+i]!=expected[guard+i])++bit_mismatches;
        const bool guards_ok=std::all_of(actual.begin(),actual.begin()+guard,
            [](std::uint16_t value){return value==sentinel;})&&
            std::all_of(actual.begin()+guard+output_values,actual.end(),
            [](std::uint16_t value){return value==sentinel;});
        require(bit_mismatches==0&&guards_ok&&actual==repeated&&
            input_after==item.input,
            "predecoded M1 discriminator exactness contract failed");
        for(int warmup=0;warmup<4;++warmup) {
            workspace->forward(item.weights,item.metadata,input_device,
                reference_base+guard,1);
            workspace->forward_predecoded_m1_k6_for_test(
                item.weights,item.metadata,input_device,candidate_base+guard,1,
                decoded_device,decoded_bytes);
        }
        constexpr int repetitions=32,samples=31;
        auto measure=[&](bool predecoded) {
            cuda_check(cudaEventRecord(begin),"record predecoded M1 sample begin");
            for(int repetition=0;repetition<repetitions;++repetition) {
                if(predecoded)
                    workspace->forward_predecoded_m1_k6_for_test(
                        item.weights,item.metadata,input_device,candidate_base+guard,
                        1,decoded_device,decoded_bytes);
                else workspace->forward(item.weights,item.metadata,input_device,
                    reference_base+guard,1);
            }
            cuda_check(cudaEventRecord(end),"record predecoded M1 sample end");
            cuda_check(cudaEventSynchronize(end),
                "synchronize predecoded M1 sample");
            float milliseconds=0.0F;
            cuda_check(cudaEventElapsedTime(&milliseconds,begin,end),
                "resolve predecoded M1 sample");
            return static_cast<double>(milliseconds)*1000.0/repetitions;
        };
        std::vector<double> packed_samples,predecoded_samples;
        packed_samples.reserve(samples);predecoded_samples.reserve(samples);
        for(int sample=0;sample<samples;++sample) {
            if((sample&1)==0) {
                packed_samples.push_back(measure(false));
                predecoded_samples.push_back(measure(true));
            } else {
                predecoded_samples.push_back(measure(true));
                packed_samples.push_back(measure(false));
            }
        }
        std::sort(packed_samples.begin(),packed_samples.end());
        std::sort(predecoded_samples.begin(),predecoded_samples.end());
        const double packed_us=packed_samples[samples/2];
        const double predecoded_us=predecoded_samples[samples/2];
        std::cout<<std::fixed<<std::setprecision(3)
            <<"M1_PREDECODED_DISCRIMINATOR PASS"
            <<" layer="<<item.layer<<" operator="<<item.operation
            <<" in_features="<<item.metadata.in_features
            <<" out_features="<<item.metadata.out_features
            <<" decoded_bytes="<<decoded_bytes
            <<" setup_us="<<static_cast<double>(setup_ms)*1000.0
            <<" packed_median_us="<<packed_us
            <<" predecoded_median_us="<<predecoded_us
            <<" gain_percent="<<(100.0*(packed_us-predecoded_us)/packed_us)
            <<" bit_mismatches="<<bit_mismatches
            <<" guards_ok=1 repeat_equal=1 input_unchanged=1"
            <<std::endl;
    } catch(...) {
        cleanup();
        throw;
    }
    cleanup();
}

void run_target_m1_gate_up_pair_discriminator(
    Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,bool oscar) {
    require(env("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A")=="0",
        "target M1 gate/up pair discriminator requires flag0 capture");
    ninfer::test::ScopedEnvironmentRestore restore_environment(kTargetM1K6EnvironmentOptions);
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS","");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16","0");
    auto captured=run_target_m1_k6_capture(target,draft,prompt,oscar);
    const auto find_owner=[&](const char* operation) {
        return std::find_if(captured.begin(),captured.end(),[&](const auto& item) {
            return item.layer==4&&item.operation==operation&&
                item.metadata.in_features==5120&&item.metadata.out_features==17408;
        });
    };
    const auto gate=find_owner("gate"),up=find_owner("up");
    require(gate!=captured.end()&&up!=captured.end(),
        "target M1 gate/up pair discriminator missed layer-4 owners");
    require(gate->input==up->input,
        "target M1 gate/up pair discriminator inputs differ");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A","1");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16","1");
    auto gate_workspace=make_target_m1_k6_workspace(*gate,true);
    auto up_workspace=make_target_m1_k6_workspace(*up,true);
    require(std::string(gate_workspace->dispatch_name(gate->metadata,1))==
        "target_m1_k6_n16_async_a_stream_reduction"&&
        std::string(up_workspace->dispatch_name(up->metadata,1))==
        "target_m1_k6_n16_async_a_stream_reduction",
        "target M1 gate/up pair selected route mismatch");
    constexpr std::size_t guard=128;
    constexpr std::uint16_t sentinel=0x3555;
    const std::size_t output_values=17408,guarded_values=output_values+2*guard;
    DeviceBuffer input(gate->input.size()*sizeof(std::uint16_t));
    std::array<std::unique_ptr<DeviceBuffer>,4> outputs;
    for(auto& output:outputs)
        output=std::make_unique<DeviceBuffer>(guarded_values*sizeof(std::uint16_t));
    auto* input_device=static_cast<std::uint16_t*>(input.get());
    cuda_check(cudaMemcpy(input_device,gate->input.data(),
        gate->input.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
        "upload target M1 gate/up pair input");
    std::vector<std::uint16_t> canary(guarded_values,sentinel);
    for(auto& output:outputs)
        cuda_check(cudaMemcpy(output->get(),canary.data(),
            guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
            "upload target M1 gate/up pair canary");
    auto* reference_gate=static_cast<std::uint16_t*>(outputs[0]->get())+guard;
    auto* reference_up=static_cast<std::uint16_t*>(outputs[1]->get())+guard;
    auto* candidate_gate=static_cast<std::uint16_t*>(outputs[2]->get())+guard;
    auto* candidate_up=static_cast<std::uint16_t*>(outputs[3]->get())+guard;
    const auto run_separate=[&] {
        gate_workspace->forward(gate->weights,gate->metadata,input_device,
            reference_gate,1);
        up_workspace->forward(up->weights,up->metadata,input_device,reference_up,1);
    };
    const auto run_pair=[&] {
        gate_workspace->forward_target_m1_gate_up_pair_for_test(*up_workspace,
            gate->weights,gate->metadata,up->weights,up->metadata,input_device,
            candidate_gate,candidate_up);
    };
    cudaEvent_t begin=nullptr,end=nullptr;
    cuda_check(cudaEventCreate(&begin),"create target M1 gate/up pair begin");
    cuda_check(cudaEventCreate(&end),"create target M1 gate/up pair end");
    auto cleanup=[&] {if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);};
    try {
        run_separate();run_pair();
        cuda_check(cudaDeviceSynchronize(),"target M1 gate/up pair exactness sync");
        std::array<std::vector<std::uint16_t>,4> host;
        for(std::size_t i=0;i<host.size();++i) {
            host[i].resize(guarded_values);
            cuda_check(cudaMemcpy(host[i].data(),outputs[i]->get(),
                guarded_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                "download target M1 gate/up pair output");
        }
        std::vector<std::uint16_t> input_after(gate->input.size());
        cuda_check(cudaMemcpy(input_after.data(),input_device,
            input_after.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download target M1 gate/up pair input");
        const auto guarded=[&](const auto& values) {
            return std::all_of(values.begin(),values.begin()+guard,
                [](auto value){return value==sentinel;})&&
                std::all_of(values.begin()+guard+output_values,values.end(),
                [](auto value){return value==sentinel;});
        };
        require(host[0]==host[2]&&host[1]==host[3]&&guarded(host[2])&&
            guarded(host[3])&&input_after==gate->input,
            "target M1 gate/up pair exactness contract failed");
        cuda_check(cudaMemcpy(outputs[2]->get(),canary.data(),
            guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
            "reset target M1 gate candidate canary");
        cuda_check(cudaMemcpy(outputs[3]->get(),canary.data(),
            guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
            "reset target M1 up candidate canary");
        run_pair();cuda_check(cudaDeviceSynchronize(),
            "target M1 gate/up pair repeat sync");
        std::array<std::vector<std::uint16_t>,2> repeated;
        for(std::size_t i=0;i<2;++i) {
            repeated[i].resize(guarded_values);
            cuda_check(cudaMemcpy(repeated[i].data(),outputs[i+2]->get(),
                guarded_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                "download target M1 gate/up pair repeat");
        }
        require(repeated[0]==host[2]&&repeated[1]==host[3],
            "target M1 gate/up pair repeat mismatch");
        for(int warmup=0;warmup<4;++warmup){run_separate();run_pair();}
        constexpr int samples=31,repetitions=16;
        auto measure=[&](bool pair) {
            cuda_check(cudaEventRecord(begin),"record target M1 gate/up pair begin");
            for(int i=0;i<repetitions;++i)pair?run_pair():run_separate();
            cuda_check(cudaEventRecord(end),"record target M1 gate/up pair end");
            cuda_check(cudaEventSynchronize(end),"sync target M1 gate/up pair sample");
            float milliseconds=0.0F;
            cuda_check(cudaEventElapsedTime(&milliseconds,begin,end),
                "resolve target M1 gate/up pair sample");
            return static_cast<double>(milliseconds)*1000.0/repetitions;
        };
        std::vector<double> separate_samples,pair_samples;
        for(int sample=0;sample<samples;++sample) {
            if((sample&1)==0){separate_samples.push_back(measure(false));
                pair_samples.push_back(measure(true));}
            else {pair_samples.push_back(measure(true));
                separate_samples.push_back(measure(false));}
        }
        std::sort(separate_samples.begin(),separate_samples.end());
        std::sort(pair_samples.begin(),pair_samples.end());
        const double separate_us=separate_samples[samples/2];
        const double pair_us=pair_samples[samples/2];
        std::cout<<std::fixed<<std::setprecision(3)
            <<"M1_GATE_UP_PAIR_DISCRIMINATOR PASS layer=4 rows=1"
            <<" in_features=5120 out_features=17408"
            <<" separate_median_us="<<separate_us
            <<" pair_median_us="<<pair_us
            <<" gain_percent="<<(100.0*(separate_us-pair_us)/separate_us)
            <<" values="<<(2*output_values)
            <<" exact=1 repeat=1 guards=1 input_unchanged=1"<<std::endl;
    } catch(...) {cleanup();throw;}
    cleanup();
}

void run_target_m1_fast_decode_discriminator(
    Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,bool oscar) {
    require(env("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A")=="0",
        "fast-decode M1 discriminator requires flag0 capture");
    ninfer::test::ScopedEnvironmentRestore restore_environment(kTargetM1K6EnvironmentOptions);
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS","");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16","0");
    auto captured=run_target_m1_k6_capture(target,draft,prompt,oscar);
    const auto selected=std::find_if(captured.begin(),captured.end(),
        [](const auto& item) {
            return item.layer==4&&item.operation=="gate"&&
                item.metadata.in_features==5120&&
                item.metadata.out_features==17408;
        });
    require(selected!=captured.end(),
        "fast-decode M1 discriminator missed layer-4 gate owner");
    const auto& item=*selected;
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A","1");
    _putenv_s("NINFER_EXL3_TARGET_M1_K6_N16","1");
    auto workspace=make_target_m1_k6_workspace(item,true);
    require(std::string(workspace->dispatch_name(item.metadata,1))==
        "target_m1_k6_n16_async_a_stream_reduction",
        "fast-decode M1 discriminator selected route mismatch");
    constexpr std::size_t guard=128;
    constexpr std::uint16_t sentinel=0x3555;
    std::uint64_t checked_values=0;
    for(const auto& owner:captured) {
        auto owner_workspace=make_target_m1_k6_workspace(owner,true);
        const std::size_t owner_values=owner.metadata.out_features;
        const std::size_t owner_guarded=owner_values+2*guard;
        DeviceBuffer owner_input(owner.input.size()*sizeof(std::uint16_t));
        DeviceBuffer owner_reference(owner_guarded*sizeof(std::uint16_t));
        DeviceBuffer owner_candidate(owner_guarded*sizeof(std::uint16_t));
        std::vector<std::uint16_t> owner_canary(owner_guarded,sentinel);
        cuda_check(cudaMemcpy(owner_input.get(),owner.input.data(),
            owner.input.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
            "upload fast-decode owner input");
        for(auto* output:{owner_reference.get(),owner_candidate.get()})
            cuda_check(cudaMemcpy(output,owner_canary.data(),
                owner_guarded*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
                "upload fast-decode owner canary");
        auto* owner_reference_output=
            static_cast<std::uint16_t*>(owner_reference.get())+guard;
        auto* owner_candidate_output=
            static_cast<std::uint16_t*>(owner_candidate.get())+guard;
        owner_workspace->forward(owner.weights,owner.metadata,
            static_cast<std::uint16_t*>(owner_input.get()),
            owner_reference_output,1);
        owner_workspace->forward_fast_decode_m1_k6_for_test(
            owner.weights,owner.metadata,
            static_cast<std::uint16_t*>(owner_input.get()),
            owner_candidate_output,1);
        cuda_check(cudaDeviceSynchronize(),"fast-decode owner exactness sync");
        std::vector<std::uint16_t> owner_expected(owner_guarded),
            owner_actual(owner_guarded),owner_input_after(owner.input.size());
        cuda_check(cudaMemcpy(owner_expected.data(),owner_reference.get(),
            owner_guarded*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download fast-decode owner reference");
        cuda_check(cudaMemcpy(owner_actual.data(),owner_candidate.get(),
            owner_guarded*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download fast-decode owner candidate");
        cuda_check(cudaMemcpy(owner_input_after.data(),owner_input.get(),
            owner.input.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download fast-decode owner input");
        require(owner_expected==owner_actual&&owner_input_after==owner.input,
            "fast-decode all-owner exactness mismatch");
        checked_values+=owner_values;
    }
    const std::size_t output_values=item.metadata.out_features;
    const std::size_t guarded_values=output_values+2*guard;
    DeviceBuffer input(item.input.size()*sizeof(std::uint16_t));
    DeviceBuffer reference(guarded_values*sizeof(std::uint16_t));
    DeviceBuffer candidate(guarded_values*sizeof(std::uint16_t));
    auto* input_device=static_cast<std::uint16_t*>(input.get());
    auto* reference_base=static_cast<std::uint16_t*>(reference.get());
    auto* candidate_base=static_cast<std::uint16_t*>(candidate.get());
    cuda_check(cudaMemcpy(input_device,item.input.data(),
        item.input.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
        "upload fast-decode M1 input");
    std::vector<std::uint16_t> canary(guarded_values,sentinel);
    cuda_check(cudaMemcpy(reference_base,canary.data(),
        guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
        "upload fast-decode M1 reference canary");
    cuda_check(cudaMemcpy(candidate_base,canary.data(),
        guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
        "upload fast-decode M1 candidate canary");
    const auto run_selected=[&] {
        workspace->forward(item.weights,item.metadata,input_device,
            reference_base+guard,1);
    };
    const auto run_fast=[&] {
        workspace->forward_fast_decode_m1_k6_for_test(
            item.weights,item.metadata,input_device,candidate_base+guard,1);
    };
    cudaEvent_t begin=nullptr,end=nullptr;
    cuda_check(cudaEventCreate(&begin),"create fast-decode M1 begin");
    cuda_check(cudaEventCreate(&end),"create fast-decode M1 end");
    auto cleanup=[&] {if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);};
    try {
        run_selected();run_fast();cuda_check(cudaDeviceSynchronize(),
            "fast-decode M1 exactness sync");
        std::vector<std::uint16_t> expected(guarded_values),actual(guarded_values),
            repeated(guarded_values),input_after(item.input.size());
        cuda_check(cudaMemcpy(expected.data(),reference_base,
            guarded_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download fast-decode M1 reference");
        cuda_check(cudaMemcpy(actual.data(),candidate_base,
            guarded_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download fast-decode M1 candidate");
        cuda_check(cudaMemcpy(candidate_base,canary.data(),
            guarded_values*sizeof(std::uint16_t),cudaMemcpyHostToDevice),
            "reset fast-decode M1 candidate canary");
        run_fast();cuda_check(cudaDeviceSynchronize(),
            "fast-decode M1 repeat sync");
        cuda_check(cudaMemcpy(repeated.data(),candidate_base,
            guarded_values*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download fast-decode M1 repeat");
        cuda_check(cudaMemcpy(input_after.data(),input_device,
            item.input.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
            "download fast-decode M1 input");
        const bool guards_ok=std::all_of(actual.begin(),actual.begin()+guard,
            [](auto value){return value==sentinel;})&&
            std::all_of(actual.begin()+guard+output_values,actual.end(),
            [](auto value){return value==sentinel;});
        require(expected==actual&&actual==repeated&&guards_ok&&
            input_after==item.input,
            "fast-decode M1 exactness contract failed");
        for(int warmup=0;warmup<4;++warmup){run_selected();run_fast();}
        constexpr int repetitions=32,samples=31;
        auto measure=[&](bool fast) {
            cuda_check(cudaEventRecord(begin),"record fast-decode M1 begin");
            for(int i=0;i<repetitions;++i)fast?run_fast():run_selected();
            cuda_check(cudaEventRecord(end),"record fast-decode M1 end");
            cuda_check(cudaEventSynchronize(end),"sync fast-decode M1 sample");
            float milliseconds=0.0F;
            cuda_check(cudaEventElapsedTime(&milliseconds,begin,end),
                "resolve fast-decode M1 sample");
            return static_cast<double>(milliseconds)*1000.0/repetitions;
        };
        std::vector<double> selected_samples,fast_samples;
        for(int sample=0;sample<samples;++sample) {
            if((sample&1)==0){selected_samples.push_back(measure(false));
                fast_samples.push_back(measure(true));}
            else {fast_samples.push_back(measure(true));
                selected_samples.push_back(measure(false));}
        }
        std::sort(selected_samples.begin(),selected_samples.end());
        std::sort(fast_samples.begin(),fast_samples.end());
        const double selected_us=selected_samples[samples/2];
        const double fast_us=fast_samples[samples/2];
        std::cout<<std::fixed<<std::setprecision(3)
            <<"M1_FAST_DECODE_DISCRIMINATOR PASS layer="<<item.layer
            <<" operation="<<item.operation
            <<" in_features="<<item.metadata.in_features
            <<" out_features="<<item.metadata.out_features
            <<" selected_median_us="<<selected_us
            <<" fast_median_us="<<fast_us
            <<" gain_percent="<<(100.0*(selected_us-fast_us)/selected_us)
            <<" values="<<output_values
            <<" owners="<<captured.size()
            <<" checked_values="<<checked_values
            <<" exact=1 repeat=1 guards=1 input_unchanged=1"<<std::endl;
    } catch(...) {cleanup();throw;}
    cleanup();
}
