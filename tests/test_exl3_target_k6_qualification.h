#pragma once

// Included by the acceptance harness after the independent H6/K6 scalar
// oracle helpers.  This checks only the explicitly admitted target
// continuation gate/up K6 candidate on the 94 real represented inputs.

class TargetK6EnvironmentRestore {
public:
    TargetK6EnvironmentRestore()
        : candidate_(env("NINFER_EXL3_TARGET_GATEUP_SMALL_M")),
          async_a_(env("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A")),
          n16_(env("NINFER_EXL3_TARGET_GATEUP_K6_N16")),
          splits_(env("NINFER_EXL3_GENERIC_SPLITS")) {}
    ~TargetK6EnvironmentRestore() {
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", candidate_.c_str());
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A", async_a_.c_str());
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_N16", n16_.c_str());
        _putenv_s("NINFER_EXL3_GENERIC_SPLITS", splits_.c_str());
    }
private:
    std::string candidate_;
    std::string async_a_;
    std::string n16_;
    std::string splits_;
};

void run_target_k6_qualification(Exl3TextModel& target,
                                 Exl3Dflash2DraftModel& draft,
                                 const std::vector<std::int64_t>& prompt,
                                 bool oscar,
                                 std::ostream& out) {
    require(prompt.size() == 512, "target K6 qualification requires ctx512");
    require(oscar, "target K6 qualification requires canonical OSCAR");
    require(env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "0",
            "target K6 real capture must start from explicit flag0 baseline");
    TargetK6EnvironmentRestore restore_environment;
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");

    auto captured = run_target_k6_capture(
        target, draft, prompt, std::filesystem::path{}, oscar);
    require(captured.size() == 94, "target K6 qualification capture count mismatch");

    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    constexpr Admission admit = Admission::target_continuation_gate_up;
    constexpr std::size_t guard = 128;
    constexpr std::uint16_t sentinel = 0x3555;
    constexpr int out_features = 17408;
    const std::vector<int> oracle_groups = {0, 3, 4, 67, 68, 135};

    // Dispatch ownership and construction-time policy latching are checked
    // before any numerical work.  The same shape stays generic without both
    // target ownership and explicit continuation gate/up admission.  Preserve
    // the parent route's historical nonzero-enable flag contract; the new
    // async-A sub-route is the exact 0/1 flag being qualified here.
    const auto& representative = captured.front();
    const std::string requested_n16 =
        env("NINFER_EXL3_TARGET_GATEUP_K6_N16");
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "1");
    Exl3CudaLinearWorkspace enabled(kHidden, out_features, 8, false, true);
    Exl3CudaLinearWorkspace generic_owner(kHidden, out_features, 8);
    const std::string expected_dispatch =
        env("NINFER_EXL3_TARGET_GATEUP_K6_N16") == "1"
            ? "target_gateup_k6_n16_async_a_stream_reduction"
        : env("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A") == "1"
            ? "target_gateup_k6_small_m_async_a"
            : "target_gateup_small_m_mma_split";
    require(std::string(enabled.dispatch_name(representative.metadata, 8, admit)) ==
                expected_dispatch,
            "target K6 owned/admitted M8 dispatch missing");
    require(std::string(enabled.dispatch_name(representative.metadata, 8)) ==
                "generic_tile" &&
                std::string(generic_owner.dispatch_name(
                    representative.metadata, 8, admit)) == "generic_tile",
            "target K6 candidate escaped explicit call-site ownership");
    require(std::string(enabled.dispatch_name(representative.metadata, 1, admit)) ==
                "generic_mma_split" &&
                std::string(enabled.dispatch_name(representative.metadata, 9, admit)) ==
                "generic_tile",
            "target K6 candidate changed M1 or admitted M9");
    Exl3CudaLinearWorkspace owner_max4(kHidden, out_features, 4, false, true);
    require(!owner_max4.target_gateup_small_m_candidate(
                representative.metadata, 5, admit),
            "target K6 candidate admitted rows beyond workspace capacity");
    auto mismatched_dimensions = representative.metadata;
    mismatched_dimensions.out_features = out_features + 128;
    require(!enabled.target_gateup_small_m_candidate(
                mismatched_dimensions, 8, admit),
            "target K6 candidate admitted mismatched metadata dimensions");
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "0");
    Exl3CudaLinearWorkspace off(kHidden, out_features, 8, false, true);
    require(std::string(off.dispatch_name(representative.metadata, 8, admit)) ==
                "generic_tile",
            "target K6 parent route ignored explicit flag0");
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "1");
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_N16", "0");
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A", "0");
    Exl3CudaLinearWorkspace synchronous(kHidden, out_features, 8, false, true);
    require(std::string(synchronous.dispatch_name(
                representative.metadata, 8, admit)) ==
                "target_gateup_small_m_mma_split",
            "target K6 async-A flag0 changed the parent route");
    for (const char* malformed : {"2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A", malformed);
        bool rejected = false;
        try {
            Exl3CudaLinearWorkspace invalid(kHidden, out_features, 8, false, true);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "target K6 async-A accepted malformed opt-in value");
    }
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A", "1");
    for (const char* malformed : {"2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_N16", malformed);
        bool rejected = false;
        try {
            Exl3CudaLinearWorkspace invalid(kHidden, out_features, 8, false, true);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "target K6 N16 accepted malformed opt-in value");
    }
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_N16", requested_n16.c_str());
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A", "1");
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "0");
    Exl3CudaLinearWorkspace latched_off(kHidden, out_features, 8, false, true);
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "1");
    require(std::string(latched_off.dispatch_name(
                representative.metadata, 8, admit)) == "generic_tile" &&
                std::string(enabled.dispatch_name(
                    representative.metadata, 8, admit)) ==
                    expected_dispatch,
            "target K6 workspace did not latch construction policy");
    auto unrelated = representative.metadata;
    unrelated.K = 5;
    const std::string unrelated_ordinary = enabled.dispatch_name(unrelated, 8);
    const std::string unrelated_admitted =
        enabled.dispatch_name(unrelated, 8, admit);
    require(unrelated_admitted == unrelated_ordinary &&
                unrelated_admitted != "target_gateup_small_m_mma_split",
            "target K6 admission changed the existing K5 shape dispatch");

    out << "layer,operator,M,dispatch,repeat_equal,guards_ok,finite,"
           "bit_mismatches,changed_input,oracle_values,oracle_max_abs,"
           "oracle_rel_l2,oracle_max_error_over_bound,oracle_max_bound,"
           "oracle_fp64_bound,oracle_worst_group,oracle_worst_row,"
           "oracle_worst_column,oracle_within_bound\n";
    std::size_t checked_values = 0;
    std::size_t oracle_values = 0;
    double oracle_max_abs = 0.0;
    double oracle_max_ratio = 0.0;

    for (const auto& item : captured) {
        require(item.metadata.K == 6 && item.metadata.mul1 && !item.metadata.mcg &&
                    !item.metadata.has_bias && item.metadata.in_features == kHidden &&
                    item.metadata.out_features == out_features,
                "target K6 qualification metadata mismatch");
        const std::size_t capacity = 8u * out_features;
        const std::size_t guarded_elements = capacity + 2u * guard;
        DeviceBuffer input(item.input.size() * sizeof(std::uint16_t));
        DeviceBuffer output(guarded_elements * sizeof(std::uint16_t));
        DeviceBuffer reference(static_cast<std::size_t>(out_features) *
                               sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* output_base = static_cast<std::uint16_t*>(output.get());
        auto* output_device = output_base + guard;
        auto* reference_device = static_cast<std::uint16_t*>(reference.get());
        cuda_check(cudaMemcpy(input_device, item.input.data(),
                              item.input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target K6 qualification input upload");
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "1");
        Exl3CudaLinearWorkspace candidate(kHidden, out_features, 8, false, true);
        Exl3CudaLinearWorkspace m1_reference(kHidden, out_features, 1);
        std::vector<std::uint16_t> initial(guarded_elements, sentinel);
        std::vector<std::uint16_t> actual(guarded_elements);
        std::vector<std::uint16_t> repeated(guarded_elements);
        std::vector<std::uint16_t> row_reference(out_features);
        std::vector<std::uint16_t> m8_actual;

        for (int m = 2; m <= 8; ++m) {
            require(std::string(candidate.dispatch_name(item.metadata, m, admit)) ==
                        expected_dispatch,
                    "target K6 candidate failed M2..8 dispatch");
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target K6 qualification canary upload");
            candidate.forward(item.weights, item.metadata, input_device,
                              output_device, m, nullptr, admit);
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target K6 qualification output download");
            candidate.forward(item.weights, item.metadata, input_device,
                              output_device, m, nullptr, admit);
            cuda_check(cudaMemcpy(repeated.data(), output_base,
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target K6 qualification repeat download");
            const bool repeat_equal = actual == repeated;
            const bool guards_ok =
                std::all_of(actual.begin(), actual.begin() + guard,
                            [](std::uint16_t v) { return v == sentinel; }) &&
                std::all_of(actual.begin() + guard +
                                static_cast<std::size_t>(m) * out_features,
                            actual.end(),
                            [](std::uint16_t v) { return v == sentinel; });
            bool finite = true;
            std::size_t mismatches = 0;
            for (int row = 0; row < m; ++row) {
                m1_reference.forward(
                    item.weights, item.metadata,
                    input_device + static_cast<std::size_t>(row) * kHidden,
                    reference_device, 1);
                cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                      row_reference.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost),
                           "target K6 M1 reference download");
                const auto offset = guard +
                    static_cast<std::size_t>(row) * out_features;
                for (int column = 0; column < out_features; ++column) {
                    const auto value = actual[offset + static_cast<std::size_t>(column)];
                    finite = finite && std::isfinite(half_to_float(value));
                    mismatches += value != row_reference[static_cast<std::size_t>(column)];
                }
            }
            H6OracleStats oracle{};
            if (m == 8) {
                m8_actual = actual;
                oracle = target_k6_check_oracle_groups(
                    item.weights, item.metadata, item.input, actual, guard,
                    oracle_groups);
                oracle_values += oracle.values;
                oracle_max_abs = std::max(oracle_max_abs, oracle.max_abs);
                oracle_max_ratio = std::max(oracle_max_ratio, oracle.max_ratio);
                require(oracle.within_bound,
                        "target K6 candidate exceeded independent FP64 bound");
            }
            checked_values += static_cast<std::size_t>(m) * out_features;
            out << item.layer << ',' << item.operation << ',' << m << ','
                << candidate.dispatch_name(item.metadata, m, admit) << ','
                << repeat_equal << ',' << guards_ok << ',' << finite << ','
                << mismatches << ",0,";
            if (m == 8) {
                out << oracle.values << ',' << oracle.max_abs << ','
                    << (oracle.norm_sq > 0.0
                            ? std::sqrt(oracle.error_sq / oracle.norm_sq)
                            : 0.0)
                    << ',' << oracle.max_ratio << ',' << oracle.max_bound << ','
                    << oracle.max_fp64_bound << ',' << oracle.worst_group << ','
                    << oracle.worst_row << ',' << oracle.worst_column << ','
                    << oracle.within_bound << '\n';
            } else {
                out << "0,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";
            }
            require(repeat_equal && guards_ok && finite && mismatches == 0,
                    "target K6 candidate failed exact M1 differential gate");
        }

        // Reuse the same candidate after varying M, with a bitwise changed but
        // still finite represented input.  Compare its full M3 output to M1.
        auto changed = item.input;
        for (auto& bits : changed) bits ^= 0x8000u;
        require(std::all_of(changed.begin(), changed.end(), [](std::uint16_t bits) {
                    return std::isfinite(half_to_float(bits));
                }),
                "target K6 changed input became nonfinite");
        cuda_check(cudaMemcpy(input_device, changed.data(),
                              changed.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target K6 changed input upload");
        cuda_check(cudaMemcpy(output_base, initial.data(),
                              guarded_elements * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target K6 changed-input canary upload");
        candidate.forward(item.weights, item.metadata, input_device,
                          output_device, 3, nullptr, admit);
        cuda_check(cudaMemcpy(actual.data(), output_base,
                              guarded_elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "target K6 changed output download");
        const bool changed_guards =
            std::all_of(actual.begin(), actual.begin() + guard,
                        [](std::uint16_t v) { return v == sentinel; }) &&
            std::all_of(actual.begin() + guard + 3u * out_features,
                        actual.end(),
                        [](std::uint16_t v) { return v == sentinel; });
        bool changed_finite = true;
        for (std::size_t i = guard; i < guard + 3u * out_features; ++i)
            changed_finite = changed_finite &&
                std::isfinite(half_to_float(actual[i]));
        std::size_t changed_mismatches = 0;
        for (int row = 0; row < 3; ++row) {
            m1_reference.forward(
                item.weights, item.metadata,
                input_device + static_cast<std::size_t>(row) * kHidden,
                reference_device, 1);
            cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                  row_reference.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target K6 changed M1 reference download");
            const auto offset = guard + static_cast<std::size_t>(row) * out_features;
            for (int column = 0; column < out_features; ++column)
                changed_mismatches +=
                    actual[offset + static_cast<std::size_t>(column)] !=
                    row_reference[static_cast<std::size_t>(column)];
        }
        require(changed_guards && changed_finite && changed_mismatches == 0 &&
                    !std::equal(actual.begin() + guard,
                                actual.begin() + guard + 3u * out_features,
                                m8_actual.begin() + guard),
                "target K6 changed-input reuse failed or did not change output");
        out << item.layer << ',' << item.operation << ",3,"
            << candidate.dispatch_name(item.metadata, 3, admit)
            << ",NA," << changed_guards << ',' << changed_finite
            << ",0,1,0,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";

        // Invalid row counts reject before launch and leave the canary intact.
        for (const int invalid_rows : {0, 9}) {
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target K6 invalid-row canary upload");
            bool rejected = false;
            try {
                candidate.forward(item.weights, item.metadata, input_device,
                                  output_device, invalid_rows, nullptr, admit);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target K6 invalid-row canary download");
            require(rejected && actual == initial,
                    "target K6 invalid row did not reject before launch");
        }
    }
    out.flush();
    require(out.good(), "target K6 qualification CSV write failed");
    const std::size_t expected_oracle_values =
        94u * 8u * oracle_groups.size() * 128u;
    require(oracle_values == expected_oracle_values,
            "target K6 oracle coverage count mismatch");

    // Optional bounded causal stage screen.  It uses the same 94 represented
    // captures and weights as the exactness proof, but directly times the
    // preserved synchronous-A route against async-A with CUDA events on one
    // stream.  Each M2..8 cell alternates AB/BA to limit order drift.
    const std::string timing_path = env("NINFER_E5A4_TARGET_K6_ASYNC_A_TIMING_OUT");
    if (!timing_path.empty()) {
        std::ofstream timing(timing_path, std::ios::trunc);
        require(timing.good(), "cannot create target K6 async-A timing CSV");
        timing << "capture,layer,operator,M,order,repetitions,control_us,candidate_us\n";
        cudaEvent_t begin = nullptr;
        cudaEvent_t end = nullptr;
        cuda_check(cudaEventCreate(&begin), "create target K6 timing begin event");
        cuda_check(cudaEventCreate(&end), "create target K6 timing end event");
        constexpr int repetitions = 16;
        std::size_t timing_cells = 0;
        try {
            for (std::size_t capture = 0; capture < captured.size(); ++capture) {
                const auto& item = captured[capture];
                DeviceBuffer input(item.input.size() * sizeof(std::uint16_t));
                DeviceBuffer output(static_cast<std::size_t>(8) * out_features *
                                    sizeof(std::uint16_t));
                auto* input_device = static_cast<std::uint16_t*>(input.get());
                auto* output_device = static_cast<std::uint16_t*>(output.get());
                cuda_check(cudaMemcpy(input_device, item.input.data(),
                                      item.input.size() * sizeof(std::uint16_t),
                                      cudaMemcpyHostToDevice),
                           "target K6 timing input upload");
                _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "1");
                _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A", "0");
                Exl3CudaLinearWorkspace control(kHidden, out_features, 8, false, true);
                _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A", "1");
                Exl3CudaLinearWorkspace candidate(kHidden, out_features, 8, false, true);
                for (int m = 2; m <= 8; ++m) {
                    require(std::string(control.dispatch_name(
                                item.metadata, m, admit)) ==
                                "target_gateup_small_m_mma_split" &&
                                std::string(candidate.dispatch_name(
                                item.metadata, m, admit)) ==
                                "target_gateup_k6_small_m_async_a",
                            "target K6 timing dispatch mismatch");
                    auto measure = [&](Exl3CudaLinearWorkspace& workspace) {
                        workspace.forward(item.weights, item.metadata, input_device,
                                          output_device, m, nullptr, admit);
                        cuda_check(cudaDeviceSynchronize(),
                                   "warm target K6 timing candidate");
                        cuda_check(cudaEventRecord(begin),
                                   "record target K6 timing begin");
                        for (int repetition = 0; repetition < repetitions; ++repetition)
                            workspace.forward(item.weights, item.metadata, input_device,
                                              output_device, m, nullptr, admit);
                        cuda_check(cudaEventRecord(end),
                                   "record target K6 timing end");
                        cuda_check(cudaEventSynchronize(end),
                                   "synchronize target K6 timing end");
                        float milliseconds = 0.0f;
                        cuda_check(cudaEventElapsedTime(&milliseconds, begin, end),
                                   "resolve target K6 timing events");
                        return static_cast<double>(milliseconds) * 1000.0 /
                               repetitions;
                    };
                    double control_us = 0.0;
                    double candidate_us = 0.0;
                    const bool control_first = ((capture * 7u +
                                                 static_cast<std::size_t>(m - 2)) % 2u) == 0;
                    if (control_first) {
                        control_us = measure(control);
                        candidate_us = measure(candidate);
                    } else {
                        candidate_us = measure(candidate);
                        control_us = measure(control);
                    }
                    timing << capture << ',' << item.layer << ',' << item.operation
                           << ',' << m << ',' << (control_first ? "AB" : "BA")
                           << ',' << repetitions << ',' << control_us << ','
                           << candidate_us << '\n';
                    ++timing_cells;
                }
            }
        } catch (...) {
            if (begin) cudaEventDestroy(begin);
            if (end) cudaEventDestroy(end);
            throw;
        }
        cuda_check(cudaEventDestroy(begin), "destroy target K6 timing begin event");
        cuda_check(cudaEventDestroy(end), "destroy target K6 timing end event");
        timing.flush();
        require(timing.good() && timing_cells == 658,
                "target K6 async-A timing coverage mismatch");
        std::cout << "TARGETK6_ASYNC_A_TIMING PASS captures=94 M=2..8 cells="
                  << timing_cells << " repetitions=" << repetitions << '\n';
    }

    // Candidate-specific in-process allocation/lifetime discriminator.  Warm
    // both instantiated kernels first, then repeatedly construct, use, and
    // destroy identical workspaces while sampling physical device free bytes
    // at identical boundaries.  The candidate changes dynamic shared bytes
    // only; no endpoint may lose more than the existing 1 MiB gate.
    const std::string lifecycle_path =
        env("NINFER_E5A4_TARGET_K6_ASYNC_A_LIFECYCLE_OUT");
    if (!lifecycle_path.empty()) {
        std::ofstream lifecycle(lifecycle_path, std::ios::trunc);
        require(lifecycle.good(),
                "cannot create target K6 async-A lifecycle CSV");
        lifecycle << "pair,launch,arm,free_before,free_live,free_after,"
                     "live_used_bytes,endpoint_loss_bytes,gate\n";
        const auto& item = captured.front();
        auto run_cycle = [&](int pair, int launch, const char* arm,
                             const char* async_value, bool record) {
            std::size_t free_before = 0;
            std::size_t total_before = 0;
            cuda_check(cudaMemGetInfo(&free_before, &total_before),
                       "query target K6 lifecycle before");
            std::size_t free_live = 0;
            std::size_t total_live = 0;
            {
                DeviceBuffer input(item.input.size() * sizeof(std::uint16_t));
                DeviceBuffer output(static_cast<std::size_t>(8) * out_features *
                                    sizeof(std::uint16_t));
                auto* input_device = static_cast<std::uint16_t*>(input.get());
                auto* output_device = static_cast<std::uint16_t*>(output.get());
                cuda_check(cudaMemcpy(input_device, item.input.data(),
                                      item.input.size() * sizeof(std::uint16_t),
                                      cudaMemcpyHostToDevice),
                           "target K6 lifecycle input upload");
                _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "1");
                _putenv_s("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A", async_value);
                Exl3CudaLinearWorkspace workspace(
                    kHidden, out_features, 8, false, true);
                workspace.forward(item.weights, item.metadata, input_device,
                                  output_device, 8, nullptr, admit);
                cuda_check(cudaDeviceSynchronize(),
                           "synchronize target K6 lifecycle live workspace");
                cuda_check(cudaMemGetInfo(&free_live, &total_live),
                           "query target K6 lifecycle live");
                require(total_live == total_before,
                        "target K6 lifecycle total-memory identity changed");
            }
            cuda_check(cudaDeviceSynchronize(),
                       "synchronize target K6 lifecycle teardown");
            std::size_t free_after = 0;
            std::size_t total_after = 0;
            cuda_check(cudaMemGetInfo(&free_after, &total_after),
                       "query target K6 lifecycle after");
            require(total_after == total_before,
                    "target K6 lifecycle total-memory endpoint changed");
            const std::size_t live_used =
                free_before > free_live ? free_before - free_live : 0;
            const std::size_t endpoint_loss =
                free_before > free_after ? free_before - free_after : 0;
            const bool gate = endpoint_loss <= 1048576u;
            if (record) {
                lifecycle << pair << ',' << launch << ',' << arm << ','
                          << free_before << ',' << free_live << ',' << free_after
                          << ',' << live_used << ',' << endpoint_loss << ','
                          << gate << '\n';
            }
            require(gate, "target K6 lifecycle exceeded 1 MiB endpoint gate");
        };
        run_cycle(-1, 0, "control_warm", "0", false);
        run_cycle(-1, 1, "candidate_warm", "1", false);
        for (int pair = 0; pair < 8; ++pair) {
            const bool control_first = (pair % 2) == 0;
            if (control_first) {
                run_cycle(pair, 0, "control", "0", true);
                run_cycle(pair, 1, "candidate", "1", true);
            } else {
                run_cycle(pair, 0, "candidate", "1", true);
                run_cycle(pair, 1, "control", "0", true);
            }
        }
        lifecycle.flush();
        require(lifecycle.good(), "target K6 lifecycle CSV write failed");
        std::cout << "TARGETK6_ASYNC_A_LIFECYCLE PASS pairs=8 cycles=16 "
                     "endpoint_gate_bytes=1048576\n";
    }

    std::cout << "TARGETK6QUAL_ORACLE PASS pairs=94 rows=8 groups=6 values="
              << oracle_values << " max_abs=" << oracle_max_abs
              << " max_error_over_bound=" << oracle_max_ratio << '\n';
    std::cout << "TARGETK6QUAL PASS pairs=94 M=2..8 checked_values="
              << checked_values
              << " exact_m1=1 repeat=1 guards=1 finite=1 changed_input=1"
              << std::endl;
}
