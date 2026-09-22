#pragma once

struct PrefillK6GateupWarpgroupQualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit = Admission::target_wide_prefill;
    static constexpr std::size_t guard = 128;
    static constexpr int max_rows = 128;
    static constexpr int intermediate = 17408;

    std::filesystem::path directory;
    bool checked = false;
    int observations = 0;

    explicit PrefillK6GateupWarpgroupQualification(
        const std::filesystem::path& path) : directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),
                "K6 gate/up warp-group output must be new");
        std::filesystem::create_directories(path);
    }

    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& observation,
        void* user) {
        static_cast<PrefillK6GateupWarpgroupQualification*>(user)->check(
            observation);
    }

    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        ++observations;
        if (checked) return;
        checked = true;
        require(x.rows == max_rows && x.metadata.K == 6 &&
                    x.metadata.in_features == kHidden &&
                    x.metadata.out_features == intermediate,
                "K6 gate/up warp-group observed extent");
        cuda_check(cudaStreamSynchronize(x.stream),
                   "K6 gate/up warp-group input ready");

        _putenv_s("NINFER_EXL3_PREFILL_K6_GATEUP_WARPGROUP_ASYNC", "0");
        Exl3CudaLinearWorkspace baseline(
            kHidden, intermediate, max_rows, false,
            true, false, false, false, true);
        _putenv_s("NINFER_EXL3_PREFILL_K6_GATEUP_WARPGROUP_ASYNC", "1");
        Exl3CudaLinearWorkspace candidate(
            kHidden, intermediate, max_rows, false,
            true, false, false, false, true);
        _putenv_s("NINFER_EXL3_PREFILL_K6_GATEUP_WARPGROUP_ASYNC", "0");

        require(std::string(baseline.dispatch_name(
                    x.metadata, max_rows, admit)) ==
                    "target_wide_prefill_staged",
                "K6 gate/up warp-group baseline route");
        require(std::string(candidate.dispatch_name(
                    x.metadata, max_rows, admit)) ==
                    "target_k6_gateup_warpgroup_async",
                "K6 gate/up warp-group candidate route");
        require(!candidate.target_k6_gateup_warpgroup_async_candidate(
                    x.metadata, 16, admit) &&
                    candidate.target_k6_gateup_warpgroup_async_candidate(
                        x.metadata, 17, admit) &&
                    !candidate.target_k6_gateup_warpgroup_async_candidate(
                        x.metadata, max_rows + 1, admit) &&
                    !candidate.target_k6_gateup_warpgroup_async_candidate(
                        x.metadata, max_rows, Admission::target_prefill_gate_up),
                "K6 gate/up warp-group admission boundary");

        constexpr std::size_t extent =
            static_cast<std::size_t>(max_rows) * intermediate;
        std::vector<std::uint16_t> initial(extent + 2 * guard, 0x3555);
        std::vector<std::uint16_t> control(initial), trial(initial);
        DeviceBuffer a(initial.size() * sizeof(std::uint16_t));
        DeviceBuffer b(initial.size() * sizeof(std::uint16_t));
        auto* pa = static_cast<std::uint16_t*>(a.get()) + guard;
        auto* pb = static_cast<std::uint16_t*>(b.get()) + guard;
        std::ofstream csv(directory / "differential.csv");
        csv << "rows,exact,control_route,candidate_route,candidate_calls\n";

        std::uint64_t expected_calls = 0;
        for (int rows : {17, 31, 32, 127, 128}) {
            cuda_check(cudaMemcpy(a.get(), initial.data(),
                                  initial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "K6 gate/up warp-group control guards");
            cuda_check(cudaMemcpy(b.get(), initial.data(),
                                  initial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "K6 gate/up warp-group candidate guards");
            baseline.forward(x.weights, x.metadata, x.input, pa, rows,
                             x.stream, admit);
            candidate.forward(x.weights, x.metadata, x.input, pb, rows,
                              x.stream, admit);
            cuda_check(cudaStreamSynchronize(x.stream),
                       "K6 gate/up warp-group compare ready");
            cuda_check(cudaMemcpy(control.data(), a.get(),
                                  control.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "K6 gate/up warp-group control copy");
            cuda_check(cudaMemcpy(trial.data(), b.get(),
                                  trial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "K6 gate/up warp-group candidate copy");
            require(control == trial,
                    "K6 gate/up warp-group exact differential mismatch M=" +
                        std::to_string(rows));
            ++expected_calls;
            require(candidate.k6_gateup_warpgroup_async_calls() ==
                        expected_calls &&
                        baseline.k6_gateup_warpgroup_async_calls() == 0,
                    "K6 gate/up warp-group dispatch counter");
            csv << rows << ",1,target_wide_prefill_staged,"
                << "target_k6_gateup_warpgroup_async," << expected_calls
                << '\n';
        }
        require(csv.good(), "K6 gate/up warp-group differential evidence");
    }
};

void run_prefill_k6_gateup_warpgroup_qualification(Exl3TextModel& target) {
    require(env("NINFER_EXL3_PREFILL_K6_GATEUP_WARPGROUP_ASYNC") == "0",
            "K6 gate/up warp-group capture needs control flag");
    auto ids = load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    constexpr int initial_rows = 16;
    require(ids.size() >= initial_rows +
                PrefillK6GateupWarpgroupQualification::max_rows,
            "K6 gate/up warp-group prompt extent");
    ids.resize(initial_rows +
               PrefillK6GateupWarpgroupQualification::max_rows);
    auto context = target.create_context(true);
    PrefillK6GateupWarpgroupQualification qualification(
        env("NINFER_E5A4_OUT"));
    context->set_target_projection_observer_for_test(
        PrefillK6GateupWarpgroupQualification::callback, &qualification,
        nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::gate_up_k6);
    context->prefill(std::span<const std::int64_t>(ids.data(), initial_rows));
    context->append_exact_prefill_wide(std::span<const std::int64_t>(
        ids.data() + initial_rows,
        PrefillK6GateupWarpgroupQualification::max_rows));
    context->finish_exact_prefill();
    cuda_check(cudaDeviceSynchronize(),
               "K6 gate/up warp-group qualification complete");
    context->set_target_projection_observer_for_test(nullptr, nullptr);
    require(qualification.checked && qualification.observations == 94 &&
                context->position() == initial_rows +
                    PrefillK6GateupWarpgroupQualification::max_rows,
            "K6 gate/up warp-group real-caller coverage");
    std::cout << "PREFILL_K6_GATEUP_WARPGROUP PASS observations="
              << qualification.observations << " rows=17,31,32,127,128\n";
}
