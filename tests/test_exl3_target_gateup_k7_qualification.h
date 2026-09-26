#pragma once


void run_target_gateup_k7_qualification(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt, bool oscar,
    std::ostream& out) {
    require(prompt.size() == 512,
            "target gate/up K7 qualification requires ctx512");
    require(oscar, "target gate/up K7 qualification requires canonical OSCAR");
    require(env("NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M") == "0",
            "target gate/up K7 qualification must start at explicit flag0");
    ninfer::test::ScopedEnvironmentRestore restore{
        "NINFER_EXL3_TARGET_GATEUP_SMALL_M",
        "NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M",
        "NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A",
        "NINFER_EXL3_GENERIC_SPLITS"};
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_SMALL_M", "0");
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "5");
    _putenv_s("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A", "0");

    const auto captured = run_target_gateup_k7_capture(
        target, draft, prompt, oscar);
    require(captured.size() == 4,
            "target gate/up K7 qualification capture count mismatch");
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    constexpr Admission admit = Admission::target_continuation_gate_up;
    constexpr int output_features = 17408;
    constexpr std::size_t guard = 128;
    constexpr std::uint16_t sentinel = 0x3555;

    const auto& representative = captured.front();
    for (const char* malformed : {"2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A", malformed);
        bool rejected = false;
        try {
            Exl3CudaLinearWorkspace invalid(
                kHidden, output_features, 1, false, true);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "target M1 K7 async-A accepted malformed opt-in");
    }
    _putenv_s("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A", "0");
    Exl3CudaLinearWorkspace m1_latched_off(
        kHidden, output_features, 1, false, true);
    _putenv_s("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A", "1");
    require(std::string(m1_latched_off.dispatch_name(
                representative.metadata, 1, admit)) == "generic_mma_split",
            "target M1 K7 workspace did not latch flag0");
    Exl3CudaLinearWorkspace m1_enabled(
        kHidden, output_features, 1, false, true);
    Exl3CudaLinearWorkspace m1_generic_owner(
        kHidden, output_features, 1);
    require(std::string(m1_enabled.dispatch_name(
                representative.metadata, 1, admit)) ==
                "target_m1_k7_n32_async_a" &&
                std::string(m1_generic_owner.dispatch_name(
                    representative.metadata, 1, admit)) ==
                "generic_mma_split",
            "target M1 K7 async-A ownership/dispatch mismatch");
    _putenv_s("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A", "0");
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M", "1");
    Exl3CudaLinearWorkspace enabled(kHidden, output_features, 8, false, true);
    Exl3CudaLinearWorkspace generic_owner(kHidden, output_features, 8);
    require(std::string(enabled.dispatch_name(
                representative.metadata, 8, admit)) ==
                "target_gateup_k7_small_m_mma_split",
            "target gate/up K7 owned M8 dispatch missing");
    require(std::string(enabled.dispatch_name(representative.metadata, 8)) ==
                "generic_tile" &&
                std::string(generic_owner.dispatch_name(
                    representative.metadata, 8, admit)) == "generic_tile",
            "target gate/up K7 escaped target-owner admission");
    require(std::string(enabled.dispatch_name(
                representative.metadata, 1, admit)) == "generic_mma_split" &&
                std::string(enabled.dispatch_name(
                    representative.metadata, 9, admit)) == "generic_tile",
            "target gate/up K7 changed M1 or admitted M9");
    Exl3CudaLinearWorkspace owner_max4(
        kHidden, output_features, 4, false, true);
    require(!owner_max4.target_gateup_small_m_candidate(
                representative.metadata, 5, admit),
            "target gate/up K7 admitted beyond workspace rows");
    auto wrong = representative.metadata;
    wrong.out_features += 128;
    require(!enabled.target_gateup_small_m_candidate(wrong, 8, admit),
            "target gate/up K7 admitted mismatched metadata");
    for (const char* disabled : {"", "0", "2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M", disabled);
        Exl3CudaLinearWorkspace off(
            kHidden, output_features, 8, false, true);
        require(std::string(off.dispatch_name(
                    representative.metadata, 8, admit)) == "generic_tile",
                "target gate/up K7 accepted a non-exact opt-in");
    }
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M", "1");

    out << "layer,operator,M,dispatch,repeat_equal,guards_ok,finite,"
           "bit_mismatches,values\n";
    std::size_t total_values = 0;
    std::size_t m1_async_values = 0;
    for (const auto& item : captured) {
        require((item.layer == 58 || item.layer == 63) &&
                    (item.operation == "gate" || item.operation == "up") &&
                    item.metadata.K == 7 && item.metadata.mul1 &&
                    !item.metadata.mcg && !item.metadata.has_bias &&
                    item.metadata.in_features == kHidden &&
                    item.metadata.out_features == output_features &&
                    item.input.size() == 8u * kHidden,
                "target gate/up K7 actual capture mismatch");
        DeviceBuffer input(item.input.size() * sizeof(std::uint16_t));
        const std::size_t capacity = 8u * output_features;
        const std::size_t guarded = capacity + 2u * guard;
        DeviceBuffer output(guarded * sizeof(std::uint16_t));
        DeviceBuffer reference(output_features * sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* output_base = static_cast<std::uint16_t*>(output.get());
        auto* output_device = output_base + guard;
        auto* reference_device = static_cast<std::uint16_t*>(reference.get());
        cuda_check(cudaMemcpy(input_device, item.input.data(),
                              item.input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target gate/up K7 input upload");
        _putenv_s("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A", "0");
        Exl3CudaLinearWorkspace m1_control(
            kHidden, output_features, 1, false, true);
        _putenv_s("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A", "1");
        Exl3CudaLinearWorkspace m1_candidate(
            kHidden, output_features, 1, false, true);
        require(std::string(m1_control.dispatch_name(
                    item.metadata, 1, admit)) == "generic_mma_split" &&
                    std::string(m1_candidate.dispatch_name(
                        item.metadata, 1, admit)) ==
                    "target_m1_k7_n32_async_a",
                "target M1 K7 actual dispatch mismatch");
        m1_control.forward(item.weights, item.metadata, input_device,
                           reference_device, 1, nullptr, admit);
        m1_candidate.forward(item.weights, item.metadata, input_device,
                             output_device, 1, nullptr, admit);
        std::vector<std::uint16_t> m1_expected(output_features);
        std::vector<std::uint16_t> m1_actual(output_features);
        cuda_check(cudaMemcpy(m1_expected.data(), reference_device,
                              output_features * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "target M1 K7 control download");
        cuda_check(cudaMemcpy(m1_actual.data(), output_device,
                              output_features * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "target M1 K7 candidate download");
        require(m1_actual == m1_expected &&
                    std::all_of(m1_actual.begin(), m1_actual.end(),
                        [](std::uint16_t bits) {
                            return std::isfinite(half_to_float(bits));
                        }),
                "target M1 K7 exact differential failed");
        m1_async_values += m1_actual.size();
        _putenv_s("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A", "0");
        Exl3CudaLinearWorkspace candidate(
            kHidden, output_features, 8, false, true);
        Exl3CudaLinearWorkspace m1_reference(
            kHidden, output_features, 1);
        std::vector<std::uint16_t> initial(guarded, sentinel);
        std::vector<std::uint16_t> actual(guarded);
        std::vector<std::uint16_t> repeated(guarded);
        std::vector<std::uint16_t> row_reference(output_features);
        for (int rows = 2; rows <= 8; ++rows) {
            require(std::string(candidate.dispatch_name(
                        item.metadata, rows, admit)) ==
                        "target_gateup_k7_small_m_mma_split",
                    "target gate/up K7 M2..8 dispatch missing");
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  guarded * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target gate/up K7 canary upload");
            candidate.forward(item.weights, item.metadata, input_device,
                              output_device, rows, nullptr, admit);
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  guarded * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target gate/up K7 output download");
            candidate.forward(item.weights, item.metadata, input_device,
                              output_device, rows, nullptr, admit);
            cuda_check(cudaMemcpy(repeated.data(), output_base,
                                  guarded * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target gate/up K7 repeat download");
            const bool repeat_equal = actual == repeated;
            const bool guards_ok =
                std::all_of(actual.begin(), actual.begin() + guard,
                            [](std::uint16_t v) { return v == sentinel; }) &&
                std::all_of(actual.begin() + guard +
                                static_cast<std::size_t>(rows) * output_features,
                            actual.end(),
                            [](std::uint16_t v) { return v == sentinel; });
            bool finite = true;
            std::size_t mismatches = 0;
            for (int row = 0; row < rows; ++row) {
                m1_reference.forward(
                    item.weights, item.metadata,
                    input_device + static_cast<std::size_t>(row) * kHidden,
                    reference_device, 1);
                cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                      output_features * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost),
                           "target gate/up K7 M1 reference download");
                const std::size_t offset = guard +
                    static_cast<std::size_t>(row) * output_features;
                for (int column = 0; column < output_features; ++column) {
                    const auto bits = actual[offset + column];
                    finite = finite && std::isfinite(half_to_float(bits));
                    mismatches += bits != row_reference[column];
                }
            }
            require(repeat_equal && guards_ok && finite && mismatches == 0,
                    "target gate/up K7 M2..8 exactness gate failed");
            const std::size_t values =
                static_cast<std::size_t>(rows) * output_features;
            total_values += values;
            out << item.layer << ',' << item.operation << ',' << rows << ','
                << candidate.dispatch_name(item.metadata, rows, admit) << ','
                << repeat_equal << ',' << guards_ok << ',' << finite << ','
                << mismatches << ',' << values << '\n';
        }
    }
    require(total_values == 2437120,
            "target gate/up K7 qualification value extent mismatch");
    require(m1_async_values == 69632,
            "target M1 K7 exact differential value extent mismatch");
    out.flush();
    require(out.good(), "target gate/up K7 qualification report write failed");
    std::cout << "TARGET_GATEUP_K7_QUALIFICATION PASS records=4 cases=28 values="
              << total_values
              << " m1_async_values=" << m1_async_values
              << " bit_mismatches=0 guards=1 repeat=1 finite=1 malformed=1 rollback=1"
              << std::endl;
}
