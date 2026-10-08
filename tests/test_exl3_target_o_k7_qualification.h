#pragma once

// Operator qualification for the explicitly owned target-continuation K7 O
// candidate. The production route reuses the accepted 32-output-tile generic
// MMA kernel, split policy, transforms and reduction unchanged.


void run_target_o_k7_qualification(Exl3TextModel& target,
                                      Exl3Dflash2DraftModel& draft,
                                      const std::vector<std::int64_t>& prompt,
                                      bool oscar,
                                      std::ostream& out) {
    require(prompt.size() == 512, "target O K7 qualification requires ctx512");
    require(oscar, "target O K7 qualification requires canonical OSCAR");
    require(env("NINFER_EXL3_TARGET_O_K7_SMALL_M") == "0" &&
                env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "1" &&
                env("NINFER_EXL3_TARGET_DOWN_SMALL_M") == "1",
            "target O K7 capture requires O flag0 and accepted K6 flags1");
    ninfer::test::ScopedEnvironmentRestore restore_environment{
        "NINFER_EXL3_TARGET_O_K7_SMALL_M",
        "NINFER_EXL3_TARGET_GATEUP_SMALL_M",
        "NINFER_EXL3_TARGET_DOWN_SMALL_M",
        "NINFER_EXL3_GENERIC_SPLITS"};
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");
    auto captured = run_target_o_k7_capture(
        target, draft, prompt, std::filesystem::path{}, oscar);
    require(captured.size() == 32,
            "target O K7 qualification capture count mismatch");

    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    constexpr Admission admit = Admission::target_continuation_o;
    constexpr int input_features = 6144;
    constexpr int output_features = 5120;
    constexpr std::size_t guard = 128;
    constexpr std::uint16_t sentinel = 0x3555;
    const std::vector<int> oracle_groups = {0, 3, 4, 19, 20, 39};
    const auto& representative = captured.front();

    _putenv_s("NINFER_EXL3_TARGET_O_K7_SMALL_M", "1");
    Exl3CudaLinearWorkspace enabled(
        input_features, output_features, 8, false, false, false, true);
    Exl3CudaLinearWorkspace generic_owner(input_features, output_features, 8);
    require(std::string(enabled.dispatch_name(representative.metadata, 8, admit)) ==
                "target_o_k7_small_m_mma_split" &&
                std::string(enabled.dispatch_name(representative.metadata, 1, admit)) ==
                "generic_mma_split",
            "target O K7 owned dispatch or M1 preservation failed");
    require(std::string(enabled.dispatch_name(representative.metadata, 8)) ==
                "generic_tile" &&
                std::string(enabled.dispatch_name(
                    representative.metadata, 8,
                    Admission::target_continuation_down)) == "generic_tile" &&
                std::string(generic_owner.dispatch_name(
                    representative.metadata, 8, admit)) == "generic_tile",
            "target O K7 candidate escaped ownership/admission");
    for (const char* disabled : {"", "0", "2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_TARGET_O_K7_SMALL_M", disabled);
        Exl3CudaLinearWorkspace off(
            input_features, output_features, 8, false, false, false, true);
        require(std::string(off.dispatch_name(representative.metadata, 8, admit)) ==
                    "generic_tile",
                "target O K7 accepted a non-exact opt-in value");
    }
    _putenv_s("NINFER_EXL3_TARGET_O_K7_SMALL_M", "0");
    Exl3CudaLinearWorkspace latched_off(
        input_features, output_features, 8, false, false, false, true);
    require(std::string(enabled.dispatch_name(
                representative.metadata, 8, admit)) ==
                    "target_o_k7_small_m_mma_split",
            "target O K7 enabled workspace did not retain latched flag1");
    _putenv_s("NINFER_EXL3_TARGET_O_K7_SMALL_M", "1");
    require(std::string(latched_off.dispatch_name(
                representative.metadata, 8, admit)) == "generic_tile" &&
                std::string(enabled.dispatch_name(
                    representative.metadata, 8, admit)) ==
                    "target_o_k7_small_m_mma_split",
            "target O K7 workspace did not latch candidate flag");
    Exl3CudaLinearWorkspace owner_max4(
        input_features, output_features, 4, false, false, false, true);
    Exl3CudaLinearWorkspace wrong_input_owner(
        input_features - 128, output_features, 8, false, false, false, true);
    Exl3CudaLinearWorkspace wrong_output_owner(
        input_features, output_features + 128, 8, false, false, false, true);
    Exl3CudaLinearWorkspace unrelated_k7_owner(
        17408, output_features, 8, false, false, false, true);
    auto unrelated_k7 = representative.metadata;
    unrelated_k7.in_features = 17408;
    require(!owner_max4.target_o_k7_small_m_candidate(
                representative.metadata, 5, admit),
            "target O K7 admitted rows beyond workspace capacity");
    require(!wrong_input_owner.target_o_k7_small_m_candidate(
                representative.metadata, 8, admit) &&
                !wrong_output_owner.target_o_k7_small_m_candidate(
                    representative.metadata, 8, admit),
            "target O K7 admitted a mismatched workspace shape");
    require(!unrelated_k7_owner.target_o_k7_small_m_candidate(
                unrelated_k7, 8, admit),
            "target O K7 admitted an unrelated K7 family");
    require(!enabled.target_o_k7_small_m_candidate(
                representative.metadata, 1, admit) &&
                !enabled.target_o_k7_small_m_candidate(
                    representative.metadata, 9, admit),
            "target O K7 admitted unsupported row count");
    for (int variant = 0; variant < 6; ++variant) {
        auto unsupported = representative.metadata;
        if (variant == 0) unsupported.K = 6;
        if (variant == 1) unsupported.mcg = true;
        if (variant == 2) unsupported.mul1 = false;
        if (variant == 3) unsupported.has_bias = true;
        if (variant == 4) unsupported.in_features = input_features - 128;
        if (variant == 5) unsupported.out_features = output_features + 128;
        require(!enabled.target_o_k7_small_m_candidate(unsupported, 8, admit),
                "target O K7 admitted unsupported metadata");
    }

    out << "layer,operator,M,dispatch,split_override,repeat_equal,guards_ok,"
           "finite,bit_mismatches,changed_input,oracle_values,oracle_max_abs,"
           "oracle_rel_l2,oracle_max_error_over_bound,oracle_max_bound,"
           "oracle_fp64_bound,oracle_worst_group,oracle_worst_row,"
           "oracle_worst_column,oracle_within_bound\n";
    std::size_t checked_values = 0;
    std::size_t oracle_values = 0;
    double oracle_max_abs = 0.0;
    double oracle_max_ratio = 0.0;

    for (const auto& item : captured) {
        const std::size_t guarded_elements =
            8u * output_features + 2u * guard;
        DeviceBuffer input(item.input.size() * sizeof(std::uint16_t));
        DeviceBuffer output(guarded_elements * sizeof(std::uint16_t));
        DeviceBuffer reference(static_cast<std::size_t>(output_features) *
                               sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* output_base = static_cast<std::uint16_t*>(output.get());
        auto* output_device = output_base + guard;
        auto* reference_device = static_cast<std::uint16_t*>(reference.get());
        cuda_check(cudaMemcpy(input_device, item.input.data(),
                              item.input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target O K7 input upload");
        _putenv_s("NINFER_EXL3_TARGET_O_K7_SMALL_M", "1");
        _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");
        Exl3CudaLinearWorkspace candidate(
            input_features, output_features, 8, false, false, false, true);
        Exl3CudaLinearWorkspace reference_ws(input_features, output_features, 1);
        std::vector<std::uint16_t> initial(guarded_elements, sentinel);
        std::vector<std::uint16_t> actual(guarded_elements);
        std::vector<std::uint16_t> repeated(guarded_elements);
        std::vector<std::uint16_t> row_reference(output_features);
        std::vector<std::uint16_t> m8_actual;
        for (int m = 2; m <= 8; ++m) {
            require(std::string(candidate.dispatch_name(item.metadata, m, admit)) ==
                        "target_o_k7_small_m_mma_split",
                    "target O K7 candidate failed M2..8 dispatch");
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target O K7 canary upload");
            candidate.forward(item.weights, item.metadata, input_device,
                              output_device, m, nullptr, admit);
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target O K7 output download");
            candidate.forward(item.weights, item.metadata, input_device,
                              output_device, m, nullptr, admit);
            cuda_check(cudaMemcpy(repeated.data(), output_base,
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target O K7 repeat download");
            const bool repeat_equal = actual == repeated;
            const bool guards_ok =
                std::all_of(actual.begin(), actual.begin() + guard,
                            [](std::uint16_t v) { return v == sentinel; }) &&
                std::all_of(actual.begin() + guard +
                                static_cast<std::size_t>(m) * output_features,
                            actual.end(),
                            [](std::uint16_t v) { return v == sentinel; });
            bool finite = true;
            std::size_t mismatches = 0;
            for (int row = 0; row < m; ++row) {
                reference_ws.forward(
                    item.weights, item.metadata,
                    input_device + static_cast<std::size_t>(row) * input_features,
                    reference_device, 1);
                cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                      row_reference.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost),
                           "target O K7 M1 download");
                const auto offset = guard +
                    static_cast<std::size_t>(row) * output_features;
                for (int column = 0; column < output_features; ++column) {
                    const auto value = actual[offset + static_cast<std::size_t>(column)];
                    finite = finite && std::isfinite(half_to_float(value));
                    mismatches += value != row_reference[static_cast<std::size_t>(column)];
                }
            }
            H6OracleStats oracle{};
            if (m == 8) {
                m8_actual = actual;
                oracle = target_o_k7_check_oracle_groups(
                    item.weights, item.metadata, item.input, actual, guard,
                    oracle_groups);
                oracle_values += oracle.values;
                oracle_max_abs = std::max(oracle_max_abs, oracle.max_abs);
                oracle_max_ratio = std::max(oracle_max_ratio, oracle.max_ratio);
                require(oracle.within_bound,
                        "target O K7 exceeded independent FP64 bound");
            }
            checked_values += static_cast<std::size_t>(m) * output_features;
            out << item.layer << ',' << item.operation << ',' << m << ','
                << candidate.dispatch_name(item.metadata, m, admit)
                << ",0," << repeat_equal << ',' << guards_ok << ',' << finite
                << ',' << mismatches << ",0,";
            if (m == 8) {
                out << oracle.values << ',' << oracle.max_abs << ','
                    << (oracle.norm_sq > 0.0
                            ? std::sqrt(oracle.error_sq / oracle.norm_sq) : 0.0)
                    << ',' << oracle.max_ratio << ',' << oracle.max_bound << ','
                    << oracle.max_fp64_bound << ',' << oracle.worst_group << ','
                    << oracle.worst_row << ',' << oracle.worst_column << ','
                    << oracle.within_bound << '\n';
            } else {
                out << "0,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";
            }
            require(repeat_equal && guards_ok && finite && mismatches == 0,
                    "target O K7 failed exact M1 differential gate");
        }

        auto changed = item.input;
        for (auto& bits : changed) bits ^= 0x8000u;
        cuda_check(cudaMemcpy(input_device, changed.data(),
                              changed.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target O K7 changed input upload");
        cuda_check(cudaMemcpy(output_base, initial.data(),
                              guarded_elements * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target O K7 changed canary upload");
        candidate.forward(item.weights, item.metadata, input_device,
                          output_device, 3, nullptr, admit);
        cuda_check(cudaMemcpy(actual.data(), output_base,
                              guarded_elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "target O K7 changed output download");
        candidate.forward(item.weights, item.metadata, input_device,
                          output_device, 3, nullptr, admit);
        cuda_check(cudaMemcpy(repeated.data(), output_base,
                              guarded_elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "target O K7 changed repeat download");
        const bool changed_repeat = actual == repeated;
        const bool changed_guards =
            std::all_of(actual.begin(), actual.begin() + guard,
                        [](std::uint16_t v) { return v == sentinel; }) &&
            std::all_of(actual.begin() + guard + 3u * output_features,
                        actual.end(),
                        [](std::uint16_t v) { return v == sentinel; });
        bool changed_finite = true;
        std::size_t changed_mismatches = 0;
        for (int row = 0; row < 3; ++row) {
            reference_ws.forward(
                item.weights, item.metadata,
                input_device + static_cast<std::size_t>(row) * input_features,
                reference_device, 1);
            cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                  row_reference.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target O K7 changed M1 download");
            const auto offset = guard +
                static_cast<std::size_t>(row) * output_features;
            for (int column = 0; column < output_features; ++column) {
                const auto value = actual[offset + static_cast<std::size_t>(column)];
                changed_finite = changed_finite &&
                    std::isfinite(half_to_float(value));
                changed_mismatches +=
                    value != row_reference[static_cast<std::size_t>(column)];
            }
        }
        require(changed_repeat && changed_guards && changed_finite &&
                    changed_mismatches == 0 &&
                    !std::equal(actual.begin() + guard,
                                actual.begin() + guard + 3u * output_features,
                                m8_actual.begin() + guard),
                "target O K7 changed-input reuse failed");
        out << item.layer << ',' << item.operation << ",3,"
            << candidate.dispatch_name(item.metadata, 3, admit)
            << ",0," << changed_repeat << ',' << changed_guards << ',' << changed_finite
            << ",0,1,0,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";

        for (const int invalid_rows : {-1, 0, 9}) {
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target O K7 invalid-row canary upload");
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
                       "target O K7 invalid-row canary download");
            require(rejected && actual == initial,
                    "target O K7 invalid row modified output");
        }
    }

    // Explicit S1 differential: real GDN layer 2 and full-attention layer 23,
    // each at M2 and M8, with candidate/reference sharing the override.
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "1");
    for (const int layer : {2, 23}) {
        const auto found = std::find_if(captured.begin(), captured.end(),
            [layer](const auto& item) { return item.layer == layer; });
        require(found != captured.end(), "target O K7 S1 layer missing");
        DeviceBuffer input(found->input.size() * sizeof(std::uint16_t));
        DeviceBuffer output((8u * output_features + 2u * guard) * sizeof(std::uint16_t));
        DeviceBuffer reference(static_cast<std::size_t>(output_features) * sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* output_base = static_cast<std::uint16_t*>(output.get());
        auto* output_device = output_base + guard;
        auto* reference_device = static_cast<std::uint16_t*>(reference.get());
        cuda_check(cudaMemcpy(input_device, found->input.data(),
                              found->input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "target O K7 S1 input upload");
        Exl3CudaLinearWorkspace candidate(
            input_features, output_features, 8, false, false, false, true);
        Exl3CudaLinearWorkspace reference_ws(input_features, output_features, 1);
        std::vector<std::uint16_t> initial(8u * output_features + 2u * guard, sentinel);
        std::vector<std::uint16_t> actual(initial.size());
        std::vector<std::uint16_t> repeated(initial.size());
        std::vector<std::uint16_t> row_reference(output_features);
        for (const int m : {2, 8}) {
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  initial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice), "target O K7 S1 canary upload");
            candidate.forward(found->weights, found->metadata, input_device,
                              output_device, m, nullptr, admit);
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  actual.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "target O K7 S1 output download");
            candidate.forward(found->weights, found->metadata, input_device,
                              output_device, m, nullptr, admit);
            cuda_check(cudaMemcpy(repeated.data(), output_base,
                                  repeated.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target O K7 S1 repeat download");
            const bool repeat_equal = actual == repeated;
            std::size_t mismatches = 0;
            for (int row = 0; row < m; ++row) {
                reference_ws.forward(
                    found->weights, found->metadata,
                    input_device + static_cast<std::size_t>(row) * input_features,
                    reference_device, 1);
                cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                      row_reference.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost), "target O K7 S1 M1 download");
                const auto offset = guard +
                    static_cast<std::size_t>(row) * output_features;
                for (int column = 0; column < output_features; ++column)
                    mismatches += actual[offset + static_cast<std::size_t>(column)] !=
                        row_reference[static_cast<std::size_t>(column)];
            }
            const bool guards =
                std::all_of(actual.begin(), actual.begin() + guard,
                            [](std::uint16_t v) { return v == sentinel; }) &&
                std::all_of(actual.begin() + guard +
                                static_cast<std::size_t>(m) * output_features,
                            actual.end(),
                            [](std::uint16_t v) { return v == sentinel; });
            bool finite = true;
            for (std::size_t i = guard;
                 i < guard + static_cast<std::size_t>(m) * output_features; ++i)
                finite = finite && std::isfinite(half_to_float(actual[i]));
            require(repeat_equal && guards && finite && mismatches == 0,
                    "target O K7 S1 differential failed");
            out << found->layer << ',' << found->operation << ',' << m << ','
                << candidate.dispatch_name(found->metadata, m, admit)
                << ",1," << repeat_equal << ',' << guards
                << ',' << finite << ',' << mismatches
                << ",0,0,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";
        }
    }
    out.flush();
    require(out.good(), "target O K7 qualification CSV write failed");
    require(oracle_values == 196608,
            "target O K7 oracle coverage count mismatch");
    std::cout << "TARGETOK7QUAL_ORACLE PASS pairs=32 rows=8 groups=6"
              << " values=" << oracle_values << " max_abs=" << oracle_max_abs
              << " max_error_over_bound=" << oracle_max_ratio << '\n';
    std::cout << "TARGETOK7QUAL PASS pairs=32 M=2..8 checked_values="
              << checked_values
              << " bit_mismatches=0 repeat=1 guards=1 finite=1 changed_input=32"
              << " dispatch=1 s1_cases=4" << std::endl;
}
