#pragma once

// Operator qualification for the explicitly owned target-continuation K6
// down candidate.  The production route reuses the accepted 16-output-tile
// large-down MMA kernel, split policy, transforms and reduction unchanged.


void run_target_down_k6_qualification(Exl3TextModel& target,
                                      Exl3Dflash2DraftModel& draft,
                                      const std::vector<std::int64_t>& prompt,
                                      bool oscar,
                                      std::ostream& out) {
    require(prompt.size() == 512, "target down K6 qualification requires ctx512");
    require(oscar, "target down K6 qualification requires canonical OSCAR");
    require(env("NINFER_EXL3_TARGET_DOWN_SMALL_M") == "0" &&
                env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "0" &&
                env("NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A") == "0" &&
                env("NINFER_EXL3_EXTENDED_STREAM_REDUCTION") == "0",
            "target down K6 capture requires all candidates explicit flag0");
    ninfer::test::ScopedEnvironmentRestore restore_environment{
        "NINFER_EXL3_TARGET_DOWN_SMALL_M",
        "NINFER_EXL3_TARGET_GATEUP_SMALL_M",
        "NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A",
        "NINFER_EXL3_EXTENDED_STREAM_REDUCTION",
        "NINFER_EXL3_LARGE_DOWN_TOPOLOGY",
        "NINFER_EXL3_LARGE_DOWN_SPLITS"};
    _putenv_s("NINFER_EXL3_LARGE_DOWN_TOPOLOGY", "16");
    _putenv_s("NINFER_EXL3_LARGE_DOWN_SPLITS", "");
    auto captured = run_target_down_k6_capture(
        target, draft, prompt, std::filesystem::path{}, oscar);
    require(captured.size() == 43,
            "target down K6 qualification capture count mismatch");

    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    constexpr Admission admit = Admission::target_continuation_down;
    constexpr Admission ordinary = Admission::ordinary;
    constexpr int input_features = 17408;
    constexpr int output_features = 5120;
    constexpr std::size_t guard = 128;
    constexpr std::uint16_t sentinel = 0x3555;
    const std::vector<int> oracle_groups = {0, 1, 2, 19, 20, 39};
    const auto& representative = captured.front();

    _putenv_s("NINFER_EXL3_TARGET_DOWN_SMALL_M", "1");
    _putenv_s("NINFER_EXL3_EXTENDED_STREAM_REDUCTION", "1");
    _putenv_s("NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A", "0");
    Exl3CudaLinearWorkspace stream_control(
        input_features, output_features, 8, false, false, true,
        false, false, true);
    require(std::string(stream_control.dispatch_name(
                representative.metadata, 8, admit)) ==
                "target_extended_stream_reduction" &&
                std::string(stream_control.dispatch_name(
                    representative.metadata, 1, ordinary)) ==
                "target_extended_stream_reduction",
            "target down K6 async-A flag0 changed the selected stream route");
    _putenv_s("NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A", "1");
    Exl3CudaLinearWorkspace enabled(
        input_features, output_features, 8, false, false, true,
        false, false, true);
    Exl3CudaLinearWorkspace generic_owner(input_features, output_features, 8);
    require(std::string(enabled.dispatch_name(representative.metadata, 8, admit)) ==
                "target_down_k6_async_a_stream_reduction" &&
                std::string(enabled.dispatch_name(
                    representative.metadata, 1, ordinary)) ==
                "target_down_k6_async_a_stream_reduction",
            "target down K6 async-A M1/M8 dispatch failed");
    require(std::string(enabled.dispatch_name(representative.metadata, 8)) ==
                "generic_tile" &&
                std::string(generic_owner.dispatch_name(
                    representative.metadata, 8, admit)) == "generic_tile",
            "target down K6 candidate escaped ownership/admission");
    _putenv_s("NINFER_EXL3_TARGET_DOWN_SMALL_M", "0");
    Exl3CudaLinearWorkspace off(
        input_features, output_features, 8, false, false, true,
        false, false, true);
    require(std::string(off.dispatch_name(representative.metadata, 8, admit)) ==
                "generic_tile",
            "target down K6 explicit flag0 did not disable the inherited route");
    _putenv_s("NINFER_EXL3_TARGET_DOWN_SMALL_M", "1");
    for (const char* invalid : {"2", "true", "01"}) {
        _putenv_s("NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A", invalid);
        bool rejected=false;
        try {
            Exl3CudaLinearWorkspace invalid_async(
                input_features, output_features, 8, false, false, true,
                false, false, true);
        } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"target down K6 async-A accepted a non-exact opt-in value");
    }
    _putenv_s("NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A", "1");
    _putenv_s("NINFER_EXL3_TARGET_DOWN_SMALL_M", "0");
    Exl3CudaLinearWorkspace latched_off(
        input_features, output_features, 8, false, false, true,
        false, false, true);
    require(std::string(enabled.dispatch_name(
                representative.metadata, 8, admit)) ==
                    "target_down_k6_async_a_stream_reduction",
            "target down K6 enabled workspace did not retain latched flag1");
    _putenv_s("NINFER_EXL3_TARGET_DOWN_SMALL_M", "1");
    require(std::string(latched_off.dispatch_name(
                representative.metadata, 8, admit)) == "generic_tile" &&
                std::string(enabled.dispatch_name(
                representative.metadata, 8, admit)) ==
                    "target_down_k6_async_a_stream_reduction",
            "target down K6 workspace did not latch candidate flag");
    Exl3CudaLinearWorkspace owner_max4(
        input_features, output_features, 4, false, false, true,
        false, false, true);
    require(!owner_max4.target_down_small_m_candidate(
                representative.metadata, 5, admit),
            "target down K6 admitted rows beyond workspace capacity");
    require(!enabled.target_down_small_m_candidate(
                representative.metadata, 1, admit) &&
                !enabled.target_down_small_m_candidate(
                    representative.metadata, 9, admit),
            "target down K6 admitted unsupported row count");
    for (const char* declined : {"8", "32", "off", "16-k7"}) {
        _putenv_s("NINFER_EXL3_LARGE_DOWN_TOPOLOGY", declined);
        require(!enabled.target_down_small_m_candidate(
                    representative.metadata, 8, admit) &&
                    std::string(enabled.dispatch_name(
                        representative.metadata, 8, admit)) == "generic_tile",
                "target down K6 admitted a declined topology");
    }
    for (const char* admitted : {"", "16", "16-k6"}) {
        _putenv_s("NINFER_EXL3_LARGE_DOWN_TOPOLOGY", admitted);
        require(enabled.target_down_small_m_candidate(
                    representative.metadata, 8, admit),
                "target down K6 declined an allowed 16-tile topology");
    }
    _putenv_s("NINFER_EXL3_LARGE_DOWN_TOPOLOGY", "16");
    for (int variant = 0; variant < 6; ++variant) {
        auto unsupported = representative.metadata;
        if (variant == 0) unsupported.K = 5;
        if (variant == 1) unsupported.mcg = true;
        if (variant == 2) unsupported.mul1 = false;
        if (variant == 3) unsupported.has_bias = true;
        if (variant == 4) unsupported.in_features = input_features - 128;
        if (variant == 5) unsupported.out_features = output_features + 128;
        require(!enabled.target_down_small_m_candidate(unsupported, 8, admit),
                "target down K6 admitted unsupported metadata");
        require(!enabled.target_down_k6_async_a_candidate(
                    unsupported, 8, admit),
                "target down K6 async-A admitted unsupported metadata");
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
                   "target down K6 input upload");
        _putenv_s("NINFER_EXL3_TARGET_DOWN_SMALL_M", "1");
        _putenv_s("NINFER_EXL3_EXTENDED_STREAM_REDUCTION", "1");
        _putenv_s("NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A", "1");
        _putenv_s("NINFER_EXL3_LARGE_DOWN_TOPOLOGY", "16");
        _putenv_s("NINFER_EXL3_LARGE_DOWN_SPLITS", "");
        Exl3CudaLinearWorkspace candidate(
            input_features, output_features, 8, false, false, true,
            false, false, true);
        Exl3CudaLinearWorkspace reference_ws(input_features, output_features, 1);
        std::vector<std::uint16_t> initial(guarded_elements, sentinel);
        std::vector<std::uint16_t> actual(guarded_elements);
        std::vector<std::uint16_t> repeated(guarded_elements);
        std::vector<std::uint16_t> row_reference(output_features);
        std::vector<std::uint16_t> m8_actual;
        for (int m = 1; m <= 8; ++m) {
            const auto candidate_admission = m == 1 ? ordinary : admit;
            require(std::string(candidate.dispatch_name(
                        item.metadata, m, candidate_admission)) ==
                        "target_down_k6_async_a_stream_reduction",
                    "target down K6 async-A candidate failed M1..8 dispatch");
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target down K6 canary upload");
            candidate.forward(item.weights, item.metadata, input_device,
                              output_device, m, nullptr, candidate_admission);
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target down K6 output download");
            candidate.forward(item.weights, item.metadata, input_device,
                              output_device, m, nullptr, candidate_admission);
            cuda_check(cudaMemcpy(repeated.data(), output_base,
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target down K6 repeat download");
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
                           "target down K6 M1 download");
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
                oracle = target_down_k6_check_oracle_groups(
                    item.weights, item.metadata, item.input, actual, guard,
                    oracle_groups);
                oracle_values += oracle.values;
                oracle_max_abs = std::max(oracle_max_abs, oracle.max_abs);
                oracle_max_ratio = std::max(oracle_max_ratio, oracle.max_ratio);
                require(oracle.within_bound,
                        "target down K6 exceeded independent FP64 bound");
            }
            checked_values += static_cast<std::size_t>(m) * output_features;
            out << item.layer << ',' << item.operation << ',' << m << ','
                << candidate.dispatch_name(item.metadata, m, candidate_admission)
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
                    "target down K6 failed exact M1 differential gate");
        }
        require(candidate.target_down_k6_async_a_calls() == 16,
                "target down K6 async-A M1..8 dispatch count mismatch");

        auto changed = item.input;
        for (auto& bits : changed) bits ^= 0x8000u;
        cuda_check(cudaMemcpy(input_device, changed.data(),
                              changed.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target down K6 changed input upload");
        cuda_check(cudaMemcpy(output_base, initial.data(),
                              guarded_elements * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice),
                   "target down K6 changed canary upload");
        candidate.forward(item.weights, item.metadata, input_device,
                          output_device, 3, nullptr, admit);
        cuda_check(cudaMemcpy(actual.data(), output_base,
                              guarded_elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "target down K6 changed output download");
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
                       "target down K6 changed M1 download");
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
        require(changed_guards && changed_finite && changed_mismatches == 0 &&
                    !std::equal(actual.begin() + guard,
                                actual.begin() + guard + 3u * output_features,
                                m8_actual.begin() + guard),
                "target down K6 changed-input reuse failed");
        out << item.layer << ',' << item.operation << ",3,"
            << candidate.dispatch_name(item.metadata, 3, admit)
            << ",0,NA," << changed_guards << ',' << changed_finite
            << ",0,1,0,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";

        for (const int invalid_rows : {-1, 0, 9}) {
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  guarded_elements * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target down K6 invalid-row canary upload");
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
                       "target down K6 invalid-row canary download");
            require(rejected && actual == initial,
                    "target down K6 invalid row modified output");
        }
    }

    // Explicit S1 differential: one real GDN layer (2) and one real full layer
    // (3), each at M2 and M8, with candidate/reference sharing the override.
    _putenv_s("NINFER_EXL3_LARGE_DOWN_SPLITS", "1");
    for (const int layer : {2, 3}) {
        const auto found = std::find_if(captured.begin(), captured.end(),
            [layer](const auto& item) { return item.layer == layer; });
        require(found != captured.end(), "target down K6 S1 layer missing");
        DeviceBuffer input(found->input.size() * sizeof(std::uint16_t));
        DeviceBuffer output((8u * output_features + 2u * guard) * sizeof(std::uint16_t));
        DeviceBuffer reference(static_cast<std::size_t>(output_features) * sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* output_base = static_cast<std::uint16_t*>(output.get());
        auto* output_device = output_base + guard;
        auto* reference_device = static_cast<std::uint16_t*>(reference.get());
        cuda_check(cudaMemcpy(input_device, found->input.data(),
                              found->input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "target down K6 S1 input upload");
        Exl3CudaLinearWorkspace candidate(
            input_features, output_features, 8, false, false, true,
            false, false, true);
        Exl3CudaLinearWorkspace reference_ws(input_features, output_features, 1);
        std::vector<std::uint16_t> initial(8u * output_features + 2u * guard, sentinel);
        std::vector<std::uint16_t> actual(initial.size());
        std::vector<std::uint16_t> row_reference(output_features);
        for (const int m : {2, 8}) {
            cuda_check(cudaMemcpy(output_base, initial.data(),
                                  initial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice), "target down K6 S1 canary upload");
            candidate.forward(found->weights, found->metadata, input_device,
                              output_device, m, nullptr, admit);
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  actual.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "target down K6 S1 output download");
            std::size_t mismatches = 0;
            for (int row = 0; row < m; ++row) {
                reference_ws.forward(
                    found->weights, found->metadata,
                    input_device + static_cast<std::size_t>(row) * input_features,
                    reference_device, 1);
                cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                      row_reference.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost), "target down K6 S1 M1 download");
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
            require(guards && finite && mismatches == 0,
                    "target down K6 S1 differential failed");
            out << found->layer << ',' << found->operation << ',' << m << ','
                << candidate.dispatch_name(found->metadata, m, admit)
                << ",1,NA," << guards
                << ',' << finite << ',' << mismatches
                << ",0,0,NA,NA,NA,NA,NA,NA,NA,NA,NA\n";
        }
    }
    out.flush();
    require(out.good(), "target down K6 qualification CSV write failed");
    require(oracle_values == 264192,
            "target down K6 oracle coverage count mismatch");
    std::cout << "TARGETDOWNK6QUAL_ORACLE PASS pairs=43 rows=8 groups=6"
              << " values=" << oracle_values << " max_abs=" << oracle_max_abs
              << " max_error_over_bound=" << oracle_max_ratio << '\n';
    std::cout << "TARGETDOWNK6QUAL PASS pairs=43 M=1..8 checked_values="
              << checked_values
              << " bit_mismatches=0 repeat=1 guards=1 finite=1 changed_input=43"
              << " dispatch=1 s1_cases=4" << std::endl;
}
