#pragma once

struct PrefillK7Tiles64ExactSplitsQualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit = Admission::target_wide_prefill;
    static constexpr int max_rows = 128;
    static constexpr std::size_t guard = 128;

    struct EnvironmentRestore {
        std::string candidate = env("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS");
        std::string rowpair = env("NINFER_EXL3_PREFILL_ROWPAIR_K7");
        std::string direct_a = env("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A");
        std::string direct_all = env("NINFER_EXL3_PREFILL_DIRECT_ASYNC_ALL");
        std::string partials = env("NINFER_EXL3_PREFILL_DIRECT_PARTIALS");
        ~EnvironmentRestore() {
            _putenv_s("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS", candidate.c_str());
            _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K7", rowpair.c_str());
            _putenv_s("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A", direct_a.c_str());
            _putenv_s("NINFER_EXL3_PREFILL_DIRECT_ASYNC_ALL", direct_all.c_str());
            _putenv_s("NINFER_EXL3_PREFILL_DIRECT_PARTIALS", partials.c_str());
        }
    } restore;

    std::filesystem::path directory;
    std::ofstream csv;
    std::set<std::pair<int, int>> shapes;
    int observations = 0;
    int cases = 0;
    std::uint64_t values = 0;
    bool strict_checked = false;

    explicit PrefillK7Tiles64ExactSplitsQualification(
        const std::filesystem::path& path) : directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),
                "K7 tiles64 exact-splits output must be new");
        std::filesystem::create_directories(path);
        csv.open(path / "differential.csv");
        csv << "input,output,operation,rows,values,control_route,candidate_route,"
               "exact,repeat,guards,input_unchanged,candidate_calls\n";
        _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K7", "1");
        _putenv_s("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A", "1");
        _putenv_s("NINFER_EXL3_PREFILL_DIRECT_ASYNC_ALL", "1");
        _putenv_s("NINFER_EXL3_PREFILL_DIRECT_PARTIALS", "1");
        _putenv_s("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS", "0");
    }

    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& observation,
        void* user) {
        static_cast<PrefillK7Tiles64ExactSplitsQualification*>(user)->check(
            observation);
    }

    static bool expected_shape(int input, int output) noexcept {
        return (input == 5120 &&
                (output == 1024 || output == 17408)) ||
               (input == 6144 && output == 5120);
    }

    static std::unique_ptr<Exl3CudaLinearWorkspace> make_workspace(
        const ninfer::exl3::Exl3TargetProjectionObservation& x,
        bool candidate) {
        const std::string operation = x.operation;
        _putenv_s("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS",
                  candidate ? "1" : "0");
        return std::make_unique<Exl3CudaLinearWorkspace>(
            x.metadata.in_features, x.metadata.out_features, max_rows, false,
            operation == "gate" || operation == "up", operation == "down",
            operation == "o", operation == "z", true, false,
            ninfer::exl3::Exl3CudaAccumulationView{},
            ninfer::exl3::Exl3CudaTransformView{}, operation == "qkv",
            operation == "k" || operation == "v", operation == "q");
    }

    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        ++observations;
        const int input_width = x.metadata.in_features;
        const int output_width = x.metadata.out_features;
        if (x.rows != max_rows || x.metadata.K != 7 ||
            !expected_shape(input_width, output_width) ||
            !shapes.emplace(input_width, output_width).second)
            return;
        cuda_check(cudaStreamSynchronize(x.stream),
                   "K7 tiles64 exact-splits input ready");

        auto control = make_workspace(x, false);
        auto candidate = make_workspace(x, true);
        _putenv_s("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS", "0");

        if (!strict_checked) {
            for (const char* malformed : {"2", "true", "01"}) {
                _putenv_s("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS",
                          malformed);
                bool rejected = false;
                try {
                    const std::string operation = x.operation;
                    Exl3CudaLinearWorkspace invalid(
                        x.metadata.in_features, x.metadata.out_features,
                        max_rows, false,
                        operation == "gate" || operation == "up",
                        operation == "down", operation == "o",
                        operation == "z", true, false,
                        ninfer::exl3::Exl3CudaAccumulationView{},
                        ninfer::exl3::Exl3CudaTransformView{},
                        operation == "qkv",
                        operation == "k" || operation == "v",
                        operation == "q");
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                require(rejected,
                        "K7 tiles64 exact-splits accepted malformed opt-in");
            }
            strict_checked = true;
            _putenv_s("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS", "0");
        }

        require(!candidate->target_k7_tiles64_exact_splits_candidate(
                    x.metadata, 16, admit) &&
                    candidate->target_k7_tiles64_exact_splits_candidate(
                        x.metadata, 17, admit) &&
                    !candidate->target_k7_tiles64_exact_splits_candidate(
                        x.metadata, max_rows + 1, admit) &&
                    !candidate->target_k7_tiles64_exact_splits_candidate(
                        x.metadata, max_rows, Admission::ordinary),
                "K7 tiles64 exact-splits admission boundary");

        const std::size_t input_extent =
            static_cast<std::size_t>(max_rows) * input_width;
        const std::size_t output_extent =
            static_cast<std::size_t>(max_rows) * output_width;
        std::vector<std::uint16_t> input_before(input_extent),
            input_after(input_extent);
        cuda_check(cudaMemcpy(input_before.data(), x.input,
                              input_extent * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),
                   "K7 tiles64 exact-splits input snapshot");
        const std::vector<std::uint16_t> initial(
            output_extent + 2 * guard, 0x3555);
        std::vector<std::uint16_t> a(initial.size()), b(initial.size()),
            repeat(initial.size());
        DeviceBuffer old_output(initial.size() * sizeof(std::uint16_t));
        DeviceBuffer new_output(initial.size() * sizeof(std::uint16_t));
        auto* old_base = static_cast<std::uint16_t*>(old_output.get());
        auto* new_base = static_cast<std::uint16_t*>(new_output.get());
        std::uint64_t expected_calls = 0;

        for (int rows : {17, 31, 32, 127, 128}) {
            const std::string control_route =
                control->dispatch_name(x.metadata, rows, admit);
            const std::string candidate_route =
                candidate->dispatch_name(x.metadata, rows, admit);
            require(candidate_route ==
                        "target_wide_prefill_k7_tiles64_exact_splits",
                    "K7 tiles64 exact-splits candidate route");
            require(control_route == (rows < 32
                        ? "target_wide_prefill_staged"
                        : "target_wide_prefill_rowpair_k7"),
                    "K7 tiles64 exact-splits retained control route");
            cuda_check(cudaMemcpy(old_base, initial.data(),
                                  initial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "K7 tiles64 exact-splits control guards");
            cuda_check(cudaMemcpy(new_base, initial.data(),
                                  initial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "K7 tiles64 exact-splits candidate guards");
            control->forward(x.weights, x.metadata, x.input, old_base + guard,
                             rows, x.stream, admit);
            candidate->forward(x.weights, x.metadata, x.input,
                               new_base + guard, rows, x.stream, admit);
            cuda_check(cudaStreamSynchronize(x.stream),
                       "K7 tiles64 exact-splits compare ready");
            cuda_check(cudaMemcpy(a.data(), old_base,
                                  a.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "K7 tiles64 exact-splits control output");
            cuda_check(cudaMemcpy(b.data(), new_base,
                                  b.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "K7 tiles64 exact-splits candidate output");
            candidate->forward(x.weights, x.metadata, x.input,
                               new_base + guard, rows, x.stream, admit);
            cuda_check(cudaStreamSynchronize(x.stream),
                       "K7 tiles64 exact-splits repeat ready");
            cuda_check(cudaMemcpy(repeat.data(), new_base,
                                  repeat.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "K7 tiles64 exact-splits repeat output");
            cuda_check(cudaMemcpy(input_after.data(), x.input,
                                  input_extent * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "K7 tiles64 exact-splits input after");
            const std::size_t active =
                static_cast<std::size_t>(rows) * output_width;
            const bool guards =
                std::all_of(b.begin(), b.begin() + guard,
                            [](auto value) { return value == 0x3555; }) &&
                std::all_of(b.begin() + guard + active, b.end(),
                            [](auto value) { return value == 0x3555; });
            const bool exact = a == b;
            const bool repeated = b == repeat;
            const bool unchanged = input_before == input_after;
            ++expected_calls;
            const auto candidate_calls =
                candidate->k7_tiles64_exact_splits_calls();
            csv << input_width << ',' << output_width << ',' << x.operation
                << ',' << rows << ',' << active << ',' << control_route << ','
                << candidate_route << ',' << exact << ',' << repeated << ','
                << guards << ',' << unchanged << ',' << candidate_calls << '\n';
            csv.flush();
            require(exact && repeated && guards && unchanged &&
                        control->k7_tiles64_exact_splits_calls() == 0 &&
                        candidate_calls == 2 * expected_calls,
                    "K7 tiles64 exact-splits differential failure");
            ++cases;
            values += active;
        }
        require(csv.good(), "K7 tiles64 exact-splits differential evidence");
    }
};

void run_prefill_k7_tiles64_exact_splits_qualification(Exl3TextModel& target) {
    require(env("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS") == "0" &&
                env("NINFER_EXL3_PREFILL_ROWPAIR_K7") == "1" &&
                env("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A") == "1" &&
                env("NINFER_EXL3_PREFILL_DIRECT_ASYNC_ALL") == "1" &&
                env("NINFER_EXL3_PREFILL_DIRECT_PARTIALS") == "1",
            "K7 tiles64 exact-splits qualification flags");
    auto ids = load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size() >= 16 +
                PrefillK7Tiles64ExactSplitsQualification::max_rows,
            "K7 tiles64 exact-splits prompt extent");
    auto context = target.create_context(true);
    PrefillK7Tiles64ExactSplitsQualification qualification(
        env("NINFER_E5A4_OUT"));
    context->prefill(std::span<const std::int64_t>(ids.data(), 16));
    context->set_target_projection_observer_for_test(
        PrefillK7Tiles64ExactSplitsQualification::callback, &qualification,
        nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(std::span<const std::int64_t>(
        ids.data() + 16,
        PrefillK7Tiles64ExactSplitsQualification::max_rows));
    context->finish_exact_prefill();
    cuda_check(cudaDeviceSynchronize(),
               "K7 tiles64 exact-splits qualification complete");
    context->set_target_projection_observer_for_test(nullptr, nullptr);
    const std::set<std::pair<int, int>> expected{
        {5120, 1024}, {5120, 17408}, {6144, 5120}};
    require(qualification.observations == 400 &&
                qualification.shapes == expected &&
                qualification.cases == 15,
            "K7 tiles64 exact-splits real-caller shape coverage");
    std::ofstream result(qualification.directory / "result.json");
    result << nlohmann::ordered_json{
        {"status", "PASS_PREFILL_K7_TILES64_EXACT_SPLITS"},
        {"observations", qualification.observations},
        {"shapes", qualification.shapes.size()},
        {"cases", qualification.cases},
        {"values", qualification.values},
        {"exact", true}, {"repeat", true}, {"guards", true},
        {"input_unchanged", true}, {"strict_flag", true}}.dump(2);
    require(result.good(), "K7 tiles64 exact-splits result evidence");
    std::cout << "PREFILL_K7_TILES64_EXACT_SPLITS PASS observations="
              << qualification.observations << " shapes="
              << qualification.shapes.size() << " cases="
              << qualification.cases << " values=" << qualification.values
              << " strict_flag=1 exact=1 repeat=1 guards=1 input_unchanged=1\n";
}
