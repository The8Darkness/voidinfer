#pragma once

struct TargetK5SmallMBatchQualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    std::filesystem::path directory;
    std::ofstream csv;
    std::vector<std::string> shapes;

    explicit TargetK5SmallMBatchQualification(const std::filesystem::path& path)
        : directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),
                "target K5 batch qualification output must be new");
        std::filesystem::create_directories(path);
        csv.open(path / "operators.csv");
        csv << "operator,in_features,out_features,rows,elements,exact,guards,dispatch,calls\n";
    }

    static Admission admission(const char* operation) noexcept {
        const std::string op(operation ? operation : "");
        if (op == "q") return Admission::target_continuation_q;
        if (op == "qkv") return Admission::target_continuation_qkv;
        if (op == "z") return Admission::target_continuation_z;
        if (op == "k" || op == "v") return Admission::target_continuation_kv;
        if (op == "o") return Admission::target_continuation_o;
        if (op == "down") return Admission::target_continuation_down;
        return Admission::ordinary;
    }

    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,
                         void* user) {
        auto* self = static_cast<TargetK5SmallMBatchQualification*>(user);
        if (x.metadata.K != 5 || admission(x.operation) == Admission::ordinary)
            return;
        const std::string key = std::string(x.operation) + "-" +
            std::to_string(x.metadata.in_features) + "x" +
            std::to_string(x.metadata.out_features);
        if (std::find(self->shapes.begin(), self->shapes.end(), key) !=
            self->shapes.end())
            return;
        self->check(x, key);
        self->shapes.push_back(key);
    }

    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x,
               const std::string& key) {
        require(x.rows == 8 && x.metadata.mul1 && !x.metadata.mcg &&
                    !x.metadata.has_bias,
                "target K5 batch capture contract");
        const auto admit = admission(x.operation);
        _putenv_s("NINFER_EXL3_TARGET_K5_SMALL_M_BATCH", "0");
        Exl3CudaLinearWorkspace baseline(
            x.metadata.in_features, x.metadata.out_features, 8,
            false, true, true, true, true, false, false, {}, {}, true, true, true);
        _putenv_s("NINFER_EXL3_TARGET_K5_SMALL_M_BATCH", "1");
        Exl3CudaLinearWorkspace candidate(
            x.metadata.in_features, x.metadata.out_features, 8,
            false, true, true, true, true, false, false, {}, {}, true, true, true);
        _putenv_s("NINFER_EXL3_TARGET_K5_SMALL_M_BATCH", "0");
        require(!candidate.target_k5_small_m_batch_candidate(
                    x.metadata, 1, admit) &&
                    candidate.target_k5_small_m_batch_candidate(
                        x.metadata, 2, admit) &&
                    candidate.target_k5_small_m_batch_candidate(
                        x.metadata, 8, admit) &&
                    !candidate.target_k5_small_m_batch_candidate(
                        x.metadata, 9, admit) &&
                    !candidate.target_k5_small_m_batch_candidate(
                        x.metadata, 8, Admission::ordinary),
                "target K5 batch width/owner admission");
        require(std::string(candidate.dispatch_name(x.metadata, 8, admit)) ==
                    "target_k5_small_m_batch",
                "target K5 batch dispatch identity");

        constexpr std::size_t guard = 128;
        const std::size_t extent = 8u * x.metadata.out_features;
        DeviceBuffer reference_device((extent + 2 * guard) * sizeof(std::uint16_t));
        DeviceBuffer candidate_device((extent + 2 * guard) * sizeof(std::uint16_t));
        auto* reference_output =
            static_cast<std::uint16_t*>(reference_device.get()) + guard;
        auto* candidate_output =
            static_cast<std::uint16_t*>(candidate_device.get()) + guard;
        std::vector<std::uint16_t> initial(extent + 2 * guard, 0x3555);
        std::vector<std::uint16_t> reference(initial), actual(initial);

        for (const int rows : {2, 8}) {
            cuda_check(cudaMemcpy(reference_device.get(), initial.data(),
                                  initial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target K5 batch reference guards");
            cuda_check(cudaMemcpy(candidate_device.get(), initial.data(),
                                  initial.size() * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice),
                       "target K5 batch candidate guards");
            for (int row = 0; row < rows; ++row)
                baseline.forward(
                    x.weights, x.metadata,
                    x.input + static_cast<std::size_t>(row) * x.metadata.in_features,
                    reference_output +
                        static_cast<std::size_t>(row) * x.metadata.out_features,
                    1, x.stream);
            candidate.forward(x.weights, x.metadata, x.input, candidate_output,
                              rows, x.stream, admit);
            cuda_check(cudaStreamSynchronize(x.stream),
                       "target K5 batch exact ready");
            cuda_check(cudaMemcpy(reference.data(), reference_device.get(),
                                  reference.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target K5 batch reference copy");
            cuda_check(cudaMemcpy(actual.data(), candidate_device.get(),
                                  actual.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost),
                       "target K5 batch candidate copy");
            require(reference == actual,
                    "target K5 batch exact mismatch " + key + " M" +
                        std::to_string(rows));
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (i < guard || i >= guard +
                    static_cast<std::size_t>(rows) * x.metadata.out_features)
                    require(actual[i] == 0x3555,
                            "target K5 batch inactive output/canary");
            }
            csv << x.operation << ',' << x.metadata.in_features << ','
                << x.metadata.out_features << ',' << rows << ','
                << static_cast<std::size_t>(rows) * x.metadata.out_features
                << ",1,1," << candidate.dispatch_name(x.metadata, rows, admit)
                << ',' << candidate.target_k5_small_m_batch_calls() << '\n';
        }
        require(candidate.target_k5_small_m_batch_calls() == 2,
                "target K5 batch dispatch count");
        csv.flush();
        require(csv.good(), "target K5 batch result write");
    }
};

void run_target_k5_small_m_batch_qualification(Exl3TextModel& target,
                                                Exl3Dflash2DraftModel& draft) {
    require(env("NINFER_EXL3_TARGET_K5_SMALL_M_BATCH") == "0",
            "target K5 batch capture requires construction flag0");
    auto ids = load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size() >= 512, "target K5 batch prompt extent");
    std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
    TapStage stage;
    for (int tap = 0; tap < kTapCount; ++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16u * kHidden * 2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden * 2));
        stage.bulk_ptrs.push_back(
            static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(
            static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    auto context = target.create_context(true);
    draft.reset();
    context->prefill(std::span<const std::int64_t>(prompt.data(), 16));
    commit_current_rows(*context, draft, stage, 16, 0);
    int cursor = 16;
    while (cursor < static_cast<int>(prompt.size())) {
        const int remaining = static_cast<int>(prompt.size()) - cursor;
        const int count = remaining >= 128 ? 128 : 16;
        require(count == 16 || count == 128,
                "target K5 batch exact-host prefill partition");
        context->append_exact_prefill_wide(
            std::span<const std::int64_t>(prompt.data() + cursor, count));
        commit_current_rows(*context, draft, stage, count, cursor);
        cursor += count;
    }
    require(context->position() == 512 && draft.ring_count() == 512,
            "target K5 batch exact-host prefix");
    const auto pending = sample_target(*context);
    std::vector<std::int64_t> block(8, kMaskToken);
    block[0] = pending;
    const auto proposals = draft.propose_cached(
        block, 512, context->target_embedding(),
        context->target_lm_head_weights(), context->target_lm_head_metadata(),
        kMaskToken);
    require(proposals.size() == 7, "target K5 batch proposal extent");
    block.assign(1, pending);
    block.insert(block.end(), proposals.begin(), proposals.end());

    TargetK5SmallMBatchQualification check(env("NINFER_E5A4_OUT"));
    context->prepare_transaction();
    context->prepare_continuation(8);
    context->set_target_projection_observer_for_test(
        TargetK5SmallMBatchQualification::callback, &check, nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->begin_transaction();
    context->continue_rows(block);
    context->set_target_projection_observer_for_test(nullptr, nullptr);
    context->rollback_transaction();
    cuda_check(cudaDeviceSynchronize(), "target K5 batch sweep ready");
    require(check.shapes.size() >= 4,
            "target K5 batch real caller shape coverage");
    std::ofstream result(check.directory / "result.txt");
    result << "PASS shapes=" << check.shapes.size()
           << " widths=2,8 bit_exact=1 guards=1 dispatch_counted=1\n";
    require(result.good(), "target K5 batch summary write");
    std::cout << "TARGET_K5_SMALL_M_BATCH PASS shapes=" << check.shapes.size()
              << " cases=" << check.shapes.size() * 2 << "\n";
}
