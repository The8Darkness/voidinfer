#pragma once

struct TargetQK6Qualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit = Admission::target_continuation_q;
    static constexpr int output = 12288;
    static constexpr std::size_t guard = 128;
    static constexpr std::size_t extent = 8u * output;
    std::filesystem::path directory;
    std::ofstream csv;
    std::vector<int> layers;

    explicit TargetQK6Qualification(const std::filesystem::path& path) : directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),
                "Q K6 qualification output must be new");
        std::filesystem::create_directories(path);
        csv.open(path / "operators.csv");
        csv << "layer,rows,elements,exact,dispatch,canaries,repeat,finite\n";
    }

    template<class T>
    void save(const std::string& name, const std::vector<T>& data) {
        std::ofstream file(directory / name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
        require(file.good(), "Q K6 raw evidence write");
    }

    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& observation,
                         void* user) {
        if (std::string(observation.operation) == "q" && observation.metadata.K == 6)
            static_cast<TargetQK6Qualification*>(user)->check(observation);
    }

    void check(const ninfer::exl3::Exl3TargetProjectionObservation& observation) {
        require(observation.rows == 8 && observation.metadata.K == 6 &&
                    observation.metadata.in_features == kHidden &&
                    observation.metadata.out_features == output &&
                    std::string(observation.operation) == "q",
                "Q K6 capture extent");
        require(std::find(layers.begin(), layers.end(), observation.layer) == layers.end(),
                "duplicate Q K6 capture");
        require(observation.layer >= 0 && observation.layer < 64 &&
                    observation.layer % 4 == 3,
                "Q K6 full-attention owner layer");
        layers.push_back(observation.layer);

        cuda_check(cudaStreamSynchronize(observation.stream), "Q K6 input ready");
        std::vector<std::uint16_t> input(8u * kHidden);
        cuda_check(cudaMemcpy(input.data(), observation.input, input.size() * 2,
                              cudaMemcpyDeviceToHost),
                   "Q K6 input copy");
        require(std::all_of(input.begin(), input.end(), [](std::uint16_t value) {
                    return (value & 0x7c00) != 0x7c00;
                }),
                "Q K6 finite represented inputs");

        _putenv_s("NINFER_EXL3_TARGET_Q_K6_SMALL_M", "0");
        Exl3CudaLinearWorkspace baseline(kHidden, output, 8, false, false, false,
                                         false, false, false, false, {}, {}, false,
                                         false, true);
        _putenv_s("NINFER_EXL3_TARGET_Q_K6_SMALL_M", "1");
        Exl3CudaLinearWorkspace candidate(kHidden, output, 8, false, false, false,
                                          false, false, false, false, {}, {}, false,
                                          false, true);
        Exl3CudaLinearWorkspace unowned(kHidden, output, 8);
        Exl3CudaLinearWorkspace short_owner(kHidden, output, 4, false, false, false,
                                            false, false, false, false, {}, {}, false,
                                            false, true);
        _putenv_s("NINFER_EXL3_TARGET_Q_K6_SMALL_M", "0");

        require(candidate.target_q_k6_small_m_candidate(observation.metadata, 8, admit) &&
                    !baseline.target_q_k6_small_m_candidate(observation.metadata, 8, admit),
                "Q K6 latched enable");
        require(!unowned.target_q_k6_small_m_candidate(observation.metadata, 8, admit) &&
                    !candidate.target_q_k6_small_m_candidate(
                        observation.metadata, 8, Admission::ordinary) &&
                    !short_owner.target_q_k6_small_m_candidate(observation.metadata, 8, admit),
                "Q K6 ownership/capacity");
        for (int rows = 0; rows <= 17; ++rows)
            require(candidate.target_q_k6_small_m_candidate(
                        observation.metadata, rows, admit) == (rows >= 2 && rows <= 8),
                    "Q K6 width admission");
        for (int bad = 0; bad < 6; ++bad) {
            auto metadata = observation.metadata;
            if (bad == 0) metadata.K = 7;
            if (bad == 1) metadata.mcg = true;
            if (bad == 2) metadata.mul1 = false;
            if (bad == 3) metadata.has_bias = true;
            if (bad == 4) metadata.in_features = 4096;
            if (bad == 5) metadata.out_features = 10240;
            require(!candidate.target_q_k6_small_m_candidate(metadata, 8, admit),
                    "Q K6 metadata admission");
        }
        if (layers.size() == 1) {
            for (const char* disabled : {"", "0", "2", "true", "01"}) {
                _putenv_s("NINFER_EXL3_TARGET_Q_K6_SMALL_M", disabled);
                Exl3CudaLinearWorkspace off(kHidden, output, 8, false, false, false,
                                             false, false, false, false, {}, {}, false,
                                             false, true);
                require(!off.target_q_k6_small_m_candidate(observation.metadata, 8, admit),
                        "Q K6 exact flag1 required");
            }
        }
        _putenv_s("NINFER_EXL3_TARGET_Q_K6_SMALL_M", "0");
        require(std::string(candidate.dispatch_name(observation.metadata, 8, admit)) ==
                    "target_q_k6_small_m_mma_split" &&
                    std::string(candidate.dispatch_name(observation.metadata, 1, admit)) ==
                    "generic_mma_split",
                "Q K6 dispatch");

        DeviceBuffer reference_device((extent + 2 * guard) * 2);
        DeviceBuffer candidate_device((extent + 2 * guard) * 2);
        auto* reference_output = static_cast<std::uint16_t*>(reference_device.get()) + guard;
        auto* candidate_output = static_cast<std::uint16_t*>(candidate_device.get()) + guard;
        std::vector<std::uint16_t> initial(extent + 2 * guard, 0x3555);
        std::vector<std::uint16_t> reference(initial), got(initial), repeat(initial);
        const auto serial = [&](int rows, const std::uint16_t* source) {
            for (int row = 0; row < rows; ++row)
                baseline.forward(observation.weights, observation.metadata,
                                 source + static_cast<std::size_t>(row) * kHidden,
                                 reference_output + static_cast<std::size_t>(row) * output,
                                 1, observation.stream);
        };

        for (int rows = 1; rows <= 8; ++rows) {
            cuda_check(cudaMemcpy(reference_device.get(), initial.data(), initial.size() * 2,
                                  cudaMemcpyHostToDevice),
                       "Q K6 reference guards");
            cuda_check(cudaMemcpy(candidate_device.get(), initial.data(), initial.size() * 2,
                                  cudaMemcpyHostToDevice),
                       "Q K6 candidate guards");
            serial(rows, observation.input);
            candidate.forward(observation.weights, observation.metadata, observation.input,
                              candidate_output, rows, observation.stream, admit);
            cuda_check(cudaStreamSynchronize(observation.stream), "Q K6 exact ready");
            cuda_check(cudaMemcpy(reference.data(), reference_device.get(), reference.size() * 2,
                                  cudaMemcpyDeviceToHost),
                       "Q K6 reference copy");
            cuda_check(cudaMemcpy(got.data(), candidate_device.get(), got.size() * 2,
                                  cudaMemcpyDeviceToHost),
                       "Q K6 candidate copy");
            const auto tag = std::to_string(observation.layer) + "-m" + std::to_string(rows);
            if (rows == 8 || reference != got) {
                save(tag + "-reference.bin", reference);
                save(tag + "-candidate.bin", got);
            }
            require(reference == got, "Q K6 exact output mismatch " + tag);
            for (std::size_t i = 0; i < got.size(); ++i) {
                if (i < guard || i >= guard + static_cast<std::size_t>(rows) * output)
                    require(got[i] == 0x3555, "Q K6 output tail/canary");
                else
                    require((got[i] & 0x7c00) != 0x7c00, "Q K6 nonfinite output");
            }
            candidate.forward(observation.weights, observation.metadata, observation.input,
                              candidate_output, rows, observation.stream, admit);
            cuda_check(cudaStreamSynchronize(observation.stream), "Q K6 repeat ready");
            cuda_check(cudaMemcpy(repeat.data(), candidate_device.get(), repeat.size() * 2,
                                  cudaMemcpyDeviceToHost),
                       "Q K6 repeat copy");
            require(repeat == got, "Q K6 repeat mismatch");
            csv << observation.layer << ',' << rows << ',' << rows * output << ",1,"
                << candidate.dispatch_name(observation.metadata, rows, admit)
                << ",1,1,1\n";
            csv.flush();
            require(csv.good(), "Q K6 csv write");
        }

        save(std::to_string(observation.layer) + "-input.bin", input);
        std::vector<std::uint16_t> after(input.size());
        cuda_check(cudaMemcpy(after.data(), observation.input, input.size() * 2,
                              cudaMemcpyDeviceToHost),
                   "Q K6 immutable input");
        require(after == input, "Q K6 input mutation");

        auto changed = input;
        for (auto& bits : changed) bits ^= 0x8000;
        DeviceBuffer changed_device(changed.size() * 2);
        cuda_check(cudaMemcpy(changed_device.get(), changed.data(), changed.size() * 2,
                              cudaMemcpyHostToDevice),
                   "Q K6 changed input");
        auto* changed_input = static_cast<const std::uint16_t*>(changed_device.get());
        cuda_check(cudaMemcpy(reference_device.get(), initial.data(), initial.size() * 2,
                              cudaMemcpyHostToDevice),
                   "Q K6 reuse reference guards");
        cuda_check(cudaMemcpy(candidate_device.get(), initial.data(), initial.size() * 2,
                              cudaMemcpyHostToDevice),
                   "Q K6 reuse candidate guards");
        serial(3, changed_input);
        candidate.forward(observation.weights, observation.metadata, changed_input,
                          candidate_output, 3, observation.stream, admit);
        cuda_check(cudaStreamSynchronize(observation.stream), "Q K6 reuse ready");
        cuda_check(cudaMemcpy(reference.data(), reference_device.get(), reference.size() * 2,
                              cudaMemcpyDeviceToHost),
                   "Q K6 reused reference");
        cuda_check(cudaMemcpy(repeat.data(), candidate_device.get(), repeat.size() * 2,
                              cudaMemcpyDeviceToHost),
                   "Q K6 reused candidate");
        require(reference == repeat, "Q K6 changed-input M8-to-M3 workspace reuse");
    }
};

void run_target_q_k6_qualification(Exl3TextModel& target, Exl3Dflash2DraftModel& draft) {
    require(env("NINFER_EXL3_TARGET_Q_K6_SMALL_M") == "0",
            "Q K6 capture requires construction flag0");
    auto ids = load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size() >= 512, "Q K6 prompt extent");
    std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 512);
    TapStage stage;
    for (int tap = 0; tap < kTapCount; ++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16u * kHidden * 2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden * 2));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    auto context = target.create_context(true);
    require(context->try_enable_oscar_from_environment(), "Q K6 OSCAR required");
    draft.reset();
    ingest_prefix(*context, prompt, CommitSink{&draft, &stage});
    require(context->position() == 512 && draft.ring_count() == 512,
            "Q K6 capture prefix");
    const auto pending = sample_target(*context);
    std::vector<std::int64_t> block(8, kMaskToken);
    block[0] = pending;
    const auto proposals = draft.propose_cached(
        block, 512, context->target_embedding(), context->target_lm_head_weights(),
        context->target_lm_head_metadata(), kMaskToken);
    require(proposals.size() == 7, "Q K6 real proposals");
    block.assign(1, pending);
    block.insert(block.end(), proposals.begin(), proposals.end());

    TargetQK6Qualification qualification(env("NINFER_E5A4_OUT"));
    context->prepare_transaction();
    context->prepare_continuation(8);
    const auto ring = draft.ring_digest();
    context->set_target_projection_observer_for_test(
        TargetQK6Qualification::callback, &qualification, nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->begin_transaction();
    context->continue_rows(block);
    context->set_target_projection_observer_for_test(nullptr, nullptr);
    context->rollback_transaction();
    cuda_check(cudaDeviceSynchronize(), "Q K6 sweep ready");
    require(qualification.layers.size() == 12 && context->position() == 512 &&
                context->device_position_host() == 511 && sample_target(*context) == pending &&
                !context->transaction_active(),
            "Q K6 full layer coverage/rollback");
    require(draft.ring_digest() == ring && draft.ring_count() == 512 &&
                draft.ring_base_abs() == 0,
            "Q K6 ring unchanged");
    std::ofstream result(qualification.directory / "result.txt");
    result << "PASS operators=12 widths=1..8 cases=96 exact=bitwise guards=pass "
              "repeat=pass changed_input_reuse=pass\n";
    require(result.good(), "Q K6 result write");
    std::cout << "TARGET_Q_K6 PASS operators=12 cases=96\n";
}
