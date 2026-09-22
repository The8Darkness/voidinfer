#pragma once

// Qualification-only capture of represented inputs for the exact target K7
// O-projection family. Borrowed weight pointers remain valid only while the
// caller keeps the target model alive; captured input vectors own their data.
using TargetOK7CapturedProjection = TargetK6CapturedProjection;

void capture_target_o_k7_projection(
    const ninfer::exl3::Exl3TargetProjectionObservation& observation,
    void* user) {
    require(observation.rows == 8 && observation.operation != nullptr &&
                std::strcmp(observation.operation, "o") == 0 &&
                observation.metadata.K == 7 && observation.metadata.mul1 &&
                !observation.metadata.mcg && !observation.metadata.has_bias &&
                observation.metadata.in_features == 6144 &&
                observation.metadata.out_features == 5120,
            "target O K7 capture received an unexpected observation");
    capture_target_k6_projection(observation, user);
}

std::vector<TargetOK7CapturedProjection> run_target_o_k7_capture(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,
    const std::filesystem::path& output_directory, bool oscar) {
    require(prompt.size() == 512, "target O K7 capture requires ctx512");
    require(target.max_context() >= 520,
            "target O K7 capture requires context capacity at least 520");
    require(oscar, "target O K7 capture requires canonical OSCAR");
    require(env("NINFER_EXL3_TARGET_GATEUP_SMALL_M") == "1" &&
                env("NINFER_EXL3_TARGET_DOWN_SMALL_M") == "1",
            "target O K7 capture requires accepted gate/up and down candidates");
    for (const char* name : {
             "NINFER_EXL3_TARGET_PROJECTION_TIMING",
             "NINFER_EXL3_TARGET_PROJECTION_TIMING_OUT",
             "NINFER_EXL3_LARGE_DOWN_TOPOLOGY",
             "NINFER_EXL3_LARGE_DOWN_SPLITS",
             "NINFER_EXL3_GENERIC_SPLITS",
             "NINFER_EXL3_H6_SMALL_M",
             "NINFER_EXL3_DRAFT_SMALL_M"}) {
        require(env(name).empty(),
                std::string("target O K7 capture forbids override ") + name);
    }

    constexpr int input_features = 6144;
    constexpr int output_features = 5120;
    constexpr int expected_records = 32;
    const std::array<int, expected_records> expected_layers = {{
        2, 4, 8, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 39, 40, 43, 44, 46, 47, 50, 51, 52, 53, 56, 57}};
    std::array<bool, 64> expected{};
    for (const int layer : expected_layers)
        expected[static_cast<std::size_t>(layer)] = true;

    const bool write_capture = !output_directory.empty();
    if (write_capture) {
        require(!std::filesystem::exists(output_directory),
                "target O K7 capture output path must be new");
        require(std::filesystem::create_directories(output_directory),
                "cannot create target O K7 capture output directory");
    }

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
            "target O K7 capture could not enable OSCAR");
    draft.reset();
    ingest_prefix(*context, prompt, CommitSink{&draft, &stage});
    require(context->position() == 512 && context->device_position_host() == 511 &&
                draft.ring_count() == 512 && draft.ring_base_abs() == 0,
            "target O K7 capture prefix state mismatch");

    const std::int64_t pending = sample_target(*context);
    std::vector<std::int64_t> draft_block(8, kMaskToken);
    draft_block[0] = pending;
    const auto proposals = draft.propose_cached(
        draft_block, 512, context->target_embedding(),
        context->target_lm_head_weights(), context->target_lm_head_metadata(),
        kMaskToken);
    const std::vector<std::int64_t> qualified_proposals =
        {198, 27, 15704, 28, 2447, 29, 198};
    require(pending == 29 && proposals == qualified_proposals,
            "target O K7 request differs from qualified code512 ledger");
    std::ostringstream proposal_ids;
    for (std::size_t index = 0; index < proposals.size(); ++index) {
        if (index) proposal_ids << '|';
        proposal_ids << proposals[index];
    }
    std::vector<std::int64_t> target_block{pending};
    target_block.insert(target_block.end(), proposals.begin(), proposals.end());

    context->prepare_transaction();
    context->prepare_continuation(8);
    const auto ring_before = draft.ring_digest();
    std::vector<TargetOK7CapturedProjection> captured;
    captured.reserve(expected_records);
    context->set_target_projection_observer_for_test(
        capture_target_o_k7_projection, &captured, nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::output_k7);
    bool observer_installed = true;
    try {
        context->begin_transaction();
        context->continue_rows(target_block);
        context->set_target_projection_observer_for_test(nullptr);
        observer_installed = false;
        context->rollback_transaction();
        cuda_check(cudaStreamSynchronize(nullptr),
                   "synchronize target O K7 capture rollback");
    } catch (...) {
        if (observer_installed)
            context->set_target_projection_observer_for_test(nullptr);
        if (context->transaction_active())
            context->rollback_transaction();
        cudaStreamSynchronize(nullptr);
        throw;
    }
    require(!context->transaction_active() && context->position() == 512 &&
                context->device_position_host() == 511 &&
                sample_target(*context) == pending,
            "target O K7 capture did not restore target state");
    require(draft.ring_digest() == ring_before && draft.ring_count() == 512 &&
                draft.ring_base_abs() == 0,
            "target O K7 capture changed the draft ring");

    require(captured.size() == expected_records,
            "target O K7 capture observed the wrong record count");
    std::array<bool, 64> seen{};
    int full_layers = 0;
    int gdn_layers = 0;
    std::size_t total_input_bytes = 0;
    std::ofstream manifest;
    if (write_capture) {
        manifest.open(output_directory / "manifest.csv");
        require(manifest.good(), "cannot create target O K7 capture manifest");
        manifest << "layer,operator,rows,in_features,out_features,K,mul1,mcg,has_bias,"
                    "m1_dispatch,m1_calls,input_bytes,distinct_rows,file,pending,proposal_ids\n";
    }
    for (const auto& item : captured) {
        require(item.layer >= 0 && item.layer < 64 && item.operation == "o" &&
                    expected[static_cast<std::size_t>(item.layer)] &&
                    !seen[static_cast<std::size_t>(item.layer)],
                "target O K7 capture observed unexpected or duplicate projection");
        seen[static_cast<std::size_t>(item.layer)] = true;
        require(item.metadata.K == 7 && item.metadata.mul1 &&
                    !item.metadata.mcg && !item.metadata.has_bias &&
                    item.metadata.in_features == input_features &&
                    item.metadata.out_features == output_features &&
                    item.input.size() == 8u * input_features &&
                    item.m1_dispatch == "generic_mma_split",
                "target O K7 capture shape, metadata or M1 dispatch mismatch");
        require(item.weights.trellis && item.weights.suh && item.weights.svh &&
                    item.weights.mul1,
                "target O K7 capture borrowed null weights");
        require(std::all_of(item.input.begin(), item.input.end(),
                            [](std::uint16_t bits) {
                                return std::isfinite(half_to_float(bits));
                            }) &&
                    std::any_of(item.input.begin(), item.input.end(),
                                [](std::uint16_t bits) {
                                    return (bits & 0x7fffU) != 0U;
                                }),
                "target O K7 capture input is nonfinite or all zero");
        const int distinct_rows = target_k6_distinct_rows(item);
        require(distinct_rows == 8,
                "target O K7 capture requires eight bitwise-distinct rows");
        const std::size_t input_bytes =
            item.input.size() * sizeof(std::uint16_t);
        total_input_bytes += input_bytes;
        if (item.layer % 4 == 3) ++full_layers;
        else ++gdn_layers;

        if (write_capture) {
            const std::string filename = "layer" + std::to_string(item.layer) +
                "-o-6144x5120.f16";
            const auto path = output_directory / filename;
            require(!std::filesystem::exists(path),
                    "target O K7 capture refuses to overwrite input file");
            std::ofstream output(path, std::ios::binary);
            require(output.good(), "cannot create target O K7 captured input");
            output.write(reinterpret_cast<const char*>(item.input.data()),
                         static_cast<std::streamsize>(input_bytes));
            require(output.good(), "cannot write target O K7 captured input");
            manifest << item.layer << ",o,8," << item.metadata.in_features << ','
                     << item.metadata.out_features << ',' << item.metadata.K << ','
                     << item.metadata.mul1 << ',' << item.metadata.mcg << ','
                     << item.metadata.has_bias << ',' << item.m1_dispatch << ",8,"
                     << input_bytes << ',' << distinct_rows << ',' << filename << ','
                     << pending << ',' << proposal_ids.str() << '\n';
        }
    }
    for (int layer = 0; layer < 64; ++layer)
        require(seen[static_cast<std::size_t>(layer)] ==
                    expected[static_cast<std::size_t>(layer)],
                "target O K7 canonical layer inventory mismatch");
    require(full_layers == 8 && gdn_layers == 24,
            "target O K7 full/GDN inventory mismatch");
    require(total_input_bytes == 3145728u,
            "target O K7 total input byte count mismatch");
    if (write_capture) {
        manifest.flush();
        require(manifest.good(), "target O K7 capture manifest write failed");
        std::cout << "TARGETOK7CAPTURE PASS records=32 ctx=512 rows=8"
                  << " pending=" << pending
                  << " proposal_ids=" << proposal_ids.str()
                  << " total_input_bytes=" << total_input_bytes
                  << " full_layers=" << full_layers
                  << " gdn_layers=" << gdn_layers
                  << " files=32 rollback=1 target_restored=1 ring_unchanged=1"
                  << std::endl;
    }
    return captured;
}
