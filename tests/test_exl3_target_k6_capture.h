#pragma once

struct TargetK6CapturedProjection {
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    std::vector<std::uint16_t> input;
    int layer = -1;
    std::string operation;
    std::string m1_dispatch;
};

void capture_target_k6_projection(
    const ninfer::exl3::Exl3TargetProjectionObservation& observation,
    void* user) {
    auto& captured = *static_cast<std::vector<TargetK6CapturedProjection>*>(user);
    require(observation.input != nullptr && observation.operation != nullptr &&
                observation.m1_dispatch != nullptr,
            "target K6 capture received an incomplete observation");
    TargetK6CapturedProjection item;
    item.weights = observation.weights;
    item.metadata = observation.metadata;
    item.layer = observation.layer;
    item.operation = observation.operation;
    item.m1_dispatch = observation.m1_dispatch;
    item.input.resize(static_cast<std::size_t>(observation.rows) *
                      observation.metadata.in_features);
    cuda_check(cudaMemcpyAsync(item.input.data(), observation.input,
                               item.input.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, observation.stream),
               "capture target K6 represented input");
    cuda_check(cudaStreamSynchronize(observation.stream),
               "synchronize target K6 represented input capture");
    captured.push_back(std::move(item));
}

int target_k6_distinct_rows(const TargetK6CapturedProjection& item) {
    int distinct = 0;
    const std::size_t width = static_cast<std::size_t>(item.metadata.in_features);
    for (int row = 0; row < 8; ++row) {
        bool unique = true;
        const auto begin = item.input.begin() + static_cast<std::size_t>(row) * width;
        for (int prior = 0; prior < row; ++prior) {
            const auto prior_begin =
                item.input.begin() + static_cast<std::size_t>(prior) * width;
            if (std::equal(begin, begin + width, prior_begin)) {
                unique = false;
                break;
            }
        }
        distinct += unique;
    }
    return distinct;
}

std::vector<TargetK6CapturedProjection> run_target_k6_capture_selected(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,
    const std::filesystem::path& output_directory, bool oscar,
    ninfer::exl3::Exl3TargetProjectionObserverSelection selection) {
    require(prompt.size() == 512, "target K6 capture requires ctx512");
    require(target.max_context() >= 520,
            "target K6 capture requires context capacity at least 520");
    require(oscar, "target K6 capture requires canonical OSCAR");
    const bool write_capture = !output_directory.empty();
    const bool capture_down_k7 = selection ==
        ninfer::exl3::Exl3TargetProjectionObserverSelection::down_k7;
    const bool capture_gateup_k7 = selection ==
        ninfer::exl3::Exl3TargetProjectionObserverSelection::gate_up_k7;
    const bool capture_down = capture_down_k7 || selection ==
        ninfer::exl3::Exl3TargetProjectionObserverSelection::down_k6;
    require(capture_down || capture_gateup_k7 || selection ==
                ninfer::exl3::Exl3TargetProjectionObserverSelection::gate_up_k6,
            "unsupported target projection capture selection");
    const int expected_k = (capture_down_k7 || capture_gateup_k7) ? 7 : 6;
    const int expected_records = capture_gateup_k7 ? 4 :
        (capture_down_k7 ? 19 : (capture_down ? 43 : 94));
    const int input_features = capture_down ? 17408 : kHidden;
    const int output_features = capture_down ? kHidden : 17408;
    if (write_capture) {
        require(!std::filesystem::exists(output_directory),
                "target K6 capture output path must be new");
        require(std::filesystem::create_directories(output_directory),
                "cannot create target K6 capture output directory");
    }

    TapStage stage;
    for (int tap = 0; tap < kTapCount; ++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(16) * kHidden * sizeof(std::uint16_t)));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t)));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }

    auto context = target.create_context(true);
    require(context->try_enable_oscar_from_environment(),
            "target K6 capture could not enable OSCAR");
    draft.reset();
    ingest_prefix(*context, prompt, CommitSink{&draft, &stage});
    require(context->position() == 512 && context->device_position_host() == 511 &&
                draft.ring_count() == 512 && draft.ring_base_abs() == 0,
            "target K6 capture prefix state mismatch");

    const std::int64_t pending = sample_target(*context);
    std::vector<std::int64_t> draft_block(8, kMaskToken);
    draft_block[0] = pending;
    const auto proposals = draft.propose_cached(
        draft_block, 512, context->target_embedding(),
        context->target_lm_head_weights(), context->target_lm_head_metadata(),
        kMaskToken);
    require(proposals.size() == 7, "target K6 capture proposal count mismatch");
    require(std::all_of(proposals.begin(), proposals.end(), [](std::int64_t token) {
                return token >= 0 && token < kVocab;
            }),
            "target K6 capture proposal outside vocabulary");
    if (capture_down) {
        const std::vector<std::int64_t> qualified_proposals =
            {198, 27, 15704, 28, 2447, 29, 198};
        require(pending == 29 && proposals == qualified_proposals,
                "target down K6 request differs from qualified code512 ledger");
    }
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
    std::vector<TargetK6CapturedProjection> captured;
    captured.reserve(static_cast<std::size_t>(expected_records));
    context->set_target_projection_observer_for_test(
        capture_target_k6_projection, &captured, nullptr, selection);
    struct ObserverReset {
        Exl3TextContext* context = nullptr;
        ~ObserverReset() {
            if (context) context->set_target_projection_observer_for_test(nullptr);
        }
    } observer_reset{context.get()};
    context->begin_transaction();
    context->continue_rows(target_block);
    context->set_target_projection_observer_for_test(nullptr);
    observer_reset.context = nullptr;
    context->rollback_transaction();
    cuda_check(cudaStreamSynchronize(nullptr),
               "synchronize target K6 capture rollback");
    require(!context->transaction_active() && context->position() == 512 &&
                context->device_position_host() == 511 &&
                sample_target(*context) == pending,
            "target K6 capture did not restore target state");
    require(draft.ring_digest() == ring_before && draft.ring_count() == 512 &&
                draft.ring_base_abs() == 0,
            "target K6 capture changed the draft ring");

    require(captured.size() == static_cast<std::size_t>(expected_records),
            "target K6 capture observed the wrong record count");
    std::array<std::array<bool, 2>, 64> seen{};
    std::array<bool, 64> captured_layers{};
    int full_layers = 0;
    int gdn_layers = 0;
    std::size_t total_input_bytes = 0;
    std::ofstream manifest;
    if (write_capture) {
        manifest.open(output_directory / "manifest.csv");
        require(manifest.good(), "cannot create target K6 capture manifest");
        manifest << "layer,operator,rows,in_features,out_features,K,mul1,mcg,has_bias,"
                    "m1_dispatch,m1_calls,input_bytes,distinct_rows,file,pending,proposal_ids\n";
    }
    for (const auto& item : captured) {
        require(item.layer >= 0 && item.layer < 64 &&
                    (capture_down ? item.operation == "down"
                                  : (item.operation == "gate" ||
                                     item.operation == "up")),
                "target K6 capture observed an invalid layer/operator");
        require(item.metadata.K == expected_k && item.metadata.mul1 && !item.metadata.mcg &&
                    !item.metadata.has_bias &&
                    item.metadata.in_features == input_features &&
                    item.metadata.out_features == output_features &&
                    item.input.size() ==
                        static_cast<std::size_t>(8) * input_features,
                "target K6 capture shape/metadata mismatch");
        require(item.weights.trellis && item.weights.suh && item.weights.svh &&
                    item.weights.mul1,
                "target K6 capture borrowed null weights");
        total_input_bytes += item.input.size() * sizeof(std::uint16_t);
        require(std::all_of(item.input.begin(), item.input.end(), [](std::uint16_t bits) {
                    return std::isfinite(half_to_float(bits));
                }),
                "target K6 capture contains nonfinite represented input");
        require(std::any_of(item.input.begin(), item.input.end(), [](std::uint16_t bits) {
                    return (bits & 0x7fffU) != 0U;
                }),
                "target K6 capture contains only represented zeros");
        const int operation = item.operation == "up" ? 1 : 0;
        require(!seen[static_cast<std::size_t>(item.layer)]
                     [static_cast<std::size_t>(operation)],
                "target K6 capture observed a duplicate layer/operator");
        seen[static_cast<std::size_t>(item.layer)]
            [static_cast<std::size_t>(operation)] = true;
        const int distinct_rows = target_k6_distinct_rows(item);
        require(distinct_rows == 8,
                "target K6 capture requires eight bitwise-distinct rows");
        captured_layers[static_cast<std::size_t>(item.layer)] = true;
        if (write_capture) {
            const std::string filename = "layer" + std::to_string(item.layer) + "-" +
                item.operation + "-" + std::to_string(input_features) + "x" +
                std::to_string(output_features) + ".f16";
            const auto path = output_directory / filename;
            require(!std::filesystem::exists(path),
                    "target K6 capture refuses to overwrite input file");
            std::ofstream output(path, std::ios::binary);
            require(output.good(), "cannot create target K6 captured input");
            output.write(reinterpret_cast<const char*>(item.input.data()),
                         static_cast<std::streamsize>(item.input.size() *
                                                      sizeof(std::uint16_t)));
            require(output.good(), "cannot write target K6 captured input");
            manifest << item.layer << ',' << item.operation << ",8,"
                     << item.metadata.in_features << ',' << item.metadata.out_features << ','
                     << item.metadata.K << ',' << item.metadata.mul1 << ','
                     << item.metadata.mcg << ',' << item.metadata.has_bias << ','
                     << item.m1_dispatch << ",8," << item.input.size() * sizeof(std::uint16_t)
                     << ',' << distinct_rows << ',' << filename << ',' << pending << ','
                     << proposal_ids.str() << '\n';
        }
    }
    for (int layer = 0; layer < 64; ++layer) {
        if (capture_down) {
            const bool expected = capture_down_k7
                ? ((layer >= 22 && layer <= 26) ||
                   (layer >= 35 && layer <= 41) ||
                   (layer >= 51 && layer <= 57))
                : (layer >= 2 && layer <= 21) ||
                (layer >= 27 && layer <= 34) ||
                (layer >= 42 && layer <= 50) ||
                (layer >= 58 && layer <= 63);
            require(captured_layers[static_cast<std::size_t>(layer)] == expected,
                    "target down K6 canonical layer inventory mismatch");
        }
        if (!captured_layers[static_cast<std::size_t>(layer)]) continue;
        require(seen[static_cast<std::size_t>(layer)][0] &&
                    (capture_down || seen[static_cast<std::size_t>(layer)][1]),
                "target K6 capture layer is missing an expected operator");
        if (layer % 4 == 3) ++full_layers;
        else ++gdn_layers;
    }
    require(full_layers == (capture_gateup_k7 ? 1 :
                              (capture_down_k7 ? 5 : (capture_down ? 11 : 12))) &&
                gdn_layers == (capture_gateup_k7 ? 1 :
                               (capture_down_k7 ? 14 : (capture_down ? 32 : 35))),
            "target K6 capture canonical layer inventory mismatch");
    require(total_input_bytes == static_cast<std::size_t>(expected_records) *
                8u * static_cast<std::size_t>(input_features) *
                sizeof(std::uint16_t),
            "target K6 capture total input byte count mismatch");
    if (write_capture) {
        manifest.flush();
        require(manifest.good(), "target K6 capture manifest write failed");
        if (capture_down) {
            std::cout << (capture_down_k7 ? "TARGETDOWNK7CAPTURE" : "TARGETDOWNK6CAPTURE")
                      << " PASS records=" << expected_records << " ctx=512 rows=8"
                      << " pending=" << pending
                      << " proposal_ids=" << proposal_ids.str()
                      << " total_input_bytes=" << total_input_bytes
                      << " full_layers=" << full_layers
                      << " gdn_layers=" << gdn_layers
                      << " files=" << expected_records << " rollback=1 target_restored=1 ring_unchanged=1"
                      << std::endl;
        } else {
            std::cout << "TARGETK6CAPTURE PASS pairs=94 rows=8 full_layers="
                      << full_layers << " gdn_layers=" << gdn_layers
                      << " files=94 target_restored=1 ring_unchanged=1"
                      << std::endl;
        }
    }
    return captured;
}

std::vector<TargetK6CapturedProjection> run_target_k6_capture(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,
    const std::filesystem::path& output_directory, bool oscar) {
    return run_target_k6_capture_selected(
        target, draft, prompt, output_directory, oscar,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::gate_up_k6);
}

std::vector<TargetK6CapturedProjection> run_target_down_k6_capture(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,
    const std::filesystem::path& output_directory, bool oscar) {
    return run_target_k6_capture_selected(
        target, draft, prompt, output_directory, oscar,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::down_k6);
}

std::vector<TargetK6CapturedProjection> run_target_down_k7_capture(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,
    const std::filesystem::path& output_directory, bool oscar) {
    return run_target_k6_capture_selected(
        target, draft, prompt, output_directory, oscar,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::down_k7);
}

std::vector<TargetK6CapturedProjection> run_target_gateup_k7_capture(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt, bool oscar) {
    return run_target_k6_capture_selected(
        target, draft, prompt, std::filesystem::path{}, oscar,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::gate_up_k7);
}
