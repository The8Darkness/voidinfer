#pragma once

template <class Fn>
void require_target_continue_reject(Fn&& fn, const std::string& label) {
    bool rejected = false;
    try { fn(); } catch (const std::exception&) { rejected = true; }
    require(rejected, label + " did not reject");
}

void require_target_continue_exact(const std::vector<std::uint16_t>& actual,
                                   const std::vector<std::uint16_t>& expected,
                                   std::size_t row_width,
                                   const std::string& label) {
    require(actual.size() == expected.size(), label + " size mismatch");
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end()) {
        const std::size_t index = static_cast<std::size_t>(mismatch.first - actual.begin());
        throw std::runtime_error(label + " first mismatch row=" +
            std::to_string(index / row_width) + " index=" +
            std::to_string(index % row_width) + " actual=" +
            std::to_string(*mismatch.first) + " expected=" +
            std::to_string(*(expected.begin() + index)));
    }
}

std::vector<std::uint16_t> target_continue_device_bits(const std::uint16_t* source,
                                                       std::size_t count,
                                                       const std::string& label) {
    std::vector<std::uint16_t> result(count);
    cuda_check(cudaMemcpy(result.data(), source, count * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), label.c_str());
    return result;
}

std::vector<std::uint16_t> target_continue_tap_bits(Exl3TextContext& context,
                                                    int layer, int rows) {
    DeviceBuffer staging(static_cast<std::size_t>(rows) * kHidden * sizeof(std::uint16_t));
    context.copy_tap_rows_to_device(layer, 0,
        static_cast<std::uint16_t*>(staging.get()), rows);
    return target_continue_device_bits(static_cast<const std::uint16_t*>(staging.get()),
        static_cast<std::size_t>(rows) * kHidden, "targetcontinue download tap bits");
}

std::uint64_t target_continue_hash(const std::vector<float>& values) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    const std::size_t count = values.size() * sizeof(float);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < count; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

void require_target_continue_state_equal(Exl3TextContext& actual,
                                         Exl3TextContext& expected,
                                         const std::string& label,
                                         bool compare_taps = true) {
    require(actual.position() == expected.position(), label + " host position mismatch");
    require(actual.device_position_host() == expected.device_position_host(),
            label + " device position mismatch");
    require(target_continue_device_bits(actual.logits_device(), kVocab,
                "targetcontinue download actual logits") ==
            target_continue_device_bits(expected.logits_device(), kVocab,
                "targetcontinue download expected logits"),
            label + " legacy logits mismatch");
    if (compare_taps) {
        for (const int layer : kTapLayers)
            require(target_continue_tap_bits(actual, layer, actual.captured_tap_rows()) ==
                        target_continue_tap_bits(expected, layer, expected.captured_tap_rows()),
                    label + " tap mismatch at layer " + std::to_string(layer));
    }
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        const auto a = actual.gdn_state_host(layer);
        const auto b = expected.gdn_state_host(layer);
        require(a.size() == b.size() &&
                    std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                label + " GDN state mismatch at layer " + std::to_string(layer));
    }
}

void run_target_continuation_qualification(Exl3TextModel& target,
                                           const std::vector<std::int64_t>& source_ids) {
    require(target.max_context() == 1024, "targetcontinue requires max context 1024");
    require(env("NINFER_OSCAR_EXL3") == "1",
            "targetcontinue requires canonical OSCAR");
    require(!source_ids.empty(), "targetcontinue prompt fixture is empty");

    constexpr int kCapacity = 8;
    // B8: normalized rows 81,920 + logits 3,973,120 + H6 workspace
    // (transformed 81,920 + five FP32 accumulation splits 39,731,200).
    constexpr std::size_t kExpectedBytes = 43868160ULL;
    const std::array<int, 4> prefixes{63, 319, 512, 575};
    const std::array<int, 3> row_counts{2, 4, 8};
    std::vector<std::int64_t> ids(1032);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = source_ids[i % source_ids.size()];

    int cases = 0;
    for (const int prefix_rows : prefixes) {
        auto serial = target.create_context(true);
        auto block = target.create_context(true);
        require(serial->try_enable_oscar_from_environment(),
                "targetcontinue serial OSCAR enable failed");
        require(block->try_enable_oscar_from_environment(),
                "targetcontinue block OSCAR enable failed");
        serial->prepare_transaction();
        block->prepare_transaction();
        const std::size_t persistent_before = block->persistent_bytes();
        block->prepare_continuation(kCapacity);
        require(block->continuation_capacity() == kCapacity &&
                    block->continuation_bytes() == kExpectedBytes &&
                    block->persistent_bytes() - persistent_before == kExpectedBytes,
                "targetcontinue allocation accounting mismatch");
        const std::string head = block->continuation_head_dispatch();
        const bool h6_opt_out = env("NINFER_EXL3_H6_SMALL_M") == "0";
        require(head == (h6_opt_out ? "h6_single_split_per_row"
                                   : "h6_small_m_single_split"),
                "targetcontinue H6 route mismatch");

        const std::vector<std::int64_t> prefix(ids.begin(), ids.begin() + prefix_rows);
        ingest_prefix(*serial, prefix, CommitSink{});
        ingest_prefix(*block, prefix, CommitSink{});
        require_target_continue_state_equal(*block, *serial,
                                            "targetcontinue paired prefix");

        for (const int rows : row_counts) {
            serial->begin_transaction();
            block->begin_transaction();
            if (prefix_rows == prefixes.front() && rows == row_counts.front()) {
                require_target_continue_reject([&] { serial->prepare_continuation(kCapacity); },
                    "targetcontinue allocation during active transaction");
            }
            std::vector<std::uint16_t> serial_logits;
            std::array<std::vector<std::uint16_t>, 5> serial_taps;
            for (int row = 0; row < rows; ++row) {
                serial->decode(ids[static_cast<std::size_t>(prefix_rows + row)]);
                const auto logits = target_continue_device_bits(serial->logits_device(), kVocab,
                    "targetcontinue download serial logits");
                serial_logits.insert(serial_logits.end(), logits.begin(), logits.end());
                for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
                    const auto values = target_continue_tap_bits(*serial, kTapLayers[tap], 1);
                    serial_taps[tap].insert(serial_taps[tap].end(), values.begin(), values.end());
                }
            }
            const std::span<const std::int64_t> teacher(
                ids.data() + prefix_rows, static_cast<std::size_t>(rows));
            std::uint64_t oscar_before = 0, ordinary_before = 0;
            block->oscar_routing_counts(oscar_before, ordinary_before);
            block->continue_rows(teacher);
            cuda_check(cudaDeviceSynchronize(), "targetcontinue block sync");
            std::uint64_t oscar_after = 0, ordinary_after = 0;
            block->oscar_routing_counts(oscar_after, ordinary_after);
            require(oscar_after - oscar_before == static_cast<std::uint64_t>(16 * rows) &&
                        ordinary_after == ordinary_before,
                    "targetcontinue OSCAR chronology/count mismatch");
            require(block->continuation_rows() == rows &&
                        block->continuation_logits_device() != nullptr,
                    "targetcontinue output row metadata mismatch");
            const auto all_logits = block->continuation_logits_bits_host();
            require_target_continue_exact(all_logits, serial_logits, kVocab,
                    "targetcontinue all-row full-vocabulary logits");
            for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap)
                require_target_continue_exact(target_continue_tap_bits(
                    *block, kTapLayers[tap], rows),
                    serial_taps[tap], kHidden,
                    "targetcontinue all-row tap layer=" +
                        std::to_string(kTapLayers[tap]));
            require(target_continue_device_bits(block->logits_device(), kVocab,
                        "targetcontinue download legacy final logits") ==
                    std::vector<std::uint16_t>(all_logits.end() - kVocab, all_logits.end()),
                    "targetcontinue legacy final-row logits mismatch");
            require_target_continue_state_equal(*block, *serial,
                "targetcontinue completed block prefix=" + std::to_string(prefix_rows) +
                " rows=" + std::to_string(rows), false);

            require(block->continuation_rows() == rows,
                    "targetcontinue commit-visible scratch missing");
            const bool commit_case = rows == row_counts.back();
            if (commit_case) {
                serial->commit_transaction();
                block->commit_transaction();
                require(block->continuation_rows() == rows,
                        "targetcontinue commit unexpectedly invalidated derived output");
            }
            for (int step = 0; step < 4; ++step) {
                const auto token = ids[static_cast<std::size_t>(prefix_rows + rows + step)];
                serial->decode(token);
                block->decode(token);
                if (step == 0)
                    require(block->continuation_rows() == 0 &&
                                block->continuation_logits_device() == nullptr,
                            "targetcontinue ordinary decode did not invalidate scratch");
                require_target_continue_state_equal(*block, *serial,
                    "targetcontinue serial continuation step " + std::to_string(step));
            }

            if (!commit_case) {
                serial->rollback_transaction();
                block->rollback_transaction();
                cuda_check(cudaDeviceSynchronize(), "targetcontinue rollback sync");
                require(block->continuation_rows() == 0,
                        "targetcontinue rollback did not invalidate scratch");
                require_target_continue_state_equal(*block, *serial,
                                                    "targetcontinue rollback baseline");
            }
            ++cases;
            std::cout << "TARGETCONTINUE_CASE PASS prefix=" << prefix_rows
                      << " rows=" << rows << " head=" << head << std::endl;
        }
    }

    // Rejections below occur before target state mutation. The paired serial
    // context proves logits, taps, position and recurrent state remain intact.
    const std::vector<std::int64_t> prefix(ids.begin(), ids.begin() + 63);
    {
    auto invalid = target.create_context(true);
    auto invalid_reference = target.create_context(true);
    require(invalid->try_enable_oscar_from_environment() &&
                invalid_reference->try_enable_oscar_from_environment(),
            "targetcontinue invalid-case OSCAR enable failed");
    invalid->prepare_continuation(kCapacity);
    require_target_continue_reject([&] {
        invalid->continue_rows(std::span<const std::int64_t>(ids.data(), 2));
    }, "targetcontinue empty prefix");
    require_target_continue_reject([&] {
        invalid->continue_rows(std::span<const std::int64_t>{});
    },
                                   "targetcontinue empty-prefix/empty-rows");
    ingest_prefix(*invalid, prefix, CommitSink{});
    ingest_prefix(*invalid_reference, prefix, CommitSink{});
    require_target_continue_reject([&] {
        invalid->continue_rows(std::span<const std::int64_t>(ids.data() + 63, 1));
    }, "targetcontinue M1");
    require_target_continue_reject([&] {
        invalid->continue_rows(std::span<const std::int64_t>(ids.data() + 63, 9));
    }, "targetcontinue over-capacity");
    const std::array<std::int64_t, 2> negative_token{-1, ids[64]};
    require_target_continue_reject([&] {
        invalid->continue_rows(std::span<const std::int64_t>(negative_token));
    },
                                   "targetcontinue negative token");
    const std::array<std::int64_t, 2> upper_token{kVocab, ids[64]};
    require_target_continue_reject([&] {
        invalid->continue_rows(std::span<const std::int64_t>(upper_token));
    },
                                   "targetcontinue token at vocabulary bound");
    require_target_continue_state_equal(*invalid, *invalid_reference,
                                        "targetcontinue invalid requests");
    cudaStream_t external_stream = nullptr;
    cudaGraph_t external_graph = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&external_stream, cudaStreamNonBlocking),
               "targetcontinue create external capture stream");
    cuda_check(cudaStreamBeginCapture(external_stream, cudaStreamCaptureModeRelaxed),
               "targetcontinue begin external stream capture");
    bool external_rejected = false;
    try {
        invalid->continue_rows(std::span<const std::int64_t>(ids.data() + 63, 2),
                               external_stream);
    } catch (const std::exception&) {
        external_rejected = true;
    }
    const cudaError_t capture_end = cudaStreamEndCapture(external_stream, &external_graph);
    const cudaError_t external_graph_destroy = external_graph != nullptr
        ? cudaGraphDestroy(external_graph) : cudaSuccess;
    const cudaError_t external_stream_destroy = cudaStreamDestroy(external_stream);
    cuda_check(capture_end, "targetcontinue end external stream capture");
    cuda_check(external_graph_destroy, "targetcontinue destroy external capture graph");
    cuda_check(external_stream_destroy, "targetcontinue destroy external capture stream");
    require(external_rejected, "targetcontinue external stream capture did not reject");
    require_target_continue_state_equal(*invalid, *invalid_reference,
                                        "targetcontinue external capture rejection");
    cudaStream_t graph_stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking),
               "targetcontinue create internal graph stream");
    invalid->oscar_set_graph_class(16);
    invalid->oscar_sync_device_state(graph_stream);
    const bool graph_captured = invalid->capture_decode_graph(graph_stream);
    cuda_check(cudaStreamSynchronize(graph_stream),
               "targetcontinue synchronize internal graph stream");
    cuda_check(cudaStreamDestroy(graph_stream),
               "targetcontinue destroy internal graph stream");
    require(graph_captured, "targetcontinue graph capture failed");
    require_target_continue_reject([&] {
        invalid->continue_rows(std::span<const std::int64_t>(ids.data() + 63, 2));
    }, "targetcontinue graph mode");
    }

    for (const int capacity : std::array<int, 4>{-1, 0, 1, 9}) {
        auto context = target.create_context(true);
        require_target_continue_reject([&] { context->prepare_continuation(capacity); },
                                       "targetcontinue invalid capacity");
    }
    {
        auto no_oscar = target.create_context(true);
        require_target_continue_reject([&] { no_oscar->prepare_continuation(kCapacity); },
                                       "targetcontinue setup without OSCAR");
    }
    {
        auto no_taps = target.create_context(false);
        require(no_taps->try_enable_oscar_from_environment(),
                "targetcontinue no-taps OSCAR enable failed");
        require_target_continue_reject([&] { no_taps->prepare_continuation(kCapacity); },
                                       "targetcontinue setup without tap capture");
    }

    {
        auto rollback_case = target.create_context(true);
        auto rollback_reference = target.create_context(true);
        require(rollback_case->try_enable_oscar_from_environment() &&
                    rollback_reference->try_enable_oscar_from_environment(),
                "targetcontinue live rollback OSCAR enable failed");
        rollback_case->prepare_transaction();
        rollback_case->prepare_continuation(kCapacity);
        ingest_prefix(*rollback_case, prefix, CommitSink{});
        ingest_prefix(*rollback_reference, prefix, CommitSink{});
        require_target_continue_reject([&] {
            rollback_reference->continue_rows(
                std::span<const std::int64_t>(ids.data() + 63, 2));
        }, "targetcontinue unprepared context");
        require_target_continue_state_equal(*rollback_case, *rollback_reference,
                                            "targetcontinue unprepared rejection");
        rollback_case->begin_transaction();
        rollback_case->continue_rows(
            std::span<const std::int64_t>(ids.data() + 63, 2));
        require(rollback_case->continuation_rows() == 2,
                "targetcontinue live rollback setup did not publish scratch");
        rollback_case->rollback_transaction();
        cuda_check(cudaDeviceSynchronize(), "targetcontinue live rollback sync");
        require(rollback_case->continuation_rows() == 0 &&
                    rollback_case->continuation_logits_device() == nullptr,
                "targetcontinue live rollback did not invalidate scratch");
        require_target_continue_reject([&] {
            (void)rollback_case->continuation_logits_bits_host();
        }, "targetcontinue host getter after rollback");
        require_target_continue_state_equal(*rollback_case, *rollback_reference,
                                            "targetcontinue live rollback baseline");
    }

    auto reset_case = target.create_context(true);
    require(reset_case->try_enable_oscar_from_environment(),
            "targetcontinue reset OSCAR enable failed");
    reset_case->prepare_continuation(kCapacity);
    ingest_prefix(*reset_case, prefix, CommitSink{});
    while (reset_case->position() < 1023) {
        const int rows = std::min(kCapacity, 1023 - reset_case->position());
        reset_case->continue_rows(std::span<const std::int64_t>(
            ids.data() + reset_case->position(), static_cast<std::size_t>(rows)));
    }
    require(reset_case->position() == 1023 && reset_case->continuation_rows() > 0,
            "targetcontinue near-capacity setup mismatch");
    const int capacity_position = reset_case->position();
    const int capacity_device_position = reset_case->device_position_host();
    const int capacity_rows = reset_case->continuation_rows();
    const auto capacity_logits = target_continue_device_bits(
        reset_case->logits_device(), kVocab, "targetcontinue near-capacity logits");
    std::array<std::vector<std::uint16_t>, 5> capacity_taps;
    for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap)
        capacity_taps[tap] = target_continue_tap_bits(
            *reset_case, kTapLayers[tap], capacity_rows);
    std::array<std::uint64_t, 64> capacity_gdn{};
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        capacity_gdn[static_cast<std::size_t>(layer)] =
            target_continue_hash(reset_case->gdn_state_host(layer));
    }
    require_target_continue_reject([&] {
        reset_case->continue_rows(std::span<const std::int64_t>(ids.data() + 1023, 2));
    }, "targetcontinue context capacity");
    require(reset_case->position() == capacity_position &&
                reset_case->device_position_host() == capacity_device_position &&
                reset_case->continuation_rows() == capacity_rows &&
                target_continue_device_bits(reset_case->logits_device(), kVocab,
                    "targetcontinue near-capacity logits after rejection") == capacity_logits,
            "targetcontinue near-capacity rejection mutated metadata/logits");
    for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap)
        require(target_continue_tap_bits(*reset_case, kTapLayers[tap], capacity_rows) ==
                    capacity_taps[tap],
                "targetcontinue near-capacity rejection mutated tap");
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        require(target_continue_hash(reset_case->gdn_state_host(layer)) ==
                    capacity_gdn[static_cast<std::size_t>(layer)],
                "targetcontinue near-capacity rejection mutated GDN state");
    }
    reset_case->reset();
    require(reset_case->continuation_rows() == 0 &&
                reset_case->continuation_logits_device() == nullptr,
            "targetcontinue reset did not invalidate scratch");

    // Raw physical-convolution and OSCAR-cache bytes are intentionally not
    // exposed here. Exact recurrent state plus four untouched serial steps
    // verify their observable continuation semantics.
    std::cout << "TARGETCONTINUE PASS cases=" << cases
              << " capacity=" << kCapacity
              << " bytes=" << kExpectedBytes << std::endl;
}
