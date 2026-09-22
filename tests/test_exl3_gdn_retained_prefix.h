#pragma once

void run_gdn_retained_prefix_qualification(
    const Exl3GdnLayerWeights& weights, const SafetensorsHeader& fixture,
    const std::filesystem::path& fixture_path) {
    constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
    const auto real_values = as_float(read_tensor(
        fixture_path, fixture, "prefill11_layer_output_input"));
    require(real_values.size() == static_cast<std::size_t>(PREFILL) * HIDDEN,
            "GDNPREFIX real input fixture shape mismatch");
    std::vector<std::uint16_t> real_bits(real_values.size());
    for (std::size_t i = 0; i < real_values.size(); ++i) {
        if (!std::isfinite(real_values[i]))
            throw std::runtime_error("GDNPREFIX real input fixture is nonfinite");
        real_bits[i] = half_bits(real_values[i]);
        if ((real_bits[i] & 0x7c00u) == 0x7c00u)
            throw std::runtime_error(
                "GDNPREFIX represented real input is nonfinite");
    }
    for (int lhs = 0; lhs < 8; ++lhs) {
        for (int rhs = lhs + 1; rhs < 8; ++rhs) {
            require(std::memcmp(real_bits.data() + static_cast<std::size_t>(lhs) * HIDDEN,
                                real_bits.data() + static_cast<std::size_t>(rhs) * HIDDEN,
                                static_cast<std::size_t>(HIDDEN) * kHalfBytes) != 0,
                    "GDNPREFIX fixture rows are not bitwise distinct");
        }
    }
    const auto continuation_values = as_float(read_tensor(
        fixture_path, fixture, "decode0_layer_output_input"));
    require(continuation_values.size() == HIDDEN,
            "GDNPREFIX continuation fixture shape mismatch");
    std::vector<std::uint16_t> continuation_bits(HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) {
        if (!std::isfinite(continuation_values[static_cast<std::size_t>(i)]))
            throw std::runtime_error(
                "GDNPREFIX continuation fixture is nonfinite");
        continuation_bits[static_cast<std::size_t>(i)] =
            half_bits(continuation_values[static_cast<std::size_t>(i)]);
        if ((continuation_bits[static_cast<std::size_t>(i)] & 0x7c00u) ==
            0x7c00u)
            throw std::runtime_error(
                "GDNPREFIX represented continuation is nonfinite");
    }

    Exl3GdnLayer candidate(weights, PREFILL);
    Exl3GdnLayer serial(weights, PREFILL);
    auto recurrent_model_owner=std::make_shared<int>(5);
    candidate.bind_recurrent_layout(recurrent_model_owner,5);
    serial.bind_recurrent_layout(recurrent_model_owner,5);
    DeviceBuffer candidate_input, serial_input, candidate_output, serial_output;
    cuda_check(cudaMalloc(&candidate_input.ptr,
                          static_cast<std::size_t>(PREFILL) * HIDDEN * kHalfBytes),
               "allocate GDNPREFIX candidate input");
    cuda_check(cudaMalloc(&serial_input.ptr,
                          static_cast<std::size_t>(PREFILL) * HIDDEN * kHalfBytes),
               "allocate GDNPREFIX serial input");
    cuda_check(cudaMalloc(&candidate_output.ptr,
                          static_cast<std::size_t>(PREFILL) * HIDDEN * kHalfBytes),
               "allocate GDNPREFIX candidate output");
    cuda_check(cudaMalloc(&serial_output.ptr,
                          static_cast<std::size_t>(PREFILL) * HIDDEN * kHalfBytes),
               "allocate GDNPREFIX serial output");
    DeviceBuffer candidate_base_recurrent, candidate_base_conv;
    DeviceBuffer serial_base_recurrent, serial_base_conv;
    DeviceBuffer replacement_recurrent, replacement_conv;
    cuda_check(cudaMalloc(&candidate_base_recurrent.ptr,
                          candidate.recurrent_state_bytes()),
               "allocate GDNPREFIX candidate recurrent checkpoint");
    cuda_check(cudaMalloc(&candidate_base_conv.ptr,
                          candidate.physical_conv_state_bytes()),
               "allocate GDNPREFIX candidate convolution checkpoint");
    cuda_check(cudaMalloc(&serial_base_recurrent.ptr,
                          serial.recurrent_state_bytes()),
               "allocate GDNPREFIX serial recurrent checkpoint");
    cuda_check(cudaMalloc(&serial_base_conv.ptr,
                          serial.physical_conv_state_bytes()),
               "allocate GDNPREFIX serial convolution checkpoint");
    cuda_check(cudaMalloc(&replacement_recurrent.ptr,
                          candidate.recurrent_state_bytes()),
               "allocate GDNPREFIX replacement recurrent checkpoint");
    cuda_check(cudaMalloc(&replacement_conv.ptr,
                          candidate.physical_conv_state_bytes()),
               "allocate GDNPREFIX replacement convolution checkpoint");
    Exl3GdnLayerCheckpoint candidate_base{
        candidate_base_recurrent.ptr, candidate.recurrent_state_bytes(),
        candidate_base_conv.ptr, candidate.physical_conv_state_bytes()};
    Exl3GdnLayerCheckpoint serial_base{
        serial_base_recurrent.ptr, serial.recurrent_state_bytes(),
        serial_base_conv.ptr, serial.physical_conv_state_bytes()};
    Exl3GdnLayerCheckpoint replacement{
        replacement_recurrent.ptr, candidate.recurrent_state_bytes(),
        replacement_conv.ptr, candidate.physical_conv_state_bytes()};
    const auto bind_checkpoint=[&](Exl3GdnLayerCheckpoint& checkpoint) {
        checkpoint.model_owner=recurrent_model_owner;
        checkpoint.model_layer=5;
        checkpoint.recurrent_layer_stride_bytes=
            Exl3GdnRecurrentLayout::recurrent_bytes;
        checkpoint.convolution_layer_stride_bytes=
            Exl3GdnRecurrentLayout::convolution_storage_bytes;
    };
    bind_checkpoint(candidate_base);
    bind_checkpoint(serial_base);
    bind_checkpoint(replacement);

    const auto recurrent = [](const Exl3GdnLayer& layer) {
        return download_bytes(layer.recurrent_state_device(),
                              layer.recurrent_state_bytes(),
                              "download GDNPREFIX recurrent state");
    };
    const auto physical_conv = [](const Exl3GdnLayer& layer) {
        return download_bytes(layer.physical_conv_state_device(),
                              layer.physical_conv_state_bytes(),
                              "download GDNPREFIX physical convolution state");
    };
    const auto output = [](const DeviceBuffer& buffer) {
        return download_bytes(buffer.ptr,
                              static_cast<std::size_t>(HIDDEN) * sizeof(std::uint16_t),
                              "download GDNPREFIX continuation output");
    };
    const auto require_finite_f32 = [&](const std::vector<std::byte>& bytes,
                                        const std::string& label) {
        require(bytes.size() % sizeof(float) == 0,
                label + " FP32 byte count mismatch");
        for (std::size_t offset = 0; offset < bytes.size();
             offset += sizeof(float)) {
            float value = 0.0f;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            if (!std::isfinite(value))
                throw std::runtime_error(label + " contains nonfinite FP32");
        }
    };
    const auto require_finite_u16 = [&](const std::vector<std::byte>& bytes,
                                        std::uint16_t exponent_mask,
                                        const std::string& label) {
        require(bytes.size() % sizeof(std::uint16_t) == 0,
                label + " 16-bit byte count mismatch");
        for (std::size_t offset = 0; offset < bytes.size();
             offset += sizeof(std::uint16_t)) {
            std::uint16_t bits = 0;
            std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
            if ((bits & exponent_mask) == exponent_mask)
                throw std::runtime_error(
                    label + " contains nonfinite represented value");
        }
    };
    const auto require_finite_state = [&](const Exl3GdnLayer& layer,
                                          const std::string& label) {
        require_finite_f32(recurrent(layer), label + " recurrent state");
        require_finite_u16(physical_conv(layer), 0x7f80u,
                           label + " physical convolution state");
    };
    const auto require_finite_output = [&](const DeviceBuffer& buffer,
                                           const std::string& label) {
        require_finite_u16(output(buffer), 0x7c00u, label + " output");
    };
    const auto upload_rows = [&](DeviceBuffer& buffer, int rows) {
        cuda_check(cudaMemcpy(buffer.ptr, real_bits.data(),
                              static_cast<std::size_t>(rows) * HIDDEN * kHalfBytes,
                              cudaMemcpyHostToDevice),
                   "upload GDNPREFIX real rows");
    };
    const auto upload_continuation = [&](DeviceBuffer& buffer) {
        cuda_check(cudaMemcpy(buffer.ptr, continuation_bits.data(),
                              static_cast<std::size_t>(HIDDEN) * kHalfBytes,
                              cudaMemcpyHostToDevice),
                   "upload GDNPREFIX continuation row");
    };
    const auto require_state_equal = [&](const std::string& label) {
        require(recurrent(candidate) == recurrent(serial),
                label + " recurrent state mismatch");
        require(physical_conv(candidate) == physical_conv(serial),
                label + " four-slot convolution state mismatch");
    };
    const auto require_rejected_without_mutation =
        [&](auto&& operation, const std::string& reason,
            const std::string& label) {
            cuda_check(cudaDeviceSynchronize(),
                       "synchronize GDNPREFIX rejection baseline");
            const auto before_recurrent = recurrent(candidate);
            const auto before_conv = physical_conv(candidate);
            bool rejected = false;
            try { operation(); }
            catch (const std::invalid_argument& error) {
                rejected = std::string(error.what()).find(reason) !=
                    std::string::npos;
            }
            cuda_check(cudaDeviceSynchronize(),
                       "synchronize GDNPREFIX rejected operation");
            require(rejected && recurrent(candidate) == before_recurrent &&
                        physical_conv(candidate) == before_conv,
                    label + " did not reject before mutation");
        };

    candidate.reset();
    serial.reset();
    upload_rows(candidate_input, PREFILL);
    upload_rows(serial_input, PREFILL);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr), PREFILL);
    serial.forward(static_cast<const std::uint16_t*>(serial_input.ptr),
                   static_cast<std::uint16_t*>(serial_output.ptr), PREFILL);
    candidate.save_checkpoint(candidate_base,PREFILL);
    serial.save_checkpoint(serial_base,PREFILL);
    cuda_check(cudaDeviceSynchronize(), "synchronize GDNPREFIX nonzero base");
    const auto base_recurrent = recurrent(candidate);
    const auto base_conv = physical_conv(candidate);
    require(any_nonzero(base_recurrent) && any_nonzero(base_conv),
            "GDNPREFIX eleven-row base state is zero");
    require_state_equal("GDNPREFIX base");
    require_finite_state(candidate, "GDNPREFIX base");

    int cases = 0;
    int guard_checks = 0;
    for (const int block_rows : {2, 4, 8}) {
        for (int retained = 1; retained <= block_rows; ++retained) {
            candidate.restore_checkpoint(candidate_base);
            candidate.save_checkpoint(candidate_base,PREFILL);
            serial.restore_checkpoint(serial_base);
            upload_rows(candidate_input, block_rows);
            upload_rows(serial_input, block_rows);
            candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                              static_cast<std::uint16_t*>(candidate_output.ptr),
                              block_rows, nullptr, false, true);

            if (block_rows == 2 && retained == 1) {
                require_rejected_without_mutation(
                    [&] { candidate.reconstruct_retained_prefix(candidate_base, 0); },
                    "row count",
                    "GDNPREFIX zero-row guard");
                ++guard_checks;
                require_rejected_without_mutation(
                    [&] { candidate.reconstruct_retained_prefix(
                        candidate_base, block_rows + 1); },
                    "row count",
                    "GDNPREFIX oversized-row guard");
                ++guard_checks;
                require_rejected_without_mutation(
                    [&] { candidate.reconstruct_retained_prefix(serial_base, retained); },
                    "provenance",
                    "GDNPREFIX foreign-checkpoint guard");
                ++guard_checks;
                Exl3GdnLayerCheckpoint wrong_buffer = candidate_base;
                wrong_buffer.recurrent_state_device = serial_base.recurrent_state_device;
                require_rejected_without_mutation(
                    [&] { candidate.reconstruct_retained_prefix(
                        wrong_buffer, retained); },
                    "buffer mismatch",
                    "GDNPREFIX wrong-buffer guard");
                ++guard_checks;
                cudaStream_t wrong_stream = nullptr;
                cuda_check(cudaStreamCreateWithFlags(&wrong_stream, cudaStreamNonBlocking),
                           "create GDNPREFIX wrong stream");
                try {
                    require_rejected_without_mutation(
                        [&] { candidate.reconstruct_retained_prefix(
                            candidate_base, retained, wrong_stream); },
                        "stream mismatch",
                        "GDNPREFIX wrong-stream guard");
                } catch (...) {
                    cudaStreamDestroy(wrong_stream);
                    throw;
                }
                cuda_check(cudaStreamDestroy(wrong_stream),
                           "destroy GDNPREFIX wrong stream");
                ++guard_checks;
            }

            candidate.reconstruct_retained_prefix(candidate_base, retained);
            for (int row = 0; row < retained; ++row) {
                serial.forward(
                    static_cast<const std::uint16_t*>(serial_input.ptr) +
                        static_cast<std::size_t>(row) * HIDDEN,
                    static_cast<std::uint16_t*>(serial_output.ptr), 1);
            }
            cuda_check(cudaDeviceSynchronize(),
                       "synchronize GDNPREFIX reconstructed state");
            require_state_equal("GDNPREFIX B" + std::to_string(block_rows) +
                                " retained" + std::to_string(retained));
            require_finite_state(candidate,
                                 "GDNPREFIX reconstructed B" +
                                     std::to_string(block_rows) + " retained" +
                                     std::to_string(retained));
            const auto reconstructed_recurrent = recurrent(candidate);
            const auto reconstructed_conv = physical_conv(candidate);

            require_rejected_without_mutation(
                [&] { candidate.reconstruct_retained_prefix(candidate_base, retained); },
                "absent",
                "GDNPREFIX consumed-capability guard");
            ++guard_checks;

            upload_continuation(candidate_input);
            upload_continuation(serial_input);
            candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                              static_cast<std::uint16_t*>(candidate_output.ptr), 1);
            serial.forward(static_cast<const std::uint16_t*>(serial_input.ptr),
                           static_cast<std::uint16_t*>(serial_output.ptr), 1);
            cuda_check(cudaDeviceSynchronize(),
                       "synchronize GDNPREFIX continuation");
            require(output(candidate_output) == output(serial_output),
                    "GDNPREFIX subsequent M1 output mismatch");
            require_state_equal("GDNPREFIX subsequent M1");
            require_finite_output(candidate_output,
                                  "GDNPREFIX subsequent M1");
            require_finite_state(candidate, "GDNPREFIX subsequent M1");

            candidate.restore_checkpoint(candidate_base);
            cuda_check(cudaDeviceSynchronize(),
                       "synchronize GDNPREFIX reusable base rollback");
            require(recurrent(candidate) == base_recurrent &&
                        physical_conv(candidate) == base_conv,
                    "GDNPREFIX reusable base rollback mismatch");

            if (block_rows == 8 && retained == 4) {
                candidate.save_checkpoint(candidate_base,PREFILL);
                upload_rows(candidate_input, block_rows);
                candidate.forward(
                    static_cast<const std::uint16_t*>(candidate_input.ptr),
                    static_cast<std::uint16_t*>(candidate_output.ptr),
                    block_rows, nullptr, false, true);
                candidate.reconstruct_retained_prefix(candidate_base, retained);
                cuda_check(cudaDeviceSynchronize(),
                           "synchronize GDNPREFIX repeated reconstruction");
                require(recurrent(candidate) == reconstructed_recurrent &&
                            physical_conv(candidate) == reconstructed_conv,
                        "GDNPREFIX repeated reconstruction mismatch");
                candidate.restore_checkpoint(candidate_base);
            }
            ++cases;
            std::cout << "GDNPREFIX_CASE PASS B=" << block_rows
                      << " retained=" << retained
                      << " recurrent=exact conv_slots=4 continuation=exact"
                      << std::endl;
        }
    }

    // Capability generation guards: absence, restore, reset and an intervening
    // forward must all reject without changing the state they observe.
    require_rejected_without_mutation(
        [&] { candidate.reconstruct_retained_prefix(candidate_base, 1); },
        "absent",
        "GDNPREFIX absent-capability guard");
    ++guard_checks;
    candidate.restore_checkpoint(candidate_base);
    candidate.save_checkpoint(candidate_base,PREFILL);
    upload_rows(candidate_input, 2);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr),
                      2, nullptr, false, true);
    candidate.restore_checkpoint(candidate_base);
    require_rejected_without_mutation(
        [&] { candidate.reconstruct_retained_prefix(candidate_base, 1); },
        "absent",
        "GDNPREFIX restore-invalidated guard");
    ++guard_checks;
    candidate.save_checkpoint(candidate_base,PREFILL);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr),
                      2, nullptr, false, true);
    upload_continuation(candidate_input);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr), 1);
    require_rejected_without_mutation(
        [&] { candidate.reconstruct_retained_prefix(candidate_base, 1); },
        "absent",
        "GDNPREFIX intervening-forward guard");
    ++guard_checks;

    // A preserve forward captured rather than executed must not publish an
    // eager reconstruction capability after capture ends.
    cudaStream_t capture_stream = nullptr;
    cudaGraph_t captured_graph = nullptr;
    cudaGraph_t captured_save_graph = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking),
               "create GDNPREFIX capture stream");
    try {
        candidate.restore_checkpoint(candidate_base, capture_stream);
        cuda_check(cudaStreamSynchronize(capture_stream),
                   "synchronize GDNPREFIX capture base restore");
        candidate.save_checkpoint(candidate_base,PREFILL,capture_stream);
        cuda_check(cudaStreamSynchronize(capture_stream),
                   "synchronize GDNPREFIX capture-stream checkpoint");
        upload_rows(candidate_input, 2);
        cuda_check(cudaStreamBeginCapture(
                       capture_stream, cudaStreamCaptureModeThreadLocal),
                   "begin GDNPREFIX source capture");
        candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                          static_cast<std::uint16_t*>(candidate_output.ptr),
                          2, capture_stream, false, true);
        cuda_check(cudaStreamEndCapture(capture_stream, &captured_graph),
                   "end GDNPREFIX source capture");
        require(captured_graph != nullptr,
                "GDNPREFIX source capture produced no graph");
        require_rejected_without_mutation(
            [&] { candidate.reconstruct_retained_prefix(
                candidate_base, 1, capture_stream); },
            "absent", "GDNPREFIX captured-source guard");
        ++guard_checks;

        // A save recorded into a graph may never authorize a later eager
        // forward, regardless of whether its descriptor previously named a
        // valid eager checkpoint.
        candidate.restore_checkpoint(candidate_base, capture_stream);
        cuda_check(cudaStreamSynchronize(capture_stream),
                   "synchronize GDNPREFIX captured-save base restore");
        candidate.save_checkpoint(replacement,PREFILL,capture_stream);
        cuda_check(cudaStreamSynchronize(capture_stream),
                   "synchronize GDNPREFIX captured-save backup");
        cuda_check(cudaStreamBeginCapture(
                       capture_stream, cudaStreamCaptureModeThreadLocal),
                   "begin GDNPREFIX checkpoint save capture");
        candidate.save_checkpoint(candidate_base,PREFILL,capture_stream);
        cuda_check(cudaStreamEndCapture(capture_stream, &captured_save_graph),
                   "end GDNPREFIX checkpoint save capture");
        require(captured_save_graph != nullptr,
                "GDNPREFIX checkpoint save capture produced no graph");
        // Do not restore here: restore independently invalidates eligibility
        // and would mask a captured save incorrectly authorizing the forward.
        candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                          static_cast<std::uint16_t*>(candidate_output.ptr),
                          2, capture_stream, false, true);
        require_rejected_without_mutation(
            [&] { candidate.reconstruct_retained_prefix(
                candidate_base, 1, capture_stream); },
            "absent", "GDNPREFIX captured-save guard");
        ++guard_checks;
        cuda_check(cudaGraphDestroy(captured_save_graph),
                   "destroy GDNPREFIX checkpoint save graph");
        captured_save_graph = nullptr;

        candidate.restore_checkpoint(replacement,capture_stream);
        candidate.save_checkpoint(candidate_base,PREFILL,capture_stream);
        candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                          static_cast<std::uint16_t*>(candidate_output.ptr),
                          2, capture_stream, false, true);
        cuda_check(cudaStreamSynchronize(capture_stream),
                   "synchronize GDNPREFIX eager capture-stream source");
        const auto active_capture_recurrent = recurrent(candidate);
        const auto active_capture_conv = physical_conv(candidate);
        cuda_check(cudaStreamBeginCapture(
                       capture_stream, cudaStreamCaptureModeThreadLocal),
                   "begin GDNPREFIX active reconstruction capture");
        bool active_capture_rejected = false;
        try {
            candidate.reconstruct_retained_prefix(
                candidate_base, 1, capture_stream);
        } catch (const std::invalid_argument& error) {
            active_capture_rejected = std::string(error.what()).find(
                "requires an eager stream") != std::string::npos;
        }
        cuda_check(cudaMemsetAsync(candidate_output.ptr, 0,
                                   sizeof(std::uint16_t), capture_stream),
                   "record GDNPREFIX active-capture sentinel node");
        cudaGraph_t active_capture_graph = nullptr;
        cuda_check(cudaStreamEndCapture(capture_stream, &active_capture_graph),
                   "end GDNPREFIX active reconstruction capture");
        require(active_capture_graph != nullptr,
                "GDNPREFIX active reconstruction capture produced no graph");
        cuda_check(cudaGraphDestroy(active_capture_graph),
                   "destroy GDNPREFIX active reconstruction graph");
        cuda_check(cudaStreamSynchronize(capture_stream),
                   "synchronize GDNPREFIX active-capture guard");
        require(active_capture_rejected &&
                    recurrent(candidate) == active_capture_recurrent &&
                    physical_conv(candidate) == active_capture_conv,
                "GDNPREFIX active capture did not reject before mutation");
        candidate.reconstruct_retained_prefix(
            candidate_base, 1, capture_stream);
        cuda_check(cudaStreamSynchronize(capture_stream),
                   "synchronize GDNPREFIX post-capture reconstruction");
        ++guard_checks;
    } catch (...) {
        if (captured_graph) cudaGraphDestroy(captured_graph);
        if (captured_save_graph) cudaGraphDestroy(captured_save_graph);
        cudaStreamDestroy(capture_stream);
        throw;
    }
    cuda_check(cudaGraphDestroy(captured_graph),
               "destroy GDNPREFIX source graph");
    cuda_check(cudaStreamDestroy(capture_stream),
               "destroy GDNPREFIX capture stream");
    candidate.restore_checkpoint(candidate_base);
    candidate.save_checkpoint(candidate_base,PREFILL);
    upload_rows(candidate_input, 2);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr),
                      2, nullptr, false, true);
    candidate.save_checkpoint(replacement,PREFILL+2);
    require_rejected_without_mutation(
        [&] { candidate.reconstruct_retained_prefix(candidate_base, 1); },
        "absent", "GDNPREFIX replacement-checkpoint guard");
    ++guard_checks;

    // Copy an authorized descriptor, overwrite the same backing ranges with a
    // newer save, then prove restoring the stale copy cannot mint authority.
    candidate.restore_checkpoint(candidate_base);
    candidate.save_checkpoint(replacement,PREFILL);
    const Exl3GdnLayerCheckpoint stale_same_buffers = replacement;
    upload_rows(candidate_input, 2);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr),
                      2, nullptr, false, true);
    candidate.save_checkpoint(replacement,PREFILL+2);
    require_rejected_without_mutation(
        [&] { candidate.restore_checkpoint(stale_same_buffers); },
        "stale", "GDNPREFIX overwritten-stale-restore guard");
    ++guard_checks;
    candidate.restore_checkpoint(replacement);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr),
                      2, nullptr, false, true);
    require_rejected_without_mutation(
        [&] { candidate.reconstruct_retained_prefix(stale_same_buffers, 1); },
        "absent", "GDNPREFIX overwritten-stale-descriptor guard");
    ++guard_checks;

    candidate.restore_checkpoint(replacement);
    candidate.save_checkpoint(replacement,PREFILL+2);
    upload_rows(candidate_input, 2);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr),
                      2, nullptr, false, true);
    candidate.reconstruct_retained_prefix(replacement, 1);
    cuda_check(cudaDeviceSynchronize(),
               "synchronize GDNPREFIX replacement generation reconstruction");
    require_finite_state(candidate,
                         "GDNPREFIX replacement generation reconstruction");

    candidate.restore_checkpoint(candidate_base);
    candidate.save_checkpoint(candidate_base,PREFILL);
    upload_rows(candidate_input, 2);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr),
                      2, nullptr, false, true);
    candidate.reset();
    require_rejected_without_mutation(
        [&] { candidate.reconstruct_retained_prefix(candidate_base, 1); },
        "absent",
        "GDNPREFIX reset-invalidated guard");
    ++guard_checks;
    candidate.restore_checkpoint(candidate_base);
    candidate.save_checkpoint(candidate_base,PREFILL);
    upload_rows(candidate_input, 2);
    candidate.forward(static_cast<const std::uint16_t*>(candidate_input.ptr),
                      static_cast<std::uint16_t*>(candidate_output.ptr),
                      2, nullptr, false, true);
    candidate.reconstruct_retained_prefix(candidate_base, 1);
    cuda_check(cudaDeviceSynchronize(),
               "synchronize GDNPREFIX post-reset reconstruction");
    require_finite_state(candidate, "GDNPREFIX post-reset reconstruction");

    require(cases == 14, "GDNPREFIX case count mismatch");
    require(guard_checks == 29, "GDNPREFIX guard count mismatch");
    std::cout << "GDNPREFIX PASS cases=14 blocks=2|4|8 all_retained=1"
              << " recurrent=exact conv_slots=4 continuation=exact"
              << " base_rows=11 base_reusable=1 repeat=1 guards="
              << guard_checks << std::endl;
}
