#pragma once

struct PrefixRetentionSnapshot {
    std::array<int, 6> metadata{};
    std::vector<std::uint16_t> logits, continuation_logits, embedding;
    std::array<std::vector<std::uint16_t>, 5> taps;
    std::vector<std::byte> recurrent, convolution, oscar;
    bool operator==(const PrefixRetentionSnapshot&) const = default;
};

void validate_prefix_retention_oscar(const std::vector<std::byte>& state, std::uint32_t context_limit = 4096) {
    std::size_t cursor = 0;
    const auto u32 = [&]() {
        require(cursor + 4 <= state.size(), "prefixretain truncated OSCAR header");
        std::uint32_t value = 0;
        std::memcpy(&value, state.data() + cursor, 4); cursor += 4;
        return value;
    };
    const auto plane = [&](std::size_t bytes, int element_kind) {
        require(bytes <= state.size() - cursor, "prefixretain truncated OSCAR plane");
        if (element_kind == 2) {
            for (std::size_t index = 0; index < bytes; index += 2) {
                std::uint16_t bits = 0; std::memcpy(&bits, state.data() + cursor + index, 2);
                if ((bits & 0x7f80U) == 0x7f80U) throw std::runtime_error("prefixretain OSCAR BF16 nonfinite");
            }
        } else if (element_kind == 4) {
            for (std::size_t index = 0; index < bytes; index += 4) {
                float value = 0; std::memcpy(&value, state.data() + cursor + index, 4);
                if (!std::isfinite(value)) throw std::runtime_error("prefixretain OSCAR metadata nonfinite");
            }
        }
        cursor += bytes;
    };
    for (std::uint32_t bank = 0; bank < 16; ++bank) {
        const auto layer = u32(), context = u32(), prefix = u32(), history = u32(), recent = u32(), head = u32();
        require(layer == 3 + 4 * bank && context <= context_limit, "prefixretain OSCAR layer/context");
        const auto expected_prefix = std::min(context, 64U);
        const auto begin = context <= 64 ? context : std::max(64U, context > 256 ? context - 256 : 0U);
        require(prefix == expected_prefix && history == begin - prefix && recent == context - begin &&
                    head == (begin & 255U),
                "prefixretain OSCAR live extents/head mismatch");
        plane(prefix * 2048ULL, 2); plane(prefix * 2048ULL, 2);
        plane(history * 256ULL, 1); plane(history * 64ULL, 4);
        plane(history * 256ULL, 1); plane(history * 64ULL, 4);
        plane(recent * 2048ULL, 2); plane(recent * 2048ULL, 2);
    }
    require(cursor == state.size(), "prefixretain OSCAR export has unexpected trailing data");
}

void prefix_retention_finite(const std::vector<std::uint16_t>& values,
                             bool bf16, const std::string& label) {
    const std::uint16_t exponent = bf16 ? 0x7f80U : 0x7c00U;
    for (const auto value : values)
        if ((value & exponent) == exponent) throw std::runtime_error(label + " nonfinite");
}

PrefixRetentionSnapshot prefix_retention_snapshot(Exl3TextContext& context, std::uint32_t context_limit = 4096) {
    cuda_check(cudaDeviceSynchronize(), "prefixretain snapshot ordering");
    PrefixRetentionSnapshot result;
    result.metadata = {context.position(), context.device_position_host(),
        context.last_forward_rows(), context.captured_tap_rows(),
        context.captured_embedding_rows(), context.continuation_rows()};
    if (context.position() > 0 && context.logits_device() != nullptr) {
        result.logits = target_continue_device_bits(context.logits_device(), kVocab,
                                                    "prefixretain snapshot logits");
        prefix_retention_finite(result.logits, false, "prefixretain logits");
    }
    if (context.continuation_rows() > 0) {
        result.continuation_logits = context.continuation_logits_bits_host();
        prefix_retention_finite(result.continuation_logits, false, "prefixretain continuation logits");
    }
    if (context.captured_embedding_rows() > 0) {
        result.embedding = context.embedding_bits_host_for_test();
        prefix_retention_finite(result.embedding, false, "prefixretain embedding");
    }
    if (context.captured_tap_rows() > 0) {
        for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
            result.taps[tap] = target_continue_tap_bits(context, kTapLayers[tap],
                                                       context.captured_tap_rows());
            prefix_retention_finite(result.taps[tap], false, "prefixretain tap");
        }
    }
    result.recurrent.reserve(48ULL * 3145728);
    result.convolution.reserve(48ULL * 81920);
    for (int layer = 0; layer < 64; ++layer) {
        if ((layer + 1) % 4 == 0) continue;
        const auto state = context.gdn_state_host(layer);
        require(state.size() * sizeof(float) == 3145728, "prefixretain recurrent size");
        for (const float value : state)
            if (!std::isfinite(value)) throw std::runtime_error("prefixretain recurrent nonfinite");
        const auto* bytes = reinterpret_cast<const std::byte*>(state.data());
        result.recurrent.insert(result.recurrent.end(), bytes, bytes + state.size() * sizeof(float));
        const auto conv = context.gdn_physical_conv_host(layer);
        require(conv.size() * sizeof(std::uint16_t) == 81920, "prefixretain four-slot conv size");
        prefix_retention_finite(conv, true, "prefixretain physical conv");
        bytes = reinterpret_cast<const std::byte*>(conv.data());
        result.convolution.insert(result.convolution.end(), bytes,
                                  bytes + conv.size() * sizeof(std::uint16_t));
    }
    result.oscar = context.oscar_live_state_host_for_test();
    require(!result.oscar.empty(), "prefixretain OSCAR export empty");
    validate_prefix_retention_oscar(result.oscar, context_limit);
    return result;
}

void require_prefix_retention_semantic(const PrefixRetentionSnapshot& actual,
                                       const PrefixRetentionSnapshot& expected,
                                       const std::string& label) {
    require(actual.metadata[0] == expected.metadata[0] &&
                actual.metadata[1] == expected.metadata[1], label + " position mismatch");
    require(actual.logits == expected.logits, label + " logits mismatch");
    require(actual.recurrent == expected.recurrent, label + " recurrent mismatch");
    require(actual.convolution == expected.convolution, label + " four-slot conv mismatch");
    require(actual.oscar == expected.oscar, label + " live OSCAR mismatch");
}

template<class Fn>
void prefix_retention_reject_unchanged(Exl3TextContext& context, Fn&& operation,
                                      const std::string& label) {
    const auto before = prefix_retention_snapshot(context);
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, label + " did not reject");
    require(prefix_retention_snapshot(context) == before, label + " mutated state/payload");
}

void run_prefix_retention_qualification(Exl3TextModel& target,
                                        const std::vector<std::int64_t>& source_ids) {
    require(target.max_context() == 4096 && !source_ids.empty(),
            "prefixretain requires maxctx4096 and real prompt IDs");
    require(env("NINFER_OSCAR_EXL3") == "1", "prefixretain requires canonical OSCAR");
    std::vector<std::int64_t> ids(4096);
    for (std::size_t index = 0; index < ids.size(); ++index)
        ids[index] = source_ids[index % source_ids.size()];
    const std::array<int, 7> prefixes{63, 64, 319, 320, 575, 576, 2047};
    int cases = 0;
    for (const int prefix : prefixes) {
        auto serial = target.create_context(true);
        auto candidate = target.create_context(true);
        require(serial->try_enable_oscar_from_environment() &&
                    candidate->try_enable_oscar_from_environment(), "prefixretain OSCAR enable");
        serial->prepare_transaction();
        candidate->prepare_transaction();
        candidate->prepare_continuation(8);
        const std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + prefix);
        ingest_prefix(*serial, prompt, CommitSink{});
        ingest_prefix(*candidate, prompt, CommitSink{});
        const auto base = prefix_retention_snapshot(*candidate);
        require(base == prefix_retention_snapshot(*serial), "prefixretain paired base mismatch");
        const std::size_t persistent = candidate->persistent_bytes();
        for (const int block : {2, 4, 8}) {
            for (int retained = 1; retained <= block; ++retained) {
                candidate->begin_transaction();
                serial->begin_transaction();
                const std::span<const std::int64_t> teacher(ids.data() + prefix, block);
                candidate->continue_rows(teacher);
                const auto attempted_history=
                    candidate->gdn_continuation_history(0,0,block);
                const auto final_history=
                    candidate->gdn_continuation_history(0,block-1,1);
                require(attempted_history.current() && final_history.current() &&
                            candidate->persistent_bytes()==persistent &&
                            attempted_history.base_position==prefix &&
                            attempted_history.source_rows==block &&
                            final_history.first_row==block-1 &&
                            final_history.q==attempted_history.q+
                                static_cast<std::size_t>(block-1)*16*128,
                        "prefixretain bounded native history view");
                const auto attempted_logits = candidate->continuation_logits_bits_host();
                prefix_retention_finite(attempted_logits, false, "prefixretain attempted logits");
                std::vector<std::uint16_t> serial_logits, serial_embedding;
                std::array<std::vector<std::uint16_t>, 5> serial_taps;
                for (int row = 0; row < retained; ++row) {
                    serial->decode(teacher[static_cast<std::size_t>(row)]);
                    const auto logits = target_continue_device_bits(serial->logits_device(), kVocab,
                                                                    "prefixretain serial logits");
                    serial_logits.insert(serial_logits.end(), logits.begin(), logits.end());
                    const auto embedding = serial->embedding_bits_host_for_test();
                    serial_embedding.insert(serial_embedding.end(), embedding.begin(), embedding.end());
                    for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
                        const auto values = target_continue_tap_bits(*serial, kTapLayers[tap], 1);
                        serial_taps[tap].insert(serial_taps[tap].end(), values.begin(), values.end());
                    }
                }
                candidate->retain_transaction_prefix(retained);
                require(attempted_history.current()==(retained==block),
                        "prefixretain history generation did not follow repair");
                require(candidate->transaction_active(), "prefixretain closed recovery checkpoint");
                const auto actual = prefix_retention_snapshot(*candidate);
                const auto expected = prefix_retention_snapshot(*serial);
                require_prefix_retention_semantic(actual, expected, "prefixretain retained state");
                require(actual.metadata == std::array<int, 6>{prefix + retained, prefix + retained - 1,
                            retained, retained, retained, retained}, "prefixretain retained metadata");
                require(actual.embedding == serial_embedding && actual.taps == serial_taps,
                        "prefixretain retained payload mismatch");
                require(candidate->continuation_logits_bits_host() == serial_logits,
                        "prefixretain retained all-row logits mismatch");
                require(std::equal(serial_logits.begin(), serial_logits.end(), attempted_logits.begin()),
                        "prefixretain modified precomputed logits");
                require(candidate->persistent_bytes() == persistent,
                        "prefixretain allocated persistent bytes during retention");
                for (int step = 0; step < 4; ++step) {
                    const auto token = ids[static_cast<std::size_t>(prefix + retained + step)];
                    candidate->decode(token);
                    serial->decode(token);
                    require(prefix_retention_snapshot(*candidate) == prefix_retention_snapshot(*serial),
                            "prefixretain following M1 state/payload mismatch");
                }
                candidate->rollback_transaction();
                serial->rollback_transaction();
                require(prefix_retention_snapshot(*candidate) == base &&
                            prefix_retention_snapshot(*serial) == base,
                        "prefixretain reusable base rollback mismatch");
                ++cases;
                std::cout << "PREFIXRETAIN_CASE PASS prefix=" << prefix << " B=" << block
                          << " retained=" << retained
                          << " gdn=48 conv_slots=4 oscar_live=exact payload=exact continuation=4"
                          << std::endl;
            }
        }
    }
    require(cases == 98, "prefixretain matrix count mismatch");
    // Context lifecycle and injected failures are a separate small matrix.
    auto candidate = target.create_context(true);
    require(candidate->try_enable_oscar_from_environment(), "prefixretain guard OSCAR enable");
    candidate->prepare_transaction();
    candidate->prepare_continuation(8);
    const std::vector<std::int64_t> prompt(ids.begin(), ids.begin() + 63);
    ingest_prefix(*candidate, prompt, CommitSink{});
    const auto base = prefix_retention_snapshot(*candidate);
    const std::span<const std::int64_t> teacher(ids.data() + 63, 8);
    int guards = 0;
    prefix_retention_reject_unchanged(*candidate,
        [&] { candidate->retain_transaction_prefix(1); }, "prefixretain absent transaction");
    ++guards;
    candidate->begin_transaction();
    prefix_retention_reject_unchanged(*candidate,
        [&] { candidate->retain_transaction_prefix(1); }, "prefixretain absent attempt");
    ++guards;
    candidate->continue_rows(teacher);
    const auto guard_history=candidate->gdn_continuation_history(0,0,8);
    bool missing_history_row=false;
    try{(void)candidate->gdn_continuation_history(0,8,1);}
    catch(const std::invalid_argument&){missing_history_row=true;}
    require(guard_history.current()&&missing_history_row,
            "prefixretain missing history row was exposed");
    ++guards;
    for (const int invalid : {-1, 0, 9}) {
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->retain_transaction_prefix(invalid); }, "prefixretain invalid count");
        ++guards;
    }
    cudaStream_t other = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "prefixretain guard stream");
    try {
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->retain_transaction_prefix(1, other); }, "prefixretain wrong stream");
        ++guards;
    } catch (...) { cudaStreamDestroy(other); throw; }
    cuda_check(cudaStreamDestroy(other), "prefixretain destroy guard stream");
    candidate->retain_transaction_prefix(1);
    require(!guard_history.current(),
            "prefixretain partial repair left old history current");
    prefix_retention_reject_unchanged(*candidate,
        [&] { candidate->retain_transaction_prefix(1); }, "prefixretain consumed capability");
    ++guards;
    candidate->rollback_transaction();
    require(prefix_retention_snapshot(*candidate) == base, "prefixretain guard rollback");
    for (const bool another_continuation : {false, true}) {
        candidate->begin_transaction();
        candidate->continue_rows(teacher);
        if (another_continuation) candidate->continue_rows(teacher.first(2));
        else candidate->decode(ids[71]);
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->retain_transaction_prefix(1); }, "prefixretain intervening forward");
        ++guards;
        candidate->rollback_transaction();
        require(prefix_retention_snapshot(*candidate) == base, "prefixretain intervening rollback");
    }
    int failures = 0;
    for (const int fail_layer : {0, 31, 63}) {
        candidate->begin_transaction();
        candidate->continue_rows(teacher);
        const auto attempt = prefix_retention_snapshot(*candidate);
        bool failed = false;
        try { candidate->retain_transaction_prefix_for_test(1, fail_layer); }
        catch (const std::runtime_error& error) {
            require(std::string(error.what()) ==
                        "P2 injected retained-prefix failure after model layer " + std::to_string(fail_layer),
                    "prefixretain unexpected reconstruction exception: " + std::string(error.what()));
            failed = true;
        }
        require(failed && candidate->transaction_active(), "prefixretain injection lost recovery checkpoint");
        require(candidate->logits_device() == nullptr && candidate->continuation_logits_device() == nullptr &&
                    candidate->last_forward_rows() == 0 && candidate->captured_tap_rows() == 0 &&
                    candidate->captured_embedding_rows() == 0,
                "prefixretain failure exposed valid-looking derived outputs");
        const auto partial = prefix_retention_snapshot(*candidate);
        require(partial.recurrent != attempt.recurrent || partial.oscar != attempt.oscar,
                "prefixretain failure was injected before reconstruction");
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->retain_transaction_prefix(1); }, "prefixretain failed capability retry");
        ++guards;
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->commit_transaction(); }, "prefixretain failed commit");
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->decode(ids[63]); }, "prefixretain failed decode");
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->continue_rows(teacher); }, "prefixretain failed continuation");
        prefix_retention_reject_unchanged(*candidate,
            [&] { (void)candidate->profile_decode(ids[63]); }, "prefixretain failed profile");
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->prefill(teacher); }, "prefixretain failed prefill");
        prefix_retention_reject_unchanged(*candidate,
            [&] { (void)candidate->logits_host(); }, "prefixretain failed logits getter");
        guards += 6;
        candidate->rollback_transaction();
        require(prefix_retention_snapshot(*candidate) == base, "prefixretain injected failure rollback");
        ++failures;
        std::cout << "PREFIXRETAIN_FAILURE PASS after_layer=" << fail_layer
                  << " rollback=exact checkpoint_active=1" << std::endl;
    }
    // Capture tests keep all snapshots and synchronization outside capture.
    cuda_check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "prefixretain capture stream");
    cudaEvent_t sentinel = nullptr;
    cudaGraph_t graph = nullptr;
    cuda_check(cudaEventCreateWithFlags(&sentinel, cudaEventDisableTiming), "prefixretain capture sentinel");
    try {
        const auto before_begin_capture = prefix_retention_snapshot(*candidate);
        cuda_check(cudaStreamBeginCapture(other, cudaStreamCaptureModeThreadLocal), "prefixretain begin save capture");
        bool rejected = false;
        try { candidate->begin_transaction(other); } catch (const std::exception&) { rejected = true; }
        cuda_check(cudaEventRecord(sentinel, other), "prefixretain save capture sentinel");
        cuda_check(cudaStreamEndCapture(other, &graph), "prefixretain end save capture");
        cuda_check(cudaGraphDestroy(graph), "prefixretain destroy save capture"); graph = nullptr;
        require(rejected && !candidate->transaction_active() &&
                    prefix_retention_snapshot(*candidate) == before_begin_capture,
                "prefixretain captured begin did not reject unchanged");
        ++guards;
        candidate->begin_transaction(other);
        candidate->continue_rows(teacher.first(2), other);
        cuda_check(cudaStreamSynchronize(other), "prefixretain explicit stream attempt");
        const auto before_capture = prefix_retention_snapshot(*candidate);
        cuda_check(cudaStreamBeginCapture(other, cudaStreamCaptureModeThreadLocal), "prefixretain begin active capture");
        rejected = false;
        try { candidate->retain_transaction_prefix(1, other); } catch (const std::exception&) { rejected = true; }
        cuda_check(cudaEventRecord(sentinel, other), "prefixretain active capture sentinel");
        cuda_check(cudaStreamEndCapture(other, &graph), "prefixretain end active capture");
        cuda_check(cudaGraphDestroy(graph), "prefixretain destroy active capture"); graph = nullptr;
        require(rejected && prefix_retention_snapshot(*candidate) == before_capture,
                "prefixretain active capture did not reject unchanged");
        ++guards;
        candidate->retain_transaction_prefix(1, other);
        cuda_check(cudaStreamSynchronize(other), "prefixretain explicit stream retention");
        auto control = target.create_context(true);
        require(control->try_enable_oscar_from_environment(), "prefixretain stream control OSCAR");
        ingest_prefix(*control, prompt, CommitSink{});
        control->decode(ids[63]);
        const auto actual = prefix_retention_snapshot(*candidate);
        const auto expected = prefix_retention_snapshot(*control);
        require_prefix_retention_semantic(actual, expected, "prefixretain explicit stream");
        require(actual.taps == expected.taps && actual.embedding == expected.embedding,
                "prefixretain explicit stream payload");
        candidate->rollback_transaction(other);
        cuda_check(cudaStreamSynchronize(other), "prefixretain explicit stream rollback");
        require(prefix_retention_snapshot(*candidate) == base, "prefixretain stream base rollback");
        // A legacy forward on another explicitly ordered stream may execute,
        // but it must not authorize retention against a different-stream save.
        candidate->begin_transaction();
        cuda_check(cudaDeviceSynchronize(), "prefixretain cross-stream save dependency");
        candidate->continue_rows(teacher.first(2), other);
        cuda_check(cudaStreamSynchronize(other), "prefixretain cross-stream forward dependency");
        prefix_retention_reject_unchanged(*candidate,
            [&] { candidate->retain_transaction_prefix(1, other); }, "prefixretain mismatched source stream");
        ++guards;
        candidate->rollback_transaction();
        require(prefix_retention_snapshot(*candidate) == base, "prefixretain mismatched stream rollback");
    } catch (...) {
        if (graph) cudaGraphDestroy(graph);
        cudaEventDestroy(sentinel); cudaStreamDestroy(other);
        throw;
    }
    cuda_check(cudaEventDestroy(sentinel), "prefixretain destroy sentinel");
    cuda_check(cudaStreamDestroy(other), "prefixretain destroy capture stream");
    // A successful retained transaction commits without losing its payload.
    candidate->begin_transaction();
    candidate->continue_rows(teacher);
    candidate->retain_transaction_prefix(3);
    const auto retained = prefix_retention_snapshot(*candidate);
    candidate->commit_transaction();
    require(prefix_retention_snapshot(*candidate) == retained && !candidate->transaction_active(),
            "prefixretain commit changed retained payload");
    prefix_retention_reject_unchanged(*candidate,
        [&] { candidate->retain_transaction_prefix(1); }, "prefixretain committed capability");
    ++guards;
    candidate->reset();
    prefix_retention_reject_unchanged(*candidate,
        [&] { candidate->retain_transaction_prefix(1); }, "prefixretain reset capability");
    ++guards;
    ingest_prefix(*candidate, prompt, CommitSink{});
    require(prefix_retention_snapshot(*candidate) == base, "prefixretain reset reingest base");
    candidate->begin_transaction();
    candidate->continue_rows(teacher.first(2));
    candidate->retain_transaction_prefix(1);
    candidate->rollback_transaction();
    require(prefix_retention_snapshot(*candidate) == base, "prefixretain reset reuse rollback");
    require(guards == 36 && failures == 3, "prefixretain guard/failure matrix count mismatch");
    std::cout << "PREFIXRETAIN PASS cases=" << cases << " guards=" << guards
              << " failures=" << failures
              << " gdn=48 conv_slots=4 oscar_live=exact payload=exact continuation=4"
              << " new_device_bytes=0 caller_ring=unqualified" << std::endl;
}
