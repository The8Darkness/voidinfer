#pragma once

#include <cstddef>

// Focused eager C2 request-isolation differential.  Logits/taps are exact
// represented F16 bits, positions are exact metadata, and the exported GDN
// state, physical GDN convolution state, and canonical OSCAR live state are
// retained and compared byte-for-byte.  The public test surface does not
// export the complete full-attention KV image, so this mode makes no claim
// about that unobserved representation.

struct C2IsolationSnapshot {
    int position = 0;
    int device_position = 0;
    int last_forward_rows = 0;
    int captured_tap_rows = 0;
    int captured_embedding_rows = 0;
    std::vector<std::uint16_t> logits;
    std::array<std::vector<std::uint16_t>, kTapCount> taps;
    std::vector<std::byte> gdn;
    std::vector<std::byte> convolution;
    std::vector<std::byte> oscar;
};

C2IsolationSnapshot c2_isolation_snapshot(Exl3TextContext& context,
                                          const std::string& label) {
    cuda_check(cudaDeviceSynchronize(), (label + " ordering").c_str());
    C2IsolationSnapshot result;
    result.position = context.position();
    result.device_position = context.device_position_host();
    result.last_forward_rows = context.last_forward_rows();
    result.captured_tap_rows = context.captured_tap_rows();
    result.captured_embedding_rows = context.captured_embedding_rows();
    if (result.position > 0 && context.logits_device() != nullptr) {
        result.logits = target_continue_device_bits(
            context.logits_device(), kVocab, label + " logits");
    }
    if (result.captured_tap_rows > 0) {
        for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
            result.taps[tap] = target_continue_tap_bits(
                context, kTapLayers[tap], result.captured_tap_rows);
        }
    }
    result.gdn.reserve(48ULL * 3145728ULL);
    result.convolution.reserve(48ULL * 81920ULL * sizeof(std::uint16_t));
    for (int layer = 0; layer < 64; ++layer) {
        if (layer >= 3 && (layer - 3) % 4 == 0) continue;
        const auto state = context.gdn_state_host(layer);
        const auto* bytes = reinterpret_cast<const std::byte*>(state.data());
        result.gdn.insert(result.gdn.end(), bytes, bytes + state.size() * sizeof(float));
        const auto convolution = context.gdn_physical_conv_host(layer);
        const auto* convolution_bytes = reinterpret_cast<const std::byte*>(convolution.data());
        result.convolution.insert(result.convolution.end(), convolution_bytes,
                                  convolution_bytes + convolution.size() * sizeof(std::uint16_t));
    }
    result.oscar = context.oscar_live_state_host_for_test();
    return result;
}

void c2_isolation_require_equal(const C2IsolationSnapshot& actual,
                                const C2IsolationSnapshot& expected,
                                const std::string& label) {
    require(actual.position == expected.position &&
                actual.device_position == expected.device_position &&
                actual.last_forward_rows == expected.last_forward_rows &&
                actual.captured_tap_rows == expected.captured_tap_rows &&
                actual.captured_embedding_rows == expected.captured_embedding_rows,
            label + " metadata mismatch");
    require(actual.logits == expected.logits, label + " logits mismatch");
    require(actual.taps == expected.taps, label + " tap mismatch");
    require(actual.gdn == expected.gdn, label + " GDN mismatch");
    require(actual.convolution == expected.convolution,
            label + " physical convolution mismatch");
    require(actual.oscar == expected.oscar, label + " OSCAR live state mismatch");
}

void c2_isolation_ingest(Exl3TextContext& context,
                         const std::vector<std::int64_t>& prompt,
                         int prefill_width = 1024) {
    require(prefill_width == 1024 || prefill_width == 128 || prefill_width == 16,
            "c2isolation prefill width must be 1024, 128, or 16");
    require(!prompt.empty() && prompt.size() <= 1024,
            "c2isolation prefix extent");
    if (prompt.size() <= 16) {
        context.prefill(prompt);
    } else {
        // Match the acceptance harness's qualified eager route: initial16,
        // followed by deterministic wide suffix chunks.  Width 1024 retains
        // the existing one-suffix C2/C4 sequence; width 128 is the bounded
        // C8 memory candidate and uses 128,128,48 for the 320-token prefix.
        context.prefill(std::span<const std::int64_t>(prompt.data(), 16));
        std::size_t offset = 16;
        while (offset < prompt.size()) {
            const std::size_t rows = std::min<std::size_t>(
                static_cast<std::size_t>(prefill_width), prompt.size() - offset);
            if (prefill_width == 16) {
                // The narrow constructor has no wide admission.  Its
                // established append API accepts at most eight rows, so
                // preserve the width-16 sequence as two ordered <=8 chunks.
                const std::size_t narrow_rows = std::min<std::size_t>(8, rows);
                context.append_prefill(std::span<const std::int64_t>(
                    prompt.data() + offset, narrow_rows));
                offset += narrow_rows;
            } else {
                context.append_prefill_wide(std::span<const std::int64_t>(
                    prompt.data() + offset, rows));
                offset += rows;
            }
        }
    }
    cuda_check(cudaDeviceSynchronize(), "c2isolation prefix sync");
}

struct C2IsolationSequence {
    std::vector<C2IsolationSnapshot> snapshots;
};

C2IsolationSequence c2_isolation_oracle(Exl3TextModel& target,
                                        const std::vector<std::int64_t>& prompt,
                                        const std::vector<std::int64_t>& teacher,
                                        const std::string& label,
                                        int prefill_width = 1024) {
    // P17's wide slab is an opt-in candidate allocation.  Keep each
    // single-context oracle on the pre-P17 path, then restore the caller's
    // candidate setting before the paired contexts are constructed.
    struct ScopedSerialGdnSlab {
        std::string saved = env("NINFER_EXL3_GDN_WIDE_SLAB");
        ScopedSerialGdnSlab() { _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB", "0"); }
        ~ScopedSerialGdnSlab() { _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB", saved.c_str()); }
    } serial_slab;
    auto context = target.create_context(true);
    require(context->try_enable_oscar_from_environment(),
            label + " OSCAR enable failed");
    c2_isolation_ingest(*context, prompt, prefill_width);
    C2IsolationSequence result;
    result.snapshots.push_back(c2_isolation_snapshot(*context, label + " prefix"));
    for (std::size_t row = 0; row < teacher.size(); ++row) {
        context->decode(teacher[row]);
        result.snapshots.push_back(c2_isolation_snapshot(
            *context, label + " continuation " + std::to_string(row)));
    }
    return result;
}

std::vector<std::int64_t> c2_isolation_slice(
    const std::vector<std::int64_t>& source, std::size_t offset, int rows,
    const std::string& label) {
    require(rows >= 1 && offset + static_cast<std::size_t>(rows) <= source.size(),
            label + " source slice extent");
    std::vector<std::int64_t> result(source.begin() + offset,
                                     source.begin() + offset + rows);
    for (const auto token : result)
        require(token >= 0 && token < kVocab, label + " token extent");
    require(result.back() != kMaskToken, label + " ends in mask token");
    return result;
}

void run_context_isolation_qualification(
    Exl3TextModel& target, const std::vector<std::int64_t>& source_ids,
    const std::string& output_path) {
    require(target.max_context() == 1024,
            "c2isolation requires max context 1024");
    require(env("NINFER_OSCAR_EXL3") == "1",
            "c2isolation requires canonical OSCAR");
    require(env("NINFER_EXL3_PREFILL_ROWPAIR_K6") == "0",
            "c2isolation requires explicit P16 rowpair0");
    const std::string count_setting = env("NINFER_CONTEXT_ISOLATION_COUNT");
    require(count_setting.empty() || count_setting == "2" || count_setting == "4" ||
                count_setting == "8",
            "c2isolation count must be unset, 2, 4, or 8");
    const int context_count = count_setting.empty() ? 2 : std::stoi(count_setting);
    const std::string prefill_setting = env("NINFER_CONTEXT_ISOLATION_PREFILL");
    require(prefill_setting.empty() || prefill_setting == "1024" ||
                prefill_setting == "128" || prefill_setting == "16",
            "c2isolation prefill must be unset, 1024, 128, or 16");
    const int prefill_width = prefill_setting.empty() ? 1024 : std::stoi(prefill_setting);
    if (prefill_width == 16) {
        require(context_count == 2 && env("NINFER_EXL3_WIDE_PREFILL") == "0" &&
                    env("NINFER_EXL3_PREFILL_STAGED_REDUCTION") == "0" &&
                    env("NINFER_EXL3_PREFILL_WIDE64") == "0" &&
                    env("NINFER_EXL3_PREFILL_WIDE128") == "0" &&
                    env("NINFER_EXL3_PREFILL_WIDE256") == "0" &&
                    env("NINFER_EXL3_PREFILL_WIDE512") == "0" &&
                    env("NINFER_EXL3_PREFILL_WIDE1024") == "0" &&
                    env("NINFER_EXL3_PREFILL_CHUNK") == "16",
                "c2isolation width16 requires the explicit narrow control selectors");
    } else {
        require(env("NINFER_EXL3_WIDE_PREFILL") == "1",
                "c2isolation requires wide eager prefill");
    }
    if (prefill_width == 1024) {
        require(env("NINFER_EXL3_PREFILL_WIDE1024") == "1" &&
                    env("NINFER_EXL3_PREFILL_CHUNK") == "1024",
                "c2isolation requires the P14 1024-row eager overlay");
    } else if (prefill_width == 128) {
        require(env("NINFER_EXL3_PREFILL_WIDE128") == "1" &&
                    env("NINFER_EXL3_PREFILL_WIDE64") == "1" &&
                    env("NINFER_EXL3_PREFILL_WIDE256") == "0" &&
                    env("NINFER_EXL3_PREFILL_WIDE512") == "0" &&
                    env("NINFER_EXL3_PREFILL_WIDE1024") == "0" &&
                    env("NINFER_EXL3_PREFILL_STAGED_REDUCTION") == "1" &&
                    env("NINFER_EXL3_SHARED_ACCUM") == "1" &&
                    env("NINFER_EXL3_PREFILL_CHUNK") == "128",
                "c8isolation requires the P17 wide128 overlay");
    }
    if (context_count == 4 || context_count == 8) {
        const std::string slab = env("NINFER_EXL3_GDN_WIDE_SLAB");
        require((slab.empty() || slab == "1") &&
                    (env("NINFER_EXL3_SHARED_LAYER_SCRATCH").empty() ||
                     env("NINFER_EXL3_SHARED_LAYER_SCRATCH") == "1"),
                "c4isolation requires effective P17 GDN wide slab/shared scratch");
    }
    if (context_count == 8) {
        require(prefill_width == 128,
                "c8isolation requires prefill width 128");
    }
    const std::string admission_setting = env("NINFER_CONTEXT_ISOLATION_ADMISSION_ONLY");
    require(admission_setting.empty() || admission_setting == "0" || admission_setting == "1",
            "c8isolation admission-only must be unset, 0, or 1");
    const bool admission_only = admission_setting == "1";
    require(!admission_only || (context_count == 8 && prefill_width == 128),
            "admission-only requires count8 and prefill width128");
    require(!source_ids.empty(), "c2isolation prompt fixture is empty");
    require(!output_path.empty() && !std::filesystem::exists(output_path),
            "c2isolation output must be a new file");

    constexpr int kTeacherRows = 4;
    const std::array<int, 3> prefixes{16, 32, 320};
    const int max_prefix = prefixes.back();
    constexpr std::size_t kStreamOffset = 512;
    const std::size_t required_source =
        static_cast<std::size_t>(context_count - 1) * kStreamOffset +
        static_cast<std::size_t>(max_prefix + kTeacherRows);
    require(source_ids.size() >= required_source,
            "c2isolation prompt fixture is too short for requested slices");

    std::ofstream report(output_path);
    require(report.good(), "c2isolation cannot open report");
    report << "# C2/C4/C8 eager request isolation\n\n"
           << "Scope: " << context_count << " contexts, prefill_width=" << prefill_width
           << ", serialized per-context oracle versus eager "
              "interleaving under P14 full overlay with P16 rowpair=0.\n"
           << "Oracle: exact F16 logits/taps, position metadata, byte-exact exported "
              "GDN recurrent state, physical convolution state, and OSCAR live state. "
              "Full-attention KV is not exported by this helper.\n"
           << "Termination: nonzero on any mismatch; each case is flushed below.\n\n"
           << "|contexts|prefix|teacher_rows|reset_other|result|\n"
           << "|---:|---:|---:|---|---|\n";
    report.flush();

    if (admission_only) {
        constexpr std::size_t kAdmissionFloorBytes = 1ULL << 30;
        std::size_t free_before = 0;
        std::size_t total_before = 0;
        bool have_before = false;
        std::size_t free_after = 0;
        std::size_t total_after = 0;
        bool have_after = false;
        try {
            if (cudaMemGetInfo(&free_before, &total_before) == cudaSuccess) {
                have_before = true;
                report << "Admission free_before_bytes=" << free_before
                       << " total_before_bytes=" << total_before << "\n";
                report.flush();
            }
            std::vector<std::unique_ptr<Exl3TextContext>> admission_contexts;
            admission_contexts.reserve(static_cast<std::size_t>(context_count));
            std::uint64_t persistent_bytes = 0;
            for (int stream = 0; stream < context_count; ++stream) {
                admission_contexts.push_back(target.create_context(true));
                require(admission_contexts.back()->try_enable_oscar_from_environment(),
                        "c8isolation admission OSCAR enable failed");
                persistent_bytes += admission_contexts.back()->persistent_bytes();
            }
            cuda_check(cudaDeviceSynchronize(), "c8isolation admission sync");
            cuda_check(cudaMemGetInfo(&free_after, &total_after),
                       "c8isolation admission memory query");
            have_after = true;
            report << "Admission contexts=" << context_count
                   << " prefill_width=" << prefill_width
                   << " persistent_bytes_total=" << persistent_bytes
                   << " free_bytes=" << free_after
                   << " total_bytes=" << total_after << "\n";
            require(free_after >= kAdmissionFloorBytes,
                    "c8isolation admission free-memory floor failed");
            report << "Status: PASS\n";
            report.flush();
            std::cout << "CONTEXT_ISOLATION_ADMISSION PASS contexts=" << context_count
                      << " prefill_width=" << prefill_width
                      << " free_bytes=" << free_after
                      << " total_bytes=" << total_after
                      << " persistent_bytes=" << persistent_bytes << '\n';
        } catch (const std::exception& error) {
            report << "Admission contexts=" << context_count
                   << " prefill_width=" << prefill_width;
            if (have_before)
                report << " free_before_bytes=" << free_before
                       << " total_before_bytes=" << total_before;
            if (have_after)
                report << " free_bytes=" << free_after
                       << " total_bytes=" << total_after;
            report << "\nStatus: FAIL\nFailure: " << error.what() << "\n";
            report.flush();
            throw;
        }
        return;
    }

    int cases = 0;
    bool allocation_reported = false;
    for (const int prefix_rows : prefixes) {
        std::vector<std::vector<std::int64_t>> prompts(
            static_cast<std::size_t>(context_count));
        std::vector<std::vector<std::int64_t>> teachers(
            static_cast<std::size_t>(context_count));
        for (int stream = 0; stream < context_count; ++stream) {
            const std::size_t offset = static_cast<std::size_t>(stream) * kStreamOffset;
            const std::string stream_label =
                "c2isolation stream " + std::to_string(stream);
            prompts[static_cast<std::size_t>(stream)] = c2_isolation_slice(
                source_ids, offset, prefix_rows, stream_label + " prompt");
            teachers[static_cast<std::size_t>(stream)] = c2_isolation_slice(
                source_ids, offset + static_cast<std::size_t>(prefix_rows),
                kTeacherRows, stream_label + " teacher");
            for (int prior = 0; prior < stream; ++prior) {
                require(prompts[static_cast<std::size_t>(stream)] !=
                            prompts[static_cast<std::size_t>(prior)],
                        "c2isolation prompt slices are not distinct");
            }
        }

        // Build and retain only host snapshots before creating the paired
        // contexts, keeping the GPU oracle phase single-context.
        std::vector<C2IsolationSequence> oracles;
        oracles.reserve(static_cast<std::size_t>(context_count));
        for (int stream = 0; stream < context_count; ++stream) {
            oracles.push_back(c2_isolation_oracle(
                target, prompts[static_cast<std::size_t>(stream)],
                teachers[static_cast<std::size_t>(stream)],
                "c2isolation oracle " + std::to_string(stream), prefill_width));
            require(oracles.back().snapshots.size() ==
                        static_cast<std::size_t>(kTeacherRows + 1),
                    "c2isolation oracle snapshot count");
        }

        // Every oracle context is destroyed on return above before the live
        // target-context vector is constructed.
        std::vector<std::unique_ptr<Exl3TextContext>> contexts;
        contexts.reserve(static_cast<std::size_t>(context_count));
        for (int stream = 0; stream < context_count; ++stream) {
            contexts.push_back(target.create_context(true));
            require(contexts.back()->try_enable_oscar_from_environment(),
                    "c2isolation interleaved OSCAR enable failed");
        }
        if (!allocation_reported) {
            const char* slab = std::getenv("NINFER_EXL3_GDN_WIDE_SLAB");
            report << "Allocation telemetry: contexts=" << context_count
                   << " prefill_width=" << prefill_width
                   << " gdn_wide_slab=" << (slab ? slab : "");
            for (int stream = 0; stream < context_count; ++stream) {
                report << " context_" << stream << "_persistent_bytes="
                       << contexts[static_cast<std::size_t>(stream)]->persistent_bytes();
            }
            report << "\n\n";
            report.flush();
            std::cout << "C2_ISOLATION_ALLOCATION contexts=" << context_count
                      << " prefill_width=" << prefill_width
                      << " gdn_wide_slab=" << (slab ? slab : "");
            for (int stream = 0; stream < context_count; ++stream) {
                std::cout << " context_" << stream << "_persistent_bytes="
                          << contexts[static_cast<std::size_t>(stream)]->persistent_bytes();
            }
            std::cout << '\n';
            allocation_reported = true;
        }
        for (int stream = 0; stream < context_count; ++stream) {
            c2_isolation_ingest(*contexts[static_cast<std::size_t>(stream)],
                                prompts[static_cast<std::size_t>(stream)], prefill_width);
            c2_isolation_require_equal(
                c2_isolation_snapshot(
                    *contexts[static_cast<std::size_t>(stream)],
                    "c2isolation interleaved prefix " + std::to_string(stream)),
                oracles[static_cast<std::size_t>(stream)].snapshots[0],
                "c2isolation stream " + std::to_string(stream) + " prefix");
        }

        // Round-robin the first three rows.  Stream zero completes its fourth
        // row before reset while every other stream still has that row pending.
        for (int row = 0; row < kTeacherRows - 1; ++row) {
            for (int stream = 0; stream < context_count; ++stream) {
                auto& context = *contexts[static_cast<std::size_t>(stream)];
                context.decode(teachers[static_cast<std::size_t>(stream)]
                                  [static_cast<std::size_t>(row)]);
                c2_isolation_require_equal(
                    c2_isolation_snapshot(
                        context, "c2isolation interleaved continuation"),
                    oracles[static_cast<std::size_t>(stream)]
                        .snapshots[static_cast<std::size_t>(row + 1)],
                    "c2isolation stream " + std::to_string(stream) +
                        " continuation " + std::to_string(row));
            }
        }
        auto& first_context = *contexts[0];
        first_context.decode(teachers[0][kTeacherRows - 1]);
        c2_isolation_require_equal(
            c2_isolation_snapshot(first_context, "c2isolation stream 0 final"),
            oracles[0].snapshots[static_cast<std::size_t>(kTeacherRows)],
            "c2isolation stream 0 continuation 3");
        std::vector<C2IsolationSnapshot> before_reset;
        before_reset.reserve(static_cast<std::size_t>(context_count - 1));
        for (int stream = 1; stream < context_count; ++stream) {
            before_reset.push_back(c2_isolation_snapshot(
                *contexts[static_cast<std::size_t>(stream)],
                "c2isolation pending before stream 0 reset"));
        }
        first_context.reset();
        for (int stream = 1; stream < context_count; ++stream) {
            c2_isolation_require_equal(
                c2_isolation_snapshot(
                    *contexts[static_cast<std::size_t>(stream)],
                    "c2isolation pending after stream 0 reset"),
                before_reset[static_cast<std::size_t>(stream - 1)],
                "c2isolation stream " + std::to_string(stream) +
                    " survives stream 0 reset");
            auto& context = *contexts[static_cast<std::size_t>(stream)];
            context.decode(teachers[static_cast<std::size_t>(stream)]
                              [static_cast<std::size_t>(kTeacherRows - 1)]);
            c2_isolation_require_equal(
                c2_isolation_snapshot(context, "c2isolation stream final"),
                oracles[static_cast<std::size_t>(stream)]
                    .snapshots[static_cast<std::size_t>(kTeacherRows)],
                "c2isolation stream " + std::to_string(stream) +
                    " continuation 3");
        }
        ++cases;
        report << '|' << context_count << '|' << prefix_rows << '|' << kTeacherRows
               << "|stream 0 reset after row 3, before other row 3|PASS|\n";
        report.flush();
        std::cout << "C2_ISOLATION_CASE PASS prefix=" << prefix_rows
                  << " teacher_rows=" << kTeacherRows
                  << " reset=A_after_row3_before_B_row3 contexts=" << context_count
                  << " prefill_width=" << prefill_width << '\n';
    }
    report << "\nCases: " << cases << "\nStatus: PASS\n";
    report.flush();
    std::cout << "CONTEXT_ISOLATION PASS cases=" << cases
              << " contexts=" << context_count
              << " prefill_width=" << prefill_width
              << " oracle=logits+taps+position+gdn+conv+oscar_exact full_kv=unobserved\n";
}
