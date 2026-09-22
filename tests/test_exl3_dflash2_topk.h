#pragma once

std::uint32_t topk_float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Independent library partial sort for numerically ordered rows. For NaN rows,
// the reference is the legacy insertion sequence, not an invented NaN ordering.
std::vector<int> topk_cpu_ids(const std::uint16_t* input, int vocab) {
    std::vector<float> values(vocab);
    bool nan = false;
    for (int i = 0; i < vocab; ++i) {
        values[i] = half_to_float(input[i]);
        nan = nan || std::isnan(values[i]);
    }
    if (!nan) {
        std::vector<int> ids(vocab);
        std::iota(ids.begin(), ids.end(), 0);
        std::partial_sort(ids.begin(), ids.begin() + 16, ids.end(),
            [&](int a, int b) {
                return values[a] > values[b] || (values[a] == values[b] && a < b);
            });
        ids.resize(16);
        return ids;
    }
    std::vector<int> ids;
    for (int i = 0; i < vocab; ++i) {
        if (ids.size() == 16 && !(values[i] > values[ids.back()])) continue;
        if (ids.size() == 16) ids.pop_back();
        auto position = ids.end();
        while (position != ids.begin() && values[*(position - 1)] < values[i]) --position;
        ids.insert(position, i);
    }
    return ids;
}

void check_topk_case(const std::vector<std::uint16_t>& input, int rows, int vocab,
                     const std::string& name, std::ostream& out,
                     std::ostream* timing = nullptr) {
    require(rows >= 1 && rows <= 7 && input.size() == static_cast<std::size_t>(rows) * vocab,
            "top-K test input extent");
    constexpr std::size_t guard = 32;
    constexpr std::int64_t id_canary = 0x1122334455667788LL;
    constexpr std::uint32_t value_canary = 0x3f123456U;
    const std::size_t count = static_cast<std::size_t>(rows) * 16;
    DeviceBuffer in(input.size() * sizeof(std::uint16_t));
    DeviceBuffer ids((count + 2 * guard) * sizeof(std::int64_t));
    DeviceBuffer values((count + 2 * guard) * sizeof(std::uint32_t));
    auto* ip = static_cast<std::uint16_t*>(in.get());
    auto* id_base = static_cast<std::int64_t*>(ids.get());
    auto* value_base = static_cast<std::uint32_t*>(values.get());
    auto* idp = id_base + guard;
    auto* vp = reinterpret_cast<float*>(value_base + guard);
    cuda_check(cudaMemcpy(ip, input.data(), input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice), "top-K input upload");
    const std::vector<std::int64_t> id_initial(count + 2 * guard, id_canary);
    const std::vector<std::uint32_t> value_initial(count + 2 * guard, value_canary);
    std::vector<std::int64_t> first_ids;
    std::vector<std::uint32_t> first_values;
    int nan_rows = 0;
    for (int r = 0; r < rows; ++r)
        nan_rows += std::any_of(input.begin() + static_cast<std::size_t>(r) * vocab,
            input.begin() + static_cast<std::size_t>(r + 1) * vocab,
            [](auto bits) { return std::isnan(half_to_float(bits)); });
    for (const int mode : {0, 1, 2}) {
        const bool parallel = mode == 1;
        const bool local_merge = mode == 2;
        cuda_check(cudaMemcpy(id_base, id_initial.data(), ids.bytes(), cudaMemcpyHostToDevice), "top-K ID guards");
        cuda_check(cudaMemcpy(value_base, value_initial.data(), values.bytes(), cudaMemcpyHostToDevice), "top-K value guards");
        std::vector<std::int64_t> got_ids(count + 2 * guard), repeat_ids(count + 2 * guard);
        std::vector<std::uint32_t> got_values(count + 2 * guard), repeat_values(count + 2 * guard);
        ninfer::exl3::dflash2_topk16_for_test(
            ip, rows, vocab, idp, vp, parallel, local_merge);
        cuda_check(cudaMemcpy(got_ids.data(), id_base, ids.bytes(), cudaMemcpyDeviceToHost), "top-K IDs download");
        cuda_check(cudaMemcpy(got_values.data(), value_base, values.bytes(), cudaMemcpyDeviceToHost), "top-K values download");
        ninfer::exl3::dflash2_topk16_for_test(
            ip, rows, vocab, idp, vp, parallel, local_merge);
        cuda_check(cudaMemcpy(repeat_ids.data(), id_base, ids.bytes(), cudaMemcpyDeviceToHost), "top-K repeat IDs");
        cuda_check(cudaMemcpy(repeat_values.data(), value_base, values.bytes(), cudaMemcpyDeviceToHost), "top-K repeat values");
        require(got_ids == repeat_ids && got_values == repeat_values, "top-K repeat mismatch");
        for (std::size_t i = 0; i < count + 2 * guard; ++i)
            if (i < guard || i >= guard + count)
                require(got_ids[i] == id_canary && got_values[i] == value_canary, "top-K output guard overwritten");
        for (int r = 0; r < rows; ++r) {
            const auto* row = input.data() + static_cast<std::size_t>(r) * vocab;
            const auto expected = topk_cpu_ids(row, vocab);
            require(expected.size() == 16, "top-K CPU extent");
            for (int k = 0; k < 16; ++k) {
                const auto offset = guard + static_cast<std::size_t>(r) * 16 + k;
                require(got_ids[offset] == expected[k], "top-K independent CPU ID mismatch: " + name);
                const float expected_value = half_to_float(row[expected[k]]);
                float actual_value;
                std::memcpy(&actual_value, &got_values[offset], sizeof(actual_value));
                require(std::isnan(expected_value) ? std::isnan(actual_value)
                    : got_values[offset] == topk_float_bits(expected_value), "top-K CPU value mismatch");
            }
        }
        if (mode == 0) { first_ids = got_ids; first_values = got_values; }
        else require(first_ids == got_ids && first_values == got_values,
                     "top-K exact original-GPU differential failed");
    }
    std::vector<std::uint16_t> unchanged(input.size());
    cuda_check(cudaMemcpy(unchanged.data(), ip, in.bytes(), cudaMemcpyDeviceToHost), "top-K input integrity");
    require(unchanged == input, "top-K mutated input");
    out << name << ',' << rows << ',' << vocab << ',' << nan_rows
        << ",PASS,PASS,PASS,PASS,PASS\n";
    out.flush();
    if (timing) {
        cudaEvent_t a = nullptr, b = nullptr;
        cuda_check(cudaEventCreate(&a), "top-K event create");
        cuda_check(cudaEventCreate(&b), "top-K event create");
        for (int repeat = -3; repeat < 21; ++repeat) {
            for (const int mode : {0, 1, 2}) {
                const bool parallel = mode == 1;
                const bool local_merge = mode == 2;
                cuda_check(cudaEventRecord(a), "top-K timer begin");
                ninfer::exl3::dflash2_topk16_for_test(
                    ip, rows, vocab, idp, vp, parallel, local_merge);
                cuda_check(cudaEventRecord(b), "top-K timer end");
                cuda_check(cudaEventSynchronize(b), "top-K timer sync");
                float ms = 0;
                cuda_check(cudaEventElapsedTime(&ms, a, b), "top-K timer elapsed");
                if (repeat >= 0) *timing << name << ',' << repeat << ',' << mode << ',' << ms * 1000 << '\n';
            }
        }
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    // Invalid/empty entry points must leave both output buffers untouched.
    for (const int mode : {0, 1, 2}) {
        const bool parallel = mode == 1;
        const bool local_merge = mode == 2;
        cuda_check(cudaMemcpy(id_base, id_initial.data(), ids.bytes(), cudaMemcpyHostToDevice), "top-K invalid ID guards");
        cuda_check(cudaMemcpy(value_base, value_initial.data(), values.bytes(), cudaMemcpyHostToDevice), "top-K invalid value guards");
        ninfer::exl3::dflash2_topk16_for_test(
            nullptr, 0, vocab, idp, vp, parallel, local_merge);
        for (const int bad_rows : {-1, 8}) {
            bool rejected = false;
            try {
                ninfer::exl3::dflash2_topk16_for_test(
                    ip, bad_rows, vocab, idp, vp, parallel, local_merge);
            }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "top-K accepted invalid rows");
        }
        bool rejected = false;
        try {
            ninfer::exl3::dflash2_topk16_for_test(
                ip, 1, 15, idp, vp, parallel, local_merge);
        }
        catch (const std::exception&) { rejected = true; }
        require(rejected, "top-K accepted short vocabulary");
        std::vector<std::int64_t> after_ids(id_initial.size());
        std::vector<std::uint32_t> after_values(value_initial.size());
        cuda_check(cudaMemcpy(after_ids.data(), id_base, ids.bytes(), cudaMemcpyDeviceToHost), "top-K invalid IDs");
        cuda_check(cudaMemcpy(after_values.data(), value_base, values.bytes(), cudaMemcpyDeviceToHost), "top-K invalid values");
        require(after_ids == id_initial && after_values == value_initial, "top-K rejected entry mutated output");
    }
}

void run_topk_unit_qualification(const std::string& path) {
    require(!path.empty(), "top-K qualification output required");
    std::ofstream out(path), timing(path + ".timing.csv");
    require(out.good() && timing.good(), "top-K qualification output open");
    out << "case,rows,vocab,nan_rows,cpu_ids,cpu_values,gpu_exact,repeat,guards_input\n";
    timing << "case,repeat,mode,gpu_us\n";
    int cases = 0;
    for (const int vocab : {16, 17, 31, 255, 257, 248319, 248320, 248321}) {
        for (int rows = 1; rows <= 7; ++rows) {
            if (vocab != kVocab && rows != 1 && rows != 7) continue;
            for (int pattern = 0; pattern < 8; ++pattern) {
                std::vector<std::uint16_t> input(static_cast<std::size_t>(rows) * vocab);
                std::uint32_t state = 0x12345678U;
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < vocab; ++c) {
                        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                        std::uint16_t bits = static_cast<std::uint16_t>((state & 0x8000U) | (state % 0x7c00U));
                        if (pattern == 1) bits = 0x3c00;
                        if (pattern == 2) bits = (c & 1) ? 0x8000 : 0;
                        if (pattern == 3) bits = (c % 17 == 0) ? 0x7c00 : 0xfc00;
                        if (pattern == 4) bits = static_cast<std::uint16_t>(0x3c00 + c % 3);
                        if (pattern == 5 && (c == 0 || c == 7 || c == vocab - 1)) bits = c == 7 ? 0xfc11 : 0x7e11;
                        if (pattern == 6) bits = static_cast<std::uint16_t>(0x7c01U | (state & 0x83ffU));
                        if (pattern == 7) bits = c >= vocab - 2 ? 0x4000 : 0xbc00;
                        input[static_cast<std::size_t>(r) * vocab + c] = bits;
                    }
                }
                const std::string name = "synthetic-p" + std::to_string(pattern) + "-r" + std::to_string(rows) + "-v" + std::to_string(vocab);
                check_topk_case(input, rows, vocab, name, out,
                    vocab == kVocab && rows == 7 && pattern == 0 ? &timing : nullptr);
                ++cases;
            }
        }
    }
    require(cases == 168 && out.good() && timing.good(), "top-K qualification count/write");
    std::cout << "TOPKQUAL PASS cases=" << cases << " cpu_ids=1 cpu_values=1 gpu_exact=1 repeat=1 guards=1 input_unchanged=1\n";
}

void check_real_topk(Exl3Dflash2DraftModel& draft, int rows, const std::string& directory) {
    const std::filesystem::path dir(directory);
    require(!std::filesystem::exists(dir) && std::filesystem::create_directories(dir), "top-K capture path must be new");
    std::vector<std::uint16_t> input(static_cast<std::size_t>(rows) * kVocab);
    std::vector<std::int64_t> ids(static_cast<std::size_t>(rows) * 16);
    std::vector<std::uint32_t> values(ids.size());
    cuda_check(cudaMemcpy(input.data(), draft.last_head_logits_device_for_test(), input.size() * 2, cudaMemcpyDeviceToHost), "top-K real logits");
    require(std::all_of(input.begin(), input.end(), [](auto bits) {
                return std::isfinite(half_to_float(bits)); }), "real top-K logits must be finite");
    cuda_check(cudaMemcpy(ids.data(), draft.last_topk_ids_device_for_test(), ids.size() * 8, cudaMemcpyDeviceToHost), "top-K actual selector IDs");
    cuda_check(cudaMemcpy(values.data(), draft.last_topk_values_device_for_test(), values.size() * 4, cudaMemcpyDeviceToHost), "top-K actual selector values");
    for (int r = 0; r < rows; ++r) {
        const auto* row = input.data() + static_cast<std::size_t>(r) * kVocab;
        const auto expected = topk_cpu_ids(row, kVocab);
        for (int k = 0; k < 16; ++k) {
            const auto i = static_cast<std::size_t>(r) * 16 + k;
            require(ids[i] == expected[k] && values[i] == topk_float_bits(half_to_float(row[expected[k]])), "real production selector differs from CPU reference");
        }
    }
    std::ofstream raw(dir / "logits.f16", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(input.data()), static_cast<std::streamsize>(input.size() * 2));
    require(raw.good(), "top-K raw capture write");
    std::ofstream out(dir / "cases.csv"), timing(dir / "timing.csv");
    out << "case,rows,vocab,nan_rows,cpu_ids,cpu_values,gpu_exact,repeat,guards_input\n";
    timing << "case,repeat,parallel,gpu_us\n";
    check_topk_case(input, rows, kVocab, "real", out, &timing);
    require(out.good() && timing.good(), "top-K real evidence write");
    std::cout << "TOPKREAL PASS rows=" << rows << " vocab=" << kVocab << " cpu_selector=1 gpu_exact=1\n";
}
