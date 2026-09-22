#pragma once

struct T69ProjectionCaptureItem {
    int layer = -1;
    std::string operation;
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    std::unique_ptr<DeviceBuffer> input;
};

struct T69ProjectionCapture {
    std::array<T69ProjectionCaptureItem, 2> items;

    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& observation,
        void* user) {
        static_cast<T69ProjectionCapture*>(user)->observe(observation);
    }

    void observe(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if (x.rows != 1024 || x.metadata.mcg || !x.metadata.mul1 ||
            x.metadata.has_bias) return;
        int slot = -1;
        if (x.metadata.K == 6 && x.metadata.in_features == 5120 &&
            x.metadata.out_features == 17408 &&
            (x.operation == "gate" || x.operation == "up")) slot = 0;
        if (x.metadata.K == 7 && x.metadata.in_features == 17408 &&
            x.metadata.out_features == 5120 && x.operation == "down") slot = 1;
        if (slot < 0 || items[slot].input) return;
        const std::size_t bytes = static_cast<std::size_t>(x.rows) *
            x.metadata.in_features * sizeof(std::uint16_t);
        items[slot].input = std::make_unique<DeviceBuffer>(bytes);
        cuda_check(cudaMemcpyAsync(items[slot].input->get(), x.input, bytes,
                                   cudaMemcpyDeviceToDevice, x.stream),
                   "T69 capture real projection input");
        cuda_check(cudaStreamSynchronize(x.stream),
                   "T69 preserve captured input lifetime");
        items[slot].layer = x.layer;
        items[slot].operation = x.operation;
        items[slot].weights = x.weights;
        items[slot].metadata = x.metadata;
    }
};

struct T69ErrorMetric {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
    std::size_t values = 0;
};

T69ErrorMetric t69_compare_fp16(const std::vector<std::uint16_t>& actual,
                                const std::vector<std::uint16_t>& reference,
                                std::size_t elements) {
    require(actual.size() >= elements && reference.size() >= elements,
            "T69 compare extent");
    double error_sq = 0.0, norm_sq = 0.0, max_abs = 0.0;
    for (std::size_t i = 0; i < elements; ++i) {
        const double a = half_to_float(actual[i]);
        const double b = half_to_float(reference[i]);
        require(std::isfinite(a) && std::isfinite(b),
                "T69 nonfinite projection output");
        const double d = a - b;
        error_sq += d * d;
        norm_sq += b * b;
        max_abs = std::max(max_abs, std::abs(d));
    }
    return {max_abs, norm_sq > 0.0 ? std::sqrt(error_sq / norm_sq) : 0.0,
            elements};
}

std::uint16_t t69_decode_state_bitwise(const std::uint16_t* packed,
                                       int bits, int encoded) {
    const int tile_bits = 256 * bits;
    const int first = (encoded + 1) * bits - 16 + tile_bits;
    std::uint16_t state = 0;
    for (int offset = 0; offset < 16; ++offset) {
        const int bit = (first + offset) % tile_bits;
        const std::uint32_t word = h6_load_u32(packed, bit / 32);
        state = static_cast<std::uint16_t>(
            (static_cast<std::uint32_t>(state) << 1U) |
            ((word >> (31 - bit % 32)) & 1U));
    }
    return state;
}

struct T69OracleMetrics {
    T69ErrorMetric direct;
    T69ErrorMetric candidate;
};

T69OracleMetrics t69_check_fp64_oracle(
    const T69ProjectionCaptureItem& site,
    const std::vector<std::uint16_t>& direct,
    const std::vector<std::uint16_t>& candidate,
    int rows) {
    const int ni = site.metadata.in_features;
    const int no = site.metadata.out_features;
    const int bits = site.metadata.K;
    require(rows == 1024 && (bits == 6 || bits == 7),
            "T69 FP64 oracle scope");
    std::vector<std::uint16_t> suh(ni), svh(no);
    std::uint32_t multiplier = 0;
    cuda_check(cudaMemcpy(suh.data(), site.weights.suh, suh.size() * 2,
                          cudaMemcpyDeviceToHost), "T69 FP64 copy suh");
    cuda_check(cudaMemcpy(svh.data(), site.weights.svh, svh.size() * 2,
                          cudaMemcpyDeviceToHost), "T69 FP64 copy svh");
    cuda_check(cudaMemcpy(&multiplier, site.weights.mul1, sizeof(multiplier),
                          cudaMemcpyDeviceToHost), "T69 FP64 copy mul1");

    const std::array<int, 3> sample_rows = {0, rows / 2, rows - 1};
    const int groups = no / 128;
    const std::array<int, 3> sample_groups = {0, groups / 2, groups - 1};
    std::array<std::vector<std::uint16_t>, 3> raw;
    std::array<std::vector<double>, 3> transformed;
    constexpr double scale = 1.0 / 11.3137084989847603904135097936775846;
    for (int r = 0; r < 3; ++r) {
        raw[r].resize(ni);
        cuda_check(cudaMemcpy(raw[r].data(),
                              static_cast<const std::uint16_t*>(site.input->get()) +
                                  static_cast<std::size_t>(sample_rows[r]) * ni,
                              raw[r].size() * 2, cudaMemcpyDeviceToHost),
                   "T69 FP64 copy input row");
        transformed[r].resize(ni);
        for (int block = 0; block < ni; block += 128) {
            std::array<double, 128> values{};
            for (int i = 0; i < 128; ++i)
                values[i] = static_cast<double>(half_to_float(raw[r][block + i])) *
                    static_cast<double>(half_to_float(suh[block + i]));
            h6_hadamard(values);
            for (int i = 0; i < 128; ++i)
                transformed[r][block + i] = values[i] * scale;
        }
    }

    double direct_error_sq = 0.0, candidate_error_sq = 0.0, norm_sq = 0.0;
    double direct_max = 0.0, candidate_max = 0.0;
    std::size_t values_checked = 0;
    const int input_tiles = ni / 16;
    const int output_tiles = no / 16;
    const std::size_t packed_tile_words = static_cast<std::size_t>(16 * bits);
    const std::size_t group_words = 8 * packed_tile_words;
    for (const int group : sample_groups) {
        std::vector<std::uint16_t> packed(
            static_cast<std::size_t>(input_tiles) * group_words);
        const auto* source = site.weights.trellis +
            static_cast<std::size_t>(group * 8) * packed_tile_words;
        cuda_check(cudaMemcpy2D(
            packed.data(), group_words * sizeof(std::uint16_t),
            source, static_cast<std::size_t>(output_tiles) * packed_tile_words *
                        sizeof(std::uint16_t),
            group_words * sizeof(std::uint16_t), input_tiles,
            cudaMemcpyDeviceToHost), "T69 FP64 copy trellis group");
        for (int sample = 0; sample < 3; ++sample) {
            std::array<double, 128> accum{};
            for (int tile_k = 0; tile_k < input_tiles; ++tile_k) {
                for (int ir = 0; ir < 16; ++ir) {
                    const double x = transformed[sample][tile_k * 16 + ir];
                    for (int oc = 0; oc < 128; ++oc) {
                        const int tile_n = oc / 16;
                        const int column = oc % 16;
                        const auto* tile = packed.data() +
                            (static_cast<std::size_t>(tile_k) * 8 + tile_n) *
                                packed_tile_words;
                        const int encoded = h6_trellis_index(ir, column);
                        const auto state = t69_decode_state_bitwise(
                            tile, bits, encoded);
                        const auto weight = h6_decode_mul1(state, multiplier);
                        accum[oc] += x * static_cast<double>(half_to_float(weight));
                    }
                }
            }
            h6_hadamard(accum);
            for (int oc = 0; oc < 128; ++oc) {
                const int column = group * 128 + oc;
                const double oracle = accum[oc] * scale *
                    static_cast<double>(half_to_float(svh[column]));
                const std::size_t index =
                    static_cast<std::size_t>(sample_rows[sample]) * no + column;
                const double d = static_cast<double>(half_to_float(direct[index])) - oracle;
                const double c = static_cast<double>(half_to_float(candidate[index])) - oracle;
                direct_error_sq += d * d;
                candidate_error_sq += c * c;
                norm_sq += oracle * oracle;
                direct_max = std::max(direct_max, std::abs(d));
                candidate_max = std::max(candidate_max, std::abs(c));
                ++values_checked;
            }
        }
    }
    const double denominator = norm_sq > 0.0 ? norm_sq : 1.0;
    return {{direct_max, std::sqrt(direct_error_sq / denominator), values_checked},
            {candidate_max, std::sqrt(candidate_error_sq / denominator),
             values_checked}};
}

void run_projection_reconstruct_gemm_t69(
    Exl3TextModel& target, const std::vector<std::int64_t>& source) {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    using Candidate = ninfer::exl3::Exl3CudaReconstructGemmWorkspace;
    using Phase = ninfer::exl3::Exl3ReconstructGemmPhaseTiming;
    for(int rows:{1,16,128,1024}) {
        const auto weights=5120ull*17408*2;
        const auto scratch=16ull*1024*1024;
        require(Candidate::workspace_bytes_required(5120,17408,rows,false)==
            weights+scratch+rows*(5120ull*2+17408ull*4),"T69 independent forward requirement");
        require(Candidate::workspace_bytes_required(17408,5120,rows,false)==
            weights+scratch+rows*(17408ull*2+5120ull*4),"T69 independent reverse requirement");
        for(bool reverse:{false,true})
            require(Candidate::workspace_bytes_required(reverse?17408:5120,reverse?5120:17408,rows,true)==
                weights+scratch+rows*17408ull*6,"T69 transpose capacity requirement");
    }
    for(const auto shape:std::array<std::array<int,3>,4>{{{0,128,1},{128,-1,1},{129,128,1},{128,128,0}}}) {
        bool refused=false;
        try{(void)Candidate::workspace_bytes_required(shape[0],shape[1],shape[2],false);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"T69 invalid requirement shape accepted");
    }
    bool overflow_refused=false;
    try{(void)Candidate::workspace_bytes_required(2147483520,2147483520,2147483647,true);}
    catch(const std::overflow_error&){overflow_refused=true;}
    require(overflow_refused,"T69 complete allocation overflow accepted");
    require(target.max_context() >= 1040 && source.size() >= 1040,
            "T69 source/context extent");
    for (const char* key : {"NINFER_EXL3_WIDE_PREFILL",
                            "NINFER_EXL3_PREFILL_STAGED_REDUCTION",
                            "NINFER_EXL3_PREFILL_WIDE1024",
                            "NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
                            "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A"})
        require(env(key) == "1", std::string("T69 current direct flag ") + key);

    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    auto context = target.create_context(true);
    context->prefill(std::span<const std::int64_t>(source.data(), 16));
    T69ProjectionCapture capture;
    context->set_target_projection_observer_for_test(
        T69ProjectionCapture::callback, &capture, nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(
        std::span<const std::int64_t>(source.data() + 16, 1024));
    cuda_check(cudaDeviceSynchronize(), "T69 capture complete");
    context->set_target_projection_observer_for_test(nullptr);
    require(capture.items[0].input && capture.items[1].input,
            "T69 missing K6/K7 real input");

    const std::filesystem::path directory = env("NINFER_T69_OUT");
    require(!directory.empty() && !std::filesystem::exists(directory),
            "T69 output directory must be new");
    std::filesystem::create_directories(directory);
    std::ofstream summary(directory / "summary.csv");
    std::ofstream pairs(directory / "pairs.csv");
    require(summary.good() && pairs.good(), "T69 output creation");
    summary << "site,layer,operation,K,input,output,rows,direct_dispatch,workspace_bytes,"
               "direct_candidate_max_abs,direct_candidate_rel_l2,input_us,reconstruct_us,"
               "gemm_us,output_us,candidate_total_us,oracle_values,direct_oracle_max_abs,"
               "direct_oracle_rel_l2,candidate_oracle_max_abs,candidate_oracle_rel_l2,numeric_gate\n";
    pairs << "site,rows,pair,order,repetitions,direct_us,candidate_us,gain_pct\n";
    bool all_numeric = true;
    for (int slot = 0; slot < 2; ++slot) {
        const auto& site = capture.items[slot];
        const int ni = site.metadata.in_features, no = site.metadata.out_features;
        const std::string name = site.metadata.K == 6 ? "k6_gateup" : "k7_down";
        Exl3CudaLinearWorkspace direct_ws(
            ni, no, 1024, false, false, false, false, false, true);
        Candidate candidate_ws(ni, no, 1024);
        require(candidate_ws.workspace_bytes()==static_cast<std::size_t>(ni)*no*2+
            1024ull*(static_cast<std::size_t>(ni)*2+static_cast<std::size_t>(no)*4)+(16ull<<20),
            "T69 independent actual workspace ledger");
        constexpr std::size_t guard = 128;
        const std::size_t maximum = 1024ull * no;
        std::vector<std::uint16_t> poison(maximum + 2 * guard, 0x3555);
        DeviceBuffer direct_out(poison.size() * 2), candidate_out(poison.size() * 2);
        for (const int rows : {256, 512, 1024}) {
            const std::size_t active = static_cast<std::size_t>(rows) * no;
            cuda_check(cudaMemcpy(direct_out.get(), poison.data(), poison.size() * 2,
                                  cudaMemcpyHostToDevice), "T69 poison direct");
            cuda_check(cudaMemcpy(candidate_out.get(), poison.data(), poison.size() * 2,
                                  cudaMemcpyHostToDevice), "T69 poison candidate");
            direct_ws.forward(site.weights, site.metadata,
                static_cast<const std::uint16_t*>(site.input->get()),
                static_cast<std::uint16_t*>(direct_out.get()) + guard, rows, nullptr,
                Admission::target_wide_prefill);
            candidate_ws.forward_numeric_candidate(site.weights, site.metadata,
                static_cast<const std::uint16_t*>(site.input->get()),
                static_cast<std::uint16_t*>(candidate_out.get()) + guard, rows);
            cuda_check(cudaDeviceSynchronize(), "T69 correctness complete");
            std::vector<std::uint16_t> direct_host(poison.size()),
                candidate_host(poison.size());
            cuda_check(cudaMemcpy(direct_host.data(), direct_out.get(), poison.size() * 2,
                                  cudaMemcpyDeviceToHost), "T69 copy direct");
            cuda_check(cudaMemcpy(candidate_host.data(), candidate_out.get(), poison.size() * 2,
                                  cudaMemcpyDeviceToHost), "T69 copy candidate");
            for (std::size_t i = 0; i < guard; ++i) {
                require(direct_host[i] == 0x3555 && candidate_host[i] == 0x3555 &&
                        direct_host[guard + active + i] == 0x3555 &&
                        candidate_host[guard + active + i] == 0x3555,
                        "T69 output canary");
            }
            std::vector<std::uint16_t> direct_active(
                direct_host.begin() + guard, direct_host.begin() + guard + active);
            std::vector<std::uint16_t> candidate_active(
                candidate_host.begin() + guard,
                candidate_host.begin() + guard + active);
            const auto parity = t69_compare_fp16(candidate_active, direct_active, active);
            T69OracleMetrics oracle{};
            bool numeric_gate = true;
            if (rows == 1024) {
                oracle = t69_check_fp64_oracle(site, direct_active, candidate_active, rows);
                numeric_gate = oracle.candidate.relative_l2 <= 0.003 &&
                    oracle.candidate.max_abs <= 0.10 &&
                    oracle.candidate.relative_l2 <= oracle.direct.relative_l2 + 0.0005 &&
                    oracle.candidate.max_abs <= oracle.direct.max_abs + 0.02;
                all_numeric = all_numeric && numeric_gate;
            }

            direct_ws.forward(site.weights, site.metadata,
                static_cast<const std::uint16_t*>(site.input->get()),
                static_cast<std::uint16_t*>(direct_out.get()) + guard, rows, nullptr,
                Admission::target_wide_prefill);
            candidate_ws.forward_numeric_candidate(site.weights, site.metadata,
                static_cast<const std::uint16_t*>(site.input->get()),
                static_cast<std::uint16_t*>(candidate_out.get()) + guard, rows);
            cuda_check(cudaDeviceSynchronize(), "T69 warmup complete");
            Phase phase{};
            candidate_ws.forward_numeric_candidate(site.weights, site.metadata,
                static_cast<const std::uint16_t*>(site.input->get()),
                static_cast<std::uint16_t*>(candidate_out.get()) + guard, rows, nullptr,
                &phase);

            cudaEvent_t begin = nullptr, end = nullptr;
            cuda_check(cudaEventCreate(&begin), "T69 timing begin create");
            cuda_check(cudaEventCreate(&end), "T69 timing end create");
            constexpr int repetitions = 3;
            auto measure = [&](bool candidate_path) {
                cuda_check(cudaEventRecord(begin), "T69 timing begin");
                for (int repeat = 0; repeat < repetitions; ++repeat) {
                    if (candidate_path)
                        candidate_ws.forward_numeric_candidate(
                            site.weights, site.metadata,
                            static_cast<const std::uint16_t*>(site.input->get()),
                            static_cast<std::uint16_t*>(candidate_out.get()) + guard,
                            rows);
                    else
                        direct_ws.forward(
                            site.weights, site.metadata,
                            static_cast<const std::uint16_t*>(site.input->get()),
                            static_cast<std::uint16_t*>(direct_out.get()) + guard,
                            rows, nullptr, Admission::target_wide_prefill);
                }
                cuda_check(cudaEventRecord(end), "T69 timing end");
                cuda_check(cudaEventSynchronize(end), "T69 timing sync");
                float ms = 0.0f;
                cuda_check(cudaEventElapsedTime(&ms, begin, end),
                           "T69 timing resolve");
                return static_cast<double>(ms) * 1000.0 / repetitions;
            };
            for (int pair = 0; pair < 6; ++pair) {
                double direct_us = 0.0, candidate_us = 0.0;
                if ((pair & 1) == 0) {
                    direct_us = measure(false); candidate_us = measure(true);
                } else {
                    candidate_us = measure(true); direct_us = measure(false);
                }
                pairs << name << ',' << rows << ',' << pair << ','
                      << ((pair & 1) == 0 ? "AB" : "BA") << ',' << repetitions
                      << ',' << direct_us << ',' << candidate_us << ','
                      << (direct_us - candidate_us) * 100.0 / direct_us << '\n';
            }
            cuda_check(cudaEventDestroy(begin), "T69 timing begin destroy");
            cuda_check(cudaEventDestroy(end), "T69 timing end destroy");
            summary << name << ',' << site.layer << ',' << site.operation << ','
                    << site.metadata.K << ',' << ni << ',' << no << ',' << rows
                    << ',' << direct_ws.dispatch_name(
                           site.metadata, rows, Admission::target_wide_prefill)
                    << ',' << candidate_ws.workspace_bytes() << ','
                    << parity.max_abs << ',' << parity.relative_l2 << ','
                    << phase.input_transform_us << ',' << phase.reconstruct_us << ','
                    << phase.gemm_us << ',' << phase.output_transform_us << ','
                    << phase.total_us << ',' << oracle.candidate.values << ','
                    << oracle.direct.max_abs << ',' << oracle.direct.relative_l2 << ','
                    << oracle.candidate.max_abs << ','
                    << oracle.candidate.relative_l2 << ','
                    << (numeric_gate ? 1 : 0) << '\n';
            summary.flush(); pairs.flush();
            require(summary.good() && pairs.good(), "T69 evidence write");
        }
    }
    require(all_numeric, "T69 predeclared FP64 numeric gate failed");
    std::cout << "T69_RECONSTRUCT_GEMM PASS sites=2 rows=256,512,1024 "
                 "lane=NUMERIC_CANDIDATE fp64_gate=1 production_changed=0"
              << std::endl;
}
