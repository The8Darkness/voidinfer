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
    require(rows == 1024 && (bits == 5 || bits == 6 || bits == 7),
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

// Reached K5 GDN MLP projections have a different packed representation from
// the earlier K6/K7 T69 sites. Capture their real inputs, then compare the
// proposed transformed-basis reconstruction against both the existing wide
// projection and a bitwise packed-state FP64 oracle.
struct K5PrefillCapture {
    std::array<T69ProjectionCaptureItem,3> items;
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,
                         void* user) {
        static_cast<K5PrefillCapture*>(user)->observe(x);
    }
    void observe(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if(x.rows!=1024 || x.layer!=0 || x.metadata.K!=5 ||
           x.metadata.mcg || !x.metadata.mul1 || x.metadata.has_bias) return;
        const int slot=x.operation=="gate"?0:x.operation=="up"?1:
                       x.operation=="down"?2:-1;
        if(slot<0 || items[slot].input) return;
        auto& item=items[slot];
        item.input=std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(x.rows)*x.metadata.in_features*2);
        cuda_check(cudaMemcpyAsync(item.input->get(),x.input,
            static_cast<std::size_t>(x.rows)*x.metadata.in_features*2,
            cudaMemcpyDeviceToDevice,x.stream),"K5 capture projection input");
        cuda_check(cudaStreamSynchronize(x.stream),"K5 capture input lifetime");
        item.layer=x.layer;item.operation=x.operation;
        item.weights=x.weights;item.metadata=x.metadata;
    }
};

std::uint16_t t69_float_to_half_rn(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t absolute = bits & 0x7fffffffu;
    if (absolute >= 0x7f800000u)
        return static_cast<std::uint16_t>(sign |
            (absolute > 0x7f800000u ? 0x7e00u : 0x7c00u));
    int exponent = static_cast<int>((absolute >> 23u) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = absolute & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const unsigned shift = static_cast<unsigned>(14 - exponent);
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway ||
            (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    std::uint32_t rounded = mantissa >> 13u;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        if (++rounded == 0x400u) {
            rounded = 0;
            if (++exponent >= 31)
                return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign |
        (static_cast<unsigned>(exponent) << 10u) | rounded);
}

void run_k5_prefill_operator(Exl3TextModel& target,
                             const std::vector<std::int64_t>& source) {
    using Candidate=ninfer::exl3::Exl3CudaReconstructGemmWorkspace;
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    require(source.size()>=1040 && target.max_context()>=1040,
            "K5 prefill source capacity");
    require(env("NINFER_EXL3_FAST_LAYER_MAJOR_K5_RECONSTRUCT")=="1",
            "K5 prefill test candidate flag");
    auto context=target.create_context(true);
    context->prefill(std::span<const std::int64_t>(source.data(),16));
    K5PrefillCapture capture;
    context->set_target_projection_observer_for_test(
        K5PrefillCapture::callback,&capture,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_prefill_wide(
        std::span<const std::int64_t>(source.data()+16,1024));
    cuda_check(cudaDeviceSynchronize(),"K5 capture complete");
    context->set_target_projection_observer_for_test(nullptr);
    for(const auto& item:capture.items)
        require(item.input && item.metadata.K==5,
                "missing reached K5 GDN projection");
    require(capture.items[0].weights.trellis!=capture.items[1].weights.trellis,
            "K5 gate/up weight identities aliased");

    Candidate gate_up(5120,17408,1024,false,true);
    Candidate down(17408,5120,1024,false,true);
    gate_up.begin_layer_reuse(512ull*1024*1024,true);
    down.begin_layer_reuse(256ull*1024*1024,true);
    for(int site_index=0;site_index<3;++site_index) {
        const auto& site=capture.items[site_index];
        const int ni=site.metadata.in_features,no=site.metadata.out_features;
        require((site_index<2 && ni==5120 && no==17408) ||
                (site_index==2 && ni==17408 && no==5120),
                "reached K5 shape changed");
        auto& candidate=site_index<2?gate_up:down;
        require(candidate.supports(site.metadata,1024),
                "K5 candidate refused reached shape");
        Exl3CudaLinearWorkspace direct(ni,no,1024,false,false,false,false,
                                       false,true);
        constexpr std::size_t guard=128;
        const std::size_t maximum=1024ull*no;
        const std::vector<std::uint16_t> poison(maximum+2*guard,0x3555);
        DeviceBuffer direct_out(poison.size()*2),candidate_out(poison.size()*2);
        for(int rows:{256,1024}) {
            cuda_check(cudaMemcpy(direct_out.get(),poison.data(),poison.size()*2,
                                  cudaMemcpyHostToDevice),"K5 poison direct");
            cuda_check(cudaMemcpy(candidate_out.get(),poison.data(),poison.size()*2,
                                  cudaMemcpyHostToDevice),"K5 poison candidate");
            direct.forward(site.weights,site.metadata,
                static_cast<const std::uint16_t*>(site.input->get()),
                static_cast<std::uint16_t*>(direct_out.get())+guard,rows,nullptr,
                Admission::target_wide_prefill);
            candidate.forward_numeric_candidate(site.weights,site.metadata,
                static_cast<const std::uint16_t*>(site.input->get()),
                static_cast<std::uint16_t*>(candidate_out.get())+guard,rows);
            cuda_check(cudaDeviceSynchronize(),"K5 differential complete");
            std::vector<std::uint16_t> direct_host(poison.size()),
                candidate_host(poison.size());
            cuda_check(cudaMemcpy(direct_host.data(),direct_out.get(),
                poison.size()*2,cudaMemcpyDeviceToHost),"K5 copy direct");
            cuda_check(cudaMemcpy(candidate_host.data(),candidate_out.get(),
                poison.size()*2,cudaMemcpyDeviceToHost),"K5 copy candidate");
            const auto active=static_cast<std::size_t>(rows)*no;
            for(std::size_t i=0;i<guard;++i)
                require(direct_host[i]==0x3555 && candidate_host[i]==0x3555 &&
                        direct_host[guard+active+i]==0x3555 &&
                        candidate_host[guard+active+i]==0x3555,
                        "K5 output canary");
            std::vector<std::uint16_t> direct_active(
                direct_host.begin()+guard,direct_host.begin()+guard+active);
            std::vector<std::uint16_t> candidate_active(
                candidate_host.begin()+guard,candidate_host.begin()+guard+active);
            const auto parity=t69_compare_fp16(candidate_active,direct_active,active);
            std::cout<<"K5_PREFILL_OPERATOR site="<<site.operation<<" rows="<<rows
                     <<" direct_candidate_max_abs="<<parity.max_abs
                     <<" direct_candidate_rel_l2="<<parity.relative_l2;
            if(rows==1024) {
                const auto oracle=t69_check_fp64_oracle(
                    site,direct_active,candidate_active,rows);
                std::cout<<" direct_oracle_rel_l2="<<oracle.direct.relative_l2
                         <<" candidate_oracle_rel_l2="<<oracle.candidate.relative_l2
                         <<" candidate_oracle_max_abs="<<oracle.candidate.max_abs;
                require(oracle.candidate.relative_l2<=0.003 &&
                        oracle.candidate.max_abs<=0.10 &&
                        oracle.candidate.relative_l2<=oracle.direct.relative_l2+0.0005 &&
                        oracle.candidate.max_abs<=oracle.direct.max_abs+0.02,
                        "K5 independent FP64 projection gate");
            }
            std::cout<<'\n';
        }
    }
    const auto reused=gate_up.stats();
    require(reused.reconstructed_weight_calls==2 &&
            reused.reused_weight_calls==2 &&
            reused.cached_weight_capacity_bytes>=2ull*5120*17408*2,
            "K5 gate/up distinct weight reuse");
    const char* lt=std::getenv("NINFER_EXL3_FAST_PREFILL_K5_CUBLAS_LT");
    if (lt && std::strcmp(lt,"1")==0)
        require(reused.k5_lt_calls==4 && down.stats().k5_lt_calls==2,
                "K5 cuBLASLt candidate did not dispatch");
    gate_up.end_layer_reuse();down.end_layer_reuse();
    // Reached layer-0 K5 gate/up/down composition. The ordinary two-launch
    // activation and down transform is the bitwise control; the captured K5
    // weights and the independent projection oracle above remain the math
    // reference for the down operator.
    Exl3CudaLinearWorkspace down_transform(17408,5120,1024,false,false,
                                            false,false,false,true);
    constexpr std::size_t fusion_guard=128;
    constexpr std::size_t fusion_middle=1024ull*17408;
    constexpr std::size_t fusion_output=1024ull*5120;
    const std::vector<std::uint16_t> middle_poison(
        fusion_middle+2*fusion_guard,0x3555);
    const std::vector<std::uint16_t> output_poison(
        fusion_output+2*fusion_guard,0x3555);
    DeviceBuffer gate_buffer(fusion_middle*2),up_buffer(fusion_middle*2);
    DeviceBuffer ordinary_act(middle_poison.size()*2);
    DeviceBuffer fused_act(middle_poison.size()*2);
    DeviceBuffer ordinary_down(output_poison.size()*2);
    DeviceBuffer fused_down(output_poison.size()*2);
    gate_up.begin_layer_reuse(512ull*1024*1024,true);
    down.begin_layer_reuse(256ull*1024*1024,true);
    for(const int rows:{256,1024}) {
        const auto& gate_site=capture.items[0];
        const auto& up_site=capture.items[1];
        const auto& down_site=capture.items[2];
        gate_up.forward_numeric_candidate(gate_site.weights,gate_site.metadata,
            static_cast<const std::uint16_t*>(gate_site.input->get()),
            static_cast<std::uint16_t*>(gate_buffer.get()),rows);
        gate_up.forward_numeric_candidate(up_site.weights,up_site.metadata,
            static_cast<const std::uint16_t*>(up_site.input->get()),
            static_cast<std::uint16_t*>(up_buffer.get()),rows);
        cuda_check(cudaMemcpy(ordinary_act.get(),middle_poison.data(),
            middle_poison.size()*2,cudaMemcpyHostToDevice),
            "K5 poison ordinary activation");
        cuda_check(cudaMemcpy(fused_act.get(),middle_poison.data(),
            middle_poison.size()*2,cudaMemcpyHostToDevice),
            "K5 poison fused activation");
        cuda_check(cudaMemcpy(ordinary_down.get(),output_poison.data(),
            output_poison.size()*2,cudaMemcpyHostToDevice),
            "K5 poison ordinary down");
        cuda_check(cudaMemcpy(fused_down.get(),output_poison.data(),
            output_poison.size()*2,cudaMemcpyHostToDevice),
            "K5 poison fused down");
        auto* ordinary_act_data=static_cast<std::uint16_t*>(ordinary_act.get())+
            fusion_guard;
        auto* fused_act_data=static_cast<std::uint16_t*>(fused_act.get())+
            fusion_guard;
        auto* ordinary_down_data=static_cast<std::uint16_t*>(ordinary_down.get())+
            fusion_guard;
        auto* fused_down_data=static_cast<std::uint16_t*>(fused_down.get())+
            fusion_guard;
        down_transform.transform_gate_up(down_site.weights,down_site.metadata,
            static_cast<const std::uint16_t*>(gate_buffer.get()),
            static_cast<const std::uint16_t*>(up_buffer.get()),
            ordinary_act_data,rows,nullptr,Admission::target_wide_prefill);
        down.forward_numeric_candidate(down_site.weights,down_site.metadata,
            ordinary_act_data,ordinary_down_data,rows);
        down.forward_numeric_gate_up_down(down_site.weights,down_site.metadata,
            static_cast<const std::uint16_t*>(gate_buffer.get()),
            static_cast<const std::uint16_t*>(up_buffer.get()),
            fused_act_data,fused_down_data,rows);
        cuda_check(cudaDeviceSynchronize(),"K5 fused down composition complete");
        std::vector<std::uint16_t> ordinary_act_host(middle_poison.size());
        std::vector<std::uint16_t> fused_act_host(middle_poison.size());
        std::vector<std::uint16_t> ordinary_down_host(output_poison.size());
        std::vector<std::uint16_t> fused_down_host(output_poison.size());
        cuda_check(cudaMemcpy(ordinary_act_host.data(),ordinary_act.get(),
            middle_poison.size()*2,cudaMemcpyDeviceToHost),
            "K5 copy ordinary activation");
        cuda_check(cudaMemcpy(fused_act_host.data(),fused_act.get(),
            middle_poison.size()*2,cudaMemcpyDeviceToHost),
            "K5 copy fused activation");
        cuda_check(cudaMemcpy(ordinary_down_host.data(),ordinary_down.get(),
            output_poison.size()*2,cudaMemcpyDeviceToHost),
            "K5 copy ordinary down");
        cuda_check(cudaMemcpy(fused_down_host.data(),fused_down.get(),
            output_poison.size()*2,cudaMemcpyDeviceToHost),
            "K5 copy fused down");
        const auto active_middle=static_cast<std::size_t>(rows)*17408;
        const auto active_output=static_cast<std::size_t>(rows)*5120;
        require(ordinary_act_host==fused_act_host &&
                ordinary_down_host==fused_down_host,
                "K5 fused down changed activation, down output or canary");
        for(const std::size_t i:{std::size_t{0},active_middle/2,
                                 active_middle-1}) {
            const float g=half_to_float(
                [&] { std::uint16_t value=0;
                    cuda_check(cudaMemcpy(&value,
                        static_cast<const std::uint16_t*>(gate_buffer.get())+i,
                        sizeof(value),cudaMemcpyDeviceToHost),
                        "K5 sample gate"); return value; }());
            const float u=half_to_float(
                [&] { std::uint16_t value=0;
                    cuda_check(cudaMemcpy(&value,
                        static_cast<const std::uint16_t*>(up_buffer.get())+i,
                        sizeof(value),cudaMemcpyDeviceToHost),
                        "K5 sample up"); return value; }());
            const double expected=static_cast<double>(g)*u/
                (1.0+std::exp(-static_cast<double>(g)));
            const double actual=half_to_float(
                fused_act_host[fusion_guard+i]);
            require(std::isfinite(actual) &&
                    std::abs(actual-expected)<=0.003*std::max(1.0,std::abs(expected)),
                    "K5 fused down activation oracle mismatch");
        }
        require(down.stats().fused_gate_up_down_calls>0,
                "K5 fused down did not dispatch");
        std::cout<<"K5_FUSED_DOWN_OPERATOR rows="<<rows
                 <<" active_middle="<<active_middle
                 <<" active_output="<<active_output<<" bitwise=1 canary=1\n";
    }
    gate_up.end_layer_reuse();down.end_layer_reuse();
    // The reached K5 down owner aliases down with final layer output. Check
    // the fused epilogue against a separate projection plus represented FP32
    // residual add, including the requested last-row down trace and canaries.
    const auto& down_site=capture.items[2];
    const auto& residual_site=capture.items[0];
    constexpr std::size_t residual_guard=128;
    constexpr int traced_rows=17;
    const std::vector<std::uint16_t> residual_poison(
        1024ull*5120+2*residual_guard,0x3555);
    const std::vector<std::uint16_t> trace_poison(
        traced_rows*5120+2*residual_guard,0x3555);
    DeviceBuffer ordinary_residual_down(residual_poison.size()*2);
    DeviceBuffer fused_residual_output(residual_poison.size()*2);
    DeviceBuffer fused_residual_trace(trace_poison.size()*2);
    std::vector<std::uint16_t> residual_host(1024ull*5120);
    cuda_check(cudaMemcpy(residual_host.data(),residual_site.input->get(),
        residual_host.size()*2,cudaMemcpyDeviceToHost),
        "K5 copy reached residual test operand");
    down.begin_layer_reuse(256ull*1024*1024,true);
    for(const int rows:{256,1024}) {
        cuda_check(cudaMemcpy(ordinary_residual_down.get(),
            residual_poison.data(),residual_poison.size()*2,
            cudaMemcpyHostToDevice),"K5 poison separate down");
        cuda_check(cudaMemcpy(fused_residual_output.get(),
            residual_poison.data(),residual_poison.size()*2,
            cudaMemcpyHostToDevice),"K5 poison fused residual");
        cuda_check(cudaMemcpy(fused_residual_trace.get(),trace_poison.data(),
            trace_poison.size()*2,cudaMemcpyHostToDevice),
            "K5 poison fused residual trace");
        auto* ordinary_data=static_cast<std::uint16_t*>(
            ordinary_residual_down.get())+residual_guard;
        auto* fused_data=static_cast<std::uint16_t*>(
            fused_residual_output.get())+residual_guard;
        auto* trace_data=static_cast<std::uint16_t*>(
            fused_residual_trace.get())+residual_guard;
        down.forward_numeric_candidate(down_site.weights,down_site.metadata,
            static_cast<const std::uint16_t*>(down_site.input->get()),
            ordinary_data,rows);
        down.forward_numeric_candidate(down_site.weights,down_site.metadata,
            static_cast<const std::uint16_t*>(down_site.input->get()),
            fused_data,rows,nullptr,nullptr,nullptr,nullptr,
            static_cast<const std::uint16_t*>(residual_site.input->get()),
            trace_data,rows-traced_rows);
        cuda_check(cudaDeviceSynchronize(),"K5 fused residual epilogue complete");
        std::vector<std::uint16_t> ordinary(residual_poison.size());
        std::vector<std::uint16_t> fused(residual_poison.size());
        std::vector<std::uint16_t> trace(trace_poison.size());
        cuda_check(cudaMemcpy(ordinary.data(),ordinary_residual_down.get(),
            ordinary.size()*2,cudaMemcpyDeviceToHost),
            "K5 copy separate down");
        cuda_check(cudaMemcpy(fused.data(),fused_residual_output.get(),
            fused.size()*2,cudaMemcpyDeviceToHost),
            "K5 copy fused residual");
        cuda_check(cudaMemcpy(trace.data(),fused_residual_trace.get(),
            trace.size()*2,cudaMemcpyDeviceToHost),
            "K5 copy fused residual trace");
        const auto count=static_cast<std::size_t>(rows)*5120;
        require(std::all_of(ordinary.begin(),
                ordinary.begin()+residual_guard,
                [](std::uint16_t x){return x==0x3555;}) &&
            std::all_of(ordinary.begin()+residual_guard+count,
                ordinary.end(),[](std::uint16_t x){return x==0x3555;}) &&
            std::all_of(fused.begin(),fused.begin()+residual_guard,
                [](std::uint16_t x){return x==0x3555;}) &&
            std::all_of(fused.begin()+residual_guard+count,fused.end(),
                [](std::uint16_t x){return x==0x3555;}) &&
            std::all_of(trace.begin(),trace.begin()+residual_guard,
                [](std::uint16_t x){return x==0x3555;}) &&
            std::all_of(trace.begin()+residual_guard+traced_rows*5120,
                trace.end(),[](std::uint16_t x){return x==0x3555;}),
            "K5 fused residual guard changed");
        for(std::size_t i=0;i<count;++i) {
            const auto expected=t69_float_to_half_rn(
                half_to_float(ordinary[residual_guard+i])+
                half_to_float(residual_host[i]));
            require(fused[residual_guard+i]==expected,
                "K5 fused residual changed represented final value");
        }
        const auto final_offset=static_cast<std::size_t>(rows-traced_rows)*5120;
        for(std::size_t i=0;i<traced_rows*5120;++i)
            require(trace[residual_guard+i]==
                ordinary[residual_guard+final_offset+i],
                "K5 fused residual changed down trace");
        std::cout<<"K5_FUSED_RESIDUAL_OPERATOR rows="<<rows
                 <<" final_values="<<count<<" trace_rows="<<traced_rows
                 <<" bitwise=1 canary=1\n";
    }
    require(down.stats().fused_down_residual_calls==2 &&
            down.stats().fused_down_residual_rows==1280,
            "K5 fused residual did not dispatch at both row counts");
    down.end_layer_reuse();
    std::cout<<"K5_PREFILL_OPERATOR PASS sites=3 rows=256,1024"
             <<std::endl;
}

void run_k5_direct_bulk_operator(Exl3TextModel& target,
                                 const std::vector<std::int64_t>& source) {
    using Candidate=ninfer::exl3::Exl3CudaReconstructGemmWorkspace;
    require(source.size()>=1040 && target.max_context()>=1040,
            "K5 direct bulk source capacity");
    require(env("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL")=="1",
            "K5 direct bulk requires FP16 GEMM destination policy");
    _putenv_s("NINFER_EXL3_FAST_GDN_BULK_MLP_PACKED_K5","0");
    auto context=target.create_context(true);
    context->prefill(std::span<const std::int64_t>(source.data(),16));
    K5PrefillCapture capture;
    context->set_target_projection_observer_for_test(
        K5PrefillCapture::callback,&capture,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_prefill_wide(
        std::span<const std::int64_t>(source.data()+16,1024));
    cuda_check(cudaDeviceSynchronize(),"K5 direct bulk capture complete");
    context->set_target_projection_observer_for_test(nullptr);
    for(const auto& item:capture.items)
        require(item.input && item.metadata.K==5,
                "missing reached K5 direct bulk projection");
    require(capture.items[0].weights.trellis!=capture.items[1].weights.trellis &&
            capture.items[0].weights.suh!=capture.items[1].weights.suh &&
            capture.items[0].weights.svh!=capture.items[1].weights.svh,
            "K5 gate/up represented weights or transforms aliased");

    constexpr int rows=1024;
    constexpr std::size_t guard=128;
    for(int site_index=0;site_index<3;++site_index) {
        const auto& site=capture.items[site_index];
        const int ni=site.metadata.in_features,no=site.metadata.out_features;
        require((site_index<2 && ni==5120 && no==17408) ||
                (site_index==2 && ni==17408 && no==5120),
                "K5 direct bulk reached shape changed");
        _putenv_s("NINFER_EXL3_FAST_GDN_BULK_MLP_PACKED_K5","0");
        Candidate control(ni,no,rows,false,true);
        _putenv_s("NINFER_EXL3_FAST_GDN_BULK_MLP_PACKED_K5","1");
        Candidate candidate(ni,no,rows,false,true);
        control.begin_layer_reuse(512ull*1024*1024,true);
        candidate.begin_layer_reuse(512ull*1024*1024,true);
        require(control.supports(site.metadata,rows) &&
                candidate.supports(site.metadata,rows),
                "K5 direct bulk layer-reuse admission");
        const std::size_t active=static_cast<std::size_t>(rows)*no;
        const std::vector<std::uint16_t> poison(active+2*guard,0x3555);
        DeviceBuffer control_out(poison.size()*2),candidate_out(poison.size()*2);
        cuda_check(cudaMemcpy(control_out.get(),poison.data(),poison.size()*2,
                              cudaMemcpyHostToDevice),"K5 direct poison control");
        cuda_check(cudaMemcpy(candidate_out.get(),poison.data(),poison.size()*2,
                              cudaMemcpyHostToDevice),"K5 direct poison candidate");
        const auto* input=static_cast<const std::uint16_t*>(site.input->get());
        control.forward_numeric_candidate(site.weights,site.metadata,input,
            static_cast<std::uint16_t*>(control_out.get())+guard,rows);
        candidate.forward_numeric_candidate(site.weights,site.metadata,input,
            static_cast<std::uint16_t*>(candidate_out.get())+guard,rows);
        cuda_check(cudaDeviceSynchronize(),"K5 direct bulk operator complete");
        std::vector<std::uint16_t> control_host(poison.size()),
                                   candidate_host(poison.size());
        cuda_check(cudaMemcpy(control_host.data(),control_out.get(),
            poison.size()*2,cudaMemcpyDeviceToHost),"K5 direct copy control");
        cuda_check(cudaMemcpy(candidate_host.data(),candidate_out.get(),
            poison.size()*2,cudaMemcpyDeviceToHost),"K5 direct copy candidate");
        for(std::size_t i=0;i<guard;++i)
            require(control_host[i]==0x3555 && candidate_host[i]==0x3555 &&
                    control_host[guard+active+i]==0x3555 &&
                    candidate_host[guard+active+i]==0x3555,
                    "K5 direct bulk output canary");
        std::vector<std::uint16_t> control_active(
            control_host.begin()+guard,control_host.begin()+guard+active);
        std::vector<std::uint16_t> candidate_active(
            candidate_host.begin()+guard,candidate_host.begin()+guard+active);
        const auto parity=t69_compare_fp16(candidate_active,control_active,active);
        const auto oracle=t69_check_fp64_oracle(
            site,control_active,candidate_active,rows);
        require(parity.relative_l2<=0.003 && parity.max_abs<=0.10 &&
                oracle.candidate.relative_l2<=0.003 &&
                oracle.candidate.max_abs<=0.10 &&
                oracle.candidate.relative_l2<=oracle.direct.relative_l2+0.0005 &&
                oracle.candidate.max_abs<=oracle.direct.max_abs+0.02,
                "K5 direct bulk independent numeric gate");
        const auto a=control.stats(),b=candidate.stats();
        require(a.reconstructed_weight_calls==1 &&
                a.packed_direct_k5_gate_up_calls==0 &&
                a.packed_direct_k5_down_calls==0 &&
                b.reconstructed_weight_calls==0 &&
                b.packed_direct_k5_rows==rows &&
                b.packed_direct_k5_gate_up_calls==(site_index<2?1:0) &&
                b.packed_direct_k5_down_calls==(site_index==2?1:0),
                "K5 direct bulk source and operation dispatch");
        std::cout<<"K5_DIRECT_BULK site="<<site.operation
                 <<" rows="<<rows
                 <<" control_candidate_max_abs="<<parity.max_abs
                 <<" control_candidate_rel_l2="<<parity.relative_l2
                 <<" control_oracle_rel_l2="<<oracle.direct.relative_l2
                 <<" candidate_oracle_rel_l2="<<oracle.candidate.relative_l2
                 <<" candidate_oracle_max_abs="<<oracle.candidate.max_abs
                 <<" canary=1 dispatch=1\n";
        control.end_layer_reuse();
        candidate.end_layer_reuse();
    }
    _putenv_s("NINFER_EXL3_FAST_GDN_BULK_MLP_PACKED_K5","0");
    std::cout<<"K5_DIRECT_BULK_OPERATOR PASS sites=3 rows=1024"
             <<std::endl;
}
