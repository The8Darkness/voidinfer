// Included by the EXL3 acceptance harness inside its anonymous namespace.
// Independent-row comparator: the unchanged, qualified M1 H6 implementation.

namespace {

constexpr double kH6ParityMaxAbs = 0.0015;
constexpr double kH6ParityRelL2 = 0.0007;
constexpr int kH6 = 6;
constexpr int kH6Hadamard = 128;
constexpr int kH6InputTiles = kHidden / 16;
constexpr int kH6OutputTiles = kVocab / 16;
constexpr int kH6PackedWords16 = 16 * kH6;
constexpr double kH6UnitRound16 = 0x1p-11;
constexpr double kH6UnitRound32 = 0x1p-24;
constexpr double kH6UnitRound64 = 0x1p-53;
constexpr double kH6HalfEta = 0x1p-25;

double h6_gamma(double unit_roundoff, int operations) {
    const double product = unit_roundoff * static_cast<double>(operations);
    return product / (1.0 - product);
}

std::uint16_t h6_float_to_half_bits(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

void h6_hadamard(std::array<double, kH6Hadamard>& values) {
    for (int width = 1; width < kH6Hadamard; width *= 2) {
        for (int base = 0; base < kH6Hadamard; base += 2 * width) {
            for (int lane = 0; lane < width; ++lane) {
                const double left = values[static_cast<std::size_t>(base + lane)];
                const double right = values[static_cast<std::size_t>(base + lane + width)];
                values[static_cast<std::size_t>(base + lane)] = left + right;
                values[static_cast<std::size_t>(base + lane + width)] = left - right;
            }
        }
    }
}

std::uint32_t h6_load_u32(const std::uint16_t* packed, int index) {
    return static_cast<std::uint32_t>(packed[index * 2]) |
           (static_cast<std::uint32_t>(packed[index * 2 + 1]) << 16U);
}

// The trellis stores each logical 16x16 tile in the tensor-core fragment order
// documented by upstream reconstruct.cu. This inverse is derived from that
// row-major scatter: output lane selects the 4-entry quartet, input row selects
// the quartet member, and the high halves select the 8-column/8-row variants.
int h6_trellis_index(int input_row, int output_column) {
    const int quartet = (input_row & 7) / 2;
    const int row_variant = ((input_row >> 3) * 2) + (input_row & 1);
    const int output_lane = output_column & 7;
    const int column_variant = (output_column >> 3) * 4;
    return (output_lane * 4 + quartet) * 8 + column_variant + row_variant;
}

std::uint16_t h6_decode_state(const std::uint16_t* packed, int encoded) {
    constexpr int words32 = kH6 * 8;
    const int b0 = encoded * kH6 + kH6 - 16 + 256 * kH6;
    const int b1 = b0 + 16;
    const int i0 = b0 / 32;
    const int i1 = (b1 - 1) / 32;
    const int shift = (i1 + 1) * 32 - b1;
    const std::uint64_t window =
        (static_cast<std::uint64_t>(h6_load_u32(packed, i0 % words32)) << 32U) |
        h6_load_u32(packed, i1 % words32);
    return static_cast<std::uint16_t>((window >> shift) & 0xffffU);
}

std::uint16_t h6_decode_mul1(std::uint16_t state, std::uint32_t multiplier) {
    const std::uint32_t product = static_cast<std::uint32_t>(state) * multiplier;
    const std::uint32_t byte_sum = (product & 0xffU) + ((product >> 8U) & 0xffU) +
                                   ((product >> 16U) & 0xffU) + (product >> 24U);
    const float accumulator = half_to_float(static_cast<std::uint16_t>(0x6400U + byte_sum));
    const float inverse = half_to_float(0x1eeeU);
    const float bias = half_to_float(0xc931U);
    // One correctly-rounded FMA followed by the represented FP16 codebook value.
    return h6_float_to_half_bits(std::fma(accumulator, inverse, bias));
}

struct H6OracleStats {
    double max_abs = 0.0;
    double error_sq = 0.0;
    double norm_sq = 0.0;
    double max_ratio = 0.0;
    double max_bound = 0.0;
    double max_fp64_bound = 0.0;
    std::size_t values = 0;
    int worst_group = -1;
    int worst_row = -1;
    int worst_column = -1;
    bool within_bound = true;
};

// Independent scalar oracle for COMPLETE 128-column output-Hadamard groups.
// It never calls an EXL3 kernel and never materializes dense weights. For each
// selected group it copies only the packed 320x8x96-u16 trellis slice and decodes
// each represented weight while reducing the complete 5120-input dot product.
//
// Formula (c=1/sqrt(128), H=natural Sylvester order):
//   xhat = c H(x * su), a_j = sum_i xhat_i decode(trellis_ij,mul1),
//   z = c H(a), y = z * sv.
// The center is evaluated in FP64. The acceptance radius independently bounds
// the production FP32 butterflies/dot and both FP16 stores. gamma(10) covers a
// seven-stage butterfly plus product/normalization operations; gamma(5120)
// covers the full dot. A separately propagated FP64 evaluation bound is added.
H6OracleStats h6_check_oracle_groups(
    const ninfer::exl3::Exl3CudaLinearWeights& weights,
    const std::vector<std::uint16_t>& input,
    const std::vector<std::uint16_t>& actual_with_guards,
    std::size_t guard,
    const std::vector<int>& groups) {
    require(input.size() == 8u * kHidden, "H6 oracle requires eight complete rows");
    require(!groups.empty(), "H6 oracle group set is empty");

    std::vector<std::uint16_t> su(kHidden), sv(kVocab);
    std::uint32_t multiplier = 0;
    cuda_check(cudaMemcpy(su.data(), weights.suh, su.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "H6 oracle copy su");
    cuda_check(cudaMemcpy(sv.data(), weights.svh, sv.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "H6 oracle copy sv");
    cuda_check(cudaMemcpy(&multiplier, weights.mul1, sizeof(multiplier), cudaMemcpyDeviceToHost),
               "H6 oracle copy mul1");

    constexpr double c = 1.0 / 11.3137084989847603904135097936775846;
    constexpr double c_float = static_cast<double>(0.088388347648F);
    const double gamma10 = h6_gamma(kH6UnitRound32, 10);
    const double gamma_dot = h6_gamma(kH6UnitRound32, kHidden);
    const double gamma10_64 = h6_gamma(kH6UnitRound64, 10);
    const double gamma_dot_64 = h6_gamma(kH6UnitRound64, kHidden);

    std::vector<double> xhat(8u * kHidden), ex(8u * kHidden), ex64(8u * kHidden);
    for (int row = 0; row < 8; ++row) {
        for (int block = 0; block < kHidden / kH6Hadamard; ++block) {
            std::array<double, kH6Hadamard> transformed{};
            double sum_abs = 0.0;
            for (int lane = 0; lane < kH6Hadamard; ++lane) {
                const std::size_t i = static_cast<std::size_t>(row) * kHidden +
                                      block * kH6Hadamard + lane;
                transformed[static_cast<std::size_t>(lane)] =
                    static_cast<double>(half_to_float(input[i])) *
                    static_cast<double>(half_to_float(su[block * kH6Hadamard + lane]));
                sum_abs += std::abs(transformed[static_cast<std::size_t>(lane)]);
            }
            h6_hadamard(transformed);
            const double transform_error =
                gamma10 * c * sum_abs + std::abs(c_float - c) * sum_abs;
            const double transform_error64 = gamma10_64 * c * sum_abs;
            for (int lane = 0; lane < kH6Hadamard; ++lane) {
                const std::size_t i = static_cast<std::size_t>(row) * kHidden +
                                      block * kH6Hadamard + lane;
                xhat[i] = c * transformed[static_cast<std::size_t>(lane)];
                ex[i] = transform_error +
                        kH6UnitRound16 * (std::abs(xhat[i]) + transform_error) + kH6HalfEta;
                ex64[i] = transform_error64;
            }
        }
    }

    H6OracleStats stats;
    for (int group : groups) {
        require(group >= 0 && group < kVocab / kH6Hadamard, "H6 oracle group outside vocabulary");
        std::vector<std::uint16_t> packed(
            static_cast<std::size_t>(kH6InputTiles) * 8u * kH6PackedWords16);
        const int tile0 = group * (kH6Hadamard / 16);
        for (int tile_k = 0; tile_k < kH6InputTiles; ++tile_k) {
            const auto* source = weights.trellis +
                (static_cast<std::size_t>(tile_k) * kH6OutputTiles + tile0) * kH6PackedWords16;
            auto* destination = packed.data() +
                static_cast<std::size_t>(tile_k) * 8u * kH6PackedWords16;
            cuda_check(cudaMemcpy(destination, source,
                                  8u * kH6PackedWords16 * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "H6 oracle copy packed group");
        }

        std::array<std::array<double, kH6Hadamard>, 8> a{};
        std::array<std::array<double, kH6Hadamard>, 8> ea{};
        std::array<std::array<double, kH6Hadamard>, 8> ea64{};
        std::array<std::array<double, kH6Hadamard>, 8> dot_abs{};
        std::array<std::array<double, kH6Hadamard>, 8> dot_abs64{};
        for (int i = 0; i < kHidden; ++i) {
            const int tile_k = i / 16;
            const int input_row = i % 16;
            for (int local_j = 0; local_j < kH6Hadamard; ++local_j) {
                const int tile_n = local_j / 16;
                const int output_column = local_j % 16;
                const auto* tile = packed.data() +
                    (static_cast<std::size_t>(tile_k) * 8u + tile_n) * kH6PackedWords16;
                const std::uint16_t state =
                    h6_decode_state(tile, h6_trellis_index(input_row, output_column));
                const double weight = static_cast<double>(half_to_float(h6_decode_mul1(state, multiplier)));
                for (int row = 0; row < 8; ++row) {
                    const std::size_t xi = static_cast<std::size_t>(row) * kHidden + i;
                    const double abs_weight = std::abs(weight);
                    a[static_cast<std::size_t>(row)][static_cast<std::size_t>(local_j)] += xhat[xi] * weight;
                    ea[static_cast<std::size_t>(row)][static_cast<std::size_t>(local_j)] += ex[xi] * abs_weight;
                    ea64[static_cast<std::size_t>(row)][static_cast<std::size_t>(local_j)] += ex64[xi] * abs_weight;
                    dot_abs[static_cast<std::size_t>(row)][static_cast<std::size_t>(local_j)] +=
                        (std::abs(xhat[xi]) + ex[xi]) * abs_weight;
                    dot_abs64[static_cast<std::size_t>(row)][static_cast<std::size_t>(local_j)] +=
                        (std::abs(xhat[xi]) + ex64[xi]) * abs_weight;
                }
            }
        }

        for (int row = 0; row < 8; ++row) {
            double sum_abs_a_e = 0.0, sum_ea = 0.0;
            double sum_abs_a_e64 = 0.0, sum_ea64 = 0.0;
            for (int j = 0; j < kH6Hadamard; ++j) {
                ea[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] +=
                    gamma_dot * dot_abs[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
                ea64[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] +=
                    gamma_dot_64 * dot_abs64[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
                sum_ea += ea[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
                sum_abs_a_e += std::abs(a[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)]) +
                               ea[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
                sum_ea64 += ea64[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
                sum_abs_a_e64 += std::abs(a[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)]) +
                                 ea64[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
            }
            auto z = a[static_cast<std::size_t>(row)];
            h6_hadamard(z);
            for (double& value : z) value *= c;
            const double ez = c * sum_ea + gamma10 * c * sum_abs_a_e +
                              std::abs(c_float - c) * sum_abs_a_e;
            const double ez64 = c * sum_ea64 + gamma10_64 * c * sum_abs_a_e64;
            for (int j = 0; j < kH6Hadamard; ++j) {
                const int output_column = group * kH6Hadamard + j;
                const double scale = static_cast<double>(half_to_float(sv[output_column]));
                const double expected = z[static_cast<std::size_t>(j)] * scale;
                const double ez_half = ez +
                    kH6UnitRound16 * (std::abs(z[static_cast<std::size_t>(j)]) + ez) + kH6HalfEta;
                const double bound = std::abs(scale) * ez_half +
                    kH6UnitRound16 * (std::abs(expected) + std::abs(scale) * ez_half) + kH6HalfEta;
                const double fp64_bound = std::abs(scale) * ez64 +
                                           kH6UnitRound64 * std::abs(expected);
                const double total_bound = bound + fp64_bound;
                const double actual = half_to_float(actual_with_guards[
                    guard + static_cast<std::size_t>(row) * kVocab + output_column]);
                require(std::isfinite(expected) && std::isfinite(total_bound) && std::isfinite(actual),
                        "H6 scalar oracle encountered a nonfinite value");
                const double error = std::abs(actual - expected);
                stats.max_abs = std::max(stats.max_abs, error);
                stats.error_sq += error * error;
                stats.norm_sq += expected * expected;
                const double ratio = total_bound > 0.0
                    ? error / total_bound
                    : (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
                if (ratio > stats.max_ratio) {
                    stats.max_ratio = ratio;
                    stats.worst_group = group;
                    stats.worst_row = row;
                    stats.worst_column = output_column;
                }
                stats.max_bound = std::max(stats.max_bound, total_bound);
                stats.max_fp64_bound = std::max(stats.max_fp64_bound, fp64_bound);
                ++stats.values;
                stats.within_bound = stats.within_bound && error <= total_bound;
            }
        }
    }
    return stats;
}

class H6EnvironmentRestore {
public:
    H6EnvironmentRestore() : original_(env("NINFER_EXL3_H6_SMALL_M")) {}
    ~H6EnvironmentRestore() { _putenv_s("NINFER_EXL3_H6_SMALL_M", original_.c_str()); }
private:
    std::string original_;
};

struct H6Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~H6Graph() {
        if (executable != nullptr) cudaGraphExecDestroy(executable);
        if (graph != nullptr) cudaGraphDestroy(graph);
    }
};

} // namespace

void run_h6qualification(Exl3TextModel& target, std::ostream& out) {
    auto ctx = target.create_context(false);
    const auto& weights = ctx->target_lm_head_weights();
    const auto& metadata = ctx->target_lm_head_metadata();
    require(metadata.in_features == kHidden && metadata.out_features == kVocab &&
            metadata.K == 6 && metadata.mul1, "H6 qualification artifact mismatch");
    constexpr std::size_t guard = 128;
    constexpr std::uint16_t sentinel = 0x3555;
    const std::size_t elements = 8u * kVocab + 2u * guard;
    DeviceBuffer input(8u * kHidden * 2u), output(elements * 2u), reference(kVocab * 2u);
    auto* in = static_cast<std::uint16_t*>(input.get());
    auto* base = static_cast<std::uint16_t*>(output.get());
    auto* dst = base + guard;
    auto* ref_dst = static_cast<std::uint16_t*>(reference.get());
    H6EnvironmentRestore restore_environment;
    _putenv_s("NINFER_EXL3_H6_SMALL_M", "1");
    Exl3CudaLinearWorkspace ws(kHidden, kVocab, 8);
    require(std::string(ws.dispatch_name(metadata, 1)) == "h6_single_split",
            "H6 candidate changed M1 dispatch");
    for (int m = 2; m <= 8; ++m)
        require(std::string(ws.dispatch_name(metadata, m)) == "h6_small_m_single_split",
                "H6 candidate did not dispatch M2..8");
    _putenv_s("NINFER_EXL3_H6_SMALL_M", "0");
    Exl3CudaLinearWorkspace generic_ws(kHidden, kVocab, 8);
    for (int m = 2; m <= 8; ++m) {
        require(std::string(generic_ws.dispatch_name(metadata, m)) == "generic_tile",
                "H6 explicit env=0 did not select generic M2..8");
        require(std::string(ws.dispatch_name(metadata, m)) == "h6_small_m_single_split",
                "H6 candidate workspace did not latch construction-time flag");
    }
    _putenv_s("NINFER_EXL3_H6_SMALL_M", "");
    {
        Exl3CudaLinearWorkspace default_ws(kHidden, kVocab, 8);
        for (int m = 2; m <= 8; ++m)
            require(std::string(default_ws.dispatch_name(metadata, m)) == "h6_small_m_single_split",
                    "H6 unset environment did not select default candidate M2..8");
    }
    _putenv_s("NINFER_EXL3_H6_SMALL_M", "1");
    for (int m = 2; m <= 8; ++m)
        require(std::string(generic_ws.dispatch_name(metadata, m)) == "generic_tile",
                "H6 generic workspace observed a post-construction env flip");
    Exl3CudaLinearWorkspace ws9(kHidden, kVocab, 9);
    require(std::string(ws9.dispatch_name(metadata, 9)) == "generic_tile", "M9 escaped bounded specialization");
    for (int invalid : {-1, 0, 9}) {
        bool rejected = false;
        try { ws.forward(weights, metadata, in, dst, invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "H6 failed to reject invalid workspace row count");
    }
    struct Case { std::string name; std::vector<std::uint16_t> input; bool real = false; };
    std::vector<Case> cases;
    auto add = [&](const std::string& name) -> std::vector<std::uint16_t>& {
        cases.push_back(Case{name, std::vector<std::uint16_t>(8u * kHidden, 0), false});
        return cases.back().input;
    };
    auto& sine = add("sine");
    for (std::size_t i = 0; i < sine.size(); ++i)
        sine[i] = __half_as_ushort(__float2half_rn(std::sin(static_cast<float>(i) * 0.001f)));
    auto& alternating = add("alternating64");
    for (std::size_t i = 0; i < alternating.size(); ++i)
        alternating[i] = __half_as_ushort(__float2half_rn((i % 3) ? 64.0f : -64.0f));
    add("zero");
    auto& impulse = add("finite_max_impulse");
    for (int r = 0; r < 8; ++r)
        impulse[static_cast<std::size_t>(r) * kHidden + (r * 641) % kHidden] = (r & 1) ? 0xfbff : 0x7bff;
    auto& tail = add("last_column_impulse");
    for (int r = 0; r < 8; ++r)
        tail[static_cast<std::size_t>(r + 1) * kHidden - 1] = (r & 1) ? 0xd400 : 0x5400;
    std::stringstream paths(env("NINFER_E5A5H_INPUT_FILES"));
    std::string path;
    while (std::getline(paths, path, ';')) {
        require(!path.empty(), "empty H6 capture path");
        const auto bytes = std::filesystem::file_size(path);
        require(bytes >= 2u * kHidden && bytes <= 16u * kHidden && bytes % (2u * kHidden) == 0,
                "invalid H6 capture dimensions");
        std::vector<std::uint16_t> captured(static_cast<std::size_t>(bytes / 2));
        std::ifstream file(path, std::ios::binary);
        file.read(reinterpret_cast<char*>(captured.data()), static_cast<std::streamsize>(bytes));
        require(file.good(), "cannot read H6 capture");
        auto& real_case = cases.emplace_back(Case{
            std::filesystem::path(path).stem().string() + (captured.size() / kHidden < 8
                ? "+derived_row8" : ""),
            std::vector<std::uint16_t>(8u * kHidden, 0), true});
        auto& real = real_case.input;
        const std::size_t rows = captured.size() / kHidden;
        for (std::size_t r = 0; r < std::min<std::size_t>(rows, 8); ++r)
            std::copy_n(captured.data() + r * kHidden, kHidden, real.data() + r * kHidden);
        // Block-8 captures conventionally contain seven proposal rows. Preserve
        // their represented values and form one bounded real-derived eighth row
        // so M8 exercises eight distinct lanes instead of duplicating row zero.
        for (std::size_t r = rows; r < 8; ++r) {
            for (int i = 0; i < kHidden; ++i) {
                const float a = half_to_float(captured[static_cast<std::size_t>(i)]);
                const float b = half_to_float(captured[(1u % rows) * kHidden + static_cast<std::size_t>(i)]);
                real[r * kHidden + i] = h6_float_to_half_bits(0.75F * a + 0.25F * b);
            }
        }
    }
    out << "case,M,dispatch,max_abs,rel_l2,bit_mismatches,greedy_mismatches,min_top2_margin,repeat_equal,guards_ok,workspace_bytes,oracle_group_count,oracle_group_ids,oracle_values,oracle_max_abs,oracle_rel_l2,oracle_max_error_over_bound,oracle_max_bound,oracle_fp64_bound,oracle_worst_group,oracle_worst_row,oracle_worst_column,oracle_within_bound\n";
    std::vector<std::uint16_t> storage(elements, sentinel), got(elements), repeated(elements), ref(kVocab);
    for (const auto& test : cases) {
        cuda_check(cudaMemcpy(in, test.input.data(), test.input.size() * 2u, cudaMemcpyHostToDevice), "H6 test input");
        for (int m = 1; m <= 8; ++m) {
            cuda_check(cudaMemcpy(base, storage.data(), elements * 2u, cudaMemcpyHostToDevice), "H6 canaries");
            ws.forward(weights, metadata, in, dst, m);
            cuda_check(cudaMemcpy(got.data(), base, elements * 2u, cudaMemcpyDeviceToHost), "H6 result");
            bool guards_ok = std::all_of(got.begin(), got.begin() + guard, [](auto v) { return v == sentinel; }) &&
                std::all_of(got.begin() + guard + static_cast<std::size_t>(m) * kVocab, got.end(),
                            [](auto v) { return v == sentinel; });
            require(guards_ok, "H6 output canary or masked row overwritten");
            bool repeat_equal = true;
            for (int repeat = 0; repeat < 3; ++repeat) {
                ws.forward(weights, metadata, in, dst, m);
                cuda_check(cudaMemcpy(repeated.data(), base, elements * 2u, cudaMemcpyDeviceToHost), "H6 repeat");
                repeat_equal = repeat_equal && got == repeated;
            }
            double max_abs = 0.0, error_sq = 0.0, norm_sq = 0.0;
            double min_margin = std::numeric_limits<double>::infinity();
            std::size_t bit_mismatches = 0;
            int decisions = 0;
            std::vector<int> oracle_groups;
            auto add_oracle_group = [&](int group) {
                if (std::find(oracle_groups.begin(), oracle_groups.end(), group) == oracle_groups.end())
                    oracle_groups.push_back(group);
            };
            if (m == 8 && (test.name == "sine" || test.real)) {
                add_oracle_group(0);
                add_oracle_group((kVocab / kH6Hadamard) / 2);
                add_oracle_group(kVocab / kH6Hadamard - 1);
            }
            for (int r = 0; r < m; ++r) {
                ws.forward(weights, metadata, in + static_cast<std::size_t>(r) * kHidden, ref_dst, 1);
                cuda_check(cudaMemcpy(ref.data(), ref_dst, kVocab * 2u, cudaMemcpyDeviceToHost), "H6 independent M1 row");
                int got_best = 0, ref_best = 0;
                double best = -std::numeric_limits<double>::infinity(), second = best;
                const auto offset = guard + static_cast<std::size_t>(r) * kVocab;
                for (int c = 0; c < kVocab; ++c) {
                    const double a = half_to_float(got[offset + c]), b = half_to_float(ref[c]);
                    require(std::isfinite(a) && std::isfinite(b), "H6 nonfinite output");
                    const double delta = a - b;
                    max_abs = std::max(max_abs, std::abs(delta));
                    error_sq += delta * delta;
                    norm_sq += b * b;
                    bit_mismatches += got[offset + c] != ref[c];
                    if (a > half_to_float(got[offset + got_best])) got_best = c;
                    if (b > half_to_float(ref[ref_best])) ref_best = c;
                    if (b > best) { second = best; best = b; }
                    else if (b > second) second = b;
                }
                decisions += got_best != ref_best;
                min_margin = std::min(min_margin, best - second);
                if (m == 8 && test.real) add_oracle_group(ref_best / kH6Hadamard);
            }
            const double rel = norm_sq > 0 ? std::sqrt(error_sq / norm_sq) : std::sqrt(error_sq);
            H6OracleStats oracle;
            if (!oracle_groups.empty()) {
                for (int left = 0; left < 8; ++left) {
                    for (int right = left + 1; right < 8; ++right) {
                        require(!std::equal(test.input.begin() + static_cast<std::size_t>(left) * kHidden,
                                            test.input.begin() + static_cast<std::size_t>(left + 1) * kHidden,
                                            test.input.begin() + static_cast<std::size_t>(right) * kHidden),
                                "H6 oracle requires eight distinct represented rows");
                    }
                }
                oracle = h6_check_oracle_groups(weights, test.input, got, guard, oracle_groups);
            }
            std::ostringstream oracle_group_ids;
            for (std::size_t i = 0; i < oracle_groups.size(); ++i) {
                if (i != 0) oracle_group_ids << ';';
                oracle_group_ids << oracle_groups[i];
            }
            out << test.name << ',' << m << ',' << ws.dispatch_name(metadata, m) << ','
                << std::setprecision(12) << max_abs << ',' << rel << ',' << bit_mismatches << ','
                << decisions << ',' << min_margin << ',' << repeat_equal << ',' << guards_ok << ','
                << ws.workspace_bytes() << ',' << oracle_groups.size() << ',' << oracle_group_ids.str()
                << ',' << oracle.values << ','
                << oracle.max_abs << ','
                << (oracle.norm_sq > 0.0 ? std::sqrt(oracle.error_sq / oracle.norm_sq)
                                         : std::sqrt(oracle.error_sq)) << ','
                << oracle.max_ratio << ',' << oracle.max_bound << ',' << oracle.max_fp64_bound << ','
                << oracle.worst_group << ',' << oracle.worst_row << ',' << oracle.worst_column << ','
                << oracle.within_bound << '\n';
            out.flush();
            require(repeat_equal && bit_mismatches == 0 && max_abs <= kH6ParityMaxAbs &&
                        rel <= kH6ParityRelL2 && decisions == 0,
                    "H6 finite/repeat/numerical/token qualification failed");
            require(oracle.within_bound, "H6 scalar packed-code oracle bound exceeded; see CSV worst coordinate");
        }
    }

    // The retained explicit env=0 route has a different FP32 reduction profile
    // and is not required to match M1 bit-for-bit. Qualify its actual M8 output
    // against the same predeclared packed-code oracle used above, while reporting
    // the complete-vocabulary M1 discrepancy without relaxing either route's gate.
    int generic_real_profiles = 0;
    _putenv_s("NINFER_EXL3_H6_SMALL_M", "0");
    for (const auto& test : cases) {
        const bool prose = test.name.rfind("prose", 0) == 0;
        const bool code = test.name.rfind("code", 0) == 0;
        if (test.name != "sine" && !prose && !code) continue;
        generic_real_profiles += prose || code;
        cuda_check(cudaMemcpy(in, test.input.data(), test.input.size() * 2u,
                              cudaMemcpyHostToDevice), "H6 generic oracle input");
        cuda_check(cudaMemcpy(base, storage.data(), elements * 2u, cudaMemcpyHostToDevice),
                   "H6 generic oracle canaries");
        require(std::string(generic_ws.dispatch_name(metadata, 8)) == "generic_tile",
                "H6 env=0 workspace did not retain generic M8 dispatch");
        generic_ws.forward(weights, metadata, in, dst, 8);
        cuda_check(cudaMemcpy(got.data(), base, elements * 2u, cudaMemcpyDeviceToHost),
                   "H6 generic oracle result");
        const bool guards_ok =
            std::all_of(got.begin(), got.begin() + guard, [](auto v) { return v == sentinel; }) &&
            std::all_of(got.begin() + guard + 8u * kVocab, got.end(),
                        [](auto v) { return v == sentinel; });
        require(guards_ok, "H6 generic M8 output canary overwritten");

        std::vector<int> groups{0, (kVocab / kH6Hadamard) / 2,
                                kVocab / kH6Hadamard - 1};
        auto add_group = [&](int group) {
            if (std::find(groups.begin(), groups.end(), group) == groups.end())
                groups.push_back(group);
        };
        double m1_max_abs = 0.0, m1_error_sq = 0.0, m1_norm_sq = 0.0;
        std::size_t m1_bit_mismatches = 0;
        int m1_greedy_mismatches = 0;
        for (int row = 0; row < 8; ++row) {
            ws.forward(weights, metadata,
                       in + static_cast<std::size_t>(row) * kHidden, ref_dst, 1);
            cuda_check(cudaMemcpy(ref.data(), ref_dst, kVocab * 2u, cudaMemcpyDeviceToHost),
                       "H6 generic oracle M1 evidence");
            int actual_best = 0, m1_best = 0;
            const std::size_t offset = guard + static_cast<std::size_t>(row) * kVocab;
            for (int column = 0; column < kVocab; ++column) {
                const double actual = half_to_float(got[offset + column]);
                const double expected_m1 = half_to_float(ref[column]);
                require(std::isfinite(actual) && std::isfinite(expected_m1),
                        "H6 generic M8 produced nonfinite output");
                const double delta = actual - expected_m1;
                m1_max_abs = std::max(m1_max_abs, std::abs(delta));
                m1_error_sq += delta * delta;
                m1_norm_sq += expected_m1 * expected_m1;
                m1_bit_mismatches += got[offset + column] != ref[column];
                if (actual > half_to_float(got[offset + actual_best])) actual_best = column;
                if (expected_m1 > half_to_float(ref[m1_best])) m1_best = column;
            }
            m1_greedy_mismatches += actual_best != m1_best;
            if (prose || code) add_group(m1_best / kH6Hadamard);
        }
        const H6OracleStats oracle =
            h6_check_oracle_groups(weights, test.input, got, guard, groups);
        const double oracle_rel = oracle.norm_sq > 0.0
            ? std::sqrt(oracle.error_sq / oracle.norm_sq)
            : std::sqrt(oracle.error_sq);
        const double m1_rel = m1_norm_sq > 0.0
            ? std::sqrt(m1_error_sq / m1_norm_sq)
            : std::sqrt(m1_error_sq);
        std::cout << "E5A5H_GENERIC_ORACLE profile=" << test.name
                  << " dispatch=" << generic_ws.dispatch_name(metadata, 8)
                  << " groups=" << groups.size()
                  << " values=" << oracle.values
                  << " max_abs=" << std::setprecision(12) << oracle.max_abs
                  << " rel_l2=" << oracle_rel
                  << " max_error_over_bound=" << oracle.max_ratio
                  << " max_bound=" << oracle.max_bound
                  << " fp64_bound=" << oracle.max_fp64_bound
                  << " within_bound=" << oracle.within_bound
                  << " m1_max_abs=" << m1_max_abs
                  << " m1_rel_l2=" << m1_rel
                  << " m1_bit_mismatches=" << m1_bit_mismatches
                  << " m1_greedy_mismatches=" << m1_greedy_mismatches
                  << std::defaultfloat << '\n';
        require(oracle.within_bound,
                "H6 explicit env=0 generic M8 exceeded the independent oracle bound");
    }
    if (!env("NINFER_E5A5H_INPUT_FILES").empty())
        require(generic_real_profiles == 2,
                "H6 generic oracle requires unique prose and code real profiles");

    // Capture M7 and M8 against one stable captured-path input/output allocation
    // and the same workspace. Replays consume changed input contents after env=0;
    // exact eager parity proves graph address stability and flag latching.
    cudaStream_t graph_stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking), "H6 graph stream create");
    std::array<H6Graph, 2> graphs{};
    _putenv_s("NINFER_EXL3_H6_SMALL_M", "1");
    Exl3CudaLinearWorkspace graph_reference_ws(kHidden, kVocab, 8);
    DeviceBuffer graph_reference_output(8u * kVocab * sizeof(std::uint16_t));
    auto* graph_reference_dst =
        static_cast<std::uint16_t*>(graph_reference_output.get());
    for (int index = 0; index < 2; ++index) {
        const int m = 7 + index;
        cuda_check(cudaMemcpy(in, cases[0].input.data(), cases[0].input.size() * 2u, cudaMemcpyHostToDevice),
                   "H6 graph capture seed");
        cuda_check(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal),
                   "H6 graph begin capture");
        ws.forward(weights, metadata, in, dst, m, graph_stream);
        cuda_check(cudaStreamEndCapture(graph_stream, &graphs[static_cast<std::size_t>(index)].graph),
                   "H6 graph end capture");
        std::size_t graph_nodes = 0;
        cuda_check(cudaGraphGetNodes(graphs[static_cast<std::size_t>(index)].graph, nullptr, &graph_nodes),
                   "H6 graph node count");
        require(graph_nodes > 0, "H6 captured an empty CUDA Graph");
        cuda_check(cudaGraphInstantiate(&graphs[static_cast<std::size_t>(index)].executable,
                                        graphs[static_cast<std::size_t>(index)].graph,
                                        nullptr, nullptr, 0), "H6 graph instantiate");
    }
    _putenv_s("NINFER_EXL3_H6_SMALL_M", "0");
    for (int index = 0; index < 2; ++index) {
        const int m = 7 + index;
        std::vector<std::uint16_t> changed = cases[1].input;
        changed[static_cast<std::size_t>(index) * kHidden + 17] =
            h6_float_to_half_bits(index == 0 ? 3.25F : -5.5F);
        cuda_check(cudaMemcpyAsync(in, changed.data(), changed.size() * 2u, cudaMemcpyHostToDevice,
                                   graph_stream), "H6 graph changed input");
        // The eager reference owns separate transformed/accum/output storage,
        // so the captured workspace never contains the changed-input answer.
        graph_reference_ws.forward(weights, metadata, in, graph_reference_dst, m, graph_stream);
        cuda_check(cudaStreamSynchronize(graph_stream), "H6 graph eager synchronize");
        std::vector<std::uint16_t> eager(static_cast<std::size_t>(m) * kVocab);
        cuda_check(cudaMemcpy(eager.data(), graph_reference_dst, eager.size() * 2u,
                              cudaMemcpyDeviceToHost),
                   "H6 graph eager result");
        cuda_check(cudaMemsetAsync(dst, 0xa5, eager.size() * 2u, graph_stream),
                   "H6 graph poison output");
        cuda_check(cudaGraphLaunch(graphs[static_cast<std::size_t>(index)].executable, graph_stream),
                   "H6 graph replay");
        cuda_check(cudaStreamSynchronize(graph_stream), "H6 graph replay synchronize");
        std::vector<std::uint16_t> replay(eager.size());
        cuda_check(cudaMemcpy(replay.data(), dst, replay.size() * 2u, cudaMemcpyDeviceToHost),
                   "H6 graph replay result");
        require(eager == replay, "H6 M7/M8 CUDA Graph replay diverged from eager changed-input output");
        require(std::string(ws.dispatch_name(metadata, m)) == "h6_small_m_single_split",
                "H6 graph workspace observed post-construction env flip");
        std::cout << "E5A5H_GRAPH_PASS M=" << m << " changed_input=1 env_after_capture=0\n";
    }
    cuda_check(cudaStreamDestroy(graph_stream), "H6 graph stream destroy");
    std::cout << "E5A5H_QUALIFICATION_PASS cases=" << cases.size() << " rows=1..8\n";
}
