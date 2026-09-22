#pragma once

// Requires test_exl3_target_o_k7_oracle.h for independent K7 bit extraction.
// Qualification-only independent FP64 formula for K7 down17408->5120.
// Uses independent bit-by-bit K7 extraction and the accepted direct unsigned
// MUL1 byte sum, Sylvester transform and error helpers. Requires the H6 helper first.
// All 17408 represented input terms enter every selected output value; input
// tile count is1088 and packed output stride320, independent of H6 dimensions.
// The center has no production staging casts, split tree, MMA or dense matrix.
// gamma(17408) bounds the shorter per-split FP32 accumulation and merge; input
// and output transforms, normalization constants and half storage are bounded
// separately from FP64 evaluation error. No CPU inference fallback is added.
// Planned groups{0,1,2,19,20,39} cover256-column tile boundaries and the tail.
// Selected-column oracle coverage is separate from full-output M1 parity.
H6OracleStats target_down_k7_check_oracle_groups(
    const ninfer::exl3::Exl3CudaLinearWeights& weights,
    const ninfer::exl3::Exl3CudaLinearMetadata& metadata,
    const std::vector<std::uint16_t>& input,
    const std::vector<std::uint16_t>& actual_with_guards,
    std::size_t guard,
    const std::vector<int>& groups) {
    // Same represented transform/MUL1 formula as the accepted H6 oracle; only the output
    // and input extents and packed row stride differ. Keep the center in FP64 and bound
    // FP32 reduction plus storage rounding separately, without stage casts.
    constexpr int input_features = 17408;
    constexpr int input_tiles = input_features / 16;
    constexpr int output_features = 5120;
    constexpr int output_tiles = output_features / 16;
    require(metadata.in_features == input_features &&
                metadata.out_features == output_features && metadata.K == 7 &&
                metadata.mul1 && !metadata.mcg && !metadata.has_bias,
            "target down K7 oracle requires exact 17408->5120 K7 MUL1 metadata");
    require(weights.trellis && weights.suh && weights.svh && weights.mul1,
            "target down K7 oracle received null represented weight tensors");
    require(actual_with_guards.size() ==
                2u * guard + 8u * static_cast<std::size_t>(output_features),
            "target down K7 oracle guarded output extent mismatch");
    std::array<bool, output_features / kH6Hadamard> seen{};
    for (const int group : groups) {
        require(group >= 0 && group < output_features / kH6Hadamard,
                "target down K7 oracle group outside output extent");
        require(!seen[static_cast<std::size_t>(group)],
                "target down K7 oracle repeated an output group");
        seen[static_cast<std::size_t>(group)] = true;
    }
    require(input.size() == 8u * input_features, "target down K7 oracle requires eight complete rows");
    require(!groups.empty(), "target down K7 oracle group set is empty");

    std::vector<std::uint16_t> su(input_features), sv(output_features);
    std::uint32_t multiplier = 0;
    cuda_check(cudaMemcpy(su.data(), weights.suh, su.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "target down K7 oracle copy su");
    cuda_check(cudaMemcpy(sv.data(), weights.svh, sv.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "target down K7 oracle copy sv");
    cuda_check(cudaMemcpy(&multiplier, weights.mul1, sizeof(multiplier), cudaMemcpyDeviceToHost),
               "target down K7 oracle copy mul1");

    constexpr double c = 1.0 / 11.3137084989847603904135097936775846;
    constexpr double c_float = static_cast<double>(0.088388347648F);
    const double gamma10 = h6_gamma(kH6UnitRound32, 10);
    const double gamma_dot = h6_gamma(kH6UnitRound32, input_features);
    const double gamma10_64 = h6_gamma(kH6UnitRound64, 10);
    const double gamma_dot_64 = h6_gamma(kH6UnitRound64, input_features);

    std::vector<double> xhat(8u * input_features), ex(8u * input_features), ex64(8u * input_features);
    for (int row = 0; row < 8; ++row) {
        for (int block = 0; block < input_features / kH6Hadamard; ++block) {
            std::array<double, kH6Hadamard> transformed{};
            double sum_abs = 0.0;
            for (int lane = 0; lane < kH6Hadamard; ++lane) {
                const std::size_t i = static_cast<std::size_t>(row) * input_features +
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
                const std::size_t i = static_cast<std::size_t>(row) * input_features +
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
        require(group >= 0 && group < output_features / kH6Hadamard, "target down K7 oracle group outside output extent");
        std::vector<std::uint16_t> packed(
            static_cast<std::size_t>(input_tiles) * 8u * kOK7PackedWords16);
        const int tile0 = group * (kH6Hadamard / 16);
        for (int tile_k = 0; tile_k < input_tiles; ++tile_k) {
            const auto* source = weights.trellis +
                (static_cast<std::size_t>(tile_k) * output_tiles + tile0) * kOK7PackedWords16;
            auto* destination = packed.data() +
                static_cast<std::size_t>(tile_k) * 8u * kOK7PackedWords16;
            cuda_check(cudaMemcpy(destination, source,
                                  8u * kOK7PackedWords16 * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "target down K7 oracle copy packed group");
        }

        std::array<std::array<double, kH6Hadamard>, 8> a{};
        std::array<std::array<double, kH6Hadamard>, 8> ea{};
        std::array<std::array<double, kH6Hadamard>, 8> ea64{};
        std::array<std::array<double, kH6Hadamard>, 8> dot_abs{};
        std::array<std::array<double, kH6Hadamard>, 8> dot_abs64{};
        for (int i = 0; i < input_features; ++i) {
            const int tile_k = i / 16;
            const int input_row = i % 16;
            for (int local_j = 0; local_j < kH6Hadamard; ++local_j) {
                const int tile_n = local_j / 16;
                const int output_column = local_j % 16;
                const auto* tile = packed.data() +
                    (static_cast<std::size_t>(tile_k) * 8u + tile_n) * kOK7PackedWords16;
                const std::uint16_t state =
                    target_o_k7_decode_state(tile, h6_trellis_index(input_row, output_column));
                const double weight = static_cast<double>(half_to_float(h6_decode_mul1(state, multiplier)));
                for (int row = 0; row < 8; ++row) {
                    const std::size_t xi = static_cast<std::size_t>(row) * input_features + i;
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
                    guard + static_cast<std::size_t>(row) * output_features + output_column]);
                require(std::isfinite(expected) && std::isfinite(total_bound) && std::isfinite(actual),
                        "target down K7 scalar oracle encountered a nonfinite value");
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
