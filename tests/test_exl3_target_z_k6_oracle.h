#pragma once

// Qualification-only FP64 center for the measured target K6 z family.
// Requires test_exl3_h6_qualification.h to have defined its independent scalar
// K6 bit decoder, represented MUL1 decoder, Sylvester transform and error model.
// Derived directly from the already-qualified H6 full-formula oracle, with
// output extent/trellis stride bound to 6144 instead of the vocabulary size.
// No production transform, staging cast, MMA reduction, dense matrix or CPU
// inference fallback is used. Each selected complete 128-column output group
// evaluates all 5120 input terms on all 8 represented rows. Selected groups are
// column coverage, not an assertion that unselected columns were oracle-tested.
// gamma(5120) also bounds the shorter per-split FP32 dot/reduction paths; it does
// not prescribe the production split order or alter the mathematical center.
H6OracleStats target_z_k6_check_oracle_groups(
    const ninfer::exl3::Exl3CudaLinearWeights& weights,
    const ninfer::exl3::Exl3CudaLinearMetadata& metadata,
    const std::vector<std::uint16_t>& input,
    const std::vector<std::uint16_t>& actual_with_guards,
    std::size_t guard,
    const std::vector<int>& groups) {
    // Same represented K6 formula as the accepted H6 oracle; only the output
    // extent and packed row stride differ. Keep the center in FP64 and bound
    // FP32 reduction plus storage rounding separately, without stage casts.
    constexpr int output_features = 6144;
    constexpr int output_tiles = output_features / 16;
    require(metadata.in_features == kHidden &&
                metadata.out_features == output_features && metadata.K == 6 &&
                metadata.mul1 && !metadata.mcg && !metadata.has_bias,
            "target K6 oracle requires exact 5120->6144 K6 MUL1 metadata");
    require(weights.trellis && weights.suh && weights.svh && weights.mul1,
            "target K6 oracle received null represented weight tensors");
    require(actual_with_guards.size() ==
                2u * guard + 8u * static_cast<std::size_t>(output_features),
            "target K6 oracle guarded output extent mismatch");
    std::array<bool, output_features / kH6Hadamard> seen{};
    for (const int group : groups) {
        require(group >= 0 && group < output_features / kH6Hadamard,
                "target K6 oracle group outside output extent");
        require(!seen[static_cast<std::size_t>(group)],
                "target K6 oracle repeated an output group");
        seen[static_cast<std::size_t>(group)] = true;
    }
    require(input.size() == 8u * kHidden, "target K6 oracle requires eight complete rows");
    require(!groups.empty(), "target K6 oracle group set is empty");

    std::vector<std::uint16_t> su(kHidden), sv(output_features);
    std::uint32_t multiplier = 0;
    cuda_check(cudaMemcpy(su.data(), weights.suh, su.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "target K6 oracle copy su");
    cuda_check(cudaMemcpy(sv.data(), weights.svh, sv.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "target K6 oracle copy sv");
    cuda_check(cudaMemcpy(&multiplier, weights.mul1, sizeof(multiplier), cudaMemcpyDeviceToHost),
               "target K6 oracle copy mul1");

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
        require(group >= 0 && group < output_features / kH6Hadamard, "target K6 oracle group outside vocabulary");
        std::vector<std::uint16_t> packed(
            static_cast<std::size_t>(kH6InputTiles) * 8u * kH6PackedWords16);
        const int tile0 = group * (kH6Hadamard / 16);
        for (int tile_k = 0; tile_k < kH6InputTiles; ++tile_k) {
            const auto* source = weights.trellis +
                (static_cast<std::size_t>(tile_k) * output_tiles + tile0) * kH6PackedWords16;
            auto* destination = packed.data() +
                static_cast<std::size_t>(tile_k) * 8u * kH6PackedWords16;
            cuda_check(cudaMemcpy(destination, source,
                                  8u * kH6PackedWords16 * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "target K6 oracle copy packed group");
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
                    guard + static_cast<std::size_t>(row) * output_features + output_column]);
                require(std::isfinite(expected) && std::isfinite(total_bound) && std::isfinite(actual),
                        "target K6 scalar oracle encountered a nonfinite value");
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
