#pragma once

using ninfer::exl3::Exl3CudaLinearMetadata;
using ninfer::exl3::Exl3CudaLinearWeights;
using K5OracleStats = H6OracleStats;

std::uint32_t k5_oracle_load_u32(const std::uint16_t* packed, int index) {
    return static_cast<std::uint32_t>(packed[index * 2]) |
           (static_cast<std::uint32_t>(packed[index * 2 + 1]) << 16U);
}

// Independent K5 state-bit extraction. The circular 16-bit state window is
// reconstructed directly from the represented 80-u16 packed tile.
std::uint16_t k5_oracle_decode_state(const std::uint16_t* packed, int encoded) {
    constexpr int bits = 5;
    constexpr int words32 = bits * 8;
    const int bit_end = encoded * bits + bits - 16 + 256 * bits;
    const int next_end = bit_end + 16;
    const int first_word = bit_end / 32;
    const int last_word = (next_end - 1) / 32;
    const int shift = (last_word + 1) * 32 - next_end;
    const std::uint64_t window =
        (static_cast<std::uint64_t>(
             k5_oracle_load_u32(packed, first_word % words32)) << 32U) |
        k5_oracle_load_u32(packed, last_word % words32);
    return static_cast<std::uint16_t>((window >> shift) & 0xffffU);
}

// Independent scalar oracle for complete 128-column output-Hadamard groups of
// the admitted K5 projection shapes. It copies only SU/SV, MUL1, and selected
// packed group slices. The represented center is evaluated in FP64 without a
// production kernel, dense weights, or production staging casts.
K5OracleStats k5_check_oracle_groups(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,
    const std::vector<std::uint16_t>& input,
    const std::vector<std::uint16_t>& actual_with_guards,
    std::size_t guard,
    const std::vector<int>& groups) {
    constexpr int rows = 8;
    constexpr int hadamard = 128;
    constexpr int bits = 5;
    constexpr int packed_words16 = 16 * bits;
    constexpr double unit_round16 = 0x1p-11;
    constexpr double unit_round32 = 0x1p-24;
    constexpr double unit_round64 = 0x1p-53;
    constexpr double half_eta = 0x1p-25;
    constexpr double normalization =
        1.0 / 11.3137084989847603904135097936775846;
    constexpr double normalization_f32 = static_cast<double>(0.088388347648F);

    const int input_features = metadata.in_features;
    const int output_features = metadata.out_features;
    const bool supported_shape =
        (input_features == 5120 && output_features == 17408) ||
        (input_features == 5120 && output_features == 4096) ||
        (input_features == 5120 && output_features == 1024) ||
        (input_features == 4096 && output_features == 5120) ||
        (input_features == 17408 && output_features == 5120);
    require(supported_shape, "K5 oracle received an unsupported projection shape");
    require(metadata.K == bits && !metadata.mcg && metadata.mul1 && !metadata.has_bias,
            "K5 oracle requires exact K5 MUL1 metadata");
    require(input_features % 16 == 0 && output_features % hadamard == 0,
            "K5 oracle shape is not tile aligned");
    require(weights.trellis != nullptr && weights.suh != nullptr &&
                weights.svh != nullptr && weights.mul1 != nullptr,
            "K5 oracle received a null model tensor");
    require(input.size() == static_cast<std::size_t>(rows) * input_features,
            "K5 oracle requires eight complete represented input rows");
    require(actual_with_guards.size() ==
                2 * guard + static_cast<std::size_t>(rows) * output_features,
            "K5 oracle guarded output extent mismatch");
    require(!groups.empty(), "K5 oracle group set is empty");

    const int input_tiles = input_features / 16;
    const int output_tiles = output_features / 16;
    const int output_groups = output_features / hadamard;
    std::vector<bool> seen(static_cast<std::size_t>(output_groups), false);
    for (int group : groups) {
        require(group >= 0 && group < output_groups,
                "K5 oracle group is outside the output extent");
        require(!seen[static_cast<std::size_t>(group)],
                "K5 oracle group set contains a duplicate");
        seen[static_cast<std::size_t>(group)] = true;
    }

    std::vector<std::uint16_t> su(static_cast<std::size_t>(input_features));
    std::vector<std::uint16_t> sv(static_cast<std::size_t>(output_features));
    std::uint32_t multiplier = 0;
    cuda_check(cudaMemcpy(su.data(), weights.suh, su.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "K5 oracle copy SU");
    cuda_check(cudaMemcpy(sv.data(), weights.svh, sv.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost), "K5 oracle copy SV");
    cuda_check(cudaMemcpy(&multiplier, weights.mul1, sizeof(multiplier),
                          cudaMemcpyDeviceToHost), "K5 oracle copy MUL1");

    const double gamma10 = h6_gamma(unit_round32, 10);
    const double gamma_dot = h6_gamma(unit_round32, input_features);
    const double gamma10_64 = h6_gamma(unit_round64, 10);
    const double gamma_dot_64 = h6_gamma(unit_round64, input_features);

    const std::size_t input_values = static_cast<std::size_t>(rows) * input_features;
    std::vector<double> transformed(input_values);
    std::vector<double> transformed_error(input_values);
    std::vector<double> transformed_error64(input_values);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < input_features / hadamard; ++block) {
            std::array<double, hadamard> values{};
            double sum_abs = 0.0;
            for (int lane = 0; lane < hadamard; ++lane) {
                const std::size_t index = static_cast<std::size_t>(row) * input_features +
                    static_cast<std::size_t>(block) * hadamard + lane;
                values[static_cast<std::size_t>(lane)] =
                    static_cast<double>(half_to_float(input[index])) *
                    static_cast<double>(half_to_float(
                        su[static_cast<std::size_t>(block) * hadamard + lane]));
                sum_abs += std::abs(values[static_cast<std::size_t>(lane)]);
            }
            h6_hadamard(values);
            const double transform_bound = gamma10 * normalization * sum_abs +
                std::abs(normalization_f32 - normalization) * sum_abs;
            const double transform_bound64 = gamma10_64 * normalization * sum_abs;
            for (int lane = 0; lane < hadamard; ++lane) {
                const std::size_t index = static_cast<std::size_t>(row) * input_features +
                    static_cast<std::size_t>(block) * hadamard + lane;
                transformed[index] = normalization * values[static_cast<std::size_t>(lane)];
                transformed_error[index] = transform_bound +
                    unit_round16 * (std::abs(transformed[index]) + transform_bound) + half_eta;
                transformed_error64[index] = transform_bound64;
            }
        }
    }

    K5OracleStats stats;
    for (int group : groups) {
        std::vector<std::uint16_t> packed(
            static_cast<std::size_t>(input_tiles) * 8U * packed_words16);
        const int first_output_tile = group * (hadamard / 16);
        for (int input_tile = 0; input_tile < input_tiles; ++input_tile) {
            const auto* source = weights.trellis +
                (static_cast<std::size_t>(input_tile) * output_tiles + first_output_tile) *
                    packed_words16;
            auto* destination = packed.data() +
                static_cast<std::size_t>(input_tile) * 8U * packed_words16;
            cuda_check(cudaMemcpy(destination, source,
                                  8U * packed_words16 * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "K5 oracle copy packed group");
        }

        std::array<std::array<double, hadamard>, rows> accumulator{};
        std::array<std::array<double, hadamard>, rows> accumulator_error{};
        std::array<std::array<double, hadamard>, rows> accumulator_error64{};
        std::array<std::array<double, hadamard>, rows> dot_magnitude{};
        std::array<std::array<double, hadamard>, rows> dot_magnitude64{};
        for (int input_column = 0; input_column < input_features; ++input_column) {
            const int input_tile = input_column / 16;
            const int input_row = input_column % 16;
            for (int local_output = 0; local_output < hadamard; ++local_output) {
                const int output_tile = local_output / 16;
                const int output_column = local_output % 16;
                const auto* tile = packed.data() +
                    (static_cast<std::size_t>(input_tile) * 8U + output_tile) *
                        packed_words16;
                const std::uint16_t state = k5_oracle_decode_state(
                    tile, h6_trellis_index(input_row, output_column));
                const double weight = static_cast<double>(half_to_float(
                    h6_decode_mul1(state, multiplier)));
                const double abs_weight = std::abs(weight);
                for (int row = 0; row < rows; ++row) {
                    const std::size_t index =
                        static_cast<std::size_t>(row) * input_features + input_column;
                    accumulator[static_cast<std::size_t>(row)]
                               [static_cast<std::size_t>(local_output)] +=
                        transformed[index] * weight;
                    accumulator_error[static_cast<std::size_t>(row)]
                                     [static_cast<std::size_t>(local_output)] +=
                        transformed_error[index] * abs_weight;
                    accumulator_error64[static_cast<std::size_t>(row)]
                                       [static_cast<std::size_t>(local_output)] +=
                        transformed_error64[index] * abs_weight;
                    dot_magnitude[static_cast<std::size_t>(row)]
                                 [static_cast<std::size_t>(local_output)] +=
                        (std::abs(transformed[index]) + transformed_error[index]) * abs_weight;
                    dot_magnitude64[static_cast<std::size_t>(row)]
                                   [static_cast<std::size_t>(local_output)] +=
                        (std::abs(transformed[index]) + transformed_error64[index]) * abs_weight;
                }
            }
        }

        for (int row = 0; row < rows; ++row) {
            double sum_error = 0.0;
            double sum_magnitude = 0.0;
            double sum_error64 = 0.0;
            double sum_magnitude64 = 0.0;
            for (int column = 0; column < hadamard; ++column) {
                auto& error = accumulator_error[static_cast<std::size_t>(row)]
                                               [static_cast<std::size_t>(column)];
                auto& error64 = accumulator_error64[static_cast<std::size_t>(row)]
                                                   [static_cast<std::size_t>(column)];
                error += gamma_dot * dot_magnitude[static_cast<std::size_t>(row)]
                                              [static_cast<std::size_t>(column)];
                error64 += gamma_dot_64 * dot_magnitude64[static_cast<std::size_t>(row)]
                                                    [static_cast<std::size_t>(column)];
                sum_error += error;
                sum_magnitude += std::abs(accumulator[static_cast<std::size_t>(row)]
                                                     [static_cast<std::size_t>(column)]) + error;
                sum_error64 += error64;
                sum_magnitude64 += std::abs(accumulator[static_cast<std::size_t>(row)]
                                                       [static_cast<std::size_t>(column)]) + error64;
            }

            auto output_group = accumulator[static_cast<std::size_t>(row)];
            h6_hadamard(output_group);
            for (double& value : output_group) value *= normalization;
            const double output_error = normalization * sum_error +
                gamma10 * normalization * sum_magnitude +
                std::abs(normalization_f32 - normalization) * sum_magnitude;
            const double output_error64 = normalization * sum_error64 +
                gamma10_64 * normalization * sum_magnitude64;

            for (int local_output = 0; local_output < hadamard; ++local_output) {
                const int output_column = group * hadamard + local_output;
                const double scale = static_cast<double>(half_to_float(
                    sv[static_cast<std::size_t>(output_column)]));
                const double center =
                    output_group[static_cast<std::size_t>(local_output)] * scale;
                const double intermediate_half_bound = output_error +
                    unit_round16 * (std::abs(output_group[static_cast<std::size_t>(local_output)]) +
                                    output_error) + half_eta;
                const double fp32_bound = std::abs(scale) * intermediate_half_bound +
                    unit_round16 * (std::abs(center) +
                                    std::abs(scale) * intermediate_half_bound) + half_eta;
                const double fp64_bound = std::abs(scale) * output_error64 +
                    unit_round64 * std::abs(center);
                const double total_bound = fp32_bound + fp64_bound;
                const std::size_t actual_index = guard +
                    static_cast<std::size_t>(row) * output_features + output_column;
                const double actual = half_to_float(actual_with_guards[actual_index]);
                require(std::isfinite(center) && std::isfinite(total_bound) &&
                            std::isfinite(actual),
                        "K5 scalar oracle encountered a nonfinite value");
                const double error = std::abs(actual - center);
                stats.max_abs = std::max(stats.max_abs, error);
                stats.error_sq += error * error;
                stats.norm_sq += center * center;
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
