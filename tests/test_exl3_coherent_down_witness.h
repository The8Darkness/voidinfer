// CPU qualification for one complete 128-column group of the reached K6 down
// owner. Included by test_exl3_cuda_linear.cpp.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace coherent_down_witness {

constexpr int input_features = 17408;
constexpr int output_features = 5120;
constexpr int group = 19;
constexpr int width = 128;
constexpr double hadamard_scale = 1.0 / 11.3137084989847603904135097936775846;

inline float from_half(std::uint16_t bits) {
    const unsigned sign = (bits & 0x8000u) << 16u;
    unsigned exponent = (bits >> 10u) & 31u;
    unsigned mantissa = bits & 1023u;
    unsigned word = sign;
    if (exponent == 0 && mantissa != 0) {
        int power = -14;
        while ((mantissa & 1024u) == 0) { mantissa <<= 1; --power; }
        word |= static_cast<unsigned>(power + 127) << 23u;
        word |= (mantissa & 1023u) << 13u;
    } else if (exponent == 31) {
        word |= 0x7f800000u | (mantissa << 13u);
    } else if (exponent != 0) {
        word |= (exponent + 112u) << 23u | (mantissa << 13u);
    }
    return std::bit_cast<float>(word);
}

inline std::uint16_t to_half(float value) {
    const unsigned word = std::bit_cast<unsigned>(value);
    const unsigned sign = (word >> 16u) & 0x8000u;
    const unsigned exponent = (word >> 23u) & 255u;
    unsigned fraction = word & 0x7fffffu;
    if (exponent == 255u) return static_cast<std::uint16_t>(
        sign | (fraction == 0 ? 0x7c00u : 0x7e00u));
    const int power = static_cast<int>(exponent) - 127;
    if (power > 15) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (power >= -14) {
        unsigned result_exponent = static_cast<unsigned>(power + 15);
        unsigned result_fraction = fraction >> 13u;
        const unsigned remainder = fraction & 0x1fffu;
        if (remainder > 0x1000u ||
            (remainder == 0x1000u && (result_fraction & 1u))) {
            ++result_fraction;
            if (result_fraction == 1024u) {
                result_fraction = 0;
                ++result_exponent;
            }
        }
        return static_cast<std::uint16_t>(sign | (result_exponent << 10u) |
                                          result_fraction);
    }
    if (power < -25) return static_cast<std::uint16_t>(sign);
    const unsigned mantissa = fraction | 0x800000u;
    const int shift = -power - 1;
    unsigned result_fraction = mantissa >> shift;
    const unsigned remainder = mantissa & ((1u << shift) - 1u);
    const unsigned halfway = 1u << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (result_fraction & 1u))) ++result_fraction;
    return static_cast<std::uint16_t>(sign | result_fraction);
}

template <typename T>
inline void hadamard(std::array<T, width>& values) {
    for (int stride = 1; stride < width; stride *= 2) {
        for (int base = 0; base < width; base += 2 * stride) {
            for (int lane = 0; lane < stride; ++lane) {
                const T left = values[base + lane];
                const T right = values[base + stride + lane];
                values[base + lane] = left + right;
                values[base + stride + lane] = left - right;
            }
        }
    }
}

inline int trellis_index(int row, int column) {
    return (((column & 7) * 4 + (row & 7) / 2) * 8 +
            (column / 8) * 4 + (row / 8) * 2 + (row & 1));
}

template<int Bits>
inline std::uint16_t packed_state(const std::uint16_t* tile, int encoded) {
    constexpr int words32 = Bits * 8;
    const int bit0 = encoded * Bits + Bits - 16 + 256 * Bits;
    const int bit1 = bit0 + 16;
    const int first = (bit0 / 32) % words32;
    const int second = ((bit1 - 1) / 32) % words32;
    const auto word = [&](int index) {
        return static_cast<std::uint32_t>(tile[2 * index]) |
               (static_cast<std::uint32_t>(tile[2 * index + 1]) << 16u);
    };
    const std::uint64_t window =
        (static_cast<std::uint64_t>(word(first)) << 32u) | word(second);
    const int shift = ((bit1 - 1) / 32 + 1) * 32 - bit1;
    return static_cast<std::uint16_t>(window >> shift);
}

inline double represented_weight(std::uint16_t state, std::uint32_t mul1) {
    const std::uint32_t product = static_cast<std::uint32_t>(state * mul1);
    const std::uint32_t byte_sum = (product & 255u) +
        ((product >> 8u) & 255u) + ((product >> 16u) & 255u) +
        (product >> 24u);
    const float decoded = std::fma(
        from_half(static_cast<std::uint16_t>(0x6400u + byte_sum)),
        from_half(0x1eeeu), from_half(0xc931u));
    return static_cast<double>(from_half(to_half(decoded)));
}

struct Result {
    std::array<double, width> fp64_output{};
    std::array<std::uint16_t, width> scalar_fp32_output{};
    std::array<std::uint16_t, width> split5_fp32_output{};
};

// All three calculations start from the same represented input and packed
// weights. FP64 evaluates the complete Op; FP32 paths expose the input FP16
// staging and K reduction partition that scalar/correction and verifier may
// choose. They are arithmetic witnesses, not CUDA MMA emulators.
template<int InputFeatures,int Bits,int Group>
inline Result evaluate_impl(std::span<const std::uint16_t> input,
                       std::span<const std::uint16_t> suh,
                       std::span<const std::uint16_t> svh,
                       std::span<const std::uint16_t> trellis,
                       std::uint32_t mul1, int row) {
    if (row < 0 || input.size() < static_cast<std::size_t>(row + 1) * InputFeatures ||
        suh.size() != InputFeatures || svh.size() != output_features ||
        trellis.size() != static_cast<std::size_t>(InputFeatures / 16) *
                          (output_features / 16) * (Bits * 16u))
        throw std::invalid_argument("coherent packed witness extent");
    std::vector<double> mathematical_input(InputFeatures);
    std::vector<float> staged_input(InputFeatures);
    for (int block = 0; block < InputFeatures; block += width) {
        std::array<double, width> exact{};
        std::array<float, width> staged{};
        for (int lane = 0; lane < width; ++lane) {
            const double product = static_cast<double>(from_half(input[
                static_cast<std::size_t>(row) * InputFeatures + block + lane])) *
                from_half(suh[block + lane]);
            exact[lane] = product;
            staged[lane] = static_cast<float>(product);
        }
        hadamard(exact);
        hadamard(staged);
        for (int lane = 0; lane < width; ++lane) {
            mathematical_input[block + lane] = exact[lane] * hadamard_scale;
            staged_input[block + lane] = from_half(to_half(
                staged[lane] * static_cast<float>(hadamard_scale)));
        }
    }
    std::array<double, width> exact_dot{};
    std::array<float, width> scalar_dot{};
    std::array<std::array<float, width>, 5> split_dot{};
    constexpr int tile_count = InputFeatures / 16;
    constexpr int tiles_per_split = (tile_count + 4) / 5;
    for (int tile_k = 0; tile_k < tile_count; ++tile_k) {
        for (int tile_n = 0; tile_n < 8; ++tile_n) {
            const std::size_t offset =
                (static_cast<std::size_t>(tile_k) * (output_features / 16) +
                 Group * 8 + tile_n) * (Bits * 16u);
            const auto* tile = trellis.data() + offset;
            for (int local_i = 0; local_i < 16; ++local_i) {
                const int i = tile_k * 16 + local_i;
                for (int local_j = 0; local_j < 16; ++local_j) {
                    const int j = tile_n * 16 + local_j;
                    const float weight = static_cast<float>(represented_weight(
                        packed_state<Bits>(tile, trellis_index(local_i, local_j)), mul1));
                    exact_dot[j] += mathematical_input[i] * weight;
                    const float product = staged_input[i] * weight;
                    scalar_dot[j] += product;
                    split_dot[std::min(tile_k / tiles_per_split, 4)][j] += product;
                }
            }
        }
    }
    std::array<float, width> reduced{};
    for (int part = 0; part < 5; ++part)
        for (int j = 0; j < width; ++j) reduced[j] += split_dot[part][j];
    hadamard(exact_dot);
    hadamard(scalar_dot);
    hadamard(reduced);
    Result result;
    for (int j = 0; j < width; ++j) {
        const double scale = from_half(svh[Group * width + j]);
        result.fp64_output[j] = exact_dot[j] * hadamard_scale * scale;
        const auto finish = [scale](float value) {
            const float normalized = from_half(to_half(
                value * static_cast<float>(hadamard_scale)));
            return to_half(normalized * static_cast<float>(scale));
        };
        result.scalar_fp32_output[j] = finish(scalar_dot[j]);
        result.split5_fp32_output[j] = finish(reduced[j]);
    }
    return result;
}

inline Result evaluate(std::span<const std::uint16_t> input,
                       std::span<const std::uint16_t> suh,
                       std::span<const std::uint16_t> svh,
                       std::span<const std::uint16_t> trellis,
                       std::uint32_t mul1,int row) {
    return evaluate_impl<input_features,6,group>(input,suh,svh,trellis,mul1,row);
}

inline Result evaluate_down_k7(std::span<const std::uint16_t> input,
                               std::span<const std::uint16_t> suh,
                               std::span<const std::uint16_t> svh,
                               std::span<const std::uint16_t> trellis,
                               std::uint32_t mul1,int row) {
    return evaluate_impl<input_features,7,group>(input,suh,svh,trellis,mul1,row);
}

inline Result evaluate_o_k7(std::span<const std::uint16_t> input,
                            std::span<const std::uint16_t> suh,
                            std::span<const std::uint16_t> svh,
                            std::span<const std::uint16_t> trellis,
                            std::uint32_t mul1,int row) {
    return evaluate_impl<6144,7,group>(input,suh,svh,trellis,mul1,row);
}

} // namespace coherent_down_witness
