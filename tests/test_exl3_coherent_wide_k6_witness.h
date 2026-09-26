// Independent real-weight FP64 witness for a 128-column group of the reached
// target-owned wide K6 projections. Included by test_exl3_cuda_linear.cpp.
#pragma once

#include "test_exl3_coherent_down_witness.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace coherent_wide_k6_witness {

constexpr int group = 19;
constexpr int width = coherent_down_witness::width;
constexpr double hadamard_scale = coherent_down_witness::hadamard_scale;
using Output = std::array<double, width>;

inline Output evaluate(std::span<const std::uint16_t> input,
                       std::span<const std::uint16_t> suh,
                       std::span<const std::uint16_t> svh,
                       std::span<const std::uint16_t> trellis,
                       std::uint32_t mul1, int row,
                       int input_features, int output_features) {
    if (row < 0 || input_features <= 0 || output_features < (group+1)*width ||
        input_features % width != 0 || output_features % width != 0 ||
        input.size() < static_cast<std::size_t>(row+1)*input_features ||
        suh.size() != static_cast<std::size_t>(input_features) ||
        svh.size() != static_cast<std::size_t>(output_features) ||
        trellis.size() != static_cast<std::size_t>(input_features/16)*
            (output_features/16)*6u*16u)
        throw std::invalid_argument("coherent wide K6 packed witness extent");
    std::vector<double> transformed(input_features);
    for (int block=0; block<input_features; block+=width) {
        Output values{};
        for (int lane=0; lane<width; ++lane)
            values[lane] = static_cast<double>(coherent_down_witness::from_half(
                input[static_cast<std::size_t>(row)*input_features+block+lane])) *
                coherent_down_witness::from_half(suh[block+lane]);
        coherent_down_witness::hadamard(values);
        for (int lane=0; lane<width; ++lane)
            transformed[block+lane] = values[lane]*hadamard_scale;
    }
    Output dot{};
    for (int tile_k=0; tile_k<input_features/16; ++tile_k) {
        for (int tile_n=0; tile_n<8; ++tile_n) {
            const auto* packed = trellis.data() +
                (static_cast<std::size_t>(tile_k)*(output_features/16)+
                 group*8+tile_n)*96u;
            for (int i=0; i<16; ++i) {
                const double a=transformed[tile_k*16+i];
                for (int j=0; j<16; ++j) {
                    const int local=tile_n*16+j;
                    const auto state=coherent_down_witness::packed_state<6>(
                        packed, coherent_down_witness::trellis_index(i,j));
                    dot[local]+=a*coherent_down_witness::represented_weight(
                        state,mul1);
                }
            }
        }
    }
    coherent_down_witness::hadamard(dot);
    for (int lane=0; lane<width; ++lane)
        dot[lane] *= hadamard_scale *
            coherent_down_witness::from_half(svh[group*width+lane]);
    return dot;
}

} // namespace coherent_wide_k6_witness
