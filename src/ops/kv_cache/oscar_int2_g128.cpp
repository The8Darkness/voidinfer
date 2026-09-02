#include "ops/kv_cache/oscar_int2_g128.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer::ops {
namespace {

int resolve_clip_index(float clip_ratio) {
    if (!std::isfinite(clip_ratio) || clip_ratio < 0.0F || clip_ratio > 1.0F) {
        throw std::invalid_argument("OSCAR INT2 clip ratio must be finite and in [0,1]");
    }
    if (clip_ratio <= 0.0F) { return -1; }
    int index = static_cast<int>(clip_ratio * static_cast<float>(kOscarInt2G128HeadDim));
    index = std::clamp(index, 0, kOscarInt2G128HeadDim - 1);
    return index;
}

} // namespace

OscarInt2G128EncodedRow oscar_int2_g128_encode(const float* values, int count,
                                               float clip_ratio) {
    if (values == nullptr || count != kOscarInt2G128HeadDim) {
        throw std::invalid_argument("OSCAR INT2 G128 requires exactly 256 input values");
    }

    OscarInt2G128EncodedRow encoded{};
    std::vector<float> absolute_values;
    absolute_values.reserve(kOscarInt2G128HeadDim);
    for (int dimension = 0; dimension < kOscarInt2G128HeadDim; ++dimension) {
        if (!std::isfinite(values[dimension])) {
            throw std::invalid_argument("OSCAR INT2 input contains NaN or Inf");
        }
        absolute_values.push_back(std::fabs(values[dimension]));
    }
    const int clip_index = resolve_clip_index(clip_ratio);
    float threshold       = std::numeric_limits<float>::infinity();
    if (clip_index >= 0) {
        std::sort(absolute_values.begin(), absolute_values.end());
        threshold = absolute_values[static_cast<std::size_t>(clip_index)];
    }
    for (int dimension = 0; dimension < kOscarInt2G128HeadDim; ++dimension) {
        encoded.clipped[static_cast<std::size_t>(dimension)] =
            std::clamp(values[dimension], -threshold, threshold);
    }

    for (int group = 0; group < kOscarInt2G128Groups; ++group) {
        const int begin = group * kOscarInt2G128GroupSize;
        float minimum   = encoded.clipped[static_cast<std::size_t>(begin)];
        float maximum   = minimum;
        for (int dimension = begin + 1; dimension < begin + kOscarInt2G128GroupSize;
             ++dimension) {
            const float value = encoded.clipped[static_cast<std::size_t>(dimension)];
            minimum           = std::min(minimum, value);
            maximum           = std::max(maximum, value);
        }
        const float range = std::max(maximum - minimum, 1.0e-8F);
        const float scale = range / static_cast<float>(kOscarInt2G128Levels);
        const float zero  = -minimum / scale;
        encoded.scales_zeros[static_cast<std::size_t>(2 * group)]     = scale;
        encoded.scales_zeros[static_cast<std::size_t>(2 * group + 1)] = zero;
        for (int dimension = begin; dimension < begin + kOscarInt2G128GroupSize; ++dimension) {
            const float normalized = encoded.clipped[static_cast<std::size_t>(dimension)] /
                                         scale +
                                     zero + 0.5F;
            const int code = static_cast<int>(std::floor(normalized));
            if (code < 0 || code > kOscarInt2G128Levels) {
                throw std::runtime_error("OSCAR INT2 quantizer produced an out-of-range code");
            }
            encoded.symbols[static_cast<std::size_t>(dimension)] =
                static_cast<std::uint8_t>(code);
        }
    }

    // Match the official quartered Triton layout: after [4,64] -> [64,4],
    // byte j contains dimensions j, j+64, j+128, j+192.
    for (int byte = 0; byte < kOscarInt2G128CodeBytes; ++byte) {
        encoded.packed[static_cast<std::size_t>(byte)] = static_cast<std::uint8_t>(
            encoded.symbols[static_cast<std::size_t>(byte)] |
            (encoded.symbols[static_cast<std::size_t>(byte + 64)] << 2U) |
            (encoded.symbols[static_cast<std::size_t>(byte + 128)] << 4U) |
            (encoded.symbols[static_cast<std::size_t>(byte + 192)] << 6U));
    }
    return encoded;
}

void oscar_int2_g128_decode(const OscarInt2G128EncodedRow& encoded, float* output, int count) {
    if (output == nullptr || count != kOscarInt2G128HeadDim) {
        throw std::invalid_argument("OSCAR INT2 G128 requires exactly 256 output values");
    }
    for (int dimension = 0; dimension < kOscarInt2G128HeadDim; ++dimension) {
        const int group = dimension / kOscarInt2G128GroupSize;
        const float scale = encoded.scales_zeros[static_cast<std::size_t>(2 * group)];
        const float zero  = encoded.scales_zeros[static_cast<std::size_t>(2 * group + 1)];
        output[dimension] =
            (static_cast<float>(encoded.symbols[static_cast<std::size_t>(dimension)]) - zero) *
            scale;
    }
}

} // namespace ninfer::ops
