#include "ops/kv_cache/oscar_int2_g128.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define NINFER_OSCAR_VALIDATOR_AVX2 1
#else
#define NINFER_OSCAR_VALIDATOR_AVX2 0
#endif

namespace {

constexpr std::uint32_t kMagic = 0x3342524FU;
constexpr std::uint32_t kVersion = 1U;
constexpr std::uint32_t kD = 256U;
constexpr std::uint32_t kQHeads = 24U;
constexpr std::uint32_t kKVHeads = 4U;
constexpr std::uint32_t kGqa = 6U;
constexpr std::uint32_t kPrefix = 64U;
constexpr std::uint32_t kRecent = 256U;
constexpr std::string_view kIdentity =
    "qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1";
constexpr std::string_view kAssetHash =
    "4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560";
constexpr std::array<std::uint32_t, 16> kLayers = {
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63};

using Clock = std::chrono::steady_clock;

double process_cpu_us() noexcept {
#if defined(_WIN32)
    FILETIME creation{}, exit_time{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit_time, &kernel, &user)) {
        return 0.0;
    }
    ULARGE_INTEGER kernel_ticks{kernel.dwLowDateTime, kernel.dwHighDateTime};
    ULARGE_INTEGER user_ticks{user.dwLowDateTime, user.dwHighDateTime};
    return static_cast<double>(kernel_ticks.QuadPart + user_ticks.QuadPart) / 10.0;
#else
    return 0.0;
#endif
}

template <typename T>
T read_value(std::ifstream& input, const char* label) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) { throw std::runtime_error(std::string("truncated live tap ") + label); }
    return value;
}

template <typename T, std::size_t N>
void read_array(std::ifstream& input, std::array<T, N>& values, const char* label) {
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(
        values.size() * sizeof(T)));
    if (!input) { throw std::runtime_error(std::string("truncated live tap ") + label); }
}

template <typename T>
std::vector<T> read_values(std::ifstream& input, std::size_t count, const char* label) {
    std::vector<T> values(count);
    if (count != 0) {
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(count * sizeof(T)));
        if (!input) { throw std::runtime_error(std::string("truncated live tap ") + label); }
    }
    return values;
}

std::string read_string(std::ifstream& input, const char* label) {
    const std::uint32_t size = read_value<std::uint32_t>(input, label);
    if (size > 4096U) { throw std::runtime_error(std::string("oversized live tap ") + label); }
    std::string value(size, '\0');
    if (size != 0) {
        input.read(value.data(), static_cast<std::streamsize>(size));
        if (!input) { throw std::runtime_error(std::string("truncated live tap ") + label); }
    }
    return value;
}

float bf16_to_float(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

std::uint16_t float_to_bf16(float value) noexcept {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
}

void require_finite(const float* values, std::size_t count, const char* label) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            throw std::runtime_error(std::string(label) + " has NaN/Inf");
        }
    }
}

struct EncodedRow {
    std::array<std::uint8_t, ninfer::ops::kOscarInt2G128CodeBytes> packed{};
    std::array<float, ninfer::ops::kOscarInt2G128MetadataItems> metadata{};
};

struct Row {
    std::uint8_t tier = 0;
    std::array<std::array<float, kD>, kKVHeads> k{};
    std::array<std::array<float, kD>, kKVHeads> v{};
};

struct TapLoadProfile {
    double io_us = 0.0;
    double int2_k_decode_us = 0.0;
    double int2_v_decode_us = 0.0;
};

struct Tap {
    std::filesystem::path path;
    std::uint32_t layer = 0;
    std::uint32_t query = 0;
    std::uint32_t context = 0;
    std::vector<float> q;
    std::vector<std::uint32_t> positions;
    std::vector<Row> rows;
    std::vector<float> actual_q;
    std::vector<float> actual_scores;
    std::vector<float> actual_softmax;
    std::vector<float> actual_av;
    std::vector<float> actual_recovered;
    TapLoadProfile profile;
};

void decode_encoded(const EncodedRow& encoded, float* output) {
    for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
        const std::uint32_t shift = (dimension / 64U) * 2U;
        const std::uint8_t symbol =
            (encoded.packed[dimension % 64U] >> shift) & static_cast<std::uint8_t>(3U);
        const std::uint32_t group = dimension / ninfer::ops::kOscarInt2G128GroupSize;
        output[dimension] =
            (static_cast<float>(symbol) - encoded.metadata[2U * group + 1U]) *
            encoded.metadata[2U * group];
    }
}

Tap read_tap(const std::filesystem::path& path) {
    const auto io_start = Clock::now();
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open live tap: " + path.string()); }
    Tap tap;
    tap.path = path;
    if (read_value<std::uint32_t>(input, "magic") != kMagic ||
        read_value<std::uint32_t>(input, "version") != kVersion) {
        throw std::runtime_error("live tap magic/version mismatch: " + path.string());
    }
    tap.layer = read_value<std::uint32_t>(input, "layer");
    tap.query = read_value<std::uint32_t>(input, "query");
    tap.context = read_value<std::uint32_t>(input, "context");
    const std::uint32_t q_count = read_value<std::uint32_t>(input, "Q count");
    const std::uint32_t position_count = read_value<std::uint32_t>(input, "position count");
    if (q_count != kQHeads * kD || position_count != tap.query + 1U ||
        tap.query >= tap.context || tap.context == 0) {
        throw std::runtime_error("live tap shape/range mismatch: " + path.string());
    }
    if (std::find(kLayers.begin(), kLayers.end(), tap.layer) == kLayers.end()) {
        throw std::runtime_error("live tap contains a non-full-attention layer");
    }
    if (read_string(input, "asset identity") != kIdentity ||
        read_string(input, "asset hash") != kAssetHash) {
        throw std::runtime_error("live tap asset identity/hash mismatch");
    }
    tap.q = read_values<float>(input, q_count, "Q");
    tap.positions = read_values<std::uint32_t>(input, position_count, "positions");
    for (std::uint32_t position = 0; position < position_count; ++position) {
        if (tap.positions[position] != position) {
            throw std::runtime_error("live tap logical positions are not contiguous");
        }
    }
    tap.rows.resize(position_count);
    for (std::uint32_t position = 0; position < position_count; ++position) {
        Row& row = tap.rows[position];
        row.tier = read_value<std::uint8_t>(input, "tier");
        std::array<std::uint8_t, 3> padding{};
        read_array(input, padding, "tier padding");
        if (row.tier > 2U) { throw std::runtime_error("live tap has an invalid tier"); }
        const bool bf16 = row.tier == 0U || row.tier == 2U;
        const std::uint32_t recent_begin = tap.context > kRecent
                                                ? tap.context - kRecent
                                                : kPrefix;
        const std::uint8_t expected = position < kPrefix
                                          ? 0U
                                          : (position < recent_begin ? 1U : 2U);
        if (row.tier != expected) {
            throw std::runtime_error("live tap tier does not match mixed-cache policy");
        }
        for (std::uint32_t head = 0; head < kKVHeads; ++head) {
            if (bf16) {
                std::array<std::uint16_t, kD> k_bits{};
                std::array<std::uint16_t, kD> v_bits{};
                read_array(input, k_bits, "BF16 K row");
                read_array(input, v_bits, "BF16 V row");
                for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
                    row.k[head][dimension] = bf16_to_float(k_bits[dimension]);
                    row.v[head][dimension] = bf16_to_float(v_bits[dimension]);
                }
            } else {
                EncodedRow k_encoded{};
                EncodedRow v_encoded{};
                const auto k_decode_start = Clock::now();
                read_array(input, k_encoded.packed, "INT2 K payload");
                read_array(input, k_encoded.metadata, "INT2 K metadata");
                decode_encoded(k_encoded, row.k[head].data());
                tap.profile.int2_k_decode_us +=
                    std::chrono::duration<double, std::micro>(Clock::now() - k_decode_start).count();
                const auto v_decode_start = Clock::now();
                read_array(input, v_encoded.packed, "INT2 V payload");
                read_array(input, v_encoded.metadata, "INT2 V metadata");
                decode_encoded(v_encoded, row.v[head].data());
                tap.profile.int2_v_decode_us +=
                    std::chrono::duration<double, std::micro>(Clock::now() - v_decode_start).count();
            }
            require_finite(row.k[head].data(), kD, "decoded K row");
            require_finite(row.v[head].data(), kD, "decoded V row");
        }
    }
    const auto read_trace = [&](const char* label) {
        const std::uint64_t count = read_value<std::uint64_t>(input, label);
        if (count > 100000000ULL) { throw std::runtime_error("live tap trace is oversized"); }
        return read_values<float>(input, static_cast<std::size_t>(count), label);
    };
    tap.actual_q = read_trace("rotated Q");
    tap.actual_scores = read_trace("scores");
    tap.actual_softmax = read_trace("softmax");
    tap.actual_av = read_trace("rotated AV");
    tap.actual_recovered = read_trace("recovered output");
    require_finite(tap.q.data(), tap.q.size(), "tap Q");
    require_finite(tap.actual_q.data(), tap.actual_q.size(), "tap rotated Q");
    require_finite(tap.actual_scores.data(), tap.actual_scores.size(), "tap scores");
    require_finite(tap.actual_softmax.data(), tap.actual_softmax.size(), "tap softmax");
    require_finite(tap.actual_av.data(), tap.actual_av.size(), "tap rotated AV");
    require_finite(tap.actual_recovered.data(), tap.actual_recovered.size(), "tap recovered output");
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("live tap has trailing bytes: " + path.string());
    }
    tap.profile.io_us =
        std::chrono::duration<double, std::micro>(Clock::now() - io_start).count();
    return tap;
}

struct RotationBanks {
    std::vector<float> k;
    std::vector<float> v;
    // [input_dimension][output_column] = R_V[output_column][input_dimension]
    // so the inverse/output rotation can use the same contiguous row-wise pass.
    std::vector<float> v_transposed;
};

RotationBanks read_banks(const std::filesystem::path& k_path,
                         const std::filesystem::path& v_path) {
    constexpr std::size_t count = 16ULL * kD * kD;
    const auto read = [&](const std::filesystem::path& path) {
        if (std::filesystem::file_size(path) != count * sizeof(float)) {
            throw std::runtime_error("rotation bank size mismatch: " + path.string());
        }
        std::ifstream input(path, std::ios::binary);
        if (!input) { throw std::runtime_error("cannot open rotation bank: " + path.string()); }
        auto values = read_values<float>(input, count, "rotation bank");
        require_finite(values.data(), values.size(), "rotation bank");
        return values;
    };
    RotationBanks banks{read(k_path), read(v_path), {}};
    banks.v_transposed.resize(count);
    for (std::size_t layer = 0; layer < kLayers.size(); ++layer) {
        const std::size_t base = layer * kD * kD;
        for (std::uint32_t output = 0; output < kD; ++output) {
            for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
                banks.v_transposed[base + static_cast<std::size_t>(dimension) * kD + output] =
                    banks.v[base + static_cast<std::size_t>(output) * kD + dimension];
            }
        }
    }
    return banks;
}

struct Metrics {
    float max_abs = 0.0F;
    double mean_abs = 0.0;
    double relative_l2 = 0.0;
};

Metrics compare(const float* expected, const float* actual, std::size_t count) {
    if (expected == nullptr || actual == nullptr || count == 0) {
        throw std::runtime_error("live tap trace shape mismatch");
    }
    double abs_sum = 0.0;
    double diff2 = 0.0;
    double ref2 = 0.0;
    float max_abs = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
        const float diff = std::abs(expected[index] - actual[index]);
        max_abs = std::max(max_abs, diff);
        abs_sum += diff;
        diff2 += static_cast<double>(diff) * diff;
        ref2 += static_cast<double>(expected[index]) * expected[index];
    }
    return {max_abs, abs_sum / static_cast<double>(count),
            std::sqrt(diff2 / std::max(ref2, std::numeric_limits<double>::min()))};
}

float dot256_scalar(const float* left, const float* right) noexcept {
    float sum = 0.0F;
    for (std::uint32_t i = 0; i < kD; ++i) { sum += left[i] * right[i]; }
    return sum;
}

#if NINFER_OSCAR_VALIDATOR_AVX2
float dot256_fast(const float* left, const float* right) noexcept {
    // Use AVX2 for the eight independent products, then accumulate those
    // products in the scalar left-to-right order used by the golden oracle.
    // This preserves the qualification contract while still eliminating the
    // scalar multiply loop in the rotation/inverse-rotation hot paths.
    alignas(32) float products[8];
    float sum = 0.0F;
    for (std::uint32_t i = 0; i < kD; i += 8U) {
        _mm256_store_ps(products, _mm256_mul_ps(_mm256_loadu_ps(left + i),
                                                _mm256_loadu_ps(right + i)));
        for (std::uint32_t lane = 0; lane < 8U; ++lane) {
            sum += products[lane];
        }
    }
    return sum;
}

void rotate_row_fast(const float* input, const float* matrix_rows, float* output) noexcept {
    // matrix_rows is laid out [input_dimension][output_column].  The inner
    // scalar update keeps each output column's exact left-to-right reduction
    // while AVX2 performs eight independent products at a time.
    alignas(32) float products[8];
    for (std::uint32_t column = 0; column < kD; column += 8U) {
        float sums[8] = {};
        for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
            const __m256 product = _mm256_mul_ps(
                _mm256_set1_ps(input[dimension]),
                _mm256_loadu_ps(matrix_rows + static_cast<std::size_t>(dimension) * kD + column));
            _mm256_store_ps(products, product);
            for (std::uint32_t lane = 0; lane < 8U; ++lane) {
                sums[lane] += products[lane];
            }
        }
        for (std::uint32_t lane = 0; lane < 8U; ++lane) {
            output[column + lane] = sums[lane];
        }
    }
}
#else
float dot256_fast(const float* left, const float* right) noexcept {
    return dot256_scalar(left, right);
}

void rotate_row_fast(const float* input, const float* matrix_rows, float* output) noexcept {
    for (std::uint32_t column = 0; column < kD; ++column) {
        float sum = 0.0F;
        for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
            sum += input[dimension] *
                matrix_rows[static_cast<std::size_t>(dimension) * kD + column];
        }
        output[column] = sum;
    }
}
#endif

void rotate_row_scalar(const float* input, const float* matrix, bool transpose, float* output) {
    for (std::uint32_t column = 0; column < kD; ++column) {
        float sum = 0.0F;
        for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
            const float matrix_value = transpose
                ? matrix[static_cast<std::size_t>(column) * kD + dimension]
                : matrix[static_cast<std::size_t>(dimension) * kD + column];
            sum += input[dimension] * matrix_value;
        }
        output[column] = sum;
    }
}

struct ReferenceOutput {
    std::vector<float> rotated_q;
    std::vector<float> scores;
    std::vector<float> softmax;
    std::vector<float> av;
    std::vector<float> recovered;
};

ReferenceOutput scalar_compute(const Tap& tap, const RotationBanks& banks) {
    const std::size_t bank_offset = static_cast<std::size_t>(
        std::distance(kLayers.begin(), std::find(kLayers.begin(), kLayers.end(), tap.layer))) * kD * kD;
    const std::uint32_t count = tap.query + 1U;
    ReferenceOutput result{
        std::vector<float>(kQHeads * kD), std::vector<float>(kQHeads * count),
        std::vector<float>(kQHeads * count), std::vector<float>(kQHeads * kD),
        std::vector<float>(kQHeads * kD)};
    for (std::uint32_t head = 0; head < kQHeads; ++head) {
        rotate_row_scalar(tap.q.data() + static_cast<std::size_t>(head) * kD,
                          banks.k.data() + bank_offset, false,
                          result.rotated_q.data() + static_cast<std::size_t>(head) * kD);
    }
    for (std::uint32_t head = 0; head < kQHeads; ++head) {
        const std::uint32_t kv_head = head / kGqa;
        float* scores = result.scores.data() + static_cast<std::size_t>(head) * count;
        float* probabilities = result.softmax.data() + static_cast<std::size_t>(head) * count;
        float* av = result.av.data() + static_cast<std::size_t>(head) * kD;
        std::fill(av, av + kD, 0.0F);
        for (std::uint32_t position = 0; position < count; ++position) {
            scores[position] = dot256_scalar(
                result.rotated_q.data() + static_cast<std::size_t>(head) * kD,
                tap.rows[position].k[kv_head].data()) / 16.0F;
        }
        const float maximum = *std::max_element(scores, scores + count);
        float sum = 0.0F;
        for (std::uint32_t position = 0; position < count; ++position) {
            probabilities[position] = std::exp(scores[position] - maximum);
            sum += probabilities[position];
        }
        for (std::uint32_t position = 0; position < count; ++position) {
            probabilities[position] /= sum;
            const float* value = tap.rows[position].v[kv_head].data();
            for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
                av[dimension] += probabilities[position] * value[dimension];
            }
        }
        rotate_row_scalar(av, banks.v.data() + bank_offset, true,
                          result.recovered.data() + static_cast<std::size_t>(head) * kD);
    }
    return result;
}

struct FastWorkspace {
    std::vector<float> rotated_q;
    std::vector<float> scores;
    std::vector<float> softmax;
    std::vector<float> av;
    std::vector<float> recovered;

    double ensure(std::uint32_t count) {
        const auto start = Clock::now();
        rotated_q.resize(kQHeads * kD);
        scores.resize(static_cast<std::size_t>(kQHeads) * count);
        softmax.resize(scores.size());
        av.resize(kQHeads * kD);
        recovered.resize(kQHeads * kD);
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    }
};

struct FastProfile {
    double allocation_us = 0.0;
    double q_rotation_us = 0.0;
    double qk_us = 0.0;
    double softmax_us = 0.0;
    double av_us = 0.0;
    double rv_inverse_us = 0.0;
    double total_us = 0.0;
};

struct FastResult {
    std::array<Metrics, 5> metrics{};
    FastProfile profile{};
};

FastResult fast_compute(const Tap& tap, const RotationBanks& banks, FastWorkspace& workspace) {
    const auto total_start = Clock::now();
    const std::size_t bank_offset = static_cast<std::size_t>(
        std::distance(kLayers.begin(), std::find(kLayers.begin(), kLayers.end(), tap.layer))) * kD * kD;
    const std::uint32_t count = tap.query + 1U;
    FastResult result;
    result.profile.allocation_us = workspace.ensure(count);

    auto phase_start = Clock::now();
    for (std::uint32_t head = 0; head < kQHeads; ++head) {
        rotate_row_fast(tap.q.data() + static_cast<std::size_t>(head) * kD,
                        banks.k.data() + bank_offset,
                        workspace.rotated_q.data() + static_cast<std::size_t>(head) * kD);
    }
    result.profile.q_rotation_us =
        std::chrono::duration<double, std::micro>(Clock::now() - phase_start).count();

    phase_start = Clock::now();
    for (std::uint32_t kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (std::uint32_t position = 0; position < count; ++position) {
#if NINFER_OSCAR_VALIDATOR_AVX2
            alignas(32) float products[8];
            float sums[kGqa] = {};
            const float* key = tap.rows[position].k[kv_head].data();
            for (std::uint32_t dimension = 0; dimension < kD; dimension += 8U) {
                const __m256 key_values = _mm256_loadu_ps(key + dimension);
                for (std::uint32_t slot = 0; slot < kGqa; ++slot) {
                    const float* query = workspace.rotated_q.data() +
                        static_cast<std::size_t>(kv_head * kGqa + slot) * kD + dimension;
                    _mm256_store_ps(products, _mm256_mul_ps(
                        key_values, _mm256_loadu_ps(query)));
                    for (std::uint32_t lane = 0; lane < 8U; ++lane) {
                        sums[slot] += products[lane];
                    }
                }
            }
            for (std::uint32_t slot = 0; slot < kGqa; ++slot) {
                workspace.scores[static_cast<std::size_t>(kv_head * kGqa + slot) * count + position] =
                    sums[slot] * (1.0F / 16.0F);
            }
#else
            for (std::uint32_t slot = 0; slot < kGqa; ++slot) {
                workspace.scores[static_cast<std::size_t>(kv_head * kGqa + slot) * count + position] =
                    dot256_scalar(workspace.rotated_q.data() +
                                    static_cast<std::size_t>(kv_head * kGqa + slot) * kD,
                                tap.rows[position].k[kv_head].data()) / 16.0F;
            }
#endif
        }
    }
    result.profile.qk_us =
        std::chrono::duration<double, std::micro>(Clock::now() - phase_start).count();

    phase_start = Clock::now();
    for (std::uint32_t head = 0; head < kQHeads; ++head) {
        float* scores = workspace.scores.data() + static_cast<std::size_t>(head) * count;
        float* probabilities = workspace.softmax.data() + static_cast<std::size_t>(head) * count;
        const float maximum = *std::max_element(scores, scores + count);
        float sum = 0.0F;
        for (std::uint32_t position = 0; position < count; ++position) {
            probabilities[position] = std::exp(scores[position] - maximum);
            sum += probabilities[position];
        }
        if (!std::isfinite(sum) || sum <= 0.0F) {
            throw std::runtime_error("optimized OSCAR validator softmax sum is invalid");
        }
        for (std::uint32_t position = 0; position < count; ++position) {
            probabilities[position] /= sum;
        }
    }
    result.profile.softmax_us =
        std::chrono::duration<double, std::micro>(Clock::now() - phase_start).count();

    phase_start = Clock::now();
    std::fill(workspace.av.begin(), workspace.av.end(), 0.0F);
    for (std::uint32_t kv_head = 0; kv_head < kKVHeads; ++kv_head) {
        for (std::uint32_t position = 0; position < count; ++position) {
            const float* value = tap.rows[position].v[kv_head].data();
            for (std::uint32_t slot = 0; slot < kGqa; ++slot) {
                float* output = workspace.av.data() +
                    static_cast<std::size_t>(kv_head * kGqa + slot) * kD;
#if NINFER_OSCAR_VALIDATOR_AVX2
                const __m256 weight = _mm256_set1_ps(
                    workspace.softmax[static_cast<std::size_t>(kv_head * kGqa + slot) * count + position]);
                for (std::uint32_t dimension = 0; dimension < kD; dimension += 8U) {
                    _mm256_storeu_ps(output + dimension, _mm256_add_ps(
                        _mm256_loadu_ps(output + dimension),
                        _mm256_mul_ps(weight, _mm256_loadu_ps(value + dimension))));
                }
#else
                const float weight = workspace.softmax[
                    static_cast<std::size_t>(kv_head * kGqa + slot) * count + position];
                for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
                    output[dimension] += weight * value[dimension];
                }
#endif
            }
        }
    }
    result.profile.av_us =
        std::chrono::duration<double, std::micro>(Clock::now() - phase_start).count();

    phase_start = Clock::now();
    for (std::uint32_t head = 0; head < kQHeads; ++head) {
        rotate_row_fast(workspace.av.data() + static_cast<std::size_t>(head) * kD,
                        banks.v_transposed.data() + bank_offset,
                        workspace.recovered.data() + static_cast<std::size_t>(head) * kD);
    }
    result.profile.rv_inverse_us =
        std::chrono::duration<double, std::micro>(Clock::now() - phase_start).count();
    result.profile.total_us =
        std::chrono::duration<double, std::micro>(Clock::now() - total_start).count();

    if (!tap.actual_q.empty()) {
        const std::size_t q_values = kQHeads * kD;
        const std::size_t attention_values = static_cast<std::size_t>(kQHeads) * count;
        result.metrics[0] = compare(workspace.rotated_q.data(), tap.actual_q.data(), q_values);
        result.metrics[1] = compare(workspace.scores.data(), tap.actual_scores.data(), attention_values);
        result.metrics[2] = compare(workspace.softmax.data(), tap.actual_softmax.data(), attention_values);
        result.metrics[3] = compare(workspace.av.data(), tap.actual_av.data(), q_values);
        result.metrics[4] = compare(workspace.recovered.data(), tap.actual_recovered.data(), q_values);
    }
    return result;
}

void fill_fixture(Tap& tap, std::uint32_t layer, std::uint32_t context) {
    tap.layer = layer;
    tap.query = context - 1U;
    tap.context = context;
    tap.q.resize(kQHeads * kD);
    tap.rows.resize(context);
    for (std::uint32_t head = 0; head < kQHeads; ++head) {
        for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
            const float value = static_cast<float>(
                static_cast<int>((head * 17U + dimension * 13U) % 61U) - 30) * 0.03125F;
            tap.q[static_cast<std::size_t>(head) * kD + dimension] = value;
        }
    }
    const std::uint32_t recent_begin = context > kRecent ? context - kRecent : kPrefix;
    for (std::uint32_t position = 0; position < context; ++position) {
        tap.rows[position].tier = position < kPrefix ? 0U : (position < recent_begin ? 1U : 2U);
        for (std::uint32_t head = 0; head < kKVHeads; ++head) {
            std::array<float, kD> source_k{};
            std::array<float, kD> source_v{};
            for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
                source_k[dimension] = static_cast<float>(
                    static_cast<int>((position * 7U + head * 19U + dimension * 3U) % 97U) - 48) *
                    0.015625F;
                source_v[dimension] = static_cast<float>(
                    static_cast<int>((position * 11U + head * 23U + dimension * 5U) % 89U) - 44) *
                    0.015625F;
            }
            if (tap.rows[position].tier == 1U) {
                const auto encoded_k = ninfer::ops::oscar_int2_g128_encode(source_k.data(), kD, 0.96F);
                const auto encoded_v = ninfer::ops::oscar_int2_g128_encode(source_v.data(), kD, 0.92F);
                ninfer::ops::oscar_int2_g128_decode(encoded_k, tap.rows[position].k[head].data(), kD);
                ninfer::ops::oscar_int2_g128_decode(encoded_v, tap.rows[position].v[head].data(), kD);
            } else {
                for (std::uint32_t dimension = 0; dimension < kD; ++dimension) {
                    tap.rows[position].k[head][dimension] =
                        bf16_to_float(float_to_bf16(source_k[dimension]));
                    tap.rows[position].v[head][dimension] =
                        bf16_to_float(float_to_bf16(source_v[dimension]));
                }
            }
        }
    }
}

void require_close(const ReferenceOutput& scalar, const FastWorkspace& fast,
                   const char* label) {
    constexpr std::array<const char*, 5> labels = {
        "rotated_q", "scores", "softmax", "av", "recovered"};
    const std::array<std::pair<const std::vector<float>*, const std::vector<float>*>, 5> stages = {{
        {&scalar.rotated_q, &fast.rotated_q}, {&scalar.scores, &fast.scores},
        {&scalar.softmax, &fast.softmax}, {&scalar.av, &fast.av},
        {&scalar.recovered, &fast.recovered}}};
    for (std::size_t stage = 0; stage < stages.size(); ++stage) {
        const auto& [expected, actual] = stages[stage];
        const Metrics metrics = compare(expected->data(), actual->data(), expected->size());
        if (metrics.relative_l2 > 2.0e-6 || metrics.max_abs > 1.0e-4F) {
            throw std::runtime_error(std::string("optimized/scalar golden mismatch at ") + label +
                                     " stage=" + labels[stage] +
                                     " rel_l2=" + std::to_string(metrics.relative_l2) +
                                     " max_abs=" + std::to_string(metrics.max_abs));
        }
    }
}

void run_scalar_golden(const RotationBanks& banks) {
    constexpr std::array<std::uint32_t, 7> contexts = {64, 320, 321, 332, 512, 2048, 4096};
    constexpr std::array<std::uint32_t, 3> layers = {3, 35, 63};
    FastWorkspace workspace;
    std::size_t checks = 0;
    for (const std::uint32_t context : contexts) {
        for (const std::uint32_t layer : layers) {
            Tap tap;
            fill_fixture(tap, layer, context);
            const auto scalar = scalar_compute(tap, banks);
            (void)fast_compute(tap, banks, workspace);
            require_close(scalar, workspace,
                          (std::to_string(context) + "/" + std::to_string(layer)).c_str());
            ++checks;
        }
    }
    std::cout << "OSCAR optimized validator scalar-golden: PASS checks=" << checks
              << " contexts=64,320,321,332,512,2048,4096 layers=3,35,63"
              << " tolerance_rel_l2=2e-6 tolerance_max_abs=1e-4"
              << " avx2=" << (NINFER_OSCAR_VALIDATOR_AVX2 ? "true" : "false") << '\n';
}

void run_scalar_golden_one(const RotationBanks& banks, std::uint32_t context,
                           std::uint32_t layer) {
    FastWorkspace workspace;
    Tap tap;
    fill_fixture(tap, layer, context);
    const auto scalar = scalar_compute(tap, banks);
    (void)fast_compute(tap, banks, workspace);
    require_close(scalar, workspace, "single");
    std::cout << "OSCAR optimized validator scalar-golden: PASS context=" << context
              << " layer=" << layer << '\n';
}

void run_oracle_benchmark(const RotationBanks& banks) {
    constexpr std::array<std::uint32_t, 3> contexts = {512, 2048, 4096};
    FastWorkspace workspace;
    for (const std::uint32_t context : contexts) {
        Tap tap;
        fill_fixture(tap, kLayers.front(), context);
        FastProfile total{};
        const auto wall_start = Clock::now();
        for (const std::uint32_t layer : kLayers) {
            tap.layer = layer;
            const FastResult result = fast_compute(tap, banks, workspace);
            total.allocation_us += result.profile.allocation_us;
            total.q_rotation_us += result.profile.q_rotation_us;
            total.qk_us += result.profile.qk_us;
            total.softmax_us += result.profile.softmax_us;
            total.av_us += result.profile.av_us;
            total.rv_inverse_us += result.profile.rv_inverse_us;
            total.total_us += result.profile.total_us;
        }
        const double wall_us =
            std::chrono::duration<double, std::micro>(Clock::now() - wall_start).count();
        std::cout << std::setprecision(9)
                  << "OSCAR optimized oracle benchmark context=" << context
                  << " taps=16 layers=all-full synthetic_fixture=true wall_us=" << wall_us
                  << " allocation_resize_us=" << total.allocation_us
                  << " q_rotation_us=" << total.q_rotation_us
                  << " qk_us=" << total.qk_us
                  << " softmax_us=" << total.softmax_us
                  << " av_us=" << total.av_us
                  << " rv_inverse_us=" << total.rv_inverse_us
                  << " compute_us=" << total.total_us << '\n';
    }
}

struct TapResult {
    FastResult fast;
    TapLoadProfile load;
    std::uint32_t layer = 0;
    std::uint32_t query = 0;
    std::uint32_t context = 0;
};

void validate_taps(const std::filesystem::path& tap_dir, const RotationBanks& banks) {
    const auto validation_start = Clock::now();
    const double cpu_start_us = process_cpu_us();
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(tap_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) { throw std::runtime_error("no live reference taps found"); }

    std::vector<TapResult> results(paths.size());
    std::atomic<std::size_t> next{0};
    std::mutex failure_mutex;
    std::string failure;
    const std::size_t hardware = std::max(1U, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min<std::size_t>({hardware, 8U, paths.size()});
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&] {
            FastWorkspace workspace;
            for (;;) {
                const std::size_t index = next.fetch_add(1);
                if (index >= paths.size()) { break; }
                try {
                    Tap tap = read_tap(paths[index]);
                    results[index].load = tap.profile;
                    results[index].layer = tap.layer;
                    results[index].query = tap.query;
                    results[index].context = tap.context;
                    results[index].fast = fast_compute(tap, banks, workspace);
                    for (std::size_t stage = 0; stage < results[index].fast.metrics.size(); ++stage) {
                        const auto& metrics = results[index].fast.metrics[stage];
                        if (metrics.relative_l2 > 2.0e-6 || metrics.max_abs > 1.0e-4F) {
                            throw std::runtime_error("live/reference mismatch at stage " +
                                                     std::to_string(stage) + " in " +
                                                     paths[index].string());
                        }
                    }
                } catch (const std::exception& error) {
                    std::lock_guard lock(failure_mutex);
                    if (failure.empty()) { failure = error.what(); }
                }
            }
        });
    }
    for (auto& worker : workers) { worker.join(); }
    if (!failure.empty()) { throw std::runtime_error(failure); }
    const double wall_us = std::chrono::duration<double, std::micro>(
        Clock::now() - validation_start).count();
    const double cpu_end_us = process_cpu_us();
    const double cpu_us = cpu_end_us >= cpu_start_us ? cpu_end_us - cpu_start_us : 0.0;

    const std::array<const char*, 5> labels = {
        "rotated_q", "scores", "softmax", "rotated_av", "recovered"};
    std::array<double, 9> totals{};
    std::array<double, 5> worst_rel{};
    std::array<float, 5> worst_abs{};
    for (std::size_t index = 0; index < paths.size(); ++index) {
        const auto& tap = paths[index];
        const auto& result = results[index];
        totals[0] += result.load.io_us;
        totals[1] += result.load.int2_k_decode_us;
        totals[2] += result.load.int2_v_decode_us;
        totals[3] += result.fast.profile.allocation_us;
        totals[4] += result.fast.profile.q_rotation_us;
        totals[5] += result.fast.profile.qk_us;
        totals[6] += result.fast.profile.softmax_us;
        totals[7] += result.fast.profile.av_us;
        totals[8] += result.fast.profile.rv_inverse_us;
        for (std::size_t stage = 0; stage < 5; ++stage) {
            worst_rel[stage] = std::max(worst_rel[stage], result.fast.metrics[stage].relative_l2);
            worst_abs[stage] = std::max(worst_abs[stage], result.fast.metrics[stage].max_abs);
        }
        const std::uint32_t prefix = std::min(result.context, kPrefix);
        const std::uint32_t recent_begin = result.context > kRecent
                                                ? result.context - kRecent
                                                : kPrefix;
        const std::uint32_t historical = recent_begin > prefix ? recent_begin - prefix : 0U;
        const std::uint32_t recent = result.query + 1U > recent_begin
                                         ? result.query + 1U - recent_begin
                                         : 0U;
        const auto worst_it = std::max_element(
            result.fast.metrics.begin(), result.fast.metrics.end(),
            [](const Metrics& left, const Metrics& right) {
                return left.relative_l2 < right.relative_l2;
            });
        std::cout << "tap=" << tap.filename().string() << " layer=" << result.layer
                  << " query=" << result.query << " context=" << result.context
                  << " prefix=" << prefix << " historical=" << historical
                  << " recent=" << recent << " worst_rel_l2="
                  << worst_it->relative_l2 << " PASS\n";
    }
    std::cout << std::setprecision(9)
              << "OSCAR optimized full validator: PASS taps=" << paths.size()
              << " workers=" << worker_count
              << " avx2=" << (NINFER_OSCAR_VALIDATOR_AVX2 ? "true" : "false") << '\n'
              << "OSCAR validator timing_us: wall=" << wall_us
              << " process_cpu=" << cpu_us
              << " process_cpu_utilization_pct="
              << (wall_us > 0.0 ? (100.0 * cpu_us / wall_us) : 0.0) << '\n'
              << "OSCAR validator profile_us: file_io_inclusive_decode=" << totals[0]
              << " int2_k_decode=" << totals[1]
              << " int2_v_decode=" << totals[2]
              << " allocation_resize=" << totals[3]
              << " q_rotation=" << totals[4]
              << " qk=" << totals[5]
              << " softmax=" << totals[6]
              << " av=" << totals[7]
              << " rv_inverse=" << totals[8]
              << " synchronization=0\n";
    for (std::size_t stage = 0; stage < labels.size(); ++stage) {
        std::cout << "OSCAR validator worst stage=" << labels[stage]
                  << " max_abs=" << worst_abs[stage]
                  << " relative_l2=" << worst_rel[stage] << " PASS\n";
    }
    std::cout << "OSCAR D2.3b independent live/reference parity: PASS taps=" << paths.size()
              << " worst_relative_l2=" << *std::max_element(worst_rel.begin(), worst_rel.end())
              << " legacy_q2=false bf16_history_shadow=false\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string_view(argv[1]) == "--self-check") {
            const auto banks = read_banks(argv[2], argv[3]);
            run_scalar_golden(banks);
            return 0;
        }
        if (argc == 6 && std::string_view(argv[1]) == "--golden-one") {
            const auto banks = read_banks(argv[4], argv[5]);
            run_scalar_golden_one(banks, static_cast<std::uint32_t>(std::stoul(argv[2])),
                                  static_cast<std::uint32_t>(std::stoul(argv[3])));
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "--benchmark") {
            const auto banks = read_banks(argv[2], argv[3]);
            run_oracle_benchmark(banks);
            return 0;
        }
        if (argc != 4) {
            std::cerr << "usage: validator <tap-dir> <k-bank.bin> <v-bank.bin>\n"
                      << "       validator --self-check <k-bank.bin> <v-bank.bin>\n"
                      << "       validator --golden-one <context> <layer> <k-bank.bin> <v-bank.bin>\n"
                      << "       validator --benchmark <k-bank.bin> <v-bank.bin>\n";
            return 2;
        }
        const auto banks = read_banks(argv[2], argv[3]);
        validate_taps(argv[1], banks);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OSCAR optimized independent validator: FAIL: " << error.what() << '\n';
        return 1;
    }
}
