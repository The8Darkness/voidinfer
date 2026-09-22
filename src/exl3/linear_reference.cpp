#include "exl3/linear_reference.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::exl3 {
namespace {

constexpr float kHadamardScale = 0.088388347648f;
constexpr std::uint32_t kMul1Multiplier = 0x83DCD12Du;
constexpr std::uint16_t kMul1AccumulatorHalf = 0x6400u;
constexpr std::uint16_t kMul1InverseHalf = 0x1eeeu;
constexpr std::uint16_t kMul1BiasHalf = 0xc931u;

[[noreturn]] void fail(std::string message) {
    throw SafetensorsError(std::move(message));
}

void require(bool condition, std::string message) {
    if (!condition) { fail(std::move(message)); }
}

std::uint16_t read_u16(const std::byte* data) noexcept {
    return static_cast<std::uint16_t>(std::to_integer<unsigned char>(data[0])) |
           static_cast<std::uint16_t>(std::to_integer<unsigned char>(data[1])) << 8u;
}

std::int16_t read_i16(const std::byte* data) noexcept {
    return static_cast<std::int16_t>(read_u16(data));
}

std::int32_t read_i32(const std::byte* data) noexcept {
    const std::uint32_t value = static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[0])) |
                                static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[1])) << 8u |
                                static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[2])) << 16u |
                                static_cast<std::uint32_t>(std::to_integer<unsigned char>(data[3])) << 24u;
    return static_cast<std::int32_t>(value);
}

float half_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign = (value & 0x8000u) << 16u;
    const std::uint32_t exponent = (value >> 10u) & 0x1fu;
    const std::uint32_t fraction = value & 0x03ffu;
    std::uint32_t bits = sign;

    if (exponent == 0) {
        if (fraction != 0) {
            std::uint32_t normalized = fraction;
            std::uint32_t exp = 0;
            while ((normalized & 0x0400u) == 0) {
                normalized <<= 1u;
                ++exp;
            }
            normalized &= 0x03ffu;
            bits |= (127u - 14u - exp) << 23u;
            bits |= normalized << 13u;
        }
    } else if (exponent == 0x1fu) {
        bits |= 0x7f800000u | (fraction << 13u);
    } else {
        bits |= (exponent + (127u - 15u)) << 23u;
        bits |= fraction << 13u;
    }

    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t float_to_half(float value) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16u) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23u) & 0xffu;
    const std::uint32_t fraction = bits & 0x007fffffu;

    if (exponent == 0xffu) {
        if (fraction == 0) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        return static_cast<std::uint16_t>(sign | 0x7e00u | (fraction >> 13u));
    }

    const int unbiased = static_cast<int>(exponent) - 127;
    if (unbiased > 15) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (unbiased >= -14) {
        std::uint32_t half_exp = static_cast<std::uint32_t>(unbiased + 15);
        std::uint32_t half_frac = fraction >> 13u;
        const std::uint32_t remainder = fraction & 0x1fffu;
        if (remainder > 0x1000u || (remainder == 0x1000u && (half_frac & 1u))) {
            ++half_frac;
            if (half_frac == 0x400u) {
                half_frac = 0;
                ++half_exp;
                if (half_exp == 0x1fu) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
            }
        }
        return static_cast<std::uint16_t>(sign | (half_exp << 10u) | half_frac);
    }

    if (unbiased < -25) { return static_cast<std::uint16_t>(sign); }
    const std::uint32_t mantissa = fraction | 0x00800000u;
    const int shift = -unbiased - 14;
    std::uint32_t half_frac = mantissa >> (shift + 13);
    const std::uint32_t remainder_mask = (1u << (shift + 13)) - 1u;
    const std::uint32_t remainder = mantissa & remainder_mask;
    const std::uint32_t halfway = 1u << (shift + 12);
    if (remainder > halfway || (remainder == halfway && (half_frac & 1u))) { ++half_frac; }
    return static_cast<std::uint16_t>(sign | half_frac);
}

std::uint16_t half_mul(std::uint16_t left, std::uint16_t right) noexcept {
    return float_to_half(half_to_float(left) * half_to_float(right));
}

std::uint16_t half_fma(std::uint16_t left, std::uint16_t right, std::uint16_t addend) noexcept {
    return float_to_half(std::fma(half_to_float(left), half_to_float(right), half_to_float(addend)));
}

void hadamard_128(float* values) noexcept {
    for (int width = 1; width < 128; width *= 2) {
        for (int base = 0; base < 128; base += 2 * width) {
            for (int i = 0; i < width; ++i) {
                const float left = values[base + i];
                const float right = values[base + width + i];
                values[base + i] = left + right;
                values[base + width + i] = left - right;
            }
        }
    }
}

constexpr std::array<std::uint16_t, 256> make_tensor_core_perm() {
    std::array<std::uint16_t, 256> permutation{};
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2;
        const int r1 = r0 + 1;
        const int r2 = r0 + 8;
        const int r3 = r0 + 9;
        const int c0 = t / 4;
        const int c1 = c0 + 8;
        permutation[t * 8 + 0] = static_cast<std::uint16_t>(r0 * 16 + c0);
        permutation[t * 8 + 1] = static_cast<std::uint16_t>(r1 * 16 + c0);
        permutation[t * 8 + 2] = static_cast<std::uint16_t>(r2 * 16 + c0);
        permutation[t * 8 + 3] = static_cast<std::uint16_t>(r3 * 16 + c0);
        permutation[t * 8 + 4] = static_cast<std::uint16_t>(r0 * 16 + c1);
        permutation[t * 8 + 5] = static_cast<std::uint16_t>(r1 * 16 + c1);
        permutation[t * 8 + 6] = static_cast<std::uint16_t>(r2 * 16 + c1);
        permutation[t * 8 + 7] = static_cast<std::uint16_t>(r3 * 16 + c1);
    }
    return permutation;
}

std::uint32_t load_u32(const std::uint16_t* words, int index) noexcept {
    return static_cast<std::uint32_t>(words[index * 2]) |
           static_cast<std::uint32_t>(words[index * 2 + 1]) << 16u;
}

std::uint16_t decode_state(const std::uint16_t* packed, int bits, int t_offset) noexcept {
    const int words32 = bits * 256 / 32;
    const int b0 = t_offset * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const std::uint64_t merged = (static_cast<std::uint64_t>(load_u32(packed, (b0 / 32) % words32)) << 32u) |
                                 load_u32(packed, ((b1 - 1) / 32) % words32);
    return static_cast<std::uint16_t>((merged >> shift) & 0xffffu);
}

std::uint16_t decode_mul1(std::uint16_t state) noexcept {
    const std::uint32_t product = static_cast<std::uint32_t>(state) * kMul1Multiplier;
    const std::uint32_t byte_sum = (product & 0xffu) + ((product >> 8u) & 0xffu) +
                                    ((product >> 16u) & 0xffu) + (product >> 24u);
    const auto codebook_input = static_cast<std::uint16_t>(kMul1AccumulatorHalf + byte_sum);
    return half_fma(codebook_input, kMul1InverseHalf, kMul1BiasHalf);
}

void validate_shape(const TensorPayload& tensor, std::string_view name,
                    std::initializer_list<std::uint64_t> expected) {
    require(tensor.info.name == name, "tensor name mismatch for " + std::string(name));
    require(tensor.info.shape == std::vector<std::uint64_t>(expected),
            "tensor shape mismatch for " + std::string(name));
}

} // namespace

void validate_linear_tensors(const Exl3LinearTensors& tensors,
                             const Exl3LinearMetadata& metadata) {
    require(std::endian::native == std::endian::little,
            "EXL3 reference path requires a little-endian host");
    require(tensors.trellis != nullptr, "missing trellis");
    require(tensors.suh != nullptr, "missing suh");
    require(tensors.svh != nullptr, "missing svh");
    if (metadata.mul1) { require(tensors.mul1 != nullptr, "missing mul1 when required"); }
    require(!metadata.mcg, "unsupported EXL3 codebook: mcg");
    require(metadata.mul1, "unsupported EXL3 codebook: mul1 is not enabled");
    require(metadata.codebook == "mul1", "unsupported EXL3 codebook: " + std::string(metadata.codebook));
    require(!metadata.has_bias, "unsupported EXL3 bias");
    require(metadata.K == 5, "unsupported EXL3 module K; E2 reference requires K=5");
    require(metadata.in_features == 5120 && metadata.out_features == 17408,
            "E2 reference dimensions must be 5120 x 17408");

    validate_shape(*tensors.trellis,
                   "model.language_model.layers.5.mlp.gate_proj.trellis",
                   {320, 1088, 80});
    require(tensors.trellis->info.dtype == "I16", "trellis must have dtype I16");
    validate_shape(*tensors.suh,
                   "model.language_model.layers.5.mlp.gate_proj.suh", {5120});
    require(tensors.suh->info.dtype == "F16", "suh must have dtype F16");
    validate_shape(*tensors.svh,
                   "model.language_model.layers.5.mlp.gate_proj.svh", {17408});
    require(tensors.svh->info.dtype == "F16", "svh must have dtype F16");
    validate_shape(*tensors.mul1,
                   "model.language_model.layers.5.mlp.gate_proj.mul1", {});
    require(tensors.mul1->info.dtype == "I32", "mul1 must have dtype I32");
    require(tensors.trellis->bytes().size() == 55705600, "trellis byte size mismatch");
    require(tensors.suh->bytes().size() == 10240, "suh byte size mismatch");
    require(tensors.svh->bytes().size() == 34816, "svh byte size mismatch");
    require(tensors.mul1->bytes().size() == 4, "mul1 byte size mismatch");
}

Exl3ReferenceOutput exl3_linear_reference(const TensorPayload& input,
                                          const Exl3LinearTensors& tensors,
                                          const Exl3LinearMetadata& metadata) {
    validate_linear_tensors(tensors, metadata);
    require(input.info.dtype == "F16", "linear input must have dtype F16");
    require(!input.info.shape.empty() && input.info.shape.back() == 5120,
            "incompatible input dimension");

    std::uint64_t rows_u64 = 1;
    for (std::size_t i = 0; i + 1 < input.info.shape.size(); ++i) {
        if (input.info.shape[i] != 0 && rows_u64 > std::numeric_limits<std::uint64_t>::max() / input.info.shape[i]) {
            fail("input row count overflows u64");
        }
        rows_u64 *= input.info.shape[i];
    }
    require(rows_u64 <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
            "input row count is too large");
    const int rows = static_cast<int>(rows_u64);
    const auto input_bytes = input.bytes();
    require(input_bytes.size() == rows_u64 * 5120u * sizeof(std::uint16_t),
            "linear input payload size mismatch");

    const auto suh = tensors.suh->typed<std::uint16_t>("F16");
    const auto svh = tensors.svh->typed<std::uint16_t>("F16");
    const auto trellis = tensors.trellis->typed<std::uint16_t>("I16");
    const auto* input_half = reinterpret_cast<const std::uint16_t*>(input_bytes.data());

    std::vector<std::uint16_t> transformed(static_cast<std::size_t>(rows) * 5120u);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < 5120; block += 128) {
            std::array<float, 128> values{};
            for (int i = 0; i < 128; ++i) {
                const auto scaled = half_mul(input_half[static_cast<std::size_t>(row) * 5120 + block + i],
                                             suh[block + i]);
                values[i] = half_to_float(scaled);
            }
            hadamard_128(values.data());
            for (int i = 0; i < 128; ++i) {
                transformed[static_cast<std::size_t>(row) * 5120 + block + i] =
                    float_to_half(values[i] * kHadamardScale);
            }
        }
    }

    const int tiles_k = 320;
    const int tiles_n = 1088;
    const int packed_size = 16 * metadata.K;
    std::vector<float> accum(static_cast<std::size_t>(rows) * 17408u, 0.0f);
    constexpr auto permutation = make_tensor_core_perm();
    std::array<std::uint16_t, 256> tile{};

    for (int tile_n = 0; tile_n < tiles_n; ++tile_n) {
        std::fill(tile.begin(), tile.end(), 0);
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k) {
            const auto* packed = trellis.data() +
                (static_cast<std::size_t>(tile_k) * tiles_n + tile_n) * packed_size;
            for (int t = 0; t < 256; ++t) {
                tile[permutation[t]] = decode_mul1(decode_state(packed, metadata.K, t));
            }
            for (int row = 0; row < rows; ++row) {
                const auto* x = transformed.data() + static_cast<std::size_t>(row) * 5120u + tile_k * 16;
                auto* y = accum.data() + static_cast<std::size_t>(row) * 17408u + tile_n * 16;
                for (int r = 0; r < 16; ++r) {
                    const float xv = half_to_float(x[r]);
                    for (int c = 0; c < 16; ++c) {
                        y[c] += xv * half_to_float(tile[r * 16 + c]);
                    }
                }
            }
        }
    }

    Exl3ReferenceOutput output;
    output.shape = input.info.shape;
    output.shape.back() = 17408;
    output.fp16_bits.resize(accum.size());
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < 17408; block += 128) {
            std::array<float, 128> values{};
            std::copy_n(accum.data() + static_cast<std::size_t>(row) * 17408u + block,
                        128, values.data());
            hadamard_128(values.data());
            for (int i = 0; i < 128; ++i) {
                const auto normalized = float_to_half(values[i] * kHadamardScale);
                output.fp16_bits[static_cast<std::size_t>(row) * 17408u + block + i] =
                    half_mul(normalized, svh[block + i]);
            }
        }
    }
    return output;
}

} // namespace ninfer::exl3
