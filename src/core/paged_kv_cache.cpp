#include "core/paged_kv_cache.h"

#include "core/device.h"
#include "core/host_kv_arena.h"
#include "ops/kv_cache/d256_profile.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer {
namespace {

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must fit positive int32");
    }
    return static_cast<std::int32_t>(value);
}

std::size_t checked_table_bytes(const KVExecutionTableSpec& spec) {
    const std::size_t logical =
        checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity");
    if (spec.table_rows <= 0) {
        throw std::invalid_argument("Paged KV table row count must be positive");
    }
    const std::size_t rows = static_cast<std::size_t>(spec.table_rows);
    if (logical > std::numeric_limits<std::size_t>::max() / rows / sizeof(std::int32_t)) {
        throw std::overflow_error("Paged KV execution table size overflow");
    }
    return logical * rows * sizeof(std::int32_t);
}

void validate_geometry(const KVPageGeometry& geometry) {
    if (geometry.page_tokens != static_cast<std::uint32_t>(kPagedKVPageSize)) {
        throw std::invalid_argument("Paged KV device geometry requires 64-token pages");
    }
    if (geometry.planes.empty()) { throw std::invalid_argument("Paged KV geometry has no planes"); }
    if (!is_valid_oscar_kv_layout(geometry.oscar_layout)) {
        throw std::invalid_argument("Paged KV geometry has an invalid OSCAR layout");
    }
    for (const KVPlaneGeometry& plane : geometry.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Paged KV plane extents must be positive");
        }
    }
}

void increment_generation(std::uint32_t& generation) noexcept {
    ++generation;
    if (generation == 0) { ++generation; }
}

std::uint32_t float_bits(float value) noexcept {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(std::uint32_t bits) noexcept {
    float value = 0.0F;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint16_t float_to_fp16_bits(float value) noexcept {
    const std::uint32_t bits      = float_bits(value);
    const std::uint16_t sign      = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const std::uint32_t exponent = (bits >> 23U) & 0xffU;
    std::uint32_t mantissa       = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        if (mantissa == 0) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
        mantissa >>= 13U;
        return static_cast<std::uint16_t>(sign | 0x7c00U | (mantissa == 0 ? 1U : mantissa));
    }

    int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent <= 0) {
        if (half_exponent < -10) { return sign; }
        mantissa |= 0x800000U;
        const int shift = 14 - half_exponent;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
        const std::uint32_t halfway   = 1U << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1U) != 0)) {
            ++half_mantissa;
        }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    if (half_exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }

    std::uint32_t half_mantissa = mantissa >> 13U;
    const std::uint32_t remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
        ++half_mantissa;
        if (half_mantissa == 0x400U) {
            half_mantissa = 0;
            ++half_exponent;
            if (half_exponent >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00U); }
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(half_exponent) << 10U) |
                                       half_mantissa);
}

float fp16_bits_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign     = (static_cast<std::uint32_t>(value & 0x8000U) << 16U);
    std::uint32_t exponent       = (value >> 10U) & 0x1fU;
    std::uint32_t mantissa       = value & 0x3ffU;
    std::uint32_t float_exponent = 0;
    if (exponent == 0) {
        if (mantissa == 0) { return bits_float(sign); }
        int unbiased = -14;
        while ((mantissa & 0x400U) == 0) {
            mantissa <<= 1U;
            --unbiased;
        }
        mantissa &= 0x3ffU;
        float_exponent = static_cast<std::uint32_t>(unbiased + 127);
    } else if (exponent == 0x1fU) {
        float_exponent = 0xffU;
    } else {
        float_exponent = exponent - 15U + 127U;
    }
    return bits_float(sign | (float_exponent << 23U) | (mantissa << 13U));
}

std::uint16_t bf16_to_fp16(std::uint16_t value) noexcept {
    return float_to_fp16_bits(bits_float(static_cast<std::uint32_t>(value) << 16U));
}

std::uint16_t fp16_to_bf16(std::uint16_t value) noexcept {
    const std::uint32_t bits = float_bits(fp16_bits_to_float(value));
    const std::uint32_t bias = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

bool host_uses_fp16_for_bf16(const HostKVPageLayout& host,
                             const std::vector<Tensor>& device_planes) {
    if (host.storage_dtypes.empty()) { return false; }
    if (host.storage_dtypes.size() != device_planes.size() ||
        host.planes.size() != device_planes.size()) {
        throw std::invalid_argument("Paged KV host storage dtype inventory is inconsistent");
    }
    bool converts = false;
    for (std::size_t index = 0; index < device_planes.size(); ++index) {
        const DType source = device_planes[index].dtype;
        const DType target = host.storage_dtypes[index];
        if (source == target) { continue; }
        if (source != DType::BF16 || target != DType::FP16) {
            throw std::invalid_argument(
                "Paged KV only supports BF16-to-FP16 host storage conversion");
        }
        converts = true;
    }
    return converts;
}

void convert_host_records_bf16_to_fp16(std::byte* records, std::uint32_t pages,
                                       const HostKVPageLayout& host,
                                       const std::vector<Tensor>& device_planes) {
    for (std::uint32_t page = 0; page < pages; ++page) {
        for (std::size_t index = 0; index < device_planes.size(); ++index) {
            if (device_planes[index].dtype != DType::BF16 ||
                host.storage_dtypes[index] != DType::FP16) {
                continue;
            }
            const HostKVPlaneLayout& plane = host.planes[index];
            auto* values = reinterpret_cast<std::uint16_t*>(
                records + static_cast<std::size_t>(page) * host.page_stride + plane.offset);
            const std::size_t count = plane.page_payload_bytes / sizeof(std::uint16_t);
            for (std::size_t element = 0; element < count; ++element) {
                values[element] = bf16_to_fp16(values[element]);
            }
        }
    }
}

void convert_host_records_fp16_to_bf16(std::byte* records, std::uint32_t pages,
                                       const HostKVPageLayout& host,
                                       const std::vector<Tensor>& device_planes) {
    for (std::uint32_t page = 0; page < pages; ++page) {
        for (std::size_t index = 0; index < device_planes.size(); ++index) {
            if (device_planes[index].dtype != DType::BF16 ||
                host.storage_dtypes[index] != DType::FP16) {
                continue;
            }
            const HostKVPlaneLayout& plane = host.planes[index];
            auto* values = reinterpret_cast<std::uint16_t*>(
                records + static_cast<std::size_t>(page) * host.page_stride + plane.offset);
            const std::size_t count = plane.page_payload_bytes / sizeof(std::uint16_t);
            for (std::size_t element = 0; element < count; ++element) {
                values[element] = fp16_to_bf16(values[element]);
            }
        }
    }
}

bool is_hierarchical_oscar_host_layout(const HostKVPageLayout& layout) noexcept {
    return layout.storage_format == HostKVStorageFormat::OscarQ4AndFp16;
}

std::uint16_t float_to_bf16_bits(float value) noexcept {
    const std::uint32_t bits = float_bits(value);
    const std::uint32_t bias = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

float bf16_bits_to_float(std::uint16_t value) noexcept {
    return bits_float(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t load_u16(const std::byte* address) noexcept {
    std::uint16_t value = 0;
    std::memcpy(&value, address, sizeof(value));
    return value;
}

void store_u16(std::byte* address, std::uint16_t value) noexcept {
    std::memcpy(address, &value, sizeof(value));
}

// CPU mirror of normalized_hadamard_d256_inplace().  Host conversion is deliberately off the
// serving stream; matching the device butterfly order keeps L2 restores deterministic without
// requiring a temporary FP16 allocation on the GPU.
void normalized_hadamard_d256_host(std::array<float, 256>& values) noexcept {
    for (int block = 0; block < 8; ++block) {
        float* column = values.data() + block * 32;
        for (int stride = 1; stride <= 16; stride <<= 1) {
            for (int base = 0; base < 32; base += 2 * stride) {
                for (int offset = 0; offset < stride; ++offset) {
                    const float low  = column[base + offset];
                    const float high = column[base + offset + stride];
                    column[base + offset]        = low + high;
                    column[base + offset + stride] = low - high;
                }
            }
        }
    }
    for (int stride = 1; stride < 8; stride <<= 1) {
        for (int base = 0; base < 8; base += 2 * stride) {
            for (int offset = 0; offset < stride; ++offset) {
                for (int dimension = 0; dimension < 32; ++dimension) {
                    const int low_index  = (base + offset) * 32 + dimension;
                    const int high_index = (base + offset + stride) * 32 + dimension;
                    const float low       = values[low_index];
                    const float high      = values[high_index];
                    values[low_index]     = low + high;
                    values[high_index]    = low - high;
                }
            }
        }
    }
    for (float& value : values) { value *= 0.0625F; }
}

struct OscarHostQuantParams {
    float scale = 1.0F;
    float zero  = 0.0F;
};

template <bool IsValue>
OscarHostQuantParams oscar_host_quant_params(const std::array<float, 256>& values,
                                             int bits) noexcept {
    float minimum = values[0];
    float maximum = values[0];
    for (float value : values) {
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    const float ratio = bits == 2 ? (IsValue ? 0.91F : 0.93F) :
                         bits == 4 ? (IsValue ? 0.98F : 0.985F) : 0.95F;
    const float center    = 0.5F * (minimum + maximum);
    const float half_span = 0.5F * (maximum - minimum) * ratio;
    const float zero      = center - half_span;
    const float span      = 2.0F * half_span;
    return {.scale = span > 1.0e-8F ? span / static_cast<float>((1 << bits) - 1) : 1.0F,
            .zero  = zero};
}

int oscar_host_quantize(float value, OscarHostQuantParams params, int bits) noexcept {
    const float normalized = (value - params.zero) / params.scale;
    const int rounded       = static_cast<int>(std::lrintf(normalized));
    return std::clamp(rounded, 0, (1 << bits) - 1);
}

void pack_oscar_host_codes(const std::array<float, 256>& values, OscarHostQuantParams params,
                           int bits, std::byte* destination) noexcept {
    const int code_extent = (256 * bits + 7) / 8;
    std::memset(destination, 0, static_cast<std::size_t>(code_extent));
    for (int dimension = 0; dimension < 256; ++dimension) {
        const int code       = oscar_host_quantize(values[dimension], params, bits);
        const int bit_offset = dimension * bits;
        auto* byte            = reinterpret_cast<std::uint8_t*>(destination) + (bit_offset >> 3);
        *byte = static_cast<std::uint8_t>(*byte | (code << (bit_offset & 7)));
        if ((bit_offset & 7) + bits > 8) {
            byte[1] = static_cast<std::uint8_t>(byte[1] | (code >> (8 - (bit_offset & 7))));
        }
    }
}

float load_source_oscar_value(const std::byte* source, const HostKVPlaneLayout& code_plane,
                              const HostKVPlaneLayout& scale_plane, int token, int head,
                              int dimension, int bits, int page_tokens) noexcept {
    const std::size_t code_leading_bytes =
        code_plane.head_payload_bytes / static_cast<std::size_t>(page_tokens);
    const auto* code = reinterpret_cast<const std::uint8_t*>(
        source + code_plane.offset +
        (static_cast<std::size_t>(dimension * bits) >> 3) +
        static_cast<std::size_t>(code_plane.head_payload_bytes) * head +
        code_leading_bytes * static_cast<std::size_t>(token));
    const int shift = (dimension * bits) & 7;
    std::uint16_t scale_bits = load_u16(
        source + scale_plane.offset + sizeof(std::uint16_t) *
                                           (2 * (token + page_tokens * head)));
    std::uint16_t zero_bits = load_u16(
        source + scale_plane.offset + sizeof(std::uint16_t) *
                                           (2 * (token + page_tokens * head) + 1));
    const std::uint32_t word = static_cast<std::uint32_t>(*code) |
                               (static_cast<std::uint32_t>(code[1]) << 8U);
    const int code_value = (static_cast<int>(word) >> shift) & ((1 << bits) - 1);
    return static_cast<float>(code_value) * bf16_bits_to_float(scale_bits) +
           bf16_bits_to_float(zero_bits);
}

void copy_device_pages_to_native(std::span<const std::int32_t> physical_pages,
                                 std::byte* destination, const HostKVPageLayout& native,
                                 const KVPageGeometry& geometry,
                                 const std::vector<Tensor>& device_planes,
                                 cudaStream_t stream) {
    std::size_t begin = 0;
    while (begin < physical_pages.size()) {
        std::size_t end = begin + 1;
        while (end < physical_pages.size() &&
               physical_pages[end] == physical_pages[end - 1] + 1) {
            ++end;
        }
        const std::size_t count = end - begin;
        const std::int32_t first = physical_pages[begin];
        for (std::size_t plane_index = 0; plane_index < device_planes.size(); ++plane_index) {
            const Tensor& plane                 = device_planes[plane_index];
            const HostKVPlaneLayout& host_plane = native.planes[plane_index];
            auto* host_base = destination + begin * native.page_stride + host_plane.offset;
            const auto* device_base = static_cast<const unsigned char*>(plane.data);
            if (geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    host_base, native.page_stride,
                    device_base + static_cast<std::int64_t>(first) * plane.nb[3], plane.nb[3],
                    host_plane.page_payload_bytes, count, cudaMemcpyDeviceToHost, stream));
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        host_base + static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                        native.page_stride,
                        device_base + static_cast<std::int64_t>(head) * plane.nb[3] +
                            static_cast<std::int64_t>(first) * plane.nb[2],
                        plane.nb[2], host_plane.head_payload_bytes, count,
                        cudaMemcpyDeviceToHost, stream));
                }
            }
        }
        begin = end;
    }
}

void copy_native_to_device(std::span<const std::int32_t> physical_pages,
                           const std::byte* source, const HostKVPageLayout& native,
                           const KVPageGeometry& geometry, const std::vector<Tensor>& device_planes,
                           cudaStream_t stream) {
    std::size_t begin = 0;
    while (begin < physical_pages.size()) {
        std::size_t end = begin + 1;
        while (end < physical_pages.size() &&
               physical_pages[end] == physical_pages[end - 1] + 1) {
            ++end;
        }
        const std::size_t count = end - begin;
        const std::int32_t first = physical_pages[begin];
        for (std::size_t plane_index = 0; plane_index < device_planes.size(); ++plane_index) {
            const Tensor& plane                 = device_planes[plane_index];
            const HostKVPlaneLayout& host_plane = native.planes[plane_index];
            const auto* host_base = source + begin * native.page_stride + host_plane.offset;
            auto* device_base     = static_cast<unsigned char*>(plane.data);
            if (geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    device_base + static_cast<std::int64_t>(first) * plane.nb[3], plane.nb[3],
                    host_base, native.page_stride, host_plane.page_payload_bytes, count,
                    cudaMemcpyHostToDevice, stream));
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        device_base + static_cast<std::int64_t>(head) * plane.nb[3] +
                            static_cast<std::int64_t>(first) * plane.nb[2],
                        plane.nb[2],
                        host_base + static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                        native.page_stride, host_plane.head_payload_bytes, count,
                        cudaMemcpyHostToDevice, stream));
                }
            }
        }
        begin = end;
    }
}

void native_oscar_to_hierarchical(const std::byte* source, std::byte* destination,
                                  std::uint32_t pages, const HostKVPageLayout& native,
                                  const HostKVPageLayout& hierarchical) {
    const std::size_t layers = native.geometry.planes.size() / 4U;
    const int page_tokens    = static_cast<int>(native.geometry.page_tokens);
    for (std::uint32_t page = 0; page < pages; ++page) {
        const std::byte* source_page = source + static_cast<std::size_t>(page) * native.page_stride;
        std::byte* destination_page =
            destination + static_cast<std::size_t>(page) * hierarchical.page_stride;
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const auto& source_k_code = native.planes[layer * 4U];
            const auto& source_v_code = native.planes[layer * 4U + 1U];
            const auto& source_k_scale = native.planes[layer * 4U + 2U];
            const auto& source_v_scale = native.planes[layer * 4U + 3U];
            const int heads = source_k_code.head_payload_bytes == 0
                                  ? 0
                                  : static_cast<int>(source_k_code.page_payload_bytes /
                                                      source_k_code.head_payload_bytes);
            for (int head = 0; head < heads; ++head) {
                for (int token = 0; token < page_tokens; ++token) {
                    std::array<float, 256> k_rotated{};
                    std::array<float, 256> v_rotated{};
                    for (int dimension = 0; dimension < 256; ++dimension) {
                        k_rotated[dimension] = load_source_oscar_value(
                            source_page, source_k_code, source_k_scale, token, head, dimension, 2,
                            page_tokens);
                        v_rotated[dimension] = load_source_oscar_value(
                            source_page, source_v_code, source_v_scale, token, head, dimension, 2,
                            page_tokens);
                    }
                    const auto k_quant = oscar_host_quant_params<false>(k_rotated, 4);
                    const auto v_quant = oscar_host_quant_params<true>(v_rotated, 4);
                    const std::size_t q4_base = layer * 4U;
                    pack_oscar_host_codes(
                        k_rotated, k_quant, 4,
                        destination_page + hierarchical.planes[q4_base].offset +
                            static_cast<std::size_t>(head) * hierarchical.planes[q4_base].head_payload_bytes +
                            static_cast<std::size_t>(token) * 128U);
                    pack_oscar_host_codes(
                        v_rotated, v_quant, 4,
                        destination_page + hierarchical.planes[q4_base + 1U].offset +
                            static_cast<std::size_t>(head) * hierarchical.planes[q4_base + 1U].head_payload_bytes +
                            static_cast<std::size_t>(token) * 128U);
                    const std::size_t scale_index =
                        static_cast<std::size_t>(2 * (token + page_tokens * head));
                    store_u16(destination_page + hierarchical.planes[q4_base + 2U].offset +
                                  scale_index * sizeof(std::uint16_t),
                              float_to_bf16_bits(k_quant.scale));
                    store_u16(destination_page + hierarchical.planes[q4_base + 2U].offset +
                                  (scale_index + 1U) * sizeof(std::uint16_t),
                              float_to_bf16_bits(k_quant.zero));
                    store_u16(destination_page + hierarchical.planes[q4_base + 3U].offset +
                                  scale_index * sizeof(std::uint16_t),
                              float_to_bf16_bits(v_quant.scale));
                    store_u16(destination_page + hierarchical.planes[q4_base + 3U].offset +
                                  (scale_index + 1U) * sizeof(std::uint16_t),
                              float_to_bf16_bits(v_quant.zero));

                    std::array<float, 256> k_original = k_rotated;
                    std::array<float, 256> v_original = v_rotated;
                    normalized_hadamard_d256_host(k_original);
                    normalized_hadamard_d256_host(v_original);
                    const std::size_t l2_base = layers * 4U + layer * 2U;
                    for (int dimension = 0; dimension < 256; ++dimension) {
                        const std::size_t element =
                            static_cast<std::size_t>(dimension) +
                            static_cast<std::size_t>(256) * (token + page_tokens * head);
                        store_u16(destination_page + hierarchical.planes[l2_base].offset +
                                      element * sizeof(std::uint16_t),
                                  float_to_fp16_bits(k_original[dimension]));
                        store_u16(destination_page + hierarchical.planes[l2_base + 1U].offset +
                                      element * sizeof(std::uint16_t),
                                  float_to_fp16_bits(v_original[dimension]));
                    }
                }
            }
        }
    }
}

void hierarchical_to_native_oscar(const std::byte* source, std::byte* destination,
                                  std::uint32_t pages, const HostKVPageLayout& native,
                                  const HostKVPageLayout& hierarchical) {
    const std::size_t layers = native.geometry.planes.size() / 4U;
    const int page_tokens    = static_cast<int>(native.geometry.page_tokens);
    for (std::uint32_t page = 0; page < pages; ++page) {
        const std::byte* source_page =
            source + static_cast<std::size_t>(page) * hierarchical.page_stride;
        std::byte* destination_page =
            destination + static_cast<std::size_t>(page) * native.page_stride;
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const int heads = hierarchical.planes[layer * 4U].head_payload_bytes == 0
                                  ? 0
                                  : static_cast<int>(hierarchical.planes[layer * 4U].page_payload_bytes /
                                                      hierarchical.planes[layer * 4U].head_payload_bytes);
            const std::size_t l2_base = layers * 4U + layer * 2U;
            for (int head = 0; head < heads; ++head) {
                for (int token = 0; token < page_tokens; ++token) {
                    std::array<float, 256> k_original{};
                    std::array<float, 256> v_original{};
                    for (int dimension = 0; dimension < 256; ++dimension) {
                        const std::size_t element =
                            static_cast<std::size_t>(dimension) +
                            static_cast<std::size_t>(256) * (token + page_tokens * head);
                        k_original[dimension] = fp16_bits_to_float(load_u16(
                            source_page + hierarchical.planes[l2_base].offset +
                            element * sizeof(std::uint16_t)));
                        v_original[dimension] = fp16_bits_to_float(load_u16(
                            source_page + hierarchical.planes[l2_base + 1U].offset +
                            element * sizeof(std::uint16_t)));
                    }
                    normalized_hadamard_d256_host(k_original);
                    normalized_hadamard_d256_host(v_original);
                    const auto k_quant = oscar_host_quant_params<false>(k_original, 2);
                    const auto v_quant = oscar_host_quant_params<true>(v_original, 2);
                    const auto& k_code = native.planes[layer * 4U];
                    const auto& v_code = native.planes[layer * 4U + 1U];
                    const auto& k_scale = native.planes[layer * 4U + 2U];
                    const auto& v_scale = native.planes[layer * 4U + 3U];
                    const std::size_t code_token_bytes =
                        k_code.head_payload_bytes / static_cast<std::size_t>(page_tokens);
                    const std::size_t code_offset =
                        static_cast<std::size_t>(head) * k_code.head_payload_bytes +
                        static_cast<std::size_t>(token) * code_token_bytes;
                    std::memset(destination_page + k_code.offset + code_offset, 0,
                                code_token_bytes);
                    std::memset(destination_page + v_code.offset + code_offset, 0,
                                code_token_bytes);
                    for (int dimension = 0; dimension < 256; ++dimension) {
                        const int k_value = oscar_host_quantize(k_original[dimension], k_quant, 2);
                        const int v_value = oscar_host_quantize(v_original[dimension], v_quant, 2);
                        const int bit      = dimension * 2;
                        auto* k_byte = reinterpret_cast<std::uint8_t*>(
                            destination_page + k_code.offset + code_offset + (bit >> 3));
                        auto* v_byte = reinterpret_cast<std::uint8_t*>(
                            destination_page + v_code.offset + code_offset + (bit >> 3));
                        *k_byte = static_cast<std::uint8_t>(*k_byte | (k_value << (bit & 7)));
                        *v_byte = static_cast<std::uint8_t>(*v_byte | (v_value << (bit & 7)));
                    }
                    const std::size_t scale_index =
                        static_cast<std::size_t>(2 * (token + page_tokens * head));
                    store_u16(destination_page + k_scale.offset + scale_index * sizeof(std::uint16_t),
                              float_to_bf16_bits(k_quant.scale));
                    store_u16(destination_page + k_scale.offset +
                                  (scale_index + 1U) * sizeof(std::uint16_t),
                              float_to_bf16_bits(k_quant.zero));
                    store_u16(destination_page + v_scale.offset + scale_index * sizeof(std::uint16_t),
                              float_to_bf16_bits(v_quant.scale));
                    store_u16(destination_page + v_scale.offset +
                                  (scale_index + 1U) * sizeof(std::uint16_t),
                              float_to_bf16_bits(v_quant.zero));
                }
            }
        }
    }
}

} // namespace

DeviceKVPagePoolLayout plan_device_kv_page_pool(LayoutBuilder& builder,
                                                const DeviceKVPagePoolSpec& spec) {
    const std::int32_t physical_pages =
        checked_i32(spec.page_group_count, "Paged KV physical page count");
    validate_geometry(spec.geometry);

    DeviceKVPagePoolLayout layout;
    layout.spec = spec;
    layout.planes.reserve(spec.geometry.planes.size());
    for (std::size_t index = 0; index < spec.geometry.planes.size(); ++index) {
        const KVPlaneGeometry& plane = spec.geometry.planes[index];
        DeviceKVPlaneLayout planned;
        planned.geometry        = plane;
        const std::string label = "Paged KV plane " + std::to_string(index);
        if (spec.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, plane.head_extent, physical_pages},
                plane.alignment, label);
        } else {
            planned.storage = builder.add_tensor(
                plane.dtype,
                {plane.leading_extent, kPagedKVPageSize, physical_pages, plane.head_extent},
                plane.alignment, label);
        }
        layout.planes.push_back(planned);
    }
    return layout;
}

KVExecutionTableLayout plan_kv_execution_tables(LayoutBuilder& builder,
                                                const KVExecutionTableSpec& spec) {
    (void)checked_table_bytes(spec);
    KVExecutionTableLayout layout;
    layout.spec         = spec;
    layout.block_tables = builder.add_tensor(
        DType::I32,
        {checked_i32(spec.logical_page_capacity, "Paged KV logical page capacity"),
         spec.table_rows},
        256, "Paged KV execution tables");
    return layout;
}

std::size_t DeviceKVPagePoolLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const DeviceKVPlaneLayout& plane : planes) { total += plane.storage.region.bytes; }
    return total;
}

std::size_t KVExecutionTableLayout::metadata_bytes() const noexcept {
    return block_tables.region.bytes;
}

DeviceKVPageLease::~DeviceKVPageLease() { (void)release(); }

DeviceKVPageLease::DeviceKVPageLease(DeviceKVPageLease&& other) noexcept
    : owner_(other.owner_), index_(other.index_), generation_(other.generation_) {
    other.owner_      = nullptr;
    other.index_      = -1;
    other.generation_ = 0;
}

DeviceKVPageLease& DeviceKVPageLease::operator=(DeviceKVPageLease&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_            = other.owner_;
    index_            = other.index_;
    generation_       = other.generation_;
    other.owner_      = nullptr;
    other.index_      = -1;
    other.generation_ = 0;
    return *this;
}

DeviceKVPageHandle DeviceKVPageLease::handle() const noexcept {
    return valid() ? DeviceKVPageHandle(owner_, index_, generation_) : DeviceKVPageHandle();
}

bool DeviceKVPageLease::belongs_to(const DeviceKVPagePool& pool) const noexcept {
    return owner_ == &pool;
}

bool DeviceKVPageLease::release() noexcept {
    if (!valid()) { return false; }
    DeviceKVPagePool* owner        = owner_;
    const std::int32_t index       = index_;
    const std::uint32_t generation = generation_;
    owner_                         = nullptr;
    index_                         = -1;
    generation_                    = 0;
    if (!owner->valid_handle(DeviceKVPageHandle(owner, index, generation))) { return false; }
    owner->release_page(index, generation);
    return true;
}

DeviceKVPageReservation::~DeviceKVPageReservation() { release(); }

DeviceKVPageReservation::DeviceKVPageReservation(DeviceKVPageReservation&& other) noexcept
    : owner_(other.owner_), pages_(other.pages_) {
    other.owner_ = nullptr;
    other.pages_ = 0;
}

DeviceKVPageReservation&
DeviceKVPageReservation::operator=(DeviceKVPageReservation&& other) noexcept {
    if (this == &other) { return *this; }
    release();
    owner_       = other.owner_;
    pages_       = other.pages_;
    other.owner_ = nullptr;
    other.pages_ = 0;
    return *this;
}

bool DeviceKVPageReservation::belongs_to(const DeviceKVPagePool& pool) const noexcept {
    return owner_ == &pool;
}

void DeviceKVPageReservation::clear() noexcept {
    if (!valid() || pages_ == 0) { return; }
    owner_->release_reservation(pages_);
    pages_ = 0;
}

void DeviceKVPageReservation::release() noexcept {
    if (!valid()) { return; }
    owner_->release_reservation(pages_);
    owner_ = nullptr;
    pages_ = 0;
}

DeviceKVPagePool::DeviceKVPagePool(DeviceSpan backing, const DeviceKVPagePoolLayout& layout)
    : spec_(layout.spec) {
    validate_geometry(spec_.geometry);
    if (layout.planes.size() != spec_.geometry.planes.size() || layout.planes.empty()) {
        throw std::invalid_argument("Paged KV device layout plane inventory is inconsistent");
    }

    const std::int32_t physical_pages =
        checked_i32(spec_.page_group_count, "Paged KV physical page count");
    planes_.reserve(layout.planes.size());
    for (std::size_t index = 0; index < layout.planes.size(); ++index) {
        const DeviceKVPlaneLayout& planned = layout.planes[index];
        const KVPlaneGeometry& expected    = spec_.geometry.planes[index];
        if (planned.geometry != expected) {
            throw std::logic_error("Paged KV device plane layout does not match its geometry");
        }
        Tensor plane = planned.storage.bind(backing);
        if (plane.dtype != expected.dtype || plane.ne[0] != expected.leading_extent ||
            plane.ne[1] != kPagedKVPageSize) {
            throw std::logic_error("Paged KV device plane tensor is inconsistent");
        }
        if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            if (plane.ne[2] != expected.head_extent || plane.ne[3] != physical_pages) {
                throw std::logic_error("Paged KV PageMajor plane shape is inconsistent");
            }
        } else if (plane.ne[2] != physical_pages || plane.ne[3] != expected.head_extent) {
            throw std::logic_error("Paged KV HeadMajor plane shape is inconsistent");
        }
        planes_.push_back(plane);
    }

    free_page_runs_.reserve(spec_.page_group_count);
    page_generations_.assign(spec_.page_group_count, 1);
    page_allocated_.assign(spec_.page_group_count, false);
    validation_marks_.assign(spec_.page_group_count, 0);
    free_page_runs_.push_back(FreePageRun{.begin = 0, .count = spec_.page_group_count});
}

std::uint32_t DeviceKVPagePool::capacity_pages() const noexcept { return spec_.page_group_count; }

std::uint32_t DeviceKVPagePool::allocated_pages() const noexcept { return allocated_pages_; }

std::uint32_t DeviceKVPagePool::reserved_pages() const noexcept { return reserved_pages_; }

std::uint32_t DeviceKVPagePool::available_pages() const noexcept {
    return capacity_pages() - allocated_pages_ - reserved_pages_;
}

std::size_t DeviceKVPagePool::plane_count() const noexcept { return planes_.size(); }

const Tensor& DeviceKVPagePool::plane(std::size_t index) const { return planes_.at(index); }

std::uint32_t
DeviceKVPagePool::contiguous_run_count(std::span<const DeviceKVPageHandle> pages) const {
    if (pages.empty()) { return 0; }
    std::uint32_t runs    = 0;
    std::int32_t previous = -2;
    for (const DeviceKVPageHandle page : pages) {
        const std::int32_t current = physical_index(page);
        if (runs == 0 || current != previous + 1) { ++runs; }
        previous = current;
    }
    return runs;
}

std::optional<DeviceKVPageReservation> DeviceKVPagePool::reserve(std::uint32_t pages) noexcept {
    if (pages == 0 || pages > available_pages()) { return std::nullopt; }
    reserved_pages_ += pages;
    return DeviceKVPageReservation(*this, pages);
}

bool DeviceKVPagePool::can_resize_reservation(const DeviceKVPageReservation& reservation,
                                              std::uint32_t new_reserved_pages) const noexcept {
    if (!reservation.belongs_to(*this) || reservation.pages_ > reserved_pages_) { return false; }
    const std::uint64_t used = static_cast<std::uint64_t>(allocated_pages_) +
                               static_cast<std::uint64_t>(reserved_pages_ - reservation.pages_) +
                               new_reserved_pages;
    return used <= capacity_pages();
}

void DeviceKVPagePool::resize_reservation(DeviceKVPageReservation& reservation,
                                          std::uint32_t new_reserved_pages) {
    if (!can_resize_reservation(reservation, new_reserved_pages)) { throw std::bad_alloc(); }
    reserved_pages_    = reserved_pages_ - reservation.pages_ + new_reserved_pages;
    reservation.pages_ = new_reserved_pages;
}

void DeviceKVPagePool::materialize(DeviceKVPageReservation& reservation,
                                   std::uint32_t target_page_count,
                                   std::vector<DeviceKVPageLease>& destination,
                                   std::optional<DeviceKVPageHandle> preferred_predecessor) {
    static_assert(std::is_nothrow_move_constructible_v<DeviceKVPageLease>);
    if (!reservation.belongs_to(*this) || target_page_count < destination.size()) {
        throw std::invalid_argument("Paged KV materialization has an invalid owner or extent");
    }
    const std::uint32_t old_count = static_cast<std::uint32_t>(destination.size());
    const std::uint32_t count     = target_page_count - old_count;
    if (count > reservation.pages_ || target_page_count > destination.capacity()) {
        throw std::invalid_argument("Paged KV materialization exceeds reserved capacity");
    }
    for (const DeviceKVPageLease& page : destination) {
        if (!page.belongs_to(*this) || !valid_handle(page.handle())) {
            throw std::invalid_argument("Paged KV materialization destination is stale");
        }
    }
    if (count == 0) { return; }
    if (count > capacity_pages() - allocated_pages_) {
        throw std::logic_error("Paged KV reservation invariant was violated");
    }

    std::optional<std::int32_t> preferred;
    if (preferred_predecessor) {
        preferred = physical_index(*preferred_predecessor) + 1;
    } else if (!destination.empty()) {
        preferred = destination.back().index_ + 1;
    }

    std::size_t selected        = free_page_runs_.size();
    std::int32_t selected_begin = 0;
    if (preferred && *preferred >= 0) {
        const auto upper = std::upper_bound(
            free_page_runs_.begin(), free_page_runs_.end(), *preferred,
            [](std::int32_t page, const FreePageRun& run) { return page < run.begin; });
        if (upper != free_page_runs_.begin()) {
            const auto candidate = upper - 1;
            const std::uint64_t run_end =
                static_cast<std::uint64_t>(candidate->begin) + candidate->count;
            const std::uint64_t requested_end = static_cast<std::uint64_t>(*preferred) + count;
            if (*preferred >= candidate->begin && requested_end <= run_end) {
                selected       = static_cast<std::size_t>(candidate - free_page_runs_.begin());
                selected_begin = *preferred;
            }
        }
    }
    if (selected == free_page_runs_.size()) {
        const auto contiguous =
            std::find_if(free_page_runs_.begin(), free_page_runs_.end(),
                         [count](const FreePageRun& run) { return run.count >= count; });
        if (contiguous != free_page_runs_.end()) {
            selected       = static_cast<std::size_t>(contiguous - free_page_runs_.begin());
            selected_begin = contiguous->begin;
        }
    }

    const auto append_page = [&](std::int32_t page) {
        destination.push_back(
            DeviceKVPageLease(*this, page, page_generations_[static_cast<std::size_t>(page)]));
        page_allocated_[static_cast<std::size_t>(page)] = true;
    };
    if (selected != free_page_runs_.size()) {
        for (std::uint32_t offset = 0; offset < count; ++offset) {
            append_page(selected_begin + static_cast<std::int32_t>(offset));
        }
        consume_free_run(selected, selected_begin, count);
    } else {
        std::uint32_t remaining   = count;
        std::size_t consumed_runs = 0;
        for (std::size_t run_index = 0; remaining != 0; ++run_index) {
            FreePageRun& run         = free_page_runs_[run_index];
            const std::uint32_t take = std::min(remaining, run.count);
            for (std::uint32_t offset = 0; offset < take; ++offset) {
                append_page(run.begin + static_cast<std::int32_t>(offset));
            }
            remaining -= take;
            if (take == run.count) {
                ++consumed_runs;
            } else {
                run.begin += static_cast<std::int32_t>(take);
                run.count -= take;
            }
        }
        free_page_runs_.erase(free_page_runs_.begin(),
                              free_page_runs_.begin() + static_cast<std::ptrdiff_t>(consumed_runs));
    }
    allocated_pages_ += count;
    reserved_pages_ -= count;
    reservation.pages_ -= count;
}

DeviceKVPageLease DeviceKVPagePool::materialize_one(DeviceKVPageReservation& reservation) {
    if (!reservation.belongs_to(*this) || reservation.pages_ == 0) {
        throw std::invalid_argument("Paged KV single-page materialization exceeds reservation");
    }
    if (free_page_runs_.empty()) {
        throw std::logic_error("Paged KV reservation invariant was violated");
    }
    FreePageRun& run        = free_page_runs_.front();
    const std::int32_t page = run.begin++;
    if (--run.count == 0) { free_page_runs_.erase(free_page_runs_.begin()); }
    page_allocated_[static_cast<std::size_t>(page)] = true;
    ++allocated_pages_;
    --reserved_pages_;
    --reservation.pages_;
    return DeviceKVPageLease(*this, page, page_generations_[static_cast<std::size_t>(page)]);
}

void DeviceKVPagePool::dematerialize(DeviceKVPageReservation& reservation,
                                     std::uint32_t target_page_count,
                                     std::vector<DeviceKVPageLease>& source) {
    if (!reservation.belongs_to(*this) || target_page_count > source.size()) {
        throw std::invalid_argument("Paged KV dematerialization has an invalid owner or extent");
    }
    for (const DeviceKVPageLease& page : source) {
        if (!page.belongs_to(*this) || !valid_handle(page.handle())) {
            throw std::invalid_argument("Paged KV dematerialization source is stale");
        }
    }
    const std::uint32_t released = static_cast<std::uint32_t>(source.size() - target_page_count);
    if (released == 0) { return; }
    if (released > std::numeric_limits<std::uint32_t>::max() - reservation.pages_) {
        throw std::overflow_error("Paged KV reservation size overflow");
    }

    // Shrinking a vector of nothrow-destructible leases cannot fail. Each destructor first returns
    // its physical page; the capacity is then adopted by the same reservation before this method
    // becomes observable to its single owner.
    source.resize(target_page_count);
    reserved_pages_ += released;
    reservation.pages_ += released;
}

void DeviceKVPagePool::dematerialize_one(DeviceKVPageReservation& reservation,
                                         DeviceKVPageLease&& page) {
    if (!reservation.belongs_to(*this) || !page.belongs_to(*this) || !valid_handle(page.handle()) ||
        reservation.pages_ == std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Paged KV single-page dematerialization is invalid");
    }
    if (!page.release()) {
        throw std::logic_error("Paged KV single-page dematerialization lost its lease");
    }
    ++reserved_pages_;
    ++reservation.pages_;
}

bool DeviceKVPagePool::valid_handle(DeviceKVPageHandle handle) const noexcept {
    if (handle.owner_ != this || handle.index_ < 0 ||
        handle.index_ >= static_cast<std::int32_t>(capacity_pages())) {
        return false;
    }
    const std::size_t index = static_cast<std::size_t>(handle.index_);
    return page_allocated_[index] && page_generations_[index] == handle.generation_;
}

std::int32_t DeviceKVPagePool::physical_index(DeviceKVPageHandle handle) const {
    if (!valid_handle(handle)) { throw std::invalid_argument("Paged KV page handle is stale"); }
    return handle.index_;
}

void DeviceKVPagePool::validate_distinct_pages(std::span<const DeviceKVPageHandle> pages,
                                               const char* duplicate_message) const {
    ++validation_stamp_;
    if (validation_stamp_ == 0) {
        std::fill(validation_marks_.begin(), validation_marks_.end(), 0);
        validation_stamp_ = 1;
    }
    for (const DeviceKVPageHandle page : pages) {
        const std::size_t index = static_cast<std::size_t>(physical_index(page));
        if (validation_marks_[index] == validation_stamp_) {
            throw std::invalid_argument(duplicate_message);
        }
        validation_marks_[index] = validation_stamp_;
    }
}

void DeviceKVPagePool::release_page(std::int32_t index, std::uint32_t generation) noexcept {
    if (index < 0 || index >= static_cast<std::int32_t>(capacity_pages())) { return; }
    const std::size_t position = static_cast<std::size_t>(index);
    if (!page_allocated_[position] || page_generations_[position] != generation) { return; }
    page_allocated_[position] = false;
    increment_generation(page_generations_[position]);
    release_free_page(index);
    --allocated_pages_;
}

void DeviceKVPagePool::consume_free_run(std::size_t run_index, std::int32_t begin,
                                        std::uint32_t count) noexcept {
    if (run_index >= free_page_runs_.size() || count == 0) { std::terminate(); }
    FreePageRun& run                = free_page_runs_[run_index];
    const std::int64_t run_end      = static_cast<std::int64_t>(run.begin) + run.count;
    const std::int64_t consumed_end = static_cast<std::int64_t>(begin) + count;
    if (begin < run.begin || consumed_end > run_end) { std::terminate(); }
    if (begin == run.begin && consumed_end == run_end) {
        free_page_runs_.erase(free_page_runs_.begin() + static_cast<std::ptrdiff_t>(run_index));
        return;
    }
    if (begin == run.begin) {
        run.begin += static_cast<std::int32_t>(count);
        run.count -= count;
        return;
    }
    if (consumed_end == run_end) {
        run.count = static_cast<std::uint32_t>(begin - run.begin);
        return;
    }
    const FreePageRun right{.begin = static_cast<std::int32_t>(consumed_end),
                            .count = static_cast<std::uint32_t>(run_end - consumed_end)};
    run.count = static_cast<std::uint32_t>(begin - run.begin);
    free_page_runs_.insert(free_page_runs_.begin() + static_cast<std::ptrdiff_t>(run_index + 1),
                           right);
}

void DeviceKVPagePool::release_free_page(std::int32_t index) noexcept {
    const auto next = std::lower_bound(
        free_page_runs_.begin(), free_page_runs_.end(), index,
        [](const FreePageRun& run, std::int32_t page) { return run.begin < page; });
    const bool joins_right = next != free_page_runs_.end() && index + 1 == next->begin;
    const bool joins_left =
        next != free_page_runs_.begin() &&
        static_cast<std::int64_t>((next - 1)->begin) + (next - 1)->count == index;
    if (joins_left && joins_right) {
        auto& left = *(next - 1);
        left.count += 1U + next->count;
        free_page_runs_.erase(next);
    } else if (joins_left) {
        ++(next - 1)->count;
    } else if (joins_right) {
        next->begin = index;
        ++next->count;
    } else {
        free_page_runs_.insert(next, FreePageRun{.begin = index, .count = 1});
    }
}

void DeviceKVPagePool::release_reservation(std::uint32_t pages) noexcept {
    if (pages > reserved_pages_) { std::terminate(); }
    reserved_pages_ -= pages;
}

void DeviceKVPagePool::zero_pages(std::span<const DeviceKVPageHandle> pages,
                                  cudaStream_t stream) const {
    validate_distinct_pages(pages, "Paged KV zero destination contains duplicate pages");
    std::size_t begin = 0;
    while (begin < pages.size()) {
        std::size_t end = begin + 1;
        while (end < pages.size() && pages[end].index_ == pages[end - 1].index_ + 1) { ++end; }
        const std::int32_t first = pages[begin].index_;
        const std::int32_t count = static_cast<std::int32_t>(end - begin);
        for (const Tensor& plane : planes_) {
            auto* base = static_cast<unsigned char*>(plane.data);
            if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemsetAsync(base + static_cast<std::int64_t>(first) * plane.nb[3], 0,
                                           static_cast<std::size_t>(count) * plane.nb[3], stream));
            } else {
                CUDA_CHECK(cudaMemset2DAsync(base + static_cast<std::int64_t>(first) * plane.nb[2],
                                             plane.nb[3], 0,
                                             static_cast<std::size_t>(count) * plane.nb[2],
                                             static_cast<std::size_t>(plane.ne[3]), stream));
            }
        }
        begin = end;
    }
}

void DeviceKVPagePool::copy_page(DeviceKVPageHandle source, DeviceKVPageHandle destination,
                                 cudaStream_t stream) const {
    const std::int32_t source_index      = physical_index(source);
    const std::int32_t destination_index = physical_index(destination);
    if (source_index == destination_index) { return; }
    for (const Tensor& plane : planes_) {
        auto* base = static_cast<unsigned char*>(plane.data);
        if (spec_.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor) {
            CUDA_CHECK(
                cudaMemcpyAsync(base + static_cast<std::int64_t>(destination_index) * plane.nb[3],
                                base + static_cast<std::int64_t>(source_index) * plane.nb[3],
                                plane.nb[3], cudaMemcpyDeviceToDevice, stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                base + static_cast<std::int64_t>(destination_index) * plane.nb[2], plane.nb[3],
                base + static_cast<std::int64_t>(source_index) * plane.nb[2], plane.nb[3],
                plane.nb[2], static_cast<std::size_t>(plane.ne[3]), cudaMemcpyDeviceToDevice,
                stream));
        }
    }
}

void DeviceKVPagePool::copy_to_host(std::span<const DeviceKVPageHandle> source,
                                    HostKVAllocationView destination, cudaStream_t stream) const {
    if (!destination.valid() || destination.page_count() != source.size() ||
        destination.layout().geometry != geometry()) {
        throw std::invalid_argument("Paged KV D2H geometry or extent is inconsistent");
    }
    if (source.empty()) { return; }
    for (DeviceKVPageHandle page : source) { (void)physical_index(page); }

    const HostKVPageLayout& host = destination.layout();
    if (is_hierarchical_oscar_host_layout(host)) {
        const HostKVPageLayout native = plan_host_kv_page_layout(geometry());
        if (native.geometry != host.geometry || source.size() >
                                                     std::numeric_limits<std::uint32_t>::max() ||
            native.page_stride > std::numeric_limits<std::size_t>::max() / source.size()) {
            throw std::invalid_argument("Paged KV hierarchical host conversion geometry is invalid");
        }
        std::vector<std::int32_t> physical_pages;
        physical_pages.reserve(source.size());
        for (const DeviceKVPageHandle page : source) { physical_pages.push_back(physical_index(page)); }
        PinnedHostBuffer staging(native.page_stride * source.size());
        copy_device_pages_to_native(
            std::span<const std::int32_t>(physical_pages), static_cast<std::byte*>(staging.data()),
            native, geometry(), planes_, stream);
        // The conversion is intentionally outside the serving stream.  Host snapshots are
        // checkpoint work, and synchronizing here is what makes the pinned record a complete
        // Q4+FP16 pair before its extent is published.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        native_oscar_to_hierarchical(static_cast<const std::byte*>(staging.data()),
                                     destination.data(), static_cast<std::uint32_t>(source.size()),
                                     native, host);
        return;
    }
    const bool convert_bf16_to_fp16 = host_uses_fp16_for_bf16(host, planes_);
    std::size_t begin            = 0;
    while (begin < source.size()) {
        std::size_t end = begin + 1;
        while (end < source.size() && source[end].index_ == source[end - 1].index_ + 1) { ++end; }
        const std::size_t count  = end - begin;
        const std::int32_t first = source[begin].index_;
        for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
            const Tensor& plane                 = planes_[plane_index];
            const HostKVPlaneLayout& host_plane = host.planes[plane_index];
            auto* host_base = destination.data() + begin * host.page_stride + host_plane.offset;
            const auto* device_base = static_cast<const unsigned char*>(plane.data);
            if (geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    host_base, host.page_stride,
                    device_base + static_cast<std::int64_t>(first) * plane.nb[3], plane.nb[3],
                    host_plane.page_payload_bytes, count, cudaMemcpyDeviceToHost, stream));
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        host_base + static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                        host.page_stride,
                        device_base + static_cast<std::int64_t>(head) * plane.nb[3] +
                            static_cast<std::int64_t>(first) * plane.nb[2],
                        plane.nb[2], host_plane.head_payload_bytes, count, cudaMemcpyDeviceToHost,
                        stream));
                }
            }
        }
        begin = end;
    }
    if (convert_bf16_to_fp16) {
        // The destination is the canonical pinned host record. Wait for the raw DMA before
        // converting in place; this keeps the conversion bounded by the destination allocation
        // and avoids a second context-sized staging buffer.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        convert_host_records_bf16_to_fp16(destination.data(),
                                          static_cast<std::uint32_t>(source.size()), host, planes_);
    }
}

void DeviceKVPagePool::copy_from_host(HostKVAllocationConstView source,
                                      std::span<const DeviceKVPageHandle> destination,
                                      cudaStream_t stream) const {
    if (!source.valid() || source.page_count() != destination.size() ||
        source.layout().geometry != geometry()) {
        throw std::invalid_argument("Paged KV H2D geometry or extent is inconsistent");
    }
    if (destination.empty()) { return; }
    validate_distinct_pages(destination, "Paged KV H2D destination contains duplicate pages");

    const HostKVPageLayout& host = source.layout();
    if (is_hierarchical_oscar_host_layout(host)) {
        const HostKVPageLayout native = plan_host_kv_page_layout(geometry());
        if (native.geometry != host.geometry || destination.size() >
                                                     std::numeric_limits<std::uint32_t>::max() ||
            native.page_stride > std::numeric_limits<std::size_t>::max() / destination.size()) {
            throw std::invalid_argument("Paged KV hierarchical host restore geometry is invalid");
        }
        std::vector<std::int32_t> physical_pages;
        physical_pages.reserve(destination.size());
        for (const DeviceKVPageHandle page : destination) {
            physical_pages.push_back(physical_index(page));
        }
        PinnedHostBuffer staging(native.page_stride * destination.size());
        hierarchical_to_native_oscar(source.data(), static_cast<std::byte*>(staging.data()),
                                      static_cast<std::uint32_t>(destination.size()), native, host);
        copy_native_to_device(
            std::span<const std::int32_t>(physical_pages), static_cast<const std::byte*>(staging.data()),
            native, geometry(), planes_, stream);
        // The staging buffer is scoped to this call.  Wait for the H2D work before releasing it;
        // the normal native path remains asynchronous.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    const bool convert_fp16_to_bf16 = host_uses_fp16_for_bf16(host, planes_);
    if (convert_fp16_to_bf16) {
        // H2D restores are infrequent relative to decode. Convert bounded chunks into a temporary
        // canonical record and use synchronous copies so the staging storage can be reused safely
        // without extending the Host KV allocation lifetime.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        constexpr std::size_t kConversionChunkBytes = 8U * 1024U * 1024U;
        const std::size_t pages_per_chunk =
            std::max<std::size_t>(1U, kConversionChunkBytes / host.page_stride);
        std::vector<std::byte> staging;
        std::size_t begin = 0;
        while (begin < destination.size()) {
            std::size_t end = begin + 1;
            while (end < destination.size() &&
                   destination[end].index_ == destination[end - 1].index_ + 1) {
                ++end;
            }
            for (std::size_t chunk_begin = begin; chunk_begin < end;
                 chunk_begin += pages_per_chunk) {
                const std::size_t count = std::min(pages_per_chunk, end - chunk_begin);
                if (count > std::numeric_limits<std::size_t>::max() / host.page_stride) {
                    throw std::overflow_error("Paged KV FP16 restore staging size overflow");
                }
                const std::size_t bytes = count * host.page_stride;
                staging.resize(bytes);
                std::memcpy(staging.data(),
                            source.data() + chunk_begin * host.page_stride, bytes);
                convert_host_records_fp16_to_bf16(staging.data(),
                                                  static_cast<std::uint32_t>(count), host, planes_);
                const std::int32_t first = destination[chunk_begin].index_;
                for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
                    const Tensor& plane                 = planes_[plane_index];
                    const HostKVPlaneLayout& host_plane = host.planes[plane_index];
                    const auto* host_base = staging.data() + host_plane.offset;
                    auto* device_base     = static_cast<unsigned char*>(plane.data);
                    if (geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
                        CUDA_CHECK(cudaMemcpy2D(
                            device_base + static_cast<std::int64_t>(first) * plane.nb[3],
                            plane.nb[3], host_base, host.page_stride,
                            host_plane.page_payload_bytes, count, cudaMemcpyHostToDevice));
                    } else {
                        for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                            CUDA_CHECK(cudaMemcpy2D(
                                device_base + static_cast<std::int64_t>(head) * plane.nb[3] +
                                    static_cast<std::int64_t>(first) * plane.nb[2],
                                plane.nb[2],
                                host_base + static_cast<std::size_t>(head) *
                                                host_plane.head_payload_bytes,
                                host.page_stride, host_plane.head_payload_bytes, count,
                                cudaMemcpyHostToDevice));
                        }
                    }
                }
            }
            begin = end;
        }
        return;
    }
    std::size_t begin            = 0;
    while (begin < destination.size()) {
        std::size_t end = begin + 1;
        while (end < destination.size() &&
               destination[end].index_ == destination[end - 1].index_ + 1) {
            ++end;
        }
        const std::size_t count  = end - begin;
        const std::int32_t first = destination[begin].index_;
        for (std::size_t plane_index = 0; plane_index < planes_.size(); ++plane_index) {
            const Tensor& plane                 = planes_[plane_index];
            const HostKVPlaneLayout& host_plane = host.planes[plane_index];
            const auto* host_base = source.data() + begin * host.page_stride + host_plane.offset;
            auto* device_base     = static_cast<unsigned char*>(plane.data);
            if (geometry().device_plane_order == PagedKVPlaneOrder::PageMajor) {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    device_base + static_cast<std::int64_t>(first) * plane.nb[3], plane.nb[3],
                    host_base, host.page_stride, host_plane.page_payload_bytes, count,
                    cudaMemcpyHostToDevice, stream));
            } else {
                for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        device_base + static_cast<std::int64_t>(head) * plane.nb[3] +
                            static_cast<std::int64_t>(first) * plane.nb[2],
                        plane.nb[2],
                        host_base + static_cast<std::size_t>(head) * host_plane.head_payload_bytes,
                        host.page_stride, host_plane.head_payload_bytes, count,
                        cudaMemcpyHostToDevice, stream));
                }
            }
        }
        begin = end;
    }
}

std::vector<DeviceKVPageReservation>
reserve_device_kv_page_bundle(std::span<const DeviceKVPageReservationRequest> requests) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const DeviceKVPageReservationRequest& request = requests[index];
        if (request.pool == nullptr || request.pages == 0) {
            throw std::invalid_argument("Paged KV bundle reservation is empty");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (requests[previous].pool == request.pool) {
                throw std::invalid_argument("Paged KV bundle names the same pool twice");
            }
        }
        if (request.pages > request.pool->available_pages()) { throw std::bad_alloc(); }
    }

    std::vector<DeviceKVPageReservation> reservations;
    reservations.reserve(requests.size());
    for (const DeviceKVPageReservationRequest& request : requests) {
        std::optional<DeviceKVPageReservation> reservation = request.pool->reserve(request.pages);
        if (!reservation) {
            throw std::logic_error("Paged KV bundle prevalidation was not stable");
        }
        reservations.push_back(std::move(*reservation));
    }
    return reservations;
}

KVExecutionRowLease::~KVExecutionRowLease() { (void)release(); }

KVExecutionRowLease::KVExecutionRowLease(KVExecutionRowLease&& other) noexcept
    : owner_(other.owner_), row_(other.row_), generation_(other.generation_) {
    other.owner_      = nullptr;
    other.row_        = -1;
    other.generation_ = 0;
}

KVExecutionRowLease& KVExecutionRowLease::operator=(KVExecutionRowLease&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_            = other.owner_;
    row_              = other.row_;
    generation_       = other.generation_;
    other.owner_      = nullptr;
    other.row_        = -1;
    other.generation_ = 0;
    return *this;
}

KVExecutionRowHandle KVExecutionRowLease::handle() const noexcept {
    return valid() ? KVExecutionRowHandle(owner_, row_, generation_) : KVExecutionRowHandle();
}

bool KVExecutionRowLease::belongs_to(const KVExecutionTablePool& pool) const noexcept {
    return owner_ == &pool;
}

bool KVExecutionRowLease::release() noexcept {
    if (!valid()) { return false; }
    KVExecutionTablePool* owner    = owner_;
    const std::int32_t row         = row_;
    const std::uint32_t generation = generation_;
    owner_                         = nullptr;
    row_                           = -1;
    generation_                    = 0;
    return owner->release_row(row, generation);
}

KVExecutionTablePool::KVExecutionTablePool(DeviceSpan backing, const KVExecutionTableLayout& layout,
                                           const DeviceKVPagePool& pages)
    : spec_(layout.spec), pages_(&pages), block_tables_(layout.block_tables.bind(backing)),
      host_shadow_(checked_table_bytes(layout.spec)),
      row_in_use_(static_cast<std::size_t>(layout.spec.table_rows), false),
      row_generations_(static_cast<std::size_t>(layout.spec.table_rows), 1) {
    if (block_tables_.dtype != DType::I32 ||
        block_tables_.ne[0] !=
            checked_i32(spec_.logical_page_capacity, "Paged KV logical page capacity") ||
        block_tables_.ne[1] != spec_.table_rows) {
        throw std::logic_error("Paged KV execution-table layout is inconsistent");
    }
}

std::uint32_t KVExecutionTablePool::logical_page_capacity() const noexcept {
    return spec_.logical_page_capacity;
}

std::int32_t KVExecutionTablePool::row_count() const noexcept { return spec_.table_rows; }

KVExecutionRowLease KVExecutionTablePool::acquire(std::int32_t row_index) {
    if (row_index < 0 || row_index >= row_count()) {
        throw std::out_of_range("Paged KV execution row is out of range");
    }
    const std::size_t index = static_cast<std::size_t>(row_index);
    if (row_in_use_[index]) { throw std::logic_error("Paged KV execution row is already bound"); }
    row_in_use_[index] = true;
    return KVExecutionRowLease(*this, row_index, row_generations_[index]);
}

bool KVExecutionTablePool::valid_handle(KVExecutionRowHandle handle) const noexcept {
    if (handle.owner_ != this || handle.row_ < 0 || handle.row_ >= row_count()) { return false; }
    const std::size_t row_index = static_cast<std::size_t>(handle.row_);
    return row_in_use_[row_index] && row_generations_[row_index] == handle.generation_;
}

bool KVExecutionTablePool::release_row(std::int32_t row_index, std::uint32_t generation) noexcept {
    if (row_index < 0 || row_index >= row_count()) { return false; }
    const std::size_t index = static_cast<std::size_t>(row_index);
    if (!row_in_use_[index] || row_generations_[index] != generation) { return false; }
    row_in_use_[index] = false;
    increment_generation(row_generations_[index]);
    return true;
}

void KVExecutionTablePool::publish(KVExecutionRowHandle row_handle, std::uint32_t logical_begin,
                                   std::span<const DeviceKVPageHandle> page_handles,
                                   cudaStream_t stream) {
    if (!valid_handle(row_handle) || logical_begin > logical_page_capacity() ||
        page_handles.size() > logical_page_capacity() - logical_begin) {
        throw std::invalid_argument("Paged KV mapping publication is outside its execution row");
    }
    auto* shadow = static_cast<std::int32_t*>(host_shadow_.data()) +
                   static_cast<std::size_t>(row_handle.row_) * logical_page_capacity() +
                   logical_begin;
    for (std::size_t index = 0; index < page_handles.size(); ++index) {
        shadow[index] = pages_->physical_index(page_handles[index]);
    }
    publish_indices(row_handle, logical_begin,
                    std::span<const std::int32_t>(shadow, page_handles.size()), stream);
}

void KVExecutionTablePool::publish(KVExecutionRowHandle row_handle, std::uint32_t logical_begin,
                                   std::span<const DeviceKVPageLease> page_leases,
                                   cudaStream_t stream) {
    if (!valid_handle(row_handle) || logical_begin > logical_page_capacity() ||
        page_leases.size() > logical_page_capacity() - logical_begin) {
        throw std::invalid_argument("Paged KV mapping publication is outside its execution row");
    }
    auto* shadow = static_cast<std::int32_t*>(host_shadow_.data()) +
                   static_cast<std::size_t>(row_handle.row_) * logical_page_capacity() +
                   logical_begin;
    for (std::size_t index = 0; index < page_leases.size(); ++index) {
        if (!page_leases[index].belongs_to(*pages_)) {
            throw std::invalid_argument("Paged KV execution mapping names another page pool");
        }
        shadow[index] = pages_->physical_index(page_leases[index].handle());
    }
    publish_indices(row_handle, logical_begin,
                    std::span<const std::int32_t>(shadow, page_leases.size()), stream);
}

void KVExecutionTablePool::publish_repeated(KVExecutionRowHandle row_handle,
                                            DeviceKVPageHandle page, std::uint32_t count,
                                            cudaStream_t stream) {
    if (!valid_handle(row_handle) || count > logical_page_capacity()) {
        throw std::invalid_argument("Repeated Paged KV mapping is outside its execution row");
    }
    const std::int32_t physical = pages_->physical_index(page);
    auto* shadow                = static_cast<std::int32_t*>(host_shadow_.data()) +
                   static_cast<std::size_t>(row_handle.row_) * logical_page_capacity();
    std::fill_n(shadow, count, physical);
    publish_indices(row_handle, 0, std::span<const std::int32_t>(shadow, count), stream);
}

void KVExecutionTablePool::publish_indices(KVExecutionRowHandle row_handle,
                                           std::uint32_t logical_begin,
                                           std::span<const std::int32_t> indices,
                                           cudaStream_t stream) {
    if (indices.empty()) { return; }
    Tensor destination_row = row(row_handle);
    auto* destination      = static_cast<std::int32_t*>(destination_row.data) + logical_begin;
    CUDA_CHECK(cudaMemcpyAsync(destination, indices.data(), indices.size_bytes(),
                               cudaMemcpyHostToDevice, stream));
}

Tensor KVExecutionTablePool::row(KVExecutionRowHandle handle) const {
    if (!valid_handle(handle)) { throw std::invalid_argument("Paged KV execution row is stale"); }
    return block_tables_.slice(1, handle.row_, 1)
        .view({static_cast<std::int32_t>(logical_page_capacity())});
}

} // namespace ninfer
