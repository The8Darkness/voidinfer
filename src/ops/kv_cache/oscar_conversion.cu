#include "ops/kv_cache/oscar_conversion.h"

#include "core/device.h"
#include "ops/kv_cache/oscar_codec.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::ops {
namespace {

template <int Bits>
__device__ __forceinline__ std::uint8_t pack_oscar_byte(const std::uint8_t (&codes)[4], int byte) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr int CodeExtent     = cyclic_oscar_code_extent<Bits>(kCyclicKVCacheOscarHeadDim);
    (void)CodeExtent;
    std::uint8_t packed = 0;
    int bit_cursor       = 0;
    while (bit_cursor < 8) {
        const int global_bit = byte * 8 + bit_cursor;
        const int dimension  = global_bit / Bits;
        const int bit_offset = global_bit - dimension * Bits;
        const int take       = min(Bits - bit_offset, 8 - bit_cursor);
        const int source_lane = dimension & 31;
        const int source_item = dimension >> 5;
        const int code = __shfl_sync(FullMask, static_cast<int>(codes[source_item]), source_lane);
        const int mask = (1 << take) - 1;
        packed |= static_cast<std::uint8_t>(((code >> bit_offset) & mask) << bit_cursor);
        bit_cursor += take;
    }
    return packed;
}

template <int Bits>
__device__ __forceinline__ std::uint8_t unpack_oscar_conversion_code(
    const std::uint8_t* codes, int dimension, bool q2_transposed) {
    // The DFlash Q2 hot layout stores byte l as the four symbols at
    // dimensions l + {0,32,64,96}. Other OSCAR images retain the ordinary
    // contiguous bitstream. The runtime flag is only used on state-boundary
    // conversion kernels, never on the decode hot path.
    if (q2_transposed && Bits == 2) {
        const int lane = dimension & 31;
        const int item = dimension >> 5;
        return static_cast<std::uint8_t>((codes[lane] >> (item << 1)) & 3);
    }
    return cyclic_oscar_unpack_code<Bits>(codes, dimension);
}

template <int SourceBits, int DestinationBits>
__global__ void cyclic_oscar_requantize_kernel(
    const std::uint8_t* __restrict__ source_k,
    const __nv_bfloat16* __restrict__ source_k_metadata,
    const std::uint8_t* __restrict__ source_v,
    const __nv_bfloat16* __restrict__ source_v_metadata,
    std::uint8_t* __restrict__ destination_k,
    __nv_bfloat16* __restrict__ destination_k_metadata,
    std::uint8_t* __restrict__ destination_v,
    __nv_bfloat16* __restrict__ destination_v_metadata, int padded_capacity, int kv_heads,
    bool source_q2_transposed, bool destination_q2_transposed) {
    constexpr int D                  = kCyclicKVCacheOscarHeadDim;
    constexpr int SourceCodeExtent   = cyclic_oscar_code_extent<SourceBits>(D);
    constexpr int DestinationCodeExtent = cyclic_oscar_code_extent<DestinationBits>(D);
    constexpr int ScaleExtent        = kCyclicKVCacheOscarScaleExtent;
    const int lane                  = static_cast<int>(threadIdx.x);
    const int row                   = static_cast<int>(blockIdx.x);
    const int rows                  = padded_capacity * kv_heads;
    if (lane >= 32 || row >= rows) return;
    const int slot    = row % padded_capacity;
    const int kv_head = row / padded_capacity;
    const std::int64_t source_row =
        static_cast<std::int64_t>(SourceCodeExtent) *
        (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);
    const std::int64_t source_scale_row =
        static_cast<std::int64_t>(ScaleExtent) *
        (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);
    const std::int64_t destination_row =
        static_cast<std::int64_t>(DestinationCodeExtent) *
        (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);
    const std::int64_t destination_scale_row =
        static_cast<std::int64_t>(ScaleExtent) *
        (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);

    float k_values[4];
    float v_values[4];
    const float k_scale = __bfloat162float(source_k_metadata[source_scale_row]);
    const float k_zero  = __bfloat162float(source_k_metadata[source_scale_row + 1]);
    const float v_scale = __bfloat162float(source_v_metadata[source_scale_row]);
    const float v_zero  = __bfloat162float(source_v_metadata[source_scale_row + 1]);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * 32;
        k_values[item] = fmaf(static_cast<float>(unpack_oscar_conversion_code<SourceBits>(
                                        source_k + source_row, d, source_q2_transposed)),
                              k_scale, k_zero);
        v_values[item] = fmaf(static_cast<float>(unpack_oscar_conversion_code<SourceBits>(
                                        source_v + source_row, d, source_q2_transposed)),
                              v_scale, v_zero);
    }
    const auto k_quant = cyclic_oscar_quant_params<DestinationBits, false>(k_values, lane);
    const auto v_quant = cyclic_oscar_quant_params<DestinationBits, true>(v_values, lane);
    std::uint8_t k_codes[4];
    std::uint8_t v_codes[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        k_codes[item] = cyclic_oscar_quantize<DestinationBits>(k_values[item], k_quant);
        v_codes[item] = cyclic_oscar_quantize_value<DestinationBits>(v_values[item], v_quant);
    }
    if (destination_q2_transposed && DestinationBits == 2) {
        const std::uint8_t packed_k = static_cast<std::uint8_t>(
            k_codes[0] | (k_codes[1] << 2) | (k_codes[2] << 4) | (k_codes[3] << 6));
        const std::uint8_t packed_v = static_cast<std::uint8_t>(
            v_codes[0] | (v_codes[1] << 2) | (v_codes[2] << 4) | (v_codes[3] << 6));
        destination_k[destination_row + lane] = packed_k;
        destination_v[destination_row + lane] = packed_v;
    } else {
        for (int byte = 0; byte < DestinationCodeExtent; ++byte) {
            const std::uint8_t packed_k = pack_oscar_byte<DestinationBits>(k_codes, byte);
            const std::uint8_t packed_v = pack_oscar_byte<DestinationBits>(v_codes, byte);
            if (lane == 0) {
                destination_k[destination_row + byte] = packed_k;
                destination_v[destination_row + byte] = packed_v;
            }
        }
    }
    if (lane == 0) {
        destination_k_metadata[destination_scale_row]     = __float2bfloat16(k_quant.scale);
        destination_k_metadata[destination_scale_row + 1] = __float2bfloat16(k_quant.zero);
        destination_v_metadata[destination_scale_row]     = __float2bfloat16(v_quant.scale);
        destination_v_metadata[destination_scale_row + 1] = __float2bfloat16(v_quant.zero);
    }
}

template <int SourceBits, int DestinationBits>
void launch_typed(const std::uint8_t* source_k, const __nv_bfloat16* source_k_metadata,
                  const std::uint8_t* source_v, const __nv_bfloat16* source_v_metadata,
                  std::uint8_t* destination_k, __nv_bfloat16* destination_k_metadata,
                  std::uint8_t* destination_v, __nv_bfloat16* destination_v_metadata,
                  int padded_capacity, int kv_heads, cudaStream_t stream,
                  bool source_q2_transposed, bool destination_q2_transposed) {
    const int rows = padded_capacity * kv_heads;
    cyclic_oscar_requantize_kernel<SourceBits, DestinationBits><<<rows, 32, 0, stream>>>(
        source_k, source_k_metadata, source_v, source_v_metadata, destination_k,
        destination_k_metadata, destination_v, destination_v_metadata, padded_capacity, kv_heads,
        source_q2_transposed, destination_q2_transposed);
}

} // namespace

void cyclic_oscar_requantize_launch(
    const std::uint8_t* source_k, const __nv_bfloat16* source_k_metadata,
    const std::uint8_t* source_v, const __nv_bfloat16* source_v_metadata,
    std::uint8_t* destination_k, __nv_bfloat16* destination_k_metadata,
    std::uint8_t* destination_v, __nv_bfloat16* destination_v_metadata, int source_bits,
    int destination_bits, int padded_capacity, int kv_heads, cudaStream_t stream,
    bool source_q2_transposed, bool destination_q2_transposed) {
    if (source_bits < 2 || source_bits > 4 || destination_bits < 2 || destination_bits > 4 ||
        padded_capacity <= 0 || kv_heads <= 0) {
        throw std::invalid_argument("OSCAR requantization geometry is invalid");
    }
#define NINFER_OSCAR_REQUANTIZE_DESTINATION(SOURCE_BITS)                                      \
    switch (destination_bits) {                                                               \
    case 2:                                                                                   \
        launch_typed<SOURCE_BITS, 2>(                                                         \
            source_k, source_k_metadata, source_v, source_v_metadata, destination_k,           \
            destination_k_metadata, destination_v, destination_v_metadata, padded_capacity,   \
            kv_heads, stream, source_q2_transposed, destination_q2_transposed);                \
        break;                                                                                 \
    case 3:                                                                                   \
        launch_typed<SOURCE_BITS, 3>(                                                         \
            source_k, source_k_metadata, source_v, source_v_metadata, destination_k,           \
            destination_k_metadata, destination_v, destination_v_metadata, padded_capacity,   \
            kv_heads, stream, source_q2_transposed, destination_q2_transposed);                \
        break;                                                                                 \
    case 4:                                                                                   \
        launch_typed<SOURCE_BITS, 4>(                                                         \
            source_k, source_k_metadata, source_v, source_v_metadata, destination_k,           \
            destination_k_metadata, destination_v, destination_v_metadata, padded_capacity,   \
            kv_heads, stream, source_q2_transposed, destination_q2_transposed);                \
        break;                                                                                 \
    default:                                                                                  \
        throw std::invalid_argument("OSCAR destination bit width is invalid");               \
    }
    switch (source_bits) {
    case 2:
        NINFER_OSCAR_REQUANTIZE_DESTINATION(2)
        break;
    case 3:
        NINFER_OSCAR_REQUANTIZE_DESTINATION(3)
        break;
    case 4:
        NINFER_OSCAR_REQUANTIZE_DESTINATION(4)
        break;
    default:
        throw std::invalid_argument("OSCAR source bit width is invalid");
    }
#undef NINFER_OSCAR_REQUANTIZE_DESTINATION
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
