#pragma once

// Experimental low-bit codec for the DFlash2 local cyclic KV cache. The cache is still backed by
// U8 tensors, but 2- and 3-bit symbols are packed consecutively inside each 16-value scale group.
// The format is deliberately separate from NVFP4: protected ranges remain BF16. OSCAR-Q2/Q3
// host snapshots use the separate affine Q4 path; this helper remains the legacy 16-value-group
// low-bit codec used only by non-OSCAR profiles.

#include "ops/kv_cache/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCyclicKVLowBitQuantGroup = 16;

template <int Bits>
__host__ __device__ constexpr int cyclic_kv_lowbit_code_extent(int head_dim) {
    return (head_dim * Bits + 7) / 8;
}

template <int Bits>
__host__ __device__ constexpr int cyclic_kv_lowbit_group_bytes() {
    return (kCyclicKVLowBitQuantGroup * Bits + 7) / 8;
}

template <int Bits>
__device__ __forceinline__ float cyclic_kv_lowbit_code_value(int code) {
    constexpr int levels = 1 << Bits;
    return -1.0F + 2.0F * static_cast<float>(code) / static_cast<float>(levels - 1);
}

template <int Bits>
__device__ __forceinline__ std::uint8_t cyclic_kv_lowbit_decode_code(
    const std::uint8_t* codes, int dimension) {
    constexpr int mask = (1 << Bits) - 1;
    const int group = dimension / kCyclicKVLowBitQuantGroup;
    const int in_group = dimension % kCyclicKVLowBitQuantGroup;
    const int bit = in_group * Bits;
    const int byte = group * cyclic_kv_lowbit_group_bytes<Bits>() + (bit >> 3);
    const int shift = bit & 7;
    std::uint32_t word = codes[byte];
    if (shift + Bits > 8) { word |= static_cast<std::uint32_t>(codes[byte + 1]) << 8U; }
    return static_cast<std::uint8_t>((word >> shift) & mask);
}

template <int Bits>
__device__ __forceinline__ float cyclic_kv_lowbit_decode_value(const std::uint8_t* codes,
                                                                const std::uint8_t* scales,
                                                                int dimension) {
    const float scale = kv_cache_nvfp4_decode_scale(
        scales[dimension / kCyclicKVLowBitQuantGroup]);
    return cyclic_kv_lowbit_code_value<Bits>(
               cyclic_kv_lowbit_decode_code<Bits>(codes, dimension)) *
           scale;
}

template <int Bits>
struct CyclicKVLowBitQuantParams {
    std::uint8_t scale;
    float inverse_scale;
};

template <int Bits>
__device__ __forceinline__ CyclicKVLowBitQuantParams<Bits>
cyclic_kv_lowbit_quant_params(float absmax) {
    if (!(absmax > 0.0F)) { return {.scale = 0, .inverse_scale = 0.0F}; }
    const std::uint8_t scale =
        __nv_cvt_float_to_fp8(absmax, __NV_SATFINITE, __NV_E4M3);
    const float represented = kv_cache_nvfp4_decode_scale(scale);
    return {.scale = scale, .inverse_scale = represented > 0.0F ? 1.0F / represented : 0.0F};
}

template <int Bits>
__device__ __forceinline__ std::uint8_t cyclic_kv_lowbit_quantize(float value,
                                                                   float inverse_scale) {
    if (!(inverse_scale > 0.0F)) { return 0; }
    constexpr int levels = 1 << Bits;
    const float normalized = fminf(1.0F, fmaxf(-1.0F, value * inverse_scale));
    int best = 0;
    float best_error = CUDART_INF_F;
#pragma unroll
    for (int code = 0; code < levels; ++code) {
        const float error = fabsf(normalized - cyclic_kv_lowbit_code_value<Bits>(code));
        if (error < best_error) {
            best_error = error;
            best = code;
        }
    }
    return static_cast<std::uint8_t>(best);
}

template <int Bits>
__device__ __forceinline__ std::int64_t cyclic_kv_lowbit_code_index(
    int slot, int kv_head, int lane, int padded_capacity, int code_byte) {
    const std::int64_t lane_stride = static_cast<std::int64_t>(
        cyclic_kv_lowbit_code_extent<Bits>(kCyclicKVCacheNvfp4HeadDim)) * padded_capacity *
                                      kCyclicKVCacheNvfp4KVHeads;
    return static_cast<std::int64_t>(code_byte) +
           static_cast<std::int64_t>(cyclic_kv_lowbit_code_extent<Bits>(
               kCyclicKVCacheNvfp4HeadDim)) *
               (slot + static_cast<std::int64_t>(padded_capacity) * kv_head) +
           lane_stride * lane;
}

template <int Bits>
__device__ __forceinline__ std::int64_t cyclic_kv_lowbit_scale_index(
    int slot, int kv_head, int lane, int padded_capacity, int group) {
    constexpr int ScaleExtent = kCyclicKVCacheNvfp4HeadDim / kCyclicKVLowBitQuantGroup;
    const std::int64_t lane_stride = static_cast<std::int64_t>(ScaleExtent) * padded_capacity *
                                      kCyclicKVCacheNvfp4KVHeads;
    return static_cast<std::int64_t>(group) +
           static_cast<std::int64_t>(ScaleExtent) *
               (slot + static_cast<std::int64_t>(padded_capacity) * kv_head) +
           lane_stride * lane;
}

} // namespace ninfer::ops
