#pragma once

// NVFP4 KV-cache codec used by the opt-in VeriCache draft path. Values are packed as two
// signed E2M1 nibbles per byte. Each contiguous group of sixteen values has one positive E4M3
// scale byte, matching the existing NVFP4 weight codec's scale/code contract.

#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheNvfp4HeadDim       = 256;
inline constexpr int kKVCacheNvfp4Group         = 16;
inline constexpr int kKVCacheNvfp4Groups        =
    kKVCacheNvfp4HeadDim / kKVCacheNvfp4Group;
inline constexpr int kKVCacheNvfp4CodeExtent    = kKVCacheNvfp4HeadDim / 2;
inline constexpr int kKVCacheNvfp4ScaleExtent   = kKVCacheNvfp4Groups;
inline constexpr float kKVCacheNvfp4MaxCode     = 6.0F;

// DFlash2 uses a smaller 128-wide local head.  It shares the same packed E2M1/E4M3 contract but
// has its own leading extents and cyclic addressing, so it cannot reuse the paged D256 helpers.
inline constexpr int kCyclicKVCacheNvfp4HeadDim     = 128;
inline constexpr int kCyclicKVCacheNvfp4KVHeads     = 8;
inline constexpr int kCyclicKVCacheNvfp4Group       = 16;
inline constexpr int kCyclicKVCacheNvfp4CodeExtent  = kCyclicKVCacheNvfp4HeadDim / 2;
inline constexpr int kCyclicKVCacheNvfp4ScaleExtent =
    kCyclicKVCacheNvfp4HeadDim / kCyclicKVCacheNvfp4Group;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_nvfp4_code_index(int physical_page, int kv_head, int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheNvfp4CodeExtent, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_nvfp4_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheNvfp4ScaleExtent, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_nvfp4_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheNvfp4HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

__device__ __forceinline__ std::int64_t
cyclic_nvfp4_code_index(int slot, int kv_head, int lane, int padded_capacity, int d_pair) {
    const std::int64_t lane_stride = static_cast<std::int64_t>(
        kCyclicKVCacheNvfp4CodeExtent) * padded_capacity * kCyclicKVCacheNvfp4KVHeads;
    return static_cast<std::int64_t>(d_pair) +
           static_cast<std::int64_t>(kCyclicKVCacheNvfp4CodeExtent) *
               (slot + static_cast<std::int64_t>(padded_capacity) * kv_head) +
           lane_stride * lane;
}

__device__ __forceinline__ std::int64_t
cyclic_nvfp4_scale_index(int slot, int kv_head, int lane, int padded_capacity, int group) {
    const std::int64_t lane_stride = static_cast<std::int64_t>(
        kCyclicKVCacheNvfp4ScaleExtent) * padded_capacity * kCyclicKVCacheNvfp4KVHeads;
    return static_cast<std::int64_t>(group) +
           static_cast<std::int64_t>(kCyclicKVCacheNvfp4ScaleExtent) *
               (slot + static_cast<std::int64_t>(padded_capacity) * kv_head) +
           lane_stride * lane;
}

__device__ __forceinline__ float kv_cache_nvfp4_decode_scale(std::uint8_t encoded) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(encoded) |
                (static_cast<std::uint16_t>(encoded) << 8U);
    return static_cast<float2>(value).x;
}

__device__ __forceinline__ float2 kv_cache_nvfp4_decode_pair(std::uint8_t encoded) {
    __nv_fp4x2_e2m1 value;
    value.__x = encoded;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float kv_cache_nvfp4_decode_value(const std::uint8_t* codes,
                                                              const std::uint8_t* scales, int d,
                                                              float scale_multiplier = 1.0F) {
    const float2 pair = kv_cache_nvfp4_decode_pair(codes[d >> 1]);
    const float code   = (d & 1) == 0 ? pair.x : pair.y;
    return code * kv_cache_nvfp4_decode_scale(scales[d >> 4]) * scale_multiplier;
}

struct KVCacheNvfp4QuantParams {
    std::uint8_t scale;
    float inverse_scale;
};

__device__ __forceinline__ KVCacheNvfp4QuantParams
kv_cache_nvfp4_quant_params(float absmax) {
    if (!(absmax > 0.0F)) { return {.scale = 0, .inverse_scale = 0.0F}; }
    const std::uint8_t scale = __nv_cvt_float_to_fp8(
        absmax / kKVCacheNvfp4MaxCode, __NV_SATFINITE, __NV_E4M3);
    const float represented = kv_cache_nvfp4_decode_scale(scale);
    return {.scale = scale, .inverse_scale = represented > 0.0F ? 1.0F / represented : 0.0F};
}

__device__ __forceinline__ std::uint8_t
kv_cache_nvfp4_quantize_pair(float x0, float x1, float inverse_scale) {
    if (!(inverse_scale > 0.0F)) { return 0; }
    return static_cast<std::uint8_t>(__nv_cvt_float2_to_fp4x2(
        make_float2(x0 * inverse_scale, x1 * inverse_scale), __NV_E2M1, cudaRoundNearest));
}

} // namespace ninfer::ops
