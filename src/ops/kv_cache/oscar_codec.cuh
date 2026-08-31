#pragma once

// Experimental OSCAR-style affine KV codec for the Qwen3.8 DFlash2 local cache.
//
// The current branch has no model-specific OSCAR covariance/bit-reversal assets, so the runtime
// uses the available normalized Hadamard factor as the attention-aware rotation and keeps the
// calibration boundary explicit: every token gets independently fitted affine K/V parameters,
// with separate clipping ratios for K and V and a BF16 (scale, zero) metadata pair.  This is a
// real packed Q2/Q4 path, not an alias for NVFP4.  Replacing the fixed rotation with trained
// OSCAR U/P_br matrices later does not change the storage or attention contract.

#include "ops/common/warp.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCyclicKVCacheOscarHeadDim       = 128;
inline constexpr int kCyclicKVCacheOscarKVHeads       = 8;
inline constexpr int kCyclicKVCacheOscarQuantGroup    = 128;
inline constexpr int kCyclicKVCacheOscarScaleExtent   = 2;
inline constexpr float kCyclicKVCacheOscarRotationScale = 0.08838834764831844F;

// Full-attention Qwen3.8 pages use D=256 and four (or two) KV heads.  This is the same affine
// OSCAR contract as the DFlash cyclic cache, but with paged head-major addressing and a Q2 code
// stream of 64 bytes per row.  Keeping the constants separate prevents the DFlash D=128 helper
// from accidentally being used for the target cache.
inline constexpr int kPagedKVCacheOscarHeadDim     = 256;
inline constexpr int kPagedKVCacheOscarQuantGroup  = 128;
inline constexpr int kPagedKVCacheOscarCodeExtent  = 64;
inline constexpr int kPagedKVCacheOscarScaleExtent = 2;

template <int Bits>
__host__ __device__ constexpr int cyclic_oscar_code_extent(int head_dim) {
    static_assert(Bits >= 2 && Bits <= 4);
    return (head_dim * Bits + 7) / 8;
}

template <int Bits>
__device__ __forceinline__ std::int64_t cyclic_oscar_code_index(
    int slot, int kv_head, int lane, int padded_capacity, int code_byte) {
    constexpr int CodeExtent = cyclic_oscar_code_extent<Bits>(kCyclicKVCacheOscarHeadDim);
    const std::int64_t lane_stride = static_cast<std::int64_t>(CodeExtent) * padded_capacity *
                                     kCyclicKVCacheOscarKVHeads;
    return static_cast<std::int64_t>(code_byte) +
           static_cast<std::int64_t>(CodeExtent) *
               (slot + static_cast<std::int64_t>(padded_capacity) * kv_head) +
           lane_stride * lane;
}

__device__ __forceinline__ std::int64_t cyclic_oscar_scale_index(
    int slot, int kv_head, int lane, int padded_capacity, int item) {
    const std::int64_t lane_stride = static_cast<std::int64_t>(
        kCyclicKVCacheOscarScaleExtent) * padded_capacity * kCyclicKVCacheOscarKVHeads;
    return static_cast<std::int64_t>(item) +
           static_cast<std::int64_t>(kCyclicKVCacheOscarScaleExtent) *
               (slot + static_cast<std::int64_t>(padded_capacity) * kv_head) +
           lane_stride * lane;
}

// H128 = H4(H32 columns), normalized by 1/sqrt(128).  It is self-inverse, so the same helper is
// used when preparing Q/K/V and when returning an attention accumulator to model coordinates.
__device__ __forceinline__ void normalized_hadamard_d128_inplace(float (&values)[4], int lane) {
    hadamard_d32_columns_inplace(values, lane);
#pragma unroll
    for (int span = 1; span < 4; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 4; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low       = values[base + offset];
                const float high      = values[base + offset + span];
                values[base + offset] = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }
#pragma unroll
    for (float& value : values) {
        value = __fmul_rn(value, kCyclicKVCacheOscarRotationScale);
    }
}

// DFlash-Q2 packed-row layout. Lane l owns dimensions [4*l, 4*l+4), so one
// Q2 byte supplies all four values for the lane. H128 is factored over the
// same dimension bits: H4 within the lane registers followed by H32 across
// lanes. The helper is also used for the inverse transform because H128 is
// normalized and self-inverse.
__device__ __forceinline__ void normalized_hadamard_d128_packed_inplace(
    float (&values)[4], int lane) {
#pragma unroll
    for (int stride = 1; stride <= 2; stride <<= 1) {
#pragma unroll
        for (int base = 0; base < 4; base += 2 * stride) {
#pragma unroll
            for (int offset = 0; offset < stride; ++offset) {
                const float low       = values[base + offset];
                const float high      = values[base + offset + stride];
                values[base + offset] = __fadd_rn(low, high);
                values[base + offset + stride] = __fsub_rn(low, high);
            }
        }
    }
    hadamard_d32_columns_inplace(values, lane);
#pragma unroll
    for (float& value : values) {
        value = __fmul_rn(value, kCyclicKVCacheOscarRotationScale);
    }
}

template <int Bits>
__device__ __forceinline__ std::uint8_t cyclic_oscar_unpack_code(const std::uint8_t* codes,
                                                                  int dimension) {
    static_assert(Bits >= 2 && Bits <= 4);
    constexpr int Mask = (1 << Bits) - 1;
    const int bit       = dimension * Bits;
    const int byte      = bit >> 3;
    const int shift     = bit & 7;
    std::uint32_t word  = codes[byte];
    if (shift + Bits > 8) { word |= static_cast<std::uint32_t>(codes[byte + 1]) << 8U; }
    return static_cast<std::uint8_t>((word >> shift) & Mask);
}

template <int Bits>
__device__ __forceinline__ float cyclic_oscar_decode_value(
    const std::uint8_t* codes, const __nv_bfloat16* metadata, int dimension) {
    const float scale = __bfloat162float(metadata[0]);
    const float zero  = __bfloat162float(metadata[1]);
    return fmaf(static_cast<float>(cyclic_oscar_unpack_code<Bits>(codes, dimension)), scale, zero);
}

template <int Bits, bool IsValue>
__device__ __forceinline__ float cyclic_oscar_clip_ratio() {
    // V is more sensitive to outliers in the weighted sum, so it receives a slightly tighter
    // fit. Q2 uses more aggressive clipping to spend its smaller codebook on the dense center;
    // Q4 retains more tails. K/V and each bit width are intentionally calibrated independently.
    if constexpr (Bits == 2) { return IsValue ? 0.91F : 0.93F; }
    if constexpr (Bits == 3) { return IsValue ? 0.95F : 0.965F; }
    return IsValue ? 0.98F : 0.985F;
}

template <int Bits, bool IsValue>
struct OscarAffineQuantParams {
    float scale;
    float zero;
};

template <int Bits, bool IsValue>
__device__ __forceinline__ OscarAffineQuantParams<Bits, IsValue>
cyclic_oscar_quant_params(float (&values)[4], int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr int Levels         = (1 << Bits) - 1;
    float local_min              = values[0];
    float local_max              = values[0];
#pragma unroll
    for (int item = 1; item < 4; ++item) {
        local_min = fminf(local_min, values[item]);
        local_max = fmaxf(local_max, values[item]);
    }
    const float min_value = -warp_max(-local_min, FullMask);
    const float max_value = warp_max(local_max, FullMask);
    const float center    = __fmul_rn(__fadd_rn(min_value, max_value), 0.5F);
    const float half_span = __fmul_rn(__fsub_rn(max_value, min_value),
                                      0.5F * cyclic_oscar_clip_ratio<Bits, IsValue>());
    const float zero      = __fsub_rn(center, half_span);
    const float span      = __fmul_rn(half_span, 2.0F);
    const float scale     = span > 1.0e-8F ? span / static_cast<float>(Levels) : 1.0F;
    (void)lane;
    return {.scale = scale, .zero = zero};
}

template <int Bits>
__device__ __forceinline__ std::uint8_t cyclic_oscar_quantize(float value,
                                                                OscarAffineQuantParams<Bits, false> q) {
    const float normalized = (value - q.zero) / q.scale;
    const int code         = max(0, min((1 << Bits) - 1, __float2int_rn(normalized)));
    return static_cast<std::uint8_t>(code);
}

template <int Bits>
__device__ __forceinline__ std::uint8_t cyclic_oscar_quantize_value(
    float value, OscarAffineQuantParams<Bits, true> q) {
    const float normalized = (value - q.zero) / q.scale;
    const int code         = max(0, min((1 << Bits) - 1, __float2int_rn(normalized)));
    return static_cast<std::uint8_t>(code);
}

template <int Bits, typename Geometry>
__device__ __forceinline__ std::int64_t paged_oscar_code_index(int physical_page, int kv_head,
                                                               int page_offset, int code_byte) {
    constexpr int CodeExtent = cyclic_oscar_code_extent<Bits>(kPagedKVCacheOscarHeadDim);
    return paged_kv_element_offset<CodeExtent, Geometry::KVHeads>(physical_page, kv_head,
                                                                   page_offset, code_byte);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t paged_oscar_scale_index(int physical_page, int kv_head,
                                                                int page_offset, int item) {
    return paged_kv_element_offset<kPagedKVCacheOscarScaleExtent, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, item);
}

template <int Bits>
__device__ __forceinline__ float paged_oscar_decode_value(const std::uint8_t* codes,
                                                         const __nv_bfloat16* metadata,
                                                         int dimension) {
    return cyclic_oscar_decode_value<Bits>(codes, metadata, dimension);
}

// Full-attention rows distribute D=256 over one warp as eight values per lane.  This overload
// keeps K and V calibration independent while sharing the same attention-aware rotation.
template <int Bits, bool IsValue>
__device__ __forceinline__ OscarAffineQuantParams<Bits, IsValue>
cyclic_oscar_quant_params(float (&values)[8], int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr int Levels         = (1 << Bits) - 1;
    float local_min              = values[0];
    float local_max              = values[0];
#pragma unroll
    for (int item = 1; item < 8; ++item) {
        local_min = fminf(local_min, values[item]);
        local_max = fmaxf(local_max, values[item]);
    }
    const float min_value = -warp_max(-local_min, FullMask);
    const float max_value = warp_max(local_max, FullMask);
    const float center    = __fmul_rn(__fadd_rn(min_value, max_value), 0.5F);
    const float half_span = __fmul_rn(__fsub_rn(max_value, min_value),
                                      0.5F * cyclic_oscar_clip_ratio<Bits, IsValue>());
    const float zero      = __fsub_rn(center, half_span);
    const float span      = __fmul_rn(half_span, 2.0F);
    const float scale     = span > 1.0e-8F ? span / static_cast<float>(Levels) : 1.0F;
    (void)lane;
    return {.scale = scale, .zero = zero};
}

} // namespace ninfer::ops
