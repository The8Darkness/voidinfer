#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/low_bit_codec.cuh"
#include "ops/kv_cache/nvfp4_codec.cuh"
#include "ops/kv_cache/oscar_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheAppendFullHeadDim = 256;

template <int KVHeadsValue>
struct KVCacheAppendFullGeometry {
    static_assert(KVHeadsValue == 4 || KVHeadsValue == 2);
    static constexpr int KVHeads = KVHeadsValue;
};

using KVCacheAppendD256Kv4 = KVCacheAppendFullGeometry<4>;
using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<2>;

struct KVCacheAppendDirectMetadata {
    const std::int32_t* table;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }
};

template <bool Masked>
struct KVCacheAppendBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

template <typename Geometry>
__device__ __forceinline__ void
kv_cache_append_full_fp8_row(const __nv_bfloat16* __restrict__ k,
                             const __nv_bfloat16* __restrict__ v,
                             std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                             __half* __restrict__ scale_k, __half* __restrict__ scale_v, int token,
                             int kv_head, int physical_page, int page_off, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    float values[8];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(k[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (float value : values) { local_absmax = fmaxf(local_absmax, fabsf(value)); }
    const auto k_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_k[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_off)] =
            kv_cache_fp8_quant_code(values[r], k_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_k[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_off)] =
            k_quant.scale;
    }

    local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d  = lane + 32 * r;
        values[r]    = __bfloat162float(v[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
        local_absmax = fmaxf(local_absmax, fabsf(values[r]));
    }
    const auto v_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_v[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_off)] =
            kv_cache_fp8_quant_code(values[r], v_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_v[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_off)] =
            v_quant.scale;
    }
}

template <typename Geometry, typename Metadata>
__global__ void kv_cache_append_full_bf16_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, std::int32_t width) {
    constexpr int VecElems = 8;
    const int tokens       = metadata.valid_tokens(width);
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n   = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                           (kKVCacheAppendFullHeadDim / VecElems);
    if (idx >= n) return;

    const int vec      = static_cast<int>(idx % (kKVCacheAppendFullHeadDim / VecElems));
    const int tmp      = static_cast<int>(idx / (kKVCacheAppendFullHeadDim / VecElems));
    const int kv_head  = tmp % Geometry::KVHeads;
    const int token    = tmp / Geometry::KVHeads;
    const int d        = vec * VecElems;
    const int position = positions[0] + token;
    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const std::int64_t src_off =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendFullHeadDim) *
                                           (kv_head + Geometry::KVHeads * token);
    const int4 k_value = load_vec<int4>(&k[src_off]);
    const int4 v_value = load_vec<int4>(&v[src_off]);
    physical_page      = __shfl_sync(0xffffffffu, physical_page, 0);
    const std::int64_t cache_off =
        paged_kv_element_offset<kKVCacheAppendFullHeadDim, Geometry::KVHeads>(
            physical_page, kv_head, position & kPagedKVPageMask, d);
    store_vec(&cache_k[cache_off], k_value);
    store_vec(&cache_v[cache_off], v_value);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_kernel(const __nv_bfloat16* __restrict__ k,
                                         const __nv_bfloat16* __restrict__ v,
                                         const std::int32_t* __restrict__ positions,
                                         Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                         std::uint8_t* __restrict__ cache_v,
                                         __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                         std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                           physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_fp8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                              const __nv_bfloat16* __restrict__ v,
                                              const std::int32_t* __restrict__ positions,
                                              Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                              std::uint8_t* __restrict__ cache_v,
                                              __half* __restrict__ scale_k,
                                              __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    const int position              = base_position + token;
    kv_cache_append_full_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head,
                                           physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i8_kernel(const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v,
                                        const std::int32_t* __restrict__ positions,
                                        Metadata metadata, std::int8_t* __restrict__ cache_k,
                                        std::int8_t* __restrict__ cache_v,
                                        __half* __restrict__ scale_k, __half* __restrict__ scale_v,
                                        std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int page                        = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const int page_off              = position & kPagedKVPageMask;
    page                            = __shfl_sync(FullMask, page, 0);

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const int d0                 = group * kKVCacheInt8Group + lane;
        const int d1                 = d0 + 32;
        const float k0               = k_values[2 * group];
        const float k1               = k_values[2 * group + 1];
        const std::int64_t src0      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        const float v0               = __bfloat162float(v[src0]);
        const float v1               = __bfloat162float(v[src1]);
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs            = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const auto v_quant           = kv_cache_int8_quant_params(v_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_off =
                kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, group, page_off);
            scale_k[scale_off] = k_quant.scale;
            scale_v[scale_off] = v_quant.scale;
        }
    }
}

template <typename Geometry>
__device__ __forceinline__ void kv_cache_append_full_nvfp4_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, int token,
    int kv_head, int physical_page, int page_off, int lane) {
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr int ValuesPerLane = kKVCacheNvfp4HeadDim / 32;
    const int group             = lane >> 1;
    const int group_lane        = lane & 1;
    const int d_base            = group * kKVCacheNvfp4Group + group_lane * ValuesPerLane;
    float k_values[ValuesPerLane];
    float v_values[ValuesPerLane];
    float k_absmax = 0.0F;
    float v_absmax = 0.0F;
#pragma unroll
    for (int i = 0; i < ValuesPerLane; ++i) {
        const int d = d_base + i;
        k_values[i] = __bfloat162float(k[kv_cache_nvfp4_src_index<Geometry>(kv_head, d, token)]);
        v_values[i] = __bfloat162float(v[kv_cache_nvfp4_src_index<Geometry>(kv_head, d, token)]);
        k_absmax   = fmaxf(k_absmax, fabsf(k_values[i]));
        v_absmax   = fmaxf(v_absmax, fabsf(v_values[i]));
    }
    // Two lanes own one sixteen-value group. Restrict the reduction to that pair so neighboring
    // groups can be quantized independently without a second synchronization point.
    k_absmax = fmaxf(k_absmax, __shfl_xor_sync(FullMask, k_absmax, 1));
    v_absmax = fmaxf(v_absmax, __shfl_xor_sync(FullMask, v_absmax, 1));
    const KVCacheNvfp4QuantParams k_quant = kv_cache_nvfp4_quant_params(k_absmax);
    const KVCacheNvfp4QuantParams v_quant = kv_cache_nvfp4_quant_params(v_absmax);

#pragma unroll
    for (int pair = 0; pair < ValuesPerLane / 2; ++pair) {
        const int d = d_base + 2 * pair;
        cache_k[kv_cache_nvfp4_code_index<Geometry>(physical_page, kv_head, d >> 1, page_off)] =
            kv_cache_nvfp4_quantize_pair(k_values[2 * pair], k_values[2 * pair + 1],
                                         k_quant.inverse_scale);
        cache_v[kv_cache_nvfp4_code_index<Geometry>(physical_page, kv_head, d >> 1, page_off)] =
            kv_cache_nvfp4_quantize_pair(v_values[2 * pair], v_values[2 * pair + 1],
                                         v_quant.inverse_scale);
    }
    if (group_lane == 0) {
        scale_k[kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, group, page_off)] =
            k_quant.scale;
        scale_v[kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, group, page_off)] =
            v_quant.scale;
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_nvfp4_kernel(const __nv_bfloat16* __restrict__ k,
                                           const __nv_bfloat16* __restrict__ v,
                                           const std::int32_t* __restrict__ positions,
                                           Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                           std::uint8_t* __restrict__ cache_v,
                                           std::uint8_t* __restrict__ scale_k,
                                           std::uint8_t* __restrict__ scale_v,
                                           std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_nvfp4_row<Geometry>(
        k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head, physical_page,
        position & kPagedKVPageMask, lane);
}

template <typename Geometry, int Bits, bool TransposedQ2>
__device__ __forceinline__ void kv_cache_append_full_oscar_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __nv_bfloat16* __restrict__ scale_k, __nv_bfloat16* __restrict__ scale_v, int token,
    int kv_head, int physical_page, int page_off, int lane) {
    static_assert(Bits == 2 || Bits == 4);
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr int D             = kPagedKVCacheOscarHeadDim;
    constexpr int CodeExtent    = cyclic_oscar_code_extent<Bits>(D);
    float k_values[8];
    float v_values[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int d = lane + 32 * item;
        k_values[item] = __bfloat162float(k[kv_cache_nvfp4_src_index<Geometry>(kv_head, d, token)]);
        v_values[item] = __bfloat162float(v[kv_cache_nvfp4_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);
    normalized_hadamard_d256_inplace(v_values, lane);
    const auto k_quant = cyclic_oscar_quant_params<Bits, false>(k_values, lane);
    const auto v_quant = cyclic_oscar_quant_params<Bits, true>(v_values, lane);
    std::uint8_t k_codes[8];
    std::uint8_t v_codes[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        k_codes[item] = cyclic_oscar_quantize<Bits>(k_values[item], k_quant);
        v_codes[item] = cyclic_oscar_quantize_value<Bits>(v_values[item], v_quant);
    }

    if constexpr (TransposedQ2) {
        static_assert(Bits == 2);
        // The attention decoder owns one lane per byte.  Keep the four values separated by the
        // natural H256 row stride in adjacent bit fields, so append becomes two coalesced stores
        // per lane instead of rebuilding all 64 contiguous bytes through warp shuffles.
        const std::uint8_t packed_k0 = static_cast<std::uint8_t>(
            k_codes[0] | (k_codes[1] << 2) | (k_codes[2] << 4) | (k_codes[3] << 6));
        const std::uint8_t packed_k1 = static_cast<std::uint8_t>(
            k_codes[4] | (k_codes[5] << 2) | (k_codes[6] << 4) | (k_codes[7] << 6));
        const std::uint8_t packed_v0 = static_cast<std::uint8_t>(
            v_codes[0] | (v_codes[1] << 2) | (v_codes[2] << 4) | (v_codes[3] << 6));
        const std::uint8_t packed_v1 = static_cast<std::uint8_t>(
            v_codes[4] | (v_codes[5] << 2) | (v_codes[6] << 4) | (v_codes[7] << 6));
        cache_k[paged_oscar_code_index<Bits, Geometry>(physical_page, kv_head, page_off, lane)] =
            packed_k0;
        cache_k[paged_oscar_code_index<Bits, Geometry>(physical_page, kv_head, page_off,
                                                        lane + 32)] = packed_k1;
        cache_v[paged_oscar_code_index<Bits, Geometry>(physical_page, kv_head, page_off, lane)] =
            packed_v0;
        cache_v[paged_oscar_code_index<Bits, Geometry>(physical_page, kv_head, page_off,
                                                        lane + 32)] = packed_v1;
    } else {
        // Pack the contiguous symbol stream from lane-local rotated values.  Lane zero owns each
        // byte, so Q2/Q4 writes are race-free even when the page is shared by concurrent rows.
        for (int byte = 0; byte < CodeExtent; ++byte) {
            std::uint8_t packed_k = 0;
            std::uint8_t packed_v = 0;
            int bit_cursor         = 0;
            while (bit_cursor < 8) {
                const int global_bit = byte * 8 + bit_cursor;
                const int dimension  = global_bit / Bits;
                const int bit_offset = global_bit - dimension * Bits;
                const int take       = min(Bits - bit_offset, 8 - bit_cursor);
                const int source_lane = dimension & 31;
                const int source_item = dimension >> 5;
                const int k_code = __shfl_sync(FullMask, static_cast<int>(k_codes[source_item]),
                                               source_lane);
                const int v_code = __shfl_sync(FullMask, static_cast<int>(v_codes[source_item]),
                                               source_lane);
                const int mask = (1 << take) - 1;
                packed_k |=
                    static_cast<std::uint8_t>(((k_code >> bit_offset) & mask) << bit_cursor);
                packed_v |=
                    static_cast<std::uint8_t>(((v_code >> bit_offset) & mask) << bit_cursor);
                bit_cursor += take;
            }
            if (lane == 0) {
                cache_k[paged_oscar_code_index<Bits, Geometry>(physical_page, kv_head, page_off,
                                                                byte)] = packed_k;
                cache_v[paged_oscar_code_index<Bits, Geometry>(physical_page, kv_head, page_off,
                                                                byte)] = packed_v;
            }
        }
    }
    if (lane == 0) {
        const std::int64_t scale_offset =
            paged_oscar_scale_index<Geometry>(physical_page, kv_head, page_off, 0);
        scale_k[scale_offset]     = __float2bfloat16(k_quant.scale);
        scale_k[scale_offset + 1] = __float2bfloat16(k_quant.zero);
        scale_v[scale_offset]     = __float2bfloat16(v_quant.scale);
        scale_v[scale_offset + 1] = __float2bfloat16(v_quant.zero);
    }
}

template <typename Geometry, int Bits, typename Metadata, bool TransposedQ2>
__launch_bounds__(256) __global__ void kv_cache_append_full_oscar_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __nv_bfloat16* __restrict__ scale_k, __nv_bfloat16* __restrict__ scale_v,
    std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_oscar_row<Geometry, Bits, TransposedQ2>(
        k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head, physical_page,
        position & kPagedKVPageMask, lane);
}

template <typename Geometry, int Bits, bool Masked, bool TransposedQ2>
__launch_bounds__(32) __global__ void kv_cache_append_oscar_batch_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ table_rows, std::uint8_t* __restrict__ cache_k,
    std::uint8_t* __restrict__ cache_v, __nv_bfloat16* __restrict__ scale_k,
    __nv_bfloat16* __restrict__ scale_v, const std::int32_t* __restrict__ block_tables,
    std::int32_t table_stride, std::int32_t logical_pages, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t width) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int token   = static_cast<int>(blockIdx.y);
    const int batch   = static_cast<int>(blockIdx.z);
    const int lane    = static_cast<int>(threadIdx.x);
    int valid_tokens  = width;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, width);
    }
    if (kv_head >= Geometry::KVHeads || token >= valid_tokens) return;

    const std::int64_t batch_column = static_cast<std::int64_t>(batch) * full_width +
                                      column_begin;
    const int position               = positions[batch_column + token];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_rows[batch]) * table_stride;
    const int physical_page = paged_kv_physical_page(block_table, position);
    const std::int64_t input_offset =
        static_cast<std::int64_t>(kPagedKVCacheOscarHeadDim) * Geometry::KVHeads * batch_column;
    (void)logical_pages;
    kv_cache_append_full_oscar_row<Geometry, Bits, TransposedQ2>(
        k + input_offset, v + input_offset, cache_k, cache_v, scale_k, scale_v, token, kv_head,
        physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, bool Masked>
__launch_bounds__(32) __global__ void kv_cache_append_nvfp4_batch_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ table_rows, std::uint8_t* __restrict__ cache_k,
    std::uint8_t* __restrict__ cache_v, std::uint8_t* __restrict__ scale_k,
    std::uint8_t* __restrict__ scale_v, const std::int32_t* __restrict__ block_tables,
    std::int32_t table_stride, std::int32_t logical_pages, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t width) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int token   = static_cast<int>(blockIdx.y);
    const int batch   = static_cast<int>(blockIdx.z);
    const int lane    = static_cast<int>(threadIdx.x);
    int valid_tokens  = width;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, width);
    }
    if (kv_head >= Geometry::KVHeads || token >= valid_tokens) return;

    const std::int64_t batch_column = static_cast<std::int64_t>(batch) * full_width +
                                      column_begin;
    const int position               = positions[batch_column + token];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_rows[batch]) * table_stride;
    const int physical_page = paged_kv_physical_page(block_table, position);
    const std::int64_t input_offset =
        static_cast<std::int64_t>(kKVCacheNvfp4HeadDim) * Geometry::KVHeads * batch_column;
    kv_cache_append_full_nvfp4_row<Geometry>(
        k + input_offset, v + input_offset, cache_k, cache_v, scale_k, scale_v, token, kv_head,
        physical_page, position & kPagedKVPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_i8_page_kernel(const __nv_bfloat16* __restrict__ k,
                                             const __nv_bfloat16* __restrict__ v,
                                             const std::int32_t* __restrict__ positions,
                                             Metadata metadata, std::int8_t* __restrict__ cache_k,
                                             std::int8_t* __restrict__ cache_v,
                                             __half* __restrict__ scale_k,
                                             __half* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffu;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);

    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;

    float k_values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        k_values[r] =
            __bfloat162float(k[kv_cache_int8_quant_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(k_values, lane);

#pragma unroll
    for (int group = 0; group < kKVCacheInt8Groups; ++group) {
        const int d0                 = group * kKVCacheInt8Group + lane;
        const int d1                 = d0 + 32;
        const float k0               = k_values[2 * group];
        const float k1               = k_values[2 * group + 1];
        const std::int64_t src0      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d0, token);
        const std::int64_t src1      = kv_cache_int8_quant_src_index<Geometry>(kv_head, d1, token);
        const float v0               = __bfloat162float(v[src0]);
        const float v1               = __bfloat162float(v[src1]);
        const float k_abs            = warp_max(fmaxf(fabsf(k0), fabsf(k1)), FullMask);
        const float v_abs            = warp_max(fmaxf(fabsf(v0), fabsf(v1)), FullMask);
        const auto k_quant           = kv_cache_int8_quant_params(k_abs);
        const auto v_quant           = kv_cache_int8_quant_params(v_abs);
        const std::int64_t code_base = kv_cache_int8_quant_code_index<Geometry>(
            physical_page, kv_head, group * kKVCacheInt8Group, page_off);
        cache_k[code_base + lane]      = kv_cache_int8_quant_code(k0, k_quant.inverse_scale);
        cache_k[code_base + lane + 32] = kv_cache_int8_quant_code(k1, k_quant.inverse_scale);
        cache_v[code_base + lane]      = kv_cache_int8_quant_code(v0, v_quant.inverse_scale);
        cache_v[code_base + lane + 32] = kv_cache_int8_quant_code(v1, v_quant.inverse_scale);
        if (lane == 0) {
            const std::int64_t scale_offset =
                kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, group, page_off);
            scale_k[scale_offset] = k_quant.scale;
            scale_v[scale_offset] = v_quant.scale;
        }
    }
}

inline constexpr int kKVCacheAppendPrefixHeadDim = 128;
inline constexpr int kKVCacheAppendPrefixHeads   = 8;
inline constexpr int kKVCacheAppendPrefixPage    = 64;

__device__ __forceinline__ void kv_cache_append_prefix_copy_cyclic_unit(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, int token,
    int unit_in_token, int slot, int padded_capacity) {
    constexpr int Bf16PerUnit  = 16;
    constexpr int UnitsPerHead = kKVCacheAppendPrefixHeadDim / Bf16PerUnit;
    const int kv_head          = unit_in_token / UnitsPerHead;
    const int d                = (unit_in_token - kv_head * UnitsPerHead) * Bf16PerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst = static_cast<std::int64_t>(d) +
                             static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                 (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);

    const int4 k0                               = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0                               = *reinterpret_cast<const int4*>(&v[src]);
    *reinterpret_cast<int4*>(&cache_k[dst])     = k0;
    *reinterpret_cast<int4*>(&cache_v[dst])     = v0;
    const int4 k1                               = *reinterpret_cast<const int4*>(&k[src + 8]);
    const int4 v1                               = *reinterpret_cast<const int4*>(&v[src + 8]);
    *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
    *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
}

__device__ __forceinline__ void kv_cache_append_prefix_copy_paged_unit(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, int token,
    int unit_in_token, int page_offset, int physical_page, int physical_pages) {
    constexpr int Bf16PerUnit  = 16;
    constexpr int UnitsPerHead = kKVCacheAppendPrefixHeadDim / Bf16PerUnit;
    const int kv_head          = unit_in_token / UnitsPerHead;
    const int d                = (unit_in_token - kv_head * UnitsPerHead) * Bf16PerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
            (page_offset + kKVCacheAppendPrefixPage * (physical_page + physical_pages * kv_head));

    const int4 k0                               = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0                               = *reinterpret_cast<const int4*>(&v[src]);
    *reinterpret_cast<int4*>(&cache_k[dst])     = k0;
    *reinterpret_cast<int4*>(&cache_v[dst])     = v0;
    const int4 k1                               = *reinterpret_cast<const int4*>(&k[src + 8]);
    const int4 v1                               = *reinterpret_cast<const int4*>(&v[src + 8]);
    *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
    *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
}

__global__ void kv_cache_append_prefix_cyclic_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ lanes, __nv_bfloat16* __restrict__ cache_k,
    __nv_bfloat16* __restrict__ cache_v, int min_count, int max_count, int width,
    int window, int padded_capacity) {
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * 8;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256);
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count < min_count || count > max_count) return;

    constexpr std::int64_t ElementsPerToken =
        kKVCacheAppendPrefixHeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset = ElementsPerToken * width * batch;
    const std::int64_t cache_offset =
        ElementsPerToken * static_cast<std::int64_t>(padded_capacity) * lanes[batch];
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    cache_k += cache_offset;
    cache_v += cache_offset;

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    if (token >= count) return;
    const int position = positions[token];
    const int slot     = position & (window - 1);
    kv_cache_append_prefix_copy_cyclic_unit(k, v, cache_k, cache_v, token, unit_in_token, slot,
                                            padded_capacity);
}

template <int Bits, bool Dual>
__device__ __forceinline__ void kv_cache_append_prefix_cyclic_oscar_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __nv_bfloat16* __restrict__ scale_k, __nv_bfloat16* __restrict__ scale_v, int token,
    __nv_bfloat16* __restrict__ protected_k, __nv_bfloat16* __restrict__ protected_v, int kv_head,
    std::uint8_t* __restrict__ q4_cache_k, std::uint8_t* __restrict__ q4_cache_v,
    __nv_bfloat16* __restrict__ q4_scale_k, __nv_bfloat16* __restrict__ q4_scale_v,
    __nv_bfloat16* __restrict__ q4_protected_k, __nv_bfloat16* __restrict__ q4_protected_v,
    int lane_id, int slot, int padded_capacity, int protected_slot, int protected_padded_capacity,
    int lane) {
    static_assert(Bits >= 2 && Bits <= 4);
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr int D             = kCyclicKVCacheOscarHeadDim;
    constexpr int Heads         = kCyclicKVCacheOscarKVHeads;
    constexpr int CodeExtent    = cyclic_oscar_code_extent<Bits>(D);
    const std::int64_t src_base = static_cast<std::int64_t>(D) *
                                  (kv_head + static_cast<std::int64_t>(Heads) * token);
    float k_values[4];
    float v_values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * 32;
        k_values[item] = __bfloat162float(k[src_base + d]);
        v_values[item] = __bfloat162float(v[src_base + d]);
    }
    normalized_hadamard_d128_inplace(k_values, lane);
    normalized_hadamard_d128_inplace(v_values, lane);
    const auto k_quant = cyclic_oscar_quant_params<Bits, false>(k_values, lane);
    const auto v_quant = cyclic_oscar_quant_params<Bits, true>(v_values, lane);
    std::uint8_t k_codes[4];
    std::uint8_t v_codes[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        k_codes[item] = cyclic_oscar_quantize<Bits>(k_values[item], k_quant);
        v_codes[item] = cyclic_oscar_quantize_value<Bits>(v_values[item], v_quant);
    }

    OscarAffineQuantParams<4, false> q4_k_quant{};
    OscarAffineQuantParams<4, true> q4_v_quant{};
    std::uint8_t q4_k_codes[4]{};
    std::uint8_t q4_v_codes[4]{};
    if constexpr (Dual) {
        q4_k_quant = cyclic_oscar_quant_params<4, false>(k_values, lane);
        q4_v_quant = cyclic_oscar_quant_params<4, true>(v_values, lane);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            q4_k_codes[item] = cyclic_oscar_quantize<4>(k_values[item], q4_k_quant);
            q4_v_codes[item] = cyclic_oscar_quantize_value<4>(v_values[item], q4_v_quant);
        }
    }

    if constexpr (Bits == 2) {
        // DFlash attention consumes Q2 in a lane-local layout: byte l contains
        // the four symbols for dimensions l, l+32, l+64 and l+96. This is a
        // storage-only transpose of the normal contiguous bitstream. Every
        // lane writes a distinct byte, removing the 32-lane packing shuffles
        // from the hot append path as well as from attention.
        const std::uint8_t packed_k = static_cast<std::uint8_t>(
            k_codes[0] | (k_codes[1] << 2) | (k_codes[2] << 4) | (k_codes[3] << 6));
        const std::uint8_t packed_v = static_cast<std::uint8_t>(
            v_codes[0] | (v_codes[1] << 2) | (v_codes[2] << 4) | (v_codes[3] << 6));
        cache_k[cyclic_oscar_code_index<Bits>(slot, kv_head, lane_id, padded_capacity, lane)] =
            packed_k;
        cache_v[cyclic_oscar_code_index<Bits>(slot, kv_head, lane_id, padded_capacity, lane)] =
            packed_v;
    } else {
        // Pack a contiguous symbol stream without byte races. Every lane executes the same
        // shuffles; lane zero owns the final byte stores. Q4 shadow/snapshot data stays in the
        // conventional OSCAR stream layout.
        for (int byte = 0; byte < CodeExtent; ++byte) {
            std::uint8_t packed_k = 0;
            std::uint8_t packed_v = 0;
            int bit_cursor         = 0;
            while (bit_cursor < 8) {
                const int global_bit = byte * 8 + bit_cursor;
                const int dimension  = global_bit / Bits;
                const int bit_offset = global_bit - dimension * Bits;
                const int take       = min(Bits - bit_offset, 8 - bit_cursor);
                const int source_lane = dimension & 31;
                const int source_item = dimension >> 5;
                const int k_code = __shfl_sync(FullMask, static_cast<int>(k_codes[source_item]),
                                               source_lane);
                const int v_code = __shfl_sync(FullMask, static_cast<int>(v_codes[source_item]),
                                               source_lane);
                const int mask = (1 << take) - 1;
                packed_k |= static_cast<std::uint8_t>(((k_code >> bit_offset) & mask) << bit_cursor);
                packed_v |= static_cast<std::uint8_t>(((v_code >> bit_offset) & mask) << bit_cursor);
                bit_cursor += take;
            }
            if (lane == 0) {
                cache_k[cyclic_oscar_code_index<Bits>(slot, kv_head, lane_id, padded_capacity,
                                                      byte)] = packed_k;
                cache_v[cyclic_oscar_code_index<Bits>(slot, kv_head, lane_id, padded_capacity,
                                                      byte)] = packed_v;
            }
        }
    }
    if constexpr (Dual) {
        constexpr int Q4CodeExtent = cyclic_oscar_code_extent<4>(D);
#pragma unroll
        for (int byte = 0; byte < Q4CodeExtent; ++byte) {
            std::uint8_t packed_k = 0;
            std::uint8_t packed_v = 0;
            int bit_cursor         = 0;
            while (bit_cursor < 8) {
                const int global_bit = byte * 8 + bit_cursor;
                const int dimension  = global_bit / 4;
                const int bit_offset = global_bit - dimension * 4;
                const int take       = min(4 - bit_offset, 8 - bit_cursor);
                const int source_lane = dimension & 31;
                const int source_item = dimension >> 5;
                const int k_code = __shfl_sync(FullMask, static_cast<int>(q4_k_codes[source_item]),
                                               source_lane);
                const int v_code = __shfl_sync(FullMask, static_cast<int>(q4_v_codes[source_item]),
                                               source_lane);
                const int mask = (1 << take) - 1;
                packed_k |= static_cast<std::uint8_t>(((k_code >> bit_offset) & mask) << bit_cursor);
                packed_v |= static_cast<std::uint8_t>(((v_code >> bit_offset) & mask) << bit_cursor);
                bit_cursor += take;
            }
            if (lane == 0) {
                q4_cache_k[cyclic_oscar_code_index<4>(slot, kv_head, lane_id, padded_capacity,
                                                      byte)] = packed_k;
                q4_cache_v[cyclic_oscar_code_index<4>(slot, kv_head, lane_id, padded_capacity,
                                                      byte)] = packed_v;
            }
        }
    }
    if (lane == 0) {
        const std::int64_t scale_offset =
            cyclic_oscar_scale_index(slot, kv_head, lane_id, padded_capacity, 0);
        scale_k[scale_offset]     = __float2bfloat16(k_quant.scale);
        scale_k[scale_offset + 1] = __float2bfloat16(k_quant.zero);
        scale_v[scale_offset]     = __float2bfloat16(v_quant.scale);
        scale_v[scale_offset + 1] = __float2bfloat16(v_quant.zero);
        if constexpr (Dual) {
            q4_scale_k[cyclic_oscar_scale_index(slot, kv_head, lane_id, padded_capacity, 0)] =
                __float2bfloat16(q4_k_quant.scale);
            q4_scale_k[cyclic_oscar_scale_index(slot, kv_head, lane_id, padded_capacity, 1)] =
                __float2bfloat16(q4_k_quant.zero);
            q4_scale_v[cyclic_oscar_scale_index(slot, kv_head, lane_id, padded_capacity, 0)] =
                __float2bfloat16(q4_v_quant.scale);
            q4_scale_v[cyclic_oscar_scale_index(slot, kv_head, lane_id, padded_capacity, 1)] =
                __float2bfloat16(q4_v_quant.zero);
        }
    }
    if (protected_k != nullptr && lane < 32) {
        const std::int64_t protected_base =
            static_cast<std::int64_t>(D) *
            (protected_slot + static_cast<std::int64_t>(protected_padded_capacity) *
                                  (kv_head + static_cast<std::int64_t>(Heads) * lane_id));
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            protected_k[protected_base + lane + item * 32] = __float2bfloat16(k_values[item]);
            protected_v[protected_base + lane + item * 32] = __float2bfloat16(v_values[item]);
        }
    }
    if constexpr (Dual) {
        if (q4_protected_k != nullptr && lane < 32) {
            const std::int64_t protected_base =
                static_cast<std::int64_t>(D) *
                (protected_slot + static_cast<std::int64_t>(protected_padded_capacity) *
                                      (kv_head + static_cast<std::int64_t>(Heads) * lane_id));
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                q4_protected_k[protected_base + lane + item * 32] = __float2bfloat16(k_values[item]);
                q4_protected_v[protected_base + lane + item * 32] = __float2bfloat16(v_values[item]);
            }
        }
    }
}

template <int Bits, bool Dual>
__global__ void kv_cache_append_prefix_cyclic_oscar_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ lanes, std::uint8_t* __restrict__ cache_k,
    std::uint8_t* __restrict__ cache_v, __nv_bfloat16* __restrict__ scale_k,
    __nv_bfloat16* __restrict__ scale_v, __nv_bfloat16* __restrict__ protected_k,
    __nv_bfloat16* __restrict__ protected_v, std::uint8_t* __restrict__ q4_cache_k,
    std::uint8_t* __restrict__ q4_cache_v, __nv_bfloat16* __restrict__ q4_scale_k,
    __nv_bfloat16* __restrict__ q4_scale_v, __nv_bfloat16* __restrict__ q4_protected_k,
    __nv_bfloat16* __restrict__ q4_protected_v, int min_count, int max_count, int width,
    int window, int padded_capacity, int protected_capacity, int protected_anchor_capacity,
    int protected_padded_capacity) {
    constexpr int Warps = 8;
    const int batch       = static_cast<int>(blockIdx.y);
    const int count       = counts[batch];
    if (count < min_count || count > max_count) return;
    const int token       = static_cast<int>(blockIdx.x);
    const int kv_head     = static_cast<int>(threadIdx.x) >> 5;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    if (token >= count || kv_head >= kCyclicKVCacheOscarKVHeads ||
        static_cast<int>(blockDim.x) != Warps * 32) {
        return;
    }
    constexpr std::int64_t ElementsPerToken =
        kCyclicKVCacheOscarHeadDim * kCyclicKVCacheOscarKVHeads;
    const std::int64_t batch_offset = ElementsPerToken * static_cast<std::int64_t>(width) * batch;
    const int position = positions[static_cast<std::int64_t>(width) * batch + token];
    const int slot     = position & (window - 1);
    const bool anchor_protected =
        protected_anchor_capacity != 0 && position < protected_anchor_capacity;
    // Only the suffix that is recent at the end of this append owns the narrow BF16 sidecar.
    // Writing every token into a smaller recent ring would make a multi-token prefix append race
    // with itself when positions wrap the sidecar. The cyclic append contract guarantees a
    // sequential prefix, so the last input position is the append frontier.
    const int append_frontier = positions[static_cast<std::int64_t>(width) * batch + count - 1];
    const int recent_begin =
        protected_capacity == 0
            ? append_frontier + 1
            : max(0, append_frontier - static_cast<int>(protected_capacity) + 1);
    const bool recent_protected = protected_capacity != 0 && position >= recent_begin;
    const int protected_slot =
        anchor_protected
            ? position
            : protected_anchor_capacity +
                  (recent_protected ? position & (protected_capacity - 1) : 0);
    __nv_bfloat16* protected_k_for_token =
        (anchor_protected || recent_protected) ? protected_k : nullptr;
    __nv_bfloat16* protected_v_for_token =
        (anchor_protected || recent_protected) ? protected_v : nullptr;
    kv_cache_append_prefix_cyclic_oscar_row<Bits, Dual>(
        k + batch_offset, v + batch_offset, cache_k, cache_v, scale_k, scale_v, token,
        protected_k_for_token, protected_v_for_token, kv_head, q4_cache_k, q4_cache_v, q4_scale_k,
        q4_scale_v,
        (q4_protected_k != nullptr && (anchor_protected || recent_protected))
            ? q4_protected_k
            : nullptr,
        (q4_protected_v != nullptr && (anchor_protected || recent_protected))
            ? q4_protected_v
            : nullptr,
        lanes[batch], slot, padded_capacity, protected_slot, protected_padded_capacity, lane);
}

template <int Bits>
__device__ __forceinline__ void kv_cache_append_prefix_cyclic_lowbit_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, int token,
    __nv_bfloat16* __restrict__ protected_k, __nv_bfloat16* __restrict__ protected_v,
    int kv_head, int lane_id, int slot, int padded_capacity, int protected_slot,
    int protected_padded_capacity, int lane) {
    static_assert(Bits == 2 || Bits == 3);
    constexpr unsigned FullMask = 0xffffffffU;
    constexpr int ValuesPerLane = kCyclicKVLowBitQuantGroup / 2;
    const int logical_lane = lane & 15;
    const int group = logical_lane >> 1;
    const int group_lane = logical_lane & 1;
    const int d_base = group * kCyclicKVLowBitQuantGroup + group_lane * ValuesPerLane;
    const std::int64_t src_base = static_cast<std::int64_t>(kCyclicKVCacheNvfp4HeadDim) *
                                  (kv_head + kCyclicKVCacheNvfp4KVHeads * token);
    float k_values[ValuesPerLane];
    float v_values[ValuesPerLane];
    float k_absmax = 0.0F;
    float v_absmax = 0.0F;
#pragma unroll
    for (int i = 0; i < ValuesPerLane; ++i) {
        const int d = d_base + i;
        k_values[i] = __bfloat162float(k[src_base + d]);
        v_values[i] = __bfloat162float(v[src_base + d]);
        k_absmax = fmaxf(k_absmax, fabsf(k_values[i]));
        v_absmax = fmaxf(v_absmax, fabsf(v_values[i]));
    }
    k_absmax = fmaxf(k_absmax, __shfl_xor_sync(FullMask, k_absmax, 1));
    v_absmax = fmaxf(v_absmax, __shfl_xor_sync(FullMask, v_absmax, 1));
    const auto k_quant = cyclic_kv_lowbit_quant_params<Bits>(k_absmax);
    const auto v_quant = cyclic_kv_lowbit_quant_params<Bits>(v_absmax);
    std::uint8_t k_codes[ValuesPerLane];
    std::uint8_t v_codes[ValuesPerLane];
#pragma unroll
    for (int i = 0; i < ValuesPerLane; ++i) {
        k_codes[i] = cyclic_kv_lowbit_quantize<Bits>(k_values[i], k_quant.inverse_scale);
        v_codes[i] = cyclic_kv_lowbit_quantize<Bits>(v_values[i], v_quant.inverse_scale);
    }

    // One writer lane owns each 16-value group. It obtains the other half of the group from its
    // paired lane, then packs complete bytes without read/modify/write races between lanes.
    if (lane < 16 && group_lane == 0) {
        constexpr int GroupBytes = cyclic_kv_lowbit_group_bytes<Bits>();
        constexpr int ValuesPerByte = 8 / Bits;
        for (int byte = 0; byte < GroupBytes; ++byte) {
            std::uint8_t packed_k = 0;
            std::uint8_t packed_v = 0;
#pragma unroll
            for (int item = 0; item < ValuesPerByte; ++item) {
                const int value_index = byte * ValuesPerByte + item;
                const int local_index = value_index < ValuesPerLane
                                            ? value_index
                                            : value_index - ValuesPerLane;
                const std::uint8_t k_code = value_index < ValuesPerLane
                                                 ? k_codes[local_index]
                                                 : __shfl_sync(FullMask, k_codes[local_index], lane + 1);
                const std::uint8_t v_code = value_index < ValuesPerLane
                                                 ? v_codes[local_index]
                                                 : __shfl_sync(FullMask, v_codes[local_index], lane + 1);
                packed_k |= static_cast<std::uint8_t>(k_code << (item * Bits));
                packed_v |= static_cast<std::uint8_t>(v_code << (item * Bits));
            }
            const int code_byte = group * GroupBytes + byte;
            cache_k[cyclic_kv_lowbit_code_index<Bits>(slot, kv_head, lane_id, padded_capacity,
                                                      code_byte)] = packed_k;
            cache_v[cyclic_kv_lowbit_code_index<Bits>(slot, kv_head, lane_id, padded_capacity,
                                                      code_byte)] = packed_v;
        }
        scale_k[cyclic_kv_lowbit_scale_index<Bits>(slot, kv_head, lane_id, padded_capacity, group)] =
            k_quant.scale;
        scale_v[cyclic_kv_lowbit_scale_index<Bits>(slot, kv_head, lane_id, padded_capacity, group)] =
            v_quant.scale;
    }
    if (protected_k != nullptr && lane < 16) {
        const std::int64_t protected_base =
            static_cast<std::int64_t>(kCyclicKVCacheNvfp4HeadDim) *
            (protected_slot + static_cast<std::int64_t>(protected_padded_capacity) *
                                  (kv_head + kCyclicKVCacheNvfp4KVHeads * lane_id));
#pragma unroll
        for (int i = 0; i < ValuesPerLane; ++i) {
            protected_k[protected_base + d_base + i] = k[src_base + d_base + i];
            protected_v[protected_base + d_base + i] = v[src_base + d_base + i];
        }
    }
}

template <int Bits>
__global__ void kv_cache_append_prefix_cyclic_lowbit_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ lanes, std::uint8_t* __restrict__ cache_k,
    std::uint8_t* __restrict__ cache_v, std::uint8_t* __restrict__ scale_k,
    std::uint8_t* __restrict__ scale_v, __nv_bfloat16* __restrict__ protected_k,
    __nv_bfloat16* __restrict__ protected_v, int min_count, int max_count, int width, int window,
    int padded_capacity, int protected_capacity, int protected_anchor_capacity,
    int protected_padded_capacity) {
    constexpr int Warps = 8;
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count < min_count || count > max_count) return;
    const int token = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (token >= count || kv_head >= kCyclicKVCacheNvfp4KVHeads ||
        static_cast<int>(blockDim.x) != Warps * 32) {
        return;
    }
    constexpr std::int64_t ElementsPerToken =
        kCyclicKVCacheNvfp4HeadDim * kCyclicKVCacheNvfp4KVHeads;
    const std::int64_t batch_offset = ElementsPerToken * static_cast<std::int64_t>(width) * batch;
    const int position = positions[static_cast<std::int64_t>(width) * batch + token];
    const int slot = position & (window - 1);
    const bool anchor_protected = protected_anchor_capacity != 0 && position < protected_anchor_capacity;
    const int append_frontier = positions[static_cast<std::int64_t>(width) * batch + count - 1];
    const int recent_begin =
        protected_capacity == 0
            ? append_frontier + 1
            : max(0, append_frontier - static_cast<int>(protected_capacity) + 1);
    const bool recent_protected = protected_capacity != 0 && position >= recent_begin;
    const int protected_slot = anchor_protected
                                   ? position
                                   : protected_anchor_capacity +
                                         (recent_protected ? position & (protected_capacity - 1) : 0);
    __nv_bfloat16* protected_k_for_token =
        (anchor_protected || recent_protected) ? protected_k : nullptr;
    __nv_bfloat16* protected_v_for_token =
        (anchor_protected || recent_protected) ? protected_v : nullptr;
    kv_cache_append_prefix_cyclic_lowbit_row<Bits>(
        k + batch_offset, v + batch_offset, cache_k, cache_v, scale_k, scale_v, token,
        protected_k_for_token, protected_v_for_token, kv_head, lanes[batch], slot, padded_capacity,
        protected_slot, protected_padded_capacity, lane);
}

__device__ __forceinline__ void kv_cache_append_prefix_cyclic_nvfp4_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, int token,
    __nv_bfloat16* __restrict__ protected_k, __nv_bfloat16* __restrict__ protected_v,
    int kv_head, int lane_id, int slot, int padded_capacity, int protected_slot,
    int protected_padded_capacity, int lane) {
    constexpr unsigned FullMask   = 0xffffffffU;
    constexpr int ValuesPerLane   = kCyclicKVCacheNvfp4Group / 2;
    // A warp has 32 lanes while a 128-wide head has eight 16-value groups. Reuse the second
    // half of the warp for the first half; the duplicate stores are intentional and keep the
    // pairwise shuffle mask warp-uniform.
    const int group               = (lane & 15) >> 1;
    const int group_lane          = lane & 1;
    const int d_base              = group * kCyclicKVCacheNvfp4Group + group_lane * ValuesPerLane;
    const std::int64_t src_base   = static_cast<std::int64_t>(kCyclicKVCacheNvfp4HeadDim) *
                                  (kv_head + kCyclicKVCacheNvfp4KVHeads * token);
    float k_values[ValuesPerLane];
    float v_values[ValuesPerLane];
    float k_absmax = 0.0F;
    float v_absmax = 0.0F;
#pragma unroll
    for (int i = 0; i < ValuesPerLane; ++i) {
        const int d = d_base + i;
        k_values[i] = __bfloat162float(k[src_base + d]);
        v_values[i] = __bfloat162float(v[src_base + d]);
        k_absmax    = fmaxf(k_absmax, fabsf(k_values[i]));
        v_absmax    = fmaxf(v_absmax, fabsf(v_values[i]));
    }
    k_absmax = fmaxf(k_absmax, __shfl_xor_sync(FullMask, k_absmax, 1));
    v_absmax = fmaxf(v_absmax, __shfl_xor_sync(FullMask, v_absmax, 1));
    const KVCacheNvfp4QuantParams k_quant = kv_cache_nvfp4_quant_params(k_absmax);
    const KVCacheNvfp4QuantParams v_quant = kv_cache_nvfp4_quant_params(v_absmax);
#pragma unroll
    for (int pair = 0; pair < ValuesPerLane / 2; ++pair) {
        const int d = d_base + 2 * pair;
        cache_k[cyclic_nvfp4_code_index(slot, kv_head, lane_id, padded_capacity, d >> 1)] =
            kv_cache_nvfp4_quantize_pair(k_values[2 * pair], k_values[2 * pair + 1],
                                         k_quant.inverse_scale);
        cache_v[cyclic_nvfp4_code_index(slot, kv_head, lane_id, padded_capacity, d >> 1)] =
            kv_cache_nvfp4_quantize_pair(v_values[2 * pair], v_values[2 * pair + 1],
                                         v_quant.inverse_scale);
    }
    if (group_lane == 0) {
        scale_k[cyclic_nvfp4_scale_index(slot, kv_head, lane_id, padded_capacity, group)] =
            k_quant.scale;
        scale_v[cyclic_nvfp4_scale_index(slot, kv_head, lane_id, padded_capacity, group)] =
            v_quant.scale;
    }
    if (protected_k != nullptr && lane < 16) {
        const std::int64_t protected_base =
            static_cast<std::int64_t>(kCyclicKVCacheNvfp4HeadDim) *
            (protected_slot + static_cast<std::int64_t>(protected_padded_capacity) *
                                  (kv_head + kCyclicKVCacheNvfp4KVHeads * lane_id));
        // The first 16 lanes cover the complete 128-wide head once.  Keep the sidecar BF16 so
        // recent/high-sensitivity keys can be consumed without quantization error.
#pragma unroll
        for (int i = 0; i < ValuesPerLane; ++i) {
            protected_k[protected_base + d_base + i] = k[src_base + d_base + i];
            protected_v[protected_base + d_base + i] = v[src_base + d_base + i];
        }
    }
}

__global__ void kv_cache_append_prefix_cyclic_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ lanes, std::uint8_t* __restrict__ cache_k,
    std::uint8_t* __restrict__ cache_v, std::uint8_t* __restrict__ scale_k,
    std::uint8_t* __restrict__ scale_v, __nv_bfloat16* __restrict__ protected_k,
    __nv_bfloat16* __restrict__ protected_v, int min_count, int max_count, int width, int window,
    int padded_capacity, int protected_capacity, int protected_anchor_capacity,
    int protected_padded_capacity) {
    constexpr int Warps = 8;
    const int batch      = static_cast<int>(blockIdx.y);
    const int count      = counts[batch];
    if (count < min_count || count > max_count) return;
    const int token = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (token >= count || kv_head >= kCyclicKVCacheNvfp4KVHeads ||
        static_cast<int>(blockDim.x) != Warps * 32) {
        return;
    }
    constexpr std::int64_t ElementsPerToken =
        kCyclicKVCacheNvfp4HeadDim * kCyclicKVCacheNvfp4KVHeads;
    const std::int64_t batch_offset = ElementsPerToken * static_cast<std::int64_t>(width) * batch;
    const int position = positions[static_cast<std::int64_t>(width) * batch + token];
    const int slot     = position & (window - 1);
    const bool anchor_protected =
        protected_anchor_capacity != 0 && position < protected_anchor_capacity;
    const int append_frontier = positions[static_cast<std::int64_t>(width) * batch + count - 1];
    const int recent_begin =
        protected_capacity == 0
            ? append_frontier + 1
            : max(0, append_frontier - static_cast<int>(protected_capacity) + 1);
    const bool recent_protected = protected_capacity != 0 && position >= recent_begin;
    const int protected_slot =
        anchor_protected
            ? position
            : protected_anchor_capacity +
                  (recent_protected ? position & (protected_capacity - 1) : 0);
    __nv_bfloat16* protected_k_for_token =
        (anchor_protected || recent_protected) ? protected_k : nullptr;
    __nv_bfloat16* protected_v_for_token =
        (anchor_protected || recent_protected) ? protected_v : nullptr;
    kv_cache_append_prefix_cyclic_nvfp4_row(
        k + batch_offset, v + batch_offset, cache_k, cache_v, scale_k, scale_v, token,
        protected_k_for_token, protected_v_for_token, kv_head, lanes[batch], slot, padded_capacity,
        protected_slot, protected_padded_capacity, lane);
}

__global__ void kv_cache_append_prefix_paged_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ table_rows, __nv_bfloat16* __restrict__ cache_k,
    __nv_bfloat16* __restrict__ cache_v, const std::int32_t* __restrict__ block_tables,
    int physical_pages, int logical_pages, int min_count, int max_count, int width) {
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * 8;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256);
    const int batch = static_cast<int>(blockIdx.y);
    const int count = counts[batch];
    if (count < min_count || count > max_count) return;

    constexpr std::int64_t ElementsPerToken =
        kKVCacheAppendPrefixHeadDim * kKVCacheAppendPrefixHeads;
    const std::int64_t input_offset = ElementsPerToken * width * batch;
    k += input_offset;
    v += input_offset;
    positions += static_cast<std::int64_t>(width) * batch;
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(logical_pages) * table_rows[batch];

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int lane          = local & 31;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    int position            = 0;
    int physical_page       = 0;
    if (lane == 0 && token < count) {
        position      = positions[token];
        physical_page = block_table[position >> 6];
    }
    position      = __shfl_sync(0xffffffffu, position, 0);
    physical_page = __shfl_sync(0xffffffffu, physical_page, 0);
    if (token < count) {
        kv_cache_append_prefix_copy_paged_unit(k, v, cache_k, cache_v, token, unit_in_token,
                                               position & (kKVCacheAppendPrefixPage - 1),
                                               physical_page, physical_pages);
    }
}

} // namespace ninfer::ops
