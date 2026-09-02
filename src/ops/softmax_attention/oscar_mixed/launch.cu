#include "ops/softmax_attention/oscar_mixed/launch.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kGqa = 6;
constexpr int kHeadDim = 256;
constexpr int kGroups = 2;
constexpr int kGroupSize = 128;
constexpr int kCodeBytes = 64;
constexpr int kPrefixTokens = 64;
constexpr int kRecentTokens = 256;
constexpr int kInt2Levels = 3;
constexpr int kInt2MetadataItems = 4;
constexpr int kThreads = 256;
constexpr int kTokenTile = 256;
// The fused reader uses a smaller internal query/tile shape than the D4.5 launch tile. Four
// queries and 32 K/V rows keep useful GQA reuse while fitting Q, one K/V slab, and score tiles in
// the actual SM120a per-block dynamic shared-memory budget. The K slab is reused for V after QK.
constexpr int kFusedTokenTile = 32;
constexpr int kFusedDecodeTokenTile = 64;
constexpr int kFusedQueryTile = 4;
static_assert(kGroupSize == 128 && kGroups == 2 && kCodeBytes == 64,
              "OscarInt2G128 mixed layout changed");

__device__ __forceinline__ float decode_int2(const std::uint8_t* packed,
                                              const float* metadata, int dimension) {
    const int byte = dimension & 63;
    const int shift = (dimension >> 6) << 1;
    const std::uint8_t symbol = static_cast<std::uint8_t>((packed[byte] >> shift) & 3U);
    const int group = dimension >> 7;
    return (static_cast<float>(symbol) - metadata[group * 2 + 1]) * metadata[group * 2];
}

__device__ __forceinline__ float decode_bf16(std::uint16_t bits) {
    return __int_as_float(static_cast<int>(static_cast<std::uint32_t>(bits) << 16U));
}

__device__ __forceinline__ float load_key(
    int logical, int kv_head, int dimension, const std::uint16_t* prefix_k, int prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    int historical_tokens, const std::uint16_t* recent_k, int recent_tokens,
    int recent_ring_head) {
    if (logical < prefix_tokens) {
        const std::size_t row = (static_cast<std::size_t>(logical) * kKVHeads + kv_head) * kHeadDim;
        return decode_bf16(prefix_k[row + dimension]);
    }
    const int historical_logical = logical - prefix_tokens;
    if (historical_logical < historical_tokens) {
        const std::size_t row = static_cast<std::size_t>(historical_logical) * kKVHeads + kv_head;
        return decode_int2(historical_k_packed + row * kCodeBytes,
                           historical_k_metadata + row * (kGroups * 2), dimension);
    }
    const int recent_logical = historical_logical - historical_tokens;
    if (recent_logical < recent_tokens) {
        const int physical_recent = (recent_ring_head + recent_logical) & (kTokenTile - 1);
        const std::size_t row =
            (static_cast<std::size_t>(physical_recent) * kKVHeads + kv_head) * kHeadDim;
        return decode_bf16(recent_k[row + dimension]);
    }
    return 0.0F;
}

__device__ __forceinline__ float load_value(
    int logical, int kv_head, int dimension, const std::uint16_t* prefix_v, int prefix_tokens,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    int historical_tokens, const std::uint16_t* recent_v, int recent_tokens,
    int recent_ring_head) {
    if (logical < prefix_tokens) {
        const std::size_t row = (static_cast<std::size_t>(logical) * kKVHeads + kv_head) * kHeadDim;
        return decode_bf16(prefix_v[row + dimension]);
    }
    const int historical_logical = logical - prefix_tokens;
    if (historical_logical < historical_tokens) {
        const std::size_t row = static_cast<std::size_t>(historical_logical) * kKVHeads + kv_head;
        return decode_int2(historical_v_packed + row * kCodeBytes,
                           historical_v_metadata + row * (kGroups * 2), dimension);
    }
    const int recent_logical = historical_logical - historical_tokens;
    if (recent_logical < recent_tokens) {
        const int physical_recent = (recent_ring_head + recent_logical) & (kTokenTile - 1);
        const std::size_t row =
            (static_cast<std::size_t>(physical_recent) * kKVHeads + kv_head) * kHeadDim;
        return decode_bf16(recent_v[row + dimension]);
    }
    return 0.0F;
}

// One block owns one KV head and one logical 256-token tile. K is decoded at the
// point of use and each decoded scalar feeds all six Q heads sharing the KV head.
__global__ void oscar_mixed_score_kernel(
    const float* __restrict__ q_rotated, const std::uint16_t* __restrict__ prefix_k,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_k_packed,
    const float* __restrict__ historical_k_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_k, int recent_tokens, int recent_ring_head,
    int total_tokens, float attention_scale, float* __restrict__ scores) {
    __shared__ float q_shared[kGqa * kHeadDim];
    const int kv_head = static_cast<int>(blockIdx.x);
    const int logical = static_cast<int>(blockIdx.y) * kTokenTile + static_cast<int>(threadIdx.x);
    for (int index = static_cast<int>(threadIdx.x); index < kGqa * kHeadDim; index += kThreads) {
        const int query_head = kv_head * kGqa + index / kHeadDim;
        q_shared[index] = q_rotated[query_head * kHeadDim + index % kHeadDim];
    }
    __syncthreads();
    if (logical >= total_tokens) return;

    float accum[kGqa] = {};
    for (int dimension = 0; dimension < kHeadDim; ++dimension) {
        const float key = load_key(logical, kv_head, dimension, prefix_k, prefix_tokens,
                                   historical_k_packed, historical_k_metadata, historical_tokens,
                                   recent_k, recent_tokens, recent_ring_head);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            accum[query_group] += q_shared[query_group * kHeadDim + dimension] * key;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        scores[query_head * total_tokens + logical] = accum[query_group] * attention_scale;
    }
}

__device__ __forceinline__ float reduce_max(float value) {
    __shared__ float values[kThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return values[0];
}

__device__ __forceinline__ float reduce_sum(float value) {
    __shared__ float values[kThreads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    return values[0];
}

__global__ void oscar_mixed_softmax_kernel(const float* __restrict__ scores, int total_tokens,
                                           float* __restrict__ softmax) {
    const int query_head = static_cast<int>(blockIdx.x);
    const float* row = scores + query_head * total_tokens;
    float maximum = -3.402823466e+38F;
    for (int logical = static_cast<int>(threadIdx.x); logical < total_tokens; logical += kThreads) {
        maximum = fmaxf(maximum, row[logical]);
    }
    maximum = reduce_max(maximum);
    float sum = 0.0F;
    for (int logical = static_cast<int>(threadIdx.x); logical < total_tokens; logical += kThreads) {
        const float probability = expf(row[logical] - maximum);
        softmax[query_head * total_tokens + logical] = probability;
        sum += probability;
    }
    sum = reduce_sum(sum);
    for (int logical = static_cast<int>(threadIdx.x); logical < total_tokens; logical += kThreads) {
        softmax[query_head * total_tokens + logical] /= sum;
    }
}

// One block owns one KV head and one output dimension per thread. V is decoded
// directly into six FP32 AV accumulators; there is no decoded-V allocation.
__global__ void oscar_mixed_av_kernel(
    const float* __restrict__ softmax, const std::uint16_t* __restrict__ prefix_v,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_v_packed,
    const float* __restrict__ historical_v_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_v, int recent_tokens, int recent_ring_head,
    int total_tokens, float* __restrict__ output) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    if (dimension >= kHeadDim) return;

    float accum[kGqa] = {};
    for (int logical = 0; logical < total_tokens; ++logical) {
        const float value = load_value(logical, kv_head, dimension, prefix_v, prefix_tokens,
                                       historical_v_packed, historical_v_metadata, historical_tokens,
                                       recent_v, recent_tokens, recent_ring_head);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            const int query_head = kv_head * kGqa + query_group;
            accum[query_group] += softmax[query_head * total_tokens + logical] * value;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        output[query_head * kHeadDim + dimension] = accum[query_group];
    }
}

// The batched prefill kernels intentionally retain the D4.4 scalar-query operation order.  A
// query is an independent grid-z slice, so batching changes launch/setup overhead without
// changing the mixed-cache decode, softmax, GQA, or AV arithmetic for any query column.
__global__ void oscar_mixed_score_batch_kernel(
    const float* __restrict__ q_rotated, const std::uint16_t* __restrict__ prefix_k,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_k_packed,
    const float* __restrict__ historical_k_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_k, int recent_tokens, int recent_ring_head,
    int total_tokens, int query_start, int workspace_stride, float attention_scale,
    float* __restrict__ scores) {
    __shared__ float q_shared[kGqa * kHeadDim];
    const int kv_head = static_cast<int>(blockIdx.x);
    const int query_index = static_cast<int>(blockIdx.z);
    const int query_token = query_start + query_index;
    const int logical = static_cast<int>(blockIdx.y) * kTokenTile + static_cast<int>(threadIdx.x);
    for (int index = static_cast<int>(threadIdx.x); index < kGqa * kHeadDim; index += kThreads) {
        const int query_head = kv_head * kGqa + index / kHeadDim;
        q_shared[index] = q_rotated[(static_cast<std::size_t>(query_index) * kQHeads +
                                     query_head) * kHeadDim + index % kHeadDim];
    }
    __syncthreads();
    if (logical >= total_tokens || logical > query_token) return;

    float accum[kGqa] = {};
    for (int dimension = 0; dimension < kHeadDim; ++dimension) {
        const float key = load_key(logical, kv_head, dimension, prefix_k, prefix_tokens,
                                   historical_k_packed, historical_k_metadata, historical_tokens,
                                   recent_k, recent_tokens, recent_ring_head);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            accum[query_group] += q_shared[query_group * kHeadDim + dimension] * key;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        const std::size_t row = (static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                                static_cast<std::size_t>(workspace_stride);
        scores[row + logical] = accum[query_group] * attention_scale;
    }
}

__global__ void oscar_mixed_softmax_batch_kernel(const float* __restrict__ scores,
                                                 int total_tokens, int query_start,
                                                 int query_count, int workspace_stride,
                                                 float* __restrict__ softmax) {
    const int flat = static_cast<int>(blockIdx.x);
    const int query_index = flat / kQHeads;
    const int query_head = flat % kQHeads;
    if (query_index >= query_count) return;
    const int visible_tokens = min(total_tokens, query_start + query_index + 1);
    const float* row = scores +
                       (static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                           static_cast<std::size_t>(workspace_stride);
    float maximum = -3.402823466e+38F;
    for (int logical = static_cast<int>(threadIdx.x); logical < visible_tokens;
         logical += kThreads) {
        maximum = fmaxf(maximum, row[logical]);
    }
    maximum = reduce_max(maximum);
    float sum = 0.0F;
    for (int logical = static_cast<int>(threadIdx.x); logical < visible_tokens;
         logical += kThreads) {
        const float probability = expf(row[logical] - maximum);
        softmax[(static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                    static_cast<std::size_t>(workspace_stride) + logical] = probability;
        sum += probability;
    }
    sum = reduce_sum(sum);
    for (int logical = static_cast<int>(threadIdx.x); logical < visible_tokens;
         logical += kThreads) {
        softmax[(static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                    static_cast<std::size_t>(workspace_stride) + logical] /= sum;
    }
}

__global__ void oscar_mixed_av_batch_kernel(
    const float* __restrict__ softmax, const std::uint16_t* __restrict__ prefix_v,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_v_packed,
    const float* __restrict__ historical_v_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_v, int recent_tokens, int recent_ring_head,
    int total_tokens, int query_start, int query_count, int workspace_stride,
    float* __restrict__ output) {
    const int flat = static_cast<int>(blockIdx.x);
    const int query_index = flat / kKVHeads;
    const int kv_head = flat % kKVHeads;
    const int dimension = static_cast<int>(threadIdx.x);
    if (query_index >= query_count || dimension >= kHeadDim) return;
    const int visible_tokens = min(total_tokens, query_start + query_index + 1);

    float accum[kGqa] = {};
    for (int logical = 0; logical < visible_tokens; ++logical) {
        const float value = load_value(logical, kv_head, dimension, prefix_v, prefix_tokens,
                                       historical_v_packed, historical_v_metadata, historical_tokens,
                                       recent_v, recent_tokens, recent_ring_head);
        for (int query_group = 0; query_group < kGqa; ++query_group) {
            const int query_head = kv_head * kGqa + query_group;
            const std::size_t probability =
                (static_cast<std::size_t>(query_index) * kQHeads + query_head) *
                    static_cast<std::size_t>(workspace_stride) + logical;
            accum[query_group] += softmax[probability] * value;
        }
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        output[(static_cast<std::size_t>(query_index) * kQHeads + query_head) * kHeadDim +
               dimension] = accum[query_group];
    }
}

// D4.6 fused resident mixed attention.  A block owns one KV head and four neighboring query
// columns.  The block walks the logical sequence in 32-token tiles, decodes each resident K/V
// tile exactly once into a shared-memory slab, computes all six GQA score rows for the query tile,
// reuses the slab for V, and immediately folds the tile into a per-query online-softmax/AV
// accumulator. No score, softmax, or decoded K/V history is written to global memory.
__global__ void oscar_mixed_fused_batch_kernel(
    const float* __restrict__ q_rotated, const std::uint16_t* __restrict__ prefix_k,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_k_packed,
    const float* __restrict__ historical_k_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_k, int recent_tokens, int recent_ring_head,
    const std::uint16_t* __restrict__ prefix_v,
    const std::uint8_t* __restrict__ historical_v_packed,
    const float* __restrict__ historical_v_metadata, const std::uint16_t* __restrict__ recent_v,
    int total_tokens, int query_start, int query_count, float attention_scale,
    float* __restrict__ output) {
    extern __shared__ float shared[];
    constexpr int kQueryValues = kFusedQueryTile * kGqa * kHeadDim;
    constexpr int kTileValues = kFusedTokenTile * kHeadDim;
    constexpr int kScoreValues = kFusedQueryTile * kGqa * kFusedTokenTile;
    constexpr int kStateValues = kFusedQueryTile * kGqa;
    float* q_shared = shared;
    float* k_shared = q_shared + kQueryValues;
    float* score_shared = k_shared + kTileValues;
    float* tile_max = score_shared + kScoreValues;
    float* tile_sum = tile_max + kStateValues;
    float* m_state = tile_sum + kStateValues;
    float* l_state = m_state + kStateValues;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int query_base = static_cast<int>(blockIdx.y) * kFusedQueryTile;
    const int active_queries = min(kFusedQueryTile, query_count - query_base);
    const int active_groups = active_queries * kGqa;
    if (active_queries <= 0) return;
    float accum[kStateValues] = {};

    for (int index = static_cast<int>(threadIdx.x); index < kQueryValues; index += kThreads) {
        const int query_index = index / (kGqa * kHeadDim);
        const int group = (index / kHeadDim) % kGqa;
        const int dimension = index % kHeadDim;
        if (query_index < active_queries) {
            const int query_head = kv_head * kGqa + group;
            q_shared[index] = q_rotated[
                (static_cast<std::size_t>(query_base + query_index) * kQHeads + query_head) *
                    kHeadDim + dimension];
        } else {
            q_shared[index] = 0.0F;
        }
    }
    for (int group = static_cast<int>(threadIdx.x); group < kStateValues; group += kThreads) {
        m_state[group] = -3.402823466e+38F;
        l_state[group] = 0.0F;
    }
    __syncthreads();

    const int maximum_visible = min(total_tokens, query_start + query_base + active_queries);
    for (int tile_start = 0; tile_start < maximum_visible; tile_start += kFusedTokenTile) {
        const int tile_tokens = min(kFusedTokenTile, maximum_visible - tile_start);
        for (int index = static_cast<int>(threadIdx.x); index < kTileValues; index += kThreads) {
            const int tile_token = index / kHeadDim;
            if (tile_token < tile_tokens) {
                const int logical = tile_start + tile_token;
                const int dimension = index % kHeadDim;
                k_shared[index] = load_key(logical, kv_head, dimension, prefix_k, prefix_tokens,
                                           historical_k_packed, historical_k_metadata,
                                           historical_tokens, recent_k, recent_tokens,
                                           recent_ring_head);
            }
        }
        __syncthreads();

        // One warp owns one query in the tile and one lane owns one K row. This gives each
        // thread the same six-GQA accumulator pattern as the qualified scalar reader, so a K
        // scalar is loaded once per dimension and reused across all six associated Q heads.
        for (int slot = static_cast<int>(threadIdx.x);
             slot < active_queries * kFusedTokenTile; slot += kThreads) {
            const int query_index = slot / kFusedTokenTile;
            const int tile_token = slot % kFusedTokenTile;
            if (tile_token < tile_tokens) {
                const float* key = k_shared + tile_token * kHeadDim;
                float accum[kGqa] = {};
                for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                    const float key_value = key[dimension];
                    #pragma unroll
                    for (int query_group = 0; query_group < kGqa; ++query_group) {
                        const float* q = q_shared +
                                         (query_index * kGqa + query_group) * kHeadDim;
                        accum[query_group] += q[dimension] * key_value;
                    }
                }
                #pragma unroll
                for (int query_group = 0; query_group < kGqa; ++query_group) {
                    const int group_index = query_index * kGqa + query_group;
                    score_shared[group_index * kFusedTokenTile + tile_token] =
                        accum[query_group] * attention_scale;
                }
            }
        }
        __syncthreads();

        // QK has consumed the K slab. Reuse that same storage for V; the load can overlap the
        // small per-query tile-statistics loop because the two regions are independent.
        for (int index = static_cast<int>(threadIdx.x); index < kTileValues; index += kThreads) {
            const int tile_token = index / kHeadDim;
            if (tile_token < tile_tokens) {
                const int logical = tile_start + tile_token;
                const int dimension = index % kHeadDim;
                k_shared[index] = load_value(logical, kv_head, dimension, prefix_v, prefix_tokens,
                                              historical_v_packed, historical_v_metadata,
                                              historical_tokens, recent_v, recent_tokens,
                                              recent_ring_head);
            }
        }

        // Tile statistics are small (at most 24 rows), so one thread per row avoids a second
        // reduction launch and keeps the update order deterministic.
        for (int group_index = static_cast<int>(threadIdx.x); group_index < active_groups;
             group_index += kThreads) {
            const int query_index = group_index / kGqa;
            const int visible_in_tile = min(tile_tokens,
                                            query_start + query_base + query_index + 1 - tile_start);
            float maximum = -3.402823466e+38F;
            for (int tile_token = 0; tile_token < visible_in_tile; ++tile_token) {
                maximum = fmaxf(maximum,
                                score_shared[group_index * kFusedTokenTile + tile_token]);
            }
            float sum = 0.0F;
            for (int tile_token = 0; tile_token < visible_in_tile; ++tile_token) {
                const float probability =
                    expf(score_shared[group_index * kFusedTokenTile + tile_token] - maximum);
                score_shared[group_index * kFusedTokenTile + tile_token] = probability;
                sum += probability;
            }
            tile_max[group_index] = maximum;
            tile_sum[group_index] = sum;
        }
        __syncthreads();

        // Compute the online-softmax scales once per query/group. Reusing them across the 256
        // output dimensions avoids issuing the same two transcendental operations once per
        // dimension while retaining the same stable merge.
        if (threadIdx.x < active_groups) {
            const float old_m = m_state[threadIdx.x];
            const float new_m = fmaxf(old_m, tile_max[threadIdx.x]);
            const float old_scale = expf(old_m - new_m);
            const float tile_scale = expf(tile_max[threadIdx.x] - new_m);
            l_state[threadIdx.x] = l_state[threadIdx.x] * old_scale +
                                    tile_sum[threadIdx.x] * tile_scale;
            m_state[threadIdx.x] = new_m;
            tile_max[threadIdx.x] = old_scale;
            tile_sum[threadIdx.x] = tile_scale;
        }
        __syncthreads();

        // The AV update is performed by the 256 output-dimension threads. The two scales above
        // are shared by all dimensions for one query/group.
        for (int group_index = 0; group_index < active_groups; ++group_index) {
            const int query_index = group_index / kGqa;
            const int visible_in_tile = min(tile_tokens,
                                            query_start + query_base + query_index + 1 - tile_start);
            const float old_scale = tile_max[group_index];
            const float tile_scale = tile_sum[group_index];
            accum[group_index] *= old_scale;
            for (int tile_token = 0; tile_token < visible_in_tile; ++tile_token) {
                const float probability = score_shared[group_index * kFusedTokenTile + tile_token] *
                                          tile_scale;
                accum[group_index] +=
                    probability * k_shared[tile_token * kHeadDim + threadIdx.x];
            }
        }
        __syncthreads();
    }

    for (int group_index = 0; group_index < active_groups; ++group_index) {
        const int query_index = group_index / kGqa;
        const int query_group = group_index % kGqa;
        const int query_head = kv_head * kGqa + query_group;
        output[(static_cast<std::size_t>(query_base + query_index) * kQHeads + query_head) *
                   kHeadDim +
               threadIdx.x] = accum[group_index] / l_state[group_index];
    }
}

constexpr std::size_t kFusedSharedBytes =
    static_cast<std::size_t>(kFusedQueryTile * kGqa * kHeadDim +
                              2 * kFusedTokenTile * kHeadDim +
                              kFusedQueryTile * kGqa * kFusedTokenTile +
                              4 * kFusedQueryTile * kGqa) *
    sizeof(float);

// Decode has one query and therefore does not need the four-query accumulator and Q tile. It
// uses a 64-row slab, reusing the K storage for V after QK, to cut the per-token tile loop while
// staying below the opt-in SM120a dynamic shared-memory limit.
__global__ void oscar_mixed_fused_decode_kernel(
    const float* __restrict__ q_rotated, const std::uint16_t* __restrict__ prefix_k,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_k_packed,
    const float* __restrict__ historical_k_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_k, int recent_tokens, int recent_ring_head,
    const std::uint16_t* __restrict__ prefix_v,
    const std::uint8_t* __restrict__ historical_v_packed,
    const float* __restrict__ historical_v_metadata, const std::uint16_t* __restrict__ recent_v,
    int total_tokens, int query_token, float attention_scale, float* __restrict__ output) {
    extern __shared__ float shared[];
    constexpr int kQueryValues = kGqa * kHeadDim;
    constexpr int kTileValues = kFusedDecodeTokenTile * kHeadDim;
    constexpr int kScoreValues = kGqa * kFusedDecodeTokenTile;
    float* q_shared = shared;
    float* k_shared = q_shared + kQueryValues;
    float* score_shared = k_shared + kTileValues;
    float* tile_max = score_shared + kScoreValues;
    float* tile_sum = tile_max + kGqa;
    float* m_state = tile_sum + kGqa;
    float* l_state = m_state + kGqa;

    const int kv_head = static_cast<int>(blockIdx.x);
    for (int index = static_cast<int>(threadIdx.x); index < kQueryValues; index += kThreads) {
        const int query_group = index / kHeadDim;
        q_shared[index] = q_rotated[(kv_head * kGqa + query_group) * kHeadDim + index % kHeadDim];
    }
    if (threadIdx.x < kGqa) {
        m_state[threadIdx.x] = -3.402823466e+38F;
        l_state[threadIdx.x] = 0.0F;
    }
    __syncthreads();

    const int maximum_visible = min(total_tokens, query_token + 1);
    float accum[kGqa] = {};
    for (int tile_start = 0; tile_start < maximum_visible; tile_start += kFusedDecodeTokenTile) {
        const int tile_tokens = min(kFusedDecodeTokenTile, maximum_visible - tile_start);
        for (int index = static_cast<int>(threadIdx.x); index < kTileValues; index += kThreads) {
            const int tile_token = index / kHeadDim;
            if (tile_token < tile_tokens) {
                const int logical = tile_start + tile_token;
                k_shared[index] = load_key(logical, kv_head, index % kHeadDim, prefix_k,
                                           prefix_tokens, historical_k_packed,
                                           historical_k_metadata, historical_tokens, recent_k,
                                           recent_tokens, recent_ring_head);
            }
        }
        __syncthreads();

        // One warp owns the score row: a lane owns one K row and reuses its K scalars across GQA.
        for (int tile_token = static_cast<int>(threadIdx.x); tile_token < tile_tokens;
             tile_token += kThreads) {
            const float* key = k_shared + tile_token * kHeadDim;
            float scores[kGqa] = {};
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                const float key_value = key[dimension];
                #pragma unroll
                for (int query_group = 0; query_group < kGqa; ++query_group) {
                    scores[query_group] +=
                        q_shared[query_group * kHeadDim + dimension] * key_value;
                }
            }
            #pragma unroll
            for (int query_group = 0; query_group < kGqa; ++query_group) {
                score_shared[query_group * kFusedDecodeTokenTile + tile_token] =
                    scores[query_group] * attention_scale;
            }
        }
        __syncthreads();

        // The K slab is no longer needed after QK, so use it for V while the six row statistics
        // are computed. The barrier below joins both independent operations.
        for (int index = static_cast<int>(threadIdx.x); index < kTileValues; index += kThreads) {
            const int tile_token = index / kHeadDim;
            if (tile_token < tile_tokens) {
                const int logical = tile_start + tile_token;
                k_shared[index] = load_value(logical, kv_head, index % kHeadDim, prefix_v,
                                             prefix_tokens, historical_v_packed,
                                             historical_v_metadata, historical_tokens, recent_v,
                                             recent_tokens, recent_ring_head);
            }
        }
        for (int query_group = static_cast<int>(threadIdx.x); query_group < kGqa;
             query_group += kThreads) {
            const int visible_in_tile = min(tile_tokens, query_token + 1 - tile_start);
            float maximum = -3.402823466e+38F;
            for (int tile_token = 0; tile_token < visible_in_tile; ++tile_token) {
                maximum = fmaxf(maximum,
                                score_shared[query_group * kFusedDecodeTokenTile + tile_token]);
            }
            float sum = 0.0F;
            for (int tile_token = 0; tile_token < visible_in_tile; ++tile_token) {
                const float probability = expf(
                    score_shared[query_group * kFusedDecodeTokenTile + tile_token] - maximum);
                score_shared[query_group * kFusedDecodeTokenTile + tile_token] = probability;
                sum += probability;
            }
            tile_max[query_group] = maximum;
            tile_sum[query_group] = sum;
        }
        __syncthreads();

        if (threadIdx.x < kGqa) {
            const float old_m = m_state[threadIdx.x];
            const float new_m = fmaxf(old_m, tile_max[threadIdx.x]);
            const float old_scale = expf(old_m - new_m);
            const float tile_scale = expf(tile_max[threadIdx.x] - new_m);
            l_state[threadIdx.x] = l_state[threadIdx.x] * old_scale +
                                   tile_sum[threadIdx.x] * tile_scale;
            m_state[threadIdx.x] = new_m;
            tile_max[threadIdx.x] = old_scale;
            tile_sum[threadIdx.x] = tile_scale;
        }
        __syncthreads();

        for (int query_group = 0; query_group < kGqa; ++query_group) {
            const float old_scale = tile_max[query_group];
            const float tile_scale = tile_sum[query_group];
            accum[query_group] *= old_scale;
            const int visible_in_tile = min(tile_tokens, query_token + 1 - tile_start);
            for (int tile_token = 0; tile_token < visible_in_tile; ++tile_token) {
                const float probability =
                    score_shared[query_group * kFusedDecodeTokenTile + tile_token] * tile_scale;
                accum[query_group] +=
                    probability * k_shared[tile_token * kHeadDim + threadIdx.x];
            }
        }
        __syncthreads();
    }

    for (int query_group = 0; query_group < kGqa; ++query_group) {
        const int query_head = kv_head * kGqa + query_group;
        output[query_head * kHeadDim + threadIdx.x] = accum[query_group] / l_state[query_group];
    }
}

constexpr std::size_t kFusedDecodeSharedBytes =
    static_cast<std::size_t>(kGqa * kHeadDim + kFusedDecodeTokenTile * kHeadDim +
                              kGqa * kFusedDecodeTokenTile + 4 * kGqa) *
    sizeof(float);

constexpr int kFusedDecodeMaxSplits = kOscarMixedFusedDecodeMaxSplits;
constexpr std::size_t kFusedDecodePartialValues =
    static_cast<std::size_t>(kFusedDecodeMaxSplits) * kKVHeads * kGqa * kHeadDim;
constexpr std::size_t kFusedDecodePartialRows =
    static_cast<std::size_t>(kFusedDecodeMaxSplits) * kKVHeads * kGqa;
static_assert(kFusedDecodePartialValues * sizeof(float) +
                      2 * kFusedDecodePartialRows * sizeof(float) ==
                  kOscarMixedFusedDecodeWorkspaceBytes,
              "fused decode workspace contract changed");

__global__ void oscar_mixed_fused_decode_split_kernel(
    const float* __restrict__ q_rotated, const std::uint16_t* __restrict__ prefix_k,
    int prefix_tokens, const std::uint8_t* __restrict__ historical_k_packed,
    const float* __restrict__ historical_k_metadata, int historical_tokens,
    const std::uint16_t* __restrict__ recent_k, int recent_tokens, int recent_ring_head,
    const std::uint16_t* __restrict__ prefix_v,
    const std::uint8_t* __restrict__ historical_v_packed,
    const float* __restrict__ historical_v_metadata, const std::uint16_t* __restrict__ recent_v,
    int total_tokens, int query_token, int split_count, float attention_scale,
    float* __restrict__ workspace) {
    extern __shared__ float shared[];
    constexpr int kQueryValues = kGqa * kHeadDim;
    constexpr int kTileValues = kFusedDecodeTokenTile * kHeadDim;
    constexpr int kScoreValues = kGqa * kFusedDecodeTokenTile;
    float* q_shared = shared;
    float* k_shared = q_shared + kQueryValues;
    float* score_shared = k_shared + kTileValues;
    float* tile_max = score_shared + kScoreValues;
    float* tile_sum = tile_max + kGqa;
    float* m_state = tile_sum + kGqa;
    float* l_state = m_state + kGqa;

    const int kv_head = static_cast<int>(blockIdx.x);
    const int split_index = static_cast<int>(blockIdx.y);
    for (int index = static_cast<int>(threadIdx.x); index < kQueryValues; index += kThreads) {
        const int query_group = index / kHeadDim;
        q_shared[index] = q_rotated[(kv_head * kGqa + query_group) * kHeadDim + index % kHeadDim];
    }
    if (threadIdx.x < kGqa) {
        m_state[threadIdx.x] = -3.402823466e+38F;
        l_state[threadIdx.x] = 0.0F;
    }
    __syncthreads();

    const int maximum_visible = min(total_tokens, query_token + 1);
    const int split_start = (maximum_visible * split_index) / split_count;
    const int split_end = (maximum_visible * (split_index + 1)) / split_count;
    float accum[kGqa] = {};
    for (int tile_start = split_start; tile_start < split_end;
         tile_start += kFusedDecodeTokenTile) {
        const int tile_tokens = min(kFusedDecodeTokenTile, split_end - tile_start);
        for (int index = static_cast<int>(threadIdx.x); index < kTileValues; index += kThreads) {
            const int tile_token = index / kHeadDim;
            if (tile_token < tile_tokens) {
                const int logical = tile_start + tile_token;
                k_shared[index] = load_key(logical, kv_head, index % kHeadDim, prefix_k,
                                           prefix_tokens, historical_k_packed,
                                           historical_k_metadata, historical_tokens, recent_k,
                                           recent_tokens, recent_ring_head);
            }
        }
        __syncthreads();

        for (int tile_token = static_cast<int>(threadIdx.x); tile_token < tile_tokens;
             tile_token += kThreads) {
            const float* key = k_shared + tile_token * kHeadDim;
            float scores[kGqa] = {};
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                const float key_value = key[dimension];
                #pragma unroll
                for (int query_group = 0; query_group < kGqa; ++query_group) {
                    scores[query_group] +=
                        q_shared[query_group * kHeadDim + dimension] * key_value;
                }
            }
            #pragma unroll
            for (int query_group = 0; query_group < kGqa; ++query_group) {
                score_shared[query_group * kFusedDecodeTokenTile + tile_token] =
                    scores[query_group] * attention_scale;
            }
        }
        __syncthreads();

        for (int index = static_cast<int>(threadIdx.x); index < kTileValues; index += kThreads) {
            const int tile_token = index / kHeadDim;
            if (tile_token < tile_tokens) {
                const int logical = tile_start + tile_token;
                k_shared[index] = load_value(logical, kv_head, index % kHeadDim, prefix_v,
                                             prefix_tokens, historical_v_packed,
                                             historical_v_metadata, historical_tokens, recent_v,
                                             recent_tokens, recent_ring_head);
            }
        }
        for (int query_group = static_cast<int>(threadIdx.x); query_group < kGqa;
             query_group += kThreads) {
            float maximum = -3.402823466e+38F;
            for (int tile_token = 0; tile_token < tile_tokens; ++tile_token) {
                maximum = fmaxf(maximum,
                                score_shared[query_group * kFusedDecodeTokenTile + tile_token]);
            }
            float sum = 0.0F;
            for (int tile_token = 0; tile_token < tile_tokens; ++tile_token) {
                const float probability = expf(
                    score_shared[query_group * kFusedDecodeTokenTile + tile_token] - maximum);
                score_shared[query_group * kFusedDecodeTokenTile + tile_token] = probability;
                sum += probability;
            }
            tile_max[query_group] = maximum;
            tile_sum[query_group] = sum;
        }
        __syncthreads();

        if (threadIdx.x < kGqa) {
            const float old_m = m_state[threadIdx.x];
            const float new_m = fmaxf(old_m, tile_max[threadIdx.x]);
            const float old_scale = expf(old_m - new_m);
            const float tile_scale = expf(tile_max[threadIdx.x] - new_m);
            l_state[threadIdx.x] = l_state[threadIdx.x] * old_scale +
                                   tile_sum[threadIdx.x] * tile_scale;
            m_state[threadIdx.x] = new_m;
            tile_max[threadIdx.x] = old_scale;
            tile_sum[threadIdx.x] = tile_scale;
        }
        __syncthreads();

        for (int query_group = 0; query_group < kGqa; ++query_group) {
            const float old_scale = tile_max[query_group];
            const float tile_scale = tile_sum[query_group];
            accum[query_group] *= old_scale;
            for (int tile_token = 0; tile_token < tile_tokens; ++tile_token) {
                const float probability =
                    score_shared[query_group * kFusedDecodeTokenTile + tile_token] * tile_scale;
                accum[query_group] +=
                    probability * k_shared[tile_token * kHeadDim + threadIdx.x];
            }
        }
        __syncthreads();
    }

    const std::size_t split_head =
        static_cast<std::size_t>(split_index) * kKVHeads + kv_head;
    float* partial_m = workspace + kFusedDecodePartialValues;
    float* partial_l = partial_m + kFusedDecodePartialRows;
    for (int query_group = static_cast<int>(threadIdx.x); query_group < kGqa;
         query_group += kThreads) {
        partial_m[split_head * kGqa + query_group] = m_state[query_group];
        partial_l[split_head * kGqa + query_group] = l_state[query_group];
    }
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        workspace[(split_head * kGqa + query_group) * kHeadDim + threadIdx.x] =
            accum[query_group];
    }
}

__global__ void oscar_mixed_fused_decode_merge_kernel(const float* __restrict__ workspace,
                                                      int split_count,
                                                      float* __restrict__ output) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    const float* partial_m = workspace + kFusedDecodePartialValues;
    const float* partial_l = partial_m + kFusedDecodePartialRows;
    for (int query_group = 0; query_group < kGqa; ++query_group) {
        float maximum = -3.402823466e+38F;
        for (int split = 0; split < split_count; ++split) {
            const std::size_t row = static_cast<std::size_t>(split) * kKVHeads + kv_head;
            maximum = fmaxf(maximum, partial_m[row * kGqa + query_group]);
        }
        float total_l = 0.0F;
        float total_value = 0.0F;
        for (int split = 0; split < split_count; ++split) {
            const std::size_t row = static_cast<std::size_t>(split) * kKVHeads + kv_head;
            const float scale = expf(partial_m[row * kGqa + query_group] - maximum);
            total_l += partial_l[row * kGqa + query_group] * scale;
            const std::size_t value =
                (row * kGqa + query_group) * kHeadDim + dimension;
            total_value += workspace[value] * scale;
        }
        const int query_head = kv_head * kGqa + query_group;
        output[query_head * kHeadDim + dimension] = total_value / total_l;
    }
}

__device__ __forceinline__ std::uint16_t oscar_float_to_bf16(float value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(__float_as_uint(value));
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

__device__ __forceinline__ float oscar_bf16_to_float(std::uint16_t bits) {
    return __int_as_float(static_cast<int>(static_cast<std::uint32_t>(bits) << 16U));
}

// The official host reference first rounds a newly produced FP32 row to BF16 while it is in the
// protected/recent tier, then quantizes that BF16 value when it ages.  The resident path keeps
// that contract by applying the same round-to-nearest-even conversion before either storing or
// encoding a value.
__device__ __forceinline__ std::uint16_t resident_input_bf16(
    const float* rotated, std::uint32_t input_token, int kv_head, int dimension,
    int token_count) {
    const std::size_t index = static_cast<std::size_t>(dimension) +
                              static_cast<std::size_t>(kHeadDim) *
                                  (static_cast<std::size_t>(kv_head) +
                                   static_cast<std::size_t>(kKVHeads) * input_token);
    (void)token_count;
    return oscar_float_to_bf16(rotated[index]);
}

__device__ __forceinline__ std::uint16_t resident_recent_bf16(
    const std::uint16_t* recent, std::uint32_t logical_token, int kv_head, int dimension,
    std::uint32_t old_recent_begin, std::uint32_t old_recent_head) {
    const std::uint32_t physical =
        (old_recent_head + logical_token - old_recent_begin) & (kTokenTile - 1);
    const std::size_t index =
        (static_cast<std::size_t>(physical) * kKVHeads + kv_head) * kHeadDim + dimension;
    return recent[index];
}

// This is a deliberately simple GPU encoder used only at the cache-publish boundary.  It uses a
// 256-element bitonic sort to reproduce the official sorted-absolute percentile clip, then the
// exact two independent g128 asymmetric groups and quartered byte layout.  The historical
// attention kernel still decodes the resulting bytes directly; no decoded historical buffer is
// created.
__device__ void encode_resident_row(
    const float* input, const std::uint16_t* old_recent, bool from_recent,
    std::uint32_t logical_token, std::uint32_t logical_start, int token_count, int kv_head,
    std::uint32_t old_recent_begin, std::uint32_t old_recent_head, int clip_index,
    std::uint8_t* packed_output, float* metadata_output, float* abs_values,
    float* clipped_values, std::uint8_t* symbols) {
    const int dimension = static_cast<int>(threadIdx.x);
    const std::uint32_t input_token = logical_token - logical_start;
    const std::uint16_t bits = from_recent
                                   ? resident_recent_bf16(old_recent, logical_token, kv_head,
                                                          dimension, old_recent_begin,
                                                          old_recent_head)
                                   : resident_input_bf16(input, input_token, kv_head, dimension,
                                                         token_count);
    const float value = oscar_bf16_to_float(bits);
    abs_values[dimension] = fabsf(value);
    clipped_values[dimension] = value;
    __syncthreads();

    for (int size = 2; size <= kHeadDim; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int partner = dimension ^ stride;
            if (partner > dimension) {
                const bool ascending = (dimension & size) == 0;
                const float left = abs_values[dimension];
                const float right = abs_values[partner];
                if ((ascending && left > right) || (!ascending && left < right)) {
                    abs_values[dimension] = right;
                    abs_values[partner] = left;
                }
            }
            __syncthreads();
        }
    }
    const float threshold = abs_values[clip_index];
    const float clipped = fminf(fmaxf(value, -threshold), threshold);
    clipped_values[dimension] = clipped;
    __syncthreads();

    if (dimension == 0) {
        for (int group = 0; group < kGroups; ++group) {
            const int begin = group * kGroupSize;
            float minimum = clipped_values[begin];
            float maximum = minimum;
            for (int offset = 1; offset < kGroupSize; ++offset) {
                const float current = clipped_values[begin + offset];
                minimum = fminf(minimum, current);
                maximum = fmaxf(maximum, current);
            }
            const float range = fmaxf(maximum - minimum, 1.0e-8F);
            metadata_output[group * 2] = range / static_cast<float>(kInt2Levels);
            metadata_output[group * 2 + 1] = -minimum / metadata_output[group * 2];
        }
    }
    __syncthreads();
    float normalized = clipped_values[dimension] /
                           metadata_output[(dimension >> 7) * 2] +
                       metadata_output[(dimension >> 7) * 2 + 1] + 0.5F;
    int code = static_cast<int>(floorf(normalized));
    code = max(0, min(kInt2Levels, code));
    symbols[dimension] = static_cast<std::uint8_t>(code);
    __syncthreads();
    if (dimension < kCodeBytes) {
        packed_output[dimension] = static_cast<std::uint8_t>(
            symbols[dimension] | (symbols[dimension + 64] << 2U) |
            (symbols[dimension + 128] << 4U) | (symbols[dimension + 192] << 6U));
    }
    __syncthreads();
}

__global__ void oscar_resident_write_bf16_kernel(
    const float* __restrict__ rotated_k, const float* __restrict__ rotated_v, int token_count,
    std::uint32_t logical_start, std::uint32_t final_recent_begin,
    std::uint32_t final_recent_head, std::uint16_t* __restrict__ prefix_k,
    std::uint16_t* __restrict__ prefix_v, std::uint16_t* __restrict__ recent_k,
    std::uint16_t* __restrict__ recent_v) {
    const std::uint32_t input_token = static_cast<std::uint32_t>(blockIdx.x);
    if (input_token >= static_cast<std::uint32_t>(token_count)) return;
    const std::uint32_t logical = logical_start + input_token;
    if (logical >= final_recent_begin && logical < final_recent_begin + kRecentTokens) {
        const std::uint32_t physical = (final_recent_head + logical - final_recent_begin) &
                                       (kTokenTile - 1);
        for (int flat = static_cast<int>(threadIdx.x); flat < kKVHeads * kHeadDim;
             flat += kThreads) {
            const int head = flat / kHeadDim;
            const int dimension = flat % kHeadDim;
            const std::size_t source = static_cast<std::size_t>(dimension) +
                                       static_cast<std::size_t>(kHeadDim) *
                                           (static_cast<std::size_t>(head) +
                                            static_cast<std::size_t>(kKVHeads) * input_token);
            const std::size_t destination =
                (static_cast<std::size_t>(physical) * kKVHeads + head) * kHeadDim + dimension;
            recent_k[destination] = oscar_float_to_bf16(rotated_k[source]);
            recent_v[destination] = oscar_float_to_bf16(rotated_v[source]);
        }
    } else if (logical < kPrefixTokens) {
        for (int flat = static_cast<int>(threadIdx.x); flat < kKVHeads * kHeadDim;
             flat += kThreads) {
            const int head = flat / kHeadDim;
            const int dimension = flat % kHeadDim;
            const std::size_t source = static_cast<std::size_t>(dimension) +
                                       static_cast<std::size_t>(kHeadDim) *
                                           (static_cast<std::size_t>(head) +
                                            static_cast<std::size_t>(kKVHeads) * input_token);
            const std::size_t destination =
                (static_cast<std::size_t>(logical) * kKVHeads + head) * kHeadDim + dimension;
            prefix_k[destination] = oscar_float_to_bf16(rotated_k[source]);
            prefix_v[destination] = oscar_float_to_bf16(rotated_v[source]);
        }
    }
}

__global__ void oscar_resident_encode_kernel(
    const float* __restrict__ rotated_k, const float* __restrict__ rotated_v,
    int token_count, std::uint32_t logical_start, std::uint32_t old_context,
    std::uint32_t old_recent_begin, std::uint32_t old_recent_head,
    std::uint32_t final_recent_begin, std::uint32_t new_historical_tokens,
    std::uint32_t old_aging_tokens, std::uint16_t* __restrict__ old_recent_k,
    std::uint16_t* __restrict__ old_recent_v, std::uint8_t* __restrict__ historical_k,
    std::uint8_t* __restrict__ historical_v, float* __restrict__ historical_k_metadata,
    float* __restrict__ historical_v_metadata) {
    const std::uint32_t work = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t total_work = (new_historical_tokens + old_aging_tokens) * kKVHeads;
    if (work >= total_work) return;
    const std::uint32_t row = work / kKVHeads;
    const int head = static_cast<int>(work % kKVHeads);
    const bool from_recent = row >= new_historical_tokens;
    const std::uint32_t logical = from_recent
                                      ? old_recent_begin + row - new_historical_tokens
                                      : logical_start + row;
    if (logical < kPrefixTokens || logical >= final_recent_begin ||
        logical >= old_context + (from_recent ? 0U : static_cast<std::uint32_t>(token_count)) ||
        logical < logical_start && !from_recent) {
        return;
    }
    const std::size_t output_row =
        (static_cast<std::size_t>(logical - kPrefixTokens) * kKVHeads) + head;
    extern __shared__ std::uint8_t shared_bytes[];
    float* abs_values = reinterpret_cast<float*>(shared_bytes);
    float* clipped_values = abs_values + kHeadDim;
    std::uint8_t* symbols = reinterpret_cast<std::uint8_t*>(clipped_values + kHeadDim);
    std::uint8_t* k_packed = symbols + kHeadDim;
    float* k_metadata = reinterpret_cast<float*>(k_packed + kCodeBytes);
    std::uint8_t* v_packed = reinterpret_cast<std::uint8_t*>(k_metadata + kInt2MetadataItems);
    float* v_metadata = reinterpret_cast<float*>(v_packed + kCodeBytes);

    const float* k_input = rotated_k;
    const float* v_input = rotated_v;
    float* k_scratch = abs_values;
    float* v_scratch = abs_values;
    encode_resident_row(k_input, old_recent_k, from_recent, logical, logical_start, token_count,
                        head, old_recent_begin, old_recent_head, 245, k_packed, k_metadata,
                        k_scratch, clipped_values, symbols);
    encode_resident_row(v_input, old_recent_v, from_recent, logical, logical_start, token_count,
                        head, old_recent_begin, old_recent_head, 235, v_packed, v_metadata,
                        v_scratch, clipped_values, symbols);
    if (threadIdx.x == 0) {
        const std::size_t payload_offset = output_row * kCodeBytes;
        const std::size_t metadata_offset = output_row * kInt2MetadataItems;
        for (int byte = 0; byte < kCodeBytes; ++byte) {
            historical_k[payload_offset + byte] = k_packed[byte];
            historical_v[payload_offset + byte] = v_packed[byte];
        }
        for (int item = 0; item < kInt2MetadataItems; ++item) {
            historical_k_metadata[metadata_offset + item] = k_metadata[item];
            historical_v_metadata[metadata_offset + item] = v_metadata[item];
        }
    }
}

} // namespace

void oscar_int2_g128_mixed_attention_launch(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens, float attention_scale,
    float* scores, float* softmax, float* output, cudaStream_t stream) {
    oscar_int2_g128_mixed_attention_launch_ring(
        q_rotated, prefix_k_bf16, prefix_v_bf16, prefix_tokens, historical_k_packed,
        historical_k_metadata, historical_v_packed, historical_v_metadata, historical_tokens,
        recent_k_bf16, recent_v_bf16, recent_tokens, 0, attention_scale, scores, softmax, output,
        stream);
}

void oscar_int2_g128_mixed_attention_launch_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, float attention_scale, float* scores, float* softmax,
    float* output, cudaStream_t stream) {
    if (prefix_tokens < 0 || prefix_tokens > 64 || historical_tokens < 0 ||
        recent_tokens < 0 || recent_tokens > kRecentTokens || recent_ring_head < 0 ||
        recent_ring_head >= kRecentTokens) {
        throw std::invalid_argument("OSCAR mixed attention tier count is invalid");
    }
    const int total_tokens = prefix_tokens + historical_tokens + recent_tokens;
    if (total_tokens <= 0) throw std::invalid_argument("OSCAR mixed attention is empty");
    const dim3 score_grid(kKVHeads,
                          static_cast<unsigned>((total_tokens + kTokenTile - 1) / kTokenTile), 1u);
    oscar_mixed_score_kernel<<<score_grid, kThreads, 0, stream>>>(
        q_rotated, prefix_k_bf16, prefix_tokens, historical_k_packed, historical_k_metadata,
        historical_tokens, recent_k_bf16, recent_tokens, recent_ring_head, total_tokens,
        attention_scale, scores);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_softmax_kernel<<<kQHeads, kThreads, 0, stream>>>(scores, total_tokens, softmax);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_av_kernel<<<kKVHeads, kThreads, 0, stream>>>(
        softmax, prefix_v_bf16, prefix_tokens, historical_v_packed, historical_v_metadata,
        historical_tokens, recent_v_bf16, recent_tokens, recent_ring_head, total_tokens, output);
    CUDA_CHECK(cudaGetLastError());
}

void oscar_int2_g128_mixed_attention_launch_batch_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, std::int32_t query_start, std::int32_t query_count,
    std::int32_t workspace_stride, float attention_scale, float* scores, float* softmax,
    float* output, cudaStream_t stream) {
    if (prefix_tokens < 0 || prefix_tokens > kPrefixTokens || historical_tokens < 0 ||
        recent_tokens < 0 || recent_tokens > kRecentTokens || recent_ring_head < 0 ||
        recent_ring_head >= kRecentTokens || query_start < 0 || query_count <= 0 ||
        query_count > kRecentTokens / 4 || workspace_stride <= 0) {
        throw std::invalid_argument("OSCAR mixed batched attention arguments are invalid");
    }
    const int total_tokens = prefix_tokens + historical_tokens + recent_tokens;
    if (total_tokens <= 0 || workspace_stride < total_tokens ||
        query_start + query_count > total_tokens) {
        throw std::invalid_argument("OSCAR mixed batched attention extent is invalid");
    }
    const dim3 score_grid(
        kKVHeads, static_cast<unsigned>((total_tokens + kTokenTile - 1) / kTokenTile),
        static_cast<unsigned>(query_count));
    oscar_mixed_score_batch_kernel<<<score_grid, kThreads, 0, stream>>>(
        q_rotated, prefix_k_bf16, prefix_tokens, historical_k_packed, historical_k_metadata,
        historical_tokens, recent_k_bf16, recent_tokens, recent_ring_head, total_tokens,
        query_start, workspace_stride, attention_scale, scores);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_softmax_batch_kernel<<<static_cast<unsigned>(query_count * kQHeads), kThreads, 0,
                                       stream>>>(scores, total_tokens, query_start, query_count,
                                                workspace_stride, softmax);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_av_batch_kernel<<<static_cast<unsigned>(query_count * kKVHeads), kThreads, 0,
                                  stream>>>(
        softmax, prefix_v_bf16, prefix_tokens, historical_v_packed, historical_v_metadata,
        historical_tokens, recent_v_bf16, recent_tokens, recent_ring_head, total_tokens,
        query_start, query_count, workspace_stride, output);
    CUDA_CHECK(cudaGetLastError());
}

void oscar_int2_g128_mixed_attention_launch_fused_batch_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, std::int32_t query_start, std::int32_t query_count,
    float attention_scale, float* output, cudaStream_t stream) {
    if (q_rotated == nullptr || output == nullptr || prefix_tokens < 0 ||
        prefix_tokens > kPrefixTokens || historical_tokens < 0 || recent_tokens < 0 ||
        recent_tokens > kRecentTokens || recent_ring_head < 0 ||
        recent_ring_head >= kRecentTokens || query_start < 0 || query_count <= 0 ||
        query_count > 64) {
        throw std::invalid_argument("OSCAR fused attention arguments are invalid");
    }
    const int total_tokens = prefix_tokens + historical_tokens + recent_tokens;
    if (total_tokens <= 0 || query_start + query_count > total_tokens) {
        throw std::invalid_argument("OSCAR fused attention extent is invalid");
    }
    static std::once_flag fused_attribute_once;
    std::call_once(fused_attribute_once, [] {
        CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(oscar_mixed_fused_batch_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kFusedSharedBytes)));
        CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(oscar_mixed_fused_decode_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kFusedDecodeSharedBytes)));
    });
    if (query_count == 1) {
        const dim3 decode_grid(kKVHeads, 1u, 1u);
        oscar_mixed_fused_decode_kernel<<<decode_grid, kThreads, kFusedDecodeSharedBytes, stream>>>(
            q_rotated, prefix_k_bf16, prefix_tokens, historical_k_packed,
            historical_k_metadata, historical_tokens, recent_k_bf16, recent_tokens,
            recent_ring_head, prefix_v_bf16, historical_v_packed, historical_v_metadata,
            recent_v_bf16, total_tokens, query_start, attention_scale, output);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 grid(kKVHeads,
                    static_cast<unsigned>((query_count + kFusedQueryTile - 1) /
                                          kFusedQueryTile),
                    1u);
    oscar_mixed_fused_batch_kernel<<<grid, kThreads, kFusedSharedBytes, stream>>>(
        q_rotated, prefix_k_bf16, prefix_tokens, historical_k_packed, historical_k_metadata,
        historical_tokens, recent_k_bf16, recent_tokens, recent_ring_head, prefix_v_bf16,
        historical_v_packed, historical_v_metadata, recent_v_bf16, total_tokens, query_start,
        query_count, attention_scale, output);
    CUDA_CHECK(cudaGetLastError());
}

int oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(std::int32_t visible_tokens) {
    if (visible_tokens <= 512) return 16;
    if (visible_tokens <= 8192) return 32;
    return 64;
}

void oscar_int2_g128_mixed_attention_launch_fused_decode_split_ring(
    const float* q_rotated, const std::uint16_t* prefix_k_bf16,
    const std::uint16_t* prefix_v_bf16, std::int32_t prefix_tokens,
    const std::uint8_t* historical_k_packed, const float* historical_k_metadata,
    const std::uint8_t* historical_v_packed, const float* historical_v_metadata,
    std::int32_t historical_tokens, const std::uint16_t* recent_k_bf16,
    const std::uint16_t* recent_v_bf16, std::int32_t recent_tokens,
    std::int32_t recent_ring_head, std::int32_t query_token, std::int32_t split_count,
    float attention_scale, float* partial_workspace, float* output, cudaStream_t stream) {
    if (q_rotated == nullptr || partial_workspace == nullptr || output == nullptr ||
        prefix_tokens < 0 || prefix_tokens > kPrefixTokens || historical_tokens < 0 ||
        recent_tokens < 0 || recent_tokens > kRecentTokens || recent_ring_head < 0 ||
        recent_ring_head >= kRecentTokens || query_token < 0 ||
        (split_count != kOscarMixedFusedDecodeAdaptiveSplits && split_count != 1 &&
         split_count != 2 && split_count != 4 && split_count != 8 && split_count != 16 &&
         split_count != 32 && split_count != 64)) {
        throw std::invalid_argument("OSCAR split fused decode arguments are invalid");
    }
    const int total_tokens = prefix_tokens + historical_tokens + recent_tokens;
    if (total_tokens <= 0 || query_token >= total_tokens) {
        throw std::invalid_argument("OSCAR split fused decode extent is invalid");
    }
    const int maximum_visible = std::min(total_tokens, query_token + 1);
    const int selected_splits = split_count == kOscarMixedFusedDecodeAdaptiveSplits
                                    ? oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
                                          maximum_visible)
                                    : split_count;
    if (selected_splits < 1 || selected_splits > kFusedDecodeMaxSplits) {
        throw std::logic_error("OSCAR adaptive fused decode selected an invalid split count");
    }
    static std::once_flag fused_decode_attribute_once;
    std::call_once(fused_decode_attribute_once, [] {
        CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(oscar_mixed_fused_decode_split_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kFusedDecodeSharedBytes)));
    });
    const dim3 split_grid(kKVHeads, static_cast<unsigned>(selected_splits), 1u);
    oscar_mixed_fused_decode_split_kernel<<<split_grid, kThreads, kFusedDecodeSharedBytes, stream>>>(
        q_rotated, prefix_k_bf16, prefix_tokens, historical_k_packed, historical_k_metadata,
        historical_tokens, recent_k_bf16, recent_tokens, recent_ring_head, prefix_v_bf16,
        historical_v_packed, historical_v_metadata, recent_v_bf16, total_tokens, query_token,
        selected_splits, attention_scale, partial_workspace);
    CUDA_CHECK(cudaGetLastError());
    oscar_mixed_fused_decode_merge_kernel<<<static_cast<unsigned>(kKVHeads), kThreads, 0,
                                            stream>>>(partial_workspace, selected_splits, output);
    CUDA_CHECK(cudaGetLastError());
}

void oscar_int2_g128_cache_write_bf16_launch(
    const float* rotated_k, const float* rotated_v, std::int32_t token_count,
    std::uint32_t logical_start, std::uint32_t final_recent_begin,
    std::uint32_t final_recent_head, const OscarInt2G128ResidentCacheView& cache,
    cudaStream_t stream) {
    if (rotated_k == nullptr || rotated_v == nullptr || token_count <= 0 ||
        final_recent_head >= kRecentTokens || cache.prefix_k_bf16 == nullptr ||
        cache.prefix_v_bf16 == nullptr || cache.recent_k_bf16 == nullptr ||
        cache.recent_v_bf16 == nullptr) {
        throw std::invalid_argument("OSCAR resident BF16 append arguments are invalid");
    }
    oscar_resident_write_bf16_kernel<<<static_cast<unsigned>(token_count), kThreads, 0, stream>>>(
        rotated_k, rotated_v, token_count, logical_start, final_recent_begin, final_recent_head,
        cache.prefix_k_bf16, cache.prefix_v_bf16, cache.recent_k_bf16, cache.recent_v_bf16);
    CUDA_CHECK(cudaGetLastError());
}

void oscar_int2_g128_cache_encode_launch(
    const float* rotated_k, const float* rotated_v, std::int32_t token_count,
    std::uint32_t logical_start, std::uint32_t old_context, std::uint32_t old_recent_begin,
    std::uint32_t old_recent_head, std::uint32_t final_recent_begin,
    std::uint32_t new_historical_tokens, std::uint32_t old_aging_tokens,
    const OscarInt2G128ResidentCacheView& cache, cudaStream_t stream) {
    if (rotated_k == nullptr || rotated_v == nullptr || token_count <= 0 ||
        old_recent_head >= kRecentTokens || cache.historical_k_packed == nullptr ||
        cache.historical_v_packed == nullptr || cache.historical_k_metadata == nullptr ||
        cache.historical_v_metadata == nullptr || cache.max_context <= 0) {
        throw std::invalid_argument("OSCAR resident INT2 append arguments are invalid");
    }
    const std::uint32_t work_items = (new_historical_tokens + old_aging_tokens) * kKVHeads;
    if (work_items == 0) return;
    constexpr std::size_t kSharedBytes =
        2ULL * kHeadDim * sizeof(float) + kHeadDim * sizeof(std::uint8_t) +
        kCodeBytes * 2ULL * sizeof(std::uint8_t) +
        kInt2MetadataItems * 2ULL * sizeof(float);
    oscar_resident_encode_kernel<<<static_cast<unsigned>(work_items), kThreads, kSharedBytes,
                                   stream>>>(
        rotated_k, rotated_v, token_count, logical_start, old_context, old_recent_begin,
        old_recent_head, final_recent_begin, new_historical_tokens, old_aging_tokens,
        cache.recent_k_bf16, cache.recent_v_bf16, cache.historical_k_packed,
        cache.historical_v_packed, cache.historical_k_metadata, cache.historical_v_metadata);
    CUDA_CHECK(cudaGetLastError());
}

OscarMixedKernelResources oscar_int2_g128_mixed_kernel_resources() {
    OscarMixedKernelResources result;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_mixed_score_kernel)));
    result.score_registers = attributes.numRegs;
    result.score_static_shared_bytes = attributes.sharedSizeBytes;
    result.score_max_threads = attributes.maxThreadsPerBlock;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_mixed_softmax_kernel)));
    result.softmax_registers = attributes.numRegs;
    result.softmax_static_shared_bytes = attributes.sharedSizeBytes;
    result.softmax_max_threads = attributes.maxThreadsPerBlock;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     reinterpret_cast<const void*>(oscar_mixed_av_kernel)));
    result.av_registers = attributes.numRegs;
    result.av_static_shared_bytes = attributes.sharedSizeBytes;
    result.av_max_threads = attributes.maxThreadsPerBlock;
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, reinterpret_cast<const void*>(oscar_mixed_fused_batch_kernel)));
    result.fused_registers = attributes.numRegs;
    result.fused_dynamic_shared_bytes = kFusedSharedBytes;
    result.fused_max_threads = attributes.maxThreadsPerBlock;
    return result;
}

} // namespace ninfer::ops::detail
