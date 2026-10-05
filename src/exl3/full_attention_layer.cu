#include "exl3/pdl_small.cuh"
#include "exl3/full_attention_layer.h"
#include "exl3/environment_options.h"
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/linear_workspace_requirements.h"
#include "exl3/block_tree_sum.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <type_traits>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3 {
namespace {

constexpr int kHidden = 5120;
constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kQProjection = kQHeads * kHeadDim * 2;
constexpr int kKVProjection = kKVHeads * kHeadDim;
constexpr int kIntermediate = 17408;
constexpr int kRopeDim = 64;
constexpr float kRopeTheta = 10000000.0f;
constexpr float kRmsEps = 1.0e-6f;
constexpr std::array<int,17> kBufferFeatures={{kHidden,kQProjection,kKVProjection,kKVProjection,
    kQHeads*kHeadDim,kKVHeads*kHeadDim,kQHeads*kHeadDim,kKVHeads*kHeadDim,
    kQHeads*kHeadDim,kHidden,kHidden,kHidden,kIntermediate,kIntermediate,kIntermediate,kHidden,kHidden}};
constexpr bool transient_buffer(std::size_t i) {return i!=3 && i!=6 && i!=7;}

std::atomic<std::uint64_t> prefill_shared_score_launch_counter{0};
std::atomic<std::uint64_t> prefill_shared_score_row_counter{0};
std::atomic<std::uint64_t> prefill_shared_score_bytes_counter{0};
std::atomic<std::uint64_t> prefill_shared_score_threads256_counter{0};
std::atomic<std::uint64_t> prefill_shared_score_parallel_softmax_counter{0};
std::atomic<std::uint64_t> prefill_shared_score_head_split256_counter{0};
std::atomic<std::uint64_t> prefill_shared_score_dimension_split256_counter{0};
std::atomic<std::uint64_t> fast_fused_flash_attention_counter{0};
std::atomic<std::uint64_t> fast_fused_flash_multirow_attention_counter{0};
std::atomic<std::uint64_t> fast_whole_context_fused_attention_counter{0};
std::atomic<std::uint64_t> fast_prefill_tiled_attention_counter{0};
std::atomic<std::uint64_t> fast_prefill_rows4_attention_counter{0};
std::atomic<std::uint64_t> fast_prefill_rows8_attention_counter{0};
std::atomic<std::uint64_t> fast_prefill_wmma_attention_counter{0};
std::atomic<std::uint64_t> fast_prefill_wmma32_attention_counter{0};
std::atomic<std::uint64_t> fast_prefill_wmma64_attention_counter{0};
std::atomic<std::uint64_t> fast_prefill_rows2_attention_counter{0};
std::atomic<std::uint64_t> fast_online_decode_attention_counter{0};
std::atomic<std::uint64_t> fast_cublas_attention_counter{0};

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__global__ void split_qg_kernel(const std::uint16_t* qg,
                                std::uint16_t* q,
                                std::uint16_t* gate,
                                int rows) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kQHeads * kHeadDim;
    if (index >= total) return;
    const int head = index / kHeadDim;
    const int head_in_row = head % kQHeads;
    const int channel = index % kHeadDim;
    const int source = head_in_row * (2 * kHeadDim) + channel;
    q[index] = qg[(head / kQHeads) * kQProjection + source];
    gate[index] = qg[(head / kQHeads) * kQProjection + source + kHeadDim];
}

__global__ void rms_norm_kernel(const std::uint16_t* input,
                                const std::uint16_t* weight,
                                std::uint16_t* output,
                                int rows,
                                int features) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= features) return;
    extern __shared__ float shared[];
    // Decode-shaped rows (5120 over 512 threads, 256-wide heads over 256):
    // loads issued together and kept in registers; the per-thread x*x order
    // and the pairwise tree are unchanged (block_tree_sum_exact).
    const auto cached = [&](auto width, auto count) {
        constexpr int B = decltype(width)::value, C = decltype(count)::value;
        std::uint16_t x_bits[C], w_bits[C];
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            x_bits[j] = input[row * features + lane + j * B];
            w_bits[j] = weight[lane + j * B];
        }
        float partial = 0.0f;
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            const float value = __half2float(__ushort_as_half(x_bits[j]));
            partial += value * value;
        }
        const float inv = rsqrtf(block_tree_sum_exact<B>(partial, shared, lane) /
                                 static_cast<float>(features) + kRmsEps);
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            const float x = __half2float(__ushort_as_half(x_bits[j])) * inv;
            const float w = __half2float(__ushort_as_half(w_bits[j]));
            output[row * features + lane + j * B] =
                __half_as_ushort(__float2half_rn(x * (w + 1.0f)));
        }
    };
    if (blockDim.x == 512 && features == 5120) {
        cached(std::integral_constant<int, 512>{}, std::integral_constant<int, 10>{});
        return;
    }
    if (blockDim.x == 256 && features == 256) {
        cached(std::integral_constant<int, 256>{}, std::integral_constant<int, 1>{});
        return;
    }
    float sum = 0.0f;
    for (int i = lane; i < features; i += blockDim.x) {
        const float value = __half2float(__ushort_as_half(input[row * features + i]));
        sum += value * value;
    }
    shared[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(shared[0] / static_cast<float>(features) + kRmsEps);
    for (int i = lane; i < features; i += blockDim.x) {
        const float x = __half2float(__ushort_as_half(input[row * features + i])) * inv;
        const float w = __half2float(__ushort_as_half(weight[i]));
        output[row * features + i] = __half_as_ushort(__float2half_rn(x * (w + 1.0f)));
    }
}

template<bool Mrope=false>
__global__ void rope_kernel(const std::uint16_t* q_in,
                            const std::uint16_t* k_in,
                            std::uint16_t* q_out,
                            std::uint16_t* k_out,
                            int rows,
                            int position,
                            const int* position_device,const int* positions_xyz,int offset) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kQHeads * kHeadDim;
    if (index >= total) return;
    const int channel = index % kHeadDim;
    const int row = index / (kQHeads * kHeadDim);
    const int kv_index = row * kKVHeads * kHeadDim +
                         ((index / kHeadDim) % kKVHeads) * kHeadDim + channel;
    const int base_position = position_device ? *position_device : position;
    int pos = base_position + row;
    if constexpr(Mrope)pos=positions_xyz?positions_xyz[row*3+(channel%(kRopeDim/2))%3]:pos+offset;
    float q = __half2float(__ushort_as_half(q_in[index]));
    float k = __half2float(__ushort_as_half(k_in[kv_index]));
    if (channel < kRopeDim) {
        const int pair = channel < kRopeDim / 2 ? channel : channel - kRopeDim / 2;
        const float inv = powf(kRopeTheta, -2.0f * static_cast<float>(pair) /
                                           static_cast<float>(kRopeDim));
        const float angle = static_cast<float>(pos) * inv;
        const float sine = sinf(angle);
        const float cosine = cosf(angle);
        const int mate = channel < kRopeDim / 2 ? channel + kRopeDim / 2 : channel - kRopeDim / 2;
        const float qmate = __half2float(__ushort_as_half(q_in[row * kQHeads * kHeadDim +
                                                               (index % (kQHeads * kHeadDim)) / kHeadDim * kHeadDim + mate]));
        const int kv_head = (index / kHeadDim) % kKVHeads;
        const float kmate = __half2float(__ushort_as_half(k_in[row * kKVHeads * kHeadDim +
                                                               kv_head * kHeadDim + mate]));
        q = q * cosine + (channel < kRopeDim / 2 ? -qmate : qmate) * sine;
        k = k * cosine + (channel < kRopeDim / 2 ? -kmate : kmate) * sine;
    }
    q_out[index] = __half_as_ushort(__float2half_rn(q));
    if (channel < kHeadDim) {
        // One KV row is shared by six query heads; only the first four head blocks
        // of this grid write K, so this branch is overridden by the dedicated K launch below.
    }
}

template<bool Mrope=false>
__global__ void rope_k_kernel(const std::uint16_t* k_in,
                              std::uint16_t* k_out,
                              int rows,
                              int position,
                              const int* position_device,const int* positions_xyz,int offset) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kKVHeads * kHeadDim;
    if (index >= total) return;
    const int channel = index % kHeadDim;
    const int row = index / (kKVHeads * kHeadDim);
    const int base_position = position_device ? *position_device : position;
    int pos = base_position + row;
    if constexpr(Mrope)pos=positions_xyz?positions_xyz[row*3+(channel%(kRopeDim/2))%3]:pos+offset;
    float value = __half2float(__ushort_as_half(k_in[index]));
    if (channel < kRopeDim) {
        const int pair = channel < kRopeDim / 2 ? channel : channel - kRopeDim / 2;
        const float angle = static_cast<float>(pos) *
            powf(kRopeTheta, -2.0f * static_cast<float>(pair) / static_cast<float>(kRopeDim));
        const float sine = sinf(angle);
        const float cosine = cosf(angle);
        const int mate = channel < kRopeDim / 2 ? channel + kRopeDim / 2 : channel - kRopeDim / 2;
        const float mate_value = __half2float(__ushort_as_half(
            k_in[row * kKVHeads * kHeadDim + (index % (kKVHeads * kHeadDim)) / kHeadDim * kHeadDim + mate]));
        value = value * cosine + (channel < kRopeDim / 2 ? -mate_value : mate_value) * sine;
    }
    k_out[index] = __half_as_ushort(__float2half_rn(value));
}

__global__ void attention_kernel(const std::uint16_t* q,
                                 const std::uint16_t* k,
                                 const std::uint16_t* v,
                                 std::uint16_t* output,
                                 int rows) {
    __shared__ float scores[16];
    const int query = static_cast<int>(blockIdx.x) / kQHeads;
    const int head = static_cast<int>(blockIdx.x) % kQHeads;
    if (query >= rows) return;
    if (threadIdx.x == 0) {
        const int kv_head = head / (kQHeads / kKVHeads);
        float maximum = -3.402823466e+38F;
        for (int key = 0; key <= query; ++key) {
            float dot = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                const float qv = __half2float(__ushort_as_half(q[(query * kQHeads + head) * kHeadDim + d]));
                const float kv = __half2float(__ushort_as_half(k[(key * kKVHeads + kv_head) * kHeadDim + d]));
                dot += qv * kv;
            }
            scores[key] = dot * 0.0625f;
            maximum = fmaxf(maximum, scores[key]);
        }
        float denom = 0.0f;
        for (int key = 0; key <= query; ++key) {
            scores[key] = expf(scores[key] - maximum);
            denom += scores[key];
        }
        for (int d = 0; d < kHeadDim; ++d) {
            float value = 0.0f;
            for (int key = 0; key <= query; ++key) {
                value += scores[key] / denom * __half2float(__ushort_as_half(
                    v[(key * kKVHeads + kv_head) * kHeadDim + d]));
            }
            output[(query * kQHeads + head) * kHeadDim + d] =
                __half_as_ushort(__float2half_rn(value));
        }
    }
}

__global__ void append_kv_cache_kernel(const std::uint16_t* k,
                                       const std::uint16_t* v,
                                       std::uint16_t* k_cache,
                                       std::uint16_t* v_cache,
                                       int rows,
                                       int position,
                                       int capacity,
                                       const int* position_device) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kKVHeads * kHeadDim;
    if (index >= total) return;
    const int row = index / (kKVHeads * kHeadDim);
    const int base_position = position_device ? *position_device : position;
    const int slot = base_position + row;
    if (slot < 0 || slot >= capacity) return;
    const int offset = index % (kKVHeads * kHeadDim);
    const std::size_t destination = static_cast<std::size_t>(slot) * kKVHeads * kHeadDim + offset;
    k_cache[destination] = k[index];
    v_cache[destination] = v[index];
}

__global__ void attention_cached_kernel(const std::uint16_t* q,
                                        const std::uint16_t* k_cache,
                                        const std::uint16_t* v_cache,
                                        std::uint16_t* output,
                                        int rows,
                                        int position,
                                        int capacity,
                                        const int* position_device) {
    extern __shared__ float scores[];
    const int query = static_cast<int>(blockIdx.x) / kQHeads;
    const int head = static_cast<int>(blockIdx.x) % kQHeads;
    if (query >= rows || threadIdx.x != 0) return;
    const int base_position = position_device ? *position_device : position;
    const int key_count = base_position + query + 1;
    if (key_count < 0 || key_count > capacity) return;
    const int kv_head = head / (kQHeads / kKVHeads);
    float maximum = -3.402823466e+38F;
    for (int key = 0; key < key_count; ++key) {
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            const float qv = __half2float(__ushort_as_half(q[(query * kQHeads + head) * kHeadDim + d]));
            const float kv = __half2float(__ushort_as_half(
                k_cache[(static_cast<std::size_t>(key) * kKVHeads + kv_head) * kHeadDim + d]));
            dot += qv * kv;
        }
        scores[key] = dot * 0.0625f;
        maximum = fmaxf(maximum, scores[key]);
    }
    float denominator = 0.0f;
    for (int key = 0; key < key_count; ++key) {
        scores[key] = expf(scores[key] - maximum);
        denominator += scores[key];
    }
    for (int d = 0; d < kHeadDim; ++d) {
        float value = 0.0f;
        for (int key = 0; key < key_count; ++key) {
            value += scores[key] / denominator * __half2float(__ushort_as_half(
                v_cache[(static_cast<std::size_t>(key) * kKVHeads + kv_head) * kHeadDim + d]));
        }
        output[(query * kQHeads + head) * kHeadDim + d] = __half_as_ushort(__float2half_rn(value));
    }
}

// Same logical FP32 sum order as attention_cached_kernel. Parallelize only
// independent key scores and independent output dimensions. A context-owned
// global score plane replaces context-length dynamic shared memory.
template<bool segmented=false>
__global__ void attention_cached_parallel_kernel(const std::uint16_t* q,
    const std::uint16_t* k_cache,const std::uint16_t* v_cache,std::uint16_t* output,
    float* workspace,int rows,int position,int capacity,const int* position_device,int query_offset,
    const std::uint16_t* prefix_k=nullptr,const std::uint16_t* prefix_v=nullptr,int prefix_rows=0,int prefix_first=0,
    Exl3AttentionPageRanges page_ranges={}) {
    const int query=static_cast<int>(blockIdx.x)/kQHeads;
    const int head=static_cast<int>(blockIdx.x)%kQHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=Exl3AttentionCausalRows{base,rows,capacity}.count(query);
    if(!count) return;
    const int kv_head=head/(kQHeads/kKVHeads);
    float* scores=workspace+(static_cast<std::size_t>(query)*kQHeads+head)*capacity;
    __shared__ float maximum,denominator;
    for(int key=tid;key<count;key+=blockDim.x) {
        float dot=0.0f;
        const auto* key_plane=k_cache;
        int key_row=key;
        if constexpr(segmented) if(key>=prefix_first && key-prefix_first<prefix_rows) {
            key_plane=prefix_k;key_row=key-prefix_first;
        }
        if constexpr(segmented) for(int i=0;i<page_ranges.count;++i) {
            const auto r=page_ranges.ranges[i];
            if(key>=r.first && key-r.first<r.rows){key_plane=r.k;key_row=key-r.first;break;}
        }
        for(int d=0;d<kHeadDim;++d) {
            const float qv=__half2float(__ushort_as_half(q[(query*kQHeads+head)*kHeadDim+d]));
            const float kv=__half2float(__ushort_as_half(key_plane[(static_cast<std::size_t>(key_row)*kKVHeads+kv_head)*kHeadDim+d]));
            dot+=qv*kv;
        }
        scores[key]=dot*0.0625f;
    }
    __syncthreads();
    if(tid==0) {
        float value=-3.402823466e+38F;
        for(int key=0;key<count;++key) value=fmaxf(value,scores[key]);
        maximum=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) scores[key]=expf(scores[key]-maximum);
    __syncthreads();
    if(tid==0) {
        float value=0.0f;
        for(int key=0;key<count;++key) value+=scores[key];
        denominator=value;
    }
    __syncthreads();
    // Reuse the identical FP32 division for every value dimension. Materialized
    // FP32 weights introduce no additional cast relative to the original loop.
    for(int key=tid;key<count;key+=blockDim.x) scores[key]=scores[key]/denominator;
    __syncthreads();
    if(tid<kHeadDim) {
        float value=0.0f;
        for(int key=0;key<count;++key) {
            const auto* value_plane=v_cache;
            int value_row=key;
            if constexpr(segmented) if(key>=prefix_first && key-prefix_first<prefix_rows) {
                value_plane=prefix_v;value_row=key-prefix_first;
            }
            if constexpr(segmented) for(int i=0;i<page_ranges.count;++i) {
                const auto r=page_ranges.ranges[i];
                if(key>=r.first && key-r.first<r.rows){value_plane=r.v;value_row=key-r.first;break;}
            }
            value+=scores[key]*__half2float(__ushort_as_half(value_plane[(static_cast<std::size_t>(value_row)*kKVHeads+kv_head)*kHeadDim+tid]));
        }
        output[(query*kQHeads+head)*kHeadDim+tid]=__half_as_ushort(__float2half_rn(value));
    }
}

void cublas_check_attention(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}

__global__ void attention_scores_to_half_kernel(const float* scores,
                                                std::uint16_t* probabilities,
                                                int count,int capacity) {
    const int index=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    const int total=kQHeads*count;
    if(index>=total)return;
    const int head=index/count;
    const int key=index%count;
    probabilities[static_cast<std::size_t>(head)*capacity+key]=
        __half_as_ushort(__float2half_rn(
            scores[static_cast<std::size_t>(head)*capacity+key]));
}

// Bit-preserving variant: the represented FP16 query is converted once per
// dimension and block. Each key still performs the same d=0..255 FP32 FMA
// sequence, and softmax/value accumulation order is unchanged.
template<bool k_half2,bool v_half2=false,bool segmented=false>
__global__ void attention_cached_parallel_q_shared_kernel(const std::uint16_t* q,
    const std::uint16_t* k_cache,const std::uint16_t* v_cache,std::uint16_t* output,
    float* workspace,int rows,int position,int capacity,const int* position_device,int query_offset,
    Exl3AttentionPageRanges pages={}) {
    static_assert(!segmented || (!k_half2 && !v_half2));
    const int query=static_cast<int>(blockIdx.x)/kQHeads;
    const int head=static_cast<int>(blockIdx.x)%kQHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=Exl3AttentionCausalRows{base,rows,capacity}.count(query);
    if(!count) return;
    const int kv_head=head/(kQHeads/kKVHeads);
    __shared__ float query_values[kHeadDim];
    __shared__ float maximum,denominator;
    if(tid<kHeadDim)
        query_values[tid]=__half2float(__ushort_as_half(q[(query*kQHeads+head)*kHeadDim+tid]));
    __syncthreads();
    float* scores=workspace+(static_cast<std::size_t>(query)*kQHeads+head)*capacity;
    for(int key=tid;key<count;key+=blockDim.x) {
        float dot=0.0f;
        const auto* key_values=k_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
        if constexpr(segmented) for(int i=0;i<pages.count;++i) {
            const auto r=pages.ranges[i];
            if(key>=r.first && key-r.first<r.rows) {
                key_values=r.k+(static_cast<std::size_t>(key-r.first)*kKVHeads+kv_head)*kHeadDim;break;
            }
        }
        if constexpr(k_half2) for(int d=0;d<kHeadDim;d+=2) {
            const float2 kv=__half22float2(*reinterpret_cast<const __half2*>(key_values+d));
            dot+=query_values[d]*kv.x;
            dot+=query_values[d+1]*kv.y;
        } else for(int d=0;d<kHeadDim;++d)
            dot+=query_values[d]*__half2float(__ushort_as_half(key_values[d]));
        scores[key]=dot*0.0625f;
    }
    __syncthreads();
    if(tid==0) {
        float value=-3.402823466e+38F;
        for(int key=0;key<count;++key) value=fmaxf(value,scores[key]);
        maximum=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) scores[key]=expf(scores[key]-maximum);
    __syncthreads();
    if(tid==0) {
        float value=0.0f;
        for(int key=0;key<count;++key) value+=scores[key];
        denominator=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) scores[key]=scores[key]/denominator;
    __syncthreads();
    if constexpr(v_half2) {
      if(tid<kHeadDim/2) {
        float value0=0.0f,value1=0.0f;
        const int d=tid*2;
        for(int key=0;key<count;++key) {
            const float weight=scores[key];
            const auto* values=v_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            const float2 pair=__half22float2(*reinterpret_cast<const __half2*>(values));
            value0+=weight*pair.x;
            value1+=weight*pair.y;
        }
        auto* destination=output+(query*kQHeads+head)*kHeadDim+d;
        destination[0]=__half_as_ushort(__float2half_rn(value0));
        destination[1]=__half_as_ushort(__float2half_rn(value1));
      }
    } else if(tid<kHeadDim) {
        float value=0.0f;
        for(int key=0;key<count;++key) {
            const auto* values=v_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
            if constexpr(segmented) for(int i=0;i<pages.count;++i) {
                const auto r=pages.ranges[i];
                if(key>=r.first && key-r.first<r.rows) {
                    values=r.v+(static_cast<std::size_t>(key-r.first)*kKVHeads+kv_head)*kHeadDim;break;
                }
            }
            value+=scores[key]*__half2float(__ushort_as_half(values[tid]));
        }
        output[(query*kQHeads+head)*kHeadDim+tid]=__half_as_ushort(__float2half_rn(value));
    }
}

// Two chronological verifier rows share represented history loads. Each row
// keeps its own dot/softmax/value chain and never reads its future score slot.
__global__ void attention_query_pair_kernel(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,std::uint16_t* output,float* scores,int rows,int position,int capacity,
    const int* position_device,int query_offset,Exl3AttentionPageRanges pages) {
    const int first=(blockIdx.x/kQHeads)*2,head=blockIdx.x%kQHeads,tid=threadIdx.x;
    const int base=(position_device?*position_device:position)+query_offset;
    const Exl3AttentionCausalRows mask{base,rows,capacity};
    const int count0=mask.count(first),count1=mask.count(first+1);
    if(!count0)return;
    const int count=count1?count1:count0,kv_head=head/(kQHeads/kKVHeads);
    __shared__ float query[2][kHeadDim],maximum[2],denominator[2];
    if(tid<kHeadDim) {
        query[0][tid]=__half2float(__ushort_as_half(q[(first*kQHeads+head)*kHeadDim+tid]));
        query[1][tid]=count1?__half2float(__ushort_as_half(q[((first+1)*kQHeads+head)*kHeadDim+tid])):0.0f;
    }
    __syncthreads();
    float* score0=scores+(static_cast<std::size_t>(first)*kQHeads+head)*capacity;
    float* score1=count1?scores+(static_cast<std::size_t>(first+1)*kQHeads+head)*capacity:nullptr;
    for(int key=tid;key<count;key+=blockDim.x) {
        const auto* values=k+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
        for(int i=0;i<pages.count;++i) {
            const auto r=pages.ranges[i];
            if(key>=r.first && key-r.first<r.rows){values=r.k+(static_cast<std::size_t>(key-r.first)*kKVHeads+kv_head)*kHeadDim;break;}
        }
        float dot0=0.0f,dot1=0.0f;
        for(int d=0;d<kHeadDim;++d) {
            const float value=__half2float(__ushort_as_half(values[d]));
            if(key<count0)dot0+=query[0][d]*value;
            if(count1)dot1+=query[1][d]*value;
        }
        if(key<count0)score0[key]=dot0*0.0625f;
        if(count1)score1[key]=dot1*0.0625f;
    }
    __syncthreads();
    if(tid<2) {
        const int end=tid?count1:count0;const auto* s=tid?score1:score0;
        float value=-3.402823466e+38F;
        for(int key=0;key<end;++key)value=fmaxf(value,s[key]);
        maximum[tid]=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) {
        if(key<count0)score0[key]=expf(score0[key]-maximum[0]);
        if(count1)score1[key]=expf(score1[key]-maximum[1]);
    }
    __syncthreads();
    if(tid<2) {
        const int end=tid?count1:count0;const auto* s=tid?score1:score0;float value=0.0f;
        for(int key=0;key<end;++key)value+=s[key];denominator[tid]=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) {
        if(key<count0)score0[key]/=denominator[0];
        if(count1)score1[key]/=denominator[1];
    }
    __syncthreads();
    if(tid<kHeadDim) {
        float value0=0.0f,value1=0.0f;
        for(int key=0;key<count;++key) {
            const auto* values=v+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
            for(int i=0;i<pages.count;++i) {
                const auto r=pages.ranges[i];
                if(key>=r.first && key-r.first<r.rows){values=r.v+(static_cast<std::size_t>(key-r.first)*kKVHeads+kv_head)*kHeadDim;break;}
            }
            const float value=__half2float(__ushort_as_half(values[tid]));
            if(key<count0)value0+=score0[key]*value;
            if(count1)value1+=score1[key]*value;
        }
        output[(first*kQHeads+head)*kHeadDim+tid]=__half_as_ushort(__float2half_rn(value0));
        if(count1)output[((first+1)*kQHeads+head)*kHeadDim+tid]=__half_as_ushort(__float2half_rn(value1));
    }
}

// GQA24:4 gives six query heads the same KV head. Pair two query heads, but
// retain the original score-grid concurrency by assigning alternating 256-key
// tiles to two blocks. Each head keeps its own d=0..255 FP32 dependency chain;
// only the represented FP16 K load is shared.
template<bool segmented=false>
__global__ void attention_cached_gqa_pair_scores_kernel(const std::uint16_t* q,
    const std::uint16_t* k_cache,float* workspace,int rows,int position,
    int capacity,const int* position_device,int query_offset,Exl3AttentionPageRanges pages={}) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kPairsPerKV=kHeadsPerKV/2;
    constexpr int kScoreShards=2;
    constexpr int kBlocksPerQuery=kKVHeads*kPairsPerKV*kScoreShards;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/(kPairsPerKV*kScoreShards);
    const int pair=(local/kScoreShards)%kPairsPerKV;
    const int shard=local%kScoreShards;
    const int head0=kv_head*kHeadsPerKV+pair*2;
    const int head1=head0+1;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=Exl3AttentionCausalRows{base,rows,capacity}.count(query);
    if(!count) return;
    __shared__ float query0[kHeadDim];
    __shared__ float query1[kHeadDim];
    if(tid<kHeadDim) {
        query0[tid]=__half2float(__ushort_as_half(q[(query*kQHeads+head0)*kHeadDim+tid]));
        query1[tid]=__half2float(__ushort_as_half(q[(query*kQHeads+head1)*kHeadDim+tid]));
    }
    __syncthreads();
    float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    for(int key=shard*blockDim.x+tid;key<count;key+=kScoreShards*blockDim.x) {
        float dot0=0.0f,dot1=0.0f;
        const auto* key_values=k_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
        if constexpr(segmented) for(int i=0;i<pages.count;++i) {
            const auto r=pages.ranges[i];
            if(key>=r.first && key-r.first<r.rows) {
                key_values=r.k+(static_cast<std::size_t>(key-r.first)*kKVHeads+kv_head)*kHeadDim;break;
            }
        }
        for(int d=0;d<kHeadDim;d+=2) {
            const float2 kv=__half22float2(*reinterpret_cast<const __half2*>(key_values+d));
            dot0+=query0[d]*kv.x;
            dot0+=query0[d+1]*kv.y;
            dot1+=query1[d]*kv.x;
            dot1+=query1[d+1]*kv.y;
        }
        scores0[key]=dot0*0.0625f;
        scores1[key]=dot1*0.0625f;
    }
}

// Stream ordering guarantees that both score shards are complete. Softmax and
// value accumulation preserve the former chronological order independently for
// each head, while one half2 V load feeds both heads.
template<bool segmented=false>
__global__ void attention_cached_gqa_pair_values_kernel(const std::uint16_t* v_cache,
    std::uint16_t* output,float* workspace,int rows,int position,int capacity,
    const int* position_device,int query_offset,Exl3AttentionPageRanges pages={}) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kPairsPerKV=kHeadsPerKV/2;
    constexpr int kBlocksPerQuery=kKVHeads*kPairsPerKV;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/kPairsPerKV;
    const int pair=local%kPairsPerKV;
    const int head0=kv_head*kHeadsPerKV+pair*2;
    const int head1=head0+1;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=Exl3AttentionCausalRows{base,rows,capacity}.count(query);
    if(!count) return;
    float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    __shared__ float maximum[2],denominator[2];
    if(tid<2) {
        float value=-3.402823466e+38F;
        const float* scores=tid?scores1:scores0;
        for(int key=0;key<count;++key) value=fmaxf(value,scores[key]);
        maximum[tid]=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) {
        scores0[key]=expf(scores0[key]-maximum[0]);
        scores1[key]=expf(scores1[key]-maximum[1]);
    }
    __syncthreads();
    if(tid<2) {
        float value=0.0f;
        const float* scores=tid?scores1:scores0;
        for(int key=0;key<count;++key) value+=scores[key];
        denominator[tid]=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) {
        scores0[key]=scores0[key]/denominator[0];
        scores1[key]=scores1[key]/denominator[1];
    }
    __syncthreads();
    if(tid<kHeadDim/2) {
        float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
        const int d=tid*2;
        for(int key=0;key<count;++key) {
            const auto* values=v_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            if constexpr(segmented) for(int i=0;i<pages.count;++i) {
                const auto r=pages.ranges[i];
                if(key>=r.first && key-r.first<r.rows) {
                    values=r.v+(static_cast<std::size_t>(key-r.first)*kKVHeads+kv_head)*kHeadDim+d;break;
                }
            }
            const float2 represented=__half22float2(*reinterpret_cast<const __half2*>(values));
            value00+=scores0[key]*represented.x;
            value01+=scores0[key]*represented.y;
            value10+=scores1[key]*represented.x;
            value11+=scores1[key]*represented.y;
        }
        auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
        auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
        destination0[0]=__half_as_ushort(__float2half_rn(value00));
        destination0[1]=__half_as_ushort(__float2half_rn(value01));
        destination1[0]=__half_as_ushort(__float2half_rn(value10));
        destination1[1]=__half_as_ushort(__float2half_rn(value11));
    }
}

// Incremental GQA load reuse: three query heads share each represented K load.
// Three disjoint key shards keep the score launch at 24 blocks/query, identical
// to the promoted two-head schedule's aggregate concurrency.
__global__ void attention_cached_gqa_triple_scores_kernel(const std::uint16_t* q,
    const std::uint16_t* k_cache,float* workspace,int rows,int position,
    int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kGroupsPerKV=kHeadsPerKV/3;
    constexpr int kScoreShards=3;
    constexpr int kBlocksPerQuery=kKVHeads*kGroupsPerKV*kScoreShards;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/(kGroupsPerKV*kScoreShards);
    const int group=(local/kScoreShards)%kGroupsPerKV;
    const int shard=local%kScoreShards;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1 || count>capacity) return;
    __shared__ float query0[kHeadDim],query1[kHeadDim],query2[kHeadDim];
    if(tid<kHeadDim) {
        query0[tid]=__half2float(__ushort_as_half(q[(query*kQHeads+head0)*kHeadDim+tid]));
        query1[tid]=__half2float(__ushort_as_half(q[(query*kQHeads+head1)*kHeadDim+tid]));
        query2[tid]=__half2float(__ushort_as_half(q[(query*kQHeads+head2)*kHeadDim+tid]));
    }
    __syncthreads();
    float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    float* scores2=workspace+(static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    for(int key=shard*blockDim.x+tid;key<count;key+=kScoreShards*blockDim.x) {
        float dot0=0.0f,dot1=0.0f,dot2=0.0f;
        const auto* key_values=k_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
        for(int d=0;d<kHeadDim;d+=2) {
            const float2 kv=__half22float2(*reinterpret_cast<const __half2*>(key_values+d));
            dot0+=query0[d]*kv.x;dot0+=query0[d+1]*kv.y;
            dot1+=query1[d]*kv.x;dot1+=query1[d+1]*kv.y;
            dot2+=query2[d]*kv.x;dot2+=query2[d+1]*kv.y;
        }
        scores0[key]=dot0*0.0625f;
        scores1[key]=dot1*0.0625f;
        scores2[key]=dot2*0.0625f;
    }
}

template <bool StagedSoftmax,bool PairDimensions=false>
__global__ void attention_cached_gqa_triple_values_kernel(const std::uint16_t* v_cache,
    std::uint16_t* output,float* workspace,int rows,int position,int capacity,
    const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kGroupsPerKV=kHeadsPerKV/3;
    constexpr int kBlocksPerQuery=kKVHeads*kGroupsPerKV;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/kGroupsPerKV;
    const int group=local%kGroupsPerKV;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1 || count>capacity) return;
    float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    float* scores2=workspace+(static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    __shared__ float maximum[3],denominator[3];
    __shared__ float score_tile[StagedSoftmax?3*256:1];
    if constexpr(StagedSoftmax) {
        if(tid<3) maximum[tid]=-3.402823466e+38F;
        __syncthreads();
        for(int first=0;first<count;first+=blockDim.x) {
            const int key=first+tid;
            if(key<count) {score_tile[tid]=scores0[key];score_tile[256+tid]=scores1[key];score_tile[512+tid]=scores2[key];}
            __syncthreads();
            if(tid<3) {float value=maximum[tid];const int extent=min(blockDim.x,count-first);const float* tile=score_tile+tid*256;for(int offset=0;offset<extent;++offset)value=fmaxf(value,tile[offset]);maximum[tid]=value;}
            __syncthreads();
        }
    } else if(tid<3) {
        float value=-3.402823466e+38F;
        const float* scores=tid==0?scores0:(tid==1?scores1:scores2);
        for(int key=0;key<count;++key) value=fmaxf(value,scores[key]);
        maximum[tid]=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) {
        scores0[key]=expf(scores0[key]-maximum[0]);
        scores1[key]=expf(scores1[key]-maximum[1]);
        scores2[key]=expf(scores2[key]-maximum[2]);
    }
    __syncthreads();
    if constexpr(StagedSoftmax) {
        if(tid<3) denominator[tid]=0.0f;
        __syncthreads();
        for(int first=0;first<count;first+=blockDim.x) {
            const int key=first+tid;
            if(key<count) {score_tile[tid]=scores0[key];score_tile[256+tid]=scores1[key];score_tile[512+tid]=scores2[key];}
            __syncthreads();
            if(tid<3) {float value=denominator[tid];const int extent=min(blockDim.x,count-first);const float* tile=score_tile+tid*256;for(int offset=0;offset<extent;++offset)value+=tile[offset];denominator[tid]=value;}
            __syncthreads();
        }
    } else if(tid<3) {
        float value=0.0f;
        const float* scores=tid==0?scores0:(tid==1?scores1:scores2);
        for(int key=0;key<count;++key) value+=scores[key];
        denominator[tid]=value;
    }
    __syncthreads();
    for(int key=tid;key<count;key+=blockDim.x) {
        scores0[key]=scores0[key]/denominator[0];
        scores1[key]=scores1[key]/denominator[1];
        scores2[key]=scores2[key]/denominator[2];
    }
    __syncthreads();
    if constexpr(PairDimensions) {
      if(tid<kHeadDim/4) {
        float value00=0.0f,value01=0.0f,value02=0.0f,value03=0.0f;
        float value10=0.0f,value11=0.0f,value12=0.0f,value13=0.0f;
        float value20=0.0f,value21=0.0f,value22=0.0f,value23=0.0f;
        const int d=tid*4;
        for(int key=0;key<count;++key) {
            const auto* values=v_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            const float2 represented0=__half22float2(
                *reinterpret_cast<const __half2*>(values));
            const float2 represented1=__half22float2(
                *reinterpret_cast<const __half2*>(values+2));
            const float weight0=scores0[key],weight1=scores1[key],weight2=scores2[key];
            value00+=weight0*represented0.x;value01+=weight0*represented0.y;
            value02+=weight0*represented1.x;value03+=weight0*represented1.y;
            value10+=weight1*represented0.x;value11+=weight1*represented0.y;
            value12+=weight1*represented1.x;value13+=weight1*represented1.y;
            value20+=weight2*represented0.x;value21+=weight2*represented0.y;
            value22+=weight2*represented1.x;value23+=weight2*represented1.y;
        }
        auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
        auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
        auto* destination2=output+(query*kQHeads+head2)*kHeadDim+d;
        destination0[0]=__half_as_ushort(__float2half_rn(value00));destination0[1]=__half_as_ushort(__float2half_rn(value01));
        destination0[2]=__half_as_ushort(__float2half_rn(value02));destination0[3]=__half_as_ushort(__float2half_rn(value03));
        destination1[0]=__half_as_ushort(__float2half_rn(value10));destination1[1]=__half_as_ushort(__float2half_rn(value11));
        destination1[2]=__half_as_ushort(__float2half_rn(value12));destination1[3]=__half_as_ushort(__float2half_rn(value13));
        destination2[0]=__half_as_ushort(__float2half_rn(value20));destination2[1]=__half_as_ushort(__float2half_rn(value21));
        destination2[2]=__half_as_ushort(__float2half_rn(value22));destination2[3]=__half_as_ushort(__float2half_rn(value23));
      }
    } else if(tid<kHeadDim/2) {
        float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f,value20=0.0f,value21=0.0f;
        const int d=tid*2;
        for(int key=0;key<count;++key) {
            const auto* values=v_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            const float2 represented=__half22float2(*reinterpret_cast<const __half2*>(values));
            value00+=scores0[key]*represented.x;value01+=scores0[key]*represented.y;
            value10+=scores1[key]*represented.x;value11+=scores1[key]*represented.y;
            value20+=scores2[key]*represented.x;value21+=scores2[key]*represented.y;
        }
        auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
        auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
        auto* destination2=output+(query*kQHeads+head2)*kHeadDim+d;
        destination0[0]=__half_as_ushort(__float2half_rn(value00));destination0[1]=__half_as_ushort(__float2half_rn(value01));
        destination1[0]=__half_as_ushort(__float2half_rn(value10));destination1[1]=__half_as_ushort(__float2half_rn(value11));
        destination2[0]=__half_as_ushort(__float2half_rn(value20));destination2[1]=__half_as_ushort(__float2half_rn(value21));
    }
}

// One CTA owns both independent three-head groups of a KV head. All six
// softmax streams retain the staged chronological maximum/sum sequence. The
// value phase uses two disjoint 128-thread groups, so each thread retains the
// qualified six accumulators rather than the regressive full-six kernel's
// twelve; the two groups intentionally issue their own represented V load.
__global__ void attention_cached_gqa_six_packed_triples_values_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    float* scores[kHeadsPerKV]{};
    #pragma unroll
    for(int head=0;head<kHeadsPerKV;++head)
        scores[head]=workspace+(static_cast<std::size_t>(query)*kQHeads+
            kv_head*kHeadsPerKV+head)*capacity;
    __shared__ float maximum[kHeadsPerKV],denominator[kHeadsPerKV];
    __shared__ float score_tile[kHeadsPerKV*256];
    if(tid<kHeadsPerKV) maximum[tid]=-3.402823466e+38F;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int head=0;head<kHeadsPerKV;++head)
                score_tile[head*256+tid]=scores[head][key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=maximum[tid];const int extent=min(blockDim.x,count-first);
            const float* tile=score_tile+tid*256;
            for(int offset=0;offset<extent;++offset)value=fmaxf(value,tile[offset]);
            maximum[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int head=0;head<kHeadsPerKV;++head)
            scores[head][key]=expf(scores[head][key]-maximum[head]);
    }
    __syncthreads();
    if(tid<kHeadsPerKV) denominator[tid]=0.0f;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int head=0;head<kHeadsPerKV;++head)
                score_tile[head*256+tid]=scores[head][key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=denominator[tid];const int extent=min(blockDim.x,count-first);
            const float* tile=score_tile+tid*256;
            for(int offset=0;offset<extent;++offset)value+=tile[offset];
            denominator[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int head=0;head<kHeadsPerKV;++head)
            scores[head][key]=scores[head][key]/denominator[head];
    }
    __syncthreads();
    {
        const int group=tid/128,lane=tid%128,head0=kv_head*kHeadsPerKV+group*3;
        const int d=lane*2;
        float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
        float value20=0.0f,value21=0.0f;
        const float* scores0=scores[group*3];
        const float* scores1=scores[group*3+1];
        const float* scores2=scores[group*3+2];
        for(int key=0;key<count;++key) {
            const auto* values=v_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(values));
            value00+=scores0[key]*represented.x;value01+=scores0[key]*represented.y;
            value10+=scores1[key]*represented.x;value11+=scores1[key]*represented.y;
            value20+=scores2[key]*represented.x;value21+=scores2[key]*represented.y;
        }
        auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
        auto* destination1=destination0+kHeadDim;
        auto* destination2=destination1+kHeadDim;
        destination0[0]=__half_as_ushort(__float2half_rn(value00));
        destination0[1]=__half_as_ushort(__float2half_rn(value01));
        destination1[0]=__half_as_ushort(__float2half_rn(value10));
        destination1[1]=__half_as_ushort(__float2half_rn(value11));
        destination2[0]=__half_as_ushort(__float2half_rn(value20));
        destination2[1]=__half_as_ushort(__float2half_rn(value21));
    }
}

// Scores have already been normalized once for all six heads by the exact
// six-head softmax kernel. The default retains the qualified three-head/two-value
// schedule. The default-off paired-dimension specialization shares each three-score
// tuple across two adjacent half2 values while keeping twelve independent,
// chronological FP32 accumulator chains.
template<bool FuseGate>
__device__ __forceinline__ std::uint16_t attention_value_output_bits(
    float value,const std::uint16_t* gate,std::size_t index) {
    const auto represented=__float2half_rn(value);
    if constexpr(!FuseGate) return __half_as_ushort(represented);
    const float x=__half2float(represented);
    const float g=__half2float(__ushort_as_half(gate[index]));
    return __half_as_ushort(__float2half_rn(x/(1.0f+expf(-g))));
}

template<bool segmented=false,bool PairDimensions=false,bool WarpScoreBroadcast=false,
         bool KeyPairPipeline=false,bool FuseGate=false,bool DeferredNormalize=false>
__global__ void attention_cached_gqa_triple_normalized_values_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset,
    Exl3AttentionPageRanges pages={},const std::uint16_t* gate=nullptr,
    const float* deferred_denominators=nullptr) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kGroupsPerKV=kHeadsPerKV/3;
    constexpr int kBlocksPerQuery=kKVHeads*kGroupsPerKV;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/kGroupsPerKV;
    const int group=local%kGroupsPerKV;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const int tid=threadIdx.x;
    if(query>=rows||tid>=(PairDimensions?kHeadDim/4:kHeadDim/2)) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    const float* scores2=workspace+(static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
    float value20=0.0f,value21=0.0f;
    float value02=0.0f,value03=0.0f,value12=0.0f,value13=0.0f;
    float value22=0.0f,value23=0.0f;
    const int d=tid*(PairDimensions?4:2);
    int key=0;
    if constexpr(KeyPairPipeline) for(;key+1<count;key+=2) {
        const auto* values0=v_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        const auto* values1=values0+static_cast<std::size_t>(kKVHeads)*kHeadDim;
        const float2 represented0=__half22float2(
            *reinterpret_cast<const __half2*>(values0));
        const float2 represented1=__half22float2(
            *reinterpret_cast<const __half2*>(values1));
        const float weight00=scores0[key],weight10=scores1[key],weight20=scores2[key];
        const float weight01=scores0[key+1],weight11=scores1[key+1],weight21=scores2[key+1];
        value00+=weight00*represented0.x;value01+=weight00*represented0.y;
        value10+=weight10*represented0.x;value11+=weight10*represented0.y;
        value20+=weight20*represented0.x;value21+=weight20*represented0.y;
        value00+=weight01*represented1.x;value01+=weight01*represented1.y;
        value10+=weight11*represented1.x;value11+=weight11*represented1.y;
        value20+=weight21*represented1.x;value21+=weight21*represented1.y;
    }
    for(;key<count;++key) {
        const auto* values=v_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        if constexpr(segmented) for(int i=0;i<pages.count;++i) {
            const auto range=pages.ranges[i];
            if(key>=range.first && key-range.first<range.rows) {
                values=range.v+
                    (static_cast<std::size_t>(key-range.first)*kKVHeads+
                     kv_head)*kHeadDim+d;
                break;
            }
        }
        if constexpr(PairDimensions) {
            const float2 represented0=__half22float2(
                *reinterpret_cast<const __half2*>(values));
            const float2 represented1=__half22float2(
                *reinterpret_cast<const __half2*>(values+2));
            const float weight0=scores0[key],weight1=scores1[key],weight2=scores2[key];
            value00+=weight0*represented0.x;value01+=weight0*represented0.y;
            value02+=weight0*represented1.x;value03+=weight0*represented1.y;
            value10+=weight1*represented0.x;value11+=weight1*represented0.y;
            value12+=weight1*represented1.x;value13+=weight1*represented1.y;
            value20+=weight2*represented0.x;value21+=weight2*represented0.y;
            value22+=weight2*represented1.x;value23+=weight2*represented1.y;
        } else if constexpr(WarpScoreBroadcast) {
            const unsigned mask=__activemask();
            const int lane=tid&31;
            const float weight0=__shfl_sync(
                mask,lane==0?scores0[key]:0.0f,0);
            const float weight1=__shfl_sync(
                mask,lane==0?scores1[key]:0.0f,0);
            const float weight2=__shfl_sync(
                mask,lane==0?scores2[key]:0.0f,0);
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(values));
            value00+=weight0*represented.x;value01+=weight0*represented.y;
            value10+=weight1*represented.x;value11+=weight1*represented.y;
            value20+=weight2*represented.x;value21+=weight2*represented.y;
        } else {
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(values));
            float weight0=scores0[key],weight1=scores1[key],weight2=scores2[key];
            if constexpr(DeferredNormalize) {
                weight0=weight0/deferred_denominators[query*kQHeads+head0];
                weight1=weight1/deferred_denominators[query*kQHeads+head1];
                weight2=weight2/deferred_denominators[query*kQHeads+head2];
            }
            value00+=weight0*represented.x;value01+=weight0*represented.y;
            value10+=weight1*represented.x;value11+=weight1*represented.y;
            value20+=weight2*represented.x;value21+=weight2*represented.y;
        }
    }
    auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
    auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
    auto* destination2=output+(query*kQHeads+head2)*kHeadDim+d;
    const auto index0=static_cast<std::size_t>(destination0-output);
    const auto index1=static_cast<std::size_t>(destination1-output);
    const auto index2=static_cast<std::size_t>(destination2-output);
    destination0[0]=attention_value_output_bits<FuseGate>(value00,gate,index0);
    destination0[1]=attention_value_output_bits<FuseGate>(value01,gate,index0+1);
    destination1[0]=attention_value_output_bits<FuseGate>(value10,gate,index1);
    destination1[1]=attention_value_output_bits<FuseGate>(value11,gate,index1+1);
    destination2[0]=attention_value_output_bits<FuseGate>(value20,gate,index2);
    destination2[1]=attention_value_output_bits<FuseGate>(value21,gate,index2+1);
    if constexpr(PairDimensions) {
        destination0[2]=attention_value_output_bits<FuseGate>(value02,gate,index0+2);
        destination0[3]=attention_value_output_bits<FuseGate>(value03,gate,index0+3);
        destination1[2]=attention_value_output_bits<FuseGate>(value12,gate,index1+2);
        destination1[3]=attention_value_output_bits<FuseGate>(value13,gate,index1+3);
        destination2[2]=attention_value_output_bits<FuseGate>(value22,gate,index2+2);
        destination2[3]=attention_value_output_bits<FuseGate>(value23,gate,index2+3);
    }
}

// Discriminator-only counterpart to the selected normalized-value consumer.
// Scores are laid out [KV head][three-head group][key][head-in-group] so one
// chronological tuple is adjacent. Arithmetic and represented V traversal are
// intentionally identical to the selected head-major kernel.
__global__ void attention_cached_gqa_triple_interleaved_values_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int count,int capacity) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kGroupsPerKV=kHeadsPerKV/3;
    constexpr int kBlocks=kKVHeads*kGroupsPerKV;
    const int local=static_cast<int>(blockIdx.x)%kBlocks;
    const int kv_head=local/kGroupsPerKV;
    const int group=local%kGroupsPerKV;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int tid=threadIdx.x;
    if(tid>=kHeadDim/2) return;
    const float* scores=workspace+
        (static_cast<std::size_t>(kv_head*kGroupsPerKV+group)*capacity)*3;
    float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
    float value20=0.0f,value21=0.0f;
    const int d=tid*2;
    for(int key=0;key<count;++key) {
        const auto* values=v_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        const float2 represented=__half22float2(
            *reinterpret_cast<const __half2*>(values));
        const auto* weights=scores+static_cast<std::size_t>(key)*3;
        value00+=weights[0]*represented.x;value01+=weights[0]*represented.y;
        value10+=weights[1]*represented.x;value11+=weights[1]*represented.y;
        value20+=weights[2]*represented.x;value21+=weights[2]*represented.y;
    }
    auto* destination0=output+head0*kHeadDim+d;
    auto* destination1=destination0+kHeadDim;
    auto* destination2=destination1+kHeadDim;
    destination0[0]=__half_as_ushort(__float2half_rn(value00));
    destination0[1]=__half_as_ushort(__float2half_rn(value01));
    destination1[0]=__half_as_ushort(__float2half_rn(value10));
    destination1[1]=__half_as_ushort(__float2half_rn(value11));
    destination2[0]=__half_as_ushort(__float2half_rn(value20));
    destination2[1]=__half_as_ushort(__float2half_rn(value21));
}

__global__ void attention_triple_layout_discriminator_init_kernel(
    std::uint16_t* values,float* head_major,float* interleaved) {
    constexpr int capacity=4096;
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kGroupsPerKV=kHeadsPerKV/3;
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    const std::size_t value_count=
        static_cast<std::size_t>(capacity)*kKVHeads*kHeadDim;
    if(index<value_count) {
        const int signed_value=static_cast<int>((index*17u+11u)%31u)-15;
        values[index]=__half_as_ushort(__float2half_rn(signed_value*0.001f));
    }
    const std::size_t score_count=static_cast<std::size_t>(kQHeads)*capacity;
    if(index<score_count) {
        const int head=static_cast<int>(index/capacity);
        const int key=static_cast<int>(index%capacity);
        const float score=static_cast<float>((key*13+head*7)%97+1)*0.00001f;
        head_major[index]=score;
        const int kv_head=head/kHeadsPerKV;
        const int local_head=head%kHeadsPerKV;
        const int group=local_head/3;
        const int in_group=local_head%3;
        interleaved[(static_cast<std::size_t>(kv_head*kGroupsPerKV+group)*
            capacity+key)*3+in_group]=score;
    }
}

__global__ void attention_triple_layout_discriminator_compare_kernel(
    const std::uint16_t* reference,const std::uint16_t* candidate,
    std::size_t count,unsigned long long* mismatches) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count && reference[index]!=candidate[index])
        atomicAdd(mismatches,1ULL);
}

__global__ void attention_fusion_discriminator_compare_scores_kernel(
    const float* reference,const float* candidate,std::size_t count,
    unsigned long long* mismatches) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count && __float_as_uint(reference[index])!=
                      __float_as_uint(candidate[index]))
        atomicAdd(mismatches,1ULL);
}

__global__ void attention_deferred_normalization_compare_scores_kernel(
    const float* normalized,const float* exponentiated,const float* denominators,
    int capacity,std::size_t count,unsigned long long* mismatches) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count) {
        const int head=static_cast<int>(index/static_cast<std::size_t>(capacity));
        const float candidate=exponentiated[index]/denominators[head];
        if(__float_as_uint(normalized[index])!=__float_as_uint(candidate))
            atomicAdd(mismatches,1ULL);
    }
}

__global__ void attention_gate_discriminator_init_kernel(
    std::uint16_t* gate,std::size_t count) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count) {
        const int signed_value=static_cast<int>((index*29u+5u)%41u)-20;
        gate[index]=__half_as_ushort(__float2half_rn(signed_value*0.05f));
    }
}

// The selected value stage broadcasts the same three normalized scores to all
// 128 dimension lanes. Stage one chronological 128-key score tile per CTA so
// each score plane is read from global memory once per key instead of once per
// lane. Every lane still consumes keys in ascending order and keeps the same
// six independent FP32 accumulator chains and represented V half2 loads.
__global__ void attention_cached_gqa_triple_normalized_values_score_tile_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kGroupsPerKV=kHeadsPerKV/3;
    constexpr int kBlocksPerQuery=kKVHeads*kGroupsPerKV;
    constexpr int kTileKeys=128;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/kGroupsPerKV;
    const int group=local%kGroupsPerKV;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const int tid=threadIdx.x;
    if(query>=rows||tid>=kHeadDim/2) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const float* scores0=workspace+
        (static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=workspace+
        (static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    const float* scores2=workspace+
        (static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    __shared__ float score_tile[3][kTileKeys];
    float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
    float value20=0.0f,value21=0.0f;
    const int d=tid*2;
    for(int first=0;first<count;first+=kTileKeys) {
        const int key=first+tid;
        if(key<count) {
            score_tile[0][tid]=scores0[key];
            score_tile[1][tid]=scores1[key];
            score_tile[2][tid]=scores2[key];
        }
        __syncthreads();
        const int extent=min(kTileKeys,count-first);
        for(int offset=0;offset<extent;++offset) {
            const auto* values=v_cache+
                (static_cast<std::size_t>(first+offset)*kKVHeads+kv_head)*
                    kHeadDim+d;
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(values));
            value00+=score_tile[0][offset]*represented.x;
            value01+=score_tile[0][offset]*represented.y;
            value10+=score_tile[1][offset]*represented.x;
            value11+=score_tile[1][offset]*represented.y;
            value20+=score_tile[2][offset]*represented.x;
            value21+=score_tile[2][offset]*represented.y;
        }
        __syncthreads();
    }
    auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
    auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
    auto* destination2=output+(query*kQHeads+head2)*kHeadDim+d;
    destination0[0]=__half_as_ushort(__float2half_rn(value00));
    destination0[1]=__half_as_ushort(__float2half_rn(value01));
    destination1[0]=__half_as_ushort(__float2half_rn(value10));
    destination1[1]=__half_as_ushort(__float2half_rn(value11));
    destination2[0]=__half_as_ushort(__float2half_rn(value20));
    destination2[1]=__half_as_ushort(__float2half_rn(value21));
}

// The selected six-head softmax supplies six independent normalized score
// planes for every query/KV head.  One 128-thread CTA owns all six heads so
// each chronological represented V half2 is fetched once, then consumed by
// six independent FP32 accumulator chains.  Per-head arithmetic order and
// output conversion are identical to the qualified two-CTA triple route.
__global__ void attention_cached_gqa_six_normalized_values_single_load_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows||tid>=kHeadDim/2) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const int head0=kv_head*kHeadsPerKV;
    const float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=scores0+capacity;
    const float* scores2=scores1+capacity;
    const float* scores3=scores2+capacity;
    const float* scores4=scores3+capacity;
    const float* scores5=scores4+capacity;
    float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
    float value20=0.0f,value21=0.0f,value30=0.0f,value31=0.0f;
    float value40=0.0f,value41=0.0f,value50=0.0f,value51=0.0f;
    const int d=tid*2;
    for(int key=0;key<count;++key) {
        const auto* values=v_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        const float2 represented=__half22float2(
            *reinterpret_cast<const __half2*>(values));
        value00+=scores0[key]*represented.x;value01+=scores0[key]*represented.y;
        value10+=scores1[key]*represented.x;value11+=scores1[key]*represented.y;
        value20+=scores2[key]*represented.x;value21+=scores2[key]*represented.y;
        value30+=scores3[key]*represented.x;value31+=scores3[key]*represented.y;
        value40+=scores4[key]*represented.x;value41+=scores4[key]*represented.y;
        value50+=scores5[key]*represented.x;value51+=scores5[key]*represented.y;
    }
    const auto store=[&](int head,float x,float y) {
        auto* destination=output+(query*kQHeads+head)*kHeadDim+d;
        destination[0]=__half_as_ushort(__float2half_rn(x));
        destination[1]=__half_as_ushort(__float2half_rn(y));
    };
    store(head0,value00,value01);store(head0+1,value10,value11);
    store(head0+2,value20,value21);store(head0+3,value30,value31);
    store(head0+4,value40,value41);store(head0+5,value50,value51);
}

// The earlier six-head single-load route halved represented-V traffic but
// carried twelve FP32 accumulators in each half2 lane.  This bounded variant
// assigns one represented dimension to each of 256 lanes, retaining the same
// chronological per-head FP32 chains with six accumulators per lane.  Across
// the CTA each represented V scalar is loaded once instead of once per
// three-head group, while the selected two-CTA route and its evidence remain
// available unchanged.
__global__ void attention_cached_gqa_six_normalized_values_scalar_single_load_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int d=threadIdx.x;
    if(query>=rows||d>=kHeadDim) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const int head0=kv_head*kHeadsPerKV;
    const float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=scores0+capacity;
    const float* scores2=scores1+capacity;
    const float* scores3=scores2+capacity;
    const float* scores4=scores3+capacity;
    const float* scores5=scores4+capacity;
    float value0=0.0f,value1=0.0f,value2=0.0f;
    float value3=0.0f,value4=0.0f,value5=0.0f;
    for(int key=0;key<count;++key) {
        const float represented=__half2float(__ushort_as_half(v_cache[
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d]));
        value0+=scores0[key]*represented;value1+=scores1[key]*represented;
        value2+=scores2[key]*represented;value3+=scores3[key]*represented;
        value4+=scores4[key]*represented;value5+=scores5[key]*represented;
    }
    output[(query*kQHeads+head0)*kHeadDim+d]=__half_as_ushort(__float2half_rn(value0));
    output[(query*kQHeads+head0+1)*kHeadDim+d]=__half_as_ushort(__float2half_rn(value1));
    output[(query*kQHeads+head0+2)*kHeadDim+d]=__half_as_ushort(__float2half_rn(value2));
    output[(query*kQHeads+head0+3)*kHeadDim+d]=__half_as_ushort(__float2half_rn(value3));
    output[(query*kQHeads+head0+4)*kHeadDim+d]=__half_as_ushort(__float2half_rn(value4));
    output[(query*kQHeads+head0+5)*kHeadDim+d]=__half_as_ushort(__float2half_rn(value5));
}

// Pair adjacent causal queries inside one 128-thread CTA.  Both queries read
// the same represented V prefix, so each chronological V half2 load feeds the
// six accumulator chains for both rows.  The second row then consumes its one
// additional causal value.  Every output retains the baseline key order and
// FP32 accumulation sequence; only redundant V traffic and CTA count change.
__global__ void attention_cached_gqa_triple_normalized_values_two_query_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kGroupsPerKV=kHeadsPerKV/3;
    constexpr int kBlocksPerPair=kKVHeads*kGroupsPerKV;
    const int pair=static_cast<int>(blockIdx.x)/kBlocksPerPair;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerPair;
    const int query0=pair*2,query1=query0+1;
    const int kv_head=local/kGroupsPerKV;
    const int group=local%kGroupsPerKV;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const int tid=threadIdx.x;
    if(query0>=rows||tid>=kHeadDim/2) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count0=base+query0+1;
    if(count0<1||count0>capacity) return;
    const bool have_query1=query1<rows;
    const int d=tid*2;
    const auto score=[&](int query,int head) {
        return workspace+(static_cast<std::size_t>(query)*kQHeads+head)*capacity;
    };
    const float* scores00=score(query0,head0);
    const float* scores01=score(query0,head1);
    const float* scores02=score(query0,head2);
    const float* scores10=have_query1?score(query1,head0):nullptr;
    const float* scores11=have_query1?score(query1,head1):nullptr;
    const float* scores12=have_query1?score(query1,head2):nullptr;
    float value000=0.0f,value001=0.0f,value010=0.0f,value011=0.0f;
    float value020=0.0f,value021=0.0f;
    float value100=0.0f,value101=0.0f,value110=0.0f,value111=0.0f;
    float value120=0.0f,value121=0.0f;
    for(int key=0;key<count0;++key) {
        const auto* values=v_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        const float2 represented=__half22float2(
            *reinterpret_cast<const __half2*>(values));
        value000+=scores00[key]*represented.x;value001+=scores00[key]*represented.y;
        value010+=scores01[key]*represented.x;value011+=scores01[key]*represented.y;
        value020+=scores02[key]*represented.x;value021+=scores02[key]*represented.y;
        if(have_query1) {
            value100+=scores10[key]*represented.x;value101+=scores10[key]*represented.y;
            value110+=scores11[key]*represented.x;value111+=scores11[key]*represented.y;
            value120+=scores12[key]*represented.x;value121+=scores12[key]*represented.y;
        }
    }
    if(have_query1) {
        const auto* values=v_cache+
            (static_cast<std::size_t>(count0)*kKVHeads+kv_head)*kHeadDim+d;
        const float2 represented=__half22float2(
            *reinterpret_cast<const __half2*>(values));
        value100+=scores10[count0]*represented.x;value101+=scores10[count0]*represented.y;
        value110+=scores11[count0]*represented.x;value111+=scores11[count0]*represented.y;
        value120+=scores12[count0]*represented.x;value121+=scores12[count0]*represented.y;
    }
    const auto store=[&](int query,int head,float x,float y) {
        auto* destination=output+(query*kQHeads+head)*kHeadDim+d;
        destination[0]=__half_as_ushort(__float2half_rn(x));
        destination[1]=__half_as_ushort(__float2half_rn(y));
    };
    store(query0,head0,value000,value001);
    store(query0,head1,value010,value011);
    store(query0,head2,value020,value021);
    if(have_query1) {
        store(query1,head0,value100,value101);
        store(query1,head1,value110,value111);
        store(query1,head2,value120,value121);
    }
}

// Exact scalar-dimension topology for the selected three-head value stage.
// Each of the 256 threads owns one represented dimension, preserving the
// baseline key order and FP32 accumulation sequence for that dimension while
// halving the per-thread accumulator chains from six to three.
__global__ void attention_cached_gqa_triple_normalized_values_scalar_dim_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kGroupsPerKV=kHeadsPerKV/3;
    constexpr int kBlocksPerQuery=kKVHeads*kGroupsPerKV;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/kGroupsPerKV;
    const int group=local%kGroupsPerKV;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const int d=threadIdx.x;
    if(query>=rows||d>=kHeadDim) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    const float* scores2=workspace+(static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    float value0=0.0f,value1=0.0f,value2=0.0f;
    for(int key=0;key<count;++key) {
        const auto represented=__half2float(__ushort_as_half(v_cache[
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d]));
        value0+=scores0[key]*represented;
        value1+=scores1[key]*represented;
        value2+=scores2[key]*represented;
    }
    output[(query*kQHeads+head0)*kHeadDim+d]=
        __half_as_ushort(__float2half_rn(value0));
    output[(query*kQHeads+head1)*kHeadDim+d]=
        __half_as_ushort(__float2half_rn(value1));
    output[(query*kQHeads+head2)*kHeadDim+d]=
        __half_as_ushort(__float2half_rn(value2));
}

// Barrier-free topology alternative for the selected six-softmax value stage.
// One CTA carries both original 128-thread triple groups. Each group still
// issues its own represented V half2 load and retains its original three-head,
// six-accumulator chronological dependency chains; only the block topology is
// merged so all 256 launched threads perform useful work.
__global__ void attention_cached_gqa_triple_normalized_values_full_cta_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const int group=tid/128,lane=tid%128;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    const float* scores2=workspace+(static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
    float value20=0.0f,value21=0.0f;
    const int d=lane*2;
    for(int key=0;key<count;++key) {
        const auto* values=v_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        const float2 represented=__half22float2(
            *reinterpret_cast<const __half2*>(values));
        value00+=scores0[key]*represented.x;value01+=scores0[key]*represented.y;
        value10+=scores1[key]*represented.x;value11+=scores1[key]*represented.y;
        value20+=scores2[key]*represented.x;value21+=scores2[key]*represented.y;
    }
    auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
    auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
    auto* destination2=output+(query*kQHeads+head2)*kHeadDim+d;
    destination0[0]=__half_as_ushort(__float2half_rn(value00));
    destination0[1]=__half_as_ushort(__float2half_rn(value01));
    destination1[0]=__half_as_ushort(__float2half_rn(value10));
    destination1[1]=__half_as_ushort(__float2half_rn(value11));
    destination2[0]=__half_as_ushort(__float2half_rn(value20));
    destination2[1]=__half_as_ushort(__float2half_rn(value21));
}

// The selected six-softmax route has two independent three-head groups for
// every query/KV head. Stage a short chronological run of represented V half2
// values once, then let both 128-thread groups consume it. Each thread retains
// its original key order and six FP32 accumulator dependency chains.
template<int kTileKeys>
__global__ void attention_cached_gqa_triple_normalized_values_v_tile_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    static_assert(kTileKeys==8||kTileKeys==16||kTileKeys==64);
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const int group=tid/128,lane=tid%128;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    const float* scores2=workspace+(static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    extern __shared__ std::uint32_t staged_v[];
    float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
    float value20=0.0f,value21=0.0f;
    for(int first=0;first<count;first+=kTileKeys) {
        const int extent=min(kTileKeys,count-first);
        for(int item=tid;item<extent*128;item+=blockDim.x) {
            const int key=first+item/128;
            const int d=(item%128)*2;
            const auto* values=v_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            staged_v[item]=*reinterpret_cast<const std::uint32_t*>(values);
        }
        __syncthreads();
        #pragma unroll
        for(int offset=0;offset<kTileKeys;++offset) {
            if(offset<extent) {
                const int key=first+offset;
                const float2 represented=__half22float2(
                    *reinterpret_cast<const __half2*>(staged_v+offset*128+lane));
                value00+=scores0[key]*represented.x;value01+=scores0[key]*represented.y;
                value10+=scores1[key]*represented.x;value11+=scores1[key]*represented.y;
                value20+=scores2[key]*represented.x;value21+=scores2[key]*represented.y;
            }
        }
        __syncthreads();
    }
    const int d=lane*2;
    auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
    auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
    auto* destination2=output+(query*kQHeads+head2)*kHeadDim+d;
    destination0[0]=__half_as_ushort(__float2half_rn(value00));
    destination0[1]=__half_as_ushort(__float2half_rn(value01));
    destination1[0]=__half_as_ushort(__float2half_rn(value10));
    destination1[1]=__half_as_ushort(__float2half_rn(value11));
    destination2[0]=__half_as_ushort(__float2half_rn(value20));
    destination2[1]=__half_as_ushort(__float2half_rn(value21));
}

// Natural GQA24:4 grouping. Disjoint score shards let each represented K load
// feed all six query heads. The qualified route uses six shards. Smaller
// compile-time extents preserve each key's lane and FP32 dot dependency chain.
template<int kScoreShards,bool segmented=false>
__global__ void attention_cached_gqa_six_scores_kernel(const std::uint16_t* q,
    const std::uint16_t* k_cache,float* workspace,int rows,int position,
    int capacity,const int* position_device,int query_offset,
    Exl3AttentionPageRanges pages={}) {
    static_assert(kScoreShards>=1&&kScoreShards<=6);
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kBlocksPerQuery=kKVHeads*kScoreShards;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/kScoreShards,shard=local%kScoreShards;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    __shared__ float query_values[kHeadsPerKV][kHeadDim];
    if(tid<kHeadDim) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h)
            query_values[h][tid]=__half2float(__ushort_as_half(
                q[(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]));
    }
    __syncthreads();
    for(int key=shard*blockDim.x+tid;key<count;key+=kScoreShards*blockDim.x) {
        float dot[kHeadsPerKV]={};
        const auto* key_values=k_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
        if constexpr(segmented) for(int i=0;i<pages.count;++i) {
            const auto range=pages.ranges[i];
            if(key>=range.first && key-range.first<range.rows) {
                key_values=range.k+
                    (static_cast<std::size_t>(key-range.first)*kKVHeads+
                     kv_head)*kHeadDim;
                break;
            }
        }
        for(int d=0;d<kHeadDim;d+=2) {
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(key_values+d));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h) {
                dot[h]+=query_values[h][d]*represented.x;
                dot[h]+=query_values[h][d+1]*represented.y;
            }
        }
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h)
            workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key]=dot[h]*0.0625f;
    }
}

// Coalesce the physically strided K history reads without changing any score
// arithmetic. Each warp loads eight complete, adjacent 256-half K rows into a
// private shared tile using aligned 16-byte vectors. Four lanes then own each
// key: three lanes retain two of the original six independent head chains and
// the fourth is inactive. Every active lane consumes d=0..255 in the same
// half2 order and writes the unchanged score plane.
template<int kScoreShards>
__global__ void attention_cached_gqa_six_scores_k_tile64_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,float* workspace,
    int rows,int position,int capacity,const int* position_device,
    int query_offset) {
    static_assert(kScoreShards>=1&&kScoreShards<=6);
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kKeysPerWarp=8;
    constexpr int kKeysPerTile=8*kKeysPerWarp;
    constexpr int kBlocksPerQuery=kKVHeads*kScoreShards;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/kScoreShards,shard=local%kScoreShards;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    __shared__ float query_values[kHeadsPerKV][kHeadDim];
    __shared__ __align__(16) std::uint16_t key_tile[8][kKeysPerWarp][kHeadDim];
    if(tid<kHeadDim) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h)
            query_values[h][tid]=__half2float(__ushort_as_half(
                q[(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]));
    }
    __syncthreads();
    for(int slab=shard*blockDim.x;slab<count;
        slab+=kScoreShards*blockDim.x) {
        #pragma unroll
        for(int first=0;first<blockDim.x;first+=kKeysPerTile) {
            const int key_slot=lane/4;
            const int key=slab+first+warp*kKeysPerWarp+key_slot;
            // All 32 lanes cover one key's 256 halves as one coalesced row.
            #pragma unroll
            for(int staged_key=0;staged_key<kKeysPerWarp;++staged_key) {
                const int load_key=slab+first+warp*kKeysPerWarp+staged_key;
                if(load_key<count) {
                    const auto* source=k_cache+
                        (static_cast<std::size_t>(load_key)*kKVHeads+kv_head)*
                            kHeadDim+lane*8;
                    auto* destination=key_tile[warp][staged_key]+lane*8;
                    *reinterpret_cast<uint4*>(destination)=
                        *reinterpret_cast<const uint4*>(source);
                }
            }
            __syncthreads();
            const int head_group=lane%4;
            if(key<count&&head_group<3) {
                const int head0=head_group*2,head1=head0+1;
                float dot0=0.0f,dot1=0.0f;
                const auto* represented_key=key_tile[warp][key_slot];
                for(int d=0;d<kHeadDim;d+=2) {
                    const float2 represented=__half22float2(
                        *reinterpret_cast<const __half2*>(represented_key+d));
                    dot0+=query_values[head0][d]*represented.x;
                    dot0+=query_values[head0][d+1]*represented.y;
                    dot1+=query_values[head1][d]*represented.x;
                    dot1+=query_values[head1][d+1]*represented.y;
                }
                workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+head0)*capacity+key]=dot0*0.0625f;
                workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+head1)*capacity+key]=dot1*0.0625f;
            }
            __syncthreads();
        }
    }
}

// Adjacent chronological queries share every represented K load. The two
// query rows retain disjoint score planes, their original d=0..255 FP32 FMA
// chains, and independent causal extents. An odd final row is handled by the
// first half only. The qualified triple staged-value kernel remains unchanged.
template<int kScoreShards>
__global__ void attention_cached_gqa_six_query_pair_scores_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    static_assert(kScoreShards>=1&&kScoreShards<=6);
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kBlocksPerPair=kKVHeads*kScoreShards;
    const int pair=static_cast<int>(blockIdx.x)/kBlocksPerPair;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerPair;
    const int kv_head=local/kScoreShards,shard=local%kScoreShards;
    const int query0=pair*2,query1=query0+1;
    const int tid=threadIdx.x;
    if(query0>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count0=base+query0+1;
    const int count1=query1<rows?base+query1+1:0;
    const int count=count1?count1:count0;
    if(count0<1||count>capacity) return;
    __shared__ float query_values[2][kHeadsPerKV][kHeadDim];
    if(tid<kHeadDim) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            query_values[0][h][tid]=__half2float(__ushort_as_half(
                q[(query0*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]));
            if(count1) query_values[1][h][tid]=__half2float(__ushort_as_half(
                q[(query1*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]));
        }
    }
    __syncthreads();
    for(int key=shard*blockDim.x+tid;key<count;key+=kScoreShards*blockDim.x) {
        float dot0[kHeadsPerKV]={},dot1[kHeadsPerKV]={};
        const auto* key_values=k_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
        for(int d=0;d<kHeadDim;d+=2) {
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(key_values+d));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h) {
                if(key<count0) {
                    dot0[h]+=query_values[0][h][d]*represented.x;
                    dot0[h]+=query_values[0][h][d+1]*represented.y;
                }
                if(count1) {
                    dot1[h]+=query_values[1][h][d]*represented.x;
                    dot1[h]+=query_values[1][h][d+1]*represented.y;
                }
            }
        }
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            if(key<count0) workspace[(static_cast<std::size_t>(query0)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key]=dot0[h]*0.0625f;
            if(count1) workspace[(static_cast<std::size_t>(query1)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key]=dot1[h]*0.0625f;
        }
    }
}

void launch_attention_cached_gqa_six_scores(const std::uint16_t* q,
    const std::uint16_t* k_cache,float* workspace,int rows,int position,
    int capacity,const int* position_device,int query_offset,int score_shards,
    cudaStream_t stream) {
    const int blocks=rows*kKVHeads*score_shards;
    switch(score_shards) {
    case 1: attention_cached_gqa_six_scores_kernel<1><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 2: attention_cached_gqa_six_scores_kernel<2><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 3: attention_cached_gqa_six_scores_kernel<3><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 4: attention_cached_gqa_six_scores_kernel<4><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 5: attention_cached_gqa_six_scores_kernel<5><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 6: attention_cached_gqa_six_scores_kernel<6><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    default: throw std::invalid_argument("exact attention GQA six score shards");
    }
}

void launch_attention_cached_gqa_six_segmented_scores(const std::uint16_t* q,
    const std::uint16_t* k_cache,float* workspace,int rows,int position,
    int capacity,const int* position_device,int query_offset,int score_shards,
    Exl3AttentionPageRanges pages,cudaStream_t stream) {
    const int blocks=rows*kKVHeads*score_shards;
    switch(score_shards) {
    case 1: attention_cached_gqa_six_scores_kernel<1,true><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset,pages);break;
    case 2: attention_cached_gqa_six_scores_kernel<2,true><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset,pages);break;
    case 3: attention_cached_gqa_six_scores_kernel<3,true><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset,pages);break;
    case 4: attention_cached_gqa_six_scores_kernel<4,true><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset,pages);break;
    case 5: attention_cached_gqa_six_scores_kernel<5,true><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset,pages);break;
    case 6: attention_cached_gqa_six_scores_kernel<6,true><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset,pages);break;
    default: throw std::invalid_argument("segmented exact attention GQA six score shards");
    }
}

void launch_attention_cached_gqa_six_scores_k_tile64(
    const std::uint16_t* q,const std::uint16_t* k_cache,float* workspace,
    int rows,int position,int capacity,const int* position_device,
    int query_offset,int score_shards,cudaStream_t stream) {
    const int blocks=rows*kKVHeads*score_shards;
    switch(score_shards) {
    case 1: attention_cached_gqa_six_scores_k_tile64_kernel<1><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 2: attention_cached_gqa_six_scores_k_tile64_kernel<2><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 3: attention_cached_gqa_six_scores_k_tile64_kernel<3><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 4: attention_cached_gqa_six_scores_k_tile64_kernel<4><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 5: attention_cached_gqa_six_scores_k_tile64_kernel<5><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 6: attention_cached_gqa_six_scores_k_tile64_kernel<6><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    default: throw std::invalid_argument("exact attention GQA six K-tile score shards");
    }
}

void launch_attention_cached_gqa_six_query_pair_scores(const std::uint16_t* q,
    const std::uint16_t* k_cache,float* workspace,int rows,int position,
    int capacity,const int* position_device,int query_offset,int score_shards,
    cudaStream_t stream) {
    const int blocks=((rows+1)/2)*kKVHeads*score_shards;
    switch(score_shards) {
    case 1: attention_cached_gqa_six_query_pair_scores_kernel<1><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 2: attention_cached_gqa_six_query_pair_scores_kernel<2><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 3: attention_cached_gqa_six_query_pair_scores_kernel<3><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 4: attention_cached_gqa_six_query_pair_scores_kernel<4><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 5: attention_cached_gqa_six_query_pair_scores_kernel<5><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    case 6: attention_cached_gqa_six_query_pair_scores_kernel<6><<<blocks,256,0,stream>>>(
        q,k_cache,workspace,rows,position,capacity,position_device,query_offset);break;
    default: throw std::invalid_argument("exact attention GQA six query-pair score shards");
    }
}

__global__ void attention_cached_gqa_six_values_kernel(const std::uint16_t* v_cache,
    std::uint16_t* output,float* workspace,int rows,int position,int capacity,
    const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    __shared__ float maximum[kHeadsPerKV],denominator[kHeadsPerKV];
    __shared__ float score_tile[kHeadsPerKV*256];
    if(tid<kHeadsPerKV) maximum[tid]=-3.402823466e+38F;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*256+tid]=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=maximum[tid];const int extent=min(blockDim.x,count-first);
            for(int offset=0;offset<extent;++offset)
                value=fmaxf(value,score_tile[tid*256+offset]);
            maximum[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=expf(score-maximum[h]);
        }
    }
    __syncthreads();
    if(tid<kHeadsPerKV) denominator[tid]=0.0f;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*256+tid]=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=denominator[tid];const int extent=min(blockDim.x,count-first);
            for(int offset=0;offset<extent;++offset)value+=score_tile[tid*256+offset];
            denominator[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=score/denominator[h];
        }
    }
    __syncthreads();
    if(tid<kHeadDim/2) {
        float value[kHeadsPerKV][2]={};const int d=tid*2;
        for(int key=0;key<count;++key) {
            const auto* values=v_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            const float2 represented=__half22float2(*reinterpret_cast<const __half2*>(values));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h) {
                const float weight=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
                value[h][0]+=weight*represented.x;value[h][1]+=weight*represented.y;
            }
        }
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            auto* destination=output+(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+d;
            destination[0]=__half_as_ushort(__float2half_rn(value[h][0]));
            destination[1]=__half_as_ushort(__float2half_rn(value[h][1]));
        }
    }
}

// Split the exact six-head softmax from value accumulation so the value phase
// can restore two independently schedulable dimension shards per KV head.
template<bool DeferredNormalize=false>
__global__ void attention_cached_gqa_six_softmax_kernel(float* workspace,int rows,
    int position,int capacity,const int* position_device,int query_offset,
    float* deferred_denominators=nullptr) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    __shared__ float maximum[kHeadsPerKV],denominator[kHeadsPerKV];
    __shared__ float score_tile[kHeadsPerKV*256];
    if(tid<kHeadsPerKV) maximum[tid]=-3.402823466e+38F;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*256+tid]=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=maximum[tid];const int extent=min(blockDim.x,count-first);
            for(int offset=0;offset<extent;++offset)
                value=fmaxf(value,score_tile[tid*256+offset]);
            maximum[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=expf(score-maximum[h]);
        }
    }
    __syncthreads();
    if(tid<kHeadsPerKV) denominator[tid]=0.0f;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*256+tid]=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=denominator[tid];const int extent=min(blockDim.x,count-first);
            for(int offset=0;offset<extent;++offset)value+=score_tile[tid*256+offset];
            denominator[tid]=value;
        }
        __syncthreads();
    }
    if constexpr(DeferredNormalize) {
        if(tid<kHeadsPerKV)
            deferred_denominators[query*kQHeads+kv_head*kHeadsPerKV+tid]=
                denominator[tid];
    } else {
        for(int key=tid;key<count;key+=blockDim.x) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h) {
                float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
                score=score/denominator[h];
            }
        }
    }
}

// Exact 512-key staging for the selected six-head softmax. The 256-thread CTA
// loads two chronological score positions per lane, then the same six scalar
// FP32 chains consume offsets 0..511 in order. This halves tile barriers while
// retaining the score workspace and every max/add/exp/div statement.
__global__ void attention_cached_gqa_six_softmax_tile512_kernel(float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kTileKeys=512;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    __shared__ float maximum[kHeadsPerKV],denominator[kHeadsPerKV];
    __shared__ float score_tile[kHeadsPerKV*kTileKeys];
    if(tid<kHeadsPerKV) maximum[tid]=-3.402823466e+38F;
    __syncthreads();
    for(int first=0;first<count;first+=kTileKeys) {
        #pragma unroll
        for(int half=0;half<2;++half) {
            const int offset=half*blockDim.x+tid;
            const int key=first+offset;
            if(key<count) {
                #pragma unroll
                for(int h=0;h<kHeadsPerKV;++h)
                    score_tile[h*kTileKeys+offset]=workspace[
                        (static_cast<std::size_t>(query)*kQHeads+
                         kv_head*kHeadsPerKV+h)*capacity+key];
            }
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=maximum[tid];const int extent=min(kTileKeys,count-first);
            for(int offset=0;offset<extent;++offset)
                value=fmaxf(value,score_tile[tid*kTileKeys+offset]);
            maximum[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=expf(score-maximum[h]);
        }
    }
    __syncthreads();
    if(tid<kHeadsPerKV) denominator[tid]=0.0f;
    __syncthreads();
    for(int first=0;first<count;first+=kTileKeys) {
        #pragma unroll
        for(int half=0;half<2;++half) {
            const int offset=half*blockDim.x+tid;
            const int key=first+offset;
            if(key<count) {
                #pragma unroll
                for(int h=0;h<kHeadsPerKV;++h)
                    score_tile[h*kTileKeys+offset]=workspace[
                        (static_cast<std::size_t>(query)*kQHeads+
                         kv_head*kHeadsPerKV+h)*capacity+key];
            }
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=denominator[tid];const int extent=min(kTileKeys,count-first);
            for(int offset=0;offset<extent;++offset)
                value+=score_tile[tid*kTileKeys+offset];
            denominator[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=score/denominator[h];
        }
    }
}

// Exact fusion of the selected six-head softmax and six scalar value chains.
// The normalized scores remain in the caller's score plane, but are staged in
// shared memory while the value phase consumes them. Each thread owns one
// represented FP16 V dimension, so a KV-head V element is loaded once for all
// six query heads. The FP32 max/exp/sum/div and chronological value order match
// the established separate routes.
__global__ void attention_cached_gqa_six_softmax_fused_scalar_values_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kTileKeys=256;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const std::size_t score_base=
        (static_cast<std::size_t>(query)*kQHeads+kv_head*kHeadsPerKV)*capacity;
    __shared__ float maximum[kHeadsPerKV],denominator[kHeadsPerKV];
    __shared__ float score_tile[kHeadsPerKV*kTileKeys];

    if(tid<kHeadsPerKV) maximum[tid]=-3.402823466e+38F;
    __syncthreads();
    for(int first=0;first<count;first+=kTileKeys) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*kTileKeys+tid]=workspace[
                    score_base+static_cast<std::size_t>(h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=maximum[tid];
            const int extent=min(kTileKeys,count-first);
            for(int offset=0;offset<extent;++offset)
                value=fmaxf(value,score_tile[tid*kTileKeys+offset]);
            maximum[tid]=value;
        }
        __syncthreads();
    }

    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[
                score_base+static_cast<std::size_t>(h)*capacity+key];
            score=expf(score-maximum[h]);
        }
    }
    __syncthreads();
    if(tid<kHeadsPerKV) denominator[tid]=0.0f;
    __syncthreads();
    for(int first=0;first<count;first+=kTileKeys) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*kTileKeys+tid]=workspace[
                    score_base+static_cast<std::size_t>(h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=denominator[tid];
            const int extent=min(kTileKeys,count-first);
            for(int offset=0;offset<extent;++offset)
                value+=score_tile[tid*kTileKeys+offset];
            denominator[tid]=value;
        }
        __syncthreads();
    }

    float value[kHeadsPerKV]={};
    const int head0=kv_head*kHeadsPerKV;
    for(int first=0;first<count;first+=kTileKeys) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*kTileKeys+tid]=workspace[
                    score_base+static_cast<std::size_t>(h)*capacity+key];
        }
        __syncthreads();
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h) {
                const float normalized=score_tile[h*kTileKeys+tid]/denominator[h];
                score_tile[h*kTileKeys+tid]=normalized;
                workspace[score_base+static_cast<std::size_t>(h)*capacity+key]=
                    normalized;
            }
        }
        __syncthreads();
        const int extent=min(kTileKeys,count-first);
        for(int offset=0;offset<extent;++offset) {
            const int value_key=first+offset;
            const float represented=__half2float(__ushort_as_half(v_cache[
                (static_cast<std::size_t>(value_key)*kKVHeads+kv_head)*
                    kHeadDim+tid]));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                value[h]+=score_tile[h*kTileKeys+offset]*represented;
        }
        __syncthreads();
    }

    #pragma unroll
    for(int h=0;h<kHeadsPerKV;++h)
        output[(query*kQHeads+head0+h)*kHeadDim+tid]=
            __half_as_ushort(__float2half_rn(value[h]));
}

// Exact fusion of the selected six-head softmax and two three-head value
// groups. The normalization statements are kept in the same order as
// attention_cached_gqa_six_softmax_kernel. After the normalized score stores
// become CTA-visible, each 128-thread half executes the original chronological
// three-head value chain with independent represented V half2 loads.
__global__ void attention_cached_gqa_six_softmax_triple_values_fused_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    __shared__ float maximum[kHeadsPerKV],denominator[kHeadsPerKV];
    __shared__ float score_tile[kHeadsPerKV*256];
    if(tid<kHeadsPerKV) maximum[tid]=-3.402823466e+38F;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*256+tid]=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=maximum[tid];const int extent=min(blockDim.x,count-first);
            for(int offset=0;offset<extent;++offset)
                value=fmaxf(value,score_tile[tid*256+offset]);
            maximum[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=expf(score-maximum[h]);
        }
    }
    __syncthreads();
    if(tid<kHeadsPerKV) denominator[tid]=0.0f;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*256+tid]=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=denominator[tid];const int extent=min(blockDim.x,count-first);
            for(int offset=0;offset<extent;++offset)value+=score_tile[tid*256+offset];
            denominator[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=score/denominator[h];
        }
    }
    __syncthreads();

    const int group=tid/128,lane=tid%128;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const float* scores0=workspace+(static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=workspace+(static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    const float* scores2=workspace+(static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
    float value20=0.0f,value21=0.0f;
    const int d=lane*2;
    for(int key=0;key<count;++key) {
        const auto* values=v_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        const float2 represented=__half22float2(
            *reinterpret_cast<const __half2*>(values));
        value00+=scores0[key]*represented.x;value01+=scores0[key]*represented.y;
        value10+=scores1[key]*represented.x;value11+=scores1[key]*represented.y;
        value20+=scores2[key]*represented.x;value21+=scores2[key]*represented.y;
    }
    auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
    auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
    auto* destination2=output+(query*kQHeads+head2)*kHeadDim+d;
    destination0[0]=__half_as_ushort(__float2half_rn(value00));
    destination0[1]=__half_as_ushort(__float2half_rn(value01));
    destination1[0]=__half_as_ushort(__float2half_rn(value10));
    destination1[1]=__half_as_ushort(__float2half_rn(value11));
    destination2[0]=__half_as_ushort(__float2half_rn(value20));
    destination2[1]=__half_as_ushort(__float2half_rn(value21));
}

// Contiguous target-only decode candidate. Each CTA owns one KV head and
// preserves the established score, softmax, and three-head value arithmetic
// while removing the score/softmax/value launch boundaries. The exact score
// plane remains materialized in the caller-owned workspace so the original
// route's represented state and fallback ownership are unchanged.
__global__ void attention_cached_gqa_six_decode_fused_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;

    __shared__ float query_values[kHeadsPerKV][kHeadDim];
    __shared__ float maximum[kHeadsPerKV],denominator[kHeadsPerKV];
    __shared__ float score_tile[kHeadsPerKV*256];
    if(tid<kHeadDim) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h)
            query_values[h][tid]=__half2float(__ushort_as_half(
                q[(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]));
    }
    __syncthreads();

    // This is the same d=0..255 half2 FP32 dot sequence as the selected
    // six-head score kernel; only the score launch ownership is different.
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            float dot[kHeadsPerKV]={};
            const auto* key_values=k_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
            for(int d=0;d<kHeadDim;d+=2) {
                const float2 represented=__half22float2(
                    *reinterpret_cast<const __half2*>(key_values+d));
                #pragma unroll
                for(int h=0;h<kHeadsPerKV;++h) {
                    dot[h]+=query_values[h][d]*represented.x;
                    dot[h]+=query_values[h][d+1]*represented.y;
                }
            }
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key]=dot[h]*0.0625f;
        }
    }
    __syncthreads();

    // Keep the exact route's chronological max, exp, sum, and division
    // statements. score_tile only stages the same global score values.
    if(tid<kHeadsPerKV) maximum[tid]=-3.402823466e+38F;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*256+tid]=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=maximum[tid];
            const int extent=min(blockDim.x,count-first);
            for(int offset=0;offset<extent;++offset)
                value=fmaxf(value,score_tile[tid*256+offset]);
            maximum[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=expf(score-maximum[h]);
        }
    }
    __syncthreads();
    if(tid<kHeadsPerKV) denominator[tid]=0.0f;
    __syncthreads();
    for(int first=0;first<count;first+=blockDim.x) {
        const int key=first+tid;
        if(key<count) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h*256+tid]=workspace[(static_cast<std::size_t>(query)*kQHeads+
                    kv_head*kHeadsPerKV+h)*capacity+key];
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float value=denominator[tid];
            const int extent=min(blockDim.x,count-first);
            for(int offset=0;offset<extent;++offset)
                value+=score_tile[tid*256+offset];
            denominator[tid]=value;
        }
        __syncthreads();
    }
    for(int key=tid;key<count;key+=blockDim.x) {
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            float& score=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            score=score/denominator[h];
        }
    }
    __syncthreads();

    const int group=tid/128, lane=tid%128;
    const int head0=kv_head*kHeadsPerKV+group*3;
    const int head1=head0+1,head2=head0+2;
    const float* scores0=workspace+
        (static_cast<std::size_t>(query)*kQHeads+head0)*capacity;
    const float* scores1=workspace+
        (static_cast<std::size_t>(query)*kQHeads+head1)*capacity;
    const float* scores2=workspace+
        (static_cast<std::size_t>(query)*kQHeads+head2)*capacity;
    float value00=0.0f,value01=0.0f,value10=0.0f,value11=0.0f;
    float value20=0.0f,value21=0.0f;
    const int d=lane*2;
    for(int key=0;key<count;++key) {
        const auto* values=v_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        const float2 represented=__half22float2(
            *reinterpret_cast<const __half2*>(values));
        value00+=scores0[key]*represented.x;value01+=scores0[key]*represented.y;
        value10+=scores1[key]*represented.x;value11+=scores1[key]*represented.y;
        value20+=scores2[key]*represented.x;value21+=scores2[key]*represented.y;
    }
    auto* destination0=output+(query*kQHeads+head0)*kHeadDim+d;
    auto* destination1=output+(query*kQHeads+head1)*kHeadDim+d;
    auto* destination2=output+(query*kQHeads+head2)*kHeadDim+d;
    destination0[0]=__half_as_ushort(__float2half_rn(value00));
    destination0[1]=__half_as_ushort(__float2half_rn(value01));
    destination1[0]=__half_as_ushort(__float2half_rn(value10));
    destination1[1]=__half_as_ushort(__float2half_rn(value11));
    destination2[0]=__half_as_ushort(__float2half_rn(value20));
    destination2[1]=__half_as_ushort(__float2half_rn(value21));
}

__global__ void attention_cached_gqa_six_values_sharded_kernel(
    const std::uint16_t* v_cache,std::uint16_t* output,const float* workspace,
    int rows,int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads,kDimensionShards=2;
    constexpr int kBlocksPerQuery=kKVHeads*kDimensionShards;
    const int query=static_cast<int>(blockIdx.x)/kBlocksPerQuery;
    const int local=static_cast<int>(blockIdx.x)%kBlocksPerQuery;
    const int kv_head=local/kDimensionShards,shard=local%kDimensionShards;
    const int tid=threadIdx.x;
    if(query>=rows||tid>=kHeadDim/(2*kDimensionShards)) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;
    const int d=(shard*(kHeadDim/(2*kDimensionShards))+tid)*2;
    float value[kHeadsPerKV][2]={};
    for(int key=0;key<count;++key) {
        const auto* values=v_cache+(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
        const float2 represented=__half22float2(*reinterpret_cast<const __half2*>(values));
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            const float weight=workspace[(static_cast<std::size_t>(query)*kQHeads+
                kv_head*kHeadsPerKV+h)*capacity+key];
            value[h][0]+=weight*represented.x;value[h][1]+=weight*represented.y;
        }
    }
    #pragma unroll
    for(int h=0;h<kHeadsPerKV;++h) {
        auto* destination=output+(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+d;
        destination[0]=__half_as_ushort(__float2half_rn(value[h][0]));
        destination[1]=__half_as_ushort(__float2half_rn(value[h][1]));
    }
}

// Exact large-M prefill route. One CTA owns a query/KV-head pair and keeps the
// full six-head score plane in dynamic shared memory. This preserves the
// selected route's per-score dot order, chronological max/sum chains,
// represented exp/div stores, and chronological value accumulation while
// eliminating the global score plane and its repeated traversals.
template<int Threads,bool ParallelSoftmax,bool HeadSplit256,
         bool DimensionSplit256=false>
__launch_bounds__(Threads,1)
__global__ void attention_cached_gqa_six_shared_scores_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,int score_stride) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    static_assert(Threads==128||Threads==256);
    static_assert(!HeadSplit256||Threads==256);
    static_assert(!DimensionSplit256||Threads==256);
    static_assert(!(HeadSplit256&&DimensionSplit256));
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int count=position+query+1;
    if(count<1||count>capacity||count>score_stride) return;

    extern __shared__ unsigned char shared_storage[];
    auto* query_bits=reinterpret_cast<std::uint16_t*>(shared_storage);
    auto* scores=reinterpret_cast<float*>(query_bits+kHeadsPerKV*kHeadDim);
    for(int index=tid;index<kHeadsPerKV*kHeadDim;index+=blockDim.x)
        query_bits[index]=q[(query*kQHeads+kv_head*kHeadsPerKV)*kHeadDim+index];
    __syncthreads();

    for(int key=tid;key<count;key+=blockDim.x) {
        float dot[kHeadsPerKV]={};
        const auto* represented_key=k_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
        for(int d=0;d<kHeadDim;d+=2) {
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(represented_key+d));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h) {
                dot[h]+=__half2float(__ushort_as_half(
                    query_bits[h*kHeadDim+d]))*represented.x;
                dot[h]+=__half2float(__ushort_as_half(
                    query_bits[h*kHeadDim+d+1]))*represented.y;
            }
        }
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h)
            scores[h*score_stride+key]=dot[h]*0.0625f;
    }
    __syncthreads();

    if constexpr(ParallelSoftmax) {
        // Q is dead after score construction, so reuse its shared allocation
        // for six maxima and six denominators without increasing the 4K
        // dynamic-shared footprint. Max and sum retain their exact
        // chronological chains; exp and division are per-element operations.
        auto* softmax_scalars=reinterpret_cast<float*>(query_bits);
        if(tid<kHeadsPerKV) {
            float maximum=-3.402823466e+38F;
            for(int key=0;key<count;++key)
                maximum=fmaxf(maximum,scores[tid*score_stride+key]);
            softmax_scalars[tid]=maximum;
        }
        __syncthreads();
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h)
            for(int key=tid;key<count;key+=blockDim.x) {
                float& score=scores[h*score_stride+key];
                score=expf(score-softmax_scalars[h]);
            }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float denominator=0.0f;
            for(int key=0;key<count;++key)
                denominator+=scores[tid*score_stride+key];
            softmax_scalars[kHeadsPerKV+tid]=denominator;
        }
        __syncthreads();
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h)
            for(int key=tid;key<count;key+=blockDim.x) {
                float& score=scores[h*score_stride+key];
                score=score/softmax_scalars[kHeadsPerKV+h];
            }
    } else if(tid<kHeadsPerKV) {
        float maximum=-3.402823466e+38F;
        for(int key=0;key<count;++key)
            maximum=fmaxf(maximum,scores[tid*score_stride+key]);
        for(int key=0;key<count;++key) {
            float& score=scores[tid*score_stride+key];
            score=expf(score-maximum);
        }
        float denominator=0.0f;
        for(int key=0;key<count;++key)
            denominator+=scores[tid*score_stride+key];
        for(int key=0;key<count;++key) {
            float& score=scores[tid*score_stride+key];
            score=score/denominator;
        }
    }
    __syncthreads();

    if constexpr(DimensionSplit256) {
        // All 256 threads own one scalar output dimension. This retains each
        // output scalar's chronological FP32 key/head chain while halving the
        // per-thread accumulator footprint. Unlike head-split256, every V
        // scalar is loaded exactly once rather than once per head group.
        const int d=tid;
        float values[kHeadsPerKV]={};
        for(int key=0;key<count;++key) {
            const auto* represented_value=v_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            const float represented=__half2float(
                __ushort_as_half(*represented_value));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                values[h]+=scores[h*score_stride+key]*represented;
        }
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            auto* destination=output+
                (query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+d;
            *destination=__half_as_ushort(__float2half_rn(values[h]));
        }
    } else if constexpr(HeadSplit256) {
        constexpr int kHeadsPerThread=kHeadsPerKV/2;
        const int d=(tid%(kHeadDim/2))*2;
        const int first_head=(tid/(kHeadDim/2))*kHeadsPerThread;
        float values[kHeadsPerThread][2]={};
        for(int key=0;key<count;++key) {
            const auto* represented_value=v_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(represented_value));
            #pragma unroll
            for(int local_head=0;local_head<kHeadsPerThread;++local_head) {
                const int h=first_head+local_head;
                const float weight=scores[h*score_stride+key];
                values[local_head][0]+=weight*represented.x;
                values[local_head][1]+=weight*represented.y;
            }
        }
        #pragma unroll
        for(int local_head=0;local_head<kHeadsPerThread;++local_head) {
            const int h=first_head+local_head;
            auto* destination=output+
                (query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+d;
            destination[0]=__half_as_ushort(__float2half_rn(values[local_head][0]));
            destination[1]=__half_as_ushort(__float2half_rn(values[local_head][1]));
        }
    } else if(tid<kHeadDim/2) {
        const int d=tid*2;
        float values[kHeadsPerKV][2]={};
        for(int key=0;key<count;++key) {
            const auto* represented_value=v_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+d;
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(represented_value));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h) {
                const float weight=scores[h*score_stride+key];
                values[h][0]+=weight*represented.x;
                values[h][1]+=weight*represented.y;
            }
        }
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) {
            auto* destination=output+
                (query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+d;
            destination[0]=__half_as_ushort(__float2half_rn(values[h][0]));
            destination[1]=__half_as_ushort(__float2half_rn(values[h][1]));
        }
    }
}

constexpr std::size_t prefill_shared_score_bytes(int score_stride) noexcept {
    return static_cast<std::size_t>(kQHeads/kKVHeads)*kHeadDim*sizeof(std::uint16_t)+
        static_cast<std::size_t>(kQHeads/kKVHeads)*score_stride*sizeof(float);
}

void launch_prefill_attention_shared_scores(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream,bool threads256,
    bool parallel_softmax,bool head_split256,bool dimension_split256) {
    if(!q||!k||!v||!output||rows<1||position<0||position+rows>capacity||
       position+rows>4096)
        throw std::invalid_argument("exact shared-score prefill shape");
    constexpr int kMaximumScoreStride=4096;
    constexpr std::size_t kMaximumSharedBytes=
        prefill_shared_score_bytes(kMaximumScoreStride);
    static const cudaError_t attribute128=cudaFuncSetAttribute(
        attention_cached_gqa_six_shared_scores_kernel<128,false,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kMaximumSharedBytes));
    static const cudaError_t attribute256=cudaFuncSetAttribute(
        attention_cached_gqa_six_shared_scores_kernel<256,false,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kMaximumSharedBytes));
    static const cudaError_t parallel_attribute128=cudaFuncSetAttribute(
        attention_cached_gqa_six_shared_scores_kernel<128,true,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kMaximumSharedBytes));
    static const cudaError_t parallel_attribute256=cudaFuncSetAttribute(
        attention_cached_gqa_six_shared_scores_kernel<256,true,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kMaximumSharedBytes));
    static const cudaError_t head_split_attribute=cudaFuncSetAttribute(
        attention_cached_gqa_six_shared_scores_kernel<256,true,true,false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kMaximumSharedBytes));
    static const cudaError_t dimension_split_attribute=cudaFuncSetAttribute(
        attention_cached_gqa_six_shared_scores_kernel<256,true,false,true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kMaximumSharedBytes));
    if((head_split256||dimension_split256)&&!parallel_softmax)
        throw std::invalid_argument(
            "exact shared-score value split requires parallel softmax");
    if(dimension_split256&&(threads256||head_split256))
        throw std::invalid_argument(
            "exact shared-score dimension-split256 requires exclusive value topology");
    const cudaError_t attribute=parallel_softmax?
        (dimension_split256?dimension_split_attribute:
         head_split256?head_split_attribute:
            (threads256?parallel_attribute256:parallel_attribute128)):
        (threads256?attribute256:attribute128);
    cuda_check(attribute,
        "set exact shared-score prefill shared memory");
    const int score_stride=position+rows;
    if(dimension_split256)
        attention_cached_gqa_six_shared_scores_kernel<256,true,false,true><<<
            rows*kKVHeads,256,prefill_shared_score_bytes(score_stride),stream>>>(
                q,k,v,output,rows,position,capacity,score_stride);
    else if(head_split256)
        attention_cached_gqa_six_shared_scores_kernel<256,true,true,false><<<
            rows*kKVHeads,256,prefill_shared_score_bytes(score_stride),stream>>>(
                q,k,v,output,rows,position,capacity,score_stride);
    else if(threads256) {
        if(parallel_softmax)
            attention_cached_gqa_six_shared_scores_kernel<256,true,false><<<
                rows*kKVHeads,256,prefill_shared_score_bytes(score_stride),stream>>>(
                    q,k,v,output,rows,position,capacity,score_stride);
        else
            attention_cached_gqa_six_shared_scores_kernel<256,false,false><<<
                rows*kKVHeads,256,prefill_shared_score_bytes(score_stride),stream>>>(
                    q,k,v,output,rows,position,capacity,score_stride);
    } else {
        if(parallel_softmax)
            attention_cached_gqa_six_shared_scores_kernel<128,true,false><<<
                rows*kKVHeads,128,prefill_shared_score_bytes(score_stride),stream>>>(
                    q,k,v,output,rows,position,capacity,score_stride);
        else
            attention_cached_gqa_six_shared_scores_kernel<128,false,false><<<
                rows*kKVHeads,128,prefill_shared_score_bytes(score_stride),stream>>>(
                    q,k,v,output,rows,position,capacity,score_stride);
    }
}

// T71 NUMERIC_CANDIDATE. One CTA owns a query/KV-head pair and streams
// chronological 256-key tiles. Represented K and V are each loaded once per
// Separately gated native FP16-KV flash candidate. Each block owns one
// chronological key tile for one KV head and all six sibling query heads. It
// keeps the K score, softmax, and V accumulation in one CTA and leaves only
// FP32 partial numerators plus (max, denominator) in the existing exact-score
// scratch. A small merge CTA restores the six output heads. This is deliberately
// not wired into the exact/reference path: split-local online normalization has
// a different reduction order and must pass the frozen trajectory gate before
// it can be considered for promotion.
constexpr int kFastFusedFlashKeys=128;
constexpr int kFastFusedFlashKeys256=256;
constexpr int kFastFusedFlashHeads=kQHeads/kKVHeads;
constexpr int kFastFusedFlashValues=kFastFusedFlashHeads*kHeadDim;
constexpr int kFastFusedFlashStride=kFastFusedFlashValues+2*kFastFusedFlashHeads;

// Staged-K dimension chunk. Each key lane keeps its sequential FMA chain over
// d; only the K source moves from strided global loads to a coalesced shared
// tile, so the scores are bitwise identical to the unstaged kernel.
constexpr int kFastFusedFlashStageDims=16;

template<int kKeys,bool kStagedK=false>
__global__ void attention_cached_gqa_six_fused_flash_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,float* workspace,std::uint16_t* output,
    int rows,int position,int capacity,int segments,
    const int* position_device,int query_offset) {
    const int block=static_cast<int>(blockIdx.x);
    const int blocks_per_query=kKVHeads*segments;
    const int query=block/blocks_per_query;
    const int local=block%blocks_per_query;
    const int kv_head=local/segments;
    const int segment=local%segments;
    const int tid=threadIdx.x;
    if(query>=rows) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    const int first=segment*kKeys;
    const int extent=min(kKeys,count-first);
    if(extent<=0 || count<1 || count>capacity) return;

    __shared__ __align__(16) float query_values[kFastFusedFlashHeads][kHeadDim];
    __shared__ __align__(16) float score_tile[kFastFusedFlashHeads][kKeys];
    __shared__ float maximum[kFastFusedFlashHeads];
    __shared__ float denominator[kFastFusedFlashHeads];
    if(tid<kHeadDim) {
        #pragma unroll
        for(int head=0;head<kFastFusedFlashHeads;++head)
            query_values[head][tid]=__half2float(__ushort_as_half(
                q[(query*kQHeads+kv_head*kFastFusedFlashHeads+head)*kHeadDim+tid]));
    }
    __syncthreads();

    // The first 128 lanes own K rows. All six score chains use the same
    // represented FP16 K load, while the d lanes below own the V output.
    if constexpr(kStagedK) {
        constexpr int kWords=kFastFusedFlashStageDims/2;
        constexpr int kParts=kFastFusedFlashStageDims/8;
        __shared__ __half2 key_tile[kKeys][kWords+1];
        float dot[kFastFusedFlashHeads]={};
        for(int c=0;c<kHeadDim;c+=kFastFusedFlashStageDims) {
            for(int w=tid;w<extent*kParts;w+=static_cast<int>(blockDim.x)) {
                const int key=w/kParts;
                const int part=w%kParts;
                const uint4 packed=*reinterpret_cast<const uint4*>(k_cache+
                    (static_cast<std::size_t>(first+key)*kKVHeads+kv_head)*kHeadDim+
                        c+part*8);
                key_tile[key][part*4+0]=*reinterpret_cast<const __half2*>(&packed.x);
                key_tile[key][part*4+1]=*reinterpret_cast<const __half2*>(&packed.y);
                key_tile[key][part*4+2]=*reinterpret_cast<const __half2*>(&packed.z);
                key_tile[key][part*4+3]=*reinterpret_cast<const __half2*>(&packed.w);
            }
            __syncthreads();
            if(tid<extent) {
                #pragma unroll
                for(int word=0;word<kWords;word+=2) {
                    const int d=c+2*word;
                    const float2 low=__half22float2(key_tile[tid][word]);
                    const float2 high=__half22float2(key_tile[tid][word+1]);
                    #pragma unroll
                    for(int head=0;head<kFastFusedFlashHeads;++head) {
                        const float4 query4=
                            *reinterpret_cast<const float4*>(&query_values[head][d]);
                        dot[head]+=query4.x*low.x;
                        dot[head]+=query4.y*low.y;
                        dot[head]+=query4.z*high.x;
                        dot[head]+=query4.w*high.y;
                    }
                }
            }
            __syncthreads();
        }
        if(tid<extent) {
            #pragma unroll
            for(int head=0;head<kFastFusedFlashHeads;++head)
                score_tile[head][tid]=dot[head]*0.0625f;
        }
    } else if(tid<extent) {
        const int key=first+tid;
        const auto* key_values=k_cache+
            (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
        float dot[kFastFusedFlashHeads]={};
        for(int d=0;d<kHeadDim;d+=2) {
            const float2 represented=__half22float2(
                *reinterpret_cast<const __half2*>(key_values+d));
            #pragma unroll
            for(int head=0;head<kFastFusedFlashHeads;++head) {
                dot[head]+=query_values[head][d]*represented.x;
                dot[head]+=query_values[head][d+1]*represented.y;
            }
        }
        #pragma unroll
        for(int head=0;head<kFastFusedFlashHeads;++head)
            score_tile[head][tid]=dot[head]*0.0625f;
    }
    __syncthreads();
    if(tid<kFastFusedFlashHeads) {
        float value=-3.402823466e+38F;
        for(int offset=0;offset<extent;++offset)
            value=fmaxf(value,score_tile[tid][offset]);
        maximum[tid]=value;
    }
    __syncthreads();
    if(tid<extent) {
        #pragma unroll
        for(int head=0;head<kFastFusedFlashHeads;++head)
            score_tile[head][tid]=expf(score_tile[head][tid]-maximum[head]);
    }
    __syncthreads();
    if(tid<kFastFusedFlashHeads) {
        float value=0.0f;
        for(int offset=0;offset<extent;++offset)
            value+=score_tile[tid][offset];
        denominator[tid]=value;
    }
    __syncthreads();

    float accumulated[kFastFusedFlashHeads]={};
    if(kStagedK && tid<kHeadDim) {
        // Batch eight independent V loads, then apply them in chronological
        // order: each head keeps the same FP32 chain as the scalar loop.
        int offset=0;
        for(;offset+8<=extent;offset+=8) {
            float values[8];
            #pragma unroll
            for(int j=0;j<8;++j)
                values[j]=__half2float(__ushort_as_half(
                    v_cache[(static_cast<std::size_t>(first+offset+j)*kKVHeads+kv_head)*
                        kHeadDim+tid]));
            #pragma unroll
            for(int head=0;head<kFastFusedFlashHeads;++head) {
                const float4 a=*reinterpret_cast<const float4*>(&score_tile[head][offset]);
                const float4 b=*reinterpret_cast<const float4*>(&score_tile[head][offset+4]);
                accumulated[head]+=a.x*values[0];
                accumulated[head]+=a.y*values[1];
                accumulated[head]+=a.z*values[2];
                accumulated[head]+=a.w*values[3];
                accumulated[head]+=b.x*values[4];
                accumulated[head]+=b.y*values[5];
                accumulated[head]+=b.z*values[6];
                accumulated[head]+=b.w*values[7];
            }
        }
        for(;offset<extent;++offset) {
            const auto represented=__half2float(__ushort_as_half(
                v_cache[(static_cast<std::size_t>(first+offset)*kKVHeads+kv_head)*
                    kHeadDim+tid]));
            #pragma unroll
            for(int head=0;head<kFastFusedFlashHeads;++head)
                accumulated[head]+=score_tile[head][offset]*represented;
        }
    } else if(tid<kHeadDim) {
        for(int offset=0;offset<extent;++offset) {
            const auto represented=__half2float(__ushort_as_half(
                v_cache[(static_cast<std::size_t>(first+offset)*kKVHeads+kv_head)*
                    kHeadDim+tid]));
            #pragma unroll
            for(int head=0;head<kFastFusedFlashHeads;++head)
                accumulated[head]+=score_tile[head][offset]*represented;
        }
    }
    float* slot=workspace+
        ((static_cast<std::size_t>(query)*kKVHeads+kv_head)*segments+segment)*
            kFastFusedFlashStride;
    if(tid<kHeadDim) {
        #pragma unroll
        for(int head=0;head<kFastFusedFlashHeads;++head)
            slot[head*kHeadDim+tid]=accumulated[head];
    }
    if(tid<kFastFusedFlashHeads) {
        slot[kFastFusedFlashValues+tid]=maximum[tid];
        slot[kFastFusedFlashValues+kFastFusedFlashHeads+tid]=denominator[tid];
    }
}

bool wmma32_vector_loads_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_WMMA32_VECTOR_LOADS");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

bool fused_flash_staged_k_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_FLASH_STAGED_K");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

__global__ void attention_cached_gqa_six_fused_flash_merge_kernel(
    const float* workspace,std::uint16_t* output,int rows,int segments,
    int position,int capacity,int keys,const int* position_device,
    int query_offset) {
    const int block=static_cast<int>(blockIdx.x);
    const int query=block/kKVHeads;
    const int kv_head=block%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows || tid>=kHeadDim) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1 || count>capacity) {
        for(int head=0;head<kFastFusedFlashHeads;++head)
            output[(query*kQHeads+kv_head*kFastFusedFlashHeads+head)*kHeadDim+tid]=0;
        return;
    }
    // A graph may launch the capacity-sized grid before the live frontier
    // reaches its final tile. Producers for those tiles return without writing
    // scratch; the merge must never consume their old contents.
    const int live_segments=min(segments,(count+keys-1)/keys);
    __shared__ float global_max[kFastFusedFlashHeads];
    if(tid<kFastFusedFlashHeads) {
        float value=-3.402823466e+38F;
        for(int segment=0;segment<live_segments;++segment) {
            const float* slot=workspace+
                ((static_cast<std::size_t>(query)*kKVHeads+kv_head)*segments+segment)*
                    kFastFusedFlashStride;
            value=fmaxf(value,slot[kFastFusedFlashValues+tid]);
        }
        global_max[tid]=value;
    }
    __syncthreads();
    for(int head=0;head<kFastFusedFlashHeads;++head) {
        float denominator=0.0f,numerator=0.0f;
        for(int segment=0;segment<live_segments;++segment) {
            const float* slot=workspace+
                ((static_cast<std::size_t>(query)*kKVHeads+kv_head)*segments+segment)*
                    kFastFusedFlashStride;
            const float scale=expf(slot[kFastFusedFlashValues+head]-global_max[head]);
            denominator+=slot[kFastFusedFlashValues+kFastFusedFlashHeads+head]*scale;
            numerator+=slot[head*kHeadDim+tid]*scale;
        }
        output[(query*kQHeads+kv_head*kFastFusedFlashHeads+head)*kHeadDim+tid]=
            __half_as_ushort(__float2half_rn(numerator/denominator));
    }
}

// dimension and reused across all six GQA heads. The running-max rescale is a
// mathematically equivalent online softmax, but deliberately changes FP32
// reduction topology and therefore is never an EXACT path.
__global__ void attention_cached_gqa_six_online_tiled_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows||tid>=kHeadDim) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;

    __shared__ float query_values[kHeadsPerKV][kHeadDim];
    __shared__ float score_tile[kHeadsPerKV][256];
    __shared__ float running_max[kHeadsPerKV];
    __shared__ float denominator[kHeadsPerKV];
    __shared__ float old_scale[kHeadsPerKV];
    #pragma unroll
    for(int h=0;h<kHeadsPerKV;++h)
        query_values[h][tid]=__half2float(__ushort_as_half(
            q[(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]));
    if(tid<kHeadsPerKV) {
        running_max[tid]=-3.402823466e+38F;
        denominator[tid]=0.0f;
        old_scale[tid]=0.0f;
    }
    float accumulated[kHeadsPerKV]={};
    __syncthreads();

    for(int first=0;first<count;first+=256) {
        const int extent=min(256,count-first);
        const int key=first+tid;
        if(tid<extent) {
            float dot[kHeadsPerKV]={};
            const auto* represented_key=k_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
            for(int d=0;d<kHeadDim;d+=2) {
                const float2 represented=__half22float2(
                    *reinterpret_cast<const __half2*>(represented_key+d));
                #pragma unroll
                for(int h=0;h<kHeadsPerKV;++h) {
                    dot[h]+=query_values[h][d]*represented.x;
                    dot[h]+=query_values[h][d+1]*represented.y;
                }
            }
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h][tid]=dot[h]*0.0625f;
        }
        __syncthreads();

        if(tid<kHeadsPerKV) {
            float tile_max=-3.402823466e+38F;
            for(int offset=0;offset<extent;++offset)
                tile_max=fmaxf(tile_max,score_tile[tid][offset]);
            const float next_max=fmaxf(running_max[tid],tile_max);
            old_scale[tid]=denominator[tid]==0.0f?0.0f:
                expf(running_max[tid]-next_max);
            running_max[tid]=next_max;
        }
        __syncthreads();

        if(tid<extent) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h][tid]=expf(score_tile[h][tid]-running_max[h]);
        }
        __syncthreads();

        if(tid<kHeadsPerKV) {
            float tile_sum=0.0f;
            for(int offset=0;offset<extent;++offset)
                tile_sum+=score_tile[tid][offset];
            denominator[tid]=denominator[tid]*old_scale[tid]+tile_sum;
        }
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) accumulated[h]*=old_scale[h];
        for(int offset=0;offset<extent;++offset) {
            const float represented_value=__half2float(__ushort_as_half(
                v_cache[((static_cast<std::size_t>(first+offset)*kKVHeads+
                    kv_head)*kHeadDim)+tid]));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                accumulated[h]+=score_tile[h][offset]*represented_value;
        }
        __syncthreads();
    }

    #pragma unroll
    for(int h=0;h<kHeadsPerKV;++h)
        output[(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]=
            __half_as_ushort(__float2half_rn(accumulated[h]/denominator[h]));
}

// T76 NUMERIC_CANDIDATE. Decode-specialized peer-style online attention for
// the ordinary contiguous FP16-KV cache. One CTA owns one KV head; each thread
// owns one head dimension and keeps six sibling GQA query/value accumulators in
// registers. K is reduced with warp shuffles, so the cache is read once per
// dimension and the score/value path has no materialized score plane. This is
// a separately gated FAST_MIA_PARITY route: its online max/rescale order is
// intentionally not the exact/reference order, and no cache or weight
// representation is changed.
__global__ void attention_cached_gqa_six_online_decode_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kWarps=256/32;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    const int warp=tid/32;
    const int lane=tid%32;
    if(query>=rows || tid>=kHeadDim) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1 || count>capacity) return;

    __shared__ float partials[kHeadsPerKV*kWarps];
    __shared__ float scores[kHeadsPerKV];
    float query_values[kHeadsPerKV];
    #pragma unroll
    for(int head=0;head<kHeadsPerKV;++head)
        query_values[head]=__half2float(__ushort_as_half(
            q[(query*kQHeads+kv_head*kHeadsPerKV+head)*kHeadDim+tid]));

    float running_max[kHeadsPerKV];
    float denominator[kHeadsPerKV]={};
    float accumulated[kHeadsPerKV]={};
    #pragma unroll
    for(int head=0;head<kHeadsPerKV;++head)
        running_max[head]=-3.402823466e+38F;
    __syncthreads();

    for(int key=0;key<count;++key) {
        const float represented_key=__half2float(__ushort_as_half(
            k_cache[(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+tid]));
        float partial[kHeadsPerKV];
        #pragma unroll
        for(int head=0;head<kHeadsPerKV;++head)
            partial[head]=query_values[head]*represented_key;
        #pragma unroll
        for(int mask=16;mask>0;mask>>=1)
            for(int head=0;head<kHeadsPerKV;++head)
                partial[head]+=__shfl_xor_sync(0xffffffff,partial[head],mask);
        if(lane==0) {
            #pragma unroll
            for(int head=0;head<kHeadsPerKV;++head)
                partials[head*kWarps+warp]=partial[head];
        }
        __syncthreads();
        if(warp==0 && lane<kHeadsPerKV) {
            float score=0.0f;
            #pragma unroll
            for(int other=0;other<kWarps;++other)
                score+=partials[lane*kWarps+other];
            scores[lane]=score*0.0625f;
        }
        __syncthreads();

        const float value=__half2float(__ushort_as_half(
            v_cache[(static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim+tid]));
        #pragma unroll
        for(int head=0;head<kHeadsPerKV;++head) {
            const float score=scores[head];
            const float next_max=fmaxf(running_max[head],score);
            const float old_scale=__expf(running_max[head]-next_max);
            const float weight=__expf(score-next_max);
            accumulated[head]=old_scale*accumulated[head]+weight*value;
            denominator[head]=old_scale*denominator[head]+weight;
            running_max[head]=next_max;
        }
        __syncthreads();
    }

    #pragma unroll
    for(int head=0;head<kHeadsPerKV;++head)
        output[(query*kQHeads+kv_head*kHeadsPerKV+head)*kHeadDim+tid]=
            __half_as_ushort(__float2half_rn(accumulated[head]/denominator[head]));
}

// T77 NUMERIC_CANDIDATE. The T76 online route uses one CTA for a KV head,
// but every represented key crosses two CTA-wide barriers while all six GQA
// heads are carried by every thread. This native-only variant assigns one
// warp to each of the six query heads. A lane owns eight adjacent dimensions,
// so its score reduction and value accumulation are entirely warp-local; the
// cache remains the ordinary token-major FP16-KV representation and the
// chronological online max/sum/value order is retained per output head.
// It is separately gated because the score reduction topology is different
// from the established T76 candidate and is not an exact/reference route.
__global__ void attention_cached_gqa_six_online_decode_warp_heads_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kDimensionsPerLane=kHeadDim/32;
    static_assert(kHeadsPerKV==6 && kDimensionsPerLane==8);
    const int block=static_cast<int>(blockIdx.x);
    const int query=block/kKVHeads;
    const int kv_head=block%kKVHeads;
    const int tid=static_cast<int>(threadIdx.x);
    const int head=tid/32;
    const int lane=tid%32;
    if(query>=rows || head>=kHeadsPerKV) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1 || count>capacity) return;

    const std::size_t q_base=(static_cast<std::size_t>(query)*kQHeads+
        kv_head*kHeadsPerKV+head)*kHeadDim;
    const int dimension_base=lane*kDimensionsPerLane;
    float query_values[kDimensionsPerLane];
    #pragma unroll
    for(int i=0;i<kDimensionsPerLane;++i)
        query_values[i]=__half2float(__ushort_as_half(q[q_base+
            dimension_base+i]));

    float running_max=-3.402823466e+38F;
    float denominator=0.0f;
    float accumulated[kDimensionsPerLane]={};
    for(int key=0;key<count;++key) {
        const std::size_t cache_base=(static_cast<std::size_t>(key)*kKVHeads+
            kv_head)*kHeadDim+dimension_base;
        float score=0.0f;
        #pragma unroll
        for(int i=0;i<kDimensionsPerLane;++i)
            score+=query_values[i]*__half2float(__ushort_as_half(
                k_cache[cache_base+i]));
        #pragma unroll
        for(int mask=16;mask>0;mask>>=1)
            score+=__shfl_xor_sync(0xffffffff,score,mask);
        score*=0.0625f;
        const float next_max=fmaxf(running_max,score);
        const float old_scale=__expf(running_max-next_max);
        const float weight=__expf(score-next_max);
        #pragma unroll
        for(int i=0;i<kDimensionsPerLane;++i)
            accumulated[i]=old_scale*accumulated[i]+weight*
                __half2float(__ushort_as_half(v_cache[cache_base+i]));
        denominator=old_scale*denominator+weight;
        running_max=next_max;
    }

    const std::size_t output_base=(static_cast<std::size_t>(query)*kQHeads+
        kv_head*kHeadsPerKV+head)*kHeadDim+dimension_base;
    #pragma unroll
    for(int i=0;i<kDimensionsPerLane;++i)
        output[output_base+i]=__half_as_ushort(__float2half_rn(
            accumulated[i]/denominator));
}

// T73 NUMERIC_CANDIDATE. Four adjacent causal query rows share the K load
// for each KV head. This keeps the six sibling GQA score chains independent,
// but removes the one-CTA-per-row duplication that dominates the ordinary
// Fast prefill attention route. The online normalization order is intentionally
// different from the exact/reference score-plane route, so this kernel is
// reachable only through its separate numerical Fast gate.
template<int RowsPerBlock,int KeysPerTile=256,int DimsPerThread=1>
__global__ void attention_cached_gqa_six_online_rows_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset) {
    static_assert(RowsPerBlock==2 || RowsPerBlock==4 || RowsPerBlock==8);
    static_assert(RowsPerBlock!=8 || KeysPerTile==128);
    static_assert(DimsPerThread==1 || DimsPerThread==2);
    static_assert(kHeadDim % DimsPerThread == 0);
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kKeysPerTile=KeysPerTile;
    const int query_group=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int query_base=query_group*RowsPerBlock;
    const int tid=threadIdx.x;
    const int dim_base=tid*DimsPerThread;
    if(query_base>=rows || dim_base>=kHeadDim) return;
    const int active_rows=min(RowsPerBlock,rows-query_base);
    const int base=(position_device?*position_device:position)+query_offset;
    const int maximum_count=base+query_base+active_rows;
    if(base+query_base+1<1 || maximum_count>capacity) return;

    __shared__ float query_values[RowsPerBlock][kHeadsPerKV][kHeadDim];
    __shared__ float score_tile[RowsPerBlock][kHeadsPerKV][kKeysPerTile];
    __shared__ float running_max[RowsPerBlock][kHeadsPerKV];
    __shared__ float denominator[RowsPerBlock][kHeadsPerKV];
    __shared__ float old_scale[RowsPerBlock][kHeadsPerKV];

    #pragma unroll
    for(int row=0;row<RowsPerBlock;++row) {
        if(row<active_rows) {
            #pragma unroll
            for(int head=0;head<kHeadsPerKV;++head)
                #pragma unroll
                for(int lane=0;lane<DimsPerThread;++lane)
                    query_values[row][head][dim_base+lane]=__half2float(
                        __ushort_as_half(q[((query_base+row)*kQHeads+
                            kv_head*kHeadsPerKV+head)*kHeadDim+
                            dim_base+lane]));
        }
    }
    if(tid<active_rows*kHeadsPerKV) {
        const int row=tid/kHeadsPerKV;
        const int head=tid%kHeadsPerKV;
        running_max[row][head]=-3.402823466e+38F;
        denominator[row][head]=0.0f;
        old_scale[row][head]=0.0f;
    }
    float accumulated[RowsPerBlock][kHeadsPerKV][DimsPerThread]={};
    __syncthreads();

    for(int first=0;first<maximum_count;first+=kKeysPerTile) {
        const int extent_max=min(kKeysPerTile,maximum_count-first);
        const int key=first+tid;
        if(tid<extent_max) {
            const auto* represented_key=k_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
            #pragma unroll
            for(int row=0;row<RowsPerBlock;++row) {
                if(row<active_rows && key<base+query_base+row+1) {
                    float dot[kHeadsPerKV]={};
                    for(int d=0;d<kHeadDim;d+=2) {
                        const float2 represented=__half22float2(
                            *reinterpret_cast<const __half2*>(represented_key+d));
                        #pragma unroll
                        for(int head=0;head<kHeadsPerKV;++head) {
                            dot[head]+=query_values[row][head][d]*represented.x;
                            dot[head]+=query_values[row][head][d+1]*represented.y;
                        }
                    }
                    #pragma unroll
                    for(int head=0;head<kHeadsPerKV;++head)
                        score_tile[row][head][tid]=dot[head]*0.0625f;
                }
            }
        }
        __syncthreads();

        if(tid<active_rows*kHeadsPerKV) {
            const int row=tid/kHeadsPerKV;
            const int head=tid%kHeadsPerKV;
            const int extent=min(kKeysPerTile,
                max(0,base+query_base+row+1-first));
            float tile_max=-3.402823466e+38F;
            for(int offset=0;offset<extent;++offset)
                tile_max=fmaxf(tile_max,score_tile[row][head][offset]);
            const float next_max=fmaxf(running_max[row][head],tile_max);
            old_scale[row][head]=denominator[row][head]==0.0f?0.0f:
                expf(running_max[row][head]-next_max);
            running_max[row][head]=next_max;
        }
        __syncthreads();

        if(tid<extent_max) {
            #pragma unroll
            for(int row=0;row<RowsPerBlock;++row) {
                if(row<active_rows && key<base+query_base+row+1) {
                    #pragma unroll
                    for(int head=0;head<kHeadsPerKV;++head)
                        score_tile[row][head][tid]=expf(
                            score_tile[row][head][tid]-running_max[row][head]);
                }
            }
        }
        __syncthreads();

        if(tid<active_rows*kHeadsPerKV) {
            const int row=tid/kHeadsPerKV;
            const int head=tid%kHeadsPerKV;
            const int extent=min(kKeysPerTile,
                max(0,base+query_base+row+1-first));
            float tile_sum=0.0f;
            for(int offset=0;offset<extent;++offset)
                tile_sum+=score_tile[row][head][offset];
            denominator[row][head]=denominator[row][head]*old_scale[row][head]+
                tile_sum;
        }
        #pragma unroll
        for(int row=0;row<RowsPerBlock;++row)
            if(row<active_rows)
                for(int head=0;head<kHeadsPerKV;++head)
                    for(int lane=0;lane<DimsPerThread;++lane)
                        accumulated[row][head][lane]*=old_scale[row][head];

        for(int offset=0;offset<extent_max;++offset) {
            const int current_key=first+offset;
            #pragma unroll
            for(int lane=0;lane<DimsPerThread;++lane) {
                const int dim=dim_base+lane;
                const float represented_value=__half2float(__ushort_as_half(
                    v_cache[((static_cast<std::size_t>(current_key)*kKVHeads+
                        kv_head)*kHeadDim)+dim]));
                #pragma unroll
                for(int row=0;row<RowsPerBlock;++row) {
                    if(row<active_rows && current_key<base+query_base+row+1) {
                        #pragma unroll
                        for(int head=0;head<kHeadsPerKV;++head)
                            accumulated[row][head][lane]+=score_tile[row][head][offset]*
                                represented_value;
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int row=0;row<RowsPerBlock;++row) {
        if(row<active_rows) {
            #pragma unroll
            for(int head=0;head<kHeadsPerKV;++head)
                #pragma unroll
                for(int lane=0;lane<DimsPerThread;++lane)
                    output[((query_base+row)*kQHeads+kv_head*kHeadsPerKV+head)*
                        kHeadDim+dim_base+lane]=__half_as_ushort(__float2half_rn(
                            accumulated[row][head][lane]/denominator[row][head]));
        }
    }
}

// T75 NUMERIC_CANDIDATE.  The pinned Mia prefill path uses a tensor-core
// tiled attention kernel (M=64, N=32 for this head dimension), while the
// established native Fast candidates are scalar dot/value loops.  This
// separately gated kernel keeps one query-head block at M=16 and streams
// N=64 represented FP16-KV rows.  WMMA handles both Q*K^T and P*V; the
// online softmax and causal boundary remain explicit so the exact/reference
// route and its score-plane ownership are untouched.  Arithmetic order is
// intentionally a Fast numerical candidate, never an exact route.
template<int BlockM=16,int BlockN=32,int HeadsPerBlock=2>
__global__ void attention_cached_gqa_six_wmma_prefill_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset) {
    static_assert(BlockM==16 && BlockN==32 && HeadsPerBlock==2);
    constexpr int kValueGroups=kHeadDim/64;
    const int query_base=static_cast<int>(blockIdx.x)*BlockM;
    const int kv_head=static_cast<int>(blockIdx.y);
    const int head_group=static_cast<int>(blockIdx.z);
    const int tid=static_cast<int>(threadIdx.x);
    const int warp=tid/32;
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int q_head_base=kv_head*kHeadsPerKV+head_group*HeadsPerBlock;
    if(query_base>=rows || kv_head>=kKVHeads ||
       head_group*HeadsPerBlock>=kHeadsPerKV) return;
    const int active_rows=min(BlockM,rows-query_base);
    const int base=(position_device?*position_device:position)+query_offset;
    const int maximum_count=base+query_base+active_rows;
    if(base+query_base+1<1 || maximum_count>capacity) return;

    __shared__ half q_tile[HeadsPerBlock][BlockM][kHeadDim];
    __shared__ half k_tile[BlockN][kHeadDim];
    __shared__ half v_tile[BlockN][kHeadDim];
    __shared__ float score_tile[HeadsPerBlock][BlockM][BlockN];
    __shared__ half probability_tile[HeadsPerBlock][BlockM][BlockN];
    __shared__ float accumulated[HeadsPerBlock][BlockM][kHeadDim];
    __shared__ float running_max[HeadsPerBlock][BlockM];
    __shared__ float denominator[HeadsPerBlock][BlockM];
    __shared__ float old_scale[HeadsPerBlock][BlockM];

    for(int index=tid;index<HeadsPerBlock*BlockM*kHeadDim;index+=blockDim.x) {
        const int head=index/(BlockM*kHeadDim);
        const int rem=index%(BlockM*kHeadDim);
        const int row=rem/kHeadDim;
        const int dim=rem%kHeadDim;
        q_tile[head][row][dim]=row<active_rows?
            __ushort_as_half(q[((query_base+row)*kQHeads+q_head_base+head)*
                kHeadDim+dim]):__float2half(0.0f);
        accumulated[head][row][dim]=0.0f;
    }
    for(int index=tid;index<HeadsPerBlock*BlockM;index+=blockDim.x) {
        const int head=index/BlockM;
        const int row=index%BlockM;
        running_max[head][row]=-3.402823466e+38F;
        denominator[head][row]=0.0f;
        old_scale[head][row]=0.0f;
    }
    __syncthreads();

    for(int first=0;first<maximum_count;first+=BlockN) {
        for(int index=tid;index<BlockN*kHeadDim;index+=blockDim.x) {
            const int key_offset=index/kHeadDim;
            const int dim=index%kHeadDim;
            const int key=first+key_offset;
            if(key<maximum_count && key<capacity) {
                k_tile[key_offset][dim]=__ushort_as_half(
                    k_cache[(static_cast<std::size_t>(key)*kKVHeads+
                        kv_head)*kHeadDim+dim]);
                v_tile[key_offset][dim]=__ushort_as_half(
                    v_cache[(static_cast<std::size_t>(key)*kKVHeads+
                        kv_head)*kHeadDim+dim]);
            } else {
                k_tile[key_offset][dim]=__float2half(0.0f);
                v_tile[key_offset][dim]=__float2half(0.0f);
            }
        }
        __syncthreads();

        if(warp<HeadsPerBlock*4) {
            const int head=warp/4;
            const int warp_in_head=warp%4;
            if(warp_in_head < BlockN/16) {
                const int n_base=warp_in_head*16;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,
                nvcuda::wmma::row_major> a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,
                nvcuda::wmma::col_major> b;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> c;
            nvcuda::wmma::fill_fragment(c,0.0f);
            #pragma unroll
            for(int k_tile_base=0;k_tile_base<kHeadDim;k_tile_base+=16) {
                nvcuda::wmma::load_matrix_sync(
                    a,&q_tile[head][0][k_tile_base],kHeadDim);
                nvcuda::wmma::load_matrix_sync(
                    b,&k_tile[n_base][k_tile_base],kHeadDim);
                nvcuda::wmma::mma_sync(c,a,b,c);
            }
            nvcuda::wmma::store_matrix_sync(
                &score_tile[head][0][n_base],c,BlockN,
                nvcuda::wmma::mem_row_major);
            }
        }
        __syncthreads();

        for(int head=0;head<HeadsPerBlock;++head) {
            if(tid<BlockM) {
                float tile_max=-3.402823466e+38F;
                for(int offset=0;offset<BlockN;++offset) {
                    const int key=first+offset;
                    if(tid<active_rows && offset<maximum_count-first &&
                       key<=base+query_base+tid)
                        tile_max=fmaxf(tile_max,
                            score_tile[head][tid][offset]*0.0625f);
                }
                const float next_max=fmaxf(running_max[head][tid],tile_max);
                old_scale[head][tid]=denominator[head][tid]==0.0f?0.0f:
                    exp2f((running_max[head][tid]-next_max)*
                        1.4426950408889634f);
                running_max[head][tid]=next_max;
            }
            __syncthreads();

            for(int index=tid;index<BlockM*BlockN;index+=blockDim.x) {
                const int row=index/BlockN;
                const int offset=index%BlockN;
                const int key=first+offset;
                const bool valid=row<active_rows &&
                    offset<maximum_count-first &&
                    key<=base+query_base+row;
                const float score=score_tile[head][row][offset]*0.0625f;
                probability_tile[head][row][offset]=__float2half(
                    valid?exp2f((score-running_max[head][row])*
                        1.4426950408889634f):0.0f);
            }
            __syncthreads();

            if(tid<BlockM) {
                float tile_sum=0.0f;
                const int extent=min(BlockN,maximum_count-first);
                for(int offset=0;offset<extent;++offset)
                    tile_sum+=__half2float(probability_tile[head][tid][offset]);
                denominator[head][tid]=denominator[head][tid]*
                    old_scale[head][tid]+tile_sum;
            }
            __syncthreads();

            for(int index=tid;index<BlockM*kHeadDim;index+=blockDim.x) {
                const int row=index/kHeadDim;
                accumulated[head][row][index%kHeadDim]*=
                    old_scale[head][row];
            }
            __syncthreads();

            const int head_warp=warp-head*4;
            if(head_warp>=0 && head_warp<4) {
                const int k_base=head_warp*64;
                nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float>
                    c[kValueGroups];
                #pragma unroll
                for(int group=0;group<kValueGroups;++group)
                    nvcuda::wmma::load_matrix_sync(
                        c[group],&accumulated[head][0][k_base+group*16],
                        kHeadDim,nvcuda::wmma::mem_row_major);
                #pragma unroll
                for(int n_base=0;n_base<BlockN;n_base+=16) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,
                        nvcuda::wmma::row_major> a;
                    nvcuda::wmma::load_matrix_sync(
                        a,&probability_tile[head][0][n_base],BlockN);
                    #pragma unroll
                    for(int group=0;group<kValueGroups;++group) {
                        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,
                            nvcuda::wmma::row_major> b;
                        nvcuda::wmma::load_matrix_sync(
                            b,&v_tile[n_base][k_base+group*16],kHeadDim);
                        nvcuda::wmma::mma_sync(c[group],a,b,c[group]);
                    }
                }
                #pragma unroll
                for(int group=0;group<kValueGroups;++group)
                    nvcuda::wmma::store_matrix_sync(
                        &accumulated[head][0][k_base+group*16],c[group],
                        kHeadDim,nvcuda::wmma::mem_row_major);
            }
            __syncthreads();
        }
    }

    for(int index=tid;index<HeadsPerBlock*BlockM*kHeadDim;index+=blockDim.x) {
        const int head=index/(BlockM*kHeadDim);
        const int rem=index%(BlockM*kHeadDim);
        const int row=rem/kHeadDim;
        const int dim=rem%kHeadDim;
        if(row<active_rows)
            output[((query_base+row)*kQHeads+q_head_base+head)*kHeadDim+dim]=
                __half_as_ushort(__float2half_rn(
                    accumulated[head][row][dim]/denominator[head][row]));
    }
}

// T77 NUMERIC_CANDIDATE.  The established WMMA candidate uses M=16 and two
// query heads per CTA.  This corrected variant doubles the query tile while
// assigning one query head per CTA, keeping its static shared-memory footprint
// within the physical-C1 limit while reusing each K/V tile across 32 rows. It is
// construction-latched and default-off: the exact score/value path remains
// the fallback for every unsupported or disabled configuration.
// VectorLoads moves the same represented Q/K/V halves into the same shared
// tiles with independent 16-byte loads instead of a dependent 2-byte loop.
template<int TileStride,int SplitCount=1,int BlockM=32,bool VectorLoads=false>
__global__ void attention_cached_gqa_six_wmma32_prefill_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset,
    float* split_output,float* split_stats) {
    static_assert(BlockM==32 || BlockM==64);
    constexpr int BlockN=32;
    constexpr int HeadsPerBlock=1;
    constexpr int ActiveWarps=BlockM==64?8:4;
    constexpr int kValueGroups=kHeadDim/(ActiveWarps*16);
    const int query_base=static_cast<int>(blockIdx.x)*BlockM;
    const int kv_head=static_cast<int>(blockIdx.y);
    static_assert(SplitCount==1 || SplitCount==2 || SplitCount==4);
    const int head_group=static_cast<int>(blockIdx.z)/SplitCount;
    const int split=static_cast<int>(blockIdx.z)%SplitCount;
    const int tid=static_cast<int>(threadIdx.x);
    const int warp=tid/32;
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int q_head_base=kv_head*kHeadsPerKV+head_group*HeadsPerBlock;
    if(query_base>=rows || kv_head>=kKVHeads ||
       head_group*HeadsPerBlock>=kHeadsPerKV) return;
    const int active_rows=min(BlockM,rows-query_base);
    const int base=(position_device?*position_device:position)+query_offset;
    const int maximum_count=base+query_base+active_rows;
    if(base+query_base+1<1 || maximum_count>capacity) return;
    const int segment_tiles=(maximum_count+SplitCount*BlockN-1)/
        (SplitCount*BlockN);
    const int first_begin=SplitCount==1?0:
        min(split*segment_tiles*BlockN,maximum_count);
    const int first_end=SplitCount==1?maximum_count:
        min((split+1)*segment_tiles*BlockN,maximum_count);

    // M64 streams Q from its original row-major positions and overlays K/V.
    // That keeps the FP32 online output accumulator within SM120 shared memory.
    __shared__ __align__(16) half q_tile[HeadsPerBlock][BlockM==64?1:BlockM][TileStride];
    __shared__ __align__(16) half k_tile[BlockN][TileStride];
    __shared__ __align__(16) half v_tile[BlockM==64?1:BlockN][TileStride];
    static_assert(!VectorLoads || (BlockM==32 && HeadsPerBlock==1 && TileStride%8==0));
    __shared__ float score_tile[HeadsPerBlock][BlockM][BlockN];
    __shared__ half probability_tile[HeadsPerBlock][BlockM][BlockN];
    __shared__ float accumulated[HeadsPerBlock][BlockM][kHeadDim];
    __shared__ float running_max[HeadsPerBlock][BlockM];
    __shared__ float denominator[HeadsPerBlock][BlockM];
    __shared__ float old_scale[HeadsPerBlock][BlockM];

    for(int index=tid;index<HeadsPerBlock*BlockM*kHeadDim;index+=blockDim.x) {
        const int head=index/(BlockM*kHeadDim);
        const int rem=index%(BlockM*kHeadDim);
        const int row=rem/kHeadDim;
        const int dim=rem%kHeadDim;
        if constexpr(BlockM==32)
            q_tile[head][row][dim]=row<active_rows?
                __ushort_as_half(q[((query_base+row)*kQHeads+q_head_base+head)*
                    kHeadDim+dim]):__float2half(0.0f);
        accumulated[head][row][dim]=0.0f;
    }
    for(int index=tid;index<HeadsPerBlock*BlockM;index+=blockDim.x) {
        const int head=index/BlockM;
        const int row=index%BlockM;
        running_max[head][row]=-3.402823466e+38F;
        denominator[head][row]=0.0f;
        old_scale[head][row]=0.0f;
    }
    __syncthreads();

    for(int first=first_begin;first<first_end;first+=BlockN) {
        if constexpr(VectorLoads) {
            constexpr int kVectors=kHeadDim/8;
            for(int index=tid;index<BlockN*kVectors;index+=blockDim.x) {
                const int key_offset=index/kVectors;
                const int dim=(index%kVectors)*8;
                const int key=first+key_offset;
                uint4 key_bits=make_uint4(0,0,0,0),value_bits=make_uint4(0,0,0,0);
                if(key<first_end && key<capacity) {
                    const std::size_t offset=(static_cast<std::size_t>(key)*kKVHeads+
                        kv_head)*kHeadDim+dim;
                    key_bits=*reinterpret_cast<const uint4*>(k_cache+offset);
                    value_bits=*reinterpret_cast<const uint4*>(v_cache+offset);
                }
                *reinterpret_cast<uint4*>(&k_tile[key_offset][dim])=key_bits;
                *reinterpret_cast<uint4*>(&v_tile[key_offset][dim])=value_bits;
            }
        } else
        for(int index=tid;index<BlockN*kHeadDim;index+=blockDim.x) {
            const int key_offset=index/kHeadDim;
            const int dim=index%kHeadDim;
            const int key=first+key_offset;
            if(key<first_end && key<capacity) {
                k_tile[key_offset][dim]=__ushort_as_half(
                    k_cache[(static_cast<std::size_t>(key)*kKVHeads+
                        kv_head)*kHeadDim+dim]);
                if constexpr(BlockM==32)
                    v_tile[key_offset][dim]=__ushort_as_half(
                        v_cache[(static_cast<std::size_t>(key)*kKVHeads+
                            kv_head)*kHeadDim+dim]);
            } else {
                k_tile[key_offset][dim]=__float2half(0.0f);
                if constexpr(BlockM==32)
                    v_tile[key_offset][dim]=__float2half(0.0f);
            }
        }
        __syncthreads();

        // M64 assigns all eight warps to four M tiles x two N tiles.
        if(warp<HeadsPerBlock*(BlockM/16)*2) {
            const int head=warp/((BlockM/16)*2);
            const int task=warp%((BlockM/16)*2);
            const int m_base=(task/2)*16;
            const int n_base=(task%2)*16;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,
                nvcuda::wmma::row_major> a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,
                nvcuda::wmma::col_major> b;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> c;
            nvcuda::wmma::fill_fragment(c,0.0f);
            #pragma unroll
            for(int k_tile_base=0;k_tile_base<kHeadDim;k_tile_base+=16) {
                if constexpr(BlockM==64)
                    nvcuda::wmma::load_matrix_sync(a,
                        reinterpret_cast<const half*>(q)+
                        ((query_base+m_base)*kQHeads+q_head_base+head)*
                            kHeadDim+k_tile_base,kQHeads*kHeadDim);
                else
                    nvcuda::wmma::load_matrix_sync(
                        a,&q_tile[head][m_base][k_tile_base],TileStride);
                nvcuda::wmma::load_matrix_sync(
                    b,&k_tile[n_base][k_tile_base],TileStride);
                nvcuda::wmma::mma_sync(c,a,b,c);
            }
            nvcuda::wmma::store_matrix_sync(
                &score_tile[head][m_base][n_base],c,BlockN,
                nvcuda::wmma::mem_row_major);
        }
        __syncthreads();

        for(int head=0;head<HeadsPerBlock;++head) {
            if(tid<BlockM) {
                float tile_max=-3.402823466e+38F;
                for(int offset=0;offset<BlockN;++offset) {
                    const int key=first+offset;
                    if(tid<active_rows && offset<first_end-first &&
                       key<=base+query_base+tid)
                        tile_max=fmaxf(tile_max,
                            score_tile[head][tid][offset]*0.0625f);
                }
                const float next_max=fmaxf(running_max[head][tid],tile_max);
                old_scale[head][tid]=denominator[head][tid]==0.0f?0.0f:
                    exp2f((running_max[head][tid]-next_max)*
                        1.4426950408889634f);
                running_max[head][tid]=next_max;
            }
            __syncthreads();

            for(int index=tid;index<BlockM*BlockN;index+=blockDim.x) {
                const int row=index/BlockN;
                const int offset=index%BlockN;
                const int key=first+offset;
                const bool valid=row<active_rows &&
                    offset<first_end-first &&
                    key<=base+query_base+row;
                const float score=score_tile[head][row][offset]*0.0625f;
                probability_tile[head][row][offset]=__float2half(
                    valid?exp2f((score-running_max[head][row])*
                        1.4426950408889634f):0.0f);
            }
            __syncthreads();

            if(tid<BlockM) {
                float tile_sum=0.0f;
                const int extent=min(BlockN,first_end-first);
                for(int offset=0;offset<extent;++offset)
                    tile_sum+=__half2float(probability_tile[head][tid][offset]);
                denominator[head][tid]=denominator[head][tid]*
                    old_scale[head][tid]+tile_sum;
            }
            __syncthreads();

            for(int index=tid;index<BlockM*kHeadDim;index+=blockDim.x) {
                const int row=index/kHeadDim;
                accumulated[head][row][index%kHeadDim]*=
                    old_scale[head][row];
            }
            __syncthreads();

            if constexpr(BlockM==64) {
                // QK is complete. Reuse the K tile as V storage before PV.
                for(int index=tid;index<BlockN*kHeadDim;index+=blockDim.x) {
                    const int key_offset=index/kHeadDim;
                    const int dim=index%kHeadDim;
                    const int key=first+key_offset;
                    k_tile[key_offset][dim]=key<first_end && key<capacity?
                        __ushort_as_half(v_cache[
                            (static_cast<std::size_t>(key)*kKVHeads+
                                kv_head)*kHeadDim+dim]):__float2half(0.0f);
                }
                __syncthreads();
            }

            const int head_warp=warp-head*ActiveWarps;
            if(head_warp>=0 && head_warp<ActiveWarps) {
                const int k_base=head_warp*(kHeadDim/ActiveWarps);
                const half* value_tile=BlockM==64?
                    &k_tile[0][0]:&v_tile[0][0];
                for(int m_base=0;m_base<BlockM;m_base+=16) {
                    nvcuda::wmma::fragment<
                        nvcuda::wmma::accumulator,16,16,16,float> c[kValueGroups];
                    #pragma unroll
                    for(int group=0;group<kValueGroups;++group)
                        nvcuda::wmma::load_matrix_sync(
                            c[group],&accumulated[head][m_base][k_base+group*16],
                            kHeadDim,nvcuda::wmma::mem_row_major);
                    #pragma unroll
                    for(int n_base=0;n_base<BlockN;n_base+=16) {
                        nvcuda::wmma::fragment<
                            nvcuda::wmma::matrix_a,16,16,16,half,
                            nvcuda::wmma::row_major> a;
                        nvcuda::wmma::load_matrix_sync(
                            a,&probability_tile[head][m_base][n_base],BlockN);
                        #pragma unroll
                        for(int group=0;group<kValueGroups;++group) {
                            nvcuda::wmma::fragment<
                                nvcuda::wmma::matrix_b,16,16,16,half,
                                nvcuda::wmma::row_major> b;
                            nvcuda::wmma::load_matrix_sync(
                                b,value_tile+n_base*TileStride+
                                    k_base+group*16,TileStride);
                            nvcuda::wmma::mma_sync(c[group],a,b,c[group]);
                        }
                    }
                    #pragma unroll
                    for(int group=0;group<kValueGroups;++group)
                        nvcuda::wmma::store_matrix_sync(
                            &accumulated[head][m_base][k_base+group*16],c[group],
                            kHeadDim,nvcuda::wmma::mem_row_major);
                }
            }
            __syncthreads();
        }
    }

    for(int index=tid;index<HeadsPerBlock*BlockM*kHeadDim;index+=blockDim.x) {
        const int head=index/(BlockM*kHeadDim);
        const int rem=index%(BlockM*kHeadDim);
        const int row=rem/kHeadDim;
        const int dim=rem%kHeadDim;
        if(row<active_rows) {
            const int element=((query_base+row)*kQHeads+q_head_base+head)*
                kHeadDim+dim;
            const float normalized=denominator[head][row]>0.0f?
                accumulated[head][row][dim]/denominator[head][row]:0.0f;
            if constexpr(SplitCount>1)
                split_output[static_cast<std::size_t>(split)*rows*kQHeads*
                    kHeadDim+element]=normalized;
            else
                output[element]=__half_as_ushort(__float2half_rn(normalized));
        }
    }
    if constexpr(SplitCount>1) {
        for(int index=tid;index<HeadsPerBlock*active_rows;index+=blockDim.x) {
            const int head=index/active_rows;
            const int row=index%active_rows;
            const std::size_t stat=(static_cast<std::size_t>(split)*rows*kQHeads+
                (query_base+row)*kQHeads+q_head_base+head)*2;
            split_stats[stat]=running_max[head][row];
            split_stats[stat+1]=denominator[head][row];
        }
    }
}

// Separate numeric candidate: a warp owns 16 query rows x 128 value columns
// across the entire key scan. Unlike the selected WMMA64 path, its FP32 output
// fragments stay in registers until the final store. A CTA-wide maximum gives
// every fragment one rescale factor without assuming a WMMA fragment lane map.
// The changed FP16 probability range requires an independent quality gate.
template<int TileStride,int SplitCount,int HeadsPerCTA=1,int BlockN=32>
__global__ void attention_cached_gqa_six_wmma64_register_prefill_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset,
    float* split_output,float* split_stats) {
    static_assert(SplitCount==1 || SplitCount==2);
    static_assert(HeadsPerCTA==1 || HeadsPerCTA==2);
    static_assert(BlockN==32 || BlockN==64);
    constexpr int BlockM=64/HeadsPerCTA;
    constexpr int ValueColumnsPerWarp=128,WarpsPerHead=8/HeadsPerCTA;
    constexpr int KeyStage=32;
    constexpr int FinalColumns=BlockN==64?64:ValueColumnsPerWarp;
    const int query_base=static_cast<int>(blockIdx.x)*BlockM;
    const int kv_head=static_cast<int>(blockIdx.y);
    constexpr int heads_per_kv=kQHeads/kKVHeads;
    const int head_group=static_cast<int>(blockIdx.z)/SplitCount;
    const int split=static_cast<int>(blockIdx.z)%SplitCount;
    const int q_head_base=kv_head*heads_per_kv+head_group*HeadsPerCTA;
    const int tid=static_cast<int>(threadIdx.x);
    const int warp=tid/32;
    const int lane=tid%32;
    if(query_base+BlockM>rows || kv_head>=kKVHeads ||
       head_group*HeadsPerCTA>=heads_per_kv) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int maximum_count=base+query_base+BlockM;
    if(base+query_base+1<1 || maximum_count>capacity) return;
    const int segment_tiles=(maximum_count+SplitCount*BlockN-1)/
        (SplitCount*BlockN);
    const int first_begin=SplitCount==1?0:
        min(split*segment_tiles*BlockN,maximum_count);
    const int first_end=SplitCount==1?maximum_count:
        min((split+1)*segment_tiles*BlockN,maximum_count);

    __shared__ __align__(16) half key_value_tile[KeyStage][TileStride];
    __shared__ __align__(16) float scores[HeadsPerCTA][BlockM][BlockN];
    __shared__ __align__(16) half probabilities[HeadsPerCTA][BlockM][BlockN];
    __shared__ float denominator[HeadsPerCTA][BlockM];
    __shared__ float tile_row_max[HeadsPerCTA][BlockM];
    __shared__ float current_max[HeadsPerCTA],rescale[HeadsPerCTA];
    __shared__ __align__(16) float final_tile[16][FinalColumns];
    if(tid<HeadsPerCTA*BlockM)
        denominator[tid/BlockM][tid%BlockM]=0.0f;
    if(tid<HeadsPerCTA) current_max[tid]=-3.402823466e+38F;
    using Accumulator=nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,16,16,16,float>;
    Accumulator value[ValueColumnsPerWarp/16];
    #pragma unroll
    for(int group=0;group<ValueColumnsPerWarp/16;++group)
        nvcuda::wmma::fill_fragment(value[group],0.0f);
    __syncthreads();

    for(int first=first_begin;first<first_end;first+=BlockN) {
        const int head=warp/WarpsPerHead;
        const int task=warp%WarpsPerHead;
        const int m_base=(task/2)*16;
        // A 64-key statistics tile uses two bounded 32-key K/V stages. The
        // larger score plane halves online-max and rescale iterations without
        // doubling shared K/V storage or changing register-owned output.
        for(int key_window=0;key_window<BlockN;key_window+=KeyStage) {
            for(int index=tid;index<KeyStage*kHeadDim;index+=blockDim.x) {
                const int key_offset=index/kHeadDim;
                const int dimension=index%kHeadDim;
                const int key=first+key_window+key_offset;
                key_value_tile[key_offset][dimension]=
                    key<first_end && key<capacity?
                    __ushort_as_half(k_cache[
                        (static_cast<std::size_t>(key)*kKVHeads+kv_head)*
                            kHeadDim+dimension]):__float2half(0.0f);
            }
            __syncthreads();
            const int n_base=key_window+(task%2)*16;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,
                nvcuda::wmma::row_major> query_fragment;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,
                nvcuda::wmma::col_major> key_fragment;
            Accumulator score_fragment;
            nvcuda::wmma::fill_fragment(score_fragment,0.0f);
            #pragma unroll
            for(int dimension=0;dimension<kHeadDim;dimension+=16) {
                nvcuda::wmma::load_matrix_sync(query_fragment,
                    reinterpret_cast<const half*>(q)+
                    ((query_base+m_base)*kQHeads+q_head_base+head)*
                        kHeadDim+dimension,
                    kQHeads*kHeadDim);
                nvcuda::wmma::load_matrix_sync(key_fragment,
                    &key_value_tile[(task%2)*16][dimension],TileStride);
                nvcuda::wmma::mma_sync(score_fragment,query_fragment,
                    key_fragment,score_fragment);
            }
            nvcuda::wmma::store_matrix_sync(&scores[head][m_base][n_base],
                score_fragment,BlockN,nvcuda::wmma::mem_row_major);
            __syncthreads();
        }

        if(tid<HeadsPerCTA*BlockM) {
            const int score_head=tid/BlockM;
            const int row=tid%BlockM;
            float maximum=-3.402823466e+38F;
            #pragma unroll
            for(int offset=0;offset<BlockN;++offset) {
                const int key=first+offset;
                if(key<first_end && key<=base+query_base+row)
                    maximum=fmaxf(maximum,
                        scores[score_head][row][offset]*0.0625f);
            }
            tile_row_max[score_head][row]=maximum;
        }
        __syncthreads();
        if(tid<HeadsPerCTA) {
            float next_max=current_max[tid];
            #pragma unroll
            for(int row=0;row<BlockM;++row)
                next_max=fmaxf(next_max,tile_row_max[tid][row]);
            rescale[tid]=next_max==current_max[tid]?1.0f:
                (current_max[tid]< -1.0e30f?0.0f:
                    exp2f((current_max[tid]-next_max)*
                        1.4426950408889634f));
            current_max[tid]=next_max;
        }
        __syncthreads();
        for(int index=tid;index<HeadsPerCTA*BlockM*BlockN;
            index+=blockDim.x) {
            const int probability_head=index/(BlockM*BlockN);
            const int row=(index/BlockN)%BlockM;
            const int offset=index%BlockN;
            const int key=first+offset;
            const bool valid=key<first_end && key<=base+query_base+row;
            probabilities[probability_head][row][offset]=__float2half_rn(
                valid?exp2f((scores[probability_head][row][offset]*
                    0.0625f-current_max[probability_head])*
                    1.4426950408889634f):0.0f);
        }
        __syncthreads();
        if(tid<HeadsPerCTA*BlockM) {
            const int probability_head=tid/BlockM;
            const int row=tid%BlockM;
            float sum=0.0f;
            const int extent=min(BlockN,first_end-first);
            for(int offset=0;offset<extent;++offset)
                sum+=__half2float(
                    probabilities[probability_head][row][offset]);
            denominator[probability_head][row]=
                denominator[probability_head][row]*
                    rescale[probability_head]+sum;
        }
        #pragma unroll
        for(int group=0;group<ValueColumnsPerWarp/16;++group)
            #pragma unroll
            for(int element=0;element<value[group].num_elements;++element)
                value[group].x[element]*=rescale[warp/WarpsPerHead];
        __syncthreads();

        const int value_head=warp/WarpsPerHead;
        const int value_m_base=((warp%WarpsPerHead)/2)*16;
        const int value_d_base=(warp%2)*ValueColumnsPerWarp;
        // QK has consumed the key stage. Reuse it for the matching V half.
        for(int key_window=0;key_window<BlockN;key_window+=KeyStage) {
            for(int index=tid;index<KeyStage*kHeadDim;index+=blockDim.x) {
                const int key_offset=index/kHeadDim;
                const int dimension=index%kHeadDim;
                const int key=first+key_window+key_offset;
                key_value_tile[key_offset][dimension]=
                    key<first_end && key<capacity?
                    __ushort_as_half(v_cache[
                        (static_cast<std::size_t>(key)*kKVHeads+kv_head)*
                            kHeadDim+dimension]):__float2half(0.0f);
            }
            __syncthreads();
            #pragma unroll
            for(int probability_offset=0;probability_offset<KeyStage;
                probability_offset+=16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,half,
                    nvcuda::wmma::row_major> probability_fragment;
                nvcuda::wmma::load_matrix_sync(probability_fragment,
                    &probabilities[value_head][value_m_base]
                        [key_window+probability_offset],BlockN);
                #pragma unroll
                for(int group=0;group<ValueColumnsPerWarp/16;++group) {
                    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,half,
                        nvcuda::wmma::row_major> v_fragment;
                    nvcuda::wmma::load_matrix_sync(v_fragment,
                        &key_value_tile[probability_offset]
                            [value_d_base+group*16],TileStride);
                    nvcuda::wmma::mma_sync(value[group],probability_fragment,
                        v_fragment,value[group]);
                }
            }
            __syncthreads();
        }
    }

    // Reuse one 16 x 128 tile across warps only once, after the key scan.
    // Every warp participates in the CTA barriers, even on an empty split.
    for(int owner=0;owner<8;++owner) {
        for(int part=0;part<ValueColumnsPerWarp/FinalColumns;++part) {
            if(warp==owner) {
                #pragma unroll
                for(int group=0;group<FinalColumns/16;++group)
                    nvcuda::wmma::store_matrix_sync(
                        &final_tile[0][group*16],
                        value[part*(FinalColumns/16)+group],
                        FinalColumns,nvcuda::wmma::mem_row_major);
            }
            __syncthreads();
            if(warp==owner) {
                const int output_head=warp/WarpsPerHead;
                const int m_base=((warp%WarpsPerHead)/2)*16;
                const int d_base=(warp%2)*ValueColumnsPerWarp+part*FinalColumns;
                for(int index=lane;index<16*FinalColumns;index+=32) {
                    const int row=m_base+index/FinalColumns;
                    const int dimension=d_base+index%FinalColumns;
                    const int element=((query_base+row)*kQHeads+
                        q_head_base+output_head)*kHeadDim+dimension;
                    const float result=denominator[output_head][row]>0.0f?
                        final_tile[index/FinalColumns]
                            [index%FinalColumns]/
                                denominator[output_head][row]:0.0f;
                    if constexpr(SplitCount==2)
                        split_output[static_cast<std::size_t>(split)*rows*
                            kQHeads*kHeadDim+element]=result;
                    else
                        output[element]=__half_as_ushort(__float2half_rn(result));
                }
            }
            __syncthreads();
        }
    }
    if constexpr(SplitCount==2) {
        if(tid<HeadsPerCTA*BlockM) {
            const int output_head=tid/BlockM;
            const int row=tid%BlockM;
            const std::size_t stat=(static_cast<std::size_t>(split)*rows*
                kQHeads+(query_base+row)*kQHeads+q_head_base+
                output_head)*2;
            split_stats[stat]=current_max[output_head];
            split_stats[stat+1]=denominator[output_head][row];
        }
    }
}

template<int SplitCount>
__global__ void attention_wmma32_split_merge_kernel(
    const float* split_output,const float* split_stats,
    std::uint16_t* output,int rows) {
    const int element=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    const int elements=rows*kQHeads*kHeadDim;
    if(element>=elements) return;
    const int head_row=element/kHeadDim;
    float maximum=-3.402823466e+38F;
    #pragma unroll
    for(int split=0;split<SplitCount;++split) {
        const std::size_t stat=(static_cast<std::size_t>(split)*rows*kQHeads+
            head_row)*2;
        if(split_stats[stat+1]>0.0f)
            maximum=fmaxf(maximum,split_stats[stat]);
    }
    float numerator=0.0f,denominator=0.0f;
    #pragma unroll
    for(int split=0;split<SplitCount;++split) {
        const std::size_t stat=(static_cast<std::size_t>(split)*rows*kQHeads+
            head_row)*2;
        const float sum=split_stats[stat+1];
        if(sum>0.0f) {
            const float scale=sum*exp2f((split_stats[stat]-maximum)*
                1.4426950408889634f);
            numerator+=split_output[static_cast<std::size_t>(split)*elements+
                element]*scale;
            denominator+=scale;
        }
    }
    const float value=numerator/denominator;
    output[element]=__half_as_ushort(__float2half_rn(value));
}

// T72 diagnostic candidate. One CTA owns a query/KV-head pair and scans the
// complete valid cache prefix in chronological tiles. Unlike T71's segmented
// fused-flash candidate, this route has no per-segment workspace or merge
// launch: the CTA keeps six FP32 online-softmax/value accumulators and writes
// the final FP16 heads directly. The online rescaling and FP16-KV arithmetic
// deliberately remain outside the exact/reference trajectory until a matched
// numerical gate accepts them. The ordinary exact branch below is retained as
// the explicit fallback for every unsupported or disabled configuration.
__global__ void attention_cached_gqa_six_whole_context_fused_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kKeysPerTile=256;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows||tid>=kHeadDim) return;

    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    if(count<1||count>capacity) return;

    __shared__ float query_values[kHeadsPerKV][kHeadDim];
    __shared__ float score_tile[kHeadsPerKV][kKeysPerTile];
    __shared__ float running_max[kHeadsPerKV];
    __shared__ float denominator[kHeadsPerKV];
    __shared__ float old_scale[kHeadsPerKV];

    #pragma unroll
    for(int head=0;head<kHeadsPerKV;++head)
        query_values[head][tid]=__half2float(__ushort_as_half(
            q[(query*kQHeads+kv_head*kHeadsPerKV+head)*kHeadDim+tid]));
    if(tid<kHeadsPerKV) {
        running_max[tid]=-3.402823466e+38F;
        denominator[tid]=0.0f;
        old_scale[tid]=0.0f;
    }
    float accumulated[kHeadsPerKV]={};
    __syncthreads();

    for(int first=0;first<count;first+=kKeysPerTile) {
        const int extent=min(kKeysPerTile,count-first);
        if(tid<extent) {
            float dot[kHeadsPerKV]={};
            const auto* represented_key=k_cache+
                (static_cast<std::size_t>(first+tid)*kKVHeads+kv_head)*kHeadDim;
            for(int d=0;d<kHeadDim;d+=2) {
                const float2 represented=__half22float2(
                    *reinterpret_cast<const __half2*>(represented_key+d));
                #pragma unroll
                for(int head=0;head<kHeadsPerKV;++head) {
                    dot[head]+=query_values[head][d]*represented.x;
                    dot[head]+=query_values[head][d+1]*represented.y;
                }
            }
            #pragma unroll
            for(int head=0;head<kHeadsPerKV;++head)
                score_tile[head][tid]=dot[head]*0.0625f;
        }
        __syncthreads();

        if(tid<kHeadsPerKV) {
            float tile_max=-3.402823466e+38F;
            for(int offset=0;offset<extent;++offset)
                tile_max=fmaxf(tile_max,score_tile[tid][offset]);
            const float next_max=fmaxf(running_max[tid],tile_max);
            old_scale[tid]=denominator[tid]==0.0f?0.0f:
                expf(running_max[tid]-next_max);
            running_max[tid]=next_max;
        }
        __syncthreads();

        if(tid<extent) {
            #pragma unroll
            for(int head=0;head<kHeadsPerKV;++head)
                score_tile[head][tid]=expf(score_tile[head][tid]-running_max[head]);
        }
        __syncthreads();

        if(tid<kHeadsPerKV) {
            float tile_sum=0.0f;
            for(int offset=0;offset<extent;++offset)
                tile_sum+=score_tile[tid][offset];
            denominator[tid]=denominator[tid]*old_scale[tid]+tile_sum;
        }
        #pragma unroll
        for(int head=0;head<kHeadsPerKV;++head)
            accumulated[head]*=old_scale[head];
        for(int offset=0;offset<extent;++offset) {
            const float represented_value=__half2float(__ushort_as_half(
                v_cache[((static_cast<std::size_t>(first+offset)*kKVHeads+
                    kv_head)*kHeadDim)+tid]));
            #pragma unroll
            for(int head=0;head<kHeadsPerKV;++head)
                accumulated[head]+=score_tile[head][offset]*represented_value;
        }
        __syncthreads();
    }

    #pragma unroll
    for(int head=0;head<kHeadsPerKV;++head)
        output[(query*kQHeads+kv_head*kHeadsPerKV+head)*kHeadDim+tid]=
            __half_as_ushort(__float2half_rn(accumulated[head]/denominator[head]));
}

constexpr int kT71SegmentKeys=1024;
constexpr int kT71PartialValues=(kQHeads/kKVHeads)*kHeadDim;
constexpr int kT71PartialStats=2*(kQHeads/kKVHeads);
constexpr int kT71PartialStride=kT71PartialValues+kT71PartialStats;

// T71B: independent bounded key segments restore the key-extent concurrency
// that T71A lacked. Each block leaves an unnormalized numerator and its local
// online-softmax statistics; no represented state is changed.
__global__ void attention_cached_gqa_six_splitk_partial_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,float* workspace,int rows,int position,
    int capacity,int segments,const int* position_device,int query_offset) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int blocks_per_query=kKVHeads*segments;
    const int query=static_cast<int>(blockIdx.x)/blocks_per_query;
    const int local=static_cast<int>(blockIdx.x)%blocks_per_query;
    const int kv_head=local/segments,segment=local%segments,tid=threadIdx.x;
    if(query>=rows||tid>=kHeadDim) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    const int segment_begin=segment*kT71SegmentKeys;
    const int segment_end=min(count,segment_begin+kT71SegmentKeys);
    float* slot=workspace+((static_cast<std::size_t>(query)*kKVHeads+kv_head)*
        segments+segment)*kT71PartialStride;
    float* partial=slot;
    float* maxima=partial+kT71PartialValues;
    float* denominators=maxima+kHeadsPerKV;

    __shared__ float query_values[kHeadsPerKV][kHeadDim];
    __shared__ float score_tile[kHeadsPerKV][256];
    __shared__ float running_max[kHeadsPerKV];
    __shared__ float denominator[kHeadsPerKV];
    __shared__ float old_scale[kHeadsPerKV];
    #pragma unroll
    for(int h=0;h<kHeadsPerKV;++h)
        query_values[h][tid]=__half2float(__ushort_as_half(
            q[(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]));
    if(tid<kHeadsPerKV) {
        running_max[tid]=-3.402823466e+38F;
        denominator[tid]=0.0f;
        old_scale[tid]=0.0f;
    }
    float accumulated[kHeadsPerKV]={};
    __syncthreads();

    for(int first=segment_begin;first<segment_end;first+=256) {
        const int extent=min(256,segment_end-first);
        const int key=first+tid;
        if(tid<extent) {
            float dot[kHeadsPerKV]={};
            const auto* represented_key=k_cache+
                (static_cast<std::size_t>(key)*kKVHeads+kv_head)*kHeadDim;
            for(int d=0;d<kHeadDim;d+=2) {
                const float2 represented=__half22float2(
                    *reinterpret_cast<const __half2*>(represented_key+d));
                #pragma unroll
                for(int h=0;h<kHeadsPerKV;++h) {
                    dot[h]+=query_values[h][d]*represented.x;
                    dot[h]+=query_values[h][d+1]*represented.y;
                }
            }
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h][tid]=dot[h]*0.0625f;
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float tile_max=-3.402823466e+38F;
            for(int offset=0;offset<extent;++offset)
                tile_max=fmaxf(tile_max,score_tile[tid][offset]);
            const float next_max=fmaxf(running_max[tid],tile_max);
            old_scale[tid]=denominator[tid]==0.0f?0.0f:
                expf(running_max[tid]-next_max);
            running_max[tid]=next_max;
        }
        __syncthreads();
        if(tid<extent) {
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                score_tile[h][tid]=expf(score_tile[h][tid]-running_max[h]);
        }
        __syncthreads();
        if(tid<kHeadsPerKV) {
            float tile_sum=0.0f;
            for(int offset=0;offset<extent;++offset)
                tile_sum+=score_tile[tid][offset];
            denominator[tid]=denominator[tid]*old_scale[tid]+tile_sum;
        }
        #pragma unroll
        for(int h=0;h<kHeadsPerKV;++h) accumulated[h]*=old_scale[h];
        for(int offset=0;offset<extent;++offset) {
            const float represented_value=__half2float(__ushort_as_half(
                v_cache[((static_cast<std::size_t>(first+offset)*kKVHeads+
                    kv_head)*kHeadDim)+tid]));
            #pragma unroll
            for(int h=0;h<kHeadsPerKV;++h)
                accumulated[h]+=score_tile[h][offset]*represented_value;
        }
        __syncthreads();
    }
    #pragma unroll
    for(int h=0;h<kHeadsPerKV;++h)
        partial[h*kHeadDim+tid]=accumulated[h];
    if(tid<kHeadsPerKV) {
        maxima[tid]=running_max[tid];
        denominators[tid]=denominator[tid];
    }
}

__global__ void attention_cached_gqa_six_splitk_merge_kernel(
    const float* workspace,std::uint16_t* output,int rows,int segments) {
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    const int query=static_cast<int>(blockIdx.x)/kKVHeads;
    const int kv_head=static_cast<int>(blockIdx.x)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows||tid>=kHeadDim) return;
    const auto* group=workspace+(static_cast<std::size_t>(query)*kKVHeads+
        kv_head)*segments*kT71PartialStride;
    __shared__ float maximum[kHeadsPerKV],denominator[kHeadsPerKV];
    if(tid<kHeadsPerKV) {
        float global_max=-3.402823466e+38F;
        for(int segment=0;segment<segments;++segment) {
            const auto* slot=group+segment*kT71PartialStride;
            const float local_den=slot[kT71PartialValues+kHeadsPerKV+tid];
            if(local_den>0.0f)
                global_max=fmaxf(global_max,slot[kT71PartialValues+tid]);
        }
        float global_den=0.0f;
        for(int segment=0;segment<segments;++segment) {
            const auto* slot=group+segment*kT71PartialStride;
            const float local_den=slot[kT71PartialValues+kHeadsPerKV+tid];
            if(local_den>0.0f)
                global_den+=local_den*expf(slot[kT71PartialValues+tid]-global_max);
        }
        maximum[tid]=global_max;denominator[tid]=global_den;
    }
    __syncthreads();
    #pragma unroll
    for(int h=0;h<kHeadsPerKV;++h) {
        float numerator=0.0f;
        for(int segment=0;segment<segments;++segment) {
            const auto* slot=group+segment*kT71PartialStride;
            const float local_den=slot[kT71PartialValues+kHeadsPerKV+h];
            if(local_den>0.0f)
                numerator+=slot[h*kHeadDim+tid]*
                    expf(slot[kT71PartialValues+h]-maximum[h]);
        }
        output[(query*kQHeads+kv_head*kHeadsPerKV+h)*kHeadDim+tid]=
            __half_as_ushort(__float2half_rn(numerator/denominator[h]));
    }
}

__global__ void sigmoid_mul_kernel(const std::uint16_t* gate,
                                   std::uint16_t* values,
                                   int count) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = __half2float(__ushort_as_half(gate[i]));
    const float x = __half2float(__ushort_as_half(values[i]));
    values[i] = __half_as_ushort(__float2half_rn(x / (1.0f + expf(-g))));
}

__global__ void silu_mul_kernel(const std::uint16_t* gate,
                                const std::uint16_t* up,
                                std::uint16_t* output,
                                int count) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = __half2float(__ushort_as_half(gate[i]));
    const float u = __half2float(__ushort_as_half(up[i]));
    output[i] = __half_as_ushort(__float2half_rn((g / (1.0f + expf(-g))) * u));
}

__global__ void residual_kernel(const std::uint16_t* left,
                                const std::uint16_t* right,
                                std::uint16_t* output,
                                int count) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float a = __half2float(__ushort_as_half(left[i]));
    const float b = __half2float(__ushort_as_half(right[i]));
    output[i] = __half_as_ushort(__float2half_rn(a + b));
}

void launch_profile_event(cudaEvent_t event, cudaStream_t stream) {
    cuda_check(cudaEventRecord(event, stream), "record EXL3 layer timing event");
}

} // namespace

Exl3TripleScoreLayoutDiscriminator
exl3_triple_score_layout_discriminator_for_test() {
    constexpr int capacity=4096;
    constexpr int blocks=kKVHeads*(kQHeads/kKVHeads/3);
    constexpr int threads=128;
    constexpr int samples=31;
    constexpr std::size_t value_count=
        static_cast<std::size_t>(capacity)*kKVHeads*kHeadDim;
    constexpr std::size_t score_count=
        static_cast<std::size_t>(kQHeads)*capacity;
    constexpr std::size_t output_count=kQHeads*kHeadDim;
    std::uint16_t* values=nullptr;
    float* head_major=nullptr;
    float* interleaved=nullptr;
    std::uint16_t* reference=nullptr;
    std::uint16_t* candidate=nullptr;
    unsigned long long* mismatches=nullptr;
    cudaEvent_t begin=nullptr,end=nullptr;
    auto cleanup=[&]() noexcept {
        if(end)(void)cudaEventDestroy(end);
        if(begin)(void)cudaEventDestroy(begin);
        if(mismatches)(void)cudaFree(mismatches);
        if(candidate)(void)cudaFree(candidate);
        if(reference)(void)cudaFree(reference);
        if(interleaved)(void)cudaFree(interleaved);
        if(head_major)(void)cudaFree(head_major);
        if(values)(void)cudaFree(values);
    };
    try {
        cuda_check(cudaMalloc(&values,value_count*sizeof(*values)),
            "allocate score-layout values");
        cuda_check(cudaMalloc(&head_major,score_count*sizeof(*head_major)),
            "allocate head-major scores");
        cuda_check(cudaMalloc(&interleaved,score_count*sizeof(*interleaved)),
            "allocate interleaved scores");
        cuda_check(cudaMalloc(&reference,output_count*sizeof(*reference)),
            "allocate score-layout reference");
        cuda_check(cudaMalloc(&candidate,output_count*sizeof(*candidate)),
            "allocate score-layout candidate");
        cuda_check(cudaMalloc(&mismatches,sizeof(*mismatches)),
            "allocate score-layout mismatches");
        cuda_check(cudaMemset(mismatches,0,sizeof(*mismatches)),
            "clear score-layout mismatches");
        cuda_check(cudaEventCreate(&begin),"create score-layout begin event");
        cuda_check(cudaEventCreate(&end),"create score-layout end event");
        const int init_blocks=static_cast<int>((value_count+255)/256);
        attention_triple_layout_discriminator_init_kernel<<<init_blocks,256>>>(
            values,head_major,interleaved);
        cuda_check(cudaGetLastError(),"initialize score-layout discriminator");

        const auto launch=[&](bool use_interleaved) {
            if(use_interleaved)
                attention_cached_gqa_triple_interleaved_values_kernel<<<blocks,threads>>>(
                    values,candidate,interleaved,capacity,capacity);
            else
                attention_cached_gqa_triple_normalized_values_kernel<false><<<blocks,threads>>>(
                    values,reference,head_major,1,capacity-1,capacity,nullptr,0);
        };
        for(int warmup=0;warmup<4;++warmup){launch(false);launch(true);}
        cuda_check(cudaGetLastError(),"warm score-layout discriminator");
        cuda_check(cudaDeviceSynchronize(),"synchronize score-layout warmup");
        const auto time_kernel=[&](bool use_interleaved) {
            cuda_check(cudaEventRecord(begin),"record score-layout begin");
            launch(use_interleaved);
            cuda_check(cudaGetLastError(),"launch score-layout sample");
            cuda_check(cudaEventRecord(end),"record score-layout end");
            cuda_check(cudaEventSynchronize(end),"synchronize score-layout sample");
            float elapsed_ms=0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms,begin,end),
                "time score-layout sample");
            return static_cast<double>(elapsed_ms)*1000.0;
        };
        std::vector<double> head_major_samples,interleaved_samples;
        head_major_samples.reserve(samples);interleaved_samples.reserve(samples);
        for(int sample=0;sample<samples;++sample) {
            if((sample&1)==0) {
                head_major_samples.push_back(time_kernel(false));
                interleaved_samples.push_back(time_kernel(true));
            } else {
                interleaved_samples.push_back(time_kernel(true));
                head_major_samples.push_back(time_kernel(false));
            }
        }
        attention_triple_layout_discriminator_compare_kernel<<<
            static_cast<int>((output_count+255)/256),256>>>(
                reference,candidate,output_count,mismatches);
        cuda_check(cudaGetLastError(),"compare score-layout outputs");
        unsigned long long mismatch_count=0;
        cuda_check(cudaMemcpy(&mismatch_count,mismatches,sizeof(mismatch_count),
                              cudaMemcpyDeviceToHost),
            "read score-layout mismatches");
        const auto median=[](std::vector<double>& values) {
            std::sort(values.begin(),values.end());
            return values[values.size()/2];
        };
        Exl3TripleScoreLayoutDiscriminator result;
        result.head_major_median_us=median(head_major_samples);
        result.interleaved_median_us=median(interleaved_samples);
        result.score_bytes=score_count*sizeof(float);
        result.history_rows=capacity;
        result.mismatches=static_cast<std::uint64_t>(mismatch_count);
        cleanup();
        return result;
    } catch(...) {
        cleanup();
        throw;
    }
}

Exl3SoftmaxValueFusionDiscriminator
exl3_softmax_value_fusion_discriminator_for_test() {
    constexpr int capacity=4096;
    constexpr int softmax_blocks=kKVHeads;
    constexpr int value_blocks=kKVHeads*(kQHeads/kKVHeads/3);
    constexpr int samples=31;
    constexpr std::size_t value_count=
        static_cast<std::size_t>(capacity)*kKVHeads*kHeadDim;
    constexpr std::size_t score_count=
        static_cast<std::size_t>(kQHeads)*capacity;
    constexpr std::size_t output_count=kQHeads*kHeadDim;
    std::uint16_t* values=nullptr;
    float* raw_scores=nullptr;
    float* selected_scores=nullptr;
    float* fused_scores=nullptr;
    float* unused_interleaved=nullptr;
    std::uint16_t* selected_output=nullptr;
    std::uint16_t* fused_output=nullptr;
    unsigned long long* output_mismatches=nullptr;
    unsigned long long* score_mismatches=nullptr;
    cudaEvent_t begin=nullptr,end=nullptr;
    auto cleanup=[&]() noexcept {
        if(end)(void)cudaEventDestroy(end);
        if(begin)(void)cudaEventDestroy(begin);
        if(score_mismatches)(void)cudaFree(score_mismatches);
        if(output_mismatches)(void)cudaFree(output_mismatches);
        if(fused_output)(void)cudaFree(fused_output);
        if(selected_output)(void)cudaFree(selected_output);
        if(unused_interleaved)(void)cudaFree(unused_interleaved);
        if(fused_scores)(void)cudaFree(fused_scores);
        if(selected_scores)(void)cudaFree(selected_scores);
        if(raw_scores)(void)cudaFree(raw_scores);
        if(values)(void)cudaFree(values);
    };
    try {
        cuda_check(cudaMalloc(&values,value_count*sizeof(*values)),
            "allocate fusion values");
        cuda_check(cudaMalloc(&raw_scores,score_count*sizeof(*raw_scores)),
            "allocate fusion raw scores");
        cuda_check(cudaMalloc(&selected_scores,score_count*sizeof(*selected_scores)),
            "allocate selected fusion scores");
        cuda_check(cudaMalloc(&fused_scores,score_count*sizeof(*fused_scores)),
            "allocate candidate fusion scores");
        cuda_check(cudaMalloc(&unused_interleaved,score_count*sizeof(*unused_interleaved)),
            "allocate fusion init scratch");
        cuda_check(cudaMalloc(&selected_output,output_count*sizeof(*selected_output)),
            "allocate selected fusion output");
        cuda_check(cudaMalloc(&fused_output,output_count*sizeof(*fused_output)),
            "allocate candidate fusion output");
        cuda_check(cudaMalloc(&output_mismatches,sizeof(*output_mismatches)),
            "allocate fusion output mismatches");
        cuda_check(cudaMalloc(&score_mismatches,sizeof(*score_mismatches)),
            "allocate fusion score mismatches");
        cuda_check(cudaEventCreate(&begin),"create fusion begin event");
        cuda_check(cudaEventCreate(&end),"create fusion end event");
        const int init_blocks=static_cast<int>((value_count+255)/256);
        attention_triple_layout_discriminator_init_kernel<<<init_blocks,256>>>(
            values,raw_scores,unused_interleaved);
        cuda_check(cudaGetLastError(),"initialize fusion discriminator");

        const auto reset=[&](float* scores) {
            cuda_check(cudaMemcpyAsync(scores,raw_scores,score_count*sizeof(float),
                                       cudaMemcpyDeviceToDevice),
                "reset fusion scores");
        };
        const auto launch=[&](bool fused) {
            if(fused)
                attention_cached_gqa_six_softmax_triple_values_fused_kernel<<<
                    softmax_blocks,256>>>(values,fused_output,fused_scores,1,
                    capacity-1,capacity,nullptr,0);
            else {
                attention_cached_gqa_six_softmax_kernel<<<softmax_blocks,256>>>(
                    selected_scores,1,capacity-1,capacity,nullptr,0);
                attention_cached_gqa_triple_normalized_values_kernel<false><<<
                    value_blocks,128>>>(values,selected_output,selected_scores,1,
                    capacity-1,capacity,nullptr,0);
            }
        };
        for(int warmup=0;warmup<4;++warmup) {
            reset(selected_scores);launch(false);
            reset(fused_scores);launch(true);
        }
        cuda_check(cudaGetLastError(),"warm fusion discriminator");
        cuda_check(cudaDeviceSynchronize(),"synchronize fusion warmup");
        const auto time_stage=[&](bool fused) {
            reset(fused?fused_scores:selected_scores);
            cuda_check(cudaEventRecord(begin),"record fusion begin");
            launch(fused);
            cuda_check(cudaGetLastError(),"launch fusion sample");
            cuda_check(cudaEventRecord(end),"record fusion end");
            cuda_check(cudaEventSynchronize(end),"synchronize fusion sample");
            float elapsed_ms=0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms,begin,end),
                "time fusion sample");
            return static_cast<double>(elapsed_ms)*1000.0;
        };
        std::vector<double> selected_samples,fused_samples;
        selected_samples.reserve(samples);fused_samples.reserve(samples);
        for(int sample=0;sample<samples;++sample) {
            if((sample&1)==0) {
                selected_samples.push_back(time_stage(false));
                fused_samples.push_back(time_stage(true));
            } else {
                fused_samples.push_back(time_stage(true));
                selected_samples.push_back(time_stage(false));
            }
        }
        reset(selected_scores);launch(false);
        reset(fused_scores);launch(true);
        cuda_check(cudaMemsetAsync(output_mismatches,0,sizeof(*output_mismatches)),
            "clear fusion output mismatches");
        cuda_check(cudaMemsetAsync(score_mismatches,0,sizeof(*score_mismatches)),
            "clear fusion score mismatches");
        attention_triple_layout_discriminator_compare_kernel<<<
            static_cast<int>((output_count+255)/256),256>>>(selected_output,
                fused_output,output_count,output_mismatches);
        attention_fusion_discriminator_compare_scores_kernel<<<
            static_cast<int>((score_count+255)/256),256>>>(selected_scores,
                fused_scores,score_count,score_mismatches);
        cuda_check(cudaGetLastError(),"compare fusion discriminator");
        unsigned long long output_mismatch_count=0,score_mismatch_count=0;
        cuda_check(cudaMemcpy(&output_mismatch_count,output_mismatches,
                              sizeof(output_mismatch_count),cudaMemcpyDeviceToHost),
            "read fusion output mismatches");
        cuda_check(cudaMemcpy(&score_mismatch_count,score_mismatches,
                              sizeof(score_mismatch_count),cudaMemcpyDeviceToHost),
            "read fusion score mismatches");
        const auto median=[](std::vector<double>& values) {
            std::sort(values.begin(),values.end());
            return values[values.size()/2];
        };
        Exl3SoftmaxValueFusionDiscriminator result;
        result.selected_median_us=median(selected_samples);
        result.fused_median_us=median(fused_samples);
        result.score_bytes=score_count*sizeof(float);
        result.history_rows=capacity;
        result.output_mismatches=static_cast<std::uint64_t>(output_mismatch_count);
        result.score_mismatches=static_cast<std::uint64_t>(score_mismatch_count);
        cleanup();
        return result;
    } catch(...) {
        cleanup();
        throw;
    }
}

Exl3DeferredNormalizationDiscriminator
exl3_deferred_normalization_discriminator_for_test() {
    constexpr int capacity=4096;
    constexpr int softmax_blocks=kKVHeads;
    constexpr int value_blocks=kKVHeads*(kQHeads/kKVHeads/3);
    constexpr int samples=31;
    constexpr std::size_t value_count=
        static_cast<std::size_t>(capacity)*kKVHeads*kHeadDim;
    constexpr std::size_t score_count=static_cast<std::size_t>(kQHeads)*capacity;
    constexpr std::size_t output_count=kQHeads*kHeadDim;
    std::uint16_t* values=nullptr;
    float* raw_scores=nullptr;
    float* selected_scores=nullptr;
    float* deferred_scores=nullptr;
    float* unused_interleaved=nullptr;
    float* denominators=nullptr;
    std::uint16_t* selected_output=nullptr;
    std::uint16_t* deferred_output=nullptr;
    unsigned long long* output_mismatches=nullptr;
    unsigned long long* score_mismatches=nullptr;
    cudaEvent_t begin=nullptr,end=nullptr;
    auto cleanup=[&]() noexcept {
        if(end)(void)cudaEventDestroy(end);
        if(begin)(void)cudaEventDestroy(begin);
        if(score_mismatches)(void)cudaFree(score_mismatches);
        if(output_mismatches)(void)cudaFree(output_mismatches);
        if(deferred_output)(void)cudaFree(deferred_output);
        if(selected_output)(void)cudaFree(selected_output);
        if(denominators)(void)cudaFree(denominators);
        if(unused_interleaved)(void)cudaFree(unused_interleaved);
        if(deferred_scores)(void)cudaFree(deferred_scores);
        if(selected_scores)(void)cudaFree(selected_scores);
        if(raw_scores)(void)cudaFree(raw_scores);
        if(values)(void)cudaFree(values);
    };
    try {
        cuda_check(cudaMalloc(&values,value_count*sizeof(*values)),
            "allocate deferred-normalization values");
        cuda_check(cudaMalloc(&raw_scores,score_count*sizeof(*raw_scores)),
            "allocate deferred-normalization raw scores");
        cuda_check(cudaMalloc(&selected_scores,score_count*sizeof(*selected_scores)),
            "allocate selected normalization scores");
        cuda_check(cudaMalloc(&deferred_scores,score_count*sizeof(*deferred_scores)),
            "allocate deferred normalization scores");
        cuda_check(cudaMalloc(&unused_interleaved,score_count*sizeof(*unused_interleaved)),
            "allocate deferred-normalization init scratch");
        cuda_check(cudaMalloc(&denominators,kQHeads*sizeof(*denominators)),
            "allocate deferred denominators");
        cuda_check(cudaMalloc(&selected_output,output_count*sizeof(*selected_output)),
            "allocate selected normalization output");
        cuda_check(cudaMalloc(&deferred_output,output_count*sizeof(*deferred_output)),
            "allocate deferred normalization output");
        cuda_check(cudaMalloc(&output_mismatches,sizeof(*output_mismatches)),
            "allocate deferred output mismatches");
        cuda_check(cudaMalloc(&score_mismatches,sizeof(*score_mismatches)),
            "allocate deferred score mismatches");
        cuda_check(cudaEventCreate(&begin),"create deferred begin event");
        cuda_check(cudaEventCreate(&end),"create deferred end event");
        attention_triple_layout_discriminator_init_kernel<<<
            static_cast<int>((value_count+255)/256),256>>>(
                values,raw_scores,unused_interleaved);
        cuda_check(cudaGetLastError(),"initialize deferred-normalization discriminator");
        const auto reset=[&](float* scores) {
            cuda_check(cudaMemcpyAsync(scores,raw_scores,score_count*sizeof(float),
                                       cudaMemcpyDeviceToDevice),
                "reset deferred-normalization scores");
        };
        const auto launch=[&](bool deferred) {
            if(deferred) {
                attention_cached_gqa_six_softmax_kernel<true><<<softmax_blocks,256>>>(
                    deferred_scores,1,capacity-1,capacity,nullptr,0,denominators);
                attention_cached_gqa_triple_normalized_values_kernel<
                    false,false,false,false,false,true><<<value_blocks,128>>>(
                        values,deferred_output,deferred_scores,1,capacity-1,capacity,
                        nullptr,0,{},nullptr,denominators);
            } else {
                attention_cached_gqa_six_softmax_kernel<false><<<softmax_blocks,256>>>(
                    selected_scores,1,capacity-1,capacity,nullptr,0);
                attention_cached_gqa_triple_normalized_values_kernel<false><<<
                    value_blocks,128>>>(values,selected_output,selected_scores,1,
                    capacity-1,capacity,nullptr,0);
            }
        };
        for(int warmup=0;warmup<4;++warmup) {
            reset(selected_scores);launch(false);
            reset(deferred_scores);launch(true);
        }
        cuda_check(cudaGetLastError(),"warm deferred-normalization discriminator");
        cuda_check(cudaDeviceSynchronize(),"synchronize deferred-normalization warmup");
        const auto time_stage=[&](bool deferred) {
            reset(deferred?deferred_scores:selected_scores);
            cuda_check(cudaEventRecord(begin),"record deferred-normalization begin");
            launch(deferred);
            cuda_check(cudaGetLastError(),"launch deferred-normalization sample");
            cuda_check(cudaEventRecord(end),"record deferred-normalization end");
            cuda_check(cudaEventSynchronize(end),"synchronize deferred-normalization sample");
            float elapsed_ms=0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms,begin,end),
                "time deferred-normalization sample");
            return static_cast<double>(elapsed_ms)*1000.0;
        };
        std::vector<double> selected_samples,deferred_samples;
        selected_samples.reserve(samples);deferred_samples.reserve(samples);
        for(int sample=0;sample<samples;++sample) {
            if((sample&1)==0) {
                selected_samples.push_back(time_stage(false));
                deferred_samples.push_back(time_stage(true));
            } else {
                deferred_samples.push_back(time_stage(true));
                selected_samples.push_back(time_stage(false));
            }
        }
        reset(selected_scores);launch(false);
        reset(deferred_scores);launch(true);
        cuda_check(cudaMemsetAsync(output_mismatches,0,sizeof(*output_mismatches)),
            "clear deferred output mismatches");
        cuda_check(cudaMemsetAsync(score_mismatches,0,sizeof(*score_mismatches)),
            "clear deferred score mismatches");
        attention_triple_layout_discriminator_compare_kernel<<<
            static_cast<int>((output_count+255)/256),256>>>(selected_output,
                deferred_output,output_count,output_mismatches);
        attention_deferred_normalization_compare_scores_kernel<<<
            static_cast<int>((score_count+255)/256),256>>>(selected_scores,
                deferred_scores,denominators,capacity,score_count,score_mismatches);
        cuda_check(cudaGetLastError(),"compare deferred-normalization discriminator");
        unsigned long long output_mismatch_count=0,score_mismatch_count=0;
        cuda_check(cudaMemcpy(&output_mismatch_count,output_mismatches,
                              sizeof(output_mismatch_count),cudaMemcpyDeviceToHost),
            "read deferred output mismatches");
        cuda_check(cudaMemcpy(&score_mismatch_count,score_mismatches,
                              sizeof(score_mismatch_count),cudaMemcpyDeviceToHost),
            "read deferred score mismatches");
        const auto median=[](std::vector<double>& values) {
            std::sort(values.begin(),values.end());
            return values[values.size()/2];
        };
        Exl3DeferredNormalizationDiscriminator result;
        result.selected_median_us=median(selected_samples);
        result.deferred_median_us=median(deferred_samples);
        result.score_bytes=score_count*sizeof(float);
        result.denominator_bytes=kQHeads*sizeof(float);
        result.history_rows=capacity;
        result.output_mismatches=static_cast<std::uint64_t>(output_mismatch_count);
        result.normalized_score_mismatches=static_cast<std::uint64_t>(score_mismatch_count);
        cleanup();
        return result;
    } catch(...) {
        cleanup();
        throw;
    }
}

Exl3AttentionGateFusionDiscriminator
exl3_attention_gate_fusion_discriminator_for_test() {
    constexpr int capacity=4096;
    constexpr int value_blocks=kKVHeads*(kQHeads/kKVHeads/3);
    constexpr int value_threads=256;
    constexpr int samples=31;
    constexpr std::size_t value_count=
        static_cast<std::size_t>(capacity)*kKVHeads*kHeadDim;
    constexpr std::size_t score_count=
        static_cast<std::size_t>(kQHeads)*capacity;
    constexpr std::size_t output_count=kQHeads*kHeadDim;
    std::uint16_t* values=nullptr;
    float* scores=nullptr;
    float* unused_interleaved=nullptr;
    std::uint16_t* gate=nullptr;
    std::uint16_t* selected_output=nullptr;
    std::uint16_t* fused_output=nullptr;
    unsigned long long* mismatches=nullptr;
    cudaEvent_t begin=nullptr,end=nullptr;
    auto cleanup=[&]() noexcept {
        if(end)(void)cudaEventDestroy(end);
        if(begin)(void)cudaEventDestroy(begin);
        if(mismatches)(void)cudaFree(mismatches);
        if(fused_output)(void)cudaFree(fused_output);
        if(selected_output)(void)cudaFree(selected_output);
        if(gate)(void)cudaFree(gate);
        if(unused_interleaved)(void)cudaFree(unused_interleaved);
        if(scores)(void)cudaFree(scores);
        if(values)(void)cudaFree(values);
    };
    try {
        cuda_check(cudaMalloc(&values,value_count*sizeof(*values)),
            "allocate gate-fusion values");
        cuda_check(cudaMalloc(&scores,score_count*sizeof(*scores)),
            "allocate gate-fusion scores");
        cuda_check(cudaMalloc(&unused_interleaved,score_count*sizeof(*unused_interleaved)),
            "allocate gate-fusion init scratch");
        cuda_check(cudaMalloc(&gate,output_count*sizeof(*gate)),
            "allocate gate-fusion gate");
        cuda_check(cudaMalloc(&selected_output,output_count*sizeof(*selected_output)),
            "allocate gate-fusion selected output");
        cuda_check(cudaMalloc(&fused_output,output_count*sizeof(*fused_output)),
            "allocate gate-fusion fused output");
        cuda_check(cudaMalloc(&mismatches,sizeof(*mismatches)),
            "allocate gate-fusion mismatches");
        cuda_check(cudaEventCreate(&begin),"create gate-fusion begin event");
        cuda_check(cudaEventCreate(&end),"create gate-fusion end event");
        attention_triple_layout_discriminator_init_kernel<<<
            static_cast<int>((value_count+255)/256),256>>>(
                values,scores,unused_interleaved);
        attention_gate_discriminator_init_kernel<<<
            static_cast<int>((output_count+255)/256),256>>>(gate,output_count);
        cuda_check(cudaGetLastError(),"initialize attention gate-fusion discriminator");
        const auto launch=[&](bool fused) {
            if(fused)
                attention_cached_gqa_triple_normalized_values_kernel<
                    false,false,false,false,true><<<value_blocks,value_threads>>>(
                        values,fused_output,scores,1,capacity-1,capacity,nullptr,0,
                        {},gate);
            else {
                attention_cached_gqa_triple_normalized_values_kernel<false><<<
                    value_blocks,value_threads>>>(values,selected_output,scores,1,
                        capacity-1,capacity,nullptr,0);
                exl3_launch_small(sigmoid_mul_kernel,dim3(static_cast<int>((output_count+255)/256)),dim3(256),0,nullptr,
                    gate,selected_output,static_cast<int>(output_count));
            }
        };
        for(int warmup=0;warmup<4;++warmup){launch(false);launch(true);}
        cuda_check(cudaGetLastError(),"warm attention gate-fusion discriminator");
        cuda_check(cudaDeviceSynchronize(),
            "synchronize attention gate-fusion warmup");
        const auto time_stage=[&](bool fused) {
            cuda_check(cudaEventRecord(begin),"record gate-fusion begin");
            launch(fused);
            cuda_check(cudaGetLastError(),"launch gate-fusion sample");
            cuda_check(cudaEventRecord(end),"record gate-fusion end");
            cuda_check(cudaEventSynchronize(end),"synchronize gate-fusion sample");
            float elapsed_ms=0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms,begin,end),
                "time gate-fusion sample");
            return static_cast<double>(elapsed_ms)*1000.0;
        };
        std::vector<double> selected_samples,fused_samples;
        selected_samples.reserve(samples);fused_samples.reserve(samples);
        for(int sample=0;sample<samples;++sample) {
            if((sample&1)==0) {
                selected_samples.push_back(time_stage(false));
                fused_samples.push_back(time_stage(true));
            } else {
                fused_samples.push_back(time_stage(true));
                selected_samples.push_back(time_stage(false));
            }
        }
        launch(false);launch(true);
        cuda_check(cudaMemsetAsync(mismatches,0,sizeof(*mismatches)),
            "clear gate-fusion mismatches");
        attention_triple_layout_discriminator_compare_kernel<<<
            static_cast<int>((output_count+255)/256),256>>>(
                selected_output,fused_output,output_count,mismatches);
        cuda_check(cudaGetLastError(),"compare attention gate-fusion outputs");
        unsigned long long mismatch_count=0;
        cuda_check(cudaMemcpy(&mismatch_count,mismatches,sizeof(mismatch_count),
                              cudaMemcpyDeviceToHost),
            "read gate-fusion mismatches");
        const auto median=[](std::vector<double>& values) {
            std::sort(values.begin(),values.end());
            return values[values.size()/2];
        };
        Exl3AttentionGateFusionDiscriminator result;
        result.selected_median_us=median(selected_samples);
        result.fused_median_us=median(fused_samples);
        result.history_rows=capacity;
        result.output_bytes=output_count*sizeof(std::uint16_t);
        result.mismatches=static_cast<std::uint64_t>(mismatch_count);
        cleanup();
        return result;
    } catch(...) {
        cleanup();
        throw;
    }
}

void exl3_exact_segmented_attention_for_test(const std::uint16_t* q,
    const std::uint16_t* prefix_k,const std::uint16_t* prefix_v,int prefix_rows,
    const std::uint16_t* tail_k,const std::uint16_t* tail_v,std::uint16_t* output,
    float* scores,int rows,int position,int capacity,cudaStream_t stream,int prefix_first) {
    if(!q || !tail_k || !tail_v || !output || !scores || rows<1 || rows>16 ||
       position<0 || capacity<rows || position>capacity-rows || prefix_rows<0 || prefix_first<0 ||
       prefix_first>position || prefix_rows>position-prefix_first ||
       (prefix_rows && (!prefix_k || !prefix_v)))
        throw std::invalid_argument("segmented attention extent");
    attention_cached_parallel_kernel<true><<<rows*kQHeads,256,0,stream>>>(
        q,tail_k,tail_v,output,scores,rows,position,capacity,nullptr,0,prefix_k,prefix_v,prefix_rows,prefix_first);
    cuda_check(cudaGetLastError(),"segmented attention");
}

void exl3_exact_page_attention_for_test(const std::uint16_t* q,const Exl3AttentionPageRanges& ranges,
    const std::uint16_t* private_k,const std::uint16_t* private_v,std::uint16_t* output,float* scores,
    int rows,int position,int capacity,cudaStream_t stream,bool q_shared,bool gqa_pair,bool query_pair) {
    if(!q || !private_k || !private_v || !output || !scores || rows<1 || rows>16 ||
        position<0 || capacity<rows || position>capacity-rows || ranges.count<0 ||
        ranges.count>Exl3AttentionPageRanges::capacity)throw std::invalid_argument("page attention extent");
    Exl3AttentionPageRanges checked;
    for(int i=0;i<ranges.count;++i) {
        const auto& range=ranges.ranges[i];checked.append(range.k,range.v,range.first,range.rows,position);
    }
    if(query_pair)attention_query_pair_kernel<<<((rows+1)/2)*kQHeads,256,0,stream>>>(
        q,private_k,private_v,output,scores,rows,position,capacity,nullptr,0,checked);
    else if(gqa_pair) {
        attention_cached_gqa_pair_scores_kernel<true><<<rows*kQHeads,256,0,stream>>>(q,private_k,scores,rows,position,capacity,nullptr,0,checked);
        attention_cached_gqa_pair_values_kernel<true><<<rows*(kQHeads/2),256,0,stream>>>(private_v,output,scores,rows,position,capacity,nullptr,0,checked);
    } else if(q_shared)attention_cached_parallel_q_shared_kernel<false,false,true><<<rows*kQHeads,256,0,stream>>>(
        q,private_k,private_v,output,scores,rows,position,capacity,nullptr,0,checked);
    else attention_cached_parallel_kernel<true><<<rows*kQHeads,256,0,stream>>>(q,private_k,private_v,
        output,scores,rows,position,capacity,nullptr,0,nullptr,nullptr,0,0,checked);
    cuda_check(cudaGetLastError(),"page attention");
}

void exl3_exact_attention_for_test(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,std::uint16_t* output,float* scores,int rows,
    int position,int capacity,bool parallel,cudaStream_t stream,bool q_shared,bool k_half2,bool v_half2,bool gqa_pair,bool gqa_triple,
    bool gqa_triple_values128,bool gqa_triple_softmax_staged,bool gqa_six,
    bool gqa_six_scores,bool gqa_six_values_sharded,bool gqa_triple_values4,
    bool gqa_six_softmax_triple_values,bool gqa_six_packed_triples,
    bool gqa_six_extent_shards,bool gqa_six_query_pair_scores,
    bool gqa_six_softmax_triple_values_v_tile,
    bool gqa_six_softmax_triple_values_full_cta,
    bool gqa_six_softmax_triple_values_fused,
    bool gqa_six_softmax_tile512,
    bool gqa_six_score_k_tile64,
    bool gqa_six_softmax_triple_values_scalar_dim,
    bool gqa_six_softmax_triple_values_two_query,
    Exl3AttentionPageRanges segmented_pages,
    bool gqa_six_softmax_triple_values_pair_dimensions,
    bool gqa_six_softmax_six_values_single_load,
    bool gqa_six_softmax_triple_values_v_tile64,
    bool gqa_six_softmax_triple_values_warp_score_broadcast,
    bool gqa_six_softmax_six_values_scalar_single_load,
    bool gqa_six_softmax_triple_values_key_pair_pipeline,
    bool gqa_six_softmax_fused_scalar_values) {
    const bool v_tile_variant=gqa_six_softmax_triple_values_v_tile ||
        gqa_six_softmax_triple_values_v_tile64;
    const bool segmented_selected_six=segmented_pages.count>0 &&
        gqa_six_scores && gqa_six_softmax_triple_values &&
        !gqa_six_query_pair_scores && !gqa_six_softmax_triple_values_v_tile &&
        !gqa_six_softmax_triple_values_v_tile64 &&
        !gqa_six_softmax_triple_values_full_cta &&
        !gqa_six_softmax_triple_values_fused && !gqa_six_softmax_tile512 &&
        !gqa_six_softmax_fused_scalar_values &&
        !gqa_six_score_k_tile64 &&
        !gqa_six_softmax_triple_values_scalar_dim &&
        !gqa_six_softmax_triple_values_two_query &&
        !gqa_six_softmax_triple_values_pair_dimensions &&
        !gqa_six_softmax_six_values_single_load &&
        !gqa_six_softmax_six_values_scalar_single_load &&
        !gqa_six_softmax_triple_values_key_pair_pipeline &&
        !gqa_six_softmax_triple_values_warp_score_broadcast;
    if(!q||!k||!v||!output||rows<1||rows>1024||position<0||capacity<rows||position>capacity-rows||
       (parallel&&!scores)||(!parallel&&position+rows>4096)||(q_shared&&!parallel)||
       (k_half2&&!q_shared)||(v_half2&&!k_half2)||(gqa_pair&&!v_half2)||(gqa_triple&&!gqa_pair)||
       (gqa_triple_values128&&!gqa_triple)||(gqa_triple_softmax_staged&&!gqa_triple)||
       (gqa_six&&!gqa_triple_softmax_staged)||
       (gqa_six_scores&&!gqa_triple_softmax_staged)||
       (gqa_six&&gqa_six_scores)||(gqa_six_values_sharded&&!gqa_six_scores)||
       (gqa_six&&gqa_six_values_sharded)||(gqa_triple_values4&&!gqa_triple_softmax_staged)||
       (gqa_triple_values4&&(gqa_six||gqa_six_values_sharded||gqa_triple_values128))||
       (gqa_six_softmax_triple_values&&(!gqa_six_scores||gqa_six||
          gqa_six_values_sharded||gqa_triple_values4||gqa_six_packed_triples))||
       (gqa_six_packed_triples&&(!gqa_six_scores||gqa_six||
          gqa_six_values_sharded||gqa_triple_values4))||
       (gqa_six_extent_shards&&!gqa_six_scores)||
       (gqa_six_query_pair_scores&&!gqa_six_scores)||
       (v_tile_variant&&!gqa_six_softmax_triple_values)||
       (gqa_six_softmax_triple_values_v_tile&&
          gqa_six_softmax_triple_values_v_tile64)||
       (gqa_six_softmax_triple_values_full_cta&&
          (!gqa_six_softmax_triple_values||v_tile_variant))||
       (gqa_six_softmax_triple_values_fused&&
          (!gqa_six_softmax_triple_values||v_tile_variant||
           gqa_six_softmax_triple_values_full_cta))||
       (gqa_six_softmax_tile512&&
          (!gqa_six_softmax_triple_values||gqa_six_softmax_triple_values_fused))||
       (gqa_six_score_k_tile64&&
          (!gqa_six_softmax_triple_values||gqa_six_query_pair_scores))||
       (gqa_six_softmax_triple_values_scalar_dim&&
          (!gqa_six_softmax_triple_values||v_tile_variant||
           gqa_six_softmax_triple_values_full_cta||gqa_six_softmax_triple_values_fused))||
       (gqa_six_softmax_triple_values_two_query&&
          (!gqa_six_softmax_triple_values||v_tile_variant||
           gqa_six_softmax_triple_values_full_cta||gqa_six_softmax_triple_values_fused||
           gqa_six_softmax_triple_values_scalar_dim||
           gqa_six_softmax_triple_values_pair_dimensions)) ||
       (gqa_six_softmax_triple_values_pair_dimensions&&
          (!gqa_six_softmax_triple_values||v_tile_variant||
           gqa_six_softmax_triple_values_full_cta||gqa_six_softmax_triple_values_fused||
           gqa_six_softmax_triple_values_scalar_dim||
           gqa_six_softmax_six_values_single_load)) ||
       (gqa_six_softmax_six_values_single_load&&
          (!gqa_six_softmax_triple_values||v_tile_variant||
           gqa_six_softmax_triple_values_full_cta||gqa_six_softmax_triple_values_fused||
           gqa_six_softmax_triple_values_scalar_dim||
           gqa_six_softmax_triple_values_two_query)) ||
       (gqa_six_softmax_six_values_scalar_single_load&&
          (!gqa_six_softmax_triple_values||v_tile_variant||
           gqa_six_softmax_triple_values_full_cta||gqa_six_softmax_triple_values_fused||
           gqa_six_softmax_triple_values_scalar_dim||
           gqa_six_softmax_triple_values_two_query||
           gqa_six_softmax_triple_values_pair_dimensions||
           gqa_six_softmax_six_values_single_load)) ||
        (gqa_six_softmax_triple_values_warp_score_broadcast&&
          (!gqa_six_softmax_triple_values||v_tile_variant||
           gqa_six_softmax_triple_values_full_cta||
           gqa_six_softmax_triple_values_fused||
           gqa_six_softmax_triple_values_scalar_dim||
           gqa_six_softmax_triple_values_two_query||
           gqa_six_softmax_triple_values_pair_dimensions||
            gqa_six_softmax_six_values_single_load||
            gqa_six_softmax_six_values_scalar_single_load)) ||
       (gqa_six_softmax_triple_values_key_pair_pipeline&&
           (!gqa_six_softmax_triple_values||v_tile_variant||
            gqa_six_softmax_triple_values_full_cta||
            gqa_six_softmax_triple_values_fused||
            gqa_six_softmax_triple_values_scalar_dim||
            gqa_six_softmax_triple_values_two_query||
            gqa_six_softmax_triple_values_pair_dimensions||
            gqa_six_softmax_six_values_single_load||
            gqa_six_softmax_six_values_scalar_single_load||
            gqa_six_softmax_triple_values_warp_score_broadcast)) ||
       (gqa_six_softmax_fused_scalar_values &&
           (!gqa_six_softmax_triple_values || !gqa_six_scores || v_tile_variant ||
            gqa_six_softmax_triple_values_full_cta ||
            gqa_six_softmax_triple_values_fused || gqa_six_softmax_tile512 ||
            gqa_six_softmax_triple_values_scalar_dim ||
            gqa_six_softmax_triple_values_two_query ||
            gqa_six_softmax_triple_values_pair_dimensions ||
            gqa_six_softmax_six_values_single_load ||
            gqa_six_softmax_six_values_scalar_single_load ||
            gqa_six_softmax_triple_values_warp_score_broadcast ||
            gqa_six_softmax_triple_values_key_pair_pipeline)) ||
       (segmented_pages.count && !segmented_selected_six))
        throw std::invalid_argument("exact attention qualification extent");
    if(segmented_pages.count) for(int page=0;page<segmented_pages.count;++page) {
        const auto range=segmented_pages.ranges[page];
        if(range.first<0 || range.rows<1 || range.first>position ||
           range.rows>position-range.first || !range.k || !range.v)
            throw std::invalid_argument("segmented selected-six qualification range");
    }
    if(parallel) for(int first=0;first<rows;first+=16) {
        const int count=std::min(16,rows-first);
        const int score_shards=gqa_six_extent_shards?
            std::min(6,std::max(1,(position+first+count+255)/256)):6;
        if(gqa_six) {
            launch_attention_cached_gqa_six_scores(
                q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,scores,count,
                position,capacity,nullptr,first,6,stream);
            attention_cached_gqa_six_values_kernel<<<count*kKVHeads,256,0,stream>>>(
                v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
        } else if(gqa_six_values_sharded) {
            launch_attention_cached_gqa_six_scores(
                q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,scores,count,
                position,capacity,nullptr,first,6,stream);
            attention_cached_gqa_six_softmax_kernel<<<count*kKVHeads,256,0,stream>>>(
                scores,count,position,capacity,nullptr,first);
            attention_cached_gqa_six_values_sharded_kernel<<<count*kKVHeads*2,64,0,stream>>>(
                v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
        } else if(gqa_six_scores) {
            if(gqa_six_score_k_tile64)
                launch_attention_cached_gqa_six_scores_k_tile64(
                    q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,scores,
                    count,position,capacity,nullptr,first,score_shards,stream);
            else if(gqa_six_query_pair_scores && count>1)
                launch_attention_cached_gqa_six_query_pair_scores(
                    q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,scores,count,
                    position,capacity,nullptr,first,score_shards,stream);
            else if(segmented_pages.count)
                launch_attention_cached_gqa_six_segmented_scores(
                    q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,scores,
                    count,position,capacity,nullptr,first,score_shards,
                    segmented_pages,stream);
            else launch_attention_cached_gqa_six_scores(
                    q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,scores,count,
                    position,capacity,nullptr,first,score_shards,stream);
            if(gqa_six_packed_triples) {
                attention_cached_gqa_six_packed_triples_values_kernel<<<
                    count*kKVHeads,256,0,stream>>>(v,
                    output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                    scores,count,position,capacity,nullptr,first);
            } else if(gqa_six_softmax_triple_values) {
                if(gqa_six_softmax_fused_scalar_values)
                    attention_cached_gqa_six_softmax_fused_scalar_values_kernel<<<
                        count*kKVHeads,256,0,stream>>>(v,
                            output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_triple_values_fused)
                    attention_cached_gqa_six_softmax_triple_values_fused_kernel<<<
                        count*kKVHeads,256,0,stream>>>(v,
                            output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            scores,count,position,capacity,nullptr,first);
                else {
                    if(gqa_six_softmax_tile512)
                        attention_cached_gqa_six_softmax_tile512_kernel<<<
                            count*kKVHeads,256,0,stream>>>(
                                scores,count,position,capacity,nullptr,first);
                    else attention_cached_gqa_six_softmax_kernel<<<
                            count*kKVHeads,256,0,stream>>>(
                                scores,count,position,capacity,nullptr,first);
                if(gqa_six_softmax_triple_values_v_tile64)
                    attention_cached_gqa_triple_normalized_values_v_tile_kernel<64><<<
                        count*kKVHeads,256,
                        Exl3FullAttentionLayer::gqa_six_softmax_triple_v_tile_shared_bytes(64),stream>>>(
                            v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_triple_values_v_tile)
                    attention_cached_gqa_triple_normalized_values_v_tile_kernel<16><<<
                        count*kKVHeads,256,
                        Exl3FullAttentionLayer::gqa_six_softmax_triple_v_tile_shared_bytes(),stream>>>(
                            v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_triple_values_full_cta)
                    attention_cached_gqa_triple_normalized_values_full_cta_kernel<<<
                        count*kKVHeads,256,0,stream>>>(v,
                            output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_triple_values_scalar_dim)
                    attention_cached_gqa_triple_normalized_values_scalar_dim_kernel<<<
                        count*(kQHeads/3),256,0,stream>>>(v,
                            output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_triple_values_two_query&&count>1)
                    attention_cached_gqa_triple_normalized_values_two_query_kernel<<<
                        ((count+1)/2)*(kQHeads/3),128,0,stream>>>(v,
                        output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                        scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_triple_values_pair_dimensions)
                    attention_cached_gqa_triple_normalized_values_kernel<false,true><<<
                        count*(kQHeads/3),64,0,stream>>>(v,
                        output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                        scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_six_values_single_load)
                    attention_cached_gqa_six_normalized_values_single_load_kernel<<<
                        count*kKVHeads,128,0,stream>>>(v,
                            output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_six_values_scalar_single_load)
                    attention_cached_gqa_six_normalized_values_scalar_single_load_kernel<<<
                        count*kKVHeads,256,0,stream>>>(v,
                            output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_triple_values_warp_score_broadcast)
                    attention_cached_gqa_triple_normalized_values_kernel<false,false,true><<<
                        count*(kQHeads/3),256,0,stream>>>(v,
                        output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                        scores,count,position,capacity,nullptr,first);
                else if(gqa_six_softmax_triple_values_key_pair_pipeline)
                    attention_cached_gqa_triple_normalized_values_kernel<false,false,false,true><<<
                        count*(kQHeads/3),256,0,stream>>>(v,
                        output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                        scores,count,position,capacity,nullptr,first);
                else if(segmented_pages.count)
                    attention_cached_gqa_triple_normalized_values_kernel<true><<<
                        count*(kQHeads/3),256,0,stream>>>(v,
                        output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                        scores,count,position,capacity,nullptr,first,
                        segmented_pages);
                else attention_cached_gqa_triple_normalized_values_kernel<false><<<
                    count*(kQHeads/3),256,0,stream>>>(v,
                    output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                    scores,count,position,capacity,nullptr,first);
                }
            } else if(gqa_triple_values4)
                attention_cached_gqa_triple_values_kernel<true,true><<<count*(kQHeads/3),256,0,stream>>>(v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
            else
                attention_cached_gqa_triple_values_kernel<true><<<count*(kQHeads/3),256,0,stream>>>(v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
        } else if(gqa_triple) {
            attention_cached_gqa_triple_scores_kernel<<<count*kQHeads,256,0,stream>>>(
                q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,scores,count,position,capacity,nullptr,first);
            const int value_threads=gqa_triple_values128?128:256;
            if(gqa_triple_softmax_staged) attention_cached_gqa_triple_values_kernel<true><<<count*(kQHeads/3),value_threads,0,stream>>>(v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
            else attention_cached_gqa_triple_values_kernel<false><<<count*(kQHeads/3),value_threads,0,stream>>>(v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
        } else if(gqa_pair) {
            attention_cached_gqa_pair_scores_kernel<false><<<count*kQHeads,256,0,stream>>>(
                q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,scores,count,position,capacity,nullptr,first);
            attention_cached_gqa_pair_values_kernel<false><<<count*(kQHeads/2),256,0,stream>>>(
                v,output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
        } else if(v_half2)
            attention_cached_parallel_q_shared_kernel<true,true><<<count*kQHeads,256,0,stream>>>(
                q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,v,
                output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
        else if(k_half2)
            attention_cached_parallel_q_shared_kernel<true,false><<<count*kQHeads,256,0,stream>>>(
                q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,v,
                output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
        else if(q_shared)
            attention_cached_parallel_q_shared_kernel<false,false><<<count*kQHeads,256,0,stream>>>(
                q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,v,
                output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
        else
            attention_cached_parallel_kernel<false><<<count*kQHeads,256,0,stream>>>(
                q+static_cast<std::size_t>(first)*kQHeads*kHeadDim,k,v,
                output+static_cast<std::size_t>(first)*kQHeads*kHeadDim,scores,count,position,capacity,nullptr,first);
    }
    else attention_cached_kernel<<<rows*kQHeads,256,(position+rows)*sizeof(float),stream>>>(
        q,k,v,output,rows,position,capacity,nullptr);
    cuda_check(cudaGetLastError(),"exact attention qualification launch");
}

void exl3_numeric_attention_tiled_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream) {
    if(!q||!k||!v||!output||rows<1||rows>1024||position<0||
       capacity<rows||position>capacity-rows)
        throw std::invalid_argument("numeric tiled attention qualification extent");
    attention_cached_gqa_six_online_tiled_kernel<<<rows*kKVHeads,256,0,stream>>>(
        q,k,v,output,rows,position,capacity,nullptr,0);
    cuda_check(cudaGetLastError(),"numeric tiled attention qualification launch");
}

void exl3_numeric_attention_rows4_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream) {
    if(!q||!k||!v||!output||rows<1||rows>1024||position<0||
       capacity<rows||position>capacity-rows)
        throw std::invalid_argument("numeric rows4 attention qualification extent");
    const int blocks=((rows+3)/4)*kKVHeads;
    attention_cached_gqa_six_online_rows_kernel<4><<<blocks,256,0,stream>>>(
        q,k,v,output,rows,position,capacity,nullptr,0);
    cuda_check(cudaGetLastError(),"numeric rows4 attention qualification launch");
}

void exl3_numeric_attention_rows2_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream) {
    if(!q||!k||!v||!output||rows<1||rows>1024||position<0||
       capacity<rows||position>capacity-rows)
        throw std::invalid_argument("numeric rows2 attention qualification extent");
    const int blocks=((rows+1)/2)*kKVHeads;
    attention_cached_gqa_six_online_rows_kernel<2><<<blocks,256,0,stream>>>(
        q,k,v,output,rows,position,capacity,nullptr,0);
    cuda_check(cudaGetLastError(),"numeric rows2 attention qualification launch");
}

void exl3_prefill_attention_shared_scores_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream,bool threads256,
    bool parallel_softmax,bool head_split256,bool dimension_split256) {
    launch_prefill_attention_shared_scores(
        q,k,v,output,rows,position,capacity,stream,threads256,parallel_softmax,
        head_split256,dimension_split256);
    cuda_check(cudaGetLastError(),"exact shared-score prefill qualification launch");
}

Exl3PrefillAttentionSharedScoreSnapshot
exl3_prefill_attention_shared_score_global_snapshot() noexcept {
    return {
        prefill_shared_score_launch_counter.load(std::memory_order_relaxed),
        prefill_shared_score_row_counter.load(std::memory_order_relaxed),
        prefill_shared_score_bytes_counter.load(std::memory_order_relaxed),
        prefill_shared_score_threads256_counter.load(std::memory_order_relaxed),
        prefill_shared_score_parallel_softmax_counter.load(std::memory_order_relaxed),
        prefill_shared_score_head_split256_counter.load(std::memory_order_relaxed),
        prefill_shared_score_dimension_split256_counter.load(std::memory_order_relaxed)};
}

std::uint64_t exl3_fast_fused_flash_attention_calls_for_test() noexcept {
    return fast_fused_flash_attention_counter.load(std::memory_order_relaxed);
}
std::uint64_t exl3_fast_fused_flash_multirow_attention_calls_for_test() noexcept {
    return fast_fused_flash_multirow_attention_counter.load(std::memory_order_relaxed);
}

void fast_wmma_attention_fixture(bool rows32,const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,const int* position_device,
    int query_offset,cudaStream_t stream,bool padded) {
    if(rows32 && padded)
        attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim+16><<<
            dim3((rows+31)/32,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,query_offset,
                nullptr,nullptr);
    else if(rows32)
        attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim><<<
            dim3((rows+31)/32,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,query_offset,
                nullptr,nullptr);
    else
        attention_cached_gqa_six_wmma_prefill_kernel<<<
            dim3((rows+15)/16,kKVHeads,kQHeads/(kKVHeads*2)),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,query_offset);
    cuda_check(cudaGetLastError(),"WMMA attention fixture launch");
}

// Register-resident twin of attention_cached_gqa_six_wmma32_prefill_kernel
// <*,SplitCount,32>. Each warp owns 16 query rows of one query head across the
// whole key scan; Heads query heads of one KV head share every K/V tile. The
// arithmetic is the reference's: 32-key tiles in the same order, the same
// m16n8k16 HMMA k-sequence for S and for P x V, exact quad max, the same
// exp2f/half rounding of P, an in-order FP32 tile sum and the same running
// max/denominator/rescale updates. Only operand residency changes: the FP32
// output accumulators stay in registers instead of shared memory round trips.
constexpr int kRegAttnStride=kHeadDim+8;
constexpr int kRegAttnPStride=32+8;

template<int Heads,bool QGlobal=false>
constexpr std::size_t reg_attn_shared_bytes() {
    return ((QGlobal?0:static_cast<std::size_t>(Heads)*32*kRegAttnStride)+
            2*32*kRegAttnStride+2*Heads*16*kRegAttnPStride)*sizeof(half);
}

bool wmma32_register_q_global() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_WMMA32_REGISTER_Q_GLOBAL");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

__device__ __forceinline__ void reg_attn_ldmatrix_x4(unsigned (&r)[4],const half* p) {
    const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
}

__device__ __forceinline__ void reg_attn_ldmatrix_x4_trans(unsigned (&r)[4],const half* p) {
    const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 :"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address));
}

__device__ __forceinline__ void reg_attn_mma(float (&c)[4],const unsigned (&a)[4],
                                             unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}

__device__ __forceinline__ unsigned reg_attn_pack(half low,half high) {
    return static_cast<unsigned>(__half_as_ushort(low))|
        (static_cast<unsigned>(__half_as_ushort(high))<<16);
}

// SplitBlockM names the reference row block whose frontier defines the split
// boundaries and capacity check (32 for WMMA32, 64 for the M64 twin). Extra
// fully masked tiles are exact no-ops for rows of the smaller block.
// QGlobal reads the identical m16n8k16 A fragments of Q straight from global
// memory (zero for rows beyond the live block) instead of staging Q in shared
// memory, halving the CTA footprint so two CTAs fit per SM.
template<int SplitCount,int Heads,int SplitBlockM=32,bool QGlobal=false>
__global__ void __launch_bounds__(64*Heads) attention_gqa_six_wmma32_register_prefill_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,
    int position,int capacity,const int* position_device,int query_offset,
    float* split_output,float* split_stats) {
    constexpr int BlockM=32;
    constexpr int BlockN=32;
    constexpr int kHeadsPerKV=kQHeads/kKVHeads;
    constexpr int kVectors=kHeadDim/8;
    static_assert(kHeadsPerKV%Heads==0);
    static_assert(SplitCount==1 || SplitCount==2 || SplitCount==4);
    extern __shared__ __align__(16) unsigned char reg_attn_shared[];
    half* q_s=reinterpret_cast<half*>(reg_attn_shared);
    half* k_s=q_s+(QGlobal?0:Heads*BlockM*kRegAttnStride);
    half* v_s=k_s+BlockN*kRegAttnStride;
    half* p_s=v_s+BlockN*kRegAttnStride;

    const int query_base=static_cast<int>(blockIdx.x)*BlockM;
    const int kv_head=static_cast<int>(blockIdx.y);
    const int head_group=static_cast<int>(blockIdx.z)/SplitCount;
    const int split=static_cast<int>(blockIdx.z)%SplitCount;
    const int tid=static_cast<int>(threadIdx.x);
    const int warp=tid>>5;
    const int lane=tid&31;
    const int q_head_base=kv_head*kHeadsPerKV+head_group*Heads;
    if(query_base>=rows || kv_head>=kKVHeads) return;
    static_assert(SplitBlockM==32 || SplitBlockM==64);
    const int active_rows=min(BlockM,rows-query_base);
    const int base=(position_device?*position_device:position)+query_offset;
    const int maximum_count=base+query_base+active_rows;
    const int split_base=(query_base/SplitBlockM)*SplitBlockM;
    const int split_count=base+split_base+min(SplitBlockM,rows-split_base);
    if(base+split_base+1<1 || split_count>capacity) return;
    const int segment_tiles=(split_count+SplitCount*BlockN-1)/(SplitCount*BlockN);
    const int first_begin=SplitCount==1?0:
        min(split*segment_tiles*BlockN,split_count);
    const int first_end=SplitCount==1?maximum_count:
        min((split+1)*segment_tiles*BlockN,split_count);

    if constexpr(!QGlobal)
    for(int index=tid;index<Heads*BlockM*kVectors;index+=blockDim.x) {
        const int head=index/(BlockM*kVectors);
        const int rem=index%(BlockM*kVectors);
        const int row=rem/kVectors;
        const int dim=(rem%kVectors)*8;
        uint4 bits=make_uint4(0,0,0,0);
        if(row<active_rows)
            bits=*reinterpret_cast<const uint4*>(q+
                (static_cast<std::size_t>(query_base+row)*kQHeads+q_head_base+head)*
                    kHeadDim+dim);
        *reinterpret_cast<uint4*>(q_s+(head*BlockM+row)*kRegAttnStride+dim)=bits;
    }

    const int head_local=warp>>1;
    const int m_base=(warp&1)*16;
    const int g=lane>>2;
    const int t=lane&3;
    const int row0=m_base+g;
    const int row1=row0+8;
    const half* q_warp=q_s+(QGlobal?0:(head_local*BlockM+m_base)*kRegAttnStride);
    const bool q_row0_live=row0<active_rows;
    const bool q_row1_live=row1<active_rows;
    const auto* q_global0=reinterpret_cast<const unsigned*>(q+
        (static_cast<std::size_t>(query_base+(q_row0_live?row0:0))*kQHeads+
            q_head_base+head_local)*kHeadDim+2*t);
    const auto* q_global1=reinterpret_cast<const unsigned*>(q+
        (static_cast<std::size_t>(query_base+(q_row1_live?row1:0))*kQHeads+
            q_head_base+head_local)*kHeadDim+2*t);
    half* p_warp=p_s+warp*16*kRegAttnPStride;
    float acc[kHeadDim/8][4];
    #pragma unroll
    for(int n=0;n<kHeadDim/8;++n)
        acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.0f;
    float running_max[2]={-3.402823466e+38F,-3.402823466e+38F};
    float denominator[2]={0.0f,0.0f};

    for(int first=first_begin;first<first_end;first+=BlockN) {
        __syncthreads();
        for(int index=tid;index<BlockN*kVectors;index+=blockDim.x) {
            const int key_offset=index/kVectors;
            const int dim=(index%kVectors)*8;
            const int key=first+key_offset;
            uint4 key_bits=make_uint4(0,0,0,0),value_bits=make_uint4(0,0,0,0);
            if(key<first_end && key<capacity) {
                const std::size_t offset=(static_cast<std::size_t>(key)*kKVHeads+
                    kv_head)*kHeadDim+dim;
                key_bits=*reinterpret_cast<const uint4*>(k_cache+offset);
                value_bits=*reinterpret_cast<const uint4*>(v_cache+offset);
            }
            *reinterpret_cast<uint4*>(k_s+key_offset*kRegAttnStride+dim)=key_bits;
            *reinterpret_cast<uint4*>(v_s+key_offset*kRegAttnStride+dim)=value_bits;
        }
        __syncthreads();

        float s[4][4];
        #pragma unroll
        for(int j=0;j<4;++j) s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.0f;
        #pragma unroll
        for(int kk=0;kk<kHeadDim;kk+=16) {
            unsigned a[4];
            if constexpr(QGlobal) {
                a[0]=q_row0_live?q_global0[kk/2]:0u;
                a[1]=q_row1_live?q_global1[kk/2]:0u;
                a[2]=q_row0_live?q_global0[kk/2+4]:0u;
                a[3]=q_row1_live?q_global1[kk/2+4]:0u;
            } else
            reg_attn_ldmatrix_x4(a,q_warp+(lane&15)*kRegAttnStride+kk+(lane>>4)*8);
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
                unsigned b[4];
                reg_attn_ldmatrix_x4(b,k_s+(pair*16+(lane&7)+((lane>>4)<<3))*
                    kRegAttnStride+kk+((lane>>3)&1)*8);
                reg_attn_mma(s[2*pair],a,b[0],b[1]);
                reg_attn_mma(s[2*pair+1],a,b[2],b[3]);
            }
        }

        const int in_range=first_end-first;
        const int causal=base+query_base-first;
        float tile_max[2]={-3.402823466e+38F,-3.402823466e+38F};
        #pragma unroll
        for(int j=0;j<4;++j) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                const int row=e<2?row0:row1;
                const int offset=8*j+2*t+(e&1);
                if(row<active_rows && offset<in_range && offset<=causal+row)
                    tile_max[e>>1]=fmaxf(tile_max[e>>1],s[j][e]*0.0625f);
            }
        }
        #pragma unroll
        for(int i=0;i<2;++i) {
            tile_max[i]=fmaxf(tile_max[i],__shfl_xor_sync(0xffffffffU,tile_max[i],1));
            tile_max[i]=fmaxf(tile_max[i],__shfl_xor_sync(0xffffffffU,tile_max[i],2));
        }
        float old_scale[2];
        #pragma unroll
        for(int i=0;i<2;++i) {
            const float next_max=fmaxf(running_max[i],tile_max[i]);
            old_scale[i]=denominator[i]==0.0f?0.0f:
                exp2f((running_max[i]-next_max)*1.4426950408889634f);
            running_max[i]=next_max;
        }
        half p[4][4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            #pragma unroll
            for(int e=0;e<4;++e) {
                const int row=e<2?row0:row1;
                const int offset=8*j+2*t+(e&1);
                const bool valid=row<active_rows && offset<in_range &&
                    offset<=causal+row;
                const float score=s[j][e]*0.0625f;
                p[j][e]=__float2half(valid?exp2f((score-running_max[e>>1])*
                    1.4426950408889634f):0.0f);
            }
            *reinterpret_cast<unsigned*>(p_warp+g*kRegAttnPStride+8*j+2*t)=
                reg_attn_pack(p[j][0],p[j][1]);
            *reinterpret_cast<unsigned*>(p_warp+(g+8)*kRegAttnPStride+8*j+2*t)=
                reg_attn_pack(p[j][2],p[j][3]);
        }
        __syncwarp();
        float tile_sum=0.0f;
        if(lane<16) {
            const int extent=min(BlockN,in_range);
            for(int offset=0;offset<extent;++offset)
                tile_sum+=__half2float(p_warp[lane*kRegAttnPStride+offset]);
        }
        const float sums[2]={__shfl_sync(0xffffffffU,tile_sum,g),
                             __shfl_sync(0xffffffffU,tile_sum,g+8)};
        #pragma unroll
        for(int i=0;i<2;++i)
            denominator[i]=denominator[i]*old_scale[i]+sums[i];
        #pragma unroll
        for(int n=0;n<kHeadDim/8;++n) {
            acc[n][0]*=old_scale[0];
            acc[n][1]*=old_scale[0];
            acc[n][2]*=old_scale[1];
            acc[n][3]*=old_scale[1];
        }
        #pragma unroll
        for(int kk=0;kk<2;++kk) {
            const unsigned a[4]={reg_attn_pack(p[2*kk][0],p[2*kk][1]),
                                 reg_attn_pack(p[2*kk][2],p[2*kk][3]),
                                 reg_attn_pack(p[2*kk+1][0],p[2*kk+1][1]),
                                 reg_attn_pack(p[2*kk+1][2],p[2*kk+1][3])};
            #pragma unroll
            for(int pair=0;pair<kHeadDim/16;++pair) {
                unsigned b[4];
                reg_attn_ldmatrix_x4_trans(b,v_s+(16*kk+(lane&7)+((lane>>3)&1)*8)*
                    kRegAttnStride+pair*16+(lane>>4)*8);
                reg_attn_mma(acc[2*pair],a,b[0],b[1]);
                reg_attn_mma(acc[2*pair+1],a,b[2],b[3]);
            }
        }
        __syncwarp();
    }

    const int q_head=q_head_base+head_local;
    #pragma unroll
    for(int i=0;i<2;++i) {
        const int row=i==0?row0:row1;
        if(row>=active_rows) continue;
        const std::size_t row_base=(static_cast<std::size_t>(query_base+row)*kQHeads+
            q_head)*kHeadDim;
        #pragma unroll
        for(int n=0;n<kHeadDim/8;++n) {
            const int dim=8*n+2*t;
            const float first_value=denominator[i]>0.0f?
                acc[n][2*i]/denominator[i]:0.0f;
            const float second_value=denominator[i]>0.0f?
                acc[n][2*i+1]/denominator[i]:0.0f;
            if constexpr(SplitCount>1) {
                float* destination=split_output+static_cast<std::size_t>(split)*rows*
                    kQHeads*kHeadDim+row_base+dim;
                destination[0]=first_value;
                destination[1]=second_value;
            } else {
                output[row_base+dim]=__half_as_ushort(__float2half_rn(first_value));
                output[row_base+dim+1]=__half_as_ushort(__float2half_rn(second_value));
            }
        }
        if constexpr(SplitCount>1) {
            if(t==0) {
                const std::size_t stat=(static_cast<std::size_t>(split)*rows*kQHeads+
                    (query_base+row)*kQHeads+q_head)*2;
                split_stats[stat]=running_max[i];
                split_stats[stat+1]=denominator[i];
            }
        }
    }
}


// FA2-style GQA-6 prefill attention (NINFER_EXL3_FA2_PREFILL, default 1; 0 =
// the register WMMA32 route). Eight warps cover two query heads x 64 rows and
// share every 32-key K/V tile; Q A-fragments stay in registers; K/V tiles are
// double buffered with cp.async; causal masking only on diagonal tiles; row
// sums come from the register P values. Chunks whose causal extent reaches
// 2048 keys split the key range four ways (longest row blocks first) and merge
// through attention_wmma32_split_merge_kernel. Numerical profile: FP16 Q/K/V
// and P, FP32 scores/accumulators; differs from the WMMA32 route only in FP32
// summation order.
namespace fa2_prefill {
constexpr int kStride=kHeadDim+8;
constexpr int BM=64, BN=32;
constexpr int kStage=2*BN*kStride; // halfs (K then V)
constexpr std::size_t smem_bytes(){ return 2u*kStage*sizeof(half); }
__device__ __forceinline__ void cp16(void* dst,const void* src,bool valid){
  const unsigned d=static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"::"r"(d),"l"(src),"r"(valid?16:0));
}
__device__ __forceinline__ void commit(){ asm volatile("cp.async.commit_group;\n"); }
template<int N> __device__ __forceinline__ void wait(){ asm volatile("cp.async.wait_group %0;\n"::"n"(N)); }
template<int S>
__global__ void __launch_bounds__(256,1) attention_gqa_six_fa2_prefill_kernel(const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,std::uint16_t* output,int rows,int base,int capacity,
    float* split_output,float* split_stats){
  extern __shared__ __align__(16) unsigned char smem_raw[];
  half* sm=reinterpret_cast<half*>(smem_raw);
  const int query_base=(static_cast<int>(gridDim.x)-1-static_cast<int>(blockIdx.x))*BM;
  const int kv_head=blockIdx.y;
  const int head_group=blockIdx.z/S, split=blockIdx.z%S;
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31;
  const int head_local=warp>>2, m_base=(warp&3)*16;
  const int g=lane>>2, t=lane&3;
  const int row0=m_base+g, row1=row0+8;
  const int q_head=kv_head*(kQHeads/kKVHeads)+head_group*2+head_local;
  const int active_rows=min(BM,rows-query_base);
  if(active_rows<=0) return;
  const int maximum_count=min(base+query_base+active_rows,capacity);
  const int tiles=(maximum_count+BN-1)/BN;
  const int per=(tiles+S-1)/S;
  const int t_begin=split*per, t_end=min(tiles,t_begin+per);
  unsigned qa[kHeadDim/16][4];
  {
    const bool l0=row0<active_rows, l1=row1<active_rows;
    const unsigned* q0=reinterpret_cast<const unsigned*>(q+(static_cast<std::size_t>(query_base+(l0?row0:0))*kQHeads+q_head)*kHeadDim+2*t);
    const unsigned* q1=reinterpret_cast<const unsigned*>(q+(static_cast<std::size_t>(query_base+(l1?row1:0))*kQHeads+q_head)*kHeadDim+2*t);
    #pragma unroll
    for(int kk=0;kk<kHeadDim/16;++kk){
      qa[kk][0]=l0?q0[kk*8]:0u; qa[kk][1]=l1?q1[kk*8]:0u;
      qa[kk][2]=l0?q0[kk*8+4]:0u; qa[kk][3]=l1?q1[kk*8+4]:0u;
    }
  }
  float acc[kHeadDim/8][4];
  #pragma unroll
  for(int n=0;n<kHeadDim/8;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.f;
  float run_max[2]={-3.402823466e+38F,-3.402823466e+38F}, den[2]={0.f,0.f};
  auto load=[&](int tile,int stage){
    half* ks=sm+stage*kStage; half* vs=ks+BN*kStride;
    const int first=tile*BN;
    #pragma unroll
    for(int i=0;i<(BN*kHeadDim/8)/256;++i){
      const int idx=tid+i*256; const int key_off=idx/(kHeadDim/8); const int dim=(idx%(kHeadDim/8))*8;
      const int key=first+key_off; const bool valid=key<maximum_count;
      const std::size_t off=(static_cast<std::size_t>(valid?key:0)*kKVHeads+kv_head)*kHeadDim+dim;
      cp16(ks+key_off*kStride+dim,k_cache+off,valid);
      cp16(vs+key_off*kStride+dim,v_cache+off,valid);
    }
    commit();
  };
  if(t_begin<t_end) load(t_begin,0);
  const int warp_min_limit=base+query_base+m_base;
  const int warp_max_limit=base+query_base+m_base+15;
  for(int tile=t_begin;tile<t_end;++tile){
    const int stage=(tile-t_begin)&1;
    if(tile+1<t_end){ load(tile+1,stage^1); wait<1>(); } else wait<0>();
    __syncthreads();
    const int first=tile*BN;
    const half* ks=sm+stage*kStage; const half* vs=ks+BN*kStride;
    const bool warp_live=(m_base<active_rows) && first<=warp_max_limit;
    if(warp_live){
      float s[BN/8][4];
      #pragma unroll
      for(int j=0;j<BN/8;++j) s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.f;
      #pragma unroll
      for(int kk=0;kk<kHeadDim/16;++kk){
        #pragma unroll
        for(int pair=0;pair<BN/16;++pair){
          unsigned b[4];
          reg_attn_ldmatrix_x4(b,ks+(pair*16+(lane&7)+((lane>>4)<<3))*kStride+kk*16+((lane>>3)&1)*8);
          reg_attn_mma(s[2*pair],qa[kk],b[0],b[1]);
          reg_attn_mma(s[2*pair+1],qa[kk],b[2],b[3]);
        }
      }
      const bool need_mask=first+BN-1>warp_min_limit;
      float tmax[2]={-3.402823466e+38F,-3.402823466e+38F};
      #pragma unroll
      for(int j=0;j<BN/8;++j)
        #pragma unroll
        for(int e=0;e<4;++e){
          const int row=e<2?row0:row1; const int key=first+8*j+2*t+(e&1);
          float sc=s[j][e]*0.0625f;
          if(need_mask && (key>base+query_base+row || row>=active_rows)) sc=-3.402823466e+38F;
          s[j][e]=sc; tmax[e>>1]=fmaxf(tmax[e>>1],sc);
        }
      #pragma unroll
      for(int i=0;i<2;++i){ tmax[i]=fmaxf(tmax[i],__shfl_xor_sync(0xffffffffu,tmax[i],1)); tmax[i]=fmaxf(tmax[i],__shfl_xor_sync(0xffffffffu,tmax[i],2)); }
      float scale[2];
      #pragma unroll
      for(int i=0;i<2;++i){ const float nm=fmaxf(run_max[i],tmax[i]); scale[i]=(nm==run_max[i])?1.f:exp2f((run_max[i]-nm)*1.4426950408889634f); run_max[i]=nm; }
      unsigned pp[BN/8][2]; float rs[2]={0.f,0.f};
      #pragma unroll
      for(int j=0;j<BN/8;++j){
        half p[4];
        #pragma unroll
        for(int e=0;e<4;++e){
          const float x=s[j][e];
          p[e]=__float2half(x==-3.402823466e+38F?0.f:exp2f((x-run_max[e>>1])*1.4426950408889634f));
          rs[e>>1]+=__half2float(p[e]);
        }
        pp[j][0]=reg_attn_pack(p[0],p[1]); pp[j][1]=reg_attn_pack(p[2],p[3]);
      }
      #pragma unroll
      for(int i=0;i<2;++i){ rs[i]+=__shfl_xor_sync(0xffffffffu,rs[i],1); rs[i]+=__shfl_xor_sync(0xffffffffu,rs[i],2); den[i]=den[i]*scale[i]+rs[i]; }
      if(!__all_sync(0xffffffffu,scale[0]==1.f && scale[1]==1.f)){
        #pragma unroll
        for(int n=0;n<kHeadDim/8;++n){ acc[n][0]*=scale[0]; acc[n][1]*=scale[0]; acc[n][2]*=scale[1]; acc[n][3]*=scale[1]; }
      }
#if 0
      unsigned hacc[kHeadDim/8][2];
      #pragma unroll
      for(int n=0;n<kHeadDim/8;++n) hacc[n][0]=hacc[n][1]=0u;
#endif
      #pragma unroll
      for(int kk=0;kk<BN/16;++kk){
        const unsigned a[4]={pp[2*kk][0],pp[2*kk][1],pp[2*kk+1][0],pp[2*kk+1][1]};
        #pragma unroll
        for(int pair=0;pair<kHeadDim/16;++pair){
          unsigned b[4];
          reg_attn_ldmatrix_x4_trans(b,vs+(16*kk+(lane&7)+((lane>>3)&1)*8)*kStride+pair*16+(lane>>4)*8);
#if 0
          mma_h(hacc[2*pair],a,b[0],b[1]); mma_h(hacc[2*pair+1],a,b[2],b[3]);
#else
          reg_attn_mma(acc[2*pair],a,b[0],b[1]); reg_attn_mma(acc[2*pair+1],a,b[2],b[3]);
#endif
        }
      }
#if 0
      #pragma unroll
      for(int n=0;n<kHeadDim/8;++n){
        const float2 lo=__half22float2(*reinterpret_cast<half2*>(&hacc[n][0]));
        const float2 hi=__half22float2(*reinterpret_cast<half2*>(&hacc[n][1]));
        acc[n][0]+=lo.x; acc[n][1]+=lo.y; acc[n][2]+=hi.x; acc[n][3]+=hi.y;
      }
#endif
    }
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<2;++i){
    const int row=i==0?row0:row1;
    if(row>=active_rows) continue;
    const std::size_t row_base=(static_cast<std::size_t>(query_base+row)*kQHeads+q_head)*kHeadDim;
    const float inv=den[i]>0.f?1.f/den[i]:0.f;
    #pragma unroll
    for(int n=0;n<kHeadDim/8;++n){
      const int dim=8*n+2*t;
      const float a0=acc[n][2*i]*inv, a1=acc[n][2*i+1]*inv;
      if constexpr(S>1){
        float2* d=reinterpret_cast<float2*>(split_output+static_cast<std::size_t>(split)*rows*kQHeads*kHeadDim+row_base+dim);
        *d=make_float2(a0,a1);
      } else {
        *reinterpret_cast<unsigned*>(output+row_base+dim)=reg_attn_pack(__float2half_rn(a0),__float2half_rn(a1));
      }
    }
    if constexpr(S>1) if(t==0){
      const std::size_t stat=(static_cast<std::size_t>(split)*rows*kQHeads+(query_base+row)*kQHeads+q_head)*2;
      split_stats[stat]=run_max[i]; split_stats[stat+1]=den[i];
    }
  }
}
} // namespace fa2_prefill

bool exl3_fa2_prefill_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_FA2_PREFILL");
        if(!value || std::strcmp(value,"1")==0) return true;
        if(std::strcmp(value,"0")==0) return false;
        throw std::invalid_argument("NINFER_EXL3_FA2_PREFILL must be 0 or 1");
    }();
    return enabled;
}

template<int S>
static void launch_fa2_prefill_variant(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,std::uint16_t* output,int rows,int position,int capacity,
    float* split_output,float* split_stats,cudaStream_t stream) {
    constexpr std::size_t bytes=fa2_prefill::smem_bytes();
    static const bool configured=[] {
        cuda_check(cudaFuncSetAttribute(fa2_prefill::attention_gqa_six_fa2_prefill_kernel<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(bytes)),
            "configure FA2 prefill shared memory");
        return true;
    }();
    (void)configured;
    fa2_prefill::attention_gqa_six_fa2_prefill_kernel<S><<<
        dim3((rows+fa2_prefill::BM-1)/fa2_prefill::BM,kKVHeads,(kQHeads/kKVHeads/2)*S),256,
        bytes,stream>>>(q,k,v,output,rows,position,capacity,split_output,split_stats);
    cuda_check(cudaGetLastError(),"launch FA2 prefill attention");
    if constexpr(S>1) {
        attention_wmma32_split_merge_kernel<S><<<(rows*kQHeads*kHeadDim+255)/256,256,0,stream>>>(
            split_output,split_stats,output,rows);
        cuda_check(cudaGetLastError(),"launch FA2 prefill split merge");
    }
}

static void launch_fa2_prefill(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,std::uint16_t* output,int rows,int position,int capacity,
    float* split_output,float* split_stats,cudaStream_t stream) {
    if(position+rows>=2048)
        launch_fa2_prefill_variant<4>(q,k,v,output,rows,position,capacity,split_output,split_stats,stream);
    else
        launch_fa2_prefill_variant<1>(q,k,v,output,rows,position,capacity,split_output,split_stats,stream);
}

template<int SplitCount,int Heads,int SplitBlockM=32,bool QGlobal=false>
void launch_wmma32_register_prefill_variant(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,std::uint16_t* output,int rows,int position,int capacity,
    const int* position_device,int query_offset,float* split_output,float* split_stats,
    cudaStream_t stream) {
    constexpr std::size_t bytes=reg_attn_shared_bytes<Heads,QGlobal>();
    static const bool configured=[] {
        cuda_check(cudaFuncSetAttribute(
            attention_gqa_six_wmma32_register_prefill_kernel<SplitCount,Heads,SplitBlockM,QGlobal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(bytes)),
            "configure register WMMA32 prefill shared memory");
        return true;
    }();
    (void)configured;
    attention_gqa_six_wmma32_register_prefill_kernel<SplitCount,Heads,SplitBlockM,QGlobal><<<
        dim3((rows+31)/32,kKVHeads,(kQHeads/kKVHeads/Heads)*SplitCount),64*Heads,
        bytes,stream>>>(q,k,v,output,rows,position,capacity,position_device,
            query_offset,split_output,split_stats);
}

template<int SplitCount,int Heads,int SplitBlockM=32>
void launch_wmma32_register_prefill(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,std::uint16_t* output,int rows,int position,int capacity,
    const int* position_device,int query_offset,float* split_output,float* split_stats,
    cudaStream_t stream) {
    if(wmma32_register_q_global())
        launch_wmma32_register_prefill_variant<SplitCount,Heads,SplitBlockM,true>(q,k,v,
            output,rows,position,capacity,position_device,query_offset,split_output,
            split_stats,stream);
    else
        launch_wmma32_register_prefill_variant<SplitCount,Heads,SplitBlockM,false>(q,k,v,
            output,rows,position,capacity,position_device,query_offset,split_output,
            split_stats,stream);
}

int wmma32_register_heads() {
    static const int heads=[] {
        const char* value=std::getenv("NINFER_EXL3_WMMA32_REGISTER");
        if(!value) return 2;
        const int parsed=std::atoi(value);
        if(parsed!=0 && parsed!=1 && parsed!=2 && parsed!=3)
            throw std::invalid_argument("NINFER_EXL3_WMMA32_REGISTER must be 0, 1, 2 or 3");
        return parsed;
    }();
    return heads;
}

// Returns false when disabled; otherwise launches the split partials (or the
// final output for SplitCount 1).
template<int SplitCount,int SplitBlockM=32>
bool launch_wmma32_register_selected(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,std::uint16_t* output,int rows,int position,int capacity,
    const int* position_device,int query_offset,float* split_output,float* split_stats,
    cudaStream_t stream) {
    switch(wmma32_register_heads()) {
    case 1: launch_wmma32_register_prefill<SplitCount,1,SplitBlockM>(q,k,v,output,rows,position,capacity,
        position_device,query_offset,split_output,split_stats,stream); return true;
    case 2: launch_wmma32_register_prefill<SplitCount,2,SplitBlockM>(q,k,v,output,rows,position,capacity,
        position_device,query_offset,split_output,split_stats,stream); return true;
    case 3: launch_wmma32_register_prefill<SplitCount,3,SplitBlockM>(q,k,v,output,rows,position,capacity,
        position_device,query_offset,split_output,split_stats,stream); return true;
    default: return false;
    }
}

void fast_wmma32_register_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* split_output,float* split_stats,int rows,int position,int capacity,
    const int* position_device,int query_offset,int split_count,int heads,
    cudaStream_t stream,int split_block_m) {
    if(heads<1 || heads>3 || (split_count!=1 && split_count!=2 && split_count!=4) ||
       (split_block_m!=32 && split_block_m!=64))
        throw std::invalid_argument("register WMMA32 fixture geometry");
    auto run=[&](auto split_tag) {
        constexpr int S=decltype(split_tag)::value;
        auto launch=[&](auto heads_tag,auto block_tag) {
            launch_wmma32_register_prefill<S,decltype(heads_tag)::value,
                decltype(block_tag)::value>(q,k,v,output,rows,position,capacity,
                position_device,query_offset,split_output,split_stats,stream);
        };
        auto by_block=[&](auto heads_tag) {
            if(split_block_m==64) launch(heads_tag,std::integral_constant<int,64>{});
            else launch(heads_tag,std::integral_constant<int,32>{});
        };
        if(heads==1) by_block(std::integral_constant<int,1>{});
        else if(heads==2) by_block(std::integral_constant<int,2>{});
        else by_block(std::integral_constant<int,3>{});
        cuda_check(cudaGetLastError(),"register WMMA32 fixture launch");
        if constexpr(S>1) {
            attention_wmma32_split_merge_kernel<S><<<
                (rows*kQHeads*kHeadDim+255)/256,256,0,stream>>>(
                    split_output,split_stats,output,rows);
            cuda_check(cudaGetLastError(),"register WMMA32 fixture merge");
        }
    };
    if(split_count==1) run(std::integral_constant<int,1>{});
    else if(split_count==2) run(std::integral_constant<int,2>{});
    else run(std::integral_constant<int,4>{});
}

void fast_wmma_split2_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* split_output,float* split_stats,int rows,int position,int capacity,
    const int* position_device,int query_offset,cudaStream_t stream,bool padded) {
    if(padded)
        attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim+16,2><<<
            dim3((rows+31)/32,kKVHeads,2*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,query_offset,
                split_output,split_stats);
    else
        attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim,2><<<
            dim3((rows+31)/32,kKVHeads,2*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,query_offset,
                split_output,split_stats);
    cuda_check(cudaGetLastError(),"WMMA split2 attention fixture launch");
    attention_wmma32_split_merge_kernel<2><<<
        (rows*kQHeads*kHeadDim+255)/256,256,0,stream>>>(
            split_output,split_stats,output,rows);
    cuda_check(cudaGetLastError(),"WMMA split2 attention fixture merge");
}

void fast_wmma_split4_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* split_output,float* split_stats,int rows,int position,int capacity,
    const int* position_device,int query_offset,cudaStream_t stream,bool padded) {
    if(padded)
        attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim+16,4><<<
            dim3((rows+31)/32,kKVHeads,4*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,query_offset,
                split_output,split_stats);
    else
        attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim,4><<<
            dim3((rows+31)/32,kKVHeads,4*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,query_offset,
                split_output,split_stats);
    cuda_check(cudaGetLastError(),"WMMA split4 attention fixture launch");
    attention_wmma32_split_merge_kernel<4><<<
        (rows*kQHeads*kHeadDim+255)/256,256,0,stream>>>(
            split_output,split_stats,output,rows);
    cuda_check(cudaGetLastError(),"WMMA split4 attention fixture merge");
}

void fast_wmma64_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* split_output,float* split_stats,int rows,int position,int capacity,
    const int* position_device,int query_offset,cudaStream_t stream,
    bool padded,bool split2,bool register_owned,bool shared_heads,
    bool keys64) {
    if(keys64 && (!register_owned || shared_heads))
        throw std::invalid_argument("WMMA64 keys64 fixture requires one-head register route");
    // Partial blocks retain the repaired WMMA32 path; no Q fragment may read
    // beyond the represented query span.
    if(rows%64) {
        if(split2)
            fast_wmma_split2_attention_fixture(q,k,v,output,split_output,
                split_stats,rows,position,capacity,position_device,
                query_offset,stream,padded);
        else
            fast_wmma_attention_fixture(true,q,k,v,output,rows,position,
                capacity,position_device,query_offset,stream,padded);
        return;
    }
    if(split2) {
        if(shared_heads && padded)
            attention_cached_gqa_six_wmma64_register_prefill_kernel<
                kHeadDim+16,2,2><<<
                dim3(rows/32,kKVHeads,2*kQHeads/(kKVHeads*2)),
                256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,
                query_offset,split_output,split_stats);
        else if(shared_heads)
            attention_cached_gqa_six_wmma64_register_prefill_kernel<
                kHeadDim,2,2><<<
                dim3(rows/32,kKVHeads,2*kQHeads/(kKVHeads*2)),
                256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,
                query_offset,split_output,split_stats);
        else if(keys64 && padded)
            attention_cached_gqa_six_wmma64_register_prefill_kernel<
                kHeadDim+16,2,1,64><<<
                dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,
                query_offset,split_output,split_stats);
        else if(keys64)
            attention_cached_gqa_six_wmma64_register_prefill_kernel<
                kHeadDim,2,1,64><<<
                dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,
                query_offset,split_output,split_stats);
        else if(register_owned && padded)
            attention_cached_gqa_six_wmma64_register_prefill_kernel<
                kHeadDim+16,2><<<
                dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,
                query_offset,split_output,split_stats);
        else if(register_owned)
            attention_cached_gqa_six_wmma64_register_prefill_kernel<
                kHeadDim,2><<<
                dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,
                query_offset,split_output,split_stats);
        else if(padded)
            attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim+16,2,64><<<
                dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,
                query_offset,split_output,split_stats);
        else
            attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim,2,64><<<
                dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),256,0,stream>>>(
                q,k,v,output,rows,position,capacity,position_device,
                query_offset,split_output,split_stats);
        cuda_check(cudaGetLastError(),"WMMA64 split2 fixture partials");
        attention_wmma32_split_merge_kernel<2><<<
            (rows*kQHeads*kHeadDim+255)/256,256,0,stream>>>(
            split_output,split_stats,output,rows);
    } else if(shared_heads && padded)
        attention_cached_gqa_six_wmma64_register_prefill_kernel<
            kHeadDim+16,1,2><<<
            dim3(rows/32,kKVHeads,kQHeads/(kKVHeads*2)),256,0,stream>>>(
            q,k,v,output,rows,position,capacity,position_device,
            query_offset,nullptr,nullptr);
    else if(shared_heads)
        attention_cached_gqa_six_wmma64_register_prefill_kernel<
            kHeadDim,1,2><<<
            dim3(rows/32,kKVHeads,kQHeads/(kKVHeads*2)),256,0,stream>>>(
            q,k,v,output,rows,position,capacity,position_device,
            query_offset,nullptr,nullptr);
    else if(keys64 && padded)
        attention_cached_gqa_six_wmma64_register_prefill_kernel<
            kHeadDim+16,1,1,64><<<
            dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
            q,k,v,output,rows,position,capacity,position_device,
            query_offset,nullptr,nullptr);
    else if(keys64)
        attention_cached_gqa_six_wmma64_register_prefill_kernel<
            kHeadDim,1,1,64><<<
            dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
            q,k,v,output,rows,position,capacity,position_device,
            query_offset,nullptr,nullptr);
    else if(register_owned && padded)
        attention_cached_gqa_six_wmma64_register_prefill_kernel<
            kHeadDim+16,1><<<
            dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
            q,k,v,output,rows,position,capacity,position_device,
            query_offset,nullptr,nullptr);
    else if(register_owned)
        attention_cached_gqa_six_wmma64_register_prefill_kernel<
            kHeadDim,1><<<
            dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
            q,k,v,output,rows,position,capacity,position_device,
            query_offset,nullptr,nullptr);
    else if(padded)
        attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim+16,1,64><<<
            dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
            q,k,v,output,rows,position,capacity,position_device,
            query_offset,nullptr,nullptr);
    else
        attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim,1,64><<<
            dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
            q,k,v,output,rows,position,capacity,position_device,
            query_offset,nullptr,nullptr);
    cuda_check(cudaGetLastError(),"WMMA64 attention fixture launch");
}

// Tensor-core flash-decode for FP16-KV verifier/decode rows (numerics policy
// candidate; quality-gated). One CTA owns one KV head and one 256-key segment
// for up to eight query rows; the 6 sibling query heads of each row form 48
// query vectors, handled as three m16 tiles by three warps. QK^T and P x V use
// m16n8k16 HMMA with FP32 accumulation over 32-key chunks with an FP32 online
// softmax (natural-exponent domain, P rounded to FP16 for P x V). Results are
// written into the fused-flash scratch slot layout (unnormalized values plus
// segment max and denominator), so the existing segment merge is reused.
constexpr int kVerifyMmaChunk=32;
constexpr int kVerifyMmaStride=kHeadDim+8;

// Per-head twin of attention_cached_gqa_six_fused_flash_merge_kernel: one CTA
// per (row, KV head, query head) instead of one per (row, KV head). Every
// output performs the identical ordered max, scale, denominator and numerator
// chain, so the result is bitwise equal; only the parallel extent changes.
__global__ void attention_fused_flash_merge_heads_kernel(
    const float* workspace,std::uint16_t* output,int rows,int segments,
    int position,int capacity,int keys,const int* position_device,
    int query_offset) {
    EXL3_PDL_SMALL_PROLOGUE();
    constexpr int H=kFastFusedFlashHeads;
    const int block=static_cast<int>(blockIdx.x);
    const int head=block%H;
    const int query=block/(H*kKVHeads);
    const int kv_head=(block/H)%kKVHeads;
    const int tid=threadIdx.x;
    if(query>=rows || tid>=kHeadDim) return;
    const int base=(position_device?*position_device:position)+query_offset;
    const int count=base+query+1;
    std::uint16_t* out=output+(query*kQHeads+kv_head*H+head)*kHeadDim+tid;
    if(count<1 || count>capacity) { *out=0; return; }
    const int live_segments=min(segments,(count+keys-1)/keys);
    const float* slots=workspace+
        (static_cast<std::size_t>(query)*kKVHeads+kv_head)*segments*kFastFusedFlashStride;
    float global_max=-3.402823466e+38F;
    for(int segment=0;segment<live_segments;++segment)
        global_max=fmaxf(global_max,
            slots[segment*kFastFusedFlashStride+kFastFusedFlashValues+head]);
    float denominator=0.0f,numerator=0.0f;
    for(int segment=0;segment<live_segments;++segment) {
        const float* slot=slots+segment*kFastFusedFlashStride;
        const float scale=expf(slot[kFastFusedFlashValues+head]-global_max);
        denominator+=slot[kFastFusedFlashValues+H+head]*scale;
        numerator+=slot[head*kHeadDim+tid]*scale;
    }
    *out=__half_as_ushort(__float2half_rn(numerator/denominator));
}

bool fused_flash_merge_heads_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_FLASH_MERGE_HEADS");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

void launch_fused_flash_merge(const float* workspace,std::uint16_t* output,int rows,
    int segments,int position,int capacity,int keys,const int* position_device,
    cudaStream_t stream) {
    if(fused_flash_merge_heads_enabled())
        exl3_launch_small(attention_fused_flash_merge_heads_kernel,dim3(rows*kKVHeads*kFastFusedFlashHeads),dim3(256),0,stream,workspace,output,rows,
                segments,position,capacity,keys,position_device,0);
    else
        attention_cached_gqa_six_fused_flash_merge_kernel<<<
            rows*kKVHeads,256,0,stream>>>(workspace,output,rows,segments,position,
                capacity,keys,position_device,0);
}

template<int kVerifyMmaKeys>
__global__ void __launch_bounds__(96) attention_verify_flash_mma_kernel(
    const std::uint16_t* q,const std::uint16_t* k_cache,
    const std::uint16_t* v_cache,float* workspace,int rows,int position,
    int capacity,int segments,const int* position_device,int query_offset) {
    EXL3_PDL_SMALL_PROLOGUE();
    constexpr int H=kFastFusedFlashHeads;
    constexpr int kVectors=kHeadDim/8;
    __shared__ __align__(16) half k_s[kVerifyMmaChunk*kVerifyMmaStride];
    __shared__ __align__(16) half v_s[kVerifyMmaChunk*kVerifyMmaStride];
    const int segment=static_cast<int>(blockIdx.x);
    const int kv_head=static_cast<int>(blockIdx.y);
    const int tid=static_cast<int>(threadIdx.x);
    const int warp=tid>>5;
    const int lane=tid&31;
    const int g=lane>>2;
    const int t=lane&3;
    const int base=(position_device?*position_device:position)+query_offset;
    const int first=segment*kVerifyMmaKeys;
    const int last_count=base+rows;  // largest live key count of the block
    if(first>=last_count || base+1<1 || last_count>capacity) return;
    const int segment_end=min(first+kVerifyMmaKeys,last_count);

    // Fragment rows: pair = row*6 + head for this warp's m16 tile.
    const int pair0=warp*16+g;
    const int pair1=pair0+8;
    const int row0=pair0/H,head0=pair0%H;
    const int row1=pair1/H,head1=pair1%H;
    const bool live0=row0<rows,live1=row1<rows;
    const int count0=base+row0+1,count1=base+row1+1;
    const auto* q0=reinterpret_cast<const unsigned*>(q+
        (static_cast<std::size_t>(live0?row0:0)*kQHeads+kv_head*H+head0)*kHeadDim+2*t);
    const auto* q1=reinterpret_cast<const unsigned*>(q+
        (static_cast<std::size_t>(live1?row1:0)*kQHeads+kv_head*H+head1)*kHeadDim+2*t);

    float acc[kHeadDim/8][4];
    #pragma unroll
    for(int n=0;n<kHeadDim/8;++n) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0.0f;
    float running_max[2]={-INFINITY,-INFINITY};
    float denominator[2]={0.0f,0.0f};

    for(int chunk_first=first;chunk_first<segment_end;chunk_first+=kVerifyMmaChunk) {
        __syncthreads();
        for(int index=tid;index<kVerifyMmaChunk*kVectors;index+=blockDim.x) {
            const int key_offset=index/kVectors;
            const int dim=(index%kVectors)*8;
            const int key=chunk_first+key_offset;
            uint4 key_bits=make_uint4(0,0,0,0),value_bits=make_uint4(0,0,0,0);
            if(key<segment_end) {
                const std::size_t offset=(static_cast<std::size_t>(key)*kKVHeads+kv_head)*
                    kHeadDim+dim;
                key_bits=*reinterpret_cast<const uint4*>(k_cache+offset);
                value_bits=*reinterpret_cast<const uint4*>(v_cache+offset);
            }
            *reinterpret_cast<uint4*>(k_s+key_offset*kVerifyMmaStride+dim)=key_bits;
            *reinterpret_cast<uint4*>(v_s+key_offset*kVerifyMmaStride+dim)=value_bits;
        }
        __syncthreads();

        float s[4][4];
        #pragma unroll
        for(int j=0;j<4;++j) s[j][0]=s[j][1]=s[j][2]=s[j][3]=0.0f;
        #pragma unroll
        for(int kk=0;kk<kHeadDim;kk+=16) {
            const unsigned a[4]={live0?q0[kk/2]:0u,live1?q1[kk/2]:0u,
                                 live0?q0[kk/2+4]:0u,live1?q1[kk/2+4]:0u};
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
                unsigned b[4];
                reg_attn_ldmatrix_x4(b,k_s+(pair*16+(lane&7)+((lane>>4)<<3))*
                    kVerifyMmaStride+kk+((lane>>3)&1)*8);
                reg_attn_mma(s[2*pair],a,b[0],b[1]);
                reg_attn_mma(s[2*pair+1],a,b[2],b[3]);
            }
        }
        float tile_max[2]={-INFINITY,-INFINITY};
        #pragma unroll
        for(int j=0;j<4;++j)
            #pragma unroll
            for(int e=0;e<4;++e) {
                const int key=chunk_first+8*j+2*t+(e&1);
                const bool valid=e<2?(live0&&key<count0):(live1&&key<count1);
                s[j][e]=valid?s[j][e]*0.0625f:-INFINITY;
                tile_max[e>>1]=fmaxf(tile_max[e>>1],s[j][e]);
            }
        #pragma unroll
        for(int i=0;i<2;++i) {
            tile_max[i]=fmaxf(tile_max[i],__shfl_xor_sync(0xffffffffU,tile_max[i],1));
            tile_max[i]=fmaxf(tile_max[i],__shfl_xor_sync(0xffffffffU,tile_max[i],2));
        }
        float scale[2];
        #pragma unroll
        for(int i=0;i<2;++i) {
            const float next=fmaxf(running_max[i],tile_max[i]);
            scale[i]=next==-INFINITY?1.0f:expf(running_max[i]-next);
            running_max[i]=next;
        }
        half p[4][4];
        float sums[2]={0.0f,0.0f};
        #pragma unroll
        for(int j=0;j<4;++j)
            #pragma unroll
            for(int e=0;e<4;++e) {
                const float m=running_max[e>>1];
                p[j][e]=__float2half_rn(s[j][e]==-INFINITY?0.0f:expf(s[j][e]-m));
                sums[e>>1]+=__half2float(p[j][e]);
            }
        #pragma unroll
        for(int i=0;i<2;++i) {
            sums[i]+=__shfl_xor_sync(0xffffffffU,sums[i],1);
            sums[i]+=__shfl_xor_sync(0xffffffffU,sums[i],2);
            denominator[i]=denominator[i]*scale[i]+sums[i];
        }
        #pragma unroll
        for(int n=0;n<kHeadDim/8;++n) {
            acc[n][0]*=scale[0];acc[n][1]*=scale[0];
            acc[n][2]*=scale[1];acc[n][3]*=scale[1];
        }
        #pragma unroll
        for(int kk=0;kk<2;++kk) {
            const unsigned a[4]={reg_attn_pack(p[2*kk][0],p[2*kk][1]),
                                 reg_attn_pack(p[2*kk][2],p[2*kk][3]),
                                 reg_attn_pack(p[2*kk+1][0],p[2*kk+1][1]),
                                 reg_attn_pack(p[2*kk+1][2],p[2*kk+1][3])};
            #pragma unroll
            for(int pair=0;pair<kHeadDim/16;++pair) {
                unsigned b[4];
                reg_attn_ldmatrix_x4_trans(b,v_s+(16*kk+(lane&7)+((lane>>3)&1)*8)*
                    kVerifyMmaStride+pair*16+(lane>>4)*8);
                reg_attn_mma(acc[2*pair],a,b[0],b[1]);
                reg_attn_mma(acc[2*pair+1],a,b[2],b[3]);
            }
        }
    }

    #pragma unroll
    for(int i=0;i<2;++i) {
        const int row=i==0?row0:row1;
        const int head=i==0?head0:head1;
        const int count=i==0?count0:count1;
        if(!(i==0?live0:live1) || first>=count) continue;
        float* slot=workspace+
            ((static_cast<std::size_t>(row)*kKVHeads+kv_head)*segments+segment)*
                kFastFusedFlashStride;
        #pragma unroll
        for(int n=0;n<kHeadDim/8;++n) {
            const int dim=8*n+2*t;
            slot[head*kHeadDim+dim]=acc[n][2*i];
            slot[head*kHeadDim+dim+1]=acc[n][2*i+1];
        }
        if(t==0) {
            slot[kFastFusedFlashValues+head]=running_max[i];
            slot[kFastFusedFlashValues+H+head]=denominator[i];
        }
    }
}

bool verify_flash_mma_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_VERIFY_FLASH_MMA");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

// Key-segment length of the tensor-core flash path (64 default, 128 or 256;
// quality-gated: 2-row verify NLL within +0.0003 nats/token). Shorter
// segments raise the CTA count for decode-sized row counts; a length whose
// scratch would exceed the owned plane falls back to the next longer one.
int verify_flash_mma_keys() {
    static const int keys=[] {
        const char* value=std::getenv("NINFER_EXL3_VERIFY_FLASH_MMA_KEYS");
        if(!value) return 64;
        const int parsed=std::atoi(value);
        if(parsed!=64 && parsed!=128 && parsed!=256)
            throw std::invalid_argument("NINFER_EXL3_VERIFY_FLASH_MMA_KEYS must be 64, 128 or 256");
        return parsed;
    }();
    return keys;
}

// Tensor-core flash attention plus segment merge. Returns the key-segment
// length used.
int launch_verify_flash_mma(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,float* workspace,std::size_t workspace_bytes,
    std::uint16_t* output,int rows,int position,int capacity,
    const int* position_device,cudaStream_t stream,int requested_keys=0) {
    const int count=position+rows;
    int keys=requested_keys?requested_keys:verify_flash_mma_keys();
    const auto required=[&](int length) {
        return static_cast<std::size_t>(rows)*kKVHeads*((count+length-1)/length)*
            kFastFusedFlashStride*sizeof(float);
    };
    while(keys<256 && required(keys)>workspace_bytes) keys*=2;
    if(required(keys)>workspace_bytes)
        throw std::invalid_argument("verify flash MMA scratch extent");
    const int segments=(count+keys-1)/keys;
    const dim3 grid(segments,kKVHeads);
    if(keys==64)
        exl3_launch_small(attention_verify_flash_mma_kernel<64>,dim3(grid),dim3(96),0,stream,
            q,k,v,workspace,rows,position,capacity,segments,position_device,0);
    else if(keys==128)
        exl3_launch_small(attention_verify_flash_mma_kernel<128>,dim3(grid),dim3(96),0,stream,
            q,k,v,workspace,rows,position,capacity,segments,position_device,0);
    else
        exl3_launch_small(attention_verify_flash_mma_kernel<256>,dim3(grid),dim3(96),0,stream,
            q,k,v,workspace,rows,position,capacity,segments,position_device,0);
    launch_fused_flash_merge(workspace,output,rows,segments,position,capacity,keys,
        position_device,stream);
    return keys;
}

void fast_verify_flash_mma_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,float* workspace,
    std::uint16_t* output,int position,int capacity,int segments,int rows,
    cudaStream_t stream,int keys) {
    if(rows<1 || rows>8 || position<0 || position+rows>capacity)
        throw std::invalid_argument("verify flash MMA fixture rows/frontier");
    const std::size_t bytes=static_cast<std::size_t>(rows)*kKVHeads*
        ((position+rows+keys-1)/keys)*kFastFusedFlashStride*sizeof(float);
    (void)segments;
    if(launch_verify_flash_mma(q,k,v,workspace,bytes,output,rows,position,capacity,
           nullptr,stream,keys)!=keys)
        throw std::invalid_argument("verify flash MMA fixture segment length");
    cuda_check(cudaGetLastError(),"verify flash MMA fixture launch");
}

void fast_fused_attention_fixture(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,float* workspace,
    std::uint16_t* output,int position,int capacity,int segments,
    const int* position_device,cudaStream_t stream,int rows) {
    if(rows<1 || rows>8 || position<0 || position+rows>capacity)
        throw std::invalid_argument("fused attention fixture rows/frontier");
    attention_cached_gqa_six_fused_flash_kernel<kFastFusedFlashKeys256><<<
        rows*kKVHeads*segments,256,0,stream>>>(q,k,v,workspace,output,rows,position,
            capacity,segments,position_device,0);
    attention_cached_gqa_six_fused_flash_merge_kernel<<<
        rows*kKVHeads,256,0,stream>>>(workspace,output,rows,segments,position,
            capacity,kFastFusedFlashKeys256,position_device,0);
    cuda_check(cudaGetLastError(),"fused attention fixture launch");
}

std::uint64_t exl3_fast_whole_context_fused_attention_calls_for_test() noexcept {
    return fast_whole_context_fused_attention_counter.load(std::memory_order_relaxed);
}

std::uint64_t exl3_fast_prefill_tiled_attention_calls_for_test() noexcept {
    return fast_prefill_tiled_attention_counter.load(std::memory_order_relaxed);
}

std::uint64_t exl3_fast_prefill_rows4_attention_calls_for_test() noexcept {
    return fast_prefill_rows4_attention_counter.load(std::memory_order_relaxed);
}

std::uint64_t exl3_fast_prefill_rows8_attention_calls_for_test() noexcept {
    return fast_prefill_rows8_attention_counter.load(std::memory_order_relaxed);
}

std::uint64_t exl3_fast_prefill_wmma_attention_calls_for_test() noexcept {
    return fast_prefill_wmma_attention_counter.load(std::memory_order_relaxed);
}

std::uint64_t exl3_fast_prefill_wmma32_attention_calls_for_test() noexcept {
    return fast_prefill_wmma32_attention_counter.load(std::memory_order_relaxed);
}

std::uint64_t exl3_fast_prefill_rows2_attention_calls_for_test() noexcept {
    return fast_prefill_rows2_attention_counter.load(std::memory_order_relaxed);
}

std::uint64_t exl3_fast_online_decode_attention_calls_for_test() noexcept {
    return fast_online_decode_attention_counter.load(std::memory_order_relaxed);
}

std::uint64_t exl3_fast_cublas_attention_calls_for_test() noexcept {
    return fast_cublas_attention_counter.load(std::memory_order_relaxed);
}

std::size_t exl3_numeric_attention_splitk_workspace_bytes(int rows,int capacity) {
    if(rows<1||rows>1024||capacity<rows) return 0;
    const int segments=(capacity+kT71SegmentKeys-1)/kT71SegmentKeys;
    return static_cast<std::size_t>(rows)*kKVHeads*segments*kT71PartialStride*
        sizeof(float);
}

void exl3_numeric_attention_splitk_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* workspace,std::size_t workspace_bytes,int rows,int position,
    int capacity,cudaStream_t stream) {
    const std::size_t required=exl3_numeric_attention_splitk_workspace_bytes(rows,capacity);
    if(!q||!k||!v||!output||!workspace||required==0||workspace_bytes!=required||
       position<0||position>capacity-rows)
        throw std::invalid_argument("numeric split-K attention qualification extent");
    const int segments=(capacity+kT71SegmentKeys-1)/kT71SegmentKeys;
    attention_cached_gqa_six_splitk_partial_kernel<<<
        rows*kKVHeads*segments,256,0,stream>>>(q,k,v,workspace,rows,position,
            capacity,segments,nullptr,0);
    attention_cached_gqa_six_splitk_merge_kernel<<<rows*kKVHeads,256,0,stream>>>(
        workspace,output,rows,segments);
    cuda_check(cudaGetLastError(),"numeric split-K attention qualification launch");
}

std::size_t Exl3FullAttentionLayer::shared_scratch_bytes(int rows,bool coalesce_input_mlp) {
    if(rows<=0) throw std::invalid_argument("full-attention shared scratch rows");
    std::size_t per_row=0;
    for(std::size_t i=0;i<kBufferFeatures.size();++i)
        if(transient_buffer(i) && !(coalesce_input_mlp && i==11)) per_row+=kBufferFeatures[i]*sizeof(std::uint16_t);
    return static_cast<std::size_t>(rows)*per_row;
}

std::size_t Exl3FullAttentionLayer::fixed_owner_metadata_required() noexcept {
    return sizeof(Exl3FullAttentionLayer)+7*Exl3CudaLinearWorkspace::metadata_bytes()+18*Exl3LayerBufferRetirement::record_bytes();
}
std::size_t Exl3FullAttentionLayer::fixed_owner_metadata_bytes() const noexcept {
    auto bytes=sizeof(Exl3FullAttentionLayer);
    for(const auto& owner:buffer_retirements_)if(owner)bytes+=Exl3LayerBufferRetirement::record_bytes();
    for(const auto* workspace:linear_workspaces_)if(workspace)bytes+=Exl3CudaLinearWorkspace::metadata_bytes();
    return bytes;
}
std::size_t Exl3FullAttentionLayer::workspace_bytes_required(int rows,bool borrowed_accumulation,
    bool borrowed_transform,bool borrowed_scratch,bool coalesce_input_mlp) {
    if(rows<=0 || rows>1024)throw std::invalid_argument("EXL3 full-attention layer max_rows must be 1..1024");
    std::size_t bytes=0;
    const auto add=[&](std::size_t extent) {
        if(extent)bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(bytes,extent);
    };
    for(const auto shape:std::array<std::pair<int,int>,7>{{
        {kHidden,kQProjection},{kHidden,kKVProjection},{kHidden,kKVProjection},
        {kQHeads*kHeadDim,kHidden},{kHidden,kIntermediate},
        {kHidden,kIntermediate},{kIntermediate,kHidden}}})
        add(Exl3LinearWorkspaceRequirements::derive(shape.first,shape.second,rows,
            borrowed_transform,borrowed_accumulation).owned_bytes);
    for(std::size_t i=0;i<kBufferFeatures.size();++i) {
        if((coalesce_input_mlp && i==11) || (borrowed_scratch && transient_buffer(i)))continue;
        add(static_cast<std::size_t>(rows)*kBufferFeatures[i]*sizeof(std::uint16_t));
    }
    return bytes;
}

void Exl3FullAttentionLayer::ensure_fast_cublas_attention_resources(int capacity) {
    if (!fast_cublas_attention_) return;
    if (capacity < 1) throw std::invalid_argument(
        "FAST cuBLAS attention cache capacity");
    if (fast_cublas_handle_ != nullptr) {
        if (fast_cublas_capacity_ != capacity)
            throw std::invalid_argument(
                "FAST cuBLAS attention cache capacity changed");
        return;
    }
    cublasHandle_t handle = nullptr;
    try {
        cublas_check_attention(cublasCreate(&handle),
            "create FAST same-weight FP16-KV cuBLAS attention handle");
        cublas_check_attention(cublasSetPointerMode(handle,
            CUBLAS_POINTER_MODE_HOST),
            "set FAST same-weight FP16-KV cuBLAS attention pointer mode");
        cublas_check_attention(cublasSetMathMode(handle,
            CUBLAS_TENSOR_OP_MATH),
            "set FAST same-weight FP16-KV cuBLAS attention math mode");
        fast_cublas_handle_ = reinterpret_cast<void*>(handle);
        fast_cublas_capacity_ = capacity;
    } catch (...) {
        if (handle) (void)cublasDestroy(handle);
        throw;
    }
}

Exl3FullAttentionLayer::Exl3FullAttentionLayer(
    const Exl3FullAttentionLayerWeights& weights, int max_rows,
    Exl3CudaAccumulationView accumulation, Exl3CudaTransformView transformed,
    Exl3CudaLayerScratchView scratch,
    Exl3CudaReconstructGemmWorkspace* reconstruct_gemm,bool coalesce_input_mlp,
    Exl3VeriCacheServingCoordinator* constructor_authority,unsigned buffer_fault_for_test,
    Exl3CudaAccumulationView paired_up_accumulation,
    Exl3CudaTransformView paired_up_transformed)
    : weights_(weights), max_rows_(max_rows),
      reconstruct_gemm_(reconstruct_gemm) {
    coalesce_input_mlp_=coalesce_input_mlp;
    small_m_fused_gate_up_transform_ = read_binary_option("NINFER_EXL3_SMALL_M_FUSED_GATE_UP_TRANSFORM",
        "small-M fused gate/up transform must be 0 or 1");
    fast_fused_flash_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_FUSED_FLASH",
        "FAST same-weight FP16-KV fused flash must be 0 or 1");
    fast_fused_flash_attention_keys256_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_FUSED_FLASH_KEYS256",
        "FAST same-weight FP16-KV fused flash keys256 must be 0 or 1");
    fast_fused_flash_multirow_ = read_binary_option(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_FUSED_FLASH_MULTIROW",
        "FAST same-weight FP16-KV fused flash multirow must be 0 or 1");
    fast_cublas_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_CUBLAS_ATTENTION",
        "FAST same-weight FP16-KV cuBLAS attention must be 0 or 1");
    fast_online_decode_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_ONLINE_DECODE",
        "FAST same-weight FP16-KV online decode attention must be 0 or 1");
    fast_online_decode_attention_warp_heads_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_ONLINE_DECODE_WARP_HEADS",
        "FAST same-weight FP16-KV online decode warp-heads must be 0 or 1");
    fast_whole_context_fused_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_WHOLE_CONTEXT",
        "FAST same-weight FP16-KV whole-context fused attention must be 0 or 1");
    fast_prefill_tiled_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_TILED",
        "FAST same-weight FP16-KV tiled prefill attention must be 0 or 1");
    fast_prefill_rows4_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_ROWS4",
        "FAST same-weight FP16-KV four-row prefill attention must be 0 or 1");
    fast_prefill_rows8_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_ROWS8",
        "FAST same-weight FP16-KV eight-row prefill attention must be 0 or 1");
    fast_prefill_rows8_threads128_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_ROWS8_THREADS128",
        "FAST same-weight FP16-KV eight-row prefill 128-thread launch must be 0 or 1");
    fast_prefill_wmma_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA",
        "FAST same-weight FP16-KV WMMA prefill must be 0 or 1");
    const char* explicit_wmma32 = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA32");
    const char* fast_prefill_scope = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL");
    fast_prefill_wmma32_attention_ = read_binary_option(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA32",
        "FAST same-weight FP16-KV WMMA32 prefill must be 0 or 1") ||
        (!explicit_wmma32 && fast_prefill_scope &&
         std::strcmp(fast_prefill_scope, "1") == 0);
    fast_prefill_wmma64_attention_ = read_binary_option(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA64",
        "FAST same-weight FP16-KV WMMA64 prefill must be 0 or 1");
    if (fast_prefill_wmma64_attention_ && !fast_prefill_wmma32_attention_)
        throw std::invalid_argument("WMMA64 requires WMMA32 prefill policy");
    fast_prefill_wmma64_register_attention_ = read_binary_option(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA64_REGISTER",
        "FAST same-weight FP16-KV WMMA64 register prefill must be 0 or 1");
    if (fast_prefill_wmma64_register_attention_ &&
        !fast_prefill_wmma64_attention_)
        throw std::invalid_argument("WMMA64 register candidate requires WMMA64");
    fast_prefill_wmma64_register_keys64_attention_ = read_binary_option(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA64_REGISTER_KEYS64",
        "FAST same-weight FP16-KV WMMA64 register keys64 must be 0 or 1");
    if (fast_prefill_wmma64_register_keys64_attention_ &&
        !fast_prefill_wmma64_register_attention_)
        throw std::invalid_argument("WMMA64 keys64 requires register candidate");
    fast_prefill_wmma64_shared_heads_attention_ = read_binary_option(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA64_HEADS2",
        "FAST same-weight FP16-KV WMMA64 shared heads must be 0 or 1");
    if (fast_prefill_wmma64_shared_heads_attention_ &&
        !fast_prefill_wmma64_register_attention_)
        throw std::invalid_argument("WMMA64 shared heads requires register candidate");
    if (fast_prefill_wmma64_register_keys64_attention_ &&
        fast_prefill_wmma64_shared_heads_attention_)
        throw std::invalid_argument("WMMA64 keys64 excludes shared heads");
    fast_prefill_wmma32_padded_attention_ = read_binary_option(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA32_PADDED",
        "FAST same-weight FP16-KV WMMA32 padded tile must be 0 or 1");
    if (fast_prefill_wmma32_padded_attention_ &&
        !fast_prefill_wmma32_attention_)
        throw std::invalid_argument("padded WMMA32 requires WMMA32 prefill");
    fast_prefill_rows2_attention_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_ROWS2",
        "FAST same-weight FP16-KV two-row prefill attention must be 0 or 1");
    fast_same_weights_fp16kv_m1_gate_up_pair_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_GATE_UP_PAIR",
        "FAST same-weight FP16-KV M1 gate/up pair must be 0 or 1");
    const auto* m1_kv_pair=std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_KV_PAIR");
    if(m1_kv_pair && std::strcmp(m1_kv_pair,"0")!=0 &&
       std::strcmp(m1_kv_pair,"1")!=0)
        throw std::invalid_argument(
            "FAST same-weight FP16-KV M1 K/V pair must be 0 or 1");
    fast_same_weights_fp16kv_m1_kv_pair_=m1_kv_pair &&
        std::strcmp(m1_kv_pair,"1")==0;
    fast_same_weights_fp16kv_m1_kv_wide_pair_ = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_KV_WIDE_PAIR",
        "FAST same-weight FP16-KV M1 wide K/V pair must be 0 or 1");
    const bool m1_kv_pair_graph = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_KV_PAIR_GRAPH",
        "FAST same-weight FP16-KV M1 K/V graph pair must be 0 or 1");
    fast_same_weights_fp16kv_m1_kv_pair_graph_ = fast_same_weights_fp16kv_m1_kv_pair_ && m1_kv_pair_graph;
    if(fast_fused_flash_attention_ && fast_whole_context_fused_attention_)
        throw std::invalid_argument(
            "FAST segmented and whole-context fused attention are mutually exclusive");
    const auto* query_pair_scores=std::getenv(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_QUERY_PAIR_SCORES");
    if(query_pair_scores && std::strcmp(query_pair_scores,"0")!=0 &&
       std::strcmp(query_pair_scores,"1")!=0)
        throw std::invalid_argument(
            "exact attention GQA six query-pair scores must be 0 or 1");
    exact_attention_gqa_six_query_pair_scores_=query_pair_scores &&
        std::strcmp(query_pair_scores,"1")==0;
    const auto* score_k_tile64=std::getenv(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SCORE_K_TILE64");
    if(score_k_tile64 && std::strcmp(score_k_tile64,"0")!=0 &&
       std::strcmp(score_k_tile64,"1")!=0)
        throw std::invalid_argument(
            "exact attention GQA six score K-tile64 must be 0 or 1");
    exact_attention_gqa_six_score_k_tile64_=score_k_tile64 &&
        std::strcmp(score_k_tile64,"1")==0;
    if(exact_attention_gqa_six_score_k_tile64_ &&
       exact_attention_gqa_six_query_pair_scores_)
        throw std::invalid_argument(
            "exact attention K-tile64 and query-pair scores are mutually exclusive");
    exact_attention_gqa_six_softmax_triple_values_v_tile_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE",
        "exact attention GQA six-softmax triple-values V tile must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_v_tile64_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE64",
        "exact attention GQA six-softmax triple-values V tile64 must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_full_cta_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FULL_CTA",
        "exact attention GQA six-softmax triple-values full CTA must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_threads128_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_THREADS128",
        "exact attention GQA six-softmax triple-values threads128 must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_score_tile_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCORE_TILE",
        "exact attention GQA six-softmax triple-values score tile must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_key_pair_pipeline_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_KEY_PAIR_PIPELINE",
        "exact attention GQA triple-values key-pair pipeline must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_warp_score_broadcast_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_WARP_SCORE_BROADCAST",
        "exact attention GQA triple-values warp score broadcast must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_pair_dimensions_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS",
        "exact attention GQA six-softmax triple-values pair-dimensions must be 0 or 1");
    exact_attention_gqa_six_softmax_six_values_single_load_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD",
        "exact attention GQA six-softmax six-values single-load must be 0 or 1");
    exact_attention_gqa_six_softmax_six_values_scalar_single_load_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SCALAR_SINGLE_LOAD",
        "exact attention GQA six-softmax six-values scalar single-load must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_scalar_dim_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCALAR_DIM",
        "exact attention GQA six-softmax triple-values scalar-dim must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_two_query_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_TWO_QUERY",
        "exact attention GQA six-softmax triple-values two-query must be 0 or 1");
    exact_attention_gqa_six_softmax_triple_values_fused_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FUSED",
        "exact attention GQA six-softmax triple-values fused must be 0 or 1");
    exact_attention_gqa_six_softmax_fused_scalar_values_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_FUSED_SCALAR_VALUES",
        "exact attention GQA six-softmax fused scalar values must be 0 or 1");
    const auto* decode_fused=std::getenv(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_DECODE_FUSED");
    if(decode_fused && std::strcmp(decode_fused,"0")!=0 &&
       std::strcmp(decode_fused,"1")!=0)
        throw std::invalid_argument(
            "exact attention GQA six decode-fused route must be 0 or 1");
    exact_attention_gqa_six_decode_fused_=decode_fused &&
        std::strcmp(decode_fused,"1")==0;
    exact_attention_gqa_six_softmax_tile512_ = read_binary_option("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TILE512",
        "exact attention GQA six softmax tile512 must be 0 or 1");
    if(exact_attention_gqa_six_softmax_tile512_ &&
       (exact_attention_gqa_six_softmax_triple_values_fused_ ||
        exact_attention_gqa_six_softmax_fused_scalar_values_))
        throw std::invalid_argument(
            "exact attention softmax tile512 and fused value routes are mutually exclusive");
    const int triple_value_candidates=
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_v_tile_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_v_tile64_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_full_cta_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_threads128_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_score_tile_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_key_pair_pipeline_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_warp_score_broadcast_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_pair_dimensions_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_scalar_dim_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_two_query_)+
        static_cast<int>(exact_attention_gqa_six_softmax_six_values_single_load_)+
        static_cast<int>(exact_attention_gqa_six_softmax_six_values_scalar_single_load_)+
        static_cast<int>(exact_attention_gqa_six_softmax_triple_values_fused_)+
        static_cast<int>(exact_attention_gqa_six_softmax_fused_scalar_values_);
    if(triple_value_candidates>1)
        throw std::invalid_argument(
            "exact attention triple-values candidates are mutually exclusive");
    if (max_rows_ <= 0 || max_rows_ > 1024) {
        throw std::invalid_argument("EXL3 full-attention layer max_rows must be 1..1024");
    }
    if((!scratch.data && scratch.bytes) || (scratch.data && scratch.bytes<shared_scratch_bytes(max_rows_,coalesce_input_mlp_)))
        throw std::invalid_argument("full-attention shared scratch capacity");
    if((!paired_up_accumulation.data && paired_up_accumulation.bytes) ||
       (paired_up_accumulation.data &&
        (!accumulation.data || paired_up_accumulation.data==accumulation.data)))
        throw std::invalid_argument("full-attention paired up accumulation storage");
    if((!paired_up_transformed.data && paired_up_transformed.bytes) ||
       (paired_up_transformed.data &&
        (!transformed.data || paired_up_transformed.data==transformed.data)))
        throw std::invalid_argument("full-attention paired up transform storage");
    auto* cursor=static_cast<std::byte*>(scratch.data);
    const std::array<std::pair<int, int>, 7> shapes = {{
        {kHidden, kQProjection}, {kHidden, kKVProjection}, {kHidden, kKVProjection},
        {kQHeads * kHeadDim, kHidden}, {kHidden, kIntermediate},
        {kHidden, kIntermediate}, {kIntermediate, kHidden}}};
    std::size_t required_linear_bytes=0;
    for(const auto& shape:shapes) {
        const auto required=Exl3LinearWorkspaceRequirements::derive(shape.first,shape.second,max_rows_,
            transformed.data!=nullptr,accumulation.data!=nullptr);
        if(required.owned_bytes)required_linear_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            required_linear_bytes,required.owned_bytes);
    }
    std::array<std::size_t,kBufferFeatures.size()> buffer_bytes{};
    const auto required_workspace_bytes=workspace_bytes_required(max_rows_,accumulation.data!=nullptr,
        transformed.data!=nullptr,scratch.data!=nullptr,coalesce_input_mlp_);
    for(std::size_t i=0;i<buffer_bytes.size();++i) {
        if(coalesce_input_mlp_ && i==11)continue;
        buffer_bytes[i]=static_cast<std::size_t>(max_rows_)*kBufferFeatures[i]*sizeof(std::uint16_t);
    }
    try {
        for (std::size_t i = 0; i < linear_workspaces_.size(); ++i) {
            std::optional<RetainedDeviceLedger::Ticket> device;
            std::optional<RetainedDescriptorLedger::Ticket> metadata;
            const auto plan=Exl3LinearWorkspaceRequirements::derive(shapes[i].first,shapes[i].second,max_rows_,
                transformed.data!=nullptr,accumulation.data!=nullptr);
            if(constructor_authority && plan.owned_bytes) {
                auto credits=constructor_authority->reserve_constructor_credits(plan.owned_bytes,Exl3CudaLinearWorkspace::metadata_bytes());
                device.emplace(std::move(credits.device));metadata.emplace(std::move(credits.metadata));
            }
            const bool paired_workspace=(i==5 &&
                    fast_same_weights_fp16kv_m1_gate_up_pair_) ||
                (i==2 && (fast_same_weights_fp16kv_m1_kv_pair_ ||
                    fast_same_weights_fp16kv_m1_kv_wide_pair_));
            const auto workspace_accumulation=paired_workspace &&
                    paired_up_accumulation.data?
                paired_up_accumulation:accumulation;
            const auto workspace_transformed=paired_workspace &&
                    paired_up_transformed.data?
                paired_up_transformed:transformed;
            linear_workspaces_[i] = new Exl3CudaLinearWorkspace(
                shapes[i].first, shapes[i].second, max_rows_, false,
                i == 4 || i == 5, i == 6, i == 3, false, true, false,
                workspace_accumulation, workspace_transformed,
                false, i == 1 || i == 2, i == 0,std::move(device),std::move(metadata));
            workspace_bytes_ += linear_workspaces_[i]->workspace_bytes();
        }
        if(workspace_bytes_!=required_linear_bytes)throw std::runtime_error("full-attention linear allocation requirement mismatch");
        for(auto& owner:buffer_retirements_)owner.emplace(+[](void* pointer,int device) noexcept -> int {
            int current=-1;const auto query=cudaGetDevice(&current);
            if(query!=cudaSuccess)return static_cast<int>(query);
            if(current!=device)return static_cast<int>(cudaErrorInvalidDevice);
            return static_cast<int>(cudaFree(pointer));
        });
        int buffer_device=-1;
        cuda_check(cudaGetDevice(&buffer_device),"capture full-attention buffer device");
        for (std::size_t i=0;i<kBufferFeatures.size();++i) {
            if(coalesce_input_mlp_ && i==11) {buffers_[i]=buffers_[0];continue;}
            const auto bytes=buffer_bytes[i];
            if(scratch.data && transient_buffer(i)) {
                buffers_[i]=reinterpret_cast<std::uint16_t*>(cursor);cursor+=bytes;
            } else {
                std::optional<Exl3VeriCacheServingCoordinator::ConstructorCredits> credits;
                if(constructor_authority)credits.emplace(constructor_authority->reserve_constructor_credits(
                    bytes,Exl3LayerBufferRetirement::record_bytes()));
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&buffers_[i]),bytes),
                           "allocate EXL3 full-attention workspace");
                buffer_retirements_[i]->adopt(buffers_[i],bytes,buffer_device);
                if(credits)buffer_retirements_[i]->attach(std::move(credits->device),std::move(credits->metadata));
                if(buffer_fault_for_test==1 || buffer_fault_for_test==3) {
                    if(buffer_fault_for_test==1)buffer_retirements_[i]->fail_cleanup_for_test(static_cast<int>(cudaErrorUnknown));
                    throw std::runtime_error("injected layer buffer postallocation failure");
                }
                buffer_owned_[i]=true;workspace_bytes_+=bytes;
            }
        }
        if(workspace_bytes_!=required_workspace_bytes)throw std::runtime_error("full-attention complete allocation requirement mismatch");
        // E4C1: raise the cached-decode dynamic-shared ceiling as far as the
        // device allows (32K needs 128 KiB). Never throws: long-context eager
        // runs fail later with a clear launch error if the ceiling is short.
        int optin_max = 0;
        if (cudaDeviceGetAttribute(&optin_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)
            == cudaSuccess && optin_max > 0) {
            cudaFuncSetAttribute(reinterpret_cast<const void*>(attention_cached_kernel),
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   std::min(optin_max, 132 * 1024));
        }
    } catch (...) {
        for(auto& owner:buffer_retirements_)if(owner)owner->retire();
        for (auto*& workspace : linear_workspaces_) Exl3CudaLinearWorkspace::retire_slot(workspace);
        throw;
    }
}

Exl3FullAttentionLayer::~Exl3FullAttentionLayer() {
    if (fast_cublas_handle_) {
        (void)cublasDestroy(reinterpret_cast<cublasHandle_t>(fast_cublas_handle_));
        fast_cublas_handle_ = nullptr;
    }
    for(auto& owner:buffer_retirements_)if(owner)owner->retire();
    for (auto*& workspace : linear_workspaces_) Exl3CudaLinearWorkspace::retire_slot(workspace);
}

void Exl3FullAttentionLayer::prepare_eager_mlp_gateup_concurrency(
    Exl3MlpGateUpConcurrencyView view) {
    const bool any = view.up_stream || view.fork || view.up_done ||
        view.up_workspace;
    if (!any) return;
    if (!view.complete() || eager_mlp_gateup_concurrency_.complete() ||
        max_rows_ < 8)
        throw std::invalid_argument(
            "eager full-attention MLP gate/up concurrency resources");
    for (int rows = 1; rows <= 8; ++rows) {
        const auto admission = rows == 1
            ? Exl3CudaLinearAdmission::ordinary
            : Exl3CudaLinearAdmission::target_continuation_gate_up;
        const auto original_route = linear_workspaces_[5]->dispatch_name(
            weights_.up_metadata, rows, admission);
        const auto private_route = view.up_workspace->dispatch_name(
            weights_.up_metadata, rows, admission);
        if (std::strcmp(original_route, private_route) != 0)
            throw std::invalid_argument(std::string(
                "eager full-attention private up route mismatch rows=")+
                std::to_string(rows)+" original="+original_route+
                " private="+private_route);
    }
    eager_mlp_gateup_concurrency_ = view;
}

void Exl3FullAttentionLayer::prepare_prefill_qkv_concurrency(
    Exl3PrefillQkvConcurrencyView view) {
    const bool any=view.k_stream || view.v_stream || view.fork ||
        view.k_done || view.v_done || view.k_workspace || view.v_workspace;
    if(!any)return;
    if(!view.complete() || prefill_qkv_concurrency_.complete() || max_rows_<17)
        throw std::invalid_argument("wide-prefill QKV concurrency resources");
    constexpr std::array<int,7> rows_to_check{{17,32,64,128,256,512,1024}};
    for(const int rows:rows_to_check) {
        if(rows>max_rows_)continue;
        const auto admission=Exl3CudaLinearAdmission::target_wide_prefill;
        if(std::strcmp(linear_workspaces_[1]->dispatch_name(weights_.k_metadata,rows,admission),
                       view.k_workspace->dispatch_name(weights_.k_metadata,rows,admission))!=0 ||
           std::strcmp(linear_workspaces_[2]->dispatch_name(weights_.v_metadata,rows,admission),
                       view.v_workspace->dispatch_name(weights_.v_metadata,rows,admission))!=0)
            throw std::invalid_argument("wide-prefill private K/V route mismatch");
    }
    prefill_qkv_concurrency_=view;
}

void Exl3FullAttentionLayer::set_kv_cache(std::uint16_t* k_cache,
                                          std::uint16_t* v_cache,
                                          int capacity) noexcept {
    k_cache_ = k_cache;
    v_cache_ = v_cache;
    cache_capacity_ = capacity;
}

void Exl3FullAttentionLayer::set_oscar(Exl3OscarContext* oscar, int model_layer) noexcept {
    oscar_ = oscar;
    oscar_layer_ = model_layer;
}

void Exl3FullAttentionLayer::set_position_device(const int* position_device) noexcept {
    position_device_ = position_device;
}

void Exl3FullAttentionLayer::validate_retained_prefix_reappend(
    int retained_rows, int attempted_rows, int base_position,
    cudaStream_t stream) const {
    if (!retained_prefix_available_ || retained_rows <= 0 ||
        retained_rows > attempted_rows || attempted_rows != retained_prefix_rows_) {
        throw std::invalid_argument(
            "EXL3 full-attention retained prefix is absent or row count is invalid");
    }
    if (base_position != retained_prefix_position_) {
        throw std::invalid_argument(
            "EXL3 full-attention retained prefix position mismatch");
    }
    if (stream != retained_prefix_stream_) {
        throw std::invalid_argument(
            "EXL3 full-attention retained prefix stream mismatch");
    }
    if ((!oscar_ && (!k_cache_ || !v_cache_)) ||
        (oscar_ && oscar_->graph_class() != 0)) {
        throw std::invalid_argument(
            "EXL3 full-attention retained prefix requires eager OSCAR or ordinary K/V cache");
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status),
               "query EXL3 full-attention retained-prefix reappend capture");
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::invalid_argument(
            "EXL3 full-attention retained prefix requires an eager stream");
    }
}

void Exl3FullAttentionLayer::reappend_retained_prefix(
    int retained_rows, int attempted_rows, int base_position,
    cudaStream_t stream) {
    validate_retained_prefix_reappend(
        retained_rows, attempted_rows, base_position, stream);
    retained_prefix_available_ = false;
    if (oscar_) {
        const auto* qr = buffers_[6];
        const auto* kr = buffers_[7];
        const auto* vp = buffers_[3];
        for (int row = 0; row < retained_rows; ++row) {
            oscar_->append_layer(
                oscar_layer_,
                qr + static_cast<std::size_t>(row) * kQHeads * kHeadDim,
                kr + static_cast<std::size_t>(row) * kKVHeads * kHeadDim,
                vp + static_cast<std::size_t>(row) * kKVHeads * kHeadDim,
                1, static_cast<std::uint32_t>(base_position + row), stream);
        }
    }
    // Ordinary exact K/V already contains every attempted row. Shortening the
    // authoritative cursor retains the prefix; the correction overwrites the
    // first discarded row and later rows remain unreachable.
}

void Exl3FullAttentionLayer::arm_captured_retained_prefix(
    int attempted_rows,int base_position,cudaStream_t stream) {
    if ((!oscar_ && (!k_cache_ || !v_cache_)) ||
        attempted_rows < 2 || attempted_rows > 8 ||
        base_position < 0 || capture_active_)
        throw std::invalid_argument("captured full-attention retained prefix provenance");
    retained_prefix_available_ = true;
    retained_prefix_rows_ = attempted_rows;
    retained_prefix_position_ = base_position;
    retained_prefix_stream_ = stream;
    oscar_count_ += static_cast<std::uint64_t>(attempted_rows);
}

void Exl3FullAttentionLayer::complete_prefill_projection_chain_graph_after_drain(
    cudaStream_t stream) {
    const auto snapshot=prefill_projection_chain_graph_.snapshot();
    if(snapshot.pending &&
       !prefill_projection_chain_graph_.complete_after_drain(stream))
        throw std::logic_error("full-attention prefill projection-chain completion");
    const auto attention_snapshot=prefill_attention_chain_graph_.snapshot();
    if(attention_snapshot.pending_slots &&
       !prefill_attention_chain_graph_.complete_after_drain(stream))
        throw std::logic_error("full-attention prefill attention-chain completion");
}

void Exl3FullAttentionLayer::capture_mlp_tail_graph(
    std::uint16_t* output,int rows,cudaStream_t stream) {
    if(!capture_active_ || !output || rows<1 || rows>8 || !stream || oscar_ ||
       projection_timing_ || projection_observer_ ||
       eager_mlp_gateup_concurrency_.complete() ||
       small_m_fused_gate_up_transform_ || target_q_executor_)
        throw std::invalid_argument(
            "HostKV MLP-tail graph requires fixed ordinary continuation resources");
    auto* post_resid=buffers_[10];
    auto* mlp_in=buffers_[11];
    auto* gp=buffers_[12];
    auto* up=buffers_[13];
    auto* act=buffers_[14];
    auto* down=buffers_[15];
    exl3_launch_small(rms_norm_kernel,dim3(rows),dim3(512),512*sizeof(float),stream,post_resid,
        weights_.post_attention_norm,mlp_in,rows,kHidden);
    cuda_check(cudaGetLastError(),"capture HostKV MLP-tail norm");
    const auto project=[&](Exl3CudaLinearWorkspace* workspace,
                           const Exl3CudaLinearWeights& weights,
                           const Exl3CudaLinearMetadata& metadata,
                           const std::uint16_t* source,
                           std::uint16_t* destination,
                           bool small_m,
                           Exl3CudaLinearAdmission admission) {
        if(rows>1 && !small_m) {
            for(int row=0;row<rows;++row)
                workspace->forward(weights,metadata,
                    source+static_cast<std::size_t>(row)*metadata.in_features,
                    destination+static_cast<std::size_t>(row)*metadata.out_features,
                    1,stream,Exl3CudaLinearAdmission::ordinary);
        } else workspace->forward(weights,metadata,source,destination,rows,
            stream,rows==1?Exl3CudaLinearAdmission::ordinary:admission);
    };
    const bool gate_small_m=rows>1 &&
        (linear_workspaces_[4]->target_gateup_small_m_candidate(
             weights_.gate_metadata,rows,
             Exl3CudaLinearAdmission::target_continuation_gate_up) ||
         linear_workspaces_[4]->target_gateup_k5_small_m_candidate(
             weights_.gate_metadata,rows,
             Exl3CudaLinearAdmission::target_continuation_gate_up));
    const bool up_small_m=rows>1 &&
        (linear_workspaces_[5]->target_gateup_small_m_candidate(
             weights_.up_metadata,rows,
             Exl3CudaLinearAdmission::target_continuation_gate_up) ||
         linear_workspaces_[5]->target_gateup_k5_small_m_candidate(
             weights_.up_metadata,rows,
             Exl3CudaLinearAdmission::target_continuation_gate_up));
    // The ordinary device-KV graph is a separately labeled opt-in lane. When
    // its existing same-weight M1 gate/up pair is enabled, capture the native
    // two-output packed-MMA kernel as one graph node sequence. The pair keeps
    // the two native trellises, SUH/SVH transforms, FP32 partials, and output
    // reductions independent; it only shares the launch/activation staging.
    // The frozen/reference and default eager routes never enter this capture.
    const bool graph_gate_up_pair = rows == 1 &&
        fast_same_weights_fp16kv_m1_gate_up_pair_ &&
        (weights_.gate_metadata.K == 5 || weights_.gate_metadata.K == 6 ||
         weights_.gate_metadata.K == 7) &&
        weights_.up_metadata.K == weights_.gate_metadata.K &&
        !weights_.gate_metadata.mcg && !weights_.up_metadata.mcg &&
        weights_.gate_metadata.mul1 && weights_.up_metadata.mul1 &&
        !weights_.gate_metadata.has_bias && !weights_.up_metadata.has_bias;
    const auto* graph_diag = std::getenv("NINFER_EXL3_MLP_GRAPH_DIAGNOSTIC");
    if (graph_diag && std::strcmp(graph_diag, "1") == 0) {
        std::fprintf(stdout,
            "MLP_GRAPH_DIAG layer=%d rows=%d flag=%d gateK=%d upK=%d gateM=%d upM=%d gateMul1=%d upMul1=%d gateBias=%d upBias=%d pair=%d\n",
            model_layer_, rows, fast_same_weights_fp16kv_m1_gate_up_pair_ ? 1 : 0,
            weights_.gate_metadata.K, weights_.up_metadata.K,
            weights_.gate_metadata.mcg ? 1 : 0, weights_.up_metadata.mcg ? 1 : 0,
            weights_.gate_metadata.mul1 ? 1 : 0, weights_.up_metadata.mul1 ? 1 : 0,
            weights_.gate_metadata.has_bias ? 1 : 0,
            weights_.up_metadata.has_bias ? 1 : 0, graph_gate_up_pair ? 1 : 0);
        std::fflush(stdout);
    }
    if (graph_gate_up_pair) {
        linear_workspaces_[4]->forward_target_m1_gate_up_pair_for_test(
            *linear_workspaces_[5],weights_.gate,weights_.gate_metadata,
            weights_.up,weights_.up_metadata,mlp_in,gp,up,stream);
    } else {
        project(linear_workspaces_[4],weights_.gate,weights_.gate_metadata,
            mlp_in,gp,gate_small_m,
            Exl3CudaLinearAdmission::target_continuation_gate_up);
        cuda_check(cudaGetLastError(),"capture HostKV MLP-tail gate");
        project(linear_workspaces_[5],weights_.up,weights_.up_metadata,
            mlp_in,up,up_small_m,
            Exl3CudaLinearAdmission::target_continuation_gate_up);
        cuda_check(cudaGetLastError(),"capture HostKV MLP-tail up");
    }
    exl3_launch_small(silu_mul_kernel,dim3((rows*kIntermediate+255)/256),dim3(256),0,stream,
        gp,up,act,rows*kIntermediate);
    cuda_check(cudaGetLastError(),"capture HostKV MLP-tail SiLU");
    const bool down_small_m=rows>1 &&
        linear_workspaces_[6]->target_down_small_m_candidate(
            weights_.down_metadata,rows,
            Exl3CudaLinearAdmission::target_continuation_down);
    project(linear_workspaces_[6],weights_.down,weights_.down_metadata,
        act,down,down_small_m,
        Exl3CudaLinearAdmission::target_continuation_down);
    cuda_check(cudaGetLastError(),"capture HostKV MLP-tail down");
    exl3_launch_small(residual_kernel,dim3((rows*kHidden+255)/256),dim3(256),0,stream,
        post_resid,down,output,rows*kHidden);
    cuda_check(cudaGetLastError(),"capture HostKV MLP-tail residual");
}

void Exl3FullAttentionLayer::forward(const std::uint16_t* input,
                                     std::uint16_t* output,
                                     int rows,
                                     int position,
                                     cudaStream_t stream,
                                     bool profile,
                                     bool preserve_m1_topology,
                                     bool wide_prefill,
                                     DecodeGraphExecutable* mlp_tail_graph) {
    if(coalesce_input_mlp_ && (profile || capture_active_ || oscar_))
        throw std::invalid_argument("coalesced attention scratch requires ordinary eager nondiagnostic execution");
    if (!input || !output || rows <= 0 || rows > max_rows_) {
        throw std::invalid_argument("invalid EXL3 full-attention layer input/output/rows");
    }
    if(exact_prefix_rows_ || exact_page_ranges_.count) {
        exact_position_contract_.require_current(mrope_positions_,rope_offset_);
        if(!supports_segmented_exact_prefix())
            throw std::invalid_argument("segmented attention precompute extent/profile");
        Exl3AttentionInputView input_view{k_cache_,v_cache_,cache_capacity_,1024,exact_page_ranges_};
        if(exact_prefix_rows_)input_view.shared.append(exact_prefix_k_,exact_prefix_v_,
            exact_prefix_first_,exact_prefix_rows_,position);
        input_view.require_geometry(position,rows);
    }
    if(direct_staged_rows_) {
        if(!direct_staged_k_ || !direct_staged_v_ ||
           direct_staged_rows_!=position || !supports_direct_staged_history(rows,position))
            throw std::invalid_argument("direct staged history extent/profile");
    }
    retained_prefix_available_ = false;
    bool eligible_retained_prefix = preserve_m1_topology && rows >= 2 && rows <= 8 &&
        ((oscar_ != nullptr && oscar_->graph_class() == 0) ||
         (oscar_ == nullptr && k_cache_ != nullptr && v_cache_ != nullptr)) &&
        !capture_active_;
    if (eligible_retained_prefix) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture_status),
                   "query EXL3 full-attention retained-prefix source capture");
        eligible_retained_prefix =
            capture_status == cudaStreamCaptureStatusNone;
    }
    cudaEvent_t starts[13]{};
    cudaEvent_t ends[13]{};
    if (profile) {
        for (int i = 0; i < 13; ++i) {
            cuda_check(cudaEventCreate(&starts[i]), "create EXL3 layer start event");
            cuda_check(cudaEventCreate(&ends[i]), "create EXL3 layer end event");
        }
    }
    const auto begin = [&](int i) { if (profile) launch_profile_event(starts[i], stream); };
    const auto end = [&](int i) { if (profile) launch_profile_event(ends[i], stream); };
    const auto launch = [&](cudaError_t error, const char* op) { cuda_check(error, op); };
    auto* input_norm = buffers_[0];
    auto* qg = buffers_[1];
    auto* kp = buffers_[2];
    auto* vp = buffers_[3];
    auto* qn = buffers_[4];
    auto* kn = buffers_[5];
    auto* qr = buffers_[6];
    auto* kr = buffers_[7];
    auto* attn = buffers_[8];
    auto* op = buffers_[9];
    auto* post_resid = buffers_[10];
    auto* mlp_in = buffers_[11];
    auto* gp = buffers_[12];
    auto* up = buffers_[13];
    auto* act = buffers_[14];
    auto* down = buffers_[15];
    const auto project_on = [&](Exl3CudaLinearWorkspace* workspace,
                             const Exl3CudaLinearWeights& weights,
                             const Exl3CudaLinearMetadata& metadata,
                              const std::uint16_t* source,
                              std::uint16_t* destination,
                              Exl3TargetProjectionOperator operation,
                              cudaStream_t projection_stream,
                              const std::uint16_t* transformed_input=nullptr) {
        const bool rowwise = preserve_m1_topology && rows > 1;
        const bool target_initial16 = !rowwise && rows == 16 &&
            workspace->target_initial16_candidate(metadata, rows, Exl3CudaLinearAdmission::target_initial16);
        const bool target_wide_candidate = rowwise && wide_prefill &&
            workspace->target_wide_prefill_candidate(metadata, rows, Exl3CudaLinearAdmission::target_wide_prefill);
        const bool target_gateup_m16 = !rowwise && rows == 16 &&
            (operation == Exl3TargetProjectionOperator::gate ||
             operation == Exl3TargetProjectionOperator::up) &&
            workspace->target_gateup_m16_candidate(metadata, rows,
                Exl3CudaLinearAdmission::target_prefill_gate_up);
        const bool target_gateup_candidate = rowwise &&
            (operation == Exl3TargetProjectionOperator::gate ||
             operation == Exl3TargetProjectionOperator::up) &&
            (workspace->target_gateup_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_gate_up) ||
             workspace->target_gateup_k5_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_gate_up));
        const bool target_down_candidate = rowwise &&
            operation == Exl3TargetProjectionOperator::down &&
            workspace->target_down_small_m_candidate(
                metadata, rows,
                Exl3CudaLinearAdmission::target_continuation_down);
        const bool target_o_k7_candidate = rowwise &&
            operation == Exl3TargetProjectionOperator::o &&
            (workspace->target_o_k7_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_o) ||
             workspace->target_o_k6_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_o));
        const bool target_kv_candidate = rowwise && !wide_prefill &&
            (operation == Exl3TargetProjectionOperator::k ||
             operation == Exl3TargetProjectionOperator::v) &&
            workspace->target_kv_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_kv);
        const bool target_q_k6_candidate = rowwise && !wide_prefill &&
            operation == Exl3TargetProjectionOperator::q &&
            workspace->target_q_k6_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_q);
        const auto target_k5_admission =
            operation == Exl3TargetProjectionOperator::q
                ? Exl3CudaLinearAdmission::target_continuation_q
            : (operation == Exl3TargetProjectionOperator::k ||
               operation == Exl3TargetProjectionOperator::v)
                ? Exl3CudaLinearAdmission::target_continuation_kv
            : operation == Exl3TargetProjectionOperator::o
                ? Exl3CudaLinearAdmission::target_continuation_o
            : operation == Exl3TargetProjectionOperator::down
                ? Exl3CudaLinearAdmission::target_continuation_down
                : Exl3CudaLinearAdmission::ordinary;
        const bool target_k5_candidate = rowwise && !wide_prefill &&
            workspace->target_k5_small_m_batch_candidate(
                metadata, rows, target_k5_admission);
        const bool target_small_m_candidate =
            target_gateup_candidate || target_down_candidate ||
            target_o_k7_candidate || target_wide_candidate || target_kv_candidate ||
            target_q_k6_candidate || target_k5_candidate;
        const char* fast_same_weights_fp16kv_decode = std::getenv(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_DECODE");
        const bool fast_same_weights_fp16kv_decode_candidate =
            reconstruct_gemm_ && rows == 1 && !wide_prefill && !capture_active_ &&
            fast_same_weights_fp16kv_decode &&
            std::strcmp(fast_same_weights_fp16kv_decode, "1") == 0 &&
            reconstruct_gemm_->accepts_all_model_shapes() &&
            reconstruct_gemm_->supports(metadata, rows) && metadata.mul1 &&
            !metadata.mcg && !metadata.has_bias;
        const bool numeric_reconstruct_candidate =
            fast_same_weights_fp16kv_decode_candidate ||
            (reconstruct_gemm_ && rowwise && wide_prefill && !capture_active_ &&
             target_wide_candidate && reconstruct_gemm_->supports(metadata, rows) &&
             metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
             ((reconstruct_gemm_->accepts_all_model_shapes() &&
               metadata.K >= 5 && metadata.K <= 8 && rows >= 256) ||
              ((operation == Exl3TargetProjectionOperator::gate ||
                operation == Exl3TargetProjectionOperator::up) &&
               metadata.K == 6 && metadata.in_features == kHidden &&
               metadata.out_features == kIntermediate && rows >= 512) ||
              (operation == Exl3TargetProjectionOperator::down &&
               metadata.K == 7 && metadata.in_features == kIntermediate &&
               metadata.out_features == kHidden && rows >= 256)));
        bool observed_projection = false;
        const bool observe_prefill = !rowwise && rows == 16 &&
            projection_observer_selection_ ==
                Exl3TargetProjectionObserverSelection::prefill_gate_up_k6;
        const bool observe_initial16 = !rowwise && rows == 16 &&
            projection_observer_selection_ == Exl3TargetProjectionObserverSelection::initial16_generic;
        const bool observe_m1_k6_n32 = rowwise &&
            projection_observer_selection_ ==
                Exl3TargetProjectionObserverSelection::m1_k6_n32;
        if (projection_observer_ != nullptr &&
            (rowwise || observe_prefill || observe_initial16 || observe_m1_k6_n32)) {
            const bool observed_gate_up =
                (projection_observer_selection_ ==
                    Exl3TargetProjectionObserverSelection::gate_up_k6 || observe_prefill) &&
                (operation == Exl3TargetProjectionOperator::gate ||
                 operation == Exl3TargetProjectionOperator::up) &&
                metadata.in_features == kHidden &&
                metadata.out_features == kIntermediate;
            const bool observed_gate_up_k7 =
                projection_observer_selection_ ==
                    Exl3TargetProjectionObserverSelection::gate_up_k7 &&
                (operation == Exl3TargetProjectionOperator::gate ||
                 operation == Exl3TargetProjectionOperator::up) &&
                metadata.in_features == kHidden &&
                metadata.out_features == kIntermediate;
            const bool observed_down =
                projection_observer_selection_ ==
                    Exl3TargetProjectionObserverSelection::down_k6 &&
                operation == Exl3TargetProjectionOperator::down &&
                metadata.in_features == kIntermediate &&
                metadata.out_features == kHidden;
            const bool observed_down_k7 =
                projection_observer_selection_ ==
                    Exl3TargetProjectionObserverSelection::down_k7 &&
                operation == Exl3TargetProjectionOperator::down &&
                metadata.in_features == kIntermediate &&
                metadata.out_features == kHidden;
            const bool observed_output =
                projection_observer_selection_ ==
                    Exl3TargetProjectionObserverSelection::output_k7 &&
                operation == Exl3TargetProjectionOperator::o &&
                metadata.in_features == kQHeads * kHeadDim &&
                metadata.out_features == kHidden;
            observed_projection =
                (((observed_gate_up || observed_down) && metadata.K == 6) ||
                 ((observed_gate_up_k7 || observed_output || observed_down_k7) &&
                  metadata.K == 7)) &&
                metadata.mul1 && !metadata.mcg && !metadata.has_bias;
            if (observe_initial16)
                observed_projection = std::string(workspace->dispatch_name(metadata,16)) == "generic_tile" &&
                    !(metadata.in_features == 5120 && metadata.out_features == 17408 && metadata.K == 6);
            if (observe_m1_k6_n32)
                observed_projection = metadata.K == 6 && metadata.mul1 &&
                    !metadata.mcg && !metadata.has_bias &&
                    std::string(workspace->dispatch_name(metadata, 1)) ==
                        "generic_mma_split";
            if (projection_observer_selection_ == Exl3TargetProjectionObserverSelection::wide_prefill_all)
                observed_projection = rowwise;
        }
        if (observed_projection) {
            const Exl3TargetProjectionObservation observation{
                weights, metadata, source, rows, model_layer_,
                (projection_observer_selection_ == Exl3TargetProjectionObserverSelection::wide_prefill_all ||
                 observe_initial16 || observe_m1_k6_n32)
                    ? target_projection_operator_name(operation) :
                operation == Exl3TargetProjectionOperator::gate ? "gate" :
                    (operation == Exl3TargetProjectionOperator::up ? "up" :
                     (operation == Exl3TargetProjectionOperator::o ? "o" : "down")),
                workspace->dispatch_name(metadata, 1), projection_stream};
            projection_observer_(observation, projection_observer_user_);
        }
        const auto topology = numeric_reconstruct_candidate
            ? Exl3TargetProjectionTopology::batched
            : target_small_m_candidate
            ? Exl3TargetProjectionTopology::small_m_mma_split
            : (rowwise ? Exl3TargetProjectionTopology::m1_per_row
            : (rows == 1 ? Exl3TargetProjectionTopology::m1
                         : Exl3TargetProjectionTopology::batched));
        const int timing_slot = projection_timing_
            ? projection_timing_->begin(model_layer_, operation, rows, metadata.K,
                                        metadata.in_features, metadata.out_features,
                                        topology,
                                        rowwise && !target_small_m_candidate ? rows : 1,
                                        projection_stream)
            : -1;
        if (numeric_reconstruct_candidate) {
            if(transformed_input)
                throw std::logic_error(
                    "numeric reconstruct cannot consume borrowed transformed input");
            reconstruct_gemm_->forward_numeric_candidate(
                weights, metadata, source, destination, rows, projection_stream);
        } else if (rowwise && !target_small_m_candidate) {
            for (int row = 0; row < rows; ++row) {
                if(transformed_input)
                    workspace->forward_from_transformed(weights,metadata,
                        transformed_input+static_cast<std::size_t>(row)*metadata.in_features,
                        destination+static_cast<std::size_t>(row)*metadata.out_features,
                        1,projection_stream,Exl3CudaLinearAdmission::ordinary);
                else workspace->forward(weights, metadata,
                        source + static_cast<std::size_t>(row) * metadata.in_features,
                        destination + static_cast<std::size_t>(row) * metadata.out_features,
                        1, projection_stream);
            }
        } else {
            const auto admission=target_kv_candidate
                    ? Exl3CudaLinearAdmission::target_continuation_kv
                    : target_q_k6_candidate
                    ? Exl3CudaLinearAdmission::target_continuation_q
                    : target_k5_candidate
                    ? target_k5_admission
                    : target_initial16
                    ? Exl3CudaLinearAdmission::target_initial16
                    : target_wide_candidate
                    ? Exl3CudaLinearAdmission::target_wide_prefill
                    : target_gateup_m16
                    ? Exl3CudaLinearAdmission::target_prefill_gate_up
                    : target_gateup_candidate
                    ? Exl3CudaLinearAdmission::target_continuation_gate_up
                    : (target_down_candidate
                           ? Exl3CudaLinearAdmission::target_continuation_down
                           : (target_o_k7_candidate
                                   ? Exl3CudaLinearAdmission::target_continuation_o
                                   : Exl3CudaLinearAdmission::ordinary));
            if(transformed_input)
                workspace->forward_from_transformed(
                    weights,metadata,transformed_input,destination,rows,
                    projection_stream,admission);
            else workspace->forward(
                weights,metadata,source,destination,rows,projection_stream,
                admission);
        }
        if (projection_timing_)
            projection_timing_->end(timing_slot, projection_stream);
    };
    const auto project = [&](Exl3CudaLinearWorkspace* workspace,
                             const Exl3CudaLinearWeights& weights,
                             const Exl3CudaLinearMetadata& metadata,
                              const std::uint16_t* source,
                              std::uint16_t* destination,
                              Exl3TargetProjectionOperator operation,
                              const std::uint16_t* transformed_input=nullptr) {
        project_on(workspace, weights, metadata, source, destination, operation,
                   stream,transformed_input);
    };

    begin(0);
    exl3_launch_small(rms_norm_kernel,dim3(rows),dim3(512),512 * sizeof(float),stream,input, weights_.input_norm,
        input_norm, rows, kHidden);
    launch(cudaGetLastError(), "launch EXL3 input RMSNorm"); end(0);

    const bool can_share_target=target_q_executor_ && !capture_active_ && !oscar_ &&
        !profile && !projection_timing_ && !projection_observer_ &&
        preserve_m1_topology && !wide_prefill && rows>=1 && rows<=8;
    const bool concurrent_qkv=prefill_qkv_concurrency_.complete() &&
        !capture_active_ && !oscar_ && !profile && !projection_timing_ &&
        !projection_observer_ && preserve_m1_topology && wide_prefill &&
        rows>=17 && rows<=1024 && !can_share_target;
    const auto* projection_chain_value=
        std::getenv("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS");
    const auto* projection_chain_full_value=
        std::getenv("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_FULL");
    if(projection_chain_full_value &&
       std::strcmp(projection_chain_full_value,"0")!=0 &&
       std::strcmp(projection_chain_full_value,"1")!=0)
        throw std::invalid_argument(
            "NINFER_EXL3_PREFILL_PROJECTION_CHAIN_FULL must be 0 or 1");
    const bool projection_chain=projection_chain_value &&
        std::strcmp(projection_chain_value,"1")==0 &&
        (!projection_chain_full_value ||
         std::strcmp(projection_chain_full_value,"1")==0) && !capture_active_ &&
        !oscar_ && !profile && !projection_timing_ && !projection_observer_ &&
        preserve_m1_topology && wide_prefill && rows==1024 && !can_share_target &&
        !concurrent_qkv;
    if(projection_chain) {
        std::array<Exl3GraphBufferIdentity,16> buffers{};
        std::array<Exl3GraphBoundResource,16> resources{};
        std::size_t count=0;
        const auto append=[&](const void* address,std::size_t bytes,
                              const std::shared_ptr<const void>& owner) {
            if(!address || !bytes || count==buffers.size())
                throw std::logic_error("full-attention projection-chain resource");
            buffers[count]={address,bytes};
            resources[count]={owner,address,bytes,count+1};
            ++count;
        };
        append(input_norm,static_cast<std::size_t>(rows)*kHidden*2,
               prefill_projection_chain_scratch_owner_);
        append(qg,static_cast<std::size_t>(rows)*weights_.q_metadata.out_features*2,
               prefill_projection_chain_scratch_owner_);
        append(kp,static_cast<std::size_t>(rows)*weights_.k_metadata.out_features*2,
               prefill_projection_chain_scratch_owner_);
        append(vp,static_cast<std::size_t>(rows)*weights_.v_metadata.out_features*2,
               prefill_projection_chain_scratch_owner_);
        const auto append_weights=[&](const Exl3CudaLinearWeights& value) {
            if(value.trellis)append(value.trellis,2,prefill_projection_chain_model_owner_);
            if(value.suh)append(value.suh,2,prefill_projection_chain_model_owner_);
            if(value.svh)append(value.svh,2,prefill_projection_chain_model_owner_);
            if(value.mul1)append(value.mul1,4,prefill_projection_chain_model_owner_);
        };
        append_weights(weights_.q);append_weights(weights_.k);append_weights(weights_.v);
        Exl3GraphCompatibilityFingerprint fingerprint;
        fingerprint.bind(prefill_projection_chain_context_owner_,
            prefill_projection_chain_model_owner_,
            prefill_projection_chain_scratch_owner_,
            std::span<const Exl3GraphBufferIdentity>(buffers.data(),count),
            rows,max_rows_,rows,kHidden,1,0x46514b56u,
            Exl3GraphPrecision::oscar_int2_fp16,
            Exl3GraphPositionPolicy::oscar_split_class,1,stream);
        Exl3GraphCaptureExtent extent;extent.known=true;
        extent.retained[static_cast<unsigned>(
            Exl3ResourceInventory::Domain::graph_count)]=2;
        Exl3PrefillProjectionChainGraph::Request request{
            std::move(fingerprint),
            std::span<const Exl3GraphBoundResource>(resources.data(),count),
            extent,1,0x46514b560001ull,stream,true};
        begin(1);
        prefill_projection_chain_graph_.execute(request,[&](cudaStream_t graph_stream) {
            project_on(linear_workspaces_[0],weights_.q,weights_.q_metadata,
                input_norm,qg,Exl3TargetProjectionOperator::q,graph_stream);
            project_on(linear_workspaces_[1],weights_.k,weights_.k_metadata,
                input_norm,kp,Exl3TargetProjectionOperator::k,graph_stream);
            project_on(linear_workspaces_[2],weights_.v,weights_.v_metadata,
                input_norm,vp,Exl3TargetProjectionOperator::v,graph_stream);
        });
        launch(cudaGetLastError(),"launch prefill projection-chain QKV graph");
        end(1);
    } else if(concurrent_qkv) {
        launch(cudaEventRecord(prefill_qkv_concurrency_.fork,stream),
               "record wide-prefill QKV fork");
        launch(cudaStreamWaitEvent(prefill_qkv_concurrency_.k_stream,
                                   prefill_qkv_concurrency_.fork,0),
               "fork wide-prefill K stream");
        launch(cudaStreamWaitEvent(prefill_qkv_concurrency_.v_stream,
                                   prefill_qkv_concurrency_.fork,0),
               "fork wide-prefill V stream");
        try {
            project_on(prefill_qkv_concurrency_.k_workspace,weights_.k,
                weights_.k_metadata,input_norm,kp,Exl3TargetProjectionOperator::k,
                prefill_qkv_concurrency_.k_stream);
            launch(cudaGetLastError(),"launch concurrent wide-prefill K projection");
            launch(cudaEventRecord(prefill_qkv_concurrency_.k_done,
                                   prefill_qkv_concurrency_.k_stream),
                   "record wide-prefill K completion");
            project_on(prefill_qkv_concurrency_.v_workspace,weights_.v,
                weights_.v_metadata,input_norm,vp,Exl3TargetProjectionOperator::v,
                prefill_qkv_concurrency_.v_stream);
            launch(cudaGetLastError(),"launch concurrent wide-prefill V projection");
            launch(cudaEventRecord(prefill_qkv_concurrency_.v_done,
                                   prefill_qkv_concurrency_.v_stream),
                   "record wide-prefill V completion");
            project(linear_workspaces_[0],weights_.q,weights_.q_metadata,input_norm,qg,
                    Exl3TargetProjectionOperator::q);
            launch(cudaGetLastError(),"launch concurrent wide-prefill Q projection");
            launch(cudaStreamWaitEvent(stream,prefill_qkv_concurrency_.k_done,0),
                   "join wide-prefill K stream");
            launch(cudaStreamWaitEvent(stream,prefill_qkv_concurrency_.v_done,0),
                   "join wide-prefill V stream");
            ++prefill_qkv_concurrent_calls_;
            prefill_qkv_concurrent_rows_+=static_cast<std::uint64_t>(rows);
        } catch(...) {
            (void)cudaEventRecord(prefill_qkv_concurrency_.k_done,
                                  prefill_qkv_concurrency_.k_stream);
            (void)cudaEventRecord(prefill_qkv_concurrency_.v_done,
                                  prefill_qkv_concurrency_.v_stream);
            (void)cudaStreamWaitEvent(stream,prefill_qkv_concurrency_.k_done,0);
            (void)cudaStreamWaitEvent(stream,prefill_qkv_concurrency_.v_done,0);
            throw;
        }
    } else {
        begin(1);
        // Suspension is explicit at the layer boundary, outside ordinary linear
        // dispatch. Private attention/KV/MLP state resumes only after Q is complete.
        const bool shared_q=can_share_target &&
            target_shared_admission(Exl3TargetSharedFamily::q,weights_.q_metadata).has_value() &&
            target_q_executor_(Exl3TargetQContinuation{weights_.q,weights_.q_metadata,
                input_norm,qg,rows,position,model_layer_,stream});
        if(!shared_q)project(linear_workspaces_[0],weights_.q,weights_.q_metadata,
            input_norm,qg,Exl3TargetProjectionOperator::q);
        launch(cudaGetLastError(),"launch EXL3 Q projection");end(1);
        const bool m1_kv_pair=(fast_same_weights_fp16kv_m1_kv_pair_ ||
            fast_same_weights_fp16kv_m1_kv_wide_pair_) &&
            !preserve_m1_topology && !can_share_target && rows==1 &&
            !wide_prefill && (!capture_active_ ||
                fast_same_weights_fp16kv_m1_kv_pair_graph_) &&
            !oscar_ && !profile &&
            !projection_timing_ &&
            !projection_observer_ && linear_workspaces_[1] &&
            linear_workspaces_[2] &&
            (weights_.k_metadata.K==6 || weights_.k_metadata.K==7) &&
            weights_.v_metadata.K==weights_.k_metadata.K &&
            weights_.k_metadata.mul1 &&
            weights_.v_metadata.mul1 && !weights_.k_metadata.mcg &&
            !weights_.v_metadata.mcg && !weights_.k_metadata.has_bias &&
            !weights_.v_metadata.has_bias &&
            weights_.k_metadata.in_features==kHidden &&
            weights_.v_metadata.in_features==kHidden &&
            weights_.k_metadata.out_features==kKVProjection &&
            weights_.v_metadata.out_features==kKVProjection;
        begin(2);
        const bool merged_kv=!m1_kv_pair && rows==1 && !can_share_target && !oscar_ &&
            !profile && !projection_timing_ && !projection_observer_ && !wide_prefill &&
            linear_workspaces_[1] && linear_workspaces_[2] &&
            linear_workspaces_[1]->forward_m1_pair(*linear_workspaces_[2],
                weights_.k,weights_.k_metadata,kp,weights_.v,weights_.v_metadata,vp,
                input_norm,stream);
        if(merged_kv) {
            launch(cudaGetLastError(),"launch merged EXL3 K/V projection");
        } else if(m1_kv_pair) {
            if(fast_same_weights_fp16kv_m1_kv_wide_pair_)
                linear_workspaces_[1]->forward_target_m1_kv_wide_pair_for_test(
                    *linear_workspaces_[2],weights_.k,weights_.k_metadata,
                    weights_.v,weights_.v_metadata,input_norm,kp,vp,stream);
            else linear_workspaces_[1]->forward_target_m1_kv_pair_for_test(
                    *linear_workspaces_[2],weights_.k,weights_.k_metadata,
                    weights_.v,weights_.v_metadata,input_norm,kp,vp,stream);
            launch(cudaGetLastError(),"launch EXL3 paired K/V projection");
            ++fast_same_weights_fp16kv_m1_kv_pair_submissions_;
        } else {
            const bool shared_k=can_share_target && target_kv_executor_enabled_ &&
                target_shared_admission(Exl3TargetSharedFamily::k,weights_.k_metadata).has_value() &&
                target_q_executor_(Exl3TargetQContinuation{weights_.k,weights_.k_metadata,
                    input_norm,kp,rows,position,model_layer_,stream,Exl3TargetSharedFamily::k});
            if(!shared_k)project(linear_workspaces_[1],weights_.k,weights_.k_metadata,input_norm,kp,
                Exl3TargetProjectionOperator::k);
            launch(cudaGetLastError(),"launch EXL3 K projection");
        }
        end(2);
        begin(3);
        if(!m1_kv_pair && !merged_kv) {
            const bool shared_v=can_share_target && target_kv_executor_enabled_ &&
                target_shared_admission(Exl3TargetSharedFamily::v,weights_.v_metadata).has_value() &&
                target_q_executor_(Exl3TargetQContinuation{weights_.v,weights_.v_metadata,
                    input_norm,vp,rows,position,model_layer_,stream,Exl3TargetSharedFamily::v});
            if(!shared_v)project(linear_workspaces_[2],weights_.v,weights_.v_metadata,input_norm,vp,
                Exl3TargetProjectionOperator::v);
            launch(cudaGetLastError(),"launch EXL3 V projection");
        }
        end(3);
    }

    exl3_launch_small(split_qg_kernel,dim3((rows * kQHeads * kHeadDim + 255) / 256),dim3(256),0,stream,qg, qn, gp, rows);
    launch(cudaGetLastError(), "launch EXL3 Q/gate split");
    begin(4);
    exl3_launch_small(rms_norm_kernel,dim3(rows * kQHeads),dim3(256),256 * sizeof(float),stream,qn, weights_.q_norm,
        qn, rows * kQHeads, kHeadDim);
    exl3_launch_small(rms_norm_kernel,dim3(rows * kKVHeads),dim3(256),256 * sizeof(float),stream,kp, weights_.k_norm,
        kn, rows * kKVHeads, kHeadDim);
    launch(cudaGetLastError(), "launch EXL3 Q/K RMSNorm"); end(4);

    begin(5);
    if(mrope_positions_||rope_offset_){
        exl3_launch_small(rope_kernel<true>,dim3((rows * kQHeads * kHeadDim + 255) / 256),dim3(256),0,stream,qn,kn,qr,kr,rows,position,position_device_,mrope_positions_,rope_offset_);
        exl3_launch_small(rope_k_kernel<true>,dim3((rows * kKVHeads * kHeadDim + 255) / 256),dim3(256),0,stream,kn,kr,rows,position,position_device_,mrope_positions_,rope_offset_);
    }else{
        exl3_launch_small(rope_kernel<false>,dim3((rows * kQHeads * kHeadDim + 255) / 256),dim3(256),0,stream,qn,kn,qr,kr,rows,position,position_device_,nullptr,0);
        exl3_launch_small(rope_k_kernel<false>,dim3((rows * kKVHeads * kHeadDim + 255) / 256),dim3(256),0,stream,kn,kr,rows,position,position_device_,nullptr,0);
    }
    launch(cudaGetLastError(), "launch EXL3 RoPE"); end(5);

    const char* attention_core_sample_option=std::getenv(
        "NINFER_EXL3_TEST_ATTENTION_CORE_SAMPLE");
    static std::atomic<int> attention_core_samples{0};
    const int attention_core_sample_index=rows>=256 && wide_prefill &&
        !capture_active_ && attention_core_sample_option &&
        std::strcmp(attention_core_sample_option,"1")==0 ?
        attention_core_samples.fetch_add(1,std::memory_order_relaxed):-1;
    const bool attention_core_sample=rows>=256 && wide_prefill &&
        !capture_active_ && attention_core_sample_index>=0 &&
        attention_core_sample_index<16;
    if(attention_core_sample && attention_core_sample_index==0) {
        int max_shared=0;
        cuda_check(cudaDeviceGetAttribute(&max_shared,
            cudaDevAttrMaxSharedMemoryPerBlockOptin,0),
            "query SM120 shared-memory ceiling");
        cudaFuncAttributes attributes{};
        cuda_check(cudaFuncGetAttributes(&attributes,
            attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim+16>),
            "query WMMA32 kernel resources");
        std::fprintf(stderr,
            "ATTENTION_RESOURCE_SAMPLE optin_shared_bytes=%d wmma32_static_shared_bytes=%zu wmma32_registers_per_thread=%d\n",
            max_shared,attributes.sharedSizeBytes,attributes.numRegs);
    }
    cudaEvent_t attention_core_start{},attention_core_end{};
    if(attention_core_sample) {
        cuda_check(cudaEventCreate(&attention_core_start),
            "create attention core sample start");
        cuda_check(cudaEventCreate(&attention_core_end),
            "create attention core sample end");
        cuda_check(cudaEventRecord(attention_core_start,stream),
            "record attention core sample start");
    }
    begin(6);
    if (oscar_ != nullptr && capture_active_ && preserve_m1_topology && rows > 1) {
        if (rows <= 8 && oscar_->continuation_cohort_b8_enabled()) {
            oscar_->forward_continuation_device_cohort(
                oscar_layer_,qr,kr,vp,attn,rows,position_device_,
                oscar_->graph_class(),stream);
            launch(cudaGetLastError(),
                   "launch OSCAR chronological cohort graph attention");
        } else {
            oscar_->forward_continuation_device(oscar_layer_,qr,kr,vp,attn,rows,
                position_device_,oscar_->graph_class(),stream);
            launch(cudaGetLastError(), "launch B8 OSCAR continuation graph attention");
        }
    } else if (oscar_ != nullptr && oscar_->graph_class() != 0) {
        // E4C1 graph-safe device path: single-token append + decode with all
        // extents from device memory. Host position/counts stay frozen.
        if (position_device_ == nullptr) {
            throw std::invalid_argument("E4C1 OSCAR graph path needs a position device");
        }
        if (rows != 1) {
            throw std::invalid_argument("E4C1 OSCAR graph path takes single rows");
        }
        oscar_->forward_decode_device(oscar_layer_, qr, kr, vp, attn, position_device_,
                                          stream);
        launch(cudaGetLastError(), "launch E4C1 OSCAR graph attention");
    } else if (oscar_ != nullptr) {
        // E4C1 OSCAR owns attention and cache for this context. Single-row
        // decode uses the fused split-KV path; multi-row ranges use the fused
        // batch prefill path (causal over the resident cache). The F16 KV
        // cache is intentionally not appended: no historical BF16 shadow.
        if (preserve_m1_topology && wide_prefill && rows > 1 &&
            oscar_->prefill_batch_rotations_enabled()) {
            oscar_->prefill_chronological_layer(oscar_layer_, qr, kr, vp, rows,
                static_cast<std::uint32_t>(position), attn, stream);
        } else if (preserve_m1_topology && rows > 1) {
            // The eager continuation cohort is explicit/default-off. It is
            // admitted only for target-verifier rows and a single adaptive
            // split class; the helper returns false for the default latch,
            // malformed inputs, or 512/8192 boundary crossings, preserving
            // the exact row-ordered append/decode fallback below.
            const bool eager_cohort = !wide_prefill &&
                oscar_->try_forward_continuation_device_cohort_eager(
                    oscar_layer_,qr,kr,vp,attn,rows,position,
                    position_device_,stream);
            if (!eager_cohort) {
                for (int row = 0; row < rows; ++row) {
                    oscar_->append_layer(oscar_layer_,
                        qr + static_cast<std::size_t>(row) * kQHeads * kHeadDim,
                        kr + static_cast<std::size_t>(row) * kKVHeads * kHeadDim,
                        vp + static_cast<std::size_t>(row) * kKVHeads * kHeadDim,
                        1, static_cast<std::uint32_t>(position + row), stream);
                    oscar_->decode_layer(oscar_layer_,
                        qr + static_cast<std::size_t>(row) * kQHeads * kHeadDim,
                        attn + static_cast<std::size_t>(row) * kQHeads * kHeadDim,
                        static_cast<std::uint32_t>(position + row), 0, stream);
                }
            }
        } else if (rows == 1) {
            oscar_->append_layer(oscar_layer_, qr, kr, vp, rows,
                                 static_cast<std::uint32_t>(position), stream);
            oscar_->decode_layer(oscar_layer_, qr, attn,
                                 static_cast<std::uint32_t>(position), 0, stream);
        } else {
            oscar_->prefill_layer(oscar_layer_, qr, kr, vp, rows,
                                  static_cast<std::uint32_t>(position), attn, stream);
        }
        launch(cudaGetLastError(), "launch E4C1 OSCAR attention");
        oscar_count_ += preserve_m1_topology ? static_cast<std::uint64_t>(rows) : 1ULL;
    } else if (k_cache_ != nullptr || v_cache_ != nullptr) {
        ++ordinary_count_;
        if (k_cache_ == nullptr || v_cache_ == nullptr || cache_capacity_ <= 0 ||
            position < 0 || position + rows > cache_capacity_) {
            throw std::invalid_argument("invalid EXL3 attention cache configuration");
        }
        exl3_launch_small(append_kv_cache_kernel,dim3((rows * kKVHeads * kHeadDim + 255) / 256),dim3(256),0,stream,
            kr, vp, k_cache_, v_cache_, rows, position, cache_capacity_, position_device_);
        launch(cudaGetLastError(), "append EXL3 attention KV cache");
        if(direct_staged_rows_) {
            exl3_launch_small(append_kv_cache_kernel,dim3((rows * kKVHeads * kHeadDim + 255) / 256),dim3(256),0,stream,
                kr, vp, direct_staged_k_, direct_staged_v_, rows, position,
                cache_capacity_, position_device_);
            launch(cudaGetLastError(), "append direct staged EXL3 attention KV");
        }
        const auto* attention_k=direct_staged_rows_?direct_staged_k_:k_cache_;
        const auto* attention_v=direct_staged_rows_?direct_staged_v_:v_cache_;
        // The existing E3A kernel is the qualified prefill reference.  Use it
        // while appending the complete prefill block, then use the persistent
        // cache path for incremental decode.  This also keeps the two paths
        // directly comparable during E4A bring-up.
        // Opt-in denominator includes prefill and incompatible-profile fallback.
        // This counts dispatch attempts, including later failures, not completions
        // or physical history transactions. Disabled requests do not update it.
        if(exact_attention_query_pair_)
            query_pair_requested_row_attempts_+=static_cast<std::uint64_t>(rows);
        // HostKV layer-graph capture uses a host placeholder position of zero
        // while position_device_ supplies the live decode frontier at replay.
        // Do not freeze the prefill-only attention branch into M2..M8 graphs.
        if (ordinary_full_layer_graph_capture_ && rows == 1 && position > 0 &&
            std::getenv("NINFER_EXL3_ORDINARY_FULL_LAYER_GRAPH_DIAGNOSTIC")) {
            std::fprintf(stderr,
                "ordinary_full_layer_graph_attention_gate layer=%d capture=%d marker=%d fast_flash=%d direct=%d prefix=%d pages=%d exact_scores=%d score_rows=%d six_scores=%d triple_values=%d scalar_values=%d profile=%d position=%d\\n",
                model_layer_, capture_active_ ? 1 : 0,
                ordinary_full_layer_graph_capture_ ? 1 : 0,
                fast_fused_flash_attention_ ? 1 : 0,
                direct_staged_rows_, exact_prefix_rows_, exact_page_ranges_.count,
                exact_scores_ ? 1 : 0, exact_score_rows_,
                exact_attention_gqa_six_scores_ ? 1 : 0,
                exact_attention_gqa_six_softmax_triple_values_ ? 1 : 0,
                exact_attention_gqa_six_softmax_fused_scalar_values_ ? 1 : 0,
                profile ? 1 : 0, position);
        }
        // The guarded verifier route uses the same fused attention reduction
        // for scalar seeds and B2..B8 proposals.  The older tiled prefill
        // branch otherwise intercepts those small batches first.
        const bool fused_flash_eligible=fast_fused_flash_attention_ &&
            (rows==1 || (fast_fused_flash_multirow_ && rows>=2 && rows<=8)) &&
            position>0 && (!capture_active_ || ordinary_full_layer_graph_capture_) &&
            !direct_staged_rows_ && !exact_prefix_rows_ &&
            !exact_page_ranges_.count &&
            exact_scores_ && exact_score_rows_>=rows &&
            exact_attention_gqa_six_scores_ &&
            exact_attention_gqa_six_softmax_triple_values_ &&
            !exact_attention_gqa_six_softmax_fused_scalar_values_;
        if ((fast_prefill_wmma32_attention_ || fast_prefill_wmma_attention_) &&
            fast_prefill_tiled_attention_ &&
            rows > 1 && rows <= 1024 && !capture_active_ &&
            !direct_staged_rows_ && !exact_prefix_rows_ &&
            !exact_page_ranges_.count && !numeric_attention_splitk_ &&
            !fused_flash_eligible) {
            // T77 is selected explicitly when requested; T75 remains the
            // fallback WMMA candidate under its original gate.
            if (fast_wmma32_split2_output_ && fast_wmma32_split2_capacity_splits_>=4 &&
                fast_wmma32_split2_capacity_rows_>=rows && exl3_fa2_prefill_enabled()) {
                launch_fa2_prefill(qr,attention_k,attention_v,attn,rows,position,
                    cache_capacity_,fast_wmma32_split2_output_,fast_wmma32_split2_stats_,stream);
                static std::atomic<int> fa2_prefill_dispatches{0};
                if(fa2_prefill_dispatches.fetch_add(1,std::memory_order_relaxed)==0)
                    std::fprintf(stderr,"FA2_PREFILL_DISPATCH rows=%d position=%d\n",rows,position);
            } else if (fast_wmma32_split2_output_ && fast_prefill_wmma32_attention_ &&
                position+rows>=8192) {
                if(!fast_wmma32_split2_stats_ ||
                   fast_wmma32_split2_capacity_rows_<rows)
                    throw std::logic_error("WMMA32 split2 workspace extent");
                if(fast_prefill_wmma64_attention_ && rows%64==0 &&
                   fast_wmma32_split_count_==2) {
                    if(fast_prefill_wmma64_shared_heads_attention_ &&
                       fast_prefill_wmma32_padded_attention_)
                        attention_cached_gqa_six_wmma64_register_prefill_kernel<
                            kHeadDim+16,2,2><<<
                            dim3(rows/32,kKVHeads,2*kQHeads/(kKVHeads*2)),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else if(fast_prefill_wmma64_shared_heads_attention_)
                        attention_cached_gqa_six_wmma64_register_prefill_kernel<
                            kHeadDim,2,2><<<
                            dim3(rows/32,kKVHeads,2*kQHeads/(kKVHeads*2)),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else if(fast_prefill_wmma64_register_keys64_attention_ &&
                            fast_prefill_wmma32_padded_attention_)
                        attention_cached_gqa_six_wmma64_register_prefill_kernel<
                            kHeadDim+16,2,1,64><<<
                            dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else if(fast_prefill_wmma64_register_keys64_attention_)
                        attention_cached_gqa_six_wmma64_register_prefill_kernel<
                            kHeadDim,2,1,64><<<
                            dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else if(fast_prefill_wmma64_register_attention_ &&
                       fast_prefill_wmma32_padded_attention_)
                        attention_cached_gqa_six_wmma64_register_prefill_kernel<
                            kHeadDim+16,2><<<
                            dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else if(fast_prefill_wmma64_register_attention_)
                        attention_cached_gqa_six_wmma64_register_prefill_kernel<
                            kHeadDim,2><<<
                            dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else if(launch_wmma32_register_selected<2,64>(qr,attention_k,
                                attention_v,attn,rows,position,cache_capacity_,nullptr,0,
                                fast_wmma32_split2_output_,fast_wmma32_split2_stats_,stream)) {
                    } else if(fast_prefill_wmma32_padded_attention_)
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim+16,2,64><<<
                            dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim,2,64><<<
                            dim3(rows/64,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    if(fast_prefill_wmma64_attention_counter.fetch_add(
                           1,std::memory_order_relaxed)==0)
                        std::fprintf(stderr,
                            "WMMA64_PREFILL_DISPATCH split=2 rows=%d register=%d heads=%d keys=%d\n",
                            rows,fast_prefill_wmma64_register_attention_?1:0,
                            fast_prefill_wmma64_shared_heads_attention_?2:1,
                            fast_prefill_wmma64_register_keys64_attention_?64:32);
                } else if(fast_wmma32_split_count_==4) {
                    if(fast_prefill_wmma32_padded_attention_)
                    {
                    if(launch_wmma32_register_selected<4>(qr,attention_k,
                           attention_v,attn,rows,position,cache_capacity_,nullptr,0,
                           fast_wmma32_split2_output_,fast_wmma32_split2_stats_,stream)) {
                    } else if(wmma32_vector_loads_enabled())
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim+16,4,32,true><<<
                            dim3((rows+31)/32,kKVHeads,4*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim+16,4><<<
                            dim3((rows+31)/32,kKVHeads,4*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    }
                    else
                    {
                    if(launch_wmma32_register_selected<4>(qr,attention_k,
                           attention_v,attn,rows,position,cache_capacity_,nullptr,0,
                           fast_wmma32_split2_output_,fast_wmma32_split2_stats_,stream)) {
                    } else if(wmma32_vector_loads_enabled())
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim,4,32,true><<<
                            dim3((rows+31)/32,kKVHeads,4*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim,4><<<
                            dim3((rows+31)/32,kKVHeads,4*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    }
                } else {
                    if(fast_wmma32_split_count_!=2)
                        throw std::logic_error("WMMA32 split count");
                    if(fast_prefill_wmma32_padded_attention_)
                    {
                    if(launch_wmma32_register_selected<2>(qr,attention_k,
                           attention_v,attn,rows,position,cache_capacity_,nullptr,0,
                           fast_wmma32_split2_output_,fast_wmma32_split2_stats_,stream)) {
                    } else if(wmma32_vector_loads_enabled())
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim+16,2,32,true><<<
                            dim3((rows+31)/32,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim+16,2><<<
                            dim3((rows+31)/32,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    }
                    else
                    {
                    if(launch_wmma32_register_selected<2>(qr,attention_k,
                           attention_v,attn,rows,position,cache_capacity_,nullptr,0,
                           fast_wmma32_split2_output_,fast_wmma32_split2_stats_,stream)) {
                    } else if(wmma32_vector_loads_enabled())
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim,2,32,true><<<
                            dim3((rows+31)/32,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    else
                        attention_cached_gqa_six_wmma32_prefill_kernel<
                            kHeadDim,2><<<
                            dim3((rows+31)/32,kKVHeads,2*kQHeads/kKVHeads),
                            256,0,stream>>>(qr,attention_k,attention_v,attn,
                            rows,position,cache_capacity_,nullptr,0,
                            fast_wmma32_split2_output_,fast_wmma32_split2_stats_);
                    }
                }
                launch(cudaGetLastError(),"launch WMMA32 split2 partials");
                if(fast_wmma32_split_count_==4)
                    attention_wmma32_split_merge_kernel<4><<<
                        (rows*kQHeads*kHeadDim+255)/256,256,0,stream>>>(
                        fast_wmma32_split2_output_,fast_wmma32_split2_stats_,
                        attn,rows);
                else
                    attention_wmma32_split_merge_kernel<2><<<
                        (rows*kQHeads*kHeadDim+255)/256,256,0,stream>>>(
                        fast_wmma32_split2_output_,fast_wmma32_split2_stats_,
                        attn,rows);
            } else if (fast_prefill_wmma64_attention_ && rows%64==0) {
                if(fast_prefill_wmma64_shared_heads_attention_ &&
                   fast_prefill_wmma32_padded_attention_ &&
                   position+rows>=8192)
                    attention_cached_gqa_six_wmma64_register_prefill_kernel<
                        kHeadDim+16,1,2><<<
                        dim3(rows/32,kKVHeads,kQHeads/(kKVHeads*2)),
                        256,0,stream>>>(
                        qr,attention_k,attention_v,attn,rows,position,
                        cache_capacity_,nullptr,0,nullptr,nullptr);
                else if(fast_prefill_wmma64_shared_heads_attention_)
                    attention_cached_gqa_six_wmma64_register_prefill_kernel<
                        kHeadDim,1,2><<<
                        dim3(rows/32,kKVHeads,kQHeads/(kKVHeads*2)),
                        256,0,stream>>>(
                        qr,attention_k,attention_v,attn,rows,position,
                        cache_capacity_,nullptr,0,nullptr,nullptr);
                else if(fast_prefill_wmma64_register_keys64_attention_ &&
                   fast_prefill_wmma32_padded_attention_ &&
                   position+rows>=8192)
                    attention_cached_gqa_six_wmma64_register_prefill_kernel<
                        kHeadDim+16,1,1,64><<<
                        dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
                        qr,attention_k,attention_v,attn,rows,position,
                        cache_capacity_,nullptr,0,nullptr,nullptr);
                else if(fast_prefill_wmma64_register_keys64_attention_)
                    attention_cached_gqa_six_wmma64_register_prefill_kernel<
                        kHeadDim,1,1,64><<<
                        dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
                        qr,attention_k,attention_v,attn,rows,position,
                        cache_capacity_,nullptr,0,nullptr,nullptr);
                else if(fast_prefill_wmma64_register_attention_ &&
                   fast_prefill_wmma32_padded_attention_ &&
                   position+rows>=8192)
                    attention_cached_gqa_six_wmma64_register_prefill_kernel<
                        kHeadDim+16,1><<<
                        dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
                        qr,attention_k,attention_v,attn,rows,position,
                        cache_capacity_,nullptr,0,nullptr,nullptr);
                else if(fast_prefill_wmma64_register_attention_)
                    attention_cached_gqa_six_wmma64_register_prefill_kernel<
                        kHeadDim,1><<<
                        dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
                        qr,attention_k,attention_v,attn,rows,position,
                        cache_capacity_,nullptr,0,nullptr,nullptr);
                else if(launch_wmma32_register_selected<1,64>(qr,attention_k,
                            attention_v,attn,rows,position,cache_capacity_,nullptr,0,
                            nullptr,nullptr,stream)) {
                } else if(fast_prefill_wmma32_padded_attention_ && position+rows>=8192)
                    attention_cached_gqa_six_wmma32_prefill_kernel<
                        kHeadDim+16,1,64><<<
                        dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
                        qr,attention_k,attention_v,attn,rows,position,
                        cache_capacity_,nullptr,0,nullptr,nullptr);
                else
                    attention_cached_gqa_six_wmma32_prefill_kernel<
                        kHeadDim,1,64><<<
                        dim3(rows/64,kKVHeads,kQHeads/kKVHeads),256,0,stream>>>(
                        qr,attention_k,attention_v,attn,rows,position,
                        cache_capacity_,nullptr,0,nullptr,nullptr);
                if(fast_prefill_wmma64_attention_counter.fetch_add(
                       1,std::memory_order_relaxed)==0)
                    std::fprintf(stderr,
                        "WMMA64_PREFILL_DISPATCH split=1 rows=%d register=%d heads=%d keys=%d\n",
                        rows,fast_prefill_wmma64_register_attention_?1:0,
                        fast_prefill_wmma64_shared_heads_attention_?2:1,
                        fast_prefill_wmma64_register_keys64_attention_?64:32);
            } else if (fast_prefill_wmma32_padded_attention_ &&
                position+rows>=8192 && wmma32_vector_loads_enabled()) {
                if(!launch_wmma32_register_selected<1>(qr,attention_k,attention_v,
                       attn,rows,position,cache_capacity_,nullptr,0,nullptr,nullptr,stream))
                attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim+16,1,32,true><<<
                    dim3((rows + 31) / 32, kKVHeads,
                         kQHeads / kKVHeads), 256, 0, stream>>>(
                    qr, attention_k, attention_v, attn, rows, position,
                    cache_capacity_, nullptr, 0,nullptr,nullptr);
            } else if (fast_prefill_wmma32_padded_attention_ &&
                position+rows>=8192) {
                attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim+16><<<
                    dim3((rows + 31) / 32, kKVHeads,
                         kQHeads / kKVHeads), 256, 0, stream>>>(
                    qr, attention_k, attention_v, attn, rows, position,
                    cache_capacity_, nullptr, 0,nullptr,nullptr);
            } else if (fast_prefill_wmma32_attention_ && wmma32_vector_loads_enabled()) {
                if(!launch_wmma32_register_selected<1>(qr,attention_k,attention_v,
                       attn,rows,position,cache_capacity_,nullptr,0,nullptr,nullptr,stream))
                attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim,1,32,true><<<
                    dim3((rows + 31) / 32, kKVHeads,
                         kQHeads / kKVHeads), 256, 0, stream>>>(
                    qr, attention_k, attention_v, attn, rows, position,
                    cache_capacity_, nullptr, 0,nullptr,nullptr);
            } else if (fast_prefill_wmma32_attention_) {
                attention_cached_gqa_six_wmma32_prefill_kernel<kHeadDim><<<
                    dim3((rows + 31) / 32, kKVHeads,
                         kQHeads / kKVHeads), 256, 0, stream>>>(
                    qr, attention_k, attention_v, attn, rows, position,
                    cache_capacity_, nullptr, 0,nullptr,nullptr);
            } else {
                attention_cached_gqa_six_wmma_prefill_kernel<<<
                    dim3((rows + 15) / 16, kKVHeads,
                         kQHeads / (kKVHeads * 2)), 256, 0, stream>>>(
                    qr, attention_k, attention_v, attn, rows, position,
                    cache_capacity_, nullptr, 0);
            }
            launch(cudaGetLastError(),
                   fast_prefill_wmma32_attention_
                       ? "launch FAST same-weight FP16-KV WMMA32 prefill attention"
                       : "launch FAST same-weight FP16-KV WMMA prefill attention");
            fast_prefill_tiled_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
            fast_prefill_wmma_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
            if (fast_prefill_wmma32_attention_)
                fast_prefill_wmma32_attention_counter.fetch_add(
                    1, std::memory_order_relaxed);
        } else if (fast_prefill_rows8_attention_ && fast_prefill_tiled_attention_ &&
            rows > 1 && rows <= 1024 && !capture_active_ &&
            !direct_staged_rows_ && !exact_prefix_rows_ &&
            !exact_page_ranges_.count && !numeric_attention_splitk_ &&
            !fused_flash_eligible) {
            // T74 numerical candidate: eight adjacent causal rows share the
            // represented K/V load.  Its smaller key tile keeps the larger
            // query block below the physical-C1 shared-memory limit; the
            // separate gate and counter keep the exact/reference lane intact.
            if(fast_prefill_rows8_threads128_)
                attention_cached_gqa_six_online_rows_kernel<8,128,2><<<
                    ((rows + 7) / 8) * kKVHeads, 128, 0, stream>>>(
                        qr, attention_k, attention_v, attn, rows, position,
                        cache_capacity_, nullptr, 0);
            else
                attention_cached_gqa_six_online_rows_kernel<8,128,1><<<
                    ((rows + 7) / 8) * kKVHeads, 256, 0, stream>>>(
                        qr, attention_k, attention_v, attn, rows, position,
                        cache_capacity_, nullptr, 0);
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV eight-row prefill attention");
            fast_prefill_tiled_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
            fast_prefill_rows8_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
        } else if (fast_prefill_rows2_attention_ && fast_prefill_tiled_attention_ &&
            rows > 1 && rows <= 1024 && !capture_active_ &&
            !direct_staged_rows_ && !exact_prefix_rows_ &&
            !exact_page_ranges_.count && !numeric_attention_splitk_ &&
            !fused_flash_eligible) {
            attention_cached_gqa_six_online_rows_kernel<2><<<
                ((rows + 1) / 2) * kKVHeads, 256, 0, stream>>>(
                    qr, attention_k, attention_v, attn, rows, position,
                    cache_capacity_, nullptr, 0);
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV two-row prefill attention");
            fast_prefill_tiled_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
            fast_prefill_rows2_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
        } else if (fast_prefill_rows4_attention_ && fast_prefill_tiled_attention_ &&
            rows > 1 && rows <= 1024 && !capture_active_ &&
            !direct_staged_rows_ && !exact_prefix_rows_ &&
            !exact_page_ranges_.count && !numeric_attention_splitk_ &&
            !fused_flash_eligible) {
            // T73 numerical candidate: four adjacent causal rows share each
            // represented K/V load within one CTA. Keep the established T71
            // counter as the umbrella Fast-prefill dispatch and expose the
            // rows4 counter separately for campaign provenance.
            attention_cached_gqa_six_online_rows_kernel<4><<<
                ((rows + 3) / 4) * kKVHeads, 256, 0, stream>>>(
                    qr, attention_k, attention_v, attn, rows, position,
                    cache_capacity_, nullptr, 0);
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV four-row prefill attention");
            fast_prefill_tiled_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
            fast_prefill_rows4_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
        } else if (fast_prefill_tiled_attention_ && rows > 1 && rows <= 1024 &&
                   !capture_active_ && !direct_staged_rows_ &&
                   !exact_prefix_rows_ && !exact_page_ranges_.count &&
                   !numeric_attention_splitk_ && !fused_flash_eligible) {
            // Numerical Fast prefill candidate. T71 qualified this online
            // softmax/value organization against the exact represented-input
            // route; it is explicitly excluded from the exact/reference lane.
            attention_cached_gqa_six_online_tiled_kernel<<<
                rows * kKVHeads, 256, 0, stream>>>(
                    qr, attention_k, attention_v, attn, rows, position,
                    cache_capacity_, nullptr, 0);
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV tiled prefill attention");
            fast_prefill_tiled_attention_counter.fetch_add(
                1, std::memory_order_relaxed);
        } else if (position == 0 && rows > 1 && !capture_active_) {
            attention_kernel<<<rows * kQHeads, 256, 0, stream>>>(qr, kr, vp, attn, rows);
        } else if(fast_online_decode_attention_ &&
                  fast_online_decode_attention_warp_heads_ && rows==1 &&
                  position>0 && !capture_active_ && !direct_staged_rows_ &&
                  !exact_prefix_rows_ && !exact_page_ranges_.count &&
                  !numeric_attention_splitk_ && exact_scores_ &&
                  exact_score_rows_>=rows && exact_attention_gqa_six_scores_ &&
                  exact_attention_gqa_six_softmax_triple_values_ &&
                  !exact_attention_gqa_six_softmax_fused_scalar_values_) {
            attention_cached_gqa_six_online_decode_warp_heads_kernel<<<
                rows*kKVHeads, (kQHeads/kKVHeads)*32, 0, stream>>>(
                    qr,attention_k,attention_v,attn,rows,position,
                    cache_capacity_,position_device_,0);
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV online decode warp-head attention");
            fast_online_decode_attention_counter.fetch_add(
                1,std::memory_order_relaxed);
        } else if(fast_online_decode_attention_ && rows==1 && position>0 &&
                  !capture_active_ && !direct_staged_rows_ &&
                  !exact_prefix_rows_ && !exact_page_ranges_.count &&
                  !numeric_attention_splitk_ && exact_scores_ &&
                  exact_score_rows_>=rows && exact_attention_gqa_six_scores_ &&
                  exact_attention_gqa_six_softmax_triple_values_ &&
                  !exact_attention_gqa_six_softmax_fused_scalar_values_) {
            attention_cached_gqa_six_online_decode_kernel<<<
                rows*kKVHeads,256,0,stream>>>(
                    qr,attention_k,attention_v,attn,rows,position,
                    cache_capacity_,position_device_,0);
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV online decode attention");
            fast_online_decode_attention_counter.fetch_add(
                1,std::memory_order_relaxed);
        } else if(fast_cublas_attention_ && rows==1 && position>0 &&
                  !capture_active_ && !direct_staged_rows_ &&
                  !exact_prefix_rows_ && !exact_page_ranges_.count &&
                  !numeric_attention_splitk_ && exact_scores_ &&
                  exact_score_rows_>=2) {
            // Separately labeled native Fast backend: use the same ordinary
            // token-major FP16 K/V cache, replace QK and V*P with batched
            // tensor-core GEMMs, and retain the established FP32 softmax
            // kernel between them. This candidate never proxies through Mia
            // or changes the cache weights/layout. Eager C1 owns the host
            // position; graph capture is excluded by the admission gate.
            const int count=position+1;
            if(count<1 || count>cache_capacity_)
                throw std::invalid_argument(
                    "FAST cuBLAS attention position extent");
            ensure_fast_cublas_attention_resources(cache_capacity_);
            auto handle=reinterpret_cast<cublasHandle_t>(fast_cublas_handle_);
            cublas_check_attention(cublasSetStream(handle,stream),
                "set FAST same-weight FP16-KV cuBLAS attention stream");
            const float alpha=0.0625f,beta=0.0f;
            cublas_check_attention(cublasGemmStridedBatchedEx(
                handle,CUBLAS_OP_T,CUBLAS_OP_N,
                count,kQHeads/kKVHeads,kHeadDim,
                &alpha,attention_k,CUDA_R_16F,kKVHeads*kHeadDim,
                static_cast<long long>(kHeadDim),
                qr,CUDA_R_16F,kHeadDim,
                static_cast<long long>((kQHeads/kKVHeads)*kHeadDim),
                &beta,exact_scores_,CUDA_R_32F,cache_capacity_,
                static_cast<long long>((kQHeads/kKVHeads)*cache_capacity_),
                kKVHeads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "FAST same-weight FP16-KV cuBLAS GQA QK GEMM");
            attention_cached_gqa_six_softmax_kernel<<<kKVHeads,256,0,stream>>>(
                exact_scores_,rows,position,cache_capacity_,nullptr,0);
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV cuBLAS GQA softmax");
            // The second score row is part of the existing context-owned
            // score allocation. It is disjoint from row zero, so the
            // FP32-normalized scores can be narrowed without an allocation
            // or an in-place read/write race.
            auto* half_probabilities=reinterpret_cast<std::uint16_t*>(
                exact_scores_+static_cast<std::size_t>(kQHeads)*cache_capacity_);
            attention_scores_to_half_kernel<<<
                (kQHeads*count+255)/256,256,0,stream>>>(
                    exact_scores_,half_probabilities,count,cache_capacity_);
            launch(cudaGetLastError(),
                   "convert FAST same-weight FP16-KV cuBLAS GQA probabilities");
            const float value_alpha=1.0f,value_beta=0.0f;
            cublas_check_attention(cublasGemmStridedBatchedEx(
                handle,CUBLAS_OP_N,CUBLAS_OP_N,
                kHeadDim,kQHeads/kKVHeads,count,
                &value_alpha,attention_v,CUDA_R_16F,kKVHeads*kHeadDim,
                static_cast<long long>(kHeadDim),
                half_probabilities,CUDA_R_16F,cache_capacity_,
                static_cast<long long>((kQHeads/kKVHeads)*cache_capacity_),
                &value_beta,attn,CUDA_R_16F,kHeadDim,
                static_cast<long long>((kQHeads/kKVHeads)*kHeadDim),
                kKVHeads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "FAST same-weight FP16-KV cuBLAS GQA V-P GEMM");
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV cuBLAS GQA attention");
            fast_cublas_attention_counter.fetch_add(
                1,std::memory_order_relaxed);
        } else if(fast_whole_context_fused_attention_ && rows==1 && position>0 &&
                  !capture_active_ && !direct_staged_rows_ &&
                  !exact_prefix_rows_ && !exact_page_ranges_.count &&
                  !numeric_attention_splitk_ && exact_scores_ &&
                  exact_score_rows_>=rows && exact_attention_gqa_six_scores_ &&
                  exact_attention_gqa_six_softmax_triple_values_ &&
                  !exact_attention_gqa_six_softmax_fused_scalar_values_) {
            // Diagnostic-only whole-context route: one launch owns the four
            // KV heads and each CTA scans the complete valid cache prefix.
            // It does not consume exact-score scratch, alter cache ownership,
            // or run during graph capture. Every unmet gate falls through to
            // the qualified exact/reference path below.
            attention_cached_gqa_six_whole_context_fused_kernel<<<
                rows*kKVHeads,256,0,stream>>>(
                    qr,attention_k,attention_v,attn,rows,position,
                    cache_capacity_,position_device_,0);
            launch(cudaGetLastError(),
                   "launch FAST whole-context same-weight FP16-KV attention");
            fast_whole_context_fused_attention_counter.fetch_add(
                1,std::memory_order_relaxed);
        } else if(fused_flash_eligible) {
            if (ordinary_full_layer_graph_capture_ &&
                std::getenv("NINFER_EXL3_ORDINARY_FULL_LAYER_GRAPH_DIAGNOSTIC"))
                std::fprintf(stderr,
                    "ordinary_full_layer_graph_fused_flash_branch_entered\\n");
            // The scratch allocation is the larger exact score plane.  Keep
            // the fused candidate inside that existing ownership envelope;
            // no hidden allocation or weight/cache transformation is allowed.
            const int count=position+rows;
            const int fused_keys=fast_fused_flash_attention_keys256_
                ? kFastFusedFlashKeys256 : kFastFusedFlashKeys;
            const int segments=(count+fused_keys-1)/fused_keys;
            const std::size_t required=static_cast<std::size_t>(rows)*kKVHeads*
                segments*kFastFusedFlashStride*sizeof(float);
            if(required>exl3_exact_attention_score_bytes(exact_score_rows_,
                                                         cache_capacity_))
                throw std::invalid_argument(
                    "FAST fused flash attention scratch extent");
            const bool mma=fast_fused_flash_attention_keys256_&&verify_flash_mma_enabled();
            if(mma)
                launch_verify_flash_mma(qr,attention_k,attention_v,exact_scores_,
                    exl3_exact_attention_score_bytes(exact_score_rows_,cache_capacity_),
                    attn,rows,position,cache_capacity_,position_device_,stream);
            else if(fast_fused_flash_attention_keys256_&&fused_flash_staged_k_enabled())
                attention_cached_gqa_six_fused_flash_kernel<kFastFusedFlashKeys256,true><<<
                    rows*kKVHeads*segments,256,0,stream>>>(
                        qr,attention_k,attention_v,exact_scores_,attn,rows,position,
                        cache_capacity_,segments,position_device_,0);
            else if(fast_fused_flash_attention_keys256_)
                attention_cached_gqa_six_fused_flash_kernel<kFastFusedFlashKeys256><<<
                    rows*kKVHeads*segments,256,0,stream>>>(
                        qr,attention_k,attention_v,exact_scores_,attn,rows,position,
                        cache_capacity_,segments,position_device_,0);
            else
                attention_cached_gqa_six_fused_flash_kernel<kFastFusedFlashKeys><<<
                    rows*kKVHeads*segments,256,0,stream>>>(
                        qr,attention_k,attention_v,exact_scores_,attn,rows,position,
                        cache_capacity_,segments,position_device_,0);
            if(!mma)
                launch_fused_flash_merge(exact_scores_,attn,rows,segments,position,
                    cache_capacity_,fused_keys,position_device_,stream);
            launch(cudaGetLastError(),
                   "launch FAST same-weight FP16-KV fused flash attention");
            fast_fused_flash_attention_counter.fetch_add(
                1,std::memory_order_relaxed);
            if(rows>1)fast_fused_flash_multirow_attention_counter.fetch_add(
                1,std::memory_order_relaxed);
        } else if(numeric_attention_splitk_) {
            const std::size_t required=exl3_numeric_attention_splitk_workspace_bytes(
                exact_score_rows_,cache_capacity_);
            if(!numeric_attention_splitk_workspace_||exact_score_rows_<1||
               numeric_attention_splitk_workspace_bytes_!=required)
                throw std::invalid_argument("numeric split-K attention workspace");
            const int segments=(cache_capacity_+kT71SegmentKeys-1)/kT71SegmentKeys;
            for(int first=0;first<rows;first+=exact_score_rows_) {
                const int count=std::min(exact_score_rows_,rows-first);
                attention_cached_gqa_six_splitk_partial_kernel<<<
                    count*kKVHeads*segments,256,0,stream>>>(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                        attention_k,attention_v,numeric_attention_splitk_workspace_,
                        count,position,cache_capacity_,segments,position_device_,first);
                attention_cached_gqa_six_splitk_merge_kernel<<<
                    count*kKVHeads,256,0,stream>>>(numeric_attention_splitk_workspace_,
                        attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                        count,segments);
            }
        } else if(exact_scores_) {
            if(exact_score_rows_<1) throw std::invalid_argument("exact attention score row capacity");
            const auto* shared_score_value=
                std::getenv("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE");
            if(shared_score_value && std::strcmp(shared_score_value,"0")!=0 &&
               std::strcmp(shared_score_value,"1")!=0)
                throw std::invalid_argument(
                    "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE must be 0 or 1");
            const auto* shared_score_rows128_value=
                std::getenv("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_ROWS128");
            if(shared_score_rows128_value &&
               std::strcmp(shared_score_rows128_value,"0")!=0 &&
               std::strcmp(shared_score_rows128_value,"1")!=0)
                throw std::invalid_argument(
                    "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_ROWS128 must be 0 or 1");
            const bool shared_score_rows128=shared_score_rows128_value &&
                std::strcmp(shared_score_rows128_value,"1")==0;
            const auto* shared_score_threads256_value=std::getenv(
                "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_THREADS256");
            if(shared_score_threads256_value &&
               std::strcmp(shared_score_threads256_value,"0")!=0 &&
               std::strcmp(shared_score_threads256_value,"1")!=0)
                throw std::invalid_argument(
                    "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_THREADS256 must be 0 or 1");
            const bool shared_score_threads256=shared_score_threads256_value &&
                std::strcmp(shared_score_threads256_value,"1")==0;
            const auto* shared_score_parallel_softmax_value=std::getenv(
                "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_PARALLEL_SOFTMAX");
            if(shared_score_parallel_softmax_value &&
               std::strcmp(shared_score_parallel_softmax_value,"0")!=0 &&
               std::strcmp(shared_score_parallel_softmax_value,"1")!=0)
                throw std::invalid_argument(
                    "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_PARALLEL_SOFTMAX must be 0 or 1");
            const bool shared_score_parallel_softmax=
                shared_score_parallel_softmax_value &&
                std::strcmp(shared_score_parallel_softmax_value,"1")==0;
            const auto* shared_score_head_split256_value=std::getenv(
                "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_HEAD_SPLIT256");
            if(shared_score_head_split256_value &&
               std::strcmp(shared_score_head_split256_value,"0")!=0 &&
               std::strcmp(shared_score_head_split256_value,"1")!=0)
                throw std::invalid_argument(
                    "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_HEAD_SPLIT256 must be 0 or 1");
            const bool shared_score_head_split256=
                shared_score_head_split256_value &&
                std::strcmp(shared_score_head_split256_value,"1")==0;
            if(shared_score_head_split256 &&
               (shared_score_threads256 || !shared_score_parallel_softmax))
                throw std::invalid_argument(
                    "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_HEAD_SPLIT256 requires parallel softmax and threads256=0");
            const auto* shared_score_dimension_split256_value=std::getenv(
                "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_DIMENSION_SPLIT256");
            if(shared_score_dimension_split256_value &&
               std::strcmp(shared_score_dimension_split256_value,"0")!=0 &&
               std::strcmp(shared_score_dimension_split256_value,"1")!=0)
                throw std::invalid_argument(
                    "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_DIMENSION_SPLIT256 must be 0 or 1");
            const bool shared_score_dimension_split256=
                shared_score_dimension_split256_value &&
                std::strcmp(shared_score_dimension_split256_value,"1")==0;
            if(shared_score_dimension_split256 &&
               (shared_score_threads256 || shared_score_head_split256 ||
                !shared_score_parallel_softmax))
                throw std::invalid_argument(
                    "NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_DIMENSION_SPLIT256 requires parallel softmax, threads256=0 and head_split256=0");
            const bool shared_score_attention=shared_score_value &&
                std::strcmp(shared_score_value,"1")==0 &&
                wide_prefill && !profile &&
                !capture_active_ &&
                (rows==1024 || (rows==128 && shared_score_rows128)) && position>0 &&
                position+rows<=4096 && !direct_staged_rows_ &&
                !exact_prefix_rows_ && !exact_page_ranges_.count &&
                exact_attention_gqa_six_scores_ &&
                exact_attention_gqa_six_softmax_triple_values_ &&
                !exact_attention_gqa_six_ &&
                !exact_attention_gqa_six_values_sharded_ &&
                !exact_attention_gqa_six_packed_triples_ &&
                !exact_attention_gqa_six_query_pair_scores_ &&
                !exact_attention_gqa_six_score_k_tile64_ &&
                !exact_attention_gqa_six_softmax_triple_values_v_tile_ &&
                !exact_attention_gqa_six_softmax_triple_values_v_tile64_ &&
                !exact_attention_gqa_six_softmax_triple_values_full_cta_ &&
                !exact_attention_gqa_six_softmax_triple_values_threads128_ &&
                !exact_attention_gqa_six_softmax_triple_values_pair_dimensions_ &&
                !exact_attention_gqa_six_softmax_six_values_single_load_ &&
                !exact_attention_gqa_six_softmax_triple_values_scalar_dim_ &&
                !exact_attention_gqa_six_softmax_triple_values_two_query_ &&
                !exact_attention_gqa_six_softmax_triple_values_fused_ &&
                !exact_attention_gqa_six_softmax_tile512_;
            const auto* attention_chain_value=
                std::getenv("NINFER_EXL3_PREFILL_ATTENTION_CHAIN_GRAPH");
            const bool attention_chain=attention_chain_value &&
                std::strcmp(attention_chain_value,"1")==0 &&
                preserve_m1_topology && wide_prefill && !profile &&
                !capture_active_ && rows==1024 && exact_score_rows_==16 &&
                position>0 && position<=3072 && position+rows<=4096 &&
                !direct_staged_rows_ && !exact_prefix_rows_ &&
                !exact_page_ranges_.count && exact_attention_gqa_six_scores_ &&
                exact_attention_gqa_six_softmax_triple_values_ &&
                !exact_attention_gqa_six_ &&
                !exact_attention_gqa_six_values_sharded_ &&
                !exact_attention_gqa_six_packed_triples_ &&
                !exact_attention_gqa_six_query_pair_scores_ &&
                !exact_attention_gqa_six_score_k_tile64_ &&
                !exact_attention_gqa_six_softmax_triple_values_v_tile_ &&
                !exact_attention_gqa_six_softmax_triple_values_v_tile64_ &&
                !exact_attention_gqa_six_softmax_triple_values_full_cta_ &&
                !exact_attention_gqa_six_softmax_triple_values_threads128_ &&
                !exact_attention_gqa_six_softmax_triple_values_score_tile_ &&
                !exact_attention_gqa_six_softmax_triple_values_pair_dimensions_ &&
                !exact_attention_gqa_six_softmax_six_values_single_load_ &&
                !exact_attention_gqa_six_softmax_triple_values_scalar_dim_ &&
                !exact_attention_gqa_six_softmax_triple_values_two_query_ &&
                !exact_attention_gqa_six_softmax_triple_values_fused_ &&
                !exact_attention_gqa_six_softmax_tile512_;
            if(shared_score_attention) {
                launch_prefill_attention_shared_scores(qr,attention_k,attention_v,
                    attn,rows,position,cache_capacity_,stream,
                    shared_score_threads256,shared_score_parallel_softmax,
                    shared_score_head_split256,
                    shared_score_dimension_split256);
                launch(cudaGetLastError(),
                    "launch exact shared-score large-M prefill attention");
                ++prefill_shared_score_launch_attempts_;
                prefill_shared_score_row_attempts_+=
                    static_cast<std::uint64_t>(rows);
                const std::uint64_t first_count=
                    static_cast<std::uint64_t>(position+1);
                const std::uint64_t last_count=
                    static_cast<std::uint64_t>(position+rows);
                const std::uint64_t score_elements=
                    (first_count+last_count)*static_cast<std::uint64_t>(rows)/2u*
                    static_cast<std::uint64_t>(kQHeads);
                prefill_shared_score_global_bytes_eliminated_+=
                    score_elements*sizeof(float);
                prefill_shared_score_launch_counter.fetch_add(
                    1,std::memory_order_relaxed);
                prefill_shared_score_row_counter.fetch_add(
                    static_cast<std::uint64_t>(rows),std::memory_order_relaxed);
                prefill_shared_score_bytes_counter.fetch_add(
                    score_elements*sizeof(float),std::memory_order_relaxed);
                if(shared_score_threads256)
                    prefill_shared_score_threads256_counter.fetch_add(
                        1,std::memory_order_relaxed);
                if(shared_score_parallel_softmax)
                    prefill_shared_score_parallel_softmax_counter.fetch_add(
                        1,std::memory_order_relaxed);
                if(shared_score_head_split256)
                    prefill_shared_score_head_split256_counter.fetch_add(
                        1,std::memory_order_relaxed);
                if(shared_score_dimension_split256)
                    prefill_shared_score_dimension_split256_counter.fetch_add(
                        1,std::memory_order_relaxed);
            } else if(attention_chain) {
                std::array<Exl3GraphBufferIdentity,5> identities{};
                std::array<Exl3GraphBoundResource,5> resources{};
                std::size_t resource_count=0;
                const auto append=[&](const void* address,std::size_t bytes,
                                      const std::shared_ptr<const void>& owner) {
                    if(!address || !bytes || resource_count==identities.size())
                        throw std::logic_error("prefill attention-chain resource");
                    identities[resource_count]={address,bytes};
                    resources[resource_count]={owner,address,bytes,resource_count+1};
                    ++resource_count;
                };
                append(qr,static_cast<std::size_t>(rows)*kQHeads*kHeadDim*2,
                    prefill_projection_chain_scratch_owner_);
                append(attention_k,static_cast<std::size_t>(cache_capacity_)*
                    kKVHeads*kHeadDim*2,prefill_projection_chain_context_owner_);
                append(attention_v,static_cast<std::size_t>(cache_capacity_)*
                    kKVHeads*kHeadDim*2,prefill_projection_chain_context_owner_);
                append(attn,static_cast<std::size_t>(rows)*kQHeads*kHeadDim*2,
                    prefill_projection_chain_scratch_owner_);
                append(exact_scores_,static_cast<std::size_t>(exact_score_rows_)*
                    kQHeads*cache_capacity_*sizeof(float),
                    prefill_projection_chain_scratch_owner_);
                Exl3GraphCompatibilityFingerprint fingerprint;
                const std::uint32_t route_bits=0x4154544eu |
                    (exact_attention_gqa_six_extent_shards_?0x80000000u:0u);
                fingerprint.bind(prefill_projection_chain_context_owner_,
                    prefill_projection_chain_model_owner_,
                    prefill_projection_chain_scratch_owner_,
                    std::span<const Exl3GraphBufferIdentity>(
                        identities.data(),resource_count),rows,max_rows_,
                    static_cast<unsigned>(position),
                    static_cast<unsigned>(cache_capacity_),
                    exact_score_rows_,route_bits,
                    Exl3GraphPrecision::oscar_int2_fp16,
                    Exl3GraphPositionPolicy::oscar_split_class,1,stream);
                Exl3GraphCaptureExtent extent;extent.known=true;
                extent.retained[static_cast<unsigned>(
                    Exl3ResourceInventory::Domain::graph_count)]=2;
                Exl3PrefillAttentionChainGraph::Request request{
                    std::move(fingerprint),
                    std::span<const Exl3GraphBoundResource>(
                        resources.data(),resource_count),extent,1,
                    0x4154544e0001ull,stream,position/1024,true};
                prefill_attention_chain_graph_.execute(request,
                    [&](cudaStream_t graph_stream) {
                    // The graph is keyed by this exact host position. Do not
                    // retain the mutable device cursor: append_prefill_impl
                    // republishes that scalar after the layer stack, whereas
                    // every node in this bounded numerical slice needs the
                    // immutable chunk base captured in its launch arguments.
                    for(int first=0;first<rows;first+=exact_score_rows_) {
                        const int count=std::min(exact_score_rows_,rows-first);
                        const int score_shards=exact_attention_gqa_six_extent_shards_?
                            std::min(6,std::max(1,
                                (position+first+count+255)/256)):6;
                        launch_attention_cached_gqa_six_scores(
                            qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            attention_k,exact_scores_,count,position,
                            cache_capacity_,nullptr,first,score_shards,
                            graph_stream);
                        attention_cached_gqa_six_softmax_kernel<<<
                            count*kKVHeads,256,0,graph_stream>>>(exact_scores_,
                                count,position,cache_capacity_,nullptr,first);
                        attention_cached_gqa_triple_normalized_values_kernel<false><<<
                            count*(kQHeads/3),256,0,graph_stream>>>(attention_v,
                                attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                exact_scores_,count,position,cache_capacity_,
                                nullptr,first);
                    }
                });
                // Logical selected-kernel accounting is independent of whether
                // CUDA received eager launches or one graph replay launch.
                gqa_six_softmax_triple_value_launch_attempts_+=
                    static_cast<std::uint64_t>(rows/exact_score_rows_);
                gqa_six_softmax_triple_value_row_attempts_+=
                    static_cast<std::uint64_t>(rows);
            } else for(int first=0;first<rows;first+=exact_score_rows_) {
                const int count=std::min(exact_score_rows_,rows-first);
                const auto segmented_route=exl3_segmented_attention_route(supports_segmented_exact_prefix(),
                    !oscar_,count,position,first,cache_capacity_,kQHeads,kKVHeads,kHeadDim,
                    exact_attention_query_pair_,exact_attention_gqa_pair_,exact_attention_q_shared_);
                const int score_shards=exact_attention_gqa_six_extent_shards_&&!capture_active_?
                    std::min(6,std::max(1,(position+first+count+255)/256)):6;
                if(exact_prefix_rows_>0 || exact_page_ranges_.count>0 ||
                    (exact_attention_query_pair_ && segmented_route!=Exl3SegmentedAttentionRoute::inherited)) {
                    for(int page=0;page<exact_page_ranges_.count;++page) {
                        const auto& range=exact_page_ranges_.ranges[page];
                        if(range.first>position || range.rows>position-range.first)
                            throw std::invalid_argument("stale segmented page range");
                    }
                    if(segmented_route==Exl3SegmentedAttentionRoute::inherited || exact_prefix_first_>position || exact_prefix_rows_>position-exact_prefix_first_)
                        throw std::invalid_argument("stale segmented exact prefix");
                    const bool segmented_selected_six=
                        exact_attention_gqa_six_scores_ &&
                        exact_attention_gqa_six_softmax_triple_values_;
                    if(segmented_selected_six) {
                        auto pages=exact_page_ranges_;
                        if(exact_prefix_rows_)pages.append(exact_prefix_k_,exact_prefix_v_,
                            exact_prefix_first_,exact_prefix_rows_,position);
                        launch_attention_cached_gqa_six_segmented_scores(
                            qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            attention_k,exact_scores_,count,position,cache_capacity_,
                            position_device_,first,score_shards,pages,stream);
                        attention_cached_gqa_six_softmax_kernel<<<
                            count*kKVHeads,256,0,stream>>>(exact_scores_,count,
                                position,cache_capacity_,position_device_,first);
                        ++gqa_six_softmax_triple_value_launch_attempts_;
                        gqa_six_softmax_triple_value_row_attempts_+=
                            static_cast<std::uint64_t>(count);
                        attention_cached_gqa_triple_normalized_values_kernel<true><<<
                            count*(kQHeads/3),256,0,stream>>>(attention_v,
                                attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                exact_scores_,count,position,cache_capacity_,
                                position_device_,first,pages);
                    } else if(segmented_route==Exl3SegmentedAttentionRoute::query_pair) {
                        auto pages=exact_page_ranges_;
                        if(exact_prefix_rows_)pages.append(exact_prefix_k_,exact_prefix_v_,exact_prefix_first_,exact_prefix_rows_,position);
                        attention_query_pair_kernel<<<((count+1)/2)*kQHeads,256,0,stream>>>(
                            qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,attention_v,
                            attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,
                            cache_capacity_,position_device_,first,pages);
                        ++query_pair_launch_attempts_;
                        query_pair_row_attempts_+=static_cast<std::uint64_t>(count);
                    } else if(segmented_route==Exl3SegmentedAttentionRoute::gqa_pair) {
                        auto pages=exact_page_ranges_;
                        if(exact_prefix_rows_)pages.append(exact_prefix_k_,exact_prefix_v_,exact_prefix_first_,exact_prefix_rows_,position);
                        attention_cached_gqa_pair_scores_kernel<true><<<count*kQHeads,256,0,stream>>>(
                            qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,exact_scores_,count,position,
                            cache_capacity_,position_device_,first,pages);
                        attention_cached_gqa_pair_values_kernel<true><<<count*(kQHeads/2),256,0,stream>>>(
                            attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,
                            cache_capacity_,position_device_,first,pages);
                    } else if(segmented_route==Exl3SegmentedAttentionRoute::q_shared) {
                        auto pages=exact_page_ranges_;
                        if(exact_prefix_rows_)pages.append(exact_prefix_k_,exact_prefix_v_,exact_prefix_first_,exact_prefix_rows_,position);
                        attention_cached_parallel_q_shared_kernel<false,false,true><<<count*kQHeads,256,0,stream>>>(
                            qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,attention_v,
                            attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,
                            cache_capacity_,position_device_,first,pages);
                    } else attention_cached_parallel_kernel<true><<<count*kQHeads,256,0,stream>>>(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,attention_v,
                        attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,
                        cache_capacity_,position_device_,first,exact_prefix_k_,exact_prefix_v_,exact_prefix_rows_,exact_prefix_first_,exact_page_ranges_);
                } else if(exact_attention_gqa_six_) {
                    launch_attention_cached_gqa_six_scores(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,
                        exact_scores_,count,position,cache_capacity_,position_device_,first,6,stream);
                    attention_cached_gqa_six_values_kernel<<<count*kKVHeads,256,0,stream>>>(
                        attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                } else if(exact_attention_gqa_six_values_sharded_) {
                    launch_attention_cached_gqa_six_scores(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,
                        exact_scores_,count,position,cache_capacity_,position_device_,first,6,stream);
                    attention_cached_gqa_six_softmax_kernel<<<count*kKVHeads,256,0,stream>>>(
                        exact_scores_,count,position,cache_capacity_,position_device_,first);
                    attention_cached_gqa_six_values_sharded_kernel<<<count*kKVHeads*2,64,0,stream>>>(
                        attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                } else if(exact_attention_gqa_six_scores_) {
                    const bool query_pair_scores=
                        exact_attention_gqa_six_query_pair_scores_ && !capture_active_ &&
                        count>1;
                    const bool score_k_tile64=
                        exact_attention_gqa_six_score_k_tile64_ &&
                        exact_attention_gqa_six_softmax_triple_values_;
                    const bool decode_fused=
                        exact_attention_gqa_six_decode_fused_ &&
                        rows==1 && position>0 && !profile &&
                        (!capture_active_ || ordinary_full_layer_graph_capture_) &&
                        !direct_staged_rows_ &&
                        !exact_prefix_rows_ && !exact_page_ranges_.count &&
                        exact_attention_gqa_six_softmax_triple_values_ &&
                        !exact_attention_gqa_six_ &&
                        !exact_attention_gqa_six_values_sharded_ &&
                        !exact_attention_gqa_six_packed_triples_ &&
                        !exact_attention_gqa_six_query_pair_scores_ &&
                        !exact_attention_gqa_six_score_k_tile64_ &&
                        !exact_attention_gqa_six_softmax_triple_values_v_tile_ &&
                        !exact_attention_gqa_six_softmax_triple_values_v_tile64_ &&
                        !exact_attention_gqa_six_softmax_triple_values_full_cta_ &&
                        !exact_attention_gqa_six_softmax_triple_values_threads128_ &&
                        !exact_attention_gqa_six_softmax_triple_values_score_tile_ &&
                        !exact_attention_gqa_six_softmax_triple_values_key_pair_pipeline_ &&
                        !exact_attention_gqa_six_softmax_triple_values_warp_score_broadcast_ &&
                        !exact_attention_gqa_six_softmax_triple_values_pair_dimensions_ &&
                        !exact_attention_gqa_six_softmax_six_values_single_load_ &&
                        !exact_attention_gqa_six_softmax_six_values_scalar_single_load_ &&
                        !exact_attention_gqa_six_softmax_triple_values_scalar_dim_ &&
                        !exact_attention_gqa_six_softmax_triple_values_two_query_ &&
                        !exact_attention_gqa_six_softmax_triple_values_fused_ &&
                        !exact_attention_gqa_six_softmax_fused_scalar_values_;
                    if(!decode_fused && score_k_tile64) {
                        launch_attention_cached_gqa_six_scores_k_tile64(
                            qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            attention_k,exact_scores_,count,position,cache_capacity_,
                            position_device_,first,score_shards,stream);
                        ++gqa_six_score_k_tile64_launch_attempts_;
                        gqa_six_score_k_tile64_row_attempts_+=
                            static_cast<std::uint64_t>(count);
                    } else if(!decode_fused && query_pair_scores) {
                        launch_attention_cached_gqa_six_query_pair_scores(
                            qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,
                            exact_scores_,count,position,cache_capacity_,position_device_,first,
                            score_shards,stream);
                        ++gqa_six_query_pair_score_launch_attempts_;
                        gqa_six_query_pair_score_row_attempts_+=
                            static_cast<std::uint64_t>(count);
                    } else if(!decode_fused) launch_attention_cached_gqa_six_scores(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,
                        exact_scores_,count,position,cache_capacity_,position_device_,first,
                        score_shards,stream);
                    if(decode_fused) {
                        if (ordinary_full_layer_graph_capture_ &&
                            std::getenv("NINFER_EXL3_ORDINARY_FULL_LAYER_GRAPH_DIAGNOSTIC"))
                            std::fprintf(stderr,
                                "ordinary_full_layer_graph_decode_fused_branch_entered\\n");
                        attention_cached_gqa_six_decode_fused_kernel<<<
                            rows*kKVHeads,256,0,stream>>>(
                                qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                attention_k,attention_v,
                                attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                exact_scores_,count,position,cache_capacity_,
                                position_device_,first);
                        ++exact_attention_gqa_six_decode_fused_calls_;
                    } else if(exact_attention_gqa_six_packed_triples_) {
                        attention_cached_gqa_six_packed_triples_values_kernel<<<
                            count*kKVHeads,256,0,stream>>>(attention_v,
                            attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                            exact_scores_,count,position,cache_capacity_,position_device_,first);
                    } else if(exact_attention_gqa_six_softmax_triple_values_ ||
                              exact_attention_gqa_six_softmax_fused_scalar_values_) {
                        ++gqa_six_softmax_triple_value_launch_attempts_;
                        gqa_six_softmax_triple_value_row_attempts_+=
                            static_cast<std::uint64_t>(count);
                        if(exact_attention_gqa_six_softmax_fused_scalar_values_) {
                            ++gqa_six_softmax_fused_scalar_values_launch_attempts_;
                            gqa_six_softmax_fused_scalar_values_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_six_softmax_fused_scalar_values_kernel<<<
                                count*kKVHeads,256,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(exact_attention_gqa_six_softmax_triple_values_fused_) {
                            ++gqa_six_softmax_triple_fused_launch_attempts_;
                            gqa_six_softmax_triple_fused_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_six_softmax_triple_values_fused_kernel<<<
                                count*kKVHeads,256,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else {
                            if(exact_attention_gqa_six_softmax_tile512_) {
                                ++gqa_six_softmax_tile512_launch_attempts_;
                                gqa_six_softmax_tile512_row_attempts_+=
                                    static_cast<std::uint64_t>(count);
                                attention_cached_gqa_six_softmax_tile512_kernel<<<
                                    count*kKVHeads,256,0,stream>>>(exact_scores_,count,
                                        position,cache_capacity_,position_device_,first);
                            } else attention_cached_gqa_six_softmax_kernel<<<
                                    count*kKVHeads,256,0,stream>>>(exact_scores_,count,
                                        position,cache_capacity_,position_device_,first);
                        if(exact_attention_gqa_six_softmax_triple_values_v_tile64_) {
                            ++gqa_six_softmax_triple_v_tile_launch_attempts_;
                            gqa_six_softmax_triple_v_tile_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_v_tile_kernel<64><<<
                                count*kKVHeads,256,
                                gqa_six_softmax_triple_v_tile_shared_bytes(64),stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(exact_attention_gqa_six_softmax_triple_values_v_tile_) {
                            ++gqa_six_softmax_triple_v_tile_launch_attempts_;
                            gqa_six_softmax_triple_v_tile_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_v_tile_kernel<16><<<
                                count*kKVHeads,256,
                                gqa_six_softmax_triple_v_tile_shared_bytes(),stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(exact_attention_gqa_six_softmax_triple_values_full_cta_) {
                            ++gqa_six_softmax_triple_full_cta_launch_attempts_;
                            gqa_six_softmax_triple_full_cta_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_full_cta_kernel<<<
                                count*kKVHeads,256,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(
                            exact_attention_gqa_six_softmax_triple_values_pair_dimensions_) {
                            ++gqa_six_softmax_triple_pair_dimensions_launch_attempts_;
                            gqa_six_softmax_triple_pair_dimensions_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_kernel<false,true><<<
                                count*(kQHeads/3),64,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(
                            exact_attention_gqa_six_softmax_six_values_single_load_) {
                            ++gqa_six_softmax_six_values_single_load_launch_attempts_;
                            gqa_six_softmax_six_values_single_load_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_six_normalized_values_single_load_kernel<<<
                                count*kKVHeads,128,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(
                            exact_attention_gqa_six_softmax_six_values_scalar_single_load_ &&
                            count<=8) {
                            ++gqa_six_softmax_six_values_single_load_launch_attempts_;
                            gqa_six_softmax_six_values_single_load_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_six_normalized_values_scalar_single_load_kernel<<<
                                count*kKVHeads,256,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(exact_attention_gqa_six_softmax_triple_values_scalar_dim_) {
                            ++gqa_six_softmax_triple_scalar_dim_launch_attempts_;
                            gqa_six_softmax_triple_scalar_dim_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_scalar_dim_kernel<<<
                                count*(kQHeads/3),256,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(exact_attention_gqa_six_softmax_triple_values_score_tile_) {
                            ++gqa_six_softmax_triple_score_tile_launch_attempts_;
                            gqa_six_softmax_triple_score_tile_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_score_tile_kernel<<<
                                count*(kQHeads/3),128,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(
                            exact_attention_gqa_six_softmax_triple_values_key_pair_pipeline_) {
                            ++gqa_six_softmax_triple_key_pair_launch_attempts_;
                            gqa_six_softmax_triple_key_pair_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_kernel<false,false,false,true><<<
                                count*(kQHeads/3),256,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(
                            exact_attention_gqa_six_softmax_triple_values_warp_score_broadcast_) {
                            ++gqa_six_softmax_triple_warp_score_broadcast_launch_attempts_;
                            gqa_six_softmax_triple_warp_score_broadcast_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_kernel<false,false,true><<<
                                count*(kQHeads/3),256,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else if(exact_attention_gqa_six_softmax_triple_values_two_query_ &&
                                  count>1) {
                            ++gqa_six_softmax_triple_two_query_launch_attempts_;
                            gqa_six_softmax_triple_two_query_row_attempts_+=
                                static_cast<std::uint64_t>(count);
                            attention_cached_gqa_triple_normalized_values_two_query_kernel<<<
                                ((count+1)/2)*(kQHeads/3),128,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        } else {
                            const int value_threads=
                                exact_attention_gqa_six_softmax_triple_values_threads128_?128:256;
                            if(exact_attention_gqa_six_softmax_triple_values_threads128_) {
                                ++gqa_six_softmax_triple_threads128_launch_attempts_;
                                gqa_six_softmax_triple_threads128_row_attempts_+=
                                    static_cast<std::uint64_t>(count);
                            }
                            attention_cached_gqa_triple_normalized_values_kernel<false><<<
                                count*(kQHeads/3),value_threads,0,stream>>>(attention_v,
                                    attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,
                                    exact_scores_,count,position,cache_capacity_,position_device_,first);
                        }
                        }
                    } else if(exact_attention_gqa_triple_values4_)
                        attention_cached_gqa_triple_values_kernel<true,true><<<count*(kQHeads/3),256,0,stream>>>(attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                    else
                        attention_cached_gqa_triple_values_kernel<true><<<count*(kQHeads/3),256,0,stream>>>(attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                } else if(exact_attention_gqa_triple_) {
                    attention_cached_gqa_triple_scores_kernel<<<count*kQHeads,256,0,stream>>>(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,exact_scores_,count,position,cache_capacity_,position_device_,first);
                    const int value_threads=exact_attention_gqa_triple_values128_?128:256;
                    if(exact_attention_gqa_triple_softmax_staged_&&exact_attention_gqa_triple_values4_) attention_cached_gqa_triple_values_kernel<true,true><<<count*(kQHeads/3),value_threads,0,stream>>>(attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                    else if(exact_attention_gqa_triple_softmax_staged_) attention_cached_gqa_triple_values_kernel<true><<<count*(kQHeads/3),value_threads,0,stream>>>(attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                    else attention_cached_gqa_triple_values_kernel<false><<<count*(kQHeads/3),value_threads,0,stream>>>(attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                } else if(exact_attention_gqa_pair_) {
                    attention_cached_gqa_pair_scores_kernel<false><<<count*kQHeads,256,0,stream>>>(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,exact_scores_,count,position,cache_capacity_,position_device_,first);
                    attention_cached_gqa_pair_values_kernel<false><<<count*(kQHeads/2),256,0,stream>>>(
                        attention_v,attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                } else if(exact_attention_v_half2_)
                    attention_cached_parallel_q_shared_kernel<true,true><<<count*kQHeads,256,0,stream>>>(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,attention_v,
                        attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                else if(exact_attention_k_half2_)
                    attention_cached_parallel_q_shared_kernel<true,false><<<count*kQHeads,256,0,stream>>>(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,attention_v,
                        attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                else if(exact_attention_q_shared_)
                    attention_cached_parallel_q_shared_kernel<false,false><<<count*kQHeads,256,0,stream>>>(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,attention_v,
                        attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
                else
                    attention_cached_parallel_kernel<false><<<count*kQHeads,256,0,stream>>>(
                        qr+static_cast<std::size_t>(first)*kQHeads*kHeadDim,attention_k,attention_v,
                        attn+static_cast<std::size_t>(first)*kQHeads*kHeadDim,exact_scores_,count,position,cache_capacity_,position_device_,first);
            }
        } else {
            // Eager sizes shared by actual need; graph capture freezes launch
            // parameters, so captured launches keep full capacity.
            const std::size_t scores_bytes = capture_active_
                ? static_cast<std::size_t>(cache_capacity_) * sizeof(float)
                : static_cast<std::size_t>(position + rows) * sizeof(float);
            attention_cached_kernel<<<rows * kQHeads, 256, scores_bytes, stream>>>(
                qr, attention_k, attention_v, attn, rows, position, cache_capacity_, position_device_);
        }
    } else {
        attention_kernel<<<rows * kQHeads, 256, 0, stream>>>(qr, kr, vp, attn, rows);
    }
    launch(cudaGetLastError(), "launch EXL3 causal GQA");
    exl3_launch_small(sigmoid_mul_kernel,dim3((rows * kQHeads * kHeadDim + 255) / 256),dim3(256),0,stream,gp, attn,
        rows * kQHeads * kHeadDim);
    launch(cudaGetLastError(), "launch EXL3 attention output gate"); end(6);
    if(attention_core_sample) {
        cuda_check(cudaEventRecord(attention_core_end,stream),
            "record attention core sample end");
        cuda_check(cudaEventSynchronize(attention_core_end),
            "resolve attention core sample");
        float attention_core_ms=0.0f;
        cuda_check(cudaEventElapsedTime(&attention_core_ms,
            attention_core_start,attention_core_end),
            "read attention core sample");
        std::fprintf(stderr,
            "ATTENTION_CORE_SAMPLE layer=%d rows=%d position=%d ms=%.6f\n",
            model_layer_,rows,position,attention_core_ms);
        cudaEventDestroy(attention_core_start);
        cudaEventDestroy(attention_core_end);
    }

    begin(7);
    const bool shared_o=can_share_target && target_o_executor_enabled_ &&
        target_shared_admission(Exl3TargetSharedFamily::o,weights_.o_metadata).has_value() &&
        target_q_executor_(Exl3TargetQContinuation{weights_.o,weights_.o_metadata,
            attn,op,rows,position,model_layer_,stream,Exl3TargetSharedFamily::o});
    if(!shared_o)project(linear_workspaces_[3], weights_.o, weights_.o_metadata, attn, op,
            Exl3TargetProjectionOperator::o);
    launch(cudaGetLastError(), "launch EXL3 output projection"); end(7);

    begin(8);
    exl3_launch_small(residual_kernel,dim3((rows * kHidden + 255) / 256),dim3(256),0,stream,input, op, post_resid,
        rows * kHidden);
    launch(cudaGetLastError(), "launch EXL3 attention residual");
    if(mlp_tail_graph) {
        if(profile || (rows>1 && !preserve_m1_topology) || wide_prefill ||
           rows<1 || rows>8 ||
           oscar_ || capture_active_ || projection_timing_ || projection_observer_ ||
           eager_mlp_gateup_concurrency_.complete() ||
           small_m_fused_gate_up_transform_ || target_q_executor_)
            throw std::invalid_argument(
                "HostKV MLP-tail graph replay compatibility");
        end(8);
        mlp_tail_graph->launch(stream);
    } else {
    exl3_launch_small(rms_norm_kernel,dim3(rows),dim3(512),512 * sizeof(float),stream,post_resid, weights_.post_attention_norm,
        mlp_in, rows, kHidden);
    launch(cudaGetLastError(), "launch EXL3 attention residual and norm"); end(8);

    // Fused prefill MLP (quantized route): gate/up stay untransformed and the
    // quantized down input is produced in one pass.
    const bool fused_prefill_mlp = wide_prefill && reconstruct_gemm_ &&
        !capture_active_ && !oscar_ && !profile && !projection_timing_ &&
        !projection_observer_ && !can_share_target && rows >= 256 &&
        reconstruct_gemm_->forward_numeric_mlp(weights_.gate, weights_.gate_metadata,
            weights_.up, weights_.up_metadata, weights_.down, weights_.down_metadata,
            mlp_in, gp, up, act, down, rows, stream);
    if (fused_prefill_mlp) {
        launch(cudaGetLastError(), "launch fused EXL3 prefill MLP");
        exl3_launch_small(residual_kernel,dim3((rows * kHidden + 255) / 256),dim3(256),0,stream,
            post_resid, down, output, rows * kHidden);
        launch(cudaGetLastError(), "launch EXL3 final residual");
    } else {
    const bool concurrent_mlp_gateup =
        eager_mlp_gateup_concurrency_.complete() && !capture_active_ && !oscar_ &&
        !profile && !projection_timing_ && !projection_observer_ &&
        (preserve_m1_topology || rows == 1) && !wide_prefill &&
        rows >= 1 && rows <= 8 &&
        !(can_share_target && target_gateup_executor_enabled_);
    bool merged_gate_up = false;
    if (concurrent_mlp_gateup) {
        launch(cudaEventRecord(eager_mlp_gateup_concurrency_.fork, stream),
               "record eager full-attention MLP gate/up fork");
        launch(cudaStreamWaitEvent(eager_mlp_gateup_concurrency_.up_stream,
                                   eager_mlp_gateup_concurrency_.fork, 0),
               "fork eager full-attention up projection stream");
        try {
            project_on(eager_mlp_gateup_concurrency_.up_workspace, weights_.up,
                       weights_.up_metadata, mlp_in, up,
                       Exl3TargetProjectionOperator::up,
                       eager_mlp_gateup_concurrency_.up_stream);
            launch(cudaGetLastError(), "launch concurrent EXL3 up projection");
            launch(cudaEventRecord(eager_mlp_gateup_concurrency_.up_done,
                                   eager_mlp_gateup_concurrency_.up_stream),
                   "record eager full-attention up projection completion");
            begin(9);
            project(linear_workspaces_[4], weights_.gate, weights_.gate_metadata,
                    mlp_in, gp, Exl3TargetProjectionOperator::gate);
            launch(cudaGetLastError(), "launch EXL3 gate projection");
            launch(cudaStreamWaitEvent(stream,
                                       eager_mlp_gateup_concurrency_.up_done, 0),
                   "join eager full-attention up projection stream");
            end(9);
            ++eager_mlp_gateup_concurrent_calls_;
        } catch (...) {
            (void)cudaEventRecord(eager_mlp_gateup_concurrency_.up_done,
                                  eager_mlp_gateup_concurrency_.up_stream);
            (void)cudaStreamWaitEvent(stream,
                                      eager_mlp_gateup_concurrency_.up_done, 0);
            throw;
        }
    } else {
        begin(9);
        std::optional<Exl3ActivationLifetime::Scope> activation_scope;
        if(can_share_target && target_gateup_executor_enabled_)
            activation_scope.emplace(mlp_activation_lifetime_);
        const auto activation_witness=activation_scope
            ? activation_scope->witness():Exl3ActivationLifetime::Witness{};
        const bool paired_m1_gate_up=fast_same_weights_fp16kv_m1_gate_up_pair_ &&
            !can_share_target && !capture_active_ && !oscar_ && !profile &&
            !projection_timing_ && !projection_observer_ && !wide_prefill &&
            !preserve_m1_topology && rows==1 && linear_workspaces_[4] &&
            linear_workspaces_[5] &&
            (weights_.gate_metadata.K==5 || weights_.gate_metadata.K==6 ||
             weights_.gate_metadata.K==7) &&
            weights_.up_metadata.K==weights_.gate_metadata.K;
        if(paired_m1_gate_up) {
            linear_workspaces_[4]->forward_target_m1_gate_up_pair_for_test(
                *linear_workspaces_[5],weights_.gate,weights_.gate_metadata,
                weights_.up,weights_.up_metadata,mlp_in,gp,up,stream);
            ++fast_same_weights_fp16kv_m1_gate_up_pair_submissions_;
        }
        merged_gate_up=!paired_m1_gate_up && rows==1 && !can_share_target &&
            !oscar_ && !profile && !projection_timing_ && !projection_observer_ &&
            !wide_prefill && !small_m_fused_gate_up_transform_ &&
            linear_workspaces_[4]->forward_m1_gate_up_silu(*linear_workspaces_[5],
                weights_.gate,weights_.gate_metadata,weights_.up,weights_.up_metadata,
                mlp_in,gp,up,act,stream);
        const bool paired_gate_up=!paired_m1_gate_up && !merged_gate_up && !can_share_target &&
            !profile && !projection_timing_ &&
            !projection_observer_ && linear_workspaces_[4]->forward_target_prefill_gate_up_pair(
                *linear_workspaces_[5],weights_.gate,weights_.gate_metadata,
                weights_.up,weights_.up_metadata,mlp_in,gp,up,rows,stream);
        const bool shared_gate=paired_m1_gate_up || merged_gate_up || paired_gate_up ||
            (can_share_target && target_gateup_executor_enabled_ &&
            target_shared_admission(Exl3TargetSharedFamily::gate,weights_.gate_metadata).has_value() &&
            target_q_executor_(Exl3TargetQContinuation{weights_.gate,weights_.gate_metadata,
                mlp_in,gp,rows,position,model_layer_,stream,Exl3TargetSharedFamily::gate,nullptr,activation_witness}));
        if(!shared_gate)project(linear_workspaces_[4], weights_.gate, weights_.gate_metadata, mlp_in, gp,
                Exl3TargetProjectionOperator::gate);
        launch(cudaGetLastError(), "launch EXL3 gate projection"); end(9);
        begin(10);
        const bool shared_up=paired_m1_gate_up || merged_gate_up || paired_gate_up ||
            (can_share_target && target_gateup_executor_enabled_ &&
            target_shared_admission(Exl3TargetSharedFamily::up,weights_.up_metadata).has_value() &&
            target_q_executor_(Exl3TargetQContinuation{weights_.up,weights_.up_metadata,
                mlp_in,up,rows,position,model_layer_,stream,Exl3TargetSharedFamily::up,nullptr,activation_witness}));
        if(!shared_up)project(linear_workspaces_[5], weights_.up, weights_.up_metadata, mlp_in, up,
                Exl3TargetProjectionOperator::up);
        launch(cudaGetLastError(), "launch EXL3 up projection"); end(10);
        activation_scope.reset(); // no cross-SiLU/residual gather reuse
    }
    const bool fused_gate_up=small_m_fused_gate_up_transform_ &&
        !capture_active_ && !oscar_ && !profile && !projection_timing_ &&
        !projection_observer_ && !wide_prefill && rows>=1 && rows<=8 &&
        !(can_share_target && target_down_executor_enabled_) &&
        !weights_.down_metadata.mcg && weights_.down_metadata.mul1 &&
        !weights_.down_metadata.has_bias;
    begin(11);
    if(merged_gate_up) {
    } else if(fused_gate_up) {
        linear_workspaces_[6]->transform_gate_up(
            weights_.down,weights_.down_metadata,gp,up,act,rows,stream,
            rows==1 ? Exl3CudaLinearAdmission::ordinary :
                Exl3CudaLinearAdmission::target_continuation_down);
        ++fused_gate_up_submissions_;
    } else {
        exl3_launch_small(silu_mul_kernel,dim3((rows * kIntermediate + 255) / 256),dim3(256),0,stream,
            gp,up,act,rows*kIntermediate);
        launch(cudaGetLastError(), "launch EXL3 SiLU gate");
    }
    end(11);
    begin(12);
    const bool shared_down=can_share_target && target_down_executor_enabled_ &&
        target_shared_admission(Exl3TargetSharedFamily::down,weights_.down_metadata).has_value() &&
        target_q_executor_(Exl3TargetQContinuation{weights_.down,weights_.down_metadata,
            act,down,rows,position,model_layer_,stream,Exl3TargetSharedFamily::down});
    if(!shared_down)project(linear_workspaces_[6], weights_.down, weights_.down_metadata, act, down,
            Exl3TargetProjectionOperator::down,
            fused_gate_up?linear_workspaces_[6]->transformed_device():nullptr);
    launch(cudaGetLastError(), "launch EXL3 down projection");
    exl3_launch_small(residual_kernel,dim3((rows * kHidden + 255) / 256),dim3(256),0,stream,post_resid, down, output,
        rows * kHidden);
    launch(cudaGetLastError(), "launch EXL3 final residual"); end(12);
    }
    }

    if (profile) {
        cudaEvent_t total_end{};
        cuda_check(cudaEventCreate(&total_end), "create EXL3 total timing event");
        launch_profile_event(total_end, stream);
        cuda_check(cudaEventSynchronize(total_end), "synchronize EXL3 total timing event");
        float total_milliseconds = 0.0f;
        cuda_check(cudaEventElapsedTime(&total_milliseconds, starts[0], total_end),
                   "read EXL3 total layer timing");
        for (int i = 0; i < 13; ++i) {
            float milliseconds = 0.0f;
            cuda_check(cudaEventElapsedTime(&milliseconds, starts[i], ends[i]),
                       "read EXL3 layer timing");
            timings_.microseconds[i] = static_cast<double>(milliseconds) * 1000.0;
            cudaEventDestroy(starts[i]);
            cudaEventDestroy(ends[i]);
        }
        timings_.total_microseconds = static_cast<double>(total_milliseconds) * 1000.0;
        cudaEventDestroy(total_end);
    }

    trace_ = {
        input, input_norm, qg, kp, vp, qn, kn, qr, kr, attn, op, post_resid,
        mlp_in, gp, up, act, down, output};
    if (eligible_retained_prefix) {
        retained_prefix_available_ = true;
        retained_prefix_rows_ = rows;
        retained_prefix_position_ = position;
        retained_prefix_stream_ = stream;
    }
}

} // namespace ninfer::exl3
