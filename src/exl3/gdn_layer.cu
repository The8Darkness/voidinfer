#include "exl3/pdl_small.cuh"
#include "exl3/gaming_operator_fixture.h"
#include "exl3/gdn_layer.h"
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/linear_workspace_requirements.h"
#include "exl3/paired_transform_extent.h"
#include "exl3/residual_norm_extent.h"
#include "exl3/block_tree_sum.cuh"
#include "exl3/gdn_chunked_recurrence.cuh"

#include "core/arena.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace ninfer::exl3 {
namespace {

// Prefill numerical policy (default 1, quality-gated): wide-prefill GDN
// recurrences of >= 64 rows use the chunk-parallel WY form (FP32 state,
// TF32/FP16 tensor-core products) instead of the sequential resident
// recurrence. 0 = sequential control.
bool gdn_chunked_prefill_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_EXL3_GDN_CHUNKED_PREFILL");
        if (!value || std::strcmp(value, "1") == 0) return true;
        if (std::strcmp(value, "0") == 0) return false;
        throw std::invalid_argument("NINFER_EXL3_GDN_CHUNKED_PREFILL must be 0 or 1");
    }();
    return enabled;
}

// One process-wide chunk workspace: GDN layers execute their prefill
// recurrences sequentially on the owning stream, at most 1024 rows per call.
gdn_chunked::Workspace gdn_chunked_workspace() {
    static void* base = [] {
        void* pointer = nullptr;
        if (cudaMalloc(&pointer, gdn_chunked::workspace_bytes(1024)) != cudaSuccess)
            throw std::runtime_error("allocate chunked GDN prefill workspace");
        gdn_chunked::configure();
        return pointer;
    }();
    return gdn_chunked::carve(base, 1024);
}

constexpr int kHidden = 5120;
constexpr int kQkv = 10240;
constexpr int kZ = 6144;
constexpr int kHeads = 48;
constexpr int kKeyHeads = 16;
constexpr int kHeadDim = 128;
constexpr int kIntermediate = 17408;
constexpr int kConvStorage = 4;
constexpr float kRmsEps = 1.0e-6f;
constexpr std::array<int,22> kHalfBufferFeatures={{kHidden,kQkv,kZ,kQkv,2048,2048,6144,kQkv,
    kZ,kZ,kZ,kZ,kZ,kHidden,kHidden,kHidden,kIntermediate,kIntermediate,
    kIntermediate,kHidden,kHidden,kZ}};
constexpr std::array<std::size_t,5> kSplitHalfIndices{{3,4,5,6,9}};
constexpr int kPrivateRetentionRows = 16;
constexpr std::size_t kSplitHalfFeatures =
    static_cast<std::size_t>(kHalfBufferFeatures[3]) +
    static_cast<std::size_t>(kHalfBufferFeatures[4]) +
    static_cast<std::size_t>(kHalfBufferFeatures[5]) +
    static_cast<std::size_t>(kHalfBufferFeatures[6]) +
    static_cast<std::size_t>(kHalfBufferFeatures[9]);
constexpr bool transient_half(std::size_t i) {
    return i!=3 && i!=4 && i!=5 && i!=6 && i!=9;
}

bool is_split_half(std::size_t i) {
    return std::find(kSplitHalfIndices.begin(), kSplitHalfIndices.end(), i) !=
        kSplitHalfIndices.end();
}

std::size_t split_wide_bytes(int rows) {
    return static_cast<std::size_t>(rows) * kSplitHalfFeatures * sizeof(std::uint16_t);
}

void check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

__device__ __forceinline__ float half_value(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float bf16_value(std::uint16_t bits) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}

__device__ __forceinline__ float silu_f32(float x) { return x / (1.0f + expf(-x)); }

template<bool Residual=false>
__global__ void rms_norm_f16_kernel(const std::uint16_t* input, const std::uint16_t* weight,
                                    std::uint16_t* output, int rows, int features,
                                    const std::uint16_t* right=nullptr,std::uint16_t* materialized=nullptr) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= features) return;
    extern __shared__ float shared[];
    constexpr int kCached = 10;
    const bool cached = features == kCached * static_cast<int>(blockDim.x);
    float values[kCached];
    std::uint16_t weight_bits[kCached];
    float sum = 0.0f;
    if (cached) {
        #pragma unroll
        for (int j = 0; j < kCached; ++j)
            weight_bits[j] = weight[lane + j * static_cast<int>(blockDim.x)];
        // Same elements, same x*x accumulation order; loads issued together and
        // the represented inputs kept in registers for the output pass.
        std::uint16_t represented[kCached];
        #pragma unroll
        for (int j = 0; j < kCached; ++j)
            represented[j] = input[row * features + lane + j * static_cast<int>(blockDim.x)];
        if constexpr(Residual) {
            std::uint16_t right_bits[kCached];
            #pragma unroll
            for (int j = 0; j < kCached; ++j)
                right_bits[j] = right[row * features + lane + j * static_cast<int>(blockDim.x)];
            #pragma unroll
            for (int j = 0; j < kCached; ++j) {
                represented[j] = __half_as_ushort(__float2half_rn(
                    half_value(represented[j]) + half_value(right_bits[j])));
                materialized[row * features + lane + j * static_cast<int>(blockDim.x)] =
                    represented[j];
            }
        }
        #pragma unroll
        for (int j = 0; j < kCached; ++j) {
            values[j] = half_value(represented[j]);
            sum += values[j] * values[j];
        }
    } else
    for (int i = lane; i < features; i += blockDim.x) {
        const int index=row * features + i;
        std::uint16_t represented=input[index];
        if constexpr(Residual) {
            represented=__half_as_ushort(__float2half_rn(half_value(represented)+half_value(right[index])));
            materialized[index]=represented;
        }
        const float x = half_value(represented);
        sum += x * x;
    }
    float total;
    if (blockDim.x == 512) {
        total = block_tree_sum_exact<512>(sum, shared, lane);
    } else {
    shared[lane] = sum;
    __syncthreads();
    // Cross-warp strides in shared memory; strides 16..1 are the same pairwise
    // adds done with shfl_down in warp 0.
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (lane < stride) shared[lane] += shared[lane + stride];
        __syncthreads();
    }
    if (lane < 32) {
        float value = shared[lane];
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    total = shared[0];
    }
    const float inv = rsqrtf(total / static_cast<float>(features) + kRmsEps);
    if (cached) {
        #pragma unroll
        for (int j = 0; j < kCached; ++j) {
            const int i = lane + j * static_cast<int>(blockDim.x);
            const float x = values[j] * inv;
            const float w = half_value(weight_bits[j]);
            output[row * features + i] = __half_as_ushort(__float2half_rn(x * (w + 1.0f)));
        }
        return;
    }
    for (int i = lane; i < features; i += blockDim.x) {
        const float x = half_value((Residual?materialized:input)[row * features + i]) * inv;
        const float w = half_value(weight[i]);
        output[row * features + i] = __half_as_ushort(__float2half_rn(x * (w + 1.0f)));
    }
}

// NINFER_EXL3_RESIDUAL_NORM_FUSED (default 1): the post-attention residual
// add and RMS norm run as rms_norm_f16_kernel<true> (identical half(a+b)
// residual, materialized, then the same norm); 0 restores two launches.
bool residual_norm_fused_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_RESIDUAL_NORM_FUSED");
        if(!value||std::strcmp(value,"1")==0) return true;
        if(std::strcmp(value,"0")==0) return false;
        throw std::invalid_argument("NINFER_EXL3_RESIDUAL_NORM_FUSED must be 0 or 1");
    }();
    return enabled;
}

__global__ void transpose_f16_to_bf16_kernel(const std::uint16_t* input, std::uint16_t* output,
                                             int rows, int features) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * features;
    if (index >= total) return;
    const int row = index / features;
    const int feature = index % features;
    reinterpret_cast<__nv_bfloat16*>(output)[feature * rows + row] =
        __float2bfloat16_rn(half_value(input[index]));
}

__global__ void convert_f16_to_bf16_kernel(const std::uint16_t* input, std::uint16_t* output, int count) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    reinterpret_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16_rn(half_value(input[index]));
}

__global__ void transpose_bf16_to_f16_kernel(const std::uint16_t* input, std::uint16_t* output,
                                             int rows, int features) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * features;
    if (index >= total) return;
    output[index] = __half_as_ushort(__float2half_rn(
        __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(input)[index])));
}

__global__ void gopt_gdn_output_pack_kernel(const std::uint16_t* core,
    const std::uint16_t* norm,std::uint16_t* trace,std::uint16_t* projection,int count) {
    const int i=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    trace[i]=core[i];
    projection[i]=__half_as_ushort(__float2half_rn(bf16_value(norm[i])));
}

__global__ void transpose_f32_to_row_major_kernel(const float* input, float* output, int rows, int features) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * features) return;
    const int row = index / features;
    const int feature = index % features;
    output[row * features + feature] = input[feature * rows + row];
}

__global__ void control_kernel(const float* b, const float* a, const float* a_log,
                               const float* dt_bias, float* beta, float* g, int rows) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kHeads;
    if (index >= total) return;
    const int row = index / kHeads;
    const int head = index % kHeads;
    const float bv = b[index];
    const float av = a[index] + dt_bias[head];
    const float beta_f = 1.0f / (1.0f + expf(-bv));
    const float softplus = av > 20.0f ? av : log1pf(expf(av));
    beta[head * rows + row] = __bfloat162float(__float2bfloat16_rn(beta_f));
    g[head * rows + row] = -expf(a_log[head]) * softplus;
}

__global__ void silu_mul_kernel(const std::uint16_t* gate, const std::uint16_t* up,
                                std::uint16_t* output, int count) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = half_value(gate[i]);
    const float u = half_value(up[i]);
    output[i] = __half_as_ushort(__float2half_rn((g / (1.0f + expf(-g))) * u));
}

__global__ void residual_kernel(const std::uint16_t* left, const std::uint16_t* right,
                                std::uint16_t* output, int count) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = __half_as_ushort(__float2half_rn(half_value(left[i]) + half_value(right[i])));
}

__global__ void pack_qkv_bf16_kernel(const std::uint16_t* q, const std::uint16_t* k,
                                     const std::uint16_t* v, std::uint16_t* output, int rows) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kQkv;
    if (index >= total) return;
    const int row = index / kQkv;
    const int channel = index % kQkv;
    const auto* source = channel < 2048 ? q : (channel < 4096 ? k : v);
    const int source_channel = channel < 2048 ? channel : (channel < 4096 ? channel - 2048 : channel - 4096);
    reinterpret_cast<__nv_bfloat16*>(output)[row * kQkv + channel] =
        reinterpret_cast<const __nv_bfloat16*>(source)[row * (channel < 4096 ? 2048 : 6144) + source_channel];
}

__global__ void pack_heads_bf16_kernel(const std::uint16_t* input, std::uint16_t* output, int rows) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kZ;
    if (index >= total) return;
    reinterpret_cast<__nv_bfloat16*>(output)[index] = reinterpret_cast<const __nv_bfloat16*>(input)[index];
}

// The checkpoint stores the depthwise filter as [channel, kernel].  This local
// kernel keeps that representation and the [channel, history] state layout
// explicit, avoiding an ambiguous transpose at the public convolution API.
template<bool Trace=false>
__global__ void gdn_conv_kernel(const std::uint16_t* input, const std::uint16_t* weight,
                                std::uint16_t* state, std::uint16_t* q, std::uint16_t* k,
                                std::uint16_t* v, int rows,
                                std::uint16_t* state_trace=nullptr) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= kQkv) return;
    float s0 = bf16_value(state[channel * kConvStorage + 0]);
    float s1 = bf16_value(state[channel * kConvStorage + 1]);
    float s2 = bf16_value(state[channel * kConvStorage + 2]);
    float s3 = bf16_value(state[channel * kConvStorage + 3]);
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(input);
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(weight);
    auto* q_out = reinterpret_cast<__nv_bfloat16*>(q);
    auto* k_out = reinterpret_cast<__nv_bfloat16*>(k);
    auto* v_out = reinterpret_cast<__nv_bfloat16*>(v);
    for (int row = 0; row < rows; ++row) {
        const float x0 = __bfloat162float(x[channel * rows + row]);
        const float acc = __bfloat162float(w[channel * 4 + 0]) * s1 +
                          __bfloat162float(w[channel * 4 + 1]) * s2 +
                          __bfloat162float(w[channel * 4 + 2]) * s3 +
                          __bfloat162float(w[channel * 4 + 3]) * x0;
        const __nv_bfloat16 y = __float2bfloat16_rn(silu_f32(acc));
        if (channel < 2048) q_out[row * 2048 + channel] = y;
        else if (channel < 4096) k_out[row * 2048 + (channel - 2048)] = y;
        else v_out[row * 6144 + (channel - 4096)] = y;
        s0 = s1; s1 = s2; s2 = s3; s3 = x0;
    }
    auto* state_out = reinterpret_cast<__nv_bfloat16*>(state);
    state_out[channel * kConvStorage + 0] = __float2bfloat16_rn(s0);
    state_out[channel * kConvStorage + 1] = __float2bfloat16_rn(s1);
    state_out[channel * kConvStorage + 2] = __float2bfloat16_rn(s2);
    state_out[channel * kConvStorage + 3] = __float2bfloat16_rn(s3);
    if constexpr(Trace) {
        auto* trace=reinterpret_cast<__nv_bfloat16*>(state_trace);
        trace[channel*3+0]=__float2bfloat16_rn(s0);
        trace[channel*3+1]=__float2bfloat16_rn(s1);
        trace[channel*3+2]=__float2bfloat16_rn(s2);
    }
}

// Same-weight FP16-KV decode candidate.  The ordinary decode route submits a
// row-major F16->channel-major BF16 transpose, the recurrent convolution, and
// a separate Q/K/V->packed-BF16 copy.  For the small-M decode geometry, one
// channel owns the complete chronological row loop and can preserve all four
// representations in one exact-order pass:
//   - conv_input remains channel-major BF16 for retained-prefix reconstruction;
//   - q/k/v remain the existing row-major BF16 recurrence inputs;
//   - packed_output remains the existing row-major BF16 trace;
//   - the physical four-slot BF16 convolution state is updated identically.
// The source weights and activations are unchanged; this only removes staging
// launches and their intermediate global-memory reread.  Prefill deliberately
// does not use this kernel because its direct row-major reads are strided.
template<bool Trace=false>
__global__ void gdn_conv_f16_decode_fused_kernel(
    const std::uint16_t* input_f16, const std::uint16_t* weight,
    std::uint16_t* state, std::uint16_t* conv_input_bf16,
    std::uint16_t* q, std::uint16_t* k, std::uint16_t* v,
    std::uint16_t* packed_output_bf16, int rows,
    std::uint16_t* state_trace=nullptr) {
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= kQkv) return;
    float s0 = bf16_value(state[channel * kConvStorage + 0]);
    float s1 = bf16_value(state[channel * kConvStorage + 1]);
    float s2 = bf16_value(state[channel * kConvStorage + 2]);
    float s3 = bf16_value(state[channel * kConvStorage + 3]);
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(weight);
    auto* conv_input = reinterpret_cast<__nv_bfloat16*>(conv_input_bf16);
    auto* q_out = reinterpret_cast<__nv_bfloat16*>(q);
    auto* k_out = reinterpret_cast<__nv_bfloat16*>(k);
    auto* v_out = reinterpret_cast<__nv_bfloat16*>(v);
    auto* packed = reinterpret_cast<__nv_bfloat16*>(packed_output_bf16);
    for (int row = 0; row < rows; ++row) {
        const float x0 = half_value(input_f16[row * kQkv + channel]);
        const __nv_bfloat16 x_bf16 = __float2bfloat16_rn(x0);
        const float x_bf16_value = __bfloat162float(x_bf16);
        conv_input[channel * rows + row] = x_bf16;
        const float acc = __bfloat162float(w[channel * kConvStorage + 0]) * s1 +
                          __bfloat162float(w[channel * kConvStorage + 1]) * s2 +
                          __bfloat162float(w[channel * kConvStorage + 2]) * s3 +
                          __bfloat162float(w[channel * kConvStorage + 3]) * x_bf16_value;
        const __nv_bfloat16 y = __float2bfloat16_rn(silu_f32(acc));
        if (channel < 2048) {
            q_out[row * 2048 + channel] = y;
            packed[row * kQkv + channel] = y;
        } else if (channel < 4096) {
            k_out[row * 2048 + (channel - 2048)] = y;
            packed[row * kQkv + channel] = y;
        } else {
            v_out[row * 6144 + (channel - 4096)] = y;
            packed[row * kQkv + channel] = y;
        }
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = x_bf16_value;
    }
    auto* state_out = reinterpret_cast<__nv_bfloat16*>(state);
    state_out[channel * kConvStorage + 0] = __float2bfloat16_rn(s0);
    state_out[channel * kConvStorage + 1] = __float2bfloat16_rn(s1);
    state_out[channel * kConvStorage + 2] = __float2bfloat16_rn(s2);
    state_out[channel * kConvStorage + 3] = __float2bfloat16_rn(s3);
    if constexpr(Trace) {
        auto* trace=reinterpret_cast<__nv_bfloat16*>(state_trace);
        trace[channel*3+0]=__float2bfloat16_rn(s0);
        trace[channel*3+1]=__float2bfloat16_rn(s1);
        trace[channel*3+2]=__float2bfloat16_rn(s2);
    }
}

// Tiled prefill twin of transpose_f16_to_bf16 + gdn_conv_kernel + pack_qkv.
// The depthwise convolution has no recurrence beyond its three-row window, so
// each (row, channel) output reads its represented BF16 window directly: the
// saved state for rows before zero, otherwise the F16->BF16 input. The
// accumulation expression, SiLU and BF16 rounding are those of gdn_conv_kernel;
// conv_input keeps its channel-major BF16 contents for retained-prefix repair.
// The four-slot state (and optional trace) is written by a second kernel after
// every tile has read the old state.
constexpr int kConvTileRows=64;
constexpr int kConvTileChannels=64;

bool gdn_conv_prefill_tiled_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_GDN_CONV_TILED");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

__global__ void __launch_bounds__(256) gdn_conv_prefill_tiled_kernel(
    const std::uint16_t* input_f16,const std::uint16_t* weight,const std::uint16_t* state,
    std::uint16_t* conv_input_bf16,std::uint16_t* q,std::uint16_t* k,std::uint16_t* v,
    std::uint16_t* packed_bf16,int rows) {
    __shared__ float window[kConvTileRows+3][kConvTileChannels+1];
    const int c0=static_cast<int>(blockIdx.x)*kConvTileChannels;
    const int r0=static_cast<int>(blockIdx.y)*kConvTileRows;
    const int tid=static_cast<int>(threadIdx.x);
    const int lane_channel=tid%kConvTileChannels;
    const int group=tid/kConvTileChannels;
    constexpr int kGroups=256/kConvTileChannels;
    const int channel=c0+lane_channel;
    for(int local=group;local<kConvTileRows+3;local+=kGroups) {
        const int row=r0-3+local;
        float value=0.0f;
        if(row>=0) {
            if(row<rows)
                value=__bfloat162float(__float2bfloat16_rn(
                    half_value(input_f16[row*kQkv+channel])));
        } else {
            value=bf16_value(state[channel*kConvStorage+4+row]);
        }
        window[local][lane_channel]=value;
    }
    __syncthreads();
    const auto* w=reinterpret_cast<const __nv_bfloat16*>(weight);
    const float w0=__bfloat162float(w[channel*4+0]);
    const float w1=__bfloat162float(w[channel*4+1]);
    const float w2=__bfloat162float(w[channel*4+2]);
    const float w3=__bfloat162float(w[channel*4+3]);
    auto* q_out=reinterpret_cast<__nv_bfloat16*>(q);
    auto* k_out=reinterpret_cast<__nv_bfloat16*>(k);
    auto* v_out=reinterpret_cast<__nv_bfloat16*>(v);
    auto* packed=reinterpret_cast<__nv_bfloat16*>(packed_bf16);
    for(int local=group;local<kConvTileRows;local+=kGroups) {
        const int row=r0+local;
        if(row>=rows) break;
        const float s1=window[local][lane_channel];
        const float s2=window[local+1][lane_channel];
        const float s3=window[local+2][lane_channel];
        const float x0=window[local+3][lane_channel];
        const float acc = w0 * s1 +
                          w1 * s2 +
                          w2 * s3 +
                          w3 * x0;
        const __nv_bfloat16 y=__float2bfloat16_rn(silu_f32(acc));
        if(channel<2048) q_out[row*2048+channel]=y;
        else if(channel<4096) k_out[row*2048+(channel-2048)]=y;
        else v_out[row*6144+(channel-4096)]=y;
        packed[row*kQkv+channel]=y;
    }
    auto* conv_input=reinterpret_cast<__nv_bfloat16*>(conv_input_bf16);
    const int row=r0+lane_channel;
    if(row<rows) {
        for(int local_channel=group;local_channel<kConvTileChannels;local_channel+=kGroups)
            conv_input[static_cast<std::size_t>(c0+local_channel)*rows+row]=
                __float2bfloat16_rn(window[lane_channel+3][local_channel]);
    }
}

template<bool Trace>
__global__ void gdn_conv_prefill_state_kernel(const std::uint16_t* input_f16,
    std::uint16_t* state,int rows,std::uint16_t* state_trace) {
    const int channel=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(channel>=kQkv) return;
    float history[4];
    #pragma unroll
    for(int slot=0;slot<4;++slot) {
        const int row=rows-4+slot;
        history[slot]=row>=0?
            __bfloat162float(__float2bfloat16_rn(half_value(input_f16[row*kQkv+channel]))):
            bf16_value(state[channel*kConvStorage+4+row]);
    }
    auto* state_out=reinterpret_cast<__nv_bfloat16*>(state);
    #pragma unroll
    for(int slot=0;slot<4;++slot)
        state_out[channel*kConvStorage+slot]=__float2bfloat16_rn(history[slot]);
    if constexpr(Trace) {
        auto* trace=reinterpret_cast<__nv_bfloat16*>(state_trace);
        #pragma unroll
        for(int slot=0;slot<3;++slot)
            trace[channel*3+slot]=__float2bfloat16_rn(history[slot]);
    }
}

// Select specialized leaves on the host: all-off convolution has no trace
// branch or candidate stores, matching the inherited operation sequence.
void launch_gopt_conv(bool trace,bool decode,const std::uint16_t* input,
    const std::uint16_t* weight,std::uint16_t* state,std::uint16_t* conv_input,
    std::uint16_t* q,std::uint16_t* k,std::uint16_t* v,std::uint16_t* packed,
    std::uint16_t* state_trace,int rows,cudaStream_t stream) {
    if(decode) {
        if(trace)gdn_conv_f16_decode_fused_kernel<true><<<(kQkv+255)/256,256,0,stream>>>(input,weight,state,conv_input,q,k,v,packed,rows,state_trace);
        else gdn_conv_f16_decode_fused_kernel<false><<<(kQkv+255)/256,256,0,stream>>>(input,weight,state,conv_input,q,k,v,packed,rows);
    } else {
        if(trace)exl3_launch_small(gdn_conv_kernel<true>,dim3((kQkv+255)/256),dim3(256),0,stream,conv_input,weight,state,q,k,v,rows,state_trace);
        else exl3_launch_small(gdn_conv_kernel<false>,dim3((kQkv+255)/256),dim3(256),0,stream,conv_input,weight,state,q,k,v,rows);
    }
}

__global__ void copy_conv_state_trace_kernel(const std::uint16_t* state, std::uint16_t* trace) {
    EXL3_PDL_SMALL_PROLOGUE();
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int>(Exl3GdnRecurrentLayout::convolution_history_elements)) return;
    const int channel = index / static_cast<int>(Exl3GdnRecurrentLayout::convolution_history);
    const int slot = index % static_cast<int>(Exl3GdnRecurrentLayout::convolution_history);
    trace[index] = state[channel * kConvStorage + slot];
}

// Rebuild the physical four-slot state without repeating convolution
// arithmetic. conv_input is the preserved channel-major [channel, original B]
// representation from the immediately preceding verifier forward.
__global__ void reconstruct_conv_prefix_kernel(
    const std::uint16_t* base_state, const std::uint16_t* conv_input,
    std::uint16_t* state, std::uint16_t* trace,
    int retained_rows, int original_rows) {
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= kQkv) return;
    std::uint16_t s0 = base_state[channel * kConvStorage + 0];
    std::uint16_t s1 = base_state[channel * kConvStorage + 1];
    std::uint16_t s2 = base_state[channel * kConvStorage + 2];
    std::uint16_t s3 = base_state[channel * kConvStorage + 3];
    for (int row = 0; row < retained_rows; ++row) {
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = conv_input[channel * original_rows + row];
    }
    state[channel * kConvStorage + 0] = s0;
    state[channel * kConvStorage + 1] = s1;
    state[channel * kConvStorage + 2] = s2;
    state[channel * kConvStorage + 3] = s3;
    trace[channel * 3 + 0] = s0;
    trace[channel * 3 + 1] = s1;
    trace[channel * 3 + 2] = s2;
}

// Exact recurrent reference geometry retained for diagnosis.  One thread owns
// one value head, so this deliberately mirrors the correctness-first E3B path.
__global__ void gdn_recurrence_reference_kernel(const std::uint16_t* q, const std::uint16_t* k,
                                      const std::uint16_t* v, const float* g, const float* beta,
                                      float* state, std::uint16_t* output, int rows) {
    const int head = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head >= kHeads) return;
    const int qk_head = head / 3;
    auto* qv = reinterpret_cast<const __nv_bfloat16*>(q);
    auto* kv = reinterpret_cast<const __nv_bfloat16*>(k);
    auto* vv = reinterpret_cast<const __nv_bfloat16*>(v);
    auto* out = reinterpret_cast<__nv_bfloat16*>(output);
    float* s = state + head * kHeadDim * kHeadDim;
    float q_norm[kHeadDim], k_norm[kHeadDim];
    for (int row = 0; row < rows; ++row) {
        float q_sum = 0.0f, k_sum = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            q_norm[d] = __bfloat162float(qv[row * (kKeyHeads * kHeadDim) + qk_head * kHeadDim + d]);
            k_norm[d] = __bfloat162float(kv[row * (kKeyHeads * kHeadDim) + qk_head * kHeadDim + d]);
            q_sum += q_norm[d] * q_norm[d]; k_sum += k_norm[d] * k_norm[d];
        }
        const float q_inv = rsqrtf(q_sum + 1.0e-6f), k_inv = rsqrtf(k_sum + 1.0e-6f);
        for (int d = 0; d < kHeadDim; ++d) { q_norm[d] *= q_inv; k_norm[d] *= k_inv; }
        const float alpha = expf(g[row * kHeads + head]);
        const float beta_value = beta[row * kHeads + head];
        for (int kd = 0; kd < kHeadDim; ++kd) {
            for (int vd = 0; vd < kHeadDim; ++vd) s[kd * kHeadDim + vd] *= alpha;
        }
        for (int vd = 0; vd < kHeadDim; ++vd) {
            float kv_mem = 0.0f;
            for (int kd = 0; kd < kHeadDim; ++kd) kv_mem += s[kd * kHeadDim + vd] * k_norm[kd];
            const float value = __bfloat162float(vv[row * (kHeads * kHeadDim) + head * kHeadDim + vd]);
            const float delta = beta_value * (value - kv_mem);
            for (int kd = 0; kd < kHeadDim; ++kd) s[kd * kHeadDim + vd] += k_norm[kd] * delta;
        }
        for (int vd = 0; vd < kHeadDim; ++vd) {
            float result = 0.0f;
            for (int kd = 0; kd < kHeadDim; ++kd) result += s[kd * kHeadDim + vd] * q_norm[kd];
            out[row * (kHeads * kHeadDim) + head * kHeadDim + vd] = __float2bfloat16_rn(result / sqrtf(128.0f));
        }
    }
}

// E3C2 SM120 recurrence leaf.  A warp owns one value-column of one head state;
// lane == key dimension.  This lets every state element be read once, decayed,
// used for the prediction, updated, and written once, while the two reductions
// remain entirely in registers/shuffle lanes.  Four warps per CTA cover four
// value columns, and the grid tiles all 128 columns independently.  Rows remain
// sequential within a tile so causal state semantics are unchanged.
__global__ void gdn_recurrence_sm120_kernel(const std::uint16_t* q, const std::uint16_t* k,
                                            const std::uint16_t* v, const float* g, const float* beta,
                                            float* state, std::uint16_t* output, int rows) {
    constexpr int kWarpsPerBlock = 4;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int tile = static_cast<int>(blockIdx.x);
    const int head = tile / (kHeadDim / kWarpsPerBlock);
    const int value_tile = tile % (kHeadDim / kWarpsPerBlock);
    if (head >= kHeads) return;

    const int value_dim = value_tile * kWarpsPerBlock + warp;
    const int qk_head = head / 3;
    const auto* qv = reinterpret_cast<const __nv_bfloat16*>(q);
    const auto* kv = reinterpret_cast<const __nv_bfloat16*>(k);
    const auto* vv = reinterpret_cast<const __nv_bfloat16*>(v);
    auto* out = reinterpret_cast<__nv_bfloat16*>(output);
    float* s = state + head * kHeadDim * kHeadDim;
    constexpr unsigned mask = 0xffffffffu;
    constexpr float output_scale = 0.08838834764831843f; // 1 / sqrt(128)
    // The warp's state column stays in registers across rows: loaded once,
    // stored once. Per-row arithmetic is unchanged.
    float column[4];
    #pragma unroll
    for (int part = 0; part < 4; ++part)
        column[part] = s[(lane + part * 32) * kHeadDim + value_dim];
    for (int row = 0; row < rows; ++row) {
        const int qk_base = row * (kKeyHeads * kHeadDim) + qk_head * kHeadDim;
        float q_norm[4], k_norm[4];
        float q_sum = 0.0f, k_sum = 0.0f;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int kd = lane + part * 32;
            const float qd = __bfloat162float(qv[qk_base + kd]);
            const float kval = __bfloat162float(kv[qk_base + kd]);
            q_norm[part] = qd;
            k_norm[part] = kval;
            q_sum += qd * qd;
            k_sum += kval * kval;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            q_sum += __shfl_down_sync(mask, q_sum, offset);
            k_sum += __shfl_down_sync(mask, k_sum, offset);
        }
        q_sum = __shfl_sync(mask, q_sum, 0);
        k_sum = __shfl_sync(mask, k_sum, 0);
        const float q_inv = rsqrtf(q_sum + kRmsEps);
        const float k_inv = rsqrtf(k_sum + kRmsEps);
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            q_norm[part] *= q_inv;
            k_norm[part] *= k_inv;
        }
        const float alpha = expf(g[row * kHeads + head]);
        const float beta_value = beta[row * kHeads + head];

        float updated[4];
        float kv_mem = 0.0f;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            updated[part] = column[part] * alpha;
            kv_mem += updated[part] * k_norm[part];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            kv_mem += __shfl_down_sync(mask, kv_mem, offset);
        }
        kv_mem = __shfl_sync(mask, kv_mem, 0);
        const float value = __bfloat162float(vv[row * (kHeads * kHeadDim) + head * kHeadDim + value_dim]);
        const float delta = beta_value * (value - kv_mem);
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            updated[part] += k_norm[part] * delta;
            column[part] = updated[part];
        }

        float result = 0.0f;
        #pragma unroll
        for (int part = 0; part < 4; ++part) result += updated[part] * q_norm[part];
        for (int offset = 16; offset > 0; offset >>= 1) {
            result += __shfl_down_sync(mask, result, offset);
        }
        if (lane == 0) {
            out[row * (kHeads * kHeadDim) + head * kHeadDim + value_dim] = __float2bfloat16_rn(result * output_scale);
        }
    }
    #pragma unroll
    for (int part = 0; part < 4; ++part)
        s[(lane + part * 32) * kHeadDim + value_dim] = column[part];
}

// Coalesced-state twin of gdn_recurrence_sm120_kernel for verifier-sized
// batches. A CTA stages a [128 key][32 value] block of the recurrent state
// with 16-byte loads, each warp carries four value columns as independent
// chains (lane == key dimension, exactly as the sm120 leaf), and the block is
// written back with 16-byte stores. Per-column arithmetic, reduction trees and
// row order are those of gdn_recurrence_sm120_kernel.
constexpr int kGdnTiledColumns=32;
constexpr int kGdnTiledWarps=8;

bool gdn_decode_tiled_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_GDN_DECODE_TILED");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

__global__ void __launch_bounds__(kGdnTiledWarps*32) gdn_recurrence_sm120_tiled_kernel(
    const std::uint16_t* q,const std::uint16_t* k,const std::uint16_t* v,const float* g,
    const float* beta,float* state,std::uint16_t* output,int rows) {
    EXL3_PDL_SMALL_PROLOGUE();
    constexpr int kColumnsPerWarp=kGdnTiledColumns/kGdnTiledWarps;
    __shared__ float tile[kHeadDim][kGdnTiledColumns+1];
    const int tid=static_cast<int>(threadIdx.x);
    const int lane=tid&31;
    const int warp=tid>>5;
    constexpr int kTilesPerHead=kHeadDim/kGdnTiledColumns;
    const int head=static_cast<int>(blockIdx.x)/kTilesPerHead;
    const int column_base=(static_cast<int>(blockIdx.x)%kTilesPerHead)*kGdnTiledColumns;
    if(head>=kHeads) return;
    float* s=state+head*kHeadDim*kHeadDim;
    constexpr int kVectorsPerRow=kGdnTiledColumns/4;
    for(int index=tid;index<kHeadDim*kVectorsPerRow;index+=blockDim.x) {
        const int key=index/kVectorsPerRow;
        const int part=(index%kVectorsPerRow)*4;
        const float4 values=*reinterpret_cast<const float4*>(s+key*kHeadDim+column_base+part);
        tile[key][part+0]=values.x;
        tile[key][part+1]=values.y;
        tile[key][part+2]=values.z;
        tile[key][part+3]=values.w;
    }
    __syncthreads();
    const int qk_head=head/3;
    const auto* qv=reinterpret_cast<const __nv_bfloat16*>(q);
    const auto* kv=reinterpret_cast<const __nv_bfloat16*>(k);
    const auto* vv=reinterpret_cast<const __nv_bfloat16*>(v);
    auto* out=reinterpret_cast<__nv_bfloat16*>(output);
    constexpr unsigned mask=0xffffffffu;
    constexpr float output_scale=0.08838834764831843f;
    float column[kColumnsPerWarp][4];
    #pragma unroll
    for(int c=0;c<kColumnsPerWarp;++c)
        #pragma unroll
        for(int part=0;part<4;++part)
            column[c][part]=tile[lane+part*32][warp*kColumnsPerWarp+c];
    for(int row=0;row<rows;++row) {
        const int qk_base=row*(kKeyHeads*kHeadDim)+qk_head*kHeadDim;
        float q_norm[4],k_norm[4];
        float q_sum=0.0f,k_sum=0.0f;
        #pragma unroll
        for(int part=0;part<4;++part) {
            const int kd=lane+part*32;
            const float qd=__bfloat162float(qv[qk_base+kd]);
            const float kval=__bfloat162float(kv[qk_base+kd]);
            q_norm[part]=qd;
            k_norm[part]=kval;
            q_sum+=qd*qd;
            k_sum+=kval*kval;
        }
        for(int offset=16;offset>0;offset>>=1) {
            q_sum+=__shfl_down_sync(mask,q_sum,offset);
            k_sum+=__shfl_down_sync(mask,k_sum,offset);
        }
        q_sum=__shfl_sync(mask,q_sum,0);
        k_sum=__shfl_sync(mask,k_sum,0);
        const float q_inv=rsqrtf(q_sum+kRmsEps);
        const float k_inv=rsqrtf(k_sum+kRmsEps);
        #pragma unroll
        for(int part=0;part<4;++part) {
            q_norm[part]*=q_inv;
            k_norm[part]*=k_inv;
        }
        const float alpha=expf(g[row*kHeads+head]);
        const float beta_value=beta[row*kHeads+head];
        float updated[kColumnsPerWarp][4];
        float kv_mem[kColumnsPerWarp];
        #pragma unroll
        for(int c=0;c<kColumnsPerWarp;++c) {
            kv_mem[c]=0.0f;
            #pragma unroll
            for(int part=0;part<4;++part) {
                updated[c][part]=column[c][part]*alpha;
                kv_mem[c]+=updated[c][part]*k_norm[part];
            }
        }
        for(int offset=16;offset>0;offset>>=1) {
            #pragma unroll
            for(int c=0;c<kColumnsPerWarp;++c)
                kv_mem[c]+=__shfl_down_sync(mask,kv_mem[c],offset);
        }
        float result[kColumnsPerWarp];
        #pragma unroll
        for(int c=0;c<kColumnsPerWarp;++c) {
            kv_mem[c]=__shfl_sync(mask,kv_mem[c],0);
            const int value_dim=column_base+warp*kColumnsPerWarp+c;
            const float value=__bfloat162float(vv[row*(kHeads*kHeadDim)+head*kHeadDim+value_dim]);
            const float delta=beta_value*(value-kv_mem[c]);
            #pragma unroll
            for(int part=0;part<4;++part) {
                updated[c][part]+=k_norm[part]*delta;
                column[c][part]=updated[c][part];
            }
            result[c]=0.0f;
            #pragma unroll
            for(int part=0;part<4;++part) result[c]+=updated[c][part]*q_norm[part];
        }
        for(int offset=16;offset>0;offset>>=1) {
            #pragma unroll
            for(int c=0;c<kColumnsPerWarp;++c)
                result[c]+=__shfl_down_sync(mask,result[c],offset);
        }
        if(lane==0) {
            #pragma unroll
            for(int c=0;c<kColumnsPerWarp;++c) {
                const int value_dim=column_base+warp*kColumnsPerWarp+c;
                out[row*(kHeads*kHeadDim)+head*kHeadDim+value_dim]=
                    __float2bfloat16_rn(result[c]*output_scale);
            }
        }
    }
    #pragma unroll
    for(int c=0;c<kColumnsPerWarp;++c)
        #pragma unroll
        for(int part=0;part<4;++part)
            tile[lane+part*32][warp*kColumnsPerWarp+c]=column[c][part];
    __syncthreads();
    for(int index=tid;index<kHeadDim*kVectorsPerRow;index+=blockDim.x) {
        const int key=index/kVectorsPerRow;
        const int part=(index%kVectorsPerRow)*4;
        *reinterpret_cast<float4*>(s+key*kHeadDim+column_base+part)=make_float4(
            tile[key][part+0],tile[key][part+1],tile[key][part+2],tile[key][part+3]);
    }
}

void launch_gdn_recurrence_sm120(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,const float* g,const float* beta,float* state,
    std::uint16_t* output,int rows,cudaStream_t stream) {
    if(rows>=1 && rows<=8 && gdn_decode_tiled_enabled())
        exl3_launch_small(gdn_recurrence_sm120_tiled_kernel,dim3(kHeads*(kHeadDim/kGdnTiledColumns)),dim3(kGdnTiledWarps*32),0,stream,q,k,v,g,beta,state,output,rows);
    else
        gdn_recurrence_sm120_kernel<<<kHeads*(kHeadDim/4),4*32,0,stream>>>(
            q,k,v,g,beta,state,output,rows);
}

__global__ void gdn_prefill_normalize_kernel(const std::uint16_t* q,
    const std::uint16_t* k, const float* g, float* qout, float* kout, float* alpha) {
    const int lane = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.x) / kKeyHeads;
    const int qk_head = static_cast<int>(blockIdx.x) % kKeyHeads;
    const int qk_base = row * (kKeyHeads * kHeadDim) + qk_head * kHeadDim;
    const auto* qv = reinterpret_cast<const __nv_bfloat16*>(q);
    const auto* kv = reinterpret_cast<const __nv_bfloat16*>(k);
    constexpr unsigned mask = 0xffffffffu;
    float q_norm[4], k_norm[4];
    float q_sum = 0.0f, k_sum = 0.0f;
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        const int kd = lane + part * 32;
        const float qd = __bfloat162float(qv[qk_base + kd]);
        const float kval = __bfloat162float(kv[qk_base + kd]);
        q_norm[part] = qd;
        k_norm[part] = kval;
        q_sum += qd * qd;
        k_sum += kval * kval;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(mask, q_sum, offset);
        k_sum += __shfl_down_sync(mask, k_sum, offset);
    }
    q_sum = __shfl_sync(mask, q_sum, 0);
    k_sum = __shfl_sync(mask, k_sum, 0);
    const float q_inv = rsqrtf(q_sum + kRmsEps);
    const float k_inv = rsqrtf(k_sum + kRmsEps);
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        q_norm[part] *= q_inv;
        k_norm[part] *= k_inv;
    }
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
        qout[qk_base + lane + part * 32] = q_norm[part];
        kout[qk_base + lane + part * 32] = k_norm[part];
    }
    if (lane < 3) {
        const int head = qk_head * 3 + lane;
        alpha[row * kHeads + head] = expf(g[row * kHeads + head]);
    }
}

__global__ void gdn_recurrence_prefill_resident_kernel(const float* q, const float* k,
                                            const std::uint16_t* v, const float* g, const float* beta,
                                            float* state, std::uint16_t* output, int rows) {
    constexpr int kWarpsPerBlock = 4;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int tile = static_cast<int>(blockIdx.x);
    const int head = tile / (kHeadDim / kWarpsPerBlock);
    const int value_tile = tile % (kHeadDim / kWarpsPerBlock);
    if (head >= kHeads) return;

    const int value_dim = value_tile * kWarpsPerBlock + warp;
    const int qk_head = head / 3;
    const auto* vv = reinterpret_cast<const __nv_bfloat16*>(v);
    auto* out = reinterpret_cast<__nv_bfloat16*>(output);
    float* s = state + head * kHeadDim * kHeadDim;
    constexpr unsigned mask = 0xffffffffu;
    constexpr float output_scale = 0.08838834764831843f; // 1 / sqrt(128)
    float resident[4];
    #pragma unroll
    for (int part = 0; part < 4; ++part)
        resident[part] = s[(lane + part * 32) * kHeadDim + value_dim];
    for (int row = 0; row < rows; ++row) {
        const int qk_base = row * (kKeyHeads * kHeadDim) + qk_head * kHeadDim;
        float q_norm[4], k_norm[4];
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            q_norm[part] = q[qk_base + lane + part * 32];
            k_norm[part] = k[qk_base + lane + part * 32];
        }
        const float alpha = g[row * kHeads + head];
        const float beta_value = beta[row * kHeads + head];

        float updated[4];
        float kv_mem = 0.0f;
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            updated[part] = resident[part] * alpha;
            kv_mem += updated[part] * k_norm[part];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            kv_mem += __shfl_down_sync(mask, kv_mem, offset);
        }
        kv_mem = __shfl_sync(mask, kv_mem, 0);
        const float value = __bfloat162float(vv[row * (kHeads * kHeadDim) + head * kHeadDim + value_dim]);
        const float delta = beta_value * (value - kv_mem);
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            updated[part] += k_norm[part] * delta;
            resident[part] = updated[part];
        }

        float result = 0.0f;
        #pragma unroll
        for (int part = 0; part < 4; ++part) result += updated[part] * q_norm[part];
        for (int offset = 16; offset > 0; offset >>= 1) {
            result += __shfl_down_sync(mask, result, offset);
        }
        if (lane == 0) {
            out[row * (kHeads * kHeadDim) + head * kHeadDim + value_dim] = __float2bfloat16_rn(result * output_scale);
        }
    }
    #pragma unroll
    for (int part = 0; part < 4; ++part)
        s[(lane + part * 32) * kHeadDim + value_dim] = resident[part];
}

// Each warp owns two independent value columns. The two FP32 recurrence and
// reduction chains retain their former part/lane order, while normalized Q/K,
// alpha and beta are loaded once for the pair instead of once per column.
template<bool VectorIO>
__global__ void gdn_recurrence_prefill_resident_pair_columns_kernel(
    const float* q,const float* k,const std::uint16_t* v,const float* g,
    const float* beta,float* state,std::uint16_t* output,int rows) {
    constexpr int kWarpsPerBlock=4,kColumnsPerWarp=2;
    const int lane=static_cast<int>(threadIdx.x)&31;
    const int warp=static_cast<int>(threadIdx.x)>>5;
    const int tiles_per_head=kHeadDim/(kWarpsPerBlock*kColumnsPerWarp);
    const int tile=static_cast<int>(blockIdx.x);
    const int head=tile/tiles_per_head;
    const int value_tile=tile%tiles_per_head;
    if(head>=kHeads) return;
    const int value0=(value_tile*kWarpsPerBlock+warp)*kColumnsPerWarp;
    const int value1=value0+1;
    const int qk_head=head/3;
    const auto* vv=reinterpret_cast<const __nv_bfloat16*>(v);
    auto* out=reinterpret_cast<__nv_bfloat16*>(output);
    float* s=state+head*kHeadDim*kHeadDim;
    constexpr unsigned mask=0xffffffffu;
    constexpr float output_scale=0.08838834764831843f;
    float resident0[4],resident1[4];
    #pragma unroll
    for(int part=0;part<4;++part) {
        const int key=lane+part*32;
        if constexpr(VectorIO) {
            const float2 resident=
                *reinterpret_cast<const float2*>(s+key*kHeadDim+value0);
            resident0[part]=resident.x;
            resident1[part]=resident.y;
        } else {
            resident0[part]=s[key*kHeadDim+value0];
            resident1[part]=s[key*kHeadDim+value1];
        }
    }
    for(int row=0;row<rows;++row) {
        const int qk_base=row*(kKeyHeads*kHeadDim)+qk_head*kHeadDim;
        float q_norm[4],k_norm[4];
        #pragma unroll
        for(int part=0;part<4;++part) {
            q_norm[part]=q[qk_base+lane+part*32];
            k_norm[part]=k[qk_base+lane+part*32];
        }
        const float alpha=g[row*kHeads+head];
        const float beta_value=beta[row*kHeads+head];
        float updated0[4],updated1[4];
        float kv_mem0=0.0f,kv_mem1=0.0f;
        #pragma unroll
        for(int part=0;part<4;++part) {
            updated0[part]=resident0[part]*alpha;
            updated1[part]=resident1[part]*alpha;
            kv_mem0+=updated0[part]*k_norm[part];
            kv_mem1+=updated1[part]*k_norm[part];
        }
        for(int offset=16;offset>0;offset>>=1) {
            kv_mem0+=__shfl_down_sync(mask,kv_mem0,offset);
            kv_mem1+=__shfl_down_sync(mask,kv_mem1,offset);
        }
        kv_mem0=__shfl_sync(mask,kv_mem0,0);
        kv_mem1=__shfl_sync(mask,kv_mem1,0);
        const int value_base=row*(kHeads*kHeadDim)+head*kHeadDim;
        float observed0,observed1;
        if constexpr(VectorIO) {
            const float2 observed=__bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(vv+value_base+value0));
            observed0=observed.x;
            observed1=observed.y;
        } else {
            observed0=__bfloat162float(vv[value_base+value0]);
            observed1=__bfloat162float(vv[value_base+value1]);
        }
        const float delta0=beta_value*(observed0-kv_mem0);
        const float delta1=beta_value*(observed1-kv_mem1);
        #pragma unroll
        for(int part=0;part<4;++part) {
            updated0[part]+=k_norm[part]*delta0;
            updated1[part]+=k_norm[part]*delta1;
            resident0[part]=updated0[part];
            resident1[part]=updated1[part];
        }
        float result0=0.0f,result1=0.0f;
        #pragma unroll
        for(int part=0;part<4;++part) {
            result0+=updated0[part]*q_norm[part];
            result1+=updated1[part]*q_norm[part];
        }
        for(int offset=16;offset>0;offset>>=1) {
            result0+=__shfl_down_sync(mask,result0,offset);
            result1+=__shfl_down_sync(mask,result1,offset);
        }
        if(lane==0) {
            if constexpr(VectorIO)
                *reinterpret_cast<__nv_bfloat162*>(out+value_base+value0)=
                    __floats2bfloat162_rn(result0*output_scale,result1*output_scale);
            else {
                out[value_base+value0]=__float2bfloat16_rn(result0*output_scale);
                out[value_base+value1]=__float2bfloat16_rn(result1*output_scale);
            }
        }
    }
    #pragma unroll
    for(int part=0;part<4;++part) {
        const int key=lane+part*32;
        if constexpr(VectorIO)
            *reinterpret_cast<float2*>(s+key*kHeadDim+value0)=
                make_float2(resident0[part],resident1[part]);
        else {
            s[key*kHeadDim+value0]=resident0[part];
            s[key*kHeadDim+value1]=resident1[part];
        }
    }
}

// Profile-guided extension of the qualified paired-column kernel. Four
// independent value columns share Q/K, alpha and beta; every column retains
// its own chronological FP32 recurrence and lane-reduction dependency chain.
__global__ void gdn_recurrence_prefill_resident_quad_columns_kernel(
    const float* q,const float* k,const std::uint16_t* v,const float* g,
    const float* beta,float* state,std::uint16_t* output,int rows) {
    constexpr int kWarpsPerBlock=4,kColumnsPerWarp=4;
    const int lane=static_cast<int>(threadIdx.x)&31;
    const int warp=static_cast<int>(threadIdx.x)>>5;
    const int tiles_per_head=kHeadDim/(kWarpsPerBlock*kColumnsPerWarp);
    const int tile=static_cast<int>(blockIdx.x);
    const int head=tile/tiles_per_head,value_tile=tile%tiles_per_head;
    if(head>=kHeads) return;
    const int value_base_column=(value_tile*kWarpsPerBlock+warp)*kColumnsPerWarp;
    const int qk_head=head/3;
    const auto* vv=reinterpret_cast<const __nv_bfloat16*>(v);
    auto* out=reinterpret_cast<__nv_bfloat16*>(output);
    float* s=state+head*kHeadDim*kHeadDim;
    constexpr unsigned mask=0xffffffffu;
    constexpr float output_scale=0.08838834764831843f;
    float resident[kColumnsPerWarp][4];
    #pragma unroll
    for(int column=0;column<kColumnsPerWarp;++column)
        #pragma unroll
        for(int part=0;part<4;++part) {
            const int key=lane+part*32;
            resident[column][part]=s[key*kHeadDim+value_base_column+column];
        }
    for(int row=0;row<rows;++row) {
        const int qk_base=row*(kKeyHeads*kHeadDim)+qk_head*kHeadDim;
        float q_norm[4],k_norm[4];
        #pragma unroll
        for(int part=0;part<4;++part) {
            q_norm[part]=q[qk_base+lane+part*32];
            k_norm[part]=k[qk_base+lane+part*32];
        }
        const float alpha=g[row*kHeads+head],beta_value=beta[row*kHeads+head];
        float updated[kColumnsPerWarp][4];
        float kv_mem[kColumnsPerWarp]={};
        #pragma unroll
        for(int column=0;column<kColumnsPerWarp;++column)
            #pragma unroll
            for(int part=0;part<4;++part) {
                updated[column][part]=resident[column][part]*alpha;
                kv_mem[column]+=updated[column][part]*k_norm[part];
            }
        for(int offset=16;offset>0;offset>>=1)
            #pragma unroll
            for(int column=0;column<kColumnsPerWarp;++column)
                kv_mem[column]+=__shfl_down_sync(mask,kv_mem[column],offset);
        #pragma unroll
        for(int column=0;column<kColumnsPerWarp;++column)
            kv_mem[column]=__shfl_sync(mask,kv_mem[column],0);
        const int output_base=row*(kHeads*kHeadDim)+head*kHeadDim;
        float delta[kColumnsPerWarp];
        #pragma unroll
        for(int column=0;column<kColumnsPerWarp;++column)
            delta[column]=beta_value*(
                __bfloat162float(vv[output_base+value_base_column+column])-kv_mem[column]);
        #pragma unroll
        for(int column=0;column<kColumnsPerWarp;++column)
            #pragma unroll
            for(int part=0;part<4;++part) {
                updated[column][part]+=k_norm[part]*delta[column];
                resident[column][part]=updated[column][part];
            }
        float result[kColumnsPerWarp]={};
        #pragma unroll
        for(int column=0;column<kColumnsPerWarp;++column)
            #pragma unroll
            for(int part=0;part<4;++part)
                result[column]+=updated[column][part]*q_norm[part];
        for(int offset=16;offset>0;offset>>=1)
            #pragma unroll
            for(int column=0;column<kColumnsPerWarp;++column)
                result[column]+=__shfl_down_sync(mask,result[column],offset);
        if(lane==0)
            #pragma unroll
            for(int column=0;column<kColumnsPerWarp;++column)
                out[output_base+value_base_column+column]=
                    __float2bfloat16_rn(result[column]*output_scale);
    }
    #pragma unroll
    for(int column=0;column<kColumnsPerWarp;++column)
        #pragma unroll
        for(int part=0;part<4;++part) {
            const int key=lane+part*32;
            s[key*kHeadDim+value_base_column+column]=resident[column][part];
        }
}

__global__ void control_projection_kernel(const std::uint16_t* input, const std::uint16_t* weight,
                                          float* output, int rows) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kHeads;
    if (index >= total) return;
    const int row = index / kHeads;
    const int out = index % kHeads;
    float sum = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
        sum = fmaf(half_value(input[row * kHidden + i]), half_value(weight[out * kHidden + i]), sum);
    }
    output[index] = sum;
}

// The two Qwen3.5 GDN control projections consume the same normalized hidden
// vector and have only 48 outputs each.  Fuse them with the control transforms
// so the activation is traversed once and no intermediate control kernels or
// beta/g transposes are needed.  One CTA owns one (row, value-head) pair; each
// thread accumulates a strided slice of both F16 checkpoint-layout rows.
__global__ void control_fused_kernel(const std::uint16_t* input,
                                     const std::uint16_t* a_weight,
                                     const std::uint16_t* b_weight,
                                     const float* a_log, const float* dt_bias,
                                     float* a_output, float* b_output,
                                     float* beta_trace, float* g_trace, int rows) {
    EXL3_PDL_SMALL_PROLOGUE();
    constexpr int kThreads = 128;
    constexpr int kWarps = kThreads / 32;
    __shared__ float partial_a[kWarps];
    __shared__ float partial_b[kWarps];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int index = static_cast<int>(blockIdx.x);
    const int row = index / kHeads;
    const int head = index % kHeads;
    if (row >= rows) return;

    const int input_base = row * kHidden;
    const int weight_base = head * kHidden;
    float asum = 0.0f, bsum = 0.0f;
    // Loads for eight consecutive strided elements are issued together; the
    // fmaf chain still visits d = t, t+128, ... in the same order.
    constexpr int kBatch = 8;
    static_assert(kHidden % (kThreads * kBatch) == 0);
    for (int step = 0; step < kHidden / kThreads; step += kBatch) {
        float x[kBatch], wa[kBatch], wb[kBatch];
        #pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            const int d = static_cast<int>(threadIdx.x) + (step + j) * kThreads;
            x[j] = half_value(input[input_base + d]);
            wa[j] = half_value(a_weight[weight_base + d]);
            wb[j] = half_value(b_weight[weight_base + d]);
        }
        #pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            asum = fmaf(x[j], wa[j], asum);
            bsum = fmaf(x[j], wb[j], bsum);
        }
    }
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        asum += __shfl_down_sync(mask, asum, offset);
        bsum += __shfl_down_sync(mask, bsum, offset);
    }
    if (lane == 0) {
        partial_a[warp] = asum;
        partial_b[warp] = bsum;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        asum = 0.0f;
        bsum = 0.0f;
        for (int w = 0; w < kWarps; ++w) {
            asum += partial_a[w];
            bsum += partial_b[w];
        }
        a_output[index] = asum;
        b_output[index] = bsum;
        const float beta_f = 1.0f / (1.0f + expf(-bsum));
        const float av = asum + dt_bias[head];
        const float softplus = av > 20.0f ? av : log1pf(expf(av));
        beta_trace[index] = __bfloat162float(__float2bfloat16_rn(beta_f));
        g_trace[index] = -expf(a_log[head]) * softplus;
    }
}

// Staged twin of control_fused_kernel: the CTA copies the activation row and
// both weight rows to shared memory with 16-byte loads in one pass, then each
// thread runs the identical fmaf chain over d = t, t+128, ... and the same
// shuffle / four-warp reduction, so every output is bitwise unchanged.
__global__ void __launch_bounds__(128) control_fused_staged_kernel(
    const std::uint16_t* input, const std::uint16_t* a_weight,
    const std::uint16_t* b_weight, const float* a_log, const float* dt_bias,
    float* a_output, float* b_output, float* beta_trace, float* g_trace, int rows) {
    EXL3_PDL_SMALL_PROLOGUE();
    constexpr int kThreads = 128;
    constexpr int kWarps = kThreads / 32;
    constexpr int kVectors = kHidden / 8;
    __shared__ uint4 staged_x[kVectors], staged_a[kVectors], staged_b[kVectors];
    __shared__ float partial_a[kWarps];
    __shared__ float partial_b[kWarps];
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int index = static_cast<int>(blockIdx.x);
    const int row = index / kHeads;
    const int head = index % kHeads;
    if (row >= rows) return;
    const auto* x_source = reinterpret_cast<const uint4*>(input + row * kHidden);
    const auto* a_source = reinterpret_cast<const uint4*>(a_weight + head * kHidden);
    const auto* b_source = reinterpret_cast<const uint4*>(b_weight + head * kHidden);
    #pragma unroll
    for (int i = tid; i < kVectors; i += kThreads) {
        staged_x[i] = x_source[i];
        staged_a[i] = a_source[i];
        staged_b[i] = b_source[i];
    }
    __syncthreads();
    const auto* xs = reinterpret_cast<const std::uint16_t*>(staged_x);
    const auto* as = reinterpret_cast<const std::uint16_t*>(staged_a);
    const auto* bs = reinterpret_cast<const std::uint16_t*>(staged_b);
    float asum = 0.0f, bsum = 0.0f;
    #pragma unroll 8
    for (int step = 0; step < kHidden / kThreads; ++step) {
        const int d = tid + step * kThreads;
        const float x = half_value(xs[d]);
        asum = fmaf(x, half_value(as[d]), asum);
        bsum = fmaf(x, half_value(bs[d]), bsum);
    }
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        asum += __shfl_down_sync(mask, asum, offset);
        bsum += __shfl_down_sync(mask, bsum, offset);
    }
    if (lane == 0) {
        partial_a[warp] = asum;
        partial_b[warp] = bsum;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        asum = 0.0f;
        bsum = 0.0f;
        for (int w = 0; w < kWarps; ++w) {
            asum += partial_a[w];
            bsum += partial_b[w];
        }
        a_output[index] = asum;
        b_output[index] = bsum;
        const float beta_f = 1.0f / (1.0f + expf(-bsum));
        const float av = asum + dt_bias[head];
        const float softplus = av > 20.0f ? av : log1pf(expf(av));
        beta_trace[index] = __bfloat162float(__float2bfloat16_rn(beta_f));
        g_trace[index] = -expf(a_log[head]) * softplus;
    }
}

bool control_aligned(const void* a,const void* b,const void* c) {
    return ((reinterpret_cast<std::uintptr_t>(a)|reinterpret_cast<std::uintptr_t>(b)|
             reinterpret_cast<std::uintptr_t>(c))&15u)==0;
}

bool gdn_control_staged_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_GDN_CONTROL_STAGED");
        if(!value||std::strcmp(value,"1")==0) return true;
        if(std::strcmp(value,"0")==0) return false;
        throw std::invalid_argument("NINFER_EXL3_GDN_CONTROL_STAGED must be 0 or 1");
    }();
    return enabled;
}

// Tiled twin of control_fused_kernel for prefill: one CTA owns R rows x H
// heads. Thread t keeps, for every (row, head) pair, the same ascending fmaf
// chain over d = t, t+128, ... and the same shuffle and four-warp reductions,
// while each loaded activation and weight element now feeds R*H chains
// instead of one, removing the repeated L2 traversal of rows and weights.
bool gdn_control_tiled_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_GDN_CONTROL_TILED");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

bool gdn_control_decode_tiled_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_GDN_CONTROL_DECODE_TILED");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

template<int R,int H>
__global__ void __launch_bounds__(128) control_fused_tiled_kernel(
    const std::uint16_t* input,const std::uint16_t* a_weight,const std::uint16_t* b_weight,
    const float* a_log,const float* dt_bias,float* a_output,float* b_output,
    float* beta_trace,float* g_trace,int rows) {
    constexpr int kThreads=128;
    constexpr int kWarps=kThreads/32;
    static_assert(kHeads%H==0);
    __shared__ float partial_a[R*H][kWarps];
    __shared__ float partial_b[R*H][kWarps];
    const int tid=static_cast<int>(threadIdx.x);
    const int lane=tid&31;
    const int warp=tid>>5;
    constexpr int kHeadGroups=kHeads/H;
    const int row0=(static_cast<int>(blockIdx.x)/kHeadGroups)*R;
    const int head0=(static_cast<int>(blockIdx.x)%kHeadGroups)*H;
    if(row0>=rows) return;
    float asum[R][H],bsum[R][H];
    #pragma unroll
    for(int r=0;r<R;++r)
        #pragma unroll
        for(int h=0;h<H;++h) asum[r][h]=bsum[r][h]=0.0f;
    for(int d=tid;d<kHidden;d+=kThreads) {
        float x[R];
        #pragma unroll
        for(int r=0;r<R;++r)
            x[r]=row0+r<rows?half_value(input[(row0+r)*kHidden+d]):0.0f;
        #pragma unroll
        for(int h=0;h<H;++h) {
            const float a=half_value(a_weight[(head0+h)*kHidden+d]);
            const float b=half_value(b_weight[(head0+h)*kHidden+d]);
            #pragma unroll
            for(int r=0;r<R;++r) {
                asum[r][h]=fmaf(x[r],a,asum[r][h]);
                bsum[r][h]=fmaf(x[r],b,bsum[r][h]);
            }
        }
    }
    constexpr unsigned mask=0xffffffffu;
    #pragma unroll
    for(int r=0;r<R;++r) {
        #pragma unroll
        for(int h=0;h<H;++h) {
            float a=asum[r][h],b=bsum[r][h];
            for(int offset=16;offset>0;offset>>=1) {
                a+=__shfl_down_sync(mask,a,offset);
                b+=__shfl_down_sync(mask,b,offset);
            }
            if(lane==0) {
                partial_a[r*H+h][warp]=a;
                partial_b[r*H+h][warp]=b;
            }
        }
    }
    __syncthreads();
    if(tid<R*H) {
        const int r=tid/H;
        const int h=tid%H;
        const int row=row0+r;
        if(row<rows) {
            const int head=head0+h;
            const int index=row*kHeads+head;
            float a=0.0f,b=0.0f;
            for(int w=0;w<kWarps;++w) {
                a+=partial_a[tid][w];
                b+=partial_b[tid][w];
            }
            a_output[index]=a;
            b_output[index]=b;
            const float beta_f=1.0f/(1.0f+expf(-b));
            const float av=a+dt_bias[head];
            const float softplus=av>20.0f?av:log1pf(expf(av));
            beta_trace[index]=__bfloat162float(__float2bfloat16_rn(beta_f));
            g_trace[index]=-expf(a_log[head])*softplus;
        }
    }
}

// Two verifier rows share each represented coefficient load. Each row keeps
// the canonical per-thread FMA and warp/CTA reduction order.
__global__ void control_fused_row_pair_kernel(const std::uint16_t* input,
                                     const std::uint16_t* a_weight,
                                     const std::uint16_t* b_weight,
                                     const float* a_log, const float* dt_bias,
                                     float* a_output, float* b_output,
                                     float* beta_trace, float* g_trace, int rows) {
    __shared__ float partial_a[2][4], partial_b[2][4];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int head = static_cast<int>(blockIdx.x) % kHeads;
    const int first = (static_cast<int>(blockIdx.x) / kHeads) * 2;
    const int count = min(2, rows - first);
    const int weight_base = head * kHidden;
    float asum0 = 0.0f, bsum0 = 0.0f;
    float asum1 = 0.0f, bsum1 = 0.0f;
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += 128) {
        const float aw = half_value(a_weight[weight_base + d]);
        const float bw = half_value(b_weight[weight_base + d]);
        const float x0 = half_value(input[first * kHidden + d]);
        asum0 = fmaf(x0, aw, asum0);
        bsum0 = fmaf(x0, bw, bsum0);
        if (count == 2) {
            const float x1 = half_value(input[(first + 1) * kHidden + d]);
            asum1 = fmaf(x1, aw, asum1);
            bsum1 = fmaf(x1, bw, bsum1);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        asum0 += __shfl_down_sync(0xffffffffu, asum0, offset);
        bsum0 += __shfl_down_sync(0xffffffffu, bsum0, offset);
        asum1 += __shfl_down_sync(0xffffffffu, asum1, offset);
        bsum1 += __shfl_down_sync(0xffffffffu, bsum1, offset);
    }
    if (lane == 0) {
        partial_a[0][warp] = asum0; partial_b[0][warp] = bsum0;
        partial_a[1][warp] = asum1; partial_b[1][warp] = bsum1;
    }
    __syncthreads();
    if (warp == 0 && lane == 0) {
        for (int r = 0; r < count; ++r) {
            float asum = 0.0f, bsum = 0.0f;
            for (int w = 0; w < 4; ++w) {
                asum += partial_a[r][w]; bsum += partial_b[r][w];
            }
            const int index = (first + r) * kHeads + head;
            a_output[index] = asum; b_output[index] = bsum;
            const float beta_f = 1.0f / (1.0f + expf(-bsum));
            const float av = asum + dt_bias[head];
            const float softplus = av > 20.0f ? av : log1pf(expf(av));
            beta_trace[index] = __bfloat162float(__float2bfloat16_rn(beta_f));
            g_trace[index] = -expf(a_log[head]) * softplus;
        }
    }
}

void record(cudaEvent_t event, cudaStream_t stream) { check(cudaEventRecord(event, stream), "record EXL3 GDN timing event"); }

} // namespace

void gopt_gdn_control_row_pair_fixture(bool paired,const std::uint16_t* input,
    const std::uint16_t* a_weight,const std::uint16_t* b_weight,
    const float* a_log,const float* dt_bias,float* a_output,float* b_output,
    float* beta_trace,float* g_trace,int rows,cudaStream_t stream) {
    if(rows<1 || rows>16) throw std::invalid_argument("GOPT control fixture rows");
    if(paired) control_fused_row_pair_kernel<<<((rows+1)/2)*kHeads,128,0,stream>>>(
        input,a_weight,b_weight,a_log,dt_bias,a_output,b_output,beta_trace,g_trace,rows);
    else exl3_launch_small(control_fused_kernel,dim3(rows*kHeads),dim3(128),0,stream,
        input,a_weight,b_weight,a_log,dt_bias,a_output,b_output,beta_trace,g_trace,rows);
    check(cudaGetLastError(),"GOPT control fixture launch");
}

void exl3_gdn_residual_norm(const std::uint16_t* left,const std::uint16_t* right,
    const std::uint16_t* weight,std::uint16_t* residual,std::uint16_t* normalized,
    int rows,bool fused,cudaStream_t stream) {
    exl3_require_residual_norm_extents(rows,
        {reinterpret_cast<std::uintptr_t>(left),reinterpret_cast<std::uintptr_t>(right),
         reinterpret_cast<std::uintptr_t>(weight),reinterpret_cast<std::uintptr_t>(residual),
         reinterpret_cast<std::uintptr_t>(normalized)});
    if(fused)
        exl3_launch_small(rms_norm_f16_kernel<true>,dim3(rows),dim3(512),512*sizeof(float),stream,left,
            weight,normalized,rows,kHidden,right,residual);
    else {
        exl3_launch_small(residual_kernel,dim3((rows*kHidden+255)/256),dim3(256),0,stream,left,right,residual,rows*kHidden);
        check(cudaGetLastError(),"launch GDN canonical residual");
        exl3_launch_small(rms_norm_f16_kernel<>,dim3(rows),dim3(512),512*sizeof(float),stream,residual,weight,normalized,rows,kHidden);
    }
    check(cudaGetLastError(),"launch GDN residual norm operator");
}

void exl3_gdn_stage_fusion_fixture(
    const Exl3GdnStageFusionFixtureView& view,cudaStream_t stream,bool pair_columns) {
    if(!view.valid())
        throw std::invalid_argument("GDN stage-fusion fixture extent");
    gdn_recurrence_sm120_kernel<<<kHeads*(kHeadDim/4),4*32,0,stream>>>(
        view.q,view.k,view.v,view.g,view.beta,view.canonical_state,
        view.canonical_output,view.rows);
    check(cudaGetLastError(),"launch canonical GDN stage-fusion fixture");
    gdn_prefill_normalize_kernel<<<view.rows*kKeyHeads,32,0,stream>>>(
        view.q,view.k,view.g,view.normalized_q,view.normalized_k,view.alpha);
    check(cudaGetLastError(),"launch normalized GDN stage-fusion fixture");
    if(pair_columns)
        gdn_recurrence_prefill_resident_pair_columns_kernel<false><<<
            kHeads*(kHeadDim/8),128,0,stream>>>(view.normalized_q,view.normalized_k,
                view.v,view.alpha,view.beta,view.fused_state,view.fused_output,view.rows);
    else gdn_recurrence_prefill_resident_kernel<<<
        kHeads*(kHeadDim/4),4*32,0,stream>>>(
        view.normalized_q,view.normalized_k,view.v,view.alpha,view.beta,
        view.fused_state,view.fused_output,view.rows);
    check(cudaGetLastError(),"launch resident GDN stage-fusion fixture");
}

void gopt_gdn_conv_fixture(bool fused,const std::uint16_t* input,
    const std::uint16_t* weight,std::uint16_t* state,std::uint16_t* conv_input,
    std::uint16_t* q,std::uint16_t* k,std::uint16_t* v,std::uint16_t* packed,
    std::uint16_t* trace,int rows,cudaStream_t stream,bool decode_fused) {
    if(!decode_fused)exl3_launch_small(transpose_f16_to_bf16_kernel,dim3((rows*kQkv+255)/256),dim3(256),0,stream,input,conv_input,rows,kQkv);
    launch_gopt_conv(fused,decode_fused,input,weight,state,conv_input,q,k,v,packed,trace,rows,stream);
    if(!decode_fused)exl3_launch_small(pack_qkv_bf16_kernel,dim3((rows*kQkv+255)/256),dim3(256),0,stream,q,k,v,packed,rows);
    check(cudaGetLastError(),"GOPT conv fixture");
    if(!fused)exl3_launch_small(copy_conv_state_trace_kernel,dim3((kQkv*3+255)/256),dim3(256),0,stream,state,trace);
    check(cudaGetLastError(),"GOPT conv trace fixture");
}
void gopt_gdn_pack_fixture(bool fused,const std::uint16_t* core,
    const std::uint16_t* norm,std::uint16_t* trace,std::uint16_t* projection,
    int rows,cudaStream_t stream) {
    if(fused)gopt_gdn_output_pack_kernel<<<(rows*kZ+255)/256,256,0,stream>>>(core,norm,trace,projection,rows*kZ);
    else {
        exl3_launch_small(pack_heads_bf16_kernel,dim3((rows*kZ+255)/256),dim3(256),0,stream,core,trace,rows);
        exl3_launch_small(transpose_bf16_to_f16_kernel,dim3((rows*kZ+255)/256),dim3(256),0,stream,norm,projection,rows,kZ);
    }
    check(cudaGetLastError(),"GOPT output pack fixture");
}

std::size_t Exl3GdnLayer::shared_scratch_bytes(int rows) {
    if(rows<=0) throw std::invalid_argument("GDN shared scratch rows");
    std::size_t per_row=(4*kHeads+2*kKeyHeads*kHeadDim+kHeads)*sizeof(float);
    for(std::size_t i=0;i<kHalfBufferFeatures.size();++i)
        if(transient_half(i)) per_row+=kHalfBufferFeatures[i]*sizeof(std::uint16_t);
    return static_cast<std::size_t>(rows)*per_row;
}

std::size_t Exl3GdnLayer::wide_scratch_bytes(int rows) {
    if (rows <= 0) throw std::invalid_argument("GDN wide scratch rows");
    return split_wide_bytes(rows);
}

std::size_t Exl3GdnLayer::fixed_owner_metadata_required() noexcept {
    return bounded_shared_allocation_bytes<Exl3GdnLayer>()+
        6*Exl3CudaLinearWorkspace::metadata_bytes()+sizeof(DeviceArena)+
        34*Exl3LayerBufferRetirement::record_bytes();
}
std::size_t Exl3GdnLayer::fixed_owner_metadata_bytes() const noexcept {
    auto bytes=bounded_shared_allocation_bytes<Exl3GdnLayer>();
    for(const auto& owner:buffer_retirements_)if(owner)bytes+=Exl3LayerBufferRetirement::record_bytes();
    for(const auto* workspace:linear_workspaces_)if(workspace)bytes+=Exl3CudaLinearWorkspace::metadata_bytes();
    if(op_workspace_)bytes+=sizeof(DeviceArena);
    return bytes;
}
std::size_t Exl3GdnLayer::workspace_bytes_required(int rows,bool borrowed_accumulation,
    bool borrowed_transform,bool borrowed_scratch,bool split_wide,bool resident) {
    if(rows<=0 || rows>1024)throw std::invalid_argument("EXL3 GDN max_rows must be 1..1024");
    std::size_t bytes=0;
    const auto add=[&](std::size_t extent) {
        if(extent)bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(bytes,extent);
    };
    for(const auto shape:std::array<std::pair<int,int>,6>{{{kHidden,kQkv},{kHidden,kZ},
        {kZ,kHidden},{kHidden,kIntermediate},{kHidden,kIntermediate},{kIntermediate,kHidden}}})
        add(Exl3LinearWorkspaceRequirements::derive(shape.first,shape.second,rows,
            borrowed_transform,borrowed_accumulation).owned_bytes);
    for(std::size_t i=0;i<kHalfBufferFeatures.size();++i) {
        if(borrowed_scratch && transient_half(i))continue;
        const auto extent_rows=split_wide && is_split_half(i)?std::min(rows,kPrivateRetentionRows):rows;
        add(static_cast<std::size_t>(extent_rows)*kHalfBufferFeatures[i]*sizeof(std::uint16_t));
    }
    add(static_cast<std::size_t>(rows)*(borrowed_scratch?2:6)*kHeads*sizeof(float));
    if(resident && !borrowed_scratch)add(static_cast<std::size_t>(rows)*(2*kKeyHeads*kHeadDim+kHeads)*sizeof(float));
    add(kStateBytes);add(kStateBytes);add(kConvStateStorageBytes);add(kConvStateBytes);add(1u<<20);
    return bytes;
}

Exl3GdnLayer::Exl3GdnLayer(const Exl3GdnLayerWeights& weights, int max_rows,
                           Exl3CudaAccumulationView accumulation, Exl3CudaTransformView transformed,
                           Exl3CudaLayerScratchView scratch,
                           Exl3GdnWideScratchView wide_scratch,
                           Exl3CudaReconstructGemmWorkspace* reconstruct_gemm,
                           Exl3VeriCacheServingCoordinator* constructor_authority,
                           unsigned buffer_fault_for_test,
                           std::shared_ptr<const Exl3GdnImmutableCoefficients>
                               immutable_coefficients,
                           Exl3CudaAccumulationView paired_up_accumulation,
                           Exl3CudaTransformView paired_up_transformed)
    : weights_(immutable_coefficients?immutable_coefficients->weights:weights),
      max_rows_(max_rows), reconstruct_gemm_(reconstruct_gemm),
      immutable_coefficients_(std::move(immutable_coefficients)) {
    if(immutable_coefficients_ &&
       !immutable_coefficients_->same_representation(weights))
        throw std::invalid_argument("GDN immutable coefficient representation mismatch");
    const char* resident = std::getenv("NINFER_EXL3_PREFILL_GDN_RESIDENT");
    const char* wide_prefill = std::getenv("NINFER_EXL3_WIDE_PREFILL");
    const char* staged_prefill = std::getenv("NINFER_EXL3_PREFILL_STAGED_REDUCTION");
    const char* shared_accum = std::getenv("NINFER_EXL3_SHARED_ACCUM");
    const bool qualified_default_group = exl3_prefill_qualified_default_group(
        max_rows_, wide_prefill && std::strcmp(wide_prefill, "1") == 0,
        staged_prefill && std::strcmp(staged_prefill, "1") == 0,
        shared_accum && std::strcmp(shared_accum, "1") == 0);
    prefill_resident_ = resident
        ? std::string(resident) == "1" :
          (qualified_default_group && scratch.data != nullptr);
    const char* resident_pair_columns=std::getenv(
        "NINFER_EXL3_PREFILL_GDN_RESIDENT_PAIR_COLUMNS");
    if(resident_pair_columns && std::strcmp(resident_pair_columns,"0")!=0 &&
        std::strcmp(resident_pair_columns,"1")!=0)
        throw std::invalid_argument("GDN resident pair-columns must be 0 or 1");
    prefill_resident_pair_columns_=prefill_resident_ && (resident_pair_columns
        ? std::strcmp(resident_pair_columns,"1")==0
        : (qualified_default_group && scratch.data!=nullptr));
    const char* resident_pair_vector_io=std::getenv(
        "NINFER_EXL3_PREFILL_GDN_RESIDENT_PAIR_VECTOR_IO");
    if(resident_pair_vector_io && std::strcmp(resident_pair_vector_io,"0")!=0 &&
        std::strcmp(resident_pair_vector_io,"1")!=0)
        throw std::invalid_argument("GDN resident pair vector IO must be 0 or 1");
    prefill_resident_pair_vector_io_=prefill_resident_pair_columns_ &&
        resident_pair_vector_io && std::strcmp(resident_pair_vector_io,"1")==0;
    const char* resident_quad_columns=std::getenv(
        "NINFER_EXL3_PREFILL_GDN_RESIDENT_QUAD_COLUMNS");
    if(resident_quad_columns && std::strcmp(resident_quad_columns,"0")!=0 &&
        std::strcmp(resident_quad_columns,"1")!=0)
        throw std::invalid_argument("GDN resident quad-columns must be 0 or 1");
    prefill_resident_quad_columns_=prefill_resident_pair_columns_ &&
        resident_quad_columns && std::strcmp(resident_quad_columns,"1")==0;
    if(prefill_resident_quad_columns_&&prefill_resident_pair_vector_io_)
        throw std::invalid_argument(
            "GDN resident quad columns and pair vector IO are separate controls");
    // Validate even if this layer's diagnostic or shape route excludes pairing.
    (void)exl3_equal_transform_reuse_enabled(
        std::getenv("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS"));
    const char* dual_input_transform = std::getenv(
        "NINFER_EXL3_GDN_FUSED_GATE_UP_TRANSFORM");
    if(dual_input_transform && std::strcmp(dual_input_transform,"0")!=0 &&
       std::strcmp(dual_input_transform,"1")!=0)
        throw std::invalid_argument("GDN fused gate/up transform must be 0 or 1");
    fused_gate_up_transform_=dual_input_transform && std::strcmp(dual_input_transform,"1")==0;
    const char* small_m_fused_gate_up=std::getenv(
        "NINFER_EXL3_SMALL_M_FUSED_GATE_UP_TRANSFORM");
    if(small_m_fused_gate_up && std::strcmp(small_m_fused_gate_up,"0")!=0 &&
       std::strcmp(small_m_fused_gate_up,"1")!=0)
        throw std::invalid_argument(
            "small-M fused gate/up transform must be 0 or 1");
    small_m_fused_gate_up_transform_=small_m_fused_gate_up &&
        std::strcmp(small_m_fused_gate_up,"1")==0;
    const char* bulk_mlp_fused_down=std::getenv(
        "NINFER_EXL3_FAST_GDN_BULK_MLP_FUSED_DOWN");
    if(bulk_mlp_fused_down && std::strcmp(bulk_mlp_fused_down,"0")!=0 &&
       std::strcmp(bulk_mlp_fused_down,"1")!=0)
        throw std::invalid_argument("GDN bulk MLP fused down must be 0 or 1");
    bulk_mlp_fused_down_=bulk_mlp_fused_down &&
        std::strcmp(bulk_mlp_fused_down,"1")==0;
    const char* bulk_mlp_fused_residual=std::getenv(
        "NINFER_EXL3_FAST_GDN_BULK_MLP_FUSED_RESIDUAL");
    if(bulk_mlp_fused_residual &&
       std::strcmp(bulk_mlp_fused_residual,"0")!=0 &&
       std::strcmp(bulk_mlp_fused_residual,"1")!=0)
        throw std::invalid_argument("GDN bulk MLP fused residual must be 0 or 1");
    bulk_mlp_fused_residual_=bulk_mlp_fused_residual &&
        std::strcmp(bulk_mlp_fused_residual,"1")==0;
    const char* bulk_mlp_weight_prefetch=std::getenv(
        "NINFER_EXL3_FAST_GDN_BULK_MLP_WEIGHT_PREFETCH");
    if (bulk_mlp_weight_prefetch &&
        std::strcmp(bulk_mlp_weight_prefetch,"0")!=0 &&
        std::strcmp(bulk_mlp_weight_prefetch,"1")!=0)
        throw std::invalid_argument("GDN bulk MLP weight prefetch must be 0 or 1");
    bulk_mlp_weight_prefetch_=bulk_mlp_weight_prefetch &&
        std::strcmp(bulk_mlp_weight_prefetch,"1")==0;
    const char* fused_residual=std::getenv("NINFER_EXL3_GDN_FUSED_RESIDUAL_NORM");
    if(fused_residual && std::strcmp(fused_residual,"0")!=0 && std::strcmp(fused_residual,"1")!=0)
        throw std::invalid_argument("GDN fused residual norm must be 0 or 1");
    fused_residual_norm_=fused_residual && std::strcmp(fused_residual,"1")==0;
    const char* fast_decode_conv = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_GDN_DECODE_CONV");
    if(fast_decode_conv && std::strcmp(fast_decode_conv,"0")!=0 &&
       std::strcmp(fast_decode_conv,"1")!=0)
        throw std::invalid_argument(
            "same-weight FP16-KV GDN decode convolution must be 0 or 1");
    fast_same_weights_fp16kv_gdn_decode_conv_ = fast_decode_conv &&
        std::strcmp(fast_decode_conv,"1")==0;
    const char* fast_gdn_m1_gate_up_pair = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_GDN_M1_GATE_UP_PAIR");
    if(fast_gdn_m1_gate_up_pair &&
       std::strcmp(fast_gdn_m1_gate_up_pair,"0")!=0 &&
       std::strcmp(fast_gdn_m1_gate_up_pair,"1")!=0)
        throw std::invalid_argument(
            "same-weight FP16-KV GDN M1 gate/up pair must be 0 or 1");
    fast_same_weights_fp16kv_gdn_m1_gate_up_pair_ =
        fast_gdn_m1_gate_up_pair &&
        std::strcmp(fast_gdn_m1_gate_up_pair,"1")==0;
    const auto pair_requirement = Exl3LinearWorkspaceRequirements::derive(
        kHidden,kIntermediate,max_rows_,paired_up_transformed.data!=nullptr,
        paired_up_accumulation.data!=nullptr);
    if((paired_up_accumulation.data==nullptr && paired_up_accumulation.bytes!=0) ||
       (paired_up_accumulation.data!=nullptr &&
        paired_up_accumulation.bytes<pair_requirement.accumulation_bytes))
        throw std::invalid_argument("GDN paired up accumulation capacity");
    if((paired_up_transformed.data==nullptr && paired_up_transformed.bytes!=0) ||
       (paired_up_transformed.data!=nullptr &&
        paired_up_transformed.bytes<pair_requirement.transformed_bytes))
        throw std::invalid_argument("GDN paired up transform capacity");
    if(paired_up_accumulation.data && accumulation.data &&
       paired_up_accumulation.data==accumulation.data)
        throw std::invalid_argument("GDN paired up accumulation must be disjoint");
    if(paired_up_transformed.data && transformed.data &&
       paired_up_transformed.data==transformed.data)
        throw std::invalid_argument("GDN paired up transform must be disjoint");
    if(fast_same_weights_fp16kv_gdn_m1_gate_up_pair_ && accumulation.data &&
       !paired_up_accumulation.data)
        throw std::invalid_argument(
            "GDN M1 gate/up pair requires a distinct borrowed accumulation view");
    if(fast_same_weights_fp16kv_gdn_m1_gate_up_pair_ && transformed.data &&
       !paired_up_transformed.data)
        throw std::invalid_argument(
            "GDN M1 gate/up pair requires a distinct borrowed transform view");
    dual_input_transform = std::getenv(
        "NINFER_EXL3_GDN_DUAL_INPUT_TRANSFORM");
    if (dual_input_transform &&
        std::strcmp(dual_input_transform, "0") != 0 &&
        std::strcmp(dual_input_transform, "1") != 0)
        throw std::invalid_argument("GDN dual input transform must be 0 or 1");
    dual_input_transform_ = dual_input_transform &&
        std::strcmp(dual_input_transform, "1") == 0;
    if (max_rows_ <= 0 || max_rows_ > 1024) throw std::invalid_argument("EXL3 GDN max_rows must be 1..1024");
    if((!scratch.data && scratch.bytes) || (scratch.data && scratch.bytes<shared_scratch_bytes(max_rows_)))
        throw std::invalid_argument("GDN shared scratch capacity");
    if ((!wide_scratch.data && wide_scratch.bytes) ||
        (wide_scratch.data && wide_scratch.bytes < split_wide_bytes(max_rows_)))
        throw std::invalid_argument("GDN wide scratch capacity");
    split_wide_storage_ = wide_scratch.data != nullptr;
    private_row_capacity_ = split_wide_storage_
        ? std::min(max_rows_, kPrivateRetentionRows) : max_rows_;
    scratch_reuse_={static_cast<std::size_t>(max_rows_),
        static_cast<std::size_t>(private_row_capacity_),scratch.bytes,
        wide_scratch.bytes};
    if(!scratch_reuse_.valid())
        throw std::invalid_argument("GDN scratch reuse contract");
    if (split_wide_storage_) {
        auto* wide_cursor = static_cast<std::byte*>(wide_scratch.data);
        for (std::size_t split = 0; split < kSplitHalfIndices.size(); ++split) {
            const std::size_t index = kSplitHalfIndices[split];
            wide_half_buffers_[split] = reinterpret_cast<std::uint16_t*>(wide_cursor);
            wide_half_buffer_bytes_[split]=static_cast<std::size_t>(max_rows_)*
                kHalfBufferFeatures[index]*sizeof(std::uint16_t);
            wide_cursor += wide_half_buffer_bytes_[split];
        }
    }
    auto* cursor=static_cast<std::byte*>(scratch.data);
    const auto borrow=[&](std::size_t bytes) {auto* p=cursor;cursor+=bytes;return p;};
    const std::array<std::pair<int, int>, 6> shapes = {{{kHidden, kQkv}, {kHidden, kZ},
        {kZ, kHidden}, {kHidden, kIntermediate}, {kHidden, kIntermediate}, {kIntermediate, kHidden}}};
    std::size_t required_linear_bytes=0;
    for(std::size_t i=0;i<shapes.size();++i) {
        const auto& shape=shapes[i];
        const bool paired_workspace=
            fast_same_weights_fp16kv_gdn_m1_gate_up_pair_ && i==4;
        const auto& workspace_accumulation=paired_workspace && paired_up_accumulation.data
            ? paired_up_accumulation : accumulation;
        const auto& workspace_transformed=paired_workspace && paired_up_transformed.data
            ? paired_up_transformed : transformed;
        const auto required=Exl3LinearWorkspaceRequirements::derive(shape.first,shape.second,max_rows_,
            workspace_transformed.data!=nullptr,workspace_accumulation.data!=nullptr);
        if(required.owned_bytes)required_linear_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            required_linear_bytes,required.owned_bytes);
    }
    std::array<std::size_t,kHalfBufferFeatures.size()> half_bytes{};
    const auto required_workspace_bytes=workspace_bytes_required(max_rows_,accumulation.data!=nullptr,
        transformed.data!=nullptr,scratch.data!=nullptr,split_wide_storage_,prefill_resident_ || gaming_[Gopt::GdnSmallResident]);
    for(std::size_t i=0;i<half_bytes.size();++i) {
        const auto rows=split_wide_storage_ && is_split_half(i)?private_row_capacity_:max_rows_;
        half_bytes[i]=static_cast<std::size_t>(rows)*kHalfBufferFeatures[i]*sizeof(std::uint16_t);
    }
    half_buffer_bytes_=half_bytes;
    const auto control_bytes=static_cast<std::size_t>(max_rows_)*kHeads*sizeof(float);
    const auto normalized_bytes=static_cast<std::size_t>(max_rows_)*(2*kKeyHeads*kHeadDim+kHeads)*sizeof(float);
    float_buffer_bytes_.fill(control_bytes);
    prefill_normalized_bytes_=(prefill_resident_ || gaming_[Gopt::GdnSmallResident])?normalized_bytes:0;
    constexpr std::size_t op_capacity=1u<<20;
    try {
        for (std::size_t i = 0; i < linear_workspaces_.size(); ++i) {
            std::optional<RetainedDeviceLedger::Ticket> device;
            std::optional<RetainedDescriptorLedger::Ticket> metadata;
            const bool paired_workspace=
                fast_same_weights_fp16kv_gdn_m1_gate_up_pair_ && i==4;
            const auto& workspace_accumulation=paired_workspace && paired_up_accumulation.data
                ? paired_up_accumulation : accumulation;
            const auto& workspace_transformed=paired_workspace && paired_up_transformed.data
                ? paired_up_transformed : transformed;
            const auto plan=Exl3LinearWorkspaceRequirements::derive(shapes[i].first,shapes[i].second,max_rows_,
                workspace_transformed.data!=nullptr,workspace_accumulation.data!=nullptr);
            if(constructor_authority && plan.owned_bytes) {
                auto credits=constructor_authority->reserve_constructor_credits(plan.owned_bytes,Exl3CudaLinearWorkspace::metadata_bytes());
                device.emplace(std::move(credits.device));metadata.emplace(std::move(credits.metadata));
            }
            linear_workspaces_[i] = new Exl3CudaLinearWorkspace(
                shapes[i].first, shapes[i].second, max_rows_, false,
                i == 3 || i == 4, i == 5, i == 2, i == 1, true, false,
                workspace_accumulation, workspace_transformed, i == 0,
                false,false,std::move(device),std::move(metadata));
            workspace_bytes_ += linear_workspaces_[i]->workspace_bytes();
        }
        if(workspace_bytes_!=required_linear_bytes)throw std::runtime_error("GDN linear allocation requirement mismatch");
        for(auto& owner:buffer_retirements_)owner.emplace(+[](void* pointer,int device) noexcept -> int {
            int current=-1;const auto query=cudaGetDevice(&current);
            if(query!=cudaSuccess)return static_cast<int>(query);
            if(current!=device)return static_cast<int>(cudaErrorInvalidDevice);
            return static_cast<int>(cudaFree(pointer));
        });
        int buffer_device=-1;check(cudaGetDevice(&buffer_device),"capture GDN buffer device");
        const auto allocate_buffer=[&](unsigned slot,void** pointer,std::size_t bytes,const char* label) {
            std::optional<Exl3VeriCacheServingCoordinator::ConstructorCredits> credits;
            if(constructor_authority)credits.emplace(constructor_authority->reserve_constructor_credits(
                bytes,Exl3LayerBufferRetirement::record_bytes()));
            check(cudaMalloc(pointer,bytes),label);
            buffer_retirements_[slot]->adopt(*pointer,bytes,buffer_device);
            if(credits)buffer_retirements_[slot]->attach(std::move(credits->device),std::move(credits->metadata));
            if(buffer_fault_for_test==1 || buffer_fault_for_test==3) {
                if(buffer_fault_for_test==1)buffer_retirements_[slot]->fail_cleanup_for_test(static_cast<int>(cudaErrorUnknown));
                throw std::runtime_error("injected layer buffer postallocation failure");
            }
        };
        for (std::size_t i=0;i<half_buffers_.size();++i) {
            const auto bytes=half_bytes[i];
            if(scratch.data && transient_half(i)) {
                half_buffers_[i]=reinterpret_cast<std::uint16_t*>(borrow(bytes));
            } else {
                allocate_buffer(static_cast<unsigned>(i),reinterpret_cast<void**>(&half_buffers_[i]),bytes,
                      "allocate EXL3 GDN half workspace");
                half_buffer_owned_[i]=true;workspace_bytes_+=bytes;
            }
        }
        for (std::size_t i=0;i<float_buffers_.size();++i) {
            const auto bytes=control_bytes;
            if(scratch.data && i<4) {
                float_buffers_[i]=reinterpret_cast<float*>(borrow(bytes));
            } else {
                allocate_buffer(22+static_cast<unsigned>(i),reinterpret_cast<void**>(&float_buffers_[i]),bytes,
                      "allocate EXL3 GDN control workspace");
                float_buffer_owned_[i]=true;workspace_bytes_+=bytes;
            }
        }
        if (prefill_resident_ || gaming_[Gopt::GdnSmallResident]) {
            const auto bytes=normalized_bytes;
            if(scratch.data) {
                prefill_normalized_=reinterpret_cast<float*>(borrow(bytes));
            } else {
                allocate_buffer(28,reinterpret_cast<void**>(&prefill_normalized_),bytes,
                      "allocate prefill GDN normalized Q/K/alpha");
                prefill_normalized_owned_=true;workspace_bytes_+=bytes;
            }
        }
        allocate_buffer(29,reinterpret_cast<void**>(&recurrent_state_),kStateBytes,"allocate EXL3 GDN recurrent state");
        allocate_buffer(30,reinterpret_cast<void**>(&recurrent_state_before_),kStateBytes,"allocate EXL3 GDN state trace");
        allocate_buffer(31,reinterpret_cast<void**>(&conv_state_),kConvStateStorageBytes,"allocate EXL3 GDN convolution state");
        allocate_buffer(32,reinterpret_cast<void**>(&conv_state_trace_),kConvStateBytes,"allocate EXL3 GDN convolution state trace");
        workspace_bytes_ += Exl3GdnLayer::kStateBytes * 2 + Exl3GdnLayer::kConvStateStorageBytes + Exl3GdnLayer::kConvStateBytes;
        const auto expected_with_arena=Exl3LinearWorkspaceRequirements::append_owned_bytes(workspace_bytes_,op_capacity);
        void* operation_storage=nullptr;
        allocate_buffer(33,&operation_storage,op_capacity,"allocate GDN operation arena backing");
        if(buffer_fault_for_test==2 || buffer_fault_for_test==4) {
            if(buffer_fault_for_test==2)buffer_retirements_[33]->fail_cleanup_for_test(static_cast<int>(cudaErrorUnknown));
            throw std::runtime_error("injected GDN arena wrapper allocation failure");
        }
        op_workspace_ = new DeviceArena(DeviceSpan{operation_storage,op_capacity});
        if(op_workspace_->capacity()!=op_capacity)throw std::runtime_error("GDN operation arena requirement mismatch");
        workspace_bytes_=expected_with_arena;
        if(workspace_bytes_!=required_workspace_bytes)throw std::runtime_error("GDN complete allocation requirement mismatch");
        reset();
    } catch (...) {
        for(auto& owner:buffer_retirements_)if(owner)owner->retire();
        delete op_workspace_;
        for (auto*& workspace : linear_workspaces_) Exl3CudaLinearWorkspace::retire_slot(workspace);
        throw;
    }
}

Exl3GdnLayer::~Exl3GdnLayer() {
    for(auto& owner:buffer_retirements_)if(owner)owner->retire();
    delete op_workspace_;
    for (auto*& workspace : linear_workspaces_) Exl3CudaLinearWorkspace::retire_slot(workspace);
}

void Exl3GdnLayer::prepare_continuation_graph_qkvz_concurrency(
    Exl3GdnGraphQkvzConcurrencyView view) {
    const bool any = view.z_stream || view.fork || view.z_done || view.z_workspace;
    if (!any) return;
    if (!view.complete() || graph_qkvz_concurrency_.complete() || max_rows_ < 8)
        throw std::invalid_argument("fixed-B8 GDN QKV/Z graph concurrency resources");
    const auto original_route = linear_workspaces_[1]->dispatch_name(
        weights_.z_metadata, 8, Exl3CudaLinearAdmission::target_continuation_z);
    const auto private_route = view.z_workspace->dispatch_name(
        weights_.z_metadata, 8, Exl3CudaLinearAdmission::target_continuation_z);
    if (std::strcmp(original_route, private_route) != 0)
        throw std::invalid_argument("fixed-B8 GDN private Z route mismatch");
    graph_qkvz_concurrency_ = view;
}

void Exl3GdnLayer::prepare_eager_mlp_gateup_concurrency(
    Exl3MlpGateUpConcurrencyView view) {
    const bool any = view.up_stream || view.fork || view.up_done ||
        view.up_workspace;
    if (!any) return;
    if (!view.complete() || eager_mlp_gateup_concurrency_.complete() ||
        max_rows_ < 8)
        throw std::invalid_argument(
            "eager GDN MLP gate/up concurrency resources");
    for (int rows = 1; rows <= 8; ++rows) {
        const auto admission = rows == 1
            ? Exl3CudaLinearAdmission::ordinary
            : Exl3CudaLinearAdmission::target_continuation_gate_up;
        const auto original_route = linear_workspaces_[4]->dispatch_name(
            weights_.up_metadata, rows, admission);
        const auto private_route = view.up_workspace->dispatch_name(
            weights_.up_metadata, rows, admission);
        if (std::strcmp(original_route, private_route) != 0)
            throw std::invalid_argument(std::string(
                "eager GDN private up route mismatch rows=")+
                std::to_string(rows)+" original="+original_route+
                " private="+private_route);
    }
    eager_mlp_gateup_concurrency_ = view;
}

void Exl3GdnLayer::reset(cudaStream_t stream) {
    invalidate_continuation_history();
    retained_prefix_available_ = false;
    current_checkpoint_generation_ = 0;
    current_checkpoint_stream_ = nullptr;
    current_checkpoint_recurrent_ = nullptr;
    current_checkpoint_conv_ = nullptr;
    check(cudaMemsetAsync(recurrent_state_, 0, kStateBytes, stream), "reset EXL3 GDN recurrent state");
    check(cudaMemsetAsync(conv_state_, 0, kConvStateStorageBytes, stream), "reset EXL3 GDN convolution state");
    check(cudaMemsetAsync(conv_state_trace_, 0, kConvStateBytes, stream), "reset EXL3 GDN convolution state trace");
}

void Exl3GdnLayer::bind_recurrent_layout(
    std::shared_ptr<const void> model_owner,int model_layer) {
    if(!model_owner || !Exl3GdnRecurrentLayout::valid_layer(model_layer) ||
       recurrent_model_owner_ || model_layer_>=0 ||
       (immutable_coefficients_ &&
        !immutable_coefficients_->matches(model_owner,model_layer,weights_)))
        throw std::invalid_argument("EXL3 GDN recurrent layout binding");
    recurrent_model_owner_=std::move(model_owner);model_layer_=model_layer;
}

Exl3GdnContinuationHistoryView Exl3GdnLayer::continuation_history_view(
    std::shared_ptr<const void> storage_owner,int base_position,
    int first_row,int rows) const {
    if(!storage_owner || storage_owner.get()!=this || !recurrent_model_owner_ ||
       base_position<0 ||
       continuation_history_rows_<=0 || first_row<0 || rows<=0 ||
       first_row>continuation_history_rows_ ||
       rows>continuation_history_rows_-first_row)
        throw std::invalid_argument("GDN continuation history row/lifetime");
    const auto half_offset=[first_row](std::size_t features) {
        return static_cast<std::size_t>(first_row)*features;
    };
    Exl3GdnContinuationHistoryView result{
        std::move(storage_owner),recurrent_model_owner_,
        &continuation_history_generation_,continuation_history_generation_,
        model_layer_,base_position,continuation_history_rows_,first_row,rows,
        half_buffers_[3]+half_offset(kQkv),
        half_buffers_[4]+half_offset(kKeyHeads*kHeadDim),
        half_buffers_[5]+half_offset(kKeyHeads*kHeadDim),
        half_buffers_[6]+half_offset(kHeads*kHeadDim),
        float_buffers_[5]+half_offset(kHeads),
        float_buffers_[4]+half_offset(kHeads),
        half_buffers_[9]+half_offset(kHeads*kHeadDim)};
    if(!result.current())
        throw std::invalid_argument("GDN continuation history is stale");
    return result;
}

void Exl3GdnLayer::validate_checkpoint_storage(
    const Exl3GdnLayerCheckpoint& checkpoint) const {
    if (!checkpoint.layout_valid() || !recurrent_model_owner_ ||
        checkpoint.model_owner!=recurrent_model_owner_ ||
        checkpoint.model_layer!=model_layer_)
        throw std::invalid_argument("EXL3 GDN checkpoint layout/owner");
    if (checkpoint.recurrent_trace_alias &&
        checkpoint.recurrent_state_device != recurrent_state_before_)
        throw std::invalid_argument("EXL3 GDN recurrent trace checkpoint alias");
    const auto disjoint=[&](const void* pointer,std::size_t bytes) {
        return Exl3GdnLayerCheckpoint::ranges_disjoint(
                   checkpoint.recurrent_state_device,
                   Exl3GdnRecurrentLayout::recurrent_bytes,pointer,bytes) &&
            Exl3GdnLayerCheckpoint::ranges_disjoint(
                   checkpoint.conv_state_device,
                   Exl3GdnRecurrentLayout::convolution_storage_bytes,
                   pointer,bytes);
    };
    for(std::size_t i=0;i<half_buffers_.size();++i)
        if(!disjoint(half_buffers_[i],half_buffer_bytes_[i]))
            throw std::invalid_argument("EXL3 GDN checkpoint aliases half scratch");
    for(std::size_t i=0;i<wide_half_buffers_.size();++i)
        if(wide_half_buffers_[i] &&
           !disjoint(wide_half_buffers_[i],wide_half_buffer_bytes_[i]))
            throw std::invalid_argument("EXL3 GDN checkpoint aliases wide scratch");
    for(std::size_t i=0;i<float_buffers_.size();++i)
        if(!disjoint(float_buffers_[i],float_buffer_bytes_[i]))
            throw std::invalid_argument("EXL3 GDN checkpoint aliases float scratch");
    if(prefill_normalized_ &&
       !disjoint(prefill_normalized_,prefill_normalized_bytes_))
        throw std::invalid_argument("EXL3 GDN checkpoint aliases normalized scratch");
    const std::array<std::pair<const void*,std::size_t>,4> live_state{
        std::pair<const void*,std::size_t>{recurrent_state_,kStateBytes},
        {recurrent_state_before_,kStateBytes},
        {conv_state_,kConvStateStorageBytes},
        {conv_state_trace_,kConvStateBytes}};
    for(std::size_t i=0;i<live_state.size();++i)
        if(!(checkpoint.recurrent_trace_alias && i==1) &&
           !disjoint(live_state[i].first,live_state[i].second))
            throw std::invalid_argument("EXL3 GDN checkpoint aliases live state");
}

void Exl3GdnLayer::validate_saved_checkpoint(
    const Exl3GdnLayerCheckpoint& checkpoint,int expected_position) const {
    if (!checkpoint.saved_for(recurrent_model_owner_,model_layer_,expected_position,
                              checkpoint.generation,this))
        throw std::invalid_argument(
            "EXL3 GDN checkpoint stale layout/owner/position/generation");
}

void Exl3GdnLayer::save_checkpoint(const Exl3GdnLayerCheckpoint& checkpoint,
                                   int position,cudaStream_t stream) {
    validate_checkpoint_storage(checkpoint);
    if(position<0)
        throw std::invalid_argument("EXL3 GDN checkpoint position");
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &capture_status),
          "query EXL3 GDN checkpoint stream capture");
    const auto next_generation=begin_checkpoint_graph_replay(checkpoint);
    record_checkpoint_copy_nodes(checkpoint,stream);
    if (capture_status == cudaStreamCaptureStatusNone)
        publish_checkpoint_graph_replay(
            checkpoint,position,next_generation,stream);
}

void Exl3GdnLayer::record_checkpoint_copy_nodes(
    const Exl3GdnLayerCheckpoint& checkpoint,cudaStream_t stream) const {
    validate_checkpoint_storage(checkpoint);
    check(cudaMemcpyAsync(checkpoint.recurrent_state_device, recurrent_state_, kStateBytes,
                          cudaMemcpyDeviceToDevice, stream),
          "save EXL3 GDN recurrent checkpoint");
    check(cudaMemcpyAsync(checkpoint.conv_state_device, conv_state_, kConvStateStorageBytes,
                          cudaMemcpyDeviceToDevice, stream),
          "save EXL3 GDN convolution checkpoint");
}

std::uint64_t Exl3GdnLayer::begin_checkpoint_graph_replay(
    const Exl3GdnLayerCheckpoint& checkpoint) {
    validate_checkpoint_storage(checkpoint);
    if (checkpoint_generation_==std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("EXL3 GDN checkpoint generation exhausted");
    const auto next_generation=++checkpoint_generation_;
    // Invalidate this backing and every copied descriptor before the first
    // overwrite is submitted. A partial CUDA submission therefore fails closed.
    checkpoint.backing_generation=next_generation;
    checkpoint.owner=nullptr;
    checkpoint.generation=0;
    checkpoint.generation_authority=nullptr;
    checkpoint.position=-1;
    checkpoint.saved_recurrent_state_device=nullptr;
    checkpoint.saved_conv_state_device=nullptr;
    retained_prefix_available_ = false;
    current_checkpoint_generation_ = 0;
    current_checkpoint_stream_ = nullptr;
    current_checkpoint_recurrent_ = nullptr;
    current_checkpoint_conv_ = nullptr;
    return next_generation;
}

void Exl3GdnLayer::publish_checkpoint_graph_replay(
    const Exl3GdnLayerCheckpoint& checkpoint,int position,
    std::uint64_t generation,cudaStream_t stream) {
    validate_checkpoint_storage(checkpoint);
    if(position<0 || !generation || checkpoint.backing_generation!=generation ||
       checkpoint.owner || checkpoint.generation || checkpoint.position!=-1 ||
       checkpoint.saved_recurrent_state_device || checkpoint.saved_conv_state_device ||
       current_checkpoint_generation_ || current_checkpoint_stream_ ||
       current_checkpoint_recurrent_ || current_checkpoint_conv_)
        throw std::invalid_argument("EXL3 GDN checkpoint graph publication provenance");
    checkpoint.owner = this;
    checkpoint.generation = generation;
    checkpoint.generation_authority = &checkpoint.backing_generation;
    checkpoint.position = position;
    checkpoint.saved_recurrent_state_device = checkpoint.recurrent_state_device;
    checkpoint.saved_conv_state_device = checkpoint.conv_state_device;
    current_checkpoint_generation_ = checkpoint.generation;
    current_checkpoint_stream_ = stream;
    current_checkpoint_recurrent_ = checkpoint.recurrent_state_device;
    current_checkpoint_conv_ = checkpoint.conv_state_device;
}

void Exl3GdnLayer::restore_checkpoint(const Exl3GdnLayerCheckpoint& checkpoint, cudaStream_t stream) {
    validate_saved_checkpoint(checkpoint,checkpoint.position);
    invalidate_continuation_history();
    retained_prefix_available_ = false;
    current_checkpoint_generation_ = 0;
    current_checkpoint_stream_ = nullptr;
    current_checkpoint_recurrent_ = nullptr;
    current_checkpoint_conv_ = nullptr;
    check(cudaMemcpyAsync(recurrent_state_, checkpoint.recurrent_state_device, kStateBytes,
                          cudaMemcpyDeviceToDevice, stream),
          "restore EXL3 GDN recurrent checkpoint");
    if (checkpoint.recurrent_state_device != recurrent_state_before_)
        check(cudaMemcpyAsync(recurrent_state_before_, checkpoint.recurrent_state_device,
                              kStateBytes,cudaMemcpyDeviceToDevice,stream),
              "restore EXL3 GDN recurrent state trace");
    check(cudaMemcpyAsync(conv_state_, checkpoint.conv_state_device, kConvStateStorageBytes,
                          cudaMemcpyDeviceToDevice, stream),
          "restore EXL3 GDN convolution checkpoint");
    exl3_launch_small(copy_conv_state_trace_kernel,dim3((kConvStateElements + 255) / 256),dim3(256),0,stream,
        conv_state_, conv_state_trace_);
    check(cudaGetLastError(), "refresh restored EXL3 GDN convolution state trace");
    trace_.state_before = recurrent_state_before_;
    trace_.state_after = recurrent_state_;
    // Restore deliberately never authorizes retained-prefix reconstruction.
    // A fresh eager save is required after every restore, preventing a copied
    // descriptor from rearming after its device storage was overwritten.
}

void Exl3GdnLayer::restore_host_state(std::span<const float> recurrent,
                                    std::span<const std::uint16_t> convolution,
                                    cudaStream_t stream) {
    if (recurrent.size_bytes() != kStateBytes || convolution.size_bytes() != kConvStateStorageBytes)
        throw std::invalid_argument("EXL3 GDN host checkpoint extent");
    invalidate_continuation_history();
    retained_prefix_available_ = false;
    current_checkpoint_generation_ = 0;
    current_checkpoint_stream_ = nullptr;
    current_checkpoint_recurrent_ = nullptr;
    current_checkpoint_conv_ = nullptr;
    check(cudaMemcpyAsync(recurrent_state_, recurrent.data(), kStateBytes,
                          cudaMemcpyHostToDevice, stream), "restore host recurrent state");
    check(cudaMemcpyAsync(recurrent_state_before_, recurrent.data(), kStateBytes,
                          cudaMemcpyHostToDevice, stream), "restore host recurrent trace");
    check(cudaMemcpyAsync(conv_state_, convolution.data(), kConvStateStorageBytes,
                          cudaMemcpyHostToDevice, stream), "restore host convolution");
    exl3_launch_small(copy_conv_state_trace_kernel,dim3((kConvStateElements + 255) / 256),dim3(256),0,stream,
        conv_state_, conv_state_trace_);
    check(cudaGetLastError(), "refresh host convolution trace");
    check(cudaStreamSynchronize(stream), "complete GDN host restore");
    trace_.state_before = recurrent_state_before_;
    trace_.state_after = recurrent_state_;
}

void Exl3GdnLayer::reconstruct_retained_prefix(
    const Exl3GdnLayerCheckpoint& checkpoint, int retained_rows,
    cudaStream_t stream) {
    const int attempted_rows=reconstruct_retained_prefix_host(checkpoint,retained_rows,stream);
    enqueue_retained_prefix_reconstruct(checkpoint,retained_rows,attempted_rows,stream);
}

int Exl3GdnLayer::reconstruct_retained_prefix_host(
    const Exl3GdnLayerCheckpoint& checkpoint, int retained_rows,
    cudaStream_t stream) {
    if (!retained_prefix_available_ || retained_rows <= 0 ||
        retained_rows > retained_prefix_rows_) {
        throw std::invalid_argument("EXL3 GDN retained prefix is absent or row count is invalid");
    }
    if (!checkpoint.saved_for(recurrent_model_owner_,model_layer_,checkpoint.position,
                              retained_prefix_checkpoint_generation_,this)) {
        throw std::invalid_argument("EXL3 GDN retained prefix checkpoint provenance mismatch");
    }
    if (checkpoint.recurrent_state_device != retained_prefix_checkpoint_recurrent_ ||
        checkpoint.conv_state_device != retained_prefix_checkpoint_conv_ ||
        checkpoint.recurrent_state_device !=
            checkpoint.saved_recurrent_state_device ||
        checkpoint.conv_state_device != checkpoint.saved_conv_state_device) {
        throw std::invalid_argument("EXL3 GDN retained prefix checkpoint buffer mismatch");
    }
    if (!checkpoint.recurrent_state_device ||
        checkpoint.recurrent_state_capacity_bytes < kStateBytes ||
        !checkpoint.conv_state_device ||
        checkpoint.conv_state_capacity_bytes < kConvStateStorageBytes) {
        throw std::invalid_argument("EXL3 GDN retained prefix checkpoint is undersized");
    }
    if (stream != retained_prefix_stream_) {
        throw std::invalid_argument("EXL3 GDN retained prefix stream mismatch");
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    check(cudaStreamIsCapturing(stream, &capture_status),
          "query EXL3 GDN retained prefix stream capture");
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::invalid_argument("EXL3 GDN retained prefix requires an eager stream");
    }
    // Validates that the continuation history still belongs to this attempt.
    (void)continuation_history_view(
        std::shared_ptr<const void>(recurrent_model_owner_,this),
        checkpoint.position,0,retained_prefix_rows_);

    // All validation precedes mutation. The saved base remains untouched and
    // reusable if the caller later needs a full rollback.
    invalidate_continuation_history();
    retained_prefix_available_ = false;
    current_checkpoint_generation_ = 0;
    current_checkpoint_stream_ = nullptr;
    current_checkpoint_recurrent_ = nullptr;
    current_checkpoint_conv_ = nullptr;
    trace_.state_before = recurrent_state_before_;
    trace_.state_after = recurrent_state_;
    return retained_prefix_rows_;
}

void Exl3GdnLayer::enqueue_retained_prefix_reconstruct(
    const Exl3GdnLayerCheckpoint& checkpoint, int retained_rows, int attempted_rows,
    cudaStream_t stream) const {
    check(cudaMemcpyAsync(recurrent_state_, checkpoint.recurrent_state_device,
                          kStateBytes, cudaMemcpyDeviceToDevice, stream),
          "restore EXL3 GDN retained prefix recurrent base");
    // The state-before buffer is a trace: the next forward rewrites it before
    // any consumer reads it, so the repair may skip restoring it.
    static const bool skip_trace_restore=[] {
        const char* value=std::getenv("NINFER_EXL3_GDN_SKIP_TRACE_RESTORE");
        if(!value)return true;  // measured default; "0" restores the trace copy
        if(std::strcmp(value,"0")==0)return false;
        if(std::strcmp(value,"1")==0)return true;
        throw std::invalid_argument("NINFER_EXL3_GDN_SKIP_TRACE_RESTORE must be 0 or 1");
    }();
    if (checkpoint.recurrent_state_device != recurrent_state_before_ && !skip_trace_restore)
        check(cudaMemcpyAsync(recurrent_state_before_,checkpoint.recurrent_state_device,
                              kStateBytes,cudaMemcpyDeviceToDevice,stream),
              "restore EXL3 GDN retained prefix recurrent trace");
    // Continuation history rows start at row 0 of the fixed attempt buffers.
    reconstruct_conv_prefix_kernel<<<(kQkv + 255) / 256, 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(checkpoint.conv_state_device),
        half_buffers_[3], conv_state_, conv_state_trace_, retained_rows,
        attempted_rows);
    check(cudaGetLastError(), "reconstruct EXL3 GDN retained convolution prefix");
    launch_gdn_recurrence_sm120(
        half_buffers_[4], half_buffers_[5], half_buffers_[6], float_buffers_[5],
        float_buffers_[4], recurrent_state_, half_buffers_[9], retained_rows, stream);
    check(cudaGetLastError(), "reconstruct EXL3 GDN retained recurrent prefix");
}

void Exl3GdnLayer::arm_captured_retained_prefix(
    const Exl3GdnLayerCheckpoint& checkpoint,int attempted_rows,
    cudaStream_t stream) {
    if (attempted_rows < 2 || attempted_rows > 8 ||
        !checkpoint.saved_for(recurrent_model_owner_,model_layer_,checkpoint.position,
                              current_checkpoint_generation_,this) ||
        stream != current_checkpoint_stream_ ||
        checkpoint.saved_recurrent_state_device != current_checkpoint_recurrent_ ||
        checkpoint.saved_conv_state_device != current_checkpoint_conv_)
        throw std::invalid_argument("captured GDN retained prefix provenance");
    invalidate_continuation_history();
    continuation_history_rows_=attempted_rows;
    retained_prefix_available_ = true;
    retained_prefix_rows_ = attempted_rows;
    retained_prefix_stream_ = stream;
    retained_prefix_checkpoint_generation_ = checkpoint.generation;
    retained_prefix_checkpoint_recurrent_ = checkpoint.saved_recurrent_state_device;
    retained_prefix_checkpoint_conv_ = checkpoint.saved_conv_state_device;
    current_checkpoint_generation_ = 0;
    current_checkpoint_stream_ = nullptr;
    current_checkpoint_recurrent_ = nullptr;
    current_checkpoint_conv_ = nullptr;
}

void Exl3GdnLayer::complete_prefill_projection_chain_graph_after_drain(
    cudaStream_t stream) {
    const auto snapshot=prefill_projection_chain_graph_.snapshot();
    if(snapshot.pending &&
       !prefill_projection_chain_graph_.complete_after_drain(stream))
        throw std::logic_error("GDN prefill projection-chain completion");
}

bool Exl3GdnLayer::supports_bulk_prefill() const noexcept {
    return reconstruct_gemm_ && reconstruct_gemm_->accepts_all_model_shapes() &&
        !capture_active_ && !projection_timing_ && !projection_observer_ &&
        weights_.qkv_metadata.mul1 && weights_.z_metadata.mul1 &&
        !weights_.qkv_metadata.mcg && !weights_.z_metadata.mcg &&
        !weights_.qkv_metadata.has_bias && !weights_.z_metadata.has_bias &&
        (weights_.qkv_metadata.K==6 || weights_.qkv_metadata.K==7) &&
        (weights_.z_metadata.K==6 || weights_.z_metadata.K==7);
}

void Exl3GdnLayer::prepare_bulk_prefill(const std::uint16_t* input,
    BulkPrefillBuffers buffers,cudaStream_t stream) {
    if (!supports_bulk_prefill() || !input || !buffers.h || !buffers.qkv ||
        !buffers.z || buffers.rows<=0 || buffers.rows>4096 ||
        !reconstruct_gemm_->supports(weights_.qkv_metadata,buffers.rows) ||
        !reconstruct_gemm_->supports(weights_.z_metadata,buffers.rows))
        throw std::invalid_argument("GDN bulk prefill admission");
    exl3_launch_small(rms_norm_f16_kernel<>,dim3(buffers.rows),dim3(512),512*sizeof(float),stream,
        input,weights_.input_norm,buffers.h,buffers.rows,kHidden);
    check(cudaGetLastError(),"launch GDN bulk input RMSNorm");
    reconstruct_gemm_->forward_numeric_candidate(weights_.qkv,
        weights_.qkv_metadata,buffers.h,buffers.qkv,buffers.rows,stream);
    reconstruct_gemm_->forward_numeric_candidate(weights_.z,
        weights_.z_metadata,buffers.h,buffers.z,buffers.rows,stream);
}

bool Exl3GdnLayer::supports_bulk_mlp(int rows) const noexcept {
    return reconstruct_gemm_ && reconstruct_gemm_->accepts_all_model_shapes() &&
        !capture_active_ && !projection_timing_ && !projection_observer_ &&
        rows>=256 && rows<=8192 &&
        reconstruct_gemm_->supports(weights_.gate_metadata,rows) &&
        reconstruct_gemm_->supports(weights_.up_metadata,rows) &&
        reconstruct_gemm_->supports(weights_.down_metadata,rows) &&
        weights_.gate_metadata.in_features==kHidden &&
        weights_.up_metadata.in_features==kHidden &&
        weights_.gate_metadata.out_features==kIntermediate &&
        weights_.up_metadata.out_features==kIntermediate &&
        weights_.down_metadata.in_features==kIntermediate &&
        weights_.down_metadata.out_features==kHidden;
}

void Exl3GdnLayer::forward_before_bulk_mlp(const std::uint16_t* input,
    DeferredMlpBuffers buffers,cudaStream_t stream) {
    if (!buffers.post || !buffers.mlp_input || buffers.rows<=0 ||
        buffers.rows>max_rows_)
        throw std::invalid_argument("GDN bulk MLP prefix extent");
    forward(input,buffers.post,buffers.rows,stream,false,true,true,nullptr,&buffers);
    if (bulk_mlp_weight_prefetch_)
        (void)reconstruct_gemm_->prefetch_numeric_weight(weights_.gate,
            weights_.gate_metadata,1024,stream);
}

void Exl3GdnLayer::finish_bulk_mlp(DeferredMlpBuffers buffers,
    std::uint16_t* gate,std::uint16_t* up,std::uint16_t* act,
    std::uint16_t* down,
    std::uint16_t* output,cudaStream_t stream,bool preserve_trace) {
    if (!supports_bulk_mlp(buffers.rows) || !buffers.post ||
        !buffers.mlp_input || !gate || !up || !act || !down || !output)
        throw std::invalid_argument("GDN bulk MLP completion admission");
    if (bulk_mlp_weight_prefetch_)
        (void)reconstruct_gemm_->prefetch_numeric_weight(weights_.up,
            weights_.up_metadata,buffers.rows,stream);
    reconstruct_gemm_->forward_numeric_candidate(weights_.gate,
        weights_.gate_metadata,buffers.mlp_input,gate,buffers.rows,stream);
    reconstruct_gemm_->forward_numeric_candidate(weights_.up,
        weights_.up_metadata,buffers.mlp_input,up,buffers.rows,stream);
    const int final_offset=((buffers.rows-1)/1024)*1024;
    const int final_rows=buffers.rows-final_offset;
    if(preserve_trace && act==gate)
        check(cudaMemcpyAsync(half_buffers_[16],
            gate+static_cast<std::size_t>(final_offset)*kIntermediate,
            static_cast<std::size_t>(final_rows)*kIntermediate*sizeof(std::uint16_t),
            cudaMemcpyDeviceToDevice,stream),
            "preserve final GDN bulk gate trace");
    bool fused_down_residual=false;
    if(bulk_mlp_fused_down_ && reconstruct_gemm_->supports_fused_gate_up_down()) {
        reconstruct_gemm_->forward_numeric_gate_up_down(weights_.down,
            weights_.down_metadata,gate,up,act,down,buffers.rows,stream);
    } else {
        exl3_launch_small(silu_mul_kernel,dim3((buffers.rows*kIntermediate+255)/256),dim3(256),0,stream,
            gate,up,act,buffers.rows*kIntermediate);
        check(cudaGetLastError(),"launch GDN bulk MLP activation");
        fused_down_residual=bulk_mlp_fused_residual_ && down==output;
        reconstruct_gemm_->forward_numeric_candidate(weights_.down,
            weights_.down_metadata,act,down,buffers.rows,stream,
            nullptr,nullptr,nullptr,
            fused_down_residual?buffers.post:nullptr,
            fused_down_residual && preserve_trace?half_buffers_[19]:nullptr,
            final_offset);
    }
    if(!fused_down_residual && preserve_trace && down==output)
        check(cudaMemcpyAsync(half_buffers_[19],
            down+static_cast<std::size_t>(final_offset)*kHidden,
            static_cast<std::size_t>(final_rows)*kHidden*sizeof(std::uint16_t),
            cudaMemcpyDeviceToDevice,stream),
            "preserve final GDN bulk down trace");
    if(!fused_down_residual) {
        exl3_launch_small(residual_kernel,dim3((buffers.rows*kHidden+255)/256),dim3(256),0,stream,
            buffers.post,down,output,buffers.rows*kHidden);
        check(cudaGetLastError(),"launch GDN bulk final residual");
    }
    trace_.post_attention_residual=buffers.post+
        static_cast<std::size_t>(final_offset)*kHidden;
    trace_.mlp_input=buffers.mlp_input+
        static_cast<std::size_t>(final_offset)*kHidden;
    trace_.gate_projection=preserve_trace && act==gate?half_buffers_[16]:
        gate+static_cast<std::size_t>(final_offset)*kIntermediate;
    trace_.up_projection=up+
        static_cast<std::size_t>(final_offset)*kIntermediate;
    trace_.activated_mlp=act+
        static_cast<std::size_t>(final_offset)*kIntermediate;
    trace_.down_projection=preserve_trace && down==output?half_buffers_[19]:
        down+static_cast<std::size_t>(final_offset)*kHidden;
    trace_.layer_output=output+
        static_cast<std::size_t>(final_offset)*kHidden;
}

void Exl3GdnLayer::forward(const std::uint16_t* input, std::uint16_t* output, int rows,
                           cudaStream_t stream, bool profile, bool preserve_m1_topology,
                           bool wide_prefill,const BulkPrefillBuffers* prepared,
                           const DeferredMlpBuffers* deferred_mlp) {
    if (!input || !output || rows <= 0 || rows > max_rows_) throw std::invalid_argument("invalid EXL3 GDN input/output/rows");
    if (prepared && (!wide_prefill || prepared->rows!=rows ||
        !prepared->h || !prepared->qkv || !prepared->z ||
        !supports_bulk_prefill()))
        throw std::invalid_argument("GDN prepared chunk contract");
    if (deferred_mlp && (!wide_prefill || profile ||
        deferred_mlp->rows!=rows || !deferred_mlp->post ||
        !deferred_mlp->mlp_input || projection_timing_ ||
        projection_observer_ ||
        std::getenv("NINFER_EXL3_TEST_GDN_PREFILL_RECURRENCE_SAMPLE")))
        throw std::invalid_argument("GDN deferred MLP contract");
    invalidate_continuation_history();
    const std::uint64_t base_checkpoint_generation = current_checkpoint_generation_;
    const cudaStream_t base_checkpoint_stream = current_checkpoint_stream_;
    const void* base_checkpoint_recurrent = current_checkpoint_recurrent_;
    const void* base_checkpoint_conv = current_checkpoint_conv_;
    bool eligible_retained_prefix = preserve_m1_topology &&
        Exl3GdnScratchReuseContract::retained_rows_supported(
            static_cast<std::size_t>(rows)) && base_checkpoint_generation != 0 &&
        stream == base_checkpoint_stream;
    if (eligible_retained_prefix) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        check(cudaStreamIsCapturing(stream, &capture_status),
              "query EXL3 GDN retained-prefix source capture");
        eligible_retained_prefix =
            capture_status == cudaStreamCaptureStatusNone;
    }
    const auto history_storage=scratch_reuse_.history_storage(
        static_cast<std::size_t>(rows),eligible_retained_prefix,capture_active_);
    if(history_storage==Exl3GdnHistoryStorage::refused)
        throw std::invalid_argument("GDN scratch row/lifetime contract");
    const bool use_wide_slab=
        history_storage==Exl3GdnHistoryStorage::shared_wide;
    if (use_wide_slab && !wide_prefill)
        throw std::invalid_argument(
            "GDN split storage requires wide prefill for rows above private capacity");
    retained_prefix_available_ = false;
    current_checkpoint_generation_ = 0;
    current_checkpoint_stream_ = nullptr;
    current_checkpoint_recurrent_ = nullptr;
    current_checkpoint_conv_ = nullptr;
    static std::atomic<int> prefill_stage_samples{0};
    const char* stage_sample_option=std::getenv(
        "NINFER_EXL3_TEST_GDN_PREFILL_RECURRENCE_SAMPLE");
    const bool stage_sample=rows==1024 && wide_prefill &&
        stage_sample_option && std::strcmp(stage_sample_option,"1")==0 &&
        prefill_stage_samples.fetch_add(1,std::memory_order_relaxed)<2;
    const bool collect_stage_events=profile || stage_sample;
    cudaEvent_t starts[12]{}, ends[12]{}, total_end{};
    if (collect_stage_events) {
        for (int i = 0; i < 12; ++i) { check(cudaEventCreate(&starts[i]), "create GDN start event"); check(cudaEventCreate(&ends[i]), "create GDN end event"); }
        check(cudaEventCreate(&total_end), "create GDN total event");
    }
    const auto begin = [&](int i) { if (collect_stage_events) record(starts[i], stream); };
    const auto end = [&](int i) { if (collect_stage_events) record(ends[i], stream); };
    auto* h = prepared?prepared->h:half_buffers_[0];
    auto* qkv = prepared?prepared->qkv:half_buffers_[1];
    auto* z = prepared?prepared->z:half_buffers_[2];
    auto* conv_input = reinterpret_cast<std::uint16_t*>(use_wide_slab ? wide_half_buffers_[0] : half_buffers_[3]);
    auto* q = use_wide_slab ? wide_half_buffers_[1] : half_buffers_[4];
    auto* k = use_wide_slab ? wide_half_buffers_[2] : half_buffers_[5];
    auto* v = use_wide_slab ? wide_half_buffers_[3] : half_buffers_[6];
    auto* conv_output = half_buffers_[7]; auto* z_bf16 = half_buffers_[8];
    auto* core = use_wide_slab ? wide_half_buffers_[4] : half_buffers_[9];
    auto* gdn_norm_input = half_buffers_[10]; auto* gdn_norm = half_buffers_[11]; auto* o_input = half_buffers_[12];
    auto* o = half_buffers_[13];
    auto* post = deferred_mlp?deferred_mlp->post:half_buffers_[14];
    auto* mlp_input = deferred_mlp?deferred_mlp->mlp_input:half_buffers_[15];
    auto* gate = half_buffers_[16]; auto* up = half_buffers_[17]; auto* act = half_buffers_[18]; auto* down = half_buffers_[19];
    auto* final_output = half_buffers_[20]; auto* head_trace = half_buffers_[21];
    float* b = float_buffers_[0]; float* a = float_buffers_[1]; float* beta = float_buffers_[2]; float* g = float_buffers_[3];
    float* beta_trace = float_buffers_[4]; float* g_trace = float_buffers_[5];
    const bool control_pair = gaming_[Gopt::GdnControlRowPair] &&
        rows >= 2 && rows <= 8 && !wide_prefill;
    const auto launch_control = [&] {
        if (control_pair) {
            control_fused_row_pair_kernel<<<((rows+1)/2)*kHeads,128,0,stream>>>(
                h,weights_.a_weight,weights_.b_weight,weights_.a_log,weights_.dt_bias,
                a,b,beta_trace,g_trace,rows);
            gopt_record(gaming_submissions_,Gopt::GdnControlRowPair);
        } else if(rows>=2 && rows<16 && gdn_control_decode_tiled_enabled()) {
            control_fused_tiled_kernel<2,2><<<((rows+1)/2)*(kHeads/2),128,0,stream>>>(
                h,weights_.a_weight,weights_.b_weight,weights_.a_log,weights_.dt_bias,
                a,b,beta_trace,g_trace,rows);
        } else if(rows>=16 && gdn_control_tiled_enabled()) {
            control_fused_tiled_kernel<4,4><<<((rows+3)/4)*(kHeads/4),128,0,stream>>>(
                h,weights_.a_weight,weights_.b_weight,weights_.a_log,weights_.dt_bias,
                a,b,beta_trace,g_trace,rows);
        } else {
            exl3_launch_small(gdn_control_staged_enabled()&&control_aligned(h,weights_.a_weight,weights_.b_weight)?control_fused_staged_kernel:control_fused_kernel,
                dim3(rows*kHeads),dim3(128),0,stream,
                h,weights_.a_weight,weights_.b_weight,weights_.a_log,weights_.dt_bias,
                a,b,beta_trace,g_trace,rows);
        }
        check(cudaGetLastError(),"launch fused GDN control projections");
    };
    const auto project_on = [&](Exl3CudaLinearWorkspace* workspace,
                             const Exl3CudaLinearWeights& weights,
                             const Exl3CudaLinearMetadata& metadata,
                             const std::uint16_t* source,
                             std::uint16_t* destination,
                             Exl3TargetProjectionOperator operation,
                             cudaStream_t projection_stream,
                             const std::uint16_t* transformed_input = nullptr) {
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
        const bool target_z_k6_candidate = rowwise &&
            operation == Exl3TargetProjectionOperator::z &&
            workspace->target_z_k6_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_z);
        const bool target_qkv_k6_candidate = rowwise && !wide_prefill &&
            operation == Exl3TargetProjectionOperator::qkv &&
            workspace->target_qkv_k6_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_qkv);
        const auto target_k5_admission =
            operation == Exl3TargetProjectionOperator::qkv
                ? Exl3CudaLinearAdmission::target_continuation_qkv
            : operation == Exl3TargetProjectionOperator::z
                ? Exl3CudaLinearAdmission::target_continuation_z
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
            target_o_k7_candidate || target_z_k6_candidate || target_wide_candidate ||
            target_qkv_k6_candidate || target_k5_candidate;
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
               (metadata.K == 5 || metadata.K == 6 || metadata.K == 7) && rows >= 256) ||
              ((operation == Exl3TargetProjectionOperator::gate ||
                operation == Exl3TargetProjectionOperator::up) &&
               metadata.K == 6 && metadata.in_features == kHidden &&
               metadata.out_features == kIntermediate && rows >= 512) ||
              (operation == Exl3TargetProjectionOperator::down &&
               metadata.K == 7 && metadata.in_features == kIntermediate &&
               metadata.out_features == kHidden && rows >= 256)));
        static std::atomic<int> prefill_projection_route_samples{0};
        if(stage_sample_option &&
           std::strcmp(stage_sample_option,"1")==0 &&
           rows==1024 && wide_prefill &&
           prefill_projection_route_samples.fetch_add(1,std::memory_order_relaxed)<6)
            std::fprintf(stderr,
                "GDN_PREFILL_ROUTE layer=%d op=%s K=%d in=%d out=%d rows=%d numeric=%d wide=%d dispatch=%s\n",
                model_layer_,target_projection_operator_name(operation),metadata.K,
                metadata.in_features,metadata.out_features,rows,
                numeric_reconstruct_candidate?1:0,target_wide_candidate?1:0,
                workspace->dispatch_name(metadata,rows));
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
                metadata.in_features == kZ &&
                metadata.out_features == kHidden;
            const bool observed_z =
                projection_observer_selection_ == Exl3TargetProjectionObserverSelection::z_k6 &&
                operation == Exl3TargetProjectionOperator::z &&
                metadata.in_features == kHidden && metadata.out_features == kZ;
            observed_projection =
                (((observed_gate_up || observed_down || observed_z) && metadata.K == 6) ||
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
                operation == Exl3TargetProjectionOperator::z ? "z" :
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
            if (transformed_input)
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
            const auto admission = target_initial16
                    ? Exl3CudaLinearAdmission::target_initial16
                    : target_wide_candidate
                    ? Exl3CudaLinearAdmission::target_wide_prefill
                     : target_qkv_k6_candidate
                     ? Exl3CudaLinearAdmission::target_continuation_qkv
                     : target_z_k6_candidate
                     ? Exl3CudaLinearAdmission::target_continuation_z
                     : target_k5_candidate
                     ? target_k5_admission
                    : target_gateup_m16
                    ? Exl3CudaLinearAdmission::target_prefill_gate_up
                    : target_gateup_candidate
                    ? Exl3CudaLinearAdmission::target_continuation_gate_up
                    : (target_down_candidate
                           ? Exl3CudaLinearAdmission::target_continuation_down
                           : (target_o_k7_candidate
                                  ? Exl3CudaLinearAdmission::target_continuation_o
                                  : Exl3CudaLinearAdmission::ordinary));
            if (transformed_input)
                workspace->forward_from_transformed(
                    weights, metadata, transformed_input, destination, rows,
                    projection_stream, admission);
            else
                workspace->forward(
                    weights, metadata, source, destination, rows,
                    projection_stream, admission);
        }
        if (projection_timing_) projection_timing_->end(timing_slot, projection_stream);
    };
    const auto project = [&](Exl3CudaLinearWorkspace* workspace,
                             const Exl3CudaLinearWeights& weights,
                             const Exl3CudaLinearMetadata& metadata,
                             const std::uint16_t* source,
                             std::uint16_t* destination,
                             Exl3TargetProjectionOperator operation,
                             const std::uint16_t* transformed_input = nullptr) {
        project_on(workspace, weights, metadata, source, destination, operation,
                   stream, transformed_input);
    };

    begin(0);
    if(!prepared) {
        exl3_launch_small(rms_norm_f16_kernel<>,dim3(rows),dim3(512),512 * sizeof(float),stream,
            input,weights_.input_norm,h,rows,kHidden);
        check(cudaGetLastError(),"launch GDN input RMSNorm");
    }
    end(0);
    const bool concurrent_qkvz = capture_active_ && preserve_m1_topology && rows == 8 &&
        graph_qkvz_concurrency_.complete();
    const bool dual_transform = exl3_paired_transform_admission(
        dual_input_transform_,bool(projection_timing_),bool(projection_observer_),
        preserve_m1_topology,rows,wide_prefill)==Exl3PairedTransformAdmission::admitted &&
        !wide_prefill && rows<=8;
    const auto* projection_chain_value=
        std::getenv("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS");
    const auto* projection_chain_gdn_value=
        std::getenv("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GDN");
    if(projection_chain_gdn_value &&
       std::strcmp(projection_chain_gdn_value,"0")!=0 &&
       std::strcmp(projection_chain_gdn_value,"1")!=0)
        throw std::invalid_argument(
            "NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GDN must be 0 or 1");
    const bool projection_chain=projection_chain_value &&
        std::strcmp(projection_chain_value,"1")==0 &&
        (!projection_chain_gdn_value ||
         std::strcmp(projection_chain_gdn_value,"1")==0) && !capture_active_ &&
        !profile && !projection_timing_ && !projection_observer_ &&
        preserve_m1_topology && wide_prefill && rows==1024 &&
        !dual_transform && !concurrent_qkvz;
    if(prepared) {
        begin(1); end(1);
        begin(2); end(2);
        begin(3); launch_control(); end(3);
    } else if(projection_chain) {
        std::array<Exl3GraphBufferIdentity,16> buffers{};
        std::array<Exl3GraphBoundResource,16> resources{};
        std::size_t count=0;
        const auto append=[&](const void* address,std::size_t bytes,
                              const std::shared_ptr<const void>& owner) {
            if(!address || !bytes || count==buffers.size())
                throw std::logic_error("GDN projection-chain resource");
            buffers[count]={address,bytes};
            resources[count]={owner,address,bytes,count+1};
            ++count;
        };
        append(h,static_cast<std::size_t>(rows)*kHidden*2,
               prefill_projection_chain_scratch_owner_);
        append(qkv,static_cast<std::size_t>(rows)*weights_.qkv_metadata.out_features*2,
               prefill_projection_chain_scratch_owner_);
        append(z,static_cast<std::size_t>(rows)*weights_.z_metadata.out_features*2,
               prefill_projection_chain_scratch_owner_);
        const auto append_weights=[&](const Exl3CudaLinearWeights& value) {
            if(value.trellis)append(value.trellis,2,prefill_projection_chain_model_owner_);
            if(value.suh)append(value.suh,2,prefill_projection_chain_model_owner_);
            if(value.svh)append(value.svh,2,prefill_projection_chain_model_owner_);
            if(value.mul1)append(value.mul1,4,prefill_projection_chain_model_owner_);
        };
        append_weights(weights_.qkv);append_weights(weights_.z);
        Exl3GraphCompatibilityFingerprint fingerprint;
        fingerprint.bind(prefill_projection_chain_context_owner_,
            prefill_projection_chain_model_owner_,
            prefill_projection_chain_scratch_owner_,
            std::span<const Exl3GraphBufferIdentity>(buffers.data(),count),
            rows,max_rows_,rows,kHidden,1,0x47515a50u,
            Exl3GraphPrecision::oscar_int2_fp16,
            Exl3GraphPositionPolicy::oscar_split_class,1,stream);
        Exl3GraphCaptureExtent extent;extent.known=true;
        extent.retained[static_cast<unsigned>(
            Exl3ResourceInventory::Domain::graph_count)]=2;
        Exl3PrefillProjectionChainGraph::Request request{
            std::move(fingerprint),
            std::span<const Exl3GraphBoundResource>(resources.data(),count),
            extent,1,0x47515a500001ull,stream,true};
        begin(1);
        prefill_projection_chain_graph_.execute(request,[&](cudaStream_t graph_stream) {
            project_on(linear_workspaces_[0],weights_.qkv,weights_.qkv_metadata,
                h,qkv,Exl3TargetProjectionOperator::qkv,graph_stream);
            project_on(linear_workspaces_[1],weights_.z,weights_.z_metadata,
                h,z,Exl3TargetProjectionOperator::z,graph_stream);
        });
        check(cudaGetLastError(),"launch prefill projection-chain GDN QKV/Z graph");
        end(1);
        begin(3);
        launch_control(); end(3);
    } else if (dual_transform) {
        exl3_transform_input_pair(
            weights_.qkv, weights_.qkv_metadata, weights_.z, weights_.z_metadata,
            h, act, down, rows, stream);
        if(!capture_active_)++paired_transform_submissions_;
        begin(1); project(linear_workspaces_[0], weights_.qkv, weights_.qkv_metadata,
                          h, qkv, Exl3TargetProjectionOperator::qkv, act); end(1);
        begin(2); project(linear_workspaces_[1], weights_.z, weights_.z_metadata,
                          h, z, Exl3TargetProjectionOperator::z, down); end(2);
        begin(3);
        launch_control(); end(3);
    } else if (concurrent_qkvz) {
        check(cudaEventRecord(graph_qkvz_concurrency_.fork, stream),
              "record fixed-B8 GDN QKV/Z projection fork");
        check(cudaStreamWaitEvent(graph_qkvz_concurrency_.z_stream,
                                  graph_qkvz_concurrency_.fork, 0),
              "fork fixed-B8 GDN Z projection stream");
        try {
            begin(2);
            project_on(graph_qkvz_concurrency_.z_workspace, weights_.z,
                       weights_.z_metadata, h, z, Exl3TargetProjectionOperator::z,
                       graph_qkvz_concurrency_.z_stream);
            check(cudaEventRecord(graph_qkvz_concurrency_.z_done,
                                  graph_qkvz_concurrency_.z_stream),
                  "record fixed-B8 GDN Z projection completion");
            end(2);
            begin(1);
            project(linear_workspaces_[0], weights_.qkv, weights_.qkv_metadata, h, qkv,
                    Exl3TargetProjectionOperator::qkv);
            end(1);
            begin(3);
            launch_control();
            end(3);
            check(cudaStreamWaitEvent(stream, graph_qkvz_concurrency_.z_done, 0),
                  "join fixed-B8 GDN Z projection stream");
        } catch (...) {
            // Reconnect the auxiliary stream before the graph owner invalidates
            // a failed multi-stream capture.
            (void)cudaEventRecord(graph_qkvz_concurrency_.z_done,
                                  graph_qkvz_concurrency_.z_stream);
            (void)cudaStreamWaitEvent(stream, graph_qkvz_concurrency_.z_done, 0);
            throw;
        }
    } else {
        begin(1); project(linear_workspaces_[0], weights_.qkv, weights_.qkv_metadata, h, qkv,
                          Exl3TargetProjectionOperator::qkv); end(1);
        begin(2); project(linear_workspaces_[1], weights_.z, weights_.z_metadata, h, z,
                          Exl3TargetProjectionOperator::z); end(2);
        begin(3);
        launch_control(); end(3);
    }
    begin(4);
    const bool fast_same_weights_fp16kv_gdn_decode_conv =
        fast_same_weights_fp16kv_gdn_decode_conv_ && rows >= 1 && rows <= 8 &&
        !wide_prefill;
    bool conv_tiled_packed=false;
    if (fast_same_weights_fp16kv_gdn_decode_conv) {
        launch_gopt_conv(gaming_[Gopt::GdnConvTrace],true,qkv,weights_.conv_weight,
            conv_state_,conv_input,q,k,v,conv_output,conv_state_trace_,rows,stream);
        if(gaming_[Gopt::GdnConvTrace]) {
            check(cudaGetLastError(), "launch GOPT convolution trace");
            gopt_record(gaming_submissions_,Gopt::GdnConvTrace);
        }
        ++fast_same_weights_fp16kv_gdn_decode_conv_calls_;
    } else if (rows>=kConvTileRows && gdn_conv_prefill_tiled_enabled()) {
        gdn_conv_prefill_tiled_kernel<<<dim3(kQkv/kConvTileChannels,
            (rows+kConvTileRows-1)/kConvTileRows),256,0,stream>>>(
            qkv,weights_.conv_weight,conv_state_,conv_input,q,k,v,conv_output,rows);
        if(gaming_[Gopt::GdnConvTrace]) {
            gdn_conv_prefill_state_kernel<true><<<(kQkv+255)/256,256,0,stream>>>(
                qkv,conv_state_,rows,conv_state_trace_);
            check(cudaGetLastError(),"launch GOPT tiled conv trace");
            gopt_record(gaming_submissions_,Gopt::GdnConvTrace);
        } else {
            gdn_conv_prefill_state_kernel<false><<<(kQkv+255)/256,256,0,stream>>>(
                qkv,conv_state_,rows,nullptr);
        }
        conv_tiled_packed=true;
    } else {
        exl3_launch_small(transpose_f16_to_bf16_kernel,dim3((rows * kQkv + 255) / 256),dim3(256),0,stream,qkv, conv_input, rows, kQkv);
        launch_gopt_conv(gaming_[Gopt::GdnConvTrace],false,qkv,weights_.conv_weight,
            conv_state_,conv_input,q,k,v,conv_output,conv_state_trace_,rows,stream);
        if(gaming_[Gopt::GdnConvTrace]) {
            check(cudaGetLastError(),"launch GOPT ordinary conv trace");
            gopt_record(gaming_submissions_,Gopt::GdnConvTrace);
        }
    }
    if(!gaming_[Gopt::GdnConvTrace])
        exl3_launch_small(copy_conv_state_trace_kernel,dim3((kConvStateElements + 255) / 256),dim3(256),0,stream,conv_state_, conv_state_trace_);
    if (!fast_same_weights_fp16kv_gdn_decode_conv && !conv_tiled_packed)
        exl3_launch_small(pack_qkv_bf16_kernel,dim3((rows * kQkv + 255) / 256),dim3(256),0,stream,q, k, v, conv_output, rows);
    exl3_launch_small(convert_f16_to_bf16_kernel,dim3((rows * kZ + 255) / 256),dim3(256),0,stream,z, z_bf16, rows * kZ);
    check(cudaGetLastError(), "launch GDN convolution staging"); end(4);
    begin(5);
    // Diagnostic only: sample the reached recurrence on the unchanged Fast90
    // route. The synchronizing event result is never used as request timing.
    static std::atomic<int> prefill_recurrence_samples{0};
    const char* recurrence_sample_option=std::getenv(
        "NINFER_EXL3_TEST_GDN_PREFILL_RECURRENCE_SAMPLE");
    const bool recurrence_sample=rows==1024 && wide_prefill &&
        recurrence_sample_option &&
        std::strcmp(recurrence_sample_option,"1")==0 &&
        prefill_recurrence_samples.fetch_add(1,std::memory_order_relaxed)<16;
    cudaEvent_t recurrence_start=nullptr,recurrence_end=nullptr;
    if(recurrence_sample) {
        check(cudaEventCreate(&recurrence_start),"create GDN recurrence sample start");
        check(cudaEventCreate(&recurrence_end),"create GDN recurrence sample end");
    }
    // A verifier transaction may already have saved this exact root in the
    // state-before trace. Preserve it for rollback/accepted-prefix repair and
    // avoid submitting the duplicate full recurrent-state copy.
    if (!(eligible_retained_prefix &&
          base_checkpoint_recurrent == recurrent_state_before_))
        check(cudaMemcpyAsync(recurrent_state_before_, recurrent_state_, kStateBytes,
                              cudaMemcpyDeviceToDevice, stream),
              "save GDN state trace");
    const Exl3GdnStageFusionContract stage_fusion{
        static_cast<std::size_t>(rows),prefill_resident_,wide_prefill,
        preserve_m1_topology};
    const bool small_resident=gaming_[Gopt::GdnSmallResident] &&
        gopt_small_gdn(rows,preserve_m1_topology,wide_prefill,prefill_normalized_!=nullptr);
    if (small_resident || stage_fusion.exact_route_supported()) {
        float* normalized_q = prefill_normalized_;
        float* normalized_k = normalized_q + max_rows_ * kKeyHeads * kHeadDim;
        float* alpha = normalized_k + max_rows_ * kKeyHeads * kHeadDim;
        gdn_prefill_normalize_kernel<<<rows * kKeyHeads, 32, 0, stream>>>(
            q, k, g_trace, normalized_q, normalized_k, alpha);
        check(cudaGetLastError(), "launch prefill GDN normalization");
        const auto recurrent_row_bytes=static_cast<std::size_t>(rows)*
            Exl3GdnRecurrentLayout::value_heads*
            Exl3GdnRecurrentLayout::value_columns*sizeof(std::uint16_t);
        const Exl3GdnRecurrentVectorAccess vector_access{
            recurrent_state_,kStateBytes,v,recurrent_row_bytes,core,
            recurrent_row_bytes,static_cast<std::size_t>(rows),
            Exl3GdnRecurrentLayout::value_columns};
        const bool vector_pair_supported=prefill_resident_pair_vector_io_ &&
            vector_access.pair_columns_supported();
        if(recurrence_sample)
            check(cudaEventRecord(recurrence_start,stream),
                  "record GDN recurrence sample start");
        if(small_resident) {
            gdn_recurrence_prefill_resident_pair_columns_kernel<false><<<
                kHeads*(kHeadDim/(4*2)),4*32,0,stream>>>(
                normalized_q,normalized_k,v,alpha,beta_trace,recurrent_state_,core,rows);
            check(cudaGetLastError(),"launch GOPT small-M resident GDN");
            gopt_record(gaming_submissions_,Gopt::GdnSmallResident);
        } else if(prefill_resident_quad_columns_)
            gdn_recurrence_prefill_resident_quad_columns_kernel<<<
                kHeads*(kHeadDim/(4*4)),4*32,0,stream>>>(
                normalized_q,normalized_k,v,alpha,beta_trace,recurrent_state_,core,rows);
        else if(gdn_chunked_prefill_enabled() && wide_prefill && rows>=64 &&
                rows<=max_rows_ && max_rows_<=1024 && max_rows_%gdn_chunked::kC==0)
            gdn_chunked::launch(normalized_q,normalized_k,v,alpha,beta_trace,
                recurrent_state_,core,rows,gdn_chunked_workspace(),stream);
        else if(prefill_resident_pair_columns_) {
            if(vector_pair_supported)
                gdn_recurrence_prefill_resident_pair_columns_kernel<true><<<
                    kHeads*(kHeadDim/(4*2)),4*32,0,stream>>>(
                    normalized_q,normalized_k,v,alpha,beta_trace,recurrent_state_,core,rows);
            else if(!prefill_resident_pair_vector_io_)
                gdn_recurrence_prefill_resident_pair_columns_kernel<false><<<
                    kHeads*(kHeadDim/(4*2)),4*32,0,stream>>>(
                    normalized_q,normalized_k,v,alpha,beta_trace,recurrent_state_,core,rows);
            else
                gdn_recurrence_prefill_resident_kernel<<<
                    kHeads*(kHeadDim/4),4*32,0,stream>>>(
                    normalized_q,normalized_k,v,alpha,beta_trace,
                    recurrent_state_,core,rows);
        }
        else
            gdn_recurrence_prefill_resident_kernel<<<kHeads * (kHeadDim / 4), 4 * 32, 0, stream>>>(
                normalized_q, normalized_k, v, alpha, beta_trace, recurrent_state_, core, rows);
    } else {
        if(recurrence_sample)
            check(cudaEventRecord(recurrence_start,stream),
                  "record GDN recurrence sample start");
        launch_gdn_recurrence_sm120(
            q, k, v, g_trace, beta_trace, recurrent_state_, core, rows, stream);
    }
    check(cudaGetLastError(), "launch GDN recurrence"); end(5);
    if(recurrence_sample) {
        check(cudaEventRecord(recurrence_end,stream),
              "record GDN recurrence sample end");
        check(cudaEventSynchronize(recurrence_end),
              "synchronize GDN recurrence sample");
        float elapsed_ms=0.0f;
        check(cudaEventElapsedTime(&elapsed_ms,recurrence_start,recurrence_end),
              "read GDN recurrence sample");
        std::fprintf(stderr,
            "GDN_PREFILL_RECURRENCE_SAMPLE layer=%d rows=%d gpu_ms=%.6f\n",
            model_layer_,rows,elapsed_ms);
        cudaEventDestroy(recurrence_start);
        cudaEventDestroy(recurrence_end);
    }
    begin(6);
    Tensor tz(z_bf16, DType::BF16, {kHeadDim, kHeads, rows}); Tensor tcore(core, DType::BF16, {kHeadDim, kHeads, rows}); Tensor tnorm(gdn_norm, DType::BF16, {kHeadDim, kHeads, rows});
    Tensor nw(const_cast<std::uint16_t*>(weights_.gdn_norm), DType::BF16, {kHeadDim});
    ninfer::ops::gated_rmsnorm(tcore, nw, tz, kRmsEps, tnorm, stream);
    if(gaming_[Gopt::GdnOutputPack]) {
        gopt_gdn_output_pack_kernel<<<(rows*kZ+255)/256,256,0,stream>>>(
            core,gdn_norm,head_trace,o_input,rows*kZ);
        check(cudaGetLastError(),"launch GOPT GDN output packing");
        gopt_record(gaming_submissions_,Gopt::GdnOutputPack);
    } else {
        exl3_launch_small(pack_heads_bf16_kernel,dim3((rows * kZ + 255) / 256),dim3(256),0,stream,core, head_trace, rows);
        exl3_launch_small(transpose_bf16_to_f16_kernel,dim3((rows * kZ + 255) / 256),dim3(256),0,stream,gdn_norm, o_input, rows, kZ);
    }
    check(cudaGetLastError(), "launch GDN gated norm staging"); end(6);
    begin(7); project(linear_workspaces_[2], weights_.o, weights_.o_metadata, o_input, o,
                      Exl3TargetProjectionOperator::o); end(7);
    begin(8);
    const bool gopt_residual=gaming_[Gopt::GdnVerifierResidualNorm] &&
        rows>=1 && rows<=8 && !wide_prefill && preserve_m1_topology;
    if(gopt_residual) {
        exl3_gdn_residual_norm(input,o,weights_.post_attention_norm,post,mlp_input,rows,true,stream);
        gopt_record(gaming_submissions_,Gopt::GdnVerifierResidualNorm);
    } else if(fused_residual_norm_ && wide_prefill && rows>=1 && rows<=1024 && !capture_active_ && !profile &&
       !projection_timing_ && !projection_observer_) {
        exl3_gdn_residual_norm(input,o,weights_.post_attention_norm,post,mlp_input,rows,true,stream);
        ++fused_residual_norm_submissions_;
    } else {
        if(residual_norm_fused_enabled())
            exl3_launch_small(rms_norm_f16_kernel<true>,dim3(rows),dim3(512),512*sizeof(float),stream,
                input,weights_.post_attention_norm,mlp_input,rows,kHidden,o,post);
        else {
        exl3_launch_small(residual_kernel,dim3((rows*kHidden+255)/256),dim3(256),0,stream,input,o,post,rows*kHidden);
        exl3_launch_small(rms_norm_f16_kernel<>,dim3(rows),dim3(512),512*sizeof(float),stream,post,weights_.post_attention_norm,mlp_input,rows,kHidden);
        }
    }
    check(cudaGetLastError(), "launch GDN residual/norm"); end(8);
    if (deferred_mlp) {
        trace_={input,h,qkv,z,b,a,conv_input,conv_output,beta_trace,g_trace,
            recurrent_state_before_,recurrent_state_,core,core,o_input,o_input,
            o,post,mlp_input,nullptr,nullptr,nullptr,nullptr,nullptr};
        return;
    }
    begin(9);
    const char* gdn_mgemm_pair = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_MGEMM_PAIR");
    const bool gdn_m1_gate_up_pair =
        fast_same_weights_fp16kv_gdn_m1_gate_up_pair_ &&
        gdn_mgemm_pair && std::strcmp(gdn_mgemm_pair,"1")==0 &&
        rows == 1 && !profile && !projection_timing_ &&
        !projection_observer_ &&
        (weights_.gate_metadata.K == 5 || weights_.gate_metadata.K == 6 ||
         weights_.gate_metadata.K == 7) &&
        weights_.up_metadata.K == weights_.gate_metadata.K &&
        weights_.gate_metadata.mul1 && weights_.up_metadata.mul1 &&
        !weights_.gate_metadata.mcg && !weights_.up_metadata.mcg &&
        !weights_.gate_metadata.has_bias && !weights_.up_metadata.has_bias &&
        weights_.gate_metadata.in_features == kHidden &&
        weights_.up_metadata.in_features == kHidden &&
        weights_.gate_metadata.out_features == kIntermediate &&
        weights_.up_metadata.out_features == kIntermediate;
    if (gdn_m1_gate_up_pair) {
        // This production-labeled route remains default-off and graph-safe:
        // all policy, geometry and storage were fixed during construction.
        linear_workspaces_[3]->forward_target_m1_gate_up_pair(
            *linear_workspaces_[4],weights_.gate,weights_.gate_metadata,
            weights_.up,weights_.up_metadata,mlp_input,gate,up,stream);
        ++fast_same_weights_fp16kv_gdn_m1_gate_up_pair_submissions_;
    } else if (dual_transform) {
        exl3_transform_input_pair(
            weights_.gate, weights_.gate_metadata, weights_.up, weights_.up_metadata,
            mlp_input, act, down, rows, stream);
        if(!capture_active_)++paired_transform_submissions_;
        project(linear_workspaces_[3], weights_.gate, weights_.gate_metadata,
                mlp_input, gate, Exl3TargetProjectionOperator::gate, act);
        project(linear_workspaces_[4], weights_.up, weights_.up_metadata,
                mlp_input, up, Exl3TargetProjectionOperator::up, down);
    } else {
        // Recurrence and its residual/norm have already completed submission on
        // this private stream. Only stateless gate/up projections may suspend.
        const bool concurrent_mlp_gateup =
            eager_mlp_gateup_concurrency_.complete() && !capture_active_ &&
            !profile && !projection_timing_ && !projection_observer_ &&
            (preserve_m1_topology || rows == 1) && !wide_prefill &&
            rows >= 1 && rows <= 8 &&
            !shared_gateup_enabled_;
        const bool share=shared_gateup_enabled_ && shared_gateup_executor_ && !capture_active_ && !profile &&
            !projection_timing_ && !projection_observer_ && preserve_m1_topology &&
            !wide_prefill && rows>=1 && rows<=8;
        if (concurrent_mlp_gateup) {
            check(cudaEventRecord(eager_mlp_gateup_concurrency_.fork, stream),
                  "record eager GDN MLP gate/up fork");
            check(cudaStreamWaitEvent(eager_mlp_gateup_concurrency_.up_stream,
                                      eager_mlp_gateup_concurrency_.fork, 0),
                  "fork eager GDN up projection stream");
            try {
                project_on(eager_mlp_gateup_concurrency_.up_workspace,
                           weights_.up, weights_.up_metadata, mlp_input, up,
                           Exl3TargetProjectionOperator::up,
                           eager_mlp_gateup_concurrency_.up_stream);
                check(cudaGetLastError(), "launch concurrent GDN up projection");
                check(cudaEventRecord(eager_mlp_gateup_concurrency_.up_done,
                                      eager_mlp_gateup_concurrency_.up_stream),
                      "record eager GDN up projection completion");
                project(linear_workspaces_[3], weights_.gate,
                        weights_.gate_metadata, mlp_input, gate,
                        Exl3TargetProjectionOperator::gate);
                check(cudaStreamWaitEvent(stream,
                                          eager_mlp_gateup_concurrency_.up_done,
                                          0),
                      "join eager GDN up projection stream");
                ++eager_mlp_gateup_concurrent_calls_;
            } catch (...) {
                (void)cudaEventRecord(eager_mlp_gateup_concurrency_.up_done,
                                      eager_mlp_gateup_concurrency_.up_stream);
                (void)cudaStreamWaitEvent(
                    stream, eager_mlp_gateup_concurrency_.up_done, 0);
                throw;
            }
        } else {
            std::optional<Exl3ActivationLifetime::Scope> activation_scope;
            if(share)activation_scope.emplace(mlp_activation_lifetime_);
            const auto activation_witness=activation_scope?activation_scope->witness():Exl3ActivationLifetime::Witness{};
            const bool shared_gate=share && shared_gateup_executor_(Exl3TargetQContinuation{
                weights_.gate,weights_.gate_metadata,mlp_input,gate,rows,0,model_layer_,stream,
                Exl3TargetSharedFamily::gate,nullptr,activation_witness});
            if(!shared_gate)project(linear_workspaces_[3], weights_.gate, weights_.gate_metadata,
                    mlp_input, gate, Exl3TargetProjectionOperator::gate);
            const bool shared_up=share && shared_gateup_executor_(Exl3TargetQContinuation{
                weights_.up,weights_.up_metadata,mlp_input,up,rows,0,model_layer_,stream,
                Exl3TargetSharedFamily::up,nullptr,activation_witness});
            if(!shared_up)project(linear_workspaces_[4], weights_.up, weights_.up_metadata, mlp_input, up,
                    Exl3TargetProjectionOperator::up);
        }
    }
    end(9);
    const bool fused_gate_up=(
        (fused_gate_up_transform_ && wide_prefill) ||
        (small_m_fused_gate_up_transform_ && !wide_prefill &&
         (preserve_m1_topology || rows==1) && rows>=1 && rows<=8)) &&
        !capture_active_ && !profile && !projection_timing_ && !projection_observer_ &&
        !reconstruct_gemm_ && !shared_down_enabled_ &&
        !weights_.down_metadata.mcg && weights_.down_metadata.mul1 && !weights_.down_metadata.has_bias;
    begin(10);
    if(fused_gate_up) {
        linear_workspaces_[5]->transform_gate_up(weights_.down,weights_.down_metadata,
            gate,up,act,rows,stream,wide_prefill ?
                Exl3CudaLinearAdmission::target_wide_prefill :
                (rows==1 ? Exl3CudaLinearAdmission::ordinary :
                    Exl3CudaLinearAdmission::target_continuation_down));
        ++fused_gate_up_submissions_;
    } else {
        exl3_launch_small(silu_mul_kernel,dim3((rows * kIntermediate + 255) / 256),dim3(256),0,stream,gate, up, act, rows * kIntermediate);
        check(cudaGetLastError(), "launch GDN MLP activation");
    }
    end(10);
    begin(11);
    const bool shared_down=shared_down_enabled_ && shared_gateup_executor_ &&
        !capture_active_ && !profile && !projection_timing_ && !projection_observer_ &&
        preserve_m1_topology && !wide_prefill && rows>=1 && rows<=8 &&
        shared_gateup_executor_(Exl3TargetQContinuation{weights_.down,weights_.down_metadata,
            act,down,rows,0,model_layer_,stream,Exl3TargetSharedFamily::down});
    if(!shared_down)project(linear_workspaces_[5], weights_.down, weights_.down_metadata, act, down,
                       Exl3TargetProjectionOperator::down,
                       fused_gate_up?linear_workspaces_[5]->transformed_device():nullptr); exl3_launch_small(residual_kernel,dim3((rows * kHidden + 255) / 256),dim3(256),0,stream,post, down, final_output, rows * kHidden); check(cudaGetLastError(), "launch GDN final residual"); end(11);
    if (output != final_output) check(cudaMemcpyAsync(output, final_output, static_cast<std::size_t>(rows) * kHidden * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice, stream), "copy GDN output");
    if (collect_stage_events) {
        record(total_end, stream); check(cudaEventSynchronize(total_end), "synchronize GDN timing");
        float ms = 0.0f; check(cudaEventElapsedTime(&ms, starts[0], total_end), "read GDN total timing");
        if(profile)timings_.total_microseconds = ms * 1000.0;
        if(stage_sample)std::fprintf(stderr,"GDN_PREFILL_STAGE_SAMPLE layer=%d rows=%d total_gpu_ms=%.6f",model_layer_,rows,ms);
        for (int i = 0; i < 12; ++i) {
            check(cudaEventElapsedTime(&ms, starts[i], ends[i]), "read GDN timing");
            if(profile)timings_.microseconds[i] = ms * 1000.0;
            if(stage_sample)std::fprintf(stderr," stage%d_ms=%.6f",i,ms);
            cudaEventDestroy(starts[i]); cudaEventDestroy(ends[i]);
        }
        if(stage_sample)std::fprintf(stderr,"\n");
        cudaEventDestroy(total_end);
    }
    trace_ = {input, h, qkv, z, b, a, conv_input, conv_output, beta_trace, g_trace, recurrent_state_before_, recurrent_state_, core, core, o_input, o_input, o, post, mlp_input, gate, up, act, down, final_output};
    if(preserve_m1_topology &&
       history_storage==Exl3GdnHistoryStorage::private_retained &&
       Exl3GdnScratchReuseContract::verifier_rows_supported(
           static_cast<std::size_t>(rows)))
        continuation_history_rows_=rows;
    if (eligible_retained_prefix) {
        retained_prefix_available_ = true;
        retained_prefix_rows_ = rows;
        retained_prefix_stream_ = stream;
        retained_prefix_checkpoint_generation_ = base_checkpoint_generation;
        retained_prefix_checkpoint_recurrent_ = base_checkpoint_recurrent;
        retained_prefix_checkpoint_conv_ = base_checkpoint_conv;
    }
}

void Exl3GdnLayer::forward_pair_staged_serial_for_test(
    Exl3GdnLayer& first, const std::uint16_t* first_input,
    std::uint16_t* first_output, Exl3GdnLayer& second,
    const std::uint16_t* second_input, std::uint16_t* second_output,
    cudaStream_t stream, int fail_after_published_lane,
    Exl3GdnStageOracleTelemetry* telemetry) {
    constexpr int rows = 8;
    if (&first == &second || !first_input || !first_output ||
        !second_input || !second_output)
        throw std::invalid_argument("invalid staged GDN pair");
    if (first.max_rows_ < rows || second.max_rows_ < rows)
        throw std::invalid_argument("staged GDN pair requires B8 capacity");
    if (first.projection_timing_ || second.projection_timing_ ||
        first.projection_observer_ || second.projection_observer_)
        throw std::invalid_argument("staged GDN pair rejects projection instrumentation");
    if (fail_after_published_lane < -1 || fail_after_published_lane > 1)
        throw std::invalid_argument("staged GDN pair failpoint must be -1, 0, or 1");
    first.invalidate_continuation_history();
    second.invalidate_continuation_history();
    const auto same_linear = [](const Exl3CudaLinearWeights& a,
                                const Exl3CudaLinearWeights& b,
                                const Exl3CudaLinearMetadata& am,
                                const Exl3CudaLinearMetadata& bm) {
        return a.trellis == b.trellis && a.suh == b.suh && a.svh == b.svh &&
            a.mul1 == b.mul1 && am.in_features == bm.in_features &&
            am.out_features == bm.out_features && am.K == bm.K &&
            am.mcg == bm.mcg && am.mul1 == bm.mul1 &&
            am.has_bias == bm.has_bias;
    };
    const auto& fw = first.weights_;
    const auto& sw = second.weights_;
    if (!same_linear(fw.qkv, sw.qkv, fw.qkv_metadata, sw.qkv_metadata) ||
        !same_linear(fw.z, sw.z, fw.z_metadata, sw.z_metadata) ||
        !same_linear(fw.o, sw.o, fw.o_metadata, sw.o_metadata) ||
        !same_linear(fw.gate, sw.gate, fw.gate_metadata, sw.gate_metadata) ||
        !same_linear(fw.up, sw.up, fw.up_metadata, sw.up_metadata) ||
        !same_linear(fw.down, sw.down, fw.down_metadata, sw.down_metadata) ||
        fw.input_norm != sw.input_norm || fw.gdn_norm != sw.gdn_norm ||
        fw.post_attention_norm != sw.post_attention_norm ||
        fw.conv_weight != sw.conv_weight || fw.a_weight != sw.a_weight ||
        fw.b_weight != sw.b_weight || fw.a_log != sw.a_log ||
        fw.dt_bias != sw.dt_bias)
        throw std::invalid_argument("staged GDN pair weight provenance mismatch");
    if (telemetry) {
        *telemetry = {};
        telemetry->staged_pairs = 1;
    }

    struct Lane {
        Exl3GdnLayer* layer;
        const std::uint16_t* input;
        std::uint16_t* output;
        std::uint16_t* h;
        std::uint16_t* qkv;
        std::uint16_t* z;
        std::uint16_t* conv_input;
        std::uint16_t* q;
        std::uint16_t* k;
        std::uint16_t* v;
        std::uint16_t* conv_output;
        std::uint16_t* z_bf16;
        std::uint16_t* core;
        std::uint16_t* gdn_norm;
        std::uint16_t* o_input;
        std::uint16_t* o;
        std::uint16_t* post;
        std::uint16_t* mlp_input;
        std::uint16_t* gate;
        std::uint16_t* up;
        std::uint16_t* act;
        std::uint16_t* down;
        std::uint16_t* final_output;
        std::uint16_t* head_trace;
        float* b;
        float* a;
        float* beta_trace;
        float* g_trace;
    };
    const auto bind = [](Exl3GdnLayer& layer, const std::uint16_t* input,
                         std::uint16_t* output) {
        return Lane{&layer, input, output,
            layer.half_buffers_[0], layer.half_buffers_[1], layer.half_buffers_[2],
            layer.half_buffers_[3], layer.half_buffers_[4], layer.half_buffers_[5],
            layer.half_buffers_[6], layer.half_buffers_[7], layer.half_buffers_[8],
            layer.half_buffers_[9], layer.half_buffers_[11], layer.half_buffers_[12],
            layer.half_buffers_[13], layer.half_buffers_[14], layer.half_buffers_[15],
            layer.half_buffers_[16], layer.half_buffers_[17], layer.half_buffers_[18],
            layer.half_buffers_[19], layer.half_buffers_[20], layer.half_buffers_[21],
            layer.float_buffers_[0], layer.float_buffers_[1], layer.float_buffers_[4],
            layer.float_buffers_[5]};
    };
    Lane lanes[2] = {bind(first, first_input, first_output),
                     bind(second, second_input, second_output)};

    // Match ordinary forward's publication invalidation for both owners before
    // the first stateful kernel is submitted. Caller-owned transaction snapshots
    // remain reusable for rollback but cannot be mistaken for retained-prefix
    // reconstruction provenance.
    for (auto& lane : lanes) {
        auto& layer = *lane.layer;
        layer.retained_prefix_available_ = false;
        layer.current_checkpoint_generation_ = 0;
        layer.current_checkpoint_stream_ = nullptr;
        layer.current_checkpoint_recurrent_ = nullptr;
        layer.current_checkpoint_conv_ = nullptr;
    }

    const auto project = [&](Lane& lane, int workspace_index,
                             const Exl3CudaLinearWeights& weights,
                             const Exl3CudaLinearMetadata& metadata,
                             const std::uint16_t* source,
                             std::uint16_t* destination,
                             Exl3TargetProjectionOperator operation) {
        auto* workspace = lane.layer->linear_workspaces_[workspace_index];
        const bool gate_up = operation == Exl3TargetProjectionOperator::gate ||
                             operation == Exl3TargetProjectionOperator::up;
        const bool gate_up_candidate = gate_up &&
            (workspace->target_gateup_small_m_candidate(
                 metadata, rows, Exl3CudaLinearAdmission::target_continuation_gate_up) ||
             workspace->target_gateup_k5_small_m_candidate(
                 metadata, rows, Exl3CudaLinearAdmission::target_continuation_gate_up));
        const bool down_candidate = operation == Exl3TargetProjectionOperator::down &&
            workspace->target_down_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_down);
        const bool o_candidate = operation == Exl3TargetProjectionOperator::o &&
            (workspace->target_o_k7_small_m_candidate(
                 metadata, rows, Exl3CudaLinearAdmission::target_continuation_o) ||
             workspace->target_o_k6_small_m_candidate(
                 metadata, rows, Exl3CudaLinearAdmission::target_continuation_o));
        const bool z_candidate = operation == Exl3TargetProjectionOperator::z &&
            workspace->target_z_k6_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_z);
        const bool qkv_candidate = operation == Exl3TargetProjectionOperator::qkv &&
            workspace->target_qkv_k6_small_m_candidate(
                metadata, rows, Exl3CudaLinearAdmission::target_continuation_qkv);
        if (gate_up_candidate || down_candidate || o_candidate || z_candidate || qkv_candidate) {
            const auto admission = qkv_candidate
                ? Exl3CudaLinearAdmission::target_continuation_qkv
                : z_candidate
                ? Exl3CudaLinearAdmission::target_continuation_z
                : gate_up_candidate
                ? Exl3CudaLinearAdmission::target_continuation_gate_up
                : down_candidate
                ? Exl3CudaLinearAdmission::target_continuation_down
                : Exl3CudaLinearAdmission::target_continuation_o;
            workspace->forward(weights, metadata, source, destination, rows, stream, admission);
        } else {
            for (int row = 0; row < rows; ++row) {
                workspace->forward(weights, metadata,
                    source + static_cast<std::size_t>(row) * metadata.in_features,
                    destination + static_cast<std::size_t>(row) * metadata.out_features,
                    1, stream);
            }
        }
    };

    // Each phase is submitted in fixed owner order. Corresponding projections
    // are deliberately kept separate in T0a so T0b can replace only those two
    // calls with the already-qualified exact C2 route.
    for (auto& lane : lanes) {
        exl3_launch_small(rms_norm_f16_kernel<>,dim3(rows),dim3(512),512 * sizeof(float),stream,
            lane.input, lane.layer->weights_.input_norm, lane.h, rows, kHidden);
        check(cudaGetLastError(), "launch staged GDN input RMSNorm");
    }
    for (auto& lane : lanes)
        project(lane, 0, lane.layer->weights_.qkv, lane.layer->weights_.qkv_metadata,
                lane.h, lane.qkv, Exl3TargetProjectionOperator::qkv);
    if (telemetry) {
        ++telemetry->projection_phases;
        telemetry->serial_b8_projection_calls += 2;
    }
    for (auto& lane : lanes)
        project(lane, 1, lane.layer->weights_.z, lane.layer->weights_.z_metadata,
                lane.h, lane.z, Exl3TargetProjectionOperator::z);
    if (telemetry) {
        ++telemetry->projection_phases;
        telemetry->serial_b8_projection_calls += 2;
    }
    for (auto& lane : lanes) {
        exl3_launch_small(gdn_control_staged_enabled()&&control_aligned(lane.h,lane.layer->weights_.a_weight,lane.layer->weights_.b_weight)?control_fused_staged_kernel:control_fused_kernel,
            dim3(rows * kHeads),dim3(128),0,stream,
            lane.h, lane.layer->weights_.a_weight, lane.layer->weights_.b_weight,
            lane.layer->weights_.a_log, lane.layer->weights_.dt_bias,
            lane.a, lane.b, lane.beta_trace, lane.g_trace, rows);
        check(cudaGetLastError(), "launch staged fused GDN control projections");
    }
    for (int lane_index = 0; lane_index < 2; ++lane_index) {
        auto& lane = lanes[lane_index];
        exl3_launch_small(transpose_f16_to_bf16_kernel,dim3((rows * kQkv + 255) / 256),dim3(256),0,stream,
            lane.qkv, lane.conv_input, rows, kQkv);
        exl3_launch_small(gdn_conv_kernel<>,dim3((kQkv + 255) / 256),dim3(256),0,stream,
            lane.conv_input, lane.layer->weights_.conv_weight, lane.layer->conv_state_,
            lane.q, lane.k, lane.v, rows);
        exl3_launch_small(copy_conv_state_trace_kernel,dim3((kConvStateElements + 255) / 256),dim3(256),0,stream,
            lane.layer->conv_state_, lane.layer->conv_state_trace_);
        exl3_launch_small(pack_qkv_bf16_kernel,dim3((rows * kQkv + 255) / 256),dim3(256),0,stream,
            lane.q, lane.k, lane.v, lane.conv_output, rows);
        exl3_launch_small(convert_f16_to_bf16_kernel,dim3((rows * kZ + 255) / 256),dim3(256),0,stream,
            lane.z, lane.z_bf16, rows * kZ);
        check(cudaGetLastError(), "launch staged GDN convolution");
        check(cudaMemcpyAsync(lane.layer->recurrent_state_before_, lane.layer->recurrent_state_,
                              kStateBytes, cudaMemcpyDeviceToDevice, stream),
              "save staged GDN state trace");
        launch_gdn_recurrence_sm120(
            lane.q, lane.k, lane.v, lane.g_trace, lane.beta_trace,
            lane.layer->recurrent_state_, lane.core, rows, stream);
        check(cudaGetLastError(), "launch staged GDN recurrence");
        if (telemetry) ++telemetry->published_lanes;
        if (fail_after_published_lane == lane_index)
            throw std::runtime_error("injected staged GDN lane publication failure");
    }
    for (auto& lane : lanes) {
        Tensor tz(lane.z_bf16, DType::BF16, {kHeadDim, kHeads, rows});
        Tensor tcore(lane.core, DType::BF16, {kHeadDim, kHeads, rows});
        Tensor tnorm(lane.gdn_norm, DType::BF16, {kHeadDim, kHeads, rows});
        Tensor nw(const_cast<std::uint16_t*>(lane.layer->weights_.gdn_norm),
                  DType::BF16, {kHeadDim});
        ninfer::ops::gated_rmsnorm(tcore, nw, tz, kRmsEps, tnorm, stream);
        exl3_launch_small(pack_heads_bf16_kernel,dim3((rows * kZ + 255) / 256),dim3(256),0,stream,
            lane.core, lane.head_trace, rows);
        exl3_launch_small(transpose_bf16_to_f16_kernel,dim3((rows * kZ + 255) / 256),dim3(256),0,stream,
            lane.gdn_norm, lane.o_input, rows, kZ);
        check(cudaGetLastError(), "launch staged GDN norm");
    }
    for (auto& lane : lanes)
        project(lane, 2, lane.layer->weights_.o, lane.layer->weights_.o_metadata,
                lane.o_input, lane.o, Exl3TargetProjectionOperator::o);
    if (telemetry) {
        ++telemetry->projection_phases;
        telemetry->serial_b8_projection_calls += 2;
    }
    for (auto& lane : lanes) {
        if(residual_norm_fused_enabled())
            exl3_launch_small(rms_norm_f16_kernel<true>,dim3(rows),dim3(512),512 * sizeof(float),stream,
                lane.input, lane.layer->weights_.post_attention_norm,
                lane.mlp_input, rows, kHidden, lane.o, lane.post);
        else {
        exl3_launch_small(residual_kernel,dim3((rows * kHidden + 255) / 256),dim3(256),0,stream,
            lane.input, lane.o, lane.post, rows * kHidden);
        exl3_launch_small(rms_norm_f16_kernel<>,dim3(rows),dim3(512),512 * sizeof(float),stream,
            lane.post, lane.layer->weights_.post_attention_norm,
            lane.mlp_input, rows, kHidden);
        }
        check(cudaGetLastError(), "launch staged GDN residual/norm");
    }
    for (auto& lane : lanes)
        project(lane, 3, lane.layer->weights_.gate, lane.layer->weights_.gate_metadata,
                lane.mlp_input, lane.gate, Exl3TargetProjectionOperator::gate);
    if (telemetry) {
        ++telemetry->projection_phases;
        telemetry->serial_b8_projection_calls += 2;
    }
    for (auto& lane : lanes)
        project(lane, 4, lane.layer->weights_.up, lane.layer->weights_.up_metadata,
                lane.mlp_input, lane.up, Exl3TargetProjectionOperator::up);
    if (telemetry) {
        ++telemetry->projection_phases;
        telemetry->serial_b8_projection_calls += 2;
    }
    for (auto& lane : lanes) {
        exl3_launch_small(silu_mul_kernel,dim3((rows * kIntermediate + 255) / 256),dim3(256),0,stream,
            lane.gate, lane.up, lane.act, rows * kIntermediate);
        check(cudaGetLastError(), "launch staged GDN MLP activation");
    }
    for (auto& lane : lanes)
        project(lane, 5, lane.layer->weights_.down, lane.layer->weights_.down_metadata,
                lane.act, lane.down, Exl3TargetProjectionOperator::down);
    if (telemetry) {
        ++telemetry->projection_phases;
        telemetry->serial_b8_projection_calls += 2;
    }
    for (auto& lane : lanes) {
        exl3_launch_small(residual_kernel,dim3((rows * kHidden + 255) / 256),dim3(256),0,stream,
            lane.post, lane.down, lane.final_output, rows * kHidden);
        check(cudaGetLastError(), "launch staged GDN final residual");
        if (lane.output != lane.final_output)
            check(cudaMemcpyAsync(lane.output, lane.final_output,
                                  static_cast<std::size_t>(rows) * kHidden * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToDevice, stream),
                  "copy staged GDN output");
        lane.layer->trace_ = {lane.input, lane.h, lane.qkv, lane.z, lane.b, lane.a,
            lane.conv_input, lane.conv_output, lane.beta_trace, lane.g_trace,
            lane.layer->recurrent_state_before_, lane.layer->recurrent_state_,
            lane.core, lane.core, lane.o_input, lane.o_input, lane.o, lane.post,
            lane.mlp_input, lane.gate, lane.up, lane.act, lane.down, lane.final_output};
    }
}

} // namespace ninfer::exl3
