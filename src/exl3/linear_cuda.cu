#include "exl3/linear_cuda.h"
#include "exl3/environment_options.h"
#include "exl3/projection_lifetime.h"
#include "exl3/paired_transform_extent.h"
#include "exl3/reconstruction_stream.h"
#include "exl3/reconstruction_config.h"
#include "exl3/reconstruction_device_retirement.h"
#include "core/nvtx_range.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cooperative_groups.h>
#include "exl3/mia_exllamav3/quant/exl3_gemv_int8_kernel.cuh"
#include "exl3/mia_exllamav3/quant/native_persistent_gemm_inner.cuh"
#undef cuda_check
#undef cublas_check

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::exl3 {
namespace {

namespace cg = cooperative_groups;

constexpr int kInput = 5120;
constexpr int kOutput = 17408;
constexpr int kK = 5;
constexpr int kTilesK = 320;
constexpr int kTilesN = 1088;
constexpr int kHadamard = 128;
constexpr int kThreads = 256;
constexpr float kHadamardScale = 0.088388347648f;

// Programmatic dependent launch. Kernels carrying this prologue wait for full
// completion (and memory visibility) of their stream predecessor before any
// global access, so the attribute only overlaps launch/scheduling latency.
// Without the attribute both instructions are no-ops.
#define EXL3_PDL_PROLOGUE()                                              \
    do {                                                                 \
        asm volatile("griddepcontrol.wait;" ::: "memory");               \
        asm volatile("griddepcontrol.launch_dependents;");               \
    } while (0)

bool exl3_pdl_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_PDL");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

template<class... KernelArgs,class... CallArgs>
void exl3_launch_pdl(void (*kernel)(KernelArgs...),dim3 grid,dim3 block,
                     std::size_t shared,cudaStream_t stream,CallArgs&&... args) {
    if(!exl3_pdl_enabled()) {
        kernel<<<grid,block,shared,stream>>>(std::forward<CallArgs>(args)...);
        return;
    }
    cudaLaunchAttribute attribute{};
    attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed=1;
    cudaLaunchConfig_t config{};
    config.gridDim=grid;
    config.blockDim=block;
    config.dynamicSmemBytes=shared;
    config.stream=stream;
    config.attrs=&attribute;
    config.numAttrs=1;
    const cudaError_t error=cudaLaunchKernelEx(&config,kernel,std::forward<CallArgs>(args)...);
    if(error!=cudaSuccess)
        throw std::runtime_error(std::string("PDL launch: ")+cudaGetErrorString(error));
}
constexpr std::uint16_t kMul1AccumulatorHalf = 0x6400u;
constexpr std::uint16_t kMul1InverseHalf = 0x1eeeu;
constexpr std::uint16_t kMul1BiasHalf = 0xc931u;
constexpr int kShape4OutputBlocks = kTilesN / 32;
constexpr int kShape4Splits = Exl3LinearWorkspaceRequirements::accumulation_splits;
constexpr int kShape4CooperativeGrid = kShape4OutputBlocks * kShape4Splits;
// N8 x M16 scratch for the shared K6 down arithmetic. M1 and M2..8 use the
// same packed decode, FP32 row accumulators, K partitions, and output epilogue.
// The same packed N8/M16 producer serves each reached wide K6 target shape.
constexpr std::size_t kCoherentWideK6SharedBytes =
    2u * 256u * sizeof(half) + 2u * 8u * 16u * 6u * sizeof(std::uint16_t) +
    16u * 128u * sizeof(float);
constexpr std::size_t kCoherentDownK6SharedBytes =
    2u * 256u * sizeof(half) + 2u * 8u * 16u * 6u * sizeof(std::uint16_t) +
    16u * 128u * sizeof(float);
constexpr std::size_t kCoherentDownK7SharedBytes =
    2u * 256u * sizeof(half) + 2u * 8u * 16u * 7u * sizeof(std::uint16_t) +
    16u * 128u * sizeof(float);
constexpr std::size_t kCoherentOK7SharedBytes =
    2u * 256u * sizeof(half) + 2u * 8u * 16u * 7u * sizeof(std::uint16_t) +
    16u * 128u * sizeof(float);
// Coherent 128-column packed producers with a DeepStages cp.async ring.
constexpr std::size_t coherent_deep_shared_bytes(int bits, int stages) {
    return static_cast<std::size_t>(stages) * 256u * sizeof(half) +
           static_cast<std::size_t>(stages) * 8u * 16u *
               static_cast<std::size_t>(bits) * sizeof(std::uint16_t) +
           16u * 128u * sizeof(float);
}
// The coherent K6/K7 packed producers run 128-column CTAs whose grid is too
// small to cover DRAM latency with a two-stage ring. Four stages (default) is
// the measured selection; NINFER_EXL3_COHERENT_DEEP_PIPELINE=0 restores the
// two-stage ring and 8 is retained for comparison. Numerically identical.
int coherent_deep_pipeline_stages() {
    static const int stages = [] {
        const char* value = std::getenv("NINFER_EXL3_COHERENT_DEEP_PIPELINE");
        if (!value) return 4;
        if (std::strcmp(value, "0") == 0) return 0;
        if (std::strcmp(value, "4") == 0) return 4;
        if (std::strcmp(value, "8") == 0) return 8;
        throw std::invalid_argument(
            "NINFER_EXL3_COHERENT_DEEP_PIPELINE must be 0, 4 or 8");
    }();
    return stages;
}
constexpr std::size_t kShape4SharedBytes =
    256u * sizeof(half) + 2u * 32u * 80u * sizeof(std::uint16_t) +
    16u * 512u * sizeof(float);

std::mutex prefill_persisting_l2_mutex;
std::unordered_map<int,std::size_t> prefill_persisting_l2_by_device;

// Ordinary device-KV graph capture for the split stream-reduction candidates
// is an explicit differential lane.  The default remains fail-closed until a
// matched token and numerical screen has qualified the complete composition.
bool ordinary_graph_stream_reduction_allowed() noexcept {
    const char* value = std::getenv(
        "NINFER_EXL3_ALLOW_ORDINARY_GRAPH_STREAM_REDUCTION");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

std::size_t configure_prefill_persisting_l2(int device,std::size_t requested) {
    std::lock_guard<std::mutex> lock(prefill_persisting_l2_mutex);
    if(const auto found=prefill_persisting_l2_by_device.find(device);
       found!=prefill_persisting_l2_by_device.end())return found->second;
    int maximum=0;
    if(cudaDeviceGetAttribute(&maximum,cudaDevAttrMaxPersistingL2CacheSize,device)!=cudaSuccess ||
       maximum<=0) {
        (void)cudaGetLastError();
        prefill_persisting_l2_by_device.emplace(device,0);
        return 0;
    }
    const auto bytes=std::min(requested,static_cast<std::size_t>(maximum));
    if(!bytes || cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,bytes)!=cudaSuccess) {
        (void)cudaGetLastError();
        prefill_persisting_l2_by_device.emplace(device,0);
        return 0;
    }
    prefill_persisting_l2_by_device.emplace(device,bytes);
    return bytes;
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ half2 decode_mul1_product_2(std::uint32_t product0,
                                                        std::uint32_t product1) {
    const std::uint32_t sum0 =
        __dp4a(product0, 0x01010101u, static_cast<std::uint32_t>(kMul1AccumulatorHalf));
    const std::uint32_t sum1 =
        __dp4a(product1, 0x01010101u, static_cast<std::uint32_t>(kMul1AccumulatorHalf));
    const half2 input = __halves2half2(
        __ushort_as_half(static_cast<std::uint16_t>(sum0)),
        __ushort_as_half(static_cast<std::uint16_t>(sum1)));
    return __hfma2(input,
                   __half2half2(__ushort_as_half(kMul1InverseHalf)),
                   __half2half2(__ushort_as_half(kMul1BiasHalf)));
}

__device__ __forceinline__ std::uint32_t packed_fshift(std::uint32_t high,
                                                        std::uint32_t low,
                                                        int shift) {
    const std::uint64_t merged =
        (static_cast<std::uint64_t>(low) << 32u) | static_cast<std::uint64_t>(high);
    return static_cast<std::uint32_t>(merged >> shift);
}

__device__ __forceinline__ void exl3_cp_async_16(void* shared_ptr,
                                                  const void* global_ptr) {
    const std::uint32_t shared_address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(shared_address), "l"(global_ptr));
}

__device__ __forceinline__ void exl3_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void exl3_cp_async_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// Wait until at most Pending committed cp.async groups remain in flight.
template <int Pending>
__device__ __forceinline__ void exl3_cp_async_wait_pending() {
    static_assert(Pending >= 0 && Pending <= 8, "bounded cp.async pipeline depth");
    asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending));
}

template <int Groups>
__device__ __forceinline__ void exl3_cp_async_wait_group() {
    static_assert(Groups >= 0 && Groups <= 2,
                  "the native pipeline uses only wait groups 0..2");
    if constexpr (Groups == 0)
        asm volatile("cp.async.wait_group 0;\n" ::);
    else if constexpr (Groups == 1)
        asm volatile("cp.async.wait_group 1;\n" ::);
    else
        asm volatile("cp.async.wait_group 2;\n" ::);
}

// The Mia shape-4 leaf keeps all K-slice CTAs resident and orders their
// cross-CTA accumulation through a per-output-block global lock.  Keep this
// helper deliberately separate from the established split-plane reduction:
// the candidate uses an otherwise-unused accumulation row only after the
// workspace has proved that a second row exists.
__device__ __forceinline__ void exl3_global_slice_acquire(
    int* lock, int stage) {
    if (threadIdx.x == 0) {
        int state = 0;
        do {
            state = atomicAdd(lock, 0);
        } while (state != stage);
    }
    __syncthreads();
}

__device__ __forceinline__ void exl3_global_slice_release(
    int* lock, int next_stage, bool reset) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(lock, reset ? 0 : next_stage);
    }
}

template <int Bits>
bool launch_fast_same_weights_int8_gemv(
    const std::uint16_t* input, const Exl3CudaLinearWeights& weights,
    std::uint16_t* output, int input_features, int output_features,
    float* workspace, std::size_t workspace_bytes, cudaStream_t stream,
    bool occupancy_grid) {
    static_assert(Bits == 6 || Bits == 7);
    if (!input || !weights.trellis || !weights.suh || !weights.svh || !output ||
        !workspace || input_features % 128 != 0 || output_features % 256 != 0) {
        return false;
    }
    constexpr int threads = NUM_THREADS;
    constexpr int candidate_blocks_per_sm = 6;
    int device = 0;
    int sms = 0;
    cuda_check(cudaGetDevice(&device), "FAST_SAME_WEIGHTS INT8 device query");
    cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
               "FAST_SAME_WEIGHTS INT8 SM query");
    int grid = std::min(1024, std::max(1, candidate_blocks_per_sm * sms));
    const int rows_total = input_features / 16;
    const int groups = output_features / 256;
    auto kernel = exl3_gemv_int8_sq_kernel<Bits, 1, false, false>;
    // The kernel mirrors this decomposition from gridDim.x.  Choose the final
    // grid before deriving rows_per, ksplit, workspace, and dynamic shared
    // memory; otherwise an occupancy-selected grid would make the host and
    // device layouts disagree and allow asynchronous shared/workspace OOB.
    const auto rows_for_grid = [&](int grid_) {
        const int r = (rows_total * groups + grid_ - 1) / grid_;
        int rows = (std::max(r, std::min(2 * r, 32)) + 7) & ~7;
        rows = std::max(rows, SQ_MINROWS);
        rows = std::min(rows, gemv_int8_sq_rows_max(1, false));
        rows = std::min(rows, (rows_total + 7) & ~7);
        return rows;
    };
    const auto shared_for_rows = [&](int rows) {
        const std::size_t stage_bytes = gemv_int8_stage_smem(Bits)
            ? 8u * GEMV_STAGE_D * 16u * static_cast<std::size_t>(Bits) *
                  sizeof(std::uint32_t)
            : 0u;
        return static_cast<std::size_t>(rows) * 16u * sizeof(half) +
            static_cast<std::size_t>(rows) * 16u * sizeof(std::uint32_t) +
            stage_bytes + 2u * 128u * sizeof(float);
    };
    if (occupancy_grid) {
        // Match Mia's occupancy choice: query the occupancy for the shared
        // footprint implied by the initial six-block grid, while advertising
        // the rows-max upper bound to CUDA. The final host/device layout is
        // recomputed below after the selected grid is known.
        const int rows_max = gemv_int8_sq_rows_max(1, false);
        const std::size_t occupancy_shared_bytes = shared_for_rows(rows_max);
        const std::size_t smem_guess = shared_for_rows(rows_for_grid(grid));
        cuda_check(cudaFuncSetAttribute(
                       kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(occupancy_shared_bytes)),
                   "FAST_SAME_WEIGHTS INT8 occupancy shared-memory attribute");
        cuda_check(cudaFuncSetAttribute(
                       kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared),
                   "FAST_SAME_WEIGHTS INT8 shared-memory carveout");
        int active = 1;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active, kernel, threads, smem_guess),
                   "FAST_SAME_WEIGHTS INT8 occupancy query");
        grid = std::min(1024, std::max(1, active * sms));
    }
    const int rows_per = rows_for_grid(grid);
    const int splits = (rows_total + rows_per - 1) / rows_per;
    const std::size_t required_ints = SQ_WS_RESERVED +
        static_cast<std::size_t>(splits) * output_features;
    if (required_ints * sizeof(int) > workspace_bytes || groups > SQ_COUNTERS_CAP ||
        splits > SQ_KSPLIT_CAP) {
        return false;
    }
    const std::size_t shared_bytes = shared_for_rows(rows_per);
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared_bytes)),
               "FAST_SAME_WEIGHTS INT8 shared-memory attribute");
    cuda_check(cudaMemsetAsync(workspace, 0, SQ_COUNTERS_CAP * sizeof(int), stream),
               "FAST_SAME_WEIGHTS INT8 counter reset");
    kernel<<<grid, threads, shared_bytes, stream>>>(
        reinterpret_cast<const half*>(input), weights.trellis,
        reinterpret_cast<void*>(output), 1, input_features, output_features,
        reinterpret_cast<int*>(workspace), reinterpret_cast<const half*>(weights.suh),
        nullptr, reinterpret_cast<const half*>(weights.svh));
    cuda_check(cudaGetLastError(), "launch FAST_SAME_WEIGHTS INT8 GEMV");
    return true;
}

void cublas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

__device__ __forceinline__ void decode4_k5(const std::uint32_t* packed,
                                            int t0,
                                            std::uint32_t& w0,
                                            std::uint32_t& w1,
                                            std::uint32_t& w2,
                                            std::uint32_t& w3);

struct Exl3FragA { half2 values[4]; };
struct Exl3FragB { half2 values[2]; };
struct Exl3FragC { float values[4]; };
struct Exl3FragCHalf { half2 values[2]; };

__device__ __forceinline__ void exl3_ldsm4(Exl3FragA& fragment,
                                            const void* shared_ptr) {
    auto* values = reinterpret_cast<std::uint32_t*>(&fragment);
    const std::uint32_t address =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]),
                   "=r"(values[3])
                 : "r"(address));
}

__device__ __forceinline__ void exl3_mma_m16n8k16(const Exl3FragA& a,
                                                  const Exl3FragB& b,
                                                  Exl3FragC& c) {
    const auto* a_words = reinterpret_cast<const std::uint32_t*>(&a);
    const auto* b_words = reinterpret_cast<const std::uint32_t*>(&b);
    auto* c_values = reinterpret_cast<float*>(&c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c_values[0]), "+f"(c_values[1]), "+f"(c_values[2]), "+f"(c_values[3])
        : "r"(a_words[0]), "r"(a_words[1]), "r"(a_words[2]), "r"(a_words[3]),
          "r"(b_words[0]), "r"(b_words[1]));
}

__device__ __forceinline__ void exl3_dq4_k5(const std::uint32_t* packed,
                                             int t_offset,
                                             Exl3FragB& fragment,
                                             std::uint32_t mul1_multiplier) {
    std::uint32_t w0, w1, w2, w3;
    decode4_k5(packed, t_offset, w0, w1, w2, w3);
    fragment.values[0] = decode_mul1_product_2(w0 * mul1_multiplier,
                                                w1 * mul1_multiplier);
    fragment.values[1] = decode_mul1_product_2(w2 * mul1_multiplier,
                                                w3 * mul1_multiplier);
}

// A narrow port of the pinned upstream shape-4 regular EXL3 kernel. Each block
// covers 32 packed 16-column tiles (512 output columns), uses ldmatrix/mma for
// the dot product, stages original trellis bytes with cp.async, and owns one
// of five K-splits. The split reduction stays in the packed representation.
__global__ void exl3_prefill_shape4_partials_kernel(const std::uint16_t* transformed,
                                           const std::uint16_t* trellis,
                                           const std::int32_t* mul1,
                                           float* accum,
                                           int rows) {
    const int total_rows=rows;
    const int row_base=static_cast<int>(blockIdx.y)*16;
    rows=min(16,total_rows-row_base);
    transformed+=static_cast<std::size_t>(row_base)*kInput;
    constexpr int output_tiles_per_block = 32;
    constexpr int fragments_per_warp = 8;
    constexpr int raw_stage_half = output_tiles_per_block * 80;
    extern __shared__ half shared[];
    half* sh_a = shared;
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(sh_a + 256);
    float* sh_c = reinterpret_cast<float*>(sh_raw + 2 * raw_stage_half);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    constexpr int output_blocks = kShape4OutputBlocks;
    constexpr int k_splits = kShape4Splits;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (block % output_blocks) * output_tiles_per_block;
    const int split = block / output_blocks;
    const int tiles_per_split = (kTilesK + k_splits - 1) / k_splits;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, kTilesK);
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tile_k_end) { return; }
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * kTilesN + tile_base) * 80u;
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += kThreads) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        exl3_cp_async_commit();
    };

    prefetch(tile_k_begin, 0);
    exl3_cp_async_wait();
    __syncthreads();

    Exl3FragC c[fragments_per_warp];
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        #pragma unroll
        for (float& value : c[n].values) { value = 0.0f; }
    }

    int raw_stage = 0;
    for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
        // The A tile is the 16-column slice of the already transformed row.
        // Rows 1..15 are zero padding required by the m16n8k16 fragment.
        for (int i = thread; i < 256; i += kThreads) {
            const int row = i / 16;
            const int column = i % 16;
            const int source_column = (column / 8 ^ ((row >> 2) & 1)) * 8 + column % 8;
            sh_a[i] = row < rows
                ? __ushort_as_half(transformed[row * kInput + tile_k * 16 + source_column])
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        prefetch(tile_k + 1, 1 - raw_stage);

        Exl3FragA a;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a, sh_a + r * 16 + c_swizzled * 8);

        #pragma unroll
        for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
            const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
            const auto* packed = reinterpret_cast<const std::uint32_t*>(
                sh_raw + raw_stage * raw_stage_half + sub_n2 * 80);
            Exl3FragB b0, b1;
            exl3_dq4_k5(packed, lane << 3, b0, mul1_multiplier);
            exl3_dq4_k5(packed, (lane << 3) + 4, b1, mul1_multiplier);
            exl3_mma_m16n8k16(a, b0, c[n2]);
            exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
        }

        if (tile_k + 1 < tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage = 1 - raw_stage;
    }

    // The fragment layout is the same one used by ExLlamaV3's regular GEMM
    // output staging: two adjacent values per lane and eight fragments per warp.
    const int n0 = warp * fragments_per_warp;
    const int r0 = lane / 4;
    const int r1 = r0 + 8;
    const int column = (lane % 4) * 2;
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        if (r0 < rows) {
            float* destination = sh_c + r0 * 512 + (n0 + n) * 8 + column;
            destination[0] = c[n].values[0];
            destination[1] = c[n].values[1];
        }
        if (r1 < rows) {
            float* destination = sh_c + r1 * 512 + (n0 + n) * 8 + column;
            destination[0] = c[n].values[2];
            destination[1] = c[n].values[3];
        }
    }
    __syncthreads();

    // Each row group writes disjoint five-split partials. The next ordered
    // reduction/output launch replaces the original cooperative grid barrier.
    const std::size_t partial_stride = static_cast<std::size_t>(total_rows) * kOutput;
    for (int i = thread; i < rows * 512; i += kThreads) {
        const int row = i / 512;
        const int column = i % 512;
        accum[static_cast<std::size_t>(split) * partial_stride +
              (row_base+row) * kOutput + tile_base * 16 + column] = sh_c[i];
    }
}

template<int OutputTilesPerBlock=32>
__global__ void exl3_prefill_shape4_direct_kernel(const std::uint16_t* transformed,
                                           const std::uint16_t* trellis,
                                           const std::int32_t* mul1,
                                           float* accum,
                                           int rows) {
    const int total_rows=rows;
    const int row_base=static_cast<int>(blockIdx.y)*16;
    rows=min(16,total_rows-row_base);
    transformed+=static_cast<std::size_t>(row_base)*kInput;
    constexpr int output_tiles_per_block = OutputTilesPerBlock;
    constexpr int fragments_per_warp = output_tiles_per_block/4;
    constexpr int raw_stage_half = output_tiles_per_block * 80;
    static_assert(output_tiles_per_block==32||output_tiles_per_block==64,
        "shape4 direct output topology");
    extern __shared__ half shared[];
    half* sh_a = shared;
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(sh_a + 256);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    constexpr int output_blocks = kTilesN/output_tiles_per_block;
    constexpr int k_splits = kShape4Splits;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (block % output_blocks) * output_tiles_per_block;
    const int split = block / output_blocks;
    const int tiles_per_split = (kTilesK + k_splits - 1) / k_splits;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, kTilesK);
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tile_k_end) { return; }
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * kTilesN + tile_base) * 80u;
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += kThreads) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        exl3_cp_async_commit();
    };

    prefetch(tile_k_begin, 0);
    exl3_cp_async_wait();
    __syncthreads();

    Exl3FragC c[fragments_per_warp];
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        #pragma unroll
        for (float& value : c[n].values) { value = 0.0f; }
    }

    int raw_stage = 0;
    for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
        // The A tile is the 16-column slice of the already transformed row.
        // Rows 1..15 are zero padding required by the m16n8k16 fragment.
        for (int i = thread; i < 256; i += kThreads) {
            const int row = i / 16;
            const int column = i % 16;
            const int source_column = (column / 8 ^ ((row >> 2) & 1)) * 8 + column % 8;
            sh_a[i] = row < rows
                ? __ushort_as_half(transformed[row * kInput + tile_k * 16 + source_column])
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        prefetch(tile_k + 1, 1 - raw_stage);

        Exl3FragA a;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a, sh_a + r * 16 + c_swizzled * 8);

        #pragma unroll
        for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
            const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
            const auto* packed = reinterpret_cast<const std::uint32_t*>(
                sh_raw + raw_stage * raw_stage_half + sub_n2 * 80);
            Exl3FragB b0, b1;
            exl3_dq4_k5(packed, lane << 3, b0, mul1_multiplier);
            exl3_dq4_k5(packed, (lane << 3) + 4, b1, mul1_multiplier);
            exl3_mma_m16n8k16(a, b0, c[n2]);
            exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
        }

        if (tile_k + 1 < tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage = 1 - raw_stage;
    }

    const int n0=warp*fragments_per_warp;
    const int r0=lane/4, r1=r0+8, column=(lane%4)*2;
    const std::size_t stride=static_cast<std::size_t>(total_rows)*kOutput;
    #pragma unroll
    for(int n=0;n<fragments_per_warp;++n) {
        const int col=tile_base*16+(n0+n)*8+column;
        if(r0<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+static_cast<std::size_t>(row_base+r0)*kOutput+col;
            dst[0]=c[n].values[0];dst[1]=c[n].values[1];
        }
        if(r1<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+static_cast<std::size_t>(row_base+r1)*kOutput+col;
            dst[0]=c[n].values[2];dst[1]=c[n].values[3];
        }
    }
}

__global__ void exl3_gemm_m1_shape4_kernel(const std::uint16_t* transformed,
                                           const std::uint16_t* trellis,
                                           const std::int32_t* mul1,
                                           float* accum,
                                           int rows) {
    constexpr int output_tiles_per_block = 32;
    constexpr int fragments_per_warp = 8;
    constexpr int raw_stage_half = output_tiles_per_block * 80;
    extern __shared__ half shared[];
    half* sh_a = shared;
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(sh_a + 256);
    float* sh_c = reinterpret_cast<float*>(sh_raw + 2 * raw_stage_half);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    constexpr int output_blocks = kShape4OutputBlocks;
    constexpr int k_splits = kShape4Splits;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (block % output_blocks) * output_tiles_per_block;
    const int split = block / output_blocks;
    const int tiles_per_split = (kTilesK + k_splits - 1) / k_splits;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, kTilesK);
    auto grid = cg::this_grid();
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tile_k_end) { return; }
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * kTilesN + tile_base) * 80u;
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += kThreads) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        exl3_cp_async_commit();
    };

    prefetch(tile_k_begin, 0);
    exl3_cp_async_wait();
    __syncthreads();

    Exl3FragC c[fragments_per_warp];
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        #pragma unroll
        for (float& value : c[n].values) { value = 0.0f; }
    }

    int raw_stage = 0;
    for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
        // The A tile is the 16-column slice of the already transformed row.
        // Rows 1..15 are zero padding required by the m16n8k16 fragment.
        for (int i = thread; i < 256; i += kThreads) {
            const int row = i / 16;
            const int column = i % 16;
            const int source_column = (column / 8 ^ ((row >> 2) & 1)) * 8 + column % 8;
            sh_a[i] = row < rows
                ? __ushort_as_half(transformed[row * kInput + tile_k * 16 + source_column])
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        prefetch(tile_k + 1, 1 - raw_stage);

        Exl3FragA a;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a, sh_a + r * 16 + c_swizzled * 8);

        #pragma unroll
        for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
            const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
            const auto* packed = reinterpret_cast<const std::uint32_t*>(
                sh_raw + raw_stage * raw_stage_half + sub_n2 * 80);
            Exl3FragB b0, b1;
            exl3_dq4_k5(packed, lane << 3, b0, mul1_multiplier);
            exl3_dq4_k5(packed, (lane << 3) + 4, b1, mul1_multiplier);
            exl3_mma_m16n8k16(a, b0, c[n2]);
            exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
        }

        if (tile_k + 1 < tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage = 1 - raw_stage;
    }

    // The fragment layout is the same one used by ExLlamaV3's regular GEMM
    // output staging: two adjacent values per lane and eight fragments per warp.
    const int n0 = warp * fragments_per_warp;
    const int r0 = lane / 4;
    const int r1 = r0 + 8;
    const int column = (lane % 4) * 2;
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        if (r0 < rows) {
            float* destination = sh_c + r0 * 512 + (n0 + n) * 8 + column;
            destination[0] = c[n].values[0];
            destination[1] = c[n].values[1];
        }
        if (r1 < rows) {
            float* destination = sh_c + r1 * 512 + (n0 + n) * 8 + column;
            destination[0] = c[n].values[2];
            destination[1] = c[n].values[3];
        }
    }
    __syncthreads();

    // All five K-split blocks for a column tile participate in one cooperative
    // grid. Every split stages its partial sum in its own slot of the persistent
    // accumulation buffer (sized rows*kOutput*kShape4Splits); after one grid
    // barrier, split zero reduces splits 1..4 in fixed order. This mirrors the
    // generic MMA kernel deterministic reduction pattern. The previous split-0-init
    // plus atomicAdd protocol made FP summation order depend on block arrival
    // order, i.e. nondeterministic across launches (E5A3 root-caused target-tap
    // run-to-run divergence and thin-margin draft proposal flips to it). Same
    // arithmetic, fixed order; no extra launches.
    const std::size_t partial_stride = static_cast<std::size_t>(rows) * kOutput;
    for (int i = thread; i < rows * 512; i += kThreads) {
        const int row = i / 512;
        const int column = i % 512;
        accum[static_cast<std::size_t>(split) * partial_stride +
              row * kOutput + tile_base * 16 + column] = sh_c[i];
    }
    grid.sync();
    if (split == 0) {
        for (int i = thread; i < rows * 512; i += kThreads) {
            const int row = i / 512;
            const int column = i % 512;
            float value = sh_c[i];
            for (int other = 1; other < k_splits; ++other) {
                value += accum[static_cast<std::size_t>(other) * partial_stride +
                               row * kOutput + tile_base * 16 + column];
            }
            accum[row * kOutput + tile_base * 16 + column] = value;
        }
    }
}

// K=5 extraction follows ExLlamaV3's dq4 path. Four circular 16-bit windows are
// recovered from the two boundary uint32 words; the pair codebook helper then
// reconstructs eight half weights with two dp4a instructions per pair.
__device__ __forceinline__ void decode4_k5(const std::uint32_t* packed,
                                            int t0,
                                            std::uint32_t& w0,
                                            std::uint32_t& w1,
                                            std::uint32_t& w2,
                                            std::uint32_t& w3) {
    constexpr int bits = kK;
    constexpr int words32 = bits * 256 / 32;
    const int b0 = (t0 + 257) * bits - 16;
    const int b2 = b0 + 3 * bits + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int shift = (i2 + 1) * 32 - b2;
    const std::uint32_t a = packed[i0 % words32];
    const std::uint32_t b = packed[i2 % words32];
    w3 = packed_fshift(b, a, shift) & 0xffffu;
    w2 = packed_fshift(b, a, shift + bits) & 0xffffu;
    w1 = packed_fshift(b, a, shift + bits * 2) & 0xffffu;
    w0 = packed_fshift(b, a, shift + bits * 3) & 0xffffu;
}

__device__ __forceinline__ float half_product(std::uint16_t left,
                                               std::uint16_t right) {
    return __half2float(__ushort_as_half(left)) * __half2float(__ushort_as_half(right));
}

template <int Block, bool RoundProductToHalf=false, bool GateUp=false>
__global__ void input_hadamard_kernel(const std::uint16_t* input,
                                      const std::uint16_t* suh,
                                      std::uint16_t* transformed,
                                      int rows,
                                      int input_features,
                                      const std::uint16_t* up=nullptr,
                                      std::uint16_t* activation=nullptr) {
    __shared__ float values[kHadamard];
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= Block) { return; }

    const int offset = block * kHadamard + lane;
    const int element=row * input_features + offset;
    std::uint16_t represented=input[element];
    if constexpr(GateUp) {
        const float g=__half2float(__ushort_as_half(represented));
        const float u=__half2float(__ushort_as_half(up[element]));
        represented=__half_as_ushort(__float2half_rn((g/(1.0f+expf(-g)))*u));
        activation[element]=represented;
    }
    float product = half_product(represented, suh[offset]);
    if constexpr (RoundProductToHalf) product=__half2float(__float2half_rn(product));
    values[lane] = product;
    __syncthreads();
    for (int width = 1; width < kHadamard; width *= 2) {
        if ((lane % (2 * width)) < width) {
            const int index = lane;
            const float left = values[index];
            const float right = values[index + width];
            values[index] = left + right;
            values[index + width] = left - right;
        }
        __syncthreads();
    }
    transformed[row * input_features + offset] =
        __half_as_ushort(__float2half_rn(values[lane] * kHadamardScale));
}

__device__ __forceinline__ void exl3_mma_m16n8k16_half(
    const Exl3FragA& a, const Exl3FragB& b, Exl3FragCHalf& c) {
    const auto* a_words = reinterpret_cast<const std::uint32_t*>(&a);
    const auto* b_words = reinterpret_cast<const std::uint32_t*>(&b);
    auto* c_words = reinterpret_cast<std::uint32_t*>(&c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c_words[0]), "+r"(c_words[1])
        : "r"(a_words[0]), "r"(a_words[1]), "r"(a_words[2]), "r"(a_words[3]),
          "r"(b_words[0]), "r"(b_words[1]));
}

template <int Block, bool ReuseEqual = false>
__global__ void input_hadamard_pair_kernel(
    const std::uint16_t* input,
    const std::uint16_t* first_suh,
    const std::uint16_t* second_suh,
    std::uint16_t* first_transformed,
    std::uint16_t* second_transformed,
    int rows,
    int input_features) {
    __shared__ float first_values[kHadamard];
    __shared__ float second_values[kHadamard];
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= Block) { return; }

    const int offset = block * kHadamard + lane;
    const std::uint16_t input_value = input[row * input_features + offset];
    bool shared_transform=false;
    if constexpr(ReuseEqual)
        shared_transform=__syncthreads_and(first_suh[offset]==second_suh[offset]);
    first_values[lane] = half_product(input_value, first_suh[offset]);
    if(!shared_transform)second_values[lane] = half_product(input_value, second_suh[offset]);
    __syncthreads();
    for (int width = 1; width < kHadamard; width *= 2) {
        if ((lane % (2 * width)) < width) {
            const int index = lane;
            const float first_left = first_values[index];
            const float first_right = first_values[index + width];
            first_values[index] = first_left + first_right;
            first_values[index + width] = first_left - first_right;
            if(!shared_transform) {
                const float second_left = second_values[index];
                const float second_right = second_values[index + width];
                second_values[index] = second_left + second_right;
                second_values[index + width] = second_left - second_right;
            }
        }
        __syncthreads();
    }
    const auto first_result=__half_as_ushort(__float2half_rn(first_values[lane] * kHadamardScale));
    first_transformed[row * input_features + offset] = first_result;
    second_transformed[row * input_features + offset] = shared_transform ?
        first_result :
        __half_as_ushort(__float2half_rn(second_values[lane] * kHadamardScale));
}

__global__ void output_hadamard_kernel(const float* accum,
                                       const std::uint16_t* svh,
                                       std::uint16_t* output,
                                       int rows,
                                       int output_features) {
    __shared__ float values[kHadamard];
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kHadamard) { return; }

    const int offset = block * kHadamard + lane;
    values[lane] = accum[row * output_features + offset];
    __syncthreads();
    for (int width = 1; width < kHadamard; width *= 2) {
        if ((lane % (2 * width)) < width) {
            const int index = lane;
            const float left = values[index];
            const float right = values[index + width];
            values[index] = left + right;
            values[index + width] = left - right;
        }
        __syncthreads();
    }
    const auto normalized = __float2half_rn(values[lane] * kHadamardScale);
    const auto scale = __ushort_as_half(svh[offset]);
    output[row * output_features + offset] = __half_as_ushort(__hmul(normalized, scale));
}

// Mia's reconstruct+hgemm path keeps the cuBLAS destination in FP16 while
// retaining FP32 tensor-core accumulation, then applies the output Hadamard
// in place.  The established native numeric route intentionally uses a
// float destination and the historical output kernel.  Keep this separate so
// the same-weight Fast backend can match the comparator's destination
// semantics without changing the exact/reference route.
__global__ void output_hadamard_fp16_inplace_kernel(
    std::uint16_t* output,
    const std::uint16_t* svh,
    int rows,
    int output_features) {
    __shared__ float values[kHadamard];
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kHadamard) return;

    const int offset = block * kHadamard + lane;
    values[lane] = __half2float(__ushort_as_half(
        output[row * output_features + offset]));
    __syncthreads();
    for (int width = 1; width < kHadamard; width *= 2) {
        if ((lane % (2 * width)) < width) {
            const float left = values[lane];
            const float right = values[lane + width];
            values[lane] = left + right;
            values[lane + width] = left - right;
        }
        __syncthreads();
    }
    const auto normalized = __float2half_rn(values[lane] * kHadamardScale);
    const auto scale = __ushort_as_half(svh[offset]);
    output[row * output_features + offset] =
        __half_as_ushort(__hmul(normalized, scale));
}


// Warp-per-block twins of the 128-point transforms: each warp owns one
// (row, 128-column) block with four elements per lane. Widths 1 and 2 pair
// elements inside a lane, wider widths pair lanes through shfl_xor. Every
// butterfly output is the same single FP32 add (lower+upper) or subtract
// (lower-upper) of the same operands as the shared-memory kernels, so the
// transformed bits are identical; only the CTA/barrier structure changes.
bool exl3_hadamard_warp_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_HADAMARD_WARP");
        return !value || std::strcmp(value,"0")!=0;
    }();
    return enabled;
}

__device__ __forceinline__ void exl3_warp_butterflies(float (&v)[4],int lane) {
    {
        const float a=v[0],b=v[1],c=v[2],d=v[3];
        v[0]=a+b; v[1]=a-b; v[2]=c+d; v[3]=c-d;
    }
    {
        const float a=v[0],b=v[1],c=v[2],d=v[3];
        v[0]=a+c; v[2]=a-c; v[1]=b+d; v[3]=b-d;
    }
    #pragma unroll
    for(int mask=1;mask<32;mask<<=1) {
        const bool lower=(lane&mask)==0;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const float partner=__shfl_xor_sync(0xffffffffU,v[j],mask);
            v[j]=lower?v[j]+partner:partner-v[j];
        }
    }
}

__device__ __forceinline__ void exl3_load_half4(const std::uint16_t* p,std::uint16_t (&h)[4]) {
    const uint2 bits=*reinterpret_cast<const uint2*>(p);
    h[0]=static_cast<std::uint16_t>(bits.x&0xffffu);
    h[1]=static_cast<std::uint16_t>(bits.x>>16);
    h[2]=static_cast<std::uint16_t>(bits.y&0xffffu);
    h[3]=static_cast<std::uint16_t>(bits.y>>16);
}

__device__ __forceinline__ void exl3_store_half4(std::uint16_t* p,const std::uint16_t (&h)[4]) {
    uint2 bits;
    bits.x=static_cast<unsigned>(h[0])|(static_cast<unsigned>(h[1])<<16);
    bits.y=static_cast<unsigned>(h[2])|(static_cast<unsigned>(h[3])<<16);
    *reinterpret_cast<uint2*>(p)=bits;
}

constexpr int kHadamardWarpsPerBlock=8;

template<bool RoundProductToHalf,bool GateUp>
__global__ void __launch_bounds__(kHadamardWarpsPerBlock*32) input_hadamard_warp_kernel(
    const std::uint16_t* input,const std::uint16_t* suh,std::uint16_t* transformed,
    int rows,int input_features,const std::uint16_t* up,std::uint16_t* activation) {
    EXL3_PDL_PROLOGUE();
    const int lane=static_cast<int>(threadIdx.x)&31;
    const int blocks=input_features/kHadamard;
    const long long task=static_cast<long long>(blockIdx.x)*kHadamardWarpsPerBlock+
        (static_cast<int>(threadIdx.x)>>5);
    if(task>=static_cast<long long>(rows)*blocks) return;
    const int row=static_cast<int>(task/blocks);
    const int block=static_cast<int>(task%blocks);
    const int offset=block*kHadamard+lane*4;
    const std::size_t element=static_cast<std::size_t>(row)*input_features+offset;
    std::uint16_t represented[4],scale[4];
    exl3_load_half4(input+element,represented);
    exl3_load_half4(suh+offset,scale);
    if constexpr(GateUp) {
        std::uint16_t up_bits[4];
        exl3_load_half4(up+element,up_bits);
        #pragma unroll
        for(int j=0;j<4;++j) {
            const float g=__half2float(__ushort_as_half(represented[j]));
            const float u=__half2float(__ushort_as_half(up_bits[j]));
            represented[j]=__half_as_ushort(__float2half_rn((g/(1.0f+expf(-g)))*u));
        }
        exl3_store_half4(activation+element,represented);
    }
    float v[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        float product=half_product(represented[j],scale[j]);
        if constexpr(RoundProductToHalf) product=__half2float(__float2half_rn(product));
        v[j]=product;
    }
    exl3_warp_butterflies(v,lane);
    std::uint16_t result[4];
    #pragma unroll
    for(int j=0;j<4;++j) result[j]=__half_as_ushort(__float2half_rn(v[j]*kHadamardScale));
    exl3_store_half4(transformed+element,result);
}

__global__ void __launch_bounds__(kHadamardWarpsPerBlock*32) output_hadamard_warp_kernel(
    const float* accum,const std::uint16_t* svh,std::uint16_t* output,int rows,
    int output_features) {
    EXL3_PDL_PROLOGUE();
    const int lane=static_cast<int>(threadIdx.x)&31;
    const int blocks=output_features/kHadamard;
    const long long task=static_cast<long long>(blockIdx.x)*kHadamardWarpsPerBlock+
        (static_cast<int>(threadIdx.x)>>5);
    if(task>=static_cast<long long>(rows)*blocks) return;
    const int row=static_cast<int>(task/blocks);
    const int block=static_cast<int>(task%blocks);
    const int offset=block*kHadamard+lane*4;
    const std::size_t element=static_cast<std::size_t>(row)*output_features+offset;
    const float4 loaded=*reinterpret_cast<const float4*>(accum+element);
    float v[4]={loaded.x,loaded.y,loaded.z,loaded.w};
    std::uint16_t scale[4];
    exl3_load_half4(svh+offset,scale);
    exl3_warp_butterflies(v,lane);
    std::uint16_t result[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        const auto normalized=__float2half_rn(v[j]*kHadamardScale);
        result[j]=__half_as_ushort(__hmul(normalized,__ushort_as_half(scale[j])));
    }
    exl3_store_half4(output+element,result);
}

__global__ void __launch_bounds__(kHadamardWarpsPerBlock*32) output_hadamard_fp16_inplace_warp_kernel(
    std::uint16_t* output,const std::uint16_t* svh,int rows,int output_features) {
    EXL3_PDL_PROLOGUE();
    const int lane=static_cast<int>(threadIdx.x)&31;
    const int blocks=output_features/kHadamard;
    const long long task=static_cast<long long>(blockIdx.x)*kHadamardWarpsPerBlock+
        (static_cast<int>(threadIdx.x)>>5);
    if(task>=static_cast<long long>(rows)*blocks) return;
    const int row=static_cast<int>(task/blocks);
    const int block=static_cast<int>(task%blocks);
    const int offset=block*kHadamard+lane*4;
    const std::size_t element=static_cast<std::size_t>(row)*output_features+offset;
    std::uint16_t loaded[4],scale[4];
    exl3_load_half4(output+element,loaded);
    exl3_load_half4(svh+offset,scale);
    float v[4];
    #pragma unroll
    for(int j=0;j<4;++j) v[j]=__half2float(__ushort_as_half(loaded[j]));
    exl3_warp_butterflies(v,lane);
    std::uint16_t result[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        const auto normalized=__float2half_rn(v[j]*kHadamardScale);
        result[j]=__half_as_ushort(__hmul(normalized,__ushort_as_half(scale[j])));
    }
    exl3_store_half4(output+element,result);
}

inline unsigned exl3_hadamard_warp_grid(int rows,int features) {
    const long long tasks=static_cast<long long>(rows)*(features/kHadamard);
    return static_cast<unsigned>((tasks+kHadamardWarpsPerBlock-1)/kHadamardWarpsPerBlock);
}

inline bool exl3_hadamard_warp_aligned(const void* a,const void* b,const void* c=nullptr,
                                       const void* d=nullptr,const void* e=nullptr) {
    const auto bits=reinterpret_cast<std::uintptr_t>(a)|reinterpret_cast<std::uintptr_t>(b)|
        reinterpret_cast<std::uintptr_t>(c)|reinterpret_cast<std::uintptr_t>(d)|
        reinterpret_cast<std::uintptr_t>(e);
    return (bits&15u)==0;
}

template<int Block,bool RoundProductToHalf=false,bool GateUp=false>
void launch_input_hadamard(cudaStream_t stream,const std::uint16_t* input,
    const std::uint16_t* suh,std::uint16_t* transformed,int rows,int input_features,
    const std::uint16_t* up=nullptr,std::uint16_t* activation=nullptr) {
    static_assert(Block==kHadamard);
    if(rows>0 && exl3_hadamard_warp_enabled() && input_features%kHadamard==0 &&
       exl3_hadamard_warp_aligned(input,suh,transformed,up,activation))
        exl3_launch_pdl(input_hadamard_warp_kernel<RoundProductToHalf,GateUp>,
            dim3(exl3_hadamard_warp_grid(rows,input_features)),dim3(kHadamardWarpsPerBlock*32),
            0,stream,input,suh,transformed,rows,input_features,up,activation);
    else
        input_hadamard_kernel<Block,RoundProductToHalf,GateUp><<<
            dim3(rows,input_features/kHadamard),dim3(kHadamard),0,stream>>>(
                input,suh,transformed,rows,input_features,up,activation);
}

inline void launch_output_hadamard(cudaStream_t stream,const float* accum,
    const std::uint16_t* svh,std::uint16_t* output,int rows,int output_features) {
    if(rows>0 && exl3_hadamard_warp_enabled() && output_features%kHadamard==0 &&
       exl3_hadamard_warp_aligned(accum,svh,output))
        exl3_launch_pdl(output_hadamard_warp_kernel,
            dim3(exl3_hadamard_warp_grid(rows,output_features)),dim3(kHadamardWarpsPerBlock*32),
            0,stream,accum,svh,output,rows,output_features);
    else
        output_hadamard_kernel<<<dim3(rows,output_features/kHadamard),dim3(kHadamard),0,
            stream>>>(accum,svh,output,rows,output_features);
}

inline void launch_output_hadamard_fp16_inplace(cudaStream_t stream,std::uint16_t* output,
    const std::uint16_t* svh,int rows,int output_features) {
    if(rows>0 && exl3_hadamard_warp_enabled() && output_features%kHadamard==0 &&
       exl3_hadamard_warp_aligned(output,svh))
        exl3_launch_pdl(output_hadamard_fp16_inplace_warp_kernel,
            dim3(exl3_hadamard_warp_grid(rows,output_features)),dim3(kHadamardWarpsPerBlock*32),
            0,stream,output,svh,rows,output_features);
    else
        output_hadamard_fp16_inplace_kernel<<<dim3(rows,output_features/kHadamard),
            dim3(kHadamard),0,stream>>>(output,svh,rows,output_features);
}

// Bulk down projections already write the GEMM destination as FP16.
// Preserve its represented down value before performing the same FP32-add,
// FP16-store boundary as GDN's separate residual kernel. The final diagnostic
// rows have a separate owner because the normal output aliases down.
__global__ void output_hadamard_fp16_residual_kernel(
    std::uint16_t* output, const std::uint16_t* svh,
    const std::uint16_t* residual, std::uint16_t* down_trace,
    int trace_row_base, int rows, int output_features) {
    __shared__ float values[kHadamard];
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kHadamard) return;
    const int offset = block * kHadamard + lane;
    const auto index = static_cast<std::size_t>(row) * output_features + offset;
    values[lane] = __half2float(__ushort_as_half(output[index]));
    __syncthreads();
    for (int width = 1; width < kHadamard; width *= 2) {
        if ((lane % (2 * width)) < width) {
            const float left = values[lane];
            const float right = values[lane + width];
            values[lane] = left + right;
            values[lane + width] = left - right;
        }
        __syncthreads();
    }
    const auto normalized = __float2half_rn(values[lane] * kHadamardScale);
    const auto scale = __ushort_as_half(svh[offset]);
    const auto down_half = __hmul(normalized, scale);
    if (down_trace && row >= trace_row_base)
        down_trace[static_cast<std::size_t>(row - trace_row_base) *
                   output_features + offset] = __half_as_ushort(down_half);
    output[index] = __half_as_ushort(__float2half_rn(
        __half2float(__ushort_as_half(residual[index])) +
        __half2float(down_half)));
}

// Ascending split reduction followed by the unchanged output Hadamard.
// V6 donor reconstruction uses the destination dtype for the GEMM result as
// well as the Hadamard. Keep this separate from the historical T69 candidate.
template<bool FloatOutput>
__global__ void v6_output_hadamard_kernel(const float* accum,
    const std::uint16_t* svh, std::uint16_t* half_output, float* float_output,
    int output_features) {
    __shared__ float values[kHadamard];
    const int lane = threadIdx.x;
    const int offset = blockIdx.y * kHadamard + lane;
    const int index = blockIdx.x * output_features + offset;
    float value = accum[index];
    if constexpr (!FloatOutput) value = __half2float(__float2half_rn(value));
    values[lane] = value;
    __syncthreads();
    for (int width=1; width<kHadamard; width*=2) {
        if ((lane % (2*width)) < width) {
            const float a=values[lane], b=values[lane+width];
            values[lane]=a+b; values[lane+width]=a-b;
        }
        __syncthreads();
    }
    if constexpr (FloatOutput)
        float_output[index] = __fmul_rn(values[lane]*kHadamardScale,
            __half2float(__ushort_as_half(svh[offset])));
    else
        half_output[index] = __half_as_ushort(__hmul(
            __float2half_rn(values[lane]*kHadamardScale), __ushort_as_half(svh[offset])));
}

template<bool ShuffleLocal=false,bool MinimalBarriers=false,
         bool PrefetchSplitPlanes=false,bool Fp16GemmDestination=false>
__global__ void prefill_reduce_output_kernel(const float* accum,
                                       const std::uint16_t* svh,
                                       std::uint16_t* output,
                                       int rows,
                                       int output_features, int split_count) {
    __shared__ float values[kHadamard];
    const int row = static_cast<int>(blockIdx.x);
    const int block = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= rows || lane >= kHadamard) { return; }

    const int offset = block * kHadamard + lane;
    const std::size_t index = static_cast<std::size_t>(row) * output_features + offset;
    const std::size_t stride = static_cast<std::size_t>(rows) * output_features;
    float value = accum[index];
    if constexpr(PrefetchSplitPlanes) {
        if(split_count>1) {
            float next=accum[stride+index];
            for(int split=1;split<split_count;++split) {
                float following=0.0f;
                if(split+1<split_count)
                    following=accum[(split+1)*stride+index];
                value+=next;
                next=following;
            }
        }
    } else {
        for (int split = 1; split < split_count; ++split)
            value += accum[split * stride + index];
    }
    // The reached K5 bulk-MLP comparator writes its FP32-compute GEMM result
    // to FP16 before SVH/Hadamard. Retain that represented boundary when the
    // packed MMA leaf writes its partial into this FP32 scratch plane.
    if constexpr(Fp16GemmDestination)
        value=__half2float(__float2half_rn(value));
    float transformed_value;
    if constexpr(ShuffleLocal) {
        // Widths 1..16 are warp-local. Every lane computes the same output
        // formerly written by the pair's lower lane, with identical operands.
        for(int width=1;width<32;width*=2) {
            const float other=__shfl_xor_sync(0xffffffffu,value,width);
            value=(lane&width)?other-value:value+other;
        }
        values[lane]=value;
        __syncthreads();
        for(int width=32;width<kHadamard;width*=2) {
            const float other=values[lane^width];
            if constexpr(!MinimalBarriers)__syncthreads();
            else if(width==32)__syncthreads();
            value=(lane&width)?other-value:value+other;
            if(width<64) {values[lane]=value;__syncthreads();}
        }
        transformed_value=value;
    } else {
        values[lane] = value;
        __syncthreads();
        for (int width = 1; width < kHadamard; width *= 2) {
            if ((lane % (2 * width)) < width) {
                const int index = lane;
                const float left = values[index];
                const float right = values[index + width];
                values[index] = left + right;
                values[index + width] = left - right;
            }
            __syncthreads();
        }
        transformed_value=values[lane];
    }
    const auto normalized = __float2half_rn(transformed_value * kHadamardScale);
    const auto scale = __ushort_as_half(svh[offset]);
    output[row * output_features + offset] = __half_as_ushort(__hmul(normalized, scale));
}

// Warp-per-block twin of prefill_reduce_output_kernel: the split planes are
// summed in the same chronological order, the optional FP16 destination
// boundary is kept, and the transform is the exact butterfly network of the
// shared-memory kernel (see exl3_warp_butterflies).
template<bool Fp16GemmDestination>
__global__ void __launch_bounds__(kHadamardWarpsPerBlock*32) prefill_reduce_output_warp_kernel(
    const float* accum,const std::uint16_t* svh,std::uint16_t* output,int rows,
    int output_features,int split_count) {
    EXL3_PDL_PROLOGUE();
    const int lane=static_cast<int>(threadIdx.x)&31;
    const int blocks=output_features/kHadamard;
    const long long task=static_cast<long long>(blockIdx.x)*kHadamardWarpsPerBlock+
        (static_cast<int>(threadIdx.x)>>5);
    if(task>=static_cast<long long>(rows)*blocks) return;
    const int row=static_cast<int>(task/blocks);
    const int block=static_cast<int>(task%blocks);
    const int offset=block*kHadamard+lane*4;
    const std::size_t index=static_cast<std::size_t>(row)*output_features+offset;
    const std::size_t stride=static_cast<std::size_t>(rows)*output_features;
    const float4 first=*reinterpret_cast<const float4*>(accum+index);
    float v[4]={first.x,first.y,first.z,first.w};
    if(split_count<=10) {
        // All planes are loaded before the (unchanged, ascending) adds.
        float4 planes[9];
        #pragma unroll
        for(int split=1;split<10;++split)
            if(split<split_count)
                planes[split-1]=*reinterpret_cast<const float4*>(accum+split*stride+index);
        #pragma unroll
        for(int split=1;split<10;++split)
            if(split<split_count) {
                v[0]+=planes[split-1].x; v[1]+=planes[split-1].y;
                v[2]+=planes[split-1].z; v[3]+=planes[split-1].w;
            }
    } else
    for(int split=1;split<split_count;++split) {
        const float4 plane=*reinterpret_cast<const float4*>(accum+split*stride+index);
        v[0]+=plane.x; v[1]+=plane.y; v[2]+=plane.z; v[3]+=plane.w;
    }
    if constexpr(Fp16GemmDestination) {
        #pragma unroll
        for(int j=0;j<4;++j) v[j]=__half2float(__float2half_rn(v[j]));
    }
    std::uint16_t scale[4];
    exl3_load_half4(svh+offset,scale);
    exl3_warp_butterflies(v,lane);
    std::uint16_t result[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        const auto normalized=__float2half_rn(v[j]*kHadamardScale);
        result[j]=__half_as_ushort(__hmul(normalized,__ushort_as_half(scale[j])));
    }
    exl3_store_half4(output+row*output_features+offset,result);
}

template<bool ShuffleLocal=false,bool MinimalBarriers=false,
         bool PrefetchSplitPlanes=false,bool Fp16GemmDestination=false>
void launch_prefill_reduce_output(cudaStream_t stream,const float* accum,
    const std::uint16_t* svh,std::uint16_t* output,int rows,int output_features,
    int split_count) {
    if(rows>0 && exl3_hadamard_warp_enabled() && output_features%kHadamard==0 &&
       exl3_hadamard_warp_aligned(accum,svh,output))
        exl3_launch_pdl(prefill_reduce_output_warp_kernel<Fp16GemmDestination>,
            dim3(exl3_hadamard_warp_grid(rows,output_features)),dim3(kHadamardWarpsPerBlock*32),
            0,stream,accum,svh,output,rows,output_features,split_count);
    else
        prefill_reduce_output_kernel<ShuffleLocal,MinimalBarriers,PrefetchSplitPlanes,
            Fp16GemmDestination><<<dim3(rows,output_features/kHadamard),kHadamard,0,stream>>>(
                accum,svh,output,rows,output_features,split_count);
}

// Generic native EXL3 path used by E3A's K=6 and K=8 projection families.
// It deliberately keeps the packed trellis as the source of truth and decodes
// one 16x16 tile in shared memory at a time. This is a correctness-first CUDA
// path; the qualified K=5 shape-4 leaf remains the performance path for its
// original representative shape.
__device__ __forceinline__ std::uint32_t load_u32_generic(
    const std::uint16_t* packed, int index) {
    return static_cast<std::uint32_t>(packed[index * 2]) |
           (static_cast<std::uint32_t>(packed[index * 2 + 1]) << 16u);
}

__device__ __forceinline__ std::uint16_t decode_state_generic(
    const std::uint16_t* packed, int bits, int t_offset) {
    const int words32 = bits * 8;
    const int b0 = t_offset * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const std::uint64_t merged =
        (static_cast<std::uint64_t>(load_u32_generic(packed, (b0 / 32) % words32)) << 32u) |
        load_u32_generic(packed, ((b1 - 1) / 32) % words32);
    return static_cast<std::uint16_t>((merged >> shift) & 0xffffu);
}

__device__ __forceinline__ std::uint16_t decode_mul1_generic(
    std::uint16_t state, std::uint32_t multiplier) {
    const std::uint32_t product = static_cast<std::uint32_t>(state) * multiplier;
    const std::uint32_t byte_sum = (product & 0xffu) + ((product >> 8u) & 0xffu) +
                                   ((product >> 16u) & 0xffu) + (product >> 24u);
    const auto input = __ushort_as_half(static_cast<std::uint16_t>(0x6400u + byte_sum));
    const auto inverse = __ushort_as_half(static_cast<std::uint16_t>(0x1eeeu));
    const auto bias = __ushort_as_half(static_cast<std::uint16_t>(0xc931u));
    return __half_as_ushort(__hfma(input, inverse, bias));
}

// K7 uses the same exact unsigned-byte MUL1 sum as the scalar decoder, but
// unlike K5/K6/K8 its packed products may contain bytes with bit 7 set.  The
// signed-byte dp4a result is therefore corrected by 256 for every high-bit
// byte.  This preserves the shipped unsigned sum while avoiding four scalar
// byte additions and scalar half operations per decoded state.
__device__ __forceinline__ half2 decode_mul1_product_2_k7(
    std::uint32_t product0, std::uint32_t product1) {
    const int signed_sum0 = __dp4a(static_cast<int>(product0), 0x01010101, 0);
    const int signed_sum1 = __dp4a(static_cast<int>(product1), 0x01010101, 0);
    const int sum0 = signed_sum0 + (__popc(product0 & 0x80808080u) << 8);
    const int sum1 = signed_sum1 + (__popc(product1 & 0x80808080u) << 8);
    const auto input = __halves2half2(
        __ushort_as_half(static_cast<std::uint16_t>(kMul1AccumulatorHalf + sum0)),
        __ushort_as_half(static_cast<std::uint16_t>(kMul1AccumulatorHalf + sum1)));
    return __hfma2(input,
                   __half2half2(__ushort_as_half(kMul1InverseHalf)),
                   __half2half2(__ushort_as_half(kMul1BiasHalf)));
}

// Extract four adjacent packed states with the same circular 32-bit window
// used by the upstream dq4 path.  The previous generic implementation called
// decode_state_generic four times, including four 64-bit merges and modulo
// calculations.  Keeping Bits compile-time makes the shifts/constants fold
// for every real K5/K6/K7/K8 model family.
template <int Bits>
__device__ __forceinline__ void decode4_generic(const std::uint32_t* packed,
                                                 int t0,
                                                 std::uint32_t& w0,
                                                 std::uint32_t& w1,
                                                 std::uint32_t& w2,
                                                 std::uint32_t& w3) {
    constexpr int words32 = Bits * 8;
    const int b0 = (t0 + 257) * Bits - 16;
    const int b2 = b0 + 3 * Bits + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int shift = (i2 + 1) * 32 - b2;
    const std::uint32_t a = packed[i0 % words32];
    const std::uint32_t b = packed[i2 % words32];
    w3 = packed_fshift(b, a, shift) & 0xffffu;
    w2 = packed_fshift(b, a, shift + Bits) & 0xffffu;
    w1 = packed_fshift(b, a, shift + Bits * 2) & 0xffffu;
    w0 = packed_fshift(b, a, shift + Bits * 3) & 0xffffu;
}

__device__ __forceinline__ std::uint32_t decode_state_from_three_words_k7(
    std::uint32_t word0, std::uint32_t word1, std::uint32_t word2,
    int relative_bit) {
    const bool second_word = relative_bit >= 32;
    const int local_bit = relative_bit & 31;
    const std::uint32_t low = second_word ? word1 : word0;
    const std::uint32_t next = second_word ? word2 : word1;
    const bool crosses_word = local_bit + 16 > 32;
    const std::uint32_t high = crosses_word ? next : low;
    const int shift = (crosses_word ? 64 : 32) - (local_bit + 16);
    return packed_fshift(high, low, shift) & 0xffffu;
}

__device__ __forceinline__ void decode4_three_word_k7(
    const std::uint32_t* packed, int t0,
    std::uint32_t& w0, std::uint32_t& w1,
    std::uint32_t& w2, std::uint32_t& w3) {
    constexpr int words32 = 7 * 8;
    const int first_bit = (t0 + 257) * 7 - 16;
    int first_word = first_bit >> 5;
    if (first_word >= words32) first_word -= words32;
    const int second_word = first_word == words32 - 1 ? 0 : first_word + 1;
    const int third_word = second_word == words32 - 1 ? 0 : second_word + 1;
    const std::uint32_t a = packed[first_word];
    const std::uint32_t b = packed[second_word];
    const std::uint32_t c = packed[third_word];
    const int relative_bit = first_bit & 31;
    w0 = decode_state_from_three_words_k7(a, b, c, relative_bit);
    w1 = decode_state_from_three_words_k7(a, b, c, relative_bit + 7);
    w2 = decode_state_from_three_words_k7(a, b, c, relative_bit + 14);
    w3 = decode_state_from_three_words_k7(a, b, c, relative_bit + 21);
}

// Branch-free K7 four-state window for any t_offset in [0,256). States 0..2
// always lie inside the first two represented words; only state 3 can reach
// the third word (relative bit > 27), and it is selected rather than taken
// through a divergent per-lane path. Extracted states, and hence decoded
// halves, are bitwise identical to decode4_generic/decode4_three_word_k7.
__device__ __forceinline__ void exl3_dq4_k7_window(
    const std::uint32_t* packed, int t_offset, Exl3FragB& fragment,
    std::uint32_t mul1_multiplier) {
    const int b0 = t_offset * 7 + 1783;
    int i0 = b0 >> 5;
    if (i0 >= 56) i0 -= 56;
    const int i1 = i0 == 55 ? 0 : i0 + 1;
    const int i2 = i1 == 55 ? 0 : i1 + 1;
    const int rel = b0 & 31;
    const std::uint32_t a = packed[i0], b = packed[i1], c = packed[i2];
    const std::uint64_t ab = (static_cast<std::uint64_t>(a) << 32) | b;
    const std::uint64_t bc = (static_cast<std::uint64_t>(b) << 32) | c;
    const std::uint32_t w0 = static_cast<std::uint32_t>(ab >> (48 - rel)) & 0xffffu;
    const std::uint32_t w1 = static_cast<std::uint32_t>(ab >> (41 - rel)) & 0xffffu;
    const std::uint32_t w2 = static_cast<std::uint32_t>(ab >> (34 - rel)) & 0xffffu;
    const std::uint32_t w3 = (rel <= 27
        ? static_cast<std::uint32_t>(ab >> (27 - rel))
        : static_cast<std::uint32_t>(bc >> (59 - rel))) & 0xffffu;
    fragment.values[0] = decode_mul1_product_2_k7(w0 * mul1_multiplier,
                                                   w1 * mul1_multiplier);
    fragment.values[1] = decode_mul1_product_2_k7(w2 * mul1_multiplier,
                                                   w3 * mul1_multiplier);
}

template <int Bits, bool K7ThreeWord = false>
__device__ __forceinline__ void exl3_dq4_generic(const std::uint32_t* packed,
                                                  int t_offset,
                                                  Exl3FragB& fragment,
                                                  std::uint32_t mul1_multiplier) {
#ifndef NINFER_EXL3_K7_DIVERGENT_DECODE
    if constexpr (Bits == 7 && K7ThreeWord) {
        exl3_dq4_k7_window(packed, t_offset, fragment, mul1_multiplier);
        return;
    }
#endif
    const int b0 = (t_offset + 257) * Bits - 16;
    if ((b0 & 31) + 3 * Bits + 16 > 64) {
        // A four-state window can cross three 32-bit words for some K7/K8
        // lane groups.  The compact two-word form is not valid there; use
        // the already-qualified scalar extractor for only those boundary
        // groups instead of admitting an incorrect packed fast path.
        if constexpr (Bits == 7 && K7ThreeWord) {
            std::uint32_t w0, w1, w2, w3;
            decode4_three_word_k7(packed, t_offset, w0, w1, w2, w3);
            fragment.values[0] = decode_mul1_product_2_k7(
                w0 * mul1_multiplier, w1 * mul1_multiplier);
            fragment.values[1] = decode_mul1_product_2_k7(
                w2 * mul1_multiplier, w3 * mul1_multiplier);
        } else {
            const auto* packed16 = reinterpret_cast<const std::uint16_t*>(packed);
            const std::uint16_t d0 = decode_mul1_generic(decode_state_generic(packed16, Bits, t_offset), mul1_multiplier);
            const std::uint16_t d1 = decode_mul1_generic(decode_state_generic(packed16, Bits, t_offset + 1), mul1_multiplier);
            const std::uint16_t d2 = decode_mul1_generic(decode_state_generic(packed16, Bits, t_offset + 2), mul1_multiplier);
            const std::uint16_t d3 = decode_mul1_generic(decode_state_generic(packed16, Bits, t_offset + 3), mul1_multiplier);
            fragment.values[0] = __halves2half2(__ushort_as_half(d0), __ushort_as_half(d1));
            fragment.values[1] = __halves2half2(__ushort_as_half(d2), __ushort_as_half(d3));
        }
        return;
    }
    std::uint32_t w0, w1, w2, w3;
    decode4_generic<Bits>(packed, t_offset, w0, w1, w2, w3);
    if constexpr (Bits == 4) {
        const std::uint16_t d0 = decode_mul1_generic(
            static_cast<std::uint16_t>(w0), mul1_multiplier);
        const std::uint16_t d1 = decode_mul1_generic(
            static_cast<std::uint16_t>(w1), mul1_multiplier);
        const std::uint16_t d2 = decode_mul1_generic(
            static_cast<std::uint16_t>(w2), mul1_multiplier);
        const std::uint16_t d3 = decode_mul1_generic(
            static_cast<std::uint16_t>(w3), mul1_multiplier);
        fragment.values[0] = __halves2half2(__ushort_as_half(d0), __ushort_as_half(d1));
        fragment.values[1] = __halves2half2(__ushort_as_half(d2), __ushort_as_half(d3));
        return;
    }
    if constexpr (Bits == 7) {
        fragment.values[0] = decode_mul1_product_2_k7(w0 * mul1_multiplier,
                                                       w1 * mul1_multiplier);
        fragment.values[1] = decode_mul1_product_2_k7(w2 * mul1_multiplier,
                                                       w3 * mul1_multiplier);
        return;
    }
    fragment.values[0] = decode_mul1_product_2(w0 * mul1_multiplier,
                                                w1 * mul1_multiplier);
    fragment.values[1] = decode_mul1_product_2(w2 * mul1_multiplier,
                                                w3 * mul1_multiplier);
}

// Every K6 MMA caller requests t_offset=(lane*8)+{0,4}. Those 64 windows are
// exactly two represented 32-bit words wide: the generic three-word fallback
// is unreachable, and modulo 48 reduces to one wrap. Keep the extracted state
// words and MUL1 half arithmetic identical while removing the integer div/mod
// work from the dominant N64 prefill loop.
__device__ __forceinline__ void exl3_dq4_k6_lane_window(
    const std::uint32_t* packed,int t_offset,Exl3FragB& fragment,
    std::uint32_t mul1_multiplier) {
    const int b0=t_offset*6+1526;
    int i0=b0>>5;
    if(i0>=48)i0-=48;
    const int i1=i0==47?0:i0+1;
    const int shift=30-(b0&31);
    const std::uint32_t a=packed[i0],b=packed[i1];
    const std::uint32_t w3=packed_fshift(b,a,shift)&0xffffu;
    const std::uint32_t w2=packed_fshift(b,a,shift+6)&0xffffu;
    const std::uint32_t w1=packed_fshift(b,a,shift+12)&0xffffu;
    const std::uint32_t w0=packed_fshift(b,a,shift+18)&0xffffu;
    fragment.values[0]=decode_mul1_product_2(
        w0*mul1_multiplier,w1*mul1_multiplier);
    fragment.values[1]=decode_mul1_product_2(
        w2*mul1_multiplier,w3*mul1_multiplier);
}

__device__ __forceinline__ int inverse_tensor_core_index(int row, int column) {
    const int row_group = row >= 8 ? row - 8 : row;
    const int t_mod = row_group / 2;
    const int variant = (row >= 8 ? 2 : 0) + (row & 1);
    const int c_group = column >= 8 ? column - 8 : column;
    const int slot = variant + (column >= 8 ? 4 : 0);
    return (c_group * 4 + t_mod) * 8 + slot;
}

// MXFP8 (OCP MX: E4M3 elements, one E8M0 power-of-two scale per 32 elements
// along K) operands for the cuBLASLt VEC32_UE8M0 prefill route. Scale factors
// use the cuBLASLt 128x4 tile layout; `inner_blocks` is K/32 rounded up to 4.
__device__ __forceinline__ std::size_t mxfp8_scale_offset(int outer,int inner,
                                                          int inner_blocks) {
    return (static_cast<std::size_t>(outer/128)*inner_blocks+(inner/4)*4)*128+
        (outer%32)*16+((outer%128)/32)*4+(inner%4);
}

// Power-of-two scale so the block maximum maps at or below the E4M3 finite
// maximum (448); a zero block keeps the minimum scale and encodes zeros.
__device__ __forceinline__ int mxfp8_block_exponent(float amax) {
    if (!(amax>0.0f)) return -127;
    int power=0;
    const float mantissa=frexpf(amax*(1.0f/448.0f),&power);
    const int exponent=mantissa==0.5f?power-1:power;
    return exponent<-127?-127:(exponent>127?127:exponent);
}

__device__ __forceinline__ void mxfp8_encode32(const float (&values)[32],float scale,
                                               std::uint8_t* destination) {
    alignas(16) std::uint8_t bytes[32];
#pragma unroll
    for (int i=0;i<32;i+=2) {
        const __nv_fp8x2_storage_t pair=__nv_cvt_float2_to_fp8x2(
            make_float2(values[i]*scale,values[i+1]*scale),__NV_SATFINITE,__NV_E4M3);
        bytes[i]=static_cast<std::uint8_t>(pair&0xff);
        bytes[i+1]=static_cast<std::uint8_t>(pair>>8);
    }
    reinterpret_cast<uint4*>(destination)[0]=reinterpret_cast<const uint4*>(bytes)[0];
    reinterpret_cast<uint4*>(destination)[1]=reinterpret_cast<const uint4*>(bytes)[1];
}

// K-contiguous FP16 activations [rows][k] -> E4M3 [rows][k] plus scales for
// rows padded to 128 (padding scales are zero-filled as cuBLASLt requires).
__global__ void mxfp8_quantize_rows_kernel(const half* __restrict__ source,
    std::uint8_t* __restrict__ values,std::uint8_t* __restrict__ scales,
    int rows,int k,int padded_rows) {
    const int blocks=k/32;
    const int inner_blocks=(blocks+3)/4*4;
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=padded_rows*blocks) return;
    const int row=index/blocks,block=index%blocks;
    if (row>=rows) {
        scales[mxfp8_scale_offset(row,block,inner_blocks)]=0;
        return;
    }
    const auto* input=reinterpret_cast<const uint4*>(
        source+static_cast<std::size_t>(row)*k+block*32);
    float x[32];
    float amax=0.0f;
#pragma unroll
    for (int part=0;part<4;++part) {
        const uint4 packed=input[part];
        const half2* pairs=reinterpret_cast<const half2*>(&packed);
#pragma unroll
        for (int i=0;i<4;++i) {
            const float2 value=__half22float2(pairs[i]);
            x[part*8+i*2]=value.x;x[part*8+i*2+1]=value.y;
            amax=fmaxf(amax,fmaxf(fabsf(value.x),fabsf(value.y)));
        }
    }
    const int exponent=mxfp8_block_exponent(amax);
    mxfp8_encode32(x,exp2f(static_cast<float>(-exponent)),
        values+static_cast<std::size_t>(row)*k+block*32);
    scales[mxfp8_scale_offset(row,block,inner_blocks)]=
        static_cast<std::uint8_t>(exponent+127);
}

// Reconstructed FP16 weight [k][n] (N contiguous) -> K-major E4M3 [n][k] plus
// scales (outer n, inner k/32). One thread owns one output column's 32-row
// K block; grid (ceil(n/256), k/32). n and k are multiples of 128.
__global__ void mxfp8_quantize_weight_transposed_kernel(const half* __restrict__ source,
    std::uint8_t* __restrict__ values,std::uint8_t* __restrict__ scales,int k,int n) {
    const int column=blockIdx.x*blockDim.x+threadIdx.x;
    const int block=blockIdx.y;
    if (column>=n) return;
    float x[32];
    float amax=0.0f;
#pragma unroll
    for (int i=0;i<32;++i) {
        x[i]=__half2float(source[static_cast<std::size_t>(block*32+i)*n+column]);
        amax=fmaxf(amax,fabsf(x[i]));
    }
    const int exponent=mxfp8_block_exponent(amax);
    mxfp8_encode32(x,exp2f(static_cast<float>(-exponent)),
        values+static_cast<std::size_t>(column)*k+block*32);
    scales[mxfp8_scale_offset(column,block,(k/32+3)/4*4)]=
        static_cast<std::uint8_t>(exponent+127);
}

// NVFP4 operands for the cuBLASLt VEC16_UE4M3 prefill route: E2M1 elements
// (two per byte, lower K index in the low nibble), one unsigned E4M3 scale per
// 16 elements along K in the 128x4 tile layout (inner = K/16 rounded up to 4),
// and one FP32 global scale per tensor so block scales stay in E4M3 range.
// Dequantized value = e2m1 * e4m3_scale * global.
__device__ __forceinline__ float nvfp4_global_from_amax(float amax) {
    return amax > 0.0f ? amax / (6.0f * 448.0f) : 1.0f;
}

__device__ __forceinline__ float nvfp4_decode_e4m3(__nv_fp8_storage_t code) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
}

// Smallest positive E4M3 code whose value is >= target, saturating at 448.
__device__ __forceinline__ std::uint8_t nvfp4_scale_code(float target, float& value) {
    __nv_fp8_storage_t code = __nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3);
    float decoded = nvfp4_decode_e4m3(code);
    if (decoded < target && code < 0x7e) decoded = nvfp4_decode_e4m3(++code);
    if (!(decoded > 0.0f)) decoded = nvfp4_decode_e4m3(code = 0x01);
    value = decoded;
    return static_cast<std::uint8_t>(code);
}

__device__ __forceinline__ void nvfp4_encode16(const float (&x)[16], float amax,
    float global, std::uint8_t* destination, std::uint8_t& scale_code) {
    float scale = 1.0f;
    scale_code = nvfp4_scale_code(fmaxf(amax / (6.0f * global), 1e-30f), scale);
    const float inverse = 1.0f / (scale * global);
    alignas(8) std::uint8_t bytes[8];
#pragma unroll
    for (int i = 0; i < 16; i += 2)
        bytes[i / 2] = static_cast<std::uint8_t>(__nv_cvt_float2_to_fp4x2(
            make_float2(x[i] * inverse, x[i + 1] * inverse), __NV_E2M1, cudaRoundNearest));
    *reinterpret_cast<uint2*>(destination) = *reinterpret_cast<const uint2*>(bytes);
}

// Tensor absolute maximum; non-negative floats order as unsigned bit patterns.
__global__ void nvfp4_amax_kernel(const half* __restrict__ source, std::size_t count,
                                  unsigned* __restrict__ amax_bits) {
    float local = 0.0f;
    for (std::size_t i = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) * 8;
         i < count; i += static_cast<std::size_t>(gridDim.x) * blockDim.x * 8) {
        const uint4 packed = *reinterpret_cast<const uint4*>(source + i);
        const half2* pairs = reinterpret_cast<const half2*>(&packed);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 value = __half22float2(pairs[j]);
            local = fmaxf(local, fmaxf(fabsf(value.x), fabsf(value.y)));
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, offset));
    if ((threadIdx.x & 31) == 0) atomicMax(amax_bits, __float_as_uint(local));
}

// K-contiguous FP16 activations [rows][k] -> packed E2M1 [rows][k/2] plus block
// scales (rows padded to 128, padding zero-filled). Publishes the GEMM
// alpha = activation global * weight global.
__global__ void nvfp4_quantize_rows_kernel(const half* __restrict__ source,
    std::uint8_t* __restrict__ values, std::uint8_t* __restrict__ scales,
    int rows, int k, int padded_rows, const unsigned* __restrict__ amax_bits,
    const float* __restrict__ weight_global, float* __restrict__ alpha) {
    const float global = nvfp4_global_from_amax(__uint_as_float(*amax_bits));
    const int blocks = k / 16;
    const int inner_blocks = (blocks + 3) / 4 * 4;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0) *alpha = global * *weight_global;
    if (index >= padded_rows * blocks) return;
    const int row = index / blocks, block = index % blocks;
    if (row >= rows) {
        scales[mxfp8_scale_offset(row, block, inner_blocks)] = 0;
        return;
    }
    const auto* input = reinterpret_cast<const uint4*>(
        source + static_cast<std::size_t>(row) * k + block * 16);
    float x[16];
    float amax = 0.0f;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        const uint4 packed = input[part];
        const half2* pairs = reinterpret_cast<const half2*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 value = __half22float2(pairs[i]);
            x[part * 8 + i * 2] = value.x;
            x[part * 8 + i * 2 + 1] = value.y;
            amax = fmaxf(amax, fmaxf(fabsf(value.x), fabsf(value.y)));
        }
    }
    std::uint8_t code = 0;
    nvfp4_encode16(x, amax, global,
        values + (static_cast<std::size_t>(row) * k + block * 16) / 2, code);
    scales[mxfp8_scale_offset(row, block, inner_blocks)] = code;
}

// Reconstructed FP16 weight [k][n] -> K-major packed E2M1 [n][k/2] plus block
// scales (outer n, inner k/16) and the weight global scale. Grid (ceil(n/256), k/16).
__global__ void nvfp4_quantize_weight_transposed_kernel(const half* __restrict__ source,
    std::uint8_t* __restrict__ values, std::uint8_t* __restrict__ scales,
    int k, int n, const unsigned* __restrict__ amax_bits, float* __restrict__ weight_global) {
    const float global = nvfp4_global_from_amax(__uint_as_float(*amax_bits));
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int block = blockIdx.y;
    if (column == 0 && block == 0) *weight_global = global;
    if (column >= n) return;
    float x[16];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        x[i] = __half2float(source[static_cast<std::size_t>(block * 16 + i) * n + column]);
        amax = fmaxf(amax, fabsf(x[i]));
    }
    std::uint8_t code = 0;
    nvfp4_encode16(x, amax, global,
        values + (static_cast<std::size_t>(column) * k + block * 16) / 2, code);
    scales[mxfp8_scale_offset(column, block, (k / 16 + 3) / 4 * 4)] = code;
}

template <int Bits, bool FragmentOrder = false>
__global__ void exl3_reconstruct_transformed_weight_kernel(
    const std::uint16_t* trellis,
    const std::int32_t* mul1,
    std::uint16_t* reconstructed,
    int input_features,
    int output_features,
    int first_output_tile = 0,
    int decoded_output_tiles = 0) {
    __shared__ std::uint16_t packed[16 * Bits];
    const int lane = static_cast<int>(threadIdx.x);
    const int tile_k = static_cast<int>(blockIdx.y);
    const int local_tile_n = static_cast<int>(blockIdx.x);
    const int tile_n = first_output_tile + local_tile_n;
    const int decoded_stride = decoded_output_tiles ? decoded_output_tiles : output_features / 16;
    const auto* source = trellis +
        (static_cast<std::size_t>(tile_k) * (output_features / 16) + tile_n) *
            (16 * Bits);
    // Fragment-order reconstruction is already an explicit opt-in route. Load
    // complete represented groups only; misaligned source views retain scalar
    // loads. No decoded column or padding beyond this native tile is touched.
    if constexpr (FragmentOrder) {
        constexpr int words=16*Bits;
        if((reinterpret_cast<std::uintptr_t>(source)&7u)==0) {
            if(lane<words/4) {
                const uint2 value=reinterpret_cast<const uint2*>(source)[lane];
                packed[lane*4]=static_cast<std::uint16_t>(value.x);
                packed[lane*4+1]=static_cast<std::uint16_t>(value.x>>16);
                packed[lane*4+2]=static_cast<std::uint16_t>(value.y);
                packed[lane*4+3]=static_cast<std::uint16_t>(value.y>>16);
            }
            if(lane>=words/4*4 && lane<words)packed[lane]=source[lane];
        } else if(lane<words)packed[lane]=source[lane];
    } else if (lane < 16 * Bits) packed[lane] = source[lane];
    __syncthreads();
    const int row = lane / 16;
    const int column = lane % 16;
    const int encoded = inverse_tensor_core_index(row, column);
    const auto state = decode_state_generic(packed, Bits, encoded);
    const auto value = decode_mul1_generic(
        state, static_cast<std::uint32_t>(*mul1));
    if constexpr (FragmentOrder)
        reconstructed[(static_cast<std::size_t>(tile_k) * decoded_stride +
                       local_tile_n) * 256 + encoded] = value;
    else
        reconstructed[(static_cast<std::size_t>(tile_k) * 16 + row) *
                          output_features + tile_n * 16 + column] = value;
}

// EXL3 MUL1 codebook values lie in [-3.454, 3.447] for every state, so the
// NVFP4 weight global scale is a constant bound instead of a measured amax.
constexpr float kNvfp4WeightGlobal = 3.5f / (6.0f * 448.0f);

// Fused EXL3 decode + block quantization of one transformed weight into the
// K-major MXFP8 (Quant=32) or NVFP4 (Quant=16) prefill operand, bypassing the
// FP16 reconstruction plane. Each thread decodes the Quant consecutive K
// values of one output column (one quantization block) from the packed
// trellis tiles staged in shared memory. Grid (ceil(n/256), k/Quant).
template <int Bits, int Quant>
__global__ void __launch_bounds__(256) exl3_decode_quantize_weight_kernel(
    const std::uint16_t* __restrict__ trellis, const std::int32_t* __restrict__ mul1,
    std::uint8_t* __restrict__ values, std::uint8_t* __restrict__ scales,
    float* __restrict__ weight_global, int k, int n) {
    constexpr int kTilesK = Quant / 16;
    constexpr int kWords = 16 * Bits;          // packed u16 words per 16x16 tile
    __shared__ std::uint16_t packed[kTilesK][16][kWords];
    const int tid = static_cast<int>(threadIdx.x);
    const int tiles_n = n / 16;
    const int tile_n0 = static_cast<int>(blockIdx.x) * 16;
    const int live_tiles = min(16, tiles_n - tile_n0);
    const int k_tile0 = static_cast<int>(blockIdx.y) * kTilesK;
    for (int kt = 0; kt < kTilesK; ++kt) {
        const std::uint16_t* source = trellis +
            (static_cast<std::size_t>(k_tile0 + kt) * tiles_n + tile_n0) * kWords;
        for (int i = tid; i < live_tiles * kWords; i += 256)
            packed[kt][i / kWords][i % kWords] = source[i];
    }
    if (Quant == 16 && tid == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        *weight_global = kNvfp4WeightGlobal;
    __syncthreads();
    const int local_tile = tid / 16, column_in_tile = tid % 16;
    if (local_tile >= live_tiles) return;
    const int column = (tile_n0 + local_tile) * 16 + column_in_tile;
    const std::uint32_t multiplier = static_cast<std::uint32_t>(*mul1);
    float x[Quant];
    float amax = 0.0f;
#pragma unroll
    for (int kt = 0; kt < kTilesK; ++kt) {
#pragma unroll
        for (int row = 0; row < 16; ++row) {
            const int encoded = inverse_tensor_core_index(row, column_in_tile);
            const float value = __half2float(__ushort_as_half(decode_mul1_generic(
                decode_state_generic(packed[kt][local_tile], Bits, encoded), multiplier)));
            x[kt * 16 + row] = value;
            amax = fmaxf(amax, fabsf(value));
        }
    }
    const int k0 = static_cast<int>(blockIdx.y) * Quant;
    if constexpr (Quant == 32) {
        const int exponent = mxfp8_block_exponent(amax);
        mxfp8_encode32(x, exp2f(static_cast<float>(-exponent)),
            values + static_cast<std::size_t>(column) * k + k0);
        scales[mxfp8_scale_offset(column, blockIdx.y, (k / 32 + 3) / 4 * 4)] =
            static_cast<std::uint8_t>(exponent + 127);
    } else {
        std::uint8_t code = 0;
        nvfp4_encode16(x, amax, kNvfp4WeightGlobal,
            values + (static_cast<std::size_t>(column) * k + k0) / 2, code);
        scales[mxfp8_scale_offset(column, blockIdx.y, (k / 16 + 3) / 4 * 4)] = code;
    }
}

template <int Quant>
void launch_decode_quantize_weight(int bits, const std::uint16_t* trellis,
    const std::int32_t* mul1, std::uint8_t* values, std::uint8_t* scales,
    float* weight_global, int k, int n, cudaStream_t stream) {
    const dim3 grid((n / 16 + 15) / 16, k / Quant);
    switch (bits) {
    case 5: exl3_decode_quantize_weight_kernel<5, Quant><<<grid, 256, 0, stream>>>(
        trellis, mul1, values, scales, weight_global, k, n); break;
    case 6: exl3_decode_quantize_weight_kernel<6, Quant><<<grid, 256, 0, stream>>>(
        trellis, mul1, values, scales, weight_global, k, n); break;
    case 7: exl3_decode_quantize_weight_kernel<7, Quant><<<grid, 256, 0, stream>>>(
        trellis, mul1, values, scales, weight_global, k, n); break;
    default: throw std::invalid_argument("fused EXL3 decode/quantize supports K5..K7");
    }
}

// Fold both EXL3 128-point transforms into a row-major reconstructed weight.
// The input/output Hadamard launches are intentionally absent from the caller
// when this candidate is enabled: the resulting matrix is consumed directly
// by a FP32-compute FP16 GEMM against the original activation. The established
// transformed-basis route remains the fail-closed default and this candidate
// is only admitted for wide rows.
__global__ __launch_bounds__(kThreads, 1)
void exl3_fused_original_weight_hadamard_kernel(
    std::uint16_t* weight,
    const std::uint16_t* suh,
    const std::uint16_t* svh,
    int input_features,
    int output_features) {
    constexpr int tile_elements = kHadamard * kHadamard;
    __shared__ float tile[tile_elements];
    const int thread = static_cast<int>(threadIdx.x);
    const int tile_row = static_cast<int>(blockIdx.y) * kHadamard;
    const int tile_column = static_cast<int>(blockIdx.x) * kHadamard;
    const int row_stride = output_features;

    for (int index = thread; index < tile_elements; index += kThreads) {
        const int row = index / kHadamard;
        const int column = index % kHadamard;
        tile[index] = __half2float(__ushort_as_half(
            weight[(tile_row + row) * row_stride + tile_column + column]));
    }
    __syncthreads();

    // Left multiply by H128 in the same ascending Sylvester order used by the
    // native activation transform. The tile stays FP32 until the final store.
    for (int width = 1; width < kHadamard; width *= 2) {
        for (int index = thread; index < tile_elements; index += kThreads) {
            const int row = index / kHadamard;
            const int column = index % kHadamard;
            if ((row % (2 * width)) < width) {
                const float left = tile[index];
                const float right = tile[index + width * kHadamard];
                tile[index] = left + right;
                tile[index + width * kHadamard] = left - right;
            }
        }
        __syncthreads();
    }

    // Right multiply by H128.
    for (int width = 1; width < kHadamard; width *= 2) {
        for (int index = thread; index < tile_elements; index += kThreads) {
            const int column = index % kHadamard;
            if ((column % (2 * width)) < width) {
                const float left = tile[index];
                const float right = tile[index + width];
                tile[index] = left + right;
                tile[index + width] = left - right;
            }
        }
        __syncthreads();
    }

    constexpr float scale = kHadamardScale * kHadamardScale;
    for (int index = thread; index < tile_elements; index += kThreads) {
        const int row = index / kHadamard;
        const int column = index % kHadamard;
        const float input_sign = __half2float(__ushort_as_half(
            suh[tile_row + row]));
        const float output_sign = __half2float(__ushort_as_half(
            svh[tile_column + column]));
        weight[(tile_row + row) * row_stride + tile_column + column] =
            __half_as_ushort(__float2half_rn(
                tile[index] * scale * input_sign * output_sign));
    }
}

// Native fused EXL3 dequantization plus the two 128-point transforms.  The
// packed tile is decoded directly into the transform staging tile, avoiding
// the full transformed-weight write/read pair used by the bring-up kernel
// above.  The codebook/dequant and half Hadamard helpers are part of the
// repository's native EXL3 CUDA support; this launch is independently gated
// and never calls the Mia comparator or its extension.
template <int Bits>
__global__ __launch_bounds__(kThreads)
void exl3_fused_original_weight_reconstruct_kernel(
    std::uint16_t* output,
    const std::uint16_t* trellis,
    const std::uint16_t* suh,
    const std::uint16_t* svh,
    int packed_blocks_n) {
    constexpr int packed_size = 256 * Bits / 16;
    constexpr float r_scale = 0.08838834764831845f;
    const int thread = static_cast<int>(threadIdx.x);
    const int lane = thread & 31;
    const int warp = thread / 32;
    const int input_block = static_cast<int>(blockIdx.y);
    const int output_block = static_cast<int>(blockIdx.x);
    const int row_len = gridDim.x * kHadamard;

    __shared__ std::uint32_t packed[8][8][packed_size / 2];
    __shared__ half2 staging[kHadamard * 64];

    auto staging_index = [](int row, int q, int part) {
        return row * 64 + (q ^ ((row >> 2) & 31)) * 2 + part;
    };

    constexpr int packed_int4_per_tile = packed_size / 8;
    for (int index = thread; index < 8 * 8 * packed_int4_per_tile;
         index += kThreads) {
        const int input_tile = index / (8 * packed_int4_per_tile);
        const int offset = index % (8 * packed_int4_per_tile);
        const std::uint16_t* source = trellis +
            (static_cast<std::size_t>(input_block * 8 + input_tile) *
                 packed_blocks_n + output_block * 8) * packed_size;
        reinterpret_cast<int4*>(packed[input_tile])[offset] =
            reinterpret_cast<const int4*>(source)[offset];
    }
    __syncthreads();

    for (int jj = 0; jj < 8; ++jj) {
        const int input_tile = jj;
        const int output_tile = warp & 7;
        FragB fragments[2];
        dq_dispatch<Bits, 2>(packed[input_tile][output_tile], lane * 8,
                             fragments[0], fragments[1]);

        const half2 n0 = __shfl_down_sync(0xffffffffu, fragments[0][0], 4);
        const half2 n1 = __shfl_down_sync(0xffffffffu, fragments[0][1], 4);
        const half2 n2 = __shfl_down_sync(0xffffffffu, fragments[1][0], 4);
        const half2 n3 = __shfl_down_sync(0xffffffffu, fragments[1][1], 4);
        if (!(lane & 4)) {
            const half2 m0 = __halves2half2(
                __low2half(fragments[0][0]), __low2half(n0));
            const half2 m1 = __halves2half2(
                __high2half(fragments[0][0]), __high2half(n0));
            const half2 m2 = __halves2half2(
                __low2half(fragments[0][1]), __low2half(n1));
            const half2 m3 = __halves2half2(
                __high2half(fragments[0][1]), __high2half(n1));
            const half2 m4 = __halves2half2(
                __low2half(fragments[1][0]), __low2half(n2));
            const half2 m5 = __halves2half2(
                __high2half(fragments[1][0]), __high2half(n2));
            const half2 m6 = __halves2half2(
                __low2half(fragments[1][1]), __low2half(n3));
            const half2 m7 = __halves2half2(
                __high2half(fragments[1][1]), __high2half(n3));
            const int row0 = jj * 16 + (lane & 3) * 2;
            const int row1 = row0 + 1;
            const int row2 = row0 + 8;
            const int row3 = row0 + 9;
            const int column0 = lane / 8;
            const int q0 = (output_tile * 8 + column0) >> 1;
            const int p0 = column0 & 1;
            const int q1 = (output_tile * 8 + column0 + 4) >> 1;
            const int p1 = (column0 + 4) & 1;
            staging[staging_index(row0, q0, p0)] = m0;
            staging[staging_index(row1, q0, p0)] = m1;
            staging[staging_index(row2, q0, p0)] = m2;
            staging[staging_index(row3, q0, p0)] = m3;
            staging[staging_index(row0, q1, p1)] = m4;
            staging[staging_index(row1, q1, p1)] = m5;
            staging[staging_index(row2, q1, p1)] = m6;
            staging[staging_index(row3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 scale = __float2half2_rn(r_scale);
    constexpr int chunks_per_warp = 32 / (kThreads / 32);
    #pragma unroll
    for (int chunk = 0; chunk < chunks_per_warp; ++chunk) {
        const int q = warp * chunks_per_warp + chunk;
        const int shuffled_q = q ^ lane;
        half2 first[4], second[4];
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const half2* source = staging +
                (lane * 4 + part) * 64 + shuffled_q * 2;
            first[part] = source[0];
            second[part] = source[1];
        }
        #pragma unroll
        for (int side = 0; side < 2; ++side) {
            half2* values = side == 0 ? first : second;
            half2 sum0 = __hadd2(values[0], values[1]);
            half2 difference0 = __hsub2(values[0], values[1]);
            half2 sum1 = __hadd2(values[2], values[3]);
            half2 difference1 = __hsub2(values[2], values[3]);
            values[0] = __hmul2(__hadd2(sum0, sum1), scale);
            values[1] = __hmul2(__hadd2(difference0, difference1), scale);
            values[2] = __hmul2(__hsub2(sum0, sum1), scale);
            values[3] = __hmul2(__hsub2(difference0, difference1), scale);
            #pragma unroll
            for (int part = 0; part < 4; ++part)
                values[part] = shuffle_had_h2x32(values[part], lane);
        }
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            half2* destination = staging +
                (lane * 4 + part) * 64 + shuffled_q * 2;
            destination[0] = first[part];
            destination[1] = second[part];
        }
    }
    __syncthreads();

    constexpr int rows_per_warp = kHadamard / (kThreads / 32);
    const int output_base = output_block * kHadamard;
    #pragma unroll
    for (int offset = 0; offset < rows_per_warp; ++offset) {
        const int row = warp * rows_per_warp + offset;
        const int base = row * 64 + (lane ^ ((row >> 2) & 31)) * 2;
        const half2 value01 = staging[base];
        const half2 value23 = staging[base + 1];
        const float v0 = __low2float(value01);
        const float v1 = __high2float(value01);
        const float v2 = __low2float(value23);
        const float v3 = __high2float(value23);
        const float sum0 = v0 + v1;
        const float difference0 = v0 - v1;
        const float sum1 = v2 + v3;
        const float difference1 = v2 - v3;
        half2 result01 = __hmul2(__floats2half2_rn(
            sum0 + sum1, difference0 + difference1), scale);
        half2 result23 = __hmul2(__floats2half2_rn(
            sum0 - sum1, difference0 - difference1), scale);
        result01 = shuffle_had_h2x32(result01, lane);
        result23 = shuffle_had_h2x32(result23, lane);
        const half2 input_sign = __half2half2(__ushort_as_half(
            suh[input_block * kHadamard + row]));
        const half2 output_sign01 = __halves2half2(
            __ushort_as_half(svh[output_base + lane * 4]),
            __ushort_as_half(svh[output_base + lane * 4 + 1]));
        const half2 output_sign23 = __halves2half2(
            __ushort_as_half(svh[output_base + lane * 4 + 2]),
            __ushort_as_half(svh[output_base + lane * 4 + 3]));
        result01 = __hmul2(__hmul2(result01, input_sign), output_sign01);
        result23 = __hmul2(__hmul2(result23, input_sign), output_sign23);
        half2* destination = reinterpret_cast<half2*>(output) +
            static_cast<std::size_t>(input_block * kHadamard + row) *
                (row_len / 2) + output_base / 2 + lane * 2;
        destination[0] = result01;
        destination[1] = result23;
    }
}

// Native persistent M=1 GEMM adapter.  This reuses only the vendored native
// EXL3/PTX/dequant primitives and is launched from VoidInfer's ordinary
// FP16 device-KV path; it never calls the Mia comparator.  The persistent
// grid and shape-specific tile sizes mirror the independently audited native
// kernel contract while leaving the established split/reduction path intact.
template <int Bits, int TileK, int TileN, bool K7ThreeWord = false>
__global__ __launch_bounds__(256 * TileK / 16)
void exl3_native_persistent_m1_kernel(
    const half* transformed,
    const std::uint16_t* trellis,
    float* accumulation,
    int rows,
    int input_features,
    int output_features,
    int* locks,
    const half* svh) {
    // Keep the persistent reduction in FP32 and apply the established native
    // Hadamard/output conversion afterward. The vendored comparator inner
    // otherwise writes half partials between resident K slices, which changes
    // the first token even when the packed decode is correct.
    exl3_gemm_kernel_inner<Bits, true, 2, 16, TileK, TileN, 4, 3, false,
                           K7ThreeWord>(
        transformed, trellis, accumulation, rows, input_features, output_features,
        locks, svh);
}

// Differential terminal-output adapter for the comparator's normal M=1
// cooperative shapes.  Unlike the FP32 persistent adapter above, this keeps
// the vendored inner's native half-output contract: the final lock owner
// performs exactly one output Hadamard/scaling pass and writes the caller's
// FP16 buffer directly.  It consumes the packed trellis in-place; no
// reconstruction, re-quantization, or comparator delegation is involved.
template <int Bits, int TileK, int TileN, bool K7ThreeWord = false>
__global__ __launch_bounds__(256 * TileK / 16)
void exl3_native_mia_m1_fp16_kernel(
    const half* transformed,
    const std::uint16_t* trellis,
    half* output,
    int rows,
    int input_features,
    int output_features,
    int* locks,
    const half* svh) {
    exl3_gemm_kernel_inner<Bits, false, 2, 16, TileK, TileN, 4, 3, true,
                           K7ThreeWord>(
        transformed, trellis, output, rows, input_features, output_features,
        locks, svh);
}

// Native two-slot MGEMM differential for the same-input gate/up pair. The
// packed inner is the vendored EXL3 implementation already used by this
// binary; this adapter supplies cooperative two-slot scheduling and performs
// output Hadamard after both independent lock domains complete. The original
// FP32 partial/reduction pair remains the fallback.
template <int Bits, int TileK, int TileN, bool K7ThreeWord = false>
__global__ __launch_bounds__(256 * TileK / 16)
void exl3_native_mia_m1_mgemm_pair_kernel(
    const half* first_transformed,
    const std::uint16_t* first_trellis,
    half* first_output,
    int* first_locks,
    const half* first_svh,
    const half* second_transformed,
    const std::uint16_t* second_trellis,
    half* second_output,
    int* second_locks,
    const half* second_svh,
    int rows,
    int input_features,
    int output_features) {
    auto grid = cg::this_grid();
    const bool second = blockIdx.z != 0;
    const half* transformed = second ? second_transformed : first_transformed;
    const std::uint16_t* trellis = second ? second_trellis : first_trellis;
    half* output = second ? second_output : first_output;
    int* locks = second ? second_locks : first_locks;
    const half* svh = second ? second_svh : first_svh;

    // Use a raw FP16 C plane just as Mia's MGEMM inner does. Each z slot has
    // independent locks, so the two projections cannot alias reduction state.
    exl3_gemm_kernel_inner<Bits, false, 2, 16, TileK, TileN, 4, 3, false,
                           K7ThreeWord>(
        transformed, trellis, output, rows, input_features, output_features,
        locks, nullptr);
    // Mia's Blackwell MGEMM uses one inter-CTA barrier per z slot.  The
    // transformed inputs are already complete before this kernel launches, so
    // the pre-inner barrier is unnecessary here; keep the post-inner barrier
    // scoped to the selected projection instead of synchronizing both slots.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
    group_barrier(static_cast<int>(blockIdx.z),
                  static_cast<int>(gridDim.x),
                  locks + output_features / 16);
#else
    grid.sync();
#endif

    const int total_warps = rows * output_features / 128;
    const int warps_grid = gridDim.x * blockDim.x / 32;
    const int this_warp = static_cast<int>(threadIdx.x) / 32 +
        static_cast<int>(blockDim.x) / 32 * static_cast<int>(blockIdx.x);
    for (int warp = this_warp; warp < total_warps; warp += warps_grid) {
        had_hf_r_128_inner<false, true>(
            output + static_cast<std::size_t>(warp) * 128,
            output + static_cast<std::size_t>(warp) * 128,
            svh + (static_cast<std::size_t>(warp) * 128) % output_features,
            0.088388347648f);
    }
}

// Single-matrix companion used by the pinned Mia dispatch policy for K7 and
// narrow/down K6 projections. It keeps Mia's raw-C then output-Hadamard phase
// ordering while remaining a native VoidInfer launch.
template <int Bits, int TileK, int TileN, bool K7ThreeWord = false>
__global__ __launch_bounds__(256 * TileK / 16)
void exl3_native_mia_m1_mgemm_policy_kernel(
    const half* transformed,
    const std::uint16_t* trellis,
    half* output,
    int rows,
    int input_features,
    int output_features,
    int* locks,
    const half* svh) {
    auto grid = cg::this_grid();
    exl3_gemm_kernel_inner<Bits, false, 2, 16, TileK, TileN, 4, 3, false,
                           K7ThreeWord>(
        transformed, trellis, output, rows, input_features, output_features,
        locks, nullptr);
    // For a single matrix this is Mia's group-0 barrier.  It still avoids a
    // full cooperative-grid rendezvous on Blackwell while preserving the
    // raw-C then output-Hadamard phase ordering.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 890)
    group_barrier(0, static_cast<int>(gridDim.x), locks + output_features / 16);
#else
    grid.sync();
#endif

    const int total_warps = rows * output_features / 128;
    const int warps_grid = gridDim.x * blockDim.x / 32;
    const int this_warp = static_cast<int>(threadIdx.x) / 32 +
        static_cast<int>(blockDim.x) / 32 * static_cast<int>(blockIdx.x);
    for (int warp = this_warp; warp < total_warps; warp += warps_grid) {
        had_hf_r_128_inner<false, true>(
            output + static_cast<std::size_t>(warp) * 128,
            output + static_cast<std::size_t>(warp) * 128,
            svh + (static_cast<std::size_t>(warp) * 128) % output_features,
            0.088388347648f);
    }
}

// Native-semantics fused-input differential for the Mia-shaped M=1 inner.
// The first stage deliberately mirrors input_hadamard_kernel: FP16 operands
// are multiplied into FP32 shared values, the butterfly remains FP32, and the
// transformed activation is rounded to FP16 once. Only the launch boundary is
// removed; this does not call the Mia extension or alter the packed weights.
template <int Bits, int TileK, int TileN, bool K7ThreeWord = false>
__global__ __launch_bounds__(256 * TileK / 16)
void exl3_native_mia_m1_fp16_fused_input_kernel(
    const half* raw_input,
    const half* suh,
    half* transformed,
    const std::uint16_t* trellis,
    half* output,
    int rows,
    int input_features,
    int output_features,
    int* locks,
    const half* svh) {
    auto grid = cg::this_grid();
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    __shared__ float transform_values[2][kHadamard];

    // Two warps per CTA perform the input transform. Restricting the staging
    // array to those two warps avoids cross-warp aliasing while the remaining
    // resident warps are still available to the cooperative GEMM phase.
    if (warp < 2) {
        const int total_warps = rows * input_features / kHadamard;
        const int warps_grid = gridDim.x * 2;
        for (int work = static_cast<int>(blockIdx.x) * 2 + warp;
             work < total_warps; work += warps_grid) {
            const int row = work / (input_features / kHadamard);
            const int block = work % (input_features / kHadamard);
            float* values = transform_values[warp];
            for (int i = lane; i < kHadamard; i += 32) {
                const std::size_t offset =
                    static_cast<std::size_t>(row) * input_features +
                    static_cast<std::size_t>(block) * kHadamard + i;
                values[i] = half_product(
                    __half_as_ushort(raw_input[offset]),
                    __half_as_ushort(suh[i]));
            }
            __syncwarp();
            for (int width = 1; width < kHadamard; width *= 2) {
                for (int i = lane; i < kHadamard; i += 32) {
                    if ((i % (2 * width)) < width) {
                        const float left = values[i];
                        const float right = values[i + width];
                        values[i] = left + right;
                        values[i + width] = left - right;
                    }
                }
                __syncwarp();
            }
            for (int i = lane; i < kHadamard; i += 32) {
                const std::size_t offset =
                    static_cast<std::size_t>(row) * input_features +
                    static_cast<std::size_t>(block) * kHadamard + i;
                transformed[offset] = __float2half_rn(
                    values[i] * kHadamardScale);
            }
        }
    }
    grid.sync();
    exl3_gemm_kernel_inner<Bits, false, 2, 16, TileK, TileN, 4, 3, true,
                           K7ThreeWord>(
        transformed, trellis, output, rows, input_features, output_features,
        locks, svh);
}

// Prefill adapter for the same native Mia-derived inner. The inner kernel is
// deliberately a single M-tile leaf (size_m <= 16); y owns independent row
// chunks while x owns the resident K/N slices. The second accumulation plane
// supplies lock storage, so the FP32 output plane and established native
// reduction/output transform remain unchanged.
template <int Bits, int TileK, int TileN, bool K7ThreeWord = false>
__global__ __launch_bounds__(256 * TileK / 16)
void exl3_native_persistent_prefill_kernel(
    const half* transformed,
    const std::uint16_t* trellis,
    float* accumulation,
    int rows,
    int input_features,
    int output_features,
    int* locks,
    const half* svh) {
    const int row_offset = static_cast<int>(blockIdx.y) * 16;
    const int row_count = min(16, rows - row_offset);
    if (row_count <= 0) return;
    const int lock_stride = output_features / 16;
    exl3_gemm_kernel_inner<Bits, true, 2, 16, TileK, TileN, 4, 3, false,
                           K7ThreeWord>(
        transformed + static_cast<std::size_t>(row_offset) * input_features,
        trellis,
        accumulation + static_cast<std::size_t>(row_offset) * output_features,
        row_count, input_features, output_features,
        locks + static_cast<std::size_t>(blockIdx.y) * lock_stride, svh);
}

// Cooperative large-M adapter for the native Mia-shaped FP16-output route.
// Each CTA owns a K/N slice exactly as the vendored inner expects; the whole
// cooperative grid advances through 16-row tiles in lockstep, matching Mia's
// outer kernel instead of relaunching an independent slice grid per row tile.
// The input is already in VoidInfer's canonical transformed domain and the
// output is final FP16 with the inner's established output Hadamard. No weight
// reconstruction, re-quantization, or comparator call is involved.
template <int Bits, int TileK, int TileN, bool K7ThreeWord = false>
__global__ __launch_bounds__(256 * TileK / 16)
void exl3_native_mia_prefill_fp16_kernel(
    const half* transformed,
    const std::uint16_t* trellis,
    half* output,
    int rows,
    int input_features,
    int output_features,
    int* locks,
    const half* svh) {
    auto grid = cg::this_grid();
    for (int row_offset = 0; row_offset < rows; row_offset += 16) {
        const int row_count = min(16, rows - row_offset);
        exl3_gemm_kernel_inner<Bits, false, 2, 16, TileK, TileN, 4, 3, true,
                               K7ThreeWord>(
            transformed + static_cast<std::size_t>(row_offset) * input_features,
            trellis,
            output + static_cast<std::size_t>(row_offset) * output_features,
            row_count, input_features, output_features, locks, svh);
        if (row_offset + 16 < rows) grid.sync();
    }
}

template <int Bits, int TileK, int TileN>
constexpr std::size_t exl3_native_persistent_smem_bytes() {
    constexpr int tile_blocks_k = TileK / 16;
    constexpr int tile_blocks_n = TileN / 16;
    constexpr int fragments_per_warp =
        2 * tile_blocks_n / (256 / 32);
    constexpr int sh_a_stage = 16 * TileK;
    constexpr int sh_b_stage = tile_blocks_k * tile_blocks_n * 16 * Bits;
    constexpr int sh_c_elements =
        (4 * 256 * fragments_per_warp > 16 * TileN)
            ? 4 * 256 * fragments_per_warp
            : 16 * TileN;
    return 4u * (2u * static_cast<std::size_t>(sh_a_stage) * sizeof(half) +
                 2u * static_cast<std::size_t>(sh_b_stage) * sizeof(std::uint16_t)) +
           static_cast<std::size_t>(sh_c_elements) * sizeof(float);
}

// Generalized version of the E2B cooperative leaf.  The model's real EXL3
// projection dimensions are all multiples of a 512-column output block, so a
// block owns 32 packed 16-column tiles and a split owns a contiguous range of
// input tiles.  Bits remains compile-time to keep the packed staging size and
// decode arithmetic specialized for K5/K6/K7/K8.
template <int Bits, bool SingleSplit = false, int OutputTilesPerBlock = 32,
           bool AsyncA = false, bool PartialOnly = false,
           bool K7ThreeWord = false, bool PredecodedB = false,
           bool FastK6Decode = false, bool Fp16Accumulate = false,
           bool RegisterPipeline = false, bool GlobalSlices = false,
           int DeepStages = 0, int Warps = 8, int TilesPerStage = 1>
__global__ void exl3_gemm_m1_generic_mma_kernel(const std::uint16_t* transformed,
                                                 const std::uint16_t* trellis,
                                                 const std::int32_t* mul1,
                                                 float* accum,
                                                 int rows,
                                                 int input_features,
                                                 int output_features,
                                                 int split_count) {
    // Deep async-A stages may stream the constant packed weights of their
    // preload groups before the dependency wait; everything else waits first.
    constexpr bool kEarlyWeights = DeepStages > 0 && AsyncA && !RegisterPipeline;
    if constexpr (!kEarlyWeights) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;");
    constexpr int output_tiles_per_block = OutputTilesPerBlock;
    // Each warp owns fragments_per_warp N8 fragments over the CTA's complete
    // K range. Narrower CTAs (fewer warps, fewer tiles) keep that per-warp
    // work and MMA sequence unchanged while multiplying the grid.
    constexpr int threads = Warps * 32;
    constexpr int fragments_per_warp = output_tiles_per_block * 2 / Warps;
    static_assert(Warps == 8 || Warps == 4 || Warps == 2,
                  "EXL3 packed producer uses 2, 4 or 8 warps");
    static_assert(output_tiles_per_block * 2 % Warps == 0,
                  "EXL3 output tiles must divide across warps");
    static_assert(output_tiles_per_block == 2 || output_tiles_per_block == 4 ||
                      output_tiles_per_block == 8 || output_tiles_per_block == 16 ||
                      output_tiles_per_block == 32 || output_tiles_per_block == 64,
                  "EXL3 output topology must use a proven tile width");
    static_assert(fragments_per_warp == 2 || fragments_per_warp == 4 ||
                       fragments_per_warp == 8 || fragments_per_warp == 16,
                   "EXL3 output topology must map whole MMA fragments");
    static_assert(!RegisterPipeline || AsyncA,
                  "the register pipeline requires async-A staging");
    static_assert(!RegisterPipeline || !Fp16Accumulate,
                   "the register pipeline currently preserves FP32 MMA");
    static_assert(!GlobalSlices ||
                      (!SingleSplit && AsyncA && PartialOnly),
                  "global slices require async-A partial-only accumulation");
    // DeepStages keeps DeepStages-1 packed tiles in flight per CTA. It changes
    // only copy scheduling; every output sees the same MMA sequence.
    static_assert(DeepStages == 0 ||
                      (DeepStages >= 3 && DeepStages <= 8 && AsyncA &&
                       !RegisterPipeline && !Fp16Accumulate),
                  "deep cp.async staging requires the async-A FP32 path");
    constexpr int tile_half = PredecodedB ? 256 : 16 * Bits;
    constexpr int raw_stage_half = output_tiles_per_block * tile_half;
    static_assert(TilesPerStage == 1 ||
                      (DeepStages && (TilesPerStage == 2 || TilesPerStage == 4)),
                  "multi-tile stages require the deep cp.async ring");
    constexpr int raw_stage_count =
        RegisterPipeline ? 4 : (DeepStages ? DeepStages * TilesPerStage : 2);
    extern __shared__ half shared[];
    half* sh_a = shared;
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(
        sh_a + (AsyncA ? raw_stage_count * 256 : 256));
    float* sh_c = reinterpret_cast<float*>(
        sh_raw + raw_stage_count * raw_stage_half);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    const int tiles_k = input_features / 16;
    const int tiles_n = output_features / 16;
    const int output_blocks = (tiles_n + output_tiles_per_block - 1) / output_tiles_per_block;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (SingleSplit ? block : block % output_blocks) * output_tiles_per_block;
    const int split = SingleSplit ? 0 : block / output_blocks;
    const int tiles_per_split = (tiles_k + split_count - 1) / split_count;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, tiles_k);
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    // part: 0 = packed weights and A, 1 = packed weights only, 2 = A only.
    auto prefetch = [&](int tile_k, int stage, bool commit = true, int part = 0) {
        if (tile_k >= tile_k_end) return;
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * tiles_n + tile_base) *
            static_cast<std::size_t>(tile_half);
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        if (part != 2)
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += threads) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        if constexpr (AsyncA) {
            if (part != 1 && thread < 32) {
                const int row = thread / 2;
                const int column = (thread % 2) * 8;
                const int source_column =
                    (thread % 2 ^ ((row >> 2) & 1)) * 8;
                half* destination_a =
                    sh_a + stage * 256 + row * 16 + column;
                if (row < rows) {
                    exl3_cp_async_16(
                        destination_a,
                        transformed + row * input_features +
                            tile_k * 16 + source_column);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        destination_a[j] = __float2half_rn(0.0f);
                    }
                }
            }
        }
        if (commit) exl3_cp_async_commit();
    };

    if constexpr (!RegisterPipeline && DeepStages == 0) {
        prefetch(tile_k_begin, 0);
        exl3_cp_async_wait();
        __syncthreads();
    }

    Exl3FragC c[fragments_per_warp];
    Exl3FragCHalf c_half[fragments_per_warp];
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        #pragma unroll
        for (float& value : c[n].values) value = 0.0f;
        c_half[n].values[0] = __float2half2_rn(0.0f);
        c_half[n].values[1] = __float2half2_rn(0.0f);
    }
    auto fold_half_accumulators = [&]() {
        if constexpr (Fp16Accumulate) {
            #pragma unroll
            for (int n = 0; n < fragments_per_warp; ++n) {
                const float2 low = __half22float2(c_half[n].values[0]);
                const float2 high = __half22float2(c_half[n].values[1]);
                c[n].values[0] += low.x;
                c[n].values[1] += low.y;
                c[n].values[2] += high.x;
                c[n].values[3] += high.y;
                c_half[n].values[0] = __float2half2_rn(0.0f);
                c_half[n].values[1] = __float2half2_rn(0.0f);
            }
        }
    };

    int raw_stage = 0;
    if constexpr (RegisterPipeline) {
        // Mia's shape-4 leaf keeps three decoded B fragments in registers while
        // four cp.async stages feed the next packed tile.  This differential
        // retains the native packed trellis, FP32 MMA accumulators, split
        // topology, and output store; only instruction-level overlap changes.
        constexpr int fragment_stages = 3;
        Exl3FragA a_frag[fragment_stages];
        Exl3FragB b_frag[fragment_stages][fragments_per_warp];
        auto load_fragments = [&](int fragment_stage, int shared_stage) {
            const int r = (lane % 8) + 8 * ((lane / 8) % 2);
            const int base_c = lane / 16;
            const int c_swizzled = base_c ^ ((r >> 2) & 1);
            exl3_ldsm4(a_frag[fragment_stage],
                       sh_a + shared_stage * 256 + r * 16 + c_swizzled * 8);
            #pragma unroll
            for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
                const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
                const auto* packed = reinterpret_cast<const std::uint32_t*>(
                    sh_raw + shared_stage * raw_stage_half +
                    sub_n2 * tile_half);
                if constexpr (PredecodedB) {
                    const auto* values = reinterpret_cast<const half2*>(packed) +
                        lane * 4;
                    b_frag[fragment_stage][n2].values[0] = values[0];
                    b_frag[fragment_stage][n2].values[1] = values[1];
                    b_frag[fragment_stage][n2 + 1].values[0] = values[2];
                    b_frag[fragment_stage][n2 + 1].values[1] = values[3];
                } else if constexpr (FastK6Decode) {
                    static_assert(Bits == 6,
                                  "fast M1 lane-window decoder is K6-only");
                    exl3_dq4_k6_lane_window(
                        packed, lane << 3, b_frag[fragment_stage][n2],
                        mul1_multiplier);
                    exl3_dq4_k6_lane_window(
                        packed, (lane << 3) + 4,
                        b_frag[fragment_stage][n2 + 1], mul1_multiplier);
                } else {
                    exl3_dq4_generic<Bits, K7ThreeWord>(
                        packed, lane << 3, b_frag[fragment_stage][n2],
                        mul1_multiplier);
                    exl3_dq4_generic<Bits, K7ThreeWord>(
                        packed, (lane << 3) + 4,
                        b_frag[fragment_stage][n2 + 1], mul1_multiplier);
                }
            }
            __syncthreads();
        };
        auto mma_fragments = [&](int fragment_stage) {
            #pragma unroll
            for (int n = 0; n < fragments_per_warp; ++n)
                exl3_mma_m16n8k16(
                    a_frag[fragment_stage], b_frag[fragment_stage][n], c[n]);
        };

        #pragma unroll
        for (int preload = 0; preload < 3; ++preload)
            prefetch(tile_k_begin + preload, preload);
        exl3_cp_async_wait_group<2>();
        __syncthreads();
        load_fragments(0, 0);
        int fragment_stage = 0;
        for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
            const int relative_tile = tile_k - tile_k_begin;
            prefetch(tile_k + 3, (relative_tile + 3) & 3);
            exl3_cp_async_wait_group<2>();
            mma_fragments(fragment_stage);
            if (tile_k + 1 < tile_k_end) {
                // The wait-group order guarantees the next shared stage is
                // complete before every warp decodes it into registers.
                exl3_cp_async_wait_group<1>();
                // wait_group 1 intentionally permits one pending group. At
                // the final handoff that group can be the next fragment, so
                // close the tail before decoding it from shared memory.
                if (tile_k + 2 >= tile_k_end)
                    exl3_cp_async_wait_group<0>();
                __syncthreads();
                const int next_fragment = (fragment_stage + 1) % fragment_stages;
                load_fragments(next_fragment, (relative_tile + 1) & 3);
                fragment_stage = next_fragment;
            }
        }
        exl3_cp_async_wait_group<0>();
        __syncthreads();
    } else if constexpr (DeepStages > 0) {
        constexpr int stages = DeepStages;
        constexpr int per = TilesPerStage;
        // One committed group per stage of `per` consecutive k-tiles,
        // including empty tail groups, so wait_pending<stages-2> always
        // completes the current stage. Tiles are consumed in ascending k.
        auto issue = [&](int group, int stage, int part = 0) {
            #pragma unroll
            for (int t = 0; t < per; ++t)
                prefetch(tile_k_begin + group * per + t, stage * per + t, false, part);
            exl3_cp_async_commit();
        };
        if constexpr (kEarlyWeights) {
            // Weights for every preload stage, then (after the dependency
            // wait) the producer-written A slices. Commit order B0..B(s-2),
            // A0..A(s-2), then combined groups: each wait_pending<stages-2>
            // below still covers the stage it consumes.
            #pragma unroll
            for (int preload = 0; preload < stages - 1; ++preload)
                issue(preload, preload, 1);
            asm volatile("griddepcontrol.wait;" ::: "memory");
            #pragma unroll
            for (int preload = 0; preload < stages - 1; ++preload)
                issue(preload, preload, 2);
        } else {
        #pragma unroll
        for (int preload = 0; preload < stages - 1; ++preload)
            issue(preload, preload);
        }
        const int groups = (tile_k_end - tile_k_begin + per - 1) / per;
        for (int group = 0; group < groups; ++group) {
            exl3_cp_async_wait_pending<stages - 2>();
            // Also orders the previous iteration's shared reads before the
            // refill of its stage below.
            __syncthreads();
            issue(group + stages - 1, (group + stages - 1) % stages);
          #pragma unroll
          for (int t = 0; t < per; ++t) {
            if (tile_k_begin + group * per + t >= tile_k_end) break;
            const int stage = (group % stages) * per + t;
            Exl3FragA a;
            const int r = (lane % 8) + 8 * ((lane / 8) % 2);
            const int base_c = lane / 16;
            const int c_swizzled = base_c ^ ((r >> 2) & 1);
            exl3_ldsm4(a, sh_a + stage * 256 + r * 16 + c_swizzled * 8);
            #pragma unroll
            for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
                const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
                const auto* packed = reinterpret_cast<const std::uint32_t*>(
                    sh_raw + stage * raw_stage_half + sub_n2 * tile_half);
                Exl3FragB b0, b1;
                if constexpr (PredecodedB) {
                    const auto* values = reinterpret_cast<const half2*>(packed) +
                        lane * 4;
                    b0.values[0] = values[0]; b0.values[1] = values[1];
                    b1.values[0] = values[2]; b1.values[1] = values[3];
                } else if constexpr (FastK6Decode) {
                    static_assert(Bits==6,"fast M1 lane-window decoder is K6-only");
                    exl3_dq4_k6_lane_window(packed,lane<<3,b0,mul1_multiplier);
                    exl3_dq4_k6_lane_window(packed,(lane<<3)+4,b1,mul1_multiplier);
                } else {
                    exl3_dq4_generic<Bits, K7ThreeWord>(
                        packed, lane << 3, b0, mul1_multiplier);
                    exl3_dq4_generic<Bits, K7ThreeWord>(
                        packed, (lane << 3) + 4, b1, mul1_multiplier);
                }
                exl3_mma_m16n8k16(a, b0, c[n2]);
                exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
            }
          }
        }
        exl3_cp_async_wait();
        __syncthreads();
    } else {
        for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
            if constexpr (!AsyncA) {
                for (int i = thread; i < 256; i += threads) {
                    const int row = i / 16;
                    const int column = i % 16;
                    const int source_column =
                        (column / 8 ^ ((row >> 2) & 1)) * 8 + column % 8;
                    sh_a[i] = row < rows
                        ? __ushort_as_half(
                              transformed[row * input_features + tile_k * 16 +
                                          source_column])
                        : __float2half_rn(0.0f);
                }
                __syncthreads();
            }
            prefetch(tile_k + 1, 1 - raw_stage);

            Exl3FragA a;
            const int r = (lane % 8) + 8 * ((lane / 8) % 2);
            const int base_c = lane / 16;
            const int c_swizzled = base_c ^ ((r >> 2) & 1);
            exl3_ldsm4(a, sh_a + (AsyncA ? raw_stage * 256 : 0) +
                                 r * 16 + c_swizzled * 8);

            #pragma unroll
            for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
                const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
                const auto* packed = reinterpret_cast<const std::uint32_t*>(
                    sh_raw + raw_stage * raw_stage_half + sub_n2 * tile_half);
                Exl3FragB b0, b1;
                if constexpr (PredecodedB) {
                    const auto* values = reinterpret_cast<const half2*>(packed) +
                        lane * 4;
                    b0.values[0] = values[0]; b0.values[1] = values[1];
                    b1.values[0] = values[2]; b1.values[1] = values[3];
                } else if constexpr (FastK6Decode) {
                    static_assert(Bits==6,"fast M1 lane-window decoder is K6-only");
                    exl3_dq4_k6_lane_window(
                        packed,lane<<3,b0,mul1_multiplier);
                    exl3_dq4_k6_lane_window(
                        packed,(lane<<3)+4,b1,mul1_multiplier);
                } else {
                    exl3_dq4_generic<Bits, K7ThreeWord>(
                        packed, lane << 3, b0, mul1_multiplier);
                    exl3_dq4_generic<Bits, K7ThreeWord>(
                        packed, (lane << 3) + 4, b1, mul1_multiplier);
                }
                if constexpr (Fp16Accumulate) {
                    exl3_mma_m16n8k16_half(a, b0, c_half[n2]);
                    exl3_mma_m16n8k16_half(a, b1, c_half[n2 + 1]);
                } else {
                    exl3_mma_m16n8k16(a, b0, c[n2]);
                    exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
                }
            }

            if constexpr (Fp16Accumulate) {
                if (((tile_k - tile_k_begin) & 3) == 3) fold_half_accumulators();
            }

            if (tile_k + 1 < tile_k_end) {
                exl3_cp_async_wait();
                __syncthreads();
            }
            raw_stage = 1 - raw_stage;
        }
    }

    fold_half_accumulators();
    const int n0 = warp * fragments_per_warp;
    constexpr int output_tile_elements = output_tiles_per_block * 16;
    const int r0 = lane / 4;
    const int r1 = r0 + 8;
    const int column = (lane % 4) * 2;
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        if (r0 < rows) {
            float* destination = sh_c + r0 * output_tile_elements + (n0 + n) * 8 + column;
            destination[0] = c[n].values[0];
            destination[1] = c[n].values[1];
        }
        if (r1 < rows) {
            float* destination = sh_c + r1 * output_tile_elements + (n0 + n) * 8 + column;
            destination[0] = c[n].values[2];
            destination[1] = c[n].values[3];
        }
    }
    __syncthreads();
    const std::size_t partial_stride =
        static_cast<std::size_t>(rows) * output_features;
    if constexpr (GlobalSlices) {
        // The cooperative launch guarantees every producer for a lock is
        // resident, so the ordered spin cannot be starved by an unscheduled
        // predecessor.  Split N32 output blocks are the lock granularity.
        int* locks = reinterpret_cast<int*>(accum + output_features);
        const int lock_stage = split_count - split - 1;
        const bool first = split == split_count - 1;
        const bool last = split == 0;
        exl3_global_slice_acquire(locks + block % output_blocks, lock_stage);
        for (int i = thread; i < rows * output_tile_elements; i += threads) {
            const int row = i / output_tile_elements;
            const int column = i % output_tile_elements;
            float* destination = accum + row * output_features +
                tile_base * 16 + column;
            if (first)
                *destination = sh_c[i];
            else
                *destination += sh_c[i];
        }
        exl3_global_slice_release(locks + block % output_blocks,
                                  lock_stage + 1, last);
    } else {
        for (int i = thread; i < rows * output_tile_elements; i += threads) {
            const int row = i / output_tile_elements;
            const int column = i % output_tile_elements;
            if constexpr (SingleSplit) {
                accum[row * output_features + tile_base * 16 + column] = sh_c[i];
            } else {
                accum[static_cast<std::size_t>(split) * partial_stride +
                      row * output_features + tile_base * 16 + column] = sh_c[i];
            }
        }
    }
    if constexpr (!SingleSplit && !PartialOnly) {
        auto grid = cg::this_grid();
        grid.sync();
    }
    if constexpr (!SingleSplit && !PartialOnly) {
      if (split == 0) {
        for (int i = thread; i < rows * output_tile_elements; i += threads) {
            const int row = i / output_tile_elements;
            const int column = i % output_tile_elements;
            float value = sh_c[i];
            for (int other = 1; other < split_count; ++other) {
                value += accum[static_cast<std::size_t>(other) * partial_stride +
                              row * output_features + tile_base * 16 + column];
            }
            accum[row * output_features + tile_base * 16 + column] = value;
        }
      }
    }
}

// One CTA owns a complete 128-column Hadamard group for up to eight rows.
// Unlike the split small-M path, its packed MMA result stays in shared memory
// through the output transform: no global FP32 partials or reduction launch.
// The bounded K6 down shape is the first admitted owner of this dataflow.
__global__ void exl3_small_m_down_fused_output_kernel(
    const std::uint16_t* transformed, const std::uint16_t* trellis,
    const std::int32_t* mul1, const std::uint16_t* svh,
    std::uint16_t* output, int rows, int input_features,
    int output_features) {
    constexpr int output_tiles = 8;
    constexpr int raw_stage_half = output_tiles * 16 * 6;
    __shared__ half sh_a[256];
    __shared__ std::uint16_t sh_raw[2 * raw_stage_half];
    __shared__ float sh_c[16 * kHadamard];
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    const int tile_base = static_cast<int>(blockIdx.x) * output_tiles;
    const int tiles_n = output_features / 16;
    const int tiles_k = input_features / 16;
    const std::uint32_t multiplier = static_cast<std::uint32_t>(*mul1);
    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tiles_k) return;
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * tiles_n + tile_base) * 16u * 6u;
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += kThreads)
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        exl3_cp_async_commit();
    };
    prefetch(0, 0);
    exl3_cp_async_wait();
    __syncthreads();
    Exl3FragC c[2];
    #pragma unroll
    for (auto& fragment : c)
        for (float& value : fragment.values) value = 0.0f;
    int raw_stage = 0;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k) {
        for (int i = thread; i < 256; i += kThreads) {
            const int row = i / 16;
            const int column = i % 16;
            const int source_column =
                (column / 8 ^ ((row >> 2) & 1)) * 8 + column % 8;
            sh_a[i] = row < rows
                ? __ushort_as_half(transformed[row * input_features +
                                               tile_k * 16 + source_column])
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        prefetch(tile_k + 1, 1 - raw_stage);
        Exl3FragA a;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a, sh_a + r * 16 + c_swizzled * 8);
        const auto* packed = reinterpret_cast<const std::uint32_t*>(
            sh_raw + raw_stage * raw_stage_half + warp * 16 * 6);
        Exl3FragB b0, b1;
        exl3_dq4_generic<6>(packed, lane << 3, b0, multiplier);
        exl3_dq4_generic<6>(packed, (lane << 3) + 4, b1, multiplier);
        exl3_mma_m16n8k16(a, b0, c[0]);
        exl3_mma_m16n8k16(a, b1, c[1]);
        if (tile_k + 1 < tiles_k) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage = 1 - raw_stage;
    }
    const int r0 = lane / 4;
    const int r1 = r0 + 8;
    const int column = (lane % 4) * 2;
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        if (r0 < rows) {
            float* destination = sh_c + r0 * kHadamard + warp * 16 + n * 8 + column;
            destination[0] = c[n].values[0];
            destination[1] = c[n].values[1];
        }
        if (r1 < rows) {
            float* destination = sh_c + r1 * kHadamard + warp * 16 + n * 8 + column;
            destination[0] = c[n].values[2];
            destination[1] = c[n].values[3];
        }
    }
    __syncthreads();
    // Two 128-thread groups transform one row each, four passes for M=8.
    const int row_lane = thread & (kHadamard - 1);
    const int row_group = thread / kHadamard;
    for (int row_base = 0; row_base < rows; row_base += 2) {
        const int row = row_base + row_group;
        for (int width = 1; width < kHadamard; width *= 2) {
            if (row < rows && row_lane % (2 * width) < width) {
                const int index = row * kHadamard + row_lane;
                const float left = sh_c[index];
                const float right = sh_c[index + width];
                sh_c[index] = left + right;
                sh_c[index + width] = left - right;
            }
            __syncthreads();
        }
        if (row < rows) {
            const int offset = static_cast<int>(blockIdx.x) * kHadamard + row_lane;
            const float value = sh_c[row * kHadamard + row_lane];
            const auto normalized = __float2half_rn(value * kHadamardScale);
            const auto scale = __ushort_as_half(svh[offset]);
            output[row * output_features + offset] =
                __half_as_ushort(__hmul(normalized, scale));
        }
        __syncthreads();
    }
}

// Identical packed/MMA partial arithmetic; stream ordering replaces grid sync.
template <int Bits, int OutputTilesPerBlock = 32>
__global__ void exl3_prefill_partials_kernel(const std::uint16_t* transformed,
                                                 const std::uint16_t* trellis,
                                                 const std::int32_t* mul1,
                                                 float* accum,
                                                 int rows,
                                                 int input_features,
                                                 int output_features,
                                                 int split_count) {
    const int total_rows = rows;
    const int row_base = static_cast<int>(blockIdx.y) * 16;
    rows = min(16, total_rows - row_base);
    transformed += static_cast<std::size_t>(row_base) * input_features;
    constexpr bool SingleSplit = false;
    constexpr int output_tiles_per_block = OutputTilesPerBlock;
    constexpr int fragments_per_warp = output_tiles_per_block / 4;
    static_assert(output_tiles_per_block == 8 || output_tiles_per_block == 16 || output_tiles_per_block == 32,
                  "EXL3 output topology must use a proven tile width");
    static_assert(fragments_per_warp == 2 || fragments_per_warp == 4 || fragments_per_warp == 8,
                  "EXL3 output topology must map whole MMA fragments");
    constexpr int raw_stage_half = output_tiles_per_block * 16 * Bits;
    extern __shared__ half shared[];
    half* sh_a = shared;
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(sh_a + 256);
    float* sh_c = reinterpret_cast<float*>(sh_raw + 2 * raw_stage_half);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    const int tiles_k = input_features / 16;
    const int tiles_n = output_features / 16;
    const int output_blocks = (tiles_n + output_tiles_per_block - 1) / output_tiles_per_block;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (SingleSplit ? block : block % output_blocks) * output_tiles_per_block;
    const int split = SingleSplit ? 0 : block / output_blocks;
    const int tiles_per_split = (tiles_k + split_count - 1) / split_count;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, tiles_k);
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tile_k_end) return;
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * tiles_n + tile_base) *
            static_cast<std::size_t>(16 * Bits);
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += kThreads) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        exl3_cp_async_commit();
    };

    prefetch(tile_k_begin, 0);
    exl3_cp_async_wait();
    __syncthreads();

    Exl3FragC c[fragments_per_warp];
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        #pragma unroll
        for (float& value : c[n].values) value = 0.0f;
    }

    int raw_stage = 0;
    for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
        for (int i = thread; i < 256; i += kThreads) {
            const int row = i / 16;
            const int column = i % 16;
            const int source_column = (column / 8 ^ ((row >> 2) & 1)) * 8 + column % 8;
            sh_a[i] = row < rows
                ? __ushort_as_half(transformed[row * input_features + tile_k * 16 + source_column])
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        prefetch(tile_k + 1, 1 - raw_stage);

        Exl3FragA a;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a, sh_a + r * 16 + c_swizzled * 8);

        #pragma unroll
        for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
            const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
            const auto* packed = reinterpret_cast<const std::uint32_t*>(
                sh_raw + raw_stage * raw_stage_half + sub_n2 * (16 * Bits));
            Exl3FragB b0, b1;
            exl3_dq4_generic<Bits>(packed, lane << 3, b0, mul1_multiplier);
            exl3_dq4_generic<Bits>(packed, (lane << 3) + 4, b1, mul1_multiplier);
            exl3_mma_m16n8k16(a, b0, c[n2]);
            exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
        }

        if (tile_k + 1 < tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage = 1 - raw_stage;
    }

    const int n0 = warp * fragments_per_warp;
    constexpr int output_tile_elements = output_tiles_per_block * 16;
    const int r0 = lane / 4;
    const int r1 = r0 + 8;
    const int column = (lane % 4) * 2;
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        if (r0 < rows) {
            float* destination = sh_c + r0 * output_tile_elements + (n0 + n) * 8 + column;
            destination[0] = c[n].values[0];
            destination[1] = c[n].values[1];
        }
        if (r1 < rows) {
            float* destination = sh_c + r1 * output_tile_elements + (n0 + n) * 8 + column;
            destination[0] = c[n].values[2];
            destination[1] = c[n].values[3];
        }
    }
    __syncthreads();
    const std::size_t partial_stride = static_cast<std::size_t>(total_rows) * output_features;
    for (int i = thread; i < rows * output_tile_elements; i += kThreads) {
        const int row = i / output_tile_elements;
        const int column = i % output_tile_elements;
        if constexpr (SingleSplit) {
            accum[row * output_features + tile_base * 16 + column] = sh_c[i];
        } else {
            accum[static_cast<std::size_t>(split) * partial_stride +
                  (row_base + row) * output_features + tile_base * 16 + column] = sh_c[i];
        }
    }
}

template <int Bits, int OutputTilesPerBlock = 32, bool Predecoded = false>
__global__ void exl3_prefill_direct_partials_kernel(const std::uint16_t* transformed,
                                                 const std::uint16_t* trellis,
                                                 const std::int32_t* mul1,
                                                 float* accum,
                                                 int rows,
                                                 int input_features,
                                                 int output_features,
                                                 int split_count,
                                                 int first_output_tile = 0,
                                                 int decoded_output_tiles = 0) {
    const int total_rows = rows;
    const int row_base = static_cast<int>(blockIdx.y) * 16;
    rows = min(16, total_rows - row_base);
    transformed += static_cast<std::size_t>(row_base) * input_features;
    constexpr bool SingleSplit = false;
    constexpr int output_tiles_per_block = OutputTilesPerBlock;
    constexpr int fragments_per_warp = output_tiles_per_block / 4;
    static_assert(output_tiles_per_block == 8 || output_tiles_per_block == 16 || output_tiles_per_block == 32,
                  "EXL3 output topology must use a proven tile width");
    static_assert(fragments_per_warp == 2 || fragments_per_warp == 4 || fragments_per_warp == 8,
                  "EXL3 output topology must map whole MMA fragments");
    constexpr int tile_half = Predecoded ? 256 : 16 * Bits;
    constexpr int raw_stage_half = output_tiles_per_block * tile_half;
    extern __shared__ half shared[];
    half* sh_a = shared;
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(sh_a + 256);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    const int tiles_k = input_features / 16;
    const int tiles_n = Predecoded && decoded_output_tiles ? decoded_output_tiles : output_features / 16;
    const int output_blocks = (tiles_n + output_tiles_per_block - 1) / output_tiles_per_block;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (SingleSplit ? block : block % output_blocks) * output_tiles_per_block;
    const int split = SingleSplit ? 0 : block / output_blocks;
    const int tiles_per_split = (tiles_k + split_count - 1) / split_count;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, tiles_k);
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tile_k_end) return;
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * tiles_n + tile_base) *
            static_cast<std::size_t>(tile_half);
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += kThreads) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        exl3_cp_async_commit();
    };

    prefetch(tile_k_begin, 0);
    exl3_cp_async_wait();
    __syncthreads();

    Exl3FragC c[fragments_per_warp];
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        #pragma unroll
        for (float& value : c[n].values) value = 0.0f;
    }

    int raw_stage = 0;
    for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
        for (int i = thread; i < 256; i += kThreads) {
            const int row = i / 16;
            const int column = i % 16;
            const int source_column = (column / 8 ^ ((row >> 2) & 1)) * 8 + column % 8;
            sh_a[i] = row < rows
                ? __ushort_as_half(transformed[row * input_features + tile_k * 16 + source_column])
                : __float2half_rn(0.0f);
        }
        __syncthreads();
        prefetch(tile_k + 1, 1 - raw_stage);

        Exl3FragA a;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a, sh_a + r * 16 + c_swizzled * 8);

        #pragma unroll
        for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
            const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
            const auto* packed = reinterpret_cast<const std::uint32_t*>(
                sh_raw + raw_stage * raw_stage_half + sub_n2 * tile_half);
            Exl3FragB b0, b1;
            if constexpr (Predecoded) {
                const auto* values = reinterpret_cast<const half2*>(packed) + lane * 4;
                b0.values[0] = values[0]; b0.values[1] = values[1];
                b1.values[0] = values[2]; b1.values[1] = values[3];
            } else {
                exl3_dq4_generic<Bits>(packed, lane << 3, b0, mul1_multiplier);
                exl3_dq4_generic<Bits>(packed, (lane << 3) + 4, b1, mul1_multiplier);
            }
            exl3_mma_m16n8k16(a, b0, c[n2]);
            exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
        }

        if (tile_k + 1 < tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage = 1 - raw_stage;
    }

    const int n0=warp*fragments_per_warp;
    const int r0=lane/4, r1=r0+8, column=(lane%4)*2;
    const std::size_t stride=static_cast<std::size_t>(total_rows)*output_features;
    #pragma unroll
    for(int n=0;n<fragments_per_warp;++n) {
        const int col=(tile_base+(Predecoded?first_output_tile:0))*16+(n0+n)*8+column;
        if(r0<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+static_cast<std::size_t>(row_base+r0)*output_features+col;
            dst[0]=c[n].values[0];dst[1]=c[n].values[1];
        }
        if(r1<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+static_cast<std::size_t>(row_base+r1)*output_features+col;
            dst[0]=c[n].values[2];dst[1]=c[n].values[3];
        }
    }
}

template <int Bits, int OutputTilesPerBlock = 32, int MinBlocksPerSM = 4,
          bool Predecoded = false,bool FastK6Decode = false>
__global__ __launch_bounds__(256,MinBlocksPerSM) void exl3_prefill_direct_async_a_kernel(const std::uint16_t* transformed,
                                                 const std::uint16_t* trellis,
                                                 const std::int32_t* mul1,
                                                 float* accum,
                                                 int rows,
                                                 int input_features,
                                                 int output_features,
                                                 int split_count,
                                                 int first_output_tile = 0,
                                                 int decoded_output_tiles = 0) {
    const int total_rows = rows;
    const int row_base = static_cast<int>(blockIdx.y) * 16;
    rows = min(16, total_rows - row_base);
    transformed += static_cast<std::size_t>(row_base) * input_features;
    constexpr bool SingleSplit = false;
    constexpr int output_tiles_per_block = OutputTilesPerBlock;
    constexpr int fragments_per_warp = output_tiles_per_block / 4;
    static_assert(output_tiles_per_block == 8 || output_tiles_per_block == 16 || output_tiles_per_block == 32 || output_tiles_per_block == 64,
                  "EXL3 output topology must use a proven tile width");
    static_assert(fragments_per_warp == 2 || fragments_per_warp == 4 || fragments_per_warp == 8 || fragments_per_warp == 16,
                  "EXL3 output topology must map whole MMA fragments");
    constexpr int tile_half = Predecoded ? 256 : 16 * Bits;
    constexpr int raw_stage_half = output_tiles_per_block * tile_half;
    extern __shared__ half shared[];
    half* sh_a = shared;
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(sh_a + 512);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    const int tiles_k = input_features / 16;
    const int tiles_n = Predecoded && decoded_output_tiles ? decoded_output_tiles : output_features / 16;
    const int output_blocks = (tiles_n + output_tiles_per_block - 1) / output_tiles_per_block;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (SingleSplit ? block : block % output_blocks) * output_tiles_per_block;
    const int split = SingleSplit ? 0 : block / output_blocks;
    const int tiles_per_split = (tiles_k + split_count - 1) / split_count;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, tiles_k);
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tile_k_end) return;
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * tiles_n + tile_base) *
            static_cast<std::size_t>(tile_half);
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += kThreads) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        if(thread<32) {
            const int row=thread/2;
            const int column=(thread%2)*8;
            const int source_column=(thread%2 ^ ((row>>2)&1))*8;
            half* next_a=sh_a+stage*256+row*16+column;
            if(row<rows) exl3_cp_async_16(next_a,transformed+row*input_features+tile_k*16+source_column);
            else {
                #pragma unroll
                for(int j=0;j<8;++j) next_a[j]=__float2half_rn(0.0f);
            }
        }
        exl3_cp_async_commit();
    };
    prefetch(tile_k_begin, 0);
    exl3_cp_async_wait();
    __syncthreads();

    Exl3FragC c[fragments_per_warp];
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        #pragma unroll
        for (float& value : c[n].values) value = 0.0f;
    }

    int raw_stage = 0;
    for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
        prefetch(tile_k + 1, 1 - raw_stage);

        Exl3FragA a;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a, sh_a + raw_stage*256 + r * 16 + c_swizzled * 8);

        #pragma unroll
        for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
            const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
            const auto* packed = reinterpret_cast<const std::uint32_t*>(
                sh_raw + raw_stage * raw_stage_half + sub_n2 * tile_half);
            Exl3FragB b0, b1;
            if constexpr (Predecoded) {
                const auto* values = reinterpret_cast<const half2*>(packed) + lane * 4;
                b0.values[0] = values[0]; b0.values[1] = values[1];
                b1.values[0] = values[2]; b1.values[1] = values[3];
            } else if constexpr(FastK6Decode) {
                static_assert(Bits==6,"fast lane-window decoder is K6-only");
                exl3_dq4_k6_lane_window(packed,lane<<3,b0,mul1_multiplier);
                exl3_dq4_k6_lane_window(packed,(lane<<3)+4,b1,mul1_multiplier);
            } else {
                exl3_dq4_generic<Bits>(packed, lane << 3, b0, mul1_multiplier);
                exl3_dq4_generic<Bits>(packed, (lane << 3) + 4, b1, mul1_multiplier);
            }
            exl3_mma_m16n8k16(a, b0, c[n2]);
            exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
        }

        if (tile_k + 1 < tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage = 1 - raw_stage;
    }

    const int n0=warp*fragments_per_warp;
    const int r0=lane/4, r1=r0+8, column=(lane%4)*2;
    const std::size_t stride=static_cast<std::size_t>(total_rows)*output_features;
    #pragma unroll
    for(int n=0;n<fragments_per_warp;++n) {
        const int col=(tile_base+(Predecoded?first_output_tile:0))*16+(n0+n)*8+column;
        if(r0<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+static_cast<std::size_t>(row_base+r0)*output_features+col;
            dst[0]=c[n].values[0];dst[1]=c[n].values[1];
        }
        if(r1<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+static_cast<std::size_t>(row_base+r1)*output_features+col;
            dst[0]=c[n].values[2];dst[1]=c[n].values[3];
        }
    }
}

// Two independent M16 row groups retain the selected N64 output topology while
// sharing one packed K6 B stage.  Each 256-thread half owns exactly the same MMA
// fragments and chronological FP32 chain as the selected single-group kernel;
// only duplicate global staging and dequantization of immutable weights are
// removed.  The 512-thread CTA has the same sixteen resident warps as two
// control CTAs and never crosses row-group output ownership.
__global__ __launch_bounds__(512,1) void exl3_prefill_k6_rowpair_n64_kernel(
    const std::uint16_t* transformed,const std::uint16_t* trellis,
    const std::int32_t* mul1,float* accum,int rows,int input_features,
    int output_features,int split_count) {
    constexpr int output_tiles_per_block=64;
    constexpr int fragments_per_warp=16;
    constexpr int raw_stage_half=output_tiles_per_block*16*6;
    const int total_rows=rows;
    const int row_base=static_cast<int>(blockIdx.y)*32;
    rows=min(32,total_rows-row_base);
    transformed+=static_cast<std::size_t>(row_base)*input_features;
    extern __shared__ half shared[];
    half* sh_a=shared;
    auto* sh_raw=reinterpret_cast<std::uint16_t*>(sh_a+1024);
    const int thread=static_cast<int>(threadIdx.x);
    const int group=thread/256;
    const int local_thread=thread-group*256;
    const int warp=local_thread/32;
    const int lane=local_thread&31;
    const int tiles_k=input_features/16;
    const int tiles_n=output_features/16;
    const int output_blocks=(tiles_n+output_tiles_per_block-1)/output_tiles_per_block;
    const int block=static_cast<int>(blockIdx.x);
    const int tile_base=(block%output_blocks)*output_tiles_per_block;
    const int split=block/output_blocks;
    const int tiles_per_split=(tiles_k+split_count-1)/split_count;
    const int tile_k_begin=split*tiles_per_split;
    const int tile_k_end=min(tile_k_begin+tiles_per_split,tiles_k);
    const std::uint32_t mul1_multiplier=static_cast<std::uint32_t>(*mul1);

    auto prefetch=[&](int tile_k,int stage) {
        if(tile_k>=tile_k_end)return;
        const std::size_t offset=
            (static_cast<std::size_t>(tile_k)*tiles_n+tile_base)*16u*6u;
        auto* destination=sh_raw+stage*raw_stage_half;
        const auto* source=trellis+offset;
        for(int chunk=thread;chunk<raw_stage_half/8;chunk+=512)
            exl3_cp_async_16(destination+chunk*8,source+chunk*8);
        if(thread<64) {
            const int row=thread/2;
            const int column=(thread%2)*8;
            const int source_column=(thread%2^((row>>2)&1))*8;
            half* next_a=sh_a+stage*512+row*16+column;
            if(row<rows)
                exl3_cp_async_16(next_a,transformed+row*input_features+
                    tile_k*16+source_column);
            else for(int j=0;j<8;++j)next_a[j]=__float2half_rn(0.0f);
        }
        exl3_cp_async_commit();
    };
    prefetch(tile_k_begin,0);
    exl3_cp_async_wait();
    __syncthreads();

    Exl3FragC c[fragments_per_warp];
    #pragma unroll
    for(int n=0;n<fragments_per_warp;++n)
        #pragma unroll
        for(float& value:c[n].values)value=0.0f;

    int raw_stage=0;
    for(int tile_k=tile_k_begin;tile_k<tile_k_end;++tile_k) {
        prefetch(tile_k+1,1-raw_stage);
        Exl3FragA a;
        const int r=(lane%8)+8*((lane/8)%2);
        const int base_c=lane/16;
        const int c_swizzled=base_c^((r>>2)&1);
        exl3_ldsm4(a,sh_a+raw_stage*512+group*256+r*16+c_swizzled*8);
        #pragma unroll
        for(int n2=0;n2<fragments_per_warp;n2+=2) {
            const int sub_n2=warp*(fragments_per_warp/2)+n2/2;
            const auto* packed=reinterpret_cast<const std::uint32_t*>(
                sh_raw+raw_stage*raw_stage_half+sub_n2*16*6);
            Exl3FragB b0,b1;
            exl3_dq4_generic<6>(packed,lane<<3,b0,mul1_multiplier);
            exl3_dq4_generic<6>(packed,(lane<<3)+4,b1,mul1_multiplier);
            exl3_mma_m16n8k16(a,b0,c[n2]);
            exl3_mma_m16n8k16(a,b1,c[n2+1]);
        }
        if(tile_k+1<tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage=1-raw_stage;
    }

    const int n0=warp*fragments_per_warp;
    const int r0=lane/4,r1=r0+8,column=(lane%4)*2;
    const int group_row=group*16;
    const std::size_t stride=static_cast<std::size_t>(total_rows)*output_features;
    #pragma unroll
    for(int n=0;n<fragments_per_warp;++n) {
        const int col=tile_base*16+(n0+n)*8+column;
        if(group_row+r0<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+
                static_cast<std::size_t>(row_base+group_row+r0)*output_features+col;
            dst[0]=c[n].values[0];dst[1]=c[n].values[1];
        }
        if(group_row+r1<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+
                static_cast<std::size_t>(row_base+group_row+r1)*output_features+col;
            dst[0]=c[n].values[2];dst[1]=c[n].values[3];
        }
    }
}

// Discriminator-only counterpart to the row-pair N64 kernel. Both row groups
// keep independent A fragments and FP32 MMA chains, while group zero decodes
// each immutable packed B tile once into the native fragment order consumed by
// both groups. This isolates duplicate dequantization from global B staging.
__global__ __launch_bounds__(512,1)
void exl3_prefill_k6_rowpair_n64_shared_decode_kernel(
    const std::uint16_t* transformed,const std::uint16_t* trellis,
    const std::int32_t* mul1,float* accum,int rows,int input_features,
    int output_features,int split_count) {
    constexpr int output_tiles_per_block=64;
    constexpr int fragments_per_warp=16;
    constexpr int raw_stage_half=output_tiles_per_block*16*6;
    constexpr int decoded_stage_half=output_tiles_per_block*256;
    const int total_rows=rows;
    const int row_base=static_cast<int>(blockIdx.y)*32;
    rows=min(32,total_rows-row_base);
    transformed+=static_cast<std::size_t>(row_base)*input_features;
    extern __shared__ half shared[];
    half* sh_a=shared;
    auto* sh_raw=reinterpret_cast<std::uint16_t*>(sh_a+1024);
    auto* sh_decoded=reinterpret_cast<std::uint16_t*>(
        sh_raw+2*raw_stage_half);
    const int thread=static_cast<int>(threadIdx.x);
    const int group=thread/256;
    const int local_thread=thread-group*256;
    const int warp=local_thread/32;
    const int lane=local_thread&31;
    const int tiles_k=input_features/16;
    const int tiles_n=output_features/16;
    const int output_blocks=tiles_n/output_tiles_per_block;
    const int block=static_cast<int>(blockIdx.x);
    const int tile_base=(block%output_blocks)*output_tiles_per_block;
    const int split=block/output_blocks;
    const int tiles_per_split=(tiles_k+split_count-1)/split_count;
    const int tile_k_begin=split*tiles_per_split;
    const int tile_k_end=min(tile_k_begin+tiles_per_split,tiles_k);
    const std::uint32_t multiplier=static_cast<std::uint32_t>(*mul1);
    auto prefetch=[&](int tile_k,int stage) {
        if(tile_k>=tile_k_end)return;
        const std::size_t offset=(static_cast<std::size_t>(tile_k)*tiles_n+
            tile_base)*16u*6u;
        auto* destination=sh_raw+stage*raw_stage_half;
        const auto* source=trellis+offset;
        for(int chunk=thread;chunk<raw_stage_half/8;chunk+=512)
            exl3_cp_async_16(destination+chunk*8,source+chunk*8);
        if(thread<64) {
            const int row=thread/2;
            const int column=(thread%2)*8;
            const int source_column=(thread%2^((row>>2)&1))*8;
            half* next_a=sh_a+stage*512+row*16+column;
            if(row<rows)exl3_cp_async_16(next_a,transformed+row*input_features+
                tile_k*16+source_column);
            else for(int j=0;j<8;++j)next_a[j]=__float2half_rn(0.0f);
        }
        exl3_cp_async_commit();
    };
    prefetch(tile_k_begin,0);
    exl3_cp_async_wait();
    __syncthreads();
    Exl3FragC c[fragments_per_warp];
    #pragma unroll
    for(int n=0;n<fragments_per_warp;++n)
        #pragma unroll
        for(float& value:c[n].values)value=0.0f;
    int raw_stage=0;
    for(int tile_k=tile_k_begin;tile_k<tile_k_end;++tile_k) {
        prefetch(tile_k+1,1-raw_stage);
        if(group==0) {
            #pragma unroll
            for(int n2=0;n2<fragments_per_warp;n2+=2) {
                const int sub_n2=warp*(fragments_per_warp/2)+n2/2;
                const auto* packed=reinterpret_cast<const std::uint32_t*>(
                    sh_raw+raw_stage*raw_stage_half+sub_n2*16*6);
                Exl3FragB b0,b1;
                exl3_dq4_generic<6>(packed,lane<<3,b0,multiplier);
                exl3_dq4_generic<6>(packed,(lane<<3)+4,b1,multiplier);
                auto* values=reinterpret_cast<half2*>(
                    sh_decoded+sub_n2*256)+lane*4;
                values[0]=b0.values[0];values[1]=b0.values[1];
                values[2]=b1.values[0];values[3]=b1.values[1];
            }
        }
        __syncthreads();
        Exl3FragA a;
        const int r=(lane%8)+8*((lane/8)%2);
        const int base_c=lane/16;
        const int c_swizzled=base_c^((r>>2)&1);
        exl3_ldsm4(a,sh_a+raw_stage*512+group*256+r*16+c_swizzled*8);
        #pragma unroll
        for(int n2=0;n2<fragments_per_warp;n2+=2) {
            const int sub_n2=warp*(fragments_per_warp/2)+n2/2;
            const auto* values=reinterpret_cast<const half2*>(
                sh_decoded+sub_n2*256)+lane*4;
            Exl3FragB b0,b1;
            b0.values[0]=values[0];b0.values[1]=values[1];
            b1.values[0]=values[2];b1.values[1]=values[3];
            exl3_mma_m16n8k16(a,b0,c[n2]);
            exl3_mma_m16n8k16(a,b1,c[n2+1]);
        }
        if(tile_k+1<tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage=1-raw_stage;
    }
    const int n0=warp*fragments_per_warp;
    const int r0=lane/4,r1=r0+8,column=(lane%4)*2;
    const int group_row=group*16;
    const std::size_t stride=static_cast<std::size_t>(total_rows)*output_features;
    #pragma unroll
    for(int n=0;n<fragments_per_warp;++n) {
        const int col=tile_base*16+(n0+n)*8+column;
        if(group_row+r0<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+
                static_cast<std::size_t>(row_base+group_row+r0)*output_features+col;
            dst[0]=c[n].values[0];dst[1]=c[n].values[1];
        }
        if(group_row+r1<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+
                static_cast<std::size_t>(row_base+group_row+r1)*output_features+col;
            dst[0]=c[n].values[2];dst[1]=c[n].values[3];
        }
    }
}

// Two physical M16 groups share exact reconstructed B; original kernels remain unchanged.
template <int Bits>
__global__ __launch_bounds__(256,4) void exl3_prefill_rowpair_kernel(const std::uint16_t* transformed,
                                                 const std::uint16_t* trellis,
                                                 const std::int32_t* mul1,
                                                 float* accum,
                                                 int rows,
                                                 int input_features,
                                                 int output_features,
                                                 int split_count) {
    static_assert(Bits == 6 || Bits == 7,
                  "row-pair prefill is qualified only for K6/K7");
    const int total_rows = rows;
    const int row_base = static_cast<int>(blockIdx.y) * 32;
    rows = min(32, total_rows - row_base);
    transformed += static_cast<std::size_t>(row_base) * input_features;
    constexpr bool SingleSplit = false;
    constexpr int output_tiles_per_block = 16;
    constexpr int fragments_per_warp = output_tiles_per_block / 4;
    static_assert(output_tiles_per_block == 8 || output_tiles_per_block == 16 || output_tiles_per_block == 32,
                  "EXL3 output topology must use a proven tile width");
    static_assert(fragments_per_warp == 2 || fragments_per_warp == 4 || fragments_per_warp == 8,
                  "EXL3 output topology must map whole MMA fragments");
    constexpr int raw_stage_half = output_tiles_per_block * 16 * Bits;
    extern __shared__ half shared[];
    half* sh_a = shared;
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(sh_a + 1024);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    const int tiles_k = input_features / 16;
    const int tiles_n = output_features / 16;
    const int output_blocks = (tiles_n + output_tiles_per_block - 1) / output_tiles_per_block;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (SingleSplit ? block : block % output_blocks) * output_tiles_per_block;
    const int split = SingleSplit ? 0 : block / output_blocks;
    const int tiles_per_split = (tiles_k + split_count - 1) / split_count;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, tiles_k);
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tile_k_end) return;
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * tiles_n + tile_base) *
            static_cast<std::size_t>(16 * Bits);
        auto* destination = sh_raw + stage * raw_stage_half;
        const auto* source = trellis + offset;
        for (int chunk = thread; chunk < raw_stage_half / 8; chunk += kThreads) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        if(thread<64) {
            const int row=thread/2;
            const int column=(thread%2)*8;
            const int source_column=(thread%2 ^ ((row>>2)&1))*8;
            half* next_a=sh_a+stage*512+row*16+column;
            if(row<rows) exl3_cp_async_16(next_a,transformed+row*input_features+tile_k*16+source_column);
            else {
                #pragma unroll
                for(int j=0;j<8;++j) next_a[j]=__float2half_rn(0.0f);
            }
        }
        exl3_cp_async_commit();
    };

    prefetch(tile_k_begin, 0);
    exl3_cp_async_wait();
    __syncthreads();

    Exl3FragC c[2][fragments_per_warp];
    #pragma unroll
    for (int n = 0; n < fragments_per_warp; ++n) {
        #pragma unroll
        for (int group = 0; group < 2; ++group)
            for (float& value : c[group][n].values) value = 0.0f;
    }

    int raw_stage = 0;
    for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
        prefetch(tile_k + 1, 1 - raw_stage);

        Exl3FragA a0, a1;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a0, sh_a + raw_stage*512 + r * 16 + c_swizzled * 8);
        exl3_ldsm4(a1, sh_a + raw_stage*512 + 256 + r * 16 + c_swizzled * 8);

        #pragma unroll
        for (int n2 = 0; n2 < fragments_per_warp; n2 += 2) {
            const int sub_n2 = warp * (fragments_per_warp / 2) + n2 / 2;
            const auto* packed = reinterpret_cast<const std::uint32_t*>(
                sh_raw + raw_stage * raw_stage_half + sub_n2 * (16 * Bits));
            Exl3FragB b;
            exl3_dq4_generic<Bits>(packed, lane << 3, b, mul1_multiplier);
            exl3_mma_m16n8k16(a0, b, c[0][n2]);
            exl3_mma_m16n8k16(a1, b, c[1][n2]);
            exl3_dq4_generic<Bits>(packed, (lane << 3) + 4, b, mul1_multiplier);
            exl3_mma_m16n8k16(a0, b, c[0][n2 + 1]);
            exl3_mma_m16n8k16(a1, b, c[1][n2 + 1]);
        }

        if (tile_k + 1 < tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage = 1 - raw_stage;
    }

    const int n0=warp*fragments_per_warp;
    const int r0=lane/4, r1=r0+8, column=(lane%4)*2;
    const std::size_t stride=static_cast<std::size_t>(total_rows)*output_features;
    #pragma unroll
    for(int group=0;group<2;++group) {
      #pragma unroll
      for(int n=0;n<fragments_per_warp;++n) {
        const int col=tile_base*16+(n0+n)*8+column;
        if(group*16+r0<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+static_cast<std::size_t>(row_base+group*16+r0)*output_features+col;
            dst[0]=c[group][n].values[0];dst[1]=c[group][n].values[1];
        }
        if(group*16+r1<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+static_cast<std::size_t>(row_base+group*16+r1)*output_features+col;
            dst[0]=c[group][n].values[2];dst[1]=c[group][n].values[3];
        }
    }
    }
}

__global__ void exl3_generic_tile_kernel(const std::uint16_t* transformed,
                                          const std::uint16_t* trellis,
                                          const std::int32_t* mul1,
                                          float* accum,
                                          int rows,
                                          int input_features,
                                          int output_features,
                                          int bits) {
    __shared__ std::uint16_t packed[128];
    const int row = static_cast<int>(blockIdx.y);
    const int tile_n = static_cast<int>(blockIdx.x);
    const int thread = static_cast<int>(threadIdx.x);
    const int tiles_k = input_features / 16;
    const int tiles_n = output_features / 16;
    if (row >= rows || tile_n >= tiles_n) { return; }

    const std::uint32_t multiplier = static_cast<std::uint32_t>(*mul1);
    float result = 0.0f;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k) {
        const int packed_words = 16 * bits;
        const auto* source = trellis +
            (static_cast<std::size_t>(tile_k) * tiles_n + tile_n) * packed_words;
        for (int i = thread; i < packed_words; i += blockDim.x) {
            packed[i] = source[i];
        }
        __syncthreads();

        if (thread < 16) {
            const int column = thread;
            for (int r = 0; r < 16; ++r) {
                const int encoded = inverse_tensor_core_index(r, column);
                const auto state = decode_state_generic(packed, bits, encoded);
                const auto weight = decode_mul1_generic(state, multiplier);
                const auto input = transformed[row * input_features + tile_k * 16 + r];
                result += __half2float(__ushort_as_half(input)) *
                          __half2float(__ushort_as_half(weight));
            }
        }
        __syncthreads();
    }
    if (thread < 16) {
        accum[row * output_features + tile_n * 16 + thread] = result;
    }
}

// Same per-output FP32 accumulation order as generic_tile. Reconstruct each
// packed 16x16 weight tile once and reuse it across all sixteen input rows.
__global__ void exl3_gateup_m16_ordered_kernel(
    const std::uint16_t* transformed, const std::uint16_t* trellis,
    const std::int32_t* mul1, float* accum) {
    __shared__ std::uint16_t packed[96];
    __shared__ float decoded[256];
    const int thread = static_cast<int>(threadIdx.x);
    const int row = thread / 16;
    const int column = thread % 16;
    const int tile_n = static_cast<int>(blockIdx.x);
    const std::uint32_t multiplier = static_cast<std::uint32_t>(*mul1);
    float result = 0.0f;
    for (int tile_k = 0; tile_k < 320; ++tile_k) {
        if (thread < 96)
            packed[thread] = trellis[(static_cast<std::size_t>(tile_k) * 1088 + tile_n) * 96 + thread];
        __syncthreads();
        const int encoded = inverse_tensor_core_index(row, column);
        decoded[thread] = __half2float(__ushort_as_half(
            decode_mul1_generic(decode_state_generic(packed, 6, encoded), multiplier)));
        __syncthreads();
        for (int r = 0; r < 16; ++r) {
            const auto input = transformed[row * 5120 + tile_k * 16 + r];
            result += __half2float(__ushort_as_half(input)) * decoded[r * 16 + column];
        }
        __syncthreads();
    }
    accum[row * 17408 + tile_n * 16 + column] = result;
}

// Generic-tile FP32 order, with each decoded weight reused by all16 rows.
template<int Bits>
__global__ void exl3_initial16_ordered_kernel(
    const std::uint16_t* transformed, const std::uint16_t* trellis,
    const std::int32_t* mul1, float* accum, int input_features, int output_features) {
    __shared__ std::uint16_t packed[16 * Bits];
    __shared__ float decoded[256];
    const int thread = static_cast<int>(threadIdx.x);
    const int row = thread / 16;
    const int column = thread % 16;
    const int tile_n = static_cast<int>(blockIdx.x);
    const std::uint32_t multiplier = static_cast<std::uint32_t>(*mul1);
    float result = 0.0f;
    for (int tile_k = 0; tile_k < input_features / 16; ++tile_k) {
        if (thread < 16 * Bits)
            packed[thread] = trellis[(static_cast<std::size_t>(tile_k) * (output_features / 16) + tile_n) * (16 * Bits) + thread];
        __syncthreads();
        const int encoded = inverse_tensor_core_index(row, column);
        decoded[thread] = __half2float(__ushort_as_half(
            decode_mul1_generic(decode_state_generic(packed, Bits, encoded), multiplier)));
        __syncthreads();
        for (int r = 0; r < 16; ++r) {
            const auto input = transformed[row * input_features + tile_k * 16 + r];
            result += __half2float(__ushort_as_half(input)) * decoded[r * 16 + column];
        }
        __syncthreads();
    }
    accum[row * output_features + tile_n * 16 + column] = result;
}

__global__ void exl3_mul1_lookup_table_kernel(std::uint16_t* table,
                                              std::uint32_t multiplier) {
    const auto state = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (state < 65536u)
        table[state] = decode_mul1_generic(static_cast<std::uint16_t>(state), multiplier);
}

__global__ void exl3_mul1_discriminator_states_kernel(std::uint32_t* states,
                                                       std::size_t pairs) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairs) return;
    // Odd full-period affine maps cover every 16-bit state uniformly while
    // preventing adjacent lanes from collapsing to adjacent table addresses.
    const auto low = static_cast<std::uint16_t>((index * 40503u + 173u) & 0xffffu);
    const auto high = static_cast<std::uint16_t>((index * 62119u + 49157u) & 0xffffu);
    states[index] = static_cast<std::uint32_t>(low) |
                    (static_cast<std::uint32_t>(high) << 16u);
}

__global__ void exl3_mul1_discriminator_arithmetic_kernel(
    const std::uint32_t* states, std::uint32_t* output, std::size_t pairs,
    std::uint32_t multiplier) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairs) return;
    const auto packed = states[index];
    const auto low = decode_mul1_generic(static_cast<std::uint16_t>(packed), multiplier);
    const auto high = decode_mul1_generic(static_cast<std::uint16_t>(packed >> 16u), multiplier);
    output[index] = static_cast<std::uint32_t>(low) |
                    (static_cast<std::uint32_t>(high) << 16u);
}

__global__ void exl3_mul1_discriminator_lookup_kernel(
    const std::uint32_t* states, const std::uint16_t* table,
    std::uint32_t* output, std::size_t pairs) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pairs) return;
    const auto packed = states[index];
    const auto low = table[static_cast<std::uint16_t>(packed)];
    const auto high = table[static_cast<std::uint16_t>(packed >> 16u)];
    output[index] = static_cast<std::uint32_t>(low) |
                    (static_cast<std::uint32_t>(high) << 16u);
}

__global__ void exl3_mul1_discriminator_compare_kernel(
    const std::uint32_t* arithmetic, const std::uint32_t* lookup,
    std::size_t pairs, unsigned long long* mismatches) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < pairs && arithmetic[index] != lookup[index])
        atomicAdd(mismatches, 1ULL);
}

__global__ void exl3_split_plane_discriminator_init_accum_kernel(
    float* accum,std::size_t count) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count) {
        const int value=static_cast<int>((index*37u+11u)%257u)-128;
        accum[index]=static_cast<float>(value)*0.0009765625f;
    }
}

__global__ void exl3_split_plane_discriminator_init_scale_kernel(
    std::uint16_t* scale,std::size_t count) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count) {
        const float value=0.5f+static_cast<float>((index*13u+3u)%31u)*0.03125f;
        scale[index]=__half_as_ushort(__float2half_rn(value));
    }
}

__global__ void exl3_split_plane_discriminator_compare_kernel(
    const std::uint16_t* selected,const std::uint16_t* candidate,
    std::size_t count,unsigned long long* mismatches) {
    const auto index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index<count&&selected[index]!=candidate[index])atomicAdd(mismatches,1ULL);
}

} // namespace

Exl3Mul1LookupDiscriminator exl3_mul1_lookup_discriminator_for_test(
    std::uint32_t multiplier) {
    constexpr std::size_t pairs = 1u << 22u;
    constexpr int threads = 256;
    constexpr int samples = 31;
    const int blocks = static_cast<int>((pairs + threads - 1) / threads);
    std::uint32_t* states = nullptr;
    std::uint32_t* arithmetic = nullptr;
    std::uint32_t* lookup = nullptr;
    std::uint16_t* table = nullptr;
    unsigned long long* mismatches = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    auto cleanup = [&]() noexcept {
        if (end) (void)cudaEventDestroy(end);
        if (begin) (void)cudaEventDestroy(begin);
        if (mismatches) (void)cudaFree(mismatches);
        if (table) (void)cudaFree(table);
        if (lookup) (void)cudaFree(lookup);
        if (arithmetic) (void)cudaFree(arithmetic);
        if (states) (void)cudaFree(states);
    };
    try {
        cuda_check(cudaMalloc(&states, pairs * sizeof(*states)), "allocate MUL1 discriminator states");
        cuda_check(cudaMalloc(&arithmetic, pairs * sizeof(*arithmetic)), "allocate MUL1 arithmetic output");
        cuda_check(cudaMalloc(&lookup, pairs * sizeof(*lookup)), "allocate MUL1 lookup output");
        cuda_check(cudaMalloc(&table, 65536u * sizeof(*table)), "allocate MUL1 lookup table");
        cuda_check(cudaMalloc(&mismatches, sizeof(*mismatches)), "allocate MUL1 mismatch counter");
        cuda_check(cudaMemset(mismatches, 0, sizeof(*mismatches)), "clear MUL1 mismatch counter");
        cuda_check(cudaEventCreate(&begin), "create MUL1 begin event");
        cuda_check(cudaEventCreate(&end), "create MUL1 end event");
        exl3_mul1_discriminator_states_kernel<<<blocks, threads>>>(states, pairs);
        cuda_check(cudaGetLastError(), "initialize MUL1 discriminator states");

        cuda_check(cudaEventRecord(begin), "record MUL1 setup begin");
        exl3_mul1_lookup_table_kernel<<<256, 256>>>(table, multiplier);
        cuda_check(cudaGetLastError(), "build MUL1 lookup table");
        cuda_check(cudaEventRecord(end), "record MUL1 setup end");
        cuda_check(cudaEventSynchronize(end), "synchronize MUL1 lookup setup");
        float setup_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&setup_ms, begin, end), "time MUL1 lookup setup");

        for (int warmup = 0; warmup < 4; ++warmup) {
            exl3_mul1_discriminator_arithmetic_kernel<<<blocks, threads>>>(
                states, arithmetic, pairs, multiplier);
            exl3_mul1_discriminator_lookup_kernel<<<blocks, threads>>>(
                states, table, lookup, pairs);
        }
        cuda_check(cudaGetLastError(), "warm MUL1 discriminator kernels");
        cuda_check(cudaDeviceSynchronize(), "synchronize MUL1 discriminator warmup");

        std::vector<double> arithmetic_samples;
        std::vector<double> lookup_samples;
        arithmetic_samples.reserve(samples);
        lookup_samples.reserve(samples);
        const auto time_kernel = [&](bool use_lookup) {
            cuda_check(cudaEventRecord(begin), "record MUL1 sample begin");
            if (use_lookup)
                exl3_mul1_discriminator_lookup_kernel<<<blocks, threads>>>(
                    states, table, lookup, pairs);
            else
                exl3_mul1_discriminator_arithmetic_kernel<<<blocks, threads>>>(
                    states, arithmetic, pairs, multiplier);
            cuda_check(cudaGetLastError(), "launch MUL1 discriminator sample");
            cuda_check(cudaEventRecord(end), "record MUL1 sample end");
            cuda_check(cudaEventSynchronize(end), "synchronize MUL1 discriminator sample");
            float elapsed_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, begin, end), "time MUL1 discriminator sample");
            return static_cast<double>(elapsed_ms) * 1000.0;
        };
        for (int sample = 0; sample < samples; ++sample) {
            if ((sample & 1) == 0) {
                arithmetic_samples.push_back(time_kernel(false));
                lookup_samples.push_back(time_kernel(true));
            } else {
                lookup_samples.push_back(time_kernel(true));
                arithmetic_samples.push_back(time_kernel(false));
            }
        }

        exl3_mul1_discriminator_compare_kernel<<<blocks, threads>>>(
            arithmetic, lookup, pairs, mismatches);
        cuda_check(cudaGetLastError(), "compare MUL1 discriminator outputs");
        unsigned long long mismatch_count = 0;
        cuda_check(cudaMemcpy(&mismatch_count, mismatches, sizeof(mismatch_count),
                              cudaMemcpyDeviceToHost),
                   "read MUL1 mismatch counter");
        const auto median = [](std::vector<double>& values) {
            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        };
        Exl3Mul1LookupDiscriminator result;
        result.arithmetic_median_us = median(arithmetic_samples);
        result.lookup_median_us = median(lookup_samples);
        result.table_setup_us = static_cast<double>(setup_ms) * 1000.0;
        result.state_pairs = pairs;
        result.table_bytes = 65536u * sizeof(*table);
        result.multiplier = multiplier;
        result.mismatches = static_cast<std::uint64_t>(mismatch_count);
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

Exl3SplitPlanePrefetchDiscriminator
exl3_split_plane_prefetch_discriminator_for_test() {
    constexpr int rows=1024;
    constexpr int output_features=17408;
    constexpr int split_count=5;
    constexpr int samples=31;
    constexpr std::size_t guard=128;
    constexpr std::size_t output_count=
        static_cast<std::size_t>(rows)*output_features;
    constexpr std::size_t accum_count=output_count*split_count;
    float* accum=nullptr;
    std::uint16_t* scale=nullptr;
    std::uint16_t* selected_storage=nullptr;
    std::uint16_t* prefetched_storage=nullptr;
    unsigned long long* mismatches=nullptr;
    cudaEvent_t begin=nullptr,end=nullptr;
    auto cleanup=[&]() noexcept {
        if(end)(void)cudaEventDestroy(end);
        if(begin)(void)cudaEventDestroy(begin);
        if(mismatches)(void)cudaFree(mismatches);
        if(prefetched_storage)(void)cudaFree(prefetched_storage);
        if(selected_storage)(void)cudaFree(selected_storage);
        if(scale)(void)cudaFree(scale);
        if(accum)(void)cudaFree(accum);
    };
    try {
        cuda_check(cudaMalloc(&accum,accum_count*sizeof(*accum)),
            "allocate split-plane discriminator accumulation");
        cuda_check(cudaMalloc(&scale,output_features*sizeof(*scale)),
            "allocate split-plane discriminator scale");
        cuda_check(cudaMalloc(&selected_storage,
            (output_count+2*guard)*sizeof(*selected_storage)),
            "allocate selected split-plane output");
        cuda_check(cudaMalloc(&prefetched_storage,
            (output_count+2*guard)*sizeof(*prefetched_storage)),
            "allocate prefetched split-plane output");
        cuda_check(cudaMalloc(&mismatches,sizeof(*mismatches)),
            "allocate split-plane mismatch counter");
        cuda_check(cudaEventCreate(&begin),"create split-plane begin event");
        cuda_check(cudaEventCreate(&end),"create split-plane end event");
        exl3_split_plane_discriminator_init_accum_kernel<<<
            static_cast<int>((accum_count+255)/256),256>>>(accum,accum_count);
        exl3_split_plane_discriminator_init_scale_kernel<<<
            (output_features+255)/256,256>>>(scale,output_features);
        cuda_check(cudaMemset(selected_storage,0x55,
            (output_count+2*guard)*sizeof(*selected_storage)),
            "initialize selected split-plane guards");
        cuda_check(cudaMemset(prefetched_storage,0x55,
            (output_count+2*guard)*sizeof(*prefetched_storage)),
            "initialize prefetched split-plane guards");
        auto* selected=selected_storage+guard;
        auto* prefetched=prefetched_storage+guard;
        const dim3 grid(rows,output_features/kHadamard);
        const auto launch=[&](bool candidate) {
            if(candidate)
                launch_prefill_reduce_output<true,false,true>(0,
                    accum,scale,prefetched,rows,output_features,split_count);
            else
                launch_prefill_reduce_output<true,false,false>(0,
                    accum,scale,selected,rows,output_features,split_count);
        };
        for(int warmup=0;warmup<4;++warmup){launch(false);launch(true);}
        cuda_check(cudaGetLastError(),"warm split-plane discriminator");
        cuda_check(cudaDeviceSynchronize(),"synchronize split-plane warmup");
        const auto time_stage=[&](bool candidate) {
            cuda_check(cudaEventRecord(begin),"record split-plane begin");
            launch(candidate);
            cuda_check(cudaGetLastError(),"launch split-plane sample");
            cuda_check(cudaEventRecord(end),"record split-plane end");
            cuda_check(cudaEventSynchronize(end),"synchronize split-plane sample");
            float elapsed_ms=0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms,begin,end),
                "time split-plane sample");
            return static_cast<double>(elapsed_ms)*1000.0;
        };
        std::vector<double> selected_samples,prefetched_samples;
        selected_samples.reserve(samples);prefetched_samples.reserve(samples);
        for(int sample=0;sample<samples;++sample) {
            if((sample&1)==0) {
                selected_samples.push_back(time_stage(false));
                prefetched_samples.push_back(time_stage(true));
            } else {
                prefetched_samples.push_back(time_stage(true));
                selected_samples.push_back(time_stage(false));
            }
        }
        cuda_check(cudaMemset(mismatches,0,sizeof(*mismatches)),
            "clear split-plane mismatch counter");
        exl3_split_plane_discriminator_compare_kernel<<<
            static_cast<int>(((output_count+2*guard)+255)/256),256>>>(
                selected_storage,prefetched_storage,output_count+2*guard,mismatches);
        cuda_check(cudaGetLastError(),"compare split-plane discriminator");
        unsigned long long mismatch_count=0;
        cuda_check(cudaMemcpy(&mismatch_count,mismatches,sizeof(mismatch_count),
            cudaMemcpyDeviceToHost),"read split-plane mismatch counter");
        const auto median=[](std::vector<double>& values) {
            std::sort(values.begin(),values.end());return values[values.size()/2];
        };
        Exl3SplitPlanePrefetchDiscriminator result;
        result.selected_median_us=median(selected_samples);
        result.prefetched_median_us=median(prefetched_samples);
        result.rows=rows;result.output_features=output_features;
        result.split_count=split_count;
        result.accumulation_bytes=accum_count*sizeof(*accum);
        result.mismatches=static_cast<std::uint64_t>(mismatch_count);
        cleanup();return result;
    } catch(...) {cleanup();throw;}
}

void exl3_reconstruct_native_tile_for_test(const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,int input_tile,int output_tile,
    std::uint16_t* output,cudaStream_t stream) {
    if(!weights.trellis || !weights.mul1 || !output ||
        (metadata.K!=6 && metadata.K!=7) || metadata.in_features!=17408 ||
        metadata.out_features!=5120 || metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        input_tile<0 || input_tile>=metadata.in_features/16 ||
        output_tile<0 || output_tile>=metadata.out_features/16)
        throw std::invalid_argument("native fragment observation extent");
    const auto* row=weights.trellis+std::size_t(input_tile)*(metadata.out_features/16)*(16*metadata.K);
    if(metadata.K==6)
        exl3_reconstruct_transformed_weight_kernel<6,true><<<1,256,0,stream>>>(
            row,weights.mul1,output,16,metadata.out_features,output_tile,1);
    else
        exl3_reconstruct_transformed_weight_kernel<7,true><<<1,256,0,stream>>>(
            row,weights.mul1,output,16,metadata.out_features,output_tile,1);
    cuda_check(cudaGetLastError(),"observe native reconstruction fragment");
}

void exl3_reconstruct_native_weights_for_test(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,std::uint16_t* output,
    std::size_t output_bytes,cudaStream_t stream) {
    const std::size_t required=static_cast<std::size_t>(metadata.in_features)*
        metadata.out_features*sizeof(std::uint16_t);
    if(!weights.trellis||!weights.mul1||!output||metadata.K!=6||
       metadata.in_features<=0||metadata.out_features<=0||
       metadata.in_features%16!=0||metadata.out_features%16!=0||
       metadata.mcg||!metadata.mul1||metadata.has_bias||output_bytes!=required)
        throw std::invalid_argument("native full-owner reconstruction extent");
    exl3_reconstruct_transformed_weight_kernel<6,true><<<
        dim3(metadata.out_features/16,metadata.in_features/16),256,0,stream>>>(
            weights.trellis,weights.mul1,output,metadata.in_features,
            metadata.out_features);
    cuda_check(cudaGetLastError(),"materialize native full-owner reconstruction");
}

// Exact M1 discriminator for target-owned K6 continuation projections. One
// thread owns one real output column, so the kernel performs no padded rows.
// It retains the parent's five ascending K ranges and writes the same FP32
// partial layout consumed by prefill_reduce_output_kernel. The direct oracle
// decides whether its chronological scalar dot product is bit-identical to the
// selected packed-MMA route before any Engine use is permitted.
template<int Bits>
__global__ void exl3_target_m1_simt_partials_kernel(
    const std::uint16_t* transformed,const std::uint16_t* trellis,
    const std::int32_t* mul1,float* accum,int input_features,
    int output_features,int split_count) {
    static_assert(Bits==6,"target M1 SIMT probe is K6-only");
    constexpr int kOutputTilesPerBlock=16;
    constexpr int kOutputsPerBlock=kOutputTilesPerBlock*16;
    constexpr int kPackedWordsPerTile=16*Bits;
    __shared__ std::uint16_t input_tile[16];
    __shared__ std::uint16_t packed[kOutputTilesPerBlock*kPackedWordsPerTile];
    const int thread=static_cast<int>(threadIdx.x);
    const int output_block=static_cast<int>(blockIdx.x)%
        (output_features/kOutputsPerBlock);
    const int split=static_cast<int>(blockIdx.x)/
        (output_features/kOutputsPerBlock);
    const int tiles_k=input_features/16,tiles_n=output_features/16;
    const int tiles_per_split=(tiles_k+split_count-1)/split_count;
    const int tile_k_begin=split*tiles_per_split;
    const int tile_k_end=min(tile_k_begin+tiles_per_split,tiles_k);
    const int local_tile=thread/16,column=thread%16;
    const int tile_base=output_block*kOutputTilesPerBlock;
    const std::uint32_t multiplier=static_cast<std::uint32_t>(*mul1);
    float result=0.0f;
    for(int tile_k=tile_k_begin;tile_k<tile_k_end;++tile_k) {
        if(thread<16)input_tile[thread]=transformed[tile_k*16+thread];
        const auto* source=trellis+
            (static_cast<std::size_t>(tile_k)*tiles_n+tile_base)*
                kPackedWordsPerTile;
        #pragma unroll
        for(int i=thread;i<kOutputTilesPerBlock*kPackedWordsPerTile;
            i+=kOutputsPerBlock)packed[i]=source[i];
        __syncthreads();
        const auto* local_packed=packed+local_tile*kPackedWordsPerTile;
        #pragma unroll
        for(int r=0;r<16;++r) {
            const int encoded=inverse_tensor_core_index(r,column);
            const auto weight=decode_mul1_generic(
                decode_state_generic(local_packed,Bits,encoded),multiplier);
            result+=__half2float(__ushort_as_half(input_tile[r]))*
                __half2float(__ushort_as_half(weight));
        }
        __syncthreads();
    }
    const std::size_t stride=static_cast<std::size_t>(output_features);
    accum[static_cast<std::size_t>(split)*stride+
        output_block*kOutputsPerBlock+thread]=result;
}

void exl3_transform_input_pair(
    const Exl3CudaLinearWeights& first_weights,
    const Exl3CudaLinearMetadata& first_metadata,
    const Exl3CudaLinearWeights& second_weights,
    const Exl3CudaLinearMetadata& second_metadata,
    const std::uint16_t* input,
    std::uint16_t* first_transformed,
    std::uint16_t* second_transformed,
    int rows,
    cudaStream_t stream) {
    if (first_metadata.in_features <= 0 ||
        first_metadata.in_features != second_metadata.in_features ||
        first_metadata.in_features % kHadamard != 0) {
        throw std::invalid_argument("paired EXL3 input transform dimensions");
    }
    if (rows <= 0 || !input || !first_weights.suh || !second_weights.suh ||
        !first_transformed || !second_transformed ||
        first_transformed == second_transformed) {
        throw std::invalid_argument("paired EXL3 input transform buffers");
    }
    exl3_require_paired_transform_extents(rows,first_metadata.in_features,
        {reinterpret_cast<std::uintptr_t>(input),reinterpret_cast<std::uintptr_t>(first_weights.suh),
         reinterpret_cast<std::uintptr_t>(second_weights.suh),reinterpret_cast<std::uintptr_t>(first_transformed),
         reinterpret_cast<std::uintptr_t>(second_transformed)});
    if(exl3_equal_transform_reuse_enabled(std::getenv("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS")))
        input_hadamard_pair_kernel<kHadamard,true><<<
            dim3(rows, first_metadata.in_features / kHadamard),dim3(kHadamard),0,stream>>>(
                input,first_weights.suh,second_weights.suh,first_transformed,second_transformed,rows,first_metadata.in_features);
    else input_hadamard_pair_kernel<kHadamard><<<
        dim3(rows, first_metadata.in_features / kHadamard),
        dim3(kHadamard), 0, stream>>>(
            input, first_weights.suh, second_weights.suh,
            first_transformed, second_transformed, rows,
            first_metadata.in_features);
    cuda_check(cudaGetLastError(), "launch paired EXL3 input Hadamard");
}

// The wide-prefill candidate owns only a cuBLAS handle. Its reconstructed
// FP16 weights come from the context-owned exact reconstruction slab, while
// the established native path remains the fail-closed fallback.
struct Exl3CudaLinearWorkspace::FastWideGemmState {
    cublasHandle_t handle = nullptr;

    ~FastWideGemmState() noexcept {
        if (handle) {
            (void)cublasDestroy(handle);
            handle = nullptr;
        }
    }
};

Exl3CudaLinearWorkspace::Exl3CudaLinearWorkspace(int max_rows)
    : Exl3CudaLinearWorkspace(kInput, kOutput, max_rows) {
    // Preserve the E2B one-argument API's fail-closed K=5 contract.  E3A uses
    // the explicit-shape constructor for the newly audited K=6/K=8 families.
    allow_generic_variants_ = false;
}

Exl3CudaLinearWorkspace::Exl3CudaLinearWorkspace(int in_features,
                                                 int out_features,
                                                 int max_rows,
                                                 bool default_enable_draft_small_m,
                                                 bool target_gateup_owner,
                                                 bool target_down_owner,
                                                 bool target_o_k7_owner,
                                                 bool target_z_k6_owner,
                                                 bool target_wide_prefill_owner,
                                                 bool draft_prefill_fc_owner,
                                                 Exl3CudaAccumulationView accumulation,
                                                 Exl3CudaTransformView transformed,
                                                 bool target_qkv_k6_owner,
                                                 bool target_kv_owner,
                                                 bool target_q_k6_owner,
                                                 std::optional<RetainedDeviceLedger::Ticket> constructor_device_credit,
                                                 std::optional<RetainedDescriptorLedger::Ticket> constructor_metadata_credit)
    : in_features_(in_features), out_features_(out_features), max_rows_(max_rows),
      specialized_shape_(in_features == kInput && out_features == kOutput) {
    const auto requirement=Exl3LinearWorkspaceRequirements::derive(
        in_features,out_features,max_rows,transformed.data!=nullptr,accumulation.data!=nullptr);
    if((accumulation.data==nullptr && accumulation.bytes!=0) ||
       (accumulation.data!=nullptr && accumulation.bytes<requirement.accumulation_bytes))
        throw std::invalid_argument("EXL3 borrowed accumulation capacity");
    if((transformed.data==nullptr && transformed.bytes!=0) ||
       (transformed.data!=nullptr && transformed.bytes<requirement.transformed_bytes))
        throw std::invalid_argument("EXL3 borrowed transform capacity");
    requirement.require_disjoint_borrowed_views(transformed.data,accumulation.data);
    const bool stream_reduce = read_binary_option("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION",
        "K6 small-M stream reduction must be0 or1");
    k6_stream_reduction_enabled_ = target_wide_prefill_owner && stream_reduce;
    const char* extended=std::getenv("NINFER_EXL3_EXTENDED_STREAM_REDUCTION");
    if(extended&&std::strcmp(extended,"0")!=0&&std::strcmp(extended,"1")!=0)
        throw std::invalid_argument("extended target stream reduction must be0 or1");
    extended_stream_reduction_enabled_=target_wide_prefill_owner&&extended&&std::strcmp(extended,"1")==0;
    const char* host_kv_gdn_segment_graphs=
        std::getenv("NINFER_EXL3_HOST_KV_GDN_SEGMENT_GRAPHS");
    const char* host_kv_mlp_tail_graphs=
        std::getenv("NINFER_EXL3_HOST_KV_MLP_TAIL_GRAPHS");
    host_kv_gdn_segment_graph_capture_=
        (host_kv_gdn_segment_graphs &&
         std::strcmp(host_kv_gdn_segment_graphs,"1")==0) ||
        (host_kv_mlp_tail_graphs &&
         std::strcmp(host_kv_mlp_tail_graphs,"1")==0);
    const bool kv_stream = read_binary_option("NINFER_EXL3_KV_K7_STREAM_REDUCTION",
        "K7 KV stream reduction must be 0 or 1");
    kv_k7_stream_reduction_enabled_ = target_kv_owner && kv_stream;
    // Widen only explicit staged target prefill, with construction-latched opt-in.
    const char* wide64 = std::getenv("NINFER_EXL3_PREFILL_WIDE64");
    target_wide64_enabled_ = wide64 && std::strcmp(wide64, "1") == 0;
    const char* wide128 = std::getenv("NINFER_EXL3_PREFILL_WIDE128");
    target_wide128_enabled_ = target_wide64_enabled_ && wide128 && std::strcmp(wide128, "1") == 0;
    const char* wide256 = std::getenv("NINFER_EXL3_PREFILL_WIDE256");
    target_wide256_enabled_ = target_wide128_enabled_ && wide256 && std::strcmp(wide256, "1") == 0;
    const char* wide512 = std::getenv("NINFER_EXL3_PREFILL_WIDE512");
    target_wide512_enabled_ = target_wide256_enabled_ && wide512 && std::strcmp(wide512,"1")==0;
    const char* wide1024 = std::getenv("NINFER_EXL3_PREFILL_WIDE1024");
    target_wide1024_enabled_ = target_wide512_enabled_ && wide1024 && std::strcmp(wide1024,"1")==0;
    const char* projection_graphs =
        std::getenv("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS");
    if (projection_graphs && std::strcmp(projection_graphs,"0") != 0 &&
        std::strcmp(projection_graphs,"1") != 0)
        throw std::invalid_argument(
            "NINFER_EXL3_PREFILL_PROJECTION_GRAPHS must be 0 or 1");
    // T139/W13: capture only workspaces owned by the exact target-wide caller.
    // The graph records the already-selected transform/projection/reduction
    // kernels; it never selects an alternate arithmetic route.
    prefill_projection_graph_enabled_ = target_wide_prefill_owner &&
        projection_graphs && std::strcmp(projection_graphs,"1") == 0;
    const bool persisting_l2 = read_binary_option("NINFER_EXL3_PREFILL_PERSISTING_L2",
        "NINFER_EXL3_PREFILL_PERSISTING_L2 must be 0 or 1");
    prefill_persisting_l2_enabled_ = target_wide_prefill_owner &&
        persisting_l2;
    if(prefill_persisting_l2_enabled_) {
        int device=0,maximum=0;
        if(cudaGetDevice(&device)==cudaSuccess &&
           cudaDeviceGetAttribute(&maximum,
               cudaDevAttrMaxPersistingL2CacheSize,device)==cudaSuccess &&
           maximum>0)
            prefill_persisting_l2_set_aside_bytes_=
                configure_prefill_persisting_l2(device,
                    static_cast<std::size_t>(maximum));
        else (void)cudaGetLastError();
    }
    // Qualified H6 small-M remains enabled by default; zero selects comparison.
    const char* h6_small_m = std::getenv("NINFER_EXL3_H6_SMALL_M");
    h6_small_m_enabled_ = h6_small_m == nullptr || std::string(h6_small_m) != "0";
    // Draft-owned callers may select the qualified default while generic/target
    // workspaces remain off. The exact environment override is construction-latched.
    const char* draft_small_m = std::getenv("NINFER_EXL3_DRAFT_SMALL_M");
    draft_small_m_enabled_ = draft_small_m == nullptr
        ? default_enable_draft_small_m
        : std::string(draft_small_m) == "1";
    const char* draft_k5_async_a =
        std::getenv("NINFER_EXL3_DRAFT_K5_ASYNC_A");
    if (draft_k5_async_a &&
        std::strcmp(draft_k5_async_a, "0") != 0 &&
        std::strcmp(draft_k5_async_a, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_DRAFT_K5_ASYNC_A must be 0 or 1");
    }
    draft_k5_async_a_enabled_ = draft_small_m_enabled_ &&
        draft_k5_async_a && std::strcmp(draft_k5_async_a, "1") == 0;
    // Target gate/up admission is deliberately opt-in and target-owned.  The
    // exact environment value is latched once, but cannot change dispatch
    // unless a continuation gate/up call explicitly admits the candidate.
    const char* draft_prefill = std::getenv("NINFER_DFLASH2_PREFILL_BATCH");
    draft_prefill_fc_enabled_ = draft_prefill_fc_owner && draft_prefill && std::string(draft_prefill) == "1";
    const char* staged_prefill = std::getenv("NINFER_EXL3_PREFILL_STAGED_REDUCTION");
    target_staged_prefill_enabled_ = target_wide_prefill_owner && staged_prefill != nullptr &&
        std::string(staged_prefill) == "1";
    const char* target_wide_prefill = std::getenv("NINFER_EXL3_WIDE_PREFILL");
    target_wide_prefill_enabled_ = target_wide_prefill_owner &&
        target_wide_prefill != nullptr && std::string(target_wide_prefill) == "1";
    const char* fast_wide_prefill_gemm =
        std::getenv("NINFER_EXL3_FAST_MIA_PARITY_WIDE_PREFILL_GEMM");
    if (fast_wide_prefill_gemm &&
        std::strcmp(fast_wide_prefill_gemm, "0") != 0 &&
        std::strcmp(fast_wide_prefill_gemm, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_MIA_PARITY_WIDE_PREFILL_GEMM must be 0 or 1");
    }
    // The candidate is construction-latched and still requires the existing
    // target-wide owner and staged-wide admission.
    fast_wide_prefill_gemm_enabled_ = target_wide_prefill_owner &&
        target_wide_prefill_enabled_ && fast_wide_prefill_gemm &&
        std::strcmp(fast_wide_prefill_gemm, "1") == 0;
    const bool qualified_default_group = exl3_prefill_qualified_default_group(
        max_rows_, target_wide_prefill_enabled_, target_staged_prefill_enabled_,
        accumulation.data != nullptr);
    const char* initial16 = std::getenv("NINFER_EXL3_TARGET_INITIAL16_ORDERED");
    target_initial16_enabled_ = target_wide_prefill_owner &&
        (initial16 ? std::string(initial16) == "1" : qualified_default_group);
    const char* target_gateup_m16 = std::getenv("NINFER_EXL3_TARGET_GATEUP_M16");
    target_gateup_m16_enabled_ = target_gateup_owner &&
        (target_gateup_m16 == nullptr || std::string(target_gateup_m16) != "0");
    const char* target_gateup_k5 = std::getenv("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M");
    target_gateup_k5_small_m_enabled_ = target_gateup_owner && target_gateup_k5 &&
        std::strcmp(target_gateup_k5, "1") == 0;
    const char* target_k5_small_m_batch =
        std::getenv("NINFER_EXL3_TARGET_K5_SMALL_M_BATCH");
    if (target_k5_small_m_batch &&
        std::strcmp(target_k5_small_m_batch, "0") != 0 &&
        std::strcmp(target_k5_small_m_batch, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_K5_SMALL_M_BATCH must be 0 or 1");
    }
    const char* target_gateup_small_m =
        std::getenv("NINFER_EXL3_TARGET_GATEUP_SMALL_M");
    target_gateup_small_m_enabled_ = target_gateup_owner &&
        (target_gateup_small_m == nullptr || std::string(target_gateup_small_m) != "0");
    const char* target_gateup_k6_async_a =
        std::getenv("NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A");
    if (target_gateup_k6_async_a &&
        std::strcmp(target_gateup_k6_async_a, "0") != 0 &&
        std::strcmp(target_gateup_k6_async_a, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_GATEUP_K6_ASYNC_A must be 0 or 1");
    }
    target_gateup_k6_async_a_enabled_ =
        target_gateup_owner && target_gateup_k6_async_a &&
        std::strcmp(target_gateup_k6_async_a, "1") == 0;
    const char* target_gateup_k6_n16 =
        std::getenv("NINFER_EXL3_TARGET_GATEUP_K6_N16");
    if (target_gateup_k6_n16 &&
        std::strcmp(target_gateup_k6_n16, "0") != 0 &&
        std::strcmp(target_gateup_k6_n16, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_GATEUP_K6_N16 must be 0 or 1");
    }
    target_gateup_k6_n16_enabled_ = target_gateup_owner &&
        target_gateup_k6_n16 &&
        std::strcmp(target_gateup_k6_n16, "1") == 0;
    const char* target_gateup_k7_small_m =
        std::getenv("NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M");
    target_gateup_k7_small_m_enabled_ = target_gateup_owner &&
        target_gateup_k7_small_m &&
        std::strcmp(target_gateup_k7_small_m, "1") == 0;
    const char* target_down_small_m =
        std::getenv("NINFER_EXL3_TARGET_DOWN_SMALL_M");
    target_down_small_m_enabled_ = target_down_owner &&
        (target_down_small_m == nullptr || std::string(target_down_small_m) != "0");
    const char* target_down_k7_small_m =
        std::getenv("NINFER_EXL3_TARGET_DOWN_K7_SMALL_M");
    target_down_k7_small_m_enabled_ = target_down_owner &&
        (target_down_k7_small_m == nullptr || std::string(target_down_k7_small_m) != "0");
    const char* target_down_k6_async_a =
        std::getenv("NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A");
    if (target_down_k6_async_a &&
        std::strcmp(target_down_k6_async_a, "0") != 0 &&
        std::strcmp(target_down_k6_async_a, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_DOWN_K6_ASYNC_A must be 0 or 1");
    }
    target_down_k6_async_a_enabled_ = target_down_owner &&
        target_down_k6_async_a &&
        std::strcmp(target_down_k6_async_a, "1") == 0;
    const char* target_down_k7_async_a =
        std::getenv("NINFER_EXL3_TARGET_DOWN_K7_ASYNC_A");
    if (target_down_k7_async_a &&
        std::strcmp(target_down_k7_async_a, "0") != 0 &&
        std::strcmp(target_down_k7_async_a, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_DOWN_K7_ASYNC_A must be 0 or 1");
    }
    target_down_k7_async_a_enabled_ = target_down_owner &&
        target_down_k7_async_a &&
        std::strcmp(target_down_k7_async_a, "1") == 0;
    const char* target_m1_k6_n32_async_a =
        std::getenv("NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A");
    if (target_m1_k6_n32_async_a &&
        std::strcmp(target_m1_k6_n32_async_a, "0") != 0 &&
        std::strcmp(target_m1_k6_n32_async_a, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_M1_K6_N32_ASYNC_A must be 0 or 1");
    }
    const bool target_m1_owner = target_gateup_owner || target_down_owner ||
        target_o_k7_owner || target_z_k6_owner || target_wide_prefill_owner ||
        target_qkv_k6_owner || target_kv_owner || target_q_k6_owner;
    const char* fast_same_weights_fp16_accum =
        std::getenv("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_ACCUM");
    if (fast_same_weights_fp16_accum &&
        std::strcmp(fast_same_weights_fp16_accum, "0") != 0 &&
        std::strcmp(fast_same_weights_fp16_accum, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_ACCUM must be 0 or 1");
    }
    fast_same_weights_fp16_accum_enabled_ = target_m1_owner &&
        fast_same_weights_fp16_accum &&
        std::strcmp(fast_same_weights_fp16_accum, "1") == 0;
    const bool fast_same_weights_fp16_m1 = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1 must be 0 or 1");
    fast_same_weights_fp16_m1_enabled_ = target_m1_owner &&
        fast_same_weights_fp16_m1;
    fast_fp16_m2_8_down_enabled_ = target_down_owner &&
        read_binary_option("NINFER_EXL3_FAST_FP16_M2_8_DOWN_K6",
            "fast FP16 M2-8 down K6 must be 0 or 1");
    fast_fp16_m2_8_fused_down_enabled_ = target_down_owner &&
        read_binary_option("NINFER_EXL3_FAST_FP16_M2_8_FUSED_DOWN_K6",
            "fast FP16 M2-8 fused down K6 must be 0 or 1");
    const bool coherent_wide_k6_requested =
        read_binary_option("NINFER_EXL3_TARGET_COHERENT_WIDE_K6",
            "coherent wide K6 must be 0 or 1");
    coherent_wide_k6_operation_ =
        target_q_k6_owner && in_features_ == 5120 && out_features_ == 12288 ? 0 :
        target_qkv_k6_owner && in_features_ == 5120 && out_features_ == 10240 ? 1 :
        target_z_k6_owner && in_features_ == 5120 && out_features_ == 6144 ? 2 :
        target_o_k7_owner && in_features_ == 6144 && out_features_ == 5120 ? 3 :
        target_gateup_owner && in_features_ == 5120 && out_features_ == 17408 ? 4 : -1;
    coherent_wide_k6_enabled_ = coherent_wide_k6_requested &&
        coherent_wide_k6_operation_ >= 0;
    // Split10 is the quality-gated default (teacher-forced verify NLL within
    // +/-0.00012 nats/token); 0 restores the 5-split reduction order.
    coherent_wide_k6_split10_enabled_ = coherent_wide_k6_enabled_ &&
        (!std::getenv("NINFER_EXL3_TARGET_COHERENT_WIDE_K6_SPLIT10") ||
         read_binary_option("NINFER_EXL3_TARGET_COHERENT_WIDE_K6_SPLIT10",
            "coherent wide K6 split10 must be 0 or 1"));
    coherent_down_k6_enabled_ = target_down_owner &&
        read_binary_option("NINFER_EXL3_COHERENT_DOWN_K6",
            "coherent K6 down must be 0 or 1");
    coherent_down_k7_enabled_ = target_down_owner &&
        read_binary_option("NINFER_EXL3_COHERENT_DOWN_K7",
            "coherent K7 down must be 0 or 1");
    coherent_o_k7_enabled_ = target_o_k7_owner &&
        read_binary_option("NINFER_EXL3_COHERENT_O_K7",
            "coherent K7 O must be 0 or 1");
    fast_fp16_m2_8_all_enabled_ = target_m1_owner &&
        read_binary_option("NINFER_EXL3_FAST_FP16_M2_8",
            "fast FP16 M2-8 must be 0 or 1");
    fast_fp16_m2_8_async_a_enabled_ = target_m1_owner &&
        read_binary_option("NINFER_EXL3_FAST_FP16_M2_8_ASYNC_A",
            "fast FP16 M2-8 async A must be 0 or 1");
    const bool fast_same_weights_fp16_m1_n16 = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_N16",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_N16 must be 0 or 1");
    fast_same_weights_fp16_m1_n16_enabled_ = target_m1_owner &&
        fast_same_weights_fp16_m1_enabled_ && fast_same_weights_fp16_m1_n16;
    const char* fast_same_weights_fp16_m1_k6_register_pipeline =
        std::getenv("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_K6_REGISTER_PIPELINE");
    if (fast_same_weights_fp16_m1_k6_register_pipeline &&
        std::strcmp(fast_same_weights_fp16_m1_k6_register_pipeline, "0") != 0 &&
        std::strcmp(fast_same_weights_fp16_m1_k6_register_pipeline, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_K6_REGISTER_PIPELINE must be 0 or 1");
    }
    fast_same_weights_fp16_m1_k6_register_pipeline_enabled_ = target_m1_owner &&
        fast_same_weights_fp16_m1_enabled_ &&
        fast_same_weights_fp16_m1_k6_register_pipeline &&
        std::strcmp(fast_same_weights_fp16_m1_k6_register_pipeline, "1") == 0;
    const char* fast_native_persistent_m1 =
        std::getenv("NINFER_EXL3_FAST_NATIVE_PERSISTENT_M1");
    if (fast_native_persistent_m1 &&
        std::strcmp(fast_native_persistent_m1, "0") != 0 &&
        std::strcmp(fast_native_persistent_m1, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_NATIVE_PERSISTENT_M1 must be 0 or 1");
    }
    // This is an independently gated native persistent-grid differential.  It
    // uses the vendored EXL3/PTX primitives in this binary and never delegates
    // execution to the Mia comparator or changes the quantized weights.
    fast_native_persistent_m1_enabled_ = target_m1_owner &&
        fast_native_persistent_m1 &&
        std::strcmp(fast_native_persistent_m1, "1") == 0;
    const char* fast_native_mia_m1_fp16 =
        std::getenv("NINFER_EXL3_FAST_NATIVE_MIA_M1_FP16");
    if (fast_native_mia_m1_fp16 &&
        std::strcmp(fast_native_mia_m1_fp16, "0") != 0 &&
        std::strcmp(fast_native_mia_m1_fp16, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_NATIVE_MIA_M1_FP16 must be 0 or 1");
    }
    // Independently labeled same-weight FP16-output differential.  It uses
    // the vendored Mia-shaped cooperative inner directly and is never implied
    // by the historical FP32 persistent adapter.
    fast_native_mia_m1_fp16_enabled_ = target_m1_owner &&
        fast_native_mia_m1_fp16 &&
        std::strcmp(fast_native_mia_m1_fp16, "1") == 0;
    const char* fast_native_mia_m1_fp16_fused_input =
        std::getenv("NINFER_EXL3_FAST_NATIVE_MIA_M1_FP16_FUSED_INPUT");
    if (fast_native_mia_m1_fp16_fused_input &&
        std::strcmp(fast_native_mia_m1_fp16_fused_input, "0") != 0 &&
        std::strcmp(fast_native_mia_m1_fp16_fused_input, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_NATIVE_MIA_M1_FP16_FUSED_INPUT must be 0 or 1");
    }
    // This is a separately labeled native differential: it fuses the existing
    // native-semantics input Hadamard into the same cooperative grid as the
    // vendored inner, while retaining the ordinary route as the default.
    fast_native_mia_m1_fp16_fused_input_enabled_ =
        fast_native_mia_m1_fp16_enabled_ &&
        fast_native_mia_m1_fp16_fused_input &&
        std::strcmp(fast_native_mia_m1_fp16_fused_input, "1") == 0;
    const char* fast_native_persistent_prefill =
        std::getenv("NINFER_EXL3_FAST_NATIVE_PERSISTENT_PREFILL");
    if (fast_native_persistent_prefill &&
        std::strcmp(fast_native_persistent_prefill, "0") != 0 &&
        std::strcmp(fast_native_persistent_prefill, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_NATIVE_PERSISTENT_PREFILL must be 0 or 1");
    }
    // Separate prefill differential. It is target-wide-owner gated and never
    // implied by the already-qualified M=1 route.
    fast_native_persistent_prefill_enabled_ = target_wide_prefill_owner &&
        target_wide_prefill_enabled_ && fast_native_persistent_prefill &&
        std::strcmp(fast_native_persistent_prefill, "1") == 0;
    const char* fast_native_mia_target_prefill_fp16 =
        std::getenv("NINFER_EXL3_FAST_NATIVE_MIA_TARGET_PREFILL_FP16");
    if (fast_native_mia_target_prefill_fp16 &&
        std::strcmp(fast_native_mia_target_prefill_fp16, "0") != 0 &&
        std::strcmp(fast_native_mia_target_prefill_fp16, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_NATIVE_MIA_TARGET_PREFILL_FP16 must be 0 or 1");
    }
    // Separately labeled target-wide same-weight FP16-output differential.
    // It is construction-latched to the target prefill owner and does not
    // inherit the numeric reconstruction-workspace route above.
    fast_native_mia_target_prefill_fp16_enabled_ = target_wide_prefill_owner &&
        target_wide_prefill_enabled_ && fast_native_mia_target_prefill_fp16 &&
        std::strcmp(fast_native_mia_target_prefill_fp16, "1") == 0;
    const char* fast_native_persistent_m1_only =
        std::getenv("NINFER_EXL3_FAST_NATIVE_PERSISTENT_M1_ONLY");
    if (fast_native_persistent_m1_only &&
        std::strcmp(fast_native_persistent_m1_only, "6") != 0 &&
        std::strcmp(fast_native_persistent_m1_only, "7") != 0 &&
        std::strcmp(fast_native_persistent_m1_only, "both") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_NATIVE_PERSISTENT_M1_ONLY must be 6, 7, or both");
    }
    fast_native_persistent_m1_only_ =
        fast_native_persistent_m1_only &&
        std::strcmp(fast_native_persistent_m1_only, "6") == 0 ? 6 :
        fast_native_persistent_m1_only &&
        std::strcmp(fast_native_persistent_m1_only, "7") == 0 ? 7 : 0;
    const bool fast_same_weights_fp16_m1_wide = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_WIDE",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_WIDE must be 0 or 1");
    fast_same_weights_fp16_m1_wide_enabled_ = target_m1_owner &&
        fast_same_weights_fp16_m1_enabled_ && fast_same_weights_fp16_m1_wide;
    const bool fast_same_weights_fp16_m1_wide_n32 = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_WIDE_N32",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_WIDE_N32 must be 0 or 1");
    fast_same_weights_fp16_m1_wide_n32_enabled_ = fast_same_weights_fp16_m1_wide_enabled_ && fast_same_weights_fp16_m1_wide_n32;
    const char* fast_same_weights_fp16_m1_wide_k5 =
        std::getenv("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_WIDE_K5");
    if (fast_same_weights_fp16_m1_wide_k5 &&
        std::strcmp(fast_same_weights_fp16_m1_wide_k5, "0") != 0 &&
        std::strcmp(fast_same_weights_fp16_m1_wide_k5, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_WIDE_K5 must be 0 or 1");
    }
    // K5 is deliberately separate from the established K6/K7 wide lane:
    // its packed decoder has different signed-bit handling and needs its own
    // paired-C1 qualification before it can be admitted.
    fast_same_weights_fp16_m1_wide_k5_enabled_ = target_m1_owner &&
        fast_same_weights_fp16_m1_enabled_ && fast_same_weights_fp16_m1_wide_k5 &&
        std::strcmp(fast_same_weights_fp16_m1_wide_k5, "1") == 0;
    const char* fast_same_weights_fp16_m1_n64 =
        std::getenv("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_N64");
    if (fast_same_weights_fp16_m1_n64 &&
        std::strcmp(fast_same_weights_fp16_m1_n64, "0") != 0 &&
        std::strcmp(fast_same_weights_fp16_m1_n64, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_N64 must be 0 or 1");
    }
    // Separate, target-owned differential route. It is deliberately not
    // implied by FAST_SAME_WEIGHTS_FP16_M1: the 64-tile topology has a larger
    // shared-memory/register footprint and must be qualified independently.
    fast_same_weights_fp16_m1_n64_enabled_ = target_m1_owner &&
        fast_same_weights_fp16_m1_n64 &&
        std::strcmp(fast_same_weights_fp16_m1_n64, "1") == 0;
    const char* fast_same_weights_fp16_m1_n64_k5 =
        std::getenv("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_N64_K5");
    if (fast_same_weights_fp16_m1_n64_k5 &&
        std::strcmp(fast_same_weights_fp16_m1_n64_k5, "0") != 0 &&
        std::strcmp(fast_same_weights_fp16_m1_n64_k5, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16_M1_N64_K5 must be 0 or 1");
    }
    // K5-only exact route: the packed EXL3 weights and native FP32-MMA
    // arithmetic remain authoritative; this is not the INT8/re-quantized path.
    fast_same_weights_fp16_m1_n64_k5_enabled_ = target_m1_owner &&
        fast_same_weights_fp16_m1_n64_k5 &&
        std::strcmp(fast_same_weights_fp16_m1_n64_k5, "1") == 0;
    const bool fast_same_weights_int8_gemv = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV must be 0 or 1");
    fast_same_weights_int8_gemv_enabled_ = target_m1_owner &&
        fast_same_weights_int8_gemv;
    const char* fast_same_weights_int8_gemv_k7 =
        std::getenv("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV_K7");
    if (fast_same_weights_int8_gemv_k7 &&
        std::strcmp(fast_same_weights_int8_gemv_k7, "0") != 0 &&
        std::strcmp(fast_same_weights_int8_gemv_k7, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV_K7 must be 0 or 1");
    }
    fast_same_weights_int8_gemv_k7_enabled_ = target_m1_owner &&
        fast_same_weights_int8_gemv_k7 &&
        std::strcmp(fast_same_weights_int8_gemv_k7, "1") == 0;
    const char* fast_same_weights_int8_gemv_down_k6 =
        std::getenv("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV_DOWN_K6");
    if (fast_same_weights_int8_gemv_down_k6 &&
        std::strcmp(fast_same_weights_int8_gemv_down_k6, "0") != 0 &&
        std::strcmp(fast_same_weights_int8_gemv_down_k6, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV_DOWN_K6 must be 0 or 1");
    }
    fast_same_weights_int8_gemv_down_k6_enabled_ = target_m1_owner &&
        fast_same_weights_int8_gemv_down_k6 &&
        std::strcmp(fast_same_weights_int8_gemv_down_k6, "1") == 0;
    const char* fast_same_weights_int8_gemv_down_k7 =
        std::getenv("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV_DOWN_K7");
    if (fast_same_weights_int8_gemv_down_k7 &&
        std::strcmp(fast_same_weights_int8_gemv_down_k7, "0") != 0 &&
        std::strcmp(fast_same_weights_int8_gemv_down_k7, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV_DOWN_K7 must be 0 or 1");
    }
    fast_same_weights_int8_gemv_down_k7_enabled_ = target_m1_owner &&
        fast_same_weights_int8_gemv_down_k7 &&
        std::strcmp(fast_same_weights_int8_gemv_down_k7, "1") == 0;
    const bool fast_same_weights_int8_mia_policy = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_MIA_POLICY",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_MIA_POLICY must be 0 or 1");
    fast_same_weights_int8_mia_policy_enabled_ = target_m1_owner &&
        fast_same_weights_int8_mia_policy;
    const bool fast_same_weights_int8_gemv_occupancy_grid = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV_OCCUPANCY_GRID",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_INT8_GEMV_OCCUPANCY_GRID must be 0 or 1");
    fast_same_weights_int8_gemv_occupancy_grid_enabled_ = target_m1_owner &&
        fast_same_weights_int8_gemv_occupancy_grid;
    const bool fast_same_weights_fp16kv_m1_mgemm_pair = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_MGEMM_PAIR",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_MGEMM_PAIR must be 0 or 1");
    fast_same_weights_fp16kv_m1_mgemm_pair_enabled_ = target_m1_owner &&
        fast_same_weights_fp16kv_m1_mgemm_pair;
    const bool fast_same_weights_fp16kv_m1_mgemm_policy = read_binary_option("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_MGEMM_POLICY",
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_MGEMM_POLICY must be 0 or 1");
    fast_same_weights_fp16kv_m1_mgemm_policy_enabled_ = target_m1_owner &&
        fast_same_weights_fp16kv_m1_mgemm_policy;
    target_k5_small_m_batch_enabled_ = target_m1_owner &&
        target_k5_small_m_batch &&
        std::strcmp(target_k5_small_m_batch, "1") == 0;
    const char* target_k6_m1_simt=
        std::getenv("NINFER_EXL3_TARGET_K6_M1_SIMT");
    if(target_k6_m1_simt && std::strcmp(target_k6_m1_simt,"0")!=0 &&
       std::strcmp(target_k6_m1_simt,"1")!=0)
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_K6_M1_SIMT must be 0 or 1");
    target_k6_m1_simt_enabled_=target_m1_owner && target_k6_m1_simt &&
        std::strcmp(target_k6_m1_simt,"1")==0;
    const char* target_m1_k6_n16=
        std::getenv("NINFER_EXL3_TARGET_M1_K6_N16");
    if(target_m1_k6_n16 && std::strcmp(target_m1_k6_n16,"0")!=0 &&
       std::strcmp(target_m1_k6_n16,"1")!=0)
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_M1_K6_N16 must be 0 or 1");
    target_m1_k6_n16_enabled_=target_m1_owner && target_m1_k6_n16 &&
        std::strcmp(target_m1_k6_n16,"1")==0;
    const char* target_m1_k7_three_word =
        std::getenv("NINFER_EXL3_TARGET_M1_K7_THREE_WORD");
    if (target_m1_k7_three_word &&
        std::strcmp(target_m1_k7_three_word, "0") != 0 &&
        std::strcmp(target_m1_k7_three_word, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_M1_K7_THREE_WORD must be 0 or 1");
    }
    target_m1_k7_three_word_enabled_ = target_m1_owner &&
        target_m1_k7_three_word &&
        std::strcmp(target_m1_k7_three_word, "1") == 0;
    target_m1_k6_n32_async_a_enabled_ = target_m1_owner &&
        target_m1_k6_n32_async_a &&
        std::strcmp(target_m1_k6_n32_async_a, "1") == 0;
    const char* native_k6_critical_path =
        std::getenv("NINFER_EXL3_NATIVE_K6_CRITICAL_PATH");
    if (native_k6_critical_path &&
        std::strcmp(native_k6_critical_path, "0") != 0 &&
        std::strcmp(native_k6_critical_path, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_NATIVE_K6_CRITICAL_PATH must be 0 or 1");
    }
    // R49 target-only differential. The unset state is deliberately disabled;
    // ordinary callers and the existing INT8 down owners cannot reach it.
    native_k6_critical_path_enabled_ = target_m1_owner &&
        native_k6_critical_path &&
        std::strcmp(native_k6_critical_path, "1") == 0;
    const bool native_k6_global_slices = read_binary_option("NINFER_EXL3_NATIVE_K6_GLOBAL_SLICES",
        "NINFER_EXL3_NATIVE_K6_GLOBAL_SLICES must be 0 or 1");
    native_k6_global_slices_enabled_ = target_m1_owner &&
        native_k6_critical_path_enabled_ && native_k6_global_slices;
    const bool native_k6_register_pipeline = read_binary_option("NINFER_EXL3_NATIVE_K6_REGISTER_PIPELINE",
        "NINFER_EXL3_NATIVE_K6_REGISTER_PIPELINE must be 0 or 1");
    native_k6_register_pipeline_enabled_ = target_m1_owner &&
        native_k6_critical_path_enabled_ && native_k6_register_pipeline;
    const char* native_k6_shape4_register_pipeline =
        std::getenv("NINFER_EXL3_NATIVE_K6_SHAPE4_REGISTER_PIPELINE");
    if (native_k6_shape4_register_pipeline &&
        std::strcmp(native_k6_shape4_register_pipeline, "0") != 0 &&
        std::strcmp(native_k6_shape4_register_pipeline, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_NATIVE_K6_SHAPE4_REGISTER_PIPELINE must be 0 or 1");
    }
    // Shape-4-only differential: keep the existing generic K6 contract, but
    // expose the Mia-shaped 256-thread/N32 pipeline as an independently
    // labeled opt-in.  The exact shape restriction is captured here so the
    // occupancy query and all later dispatches share one frozen mode.
    native_k6_shape4_register_pipeline_enabled_ = target_m1_owner &&
        native_k6_critical_path_enabled_ &&
        in_features_ == 5120 && out_features_ == 17408 &&
        native_k6_shape4_register_pipeline &&
        std::strcmp(native_k6_shape4_register_pipeline, "1") == 0;
    const bool native_k6_register_pipeline_n16 = read_binary_option("NINFER_EXL3_NATIVE_K6_REGISTER_PIPELINE_N16",
        "NINFER_EXL3_NATIVE_K6_REGISTER_PIPELINE_N16 must be 0 or 1");
    native_k6_register_pipeline_n16_enabled_ = target_m1_owner &&
        native_k6_register_pipeline_enabled_ && native_k6_register_pipeline_n16;
    const char* target_m1_k7_n32_async_a =
        std::getenv("NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A");
    if (target_m1_k7_n32_async_a &&
        std::strcmp(target_m1_k7_n32_async_a, "0") != 0 &&
        std::strcmp(target_m1_k7_n32_async_a, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_M1_K7_N32_ASYNC_A must be 0 or 1");
    }
    target_m1_k7_n32_async_a_enabled_ = target_m1_owner &&
        target_m1_k7_n32_async_a &&
        std::strcmp(target_m1_k7_n32_async_a, "1") == 0;
    const char* target_k6_small_m_async_a =
        std::getenv("NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A");
    if (target_k6_small_m_async_a &&
        std::strcmp(target_k6_small_m_async_a, "0") != 0 &&
        std::strcmp(target_k6_small_m_async_a, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A must be 0 or 1");
    }
    target_k6_small_m_async_a_enabled_ = target_m1_owner &&
        target_k6_small_m_async_a &&
        std::strcmp(target_k6_small_m_async_a, "1") == 0;
    const char* target_k7_small_m_async_a =
        std::getenv("NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A");
    if (target_k7_small_m_async_a &&
        std::strcmp(target_k7_small_m_async_a, "0") != 0 &&
        std::strcmp(target_k7_small_m_async_a, "1") != 0) {
        throw std::invalid_argument(
            "NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A must be 0 or 1");
    }
    target_k7_small_m_async_a_enabled_ = target_m1_owner &&
        target_k7_small_m_async_a &&
        std::strcmp(target_k7_small_m_async_a, "1") == 0;
    const char* target_kv = std::getenv("NINFER_EXL3_TARGET_KV_SMALL_M");
    target_kv_small_m_enabled_ = target_kv_owner && target_kv &&
        std::strcmp(target_kv, "1") == 0;
    const char* target_o_k6_small_m = std::getenv("NINFER_EXL3_TARGET_O_K6_SMALL_M");
    target_o_k6_small_m_enabled_ = target_o_k7_owner && target_o_k6_small_m &&
        std::strcmp(target_o_k6_small_m, "1") == 0;
    const char* target_o_k7_small_m =
        std::getenv("NINFER_EXL3_TARGET_O_K7_SMALL_M");
    target_o_k7_small_m_enabled_ = target_o_k7_owner &&
        (target_o_k7_small_m == nullptr || std::string(target_o_k7_small_m) != "0");
    const char* target_qkv_k6_small_m = std::getenv("NINFER_EXL3_TARGET_QKV_K6_SMALL_M");
    target_qkv_k6_small_m_enabled_ = target_qkv_k6_owner &&
        (target_qkv_k6_small_m == nullptr || std::string(target_qkv_k6_small_m) != "0");
    const char* target_q_k6_small_m = std::getenv("NINFER_EXL3_TARGET_Q_K6_SMALL_M");
    target_q_k6_small_m_enabled_ = target_q_k6_owner && target_q_k6_small_m &&
        std::strcmp(target_q_k6_small_m, "1") == 0;
    const char* target_z_k6_small_m = std::getenv("NINFER_EXL3_TARGET_Z_K6_SMALL_M");
    target_z_k6_small_m_enabled_ = target_z_k6_owner &&
        (target_z_k6_small_m == nullptr || std::string(target_z_k6_small_m) != "0");
    const char* direct_async_a=std::getenv("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A");
    target_direct_async_a_=direct_async_a && std::strcmp(direct_async_a,"1")==0;
    const char* direct_async_all=std::getenv("NINFER_EXL3_PREFILL_DIRECT_ASYNC_ALL");
    if(direct_async_all && std::strcmp(direct_async_all,"0")!=0 &&
       std::strcmp(direct_async_all,"1")!=0)
        throw std::invalid_argument("NINFER_EXL3_PREFILL_DIRECT_ASYNC_ALL must be 0 or 1");
    target_direct_async_all_=target_direct_async_a_ &&
        (direct_async_all ? std::strcmp(direct_async_all,"1")==0
                          : qualified_default_group);
    const char* direct_tiles64=std::getenv("NINFER_EXL3_PREFILL_DIRECT_TILES64");
    if(direct_tiles64 && std::strcmp(direct_tiles64,"0")!=0 &&
       std::strcmp(direct_tiles64,"1")!=0)
        throw std::invalid_argument("NINFER_EXL3_PREFILL_DIRECT_TILES64 must be 0 or 1");
    target_direct_tiles64_=target_direct_async_a_ &&
        (direct_tiles64 ? std::strcmp(direct_tiles64,"1")==0
                        : qualified_default_group);
    const bool k6_fast_decode = read_binary_option("NINFER_EXL3_PREFILL_K6_MOD48_FAST_DECODE",
        "NINFER_EXL3_PREFILL_K6_MOD48_FAST_DECODE must be 0 or 1");
    target_k6_fast_decode_ = target_wide_prefill_owner && k6_fast_decode;
    const bool k6_rowpair_n64 = read_binary_option("NINFER_EXL3_PREFILL_K6_ROWPAIR_N64",
        "NINFER_EXL3_PREFILL_K6_ROWPAIR_N64 must be 0 or 1");
    target_k6_rowpair_n64_ = target_wide_prefill_owner && k6_rowpair_n64;
    const bool k6_down_rowpair = read_binary_option("NINFER_EXL3_PREFILL_K6_DOWN_ROWPAIR",
        "NINFER_EXL3_PREFILL_K6_DOWN_ROWPAIR must be 0 or 1");
    target_k6_down_rowpair_ = target_down_owner && target_wide_prefill_owner &&
        k6_down_rowpair;
    const bool shape4_n64 = read_binary_option("NINFER_EXL3_PREFILL_SHAPE4_N64",
        "NINFER_EXL3_PREFILL_SHAPE4_N64 must be 0 or 1");
    target_shape4_n64_ = specialized_shape_ && target_wide_prefill_owner &&
        shape4_n64;
    const bool gateup_warpgroup = read_binary_option("NINFER_EXL3_PREFILL_K6_GATEUP_WARPGROUP_ASYNC",
        "NINFER_EXL3_PREFILL_K6_GATEUP_WARPGROUP_ASYNC must be 0 or 1");
    target_k6_gateup_warpgroup_async_ = target_gateup_owner &&
        gateup_warpgroup;
    const bool gateup_n32_pair_cta = read_binary_option("NINFER_EXL3_PREFILL_K6_GATEUP_N32_PAIR_CTA",
        "NINFER_EXL3_PREFILL_K6_GATEUP_N32_PAIR_CTA must be 0 or 1");
    target_k6_gateup_n32_pair_cta_ = target_gateup_owner &&
        gateup_n32_pair_cta;
    if(target_k6_gateup_n32_pair_cta_ && target_k6_gateup_warpgroup_async_)
        throw std::invalid_argument(
            "K6 gate/up N32-pair CTA and warp-group candidates are mutually exclusive");
    const bool gate_up_pair = read_binary_option("NINFER_EXL3_TARGET_PREFILL_GATE_UP_PAIR",
        "NINFER_EXL3_TARGET_PREFILL_GATE_UP_PAIR must be 0 or 1");
    target_prefill_gate_up_pair_ = target_gateup_owner && gate_up_pair;
    const char* k7_tiles64_exact_splits =
        std::getenv("NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS");
    if (k7_tiles64_exact_splits &&
        std::strcmp(k7_tiles64_exact_splits,"0")!=0 &&
        std::strcmp(k7_tiles64_exact_splits,"1")!=0)
        throw std::invalid_argument(
            "NINFER_EXL3_PREFILL_K7_TILES64_EXACT_SPLITS must be 0 or 1");
    target_k7_tiles64_exact_splits_ = target_wide_prefill_owner &&
        k7_tiles64_exact_splits &&
        std::strcmp(k7_tiles64_exact_splits,"1")==0;
    const bool k8_kv_async_a = read_binary_option("NINFER_EXL3_PREFILL_K8_KV_ASYNC_A",
        "NINFER_EXL3_PREFILL_K8_KV_ASYNC_A must be 0 or 1");
    target_k8_kv_prefill_async_a_ = target_kv_owner && k8_kv_async_a;
    const char* reduce_shfl=std::getenv("NINFER_EXL3_PREFILL_REDUCE_SHFL");
    if(reduce_shfl && std::strcmp(reduce_shfl,"0")!=0 &&
       std::strcmp(reduce_shfl,"1")!=0)
        throw std::invalid_argument("NINFER_EXL3_PREFILL_REDUCE_SHFL must be 0 or 1");
    target_reduce_shfl_=target_wide_prefill_owner &&
        (reduce_shfl ? std::strcmp(reduce_shfl,"1")==0
                     : qualified_default_group);
    const bool reduce_min_barriers = read_binary_option("NINFER_EXL3_PREFILL_REDUCE_SHFL_MIN_BARRIERS",
        "NINFER_EXL3_PREFILL_REDUCE_SHFL_MIN_BARRIERS must be 0 or 1");
    target_reduce_shfl_min_barriers_ = target_reduce_shfl_ &&
        reduce_min_barriers;
    const char* rowpair=std::getenv("NINFER_EXL3_PREFILL_ROWPAIR_K6");
    target_rowpair_k6_=target_wide_prefill_owner && rowpair && std::strcmp(rowpair,"1")==0;
    const bool rowpair_k7 = read_binary_option("NINFER_EXL3_PREFILL_ROWPAIR_K7",
        "NINFER_EXL3_PREFILL_ROWPAIR_K7 must be 0 or 1");
    target_rowpair_k7_ = target_wide_prefill_owner && rowpair_k7;
    const char* direct_partials=std::getenv("NINFER_EXL3_PREFILL_DIRECT_PARTIALS");
    target_direct_partials_=direct_partials
        ? std::strcmp(direct_partials,"1")==0 : qualified_default_group;
    const char* staged_shape4=std::getenv("NINFER_EXL3_PREFILL_STAGED_SHAPE4");
    target_staged_shape4_=staged_shape4
        ? std::strcmp(staged_shape4,"1")==0 : qualified_default_group;
    if (in_features_ <= 0 || out_features_ <= 0 || in_features_ % kHadamard != 0 ||
        out_features_ % kHadamard != 0) {
        throw std::invalid_argument("EXL3 CUDA dimensions must be positive and H128-aligned");
    }
    if (max_rows_ <= 0) {
        throw std::invalid_argument("EXL3 CUDA workspace max_rows must be positive");
    }
    if (specialized_shape_) {
        int device = 0;
        cuda_check(cudaGetDevice(&device), "query EXL3 CUDA device");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device),
                   "query EXL3 cooperative device properties");
        int active_blocks = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &active_blocks, exl3_gemm_m1_shape4_kernel,
                       kThreads, kShape4SharedBytes),
                   "query EXL3 cooperative occupancy");
        const int cooperative_capacity = active_blocks * properties.multiProcessorCount;
        if (cooperative_capacity < kShape4CooperativeGrid) {
            throw std::runtime_error(
                "EXL3 M=1 split kernel exceeds cooperative launch capacity");
        }
        cooperative_grid_ = kShape4CooperativeGrid;
    }
    if (allow_generic_variants_) {
        int device = 0;
        cuda_check(cudaGetDevice(&device), "query EXL3 generic CUDA device");
        cudaDeviceProp properties{};
        cuda_check(cudaGetDeviceProperties(&properties, device),
                   "query EXL3 generic device properties");
        if (coherent_wide_k6_enabled_) {
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active,
                exl3_gemm_m1_generic_mma_kernel<6, false, 8, true, true>,
                kThreads, kCoherentWideK6SharedBytes),
                "query coherent wide K6 occupancy");
            cudaFuncAttributes attributes{};
            cuda_check(cudaFuncGetAttributes(&attributes,
                exl3_gemm_m1_generic_mma_kernel<6, false, 8, true, true>),
                "query coherent wide K6 kernel resources");
            coherent_wide_k6_resident_capacity_ =
                active * properties.multiProcessorCount;
            coherent_wide_k6_registers_per_thread_ = attributes.numRegs;
        }
        if (coherent_down_k6_enabled_ && in_features_ == 17408 &&
            out_features_ == 5120) {
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active,
                exl3_gemm_m1_generic_mma_kernel<6, false, 8, true, true>,
                kThreads, kCoherentDownK6SharedBytes),
                "query coherent K6 down shared-row occupancy");
            coherent_down_k6_resident_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (coherent_down_k7_enabled_ && in_features_ == 17408 &&
            out_features_ == 5120) {
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active,
                exl3_gemm_m1_generic_mma_kernel<7, false, 8, true, true>,
                kThreads, kCoherentDownK7SharedBytes),
                "query coherent K7 down shared-row occupancy");
            coherent_down_k7_resident_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (coherent_o_k7_enabled_ && in_features_ == 6144 &&
            out_features_ == 5120) {
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active,
                exl3_gemm_m1_generic_mma_kernel<7, false, 8, true, true>,
                kThreads, kCoherentOK7SharedBytes),
                "query coherent K7 O shared-row occupancy");
            coherent_o_k7_resident_capacity_ =
                active * properties.multiProcessorCount;
        }
        const std::size_t static_bytes = 256u * sizeof(half) +
            16u * 512u * sizeof(float);
        const std::size_t shared_bytes[3] = {
            static_bytes + 2u * 32u * 16u * 5u * sizeof(std::uint16_t),
            static_bytes + 2u * 32u * 16u * 6u * sizeof(std::uint16_t),
            static_bytes + 2u * 32u * 16u * 8u * sizeof(std::uint16_t)};
        const auto query_capacity = [&](int bits, std::size_t bytes) {
            int active = 0;
            if (bits == 5) {
                cuda_check(cudaFuncSetAttribute(
                    exl3_gemm_m1_generic_mma_kernel<5>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(bytes)),
                    "set EXL3 K5 generic shared-memory limit");
            } else if (bits == 6) {
                cuda_check(cudaFuncSetAttribute(
                    exl3_gemm_m1_generic_mma_kernel<6>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(bytes)),
                    "set EXL3 K6 generic shared-memory limit");
            } else if (bits == 7) {
                cuda_check(cudaFuncSetAttribute(
                    exl3_gemm_m1_generic_mma_kernel<7>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(bytes)),
                    "set EXL3 K7 generic shared-memory limit");
            } else {
                cuda_check(cudaFuncSetAttribute(
                    exl3_gemm_m1_generic_mma_kernel<8>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(bytes)),
                    "set EXL3 K8 generic shared-memory limit");
            }
            if (bits == 5) {
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &active, exl3_gemm_m1_generic_mma_kernel<5>, kThreads, bytes),
                    "query EXL3 K5 generic occupancy");
            } else if (bits == 6) {
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &active, exl3_gemm_m1_generic_mma_kernel<6>, kThreads, bytes),
                    "query EXL3 K6 generic occupancy");
            } else if (bits == 7) {
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &active, exl3_gemm_m1_generic_mma_kernel<7>, kThreads, bytes),
                    "query EXL3 K7 generic occupancy");
            } else {
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &active, exl3_gemm_m1_generic_mma_kernel<8>, kThreads, bytes),
                    "query EXL3 K8 generic occupancy");
            }
            return active * properties.multiProcessorCount;
        };
        generic_capacity_[0] = query_capacity(5, shared_bytes[0]);
        generic_capacity_[1] = query_capacity(6, shared_bytes[1]);
        generic_capacity_[2] = query_capacity(8, shared_bytes[2]);
        if ((fast_native_persistent_m1_enabled_ ||
             fast_native_mia_m1_fp16_enabled_) &&
            ((in_features_ == 5120 && out_features_ == 17408) ||
             (in_features_ == 17408 && out_features_ == 5120))) {
            const auto query_persistent = [&](int bits, std::size_t bytes,
                                              auto kernel, int& grid) {
                cuda_check(cudaFuncSetAttribute(
                               kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(bytes)),
                           "set native M1 shared-memory limit");
                int active = 0;
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &active, kernel, 256 * (bits == 6 &&
                                   in_features_ == 5120 ? 1 : 2), bytes),
                           "query native M1 occupancy");
                grid = active > 0 ? properties.multiProcessorCount : 0;
            };
            if (in_features_ == 5120 && out_features_ == 17408) {
                if (fast_native_mia_m1_fp16_enabled_) {
                    query_persistent(
                        6, exl3_native_persistent_smem_bytes<6, 16, 512>(),
                        exl3_native_mia_m1_fp16_kernel<6, 16, 512>,
                        fast_native_persistent_m1_grid_[0]);
                } else {
                    query_persistent(
                        6, exl3_native_persistent_smem_bytes<6, 16, 512>(),
                        exl3_native_persistent_m1_kernel<6, 16, 512>,
                        fast_native_persistent_m1_grid_[0]);
                }
            } else {
                if (fast_native_mia_m1_fp16_enabled_) {
                    query_persistent(
                        6, exl3_native_persistent_smem_bytes<6, 32, 256>(),
                        exl3_native_mia_m1_fp16_kernel<6, 32, 256>,
                        fast_native_persistent_m1_grid_[0]);
                } else {
                    query_persistent(
                        6, exl3_native_persistent_smem_bytes<6, 32, 256>(),
                        exl3_native_persistent_m1_kernel<6, 32, 256>,
                        fast_native_persistent_m1_grid_[0]);
                }
            }
            if (fast_native_mia_m1_fp16_enabled_) {
                query_persistent(
                    7, exl3_native_persistent_smem_bytes<7, 32, 128>(),
                    exl3_native_mia_m1_fp16_kernel<7, 32, 128, true>,
                    fast_native_persistent_m1_grid_[1]);
            } else {
                query_persistent(
                    7, exl3_native_persistent_smem_bytes<7, 32, 128>(),
                    exl3_native_persistent_m1_kernel<7, 32, 128, true>,
                        fast_native_persistent_m1_grid_[1]);
            }
            if (fast_native_mia_m1_fp16_fused_input_enabled_) {
                if (in_features_ == 5120 && out_features_ == 17408) {
                    query_persistent(
                        6, exl3_native_persistent_smem_bytes<6, 16, 512>(),
                        exl3_native_mia_m1_fp16_fused_input_kernel<6, 16, 512>,
                        fast_native_mia_m1_fp16_fused_input_grid_[0]);
                } else {
                    query_persistent(
                        6, exl3_native_persistent_smem_bytes<6, 32, 256>(),
                        exl3_native_mia_m1_fp16_fused_input_kernel<6, 32, 256>,
                        fast_native_mia_m1_fp16_fused_input_grid_[0]);
                }
                query_persistent(
                    7, exl3_native_persistent_smem_bytes<7, 32, 128>(),
                    exl3_native_mia_m1_fp16_fused_input_kernel<7, 32, 128, true>,
                    fast_native_mia_m1_fp16_fused_input_grid_[1]);
            }
        }
        if (fast_same_weights_fp16kv_m1_mgemm_pair_enabled_ &&
            in_features_ == 5120 && out_features_ == 17408) {
            const auto query_mgemm_pair = [&](std::size_t bytes, auto kernel,
                                              int threads, int& grid) {
                cuda_check(cudaFuncSetAttribute(
                               kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(bytes)),
                           "set native M1 MGEMM-pair shared-memory limit");
                int active = 0;
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &active, kernel, threads, bytes),
                           "query native M1 MGEMM-pair occupancy");
                // The z dimension carries two independent projections.
                grid = active > 0
                    ? std::max(1, active * properties.multiProcessorCount / 2)
                    : 0;
            };
            query_mgemm_pair(
                exl3_native_persistent_smem_bytes<5, 16, 512>(),
                exl3_native_mia_m1_mgemm_pair_kernel<5, 16, 512>,
                256,
                fast_same_weights_fp16kv_m1_mgemm_pair_grid_[0]);
            query_mgemm_pair(
                exl3_native_persistent_smem_bytes<6, 16, 512>(),
                exl3_native_mia_m1_mgemm_pair_kernel<6, 16, 512>,
                256,
                fast_same_weights_fp16kv_m1_mgemm_pair_grid_[1]);
            query_mgemm_pair(
                exl3_native_persistent_smem_bytes<7, 32, 128>(),
                exl3_native_mia_m1_mgemm_pair_kernel<7, 32, 128, true>,
                512,
                fast_same_weights_fp16kv_m1_mgemm_pair_grid_[2]);
        }
        if (fast_same_weights_fp16kv_m1_mgemm_policy_enabled_ &&
            ((in_features_ == 5120 && out_features_ != 248320) ||
             (in_features_ == 17408 && out_features_ == 5120))) {
            const bool gate_up = in_features_ == 5120 && out_features_ == 17408;
            // The pinned Mia single-projection path is exl3_gemm_kernel:
            // its inner kernel owns the final output Hadamard.  Query the
            // same native adapter that the dispatch below will launch;
            // querying the old MGEMM/post-Hadamard wrapper can select a grid
            // for a different register/shared-memory contract.
            const auto query_mia_gemm_policy = [&](std::size_t bytes,
                                                   auto kernel, int threads,
                                                   int& grid) {
                cuda_check(cudaFuncSetAttribute(
                               kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(bytes)),
                           "set native M1 Mia-GEMM-policy shared-memory limit");
                int active = 0;
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &active, kernel, threads, bytes),
                           "query native M1 Mia-GEMM-policy occupancy");
                grid = active > 0 ? active * properties.multiProcessorCount : 0;
            };
            if (gate_up) {
                query_mia_gemm_policy(
                    exl3_native_persistent_smem_bytes<6, 16, 512>(),
                    exl3_native_mia_m1_fp16_kernel<6, 16, 512>, 256,
                    fast_same_weights_fp16kv_m1_mgemm_policy_grid_[0]);
                query_mia_gemm_policy(
                    exl3_native_persistent_smem_bytes<7, 32, 128>(),
                    exl3_native_mia_m1_fp16_kernel<7, 32, 128, true>,
                    512, fast_same_weights_fp16kv_m1_mgemm_policy_grid_[1]);
            } else if ((in_features_ == 5120 && out_features_ == 10240) ||
                       (in_features_ == 17408 && out_features_ == 5120)) {
                query_mia_gemm_policy(
                    exl3_native_persistent_smem_bytes<6, 32, 256>(),
                    exl3_native_mia_m1_fp16_kernel<6, 32, 256>, 512,
                    fast_same_weights_fp16kv_m1_mgemm_policy_grid_[0]);
                query_mia_gemm_policy(
                    exl3_native_persistent_smem_bytes<7, 32, 128>(),
                    exl3_native_mia_m1_fp16_kernel<7, 32, 128, true>,
                    512, fast_same_weights_fp16kv_m1_mgemm_policy_grid_[1]);
            } else {
                query_mia_gemm_policy(
                    exl3_native_persistent_smem_bytes<6, 32, 128>(),
                    exl3_native_mia_m1_fp16_kernel<6, 32, 128>, 512,
                    fast_same_weights_fp16kv_m1_mgemm_policy_grid_[0]);
                query_mia_gemm_policy(
                    exl3_native_persistent_smem_bytes<7, 32, 128>(),
                    exl3_native_mia_m1_fp16_kernel<7, 32, 128, true>,
                    512, fast_same_weights_fp16kv_m1_mgemm_policy_grid_[1]);
            }
        }
        if (fast_native_persistent_prefill_enabled_ &&
            ((in_features_ == 5120 && out_features_ == 10240) ||
             (in_features_ == 5120 && out_features_ == 6144) ||
             (in_features_ == 6144 && out_features_ == 5120) ||
             (in_features_ == 5120 && out_features_ == 17408) ||
             (in_features_ == 17408 && out_features_ == 5120))) {
            const bool gate_up_family = in_features_ == 5120;
            const auto query_prefill = [&](std::size_t bytes, auto kernel,
                                           int threads, int& capacity) {
                cuda_check(cudaFuncSetAttribute(
                               kernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(bytes)),
                           "set native persistent prefill shared-memory limit");
                int active = 0;
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &active, kernel, threads, bytes),
                           "query native persistent prefill occupancy");
                capacity = active > 0 ? active * properties.multiProcessorCount : 0;
            };
            if (gate_up_family) {
                query_prefill(
                    exl3_native_persistent_smem_bytes<5, 16, 512>(),
                    exl3_native_persistent_prefill_kernel<5, 16, 512>, 256,
                    fast_native_persistent_prefill_capacity_[0]);
                query_prefill(
                    exl3_native_persistent_smem_bytes<6, 16, 512>(),
                    exl3_native_persistent_prefill_kernel<6, 16, 512>, 256,
                    fast_native_persistent_prefill_capacity_[1]);
                query_prefill(
                    exl3_native_persistent_smem_bytes<7, 16, 512>(),
                    exl3_native_persistent_prefill_kernel<7, 16, 512>, 256,
                    fast_native_persistent_prefill_capacity_[2]);
            } else {
                query_prefill(
                    exl3_native_persistent_smem_bytes<5, 32, 256>(),
                    exl3_native_persistent_prefill_kernel<5, 32, 256>, 512,
                    fast_native_persistent_prefill_capacity_[0]);
                query_prefill(
                    exl3_native_persistent_smem_bytes<6, 32, 256>(),
                    exl3_native_persistent_prefill_kernel<6, 32, 256>, 512,
                    fast_native_persistent_prefill_capacity_[1]);
                query_prefill(
                    exl3_native_persistent_smem_bytes<7, 32, 128>(),
                    exl3_native_persistent_prefill_kernel<7, 32, 128, true>, 512,
                    fast_native_persistent_prefill_capacity_[2]);
            }
        }
        if (fast_native_mia_target_prefill_fp16_enabled_ &&
            ((in_features_ == 5120 &&
              (out_features_ == 6144 || out_features_ == 10240 ||
               out_features_ == 12288 || out_features_ == 17408)) ||
             (in_features_ == 6144 && out_features_ == 5120) ||
             (in_features_ == 17408 && out_features_ == 5120))) {
            const bool down_family = in_features_ != 5120;
            const auto query_mia_target_prefill =
                [&](std::size_t bytes, auto kernel, int threads) {
                    cuda_check(cudaFuncSetAttribute(
                                   kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(bytes)),
                               "set native Mia target-prefill shared-memory limit");
                    int active = 0;
                    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &active, kernel, threads, bytes),
                               "query native Mia target-prefill occupancy");
                    fast_native_mia_target_prefill_fp16_capacity_ = active > 0
                        ? active * properties.multiProcessorCount : 0;
                };
            if (down_family) {
                query_mia_target_prefill(
                    exl3_native_persistent_smem_bytes<6, 32, 256>(),
                    exl3_native_mia_prefill_fp16_kernel<6, 32, 256>, 512);
            } else {
                query_mia_target_prefill(
                    exl3_native_persistent_smem_bytes<6, 16, 512>(),
                    exl3_native_mia_prefill_fp16_kernel<6, 16, 512>, 256);
            }
        }
        if (draft_k5_async_a_enabled_) {
            const std::size_t async_a_bytes =
                shared_bytes[0] + 256u * sizeof(half);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<5, false, 32, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set EXL3 draft K5 async-A shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<5, false, 32, true>,
                           kThreads, async_a_bytes),
                       "query EXL3 draft K5 async-A occupancy");
            draft_k5_async_a_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (target_gateup_k6_async_a_enabled_) {
            const std::size_t async_a_bytes =
                shared_bytes[1] + 256u * sizeof(half);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set EXL3 target gate/up K6 async-A shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>,
                           kThreads, async_a_bytes),
                       "query EXL3 target gate/up K6 async-A occupancy");
            target_gateup_k6_async_a_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (target_m1_k6_n32_async_a_enabled_) {
            const std::size_t async_a_bytes =
                shared_bytes[1] + 256u * sizeof(half);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set EXL3 target M1 K6 N32 async-A shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>,
                           kThreads, async_a_bytes),
                       "query EXL3 target M1 K6 N32 async-A occupancy");
            target_m1_k6_n32_async_a_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (native_k6_critical_path_enabled_) {
            // Same N32/Async-A shared layout as the R49 K6 partial path. Query
            // the exact FastK6Decode specialization so its independent launch
            // capacity remains explicit and fail-closed.
            const std::size_t async_a_bytes =
                shared_bytes[1] + 256u * sizeof(half);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<
                               6, false, 32, true, true, false, false, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set native K6 critical-path shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<
                               6, false, 32, true, true, false, false, true>,
                           kThreads, async_a_bytes),
                       "query native K6 critical-path occupancy");
            native_k6_critical_path_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (native_k6_global_slices_enabled_) {
            // Global-slice scheduling uses the two-stage async-A body and a
            // cooperative resident grid.  Its capacity is queried separately
            // from the normal five-plane launch because the launch contract
            // and lock storage are different even though shared storage is the
            // same.
            const std::size_t global_slice_bytes =
                shared_bytes[1] + 256u * sizeof(half);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<
                               6, false, 32, true, true, false, false, true,
                               false, false, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(global_slice_bytes)),
                       "set native K6 global-slice shared-memory limit");
            int global_active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &global_active,
                           exl3_gemm_m1_generic_mma_kernel<
                               6, false, 32, true, true, false, false, true,
                               false, false, true>,
                           kThreads, global_slice_bytes),
                       "query native K6 global-slice occupancy");
            native_k6_global_slices_capacity_ =
                global_active * properties.multiProcessorCount;
        }
        if (native_k6_register_pipeline_enabled_ ||
            native_k6_shape4_register_pipeline_enabled_) {
            // Four raw cp.async stages plus the existing 512-column FP32
            // output tile.  Keep capacity separate from the two-stage control
            // because the additional shared storage can change occupancy.
            const std::size_t pipeline_bytes =
                4u * 256u * sizeof(half) +
                4u * 32u * 16u * 6u * sizeof(std::uint16_t) +
                16u * 512u * sizeof(float);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<
                               6, false, 32, true, true, false, false, true,
                               false, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(pipeline_bytes)),
                       "set native K6 register-pipeline shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<
                               6, false, 32, true, true, false, false, true,
                               false, true>,
                           kThreads, pipeline_bytes),
                       "query native K6 register-pipeline occupancy");
            native_k6_register_pipeline_capacity_ =
                active * properties.multiProcessorCount;
            if (native_k6_register_pipeline_n16_enabled_) {
                // Keep the four raw cp.async stages and the three decoded
                // fragment stages, but halve the output tile.  This is an
                // independent occupancy/parallelism differential; it uses
                // the existing five split accumulation contract.
                const std::size_t pipeline_n16_bytes =
                    4u * 256u * sizeof(half) +
                    4u * 16u * 16u * 6u * sizeof(std::uint16_t) +
                    16u * 256u * sizeof(float);
                cuda_check(cudaFuncSetAttribute(
                               exl3_gemm_m1_generic_mma_kernel<
                                   6, false, 16, true, true, false, false,
                                   true, false, true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(pipeline_n16_bytes)),
                           "set native K6 N16 register-pipeline shared-memory limit");
                int n16_active = 0;
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &n16_active,
                               exl3_gemm_m1_generic_mma_kernel<
                                   6, false, 16, true, true, false, false,
                                   true, false, true>,
                               kThreads, pipeline_n16_bytes),
                           "query native K6 N16 register-pipeline occupancy");
                native_k6_register_pipeline_n16_capacity_ =
                    n16_active * properties.multiProcessorCount;
            }
        }
        if (fast_same_weights_fp16_m1_k6_register_pipeline_enabled_) {
            // This is the same K6/N32 register-fragment pipeline as the
            // native critical-path differential, but with an independent
            // capacity because this route is selected from the ordinary
            // same-weight FP16-M1 branch.
            const std::size_t pipeline_bytes =
                4u * 256u * sizeof(half) +
                4u * 32u * 16u * 6u * sizeof(std::uint16_t) +
                16u * 512u * sizeof(float);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<
                               6, false, 32, true, true, false, false, true,
                               false, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(pipeline_bytes)),
                       "set FAST same-weight K6 register-pipeline shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<
                               6, false, 32, true, true, false, false, true,
                               false, true>,
                           kThreads, pipeline_bytes),
                       "query FAST same-weight K6 register-pipeline occupancy");
            fast_same_weights_fp16_m1_k6_register_pipeline_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (target_m1_k7_n32_async_a_enabled_) {
            // K7 keeps the parent's m16n8k16 MMA topology and split order.
            // Only represented-A staging changes from the synchronous shared
            // copy to the already proven double-buffered cp.async supply.
            const std::size_t async_a_bytes =
                shared_bytes[2] + 256u * sizeof(half);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<7, false, 32, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set EXL3 target M1 K7 N32 async-A shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<7, false, 32, true>,
                           kThreads, async_a_bytes),
                       "query EXL3 target M1 K7 N32 async-A occupancy");
            target_m1_k7_n32_async_a_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (target_k6_small_m_async_a_enabled_) {
            const std::size_t async_a_bytes =
                shared_bytes[1] + 256u * sizeof(half);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set EXL3 target K6 small-M async-A shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>,
                           kThreads, async_a_bytes),
                       "query EXL3 target K6 small-M async-A occupancy");
            target_k6_small_m_async_a_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (target_k7_small_m_async_a_enabled_) {
            const std::size_t async_a_bytes =
                shared_bytes[2] + 256u * sizeof(half);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<7, false, 32, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set EXL3 target K7 small-M async-A shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<7, false, 32, true>,
                           kThreads, async_a_bytes),
                       "query EXL3 target K7 small-M async-A occupancy");
            target_k7_small_m_async_a_capacity_ =
                active * properties.multiProcessorCount;
        }
        // E4B3D's shape-specific topology uses 16 output tiles per CTA for
        // the large-down K6/K7 shapes. Keep its resource query separate from
        // the inherited 32-tile differential topology.
        const auto query_large_down_candidate = [&](int bits, int& capacity) {
            const std::size_t shared_bytes =
                256u * sizeof(half) + 2u * 16u * 16u * static_cast<std::size_t>(bits) * sizeof(std::uint16_t) +
                16u * 256u * sizeof(float);
            if (bits == 6) {
                cuda_check(cudaFuncSetAttribute(
                               exl3_gemm_m1_generic_mma_kernel<6, false, 16>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shared_bytes)),
                           "set EXL3 K6 large-down candidate shared-memory limit");
                int active = 0;
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &active, exl3_gemm_m1_generic_mma_kernel<6, false, 16>,
                               kThreads, shared_bytes),
                           "query EXL3 K6 large-down candidate occupancy");
                capacity = active * properties.multiProcessorCount;
            } else {
                cuda_check(cudaFuncSetAttribute(
                               exl3_gemm_m1_generic_mma_kernel<7, false, 16>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shared_bytes)),
                           "set EXL3 K7 large-down candidate shared-memory limit");
                int active = 0;
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &active, exl3_gemm_m1_generic_mma_kernel<7, false, 16>,
                               kThreads, shared_bytes),
                           "query EXL3 K7 large-down candidate occupancy");
                capacity = active * properties.multiProcessorCount;
            }
        };
        query_large_down_candidate(6, large_down_candidate_capacity_[0]);
        query_large_down_candidate(7, large_down_candidate_capacity_[1]);
        if (fast_same_weights_fp16_m1_enabled_ || fast_fp16_m2_8_down_enabled_ ||
            fast_fp16_m2_8_all_enabled_) {
            const std::size_t k6_wide_bytes =
                512u * sizeof(half) +
                2u * 32u * 16u * 6u * sizeof(std::uint16_t) +
                16u * 512u * sizeof(float);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(k6_wide_bytes)),
                       "set EXL3 fast FP16 M1 K6 shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>,
                           kThreads, k6_wide_bytes),
                       "query EXL3 fast FP16 M1 K6 occupancy");
            fast_same_weights_fp16_m1_capacity_[0] =
                active * properties.multiProcessorCount;

            const std::size_t k6_down_bytes =
                512u * sizeof(half) +
                2u * 16u * 16u * 6u * sizeof(std::uint16_t) +
                16u * 256u * sizeof(float);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<6, false, 16, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(k6_down_bytes)),
                       "set EXL3 fast FP16 M1 K6-down shared-memory limit");
            active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<6, false, 16, true>,
                           kThreads, k6_down_bytes),
                       "query EXL3 fast FP16 M1 K6-down occupancy");
            fast_same_weights_fp16_m1_capacity_[1] =
                active * properties.multiProcessorCount;

            const std::size_t k7_wide_bytes =
                512u * sizeof(half) +
                2u * 32u * 16u * 7u * sizeof(std::uint16_t) +
                16u * 512u * sizeof(float);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<7, false, 32, true,
                                                            false, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(k7_wide_bytes)),
                       "set EXL3 fast FP16 M1 K7 shared-memory limit");
            active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<7, false, 32, true,
                                                            false, true>,
                           kThreads, k7_wide_bytes),
                       "query EXL3 fast FP16 M1 K7 occupancy");
            fast_same_weights_fp16_m1_capacity_[2] =
                active * properties.multiProcessorCount;
        }
        if (fast_same_weights_fp16_m1_n16_enabled_) {
            const auto query_n16_capacity = [&](int bits, int& capacity) {
                const std::size_t bytes =
                    512u * sizeof(half) +
                    2u * 16u * 16u * static_cast<std::size_t>(bits) *
                        sizeof(std::uint16_t) +
                    16u * 256u * sizeof(float);
                int active = 0;
                if (bits == 6) {
                    cuda_check(cudaFuncSetAttribute(
                                   exl3_gemm_m1_generic_mma_kernel<6, false, 16, true>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(bytes)),
                               "set EXL3 fast FP16 M1 N16 K6 shared-memory limit");
                    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &active,
                                   exl3_gemm_m1_generic_mma_kernel<6, false, 16, true>,
                                   kThreads, bytes),
                               "query EXL3 fast FP16 M1 N16 K6 occupancy");
                } else {
                    cuda_check(cudaFuncSetAttribute(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 16, true,
                                                                    false, true>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(bytes)),
                               "set EXL3 fast FP16 M1 N16 K7 shared-memory limit");
                    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &active,
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 16, true,
                                                                    false, true>,
                                   kThreads, bytes),
                               "query EXL3 fast FP16 M1 N16 K7 occupancy");
                }
                capacity = active * properties.multiProcessorCount;
            };
            query_n16_capacity(6, fast_same_weights_fp16_m1_n16_capacity_[0]);
            fast_same_weights_fp16_m1_n16_capacity_[1] =
                fast_same_weights_fp16_m1_n16_capacity_[0];
            query_n16_capacity(7, fast_same_weights_fp16_m1_n16_capacity_[2]);
        }
        if (fast_same_weights_fp16_m1_n64_enabled_ ||
            fast_same_weights_fp16_m1_n64_k5_enabled_) {
            // N64 keeps the same 256-thread warp-to-fragment mapping but lets
            // each CTA own 1024 output values. This removes half the output
            // CTAs and their split-plane reduction traffic without changing
            // packed EXL3 decoding or the FP32 MMA accumulator.
            const auto query_n64_capacity = [&](int bits, int& capacity) {
                const std::size_t bytes =
                    512u * sizeof(half) +
                    2u * 64u * 16u * static_cast<std::size_t>(bits) *
                        sizeof(std::uint16_t) +
                    16u * 1024u * sizeof(float);
                if (bytes > properties.sharedMemPerBlockOptin) {
                    capacity = 0;
                    return;
                }
                int active = 0;
                if (bits == 5) {
                    cuda_check(cudaFuncSetAttribute(
                                   exl3_gemm_m1_generic_mma_kernel<5, false, 64, true>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(bytes)),
                               "set EXL3 fast FP16 M1 N64 K5 shared-memory limit");
                    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &active,
                                   exl3_gemm_m1_generic_mma_kernel<5, false, 64, true>,
                                   kThreads, bytes),
                               "query EXL3 fast FP16 M1 N64 K5 occupancy");
                } else if (bits == 6) {
                    cuda_check(cudaFuncSetAttribute(
                                   exl3_gemm_m1_generic_mma_kernel<6, false, 64, true>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(bytes)),
                               "set EXL3 fast FP16 M1 N64 K6 shared-memory limit");
                    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &active,
                                   exl3_gemm_m1_generic_mma_kernel<6, false, 64, true>,
                                   kThreads, bytes),
                               "query EXL3 fast FP16 M1 N64 K6 occupancy");
                } else {
                    cuda_check(cudaFuncSetAttribute(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 64, true,
                                                                    false, true>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(bytes)),
                               "set EXL3 fast FP16 M1 N64 K7 shared-memory limit");
                    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                   &active,
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 64, true,
                                                                    false, true>,
                                   kThreads, bytes),
                               "query EXL3 fast FP16 M1 N64 K7 occupancy");
                }
                capacity = active * properties.multiProcessorCount;
            };
            query_n64_capacity(5, fast_same_weights_fp16_m1_n64_capacity_[0]);
            if (fast_same_weights_fp16_m1_n64_enabled_) {
                query_n64_capacity(6, fast_same_weights_fp16_m1_n64_capacity_[1]);
                query_n64_capacity(7, fast_same_weights_fp16_m1_n64_capacity_[2]);
            }
        }
        if (target_down_k6_async_a_enabled_) {
            const std::size_t async_a_bytes =
                512u * sizeof(half) + 2u * 16u * 16u * 6u *
                    sizeof(std::uint16_t) + 16u * 256u * sizeof(float);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<6, false, 16, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set EXL3 target down K6 async-A shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<6, false, 16, true>,
                           kThreads, async_a_bytes),
                       "query EXL3 target down K6 async-A occupancy");
            target_down_k6_async_a_capacity_ =
                active * properties.multiProcessorCount;
        }
        if (target_down_k7_async_a_enabled_) {
            const std::size_t async_a_bytes =
                512u * sizeof(half) + 2u * 16u * 16u * 7u *
                    sizeof(std::uint16_t) + 16u * 256u * sizeof(float);
            cuda_check(cudaFuncSetAttribute(
                           exl3_gemm_m1_generic_mma_kernel<7, false, 16, true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(async_a_bytes)),
                       "set EXL3 target down K7 async-A shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active,
                           exl3_gemm_m1_generic_mma_kernel<7, false, 16, true>,
                           kThreads, async_a_bytes),
                       "query EXL3 target down K7 async-A occupancy");
            target_down_k7_async_a_capacity_ =
                active * properties.multiProcessorCount;
        }
        const auto query_large_down_candidate8 = [&](int bits, int& capacity) {
            const std::size_t bytes =
                256u * sizeof(half) + 2u * 8u * 16u * static_cast<std::size_t>(bits) * sizeof(std::uint16_t) +
                16u * 128u * sizeof(float);
            int active = 0;
            if (bits == 6) {
                cuda_check(cudaFuncSetAttribute(
                               exl3_gemm_m1_generic_mma_kernel<6, false, 8>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(bytes)),
                           "set EXL3 K6 8-tile candidate shared-memory limit");
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &active, exl3_gemm_m1_generic_mma_kernel<6, false, 8>,
                               kThreads, bytes),
                           "query EXL3 K6 8-tile candidate occupancy");
            } else {
                cuda_check(cudaFuncSetAttribute(
                               exl3_gemm_m1_generic_mma_kernel<7, false, 8>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(bytes)),
                           "set EXL3 K7 8-tile candidate shared-memory limit");
                cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                               &active, exl3_gemm_m1_generic_mma_kernel<7, false, 8>,
                               kThreads, bytes),
                           "query EXL3 K7 8-tile candidate occupancy");
            }
            capacity = active * properties.multiProcessorCount;
        };
        query_large_down_candidate8(6, large_down_candidate8_capacity_[0]);
        query_large_down_candidate8(7, large_down_candidate8_capacity_[1]);
    }
    if (target_staged_prefill_enabled_) {
        const int bytes = 256 * sizeof(half) + 2 * 32 * 16 * 8 * sizeof(std::uint16_t) + 16 * 512 * sizeof(float);
        cuda_check(cudaFuncSetAttribute(exl3_prefill_partials_kernel<8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), "set staged K8 shared memory");
    }
    const auto transformed_bytes=requirement.transformed_bytes;
    const auto accum_bytes=requirement.accumulation_bytes;
    storage_retirement_=std::make_unique<StorageRetirement>();
    if(bool(constructor_device_credit)!=bool(constructor_metadata_credit))
        throw std::invalid_argument("linear constructor requires both credit domains");
    if(constructor_device_credit) {
        if(!requirement.owned_bytes || constructor_device_credit->bytes()!=requirement.owned_bytes ||
            constructor_metadata_credit->bytes()!=metadata_bytes())
            throw std::invalid_argument("linear constructor credit extent mismatch");
        auto record=constructor_metadata_credit->split(sizeof(StorageRetirement));
        storage_retirement_->metadata_credit.emplace(std::move(*record));
        constructor_object_credit_.emplace(std::move(*constructor_metadata_credit));
        if(!transformed.data) {
            auto part=constructor_device_credit->split(transformed_bytes);
            storage_retirement_->transform_credit.emplace(std::move(*part));
        }
        if(!accumulation.data) {
            auto part=constructor_device_credit->split(accum_bytes);
            storage_retirement_->accumulation_credit.emplace(std::move(*part));
        }
    }
    transformed_capacity_bytes_=transformed_bytes;
    accumulation_capacity_bytes_=accum_bytes;
    if(requirement.owned_bytes)
        cuda_check(cudaGetDevice(&owner_device_),"query EXL3 linear allocation device");
    if (transformed.data) {
        transformed_ = transformed.data;
        transformed_owned_ = false;
    } else {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&transformed_), transformed_bytes),
                   "cudaMalloc EXL3 transformed workspace");
    }
    try {
        const auto constructor_failure=constructor_failure_for_test_;
        constructor_failure_for_test_=0;
        if(constructor_failure) {
            retirement_failure_stage_for_test_=constructor_failure==2?1:0;
            cuda_check(cudaErrorMemoryAllocation,"injected EXL3 accumulation allocation failure");
        }
        if (accumulation.data) {
            accum_ = accumulation.data;
            accum_owned_ = false;
        } else {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&accum_), accum_bytes),
                       "cudaMalloc EXL3 accumulation workspace");
        }
    } catch (...) {
        if(!accum_)storage_retirement_->accumulation_credit.reset();
        if(!transformed_)storage_retirement_->transform_credit.reset();
        retire_storage_fallback();
        throw;
    }
    workspace_bytes_ = requirement.owned_bytes;
    accumulation_capacity_bytes_ = accum_bytes;
    transformed_capacity_bytes_ = transformed_bytes;
    if (fast_wide_prefill_gemm_enabled_) {
        try {
            auto state = std::make_unique<FastWideGemmState>();
            if (cublasCreate(&state->handle) == CUBLAS_STATUS_SUCCESS) {
                fast_wide_gemm_state_ = std::move(state);
            } else {
                // Keep the established native route as the explicit fallback
                // when the optional handle cannot be prepared on this device.
                fast_wide_prefill_gemm_enabled_ = false;
            }
        } catch (...) {
            retire_storage_fallback();
            throw;
        }
    }
}

Exl3CudaLinearWorkspace::~Exl3CudaLinearWorkspace() {
    // Context retirement has already established final use. Retire the
    // executable before the definition and before their bound scratch storage.
    prefill_projection_graph_executable_.reset();
    prefill_projection_graph_definition_.reset();
    retire_storage_fallback();
}
void Exl3CudaLinearWorkspace::retire_storage_fallback() noexcept {
    if(!(accum_owned_ && accum_) && !(transformed_owned_ && transformed_))return;
    const auto result=exl3_retire_reconstruction_device(owner_device_,
        [](int* device) noexcept {return static_cast<int>(cudaGetDevice(device));},
        [](int device) noexcept {return static_cast<int>(cudaSetDevice(device));},
        [&]() noexcept {
            auto error=retirement_failure_stage_for_test_==1?cudaErrorUnknown:cudaSuccess;
            if(error==cudaSuccess && accum_owned_ && accum_) {
                error=cudaFree(accum_);if(error==cudaSuccess) {
                    accum_=nullptr;storage_retirement_->accumulation_credit.reset();
                }
            }
            if(error==cudaSuccess && retirement_failure_stage_for_test_==2)error=cudaErrorUnknown;
            if(error==cudaSuccess && transformed_owned_ && transformed_) {
                error=cudaFree(transformed_);if(error==cudaSuccess) {
                    transformed_=nullptr;storage_retirement_->transform_credit.reset();
                }
            }
            return static_cast<int>(error);
        },[]() noexcept {});
    if(result.released && !result.restore_error)return;
    auto* retained=storage_retirement_.release();
    retained->device=owner_device_;retained->error=result.error;retained->restore_error=result.restore_error;
    if(transformed_owned_ && transformed_) {
        retained->transformed=transformed_;retained->transformed_bytes=transformed_capacity_bytes_;
        transformed_=nullptr;
    }
    if(accum_owned_ && accum_) {
        retained->accumulation=accum_;retained->accumulation_bytes=accumulation_capacity_bytes_;
        accum_=nullptr;
    }
    auto* head=storage_quarantine_.load(std::memory_order_relaxed);
    do{retained->next=head;}while(!storage_quarantine_.compare_exchange_weak(
        head,retained,std::memory_order_release,std::memory_order_relaxed));
    quarantine_count_.fetch_add(1,std::memory_order_release);
}
void Exl3CudaLinearWorkspace::retire_owned(std::unique_ptr<Exl3CudaLinearWorkspace> owner) noexcept {
    if(!owner)return;
    owner->prefill_projection_graph_executable_.reset();
    owner->prefill_projection_graph_definition_.reset();
    const auto release_owner=[&]() noexcept {
        auto credit=std::move(owner->storage_retirement_->metadata_credit);
        owner.reset(); // Physical object and retirement record precede credit release.
    };
    if(!(owner->accum_owned_ && owner->accum_) && !(owner->transformed_owned_ && owner->transformed_)) {
        release_owner();return;
    }
    const auto result=exl3_retire_reconstruction_device(owner->owner_device_,
        [](int* device) noexcept {return static_cast<int>(cudaGetDevice(device));},
        [](int device) noexcept {return static_cast<int>(cudaSetDevice(device));},
        [&]() noexcept {
    cudaError_t error=owner->retirement_failure_stage_for_test_==1?cudaErrorUnknown:cudaSuccess;
    if(error==cudaSuccess && owner->accum_owned_ && owner->accum_) {
        error=cudaFree(owner->accum_);
        if(error==cudaSuccess) {
            owner->accum_=nullptr;owner->storage_retirement_->accumulation_credit.reset();
        }
    }
    if(error==cudaSuccess && owner->retirement_failure_stage_for_test_==2)error=cudaErrorUnknown;
    if(error==cudaSuccess && owner->transformed_owned_ && owner->transformed_) {
        error=cudaFree(owner->transformed_);
        if(error==cudaSuccess) {
            owner->transformed_=nullptr;owner->storage_retirement_->transform_credit.reset();
        }
    }
    return static_cast<int>(error);
        },[]() noexcept {});
    if(result.released && !result.restore_error){release_owner();return;}
    owner->retirement_error_=result.error;
    owner->retirement_restore_error_=result.restore_error;
    auto* retained=owner.release();
    auto* head=quarantine_.load(std::memory_order_relaxed);
    do {retained->quarantine_next_=head;}while(!quarantine_.compare_exchange_weak(
        head,retained,std::memory_order_release,std::memory_order_relaxed));
    quarantine_count_.fetch_add(1,std::memory_order_release);
}

struct Exl3CudaReconstructGemmWorkspace::Impl {
    struct ReuseSlot {
        std::uint16_t* data = nullptr;
        std::size_t capacity = 0;
        const std::uint16_t* trellis = nullptr;
        const std::int32_t* mul1 = nullptr;
        const std::uint16_t* suh = nullptr;
        const std::uint16_t* svh = nullptr;
        int in_features = 0;
        int out_features = 0;
        int K = 0;
        bool original_basis = false;
        bool assigned = false;
        bool valid = false;
        bool prefetch_pending = false;
        cudaEvent_t prefetch_fork = nullptr;
        cudaEvent_t prefetch_ready = nullptr;
    };
    int in_features = 0;
    int out_features = 0;
    int max_rows = 0;
    bool accept_transpose_shape = false;
    bool accept_all_model_shapes = false;
    std::uint16_t* transformed = nullptr;
    std::uint16_t* reconstructed = nullptr;
    float* accum = nullptr;
    void* cublas_workspace = nullptr;
    cublasHandle_t handle = nullptr;
    cublasLtHandle_t lt_handle = nullptr;
    struct LtPlan {
        cublasLtMatmulDesc_t operation = nullptr;
        cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
        cublasLtMatmulAlgo_t algorithm{};
        std::size_t workspace_bytes = 0;
    };
    std::unordered_map<std::uint64_t,LtPlan> lt_plans;
    bool k5_lt_enabled = false;
    bool large_lt_enabled = false;
    bool mxfp8_enabled = false;
    int nvfp4_mode = 0;
    bool fused_decode_enabled = true;
    int prefill_layer = -1;
    float* nvfp4_scalars = nullptr;
    bool fused_original_enabled = false;
    bool original_gdn_mlp_cache_enabled = false;
    bool fp16_compute_enabled = false;
    bool k5_layer_prefill_enabled = false;
    bool k5_scope_enabled = false;
    bool packed_direct_k6_enabled = false;
    bool packed_direct_k5_enabled = false;
    bool persistent_prefill_enabled = false;
    bool mia_prefill_fp16_enabled = false;
    int persistent_prefill_capacity[6] = {};
    std::size_t bytes = 0;
    Exl3ReconstructGemmStats stats{};
    std::vector<ReuseSlot> reuse_slots;
    cudaStream_t weight_prefetch_stream = nullptr;
    std::size_t reuse_budget = 0;
    std::size_t reuse_capacity = 0;
    std::size_t reuse_used = 0;
    bool reuse_active = false;
    std::size_t reuse_scope_count = 0;
    int diagnostic_remaining = 0;
};

std::size_t Exl3CudaReconstructGemmWorkspace::workspace_bytes_required(
    int in_features,int out_features,int rows,bool accept_transpose_shape) {
    if(in_features<=0 || out_features<=0 || rows<=0 ||
        in_features%kHadamard!=0 || out_features%kHadamard!=0)
        throw std::invalid_argument("T69 reconstruct GEMM dimensions");
    Exl3ResourceInventory::Requirement required;
    using Domain=Exl3ResourceInventory::Domain;
    const auto input=accept_transpose_shape?std::max(in_features,out_features):in_features;
    const auto output=accept_transpose_shape?std::max(in_features,out_features):out_features;
    required.add(Domain::device,rows,static_cast<std::uint64_t>(input)*sizeof(std::uint16_t));
    required.add(Domain::device,in_features,static_cast<std::uint64_t>(out_features)*sizeof(std::uint16_t));
    required.add(Domain::device,rows,
        static_cast<std::uint64_t>(output)*sizeof(float) +
        static_cast<std::uint64_t>(output / 16)*sizeof(int));
    required.add(Domain::device,1,16u*1024u*1024u);
    const auto bytes=required.units[static_cast<unsigned>(Domain::device)];
    if(bytes>std::numeric_limits<std::size_t>::max())throw std::overflow_error("T69 workspace extent overflow");
    return static_cast<std::size_t>(bytes);
}

Exl3CudaReconstructGemmWorkspace::Exl3CudaReconstructGemmWorkspace(
    int in_features, int out_features, int max_rows,
    bool accept_transpose_shape, bool accept_all_model_shapes) : impl_(nullptr) {
    const auto required_bytes=workspace_bytes_required(in_features,out_features,max_rows,accept_transpose_shape);
    impl_=new Impl{};
    if (in_features <= 0 || out_features <= 0 || max_rows <= 0 ||
        in_features % kHadamard != 0 || out_features % kHadamard != 0) {
        delete impl_; impl_ = nullptr;
        throw std::invalid_argument("T69 reconstruct GEMM dimensions");
    }
    impl_->in_features = in_features;
    impl_->out_features = out_features;
    impl_->max_rows = max_rows;
    impl_->accept_transpose_shape = accept_transpose_shape;
    impl_->accept_all_model_shapes = accept_all_model_shapes;
    const char* fused_original = std::getenv(
        "NINFER_EXL3_FAST_MIA_PARITY_FUSED_RECONSTRUCT");
    if (fused_original && std::strcmp(fused_original, "0") != 0 &&
        std::strcmp(fused_original, "1") != 0) {
        delete impl_; impl_ = nullptr;
        throw std::invalid_argument(
            "FAST Mia-parity fused reconstruct must be 0 or 1");
    }
    impl_->fused_original_enabled = accept_all_model_shapes && fused_original &&
        std::strcmp(fused_original, "1") == 0;
    const char* original_gdn_mlp=std::getenv(
        "NINFER_EXL3_FAST_ORIGINAL_GDN_MLP_CACHE");
    if(original_gdn_mlp && std::strcmp(original_gdn_mlp,"0")!=0 &&
       std::strcmp(original_gdn_mlp,"1")!=0) {
        delete impl_;impl_=nullptr;
        throw std::invalid_argument("original-basis GDN MLP cache must be 0 or 1");
    }
    impl_->original_gdn_mlp_cache_enabled=accept_all_model_shapes &&
        original_gdn_mlp && std::strcmp(original_gdn_mlp,"1")==0;
    const char* fp16_compute = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_GEMM_FP16_COMPUTE");
    if (fp16_compute && std::strcmp(fp16_compute, "0") != 0 &&
        std::strcmp(fp16_compute, "1") != 0) {
        delete impl_; impl_ = nullptr;
        throw std::invalid_argument(
            "FAST same-weights FP16-KV GEMM FP16 compute must be 0 or 1");
    }
    impl_->fp16_compute_enabled = accept_all_model_shapes && fp16_compute &&
        std::strcmp(fp16_compute, "1") == 0;
    const char* k5_layer_prefill=std::getenv(
        "NINFER_EXL3_FAST_LAYER_MAJOR_K5_RECONSTRUCT");
    if(k5_layer_prefill && std::strcmp(k5_layer_prefill,"0")!=0 &&
       std::strcmp(k5_layer_prefill,"1")!=0) {
        delete impl_; impl_=nullptr;
        throw std::invalid_argument("layer-major K5 reconstruction must be 0 or 1");
    }
    impl_->k5_layer_prefill_enabled=accept_all_model_shapes &&
        k5_layer_prefill && std::strcmp(k5_layer_prefill,"1")==0;
    const char* k5_lt=std::getenv("NINFER_EXL3_FAST_PREFILL_K5_CUBLAS_LT");
    if(k5_lt && std::strcmp(k5_lt,"0")!=0 && std::strcmp(k5_lt,"1")!=0) {
        delete impl_; impl_=nullptr;
        throw std::invalid_argument("K5 prefill cuBLASLt must be 0 or 1");
    }
    impl_->k5_lt_enabled=accept_all_model_shapes && k5_lt &&
        std::strcmp(k5_lt,"1")==0;
    const char* large_lt=std::getenv("NINFER_EXL3_FAST_PREFILL_LARGE_CUBLAS_LT");
    if(large_lt && std::strcmp(large_lt,"0")!=0 &&
        std::strcmp(large_lt,"1")!=0) {
        delete impl_; impl_=nullptr;
        throw std::invalid_argument("large prefill cuBLASLt must be 0 or 1");
    }
    impl_->large_lt_enabled=accept_all_model_shapes && large_lt &&
        std::strcmp(large_lt,"1")==0;
    // Prefill numerical policy (default 1, quality-gated; 0 = FP16 control):
    // large-M projections run as MXFP8 (E4M3 with block-32 E8M0 scales on
    // weights and activations, FP32 accumulation).
    const char* mxfp8=std::getenv("NINFER_EXL3_PREFILL_MXFP8");
    if(mxfp8 && std::strcmp(mxfp8,"0")!=0 && std::strcmp(mxfp8,"1")!=0) {
        delete impl_; impl_=nullptr;
        throw std::invalid_argument("NINFER_EXL3_PREFILL_MXFP8 must be 0 or 1");
    }
    impl_->mxfp8_enabled=accept_all_model_shapes &&
        (!mxfp8 || std::strcmp(mxfp8,"1")==0);
    // NVFP4 prefill numerical policy (default 2 = MLP gate/up/down only,
    // quality-gated; 0 = MXFP8 control; 1 = all admitted prefill projections,
    // rejected on held-out quality: +0.039 overall, prose +0.068).
    // Requires the MXFP8 route.
    const char* nvfp4=std::getenv("NINFER_EXL3_PREFILL_NVFP4");
    if(nvfp4 && std::strcmp(nvfp4,"0")!=0 && std::strcmp(nvfp4,"1")!=0 &&
       std::strcmp(nvfp4,"2")!=0 && std::strcmp(nvfp4,"3")!=0) {
        delete impl_; impl_=nullptr;
        throw std::invalid_argument("NINFER_EXL3_PREFILL_NVFP4 must be 0, 1, 2 or 3");
    }
    impl_->nvfp4_mode=impl_->mxfp8_enabled ? (nvfp4 ? std::atoi(nvfp4) : 2) : 0;
    const char* fused_decode=std::getenv("NINFER_EXL3_PREFILL_FUSED_DECODE");
    if(fused_decode && std::strcmp(fused_decode,"0")!=0 && std::strcmp(fused_decode,"1")!=0) {
        delete impl_; impl_=nullptr;
        throw std::invalid_argument("NINFER_EXL3_PREFILL_FUSED_DECODE must be 0 or 1");
    }
    impl_->fused_decode_enabled=!fused_decode || std::strcmp(fused_decode,"1")==0;
    const char* packed_direct_k6 = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_PACKED_DIRECT_K6");
    if (packed_direct_k6 && std::strcmp(packed_direct_k6, "0") != 0 &&
        std::strcmp(packed_direct_k6, "1") != 0) {
        delete impl_; impl_ = nullptr;
        throw std::invalid_argument(
            "FAST same-weights FP16-KV packed direct K6 must be 0 or 1");
    }
    impl_->packed_direct_k6_enabled = accept_all_model_shapes &&
        packed_direct_k6 && std::strcmp(packed_direct_k6, "1") == 0;
    const char* packed_direct_k5 = std::getenv(
        "NINFER_EXL3_FAST_GDN_BULK_MLP_PACKED_K5");
    if (packed_direct_k5 && std::strcmp(packed_direct_k5, "0") != 0 &&
        std::strcmp(packed_direct_k5, "1") != 0) {
        delete impl_; impl_ = nullptr;
        throw std::invalid_argument("GDN bulk packed K5 must be 0 or 1");
    }
    impl_->packed_direct_k5_enabled = accept_all_model_shapes &&
        packed_direct_k5 && std::strcmp(packed_direct_k5, "1") == 0;
    const char* persistent_prefill = std::getenv(
        "NINFER_EXL3_FAST_NATIVE_PERSISTENT_PREFILL");
    if (persistent_prefill && std::strcmp(persistent_prefill, "0") != 0 &&
        std::strcmp(persistent_prefill, "1") != 0) {
        delete impl_; impl_ = nullptr;
        throw std::invalid_argument(
            "FAST native persistent prefill must be 0 or 1");
    }
    impl_->persistent_prefill_enabled = accept_all_model_shapes &&
        persistent_prefill && std::strcmp(persistent_prefill, "1") == 0;
    const char* mia_prefill_fp16 = std::getenv(
        "NINFER_EXL3_FAST_NATIVE_MIA_PREFILL_FP16");
    if (mia_prefill_fp16 && std::strcmp(mia_prefill_fp16, "0") != 0 &&
        std::strcmp(mia_prefill_fp16, "1") != 0) {
        delete impl_; impl_ = nullptr;
        throw std::invalid_argument(
            "FAST native Mia-shaped FP16 prefill must be 0 or 1");
    }
    // This is an independently labeled same-weight FP16-output cooperative
    // prefill lane. It is never implied by the historical FP32 adapter.
    impl_->mia_prefill_fp16_enabled = accept_all_model_shapes &&
        mia_prefill_fp16 && std::strcmp(mia_prefill_fp16, "1") == 0;
    if (impl_->persistent_prefill_enabled || impl_->mia_prefill_fp16_enabled) {
        int device = 0;
        cudaDeviceProp properties{};
        cuda_check(cudaGetDevice(&device),
                   "query numeric persistent prefill device");
        cuda_check(cudaGetDeviceProperties(&properties, device),
                   "query numeric persistent prefill properties");
        const auto query_persistent_prefill = [&](std::size_t bytes,
                                                  auto kernel, int threads) {
            cuda_check(cudaFuncSetAttribute(
                           kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(bytes)),
                       "set numeric persistent prefill shared-memory limit");
            int active = 0;
            cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                           &active, kernel, threads, bytes),
                       "query numeric persistent prefill occupancy");
            return active > 0 ? active * properties.multiProcessorCount : 0;
        };
        if (impl_->mia_prefill_fp16_enabled) {
            impl_->persistent_prefill_capacity[0] = query_persistent_prefill(
                exl3_native_persistent_smem_bytes<6, 16, 512>(),
                exl3_native_mia_prefill_fp16_kernel<6, 16, 512>, 256);
            impl_->persistent_prefill_capacity[1] = query_persistent_prefill(
                exl3_native_persistent_smem_bytes<7, 16, 512>(),
                exl3_native_mia_prefill_fp16_kernel<7, 16, 512>, 256);
            impl_->persistent_prefill_capacity[2] = query_persistent_prefill(
                exl3_native_persistent_smem_bytes<6, 32, 256>(),
                exl3_native_mia_prefill_fp16_kernel<6, 32, 256>, 512);
            impl_->persistent_prefill_capacity[3] = query_persistent_prefill(
                exl3_native_persistent_smem_bytes<7, 32, 128>(),
                exl3_native_mia_prefill_fp16_kernel<7, 32, 128, true>, 512);
        } else {
            impl_->persistent_prefill_capacity[0] = query_persistent_prefill(
                exl3_native_persistent_smem_bytes<6, 16, 512>(),
                exl3_native_persistent_prefill_kernel<6, 16, 512>, 256);
            impl_->persistent_prefill_capacity[1] = query_persistent_prefill(
                exl3_native_persistent_smem_bytes<7, 16, 512>(),
                exl3_native_persistent_prefill_kernel<7, 16, 512>, 256);
            impl_->persistent_prefill_capacity[2] = query_persistent_prefill(
                exl3_native_persistent_smem_bytes<6, 32, 256>(),
                exl3_native_persistent_prefill_kernel<6, 32, 256>, 512);
            impl_->persistent_prefill_capacity[3] = query_persistent_prefill(
                exl3_native_persistent_smem_bytes<7, 32, 128>(),
                exl3_native_persistent_prefill_kernel<7, 32, 128, true>, 512);
        }
    }
    const int maximum_features = accept_transpose_shape
        ? std::max(in_features, out_features) : 0;
    const std::size_t transformed_bytes = static_cast<std::size_t>(max_rows) *
        (accept_transpose_shape ? maximum_features : in_features) *
        sizeof(std::uint16_t);
    const std::size_t reconstructed_bytes = static_cast<std::size_t>(in_features) *
        out_features * sizeof(std::uint16_t);
    const std::size_t accumulation_width = accept_transpose_shape
        ? static_cast<std::size_t>(maximum_features)
        : static_cast<std::size_t>(out_features);
    const std::size_t accum_bytes = static_cast<std::size_t>(max_rows) *
        accumulation_width * sizeof(float) +
        static_cast<std::size_t>(max_rows) *
            static_cast<std::size_t>(accumulation_width / 16) * sizeof(int);
    constexpr std::size_t cublas_bytes = 16u * 1024u * 1024u;
    try {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&impl_->transformed),
                              transformed_bytes),
                   "T69 allocate transformed input");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&impl_->reconstructed),
                              reconstructed_bytes),
                   "T69 allocate reconstructed weight");
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&impl_->accum), accum_bytes),
                   "T69 allocate GEMM output");
        cuda_check(cudaMalloc(&impl_->cublas_workspace, cublas_bytes),
                   "T69 allocate cuBLAS workspace");
        if (impl_->nvfp4_mode) {
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&impl_->nvfp4_scalars),
                                  4 * sizeof(float)), "allocate NVFP4 scalars");
            cuda_check(cudaMemset(impl_->nvfp4_scalars, 0, 4 * sizeof(float)),
                       "clear NVFP4 scalars");
        }
        cublas_check(cublasCreate(&impl_->handle), "T69 create cuBLAS handle");
        if (impl_->k5_lt_enabled || impl_->large_lt_enabled || impl_->mxfp8_enabled)
            cublas_check(cublasLtCreate(&impl_->lt_handle),
                         "create K5 prefill cuBLASLt handle");
        cublas_check(cublasSetPointerMode(impl_->handle, CUBLAS_POINTER_MODE_HOST),
                     "T69 set cuBLAS host pointer mode");
        cublas_check(cublasSetWorkspace(impl_->handle, impl_->cublas_workspace,
                                        cublas_bytes),
                     "T69 bind cuBLAS workspace");
    } catch (...) {
        if (impl_->lt_handle) cublasLtDestroy(impl_->lt_handle);
        if (impl_->handle) cublasDestroy(impl_->handle);
        if (impl_->cublas_workspace) cudaFree(impl_->cublas_workspace);
        if (impl_->accum) cudaFree(impl_->accum);
        if (impl_->reconstructed) cudaFree(impl_->reconstructed);
        if (impl_->transformed) cudaFree(impl_->transformed);
        delete impl_; impl_ = nullptr;
        throw;
    }
    impl_->bytes = required_bytes;
    impl_->stats.workspace_bytes = impl_->bytes;
}

Exl3CudaReconstructGemmWorkspace::~Exl3CudaReconstructGemmWorkspace() {
    if (!impl_) return;
    if (impl_->weight_prefetch_stream)
        (void)cudaStreamSynchronize(impl_->weight_prefetch_stream);
    for (auto& [key, plan] : impl_->lt_plans) {
        if (plan.a) cublasLtMatrixLayoutDestroy(plan.a);
        if (plan.b) cublasLtMatrixLayoutDestroy(plan.b);
        if (plan.c) cublasLtMatrixLayoutDestroy(plan.c);
        if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
    }
    if (impl_->lt_handle) cublasLtDestroy(impl_->lt_handle);
    for (auto& slot : impl_->reuse_slots) {
        if (slot.prefetch_fork) (void)cudaEventDestroy(slot.prefetch_fork);
        if (slot.prefetch_ready) (void)cudaEventDestroy(slot.prefetch_ready);
        if (slot.data) cudaFree(slot.data);
    }
    if (impl_->weight_prefetch_stream)
        (void)cudaStreamDestroy(impl_->weight_prefetch_stream);
    if (impl_->handle) cublasDestroy(impl_->handle);
    if (impl_->cublas_workspace) cudaFree(impl_->cublas_workspace);
    if (impl_->nvfp4_scalars) cudaFree(impl_->nvfp4_scalars);
    if (impl_->accum) cudaFree(impl_->accum);
    if (impl_->reconstructed) cudaFree(impl_->reconstructed);
    if (impl_->transformed) cudaFree(impl_->transformed);
    delete impl_;
}

std::size_t Exl3CudaReconstructGemmWorkspace::workspace_bytes() const noexcept {
    return impl_ ? impl_->bytes : 0;
}

Exl3ReconstructGemmStats Exl3CudaReconstructGemmWorkspace::stats() const noexcept {
    return impl_ ? impl_->stats : Exl3ReconstructGemmStats{};
}

void Exl3CudaReconstructGemmWorkspace::begin_layer_reuse(
    std::size_t max_cached_bytes, bool allow_k5) {
    if (!impl_ || impl_->reuse_active || !impl_->accept_all_model_shapes || !max_cached_bytes)
        throw std::invalid_argument("layer-major weight reuse admission");
    impl_->reuse_active = true;
    impl_->k5_scope_enabled = allow_k5 && impl_->k5_layer_prefill_enabled;
    impl_->reuse_budget = max_cached_bytes;
    impl_->reuse_used = 0;
    const char* profile=std::getenv("NINFER_EXL3_TEST_LAYER_MAJOR_PROFILE");
    impl_->diagnostic_remaining = impl_->reuse_scope_count==0 && profile &&
        std::strcmp(profile,"1")==0 ? 16 : 0;
    ++impl_->reuse_scope_count;
    for (auto& slot : impl_->reuse_slots) {
        slot.assigned = false;
        slot.valid = false;
        slot.prefetch_pending = false;
    }
}

void Exl3CudaReconstructGemmWorkspace::set_prefill_layer(int layer) noexcept {
    if (impl_) impl_->prefill_layer = layer;
}

void Exl3CudaReconstructGemmWorkspace::end_layer_reuse() noexcept {
    if (!impl_) return;
    impl_->prefill_layer = -1;
    if (impl_->weight_prefetch_stream) {
        for (const auto& slot : impl_->reuse_slots) {
            if (slot.prefetch_pending) {
                (void)cudaStreamSynchronize(impl_->weight_prefetch_stream);
                break;
            }
        }
    }
    impl_->reuse_active = false;
    impl_->k5_scope_enabled = false;
    impl_->reuse_used = 0;
    impl_->diagnostic_remaining = 0;
    for (auto& slot : impl_->reuse_slots) {
        slot.assigned = false;
        slot.valid = false;
        slot.prefetch_pending = false;
    }
}

bool Exl3CudaReconstructGemmWorkspace::prefetch_numeric_weight(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,int rows,cudaStream_t stream) {
    // The FP16 prefetch slot format is not the MXFP8 slot format.
    if (!impl_ || !impl_->reuse_active || !impl_->k5_scope_enabled ||
        impl_->mxfp8_enabled ||
        metadata.K!=5 || rows<1024 || !supports(metadata,rows) ||
        !weights.trellis || !weights.mul1 || !weights.suh || !weights.svh)
        return false;
    const std::size_t weight_bytes=static_cast<std::size_t>(metadata.in_features)*
        static_cast<std::size_t>(metadata.out_features)*sizeof(std::uint16_t);
    Impl::ReuseSlot* slot=nullptr;
    for (std::size_t index=0;index<impl_->reuse_used;++index) {
        auto& candidate=impl_->reuse_slots[index];
        if (candidate.assigned && candidate.trellis==weights.trellis &&
            candidate.mul1==weights.mul1 && candidate.suh==weights.suh &&
            candidate.svh==weights.svh &&
            candidate.in_features==metadata.in_features &&
            candidate.out_features==metadata.out_features &&
            candidate.K==metadata.K && !candidate.original_basis) {
            slot=&candidate;
            break;
        }
    }
    if (!slot) {
        if (impl_->reuse_used==impl_->reuse_slots.size())
            impl_->reuse_slots.emplace_back();
        slot=&impl_->reuse_slots[impl_->reuse_used++];
        slot->assigned=true;
        slot->valid=false;
        slot->trellis=weights.trellis;
        slot->mul1=weights.mul1;
        slot->suh=weights.suh;
        slot->svh=weights.svh;
        slot->in_features=metadata.in_features;
        slot->out_features=metadata.out_features;
        slot->K=metadata.K;
        slot->original_basis=false;
        if (slot->capacity<weight_bytes &&
            impl_->reuse_capacity-slot->capacity+weight_bytes<=impl_->reuse_budget) {
            if (slot->data) {
                cuda_check(cudaFree(slot->data),"retire smaller prefetched weight slot");
                impl_->reuse_capacity-=slot->capacity;
                slot->data=nullptr;
                slot->capacity=0;
            }
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot->data),weight_bytes),
                "allocate bounded prefetched weight slot");
            slot->capacity=weight_bytes;
            impl_->reuse_capacity+=weight_bytes;
            impl_->stats.cached_weight_capacity_bytes=impl_->reuse_capacity;
        }
    }
    if (!slot->data || slot->capacity<weight_bytes || slot->valid ||
        slot->prefetch_pending)
        return false;
    if (!impl_->weight_prefetch_stream)
        cuda_check(cudaStreamCreateWithFlags(&impl_->weight_prefetch_stream,
            cudaStreamNonBlocking),"create K5 weight prefetch stream");
    if (!slot->prefetch_fork)
        cuda_check(cudaEventCreateWithFlags(&slot->prefetch_fork,
            cudaEventDisableTiming),"create K5 prefetch fork event");
    if (!slot->prefetch_ready)
        cuda_check(cudaEventCreateWithFlags(&slot->prefetch_ready,
            cudaEventDisableTiming),"create K5 prefetch ready event");
    cuda_check(cudaEventRecord(slot->prefetch_fork,stream),
        "record K5 weight prefetch fork");
    cuda_check(cudaStreamWaitEvent(impl_->weight_prefetch_stream,
        slot->prefetch_fork,0),"join K5 weight prefetch fork");
    slot->prefetch_pending=true;
    exl3_reconstruct_transformed_weight_kernel<5><<<
        dim3(metadata.out_features/16,metadata.in_features/16),256,0,
        impl_->weight_prefetch_stream>>>(weights.trellis,weights.mul1,
            slot->data,metadata.in_features,metadata.out_features);
    cuda_check(cudaGetLastError(),"launch K5 weight prefetch");
    cuda_check(cudaEventRecord(slot->prefetch_ready,
        impl_->weight_prefetch_stream),"record K5 weight prefetch ready");
    ++impl_->stats.prefetched_weight_submissions;
    ++impl_->stats.reconstructed_weight_calls;
    impl_->stats.reconstructed_weight_bytes+=weight_bytes;
    return true;
}

bool Exl3CudaReconstructGemmWorkspace::supports(
    const Exl3CudaLinearMetadata& metadata, int rows) const noexcept {
    if (!impl_ || rows <= 0 || rows > impl_->max_rows) return false;
    if (impl_->accept_all_model_shapes) {
        const bool bounded_shape = metadata.in_features > 0 &&
            metadata.out_features > 0 &&
            metadata.in_features <= impl_->in_features &&
            metadata.out_features <= impl_->out_features &&
            metadata.in_features % kHadamard == 0 &&
            metadata.out_features % kHadamard == 0;
        const bool k5_layer=metadata.K==5 &&
            (impl_->k5_scope_enabled || impl_->mxfp8_enabled) &&
            impl_->reuse_active && rows>=256;
        return bounded_shape && (metadata.K == 6 || metadata.K == 7 || k5_layer) &&
            metadata.mul1 && !metadata.mcg && !metadata.has_bias;
    }
    const bool primary_shape =
        metadata.in_features == impl_->in_features &&
        metadata.out_features == impl_->out_features;
    const bool transpose_shape = impl_->accept_transpose_shape &&
        metadata.in_features == impl_->out_features &&
        metadata.out_features == impl_->in_features;
    return (primary_shape || transpose_shape) &&
        (metadata.K == 6 || metadata.K == 7) && metadata.mul1 &&
        !metadata.mcg && !metadata.has_bias;
}

bool Exl3CudaReconstructGemmWorkspace::accepts_all_model_shapes() const noexcept {
    return impl_ && impl_->accept_all_model_shapes;
}

bool Exl3CudaReconstructGemmWorkspace::supports_fused_gate_up_down() const noexcept {
    return accepts_all_model_shapes() && !impl_->fused_original_enabled &&
        !impl_->original_gdn_mlp_cache_enabled;
}

void Exl3CudaReconstructGemmWorkspace::forward_numeric_gate_up_down(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,
    const std::uint16_t* gate,
    const std::uint16_t* up,
    std::uint16_t* activation,
    std::uint16_t* output,
    int rows,
    cudaStream_t stream) {
    if (!supports_fused_gate_up_down() || !up || !activation ||
        metadata.in_features != 17408 || metadata.out_features != 5120)
        throw std::invalid_argument("GDN fused gate/up down admission");
    forward_numeric_candidate(weights,metadata,gate,output,rows,stream,
                              nullptr,up,activation);
}

void Exl3CudaReconstructGemmWorkspace::forward_numeric_candidate(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,
    const std::uint16_t* input,
    std::uint16_t* output,
    int rows,
    cudaStream_t stream,
    Exl3ReconstructGemmPhaseTiming* timing,
    const std::uint16_t* up,
    std::uint16_t* activation,
    const std::uint16_t* residual,
    std::uint16_t* down_trace,
    int trace_row_base) {
    if (!supports(metadata, rows) || !weights.trellis ||
        !weights.suh || !weights.svh || !weights.mul1 || !input || !output ||
        ((up == nullptr) != (activation == nullptr)) ||
        (down_trace && (!residual || trace_row_base < 0 ||
                        trace_row_base >= rows))) {
        throw std::invalid_argument("T69 reconstruct GEMM contract");
    }
    Exl3ReconstructGemmPhaseTiming diagnostic_timing{};
    const bool diagnostic=timing==nullptr && impl_->reuse_active &&
        impl_->diagnostic_remaining>0;
    if (diagnostic) {
        --impl_->diagnostic_remaining;
        timing=&diagnostic_timing;
    }
    struct DiagnosticPrint {
        bool active;
        const Exl3CudaLinearMetadata& metadata;
        int rows;
        const Exl3ReconstructGemmPhaseTiming& value;
        ~DiagnosticPrint() {
            if (!active) return;
            std::fprintf(stderr,
                "LAYER_MAJOR_PROJECTION_SAMPLE K=%d in=%d out=%d rows=%d "
                "input_us=%.2f reconstruct_us=%.2f gemm_us=%.2f output_us=%.2f total_us=%.2f\n",
                metadata.K,metadata.in_features,metadata.out_features,rows,
                value.input_transform_us,value.reconstruct_us,value.gemm_us,
                value.output_transform_us,value.total_us);
        }
    } diagnostic_print{diagnostic,metadata,rows,diagnostic_timing};
    cudaEvent_t events[5]{};
    if (timing) {
        for (auto& event : events)
            cuda_check(cudaEventCreate(&event), "T69 create phase event");
        cuda_check(cudaEventRecord(events[0], stream), "T69 begin phase timing");
    }
    const bool original_gdn_mlp=impl_->original_gdn_mlp_cache_enabled &&
        impl_->reuse_active && rows>=1024 &&
        ((metadata.in_features==5120 && metadata.out_features==17408) ||
         (metadata.in_features==17408 && metadata.out_features==5120));
    const bool fused_original = (impl_->fused_original_enabled || original_gdn_mlp) &&
        metadata.K!=5 && rows >= 1024 &&
        metadata.in_features % kHadamard == 0 &&
        metadata.out_features % kHadamard == 0;
    if (up && fused_original)
        throw std::invalid_argument("original-basis GEMM cannot consume transformed gate/up input");
    const char* fast_same_weights_fp16kv = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL");
    const bool fast_fp16_destination = impl_->accept_all_model_shapes &&
        rows > 1 && fast_same_weights_fp16kv &&
        std::strcmp(fast_same_weights_fp16kv, "1") == 0;
    if (residual && (metadata.in_features != 17408 ||
                     metadata.out_features != 5120 || !fast_fp16_destination ||
                     fused_original || up || activation || residual == output ||
                     impl_->persistent_prefill_enabled ||
                     impl_->mia_prefill_fp16_enabled ||
                     impl_->packed_direct_k6_enabled))
        throw std::invalid_argument("bulk down residual epilogue admission");
    const bool fp16_compute = impl_->fp16_compute_enabled &&
        fast_fp16_destination && !fused_original && metadata.K!=5;
    if (!fused_original) {
        if (up) {
            launch_input_hadamard<kHadamard,false,true>(stream,
                    input,weights.suh,impl_->transformed,rows,
                    metadata.in_features,up,activation);
            ++impl_->stats.fused_gate_up_down_calls;
            impl_->stats.fused_gate_up_down_rows += static_cast<std::uint64_t>(rows);
        } else {
            launch_input_hadamard<kHadamard>(stream,
                    input, weights.suh, impl_->transformed, rows, metadata.in_features);
        }
        cuda_check(cudaGetLastError(), "T69 input Hadamard");
    }
    if (timing) cuda_check(cudaEventRecord(events[1], stream), "T69 input event");

    // Separately labeled native same-weight FP16-KV prefill. This branch is
    // intentionally owned by the existing numeric prefill workspace because
    // that is the reached physical-C1 caller; it consumes the packed trellis
    // directly after the canonical input transform and never reconstructs or
    // re-quantizes the weights. The first accumulation plane stays FP32 and a
    // reserved tail of that allocation carries the cooperative row locks.
    const bool persistent_prefill_down =
        metadata.in_features == 17408 && metadata.out_features == 5120;
    const bool persistent_prefill_shape =
        (metadata.in_features == 5120 &&
            (metadata.out_features == 6144 || metadata.out_features == 10240 ||
             metadata.out_features == 12288 || metadata.out_features == 17408)) ||
        (metadata.in_features == 6144 && metadata.out_features == 5120) ||
        persistent_prefill_down;
    const int persistent_prefill_capacity_index =
        persistent_prefill_down ? (metadata.K == 6 ? 2 : 3)
                                : (metadata.K == 6 ? 0 : 1);
    const int persistent_prefill_chunks = (rows + 15) / 16;
    const bool persistent_prefill_candidate =
        impl_->persistent_prefill_enabled && !fused_original && rows > 16 &&
        rows <= impl_->max_rows && (metadata.K == 6 || metadata.K == 7) &&
        metadata.mcg == false && metadata.mul1 && !metadata.has_bias &&
        persistent_prefill_shape &&
        impl_->persistent_prefill_capacity[persistent_prefill_capacity_index] >=
            persistent_prefill_chunks;
    if (persistent_prefill_candidate) {
        const bool k7 = metadata.K == 7;
        const int tile_k = persistent_prefill_down ? 32 : 16;
        const int tile_n = persistent_prefill_down ? (k7 ? 128 : 256) : 512;
        const int tiles = (metadata.in_features / tile_k) *
            (metadata.out_features / tile_n);
        const int grid_x = std::max(
            1, std::min(tiles,
                impl_->persistent_prefill_capacity[persistent_prefill_capacity_index] /
                    persistent_prefill_chunks));
        const int block_threads = tile_k == 16 ? 256 : 512;
        const std::size_t shared_bytes =
            metadata.K == 6
                ? (persistent_prefill_down
                       ? exl3_native_persistent_smem_bytes<6, 32, 256>()
                       : exl3_native_persistent_smem_bytes<6, 16, 512>())
                : (persistent_prefill_down
                       ? exl3_native_persistent_smem_bytes<7, 32, 128>()
                       : exl3_native_persistent_smem_bytes<7, 16, 512>());
        const std::uint16_t* transformed = impl_->transformed;
        const std::uint16_t* trellis = weights.trellis;
        float* result = impl_->accum;
        int row_count = rows;
        int input_features = metadata.in_features;
        int output_features = metadata.out_features;
        const int lock_width = impl_->accept_transpose_shape
            ? std::max(impl_->in_features, impl_->out_features)
            : impl_->out_features;
        int* locks = reinterpret_cast<int*>(
            impl_->accum + static_cast<std::size_t>(impl_->max_rows) * lock_width);
        const half* svh = reinterpret_cast<const half*>(weights.svh);
        cuda_check(cudaMemsetAsync(
                       locks, 0,
                       static_cast<std::size_t>(persistent_prefill_chunks) *
                           static_cast<std::size_t>(output_features / 16) *
                           sizeof(int), stream),
                   "clear numeric persistent prefill locks");
        void* kernel_args[] = {&transformed, &trellis, &result, &row_count,
                               &input_features, &output_features, &locks, &svh};
        void* kernel = metadata.K == 6
            ? (persistent_prefill_down
                   ? reinterpret_cast<void*>(
                         exl3_native_persistent_prefill_kernel<6, 32, 256>)
                   : reinterpret_cast<void*>(
                         exl3_native_persistent_prefill_kernel<6, 16, 512>))
            : (persistent_prefill_down
                   ? reinterpret_cast<void*>(
                         exl3_native_persistent_prefill_kernel<7, 32, 128, true>)
                   : reinterpret_cast<void*>(
                         exl3_native_persistent_prefill_kernel<7, 16, 512>));
        if (timing)
            cuda_check(cudaEventRecord(events[2], stream),
                       "T69 persistent prefill begin");
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel,
                       dim3(grid_x, persistent_prefill_chunks),
                       dim3(block_threads), kernel_args, shared_bytes, stream),
                   "launch numeric persistent prefill GEMM");
        if (timing)
            cuda_check(cudaEventRecord(events[3], stream),
                       "T69 persistent prefill GEMM event");
        launch_output_hadamard(stream,impl_->accum, weights.svh, output, rows,
                      metadata.out_features);
        cuda_check(cudaGetLastError(), "T69 persistent prefill output Hadamard");
        if (timing) {
            cuda_check(cudaEventRecord(events[4], stream),
                       "T69 persistent prefill output event");
            cuda_check(cudaEventSynchronize(events[4]),
                       "T69 resolve persistent prefill timing");
            float ms[4]{};
            for (int i = 0; i < 4; ++i)
                cuda_check(cudaEventElapsedTime(&ms[i], events[i], events[i + 1]),
                           "T69 resolve persistent prefill phase timing");
            timing->input_transform_us = ms[0] * 1000.0;
            timing->reconstruct_us = ms[1] * 1000.0;
            timing->gemm_us = ms[2] * 1000.0;
            timing->output_transform_us = ms[3] * 1000.0;
            timing->total_us = (ms[0] + ms[1] + ms[2] + ms[3]) * 1000.0;
            for (auto event : events) cudaEventDestroy(event);
        }
        ++impl_->stats.calls;
        impl_->stats.rows += static_cast<std::uint64_t>(rows);
        if (metadata.K == 6) ++impl_->stats.k6_calls;
        else ++impl_->stats.k7_calls;
        ++impl_->stats.persistent_prefill_calls;
        impl_->stats.persistent_prefill_rows += static_cast<std::uint64_t>(rows);
        return;
    }

    // Same-weight Mia-shaped FP16-output prefill. The cooperative grid owns
    // all K/N slices once and advances through the complete row batch with a
    // grid barrier, matching the comparator's large-M outer contract while
    // keeping VoidInfer's canonical input-transform and workspace ownership.
    const bool mia_prefill_candidate =
        impl_->mia_prefill_fp16_enabled && !fused_original && rows > 16 &&
        rows <= impl_->max_rows && (metadata.K == 6 || metadata.K == 7) &&
        metadata.mcg == false && metadata.mul1 && !metadata.has_bias &&
        persistent_prefill_shape &&
        impl_->persistent_prefill_capacity[persistent_prefill_capacity_index] >=
            persistent_prefill_chunks;
    if (mia_prefill_candidate) {
        const bool k7 = metadata.K == 7;
        const int tile_k = persistent_prefill_down ? 32 : 16;
        const int tile_n = persistent_prefill_down ? (k7 ? 128 : 256) : 512;
        const int tiles = (metadata.in_features / tile_k) *
            (metadata.out_features / tile_n);
        const int grid_x = std::max(
            1, std::min(tiles,
                impl_->persistent_prefill_capacity[persistent_prefill_capacity_index]));
        const int block_threads = tile_k == 16 ? 256 : 512;
        const std::size_t shared_bytes =
            metadata.K == 6
                ? (persistent_prefill_down
                       ? exl3_native_persistent_smem_bytes<6, 32, 256>()
                       : exl3_native_persistent_smem_bytes<6, 16, 512>())
                : (persistent_prefill_down
                       ? exl3_native_persistent_smem_bytes<7, 32, 128>()
                       : exl3_native_persistent_smem_bytes<7, 16, 512>());
        const std::uint16_t* transformed = impl_->transformed;
        const std::uint16_t* trellis = weights.trellis;
        half* result = reinterpret_cast<half*>(output);
        int row_count = rows;
        int input_features = metadata.in_features;
        int output_features = metadata.out_features;
        const int lock_width = impl_->accept_transpose_shape
            ? std::max(impl_->in_features, impl_->out_features)
            : impl_->out_features;
        int* locks = reinterpret_cast<int*>(
            impl_->accum + static_cast<std::size_t>(impl_->max_rows) * lock_width);
        const half* svh = reinterpret_cast<const half*>(weights.svh);
        cuda_check(cudaMemsetAsync(
                       locks, 0,
                       static_cast<std::size_t>(output_features / 16) * sizeof(int),
                       stream),
                   "clear native Mia-shaped prefill locks");
        void* kernel_args[] = {&transformed, &trellis, &result, &row_count,
                               &input_features, &output_features, &locks, &svh};
        void* kernel = metadata.K == 6
            ? (persistent_prefill_down
                   ? reinterpret_cast<void*>(
                         exl3_native_mia_prefill_fp16_kernel<6, 32, 256>)
                   : reinterpret_cast<void*>(
                         exl3_native_mia_prefill_fp16_kernel<6, 16, 512>))
            : (persistent_prefill_down
                   ? reinterpret_cast<void*>(
                         exl3_native_mia_prefill_fp16_kernel<7, 32, 128, true>)
                   : reinterpret_cast<void*>(
                         exl3_native_mia_prefill_fp16_kernel<7, 16, 512>));
        if (timing)
            cuda_check(cudaEventRecord(events[2], stream),
                       "T69 Mia-shaped prefill begin");
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(grid_x), dim3(block_threads), kernel_args,
                       shared_bytes, stream),
                   "launch native Mia-shaped FP16 prefill GEMM");
        cuda_check(cudaGetLastError(),
                   "native Mia-shaped FP16 prefill launch");
        if (timing) {
            cuda_check(cudaEventRecord(events[3], stream),
                       "T69 Mia-shaped prefill GEMM event");
            cuda_check(cudaEventRecord(events[4], stream),
                       "T69 Mia-shaped prefill output event");
            cuda_check(cudaEventSynchronize(events[4]),
                       "T69 resolve Mia-shaped prefill timing");
            float ms[4]{};
            for (int i = 0; i < 4; ++i)
                cuda_check(cudaEventElapsedTime(&ms[i], events[i], events[i + 1]),
                           "T69 resolve Mia-shaped prefill phase timing");
            timing->input_transform_us = ms[0] * 1000.0;
            timing->reconstruct_us = ms[1] * 1000.0;
            timing->gemm_us = ms[2] * 1000.0;
            timing->output_transform_us = ms[3] * 1000.0;
            timing->total_us = (ms[0] + ms[1] + ms[2] + ms[3]) * 1000.0;
            for (auto event : events) cudaEventDestroy(event);
        }
        ++impl_->stats.calls;
        impl_->stats.rows += static_cast<std::uint64_t>(rows);
        if (metadata.K == 6) ++impl_->stats.k6_calls;
        else ++impl_->stats.k7_calls;
        ++impl_->stats.mia_prefill_fp16_calls;
        impl_->stats.mia_prefill_fp16_rows += static_cast<std::uint64_t>(rows);
        return;
    }

    // Only the reached target GDN bulk gate/up/down shapes may use this K5
    // leaf. The packed tile is decoded in the same represented transformed
    // basis as reconstruction, then each 16-row MMA group accumulates in FP32.
    // The reduction/output kernel casts that GEMM destination to FP16 before
    // the existing SVH/Hadamard boundary. No full FP16 weight slab is written.
    const bool packed_direct_k5_gate_up =
        metadata.in_features == 5120 && metadata.out_features == 17408;
    const bool packed_direct_k5_down =
        metadata.in_features == 17408 && metadata.out_features == 5120;
    const bool packed_direct_k5_candidate =
        impl_->packed_direct_k5_enabled && fast_fp16_destination &&
        !fused_original && !residual && !up && !activation &&
        rows >= 1024 && rows <= impl_->max_rows && metadata.K == 5 &&
        !metadata.mcg && !metadata.has_bias && weights.trellis &&
        weights.mul1 && weights.suh && weights.svh &&
        (packed_direct_k5_gate_up || packed_direct_k5_down);
    if (packed_direct_k5_candidate) {
        constexpr int output_tiles_per_block = 64;
        constexpr int split_count = 1;
        const int output_blocks = metadata.out_features /
            (16 * output_tiles_per_block);
        const std::size_t shared_bytes = 512u * sizeof(half) +
            2u * output_tiles_per_block * 16u * 5u * sizeof(std::uint16_t);
        if (timing)
            cuda_check(cudaEventRecord(events[2], stream),
                       "K5 packed direct begin event");
        exl3_prefill_direct_async_a_kernel<5, 64, 2><<<
            dim3(output_blocks, (rows + 15) / 16), kThreads,
            shared_bytes, stream>>>(
                impl_->transformed, weights.trellis, weights.mul1,
                impl_->accum, rows, metadata.in_features,
                metadata.out_features, split_count);
        cuda_check(cudaGetLastError(), "launch GDN bulk packed K5 GEMM");
        if (timing)
            cuda_check(cudaEventRecord(events[3], stream),
                       "K5 packed direct GEMM event");
        launch_prefill_reduce_output<false, false, false, true>(stream,impl_->accum, weights.svh, output, rows,
                         metadata.out_features, split_count);
        cuda_check(cudaGetLastError(), "launch GDN bulk packed K5 output");
        ++impl_->stats.calls;
        ++impl_->stats.k5_calls;
        impl_->stats.rows += static_cast<std::uint64_t>(rows);
        impl_->stats.packed_direct_k5_rows += static_cast<std::uint64_t>(rows);
        if (packed_direct_k5_gate_up)
            ++impl_->stats.packed_direct_k5_gate_up_calls;
        else
            ++impl_->stats.packed_direct_k5_down_calls;
        if (timing) {
            cuda_check(cudaEventRecord(events[4], stream),
                       "K5 packed direct output event");
            cuda_check(cudaEventSynchronize(events[4]),
                       "resolve K5 packed direct timing");
            float ms[4]{};
            for (int i = 0; i < 4; ++i)
                cuda_check(cudaEventElapsedTime(&ms[i], events[i], events[i + 1]),
                           "resolve K5 packed direct phase");
            timing->input_transform_us = ms[0] * 1000.0;
            timing->reconstruct_us = 0.0;
            timing->gemm_us = ms[2] * 1000.0;
            timing->output_transform_us = ms[3] * 1000.0;
            timing->total_us = (ms[0] + ms[1] + ms[2] + ms[3]) * 1000.0;
            for (auto event : events) cudaEventDestroy(event);
        }
        return;
    }

    // Same-weight Fast differential: consume the original packed K6 trellis
    // directly with the established async-A/MMA decoder. This is deliberately
    // narrower than the general reconstruct+cuBLAS candidate: only the 6-bpw
    // K6 shapes used by the large prefill are admitted, and the existing
    // numeric route remains the fallback for every other shape. The direct
    // path uses one split plane because this workspace intentionally owns one
    // output-sized FP32 accumulation buffer; it never reconstructs or
    // re-quantizes weights.
    const bool packed_direct_k6_candidate =
        impl_->packed_direct_k6_enabled && !fused_original && rows >= 256 &&
        metadata.K == 6 && metadata.in_features % 16 == 0 &&
        metadata.out_features % 16 == 0 && weights.trellis && weights.mul1;
    if (packed_direct_k6_candidate) {
        const int output_tiles_per_block =
            metadata.out_features % (16 * 64) == 0 ? 64 :
            metadata.out_features % (16 * 32) == 0 ? 32 : 16;
        constexpr int split_count = 1;
        const int output_blocks = metadata.out_features /
            (16 * output_tiles_per_block);
        const std::size_t shared_bytes = 512u * sizeof(half) +
            2u * static_cast<std::size_t>(output_tiles_per_block) * 16u * 6u *
                sizeof(std::uint16_t);
        if (output_tiles_per_block == 64) {
            exl3_prefill_direct_async_a_kernel<6, 64, 2, false, true><<<
                dim3(output_blocks * split_count, (rows + 15) / 16), kThreads,
                shared_bytes, stream>>>(
                impl_->transformed, weights.trellis, weights.mul1, impl_->accum,
                rows, metadata.in_features, metadata.out_features, split_count);
        } else if (output_tiles_per_block == 32) {
            exl3_prefill_direct_async_a_kernel<6, 32, 2, false, true><<<
                dim3(output_blocks * split_count, (rows + 15) / 16), kThreads,
                shared_bytes, stream>>>(
                impl_->transformed, weights.trellis, weights.mul1, impl_->accum,
                rows, metadata.in_features, metadata.out_features, split_count);
        } else {
            exl3_prefill_direct_async_a_kernel<6, 16, 2, false, true><<<
                dim3(output_blocks * split_count, (rows + 15) / 16), kThreads,
                shared_bytes, stream>>>(
                impl_->transformed, weights.trellis, weights.mul1, impl_->accum,
                rows, metadata.in_features, metadata.out_features, split_count);
        }
        cuda_check(cudaGetLastError(),
                   "launch FAST packed direct K6 projection");
        launch_prefill_reduce_output<false>(stream,impl_->accum, weights.svh, output, rows,
                      metadata.out_features, split_count);
        cuda_check(cudaGetLastError(),
                   "launch FAST packed direct K6 reduction/output");
        ++impl_->stats.calls;
        ++impl_->stats.k6_calls;
        impl_->stats.rows += static_cast<std::uint64_t>(rows);
        ++impl_->stats.packed_direct_k6_calls;
        impl_->stats.packed_direct_k6_rows += static_cast<std::uint64_t>(rows);
        if (timing) {
            cuda_check(cudaEventRecord(events[2], stream),
                       "T69 packed direct event");
            cuda_check(cudaEventRecord(events[3], stream),
                       "T69 packed reduction event");
            cuda_check(cudaEventRecord(events[4], stream),
                       "T69 packed output event");
            cuda_check(cudaEventSynchronize(events[4]),
                       "T69 resolve packed direct timing");
            float ms[4]{};
            for (int i = 0; i < 4; ++i)
                cuda_check(cudaEventElapsedTime(&ms[i], events[i], events[i + 1]),
                           "T69 packed direct phase timing");
            timing->input_transform_us = ms[0] * 1000.0;
            timing->reconstruct_us = 0.0;
            timing->gemm_us = ms[1] * 1000.0;
            timing->output_transform_us = ms[2] * 1000.0;
            timing->total_us = (ms[0] + ms[1] + ms[2] + ms[3]) * 1000.0;
            for (auto event : events) cudaEventDestroy(event);
        }
        return;
    }

    const dim3 reconstruct_grid(metadata.out_features / 16,
                                metadata.in_features / 16);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(metadata.in_features) *
        static_cast<std::size_t>(metadata.out_features) * sizeof(std::uint16_t);
    std::uint16_t* reconstructed = impl_->reconstructed;
    bool reconstruct_needed = true;
    if (impl_->reuse_active && (!fused_original || original_gdn_mlp)) {
        Impl::ReuseSlot* slot = nullptr;
        for (std::size_t index = 0; index < impl_->reuse_used; ++index) {
            auto& candidate = impl_->reuse_slots[index];
            if (candidate.assigned && candidate.trellis == weights.trellis &&
                candidate.mul1 == weights.mul1 &&
                candidate.suh == weights.suh && candidate.svh == weights.svh &&
                candidate.in_features == metadata.in_features &&
                candidate.out_features == metadata.out_features &&
                candidate.K == metadata.K &&
                candidate.original_basis == fused_original) {
                slot = &candidate;
                break;
            }
        }
        if (!slot) {
            if (impl_->reuse_used == impl_->reuse_slots.size())
                impl_->reuse_slots.emplace_back();
            slot = &impl_->reuse_slots[impl_->reuse_used++];
            slot->assigned = true;
            slot->valid = false;
            slot->trellis = weights.trellis;
            slot->mul1 = weights.mul1;
            slot->suh = weights.suh;
            slot->svh = weights.svh;
            slot->in_features = metadata.in_features;
            slot->out_features = metadata.out_features;
            slot->K = metadata.K;
            slot->original_basis = fused_original;
            if (slot->capacity < weight_bytes &&
                impl_->reuse_capacity - slot->capacity + weight_bytes <=
                    impl_->reuse_budget) {
                if (slot->data) {
                    cuda_check(cudaFree(slot->data),
                               "retire smaller layer-major weight slot");
                    impl_->reuse_capacity -= slot->capacity;
                    slot->data = nullptr;
                    slot->capacity = 0;
                }
                cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot->data),
                                      weight_bytes),
                           "allocate bounded layer-major weight slot");
                slot->capacity = weight_bytes;
                impl_->reuse_capacity += weight_bytes;
                impl_->stats.cached_weight_capacity_bytes = impl_->reuse_capacity;
            }
        }
        if (slot->data && slot->capacity >= weight_bytes) {
            reconstructed = slot->data;
            if (slot->prefetch_pending) {
                cuda_check(cudaStreamWaitEvent(stream,slot->prefetch_ready,0),
                    "join prefetched K5 weight before GEMM");
                slot->prefetch_pending=false;
                slot->valid=true;
                reconstruct_needed=false;
                ++impl_->stats.prefetched_weight_hits;
            } else if (slot->valid) {
                reconstruct_needed = false;
                ++impl_->stats.reused_weight_calls;
                impl_->stats.reused_weight_bytes += weight_bytes;
            } else slot->valid = true;
        }
    }
    // MXFP8 policy: the layer-reuse slot holds the K-major E4M3 weight and its
    // block scales; FP16 reconstruction goes through the scratch plane first.
    // Without a resident slot the call keeps the FP16 route.
    const bool mxfp8 = impl_->mxfp8_enabled && fast_fp16_destination &&
        !fused_original && rows >= 16 &&
        metadata.in_features % 128 == 0 && metadata.out_features % 128 == 0 &&
        reconstructed != impl_->reconstructed;
    std::uint8_t* const mx_weight =
        mxfp8 ? reinterpret_cast<std::uint8_t*>(reconstructed) : nullptr;
    const bool mlp_shape =
        (metadata.in_features == 5120 && metadata.out_features == 17408) ||
        (metadata.in_features == 17408 && metadata.out_features == 5120);
    // Mode 3 mirrors the NInfer Qwen3.8 NVFP4 artifact: MLP layers 0..55.
    const bool nvfp4 = mxfp8 && (impl_->nvfp4_mode == 1 ||
        (impl_->nvfp4_mode == 2 && mlp_shape) ||
        (impl_->nvfp4_mode == 3 && mlp_shape && impl_->prefill_layer >= 0 &&
         impl_->prefill_layer < 56));
    // NVFP4 slot: packed values (k*n/2), block scales (n x k/16), then the
    // FP32 weight global scale, all inside the FP16-sized reuse slot.
    const std::size_t nv_value_bytes =
        static_cast<std::size_t>(metadata.in_features) * metadata.out_features / 2;
    const std::size_t nv_scale_bytes =
        static_cast<std::size_t>(metadata.in_features) * metadata.out_features / 16;
    // Default 1: decode EXL3 trellis tiles straight into the quantized operand
    // (NINFER_EXL3_PREFILL_FUSED_DECODE=0 keeps the FP16 reconstruct + quantize control).
    const bool fused_decode = mxfp8 && impl_->fused_decode_enabled &&
        metadata.K >= 5 && metadata.K <= 7;
    float* const nv_weight_global = nvfp4 ? reinterpret_cast<float*>(
        mx_weight + nv_value_bytes + nv_scale_bytes) : nullptr;
    if (mxfp8) reconstructed = impl_->reconstructed;
    if (reconstruct_needed && fused_original) {
        const dim3 fused_grid(metadata.out_features / kHadamard,
                              metadata.in_features / kHadamard);
        if (metadata.K == 6) {
            exl3_fused_original_weight_reconstruct_kernel<6><<<
                fused_grid, kThreads, 0, stream>>>(
                reconstructed, weights.trellis, weights.suh,
                weights.svh, metadata.out_features / 16);
        } else {
            exl3_fused_original_weight_reconstruct_kernel<7><<<
                fused_grid, kThreads, 0, stream>>>(
                reconstructed, weights.trellis, weights.suh,
                weights.svh, metadata.out_features / 16);
        }
    } else if (reconstruct_needed && fused_decode) {
        if (nvfp4)
            launch_decode_quantize_weight<16>(metadata.K, weights.trellis, weights.mul1,
                mx_weight, mx_weight + nv_value_bytes, nv_weight_global,
                metadata.in_features, metadata.out_features, stream);
        else
            launch_decode_quantize_weight<32>(metadata.K, weights.trellis, weights.mul1,
                mx_weight, mx_weight + static_cast<std::size_t>(metadata.in_features) *
                    metadata.out_features, nullptr,
                metadata.in_features, metadata.out_features, stream);
    } else if (reconstruct_needed) {
        if (metadata.K == 5) {
            exl3_reconstruct_transformed_weight_kernel<5><<<
                reconstruct_grid, 256, 0, stream>>>(
                    weights.trellis, weights.mul1, reconstructed,
                    metadata.in_features, metadata.out_features);
        } else if (metadata.K == 6) {
            exl3_reconstruct_transformed_weight_kernel<6><<<
                reconstruct_grid, 256, 0, stream>>>(
                    weights.trellis, weights.mul1, reconstructed,
                    metadata.in_features, metadata.out_features);
        } else {
            exl3_reconstruct_transformed_weight_kernel<7><<<
                reconstruct_grid, 256, 0, stream>>>(
                    weights.trellis, weights.mul1, reconstructed,
                    metadata.in_features, metadata.out_features);
        }
    }
    if (reconstruct_needed) {
        cuda_check(cudaGetLastError(), "T69 reconstruct transformed weight");
        impl_->stats.reconstructed_weight_bytes += weight_bytes;
        ++impl_->stats.reconstructed_weight_calls;
        if (fused_decode) {
            cuda_check(cudaGetLastError(), "fused EXL3 decode/quantize");
        } else if (nvfp4) {
            auto* amax_bits = reinterpret_cast<unsigned*>(impl_->nvfp4_scalars + 3);
            cuda_check(cudaMemsetAsync(amax_bits, 0, sizeof(unsigned), stream),
                       "clear NVFP4 weight amax");
            nvfp4_amax_kernel<<<1024, 256, 0, stream>>>(
                reinterpret_cast<const half*>(reconstructed),
                static_cast<std::size_t>(metadata.in_features) * metadata.out_features,
                amax_bits);
            nvfp4_quantize_weight_transposed_kernel<<<
                dim3((metadata.out_features + 255) / 256, metadata.in_features / 16),
                256, 0, stream>>>(
                    reinterpret_cast<const half*>(reconstructed), mx_weight,
                    mx_weight + nv_value_bytes, metadata.in_features,
                    metadata.out_features, amax_bits, nv_weight_global);
            cuda_check(cudaGetLastError(), "NVFP4 weight quantization");
        } else if (mxfp8) {
            mxfp8_quantize_weight_transposed_kernel<<<
                dim3((metadata.out_features + 255) / 256, metadata.in_features / 32),
                256, 0, stream>>>(
                    reinterpret_cast<const half*>(reconstructed), mx_weight,
                    mx_weight + static_cast<std::size_t>(metadata.in_features) *
                        metadata.out_features,
                    metadata.in_features, metadata.out_features);
            cuda_check(cudaGetLastError(), "MXFP8 weight quantization");
        }
    }
    if (timing) cuda_check(cudaEventRecord(events[2], stream), "T69 reconstruct event");

    cublas_check(cublasSetStream(impl_->handle, stream), "T69 set cuBLAS stream");
    cublas_check(cublasSetWorkspace(impl_->handle, impl_->cublas_workspace,
                                    16u * 1024u * 1024u),
                 "T69 set cuBLAS workspace");
    const float alpha = 1.0f, beta = 0.0f;
    const half alpha_half = __float2half(1.0f);
    const half beta_half = __float2half(0.0f);
    const void* gemm_input = fused_original
        ? static_cast<const void*>(input)
        : static_cast<const void*>(impl_->transformed);
    void* gemm_output = fused_original
        ? static_cast<void*>(output)
        : (fast_fp16_destination ? static_cast<void*>(output)
                                 : static_cast<void*>(impl_->accum));
    const cudaDataType output_type =
        (fused_original || fast_fp16_destination) ? CUDA_R_16F : CUDA_R_32F;
    bool used_lt=false;
    if (nvfp4) {
        const int in_blocks = metadata.in_features / 16;
        const int padded_rows = (rows + 127) / 128 * 128;
        auto* act_values = reinterpret_cast<std::uint8_t*>(impl_->accum);
        auto* act_scales = act_values +
            static_cast<std::size_t>(rows) * metadata.in_features / 2;
        act_scales += (16 - reinterpret_cast<std::uintptr_t>(act_scales) % 16) % 16;
        auto* amax_bits = reinterpret_cast<unsigned*>(impl_->nvfp4_scalars);
        float* alpha_device = impl_->nvfp4_scalars + 1;
        float* beta_device = impl_->nvfp4_scalars + 2;
        cuda_check(cudaMemsetAsync(amax_bits, 0, sizeof(unsigned), stream),
                   "clear NVFP4 activation amax");
        nvfp4_amax_kernel<<<512, 256, 0, stream>>>(
            reinterpret_cast<const half*>(impl_->transformed),
            static_cast<std::size_t>(rows) * metadata.in_features, amax_bits);
        const int act_threads = padded_rows * in_blocks;
        nvfp4_quantize_rows_kernel<<<(act_threads + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const half*>(impl_->transformed), act_values, act_scales,
            rows, metadata.in_features, padded_rows, amax_bits, nv_weight_global,
            alpha_device);
        cuda_check(cudaGetLastError(), "NVFP4 activation quantization");
        const std::uint64_t key = (1ull << 61) |
            (static_cast<std::uint64_t>(metadata.in_features) << 32) |
            (static_cast<std::uint64_t>(metadata.out_features) << 16) |
            static_cast<std::uint64_t>(rows);
        auto found = impl_->lt_plans.find(key);
        if (found == impl_->lt_plans.end()) {
            Impl::LtPlan plan{};
            cublasLtMatmulPreference_t preference = nullptr;
            try {
                cublas_check(cublasLtMatmulDescCreate(&plan.operation,
                    CUBLAS_COMPUTE_32F, CUDA_R_32F), "create NVFP4 Lt operation");
                const cublasOperation_t transpose = CUBLAS_OP_T, plain = CUBLAS_OP_N;
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
                    "NVFP4 transa");
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_TRANSB, &plain, sizeof(plain)), "NVFP4 transb");
                const cublasLtPointerMode_t device_mode = CUBLASLT_POINTER_MODE_DEVICE;
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_POINTER_MODE, &device_mode, sizeof(device_mode)),
                    "NVFP4 device alpha");
                const cublasLtMatmulMatrixScale_t mode =
                    CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)),
                    "NVFP4 A scale mode");
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)),
                    "NVFP4 B scale mode");
                const void* placeholder = impl_->cublas_workspace;
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &placeholder, sizeof(placeholder)),
                    "NVFP4 placeholder A scales");
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &placeholder, sizeof(placeholder)),
                    "NVFP4 placeholder B scales");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_4F_E2M1,
                    metadata.in_features, metadata.out_features, metadata.in_features),
                    "create NVFP4 weight layout");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_4F_E2M1,
                    metadata.in_features, rows, metadata.in_features),
                    "create NVFP4 activation layout");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_16F,
                    metadata.out_features, rows, metadata.out_features),
                    "create NVFP4 output layout");
                cublas_check(cublasLtMatmulPreferenceCreate(&preference),
                    "create NVFP4 Lt preference");
                constexpr std::size_t limit = 16u * 1024u * 1024u;
                cublas_check(cublasLtMatmulPreferenceSetAttribute(preference,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)),
                    "bound NVFP4 Lt workspace");
                cublasLtMatmulHeuristicResult_t candidates[4]{};
                int count = 0;
                cublas_check(cublasLtMatmulAlgoGetHeuristic(impl_->lt_handle,
                    plan.operation, plan.a, plan.b, plan.c, plan.c, preference,
                    4, candidates, &count), "query NVFP4 Lt algorithms");
                bool selected = false;
                for (int i = 0; i < count && !selected; ++i) {
                    if (candidates[i].state == CUBLAS_STATUS_SUCCESS &&
                        candidates[i].workspaceSize <= limit) {
                        plan.algorithm = candidates[i].algo;
                        plan.workspace_bytes = candidates[i].workspaceSize;
                        selected = true;
                    }
                }
                if (!selected) throw std::runtime_error("no supported NVFP4 Lt plan");
                cublasLtMatmulPreferenceDestroy(preference);
                preference = nullptr;
                found = impl_->lt_plans.emplace(key, plan).first;
            } catch (...) {
                if (preference) cublasLtMatmulPreferenceDestroy(preference);
                if (plan.a) cublasLtMatrixLayoutDestroy(plan.a);
                if (plan.b) cublasLtMatrixLayoutDestroy(plan.b);
                if (plan.c) cublasLtMatrixLayoutDestroy(plan.c);
                if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
                throw;
            }
        }
        const auto& plan = found->second;
        const void* weight_scales = mx_weight + nv_value_bytes;
        const void* input_scales = act_scales;
        cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scales, sizeof(weight_scales)),
            "bind NVFP4 weight scales");
        cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &input_scales, sizeof(input_scales)),
            "bind NVFP4 activation scales");
        cublas_check(cublasLtMatmul(impl_->lt_handle, plan.operation, alpha_device,
            mx_weight, plan.a, act_values, plan.b, beta_device,
            output, plan.c, output, plan.c, &plan.algorithm,
            impl_->cublas_workspace, plan.workspace_bytes, stream),
            "NVFP4 prefill cuBLASLt GEMM");
        ++impl_->stats.nvfp4_calls;
        impl_->stats.nvfp4_rows += static_cast<std::uint64_t>(rows);
        used_lt = true;
    } else if (mxfp8) {
        const int in_blocks = metadata.in_features / 32;
        const int padded_rows = (rows + 127) / 128 * 128;
        auto* act_values = reinterpret_cast<std::uint8_t*>(impl_->accum);
        auto* act_scales = act_values +
            static_cast<std::size_t>(rows) * metadata.in_features;
        act_scales += (16 - reinterpret_cast<std::uintptr_t>(act_scales) % 16) % 16;
        const int act_threads = padded_rows * in_blocks;
        mxfp8_quantize_rows_kernel<<<(act_threads + 255) / 256, 256, 0, stream>>>(
            reinterpret_cast<const half*>(impl_->transformed), act_values, act_scales,
            rows, metadata.in_features, padded_rows);
        cuda_check(cudaGetLastError(), "MXFP8 activation quantization");
        const std::uint64_t key = (1ull << 62) |
            (static_cast<std::uint64_t>(metadata.in_features) << 32) |
            (static_cast<std::uint64_t>(metadata.out_features) << 16) |
            static_cast<std::uint64_t>(rows);
        auto found = impl_->lt_plans.find(key);
        if (found == impl_->lt_plans.end()) {
            Impl::LtPlan plan{};
            cublasLtMatmulPreference_t preference = nullptr;
            try {
                cublas_check(cublasLtMatmulDescCreate(&plan.operation,
                    CUBLAS_COMPUTE_32F, CUDA_R_32F), "create MXFP8 Lt operation");
                const cublasOperation_t transpose = CUBLAS_OP_T, plain = CUBLAS_OP_N;
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)),
                    "MXFP8 transa");
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_TRANSB, &plain, sizeof(plain)), "MXFP8 transb");
                const cublasLtMatmulMatrixScale_t mode =
                    CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)),
                    "MXFP8 A scale mode");
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)),
                    "MXFP8 B scale mode");
                // Block-scaled heuristics require bound scale pointers; each
                // call rebinds its own weight and activation scales.
                const void* placeholder = impl_->cublas_workspace;
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &placeholder, sizeof(placeholder)),
                    "MXFP8 placeholder A scales");
                cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &placeholder, sizeof(placeholder)),
                    "MXFP8 placeholder B scales");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_8F_E4M3,
                    metadata.in_features, metadata.out_features, metadata.in_features),
                    "create MXFP8 weight layout");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_8F_E4M3,
                    metadata.in_features, rows, metadata.in_features),
                    "create MXFP8 activation layout");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_16F,
                    metadata.out_features, rows, metadata.out_features),
                    "create MXFP8 output layout");
                cublas_check(cublasLtMatmulPreferenceCreate(&preference),
                    "create MXFP8 Lt preference");
                constexpr std::size_t limit = 16u * 1024u * 1024u;
                cublas_check(cublasLtMatmulPreferenceSetAttribute(preference,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit)),
                    "bound MXFP8 Lt workspace");
                cublasLtMatmulHeuristicResult_t candidates[4]{};
                int count = 0;
                cublas_check(cublasLtMatmulAlgoGetHeuristic(impl_->lt_handle,
                    plan.operation, plan.a, plan.b, plan.c, plan.c, preference,
                    4, candidates, &count), "query MXFP8 Lt algorithms");
                bool selected = false;
                for (int i = 0; i < count && !selected; ++i) {
                    if (candidates[i].state == CUBLAS_STATUS_SUCCESS &&
                        candidates[i].workspaceSize <= limit) {
                        plan.algorithm = candidates[i].algo;
                        plan.workspace_bytes = candidates[i].workspaceSize;
                        selected = true;
                    }
                }
                if (!selected) throw std::runtime_error("no supported MXFP8 Lt plan");
                cublasLtMatmulPreferenceDestroy(preference);
                preference = nullptr;
                found = impl_->lt_plans.emplace(key, plan).first;
            } catch (...) {
                if (preference) cublasLtMatmulPreferenceDestroy(preference);
                if (plan.a) cublasLtMatrixLayoutDestroy(plan.a);
                if (plan.b) cublasLtMatrixLayoutDestroy(plan.b);
                if (plan.c) cublasLtMatrixLayoutDestroy(plan.c);
                if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
                throw;
            }
        }
        const auto& plan = found->second;
        const void* weight_scales = mx_weight +
            static_cast<std::size_t>(metadata.in_features) * metadata.out_features;
        const void* input_scales = act_scales;
        cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
            CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &weight_scales, sizeof(weight_scales)),
            "bind MXFP8 weight scales");
        cublas_check(cublasLtMatmulDescSetAttribute(plan.operation,
            CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &input_scales, sizeof(input_scales)),
            "bind MXFP8 activation scales");
        cublas_check(cublasLtMatmul(impl_->lt_handle, plan.operation, &alpha,
            mx_weight, plan.a, act_values, plan.b, &beta,
            output, plan.c, output, plan.c, &plan.algorithm,
            impl_->cublas_workspace, plan.workspace_bytes, stream),
            "MXFP8 prefill cuBLASLt GEMM");
        ++impl_->stats.mxfp8_calls;
        impl_->stats.mxfp8_rows += static_cast<std::uint64_t>(rows);
        used_lt = true;
    }
    const bool large_lt_candidate=!mxfp8 && impl_->large_lt_enabled &&
        (metadata.K==6 || metadata.K==7) && rows>=4096;
    if (!mxfp8 && ((impl_->k5_lt_enabled && metadata.K==5 && !fp16_compute) ||
         (large_lt_candidate && fp16_compute)) &&
        fast_fp16_destination && !fused_original) {
        const std::uint64_t key=(fp16_compute?1ull<<63:0ull) |
            (static_cast<std::uint64_t>(metadata.in_features)<<32) |
            (static_cast<std::uint64_t>(metadata.out_features)<<16) |
            static_cast<std::uint64_t>(rows);
        auto found=impl_->lt_plans.find(key);
        if (found==impl_->lt_plans.end()) {
            Impl::LtPlan plan{};
            cublasLtMatmulPreference_t preference=nullptr;
            try {
                cublas_check(cublasLtMatmulDescCreate(&plan.operation,
                    fp16_compute?CUBLAS_COMPUTE_16F:CUBLAS_COMPUTE_32F,
                    fp16_compute?CUDA_R_16F:CUDA_R_32F),"create prefill Lt operation");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.a,CUDA_R_16F,
                    metadata.out_features,metadata.in_features,metadata.out_features),
                    "create K5 Lt A layout");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.b,CUDA_R_16F,
                    metadata.in_features,rows,metadata.in_features),
                    "create K5 Lt B layout");
                cublas_check(cublasLtMatrixLayoutCreate(&plan.c,CUDA_R_16F,
                    metadata.out_features,rows,metadata.out_features),
                    "create K5 Lt C layout");
                cublas_check(cublasLtMatmulPreferenceCreate(&preference),
                    "create K5 Lt preference");
                constexpr std::size_t limit=16u*1024u*1024u;
                cublas_check(cublasLtMatmulPreferenceSetAttribute(preference,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)),
                    "bound K5 Lt workspace");
                cublasLtMatmulHeuristicResult_t candidates[4]{};
                int count=0;
                cublas_check(cublasLtMatmulAlgoGetHeuristic(impl_->lt_handle,
                    plan.operation,plan.a,plan.b,plan.c,plan.c,preference,
                    4,candidates,&count),"query K5 Lt algorithms");
                bool selected=false;
                for (int i=0;i<count;++i) {
                    if (candidates[i].state==CUBLAS_STATUS_SUCCESS &&
                        candidates[i].workspaceSize<=limit) {
                        plan.algorithm=candidates[i].algo;
                        plan.workspace_bytes=candidates[i].workspaceSize;
                        selected=true;
                        break;
                    }
                }
                if (!selected)
                    throw std::runtime_error("no supported K5 Lt plan");
                cublasLtMatmulPreferenceDestroy(preference);
                preference=nullptr;
                found=impl_->lt_plans.emplace(key,plan).first;
            } catch (...) {
                if (preference) cublasLtMatmulPreferenceDestroy(preference);
                if (plan.a) cublasLtMatrixLayoutDestroy(plan.a);
                if (plan.b) cublasLtMatrixLayoutDestroy(plan.b);
                if (plan.c) cublasLtMatrixLayoutDestroy(plan.c);
                if (plan.operation) cublasLtMatmulDescDestroy(plan.operation);
                throw;
            }
        }
        const auto& plan=found->second;
        cublas_check(cublasLtMatmul(impl_->lt_handle,plan.operation,
            fp16_compute?static_cast<const void*>(&alpha_half):
                static_cast<const void*>(&alpha),
            reconstructed,plan.a,gemm_input,plan.b,
            fp16_compute?static_cast<const void*>(&beta_half):
                static_cast<const void*>(&beta),
            gemm_output,plan.c,gemm_output,plan.c,&plan.algorithm,
            impl_->cublas_workspace,plan.workspace_bytes,stream),
            "K5 prefill cuBLASLt GEMM");
        if(metadata.K==5) ++impl_->stats.k5_lt_calls;
        else ++impl_->stats.large_lt_calls;
        used_lt=true;
    }
    if (!used_lt) {
        cublas_check(cublasGemmEx(
            impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
            metadata.out_features, rows, metadata.in_features,
            fp16_compute ? static_cast<const void*>(&alpha_half)
                         : static_cast<const void*>(&alpha),
            reconstructed, CUDA_R_16F, metadata.out_features,
            gemm_input, CUDA_R_16F, metadata.in_features,
            fp16_compute ? static_cast<const void*>(&beta_half)
                         : static_cast<const void*>(&beta),
            gemm_output, output_type, metadata.out_features,
            fp16_compute ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            fp16_compute ? "T69 same-weights FP16-compute GEMM"
                         : (fused_original ? "T69 fused original-basis GEMM"
                                            : "T69 FP32-compute GEMM"));
    }
    if (timing) cuda_check(cudaEventRecord(events[3], stream), "T69 GEMM event");

    if (!fused_original) {
        if (fast_fp16_destination) {
            if (residual) {
                output_hadamard_fp16_residual_kernel<<<
                    dim3(rows, metadata.out_features / kHadamard),
                    dim3(kHadamard), 0, stream>>>(
                        output, weights.svh, residual, down_trace,
                        trace_row_base, rows, metadata.out_features);
                ++impl_->stats.fused_down_residual_calls;
                impl_->stats.fused_down_residual_rows +=
                    static_cast<std::uint64_t>(rows);
            } else {
                launch_output_hadamard_fp16_inplace(stream,
                        output, weights.svh, rows, metadata.out_features);
            }
        } else {
            launch_output_hadamard(stream,
                    impl_->accum, weights.svh, output, rows, metadata.out_features);
        }
        cuda_check(cudaGetLastError(), "T69 output Hadamard");
    }
    ++impl_->stats.calls;
    impl_->stats.rows += static_cast<std::uint64_t>(rows);
    if (metadata.K == 5) ++impl_->stats.k5_calls;
    else if (metadata.K == 6) ++impl_->stats.k6_calls;
    else ++impl_->stats.k7_calls;
    if (fused_original) {
        ++impl_->stats.fused_original_calls;
        impl_->stats.fused_original_rows += static_cast<std::uint64_t>(rows);
    }
    if (fp16_compute) {
        ++impl_->stats.fp16_compute_calls;
        impl_->stats.fp16_compute_rows += static_cast<std::uint64_t>(rows);
    }
    if (timing) {
        cuda_check(cudaEventRecord(events[4], stream), "T69 output event");
        cuda_check(cudaEventSynchronize(events[4]), "T69 phase timing sync");
        float ms[4]{};
        for (int i = 0; i < 4; ++i)
            cuda_check(cudaEventElapsedTime(&ms[i], events[i], events[i + 1]),
                       "T69 resolve phase timing");
        timing->input_transform_us = ms[0] * 1000.0;
        timing->reconstruct_us = ms[1] * 1000.0;
        timing->gemm_us = ms[2] * 1000.0;
        timing->output_transform_us = ms[3] * 1000.0;
        timing->total_us = (ms[0] + ms[1] + ms[2] + ms[3]) * 1000.0;
        for (auto event : events) cudaEventDestroy(event);
    }
}

// Exact K6 gate/up variant of the established <6,64,2> direct async-A route.
// Each four-warp group owns one half of the B stage and a private copy of A, so
// the K-tile pipeline does not rendezvous with the unrelated output half.  Warp
// output ownership, fragment order, split boundaries, and the FP32 MMA sequence
// are otherwise identical to exl3_prefill_direct_async_a_kernel<6,64,2>.
__global__ __launch_bounds__(256,2) void exl3_prefill_k6_gateup_warpgroup_async_kernel(
    const std::uint16_t* transformed,
    const std::uint16_t* trellis,
    const std::int32_t* mul1,
    float* accum,
    int rows,
    int input_features,
    int output_features,
    int split_count) {
    constexpr int kBits = 6;
    constexpr int kOutputTilesPerBlock = 64;
    constexpr int kFragmentsPerWarp = 16;
    constexpr int kTilesPerWarpGroup = 32;
    constexpr int kTileHalf = 16 * kBits;
    constexpr int kRawStageHalf = kOutputTilesPerBlock * kTileHalf;
    constexpr int kRawGroupHalf = kTilesPerWarpGroup * kTileHalf;
    constexpr int kWarpGroups = 2;

    const int total_rows = rows;
    const int row_base = static_cast<int>(blockIdx.y) * 16;
    rows = min(16, total_rows - row_base);
    transformed += static_cast<std::size_t>(row_base) * input_features;

    extern __shared__ half shared[];
    half* sh_a = shared; // [stage][warp-group][16 rows][16 columns]
    auto* sh_raw = reinterpret_cast<std::uint16_t*>(sh_a + 2 * kWarpGroups * 256);
    const int thread = static_cast<int>(threadIdx.x);
    const int warp = thread / 32;
    const int lane = thread & 31;
    const int warp_group_id = warp / 4;
    auto warp_group = cg::tiled_partition<128>(cg::this_thread_block());
    const int group_thread = static_cast<int>(warp_group.thread_rank());

    const int tiles_k = input_features / 16;
    const int tiles_n = output_features / 16;
    const int output_blocks = tiles_n / kOutputTilesPerBlock;
    const int block = static_cast<int>(blockIdx.x);
    const int tile_base = (block % output_blocks) * kOutputTilesPerBlock;
    const int split = block / output_blocks;
    const int tiles_per_split = (tiles_k + split_count - 1) / split_count;
    const int tile_k_begin = split * tiles_per_split;
    const int tile_k_end = min(tile_k_begin + tiles_per_split, tiles_k);
    const std::uint32_t mul1_multiplier = static_cast<std::uint32_t>(*mul1);

    auto prefetch = [&](int tile_k, int stage) {
        if (tile_k >= tile_k_end) return;
        const int group_tile_base = tile_base + warp_group_id * kTilesPerWarpGroup;
        const std::size_t offset =
            (static_cast<std::size_t>(tile_k) * tiles_n + group_tile_base) *
            static_cast<std::size_t>(kTileHalf);
        auto* destination = sh_raw + stage * kRawStageHalf + warp_group_id * kRawGroupHalf;
        const auto* source = trellis + offset;
        for (int chunk = group_thread; chunk < kRawGroupHalf / 8; chunk += 128) {
            exl3_cp_async_16(destination + chunk * 8, source + chunk * 8);
        }
        if (group_thread < 32) {
            const int row = group_thread / 2;
            const int column = (group_thread % 2) * 8;
            const int source_column = (group_thread % 2 ^ ((row >> 2) & 1)) * 8;
            half* next_a = sh_a + (stage * kWarpGroups + warp_group_id) * 256 +
                row * 16 + column;
            if (row < rows) {
                exl3_cp_async_16(next_a,
                    transformed + row * input_features + tile_k * 16 + source_column);
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) next_a[j] = __float2half_rn(0.0f);
            }
        }
        exl3_cp_async_commit();
    };

    prefetch(tile_k_begin, 0);
    exl3_cp_async_wait();
    warp_group.sync();

    Exl3FragC c[kFragmentsPerWarp];
    #pragma unroll
    for (int n = 0; n < kFragmentsPerWarp; ++n) {
        #pragma unroll
        for (float& value : c[n].values) value = 0.0f;
    }

    int raw_stage = 0;
    for (int tile_k = tile_k_begin; tile_k < tile_k_end; ++tile_k) {
        prefetch(tile_k + 1, 1 - raw_stage);

        Exl3FragA a;
        const int r = (lane % 8) + 8 * ((lane / 8) % 2);
        const int base_c = lane / 16;
        const int c_swizzled = base_c ^ ((r >> 2) & 1);
        exl3_ldsm4(a, sh_a + (raw_stage * kWarpGroups + warp_group_id) * 256 +
            r * 16 + c_swizzled * 8);

        #pragma unroll
        for (int n2 = 0; n2 < kFragmentsPerWarp; n2 += 2) {
            const int sub_n2 = warp * (kFragmentsPerWarp / 2) + n2 / 2;
            const auto* packed = reinterpret_cast<const std::uint32_t*>(
                sh_raw + raw_stage * kRawStageHalf + sub_n2 * kTileHalf);
            Exl3FragB b0, b1;
            exl3_dq4_generic<kBits>(packed, lane << 3, b0, mul1_multiplier);
            exl3_dq4_generic<kBits>(packed, (lane << 3) + 4, b1, mul1_multiplier);
            exl3_mma_m16n8k16(a, b0, c[n2]);
            exl3_mma_m16n8k16(a, b1, c[n2 + 1]);
        }

        if (tile_k + 1 < tile_k_end) {
            exl3_cp_async_wait();
            warp_group.sync();
        }
        raw_stage = 1 - raw_stage;
    }

    const int n0 = warp * kFragmentsPerWarp;
    const int r0 = lane / 4, r1 = r0 + 8, column = (lane % 4) * 2;
    const std::size_t stride = static_cast<std::size_t>(total_rows) * output_features;
    #pragma unroll
    for (int n = 0; n < kFragmentsPerWarp; ++n) {
        const int col = tile_base * 16 + (n0 + n) * 8 + column;
        if (r0 < rows) {
            float* dst = accum + static_cast<std::size_t>(split) * stride +
                static_cast<std::size_t>(row_base + r0) * output_features + col;
            dst[0] = c[n].values[0]; dst[1] = c[n].values[1];
        }
        if (r1 < rows) {
            float* dst = accum + static_cast<std::size_t>(split) * stride +
                static_cast<std::size_t>(row_base + r1) * output_features + col;
            dst[0] = c[n].values[2]; dst[1] = c[n].values[3];
        }
    }
}

// Two ordinary N32 groups share one double-buffered represented A tile. Each
// 256-thread half retains the established N32 warp/fragment ownership, K-split
// bounds, K6 decode order, MMA chains and FP32 partial destinations. This cuts
// duplicate A traffic and CTAs without the sixteen-accumulator N64 footprint.
__global__ __launch_bounds__(512,1) void exl3_prefill_k6_gateup_n32_pair_cta_kernel(
    const std::uint16_t* transformed,const std::uint16_t* trellis,
    const std::int32_t* mul1,float* accum,int rows,int input_features,
    int output_features,int split_count) {
    constexpr int kBits=6,kTilesPerGroup=32,kGroups=2;
    constexpr int kFragmentsPerWarp=8,kTileHalf=16*kBits;
    constexpr int kRawGroupHalf=kTilesPerGroup*kTileHalf;
    constexpr int kRawStageHalf=kGroups*kRawGroupHalf;
    const int total_rows=rows;
    const int row_base=static_cast<int>(blockIdx.y)*16;
    rows=min(16,total_rows-row_base);
    transformed+=static_cast<std::size_t>(row_base)*input_features;
    extern __shared__ half shared[];
    half* sh_a=shared;
    auto* sh_raw=reinterpret_cast<std::uint16_t*>(sh_a+512);
    const int thread=static_cast<int>(threadIdx.x);
    const int group=thread/256,group_thread=thread%256;
    const int warp=group_thread/32,lane=group_thread&31;
    const int tiles_k=input_features/16,tiles_n=output_features/16;
    const int output_pairs=tiles_n/(kGroups*kTilesPerGroup);
    const int block=static_cast<int>(blockIdx.x);
    const int tile_base=(block%output_pairs)*(kGroups*kTilesPerGroup);
    const int split=block/output_pairs;
    const int tiles_per_split=(tiles_k+split_count-1)/split_count;
    const int tile_k_begin=split*tiles_per_split;
    const int tile_k_end=min(tile_k_begin+tiles_per_split,tiles_k);
    const std::uint32_t mul1_multiplier=static_cast<std::uint32_t>(*mul1);
    auto prefetch=[&](int tile_k,int stage) {
        if(tile_k>=tile_k_end)return;
        if(thread<32) {
            const int row=thread/2,column=(thread%2)*8;
            const int source_column=(thread%2^((row>>2)&1))*8;
            half* destination=sh_a+stage*256+row*16+column;
            if(row<rows)exl3_cp_async_16(destination,
                transformed+row*input_features+tile_k*16+source_column);
            else {
                #pragma unroll
                for(int j=0;j<8;++j)destination[j]=__float2half_rn(0.0f);
            }
        }
        const int group_tile_base=tile_base+group*kTilesPerGroup;
        const std::size_t offset=
            (static_cast<std::size_t>(tile_k)*tiles_n+group_tile_base)*kTileHalf;
        auto* destination=sh_raw+stage*kRawStageHalf+group*kRawGroupHalf;
        const auto* source=trellis+offset;
        for(int chunk=group_thread;chunk<kRawGroupHalf/8;chunk+=256)
            exl3_cp_async_16(destination+chunk*8,source+chunk*8);
        exl3_cp_async_commit();
    };
    prefetch(tile_k_begin,0);
    exl3_cp_async_wait();
    __syncthreads();
    Exl3FragC c[kFragmentsPerWarp];
    #pragma unroll
    for(int n=0;n<kFragmentsPerWarp;++n) {
        #pragma unroll
        for(float& value:c[n].values)value=0.0f;
    }
    int raw_stage=0;
    for(int tile_k=tile_k_begin;tile_k<tile_k_end;++tile_k) {
        prefetch(tile_k+1,1-raw_stage);
        Exl3FragA a;
        const int r=(lane%8)+8*((lane/8)%2);
        const int base_c=lane/16;
        const int c_swizzled=base_c^((r>>2)&1);
        exl3_ldsm4(a,sh_a+raw_stage*256+r*16+c_swizzled*8);
        #pragma unroll
        for(int n2=0;n2<kFragmentsPerWarp;n2+=2) {
            const int sub_n2=warp*(kFragmentsPerWarp/2)+n2/2;
            const auto* packed=reinterpret_cast<const std::uint32_t*>(
                sh_raw+raw_stage*kRawStageHalf+group*kRawGroupHalf+
                sub_n2*kTileHalf);
            Exl3FragB b0,b1;
            exl3_dq4_k6_lane_window(packed,lane<<3,b0,mul1_multiplier);
            exl3_dq4_k6_lane_window(packed,(lane<<3)+4,b1,mul1_multiplier);
            exl3_mma_m16n8k16(a,b0,c[n2]);
            exl3_mma_m16n8k16(a,b1,c[n2+1]);
        }
        if(tile_k+1<tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        raw_stage=1-raw_stage;
    }
    const int n0=warp*kFragmentsPerWarp;
    const int r0=lane/4,r1=r0+8,column=(lane%4)*2;
    const int group_tile_base=tile_base+group*kTilesPerGroup;
    const std::size_t stride=static_cast<std::size_t>(total_rows)*output_features;
    #pragma unroll
    for(int n=0;n<kFragmentsPerWarp;++n) {
        const int col=group_tile_base*16+(n0+n)*8+column;
        if(r0<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+
                static_cast<std::size_t>(row_base+r0)*output_features+col;
            dst[0]=c[n].values[0];dst[1]=c[n].values[1];
        }
        if(r1<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+
                static_cast<std::size_t>(row_base+r1)*output_features+col;
            dst[0]=c[n].values[2];dst[1]=c[n].values[3];
        }
    }
}

// One 512-thread CTA owns the same 64 output tiles for each of two independent
// K6 projections. Warps 0..7 retain the gate fragment ownership and warps
// 8..15 retain the up ownership. Each half consumes its matrix-specific SUH
// transform; trellis decode, K-split bounds, MMA order and FP32 destinations
// stay private.
__global__ __launch_bounds__(512,1) void exl3_prefill_k6_gate_up_pair_kernel(
    const std::uint16_t* gate_transformed,const std::uint16_t* up_transformed,
    const std::uint16_t* gate_trellis,const std::int32_t* gate_mul1,
    const std::uint16_t* up_trellis,const std::int32_t* up_mul1,
    float* gate_accum,float* up_accum,int rows,int input_features,
    int output_features,int split_count) {
    constexpr int kBits=6,kOutputTiles=64,kFragmentsPerWarp=16;
    constexpr int kTileHalf=16*kBits,kRawPerProjection=kOutputTiles*kTileHalf;
    const int total_rows=rows;
    const int row_base=static_cast<int>(blockIdx.y)*16;
    rows=min(16,total_rows-row_base);
    extern __shared__ half shared[];
    half* sh_a=shared;
    auto* sh_raw=reinterpret_cast<std::uint16_t*>(sh_a+512);
    const int thread=static_cast<int>(threadIdx.x);
    const int warp=thread/32,lane=thread&31;
    const int projection=warp/8,local_warp=warp%8;
    const int projection_thread=thread&255;
    const auto* transformed=(projection?up_transformed:gate_transformed)+
        static_cast<std::size_t>(row_base)*input_features;
    const int tiles_k=input_features/16,tiles_n=output_features/16;
    const int output_blocks=tiles_n/kOutputTiles;
    const int block=static_cast<int>(blockIdx.x);
    const int tile_base=(block%output_blocks)*kOutputTiles;
    const int split=block/output_blocks;
    const int tiles_per_split=(tiles_k+split_count-1)/split_count;
    const int tile_k_begin=split*tiles_per_split;
    const int tile_k_end=min(tile_k_begin+tiles_per_split,tiles_k);
    const std::uint16_t* trellis=projection?up_trellis:gate_trellis;
    const std::uint32_t mul1_multiplier=static_cast<std::uint32_t>(
        *(projection?up_mul1:gate_mul1));
    Exl3FragC c[kFragmentsPerWarp];
    #pragma unroll
    for(int n=0;n<kFragmentsPerWarp;++n) {
        #pragma unroll
        for(float& value:c[n].values)value=0.0f;
    }
    for(int tile_k=tile_k_begin;tile_k<tile_k_end;++tile_k) {
        if(projection_thread<32) {
            const int row=projection_thread/2,column=(projection_thread%2)*8;
            const int source_column=(projection_thread%2^((row>>2)&1))*8;
            half* destination=sh_a+projection*256+row*16+column;
            if(row<rows)exl3_cp_async_16(destination,
                transformed+row*input_features+tile_k*16+source_column);
            else {
                #pragma unroll
                for(int j=0;j<8;++j)destination[j]=__float2half_rn(0.0f);
            }
        }
        const std::size_t weight_offset=
            (static_cast<std::size_t>(tile_k)*tiles_n+tile_base)*kTileHalf;
        auto* raw_destination=sh_raw+projection*kRawPerProjection;
        const auto* raw_source=trellis+weight_offset;
        for(int chunk=thread%256;chunk<kRawPerProjection/8;chunk+=256)
            exl3_cp_async_16(raw_destination+chunk*8,raw_source+chunk*8);
        exl3_cp_async_commit();
        exl3_cp_async_wait();
        __syncthreads();
        Exl3FragA a;
        const int r=(lane%8)+8*((lane/8)%2);
        const int base_c=lane/16;
        const int c_swizzled=base_c^((r>>2)&1);
        exl3_ldsm4(a,sh_a+projection*256+r*16+c_swizzled*8);
        #pragma unroll
        for(int n2=0;n2<kFragmentsPerWarp;n2+=2) {
            const int sub_n2=local_warp*(kFragmentsPerWarp/2)+n2/2;
            const auto* packed=reinterpret_cast<const std::uint32_t*>(
                sh_raw+projection*kRawPerProjection+sub_n2*kTileHalf);
            Exl3FragB b0,b1;
            exl3_dq4_k6_lane_window(packed,lane<<3,b0,mul1_multiplier);
            exl3_dq4_k6_lane_window(packed,(lane<<3)+4,b1,mul1_multiplier);
            exl3_mma_m16n8k16(a,b0,c[n2]);
            exl3_mma_m16n8k16(a,b1,c[n2+1]);
        }
        __syncthreads();
    }
    float* accum=projection?up_accum:gate_accum;
    const int n0=local_warp*kFragmentsPerWarp;
    const int r0=lane/4,r1=r0+8,column=(lane%4)*2;
    const std::size_t stride=static_cast<std::size_t>(total_rows)*output_features;
    #pragma unroll
    for(int n=0;n<kFragmentsPerWarp;++n) {
        const int col=tile_base*16+(n0+n)*8+column;
        if(r0<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+
                static_cast<std::size_t>(row_base+r0)*output_features+col;
            dst[0]=c[n].values[0];dst[1]=c[n].values[1];
        }
        if(r1<rows) {
            float* dst=accum+static_cast<std::size_t>(split)*stride+
                static_cast<std::size_t>(row_base+r1)*output_features+col;
            dst[0]=c[n].values[2];dst[1]=c[n].values[3];
        }
    }
}

// Two selected rows=1 K6/N16 projection bodies in one CTA. Each 256-thread
// half preserves its own double-buffered transformed-A and packed-B stages,
// K-split boundaries, fragment ownership and chronological FP32 MMA chain.
// The fusion removes only a kernel launch; no represented operand is shared.
template <int Bits>
__global__ __launch_bounds__(512,1) void exl3_target_m1_projection_pair_n16_kernel(
    const std::uint16_t* first_transformed,const std::uint16_t* second_transformed,
    const std::uint16_t* first_trellis,const std::int32_t* first_mul1,
    const std::uint16_t* second_trellis,const std::int32_t* second_mul1,
    float* first_accum,float* second_accum,int input_features,int output_features,
    int split_count) {
    constexpr int kBits=Bits,kOutputTiles=16,kFragmentsPerWarp=4;
    constexpr int kTileHalf=16*kBits,kRawStageHalf=kOutputTiles*kTileHalf;
    extern __shared__ half shared[];
    half* sh_a=shared; // [projection][stage][256]
    auto* sh_raw=reinterpret_cast<std::uint16_t*>(sh_a+2*2*256);
    const int thread=static_cast<int>(threadIdx.x);
    const int projection=thread/256,local_thread=thread&255;
    const int warp=local_thread/32,lane=local_thread&31;
    const auto* transformed=projection?second_transformed:first_transformed;
    const auto* trellis=projection?second_trellis:first_trellis;
    const std::uint32_t multiplier=static_cast<std::uint32_t>(
        *(projection?second_mul1:first_mul1));
    const int tiles_k=input_features/16,tiles_n=output_features/16;
    const int output_blocks=tiles_n/kOutputTiles;
    const int block=static_cast<int>(blockIdx.x);
    const int tile_base=(block%output_blocks)*kOutputTiles;
    const int split=block/output_blocks;
    const int tiles_per_split=(tiles_k+split_count-1)/split_count;
    const int tile_k_begin=split*tiles_per_split;
    const int tile_k_end=min(tile_k_begin+tiles_per_split,tiles_k);
    auto prefetch=[&](int tile_k,int stage) {
        if(tile_k>=tile_k_end)return;
        auto* raw_destination=sh_raw+
            (projection*2+stage)*kRawStageHalf;
        const auto* raw_source=trellis+
            (static_cast<std::size_t>(tile_k)*tiles_n+tile_base)*kTileHalf;
        for(int chunk=local_thread;chunk<kRawStageHalf/8;chunk+=256)
            exl3_cp_async_16(raw_destination+chunk*8,raw_source+chunk*8);
        if(local_thread<32) {
            const int row=local_thread/2,column=(local_thread%2)*8;
            const int source_column=(local_thread%2^((row>>2)&1))*8;
            half* destination=sh_a+(projection*2+stage)*256+
                row*16+column;
            if(row<1)exl3_cp_async_16(destination,
                transformed+tile_k*16+source_column);
            else {
                #pragma unroll
                for(int j=0;j<8;++j)destination[j]=__float2half_rn(0.0f);
            }
        }
        exl3_cp_async_commit();
    };
    prefetch(tile_k_begin,0);
    exl3_cp_async_wait();
    __syncthreads();
    Exl3FragC c[kFragmentsPerWarp];
    #pragma unroll
    for(int n=0;n<kFragmentsPerWarp;++n) {
        #pragma unroll
        for(float& value:c[n].values)value=0.0f;
    }
    int stage=0;
    for(int tile_k=tile_k_begin;tile_k<tile_k_end;++tile_k) {
        prefetch(tile_k+1,1-stage);
        Exl3FragA a;
        const int r=(lane%8)+8*((lane/8)%2);
        const int base_c=lane/16;
        const int c_swizzled=base_c^((r>>2)&1);
        exl3_ldsm4(a,sh_a+(projection*2+stage)*256+
            r*16+c_swizzled*8);
        #pragma unroll
        for(int n2=0;n2<kFragmentsPerWarp;n2+=2) {
            const int sub_n2=warp*(kFragmentsPerWarp/2)+n2/2;
            const auto* packed=reinterpret_cast<const std::uint32_t*>(
                sh_raw+(projection*2+stage)*kRawStageHalf+
                sub_n2*kTileHalf);
            Exl3FragB b0,b1;
            if constexpr (kBits == 6) {
                exl3_dq4_k6_lane_window(packed,lane<<3,b0,multiplier);
                exl3_dq4_k6_lane_window(packed,(lane<<3)+4,b1,multiplier);
            } else {
                exl3_dq4_generic<kBits>(packed,lane<<3,b0,multiplier);
                exl3_dq4_generic<kBits>(packed,(lane<<3)+4,b1,multiplier);
            }
            exl3_mma_m16n8k16(a,b0,c[n2]);
            exl3_mma_m16n8k16(a,b1,c[n2+1]);
        }
        if(tile_k+1<tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        stage=1-stage;
    }
    float* accum=projection?second_accum:first_accum;
    const int n0=warp*kFragmentsPerWarp;
    const int r0=lane/4,column=(lane%4)*2;
    #pragma unroll
    for(int n=0;n<kFragmentsPerWarp;++n) {
        const int col=tile_base*16+(n0+n)*8+column;
        if(r0<1) {
            float* destination=accum+
                static_cast<std::size_t>(split)*output_features+col;
            destination[0]=c[n].values[0];
            destination[1]=c[n].values[1];
        }
    }
}

// One 512-thread CTA per projection, launched with z=2 for K/V. The two
// 256-thread subgroups retain the native N16 fragment ownership but consume
// disjoint K ranges, then combine their row-1 output tile in shared memory.
// This is a native packed-weight implementation: no Mia kernel, re-quantized
// weight, or reconstructed proxy participates in the route.
template <int Bits, int OutputTiles = 16, bool FastK6Decode = false>
__global__ __launch_bounds__(512,1) void exl3_target_m1_wide_projection_pair_n16_kernel(
    const std::uint16_t* first_transformed,const std::uint16_t* second_transformed,
    const std::uint16_t* first_trellis,const std::int32_t* first_mul1,
    const std::uint16_t* second_trellis,const std::int32_t* second_mul1,
    float* first_accum,float* second_accum,int input_features,int output_features,
    int split_count) {
    constexpr int kBits=Bits,kOutputTiles=OutputTiles,
        kFragmentsPerWarp=OutputTiles/4,kStages=2;
    constexpr int kTileHalf=16*kBits,kRawStageHalf=kOutputTiles*kTileHalf;
    constexpr int kOutputTileElements=kOutputTiles*16;
    extern __shared__ half shared[];
    half* sh_a=shared; // [sub-K][stage][256]
    auto* sh_raw=reinterpret_cast<std::uint16_t*>(sh_a+2*2*256);
    auto* sh_partial=reinterpret_cast<float*>(
        sh_raw+2*2*kRawStageHalf);
    const int thread=static_cast<int>(threadIdx.x);
    const int sub_k=thread/256,local_thread=thread&255;
    const int warp=local_thread/32,lane=local_thread&31;
    const bool second=blockIdx.z!=0;
    const auto* transformed=second?second_transformed:first_transformed;
    const auto* trellis=second?second_trellis:first_trellis;
    const std::uint32_t multiplier=static_cast<std::uint32_t>(
        *(second?second_mul1:first_mul1));
    const int tiles_k=input_features/16,tiles_n=output_features/16;
    const int output_blocks=tiles_n/kOutputTiles;
    const int block=static_cast<int>(blockIdx.x);
    const int tile_base=(block%output_blocks)*kOutputTiles;
    const int split=block/output_blocks;
    const int logical_split=split*2+sub_k;
    const int logical_split_count=split_count*2;
    const int tiles_per_logical_split=
        (tiles_k+logical_split_count-1)/logical_split_count;
    const int tile_k_begin=logical_split*tiles_per_logical_split;
    const int tile_k_end=min(tile_k_begin+tiles_per_logical_split,tiles_k);
    auto prefetch=[&](int tile_k,int stage) {
        if(tile_k>=tile_k_end)return;
        auto* raw_destination=sh_raw+
            (sub_k*kStages+stage)*kRawStageHalf;
        const auto* raw_source=trellis+
            (static_cast<std::size_t>(tile_k)*tiles_n+tile_base)*kTileHalf;
        for(int chunk=local_thread;chunk<kRawStageHalf/8;chunk+=256)
            exl3_cp_async_16(raw_destination+chunk*8,raw_source+chunk*8);
        if(local_thread<32) {
            const int row=local_thread/2,column=(local_thread%2)*8;
            const int source_column=(local_thread%2^((row>>2)&1))*8;
            half* destination=sh_a+(sub_k*kStages+stage)*256+
                row*16+column;
            if(row<1)exl3_cp_async_16(destination,
                transformed+tile_k*16+source_column);
            else {
                #pragma unroll
                for(int j=0;j<8;++j)destination[j]=__float2half_rn(0.0f);
            }
        }
        exl3_cp_async_commit();
    };
    prefetch(tile_k_begin,0);
    exl3_cp_async_wait();
    __syncthreads();
    Exl3FragC c[kFragmentsPerWarp];
    #pragma unroll
    for(int n=0;n<kFragmentsPerWarp;++n) {
        #pragma unroll
        for(float& value:c[n].values)value=0.0f;
    }
    int stage=0;
    for(int tile_k=tile_k_begin;tile_k<tile_k_end;++tile_k) {
        prefetch(tile_k+1,1-stage);
        Exl3FragA a;
        const int r=(lane%8)+8*((lane/8)%2);
        const int base_c=lane/16;
        const int c_swizzled=base_c^((r>>2)&1);
        exl3_ldsm4(a,sh_a+(sub_k*kStages+stage)*256+
            r*16+c_swizzled*8);
        #pragma unroll
        for(int n2=0;n2<kFragmentsPerWarp;n2+=2) {
            const int sub_n2=warp*(kFragmentsPerWarp/2)+n2/2;
            const auto* packed=reinterpret_cast<const std::uint32_t*>(
                sh_raw+(sub_k*kStages+stage)*kRawStageHalf+
                sub_n2*kTileHalf);
            Exl3FragB b0,b1;
            if constexpr (FastK6Decode) {
                static_assert(kBits == 6,
                              "the wide fast decoder is K6-only");
                exl3_dq4_k6_lane_window(packed,lane<<3,b0,multiplier);
                exl3_dq4_k6_lane_window(packed,(lane<<3)+4,b1,multiplier);
            } else {
                exl3_dq4_generic<kBits>(packed,lane<<3,b0,multiplier);
                exl3_dq4_generic<kBits>(packed,(lane<<3)+4,b1,multiplier);
            }
            exl3_mma_m16n8k16(a,b0,c[n2]);
            exl3_mma_m16n8k16(a,b1,c[n2+1]);
        }
        if(tile_k+1<tile_k_end) {
            exl3_cp_async_wait();
            __syncthreads();
        }
        stage=1-stage;
    }
    float* partial=sh_partial+sub_k*kOutputTileElements;
    const int n0=warp*kFragmentsPerWarp;
    const int r0=lane/4,column=(lane%4)*2;
    #pragma unroll
    for(int n=0;n<kFragmentsPerWarp;++n) {
        const int col=tile_base*16+(n0+n)*8+column;
        if(r0<1) {
            partial[col-tile_base*16]=c[n].values[0];
            partial[col-tile_base*16+1]=c[n].values[1];
        }
    }
    __syncthreads();
    if(sub_k==0) {
        float* accum=second?second_accum:first_accum;
        for(int i=local_thread;i<kOutputTileElements;i+=256)
            accum[static_cast<std::size_t>(split)*output_features+
                tile_base*16+i]=partial[i]+partial[kOutputTileElements+i];
    }
}

void Exl3CudaReconstructGemmWorkspace::forward_v6_numeric(
    const Exl3CudaLinearWeights& weights, const Exl3CudaLinearMetadata& metadata,
    const std::uint16_t* input, std::uint16_t* half_output, float* float_output,
    int rows, cudaStream_t stream) {
    if (!supports(metadata,rows) || metadata.K!=6 || !weights.trellis ||
        !weights.suh || !weights.svh || !weights.mul1 || !input ||
        (bool(half_output)==bool(float_output)))
        throw std::invalid_argument("V6 numeric projection contract");
    // V6's donor multiplies the FP16 activation and SUH in FP16 before
    // the FP32 butterfly. The established text route deliberately keeps its
    // original full product; this specialization is admitted only here.
    launch_input_hadamard<kHadamard,true>(stream,input,weights.suh,impl_->transformed,rows,metadata.in_features);
    exl3_reconstruct_transformed_weight_kernel<6><<<
        dim3(metadata.out_features/16,metadata.in_features/16),256,0,stream>>>(
        weights.trellis,weights.mul1,impl_->reconstructed,metadata.in_features,metadata.out_features);
    cuda_check(cudaGetLastError(),"V6 reconstruct/input transform");
    cublas_check(cublasSetStream(impl_->handle,stream),"V6 projection stream");
    cublas_check(cublasSetWorkspace(impl_->handle,impl_->cublas_workspace,16u*1024u*1024u),
        "V6 projection workspace");
    const float alpha=1, beta=0;
    cublas_check(cublasGemmEx(impl_->handle,CUBLAS_OP_N,CUBLAS_OP_N,
        metadata.out_features,rows,metadata.in_features,&alpha,
        impl_->reconstructed,CUDA_R_16F,metadata.out_features,
        impl_->transformed,CUDA_R_16F,metadata.in_features,&beta,
        impl_->accum,CUDA_R_32F,metadata.out_features,CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP),"V6 projection GEMM");
    if(float_output) v6_output_hadamard_kernel<true><<<
        dim3(rows,metadata.out_features/kHadamard),kHadamard,0,stream>>>(
        impl_->accum,weights.svh,nullptr,float_output,metadata.out_features);
    else v6_output_hadamard_kernel<false><<<
        dim3(rows,metadata.out_features/kHadamard),kHadamard,0,stream>>>(
        impl_->accum,weights.svh,half_output,nullptr,metadata.out_features);
    cuda_check(cudaGetLastError(),"V6 projection output transform");
    ++impl_->stats.calls; ++impl_->stats.k6_calls; impl_->stats.rows+=rows;
}

void Exl3CudaLinearWorkspace::forward_reconstructed_exact_for_test(
    const Exl3CudaLinearWeights& weights, const Exl3CudaLinearMetadata& metadata,
    const std::uint16_t* input, std::uint16_t* output, int rows,
    std::uint16_t* decoded, std::size_t decoded_bytes, cudaStream_t stream) {
    transform_input(weights, metadata, input, rows, stream);
    forward_reconstructed_exact_from_transformed(weights, metadata, transformed_,
        output, rows, decoded, decoded_bytes, stream);
}

void Exl3CudaLinearWorkspace::forward_predecoded_m1_k6_for_test(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
    std::uint16_t* output,int rows,const std::uint16_t* decoded,
    std::size_t decoded_bytes,cudaStream_t stream) {
    const std::size_t required=static_cast<std::size_t>(metadata.in_features)*
        metadata.out_features*sizeof(std::uint16_t);
    if(rows!=1||metadata.K!=6||metadata.in_features!=in_features_||
       metadata.out_features!=out_features_||metadata.mcg||!metadata.mul1||
       metadata.has_bias||!weights.suh||!weights.svh||!weights.mul1||!input||
       !output||!decoded||decoded_bytes!=required||out_features_%256!=0)
        throw std::invalid_argument("predecoded M1 K6 discriminator contract");
    transform_input(weights,metadata,input,rows,stream);
    constexpr int tiles_per_block=16;
    const int output_blocks=out_features_/(tiles_per_block*16);
    int split_count=5;
    if(const char* requested=std::getenv("NINFER_EXL3_GENERIC_SPLITS");
       requested&&*requested)
        split_count=std::min(split_count,std::max(1,std::atoi(requested)));
    const int grid_blocks=output_blocks*split_count;
    constexpr std::size_t shared_bytes=(512u+2u*tiles_per_block*256u)*
        sizeof(half)+16u*tiles_per_block*16u*sizeof(float);
    exl3_gemm_m1_generic_mma_kernel<6,false,tiles_per_block,true,true,false,true>
        <<<dim3(grid_blocks),dim3(kThreads),shared_bytes,stream>>>(
            transformed_,decoded,weights.mul1,accum_,rows,in_features_,
            out_features_,split_count);
    cuda_check(cudaGetLastError(),"launch predecoded M1 K6 discriminator partials");
    launch_prefill_reduce_output<false>(stream,
            accum_,weights.svh,output,rows,out_features_,split_count);
    cuda_check(cudaGetLastError(),"launch predecoded M1 K6 discriminator output");
}

void Exl3CudaLinearWorkspace::forward_fast_decode_m1_k6_for_test(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
    std::uint16_t* output,int rows,cudaStream_t stream) {
    if(rows!=1||metadata.K!=6||metadata.in_features!=in_features_||
       metadata.out_features!=out_features_||metadata.mcg||!metadata.mul1||
       metadata.has_bias||!weights.trellis||!weights.suh||!weights.svh||
       !weights.mul1||!input||!output||out_features_%256!=0)
        throw std::invalid_argument("fast-decode M1 K6 discriminator contract");
    transform_input(weights,metadata,input,rows,stream);
    constexpr int tiles_per_block=16;
    const int output_blocks=out_features_/(tiles_per_block*16);
    int split_count=5;
    if(const char* requested=std::getenv("NINFER_EXL3_GENERIC_SPLITS");
       requested&&*requested)
        split_count=std::min(split_count,std::max(1,std::atoi(requested)));
    const int grid_blocks=output_blocks*split_count;
    constexpr std::size_t shared_bytes=512u*sizeof(half)+
        2u*tiles_per_block*16u*6u*sizeof(std::uint16_t)+
        16u*tiles_per_block*16u*sizeof(float);
    exl3_gemm_m1_generic_mma_kernel<
        6,false,tiles_per_block,true,true,false,false,true><<<
            dim3(grid_blocks),dim3(kThreads),shared_bytes,stream>>>(
                transformed_,weights.trellis,weights.mul1,accum_,rows,
                in_features_,out_features_,split_count);
    cuda_check(cudaGetLastError(),
        "launch fast-decode M1 K6 discriminator partials");
    launch_prefill_reduce_output<false>(stream,
            accum_,weights.svh,output,rows,out_features_,split_count);
    cuda_check(cudaGetLastError(),
        "launch fast-decode M1 K6 discriminator output");
}

void Exl3CudaLinearWorkspace::forward_k6_rowpair_shared_decode_for_test(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
    std::uint16_t* output,int rows,cudaStream_t stream) {
    if(rows<32||rows>max_rows_||metadata.K!=6||
       metadata.in_features!=in_features_||metadata.out_features!=out_features_||
       in_features_==17408||out_features_%1024!=0||metadata.mcg||!metadata.mul1||
       metadata.has_bias||!weights.trellis||!weights.suh||!weights.svh||
       !weights.mul1||!input||!output)
        throw std::invalid_argument("K6 rowpair shared-decode discriminator contract");
    transform_input(weights,metadata,input,rows,stream);
    constexpr int tiles=64;
    const int blocks=out_features_/(16*tiles);
    int splits=std::min(5,generic_capacity_[1]/blocks);
    if(const char* requested=std::getenv("NINFER_EXL3_GENERIC_SPLITS");
       requested&&*requested)
        splits=std::min(splits,std::max(1,std::atoi(requested)));
    if(splits<1)throw std::runtime_error(
        "K6 rowpair shared-decode discriminator split capacity");
    constexpr std::size_t shared_bytes=(1024u+2u*64u*16u*6u+
        64u*256u)*sizeof(half);
    static const bool shared_limit_configured=[] {
        constexpr std::size_t bytes=(1024u+2u*64u*16u*6u+
            64u*256u)*sizeof(half);
        cuda_check(cudaFuncSetAttribute(
            exl3_prefill_k6_rowpair_n64_shared_decode_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(bytes)),
            "set K6 rowpair shared-decode discriminator shared-memory limit");
        return true;
    }();
    (void)shared_limit_configured;
    exl3_prefill_k6_rowpair_n64_shared_decode_kernel<<<
        dim3(blocks*splits,(rows+31)/32),512,shared_bytes,stream>>>(
            transformed_,weights.trellis,weights.mul1,accum_,rows,in_features_,
            out_features_,splits);
    cuda_check(cudaGetLastError(),
        "launch K6 rowpair shared-decode discriminator partials");
    if(target_reduce_shfl_)
        launch_prefill_reduce_output<true>(stream,
                accum_,weights.svh,output,rows,out_features_,splits);
    else launch_prefill_reduce_output<false>(stream,
            accum_,weights.svh,output,rows,out_features_,splits);
    cuda_check(cudaGetLastError(),
        "launch K6 rowpair shared-decode discriminator output");
}

bool Exl3CudaLinearWorkspace::reconstructed_exact_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    const bool down = metadata.in_features == 17408 &&
        metadata.out_features == 5120 &&
        (metadata.K == 7 ||
         (metadata.K == 6 && reconstructed_exact_.allow_k6_down));
    const bool k6_gate_up = metadata.K == 6 &&
        reconstructed_exact_.allow_k6_gate_up &&
        metadata.in_features == 5120 && metadata.out_features == 17408 &&
        reconstructed_exact_.bytes ==
            std::size_t(metadata.in_features) * metadata.out_features *
                sizeof(std::uint16_t);
    return reconstructed_exact_.data &&
        exl3_reconstruction_reservation_fits(reconstructed_exact_.bytes) &&
        (down || k6_gate_up) &&
        (rows == 256 || rows == 512 || rows == 1024) &&
        target_staged_prefill_candidate(metadata, rows, admission);
}

void Exl3CudaLinearWorkspace::forward_reconstructed_exact_from_transformed(
    const Exl3CudaLinearWeights& weights, const Exl3CudaLinearMetadata& metadata,
    const std::uint16_t* transformed_input, std::uint16_t* output, int rows,
    std::uint16_t* decoded, std::size_t decoded_bytes, cudaStream_t stream) {
    const bool down = metadata.in_features == 17408 &&
        metadata.out_features == 5120 &&
        (metadata.K == 6 || metadata.K == 7);
    const bool k6_gate_up = metadata.K == 6 &&
        metadata.in_features == 5120 && metadata.out_features == 17408;
    if (!transformed_input || !output || !decoded || !weights.trellis || !weights.mul1 ||
        !weights.suh || !weights.svh || (!down && !k6_gate_up) ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        (rows != 256 && rows != 512 && rows != 1024) ||
        !target_staged_prefill_candidate(metadata, rows, Exl3CudaLinearAdmission::target_wide_prefill) ||
        !target_direct_partials_)
        throw std::invalid_argument("exact reconstructed projection canonical topology contract");
    const int decoded_columns=down ? exl3_down_slice_columns(decoded_bytes) :
        decoded_bytes == std::size_t(metadata.in_features) * metadata.out_features *
                sizeof(std::uint16_t)
            ? metadata.out_features : 0;
    if(!decoded_columns)
        throw std::invalid_argument("exact reconstructed K6 gate/up full-slab contract");
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture), "exact reconstruction capture query");
    if (capture != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("exact reconstruction qualification rejects capture");
    const int tiles=down ? 16 : 32;
    const int blocks=out_features_/(16*tiles);
    // Keep the parent split count, even if the new consumer has different occupancy.
    const int parent_blocks=down ? blocks : out_features_/(16*64);
    int splits=std::min(5,(down ? large_down_candidate_capacity_[metadata.K==6?0:1] :
        generic_capacity_[1])/parent_blocks);
    if (const char* requested=std::getenv(
            down ? "NINFER_EXL3_LARGE_DOWN_SPLITS" : "NINFER_EXL3_GENERIC_SPLITS");
        requested&&*requested)
        splits=std::min(splits,std::max(1,std::atoi(requested)));
    if(splits<1) throw std::runtime_error("exact reconstruction parent split capacity");
    const auto lifetime=Exl3ExactDownLifetimePlan::make(rows,splits,metadata.K,
        metadata.mcg,metadata.mul1,metadata.has_bias,decoded_columns,
        metadata.in_features,metadata.out_features);
    lifetime.require_binding({reinterpret_cast<std::uintptr_t>(transformed_input),
        reinterpret_cast<std::uintptr_t>(decoded),reinterpret_cast<std::uintptr_t>(accum_),
        reinterpret_cast<std::uintptr_t>(output)},
        {lifetime.buffers[0].bytes,decoded_bytes,accumulation_capacity_bytes_,lifetime.buffers[3].bytes});
    for(int first=0;first<out_features_;first+=decoded_columns) {
    const int slice_blocks=decoded_columns/(16*tiles);
    if(metadata.K==6)
        exl3_reconstruct_transformed_weight_kernel<6,true><<<
            dim3(decoded_columns / 16, in_features_ / 16),256,0,stream>>>(
                weights.trellis, weights.mul1, decoded, in_features_, out_features_,first/16,decoded_columns/16);
    else exl3_reconstruct_transformed_weight_kernel<7,true><<<
        dim3(decoded_columns / 16, in_features_ / 16),256,0,stream>>>(
            weights.trellis, weights.mul1, decoded, in_features_, out_features_,first/16,decoded_columns/16);
    cuda_check(cudaGetLastError(), "decode exact K7 fragment-order weights");
    if(target_direct_async_all_) {
        constexpr std::size_t down_shared_bytes=(512u+2u*16u*256u)*sizeof(half);
        constexpr std::size_t gate_up_shared_bytes=(512u+2u*32u*256u)*sizeof(half);
        if(k6_gate_up)
            exl3_prefill_direct_async_a_kernel<6,32,4,true><<<
                dim3(slice_blocks*splits,(rows+15)/16),kThreads,gate_up_shared_bytes,stream>>>(
                    transformed_input,decoded,weights.mul1,accum_,rows,in_features_,out_features_,splits,first/16,decoded_columns/16);
        else if(metadata.K==6)
            exl3_prefill_direct_async_a_kernel<6,16,4,true><<<
                dim3(slice_blocks*splits,(rows+15)/16),kThreads,down_shared_bytes,stream>>>(
                    transformed_input,decoded,weights.mul1,accum_,rows,in_features_,out_features_,splits,first/16,decoded_columns/16);
        else exl3_prefill_direct_async_a_kernel<7,16,4,true><<<
                dim3(slice_blocks*splits,(rows+15)/16),kThreads,down_shared_bytes,stream>>>(
                    transformed_input,decoded,weights.mul1,accum_,rows,in_features_,out_features_,splits,first/16,decoded_columns/16);
    } else {
        constexpr std::size_t down_shared_bytes=(256u+2u*16u*256u)*sizeof(half);
        constexpr std::size_t gate_up_shared_bytes=(256u+2u*32u*256u)*sizeof(half);
        if(k6_gate_up)
            exl3_prefill_direct_partials_kernel<6,32,true><<<
                dim3(slice_blocks*splits,(rows+15)/16),kThreads,gate_up_shared_bytes,stream>>>(
                    transformed_input,decoded,weights.mul1,accum_,rows,in_features_,out_features_,splits,first/16,decoded_columns/16);
        else if(metadata.K==6)
            exl3_prefill_direct_partials_kernel<6,16,true><<<
                dim3(slice_blocks*splits,(rows+15)/16),kThreads,down_shared_bytes,stream>>>(
                    transformed_input,decoded,weights.mul1,accum_,rows,in_features_,out_features_,splits,first/16,decoded_columns/16);
        else exl3_prefill_direct_partials_kernel<7,16,true><<<
            dim3(slice_blocks*splits,(rows+15)/16),kThreads,down_shared_bytes,stream>>>(
                transformed_input,decoded,weights.mul1,accum_,rows,in_features_,out_features_,splits,first/16,decoded_columns/16);
    }
    cuda_check(cudaGetLastError(), "exact reconstructed K7 canonical MMA partials");
    } // Next slice may overwrite decoded only after this slice's consumers.
    if(target_reduce_shfl_) {
        if(target_reduce_shfl_min_barriers_) {
            launch_prefill_reduce_output<true,true>(stream,
                accum_,weights.svh,output,rows,out_features_,splits);
            ++reduce_shfl_min_barrier_calls_;reduce_shfl_min_barrier_rows_+=rows;
        } else launch_prefill_reduce_output<true>(stream,
            accum_,weights.svh,output,rows,out_features_,splits);
    }
    else
        launch_prefill_reduce_output<false>(stream,
            accum_,weights.svh,output,rows,out_features_,splits);
    cuda_check(cudaGetLastError(), "exact reconstructed K7 reduction and output");
}

bool Exl3CudaLinearWorkspace::try_fast_wide_prefill_gemm_from_transformed(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,
    const std::uint16_t* transformed_input,
    std::uint16_t* output,
    int rows,
    cudaStream_t stream) {
    if (!fast_wide_prefill_gemm_candidate(
            metadata, rows, Exl3CudaLinearAdmission::target_wide_prefill) ||
        !fast_wide_gemm_state_ || !fast_wide_gemm_state_->handle ||
        !weights.trellis || !weights.mul1 || !weights.svh ||
        !transformed_input || !output)
        return false;

    reconstructed_exact_.ordered_stream->require_identity(
        reconstructed_exact_.model_owner, reconstructed_exact_.bytes,
        reconstructed_exact_.allow_k6_down, reconstructed_exact_.data,
        reconstructed_exact_.allow_k6_gate_up);
    auto submission = reconstructed_exact_.ordered_stream->acquire(
        reinterpret_cast<std::uintptr_t>(stream));
    (void)submission;
    try {
        const dim3 reconstruct_grid(
            metadata.out_features / 16, metadata.in_features / 16);
        if (metadata.K == 6) {
            exl3_reconstruct_transformed_weight_kernel<6, false><<<
                reconstruct_grid, 256, 0, stream>>>(
                weights.trellis, weights.mul1, reconstructed_exact_.data,
                metadata.in_features, metadata.out_features);
        } else if (metadata.K == 7) {
            exl3_reconstruct_transformed_weight_kernel<7, false><<<
                reconstruct_grid, 256, 0, stream>>>(
                weights.trellis, weights.mul1, reconstructed_exact_.data,
                metadata.in_features, metadata.out_features);
        } else {
            return false;
        }
        cuda_check(cudaGetLastError(),
                   "launch fast Mia-parity wide-prefill reconstruction");

        cublas_check(cublasSetStream(fast_wide_gemm_state_->handle, stream),
                     "fast Mia-parity wide-prefill cuBLAS stream");
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublas_check(cublasGemmEx(
                         fast_wide_gemm_state_->handle, CUBLAS_OP_N,
                         CUBLAS_OP_N, metadata.out_features, rows,
                         metadata.in_features, &alpha,
                         reconstructed_exact_.data, CUDA_R_16F,
                         metadata.out_features, transformed_input, CUDA_R_16F,
                         metadata.in_features, &beta, accum_, CUDA_R_32F,
                         metadata.out_features, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                     "fast Mia-parity wide-prefill FP32-compute GEMM");
        launch_output_hadamard(stream,accum_, weights.svh, output, rows, metadata.out_features);
        cuda_check(cudaGetLastError(),
                   "fast Mia-parity wide-prefill output Hadamard");
    } catch (...) {
        reconstructed_exact_.ordered_stream->fail();
        throw;
    }
    ++fast_wide_prefill_gemm_calls_;
    fast_wide_prefill_gemm_rows_ += static_cast<std::uint64_t>(rows);
    return true;
}

static int coherent_kv_split_for(const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission, int in_features, int out_features,
    std::size_t capacity);

void Exl3CudaLinearWorkspace::forward(const Exl3CudaLinearWeights& weights,
                                      const Exl3CudaLinearMetadata& metadata,
                                      const std::uint16_t* input,
                                      std::uint16_t* output,
                                      int rows,
                                      cudaStream_t stream,
                                      Exl3CudaLinearAdmission admission) {
    // A missing destination cannot be repaired by dispatch selection. Refuse
    // before transforming input into potentially shared workspace scratch.
    if(!output)throw std::invalid_argument("EXL3 CUDA received a null device buffer");
    if((admission==Exl3CudaLinearAdmission::draft_shared_q_m16 || admission==Exl3CudaLinearAdmission::draft_shared_kv_m16 || admission==Exl3CudaLinearAdmission::draft_shared_o_m16 || admission==Exl3CudaLinearAdmission::draft_shared_down_m16 || admission==Exl3CudaLinearAdmission::draft_shared_gateup_m16) &&
       !draft_shared_m16_candidate(metadata,rows,admission))
        throw std::invalid_argument("shared draft Q requires admitted M16 workspace");
    if(admission==Exl3CudaLinearAdmission::target_continuation_head &&
       !target_head_small_m_candidate(metadata,rows,admission))
        throw std::invalid_argument("shared head requires admitted H6 small-M workspace");
    if (coherent_kv_split_for(metadata, rows, admission, in_features_, out_features_,
            accumulation_capacity_bytes_)) {
        transform_input(weights, metadata, input, rows, stream);
        forward_from_transformed(weights, metadata, transformed_, output,
                                 rows, stream, admission);
        return;
    }
    if (coherent_down_k6_candidate(metadata, rows, admission)) {
        // This explicit policy wins over older M1-only INT8/fused-input probes.
        // Scalar and verifier must not silently select different down math.
        transform_input(weights, metadata, input, rows, stream);
        forward_from_transformed(weights, metadata, transformed_, output,
                                 rows, stream, admission);
        return;
    }
    if (coherent_down_k7_candidate(metadata, rows, admission)) {
        transform_input(weights, metadata, input, rows, stream);
        forward_from_transformed(weights, metadata, transformed_, output,
                                 rows, stream, admission);
        return;
    }
    if (coherent_o_k7_candidate(metadata, rows, admission)) {
        transform_input(weights, metadata, input, rows, stream);
        forward_from_transformed(weights, metadata, transformed_, output,
                                 rows, stream, admission);
        return;
    }
    if (coherent_wide_k6_candidate(metadata, rows, admission)) {
        transform_input(weights, metadata, input, rows, stream);
        forward_from_transformed(weights, metadata, transformed_, output,
                                 rows, stream, admission);
        return;
    }
    // Admission failure must precede the input-Hadamard scratch write. The
    // submission path reacquires and holds this guard through slab consumption.
    if(extended_stream_reduction_candidate(metadata,rows,admission)) {
        cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream,&capture),"extended stream pre-transform capture query");
        if(capture!=cudaStreamCaptureStatusNone &&
           !host_kv_gdn_segment_graph_capture_ &&
           !ordinary_graph_stream_reduction_allowed())
            throw std::invalid_argument("extended stream graph composition unqualified");
    }
    if(reconstructed_exact_candidate(metadata,rows,admission) && reconstructed_exact_.ordered_stream) {
        reconstructed_exact_.ordered_stream->require_identity(reconstructed_exact_.model_owner,
            reconstructed_exact_.bytes,reconstructed_exact_.allow_k6_down,
            reconstructed_exact_.data,reconstructed_exact_.allow_k6_gate_up);
        reconstructed_exact_.ordered_stream->require_ordered(reinterpret_cast<std::uintptr_t>(stream));
    }
    std::unique_ptr<ninfer::NvtxRange> projection_range;
    if (std::getenv("NINFER_EXL3_PROJECTION_NVTX") != nullptr) {
        projection_range = std::make_unique<ninfer::NvtxRange>(
            std::string("exl3.projection.") + std::to_string(metadata.in_features) + "x" +
            std::to_string(metadata.out_features) + ".k" + std::to_string(metadata.K));
    }
    if (fast_same_weights_int8_gemv_enabled_ &&
        target_m1_k6_n16_candidate(metadata, rows, admission)) {
        if (!launch_fast_same_weights_int8_gemv<6>(
                input, weights, output, in_features_, out_features_, accum_,
                accumulation_capacity_bytes_, stream,
                fast_same_weights_int8_gemv_occupancy_grid_enabled_)) {
            throw std::runtime_error(
                "FAST_SAME_WEIGHTS INT8 GEMV workspace or shape unsupported");
        }
        process_fast_same_weights_int8_gemv_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    if (target_m1_int8_down_candidate(metadata, rows, admission)) {
        const bool launched = metadata.K == 6
            ? launch_fast_same_weights_int8_gemv<6>(
                input, weights, output, in_features_, out_features_, accum_,
                accumulation_capacity_bytes_, stream,
                fast_same_weights_int8_gemv_occupancy_grid_enabled_)
            : launch_fast_same_weights_int8_gemv<7>(
                input, weights, output, in_features_, out_features_, accum_,
                accumulation_capacity_bytes_, stream,
                fast_same_weights_int8_gemv_occupancy_grid_enabled_);
        if (!launched) {
            throw std::runtime_error(
                "FAST_SAME_WEIGHTS INT8 down GEMV workspace or shape unsupported");
        }
        if (metadata.K == 6)
            process_fast_same_weights_int8_gemv_down_k6_calls_.fetch_add(
                1, std::memory_order_relaxed);
        else
            process_fast_same_weights_int8_gemv_down_k7_calls_.fetch_add(
                1, std::memory_order_relaxed);
        return;
    }
    if (target_m1_k7_int8_candidate(metadata, rows, admission)) {
        if (!launch_fast_same_weights_int8_gemv<7>(
                input, weights, output, in_features_, out_features_, accum_,
                accumulation_capacity_bytes_, stream,
                fast_same_weights_int8_gemv_occupancy_grid_enabled_)) {
            throw std::runtime_error(
                "FAST_SAME_WEIGHTS INT8 GEMV K7 workspace or shape unsupported");
        }
        process_fast_same_weights_int8_gemv_k7_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    if (fast_native_mia_m1_fp16_fused_input_enabled_ &&
        fast_native_mia_m1_fp16_candidate(metadata, rows, admission)) {
        if (!weights.trellis || !weights.suh || !weights.svh) {
            throw std::invalid_argument(
                "native Mia-shaped fused-input route received incomplete weights");
        }
        const bool k6_gate = metadata.K == 6 && in_features_ == 5120 &&
            out_features_ == 17408;
        const int grid = fast_native_mia_m1_fp16_fused_input_grid_[metadata.K - 6];
        const int block_threads = k6_gate ? 256 : 512;
        const std::size_t shared_bytes =
            metadata.K == 6
                ? (k6_gate
                       ? exl3_native_persistent_smem_bytes<6, 16, 512>()
                       : exl3_native_persistent_smem_bytes<6, 32, 256>())
                : exl3_native_persistent_smem_bytes<7, 32, 128>();
        const half* raw_input = reinterpret_cast<const half*>(input);
        const half* suh = reinterpret_cast<const half*>(weights.suh);
        half* transformed = reinterpret_cast<half*>(transformed_);
        const std::uint16_t* trellis = weights.trellis;
        half* result = reinterpret_cast<half*>(output);
        int row_count = rows;
        int input_features = in_features_;
        int output_features = out_features_;
        int* locks = reinterpret_cast<int*>(accum_ + out_features_);
        const half* svh = reinterpret_cast<const half*>(weights.svh);
        cuda_check(cudaMemsetAsync(
                       locks, 0,
                       static_cast<std::size_t>(out_features_ / 16) *
                           sizeof(int), stream),
                   "clear native fused-input Mia-shaped M1 locks");
        void* kernel_args[] = {&raw_input, &suh, &transformed, &trellis,
                               &result, &row_count, &input_features,
                               &output_features, &locks, &svh};
        void* kernel = metadata.K == 6
            ? (k6_gate
                   ? reinterpret_cast<void*>(
                         exl3_native_mia_m1_fp16_fused_input_kernel<
                             6, 16, 512>)
                   : reinterpret_cast<void*>(
                         exl3_native_mia_m1_fp16_fused_input_kernel<
                             6, 32, 256>))
            : reinterpret_cast<void*>(
                  exl3_native_mia_m1_fp16_fused_input_kernel<
                      7, 32, 128, true>);
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(grid), dim3(block_threads), kernel_args,
                       shared_bytes, stream),
                   "launch native fused-input Mia-shaped M1 FP16 GEMM");
        cuda_check(cudaGetLastError(),
                   "launch native fused-input Mia-shaped M1 output");
        process_fast_native_mia_m1_fp16_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    const auto eager_on = [&](cudaStream_t submission_stream) {
        transform_input(weights, metadata, input, rows, submission_stream);
        forward_from_transformed(
            weights, metadata, transformed_, output, rows, submission_stream, admission);
    };
    const auto eager = [&] { eager_on(stream); };
    if (!prefill_projection_graph_candidate(metadata,rows,admission)) {
        eager();
        return;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream,&capture),
               "prefill projection graph capture query");
    // A larger retained graph owns capture when present. Nesting would replace
    // rather than preserve that graph's selected fast paths.
    if (capture != cudaStreamCaptureStatusNone) {
        eager();
        return;
    }

    const auto binding_matches = [&] {
        if (!prefill_projection_graph_binding_) return false;
        const auto& bound = *prefill_projection_graph_binding_;
        return bound.weights.trellis == weights.trellis &&
            bound.weights.suh == weights.suh && bound.weights.svh == weights.svh &&
            bound.weights.mul1 == weights.mul1 &&
            bound.metadata.in_features == metadata.in_features &&
            bound.metadata.out_features == metadata.out_features &&
            bound.metadata.K == metadata.K && bound.metadata.mcg == metadata.mcg &&
            bound.metadata.mul1 == metadata.mul1 &&
            bound.metadata.has_bias == metadata.has_bias &&
            bound.input == input && bound.output == output &&
            bound.stream == stream && bound.rows == rows &&
            bound.admission == admission;
    };
    if (prefill_projection_graph_executable_.ready()) {
        if (!binding_matches()) {
            ++prefill_projection_graph_binding_fallbacks_;
            eager();
            return;
        }
        prefill_projection_graph_executable_.launch(stream);
        ++prefill_projection_graph_replays_;
        return;
    }

    // CUDA forbids beginning capture on the legacy/default stream used by the
    // physical-C1 prefill caller. Record the same retained kernels on a
    // short-lived nonblocking stream; graph launch remains on the caller's
    // original stream, so its established ordering and ownership are intact.
    cudaStream_t capture_stream=nullptr;
    cuda_check(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking),
               "prefill projection graph capture stream create");
    try {
        prefill_projection_graph_definition_.capture(
            capture_stream,[&] { eager_on(capture_stream); });
    } catch (...) {
        (void)cudaStreamDestroy(capture_stream);
        throw;
    }
    cuda_check(cudaStreamDestroy(capture_stream),
               "prefill projection graph capture stream destroy");
    prefill_projection_graph_executable_.instantiate(
        prefill_projection_graph_definition_);
    prefill_projection_graph_executable_.upload(stream);
    prefill_projection_graph_binding_.emplace(PrefillProjectionGraphBinding{
        weights,metadata,input,output,stream,rows,admission});
    ++prefill_projection_graph_captures_;
    prefill_projection_graph_executable_.launch(stream);
    ++prefill_projection_graph_replays_;
}

void Exl3CudaLinearWorkspace::transform_input(const Exl3CudaLinearWeights& weights,
                                              const Exl3CudaLinearMetadata& metadata,
                                              const std::uint16_t* input,
                                              int rows,
                                              cudaStream_t stream) {
    if (metadata.in_features != in_features_ || metadata.out_features != out_features_) {
        throw std::invalid_argument("unsupported EXL3 CUDA dimensions");
    }
    if (metadata.K < 4 || metadata.K > 8) {
        throw std::invalid_argument("unsupported EXL3 CUDA module K");
    }
    if (!allow_generic_variants_ && metadata.K != 5) {
        throw std::invalid_argument("unsupported EXL3 CUDA module K for legacy workspace");
    }
    if (metadata.mcg || !metadata.mul1 || metadata.has_bias) {
        throw std::invalid_argument("unsupported EXL3 CUDA flags");
    }
    if (rows <= 0 || rows > max_rows_) {
        throw std::invalid_argument("EXL3 CUDA row count exceeds workspace");
    }
    if (weights.trellis == nullptr || weights.suh == nullptr || weights.svh == nullptr ||
        weights.mul1 == nullptr || input == nullptr) {
        throw std::invalid_argument("EXL3 CUDA received a null device buffer");
    }

    launch_input_hadamard<kHadamard>(stream,
        input, weights.suh, transformed_, rows, in_features_);
    cuda_check(cudaGetLastError(), "launch EXL3 input Hadamard");
}

void Exl3CudaLinearWorkspace::forward_k4_prefill_for_test(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
    std::uint16_t* output,int rows,cudaStream_t stream) {
    if(metadata.K!=4 || metadata.in_features!=in_features_ ||
       metadata.out_features!=out_features_ || metadata.mcg || !metadata.mul1 ||
       metadata.has_bias || rows<=16 || rows>max_rows_ ||
       !weights.trellis || !weights.suh || !weights.svh || !weights.mul1 ||
       !input || !output || out_features_%kHadamard!=0)
        throw std::invalid_argument("K4 prefill discriminator contract");
    transform_input(weights,metadata,input,rows,stream);
    const int full_groups=rows/16;
    for(int group=0;group<full_groups;++group) {
        const auto input_offset=static_cast<std::size_t>(group)*16u*in_features_;
        const auto output_offset=static_cast<std::size_t>(group)*16u*out_features_;
        exl3_initial16_ordered_kernel<4><<<out_features_/16,256,0,stream>>>(
            transformed_+input_offset,weights.trellis,weights.mul1,
            accum_+output_offset,in_features_,out_features_);
    }
    const int tail=rows-full_groups*16;
    if(tail) {
        const auto input_offset=static_cast<std::size_t>(full_groups)*16u*in_features_;
        const auto output_offset=static_cast<std::size_t>(full_groups)*16u*out_features_;
        exl3_generic_tile_kernel<<<dim3(out_features_/16,tail),dim3(kThreads),0,stream>>>(
            transformed_+input_offset,weights.trellis,weights.mul1,
            accum_+output_offset,tail,in_features_,out_features_,4);
    }
    cuda_check(cudaGetLastError(),"launch K4 prefill discriminator projection");
    launch_output_hadamard(stream,accum_,weights.svh,output,rows,out_features_);
    cuda_check(cudaGetLastError(),"launch K4 prefill discriminator output");
}

void Exl3CudaLinearWorkspace::forward_k4_prefill_mma_for_test(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
    std::uint16_t* output,int rows,cudaStream_t stream) {
    if(metadata.K!=4 || metadata.in_features!=in_features_ ||
       metadata.out_features!=out_features_ || metadata.mcg || !metadata.mul1 ||
       metadata.has_bias || rows<=16 || rows>max_rows_ ||
       !weights.trellis || !weights.suh || !weights.svh || !weights.mul1 ||
       !input || !output || out_features_%1024!=0)
        throw std::invalid_argument("K4 MMA prefill discriminator contract");
    transform_input(weights,metadata,input,rows,stream);
    constexpr int tiles=64,splits=5;
    const int blocks=out_features_/(16*tiles);
    constexpr std::size_t shared_bytes=512u*sizeof(half)+
        2u*tiles*16u*4u*sizeof(std::uint16_t);
    exl3_prefill_direct_async_a_kernel<4,tiles,2><<<
        dim3(blocks*splits,(rows+15)/16),kThreads,shared_bytes,stream>>>(
            transformed_,weights.trellis,weights.mul1,accum_,rows,
            in_features_,out_features_,splits);
    cuda_check(cudaGetLastError(),"launch K4 MMA prefill discriminator projection");
    launch_prefill_reduce_output<false>(stream,accum_,weights.svh,output,rows,out_features_,splits);
    cuda_check(cudaGetLastError(),"launch K4 MMA prefill discriminator output");
}

bool Exl3CudaLinearWorkspace::forward_target_prefill_gate_up_pair(
    Exl3CudaLinearWorkspace& up_workspace,
    const Exl3CudaLinearWeights& gate_weights,
    const Exl3CudaLinearMetadata& gate_metadata,
    const Exl3CudaLinearWeights& up_weights,
    const Exl3CudaLinearMetadata& up_metadata,
    const std::uint16_t* input,std::uint16_t* gate_output,
    std::uint16_t* up_output,int rows,cudaStream_t stream) {
    if(!target_prefill_gate_up_pair_)return false;
    ++target_prefill_gate_up_pair_attempts_;
    constexpr auto admission=Exl3CudaLinearAdmission::target_wide_prefill;
    const bool same_shape=in_features_==5120 && out_features_==17408 &&
        up_workspace.in_features_==in_features_ &&
        up_workspace.out_features_==out_features_ && rows>16 &&
        rows<=max_rows_ && rows<=up_workspace.max_rows_;
    const bool exact_k6=gate_metadata.K==6 && up_metadata.K==6 &&
        gate_metadata.mul1 && up_metadata.mul1 && !gate_metadata.mcg &&
        !up_metadata.mcg && !gate_metadata.has_bias && !up_metadata.has_bias &&
        gate_metadata.in_features==in_features_ &&
        up_metadata.in_features==in_features_ &&
        gate_metadata.out_features==out_features_ &&
        up_metadata.out_features==out_features_;
    const bool storage=input && gate_output && up_output &&
        gate_weights.trellis && up_weights.trellis && gate_weights.suh &&
        up_weights.suh && gate_weights.svh &&
        up_weights.svh && gate_weights.mul1 && up_weights.mul1 && accum_ &&
        up_workspace.accum_ && accum_!=up_workspace.accum_;
    const bool route=target_prefill_gate_up_pair_ &&
        up_workspace.target_prefill_gate_up_pair_ && same_shape && exact_k6 &&
        storage && target_direct_partials_ &&
        up_workspace.target_direct_partials_ && target_direct_async_a_ &&
        up_workspace.target_direct_async_a_ && target_direct_tiles64_ &&
        up_workspace.target_direct_tiles64_ &&
        target_staged_prefill_candidate(gate_metadata,rows,admission) &&
        up_workspace.target_staged_prefill_candidate(up_metadata,rows,admission);
    if(!route)return false;
    constexpr int tiles=64;
    const int blocks=out_features_/(16*tiles);
    const int capacity=std::min(generic_capacity_[1],
        up_workspace.generic_capacity_[1]);
    int splits=std::min(5,capacity/blocks);
    const char* requested=std::getenv("NINFER_EXL3_GENERIC_SPLITS");
    if(requested&&*requested)
        splits=std::min(splits,std::max(1,std::atoi(requested)));
    if(splits<1)return false;
    // The output planes are dead until their reductions. Use them as the two
    // exact transformed-A planes, then overwrite them only after the paired
    // partial kernel has consumed both on this ordered stream.
    exl3_transform_input_pair(gate_weights,gate_metadata,up_weights,up_metadata,
        input,gate_output,up_output,rows,stream);
    constexpr std::size_t shared_bytes=512u*sizeof(half)+
        2u*tiles*16u*6u*sizeof(std::uint16_t);
    exl3_prefill_k6_gate_up_pair_kernel<<<
        dim3(blocks*splits,(rows+15)/16),512,shared_bytes,stream>>>(
            gate_output,up_output,gate_weights.trellis,gate_weights.mul1,
            up_weights.trellis,up_weights.mul1,accum_,up_workspace.accum_,
            rows,in_features_,out_features_,splits);
    cuda_check(cudaGetLastError(),"launch paired target K6 gate/up partials");
    if(target_reduce_shfl_)
        launch_prefill_reduce_output<true>(stream,accum_,gate_weights.svh,gate_output,rows,
                out_features_,splits);
    else launch_prefill_reduce_output<false>(stream,accum_,gate_weights.svh,gate_output,rows,
                out_features_,splits);
    if(up_workspace.target_reduce_shfl_)
        launch_prefill_reduce_output<true>(stream,up_workspace.accum_,up_weights.svh,up_output,
                rows,out_features_,splits);
    else launch_prefill_reduce_output<false>(stream,up_workspace.accum_,up_weights.svh,up_output,
                rows,out_features_,splits);
    cuda_check(cudaGetLastError(),"launch paired target K6 gate/up reductions");
    ++target_prefill_gate_up_pair_calls_;
    target_prefill_gate_up_pair_rows_+=static_cast<std::uint64_t>(rows);
    return true;
}

void Exl3CudaLinearWorkspace::forward_target_m1_gate_up_pair(
    Exl3CudaLinearWorkspace& up_workspace,
    const Exl3CudaLinearWeights& gate_weights,
    const Exl3CudaLinearMetadata& gate_metadata,
    const Exl3CudaLinearWeights& up_weights,
    const Exl3CudaLinearMetadata& up_metadata,
    const std::uint16_t* input,std::uint16_t* gate_output,
    std::uint16_t* up_output,cudaStream_t stream) {
    const bool valid=in_features_==5120&&out_features_==17408&&
        up_workspace.in_features_==in_features_&&
        up_workspace.out_features_==out_features_&&
        (gate_metadata.K==5 || gate_metadata.K==6 || gate_metadata.K==7) &&
        up_metadata.K==gate_metadata.K&&gate_metadata.mul1&&
        up_metadata.mul1&&!gate_metadata.mcg&&!up_metadata.mcg&&
        !gate_metadata.has_bias&&!up_metadata.has_bias&&
        gate_metadata.in_features==in_features_&&
        up_metadata.in_features==in_features_&&
        gate_metadata.out_features==out_features_&&
        up_metadata.out_features==out_features_&&input&&gate_output&&up_output&&
        gate_weights.trellis&&gate_weights.suh&&gate_weights.svh&&
        gate_weights.mul1&&up_weights.trellis&&up_weights.suh&&
        up_weights.svh&&up_weights.mul1&&accum_&&up_workspace.accum_&&
        accum_!=up_workspace.accum_;
    if(!valid)throw std::invalid_argument("target M1 gate/up pair discriminator contract");
    constexpr int splits=5,tiles=16;
    if (fast_same_weights_fp16kv_m1_mgemm_pair_enabled_ &&
        fast_same_weights_fp16kv_m1_mgemm_pair_grid_[gate_metadata.K - 5] > 0) {
        // Keep transformed A in the workspaces: the MGEMM inner writes its
        // final raw C plane, so using gate_output as A would create an
        // in-place read/write hazard while K slices are still resident.
        exl3_transform_input_pair(
            gate_weights, gate_metadata, up_weights, up_metadata, input,
            transformed_, up_workspace.transformed_, 1, stream);
        const int grid_x = std::min(
            (in_features_ / 16) * (out_features_ / 512),
            fast_same_weights_fp16kv_m1_mgemm_pair_grid_[gate_metadata.K - 5]);
        int* gate_locks = reinterpret_cast<int*>(accum_ + out_features_);
        int* up_locks = reinterpret_cast<int*>(up_workspace.accum_ + out_features_);
        cuda_check(cudaMemsetAsync(
                       gate_locks, 0,
                       static_cast<std::size_t>(out_features_ / 16 + 4) *
                           sizeof(int),
                       stream),
                   "clear native M1 MGEMM-pair gate locks");
        cuda_check(cudaMemsetAsync(
                       up_locks, 0,
                       static_cast<std::size_t>(out_features_ / 16 + 4) *
                           sizeof(int),
                       stream),
                   "clear native M1 MGEMM-pair up locks");
        const half* first_transformed = reinterpret_cast<const half*>(transformed_);
        const std::uint16_t* first_trellis = gate_weights.trellis;
        half* first_output = reinterpret_cast<half*>(gate_output);
        const half* second_transformed =
            reinterpret_cast<const half*>(up_workspace.transformed_);
        const std::uint16_t* second_trellis = up_weights.trellis;
        half* second_output = reinterpret_cast<half*>(up_output);
        const half* first_svh = reinterpret_cast<const half*>(gate_weights.svh);
        const half* second_svh = reinterpret_cast<const half*>(up_weights.svh);
        int rows = 1;
        int input_features = in_features_;
        int output_features = out_features_;
        void* kernel_args[] = {
            &first_transformed, &first_trellis, &first_output, &gate_locks,
            &first_svh, &second_transformed, &second_trellis, &second_output,
            &up_locks, &second_svh, &rows, &input_features, &output_features};
        void* kernel = gate_metadata.K == 5
            ? reinterpret_cast<void*>(
                  exl3_native_mia_m1_mgemm_pair_kernel<5, 16, 512>)
            : gate_metadata.K == 6
                ? reinterpret_cast<void*>(
                      exl3_native_mia_m1_mgemm_pair_kernel<6, 16, 512>)
                : reinterpret_cast<void*>(
                      exl3_native_mia_m1_mgemm_pair_kernel<7, 32, 128, true>);
        const std::size_t shared_bytes = gate_metadata.K == 5
            ? exl3_native_persistent_smem_bytes<5, 16, 512>()
            : gate_metadata.K == 6
                ? exl3_native_persistent_smem_bytes<6, 16, 512>()
                : exl3_native_persistent_smem_bytes<7, 32, 128>();
        const int block_threads = gate_metadata.K == 7 ? 512 : 256;
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(grid_x, 1, 2), dim3(block_threads), kernel_args,
                       shared_bytes, stream),
                   "launch native Mia-shaped M1 MGEMM gate/up pair");
        cuda_check(cudaGetLastError(),
                   "launch native M1 MGEMM-pair output Hadamard");
        process_fast_same_weights_fp16kv_m1_gate_up_pair_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    const bool wide_gate_up_pair = [&]() {
        const char* requested = std::getenv(
            "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_GATE_UP_N64");
        if (requested && std::strcmp(requested, "0") != 0 &&
            std::strcmp(requested, "1") != 0)
            throw std::invalid_argument(
                "M1 gate/up N64 route must be 0 or 1");
        return requested && std::strcmp(requested, "1") == 0;
    }();
    exl3_transform_input_pair(gate_weights,gate_metadata,up_weights,up_metadata,
        input,gate_output,up_output,1,stream);
    if (wide_gate_up_pair) {
        constexpr int wide_tiles = 64;
        const int wide_blocks = out_features_ / (16 * wide_tiles);
        const std::size_t shared_bytes =
            (512u + 2u * wide_tiles * 16u * 6u) * sizeof(half);
        exl3_prefill_k6_gate_up_pair_kernel<<<
            dim3(wide_blocks * splits, 1), 512, shared_bytes, stream>>>(
                gate_output, up_output, gate_weights.trellis,
                gate_weights.mul1, up_weights.trellis, up_weights.mul1,
                accum_, up_workspace.accum_, 1, in_features_, out_features_,
                splits);
        cuda_check(cudaGetLastError(),
                   "launch target M1 wide K6 gate/up pair discriminator");
    } else {
        const int blocks=out_features_/(16*tiles);
        const std::size_t shared_bytes =
            (2u*2u*256u+2u*2u*tiles*16u*gate_metadata.K)*sizeof(half);
        if (gate_metadata.K == 5) {
            exl3_target_m1_projection_pair_n16_kernel<5><<<
                blocks*splits,512,shared_bytes,stream>>>(
                    gate_output,up_output,gate_weights.trellis,gate_weights.mul1,
                    up_weights.trellis,up_weights.mul1,accum_,up_workspace.accum_,
                    in_features_,out_features_,splits);
        } else if (gate_metadata.K == 6) {
            exl3_target_m1_projection_pair_n16_kernel<6><<<
                blocks*splits,512,shared_bytes,stream>>>(
                    gate_output,up_output,gate_weights.trellis,gate_weights.mul1,
                    up_weights.trellis,up_weights.mul1,accum_,up_workspace.accum_,
                    in_features_,out_features_,splits);
        } else {
            exl3_target_m1_projection_pair_n16_kernel<7><<<
                blocks*splits,512,shared_bytes,stream>>>(
                    gate_output,up_output,gate_weights.trellis,gate_weights.mul1,
                    up_weights.trellis,up_weights.mul1,accum_,up_workspace.accum_,
                    in_features_,out_features_,splits);
        }
        cuda_check(cudaGetLastError(),"launch target M1 packed gate/up pair discriminator");
    }
    launch_prefill_reduce_output<false>(stream,accum_,gate_weights.svh,gate_output,1,
            out_features_,splits);
    launch_prefill_reduce_output<false>(stream,up_workspace.accum_,up_weights.svh,up_output,1,
            out_features_,splits);
    cuda_check(cudaGetLastError(),"reduce target M1 K6 gate/up pair discriminator");
    process_fast_same_weights_fp16kv_m1_gate_up_pair_calls_.fetch_add(
        1,std::memory_order_relaxed);
}

void Exl3CudaLinearWorkspace::forward_target_m1_gate_up_pair_for_test(
    Exl3CudaLinearWorkspace& up_workspace,
    const Exl3CudaLinearWeights& gate_weights,
    const Exl3CudaLinearMetadata& gate_metadata,
    const Exl3CudaLinearWeights& up_weights,
    const Exl3CudaLinearMetadata& up_metadata,
    const std::uint16_t* input,std::uint16_t* gate_output,
    std::uint16_t* up_output,cudaStream_t stream) {
    forward_target_m1_gate_up_pair(
        up_workspace,gate_weights,gate_metadata,up_weights,up_metadata,
        input,gate_output,up_output,stream);
}

void Exl3CudaLinearWorkspace::forward_target_m1_kv_pair_for_test(
    Exl3CudaLinearWorkspace& second_workspace,
    const Exl3CudaLinearWeights& first_weights,
    const Exl3CudaLinearMetadata& first_metadata,
    const Exl3CudaLinearWeights& second_weights,
    const Exl3CudaLinearMetadata& second_metadata,
    const std::uint16_t* input,std::uint16_t* first_output,
    std::uint16_t* second_output,cudaStream_t stream) {
    const bool valid=in_features_==5120 && out_features_==1024 &&
        second_workspace.in_features_==in_features_ &&
        second_workspace.out_features_==out_features_ &&
        (first_metadata.K==6 || first_metadata.K==7) &&
        second_metadata.K==first_metadata.K && first_metadata.mul1 &&
        second_metadata.mul1 && !first_metadata.mcg && !second_metadata.mcg &&
        !first_metadata.has_bias && !second_metadata.has_bias &&
        first_metadata.in_features==in_features_ &&
        second_metadata.in_features==in_features_ &&
        first_metadata.out_features==out_features_ &&
        second_metadata.out_features==out_features_ && input && first_output &&
        second_output && first_output!=second_output && first_weights.trellis &&
        first_weights.suh && first_weights.svh && first_weights.mul1 &&
        second_weights.trellis && second_weights.suh && second_weights.svh &&
        second_weights.mul1 && accum_ && second_workspace.accum_ &&
        accum_!=second_workspace.accum_;
    if(!valid)throw std::invalid_argument("target M1 K/V pair discriminator contract");
    constexpr int splits=5,tiles=16;
    const int blocks=out_features_/(16*tiles);
    exl3_transform_input_pair(first_weights,first_metadata,second_weights,
        second_metadata,input,first_output,second_output,1,stream);
    const std::size_t shared_bytes=(2u*2u*256u+
        2u*2u*tiles*16u*static_cast<std::size_t>(first_metadata.K))*sizeof(half);
    if(first_metadata.K==6)
        exl3_target_m1_projection_pair_n16_kernel<6><<<
            blocks*splits,512,shared_bytes,stream>>>(
                first_output,second_output,first_weights.trellis,first_weights.mul1,
                second_weights.trellis,second_weights.mul1,accum_,
                second_workspace.accum_,in_features_,out_features_,splits);
    else
        exl3_target_m1_projection_pair_n16_kernel<7><<<
            blocks*splits,512,shared_bytes,stream>>>(
                first_output,second_output,first_weights.trellis,first_weights.mul1,
                second_weights.trellis,second_weights.mul1,accum_,
                second_workspace.accum_,in_features_,out_features_,splits);
    cuda_check(cudaGetLastError(),"launch target M1 K6 K/V pair discriminator");
    launch_prefill_reduce_output<false>(stream,accum_,first_weights.svh,first_output,1,
            out_features_,splits);
    launch_prefill_reduce_output<false>(stream,second_workspace.accum_,second_weights.svh,
            second_output,1,out_features_,splits);
    cuda_check(cudaGetLastError(),"reduce target M1 K6 K/V pair discriminator");
    process_fast_same_weights_fp16kv_m1_kv_pair_calls_.fetch_add(
        1,std::memory_order_relaxed);
}

void Exl3CudaLinearWorkspace::forward_target_m1_kv_wide_pair_for_test(
    Exl3CudaLinearWorkspace& second_workspace,
    const Exl3CudaLinearWeights& first_weights,
    const Exl3CudaLinearMetadata& first_metadata,
    const Exl3CudaLinearWeights& second_weights,
    const Exl3CudaLinearMetadata& second_metadata,
    const std::uint16_t* input,std::uint16_t* first_output,
    std::uint16_t* second_output,cudaStream_t stream) {
    const bool valid=in_features_==5120 && out_features_==1024 &&
        second_workspace.in_features_==in_features_ &&
        second_workspace.out_features_==out_features_ &&
        (first_metadata.K==6 || first_metadata.K==7) &&
        second_metadata.K==first_metadata.K && first_metadata.mul1 &&
        second_metadata.mul1 && !first_metadata.mcg && !second_metadata.mcg &&
        !first_metadata.has_bias && !second_metadata.has_bias &&
        first_metadata.in_features==in_features_ &&
        second_metadata.in_features==in_features_ &&
        first_metadata.out_features==out_features_ &&
        second_metadata.out_features==out_features_ && input && first_output &&
        second_output && first_output!=second_output && first_weights.trellis &&
        first_weights.suh && first_weights.svh && first_weights.mul1 &&
        second_weights.trellis && second_weights.suh && second_weights.svh &&
        second_weights.mul1 && accum_ && second_workspace.accum_ &&
        accum_!=second_workspace.accum_;
    if(!valid)throw std::invalid_argument(
        "target M1 wide K/V pair discriminator contract");
    constexpr int splits=5,tiles=16,output_tile_elements=tiles*16;
    const int blocks=out_features_/(16*tiles);
    exl3_transform_input_pair(first_weights,first_metadata,second_weights,
        second_metadata,input,first_output,second_output,1,stream);
    const std::size_t raw_stage_half=static_cast<std::size_t>(tiles)*16u*
        static_cast<std::size_t>(first_metadata.K);
    const std::size_t shared_bytes=(4u*256u+4u*raw_stage_half)*sizeof(half)+
        static_cast<std::size_t>(2u*output_tile_elements)*sizeof(float);
    if(first_metadata.K==6)
        exl3_target_m1_wide_projection_pair_n16_kernel<6><<<
            dim3(blocks*splits,1,2),512,shared_bytes,stream>>>(
                first_output,second_output,first_weights.trellis,first_weights.mul1,
                second_weights.trellis,second_weights.mul1,accum_,
                second_workspace.accum_,in_features_,out_features_,splits);
    else
        exl3_target_m1_wide_projection_pair_n16_kernel<7><<<
            dim3(blocks*splits,1,2),512,shared_bytes,stream>>>(
                first_output,second_output,first_weights.trellis,first_weights.mul1,
                second_weights.trellis,second_weights.mul1,accum_,
                second_workspace.accum_,in_features_,out_features_,splits);
    cuda_check(cudaGetLastError(),"launch target M1 wide K6/K7 K/V pair");
    launch_prefill_reduce_output<false>(stream,accum_,first_weights.svh,first_output,1,
            out_features_,splits);
    launch_prefill_reduce_output<false>(stream,second_workspace.accum_,second_weights.svh,
            second_output,1,out_features_,splits);
    cuda_check(cudaGetLastError(),"reduce target M1 wide K6/K7 K/V pair");
    process_fast_same_weights_fp16kv_m1_kv_pair_calls_.fetch_add(
        1,std::memory_order_relaxed);
}

void Exl3CudaLinearWorkspace::transform_gate_up(const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,const std::uint16_t* gate,
    const std::uint16_t* up,std::uint16_t* activation,int rows,cudaStream_t stream,
    Exl3CudaLinearAdmission admission) {
    if(metadata.in_features!=in_features_ || metadata.out_features!=out_features_ ||
       metadata.K<4 || metadata.K>8 || (!allow_generic_variants_ && metadata.K!=5) ||
       metadata.mcg || !metadata.mul1 || metadata.has_bias || rows<=0 || rows>max_rows_ ||
       !weights.trellis || !weights.suh || !weights.svh || !weights.mul1)
        throw std::invalid_argument("unsupported gate/up transformed projection");
    if(reconstructed_exact_candidate(metadata,rows,admission) && reconstructed_exact_.ordered_stream) {
        reconstructed_exact_.ordered_stream->require_identity(reconstructed_exact_.model_owner,
            reconstructed_exact_.bytes,reconstructed_exact_.allow_k6_down,
            reconstructed_exact_.data,reconstructed_exact_.allow_k6_gate_up);
        reconstructed_exact_.ordered_stream->require_ordered(reinterpret_cast<std::uintptr_t>(stream));
    }
    for(const auto* input:{gate,up})
        exl3_require_paired_transform_extents(rows,in_features_,
            {reinterpret_cast<std::uintptr_t>(input),reinterpret_cast<std::uintptr_t>(weights.suh),
             reinterpret_cast<std::uintptr_t>(weights.suh),reinterpret_cast<std::uintptr_t>(activation),
             reinterpret_cast<std::uintptr_t>(transformed_)});
    launch_input_hadamard<kHadamard,false,true>(stream,gate,weights.suh,transformed_,rows,in_features_,up,activation);
    cuda_check(cudaGetLastError(),"launch EXL3 gate/up activation input Hadamard");
}

bool Exl3CudaLinearWorkspace::draft_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows) const noexcept {
    return draft_small_m_enabled_ && rows >= 2 && rows <= 8 && rows<=max_rows_ &&
        metadata.in_features==in_features_ && metadata.out_features==out_features_ &&
        metadata.K == 5 && !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        ((in_features_ == 5120 && out_features_ == 4096) ||
         (in_features_ == 5120 && out_features_ == 1024) ||
         (in_features_ == 4096 && out_features_ == 5120) ||
         (in_features_ == 17408 && out_features_ == 5120));
}

bool Exl3CudaLinearWorkspace::draft_prefill_fc_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows, Exl3CudaLinearAdmission admission) const noexcept {
    return draft_prefill_fc_enabled_ && admission == Exl3CudaLinearAdmission::draft_prefill_fc &&
        rows >= 2 && rows <= 8 && rows <= max_rows_ && metadata.K == 5 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        in_features_ == 25600 && out_features_ == 5120 &&
        metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
        generic_capacity_[0] >= out_features_ / (32 * 16);
}

bool Exl3CudaLinearWorkspace::draft_shared_q_m16_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows, Exl3CudaLinearAdmission admission) const noexcept {
    const auto capacity=draft_k5_async_a_enabled_?draft_k5_async_a_capacity_:generic_capacity_[0];
    return draft_small_m_enabled_ && admission == Exl3CudaLinearAdmission::draft_shared_q_m16 &&
        rows==16 && rows<=max_rows_ && metadata.K==5 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        in_features_==5120 && out_features_==4096 &&
        metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
        capacity>=out_features_/(32*16) && generic_capacity_[0]>=out_features_/(32*16);
}

bool Exl3CudaLinearWorkspace::target_rowpair_k6_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows, Exl3CudaLinearAdmission admission) const noexcept {
    return target_rowpair_k6_ && target_direct_partials_ && target_direct_async_a_ &&
        rows >= 32 && metadata.K == 6 && !(in_features_ == 17408 && out_features_ == 5120) &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_staged_prefill_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_staged_prefill_enabled_ && target_wide_prefill_candidate(metadata, rows, admission) &&
        (!(specialized_shape_ && metadata.K == kK) || target_staged_shape4_);
}

bool Exl3CudaLinearWorkspace::target_k6_fast_decode_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_k6_fast_decode_ && target_direct_partials_ &&
        target_direct_async_a_ && target_direct_tiles64_ && metadata.K==6 &&
        in_features_!=17408 && !target_rowpair_k6_ &&
        !target_k6_gateup_warpgroup_async_ && !target_k6_gateup_n32_pair_cta_ &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_k6_rowpair_n64_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_k6_rowpair_n64_ && target_direct_partials_ &&
        target_direct_async_a_ && target_direct_tiles64_ && rows>=32 &&
        metadata.K==6 && in_features_!=17408 && !target_rowpair_k6_ &&
        !target_k6_gateup_warpgroup_async_ && !target_k6_gateup_n32_pair_cta_ &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_k6_down_rowpair_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_k6_down_rowpair_ && target_direct_partials_ &&
        target_direct_async_a_ && rows>=32 && metadata.K==6 &&
        in_features_==17408 && out_features_==5120 &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_shape4_n64_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_shape4_n64_ && target_direct_partials_ && rows>=32 &&
        specialized_shape_ && metadata.K==kK &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::prefill_projection_graph_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // The current physical-C1 cold path has three full 1024-row chunks. Keep
    // the first experiment bounded to that repeated, pointer-stable extent.
    // Reconstructed Down owns a mutable decoded slab and explicitly rejects
    // capture, so it remains on its established ordered-stream route.
    // The broad K5/K6/K7/K8 integration screen corrupted downstream target
    // state even though the first observed real-caller K6 operator was exact.
    // Keep the one remaining integration experiment on that proven family;
    // the reconstructed K6 Down route has independent mutable slab ownership
    // and remains excluded.  Other quantization and shape4 families stay eager.
    const bool isolated_exact_k6 = metadata.K == 6 &&
        !(in_features_ == 17408 && out_features_ == 5120) &&
        !(specialized_shape_ && metadata.K == kK);
    return prefill_projection_graph_enabled_ && isolated_exact_k6 && rows == 1024 &&
        admission == Exl3CudaLinearAdmission::target_wide_prefill &&
        target_staged_prefill_candidate(metadata,rows,admission) &&
        !reconstructed_exact_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_prefill_persisting_l2_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return prefill_persisting_l2_enabled_ && !prefill_projection_graph_enabled_ &&
        rows==1024 && metadata.K==6 && metadata.mul1 && !metadata.mcg &&
        !metadata.has_bias && metadata.in_features==in_features_ &&
        metadata.out_features==out_features_ && in_features_!=17408 &&
        target_direct_partials_ && target_direct_async_a_ && target_direct_tiles64_ &&
        !target_rowpair_k6_ && !target_k6_gateup_warpgroup_async_ &&
        !target_k6_gateup_n32_pair_cta_ &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::apply_prefill_persisting_l2(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,cudaStream_t stream) {
    ++prefill_persisting_l2_eligible_;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(stream,&capture)!=cudaSuccess ||
       capture!=cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError();
        ++prefill_persisting_l2_unsupported_;
        return false;
    }
    int device=0,maximum_window=0;
    if(cudaGetDevice(&device)!=cudaSuccess ||
       cudaDeviceGetAttribute(&maximum_window,
           cudaDevAttrMaxAccessPolicyWindowSize,device)!=cudaSuccess ||
       maximum_window<=0 || !prefill_persisting_l2_set_aside_bytes_) {
        (void)cudaGetLastError();
        ++prefill_persisting_l2_unsupported_;
        return false;
    }
    const auto trellis_bytes=static_cast<std::size_t>(metadata.in_features)*
        static_cast<std::size_t>(metadata.out_features)*
        static_cast<std::size_t>(metadata.K)/8u;
    const auto set_aside=prefill_persisting_l2_set_aside_bytes_;
    const auto window=std::min({trellis_bytes,
        static_cast<std::size_t>(maximum_window),set_aside});
    if(!weights.trellis || !window) {
        ++prefill_persisting_l2_unsupported_;
        return false;
    }
    cudaStreamAttrValue attribute{};
    attribute.accessPolicyWindow.base_ptr=
        const_cast<std::uint16_t*>(weights.trellis);
    attribute.accessPolicyWindow.num_bytes=window;
    attribute.accessPolicyWindow.hitRatio=1.0f;
    attribute.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    attribute.accessPolicyWindow.missProp=cudaAccessPropertyNormal;
    if(cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,
        &attribute)!=cudaSuccess) {
        (void)cudaGetLastError();
        ++prefill_persisting_l2_unsupported_;
        return false;
    }
    prefill_persisting_l2_window_bytes_=window;
    prefill_persisting_l2_set_aside_bytes_=set_aside;
    ++prefill_persisting_l2_applied_;
    return true;
}

bool Exl3CudaLinearWorkspace::target_initial16_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!target_initial16_enabled_ || admission != Exl3CudaLinearAdmission::target_initial16 ||
        rows != 16 || rows > max_rows_ || metadata.K < 5 || metadata.K > 8 ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ || metadata.out_features != out_features_) return false;
    // Preserve existing shape4 and qualified K6 gate/up routes. The vocabulary
    // head is never admitted, and continuation/verifier sites use other tags.
    if (in_features_ == 5120 && out_features_ == 17408 &&
        (metadata.K == 5 || metadata.K == 6)) return false;
    return (in_features_ == 5120 && (out_features_ == 1024 || out_features_ == 6144 ||
            out_features_ == 10240 || out_features_ == 12288 || out_features_ == 17408)) ||
           ((in_features_ == 6144 || in_features_ == 17408) && out_features_ == 5120);
}

bool Exl3CudaLinearWorkspace::target_gateup_m16_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_gateup_m16_enabled_ &&
        admission == Exl3CudaLinearAdmission::target_prefill_gate_up &&
        rows == 16 && rows <= max_rows_ && metadata.K == 6 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
        in_features_ == 5120 && out_features_ == 17408;
}

bool Exl3CudaLinearWorkspace::target_gateup_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    const bool enabled =
        (metadata.K == 6 && target_gateup_small_m_enabled_) ||
        (metadata.K == 7 && target_gateup_k7_small_m_enabled_);
    return enabled &&
        admission == Exl3CudaLinearAdmission::target_continuation_gate_up &&
        rows >= 2 && rows <= native_continuation_rows_ && rows <= max_rows_ &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        metadata.in_features == in_features_ &&
        metadata.out_features == out_features_ && in_features_ == 5120 &&
        out_features_ == 17408 &&
        (out_features_ / (32 * 16)) <= generic_capacity_[metadata.K == 6 ? 1 : 2];
}

bool Exl3CudaLinearWorkspace::target_gateup_k5_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_gateup_k5_small_m_enabled_ &&
        admission == Exl3CudaLinearAdmission::target_continuation_gate_up &&
        rows >= 2 && rows <= native_continuation_rows_ && rows <= max_rows_ && metadata.K == 5 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        metadata.in_features == in_features_ &&
        metadata.out_features == out_features_ && in_features_ == 5120 &&
        out_features_ == 17408 && specialized_shape_;
}

bool Exl3CudaLinearWorkspace::target_down_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    const bool enabled = (metadata.K == 6 && target_down_small_m_enabled_) ||
        (metadata.K == 7 && target_down_k7_small_m_enabled_);
    if (!enabled ||
        admission != Exl3CudaLinearAdmission::target_continuation_down ||
        rows < 2 || rows > native_continuation_rows_ || rows > max_rows_ ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ || in_features_ != 17408 ||
        out_features_ != 5120) {
        return false;
    }
    const char* requested = std::getenv("NINFER_EXL3_LARGE_DOWN_TOPOLOGY");
    const bool topology16 = requested == nullptr || *requested == '\0' ||
        std::strcmp(requested, "16") == 0 ||
        (metadata.K == 6 && std::strcmp(requested, "16-k6") == 0) ||
        (metadata.K == 7 && std::strcmp(requested, "16-k7") == 0);
    const int output_blocks = out_features_ / (16 * 16);
    const int capacity = metadata.K == 7 && target_down_k7_async_a_enabled_
        ? target_down_k7_async_a_capacity_
        : large_down_candidate_capacity_[metadata.K == 6 ? 0 : 1];
    return topology16 && output_blocks <= capacity;
}

bool Exl3CudaLinearWorkspace::target_k5_small_m_batch_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!target_k5_small_m_batch_enabled_ || rows < 2 ||
        rows > native_continuation_rows_ || rows > max_rows_ ||
        metadata.K != 5 || metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        out_features_ % (32 * 16) != 0 ||
        out_features_ / (32 * 16) > generic_capacity_[0]) {
        return false;
    }
    // These are the established target-continuation families that currently
    // serialize the same 2..8 verifier rows into independent M1 calls.  Gate/up
    // K5 keeps its qualified shape4 owner and is intentionally excluded.
    switch (admission) {
    case Exl3CudaLinearAdmission::target_continuation_q:
        return in_features_ == 5120 && out_features_ == 12288;
    case Exl3CudaLinearAdmission::target_continuation_qkv:
        return in_features_ == 5120 && out_features_ == 10240;
    case Exl3CudaLinearAdmission::target_continuation_z:
        return in_features_ == 5120 && out_features_ == 6144;
    case Exl3CudaLinearAdmission::target_continuation_kv:
        return in_features_ == 5120 && out_features_ == 1024;
    case Exl3CudaLinearAdmission::target_continuation_o:
        return in_features_ == 6144 && out_features_ == 5120;
    case Exl3CudaLinearAdmission::target_continuation_down:
        return in_features_ == 17408 && out_features_ == 5120;
    default:
        return false;
    }
}

bool Exl3CudaLinearWorkspace::target_k6_m1_simt_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if(!target_k6_m1_simt_enabled_ || rows!=1 || rows>max_rows_ ||
       metadata.K!=6 || metadata.mcg || !metadata.mul1 || metadata.has_bias ||
       metadata.in_features!=in_features_ ||
       metadata.out_features!=out_features_ || out_features_%256!=0 ||
       in_features_==17408 || out_features_==248320 || specialized_shape_)
        return false;
    switch(admission) {
    case Exl3CudaLinearAdmission::target_continuation_q:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_o:
        return true;
    default:
        return false;
    }
}

bool Exl3CudaLinearWorkspace::target_m1_k6_n16_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if(!target_m1_k6_n16_enabled_ ||
       !target_m1_k6_n32_async_a_enabled_ || rows!=1 || rows>max_rows_ ||
       metadata.K!=6 || metadata.mcg || !metadata.mul1 || metadata.has_bias ||
       metadata.in_features!=in_features_ ||
       metadata.out_features!=out_features_ || out_features_%256!=0 ||
       (in_features_==17408 && out_features_==5120) ||
       out_features_==248320)
         return false;
    // Mia's pinned Blackwell policy only unfuses K6 when the single matrix is
    // wide enough to fill the device.  Narrow K6 tensors use the packed
    // MGEMM/GEMV fallback, preserving the same represented EXL3 weights and
    // avoiding a broad INT8-activation substitution.
    if (fast_same_weights_int8_mia_policy_enabled_ && out_features_ < 8192)
        return false;
    switch(admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_q:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_o:
        return true;
    default:
        return false;
    }
}

bool Exl3CudaLinearWorkspace::target_fast_same_weights_fp16_m1_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // This is a separately labeled native path: packed EXL3 weights are
    // decoded as represented FP16 operands and the MMA accumulator remains
    // FP32. It is intentionally independent of the FP16-accumulator and INT8
    // activation experiments above/below.
    if (!fast_same_weights_fp16_m1_enabled_ || !allow_generic_variants_ ||
        rows != 1 || rows > max_rows_ || metadata.mcg || !metadata.mul1 ||
        metadata.has_bias || metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        (in_features_ != 5120 && in_features_ != 17408) ||
        out_features_ % (32 * 16) != 0 || out_features_ == 248320 ||
        (metadata.K != 6 && metadata.K != 7))
        return false;
    switch (admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_q:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_o:
        break;
    default:
        return false;
    }
    const bool k6_down = metadata.K == 6 && in_features_ == 17408 &&
        out_features_ == 5120;
    const int output_tiles_per_block = k6_down ? 16 : 32;
    const int output_blocks = out_features_ / (16 * output_tiles_per_block);
    // N16 has its own occupancy table, but the established wide M1 path must
    // remain selectable when N16 is disabled.  Consulting the N16-only table
    // in that mode leaves its capacity at zero and silently demotes the
    // original exact baseline to generic GEMV.
    const auto& capacities = fast_same_weights_fp16_m1_n16_enabled_
        ? fast_same_weights_fp16_m1_n16_capacity_
        : fast_same_weights_fp16_m1_capacity_;
    const int capacity = metadata.K == 7
        ? capacities[2]
        : capacities[k6_down ? 1 : 0];
    return output_blocks > 0 && output_blocks <= capacity;
}

bool Exl3CudaLinearWorkspace::fast_fp16_m2_8_down_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // One M16 MMA consumes all proposal rows and decodes each packed B tile
    // once. The K6 down projection owns this first bounded vertical slice.
    return fast_fp16_m2_8_down_enabled_ && allow_generic_variants_ &&
        rows>=2 && rows<=8 && rows<=max_rows_ &&
        (admission==Exl3CudaLinearAdmission::target_continuation_down ||
         admission==Exl3CudaLinearAdmission::ordinary) &&
        in_features_==17408 && out_features_==5120 &&
        metadata.in_features==in_features_ &&
        metadata.out_features==out_features_ && metadata.K==6 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        fast_same_weights_fp16_m1_capacity_[1]>=20;
}

bool Exl3CudaLinearWorkspace::fast_fp16_m2_8_fused_down_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return fast_fp16_m2_8_fused_down_enabled_ && allow_generic_variants_ &&
        rows>=2 && rows<=8 && rows<=max_rows_ &&
        (admission==Exl3CudaLinearAdmission::target_continuation_down ||
         admission==Exl3CudaLinearAdmission::ordinary) &&
        in_features_==17408 && out_features_==5120 &&
        metadata.in_features==in_features_ &&
        metadata.out_features==out_features_ && metadata.K==6 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias;
}

std::size_t Exl3CudaLinearWorkspace::coherent_wide_k6_shared_bytes_for_test() noexcept {
    return kCoherentWideK6SharedBytes;
}

// NINFER_EXL3_COHERENT_ANY_K (default 1; quality-gated) also admits K5 and K7 weights to the coherent
// wide and down producers (numerics policy: split-plane FP32 accumulation).
static bool coherent_any_k_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_EXL3_COHERENT_ANY_K");
        if (!value || std::strcmp(value, "1") == 0) return true;
        if (std::strcmp(value, "0") == 0) return false;
        throw std::invalid_argument("NINFER_EXL3_COHERENT_ANY_K must be 0 or 1");
    }();
    return enabled;
}

bool Exl3CudaLinearWorkspace::coherent_wide_k6_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!coherent_wide_k6_enabled_ || !allow_generic_variants_ ||
        coherent_wide_k6_resident_capacity_ <= 0 ||
        rows < 1 || rows > 8 || rows > max_rows_ ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        !(metadata.K == 6 ||
          (coherent_any_k_enabled() && (metadata.K == 5 || metadata.K == 7))) ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias)
        return false;
    constexpr Exl3CudaLinearAdmission continuation[5] = {
        Exl3CudaLinearAdmission::target_continuation_q,
        Exl3CudaLinearAdmission::target_continuation_qkv,
        Exl3CudaLinearAdmission::target_continuation_z,
        Exl3CudaLinearAdmission::target_continuation_o,
        Exl3CudaLinearAdmission::target_continuation_gate_up};
    return coherent_wide_k6_operation_ >= 0 &&
        coherent_wide_k6_operation_ < 5 &&
        (admission == Exl3CudaLinearAdmission::ordinary ||
         admission == continuation[coherent_wide_k6_operation_]);
}

// NINFER_EXL3_COHERENT_DOWN_SPLIT / NINFER_EXL3_COHERENT_O_SPLIT (10 default,
// quality-gated within +/-0.00022 nats/token; 5 or 8): K partitions of the coherent down / O producers. A numerics-policy
// choice (FP32 partial boundaries move); falls back to 5 when the owned
// accumulation planes are too small.
int Exl3CudaLinearWorkspace::coherent_split_override(const char* name, int rows) const {
    const char* value = std::getenv(name);
    const int base = static_cast<int>(Exl3LinearWorkspaceRequirements::accumulation_splits);
    const int split = value ? std::atoi(value) : 10;
    if (split != 5 && split != 8 && split != 10)
        throw std::invalid_argument("coherent split override must be 5, 8 or 10");
    const auto required = static_cast<std::size_t>(rows) *
        static_cast<std::size_t>(out_features_) * split * sizeof(float);
    return accumulation_capacity_bytes_ >= required ? split : base;
}

int Exl3CudaLinearWorkspace::coherent_wide_k6_split_count(int rows) const noexcept {
    constexpr int split10 = 10;
    const auto required = static_cast<std::size_t>(rows) *
        static_cast<std::size_t>(out_features_) * split10 * sizeof(float);
    return coherent_wide_k6_split10_enabled_ &&
        accumulation_capacity_bytes_ >= required ? split10 :
        static_cast<int>(Exl3LinearWorkspaceRequirements::accumulation_splits);
}

bool Exl3CudaLinearWorkspace::coherent_down_k6_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // One N8 CTA decodes each packed B tile once for all independent active
    // rows of an M16 MMA tile. Scalar, verifier, and correction share this
    // body and five disjoint K partitions.
    return coherent_down_k6_enabled_ && allow_generic_variants_ &&
        coherent_down_k6_resident_capacity_ > 0 && rows >= 1 && rows <= 8 &&
        rows <= max_rows_ &&
        (admission == Exl3CudaLinearAdmission::ordinary ||
         admission == Exl3CudaLinearAdmission::target_continuation_down) &&
        in_features_ == 17408 && out_features_ == 5120 &&
        metadata.in_features == in_features_ &&
        metadata.out_features == out_features_ &&
        (metadata.K == 6 || (coherent_any_k_enabled() && metadata.K == 5)) &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias;
}

std::size_t Exl3CudaLinearWorkspace::coherent_down_k7_shared_bytes_for_test() noexcept {
    return kCoherentDownK7SharedBytes;
}

bool Exl3CudaLinearWorkspace::coherent_down_k7_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return coherent_down_k7_enabled_ && allow_generic_variants_ &&
        coherent_down_k7_resident_capacity_ > 0 && rows >= 1 && rows <= 8 &&
        rows <= max_rows_ &&
        (admission == Exl3CudaLinearAdmission::ordinary ||
         admission == Exl3CudaLinearAdmission::target_continuation_down) &&
        in_features_ == 17408 && out_features_ == 5120 &&
        metadata.in_features == in_features_ &&
        metadata.out_features == out_features_ && metadata.K == 7 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias;
}

bool Exl3CudaLinearWorkspace::coherent_o_k7_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return coherent_o_k7_enabled_ && allow_generic_variants_ &&
        coherent_o_k7_resident_capacity_ > 0 && rows >= 1 && rows <= 8 &&
        rows <= max_rows_ &&
        (admission == Exl3CudaLinearAdmission::ordinary ||
         admission == Exl3CudaLinearAdmission::target_continuation_o) &&
        in_features_ == 6144 && out_features_ == 5120 &&
        metadata.in_features == in_features_ &&
        metadata.out_features == out_features_ && metadata.K == 7 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias;
}

bool Exl3CudaLinearWorkspace::fast_fp16_m2_8_all_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if(!fast_fp16_m2_8_all_enabled_ || !allow_generic_variants_ ||
       rows<2 || rows>8 || rows>max_rows_ || metadata.mcg ||
       !metadata.mul1 || metadata.has_bias ||
       metadata.in_features!=in_features_ ||
       metadata.out_features!=out_features_ ||
       (in_features_!=5120 && in_features_!=17408) ||
       out_features_==248320 || (metadata.K!=6 && metadata.K!=7))
        return false;
    switch(admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_down:
    case Exl3CudaLinearAdmission::target_continuation_q:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_o:
        break;
    default:return false;
    }
    const bool down=in_features_==17408 && out_features_==5120;
    const int tiles=metadata.K==7?32:(down?16:32);
    const int blocks=out_features_/(16*tiles);
    const int capacity=fast_same_weights_fp16_m1_capacity_[
        metadata.K==7?2:(down?1:0)];
    return out_features_% (16*tiles)==0 && blocks>0 && blocks<=capacity;
}

bool Exl3CudaLinearWorkspace::fast_native_persistent_m1_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!fast_native_persistent_m1_enabled_ || !allow_generic_variants_ ||
        rows != 1 || rows > max_rows_ || metadata.mcg || !metadata.mul1 ||
        metadata.has_bias || metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        ((in_features_ != 5120 || out_features_ != 17408) &&
         (in_features_ != 17408 || out_features_ != 5120)) ||
        (metadata.K != 6 && metadata.K != 7) ||
        (fast_native_persistent_m1_only_ != 0 &&
         fast_native_persistent_m1_only_ != metadata.K) ||
        fast_native_persistent_m1_grid_[metadata.K - 6] <= 0)
        return false;
    if (metadata.K == 6 && in_features_ == 5120 &&
        out_features_ % (32 * 16) != 0)
        return false;
    if (metadata.K == 6 && in_features_ == 17408 &&
        out_features_ % (16 * 16) != 0)
        return false;
    if (metadata.K == 7 && out_features_ % (8 * 16) != 0)
        return false;
    switch (admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_prefill_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_down:
    case Exl3CudaLinearAdmission::target_continuation_o:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_q:
        return true;
    default:
        return false;
    }
}

bool Exl3CudaLinearWorkspace::fast_native_mia_m1_fp16_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    int candidate_grid = 0;
    if (metadata.K == 6 || metadata.K == 7)
        candidate_grid = fast_native_mia_m1_fp16_fused_input_enabled_
            ? fast_native_mia_m1_fp16_fused_input_grid_[metadata.K - 6]
            : fast_native_persistent_m1_grid_[metadata.K - 6];
    if (!fast_native_mia_m1_fp16_enabled_ || !allow_generic_variants_ ||
        rows != 1 || rows > max_rows_ || metadata.mcg || !metadata.mul1 ||
        metadata.has_bias || metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        ((in_features_ != 5120 || out_features_ != 17408) &&
         (in_features_ != 17408 || out_features_ != 5120)) ||
        (metadata.K != 6 && metadata.K != 7) || candidate_grid <= 0)
        return false;
    if (metadata.K == 6 && in_features_ == 5120 &&
        out_features_ % (32 * 16) != 0)
        return false;
    if (metadata.K == 6 && in_features_ == 17408 &&
        out_features_ % (16 * 16) != 0)
        return false;
    if (metadata.K == 7 && out_features_ % (8 * 16) != 0)
        return false;
    switch (admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_prefill_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_down:
    case Exl3CudaLinearAdmission::target_continuation_o:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_q:
        return true;
    default:
        return false;
    }
}

bool Exl3CudaLinearWorkspace::fast_native_persistent_prefill_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!fast_native_persistent_prefill_enabled_ || !allow_generic_variants_ ||
        reconstructed_exact_.data != nullptr || rows <= 16 ||
        rows > max_rows_ || admission != Exl3CudaLinearAdmission::target_wide_prefill ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ || metadata.out_features != out_features_ ||
        metadata.K < 5 || metadata.K > 7)
        return false;
    const bool supported_shape =
        (in_features_ == 5120 &&
            (out_features_ == 6144 || out_features_ == 10240 ||
             out_features_ == 17408)) ||
        (in_features_ == 6144 && out_features_ == 5120) ||
        (in_features_ == 17408 && out_features_ == 5120);
    if (!supported_shape) return false;
    const int chunks = (rows + 15) / 16;
    return fast_native_persistent_prefill_capacity_[metadata.K - 5] >= chunks;
}

bool Exl3CudaLinearWorkspace::fast_native_mia_target_prefill_fp16_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // K6-only first qualification: same packed trellis, same canonical
    // transformed input, and the native fallback for every other shape or
    // precision. The cooperative inner owns final FP16 output/Hadamard.
    if (!fast_native_mia_target_prefill_fp16_enabled_ ||
        !allow_generic_variants_ || reconstructed_exact_.data != nullptr ||
        rows <= 16 || rows > max_rows_ ||
        admission != Exl3CudaLinearAdmission::target_wide_prefill ||
        metadata.K != 6 || metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        fast_native_mia_target_prefill_fp16_capacity_ <= 0)
        return false;
    const bool supported_shape =
        (in_features_ == 5120 &&
            (out_features_ == 6144 || out_features_ == 10240 ||
             out_features_ == 12288 || out_features_ == 17408)) ||
        (in_features_ == 6144 && out_features_ == 5120) ||
        (in_features_ == 17408 && out_features_ == 5120);
    if (!supported_shape) return false;
    const int chunks = (rows + 15) / 16;
    // The large-M adapter advances through all row chunks inside one
    // cooperative grid, so only the resident slice grid needs capacity.
    const bool down_family = in_features_ != 5120;
    const int tile_k = down_family ? 32 : 16;
    const int tile_n = down_family ? 256 : 512;
    const int tiles = (in_features_ / tile_k) * (out_features_ / tile_n);
    return tiles > 0 &&
        std::min(tiles, fast_native_mia_target_prefill_fp16_capacity_) > 0 &&
        chunks > 0;
}

bool Exl3CudaLinearWorkspace::target_fast_same_weights_fp16_m1_n64_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // Separate from the existing N32 FP16-M1 candidate. N64 is only admitted
    // when one CTA can cover a complete 1024-output tile group; this keeps the
    // same packed weights, FP16 operand decode, FP32 MMA accumulation and
    // split/reduction semantics while changing only CTA ownership.
    if (!fast_same_weights_fp16_m1_n64_enabled_ || !allow_generic_variants_ ||
        rows != 1 || rows > max_rows_ || metadata.mcg || !metadata.mul1 ||
        metadata.has_bias || metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        out_features_ % (64 * 16) != 0 || out_features_ == 248320 ||
        metadata.K < 5 || metadata.K > 7)
        return false;
    switch (admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_q:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_o:
        break;
    default:
        return false;
    }
    const int output_blocks = out_features_ / (64 * 16);
    const int capacity =
        fast_same_weights_fp16_m1_n64_capacity_[metadata.K - 5];
    return output_blocks > 0 && output_blocks <= capacity;
}

bool Exl3CudaLinearWorkspace::target_fast_same_weights_fp16_m1_n64_k5_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // Separately labeled K5-only exact candidate. It changes only CTA ownership
    // relative to the native packed-MMA reference; no activation re-quantization
    // or external quantized-weight implementation is involved.
    if (!fast_same_weights_fp16_m1_n64_k5_enabled_ || !allow_generic_variants_ ||
        rows != 1 || rows > max_rows_ || metadata.K != 5 || metadata.mcg ||
        !metadata.mul1 || metadata.has_bias || metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ || out_features_ % (64 * 16) != 0 ||
        out_features_ == 248320)
        return false;
    switch (admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_q:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_o:
        break;
    default:
        return false;
    }
    const int output_blocks = out_features_ / (64 * 16);
    return output_blocks > 0 &&
        output_blocks <= fast_same_weights_fp16_m1_n64_capacity_[0];
}

bool Exl3CudaLinearWorkspace::target_m1_k7_int8_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if(!fast_same_weights_int8_gemv_k7_enabled_ || rows!=1 || rows>max_rows_ ||
       metadata.K!=7 || metadata.mcg || !metadata.mul1 || metadata.has_bias ||
       metadata.in_features!=in_features_ ||
       metadata.out_features!=out_features_ || out_features_%256!=0 ||
       in_features_==17408 || out_features_==248320)
        return false;
    // The pinned Mia Blackwell extension reports an INT8 ceiling of K6;
    // K7 remains on its native packed route in this differential.
    if (fast_same_weights_int8_mia_policy_enabled_)
        return false;
    switch(admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_q:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_kv:
    case Exl3CudaLinearAdmission::target_continuation_o:
        return true;
    default:
        return false;
    }
}

bool Exl3CudaLinearWorkspace::target_m1_int8_down_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if(rows!=1 || rows>max_rows_ || admission!=Exl3CudaLinearAdmission::ordinary ||
       metadata.mcg || !metadata.mul1 || metadata.has_bias ||
       metadata.in_features!=in_features_ ||
       metadata.out_features!=out_features_ || metadata.in_features!=17408 ||
       metadata.out_features!=5120 || (metadata.K!=6 && metadata.K!=7))
        return false;
    // 17408->5120 is below Mia's width threshold and therefore remains
    // packed MGEMM; do not substitute the INT8 down lane in this policy.
    if (fast_same_weights_int8_mia_policy_enabled_)
        return false;
    return metadata.K==6 ? fast_same_weights_int8_gemv_down_k6_enabled_ :
        fast_same_weights_int8_gemv_down_k7_enabled_;
}

bool Exl3CudaLinearWorkspace::target_gateup_k6_n16_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // The retained N32 path exposes only 170 CTAs for this shape at five
    // splits.  N16 keeps each output column's packed-MMA and ascending split
    // reduction arithmetic unchanged while doubling independent output CTAs.
    return target_gateup_k6_n16_enabled_ && metadata.K == 6 &&
        target_gateup_small_m_candidate(metadata, rows, admission);
}

bool Exl3CudaLinearWorkspace::target_down_k6_async_a_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!target_down_k6_async_a_enabled_ || metadata.K != 6 ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        in_features_ != 17408 || out_features_ != 5120 ||
        rows < 1 || rows > max_rows_ ||
        (rows == 1 && admission != Exl3CudaLinearAdmission::ordinary) ||
        (rows > 1 && !target_down_small_m_candidate(metadata, rows, admission)) ||
        !extended_stream_reduction_candidate(metadata, rows, admission)) {
        return false;
    }
    const char* requested = std::getenv("NINFER_EXL3_LARGE_DOWN_TOPOLOGY");
    const bool topology16 = requested == nullptr || *requested == '\0' ||
        std::strcmp(requested, "16") == 0 ||
        std::strcmp(requested, "16-k6") == 0;
    return topology16 && out_features_ / (16 * 16) <=
        target_down_k6_async_a_capacity_;
}

bool Exl3CudaLinearWorkspace::target_k6_gateup_warpgroup_async_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_k6_gateup_warpgroup_async_ &&
        admission == Exl3CudaLinearAdmission::target_wide_prefill &&
        rows > 16 && rows <= max_rows_ && metadata.K == 6 &&
        metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
        metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
        in_features_ == 5120 && out_features_ == 17408 &&
        target_direct_partials_ && target_direct_async_a_ && target_direct_tiles64_ &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_k6_gateup_n32_pair_cta_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_k6_gateup_n32_pair_cta_ &&
        admission == Exl3CudaLinearAdmission::target_wide_prefill &&
        rows > 16 && rows <= max_rows_ && metadata.K == 6 &&
        metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
        metadata.in_features == in_features_ &&
        metadata.out_features == out_features_ &&
        in_features_ == 5120 && out_features_ == 17408 &&
        out_features_ % (64*16) == 0 && target_direct_partials_ &&
        target_direct_async_a_ &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_k7_tiles64_exact_splits_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_k7_tiles64_exact_splits_ &&
        rows > 16 && rows <= max_rows_ && metadata.K == 7 &&
        metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
        metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
        !(in_features_ == 17408 && out_features_ == 5120) &&
        out_features_ % (64 * 16) == 0 &&
        target_direct_partials_ && target_direct_async_a_ && target_direct_async_all_ &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_k8_kv_prefill_async_a_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_k8_kv_prefill_async_a_ &&
        admission == Exl3CudaLinearAdmission::target_wide_prefill &&
        rows > 16 && rows <= max_rows_ && metadata.K == 8 &&
        metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
        metadata.in_features == in_features_ &&
        metadata.out_features == out_features_ &&
        in_features_ == 5120 && out_features_ == 1024 &&
        target_direct_partials_ && target_direct_async_a_ &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_rowpair_k7_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_rowpair_k7_ && target_direct_partials_ && target_direct_async_a_ &&
        rows >= 32 && metadata.K == 7 &&
        !(in_features_ == 17408 && out_features_ == 5120) &&
        target_staged_prefill_candidate(metadata,rows,admission);
}

bool Exl3CudaLinearWorkspace::target_o_k7_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!target_o_k7_small_m_enabled_ ||
        admission != Exl3CudaLinearAdmission::target_continuation_o ||
        rows < 2 || rows > native_continuation_rows_ || rows > max_rows_ || metadata.K != 7 ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ || in_features_ != 6144 ||
        out_features_ != 5120) {
        return false;
    }
    const int output_blocks = out_features_ / (32 * 16);
    return output_blocks <= generic_capacity_[2];
}

bool Exl3CudaLinearWorkspace::target_kv_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!target_kv_small_m_enabled_ ||
        admission != Exl3CudaLinearAdmission::target_continuation_kv ||
        rows < 2 || rows > native_continuation_rows_ || rows > max_rows_ || (metadata.K != 7 && metadata.K != 8) ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ || in_features_ != 5120 ||
        out_features_ != 1024) {
        return false;
    }
    const int output_blocks = out_features_ / (32 * 16);
    return output_blocks <= generic_capacity_[2];
}

bool Exl3CudaLinearWorkspace::target_o_k6_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!target_o_k6_small_m_enabled_ ||
        admission != Exl3CudaLinearAdmission::target_continuation_o ||
        rows < 2 || rows > native_continuation_rows_ || rows > max_rows_ || metadata.K != 6 ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ || in_features_ != 6144 ||
        out_features_ != 5120) {
        return false;
    }
    const int output_blocks = out_features_ / (32 * 16);
    return output_blocks <= generic_capacity_[1];
}

bool Exl3CudaLinearWorkspace::target_z_k6_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_z_k6_small_m_enabled_ &&
        admission == Exl3CudaLinearAdmission::target_continuation_z &&
        rows >= 2 && rows <= native_continuation_rows_ && rows <= max_rows_ && metadata.K == 6 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
        in_features_ == 5120 && out_features_ == 6144 &&
        out_features_ / (32 * 16) <= generic_capacity_[1];
}

bool Exl3CudaLinearWorkspace::target_qkv_k6_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_qkv_k6_small_m_enabled_ &&
        admission == Exl3CudaLinearAdmission::target_continuation_qkv &&
        rows >= 2 && rows <= native_continuation_rows_ && rows <= max_rows_ && metadata.K == 6 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
        in_features_ == 5120 && out_features_ == 10240 &&
        out_features_ / (32 * 16) <= generic_capacity_[1];
}

bool Exl3CudaLinearWorkspace::target_q_k6_small_m_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_q_k6_small_m_enabled_ &&
        admission == Exl3CudaLinearAdmission::target_continuation_q &&
        rows >= 2 && rows <= native_continuation_rows_ && rows <= max_rows_ && metadata.K == 6 &&
        !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
        in_features_ == 5120 && out_features_ == 12288 &&
        out_features_ / (32 * 16) <= generic_capacity_[1];
}

bool Exl3CudaLinearWorkspace::target_wide_prefill_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!target_wide_prefill_enabled_ || admission != Exl3CudaLinearAdmission::target_wide_prefill ||
        rows < 2 || rows > (target_staged_prefill_enabled_ ? (target_wide1024_enabled_ ? 1024 : target_wide512_enabled_ ? 512 : target_wide256_enabled_ ? 256 : target_wide128_enabled_ ? 128 : target_wide64_enabled_ ? 64 : 32) : 16) || rows > max_rows_ || metadata.K < 5 || metadata.K > 8 ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ || metadata.out_features != out_features_) return false;
    const bool target_shape =
        (in_features_ == 5120 && (out_features_ == 1024 || out_features_ == 6144 ||
         out_features_ == 10240 || out_features_ == 12288 || out_features_ == 17408)) ||
        ((in_features_ == 6144 || in_features_ == 17408) && out_features_ == 5120);
    if (!target_shape) return false;
    if (specialized_shape_ && metadata.K == kK) return true;
    if (in_features_ == 17408 && out_features_ == 5120 && (metadata.K == 6 || metadata.K == 7)) {
        const char* requested = std::getenv("NINFER_EXL3_LARGE_DOWN_TOPOLOGY");
        const std::string topology = requested ? requested : "";
        // Preserve the exact production M1 16-tile split topology. Other
        // experimental topologies retain per-row execution in this first scope.
        if (!(topology.empty() || topology == "16" ||
              (metadata.K == 6 && topology == "16-k6") ||
              (metadata.K == 7 && topology == "16-k7"))) return false;
        return out_features_ / (16 * 16) <= large_down_candidate_capacity_[metadata.K == 6 ? 0 : 1];
    }
    return out_features_ % (32 * 16) == 0 && out_features_ / (32 * 16) <=
        generic_capacity_[metadata.K == 5 ? 0 : (metadata.K == 6 ? 1 : 2)];
}

bool Exl3CudaLinearWorkspace::fast_wide_prefill_gemm_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (!fast_wide_prefill_gemm_enabled_ ||
        admission != Exl3CudaLinearAdmission::target_wide_prefill ||
        rows <= 16 || rows > max_rows_ ||
        !reconstructed_exact_.data || !reconstructed_exact_.ordered_stream ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        (metadata.K != 6 && metadata.K != 7) || metadata.mcg ||
        !metadata.mul1 || metadata.has_bias ||
        !target_wide_prefill_candidate(metadata, rows, admission))
        return false;

    // The reconstruction kernel writes a row-major FP16 matrix. Require the
    // exact full slab; sliced/fragment-order slabs keep their native fallback.
    if (metadata.in_features < 16 || metadata.out_features < 16 ||
        metadata.in_features % 16 != 0 || metadata.out_features % 16 != 0 ||
        (reinterpret_cast<std::uintptr_t>(reconstructed_exact_.data) & 0xfu) != 0)
        return false;
    const std::size_t input = static_cast<std::size_t>(metadata.in_features);
    const std::size_t output = static_cast<std::size_t>(metadata.out_features);
    if (input > std::numeric_limits<std::size_t>::max() / output)
        return false;
    const std::size_t elements = input * output;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t))
        return false;
    return reconstructed_exact_.bytes == elements * sizeof(std::uint16_t);
}

bool Exl3CudaLinearWorkspace::native_mtp_wide_prefill_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (admission != Exl3CudaLinearAdmission::native_mtp_wide_prefill ||
        rows <= 16 || rows > max_rows_ || metadata.K != 4 ||
        metadata.mcg || !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_)
        return false;

    // These are the eight closed native-MTP EXL3 projection geometries.  The
    // explicit list prevents an arbitrary K4 module from inheriting the MTP
    // route, while the multiples document the <4,64,2> kernel contract.
    const bool native_shape =
        (in_features_ == 10240 && out_features_ == 5120) ||
        (in_features_ == 5120 &&
            (out_features_ == 12288 || out_features_ == 1024 ||
             out_features_ == 17408)) ||
        (in_features_ == 6144 && out_features_ == 5120) ||
        (in_features_ == 17408 && out_features_ == 5120);
    const bool direct_geometry =
        in_features_ >= 16 && (in_features_ % 16) == 0 &&
        out_features_ >= 16 * 64 && (out_features_ % (16 * 64)) == 0;
    return native_shape && direct_geometry;
}

bool Exl3CudaLinearWorkspace::native_mtp_one_step_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if (admission != Exl3CudaLinearAdmission::native_mtp_one_step || rows != 1 ||
        rows > max_rows_ || metadata.K != 4 || metadata.mcg || !metadata.mul1 ||
        metadata.has_bias || metadata.in_features != in_features_ ||
        metadata.out_features != out_features_)
        return false;
    const bool native_shape =
        (in_features_ == 10240 && out_features_ == 5120) ||
        (in_features_ == 5120 &&
            (out_features_ == 12288 || out_features_ == 1024 ||
             out_features_ == 17408)) ||
        (in_features_ == 6144 && out_features_ == 5120) ||
        (in_features_ == 17408 && out_features_ == 5120);
    const bool direct_geometry =
        in_features_ >= 16 && (in_features_ % 16) == 0 &&
        out_features_ >= 16 * 64 && (out_features_ % (16 * 64)) == 0;
    return native_shape && direct_geometry;
}

bool Exl3CudaLinearWorkspace::extended_stream_reduction_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,Exl3CudaLinearAdmission admission) const noexcept {
    if(rows<1||rows>8||(metadata.K!=6&&metadata.K!=7))return false;
    const bool kv=kv_k7_stream_reduction_enabled_ && metadata.K==7 &&
        target_kv_small_m_candidate(metadata,rows,admission);
    if(!extended_stream_reduction_enabled_ && !kv)return false;
    if(in_features_==17408&&out_features_==5120){
        const char* topology=std::getenv("NINFER_EXL3_LARGE_DOWN_TOPOLOGY");
        if(topology&&std::strcmp(topology,"8")==0)return false;
        const bool ordinary=!topology||!*topology||std::strcmp(topology,"16")==0||
            (metadata.K==6&&std::strcmp(topology,"16-k6")==0)||
            (metadata.K==7&&std::strcmp(topology,"16-k7")==0);
        return (rows==1&&ordinary)||target_down_small_m_candidate(metadata,rows,admission);
    }
    return metadata.K==7&&in_features_!=17408&&out_features_!=248320&&out_features_%512==0&&
        out_features_/512<=generic_capacity_[2]&&
        (rows==1||target_gateup_small_m_candidate(metadata,rows,admission)||target_o_k7_small_m_candidate(metadata,rows,admission)||kv);
}

bool Exl3CudaLinearWorkspace::k6_stream_reduction_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,Exl3CudaLinearAdmission admission) const noexcept {
    // Mirror only the parent's generic N32 branch. Large-down N16, H6,
    // wide-prefill, K5/K7 and draft-owned workspaces retain their dispatch.
    return k6_stream_reduction_enabled_ && metadata.K==6 && rows>=1 && rows<=8 &&
        in_features_!=17408 && out_features_!=248320 && out_features_%512==0 &&
        out_features_/512<=generic_capacity_[1] &&
        (rows==1 || target_gateup_small_m_candidate(metadata,rows,admission) ||
         target_o_k6_small_m_candidate(metadata,rows,admission) ||
         target_kv_small_m_candidate(metadata,rows,admission) ||
         target_z_k6_small_m_candidate(metadata,rows,admission) ||
         target_qkv_k6_small_m_candidate(metadata,rows,admission) ||
         target_q_k6_small_m_candidate(metadata,rows,admission));
}

bool Exl3CudaLinearWorkspace::target_k6_small_m_async_a_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    if(!target_k6_small_m_async_a_enabled_ || rows<2 ||
       !k6_stream_reduction_candidate(metadata,rows,admission) ||
       target_gateup_small_m_candidate(metadata,rows,admission)) return false;
    return out_features_/(32*16)<=target_k6_small_m_async_a_capacity_;
}

bool Exl3CudaLinearWorkspace::native_k6_critical_path_candidate(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // R49 target-only M=1 differential. The packed trellis, MUL1 decode,
    // FP16 MMA operands, FP32 accumulators, split count and output reduction
    // remain native; only K6's exact two-word lane-window decoder changes.
    if (!native_k6_critical_path_enabled_ || !allow_generic_variants_ ||
        rows != 1 || rows > max_rows_ || metadata.K != 6 || metadata.mcg ||
        !metadata.mul1 || metadata.has_bias ||
        metadata.in_features != in_features_ ||
        metadata.out_features != out_features_ ||
        (in_features_ == 17408 && out_features_ == 5120)) {
        return false;
    }
    const bool n16_shape = out_features_ % (16 * 16) == 0;
    const bool n32_shape = out_features_ % (32 * 16) == 0;
    const bool n16_capacity = native_k6_register_pipeline_n16_enabled_ &&
        n16_shape && native_k6_register_pipeline_n16_capacity_ > 0 &&
        out_features_ / (16 * 16) <= native_k6_register_pipeline_n16_capacity_;
    const bool n32_capacity = n32_shape &&
        native_k6_critical_path_capacity_ > 0 &&
        out_features_ / (32 * 16) <= native_k6_critical_path_capacity_;
    const bool global_slice_capacity = n32_shape && max_rows_ >= 2 &&
        native_k6_global_slices_enabled_ &&
        native_k6_global_slices_capacity_ > 0 &&
        out_features_ / (32 * 16) <= native_k6_global_slices_capacity_;
    if (!n16_capacity && !n32_capacity && !global_slice_capacity)
        return false;
    switch (admission) {
    case Exl3CudaLinearAdmission::ordinary:
    case Exl3CudaLinearAdmission::target_continuation_gate_up:
    case Exl3CudaLinearAdmission::target_continuation_o:
    case Exl3CudaLinearAdmission::target_continuation_z:
    case Exl3CudaLinearAdmission::target_continuation_q:
    case Exl3CudaLinearAdmission::target_continuation_qkv:
    case Exl3CudaLinearAdmission::target_continuation_kv:
        return true;
    default:
        return false;
    }
}

bool Exl3CudaLinearWorkspace::target_k7_small_m_async_a_candidate(
    const Exl3CudaLinearMetadata& metadata,int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    return target_k7_small_m_async_a_enabled_ && rows>=2 &&
        extended_stream_reduction_candidate(metadata,rows,admission) &&
        metadata.K==7 && in_features_!=17408 &&
        out_features_/(32*16)<=target_k7_small_m_async_a_capacity_;
}

const char* Exl3CudaLinearWorkspace::dispatch_name(
    const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission) const noexcept {
    // Diagnostic mirror of the forward_from_transformed() branch order.
    if (coherent_down_k6_candidate(metadata, rows, admission))
        return "coherent_down_k6_shared_rows_n8_split5";
    if (coherent_down_k7_candidate(metadata, rows, admission))
        return "coherent_down_k7_shared_rows_n8_split5";
    if (coherent_o_k7_candidate(metadata, rows, admission))
        return "coherent_o_k7_shared_rows_n8_split5";
    if (coherent_wide_k6_candidate(metadata, rows, admission)) {
        constexpr const char* names[5] = {
            "coherent_wide_k6_q_n8_m16_split5",
            "coherent_wide_k6_qkv_n8_m16_split5",
            "coherent_wide_k6_z_n8_m16_split5",
            "coherent_wide_k6_o_n8_m16_split5",
            "coherent_wide_k6_gate_up_n8_m16_split5"};
        constexpr const char* split10_names[5] = {
            "coherent_wide_k6_q_n8_m16_split10",
            "coherent_wide_k6_qkv_n8_m16_split10",
            "coherent_wide_k6_z_n8_m16_split10",
            "coherent_wide_k6_o_n8_m16_split10",
            "coherent_wide_k6_gate_up_n8_m16_split10"};
        if (coherent_wide_k6_split_count(rows) == 10)
            return split10_names[coherent_wide_k6_operation_];
        return names[coherent_wide_k6_operation_];
    }
    if (native_mtp_one_step_candidate(metadata, rows, admission))
        return "native_mtp_one_step_mma64";
    if (native_mtp_wide_prefill_candidate(metadata, rows, admission))
        return "native_mtp_wide_prefill_mma64";
    if (native_k6_critical_path_candidate(metadata, rows, admission)) {
        const int output_blocks = metadata.out_features / (16 * 32);
        if (metadata.K == 6 && rows == 1 &&
            native_k6_shape4_register_pipeline_enabled_ &&
            native_k6_register_pipeline_capacity_ >= output_blocks) {
            return "native_k6_shape4_register_pipeline";
        }
        return "native_k6_critical_path_fast_decode";
    }
    if(target_gateup_k6_n16_candidate(metadata,rows,admission))
        return "target_gateup_k6_n16_async_a_stream_reduction";
    if(target_down_k6_async_a_candidate(metadata,rows,admission))
        return "target_down_k6_async_a_stream_reduction";
    if(target_k7_small_m_async_a_candidate(metadata,rows,admission))
        return "target_k7_small_m_async_a_stream_reduction";
    if(extended_stream_reduction_candidate(metadata,rows,admission))return "target_extended_stream_reduction";
    if(target_k6_small_m_async_a_candidate(metadata,rows,admission))
        return "target_k6_small_m_async_a_stream_reduction";
    if(k6_stream_reduction_candidate(metadata,rows,admission))return "target_k6_small_m_stream_reduction";
    if(target_k5_small_m_batch_candidate(metadata,rows,admission))
        return "target_k5_small_m_batch";
    if(target_k6_m1_simt_candidate(metadata,rows,admission))
        return "target_k6_m1_simt";
    if (target_fast_same_weights_fp16_m1_n64_k5_candidate(metadata, rows, admission))
        return "fast_same_weights_fp16_m1_n64_k5_exact";
    if (target_fast_same_weights_fp16_m1_n64_candidate(metadata, rows, admission))
        return metadata.K == 5 ? "fast_same_weights_fp16_m1_n64_k5" :
            metadata.K == 6 ? "fast_same_weights_fp16_m1_n64_k6" :
                               "fast_same_weights_fp16_m1_n64_k7";
    if(target_m1_int8_down_candidate(metadata,rows,admission))
        return metadata.K==6 ? "fast_same_weights_int8_gemv_down_k6" :
            "fast_same_weights_int8_gemv_down_k7";
    if(target_m1_k7_int8_candidate(metadata,rows,admission))
        return "fast_same_weights_int8_gemv_k7";
    if (fast_fp16_m2_8_fused_down_candidate(metadata,rows,admission))
        return "fast_fp16_m2_8_fused_down_k6";
    if (fast_fp16_m2_8_all_candidate(metadata,rows,admission))
        return "fast_fp16_m2_8_k6k7";
    if (fast_fp16_m2_8_down_candidate(metadata,rows,admission))
        return "fast_fp16_m2_8_down_k6";
    if (target_fast_same_weights_fp16_m1_candidate(metadata, rows, admission)) {
        const bool k6_down = metadata.K == 6 && metadata.in_features == 17408 &&
            metadata.out_features == 5120;
        const int output_tiles_per_block = k6_down ? 16 : 32;
        const int output_blocks = metadata.out_features /
            (16 * output_tiles_per_block);
        if (metadata.K == 6 && !k6_down &&
            fast_same_weights_fp16_m1_k6_register_pipeline_enabled_ &&
            fast_same_weights_fp16_m1_k6_register_pipeline_capacity_ > 0 &&
            output_blocks <= fast_same_weights_fp16_m1_k6_register_pipeline_capacity_)
            return "fast_same_weights_fp16_m1_k6_register_pipeline";
        return metadata.K == 6
            ? ((metadata.in_features == 17408 && metadata.out_features == 5120)
                ? "fast_same_weights_fp16_m1_k6_down"
                : "fast_same_weights_fp16_m1_k6_wide")
            : "fast_same_weights_fp16_m1_k7_wide";
    }
    if(target_m1_k6_n16_candidate(metadata,rows,admission))
        return "target_m1_k6_n16_async_a_stream_reduction";
    if (fast_native_mia_target_prefill_fp16_candidate(metadata, rows, admission))
        return "fast_native_mia_target_prefill_fp16";
    if (fast_native_persistent_prefill_candidate(metadata, rows, admission))
        return "fast_native_persistent_prefill";
    if (fast_wide_prefill_gemm_candidate(metadata, rows, admission))
        return "fast_mia_parity_wide_prefill_fp16_gemm";
    if (reconstructed_exact_candidate(metadata, rows, admission))
        return metadata.K==6 && metadata.in_features==5120 ?
            "target_wide_prefill_reconstructed_exact_k6_gate_up" :
            metadata.K==6 ? "target_wide_prefill_reconstructed_exact_k6_down" :
            "target_wide_prefill_reconstructed_exact_k7";
    if (draft_prefill_fc_candidate(metadata, rows, admission)) return "draft_prefill_fc_mma32";
    if(target_k6_fast_decode_candidate(metadata,rows,admission))
        return "target_wide_prefill_k6_mod48_fast_decode";
    if(target_k6_rowpair_n64_candidate(metadata,rows,admission))
        return "target_wide_prefill_k6_rowpair_n64";
    if(target_k6_down_rowpair_candidate(metadata,rows,admission))
        return "target_wide_prefill_k6_down_rowpair";
    if(target_shape4_n64_candidate(metadata,rows,admission))
        return "target_wide_prefill_shape4_n64";
    if (target_k6_gateup_warpgroup_async_candidate(metadata, rows, admission))
        return "target_k6_gateup_warpgroup_async";
    if (target_k6_gateup_n32_pair_cta_candidate(metadata, rows, admission))
        return "target_k6_gateup_n32_pair_cta";
    if (target_k7_tiles64_exact_splits_candidate(metadata, rows, admission))
        return "target_wide_prefill_k7_tiles64_exact_splits";
    if (target_k8_kv_prefill_async_a_candidate(metadata, rows, admission))
        return "target_wide_prefill_k8_kv_async_a";
    if (target_rowpair_k6_candidate(metadata, rows, admission)) return "target_wide_prefill_rowpair_k6";
    if (target_rowpair_k7_candidate(metadata, rows, admission)) return "target_wide_prefill_rowpair_k7";
    if (target_staged_prefill_candidate(metadata, rows, admission)) return "target_wide_prefill_staged";
    if (target_initial16_candidate(metadata, rows, admission)) return "target_initial16_ordered";
    if (target_wide_prefill_candidate(metadata, rows, admission)) {
        if (specialized_shape_ && metadata.K == kK) return "target_wide_prefill_shape4";
        if (in_features_ == 17408 && (metadata.K == 6 || metadata.K == 7)) return "target_wide_prefill_mma16";
        return "target_wide_prefill_mma32";
    }
    if (target_gateup_m16_candidate(metadata, rows, admission))
        return "target_gateup_m16_ordered";
    if (specialized_shape_ && metadata.K == kK && rows <= 16) return "m1_shape4_split";
    if (target_down_small_m_candidate(metadata, rows, admission))
        return metadata.K == 6 ? "target_down_small_m_mma16_split"
             : target_down_k7_async_a_enabled_
                 ? "target_down_k7_small_m_async_a"
                 : "target_down_k7_small_m_mma16_split";
    if (target_kv_small_m_candidate(metadata, rows, admission))
        return "target_kv_small_m_mma_split";
    if (target_o_k6_small_m_candidate(metadata, rows, admission))
        return "target_o_k6_small_m_mma_split";
    if (target_o_k7_small_m_candidate(metadata, rows, admission))
        return "target_o_k7_small_m_mma_split";
    if (target_qkv_k6_small_m_candidate(metadata, rows, admission))
        return "target_qkv_k6_small_m_mma_split";
    if (target_q_k6_small_m_candidate(metadata, rows, admission))
        return "target_q_k6_small_m_mma_split";
    if (target_z_k6_small_m_candidate(metadata, rows, admission))
        return "target_z_k6_small_m_mma_split";
    if (rows == 1 && in_features_ == 17408 && out_features_ == 5120 &&
        (metadata.K == 6 || metadata.K == 7)) {
        const char* requested = std::getenv("NINFER_EXL3_LARGE_DOWN_TOPOLOGY");
        const std::string topology =
            requested == nullptr ? std::string() : std::string(requested);
        if (topology.empty() || topology == "16" || topology == "8" ||
            (topology == "16-k6" && metadata.K == 6) ||
            (topology == "16-k7" && metadata.K == 7)) {
            return "large_down_mma";
        }
    }
    if ((rows == 1 || (h6_small_m_enabled_ && rows >= 2 && rows <= native_continuation_rows_)) &&
        in_features_ == 5120 && out_features_ == 248320 && metadata.K == 6) {
        return rows == 1 ? "h6_single_split" : "h6_small_m_single_split";
    }
    if(draft_shared_gateup_m16_candidate(metadata,rows,admission))return "draft_shared_gateup_m16_ordered";
    if(draft_shared_m16_candidate(metadata,rows,admission))return admission==Exl3CudaLinearAdmission::draft_shared_down_m16 ? "draft_shared_down_m16_mma_split" : admission==Exl3CudaLinearAdmission::draft_shared_o_m16 ? "draft_shared_o_m16_mma_split" : admission==Exl3CudaLinearAdmission::draft_shared_kv_m16 ? "draft_shared_kv_m16_mma_split" : "draft_shared_q_m16_mma_split";
    const bool draft_small_m = draft_small_m_candidate(metadata, rows);
    const bool target_gateup_small_m =
        target_gateup_small_m_candidate(metadata, rows, admission);
    const bool target_o_k7_small_m =
        (target_o_k7_small_m_candidate(metadata, rows, admission) ||
         target_o_k6_small_m_candidate(metadata, rows, admission));
    if ((rows == 1 || draft_small_m || target_gateup_small_m ||
         target_o_k7_small_m) &&
        out_features_ % (32 * 16) == 0 &&
        metadata.K >= 5 && metadata.K <= 8) {
        const int blocks = (out_features_ / 16) / 32;
        const int index = metadata.K == 5 ? 0 : (metadata.K == 6 ? 1 : 2);
        if (blocks <= generic_capacity_[index]) {
            if (rows == 1 && metadata.K == 6 &&
                target_m1_k6_n32_async_a_enabled_ &&
                blocks <= target_m1_k6_n32_async_a_capacity_)
                return "target_m1_k6_n32_async_a";
            if (rows == 1 && metadata.K == 7 &&
                target_m1_k7_n32_async_a_enabled_ &&
                blocks <= target_m1_k7_n32_async_a_capacity_)
                return "target_m1_k7_n32_async_a";
            if (draft_small_m) return "draft_small_m_mma_split";
            if (target_gateup_small_m)
                return metadata.K == 6 && target_gateup_k6_async_a_enabled_
                           ? "target_gateup_k6_small_m_async_a"
                           : metadata.K == 6 ? "target_gateup_small_m_mma_split"
                                       : "target_gateup_k7_small_m_mma_split";
            if (target_o_k7_small_m)
                return "target_o_k7_small_m_mma_split";
            return "generic_mma_split";
        }
    }
    if (rows <= 16) return "generic_tile";
    return "unsupported";
}

// Coherent packed producer (K6 down, K7 down/O, wide K6) with the selected
// cp.async depth and CTA width. Every warp keeps 16 output columns and the
// full split K range, so narrower CTAs only enlarge the grid.
template <int Bits, int Stages, int Warps, int Per = 1>
static void launch_coherent_packed_variant(
    cudaStream_t stream, const std::uint16_t* transformed,
    const std::uint16_t* trellis, const std::int32_t* mul1, float* accum,
    int rows, int input_features, int output_features, int split_count) {
    constexpr int tiles = Warps;
    constexpr int stage_count = (Stages ? Stages : 2) * Per;
    const std::size_t shared =
        static_cast<std::size_t>(stage_count) * 256u * sizeof(half) +
        static_cast<std::size_t>(stage_count) * tiles * 16u * Bits * sizeof(std::uint16_t) +
        16u * tiles * 16u * sizeof(float);
    const int grid = output_features / (16 * tiles) * split_count;
    if (shared > 48u * 1024u) {
        static const bool configured = [shared] {
            cuda_check(cudaFuncSetAttribute(
                exl3_gemm_m1_generic_mma_kernel<Bits, false, tiles, true, true, Bits == 7,
                    false, Bits == 6, false, false, false, Stages, Warps, Per>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)),
                "set deep coherent shared memory");
            cuda_check(cudaFuncSetAttribute(
                exl3_gemm_m1_generic_mma_kernel<Bits, false, tiles, true, true, false,
                    false, false, false, false, false, Stages, Warps, Per>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)),
                "set deep coherent shared memory");
            return true;
        }();
        (void)configured;
    }
    // The exact K6 lane-window and K7 three-word decoders (identical state
    // words and MUL1 arithmetic) are the measured default;
    // NINFER_EXL3_COHERENT_FAST_DECODE=0 restores the generic decoder.
    static const bool fast_decode = [] {
        const char* value = std::getenv("NINFER_EXL3_COHERENT_FAST_DECODE");
        if (!value) return true;
        if (std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        throw std::invalid_argument("NINFER_EXL3_COHERENT_FAST_DECODE must be 0 or 1");
    }();
    if (fast_decode) {
        exl3_launch_pdl(exl3_gemm_m1_generic_mma_kernel<Bits, false, tiles, true, true, Bits == 7,
            false, Bits == 6, false, false, false, Stages, Warps, Per>,
            dim3(grid), dim3(Warps * 32), shared, stream,
                transformed, trellis, mul1, accum, rows, input_features,
                output_features, split_count);
        return;
    }
    exl3_launch_pdl(exl3_gemm_m1_generic_mma_kernel<Bits, false, tiles, true, true, false,
        false, false, false, false, false, Stages, Warps, Per>,
        dim3(grid), dim3(Warps * 32), shared, stream,
            transformed, trellis, mul1, accum, rows, input_features,
            output_features, split_count);
}

// NINFER_EXL3_COHERENT_WARPS selects the CTA width: 4 warps (64 columns,
// measured default), 8 (the former 128-column CTA) or 2.
int coherent_packed_warps() {
    static const int warps = [] {
        const char* value = std::getenv("NINFER_EXL3_COHERENT_WARPS");
        if (!value || std::strcmp(value, "4") == 0) return 4;
        if (std::strcmp(value, "8") == 0) return 8;
        if (std::strcmp(value, "4") == 0) return 4;
        if (std::strcmp(value, "2") == 0) return 2;
        throw std::invalid_argument("NINFER_EXL3_COHERENT_WARPS must be 8, 4 or 2");
    }();
    return warps;
}

// NINFER_EXL3_COHERENT_TILES_PER_STAGE: k-tiles per coherent cp.async stage.
// Returns 0 when unset: single-row decode then takes two tiles per stage and
// wider verifier rows four (both measured); the choice never changes numerics.
int coherent_tiles_per_stage_setting() {
    static const int per = [] {
        const char* value = std::getenv("NINFER_EXL3_COHERENT_TILES_PER_STAGE");
        if (!value) return 0;
        if (std::strcmp(value, "4") == 0) return 4;
        if (std::strcmp(value, "1") == 0) return 1;
        if (std::strcmp(value, "2") == 0) return 2;
        throw std::invalid_argument(
            "NINFER_EXL3_COHERENT_TILES_PER_STAGE must be 1, 2 or 4");
    }();
    return per;
}

// Small-grid M1 cooperative GEMVs use narrow CTAs by default (measured);
// NINFER_EXL3_GENERIC_NARROW=0 restores the 512-column CTAs.
bool generic_narrow_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_EXL3_GENERIC_NARROW");
        if (!value) return true;
        if (std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        throw std::invalid_argument("NINFER_EXL3_GENERIC_NARROW must be 0 or 1");
    }();
    return enabled;
}

template <int Bits>
static void launch_coherent_packed_partials(
    int grid, std::size_t two_stage_bytes, cudaStream_t stream,
    const std::uint16_t* transformed, const std::uint16_t* trellis,
    const std::int32_t* mul1, float* accum, int rows, int input_features,
    int output_features, int split_count) {
    const int stages = coherent_deep_pipeline_stages();
    const int warps = coherent_packed_warps();
    const int per_setting = coherent_tiles_per_stage_setting();
    if (const int per = per_setting ? per_setting : (rows == 1 ? 2 : 4); per != 1) {
#define NINFER_COHERENT_MULTI(S, W, P)                                                 if (stages == S && warps == W && per == P) {                                       launch_coherent_packed_variant<Bits, S, W, P>(stream, transformed,                 trellis, mul1, accum, rows, input_features, output_features,                   split_count);                                                              return;                                                                    }
        NINFER_COHERENT_MULTI(4, 4, 2) NINFER_COHERENT_MULTI(4, 4, 4)
        NINFER_COHERENT_MULTI(8, 4, 2) NINFER_COHERENT_MULTI(8, 4, 4)
        NINFER_COHERENT_MULTI(4, 2, 2) NINFER_COHERENT_MULTI(4, 2, 4)
        NINFER_COHERENT_MULTI(8, 2, 2) NINFER_COHERENT_MULTI(8, 2, 4)
#undef NINFER_COHERENT_MULTI
        throw std::invalid_argument("unsupported coherent multi-tile variant");
    }
    if (warps == 8 && stages == 0) {
        exl3_gemm_m1_generic_mma_kernel<Bits, false, 8, true, true><<<
            dim3(grid), dim3(kThreads), two_stage_bytes, stream>>>(
                transformed, trellis, mul1, accum, rows, input_features,
                output_features, split_count);
        return;
    }
#define NINFER_COHERENT_VARIANT(S, W)                                          \
    if (stages == S && warps == W) {                                           \
        launch_coherent_packed_variant<Bits, S, W>(stream, transformed,        \
            trellis, mul1, accum, rows, input_features, output_features,       \
            split_count);                                                      \
        return;                                                                \
    }
    NINFER_COHERENT_VARIANT(4, 8) NINFER_COHERENT_VARIANT(8, 8)
    NINFER_COHERENT_VARIANT(0, 4) NINFER_COHERENT_VARIANT(4, 4)
    NINFER_COHERENT_VARIANT(8, 4) NINFER_COHERENT_VARIANT(0, 2)
    NINFER_COHERENT_VARIANT(4, 2) NINFER_COHERENT_VARIANT(8, 2)
#undef NINFER_COHERENT_VARIANT
    throw std::logic_error("unsupported coherent packed producer variant");
}

// NINFER_EXL3_COHERENT_KV_SPLIT=S (default 20, quality-gated; 0 = generic GEMV): the narrow 5120->1024
// K/V projections (K6..K8) run as the coherent split-plane producer with S
// K partitions instead of the 160-CTA generic GEMV. Numerics-policy candidate.
static int coherent_kv_split_setting() {
    static const int split = [] {
        const char* value = std::getenv("NINFER_EXL3_COHERENT_KV_SPLIT");
        const int parsed = value ? std::atoi(value) : 20;
        if (parsed < 0 || parsed > 80)
            throw std::invalid_argument("NINFER_EXL3_COHERENT_KV_SPLIT must be 0..80");
        return parsed;
    }();
    return split;
}

static int coherent_kv_split_for(const Exl3CudaLinearMetadata& metadata, int rows,
    Exl3CudaLinearAdmission admission, int in_features, int out_features,
    std::size_t capacity) {
    const int split = coherent_kv_split_setting();
    if (!split || rows < 1 || rows > 8 || in_features != 5120 || out_features != 1024 ||
        metadata.in_features != in_features || metadata.out_features != out_features ||
        metadata.K < 6 || metadata.K > 8 || metadata.mcg || !metadata.mul1 ||
        metadata.has_bias ||
        (admission != Exl3CudaLinearAdmission::ordinary &&
         admission != Exl3CudaLinearAdmission::target_continuation_kv))
        return 0;
    const auto required = static_cast<std::size_t>(rows) * out_features * split * sizeof(float);
    return capacity >= required ? split : 0;
}

void Exl3CudaLinearWorkspace::forward_from_transformed(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,
    const std::uint16_t* transformed_input,
    std::uint16_t* output,
    int rows,
    cudaStream_t stream,
    Exl3CudaLinearAdmission admission) {
    if((admission==Exl3CudaLinearAdmission::draft_shared_q_m16 || admission==Exl3CudaLinearAdmission::draft_shared_kv_m16 || admission==Exl3CudaLinearAdmission::draft_shared_o_m16 || admission==Exl3CudaLinearAdmission::draft_shared_down_m16 || admission==Exl3CudaLinearAdmission::draft_shared_gateup_m16) &&
       !draft_shared_m16_candidate(metadata,rows,admission))
        throw std::invalid_argument("shared transformed draft Q requires admitted M16 workspace");
    if(admission==Exl3CudaLinearAdmission::target_continuation_head &&
       !target_head_small_m_candidate(metadata,rows,admission))
        throw std::invalid_argument("shared transformed head requires admitted H6 small-M workspace");
    if (metadata.in_features != in_features_ || metadata.out_features != out_features_) {
        throw std::invalid_argument("unsupported EXL3 CUDA dimensions");
    }
    if (metadata.K < 4 || metadata.K > 8) {
        throw std::invalid_argument("unsupported EXL3 CUDA module K");
    }
    if (!allow_generic_variants_ && metadata.K != 5) {
        throw std::invalid_argument("unsupported EXL3 CUDA module K for legacy workspace");
    }
    if (metadata.mcg || !metadata.mul1 || metadata.has_bias) {
        throw std::invalid_argument("unsupported EXL3 CUDA flags");
    }
    if (rows <= 0 || rows > max_rows_) {
        throw std::invalid_argument("EXL3 CUDA row count exceeds workspace");
    }
    if (weights.trellis == nullptr || weights.svh == nullptr || weights.mul1 == nullptr ||
        transformed_input == nullptr || output == nullptr) {
        throw std::invalid_argument("EXL3 CUDA received a null device buffer");
    }
    // External pretransformed inputs obey the constructor-borrowed storage
    // contract before any reconstructed or ordinary dispatch can run.
    Exl3LinearWorkspaceRequirements::derive(in_features_,out_features_,rows,true,true)
        .require_disjoint_borrowed_views(transformed_input,accum_);

    if (const int kv_split = coherent_kv_split_for(metadata, rows, admission,
            in_features_, out_features_, accumulation_capacity_bytes_)) {
        if (metadata.K == 6)
            launch_coherent_packed_partials<6>(0, 0, stream, transformed_input,
                weights.trellis, weights.mul1, accum_, rows, in_features_, out_features_, kv_split);
        else if (metadata.K == 7)
            launch_coherent_packed_partials<7>(0, 0, stream, transformed_input,
                weights.trellis, weights.mul1, accum_, rows, in_features_, out_features_, kv_split);
        else
            launch_coherent_packed_partials<8>(0, 0, stream, transformed_input,
                weights.trellis, weights.mul1, accum_, rows, in_features_, out_features_, kv_split);
        cuda_check(cudaGetLastError(), "launch coherent K/V partials");
        launch_prefill_reduce_output<false>(stream, accum_, weights.svh, output, rows,
            out_features_, kv_split);
        cuda_check(cudaGetLastError(), "launch coherent K/V reduction/output");
        return;
    }

    if (coherent_down_k6_candidate(metadata, rows, admission)) {
        constexpr int output_blocks = 5120 / 128;
        const int split_count = coherent_split_override("NINFER_EXL3_COHERENT_DOWN_SPLIT", rows);
        // One packed-MMA producer instantiation serves M1 and M2..8. Each
        // K/N tile's B decode is reused across active independent row C
        // fragments; inactive physical M16 rows are zero and never stored.
        if (metadata.K == 5) launch_coherent_packed_partials<5>(
            output_blocks * split_count, kCoherentDownK6SharedBytes, stream,
            transformed_input, weights.trellis, weights.mul1, accum_,
            rows, in_features_, out_features_, split_count);
        else launch_coherent_packed_partials<6>(
            output_blocks * split_count, kCoherentDownK6SharedBytes, stream,
            transformed_input, weights.trellis, weights.mul1, accum_,
            rows, in_features_, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch coherent K6 down shared-row partials");
        launch_prefill_reduce_output<false>(stream,accum_, weights.svh, output, rows, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch coherent K6 down row reduction/output");
        process_coherent_down_k6_calls_.fetch_add(1, std::memory_order_relaxed);
        process_coherent_down_k6_rows_.fetch_add(
            static_cast<std::uint64_t>(rows), std::memory_order_relaxed);
        return;
    }

    if (coherent_down_k7_candidate(metadata, rows, admission)) {
        constexpr int output_blocks = 5120 / 128;
        const int split_count = coherent_split_override("NINFER_EXL3_COHERENT_DOWN_SPLIT", rows);
        launch_coherent_packed_partials<7>(
            output_blocks * split_count, kCoherentDownK7SharedBytes, stream,
            transformed_input, weights.trellis, weights.mul1, accum_,
            rows, in_features_, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch coherent K7 down shared-row partials");
        launch_prefill_reduce_output<false>(stream,accum_, weights.svh, output, rows, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch coherent K7 down row reduction/output");
        process_coherent_down_k7_calls_.fetch_add(1, std::memory_order_relaxed);
        process_coherent_down_k7_rows_.fetch_add(
            static_cast<std::uint64_t>(rows), std::memory_order_relaxed);
        return;
    }

    if (coherent_o_k7_candidate(metadata, rows, admission)) {
        constexpr int output_blocks = 5120 / 128;
        const int split_count = coherent_split_override("NINFER_EXL3_COHERENT_O_SPLIT", rows);
        launch_coherent_packed_partials<7>(
            output_blocks * split_count, kCoherentOK7SharedBytes, stream,
            transformed_input, weights.trellis, weights.mul1, accum_,
            rows, in_features_, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch coherent K7 O shared-row partials");
        launch_prefill_reduce_output<false>(stream,accum_, weights.svh, output, rows, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch coherent K7 O row reduction/output");
        process_coherent_o_k7_calls_.fetch_add(1, std::memory_order_relaxed);
        process_coherent_o_k7_rows_.fetch_add(
            static_cast<std::uint64_t>(rows), std::memory_order_relaxed);
        return;
    }

    if (coherent_wide_k6_candidate(metadata, rows, admission)) {
        const int output_blocks = out_features_ / kHadamard;
        const int split_count = coherent_wide_k6_split_count(rows);
        if (metadata.K == 5) launch_coherent_packed_partials<5>(
            output_blocks * split_count, kCoherentWideK6SharedBytes, stream,
            transformed_input, weights.trellis, weights.mul1, accum_,
            rows, in_features_, out_features_, split_count);
        else if (metadata.K == 7) launch_coherent_packed_partials<7>(
            output_blocks * split_count, kCoherentWideK6SharedBytes, stream,
            transformed_input, weights.trellis, weights.mul1, accum_,
            rows, in_features_, out_features_, split_count);
        else launch_coherent_packed_partials<6>(
            output_blocks * split_count, kCoherentWideK6SharedBytes, stream,
            transformed_input, weights.trellis, weights.mul1, accum_,
            rows, in_features_, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch coherent wide K6 shared-row partials");
        launch_prefill_reduce_output<false>(stream,accum_, weights.svh, output, rows, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch coherent wide K6 row reduction/output");
        coherent_wide_k6_calls_[coherent_wide_k6_operation_].fetch_add(
            1, std::memory_order_relaxed);
        coherent_wide_k6_rows_[coherent_wide_k6_operation_].fetch_add(
            static_cast<std::uint64_t>(rows), std::memory_order_relaxed);
        if (split_count == 10)
            coherent_wide_k6_split10_calls_[coherent_wide_k6_operation_].fetch_add(
                1, std::memory_order_relaxed);
        return;
    }

    if (fast_fp16_m2_8_fused_down_candidate(metadata,rows,admission)) {
        exl3_small_m_down_fused_output_kernel<<<
            dim3(out_features_/kHadamard),dim3(kThreads),0,stream>>>(
                transformed_input,weights.trellis,weights.mul1,weights.svh,
                output,rows,in_features_,out_features_);
        cuda_check(cudaGetLastError(),
            "launch fused-output packed FP16 M2-8 down projection");
        process_fast_fp16_m2_8_fused_down_calls_.fetch_add(
            1,std::memory_order_relaxed);
        return;
    }
    if(fast_fp16_m2_8_all_candidate(metadata,rows,admission) ||
       fast_fp16_m2_8_down_candidate(metadata,rows,admission)) {
        const bool down=in_features_==17408 && out_features_==5120;
        const int output_tiles_per_block=metadata.K==7?32:(down?16:32);
        const int output_blocks=out_features_/(16*output_tiles_per_block);
        const int capacity=fast_same_weights_fp16_m1_capacity_[
            metadata.K==7?2:(down?1:0)];
        int split_count=std::min(5,capacity/output_blocks);
        if(const char* requested=std::getenv("NINFER_EXL3_GENERIC_SPLITS");
           requested && *requested)
            split_count=std::min(split_count,std::max(1,std::atoi(requested)));
        if(split_count<1)throw std::runtime_error(
            "fast FP16 M2-8 down lacks resident capacity");
        // The prior small-M path loaded every A tile synchronously and
        // introduced an extra CTA barrier per K step. Stage A alongside the
        // already double-buffered packed B tile when explicitly requested.
        const bool async_a=fast_fp16_m2_8_async_a_enabled_;
        const std::size_t shared_bytes=(async_a?512u:256u)*sizeof(half)+
            2u*output_tiles_per_block*16u*metadata.K*sizeof(std::uint16_t)+
            16u*output_tiles_per_block*16u*sizeof(float);
        const std::uint16_t* trellis=weights.trellis;
        const std::int32_t* mul1=weights.mul1;
        float* accum=accum_;
        int input_features=in_features_,output_features=out_features_;
        void* args[]={&transformed_input,&trellis,&mul1,&accum,&rows,
            &input_features,&output_features,&split_count};
        void* kernel=metadata.K==7?
            (async_a?reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<7,false,32,true,true,true>):
                reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<7,false,32,false,true,true>)):
            down?(async_a?reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<6,false,16,true,true>):
                reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<6,false,16,false,true>)):
                (async_a?reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<6,false,32,true,true>):
                reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<6,false,32,false,true>));
        cuda_check(cudaLaunchKernel(
            kernel,
            dim3(output_blocks*split_count),dim3(kThreads),args,shared_bytes,
            stream),"launch packed FP16 M2-8 target projection");
        launch_prefill_reduce_output<false>(stream,
                accum_,weights.svh,output,rows,out_features_,split_count);
        cuda_check(cudaGetLastError(),"reduce packed FP16 M2-8 target projection");
        process_fast_fp16_m2_8_down_calls_.fetch_add(1,std::memory_order_relaxed);
        return;
    }

    // The opt-in native MGEMM policy must be evaluated before the established
    // rows==1 target selectors below.  Those selectors are intentionally kept
    // ahead of the ordinary fallback for the default configuration, but they
    // would otherwise consume the same K6/K7 projections and make this
    // separately labeled differential a dead branch during graph capture.
    if (fast_same_weights_fp16kv_m1_mgemm_policy_enabled_ &&
        rows == 1 && (metadata.K == 6 || metadata.K == 7) &&
        metadata.in_features == in_features_ &&
        metadata.out_features == out_features_ &&
        out_features_ != 248320 &&
        ((in_features_ == 5120 && out_features_ != 248320) ||
         (in_features_ == 17408 && out_features_ == 5120)) &&
        fast_same_weights_fp16kv_m1_mgemm_policy_grid_[metadata.K - 6] > 0) {
        const bool gate_up = in_features_ == 5120 && out_features_ == 17408;
        const bool k6_shape3 =
            (in_features_ == 5120 && out_features_ == 10240) ||
            (in_features_ == 17408 && out_features_ == 5120);
        const int grid = fast_same_weights_fp16kv_m1_mgemm_policy_grid_[
            metadata.K - 6];
        // Match the pinned Mia Blackwell selector: K6 wide gate/up uses
        // shape-4, K7 gate/up uses shape-2, and K6 down uses shape-3. K7
        // down and the remaining narrow routes use shape-2.
        const int block_threads = gate_up && metadata.K == 6 ? 256 : 512;
        const std::size_t shared_bytes = gate_up
            ? (metadata.K == 6
                   ? exl3_native_persistent_smem_bytes<6, 16, 512>()
                   : exl3_native_persistent_smem_bytes<7, 32, 128>())
            : (k6_shape3 && metadata.K == 6)
                   ? exl3_native_persistent_smem_bytes<6, 32, 256>()
                   : (metadata.K == 6
                          ? exl3_native_persistent_smem_bytes<6, 32, 128>()
                          : exl3_native_persistent_smem_bytes<7, 32, 128>());
        const std::uint16_t* transformed = transformed_input;
        const std::uint16_t* trellis = weights.trellis;
        half* result = reinterpret_cast<half*>(output);
        int row_count = rows;
        int input_features = in_features_;
        int output_features = out_features_;
        int* locks = reinterpret_cast<int*>(accum_ + out_features_);
        const half* svh = reinterpret_cast<const half*>(weights.svh);
        cuda_check(cudaMemsetAsync(
                       locks, 0,
                       static_cast<std::size_t>(out_features_ / 16 + 4) *
                           sizeof(int), stream),
                   "clear native Mia-GEMM-policy M1 locks");
        void* kernel_args[] = {&transformed, &trellis, &result, &row_count,
                               &input_features, &output_features, &locks, &svh};
        void* kernel = gate_up
            ? (metadata.K == 6
                   ? reinterpret_cast<void*>(
                         exl3_native_mia_m1_fp16_kernel<6, 16, 512>)
                   : reinterpret_cast<void*>(
                         exl3_native_mia_m1_fp16_kernel<7, 32, 128, true>))
                   : (k6_shape3 && metadata.K == 6
                         ? reinterpret_cast<void*>(
                               exl3_native_mia_m1_fp16_kernel<6, 32, 256>)
                         : (metadata.K == 6
                                ? reinterpret_cast<void*>(
                                      exl3_native_mia_m1_fp16_kernel<6, 32, 128>)
                                : reinterpret_cast<void*>(
                                      exl3_native_mia_m1_fp16_kernel<7, 32, 128,
                                                                     true>)));
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(grid), dim3(block_threads), kernel_args,
                       shared_bytes, stream),
                       "launch native Mia-policy single M1 GEMM");
        cuda_check(cudaGetLastError(),
                   "launch native Mia-policy single M1 output");
        process_fast_same_weights_fp16_m1_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    if (fast_native_mia_target_prefill_fp16_candidate(
            metadata, rows, admission)) {
        const bool down_family = in_features_ != 5120;
        constexpr int bits = 6;
        const int tile_k = down_family ? 32 : 16;
        const int tile_n = down_family ? 256 : 512;
        const int tiles = (in_features_ / tile_k) * (out_features_ / tile_n);
        const int grid = std::max(
            1, std::min(tiles, fast_native_mia_target_prefill_fp16_capacity_));
        const int block_threads = tile_k == 16 ? 256 : 512;
        const std::size_t shared_bytes = down_family
            ? exl3_native_persistent_smem_bytes<bits, 32, 256>()
            : exl3_native_persistent_smem_bytes<bits, 16, 512>();
        const std::uint16_t* transformed = transformed_input;
        const std::uint16_t* trellis = weights.trellis;
        half* result = reinterpret_cast<half*>(output);
        int row_count = rows;
        int input_features = in_features_;
        int output_features = out_features_;
        // The output-sized accumulation plane has a spare tail for the
        // cooperative inner's output-tile locks. It is disjoint from the
        // target output and from the canonical transformed input.
        int* locks = reinterpret_cast<int*>(
            accum_ + static_cast<std::size_t>(max_rows_) * out_features_);
        const half* svh = reinterpret_cast<const half*>(weights.svh);
        cuda_check(cudaMemsetAsync(
                       locks, 0,
                       static_cast<std::size_t>(out_features_ / 16) *
                           sizeof(int), stream),
                   "clear native Mia target-prefill locks");
        void* kernel_args[] = {&transformed, &trellis, &result, &row_count,
                               &input_features, &output_features, &locks, &svh};
        void* kernel = down_family
            ? reinterpret_cast<void*>(
                  exl3_native_mia_prefill_fp16_kernel<6, 32, 256>)
            : reinterpret_cast<void*>(
                  exl3_native_mia_prefill_fp16_kernel<6, 16, 512>);
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(grid), dim3(block_threads), kernel_args,
                       shared_bytes, stream),
                   "launch native Mia target-wide FP16 prefill GEMM");
        cuda_check(cudaGetLastError(),
                   "native Mia target-wide FP16 prefill launch");
        process_fast_native_mia_target_prefill_fp16_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    if (fast_native_persistent_prefill_candidate(metadata, rows, admission)) {
        const bool k7 = metadata.K == 7;
        const bool down_family = in_features_ == 17408;
        const int tile_k = down_family ? 32 : 16;
        const int tile_n = down_family ? (k7 ? 128 : 256) : 512;
        const int chunks = (rows + 15) / 16;
        const int capacity = fast_native_persistent_prefill_capacity_[metadata.K - 5];
        const int tiles = (in_features_ / tile_k) * (out_features_ / tile_n);
        const int grid_x = std::max(1, std::min(tiles, capacity / chunks));
        const int block_threads = tile_k == 16 ? 256 : 512;
        const std::size_t shared_bytes =
            metadata.K == 5
                ? (down_family
                       ? exl3_native_persistent_smem_bytes<5, 32, 256>()
                       : exl3_native_persistent_smem_bytes<5, 16, 512>())
                : metadata.K == 6
                    ? (down_family
                           ? exl3_native_persistent_smem_bytes<6, 32, 256>()
                           : exl3_native_persistent_smem_bytes<6, 16, 512>())
                    : (down_family
                           ? exl3_native_persistent_smem_bytes<7, 32, 128>()
                           : exl3_native_persistent_smem_bytes<7, 16, 512>());
        const std::uint16_t* transformed = transformed_input;
        const std::uint16_t* trellis = weights.trellis;
        float* result = accum_;
        int row_count = rows;
        int input_features = in_features_;
        int output_features = out_features_;
        // The first plane holds all row outputs. The following spare plane is
        // large enough for one lock per 16-wide output tile per row chunk.
        int* locks = reinterpret_cast<int*>(
            accum_ + static_cast<std::size_t>(max_rows_) * out_features_);
        const half* svh = reinterpret_cast<const half*>(weights.svh);
        cuda_check(cudaMemsetAsync(
                       locks, 0,
                       static_cast<std::size_t>(chunks) *
                           static_cast<std::size_t>(out_features_ / 16) *
                           sizeof(int), stream),
                   "clear native persistent prefill locks");
        void* kernel_args[] = {&transformed, &trellis, &result, &row_count,
                               &input_features, &output_features, &locks, &svh};
        void* kernel = nullptr;
        if (metadata.K == 5) {
            kernel = down_family
                ? reinterpret_cast<void*>(
                      exl3_native_persistent_prefill_kernel<5, 32, 256>)
                : reinterpret_cast<void*>(
                      exl3_native_persistent_prefill_kernel<5, 16, 512>);
        } else if (metadata.K == 6) {
            kernel = down_family
                ? reinterpret_cast<void*>(
                      exl3_native_persistent_prefill_kernel<6, 32, 256>)
                : reinterpret_cast<void*>(
                      exl3_native_persistent_prefill_kernel<6, 16, 512>);
        } else {
            kernel = down_family
                ? reinterpret_cast<void*>(
                      exl3_native_persistent_prefill_kernel<7, 32, 128, true>)
                : reinterpret_cast<void*>(
                      exl3_native_persistent_prefill_kernel<7, 16, 512>);
        }
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(grid_x, chunks), dim3(block_threads),
                       kernel_args, shared_bytes, stream),
                   "launch native persistent prefill GEMM");
        launch_prefill_reduce_output<false>(stream,
            accum_, weights.svh, output, rows, out_features_, 1);
        cuda_check(cudaGetLastError(),
                   "launch native persistent prefill output transform");
        return;
    }
    if (fast_wide_prefill_gemm_candidate(metadata, rows, admission)) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(
                       stream, &capture),
                   "fast Mia-parity wide-prefill capture query");
        // cuBLAS reconstruction is intentionally excluded from graph capture;
        // the established native route below remains the fallback.
        if (capture == cudaStreamCaptureStatusNone &&
            try_fast_wide_prefill_gemm_from_transformed(
                weights, metadata, transformed_input, output, rows, stream))
            return;
    }
    if (reconstructed_exact_candidate(metadata, rows, admission)) {
        if(reconstructed_exact_.ordered_stream)
            reconstructed_exact_.ordered_stream->require_identity(reconstructed_exact_.model_owner,
                reconstructed_exact_.bytes,reconstructed_exact_.allow_k6_down,
                reconstructed_exact_.data,reconstructed_exact_.allow_k6_gate_up);
        auto submission=reconstructed_exact_.ordered_stream ?
            reconstructed_exact_.ordered_stream->acquire(reinterpret_cast<std::uintptr_t>(stream)) :
            std::unique_lock<std::mutex>{};
        try {
            forward_reconstructed_exact_from_transformed(weights, metadata,
                transformed_input, output, rows, reconstructed_exact_.data,
                reconstructed_exact_.bytes, stream);
        } catch(...) {
            if(reconstructed_exact_.ordered_stream)reconstructed_exact_.ordered_stream->fail();
            throw;
        }
        if (reconstructed_exact_.stats) {
            ++reconstructed_exact_.stats->calls;
            reconstructed_exact_.stats->rows += rows;
            if(metadata.K==6) {
                if(metadata.in_features==5120) {
                    ++reconstructed_exact_.stats->k6_gate_up_calls;
                    reconstructed_exact_.stats->k6_gate_up_rows += rows;
                } else {
                    ++reconstructed_exact_.stats->k6_down_calls;
                    reconstructed_exact_.stats->k6_down_rows += rows;
                }
            }
        }
        return;
    }

    if (native_mtp_one_step_candidate(metadata, rows, admission)) {
        // The one-row bridge uses the same qualified K4 N64 body as the
        // closed wide-MTP route.  Keeping this admission separate prevents a
        // one-row caller from widening ordinary K4 dispatch.
        constexpr int output_tiles_per_block = 64;
        constexpr int split_count = 5;
        const int blocks = out_features_ / (16 * output_tiles_per_block);
        constexpr std::size_t shared_bytes =
            512u * sizeof(half) +
            2u * output_tiles_per_block * 16u * 4u * sizeof(std::uint16_t);
        exl3_prefill_direct_async_a_kernel<4, output_tiles_per_block, 2><<<
            dim3(blocks * split_count, 1), kThreads, shared_bytes, stream>>>(
            transformed_input, weights.trellis, weights.mul1, accum_,
            rows, in_features_, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch native MTP one-step K4 N64 async-A");
        launch_prefill_reduce_output<false>(stream,
            accum_, weights.svh, output, rows, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch native MTP one-step K4 reduction/output");
        return;
    }

    if (native_mtp_wide_prefill_candidate(metadata, rows, admission)) {
        // Native MTP owns this K4 route explicitly.  Keep the exact qualified
        // N64 async-A body from the discriminator and reduce its five split
        // planes in ascending order before the unchanged output Hadamard.
        constexpr int output_tiles_per_block = 64;
        constexpr int split_count = 5;
        const int blocks = out_features_ / (16 * output_tiles_per_block);
        constexpr std::size_t shared_bytes =
            512u * sizeof(half) +
            2u * output_tiles_per_block * 16u * 4u * sizeof(std::uint16_t);
        exl3_prefill_direct_async_a_kernel<4, output_tiles_per_block, 2><<<
            dim3(blocks * split_count, (rows + 15) / 16), kThreads,
            shared_bytes, stream>>>(
                transformed_input, weights.trellis, weights.mul1, accum_,
                rows, in_features_, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch native MTP K4 N64 async-A prefill");
        // prefill_reduce_output_kernel's non-shuffle path visits split planes
        // 0,1,...,4, preserving the qualified ascending reduction order.
        launch_prefill_reduce_output<false>(stream,
                accum_, weights.svh, output, rows, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch native MTP K4 N64 reduction/output");
        return;
    }

    if (fast_native_mia_m1_fp16_candidate(metadata, rows, admission)) {
        const bool k6_gate = metadata.K == 6 && in_features_ == 5120 &&
            out_features_ == 17408;
        const int grid = fast_native_persistent_m1_grid_[metadata.K - 6];
        const int block_threads = k6_gate ? 256 : 512;
        const std::size_t shared_bytes =
            metadata.K == 6
                ? (k6_gate
                       ? exl3_native_persistent_smem_bytes<6, 16, 512>()
                       : exl3_native_persistent_smem_bytes<6, 32, 256>())
                : exl3_native_persistent_smem_bytes<7, 32, 128>();
        const std::uint16_t* transformed = transformed_input;
        const std::uint16_t* trellis = weights.trellis;
        half* result = reinterpret_cast<half*>(output);
        int row_count = rows;
        int input_features = in_features_;
        int output_features = out_features_;
        // The direct Mia inner owns the final FP16 output and uses the same
        // one-lock-per-output-tile region as the FP32 differential adapter.
        int* locks = reinterpret_cast<int*>(accum_ + out_features_);
        const half* svh = reinterpret_cast<const half*>(weights.svh);
        cuda_check(cudaMemsetAsync(
                       locks, 0,
                       static_cast<std::size_t>(out_features_ / 16) *
                           sizeof(int), stream),
                   "clear native Mia-shaped M1 locks");
        void* kernel_args[] = {&transformed, &trellis, &result, &row_count,
                               &input_features, &output_features, &locks, &svh};
        void* kernel = metadata.K == 6
            ? (k6_gate
                   ? reinterpret_cast<void*>(
                         exl3_native_mia_m1_fp16_kernel<6, 16, 512>)
                   : reinterpret_cast<void*>(
                         exl3_native_mia_m1_fp16_kernel<6, 32, 256>))
            : reinterpret_cast<void*>(
                  exl3_native_mia_m1_fp16_kernel<7, 32, 128, true>);
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(grid), dim3(block_threads), kernel_args,
                       shared_bytes, stream),
                   "launch native Mia-shaped M1 FP16 GEMM");
        cuda_check(cudaGetLastError(),
                   "launch native Mia-shaped M1 FP16 output");
        process_fast_native_mia_m1_fp16_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }

    if (fast_native_persistent_m1_candidate(metadata, rows, admission)) {
        const bool k6_gate = metadata.K == 6 && in_features_ == 5120 &&
            out_features_ == 17408;
        const int grid = fast_native_persistent_m1_grid_[metadata.K - 6];
        const int block_threads = k6_gate ? 256 : 512;
        const std::size_t shared_bytes =
            metadata.K == 6
                ? (k6_gate
                       ? exl3_native_persistent_smem_bytes<6, 16, 512>()
                       : exl3_native_persistent_smem_bytes<6, 32, 256>())
                : exl3_native_persistent_smem_bytes<7, 32, 128>();
        const std::uint16_t* transformed = transformed_input;
        const std::uint16_t* trellis = weights.trellis;
        float* result = accum_;
        int row_count = rows;
        int input_features = in_features_;
        int output_features = out_features_;
        // The first accumulation plane is the persistent FP32 output. Keep
        // the one-lock-per-output-tile state in the following spare region of
        // the same five-plane workspace; it never aliases the output plane.
        int* locks = reinterpret_cast<int*>(accum_ + out_features_);
        const half* svh = reinterpret_cast<const half*>(weights.svh);
        cuda_check(cudaMemsetAsync(
                       locks, 0,
                       static_cast<std::size_t>(out_features_ / 16) *
                           sizeof(int),
                       stream),
                   "clear native persistent M1 locks");
        void* kernel_args[] = {&transformed, &trellis, &result, &row_count,
                               &input_features, &output_features, &locks, &svh};
        void* kernel = metadata.K == 6
            ? (k6_gate
                   ? reinterpret_cast<void*>(
                         exl3_native_persistent_m1_kernel<6, 16, 512>)
                   : reinterpret_cast<void*>(
                         exl3_native_persistent_m1_kernel<6, 32, 256>))
            : reinterpret_cast<void*>(
                  exl3_native_persistent_m1_kernel<7, 32, 128, true>);
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(grid), dim3(block_threads), kernel_args,
                       shared_bytes, stream),
                   "launch native persistent M1 GEMM");
        launch_prefill_reduce_output<false>(stream,
                accum_, weights.svh, output, 1, out_features_, 1);
        cuda_check(cudaGetLastError(),
                   "launch native persistent M1 FP32 output transform");
        process_fast_native_persistent_m1_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }

    if (native_k6_critical_path_candidate(metadata, rows, admission)) {
        const bool use_global_slices = native_k6_global_slices_enabled_ &&
            max_rows_ >= 2 && out_features_ % (32 * 16) == 0 &&
            native_k6_global_slices_capacity_ > 0 &&
            out_features_ / (32 * 16) <= native_k6_global_slices_capacity_;
        if (use_global_slices) {
            // Mia's shape-4 topology assigns a resident CTA to an output
            // block and a contiguous K slice, then folds those slices through
            // a global lock.  The lock row is outside the one-row result and
            // the cooperative capacity gate makes the ordered spin safe.
            constexpr int output_tiles_per_block = 32;
            const int output_blocks =
                out_features_ / (16 * output_tiles_per_block);
            int global_split_count =
                std::min(32, native_k6_global_slices_capacity_ / output_blocks);
            if (const char* requested = std::getenv(
                    "NINFER_EXL3_GLOBAL_SLICE_SPLITS");
                requested && *requested) {
                global_split_count = std::min(
                    global_split_count, std::max(1, std::atoi(requested)));
            }
            if (global_split_count < 1)
                throw std::runtime_error(
                    "native K6 global-slice launch has no resident capacity");
            const std::size_t shared_bytes =
                512u * sizeof(half) +
                2u * output_tiles_per_block * 16u * 6u *
                    sizeof(std::uint16_t) +
                16u * 512u * sizeof(float);
            const std::uint16_t* trellis = weights.trellis;
            const std::int32_t* mul1 = weights.mul1;
            float* accum = accum_;
            int input_features = in_features_;
            int output_features = out_features_;
            void* kernel_args[] = {&transformed_input, &trellis, &mul1, &accum,
                                   &rows, &input_features, &output_features,
                                   &global_split_count};
            void* kernel = reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<
                    6, false, 32, true, true, false, false, true, false,
                    false, true>);
            cuda_check(cudaMemsetAsync(
                           static_cast<void*>(accum_ + out_features_), 0,
                           static_cast<std::size_t>(output_blocks) *
                               sizeof(int), stream),
                       "clear native K6 global-slice locks");
            cuda_check(cudaLaunchCooperativeKernel(
                           kernel, dim3(output_blocks * global_split_count),
                           dim3(kThreads), kernel_args, shared_bytes, stream),
                       "launch native K6 global-slice cooperative GEMM");
            launch_prefill_reduce_output<false>(stream,
                accum_, weights.svh, output, rows, out_features_, 1);
            cuda_check(cudaGetLastError(),
                       "launch native K6 global-slice reduction/output");
            ++native_k6_critical_path_calls_;
            return;
        }
        // R49 differential: retain the current N32/Async-A partial topology and
        // ascending FP32 split reduction, replacing only K6's generic packed
        // state extraction with the exact two-word lane-window specialization.
        // The separately labeled register-pipeline differential keeps this
        // topology and arithmetic while changing only staging/fragment overlap.
        const bool use_register_pipeline_n16 =
            native_k6_register_pipeline_n16_enabled_ &&
            native_k6_register_pipeline_n16_capacity_ > 0 &&
            out_features_ % (16 * 16) == 0 &&
            out_features_ / (16 * 16) <=
                native_k6_register_pipeline_n16_capacity_;
        constexpr int register_pipeline_n16_tiles = 16;
        constexpr int register_pipeline_n32_tiles = 32;
        const int output_tiles_per_block = use_register_pipeline_n16
            ? register_pipeline_n16_tiles : register_pipeline_n32_tiles;
        const int output_blocks =
            out_features_ / (16 * output_tiles_per_block);
        const bool use_register_pipeline =
            !use_register_pipeline_n16 &&
            (native_k6_register_pipeline_enabled_ ||
             (native_k6_shape4_register_pipeline_enabled_ &&
              in_features_ == 5120 && out_features_ == 17408)) &&
            native_k6_register_pipeline_capacity_ >= output_blocks;
        const int capacity = use_register_pipeline_n16
            ? native_k6_register_pipeline_n16_capacity_
            : (use_register_pipeline ? native_k6_register_pipeline_capacity_
                                      : native_k6_critical_path_capacity_);
        int split_count = std::min(
            5, capacity / output_blocks);
        if (const char* requested = std::getenv("NINFER_EXL3_GENERIC_SPLITS");
            requested && *requested) {
            split_count = std::min(split_count,
                                   std::max(1, std::atoi(requested)));
        }
        const std::size_t shared_bytes = use_register_pipeline_n16
            ? 4u * 256u * sizeof(half) +
                4u * output_tiles_per_block * 16u * 6u *
                    sizeof(std::uint16_t) +
                16u * output_tiles_per_block * 16u * sizeof(float)
            : use_register_pipeline
            ? 4u * 256u * sizeof(half) +
                4u * output_tiles_per_block * 16u * 6u *
                    sizeof(std::uint16_t) +
                16u * 512u * sizeof(float)
            : 512u * sizeof(half) +
                2u * output_tiles_per_block * 16u * 6u *
                    sizeof(std::uint16_t) +
                16u * 512u * sizeof(float);
        const std::uint16_t* trellis = weights.trellis;
        const std::int32_t* mul1 = weights.mul1;
        float* accum = accum_;
        int input_features = in_features_;
        int output_features = out_features_;
        void* kernel_args[] = {&transformed_input, &trellis, &mul1, &accum,
                               &rows, &input_features, &output_features,
                               &split_count};
        void* kernel = use_register_pipeline_n16
            ? reinterpret_cast<void*>(
                  exl3_gemm_m1_generic_mma_kernel<
                      6, false, 16, true, true, false, false, true, false,
                      true>)
            : use_register_pipeline
            ? reinterpret_cast<void*>(
                  exl3_gemm_m1_generic_mma_kernel<
                      6, false, 32, true, true, false, false, true, false,
                      true>)
            : reinterpret_cast<void*>(
                  exl3_gemm_m1_generic_mma_kernel<
                      6, false, 32, true, true, false, false, true>);
        cuda_check(cudaLaunchKernel(
                       kernel,
                       dim3(output_blocks * split_count), dim3(kThreads),
                       kernel_args, shared_bytes, stream),
                   "launch native K6 critical-path partials");
        launch_prefill_reduce_output<false>(stream,
            accum_, weights.svh, output, rows, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch native K6 critical-path reduction/output");
        ++native_k6_critical_path_calls_;
        if (use_register_pipeline || use_register_pipeline_n16)
            ++native_k6_register_pipeline_calls_;
        return;
    }

    if (rows > 16 && target_wide_prefill_candidate(metadata, rows, admission) &&
        specialized_shape_ && metadata.K == kK && !target_staged_prefill_candidate(metadata,rows,admission)) {
        // The established shape4 kernel has sixteen physical rows. Keep its
        // exact arithmetic and finish each subrange before reusing accum_.
        for (int first = 0; first < rows; first += 16)
            forward_from_transformed(weights, metadata,
                transformed_input + static_cast<std::size_t>(first) * in_features_,
                output + static_cast<std::size_t>(first) * out_features_,
                std::min(16, rows - first), stream, admission);
        return;
    }
    if (!(fast_same_weights_fp16_m1_n16_enabled_ &&
          target_fast_same_weights_fp16_m1_candidate(metadata, rows, admission)) &&
        target_staged_prefill_candidate(metadata, rows, admission)) {
        if(specialized_shape_ && metadata.K==kK) {
            if(target_direct_partials_) {
                if(target_shape4_n64_candidate(metadata,rows,admission)) {
                    constexpr int shape4_n64_grid=(kTilesN/64)*kShape4Splits;
                    constexpr std::size_t shape4_n64_shared_bytes=
                        256u*sizeof(half)+2u*64u*80u*sizeof(std::uint16_t);
                    exl3_prefill_shape4_direct_kernel<64><<<
                        dim3(shape4_n64_grid,(rows+15)/16),kThreads,
                        shape4_n64_shared_bytes,stream>>>(
                            transformed_input,weights.trellis,weights.mul1,accum_,rows);
                    ++shape4_n64_calls_;shape4_n64_rows_+=rows;
                } else {
                    exl3_prefill_shape4_direct_kernel<32><<<
                        dim3(kShape4CooperativeGrid,(rows+15)/16),kThreads,
                        kShape4SharedBytes-16u*512u*sizeof(float),stream>>>(
                            transformed_input,weights.trellis,weights.mul1,accum_,rows);
                }
            } else
            exl3_prefill_shape4_partials_kernel<<<dim3(kShape4CooperativeGrid,(rows+15)/16),kThreads,kShape4SharedBytes,stream>>>(
                transformed_input,weights.trellis,weights.mul1,accum_,rows);
            cuda_check(cudaGetLastError(),"launch staged shape4 partials");
            if(target_reduce_shfl_) {
                if(target_reduce_shfl_min_barriers_) {
                    launch_prefill_reduce_output<true,true>(stream,
                        accum_,weights.svh,output,rows,out_features_,kShape4Splits);
                    ++reduce_shfl_min_barrier_calls_;reduce_shfl_min_barrier_rows_+=rows;
                } else launch_prefill_reduce_output<true>(stream,
                    accum_,weights.svh,output,rows,out_features_,kShape4Splits);
            }
            else
                launch_prefill_reduce_output<false>(stream,
                    accum_,weights.svh,output,rows,out_features_,kShape4Splits);
            cuda_check(cudaGetLastError(),"launch staged shape4 reduction");
            return;
        }
        const bool down = in_features_ == 17408 && out_features_ == 5120 &&
            (metadata.K == 6 || metadata.K == 7);
        const bool k7_tiles64_exact_splits =
            target_k7_tiles64_exact_splits_candidate(metadata,rows,admission);
        const bool k6_gateup_n32_pair_cta =
            target_k6_gateup_n32_pair_cta_candidate(metadata,rows,admission);
        const bool direct_tiles64 = !down &&
            ((metadata.K == 6 &&
              (target_direct_tiles64_ || k6_gateup_n32_pair_cta)) ||
             k7_tiles64_exact_splits);
        const int tiles = down ? 16 : (direct_tiles64 ? 64 : 32);
        const int blocks = out_features_ / (16 * tiles);
        const int capacity = down ? large_down_candidate_capacity_[metadata.K == 6 ? 0 : 1]
            : generic_capacity_[metadata.K == 5 ? 0 : (metadata.K == 6 ? 1 : 2)];
        // Pairing adjacent output blocks changes only CTA ownership. Select
        // splits from the established N32 control so every group retains the
        // identical K ranges and ordered partial reduction.
        const int split_blocks =
            (k7_tiles64_exact_splits || k6_gateup_n32_pair_cta)
            ? out_features_ / (16 * 32) : blocks;
        int splits = std::min(5, capacity / split_blocks);
        const char* requested = std::getenv(down ? "NINFER_EXL3_LARGE_DOWN_SPLITS" : "NINFER_EXL3_GENERIC_SPLITS");
        if (requested && *requested) splits = std::min(splits, std::max(1, std::atoi(requested)));
        const std::size_t shared_bytes = 256u * sizeof(half) +
            2u * tiles * 16u * metadata.K * sizeof(std::uint16_t) + 16u * tiles * 16u * sizeof(float);
        const dim3 grid(blocks * splits, (rows + 15) / 16);
        bool k6_gateup_warpgroup_dispatched = false;
        bool k6_gateup_n32_pair_cta_dispatched = false;
        bool k6_fast_decode_dispatched = false;
        bool k6_rowpair_n64_dispatched = false;
        bool k6_down_rowpair_dispatched = false;
        bool k7_tiles64_exact_splits_dispatched = false;
        bool k8_kv_async_a_dispatched = false;
        if(target_prefill_persisting_l2_candidate(metadata,rows,admission))
            (void)apply_prefill_persisting_l2(weights,metadata,stream);
        if(target_direct_partials_) {
        const std::size_t async_shared_bytes=shared_bytes-16u*tiles*16u*sizeof(float)+256u*sizeof(half);
        if (down && metadata.K == 6 && target_k6_down_rowpair_candidate(metadata,rows,admission)) {
            exl3_prefill_rowpair_kernel<6><<<
                dim3(blocks*splits,(rows+31)/32),kThreads,8192,stream>>>(
                    transformed_input,weights.trellis,weights.mul1,accum_,rows,
                    in_features_,out_features_,splits);
            k6_down_rowpair_dispatched=true;
        }
        else if (down && metadata.K == 6 && target_direct_async_all_) exl3_prefill_direct_async_a_kernel<6,16><<<grid,kThreads,async_shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (down && metadata.K == 6) exl3_prefill_direct_partials_kernel<6,16><<<grid,kThreads,shared_bytes-16u*tiles*16u*sizeof(float),stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (down && target_direct_async_all_) exl3_prefill_direct_async_a_kernel<7,16><<<grid,kThreads,async_shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (down) exl3_prefill_direct_partials_kernel<7,16><<<grid,kThreads,shared_bytes-16u*tiles*16u*sizeof(float),stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (metadata.K == 5) exl3_prefill_direct_partials_kernel<5><<<grid,kThreads,shared_bytes-16u*tiles*16u*sizeof(float),stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (k6_gateup_n32_pair_cta) {
            exl3_prefill_k6_gateup_n32_pair_cta_kernel<<<
                grid,512,async_shared_bytes,stream>>>(
                    transformed_input,weights.trellis,weights.mul1,accum_,rows,
                    in_features_,out_features_,splits);
            k6_gateup_n32_pair_cta_dispatched = true;
        }
        else if (target_k6_gateup_warpgroup_async_candidate(metadata,rows,admission)) {
            const std::size_t warpgroup_shared_bytes =
                2u * 2u * 256u * sizeof(half) +
                2u * 64u * 16u * 6u * sizeof(std::uint16_t);
            exl3_prefill_k6_gateup_warpgroup_async_kernel<<<
                grid,kThreads,warpgroup_shared_bytes,stream>>>(
                    transformed_input,weights.trellis,weights.mul1,accum_,rows,
                    in_features_,out_features_,splits);
            k6_gateup_warpgroup_dispatched = true;
        }
        else if (target_rowpair_k6_candidate(metadata,rows,admission)) exl3_prefill_rowpair_kernel<6><<<dim3((out_features_/(16*16))*splits,(rows+31)/32),kThreads,8192,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (k7_tiles64_exact_splits) {
            exl3_prefill_direct_async_a_kernel<7,64,2><<<
                grid,kThreads,async_shared_bytes,stream>>>(
                    transformed_input,weights.trellis,weights.mul1,accum_,rows,
                    in_features_,out_features_,splits);
            k7_tiles64_exact_splits_dispatched = true;
        }
        else if (target_rowpair_k7_candidate(metadata,rows,admission)) exl3_prefill_rowpair_kernel<7><<<dim3((out_features_/(16*16))*splits,(rows+31)/32),kThreads,9216,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if(target_k6_fast_decode_candidate(metadata,rows,admission)) {
            exl3_prefill_direct_async_a_kernel<6,64,2,false,true><<<
                grid,kThreads,async_shared_bytes,stream>>>(transformed_input,
                weights.trellis,weights.mul1,accum_,rows,in_features_,
                out_features_,splits);
            k6_fast_decode_dispatched=true;
        }
        else if(target_k6_rowpair_n64_candidate(metadata,rows,admission)) {
            constexpr std::size_t rowpair_shared_bytes=
                2u*2u*256u*sizeof(half)+
                2u*64u*16u*6u*sizeof(std::uint16_t);
            exl3_prefill_k6_rowpair_n64_kernel<<<
                dim3(blocks*splits,(rows+31)/32),512,rowpair_shared_bytes,stream>>>(
                    transformed_input,weights.trellis,weights.mul1,accum_,rows,
                    in_features_,out_features_,splits);
            k6_rowpair_n64_dispatched=true;
        }
        else if (metadata.K == 6 && target_direct_async_a_ && direct_tiles64) exl3_prefill_direct_async_a_kernel<6,64,2><<<grid,kThreads,async_shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (metadata.K == 6 && target_direct_async_a_) exl3_prefill_direct_async_a_kernel<6><<<grid,kThreads,async_shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (metadata.K == 6) exl3_prefill_direct_partials_kernel<6><<<grid,kThreads,shared_bytes-16u*tiles*16u*sizeof(float),stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (metadata.K == 7 && target_direct_async_all_) exl3_prefill_direct_async_a_kernel<7><<<grid,kThreads,async_shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (metadata.K == 7) exl3_prefill_direct_partials_kernel<7><<<grid,kThreads,shared_bytes-16u*tiles*16u*sizeof(float),stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (target_k8_kv_prefill_async_a_candidate(metadata,rows,admission)) {
            exl3_prefill_direct_async_a_kernel<8><<<grid,kThreads,async_shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
            k8_kv_async_a_dispatched = true;
        }
        else exl3_prefill_direct_partials_kernel<8><<<grid,kThreads,shared_bytes-16u*tiles*16u*sizeof(float),stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        } else {
        if (down && metadata.K == 6) exl3_prefill_partials_kernel<6,16><<<grid,kThreads,shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (down) exl3_prefill_partials_kernel<7,16><<<grid,kThreads,shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (metadata.K == 5) exl3_prefill_partials_kernel<5><<<grid,kThreads,shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (metadata.K == 6) exl3_prefill_partials_kernel<6><<<grid,kThreads,shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else if (metadata.K == 7) exl3_prefill_partials_kernel<7><<<grid,kThreads,shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        else exl3_prefill_partials_kernel<8><<<grid,kThreads,shared_bytes,stream>>>(transformed_input,weights.trellis,weights.mul1,accum_,rows,in_features_,out_features_,splits);
        }
        cuda_check(cudaGetLastError(), "launch staged prefill partials");
        if (k6_gateup_warpgroup_dispatched) ++k6_gateup_warpgroup_async_calls_;
        if (k6_gateup_n32_pair_cta_dispatched) {
            ++k6_gateup_n32_pair_cta_calls_;
            k6_gateup_n32_pair_cta_rows_ += rows;
        }
        if(k6_fast_decode_dispatched) {
            ++k6_fast_decode_calls_;
            k6_fast_decode_rows_+=rows;
        }
        if(k6_rowpair_n64_dispatched) {
            ++k6_rowpair_n64_calls_;
            k6_rowpair_n64_rows_+=rows;
        }
        if(k6_down_rowpair_dispatched) {
            ++k6_down_rowpair_calls_;
            k6_down_rowpair_rows_+=rows;
        }
        if (k7_tiles64_exact_splits_dispatched) ++k7_tiles64_exact_splits_calls_;
        if (k8_kv_async_a_dispatched) {
            ++target_k8_kv_prefill_async_a_calls_;
            target_k8_kv_prefill_async_a_rows_ += rows;
        }
        if(target_reduce_shfl_) {
            if(target_reduce_shfl_min_barriers_) {
                launch_prefill_reduce_output<true,true>(stream,accum_,weights.svh,output,rows,out_features_,splits);
                ++reduce_shfl_min_barrier_calls_;reduce_shfl_min_barrier_rows_+=rows;
            } else launch_prefill_reduce_output<true>(stream,accum_,weights.svh,output,rows,out_features_,splits);
        }
        else
            launch_prefill_reduce_output<false>(stream,accum_,weights.svh,output,rows,out_features_,splits);
        cuda_check(cudaGetLastError(), "launch staged prefill reduction/output");
        return;
    }
    if (fast_same_weights_fp16_m1_n16_enabled_ &&
        target_fast_same_weights_fp16_m1_candidate(metadata, rows, admission)) {
        // Separately labeled N16 differential: use the smaller 256-column
        // output tile used by the dense native decode family. It preserves
        // the packed FP16 operand decoder, FP32 MMA accumulation and native
        // split reduction while changing only CTA ownership.
        constexpr int output_tiles_per_block = 16;
        const int output_blocks = out_features_ / (16 * output_tiles_per_block);
        const int capacity = metadata.K == 7
            ? fast_same_weights_fp16_m1_n16_capacity_[2]
            : (in_features_ == 17408 && out_features_ == 5120)
                ? fast_same_weights_fp16_m1_n16_capacity_[1]
                : fast_same_weights_fp16_m1_n16_capacity_[0];
        int split_count = std::min(5, capacity / output_blocks);
        if (const char* requested = std::getenv("NINFER_EXL3_GENERIC_SPLITS");
            requested && *requested)
            split_count = std::min(split_count, std::max(1, std::atoi(requested)));
        if (split_count < 1)
            throw std::runtime_error("FAST same-weight FP16 M1 N16 has no resident capacity");
        const std::size_t shared_bytes =
            512u * sizeof(half) +
            2u * static_cast<std::size_t>(output_tiles_per_block) * 16u *
                static_cast<std::size_t>(metadata.K) * sizeof(std::uint16_t) +
            16u * static_cast<std::size_t>(output_tiles_per_block) * 16u *
                sizeof(float);
        const std::uint16_t* trellis = weights.trellis;
        const std::int32_t* mul1 = weights.mul1;
        float* accum = accum_;
        int input_features = in_features_;
        int output_features = out_features_;
        void* kernel_args[] = {&transformed_input, &trellis, &mul1, &accum,
                               &rows, &input_features, &output_features,
                               &split_count};
        void* kernel = metadata.K == 7
            ? reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<7, false, 16, true,
                                                 false, true>)
            : reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<6, false, 16, true>);
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(output_blocks * split_count),
                       dim3(kThreads), kernel_args, shared_bytes, stream),
                   "launch FAST_SAME_WEIGHTS FP16 M1 N16");
        launch_prefill_reduce_output<false>(stream,
            accum_, weights.svh, output, 1, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch FAST_SAME_WEIGHTS FP16 M1 N16 reduction/output");
        process_fast_same_weights_fp16_m1_calls_.fetch_add(
            1, std::memory_order_relaxed);
        process_fast_same_weights_fp16_m1_n16_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    const bool fast_same_weights_fp16_m1_n64_k5 =
        target_fast_same_weights_fp16_m1_n64_k5_candidate(metadata, rows, admission);
    if (fast_same_weights_fp16_m1_n64_k5 ||
        target_fast_same_weights_fp16_m1_n64_candidate(metadata, rows, admission)) {
        // Separately labeled N64 differential: one 256-thread CTA owns 64
        // output tiles (1024 values), halving the output-CTA count relative to
        // the rejected N32 screen. The packed EXL3 representation is decoded
        // directly into FP16 MMA operands; accumulation and final Hadamard
        // output remain the established FP32/FP16 native path.
        constexpr int output_tiles_per_block = 64;
        const int output_blocks = out_features_ / (16 * output_tiles_per_block);
        const int capacity =
            fast_same_weights_fp16_m1_n64_capacity_[metadata.K - 5];
        int split_count = std::min(5, capacity / output_blocks);
        if (const char* requested = std::getenv("NINFER_EXL3_GENERIC_SPLITS");
            requested && *requested)
            split_count = std::min(split_count, std::max(1, std::atoi(requested)));
        const std::size_t shared_bytes =
            512u * sizeof(half) +
            2u * static_cast<std::size_t>(output_tiles_per_block) * 16u *
                static_cast<std::size_t>(metadata.K) * sizeof(std::uint16_t) +
            16u * static_cast<std::size_t>(output_tiles_per_block) * 16u *
                sizeof(float);
        const std::uint16_t* trellis = weights.trellis;
        const std::int32_t* mul1 = weights.mul1;
        float* accum = accum_;
        int input_features = in_features_;
        int output_features = out_features_;
        void* kernel_args[] = {&transformed_input, &trellis, &mul1, &accum,
                               &rows, &input_features, &output_features,
                               &split_count};
        void* kernel = metadata.K == 5
            ? reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<5, false, 64, true>)
            : metadata.K == 6
                ? reinterpret_cast<void*>(
                    exl3_gemm_m1_generic_mma_kernel<6, false, 64, true>)
                : reinterpret_cast<void*>(
                    exl3_gemm_m1_generic_mma_kernel<7, false, 64, true,
                                                     false, true>);
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(output_blocks * split_count),
                       dim3(kThreads), kernel_args, shared_bytes, stream),
                   "launch FAST_SAME_WEIGHTS FP16 M1 N64 cooperative GEMV");
        launch_prefill_reduce_output<false>(stream,
            accum_, weights.svh, output, 1, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch FAST_SAME_WEIGHTS FP16 M1 N64 reduction/output");
        if (fast_same_weights_fp16_m1_n64_k5)
            process_fast_same_weights_fp16_m1_n64_k5_calls_.fetch_add(
                1, std::memory_order_relaxed);
        else
            process_fast_same_weights_fp16_m1_n64_calls_.fetch_add(
                1, std::memory_order_relaxed);
        return;
    }
    const bool fast_same_weights_fp16_m1_wide_candidate =
        ((fast_same_weights_fp16_m1_wide_enabled_ && metadata.K >= 6 &&
          metadata.K <= 7) ||
         (fast_same_weights_fp16_m1_wide_k5_enabled_ && metadata.K == 5)) &&
        rows == 1 && !metadata.mcg && metadata.mul1 && !metadata.has_bias &&
        (in_features_ == 5120 || in_features_ == 17408) &&
        out_features_ % (16 * 16) == 0 && out_features_ != 248320 &&
        (!fast_same_weights_fp16_m1_wide_n32_enabled_ ||
         out_features_ % (16 * 32) == 0);
    if (fast_same_weights_fp16_m1_wide_candidate) {
        // Separately labeled wide native M1 differential. One 512-thread CTA
        // owns a 256-column output tile; the two subgroups split K and reduce
        // into the same five native FP32 partial planes. The second operand
        // set is deliberately an alias because this is the single-projection
        // form of the same native kernel used by the opt-in K/V z-pair route.
        const int tiles=fast_same_weights_fp16_m1_wide_n32_enabled_ ? 32 : 16;
        const int output_blocks=out_features_/(16*tiles);
        int split_count=5;
        if(const char* requested=std::getenv("NINFER_EXL3_GENERIC_SPLITS");
           requested && *requested)
            split_count=std::min(split_count,std::max(1,std::atoi(requested)));
        const std::size_t raw_stage_half=static_cast<std::size_t>(tiles)*16u*
            static_cast<std::size_t>(metadata.K);
        const std::size_t shared_bytes=(4u*256u+4u*raw_stage_half)*sizeof(half)+
            2u*static_cast<std::size_t>(tiles*16)*sizeof(float);
        const std::uint16_t* trellis=weights.trellis;
        const std::int32_t* mul1=weights.mul1;
        float* accum=accum_;
        int input_features=in_features_,output_features=out_features_;
        void* kernel_args[]={&transformed_input,&transformed_input,&trellis,&mul1,
            &trellis,&mul1,&accum,&accum,&input_features,&output_features,
            &split_count};
        if(metadata.K==5) {
            if(fast_same_weights_fp16_m1_wide_n32_enabled_)
                cuda_check(cudaLaunchKernel(
                    reinterpret_cast<void*>(
                        exl3_target_m1_wide_projection_pair_n16_kernel<5,32>),
                    dim3(output_blocks*split_count,1,1),dim3(512),kernel_args,
                    shared_bytes,stream),"launch FAST wide native M1 K5 N32");
            else
                cuda_check(cudaLaunchKernel(
                    reinterpret_cast<void*>(
                        exl3_target_m1_wide_projection_pair_n16_kernel<5,16>),
                    dim3(output_blocks*split_count,1,1),dim3(512),kernel_args,
                    shared_bytes,stream),"launch FAST wide native M1 K5");
        } else if(metadata.K==6) {
            if(fast_same_weights_fp16_m1_wide_n32_enabled_)
                cuda_check(cudaLaunchKernel(
                    reinterpret_cast<void*>(
                        exl3_target_m1_wide_projection_pair_n16_kernel<6,32,true>),
                    dim3(output_blocks*split_count,1,1),dim3(512),kernel_args,
                    shared_bytes,stream),"launch FAST wide native M1 K6 N32");
            else
                cuda_check(cudaLaunchKernel(
                    reinterpret_cast<void*>(
                        exl3_target_m1_wide_projection_pair_n16_kernel<6,16,true>),
                    dim3(output_blocks*split_count,1,1),dim3(512),kernel_args,
                    shared_bytes,stream),"launch FAST wide native M1 K6");
        } else {
            if(fast_same_weights_fp16_m1_wide_n32_enabled_)
                cuda_check(cudaLaunchKernel(
                    reinterpret_cast<void*>(
                        exl3_target_m1_wide_projection_pair_n16_kernel<7,32>),
                    dim3(output_blocks*split_count,1,1),dim3(512),kernel_args,
                    shared_bytes,stream),"launch FAST wide native M1 K7 N32");
            else
                cuda_check(cudaLaunchKernel(
                    reinterpret_cast<void*>(
                        exl3_target_m1_wide_projection_pair_n16_kernel<7>),
                    dim3(output_blocks*split_count,1,1),dim3(512),kernel_args,
                    shared_bytes,stream),"launch FAST wide native M1 K7");
        }
        launch_prefill_reduce_output<false>(stream,
            accum_,weights.svh,output,1,out_features_,split_count);
        cuda_check(cudaGetLastError(),"reduce FAST wide native M1");
        process_fast_same_weights_fp16_m1_calls_.fetch_add(
            1,std::memory_order_relaxed);
        if(fast_same_weights_fp16_m1_wide_n32_enabled_)
            process_fast_same_weights_fp16_m1_wide_n32_calls_.fetch_add(
                1,std::memory_order_relaxed);
        return;
    }
    if (target_fast_same_weights_fp16_m1_candidate(metadata, rows, admission)) {
        // Mia's Blackwell M=1 reference uses FP16 operands with FP32 MMA
        // accumulation. Keep this native candidate on the existing packed
        // EXL3 decoder and cooperative split/reduction contract; the flag is
        // default-off and the fallback branches below remain untouched.
        const bool k6_down = metadata.K == 6 && in_features_ == 17408 &&
            out_features_ == 5120;
        const int output_tiles_per_block = k6_down ? 16 : 32;
        const int output_blocks = out_features_ / (16 * output_tiles_per_block);
        const int capacity = metadata.K == 7
            ? fast_same_weights_fp16_m1_capacity_[2]
            : fast_same_weights_fp16_m1_capacity_[k6_down ? 1 : 0];
        const bool use_k6_register_pipeline = metadata.K == 6 && !k6_down &&
            fast_same_weights_fp16_m1_k6_register_pipeline_enabled_ &&
            fast_same_weights_fp16_m1_k6_register_pipeline_capacity_ > 0 &&
            output_blocks <= fast_same_weights_fp16_m1_k6_register_pipeline_capacity_;
        const int launch_capacity = use_k6_register_pipeline
            ? fast_same_weights_fp16_m1_k6_register_pipeline_capacity_
            : capacity;
        int split_count = std::min(5, launch_capacity / output_blocks);
        if (const char* requested = std::getenv("NINFER_EXL3_GENERIC_SPLITS");
            requested && *requested)
            split_count = std::min(split_count, std::max(1, std::atoi(requested)));
        const std::size_t shared_bytes = use_k6_register_pipeline
            ? 4u * 256u * sizeof(half) +
                4u * 32u * 16u * 6u * sizeof(std::uint16_t) +
                16u * 512u * sizeof(float)
            : 512u * sizeof(half) +
                2u * static_cast<std::size_t>(output_tiles_per_block) * 16u *
                    static_cast<std::size_t>(metadata.K) * sizeof(std::uint16_t) +
                16u * static_cast<std::size_t>(output_tiles_per_block) * 16u *
                    sizeof(float);
        const std::uint16_t* trellis = weights.trellis;
        const std::int32_t* mul1 = weights.mul1;
        float* accum = accum_;
        int input_features = in_features_;
        int output_features = out_features_;
        void* kernel_args[] = {&transformed_input, &trellis, &mul1, &accum,
                               &rows, &input_features, &output_features,
                               &split_count};
        void* kernel = use_k6_register_pipeline
            ? reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<
                    6, false, 32, true, true, false, false, true,
                    false, true>)
            : metadata.K == 7
            ? reinterpret_cast<void*>(
                exl3_gemm_m1_generic_mma_kernel<7, false, 32, true,
                                                 false, true>)
            : k6_down
                ? reinterpret_cast<void*>(
                    exl3_gemm_m1_generic_mma_kernel<6, false, 16, true>)
                : reinterpret_cast<void*>(
                    exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>);
        cuda_check(cudaLaunchCooperativeKernel(
                       kernel, dim3(output_blocks * split_count),
                       dim3(kThreads), kernel_args, shared_bytes, stream),
                   "launch FAST_SAME_WEIGHTS FP16 M1 cooperative GEMV");
        launch_prefill_reduce_output<false>(stream,
            accum_, weights.svh, output, 1, out_features_, split_count);
        cuda_check(cudaGetLastError(),
                   "launch FAST_SAME_WEIGHTS FP16 M1 reduction/output");
        process_fast_same_weights_fp16_m1_calls_.fetch_add(
            1, std::memory_order_relaxed);
        return;
    }
    if(target_k6_m1_simt_candidate(metadata,rows,admission)) {
        const int mma_blocks=out_features_/(32*16);
        int split_count=std::min(5,generic_capacity_[1]/mma_blocks);
        if(const char* requested=std::getenv("NINFER_EXL3_GENERIC_SPLITS");
           requested && *requested)
            split_count=std::min(split_count,std::max(1,std::atoi(requested)));
        const int simt_blocks=out_features_/256;
        exl3_target_m1_simt_partials_kernel<6><<<
            simt_blocks*split_count,256,0,stream>>>(transformed_input,
                weights.trellis,weights.mul1,accum_,in_features_,out_features_,
                split_count);
        cuda_check(cudaGetLastError(),"launch target K6 M1 SIMT partials");
        launch_prefill_reduce_output<false>(stream,accum_,weights.svh,output,1,out_features_,
                split_count);
        cuda_check(cudaGetLastError(),"launch target K6 M1 SIMT reduction/output");
        ++target_k6_m1_simt_calls_;
        return;
    }
    if (target_initial16_candidate(metadata, rows, admission)) {
        switch (metadata.K) {
        case 5: exl3_initial16_ordered_kernel<5><<<out_features_ / 16, 256, 0, stream>>>(transformed_input, weights.trellis, weights.mul1, accum_, in_features_, out_features_); break;
        case 6: exl3_initial16_ordered_kernel<6><<<out_features_ / 16, 256, 0, stream>>>(transformed_input, weights.trellis, weights.mul1, accum_, in_features_, out_features_); break;
        case 7: exl3_initial16_ordered_kernel<7><<<out_features_ / 16, 256, 0, stream>>>(transformed_input, weights.trellis, weights.mul1, accum_, in_features_, out_features_); break;
        case 8: exl3_initial16_ordered_kernel<8><<<out_features_ / 16, 256, 0, stream>>>(transformed_input, weights.trellis, weights.mul1, accum_, in_features_, out_features_); break;
        }
    } else if (target_gateup_m16_candidate(metadata, rows, admission)) {
        exl3_gateup_m16_ordered_kernel<<<1088, 256, 0, stream>>>(
            transformed_input, weights.trellis, weights.mul1, accum_);
    } else if (draft_shared_gateup_m16_candidate(metadata, rows, admission)) {
        // Preserve the independent generic_tile FP32 accumulation sequence;
        // decode each K5 tile once for all sixteen private request rows.
        exl3_initial16_ordered_kernel<5><<<1088,256,0,stream>>>(
            transformed_input,weights.trellis,weights.mul1,accum_,5120,17408);
    } else if (specialized_shape_ && metadata.K == kK && rows <= 16) {
        const std::uint16_t* transformed = transformed_input;
        const std::uint16_t* trellis = weights.trellis;
        const std::int32_t* mul1 = weights.mul1;
        float* accum = accum_;
        int row_count = rows;
        void* kernel_args[] = {&transformed, &trellis, &mul1, &accum, &row_count};
        // Default: the same K5 five-split partition and fixed-order in-kernel
        // reduction through the narrow async-deep generic producer (64-column
        // CTAs; measured). NINFER_EXL3_SHAPE4_NARROW=0 keeps the shape-4 leaf.
        static const bool shape4_narrow = [] {
            const char* value = std::getenv("NINFER_EXL3_SHAPE4_NARROW");
            if (!value) return true;
            if (std::strcmp(value, "0") == 0) return false;
            if (std::strcmp(value, "1") == 0) return true;
            throw std::invalid_argument("NINFER_EXL3_SHAPE4_NARROW must be 0 or 1");
        }();
        if (shape4_narrow) {
            constexpr int warps = 4, stages = 4;
            int input_features = in_features_;
            int output_features = out_features_;
            int split_count = kShape4Splits;
            void* narrow_args[] = {&transformed, &trellis, &mul1, &accum, &row_count,
                                   &input_features, &output_features, &split_count};
            const std::size_t narrow_shared =
                static_cast<std::size_t>(stages) *
                    (256u * sizeof(half) + warps * 16u * 5u * sizeof(std::uint16_t)) +
                16u * warps * 16u * sizeof(float);
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<
                               5, false, warps, true, false, false, false, false, false,
                               false, false, stages, warps>),
                           dim3(out_features_ / (16 * warps) * split_count), dim3(warps * 32),
                           narrow_args, narrow_shared, stream),
                       "launch EXL3 narrow shape-4 split GEMV");
        } else
        cuda_check(cudaLaunchCooperativeKernel(
                       reinterpret_cast<void*>(exl3_gemm_m1_shape4_kernel),
                       dim3(cooperative_grid_), dim3(kThreads), kernel_args,
                       kShape4SharedBytes, stream),
                   "launch EXL3 cooperative split GEMV");
    } else if ((rows == 1 && in_features_ == 17408 && out_features_ == 5120 &&
                 (metadata.K == 6 || metadata.K == 7) && [&]() {
                   const char* requested = std::getenv("NINFER_EXL3_LARGE_DOWN_TOPOLOGY");
                   // The corrected 16-tile topology is the production default
                   // for this exact family. "32"/"off" retain an explicit
                   // inherited-topology control for differential benchmarks;
                   // "16-k6" and "16-k7" isolate the two qualification stages.
                   const std::string topology = requested == nullptr ? std::string() : std::string(requested);
                   return topology.empty() || topology == "16" || topology == "8" ||
                          (topology == "16-k6" && metadata.K == 6) ||
                          (topology == "16-k7" && metadata.K == 7);
               }()) || target_down_small_m_candidate(metadata, rows, admission) ||
               target_gateup_k6_n16_candidate(metadata, rows, admission) ||
               target_m1_k6_n16_candidate(metadata, rows, admission) ||
               (target_wide_prefill_candidate(metadata, rows, admission) &&
                in_features_ == 17408 && out_features_ == 5120 && (metadata.K == 6 || metadata.K == 7))) {
        const char* requested = std::getenv("NINFER_EXL3_LARGE_DOWN_TOPOLOGY");
        const bool requested_gateup_k6_n16 =
            target_gateup_k6_n16_candidate(metadata, rows, admission);
        const bool target_m1_k6_n16 =
            target_m1_k6_n16_candidate(metadata, rows, admission);
        const bool fast_same_weights_fp16_accum =
            fast_same_weights_fp16_accum_enabled_ && target_m1_k6_n16 &&
            metadata.K == 6 && rows == 1;
        const int candidate_tiles_per_block = !requested_gateup_k6_n16 &&
            !target_m1_k6_n16 &&
            requested != nullptr && std::string(requested) == "8" ? 8 : 16;
        const int tiles_n = out_features_ / 16;
        const int output_blocks = tiles_n / candidate_tiles_per_block;
        const bool target_down_k7_async_a =
            metadata.K == 7 && candidate_tiles_per_block == 16 &&
            target_down_k7_async_a_enabled_ &&
            target_down_small_m_candidate(metadata, rows, admission);
        const bool target_down_k6_async_a =
            metadata.K == 6 && candidate_tiles_per_block == 16 &&
            target_down_k6_async_a_candidate(metadata, rows, admission);
        const bool target_gateup_k6_n16 =
            metadata.K == 6 && candidate_tiles_per_block == 16 &&
            requested_gateup_k6_n16;
        const int candidate_capacity = target_gateup_k6_n16 || target_m1_k6_n16
            ? output_blocks * 5
            : target_down_k6_async_a
            ? target_down_k6_async_a_capacity_
            : target_down_k7_async_a
            ? target_down_k7_async_a_capacity_
            : candidate_tiles_per_block == 8
                ? large_down_candidate8_capacity_[metadata.K == 6 ? 0 : 1]
                : large_down_candidate_capacity_[metadata.K == 6 ? 0 : 1];
        if (candidate_capacity < output_blocks) {
            throw std::runtime_error("EXL3 large-down candidate exceeds cooperative launch capacity");
        }
        int split_count = std::min(5, candidate_capacity / output_blocks);
        const char* split_environment = target_gateup_k6_n16 || target_m1_k6_n16
            ? "NINFER_EXL3_GENERIC_SPLITS" : "NINFER_EXL3_LARGE_DOWN_SPLITS";
        if (const char* requested = std::getenv(split_environment);
            requested != nullptr && *requested != '\0') {
            split_count = std::min(split_count, std::max(1, std::atoi(requested)));
        }
        const std::size_t shared_bytes =
            ((target_down_k6_async_a || target_down_k7_async_a ||
              target_gateup_k6_n16 || target_m1_k6_n16) ? 512u : 256u) * sizeof(half) +
            2u * static_cast<std::size_t>(candidate_tiles_per_block) * 16u *
                static_cast<std::size_t>(metadata.K) * sizeof(std::uint16_t) +
            16u * static_cast<std::size_t>(candidate_tiles_per_block) * 16u * sizeof(float);
        const int grid_blocks = output_blocks * split_count;
        const std::uint16_t* trellis = weights.trellis;
        const std::int32_t* mul1 = weights.mul1;
        float* accum = accum_;
        int input_features = in_features_;
        int output_features = out_features_;
        void* kernel_args[] = {&transformed_input, &trellis, &mul1, &accum,
                               &rows, &input_features, &output_features, &split_count};
        const bool target_m1_k7_three_word =
            metadata.K == 7 && target_m1_k7_three_word_enabled_;
        if (target_m1_k7_three_word)
            process_target_m1_k7_three_word_calls_.fetch_add(
                1, std::memory_order_relaxed);
        if((extended_stream_reduction_candidate(metadata,rows,admission) ||
            target_gateup_k6_n16 || target_m1_k6_n16) &&
           candidate_tiles_per_block==16){
            cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;cuda_check(cudaStreamIsCapturing(stream,&capture),"extended N16 stream capture query");
            if(capture!=cudaStreamCaptureStatusNone &&
               !host_kv_gdn_segment_graph_capture_ &&
               !ordinary_graph_stream_reduction_allowed())
                throw std::invalid_argument("extended stream graph composition unqualified");
            auto kernel=metadata.K==6?
                (fast_same_weights_fp16_accum
                    ? reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<
                        6,false,16,true,true,false,false,false,true>)
                    : ((target_down_k6_async_a || target_gateup_k6_n16 ||
                        target_m1_k6_n16)
                        ? reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<6,false,16,true,true>)
                        : reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<6,false,16,false,true>))):
                target_down_k7_async_a
                    ? (target_m1_k7_three_word
                        ? reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<7,false,16,true,true,true>)
                        : reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<7,false,16,true,true>))
                    : (target_m1_k7_three_word
                        ? reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<7,false,16,false,true,true>)
                        : reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<7,false,16,false,true>));
            cuda_check(cudaLaunchKernel(kernel,dim3(grid_blocks),dim3(kThreads),kernel_args,shared_bytes,stream),"launch exact N16 stream partials");
            launch_prefill_reduce_output<false>(stream,accum_,weights.svh,output,rows,out_features_,split_count);
            cuda_check(cudaGetLastError(),"launch exact N16 stream reduction/output");
            ++extended_stream_reduction_calls_;
            if(target_down_k6_async_a)++target_down_k6_async_a_calls_;
            if(target_gateup_k6_n16)++target_gateup_k6_n16_calls_;
            if(target_m1_k6_n16)
                process_target_m1_k6_n16_calls_.fetch_add(
                    1,std::memory_order_relaxed);
            if(fast_same_weights_fp16_accum)
                process_fast_same_weights_fp16_accum_calls_.fetch_add(
                    1,std::memory_order_relaxed);
            return;
        }
        if (metadata.K == 6 && candidate_tiles_per_block == 16) {
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<6, false, 16>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 E4B3D K6 large-down candidate");
        } else if (metadata.K == 7 && candidate_tiles_per_block == 16 &&
                   target_down_k7_async_a) {
            cuda_check(cudaLaunchCooperativeKernel(
                           target_m1_k7_three_word
                               ? reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 16, true, false, true>)
                               : reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 16, true>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 target down K7 async-A candidate");
        } else if (metadata.K == 7 && candidate_tiles_per_block == 16) {
            cuda_check(cudaLaunchCooperativeKernel(
                           target_m1_k7_three_word
                               ? reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 16, false, false, true>)
                               : reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 16>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                       shared_bytes, stream),
                       "launch EXL3 E4B3D K7 large-down candidate");
        } else if (metadata.K == 6) {
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<6, false, 8>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 E4B3D K6 8-tile candidate");
        } else {
            cuda_check(cudaLaunchCooperativeKernel(
                           target_m1_k7_three_word
                               ? reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 8, false, false, true>)
                               : reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 8>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 E4B3D K7 8-tile candidate");
        }
    } else if ((rows == 1 || (h6_small_m_enabled_ && rows >= 2 && rows <= native_continuation_rows_)) &&
               in_features_ == 5120 && out_features_ == 248320 &&
               metadata.K == 6) {
        // The H6 vocabulary head has 485 output tiles, which cannot fit in
        // the cooperative resident-grid capacity used by the transformer
        // shapes.  It is a single K split, so a normal one-CTA-per-output
        // tile launch can use the same packed/MMA body without global
        // partial-reduction traffic or a cooperative grid barrier.
        // The unchanged body masks a 16-row A tile and writes rows*512 values;
        // E5A5H admits M2..8; an explicit exact-host owner additionally
        // admits M9..16 to fill the same physical MMA tile. Splits,
        // per-row reduction and the 45568-byte allocation are unchanged.
        const int output_blocks = out_features_ / (32 * 16);
        const std::size_t shared_bytes = 256u * sizeof(half) +
            2u * 32u * 16u * static_cast<std::size_t>(metadata.K) * sizeof(std::uint16_t) +
            16u * 512u * sizeof(float);
        const std::uint16_t* trellis = weights.trellis;
        const std::int32_t* mul1 = weights.mul1;
        float* accum = accum_;
        int input_features = in_features_;
        int output_features = out_features_;
        int split_count = 1;
        void* kernel_args[] = {&transformed_input, &trellis, &mul1, &accum,
                               &rows, &input_features, &output_features, &split_count};
        // Default: 64-column 4-warp CTAs with async-A, the 4x4 cp.async ring
        // and the exact K6 lane-window decoder (measured). Each warp still
        // accumulates its 16 columns over the whole K in order.
        // NINFER_EXL3_H6_NARROW=0 restores the 512-column launch.
        static const bool h6_narrow = [] {
            const char* value = std::getenv("NINFER_EXL3_H6_NARROW");
            if (!value) return true;
            if (std::strcmp(value, "0") == 0) return false;
            if (std::strcmp(value, "1") == 0) return true;
            throw std::invalid_argument("NINFER_EXL3_H6_NARROW must be 0 or 1");
        }();
        if (h6_narrow) {
            constexpr int warps = 4, stages = 4, per = 4;
            const std::size_t narrow_shared =
                static_cast<std::size_t>(stages * per) *
                    (256u * sizeof(half) + warps * 16u * 6u * sizeof(std::uint16_t)) +
                16u * warps * 16u * sizeof(float);
            cuda_check(cudaLaunchKernel(
                           reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<
                               6, true, warps, true, false, false, false, true, false,
                               false, false, stages, warps, per>),
                           dim3(out_features_ / (16 * warps)), dim3(warps * 32), kernel_args,
                           narrow_shared, stream),
                       "launch EXL3 H6 narrow single-split GEMV");
        } else
        cuda_check(cudaLaunchKernel(
                       reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<6, true>),
                       dim3(output_blocks), dim3(kThreads), kernel_args,
                       shared_bytes, stream),
                   "launch EXL3 H6 single-split GEMV");
    } else if (metadata.K >= 5 &&
               (rows == 1 || draft_small_m_candidate(metadata, rows) ||
                 draft_shared_m16_candidate(metadata,rows,admission) ||
                 draft_prefill_fc_candidate(metadata, rows, admission) ||
                 target_k5_small_m_batch_candidate(metadata, rows, admission) ||
                target_gateup_small_m_candidate(metadata, rows, admission) ||
                target_o_k7_small_m_candidate(metadata, rows, admission) ||
                target_o_k6_small_m_candidate(metadata, rows, admission) ||
                target_kv_small_m_candidate(metadata, rows, admission) ||
                target_z_k6_small_m_candidate(metadata, rows, admission) ||
                target_qkv_k6_small_m_candidate(metadata, rows, admission) ||
                target_q_k6_small_m_candidate(metadata, rows, admission) ||
                target_wide_prefill_candidate(metadata, rows, admission)) &&
               out_features_ % (32 * 16) == 0 &&
               (out_features_ / (32 * 16)) <=
                   generic_capacity_[metadata.K == 5 ? 0 : (metadata.K == 6 ? 1 : 2)]) {
        const int tiles_n = out_features_ / 16;
        const int output_blocks = tiles_n / 32;
        const int capacity_index = metadata.K == 5 ? 0 : (metadata.K == 6 ? 1 : 2);
        const bool target_gateup_k6_async_a =
            metadata.K == 6 && target_gateup_k6_async_a_enabled_ &&
            target_gateup_small_m_candidate(metadata, rows, admission);
        const bool draft_k5_async_a = metadata.K == 5 &&
            draft_k5_async_a_enabled_ && (draft_small_m_candidate(metadata, rows) ||
                draft_shared_m16_candidate(metadata,rows,admission));
        const bool target_m1_k6_n32_async_a = rows == 1 && metadata.K == 6 &&
            target_m1_k6_n32_async_a_enabled_;
        const bool target_m1_k7_n32_async_a = rows == 1 && metadata.K == 7 &&
            target_m1_k7_n32_async_a_enabled_;
        const bool target_k6_small_m_async_a =
            target_k6_small_m_async_a_candidate(metadata,rows,admission);
        const bool target_k7_small_m_async_a =
            target_k7_small_m_async_a_candidate(metadata,rows,admission);
        const int capacity = target_m1_k6_n32_async_a
            ? target_m1_k6_n32_async_a_capacity_
            : target_m1_k7_n32_async_a
            ? target_m1_k7_n32_async_a_capacity_
            : target_k6_small_m_async_a
            ? target_k6_small_m_async_a_capacity_
            : target_k7_small_m_async_a
            ? target_k7_small_m_async_a_capacity_
            : target_gateup_k6_async_a
            ? target_gateup_k6_async_a_capacity_
            : (draft_k5_async_a ? draft_k5_async_a_capacity_
                                : generic_capacity_[capacity_index]);
        if (capacity < output_blocks) {
            throw std::runtime_error("EXL3 generic cooperative kernel exceeds launch capacity");
        }
        int split_count = std::min(5, capacity / output_blocks);
        if (const char* requested = std::getenv("NINFER_EXL3_GENERIC_SPLITS"); requested && *requested) {
            split_count = std::min(split_count, std::max(1, std::atoi(requested)));
        }
        const std::size_t shared_bytes =
            ((target_m1_k6_n32_async_a || target_m1_k7_n32_async_a ||
              target_k6_small_m_async_a ||
              target_k7_small_m_async_a ||
              target_gateup_k6_async_a || draft_k5_async_a)
                 ? 512u : 256u) * sizeof(half) +
            2u * 32u * 16u * static_cast<std::size_t>(metadata.K) * sizeof(std::uint16_t) +
            16u * 512u * sizeof(float);
        const int grid_blocks = output_blocks * split_count;
        const std::uint16_t* trellis = weights.trellis;
        const std::int32_t* mul1 = weights.mul1;
        float* accum = accum_;
        int input_features = in_features_;
        int output_features = out_features_;
        void* kernel_args[] = {&transformed_input, &trellis, &mul1, &accum,
                               &rows, &input_features, &output_features, &split_count};
        const bool target_m1_k7_three_word =
            metadata.K == 7 && target_m1_k7_three_word_enabled_;
        if (target_m1_k7_three_word)
            process_target_m1_k7_three_word_calls_.fetch_add(
                1, std::memory_order_relaxed);
        if(extended_stream_reduction_candidate(metadata,rows,admission)&&metadata.K==7){
            cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;cuda_check(cudaStreamIsCapturing(stream,&capture),"extended K7 N32 capture query");
            if(capture!=cudaStreamCaptureStatusNone &&
               !host_kv_gdn_segment_graph_capture_ &&
               !ordinary_graph_stream_reduction_allowed())
                throw std::invalid_argument("extended stream graph composition unqualified");
            // The M1 K7 async-A option shares this exact stream-reduction
            // branch; unlike target_k7_small_m_async_a it is admitted for
            // rows==1 and must not be silently downgraded to sync-A.
            const bool k7_async_a = target_m1_k7_n32_async_a ||
                target_k7_small_m_async_a;
            auto kernel=k7_async_a
                ? (target_m1_k7_three_word
                    ? reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<7,false,32,true,true,true>)
                    : reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<7,false,32,true,true>))
                : (target_m1_k7_three_word
                    ? reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<7,false,32,false,true,true>)
                    : reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<7,false,32,false,true>));
            cuda_check(cudaLaunchKernel(kernel,dim3(grid_blocks),dim3(kThreads),kernel_args,shared_bytes,stream),"launch exact K7 N32 stream partials");
            launch_prefill_reduce_output<false>(stream,accum_,weights.svh,output,rows,out_features_,split_count);
            cuda_check(cudaGetLastError(),"launch exact K7 N32 stream reduction/output");
            ++extended_stream_reduction_calls_;
            if(target_k7_small_m_async_a)++target_k7_small_m_async_a_calls_;
            return;
        }
        if(k6_stream_reduction_candidate(metadata,rows,admission)) {
            cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
            cuda_check(cudaStreamIsCapturing(stream,&capture),"K6 stream reduction capture query");
            if(capture!=cudaStreamCaptureStatusNone &&
               !host_kv_gdn_segment_graph_capture_ &&
               !ordinary_graph_stream_reduction_allowed())
                throw std::invalid_argument("K6 stream reduction graph composition is unqualified");
            // Keep the parent's split count, raw double buffering, AsyncA
            // selection and MMA body. Only its grid barrier/reduction move:
            // the following output kernel reads completed split partials in
            // ascending order and retains the original shared Hadamard/casts.
            const bool async_a=target_m1_k6_n32_async_a ||
                target_k6_small_m_async_a || target_gateup_k6_async_a;
            auto kernel=async_a
                ? reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<6,false,32,true,true>)
                : reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<6,false,32,false,true>);
            cuda_check(cudaLaunchKernel(kernel,dim3(grid_blocks),dim3(kThreads),kernel_args,shared_bytes,stream),
                "launch exact K6 normal split partials");
            launch_prefill_reduce_output<false>(stream,
                accum_,weights.svh,output,rows,out_features_,split_count);
            cuda_check(cudaGetLastError(),"launch exact K6 stream reduction/output");
            ++stream_reduction_calls_;
            if(target_k6_small_m_async_a)++target_k6_small_m_async_a_calls_;
            return;
        }
        // Narrow M1 CTAs for small grids: each warp keeps 16 output columns
        // and the same split partition/in-kernel reduction, so results are
        // unchanged while the grid covers the SMs.
        const int narrow_warps = rows <= 16 && generic_narrow_enabled() &&
            output_blocks * split_count < 340
            ? (out_features_ <= 2048 ? 2 : 4) : 0;
        void* narrow_kernel = nullptr;
        if (narrow_warps) {
            const bool k7_async = metadata.K == 7 && target_m1_k7_n32_async_a;
            const bool k7_three = metadata.K == 7 && target_m1_k7_three_word;
// Narrow variants also stage A with cp.async and keep three packed tiles in
// flight; A values and the MMA sequence are unchanged.
#define NINFER_NARROW(B, A, T, W)                                             \
    reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<B, false, W, true,  \
        false, T, false, false, false, false, false, 4, W>)
            if (metadata.K == 5 && !draft_k5_async_a)
                narrow_kernel = narrow_warps == 2 ? NINFER_NARROW(5, false, false, 2)
                                                  : NINFER_NARROW(5, false, false, 4);
            else if (metadata.K == 8)
                narrow_kernel = narrow_warps == 2 ? NINFER_NARROW(8, false, false, 2)
                                                  : NINFER_NARROW(8, false, false, 4);
            else if (metadata.K == 7 && k7_async && k7_three)
                narrow_kernel = narrow_warps == 2 ? NINFER_NARROW(7, true, true, 2)
                                                  : NINFER_NARROW(7, true, true, 4);
            else if (metadata.K == 7 && !k7_async && k7_three)
                narrow_kernel = narrow_warps == 2 ? NINFER_NARROW(7, false, true, 2)
                                                  : NINFER_NARROW(7, false, true, 4);
            else if (metadata.K == 7 && !k7_async && !k7_three)
                narrow_kernel = narrow_warps == 2 ? NINFER_NARROW(7, false, false, 2)
                                                  : NINFER_NARROW(7, false, false, 4);
#undef NINFER_NARROW
        }
        if (narrow_kernel) {
            const std::size_t narrow_shared =
                4u * 256u * sizeof(half) +
                4u * static_cast<std::size_t>(narrow_warps) * 16u *
                    static_cast<std::size_t>(metadata.K) * sizeof(std::uint16_t) +
                16u * static_cast<std::size_t>(narrow_warps) * 16u * sizeof(float);
            const int narrow_grid = out_features_ / (16 * narrow_warps) * split_count;
            cuda_check(cudaLaunchCooperativeKernel(narrow_kernel, dim3(narrow_grid),
                           dim3(narrow_warps * 32), kernel_args, narrow_shared, stream),
                       "launch EXL3 narrow cooperative split GEMV");
            static std::atomic<unsigned> reported{0};
            if (reported.fetch_or(1u << metadata.K) & (1u << metadata.K)) {}
            else std::fprintf(stderr, "EXL3_GENERIC_NARROW K=%d in=%d out=%d warps=%d grid=%d\n",
                              metadata.K, in_features_, out_features_, narrow_warps, narrow_grid);
        } else if (metadata.K == 6 && target_m1_k6_n32_async_a) {
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(
                               exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 target M1 K6 N32 async-A cooperative split GEMV");
        } else if (metadata.K == 7 && target_m1_k7_n32_async_a) {
            cuda_check(cudaLaunchCooperativeKernel(
                           target_m1_k7_three_word
                               ? reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 32, true, false, true>)
                               : reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 32, true>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 target M1 K7 N32 async-A cooperative split GEMV");
        } else if (metadata.K == 5 && draft_k5_async_a) {
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(
                               exl3_gemm_m1_generic_mma_kernel<5, false, 32, true>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 draft K5 async-A cooperative split GEMV");
        } else if (metadata.K == 5) {
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<5>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 K5 cooperative split GEMV");
        } else if (metadata.K == 6 && target_gateup_k6_async_a) {
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(
                               exl3_gemm_m1_generic_mma_kernel<6, false, 32, true>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 target gate/up K6 async-A split GEMV");
        } else if (metadata.K == 6) {
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<6>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 K6 cooperative split GEMV");
        } else if (metadata.K == 7) {
            cuda_check(cudaLaunchCooperativeKernel(
                           target_m1_k7_three_word
                               ? reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7, false, 32, false, false, true>)
                               : reinterpret_cast<void*>(
                                   exl3_gemm_m1_generic_mma_kernel<7>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 K7 cooperative split GEMV");
        } else {
            cuda_check(cudaLaunchCooperativeKernel(
                           reinterpret_cast<void*>(exl3_gemm_m1_generic_mma_kernel<8>),
                           dim3(grid_blocks), dim3(kThreads), kernel_args,
                           shared_bytes, stream),
                       "launch EXL3 K8 cooperative split GEMV");
        }
    } else if (rows <= 16) {
        const int tiles_n = out_features_ / 16;
        exl3_generic_tile_kernel<<<dim3(tiles_n, rows), dim3(kThreads), 0, stream>>>(
            transformed_input, weights.trellis, weights.mul1, accum_, rows,
            in_features_, out_features_, metadata.K);
    } else {
        throw std::invalid_argument("EXL3 CUDA supports at most 16 rows");
    }

    cuda_check(cudaGetLastError(), "launch EXL3 packed GEMV");
    if (target_k5_small_m_batch_candidate(metadata, rows, admission))
        ++target_k5_small_m_batch_calls_;
    launch_output_hadamard(stream,
        accum_, weights.svh, output, rows, out_features_);
    cuda_check(cudaGetLastError(), "launch EXL3 output Hadamard");
}

} // namespace ninfer::exl3
