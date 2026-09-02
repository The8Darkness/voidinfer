// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry>
__launch_bounds__(32) __global__ void causal_prompt_inverse_oscar_q2_kernel(
    __nv_bfloat16* __restrict__ out, std::int32_t tokens) {
    constexpr int D             = kCausalPromptHeadDim;
    const int q_head            = static_cast<int>(blockIdx.x);
    const int token             = static_cast<int>(blockIdx.y);
    const int lane              = static_cast<int>(threadIdx.x);
    if (q_head >= Geometry::QHeads || token >= tokens) return;

    float values[D / 32];
#pragma unroll
    for (int r = 0; r < D / 32; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(out[causal_prompt_q_index<Geometry>(q_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < D / 32; ++r) {
        const int d = lane + 32 * r;
        out[causal_prompt_q_index<Geometry>(q_head, d, token)] = __float2bfloat16(values[r]);
    }
}

template <typename Geometry>
void causal_prompt_inverse_oscar_q2_launch(Tensor& out, std::int32_t tokens,
                                           cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(Geometry::QHeads), static_cast<unsigned>(tokens), 1u);
    causal_prompt_inverse_oscar_q2_kernel<Geometry><<<grid, 32, 0, stream>>>(
        static_cast<__nv_bfloat16*>(out.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata, bool OscarQ2 = false>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    const Tensor& cache_k_scale = cache.k_scale_pages;
    const Tensor& cache_v_scale = cache.v_scale_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptSmemBytes);
    CUDA_CHECK(attr_bf16);
    if constexpr (OscarQ2) {
        static const cudaError_t attr_oscar =
            cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata, true>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 kCausalPromptSmemBytes);
        CUDA_CHECK(attr_oscar);
    }
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(causal_attention_prompt_i8_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
    CUDA_CHECK(attr_i8);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if constexpr (OscarQ2) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_bf16_kernel<Geometry, Metadata, true>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), cache_k.data, cache_v.data,
                cache_k_scale.data, cache_v_scale.data, metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else if (cache.dtype == DType::I8) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        causal_attention_prompt_i8_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __nv_bfloat16*>(cache_v.data), nullptr, nullptr, metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
    if constexpr (OscarQ2) {
        causal_prompt_inverse_oscar_q2_launch<Geometry>(out, tokens, stream);
    }
}

} // namespace

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              cudaStream_t stream) {
    if (cache.dtype == DType::FP8_E4M3FN) {
        causal_attention_prompt_fp8_attention_launch(q, positions, scale, cache, out, stream);
        return;
    }
    const CausalPromptDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    if (cache.dtype == DType::FP8_E4M3FN) {
        causal_attention_prompt_fp8_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                           cache, out, stream);
        return;
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const CausalPromptBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

void causal_attention_prompt_oscar_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    kv_cache_append_oscar_batch_launch(k, v, positions, valid_columns, table_rows, 0, q.ne[2],
                                       cache, stream);
    const auto launch = [&]<bool Masked>() {
        const CausalPromptBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4, PagedKVBatchLayerView,
                                                         decltype(metadata), true>(
                q, positions, scale, cache, metadata, out, stream);
        } else {
            causal_attention_prompt_attention_launch_for<CausalD256H16Kv2, PagedKVBatchLayerView,
                                                         decltype(metadata), true>(
                q, positions, scale, cache, metadata, out, stream);
        }
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
