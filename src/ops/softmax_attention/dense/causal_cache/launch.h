#pragma once

// ninfer::ops::detail - private launch prototypes for causal_softmax_attention policies.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class CausalAttentionRoute { SmallT, ChunkedSmallT, Prompt };

struct CausalSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

std::int32_t causal_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                             DType cache_dtype,
                                             CausalAttentionExecutionEnvelope envelope);

bool causal_attention_uses_small_t(std::int32_t tokens);

CausalAttentionRoute causal_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                                    std::int32_t batch_size,
                                                    CausalAttentionExecutionEnvelope envelope);

const char* causal_attention_route_name(CausalAttentionRoute route);

void causal_attention_small_t_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                            const PagedKVLayerView& cache,
                                            CausalAttentionExecutionEnvelope envelope,
                                            Tensor& partial_acc, Tensor& partial_m,
                                            Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_small_t_fp8_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    CausalAttentionExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream);

void causal_attention_cached_small_t_fp8_launch(const Tensor& q, const Tensor& positions,
                                                float scale, const PagedKVLayerView& cache,
                                                CausalAttentionExecutionEnvelope envelope,
                                                Tensor& partial_acc, Tensor& partial_m,
                                                Tensor& partial_l, Tensor& out,
                                                cudaStream_t stream);

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream);

// Batch-1 OSCAR-Q2 prompt path: append packed rows, stage/dequantize 64-key tiles into shared
// BF16, then reuse the tensor-core prompt attention pipeline.
void causal_attention_prompt_oscar_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream);

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              cudaStream_t stream);

void causal_attention_prompt_fp8_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& positions, const Tensor& valid_columns,
                                        const Tensor& table_rows, float scale,
                                        PagedKVBatchLayerView cache, Tensor& out,
                                        cudaStream_t stream);

void causal_attention_prompt_fp8_attention_launch(const Tensor& q, const Tensor& positions,
                                                  float scale, const PagedKVLayerView& cache,
                                                  Tensor& out, cudaStream_t stream);

// NVFP4 is a packed draft-cache route. It bypasses the tensor-core partial reducers and uses a
// direct warp kernel after the current K/V chunk has been encoded into the packed page store.
void causal_attention_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream);

void causal_attention_nvfp4_cached_launch(const Tensor& q, const Tensor& positions, float scale,
                                          const PagedKVLayerView& cache, Tensor& out,
                                          cudaStream_t stream);

void causal_attention_oscar_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream);

void causal_attention_oscar_cached_launch(const Tensor& q, const Tensor& positions, float scale,
                                          const PagedKVLayerView& cache, Tensor& out,
                                          cudaStream_t stream);

} // namespace ninfer::ops::detail
