#pragma once

#include "ninfer/ops/kv_cache_append.h"

namespace ninfer::ops::detail {

void kv_cache_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                            PagedKVLayerView cache, cudaStream_t stream);

void kv_cache_append_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  const Tensor& valid_columns, const Tensor& table_rows,
                                  PagedKVBatchLayerView cache, cudaStream_t stream);

// NVFP4 attention owns an offset-aware append because its verifier can append a chunk from a
// full-width staging tensor before running the direct packed-cache attention kernel.
void kv_cache_append_nvfp4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        std::int32_t column_begin, std::int32_t width,
                                        PagedKVBatchLayerView cache, cudaStream_t stream);

struct KVCacheAppendPrefixPlan {
    std::int32_t tokens;
    std::int32_t min_count;
    std::int32_t max_count;
};

[[nodiscard]] KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope);

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& table_rows,
                                   PagedKVBatchLayerView cache, const KVCacheAppendPrefixPlan& plan,
                                   cudaStream_t stream);
void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& counts, const Tensor& lanes,
                                   CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream);
void kv_cache_append_prefix_oscar_dual_launch(
    const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
    const Tensor& lanes, CyclicKVCacheLayerView l0_cache,
    CyclicKVCacheLayerView q4_shadow_cache, const KVCacheAppendPrefixPlan& plan,
    cudaStream_t stream);

} // namespace ninfer::ops::detail
