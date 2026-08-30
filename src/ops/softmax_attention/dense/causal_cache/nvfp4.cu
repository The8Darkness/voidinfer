#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, bool Masked>
void launch_nvfp4_attention(const Tensor& q, const Tensor& positions,
                            const Tensor& valid_columns, const Tensor& table_rows, float scale,
                            const PagedKVBatchLayerView& cache, Tensor& out,
                            cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(Geometry::QHeads),
                    static_cast<unsigned>(q.ne[2]), static_cast<unsigned>(q.ne[3]));
    causal_attention_nvfp4_kernel<Geometry, Masked><<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const std::uint8_t*>(cache.k_pages.data),
        static_cast<const std::uint8_t*>(cache.v_pages.data),
        static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
        static_cast<const std::uint8_t*>(cache.v_scale_pages.data),
        static_cast<const std::int32_t*>(positions.data),
        Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        cache.block_tables.ne[0], q.ne[2], scale, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, bool Masked>
void launch_nvfp4_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache, Tensor& out,
                                   cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(Geometry::QHeads),
                    static_cast<unsigned>(q.ne[2]), 1u);
    causal_attention_nvfp4_kernel<Geometry, Masked><<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const std::uint8_t*>(cache.k_pages.data),
        static_cast<const std::uint8_t*>(cache.v_pages.data),
        static_cast<const std::uint8_t*>(cache.k_scale_pages.data),
        static_cast<const std::uint8_t*>(cache.v_scale_pages.data),
        static_cast<const std::int32_t*>(positions.data), nullptr, nullptr,
        static_cast<const std::int32_t*>(cache.block_table.data), cache.block_table.ne[0],
        cache.block_table.ne[0], q.ne[2], scale, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <bool Masked>
void launch_nvfp4_attention_for_heads(const Tensor& q, const Tensor& positions,
                                      const Tensor& valid_columns, const Tensor& table_rows,
                                      float scale, const PagedKVBatchLayerView& cache,
                                      Tensor& out, cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_nvfp4_attention<CausalD256H24Kv4, Masked>(q, positions, valid_columns, table_rows,
                                                         scale, cache, out, stream);
    } else {
        launch_nvfp4_attention<CausalD256H16Kv2, Masked>(q, positions, valid_columns, table_rows,
                                                         scale, cache, out, stream);
    }
}

} // namespace

void causal_attention_nvfp4_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    kv_cache_append_nvfp4_batch_launch(k, v, positions, valid_columns, table_rows, 0, q.ne[2],
                                       cache, stream);
    if (valid_columns.data == nullptr) {
        launch_nvfp4_attention_for_heads<false>(q, positions, valid_columns, table_rows, scale,
                                                cache, out, stream);
    } else {
        launch_nvfp4_attention_for_heads<true>(q, positions, valid_columns, table_rows, scale,
                                               cache, out, stream);
    }
}

void causal_attention_nvfp4_cached_launch(const Tensor& q, const Tensor& positions, float scale,
                                          const PagedKVLayerView& cache, Tensor& out,
                                          cudaStream_t stream) {
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        launch_nvfp4_cached_attention<CausalD256H24Kv4, false>(q, positions, scale, cache, out,
                                                               stream);
    } else {
        launch_nvfp4_cached_attention<CausalD256H16Kv2, false>(q, positions, scale, cache, out,
                                                                stream);
    }
}

} // namespace ninfer::ops::detail
