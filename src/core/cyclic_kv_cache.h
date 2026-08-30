#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

/**
 * Fixed cyclic BF16 K/V storage with absolute-position addressing.
 *
 * A logical absolute position p resides in physical slot p % capacity. The view deliberately
 * carries no mutable frontier: callers supply the live absolute interval to the consuming Op.
 */
struct CyclicKVCacheLayerView {
    Tensor k;
    Tensor v;
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::int32_t lane_capacity    = 0;
    // Quantized cyclic caches expose one scale plane per K/V stream.  They remain empty for
    // BF16, so existing callers can keep constructing the view with only K/V tensors.
    Tensor k_scale;
    Tensor v_scale;
    // Optional BF16 sidecar for recent/high-sensitivity tokens.  This is populated alongside a
    // packed NVFP4 cache and is deliberately empty for the default/stable cache profile.
    Tensor protected_k;
    Tensor protected_v;
    // protected_capacity is the power-of-two recent-token ring. The optional prefix anchor
    // region occupies slots [0, protected_anchor_capacity) in the same BF16 sidecar; recent
    // slots start immediately after it.
    std::uint32_t protected_capacity        = 0;
    std::uint32_t protected_anchor_capacity = 0;
    std::uint32_t protected_padded_capacity = 0;
    DType dtype              = DType::BF16;
    std::int32_t quant_group = 0;
};

struct CyclicKVCacheSlotView {
    Tensor k_layer0;
    Tensor v_layer0;
    std::size_t k_layer_bytes          = 0;
    std::size_t v_layer_bytes          = 0;
    std::ptrdiff_t k_layer_pitch_bytes = 0;
    std::ptrdiff_t v_layer_pitch_bytes = 0;
    std::uint32_t layers               = 0;
    Tensor k_scale_layer0;
    Tensor v_scale_layer0;
    std::size_t k_scale_layer_bytes          = 0;
    std::size_t v_scale_layer_bytes          = 0;
    std::ptrdiff_t k_scale_layer_pitch_bytes = 0;
    std::ptrdiff_t v_scale_layer_pitch_bytes = 0;
    Tensor protected_k_layer0;
    Tensor protected_v_layer0;
    std::size_t protected_k_layer_bytes          = 0;
    std::size_t protected_v_layer_bytes          = 0;
    std::ptrdiff_t protected_k_layer_pitch_bytes = 0;
    std::ptrdiff_t protected_v_layer_pitch_bytes = 0;
    std::uint32_t protected_capacity              = 0;
    std::uint32_t protected_anchor_capacity      = 0;
    std::uint32_t protected_padded_capacity      = 0;
};

struct CyclicKVCacheLayout {
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t num_kv_heads     = 0;
    std::int32_t head_dim         = 0;
    std::int32_t lane_capacity    = 0;
    std::vector<TensorRegion> k;
    std::vector<TensorRegion> v;
    std::vector<TensorRegion> k_scale;
    std::vector<TensorRegion> v_scale;
    std::vector<TensorRegion> protected_k;
    std::vector<TensorRegion> protected_v;
    std::uint32_t protected_capacity        = 0;
    std::uint32_t protected_anchor_capacity = 0;
    std::uint32_t protected_padded_capacity = 0;
    DType dtype              = DType::BF16;
    std::int32_t quant_group = 0;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

[[nodiscard]] CyclicKVCacheLayout
plan_cyclic_kv_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                     std::int32_t num_kv_heads, std::int32_t head_dim, std::int32_t lane_capacity,
                     DType dtype = DType::BF16, std::int32_t quant_group = 0,
                     std::uint32_t protected_capacity = 0,
                     std::uint32_t protected_anchor_capacity = 0);

class CyclicKVCache {
public:
    CyclicKVCache(DeviceSpan backing, const CyclicKVCacheLayout& layout);

    CyclicKVCache(const CyclicKVCache&)            = delete;
    CyclicKVCache& operator=(const CyclicKVCache&) = delete;
    CyclicKVCache(CyclicKVCache&&)                 = delete;
    CyclicKVCache& operator=(CyclicKVCache&&)      = delete;

    [[nodiscard]] std::uint32_t layer_count() const noexcept;

    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::uint32_t padded_capacity() const noexcept { return padded_capacity_; }

    [[nodiscard]] std::int32_t num_kv_heads() const noexcept { return num_kv_heads_; }

    [[nodiscard]] std::int32_t head_dim() const noexcept { return head_dim_; }

    [[nodiscard]] std::int32_t lane_capacity() const noexcept { return lane_capacity_; }

    [[nodiscard]] DType dtype() const noexcept { return dtype_; }

    [[nodiscard]] std::int32_t quant_group() const noexcept { return quant_group_; }

    [[nodiscard]] std::uint32_t protected_capacity() const noexcept {
        return protected_capacity_;
    }

    [[nodiscard]] std::uint32_t protected_padded_capacity() const noexcept {
        return protected_padded_capacity_;
    }

    [[nodiscard]] std::uint32_t protected_anchor_capacity() const noexcept {
        return protected_anchor_capacity_;
    }

    [[nodiscard]] CyclicKVCacheLayerView layer_view(std::uint32_t layer) const;
    [[nodiscard]] CyclicKVCacheSlotView slot_view(std::int32_t slot) const;

    // Copies one slot's complete fixed state. Source and destination geometry must match.
    void copy_slot_from(const CyclicKVCache& source, std::int32_t source_slot,
                        std::int32_t destination_slot, cudaStream_t stream);

private:
    std::vector<Tensor> k_;
    std::vector<Tensor> v_;
    std::uint32_t capacity_        = 0;
    std::uint32_t padded_capacity_ = 0;
    std::int32_t num_kv_heads_     = 0;
    std::int32_t head_dim_         = 0;
    std::int32_t lane_capacity_    = 0;
    std::int32_t code_head_dim_    = 0;
    std::int32_t scale_extent_     = 0;
    DType dtype_                   = DType::BF16;
    std::int32_t quant_group_      = 0;
    std::uint32_t protected_capacity_        = 0;
    std::uint32_t protected_anchor_capacity_ = 0;
    std::uint32_t protected_padded_capacity_ = 0;
    std::vector<Tensor> k_scale_;
    std::vector<Tensor> v_scale_;
    std::vector<Tensor> protected_k_;
    std::vector<Tensor> protected_v_;
};

} // namespace ninfer
