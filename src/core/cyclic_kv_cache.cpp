#include "core/cyclic_kv_cache.h"

#include "core/device.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

std::uint32_t align_up_u32(std::uint32_t value, std::uint32_t alignment) {
    const std::uint64_t mask    = static_cast<std::uint64_t>(alignment) - 1U;
    const std::uint64_t aligned = (static_cast<std::uint64_t>(value) + mask) & ~mask;
    if (aligned > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("Cyclic KV padded capacity is out of range");
    }
    return static_cast<std::uint32_t>(aligned);
}

std::ptrdiff_t layer_pitch(const std::vector<Tensor>& layers, const char* label) {
    if (layers.empty()) { throw std::logic_error(std::string("Cyclic KV has no ") + label); }
    if (layers.size() == 1) { return 0; }
    const auto first  = reinterpret_cast<std::uintptr_t>(layers[0].data);
    const auto second = reinterpret_cast<std::uintptr_t>(layers[1].data);
    if (second <= first ||
        second - first > static_cast<std::uintptr_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::logic_error(std::string("Cyclic KV ") + label + " layer pitch is invalid");
    }
    const auto pitch = static_cast<std::ptrdiff_t>(second - first);
    if (static_cast<std::size_t>(pitch) < layers.front().bytes()) {
        throw std::logic_error(std::string("Cyclic KV ") + label + " layers overlap");
    }
    for (std::size_t layer = 2; layer < layers.size(); ++layer) {
        const auto previous = reinterpret_cast<std::uintptr_t>(layers[layer - 1].data);
        const auto current  = reinterpret_cast<std::uintptr_t>(layers[layer].data);
        if (current <= previous || current - previous != static_cast<std::uintptr_t>(pitch)) {
            throw std::logic_error(std::string("Cyclic KV ") + label +
                                   " layer pitch is not constant");
        }
    }
    return pitch;
}

std::ptrdiff_t optional_layer_pitch(const std::vector<Tensor>& layers, const char* label) {
    return layers.empty() ? 0 : layer_pitch(layers, label);
}

} // namespace

CyclicKVCacheLayout plan_cyclic_kv_cache(LayoutBuilder& builder, std::uint32_t layers,
                                         std::uint32_t capacity, std::int32_t num_kv_heads,
                                         std::int32_t head_dim, std::int32_t lane_capacity,
                                         DType dtype, std::int32_t quant_group) {
    if (layers == 0 || capacity == 0 ||
        capacity > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        num_kv_heads <= 0 || head_dim <= 0 || lane_capacity <= 0) {
        throw std::invalid_argument("Cyclic KV geometry is invalid");
    }
    if (dtype != DType::BF16 && dtype != DType::U8) {
        throw std::invalid_argument("Cyclic KV supports BF16 or packed NVFP4 storage");
    }
    if ((dtype == DType::BF16 && quant_group != 0) ||
        (dtype == DType::U8 && (quant_group != 16 || head_dim % 2 != 0 ||
                                head_dim % quant_group != 0))) {
        throw std::invalid_argument("Cyclic KV quantization profile is invalid");
    }

    CyclicKVCacheLayout layout;
    layout.capacity        = capacity;
    layout.padded_capacity = align_up_u32(capacity, 128);
    layout.num_kv_heads    = num_kv_heads;
    layout.head_dim        = head_dim;
    layout.lane_capacity   = lane_capacity;
    layout.dtype           = dtype;
    layout.quant_group     = quant_group;
    layout.k.reserve(layers);
    layout.v.reserve(layers);
    if (dtype == DType::U8) {
        layout.k_scale.reserve(layers);
        layout.v_scale.reserve(layers);
    }
    const auto padded = static_cast<std::int32_t>(layout.padded_capacity);
    const auto code_head_dim = dtype == DType::U8 ? head_dim / 2 : head_dim;
    const auto scale_extent  = dtype == DType::U8 ? head_dim / quant_group : 0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "Cyclic KV layer " + std::to_string(layer);
        layout.k.push_back(builder.add_tensor(dtype,
                                              {code_head_dim, padded, num_kv_heads, lane_capacity},
                                              kArenaAlign, prefix + " K"));
        layout.v.push_back(builder.add_tensor(dtype,
                                              {code_head_dim, padded, num_kv_heads, lane_capacity},
                                              kArenaAlign, prefix + " V"));
        if (dtype == DType::U8) {
            layout.k_scale.push_back(builder.add_tensor(
                DType::FP8_E4M3FN, {scale_extent, padded, num_kv_heads, lane_capacity},
                kArenaAlign, prefix + " K scales"));
            layout.v_scale.push_back(builder.add_tensor(
                DType::FP8_E4M3FN, {scale_extent, padded, num_kv_heads, lane_capacity},
                kArenaAlign, prefix + " V scales"));
        }
    }
    return layout;
}

std::size_t CyclicKVCacheLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    for (const TensorRegion& region : k) { total += region.region.bytes; }
    for (const TensorRegion& region : v) { total += region.region.bytes; }
    for (const TensorRegion& region : k_scale) { total += region.region.bytes; }
    for (const TensorRegion& region : v_scale) { total += region.region.bytes; }
    return total;
}

CyclicKVCache::CyclicKVCache(DeviceSpan backing, const CyclicKVCacheLayout& layout)
    : capacity_(layout.capacity), padded_capacity_(layout.padded_capacity),
      num_kv_heads_(layout.num_kv_heads), head_dim_(layout.head_dim),
      lane_capacity_(layout.lane_capacity), dtype_(layout.dtype), quant_group_(layout.quant_group) {
    if (layout.k.empty() || layout.v.size() != layout.k.size() || capacity_ == 0 ||
        padded_capacity_ < capacity_ || num_kv_heads_ <= 0 || head_dim_ <= 0 ||
        lane_capacity_ <= 0) {
        throw std::invalid_argument("Cyclic KV layout is inconsistent");
    }
    if (dtype_ != DType::BF16 && dtype_ != DType::U8) {
        throw std::invalid_argument("Cyclic KV layout has an unsupported dtype");
    }
    if ((dtype_ == DType::BF16 && quant_group_ != 0) ||
        (dtype_ == DType::U8 && (quant_group_ != 16 || head_dim_ % 2 != 0 ||
                                 head_dim_ % quant_group_ != 0))) {
        throw std::invalid_argument("Cyclic KV layout has an invalid quantization profile");
    }
    if (dtype_ == DType::U8 &&
        (layout.k_scale.size() != layout.k.size() || layout.v_scale.size() != layout.v.size())) {
        throw std::invalid_argument("Cyclic KV NVFP4 layout is missing scale planes");
    }
    if (dtype_ == DType::BF16 && (!layout.k_scale.empty() || !layout.v_scale.empty())) {
        throw std::invalid_argument("Cyclic KV BF16 layout must not have scale planes");
    }
    code_head_dim_ = dtype_ == DType::U8 ? head_dim_ / 2 : head_dim_;
    scale_extent_  = dtype_ == DType::U8 ? head_dim_ / quant_group_ : 0;
    const std::array<std::int32_t, 4> expected_shape{
        code_head_dim_, static_cast<std::int32_t>(padded_capacity_), num_kv_heads_, lane_capacity_};
    const std::array<std::int32_t, 4> expected_scale_shape{
        scale_extent_, static_cast<std::int32_t>(padded_capacity_), num_kv_heads_, lane_capacity_};
    k_.reserve(layout.k.size());
    v_.reserve(layout.v.size());
    k_scale_.reserve(layout.k_scale.size());
    v_scale_.reserve(layout.v_scale.size());
    for (std::size_t layer = 0; layer < layout.k.size(); ++layer) {
        if (layout.k[layer].dtype != dtype_ || layout.v[layer].dtype != dtype_ ||
            layout.k[layer].shape != expected_shape || layout.v[layer].shape != expected_shape) {
            throw std::invalid_argument("Cyclic KV layer layout is inconsistent");
        }
        k_.push_back(layout.k[layer].bind(backing));
        v_.push_back(layout.v[layer].bind(backing));
        if (dtype_ == DType::U8) {
            if (layout.k_scale[layer].dtype != DType::FP8_E4M3FN ||
                layout.v_scale[layer].dtype != DType::FP8_E4M3FN ||
                layout.k_scale[layer].shape != expected_scale_shape ||
                layout.v_scale[layer].shape != expected_scale_shape) {
                throw std::invalid_argument("Cyclic KV scale layout is inconsistent");
            }
            k_scale_.push_back(layout.k_scale[layer].bind(backing));
            v_scale_.push_back(layout.v_scale[layer].bind(backing));
        }
    }
}

std::uint32_t CyclicKVCache::layer_count() const noexcept {
    return static_cast<std::uint32_t>(k_.size());
}

CyclicKVCacheLayerView CyclicKVCache::layer_view(std::uint32_t layer) const {
    if (layer >= layer_count()) { throw std::out_of_range("Cyclic KV layer is out of range"); }
    return {
        .k               = k_[layer],
        .v               = v_[layer],
        .capacity        = capacity_,
        .padded_capacity = padded_capacity_,
        .num_kv_heads    = num_kv_heads_,
        .head_dim        = head_dim_,
        .lane_capacity   = lane_capacity_,
        .k_scale         = dtype_ == DType::U8 ? k_scale_[layer] : Tensor{},
        .v_scale         = dtype_ == DType::U8 ? v_scale_[layer] : Tensor{},
        .dtype           = dtype_,
        .quant_group     = quant_group_,
    };
}

CyclicKVCacheSlotView CyclicKVCache::slot_view(std::int32_t slot) const {
    if (slot < 0 || slot >= lane_capacity_) {
        throw std::out_of_range("Cyclic KV slot is out of range");
    }
    const Tensor k =
        k_.front()
            .slice(3, slot, 1)
            .view({code_head_dim_, static_cast<std::int32_t>(padded_capacity_), num_kv_heads_});
    const Tensor v =
        v_.front()
            .slice(3, slot, 1)
            .view({code_head_dim_, static_cast<std::int32_t>(padded_capacity_), num_kv_heads_});
    Tensor k_scale;
    Tensor v_scale;
    if (dtype_ == DType::U8) {
        k_scale = k_scale_.front()
                      .slice(3, slot, 1)
                      .view({scale_extent_, static_cast<std::int32_t>(padded_capacity_),
                             num_kv_heads_});
        v_scale = v_scale_.front()
                      .slice(3, slot, 1)
                      .view({scale_extent_, static_cast<std::int32_t>(padded_capacity_),
                             num_kv_heads_});
    }
    return {
        .k_layer0            = k,
        .v_layer0            = v,
        .k_layer_bytes       = k.bytes(),
        .v_layer_bytes       = v.bytes(),
        .k_layer_pitch_bytes = layer_pitch(k_, "K"),
        .v_layer_pitch_bytes = layer_pitch(v_, "V"),
        .layers              = layer_count(),
        .k_scale_layer0            = k_scale,
        .v_scale_layer0            = v_scale,
        .k_scale_layer_bytes       = dtype_ == DType::U8 ? k_scale.bytes() : 0,
        .v_scale_layer_bytes       = dtype_ == DType::U8 ? v_scale.bytes() : 0,
        .k_scale_layer_pitch_bytes = optional_layer_pitch(k_scale_, "K scales"),
        .v_scale_layer_pitch_bytes = optional_layer_pitch(v_scale_, "V scales"),
    };
}

void CyclicKVCache::copy_slot_from(const CyclicKVCache& source, std::int32_t source_slot,
                                   std::int32_t destination_slot, cudaStream_t stream) {
    if (source.layer_count() != layer_count() || source.capacity_ != capacity_ ||
        source.padded_capacity_ != padded_capacity_ || source.num_kv_heads_ != num_kv_heads_ ||
        source.head_dim_ != head_dim_ || source.dtype_ != dtype_ ||
        source.quant_group_ != quant_group_) {
        throw std::invalid_argument("Cyclic KV copy requires identical component geometry");
    }
    if (source_slot < 0 || source_slot >= source.lane_capacity_) {
        throw std::out_of_range("Cyclic KV source slot is out of range");
    }
    if (destination_slot < 0 || destination_slot >= lane_capacity_) {
        throw std::out_of_range("Cyclic KV destination slot is out of range");
    }
    if (&source == this && source_slot == destination_slot) { return; }
    for (std::size_t layer = 0; layer < k_.size(); ++layer) {
        Tensor destination_k = k_[layer].slice(3, destination_slot, 1);
        Tensor destination_v = v_[layer].slice(3, destination_slot, 1);
        Tensor source_k      = source.k_[layer].slice(3, source_slot, 1);
        Tensor source_v      = source.v_[layer].slice(3, source_slot, 1);
        CUDA_CHECK(cudaMemcpyAsync(destination_k.data, source_k.data, destination_k.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(destination_v.data, source_v.data, destination_v.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        if (dtype_ == DType::U8) {
            Tensor destination_k_scale = k_scale_[layer].slice(3, destination_slot, 1);
            Tensor destination_v_scale = v_scale_[layer].slice(3, destination_slot, 1);
            Tensor source_k_scale      = source.k_scale_[layer].slice(3, source_slot, 1);
            Tensor source_v_scale      = source.v_scale_[layer].slice(3, source_slot, 1);
            CUDA_CHECK(cudaMemcpyAsync(destination_k_scale.data, source_k_scale.data,
                                       destination_k_scale.bytes(), cudaMemcpyDeviceToDevice,
                                       stream));
            CUDA_CHECK(cudaMemcpyAsync(destination_v_scale.data, source_v_scale.data,
                                       destination_v_scale.bytes(), cudaMemcpyDeviceToDevice,
                                       stream));
        }
    }
}

} // namespace ninfer
