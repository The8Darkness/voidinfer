#include "targets/qwen3_6/impl/runtime/dflash_context.h"

#include "ops/kv_cache/d256_profile.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

DFlashPersistentState::DFlashPersistentState(DeviceSpan backing,
                                             const DFlashPersistentLayout& layout,
                                             CyclicKVCache& local_state,
                                             DeviceContext& device, bool enable_phase_timing,
                                             CyclicKVCache* local_q4_state)
    : local(local_state), local_q4_shadow(local_q4_state), full(backing, layout.full),
      prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)),
      phase_timing(device, enable_phase_timing), adaptive_replay_backing(backing),
      adaptive_replay_layout(layout.adaptive_replay_records) {
    const bool full_oscar = layout.full.dtype == DType::U8 &&
                            layout.full.quant_group == ops::kD256OscarQuantGroup;
    const std::size_t expected_full_planes = full_oscar ? 4U : 2U;
    if (local.layer_count() != DFlashConfig::local_layers ||
        local.capacity() != DFlashConfig::local_capacity || full.layers() != 1 ||
        full.max_context() != layout.full.max_context ||
        full.page_pool().plane_count() != expected_full_planes ||
        local.num_kv_heads() != DFlashConfig::kv_heads ||
        local.head_dim() != DFlashConfig::head_dim ||
        (full_oscar ? (full.page_pool().plane(0).dtype != DType::U8 ||
                       full.page_pool().plane(0).ne[0] != (DFlashConfig::head_dim * 2 + 7) / 8 ||
                       full.page_pool().plane(2).dtype != DType::BF16 ||
                       full.page_pool().plane(2).ne[0] != 2)
                    : full.page_pool().plane(0).dtype != DType::BF16) ||
        (!full_oscar && full.page_pool().plane(0).ne[0] != DFlashConfig::head_dim) ||
        full.page_pool().plane(0).ne[1] != kPagedKVPageSize ||
        full.page_pool().plane(0).ne[3] != DFlashConfig::kv_heads) {
        throw std::invalid_argument("DFlash persistent cache layout is invalid");
    }
    if (local_q4_shadow != nullptr &&
        (local_q4_shadow->layer_count() != local.layer_count() ||
         local_q4_shadow->capacity() != local.capacity() ||
         local_q4_shadow->lane_capacity() != local.lane_capacity() ||
         local_q4_shadow->num_kv_heads() != local.num_kv_heads() ||
         local_q4_shadow->head_dim() != local.head_dim() ||
         local_q4_shadow->dtype() != DType::U8 ||
         local_q4_shadow->quantization() != CyclicKVCacheQuantization::OscarAffine ||
         local_q4_shadow->quant_bits() != 4)) {
        throw std::invalid_argument("DFlash OSCAR-Q4 shadow layout is invalid");
    }
    if constexpr (DFlashConfig::is_v2) {
        if (local.dtype() != DType::BF16 || local.quant_group() != 0 ||
            local.quant_bits() != 0 ||
            local.quantization() != CyclicKVCacheQuantization::Auto ||
            local_q4_shadow != nullptr) {
            throw std::invalid_argument("DFlash2 local drafter cache must be BF16");
        }
    }
}

GdnReplayRecords DFlashPersistentState::make_adaptive_replay_records(std::int32_t width) const {
    if (!adaptive_replay_layout.has_value()) {
        throw std::logic_error("Adaptive DFlash2 ReplaySSM storage is unavailable");
    }
    if (width < 2 || width > adaptive_replay_layout->spec.width) {
        throw std::invalid_argument("Adaptive DFlash2 ReplaySSM width is out of range");
    }

    GdnReplayRecordLayout layout = *adaptive_replay_layout;
    layout.spec.width             = width;
    const std::int32_t outer      = layout.spec.layers * layout.spec.record_capacity;
    const auto resize = [outer](TensorRegion& region, int width_dimension, int outer_dimension,
                                std::int32_t width_value) {
        region.shape[width_dimension] = width_value;
        region.shape[outer_dimension] = outer;
        region.region.bytes = Tensor(nullptr, region.dtype,
                                     {region.shape[0], region.shape[1], region.shape[2],
                                      region.shape[3]})
                                  .bytes();
    };
    resize(layout.conv, 1, 2, width);
    resize(layout.key, 2, 3, width);
    resize(layout.value, 2, 3, width);
    resize(layout.gate, 2, 3, width);
    return GdnReplayRecords(adaptive_replay_backing, layout);
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

CyclicKVCacheLayerView DFlashPersistentState::local_q4_layer(std::uint32_t layer) const {
    if (local_q4_shadow == nullptr) {
        throw std::logic_error("DFlash OSCAR-Q4 shadow is unavailable");
    }
    return local_q4_shadow->layer_view(layer);
}

PagedKVBatchLayerView DFlashPersistentState::full_batch_layer(std::uint32_t layer) const {
    return full.batch_layer_view(layer);
}

void DFlashPersistentState::save_rewrite_checkpoint(std::int32_t source_slot,
                                                    std::int32_t destination_slot,
                                                    cudaStream_t stream) {
    local.copy_slot_from(local, source_slot, destination_slot, stream);
    if (local_q4_shadow != nullptr) {
        local_q4_shadow->copy_slot_from(*local_q4_shadow, source_slot, destination_slot, stream);
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
