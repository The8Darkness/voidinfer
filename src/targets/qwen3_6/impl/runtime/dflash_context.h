#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include "core/cyclic_kv_cache.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

struct DFlashPersistentState {
    CyclicKVCache& local;
    // Independently written OSCAR-Q4 source shadow for the live L1 verifier and pinned host
    // mirror. It is optional so legacy/non-hierarchical DFlash keeps its original footprint.
    CyclicKVCache* local_q4_shadow = nullptr;
    // Prefer the Q4 shadow as the live DFlash proposal when the hierarchy is serving. Q2 stays
    // resident for fallback and can be re-enabled as a two-pass comparison by the plan.
    bool q4_primary = false;
    qwen3_6::PagedKVCache full;
    Tensor prefill_features;
    Tensor prefill_positions;
    Tensor pending_features;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout,
                          CyclicKVCache& local_state,
                          CyclicKVCache* local_q4_state = nullptr);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] CyclicKVCacheLayerView local_q4_layer(std::uint32_t layer) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
    void save_rewrite_checkpoint(std::int32_t source_slot, std::int32_t destination_slot,
                                 cudaStream_t stream);
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
