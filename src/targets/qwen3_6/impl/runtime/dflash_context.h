#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"

#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "ninfer/ops/gdn_replay.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

// Phase timers are opt-in because each CUDA event adds a graph/eager stream node. They are used
// by the baseline harness to separate proposal, selector, and exact-target verifier cost without
// synchronizing between phases.
struct DFlashPhaseTiming {
    struct Durations {
        float draft_ms    = 0.0F;
        float selector_ms = 0.0F;
        float verifier_ms = 0.0F;
    };

    bool enabled = false;
    std::optional<CudaEventTimer> draft;
    std::optional<CudaEventTimer> selector;
    std::optional<CudaEventTimer> verifier;

    DFlashPhaseTiming(DeviceContext& device, bool enable) : enabled(enable) {
        if (!enabled) { return; }
        draft.emplace(device);
        selector.emplace(device);
        verifier.emplace(device);
    }

    void begin_round() {
        if (enabled) { draft->start(); }
    }
    void begin_selector() {
        if (!enabled) { return; }
        draft->record_stop();
        selector->start();
    }
    void end_selector() {
        if (enabled) { selector->record_stop(); }
    }
    void begin_verifier() {
        if (enabled) { verifier->start(); }
    }
    void end_verifier() {
        if (enabled) { verifier->record_stop(); }
    }

    [[nodiscard]] Durations durations() const {
        if (!enabled) { return {}; }
        return Durations{.draft_ms = draft->elapsed_ms(),
                         .selector_ms = selector->elapsed_ms(),
                         .verifier_ms = verifier->elapsed_ms()};
    }
};

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
    DFlashPhaseTiming phase_timing;
    DeviceSpan adaptive_replay_backing;
    std::optional<GdnReplayRecordLayout> adaptive_replay_layout;
    GdnReplayRecords adaptive_replay_records;
    std::optional<ops::GdnReplayFoldPlan> adaptive_replay_fold;

    DFlashPersistentState(DeviceSpan backing, const DFlashPersistentLayout& layout,
                          CyclicKVCache& local_state, DeviceContext& device,
                          bool enable_phase_timing, CyclicKVCache* local_q4_state = nullptr);

    [[nodiscard]] CyclicKVCacheLayerView local_layer(std::uint32_t layer) const;
    [[nodiscard]] CyclicKVCacheLayerView local_q4_layer(std::uint32_t layer) const;
    [[nodiscard]] PagedKVBatchLayerView full_batch_layer(std::uint32_t layer) const;
    [[nodiscard]] GdnReplayRecords make_adaptive_replay_records(std::int32_t width) const;
    void save_rewrite_checkpoint(std::int32_t source_slot, std::int32_t destination_slot,
                                 cudaStream_t stream);
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
