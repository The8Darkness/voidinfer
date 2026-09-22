#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <vector>

namespace ninfer::exl3 {

enum class Exl3TargetProjectionPhase { attempt, replay };

enum class Exl3TargetProjectionOperator {
    q,
    k,
    v,
    o,
    gate,
    up,
    down,
    qkv,
    z,
    lm_head,
};

enum class Exl3TargetProjectionTopology {
    m1,
    m1_per_row,
    batched,
    small_m_single_split,
    small_m_mma_split,
};

struct Exl3TargetProjectionTimingRecord {
    int round = 0;
    Exl3TargetProjectionPhase phase = Exl3TargetProjectionPhase::attempt;
    int layer = -1;
    Exl3TargetProjectionOperator operation = Exl3TargetProjectionOperator::q;
    int rows = 0;
    int K = 0;
    int in_features = 0;
    int out_features = 0;
    Exl3TargetProjectionTopology topology = Exl3TargetProjectionTopology::m1;
    int calls = 0;
    double microseconds = 0.0;
};

// Diagnostic-only event owner.  401 projection groups cover one complete
// 64-layer continuation attempt (including the head); 8 serial replay rows
// need another 3208 groups.  Events are created only by explicit preparation.
class Exl3TargetProjectionTiming {
public:
    static constexpr int kGroupsPerPass = 401;
    static constexpr int kMaxSlots = kGroupsPerPass * 9;

    Exl3TargetProjectionTiming();
    ~Exl3TargetProjectionTiming();
    Exl3TargetProjectionTiming(const Exl3TargetProjectionTiming&) = delete;
    Exl3TargetProjectionTiming& operator=(const Exl3TargetProjectionTiming&) = delete;

    void begin_round(int round, cudaStream_t stream = nullptr);
    void set_phase(Exl3TargetProjectionPhase phase) noexcept { phase_ = phase; }
    int begin(int layer, Exl3TargetProjectionOperator operation, int rows, int K,
              int in_features, int out_features,
              Exl3TargetProjectionTopology topology, int calls,
              cudaStream_t stream);
    void end(int slot, cudaStream_t stream);
    void validate_execution_stream(cudaStream_t stream) const;
    bool active() const noexcept { return active_; }
    // The caller synchronizes the round before resolving; this method never
    // synchronizes and performs all vector allocation after the timed region.
    std::vector<Exl3TargetProjectionTimingRecord> finish_after_synchronize();

private:
    struct Slot {
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        Exl3TargetProjectionTimingRecord metadata{};
        bool ended = false;
    };

    std::array<Slot, kMaxSlots> slots_{};
    int round_ = 0;
    int used_ = 0;
    Exl3TargetProjectionPhase phase_ = Exl3TargetProjectionPhase::attempt;
    cudaStream_t stream_ = nullptr;
    bool active_ = false;
};

const char* target_projection_phase_name(Exl3TargetProjectionPhase phase) noexcept;
const char* target_projection_operator_name(Exl3TargetProjectionOperator operation) noexcept;
const char* target_projection_topology_name(Exl3TargetProjectionTopology topology) noexcept;

} // namespace ninfer::exl3
