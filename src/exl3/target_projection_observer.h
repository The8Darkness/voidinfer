#pragma once

#include "exl3/linear_cuda.h"

namespace ninfer::exl3 {

enum class Exl3TargetProjectionObserverSelection {
    gate_up_k6,
    gate_up_k7,
    prefill_gate_up_k6,
    down_k6,
    down_k7,
    output_k7,
    z_k6,
    wide_prefill_all,
    initial16_generic,
    m1_k6_n32,
};

struct Exl3TargetProjectionObservation {
    Exl3CudaLinearWeights weights{};
    Exl3CudaLinearMetadata metadata{};
    const std::uint16_t* input = nullptr;
    int rows = 0;
    int layer = -1;
    const char* operation = nullptr;
    const char* m1_dispatch = nullptr;
    cudaStream_t stream = nullptr;
};

using Exl3TargetProjectionObserver =
    void (*)(const Exl3TargetProjectionObservation&, void* user);

} // namespace ninfer::exl3
