#pragma once
#include <cstdint>

namespace voidinfer::exl3 {

// Per-request sampling policy of the coherent-device route (sampled_select.cuh).
// temperature 0 = greedy argmax; penalties are not part of this route.
struct Exl3DeviceSampling {
    float temperature=0.0f;
    std::int32_t top_k=0;
    float top_p=1.0f;
    float min_p=0.0f;
    std::uint64_t seed=0;
    bool active() const noexcept {return temperature>0.0f;}
};

} // namespace voidinfer::exl3
