#pragma once

#include <cstdint>
#include <vector>

namespace ninfer::exl3 {

// One copy of a verifier row's per-row state when an accepted sibling leaf is
// promoted into its chain slot: dst[i * stride] = src[i * stride] for
// i < count 16-bit elements. slot_elements > 0 marks a K/V cache row whose
// slot is relative to the attempt base (device position - attempted + 1).
struct Exl3SiblingRowCopy {
    const std::uint16_t* src = nullptr;
    std::uint16_t* dst = nullptr;
    int count = 0;
    int stride = 1;
    int slot_elements = 0;
};
using Exl3SiblingRowCopies = std::vector<Exl3SiblingRowCopy>;

} // namespace ninfer::exl3
