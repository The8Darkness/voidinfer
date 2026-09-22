#pragma once

#include <cstdint>

namespace ninfer {

// The byte layout is part of the OSCAR cache contract. Writers, metadata
// copies, prompt attention, and decode attention must receive the same value;
// they must not independently infer it from process-global environment state.
enum class OscarKVLayout : std::uint8_t {
    Contiguous = 0,
    TransposedQ2 = 1,
};

[[nodiscard]] constexpr bool is_valid_oscar_kv_layout(OscarKVLayout layout) noexcept {
    return layout == OscarKVLayout::Contiguous || layout == OscarKVLayout::TransposedQ2;
}

} // namespace ninfer
