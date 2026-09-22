#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::exl3 {
// Serialized trellis tile extent is unsigned 64-bit. Validate before narrowing:
// oversized values must never wrap to an otherwise supported K.
inline int exl3_native_bits_from_tile_extent(std::uint64_t extent) {
    if(extent<80 || extent>128 || extent%16)
        throw std::invalid_argument("unsupported native EXL3 trellis tile extent");
    return static_cast<int>(extent/16);
}
} // namespace ninfer::exl3
