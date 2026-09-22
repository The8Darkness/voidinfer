#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace ninfer::exl3 {
inline std::optional<std::size_t> pending_payload_extent(std::size_t tokens,
    const std::array<std::size_t,5>& taps,std::size_t wrapper=0) noexcept {
    auto total=wrapper;
    const auto add=[&](std::size_t count,std::size_t width) {
        if(count>(std::numeric_limits<std::size_t>::max()-total)/width)return false;
        total+=count*width;return true;
    };
    if(!add(tokens,sizeof(std::int64_t)))return std::nullopt;
    for(auto capacity:taps)if(!add(capacity,sizeof(std::uint16_t)))return std::nullopt;
    return total;
}
}
