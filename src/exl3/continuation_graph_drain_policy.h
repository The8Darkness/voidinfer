#pragma once

#include <cstdint>
#include <memory>
#include <span>

namespace ninfer::exl3 {

// Pure admission decision for the one default-off graph final-use narrowing.
// Native synchronization remains in TextContext; this value never certifies
// completion and cannot widen ownership.
class Exl3ContinuationGraphDrainPolicy {
public:
    enum class Mode : std::uint8_t { device,owned_stream };
    struct Pending {
        bool active=false;
        std::uintptr_t stream=0;
    };
    static Mode select(bool requested,
        const std::shared_ptr<const void>& stream_owner,
        std::uintptr_t owned_stream,std::span<const Pending> entries) noexcept {
        if(!requested || !stream_owner || !owned_stream)return Mode::device;
        bool found=false;
        for(const auto& entry:entries)if(entry.active) {
            if(!entry.stream || entry.stream!=owned_stream)return Mode::device;
            found=true;
        }
        return found?Mode::owned_stream:Mode::device;
    }
};

} // namespace ninfer::exl3
