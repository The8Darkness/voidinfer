#pragma once

#include "ninfer/types.h"

#include <cstddef>
#include <memory>
#include <span>

namespace ninfer {

struct DeviceContext;

// Startup-frozen device storage for the single active prefill owner. Activation only selects a
// prefix of this allocation; it never allocates or replaces device memory at request time.
class RequestTransientArena {
public:
    static constexpr std::size_t kDeviceAllocationAlignment = 256;

    struct Region {
        std::byte* data       = nullptr;
        std::size_t size      = 0;
        std::size_t alignment = 1;

        [[nodiscard]] explicit operator bool() const noexcept { return data != nullptr; }

        [[nodiscard]] std::span<std::byte> bytes() const noexcept { return {data, size}; }
    };

    RequestTransientArena(DeviceContext& device, std::size_t frozen_capacity_bytes);
    ~RequestTransientArena();

    RequestTransientArena(const RequestTransientArena&)            = delete;
    RequestTransientArena& operator=(const RequestTransientArena&) = delete;
    RequestTransientArena(RequestTransientArena&&)                 = delete;
    RequestTransientArena& operator=(RequestTransientArena&&)      = delete;

    void activate(std::size_t bytes, std::size_t alignment);
    void deactivate() noexcept;

    [[nodiscard]] Region region() const noexcept;
    [[nodiscard]] ArenaMemorySummary summary() const noexcept;
    void reset_peak() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
