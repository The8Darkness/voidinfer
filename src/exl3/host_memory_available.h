#pragma once
// Host memory a new allocation can still use: the smaller of available
// physical memory and the remaining commit charge. Available physical memory
// counts the standby cache, so it stays high while the commit charge runs out;
// then pinned allocations, page locking and plain allocations fail with tens
// of GiB still "available" (long multi-lane histories at a 262K context).
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>

namespace voidinfer::exl3 {

// Host memory every EXL3 allocation gate leaves free (resident set, prefix
// cache, recurrent pool): requests are refused below it.
inline constexpr std::uint64_t kExl3HostReserveBytes = 8ULL << 30;
// Below this, retained histories are relieved on the requesting path
// (lane_kv_source.h) and pooled host chunks are not kept.
inline constexpr std::uint64_t kExl3HostReliefBytes = kExl3HostReserveBytes + (2ULL << 30);

// The prefix cache's eviction under host memory pressure, set by the Engine
// and run by the lane sources' relief (lane_kv_source.h) once spilling cannot
// make room: evicts retained roots until `extra` more bytes fit above the
// relief floor. Calls hold the mutex, so clearing it waits for a running
// eviction.
struct Exl3HostMemoryEviction {
    std::mutex mutex;
    std::function<void(std::uint64_t)> evict;
    static Exl3HostMemoryEviction& instance() {
        static Exl3HostMemoryEviction value;
        return value;
    }
    static void set(std::function<void(std::uint64_t)> evict) {
        auto& self = instance();
        std::lock_guard lock(self.mutex);
        self.evict = std::move(evict);
    }
};

// Idle L0 plane sets give back the rows no deferred page names any more
// (registered by lane_kv_source.h): run after evicted roots are released.
inline std::atomic<void (*)()> exl3_release_idle_plane_rows{nullptr};

inline std::optional<std::uint64_t> exl3_available_host_bytes() noexcept {
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (!GlobalMemoryStatusEx(&memory)) return std::nullopt;
    return std::min<std::uint64_t>(memory.ullAvailPhys, memory.ullAvailPageFile);
}

// Fewer than `extra` bytes are available above the relief floor.
inline bool exl3_host_memory_short(std::uint64_t extra = 0) noexcept {
    const auto available = exl3_available_host_bytes();
    return available && *available < kExl3HostReliefBytes + extra;
}

}  // namespace voidinfer::exl3
