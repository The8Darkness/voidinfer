#pragma once
// Host memory a new allocation can still use: the smaller of available
// physical memory and the remaining commit charge. Available physical memory
// counts the standby cache, so it stays high while the commit charge runs out;
// then pinned allocations, page locking and plain allocations fail with tens
// of GiB still "available" (long multi-lane histories at a 262K context).
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <optional>

namespace voidinfer::exl3 {

inline std::optional<std::uint64_t> exl3_available_host_bytes() noexcept {
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (!GlobalMemoryStatusEx(&memory)) return std::nullopt;
    return std::min<std::uint64_t>(memory.ullAvailPhys, memory.ullAvailPageFile);
}

}  // namespace voidinfer::exl3
