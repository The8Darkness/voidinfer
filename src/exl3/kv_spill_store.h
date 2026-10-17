#pragma once
// Disk overflow for exact L0 KV page payloads (lane_kv_source.h). When host
// memory runs low, lane sources write idle pages' FP16 rows to one unbuffered
// temporary file in fixed 4 MiB slots (64 rows x 16 banks x K/V) instead of
// keeping them in RAM; a restore or reader brings them back.
//
// VOIDINFER_EXL3_KV_SPILL=0 disables. VOIDINFER_EXL3_KV_SPILL_DIR picks the
// directory (default: the temp directory). VOIDINFER_EXL3_KV_SPILL_FREE_MIB is
// the available physical memory to keep (default 12288).
// VOIDINFER_EXL3_KV_SPILL_DISK_RESERVE_MIB is the free space to leave on the
// spill volume (default 32768): spills that would cut into it stay in RAM, and
// a failed write pauses spilling for a minute (the pages stay resident).
#include "exl3/exact_kv_extent.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace voidinfer::exl3 {

class Exl3KvSpillStore {
public:
    static constexpr std::size_t kSlotBytes = static_cast<std::size_t>(Exl3ExactKVPage::token_capacity) * 1024 *
        sizeof(std::uint16_t) * 2 * 16;

    // The process store, or null when disabled or unavailable.
    static Exl3KvSpillStore* get() {
        static Exl3KvSpillStore* store = [] () -> Exl3KvSpillStore* {
            const char* value = std::getenv("VOIDINFER_EXL3_KV_SPILL");
            if (value && std::strcmp(value, "0") == 0) return nullptr;
            try {
                auto* created = new Exl3KvSpillStore();
                exl3_spill_release_hook().store(&release_hook, std::memory_order_release);
                return created;
            } catch (...) {
                return nullptr;
            }
        }();
        return store;
    }

    // Available physical memory below the configured floor (plus `extra`);
    // `hard`: below half the floor (plus `extra`).
    static bool under_pressure(std::uint64_t extra = 0, bool hard = false) {
        static const std::uint64_t floor = [] {
            const char* value = std::getenv("VOIDINFER_EXL3_KV_SPILL_FREE_MIB");
            return static_cast<std::uint64_t>(value ? std::strtoull(value, nullptr, 10) : 12288) << 20;
        }();
        MEMORYSTATUSEX memory{};
        memory.dwLength = sizeof(memory);
        if (!GlobalMemoryStatusEx(&memory)) return false;
        return memory.ullAvailPhys < (hard ? floor / 2 : floor) + extra;
    }

    // Room for `bytes` more of spill on the volume above the reserve, and no
    // write failure in the last minute.
    bool has_room(std::uint64_t bytes) const {
        static const std::uint64_t reserve = [] {
            const char* value = std::getenv("VOIDINFER_EXL3_KV_SPILL_DISK_RESERVE_MIB");
            return static_cast<std::uint64_t>(value ? std::strtoull(value, nullptr, 10) : 32768) << 20;
        }();
        const auto failed = failed_at_.load(std::memory_order_relaxed);
        if (failed && GetTickCount64() - failed < 60000) return false;
        ULARGE_INTEGER available{};
        if (!GetDiskFreeSpaceExW(directory_.c_str(), &available, nullptr, nullptr)) return false;
        return available.QuadPart >= reserve + bytes;
    }

    std::int64_t allocate() {
        std::lock_guard lock(mutex_);
        if (!free_.empty()) {
            const auto slot = free_.back();
            free_.pop_back();
            ++live_;
            return slot;
        }
        ++live_;
        return next_++;
    }
    void release(std::int64_t slot) noexcept {
        std::lock_guard lock(mutex_);
        free_.push_back(slot);
        --live_;
    }
    std::uint64_t live_slots() const {
        std::lock_guard lock(mutex_);
        return live_;
    }

    // `buffer` is an aligned buffer of kSlotBytes (aligned_buffer()); only
    // the first `bytes` (rounded up to the sector) are transferred.
    void write(std::int64_t slot, const void* buffer, std::size_t bytes) {
        io(slot, const_cast<void*>(buffer), bytes, true);
        const auto before = written_.fetch_add(bytes, std::memory_order_relaxed);
        if (std::getenv("VOIDINFER_EXL3_VERICACHE_STATS") && (before >> 30) != ((before + bytes) >> 30))
            std::fprintf(stderr, "KV_SPILL written_gib=%llu live_slots=%llu\n",
                static_cast<unsigned long long>((before + bytes) >> 30),
                static_cast<unsigned long long>(live_slots()));
        // VOIDINFER_EXL3_TEST_KV_SPILL=1 reads every slot back and compares.
        static const bool verify = [] {
            const char* value = std::getenv("VOIDINFER_EXL3_TEST_KV_SPILL");
            return value && std::strcmp(value, "1") == 0;
        }();
        if (verify) {
            void* check = VirtualAlloc(nullptr, kSlotBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!check) throw std::runtime_error("KV spill test buffer");
            io(slot, check, bytes, false);
            const bool same = std::memcmp(check, buffer, bytes) == 0;
            VirtualFree(check, 0, MEM_RELEASE);
            if (!same) throw std::logic_error("KV spill test mismatch");
        }
    }
    std::uint64_t written_bytes() const noexcept { return written_.load(std::memory_order_relaxed); }
    void read(std::int64_t slot, void* buffer, std::size_t bytes) { io(slot, buffer, bytes, false); }

    // A per-thread slot-sized buffer aligned for unbuffered I/O.
    static std::uint8_t* aligned_buffer() {
        struct Buffer {
            void* data = nullptr;
            ~Buffer() { if (data) VirtualFree(data, 0, MEM_RELEASE); }
        };
        thread_local Buffer buffer;
        if (!buffer.data) {
            buffer.data = VirtualAlloc(nullptr, kSlotBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!buffer.data) throw std::runtime_error("KV spill buffer");
        }
        return static_cast<std::uint8_t*>(buffer.data);
    }

private:
    Exl3KvSpillStore() {
        std::filesystem::path directory;
        if (const char* value = std::getenv("VOIDINFER_EXL3_KV_SPILL_DIR")) directory = value;
        else directory = std::filesystem::temp_directory_path();
        directory_ = std::filesystem::absolute(directory).wstring();
        const auto path = directory / ("voidinfer-kv-spill-" + std::to_string(GetCurrentProcessId()) + ".bin");
        file_ = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
            FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("KV spill file");
        sector_ = 4096;
    }
    static void release_hook(std::int64_t slot) noexcept {
        if (auto* store = get()) store->release(slot);
    }
    void io(std::int64_t slot, void* buffer, std::size_t bytes, bool write) {
        const std::size_t rounded = (bytes + sector_ - 1) / sector_ * sector_;
        if (rounded > kSlotBytes) throw std::logic_error("KV spill extent");
        OVERLAPPED overlapped{};
        const auto offset = static_cast<std::uint64_t>(slot) * kSlotBytes;
        overlapped.Offset = static_cast<DWORD>(offset);
        overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
        overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!overlapped.hEvent) throw std::runtime_error("KV spill event");
        DWORD done = 0;
        const BOOL started = write ? WriteFile(file_, buffer, static_cast<DWORD>(rounded), nullptr, &overlapped)
                                   : ReadFile(file_, buffer, static_cast<DWORD>(rounded), nullptr, &overlapped);
        const bool ok = (started || GetLastError() == ERROR_IO_PENDING) &&
            GetOverlappedResult(file_, &overlapped, &done, TRUE) && done == rounded;
        CloseHandle(overlapped.hEvent);
        if (!ok && write) failed_at_.store(GetTickCount64() | 1, std::memory_order_relaxed);
        if (!ok) throw std::runtime_error(write ? "KV spill write" : "KV spill read");
    }

    HANDLE file_ = INVALID_HANDLE_VALUE;
    std::wstring directory_;
    std::atomic<std::uint64_t> failed_at_{0};
    std::size_t sector_ = 4096;
    mutable std::mutex mutex_;
    std::vector<std::int64_t> free_;
    std::int64_t next_ = 0;
    std::uint64_t live_ = 0;
    std::atomic<std::uint64_t> written_{0};
};

}  // namespace voidinfer::exl3
