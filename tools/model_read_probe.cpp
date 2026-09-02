#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kAlignment = 4096;
constexpr std::size_t kDefaultMiB = 64;

[[noreturn]] void fail(const char* operation) {
    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
}

struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    Handle() = default;
    ~Handle() {
        if (value != INVALID_HANDLE_VALUE) { CloseHandle(value); }
    }
    Handle(const Handle&)            = delete;
    Handle& operator=(const Handle&) = delete;
};

std::uint64_t file_size(HANDLE file) {
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file, &size) || size.QuadPart < 0) { fail("GetFileSizeEx"); }
    return static_cast<std::uint64_t>(size.QuadPart);
}

double seconds_since(const Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program << " <file> [buffer_mib]\n";
}

} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc < 2 || argc > 3) {
            print_usage("model_read_probe.exe");
            return 2;
        }
        const std::filesystem::path path(argv[1]);
        std::size_t buffer_mib = kDefaultMiB;
        if (argc == 3) {
            const auto parsed = std::stoull(argv[2]);
            if (parsed == 0 || parsed > (std::numeric_limits<std::size_t>::max() / (1ULL << 20))) {
                throw std::invalid_argument("buffer_mib is out of range");
            }
            buffer_mib = static_cast<std::size_t>(parsed);
        }
        const std::uint64_t requested = static_cast<std::uint64_t>(buffer_mib) << 20U;
        if (requested % kAlignment != 0 || requested > std::numeric_limits<DWORD>::max()) {
            throw std::invalid_argument("buffer_mib must produce a 4096-aligned DWORD-sized buffer");
        }

        Handle file;
        file.value = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                     FILE_FLAG_SEQUENTIAL_SCAN,
                                 nullptr);
        if (file.value == INVALID_HANDLE_VALUE) { fail("CreateFileW"); }
        const std::uint64_t bytes = file_size(file.value);
        if (bytes == 0) { throw std::runtime_error("file must be nonempty for this probe"); }
        const std::uint64_t aligned_bytes = bytes / kAlignment * kAlignment;
        if (aligned_bytes == 0) { throw std::runtime_error("file is smaller than one I/O sector"); }

        void* buffer = VirtualAlloc(nullptr, static_cast<SIZE_T>(requested), MEM_COMMIT | MEM_RESERVE,
                                    PAGE_READWRITE);
        if (buffer == nullptr) { fail("VirtualAlloc"); }
        const auto release_buffer = [&] { VirtualFree(buffer, 0, MEM_RELEASE); };

        std::uint64_t total_read = 0;
        std::uint64_t requests  = 0;
        const auto start        = Clock::now();
        while (total_read < aligned_bytes) {
            const std::uint64_t remaining = aligned_bytes - total_read;
            const DWORD amount = static_cast<DWORD>(std::min<std::uint64_t>(requested, remaining));
            DWORD received     = 0;
            if (!ReadFile(file.value, buffer, amount, &received, nullptr)) {
                const auto error = GetLastError();
                release_buffer();
                SetLastError(error);
                fail("ReadFile");
            }
            if (received == 0 || received != amount) {
                release_buffer();
                throw std::runtime_error("short sequential read");
            }
            total_read += received;
            ++requests;
        }
        const double seconds = seconds_since(start);
        release_buffer();

        const double decimal_gb_s = static_cast<double>(total_read) / seconds / 1.0e9;
        const double mib_s        = static_cast<double>(total_read) / seconds / (1ULL << 20);
        std::cout << std::fixed << std::setprecision(3)
                  << "model_read_probe path=" << path.string() << " bytes=" << total_read
                  << " file_bytes=" << bytes << " tail_bytes_skipped=" << (bytes - aligned_bytes)
                  << " request_bytes=" << requested << " requests=" << requests
                  << " elapsed_s=" << seconds << " throughput_MB_s=" << decimal_gb_s * 1000.0
                  << " throughput_MiB_s=" << mib_s << " no_buffering=true sequential_scan=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
