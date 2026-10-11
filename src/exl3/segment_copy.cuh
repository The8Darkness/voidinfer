#pragma once
// One launch for several device-to-device copies (batched multi-agent peer
// segments). Inside the captured batched forward each cudaMemcpyAsync is its
// own graph copy node with microseconds of latency, and a peer core waited for
// eight of them in sequence before its first kernel.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <cstdint>
#include <stdexcept>

namespace ninfer::exl3 {

inline constexpr int kSegmentCopyMax = 8;

struct Exl3SegmentCopies {
    const void* source[kSegmentCopyMax]{};
    void* destination[kSegmentCopyMax]{};
    std::size_t bytes[kSegmentCopyMax]{};
    int count = 0;
    void add(void* to, const void* from, std::size_t size) {
        if (count == kSegmentCopyMax) throw std::logic_error("segment copy capacity");
        source[count] = from; destination[count] = to; bytes[count] = size; ++count;
    }
};

// blockIdx.y selects the copy; 16-byte chunks when both ends and the size
// allow it, bytes otherwise.
static __global__ void __launch_bounds__(256) exl3_segment_copy_kernel(Exl3SegmentCopies copies) {
    const int index = static_cast<int>(blockIdx.y);
    if (index >= copies.count) return;
    const auto* from = static_cast<const unsigned char*>(copies.source[index]);
    auto* to = static_cast<unsigned char*>(copies.destination[index]);
    const std::size_t bytes = copies.bytes[index];
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (((reinterpret_cast<std::uintptr_t>(from) | reinterpret_cast<std::uintptr_t>(to) | bytes) & 15) == 0) {
        const auto* src = reinterpret_cast<const uint4*>(from);
        auto* dst = reinterpret_cast<uint4*>(to);
        for (std::size_t i = first; i < bytes / 16; i += stride) dst[i] = src[i];
    } else {
        for (std::size_t i = first; i < bytes; i += stride) to[i] = from[i];
    }
}

inline cudaError_t exl3_launch_segment_copies(const Exl3SegmentCopies& copies, cudaStream_t stream) {
    if (!copies.count) return cudaSuccess;
    exl3_segment_copy_kernel<<<dim3(32, copies.count), 256, 0, stream>>>(copies);
    return cudaGetLastError();
}

// A copy on the SMs instead of a copy engine (exl3_d2h_async: decode
// readbacks must not queue behind the asynchronous VeriCache line's
// host->device history stream). Both ends must be device-accessible (device
// memory or mapped pinned host memory). NINFER_EXL3_KERNEL_COPIES=0 keeps
// cudaMemcpyAsync.
inline bool exl3_kernel_copies_enabled() {
    static const bool value = [] {
        const char* v = std::getenv("NINFER_EXL3_KERNEL_COPIES");
        return !v || std::string_view(v) != "0";
    }();
    return value;
}
inline cudaError_t exl3_copy_async(void* to, const void* from, std::size_t bytes, cudaStream_t stream) {
    if (!bytes) return cudaSuccess;
    if (!exl3_kernel_copies_enabled()) return cudaMemcpyAsync(to, from, bytes, cudaMemcpyDefault, stream);
    Exl3SegmentCopies copy;
    copy.add(to, from, bytes);
    const std::size_t blocks = (bytes + 16 * 256 * 4 - 1) / (16 * 256 * 4);
    exl3_segment_copy_kernel<<<dim3(static_cast<unsigned>(blocks < 128 ? blocks : 128), 1), 256, 0, stream>>>(copy);
    return cudaGetLastError();
}

} // namespace ninfer::exl3
