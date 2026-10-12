#pragma once
// Persistent device allocations whose size is not a multiple of the 2 MiB
// physical page: cudaMalloc backs each one with whole pages (a 3 MiB GDN state
// occupies 4 MiB), the stream-ordered pool packs them. cudaFree releases pool
// allocations like ordinary ones. VOIDINFER_EXL3_PACKED_ALLOC=0 disables.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace voidinfer::exl3 {

inline cudaError_t exl3_packed_malloc(void** pointer, std::size_t bytes) {
    constexpr std::size_t kPage = 2u << 20;
    static const bool enabled = [] {
        const char* value = std::getenv("VOIDINFER_EXL3_PACKED_ALLOC");
        return !value || std::strcmp(value, "0") != 0;
    }();
    const std::size_t waste = (kPage - bytes % kPage) % kPage;
    if (!enabled || bytes <= kPage || waste < kPage / 4) return cudaMalloc(pointer, bytes);
    static cudaStream_t stream = [] {
        cudaStream_t created = nullptr;
        if (cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking) != cudaSuccess) created = nullptr;
        return created;
    }();
    if (!stream) return cudaMalloc(pointer, bytes);
    const cudaError_t error = cudaMallocAsync(pointer, bytes, stream);
    if (error != cudaSuccess) return error;
    return cudaStreamSynchronize(stream);
}

}  // namespace voidinfer::exl3
