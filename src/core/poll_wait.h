#pragma once
// Waits that stay out of the CUDA driver while the GPU works. A thread blocked
// inside a synchronizing call (cudaStreamSynchronize, a pageable
// cudaMemcpyAsync, cudaEventSynchronize) holds the driver's context lock on
// Windows/WDDM until the work completes, so every CUDA call of every other
// thread stalls behind it (~one decode round per call while the asynchronous
// VeriCache line enqueues its pass). Polling the stream or event first leaves
// the following synchronizing call nothing to wait for.
//
// Enabled by the Engine where two host threads drive one device concurrently
// (the asynchronous VeriCache line); VOIDINFER_CUDA_POLL_WAITS=0|1 forces it.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>

namespace voidinfer {

inline std::atomic<bool>& cuda_poll_waits_requested() {
    static std::atomic<bool> value{false};
    return value;
}
inline void cuda_set_poll_waits(bool enabled) noexcept { cuda_poll_waits_requested().store(enabled); }
inline bool cuda_poll_waits_enabled() {
    static const int forced = [] {
        const char* value = std::getenv("VOIDINFER_CUDA_POLL_WAITS");
        return value ? (std::string_view(value) != "0" ? 1 : 0) : -1;
    }();
    return forced >= 0 ? forced == 1 : cuda_poll_waits_requested().load(std::memory_order_relaxed);
}

namespace detail {
inline void cuda_poll_pause() {
    // ~10 us busy pause: OS sleeps are ~1 ms granular on Windows.
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(10);
    while (std::chrono::steady_clock::now() < until) {
    }
}
}  // namespace detail

// Returns once `stream` has drained (or reports its error); no-op when disabled.
inline cudaError_t cuda_poll_stream(cudaStream_t stream) {
    if (!cuda_poll_waits_enabled()) return cudaSuccess;
    for (;;) {
        const cudaError_t error = cudaStreamQuery(stream);
        if (error != cudaErrorNotReady) return error;
        detail::cuda_poll_pause();
    }
}

inline cudaError_t cuda_poll_event(cudaEvent_t event) {
    if (!cuda_poll_waits_enabled()) return cudaEventSynchronize(event);
    for (;;) {
        const cudaError_t error = cudaEventQuery(event);
        if (error != cudaErrorNotReady) return error;
        detail::cuda_poll_pause();
    }
}

}  // namespace voidinfer
