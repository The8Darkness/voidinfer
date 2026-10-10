#pragma once
// Graph capture vs device-wide synchronization across Engine lanes. Captures
// are thread-local, but cudaDeviceSynchronize fails ("operation not permitted
// when stream is capturing") and invalidates the capture whenever any stream
// on the device is capturing, including another lane's. Captures hold the
// guard shared from begin to end; device-wide synchronization takes it
// exclusively and so waits for running captures (never call it while
// capturing: that would be illegal anyway).
#include <cuda_runtime.h>

#include <mutex>
#include <shared_mutex>

namespace ninfer {

inline std::shared_mutex& cuda_capture_guard() {
    static std::shared_mutex mutex;
    return mutex;
}

class CudaCaptureScope {
public:
    CudaCaptureScope() : lock_(cuda_capture_guard()) {}

private:
    std::shared_lock<std::shared_mutex> lock_;
};

inline cudaError_t cuda_device_synchronize_outside_capture() {
    const std::unique_lock lock(cuda_capture_guard());
    return cudaDeviceSynchronize();
}

}  // namespace ninfer
