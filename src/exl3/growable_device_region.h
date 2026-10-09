#pragma once
// Device storage with a fixed virtual address range whose physical backing
// grows (and shrinks) in allocation-granularity chunks (CUDA virtual memory
// management). Kernels and captured graphs keep one contiguous pointer;
// physical memory follows the bytes actually in use, so per-context buffers
// sized for max_context cost only what the context occupies (the paged-pool
// effect without block tables).
#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3 {

class Exl3GrowableDeviceRegion {
public:
    Exl3GrowableDeviceRegion() = default;
    Exl3GrowableDeviceRegion(const Exl3GrowableDeviceRegion&) = delete;
    Exl3GrowableDeviceRegion& operator=(const Exl3GrowableDeviceRegion&) = delete;
    ~Exl3GrowableDeviceRegion() { reset(); }

    // Reserves `capacity` bytes of virtual address space on the current device.
    // host: the backing is pinned host memory (NUMA node of the device) that
    // only the GPU addresses, through the same contiguous range; the CPU
    // reaches it by copies.
    void reserve(std::size_t capacity, bool host = false) {
        if (base_) throw std::logic_error("growable region already reserved");
        CUdevice device = 0;
        check(cuCtxGetDevice(&device), "growable region device");
        prop_ = {};
        prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        if (host) {
            int numa = -1;
            if (cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device) != CUDA_SUCCESS)
                numa = -1;
            prop_.location.type = numa >= 0 ? CU_MEM_LOCATION_TYPE_HOST_NUMA : CU_MEM_LOCATION_TYPE_HOST;
            prop_.location.id = numa >= 0 ? numa : 0;
        } else {
            prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop_.location.id = static_cast<int>(device);
        }
        check(cuMemGetAllocationGranularity(&granularity_, &prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "growable region granularity");
        capacity_ = round_up(std::max<std::size_t>(capacity, 1));
        check(cuMemAddressReserve(&base_, capacity_, 0, 0, 0), "growable region reserve");
        access_ = {};
        access_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_.location.id = static_cast<int>(device);
        access_.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    // Backs at least `bytes` from the start of the range (host call; no
    // stream ordering is needed to grow: new pages are not yet referenced).
    void ensure(std::size_t bytes) {
        if (!base_) throw std::logic_error("growable region not reserved");
        bytes = std::min(round_up(bytes), capacity_);
        while (mapped_ < bytes) {
            CUmemGenericAllocationHandle handle = 0;
            check(cuMemCreate(&handle, granularity_, &prop_, 0), "growable region create");
            const auto address = base_ + mapped_;
            if (cuMemMap(address, granularity_, 0, handle, 0) != CUDA_SUCCESS) {
                cuMemRelease(handle);
                throw std::runtime_error("growable region map");
            }
            check(cuMemSetAccess(address, granularity_, &access_, 1), "growable region access");
            handles_.push_back(handle);
            mapped_ += granularity_;
        }
    }

    // Releases backing beyond `bytes`. The caller guarantees no in-flight
    // work references the released range.
    void shrink(std::size_t bytes) {
        if (!base_) return;
        bytes = std::min(round_up(bytes), capacity_);
        while (mapped_ > bytes && !handles_.empty()) {
            mapped_ -= granularity_;
            cuMemUnmap(base_ + mapped_, granularity_);
            cuMemRelease(handles_.back());
            handles_.pop_back();
        }
    }

    void reset() noexcept {
        if (!base_) return;
        while (!handles_.empty()) {
            mapped_ -= granularity_;
            cuMemUnmap(base_ + mapped_, granularity_);
            cuMemRelease(handles_.back());
            handles_.pop_back();
        }
        cuMemAddressFree(base_, capacity_);
        base_ = 0;
        mapped_ = 0;
    }

    template <class T> T* data() const noexcept { return reinterpret_cast<T*>(base_); }
    std::size_t mapped_bytes() const noexcept { return mapped_; }
    std::size_t capacity_bytes() const noexcept { return capacity_; }

private:
    static void check(CUresult result, const char* what) {
        if (result != CUDA_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed (CUresult " +
                                     std::to_string(static_cast<int>(result)) + ")");
    }
    std::size_t round_up(std::size_t bytes) const noexcept {
        return granularity_ ? (bytes + granularity_ - 1) / granularity_ * granularity_ : bytes;
    }
    CUdeviceptr base_ = 0;
    std::size_t capacity_ = 0, mapped_ = 0, granularity_ = 0;
    CUmemAllocationProp prop_{};
    CUmemAccessDesc access_{};
    std::vector<CUmemGenericAllocationHandle> handles_;
};

}  // namespace ninfer::exl3
