#pragma once
// Device storage with a fixed virtual address range whose physical backing
// grows (and shrinks) in allocation-granularity chunks (CUDA virtual memory
// management). Kernels and captured graphs keep one contiguous pointer;
// physical memory follows the bytes actually in use, so per-context buffers
// sized for max_context cost only what the context occupies (the paged-pool
// effect without block tables).
//
// A region may also be mirrored into a window: a second reserved range that
// maps the same chunks (attach_window). Kernels and captured graphs keep the
// window address while the backing behind it changes (L0 plane sets of
// agents trade places in a lane by remapping, not by copying), and the
// backing stays reachable at its own address meanwhile.
#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace voidinfer::exl3 {

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
    // Thread-safe against concurrent ensure() calls (a helper thread may
    // pre-grow the range granule by granule while the owner ensures less).
    void ensure(std::size_t bytes) {
        if (!base_) throw std::logic_error("growable region not reserved");
        bytes = std::min(round_up(bytes), capacity_);
        while (mapped_.load(std::memory_order_acquire) < bytes) {
            const std::lock_guard<std::mutex> lock(mutex_);
            const std::size_t mapped = mapped_.load(std::memory_order_relaxed);
            if (mapped >= bytes) break;
            CUmemGenericAllocationHandle handle = take_pooled();
            if (!handle) check(cuMemCreate(&handle, granularity_, &prop_, 0), "growable region create");
            const auto address = base_ + mapped;
            if (cuMemMap(address, granularity_, 0, handle, 0) != CUDA_SUCCESS) {
                cuMemRelease(handle);
                throw std::runtime_error("growable region map");
            }
            if (cuMemSetAccess(address, granularity_, &access_, 1) != CUDA_SUCCESS) {
                cuMemUnmap(address, granularity_);
                cuMemRelease(handle);
                throw std::runtime_error("growable region access");
            }
            if (window_ && (cuMemMap(window_ + mapped, granularity_, 0, handle, 0) != CUDA_SUCCESS ||
                            cuMemSetAccess(window_ + mapped, granularity_, &access_, 1) != CUDA_SUCCESS)) {
                cuMemUnmap(window_ + mapped, granularity_);
                cuMemUnmap(address, granularity_);
                cuMemRelease(handle);
                throw std::runtime_error("growable region window map");
            }
            handles_.push_back(handle);
            mapped_.store(mapped + granularity_, std::memory_order_release);
        }
    }

    // Releases backing beyond `bytes`. The caller guarantees no in-flight
    // work references the released range.
    void shrink(std::size_t bytes) {
        if (!base_) return;
        const std::lock_guard<std::mutex> lock(mutex_);
        bytes = std::min(round_up(bytes), capacity_);
        std::size_t mapped = mapped_.load(std::memory_order_relaxed);
        while (mapped > bytes && !handles_.empty()) {
            mapped -= granularity_;
            if (window_) cuMemUnmap(window_ + mapped, granularity_);
            cuMemUnmap(base_ + mapped, granularity_);
            give_back(handles_.back());
            handles_.pop_back();
        }
        mapped_.store(mapped, std::memory_order_release);
    }

    // Maps every chunk (and later growth) also at `window` (a reserved range
    // of at least this capacity). The caller guarantees no in-flight work
    // references the window range being replaced.
    void attach_window(CUdeviceptr window) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (window_) throw std::logic_error("growable region window already attached");
        const std::size_t mapped = mapped_.load(std::memory_order_relaxed);
        for (std::size_t index = 0; index < handles_.size(); ++index)
            if (cuMemMap(window + index * granularity_, granularity_, 0, handles_[index], 0) != CUDA_SUCCESS) {
                if (index) cuMemUnmap(window, index * granularity_);
                throw std::runtime_error("growable region window map");
            }
        if (mapped && cuMemSetAccess(window, mapped, &access_, 1) != CUDA_SUCCESS) {
            cuMemUnmap(window, mapped);
            throw std::runtime_error("growable region window access");
        }
        window_ = window;
    }
    // Unmaps the window (the chunks stay mapped at this region's address).
    void detach_window() noexcept {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!window_) return;
        const std::size_t mapped = mapped_.load(std::memory_order_relaxed);
        if (mapped) cuMemUnmap(window_, mapped);
        window_ = 0;
    }

    void reset() noexcept {
        if (!base_) return;
        const std::lock_guard<std::mutex> lock(mutex_);
        if (window_ && mapped_.load(std::memory_order_relaxed))
            cuMemUnmap(window_, mapped_.load(std::memory_order_relaxed));
        window_ = 0;
        std::size_t mapped = mapped_.load(std::memory_order_relaxed);
        while (!handles_.empty()) {
            mapped -= granularity_;
            cuMemUnmap(base_ + mapped, granularity_);
            give_back(handles_.back());
            handles_.pop_back();
        }
        cuMemAddressFree(base_, capacity_);
        base_ = 0;
        mapped_.store(0, std::memory_order_release);
    }

    template <class T> T* data() const noexcept { return reinterpret_cast<T*>(base_); }
    std::size_t mapped_bytes() const noexcept { return mapped_.load(std::memory_order_acquire); }
    std::size_t granularity_bytes() const noexcept { return granularity_; }
    std::size_t capacity_bytes() const noexcept { return capacity_; }

private:
    // Pinned host chunks released by host regions (shrink, reset) wait in a
    // process-wide pool for the next region's growth: a fresh L0 plane set
    // (every new conversation) then maps them instead of pinning new host
    // memory (~2.7 GB/s; the first forward's 4096-row margin, ~260 MB, cost
    // ~100 ms). VOIDINFER_EXL3_HOST_CHUNK_POOL_MIB (default 2048, 0 = off)
    // bounds the pooled bytes; device regions never pool.
    struct HostPool {
        std::mutex mutex;
        std::vector<CUmemGenericAllocationHandle> handles;
        std::size_t granularity = 0;
        CUmemLocation location{};
    };
    static HostPool& host_pool() {
        static HostPool pool;
        return pool;
    }
    static std::size_t host_pool_cap() {
        static const std::size_t cap = [] {
            const char* value = std::getenv("VOIDINFER_EXL3_HOST_CHUNK_POOL_MIB");
            return static_cast<std::size_t>(value ? std::strtoull(value, nullptr, 10) : 2048) << 20;
        }();
        return cap;
    }
    bool host_backed() const noexcept { return prop_.location.type != CU_MEM_LOCATION_TYPE_DEVICE; }
    bool pool_matches(const HostPool& pool) const noexcept {
        return pool.granularity == granularity_ && pool.location.type == prop_.location.type &&
            pool.location.id == prop_.location.id;
    }
    CUmemGenericAllocationHandle take_pooled() {
        if (!host_backed()) return 0;
        auto& pool = host_pool();
        const std::lock_guard<std::mutex> lock(pool.mutex);
        if (pool.handles.empty() || !pool_matches(pool)) return 0;
        const auto handle = pool.handles.back();
        pool.handles.pop_back();
        return handle;
    }
    void give_back(CUmemGenericAllocationHandle handle) noexcept {
        if (host_backed()) {
            auto& pool = host_pool();
            const std::lock_guard<std::mutex> lock(pool.mutex);
            if (pool.handles.empty()) {
                pool.granularity = granularity_;
                pool.location = prop_.location;
            }
            if (pool_matches(pool) && (pool.handles.size() + 1) * granularity_ <= host_pool_cap()) {
                pool.handles.push_back(handle);
                return;
            }
        }
        cuMemRelease(handle);
    }
    static void check(CUresult result, const char* what) {
        if (result != CUDA_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed (CUresult " +
                                     std::to_string(static_cast<int>(result)) + ")");
    }
    std::size_t round_up(std::size_t bytes) const noexcept {
        return granularity_ ? (bytes + granularity_ - 1) / granularity_ * granularity_ : bytes;
    }
    CUdeviceptr base_ = 0, window_ = 0;
    std::size_t capacity_ = 0, granularity_ = 0;
    std::atomic<std::size_t> mapped_{0};
    std::mutex mutex_;
    CUmemAllocationProp prop_{};
    CUmemAccessDesc access_{};
    std::vector<CUmemGenericAllocationHandle> handles_;
};

// A reserved virtual range without backing of its own: the window that a
// growable region's chunks are mirrored into (attach_window).
class Exl3RegionWindow {
public:
    explicit Exl3RegionWindow(std::size_t capacity) {
        CUdevice device = 0;
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS) throw std::runtime_error("region window device");
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_HOST;
        std::size_t granularity = 0;
        if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
            throw std::runtime_error("region window granularity");
        capacity_ = (std::max<std::size_t>(capacity, 1) + granularity - 1) / granularity * granularity;
        if (cuMemAddressReserve(&base_, capacity_, 0, 0, 0) != CUDA_SUCCESS)
            throw std::runtime_error("region window reserve");
    }
    ~Exl3RegionWindow() { if (base_) cuMemAddressFree(base_, capacity_); }
    Exl3RegionWindow(const Exl3RegionWindow&) = delete;
    Exl3RegionWindow& operator=(const Exl3RegionWindow&) = delete;
    CUdeviceptr base() const noexcept { return base_; }
    std::size_t capacity_bytes() const noexcept { return capacity_; }

private:
    CUdeviceptr base_ = 0;
    std::size_t capacity_ = 0;
};

}  // namespace voidinfer::exl3
