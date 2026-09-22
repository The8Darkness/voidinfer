#pragma once

#include <cstdint>
#include <atomic>
#include <mutex>
#include <memory>
#include <cstddef>
#include <stdexcept>

namespace ninfer::exl3 {

// One context slab can be overwritten by many projection workspaces, but only
// on its first admitted ordered stream. No reset API: a host return is not a
// completion proof and cannot permit another stream to overwrite queued reads.
class Exl3ReconstructionStream {
public:
    void bind_model(std::shared_ptr<const void> model,std::size_t bytes,bool k6_down,
                    const void* data=nullptr,bool k6_gate_up=false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!model || !bytes || model_ || bound_)
            throw std::invalid_argument("reconstruction model binding requires pristine backing");
        model_=std::move(model);bytes_=bytes;k6_down_=k6_down;
        k6_gate_up_=k6_gate_up;data_=data;
    }
    void require_identity(const std::shared_ptr<const void>& model,std::size_t bytes,bool k6_down,
                          const void* data=nullptr,bool k6_gate_up=false) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!model_ && !model)return; // Explicit unbound operator-test backing only.
        if(!model_ || !model || model_.get()!=model.get() || model_.owner_before(model) ||
            model.owner_before(model_) || bytes!=bytes_ || k6_down!=k6_down_ ||
            k6_gate_up!=k6_gate_up_ || data!=data_)
            throw std::invalid_argument("reconstruction model or slice configuration mismatch");
    }
    void require_ordered(std::uintptr_t stream) {
        auto lock=acquire(stream);
    }
    // Hold this lock through decode, all partials and final reduction submission.
    // Otherwise two host callers could interleave slab overwrite and consumption
    // even when both enqueue on the same CUDA stream.
    std::unique_lock<std::mutex> acquire(std::uintptr_t stream) {
        std::unique_lock<std::mutex> lock(mutex_);
        if(failed_.load(std::memory_order_acquire))
            throw std::runtime_error("reconstruction slab submission previously failed");
        if(bound_ && stream_!=stream)
            throw std::invalid_argument("reconstruction slab has a different ordered stream");
        stream_=stream;
        bound_=true;
        used_.store(true,std::memory_order_release);
        return lock;
    }
    void fail() noexcept {failed_.store(true,std::memory_order_release);}
    bool failed() const noexcept {return failed_.load(std::memory_order_acquire);}
    bool used() const noexcept {return used_.load(std::memory_order_acquire);}
private:
    std::mutex mutex_;
    std::uintptr_t stream_=0;
    bool bound_=false;
    std::shared_ptr<const void> model_;
    std::size_t bytes_=0;
    bool k6_down_=false;
    bool k6_gate_up_=false;
    const void* data_=nullptr;
    std::atomic<bool> failed_{false};
    std::atomic<bool> used_{false};
};

} // namespace ninfer::exl3
