#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::exl3 {
// One serialized physical batch owns the descriptor tables until a successful
// host-side fence. This record allocates nothing and does not own CUDA handles.
class Exl3TransferDescriptorUse {
    bool pending_=false;
public:
    Exl3TransferDescriptorUse()=default;
    Exl3TransferDescriptorUse(const Exl3TransferDescriptorUse&)=delete;
    Exl3TransferDescriptorUse& operator=(const Exl3TransferDescriptorUse&)=delete;
    bool pending() const noexcept {return pending_;}
    void begin_submission() {
        if(pending_)throw std::logic_error("transfer descriptors still borrowed");
        pending_=true;
    }
    template<class Fence> bool retire_before_reuse(Fence&& fence) {
        if(!pending_)return false;
        std::forward<Fence>(fence)(); // A throw preserves uncertain ownership.
        pending_=false;
        return true;
    }
};
// Startup-sized scalar descriptor storage. clear/append never relocate buffers
// borrowed by the transfer API; callers fence a submission before reuse.
template<class T> class Exl3TransferDescriptorStorage {
    static_assert(std::is_scalar_v<T> && !std::is_const_v<T>);
    std::unique_ptr<T[]> values_;
    std::size_t capacity_=0,size_=0;
public:
    Exl3TransferDescriptorStorage()=default;
    Exl3TransferDescriptorStorage(const Exl3TransferDescriptorStorage&)=delete;
    Exl3TransferDescriptorStorage& operator=(const Exl3TransferDescriptorStorage&)=delete;
    Exl3TransferDescriptorStorage(Exl3TransferDescriptorStorage&&)=delete;
    Exl3TransferDescriptorStorage& operator=(Exl3TransferDescriptorStorage&&)=delete;
    static constexpr std::size_t max_size() noexcept {
        return static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(T);
    }
    void reserve(std::size_t count) {
        if(count<=capacity_)return;
        if(capacity_)throw std::logic_error("transfer descriptor storage cannot grow");
        if(count>max_size())throw std::length_error("transfer descriptor capacity overflow");
        auto prepared=std::make_unique<T[]>(count);
        values_=std::move(prepared);capacity_=count;
    }
    void push_back(T value) {
        if(size_==capacity_)throw std::length_error("transfer descriptor storage exhausted");
        values_[size_++]=value;
    }
    void clear() noexcept {size_=0;}
    std::size_t capacity() const noexcept {return capacity_;}
    std::size_t size() const noexcept {return size_;}
    bool empty() const noexcept {return size_==0;}
    T* data() noexcept {return values_.get();}
    const T* data() const noexcept {return values_.get();}
};
}
