#pragma once
#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace ninfer::exl3 {
struct EmbeddedSharedControl {
    static constexpr std::size_t bytes=256;
    alignas(std::max_align_t) std::byte storage[bytes];
    bool used=false;
    EmbeddedSharedControl()=default;
    EmbeddedSharedControl(const EmbeddedSharedControl&)=delete;
    EmbeddedSharedControl& operator=(const EmbeddedSharedControl&)=delete;
    template<class T> T* allocate(std::size_t count) {
        if(used || !count || alignof(T)>alignof(std::max_align_t) || count>bytes/sizeof(T))
            throw std::bad_alloc();
        used=true;return reinterpret_cast<T*>(storage);
    }
};
// Control memory belongs to an already-reserved owner. The allocator retains
// that owner through the final weak-reference deallocation call. It never
// allocates another heap block and never reuses an arena after construction.
template<class T> struct EmbeddedControlAllocator {
    using value_type=T;
    std::shared_ptr<void> owner;
    EmbeddedSharedControl* arena;
    bool fail;
    EmbeddedControlAllocator(std::shared_ptr<void> backing,EmbeddedSharedControl& storage,bool failure=false) noexcept
        :owner(std::move(backing)),arena(&storage),fail(failure) {}
    template<class U> EmbeddedControlAllocator(const EmbeddedControlAllocator<U>& other) noexcept
        :owner(other.owner),arena(other.arena),fail(other.fail) {}
    T* allocate(std::size_t count) {
        if(fail || !owner)throw std::bad_alloc();
        return arena->allocate<T>(count);
    }
    void deallocate(T*,std::size_t) noexcept {}
    template<class U> bool operator==(const EmbeddedControlAllocator<U>& other) const noexcept {
        return owner.get()==other.owner.get() && arena==other.arena && fail==other.fail;
    }
};
}
