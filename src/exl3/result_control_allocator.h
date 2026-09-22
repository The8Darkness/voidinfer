#pragma once
#include <cstddef>
#include <memory>
#include <new>

namespace ninfer::exl3 {
// Ordinary allocation in production. The explicit diagnostic survives allocator
// rebinding so shared_ptr fails at its actual control-block allocation boundary.
template<class T> struct ResultControlAllocator {
    using value_type=T;
    bool fail=false;
    explicit ResultControlAllocator(bool failure=false) noexcept:fail(failure) {}
    template<class U> ResultControlAllocator(const ResultControlAllocator<U>& other) noexcept:fail(other.fail) {}
    T* allocate(std::size_t count) {
        if(fail)throw std::bad_alloc{};
        return std::allocator<T>{}.allocate(count);
    }
    void deallocate(T* pointer,std::size_t count) noexcept {
        std::allocator<T>{}.deallocate(pointer,count);
    }
    template<class U> bool operator==(const ResultControlAllocator<U>& other) const noexcept {
        return fail==other.fail;
    }
};
}
