#pragma once

#include <cstddef>
#include <new>
#include <atomic>
#include <cstdint>
#include "exl3/retained_descriptor_ledger.h"

namespace ninfer::exl3 {
// Shared across allocator rebinds; counts blocks until their last weak owner dies.
struct Exl3SharedControlAccounting {
    inline static std::atomic<std::uint64_t> live_bytes{0};
};
struct alignas(std::max_align_t) Exl3SharedControlCredit {
    std::optional<RetainedDescriptorLedger::Ticket> ticket;
};

// Exactly one fixed-capacity allocation for one shared_ptr control block.
// The admission flag belongs to the backing during construction. Deallocation
// never accesses it: the object may already be destroyed while weak views live.
template<class T> class Exl3ReconstructionControlAllocator {
    template<class> friend class Exl3ReconstructionControlAllocator;
    bool* admitted_=nullptr;
    Exl3SharedControlCredit** credit_slot_=nullptr;
public:
    using value_type=T;
    template<class U> struct rebind { using other=Exl3ReconstructionControlAllocator<U>; };
    static constexpr std::size_t capacity=512;
    static_assert(sizeof(Exl3SharedControlCredit)<capacity,"shared control credit header exhausts block");
    static constexpr std::size_t payload_capacity=capacity-sizeof(Exl3SharedControlCredit);
    // allocator_traits is required to be able to rebind an allocator while the
    // rebound value_type (MSVC's shared_ptr control block here) is incomplete.
    // Keep the class definition independent of sizeof/alignof(T); the control
    // block is complete by the time allocate() is instantiated.
    static constexpr std::size_t alignment=alignof(std::max_align_t);
    Exl3ReconstructionControlAllocator() noexcept=default;
    explicit Exl3ReconstructionControlAllocator(bool* admitted,Exl3SharedControlCredit** credit_slot=nullptr) noexcept
        :admitted_(admitted),credit_slot_(credit_slot){}
    template<class U> Exl3ReconstructionControlAllocator(const Exl3ReconstructionControlAllocator<U>& other) noexcept
        :admitted_(other.admitted_),credit_slot_(other.credit_slot_){}
    T* allocate(std::size_t count) {
        static_assert(alignof(T)<=alignment,"over-aligned shared_ptr control block is unsupported");
        if(!admitted_ || *admitted_ || !count || count>payload_capacity/sizeof(T))throw std::bad_alloc();
        auto* storage=static_cast<std::byte*>(::operator new(capacity,std::align_val_t(alignment)));
        auto* credit=new(storage) Exl3SharedControlCredit;
        if(credit_slot_)*credit_slot_=credit;
        auto* result=reinterpret_cast<T*>(storage+sizeof(Exl3SharedControlCredit));
        *admitted_=true;
        Exl3SharedControlAccounting::live_bytes.fetch_add(capacity,std::memory_order_relaxed);
        return result;
    }
    void deallocate(T* pointer,std::size_t) noexcept {
        auto* storage=reinterpret_cast<std::byte*>(pointer)-sizeof(Exl3SharedControlCredit);
        auto* credit=reinterpret_cast<Exl3SharedControlCredit*>(storage);
        auto ticket=std::move(credit->ticket);
        credit->~Exl3SharedControlCredit();
        ::operator delete(storage,std::align_val_t(alignment));
        Exl3SharedControlAccounting::live_bytes.fetch_sub(capacity,std::memory_order_relaxed);
    }
    template<class U> bool operator==(const Exl3ReconstructionControlAllocator<U>& other) const noexcept {
        return admitted_==other.admitted_ && credit_slot_==other.credit_slot_;
    }
};
} // namespace ninfer::exl3
