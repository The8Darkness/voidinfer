#pragma once
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <utility>
#include <optional>
#include "exl3/retained_descriptor_ledger.h"

namespace ninfer::exl3 {
namespace bounded_shared_detail {
template<class T,std::size_t ControlBytes> struct Storage {
    static_assert(ControlBytes>0);
    inline static std::atomic<std::size_t> live_blocks{0};
    std::optional<RetainedDescriptorLedger::Ticket> retirement_credit;
    std::optional<RetainedDescriptorLedger::Ticket> payload_retirement_credit;
    alignas(T) std::byte payload[sizeof(T)];
    alignas(std::max_align_t) std::byte control[ControlBytes];
    bool alive=false,installed=false,control_in_use=false;
    Storage() noexcept {live_blocks.fetch_add(1,std::memory_order_relaxed);}
    Storage(const Storage&)=delete;
    Storage& operator=(const Storage&)=delete;
    void destroy_payload() noexcept {
        auto credit=std::move(payload_retirement_credit);
        if(std::exchange(alive,false))std::destroy_at(std::launder(reinterpret_cast<T*>(payload)));
    }
    ~Storage(){destroy_payload();live_blocks.fetch_sub(1,std::memory_order_relaxed);}
};
template<class U,class Owner,std::size_t ControlBytes> struct ControlAllocator {
    using value_type=U;
    Owner* owner;
    template<class V> struct rebind {using other=ControlAllocator<V,Owner,ControlBytes>;};
    explicit ControlAllocator(Owner* value) noexcept:owner(value) {}
    template<class V> ControlAllocator(const ControlAllocator<V,Owner,ControlBytes>& other) noexcept:owner(other.owner) {}
    U* allocate(std::size_t count) {
        if(!count || owner->control_in_use || alignof(U)>alignof(std::max_align_t) || count>ControlBytes/sizeof(U))
            throw std::bad_alloc();
        owner->control_in_use=true;
        return reinterpret_cast<U*>(owner->control);
    }
    void deallocate(U*,std::size_t) noexcept {
        owner->control_in_use=false;
        // During failed construction the factory's unique owner still owns
        // cleanup. Once installed, final weak retirement releases the block.
        if(owner->installed) {
            auto credit=std::move(owner->retirement_credit);
            delete owner; // Release the charge after physical block deletion.
        }
    }
    template<class V> bool operator==(const ControlAllocator<V,Owner,ControlBytes>& other) const noexcept {
        return owner==other.owner;
    }
};
template<class Owner> struct DestroyPayload {
    Owner* owner;
    template<class T> void operator()(T*) const noexcept {owner->destroy_payload();}
};
}
// Includes object, reserved shared_ptr control storage, state and alignment.
// General heap allocator bookkeeping/fragmentation is not included.
template<class T,std::size_t ControlBytes=256>
constexpr std::size_t bounded_shared_allocation_bytes() noexcept {
    return sizeof(bounded_shared_detail::Storage<T,ControlBytes>);
}
template<class T,std::size_t ControlBytes=256,class... Args>
std::shared_ptr<T> make_bounded_shared(Args&&... args) {
    using Owner=bounded_shared_detail::Storage<T,ControlBytes>;
    auto owner=std::make_unique<Owner>();
    T* payload=std::construct_at(reinterpret_cast<T*>(owner->payload),std::forward<Args>(args)...);
    owner->alive=true;
    std::shared_ptr<T> result(payload,bounded_shared_detail::DestroyPayload<Owner>{owner.get()},
        bounded_shared_detail::ControlAllocator<T,Owner,ControlBytes>{owner.get()});
    owner->installed=true;owner.release();
    return result;
}
template<class T,std::size_t ControlBytes=256>
std::size_t bounded_shared_live_blocks_for_test() noexcept {
    return bounded_shared_detail::Storage<T,ControlBytes>::live_blocks.load(std::memory_order_relaxed);
}
template<class T,class Alias,std::size_t ControlBytes=256>
bool can_attach_bounded_retirement_credit(const std::shared_ptr<Alias>& value) noexcept {
    using Owner=bounded_shared_detail::Storage<T,ControlBytes>;
    const auto* deleter=std::get_deleter<bounded_shared_detail::DestroyPayload<Owner>>(value);
    return deleter && !deleter->owner->retirement_credit;
}
template<class T,class Alias,std::size_t ControlBytes=256>
bool bounded_split_credit_belongs_to(const std::shared_ptr<Alias>& value,
    const RetainedDescriptorLedger& ledger) noexcept {
    using Owner=bounded_shared_detail::Storage<T,ControlBytes>;
    const auto* deleter=std::get_deleter<bounded_shared_detail::DestroyPayload<Owner>>(value);
    return deleter && deleter->owner->retirement_credit && deleter->owner->payload_retirement_credit &&
        ledger.owns(*deleter->owner->retirement_credit) && ledger.owns(*deleter->owner->payload_retirement_credit);
}
template<class T,class Alias,std::size_t ControlBytes=256>
bool bounded_credit_belongs_to(const std::shared_ptr<Alias>& value,
    const RetainedDescriptorLedger& ledger) noexcept {
    using Owner=bounded_shared_detail::Storage<T,ControlBytes>;
    const auto* deleter=std::get_deleter<bounded_shared_detail::DestroyPayload<Owner>>(value);
    return deleter && deleter->owner->retirement_credit && ledger.owns(*deleter->owner->retirement_credit);
}
// Dynamic payload allocations die with T; the containing block survives weak
// references. Preserve separate charges without changing T's copy semantics.
template<class T,class Alias,std::size_t ControlBytes=256>
bool attach_bounded_split_retirement_credit(const std::shared_ptr<Alias>& value,
    RetainedDescriptorLedger::Ticket credit,std::size_t payload_bytes) noexcept {
    using Owner=bounded_shared_detail::Storage<T,ControlBytes>;
    const auto* deleter=std::get_deleter<bounded_shared_detail::DestroyPayload<Owner>>(value);
    if(!deleter || deleter->owner->retirement_credit || deleter->owner->payload_retirement_credit ||
        credit.bytes()<sizeof(Owner) || credit.bytes()-sizeof(Owner)!=payload_bytes)return false;
    auto block=credit.split(sizeof(Owner));if(!block)return false;
    deleter->owner->retirement_credit.emplace(std::move(*block));
    deleter->owner->payload_retirement_credit.emplace(std::move(credit));return true;
}
template<class T,class Alias,std::size_t ControlBytes=256>
bool attach_bounded_retirement_credit(const std::shared_ptr<Alias>& value,
    RetainedDescriptorLedger::Ticket credit) noexcept {
    using Owner=bounded_shared_detail::Storage<T,ControlBytes>;
    const auto* deleter=std::get_deleter<bounded_shared_detail::DestroyPayload<Owner>>(value);
    if(!deleter || deleter->owner->retirement_credit)return false;
    deleter->owner->retirement_credit.emplace(std::move(credit));return true;
}
// Caller must serialize this diagnostic with retirement-credit attachment.
template<class T,class Alias,std::size_t ControlBytes=256>
std::optional<std::uint64_t> bounded_retirement_credit_bytes_for_test(
    const std::shared_ptr<Alias>& value) noexcept {
    using Owner=bounded_shared_detail::Storage<T,ControlBytes>;
    const auto* deleter=std::get_deleter<bounded_shared_detail::DestroyPayload<Owner>>(value);
    if(!deleter || !deleter->owner->retirement_credit)return std::nullopt;
    return deleter->owner->retirement_credit->bytes();
}
}
