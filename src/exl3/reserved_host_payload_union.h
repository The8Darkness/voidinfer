#pragma once
#include "exl3/retained_descriptor_ledger.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {
// Caller-selected fixed storage: reconciliation never grows metadata while
// deciding whether the next allocation can be admitted.
template<std::size_t Capacity> class ReservedHostPayloadUnion {
    struct Entry {
        std::shared_ptr<const void> owner;
        std::size_t plane=0;
        std::uintptr_t first=0,last=0;
    };
    std::array<Entry,Capacity> entries_{};
    std::size_t count_=0;
public:
    void add(std::shared_ptr<const void> owner,std::size_t plane,const void* data,std::size_t bytes) {
        const auto first=reinterpret_cast<std::uintptr_t>(data);
        if(!owner || !owner.use_count() || !data || !bytes || bytes>std::numeric_limits<std::uintptr_t>::max()-first)
            throw std::invalid_argument("reserved host payload identity/extent");
        const auto last=first+bytes;
        for(std::size_t i=0;i<count_;++i) {
            const auto& entry=entries_[i];
            const bool same=owner.get()==entry.owner.get() && !owner.owner_before(entry.owner) && !entry.owner.owner_before(owner);
            if(same && plane==entry.plane) {
                if(first!=entry.first || last!=entry.last)throw std::invalid_argument("reserved host payload changed allocation");
                return;
            }
            if(first<entry.last && entry.first<last)throw std::invalid_argument("reserved host payload overlapping identities");
        }
        if(count_==Capacity)throw std::length_error("reserved host payload reconciliation capacity");
        entries_[count_++]={std::move(owner),plane,first,last};
    }
    std::size_t allocations() const noexcept {return count_;}
    std::uint64_t bytes_outside(std::span<const std::pair<std::uintptr_t,std::uintptr_t>> resident,
        std::span<const std::pair<std::uintptr_t,std::uintptr_t>> incoming={}) const {
        // The resident registry supplies its sorted, disjoint page union.
        for(const auto ranges:{resident,incoming}) {
          std::uintptr_t previous=0;
          for(const auto& range:ranges) {
            if(range.second<=range.first || range.first<previous)
                throw std::invalid_argument("reserved payload resident ranges not a union");
            previous=range.second;
          }
        }
        std::uint64_t total=0;
        for(std::size_t i=0;i<count_;++i) {
            const auto& entry=entries_[i];
            std::uint64_t remaining=entry.last-entry.first;
            std::size_t old_index=0,new_index=0;
            auto covered_end=entry.first;
            while(old_index<resident.size() || new_index<incoming.size()) {
                const auto& range=(new_index==incoming.size() ||
                    (old_index<resident.size() && resident[old_index].first<=incoming[new_index].first))
                    ?resident[old_index++]:incoming[new_index++];
                if(range.first>=entry.last)break;
                const auto first=std::max(covered_end,range.first),last=std::min(entry.last,range.second);
                if(last>first) {remaining-=last-first;covered_end=last;}
            }
            if(remaining>UINT64_MAX-total)throw std::overflow_error("reserved payload uncovered bytes");
            total+=remaining;
        }
        return total;
    }
    std::uint64_t page_bytes(std::size_t page) const {
        if(!page || (page&(page-1)))throw std::invalid_argument("reserved host payload page size");
        std::array<std::pair<std::uintptr_t,std::uintptr_t>,Capacity> ranges{};
        for(std::size_t i=0;i<count_;++i) {
            if(entries_[i].last>std::numeric_limits<std::uintptr_t>::max()-(page-1))
                throw std::overflow_error("reserved host payload page rounding");
            ranges[i]={entries_[i].first&~std::uintptr_t(page-1),(entries_[i].last+page-1)&~std::uintptr_t(page-1)};
        }
        std::sort(ranges.begin(),ranges.begin()+count_);
        std::uint64_t total=0;std::uintptr_t end=0;
        for(std::size_t i=0;i<count_;++i) {
            const auto first=std::max(end,ranges[i].first),last=ranges[i].second;
            if(last>first) {
                if(last-first>UINT64_MAX-total)throw std::overflow_error("reserved host payload union bytes");
                total+=last-first;
            }
            end=std::max(end,last);
        }
        return total;
    }
};
template<std::size_t Capacity> class ReservedHostPayloadTracker {
    struct Entry {
        std::weak_ptr<const void> owner;
        std::size_t plane=0;
        const void* data=nullptr;
        std::size_t bytes=0;
    };
    std::array<Entry,Capacity> entries_{};
    mutable std::mutex mutex_;
public:
    void track(const std::shared_ptr<const void>& owner,std::size_t plane,const void* data,std::size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        ReservedHostPayloadUnion<Capacity> checked;
        std::size_t slot=Capacity;
        for(std::size_t i=0;i<Capacity;++i) {
            auto live=entries_[i].owner.lock();
            if(!live) {if(slot==Capacity)slot=i;continue;}
            checked.add(live,entries_[i].plane,entries_[i].data,entries_[i].bytes);
            if(live.get()==owner.get() && !live.owner_before(owner) && !owner.owner_before(live) &&
               plane==entries_[i].plane) {
                checked.add(owner,plane,data,bytes); // Reject a changed allocation.
                return;
            }
        }
        checked.add(owner,plane,data,bytes); // Validate before mutating registry.
        if(slot==Capacity)throw std::length_error("reserved host payload tracker full");
        entries_[slot]={owner,plane,data,bytes};
    }
    ReservedHostPayloadUnion<Capacity> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        ReservedHostPayloadUnion<Capacity> result;
        for(const auto& entry:entries_)if(auto live=entry.owner.lock())
            result.add(std::move(live),entry.plane,entry.data,entry.bytes);
        return result;
    }
    // Acquisition/registration and the resident union must be serialized by
    // the caller. Retirement may run concurrently: keep the snapshot owners
    // alive until after reading the ledger so covered credit cannot disappear
    // between the intersection calculation and subtraction. Registration must
    // already have authenticated each owner against this exact ledger.
    std::uint64_t additional_bytes(
        const RetainedHostAllocationLedger& ledger,
        std::span<const std::pair<std::uintptr_t,std::uintptr_t>> resident,
        std::span<const std::pair<std::uintptr_t,std::uintptr_t>> incoming={}) const {
        const auto live=snapshot();
        const auto tracked=live.bytes_outside({});
        const auto uncovered=live.bytes_outside(resident,incoming);
        const auto reserved=ledger.bytes();
        if(tracked>reserved)
            throw std::logic_error("reserved host payload tracking exceeds ledger");
        // Unbound promises and allocations absent from resident roots remain
        // charged. Page padding never discounts raw payload reservations.
        return reserved-(tracked-uncovered);
    }
};
}
