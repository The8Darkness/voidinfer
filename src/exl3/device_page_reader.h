#pragma once
#include "exl3/device_page_fill.h"
#include "exl3/kv_transfer_lease.h"
#include <limits>

namespace ninfer::exl3 {
// Reuse the existing retained final-use record: the ready view is its destination
// owner and the attention lane/event bundle is its independent auxiliary owner.
// No registration claim is inferred from that auxiliary reference.
class Exl3DevicePageReader {
    std::shared_ptr<const Exl3DevicePageFill::View> view_;
    Exl3KVTransferLease use_;
    bool completed_=false;
    bool failed_=false;
public:
    bool can_attach_retirement_credit() const noexcept {
        return use_.can_attach_retirement_credit();
    }
    bool attach_retirement_credit(RetainedDescriptorLedger::Ticket credit) noexcept {
        return use_.attach_retirement_credit(std::move(credit));
    }
    static constexpr std::size_t metadata_bytes() noexcept {
        return sizeof(Exl3DevicePageReader)+Exl3KVTransferLease::metadata_bytes()-sizeof(Exl3KVTransferLease);
    }
    void bind(std::shared_ptr<const Exl3DevicePageFill::View> view,
        const Exl3DevicePageKey& expected,std::uint64_t page_generation,
        std::shared_ptr<const void> event_owner) {
        if(!view || !view.use_count() || !event_owner || !event_owner.use_count() ||
            !page_generation || view->generation()!=page_generation || !view->key().same(expected))
            throw std::invalid_argument("device page reader key/generation/event owner");
        const auto& page=view->key().page();
        auto extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,page->first+64);
        use_.rebind(std::move(extent),view,std::move(event_owner));
        view_=std::move(view);completed_=false;
    }
    std::uint64_t begin(std::uint64_t acquisition,std::uint64_t execution,std::uintptr_t event) {
        if(!view_ || !view_->key().current())throw std::logic_error("device page reader unbound/stale");
        auto readers=view_->readers_->load(std::memory_order_relaxed);
        do {
            if(readers==std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("device page reader count exhausted");
        } while(!view_->readers_->compare_exchange_weak(readers,readers+1,
            std::memory_order_acq_rel,std::memory_order_relaxed));
        try {return use_.begin(acquisition,execution,event);}
        catch(...) {
            view_->readers_->fetch_sub(1,std::memory_order_release);
            throw;
        }
    }
    std::span<const std::uint16_t> plane(int bank,bool key) const {
        if(!view_ || failed_ || !use_.uncertain())throw std::logic_error("device page reader outside active use");
        return view_->plane(bank,key);
    }
    bool finish(std::uint64_t generation,int error) {
        if(!use_.finish(generation,error))return false;
        if(error)failed_=true;
        if(use_.complete()) {
            view_->readers_->fetch_sub(1,std::memory_order_release);
            use_.clear_completed();view_.reset();completed_=true;
        }
        return true;
    }
    bool complete() const noexcept {return completed_;}
    const Exl3DevicePageKey& input_key() const {
        if(!view_ || failed_ || !use_.uncertain() || !view_->key().current())
            throw std::logic_error("device page reader input outside active use");
        return view_->key();
    }
    bool uncertain() const noexcept {return use_.uncertain();}
};
}
