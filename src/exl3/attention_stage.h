#pragma once
#include "exl3/kv_transfer_lease.h"
#include "exl3/resource_inventory.h"
#include "exl3/bounded_shared_owner.h"
#include <array>
#include <atomic>
#include <limits>

namespace ninfer::exl3 {
// Two precredited staging records. The backing owner supplied to begin must own
// the destination and both CUDA events. This schedule invokes no CUDA operations;
// caller event witnesses are required before exposing or recycling a stage.
class Exl3AttentionStage {
    enum class Phase {idle,producing,ready,consuming,failed};
    struct Slot {
        Exl3KVTransferLease lease;
        Phase phase=Phase::idle;
        std::uint64_t generation=0;
        std::uintptr_t producer_event=0,consumer_event=0;
        bool cancelled=false;
    };
    std::array<Slot,2> slots_;
    inline static std::atomic<std::uint64_t> next_identity_{1};
    static std::uint64_t allocate_identity() {
        auto identity=next_identity_.load(std::memory_order_relaxed);
        do {
            if(identity==std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("attention stage identity exhausted");
        } while(!next_identity_.compare_exchange_weak(identity,identity+1,std::memory_order_relaxed));
        return identity;
    }
    const std::uint64_t identity_=allocate_identity();
public:
    struct Ticket {unsigned slot=0;std::uint64_t generation=0;std::uint64_t owner_identity=0;};
    bool uncertain() const noexcept {
        for(const auto& slot:slots_)if(slot.lease.uncertain())return true;
        return false;
    }
    static constexpr std::size_t metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3AttentionStage>()+2*Exl3KVTransferLease::state_metadata_bytes();
    }
    static bool attach_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || credit.bytes()!=metadata_bytes() ||
            !can_attach_bounded_retirement_credit<Exl3AttentionStage>(owner))return false;
        auto* stage=const_cast<Exl3AttentionStage*>(static_cast<const Exl3AttentionStage*>(owner.get()));
        for(const auto& slot:stage->slots_)if(!slot.lease.can_attach_retirement_credit())return false;
        for(auto& slot:stage->slots_) {
            auto part=credit.split(Exl3KVTransferLease::state_metadata_bytes());
            if(!part || !slot.lease.attach_retirement_credit(std::move(*part)))return false;
        }
        return attach_bounded_retirement_credit<Exl3AttentionStage>(owner,std::move(credit));
    }
    template<class Coordinator>
    static std::shared_ptr<Exl3AttentionStage> create_startup(Coordinator& authority) {
        using Inventory=Exl3ResourceInventory;
        Inventory::Requirement required;required.configuration=0x4154545354;
        required.add(Inventory::Domain::host_metadata,1,metadata_bytes());
        std::shared_ptr<Exl3AttentionStage> result;
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("attention stage reservation identity");
            auto staged=make_bounded_shared<Exl3AttentionStage>();
            Inventory actual;actual.add({staged,0,Inventory::Domain::host_metadata,metadata_bytes(),{},
                &attach_retirement_credit});
            result=std::move(staged);return actual;
        },[&]() noexcept {result.reset();});
        return result;
    }
    void require_producer(unsigned index,
        std::uint64_t acquisition,std::uint64_t execution,
        std::uintptr_t producer_event,std::uintptr_t consumer_event) const {
        if(index>=slots_.size() || !acquisition || !execution ||
           !producer_event || !consumer_event || producer_event==consumer_event)
            throw std::invalid_argument("attention stage slot/event identity");
        const auto& slot=slots_[index];
        if(slot.phase!=Phase::idle)throw std::logic_error("attention stage still owned");
        for(unsigned other=0;other<slots_.size();++other) if(other!=index) {
            const auto& peer=slots_[other];
            if(peer.phase!=Phase::idle && (producer_event==peer.producer_event ||
                producer_event==peer.consumer_event || consumer_event==peer.producer_event ||
                consumer_event==peer.consumer_event))
                throw std::invalid_argument("attention stage active event alias");
        }
    }
    Ticket begin(unsigned index,Exl3ExactKVExtent extent,std::shared_ptr<const void> backing,
        std::uint64_t acquisition,std::uint64_t execution,
        std::uintptr_t producer_event,std::uintptr_t consumer_event,std::shared_ptr<const void> source_group={},
        std::optional<Exl3KVRegistrationCache::ExternalRead> registration_read={}) {
        require_producer(index,acquisition,execution,producer_event,consumer_event);
        auto& slot=slots_[index];
        slot.lease.rebind(std::move(extent),std::move(backing),{},std::move(source_group),std::move(registration_read));
        slot.generation=slot.lease.begin(acquisition,execution,consumer_event);
        slot.producer_event=producer_event;slot.consumer_event=consumer_event;
        slot.cancelled=false;
        slot.phase=Phase::producing;
        return {index,slot.generation,identity_};
    }
    bool producer_pending(Ticket ticket,std::uintptr_t event) const noexcept {
        return ticket.owner_identity==identity_ && ticket.slot<slots_.size() && slots_[ticket.slot].generation==ticket.generation &&
            slots_[ticket.slot].phase==Phase::producing && slots_[ticket.slot].producer_event==event;
    }
    bool consumer_pending(Ticket ticket,std::uintptr_t event) const noexcept {
        return ticket.owner_identity==identity_ && ticket.slot<slots_.size() && slots_[ticket.slot].generation==ticket.generation &&
            slots_[ticket.slot].phase==Phase::consuming && slots_[ticket.slot].consumer_event==event;
    }
    std::optional<Exl3FinalUseWitness::Snapshot> final_use_for_test(Ticket ticket) const noexcept {
        if(ticket.owner_identity!=identity_ || ticket.slot>=slots_.size() ||
            slots_[ticket.slot].generation!=ticket.generation)return {};
        return slots_[ticket.slot].lease.final_use_snapshot_for_test();
    }
    bool producer_finished(Ticket ticket,std::uintptr_t event,int error) noexcept {
        if(ticket.owner_identity!=identity_ || ticket.slot>=slots_.size())return false;
        auto& slot=slots_[ticket.slot];
        if(slot.generation!=ticket.generation || slot.phase!=Phase::producing || event!=slot.producer_event)return false;
        if(error) {
            slot.phase=Phase::failed;
            slot.lease.finish_after_alternate_event(ticket.generation,event,error);
            return false;
        }
        slot.phase=Phase::ready;
        if(slot.cancelled) return cancel(ticket);
        return true;
    }
    void require_consumer(Ticket ticket,int bank,Exl3ExactKVExtent::Plane plane) const {
        if(ticket.owner_identity!=identity_ || ticket.slot>=slots_.size())throw std::invalid_argument("attention stage ticket");
        const auto& slot=slots_[ticket.slot];
        if(slot.generation!=ticket.generation || slot.phase!=Phase::ready || slot.cancelled ||
            slot.lease.source().bank()!=bank || slot.lease.source().plane()!=plane ||
            !slot.lease.source().backing_current())throw std::logic_error("attention stage not ready for consumer");
    }
    const Exl3ExactKVExtent& consume(Ticket ticket,int bank,Exl3ExactKVExtent::Plane plane) {
        require_consumer(ticket,bank,plane);
        auto& slot=slots_[ticket.slot];
        slot.phase=Phase::consuming;return slot.lease.source();
    }
    bool consumer_finished(Ticket ticket,std::uintptr_t event,int error) noexcept {
        if(ticket.owner_identity!=identity_ || ticket.slot>=slots_.size())return false;
        auto& slot=slots_[ticket.slot];
        if(slot.generation!=ticket.generation || slot.phase!=Phase::consuming || event!=slot.consumer_event)return false;
        // finish accepts an error witness but does not mark it complete.
        // Never clear retained owners on that accepted failure notification.
        if(error) {slot.lease.finish(ticket.generation,error);slot.phase=Phase::failed;return false;}
        if(!slot.lease.finish(ticket.generation,error)) {slot.phase=Phase::failed;return false;}
        slot.lease.clear_completed();slot.phase=Phase::idle;return true;
    }
    bool cancel(Ticket ticket) noexcept {
        if(ticket.owner_identity!=identity_ || ticket.slot>=slots_.size())return false;
        auto& slot=slots_[ticket.slot];
        if(slot.generation!=ticket.generation || slot.phase==Phase::idle || slot.phase==Phase::failed)return false;
        slot.cancelled=true;
        // Only a witnessed producer with no consumer can retire immediately.
        // Consuming stages still require their actual final-consumer event.
        if(slot.phase==Phase::ready) {
            // No consumer was submitted. producer_finished already validated
            // this exact producer event, so make it the recorded final-use
            // identity rather than falsely certifying the unused consumer event.
            if(!slot.lease.finish_after_alternate_event(ticket.generation,
                slot.producer_event,0)) {slot.phase=Phase::failed;return false;}
            slot.lease.clear_completed();slot.phase=Phase::idle;
        }
        return true;
    }
    // Cancellation after submission does not constitute either completion edge.
    // Destruction of pending/failed records uses the transfer-lease quarantine.
};
}
