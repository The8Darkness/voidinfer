#pragma once
#include "exl3/retirement_state.h"
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>

namespace ninfer::exl3 {
// Fixed request-local logical ownership for precreated completion events. The
// native handles remain owned by the enclosing physical lane. A slot becomes
// reusable only after its exact generation is ready and explicitly retired.
class Exl3RequestCompletionEventPool {
public:
    static constexpr std::size_t capacity=4;
    enum class Phase : std::uint8_t {
        unavailable,retired,acquired,pending,ready,failed
    };
    struct Ticket {
        std::size_t slot=capacity;
        std::uint64_t authority_generation=0,generation=0;
        std::uint64_t acquisition=0,execution=0,retirement_generation=0;
        std::uintptr_t event=0;
    };
    struct Snapshot {
        Phase phase=Phase::unavailable;
        std::uint64_t authority_generation=0,generation=0,acquisition=0,execution=0;
        std::uint64_t retirement_generation=0;
        std::uint64_t final_use_generation=0;
        std::uintptr_t event=0;
        bool cancelled=false;
        int first_error=0;
    };

    void configure(std::span<const std::uintptr_t> events,
        std::uint64_t authority_generation) {
        if(configured_ || events.empty() || events.size()>capacity ||
            !authority_generation)
            throw std::invalid_argument("completion event pool configuration");
        for(std::size_t i=0;i<events.size();++i) {
            if(!events[i])throw std::invalid_argument("completion event identity missing");
            for(std::size_t j=0;j<i;++j)if(events[j]==events[i])
                throw std::invalid_argument("completion event identity reused");
            slots_[i].event=events[i];slots_[i].phase=Phase::retired;
        }
        configured_=events.size();authority_generation_=authority_generation;
    }
    std::optional<Ticket> acquire(std::uint64_t acquisition,
        std::uint64_t execution,std::uint64_t retirement_generation,
        std::shared_ptr<const void> owner) {
        if(!configured_ || !authority_generation_ || !acquisition || !execution ||
            !retirement_generation || !owner)
            throw std::invalid_argument("completion event acquisition scope");
        if(next_generation_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("completion event generation exhausted");
        for(std::size_t i=0;i<configured_;++i) {
            auto& slot=slots_[i];
            if(slot.phase!=Phase::retired)continue;
            slot.owner=std::move(owner);slot.acquisition=acquisition;
            slot.execution=execution;slot.retirement_generation=retirement_generation;
            slot.cancelled=false;
            slot.generation=++next_generation_;slot.final_use_generation=0;
            slot.final_use={};slot.phase=Phase::acquired;
            return Ticket{i,authority_generation_,slot.generation,acquisition,
                execution,retirement_generation,slot.event};
        }
        return {};
    }
    bool submit(Ticket ticket) {
        auto* slot=match(ticket,Phase::acquired);if(!slot)return false;
        try {slot->final_use_generation=slot->final_use.begin(
            slot->acquisition,slot->execution,slot->event);}
        catch(...) {slot->phase=Phase::failed;throw;}
        slot->phase=Phase::pending;return true;
    }
    // The notifier must present both logical generation and native event.
    // Errors are accepted as sticky physical evidence but never as readiness.
    bool notify(Ticket ticket,std::uintptr_t event,int error) noexcept {
        auto* slot=match(ticket,Phase::pending);
        if(!slot || event!=slot->event ||
           !slot->final_use.finish(slot->final_use_generation,error))return false;
        if(error){slot->phase=Phase::failed;return false;}
        if(slot->cancelled){retire_slot(*slot);return true;}
        slot->phase=Phase::ready;return true;
    }
    bool cancel(Ticket ticket) noexcept {
        auto* slot=match_any(ticket);if(!slot)return false;
        if(slot->phase==Phase::acquired){retire_slot(*slot);return true;}
        if(slot->phase==Phase::pending){slot->cancelled=true;return true;}
        if(slot->phase==Phase::ready){retire_slot(*slot);return true;}
        return false;
    }
    bool retire(std::uint64_t generation) noexcept {
        auto* slot=find(generation);
        if(!slot || slot->phase!=Phase::ready ||
           !slot->final_use.matches(slot->acquisition,slot->execution,
               slot->final_use_generation))return false;
        retire_slot(*slot);return true;
    }
    bool matches(std::uint64_t authority_generation,std::uint64_t acquisition,
        std::uint64_t execution,std::uint64_t retirement_generation,
        std::uint64_t generation) const noexcept {
        const auto* slot=find(generation);
        return slot && slot->phase==Phase::ready &&
            authority_generation==authority_generation_ &&
            slot->acquisition==acquisition && slot->execution==execution &&
            slot->retirement_generation==retirement_generation &&
            slot->final_use.matches(acquisition,execution,
                slot->final_use_generation);
    }
    std::uintptr_t event(std::uint64_t generation) const noexcept {
        const auto* slot=find(generation);return slot?slot->event:0;
    }
    Snapshot snapshot(std::size_t index) const noexcept {
        if(index>=configured_)return {};
        const auto& slot=slots_[index];
        return {slot.phase,authority_generation_,slot.generation,slot.acquisition,slot.execution,
            slot.retirement_generation,
            slot.final_use_generation,slot.event,slot.cancelled,
            slot.final_use.first_error()};
    }
    Snapshot latest_snapshot() const noexcept {
        const Slot* latest=nullptr;
        for(std::size_t i=0;i<configured_;++i)
            if(!latest || slots_[i].generation>latest->generation)latest=&slots_[i];
        if(!latest)return {};
        return {latest->phase,authority_generation_,latest->generation,latest->acquisition,
            latest->execution,latest->retirement_generation,
            latest->final_use_generation,latest->event,
            latest->cancelled,latest->final_use.first_error()};
    }
    int first_error() const noexcept {
        int result=0;
        for(std::size_t i=0;i<configured_;++i)
            if(!result)result=slots_[i].final_use.first_error();
        return result;
    }
    // Caller has established a stronger all-device drain. Logical callbacks
    // remain stale, but request owners can now be dropped before native handle
    // destruction. This is teardown only and does not make slots reusable.
    void retire_all_after_device_drain() noexcept {
        for(std::size_t i=0;i<configured_;++i) {
            auto& slot=slots_[i];
            slot.owner.reset();
            if(slot.phase!=Phase::unavailable)slot.phase=Phase::retired;
        }
    }
    // One-way source-test seam. Exhaustion is a refusal, never a wrap to a
    // generation which a delayed notification could still carry.
    void exhaust_generation_for_test() noexcept {
        next_generation_=std::numeric_limits<std::uint64_t>::max();
    }
private:
    struct Slot {
        Exl3FinalUseWitness final_use;
        std::shared_ptr<const void> owner;
        std::uint64_t generation=0,acquisition=0,execution=0,retirement_generation=0;
        std::uint64_t final_use_generation=0;
        std::uintptr_t event=0;
        Phase phase=Phase::unavailable;
        bool cancelled=false;
    };
    Slot* match(Ticket ticket,Phase phase) noexcept {
        if(ticket.slot>=configured_)return nullptr;
        auto& slot=slots_[ticket.slot];
        return ticket.authority_generation==authority_generation_ &&
            slot.generation==ticket.generation && slot.event==ticket.event &&
            slot.acquisition==ticket.acquisition && slot.execution==ticket.execution &&
            slot.retirement_generation==ticket.retirement_generation &&
            slot.phase==phase?&slot:nullptr;
    }
    Slot* match_any(Ticket ticket) noexcept {
        if(ticket.slot>=configured_)return nullptr;
        auto& slot=slots_[ticket.slot];
        return ticket.authority_generation==authority_generation_ &&
            slot.generation==ticket.generation && slot.event==ticket.event &&
            slot.acquisition==ticket.acquisition && slot.execution==ticket.execution &&
            slot.retirement_generation==ticket.retirement_generation?&slot:nullptr;
    }
    Slot* find(std::uint64_t generation) noexcept {
        for(std::size_t i=0;i<configured_;++i)
            if(slots_[i].generation==generation)return &slots_[i];
        return nullptr;
    }
    const Slot* find(std::uint64_t generation) const noexcept {
        for(std::size_t i=0;i<configured_;++i)
            if(slots_[i].generation==generation)return &slots_[i];
        return nullptr;
    }
    static void retire_slot(Slot& slot) noexcept {
        slot.owner.reset();slot.cancelled=false;slot.phase=Phase::retired;
    }
    std::array<Slot,capacity> slots_{};
    std::size_t configured_=0;
    std::uint64_t authority_generation_=0,next_generation_=0;
};
}
