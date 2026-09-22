#pragma once
#include "exl3/exact_kv_extent.h"
#include "exl3/retirement_state.h"
#include "exl3/retained_descriptor_ledger.h"
#include "exl3/registered_kv_backing.h"
#include "exl3/kv_registration_cache.h"
#include <atomic>
#include <memory>
#include <optional>

namespace ninfer::exl3 {
// Construct under metadata reservation before submission. Registration is an
// independently acquired owner, not a promise inferred from page retention or
// a Windows lock. This record does not itself certify registration coverage.
class Exl3KVTransferLease {
    struct State {
        std::optional<RetainedDescriptorLedger::Ticket> retirement_credit;
        std::optional<Exl3ExactKVExtent> source;
        std::shared_ptr<const void> destination,registration,source_group;
        std::optional<Exl3RegisteredKVBacking::Reader> registered_reader;
        std::optional<Exl3KVRegistrationCache::ExternalRead> external_read;
        Exl3FinalUseWitness final_use;
        std::uint64_t acquisition=0,execution=0;
        bool submitted=false;
        State* next=nullptr;
        State()=default;
        State(Exl3ExactKVExtent extent,std::shared_ptr<const void> output,std::shared_ptr<const void> registered)
            :source(std::move(extent)),destination(std::move(output)),registration(std::move(registered)){}
    };
    struct DeleteState {
        void operator()(State* state) const noexcept {
            if(!state)return;
            auto credit=std::move(state->retirement_credit);
            delete state; // Dependencies and allocation retire before the charge.
        }
    };
    std::unique_ptr<State,DeleteState> state_;
    inline static std::atomic<State*> quarantined_{nullptr};
    inline static std::atomic<std::uint64_t> quarantined_records_{0};
public:
    static constexpr std::size_t metadata_bytes() noexcept {return sizeof(State)+sizeof(Exl3KVTransferLease);}
    static constexpr std::size_t state_metadata_bytes() noexcept {return sizeof(State);}
    bool can_attach_retirement_credit() const noexcept {return state_ && !state_->retirement_credit;}
    bool attach_retirement_credit(RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!can_attach_retirement_credit() || credit.bytes()!=sizeof(State))return false;
        state_->retirement_credit.emplace(std::move(credit));return true;
    }
    // Idle startup allocation needs no fabricated page or destination owner.
    Exl3KVTransferLease():state_(new State){}
    Exl3KVTransferLease(Exl3ExactKVExtent source,std::shared_ptr<const void> destination,
        std::shared_ptr<const void> registration={}) {
        if(!destination || !destination.use_count() || !source.bytes() || !source.backing_current())
            throw std::invalid_argument("KV transfer missing current source/destination owner");
        if(registration && !registration.use_count())
            throw std::invalid_argument("KV transfer registration has no owner");
        state_.reset(new State(std::move(source),std::move(destination),std::move(registration)));
    }
    Exl3KVTransferLease(const Exl3KVTransferLease&)=delete;
    Exl3KVTransferLease& operator=(const Exl3KVTransferLease&)=delete;
    // Reuse the precredited record only after its previous final-use witness.
    // Preserve the witness generation across bindings, including event-address
    // reuse. All validation precedes replacing any retained physical owner.
    void rebind(Exl3ExactKVExtent source,std::shared_ptr<const void> destination,
        std::shared_ptr<const void> registration={},std::shared_ptr<const void> source_group={},
        std::optional<Exl3KVRegistrationCache::ExternalRead> external_read={}) {
        if(state_->submitted && !complete())
            throw std::logic_error("KV transfer rebind before final use");
        if(!destination || !destination.use_count() || !source.bytes() || !source.backing_current())
            throw std::invalid_argument("KV transfer rebind missing current source/destination owner");
        if((registration && !registration.use_count()) || (source_group && !source_group.use_count()))
            throw std::invalid_argument("KV transfer dependency has no owner");
        if(external_read && !external_read->valid())
            throw std::invalid_argument("KV transfer external reader was moved from");
        state_->source=std::move(source);
        state_->destination=std::move(destination);
        state_->registration=std::move(registration);
        state_->source_group=std::move(source_group);
        state_->registered_reader.reset();
        state_->external_read.reset();
        if(external_read)state_->external_read.emplace(std::move(*external_read));
        state_->acquisition=0;state_->execution=0;state_->submitted=false;
    }
    void rebind_registered(Exl3ExactKVExtent source,std::shared_ptr<const void> destination,
        Exl3RegisteredKVBacking::Reader reader) {
        if(!reader.owner() || !reader.owner()->covers(source))
            throw std::invalid_argument("KV transfer registration reader does not cover source");
        rebind(std::move(source),std::move(destination),reader.owner());
        state_->registered_reader.emplace(std::move(reader));
    }
    void clear_completed() {
        if(state_->submitted && !complete())throw std::logic_error("KV transfer clear before final use");
        state_->source.reset();state_->destination.reset();state_->registration.reset();state_->source_group.reset();
        state_->registered_reader.reset();
        state_->external_read.reset();
        state_->acquisition=0;state_->execution=0;state_->submitted=false;
    }
    ~Exl3KVTransferLease() {
        if(state_->submitted && !complete()) {
            // No allocation, callback, unregistration or owner destruction on
            // uncertain completion. Keep the entire record for process lifetime.
            auto* retained=state_.release();auto* head=quarantined_.load();
            do{retained->next=head;}while(!quarantined_.compare_exchange_weak(head,retained));
            ++quarantined_records_;
        }
    }
    std::uint64_t begin(std::uint64_t acquisition,std::uint64_t execution,std::uintptr_t event) {
        if(state_->submitted || !state_->source || !state_->source->backing_current())throw std::logic_error("KV transfer reuse/stale backing");
        const auto generation=state_->final_use.begin(acquisition,execution,event);
        state_->acquisition=acquisition;state_->execution=execution;state_->submitted=true;
        return generation;
    }
    bool finish(std::uint64_t generation,int event_error) noexcept {
        return state_->submitted && state_->final_use.finish(generation,event_error);
    }
    bool finish_after_alternate_event(std::uint64_t generation,
        std::uintptr_t witnessed_event,int event_error) noexcept {
        return state_->submitted && state_->final_use.finish_after_alternate_event(
            generation,witnessed_event,event_error);
    }
    Exl3FinalUseWitness::Snapshot final_use_snapshot_for_test() const noexcept {
        return state_->final_use.snapshot();
    }
    bool complete() const noexcept {
        return state_->submitted && state_->final_use.matches(state_->acquisition,state_->execution,state_->final_use.generation());
    }
    bool uncertain() const noexcept {return state_->submitted && !complete();}
    const Exl3ExactKVExtent& source() const {
        if(!state_->source)throw std::logic_error("KV transfer source unavailable");
        return *state_->source;
    }
    static std::uint64_t quarantined_records() noexcept {return quarantined_records_.load();}
};
}
