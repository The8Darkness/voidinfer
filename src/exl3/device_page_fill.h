#pragma once
#include "exl3/device_page_key.h"
#include "exl3/retirement_state.h"
#include "exl3/kv_registration_cache.h"
#include "exl3/bounded_shared_owner.h"
#include <atomic>
#include <span>
#include <mutex>

namespace ninfer::exl3 {
// Producer supplies exclusively owned contiguous device storage under physical
// credit. This protocol never allocates device memory or publishes partial data.
class Exl3DevicePageFill {
public:
    using StorageRetirement=bool(*)(const std::shared_ptr<const void>&) noexcept;
    static constexpr std::size_t plane_elements=64*1024;
    static constexpr std::size_t elements=32*plane_elements;
    class View {
        Exl3DevicePageKey key_;
        std::shared_ptr<const void> storage_;
        const std::uint16_t* data_;
        std::uint64_t generation_;
        std::shared_ptr<std::atomic<std::uint64_t>> readers_;
        struct ConstructionKey {};
        friend class Exl3DevicePageFill;
        friend class Exl3DevicePageReader;
    public:
        View(ConstructionKey,const Exl3DevicePageKey& key,std::shared_ptr<const void> storage,
            const std::uint16_t* data,std::uint64_t generation,bool fail_after_counter_for_test=false)
            :key_(key),storage_(std::move(storage)),data_(data),generation_(generation),
             readers_(make_bounded_shared<std::atomic<std::uint64_t>>(0)) {
            if(fail_after_counter_for_test)throw std::runtime_error("injected page view after-counter failure");
        }
        const Exl3DevicePageKey& key() const noexcept {return key_;}
        std::uint64_t generation() const noexcept {return generation_;}
        std::span<const std::uint16_t> plane(int bank,bool key) const {
            if(bank<0 || bank>=16 || !key_.current())throw std::invalid_argument("device page view bank/backing");
            return {data_+(bank*2+(key?0:1))*plane_elements,plane_elements};
        }
    };
private:
    struct State {
        Exl3DevicePageKey key;
        std::shared_ptr<const void> storage;
        std::shared_ptr<const void> event_owner;
        std::optional<Exl3KVRegistrationCache::ExternalRead> registration_read;
        std::shared_ptr<View> view;
        const std::uint16_t* data;
        Exl3FinalUseWitness ready;
        std::uint64_t acquisition=0,execution=0;
        std::uint32_t planes=0;
        bool submitted=false;
        std::atomic<bool> published{false};
        std::atomic<int> failure{0};
        std::atomic<bool> sealed{false};
        mutable std::mutex access;
        StorageRetirement retire=nullptr;
        bool reclaimed=false;
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        State* next=nullptr;
        State(Exl3DevicePageKey k,std::shared_ptr<const void> owner,const std::uint16_t* p)
            :key(std::move(k)),storage(std::move(owner)),data(p){}
    };
    std::unique_ptr<State> state_;
    inline static std::atomic<State*> quarantine_{nullptr};
    inline static std::atomic<std::uint64_t> quarantine_count_{0};
public:
    static constexpr std::size_t metadata_bytes() noexcept {
        return sizeof(State)+bounded_shared_allocation_bytes<Exl3DevicePageFill>()+
            bounded_shared_allocation_bytes<View>()+bounded_shared_allocation_bytes<std::atomic<std::uint64_t>>();
    }
    static constexpr std::size_t retired_fill_metadata_bytes() noexcept {
        return sizeof(State)+bounded_shared_allocation_bytes<Exl3DevicePageFill>();
    }
    // Own fill/view/counter metadata only; storage has its own retirement owner.
    // Attach while all components still exist, before sealed physical retirement.
    std::shared_ptr<const void> storage_owner_for_retirement() const {
        std::lock_guard<std::mutex> lock(state_->access);return state_->storage;
    }
    static bool can_attach_own_metadata_credit(const std::shared_ptr<const void>& owner) noexcept {
        const auto* fill=static_cast<const Exl3DevicePageFill*>(owner.get());
        if(!fill)return false;
        std::lock_guard<std::mutex> lock(fill->state_->access);
        const auto& state=*fill->state_;
        return !state.metadata_credit && state.view && state.view->readers_ &&
            can_attach_bounded_retirement_credit<Exl3DevicePageFill>(owner) &&
            can_attach_bounded_retirement_credit<View>(state.view) &&
            can_attach_bounded_retirement_credit<std::atomic<std::uint64_t>>(state.view->readers_);
    }
    static bool attach_own_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* fill=const_cast<Exl3DevicePageFill*>(static_cast<const Exl3DevicePageFill*>(owner.get()));
        if(!fill || credit.bytes()!=metadata_bytes())return false;
        std::lock_guard<std::mutex> lock(fill->state_->access);
        auto& state=*fill->state_;
        if(state.metadata_credit || !state.view || !state.view->readers_ ||
            !can_attach_bounded_retirement_credit<Exl3DevicePageFill>(owner) ||
            !can_attach_bounded_retirement_credit<View>(state.view) ||
            !can_attach_bounded_retirement_credit<std::atomic<std::uint64_t>>(state.view->readers_))return false;
        auto fill_credit=credit.split(bounded_shared_allocation_bytes<Exl3DevicePageFill>());
        auto view_credit=credit.split(bounded_shared_allocation_bytes<View>());
        auto reader_credit=credit.split(bounded_shared_allocation_bytes<std::atomic<std::uint64_t>>());
        if(!fill_credit || !view_credit || !reader_credit || credit.bytes()!=sizeof(State))return false;
        if(!attach_bounded_retirement_credit<Exl3DevicePageFill>(owner,std::move(*fill_credit)) ||
            !attach_bounded_retirement_credit<View>(state.view,std::move(*view_credit)) ||
            !attach_bounded_retirement_credit<std::atomic<std::uint64_t>>(state.view->readers_,std::move(*reader_credit)))return false;
        state.metadata_credit.emplace(std::move(credit));return true;
    }
    std::uint64_t retained_readers() const {
        std::lock_guard<std::mutex> lock(state_->access);
        return state_->view?state_->view->readers_->load(std::memory_order_acquire):0;
    }
    bool fill_in_flight() const {
        std::lock_guard<std::mutex> lock(state_->access);
        return state_->submitted && !state_->published.load(std::memory_order_acquire) &&
            !state_->failure.load(std::memory_order_acquire);
    }
    bool abandoned_before_submission() const {
        std::lock_guard<std::mutex> lock(state_->access);
        return state_->failure.load(std::memory_order_acquire)!=0 && !state_->submitted;
    }
    Exl3DevicePageFill(Exl3DevicePageKey key,std::shared_ptr<const void> storage,
        std::span<const std::uint16_t> extent,StorageRetirement retire=nullptr,unsigned constructor_fault_for_test=0) {
        if(constructor_fault_for_test>3)throw std::invalid_argument("page fill constructor fault0..3");
        if(!storage || !storage.use_count() || !extent.data() || extent.size()!=elements || !key.current())
            throw std::invalid_argument("device page fill storage/key extent");
        state_=std::make_unique<State>(std::move(key),std::move(storage),extent.data());
        state_->retire=retire;
        if(constructor_fault_for_test==1)throw std::runtime_error("injected page fill after-state failure");
        // Allocate the immutable consumer handle inside the fill's metadata
        // reservation. Its generation becomes visible only at ready publication.
        state_->view=make_bounded_shared<View>(View::ConstructionKey{},state_->key,state_->storage,state_->data,0,
            constructor_fault_for_test==2);
        if(constructor_fault_for_test==3)throw std::runtime_error("injected page fill after-view failure");
    }
    Exl3DevicePageFill(const Exl3DevicePageFill&)=delete;
    ~Exl3DevicePageFill() {
        if(state_->submitted && !state_->published.load(std::memory_order_acquire)) {
            auto* retained=state_.release();auto* head=quarantine_.load();
            do{retained->next=head;}while(!quarantine_.compare_exchange_weak(head,retained));
            ++quarantine_count_;
        }
    }
    std::uint64_t begin(std::uint64_t acquisition,std::uint64_t execution,std::uintptr_t event,
        std::shared_ptr<const void> event_owner,
        std::optional<Exl3KVRegistrationCache::ExternalRead> registration_read={}) {
        std::lock_guard<std::mutex> lock(state_->access);
        if(state_->submitted || failed() || !state_->key.current() || !event_owner || !event_owner.use_count())
            throw std::logic_error("device page fill already submitted/stale/missing event owner");
        if(registration_read && !registration_read->valid())
            throw std::invalid_argument("device page fill registration reader was moved from");
        const auto generation=state_->ready.begin(acquisition,execution,event);
        state_->event_owner=std::move(event_owner);
        if(registration_read)state_->registration_read.emplace(std::move(*registration_read));
        state_->acquisition=acquisition;state_->execution=execution;state_->submitted=true;return generation;
    }
    void plane_submitted(std::uint64_t generation,int bank,bool key) {
        std::lock_guard<std::mutex> lock(state_->access);
        if(!state_->submitted || generation!=state_->ready.generation() || bank<0 || bank>=16 ||
            state_->ready.first_error() || ready())throw std::logic_error("device page plane submission scope");
        const auto bit=std::uint32_t{1}<<(bank*2+(key?0:1));
        if(state_->planes&bit)throw std::logic_error("device page plane submitted twice");
        state_->planes|=bit;
    }
    bool finish(std::uint64_t generation,int error) noexcept {
        std::shared_ptr<const void> retired_event;
        std::optional<Exl3KVRegistrationCache::ExternalRead> retired_registration;
        std::lock_guard<std::mutex> lock(state_->access);
        if(!state_->submitted || failed() || (!error && state_->planes!=UINT32_MAX))return false;
        const bool accepted=state_->ready.finish(generation,error);
        if(accepted) {
            if(error)state_->failure.store(error,std::memory_order_release);
            else {
                retired_event=std::move(state_->event_owner);
                if(state_->registration_read) {
                    retired_registration.emplace(std::move(*state_->registration_read));
                    state_->registration_read.reset();
                }
                state_->view->generation_=generation;
                state_->published.store(true,std::memory_order_release);
            }
        }
        // The lock is destroyed before detached owners. Their final deleters
        // may inspect this fill and must see its complete published state.
        return accepted;
    }
    bool ready() const noexcept {
        // One producer owns all mutable fill fields. Consumers inspect only the
        // release-published flag before immutable key/data/generation access.
        return !state_->sealed.load(std::memory_order_acquire) &&
            state_->published.load(std::memory_order_acquire) && state_->key.current();
    }
    bool failed() const noexcept {return state_->failure.load(std::memory_order_acquire)!=0;}
    bool sealed_for_retirement() const noexcept {return state_->sealed.load(std::memory_order_acquire);}
    // Owning shutdown calls only after producer threads have joined.
    bool uncertain_after_join() const noexcept {return state_->submitted && !state_->published.load(std::memory_order_acquire);}
    int first_error() const noexcept {return state_->failure.load(std::memory_order_acquire);}
    const Exl3DevicePageKey& key() const noexcept {return state_->key;}
    bool owns_destination(std::span<const std::uint16_t> extent) const noexcept {
        return extent.data()==state_->data && extent.size()==elements;
    }
    // Only the unique producer calls this when dropping an unfinished claim.
    // Pending storage remains retained; waiters cannot turn it into a ready hit.
    void abandon_producer() noexcept {
        std::lock_guard<std::mutex> lock(state_->access);
        if(state_->published.load(std::memory_order_acquire))return;
        if(state_->submitted)state_->ready.finish(state_->ready.generation(),-1);
        int none=0;state_->failure.compare_exchange_strong(none,-1,std::memory_order_release);
    }
    View view() const {
        std::lock_guard<std::mutex> lock(state_->access);
        if(!ready())throw std::logic_error("device page unavailable before complete readiness");
        // The published view already owns the reserved reader counter. A value
        // copy retains that counter and storage without allocating another one.
        // Readers bound through a copied view must contribute to the same count.
        return *state_->view;
    }
    std::shared_ptr<const View> ready_handle() const {
        std::lock_guard<std::mutex> lock(state_->access);
        if(!ready())return {};
        return state_->view;
    }
    bool seal_for_retirement() {
        std::lock_guard<std::mutex> lock(state_->access);
        const bool abandoned_unsubmitted=failed() && !state_->submitted;
        if(state_->sealed.load() ||
            (!state_->published.load(std::memory_order_acquire) && !abandoned_unsubmitted))return false;
        // The fill and its preallocated View are the two internal storage owners.
        // A value View adds a storage owner; a shared View adds a handle owner.
        // Both acquisition APIs hold this mutex, closing the late-reader race.
        if(state_->view.use_count()!=1 || state_->storage.use_count()!=2)return false;
        state_->sealed.store(true,std::memory_order_release);return true;
    }
    bool retire_sealed_storage() noexcept {
        std::shared_ptr<View> retired_view;
        std::shared_ptr<const void> retired_storage;
        std::lock_guard<std::mutex> lock(state_->access);
        if(!state_->sealed.load() || state_->reclaimed || !state_->retire ||
            state_->view.use_count()!=1 || state_->storage.use_count()!=2)return false;
        if(!state_->retire(state_->storage))return false;
        state_->reclaimed=true;state_->data=nullptr;
        retired_view=std::move(state_->view);
        retired_storage=std::move(state_->storage);
        return true; // Drop final owner references only after unlocking access.
    }
    static std::uint64_t quarantined_records() noexcept {return quarantine_count_.load();}
};
}
