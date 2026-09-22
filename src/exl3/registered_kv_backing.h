#pragma once
#include "exl3/exact_kv_extent.h"
#include "exl3/resource_inventory.h"
#include "exl3/bounded_shared_owner.h"
#include <cuda_runtime.h>
#include <atomic>
#include <memory>

namespace ninfer::exl3 {
struct Exl3KVRegistrationProvider {
    cudaError_t (*acquire)(void*,std::size_t) noexcept=[](void* pointer,std::size_t bytes) noexcept {
        return cudaHostRegister(pointer,bytes,cudaHostRegisterDefault);
    };
    cudaError_t (*release)(void*) noexcept=[](void* pointer) noexcept {return cudaHostUnregister(pointer);};
};
// CUDA transfer registration only. Published backing must remain immutable;
// Windows residency continues to belong exclusively to HostResidentSet.
class Exl3RegisteredKVBacking {
    struct State {
        std::optional<RetainedCudaRegistrationLedger::Ticket> registration_credit;
        std::optional<RetainedDescriptorLedger::Ticket> retirement_credit;
        Exl3ExactKVExtent extent;
        Exl3ExactKVExtent::Backing backing;
        Exl3KVRegistrationProvider provider;
        std::atomic<bool> registered{false};
        std::atomic<std::uint64_t> readers{0};
        int retirement_error=0;
        State* next=nullptr;
        State(Exl3ExactKVExtent source,Exl3KVRegistrationProvider callbacks)
            :extent(std::move(source)),backing(extent.backing()),provider(callbacks){}
    };
    struct DeleteState {
        void operator()(State* state) const noexcept {
            if(!state)return;
            auto credit=std::move(state->retirement_credit);
            delete state;
        }
    };
    std::unique_ptr<State,DeleteState> state_;
    std::optional<RetainedDescriptorLedger::Ticket> constructor_metadata_credit_;
    inline static std::atomic<State*> quarantined_{nullptr};
    inline static std::atomic<std::uint64_t> quarantined_bytes_{0};
    struct RegistrationFailure {};
    struct ConstructionKey {};
    static constexpr std::uint64_t sealed=std::uint64_t{1}<<63;
    static constexpr std::uint64_t retiring=sealed+1,retired=sealed+2,failed=sealed+3;
public:
    class Reader {
        friend class Exl3RegisteredKVBacking;
        std::shared_ptr<Exl3RegisteredKVBacking> owner_;
        explicit Reader(std::shared_ptr<Exl3RegisteredKVBacking> owner) noexcept:owner_(std::move(owner)){}
    public:
        Reader(const Reader&)=delete;
        Reader& operator=(const Reader&)=delete;
        Reader(Reader&& other) noexcept:owner_(std::move(other.owner_)){}
        ~Reader(){if(owner_)owner_->state_->readers.fetch_sub(1,std::memory_order_acq_rel);}
        const std::shared_ptr<Exl3RegisteredKVBacking>& owner() const noexcept {return owner_;}
    };
    static std::optional<Reader> acquire_reader(std::shared_ptr<Exl3RegisteredKVBacking> owner,
        const Exl3ExactKVExtent& extent) {
        if(!owner || !owner.use_count() || !owner->covers(extent))return std::nullopt;
        auto observed=owner->state_->readers.load(std::memory_order_acquire);
        while(observed<sealed-1) {
            if(owner->state_->readers.compare_exchange_weak(observed,observed+1,std::memory_order_acq_rel))
                return Reader(std::move(owner));
        }
        return std::nullopt;
    }
    std::uint64_t active_readers() const noexcept {
        const auto value=state_->readers.load(std::memory_order_acquire);
        return value<sealed?value:0;
    }
    bool seal_idle() noexcept {
        std::uint64_t expected=0;
        return state_->readers.compare_exchange_strong(expected,sealed,std::memory_order_acq_rel) ||
            expected==sealed || expected==retired;
    }
    bool retired_successfully() const noexcept {return state_->readers.load(std::memory_order_acquire)==retired;}
    std::size_t registered_bytes() const noexcept {return state_->backing.bytes;}
    bool retire_sealed() noexcept {
        std::uint64_t expected=sealed;
        if(!state_->readers.compare_exchange_strong(expected,retiring,std::memory_order_acq_rel))return expected==retired;
        if(state_->registered.load(std::memory_order_acquire)) {
            const auto error=state_->provider.release(const_cast<std::uint16_t*>(state_->backing.data));
            if(error!=cudaSuccess) {
                state_->retirement_error=static_cast<int>(error);
                quarantined_bytes_.fetch_add(state_->backing.bytes);
                state_->readers.store(failed,std::memory_order_release);return false;
            }
            state_->registered.store(false,std::memory_order_release);
            state_->registration_credit.reset();
        }
        state_->readers.store(retired,std::memory_order_release);return true;
    }
    Exl3RegisteredKVBacking(ConstructionKey,Exl3ExactKVExtent extent,Exl3KVRegistrationProvider provider)
        :state_(new State(std::move(extent),provider)){}
    static constexpr std::size_t state_metadata_bytes() noexcept {return sizeof(State);}
    static bool attach_registration_credit(const std::shared_ptr<const void>& owner,
        RetainedCudaRegistrationLedger::Ticket credit) noexcept {
        if(!owner)return false;
        auto* registration=const_cast<Exl3RegisteredKVBacking*>(static_cast<const Exl3RegisteredKVBacking*>(owner.get()));
        if(!registration->state_->registered.load(std::memory_order_acquire) ||
            registration->state_->registration_credit || credit.bytes()!=registration->state_->backing.bytes)return false;
        registration->state_->registration_credit.emplace(std::move(credit));return true;
    }
    static constexpr std::size_t metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3RegisteredKVBacking>()+sizeof(State);
    }
    static bool attach_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || credit.bytes()!=metadata_bytes() ||
           !can_attach_bounded_retirement_credit<Exl3RegisteredKVBacking>(owner))return false;
        auto* registration=const_cast<Exl3RegisteredKVBacking*>(static_cast<const Exl3RegisteredKVBacking*>(owner.get()));
        if(!registration->state_ || registration->state_->retirement_credit)return false;
        auto child=credit.split(sizeof(State));
        if(!child)return false;
        registration->state_->retirement_credit.emplace(std::move(*child));
        return attach_bounded_retirement_credit<Exl3RegisteredKVBacking>(owner,std::move(credit));
    }
    Exl3RegisteredKVBacking(const Exl3RegisteredKVBacking&)=delete;
    ~Exl3RegisteredKVBacking() {
        if(!state_->registered)return;
        // Reader owns this wrapper, so final destruction cannot precede its
        // release. A prior explicit failure must never retry unregistration.
        if(!seal_idle() || !retire_sealed()) {
            auto* retained=state_.release();auto* head=quarantined_.load();
            do{retained->next=head;}while(!quarantined_.compare_exchange_weak(head,retained));
        }
    }
    // Authority is the existing coordinator startup reservation, not a caller
    // supplied boolean credit. Factory creates idle owners and submits no copy.
    // Registration failure returns empty; other reservation errors propagate.
    template<class Coordinator>
    static std::shared_ptr<Exl3RegisteredKVBacking> acquire_startup(Coordinator& authority,
        const Exl3ExactKVExtent& extent,Exl3KVRegistrationProvider provider={}) {
        return acquire_reserved([&](const auto& required,auto&& factory,auto&& rollback,auto&& observe) {
            authority.allocate_startup_resources(required,std::forward<decltype(factory)>(factory),
                std::forward<decltype(rollback)>(rollback),std::forward<decltype(observe)>(observe));
        },[&]() noexcept {authority.seal_failed_startup_retirement();},
        [&](auto registration,auto metadata){return authority.reserve_startup_registration_constructor_credits(registration,metadata);},extent,provider);
    }
    template<class Coordinator>
    static std::shared_ptr<Exl3RegisteredKVBacking> acquire_runtime(Coordinator& authority,
        const typename Coordinator::Lease& lease,const Exl3ExactKVExtent& extent,
        Exl3KVRegistrationProvider provider={}) {
        return acquire_reserved([&](const auto& required,auto&& factory,auto&& rollback,auto&& observe) {
            authority.allocate_runtime_resources(lease,required,std::forward<decltype(factory)>(factory),
                std::forward<decltype(rollback)>(rollback),std::forward<decltype(observe)>(observe));
        },[&]() noexcept {authority.seal_failed_startup_retirement();},
        [&](auto registration,auto metadata){return authority.reserve_registration_constructor_credits(registration,metadata);},extent,provider);
    }
private:
    template<class Reserve,class Seal,class Grant>
    static std::shared_ptr<Exl3RegisteredKVBacking> acquire_reserved(Reserve&& reserve,Seal&& seal,Grant&& grant,
        const Exl3ExactKVExtent& extent,Exl3KVRegistrationProvider provider) {
        if(!provider.acquire || !provider.release)throw std::invalid_argument("KV registration provider incomplete");
        if(quarantined_bytes_.load())throw std::runtime_error("unresolved KV registration retirement");
        if(!extent.registration_eligible())return {};
        const auto backing=extent.backing();
        Exl3ResourceInventory::Requirement required;required.configuration=0x4b56524547;
        using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::cuda_registered_host,1,backing.bytes);
        required.add(Domain::host_metadata,1,metadata_bytes());
        std::shared_ptr<Exl3RegisteredKVBacking> result;
        const auto quarantine_before=quarantined_bytes_.load();
        try {
            reserve(required,[&](std::uint64_t configuration) {
                if(configuration!=required.configuration)throw std::logic_error("KV registration reservation identity");
                if(!extent.registration_eligible())
                    throw std::invalid_argument("KV registration backing changed before allocation");
                auto credits=grant(backing.bytes,metadata_bytes());
                if(credits.registration.bytes()!=backing.bytes || credits.metadata.bytes()!=metadata_bytes())
                    throw std::invalid_argument("KV registration constructor credit extent");
                auto owner=make_bounded_shared<Exl3RegisteredKVBacking>(ConstructionKey{},extent,provider);
                owner->state_->registration_credit.emplace(std::move(credits.registration));
                auto state_credit=credits.metadata.split(sizeof(State));
                if(!state_credit)throw std::logic_error("registration constructor metadata extent");
                owner->state_->retirement_credit.emplace(std::move(*state_credit));
                owner->constructor_metadata_credit_.emplace(std::move(credits.metadata));
                Exl3ResourceInventory actual;
                actual.add({owner,0,Domain::cuda_registered_host,backing.bytes,{},nullptr,nullptr,
                    &attach_registration_credit});
                actual.add({owner,1,Domain::host_metadata,metadata_bytes(),{},&attach_retirement_credit});
                // Build fallible metadata first; no consumers exist on this path.
                const auto error=provider.acquire(const_cast<std::uint16_t*>(backing.data),backing.bytes);
                if(error!=cudaSuccess)throw RegistrationFailure{};
                owner->state_->registered=true;result=std::move(owner);return actual;
            },[&]() noexcept {result.reset();},[&]() noexcept {
                if(quarantined_bytes_.load()!=quarantine_before)seal();
            });
        } catch(const RegistrationFailure&){return {};}
        result->state_->registration_credit.reset();
        result->state_->retirement_credit.reset();result->constructor_metadata_credit_.reset();
        return result;
    }
public:
    bool covers(const Exl3ExactKVExtent& extent) const {
        return state_->registered.load(std::memory_order_acquire) &&
            state_->readers.load(std::memory_order_acquire)<sealed && backing_matches(extent);
    }
    bool backing_matches(const Exl3ExactKVExtent& extent) const {
        if(!state_->extent.backing_current() || !extent.backing_current())return false;
        const auto backing=extent.backing();
        const auto& owned=state_->backing;
        const bool same_owner=!owned.owner.owner_before(backing.owner) && !backing.owner.owner_before(owned.owner);
        return same_owner && owned.data==backing.data && owned.bytes==backing.bytes &&
            owned.bank==backing.bank && owned.plane==backing.plane;
    }
    static std::uint64_t quarantined_bytes() noexcept {return quarantined_bytes_.load();}
    bool uses(Exl3KVRegistrationProvider provider) const noexcept {
        return state_->provider.acquire==provider.acquire && state_->provider.release==provider.release;
    }
    bool overlaps(const Exl3ExactKVExtent& extent) const {
        if(retired_successfully())return false;
        if(!extent.backing_current())return false;
        const auto candidate=extent.backing();
        const auto a=reinterpret_cast<std::uintptr_t>(state_->backing.data);
        const auto b=reinterpret_cast<std::uintptr_t>(candidate.data);
        if(!state_->backing.bytes || !candidate.bytes)return false;
        return a<=b?b-a<state_->backing.bytes:a-b<candidate.bytes;
    }
};
}
