#pragma once
#include "exl3/device_page_fill.h"
#include "exl3/resource_inventory.h"
#include <cuda_runtime.h>

namespace ninfer::exl3 {
struct Exl3DevicePageStorageProvider {
    cudaError_t (*allocate)(void**,std::size_t) noexcept=
        [](void** pointer,std::size_t bytes) noexcept {return cudaMalloc(pointer,bytes);};
    cudaError_t (*release)(void*) noexcept=[](void* pointer) noexcept {return cudaFree(pointer);};
};
class Exl3DevicePageStorage {
    struct State {
        void* data=nullptr;
        Exl3DevicePageStorageProvider provider;
        bool failed=false;
        std::optional<RetainedDeviceLedger::Ticket> device_credit;
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        State* next=nullptr;
        explicit State(Exl3DevicePageStorageProvider p):provider(p){}
    };
    std::unique_ptr<State> state_;
    std::optional<RetainedDescriptorLedger::Ticket> constructor_metadata_credit_;
    inline static std::atomic<State*> quarantine_{nullptr};
    inline static std::atomic<std::uint64_t> quarantine_bytes_{0};
    struct ConstructionKey {};
public:
    explicit Exl3DevicePageStorage(ConstructionKey,Exl3DevicePageStorageProvider provider)
        :state_(std::make_unique<State>(provider)){}
    static constexpr std::size_t bytes=Exl3DevicePageFill::elements*sizeof(std::uint16_t);
    static constexpr std::size_t metadata_bytes() noexcept {
        return sizeof(State)+bounded_shared_allocation_bytes<Exl3DevicePageStorage>()+Exl3DevicePageFill::metadata_bytes();
    }
    static bool attach_device_credit(const std::shared_ptr<const void>& fill_owner,RetainedDeviceLedger::Ticket credit) noexcept {
        const auto* fill=static_cast<const Exl3DevicePageFill*>(fill_owner.get());
        if(!fill || credit.bytes()!=bytes)return false;
        const auto storage=fill->storage_owner_for_retirement();
        auto* owner=const_cast<Exl3DevicePageStorage*>(static_cast<const Exl3DevicePageStorage*>(storage.get()));
        if(!owner || !owner->state_->data || owner->state_->device_credit)return false;
        owner->state_->device_credit.emplace(std::move(credit));return true;
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& fill_owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        const auto* fill=static_cast<const Exl3DevicePageFill*>(fill_owner.get());
        if(!fill || credit.bytes()!=metadata_bytes() || !Exl3DevicePageFill::can_attach_own_metadata_credit(fill_owner))return false;
        const auto storage=fill->storage_owner_for_retirement();
        auto* owner=const_cast<Exl3DevicePageStorage*>(static_cast<const Exl3DevicePageStorage*>(storage.get()));
        if(!owner || owner->state_->metadata_credit || owner->constructor_metadata_credit_ ||
            !can_attach_bounded_retirement_credit<Exl3DevicePageStorage>(storage))return false;
        auto state_credit=credit.split(sizeof(State));
        auto block_credit=credit.split(bounded_shared_allocation_bytes<Exl3DevicePageStorage>());
        if(!state_credit || !block_credit || credit.bytes()!=Exl3DevicePageFill::metadata_bytes())return false;
        if(!Exl3DevicePageFill::attach_own_metadata_credit(fill_owner,std::move(credit)) ||
            !attach_bounded_retirement_credit<Exl3DevicePageStorage>(storage,std::move(*block_credit)))return false;
        owner->state_->metadata_credit.emplace(std::move(*state_credit));return true;
    }
    static Exl3ResourceInventory::Allocation device_allocation(const std::shared_ptr<Exl3DevicePageFill>& fill) {
        return {fill,0,Exl3ResourceInventory::Domain::device,bytes,{},nullptr,&attach_device_credit};
    }
    static Exl3ResourceInventory::Allocation metadata_allocation(const std::shared_ptr<Exl3DevicePageFill>& fill) {
        return {fill,1,Exl3ResourceInventory::Domain::host_metadata,metadata_bytes(),{},&attach_metadata_credit};
    }
    struct Prepared {
        std::shared_ptr<Exl3DevicePageFill> fill;
        // Producer-only destination. Do not expose to cache readers; publication
        // uses fill->view() after every plane and the readiness event complete.
        std::span<std::uint16_t> destination;
    };
    bool retire() const noexcept {
        if(state_->failed || !state_->data)return false;
        if(state_->provider.release(state_->data)!=cudaSuccess) {
            state_->failed=true;quarantine_bytes_.fetch_add(bytes);return false;
        }
        state_->data=nullptr;state_->device_credit.reset();return true;
    }
    ~Exl3DevicePageStorage() {
        if(state_->data && (state_->failed || !retire())) {
            auto* retained=state_.release();auto* head=quarantine_.load();
            do{retained->next=head;}while(!quarantine_.compare_exchange_weak(head,retained));
        }
    }
    static std::uint64_t quarantined_bytes() noexcept {return quarantine_bytes_.load();}
    static std::array<std::uint64_t,2> retained_constructor_credits_for_test() noexcept {
        const auto* state=quarantine_.load();
        return state?std::array<std::uint64_t,2>{state->device_credit?state->device_credit->bytes():0,
            state->metadata_credit?state->metadata_credit->bytes():0}:std::array<std::uint64_t,2>{};
    }
    static constexpr std::size_t retirement_metadata_bytes() noexcept {return sizeof(State);}
    template<class Coordinator>
    static Prepared create_runtime(Coordinator& authority,const typename Coordinator::Lease& lease,
        Exl3DevicePageKey key,Exl3DevicePageStorageProvider provider={},unsigned constructor_fault_for_test=0,
        unsigned* pending_constructor_fault_for_test=nullptr) {
        if(constructor_fault_for_test>4)throw std::invalid_argument("device page constructor fault0..4");
        if(!key.current() || !provider.allocate || !provider.release)
            throw std::invalid_argument("device page allocation key/provider");
        if(quarantined_bytes() || Exl3DevicePageFill::quarantined_records())
            throw std::runtime_error("unresolved device page retirement");
        using Inventory=Exl3ResourceInventory;
        constexpr auto metadata=metadata_bytes();
        Inventory::Requirement required;required.configuration=0x4456504147;
        required.add(Inventory::Domain::device,1,bytes);
        required.add(Inventory::Domain::host_metadata,1,metadata);
        Prepared result;
        std::shared_ptr<Exl3DevicePageStorage> physical_owner;
        const auto quarantine_before=quarantined_bytes();
        try{authority.allocate_runtime_resources(lease,required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("device page allocation identity");
            // Reservation may defer this factory. Reject changed source geometry
            // before invoking the physical allocator; publication remains immutable.
            if(!key.current())throw std::invalid_argument("device page source changed before allocation");
            auto credits=authority.reserve_runtime_constructor_credits(bytes,metadata);
            if(credits.device.bytes()!=bytes || credits.metadata.bytes()!=metadata)
                throw std::invalid_argument("device page constructor credit extent");
            auto owner=make_bounded_shared<Exl3DevicePageStorage>(ConstructionKey{},provider);
            owner->state_->device_credit.emplace(std::move(credits.device));
            auto state_credit=credits.metadata.split(sizeof(State));
            if(!state_credit)throw std::logic_error("device page state metadata split");
            owner->state_->metadata_credit.emplace(std::move(*state_credit));
            owner->constructor_metadata_credit_.emplace(std::move(credits.metadata));
            physical_owner=owner;
            const auto error=provider.allocate(&owner->state_->data,bytes);
            if(error!=cudaSuccess || !owner->state_->data)
                throw std::runtime_error("device page storage allocation failed");
            // A cache-owned one-shot remains armed on hits or reservation refusal.
            // Its owner serializes this factory and consumes it only after allocation.
            if(pending_constructor_fault_for_test)
                constructor_fault_for_test=std::exchange(*pending_constructor_fault_for_test,0U);
            if(constructor_fault_for_test==1)throw std::runtime_error("injected device page post-allocation failure");
            std::span<std::uint16_t> destination(static_cast<std::uint16_t*>(owner->state_->data),
                Exl3DevicePageFill::elements);
            auto fill=make_bounded_shared<Exl3DevicePageFill>(std::move(key),owner,destination,
                [](const std::shared_ptr<const void>& retained) noexcept {
                    return static_cast<const Exl3DevicePageStorage*>(retained.get())->retire();
                },constructor_fault_for_test>=2?constructor_fault_for_test-1:0);
            Inventory actual;actual.add(device_allocation(fill));
            actual.add(metadata_allocation(fill));
            result={std::move(fill),destination};return actual;
        },[&]() noexcept {result={};physical_owner.reset();},[&]() noexcept {
            if(quarantined_bytes()!=quarantine_before)authority.seal_failed_startup_retirement();
        });}catch(...) {
            result={};physical_owner.reset();
            if(quarantined_bytes()!=quarantine_before)authority.seal_failed_startup_retirement();
            throw;
        }
        // Committed inventory now owns the full allocation. Failed constructors
        // instead retain the state/device tickets above until actual retirement.
        physical_owner->state_->device_credit.reset();
        physical_owner->state_->metadata_credit.reset();
        physical_owner->constructor_metadata_credit_.reset();
        return result;
    }
};
}
