#pragma once
#include "exl3/resource_inventory.h"
#include "exl3/bounded_shared_owner.h"
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

namespace ninfer::exl3 {
struct Exl3AttentionStageStorageProvider {
    cudaError_t (*allocate)(void**,std::size_t) noexcept=
        [](void** p,std::size_t n) noexcept {return cudaMalloc(p,n);};
    cudaError_t (*release)(void*) noexcept=[](void* p) noexcept {return cudaFree(p);};
    cudaError_t (*create)(cudaEvent_t*) noexcept=
        [](cudaEvent_t* p) noexcept {return cudaEventCreateWithFlags(p,cudaEventDisableTiming);};
    cudaError_t (*destroy)(cudaEvent_t) noexcept=[](cudaEvent_t p) noexcept {return cudaEventDestroy(p);};
};
// One additional bank destination: K and V plus distinct producer/final-use
// events per plane. Active stage leases must retain this complete owner.
class Exl3AttentionStageStorage {
    struct State {
        void* data=nullptr;
        std::array<cudaEvent_t,4> events{};
        std::size_t plane_bytes=0;
        Exl3AttentionStageStorageProvider provider;
        std::optional<RetainedDeviceLedger::Ticket> device_credit;
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        State* next=nullptr;
    };
    std::unique_ptr<State> state_;
    std::optional<RetainedDescriptorLedger::Ticket> constructor_metadata_credit_;
    inline static std::atomic<State*> quarantine_{nullptr};
    inline static std::atomic<std::uint64_t> quarantined_records_{0};
    struct ConstructionKey {};
public:
    explicit Exl3AttentionStageStorage(ConstructionKey,std::size_t plane,Exl3AttentionStageStorageProvider provider)
        :state_(std::make_unique<State>()) {state_->plane_bytes=plane;state_->provider=provider;}
    static constexpr std::size_t metadata_bytes() noexcept {
        return sizeof(State)+bounded_shared_allocation_bytes<Exl3AttentionStageStorage>();
    }
    static constexpr std::size_t retirement_metadata_bytes() noexcept {return sizeof(State);}
    static bool attach_device_credit(const std::shared_ptr<const void>& owner,
        RetainedDeviceLedger::Ticket credit) noexcept {
        if(!owner)return false;
        auto* storage=const_cast<Exl3AttentionStageStorage*>(static_cast<const Exl3AttentionStageStorage*>(owner.get()));
        if(!storage->state_->data || storage->state_->device_credit ||
            credit.bytes()!=storage->state_->plane_bytes*2)return false;
        storage->state_->device_credit.emplace(std::move(credit));return true;
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || credit.bytes()!=metadata_bytes() ||
            !can_attach_bounded_retirement_credit<Exl3AttentionStageStorage>(owner))return false;
        auto* storage=const_cast<Exl3AttentionStageStorage*>(static_cast<const Exl3AttentionStageStorage*>(owner.get()));
        if(storage->state_->metadata_credit)return false;
        auto state_credit=credit.split(sizeof(State));if(!state_credit)return false;
        storage->state_->metadata_credit.emplace(std::move(*state_credit));
        return attach_bounded_retirement_credit<Exl3AttentionStageStorage>(owner,std::move(credit));
    }
    static std::array<std::uint64_t,2> retained_credits_for_test() noexcept {
        const auto* state=quarantine_.load();
        return {state && state->device_credit?state->device_credit->bytes():0,
            state && state->metadata_credit?state->metadata_credit->bytes():0};
    }
    static std::uint64_t quarantined_records() noexcept {return quarantined_records_.load();}
    void release_constructor_credits_after_commit() noexcept {
        state_->device_credit.reset();state_->metadata_credit.reset();
        constructor_metadata_credit_.reset();
    }
    ~Exl3AttentionStageStorage() {
        // No work may remain when the last lease releases this owner. Failed
        // cleanup retains the unresolved handles without retry or further frees.
        bool failed=false;
        for(auto& event:state_->events) if(event) {
            if(state_->provider.destroy(event)!=cudaSuccess) {failed=true;break;}
            event=nullptr;
        }
        if(!failed && state_->data) {
            failed=state_->provider.release(state_->data)!=cudaSuccess;
            if(!failed)state_->data=nullptr;
        }
        if(failed) {
            auto* retained=state_.release();auto* head=quarantine_.load();
            do{retained->next=head;}while(!quarantine_.compare_exchange_weak(head,retained));
            ++quarantined_records_;
        }
    }
    std::uint16_t* plane(unsigned index) const {
        if(index>1)throw std::invalid_argument("attention stage plane");
        return reinterpret_cast<std::uint16_t*>(static_cast<std::byte*>(state_->data)+index*state_->plane_bytes);
    }
    cudaEvent_t producer_event(unsigned plane) const {return state_->events.at(plane*2);}
    cudaEvent_t consumer_event(unsigned plane) const {return state_->events.at(plane*2+1);}
    int capacity() const noexcept {return static_cast<int>(state_->plane_bytes/2048);}
    template<class Coordinator>
    static std::shared_ptr<Exl3AttentionStageStorage> create_startup(Coordinator& authority,int capacity,
        Exl3AttentionStageStorageProvider provider={}) {
        if(capacity<1 || static_cast<std::size_t>(capacity)>std::numeric_limits<std::size_t>::max()/4096 ||
            !provider.allocate || !provider.release || !provider.create || !provider.destroy)
            throw std::invalid_argument("attention stage storage capacity/provider");
        if(quarantined_records())throw std::runtime_error("unresolved attention stage storage retirement");
        using Inventory=Exl3ResourceInventory;
        const auto plane_bytes=static_cast<std::size_t>(capacity)*2048;
        Inventory::Requirement required;required.configuration=0x4154544453;
        required.add(Inventory::Domain::device,1,plane_bytes*2);
        required.add(Inventory::Domain::host_metadata,1,metadata_bytes());
        std::shared_ptr<Exl3AttentionStageStorage> result;
        const auto quarantined_before=quarantined_records();
        try {authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("attention stage storage reservation identity");
            auto credits=authority.reserve_constructor_credits(plane_bytes*2,metadata_bytes());
            if(credits.device.bytes()!=plane_bytes*2 || credits.metadata.bytes()!=metadata_bytes())
                throw std::invalid_argument("attention stage constructor credit extent");
            auto owner=make_bounded_shared<Exl3AttentionStageStorage>(ConstructionKey{},plane_bytes,provider);
            owner->state_->device_credit.emplace(std::move(credits.device));
            auto state_credit=credits.metadata.split(sizeof(State));
            if(!state_credit)throw std::logic_error("attention stage constructor metadata extent");
            owner->state_->metadata_credit.emplace(std::move(*state_credit));
            owner->constructor_metadata_credit_.emplace(std::move(credits.metadata));
            if(provider.allocate(&owner->state_->data,plane_bytes*2)!=cudaSuccess || !owner->state_->data)
                throw std::runtime_error("attention stage destination allocation failed");
            for(auto& event:owner->state_->events)
                if(provider.create(&event)!=cudaSuccess || !event)
                    throw std::runtime_error("attention stage event allocation failed");
            Inventory actual;actual.add({owner,0,Inventory::Domain::device,plane_bytes*2,{},nullptr,
                &attach_device_credit});
            actual.add({owner,1,Inventory::Domain::host_metadata,metadata_bytes(),{},&attach_metadata_credit});
            result=std::move(owner);return actual;
        },[&]() noexcept {result.reset();});} catch(...) {
            result.reset();
            if(quarantined_records()!=quarantined_before)authority.seal_failed_startup_retirement();
            throw;
        }
        // Committed inventory now carries these charges until lifetime handoff.
        if constexpr(!requires { Coordinator::defers_constructor_credits; })
            result->release_constructor_credits_after_commit();
        return result;
    }
};
}
