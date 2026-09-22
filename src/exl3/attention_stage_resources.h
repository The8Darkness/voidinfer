#pragma once
#include "exl3/attention_stage_storage.h"
#include "exl3/attention_stage.h"
#include "exl3/attention_stage_history.h"

namespace ninfer::exl3 {
// Suballocations consume an already accepted whole-transition reservation.
// This adapter cannot consult availability or increase that reservation.
template<class Coordinator>
class Exl3AttentionStageSuballocation {
    Coordinator& authority_;
    Exl3ResourceInventory::Totals remaining_;
    Exl3ResourceInventory actual_;
public:
    static constexpr bool defers_constructor_credits=true;
    explicit Exl3AttentionStageSuballocation(Coordinator& authority,Exl3ResourceInventory::Totals reserved)
        :authority_(authority),remaining_(reserved){}
    auto reserve_constructor_credits(std::uint64_t device,std::uint64_t metadata) {
        return authority_.reserve_constructor_credits(device,metadata);
    }
    void seal_failed_startup_retirement() noexcept {authority_.seal_failed_startup_retirement();}
    template<class Factory>
    void allocate_startup_resources(const Exl3ResourceInventory::Requirement& request,Factory&& factory) {
        for(std::size_t i=0;i<remaining_.size();++i)
            if(request.units[i]>remaining_[i])throw std::logic_error("attention stage suballocation exceeds reservation");
        for(std::size_t i=0;i<remaining_.size();++i)remaining_[i]-=request.units[i];
        auto actual=factory(request.configuration);
        if(actual.totals()!=request.units)throw std::logic_error("attention stage suballocation inventory mismatch");
        actual_.append(actual);
    }
    template<class Factory,class Rollback>
    void allocate_startup_resources(const Exl3ResourceInventory::Requirement& request,
        Factory&& factory,Rollback&& rollback) {
        try {allocate_startup_resources(request,std::forward<Factory>(factory));}
        catch(...) {std::forward<Rollback>(rollback)();throw;}
    }
    Exl3ResourceInventory finish() {
        for(auto remaining:remaining_)if(remaining)throw std::logic_error("attention stage incomplete transaction");
        return std::move(actual_);
    }
};
struct Exl3AttentionStageResources {
    std::shared_ptr<Exl3AttentionStageStorage> storage;
    std::shared_ptr<Exl3AttentionStage> stages;
    std::shared_ptr<Exl3AttentionStageHistory> history;
    static constexpr std::size_t metadata_bytes() noexcept {
        return Exl3AttentionStageStorage::metadata_bytes()+Exl3AttentionStage::metadata_bytes()+
            Exl3AttentionStageHistory::metadata_bytes();
    }
    template<class Coordinator>
    static Exl3AttentionStageResources create_startup(Coordinator& authority,int capacity,
        Exl3AttentionStageStorageProvider provider={}) {
        if(capacity<1 || static_cast<std::size_t>(capacity)>std::numeric_limits<std::size_t>::max()/4096)
            throw std::invalid_argument("attention stage transaction capacity");
        using Inventory=Exl3ResourceInventory;
        Inventory::Requirement required;required.configuration=0x415454414c;
        required.add(Inventory::Domain::device,1,static_cast<std::uint64_t>(capacity)*4096);
        required.add(Inventory::Domain::host_metadata,1,metadata_bytes());
        Exl3AttentionStageResources result;
        const auto before=Exl3AttentionStageStorage::quarantined_records();
        try {authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("attention stage transaction identity");
            Exl3AttentionStageSuballocation reserved(authority,required.units);
            Exl3AttentionStageResources pending;
            pending.storage=Exl3AttentionStageStorage::create_startup(reserved,capacity,provider);
            pending.stages=Exl3AttentionStage::create_startup(reserved);
            pending.history=Exl3AttentionStageHistory::create_startup(reserved);
            auto actual=reserved.finish();result=std::move(pending);return actual;
        },[&]() noexcept {result={};});} catch(...) {
            result={};
            if(Exl3AttentionStageStorage::quarantined_records()!=before)authority.seal_failed_startup_retirement();
            throw;
        }
        if constexpr(!requires { Coordinator::defers_constructor_credits; })
            result.storage->release_constructor_credits_after_commit();
        return result;
    }
};
template<class Coordinator>
std::array<Exl3AttentionStageResources,2> exl3_create_attention_stage_lanes(Coordinator& authority,
    unsigned lanes,int capacity,Exl3AttentionStageStorageProvider provider={}) {
    if(lanes<1 || lanes>2 || capacity<1)
        throw std::invalid_argument("attention stage lane transaction geometry");
    using Inventory=Exl3ResourceInventory;
    Inventory::Requirement required;required.configuration=0x4154544332;
    required.add(Inventory::Domain::device,lanes,static_cast<std::uint64_t>(capacity)*4096);
    required.add(Inventory::Domain::host_metadata,lanes,Exl3AttentionStageResources::metadata_bytes());
    std::array<Exl3AttentionStageResources,2> result;
    const auto before=Exl3AttentionStageStorage::quarantined_records();
    try {authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
        if(configuration!=required.configuration)throw std::logic_error("attention stage lane transaction identity");
        Exl3AttentionStageSuballocation reserved(authority,required.units);
        std::array<Exl3AttentionStageResources,2> pending;
        for(unsigned lane=0;lane<lanes;++lane)
            pending[lane]=Exl3AttentionStageResources::create_startup(reserved,capacity,provider);
        auto actual=reserved.finish();result=std::move(pending);return actual;
    },[&]() noexcept {result={};});} catch(...) {
        result={};
        if(Exl3AttentionStageStorage::quarantined_records()!=before)authority.seal_failed_startup_retirement();
        throw;
    }
    if constexpr(!requires { Coordinator::defers_constructor_credits; })
        for(unsigned lane=0;lane<lanes;++lane)result[lane].storage->release_constructor_credits_after_commit();
    return result;
}
}
