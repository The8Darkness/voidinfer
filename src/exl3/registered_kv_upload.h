#pragma once
#include "exl3/registered_kv_backing.h"
#include "exl3/kv_transfer_lease.h"
#include "exl3/bounded_shared_owner.h"
#include <limits>
#include <array>

namespace ninfer::exl3 {
struct Exl3KVUploadProvider {
    cudaError_t (*copy)(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t) noexcept=
        [](void* out,const void* in,std::size_t bytes,cudaMemcpyKind kind,cudaStream_t stream) noexcept {
            return cudaMemcpyAsync(out,in,bytes,kind,stream);
        };
    cudaError_t (*record)(cudaEvent_t,cudaStream_t) noexcept=
        [](cudaEvent_t event,cudaStream_t stream) noexcept {return cudaEventRecord(event,stream);};
    cudaError_t (*wait)(cudaEvent_t) noexcept=
        [](cudaEvent_t event) noexcept {return cudaEventSynchronize(event);};
};
// One precredited record per Engine transfer slot. The qualified default uses
// two slots per physical lane; a bounded 32-slot experiment can retain one
// complete C1 target forward's registered sources until the existing final
// stream drain.
class Exl3RegisteredKVUpload {
public:
    enum class Stage {none,copy,record,wait};
    struct Failure {Stage stage=Stage::none;int error=0;};
private:
    Exl3KVTransferLease transfer_;
    Failure failure_;
    cudaEvent_t event_=nullptr;
    Exl3KVUploadProvider provider_;
    std::uint64_t generation_=0;
    void fail(Stage stage,cudaError_t error) {
        if(error==cudaSuccess)return;
        if(!failure_.error)failure_={stage,static_cast<int>(error)};
        transfer_.finish(generation_,static_cast<int>(error));
        throw std::runtime_error("registered KV upload final use uncertain");
    }
public:
    static constexpr std::size_t metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3RegisteredKVUpload>()+Exl3KVTransferLease::state_metadata_bytes();
    }
    // Called once under serialized inventory retirement. The state may survive
    // as quarantine after the wrapper and its final weak owner disappear.
    static bool attach_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || credit.bytes()!=metadata_bytes() ||
           !can_attach_bounded_retirement_credit<Exl3RegisteredKVUpload>(owner))return false;
        auto* upload=const_cast<Exl3RegisteredKVUpload*>(static_cast<const Exl3RegisteredKVUpload*>(owner.get()));
        if(!upload->transfer_.can_attach_retirement_credit())return false;
        auto state_credit=credit.split(Exl3KVTransferLease::state_metadata_bytes());
        if(!state_credit)return false;
        return upload->transfer_.attach_retirement_credit(std::move(*state_credit)) &&
            attach_bounded_retirement_credit<Exl3RegisteredKVUpload>(owner,std::move(credit));
    }
    bool uncertain() const noexcept {return transfer_.uncertain();}
    static constexpr unsigned maximum_slots_per_lane=32;
    using LanePool=std::array<std::array<std::shared_ptr<Exl3RegisteredKVUpload>,
        maximum_slots_per_lane>,2>;
    static Exl3ResourceInventory::Requirement lane_requirement(unsigned lanes,
        unsigned slots_per_lane=2) {
        if(lanes<1 || lanes>2)throw std::invalid_argument("KV upload physical lane count");
        if(slots_per_lane<2 || slots_per_lane>maximum_slots_per_lane)
            throw std::invalid_argument("KV upload slots per lane");
        Exl3ResourceInventory::Requirement required;required.configuration=0x4b5655504c;
        required.add(Exl3ResourceInventory::Domain::host_metadata,
            slots_per_lane*lanes,metadata_bytes());
        return required;
    }
    template<class Coordinator>
    static LanePool create_startup_lanes(Coordinator& authority,unsigned lanes,
        unsigned fail_after_slot_for_test=0,unsigned slots_per_lane=2) {
        const auto required=lane_requirement(lanes,slots_per_lane);
        if(fail_after_slot_for_test>slots_per_lane*lanes)
            throw std::invalid_argument("KV upload constructor fault slot");
        LanePool result{};
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("KV upload lane reservation identity");
            LanePool pending{};Exl3ResourceInventory actual;unsigned constructed=0;
            for(unsigned lane=0;lane<lanes;++lane)
              for(unsigned index=0;index<slots_per_lane;++index) {
                auto& slot=pending[lane][index];
                slot=make_bounded_shared<Exl3RegisteredKVUpload>();
                actual.add({slot,0,Exl3ResourceInventory::Domain::host_metadata,metadata_bytes(),{},
                    &attach_retirement_credit});
                if(++constructed==fail_after_slot_for_test)
                    throw std::runtime_error("injected KV upload slot construction failure");
            }
            result=std::move(pending);return actual;
        },[&]() noexcept {result={};});
        return result;
    }
    static void require_submission(const Exl3ExactKVExtent& extent,const void* destination,int capacity,
        cudaEvent_t event,std::uint64_t acquisition,std::uint64_t execution) {
        if(!acquisition || !execution)throw std::invalid_argument("registered KV upload scope missing");
        if(!destination || !event)throw std::invalid_argument("registered KV upload destination unavailable");
        if(!extent.bytes())throw std::invalid_argument("registered KV upload extent empty");
        const auto offset=extent.destination_offset(capacity);
        const auto base=reinterpret_cast<std::uintptr_t>(destination);
        if(base%alignof(std::uint16_t))throw std::invalid_argument("registered KV upload destination alignment");
        const auto maximum=std::numeric_limits<std::uintptr_t>::max();
        if(offset>(maximum-base)/sizeof(std::uint16_t))
            throw std::overflow_error("registered KV upload destination offset overflow");
        const auto begin=base+offset*sizeof(std::uint16_t);
        if(extent.bytes()>maximum-begin)
            throw std::overflow_error("registered KV upload destination extent overflow");
        if(!extent.backing_current())throw std::invalid_argument("registered KV upload source changed");
    }
    Failure failure() const noexcept {return failure_;}
    template<class Coordinator>
    static std::shared_ptr<Exl3RegisteredKVUpload> create_startup(Coordinator& authority) {
        using Inventory=Exl3ResourceInventory;
        constexpr auto bytes=metadata_bytes();
        Inventory::Requirement required;required.configuration=0x4b565550;
        required.add(Inventory::Domain::host_metadata,1,bytes);
        std::shared_ptr<Exl3RegisteredKVUpload> result;
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("KV upload reservation identity");
            auto owner=make_bounded_shared<Exl3RegisteredKVUpload>();
            Inventory actual;actual.add({owner,0,Inventory::Domain::host_metadata,bytes,{},
                &attach_retirement_credit});
            result=std::move(owner);return actual;
        },[&]() noexcept {result.reset();});
        return result;
    }
    std::uint64_t submit(const Exl3ExactKVExtent& extent,void* destination,int capacity,
        cudaStream_t stream,cudaEvent_t event,std::shared_ptr<const void> destination_owner,
        std::shared_ptr<Exl3RegisteredKVBacking> registration,
        std::uint64_t acquisition,std::uint64_t execution,Exl3KVUploadProvider provider={}) {
        if(transfer_.uncertain() || failure_.error)
            throw std::logic_error("registered KV upload slot still owned/failed");
        require_submission(extent,destination,capacity,event,acquisition,execution);
        if(!destination_owner || !destination_owner.use_count())
            throw std::invalid_argument("registered KV upload destination owner missing");
        if(!provider.copy || !provider.record || !provider.wait)
            throw std::invalid_argument("registered KV upload provider incomplete");
        if(!destination || !event || !registration || !registration->backing_matches(extent))
            throw std::invalid_argument("registered KV upload coverage/destination unavailable");
        auto reader=Exl3RegisteredKVBacking::acquire_reader(std::move(registration),extent);
        if(!reader)return 0; // Retirement won before submission: exact staged fallback.
        const auto offset=extent.destination_offset(capacity);
        const auto* source=extent.data();
        transfer_.rebind_registered(extent,std::move(destination_owner),std::move(*reader));
        generation_=transfer_.begin(acquisition,execution,reinterpret_cast<std::uintptr_t>(event));
        event_=event;provider_=provider;
        fail(Stage::copy,provider.copy(static_cast<std::uint16_t*>(destination)+offset,source,
            extent.bytes(),cudaMemcpyHostToDevice,stream));
        fail(Stage::record,provider.record(event,stream));
        return generation_;
    }
    void complete(std::uint64_t generation) {
        if(!generation || generation!=generation_ || !transfer_.uncertain() || failure_.error)
            throw std::logic_error("registered KV completion unavailable/stale/failed");
        fail(Stage::wait,provider_.wait(event_));
        if(!transfer_.finish(generation,0) || !transfer_.complete())
            throw std::logic_error("registered KV upload completion scope changed");
        transfer_.clear_completed(); // break destination-owner retention after proven final use
        event_=nullptr;
    }
    void copy(const Exl3ExactKVExtent& extent,void* destination,int capacity,
        cudaStream_t stream,cudaEvent_t event,std::shared_ptr<const void> destination_owner,
        std::shared_ptr<Exl3RegisteredKVBacking> registration,
        std::uint64_t acquisition,std::uint64_t execution,Exl3KVUploadProvider provider={}) {
        const auto generation=submit(extent,destination,capacity,stream,event,std::move(destination_owner),
            std::move(registration),acquisition,execution,provider);
        if(!generation)throw std::runtime_error("registered KV upload requires staged fallback after retirement");
        complete(generation);
    }
};
}
