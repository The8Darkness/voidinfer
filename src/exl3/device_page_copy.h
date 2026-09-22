#pragma once
#include "exl3/device_page_reader.h"
#include "exl3/registered_kv_upload.h"

namespace ninfer::exl3 {
// Consumer bridge to the existing private contiguous attention destination.
// It retains the shared page only through D2D completion; attention subsequently
// reads lane-owned storage. This does not claim direct paged attention.
class Exl3DevicePageCopy {
public:
    enum class Stage {none,copy,record,wait};
    struct Failure {Stage stage=Stage::none;int error=0;};
private:
    Exl3DevicePageReader reader_;
    Failure failure_;
public:
    bool uncertain() const noexcept {return reader_.uncertain();}
    Failure failure() const noexcept {return failure_;}
    static constexpr std::size_t metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3DevicePageCopy>()+Exl3KVTransferLease::state_metadata_bytes();
    }
    static bool attach_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || credit.bytes()!=metadata_bytes() ||
            !can_attach_bounded_retirement_credit<Exl3DevicePageCopy>(owner))return false;
        auto* copy=const_cast<Exl3DevicePageCopy*>(static_cast<const Exl3DevicePageCopy*>(owner.get()));
        if(!copy->reader_.can_attach_retirement_credit())return false;
        auto state_credit=credit.split(Exl3KVTransferLease::state_metadata_bytes());
        if(!state_credit)return false;
        return copy->reader_.attach_retirement_credit(std::move(*state_credit)) &&
            attach_bounded_retirement_credit<Exl3DevicePageCopy>(owner,std::move(credit));
    }
    template<class Coordinator>
    static std::shared_ptr<Exl3DevicePageCopy> create_startup(Coordinator& authority) {
        using Inventory=Exl3ResourceInventory;
        constexpr auto metadata=metadata_bytes();
        Inventory::Requirement required;required.configuration=0x4456504350;
        required.add(Inventory::Domain::host_metadata,1,metadata);
        std::shared_ptr<Exl3DevicePageCopy> result;
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("page copy reservation identity");
            result=make_bounded_shared<Exl3DevicePageCopy>();
            Inventory actual;actual.add({result,0,Inventory::Domain::host_metadata,metadata,{},
                &attach_retirement_credit});return actual;
        },[&]() noexcept {result.reset();});
        return result;
    }
    void copy(std::shared_ptr<const Exl3DevicePageFill::View> view,const Exl3ExactKVExtent& extent,
        void* destination,int capacity,cudaStream_t stream,cudaEvent_t event,
        std::shared_ptr<const void> destination_and_event_owner,
        std::uint64_t acquisition,std::uint64_t execution,Exl3KVUploadProvider provider={}) {
        Exl3RegisteredKVUpload::require_submission(extent,destination,capacity,event,acquisition,execution);
        if(!view || !destination || !provider.copy || !provider.record || !provider.wait || !extent.backing_current())
            throw std::invalid_argument("page copy view/destination/provider");
        const auto& source_owner=view->key().page();const auto& requested=extent.owner();
        if(source_owner!=requested || source_owner.owner_before(requested) || requested.owner_before(source_owner))
            throw std::invalid_argument("page copy source ownership mismatch");
        const auto plane=view->plane(extent.bank(),extent.plane()==Exl3ExactKVExtent::Plane::key);
        const auto first=static_cast<std::size_t>(extent.first()-source_owner->first)*Exl3ExactKVExtent::stride;
        const auto count=extent.bytes()/sizeof(std::uint16_t);
        if(first>plane.size() || count>plane.size()-first || !count)
            throw std::invalid_argument("page copy source range");
        const auto offset=extent.destination_offset(capacity);
        reader_.bind(view,view->key(),view->generation(),std::move(destination_and_event_owner));
        const auto generation=reader_.begin(acquisition,execution,reinterpret_cast<std::uintptr_t>(event));
        const auto accept=[&](cudaError_t error,Stage stage) {
            if(error==cudaSuccess)return;
            if(!failure_.error)failure_={stage,static_cast<int>(error)};
            reader_.finish(generation,static_cast<int>(error));
            throw std::runtime_error("shared device page copy final use uncertain");
        };
        accept(provider.copy(static_cast<std::uint16_t*>(destination)+offset,plane.data()+first,
            extent.bytes(),cudaMemcpyDeviceToDevice,stream),Stage::copy);
        accept(provider.record(event,stream),Stage::record);accept(provider.wait(event),Stage::wait);
        if(!reader_.finish(generation,0))throw std::logic_error("shared page copy completion scope changed");
    }
};
}
