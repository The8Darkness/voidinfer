#pragma once
#include "exl3/device_page_reader.h"
#include "exl3/registered_kv_upload.h"

namespace ninfer::exl3 {
struct Exl3SharedAttentionPrefix {
    const std::uint16_t* k=nullptr;
    const std::uint16_t* v=nullptr;
    int rows=0;
    std::uint64_t generation=0;
    int first=0;
    unsigned slot=0;
    std::weak_ptr<const void> owner;
};
// One immutable page range per layer use; other ranges retain private storage.
class Exl3DevicePageAttention {
public:
    enum class Stage { none,record,wait };
    struct Failure {Stage stage=Stage::none;int error=0;};
private:
    Exl3DevicePageReader reader_;
    cudaEvent_t event_=nullptr;
    std::uint64_t generation_=0;
    std::uint64_t acquisition_=0,execution_=0;
    std::weak_ptr<const void> event_owner_;
    int bank_=0,position_=0;
    bool failed_=false;
    Failure failure_;
public:
    bool uncertain() const noexcept {return reader_.uncertain();}
    Failure failure() const noexcept {return failure_;}
    static constexpr std::size_t metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3DevicePageAttention>()+
            Exl3KVTransferLease::state_metadata_bytes();
    }
    static bool attach_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || credit.bytes()!=metadata_bytes() ||
            !can_attach_bounded_retirement_credit<Exl3DevicePageAttention>(owner))return false;
        auto* attention=const_cast<Exl3DevicePageAttention*>(
            static_cast<const Exl3DevicePageAttention*>(owner.get()));
        if(!attention->reader_.can_attach_retirement_credit())return false;
        auto state_credit=credit.split(Exl3KVTransferLease::state_metadata_bytes());
        if(!state_credit)return false;
        return attention->reader_.attach_retirement_credit(std::move(*state_credit)) &&
            attach_bounded_retirement_credit<Exl3DevicePageAttention>(owner,std::move(credit));
    }
    template<class Coordinator>
    static std::shared_ptr<Exl3DevicePageAttention> create_startup(Coordinator& authority) {
        using Inventory=Exl3ResourceInventory;
        constexpr auto metadata=metadata_bytes();
        Inventory::Requirement required;required.configuration=0x4456504154;
        required.add(Inventory::Domain::host_metadata,1,metadata);
        std::shared_ptr<Exl3DevicePageAttention> result;
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("page attention reservation identity");
            result=make_bounded_shared<Exl3DevicePageAttention>();
            Inventory actual;actual.add({result,0,Inventory::Domain::host_metadata,metadata,{},
                &attach_retirement_credit});return actual;
        },[&]() noexcept {result.reset();});
        return result;
    }
    static void require_admission(const Exl3DevicePageKey& expected,int bank,int position,
        const std::shared_ptr<const void>& event_owner,std::uint64_t acquisition,
        std::uint64_t execution,cudaEvent_t event) {
        if(!expected.current() || position<expected.page()->first ||
            position-expected.page()->first<64 || bank<0 || bank>=16 ||
            !event || !acquisition || !execution || !event_owner || !event_owner.use_count())
            throw std::invalid_argument("shared attention prefix geometry/ownership");
    }
    Exl3SharedAttentionPrefix begin(std::shared_ptr<const Exl3DevicePageFill::View> view,
        const Exl3DevicePageKey& expected,int bank,int position,std::shared_ptr<const void> event_owner,
        std::uint64_t acquisition,std::uint64_t execution,cudaEvent_t event) {
        if(failed_ || reader_.uncertain())throw std::logic_error("shared attention reader still owned");
        require_admission(expected,bank,position,event_owner,acquisition,execution,event);
        if(!view)throw std::invalid_argument("shared attention prefix view");
        const auto k=view->plane(bank,true),v=view->plane(bank,false);
        reader_.bind(view,expected,view->generation(),event_owner);
        generation_=reader_.begin(acquisition,execution,reinterpret_cast<std::uintptr_t>(event));
        acquisition_=acquisition;execution_=execution;event_owner_=event_owner;
        bank_=bank;position_=position;
        event_=event;return {k.data(),v.data(),64,generation_,expected.page()->first,0,std::move(view)};
    }
    void complete(std::uint64_t generation,cudaStream_t consumer,Exl3KVUploadProvider provider={}) {
        Completion item{this,generation};complete_group(std::span(&item,1),consumer,provider);
    }
    struct Completion {Exl3DevicePageAttention* reader;std::uint64_t generation;};
    static cudaEvent_t require_group(std::span<const Completion> items) {
        if(items.empty() || items.size()>64)
            throw std::logic_error("shared attention group capacity");
        cudaEvent_t event=nullptr;
        for(std::size_t i=0;i<items.size();++i) {
            const auto& item=items[i];const auto* reader=item.reader;
            if(!reader || reader->failed_ || !item.generation || item.generation!=reader->generation_ ||
                !reader->reader_.uncertain() || !reader->event_ || (event && event!=reader->event_))
                throw std::logic_error("shared attention completion scope/event");
            // A singleton group must prove current backing too; pairwise
            // compatibility checks below cannot provide that proof for it.
            (void)reader->reader_.input_key();
            for(std::size_t j=0;j<i;++j)if(items[j].reader==reader)
                throw std::logic_error("duplicate attention completion reader");
            if(i) {
                const auto* first=items[0].reader;
                if(reader->bank_!=first->bank_ || reader->position_!=first->position_ ||
                    !reader->reader_.input_key().same_model_position(first->reader_.input_key()))
                    throw std::logic_error("attention completion crosses numerical input identity");
                if(reader->acquisition_!=first->acquisition_ || reader->execution_!=first->execution_ ||
                    reader->event_owner_.owner_before(first->event_owner_) || first->event_owner_.owner_before(reader->event_owner_))
                    throw std::logic_error("attention completion crosses execution/event ownership");
            }
            event=reader->event_;
        }
        return event;
    }
    static void complete_group(std::span<const Completion> items,cudaStream_t consumer,Exl3KVUploadProvider provider={}) {
        if(!provider.record || !provider.wait)throw std::logic_error("shared attention completion provider");
        const auto event=require_group(items);
        const auto accept=[&](cudaError_t error,Stage stage) {
            if(error==cudaSuccess)return;
            for(const auto& item:items) {
                item.reader->failed_=true;item.reader->failure_={stage,static_cast<int>(error)};
                item.reader->reader_.finish(item.generation,static_cast<int>(error));
            }
            throw std::runtime_error("shared attention final use uncertain");
        };
        accept(provider.record(event,consumer),Stage::record);
        accept(provider.wait(event),Stage::wait);
        for(const auto& item:items) {
            if(!item.reader->reader_.finish(item.generation,0))throw std::logic_error("shared attention completion replay");
            item.reader->event_=nullptr;
            item.reader->event_owner_.reset();item.reader->acquisition_=item.reader->execution_=0;
        }
    }
};
}
