#pragma once
#include "exl3/attention_history_coverage.h"
#include "exl3/resource_inventory.h"
#include "exl3/reconstruction_control_allocator.h"
#include <array>

namespace ninfer::exl3 {
struct Exl3AttentionStageRoute {
    bool enabled=false,owners=false,host_kv=false,copy_stream=false;
    bool private_prefix=false,shared_attention=false,shared_copy=false,media=false;
    bool wide=false,profiling=false,observer=false,capture=false,graph=false,oscar=false;
    int position=0,rows=0,context_capacity=0;
    std::size_t pages=0;
    constexpr bool eligible() const noexcept {
        return enabled && owners && host_kv && copy_stream &&
            !shared_attention && !shared_copy && !media && !wide && !profiling &&
            !observer && !capture && !graph && !oscar && position>0 && rows>0 && rows<=16 &&
            context_capacity>=rows && position<=context_capacity-rows && pages>0 && pages<=1024;
    }
};
// Lane-private snapshot pool. No slot may be rebound while a command/lease holds
// its immutable view. Capacity is explicit; larger histories use another route.
class Exl3AttentionStageHistory {
public:
    using Page=std::shared_ptr<const Exl3ExactKVPage>;
    static constexpr std::size_t capacity=1024;
    class Snapshot {
        friend class Exl3AttentionStageHistory;
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit_;
        std::array<Page,capacity> pages_{};
        std::size_t count_=0;
        bool control_admitted_=false;
        Exl3SharedControlCredit* control_credit_=nullptr;
    public:
        const Page* data() const noexcept {return pages_.data();}
        std::size_t size() const noexcept {return count_;}
        const Page* begin() const noexcept {return data();}
        const Page* end() const noexcept {return data()+count_;}
        const Page& front() const {if(!count_)throw std::logic_error("empty attention history");return pages_[0];}
    };
private:
    static std::shared_ptr<Snapshot> make_snapshot(bool fail_control_for_test=false) {
        auto* snapshot=new Snapshot;
        snapshot->control_admitted_=fail_control_for_test;
        return std::shared_ptr<Snapshot>(snapshot,[](Snapshot* value){
            auto credit=std::move(value->metadata_credit_);delete value;
        },
            Exl3ReconstructionControlAllocator<std::byte>(&snapshot->control_admitted_,&snapshot->control_credit_));
    }
    std::optional<RetainedDescriptorLedger::Ticket> metadata_credit_;
    std::shared_ptr<Snapshot> snapshot_;
    bool control_admitted_=false;
    Exl3SharedControlCredit* control_credit_=nullptr;
public:
    explicit Exl3AttentionStageHistory(bool fail_snapshot_control_for_test=false)
        :snapshot_(make_snapshot(fail_snapshot_control_for_test)) {}
    static constexpr std::size_t metadata_bytes() noexcept {
        return sizeof(Exl3AttentionStageHistory)+sizeof(Snapshot)+2*Exl3ReconstructionControlAllocator<std::byte>::capacity;
    }
    static bool attach_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || credit.bytes()!=metadata_bytes())return false;
        auto* history=const_cast<Exl3AttentionStageHistory*>(static_cast<const Exl3AttentionStageHistory*>(owner.get()));
        auto* snapshot=history->snapshot_.get();
        if(!snapshot || history->metadata_credit_ || snapshot->metadata_credit_ ||
            !history->control_credit_ || !snapshot->control_credit_ ||
            history->control_credit_->ticket || snapshot->control_credit_->ticket)return false;
        constexpr auto control=Exl3ReconstructionControlAllocator<std::byte>::capacity;
        auto outer=credit.split(sizeof(Exl3AttentionStageHistory));
        auto inner=credit.split(sizeof(Snapshot));
        auto outer_control=credit.split(control);
        if(!outer || !inner || !outer_control || credit.bytes()!=control)return false;
        history->metadata_credit_.emplace(std::move(*outer));
        snapshot->metadata_credit_.emplace(std::move(*inner));
        history->control_credit_->ticket.emplace(std::move(*outer_control));
        snapshot->control_credit_->ticket.emplace(std::move(credit));return true;
    }
    template<class Coordinator>
    static std::shared_ptr<Exl3AttentionStageHistory> create_startup(Coordinator& authority,
        bool fail_outer_control_for_test=false,bool fail_snapshot_control_for_test=false) {
        using Inventory=Exl3ResourceInventory;
        Inventory::Requirement required;required.configuration=0x4154544849;
        required.add(Inventory::Domain::host_metadata,1,metadata_bytes());
        std::shared_ptr<Exl3AttentionStageHistory> result;
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("attention history reservation identity");
            auto* backing=new Exl3AttentionStageHistory(fail_snapshot_control_for_test);
            backing->control_admitted_=fail_outer_control_for_test;
            auto owner=std::shared_ptr<Exl3AttentionStageHistory>(backing,
                [](Exl3AttentionStageHistory* value){
                    auto credit=std::move(value->metadata_credit_);delete value;
                },
                Exl3ReconstructionControlAllocator<std::byte>(&backing->control_admitted_,&backing->control_credit_));
            Inventory actual;actual.add({owner,0,Inventory::Domain::host_metadata,metadata_bytes(),{},
                &attach_retirement_credit});
            result=std::move(owner);return actual;
        });
        return result;
    }
    std::shared_ptr<const Snapshot> bind(std::span<const Page> pages,int position,int bank) {
        if(snapshot_.use_count()!=1)throw std::logic_error("attention history still leased");
        if(pages.size()>capacity)throw std::invalid_argument("attention history snapshot capacity");
        exl3_require_attention_history(pages,position,bank);
        // Validation above preserves the previous snapshot on refusal.
        std::copy(pages.begin(),pages.end(),snapshot_->pages_.begin());
        for(std::size_t i=pages.size();i<snapshot_->count_;++i)snapshot_->pages_[i].reset();
        snapshot_->count_=pages.size();return snapshot_;
    }
    void clear() {
        if(snapshot_.use_count()!=1)throw std::logic_error("attention history clear before final use");
        for(std::size_t i=0;i<snapshot_->count_;++i)snapshot_->pages_[i].reset();
        snapshot_->count_=0;
    }
};
}
