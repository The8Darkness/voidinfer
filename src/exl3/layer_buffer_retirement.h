#pragma once

#include "exl3/retained_descriptor_ledger.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {
// Owning storage only: borrowed layer scratch must never be adopted here.
// Allocate this record before allocating device storage. Failed cleanup publishes
// an immutable record and never retries or switches the current CUDA device.
class Exl3LayerBufferRetirement {
public:
    using Release = int (*)(void*,int) noexcept;
    struct Record {
        void* pointer=nullptr;
        std::uint64_t bytes=0;
        int device=-1,error=0;
        std::optional<RetainedDeviceLedger::Ticket> device_credit;
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        Record* next=nullptr;
    };
private:
    inline static std::atomic<Record*> quarantine_{nullptr};
    inline static std::atomic<std::uint64_t> count_{0};
    std::unique_ptr<Record> record_=std::make_unique<Record>();
    Release release_;
    int cleanup_error_for_test_=0;
public:
    explicit Exl3LayerBufferRetirement(Release release):release_(release) {
        if(!release_)throw std::invalid_argument("layer buffer cleanup callback missing");
        if(quarantined())throw std::runtime_error("unresolved layer buffer retirement");
    }
    Exl3LayerBufferRetirement(const Exl3LayerBufferRetirement&)=delete;
    Exl3LayerBufferRetirement& operator=(const Exl3LayerBufferRetirement&)=delete;
    ~Exl3LayerBufferRetirement(){retire();}
    static constexpr std::size_t record_bytes() noexcept {return sizeof(Record);}
    std::uint64_t bytes() const noexcept {return record_?record_->bytes:0;}
    void fail_cleanup_for_test(int error) noexcept {cleanup_error_for_test_=error;}
    void release_constructor_credits_after_commit() noexcept {
        if(record_){record_->device_credit.reset();record_->metadata_credit.reset();}
    }
    static bool attach_device_credit(const std::shared_ptr<const void>& owner,RetainedDeviceLedger::Ticket credit) noexcept {
        auto* self=const_cast<Exl3LayerBufferRetirement*>(static_cast<const Exl3LayerBufferRetirement*>(owner.get()));
        if(!self || !self->record_ || !self->record_->pointer || self->record_->device_credit || credit.bytes()!=self->bytes())return false;
        self->record_->device_credit.emplace(std::move(credit));return true;
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* self=const_cast<Exl3LayerBufferRetirement*>(static_cast<const Exl3LayerBufferRetirement*>(owner.get()));
        if(!self || !self->record_ || !self->record_->pointer || self->record_->metadata_credit || credit.bytes()!=record_bytes())return false;
        self->record_->metadata_credit.emplace(std::move(credit));return true;
    }
    void adopt(void* pointer,std::uint64_t bytes,int device) {
        if(!record_ || record_->pointer || !pointer || !bytes || device<0)
            throw std::invalid_argument("invalid layer buffer adoption");
        record_->pointer=pointer;record_->bytes=bytes;record_->device=device;
    }
    void attach(RetainedDeviceLedger::Ticket device,RetainedDescriptorLedger::Ticket metadata) {
        if(!record_ || !record_->pointer || record_->device_credit || record_->metadata_credit ||
            device.bytes()!=record_->bytes || metadata.bytes()!=record_bytes())
            throw std::invalid_argument("layer buffer retirement credit mismatch");
        record_->device_credit.emplace(std::move(device));
        record_->metadata_credit.emplace(std::move(metadata));
    }
    void retire() noexcept {
        if(!record_)return;
        if(!record_->pointer){record_.reset();return;}
        record_->error=cleanup_error_for_test_?cleanup_error_for_test_:release_(record_->pointer,record_->device);
        if(!record_->error){record_.reset();return;}
        auto* failed=record_.release();
        auto* head=quarantine_.load(std::memory_order_relaxed);
        do{failed->next=head;}while(!quarantine_.compare_exchange_weak(head,failed,
            std::memory_order_release,std::memory_order_relaxed));
        count_.fetch_add(1,std::memory_order_release);
    }
    static std::uint64_t quarantined() noexcept {return count_.load(std::memory_order_acquire);}
    static const Record* latest_for_test() noexcept {return quarantine_.load(std::memory_order_acquire);}
};
}
