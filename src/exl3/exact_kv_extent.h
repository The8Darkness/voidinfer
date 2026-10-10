#pragma once
#include <array>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <functional>
#include <atomic>
#include "exl3/bounded_shared_owner.h"

namespace ninfer::exl3 {
class Exl3LaneKvSource;
// Copyable atomic flag (pages are cloned by value).
struct Exl3CopyableFlag {
    std::atomic<bool> value{false};
    Exl3CopyableFlag() = default;
    Exl3CopyableFlag(const Exl3CopyableFlag& other) : value(other.value.load(std::memory_order_acquire)) {}
    Exl3CopyableFlag& operator=(const Exl3CopyableFlag& other) {
        value.store(other.value.load(std::memory_order_acquire), std::memory_order_release);
        return *this;
    }
    bool load() const noexcept { return value.load(std::memory_order_acquire); }
    void store(bool v) noexcept { value.store(v, std::memory_order_release); }
};
template <class T> struct Exl3CopyablePointer {
    std::atomic<T*> value{nullptr};
    Exl3CopyablePointer() = default;
    Exl3CopyablePointer(const Exl3CopyablePointer& other) : value(other.value.load(std::memory_order_acquire)) {}
    Exl3CopyablePointer& operator=(const Exl3CopyablePointer& other) {
        value.store(other.value.load(std::memory_order_acquire), std::memory_order_release);
        return *this;
    }
    T* load() const noexcept { return value.load(std::memory_order_acquire); }
    void store(T* v) noexcept { value.store(v, std::memory_order_release); }
};
// Internal backing. Published images expose only const page handles. A private
// append may reserve a larger tail before upload; valid extents exclude it.
struct Exl3ExactKVPage {
    static constexpr int token_capacity=64;
    int first=0,rows=0;
    // FP16 [rows][4][256] per bank, or (L0 L2 FP8 pages, l0_l2_fp8.cuh) rows of
    // 516 words: 1024 e4m3 bytes + 4 FP16 head scales.
    bool fp8=false;
    std::array<std::vector<std::uint16_t>,16> k,v;
    // Lane-backed payload (lane_kv_source.h): while `deferred`, k/v are empty
    // and the rows live only in `source`'s lane planes, which keep them
    // unchanged until the source fills the page (before those rows change).
    std::shared_ptr<Exl3LaneKvSource> source;
    Exl3CopyablePointer<Exl3LaneKvSource> source_id;   // lock-free identity of `source`
    Exl3CopyableFlag deferred;
    bool payload_deferred() const noexcept { return deferred.load(); }
    static std::shared_ptr<Exl3ExactKVPage> create(
        const std::function<RetainedDescriptorLedger::Ticket(std::uint64_t)>& reserve={}) {
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            credit.emplace(reserve(metadata_bytes()));
            if(credit->bytes()!=metadata_bytes())throw std::invalid_argument("KV page metadata reservation extent");
        }
        auto page=make_bounded_shared<Exl3ExactKVPage>();
        if(credit && !attach_metadata_credit(page,std::move(*credit)))
            throw std::logic_error("KV page metadata attachment");
        return page;
    }
    static constexpr std::size_t metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3ExactKVPage>();
    }
    static bool metadata_credit_belongs_to(const std::shared_ptr<const void>& owner,
        const RetainedDescriptorLedger& ledger) noexcept {
        return bounded_split_credit_belongs_to<Exl3ExactKVPage>(owner,ledger);
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        // K/V vector buffers remain separate FP16 payload allocations.
        return attach_bounded_split_retirement_credit<Exl3ExactKVPage>(owner,std::move(credit),0);
    }
};
class Exl3ExactKVExtent {
public:
    enum class Plane {key,value};
    enum class Format {ordinary_fp16_kv1024};
    static constexpr std::size_t stride=1024;
    static Exl3ExactKVExtent view(std::shared_ptr<const Exl3ExactKVPage> page,
        int bank,Plane plane,int published_position,int first_row=0) {
        if(!page || !page.use_count() || bank<0 || bank>=16 || (plane!=Plane::key && plane!=Plane::value) ||
           first_row<0 || published_position<first_row ||
           page->first<0 || page->rows<1 || page->rows>Exl3ExactKVPage::token_capacity ||
           page->first>std::numeric_limits<int>::max()-page->rows)
            throw std::invalid_argument("exact KV extent identity/geometry");
        const auto& storage=plane==Plane::key?page->k[bank]:page->v[bank];
        const int begin=std::max(page->first,first_row);
        const int end=std::min(page->first+page->rows,published_position);
        Exl3ExactKVExtent result;result.owner_=std::move(page);result.bank_=bank;result.plane_=plane;
        result.backing_data_=storage.data();result.backing_elements_=storage.size();
        result.page_first_=result.owner_->first;result.page_rows_=result.owner_->rows;
        result.first_=begin;result.rows_=std::max(0,end-begin);
        if(result.rows_) {
            const auto offset=static_cast<std::size_t>(begin-result.owner_->first)*stride;
            const auto elements=static_cast<std::size_t>(result.rows_)*stride;
            if(offset>storage.size() || elements>storage.size()-offset)
                throw std::invalid_argument("exact KV extent exceeds represented plane");
            result.data_=storage.data()+offset;
        }
        return result;
    }
    const std::shared_ptr<const Exl3ExactKVPage>& owner() const noexcept {return owner_;}
    bool backing_current() const noexcept {
        if(!owner_ || !owner_.use_count())return false;
        const auto& storage=plane_==Plane::key?owner_->k[bank_]:owner_->v[bank_];
        return owner_->first==page_first_ && owner_->rows==page_rows_ &&
            storage.data()==backing_data_ && storage.size()==backing_elements_;
    }
    struct Backing {
        std::shared_ptr<const Exl3ExactKVPage> owner;
        const std::uint16_t* data;
        std::size_t bytes;
        int bank;
        Plane plane;
    };
    bool registration_eligible() const noexcept {
        // A short view may alias a complete stable plane, but rows metadata
        // alone does not prove the allocation contains that entire plane.
        return backing_current() && rows_>0 && page_rows_==Exl3ExactKVPage::token_capacity &&
            backing_elements_==static_cast<std::size_t>(Exl3ExactKVPage::token_capacity)*stride;
    }
    Backing backing() const {
        if(!backing_current())throw std::invalid_argument("exact KV backing changed after extent capture");
        return {owner_,backing_data_,backing_elements_*sizeof(std::uint16_t),bank_,plane_};
    }
    const std::uint16_t* data() const {
        if(!backing_current())throw std::invalid_argument("exact KV backing changed after extent capture");
        return data_;
    }
    int first() const noexcept {return first_;}
    int rows() const noexcept {return rows_;}
    int bank() const noexcept {return bank_;}
    Plane plane() const noexcept {return plane_;}
    Format format() const noexcept {return Format::ordinary_fp16_kv1024;}
    std::size_t bytes() const noexcept {return static_cast<std::size_t>(rows_)*stride*sizeof(std::uint16_t);}
    std::size_t destination_offset(int capacity) const {
        if(capacity<0 || first_>capacity || rows_>capacity-first_)
            throw std::invalid_argument("exact KV destination capacity");
        return static_cast<std::size_t>(first_)*stride;
    }
    // A retained page is neither a CUDA registration nor a Windows lock
    // certificate. Those authorities must be acquired and retained separately.
private:
    Exl3ExactKVExtent()=default;
    std::shared_ptr<const Exl3ExactKVPage> owner_;
    const std::uint16_t* data_=nullptr;
    const std::uint16_t* backing_data_=nullptr;
    std::size_t backing_elements_=0;
    int page_first_=0,page_rows_=0;
    int first_=0,rows_=0,bank_=0;
    Plane plane_=Plane::key;
};
}
