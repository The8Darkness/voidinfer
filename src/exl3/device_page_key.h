#pragma once
#include "exl3/exact_kv_extent.h"
#include <optional>

namespace ninfer::exl3 {
// Only ordinary FP16 sequential-text pages are admitted. Strong published page
// identity is the immutable content revision: a changed image must have a new
// COW owner. Pointer/size snapshots additionally reject changed backing geometry.
class Exl3DevicePageKey {
public:
    struct PositionContract {
        int rope_offset=0;
        bool media=false;
        bool operator==(const PositionContract&) const=default;
    };
private:
    std::shared_ptr<const void> model_;
    std::shared_ptr<const Exl3ExactKVPage> page_;
    PositionContract position_;
    std::array<std::optional<Exl3ExactKVExtent>,32> planes_;
    template<class A,class B> static bool same_owner(const A& a,const B& b) noexcept {
        return !a.owner_before(b) && !b.owner_before(a);
    }
public:
    Exl3DevicePageKey(std::shared_ptr<const void> model,
        std::shared_ptr<const Exl3ExactKVPage> page,PositionContract position,int published_position)
        :model_(std::move(model)),page_(std::move(page)),position_(position) {
        if(!model_ || !model_.use_count() || !page_ || !page_.use_count() ||
            position.media || page_->rows!=64 || page_->first<0 ||
            page_->first>std::numeric_limits<int>::max()-64 || published_position<page_->first+64)
            throw std::invalid_argument("shared device page requires published ordinary text page");
        for(int bank=0;bank<16;++bank)for(int plane=0;plane<2;++plane)
            planes_[bank*2+plane].emplace(Exl3ExactKVExtent::view(page_,bank,
                plane==0?Exl3ExactKVExtent::Plane::key:Exl3ExactKVExtent::Plane::value,page_->first+64));
    }
    bool current() const noexcept {
        for(const auto& plane:planes_)if(!plane || !plane->backing_current())return false;
        return true;
    }
    bool same(const Exl3DevicePageKey& other) const noexcept {
        return current() && other.current() && model_.get()==other.model_.get() &&
            same_owner(model_,other.model_) && page_.get()==other.page_.get() &&
            same_owner(page_,other.page_) && position_==other.position_;
    }
    bool same_model_position(const Exl3DevicePageKey& other) const noexcept {
        return current() && other.current() && model_.get()==other.model_.get() &&
            same_owner(model_,other.model_) && position_==other.position_;
    }
    // Deliberately coarse bucket selector; callers must still use same().
    std::uint64_t shortlist() const noexcept {return static_cast<std::uint64_t>(planes_[0]->first());}
    const std::shared_ptr<const Exl3ExactKVPage>& page() const noexcept {return page_;}
    // Host backing retained by this key, independent of checkpoint-index
    // membership. May overlap a live root's charge; not globally additive.
    template<class Visitor> void visit_retained_host_allocations(Visitor&& visitor) const {
        if(!current())throw std::invalid_argument("device page retained host backing changed");
        visitor(page_.get(),sizeof(Exl3ExactKVPage));
        const auto visit=[&](const auto& plane) {
            if(plane.capacity()>std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t))
                throw std::overflow_error("device page retained host bytes overflow");
            if(plane.capacity())visitor(plane.data(),plane.capacity()*sizeof(std::uint16_t));
        };
        for(int bank=0;bank<16;++bank){visit(page_->k[bank]);visit(page_->v[bank]);}
    }
    std::uint64_t retained_host_bytes() const {
        std::uint64_t bytes=0;
        visit_retained_host_allocations([&](const void*,std::size_t extent) {
            if(extent>std::numeric_limits<std::uint64_t>::max()-bytes)
                throw std::overflow_error("device page retained host bytes overflow");
            bytes+=extent;
        });
        return bytes;
    }
};
}
