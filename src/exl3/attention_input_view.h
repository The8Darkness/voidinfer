#pragma once
#include "exl3/attention_page_ranges.h"
#include <cstddef>
#include <algorithm>
#include <array>
#include <memory>

namespace ninfer::exl3 {
// Read-only numerical projection of retained input owners. Lifetime/readiness
// remain in the context's working allocation, prefix Use and page reader leases.
// This object is deliberately unable to certify ownership from a raw pointer.
struct Exl3AttentionInputView {
    const std::uint16_t* local_k=nullptr;
    const std::uint16_t* local_v=nullptr;
    int capacity=0;
    std::size_t stride=1024;
    Exl3AttentionPageRanges shared;
    void require_geometry(int position,int rows) const {
        if(!local_k || !local_v || stride!=1024 || position<0 || rows<1 || capacity<rows ||
            position>capacity-rows || shared.count<0 || shared.count>Exl3AttentionPageRanges::capacity)
            throw std::invalid_argument("attention input local geometry/representation");
        Exl3AttentionPageRanges checked;
        for(int i=0;i<shared.count;++i) {
            const auto& r=shared.ranges[i];checked.append(r.k,r.v,r.first,r.rows,position);
        }
    }
};
// Ownership accompanies geometry but does not replace final-use leases. The
// caller retains event/stream proofs until every numerical consumer completes.
class Exl3OwnedAttentionInputView {
    Exl3AttentionInputView view_;
    std::shared_ptr<const void> local_k_,local_v_;
    std::array<std::shared_ptr<const void>,Exl3AttentionPageRanges::capacity> segments_{};
    // History rows read from the local planes. Shared segments are kept in the
    // numerical view above; current request rows are produced by the layer and
    // therefore remain governed by its existing pre-attention event/order.
    std::array<Exl3AttentionPageRanges::Range,Exl3AttentionPageRanges::capacity+1> local_{};
    int local_count_=0;
    static bool overlaps(const Exl3AttentionPageRanges::Range& a,
        const Exl3AttentionPageRanges::Range& b) noexcept {
        return a.first<b.first+b.rows && b.first<a.first+a.rows;
    }
public:
    Exl3OwnedAttentionInputView(const std::uint16_t* k,const std::uint16_t* v,int capacity,
        std::shared_ptr<const void> k_owner,std::shared_ptr<const void> v_owner)
        :view_{k,v,capacity},local_k_(std::move(k_owner)),local_v_(std::move(v_owner)) {
        if(!local_k_ || !local_v_ || !local_k_.use_count() || !local_v_.use_count())
            throw std::invalid_argument("attention input missing working owner");
    }
    void append(const Exl3AttentionPageRanges::Range& range,std::shared_ptr<const void> owner,int position) {
        if(!owner || !owner.use_count())throw std::invalid_argument("attention input missing segment owner");
        auto next=view_.shared;
        next.append(range.k,range.v,range.first,range.rows,position);
        for(int i=0;i<local_count_;++i)if(overlaps(local_[i],range))
            throw std::invalid_argument("attention input local/shared overlap");
        const auto slot=view_.shared.count;
        view_.shared=next;
        segments_[slot]=std::move(owner);
    }
    void append_local(int first,int rows,int position) {
        Exl3AttentionPageRanges::Range range{view_.local_k,view_.local_v,first,rows};
        if(local_count_<0 || local_count_>=static_cast<int>(local_.size()) || first<0 || rows<=0 ||
            first>position || rows>position-first)
            throw std::invalid_argument("attention input local coverage extent");
        for(int i=0;i<local_count_;++i)if(overlaps(local_[i],range))
            throw std::invalid_argument("overlapping attention local ranges");
        for(int i=0;i<view_.shared.count;++i)if(overlaps(view_.shared.ranges[i],range))
            throw std::invalid_argument("attention input local/shared overlap");
        local_[local_count_++]=range;
    }
    // The HostKV uploader skips exactly the shared ranges passed to the
    // numerical reader. Declare the remaining chronological intervals as local
    // only after every shared segment and its owner token have been installed.
    void append_local_complement(int position) {
        if(position<0 || local_count_)throw std::invalid_argument("attention input local complement state");
        std::array<Exl3AttentionPageRanges::Range,Exl3AttentionPageRanges::capacity> ordered{};
        for(int i=0;i<view_.shared.count;++i)ordered[i]=view_.shared.ranges[i];
        std::sort(ordered.begin(),ordered.begin()+view_.shared.count,
            [](const auto& a,const auto& b){return a.first<b.first;});
        int cursor=0;
        for(int i=0;i<view_.shared.count;++i) {
            const auto& range=ordered[i];
            if(range.first>cursor)append_local(cursor,range.first-cursor,position);
            cursor=range.first+range.rows;
        }
        if(cursor<position)append_local(cursor,position-cursor,position);
    }
    const Exl3AttentionInputView& require_geometry(int position,int rows) const {
        view_.require_geometry(position,rows);
        std::array<Exl3AttentionPageRanges::Range,2*Exl3AttentionPageRanges::capacity+1> coverage{};
        int count=0;
        for(int i=0;i<local_count_;++i)coverage[count++]=local_[i];
        for(int i=0;i<view_.shared.count;++i)coverage[count++]=view_.shared.ranges[i];
        std::sort(coverage.begin(),coverage.begin()+count,
            [](const auto& a,const auto& b){return a.first<b.first;});
        int cursor=0;
        for(int i=0;i<count;++i) {
            if(coverage[i].first!=cursor || coverage[i].rows<=0 ||
                coverage[i].first>position || coverage[i].rows>position-coverage[i].first)
                throw std::invalid_argument("attention input incomplete/overlapping history coverage");
            cursor+=coverage[i].rows;
        }
        if(cursor!=position)
            throw std::invalid_argument("attention input incomplete/overlapping history coverage");
        return view_;
    }
};
}
