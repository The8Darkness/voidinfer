#pragma once
#include "exl3/exact_kv_extent.h"
#include <span>

namespace ninfer::exl3 {
// Geometry proof only. Strong published page ownership and device readiness stay
// in the caller's independent extent/reader leases. Private current rows are
// produced by the layer after this complete history prefix has been established.
inline void exl3_require_attention_history(std::span<const std::shared_ptr<const Exl3ExactKVPage>> pages,
    int position,int bank) {
    if(position<0 || bank<0 || bank>=16)throw std::invalid_argument("attention history position/bank");
    int cursor=0;
    for(const auto& page:pages) {
        if(!page || page->first!=cursor || page->rows<1 || page->rows>64 || cursor>=position)
            throw std::invalid_argument("attention history gap/overlap/order");
        const auto k=Exl3ExactKVExtent::view(page,bank,Exl3ExactKVExtent::Plane::key,position);
        const auto v=Exl3ExactKVExtent::view(page,bank,Exl3ExactKVExtent::Plane::value,position);
        if(k.first()!=cursor || v.first()!=cursor || k.rows()!=v.rows() || k.rows()<1)
            throw std::invalid_argument("attention history missing plane");
        cursor+=k.rows();
    }
    if(cursor!=position)throw std::invalid_argument("attention history incomplete published prefix");
}
}
