#pragma once
#include "exl3/exact_kv_extent.h"
#include <cstddef>

namespace ninfer::exl3 {
struct Exl3ExactPageExtensionPlan {
    std::size_t all=0,fresh=0;
    static Exl3ExactPageExtensionPlan derive(int old_position,int new_position) {
        if(old_position<0 || new_position<old_position)
            throw std::invalid_argument("exact page extension position range");
        constexpr auto capacity=static_cast<std::size_t>(Exl3ExactKVPage::token_capacity);
        const auto end=static_cast<std::size_t>(new_position);
        const auto count=end/capacity+(end%capacity!=0);
        return {count,new_position==old_position?0:count-static_cast<std::size_t>(old_position)/capacity};
    }
};
inline bool exl3_exact_tail_preallocated(const Exl3ExactKVPage& page) noexcept {
    if(page.rows<1 || page.rows>=Exl3ExactKVPage::token_capacity)return false;
    const auto used=static_cast<std::size_t>(page.rows)*Exl3ExactKVExtent::stride;
    constexpr auto full=Exl3ExactKVPage::token_capacity*Exl3ExactKVExtent::stride;
    for(int bank=0;bank<16;++bank)
        if(page.k[bank].size()!=used || page.v[bank].size()!=used ||
            page.k[bank].capacity()<full || page.v[bank].capacity()<full)return false;
    return true;
}
inline void exl3_extend_preallocated_tail(Exl3ExactKVPage& page,int rows) {
    if(!exl3_exact_tail_preallocated(page) || rows<page.rows || rows>Exl3ExactKVPage::token_capacity)
        throw std::invalid_argument("exact tail lacks complete preallocated planes");
    // uint16_t growth within existing capacity cannot allocate or throw.
    for(int bank=0;bank<16;++bank) {
        page.k[bank].resize(static_cast<std::size_t>(rows)*Exl3ExactKVExtent::stride);
        page.v[bank].resize(static_cast<std::size_t>(rows)*Exl3ExactKVExtent::stride);
    }
    page.rows=rows;
}
}
