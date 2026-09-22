#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::exl3 {
inline void exl3_require_prefill_width(int width) {
    if(width!=8 && width!=16 && width!=32 && width!=128 && width!=1024)
        throw std::invalid_argument("request prefill width must be8/16/32/128/1024");
}
inline int exl3_next_prefill_rows(int width,std::size_t remaining) {
    exl3_require_prefill_width(width);
    if(!remaining)return 0;
    const auto available=std::min<std::size_t>(width,remaining);
    for(int candidate:{1024,128,32,16})if(std::size_t(candidate)<=available)return candidate;
    return static_cast<int>(std::min<std::size_t>(8,available));
}
enum class Exl3PrefillPartitionRelation { identical, requires_numerical_qualification };
struct Exl3PrefillChunkStep { int first,rows,next_position; };
inline Exl3PrefillChunkStep exl3_prefill_chunk_step(
    int width,std::size_t remaining,int position,int capacity,bool actual_tail=false) {
    if(position<1 || capacity<position || remaining>static_cast<std::size_t>(capacity-position))
        throw std::invalid_argument("prefill suffix exceeds live context position/capacity");
    const int canonical=exl3_next_prefill_rows(width,remaining);
    const int rows=actual_tail && width>=32 && remaining>=17 && remaining<=31?
        static_cast<int>(remaining):canonical;
    return {position,rows,position+rows};
}
// Width alone is not an exactness guarantee: changed calls may change projection
// topology and recurrent history. Only identical complete partitions establish
// this scheduling equivalence; arithmetic-option/model equality is also required.
inline Exl3PrefillPartitionRelation exl3_prefill_partition_relation(
    int first_width,int second_width,std::size_t remaining) {
    exl3_require_prefill_width(first_width);exl3_require_prefill_width(second_width);
    if(first_width==second_width)return Exl3PrefillPartitionRelation::identical;
    while(remaining) {
        const int first=exl3_next_prefill_rows(first_width,remaining);
        if(first!=exl3_next_prefill_rows(second_width,remaining))
            return Exl3PrefillPartitionRelation::requires_numerical_qualification;
        remaining-=first;
    }
    return Exl3PrefillPartitionRelation::identical;
}
} // namespace ninfer::exl3
