#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::exl3 {
// Passed by value as a kernel argument. No device pointer table allocation or
// asynchronous metadata upload is needed; pointed-to pages have separate leases.
struct Exl3AttentionPageRanges {
    static constexpr int capacity=64;
    struct Range {const std::uint16_t* k=nullptr;const std::uint16_t* v=nullptr;int first=0,rows=0;};
    Range ranges[capacity]{};
    int count=0;
    void append(const std::uint16_t* k,const std::uint16_t* v,int first,int rows,int position) {
        if(count<0 || count>=capacity || !k || !v || first<0 || rows<=0 ||
            first>position || rows>position-first)
            throw std::invalid_argument("attention page range capacity/extent");
        for(int i=0;i<count;++i)
            if(first<ranges[i].first+ranges[i].rows && ranges[i].first<first+rows)
                throw std::invalid_argument("overlapping attention page ranges");
        ranges[count++]={k,v,first,rows};
    }
    bool contains_page(int first,int rows) const noexcept {
        for(int i=0;i<count;++i)if(ranges[i].first==first && ranges[i].rows==rows)return true;
        return false;
    }
};
}
