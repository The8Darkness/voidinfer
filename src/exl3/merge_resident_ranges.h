#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::exl3 {
using ResidentRange=std::pair<std::uintptr_t,std::uintptr_t>;
inline std::size_t resident_lock_chunk_count(std::span<const ResidentRange> ranges,
    std::uint64_t chunk_bytes) {
    if(!chunk_bytes)throw std::invalid_argument("resident lock chunk extent");
    std::size_t count=0;
    for(const auto& range:ranges) {
        if(range.first>=range.second)throw std::invalid_argument("resident lock range extent");
        const auto bytes=range.second-range.first;
        const auto chunks=bytes/chunk_bytes+(bytes%chunk_bytes!=0);
        if(chunks>std::numeric_limits<std::size_t>::max()-count)
            throw std::overflow_error("resident lock descriptor count overflow");
        count+=static_cast<std::size_t>(chunks);
    }
    return count;
}
// Inputs are sorted, disjoint interval unions. This visitor does not allocate.
template<class Emit> void visit_resident_difference(std::span<const ResidentRange> a,
    std::span<const ResidentRange> b,Emit&& emit) {
    std::size_t j=0;
    for(const auto& range:a) {
        auto cursor=range.first;
        while(j<b.size() && b[j].second<=cursor)++j;
        for(auto k=j;k<b.size() && b[k].first<range.second;++k) {
            if(b[k].first>cursor)emit(ResidentRange{cursor,std::min(range.second,b[k].first)});
            cursor=std::max(cursor,b[k].second);
            if(cursor>=range.second)break;
        }
        if(cursor<range.second)emit(ResidentRange{cursor,range.second});
    }
}
inline std::size_t resident_difference_count(std::span<const ResidentRange> a,
    std::span<const ResidentRange> b) {
    std::size_t count=0;
    visit_resident_difference(a,b,[&](const auto&) {
        if(count==std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("resident difference descriptor count overflow");
        ++count;
    });
    return count;
}
inline std::vector<ResidentRange> resident_difference_precounted(std::span<const ResidentRange> a,
    std::span<const ResidentRange> b,std::size_t count) {
    std::vector<ResidentRange> result;
    result.reserve(count);
    visit_resident_difference(a,b,[&](const auto& range){
        if(result.size()==count)throw std::length_error("resident difference exceeded admitted count");
        result.push_back(range);
    });
    if(result.size()!=count)throw std::length_error("resident difference below admitted count");
    return result;
}
inline std::vector<ResidentRange> resident_difference(std::span<const ResidentRange> a,
    std::span<const ResidentRange> b) {
    return resident_difference_precounted(a,b,resident_difference_count(a,b));
}
// Inputs are normalized unions, output is a distinct precredited empty buffer.
// No sorting, allocation or pointer endpoint arithmetic is needed.
inline void merge_sorted_resident_ranges(std::span<const ResidentRange> a,
    std::span<const ResidentRange> b,std::vector<ResidentRange>& output) {
    if(!output.empty() || a.size()>output.capacity() || b.size()>output.capacity()-a.size())
        throw std::length_error("resident union output capacity");
    std::size_t i=0,j=0;
    while(i<a.size() || j<b.size()) {
        const auto range=j==b.size() || (i<a.size() && a[i]<b[j])?a[i++]:b[j++];
        if(!output.empty() && range.first<=output.back().second)
            output.back().second=std::max(output.back().second,range.second);
        else output.push_back(range);
    }
}
// Half-open address ranges. Callers validate/round extents before this union.
// Sorting and compaction preserve the allocation and its retained capacity.
inline void merge_resident_ranges_in_place(
    std::vector<std::pair<std::uintptr_t,std::uintptr_t>>& ranges) {
    std::sort(ranges.begin(),ranges.end());
    std::size_t written=0;
    for(std::size_t read=0;read<ranges.size();++read) {
        const auto range=ranges[read];
        if(written && range.first<=ranges[written-1].second)
            ranges[written-1].second=std::max(ranges[written-1].second,range.second);
        else ranges[written++]=range;
    }
    ranges.resize(written);
}
}
