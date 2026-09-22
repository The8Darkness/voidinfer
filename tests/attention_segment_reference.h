#pragma once
#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace attention_reference {
using Storage=std::vector<std::uint16_t>;
using Owner=std::shared_ptr<const Storage>;
struct Segment {
    std::size_t first=0,rows=0,storage_row=0,registry_index=0;
    Owner owner;
};
// Independent host oracle: expand chronological represented rows first, rather
// than duplicate the candidate's per-key range lookup. Registry identity is
// supplied separately by the fixture that allocated the source buffers.
inline Storage expand(std::span<const std::uint16_t> local,std::size_t stride,
    std::size_t published_rows,std::span<const Segment> segments,std::span<const Owner> registry) {
    if(!stride || published_rows>local.size()/stride)
        throw std::invalid_argument("reference local extent");
    Storage result(local.begin(),local.begin()+published_rows*stride);
    std::vector<bool> assigned(published_rows,false);
    for(const auto& segment:segments) {
        if(!segment.owner || segment.registry_index>=registry.size())
            throw std::invalid_argument("reference missing registered owner");
        const auto& expected=registry[segment.registry_index];
        if(!expected || expected.get()!=segment.owner.get() ||
            expected.owner_before(segment.owner) || segment.owner.owner_before(expected))
            throw std::invalid_argument("reference wrong segment owner");
        if(!segment.rows || segment.first>published_rows || segment.rows>published_rows-segment.first ||
            segment.storage_row>segment.owner->size()/stride ||
            segment.rows>segment.owner->size()/stride-segment.storage_row)
            throw std::invalid_argument("reference segment extent");
        for(std::size_t row=0;row<segment.rows;++row) {
            const auto logical=segment.first+row;
            if(assigned[logical])throw std::invalid_argument("reference overlapping segments");
            assigned[logical]=true;
            std::copy_n(segment.owner->begin()+(segment.storage_row+row)*stride,stride,
                result.begin()+logical*stride);
        }
    }
    return result;
}
inline bool visible(int position,int queries,int query,int key) noexcept {
    // Widen before addition; no candidate endpoint or segment helper is used.
    return position>=0 && queries>0 && query>=0 && query<queries && key>=0 &&
        static_cast<std::int64_t>(key)<=static_cast<std::int64_t>(position)+query;
}
}
