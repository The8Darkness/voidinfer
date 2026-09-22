#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
namespace ninfer::exl3 {
// One actual B8 proposal, never a causal M16 sequence. Physical owners remain
// with the caller's exclusive lane; this fixed descriptor owns no ring storage.
struct Exl3DraftPrivateSegment {
    static constexpr int rows=8;
    static constexpr int ring_capacity=2048;
    std::uint64_t acquisition=0,execution=0;
    std::int64_t ring_base=0;
    int ring_count=0;
    std::array<std::int64_t,8> tokens{};
    std::array<std::int32_t,8> positions{};
    bool valid() const noexcept {
        const auto position=positions[0];
        if(!acquisition || !execution || ring_base<0 || ring_count<0 || ring_count>2047 ||
           position<0 || position>std::numeric_limits<std::int32_t>::max()-7 ||
           ring_base>position || position-ring_base!=ring_count)return false;
        for(int row=0;row<8;++row)
            if(tokens[row]<0 || tokens[row]>=248320 || positions[row]!=position+row)return false;
        return true;
    }
    static Exl3DraftPrivateSegment make(std::uint64_t acquisition,std::uint64_t execution,
        std::int64_t ring_base,int ring_count,int position,std::span<const std::int64_t> tokens);
    bool matches(std::uint64_t expected_acquisition,std::uint64_t expected_execution,
        std::int64_t expected_base,int expected_count,int expected_position,std::int64_t expected_seed) const noexcept {
        return valid() && acquisition==expected_acquisition && execution==expected_execution &&
            ring_base==expected_base && ring_count==expected_count &&
            positions[0]==expected_position && tokens[0]==expected_seed;
    }
    bool matches_tokens(std::span<const std::int64_t> expected) const noexcept {
        if(!valid() || expected.size()!=tokens.size())return false;
        for(std::size_t row=0;row<tokens.size();++row)
            if(tokens[row]!=expected[row])return false;
        return true;
    }
    // DFlash2's eight proposal rows form one private, fully visible block; they
    // are not a temporal W16 causal sequence with a peer. Committed history is
    // the lane's own contiguous ring view immediately preceding this block.
    bool block_mask_visible(int query_row,int key_row) const noexcept {
        return valid() && query_row>=0 && query_row<rows && key_row>=0 && key_row<rows;
    }
    bool ring_contains(std::int64_t absolute_position) const noexcept {
        return valid() && absolute_position>=ring_base &&
            absolute_position<ring_base+ring_count;
    }
    int ring_start_slot() const noexcept {
        return valid()?static_cast<int>(ring_base&(ring_capacity-1)):-1;
    }
    bool attention_view_matches(std::int64_t expected_base,int expected_count,
        int expected_position,int expected_rows) const noexcept {
        return valid() && expected_rows==rows && ring_base==expected_base &&
            ring_count==expected_count && positions[0]==expected_position &&
            (ring_count==0 || (ring_contains(ring_base) &&
                ring_contains(ring_base+ring_count-1))) &&
            block_mask_visible(0,rows-1) && block_mask_visible(rows-1,0);
    }
};
inline Exl3DraftPrivateSegment Exl3DraftPrivateSegment::make(std::uint64_t acquisition,std::uint64_t execution,
        std::int64_t ring_base,int ring_count,int position,std::span<const std::int64_t> tokens) {
        if(!acquisition || !execution || tokens.size()!=8 || ring_base<0 || ring_count<0 || ring_count>2047 ||
           position<0 || position>std::numeric_limits<std::int32_t>::max()-7 ||
           ring_base>position || position-ring_base!=ring_count)
            throw std::invalid_argument("private draft B8 segment scope/extent");
        Exl3DraftPrivateSegment result;result.acquisition=acquisition;result.execution=execution;
        result.ring_base=ring_base;result.ring_count=ring_count;
        for(int row=0;row<8;++row) {
            if(tokens[row]<0 || tokens[row]>=248320)throw std::invalid_argument("private draft segment token");
            result.tokens[row]=tokens[row];result.positions[row]=position+row;
        }
        return result;
    }
}
