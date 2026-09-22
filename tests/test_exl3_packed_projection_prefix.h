#pragma once
#include <array>

// Every legal two-lane row split for the target M2..M16 capability. Operator
// fixtures consume this common matrix so a newly admitted family cannot claim
// whole-menu coverage from a few representative shapes.
inline constexpr auto packed_projection_row_pairs = [] {
    std::array<std::array<int,2>,64> result{};
    std::size_t index=0;
    for(int first=1;first<=8;++first)
        for(int second=1;second<=8;++second)
            result[index++]={first,second};
    return result;
}();

inline void prefill_packed_projection_prefix(
    Exl3TextContext& context,
    std::span<const std::int64_t> tokens) {
    require(tokens.size() == 512, "packed projection prefix extent");
    context.prefill(tokens.first(16));
    std::size_t cursor = 16;
    for (int block = 0; block < 3; ++block, cursor += 128)
        context.append_exact_prefill_wide(tokens.subspan(cursor, 128));
    for (int block = 0; block < 3; ++block, cursor += 32)
        context.append_exact_prefill_wide(tokens.subspan(cursor, 32));
    context.append_exact_prefill_wide(tokens.subspan(cursor, 16));
    cursor += 16;
    require(cursor == tokens.size(), "packed projection prefix partition");
    context.finish_exact_prefill();
}
