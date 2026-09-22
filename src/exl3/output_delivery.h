#pragma once
#include <algorithm>
#include <cstddef>
#include <string_view>
#include <limits>
#include <stdexcept>

namespace ninfer::exl3 {
inline constexpr std::size_t output_delivery_byte_limit=64*1024;
inline constexpr std::size_t output_delivery_outstanding_limit=2;
// Admission policy, not a promise about an STL implementation's growth rule.
// Actual capacity is checked against this ceiling before publishing the owner.
inline std::size_t output_text_capacity_ceiling(std::size_t visible_bytes) {
    if(visible_bytes>(std::numeric_limits<std::size_t>::max()-32)/2)
        throw std::overflow_error("output text capacity ceiling overflow");
    return visible_bytes*2+32;
}
inline std::size_t output_delivery_slot_capacity_ceiling() {
    return output_text_capacity_ceiling(output_delivery_byte_limit);
}
inline bool output_result_slots_available(std::size_t used,std::size_t allowance,
    std::size_t capacity,std::size_t append) noexcept {
    return used<=allowance && used<=capacity && append<=allowance-used && append<=capacity-used;
}
inline bool output_result_capacity_accepted(std::size_t token_allowance,std::size_t token_capacity,
    std::size_t text_bound,std::size_t text_ceiling,std::size_t reasoning_capacity,
    std::size_t content_capacity) noexcept {
    return token_capacity==token_allowance && text_bound<=text_ceiling &&
        reasoning_capacity>=text_bound && reasoning_capacity<=text_ceiling &&
        content_capacity>=text_bound && content_capacity<=text_ceiling;
}
// Input is committed frontend UTF-8. Do not split a code point when limiting
// a sink message; zero means the remaining batch credit cannot fit its next one.
inline std::size_t output_delivery_prefix(std::string_view text,std::size_t credit) noexcept {
    auto bytes=std::min(text.size(),credit);
    if(bytes<text.size())while(bytes && (static_cast<unsigned char>(text[bytes])&0xc0)==0x80)--bytes;
    return bytes;
}
}
