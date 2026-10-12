#pragma once

namespace ninfer::exl3 {

// Per-context rotary position map (Exl3TextContext::set_rope_map). Device ints:
// [0] table rows, [1] delta, [2..3] device pointer to a row-major int [rows][3]
// table of the frontend's M-RoPE (t,h,w) positions. Cache rows below [0] take
// the table entry of their pair's axis (interleaved sections 11/11/10: pair%3);
// later rows rotate at position + delta. Text requests keep rows=delta=0, so
// the plain cache position is used.
__device__ __forceinline__ int exl3_rope_position(const int* map, int position, int pair) {
    if (!map) return position;
    if (position < map[0]) {
        const int* table = *reinterpret_cast<const int* const*>(map + 2);
        return table[position * 3 + pair % 3];
    }
    return position + map[1];
}

} // namespace ninfer::exl3
