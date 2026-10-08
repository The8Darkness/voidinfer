#pragma once

#include "exl3/sibling_rows.h"

namespace ninfer::exl3 {

// Verifier row layout {c, packed offsets} (exl3_sibling_layout). c == 0: row
// r is at logical offset r. With c > 0 chain rows, rows >= c are sibling
// leaves: sibling j = row - c sits at logical offset o_j (1 <= o_j < c), its
// ancestors are chain rows 0..o_j - 1, and it is never committed by the
// forward itself.
__device__ __forceinline__ int sibling_chain_rows(const int* layout) {
    return layout ? layout[0] : 0;
}
__device__ __forceinline__ int sibling_offset(int sibling, int packed) {
    return (packed >> (kExl3SiblingOffsetBits * sibling)) & kExl3SiblingOffsetMask;
}
__device__ __forceinline__ int sibling_row_offset(int row, const int* layout) {
    const int chain = sibling_chain_rows(layout);
    return chain > 0 && row >= chain ? sibling_offset(row - chain, layout[1]) : row;
}

} // namespace ninfer::exl3
