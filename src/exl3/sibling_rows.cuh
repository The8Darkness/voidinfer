#pragma once

#include "exl3/sibling_rows.h"

namespace ninfer::exl3 {

// Verifier row layout. chain_rows == 0: row r is at logical offset r. With
// c > 0 chain rows, rows >= c are sibling leaves: row c + j replaces chain
// row j + 1 (logical offset j + 1, parent chain row j) and is never committed
// by the forward itself.
__device__ __forceinline__ int sibling_chain_rows(const int* chain_rows) {
    return chain_rows ? *chain_rows : 0;
}
__device__ __forceinline__ int sibling_row_offset(int row, int chain) {
    return chain > 0 && row >= chain ? row - chain + 1 : row;
}

} // namespace ninfer::exl3
