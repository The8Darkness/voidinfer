#pragma once
#include "exl3/paired_transform_extent.h"

namespace ninfer::exl3 {
// Private GDN hidden width. Inputs may share read-only storage; each output is
// retained independently, including the FP16 residual used by trace consumers.
inline void exl3_require_residual_norm_extents(int rows,
    const std::array<std::uintptr_t,5>& addresses) {
    if(rows<1 || rows>1024)
        throw std::invalid_argument("residual norm requires 1 through 1024 private rows");
    // left, right, weight, represented residual, normalized output
    for(unsigned input:{0u,1u})
        exl3_require_paired_transform_extents(rows,5120,
            {addresses[input],addresses[2],addresses[2],addresses[3],addresses[4]});
}
} // namespace ninfer::exl3
