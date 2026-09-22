#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Projects the Qwen3.8 DFlash2 QKV parent directly into separate Q, K, and V
 * BF16 matrices. The W8G32_F16S RowSplit parent has logical shape
 * [6144, 5120] and row order [Q 4096, K 1024, V 1024].
 *
 * x is [5120,T], q is [4096,T], and k/v are [1024,T]. T is positive; the
 * implementation uses the existing W8 A16 small-T route and may tile T in
 * groups of eight. Outputs do not alias x or one another.
 */
void dflash2_qkv_proj(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                     cudaStream_t stream);

} // namespace ninfer::ops
