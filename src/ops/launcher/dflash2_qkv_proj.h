#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void dflash2_qkv_proj_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                             Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
