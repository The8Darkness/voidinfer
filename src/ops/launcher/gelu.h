#pragma once

#include "core/tensor.h"
#include "voidinfer/ops/gelu.h"

#include <cuda_runtime.h>

namespace voidinfer::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace voidinfer::ops::detail
