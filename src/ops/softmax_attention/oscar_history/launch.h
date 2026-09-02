#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Historical-only diagnostic kernel contract for the official OSCAR INT2 G128
// representation.  All tensors are contiguous FP32/byte arrays:
//   q_rotated       [24, 256]
//   packed/meta     [history, 4, 64] / [history, 4, 4]
//   scores/softmax  [24, history]
//   output          [24, 256]
//
// This API intentionally does not include prefix/recent rows, rotations, or
// production cache objects.  It is the D4.2a historical-region benchmark path.
struct OscarHistoryKernelResources {
    int score_registers = 0;
    int softmax_registers = 0;
    int av_registers = 0;
    std::size_t score_static_shared_bytes = 0;
    std::size_t softmax_static_shared_bytes = 0;
    std::size_t av_static_shared_bytes = 0;
    int score_max_threads = 0;
    int softmax_max_threads = 0;
    int av_max_threads = 0;
};

void oscar_int2_g128_history_attention_launch(
    const float* q_rotated, const std::uint8_t* k_packed, const float* k_metadata,
    const std::uint8_t* v_packed, const float* v_metadata, std::int32_t history,
    float attention_scale, float* scores, float* softmax, float* output, cudaStream_t stream);

OscarHistoryKernelResources oscar_int2_g128_history_kernel_resources();

} // namespace ninfer::ops::detail
