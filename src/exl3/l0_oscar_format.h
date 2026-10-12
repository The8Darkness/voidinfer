#pragma once
// L0 OSCAR code format (l0_oscar.cuh), shared by the device kernels and the
// host-side parking / pinned pools. Per committed row and KV head: 256 rotated
// dims as asymmetric 4-bit codes (two dims per byte, dim 2b in the low nibble
// of byte b) and FP16 (scale, zero) per 128-dim group.
#include <cstddef>

namespace voidinfer::exl3::l0_oscar_format {

constexpr int kCodeBits = 4;
constexpr int kCodeLevels = (1 << kCodeBits) - 1;
constexpr int kCodeBytes = 256 * kCodeBits / 8;   // per row and KV head
constexpr int kMetaHalves = 4;                     // scale0, zero0, scale1, zero1
// One row of all 4 KV heads, K and V: codes then FP16 meta.
constexpr std::size_t kRowCodeBytes = static_cast<std::size_t>(2) * 4 * kCodeBytes;
constexpr std::size_t kRowMetaBytes = static_cast<std::size_t>(2) * 4 * kMetaHalves * 2;

} // namespace voidinfer::exl3::l0_oscar_format
