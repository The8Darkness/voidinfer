#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops {

// Apply the five lane dimensions of a Sylvester transform to independent columns. Keeping the
// column loop inside each stage exposes their independent shuffle instructions while preserving
// the butterfly and rounding order of every column.
template <int Columns>
__device__ __forceinline__ void hadamard_d32_columns_inplace(float (&values)[Columns], int lane) {
    constexpr unsigned FullMask = 0xffffffffu;

#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const float value = values[column];
            const float peer  = __shfl_xor_sync(FullMask, value, stride);
            values[column] = (lane & stride) == 0 ? __fadd_rn(value, peer) : __fsub_rn(peer, value);
        }
    }
}

// Factorized H256 support for latency-oriented kernels that distribute one D256 row across four
// G64 warps. The H64 fragments are intentionally unnormalized; the final H4 leaf applies 2^-4.
__device__ __forceinline__ void hadamard_d64_fragment_inplace(float (&values)[2], int lane) {
    hadamard_d32_columns_inplace(values, lane);
    const float low  = values[0];
    const float high = values[1];
    values[0]        = __fadd_rn(low, high);
    values[1]        = __fsub_rn(low, high);
}

__device__ __forceinline__ float
normalized_hadamard_d256_group_value_from_h64(float x0, float x1, float x2, float x3, int group) {
    const float y0  = (group & 1) == 0 ? __fadd_rn(x0, x1) : __fsub_rn(x0, x1);
    const float y1  = (group & 1) == 0 ? __fadd_rn(x2, x3) : __fsub_rn(x2, x3);
    const float out = (group & 2) == 0 ? __fadd_rn(y0, y1) : __fsub_rn(y0, y1);
    return __fmul_rn(out, 0x1p-4f);
}

// One full warp owns one D256 row. Lane l carries dimensions l+32*r in values[r]. The fixed
// normalized Sylvester transform is shared by persistent K preparation and transient Q
// preparation; it never materializes an intermediate row outside registers.
__device__ __forceinline__ void normalized_hadamard_d256_inplace(float (&values)[8], int lane) {
    hadamard_d32_columns_inplace(values, lane);

#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low              = values[base + offset];
                const float high             = values[base + offset + span];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }

#pragma unroll
    for (float& value : values) { value = __fmul_rn(value, 0x1p-4f); }
}

// Packed-row layout used by the Q2 attention decoder.  Lane l owns dimensions
//   [4*l + 0..3] and [128 + 4*l + 0..3],
// so it can read one packed byte from each half of the 256-wide row.  This is
// the same H256 transform as normalized_hadamard_d256_inplace(), factored over
// the natural dimension bits: H4 on bits [0,1], H32 across lanes on bits
// [2,6], and H2 on bit 7.  Keeping the transform in this layout removes the
// four-lane broadcasts that otherwise duplicate every Q2 code byte.
__device__ __forceinline__ void normalized_hadamard_d256_packed_inplace(
    float (&values)[8], int lane) {
#pragma unroll
    for (int stride = 1; stride <= 2; stride <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * stride) {
#pragma unroll
            for (int offset = 0; offset < stride; ++offset) {
                const float low              = values[base + offset];
                const float high             = values[base + offset + stride];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + stride] = __fsub_rn(low, high);
            }
        }
    }

    hadamard_d32_columns_inplace(values, lane);

#pragma unroll
    for (int offset = 0; offset < 4; ++offset) {
        const float low       = values[offset];
        const float high      = values[offset + 4];
        values[offset]        = __fadd_rn(low, high);
        values[offset + 4]    = __fsub_rn(low, high);
    }

#pragma unroll
    for (float& value : values) { value = __fmul_rn(value, 0x1p-4f); }
}

} // namespace ninfer::ops
