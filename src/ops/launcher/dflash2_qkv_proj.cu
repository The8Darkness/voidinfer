#include "ops/launcher/dflash2_qkv_proj.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 6144;
constexpr int kQueryRows = 4096;
constexpr int kKvRows = 1024;
constexpr int kInputRows = 5120;
constexpr int kRowsPerBlock = 8;
constexpr int kColsPerTile = 8;
constexpr int kStages = 2;

using Output = W8SplitOutput3<kQueryRows, kKvRows, kKvRows>;

template <bool Full>
void launch_slice(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                  cudaStream_t stream) {
    const auto* xp = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (x.ne[0] % 8) == 0 &&
                           (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? x.ne[0] / 1024 : 0;
    const dim3 grid(static_cast<unsigned>(div_up(kRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(x.ne[1], kColsPerTile)), 1u);
    const Output output{static_cast<__nv_bfloat16*>(q.data),
                        static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages,
                                 Full, W8Epilogue::Store, Output>
        <<<grid, kRowsPerBlock * 32, 0, stream>>>(
            xp, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, kRows, kInputRows, x.ne[1],
            weight.padded_shape[1], full_slabs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void dflash2_qkv_proj_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                             cudaStream_t stream) {
    const bool full = (x.ne[1] % kColsPerTile) == 0;
    for_each_token_slice(x.ne[1], kColsPerTile,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor q_slice = q.slice(1, offset, count);
                             Tensor k_slice = k.slice(1, offset, count);
                             Tensor v_slice = v.slice(1, offset, count);
                             if (full) {
                                 launch_slice<true>(x_slice, weight, q_slice, k_slice, v_slice,
                                                    stream);
                             } else {
                                 launch_slice<false>(x_slice, weight, q_slice, k_slice, v_slice,
                                                     stream);
                             }
                         });
}

} // namespace ninfer::ops::detail
