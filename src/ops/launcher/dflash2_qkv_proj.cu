#include "ops/launcher/dflash2_qkv_proj.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 6144;
constexpr int kQueryRows = 4096;
constexpr int kKvRows = 1024;
constexpr int kInputRows = 5120;

using Output = W8SplitOutput3<kQueryRows, kKvRows, kKvRows>;

using Geometry      = W8LinearGeometry<kRows, kInputRows>;
using LargeSchedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2>;
using SmallLaunch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_small_slice(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                        cudaStream_t stream) {
    constexpr int kTileCols = ActiveCols <= 8    ? 8
                              : ActiveCols <= 16 ? 16
                              : ActiveCols <= 24 ? 24
                              : ActiveCols <= 32 ? 32
                              : ActiveCols <= 40 ? 40
                                                 : 48;
    using Schedule = W8SmallTMmaDefaultSchedule<kTileCols, ActiveCols>;
    static_assert((kRows % Schedule::kRowsPerCta) == 0);
    static_assert((kInputRows % Schedule::kGroupK) == 0);

    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, Output>
        <<<kRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <std::size_t... Offsets>
constexpr auto make_small_launchers(std::index_sequence<Offsets...>) {
    return std::array<SmallLaunch, sizeof...(Offsets)>{
        &launch_small_slice<1 + static_cast<int>(Offsets)>...};
}

constexpr auto kSmallLaunchers = make_small_launchers(std::make_index_sequence<48>{});

template <bool Full>
void launch_large_slice(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                        cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(div_up(kRows, LargeSchedule::BM)),
                    static_cast<unsigned>(div_up(x.ne[1], LargeSchedule::BN)), 1u);
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_gemm_mma_kernel<LargeSchedule, Full, W8Epilogue::Store, Output>
        <<<grid, LargeSchedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, kRows, kInputRows, x.ne[1],
            weight.padded_shape[1]);
}

void launch_small(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                  cudaStream_t stream) {
    // The small-T core owns one fixed token tile per launch. DFlash2 normally reaches this
    // route with T=8*batch (K=7), so use the largest compiled tile for multi-agent batches and
    // keep a short final tile exact.
    constexpr int kMaxTile = 48;
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += kMaxTile) {
        const std::int32_t count = std::min<std::int32_t>(kMaxTile, x.ne[1] - offset);
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice = q.slice(1, offset, count);
        Tensor k_slice = k.slice(1, offset, count);
        Tensor v_slice = v.slice(1, offset, count);
        kSmallLaunchers[static_cast<std::size_t>(count - 1)](x_slice, weight, q_slice, k_slice,
                                                             v_slice, stream);
    }
}

void launch_large(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                  cudaStream_t stream) {
    const bool full = (x.ne[1] % LargeSchedule::BN) == 0;
    for_each_token_slice(x.ne[1], LargeSchedule::BN,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor x_slice = x.slice(1, offset, count);
                             Tensor q_slice = q.slice(1, offset, count);
                             Tensor k_slice = k.slice(1, offset, count);
                             Tensor v_slice = v.slice(1, offset, count);
                             if (full) {
                                 launch_large_slice<true>(x_slice, weight, q_slice, k_slice, v_slice,
                                                           stream);
                             } else {
                                 launch_large_slice<false>(x_slice, weight, q_slice, k_slice, v_slice,
                                                            stream);
                             }
                         });
}

} // namespace

void dflash2_qkv_proj_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                             cudaStream_t stream) {
    if (x.ne[1] <= 48) {
        launch_small(x, weight, q, k, v, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    launch_large(x, weight, q, k, v, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
