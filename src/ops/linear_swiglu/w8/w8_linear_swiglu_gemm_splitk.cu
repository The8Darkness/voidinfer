#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kIntermediate = 6144;
constexpr int kHidden       = 2048;
constexpr int kRowsPerCta   = 8;
constexpr int kFirstExactT  = 2;
constexpr int kLastExactT   = 48;
using ProjectionLauncher    = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

constexpr int kQwenIntermediate = 17408;
constexpr int kQwenRowsPerCta   = 8;
constexpr int kQwenFirstT       = 1;
constexpr int kQwenLastT        = 40;

struct W8SwiGluExactTRows {
    static constexpr int kOutputRowsPerCta = kRowsPerCta;
    int intermediate;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & (kRowsPerCta - 1)) +
               (local_row >= kRowsPerCta ? intermediate : 0);
    }
};

struct W8SwiGluExactTEpilogue {
    __nv_bfloat16* out;
    int rows;

    template <int ActiveCols>
    __device__ __forceinline__ void store_pair(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            out[static_cast<std::int64_t>(col0) * rows + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out[static_cast<std::int64_t>(col0 + 1) * rows + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

struct W8SwiGluQwenRows {
    static constexpr int kOutputRowsPerCta = kQwenRowsPerCta;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & (kQwenRowsPerCta - 1)) +
               (local_row >= kQwenRowsPerCta ? kQwenIntermediate : 0);
    }
};

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    constexpr auto ScaleAccess =
        ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    using Geometry = W8LinearGeometry<2 * kIntermediate, kHidden>;
    using Schedule =
        std::conditional_t<(ActiveCols <= 32), W8SmallTMmaDefaultSchedule<TileCols, ActiveCols>,
                           W8SmallTMmaSchedule<4, TileCols, 3, ScaleAccess>>;
    const W8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const W8SwiGluExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const W8SwiGluExactTRows row_policy{kIntermediate};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput,
                          W8SwiGluExactTEpilogue, W8SwiGluExactTRows, true>
        <<<kIntermediate / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), ignored_output, epilogue, row_policy);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

// The Qwen3.8 gate/up research route keeps the latency-oriented production path at
// four K-warps (with the historical 22..24 exception).  The opt-in eight-warp
// schedule cuts the K-loop count in half; its one-block-per-SM variant is also
// available for register/occupancy tuning.
template <int ActiveCols>
void launch_qwen_active_cols_narrow(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream) {
    constexpr int kTileCols = ActiveCols <= 8    ? 8
                              : ActiveCols <= 16 ? 16
                              : ActiveCols <= 24 ? 24
                              : ActiveCols <= 32 ? 32
                                                 : 40;
    constexpr int kKWarps = ActiveCols >= 22 && ActiveCols <= 24 ? 8 : 4;
    constexpr auto kScaleAccess =
        ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    constexpr auto kActivationStage =
        ActiveCols <= 4 || (ActiveCols >= 9 && ActiveCols <= 15)
            ? W8SmallTMmaActivationStage::PaddedZero
            : W8SmallTMmaActivationStage::ActiveOnly;
    using Schedule = W8SmallTMmaSchedule<kKWarps, kTileCols, 2, kScaleAccess, Cache::ca, Cache::cg,
                                         kActivationStage>;
    using Geometry = W8MtpGateUpProjectionGeometry;
    static_assert((Geometry::kOutputRows / 2) % kQwenRowsPerCta == 0);
    const W8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data),
                                            kQwenIntermediate};
    const W8SwiGluExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data),
                                          kQwenIntermediate};
    const W8SwiGluQwenRows row_policy{};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput,
                          W8SwiGluExactTEpilogue, W8SwiGluQwenRows, true>
        <<<kQwenIntermediate / kQwenRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), ignored_output, epilogue, row_policy);
}

template <int ActiveCols, int KWarps, int MinBlocksPerSm>
void launch_qwen_active_cols_wide(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream) {
    constexpr int kTileCols = ActiveCols <= 8    ? 8
                              : ActiveCols <= 16 ? 16
                              : ActiveCols <= 24 ? 24
                              : ActiveCols <= 32 ? 32
                                                 : 40;
    constexpr auto kScaleAccess =
        ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    constexpr auto kActivationStage =
        ActiveCols <= 4 || (ActiveCols >= 9 && ActiveCols <= 15)
            ? W8SmallTMmaActivationStage::PaddedZero
            : W8SmallTMmaActivationStage::ActiveOnly;
    using Schedule = W8SmallTMmaSchedule<KWarps, kTileCols, MinBlocksPerSm, kScaleAccess,
                                         Cache::ca, Cache::cg, kActivationStage>;
    using Geometry = W8MtpGateUpProjectionGeometry;
    static_assert((Geometry::kOutputRows / 2) % kQwenRowsPerCta == 0);
    const W8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data),
                                            kQwenIntermediate};
    const W8SwiGluExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data),
                                          kQwenIntermediate};
    const W8SwiGluQwenRows row_policy{};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput,
                          W8SwiGluExactTEpilogue, W8SwiGluQwenRows, true>
        <<<kQwenIntermediate / kQwenRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), ignored_output, epilogue, row_policy);
}

template <std::size_t... Offsets>
constexpr auto make_qwen_narrow_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_qwen_active_cols_narrow<kQwenFirstT + static_cast<int>(Offsets)>...};
}

constexpr auto kQwenNarrowLaunchers =
    make_qwen_narrow_launchers(std::make_index_sequence<kQwenLastT - kQwenFirstT + 1>{});

template <std::size_t... Offsets>
constexpr auto make_qwen_wide_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_qwen_active_cols_wide<kQwenFirstT + static_cast<int>(Offsets), 8, 2>...};
}

constexpr auto kQwenWideLaunchers =
    make_qwen_wide_launchers(std::make_index_sequence<kQwenLastT - kQwenFirstT + 1>{});

template <std::size_t... Offsets>
constexpr auto make_qwen_wide_minblocks_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_qwen_active_cols_wide<kQwenFirstT + static_cast<int>(Offsets), 8, 1>...};
}

constexpr auto kQwenWideMinBlocksLaunchers =
    make_qwen_wide_minblocks_launchers(std::make_index_sequence<kQwenLastT - kQwenFirstT + 1>{});

template <std::size_t... Offsets>
constexpr auto make_qwen_sixteen_warp_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_qwen_active_cols_wide<kQwenFirstT + static_cast<int>(Offsets), 16, 1>...};
}

constexpr auto kQwenSixteenWarpLaunchers =
    make_qwen_sixteen_warp_launchers(std::make_index_sequence<kQwenLastT - kQwenFirstT + 1>{});

enum class QwenGateUpWarpMode : std::uint8_t {
    Auto,
    Production,
    Wide,
    WideMinBlocks,
    SixteenWarps,
};

QwenGateUpWarpMode qwen_gate_up_warp_mode() noexcept {
    static const QwenGateUpWarpMode mode = [] {
        const char* value = std::getenv("NINFER_DFLASH2_W8_GATEUP_WARPS");
        if (value == nullptr || value[0] == '\0') { return QwenGateUpWarpMode::Auto; }
        if (value[0] == '0' && value[1] == '\0') { return QwenGateUpWarpMode::Production; }
        if (value[0] == '8' && value[1] == '\0') { return QwenGateUpWarpMode::Wide; }
        if (value[0] == '8' && value[1] == '1' && value[2] == '\0') {
            return QwenGateUpWarpMode::WideMinBlocks;
        }
        if (value[0] == '1' && value[1] == '6' && value[2] == '\0') {
            return QwenGateUpWarpMode::SixteenWarps;
        }
        return QwenGateUpWarpMode::Auto;
    }();
    return mode;
}

} // namespace

void w8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream) {
    if (x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT) {
        throw std::invalid_argument("W8 LinearSwiGLU exact split-K requires T=2..48");
    }
    kLaunchers[x.ne[1] - kFirstExactT](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

void w8_linear_swiglu_qwen_small_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    if (x.ne[1] < kQwenFirstT || x.ne[1] > kQwenLastT) {
        throw std::invalid_argument("W8 Qwen LinearSwiGLU small-T requires T=1..40");
    }
    const QwenGateUpWarpMode mode = qwen_gate_up_warp_mode();
    const auto& launchers = mode == QwenGateUpWarpMode::Production
                                ? kQwenNarrowLaunchers
                                : mode == QwenGateUpWarpMode::Wide
                                    ? kQwenWideLaunchers
                                    : mode == QwenGateUpWarpMode::WideMinBlocks
                                        ? kQwenWideMinBlocksLaunchers
                                        : mode == QwenGateUpWarpMode::SixteenWarps
                                            ? kQwenSixteenWarpLaunchers
                                        : kQwenNarrowLaunchers;
    launchers[x.ne[1] - kQwenFirstT](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
