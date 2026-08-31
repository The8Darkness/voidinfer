#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a16_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
using ArgmaxLaunch =
    void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, std::int32_t, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Fp8VocabularyGeometry;
    using Schedule = typename Fp8VocabularyA16SmallTMmaProductionSchedule<ActiveTokens>::Type;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_small_t_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <int ActiveTokens>
void launch_argmax_exact(const Tensor& x, const Weight& weight, Tensor& out,
                         Tensor& scratch, std::int32_t valid_rows, cudaStream_t stream) {
    using Geometry = Fp8VocabularyGeometry;
    using Schedule = typename Fp8VocabularyA16SmallTMmaProductionSchedule<ActiveTokens>::Type;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const std::int32_t partial_rows = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const std::size_t partial_count =
        static_cast<std::size_t>(partial_rows) * static_cast<std::size_t>(ActiveTokens);
    const std::size_t index_offset =
        (partial_count * sizeof(__nv_bfloat16) + 15U) & ~std::size_t(15U);
    auto* partial_values = static_cast<__nv_bfloat16*>(scratch.data);
    auto* partial_indices = reinterpret_cast<std::int32_t*>(
        static_cast<std::uint8_t*>(scratch.data) + index_offset);
    const Fp8ArgmaxOutput output{valid_rows, nullptr, partial_values, partial_indices,
                                 partial_rows};
    fp8_a16_small_t_mma_kernel<Geometry, ActiveTokens, Schedule, Fp8ArgmaxOutput>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
    fp8_vocabulary_argmax_reduce_kernel<<<ActiveTokens, 256, 0, stream>>>(
        partial_values, partial_indices, static_cast<std::int32_t*>(out.data), partial_rows,
        ActiveTokens);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kFp8VocabularyFirstA16SmallTMmaT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kFp8VocabularyLastA16SmallTMmaT -
                                            kFp8VocabularyFirstA16SmallTMmaT + 1>{});

template <std::size_t... Offsets>
constexpr auto make_argmax_launchers(std::index_sequence<Offsets...>) {
    return std::array<ArgmaxLaunch, sizeof...(Offsets)>{
        &launch_argmax_exact<kFp8VocabularyFirstA16SmallTMmaT + static_cast<int>(Offsets)>...};
}

constexpr auto kArgmaxLaunchers =
    make_argmax_launchers(std::make_index_sequence<kFp8VocabularyLastA16SmallTMmaT -
                                                   kFp8VocabularyFirstA16SmallTMmaT + 1>{});

} // namespace

void launch_fp8_vocabulary_a16_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    if (weight.n != Fp8VocabularyGeometry::kOutputRows ||
        weight.k != Fp8VocabularyGeometry::kInputRows ||
        x.ne[1] < kFp8VocabularyFirstA16SmallTMmaT || x.ne[1] > kFp8VocabularyLastA16SmallTMmaT) {
        throw std::invalid_argument("fp8 vocabulary A16 small-T MMA: invalid exact problem");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8VocabularyFirstA16SmallTMmaT);
    kLaunchers[index](x, weight, out, stream);
}

void launch_fp8_vocabulary_a16_small_t_argmax(const Tensor& x, const Weight& weight, Tensor& out,
                                              std::int32_t valid_rows, Tensor& scratch,
                                              cudaStream_t stream) {
    if (weight.n != Fp8VocabularyGeometry::kOutputRows ||
        weight.k != Fp8VocabularyGeometry::kInputRows || valid_rows <= 0 ||
        valid_rows > Fp8VocabularyGeometry::kOutputRows ||
        x.ne[1] < kFp8VocabularyFirstA16SmallTMmaT ||
        x.ne[1] > kFp8VocabularyLastA16SmallTMmaT) {
        throw std::invalid_argument("fp8 vocabulary A16 small-T argmax: invalid exact problem");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8VocabularyFirstA16SmallTMmaT);
    kArgmaxLaunchers[index](x, weight, out, scratch, valid_rows, stream);
}

} // namespace ninfer::ops::detail
