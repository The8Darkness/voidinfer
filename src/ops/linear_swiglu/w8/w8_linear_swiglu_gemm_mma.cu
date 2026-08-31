#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGateUpRows   = 12288;
constexpr int kIntermediate = 6144;
constexpr int kHidden       = 2048;

constexpr int kQwenGateUpRows   = 34816;
constexpr int kQwenIntermediate = 17408;
constexpr int kQwenHidden       = 5120;

template <class Schedule, bool Full>
void launch_variant(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    static_assert((kIntermediate % (Schedule::BM / 2)) == 0);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const dim3 grid(kIntermediate / (Schedule::BM / 2),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::SwiGluSplitHalf>
        <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                 static_cast<const std::uint8_t*>(w.qdata),
                                                 static_cast<const std::uint8_t*>(w.scales), output,
                                                 kGateUpRows, kHidden, x.ne[1], kHidden);
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out,
                  cudaStream_t stream) {
    if ((x.ne[1] % Schedule::BN) == 0) {
        launch_variant<Schedule, true>(x, w, out, stream);
    } else {
        launch_variant<Schedule, false>(x, w, out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <bool Full>
void launch_qwen_large(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    // This conservative larger-T route keeps the same row-split SwiGLU epilogue as the tuned
    // Qwen3.6/35B kernels.  The DFlash2 block normally reaches it only at batch 8 (T=64); the
    // exact small-T route above handles the latency-critical C1/C2/C4 widths.
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kQwenIntermediate};
    const dim3 grid(kQwenIntermediate / (Schedule::BM / 2),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::SwiGluSplitHalf>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output, kQwenGateUpRows, kQwenHidden,
            x.ne[1], kQwenHidden);
}

template <bool Full>
void launch_qwen_large_bm64(const Tensor& x, const Weight& w, Tensor& out,
                            cudaStream_t stream) {
    // One CTA owns 32 output rows instead of 16. The two-row-warp SwiGLU epilogue uses the
    // existing shared up-accumulator path, while the 29 KiB schedule remains resident at two
    // CTAs/SM on SM120. This is intentionally an opt-in C8 research route.
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 32, 16, 2>;
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), kQwenIntermediate};
    const dim3 grid(kQwenIntermediate / (Schedule::BM / 2),
                    static_cast<unsigned>(div_up(x.ne[1], Schedule::BN)), 1u);
    w8_rowsplit_gemm_mma_kernel<Schedule, Full, W8Epilogue::SwiGluSplitHalf>
        <<<grid, Schedule::THREADS, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output, kQwenGateUpRows, kQwenHidden,
            x.ne[1], kQwenHidden);
}

} // namespace

void w8_linear_swiglu_mma_r32_c32_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 32, 32, 16, 4>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c48_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 48, 32, 16, 4>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c64_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c80_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 80, 32, 16, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c96_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 96, 32, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r32_c128_launch(const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c32_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 32, 64, 16, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c48_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 48, 64, 16, 3>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c64_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 64, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c80_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 80, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c96_launch(const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 96, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r128_c64_launch(const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 64, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_mma_r128_c80_launch(const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream) {
    using Schedule = W8RowSplitMmaGemmSchedule<128, 80, 64, 16, 2>;
    launch_route<Schedule>(x, w, out, stream);
}

void w8_linear_swiglu_qwen_large_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream) {
    if (x.ne[1] <= 40) {
        throw std::invalid_argument("W8 Qwen LinearSwiGLU large-T requires T>40");
    }
    const char* schedule_env = std::getenv("NINFER_DFLASH2_W8_GATEUP_LARGE_SCHEDULE");
    const bool use_bm64 = schedule_env != nullptr && schedule_env[0] == '6' &&
                          schedule_env[1] == '4' && schedule_env[2] == '\0';
    if ((x.ne[1] % 64) == 0) {
        if (use_bm64) {
            launch_qwen_large_bm64<true>(x, w, out, stream);
        } else {
            launch_qwen_large<true>(x, w, out, stream);
        }
    } else {
        if (use_bm64) {
            launch_qwen_large_bm64<false>(x, w, out, stream);
        } else {
            launch_qwen_large<false>(x, w, out, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
