// Chunk-parallel gated delta rule for GDN prefill (WY/UT form, cf.
// flash-linear-attention chunk_gated_delta_rule, MIT). Same contract as the
// sequential resident recurrence: q,k FP32 [rows][16][128] (L2-normalized),
// v BF16 [rows][48][128], alpha (decay multiplier) and beta FP32 [rows][48],
// state FP32 [48][128 key][128 value] read and written back, output BF16
// [rows][48][128] scaled by 1/sqrt(128). Value head h reads key head h/3.
// q and k must have capacity for rows rounded up to 64; launch() zero-fills
// the pad rows so tensor-core fragments load whole chunks unguarded.
//
// Per chunk of C=64 rows with in-chunk cumulative log decay L (gamma=exp(L)):
//   A_ij = beta_i exp(L_i-L_j) k_i.k_j (j<i), T=(I+A)^-1,
//   w = T diag(beta gamma) K, u = T diag(beta) V, Kt_j = exp(L_C-L_j) k_j,
//   P_ij = q_i.k_j exp(L_i-L_j) [j<=i], Qg_i = gamma_i q_i,
//   Delta = u - w S0,  O = Qg S0 + P Delta,  S_C = gamma_C S0 + Kt^T Delta.
// Prep products (Gram, T*X, Q K^T) use TF32 tensor cores; the inverse, decays
// and the recurrent state are FP32. Chunk intermediates (w, u, Kt, P, Qg) and
// the scan's operand copies of S and Delta are FP16 (the same 10-bit mantissa
// TF32 multiplies with) feeding FP16 tensor cores with FP32 accumulation.
// Reassociated numerical policy; not bitwise equal to the sequential route.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstddef>
#include <cstdint>

namespace gdn_chunked {

constexpr int kD = 128, kH = 48, kKH = 16, kC = 64, kVT = 32;
constexpr int kQkStride = kKH * kD;   // row stride of q/k in floats
constexpr int kLd = kD + 4;           // padded smem stride (floats) for 128-wide tiles
constexpr int kLd64 = kC + 4;         // padded stride (floats) for 64-wide tiles
constexpr int kLdV = kVT + 4;         // padded stride (floats) for 32-wide value tiles
constexpr int kLdH = kVT + 8;         // padded stride (halves) for 32-wide value tiles
constexpr float kOutputScale = 0.08838834764831843f;

using namespace nvcuda;
using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major>;
using FragBc = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;
using HA = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
using HAc = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major>;
using HB = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
using HC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

template <class F> __device__ __forceinline__ void to_tf32(F& f) {
#pragma unroll
    for (int i = 0; i < f.num_elements; ++i) f.x[i] = wmma::__float_to_tf32(f.x[i]);
}

// Workspace per call of at most max_rows rows (per chunk-head tile):
//   w, u, kt, qg: half [64][128]; p: half [64][64]; decay: float
struct Workspace {
    half* w; half* u; half* kt; half* qg; half* p; float* decay;
};
inline std::size_t workspace_bytes(int max_rows) {
    const std::size_t tiles = static_cast<std::size_t>((max_rows + kC - 1) / kC) * kH;
    return tiles * ((4ull * kC * kD + kC * kC) * sizeof(half) + sizeof(float));
}
inline Workspace carve(void* base, int max_rows) {
    const std::size_t tiles = static_cast<std::size_t>((max_rows + kC - 1) / kC) * kH;
    auto* h = static_cast<half*>(base);
    Workspace ws;
    ws.w = h;
    ws.u = ws.w + tiles * kC * kD;
    ws.kt = ws.u + tiles * kC * kD;
    ws.qg = ws.kt + tiles * kC * kD;
    ws.p = ws.qg + tiles * kC * kD;
    ws.decay = reinterpret_cast<float*>(ws.p + tiles * kC * kC);
    return ws;
}
__host__ __device__ inline std::size_t tile_index(int chunk, int head) {
    return static_cast<std::size_t>(chunk) * kH + head;
}

// ---- Kernel 1: per (chunk, head): decays, Gram, blocked (I+A)^-1, w, u, Kt, P, Qg ----
constexpr std::size_t kPrepSmem =
    sizeof(float) * (kC * kLd + 2 * kC * kLd64 + 3 * 16 * 17 + 2 * kC);

// out (64 x 128 half, ld 128) = T (64x64 smem, lower) * X (64x128 smem). 8 warps.
__device__ __forceinline__ void t_times(const float* Ts, const float* Xs, float* stage,
                                        half* out, int warp, int lane) {
    for (int t = warp; t < 32; t += 8) {
        const int ti = t / 8, tj = t % 8;
        FragC acc; wmma::fill_fragment(acc, 0.0f);
        for (int kk = 0; kk <= ti * 16 + 8; kk += 8) {
            FragA a; FragB b;
            wmma::load_matrix_sync(a, Ts + ti * 16 * kLd64 + kk, kLd64);
            wmma::load_matrix_sync(b, Xs + kk * kLd + tj * 16, kLd);
            to_tf32(a); to_tf32(b);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(stage, acc, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32)
            out[(ti * 16 + e / 16) * kD + tj * 16 + e % 16] = __float2half_rn(stage[e]);
        __syncwarp();
    }
}

__global__ void __launch_bounds__(256) prep_kernel(const float* __restrict__ q,
    const float* __restrict__ k, const std::uint16_t* __restrict__ v,
    const float* __restrict__ alpha, const float* __restrict__ beta, int rows, Workspace ws) {
    extern __shared__ float smem[];
    float* Xs = smem;                 // [64][kLd]  K, scaled K, scaled V
    float* As = Xs + kC * kLd;        // [64][kLd64] A, later P
    float* Ts = As + kC * kLd64;      // [64][kLd64]
    float* Ms = Ts + kC * kLd64;      // [3][16][17] off-diagonal products / warp stages
    float* Ls = Ms + 3 * 16 * 17;     // [64]
    float* Bs = Ls + kC;              // [64]
    const int chunk = blockIdx.x, head = blockIdx.y, kh = head / 3;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int row0 = chunk * kC;
    const std::size_t index = tile_index(chunk, head);
    const float* kc = k + static_cast<std::size_t>(row0) * kQkStride + kh * kD;
    const float* qc = q + static_cast<std::size_t>(row0) * kQkStride + kh * kD;
    for (int i = tid; i < kC * kD / 4; i += 256) {
        const int r = i / (kD / 4), d = (i % (kD / 4)) * 4;
        *reinterpret_cast<float4*>(Xs + r * kLd + d) =
            *reinterpret_cast<const float4*>(kc + static_cast<std::size_t>(r) * kQkStride + d);
    }
    for (int i = tid; i < kC * kC; i += 256) Ts[(i / kC) * kLd64 + i % kC] = 0.0f;
    if (tid < 32) {
        float l0 = 0.0f, l1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
        const int r0 = row0 + 2 * tid, r1 = r0 + 1;
        if (r0 < rows) { l0 = logf(fmaxf(alpha[r0 * kH + head], 1e-30f)); b0 = beta[r0 * kH + head]; }
        if (r1 < rows) { l1 = logf(fmaxf(alpha[r1 * kH + head], 1e-30f)); b1 = beta[r1 * kH + head]; }
        const float pair = l0 + l1;
        float scan = pair;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float n = __shfl_up_sync(0xffffffffu, scan, o);
            if (tid >= o) scan += n;
        }
        const float before = scan - pair;
        Ls[2 * tid] = before + l0;
        Ls[2 * tid + 1] = before + pair;
        Bs[2 * tid] = b0;
        Bs[2 * tid + 1] = b1;
        if (tid == 31) ws.decay[index] = __expf(before + pair);
    }
    __syncthreads();
    for (int t = warp; t < 16; t += 8) {
        const int ti = t / 4, tj = t % 4;
        if (tj > ti) continue;
        FragC acc; wmma::fill_fragment(acc, 0.0f);
#pragma unroll 4
        for (int kk = 0; kk < kD; kk += 8) {
            FragA a; FragBc b;
            wmma::load_matrix_sync(a, Xs + ti * 16 * kLd + kk, kLd);
            wmma::load_matrix_sync(b, Xs + tj * 16 * kLd + kk, kLd);
            to_tf32(a); to_tf32(b);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(As + ti * 16 * kLd64 + tj * 16, acc, kLd64, wmma::mem_row_major);
    }
    __syncthreads();
    for (int i = tid; i < kC * kC; i += 256) {
        const int r = i / kC, c = i % kC;
        As[r * kLd64 + c] = c < r ? Bs[r] * __expf(Ls[r] - Ls[c]) * As[r * kLd64 + c] : 0.0f;
    }
    {
        // Kt_j = exp(L_last - L_j) k_j and Qg_i = gamma_i q_i (pad rows are zero).
        const float last = Ls[kC - 1];
        half* kt = ws.kt + index * kC * kD;
        half* qg = ws.qg + index * kC * kD;
        for (int i = tid; i < kC * kD; i += 256) {
            const int r = i / kD, d = i % kD;
            kt[r * kD + d] = __float2half_rn(Xs[r * kLd + d] * __expf(last - Ls[r]));
            qg[r * kD + d] = __float2half_rn(qc[static_cast<std::size_t>(r) * kQkStride + d] *
                                             __expf(Ls[r]));
        }
    }
    __syncthreads();
    if (tid < kC) {
        const int b = tid / 16, c = tid % 16, base = b * 16;
        for (int i = 0; i < 16; ++i) {
            float value = 0.0f;
            if (i == c) value = 1.0f;
            else if (i > c)
                for (int j = c; j < i; ++j)
                    value -= As[(base + i) * kLd64 + base + j] * Ts[(base + j) * kLd64 + base + c];
            Ts[(base + i) * kLd64 + base + c] = value;
        }
    }
    __syncthreads();
    for (int d = 1; d < 4; ++d) {
        const int blocks = 4 - d;
        for (int e = tid; e < blocks * 256; e += 256) {
            const int bj = e / 256, bi = bj + d, r = (e % 256) / 16, c = e % 16;
            float sum = 0.0f;
            for (int m = bj * 16; m < bi * 16; ++m)
                sum += As[(bi * 16 + r) * kLd64 + m] * Ts[m * kLd64 + bj * 16 + c];
            Ms[(bj * 16 + r) * 17 + c] = sum;
        }
        __syncthreads();
        for (int e = tid; e < blocks * 256; e += 256) {
            const int bj = e / 256, bi = bj + d, r = (e % 256) / 16, c = e % 16;
            float sum = 0.0f;
            for (int t = 0; t <= r; ++t)
                sum += Ts[(bi * 16 + r) * kLd64 + bi * 16 + t] * Ms[(bj * 16 + t) * 17 + c];
            Ts[(bi * 16 + r) * kLd64 + bj * 16 + c] = -sum;
        }
        __syncthreads();
    }
    // P = Q K^T masked and decay-scaled (A is no longer needed): lower 10 tiles.
    for (int t = warp; t < 16; t += 8) {
        const int ti = t / 4, tj = t % 4;
        FragC acc; wmma::fill_fragment(acc, 0.0f);
        if (tj <= ti) {
#pragma unroll 4
            for (int kk = 0; kk < kD; kk += 8) {
                FragA a; FragBc b;
                wmma::load_matrix_sync(a, qc + static_cast<std::size_t>(ti * 16) * kQkStride + kk, kQkStride);
                wmma::load_matrix_sync(b, Xs + tj * 16 * kLd + kk, kLd);
                to_tf32(a); to_tf32(b);
                wmma::mma_sync(acc, a, b, acc);
            }
        }
        wmma::store_matrix_sync(As + ti * 16 * kLd64 + tj * 16, acc, kLd64, wmma::mem_row_major);
    }
    __syncthreads();
    {
        half* p = ws.p + index * kC * kC;
        for (int i = tid; i < kC * kC; i += 256) {
            const int r = i / kC, c = i % kC;
            p[i] = __float2half_rn(c <= r ? As[r * kLd64 + c] * __expf(Ls[r] - Ls[c]) : 0.0f);
        }
    }
    for (int i = tid; i < kC * kD; i += 256) {
        const int r = i / kD, d = i % kD;
        Xs[r * kLd + d] *= Bs[r] * __expf(Ls[r]);
    }
    __syncthreads();
    float* stage = Ms;   // 8 warps x 256 floats would overflow Ms; use As (free now)
    stage = As + warp * 256;
    t_times(Ts, Xs, stage, ws.w + index * kC * kD, warp, lane);
    __syncthreads();
    const auto* vv = reinterpret_cast<const __nv_bfloat16*>(v);
    for (int i = tid; i < kC * kD / 2; i += 256) {
        const int r = i / (kD / 2), d = (i % (kD / 2)) * 2, row = row0 + r;
        float2 value = make_float2(0.f, 0.f);
        if (row < rows)
            value = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                vv + (static_cast<std::size_t>(row) * kH + head) * kD + d));
        Xs[r * kLd + d] = value.x * Bs[r];
        Xs[r * kLd + d + 1] = value.y * Bs[r];
    }
    __syncthreads();
    t_times(Ts, Xs, stage, ws.u + index * kC * kD, warp, lane);
}

// ---- Kernel 2: sequential chunk scan with fused outputs, per (head, 32-wide value tile) ----
constexpr std::size_t kScanSmem = sizeof(float) * (kD * kLdV + 8 * 256) +
                                  sizeof(half) * (kD * kLdH + kC * kLdH);

__global__ void __launch_bounds__(256) scan_kernel(float* __restrict__ state,
    std::uint16_t* __restrict__ output, int rows, int chunks, Workspace ws) {
    extern __shared__ float smem[];
    float* Ss = smem;                                   // [128][kLdV] FP32 state tile
    float* stages = Ss + kD * kLdV;                     // per warp [256]
    half* Sh = reinterpret_cast<half*>(stages + 8 * 256);  // [128][kLdH] FP16 operand copy
    half* Dh = Sh + kD * kLdH;                          // [64][kLdH] Delta tile
    const int head = blockIdx.x, tile = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int v0 = tile * kVT;
    float* stage = stages + warp * 256;
    float* s_global = state + static_cast<std::size_t>(head) * kD * kD;
    auto* out = reinterpret_cast<__nv_bfloat16*>(output);
    for (int i = tid; i < kD * kVT; i += 256) {
        const int r = i / kVT, c = i % kVT;
        const float value = s_global[r * kD + v0 + c];
        Ss[r * kLdV + c] = value;
        Sh[r * kLdH + c] = __float2half_rn(value);
    }
    __syncthreads();
    const int ti = warp / 2, tj = warp % 2;   // this warp's 16x16 tile of the 64x32 chunk block
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const std::size_t index = tile_index(chunk, head);
        const half* w = ws.w + index * kC * kD;
        const half* u = ws.u + index * kC * kD;
        // Delta tile = u - w S0.
        {
            HC acc; wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int kk = 0; kk < kD; kk += 16) {
                HA a; HB b;
                wmma::load_matrix_sync(a, w + ti * 16 * kD + kk, kD);
                wmma::load_matrix_sync(b, Sh + kk * kLdH + tj * 16, kLdH);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(stage, acc, 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = ti * 16 + e / 16, c = tj * 16 + e % 16;
                Dh[r * kLdH + c] = __float2half_rn(__half2float(u[r * kD + v0 + c]) - stage[e]);
            }
        }
        __syncthreads();
        // Output tile = Qg S0 + P Delta (reads the pre-update S copy).
        {
            const half* qg = ws.qg + index * kC * kD;
            const half* p = ws.p + index * kC * kC;
            HC acc; wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int kk = 0; kk < kD; kk += 16) {
                HA a; HB b;
                wmma::load_matrix_sync(a, qg + ti * 16 * kD + kk, kD);
                wmma::load_matrix_sync(b, Sh + kk * kLdH + tj * 16, kLdH);
                wmma::mma_sync(acc, a, b, acc);
            }
            for (int kk = 0; kk <= ti * 16; kk += 16) {
                HA a; HB b;
                wmma::load_matrix_sync(a, p + ti * 16 * kC + kk, kC);
                wmma::load_matrix_sync(b, Dh + kk * kLdH + tj * 16, kLdH);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(stage, acc, 16, wmma::mem_row_major);
            __syncwarp();
            const int row0 = chunk * kC + ti * 16;
            for (int e = lane; e < 256; e += 32) {
                const int row = row0 + e / 16;
                if (row < rows)
                    out[(static_cast<std::size_t>(row) * kH + head) * kD + v0 + tj * 16 + e % 16] =
                        __float2bfloat16_rn(stage[e] * kOutputScale);
            }
            __syncwarp();
        }
        // S = gamma_C S0 + Kt^T Delta (128 x 32): 16 tiles, two per warp.
        {
            const half* kt = ws.kt + index * kC * kD;
            const float decay = ws.decay[index];
            for (int t = warp; t < 16; t += 8) {
                const int si = t / 2, sj = t % 2;
                HC acc;
                wmma::load_matrix_sync(acc, Ss + si * 16 * kLdV + sj * 16, kLdV, wmma::mem_row_major);
#pragma unroll
                for (int e = 0; e < acc.num_elements; ++e) acc.x[e] *= decay;
#pragma unroll
                for (int kk = 0; kk < kC; kk += 16) {
                    HAc a; HB b;
                    wmma::load_matrix_sync(a, kt + kk * kD + si * 16, kD);
                    wmma::load_matrix_sync(b, Dh + kk * kLdH + sj * 16, kLdH);
                    wmma::mma_sync(acc, a, b, acc);
                }
                wmma::store_matrix_sync(Ss + si * 16 * kLdV + sj * 16, acc, kLdV, wmma::mem_row_major);
            }
        }
        __syncthreads();
        for (int i = tid; i < kD * kVT; i += 256) {
            const int r = i / kVT, c = i % kVT;
            Sh[r * kLdH + c] = __float2half_rn(Ss[r * kLdV + c]);
        }
        __syncthreads();
    }
    for (int i = tid; i < kD * kVT; i += 256) {
        const int r = i / kVT, c = i % kVT;
        s_global[r * kD + v0 + c] = Ss[r * kLdV + c];
    }
}

inline void configure() {
    cudaFuncSetAttribute(prep_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(kPrepSmem));
    cudaFuncSetAttribute(scan_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(kScanSmem));
}

// q and k are written in their pad rows [rows, round_up(rows, 64)) (zeros).
inline void launch(float* q, float* k, const std::uint16_t* v, const float* alpha,
                   const float* beta, float* state, std::uint16_t* output, int rows,
                   Workspace ws, cudaStream_t stream) {
    const int chunks = (rows + kC - 1) / kC;
    const int pad = chunks * kC - rows;
    if (pad) {
        cudaMemsetAsync(q + static_cast<std::size_t>(rows) * kQkStride, 0,
                        static_cast<std::size_t>(pad) * kQkStride * sizeof(float), stream);
        cudaMemsetAsync(k + static_cast<std::size_t>(rows) * kQkStride, 0,
                        static_cast<std::size_t>(pad) * kQkStride * sizeof(float), stream);
    }
    prep_kernel<<<dim3(chunks, kH), 256, kPrepSmem, stream>>>(q, k, v, alpha, beta, rows, ws);
    scan_kernel<<<dim3(kH, kD / kVT), 256, kScanSmem, stream>>>(state, output, rows, chunks, ws);
}

}  // namespace gdn_chunked
