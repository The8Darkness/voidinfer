#pragma once
// L0 OSCAR KV history (FutureMLS-Lab OSCAR INT2, arXiv 2605.17757) for the
// ordinary FP16-KV verifier/decode route. Committed rows that leave the
// exact window (sink [0, 64) and the 256 most recent committed rows) are
// encoded per KV head as
//   K: x = (k - mu_h) R_k,h   V: x = v R_v,h          (row vector, orthogonal R)
//   clip |x| at the 0.96 (K) / 0.92 (V) per-row quantile, two 128-dim groups,
//   asymmetric INT2: scale = (max - min) / 3, zero = -min / scale,
//   code = clamp(floor(x / scale + zero + 0.5), 0, 3), x^ = (code - zero) scale.
// R and mu are per-KV-head OSCAR-2 'center' rotations fitted offline
// (NINFER_EXL3_L0_OSCAR_ROT: k_rot_perhead.bin, v_rot_perhead.bin [16][4][256][256],
// k_mean.bin [16][4][256], fp32). Attention over history uses the rotated query
// q R_k with the constant q . mu added to every history score, accumulates
// values in the rotated basis and un-rotates the merged history numerator once
// (o = o' R_v^T) before the exact-segment merge.
#include "exl3/l0_oscar_storage.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3::l0_oscar {

constexpr int kSink = 64;
constexpr int kRecent = 256;
constexpr int kAlign = 64;
constexpr int kDim = 256;
constexpr int kKVHeadsL0 = 4;
constexpr int kCodeBytes = kDim / 4;     // 2-bit codes, dims 4b..4b+3 in byte b
constexpr int kMetaFloats = 4;           // scale0, zero0, scale1, zero1
constexpr int kRingRows = 4096;          // > sink-free exact window + one prefill chunk

// Device row of absolute key `key` in a window-ring addressed K or V plane.
// With sink == nullptr and ring_mask == -1 this is the plain linear cache.
__host__ __device__ __forceinline__ std::size_t window_row(int key, bool sink, int ring_mask) {
    return static_cast<std::size_t>(sink && key < kSink ? key : (key & ring_mask));
}

// Writes appended rows into the window ring (and the sink rows).
__global__ void window_append_kernel(const std::uint16_t* k, const std::uint16_t* v,
    std::uint16_t* k_ring, std::uint16_t* v_ring, std::uint16_t* k_sink, std::uint16_t* v_sink,
    int rows, int position, const int* position_device) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * kKVHeadsL0 * kDim) return;
    const int row = index / (kKVHeadsL0 * kDim), offset = index % (kKVHeadsL0 * kDim);
    const int slot = (position_device ? *position_device : position) + row;
    const std::size_t ring = static_cast<std::size_t>(slot & (kRingRows - 1)) * kKVHeadsL0 * kDim + offset;
    k_ring[ring] = k[index];
    v_ring[ring] = v[index];
    if (slot < kSink) {
        const std::size_t sink = static_cast<std::size_t>(slot) * kKVHeadsL0 * kDim + offset;
        k_sink[sink] = k[index];
        v_sink[sink] = v[index];
    }
}

struct Assets {
    const float* rk = nullptr;   // [16][4][256][256]   x' = x R
    const float* rv = nullptr;   // [16][4][256][256]
    const float* rvt = nullptr;  // [16][4][256][256]  R_v transposed (o = o' R_v^T, coalesced)
    const float* mu = nullptr;   // [16][4][256]
};

inline bool enabled() {
    static const bool value = [] {
        const char* v = std::getenv("NINFER_EXL3_L0_OSCAR");
        if (!v || !*v || std::strcmp(v, "0") == 0) return false;
        if (std::strcmp(v, "1") != 0) throw std::invalid_argument("NINFER_EXL3_L0_OSCAR must be 0 or 1");
        return true;
    }();
    return value;
}

inline const Assets& assets() {
    static const Assets value = [] {
        const char* dir = std::getenv("NINFER_EXL3_L0_OSCAR_ROT");
        if (!dir || !*dir) throw std::invalid_argument("L0 OSCAR requires NINFER_EXL3_L0_OSCAR_ROT");
        constexpr std::size_t bank = 16ull * 4 * kDim * kDim;
        std::vector<float> rk(bank), rv(bank), rvt(bank), mu(16ull * 4 * kDim);
        const auto read = [&](const char* name, std::vector<float>& out) {
            std::ifstream f(std::string(dir) + "/" + name, std::ios::binary);
            f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * 4));
            if (!f) throw std::runtime_error(std::string("L0 OSCAR asset read: ") + name);
        };
        read("k_rot_perhead.bin", rk);
        read("v_rot_perhead.bin", rv);
        read("k_mean.bin", mu);
        for (std::size_t m = 0; m < 64; ++m)
            for (int i = 0; i < kDim; ++i)
                for (int j = 0; j < kDim; ++j)
                    rvt[m * kDim * kDim + static_cast<std::size_t>(j) * kDim + i] =
                        rv[m * kDim * kDim + static_cast<std::size_t>(i) * kDim + j];
        float* device = nullptr;
        const std::size_t total = 3 * bank + mu.size();
        if (cudaMalloc(&device, total * 4) != cudaSuccess) throw std::runtime_error("L0 OSCAR asset allocation");
        cudaMemcpy(device, rk.data(), bank * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(device + bank, rv.data(), bank * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(device + 2 * bank, rvt.data(), bank * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(device + 3 * bank, mu.data(), mu.size() * 4, cudaMemcpyHostToDevice);
        Assets a;
        a.rk = device; a.rv = device + bank; a.rvt = device + 2 * bank; a.mu = device + 3 * bank;
        return a;
    }();
    return value;
}

// History extent [kSink, history_end(base)): committed rows at least kRecent
// behind the committed base, aligned down to kAlign.
__host__ __device__ __forceinline__ int history_end(int base) {
    const int end = ((base - kRecent) / kAlign) * kAlign;
    return end > kSink ? end : kSink;
}

__device__ __forceinline__ float l0_block_max(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = scratch[0];
    for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w) r = fmaxf(r, scratch[w]);
    return r;
}

// One CTA (256 threads = dims) per (row, head, K|V) task over newly aged rows.
__global__ void __launch_bounds__(256) encode_kernel(const std::uint16_t* k_cache,
    const std::uint16_t* v_cache, int capacity, const int* position_device, int position,
    int* state, Assets a, int bank, std::uint8_t* k_codes, std::uint8_t* v_codes,
    float* k_meta, float* v_meta) {
    __shared__ float x[kDim], y[kDim], scratch[16];
    const int base = position_device ? *position_device : position;
    int wm = state[0];
    if (base < state[1]) wm = kSink;  // a new request restarted the context
    wm = max(wm, kSink);
    const int end = min(history_end(base), capacity);
    const int tasks = max(end - wm, 0) * 8;
    const int j = threadIdx.x;
    for (int t = blockIdx.x; t < tasks; t += gridDim.x) {
        const int row = wm + t / 8, head = (t / 2) % 4, kind = t % 2;
        const std::size_t cell = static_cast<std::size_t>(row) * kKVHeadsL0 + head;
        const auto* src = (kind ? v_cache : k_cache) + cell * kDim;
        const float* R = (kind ? a.rv : a.rk) + (static_cast<std::size_t>(bank) * 4 + head) * kDim * kDim;
        __syncthreads();
        float value = __half2float(__ushort_as_half(src[j]));
        if (!kind) value -= a.mu[(bank * 4 + head) * kDim + j];
        x[j] = value;
        __syncthreads();
        float acc = 0.0f;
        for (int i = 0; i < kDim; ++i) acc = fmaf(x[i], R[i * kDim + j], acc);
        y[j] = acc;
        __syncthreads();
        const float mag = fabsf(acc);
        int rank = 0;
        for (int i = 0; i < kDim; ++i) {
            const float b = fabsf(y[i]);
            rank += (b < mag) || (b == mag && i < j);
        }
        const int clip_index = kind ? 235 : 245;  // floor(0.92*256), floor(0.96*256)
        if (rank == clip_index) scratch[15] = mag;
        __syncthreads();
        const float thr = scratch[15];
        const float clipped = fminf(fmaxf(acc, -thr), thr);
        float mn = clipped, mx = clipped;
        for (int o = 16; o > 0; o >>= 1) {
            mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        }
        __syncthreads();
        if ((j & 31) == 0) { x[j >> 5] = mn; x[8 + (j >> 5)] = mx; }
        __syncthreads();
        const int g = j >> 7;
        float gmn = x[g * 4], gmx = x[8 + g * 4];
        for (int w = 1; w < 4; ++w) { gmn = fminf(gmn, x[g * 4 + w]); gmx = fmaxf(gmx, x[8 + g * 4 + w]); }
        const float scale = fmaxf(gmx - gmn, 1.0e-8f) / 3.0f;
        const float zero = -gmn / scale;
        const int code = static_cast<int>(fminf(fmaxf(floorf(clipped / scale + zero + 0.5f), 0.0f), 3.0f));
        // pack 4 consecutive dims per byte
        const unsigned packed = __shfl_sync(0xffffffffu, code, (j & 31) & ~3) |
                                (__shfl_sync(0xffffffffu, code, ((j & 31) & ~3) + 1) << 2) |
                                (__shfl_sync(0xffffffffu, code, ((j & 31) & ~3) + 2) << 4) |
                                (__shfl_sync(0xffffffffu, code, ((j & 31) & ~3) + 3) << 6);
        auto* codes = (kind ? v_codes : k_codes) + cell * kCodeBytes;
        if ((j & 3) == 0) codes[j >> 2] = static_cast<std::uint8_t>(packed);
        if ((j & 127) == 0) {
            float* meta = (kind ? v_meta : k_meta) + cell * kMetaFloats;
            meta[2 * g] = scale;
            meta[2 * g + 1] = zero;
        }
    }
}

__global__ void advance_kernel(const int* position_device, int position, int* state, int capacity) {
    const int base = position_device ? *position_device : position;
    int wm = state[0];
    if (base < state[1]) wm = kSink;
    wm = max(wm, kSink);
    state[0] = max(wm, min(history_end(base), capacity));
    state[1] = base;
}

// q_rot[row][qh] = q R_k(kv head), q_mu[row][qh] = q . mu. One CTA per (row,
// KV head, 32-column block): 192 threads = 6 query heads x 32 columns, R_k
// columns read once per CTA; column block 0 also forms q . mu.
__global__ void __launch_bounds__(192) query_kernel(const std::uint16_t* q, int rows, Assets a,
    int bank, __half* q_rot, float* q_mu) {
    __shared__ float x[6][kDim];
    const int row = blockIdx.x / 32, kv = (blockIdx.x / 8) % 4, block = blockIdx.x % 8;
    const int t = threadIdx.x, h = t / 32, c = t % 32, j = block * 32 + c;
    if (row >= rows) return;
    const std::size_t at = (static_cast<std::size_t>(row) * 24 + kv * 6) * kDim;
    for (int i = t; i < 6 * kDim; i += blockDim.x)
        x[i / kDim][i % kDim] = __half2float(__ushort_as_half(q[at + i]));
    __syncthreads();
    const float* R = a.rk + (static_cast<std::size_t>(bank) * 4 + kv) * kDim * kDim;
    float acc = 0.0f;
    for (int i = 0; i < kDim; ++i) acc = fmaf(x[h][i], R[i * kDim + j], acc);
    q_rot[at + h * kDim + j] = __float2half_rn(acc);
    if (block == 0) {
        const float* mu = a.mu + (bank * 4 + kv) * kDim;
        float m = 0.0f;
        for (int i = c; i < kDim; i += 32) m = fmaf(x[h][i], mu[i], m);
        for (int o = 16; o > 0; o >>= 1) m += __shfl_xor_sync(0xffffffffu, m, o);
        if (c == 0) q_mu[row * 24 + kv * 6 + h] = m;
    }
}

// Keys per verifier history segment for the live history [kSink, end):
// `segments` near-equal spans, whole 32-key tiles. Shared by the history
// kernel and its merge so both derive identical segment bounds on device.
__host__ __device__ __forceinline__ int history_span(int end, int segments) {
    const int keys = end > kSink ? end - kSink : 0;
    const int per = (keys + segments - 1) / segments;
    return per < 32 ? 32 : ((per + 31) / 32) * 32;
}
// One verifier history CTA per (segment, KV head): 42 x 4 CTAs fill the 170
// SMs of the target GPU once.
constexpr int kHistorySegments = 42;

// History segment length for a context capacity: at most 512 segments.
inline int history_segment_keys(int capacity) {
    int keys = 64;
    while (keys < 512 && (capacity + keys - 1) / keys > 512) keys *= 2;
    return keys;
}

}  // namespace ninfer::exl3::l0_oscar
