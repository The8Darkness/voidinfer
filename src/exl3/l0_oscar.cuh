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
// (<model directory>/l0_oscar or NINFER_EXL3_L0_OSCAR_ROT: k_rot_perhead.bin,
// v_rot_perhead.bin [16][4][256][256], k_mean.bin [16][4][256], fp32). Attention
// over history uses the rotated query q R_k with the constant q . mu added to
// every history score, accumulates values in the rotated basis and un-rotates
// the merged history numerator once (o = o' R_v^T) before the exact-segment
// merge. The history rows the verifier attends to most are served exactly
// from FP16 copies of their L2 rows instead (hot rows, l0_hot.cuh).
#include "exl3/attention_profile.h"
#include "exl3/l0_oscar_storage.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3::l0_oscar {

constexpr int kSink = 64;
// Exact recent window (exl3_l0_recent_rows): device code reads the copy
// installed at L0 storage setup (set_recent_rows) in this translation unit.
static __constant__ int recent_rows_device = 256;
inline void set_recent_rows() {
    static const bool installed = [] {
        const int rows = exl3_l0_recent_rows();
        if (cudaMemcpyToSymbol(recent_rows_device, &rows, sizeof(rows)) != cudaSuccess)
            throw std::runtime_error("L0 OSCAR recent window installation");
        return true;
    }();
    (void)installed;
}
__host__ __device__ __forceinline__ int recent_rows() {
#ifdef __CUDA_ARCH__
    return recent_rows_device;
#else
    return exl3_l0_recent_rows();
#endif
}
constexpr int kAlign = 64;
constexpr int kDim = 256;
constexpr int kKVHeadsL0 = 4;
constexpr int kCodeBytes = kDim / 4;     // 2-bit codes, dims 4b..4b+3 in byte b
constexpr int kMetaFloats = 4;           // scale0, zero0, scale1, zero1 (FP16)
constexpr int kRingRows = 2048;          // > sink-free exact window (<= 320) + one 1024-row prefill chunk

// Device row of absolute key `key` in a window-ring addressed K or V plane.
// With sink == nullptr and ring_rows == -1 this is the plain linear cache.
// The ring holds ring_rows (any multiple of 64) consecutive keys.
__host__ __device__ __forceinline__ std::size_t window_row(int key, bool sink, int ring_rows) {
    return static_cast<std::size_t>(sink && key < kSink ? key :
        (ring_rows < 0 ? key : key % ring_rows));
}

// Writes appended rows into the window ring (and the sink rows).
__global__ void window_append_kernel(const std::uint16_t* k, const std::uint16_t* v,
    std::uint16_t* k_ring, std::uint16_t* v_ring, std::uint16_t* k_sink, std::uint16_t* v_sink,
    int rows, int position, const int* position_device, int ring_rows = kRingRows) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * kKVHeadsL0 * kDim) return;
    const int row = index / (kKVHeadsL0 * kDim), offset = index % (kKVHeadsL0 * kDim);
    const int slot = (position_device ? *position_device : position) + row;
    const std::size_t ring = static_cast<std::size_t>(slot % ring_rows) * kKVHeadsL0 * kDim + offset;
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
    const __half* rk16 = nullptr;   // FP16 copies for the tensor-core rotations
    const __half* rvt16 = nullptr;
};

inline bool enabled() { return exl3_l0_oscar_enabled(); }

inline const Assets& assets() {
    static const Assets value = [] {
        const char* env = std::getenv("NINFER_EXL3_L0_OSCAR_ROT");
        const std::string dir = env && *env ? std::string(env) : exl3_l0_oscar_model_assets();
        if (dir.empty())
            throw std::invalid_argument("L0 OSCAR needs rotation assets: <model directory>/l0_oscar or "
                                        "NINFER_EXL3_L0_OSCAR_ROT (NINFER_EXL3_L0_OSCAR=0 selects FP16 device KV)");
        constexpr std::size_t bank = 16ull * 4 * kDim * kDim;
        std::vector<float> rk(bank), rv(bank), rvt(bank), mu(16ull * 4 * kDim);
        const auto read = [&](const char* name, std::vector<float>& out) {
            std::ifstream f(dir + "/" + name, std::ios::binary);
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
        std::vector<__half> half_copy(2 * bank);
        for (std::size_t i = 0; i < bank; ++i) {
            half_copy[i] = __float2half_rn(rk[i]);
            half_copy[bank + i] = __float2half_rn(rvt[i]);
        }
        __half* device16 = nullptr;
        if (cudaMalloc(&device16, half_copy.size() * sizeof(__half)) != cudaSuccess)
            throw std::runtime_error("L0 OSCAR asset allocation");
        cudaMemcpy(device16, half_copy.data(), half_copy.size() * sizeof(__half), cudaMemcpyHostToDevice);
        Assets a;
        a.rk = device; a.rv = device + bank; a.rvt = device + 2 * bank; a.mu = device + 3 * bank;
        a.rk16 = device16; a.rvt16 = device16 + bank;
        return a;
    }();
    return value;
}

// History extent [kSink, history_end(base)): committed rows at least recent_rows()
// behind the committed base, aligned down to kAlign.
__host__ __device__ __forceinline__ int history_end(int base) {
    const int end = ((base - recent_rows()) / kAlign) * kAlign;
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

// Exact hot rows (l0_hot.cuh). The verifier history kernel skips rows marked
// in `bits` and emits rows whose INT2 probability (relative to the previous
// round's history LSE, `ref`) exceeds tau; an exact FP16 pass covers the hot
// slots, and each round replaces the stalest slots with the best candidates,
// filled from the FP16 L2 planes. NINFER_EXL3_L0_HOT = slots per KV head and
// layer (default 512, a multiple of 32; 0 = off).
constexpr int kHotCandidates = 1024;   // per KV head and round
constexpr int kHotMaxInsert = 64;      // per KV head and round
struct HotConfig {
    int slots = 512;                // per KV head and layer
    float log2_tau = -8.965784f;    // candidates: p > 2e-3 of the history mass
    int insert = 32;                // slots replaced per KV head and round at most
    float lambda = 0.01f;           // priority decay per committed token (ln units)
};
inline const HotConfig& hot_config() {
    static const HotConfig value = [] {
        HotConfig c;

        if (const char* v = std::getenv("NINFER_EXL3_L0_HOT")) {
            char* end = nullptr;
            const long slots = std::strtol(v, &end, 10);
            if (!*v || *end || slots < 0 || slots > 4096 || slots % 32)
                throw std::invalid_argument("NINFER_EXL3_L0_HOT must be 0..4096 slots, a multiple of 32");
            c.slots = static_cast<int>(slots);
        }
        return c;
    }();
    return value;
}
struct HotView {
    std::uint16_t* k = nullptr;
    std::uint16_t* v = nullptr;
    int* row = nullptr;
    float* prio = nullptr;
    unsigned* bits = nullptr;
    int* count = nullptr;
    int* cand_row = nullptr;
    float* cand_val = nullptr;
    float* ref = nullptr;
    int* won = nullptr;
    int slots = 0, words = 0;
};
inline HotView hot_view(const LayerStorage& l) {
    return {l.hot_k, l.hot_v, l.hot_row, l.hot_prio, l.hot_bits, l.hot_count, l.hot_cand_row,
            l.hot_cand_val, l.hot_ref, l.hot_won, l.hot_slots, l.hot_words};
}
// HotView::won: [4] slots won last round, [4][kHotMaxInsert] slots, [4][kHotMaxInsert] rows.
constexpr int kHotWonSlots = 4;
constexpr int kHotWonRows = kHotWonSlots + 4 * kHotMaxInsert;
// Forgets every hot row (a new request or a restored history).
__device__ __forceinline__ void hot_clear(const HotView& h, int index, int stride) {
    for (int i = index; i < 4 * h.words; i += stride) h.bits[i] = 0u;
    for (int i = index; i < 4 * h.slots; i += stride) h.row[i] = -1;
    for (int i = index; i < 24; i += stride) h.ref[i] = 3.0e38f;
    for (int i = index; i < 4; i += stride) { h.count[i] = 0; h.won[i] = 0; }
}
// Fills the hot slots won last round from the FP16 planes (16-byte mapped
// reads); runs on a forked branch beside the encode of the same append.
__global__ void hot_fill_kernel(HotView hot, const std::uint16_t* k_cache, const std::uint16_t* v_cache) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= 4 * kHotMaxInsert * 2 * (kDim / 8)) return;
    const int kv = i / (kHotMaxInsert * 2 * (kDim / 8)), rest = i % (kHotMaxInsert * 2 * (kDim / 8));
    const int p = rest / (2 * (kDim / 8)), kind = (rest / (kDim / 8)) & 1, c = rest % (kDim / 8);
    if (p >= hot.won[kv]) return;
    const int slot = hot.won[kHotWonSlots + kv * kHotMaxInsert + p];
    const int row = hot.won[kHotWonRows + kv * kHotMaxInsert + p];
    reinterpret_cast<uint4*>((kind ? hot.v : hot.k) + (static_cast<std::size_t>(kv) * hot.slots + slot) * kDim)[c] =
        reinterpret_cast<const uint4*>((kind ? v_cache : k_cache) + (static_cast<std::size_t>(row) * kKVHeadsL0 + kv) * kDim)[c];
}
__global__ void hot_clear_kernel(HotView h) {
    hot_clear(h, static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x), static_cast<int>(gridDim.x * blockDim.x));
}

// One CTA (256 threads = dims) per (16-row batch, head, K|V) task over newly
// aged rows: each rotation row R[i][:] is read once per batch instead of once
// per row (the per-row 256 KB matrix re-reads were L2-bound). Every row keeps
// its own sequential fmaf chain, so codes and metadata are unchanged.
constexpr int kEncodeRows = 16;
constexpr int kEncodeBlocks = 512;
// y[r][j] = sum_i x[r][i] R[i][j] for rows 0..Rows-1 (sequential fmaf chain).
template <int Rows>
__device__ __forceinline__ void encode_rotate(const float (*x)[kDim], const float* R, int j,
                                              float (*y)[kDim]) {
    float acc[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) acc[r] = 0.0f;
    for (int i = 0; i < kDim; i += 4) {
        const float r0 = R[i * kDim + j], r1 = R[(i + 1) * kDim + j];
        const float r2 = R[(i + 2) * kDim + j], r3 = R[(i + 3) * kDim + j];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const float4 v = *reinterpret_cast<const float4*>(&x[r][i]);
            acc[r] = fmaf(v.w, r3, fmaf(v.z, r2, fmaf(v.y, r1, fmaf(v.x, r0, acc[r]))));
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) y[r][j] = acc[r];
}
__global__ void __launch_bounds__(256) encode_kernel(const std::uint16_t* k_cache,
    const std::uint16_t* v_cache, int capacity, const int* position_device, int position,
    int* state, Assets a, int bank, std::uint8_t* k_codes, std::uint8_t* v_codes,
    __half* k_meta, __half* v_meta, HotView hot) {
    __shared__ __align__(16) float x[kEncodeRows][kDim];
    __shared__ __align__(16) float y[kEncodeRows][kDim];
    __shared__ float extrema[kEncodeRows][16], thresholds[kEncodeRows];
    const int base = position_device ? *position_device : position;
    int wm = state[0];
    const int thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int threads = static_cast<int>(gridDim.x * blockDim.x);
    if (base < state[1]) {  // a new request restarted the context
        wm = kSink;
        if (hot.slots) hot_clear(hot, thread, threads);
    }
    wm = max(wm, kSink);
    const int end = min(history_end(base), capacity);
    const int batches = (max(end - wm, 0) + kEncodeRows - 1) / kEncodeRows;
    const int tasks = batches * 8;
    const int j = threadIdx.x;
    for (int t = blockIdx.x; t < tasks; t += gridDim.x) {
        const int row0 = wm + (t / 8) * kEncodeRows, head = (t / 2) % 4, kind = t % 2;
        const int count = min(kEncodeRows, end - row0);
        const float* R = (kind ? a.rv : a.rk) + (static_cast<std::size_t>(bank) * 4 + head) * kDim * kDim;
        const float mu = kind ? 0.0f : a.mu[(bank * 4 + head) * kDim + j];
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kEncodeRows; ++r) {
            float value = 0.0f;
            if (r < count) {
                const std::size_t cell = static_cast<std::size_t>(row0 + r) * kKVHeadsL0 + head;
                value = __half2float(__ushort_as_half((kind ? v_cache : k_cache)[cell * kDim + j]));
                if (!kind) value -= mu;
            }
            x[r][j] = value;
        }
        __syncthreads();
        // Decode rounds age only a few rows: size the row loop to the batch.
        if (count <= 2) encode_rotate<2>(x, R, j, y);
        else if (count <= 4) encode_rotate<4>(x, R, j, y);
        else if (count <= 8) encode_rotate<8>(x, R, j, y);
        else encode_rotate<kEncodeRows>(x, R, j, y);
        __syncthreads();
        // Clip threshold: the clip_index-th smallest |y| of the row (ascending,
        // ties by dimension), found by one warp per row with a radix select
        // over the bit patterns (non-negative floats order as unsigned).
        const int clip_index = kind ? 235 : 245;  // floor(0.92*256), floor(0.96*256)
        for (int r = j >> 5; r < count; r += 8) {
            const int lane = j & 31;
            const float4 a0 = *reinterpret_cast<const float4*>(&y[r][lane * 8]);
            const float4 a1 = *reinterpret_cast<const float4*>(&y[r][lane * 8 + 4]);
            const unsigned bits[8] = {
                __float_as_uint(fabsf(a0.x)), __float_as_uint(fabsf(a0.y)),
                __float_as_uint(fabsf(a0.z)), __float_as_uint(fabsf(a0.w)),
                __float_as_uint(fabsf(a1.x)), __float_as_uint(fabsf(a1.y)),
                __float_as_uint(fabsf(a1.z)), __float_as_uint(fabsf(a1.w))};
            unsigned prefix = 0u, mask = 0u;
            int rank = clip_index;
            for (int bit = 31; bit >= 0; --bit) {
                const unsigned b = 1u << bit;
                int local = 0;
#pragma unroll
                for (int e = 0; e < 8; ++e) local += ((bits[e] & mask) == prefix) && !(bits[e] & b);
                const int zeros = static_cast<int>(__reduce_add_sync(0xffffffffu, static_cast<unsigned>(local)));
                if (rank >= zeros) { rank -= zeros; prefix |= b; }
                mask |= b;
            }
            if (lane == 0) thresholds[r] = __uint_as_float(prefix);
        }
        __syncthreads();
        for (int r = 0; r < count; ++r) {
            const float thr = thresholds[r];
            const float clipped = fminf(fmaxf(y[r][j], -thr), thr);
            float mn = clipped, mx = clipped;
            for (int o = 16; o > 0; o >>= 1) {
                mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            }
            if ((j & 31) == 0) { extrema[r][j >> 5] = mn; extrema[r][8 + (j >> 5)] = mx; }
        }
        __syncthreads();
        const int g = j >> 7;
        for (int r = 0; r < count; ++r) {
            const float thr = thresholds[r];
            const float clipped = fminf(fmaxf(y[r][j], -thr), thr);
            float gmn = extrema[r][g * 4], gmx = extrema[r][8 + g * 4];
            for (int w = 1; w < 4; ++w) { gmn = fminf(gmn, extrema[r][g * 4 + w]); gmx = fmaxf(gmx, extrema[r][8 + g * 4 + w]); }
            // Scale and zero are stored as FP16; the codes use the stored values.
            const float scale = __half2float(__float2half_rn(fmaxf(gmx - gmn, 1.0e-6f) / 3.0f));
            const float zero = __half2float(__float2half_rn(-gmn / scale));
            const int code = static_cast<int>(fminf(fmaxf(floorf(clipped / scale + zero + 0.5f), 0.0f), 3.0f));
            // pack 4 consecutive dims per byte
            const unsigned packed = __shfl_sync(0xffffffffu, code, (j & 31) & ~3) |
                                    (__shfl_sync(0xffffffffu, code, ((j & 31) & ~3) + 1) << 2) |
                                    (__shfl_sync(0xffffffffu, code, ((j & 31) & ~3) + 2) << 4) |
                                    (__shfl_sync(0xffffffffu, code, ((j & 31) & ~3) + 3) << 6);
            const std::size_t cell = static_cast<std::size_t>(row0 + r) * kKVHeadsL0 + head;
            auto* codes = (kind ? v_codes : k_codes) + cell * kCodeBytes;
            if ((j & 3) == 0) codes[j >> 2] = static_cast<std::uint8_t>(packed);
            if ((j & 127) == 0) {
                __half* meta = (kind ? v_meta : k_meta) + cell * kMetaFloats;
                meta[2 * g] = __float2half_rn(scale);
                meta[2 * g + 1] = __float2half_rn(zero);
            }
        }
    }
}

// Rewind to a verified root at `position`: rows from history_end(position) on
// are re-encoded from the (rewritten) L2 planes, and hot rows among them are
// dropped together with pending candidates.
__global__ void rewind_kernel(int* state, HotView hot, int position) {
    const int end = history_end(position);
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index == 0) {
        state[0] = min(state[0], end);
        state[1] = position;
    }
    if (!hot.slots) return;
    if (index < 4) { hot.count[index] = 0; hot.won[index] = 0; }
    for (int i = index; i < 4 * hot.slots; i += static_cast<int>(gridDim.x * blockDim.x)) {
        const int row = hot.row[i];
        if (row < end) continue;
        const int kv = i / hot.slots;
        atomicAnd(hot.bits + kv * hot.words + (row >> 5), ~(1u << (row & 31)));
        hot.row[i] = -1;
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
