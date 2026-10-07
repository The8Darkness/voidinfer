#pragma once
// Research instrument for the KV-tier (VeriCache) fidelity study. With
// NINFER_EXL3_KV_FAKEQUANT=int2|turbo|fp8 the ordinary FP16 K/V cache rows
// that leave the exact window (positions >= prefix and older than `recent`
// rows before the committed base) are replaced in place by their codec
// reconstruction: OSCAR INT2 g128 in the calibrated rotation basis
// (NINFER_EXL3_KV_FAKEQUANT_ROT), TurboAngle K7+8/V6+4, or FP8 E4M3 row
// scale. Attention then runs unchanged over the represented values. Not a
// production storage format.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::exl3::kv_fakequant {

struct Config {
    int mode = 0;  // 0 off, 1 OSCAR INT2, 2 TurboAngle, 3 FP8, 4 OSCAR INT4, 5 OSCAR INT3
    float levels = 3.0f;
    int prefix = 64, recent = 256;
    const float* rotations = nullptr;  // device [2][16][256][256] (K banks, V banks)
    const float* signs = nullptr;      // device [256]
};

inline const Config& config() {
    static const Config value = [] {
        Config c;
        const char* mode = std::getenv("NINFER_EXL3_KV_FAKEQUANT");
        if (!mode || !*mode || std::strcmp(mode, "0") == 0) return c;
        if (std::strcmp(mode, "int2") == 0) c.mode = 1;
        else if (std::strcmp(mode, "turbo") == 0) c.mode = 2;
        else if (std::strcmp(mode, "fp8") == 0) c.mode = 3;
        else if (std::strcmp(mode, "int4") == 0) { c.mode = 4; c.levels = 15.0f; }
        else if (std::strcmp(mode, "int3") == 0) { c.mode = 5; c.levels = 7.0f; }
        else throw std::invalid_argument("NINFER_EXL3_KV_FAKEQUANT must be int2, int3, int4, turbo or fp8");
        if (const char* r = std::getenv("NINFER_EXL3_KV_FAKEQUANT_RECENT")) c.recent = std::atoi(r);
        if (const char* p = std::getenv("NINFER_EXL3_KV_FAKEQUANT_PREFIX")) c.prefix = std::atoi(p);
        std::vector<float> rot(2ull * 16 * 256 * 256);
        if (c.mode == 1 || c.mode >= 4) {
            const char* dir = std::getenv("NINFER_EXL3_KV_FAKEQUANT_ROT");
            if (!dir) throw std::invalid_argument("int2 fake-quant requires NINFER_EXL3_KV_FAKEQUANT_ROT");
            const char* names[2] = {"/runtime/k_rotation_fp32.bin", "/runtime/v_rotation_fp32.bin"};
            for (int kind = 0; kind < 2; ++kind) {
                std::ifstream f(std::string(dir) + names[kind], std::ios::binary);
                f.read(reinterpret_cast<char*>(rot.data() + kind * 16ull * 65536), 16ll * 65536 * 4);
                if (!f) throw std::runtime_error("fake-quant rotation read");
            }
        }
        std::vector<float> signs(256);
        const std::uint64_t seed = 0x74616e676c653235ULL;
        for (int i = 0; i < 256; ++i) {
            std::uint64_t x = seed + (static_cast<std::uint64_t>(i) + 1) * 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            signs[i] = ((x ^ (x >> 31)) & 1) ? -1.0f : 1.0f;
        }
        float* device = nullptr;
        if (cudaMalloc(&device, rot.size() * 4 + 256 * 4) != cudaSuccess)
            throw std::runtime_error("fake-quant allocation");
        cudaMemcpy(device, rot.data(), rot.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(device + rot.size(), signs.data(), 256 * 4, cudaMemcpyHostToDevice);
        c.rotations = device;
        c.signs = device + rot.size();
        return c;
    }();
    return value;
}

__device__ __forceinline__ float block_reduce(float v, bool is_max, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, v, o);
        v = is_max ? fmaxf(v, other) : (v + other);
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = v;
    __syncthreads();
    float r = scratch[0];
    for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w)
        r = is_max ? fmaxf(r, scratch[w]) : (r + scratch[w]);
    return r;
}

__device__ __forceinline__ void hadamard256(float* x) {
    for (int h = 1; h < 256; h <<= 1) {
        __syncthreads();
        const int j = threadIdx.x;
        float a = 0, b = 0;
        int lo = 0;
        if (j < 128) {
            lo = (j / h) * 2 * h + (j % h);
            a = x[lo]; b = x[lo + h];
        }
        __syncthreads();
        if (j < 128) { x[lo] = a + b; x[lo + h] = a - b; }
    }
    __syncthreads();
    x[threadIdx.x] *= 0.0625f;
    __syncthreads();
}

// One CTA (256 threads) per (row, head, K|V) task.
__global__ void __launch_bounds__(256) age_kernel(std::uint16_t* k_cache, std::uint16_t* v_cache,
    int capacity, const int* position_device, int position, const int* watermark, Config c, int bank) {
    __shared__ float x[256], y[256], scratch[8];
    __shared__ float group_stats[4];
    const int base = position_device ? *position_device : position;
    int wm = *watermark;
    const int end = min(base - c.recent + 1, capacity);
    if (end < wm) wm = c.prefix;
    wm = max(wm, c.prefix);
    const int tasks = (end - wm) * 8;
    const int j = threadIdx.x;
    for (int t = blockIdx.x; t < tasks; t += gridDim.x) {
        const int row = wm + t / 8, head = (t / 2) % 4, kind = t % 2;
        auto* cache = (kind ? v_cache : k_cache) + (static_cast<std::size_t>(row) * 4 + head) * 256;
        __syncthreads();
        x[j] = __half2float(__ushort_as_half(cache[j]));
        __syncthreads();
        float out = x[j];
        if (c.mode == 1 || c.mode >= 4) {
            const float* R = c.rotations + (static_cast<std::size_t>(kind) * 16 + bank) * 65536;
            float acc = 0.0f;
            for (int i = 0; i < 256; ++i) acc = fmaf(x[i], R[i * 256 + j], acc);
            y[j] = acc;
            __syncthreads();
            const float a = fabsf(acc);
            int rank = 0;
            for (int i = 0; i < 256; ++i) {
                const float b = fabsf(y[i]);
                rank += (b < a) || (b == a && i < j);
            }
            // INT2 uses the OSCAR clip ratios (0.92 V, 0.96 K); wider codes keep the full range.
            const int clip_index = c.mode == 1 ? (kind ? 235 : 245) : 255;
            if (rank == clip_index) scratch[7] = a;
            __syncthreads();
            const float thr = scratch[7];
            const float clipped = fminf(fmaxf(acc, -thr), thr);
            // group min/max over 128 lanes (warps 0-3 group 0, 4-7 group 1)
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
            const float scale = fmaxf(gmx - gmn, 1.0e-8f) / c.levels;
            const float zero = -gmn / scale;
            const float code = fminf(fmaxf(floorf(clipped / scale + zero + 0.5f), 0.0f), c.levels);
            __syncthreads();
            y[j] = (code - zero) * scale;
            __syncthreads();
            acc = 0.0f;
            for (int i = 0; i < 256; ++i) acc = fmaf(y[i], R[j * 256 + i], acc);
            out = acc;
        } else if (c.mode == 2) {
            x[j] *= c.signs[j];
            hadamard256(x);
            const bool key = kind == 0;
            float norm = 0.0f, radius = 0.0f, theta = 0.0f;
            if (j < 128) {
                radius = hypotf(x[2 * j], x[2 * j + 1]);
                theta = atan2f(x[2 * j + 1], x[2 * j]);
                norm = key ? radius : logf(fmaxf(radius, 1e-30f));
            }
            const float low = -block_reduce(j < 128 ? -norm : -INFINITY, true, scratch);
            const float high = block_reduce(j < 128 ? norm : -INFINITY, true, scratch);
            const int angle_bits = key ? 7 : 6, norm_bits = key ? 8 : 4;
            const int bins = 1 << angle_bits, levels = (1 << norm_bits) - 1;
            const float scale = fmaxf(0.0f, high - low) / levels;
            __syncthreads();
            if (j < 128) {
                const int a = static_cast<int>(lrintf(bins * theta / 6.283185307179586f)) & (bins - 1);
                const float nc = scale > 0 ? fminf(fmaxf(rintf((norm - low) / scale), 0.0f), static_cast<float>(levels)) : 0.0f;
                float r = low + nc * scale;
                if (!key) r = expf(r);
                const float th = 6.283185307179586f * a / bins;
                x[2 * j] = r * cosf(th);
                x[2 * j + 1] = r * sinf(th);
            }
            hadamard256(x);
            out = x[j] * c.signs[j];
        } else if (c.mode == 3) {
            const float amax = block_reduce(fabsf(x[j]), true, scratch);
            const float raw = amax / 448.0f;
            const float s = __half2float(__float2half_rn(fminf(65504.0f, fmaxf(0x1p-24f, raw))));
            if (amax == 0.0f) out = 0.0f;
            else {
                const __nv_fp8_e4m3 q(fminf(fmaxf(x[j] / s, -448.0f), 448.0f));
                out = static_cast<float>(q) * s;
            }
        }
        __syncthreads();
        cache[j] = __half_as_ushort(__float2half_rn(out));
    }
}

__global__ void advance_kernel(const int* position_device, int position, int* watermark, Config c,
                               int capacity) {
    const int base = position_device ? *position_device : position;
    int wm = *watermark;
    const int end = min(base - c.recent + 1, capacity);
    if (end < wm) wm = c.prefix;
    wm = max(wm, c.prefix);
    *watermark = max(wm, end);
}

inline void age(std::uint16_t* k_cache, std::uint16_t* v_cache, int capacity,
                const int* position_device, int position, int* watermark, int bank, cudaStream_t stream) {
    const Config& c = config();
    if (!c.mode) return;
    age_kernel<<<256, 256, 0, stream>>>(k_cache, v_cache, capacity, position_device, position,
                                        watermark, c, bank);
    advance_kernel<<<1, 1, 0, stream>>>(position_device, position, watermark, c, capacity);
}

} // namespace ninfer::exl3::kv_fakequant
