#pragma once
// FP8 L2 pages for L0 OSCAR contexts (NINFER_EXL3_L0_L2_FP8=1): parked exact
// host states keep their K/V history as e4m3 with one FP16 scale per (row, KV
// head) (absmax / 448). Row-interleaved layout so pages extend in place:
// row r = 1024 value bytes (4 heads x 256) + 4 FP16 scales = 516 uint16.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::exl3::l0_l2_fp8 {

constexpr int kRowBytes = 1032;
constexpr int kRowWords = kRowBytes / 2;   // 516 uint16 per row
constexpr int kStagingRows = 16384;       // bounded staging window (multiple of 64)

inline bool enabled() {
    static const bool value = [] {
        const char* v = std::getenv("NINFER_EXL3_L0_L2_FP8");
        if (!v || !*v || std::strcmp(v, "0") == 0) return false;
        if (std::strcmp(v, "1") != 0) throw std::invalid_argument("NINFER_EXL3_L0_L2_FP8 must be 0 or 1");
        return true;
    }();
    return value;
}

// One warp per (row, KV head): FP16 plane rows -> packed FP8 rows.
__global__ void pack_kernel(const std::uint16_t* plane, std::uint8_t* packed, int rows) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= rows * 4) return;
    const int row = warp >> 2, head = warp & 3;
    const auto* src = reinterpret_cast<const __half*>(plane) + (static_cast<std::size_t>(row) * 4 + head) * 256 + lane * 8;
    const uint4 raw = *reinterpret_cast<const uint4*>(src);
    const __half* h = reinterpret_cast<const __half*>(&raw);
    float x[8], amax = 0.f;
    for (int i = 0; i < 8; ++i) { x[i] = __half2float(h[i]); amax = fmaxf(amax, fabsf(x[i])); }
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const __half scale_h = __float2half_rn(amax > 0.f ? amax / 448.f : 1.f);
    const float inv = 1.f / __half2float(scale_h);
    std::uint8_t q[8];
    for (int i = 0; i < 8; ++i) {
        const __nv_fp8_e4m3 v(fminf(fmaxf(x[i] * inv, -448.f), 448.f));
        q[i] = v.__x;
    }
    std::uint8_t* out = packed + static_cast<std::size_t>(row) * kRowBytes;
    *reinterpret_cast<uint2*>(out + head * 256 + lane * 8) = *reinterpret_cast<const uint2*>(q);
    if (lane == 0) *reinterpret_cast<__half*>(out + 1024 + head * 2) = scale_h;
}

__global__ void unpack_kernel(const std::uint8_t* packed, std::uint16_t* plane, int rows) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (warp >= rows * 4) return;
    const int row = warp >> 2, head = warp & 3;
    const std::uint8_t* in = packed + static_cast<std::size_t>(row) * kRowBytes;
    const float scale = __half2float(*reinterpret_cast<const __half*>(in + 1024 + head * 2));
    const uint2 raw = *reinterpret_cast<const uint2*>(in + head * 256 + lane * 8);
    const std::uint8_t* q = reinterpret_cast<const std::uint8_t*>(&raw);
    __half h[8];
    for (int i = 0; i < 8; ++i) {
        __nv_fp8_e4m3 v; v.__x = q[i];
        h[i] = __float2half_rn(static_cast<float>(v) * scale);
    }
    *reinterpret_cast<uint4*>(reinterpret_cast<__half*>(plane) + (static_cast<std::size_t>(row) * 4 + head) * 256 + lane * 8) =
        *reinterpret_cast<const uint4*>(h);
}

// Process-wide staging (contexts export/restore one at a time on the host).
struct Staging {
    std::uint8_t* device = nullptr;
    std::uint8_t* host = nullptr;
    std::size_t bytes = 0;
    void reserve(std::size_t need) {
        if (need <= bytes) return;
        if (device) cudaFree(device);
        if (host) cudaFreeHost(host);
        device = nullptr; host = nullptr; bytes = 0;
        if (cudaMalloc(reinterpret_cast<void**>(&device), need) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void**>(&host), need) != cudaSuccess)
            throw std::runtime_error("L0 L2 FP8 staging allocation");
        bytes = need;
    }
};
inline Staging& staging() { static Staging value; return value; }

}  // namespace ninfer::exl3::l0_l2_fp8
