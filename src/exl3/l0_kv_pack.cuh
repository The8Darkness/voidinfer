#pragma once
// Lossless packing of exact L0 KV pages (lane_kv_source.h). The FP16 rows of
// a 64-row page are coded against the L0 OSCAR INT4 codes of the same rows,
// which the exported state keeps anyway (Exl3ExactHostState code chunks), so
// a packed page costs ~11.7 bits per value instead of 16 (1.36x measured on
// a 16K-token code context) and decodes back to the identical FP16 bits.
//
// Prediction: x^ = dequant(codes) R^T (+ mu for K), rounded to FP16. Every
// float operation is an explicitly rounded intrinsic in a fixed order and the
// encoder and decoder are one kernel, so both sides derive the same x^. The
// FP16 bits are coded as the zigzag difference of their monotone integer
// order to x^'s, in a width the decoder derives from x^ and the row scales:
// w = clamp(floor(log2(sigma / ulp(x^))) + 3, 1, 15), sigma^2 = (s0^2 + s1^2)
// / 24. A value that does not fit (code 2^w - 1) is an escape whose raw bits
// go to a side stream. Each stream carries a checksum of its FP16 values that
// decoding verifies; a stream that would not shrink is stored raw.
//
// Blob of one page: Header, Entry[streams], 8-byte aligned stream payloads.
// Streams are (bank, K|V, head), page rows x 256 values each, coded in
// 16-row batches: within a batch thread-major (dim i, then row).
//
// Rows, codes and meta live in host memory (L0 planes, code chunks): a fetch
// kernel stages them into device memory with wide coalesced loads and a store
// kernel writes decoded rows back, so the coding kernel never waits on PCIe.
#include "exl3/l0_oscar_storage.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace voidinfer::exl3::l0_kv_pack {

constexpr int kPageRows = 64;
constexpr int kDim = 256;
constexpr int kHeads = 4;
constexpr int kRowWords = kHeads * kDim;
constexpr int kBanks = 16;
constexpr int kMaxStreams = kBanks * 2 * kHeads;
constexpr int kBatchRows = 16;
constexpr int kTileCols = 16;
constexpr int kWidthBias = 3;
constexpr int kMaxWidth = 15;
constexpr std::uint32_t kMagic = 0x3150564bu;   // "KVP1"
// Encoder scratch per stream: main bits, escapes, then the result header.
constexpr std::size_t kMainBytes = static_cast<std::size_t>(kPageRows) * kDim * kMaxWidth / 8;
constexpr std::size_t kEscBytes = static_cast<std::size_t>(kPageRows) * kDim * 2;
constexpr std::size_t kScratchStride = 65536;
static_assert(kMainBytes + kEscBytes + 32 <= kScratchStride, "pack scratch stride");
constexpr int kBatchPages = 4;

struct Header {
    std::uint32_t magic, rows, streams, bytes;
};
struct Entry {
    std::uint32_t offset, mode, main_words, escapes;   // mode 0 coded, 1 raw
    unsigned long long checksum;
};
static_assert(sizeof(Header) == 16 && sizeof(Entry) == 24, "pack blob layout");
__host__ __device__ constexpr std::size_t payload_start(int streams) {
    return (sizeof(Header) + static_cast<std::size_t>(streams) * sizeof(Entry) + 7) / 8 * 8;
}
constexpr std::size_t kBlobCap = payload_start(kMaxStreams) +
    static_cast<std::size_t>(kMaxStreams) * kPageRows * kDim * 2;

// Rows of one plane set: [row][4][256] FP16 per bank, device-addressable
// (16-byte aligned rows).
struct Planes {
    std::uint16_t* k[kBanks];
    std::uint16_t* v[kBanks];
};
// One page: its rows [row, row + rows) of the plane set and, per bank and
// K|V, its codes ([rows][4][128] bytes) and meta ([rows][4][4] FP16) from the
// page's first row on (device-addressable, 16-byte aligned).
struct Job {
    const std::uint8_t* codes[kBanks][2];
    const std::uint16_t* meta[kBanks][2];
    int row, rows;
    unsigned long long blob;   // decode: byte offset of the page's blob
};

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ int order16(unsigned u) {
    return (u & 0x8000u) ? static_cast<int>(0xFFFFu - u) : static_cast<int>(u + 0x8000u);
}
__device__ __forceinline__ unsigned unorder16(int o) {
    return o >= 0x8000 ? static_cast<unsigned>(o - 0x8000) : static_cast<unsigned>(0xFFFF - o);
}
// floor(log2(sigma / ulp(x^))) + bias from exact exponent arithmetic.
__device__ __forceinline__ int value_width(unsigned hb, float s0, float s1) {
    const float var = __fmul_rn(__fadd_rn(__fmul_rn(s0, s0), __fmul_rn(s1, s1)), 1.0f / 24.0f);
    const int ev = static_cast<int>((__float_as_uint(var) >> 23) & 255u);
    const int e = static_cast<int>((hb >> 10) & 31u);
    if (ev == 0 || ev == 255 || e == 31) return kMaxWidth;
    const int log2_ratio_sq = (ev - 127) - 2 * (max(e, 1) - 25);
    return min(max((log2_ratio_sq >> 1) + kWidthBias, 1), kMaxWidth);
}
// Exclusive prefix of `value` over the CTA's 256 threads; `total` gets the sum.
__device__ __forceinline__ unsigned block_scan(unsigned value, unsigned* scratch, unsigned& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned inclusive = value;
    for (int o = 1; o < 32; o <<= 1) {
        const unsigned n = __shfl_up_sync(0xffffffffu, inclusive, o);
        if (lane >= o) inclusive += n;
    }
    __syncthreads();
    if (lane == 31) scratch[warp] = inclusive;
    __syncthreads();
    unsigned before = 0;
    total = 0;
    for (int w = 0; w < 8; ++w) {
        if (w < warp) before += scratch[w];
        total += scratch[w];
    }
    return before + inclusive - value;
}

// Device staging of a batch, per (job, bank, K|V) slot: 64 rows of FP16
// [4][256], codes [4][128] and meta [4][4].
constexpr std::size_t kSlotRowWords = static_cast<std::size_t>(kPageRows) * kRowWords;
constexpr std::size_t kSlotCodeBytes = static_cast<std::size_t>(kPageRows) * kHeads * kDim / 2;
constexpr std::size_t kSlotMetaWords = static_cast<std::size_t>(kPageRows) * kHeads * 4;
__device__ __forceinline__ void copy16(void* dst, const void* src, std::size_t bytes) {
    auto* d = static_cast<uint4*>(dst);
    const auto* s = static_cast<const uint4*>(src);
    const std::size_t n = bytes / 16;
    std::size_t t = threadIdx.x;
    for (; t + 3 * blockDim.x < n; t += 4 * blockDim.x) {
        const uint4 a = s[t], b = s[t + blockDim.x], c = s[t + 2 * blockDim.x], e = s[t + 3 * blockDim.x];
        d[t] = a; d[t + blockDim.x] = b; d[t + 2 * blockDim.x] = c; d[t + 3 * blockDim.x] = e;
    }
    for (; t < n; t += blockDim.x) d[t] = s[t];
}
// grid (banks * 2, jobs): stages each slot's codes, meta and (rows != null) rows.
__global__ void __launch_bounds__(256) fetch_kernel(const Job* jobs, int banks, Planes planes,
    std::uint16_t* rows, std::uint8_t* codes, std::uint16_t* meta) {
    const Job& job = jobs[blockIdx.y];
    const int bank = static_cast<int>(blockIdx.x) >> 1, plane = static_cast<int>(blockIdx.x) & 1;
    const std::size_t slot = (static_cast<std::size_t>(blockIdx.y) * banks + bank) * 2 + plane;
    copy16(codes + slot * kSlotCodeBytes, job.codes[bank][plane], static_cast<std::size_t>(job.rows) * kHeads * kDim / 2);
    copy16(meta + slot * kSlotMetaWords, job.meta[bank][plane], static_cast<std::size_t>(job.rows) * kHeads * 4 * 2);
    if (rows)
        copy16(rows + slot * kSlotRowWords, (plane ? planes.v[bank] : planes.k[bank]) +
            static_cast<std::size_t>(job.row) * kRowWords, static_cast<std::size_t>(job.rows) * kRowWords * 2);
}
// grid (banks * 2, jobs): writes each slot's decoded rows into the planes
// unless decoding flagged an error.
__global__ void __launch_bounds__(256) store_kernel(const Job* jobs, int banks, Planes planes,
    const std::uint16_t* rows, const unsigned* error) {
    if (*error) return;
    const Job& job = jobs[blockIdx.y];
    const int bank = static_cast<int>(blockIdx.x) >> 1, plane = static_cast<int>(blockIdx.x) & 1;
    const std::size_t slot = (static_cast<std::size_t>(blockIdx.y) * banks + bank) * 2 + plane;
    copy16((plane ? planes.v[bank] : planes.k[bank]) + static_cast<std::size_t>(job.row) * kRowWords,
        rows + slot * kSlotRowWords, static_cast<std::size_t>(job.rows) * kRowWords * 2);
}

// grid (streams, jobs), 256 threads (thread = dim), on staged slots.
// decode == 0: codes the slots' rows into `io` (scratch); otherwise reads
// the blobs at `io`, writes the rows into the slots and flags any mismatch
// in `error`.
__global__ void __launch_bounds__(256) code_kernel(const Job* jobs, int banks, l0_oscar::RotationView assets,
    int decode, std::uint16_t* staged_rows, const std::uint8_t* staged_codes, const std::uint16_t* staged_meta,
    std::uint8_t* io, unsigned* error) {
    __shared__ __align__(16) float yh[kDim][kBatchRows];   // [j][row]: float4 row reads
    __shared__ float tile[kTileCols][kDim + 1];
    __shared__ float scales[kBatchRows][2];
    __shared__ unsigned scan[8];
    __shared__ unsigned long long sums[8];
    const Job& job = jobs[blockIdx.y];
    const int stream = static_cast<int>(blockIdx.x);
    const int bank = stream >> 3, plane = (stream >> 2) & 1, head = stream & 3;
    const int i = static_cast<int>(threadIdx.x);
    const int rows = job.rows;
    const std::size_t slot = (static_cast<std::size_t>(blockIdx.y) * banks + bank) * 2 + plane;
    const std::uint8_t* codes = staged_codes + slot * kSlotCodeBytes + head * (kDim / 2);
    const std::uint16_t* meta = staged_meta + slot * kSlotMetaWords + head * 4;
    const float* R = (plane ? assets.rv : assets.rk) + (static_cast<std::size_t>(bank) * kHeads + head) * kDim * kDim;
    const float mu = plane ? 0.0f : assets.mu[(bank * kHeads + head) * kDim + i];
    std::uint16_t* rows_base = staged_rows + slot * kSlotRowWords + head * kDim;
    std::uint32_t* main = nullptr;
    std::uint16_t* esc = nullptr;
    const Entry* entry = nullptr;
    if (decode) {
        const std::uint8_t* blob = io + job.blob;
        entry = reinterpret_cast<const Entry*>(blob + sizeof(Header)) + stream;
        main = reinterpret_cast<std::uint32_t*>(const_cast<std::uint8_t*>(blob) + entry->offset);
        esc = reinterpret_cast<std::uint16_t*>(main + entry->main_words);
    } else {
        std::uint8_t* base = io + (static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * kScratchStride;
        main = reinterpret_cast<std::uint32_t*>(base);
        esc = reinterpret_cast<std::uint16_t*>(base + kMainBytes);
    }
    unsigned long long sum = 0;
    const auto mixed = [&](int r, unsigned u) {
        return mix64((static_cast<unsigned long long>(stream) << 40) |
                     (static_cast<unsigned long long>(r * kDim + i) << 16) | u);
    };
    unsigned carry_bits = 0, carry_esc = 0;
    if (decode && entry->mode == 1) {
        for (int r = 0; r < rows; ++r) {
            const unsigned u = reinterpret_cast<const std::uint16_t*>(main)[r * kDim + i];
            rows_base[static_cast<std::size_t>(r) * kRowWords + i] = static_cast<std::uint16_t>(u);
            sum += mixed(r, u);
        }
    } else {
        for (int r0 = 0; r0 < rows; r0 += kBatchRows) {
            const int n = min(kBatchRows, rows - r0);
            for (int r = 0; r < kBatchRows; ++r) {
                float y = 0.0f;
                if (r < n) {
                    const int row = r0 + r;
                    const unsigned byte = codes[static_cast<std::size_t>(row) * (kHeads * kDim / 2) + (i >> 1)];
                    const unsigned q = (byte >> ((i & 1) * 4)) & 15u;
                    const std::uint16_t* m = meta + static_cast<std::size_t>(row) * kHeads * 4;
                    const int g = i >> 7;
                    const float scale = __half2float(__ushort_as_half(m[2 * g]));
                    const float zero = __half2float(__ushort_as_half(m[2 * g + 1]));
                    y = __fmul_rn(__fsub_rn(static_cast<float>(q), zero), scale);
                    if (i == 0) {
                        scales[r][0] = __half2float(__ushort_as_half(m[0]));
                        scales[r][1] = __half2float(__ushort_as_half(m[2]));
                    }
                }
                yh[i][r] = y;
            }
            // x^_i = sum_j y_j R[i][j] in ascending j (one fmaf chain per value).
            float acc[kBatchRows];
#pragma unroll
            for (int r = 0; r < kBatchRows; ++r) acc[r] = 0.0f;
            for (int j0 = 0; j0 < kDim; j0 += kTileCols) {
                __syncthreads();
                for (int t = i; t < kDim * kTileCols; t += 256) {
                    const int row = t / kTileCols, col = t % kTileCols;
                    tile[col][row] = R[static_cast<std::size_t>(row) * kDim + j0 + col];
                }
                __syncthreads();
#pragma unroll
                for (int jj = 0; jj < kTileCols; ++jj) {
                    const float rij = tile[jj][i];
                    const float4* y4 = reinterpret_cast<const float4*>(yh[j0 + jj]);
#pragma unroll
                    for (int q = 0; q < kBatchRows / 4; ++q) {
                        const float4 y = y4[q];
                        acc[4 * q] = __fmaf_rn(y.x, rij, acc[4 * q]);
                        acc[4 * q + 1] = __fmaf_rn(y.y, rij, acc[4 * q + 1]);
                        acc[4 * q + 2] = __fmaf_rn(y.z, rij, acc[4 * q + 2]);
                        acc[4 * q + 3] = __fmaf_rn(y.w, rij, acc[4 * q + 3]);
                    }
                }
            }
            unsigned hb[kBatchRows], width[kBatchRows], value[kBatchRows];
            unsigned bits = 0;
#pragma unroll
            for (int r = 0; r < kBatchRows; ++r) {
                const float x = plane ? acc[r] : __fadd_rn(acc[r], mu);
                hb[r] = __half_as_ushort(__float2half_rn(x));
                width[r] = r < n ? static_cast<unsigned>(value_width(hb[r], scales[r][0], scales[r][1])) : 0u;
                bits += width[r];
            }
            unsigned total_bits = 0;
            unsigned pos = carry_bits + block_scan(bits, scan, total_bits);
            unsigned escapes = 0;
            if (!decode) {
#pragma unroll
                for (int r = 0; r < kBatchRows; ++r) {
                    if (r >= n) continue;
                    const unsigned u = rows_base[static_cast<std::size_t>(r0 + r) * kRowWords + i];
                    value[r] = u;
                    const int d = order16(u) - order16(hb[r]);
                    const unsigned z = (static_cast<unsigned>(d) << 1) ^ static_cast<unsigned>(d >> 31);
                    const unsigned top = (1u << width[r]) - 1u;
                    const unsigned code = z >= top ? top : z;
                    escapes += z >= top;
                    const unsigned word = pos >> 5, shift = pos & 31u;
                    atomicOr(main + word, code << shift);
                    if (shift + width[r] > 32u) atomicOr(main + word + 1, code >> (32u - shift));
                    pos += width[r];
                    sum += mixed(r0 + r, u);
                }
            } else {
#pragma unroll
                for (int r = 0; r < kBatchRows; ++r) {
                    if (r >= n) continue;
                    const unsigned word = pos >> 5, shift = pos & 31u;
                    unsigned code = main[word] >> shift;
                    if (shift + width[r] > 32u) code |= main[word + 1] << (32u - shift);
                    code &= (1u << width[r]) - 1u;
                    value[r] = code;
                    escapes += code == (1u << width[r]) - 1u;
                    pos += width[r];
                }
            }
            unsigned total_esc = 0;
            unsigned at = carry_esc + block_scan(escapes, scan, total_esc);
#pragma unroll
            for (int r = 0; r < kBatchRows; ++r) {
                if (r >= n) continue;
                const unsigned top = (1u << width[r]) - 1u;
                if (!decode) {
                    const int d = order16(value[r]) - order16(hb[r]);
                    const unsigned z = (static_cast<unsigned>(d) << 1) ^ static_cast<unsigned>(d >> 31);
                    if (z >= top) esc[at++] = static_cast<std::uint16_t>(value[r]);
                } else {
                    unsigned u;
                    if (value[r] == top) u = esc[at++];
                    else {
                        const int d = static_cast<int>(value[r] >> 1) ^ -static_cast<int>(value[r] & 1u);
                        u = unorder16(min(max(order16(hb[r]) + d, 0), 0xFFFF));
                    }
                    rows_base[static_cast<std::size_t>(r0 + r) * kRowWords + i] = static_cast<std::uint16_t>(u);
                    sum += mixed(r0 + r, u);
                }
            }
            carry_bits += total_bits;
            carry_esc += total_esc;
        }
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    __syncthreads();
    if ((i & 31) == 0) sums[i >> 5] = sum;
    __syncthreads();
    if (i == 0) {
        unsigned long long total = 0;
        for (int w = 0; w < 8; ++w) total += sums[w];
        if (decode) {
            const bool coded = entry->mode == 0;
            if (total != entry->checksum ||
                (coded && (entry->main_words != (carry_bits + 31) / 32 || entry->escapes != carry_esc)))
                atomicExch(error, 1u);
        } else {
            auto* result = reinterpret_cast<unsigned*>(reinterpret_cast<std::uint8_t*>(main) + kMainBytes + kEscBytes);
            result[0] = (carry_bits + 31) / 32;
            result[1] = carry_esc;
            *reinterpret_cast<unsigned long long*>(result + 2) = total;
        }
    }
}

// One 256-thread CTA per job, one thread per stream: chooses coded or raw per stream and
// lays out the job's blob table; sizes[job] gets the blob bytes.
__global__ void layout_kernel(const Job* jobs, int streams, const std::uint8_t* scratch, std::uint8_t* out,
    unsigned long long* sizes) {
    __shared__ unsigned scan[8];
    const int s = static_cast<int>(threadIdx.x), job = static_cast<int>(blockIdx.x);
    const int rows = jobs[job].rows;
    unsigned bytes = 0, mode = 0, main_words = 0, escapes = 0;
    unsigned long long checksum = 0;
    if (s < streams) {
        const auto* result = reinterpret_cast<const unsigned*>(
            scratch + (static_cast<std::size_t>(job) * streams + s) * kScratchStride + kMainBytes + kEscBytes);
        main_words = result[0];
        escapes = result[1];
        checksum = *reinterpret_cast<const unsigned long long*>(result + 2);
        const unsigned coded = (main_words * 4 + escapes * 2 + 7) / 8 * 8;
        const unsigned raw = static_cast<unsigned>(rows) * kDim * 2;
        mode = coded < raw ? 0u : 1u;
        bytes = mode ? raw : coded;
    }
    unsigned total = 0;
    const unsigned before = block_scan(bytes, scan, total);
    std::uint8_t* blob = out + static_cast<std::size_t>(job) * kBlobCap;
    const unsigned start = static_cast<unsigned>(payload_start(streams));
    if (s < streams) {
        auto* entry = reinterpret_cast<Entry*>(blob + sizeof(Header)) + s;
        entry->offset = start + before;
        entry->mode = mode;
        entry->main_words = mode ? 0u : main_words;
        entry->escapes = mode ? 0u : escapes;
        entry->checksum = checksum;
    }
    if (s == 0) {
        auto* header = reinterpret_cast<Header*>(blob);
        header->magic = kMagic;
        header->rows = static_cast<unsigned>(rows);
        header->streams = static_cast<unsigned>(streams);
        header->bytes = start + total;
        sizes[job] = start + total;
    }
}

// grid (streams, jobs): copies each stream's payload into its blob.
__global__ void gather_kernel(const Job* jobs, int banks, const std::uint16_t* staged_rows,
    const std::uint8_t* scratch, std::uint8_t* out) {
    const int stream = static_cast<int>(blockIdx.x), job = static_cast<int>(blockIdx.y);
    const int bank = stream >> 3, plane = (stream >> 2) & 1, head = stream & 3;
    std::uint8_t* blob = out + static_cast<std::size_t>(job) * kBlobCap;
    const Entry entry = reinterpret_cast<const Entry*>(blob + sizeof(Header))[stream];
    if (entry.mode == 1) {
        const std::uint16_t* rows_base = staged_rows +
            ((static_cast<std::size_t>(job) * banks + bank) * 2 + plane) * kSlotRowWords + head * kDim;
        auto* raw = reinterpret_cast<std::uint16_t*>(blob + entry.offset);
        for (int t = threadIdx.x; t < jobs[job].rows * kDim; t += blockDim.x)
            raw[t] = rows_base[static_cast<std::size_t>(t / kDim) * kRowWords + t % kDim];
        return;
    }
    const std::uint8_t* base = scratch + (static_cast<std::size_t>(job) * gridDim.x + stream) * kScratchStride;
    auto* main = reinterpret_cast<std::uint32_t*>(blob + entry.offset);
    for (unsigned t = threadIdx.x; t < entry.main_words; t += blockDim.x)
        main[t] = reinterpret_cast<const std::uint32_t*>(base)[t];
    auto* esc = reinterpret_cast<std::uint16_t*>(main + entry.main_words);
    for (unsigned t = threadIdx.x; t < entry.escapes; t += blockDim.x)
        esc[t] = reinterpret_cast<const std::uint16_t*>(base + kMainBytes)[t];
}

inline bool enabled() {
    static const bool value = [] {
        const char* v = std::getenv("VOIDINFER_EXL3_KV_PACK");
        return !v || std::strcmp(v, "0") != 0;
    }();
    return value;
}

// Process-wide packer: device scratch for kBatchPages pages, serialized.
class Packer {
public:
    // Host-side description of one page job (pointers are host pointers of
    // page-locked memory or device pointers; both are device-addressable).
    struct Page {
        int row = 0, rows = 0;
        std::array<const std::uint8_t*, kBanks> k_codes{}, v_codes{};
        std::array<const std::uint16_t*, kBanks> k_meta{}, v_meta{};
    };

    static Packer& instance() {
        static Packer packer;
        return packer;
    }

    // Packs pages[i]'s rows of `source` (banks <= 16 banks) into out[i]; an
    // empty blob means the page would not shrink (keep it raw). Batches run
    // through two host slots: the GPU writes batch n's blobs into page-locked
    // slot memory while the CPU copies batch n - 1's out.
    void pack(const Planes& source, int banks, const std::vector<Page>& pages,
              std::vector<std::vector<std::uint8_t>>& out) {
        std::lock_guard lock(mutex_);
        ensure();
        out.assign(pages.size(), {});
        const int streams = banks * 2 * kHeads;
        const auto consume = [&](std::size_t begin, int slot) {
            check(cudaEventSynchronize(slots_[slot].done), "KV pack");
            const int count = static_cast<int>(std::min<std::size_t>(kBatchPages, pages.size() - begin));
            for (int j = 0; j < count; ++j) {
                const std::size_t raw = static_cast<std::size_t>(pages[begin + j].rows) * kRowWords * 2 * 2 * banks;
                const auto bytes = slots_[slot].sizes[j];
                if (bytes >= raw) continue;
                const auto* blob = slots_[slot].host + static_cast<std::size_t>(j) * kBlobCap;
                out[begin + j].assign(blob, blob + bytes);
                packed_bytes_ += bytes;
            }
        };
        std::size_t previous = 0;
        int batch = 0;
        for (std::size_t begin = 0; begin < pages.size(); begin += kBatchPages, ++batch) {
            const int slot = batch & 1;
            const int count = static_cast<int>(std::min<std::size_t>(kBatchPages, pages.size() - begin));
            upload_jobs(pages, begin, count, banks, nullptr, &source, slot);
            check(cudaMemsetAsync(scratch_, 0, static_cast<std::size_t>(count) * streams * kScratchStride, stream_),
                "KV pack scratch clear");
            fetch_kernel<<<dim3(banks * 2, count), 256, 0, stream_>>>(jobs_device_, banks, source, rows_, codes_, meta_);
            check(cudaGetLastError(), "KV pack fetch");
            code_kernel<<<dim3(streams, count), 256, 0, stream_>>>(jobs_device_, banks, l0_oscar::rotation_view(), 0,
                rows_, codes_, meta_, scratch_, error_);
            check(cudaGetLastError(), "KV pack encode");
            layout_kernel<<<count, 256, 0, stream_>>>(jobs_device_, streams, scratch_, slots_[slot].host_device,
                slots_[slot].sizes_device);
            check(cudaGetLastError(), "KV pack layout");
            gather_kernel<<<dim3(streams, count), 256, 0, stream_>>>(jobs_device_, banks, rows_, scratch_,
                slots_[slot].host_device);
            check(cudaGetLastError(), "KV pack gather");
            check(cudaEventRecord(slots_[slot].done, stream_), "KV pack");
            if (batch) consume(previous, slot ^ 1);
            previous = begin;
        }
        if (batch) consume(previous, (batch - 1) & 1);
    }

    // Decodes blobs[i] into pages[i]'s rows of `destination`, or (no
    // destination) into the host buffers host_rows(i, bank, K|V) returns.
    // Throws if any blob is malformed or does not reproduce its checksums
    // (rows of a failing batch are not stored). The CPU stages batch n + 1's
    // blobs in the other host slot while the GPU decodes batch n.
    using HostRows = std::function<std::uint16_t*(std::size_t page, int bank, int plane)>;
    void unpack(const Planes* destination, int banks, const std::vector<Page>& pages,
                const std::vector<const std::uint8_t*>& blobs, const std::vector<std::size_t>& sizes,
                const HostRows& host_rows = {}) {
        std::lock_guard lock(mutex_);
        ensure();
        const int streams = banks * 2 * kHeads;
        check(cudaMemsetAsync(error_, 0, sizeof(unsigned), stream_), "KV unpack error clear");
        int batch = 0;
        for (std::size_t begin = 0; begin < pages.size(); begin += kBatchPages, ++batch) {
            const int slot = batch & 1;
            const int count = static_cast<int>(std::min<std::size_t>(kBatchPages, pages.size() - begin));
            // The slot's previous upload (batch - 2) has been read.
            if (batch >= 2) check(cudaEventSynchronize(slots_[slot].done), "KV unpack");
            std::array<unsigned long long, kBatchPages> offsets{};
            std::size_t offset = 0;
            for (int j = 0; j < count; ++j) {
                const auto* blob = blobs[begin + j];
                const std::size_t bytes = sizes[begin + j];
                validate(blob, bytes, pages[begin + j].rows, streams);
                std::memcpy(slots_[slot].host + offset, blob, bytes);
                offsets[j] = offset;
                offset = (offset + bytes + 15) / 16 * 16;
            }
            upload_jobs(pages, begin, count, banks, offsets.data(), destination, slot);
            check(cudaMemcpyAsync(blobs_, slots_[slot].host, offset, cudaMemcpyHostToDevice, stream_), "KV unpack upload");
            check(cudaEventRecord(slots_[slot].done, stream_), "KV unpack");
            fetch_kernel<<<dim3(banks * 2, count), 256, 0, stream_>>>(jobs_device_, banks, Planes{}, nullptr,
                codes_, meta_);
            check(cudaGetLastError(), "KV unpack fetch");
            code_kernel<<<dim3(streams, count), 256, 0, stream_>>>(jobs_device_, banks, l0_oscar::rotation_view(), 1,
                rows_, codes_, meta_, blobs_, error_);
            check(cudaGetLastError(), "KV unpack decode");
            if (destination) {
                store_kernel<<<dim3(banks * 2, count), 256, 0, stream_>>>(jobs_device_, banks, *destination, rows_, error_);
                check(cudaGetLastError(), "KV unpack store");
            } else {
                for (int j = 0; j < count; ++j)
                    for (int b = 0; b < banks; ++b)
                        for (int plane = 0; plane < 2; ++plane)
                            check(cudaMemcpyAsync(host_rows(begin + j, b, plane),
                                rows_ + ((static_cast<std::size_t>(j) * banks + b) * 2 + plane) * kSlotRowWords,
                                static_cast<std::size_t>(pages[begin + j].rows) * kRowWords * 2,
                                cudaMemcpyDeviceToHost, stream_), "KV unpack download");
            }
        }
        check(cudaMemcpyAsync(error_host_, error_, sizeof(unsigned), cudaMemcpyDeviceToHost, stream_), "KV unpack error");
        check(cudaStreamSynchronize(stream_), "KV unpack");
        if (*error_host_) throw std::runtime_error("KV pack checksum mismatch: packed page is not lossless");
    }

    std::uint64_t packed_bytes() const noexcept { return packed_bytes_; }

    static void check(cudaError_t error, const char* what) {
        if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
    // Device address of page-locked host memory (or a device pointer as is).
    template <class T> static const T* device_address(const T* pointer) {
        if (!pointer) return nullptr;
        cudaPointerAttributes attributes{};
        check(cudaPointerGetAttributes(&attributes, pointer), "KV pack pointer");
        if (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged) return pointer;
        if (attributes.type != cudaMemoryTypeHost) throw std::runtime_error("KV pack source is not device-addressable");
        if (attributes.devicePointer) {
            const auto* base = static_cast<const std::uint8_t*>(attributes.hostPointer);
            return reinterpret_cast<const T*>(static_cast<const std::uint8_t*>(attributes.devicePointer) +
                (reinterpret_cast<const std::uint8_t*>(pointer) - base));
        }
        void* device = nullptr;
        check(cudaHostGetDevicePointer(&device, const_cast<T*>(pointer), 0), "KV pack host pointer");
        return static_cast<const T*>(device);
    }

private:
    // Host side of one pipeline slot: page-locked blob memory (device-mapped:
    // pack writes its blobs there), blob sizes, job table and its last use.
    struct Slot {
        std::uint8_t* host = nullptr;
        std::uint8_t* host_device = nullptr;
        unsigned long long* sizes = nullptr;
        unsigned long long* sizes_device = nullptr;
        Job* jobs = nullptr;
        cudaEvent_t done = nullptr;
    };
    static void validate(const std::uint8_t* blob, std::size_t bytes, int rows, int streams) {
        Header header{};
        if (bytes < sizeof(Header)) throw std::runtime_error("KV pack blob truncated");
        std::memcpy(&header, blob, sizeof(header));
        if (header.magic != kMagic || header.rows != static_cast<unsigned>(rows) ||
            header.streams != static_cast<unsigned>(streams) || header.bytes != bytes || bytes > kBlobCap)
            throw std::runtime_error("KV pack blob header");
        const auto* entries = reinterpret_cast<const Entry*>(blob + sizeof(Header));
        for (int s = 0; s < streams; ++s) {
            Entry e{};
            std::memcpy(&e, entries + s, sizeof(e));
            const std::size_t need = e.mode ? static_cast<std::size_t>(header.rows) * kDim * 2
                                            : static_cast<std::size_t>(e.main_words) * 4 + e.escapes * 2ull;
            if (e.mode > 1 || e.offset % 8 || e.offset < payload_start(streams) || e.offset + need > bytes)
                throw std::runtime_error("KV pack blob entry");
        }
    }
    Packer() = default;
    void ensure() {
        if (stream_) return;
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "KV pack stream");
        check(cudaMalloc(&scratch_, static_cast<std::size_t>(kBatchPages) * kMaxStreams * kScratchStride), "KV pack scratch");
        check(cudaMalloc(&blobs_, static_cast<std::size_t>(kBatchPages) * kBlobCap), "KV pack blobs");
        check(cudaMalloc(&error_, sizeof(unsigned)), "KV pack error");
        check(cudaMalloc(&jobs_device_, kBatchPages * sizeof(Job)), "KV pack jobs");
        const std::size_t slots = static_cast<std::size_t>(kBatchPages) * kBanks * 2;
        check(cudaMalloc(&rows_, slots * kSlotRowWords * sizeof(std::uint16_t)), "KV pack staging");
        check(cudaMalloc(&codes_, slots * kSlotCodeBytes), "KV pack staging");
        check(cudaMalloc(&meta_, slots * kSlotMetaWords * sizeof(std::uint16_t)), "KV pack staging");
        check(cudaMallocHost(&error_host_, sizeof(unsigned)), "KV pack host");
        for (auto& slot : slots_) {
            check(cudaHostAlloc(&slot.host, static_cast<std::size_t>(kBatchPages) * (kBlobCap + 16),
                cudaHostAllocMapped | cudaHostAllocPortable), "KV pack host");
            check(cudaHostAlloc(&slot.sizes, kBatchPages * sizeof(unsigned long long),
                cudaHostAllocMapped | cudaHostAllocPortable), "KV pack host");
            check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&slot.host_device), slot.host, 0), "KV pack host");
            check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&slot.sizes_device), slot.sizes, 0), "KV pack host");
            check(cudaMallocHost(&slot.jobs, kBatchPages * sizeof(Job)), "KV pack host");
            check(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming), "KV pack event");
        }
    }
    static void require_aligned(const void* pointer) {
        if (reinterpret_cast<std::uintptr_t>(pointer) % 16)
            throw std::invalid_argument("KV pack source is not 16-byte aligned");
    }
    void upload_jobs(const std::vector<Page>& pages, std::size_t begin, int count, int banks,
                     const unsigned long long* blob_offsets, const Planes* planes, int slot) {
        Job* jobs_host = slots_[slot].jobs;
        for (int j = 0; j < count; ++j) {
            const Page& page = pages[begin + j];
            Job job{};
            for (int b = 0; b < banks; ++b) {
                job.codes[b][0] = device_address(page.k_codes[b]);
                job.codes[b][1] = device_address(page.v_codes[b]);
                job.meta[b][0] = device_address(page.k_meta[b]);
                job.meta[b][1] = device_address(page.v_meta[b]);
                if (!job.codes[b][0] || !job.codes[b][1] || !job.meta[b][0] || !job.meta[b][1])
                    throw std::invalid_argument("KV pack page without codes");
                require_aligned(job.codes[b][0]); require_aligned(job.codes[b][1]);
                require_aligned(job.meta[b][0]); require_aligned(job.meta[b][1]);
                if (planes) { require_aligned(planes->k[b]); require_aligned(planes->v[b]); }
            }
            if (page.rows <= 0 || page.rows > kPageRows) throw std::invalid_argument("KV pack page rows");
            job.row = page.row;
            job.rows = page.rows;
            job.blob = blob_offsets ? blob_offsets[j] : 0;
            jobs_host[j] = job;
        }
        check(cudaMemcpyAsync(jobs_device_, jobs_host, count * sizeof(Job), cudaMemcpyHostToDevice, stream_),
            "KV pack jobs");
    }

    std::mutex mutex_;
    cudaStream_t stream_ = nullptr;
    std::uint8_t* scratch_ = nullptr;
    std::uint8_t* blobs_ = nullptr;
    unsigned* error_ = nullptr;
    Job* jobs_device_ = nullptr;
    std::uint16_t* rows_ = nullptr;
    std::uint8_t* codes_ = nullptr;
    std::uint16_t* meta_ = nullptr;
    unsigned* error_host_ = nullptr;
    std::array<Slot, 2> slots_{};
    std::uint64_t packed_bytes_ = 0;
};

}  // namespace voidinfer::exl3::l0_kv_pack
