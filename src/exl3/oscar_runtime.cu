#include "exl3/oscar_runtime.h"
#include <cctype>
#include <cstdlib>
#include <map>
// E4C1 native EXL3 OSCAR runtime adapter implementation.
#include "ops/softmax_attention/oscar_mixed/launch.h"

#include <cuda_fp16.h>
#include <iostream>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace ninfer::exl3 {
namespace {

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

void* device_alloc(std::size_t bytes, const char* label) {
    if (bytes == 0) throw std::invalid_argument("E4C1 OSCAR zero allocation");
    void* ptr = nullptr;
    cuda_check(cudaMalloc(&ptr, bytes), label);
    cuda_check(cudaMemset(ptr, 0, bytes), label);
    return ptr;
}

// --- Compact public-domain SHA-256 (self-contained loader, no old-target dep). ---
struct Sha256 {
    std::uint32_t h[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                          0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::uint64_t total = 0;
    std::uint8_t block[64] = {};
    std::size_t used = 0;
    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void compress() {
        static const std::uint32_t k[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
            0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
            0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
            0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
            0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
            0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<std::uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const std::uint8_t* data, std::size_t size) {
        total += size;
        while (size > 0) {
            const std::size_t take = std::min(size, 64 - used);
            std::memcpy(block + used, data, take);
            used += take; data += take; size -= take;
            if (used == 64) { compress(); used = 0; }
        }
    }
    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bits = total * 8ULL;
        std::uint8_t pad = 0x80;
        update(&pad, 1);
        std::uint8_t zero = 0;
        while (used != 56) update(&zero, 1);
        std::uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
        update(len, 8);
        std::array<std::uint8_t, 32> out{};
        for (int i = 0; i < 8; ++i) {
            out[i * 4] = static_cast<std::uint8_t>(h[i] >> 24);
            out[i * 4 + 1] = static_cast<std::uint8_t>(h[i] >> 16);
            out[i * 4 + 2] = static_cast<std::uint8_t>(h[i] >> 8);
            out[i * 4 + 3] = static_cast<std::uint8_t>(h[i]);
        }
        return out;
    }
};

std::string sha256_hex_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("E4C1 OSCAR cannot open file for hashing");
    Sha256 sha;
    std::vector<std::uint8_t> buffer(1 << 20);
    while (input) {
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = input.gcount();
        if (got > 0) sha.update(buffer.data(), static_cast<std::size_t>(got));
    }
    static const char* digits = "0123456789abcdef";
    const auto digest = sha.finish();
    std::string out;
    for (std::uint8_t byte : digest) {
        out.push_back(digits[byte >> 4]);
        out.push_back(digits[byte & 15]);
    }
    return out;
}

std::string lower_str(std::string value) {
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

std::map<std::string, std::string> read_kv_manifest(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("E4C1 OSCAR cannot open manifest: " + path.string());
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == 13) line.pop_back();
        if (line.empty()) continue;
        const std::size_t eq = line.find(61);
        if (eq == std::string::npos) throw std::runtime_error("E4C1 OSCAR manifest malformed");
        const std::string key = line.substr(0, eq);
        if (!fields.emplace(key, line.substr(eq + 1)).second) {
            throw std::runtime_error("E4C1 OSCAR manifest duplicates " + key);
        }
    }
    return fields;
}

void require_field(const std::map<std::string, std::string>& fields, const std::string& key,
                   const std::string& expected) {
    const auto it = fields.find(key);
    if (it == fields.end() || it->second != expected) {
        throw std::runtime_error("E4C1 OSCAR manifest mismatch for " + key);
    }
}

// Rotation layout on device: FWD[d][i] = M[i][d] (contiguous per output d),
// INV[d][i] = M[d][i] (contiguous per output d for the transpose multiply).
void split_rotation_layouts(const std::vector<float>& bank, std::vector<float>& fwd,
                            std::vector<float>& inv) {
    constexpr int D = 256;
    fwd.resize(static_cast<std::size_t>(D) * D);
    inv.resize(static_cast<std::size_t>(D) * D);
    for (int d = 0; d < D; ++d) {
        for (int i = 0; i < D; ++i) {
            fwd[static_cast<std::size_t>(d) * D + i] = bank[static_cast<std::size_t>(i) * D + d];
            inv[static_cast<std::size_t>(d) * D + i] = bank[static_cast<std::size_t>(d) * D + i];
        }
    }
}

constexpr int kD = 256;

// F16 [rows, heads, D] -> FP32 rotated [rows, heads, D] (D fastest, tokens-major).
__global__ void rotate_qkv_kernel(const std::uint16_t* input_f16, const float* rot_fwd,
                                  float* output_dht, int rows, int heads) {
    const int head = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int d = static_cast<int>(threadIdx.x);
    if (head >= heads || row >= rows || d >= kD) return;
    const std::uint16_t* in_row =
        input_f16 + (static_cast<std::size_t>(row) * heads + head) * kD;
    float sum = 0.0F;
    for (int i = 0; i < kD; ++i) {
        const float v = __half2float(__ushort_as_half(in_row[i]));
        sum = fmaf(v, rot_fwd[static_cast<std::size_t>(d) * kD + i], sum);
    }
    output_dht[(static_cast<std::size_t>(row) * heads + head) * kD + d] = sum;
}

// K and V use independent CTAs with identical exact arithmetic. A single grid
// removes one launch while retaining each CTA's represented input, rotation
// bank, 256-term FP32 FMA chain and disjoint staging destination.
__global__ void rotate_kv_fused_kernel(const std::uint16_t* k_f16,
    const std::uint16_t* v_f16,const float* r_k,const float* r_v,
    float* output_k,float* output_v,int rows,int heads) {
    const int kind_head=static_cast<int>(blockIdx.x);
    const bool value=kind_head>=heads;
    const int head=value?kind_head-heads:kind_head;
    const int row=static_cast<int>(blockIdx.y);
    const int d=static_cast<int>(threadIdx.x);
    if(head>=heads||row>=rows||d>=kD) return;
    const std::uint16_t* input=value?v_f16:k_f16;
    const float* rotation=value?r_v:r_k;
    float* output=value?output_v:output_k;
    const std::uint16_t* in_row=input+
        (static_cast<std::size_t>(row)*heads+head)*kD;
    float sum=0.0F;
    for(int i=0;i<kD;++i) {
        const float represented=__half2float(__ushort_as_half(in_row[i]));
        sum=fmaf(represented,rotation[static_cast<std::size_t>(d)*kD+i],sum);
    }
    output[(static_cast<std::size_t>(row)*heads+head)*kD+d]=sum;
}

// Fused kernels are query-major: Q [heads, D] -> rotated [heads, D].
__global__ void rotate_q_decode_kernel(const std::uint16_t* q_f16, const float* rot_fwd,
                                            float* q_rot_hd, int qheads) {
    const int head = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(threadIdx.x);
    if (head >= qheads || d >= kD) return;
    const std::uint16_t* in_row = q_f16 + static_cast<std::size_t>(head) * kD;
    float sum = 0.0F;
    for (int j = 0; j < kD; ++j) {
        const float v = __half2float(__ushort_as_half(in_row[j]));
        sum = fmaf(v, rot_fwd[static_cast<std::size_t>(d) * kD + j], sum);
    }
    q_rot_hd[static_cast<std::size_t>(head) * kD + d] = sum;
}

// FP32 AV [qheads, D] (single query) x R_V^T -> F16 [qheads, D].
__global__ void recover_kernel(const float* av_dh, const float* rot_inv,
                               std::uint16_t* output_f16, int qheads) {
    const int head = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(threadIdx.x);
    if (head >= qheads || d >= kD) return;
    float sum = 0.0F;
    for (int i = 0; i < kD; ++i) {
        sum = fmaf(av_dh[static_cast<std::size_t>(head) * kD + i],
                   rot_inv[static_cast<std::size_t>(d) * kD + i], sum);
    }
    output_f16[static_cast<std::size_t>(head) * kD + d] = __half_as_ushort(__float2half(sum));
}

// FP32 AV [rows, qheads, D] x R_V^T -> F16 [rows, qheads, D].
__global__ void recover_rows_kernel(const float* av_thd, const float* rot_inv,
                                        std::uint16_t* output_f16, int rows, int qheads) {
    const int flat = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(threadIdx.x);
    const int row = flat / 24;
    const int head = flat % 24;
    if (row >= rows || head >= qheads || d >= kD) return;
    const float* av = av_thd + (static_cast<std::size_t>(row) * qheads + head) * kD;
    float sum = 0.0F;
    for (int j = 0; j < kD; ++j) {
        sum = fmaf(av[j], rot_inv[static_cast<std::size_t>(d) * kD + j], sum);
    }
    output_f16[(static_cast<std::size_t>(row) * qheads + head) * kD + d] =
        __half_as_ushort(__float2half(sum));
}

// F16 Q [rows, heads, D] -> FP32 rotated [rows, heads, D] (query-major).
__global__ void rotate_q_batch_kernel(const std::uint16_t* q_f16, const float* rot_fwd,
                                          float* q_rot_thd, int rows, int qheads) {
    const int flat = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(threadIdx.x);
    const int row = flat / 24;
    const int head = flat % 24;
    if (row >= rows || head >= qheads || d >= kD) return;
    const std::uint16_t* in_row =
        q_f16 + (static_cast<std::size_t>(row) * qheads + head) * kD;
    float sum = 0.0F;
    for (int j = 0; j < kD; ++j) {
        const float v = __half2float(__ushort_as_half(in_row[j]));
        sum = fmaf(v, rot_fwd[static_cast<std::size_t>(d) * kD + j], sum);
    }
    q_rot_thd[(static_cast<std::size_t>(row) * qheads + head) * kD + d] = sum;
}

__global__ void rotate_q_coalesced_kernel(const std::uint16_t* q_f16, const float* rot_fwd,
                                          float* q_rot_thd, int rows, int qheads) {
    const int flat = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(threadIdx.x);
    const int row = flat / 24;
    const int head = flat % 24;
    if (row >= rows || head >= qheads || d >= kD) return;
    const std::uint16_t* in_row =
        q_f16 + (static_cast<std::size_t>(row) * qheads + head) * kD;
    float sum = 0.0F;
    for (int j = 0; j < kD; ++j) {
        const float v = __half2float(__ushort_as_half(in_row[j]));
        sum = fmaf(v, rot_fwd[static_cast<std::size_t>(j) * kD + d], sum);
    }
    q_rot_thd[(static_cast<std::size_t>(row) * qheads + head) * kD + d] = sum;
}

__global__ void recover_coalesced_kernel(const float* av_thd, const float* rot_inv,
                                        std::uint16_t* output_f16, int rows, int qheads) {
    const int flat = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(threadIdx.x);
    const int row = flat / 24;
    const int head = flat % 24;
    if (row >= rows || head >= qheads || d >= kD) return;
    const float* av = av_thd + (static_cast<std::size_t>(row) * qheads + head) * kD;
    float sum = 0.0F;
    for (int j = 0; j < kD; ++j) {
        sum = fmaf(av[j], rot_inv[static_cast<std::size_t>(j) * kD + d], sum);
    }
    output_f16[(static_cast<std::size_t>(row) * qheads + head) * kD + d] =
        __half_as_ushort(__float2half(sum));
}

std::uint32_t recent_begin_for(std::uint32_t context_tokens) {
    if (context_tokens <= 64U) return context_tokens;
    return std::max(64U, context_tokens > 256U ? context_tokens - 256U : 0U);
}

}  // namespace

constexpr char kExpectedIdentity[] = "qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1";
constexpr char kExpectedManifestSha[] = "4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560";
constexpr char kExpectedLayers[] = "3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63";
constexpr char kExpectedKSha[] = "d50cd5367c886a2ac21b4336de4bc8cf53fda1e5a28bb00500879bccf8295059";
constexpr char kExpectedVSha[] = "fe2f33027175b43eb2c60ee80fb95b70f63e5d96a94742c3c9061496f0c8affe";


Exl3OscarRotations exl3_oscar_load_rotations(const std::filesystem::path& asset_dir,
                                             const std::filesystem::path& compat_manifest) {
    const auto fields = read_kv_manifest(asset_dir / "runtime" / "runtime_manifest.txt");
    require_field(fields, "asset_identity", kExpectedIdentity);
    require_field(fields, "asset_manifest_sha256", kExpectedManifestSha);
    require_field(fields, "full_attention_layers", kExpectedLayers);
    require_field(fields, "q_heads", "24");
    require_field(fields, "kv_heads", "4");
    require_field(fields, "head_dim", "256");
    require_field(fields, "k_layers", "16");
    require_field(fields, "v_layers", "16");
    require_field(fields, "dtype", "fp32");
    const auto k_path = asset_dir / "runtime" / "k_rotation_fp32.bin";
    const auto v_path = asset_dir / "runtime" / "v_rotation_fp32.bin";
    if (lower_str(sha256_hex_file(k_path)) != kExpectedKSha) {
        throw std::runtime_error("E4C1 OSCAR K rotation hash mismatch");
    }
    if (lower_str(sha256_hex_file(v_path)) != kExpectedVSha) {
        throw std::runtime_error("E4C1 OSCAR V rotation hash mismatch");
    }
    // Compatibility attestation binds this immutable asset to the EXL3 target.
    std::ifstream compat(compat_manifest);
    if (!compat) throw std::runtime_error("E4C1 OSCAR compatibility manifest missing");
    std::ostringstream attestation;
    attestation << compat.rdbuf();
    const std::string text = attestation.str();
    const auto needs = [](const std::string& hay, const char* needle, const char* label) {
        if (hay.find(needle) == std::string::npos) {
            throw std::runtime_error(std::string("E4C1 OSCAR compatibility manifest lacks ") + label);
        }
    };
    needs(text, kExpectedIdentity, "asset identity");
    needs(text, kExpectedManifestSha, "manifest digest");
    needs(text, "60d005a257b39ecb25e4ba23c1dd29df877d5c69", "EXL3 model commit");
    needs(text, "REUSE_EXISTING_ASSET", "reuse decision");
    Exl3OscarRotations rotations;
    rotations.asset_identity = kExpectedIdentity;
    rotations.manifest_sha256 = kExpectedManifestSha;
    constexpr std::size_t kBank = 256ULL * 256ULL;
    auto read_bank_file = [&](const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("E4C1 OSCAR cannot open rotation bank");
        std::vector<float> all(16ULL * kBank);
        input.read(reinterpret_cast<char*>(all.data()), static_cast<std::streamsize>(all.size() * 4));
        if (input.gcount() != static_cast<std::streamsize>(all.size() * 4)) {
            throw std::runtime_error("E4C1 OSCAR rotation bank truncated");
        }
        for (float value : all) {
            if (!std::isfinite(value)) throw std::runtime_error("E4C1 OSCAR rotation non-finite");
        }
        return all;
    };
    const auto k_all = read_bank_file(k_path);
    const auto v_all = read_bank_file(v_path);
    for (int b = 0; b < 16; ++b) {
        rotations.r_k[b].assign(k_all.begin() + b * kBank, k_all.begin() + (b + 1) * kBank);
        rotations.r_v[b].assign(v_all.begin() + b * kBank, v_all.begin() + (b + 1) * kBank);
    }
    return rotations;
}
Exl3OscarLayerCache::Exl3OscarLayerCache(int model_layer, int bank_index, int max_context)
    : model_layer_(model_layer), max_context_(max_context) {
    (void)bank_index;
    if (bank_index < 0 || bank_index >= 16) throw std::invalid_argument("E4C1 OSCAR bad bank");
    if (max_context <= 0) throw std::invalid_argument("E4C1 OSCAR bad max_context");
    const auto mc = static_cast<std::size_t>(max_context);
    prefix_k_ = device_alloc(64ULL * 4 * 256 * 2, "E4C1 OSCAR prefix K");
    prefix_v_ = device_alloc(64ULL * 4 * 256 * 2, "E4C1 OSCAR prefix V");
    hist_k_ = device_alloc(mc * 4 * 64, "E4C1 OSCAR history K");
    hist_v_ = device_alloc(mc * 4 * 64, "E4C1 OSCAR history V");
    hist_k_meta_ = device_alloc(mc * 4 * 4 * 4, "E4C1 OSCAR history K meta");
    hist_v_meta_ = device_alloc(mc * 4 * 4 * 4, "E4C1 OSCAR history V meta");
    recent_k_ = device_alloc(256ULL * 4 * 256 * 2, "E4C1 OSCAR recent K");
    recent_v_ = device_alloc(256ULL * 4 * 256 * 2, "E4C1 OSCAR recent V");
    workspace_ = device_alloc(
        ninfer::ops::detail::kOscarMixedFusedDecodeWorkspaceBytes, "E4C1 OSCAR workspace");
}

Exl3OscarLayerCache::~Exl3OscarLayerCache() {
    cudaFree(prefix_k_);
    cudaFree(prefix_v_);
    cudaFree(hist_k_);
    cudaFree(hist_v_);
    cudaFree(hist_k_meta_);
    cudaFree(hist_v_meta_);
    cudaFree(recent_k_);
    cudaFree(recent_v_);
    cudaFree(workspace_);
    cudaFree(dev_r_k_);
    cudaFree(dev_r_v_);
    cudaFree(dev_r_inv_);
    cudaFree(dev_plan_);
    cudaFree(dev_counts_);
}

void Exl3OscarLayerCache::prefix_extent(std::int32_t& prefix, std::int32_t& historical,
                                        std::int32_t& recent,
                                        std::int32_t& recent_ring_head) const {
    const std::uint32_t begin = recent_begin_for(context_);
    prefix = static_cast<std::int32_t>(std::min<std::uint32_t>(context_, 64U));
    historical = static_cast<std::int32_t>(begin) - prefix;
    recent = static_cast<std::int32_t>(context_ - begin);
    recent_ring_head = static_cast<std::int32_t>(recent_head_);
}

std::uint32_t Exl3OscarLayerCache::append(const float* rotated_k, const float* rotated_v,
                                 int token_count, std::uint32_t logical_start,
                                 cudaStream_t stream) {
    if (!rotated_k || !rotated_v || token_count <= 0) {
        throw std::invalid_argument("E4C1 OSCAR append arguments invalid");
    }
    if (logical_start != context_) {
        throw std::invalid_argument("E4C1 OSCAR append is not the next logical range");
    }
    if (static_cast<std::uint32_t>(token_count) > static_cast<std::uint32_t>(max_context_) - context_) {
        throw std::out_of_range("E4C1 OSCAR append exceeds cache capacity");
    }
    const std::uint32_t old_context = context_;
    const std::uint32_t final_context = old_context + static_cast<std::uint32_t>(token_count);
    const std::uint32_t old_begin = recent_begin_for(old_context);
    const std::uint32_t final_begin = recent_begin_for(final_context);
    const std::uint32_t final_head = (recent_head_ + final_begin - old_begin) & 255U;
    const std::uint32_t old_aging_end = std::min(final_begin, old_context);
    const std::uint32_t old_aging = old_aging_end > old_begin ? old_aging_end - old_begin : 0U;
    const std::uint32_t append_end = logical_start + static_cast<std::uint32_t>(token_count);
    const std::uint32_t new_hist_begin = std::max(logical_start, 64U);
    const std::uint32_t new_hist_end = std::min(append_end, final_begin);
    const std::uint32_t new_hist = new_hist_end > new_hist_begin ? new_hist_end - new_hist_begin : 0U;
    ninfer::ops::detail::OscarInt2G128ResidentCacheView view{};
    view.prefix_k_bf16 = static_cast<std::uint16_t*>(prefix_k_);
    view.prefix_v_bf16 = static_cast<std::uint16_t*>(prefix_v_);
    view.historical_k_packed = static_cast<std::uint8_t*>(hist_k_);
    view.historical_v_packed = static_cast<std::uint8_t*>(hist_v_);
    view.historical_k_metadata = static_cast<float*>(hist_k_meta_);
    view.historical_v_metadata = static_cast<float*>(hist_v_meta_);
    view.recent_k_bf16 = static_cast<std::uint16_t*>(recent_k_);
    view.recent_v_bf16 = static_cast<std::uint16_t*>(recent_v_);
    view.max_context = static_cast<std::int32_t>(max_context_);
    // Encode aged rows BEFORE publishing new recent rows sharing ring slots.
    if (new_hist != 0 || old_aging != 0) {
        ninfer::ops::detail::oscar_int2_g128_cache_encode_launch(
            rotated_k, rotated_v, token_count, logical_start, old_context, old_begin,
            recent_head_, final_begin, new_hist, old_aging, view, stream);
    }
    ninfer::ops::detail::oscar_int2_g128_cache_write_bf16_launch(
        rotated_k, rotated_v, token_count, logical_start, final_begin, final_head, view, stream);
    context_ = final_context;
    recent_head_ = final_head;
    return new_hist + old_aging;
}

Exl3OscarLayerCache::View Exl3OscarLayerCache::view() const noexcept {
    View v{};
    v.prefix_k = static_cast<std::uint16_t*>(prefix_k_);
    v.prefix_v = static_cast<std::uint16_t*>(prefix_v_);
    v.hist_k = static_cast<std::uint8_t*>(hist_k_);
    v.hist_v = static_cast<std::uint8_t*>(hist_v_);
    v.hist_k_meta = static_cast<float*>(hist_k_meta_);
    v.hist_v_meta = static_cast<float*>(hist_v_meta_);
    v.recent_k = static_cast<std::uint16_t*>(recent_k_);
    v.recent_v = static_cast<std::uint16_t*>(recent_v_);
    v.workspace = static_cast<float*>(workspace_);
    v.r_k = static_cast<const float*>(dev_r_k_);
    v.r_v = static_cast<const float*>(dev_r_v_);
    v.r_v_inv = static_cast<const float*>(dev_r_inv_);
    return v;
}

void Exl3OscarLayerCache::reset() noexcept {
    context_ = 0;
    recent_head_ = 0;
}

std::uint64_t Exl3OscarLayerCache::cache_bytes() const noexcept {
    const auto mc = static_cast<std::uint64_t>(max_context_);
    return 64ULL * 4 * 256 * 2 * 2 + mc * 4 * 64 * 2 + mc * 4 * 4 * 4 * 2 +
           256ULL * 4 * 256 * 2 * 2 + 2ULL * 256 * 256 * 4;
}

std::uint64_t Exl3OscarLayerCache::workspace_bytes() const noexcept {
    return ninfer::ops::detail::kOscarMixedFusedDecodeWorkspaceBytes;
}

void Exl3OscarLayerCache::save_checkpoint(void* destination, cudaStream_t stream) const {
    auto* out = static_cast<std::byte*>(destination);
    cuda_check(cudaMemcpyAsync(out, prefix_k_, kCheckpointPrefixBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "save E4C1 OSCAR prefix K checkpoint");
    out += kCheckpointPrefixBytes;
    cuda_check(cudaMemcpyAsync(out, prefix_v_, kCheckpointPrefixBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "save E4C1 OSCAR prefix V checkpoint");
    out += kCheckpointPrefixBytes;
    cuda_check(cudaMemcpyAsync(out, recent_k_, kCheckpointRecentBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "save E4C1 OSCAR recent K checkpoint");
    out += kCheckpointRecentBytes;
    cuda_check(cudaMemcpyAsync(out, recent_v_, kCheckpointRecentBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "save E4C1 OSCAR recent V checkpoint");
}

void Exl3OscarLayerCache::restore_checkpoint(const void* source, std::uint32_t context,
                                              std::uint32_t recent_head,
                                              cudaStream_t stream) {
    if (context > static_cast<std::uint32_t>(max_context_) ||
        recent_head != (recent_begin_for(context) & 255U)) {
        throw std::invalid_argument("E4C1 OSCAR checkpoint cursor is invalid");
    }
    const auto* in = static_cast<const std::byte*>(source);
    cuda_check(cudaMemcpyAsync(prefix_k_, in, kCheckpointPrefixBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "restore E4C1 OSCAR prefix K checkpoint");
    in += kCheckpointPrefixBytes;
    cuda_check(cudaMemcpyAsync(prefix_v_, in, kCheckpointPrefixBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "restore E4C1 OSCAR prefix V checkpoint");
    in += kCheckpointPrefixBytes;
    cuda_check(cudaMemcpyAsync(recent_k_, in, kCheckpointRecentBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "restore E4C1 OSCAR recent K checkpoint");
    in += kCheckpointRecentBytes;
    cuda_check(cudaMemcpyAsync(recent_v_, in, kCheckpointRecentBytes,
                               cudaMemcpyDeviceToDevice, stream),
               "restore E4C1 OSCAR recent V checkpoint");
    if (dev_plan_) {
        cuda_check(cudaMemsetAsync(dev_plan_, 0, 16 * sizeof(std::uint32_t), stream),
                   "clear restored E4C1 OSCAR device plan");
    }
    if (dev_counts_) {
        cuda_check(cudaMemsetAsync(dev_counts_, 0, 5 * sizeof(int), stream),
                   "clear restored E4C1 OSCAR device counts");
    }
    context_ = context;
    recent_head_ = recent_head;
}
Exl3OscarContext::Exl3OscarContext(const Exl3OscarRotations& rotations, int max_context,
                                   Exl3OscarTelemetry* telemetry)
    : max_context_(max_context), telemetry_(telemetry) {
    if (!telemetry) throw std::invalid_argument("E4C1 OSCAR telemetry is null");
    if (max_context <= 0) throw std::invalid_argument("E4C1 OSCAR bad max_context");
    const char* parallel_scores=std::getenv("NINFER_OSCAR_PREFILL_PARALLEL_SCORES");
    prefill_parallel_scores_=parallel_scores == nullptr || std::strcmp(parallel_scores,"1")==0;
    const char* parallel_merge=std::getenv("NINFER_OSCAR_PREFILL_PARALLEL_MERGE");
    prefill_parallel_merge_=parallel_merge == nullptr || std::strcmp(parallel_merge,"1")==0;
    const char* batch_rotations = std::getenv("NINFER_EXL3_PREFILL_BATCH_ROTATIONS");
    batch_rotations_ = batch_rotations != nullptr && std::string(batch_rotations) == "1";
    const char* coalesced=std::getenv("NINFER_OSCAR_PREFILL_COALESCED_ROTATIONS");
    prefill_coalesced_rotations_=batch_rotations_ &&
        (coalesced == nullptr || std::strcmp(coalesced,"1")==0);
    const char* decode_coalesced = std::getenv("NINFER_OSCAR_DECODE_COALESCED_ROTATIONS");
    decode_coalesced_rotations_ = prefill_coalesced_rotations_ &&
        decode_coalesced && std::strcmp(decode_coalesced, "1") == 0;
    const char* fused_kv=std::getenv("NINFER_OSCAR_DECODE_FUSED_KV_ROTATIONS");
    if(fused_kv&&std::strcmp(fused_kv,"0")!=0&&std::strcmp(fused_kv,"1")!=0)
        throw std::invalid_argument("NINFER_OSCAR_DECODE_FUSED_KV_ROTATIONS must be 0 or 1");
    decode_fused_kv_rotations_=fused_kv&&std::strcmp(fused_kv,"1")==0;
    const char* continuation_cohort =
        std::getenv("NINFER_OSCAR_CONTINUATION_COHORT_B8");
    if (continuation_cohort && std::strcmp(continuation_cohort,"0") != 0 &&
        std::strcmp(continuation_cohort,"1") != 0)
        throw std::invalid_argument(
            "NINFER_OSCAR_CONTINUATION_COHORT_B8 must be 0 or 1");
    continuation_cohort_b8_ = continuation_cohort &&
        std::strcmp(continuation_cohort,"1") == 0;
    const char* query_parallel = std::getenv("NINFER_OSCAR_PREFILL_QUERY_PARALLEL");
    prefill_query_parallel_ = query_parallel == nullptr
        ? (batch_rotations_ && prefill_parallel_merge_ && prefill_parallel_scores_)
        : std::strcmp(query_parallel,"1") == 0;
    const char* packed_byte4 = std::getenv("NINFER_OSCAR_PREFILL_PACKED_BYTE4");
    prefill_packed_byte4_ = packed_byte4 == nullptr
        ? prefill_query_parallel_ : std::strcmp(packed_byte4,"1") == 0;
    if (prefill_packed_byte4_ && !prefill_query_parallel_)
        throw std::invalid_argument("OSCAR packed-byte4 prefill requires query-parallel");
    if (prefill_query_parallel_ &&
        (!batch_rotations_ || !prefill_parallel_scores_ || !prefill_parallel_merge_)) {
        throw std::invalid_argument("OSCAR query-parallel prefill requires batch rotations and parallel scores/merge");
    }
    if (prefill_query_parallel_ || continuation_cohort_b8_) {
        constexpr std::size_t snapshot_bytes = 2ULL * 256 * 4 * 256 * sizeof(std::uint16_t);
        prefill_old_recent_ = std::shared_ptr<void>(
            device_alloc(snapshot_bytes,"OSCAR prefill old recent snapshot"),
            [](void* p){cudaFree(p);});
        prefill_query_workspace_ = std::shared_ptr<void>(
            device_alloc(ninfer::ops::detail::kOscarChronologicalWorkspaceBytes,
                         "OSCAR query cohort partials"), [](void* p){cudaFree(p);});
        telemetry_->workspace_bytes += snapshot_bytes + ninfer::ops::detail::kOscarChronologicalWorkspaceBytes;
    }
    const char* wide128 = std::getenv("NINFER_EXL3_PREFILL_WIDE128");
    if (batch_rotations_ && wide128 && std::strcmp(wide128, "1") == 0) batch_staging_rows_ = 128;
    const char* wide256 = std::getenv("NINFER_EXL3_PREFILL_WIDE256");
    if (batch_staging_rows_ == 128 && wide256 && std::strcmp(wide256, "1") == 0) batch_staging_rows_ = 256;
    const char* wide512=std::getenv("NINFER_EXL3_PREFILL_WIDE512");
    if(batch_staging_rows_==256 && wide512 && std::strcmp(wide512,"1")==0) batch_staging_rows_=512;
    const char* wide1024=std::getenv("NINFER_EXL3_PREFILL_WIDE1024");
    if(batch_staging_rows_==512 && wide1024 && std::strcmp(wide1024,"1")==0) batch_staging_rows_=1024;
    const auto mc = static_cast<std::size_t>(max_context);
    (void)mc;
    staging_rk_ = device_alloc(static_cast<std::size_t>(std::max(256,batch_staging_rows_)) * 4 * 256 * 4, "E4C1 OSCAR staging rotated K");
    staging_rv_ = device_alloc(static_cast<std::size_t>(std::max(256,batch_staging_rows_)) * 4 * 256 * 4, "E4C1 OSCAR staging rotated V");
    staging_rq_ = device_alloc(24ULL * 256 * 4, "E4C1 OSCAR staging rotated Q");
    staging_av_ = device_alloc(256ULL * 24 * 4, "E4C1 OSCAR staging AV");
    staging_bq_ = device_alloc(static_cast<std::size_t>(batch_staging_rows_) * 24 * 256 * 4, "E4C1 OSCAR staging batch Q");
    staging_bav_ = device_alloc(static_cast<std::size_t>(batch_staging_rows_) * 24 * 256 * 4, "E4C1 OSCAR staging batch AV");
    for (int b = 0; b < 16; ++b) {
        const int layer = kExl3OscarFullLayers[b];
        layers_[b] = std::make_unique<Exl3OscarLayerCache>(layer, b, max_context);
    }
    // Upload rotation layouts per layer (FWD for Q/K/V rotation, INV of R_V
    // for AV recovery). Synchronous here: creation is outside the hot loop.
    for (int b = 0; b < 16; ++b) {
        std::vector<float> fwd_k, inv_k, fwd_v, inv_v;
        split_rotation_layouts(rotations.r_k[b], fwd_k, inv_k);
        split_rotation_layouts(rotations.r_v[b], fwd_v, inv_v);
        void* dk_fwd = device_alloc(fwd_k.size() * 4, "E4C1 OSCAR device RK");
        void* dv_fwd = device_alloc(fwd_v.size() * 4, "E4C1 OSCAR device RV");
        void* dv_inv = device_alloc(inv_v.size() * 4, "E4C1 OSCAR device RVinv");
        cuda_check(cudaMemcpy(dk_fwd, fwd_k.data(), fwd_k.size() * 4, cudaMemcpyHostToDevice),
                   "E4C1 OSCAR upload RK");
        cuda_check(cudaMemcpy(dv_fwd, fwd_v.data(), fwd_v.size() * 4, cudaMemcpyHostToDevice),
                   "E4C1 OSCAR upload RV");
        cuda_check(cudaMemcpy(dv_inv, inv_v.data(), inv_v.size() * 4, cudaMemcpyHostToDevice),
                   "E4C1 OSCAR upload RVinv");
        layers_[b]->set_device_rotations(dk_fwd, dv_fwd, dv_inv);
        if(prefill_coalesced_rotations_) {
            // inv_k is the exact transpose of fwd_k: [input,output], so
            // adjacent output threads read adjacent matrix values.
            coalesced_r_k_[b]=std::shared_ptr<void>(
                device_alloc(inv_k.size()*sizeof(float),"OSCAR coalesced RK"),
                [](void* p){cudaFree(p);});
            cuda_check(cudaMemcpy(coalesced_r_k_[b].get(),inv_k.data(),inv_k.size()*sizeof(float),cudaMemcpyHostToDevice),"upload coalesced RK");
            telemetry_->workspace_bytes+=inv_k.size()*sizeof(float);
        }
    }
    for (int b = 0; b < 16; ++b) {
        telemetry_->resident_cache_bytes += layers_[b]->cache_bytes();
        telemetry_->workspace_bytes += layers_[b]->workspace_bytes();
    }
}

void Exl3OscarLayerCache::set_device_rotations(void* rk_fwd, void* rv_fwd,
                                                void* rv_inv) noexcept {
    cudaFree(dev_r_k_);
    cudaFree(dev_r_v_);
    cudaFree(dev_r_inv_);
    dev_r_k_ = rk_fwd;
    dev_r_v_ = rv_fwd;
    dev_r_inv_ = rv_inv;
}
std::unique_ptr<Exl3OscarContext> Exl3OscarContext::create(
    const Exl3OscarRotations& rotations, int max_context, Exl3OscarTelemetry* telemetry) {
    return std::unique_ptr<Exl3OscarContext>(new Exl3OscarContext(rotations, max_context, telemetry));
}

Exl3OscarContext::~Exl3OscarContext() {
    cudaFree(staging_rk_);
    cudaFree(staging_rv_);
    cudaFree(staging_rq_);
    cudaFree(staging_av_);
    cudaFree(staging_bq_);
    cudaFree(staging_bav_);
}

bool Exl3OscarContext::is_full_attention_layer(int model_layer) noexcept {
    for (int layer : kExl3OscarFullLayers) {
        if (layer == model_layer) return true;
    }
    return false;
}

int Exl3OscarContext::bank_index(int model_layer) {
    for (int b = 0; b < 16; ++b) {
        if (kExl3OscarFullLayers[b] == model_layer) return b;
    }
    throw std::invalid_argument("E4C1 OSCAR layer is not full attention");
}

void Exl3OscarContext::append_layer(int model_layer, const std::uint16_t* q_rope_f16,
                                    const std::uint16_t* k_rope_f16,
                                    const std::uint16_t* v_f16, int rows,
                                    std::uint32_t logical_start, cudaStream_t stream) {
    if (!q_rope_f16 || !k_rope_f16 || !v_f16 || rows <= 0) {
        throw std::invalid_argument("E4C1 OSCAR append taps invalid");
    }
    append_kv_layer(model_layer, k_rope_f16, v_f16, rows, logical_start, stream);
}

void Exl3OscarContext::append_kv_layer(int model_layer, const std::uint16_t* k_rope_f16,
                                     const std::uint16_t* v_f16, int rows,
                                     std::uint32_t logical_start, cudaStream_t stream) {
    if (!k_rope_f16 || !v_f16 || rows <= 0)
        throw std::invalid_argument("E4C1 OSCAR append KV invalid");
    if (!is_full_attention_layer(model_layer)) {
        throw std::invalid_argument("E4C1 OSCAR append to non-full-attention layer");
    }
    auto& cache = layers_[bank_index(model_layer)];
    const auto view = cache->view();
    if (!view.r_k || !view.r_v) throw std::logic_error("E4C1 OSCAR rotations not uploaded");
    int remaining = rows;
    std::uint32_t start = logical_start;
    std::size_t head_offset_kv = 0;
    while (remaining > 0) {
        const int chunk = std::min(remaining, chunk_rows_);
        const dim3 grid_kv(4, static_cast<unsigned>(chunk), 1u);
        if(decode_fused_kv_rotations_ && chunk == 1) {
            rotate_kv_fused_kernel<<<dim3(8,static_cast<unsigned>(chunk)),256,0,stream>>>(
                k_rope_f16+head_offset_kv*256,v_f16+head_offset_kv*256,
                view.r_k,view.r_v,static_cast<float*>(staging_rk_),
                static_cast<float*>(staging_rv_),chunk,4);
            cuda_check(cudaGetLastError(),"E4C1 OSCAR rotate fused KV chunk");
        } else {
            rotate_qkv_kernel<<<grid_kv, 256, 0, stream>>>(
                k_rope_f16 + head_offset_kv * 256, view.r_k,
                static_cast<float*>(staging_rk_), chunk, 4);
            cuda_check(cudaGetLastError(), "E4C1 OSCAR rotate K chunk");
            rotate_qkv_kernel<<<grid_kv, 256, 0, stream>>>(
                v_f16 + head_offset_kv * 256, view.r_v,
                static_cast<float*>(staging_rv_), chunk, 4);
            cuda_check(cudaGetLastError(), "E4C1 OSCAR rotate V chunk");
        }
        telemetry_->append_calls++;
        telemetry_->aging_events += cache->append(static_cast<const float*>(staging_rk_),
                                                     static_cast<const float*>(staging_rv_), chunk, start, stream);
        remaining -= chunk;
        start += static_cast<std::uint32_t>(chunk);
        head_offset_kv += static_cast<std::size_t>(chunk) * 4;
    }
    telemetry_->oscar_dispatches[model_layer]++;
}

void Exl3OscarContext::decode_rotated_layer(int model_layer, const float* rotated_q,
                                            float* av, std::uint32_t query_token,
                                            int split_count, cudaStream_t stream) {
    auto& cache = layers_[bank_index(model_layer)];
    const auto view = cache->view();
    std::int32_t prefix = 0, historical = 0, recent = 0, ring_head = 0;
    cache->prefix_extent(prefix, historical, recent, ring_head);
    const std::int32_t total = prefix + historical + recent;
    if (total <= 0 || query_token >= static_cast<std::uint32_t>(total)) {
        throw std::invalid_argument("E4C1 OSCAR decode query outside cache");
    }
    ninfer::ops::detail::OscarInt2G128ResidentCacheView resident{};
    resident.prefix_k_bf16 = view.prefix_k;
    resident.prefix_v_bf16 = view.prefix_v;
    resident.historical_k_packed = view.hist_k;
    resident.historical_v_packed = view.hist_v;
    resident.historical_k_metadata = view.hist_k_meta;
    resident.historical_v_metadata = view.hist_v_meta;
    resident.recent_k_bf16 = view.recent_k;
    resident.recent_v_bf16 = view.recent_v;
    resident.max_context = max_context_;
    const int visible = std::min<std::int32_t>(total, static_cast<std::int32_t>(query_token) + 1);
    const int selected = split_count != 0
                             ? split_count
                             : ninfer::ops::detail::
                                   oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
                                       visible);
    telemetry_->last_split_class = selected;
    ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_fused_decode_split_ring(
        rotated_q, resident.prefix_k_bf16,
        resident.prefix_v_bf16, prefix, resident.historical_k_packed,
        resident.historical_k_metadata, resident.historical_v_packed,
        resident.historical_v_metadata, historical, resident.recent_k_bf16,
        resident.recent_v_bf16, recent, ring_head, static_cast<std::int32_t>(query_token),
        split_count, 1.0F / 16.0F, view.workspace,
        av, stream, prefill_parallel_merge_, prefill_parallel_scores_);
}

void Exl3OscarContext::decode_layer(int model_layer, const std::uint16_t* q_row_f16,
                                    std::uint16_t* attn_out_f16, std::uint32_t query_token,
                                    int split_count, cudaStream_t stream) {
    if (!q_row_f16 || !attn_out_f16) {
        throw std::invalid_argument("E4C1 OSCAR decode arguments invalid");
    }
    auto& cache = layers_[bank_index(model_layer)];
    const auto view = cache->view();
    std::int32_t prefix = 0, historical = 0, recent = 0, ring_head = 0;
    cache->prefix_extent(prefix, historical, recent, ring_head);
    const std::int32_t total = prefix + historical + recent;
    if (total <= 0 || query_token >= static_cast<std::uint32_t>(total)) {
        throw std::invalid_argument("E4C1 OSCAR decode query outside cache");
    }
    const dim3 grid_q(24, 1u, 1u);
    if (decode_coalesced_rotations_) {
        rotate_q_coalesced_kernel<<<grid_q, 256, 0, stream>>>(q_row_f16,
            static_cast<const float*>(coalesced_r_k_[bank_index(model_layer)].get()),
            static_cast<float*>(staging_rq_), 1, 24);
    } else {
        rotate_q_decode_kernel<<<grid_q, 256, 0, stream>>>(q_row_f16, view.r_k,
                                                     static_cast<float*>(staging_rq_), 24);
    }
    cuda_check(cudaGetLastError(), "E4C1 OSCAR rotate decode Q");
    ninfer::ops::detail::OscarInt2G128ResidentCacheView resident{};
    resident.prefix_k_bf16 = view.prefix_k;
    resident.prefix_v_bf16 = view.prefix_v;
    resident.historical_k_packed = view.hist_k;
    resident.historical_v_packed = view.hist_v;
    resident.historical_k_metadata = view.hist_k_meta;
    resident.historical_v_metadata = view.hist_v_meta;
    resident.recent_k_bf16 = view.recent_k;
    resident.recent_v_bf16 = view.recent_v;
    resident.max_context = max_context_;
    const int visible = std::min<std::int32_t>(total, static_cast<std::int32_t>(query_token) + 1);
    const int selected = split_count != 0
                             ? split_count
                             : ninfer::ops::detail::
                                   oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
                                       visible);
    telemetry_->last_split_class = selected;
    ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_fused_decode_split_ring(
        static_cast<const float*>(staging_rq_), resident.prefix_k_bf16,
        resident.prefix_v_bf16, prefix, resident.historical_k_packed,
        resident.historical_k_metadata, resident.historical_v_packed,
        resident.historical_v_metadata, historical, resident.recent_k_bf16,
        resident.recent_v_bf16, recent, ring_head, static_cast<std::int32_t>(query_token),
        split_count, 1.0F / 16.0F, view.workspace,
        static_cast<float*>(staging_av_), stream);
    const dim3 grid_rec(24, 1u, 1u);
    if (decode_coalesced_rotations_) {
        recover_coalesced_kernel<<<grid_rec, 256, 0, stream>>>(
            static_cast<const float*>(staging_av_), view.r_v, attn_out_f16, 1, 24);
    } else {
        recover_kernel<<<grid_rec, 256, 0, stream>>>(static_cast<const float*>(staging_av_),
                                                     view.r_v_inv, attn_out_f16, 24);
    }
    cuda_check(cudaGetLastError(), "E4C1 OSCAR recover AV");
}

void Exl3OscarContext::prefill_chronological_layer(
    int model_layer, const std::uint16_t* q, const std::uint16_t* k,
    const std::uint16_t* v, int rows, std::uint32_t logical_start,
    std::uint16_t* output, cudaStream_t stream) {
    if (!batch_rotations_ || graph_class_ != 0 || !q || !k || !v || !output ||
        rows < 2 || rows > batch_staging_rows_ || logical_start > static_cast<std::uint32_t>(max_context_) ||
        static_cast<std::uint32_t>(rows) > static_cast<std::uint32_t>(max_context_) - logical_start) {
        throw std::invalid_argument("OSCAR chronological prefill admission");
    }
    auto& cache = layers_[bank_index(model_layer)];
    if (cache->context() != logical_start)
        throw std::invalid_argument("OSCAR chronological prefill position mismatch");
    const auto view = cache->view();
    if (!view.r_k || !view.r_v || !view.r_v_inv)
        throw std::logic_error("OSCAR chronological prefill rotations missing");
    rotate_qkv_kernel<<<dim3(4, rows), 256, 0, stream>>>(k, view.r_k,
        static_cast<float*>(staging_rk_), rows, 4);
    cuda_check(cudaGetLastError(), "OSCAR prefill rotate K");
    rotate_qkv_kernel<<<dim3(4, rows), 256, 0, stream>>>(v, view.r_v,
        static_cast<float*>(staging_rv_), rows, 4);
    cuda_check(cudaGetLastError(), "OSCAR prefill rotate V");
    if(prefill_coalesced_rotations_) {
        rotate_q_coalesced_kernel<<<rows*24,256,0,stream>>>(q,
            static_cast<const float*>(coalesced_r_k_[bank_index(model_layer)].get()),
            static_cast<float*>(staging_bq_),rows,24);
    } else {
    rotate_q_batch_kernel<<<rows * 24, 256, 0, stream>>>(q, view.r_k,
        static_cast<float*>(staging_bq_), rows, 24);
    }
    cuda_check(cudaGetLastError(), "OSCAR prefill rotate Q");
    if (prefill_query_parallel_) {
        constexpr std::size_t ring_bytes = 256ULL * 4 * 256 * sizeof(std::uint16_t);
        auto* old_k = static_cast<std::uint16_t*>(prefill_old_recent_.get());
        auto* old_v = old_k + ring_bytes / sizeof(std::uint16_t);
        cuda_check(cudaMemcpyAsync(old_k, view.recent_k, ring_bytes,
                                  cudaMemcpyDeviceToDevice, stream), "OSCAR snapshot recent K");
        cuda_check(cudaMemcpyAsync(old_v, view.recent_v, ring_bytes,
                                  cudaMemcpyDeviceToDevice, stream), "OSCAR snapshot recent V");
        ninfer::ops::detail::OscarChronologicalPrefillView logical{};
        logical.cache = {view.prefix_k, view.prefix_v, view.hist_k, view.hist_v,
                         view.hist_k_meta, view.hist_v_meta, view.recent_k, view.recent_v,
                         max_context_};
        logical.old_recent_k = old_k;
        logical.old_recent_v = old_v;
        logical.rotated_k = static_cast<const float*>(staging_rk_);
        logical.rotated_v = static_cast<const float*>(staging_rv_);
        logical.batch_start = static_cast<int>(logical_start);
        logical.old_recent_begin = static_cast<int>(recent_begin_for(logical_start));
        logical.old_recent_head = static_cast<int>(cache->host_head());
        // The resident encoder indexes new historical rows from logical_start.
        // Publish a prefix-crossing segment separately so it starts at >=64.
        int consumed = 0;
        if (logical_start < 64U) {
            consumed = std::min(rows, static_cast<int>(64U-logical_start));
            telemetry_->aging_events += cache->append(logical.rotated_k, logical.rotated_v,
                                                       consumed, logical_start, stream);
        }
        if (consumed < rows) {
            telemetry_->aging_events += cache->append(logical.rotated_k + consumed*4*256,
                logical.rotated_v + consumed*4*256, rows-consumed, logical_start+consumed, stream);
        }
        ninfer::ops::detail::oscar_chronological_prefill_launch(logical,
            static_cast<const float*>(staging_bq_), static_cast<int>(logical_start), rows,
            static_cast<float*>(prefill_query_workspace_.get()),
            static_cast<float*>(staging_bav_), stream, prefill_packed_byte4_);
        telemetry_->append_calls += rows;
        telemetry_->oscar_dispatches[model_layer] += rows;
        telemetry_->last_split_class = ninfer::ops::detail::
            oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(logical_start+rows);
    } else {
    for (int row = 0; row < rows; ++row) {
        // Preserve the cache representation visible to this exact query. In
        // particular, never age future rows before answering an earlier row.
        telemetry_->append_calls++;
        telemetry_->aging_events += cache->append(
            static_cast<const float*>(staging_rk_) + row * 4 * 256,
            static_cast<const float*>(staging_rv_) + row * 4 * 256,
            1, logical_start + row, stream);
        telemetry_->oscar_dispatches[model_layer]++;
        decode_rotated_layer(model_layer,
            static_cast<const float*>(staging_bq_) + row * 24 * 256,
            static_cast<float*>(staging_bav_) + row * 24 * 256,
            logical_start + row, 0, stream);
    }
    }
    if(prefill_coalesced_rotations_) {
        // Existing forward V layout is exactly the transpose of inverse V.
        recover_coalesced_kernel<<<rows*24,256,0,stream>>>(
            static_cast<const float*>(staging_bav_),view.r_v,output,rows,24);
    } else {
    recover_rows_kernel<<<rows * 24, 256, 0, stream>>>(
        static_cast<const float*>(staging_bav_), view.r_v_inv, output, rows, 24);
    }
    cuda_check(cudaGetLastError(), "OSCAR prefill recover AV");
}

void Exl3OscarContext::record_ordinary_dispatch() noexcept {
    telemetry_->ordinary_dispatches++;
}

void Exl3OscarContext::reset() noexcept {
    // Reset returns the context to eager mode: a new capture needs an
    // explicit set_graph_class after reset.
    graph_class_ = 0;
    ++checkpoint_generation_;
    for (auto& layer : layers_) {
        if (layer) layer->reset();
    }
}

void Exl3OscarContext::prefill_layer(int model_layer, const std::uint16_t* q_rope_f16,
                                     const std::uint16_t* k_rope_f16,
                                     const std::uint16_t* v_f16, int rows,
                                     std::uint32_t logical_start,
                                     std::uint16_t* attn_out_f16, cudaStream_t stream) {
    if (!q_rope_f16 || !k_rope_f16 || !v_f16 || !attn_out_f16 || rows <= 0) {
        throw std::invalid_argument("E4C1 OSCAR prefill arguments invalid");
    }
    append_layer(model_layer, q_rope_f16, k_rope_f16, v_f16, rows, logical_start, stream);
    auto& cache = layers_[bank_index(model_layer)];
    const auto view = cache->view();
    std::int32_t prefix = 0, historical = 0, recent = 0, ring_head = 0;
    cache->prefix_extent(prefix, historical, recent, ring_head);
    ninfer::ops::detail::OscarInt2G128ResidentCacheView resident{};
    resident.prefix_k_bf16 = view.prefix_k;
    resident.prefix_v_bf16 = view.prefix_v;
    resident.historical_k_packed = view.hist_k;
    resident.historical_v_packed = view.hist_v;
    resident.historical_k_metadata = view.hist_k_meta;
    resident.historical_v_metadata = view.hist_v_meta;
    resident.recent_k_bf16 = view.recent_k;
    resident.recent_v_bf16 = view.recent_v;
    resident.max_context = max_context_;
    int done = 0;
    while (done < rows) {
        const int block = std::min(rows - done, 64);
        rotate_q_batch_kernel<<<static_cast<unsigned>(block) * 24u, 256, 0, stream>>>(
            q_rope_f16 + static_cast<std::size_t>(done) * 24 * 256,
            view.r_k, static_cast<float*>(staging_bq_), block, 24);
        cuda_check(cudaGetLastError(), "E4C1 OSCAR rotate prefill Q");
        ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_fused_batch_ring(
            static_cast<const float*>(staging_bq_), resident.prefix_k_bf16,
            resident.prefix_v_bf16, prefix, resident.historical_k_packed,
            resident.historical_k_metadata, resident.historical_v_packed,
            resident.historical_v_metadata, historical, resident.recent_k_bf16,
            resident.recent_v_bf16, recent, ring_head,
            static_cast<std::int32_t>(logical_start) + done, block, 1.0F / 16.0F,
            static_cast<float*>(staging_bav_), stream);
        recover_rows_kernel<<<static_cast<unsigned>(block) * 24u, 256, 0, stream>>>(
            static_cast<const float*>(staging_bav_), view.r_v_inv,
            attn_out_f16 + static_cast<std::size_t>(done) * 24 * 256, block, 24);
        cuda_check(cudaGetLastError(), "E4C1 OSCAR recover prefill AV");
        done += block;
    }
}


// Single-token device append plan. Reads device (context, head), advances both,
// and publishes the write/encode plan plus decode counts. Fixed geometry:
// exactly one thread of one block does the integer work.
// Device append plan driven by the live position word (E4 pattern): the same
// captured work replays idempotently because every extent derives from the
// per-replay position upload, never from accumulated device state. The ring
// head is a pure function of context (telescoped begin differences), so no
// persistent head word is needed.
__global__ void oscar_exl3_plan_kernel(const int* position_device, int row_offset,
                                        std::uint32_t* plan, int* counts) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const std::uint32_t old_ctx =
        static_cast<std::uint32_t>(*position_device + row_offset);
    const std::uint32_t new_ctx = old_ctx + 1U;
    // The 64-row prefix and 256-row recent ring keep the split begin at 64
    // through context 320; only then does the recent window advance.
    const std::uint32_t old_begin = old_ctx <= 64U ? old_ctx :
        (old_ctx > 320U ? old_ctx - 256U : 64U);
    const std::uint32_t final_begin = new_ctx <= 64U ? new_ctx :
        (new_ctx > 320U ? new_ctx - 256U : 64U);
    const std::uint32_t old_head = old_begin & 255U;
    const std::uint32_t final_head = final_begin & 255U;
    const std::uint32_t old_aging_end = final_begin < old_ctx ? final_begin : old_ctx;
    const std::uint32_t old_aging = old_aging_end > old_begin ? old_aging_end - old_begin : 0U;
    const std::uint32_t append_end = old_ctx + 1U;
    const std::uint32_t new_hist_begin = old_ctx > 64U ? old_ctx : 64U;
    const std::uint32_t new_hist_end = append_end < final_begin ? append_end : final_begin;
    const std::uint32_t new_hist = new_hist_end > new_hist_begin ? new_hist_end - new_hist_begin : 0U;
    plan[0] = 1U;
    plan[1] = old_ctx;
    plan[2] = old_ctx;
    plan[3] = old_begin;
    plan[4] = old_head;
    plan[5] = final_begin;
    plan[6] = new_hist;
    plan[7] = old_aging;
    plan[8] = final_head;
    const std::uint32_t prefix = new_ctx < 64U ? new_ctx : 64U;
    counts[0] = static_cast<int>(prefix);
    counts[1] = static_cast<int>(final_begin - prefix);
    counts[2] = static_cast<int>(new_ctx - final_begin);
    counts[3] = static_cast<int>(final_head);
    counts[4] = static_cast<int>(new_ctx - 1U);
}

Exl3OscarPlanBoundaryObservation oscar_plan_boundary_observe_for_test(
    int old_position) {
    if (old_position < 0 || old_position > 1023)
        throw std::invalid_argument("OSCAR plan boundary position must be 0..1023");
    void* position_device = nullptr;
    void* plan_device = nullptr;
    void* counts_device = nullptr;
    const auto release = [&]() {
        if (counts_device) cudaFree(counts_device);
        if (plan_device) cudaFree(plan_device);
        if (position_device) cudaFree(position_device);
    };
    try {
        cuda_check(cudaMalloc(&position_device, sizeof(int)),
                   "allocate OSCAR plan boundary position");
        cuda_check(cudaMalloc(&plan_device, 16 * sizeof(std::uint32_t)),
                   "allocate OSCAR plan boundary plan");
        cuda_check(cudaMalloc(&counts_device, 5 * sizeof(std::int32_t)),
                   "allocate OSCAR plan boundary counts");
        cuda_check(cudaMemcpy(position_device, &old_position, sizeof(old_position),
                              cudaMemcpyHostToDevice),
                   "upload OSCAR plan boundary position");
        oscar_exl3_plan_kernel<<<1, 1>>>(static_cast<const int*>(position_device), 0,
                                         static_cast<std::uint32_t*>(plan_device),
                                         static_cast<int*>(counts_device));
        cuda_check(cudaGetLastError(), "launch OSCAR plan boundary kernel");
        Exl3OscarPlanBoundaryObservation result;
        cuda_check(cudaMemcpy(result.plan.data(), plan_device,
                              result.plan.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost),
                   "copy OSCAR plan boundary plan");
        cuda_check(cudaMemcpy(result.counts.data(), counts_device,
                              result.counts.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost),
                   "copy OSCAR plan boundary counts");
        release();
        return result;
    } catch (...) {
        release();
        throw;
    }
}

void Exl3OscarLayerCache::ensure_device_state() {
    if (!dev_plan_) dev_plan_ = device_alloc(16 * 4, "E4C1 OSCAR device plan");
    if (!dev_counts_) dev_counts_ = device_alloc(5 * 4, "E4C1 OSCAR device counts");
}

void Exl3OscarLayerCache::append_device(const float* rotated_k, const float* rotated_v,
                                        const int* position_device, int row_offset,
                                        cudaStream_t stream) {
    if (!rotated_k || !rotated_v || !position_device) {
        throw std::invalid_argument("E4C1 OSCAR device append null");
    }
    ensure_device_state();
    ninfer::ops::detail::OscarInt2G128ResidentCacheView view{};
    view.prefix_k_bf16 = static_cast<std::uint16_t*>(prefix_k_);
    view.prefix_v_bf16 = static_cast<std::uint16_t*>(prefix_v_);
    view.historical_k_packed = static_cast<std::uint8_t*>(hist_k_);
    view.historical_v_packed = static_cast<std::uint8_t*>(hist_v_);
    view.historical_k_metadata = static_cast<float*>(hist_k_meta_);
    view.historical_v_metadata = static_cast<float*>(hist_v_meta_);
    view.recent_k_bf16 = static_cast<std::uint16_t*>(recent_k_);
    view.recent_v_bf16 = static_cast<std::uint16_t*>(recent_v_);
    view.max_context = static_cast<std::int32_t>(max_context_);
    oscar_exl3_plan_kernel<<<1, 1, 0, stream>>>(position_device, row_offset,
                                               static_cast<std::uint32_t*>(dev_plan_),
                                               static_cast<int*>(dev_counts_));
    cuda_check(cudaGetLastError(), "E4C1 OSCAR plan kernel");
    ninfer::ops::detail::oscar_int2_g128_cache_encode_plan_launch(
        rotated_k, rotated_v, static_cast<const std::uint32_t*>(dev_plan_), view, stream);
    ninfer::ops::detail::oscar_int2_g128_cache_write_bf16_plan_launch(
        rotated_k, rotated_v, static_cast<const std::uint32_t*>(dev_plan_), view, stream);
}

const int* Exl3OscarLayerCache::device_counts() const noexcept {
    return static_cast<const int*>(dev_counts_);
}

void Exl3OscarLayerCache::sync_device_state(std::uint32_t context, std::uint32_t head,
                                               cudaStream_t stream) {
    // Device extents derive from the live position word; nothing to upload.
    // This entry point only guarantees plan/counts storage exists pre-capture.
    (void)context;
    (void)head;
    (void)stream;
}

void Exl3OscarLayerCache::reset_device(cudaStream_t stream) {
    (void)stream;
    reset();
}

void Exl3OscarLayerCache::note_host_replay(int rows) noexcept {
    for (int row = 0; row < rows; ++row) {
        const std::uint32_t old_ctx = context_;
        const std::uint32_t new_ctx = old_ctx + 1;
        const std::uint32_t old_begin = old_ctx <= 64U ? old_ctx :
            (old_ctx > 320U ? old_ctx - 256U : 64U);
        const std::uint32_t final_begin = new_ctx <= 64U ? new_ctx :
            (new_ctx > 320U ? new_ctx - 256U : 64U);
        recent_head_ = (recent_head_ + final_begin - old_begin) & 255U;
        context_ = new_ctx;
    }
}
void Exl3OscarContext::forward_decode_device(int model_layer, const std::uint16_t* q_row_f16,
                                                 const std::uint16_t* k_row_f16,
                                                 const std::uint16_t* v_row_f16,
                                                 std::uint16_t* attn_out_f16,
                                                 const int* position_device,
                                                 cudaStream_t stream) {
    if (!q_row_f16 || !k_row_f16 || !v_row_f16 || !attn_out_f16 || !position_device) {
        throw std::invalid_argument("E4C1 OSCAR device decode null");
    }
    if (graph_class_ != 16 && graph_class_ != 32 && graph_class_ != 64) {
        throw std::logic_error("E4C1 OSCAR device decode needs an explicit graph class");
    }
    auto& cache = layers_[bank_index(model_layer)];
    const auto view = cache->view();
    if(decode_fused_kv_rotations_) {
        rotate_kv_fused_kernel<<<8,256,0,stream>>>(k_row_f16,v_row_f16,
            view.r_k,view.r_v,static_cast<float*>(staging_rk_),
            static_cast<float*>(staging_rv_),1,4);
        cuda_check(cudaGetLastError(),"E4C1 OSCAR device rotate fused KV");
    } else {
        const dim3 grid_kv(4, 1u, 1u);
        rotate_qkv_kernel<<<grid_kv, 256, 0, stream>>>(k_row_f16, view.r_k,
                                                         static_cast<float*>(staging_rk_), 1, 4);
        cuda_check(cudaGetLastError(), "E4C1 OSCAR device rotate K");
        rotate_qkv_kernel<<<grid_kv, 256, 0, stream>>>(v_row_f16, view.r_v,
                                                         static_cast<float*>(staging_rv_), 1, 4);
        cuda_check(cudaGetLastError(), "E4C1 OSCAR device rotate V");
    }
    cache->append_device(static_cast<const float*>(staging_rk_),
                             static_cast<const float*>(staging_rv_), position_device, 0,
                             stream);
    const dim3 grid_q(24, 1u, 1u);
    rotate_q_decode_kernel<<<grid_q, 256, 0, stream>>>(q_row_f16, view.r_k,
                                                     static_cast<float*>(staging_rq_), 24);
    cuda_check(cudaGetLastError(), "E4C1 OSCAR device rotate Q");
    ninfer::ops::detail::OscarInt2G128ResidentCacheView resident{};
    resident.prefix_k_bf16 = view.prefix_k;
    resident.prefix_v_bf16 = view.prefix_v;
    resident.historical_k_packed = view.hist_k;
    resident.historical_v_packed = view.hist_v;
    resident.historical_k_metadata = view.hist_k_meta;
    resident.historical_v_metadata = view.hist_v_meta;
    resident.recent_k_bf16 = view.recent_k;
    resident.recent_v_bf16 = view.recent_v;
    resident.max_context = max_context_;
    ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_fused_decode_split_counts(
        static_cast<const float*>(staging_rq_), resident.prefix_k_bf16,
        resident.prefix_v_bf16, 0, resident.historical_k_packed,
        resident.historical_k_metadata, resident.historical_v_packed,
        resident.historical_v_metadata, 0, resident.recent_k_bf16,
        resident.recent_v_bf16, 0, 0, 0, graph_class_, 1.0F / 16.0F, view.workspace,
        static_cast<float*>(staging_av_), cache->device_counts(), stream);
    const dim3 grid_rec(24, 1u, 1u);
    recover_kernel<<<grid_rec, 256, 0, stream>>>(static_cast<const float*>(staging_av_),
                                                     view.r_v_inv, attn_out_f16, 24);
    cuda_check(cudaGetLastError(), "E4C1 OSCAR device recover");
}

void Exl3OscarContext::forward_continuation_device(
    int model_layer,const std::uint16_t* q_rows_f16,
    const std::uint16_t* k_rows_f16,const std::uint16_t* v_rows_f16,
    std::uint16_t* attn_out_f16,int rows,const int* position_device,
    int split_class,cudaStream_t stream) {
    if (!q_rows_f16 || !k_rows_f16 || !v_rows_f16 || !attn_out_f16 ||
        !position_device || rows < 2 || rows > 8)
        throw std::invalid_argument("B8 OSCAR continuation device arguments");
    if (split_class != 16 && split_class != 32 && split_class != 64)
        throw std::invalid_argument("B8 OSCAR continuation split class");
    auto& cache = layers_[bank_index(model_layer)];
    const auto view = cache->view();
    ninfer::ops::detail::OscarInt2G128ResidentCacheView resident{};
    resident.prefix_k_bf16 = view.prefix_k;
    resident.prefix_v_bf16 = view.prefix_v;
    resident.historical_k_packed = view.hist_k;
    resident.historical_v_packed = view.hist_v;
    resident.historical_k_metadata = view.hist_k_meta;
    resident.historical_v_metadata = view.hist_v_meta;
    resident.recent_k_bf16 = view.recent_k;
    resident.recent_v_bf16 = view.recent_v;
    resident.max_context = max_context_;
    for (int row = 0; row < rows; ++row) {
        const auto* q = q_rows_f16 + static_cast<std::size_t>(row) * 24 * 256;
        const auto* k = k_rows_f16 + static_cast<std::size_t>(row) * 4 * 256;
        const auto* v = v_rows_f16 + static_cast<std::size_t>(row) * 4 * 256;
        auto* output = attn_out_f16 + static_cast<std::size_t>(row) * 24 * 256;
        if (decode_fused_kv_rotations_) {
            rotate_kv_fused_kernel<<<8,256,0,stream>>>(k,v,view.r_k,view.r_v,
                static_cast<float*>(staging_rk_),static_cast<float*>(staging_rv_),1,4);
            cuda_check(cudaGetLastError(),"B8 OSCAR device rotate fused KV");
        } else {
            rotate_qkv_kernel<<<4,256,0,stream>>>(k,view.r_k,
                static_cast<float*>(staging_rk_),1,4);
            rotate_qkv_kernel<<<4,256,0,stream>>>(v,view.r_v,
                static_cast<float*>(staging_rv_),1,4);
            cuda_check(cudaGetLastError(),"B8 OSCAR device rotate KV");
        }
        cache->append_device(static_cast<const float*>(staging_rk_),
            static_cast<const float*>(staging_rv_),position_device,row,stream);
        if (decode_coalesced_rotations_) {
            rotate_q_coalesced_kernel<<<24,256,0,stream>>>(q,
                static_cast<const float*>(coalesced_r_k_[bank_index(model_layer)].get()),
                static_cast<float*>(staging_rq_),1,24);
        } else {
            rotate_q_decode_kernel<<<24,256,0,stream>>>(q,view.r_k,
                static_cast<float*>(staging_rq_),24);
        }
        cuda_check(cudaGetLastError(),"B8 OSCAR device rotate Q");
        ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_fused_decode_split_counts(
            static_cast<const float*>(staging_rq_),resident.prefix_k_bf16,
            resident.prefix_v_bf16,0,resident.historical_k_packed,
            resident.historical_k_metadata,resident.historical_v_packed,
            resident.historical_v_metadata,0,resident.recent_k_bf16,
            resident.recent_v_bf16,0,0,0,split_class,1.0F/16.0F,view.workspace,
            static_cast<float*>(staging_av_),cache->device_counts(),stream);
        if (decode_coalesced_rotations_) {
            recover_coalesced_kernel<<<24,256,0,stream>>>(
                static_cast<const float*>(staging_av_),view.r_v,output,1,24);
        } else {
            recover_kernel<<<24,256,0,stream>>>(static_cast<const float*>(staging_av_),
                view.r_v_inv,output,24);
        }
        cuda_check(cudaGetLastError(),"B8 OSCAR device recover");
    }
}

void Exl3OscarContext::forward_continuation_device_cohort_b8(
    int model_layer,const std::uint16_t* q_rows_f16,
    const std::uint16_t* k_rows_f16,const std::uint16_t* v_rows_f16,
    std::uint16_t* attn_out_f16,const int* position_device,
    int split_class,cudaStream_t stream) {
    forward_continuation_device_cohort(model_layer,q_rows_f16,k_rows_f16,
        v_rows_f16,attn_out_f16,8,position_device,split_class,stream);
}

void Exl3OscarContext::forward_continuation_device_cohort(
    int model_layer,const std::uint16_t* q_rows_f16,
    const std::uint16_t* k_rows_f16,const std::uint16_t* v_rows_f16,
    std::uint16_t* attn_out_f16,int rows,const int* position_device,
    int split_class,cudaStream_t stream) {
    if (!continuation_cohort_b8_ || !q_rows_f16 || !k_rows_f16 ||
        !v_rows_f16 || !attn_out_f16 || !position_device ||
        !prefill_old_recent_ || !prefill_query_workspace_ ||
        rows < 2 || rows > 8)
        throw std::invalid_argument("OSCAR chronological cohort arguments");
    if (split_class != 16 && split_class != 32 && split_class != 64)
        throw std::invalid_argument("B8 OSCAR chronological cohort split class");
    const int bank = bank_index(model_layer);
    auto& cache = layers_[bank];
    const auto view = cache->view();
    constexpr std::size_t ring_bytes =
        256ULL * 4 * 256 * sizeof(std::uint16_t);
    auto* old_k = static_cast<std::uint16_t*>(prefill_old_recent_.get());
    auto* old_v = old_k + ring_bytes / sizeof(std::uint16_t);

    // Snapshot before row zero can overwrite a physical ring slot. The
    // chronological reader combines this committed view with rotated B8 rows.
    cuda_check(cudaMemcpyAsync(old_k,view.recent_k,ring_bytes,
                               cudaMemcpyDeviceToDevice,stream),
               "snapshot B8 OSCAR chronological recent K");
    cuda_check(cudaMemcpyAsync(old_v,view.recent_v,ring_bytes,
                               cudaMemcpyDeviceToDevice,stream),
               "snapshot B8 OSCAR chronological recent V");
    if (decode_fused_kv_rotations_) {
        rotate_kv_fused_kernel<<<dim3(8,rows),256,0,stream>>>(
            k_rows_f16,v_rows_f16,view.r_k,view.r_v,
            static_cast<float*>(staging_rk_),static_cast<float*>(staging_rv_),
            rows,4);
        cuda_check(cudaGetLastError(),"B8 OSCAR cohort rotate fused KV");
    } else {
        rotate_qkv_kernel<<<dim3(4,rows),256,0,stream>>>(
            k_rows_f16,view.r_k,static_cast<float*>(staging_rk_),rows,4);
        rotate_qkv_kernel<<<dim3(4,rows),256,0,stream>>>(
            v_rows_f16,view.r_v,static_cast<float*>(staging_rv_),rows,4);
        cuda_check(cudaGetLastError(),"B8 OSCAR cohort rotate KV");
    }
    if (decode_coalesced_rotations_) {
        rotate_q_coalesced_kernel<<<rows*24,256,0,stream>>>(q_rows_f16,
            static_cast<const float*>(coalesced_r_k_[bank].get()),
            static_cast<float*>(staging_bq_),rows,24);
    } else {
        rotate_q_batch_kernel<<<rows*24,256,0,stream>>>(q_rows_f16,view.r_k,
            static_cast<float*>(staging_bq_),rows,24);
    }
    cuda_check(cudaGetLastError(),"B8 OSCAR cohort rotate Q");

    // Retain the already-qualified row-ordered plan/encode/write sequence.
    // Only attention over the eight causal views is coalesced by this probe.
    for (int row = 0; row < rows; ++row) {
        cache->append_device(
            static_cast<const float*>(staging_rk_) +
                static_cast<std::size_t>(row)*4*256,
            static_cast<const float*>(staging_rv_) +
                static_cast<std::size_t>(row)*4*256,
            position_device,row,stream);
    }

    ninfer::ops::detail::OscarChronologicalPrefillView chronological{};
    chronological.cache = {
        view.prefix_k,view.prefix_v,view.hist_k,view.hist_v,
        view.hist_k_meta,view.hist_v_meta,view.recent_k,view.recent_v,max_context_};
    chronological.old_recent_k = old_k;
    chronological.old_recent_v = old_v;
    chronological.rotated_k = static_cast<const float*>(staging_rk_);
    chronological.rotated_v = static_cast<const float*>(staging_rv_);
    chronological.position_device = position_device;
    ninfer::ops::detail::oscar_chronological_device_launch(
        chronological,static_cast<const float*>(staging_bq_),rows,split_class,
        static_cast<float*>(prefill_query_workspace_.get()),
        static_cast<float*>(staging_bav_),stream);
    if (decode_coalesced_rotations_) {
        recover_coalesced_kernel<<<rows*24,256,0,stream>>>(
            static_cast<const float*>(staging_bav_),view.r_v,
            attn_out_f16,rows,24);
    } else {
        recover_rows_kernel<<<rows*24,256,0,stream>>>(
            static_cast<const float*>(staging_bav_),view.r_v_inv,
            attn_out_f16,rows,24);
    }
    cuda_check(cudaGetLastError(),"B8 OSCAR cohort recover AV");
}

bool Exl3OscarContext::try_forward_continuation_device_cohort_eager(
    int model_layer,const std::uint16_t* q_rows_f16,
    const std::uint16_t* k_rows_f16,const std::uint16_t* v_rows_f16,
    std::uint16_t* attn_out_f16,int rows,int logical_start,
    const int* position_device,cudaStream_t stream) {
    ++telemetry_->continuation_cohort_eager_attempts;
    if (!continuation_cohort_b8_) {
        ++telemetry_->continuation_cohort_eager_latch_misses;
        return false;
    }
    if (!q_rows_f16 || !k_rows_f16 || !v_rows_f16 || !attn_out_f16 ||
        !position_device || rows < 2 || rows > 8 || logical_start < 0 ||
        logical_start > max_context_ || rows > max_context_ - logical_start) {
        ++telemetry_->continuation_cohort_eager_malformed;
        return false;
    }
    auto& cache = layers_[bank_index(model_layer)];
    if (!cache || cache->context() != static_cast<std::uint32_t>(logical_start)) {
        ++telemetry_->continuation_cohort_eager_malformed;
        return false;
    }
    // The device chronological merge selects its split class from the live
    // position word. Keep the eager cohort exact by admitting only cohorts
    // whose first and last visible query use the same adaptive class. A
    // boundary-crossing cohort remains on the existing row-ordered path.
    const int first_class = ninfer::ops::detail::
        oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
            logical_start + 1);
    const int last_class = ninfer::ops::detail::
        oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
            logical_start + rows);
    if (first_class != last_class) {
        ++telemetry_->continuation_cohort_eager_boundary_fallbacks;
        return false;
    }
    forward_continuation_device_cohort(
        model_layer,q_rows_f16,k_rows_f16,v_rows_f16,attn_out_f16,rows,
        position_device,first_class,stream);
    // The cohort uses the graph-safe device append plan, so reproduce the
    // graph caller's post-launch host mirror/telemetry update for this layer
    // only. The eager full-layer loop invokes this helper once per full layer;
    // advancing every layer here would corrupt the remaining layer cursors.
    const auto recent_begin=[](std::uint32_t context) {
        return context<=64U?context:
            (context>256U?std::max(64U,context-256U):64U);
    };
    const std::uint32_t old_context=cache->context();
    const std::uint32_t new_context=old_context+static_cast<std::uint32_t>(rows);
    cache->note_host_replay(rows);
    telemetry_->oscar_dispatches[model_layer] += static_cast<std::uint64_t>(rows);
    telemetry_->append_calls += static_cast<std::uint64_t>(rows);
    telemetry_->aging_events += static_cast<std::uint64_t>(
        recent_begin(new_context)-recent_begin(old_context));
    telemetry_->last_split_class=first_class;
    ++telemetry_->continuation_cohort_eager_dispatches;
    return true;
}

void Exl3OscarContext::sync_device_state(cudaStream_t stream) {
    for (auto& layer : layers_) {
        if (layer) layer->sync_device_state(layer->context(), layer->host_head(), stream);
    }
}

void Exl3OscarContext::note_graph_replay(int rows) noexcept {
    for (auto& layer : layers_) {
        if (layer) layer->note_host_replay(rows);
    }
}

void Exl3OscarContext::note_graph_execution(int rows,int split_class) noexcept {
    const auto recent_begin=[](std::uint32_t context) {
        return context<=64U?context:
            (context>256U?std::max(64U,context-256U):64U);
    };
    for (const int layer : kExl3OscarFullLayers) {
        const auto& cache=layers_[bank_index(layer)];
        const std::uint32_t old_context=cache->context();
        const std::uint32_t new_context=old_context+
            static_cast<std::uint32_t>(rows);
        telemetry_->oscar_dispatches[layer] += static_cast<std::uint64_t>(rows);
        telemetry_->append_calls += static_cast<std::uint64_t>(rows);
        telemetry_->aging_events +=
            static_cast<std::uint64_t>(recent_begin(new_context)-
                                       recent_begin(old_context));
    }
    if (split_class == 0 && !layers_.empty()) {
        const auto final_context=layers_.front()->context()+
            static_cast<std::uint32_t>(rows);
        split_class=final_context<=512U?16:(final_context<=8192U?32:64);
    }
    if (split_class != 0) telemetry_->last_split_class=split_class;
}

void Exl3OscarContext::ensure_device_attributes() {
    ninfer::ops::detail::oscar_int2_g128_ensure_graph_attributes();
    for (auto& layer : layers_) {
        if (layer) layer->ensure_device_state();
    }
}

std::size_t Exl3OscarContext::checkpoint_device_bytes() const noexcept {
    return layers_.size() * Exl3OscarLayerCache::kCheckpointBytes;
}

void Exl3OscarContext::save_checkpoint(Exl3OscarCheckpoint& checkpoint,
                                       cudaStream_t stream) {
    if (graph_class_ != 0) {
        throw std::logic_error("E4C1 OSCAR checkpoint is eager-only");
    }
    if (!checkpoint.cache_device || checkpoint.cache_capacity_bytes < checkpoint_device_bytes()) {
        throw std::invalid_argument("E4C1 OSCAR checkpoint device buffer is absent or undersized");
    }
    auto* destination = static_cast<std::byte*>(checkpoint.cache_device);
    for (std::size_t b = 0; b < layers_.size(); ++b) {
        layers_[b]->save_checkpoint(destination + b * Exl3OscarLayerCache::kCheckpointBytes,
                                    stream);
        checkpoint.contexts[b] = layers_[b]->context();
        checkpoint.recent_heads[b] = layers_[b]->host_head();
    }
    ++checkpoint_generation_;
    checkpoint.owner = this;
    checkpoint.generation = checkpoint_generation_;
    checkpoint.valid = true;
}

void Exl3OscarContext::restore_checkpoint(const Exl3OscarCheckpoint& checkpoint,
                                          cudaStream_t stream) {
    if (graph_class_ != 0) {
        throw std::logic_error("E4C1 OSCAR restore is eager-only");
    }
    if (!checkpoint.valid || checkpoint.owner != this ||
        checkpoint.generation != checkpoint_generation_ || !checkpoint.cache_device ||
        checkpoint.cache_capacity_bytes < checkpoint_device_bytes()) {
        throw std::invalid_argument("E4C1 OSCAR checkpoint is invalid for this context");
    }
    const auto* source = static_cast<const std::byte*>(checkpoint.cache_device);
    for (std::size_t b = 0; b < layers_.size(); ++b) {
        layers_[b]->restore_checkpoint(source + b * Exl3OscarLayerCache::kCheckpointBytes,
                                       checkpoint.contexts[b], checkpoint.recent_heads[b], stream);
    }
}

std::vector<std::byte> Exl3OscarContext::live_state_host_for_test(
    cudaStream_t stream) const {
    if (graph_class_ != 0) {
        throw std::logic_error("OSCAR live-state test export requires eager mode");
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status),
               "query OSCAR live-state test export capture");
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::logic_error("OSCAR live-state test export is unavailable during capture");
    }
    struct Extent { std::int32_t prefix, history, recent, head; };
    std::array<Extent, 16> extents{};
    constexpr std::size_t bf16_row_bytes = kExl3OscarKVHeads * kExl3OscarHeadDim * sizeof(std::uint16_t);
    constexpr std::size_t code_row_bytes = kExl3OscarKVHeads * 64;
    constexpr std::size_t meta_row_bytes = kExl3OscarKVHeads * 4 * sizeof(float);
    std::size_t total = 0;
    for (std::size_t bank = 0; bank < layers_.size(); ++bank) {
        if (!layers_[bank]) throw std::logic_error("OSCAR live-state test export has an absent layer");
        auto& extent = extents[bank];
        layers_[bank]->prefix_extent(extent.prefix, extent.history, extent.recent, extent.head);
        if (extent.prefix < 0 || extent.prefix > 64 || extent.history < 0 ||
            extent.recent < 0 || extent.recent > 256 || extent.head < 0 || extent.head >= 256 ||
            static_cast<std::uint32_t>(extent.prefix + extent.history + extent.recent) != layers_[bank]->context()) {
            throw std::logic_error("OSCAR live-state test export has inconsistent extents");
        }
        total += 6 * sizeof(std::uint32_t) +
            2 * bf16_row_bytes * static_cast<std::size_t>(extent.prefix + extent.recent) +
            2 * (code_row_bytes + meta_row_bytes) * static_cast<std::size_t>(extent.history);
    }
    std::vector<std::byte> result(total);
    // Test-only synchronous reads avoid leaving pending D2H writes into a
    // destroyed host vector if a later copy fails. Callers serialize mutation.
    cuda_check(cudaStreamSynchronize(stream), "synchronize OSCAR live-state test export");
    std::size_t offset = 0;
    const auto scalar = [&](std::uint32_t value) {
        std::memcpy(result.data() + offset, &value, sizeof(value));
        offset += sizeof(value);
    };
    const auto copy = [&](const void* source, std::size_t bytes) {
        if (bytes == 0) return;
        cuda_check(cudaMemcpy(result.data() + offset, source, bytes, cudaMemcpyDeviceToHost),
                   "copy OSCAR live logical state for test");
        offset += bytes;
    };
    for (std::size_t bank = 0; bank < layers_.size(); ++bank) {
        const auto& layer = *layers_[bank];
        const auto& extent = extents[bank];
        scalar(static_cast<std::uint32_t>(layer.model_layer()));
        scalar(layer.context());
        scalar(static_cast<std::uint32_t>(extent.prefix));
        scalar(static_cast<std::uint32_t>(extent.history));
        scalar(static_cast<std::uint32_t>(extent.recent));
        scalar(static_cast<std::uint32_t>(extent.head));
        copy(layer.prefix_k_, extent.prefix * bf16_row_bytes);
        copy(layer.prefix_v_, extent.prefix * bf16_row_bytes);
        // Historical storage is indexed by logical_token - prefix_tokens.
        copy(layer.hist_k_, extent.history * code_row_bytes);
        copy(layer.hist_k_meta_, extent.history * meta_row_bytes);
        copy(layer.hist_v_, extent.history * code_row_bytes);
        copy(layer.hist_v_meta_, extent.history * meta_row_bytes);
        const auto recent = [&](const void* source) {
            const auto* bytes = static_cast<const std::byte*>(source);
            const int first = std::min(extent.recent, 256 - extent.head);
            copy(bytes + extent.head * bf16_row_bytes, first * bf16_row_bytes);
            copy(bytes, (extent.recent - first) * bf16_row_bytes);
        };
        recent(layer.recent_k_);
        recent(layer.recent_v_);
    }
    if (offset != result.size()) throw std::logic_error("OSCAR live-state test export size mismatch");
    return result;
}

std::vector<std::byte> Exl3OscarContext::graph_layer_state_host_for_test(
    int model_layer,cudaStream_t stream) const {
    if (graph_class_ == 0)
        throw std::logic_error("OSCAR graph-layer export requires graph mode");
    const auto& layer = *layers_[bank_index(model_layer)];
    std::array<int,5> counts{};
    if (!layer.device_counts())
        throw std::logic_error("OSCAR graph-layer export lacks device counts");
    cuda_check(cudaMemcpyAsync(counts.data(),layer.device_counts(),sizeof(counts),
                               cudaMemcpyDeviceToHost,stream),
               "copy OSCAR graph-layer counts");
    cuda_check(cudaStreamSynchronize(stream),"synchronize OSCAR graph-layer counts");
    const int prefix=counts[0],history=counts[1],recent=counts[2],head=counts[3];
    if (prefix < 0 || prefix > 64 || history < 0 || recent < 0 || recent > 256 ||
        head < 0 || head >= 256 || prefix + history + recent != counts[4] + 1)
        throw std::logic_error("OSCAR graph-layer export has invalid extents");
    constexpr std::size_t bf16_row_bytes =
        kExl3OscarKVHeads*kExl3OscarHeadDim*sizeof(std::uint16_t);
    constexpr std::size_t code_row_bytes = kExl3OscarKVHeads*64;
    constexpr std::size_t meta_row_bytes = kExl3OscarKVHeads*4*sizeof(float);
    const std::size_t total = 6*sizeof(std::uint32_t) +
        2*bf16_row_bytes*static_cast<std::size_t>(prefix+recent) +
        2*(code_row_bytes+meta_row_bytes)*static_cast<std::size_t>(history);
    std::vector<std::byte> result(total);
    std::size_t offset=0;
    const auto scalar=[&](std::uint32_t value) {
        std::memcpy(result.data()+offset,&value,sizeof(value));offset+=sizeof(value);
    };
    const auto copy=[&](const void* source,std::size_t bytes) {
        if (!bytes) return;
        cuda_check(cudaMemcpy(result.data()+offset,source,bytes,cudaMemcpyDeviceToHost),
                   "copy OSCAR graph-layer state");
        offset+=bytes;
    };
    scalar(static_cast<std::uint32_t>(model_layer));
    scalar(static_cast<std::uint32_t>(counts[4]+1));
    scalar(static_cast<std::uint32_t>(prefix));
    scalar(static_cast<std::uint32_t>(history));
    scalar(static_cast<std::uint32_t>(recent));
    scalar(static_cast<std::uint32_t>(head));
    copy(layer.prefix_k_,static_cast<std::size_t>(prefix)*bf16_row_bytes);
    copy(layer.prefix_v_,static_cast<std::size_t>(prefix)*bf16_row_bytes);
    copy(layer.hist_k_,static_cast<std::size_t>(history)*code_row_bytes);
    copy(layer.hist_k_meta_,static_cast<std::size_t>(history)*meta_row_bytes);
    copy(layer.hist_v_,static_cast<std::size_t>(history)*code_row_bytes);
    copy(layer.hist_v_meta_,static_cast<std::size_t>(history)*meta_row_bytes);
    const auto copy_recent=[&](const void* source) {
        const auto* bytes=static_cast<const std::byte*>(source);
        const int first=std::min(recent,256-head);
        copy(bytes+static_cast<std::size_t>(head)*bf16_row_bytes,
             static_cast<std::size_t>(first)*bf16_row_bytes);
        copy(bytes,static_cast<std::size_t>(recent-first)*bf16_row_bytes);
    };
    copy_recent(layer.recent_k_);
    copy_recent(layer.recent_v_);
    if (offset != result.size())
        throw std::logic_error("OSCAR graph-layer export size mismatch");
    return result;
}

Exl3OscarGraphStateObservation Exl3OscarContext::graph_live_state_host_for_test(
    cudaStream_t stream) const {
    Exl3OscarGraphStateObservation result;
    try {
        if (graph_class_ == 0)
            throw std::logic_error("OSCAR graph live-state observation requires nonzero graph class");
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture_status),
                   "query OSCAR graph live-state capture state");
        if (capture_status != cudaStreamCaptureStatusNone)
            throw std::logic_error("OSCAR graph live-state observation is unavailable during capture");
        cuda_check(cudaStreamSynchronize(stream),
                   "synchronize OSCAR graph live-state observation");
        if (layers_.size() != result.device_counts.size())
            throw std::logic_error("OSCAR graph live-state layer count mismatch");

        struct Extent { std::int32_t prefix, history, recent, head, query; };
        std::array<Extent, 16> extents{};
        std::uint32_t common_context = 0;
        std::int32_t common_query = -1;
        for (std::size_t bank = 0; bank < layers_.size(); ++bank) {
            const auto& layer = layers_[bank];
            if (!layer) throw std::logic_error("OSCAR graph live-state absent layer");
            const int* device_counts = layer->device_counts();
            if (!device_counts)
                throw std::logic_error("OSCAR graph live-state device counts unavailable");
            cuda_check(cudaMemcpy(result.device_counts[bank].data(), device_counts,
                                  result.device_counts[bank].size() * sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "copy OSCAR graph live-state device counts");
            auto& extent = extents[bank];
            extent.prefix = result.device_counts[bank][0];
            extent.history = result.device_counts[bank][1];
            extent.recent = result.device_counts[bank][2];
            extent.head = result.device_counts[bank][3];
            extent.query = result.device_counts[bank][4];
            const std::int64_t context = static_cast<std::int64_t>(extent.prefix) +
                extent.history + extent.recent;
            if (extent.prefix < 0 || extent.prefix > 64 || extent.history < 0 ||
                extent.recent < 0 || extent.recent > 256 || extent.head < 0 ||
                extent.head >= 256 || context < 0 ||
                context > static_cast<std::int64_t>(max_context_) ||
                static_cast<std::int64_t>(extent.query) + 1 != context)
                throw std::logic_error("OSCAR graph live-state device extent invalid");
            if (context <= 0)
                throw std::logic_error("OSCAR graph live-state requires positive context");
            const std::int32_t expected_prefix = context < 64 ?
                static_cast<std::int32_t>(context) : 64;
            const std::int32_t expected_begin = context <= 64 ?
                static_cast<std::int32_t>(context) :
                static_cast<std::int32_t>(std::max<std::int64_t>(64, context - 256));
            const std::int32_t expected_history = expected_begin - expected_prefix;
            const std::int32_t expected_recent = static_cast<std::int32_t>(context) - expected_begin;
            if (extent.prefix != expected_prefix || extent.history != expected_history ||
                extent.recent != expected_recent || extent.head != (expected_begin & 255) ||
                extent.query != static_cast<std::int32_t>(context - 1))
                throw std::logic_error("OSCAR graph live-state device partition is noncanonical");
            if (static_cast<std::uint32_t>(extent.head) !=
                    (static_cast<std::uint32_t>(expected_begin) & 255U) ||
                layer->context() != static_cast<std::uint32_t>(context) ||
                layer->host_head() != static_cast<std::uint32_t>(extent.head))
                throw std::logic_error("OSCAR graph live-state host/device extent mismatch");
            if (bank == 0) {
                common_context = static_cast<std::uint32_t>(context);
                common_query = extent.query;
            } else if (common_context != static_cast<std::uint32_t>(context) ||
                       common_query != extent.query) {
                throw std::logic_error("OSCAR graph live-state layer extents disagree");
            }
            ++result.observed_layers;
        }

        constexpr std::size_t bf16_row_bytes =
            kExl3OscarKVHeads * kExl3OscarHeadDim * sizeof(std::uint16_t);
        constexpr std::size_t code_row_bytes = kExl3OscarKVHeads * 64;
        constexpr std::size_t meta_row_bytes =
            kExl3OscarKVHeads * 4 * sizeof(float);
        std::size_t total = 0;
        for (const auto& extent : extents) {
            total += 6 * sizeof(std::uint32_t) +
                2 * bf16_row_bytes * static_cast<std::size_t>(extent.prefix + extent.recent) +
                2 * (code_row_bytes + meta_row_bytes) *
                    static_cast<std::size_t>(extent.history);
        }
        std::vector<std::byte> candidate(total);
        std::size_t offset = 0;
        const auto scalar = [&](std::uint32_t value) {
            std::memcpy(candidate.data() + offset, &value, sizeof(value));
            offset += sizeof(value);
        };
        const auto copy = [&](const void* source, std::size_t bytes) {
            if (bytes == 0) return;
            if (!source) throw std::logic_error("OSCAR graph live-state null used buffer");
            if (bytes > candidate.size() - offset)
                throw std::logic_error("OSCAR graph live-state serialization overflow");
            cuda_check(cudaMemcpy(candidate.data() + offset, source, bytes,
                                  cudaMemcpyDeviceToHost),
                       "copy OSCAR graph live-state cache bytes");
            offset += bytes;
        };
        for (std::size_t bank = 0; bank < layers_.size(); ++bank) {
            const auto& layer = *layers_[bank];
            const auto& extent = extents[bank];
            const auto view = layer.view();
            scalar(static_cast<std::uint32_t>(layer.model_layer()));
            scalar(static_cast<std::uint32_t>(extent.prefix + extent.history + extent.recent));
            scalar(static_cast<std::uint32_t>(extent.prefix));
            scalar(static_cast<std::uint32_t>(extent.history));
            scalar(static_cast<std::uint32_t>(extent.recent));
            scalar(static_cast<std::uint32_t>(extent.head));
            copy(view.prefix_k, static_cast<std::size_t>(extent.prefix) * bf16_row_bytes);
            copy(view.prefix_v, static_cast<std::size_t>(extent.prefix) * bf16_row_bytes);
            copy(view.hist_k, static_cast<std::size_t>(extent.history) * code_row_bytes);
            copy(view.hist_k_meta, static_cast<std::size_t>(extent.history) * meta_row_bytes);
            copy(view.hist_v, static_cast<std::size_t>(extent.history) * code_row_bytes);
            copy(view.hist_v_meta, static_cast<std::size_t>(extent.history) * meta_row_bytes);
            const auto recent = [&](const void* source) {
                if (extent.recent > 0 && source == nullptr)
                    throw std::logic_error("OSCAR graph live-state null recent buffer");
                if (extent.recent == 0) return;
                const auto* bytes = static_cast<const std::byte*>(source);
                const int first = std::min(extent.recent, 256 - extent.head);
                copy(bytes + static_cast<std::size_t>(extent.head) * bf16_row_bytes,
                     static_cast<std::size_t>(first) * bf16_row_bytes);
                copy(bytes, static_cast<std::size_t>(extent.recent - first) * bf16_row_bytes);
            };
            recent(view.recent_k);
            recent(view.recent_v);
        }
        if (offset != candidate.size())
            throw std::logic_error("OSCAR graph live-state serialization size mismatch");
        result.canonical_bytes = std::move(candidate);
        result.valid = true;
    } catch (const std::exception& error) {
        result.valid = false;
        result.failure = error.what();
        result.canonical_bytes.clear();
    }
    return result;
}

}  // namespace ninfer::exl3
