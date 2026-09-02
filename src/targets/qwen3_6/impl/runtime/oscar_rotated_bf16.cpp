#include "targets/qwen3_6/impl/runtime/oscar_rotated_bf16.h"

#include "core/device.h"
#include "targets/qwen3_6/impl/frontend/digest.h"
#include "ops/softmax_attention/oscar_mixed/launch.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6::oscar_internal {
namespace {

using frontend_internal::sha256;
using frontend_internal::sha256_hex;

constexpr std::string_view kExpectedIdentity =
    "qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1";
constexpr std::string_view kExpectedModelSha256 =
    "6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e";
constexpr std::string_view kExpectedAssetManifestSha256 =
    "4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560";
constexpr std::string_view kCal10kIdentity =
    "qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal10k-v1";
constexpr std::string_view kCal10kAssetManifestSha256 =
    "7426bcd5fd34cd396d5fa2f225d590910e6c213fb3291295e0472478a6f231e9";
constexpr std::string_view kExpectedLayers =
    "3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63";
constexpr std::size_t kMatrixBytes = 16ULL * 256ULL * 256ULL * sizeof(float);

struct RuntimeAssetSpec {
    std::string_view identity;
    std::string_view asset_manifest_sha256;
};

constexpr std::array<RuntimeAssetSpec, 2> kSupportedRuntimeAssets = {{
    {kExpectedIdentity, kExpectedAssetManifestSha256},
    {kCal10kIdentity, kCal10kAssetManifestSha256},
}};

[[nodiscard]] const RuntimeAssetSpec* find_runtime_asset(std::string_view identity) {
    const auto it = std::find_if(kSupportedRuntimeAssets.begin(), kSupportedRuntimeAssets.end(),
                                 [&](const RuntimeAssetSpec& spec) {
                                     return spec.identity == identity;
                                 });
    return it == kSupportedRuntimeAssets.end() ? nullptr : &*it;
}

[[nodiscard]] std::string required_environment(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        throw std::invalid_argument(std::string("OSCAR rotated-BF16 requires ") + name);
    }
    return value;
}

[[nodiscard]] std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

void require_sha256(std::string_view value, const char* label) {
    if (value.size() != 64 ||
        !std::all_of(value.begin(), value.end(),
                     [](unsigned char c) { return std::isxdigit(c) != 0; })) {
        throw std::invalid_argument(std::string("OSCAR rotated-BF16 has invalid ") + label);
    }
}

[[nodiscard]] std::map<std::string, std::string> read_manifest(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error("cannot open OSCAR rotation manifest: " + path.string()); }
    std::map<std::string, std::string> fields;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) { continue; }
        const std::size_t equal = line.find('=');
        if (equal == std::string::npos || equal == 0 || equal + 1 == line.size()) {
            throw std::invalid_argument("OSCAR rotation manifest malformed at line " +
                                        std::to_string(line_number));
        }
        const std::string key = line.substr(0, equal);
        if (fields.contains(key)) {
            throw std::invalid_argument("OSCAR rotation manifest duplicates " + key);
        }
        fields.emplace(key, line.substr(equal + 1));
    }
    if (input.bad()) { throw std::runtime_error("cannot read OSCAR rotation manifest"); }
    return fields;
}

[[nodiscard]] const std::string& field(const std::map<std::string, std::string>& fields,
                                       std::string_view key) {
    const auto it = fields.find(std::string(key));
    if (it == fields.end() || it->second.empty()) {
        throw std::invalid_argument("OSCAR rotation manifest is missing " + std::string(key));
    }
    return it->second;
}

void require_field(const std::map<std::string, std::string>& fields, std::string_view key,
                   std::string_view expected) {
    if (field(fields, key) != expected) {
        throw std::invalid_argument("OSCAR rotation manifest mismatch for " + std::string(key));
    }
}

[[nodiscard]] std::vector<std::uint8_t> read_binary(const std::filesystem::path& path,
                                                    std::size_t expected_bytes,
                                                    std::string_view expected_sha,
                                                    const char* label) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size != expected_bytes) {
        throw std::invalid_argument(std::string("OSCAR rotation ") + label +
                                    " has an invalid byte size");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open OSCAR rotation " + path.string()); }
    std::vector<std::uint8_t> bytes(expected_bytes);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size()) || !input ||
        input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(std::string("OSCAR rotation ") + label + " is truncated");
    }
    const std::string actual = sha256_hex(sha256(std::span<const std::uint8_t>(bytes.data(), bytes.size())));
    if (lower(actual) != lower(std::string(expected_sha))) {
        throw std::invalid_argument(std::string("OSCAR rotation ") + label + " hash mismatch");
    }
    for (std::size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
        const std::uint32_t bits = static_cast<std::uint32_t>(bytes[offset]) |
                                   (static_cast<std::uint32_t>(bytes[offset + 1]) << 8U) |
                                   (static_cast<std::uint32_t>(bytes[offset + 2]) << 16U) |
                                   (static_cast<std::uint32_t>(bytes[offset + 3]) << 24U);
        if (!std::isfinite(std::bit_cast<float>(bits))) {
            throw std::invalid_argument(std::string("OSCAR rotation ") + label +
                                        " contains NaN/Inf");
        }
    }
    return bytes;
}

[[nodiscard]] std::shared_ptr<const OscarRotationSet> load_rotation_set() {
    const std::filesystem::path root =
        std::filesystem::absolute(required_environment("NINFER_OSCAR_ROTATION_ASSET_DIR"))
            .lexically_normal();
    const auto fields = read_manifest(root / "runtime_manifest.txt");
    const std::string_view identity = field(fields, "asset_identity");
    const RuntimeAssetSpec* spec = find_runtime_asset(identity);
    if (spec == nullptr) {
        throw std::invalid_argument("unsupported OSCAR runtime asset identity");
    }
    require_field(fields, "schema", "oscar-runtime-rotation-v1");
    require_field(fields, "asset_identity", spec->identity);
    require_field(fields, "model_sha256", kExpectedModelSha256);
    require_field(fields, "asset_manifest_sha256", spec->asset_manifest_sha256);
    require_field(fields, "full_attention_layers", kExpectedLayers);
    require_field(fields, "total_layers", "64");
    require_field(fields, "q_heads", "24");
    require_field(fields, "kv_heads", "4");
    require_field(fields, "gqa_ratio", "6");
    require_field(fields, "head_dim", "256");
    require_field(fields, "rotary_dim", "64");
    require_field(fields, "layout", "runtime[d,h,t];source[tokens,heads,head_dim]");
    require_field(fields, "dtype", "fp32");
    require_field(fields, "calibrated", "true");
    require_field(fields, "rotation_mode", "qqt_sst+r_h_pbr");
    require_field(fields, "k_layers", "16");
    require_field(fields, "v_layers", "16");

    const std::string model_sha = lower(required_environment("NINFER_OSCAR_MODEL_SHA256"));
    require_sha256(model_sha, "model SHA-256");
    if (model_sha != kExpectedModelSha256) {
        throw std::invalid_argument("OSCAR rotated-BF16 loaded model SHA-256 mismatch");
    }

    const std::string k_name = field(fields, "k_file");
    const std::string v_name = field(fields, "v_file");
    if (std::filesystem::path(k_name).filename() != k_name ||
        std::filesystem::path(v_name).filename() != v_name) {
        throw std::invalid_argument("OSCAR rotation file names must not escape the asset directory");
    }
    const std::vector<std::uint8_t> k_host =
        read_binary(root / k_name, kMatrixBytes, field(fields, "k_sha256"), "K asset");
    const std::vector<std::uint8_t> v_host =
        read_binary(root / v_name, kMatrixBytes, field(fields, "v_sha256"), "V asset");
    std::vector<float> k_host_fp32(k_host.size() / sizeof(float));
    std::vector<float> v_host_fp32(v_host.size() / sizeof(float));
    std::memcpy(k_host_fp32.data(), k_host.data(), k_host.size());
    std::memcpy(v_host_fp32.data(), v_host.data(), v_host.size());
    DeviceBuffer k_device(k_host.size());
    DeviceBuffer v_device(v_host.size());
    k_device.copy_from_host(k_host.data(), k_host.size());
    v_device.copy_from_host(v_host.data(), v_host.size());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto result = std::shared_ptr<const OscarRotationSet>(new OscarRotationSet(
        field(fields, "asset_identity"), field(fields, "asset_manifest_sha256"), model_sha,
        std::move(k_device), std::move(v_device), std::move(k_host_fp32), std::move(v_host_fp32)));
    const char* mode = std::getenv("NINFER_OSCAR_ROTATION_MODE");
    std::cerr << "OSCAR telemetry: asset_identity=" << result->asset_identity()
              << " asset_hash=" << result->asset_hash() << " calibrated=true"
              << " full_attention_layers=" << result->full_attention_layer_count()
              << " rotation_mode=" << (mode == nullptr ? "unknown" : mode)
              << " model_sha256=" << result->model_sha256()
              << '\n';
    return result;
}

} // namespace

// Implemented by the CUDA translation unit; keeping these declarations private to this target
// avoids exposing an OSCAR API through the public runtime headers.
void launch_oscar_rotate_qkv(const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&,
                             const float*, const float*, cudaStream_t);
void launch_oscar_inverse_value(const Tensor&, Tensor&, const float*, cudaStream_t);
void launch_oscar_rotate_qkv_fp32(const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                  Tensor&, const float*, const float*, cudaStream_t);
void launch_oscar_reference_attention_fp32(const Tensor&, const Tensor&, const Tensor&, float,
                                           Tensor&, cudaStream_t);
void launch_oscar_inverse_value_fp32(const Tensor&, Tensor&, const float*, cudaStream_t);
void launch_oscar_fp32_to_bf16(const Tensor&, Tensor&, cudaStream_t);
void launch_oscar_bf16_to_fp32(const Tensor&, Tensor&, cudaStream_t);
void launch_oscar_matched_cache_append(const Tensor&, const Tensor&, const Tensor&, float*, float*,
                                       std::uint32_t, cudaStream_t);
void launch_oscar_matched_attention(const Tensor&, const Tensor&, const float*, const float*,
                                    std::uint32_t, float, Tensor&, cudaStream_t);
void oscar_int2_g128_cache_write_bf16_launch(
    const float*, const float*, std::int32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    const ::ninfer::ops::detail::OscarInt2G128ResidentCacheView&, cudaStream_t);
void oscar_int2_g128_cache_encode_launch(
    const float*, const float*, std::int32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
    const ::ninfer::ops::detail::OscarInt2G128ResidentCacheView&, cudaStream_t);
void oscar_int2_g128_mixed_attention_launch_ring(
    const float*, const std::uint16_t*, const std::uint16_t*, std::int32_t, const std::uint8_t*,
    const float*, const std::uint8_t*, const float*, std::int32_t, const std::uint16_t*,
    const std::uint16_t*, std::int32_t, std::int32_t, float, float*, float*, float*, cudaStream_t);
void oscar_int2_g128_mixed_attention_launch_batch_ring(
    const float*, const std::uint16_t*, const std::uint16_t*, std::int32_t, const std::uint8_t*,
    const float*, const std::uint8_t*, const float*, std::int32_t, const std::uint16_t*,
    const std::uint16_t*, std::int32_t, std::int32_t, std::int32_t, std::int32_t, float, float*,
    float*, float*, cudaStream_t);


OscarRotationSet::OscarRotationSet(std::string asset_identity, std::string asset_hash,
                                   std::string model_sha256, DeviceBuffer k_rotation,
                                   DeviceBuffer v_rotation, std::vector<float> k_rotation_host,
                                   std::vector<float> v_rotation_host)
    : asset_identity_(std::move(asset_identity)), asset_hash_(std::move(asset_hash)),
      model_sha256_(std::move(model_sha256)), k_rotation_(std::move(k_rotation)),
      v_rotation_(std::move(v_rotation)), k_rotation_host_(std::move(k_rotation_host)),
      v_rotation_host_(std::move(v_rotation_host)) {
    if (k_rotation_host_.size() != 16ULL * 256ULL * 256ULL ||
        v_rotation_host_.size() != 16ULL * 256ULL * 256ULL) {
        throw std::invalid_argument("OSCAR host rotation bank shape mismatch");
    }
}

OscarRotationSet::~OscarRotationSet() = default;

namespace {

std::size_t full_attention_bank_index(std::int32_t model_layer) {
    constexpr std::array<std::int32_t, 16> layers = {
        3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63};
    const auto it = std::find(layers.begin(), layers.end(), model_layer);
    if (it == layers.end()) {
        throw std::invalid_argument("OSCAR rotation requested for a non-full-attention layer");
    }
    return static_cast<std::size_t>(it - layers.begin());
}

} // namespace

std::span<const float> OscarRotationSet::k_matrix_host(std::int32_t model_layer) const {
    const std::size_t offset = full_attention_bank_index(model_layer) * 256ULL * 256ULL;
    return std::span<const float>(k_rotation_host_.data() + offset, 256ULL * 256ULL);
}

std::span<const float> OscarRotationSet::v_matrix_host(std::int32_t model_layer) const {
    const std::size_t offset = full_attention_bank_index(model_layer) * 256ULL * 256ULL;
    return std::span<const float>(v_rotation_host_.data() + offset, 256ULL * 256ULL);
}

void OscarRotationSet::rotate_qkv(std::int32_t model_layer, const Tensor& q, const Tensor& k,
                                  const Tensor& v, Tensor& q_rotated, Tensor& k_rotated,
                                  Tensor& v_rotated, cudaStream_t stream) const {
    const std::array<int, 16> layers{3, 7, 11, 15, 19, 23, 27, 31,
                                     35, 39, 43, 47, 51, 55, 59, 63};
    const auto it = std::find(layers.begin(), layers.end(), model_layer);
    if (it == layers.end()) {
        throw std::invalid_argument("OSCAR rotated-BF16 attempted to rotate a non-full layer");
    }
    const std::size_t layer = static_cast<std::size_t>(it - layers.begin());
    const std::size_t offset = layer * 256ULL * 256ULL * sizeof(float);
    launch_oscar_rotate_qkv(q, k, v, q_rotated, k_rotated, v_rotated,
                            reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(k_rotation_.p) +
                                                           offset),
                            reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(v_rotation_.p) +
                                                           offset),
                            stream);
}

void OscarRotationSet::rotate_qkv_fp32(std::int32_t model_layer, const Tensor& q, const Tensor& k,
                                       const Tensor& v, Tensor& q_rotated, Tensor& k_rotated,
                                       Tensor& v_rotated, cudaStream_t stream) const {
    const std::array<int, 16> layers{3, 7, 11, 15, 19, 23, 27, 31,
                                     35, 39, 43, 47, 51, 55, 59, 63};
    const auto it = std::find(layers.begin(), layers.end(), model_layer);
    if (it == layers.end()) {
        throw std::invalid_argument("OSCAR rotated-BF16 attempted to rotate a non-full layer");
    }
    const std::size_t layer = static_cast<std::size_t>(it - layers.begin());
    const std::size_t offset = layer * 256ULL * 256ULL * sizeof(float);
    launch_oscar_rotate_qkv_fp32(
        q, k, v, q_rotated, k_rotated, v_rotated,
        reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(k_rotation_.p) + offset),
        reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(v_rotation_.p) + offset),
        stream);
}

void OscarRotationSet::reference_attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v,
                                                float scale, Tensor& out,
                                                cudaStream_t stream) const {
    launch_oscar_reference_attention_fp32(q, k, v, scale, out, stream);
}

void OscarRotationSet::inverse_value(std::int32_t model_layer, const Tensor& rotated,
                                     Tensor& recovered, cudaStream_t stream) const {
    const std::array<int, 16> layers{3, 7, 11, 15, 19, 23, 27, 31,
                                     35, 39, 43, 47, 51, 55, 59, 63};
    const auto it = std::find(layers.begin(), layers.end(), model_layer);
    if (it == layers.end()) {
        throw std::invalid_argument("OSCAR rotated-BF16 attempted to invert a non-full layer");
    }
    const std::size_t offset = static_cast<std::size_t>(it - layers.begin()) * 256ULL * 256ULL *
                               sizeof(float);
    launch_oscar_inverse_value(
        rotated, recovered,
        reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(v_rotation_.p) + offset),
        stream);
}

void OscarRotationSet::inverse_value_fp32(std::int32_t model_layer, const Tensor& rotated,
                                          Tensor& recovered, cudaStream_t stream) const {
    const std::array<int, 16> layers{3, 7, 11, 15, 19, 23, 27, 31,
                                     35, 39, 43, 47, 51, 55, 59, 63};
    const auto it = std::find(layers.begin(), layers.end(), model_layer);
    if (it == layers.end()) {
        throw std::invalid_argument("OSCAR rotated-BF16 attempted to invert a non-full layer");
    }
    const std::size_t offset = static_cast<std::size_t>(it - layers.begin()) * 256ULL * 256ULL *
                               sizeof(float);
    launch_oscar_inverse_value_fp32(
        rotated, recovered,
        reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(v_rotation_.p) + offset),
        stream);
}

void OscarRotationSet::fp32_to_bf16(const Tensor& input, Tensor& output,
                                    cudaStream_t stream) const {
    launch_oscar_fp32_to_bf16(input, output, stream);
}

OscarMatchedFP32Cache::OscarMatchedFP32Cache(std::uint32_t max_context)
    : max_context_(max_context),
      k_cache_(16ULL * max_context * 4ULL * 256ULL * sizeof(float)),
      v_cache_(16ULL * max_context * 4ULL * 256ULL * sizeof(float)) {
    if (max_context == 0) {
        throw std::invalid_argument("OSCAR matched FP32 cache capacity must be positive");
    }
}

OscarMatchedFP32Cache::~OscarMatchedFP32Cache() = default;

void OscarMatchedFP32Cache::reset(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(k_cache_.p, 0, k_cache_.bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(v_cache_.p, 0, v_cache_.bytes, stream));
}

void OscarMatchedFP32Cache::append(std::int32_t full_layer, const Tensor& k, const Tensor& v,
                                   const Tensor& positions, cudaStream_t stream) {
    if (full_layer < 0 || full_layer >= 16) {
        throw std::invalid_argument("OSCAR matched FP32 cache full-attention index is invalid");
    }
    const std::size_t layer_stride = static_cast<std::size_t>(max_context_) * 4ULL * 256ULL;
    const std::size_t offset         = static_cast<std::size_t>(full_layer) * layer_stride;
    launch_oscar_matched_cache_append(k, v, positions,
                                      static_cast<float*>(k_cache_.p) + offset,
                                      static_cast<float*>(v_cache_.p) + offset, max_context_, stream);
}

void OscarMatchedFP32Cache::copy_layer_prefix(std::int32_t full_layer,
                                               std::uint32_t token_count, float* k_host,
                                               float* v_host, cudaStream_t stream) const {
    if (full_layer < 0 || full_layer >= 16 || token_count == 0 || token_count > max_context_ ||
        k_host == nullptr || v_host == nullptr) {
        throw std::invalid_argument("OSCAR matched FP32 cache prefix request is invalid");
    }
    const std::size_t layer_stride = static_cast<std::size_t>(max_context_) * 4ULL * 256ULL;
    const std::size_t offset = static_cast<std::size_t>(full_layer) * layer_stride;
    const std::size_t bytes = static_cast<std::size_t>(token_count) * 4ULL * 256ULL * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(k_host, static_cast<const std::uint8_t*>(k_cache_.p) +
                                          offset * sizeof(float),
                               bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_host, static_cast<const std::uint8_t*>(v_cache_.p) +
                                          offset * sizeof(float),
                               bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < bytes / sizeof(float); ++i) {
        if (!std::isfinite(k_host[i]) || !std::isfinite(v_host[i])) {
            throw std::runtime_error("OSCAR matched FP32 cache prefix contains NaN/Inf");
        }
    }
}

void OscarMatchedFP32Cache::attention(std::int32_t full_layer, const Tensor& q,
                                      const Tensor& positions, float scale, Tensor& out,
                                      cudaStream_t stream) const {
    if (full_layer < 0 || full_layer >= 16) {
        throw std::invalid_argument("OSCAR matched FP32 cache full-attention index is invalid");
    }
    const std::size_t layer_stride = static_cast<std::size_t>(max_context_) * 4ULL * 256ULL;
    const std::size_t offset         = static_cast<std::size_t>(full_layer) * layer_stride;
    launch_oscar_matched_attention(q, positions, static_cast<const float*>(k_cache_.p) + offset,
                                   static_cast<const float*>(v_cache_.p) + offset, max_context_,
                                   scale, out, stream);
}

namespace {

constexpr std::uint32_t kLiveTapMagic   = 0x3342524FU; // "ORB3"
constexpr std::uint32_t kLiveTapVersion = 1U;
constexpr std::size_t kLiveRowValues = 4ULL * 256ULL;
constexpr std::size_t kLiveQueryValues = 24ULL * 256ULL;

float live_bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

std::uint16_t live_float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

std::vector<std::uint32_t> parse_live_list(const char* raw, const char* label) {
    if (raw == nullptr || *raw == '\0') { return {}; }
    const std::string_view text(raw);
    std::vector<std::uint32_t> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t comma = text.find(',', begin);
        const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
        if (end == begin) { throw std::invalid_argument(std::string("empty ") + label); }
        std::uint32_t value = 0;
        const auto parsed = std::from_chars(text.data() + begin, text.data() + end, value, 10);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + end) {
            throw std::invalid_argument(std::string("invalid ") + label);
        }
        result.push_back(value);
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
        if (begin == text.size()) { throw std::invalid_argument(std::string("trailing ") + label); }
    }
    return result;
}

bool list_contains(const std::vector<std::uint32_t>& values, std::uint32_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

std::uint32_t live_recent_begin(std::uint32_t context_tokens) {
    if (context_tokens <= ::ninfer::kOscarMixedPrefixTokens) { return context_tokens; }
    return std::max(::ninfer::kOscarMixedPrefixTokens,
                    context_tokens > ::ninfer::kOscarMixedRecentTokens
                        ? context_tokens - ::ninfer::kOscarMixedRecentTokens
                        : 0U);
}

template <typename T>
void write_live_tap_value(std::ofstream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) { throw std::runtime_error("cannot write OSCAR live reference tap"); }
}

template <typename T>
void write_live_tap_span(std::ofstream& output, std::span<const T> values) {
    if (!values.empty()) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size_bytes()));
        if (!output) { throw std::runtime_error("cannot write OSCAR live reference tap payload"); }
    }
}

void write_live_tap_string(std::ofstream& output, std::string_view value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("OSCAR live reference tap string is too long");
    }
    const auto size = static_cast<std::uint32_t>(value.size());
    write_live_tap_value(output, size);
    if (!value.empty()) {
        output.write(value.data(), static_cast<std::streamsize>(value.size()));
        if (!output) { throw std::runtime_error("cannot write OSCAR live reference tap string"); }
    }
}

void write_live_trace_vector(std::ofstream& output, std::span<const float> values) {
    const auto size = static_cast<std::uint64_t>(values.size());
    write_live_tap_value(output, size);
    write_live_tap_span(output, values);
}

bool oscar_d4_profile_enabled() noexcept {
    const char* d41 = std::getenv("NINFER_OSCAR_D4_1_PROFILE");
    const char* d44 = std::getenv("NINFER_OSCAR_D4_4_PROFILE");
    const char* d46 = std::getenv("NINFER_OSCAR_D4_6_PROFILE");
    return (d41 != nullptr && d41[0] == '1') || (d44 != nullptr && d44[0] == '1') ||
           (d46 != nullptr && d46[0] == '1');
}

bool oscar_gpu_oracle_enabled() noexcept {
    const char* d43 = std::getenv("NINFER_OSCAR_D4_3_VALIDATE_REFERENCE");
    const char* d44 = std::getenv("NINFER_OSCAR_D4_4_VALIDATE_REFERENCE");
    const char* d45 = std::getenv("NINFER_OSCAR_D4_5_VALIDATE_REFERENCE");
    const char* d46 = std::getenv("NINFER_OSCAR_D4_6_VALIDATE_REFERENCE");
    return (d43 != nullptr && d43[0] == '1') || (d44 != nullptr && d44[0] == '1') ||
           (d45 != nullptr && d45[0] == '1') || (d46 != nullptr && d46[0] == '1');
}

bool oscar_resident_async_profile_enabled() noexcept {
    const char* value = std::getenv("NINFER_OSCAR_D4_4_PERF_NO_ORACLE");
    return value != nullptr && value[0] == '1';
}

double oscar_elapsed_us(std::chrono::steady_clock::time_point start,
                        std::chrono::steady_clock::time_point end) noexcept {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

} // namespace

OscarLiveMixedReferenceCache::OscarLiveMixedReferenceCache(
    std::uint32_t max_context, std::uint64_t sequence_id,
    std::shared_ptr<const OscarRotationSet> rotations)
    : max_context_(max_context), sequence_id_(sequence_id), rotations_(std::move(rotations)),
      gpu_resident_enabled_(live_int2_gpu_resident_mode_enabled()),
      gpu_fused_enabled_(live_int2_gpu_fused_mode_enabled()),
      gpu_host_oracle_enabled_(oscar_gpu_oracle_enabled()),
      asset_(::ninfer::OscarMixedAgingAssetContract::from_runtime(
          rotations_ ? rotations_->asset_identity() : std::string{},
          rotations_ ? rotations_->model_sha256() : std::string{},
          rotations_ ? rotations_->asset_hash() : std::string{}, "qqt_sst+r_h_pbr")) {
    if (max_context_ == 0 || !rotations_) {
        throw std::invalid_argument("OSCAR live mixed cache requires capacity and rotations");
    }
    asset_.validate();
    if (rotations_->asset_identity() != asset_.asset_identity ||
        rotations_->model_sha256() != asset_.model_sha256 ||
        rotations_->asset_hash() != asset_.asset_manifest_sha256) {
        throw std::invalid_argument("OSCAR live mixed cache rotation contract mismatch");
    }
    refresh_live_reference_taps();
    initialize_layers();
    if (gpu_resident_enabled_) {
        gpu_prefill_query_block_size_ = live_gpu_prefill_query_block_size();
        initialize_gpu_resident_storage();
    }
}

OscarLiveMixedReferenceCache::~OscarLiveMixedReferenceCache() = default;

void OscarLiveMixedReferenceCache::initialize_layers() {
    for (std::size_t index = 0; index < asset_.full_attention_layers.size(); ++index) {
        layers_[index] = std::make_unique<::ninfer::OscarMixedAgingLayerCache>(
            asset_.full_attention_layers[index], sequence_id_, asset_);
    }
}

void OscarLiveMixedReferenceCache::initialize_gpu_resident_storage() {
    constexpr std::size_t kRowValues =
        static_cast<std::size_t>(::ninfer::kOscarMixedKVHeads) * ::ninfer::kOscarMixedHeadDim;
    constexpr std::size_t kCodeBytes = ::ninfer::ops::kOscarInt2G128CodeBytes;
    constexpr std::size_t kMetadataItems = ::ninfer::ops::kOscarInt2G128MetadataItems;
    const std::size_t prefix_bytes =
        static_cast<std::size_t>(::ninfer::kOscarMixedPrefixTokens) * kRowValues *
        sizeof(std::uint16_t);
    const std::size_t recent_bytes =
        static_cast<std::size_t>(::ninfer::kOscarMixedRecentTokens) * kRowValues *
        sizeof(std::uint16_t);
    const std::size_t historical_payload_bytes =
        static_cast<std::size_t>(max_context_) * ::ninfer::kOscarMixedKVHeads * kCodeBytes;
    const std::size_t historical_metadata_bytes =
        static_cast<std::size_t>(max_context_) * ::ninfer::kOscarMixedKVHeads * kMetadataItems *
        sizeof(float);
    for (auto& layer : gpu_resident_layers_) {
        layer.prefix_k = DeviceBuffer(prefix_bytes);
        layer.prefix_v = DeviceBuffer(prefix_bytes);
        layer.historical_k = DeviceBuffer(historical_payload_bytes);
        layer.historical_v = DeviceBuffer(historical_payload_bytes);
        layer.historical_k_metadata = DeviceBuffer(historical_metadata_bytes);
        layer.historical_v_metadata = DeviceBuffer(historical_metadata_bytes);
        layer.recent_k = DeviceBuffer(recent_bytes);
        layer.recent_v = DeviceBuffer(recent_bytes);
        gpu_resident_cache_bytes_ += 2ULL * prefix_bytes + 2ULL * recent_bytes +
                                     2ULL * historical_payload_bytes +
                                     2ULL * historical_metadata_bytes;
    }
    const std::size_t attention_workspace_bytes =
        static_cast<std::size_t>(max_context_) * ::ninfer::kOscarMixedQueryHeads * sizeof(float);
    if (!gpu_fused_enabled_) {
        const std::size_t batched_workspace_bytes =
            attention_workspace_bytes * static_cast<std::size_t>(gpu_prefill_query_block_size_);
        gpu_resident_scores_ = DeviceBuffer(batched_workspace_bytes);
        gpu_resident_softmax_ = DeviceBuffer(batched_workspace_bytes);
        gpu_resident_workspace_bytes_ = 2ULL * batched_workspace_bytes;
    } else {
        // D4.6 keeps only a fixed split-KV decode reduction buffer. The D4.5 score and softmax
        // buffers are deliberately not allocated in the production fused mode.
        gpu_resident_fused_decode_workspace_ = DeviceBuffer(
            ::ninfer::ops::detail::kOscarMixedFusedDecodeWorkspaceBytes);
        gpu_resident_workspace_bytes_ =
            ::ninfer::ops::detail::kOscarMixedFusedDecodeWorkspaceBytes;
    }
    profile_.gpu_resident_cache_bytes = gpu_resident_cache_bytes_;
    profile_.gpu_resident_workspace_bytes = gpu_resident_workspace_bytes_;
}

void OscarLiveMixedReferenceCache::reset() {
    dispatch_seen_.fill(false);
    aging_parity_checks_ = 0;
    profile_ = {};
    gpu_staged_layer_ = -1;
    gpu_staged_context_ = 0;
    gpu_staged_prefix_ = 0;
    gpu_staged_historical_ = 0;
    gpu_staged_recent_ = 0;
    for (auto& layer : gpu_resident_layers_) {
        layer.context = 0;
        layer.recent_head = 0;
    }
    profile_.gpu_resident_cache_bytes = gpu_resident_cache_bytes_;
    profile_.gpu_resident_workspace_bytes = gpu_resident_workspace_bytes_;
    initialize_layers();
}

void OscarLiveMixedReferenceCache::append_gpu(std::int32_t model_layer,
                                              std::uint32_t logical_start,
                                              const Tensor& rotated_k, const Tensor& rotated_v,
                                              cudaStream_t stream) {
    if (!gpu_resident_enabled_) {
        throw std::logic_error("OSCAR resident append called outside resident mode");
    }
    const auto valid = [](const Tensor& tensor, const char* label) {
        if (tensor.dtype != DType::FP32 || tensor.data == nullptr || !tensor.is_contiguous() ||
            tensor.ne[0] != ::ninfer::kOscarMixedHeadDim ||
            tensor.ne[1] != ::ninfer::kOscarMixedKVHeads || tensor.ne[2] <= 0 ||
            tensor.ne[3] != 1) {
            throw std::invalid_argument(std::string("OSCAR resident ") + label +
                                        " must be contiguous FP32 [256,4,T]");
        }
    };
    valid(rotated_k, "K");
    valid(rotated_v, "V");
    if (rotated_k.ne[2] != rotated_v.ne[2]) {
        throw std::invalid_argument("OSCAR resident K/V append token counts disagree");
    }
    const std::uint32_t token_count = static_cast<std::uint32_t>(rotated_k.ne[2]);
    const std::size_t index = full_attention_bank_index(model_layer);
    auto& resident = gpu_resident_layers_[index];
    if (logical_start != resident.context) {
        throw std::invalid_argument("OSCAR resident append is not the next logical range");
    }
    if (token_count > max_context_ - resident.context) {
        throw std::out_of_range("OSCAR resident append exceeds cache capacity");
    }
    const std::uint32_t old_context = resident.context;
    const std::uint32_t final_context = old_context + token_count;
    const std::uint32_t old_recent_begin = live_recent_begin(old_context);
    const std::uint32_t final_recent_begin = live_recent_begin(final_context);
    const std::uint32_t final_recent_head =
        (resident.recent_head + final_recent_begin - old_recent_begin) &
        (::ninfer::kOscarMixedRecentTokens - 1U);
    const std::uint32_t old_aging_end = std::min(final_recent_begin, old_context);
    const std::uint32_t old_aging_tokens =
        old_aging_end > old_recent_begin ? old_aging_end - old_recent_begin : 0U;
    const std::uint32_t append_end = logical_start + token_count;
    const std::uint32_t new_history_begin =
        std::max(logical_start, ::ninfer::kOscarMixedPrefixTokens);
    const std::uint32_t new_history_end = std::min(append_end, final_recent_begin);
    const std::uint32_t new_historical_tokens =
        new_history_end > new_history_begin ? new_history_end - new_history_begin : 0U;
    const ::ninfer::ops::detail::OscarInt2G128ResidentCacheView view{
        static_cast<std::uint16_t*>(resident.prefix_k.p),
        static_cast<std::uint16_t*>(resident.prefix_v.p),
        static_cast<std::uint8_t*>(resident.historical_k.p),
        static_cast<std::uint8_t*>(resident.historical_v.p),
        static_cast<float*>(resident.historical_k_metadata.p),
        static_cast<float*>(resident.historical_v_metadata.p),
        static_cast<std::uint16_t*>(resident.recent_k.p),
        static_cast<std::uint16_t*>(resident.recent_v.p),
        static_cast<std::int32_t>(max_context_)};

    const bool profiling = oscar_d4_profile_enabled();
    const bool synchronize_profiled_append = profiling && !oscar_resident_async_profile_enabled();
    // Encode before publishing the new recent rows: the first aged token and the newest token
    // intentionally reuse the same physical ring slot.  This ordering is the device equivalent
    // of the host policy's read-old-then-retire-slot transition.
    if (new_historical_tokens != 0 || old_aging_tokens != 0) {
        const auto aging_start = profiling ? std::chrono::steady_clock::now()
                                           : std::chrono::steady_clock::time_point{};
        ::ninfer::ops::detail::oscar_int2_g128_cache_encode_launch(
            static_cast<const float*>(rotated_k.data), static_cast<const float*>(rotated_v.data),
            static_cast<std::int32_t>(token_count), logical_start, old_context, old_recent_begin,
            resident.recent_head, final_recent_begin, new_historical_tokens, old_aging_tokens,
            view, stream);
        if (synchronize_profiled_append) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            profile_.gpu_resident_aging_us +=
                oscar_elapsed_us(aging_start, std::chrono::steady_clock::now());
        } else if (profiling) {
            profile_.gpu_resident_aging_us +=
                oscar_elapsed_us(aging_start, std::chrono::steady_clock::now());
        }
    }
    const auto publish_start = profiling ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
    ::ninfer::ops::detail::oscar_int2_g128_cache_write_bf16_launch(
        static_cast<const float*>(rotated_k.data), static_cast<const float*>(rotated_v.data),
        static_cast<std::int32_t>(token_count), logical_start, final_recent_begin,
        final_recent_head, view, stream);
    if (synchronize_profiled_append) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        profile_.gpu_resident_publish_us +=
            oscar_elapsed_us(publish_start, std::chrono::steady_clock::now());
    } else if (profiling) {
        profile_.gpu_resident_publish_us +=
            oscar_elapsed_us(publish_start, std::chrono::steady_clock::now());
    }
    if (profiling) {
        ++profile_.gpu_resident_append_calls;
        profile_.gpu_resident_aging_events += old_aging_tokens + new_historical_tokens;
    }

    resident.context = final_context;
    resident.recent_head = final_recent_head;

    // A host mirror is permitted only for explicit D4.4/D4.3 oracle validation.  It is never
    // consulted by the resident reader and is absent from normal optimized runs.
    if (gpu_host_oracle_enabled_) {
        // Compare device-produced historical records only after all device work for this append
        // has completed.  This readback is validation-only and is absent from production runs.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> host_k(static_cast<std::size_t>(kLiveRowValues) * token_count);
        std::vector<float> host_v(host_k.size());
        CUDA_CHECK(cudaMemcpy(host_k.data(), rotated_k.data, host_k.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_v.data(), rotated_v.data, host_v.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        std::array<std::uint16_t, kLiveRowValues> k_row{};
        std::array<std::uint16_t, kLiveRowValues> v_row{};
        for (std::uint32_t token = 0; token < token_count; ++token) {
            for (std::uint32_t head = 0; head < ::ninfer::kOscarMixedKVHeads; ++head) {
                for (std::uint32_t dimension = 0;
                     dimension < ::ninfer::kOscarMixedHeadDim; ++dimension) {
                    const std::size_t source = static_cast<std::size_t>(dimension) +
                                               static_cast<std::size_t>(::ninfer::kOscarMixedHeadDim) *
                                                   (static_cast<std::size_t>(head) +
                                                    static_cast<std::size_t>(::ninfer::kOscarMixedKVHeads) *
                                                        token);
                    const std::size_t row = static_cast<std::size_t>(head) *
                                            ::ninfer::kOscarMixedHeadDim + dimension;
                    k_row[row] = live_float_to_bf16(host_k[source]);
                    v_row[row] = live_float_to_bf16(host_v[source]);
                }
            }
            append(model_layer, logical_start + token, k_row, v_row);
        }

        const auto& host_layer = *layers_[index];
        const auto compare_device_range = [&](std::uint32_t begin, std::uint32_t end) {
            if (begin >= end) { return; }
            if (begin < ::ninfer::kOscarMixedPrefixTokens ||
                end > live_recent_begin(final_context)) {
                throw std::logic_error("OSCAR resident codec parity selected non-historical range");
            }
            const std::size_t token_count_in_range = end - begin;
            const std::size_t row_count = token_count_in_range * ::ninfer::kOscarMixedKVHeads;
            std::vector<std::uint8_t> expected_k(
                row_count * ::ninfer::ops::kOscarInt2G128CodeBytes);
            std::vector<std::uint8_t> expected_v(expected_k.size());
            std::vector<float> expected_k_meta(
                row_count * ::ninfer::ops::kOscarInt2G128MetadataItems);
            std::vector<float> expected_v_meta(expected_k_meta.size());
            for (std::size_t token = 0; token < token_count_in_range; ++token) {
                for (std::uint32_t head = 0; head < ::ninfer::kOscarMixedKVHeads; ++head) {
                    const std::size_t row = token * ::ninfer::kOscarMixedKVHeads + head;
                    const auto& host_k_row = host_layer.historical_k(
                        static_cast<std::uint32_t>(begin + token), head);
                    const auto& host_v_row = host_layer.historical_v(
                        static_cast<std::uint32_t>(begin + token), head);
                    std::copy(host_k_row.packed.begin(), host_k_row.packed.end(),
                              expected_k.begin() +
                                  row * ::ninfer::ops::kOscarInt2G128CodeBytes);
                    std::copy(host_v_row.packed.begin(), host_v_row.packed.end(),
                              expected_v.begin() +
                                  row * ::ninfer::ops::kOscarInt2G128CodeBytes);
                    std::copy(host_k_row.scales_zeros.begin(), host_k_row.scales_zeros.end(),
                              expected_k_meta.begin() +
                                  row * ::ninfer::ops::kOscarInt2G128MetadataItems);
                    std::copy(host_v_row.scales_zeros.begin(), host_v_row.scales_zeros.end(),
                              expected_v_meta.begin() +
                                  row * ::ninfer::ops::kOscarInt2G128MetadataItems);
                }
            }
            const std::size_t first_row =
                (static_cast<std::size_t>(begin) - ::ninfer::kOscarMixedPrefixTokens) *
                ::ninfer::kOscarMixedKVHeads;
            std::vector<std::uint8_t> device_k(expected_k.size());
            std::vector<std::uint8_t> device_v(expected_v.size());
            std::vector<float> device_k_meta(expected_k_meta.size());
            std::vector<float> device_v_meta(expected_v_meta.size());
            CUDA_CHECK(cudaMemcpy(
                device_k.data(), static_cast<const std::uint8_t*>(resident.historical_k.p) +
                                     first_row * ::ninfer::ops::kOscarInt2G128CodeBytes,
                device_k.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                device_v.data(), static_cast<const std::uint8_t*>(resident.historical_v.p) +
                                     first_row * ::ninfer::ops::kOscarInt2G128CodeBytes,
                device_v.size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                device_k_meta.data(),
                static_cast<const float*>(resident.historical_k_metadata.p) +
                    first_row * ::ninfer::ops::kOscarInt2G128MetadataItems,
                device_k_meta.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                device_v_meta.data(),
                static_cast<const float*>(resident.historical_v_metadata.p) +
                    first_row * ::ninfer::ops::kOscarInt2G128MetadataItems,
                device_v_meta.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (device_k != expected_k || device_v != expected_v ||
                device_k_meta != expected_k_meta || device_v_meta != expected_v_meta) {
                throw std::logic_error("OSCAR resident device codec differs from host G128");
            }
            profile_.gpu_resident_codec_parity_checks += token_count_in_range;
        };
        compare_device_range(old_recent_begin, old_aging_end);
        compare_device_range(new_history_begin, new_history_end);
    }
}

void OscarLiveMixedReferenceCache::prepare_gpu(std::int32_t model_layer,
                                               cudaStream_t stream) {
    const std::size_t index = full_attention_bank_index(model_layer);
    const auto& cache = *layers_[index];
    cache.validate();
    const std::uint32_t context = cache.context_tokens();
    const std::uint32_t prefix = std::min(context, ::ninfer::kOscarMixedPrefixTokens);
    const std::uint32_t recent_begin = live_recent_begin(context);
    const std::uint32_t historical = recent_begin - prefix;
    const std::uint32_t recent = context - recent_begin;
    if (context == 0 || context > max_context_ || prefix > 64 || historical > max_context_ ||
        recent > 256) {
        throw std::logic_error("OSCAR GPU staging cache extent is invalid");
    }

    const std::size_t row_values = kLiveRowValues;
    const std::size_t code_bytes = ::ninfer::ops::kOscarInt2G128CodeBytes;
    const std::size_t metadata_items = ::ninfer::ops::kOscarInt2G128MetadataItems;
    std::vector<std::uint16_t> prefix_k(static_cast<std::size_t>(prefix) * row_values);
    std::vector<std::uint16_t> prefix_v(prefix_k.size());
    std::vector<std::uint8_t> historical_k(
        static_cast<std::size_t>(historical) * ::ninfer::kOscarMixedKVHeads * code_bytes);
    std::vector<std::uint8_t> historical_v(historical_k.size());
    std::vector<float> historical_k_metadata(
        static_cast<std::size_t>(historical) * ::ninfer::kOscarMixedKVHeads * metadata_items);
    std::vector<float> historical_v_metadata(historical_k_metadata.size());
    std::vector<std::uint16_t> recent_k(static_cast<std::size_t>(recent) * row_values);
    std::vector<std::uint16_t> recent_v(recent_k.size());

    for (std::uint32_t logical = 0; logical < context; ++logical) {
        const auto resolved = cache.resolve(logical);
        const auto& page = cache.pages().at(resolved.page_index);
        const auto& slot = resolved.metadata;
        if (page.metadata.k_storage != page.metadata.v_storage ||
            page.metadata.layout_version != ::ninfer::kOscarMixedLayoutVersion ||
            page.metadata.group_size != ::ninfer::ops::kOscarInt2G128GroupSize ||
            slot.logical_token_begin != logical || slot.logical_token_end != logical + 1U) {
            throw std::logic_error("OSCAR GPU staging encountered invalid page metadata");
        }
        if (logical < prefix || logical >= recent_begin) {
            if (page.metadata.k_storage != ::ninfer::OscarMixedStorageType::BFloat16 ||
                !std::holds_alternative<::ninfer::OscarMixedBFloat16PageStorage>(page.storage)) {
                throw std::logic_error("OSCAR GPU staging BF16 tier is not typed BF16");
            }
            const auto& storage = std::get<::ninfer::OscarMixedBFloat16PageStorage>(page.storage);
            const std::size_t source = static_cast<std::size_t>(slot.page_offset) * row_values;
            const std::size_t destination_token = logical < prefix ? logical : logical - recent_begin;
            const std::size_t destination = destination_token * row_values;
            auto& out_k = logical < prefix ? prefix_k : recent_k;
            auto& out_v = logical < prefix ? prefix_v : recent_v;
            std::copy_n(storage.k.data() + source, row_values, out_k.data() + destination);
            std::copy_n(storage.v.data() + source, row_values, out_v.data() + destination);
        } else {
            if (page.metadata.k_storage != ::ninfer::OscarMixedStorageType::OscarInt2G128 ||
                !std::holds_alternative<::ninfer::OscarMixedInt2G128PageStorage>(page.storage)) {
                throw std::logic_error("OSCAR GPU staging historical tier is not typed INT2");
            }
            const auto& storage = std::get<::ninfer::OscarMixedInt2G128PageStorage>(page.storage);
            const std::size_t source_row = static_cast<std::size_t>(slot.page_offset) *
                                            ::ninfer::kOscarMixedKVHeads;
            const std::size_t destination_row = static_cast<std::size_t>(logical - prefix) *
                                                ::ninfer::kOscarMixedKVHeads;
            std::copy_n(storage.k_packed.data() + source_row * code_bytes,
                        ::ninfer::kOscarMixedKVHeads * code_bytes,
                        historical_k.data() + destination_row * code_bytes);
            std::copy_n(storage.v_packed.data() + source_row * code_bytes,
                        ::ninfer::kOscarMixedKVHeads * code_bytes,
                        historical_v.data() + destination_row * code_bytes);
            std::copy_n(storage.k_scales_zeros.data() + source_row * metadata_items,
                        ::ninfer::kOscarMixedKVHeads * metadata_items,
                        historical_k_metadata.data() + destination_row * metadata_items);
            std::copy_n(storage.v_scales_zeros.data() + source_row * metadata_items,
                        ::ninfer::kOscarMixedKVHeads * metadata_items,
                        historical_v_metadata.data() + destination_row * metadata_items);
        }
    }

    const auto ensure = [&](DeviceBuffer& buffer, std::size_t bytes) {
        if (buffer.bytes == 0) {
            buffer = DeviceBuffer(bytes);
        } else if (buffer.bytes < bytes) {
            throw std::logic_error("OSCAR GPU staging buffer capacity changed");
        }
    };
    ensure(gpu_prefix_k_, 64ULL * row_values * sizeof(std::uint16_t));
    ensure(gpu_prefix_v_, 64ULL * row_values * sizeof(std::uint16_t));
    ensure(gpu_historical_k_, static_cast<std::size_t>(max_context_) *
                                  ::ninfer::kOscarMixedKVHeads * code_bytes);
    ensure(gpu_historical_v_, static_cast<std::size_t>(max_context_) *
                                  ::ninfer::kOscarMixedKVHeads * code_bytes);
    ensure(gpu_historical_k_metadata_, static_cast<std::size_t>(max_context_) *
                                             ::ninfer::kOscarMixedKVHeads * metadata_items *
                                             sizeof(float));
    ensure(gpu_historical_v_metadata_, static_cast<std::size_t>(max_context_) *
                                             ::ninfer::kOscarMixedKVHeads * metadata_items *
                                             sizeof(float));
    ensure(gpu_recent_k_, 256ULL * row_values * sizeof(std::uint16_t));
    ensure(gpu_recent_v_, 256ULL * row_values * sizeof(std::uint16_t));
    ensure(gpu_scores_, static_cast<std::size_t>(max_context_) * ::ninfer::kOscarMixedQueryHeads *
                             sizeof(float));
    ensure(gpu_softmax_, static_cast<std::size_t>(max_context_) * ::ninfer::kOscarMixedQueryHeads *
                              sizeof(float));

    const auto start = oscar_d4_profile_enabled() ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
    gpu_prefix_k_.copy_from_host(prefix_k.data(), prefix_k.size() * sizeof(std::uint16_t));
    gpu_prefix_v_.copy_from_host(prefix_v.data(), prefix_v.size() * sizeof(std::uint16_t));
    gpu_historical_k_.copy_from_host(historical_k.data(), historical_k.size());
    gpu_historical_v_.copy_from_host(historical_v.data(), historical_v.size());
    gpu_historical_k_metadata_.copy_from_host(historical_k_metadata.data(),
                                              historical_k_metadata.size() * sizeof(float));
    gpu_historical_v_metadata_.copy_from_host(historical_v_metadata.data(),
                                              historical_v_metadata.size() * sizeof(float));
    gpu_recent_k_.copy_from_host(recent_k.data(), recent_k.size() * sizeof(std::uint16_t));
    gpu_recent_v_.copy_from_host(recent_v.data(), recent_v.size() * sizeof(std::uint16_t));
    (void)stream;
    if (oscar_d4_profile_enabled()) {
        const std::size_t bytes = prefix_k.size() * sizeof(std::uint16_t) +
                                  prefix_v.size() * sizeof(std::uint16_t) + historical_k.size() +
                                  historical_v.size() +
                                  historical_k_metadata.size() * sizeof(float) +
                                  historical_v_metadata.size() * sizeof(float) +
                                  recent_k.size() * sizeof(std::uint16_t) +
                                  recent_v.size() * sizeof(std::uint16_t);
        profile_.gpu_cache_staging_us += oscar_elapsed_us(start, std::chrono::steady_clock::now());
        profile_.gpu_cache_staging_bytes += bytes;
        ++profile_.gpu_cache_staging_calls;
    }
    gpu_staged_layer_ = model_layer;
    gpu_staged_context_ = context;
    gpu_staged_prefix_ = prefix;
    gpu_staged_historical_ = historical;
    gpu_staged_recent_ = recent;
}

void OscarLiveMixedReferenceCache::attention_gpu_batch(
    std::int32_t model_layer, std::uint32_t query_start, const Tensor& q_rotated,
    const Tensor& rotated_output, std::uint32_t query_count, cudaStream_t stream) {
    if (!gpu_resident_enabled_) {
        throw std::logic_error("OSCAR batched prefill requires the resident GPU cache");
    }
    if (query_count == 0 || query_count > gpu_prefill_query_block_size_ ||
        query_count > 64U) {
        throw std::invalid_argument("OSCAR batched prefill query count is invalid");
    }
    const auto valid = [query_count](const Tensor& tensor, const char* label) {
        if (tensor.dtype != DType::FP32 || tensor.data == nullptr || !tensor.is_contiguous() ||
            tensor.ne[0] != ::ninfer::kOscarMixedHeadDim ||
            tensor.ne[1] != ::ninfer::kOscarMixedQueryHeads ||
            tensor.ne[2] != static_cast<std::int32_t>(query_count) || tensor.ne[3] != 1) {
            throw std::invalid_argument(std::string("OSCAR batched GPU ") + label +
                                        " must be contiguous FP32 [256,24,Q]");
        }
    };
    valid(q_rotated, "query");
    valid(rotated_output, "output");
    const std::size_t index = full_attention_bank_index(model_layer);
    const auto& resident = gpu_resident_layers_[index];
    if (resident.context == 0 || query_start > resident.context ||
        query_count > resident.context - query_start) {
        throw std::logic_error("OSCAR batched prefill query range is outside the cache");
    }
    const std::uint32_t prefix =
        std::min(resident.context, ::ninfer::kOscarMixedPrefixTokens);
    const std::uint32_t recent_begin = live_recent_begin(resident.context);
    const std::uint32_t historical = recent_begin - prefix;
    const std::uint32_t recent = resident.context - recent_begin;
    if (prefix + historical + recent != resident.context || recent > 256U) {
        throw std::logic_error("OSCAR batched prefill resident tier split is invalid");
    }
    const ::ninfer::ops::detail::OscarInt2G128ResidentCacheView view{
        static_cast<std::uint16_t*>(resident.prefix_k.p),
        static_cast<std::uint16_t*>(resident.prefix_v.p),
        static_cast<std::uint8_t*>(resident.historical_k.p),
        static_cast<std::uint8_t*>(resident.historical_v.p),
        static_cast<float*>(resident.historical_k_metadata.p),
        static_cast<float*>(resident.historical_v_metadata.p),
        static_cast<std::uint16_t*>(resident.recent_k.p),
        static_cast<std::uint16_t*>(resident.recent_v.p),
         static_cast<std::int32_t>(max_context_)};
    if (gpu_fused_enabled_) {
        ::ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_fused_batch_ring(
            static_cast<const float*>(q_rotated.data), view.prefix_k_bf16, view.prefix_v_bf16,
            static_cast<std::int32_t>(prefix), view.historical_k_packed,
            view.historical_k_metadata, view.historical_v_packed, view.historical_v_metadata,
            static_cast<std::int32_t>(historical), view.recent_k_bf16, view.recent_v_bf16,
            static_cast<std::int32_t>(recent), static_cast<std::int32_t>(resident.recent_head),
            static_cast<std::int32_t>(query_start), static_cast<std::int32_t>(query_count),
            0.0625F, static_cast<float*>(rotated_output.data), stream);
        ++profile_.gpu_attention_batches;
        profile_.gpu_attention_calls += query_count;
        profile_.gpu_attention_kernel_launches += 1;
        profile_.gpu_fused_query_tiles += (query_count + 3U) / 4U;
        profile_.gpu_fused_kv_tiles +=
            ((query_start + query_count + 31U) / 32U) * ((query_count + 3U) / 4U);
        return;
    }
    ::ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_batch_ring(
        static_cast<const float*>(q_rotated.data), view.prefix_k_bf16, view.prefix_v_bf16,
        static_cast<std::int32_t>(prefix), view.historical_k_packed,
        view.historical_k_metadata, view.historical_v_packed, view.historical_v_metadata,
        static_cast<std::int32_t>(historical), view.recent_k_bf16, view.recent_v_bf16,
        static_cast<std::int32_t>(recent), static_cast<std::int32_t>(resident.recent_head),
        static_cast<std::int32_t>(query_start), static_cast<std::int32_t>(query_count),
        static_cast<std::int32_t>(max_context_), 0.0625F,
        static_cast<float*>(gpu_resident_scores_.p),
        static_cast<float*>(gpu_resident_softmax_.p), static_cast<float*>(rotated_output.data),
        stream);
    ++profile_.gpu_attention_batches;
    profile_.gpu_attention_calls += query_count;
    profile_.gpu_attention_kernel_launches += 3;
}

void OscarLiveMixedReferenceCache::attention_gpu(std::int32_t model_layer,
                                                 std::uint32_t query_token,
                                                 const Tensor& q_rotated,
                                                 const Tensor& rotated_output,
                                                 cudaStream_t stream) {
    const auto valid = [](const Tensor& tensor, const char* label) {
        if (tensor.dtype != DType::FP32 || tensor.data == nullptr || !tensor.is_contiguous() ||
            tensor.ne[0] != ::ninfer::kOscarMixedHeadDim ||
            tensor.ne[1] != ::ninfer::kOscarMixedQueryHeads || tensor.ne[2] != 1 ||
            tensor.ne[3] != 1) {
            throw std::invalid_argument(std::string("OSCAR GPU ") + label +
                                        " must be contiguous FP32 [256,24,1]");
        }
    };
    valid(q_rotated, "query");
    valid(rotated_output, "output");
    if (gpu_resident_enabled_) {
        const std::size_t index = full_attention_bank_index(model_layer);
        const auto& resident = gpu_resident_layers_[index];
        if (query_token >= resident.context || resident.context == 0) {
            throw std::logic_error("OSCAR resident attention query is outside the cache");
        }
        const std::uint32_t prefix =
            std::min(resident.context, ::ninfer::kOscarMixedPrefixTokens);
        const std::uint32_t recent_begin = live_recent_begin(resident.context);
        const std::uint32_t historical = recent_begin - prefix;
        const std::uint32_t history_end = prefix + historical;
        const std::uint32_t recent = resident.context - history_end;
        const std::uint32_t visible = query_token + 1U;
        const std::uint32_t visible_prefix = std::min(visible, prefix);
        const std::uint32_t visible_historical =
            visible > prefix ? std::min(visible - prefix, historical) : 0U;
        const std::uint32_t visible_history_end = prefix + visible_historical;
        const std::uint32_t visible_recent =
            visible > visible_history_end ? std::min(visible - visible_history_end,
                                                      resident.context - history_end)
                                           : 0U;
        if (visible_prefix + visible_historical + visible_recent != visible ||
            visible_recent > ::ninfer::kOscarMixedRecentTokens) {
            throw std::logic_error("OSCAR resident attention visible tier split is invalid");
        }
        const ::ninfer::ops::detail::OscarInt2G128ResidentCacheView view{
            static_cast<std::uint16_t*>(resident.prefix_k.p),
            static_cast<std::uint16_t*>(resident.prefix_v.p),
            static_cast<std::uint8_t*>(resident.historical_k.p),
            static_cast<std::uint8_t*>(resident.historical_v.p),
            static_cast<float*>(resident.historical_k_metadata.p),
            static_cast<float*>(resident.historical_v_metadata.p),
            static_cast<std::uint16_t*>(resident.recent_k.p),
            static_cast<std::uint16_t*>(resident.recent_v.p),
             static_cast<std::int32_t>(max_context_)};
        if (gpu_fused_enabled_) {
            ::ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_fused_decode_split_ring(
                static_cast<const float*>(q_rotated.data), view.prefix_k_bf16, view.prefix_v_bf16,
                static_cast<std::int32_t>(prefix), view.historical_k_packed,
                view.historical_k_metadata, view.historical_v_packed,
                view.historical_v_metadata, static_cast<std::int32_t>(historical),
                view.recent_k_bf16, view.recent_v_bf16,
                static_cast<std::int32_t>(recent), static_cast<std::int32_t>(resident.recent_head),
                static_cast<std::int32_t>(query_token),
                ::ninfer::ops::detail::kOscarMixedFusedDecodeAdaptiveSplits, 0.0625F,
                static_cast<float*>(gpu_resident_fused_decode_workspace_.p),
                static_cast<float*>(rotated_output.data), stream);
            ++profile_.gpu_attention_calls;
            ++profile_.gpu_attention_batches;
            profile_.gpu_attention_kernel_launches += 2;
            ++profile_.gpu_fused_query_tiles;
            profile_.gpu_fused_kv_tiles += (query_token + 64U) / 64U;
            return;
        }
        ::ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch_ring(
            static_cast<const float*>(q_rotated.data), view.prefix_k_bf16, view.prefix_v_bf16,
            static_cast<std::int32_t>(visible_prefix), view.historical_k_packed,
            view.historical_k_metadata, view.historical_v_packed, view.historical_v_metadata,
            static_cast<std::int32_t>(visible_historical), view.recent_k_bf16, view.recent_v_bf16,
            static_cast<std::int32_t>(visible_recent), static_cast<std::int32_t>(resident.recent_head),
            0.0625F, static_cast<float*>(gpu_resident_scores_.p),
            static_cast<float*>(gpu_resident_softmax_.p), static_cast<float*>(rotated_output.data),
            stream);
        ++profile_.gpu_attention_calls;
        ++profile_.gpu_attention_batches;
        profile_.gpu_attention_kernel_launches += 3;
        return;
    }
    if (gpu_staged_layer_ != model_layer || query_token >= gpu_staged_context_) {
        throw std::logic_error("OSCAR GPU attention was called without a matching staged cache");
    }
    const std::uint32_t visible = query_token + 1U;
    const std::uint32_t prefix = std::min(visible, gpu_staged_prefix_);
    const std::uint32_t historical = visible > gpu_staged_prefix_
                                          ? std::min(visible - gpu_staged_prefix_,
                                                      gpu_staged_historical_)
                                          : 0U;
    const std::uint32_t history_end = gpu_staged_prefix_ + gpu_staged_historical_;
    const std::uint32_t recent = visible > history_end
                                     ? std::min(visible - history_end, gpu_staged_recent_)
                                     : 0U;
    if (prefix + historical + recent != visible || prefix > 64 || historical > max_context_ ||
        recent > 256) {
        throw std::logic_error("OSCAR GPU attention visible tier split is invalid");
    }
    ::ninfer::ops::detail::oscar_int2_g128_mixed_attention_launch(
        static_cast<const float*>(q_rotated.data),
        static_cast<const std::uint16_t*>(gpu_prefix_k_.p),
        static_cast<const std::uint16_t*>(gpu_prefix_v_.p), static_cast<std::int32_t>(prefix),
        static_cast<const std::uint8_t*>(gpu_historical_k_.p),
        static_cast<const float*>(gpu_historical_k_metadata_.p),
        static_cast<const std::uint8_t*>(gpu_historical_v_.p),
        static_cast<const float*>(gpu_historical_v_metadata_.p),
        static_cast<std::int32_t>(historical), static_cast<const std::uint16_t*>(gpu_recent_k_.p),
        static_cast<const std::uint16_t*>(gpu_recent_v_.p), static_cast<std::int32_t>(recent),
        0.0625F, static_cast<float*>(gpu_scores_.p), static_cast<float*>(gpu_softmax_.p),
        static_cast<float*>(rotated_output.data), stream);
    ++profile_.gpu_attention_calls;
    ++profile_.gpu_attention_batches;
    profile_.gpu_attention_kernel_launches += 3;
}

void OscarLiveMixedReferenceCache::append(std::int32_t model_layer, std::uint32_t logical_token,
                                          std::span<const std::uint16_t> k_bf16,
                                          std::span<const std::uint16_t> v_bf16) {
    if (k_bf16.size() != kLiveRowValues || v_bf16.size() != kLiveRowValues) {
        throw std::invalid_argument("OSCAR live mixed cache K/V row shape mismatch");
    }
    const std::size_t index = full_attention_bank_index(model_layer);
    auto& layer_cache = *layers_[index];
    const bool profiling = oscar_d4_profile_enabled();
    if (profiling) { ++profile_.append_calls; }
    const std::uint32_t next_context = layer_cache.context_tokens() + 1U;
    const std::uint32_t recent_begin = live_recent_begin(next_context);
    const bool will_age = recent_begin > ::ninfer::kOscarMixedPrefixTokens &&
                          recent_begin - 1U < layer_cache.context_tokens();
    const std::uint32_t aging_target = will_age ? recent_begin - 1U : 0U;
    std::array<std::uint16_t, kLiveRowValues> aging_k_bf16{};
    std::array<std::uint16_t, kLiveRowValues> aging_v_bf16{};
    std::array<::ninfer::ops::OscarInt2G128EncodedRow, ::ninfer::kOscarMixedKVHeads> expected_k{};
    std::array<::ninfer::ops::OscarInt2G128EncodedRow, ::ninfer::kOscarMixedKVHeads> expected_v{};
    const auto aging_encode_start = profiling ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    if (will_age) {
        const auto resolved = layer_cache.resolve(aging_target);
        const auto& page = layer_cache.pages().at(resolved.page_index);
        if (page.metadata.k_storage != ::ninfer::OscarMixedStorageType::BFloat16 ||
            !std::holds_alternative<::ninfer::OscarMixedBFloat16PageStorage>(page.storage)) {
            throw std::logic_error("OSCAR live aging target is not in BF16 recent storage");
        }
        const auto& storage = std::get<::ninfer::OscarMixedBFloat16PageStorage>(page.storage);
        const std::size_t begin = static_cast<std::size_t>(resolved.metadata.page_offset) *
                                  kLiveRowValues;
        std::copy_n(storage.k.data() + begin, kLiveRowValues, aging_k_bf16.begin());
        std::copy_n(storage.v.data() + begin, kLiveRowValues, aging_v_bf16.begin());
        std::array<float, ::ninfer::kOscarMixedHeadDim> k_values{};
        std::array<float, ::ninfer::kOscarMixedHeadDim> v_values{};
        for (std::uint32_t head = 0; head < ::ninfer::kOscarMixedKVHeads; ++head) {
            const std::size_t row_begin = static_cast<std::size_t>(head) *
                                          ::ninfer::kOscarMixedHeadDim;
            for (std::uint32_t dimension = 0; dimension < ::ninfer::kOscarMixedHeadDim; ++dimension) {
                k_values[dimension] = live_bf16_to_float(aging_k_bf16[row_begin + dimension]);
                v_values[dimension] = live_bf16_to_float(aging_v_bf16[row_begin + dimension]);
            }
            expected_k[head] = ::ninfer::ops::oscar_int2_g128_encode(
                k_values.data(), static_cast<int>(::ninfer::kOscarMixedHeadDim), 0.96F);
            expected_v[head] = ::ninfer::ops::oscar_int2_g128_encode(
                v_values.data(), static_cast<int>(::ninfer::kOscarMixedHeadDim), 0.92F);
        }
    }
    if (profiling && will_age) {
        ++profile_.aging_events;
        profile_.aging_reference_encode_us +=
            oscar_elapsed_us(aging_encode_start, std::chrono::steady_clock::now());
    }
    const auto aging_conversion_start = profiling && will_age
                                            ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
    layer_cache.append(logical_token, k_bf16, v_bf16);
    if (profiling && will_age) {
        profile_.aging_us +=
            oscar_elapsed_us(aging_conversion_start, std::chrono::steady_clock::now());
    }
    if (will_age) {
        const auto parity_start = profiling ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        for (std::uint32_t head = 0; head < ::ninfer::kOscarMixedKVHeads; ++head) {
            const auto& actual_k = layer_cache.historical_k(aging_target, head);
            const auto& actual_v = layer_cache.historical_v(aging_target, head);
            if (actual_k.packed != expected_k[head].packed ||
                actual_k.scales_zeros != expected_k[head].scales_zeros ||
                actual_v.packed != expected_v[head].packed ||
                actual_v.scales_zeros != expected_v[head].scales_zeros) {
                throw std::logic_error("OSCAR live aging differs from standalone G128 encoding");
            }
        }
        if (profiling) {
            profile_.aging_parity_check_us +=
                oscar_elapsed_us(parity_start, std::chrono::steady_clock::now());
        }
        ++aging_parity_checks_;
    }
    dispatch_seen_[index] = true;
}

const ::ninfer::OscarMixedAgingLayerCache& OscarLiveMixedReferenceCache::layer(
    std::int32_t model_layer) const {
    return *layers_[full_attention_bank_index(model_layer)];
}

::ninfer::OscarMixedCacheAccounting OscarLiveMixedReferenceCache::accounting() const {
    ::ninfer::OscarMixedCacheAccounting result;
    result.full_attention_layers = static_cast<std::uint32_t>(layers_.size());
    if (gpu_resident_enabled_) {
        const std::uint32_t context = gpu_resident_layers_[0].context;
        const std::uint32_t prefix = std::min(context, ::ninfer::kOscarMixedPrefixTokens);
        const std::uint32_t recent_begin = live_recent_begin(context);
        const std::uint32_t historical = recent_begin - prefix;
        const std::uint32_t recent = context - recent_begin;
        const std::uint64_t layer_count = static_cast<std::uint64_t>(layers_.size());
        const auto pages_for = [](std::uint32_t tokens) -> std::uint64_t {
            return (static_cast<std::uint64_t>(tokens) + ::ninfer::kOscarMixedPageTokens - 1U) /
                   ::ninfer::kOscarMixedPageTokens;
        };
        const std::uint64_t prefix_pages = pages_for(prefix);
        const std::uint64_t historical_pages = pages_for(historical);
        const std::uint64_t recent_pages = pages_for(recent);
        constexpr std::uint64_t bf16_page_bytes =
            static_cast<std::uint64_t>(::ninfer::kOscarMixedPageTokens) *
            ::ninfer::kOscarMixedKVHeads * ::ninfer::kOscarMixedHeadDim * sizeof(std::uint16_t) * 2U;
        constexpr std::uint64_t int2_payload_page_bytes =
            static_cast<std::uint64_t>(::ninfer::kOscarMixedPageTokens) *
            ::ninfer::kOscarMixedKVHeads * ::ninfer::ops::kOscarInt2G128CodeBytes * 2U;
        constexpr std::uint64_t int2_metadata_page_bytes =
            static_cast<std::uint64_t>(::ninfer::kOscarMixedPageTokens) *
            ::ninfer::kOscarMixedKVHeads * ::ninfer::ops::kOscarInt2G128MetadataItems *
            sizeof(float) * 2U;
        result.context_tokens = context;
        result.prefix_tokens = prefix;
        result.historical_tokens = historical;
        result.recent_tokens = recent;
        result.page_count = (prefix_pages + historical_pages + recent_pages) * layer_count;
        result.bf16_page_count = (prefix_pages + recent_pages) * layer_count;
        result.int2_page_count = historical_pages * layer_count;
        result.physical_bf16_bytes =
            (prefix_pages + recent_pages) * bf16_page_bytes * layer_count;
        result.physical_int2_payload_bytes =
            historical_pages * int2_payload_page_bytes * layer_count;
        result.physical_int2_metadata_bytes =
            historical_pages * int2_metadata_page_bytes * layer_count;
        result.page_header_bytes = result.page_count * sizeof(::ninfer::OscarMixedPageMetadata);
        result.slot_table_bytes = static_cast<std::uint64_t>(context) *
                                  sizeof(::ninfer::OscarMixedSlotMetadata) * layer_count;
        result.logical_bf16_bytes = static_cast<std::uint64_t>(prefix + recent) *
                                    ::ninfer::kOscarMixedKVHeads *
                                    ::ninfer::kOscarMixedHeadDim * sizeof(std::uint16_t) * 2U *
                                    layer_count;
        result.logical_int2_payload_bytes =
            static_cast<std::uint64_t>(historical) * ::ninfer::kOscarMixedKVHeads *
            ::ninfer::ops::kOscarInt2G128CodeBytes * 2U * layer_count;
        result.logical_int2_metadata_bytes =
            static_cast<std::uint64_t>(historical) * ::ninfer::kOscarMixedKVHeads *
            ::ninfer::ops::kOscarInt2G128MetadataItems * sizeof(float) * 2U * layer_count;
        result.mixed_total_bytes = result.physical_bf16_bytes + result.physical_int2_payload_bytes +
                                   result.physical_int2_metadata_bytes + result.page_header_bytes +
                                   result.slot_table_bytes;
        result.historical_bulk_total_bytes = result.physical_int2_payload_bytes +
                                             result.physical_int2_metadata_bytes +
                                             result.int2_page_count *
                                                 sizeof(::ninfer::OscarMixedPageMetadata) +
                                             static_cast<std::uint64_t>(historical) *
                                                 sizeof(::ninfer::OscarMixedSlotMetadata) *
                                                 layer_count;
        result.logical_value_count = static_cast<std::uint64_t>(context) *
                                     ::ninfer::kOscarMixedKVHeads *
                                     ::ninfer::kOscarMixedHeadDim * 2U * layer_count;
        result.raw_int2_bytes_per_value =
            static_cast<double>(::ninfer::ops::kOscarInt2G128CodeBytes +
                                ::ninfer::ops::kOscarInt2G128MetadataItems * sizeof(float)) /
            ::ninfer::ops::kOscarInt2G128HeadDim;
        if (historical != 0) {
            result.historical_bulk_bytes_per_value =
                static_cast<double>(result.historical_bulk_total_bytes) /
                (static_cast<double>(historical) * ::ninfer::kOscarMixedKVHeads *
                 ::ninfer::kOscarMixedHeadDim * 2.0 * layer_count);
            result.historical_bulk_bits_per_value = result.historical_bulk_bytes_per_value * 8.0;
        }
        if (result.logical_value_count != 0) {
            result.mixed_bytes_per_value = static_cast<double>(result.mixed_total_bytes) /
                                           static_cast<double>(result.logical_value_count);
            result.mixed_bits_per_value = result.mixed_bytes_per_value * 8.0;
        }
        return result;
    }
    for (const auto& layer_cache : layers_) {
        // During a real layer sweep the current full-attention layer is populated before later
        // full-attention layers execute. Empty layers are valid transient state and must not make
        // layer-3 telemetry look like a cache-coordinate failure.
        if (layer_cache->context_tokens() == 0) { continue; }
        const auto layer_accounting = layer_cache->accounting();
        if (result.context_tokens == 0) {
            result.context_tokens = layer_accounting.context_tokens;
            result.prefix_tokens = layer_accounting.prefix_tokens;
            result.historical_tokens = layer_accounting.historical_tokens;
            result.recent_tokens = layer_accounting.recent_tokens;
        } else if (result.context_tokens != layer_accounting.context_tokens ||
                   result.prefix_tokens != layer_accounting.prefix_tokens ||
                   result.historical_tokens != layer_accounting.historical_tokens ||
                   result.recent_tokens != layer_accounting.recent_tokens) {
            throw std::logic_error("OSCAR live mixed cache layers have different logical extents");
        }
        result.page_count += layer_accounting.page_count;
        result.bf16_page_count += layer_accounting.bf16_page_count;
        result.int2_page_count += layer_accounting.int2_page_count;
        result.logical_bf16_bytes += layer_accounting.logical_bf16_bytes;
        result.logical_int2_payload_bytes += layer_accounting.logical_int2_payload_bytes;
        result.logical_int2_metadata_bytes += layer_accounting.logical_int2_metadata_bytes;
        result.physical_bf16_bytes += layer_accounting.physical_bf16_bytes;
        result.physical_int2_payload_bytes += layer_accounting.physical_int2_payload_bytes;
        result.physical_int2_metadata_bytes += layer_accounting.physical_int2_metadata_bytes;
        result.page_header_bytes += layer_accounting.page_header_bytes;
        result.slot_table_bytes += layer_accounting.slot_table_bytes;
        result.mixed_total_bytes += layer_accounting.mixed_total_bytes;
        result.historical_bulk_total_bytes += layer_accounting.historical_bulk_total_bytes;
        result.logical_value_count += layer_accounting.logical_value_count;
    }
    if (result.logical_value_count != 0) {
        result.mixed_bytes_per_value = static_cast<double>(result.mixed_total_bytes) /
                                       static_cast<double>(result.logical_value_count);
        result.mixed_bits_per_value = result.mixed_bytes_per_value * 8.0;
    }
    if (result.historical_tokens != 0) {
        const double values = static_cast<double>(result.historical_tokens) * kLiveRowValues *
                              layers_.size();
        result.historical_bulk_bytes_per_value =
            static_cast<double>(result.historical_bulk_total_bytes) / values;
        result.historical_bulk_bits_per_value = result.historical_bulk_bytes_per_value * 8.0;
    }
    result.raw_int2_bytes_per_value =
        (static_cast<double>(::ninfer::ops::kOscarInt2G128CodeBytes) +
         static_cast<double>(::ninfer::ops::kOscarInt2G128MetadataItems) * sizeof(float)) /
        static_cast<double>(::ninfer::ops::kOscarInt2G128HeadDim);
    return result;
}

::ninfer::OscarMixedAttentionTrace OscarLiveMixedReferenceCache::attention(
    std::int32_t model_layer, std::uint32_t query_token, std::span<const float> q_original) {
    const std::size_t index = full_attention_bank_index(model_layer);
    dispatch_seen_[index] = true;
    const auto& cache = *layers_[index];
    ::ninfer::OscarMixedAttentionReader reader(cache, q_original,
                                               rotations_->k_matrix_host(model_layer),
                                               rotations_->v_matrix_host(model_layer));
    auto trace = reader.read(query_token);
    if (oscar_d4_profile_enabled()) {
        ++profile_.attention_calls;
        profile_.reader_q_rotation_us += trace.q_rotation_us;
        profile_.int2_k_decode_us += trace.int2_k_decode_us;
        profile_.qk_us += trace.qk_us;
        profile_.softmax_us += trace.softmax_us;
        profile_.int2_v_decode_us += trace.int2_v_decode_us;
        profile_.av_us += trace.av_us;
        profile_.rv_inverse_us += trace.rv_inverse_us;
        profile_.reader_total_us += trace.total_us;
    }
    write_tap_if_selected(model_layer, query_token, q_original, trace);
    if (model_layer == 3 && query_token + 1U == cache.context_tokens()) {
        // Layer execution is serialized; later full-attention layers still have the prior decode
        // extent while layer 3 is already reading the newly appended row. Use this validated
        // layer's accounting for per-invocation telemetry.
        const auto counts = cache.accounting();
        std::cerr << "OSCAR live telemetry: oscar_calibrated=true"
                  << " asset_identity=" << rotations_->asset_identity()
                  << " asset_hash=" << rotations_->asset_hash()
                  << " group_size=128 k_clip=0.96 v_clip=0.92 prefix_length=64 recent_length=256"
                  << " prefix_token_count=" << counts.prefix_tokens
                  << " historical_token_count=" << counts.historical_tokens
                  << " recent_token_count=" << counts.recent_tokens
                  << " int2_payload_bytes=" << counts.physical_int2_payload_bytes
                  << " int2_metadata_bytes=" << counts.physical_int2_metadata_bytes
                  << " legacy_q2_dispatched=false bf16_historical_shadow=false fallback=false"
                  << " selected_layout=mixed-bf16-prefix-oscar-int2-g128-bf16-recent"
                  << " selected_attention_implementation="
                  << (live_int2_gpu_mode_enabled() ? "oscar-mixed-reference-cpu-oracle-only"
                                                    : "oscar-mixed-reference-cpu")
                  << '\n';
    }
    if (model_layer == 63 && query_token + 1U == cache.context_tokens()) {
        std::cerr << "OSCAR live dispatch coverage: full_layer_dispatch_bitmap=";
        for (const bool seen : dispatch_seen_) { std::cerr << (seen ? '1' : '0'); }
        std::cerr << " aging_codec_parity_checks=" << aging_parity_checks_
                  << " gdn_dispatches=0 legacy_q2_dispatches=0 bf16_history_shadow=false\n";
        if (!std::all_of(dispatch_seen_.begin(), dispatch_seen_.end(), [](bool value) { return value; })) {
            throw std::logic_error("OSCAR live full-attention dispatch coverage is incomplete");
        }
    }
    if (model_layer == 63 && query_token + 1U == cache.context_tokens() &&
        oscar_d4_profile_enabled()) {
        const auto& p = profile_;
        std::cerr << "OSCAR D4.1 profile: attention_calls=" << p.attention_calls
                  << " append_calls=" << p.append_calls << " aging_events=" << p.aging_events
                  << " qkv_rotation_us=" << p.qkv_rotation_us
                  << " aging_us=" << p.aging_us
                  << " aging_reference_encode_us=" << p.aging_reference_encode_us
                  << " aging_parity_check_us=" << p.aging_parity_check_us
                  << " reader_q_rotation_us=" << p.reader_q_rotation_us
                  << " int2_k_decode_us=" << p.int2_k_decode_us
                  << " qk_us=" << p.qk_us << " softmax_us=" << p.softmax_us
                  << " int2_v_decode_us=" << p.int2_v_decode_us << " av_us=" << p.av_us
                  << " rv_inverse_us=" << p.rv_inverse_us
                  << " reader_total_us=" << p.reader_total_us
                  << " full_attention_us=" << p.full_attention_us << '\n';
    }
    return trace;
}

void OscarLiveMixedReferenceCache::record_qkv_rotation_us(double value) noexcept {
    profile_.qkv_rotation_us += value;
}

void OscarLiveMixedReferenceCache::record_full_attention_us(double value) noexcept {
    profile_.full_attention_us += value;
}

void OscarLiveMixedReferenceCache::record_gpu_mixed_kernel_us(double value) noexcept {
    profile_.gpu_mixed_kernel_us += value;
    if (gpu_fused_enabled_) { profile_.gpu_fused_kernel_us += value; }
}

void OscarLiveMixedReferenceCache::record_gpu_fused_kernel_us(double value) noexcept {
    profile_.gpu_fused_kernel_us += value;
}

void OscarLiveMixedReferenceCache::record_gpu_recovery_us(double value) noexcept {
    profile_.gpu_recovery_us += value;
}

void OscarLiveMixedReferenceCache::record_gpu_phase_full_attention_us(
    bool prefill, double value) noexcept {
    if (prefill) {
        profile_.gpu_prefill_full_attention_us += value;
    } else {
        profile_.gpu_decode_full_attention_us += value;
    }
}

void OscarLiveMixedReferenceCache::record_gpu_incremental_host_device_bytes(
    std::uint64_t value) noexcept {
    profile_.gpu_incremental_host_device_bytes += value;
}

void OscarLiveMixedReferenceCache::record_gpu_prefill_batch(std::uint32_t query_count) noexcept {
    profile_.gpu_prefill_batches += 1;
    profile_.gpu_prefill_queries += query_count;
}

void OscarLiveMixedReferenceCache::record_gpu_decode_batch(std::uint32_t query_count) noexcept {
    profile_.gpu_decode_batches += 1;
    profile_.gpu_decode_queries += query_count;
}

void OscarLiveMixedReferenceCache::refresh_live_reference_taps() {
    tap_root_.clear();
    tap_layers_.clear();
    tap_queries_.clear();
    const char* tap_root = std::getenv("NINFER_OSCAR_LIVE_REFERENCE_TAP_DIR");
    if (tap_root == nullptr || *tap_root == '\0') { return; }
    tap_root_ = std::filesystem::absolute(tap_root).lexically_normal();
    std::error_code error;
    std::filesystem::create_directories(tap_root_, error);
    if (error) {
        throw std::runtime_error("cannot create OSCAR live reference tap directory: " +
                                 error.message());
    }
    tap_layers_ = parse_live_list(std::getenv("NINFER_OSCAR_LIVE_REFERENCE_TAP_LAYERS"),
                                  "OSCAR live tap layer list");
    tap_queries_ = parse_live_list(std::getenv("NINFER_OSCAR_LIVE_REFERENCE_TAP_QUERIES"),
                                   "OSCAR live tap query list");
    if (tap_layers_.empty()) { tap_layers_.push_back(3); }
    if (tap_queries_.empty()) { tap_queries_.push_back(0); }
    for (const std::uint32_t layer : tap_layers_) {
        if (std::find(asset_.full_attention_layers.begin(), asset_.full_attention_layers.end(),
                      layer) == asset_.full_attention_layers.end()) {
            throw std::invalid_argument("OSCAR live tap layer is not full-attention");
        }
    }
}

void OscarLiveMixedReferenceCache::write_tap_if_selected(
    std::int32_t model_layer, std::uint32_t query_token, std::span<const float> q_original,
    const ::ninfer::OscarMixedAttentionTrace& trace) const {
    if (tap_root_.empty() || !list_contains(tap_layers_, static_cast<std::uint32_t>(model_layer)) ||
        !list_contains(tap_queries_, query_token)) {
        return;
    }
    if (q_original.size() != kLiveQueryValues || trace.logical_positions.size() != query_token + 1U ||
        trace.tiers.size() != query_token + 1U) {
        throw std::logic_error("OSCAR live reference tap trace shape mismatch");
    }
    const auto& cache = layer(model_layer);
    const std::filesystem::path path = tap_root_ /
        ("tap.layer_" + std::to_string(model_layer) + ".query_" +
         std::to_string(query_token) + ".bin");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot open OSCAR live reference tap: " + path.string()); }
    write_live_tap_value(output, kLiveTapMagic);
    write_live_tap_value(output, kLiveTapVersion);
    write_live_tap_value(output, static_cast<std::uint32_t>(model_layer));
    write_live_tap_value(output, query_token);
    write_live_tap_value(output, cache.context_tokens());
    write_live_tap_value(output, static_cast<std::uint32_t>(q_original.size()));
    write_live_tap_value(output, static_cast<std::uint32_t>(trace.logical_positions.size()));
    write_live_tap_string(output, asset_.asset_identity);
    write_live_tap_string(output, asset_.asset_manifest_sha256);
    write_live_tap_span(output, q_original);
    write_live_tap_span(output, std::span<const std::uint32_t>(trace.logical_positions));
    for (std::uint32_t logical = 0; logical <= query_token; ++logical) {
        const auto resolved = cache.resolve(logical);
        const auto& page = cache.pages().at(resolved.page_index);
        const auto& slot = resolved.metadata;
        std::uint8_t tier = static_cast<std::uint8_t>(trace.tiers[logical]);
        write_live_tap_value(output, tier);
        const std::uint8_t reserved[3] = {};
        output.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));
        if (tier == static_cast<std::uint8_t>(::ninfer::OscarMixedReadTier::ProtectedPrefixBFloat16) ||
            tier == static_cast<std::uint8_t>(::ninfer::OscarMixedReadTier::RecentBFloat16)) {
            const auto& storage = std::get<::ninfer::OscarMixedBFloat16PageStorage>(page.storage);
            const std::size_t row = static_cast<std::size_t>(slot.page_offset) * kOscarMixedKVHeads;
            write_live_tap_span(output, std::span<const std::uint16_t>(
                storage.k.data() + row * kOscarMixedHeadDim, kOscarMixedHeadDim));
            write_live_tap_span(output, std::span<const std::uint16_t>(
                storage.v.data() + row * kOscarMixedHeadDim, kOscarMixedHeadDim));
            for (std::uint32_t kv_head = 1; kv_head < kOscarMixedKVHeads; ++kv_head) {
                const std::size_t head_row = row + kv_head;
                write_live_tap_span(output, std::span<const std::uint16_t>(
                    storage.k.data() + head_row * kOscarMixedHeadDim, kOscarMixedHeadDim));
                write_live_tap_span(output, std::span<const std::uint16_t>(
                    storage.v.data() + head_row * kOscarMixedHeadDim, kOscarMixedHeadDim));
            }
        } else if (tier == static_cast<std::uint8_t>(::ninfer::OscarMixedReadTier::HistoricalOscarInt2G128)) {
            const auto& storage = std::get<::ninfer::OscarMixedInt2G128PageStorage>(page.storage);
            const std::size_t row = static_cast<std::size_t>(slot.page_offset) * kOscarMixedKVHeads;
            for (std::uint32_t kv_head = 0; kv_head < kOscarMixedKVHeads; ++kv_head) {
                const std::size_t head_row = row + kv_head;
                write_live_tap_span(output, std::span<const std::uint8_t>(
                    storage.k_packed.data() + head_row * ::ninfer::ops::kOscarInt2G128CodeBytes,
                    ::ninfer::ops::kOscarInt2G128CodeBytes));
                write_live_tap_span(output, std::span<const float>(
                    storage.k_scales_zeros.data() + head_row * ::ninfer::ops::kOscarInt2G128MetadataItems,
                    ::ninfer::ops::kOscarInt2G128MetadataItems));
                write_live_tap_span(output, std::span<const std::uint8_t>(
                    storage.v_packed.data() + head_row * ::ninfer::ops::kOscarInt2G128CodeBytes,
                    ::ninfer::ops::kOscarInt2G128CodeBytes));
                write_live_tap_span(output, std::span<const float>(
                    storage.v_scales_zeros.data() + head_row * ::ninfer::ops::kOscarInt2G128MetadataItems,
                    ::ninfer::ops::kOscarInt2G128MetadataItems));
            }
        } else {
            throw std::logic_error("OSCAR live reference tap encountered unknown tier");
        }
    }
    write_live_trace_vector(output, trace.rotated_q);
    write_live_trace_vector(output, trace.score_logits);
    write_live_trace_vector(output, trace.softmax);
    write_live_trace_vector(output, trace.rotated_av);
    write_live_trace_vector(output, trace.recovered_output);
    if (!output) { throw std::runtime_error("cannot finalize OSCAR live reference tap"); }
}

std::shared_ptr<OscarLiveMixedReferenceCache>
live_mixed_cache_for(const void* owner, std::uint32_t max_context,
                     std::shared_ptr<const OscarRotationSet> rotations) {
    if (owner == nullptr) { throw std::invalid_argument("OSCAR live mixed cache owner is null"); }
    if (!rotations) { throw std::invalid_argument("OSCAR live mixed cache rotations are null"); }
    static std::mutex mutex;
    static const void* current_owner = nullptr;
    static std::shared_ptr<OscarLiveMixedReferenceCache> current_cache;
    std::lock_guard lock(mutex);
    if (current_cache && current_owner == owner) {
        if (current_cache->max_context() != max_context) {
            // A new engine can reuse the PagedKVCache address with a different capacity after
            // the previous cache has been released. An active owner changing capacity is still
            // rejected, but an idle diagnostic cache may be replaced safely.
            if (current_cache.use_count() > 1) {
                throw std::logic_error("OSCAR live mixed cache exceeded configured capacity");
            }
            current_owner = nullptr;
            current_cache.reset();
        } else if (current_cache->gpu_resident_mode_enabled() !=
                   live_int2_gpu_resident_mode_enabled()) {
            // The owner address is also reused by the D4.3 scalar oracle and the resident GPU
            // path. Do not hand a cache with the wrong storage/append contract to the new mode.
            if (current_cache.use_count() > 1) {
                throw std::logic_error("OSCAR live mixed cache mode changed while active");
            }
            current_owner = nullptr;
            current_cache.reset();
        } else {
            current_cache->refresh_live_reference_taps();
            return current_cache;
        }
    }
    if (current_cache && current_cache.use_count() > 1) {
        throw std::logic_error("OSCAR live mixed diagnostic cache is already in use");
    }
    current_owner = owner;
    current_cache = std::make_shared<OscarLiveMixedReferenceCache>(
        max_context, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(owner)),
        std::move(rotations));
    return current_cache;
}

bool rotated_bf16_mode_enabled() {
    const char* value = std::getenv("NINFER_OSCAR_ROTATION_MODE");
    if (value == nullptr || *value == '\0') { return false; }
    if (std::string_view(value) != "oscar-rotated-bf16" &&
        std::string_view(value) != "oscar-int2-reference-live" &&
        std::string_view(value) != "oscar-int2-gpu" &&
        std::string_view(value) != "oscar-int2-gpu-resident") {
        throw std::invalid_argument("unsupported NINFER_OSCAR_ROTATION_MODE");
    }
    return std::string_view(value) == "oscar-rotated-bf16";
}

bool live_int2_reference_mode_enabled() {
    const char* value = std::getenv("NINFER_OSCAR_ROTATION_MODE");
    if (value == nullptr || *value == '\0') { return false; }
    if (std::string_view(value) != "oscar-rotated-bf16" &&
        std::string_view(value) != "oscar-int2-reference-live" &&
        std::string_view(value) != "oscar-int2-gpu" &&
        std::string_view(value) != "oscar-int2-gpu-resident") {
        throw std::invalid_argument("unsupported NINFER_OSCAR_ROTATION_MODE");
    }
    return std::string_view(value) == "oscar-int2-reference-live";
}

bool live_int2_gpu_mode_enabled() {
    const char* value = std::getenv("NINFER_OSCAR_ROTATION_MODE");
    if (value == nullptr || *value == '\0') { return false; }
    if (std::string_view(value) != "oscar-rotated-bf16" &&
        std::string_view(value) != "oscar-int2-reference-live" &&
        std::string_view(value) != "oscar-int2-gpu" &&
        std::string_view(value) != "oscar-int2-gpu-resident") {
        throw std::invalid_argument("unsupported NINFER_OSCAR_ROTATION_MODE");
    }
    return std::string_view(value) == "oscar-int2-gpu" ||
           std::string_view(value) == "oscar-int2-gpu-resident";
}

bool live_int2_gpu_resident_mode_enabled() {
    const char* value = std::getenv("NINFER_OSCAR_ROTATION_MODE");
    if (value == nullptr || *value == '\0') { return false; }
    if (std::string_view(value) != "oscar-rotated-bf16" &&
        std::string_view(value) != "oscar-int2-reference-live" &&
        std::string_view(value) != "oscar-int2-gpu" &&
        std::string_view(value) != "oscar-int2-gpu-resident") {
        throw std::invalid_argument("unsupported NINFER_OSCAR_ROTATION_MODE");
    }
    return std::string_view(value) == "oscar-int2-gpu-resident";
}

bool live_int2_gpu_fused_mode_enabled() {
    if (!live_int2_gpu_resident_mode_enabled()) { return false; }
    const char* value = std::getenv("NINFER_OSCAR_D4_6_FUSED");
    if (value == nullptr || *value == '\0' || std::string_view(value) == "1" ||
        std::string_view(value) == "true") {
        return true;
    }
    if (std::string_view(value) == "0" || std::string_view(value) == "false") {
        return false;
    }
    throw std::invalid_argument("NINFER_OSCAR_D4_6_FUSED must be 0 or 1");
}

std::uint32_t live_gpu_prefill_query_block_size() {
    const char* value = std::getenv("NINFER_OSCAR_D4_5_QBLOCK");
    if (value == nullptr || *value == '\0') { return 64U; }
    const std::string_view requested(value);
    if (requested == "8") { return 8U; }
    if (requested == "16") { return 16U; }
    if (requested == "32") { return 32U; }
    if (requested == "64") { return 64U; }
    throw std::invalid_argument("NINFER_OSCAR_D4_5_QBLOCK must be one of Q8/Q16/Q32/Q64");
}

bool matched_fp32_mode_enabled() {
    const char* value = std::getenv("NINFER_OSCAR_MATCHED_FP32");
    return value != nullptr && std::string_view(value) == "1";
}

RotationPrecision rotation_precision_mode() {
    if (!rotated_bf16_mode_enabled()) { return RotationPrecision::Bf16Materialized; }
    const char* value = std::getenv("NINFER_OSCAR_ROTATION_PRECISION");
    if (value == nullptr || *value == '\0' || std::string_view(value) == "bf16-materialized") {
        return RotationPrecision::Bf16Materialized;
    }
    if (std::string_view(value) == "fp32-rotation") { return RotationPrecision::Fp32Rotation; }
    if (std::string_view(value) == "fp32-inverse") { return RotationPrecision::Fp32Inverse; }
    if (std::string_view(value) == "fp32-rotation+inverse") {
        return RotationPrecision::Fp32RotationAndInverse;
    }
    throw std::invalid_argument("unsupported NINFER_OSCAR_ROTATION_PRECISION");
}

std::shared_ptr<const OscarRotationSet> rotation_set_from_environment() {
    if (!rotated_bf16_mode_enabled() && !live_int2_reference_mode_enabled() &&
        !live_int2_gpu_mode_enabled() && !live_int2_gpu_resident_mode_enabled()) {
        return {};
    }
    static std::shared_ptr<const OscarRotationSet> cached = load_rotation_set();
    return cached;
}

std::shared_ptr<OscarMatchedFP32Cache>
matched_fp32_cache_for(const void* owner, std::uint32_t max_context) {
    if (owner == nullptr) {
        throw std::invalid_argument("OSCAR matched FP32 cache owner is null");
    }
    static std::mutex mutex;
    static const void* current_owner = nullptr;
    static std::shared_ptr<OscarMatchedFP32Cache> current_cache;
    std::lock_guard lock(mutex);
    if (current_cache && current_owner == owner) {
        if (current_cache->max_context() != max_context) {
            throw std::logic_error("OSCAR matched FP32 cache capacity changed for an owner");
        }
        return current_cache;
    }
    if (current_cache && current_cache.use_count() > 1) {
        throw std::logic_error("OSCAR matched FP32 diagnostic cache is already in use");
    }
    current_owner = owner;
    current_cache = std::make_shared<OscarMatchedFP32Cache>(max_context);
    return current_cache;
}

namespace {
thread_local std::int32_t matched_diagnostic_step = -1;
}

void set_matched_diagnostic_step(std::int32_t step) noexcept {
    matched_diagnostic_step = step;
}

std::string matched_diagnostic_step_suffix() {
    if (matched_diagnostic_step < 0) { return {}; }
    return std::string(".step_") + (matched_diagnostic_step < 10 ? "0" : "") +
           std::to_string(matched_diagnostic_step);
}

void bf16_to_fp32(const Tensor& input, Tensor& output, cudaStream_t stream) {
    launch_oscar_bf16_to_fp32(input, output, stream);
}

void fp32_to_bf16(const Tensor& input, Tensor& output, cudaStream_t stream) {
    launch_oscar_fp32_to_bf16(input, output, stream);
}

} // namespace ninfer::targets::qwen3_6::oscar_internal
