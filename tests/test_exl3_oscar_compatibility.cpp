#include "exl3/text_model.h"
#include "ops/kv_cache/oscar_int2_g128.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::exl3::Exl3FullAttentionQKVHost;
using ninfer::exl3::Exl3TextModel;
constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kGqa = 6;
constexpr float kAttentionScale = 0.0625F;
constexpr int kFullLayers[] = {3, 7, 11, 15, 19, 23, 27, 31,
                               35, 39, 43, 47, 51, 55, 59, 63};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits & 0x8000U)) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 31U;
    const std::uint32_t fraction = bits & 1023U;
    std::uint32_t value = sign;
    if (exponent == 0U) {
        if (fraction != 0U) {
            std::uint32_t mantissa = fraction;
            int shift = 0;
            while ((mantissa & 1024U) == 0U) { mantissa <<= 1U; ++shift; }
            value |= static_cast<std::uint32_t>(127 - 14 - shift) << 23U;
            value |= (mantissa & 1023U) << 13U;
        }
    } else if (exponent == 31U) {
        value |= 0x7f800000U | (fraction << 13U);
    } else {
        value |= (exponent + 112U) << 23U | (fraction << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

void write_bytes(const std::filesystem::path& path, std::span<const std::uint16_t> values) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot write EXL3 OSCAR compatibility capture");
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size_bytes()));
    require(static_cast<bool>(output), "cannot finish EXL3 OSCAR compatibility capture");
}

std::vector<float> read_f32(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open OSCAR runtime rotation: " + path.string());
    std::vector<float> values(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    require(input.gcount() == static_cast<std::streamsize>(count * sizeof(float)) && input.peek() == std::char_traits<char>::eof(),
            "invalid OSCAR runtime rotation size: " + path.string());
    for (float value : values) require(std::isfinite(value), "OSCAR rotation contains NaN/Inf");
    return values;
}

std::vector<float> rotate_row(const float* input, const float* rotation, bool inverse) {
    std::vector<float> output(kHeadDim, 0.0F);
    for (int out = 0; out < kHeadDim; ++out) {
        float sum = 0.0F;
        for (int in = 0; in < kHeadDim; ++in) {
            const float coefficient = inverse ? rotation[out * kHeadDim + in]
                                              : rotation[in * kHeadDim + out];
            sum = std::fma(input[in], coefficient, sum);
        }
        output[out] = sum;
    }
    return output;
}

struct Metrics { double max_abs = 0.0; double relative_l2 = 0.0; };

Metrics compare(std::span<const float> actual, std::span<const float> expected) {
    require(actual.size() == expected.size(), "compatibility metric size mismatch");
    double error_sq = 0.0;
    double expected_sq = 0.0;
    Metrics result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = static_cast<double>(actual[i]) - expected[i];
        result.max_abs = std::max(result.max_abs, std::abs(error));
        error_sq += error * error;
        expected_sq += static_cast<double>(expected[i]) * expected[i];
    }
    result.relative_l2 = std::sqrt(error_sq) / std::max(std::sqrt(expected_sq), 1.0e-30);
    return result;
}

struct CaseResult { Metrics attention; double rotation_roundtrip_max_abs = 0.0; };

CaseResult evaluate(const std::vector<float>& q, const std::vector<float>& k,
                    const std::vector<float>& v, int rows,
                    const std::vector<float>& rk, const std::vector<float>& rv) {
    std::vector<float> compressed;
    std::vector<float> reference;
    compressed.reserve(static_cast<std::size_t>(rows) * kQHeads * kHeadDim);
    reference.reserve(compressed.capacity());
    CaseResult result;

    std::vector<std::vector<float>> q_rot(static_cast<std::size_t>(rows) * kQHeads);
    std::vector<std::vector<float>> k_rot(static_cast<std::size_t>(rows) * kKVHeads);
    std::vector<std::vector<float>> v_rot(static_cast<std::size_t>(rows) * kKVHeads);
    std::vector<std::vector<float>> k_dec(k_rot.size());
    std::vector<std::vector<float>> v_dec(v_rot.size());
    for (int token = 0; token < rows; ++token) {
        for (int head = 0; head < kQHeads; ++head) {
            const float* source = q.data() + (static_cast<std::size_t>(token) * kQHeads + head) * kHeadDim;
            q_rot[static_cast<std::size_t>(token) * kQHeads + head] = rotate_row(source, rk.data(), false);
        }
        for (int head = 0; head < kKVHeads; ++head) {
            const float* k_source = k.data() + (static_cast<std::size_t>(token) * kKVHeads + head) * kHeadDim;
            const float* v_source = v.data() + (static_cast<std::size_t>(token) * kKVHeads + head) * kHeadDim;
            k_rot[static_cast<std::size_t>(token) * kKVHeads + head] = rotate_row(k_source, rk.data(), false);
            v_rot[static_cast<std::size_t>(token) * kKVHeads + head] = rotate_row(v_source, rv.data(), false);
            std::array<float, kHeadDim> decoded{};
            const auto ek = ninfer::ops::oscar_int2_g128_encode(k_rot[static_cast<std::size_t>(token) * kKVHeads + head].data(), kHeadDim, 0.96F);
            ninfer::ops::oscar_int2_g128_decode(ek, decoded.data(), kHeadDim);
            k_dec[static_cast<std::size_t>(token) * kKVHeads + head].assign(decoded.begin(), decoded.end());
            const auto ev = ninfer::ops::oscar_int2_g128_encode(v_rot[static_cast<std::size_t>(token) * kKVHeads + head].data(), kHeadDim, 0.92F);
            ninfer::ops::oscar_int2_g128_decode(ev, decoded.data(), kHeadDim);
            v_dec[static_cast<std::size_t>(token) * kKVHeads + head].assign(decoded.begin(), decoded.end());
        }
    }
    for (int token = 0; token < rows; ++token) {
        for (int qh = 0; qh < kQHeads; ++qh) {
            const int kvh = qh / kGqa;
            std::vector<float> original_scores(static_cast<std::size_t>(token + 1));
            std::vector<float> compressed_scores(static_cast<std::size_t>(token + 1));
            for (int key = 0; key <= token; ++key) {
                float original_score = 0.0F;
                float rotated_score = 0.0F;
                for (int d = 0; d < kHeadDim; ++d) {
                    original_score = std::fma(q[(static_cast<std::size_t>(token) * kQHeads + qh) * kHeadDim + d],
                                              k[(static_cast<std::size_t>(key) * kKVHeads + kvh) * kHeadDim + d], original_score);
                    rotated_score = std::fma(q_rot[static_cast<std::size_t>(token) * kQHeads + qh][d],
                                             k_dec[static_cast<std::size_t>(key) * kKVHeads + kvh][d], rotated_score);
                }
                original_scores[static_cast<std::size_t>(key)] = original_score * kAttentionScale;
                compressed_scores[static_cast<std::size_t>(key)] = rotated_score * kAttentionScale;
            }
            const float original_maximum = *std::max_element(original_scores.begin(), original_scores.end());
            const float compressed_maximum = *std::max_element(compressed_scores.begin(), compressed_scores.end());
            float original_denominator = 0.0F;
            float compressed_denominator = 0.0F;
            for (float& score : original_scores) { score = std::exp(score - original_maximum); original_denominator += score; }
            for (float& score : compressed_scores) { score = std::exp(score - compressed_maximum); compressed_denominator += score; }
            std::vector<float> compressed_av(kHeadDim, 0.0F);
            std::vector<float> reference_av(kHeadDim, 0.0F);
            for (int key = 0; key <= token; ++key) {
                const float compressed_weight = compressed_scores[static_cast<std::size_t>(key)] / compressed_denominator;
                const float original_weight = original_scores[static_cast<std::size_t>(key)] / original_denominator;
                for (int d = 0; d < kHeadDim; ++d) {
                    compressed_av[d] = std::fma(compressed_weight, v_dec[static_cast<std::size_t>(key) * kKVHeads + kvh][d], compressed_av[d]);
                    reference_av[d] = std::fma(original_weight, v[(static_cast<std::size_t>(key) * kKVHeads + kvh) * kHeadDim + d], reference_av[d]);
                }
            }
            const auto recovered = rotate_row(compressed_av.data(), rv.data(), true);
            for (int d = 0; d < kHeadDim; ++d) {
                compressed.push_back(recovered[d]);
                reference.push_back(reference_av[d]);
            }
        }
    }
    result.attention = compare(compressed, reference);
    for (int token = 0; token < rows; ++token) {
        const auto roundtrip = rotate_row(q.data() + static_cast<std::size_t>(token) * kQHeads * kHeadDim, rk.data(), false);
        const auto restored = rotate_row(roundtrip.data(), rk.data(), true);
        for (int d = 0; d < kHeadDim; ++d) {
            result.rotation_roundtrip_max_abs = std::max(result.rotation_roundtrip_max_abs,
                static_cast<double>(std::abs(restored[d] - q[static_cast<std::size_t>(token) * kQHeads * kHeadDim + d])));
        }
    }
    return result;
}

int run() {
    const std::string target = env("NINFER_EXL3_TARGET_PATH");
    const std::string rotation_dir = env("NINFER_OSCAR_ROTATION_ASSET_DIR");
    const std::string output_dir = env("NINFER_EXL3_OSCAR_COMPAT_DIR");
    if (target.empty() || rotation_dir.empty() || output_dir.empty()) {
        std::cout << "E4C1 skipped: set NINFER_EXL3_TARGET_PATH, NINFER_OSCAR_ROTATION_ASSET_DIR, and NINFER_EXL3_OSCAR_COMPAT_DIR\n";
        return 77;
    }
    const auto model = Exl3TextModel::load(target, 256);
    auto context = model->create_context(false);
    const std::int64_t prompt[] = {248045, 846, 198, 33963, 799, 2716, 2029, 883,
                                   47503, 13, 248046, 198, 248045, 74455};
    context->prefill(prompt);
    const auto root = std::filesystem::absolute(output_dir).lexically_normal();
    std::filesystem::create_directories(root);
    std::ofstream csv(root / "compatibility.csv", std::ios::trunc);
    require(static_cast<bool>(csv), "cannot write E4C1 compatibility CSV");
    csv << "layer,rows,q_bytes,k_bytes,v_bytes,rotation_roundtrip_max_abs,attention_max_abs,attention_relative_l2\n";
    double worst_abs = 0.0;
    double worst_rel = 0.0;
    for (int layer : kFullLayers) {
        const Exl3FullAttentionQKVHost capture = context->full_attention_qkv_host(layer);
        require(capture.rows == 14, "E4C1 compatibility capture row count changed");
        write_bytes(root / ("layer_" + std::to_string(layer) + "_q_rope.f16"), capture.q_rope);
        write_bytes(root / ("layer_" + std::to_string(layer) + "_k_rope.f16"), capture.k_rope);
        write_bytes(root / ("layer_" + std::to_string(layer) + "_v_projection.f16"), capture.v_projection);
        const auto rk = read_f32(std::filesystem::path(rotation_dir) / "runtime" / "k_rotation_fp32.bin",
                                 16U * kHeadDim * kHeadDim);
        const auto rv = read_f32(std::filesystem::path(rotation_dir) / "runtime" / "v_rotation_fp32.bin",
                                 16U * kHeadDim * kHeadDim);
        const int full_index = (layer - 3) / 4;
        const float* rk_layer = rk.data() + static_cast<std::size_t>(full_index) * kHeadDim * kHeadDim;
        const float* rv_layer = rv.data() + static_cast<std::size_t>(full_index) * kHeadDim * kHeadDim;
        std::vector<float> q(capture.q_rope.size()), k(capture.k_rope.size()), v(capture.v_projection.size());
        std::transform(capture.q_rope.begin(), capture.q_rope.end(), q.begin(), half_to_float);
        std::transform(capture.k_rope.begin(), capture.k_rope.end(), k.begin(), half_to_float);
        std::transform(capture.v_projection.begin(), capture.v_projection.end(), v.begin(), half_to_float);
        const auto metrics = evaluate(q, k, v, capture.rows,
                                      std::vector<float>(rk_layer, rk_layer + kHeadDim * kHeadDim),
                                      std::vector<float>(rv_layer, rv_layer + kHeadDim * kHeadDim));
        worst_abs = std::max(worst_abs, metrics.attention.max_abs);
        worst_rel = std::max(worst_rel, metrics.attention.relative_l2);
        csv << layer << ',' << capture.rows << ',' << capture.q_rope.size() * 2 << ','
            << capture.k_rope.size() * 2 << ',' << capture.v_projection.size() * 2 << ','
            << std::setprecision(10) << metrics.rotation_roundtrip_max_abs << ','
            << metrics.attention.max_abs << ',' << metrics.attention.relative_l2 << '\n';
    }
    std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
    require(static_cast<bool>(manifest), "cannot write E4C1 compatibility manifest");
    manifest << "schema=e4c1-exl3-oscar-compatibility-v1\n"
             << "target=" << std::filesystem::absolute(target).lexically_normal().string() << "\n"
             << "model_revision=SC_6.00bpw_H6_V6\nmodel_commit=60d005a257b39ecb25e4ba23c1dd29df877d5c69\n"
             << "capture_stage=post_qk_rmsnorm_post_rope_pre_causal_attention_cache_append\n"
             << "dtype=F16\nrows=14\nfull_attention_layers=3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63\n"
             << "q_heads=24\nkv_heads=4\nhead_dim=256\nworst_attention_max_abs=" << std::setprecision(10) << worst_abs
             << "\nworst_attention_relative_l2=" << worst_rel << "\n";
    std::cout << "E4C1 native EXL3 compatibility capture: PASS layers=16 rows=14"
              << " worst_attention_max_abs=" << std::setprecision(10) << worst_abs
              << " worst_attention_relative_l2=" << worst_rel << '\n';
    return 0;
}

} // namespace

int main() {
    try { return run(); }
    catch (const std::exception& error) {
        std::cerr << "E4C1 native EXL3 OSCAR compatibility: FAIL: " << error.what() << '\n';
        return 1;
    }
}
