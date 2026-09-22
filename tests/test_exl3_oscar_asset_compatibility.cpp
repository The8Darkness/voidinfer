// E4C1 Stage 3/4: existing cal30k OSCAR asset differential on native EXL3 QKV.
// Reference A: ordinary high-precision attention on current EXL3 Q/K/V.
#include "exl3/text_model.h"
#include "ops/kv_cache/oscar_int2_g128.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::exl3::Exl3TextModel;
constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kGqa = 6;
constexpr float kAttentionScale = 0.0625F;
constexpr int kRows = 14;
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

std::vector<float> read_f32(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open OSCAR runtime rotation");
    std::vector<float> values(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    require(input.gcount() == static_cast<std::streamsize>(count * sizeof(float)), "bad rotation size");
    for (float value : values) require(std::isfinite(value), "rotation NaN/Inf");
    return values;
}

std::vector<float> rotate_row(const float* input, const float* rotation, bool inverse) {
    std::vector<float> output(kHeadDim, 0.0F);
    for (int out = 0; out < kHeadDim; ++out) {
        float sum = 0.0F;
        for (int in = 0; in < kHeadDim; ++in) {
            const float coefficient =
                inverse ? rotation[out * kHeadDim + in] : rotation[in * kHeadDim + out];
            sum = std::fma(input[in], coefficient, sum);
        }
        output[out] = sum;
    }
    return output;
}

struct Metrics {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
};

Metrics compare(std::span<const float> actual, std::span<const float> expected) {
    require(actual.size() == expected.size(), "metric size mismatch");
    require(!actual.empty(), "metric empty");
    double error_sq = 0.0;
    double expected_sq = 0.0;
    Metrics result;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "non-finite stage value");
        const double error = static_cast<double>(actual[i]) - expected[i];
        result.max_abs = std::max(result.max_abs, std::abs(error));
        error_sq += error * error;
        expected_sq += static_cast<double>(expected[i]) * expected[i];
    }
    result.relative_l2 = std::sqrt(error_sq) / std::max(std::sqrt(expected_sq), 1.0e-30);
    return result;
}

struct LayerCase {
    Metrics k_recon;
    Metrics v_recon;
    Metrics scores;
    Metrics softmax;
    Metrics av;
    Metrics recovered;
    Metrics invariance;
    int worst_head = -1;
    double worst_head_max_abs = 0.0;
    double worst_head_rel_l2 = 0.0;
};
LayerCase evaluate_layer(const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v, const float* rk, const float* rv) {
    std::vector<float> q_rot(static_cast<std::size_t>(kRows) * kQHeads * kHeadDim);
    std::vector<float> k_rot(static_cast<std::size_t>(kRows) * kKVHeads * kHeadDim);
    std::vector<float> v_rot(static_cast<std::size_t>(kRows) * kKVHeads * kHeadDim);
    for (int t = 0; t < kRows; ++t) {
        for (int h = 0; h < kQHeads; ++h) {
            auto r = rotate_row(q.data() + (static_cast<std::size_t>(t) * kQHeads + h) * kHeadDim, rk, false);
            std::copy(r.begin(), r.end(), q_rot.begin() + (static_cast<std::size_t>(t) * kQHeads + h) * kHeadDim);
        }
        for (int h = 0; h < kKVHeads; ++h) {
            auto r0 = rotate_row(k.data() + (static_cast<std::size_t>(t) * kKVHeads + h) * kHeadDim, rk, false);
            auto r1 = rotate_row(v.data() + (static_cast<std::size_t>(t) * kKVHeads + h) * kHeadDim, rv, false);
            std::copy(r0.begin(), r0.end(), k_rot.begin() + (static_cast<std::size_t>(t) * kKVHeads + h) * kHeadDim);
            std::copy(r1.begin(), r1.end(), v_rot.begin() + (static_cast<std::size_t>(t) * kKVHeads + h) * kHeadDim);
        }
    }
    std::vector<float> k_dec(k_rot.size()), v_dec(v_rot.size());
    for (int t = 0; t < kRows; ++t) {
        for (int h = 0; h < kKVHeads; ++h) {
            const float* ks = k_rot.data() + (static_cast<std::size_t>(t) * kKVHeads + h) * kHeadDim;
            const float* vs = v_rot.data() + (static_cast<std::size_t>(t) * kKVHeads + h) * kHeadDim;
            std::array<float, kHeadDim> decoded{};
            const auto ek = ninfer::ops::oscar_int2_g128_encode(ks, kHeadDim, 0.96F);
            ninfer::ops::oscar_int2_g128_decode(ek, decoded.data(), kHeadDim);
            std::copy(decoded.begin(), decoded.end(),
                      k_dec.begin() + (static_cast<std::size_t>(t) * kKVHeads + h) * kHeadDim);
            const auto ev = ninfer::ops::oscar_int2_g128_encode(vs, kHeadDim, 0.92F);
            ninfer::ops::oscar_int2_g128_decode(ev, decoded.data(), kHeadDim);
            std::copy(decoded.begin(), decoded.end(),
                      v_dec.begin() + (static_cast<std::size_t>(t) * kKVHeads + h) * kHeadDim);
        }
    }
    LayerCase out;
    out.k_recon = compare(k_dec, k_rot);
    out.v_recon = compare(v_dec, v_rot);
    std::vector<float> all_scores_q, all_scores_e, all_prob_q, all_prob_e;
    std::vector<float> all_av_q, all_av_e, all_rec_q, all_rec_n, all_rec_e;
    for (int t = 0; t < kRows; ++t) {
        for (int qh = 0; qh < kQHeads; ++qh) {
            const int kvh = qh / kGqa;
            std::vector<float> sq(static_cast<std::size_t>(t) + 1), se(static_cast<std::size_t>(t) + 1);
            std::vector<float> sn(static_cast<std::size_t>(t) + 1);
            for (int c = 0; c <= t; ++c) {
                float aq = 0.0F, ae = 0.0F, an = 0.0F;
                for (int d = 0; d < kHeadDim; ++d) {
                    const float qq = q_rot[(static_cast<std::size_t>(t) * kQHeads + qh) * kHeadDim + d];
                    aq = std::fma(qq, k_dec[(static_cast<std::size_t>(c) * kKVHeads + kvh) * kHeadDim + d], aq);
                    ae = std::fma(qq, k_rot[(static_cast<std::size_t>(c) * kKVHeads + kvh) * kHeadDim + d], ae);
                    an = std::fma(q[(static_cast<std::size_t>(t) * kQHeads + qh) * kHeadDim + d],
                                  k[(static_cast<std::size_t>(c) * kKVHeads + kvh) * kHeadDim + d], an);
                }
                sq[c] = aq * kAttentionScale;
                se[c] = ae * kAttentionScale;
                sn[c] = an * kAttentionScale;
            }
            const float mq = *std::max_element(sq.begin(), sq.end());
            const float me = *std::max_element(se.begin(), se.end());
            const float mn = *std::max_element(sn.begin(), sn.end());
            float dq = 0.0F, de = 0.0F, dn = 0.0F;
            for (float& s : sq) { s = std::exp(s - mq); dq += s; }
            for (float& s : se) { s = std::exp(s - me); de += s; }
            for (float& s : sn) { s = std::exp(s - mn); dn += s; }
            std::vector<float> av_q(kHeadDim, 0.0F), av_e(kHeadDim, 0.0F), av_n(kHeadDim, 0.0F);
            for (int c = 0; c <= t; ++c) {
                const float wq = sq[c] / dq;
                const float we = se[c] / de;
                const float wn = sn[c] / dn;
                for (int d = 0; d < kHeadDim; ++d) {
                    av_q[d] = std::fma(wq, v_dec[(static_cast<std::size_t>(c) * kKVHeads + kvh) * kHeadDim + d], av_q[d]);
                    av_e[d] = std::fma(we, v_rot[(static_cast<std::size_t>(c) * kKVHeads + kvh) * kHeadDim + d], av_e[d]);
                    av_n[d] = std::fma(wn, v[(static_cast<std::size_t>(c) * kKVHeads + kvh) * kHeadDim + d], av_n[d]);
                }
            }
            const auto rec_q = rotate_row(av_q.data(), rv, true);
            const auto rec_e = rotate_row(av_e.data(), rv, true);
            all_scores_q.insert(all_scores_q.end(), sq.begin(), sq.end());
            all_scores_e.insert(all_scores_e.end(), se.begin(), se.end());
            for (std::size_t i = 0; i < sq.size(); ++i) {
                all_prob_q.push_back(sq[i] / dq);
                all_prob_e.push_back(se[i] / de);
            }
            all_av_q.insert(all_av_q.end(), av_q.begin(), av_q.end());
            all_av_e.insert(all_av_e.end(), av_e.begin(), av_e.end());
            all_rec_q.insert(all_rec_q.end(), rec_q.begin(), rec_q.end());
            all_rec_e.insert(all_rec_e.end(), rec_e.begin(), rec_e.end());
            all_rec_n.insert(all_rec_n.end(), av_n.begin(), av_n.end());
            const Metrics head_m = compare(rec_q, av_n);
            if (head_m.max_abs > out.worst_head_max_abs) {
                out.worst_head_max_abs = head_m.max_abs;
                out.worst_head = qh + t * kQHeads;
                out.worst_head_rel_l2 = head_m.relative_l2;
            }
        }
    }
    out.scores = compare(all_scores_q, all_scores_e);
    out.softmax = compare(all_prob_q, all_prob_e);
    out.av = compare(all_av_q, all_av_e);
    out.recovered = compare(all_rec_q, all_rec_n);
    out.invariance = compare(all_rec_e, all_rec_n);
    return out;
}
int run() {
    const std::string target = env("NINFER_EXL3_TARGET_PATH");
    const std::string rotation_dir = env("NINFER_OSCAR_ROTATION_ASSET_DIR");
    const std::string output_dir = env("NINFER_EXL3_OSCAR_COMPAT_DIR");
    if (target.empty() || rotation_dir.empty() || output_dir.empty()) {
        std::cout << "E4C1 asset compatibility skipped: set NINFER_EXL3_TARGET_PATH, "
                     "NINFER_OSCAR_ROTATION_ASSET_DIR, NINFER_EXL3_OSCAR_COMPAT_DIR\n";
        return 77;
    }
    const std::vector<std::int64_t> prose = {248045, 846, 198, 33963, 799, 2716, 2029, 883,
                                             47503, 13, 248046, 198, 248045, 74455};
    const std::vector<std::int64_t> code = {82, 11, 82, 13, 220, 17, 315, 10,
                                            82, 12, 660, 25, 198, 82};
    std::vector<std::int64_t> repetitive(kRows, 846);
    std::vector<std::int64_t> pseudorandom;
    std::uint64_t lcg = 0x12345678ULL;
    for (int i = 0; i < kRows; ++i) {
        lcg = 1664525ULL * lcg + 1013904223ULL;
        pseudorandom.push_back(static_cast<std::int64_t>((lcg >> 16U) % 248320U));
    }
    const std::vector<std::pair<std::string, std::vector<std::int64_t>>> prompts = {
        {"prose", prose}, {"code", code}, {"repetitive", repetitive}, {"pseudorandom", pseudorandom}};
    const auto model = Exl3TextModel::load(target, 256);
    const auto rk_all = read_f32(std::filesystem::path(rotation_dir) / "runtime" / "k_rotation_fp32.bin",
                                 16U * kHeadDim * kHeadDim);
    const auto rv_all = read_f32(std::filesystem::path(rotation_dir) / "runtime" / "v_rotation_fp32.bin",
                                 16U * kHeadDim * kHeadDim);
    const auto root = std::filesystem::absolute(output_dir).lexically_normal();
    std::filesystem::create_directories(root);
    std::ofstream csv(root / "asset_compatibility.csv", std::ios::trunc);
    require(static_cast<bool>(csv), "cannot write E4C1 asset compatibility CSV");
    csv << "prompt,layer,rows,k_recon_max_abs,k_recon_rel_l2,v_recon_max_abs,v_recon_rel_l2,"
           "scores_max_abs,scores_rel_l2,softmax_max_abs,softmax_rel_l2,av_max_abs,av_rel_l2,"
           "recovered_max_abs,recovered_rel_l2,invariance_max_abs,invariance_rel_l2,"
           "worst_head,worst_head_max_abs,worst_head_rel_l2\n";
    double worst_abs = 0.0, worst_rel = 0.0;
    int worst_layer = -1;
    std::string worst_prompt;
    for (const auto& named : prompts) {
        const std::string& name = named.first;
        const std::vector<std::int64_t>& prompt = named.second;
        require(prompt.size() == static_cast<std::size_t>(kRows), "prompt row count changed");
        auto context = model->create_context(false);
        context->prefill(std::span<const std::int64_t>(prompt.data(), prompt.size()));
        for (int layer : kFullLayers) {
            const auto capture = context->full_attention_qkv_host(layer);
            require(capture.rows == kRows, "E4C1 asset capture row count changed");
            const int full_index = (layer - 3) / 4;
            const float* rk_layer = rk_all.data() + static_cast<std::size_t>(full_index) * kHeadDim * kHeadDim;
            const float* rv_layer = rv_all.data() + static_cast<std::size_t>(full_index) * kHeadDim * kHeadDim;
            std::vector<float> q(capture.q_rope.size()), k(capture.k_rope.size()), v(capture.v_projection.size());
            std::transform(capture.q_rope.begin(), capture.q_rope.end(), q.begin(), half_to_float);
            std::transform(capture.k_rope.begin(), capture.k_rope.end(), k.begin(), half_to_float);
            std::transform(capture.v_projection.begin(), capture.v_projection.end(), v.begin(), half_to_float);
            const LayerCase m = evaluate_layer(q, k, v, rk_layer, rv_layer);
            require(m.invariance.relative_l2 <= 1e-3, "E4C1 rotation invariance broken");
            require(m.recovered.relative_l2 <= 1.0, "E4C1 asset incompatible with native EXL3");
            csv << name << "," << layer << "," << capture.rows << "," << std::setprecision(10)
                << m.k_recon.max_abs << "," << m.k_recon.relative_l2 << ","
                << m.v_recon.max_abs << "," << m.v_recon.relative_l2 << ","
                << m.scores.max_abs << "," << m.scores.relative_l2 << ","
                << m.softmax.max_abs << "," << m.softmax.relative_l2 << ","
                << m.av.max_abs << "," << m.av.relative_l2 << ","
                << m.recovered.max_abs << "," << m.recovered.relative_l2 << ","
                << m.invariance.max_abs << "," << m.invariance.relative_l2 << ","
                << m.worst_head << "," << m.worst_head_max_abs << "," << m.worst_head_rel_l2 << "\n";
            if (m.recovered.relative_l2 > worst_rel) {
                worst_rel = m.recovered.relative_l2;
                worst_abs = m.recovered.max_abs;
                worst_layer = layer;
                worst_prompt = name;
            }
        }
    }
    std::cout << "E4C1 EXL3 OSCAR asset compatibility: PASS prompts=4 layers=16 "
              << "worst_prompt=" << worst_prompt << " worst_layer=" << worst_layer
              << " worst_recovered_max_abs=" << std::setprecision(10) << worst_abs
              << " worst_recovered_relative_l2=" << worst_rel << "\n";
    return 0;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "E4C1 EXL3 OSCAR asset compatibility: FAIL: " << error.what() << "\n";
        return 1;
    }
}
