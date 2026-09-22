#include "core/oscar_mixed_attention_reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMatrixValues =
    static_cast<std::size_t>(ninfer::kOscarMixedHeadDim) * ninfer::kOscarMixedHeadDim;
constexpr std::size_t kQueryValues =
    static_cast<std::size_t>(ninfer::kOscarMixedQueryHeads) * ninfer::kOscarMixedHeadDim;
constexpr std::size_t kRowValues =
    static_cast<std::size_t>(ninfer::kOscarMixedKVHeads) * ninfer::kOscarMixedHeadDim;
constexpr std::size_t kLayerValues =
    kRowValues * ninfer::kOscarMixedFullAttentionLayers.size();
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr float kParityTolerance = 1.0e-6F;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t round = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + round) >> 16U);
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

std::uint32_t mix(std::uint32_t value) noexcept {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

float deterministic_value(std::size_t layer_index, std::uint32_t logical_token,
                          std::uint32_t kv_head, std::uint32_t dimension,
                          std::uint32_t branch_salt, bool value_stream) noexcept {
    std::uint32_t seed = 0xD2B30001U ^ branch_salt;
    seed ^= static_cast<std::uint32_t>(layer_index + 1U) * 0x9e3779b9U;
    seed ^= (logical_token + 17U) * 0x85ebca6bU;
    seed ^= (kv_head + 3U) * 0xc2b2ae35U;
    seed ^= (dimension + 5U) * 0x27d4eb2fU;
    if (value_stream) { seed ^= 0xA5A5A5A5U; }
    const std::int32_t centered = static_cast<std::int32_t>(mix(seed) % 4096U) - 2048;
    const float base = static_cast<float>(centered) / (value_stream ? 149.0F : 113.0F);
    const float offset = static_cast<float>(static_cast<int>(dimension % 19U) - 9) * 0.003F;
    return base + offset + (value_stream ? 0.125F : -0.25F);
}

struct TokenRows {
    std::vector<std::uint16_t> k;
    std::vector<std::uint16_t> v;
};

TokenRows make_rows(std::uint32_t logical_token, std::uint32_t branch_salt) {
    TokenRows rows;
    rows.k.resize(kLayerValues);
    rows.v.resize(kLayerValues);
    for (std::size_t layer = 0; layer < ninfer::kOscarMixedFullAttentionLayers.size(); ++layer) {
        for (std::uint32_t head = 0; head < ninfer::kOscarMixedKVHeads; ++head) {
            for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim;
                 ++dimension) {
                const std::size_t offset = layer * kRowValues +
                    static_cast<std::size_t>(head) * ninfer::kOscarMixedHeadDim + dimension;
                rows.k[offset] = float_to_bf16(deterministic_value(
                    layer, logical_token, head, dimension, branch_salt, false));
                rows.v[offset] = float_to_bf16(deterministic_value(
                    layer, logical_token, head, dimension, branch_salt, true));
            }
        }
    }
    return rows;
}

std::vector<float> make_query(std::uint32_t model_layer, std::uint32_t logical_token) {
    std::vector<float> result(kQueryValues);
    for (std::uint32_t head = 0; head < ninfer::kOscarMixedQueryHeads; ++head) {
        for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim; ++dimension) {
            const std::uint32_t seed = mix(0xC3D40001U ^ model_layer * 0x9e3779b9U ^
                                           logical_token * 0x85ebca6bU ^
                                           head * 0xc2b2ae35U ^ dimension * 0x27d4eb2fU);
            const float centered = static_cast<float>(static_cast<std::int32_t>(seed % 2048U) -
                                                       1024);
            result[static_cast<std::size_t>(head) * ninfer::kOscarMixedHeadDim + dimension] =
                centered / 97.0F;
        }
    }
    return result;
}

std::vector<float> load_bank(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) { throw std::runtime_error("cannot open OSCAR rotation bank"); }
    const auto end = input.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) !=
                       ninfer::kOscarMixedFullAttentionLayers.size() * kMatrixValues * sizeof(float)) {
        throw std::runtime_error("OSCAR rotation bank has the wrong byte count");
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> result(static_cast<std::size_t>(end) / sizeof(float));
    input.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!input || !std::all_of(result.begin(), result.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error("OSCAR rotation bank is truncated or non-finite");
    }
    return result;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open OSCAR runtime manifest"); }
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

void validate_runtime_manifest(const std::filesystem::path& rotation_dir) {
    const std::string manifest = read_text(rotation_dir / "runtime_manifest.txt");
    const std::vector<std::string> required = {
        "schema=oscar-runtime-rotation-v1",
        "asset_identity=qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1",
        "model_sha256=6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e",
        "asset_manifest_sha256=4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560",
        "full_attention_layers=3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63",
        "q_heads=24", "kv_heads=4", "gqa_ratio=6", "head_dim=256", "rotary_dim=64",
        "layout=runtime[d,h,t];source[tokens,heads,head_dim]", "dtype=fp32",
        "calibrated=true", "rotation_mode=qqt_sst+r_h_pbr", "k_layers=16", "v_layers=16",
    };
    for (const auto& line : required) {
        require(manifest.find(line) != std::string::npos,
                "C4 runtime rotation manifest is incomplete or mismatched");
    }
}

void row_rotate(const float* input, const float* matrix, bool transpose, float* output) {
    for (std::uint32_t column = 0; column < ninfer::kOscarMixedHeadDim; ++column) {
        float sum = 0.0F;
        for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim; ++dimension) {
            const std::size_t matrix_index = transpose
                ? static_cast<std::size_t>(column) * ninfer::kOscarMixedHeadDim + dimension
                : static_cast<std::size_t>(dimension) * ninfer::kOscarMixedHeadDim + column;
            sum += input[dimension] * matrix[matrix_index];
        }
        output[column] = sum;
    }
}

ninfer::OscarMixedReadTier tier_for(std::uint32_t context, std::uint32_t logical) {
    const std::uint32_t recent_begin = context <= ninfer::kOscarMixedPrefixTokens
        ? context
        : std::max(ninfer::kOscarMixedPrefixTokens,
                   context > ninfer::kOscarMixedRecentTokens
                       ? context - ninfer::kOscarMixedRecentTokens : 0U);
    if (logical < ninfer::kOscarMixedPrefixTokens) {
        return ninfer::OscarMixedReadTier::ProtectedPrefixBFloat16;
    }
    if (logical < recent_begin) { return ninfer::OscarMixedReadTier::HistoricalOscarInt2G128; }
    return ninfer::OscarMixedReadTier::RecentBFloat16;
}

struct ReferenceInput {
    std::vector<TokenRows> tokens;
};

void decode_reference_row(const std::vector<std::uint16_t>& source, std::size_t offset,
                          bool key, bool historical, float* output) {
    for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim; ++dimension) {
        output[dimension] = bf16_to_float(source[offset + dimension]);
    }
    if (!historical) { return; }
    const auto encoded = ninfer::ops::oscar_int2_g128_encode(
        output, ninfer::kOscarMixedHeadDim, key ? 0.96F : 0.92F);
    ninfer::ops::oscar_int2_g128_decode(encoded, output, ninfer::kOscarMixedHeadDim);
}

ninfer::OscarMixedAttentionTrace independent_reference(
    std::size_t layer_index, std::uint32_t context, std::uint32_t query_token,
    const ReferenceInput& input, std::span<const float> q_original,
    std::span<const float> r_k, std::span<const float> r_v) {
    require(layer_index < ninfer::kOscarMixedFullAttentionLayers.size(),
            "reference layer index is outside topology");
    require(query_token < context && context <= input.tokens.size(),
            "reference query/context is invalid");
    ninfer::OscarMixedAttentionTrace result;
    result.model_layer = ninfer::kOscarMixedFullAttentionLayers[layer_index];
    result.query_token = query_token;
    const std::uint32_t token_count = query_token + 1U;
    result.logical_positions.resize(token_count);
    result.tiers.reserve(token_count);
    result.rotated_q.resize(kQueryValues);
    result.score_logits.resize(static_cast<std::size_t>(ninfer::kOscarMixedQueryHeads) * token_count);
    result.softmax.resize(result.score_logits.size());
    result.rotated_av.resize(kQueryValues);
    result.recovered_output.resize(kQueryValues);
    for (std::uint32_t logical = 0; logical < token_count; ++logical) {
        result.logical_positions[logical] = logical;
        const auto tier = tier_for(context, logical);
        result.tiers.push_back(tier);
        if (tier == ninfer::OscarMixedReadTier::ProtectedPrefixBFloat16) {
            ++result.prefix_count;
        } else if (tier == ninfer::OscarMixedReadTier::HistoricalOscarInt2G128) {
            ++result.historical_count;
        } else {
            ++result.recent_count;
        }
    }
    for (std::uint32_t q_head = 0; q_head < ninfer::kOscarMixedQueryHeads; ++q_head) {
        row_rotate(q_original.data() + static_cast<std::size_t>(q_head) * ninfer::kOscarMixedHeadDim,
                   r_k.data(), false,
                   result.rotated_q.data() + static_cast<std::size_t>(q_head) * ninfer::kOscarMixedHeadDim);
    }
    std::vector<float> keys(static_cast<std::size_t>(token_count) * ninfer::kOscarMixedKVHeads *
                            ninfer::kOscarMixedHeadDim);
    std::vector<float> values(keys.size());
    std::vector<float> row(ninfer::kOscarMixedHeadDim);
    for (std::uint32_t logical = 0; logical < token_count; ++logical) {
        const bool historical = tier_for(context, logical) ==
                                ninfer::OscarMixedReadTier::HistoricalOscarInt2G128;
        for (std::uint32_t kv_head = 0; kv_head < ninfer::kOscarMixedKVHeads; ++kv_head) {
            const std::size_t row_offset = (static_cast<std::size_t>(logical) *
                                             ninfer::kOscarMixedKVHeads + kv_head) *
                                            ninfer::kOscarMixedHeadDim;
            const std::size_t source_offset = layer_index * kRowValues +
                                               static_cast<std::size_t>(kv_head) *
                                               ninfer::kOscarMixedHeadDim;
            decode_reference_row(input.tokens[logical].k, source_offset, true, historical,
                                 keys.data() + row_offset);
            decode_reference_row(input.tokens[logical].v, source_offset, false, historical,
                                 values.data() + row_offset);
        }
    }
    for (std::uint32_t q_head = 0; q_head < ninfer::kOscarMixedQueryHeads; ++q_head) {
        const std::uint32_t kv_head = q_head / ninfer::kOscarMixedGqaRatio;
        const float* q = result.rotated_q.data() + static_cast<std::size_t>(q_head) * ninfer::kOscarMixedHeadDim;
        float* scores = result.score_logits.data() + static_cast<std::size_t>(q_head) * token_count;
        float* probabilities = result.softmax.data() + static_cast<std::size_t>(q_head) * token_count;
        float* output = result.rotated_av.data() + static_cast<std::size_t>(q_head) * ninfer::kOscarMixedHeadDim;
        std::fill(output, output + ninfer::kOscarMixedHeadDim, 0.0F);
        for (std::uint32_t logical = 0; logical < token_count; ++logical) {
            const std::size_t row_offset = (static_cast<std::size_t>(logical) *
                                             ninfer::kOscarMixedKVHeads + kv_head) *
                                            ninfer::kOscarMixedHeadDim;
            float dot = 0.0F;
            for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim; ++dimension) {
                dot += q[dimension] * keys[row_offset + dimension];
            }
            scores[logical] = dot * kAttentionScale;
        }
        const float maximum = *std::max_element(scores, scores + token_count);
        float sum = 0.0F;
        for (std::uint32_t logical = 0; logical < token_count; ++logical) {
            probabilities[logical] = std::exp(scores[logical] - maximum);
            sum += probabilities[logical];
        }
        for (std::uint32_t logical = 0; logical < token_count; ++logical) {
            probabilities[logical] /= sum;
            const std::size_t row_offset = (static_cast<std::size_t>(logical) *
                                             ninfer::kOscarMixedKVHeads + kv_head) *
                                            ninfer::kOscarMixedHeadDim;
            std::copy_n(values.data() + row_offset, ninfer::kOscarMixedHeadDim, row.data());
            for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim; ++dimension) {
                output[dimension] += probabilities[logical] * row[dimension];
            }
        }
    }
    for (std::uint32_t q_head = 0; q_head < ninfer::kOscarMixedQueryHeads; ++q_head) {
        row_rotate(result.rotated_av.data() + static_cast<std::size_t>(q_head) * ninfer::kOscarMixedHeadDim,
                   r_v.data(), true,
                   result.recovered_output.data() + static_cast<std::size_t>(q_head) * ninfer::kOscarMixedHeadDim);
    }
    return result;
}

struct Metric {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double relative_l2 = 0.0;
};

Metric metric(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "attention trace vector shape mismatch");
    double sum_abs = 0.0;
    double sum_diff_sq = 0.0;
    double sum_expected_sq = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double difference = static_cast<double>(actual[index]) - expected[index];
        const double reference = expected[index];
        sum_abs += std::abs(difference);
        sum_diff_sq += difference * difference;
        sum_expected_sq += reference * reference;
    }
    Metric result;
    result.max_abs = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result.max_abs = std::max(result.max_abs,
                                  std::abs(static_cast<double>(actual[index]) - expected[index]));
    }
    result.mean_abs = actual.empty() ? 0.0 : sum_abs / static_cast<double>(actual.size());
    result.relative_l2 = sum_expected_sq == 0.0 ? std::sqrt(sum_diff_sq)
                                                : std::sqrt(sum_diff_sq / sum_expected_sq);
    return result;
}

struct Comparison {
    std::string suite;
    std::uint32_t layer = 0;
    std::uint32_t query = 0;
    std::uint32_t context = 0;
    std::uint32_t prefix = 0;
    std::uint32_t historical = 0;
    std::uint32_t recent = 0;
    std::vector<std::uint32_t> logical_positions;
    std::vector<ninfer::OscarMixedReadTier> tiers;
    Metric q;
    Metric scores;
    Metric softmax;
    Metric rotated_av;
    Metric recovered;
};

Comparison compare(const std::string& suite, std::size_t layer_index,
                   std::uint32_t context, std::uint32_t query,
                   const ninfer::OscarMixedAgingLayerCache& cache,
                   const ReferenceInput& input, const std::vector<float>& k_bank,
                   const std::vector<float>& v_bank) {
    const std::uint32_t model_layer = ninfer::kOscarMixedFullAttentionLayers[layer_index];
    require(cache.model_layer() == model_layer && cache.context_tokens() == context,
            "reader cache does not match requested layer/context");
    const auto q = make_query(model_layer, query);
    const std::span<const float> rk(k_bank.data() + layer_index * kMatrixValues, kMatrixValues);
    const std::span<const float> rv(v_bank.data() + layer_index * kMatrixValues, kMatrixValues);
    const ninfer::OscarMixedAttentionReader reader(cache, q, rk, rv);
    const auto actual = reader.read(query);
    const auto expected = independent_reference(layer_index, context, query, input, q, rk, rv);
    require(actual.logical_positions == expected.logical_positions && actual.tiers == expected.tiers,
            "reader/reference logical coverage differs");
    require(actual.prefix_count == expected.prefix_count &&
                actual.historical_count == expected.historical_count &&
                actual.recent_count == expected.recent_count,
            "reader/reference tier counts differ");
    Comparison result;
    result.suite = suite;
    result.layer = model_layer;
    result.query = query;
    result.context = context;
    result.prefix = actual.prefix_count;
    result.historical = actual.historical_count;
    result.recent = actual.recent_count;
    result.logical_positions = actual.logical_positions;
    result.tiers = actual.tiers;
    result.q = metric(actual.rotated_q, expected.rotated_q);
    result.scores = metric(actual.score_logits, expected.score_logits);
    result.softmax = metric(actual.softmax, expected.softmax);
    result.rotated_av = metric(actual.rotated_av, expected.rotated_av);
    result.recovered = metric(actual.recovered_output, expected.recovered_output);
    for (const Metric* value : {&result.q, &result.scores, &result.softmax,
                                &result.rotated_av, &result.recovered}) {
        require(value->relative_l2 <= kParityTolerance,
                "mixed-cache reader/reference parity exceeded the FP32 gate");
    }
    return result;
}

void run_d4_1_microbenchmark(const std::vector<float>& k_bank,
                             const std::vector<float>& v_bank) {
    const std::size_t layer_index = 0;
    const std::span<const float> rk(k_bank.data() + layer_index * kMatrixValues, kMatrixValues);
    const std::span<const float> rv(v_bank.data() + layer_index * kMatrixValues, kMatrixValues);
    const auto profile_context = [&](std::uint32_t context) {
        const auto setup_start = std::chrono::steady_clock::now();
        ninfer::OscarMixedAgingLayerCache cache(3, 9100U + context,
                                                 ninfer::OscarMixedAgingAssetContract::c4_cal30k());
        for (std::uint32_t token = 0; token < context; ++token) {
            const auto rows = make_rows(token, 0xD4010000U);
            cache.append(token, std::span<const std::uint16_t>(rows.k.data(), kRowValues),
                         std::span<const std::uint16_t>(rows.v.data(), kRowValues));
        }
        const auto setup_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - setup_start)
                                  .count();
        const std::uint32_t query = context - 1U;
        const auto q = make_query(3, query);
        const ninfer::OscarMixedAttentionReader reader(cache, q, rk, rv);
        const auto trace = reader.read(query);
        require(trace.prefix_count + trace.historical_count + trace.recent_count == context,
                "D4.1 microbenchmark tier accounting mismatch");
        std::cout << std::setprecision(10) << "d4_1_microbench context=" << context
                  << " query=" << query << " prefix=" << trace.prefix_count
                  << " historical=" << trace.historical_count << " recent=" << trace.recent_count
                  << " setup_ms=" << setup_ms << " q_rotation_us=" << trace.q_rotation_us
                  << " int2_k_decode_us=" << trace.int2_k_decode_us
                  << " qk_us=" << trace.qk_us << " softmax_us=" << trace.softmax_us
                  << " int2_v_decode_us=" << trace.int2_v_decode_us << " av_us=" << trace.av_us
                  << " rv_inverse_us=" << trace.rv_inverse_us
                  << " reader_total_us=" << trace.total_us << '\n';
    };
    for (const std::uint32_t context : {512U, 2048U, 4096U}) { profile_context(context); }
}

std::string hex_hash(std::uint64_t value) {
    std::ostringstream output;
    output << "0x" << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

void write_metric(std::ofstream& output, const Metric& value) {
    output << "{\"max_abs\":" << std::setprecision(12) << value.max_abs
           << ",\"mean_abs\":" << value.mean_abs
           << ",\"relative_l2\":" << value.relative_l2 << '}';
}

void write_comparison(std::ofstream& output, const Comparison& value) {
    output << "{\"suite\":\"" << value.suite << "\",\"layer\":" << value.layer
           << ",\"context\":" << value.context << ",\"query\":" << value.query
           << ",\"prefix\":" << value.prefix << ",\"historical\":" << value.historical
           << ",\"recent\":" << value.recent << ",\"logical_positions\":[";
    for (std::size_t index = 0; index < value.logical_positions.size(); ++index) {
        if (index != 0) { output << ','; }
        output << value.logical_positions[index];
    }
    output << "],\"tier_sequence\":[";
    for (std::size_t index = 0; index < value.tiers.size(); ++index) {
        if (index != 0) { output << ','; }
        output << static_cast<unsigned>(value.tiers[index]);
    }
    output << "],\"rotated_q\":";
    write_metric(output, value.q);
    output << ",\"scores\":";
    write_metric(output, value.scores);
    output << ",\"softmax\":";
    write_metric(output, value.softmax);
    output << ",\"rotated_av\":";
    write_metric(output, value.rotated_av);
    output << ",\"recovered_output\":";
    write_metric(output, value.recovered);
    output << ",\"verdict\":\"PASS\"}";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path rotation_dir;
        std::string report_path;
        for (int index = 1; index + 1 < argc; ++index) {
            if (std::string(argv[index]) == "--rotation-dir") {
                rotation_dir = std::filesystem::path(argv[index + 1]);
            } else if (std::string(argv[index]) == "--report") {
                report_path = argv[index + 1];
            }
        }
        require(!rotation_dir.empty() && !report_path.empty(),
                "usage: --rotation-dir <C4 runtime dir> --report <json>");
        validate_runtime_manifest(rotation_dir);
        const auto k_bank = load_bank(rotation_dir / "k_rotation_fp32.bin");
        const auto v_bank = load_bank(rotation_dir / "v_rotation_fp32.bin");
        const char* microbench = std::getenv("NINFER_OSCAR_D4_1_MICROBENCH");
        if (microbench != nullptr && microbench[0] == '1') {
            run_d4_1_microbenchmark(k_bank, v_bank);
            return 0;
        }

        ReferenceInput static_input;
        static_input.tokens.reserve(332);
        for (std::uint32_t token = 0; token < 332; ++token) {
            static_input.tokens.push_back(make_rows(token, 0U));
        }
        ReferenceInput forced_input;
        forced_input.tokens.reserve(328);
        for (std::uint32_t token = 0; token < 324; ++token) {
            forced_input.tokens.push_back(static_input.tokens[token]);
        }
        for (std::uint32_t token = 324; token < 328; ++token) {
            forced_input.tokens.push_back(make_rows(token, 0xF00D0000U));
        }

        const auto asset = ninfer::OscarMixedAgingAssetContract::c4_cal30k();
        ninfer::OscarMixedAgingCacheBundle bundle324(7100U, asset,
                                                     ninfer::kOscarMixedFullAttentionLayers);
        for (std::uint32_t token = 0; token < 324; ++token) {
            bundle324.append(token, static_input.tokens[token].k, static_input.tokens[token].v);
        }
        ninfer::OscarMixedAgingCacheBundle bundle332(7101U, asset,
                                                     ninfer::kOscarMixedFullAttentionLayers);
        for (std::uint32_t token = 0; token < 332; ++token) {
            bundle332.append(token, static_input.tokens[token].k, static_input.tokens[token].v);
        }
        bundle324.validate();
        bundle332.validate();

        std::vector<Comparison> results;
        const std::vector<std::uint32_t> base_queries = {63, 64, 68, 323};
        for (const std::uint32_t query : base_queries) {
            results.push_back(compare("static-context-324", 0, 324, query,
                                      bundle324.layer(3), static_input, k_bank, v_bank));
        }
        const std::vector<std::uint32_t> final_queries = {63, 64, 68, 320, 331};
        for (const std::uint32_t query : final_queries) {
            results.push_back(compare("static-context-332", 0, 332, query,
                                      bundle332.layer(3), static_input, k_bank, v_bank));
        }
        for (const std::uint32_t query : {323U}) {
            results.push_back(compare("layer-35-static", 8, 324, query,
                                      bundle324.layer(35), static_input, k_bank, v_bank));
            results.push_back(compare("layer-63-static", 15, 324, query,
                                      bundle324.layer(63), static_input, k_bank, v_bank));
        }

        // Forced decode advances one persistent cache, so each reader call consumes the actual
        // post-aging typed representation for that step rather than a pre-aging shadow.
        ninfer::OscarMixedAgingLayerCache forced_decode_cache(3, 7103U, asset);
        for (std::uint32_t token = 0; token < 324; ++token) {
            const auto& rows = forced_input.tokens[token];
            forced_decode_cache.append(token,
                                       std::span<const std::uint16_t>(rows.k.data(), kRowValues),
                                       std::span<const std::uint16_t>(rows.v.data(), kRowValues));
        }
        for (std::uint32_t query = 324; query < 328; ++query) {
            const auto& rows = forced_input.tokens[query];
            forced_decode_cache.append(query,
                                       std::span<const std::uint16_t>(rows.k.data(), kRowValues),
                                       std::span<const std::uint16_t>(rows.v.data(), kRowValues));
            results.push_back(compare("forced-decode-layer-3", 0, query + 1U,
                                      query, forced_decode_cache, forced_input, k_bank, v_bank));
        }

        // Lightweight all-layer coverage at the final static context. Each layer gets its own
        // validated C4 matrices and exact full-attention cache rows; no GDN layer is constructed.
        std::vector<std::uint32_t> covered_layers;
        for (std::size_t layer = 0; layer < ninfer::kOscarMixedFullAttentionLayers.size(); ++layer) {
            results.push_back(compare("all-layer-static", layer, 332, 331,
                                      bundle332.layer(ninfer::kOscarMixedFullAttentionLayers[layer]),
                                      static_input, k_bank, v_bank));
            covered_layers.push_back(ninfer::kOscarMixedFullAttentionLayers[layer]);
        }
        require(covered_layers.size() == 16 &&
                    std::equal(covered_layers.begin(), covered_layers.end(),
                               ninfer::kOscarMixedFullAttentionLayers.begin()),
                "all-layer coverage is incomplete");

        double max_q = 0.0, max_scores = 0.0, max_softmax = 0.0;
        double max_av = 0.0, max_recovered = 0.0;
        for (const auto& result : results) {
            max_q = std::max(max_q, result.q.relative_l2);
            max_scores = std::max(max_scores, result.scores.relative_l2);
            max_softmax = std::max(max_softmax, result.softmax.relative_l2);
            max_av = std::max(max_av, result.rotated_av.relative_l2);
            max_recovered = std::max(max_recovered, result.recovered.relative_l2);
        }

        if (!report_path.empty()) {
            std::ofstream report(report_path, std::ios::binary);
            if (!report) { throw std::runtime_error("cannot open OSCAR attention report"); }
            report << "{\"schema\":\"oscar-d2-3a-reference-attention-v1\",\"status\":\"PASS\""
                   << ",\"asset_identity\":\"qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\""
                   << ",\"asset_manifest_sha256\":\"4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560\""
                   << ",\"reader\":\"OscarMixedAttentionReader\""
                   << ",\"independent_reference\":\"test-side source-archive reconstruction + official OscarInt2G128 decode\""
                   << ",\"gqa_ratio\":6,\"q_heads\":24,\"kv_heads\":4,\"head_dim\":256"
                   << ",\"legacy_q2_dispatched\":false,\"gdn_state_dispatched\":false"
                   << ",\"parity_tolerance_relative_l2\":" << kParityTolerance
                   << ",\"max_relative_l2\":{\"rotated_q\":" << max_q
                   << ",\"scores\":" << max_scores << ",\"softmax\":" << max_softmax
                   << ",\"rotated_av\":" << max_av << ",\"recovered_output\":"
                   << max_recovered << "},\"covered_layers\":[";
            for (std::size_t index = 0; index < covered_layers.size(); ++index) {
                if (index != 0) { report << ','; }
                report << covered_layers[index];
            }
            report << "],\"comparisons\":[";
            for (std::size_t index = 0; index < results.size(); ++index) {
                if (index != 0) { report << ','; }
                write_comparison(report, results[index]);
            }
            report << "]}\n";
        }

        std::cout << "OscarMixed attention reference parity: PASS\n"
                  << "  asset=qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\n"
                  << "  reader=temporary per-query row decode; persistent BF16 shadow=no\n"
                  << "  static boundary contexts=324,332; layer 3 queries=63,64,68,320,323,331\n"
                  << "  forced decode=4 appends (324..327), post-aging representation each step\n"
                  << "  layers=3,35,63 plus all 16 full-attention layers\n"
                  << "  max rel-L2 Q/scores/softmax/rotated-AV/recovered="
                  << std::scientific << max_q << '/' << max_scores << '/' << max_softmax << '/'
                  << max_av << '/' << max_recovered << "\n"
                  << "  tier boundary checks and no-GDN/no-legacy-Q2 dispatch: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OscarMixed attention reference parity: FAIL: " << error.what() << '\n';
        return 1;
    }
}
