// E4C1 Stage 8: eager full-model OSCAR qualification (canonical E1 oracle).
#include "exl3/text_model.h"
#include "exl3/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::exl3;
constexpr int kPrefillRows = 14;
constexpr int kHidden = 5120;
constexpr int kVocab = 248320;

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(operation);
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

float bf16_to_float(std::uint16_t bits) {
    float result = 0.0F;
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
    std::memcpy(&result, &expanded, sizeof(result));
    return result;
}

std::vector<float> as_float(const TensorPayload& tensor) {
    std::vector<float> result;
    if (tensor.info.dtype == "F16") {
        for (const auto bits : tensor.typed<std::uint16_t>("F16")) result.push_back(half_to_float(bits));
    } else if (tensor.info.dtype == "F32") {
        const auto values = tensor.typed<float>("F32");
        result.assign(values.begin(), values.end());
    } else if (tensor.info.dtype == "BF16") {
        for (const auto bits : tensor.typed<std::uint16_t>("BF16")) result.push_back(bf16_to_float(bits));
    } else {
        throw std::runtime_error("unsupported fixture dtype");
    }
    return result;
}

std::vector<std::int64_t> as_i64(const TensorPayload& tensor) {
    require(tensor.info.dtype == "I64", "fixture is not I64");
    const auto values = tensor.typed<std::int64_t>("I64");
    return {values.begin(), values.end()};
}

struct Metric {
    double max_abs = 0.0;
    double rms = 0.0;
    double relative_l2 = 0.0;
};

Metric compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "comparison size mismatch");
    Metric result;
    double squared_error = 0.0;
    double squared_expected = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "non-finite model output");
        const double error = static_cast<double>(actual[i]) - expected[i];
        result.max_abs = std::max(result.max_abs, std::abs(error));
        squared_error += error * error;
        squared_expected += static_cast<double>(expected[i]) * expected[i];
    }
    result.rms = std::sqrt(squared_error / actual.size());
    result.relative_l2 =
        squared_expected == 0.0 ? std::sqrt(squared_error) : std::sqrt(squared_error / squared_expected);
    return result;
}

int argmax(const std::vector<float>& values) {
    return static_cast<int>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

std::vector<std::int64_t> greedy_sequence(Exl3TextContext& context, int steps) {
    std::vector<std::int64_t> result;
    for (int step = 0; step < steps; ++step) {
        const auto token = static_cast<std::int64_t>(argmax(context.logits_host()));
        result.push_back(token);
        if (step + 1 < steps) context.decode(token);
    }
    return result;
}

}  // namespace

int main() {
    try {
        const auto target = env("NINFER_EXL3_TARGET_PATH");
        const auto oracle_path = env("NINFER_EXL3_ORACLE_PATH");
        if (target.empty() || oracle_path.empty()) {
            std::cerr << "E4C1E skipped: set NINFER_EXL3_TARGET_PATH and NINFER_EXL3_ORACLE_PATH\n";
            return 77;
        }
        const auto oracle = inspect_file(oracle_path);
        auto input_ids = as_i64(read_tensor(oracle_path, oracle, "input_ids"));
        const auto expected_logits = as_float(read_tensor(oracle_path, oracle, "final_logits"));
        const auto expected_tokens = as_i64(read_tensor(oracle_path, oracle, "greedy_tokens"));
        require(input_ids.size() == 15 && expected_tokens.size() == 16, "fixture dimensions changed");
        const auto model = Exl3TextModel::load(target, 256);
        auto context = model->create_context(true);
        require(context->try_enable_oscar_from_environment(), "E4C1E OSCAR enable failed");
        require(context->oscar_enabled(), "E4C1E OSCAR not enabled");
        context->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
        for (const int layer : {5, 19, 33, 47, 61}) {
            const auto actual = context->hidden_host(layer);
            TensorPayload expected_payload = read_tensor(
                oracle_path, oracle, "hidden_layer_" + std::to_string(layer));
            const auto expected = as_float(expected_payload);
            const std::vector<float> ref(expected.begin(),
                                         expected.begin() + static_cast<std::size_t>(kPrefillRows) * kHidden);
            const auto metric = compare(actual, ref);
            std::cout << "E4C1E hidden_layer_" << layer << " max_abs=" << std::setprecision(10)
                      << metric.max_abs << " rms=" << metric.rms << " relative_l2=" << metric.relative_l2
                      << "\n";
            require(metric.relative_l2 <= 0.01 && metric.max_abs <= 2.0,
                    "E4C1E hidden-state gate failed at layer " + std::to_string(layer));
        }
        context->decode(input_ids.back());
        const auto actual_logits = context->logits_host();
        const auto logits_metric = compare(actual_logits, expected_logits);
        std::cout << "E4C1E final_logits max_abs=" << logits_metric.max_abs
                  << " rms=" << logits_metric.rms << " relative_l2=" << logits_metric.relative_l2
                  << "\n";
        require(logits_metric.relative_l2 <= 0.05, "E4C1E final-logit gate failed");
        const auto first_token = static_cast<std::int64_t>(argmax(actual_logits));
        require(first_token == expected_tokens.front(), "E4C1E first greedy token mismatch");
        const auto native_tokens = greedy_sequence(*context, static_cast<int>(expected_tokens.size()));
        int matched = 0;
        for (std::size_t i = 0; i < expected_tokens.size(); ++i) {
            if (native_tokens[i] == expected_tokens[i]) {
                ++matched;
            } else {
                std::cerr << "E4C1E greedy mismatch at " << i << " expected=" << expected_tokens[i]
                          << " actual=" << native_tokens[i] << "\n";
            }
        }
        std::cout << "E4C1E greedy_tokens_matched=" << matched << "/" << expected_tokens.size()
                  << "\n";
        require(matched == static_cast<int>(expected_tokens.size()), "E4C1E greedy mismatch");
        context->reset();
        context->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
        context->decode(input_ids.back());
        require(greedy_sequence(*context, static_cast<int>(expected_tokens.size())) == native_tokens,
                "E4C1E reset/replay mismatch");
        context->reset();
        const std::vector<std::int64_t> prompt_b = {248045, 846, 198, 33963, 799};
        context->prefill(prompt_b);
        const auto prompt_b_first = static_cast<std::int64_t>(argmax(context->logits_host()));
        require(prompt_b_first >= 0 && prompt_b_first < kVocab, "E4C1E second prompt invalid token");
        context->decode(prompt_b_first);
        std::cout << "E4C1E reset_replay=PASS second_prompt=PASS\n";
        const std::vector<std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>> suite = {
            {{248045, 846, 198, 7734, 799, 11316, 883, 12050, 13, 248046, 198, 248045, 74455, 198},
             {248068, 198, 760, 1156}},
            {{248045, 846, 198, 7734, 12654, 884, 709, 13, 248046, 198, 248045, 74455, 198},
             {248068, 198, 760, 1156}},
            {{248045, 846, 198, 5423, 4566, 440, 5226, 321, 1698, 13, 248046, 198, 248045, 74455, 198},
             {248068, 198, 760, 1156}},
        };
        int suite_ok = 0;
        for (const auto& entry : suite) {
            auto c2 = model->create_context(false);
            require(c2->try_enable_oscar_from_environment(), "E4C1E suite OSCAR enable failed");
            c2->prefill(entry.first);
            std::vector<std::int64_t> actual;
            for (std::size_t s = 0; s < entry.second.size(); ++s) {
                const auto token = static_cast<std::int64_t>(argmax(c2->logits_host()));
                actual.push_back(token);
                if (s + 1 < entry.second.size()) c2->decode(token);
            }
            require(actual == entry.second, "E4C1E additional prompt greedy mismatch");
            ++suite_ok;
        }
        std::cout << "E4C1E additional_prompts=" << suite_ok << "/3\n";
        // Routing proof: exactly the 16 full-attention layers dispatch to OSCAR.
        const auto& telemetry = context->oscar_telemetry();
        int full_dispatched = 0;
        for (int layer = 0; layer < 64; ++layer) {
            const bool full = Exl3OscarContext::is_full_attention_layer(layer);
            const auto count = telemetry.oscar_dispatches[layer];
            if (full) {
                require(count > 0, "E4C1E full-attention layer never dispatched to OSCAR");
                ++full_dispatched;
            } else {
                require(count == 0, "E4C1E non-full-attention layer dispatched to OSCAR");
            }
        }
        std::uint64_t oscar_count = 0, ordinary_count = 0;
        context->oscar_routing_counts(oscar_count, ordinary_count);
        std::cout << "E4C1E oscar_full_layers=" << full_dispatched
                  << "/16 gdn_oscar=0 split_class=" << telemetry.last_split_class
                  << " appends=" << telemetry.append_calls
                  << " aging=" << telemetry.aging_events
                  << " layer_oscar=" << oscar_count << " layer_ordinary=" << ordinary_count
                  << " cache_MB=" << telemetry.resident_cache_bytes / 1048576
                  << " workspace_MB=" << telemetry.workspace_bytes / 1048576 << "\n";
        require(ordinary_count == 0, "E4C1E ordinary path ran on OSCAR context");
        // Graph capture must refuse while OSCAR is graph-unqualified (Stage 10+ lifts this).
        require(!context->capture_decode_graph(), "E4C1E graph capture must refuse with OSCAR");
        std::cout << "E4C1E graph_refusal=PASS status=" << context->graph_status() << "\n";
        // Destroy/recreate: fresh context + fresh OSCAR attach reproduces greedy.
        {
            auto c3 = model->create_context(false);
            require(c3->try_enable_oscar_from_environment(), "E4C1E recreate OSCAR enable failed");
            c3->prefill(std::span<const std::int64_t>(input_ids.data(), kPrefillRows));
            c3->decode(input_ids.back());
            require(greedy_sequence(*c3, static_cast<int>(expected_tokens.size())) == native_tokens,
                    "E4C1E destroy/recreate mismatch");
        }
        std::cout << "E4C1E destroy_recreate=PASS\n";
        std::cout << "E4C1E eager full-model OSCAR: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "E4C1E eager full-model OSCAR: FAIL: " << error.what() << "\n";
        return 1;
    }
}
