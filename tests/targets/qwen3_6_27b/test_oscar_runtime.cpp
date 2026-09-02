#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view kArtifactEnv = "NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS";
constexpr std::string_view kModelShaEnv = "NINFER_OSCAR_MODEL_SHA256";
constexpr std::string_view kDiagnosticEnv = "NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR";

const char* required_environment(std::string_view name) {
    std::string key(name);
    const char* value = std::getenv(key.c_str());
    if (value == nullptr || *value == '\0') { return nullptr; }
    return value;
}

std::uint32_t bf16_to_f32_bits(std::uint16_t value) {
    return static_cast<std::uint32_t>(value) << 16U;
}

std::vector<float> read_bf16(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("missing diagnostic file: " + path.string()); }
    std::vector<std::uint16_t> bits(count);
    input.read(reinterpret_cast<char*>(bits.data()),
               static_cast<std::streamsize>(bits.size() * sizeof(std::uint16_t)));
    if (input.gcount() != static_cast<std::streamsize>(bits.size() * sizeof(std::uint16_t)) ||
        !input) {
        throw std::runtime_error("truncated diagnostic file: " + path.string());
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("oversized diagnostic file: " + path.string());
    }
    std::vector<float> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = std::bit_cast<float>(bf16_to_f32_bits(bits[i]));
        if (!std::isfinite(result[i])) { throw std::runtime_error("non-finite diagnostic value"); }
    }
    return result;
}

struct Metrics {
    float max_abs = 0.0F;
    double mean_abs = 0.0;
    double relative_l2 = 0.0;
};

Metrics compare(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) { throw std::runtime_error("diagnostic shape mismatch"); }
    double sum_abs = 0.0;
    double sum_diff2 = 0.0;
    double sum_ref2 = 0.0;
    float max_abs = 0.0F;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = std::abs(a[i] - b[i]);
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
        sum_diff2 += static_cast<double>(diff) * diff;
        sum_ref2 += static_cast<double>(a[i]) * a[i];
    }
    return {max_abs, sum_abs / static_cast<double>(a.size()),
            std::sqrt(sum_diff2 / std::max(sum_ref2, std::numeric_limits<double>::min()))};
}

std::vector<std::size_t> top_k(const std::vector<float>& values, std::size_t k) {
    std::vector<std::size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(k),
                      indices.end(), [&](std::size_t a, std::size_t b) {
                          if (values[a] != values[b]) { return values[a] > values[b]; }
                          return a < b;
                      });
    indices.resize(k);
    return indices;
}

struct RunRecord {
    std::vector<ninfer::TokenId> generated;
    std::vector<float> hidden;
    std::vector<float> logits;
    double elapsed_ms = 0.0;
};

struct NaturalCase {
    std::string id;
    std::string category;
    std::string prompt;
    std::string expected;
    bool long_context = false;
};

struct NaturalRecord {
    std::size_t prompt_tokens = 0;
    std::vector<ninfer::TokenId> generated;
    std::string content;
    bool objective_success = false;
};

std::vector<ninfer::TokenId> make_prompt(std::size_t tokens) {
    std::vector<ninfer::TokenId> prompt(tokens);
    for (std::size_t i = 0; i < tokens; ++i) {
        prompt[i] = ninfer::TokenId{198 + static_cast<std::int32_t>((i * 131U) % 4096U)};
    }
    return prompt;
}

RunRecord run_one(std::string_view variant, std::size_t tokens,
                 const std::filesystem::path& diagnostic_dir,
                 const std::filesystem::path& artifact) {
    const bool d13 = required_environment("NINFER_OSCAR_D1_3_FORCED_DECODE") != nullptr;
    const bool d23 = required_environment("NINFER_OSCAR_D2_3B_LIVE") != nullptr;
    const bool d31 = required_environment("NINFER_OSCAR_D3_1_LIVE") != nullptr;
    const bool d41 = required_environment("NINFER_OSCAR_D4_1_PROFILE") != nullptr;
    const bool d43 = required_environment("NINFER_OSCAR_D4_3_LIVE") != nullptr;
    const bool d44 = required_environment("NINFER_OSCAR_D4_4_LIVE") != nullptr;
    const bool d45 = required_environment("NINFER_OSCAR_D4_5_LIVE") != nullptr;
    const bool d46 = required_environment("NINFER_OSCAR_D4_6_LIVE") != nullptr;
    const bool d43_perf_only = required_environment("NINFER_OSCAR_D4_3_PERF_ONLY") != nullptr;
    const bool d44_perf_no_oracle =
        required_environment("NINFER_OSCAR_D4_4_PERF_NO_ORACLE") != nullptr;
    const bool d45_perf_no_oracle =
        required_environment("NINFER_OSCAR_D4_5_PERF_NO_ORACLE") != nullptr;
    const bool gpu = variant == "oscar-int2-gpu" || variant == "oscar-int2-gpu-resident";
    const bool live = variant == "oscar-int2-reference-live" || gpu;
    const bool matched = variant == "matched-fp32-unrotated" ||
                         variant == "matched-fp32-rotated";
    const bool rotated = variant == "matched-fp32-rotated" ||
                         live ||
                         (variant != "normal" && variant != "production-bf16" && !matched);
    const std::string precision =
        variant == "matched-fp32-rotated" ? "fp32-rotation+inverse" : std::string(variant);
    const char* rotation_mode = variant == "oscar-int2-gpu-resident"
                                    ? "oscar-int2-gpu-resident"
                                    : (gpu ? "oscar-int2-gpu"
                                    : (live ? "oscar-int2-reference-live"
                                            : (rotated ? "oscar-rotated-bf16" : "")));
    if (_putenv_s("NINFER_OSCAR_ROTATION_MODE", rotation_mode) != 0) {
        throw std::runtime_error("could not set OSCAR rotation mode");
    }
    if (_putenv_s("NINFER_OSCAR_ROTATION_PRECISION",
                  rotated && !live ? precision.c_str() : "") != 0) {
        throw std::runtime_error("could not set OSCAR rotation precision");
    }
    if (_putenv_s("NINFER_OSCAR_MATCHED_FP32", matched ? "1" : "") != 0) {
        throw std::runtime_error("could not set matched FP32 mode");
    }
    if (_putenv_s("NINFER_OSCAR_D1_3_FORCED_DECODE_TOKENS",
                  d13 ? "997,1001,1003,1005,1007,1009,1011,1013,1015" : "") != 0) {
        throw std::runtime_error("could not set forced D1.3 decode tokens");
    }
    if (_putenv_s("NINFER_OSCAR_D2_3B_FORCED_DECODE_TOKENS",
                  (d23 || d31 || d43 || d44 || d45 || d46)
                      ? "997,1001,1003,1005,1007,1009,1011,1013,1015"
                      : "") != 0) {
        throw std::runtime_error("could not set forced D2.3b decode tokens");
    }
    if (_putenv_s("NINFER_OSCAR_D4_3_VALIDATE_REFERENCE",
                  (d43 && gpu && !d43_perf_only) ? "1" : "") != 0) {
        throw std::runtime_error("could not set D4.3 reference validation mode");
    }
    if (_putenv_s("NINFER_OSCAR_D4_4_VALIDATE_REFERENCE",
                  (d44 && gpu && !d43_perf_only && !d44_perf_no_oracle) ? "1" : "") != 0) {
        throw std::runtime_error("could not set D4.4 reference validation mode");
    }
    if (_putenv_s("NINFER_OSCAR_D4_5_VALIDATE_REFERENCE",
                  (d45 && gpu && !d43_perf_only && !d45_perf_no_oracle) ? "1" : "") != 0 ||
        _putenv_s("NINFER_OSCAR_D4_6_VALIDATE_REFERENCE",
                  (d46 && gpu && !d43_perf_only && !d45_perf_no_oracle) ? "1" : "") != 0) {
        throw std::runtime_error("could not set OSCAR fused reference validation mode");
    }
    const std::string prefix =
        (diagnostic_dir / std::string(variant) / std::to_string(tokens)).string();
    std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
    if (_putenv_s("NINFER_OSCAR_RUNTIME_DIAGNOSTIC_PREFIX", prefix.c_str()) != 0) {
        throw std::runtime_error("could not set OSCAR diagnostic prefix");
    }
    if (_putenv_s("NINFER_OSCAR_RUNTIME_LAYER_DIAGNOSTIC_PREFIX", prefix.c_str()) != 0) {
        throw std::runtime_error("could not set OSCAR layer diagnostic prefix");
    }
    if (_putenv_s("NINFER_OSCAR_RUNTIME_ATTENTION_DIAGNOSTIC_PREFIX", prefix.c_str()) != 0) {
        throw std::runtime_error("could not set OSCAR attention diagnostic prefix");
    }
    const std::filesystem::path tap_dir =
        diagnostic_dir / "live_reference_taps" / std::to_string(tokens);
    if (live && (d23 || d44 || d45 || d46)) {
        std::filesystem::create_directories(tap_dir);
        if (_putenv_s("NINFER_OSCAR_LIVE_REFERENCE_TAP_DIR", tap_dir.string().c_str()) != 0 ||
            _putenv_s("NINFER_OSCAR_LIVE_REFERENCE_TAP_LAYERS",
                      tokens <= 32 ? "3" : "3,35,63") != 0 ||
            _putenv_s("NINFER_OSCAR_LIVE_REFERENCE_TAP_QUERIES",
                      tokens <= 32 ? "0,31" : "63,64,68,319,320,323,324,325,326,327") != 0) {
            throw std::runtime_error("could not set OSCAR live reference tap environment");
        }
    } else {
        _putenv_s("NINFER_OSCAR_LIVE_REFERENCE_TAP_DIR", "");
        _putenv_s("NINFER_OSCAR_LIVE_REFERENCE_TAP_LAYERS", "");
        _putenv_s("NINFER_OSCAR_LIVE_REFERENCE_TAP_QUERIES", "");
    }

    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    const std::size_t capacity = std::max<std::size_t>(4096, tokens + 16);
    options.max_context          = static_cast<std::uint32_t>(capacity);
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(
        static_cast<std::uint32_t>(capacity));
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.prefill_chunk        = 256;
    options.kv_cache             = ninfer::KvCacheStorage::BFloat16;
    options.speculative.backend  = ninfer::SpeculativeBackend::None;
    options.enable_vision        = false;
    options.use_cuda_graph       = false;
    options.context_cache.enabled = false;

    ninfer::Engine engine(std::move(options));
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens =
        (d13 || d23 || d31 || d43 || d44 || d45 || d46) ? 9 : (d41 ? 1 : 2);
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.top_k          = 1;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    const auto verifier_start = std::chrono::steady_clock::now();
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(make_prompt(tokens), false), request);
    const double verifier_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - verifier_start)
                                   .count();

    RunRecord record;
    record.generated = result.generated_token_ids;
    record.elapsed_ms = verifier_ms;
    record.hidden = read_bf16(prefix + ".hidden.bf16", 5120);
    record.logits = read_bf16(prefix + ".logits.bf16", 248320);
    return record;
}

std::string long_context_prompt(std::string_view fact, std::string_view question) {
    std::string prompt = "Remember this exact fact: ";
    prompt += fact;
    prompt += "\n\nThe following background is irrelevant but must remain in context:\n";
    for (int index = 0; index < 48; ++index) {
        prompt += "Background note ";
        prompt += std::to_string(index);
        prompt += ": calibration and retrieval tests use deterministic numbered records. ";
        prompt += "Do not replace the remembered fact.\n";
    }
    prompt += "\nQuestion: ";
    prompt += question;
    return prompt;
}

std::vector<NaturalCase> make_natural_cases() {
    return {
        {"arith-01", "reasoning/arithmetic", "Compute 17 + 25. Reply with only the integer 42.",
         "42", false},
        {"arith-02", "reasoning/arithmetic", "A box has 6 rows of 4 items. Reply with only 24.",
         "24", false},
        {"arith-03", "reasoning/arithmetic", "Which number is larger: 19 or 91? Reply only 91.",
         "91", false},
        {"code-01", "coding", "Return exactly this line and nothing else: int answer = 42;",
         "int answer = 42;", false},
        {"code-02", "coding", "Write a one-line Python expression that doubles x. Include `x * 2`.",
         "x * 2", false},
        {"code-03", "coding", "Return the exact C++ statement `return value;` and no explanation.",
         "return value;", false},
        {"json-01", "structured", "Return only this JSON object: {\"answer\":42,\"ok\":true}",
         "\"answer\":42", false},
        {"json-02", "structured", "Return valid JSON with the exact fields \"kind\":\"test\" and \"n\":7.",
         "\"kind\":\"test\"", false},
        {"retrieve-01", "retrieval", long_context_prompt(
                           "OSCAR-RETRIEVAL-7F3A",
                           "Reply with exactly the remembered identifier and nothing else."),
         "OSCAR-RETRIEVAL-7F3A", true},
        {"retrieve-02", "retrieval", long_context_prompt(
                           "QWEN-CAL-30K-ALPHA",
                           "What exact key was stored at the beginning? Copy it exactly."),
         "QWEN-CAL-30K-ALPHA", true},
        {"copy-01", "copy/exactness", long_context_prompt(
                       "VOIDINFER-COPY-2048",
                       "Copy the identifier exactly, preserving every character."),
         "VOIDINFER-COPY-2048", true},
        {"copy-02", "copy/exactness", long_context_prompt(
                       "phase-D3.1::g128::stable",
                       "Repeat the exact short string from the first line."),
         "phase-D3.1::g128::stable", true},
    };
}

bool natural_objective_success(const NaturalCase& test, std::string_view content) {
    if (test.category == "structured") {
        if (content.find('{') == std::string_view::npos) { return false; }
        if (test.id == "json-01") {
            return (content.find("\"answer\":42") != std::string_view::npos ||
                    content.find("\"answer\": 42") != std::string_view::npos) &&
                   (content.find("\"ok\":true") != std::string_view::npos ||
                    content.find("\"ok\": true") != std::string_view::npos);
        }
        if (test.id == "json-02") {
            return (content.find("\"kind\":\"test\"") != std::string_view::npos ||
                    content.find("\"kind\": \"test\"") != std::string_view::npos) &&
                   (content.find("\"n\":7") != std::string_view::npos ||
                    content.find("\"n\": 7") != std::string_view::npos);
        }
        return content.find(test.expected) != std::string_view::npos;
    }
    return content.find(test.expected) != std::string_view::npos;
}

std::vector<NaturalRecord> run_natural_suite(
    std::string_view variant, const std::filesystem::path& artifact) {
    const bool live = variant == "oscar-int2-reference-live" || variant == "oscar-int2-gpu";
    const char* rotation_mode = variant == "oscar-int2-gpu"
                                    ? "oscar-int2-gpu"
                                    : (live ? "oscar-int2-reference-live" : "");
    if (_putenv_s("NINFER_OSCAR_ROTATION_MODE", rotation_mode) != 0 ||
        _putenv_s("NINFER_OSCAR_ROTATION_PRECISION", "") != 0 ||
        _putenv_s("NINFER_OSCAR_MATCHED_FP32", "") != 0 ||
        _putenv_s("NINFER_OSCAR_D1_3_FORCED_DECODE_TOKENS", "") != 0 ||
        _putenv_s("NINFER_OSCAR_D2_3B_FORCED_DECODE_TOKENS", "") != 0 ||
        _putenv_s("NINFER_OSCAR_LIVE_REFERENCE_TAP_DIR", "") != 0) {
        throw std::runtime_error("could not configure D3.1 natural suite mode");
    }
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.prefill_chunk        = 256;
    options.kv_cache              = ninfer::KvCacheStorage::BFloat16;
    options.speculative.backend   = ninfer::SpeculativeBackend::None;
    options.enable_vision        = false;
    options.use_cuda_graph       = false;
    options.context_cache.enabled = false;
    ninfer::Engine engine(std::move(options));

    ninfer::RequestOptions request;
    std::size_t natural_output_tokens = 48;
    if (const char* configured = required_environment("NINFER_OSCAR_D3_1_NATURAL_OUTPUT")) {
        try {
            natural_output_tokens = std::max<std::size_t>(1, std::stoull(configured));
        } catch (...) {
            throw std::runtime_error("invalid NINFER_OSCAR_D3_1_NATURAL_OUTPUT");
        }
    }
    request.execution.requested_output_tokens = natural_output_tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.top_k          = 1;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;

    std::vector<NaturalCase> cases = make_natural_cases();
    const char* small_suite = std::getenv("NINFER_OSCAR_D4_3_NATURAL_SMALL");
    if (small_suite != nullptr && std::string_view(small_suite) == "1") {
        const std::array<std::string_view, 4> selected{
            "arith-01", "code-01", "json-01", "retrieve-01"};
        cases.erase(std::remove_if(cases.begin(), cases.end(), [&](const NaturalCase& test) {
                        return std::find(selected.begin(), selected.end(), test.id) ==
                               selected.end();
                    }),
                    cases.end());
    }
    std::vector<NaturalRecord> records;
    for (const NaturalCase& test : cases) {
        // Keep long-context cases bounded in the slow reference route, but allow
        // structured cases enough tokens to close their JSON object before
        // applying the syntax/objective check.
        request.execution.requested_output_tokens =
            test.category == "structured" ? std::max<std::size_t>(natural_output_tokens, 48)
                                           : natural_output_tokens;
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = test.prompt, .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking = false;
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare(std::move(input)), request);
        NaturalRecord record;
        record.prompt_tokens = result.prompt.prompt_tokens;
        record.generated = result.generated_token_ids;
        record.content = result.content;
        record.objective_success = natural_objective_success(test, record.content);
        records.push_back(record);
        std::cout << "d3_1_natural mode=" << variant << " case=" << test.id
                  << " category=" << test.category << " prompt_tokens=" << record.prompt_tokens
                  << " long_context=" << (test.long_context ? "true" : "false")
                  << " objective_success=" << (record.objective_success ? "true" : "false")
                  << " generated_tokens=" << record.generated.size()
                  << " content=" << std::quoted(record.content) << '\n';
    }
    return records;
}

double softmax_probability(const std::vector<float>& logits, std::size_t index) {
    const float maximum = *std::max_element(logits.begin(), logits.end());
    long double denominator = 0.0L;
    for (const float value : logits) {
        denominator += std::exp(static_cast<long double>(value - maximum));
    }
    return static_cast<double>(std::exp(static_cast<long double>(logits[index] - maximum)) /
                               denominator);
}

std::size_t contained_count(const std::vector<std::size_t>& expected,
                            const std::vector<std::size_t>& actual) {
    return static_cast<std::size_t>(std::count_if(
        expected.begin(), expected.end(), [&](std::size_t value) {
            return std::find(actual.begin(), actual.end(), value) != actual.end();
        }));
}

void print_fixed_fidelity(std::size_t tokens, const RunRecord& bf16, const RunRecord& oscar) {
    const Metrics hidden = compare(bf16.hidden, oscar.hidden);
    const Metrics logits = compare(bf16.logits, oscar.logits);
    const auto bf16_top10 = top_k(bf16.logits, 10);
    const auto oscar_top10 = top_k(oscar.logits, 10);
    const auto bf16_top5 = std::vector<std::size_t>(bf16_top10.begin(), bf16_top10.begin() + 5);
    const auto oscar_top5 = std::vector<std::size_t>(oscar_top10.begin(), oscar_top10.begin() + 5);
    const std::size_t bf16_top1 = bf16_top10.front();
    const std::size_t bf16_top2 = bf16_top10.at(1);
    const double bf16_probability = softmax_probability(bf16.logits, bf16_top1);
    const double oscar_probability = softmax_probability(oscar.logits, bf16_top1);
    std::size_t first_token_difference = 0;
    while (first_token_difference < bf16.generated.size() &&
           first_token_difference < oscar.generated.size() &&
           bf16.generated[first_token_difference] == oscar.generated[first_token_difference]) {
        ++first_token_difference;
    }
    if (first_token_difference == bf16.generated.size() &&
        first_token_difference == oscar.generated.size()) {
        first_token_difference = std::numeric_limits<std::size_t>::max();
    }
    std::cout << std::setprecision(10) << "d3_1_fixed case_tokens=" << tokens
              << " hidden_relative_l2=" << hidden.relative_l2
              << " logits_max_abs=" << logits.max_abs
              << " logits_mean_abs=" << logits.mean_abs
              << " logits_relative_l2=" << logits.relative_l2
              << " top1_agree=" << (bf16_top1 == oscar_top10.front() ? "true" : "false")
              << " top5_containment=" << contained_count(bf16_top5, oscar_top5)
              << " top10_overlap=" << contained_count(bf16_top10, oscar_top10)
              << " bf16_top1_margin=" << (bf16.logits[bf16_top1] - bf16.logits[bf16_top2])
              << " bf16_top1_logit_change=" << (oscar.logits[bf16_top1] - bf16.logits[bf16_top1])
              << " bf16_top1_probability_change=" << (oscar_probability - bf16_probability)
              << " generated_agree=" << (first_token_difference == std::numeric_limits<std::size_t>::max()
                                              ? "true"
                                              : "false")
              << " first_generated_difference="
              << (first_token_difference == std::numeric_limits<std::size_t>::max()
                      ? -1
                      : static_cast<long long>(first_token_difference))
              << '\n';
}

} // namespace

int main() {
    const char* artifact_value = required_environment(kArtifactEnv);
    const char* model_sha       = required_environment(kModelShaEnv);
    const char* diagnostic_root = required_environment(kDiagnosticEnv);
    if (artifact_value == nullptr || model_sha == nullptr || diagnostic_root == nullptr) {
        std::cout << "skip: artifact, NINFER_OSCAR_MODEL_SHA256 and diagnostic directory are required\n";
        return 77;
    }
    try {
        const std::filesystem::path artifact(artifact_value);
        const std::filesystem::path diagnostic_dir(diagnostic_root);
        const bool d13 = required_environment("NINFER_OSCAR_D1_3_FORCED_DECODE") != nullptr;
        const bool d23 = required_environment("NINFER_OSCAR_D2_3B_LIVE") != nullptr;
        const bool d31 = required_environment("NINFER_OSCAR_D3_1_LIVE") != nullptr;
        const bool d31_natural =
            required_environment("NINFER_OSCAR_D3_1_NATURAL") != nullptr;
        const bool d41 = required_environment("NINFER_OSCAR_D4_1_PROFILE") != nullptr;
        const bool d43 = required_environment("NINFER_OSCAR_D4_3_LIVE") != nullptr;
        const bool d44 = required_environment("NINFER_OSCAR_D4_4_LIVE") != nullptr;
        const bool d45 = required_environment("NINFER_OSCAR_D4_5_LIVE") != nullptr;
        const bool d46 = required_environment("NINFER_OSCAR_D4_6_LIVE") != nullptr;
        const bool d44_perf_no_oracle =
            required_environment("NINFER_OSCAR_D4_4_PERF_NO_ORACLE") != nullptr;
        const bool d45_perf_no_oracle =
            required_environment("NINFER_OSCAR_D4_5_PERF_NO_ORACLE") != nullptr;
        const bool d43_natural = required_environment("NINFER_OSCAR_D4_3_NATURAL") != nullptr;
        if (d43_natural) {
            _putenv_s("NINFER_OSCAR_D4_3_LIVE", "1");
            _putenv_s("NINFER_OSCAR_D4_3_PERF_ONLY", "1");
            _putenv_s("NINFER_OSCAR_D4_1_PROFILE", "1");
            const auto control = run_natural_suite("production-bf16", artifact);
            const auto oscar = run_natural_suite("oscar-int2-gpu", artifact);
            if (control.size() != oscar.size()) {
                throw std::runtime_error("D4.3 natural suite size mismatch");
            }
            std::size_t control_success = 0;
            std::size_t oscar_success = 0;
            std::size_t long_context_cases = 0;
            for (std::size_t index = 0; index < control.size(); ++index) {
                control_success += control[index].objective_success ? 1U : 0U;
                oscar_success += oscar[index].objective_success ? 1U : 0U;
                long_context_cases += control[index].prompt_tokens > 1000 ? 1U : 0U;
                std::size_t first_difference = 0;
                while (first_difference < control[index].generated.size() &&
                       first_difference < oscar[index].generated.size() &&
                       control[index].generated[first_difference] ==
                           oscar[index].generated[first_difference]) {
                    ++first_difference;
                }
                std::cout << "d4_3_natural_compare index=" << index
                          << " prompt_tokens=" << control[index].prompt_tokens
                          << " long_context="
                          << (control[index].prompt_tokens > 1000 ? "true" : "false")
                          << " control_success="
                          << (control[index].objective_success ? "true" : "false")
                          << " oscar_success="
                          << (oscar[index].objective_success ? "true" : "false")
                          << " first_token_difference=" << first_difference << '\n';
            }
            std::cout << "d4_3_natural_summary cases=" << control.size()
                      << " long_context_cases=" << long_context_cases
                      << " control_success=" << control_success
                      << " oscar_success=" << oscar_success << '\n';
            _putenv_s("NINFER_OSCAR_D4_3_NATURAL", "");
            _putenv_s("NINFER_OSCAR_D4_3_PERF_ONLY", "");
            _putenv_s("NINFER_OSCAR_D4_3_LIVE", "");
            _putenv_s("NINFER_OSCAR_D4_1_PROFILE", "");
            _putenv_s("NINFER_OSCAR_ROTATION_MODE", "");
            return 0;
        }
        if (d31_natural) {
            const auto control = run_natural_suite("production-bf16", artifact);
            const auto oscar = run_natural_suite("oscar-int2-reference-live", artifact);
            if (control.size() != oscar.size()) {
                throw std::runtime_error("D3.1 natural suite size mismatch");
            }
            std::size_t control_success = 0;
            std::size_t oscar_success = 0;
            std::size_t long_context_cases = 0;
            for (std::size_t index = 0; index < control.size(); ++index) {
                control_success += control[index].objective_success ? 1U : 0U;
                oscar_success += oscar[index].objective_success ? 1U : 0U;
                const bool long_context = control[index].prompt_tokens > 320;
                long_context_cases += long_context ? 1U : 0U;
                std::size_t first_difference = 0;
                while (first_difference < control[index].generated.size() &&
                       first_difference < oscar[index].generated.size() &&
                       control[index].generated[first_difference] ==
                           oscar[index].generated[first_difference]) {
                    ++first_difference;
                }
                if (first_difference == control[index].generated.size() &&
                    first_difference == oscar[index].generated.size()) {
                    first_difference = std::numeric_limits<std::size_t>::max();
                }
                std::cout << "d3_1_natural_compare index=" << index
                          << " prompt_tokens=" << control[index].prompt_tokens
                          << " long_context=" << (long_context ? "true" : "false")
                          << " control_success="
                          << (control[index].objective_success ? "true" : "false")
                          << " oscar_success="
                          << (oscar[index].objective_success ? "true" : "false")
                          << " control_output_tokens=" << control[index].generated.size()
                          << " oscar_output_tokens=" << oscar[index].generated.size()
                          << " first_token_difference="
                          << (first_difference == std::numeric_limits<std::size_t>::max()
                                  ? -1
                                  : static_cast<long long>(first_difference))
                          << '\n';
            }
            std::cout << "d3_1_natural_summary cases=" << control.size()
                      << " long_context_cases=" << long_context_cases
                      << " control_success=" << control_success
                      << " oscar_success=" << oscar_success << '\n';
            _putenv_s("NINFER_OSCAR_D3_1_NATURAL", "");
            _putenv_s("NINFER_OSCAR_D3_1_LIVE", "");
            _putenv_s("NINFER_OSCAR_ROTATION_MODE", "");
            return 0;
        }
        if (d31) {
            const std::vector<ninfer::TokenId> expected{
                ninfer::TokenId{997},  ninfer::TokenId{1001}, ninfer::TokenId{1003},
                ninfer::TokenId{1005}, ninfer::TokenId{1007}, ninfer::TokenId{1009},
                ninfer::TokenId{1011}, ninfer::TokenId{1013}, ninfer::TokenId{1015}};
            std::vector<std::size_t> cases{32, 324, 512};
            const char* include_1024 = std::getenv("NINFER_OSCAR_D3_1_INCLUDE_1024");
            if (include_1024 != nullptr && std::string_view(include_1024) == "1") {
                cases.push_back(1024);
            }
            for (const std::size_t tokens : cases) {
                const RunRecord control =
                    run_one("production-bf16", tokens, diagnostic_dir, artifact);
                const RunRecord oscar =
                    run_one("oscar-int2-reference-live", tokens, diagnostic_dir, artifact);
                if (control.generated != expected || oscar.generated != expected) {
                    throw std::runtime_error("D3.1 forced continuation was not committed exactly");
                }
                print_fixed_fidelity(tokens, control, oscar);
            }
            _putenv_s("NINFER_OSCAR_D2_3B_FORCED_DECODE_TOKENS", "");
            _putenv_s("NINFER_OSCAR_D3_1_LIVE", "");
            _putenv_s("NINFER_OSCAR_ROTATION_MODE", "");
            std::cout << "OSCAR D3.1 fixed-token fidelity suite: COMPLETE\n";
            return 0;
        }
        if (d43 || d44 || d45 || d46) {
            // D4.3/D4.4 own their profiling switch so the GPU paths cannot accidentally fall
            // through the older scalar D4.1 harness. D4.4 selects the resident cache explicitly.
            _putenv_s("NINFER_OSCAR_D4_1_PROFILE", "1");
            const std::string_view gpu_variant = (d44 || d45 || d46)
                                                     ? "oscar-int2-gpu-resident"
                                                     : "oscar-int2-gpu";
            const std::vector<ninfer::TokenId> expected{
                ninfer::TokenId{997},  ninfer::TokenId{1001}, ninfer::TokenId{1003},
                ninfer::TokenId{1005}, ninfer::TokenId{1007}, ninfer::TokenId{1009},
                ninfer::TokenId{1011}, ninfer::TokenId{1013}, ninfer::TokenId{1015}};
            const auto print_gpu_comparison = [&](std::string_view left_name,
                                                   const RunRecord& left,
                                                   std::string_view right_name,
                                                   const RunRecord& right, std::size_t tokens) {
                const Metrics hidden = compare(left.hidden, right.hidden);
                const Metrics logits = compare(left.logits, right.logits);
                const auto left_top10 = top_k(left.logits, 10);
                const auto right_top10 = top_k(right.logits, 10);
                const std::size_t top10_agreement = static_cast<std::size_t>(std::count_if(
                    left_top10.begin(), left_top10.end(), [&](std::size_t value) {
                        return std::find(right_top10.begin(), right_top10.end(), value) !=
                               right_top10.end();
                    }));
                std::cout << std::setprecision(9) << "d4_3_compare=" << left_name << "_vs_"
                          << right_name << " case_tokens=" << tokens
                          << " left_ms=" << left.elapsed_ms << " right_ms=" << right.elapsed_ms
                          << " hidden_max_abs=" << hidden.max_abs
                          << " hidden_mean_abs=" << hidden.mean_abs
                          << " hidden_relative_l2=" << hidden.relative_l2
                          << " logits_max_abs=" << logits.max_abs
                          << " logits_mean_abs=" << logits.mean_abs
                          << " logits_relative_l2=" << logits.relative_l2
                          << " top1_agree="
                          << (left_top10.front() == right_top10.front() ? "true" : "false")
                          << " top10_agree_count=" << top10_agreement
                          << " forced_decode_agree="
                          << (left.generated == right.generated ? "true" : "false") << '\n';
                if (!std::isfinite(hidden.relative_l2) || !std::isfinite(logits.relative_l2)) {
                    throw std::runtime_error("D4.3 non-finite model output");
                }
            };

            const bool perf_only = [] {
                const char* value = std::getenv("NINFER_OSCAR_D4_3_PERF_ONLY");
                return value != nullptr && std::string_view(value) == "1";
            }();

            // These are the required first mixed-cache contexts. The GPU branch itself
            // performs the selected layer-3/35/63 live-vs-scalar oracle tap on the final
            // prefill/decode query, while this harness records model-level diagnostics.
            for (const std::size_t tokens : (perf_only
                                                  ? std::vector<std::size_t>{}
                                                  : std::vector<std::size_t>{321, 332, 512})) {
                const RunRecord control =
                    run_one("production-bf16", tokens, diagnostic_dir, artifact);
                const RunRecord gpu = run_one(gpu_variant, tokens, diagnostic_dir, artifact);
                if (control.generated != expected || gpu.generated != expected) {
                    throw std::runtime_error("D4.6 forced continuation was not committed exactly");
                }
                print_gpu_comparison("production-bf16", control, gpu_variant, gpu, tokens);
                if (tokens == 321 && !d44_perf_no_oracle && !d45_perf_no_oracle) {
                    // One explicit scalar live run anchors the optimized mode to the
                    // already-qualified oscar-int2-reference-live contract. Larger scalar
                    // contexts are covered by the in-process attention oracle to avoid
                    // spending the entire smoke run in the intentionally slow reader.
                    const RunRecord reference =
                        run_one("oscar-int2-reference-live", tokens, diagnostic_dir, artifact);
                    if (reference.generated != expected) {
                        throw std::runtime_error("D4.3 scalar reference continuation mismatch");
                    }
                    print_gpu_comparison("oscar-int2-reference-live", reference,
                                         gpu_variant, gpu, tokens);
                }
            }

            if (d44 || d45 || d46) {
                const std::size_t tokens = 2048;
                const RunRecord control =
                    run_one("production-bf16", tokens, diagnostic_dir, artifact);
                const RunRecord gpu = run_one(gpu_variant, tokens, diagnostic_dir, artifact);
                if (gpu.generated != expected) {
                    throw std::runtime_error("D4.4 2K forced continuation mismatch");
                }
                print_gpu_comparison("production-bf16", control, gpu_variant, gpu, tokens);
            }

            const bool include_16k = [] {
                const char* value = std::getenv("NINFER_OSCAR_D4_3_INCLUDE_16K");
                return value != nullptr && std::string_view(value) == "1";
            }();
            const bool include_8k_d45 = [&] {
                const char* value = std::getenv("NINFER_OSCAR_D4_5_INCLUDE_8K");
                return (d45 || d46) && value != nullptr && std::string_view(value) == "1";
            }();
            const bool include_16k_d45 = [&] {
                const char* value = std::getenv("NINFER_OSCAR_D4_5_INCLUDE_16K");
                return (d45 || d46) && value != nullptr && std::string_view(value) == "1";
            }();
            const bool include_32k = [] {
                const char* value = std::getenv("NINFER_OSCAR_D4_3_INCLUDE_32K");
                return value != nullptr && std::string_view(value) == "1";
            }();
            std::vector<std::size_t> large_cases;
            if (!perf_only) { large_cases.push_back(4096); }
            if (include_8k_d45) { large_cases.push_back(8192); }
            if (include_16k || include_16k_d45 || perf_only) { large_cases.push_back(16384); }
            if (include_32k) { large_cases.push_back(32768); }
            for (const std::size_t tokens : large_cases) {
                if (tokens == 0) { continue; }
                const RunRecord control =
                    run_one("production-bf16", tokens, diagnostic_dir, artifact);
                const RunRecord gpu = run_one(gpu_variant, tokens, diagnostic_dir, artifact);
                if (gpu.generated != expected) {
                    throw std::runtime_error("D4.3 large-context forced continuation mismatch");
                }
                print_gpu_comparison("production-bf16", control, gpu_variant, gpu, tokens);
                std::cout << std::setprecision(9)
                          << (d46 ? "d4_6_gpu_profile case_tokens="
                                  : (d45 ? "d4_5_gpu_profile case_tokens="
                                         : (d44 ? "d4_4_gpu_profile case_tokens="
                                                : "d4_3_gpu_profile case_tokens=")))
                          << tokens
                          << " verifier_ms=" << gpu.elapsed_ms
                          << " control_ms=" << control.elapsed_ms
                          << " generated_tokens=" << gpu.generated.size() << '\n';
            }
            _putenv_s("NINFER_OSCAR_D4_3_VALIDATE_REFERENCE", "");
            _putenv_s("NINFER_OSCAR_D4_4_VALIDATE_REFERENCE", "");
            _putenv_s("NINFER_OSCAR_D4_5_VALIDATE_REFERENCE", "");
            _putenv_s("NINFER_OSCAR_D4_6_VALIDATE_REFERENCE", "");
            _putenv_s("NINFER_OSCAR_D4_4_PERF_NO_ORACLE", "");
            _putenv_s("NINFER_OSCAR_D4_5_PERF_NO_ORACLE", "");
            _putenv_s("NINFER_OSCAR_D4_3_LIVE", "");
            _putenv_s("NINFER_OSCAR_D4_4_LIVE", "");
            _putenv_s("NINFER_OSCAR_D4_5_LIVE", "");
            _putenv_s("NINFER_OSCAR_D4_6_LIVE", "");
            _putenv_s("NINFER_OSCAR_D4_1_PROFILE", "");
            _putenv_s("NINFER_OSCAR_D2_3B_FORCED_DECODE_TOKENS", "");
            _putenv_s("NINFER_OSCAR_ROTATION_MODE", "");
            _putenv_s("NINFER_OSCAR_RUNTIME_DIAGNOSTIC_PREFIX", "");
            std::cout << (d46 ? "OSCAR D4.6 fused live GPU runtime: COMPLETE\n"
                              : (d45 ? "OSCAR D4.5 batched live GPU runtime: COMPLETE\n"
                                     : "OSCAR D4.3 live GPU runtime: COMPLETE\n"));
            return 0;
        }
        if (d41) {
            const std::vector<std::size_t> cases{512, 2048, 4096};
            for (const std::size_t tokens : cases) {
                const RunRecord live =
                    run_one("oscar-int2-reference-live", tokens, diagnostic_dir, artifact);
                std::cout << std::setprecision(9) << "d4_1_profile case_tokens=" << tokens
                          << " verifier_ms=" << live.elapsed_ms
                          << " generated_tokens=" << live.generated.size() << '\n';
            }
            _putenv_s("NINFER_OSCAR_D4_1_PROFILE", "");
            _putenv_s("NINFER_OSCAR_ROTATION_MODE", "");
            _putenv_s("NINFER_OSCAR_RUNTIME_DIAGNOSTIC_PREFIX", "");
            std::cout << "OSCAR D4.1 live profile: COMPLETE\n";
            return 0;
        }
        const bool d12 = required_environment("NINFER_OSCAR_D1_2_MATCHED") != nullptr;
        if (d23) {
            const std::vector<ninfer::TokenId> expected{
                ninfer::TokenId{997},  ninfer::TokenId{1001}, ninfer::TokenId{1003},
                ninfer::TokenId{1005}, ninfer::TokenId{1007}, ninfer::TokenId{1009},
                ninfer::TokenId{1011}, ninfer::TokenId{1013}, ninfer::TokenId{1015}};
            for (const std::size_t tokens : {std::size_t{32}, std::size_t{324}}) {
                const RunRecord control =
                    run_one("production-bf16", tokens, diagnostic_dir, artifact);
                const RunRecord live = run_one("oscar-int2-reference-live", tokens,
                                               diagnostic_dir, artifact);
                if (control.generated != expected || live.generated != expected) {
                    throw std::runtime_error("D2.3b forced continuation was not committed exactly");
                }
                const Metrics hidden = compare(control.hidden, live.hidden);
                const Metrics logits = compare(control.logits, live.logits);
                const auto control_top10 = top_k(control.logits, 10);
                const auto live_top10 = top_k(live.logits, 10);
                const std::size_t top10_agreement = static_cast<std::size_t>(std::count_if(
                    control_top10.begin(), control_top10.end(), [&](std::size_t value) {
                        return std::find(live_top10.begin(), live_top10.end(), value) !=
                               live_top10.end();
                    }));
                std::cout << std::setprecision(9) << "d2_3b_model_diagnostic case_tokens=" << tokens
                          << " hidden_relative_l2=" << hidden.relative_l2
                          << " logits_relative_l2=" << logits.relative_l2
                          << " top1_agree=" << (control_top10.front() == live_top10.front() ? "true" : "false")
                          << " top10_agree_count=" << top10_agreement
                          << " forced_decode_tokens=997,1001,1003,1005,1007,1009,1011,1013,1015\n";
            }
            _putenv_s("NINFER_OSCAR_D2_3B_FORCED_DECODE_TOKENS", "");
            _putenv_s("NINFER_OSCAR_D2_3B_LIVE", "");
            _putenv_s("NINFER_OSCAR_ROTATION_MODE", "");
            _putenv_s("NINFER_OSCAR_RUNTIME_DIAGNOSTIC_PREFIX", "");
            std::cout << "OSCAR D2.3b live reference runtime: COMPLETE\n";
            return 0;
        }
        if (d13) {
            const RunRecord matched_a =
                run_one("matched-fp32-unrotated", 32, diagnostic_dir, artifact);
            const RunRecord matched_b =
                run_one("matched-fp32-rotated", 32, diagnostic_dir, artifact);
            const std::vector<ninfer::TokenId> expected{
                ninfer::TokenId{997},  ninfer::TokenId{1001}, ninfer::TokenId{1003},
                ninfer::TokenId{1005}, ninfer::TokenId{1007}, ninfer::TokenId{1009},
                ninfer::TokenId{1011}, ninfer::TokenId{1013}, ninfer::TokenId{1015}};
            if (matched_a.generated != expected || matched_b.generated != expected) {
                throw std::runtime_error("D1.3 forced continuation was not committed exactly");
            }
            const Metrics hidden = compare(matched_a.hidden, matched_b.hidden);
            const Metrics logits = compare(matched_a.logits, matched_b.logits);
            std::cout << std::setprecision(9) << "d1_3_forced_continuation="
                      << "997,1001,1003,1005,1007,1009,1011,1013,1015"
                      << " hidden_relative_l2=" << hidden.relative_l2
                      << " logits_relative_l2=" << logits.relative_l2 << '\n';
            _putenv_s("NINFER_OSCAR_D1_3_FORCED_DECODE_TOKENS", "");
            _putenv_s("NINFER_OSCAR_ROTATION_MODE", "");
            _putenv_s("NINFER_OSCAR_ROTATION_PRECISION", "");
            _putenv_s("NINFER_OSCAR_MATCHED_FP32", "");
            std::cout << "OSCAR D1.3 forced decode fixture: COMPLETE\n";
            return 0;
        }
        const std::vector<std::size_t> cases = d12 ? std::vector<std::size_t>{32, 512}
                                                   : std::vector<std::size_t>{32};
        const auto print_comparison = [&](std::string_view left_name, const RunRecord& left,
                                          std::string_view right_name, const RunRecord& right,
                                          std::size_t tokens) {
            const Metrics hidden = compare(left.hidden, right.hidden);
            const Metrics logits = compare(left.logits, right.logits);
            const auto left_top10 = top_k(left.logits, 10);
            const auto right_top10 = top_k(right.logits, 10);
            const std::size_t top10_agreement = static_cast<std::size_t>(std::count_if(
                left_top10.begin(), left_top10.end(), [&](std::size_t value) {
                    return std::find(right_top10.begin(), right_top10.end(), value) !=
                           right_top10.end();
                }));
            const bool top1_agree = left_top10.front() == right_top10.front();
            const bool generated_agree = left.generated == right.generated;
            std::cout << std::setprecision(9) << "comparison=" << left_name << "_vs_"
                      << right_name << " case_tokens=" << tokens
                      << " hidden_max_abs=" << hidden.max_abs
                      << " hidden_mean_abs=" << hidden.mean_abs
                      << " hidden_relative_l2=" << hidden.relative_l2
                      << " logits_max_abs=" << logits.max_abs
                      << " logits_mean_abs=" << logits.mean_abs
                      << " logits_relative_l2=" << logits.relative_l2
                      << " top1_agree=" << (top1_agree ? "true" : "false")
                      << " top10_agree_count=" << top10_agreement
                      << " generated_agree=" << (generated_agree ? "true" : "false")
                      << " right_generated_first="
                      << (right.generated.empty() ? -1 : right.generated.front())
                      << " right_generated_second="
                      << (right.generated.size() < 2 ? -1 : right.generated[1]) << '\n';
            if (!std::isfinite(logits.relative_l2) || !std::isfinite(hidden.relative_l2)) {
                throw std::runtime_error("OSCAR runtime variant produced non-finite output");
            }
            return std::pair<Metrics, Metrics>{hidden, logits};
        };
        for (const std::size_t tokens : cases) {
            if (d12) {
                const RunRecord production =
                    run_one("production-bf16", tokens, diagnostic_dir, artifact);
                const RunRecord matched_a =
                    run_one("matched-fp32-unrotated", tokens, diagnostic_dir, artifact);
                const RunRecord matched_b =
                    run_one("matched-fp32-rotated", tokens, diagnostic_dir, artifact);
                (void)print_comparison("production-bf16", production, "matched-fp32-unrotated",
                                       matched_a, tokens);
                (void)print_comparison("matched-fp32-unrotated", matched_a,
                                       "matched-fp32-rotated", matched_b, tokens);
                const auto matched_metrics = print_comparison(
                    "production-bf16", production, "matched-fp32-rotated", matched_b, tokens);
                if (tokens == 32 &&
                    (matched_metrics.first.relative_l2 > 1.0e-4 ||
                     matched_metrics.second.relative_l2 > 1.0e-4 ||
                     matched_a.generated != matched_b.generated)) {
                    throw std::runtime_error(
                        "D1.2 matched FP32 32-token gate failed before 512-token check");
                }
                continue;
            }
            const RunRecord normal = run_one("normal", tokens, diagnostic_dir, artifact);
            constexpr std::array<std::string_view, 4> kVariants{
                "bf16-materialized", "fp32-rotation", "fp32-inverse", "fp32-rotation+inverse"};
            for (const std::string_view variant : kVariants) {
                const RunRecord candidate = run_one(variant, tokens, diagnostic_dir, artifact);
                const Metrics hidden = compare(normal.hidden, candidate.hidden);
                const Metrics logits = compare(normal.logits, candidate.logits);
                const auto normal_top10 = top_k(normal.logits, 10);
                const auto candidate_top10 = top_k(candidate.logits, 10);
                const std::size_t top10_agreement = static_cast<std::size_t>(std::count_if(
                    normal_top10.begin(), normal_top10.end(), [&](std::size_t value) {
                        return std::find(candidate_top10.begin(), candidate_top10.end(), value) !=
                               candidate_top10.end();
                    }));
                const bool top1_agree = normal_top10.front() == candidate_top10.front();
                const bool generated_agree = normal.generated == candidate.generated;
                std::cout << std::setprecision(9) << "variant=" << variant
                          << " case_tokens=" << tokens
                          << " hidden_max_abs=" << hidden.max_abs
                          << " hidden_mean_abs=" << hidden.mean_abs
                          << " hidden_relative_l2=" << hidden.relative_l2
                          << " logits_max_abs=" << logits.max_abs
                          << " logits_mean_abs=" << logits.mean_abs
                          << " logits_relative_l2=" << logits.relative_l2
                          << " top1_agree=" << (top1_agree ? "true" : "false")
                          << " top10_agree_count=" << top10_agreement
                          << " generated_agree=" << (generated_agree ? "true" : "false") << '\n';
                if (!std::isfinite(logits.relative_l2) || !std::isfinite(hidden.relative_l2)) {
                    throw std::runtime_error("OSCAR D1.1 precision variant produced non-finite output");
                }
            }
        }
        _putenv_s("NINFER_OSCAR_ROTATION_MODE", "");
        _putenv_s("NINFER_OSCAR_ROTATION_PRECISION", "");
        _putenv_s("NINFER_OSCAR_MATCHED_FP32", "");
        _putenv_s("NINFER_OSCAR_D1_3_FORCED_DECODE_TOKENS", "");
        _putenv_s("NINFER_OSCAR_RUNTIME_DIAGNOSTIC_PREFIX", "");
        std::cout << "OSCAR D1.1 precision variants: COMPLETE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OSCAR runtime rotated-BF16: FAIL: " << error.what() << '\n';
        return 1;
    }
}
