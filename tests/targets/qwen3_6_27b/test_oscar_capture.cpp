#include "ninfer/engine.h"
#include "targets/qwen3_6/impl/runtime/oscar_qkv_capture.h"

#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kDefaultArtifactPath =
    R"(C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer)";

const char* required_environment(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

std::size_t positive_environment(const char* name, std::size_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument(std::string("invalid positive environment: ") + name);
    }
    return static_cast<std::size_t>(parsed);
}

std::int64_t nonnegative_environment(const char* name, std::int64_t fallback,
                                     bool* was_present) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        if (was_present != nullptr) {
            *was_present = false;
        }
        return fallback;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0) {
        throw std::invalid_argument(std::string("invalid nonnegative environment: ") + name);
    }
    if (was_present != nullptr) {
        *was_present = true;
    }
    return static_cast<std::int64_t>(parsed);
}

} // namespace

int main() {
    const char* capture_dir = required_environment("NINFER_OSCAR_QKV_CAPTURE_DIR");
    const char* artifact_env =
        required_environment("NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS");
    std::string default_artifact_path;
    if (artifact_env == nullptr) {
        default_artifact_path = kDefaultArtifactPath;
        if (!std::filesystem::is_regular_file(default_artifact_path)) {
            std::cout << "skip: default Qwen3.8 DFlash2 artifact not found at C: "
                      << default_artifact_path
                      << " (set NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS to override)\n";
            return 77;
        }
        artifact_env = default_artifact_path.c_str();
    }
    if (capture_dir == nullptr) {
        std::cout << "skip: OSCAR capture directory is required\n";
        return 77;
    }

    try {
        ninfer::EngineOptions options;
        options.artifact_path                    = artifact_env;
        options.max_context                      = 512;
        options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(512);
        options.max_concurrency                  = 1;
        options.max_pending_requests             = 1;
        options.prefill_chunk                    = 256;
        options.speculative.backend              = ninfer::SpeculativeBackend::None;
        options.enable_vision                    = false;
        options.use_cuda_graph                   = false;
        options.context_cache.enabled            = false;

        ninfer::Engine engine(std::move(options));
        const std::size_t request_count =
            positive_environment("NINFER_OSCAR_QKV_CAPTURE_REQUESTS", 1);
        const std::size_t prompt_tokens =
            positive_environment("NINFER_OSCAR_QKV_PROMPT_TOKENS", 256);
        if (prompt_tokens > 512) {
            throw std::invalid_argument("OSCAR capture prompt token count exceeds max_context=512");
        }
        bool varied_formula = false;
        const std::int64_t formula_seed =
            nonnegative_environment("NINFER_OSCAR_QKV_TOKEN_FORMULA_SEED", 0, &varied_formula);
        if (_putenv_s("NINFER_OSCAR_QKV_CAPTURE_ARMED", "1") != 0) {
            throw std::runtime_error("could not arm OSCAR QKV capture");
        }

        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 1;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = false;
        request.stop.include_model_defaults       = false;

        std::size_t generated_tokens = 0;
        for (std::size_t request_index = 0; request_index < request_count; ++request_index) {
            std::vector<ninfer::TokenId> prompt(prompt_tokens);
            for (std::size_t position = 0; position < prompt_tokens; ++position) {
                if (!varied_formula) {
                    prompt[position] = ninfer::TokenId{198};
                    continue;
                }
                const std::int64_t mixed =
                    formula_seed + static_cast<std::int64_t>(request_index) * 7919 +
                    static_cast<std::int64_t>(position) * 131;
                prompt[position] =
                    ninfer::TokenId{198 + static_cast<std::int32_t>(mixed % 4096)};
            }
            try {
                const ninfer::GenerationResult result =
                    engine.generate(engine.prepare_tokens(std::move(prompt), false), request);
                if (result.generated_token_ids.size() != 1 ||
                    result.finish_reason != ninfer::FinishReason::OutputLimit) {
                    std::cerr << "OSCAR capture request " << request_index
                              << " did not complete with one output token\n";
                    return 1;
                }
                generated_tokens += result.generated_token_ids.size();
            } catch (const std::exception& error) {
                // Some byte-level BPE tokens are incomplete UTF-8 when the one-token
                // smoke response is finalized. Prefill and the OSCAR capture have
                // already completed before this presentation-only failure. Continue
                // only for this exact known condition; all other failures are fatal.
                const std::string message = error.what();
                if (message.find("invalid UTF-8 leading byte in generated token stream") ==
                    std::string::npos) {
                    throw;
                }
                std::cout << "OSCAR capture request " << request_index
                          << ": output UTF-8 presentation skipped after prefill\n";
            }
        }
        ninfer::targets::qwen3_6::oscar_internal::finalize_qkv_capture_from_environment();

        std::cout << "OSCAR capture smoke: PASS requests=" << request_count
                  << " prompt_tokens=" << prompt_tokens
                  << " useful_tokens=" << request_count * prompt_tokens
                  << " generated_tokens=" << generated_tokens << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OSCAR capture smoke: FAIL: " << error.what() << '\n';
        return 1;
    }
}
