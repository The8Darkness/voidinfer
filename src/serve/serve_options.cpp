#include "serve/serve_options.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace voidinfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " --target DIR --draft DIR [--vision DIR] [--dual-manifest FILE]\n"
           "       [--host H] [--port N] [--api-key KEY] [--model-id ID] [--device N]\n"
           "       [--max-context N] [--max-concurrency N] [--max-pending-requests N]\n"
           "       [--pending-timeout-ms N] [--prefill-chunk N] [--log-stats-interval-ms N]\n"
           "       [--max-request-mib N] [--media-cache-mib N] [--media-live-mib N]\n"
           "       [--media-preprocess-threads N] [--host-state-slots N] [--host-kv-mib N]\n"
           "       [--device-state-slots N] [--max-private-continuations N] [--max-shared-prefixes N]\n"
           "       [--max-long-anchors-per-continuation N] [--max-cache-markers-per-request N]\n"
           "       [--request-log-jsonl FILE] [--response-store-max-records N]\n"
           "       [--response-store-max-mib N] [--default-max-tokens N] [--default-thinking-budget N]\n"
           "       [--no-prefix-reuse] [--no-thinking] [--no-preserve-thinking] [--cors]\n"
           "       [--webui | --webui-dir DIR] [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--seed N] [--greedy]\n"
           "\n"
           "Serves the OpenAI Responses/Chat Completions and Anthropic Messages endpoints.\n"
           "--target is the EXL3 model directory, --draft its DFlash2 draft; --vision names a\n"
           "directory with the BF16 Vision tower (`model.visual.*`, e.g. the BF16 checkpoint)\n"
           "and enables image and video input.\n"
           "--max-context defaults to 262144 (2048..262144); one to eight concurrent requests.\n"
           "--host-kv-mib defaults to max(8192, max-context x 64 KiB) of exact host history.\n"
           "--default-max-tokens defaults to " + std::to_string(kDefaultMaxTokens) + " when a request omits max_tokens.\n"
           "--max-request-mib defaults to 384 and is enforced before JSON parsing.\n"
           "--media-cache-mib defaults to 1024 (0 disables retained media reuse); --media-live-mib\n"
           "defaults to 2048; --media-preprocess-threads 0 selects up to 16 workers.\n"
           "--request-log-jsonl appends full-precision server/request records.\n"
           "Responses state is process-local: 1024 records / 256 MiB by default.\n"
           "--log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs.\n"
           "--no-prefix-reuse serves every request without reusing a cached conversation prefix.\n"
           "--default-thinking-budget caps model-origin thinking; control tokens count toward\n"
           "the output limit. Closed-turn reasoning stays in later prompts, as the model template\n"
           "defaults; --no-preserve-thinking drops it.\n"
           "Sampler defaults come from the model and thinking mode; flags and request fields\n"
           "override individual values (temperature, top_p, top_k, min_p; no penalties).\n"
           "--webui downloads the prebuilt llama.cpp web UI into --webui-dir (default\n"
           "<target>/webui) and serves it at /. --greedy forces temperature 0.\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool host_kv_explicit            = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--target") {
            options.model.target_directory = require_value("--target");
        } else if (arg == "--draft") {
            options.model.draft_directory = require_value("--draft");
        } else if (arg == "--vision") {
            options.model.vision_directory = require_value("--vision");
        } else if (arg == "--dual-manifest") {
            options.model.verified_dual_manifest = require_value("--dual-manifest");
        } else if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
        } else if (arg == "--host-state-slots") {
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
        } else if (arg == "--host-kv-mib") {
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            host_kv_explicit                             = true;
        } else if (arg == "--max-private-continuations") {
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
        } else if (arg == "--max-shared-prefixes") {
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
        } else if (arg == "--max-long-anchors-per-continuation") {
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
        } else if (arg == "--max-cache-markers-per-request") {
            options.context_cache.max_cache_markers_per_request = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-cache-markers-per-request"),
                                      "max-cache-markers-per-request"));
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--no-preserve-thinking") {
            options.preserve_thinking = false;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--webui") {
            options.webui_auto = true;
        } else if (arg == "--webui-dir") {
            options.webui_dir = require_value("--webui-dir");
            if (options.webui_dir.empty()) {
                throw std::invalid_argument("--webui-dir must not be empty");
            }
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            options.sampling_overrides.top_k =
                parse_nonnegative_int(require_value("--top-k"), "top-k");
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.model.target_directory.empty() || options.model.draft_directory.empty()) {
        throw std::invalid_argument("--target and --draft are required");
    }
    if (!host_kv_explicit) {
        // Retained conversation roots (exact FP16 history, 64 KiB per token)
        // may use 40% of physical memory, so agents beyond the lanes resume
        // without a new prefill; the KV spill tier covers memory pressure.
        std::size_t physical = 0;
#ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        if (GlobalMemoryStatusEx(&status)) physical = static_cast<std::size_t>(status.ullTotalPhys);
#endif
        options.context_cache.host_kv_capacity_bytes =
            std::max<std::size_t>({kDefaultHostKvCapacityBytes,
                                   static_cast<std::size_t>(options.max_context) * 65536,
                                   physical / 5 * 2});
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context < 2048 || options.max_context > 262144) {
        throw std::invalid_argument("--max-context must be in [2048,262144]");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view loaded_model_id) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (loaded_model_id.empty()) {
        throw std::logic_error("loaded model_id must not be empty");
    }
    return std::string(loaded_model_id);
}

} // namespace voidinfer::serve
