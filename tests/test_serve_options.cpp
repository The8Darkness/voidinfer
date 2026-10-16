#include "serve/serve_options.h"
#include "serve/translate.h"
#include "voidinfer/engine.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace voidinfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

template <class Function> bool rejects(Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

std::vector<std::string> package(std::vector<std::string> extra = {}) {
    std::vector<std::string> arguments{"voidinfer-serve", "--target", "target", "--draft", "draft"};
    arguments.insert(arguments.end(), extra.begin(), extra.end());
    return arguments;
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions defaults = parse(package());
    failures += check(defaults.model.target_directory == "target" &&
                          defaults.model.draft_directory == "draft" &&
                          defaults.model.vision_directory.empty() && !defaults.vision_enabled(),
                      "--target/--draft did not select the model package");
    failures += check(rejects([] { (void)parse({"voidinfer-serve"}); }),
                      "a server without --target/--draft was accepted");
    failures += check(rejects([] { (void)parse({"voidinfer-serve", "--target", "target"}); }),
                      "a server without --draft was accepted");
    failures += check(rejects([] { (void)parse({"voidinfer-serve", "model-dir"}); }),
                      "a positional model path was accepted");
    failures += check(defaults.max_context == 32768, "--max-context does not default to 32768");
    // Host KV default: max(8 GiB, max-context x 64 KiB, 40% of physical memory).
    std::size_t physical = 0;
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) physical = static_cast<std::size_t>(status.ullTotalPhys);
#endif
    const std::size_t memory_share = physical / 5 * 2;
    failures += check(defaults.context_cache.host_kv_capacity_bytes ==
                          std::max<std::size_t>(voidinfer::kDefaultHostKvCapacityBytes, memory_share),
                      "host KV does not default to max(8 GiB, 40% of memory) at 32K");
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures += check(defaults.preserve_thinking, "thinking history not preserved by default");
    failures += check(defaults.request_log_jsonl.empty(), "request log enabled by default");
    failures += check(defaults.log_stats_interval_ms == 5000, "stats interval default changed");
    failures += check(defaults.media_cache_bytes == voidinfer::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == voidinfer::kDefaultMediaLiveBytes,
                      "media cache defaults changed");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults changed");
    failures += check(!defaults.model_id_override && !defaults.default_thinking_budget,
                      "model id or thinking budget set by default");
    failures += check(resolve_public_model_id(defaults, "loaded-model") == "loaded-model",
                      "public model id does not follow the loaded model");

    const ServeOptions vision = parse(package({"--vision", "bf16"}));
    failures += check(vision.vision_enabled() && vision.model.vision_directory == "bf16",
                      "--vision did not select the BF16 Vision tower");

    const ServeOptions long_context = parse(package({"--max-context", "262144"}));
    failures += check(long_context.context_cache.host_kv_capacity_bytes ==
                          std::max<std::size_t>(262144ULL * 65536, memory_share),
                      "host KV does not scale to one exact snapshot of max-context");
    const ServeOptions explicit_host_kv =
        parse(package({"--max-context", "262144", "--host-kv-mib", "20000"}));
    failures += check(explicit_host_kv.context_cache.host_kv_capacity_bytes == 20000ULL << 20,
                      "--host-kv-mib did not override the default");
    failures += check(rejects([] { (void)parse(package({"--max-context", "1024"})); }) &&
                          rejects([] { (void)parse(package({"--max-context", "300000"})); }),
                      "--max-context outside [2048,262144] was accepted");
    for (const char* removed : {"--kv-dtype", "--spec", "--exl3-round", "--kv-capacity",
                                "--hierarchical-vericache", "--no-cuda-graph"})
        failures += check(rejects([&] { (void)parse(package({removed, "x"})); }),
                          "a removed inherited-route option was accepted");

    const ServeOptions configured =
        parse(package({"--max-concurrency", "4", "--max-context", "4096", "--max-pending-requests",
                       "12", "--pending-timeout-ms", "2500", "--log-stats-interval-ms", "0",
                       "--media-cache-mib", "256", "--no-preserve-thinking", "--no-prefix-reuse"}));
    failures += check(configured.max_concurrency == 4 && configured.max_context == 4096 &&
                          configured.max_pending_requests == 12 &&
                          configured.pending_timeout_ms == 2500 &&
                          configured.log_stats_interval_ms == 0 &&
                          configured.media_cache_bytes == (256ULL << 20) &&
                          !configured.preserve_thinking,
                      "configured serving options were not preserved");
    failures += check(!configured.allow_prefix_reuse && configured.context_cache.enabled,
                      "--no-prefix-reuse must keep the host-root cache and disable reuse");

    const ServeOptions model_alias = parse(package({"--model-id", "deployment-alias"}));
    failures += check(resolve_public_model_id(model_alias, "loaded-model") == "deployment-alias",
                      "--model-id did not override the public model id");
    failures += check(rejects([] { (void)parse(package({"--model-id", ""})); }),
                      "empty --model-id was accepted");
    const ServeOptions thinking_budget = parse(package({"--default-thinking-budget", "37"}));
    failures += check(thinking_budget.default_thinking_budget == 37,
                      "--default-thinking-budget did not reach serving options");
    failures += check(rejects([] { (void)parse(package({"--default-thinking-budget", "0"})); }),
                      "zero --default-thinking-budget was accepted");
    const ServeOptions response_store =
        parse(package({"--response-store-max-records", "42", "--response-store-max-mib", "7"}));
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (7ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling = parse(package(
        {"--temperature", "0", "--top-p", "0.9", "--top-k", "40", "--min-p", "0.1", "--seed", "0"}));
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 40 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");

    GenerationRequest request;
    request.max_tokens = 1;
    voidinfer::PromptCapabilities prompt_capabilities;
    prompt_capabilities.enable_thinking = true;
    const auto semantics = resolve_prompt_semantics(request, defaults, prompt_capabilities);
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const voidinfer::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.allow_prefix_reuse)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking =
        resolve_prompt_semantics(request, thinking_budget, prompt_capabilities);
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.allow_prefix_reuse)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    failures +=
        check(!resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "server preserve-thinking default was not resolved");
    request.preserve_thinking = true;
    failures +=
        check(resolve_prompt_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "request preserve-thinking override did not win");

    const std::string usage = serve_usage_text("voidinfer-serve");
    for (const char* flag : {"--target", "--draft", "--vision", "--host-kv-mib", "--no-prefix-reuse",
                             "--no-preserve-thinking", "--default-thinking-budget",
                             "--log-stats-interval-ms", "--media-preprocess-threads",
                             "--response-store-max-mib", "--request-log-jsonl"})
        failures += check(usage.find(flag) != std::string::npos, "serve help omits a flag");
    failures += check(usage.find("--kv-dtype") == std::string::npos &&
                          usage.find("--exl3-") == std::string::npos,
                      "serve help still documents removed options");

    const ServeOptions logged =
        parse(package({"--request-log-jsonl", "requests.jsonl", "--api-key", "do-not-log"}));
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
