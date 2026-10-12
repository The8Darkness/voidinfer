#include "serve/request_log.h"
#include "serve/console_log.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <cwchar>
#endif

#include <utility>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

std::uint64_t unix_time_ms() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string new_server_instance_id() {
    const auto now    = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
#ifdef _WIN32
    const auto process_id = ::_getpid();
#else
    const auto process_id = ::getpid();
#endif
    return "serve-" + std::to_string(static_cast<long long>(process_id)) + '-' +
           std::to_string(micros);
}



std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { out << '-'; }
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

const char* finish_reason_name(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output_limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context_capacity";
    case ninfer::FinishReason::StopToken:
        return "stop_token";
    case ninfer::FinishReason::StopString:
        return "stop_string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

std::string tool_choice_name(const ToolChoice& choice) {
    switch (choice.mode) {
    case ToolChoiceMode::Auto:
        return "auto";
    case ToolChoiceMode::None:
        return "none";
    case ToolChoiceMode::Required:
        return "required";
    case ToolChoiceMode::Named:
        return choice.name.empty() ? "named" : choice.name;
    }
    return "unknown";
}




const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
    case ninfer::PrefixReusePath::Root:
        return "root";
    case ninfer::PrefixReusePath::PrivateEndpoint:
        return "private_endpoint";
    case ninfer::PrefixReusePath::PrivateTurnClosure:
        return "private_turn_closure";
    case ninfer::PrefixReusePath::PrivateResponseReplay:
        return "private_response_replay";
    case ninfer::PrefixReusePath::PrivateLongAnchor:
        return "private_long_anchor";
    case ninfer::PrefixReusePath::SharedStablePrefix:
        return "shared_stable_prefix";
    }
    return "unknown";
}

Json event_base(const std::string& server_instance_id, std::uint64_t timestamp, const char* event) {
    return Json{{"artifact_type", kRequestLogArtifactType},
                {"schema_version", kRequestLogSchemaVersion},
                {"event", event},
                {"timestamp_unix_ms", timestamp},
                {"server_instance_id", server_instance_id}};
}

Json sampler_json(const ninfer::ResolvedSamplingParameters& sampling) {
    return Json{{"temperature", sampling.temperature},
                {"top_p", sampling.top_p},
                {"top_k", sampling.top_k},
                {"min_p", sampling.min_p},
                {"presence_penalty", sampling.presence_penalty},
                {"frequency_penalty", sampling.frequency_penalty},
                {"seed", sampling.seed}};
}

Json preset_json(const ninfer::SamplingPreset& preset) {
    return Json{{"temperature", preset.temperature},
                {"top_p", preset.top_p},
                {"top_k", preset.top_k},
                {"min_p", preset.min_p},
                {"presence_penalty", preset.presence_penalty},
                {"frequency_penalty", preset.frequency_penalty}};
}

Json overrides_json(const ninfer::SamplingOverrides& overrides) {
    Json result{{"temperature", nullptr},
                {"top_p", nullptr},
                {"top_k", nullptr},
                {"min_p", nullptr},
                {"presence_penalty", nullptr},
                {"frequency_penalty", nullptr},
                {"seed", nullptr}};
    if (overrides.temperature) { result["temperature"] = *overrides.temperature; }
    if (overrides.top_p) { result["top_p"] = *overrides.top_p; }
    if (overrides.top_k) { result["top_k"] = *overrides.top_k; }
    if (overrides.min_p) { result["min_p"] = *overrides.min_p; }
    if (overrides.presence_penalty) { result["presence_penalty"] = *overrides.presence_penalty; }
    if (overrides.frequency_penalty) { result["frequency_penalty"] = *overrides.frequency_penalty; }
    if (overrides.seed) { result["seed"] = *overrides.seed; }
    return result;
}


Json request_json(const RequestLogContext& context) {
    Json thinking_budget = nullptr;
    if (context.thinking_budget) { thinking_budget = *context.thinking_budget; }
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"enable_thinking", context.enable_thinking},
                {"thinking_budget", std::move(thinking_budget)},
                {"preserve_thinking", context.preserve_thinking},
                {"preserve_thinking_semantic_change", context.preserve_thinking_semantic_change},
                {"sampling", sampler_json(context.sampling)}};
}

Json preparation_json(const RequestLogContext& context) {
    const PromptPreparationStats& stats = context.preparation;
    return Json{{"total", stats.seconds},
                {"acquisition", context.acquisition_seconds},
                {"media_preprocess", stats.media_preprocess_seconds},
                {"media_preprocess_work", stats.media_preprocess_work_seconds},
                {"tokenize", stats.tokenize_seconds},
                {"media_items", stats.media_items},
                {"media_bytes", stats.media_bytes},
                {"raw_patches", stats.raw_patches},
                {"vision_tokens", stats.vision_tokens},
                {"patch_bytes", stats.patch_bytes},
                {"cache_hits", stats.media_cache_hits},
                {"cache_misses", stats.media_cache_misses},
                {"singleflight_waits", stats.media_singleflight_waits},
                {"built_patch_bytes", stats.built_patch_bytes},
                {"reused_patch_bytes", stats.reused_patch_bytes},
                {"retained_media_payload_bytes", stats.retained_media_payload_bytes}};
}

Json rejected_request_json(const RequestRejectionLogContext& context) {
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history}};
}

Json error_json(const ApiError& error) {
    Json code  = error.code.empty() ? Json(nullptr) : Json(error.code);
    Json param = error.param.empty() ? Json(nullptr) : Json(error.param);
    return Json{{"status", error.status},
                {"type", error.type},
                {"code", std::move(code)},
                {"param", std::move(param)},
                {"message", error.message}};
}

Json arena_json(const ninfer::ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}


Json speculative_json(const GenerationMetrics& metrics) {
    Json shared_projection_rows=nullptr;
    if(metrics.speculative_shared_projection_rows)
        shared_projection_rows=*metrics.speculative_shared_projection_rows;
    return Json{{"backend", metrics.speculative_enabled ? "dflash2" : "off"},
                {"draft_window", metrics.speculative_draft_window},
                {"rounds", metrics.speculative_rounds},
                {"drafted_tokens", metrics.speculative_draft_tokens},
                {"accepted_tokens", metrics.speculative_accepted_tokens},
                {"fallback_steps", metrics.speculative_fallback_steps},
                {"accepted_per_position", metrics.speculative_accepted_per_position},
                {"neural_input_rows",metrics.speculative_neural_input_rows},
                {"returned_suffix_rows",metrics.speculative_returned_suffix_rows},
                {"discarded_suffix_rows",metrics.speculative_discarded_suffix_rows},
                {"proposed_rows",metrics.speculative_proposed_rows},
                {"verified_rows",metrics.speculative_verified_rows},
                {"replayed_rows",metrics.speculative_replayed_rows},
                {"committed_model_rows",metrics.speculative_committed_model_rows},
                {"externally_visible_model_rows",
                    metrics.speculative_externally_visible_model_rows},
                {"hidden_terminal_rows",metrics.speculative_hidden_terminal_rows},
                {"diagnostic_rows",metrics.speculative_diagnostic_rows},
                {"shared_projection_rows",std::move(shared_projection_rows)}};
}

Json materialization_json(const ninfer::MaterializationDiagnostics& diagnostics) {
    return Json{
        {"predicted_now_ns", diagnostics.predicted_now_ns},
        {"predicted_future_loss_ns", diagnostics.predicted_future_loss_ns},
        {"predicted_total_ns", diagnostics.predicted_total_ns},
        {"targets_evaluated", diagnostics.targets_evaluated},
        {"projection_work", diagnostics.projection_work},
        {"planning_elapsed_ns", diagnostics.planning_elapsed_ns},
        {"search_elapsed_ns", diagnostics.search_elapsed_ns},
        {"stop_reason", ninfer::materialization_stop_reason_name(diagnostics.stop_reason)},
        {"model_optimal", diagnostics.model_optimal},
        {"budget_exhausted", diagnostics.budget_exhausted},
        {"best_remaining_lower_bound_ns", diagnostics.best_remaining_lower_bound_ns},
        {"absolute_bound_gap_ns", diagnostics.absolute_bound_gap_ns},
        {"relative_bound_gap", diagnostics.relative_bound_gap},
        {"selected_degradation_units", diagnostics.selected_degradation_units},
        {"selected_maximal_fallback", diagnostics.selected_maximal_fallback},
    };
}

double nanoseconds_to_seconds(std::uint64_t value) noexcept {
    return static_cast<double>(value) * 1.0e-9;
}

double nanoseconds_to_microseconds(std::uint64_t value) noexcept {
    return static_cast<double>(value) * 1.0e-3;
}

double request_host_exposed_seconds(const ninfer::GenerationEngineTiming& timing) noexcept {
    return timing.engine_boundary_exposed_seconds + timing.program_submit_exposed_seconds +
           timing.program_post_exposed_seconds + timing.engine_commit_output_exposed_seconds +
           timing.engine_maintenance_exposed_seconds;
}

Json request_engine_timing_json(const ninfer::GenerationEngineTiming& timing) {
    return Json{
        {"queue_wait_seconds", timing.queue_wait_seconds},
        {"host_exposed_seconds",
         Json{{"engine_boundary", timing.engine_boundary_exposed_seconds},
              {"program_submit", timing.program_submit_exposed_seconds},
              {"program_post", timing.program_post_exposed_seconds},
              {"engine_commit_output", timing.engine_commit_output_exposed_seconds},
              {"engine_maintenance", timing.engine_maintenance_exposed_seconds},
              {"total", request_host_exposed_seconds(timing)}}},
        {"device_wait_exposed_seconds", timing.device_wait_exposed_seconds},
        {"decode", Json{{"host_exposed_seconds", timing.decode_host_exposed_seconds},
                        {"device_wait_exposed_seconds", timing.decode_device_wait_exposed_seconds},
                        {"rounds", timing.decode_rounds}}},
        {"units", Json{{"prefill", timing.prefill_units}, {"control", timing.control_units}}},
    };
}



Json microseconds_per(std::uint64_t nanoseconds, std::uint64_t count) {
    if (count == 0) { return nullptr; }
    return nanoseconds_to_microseconds(nanoseconds) / static_cast<double>(count);
}

// Tokens/second with fixed precision, or "n/a" when the interval is degenerate.
std::string rate(double tokens, double seconds) {
    std::ostringstream out;
    if (seconds > 0.0 && tokens > 0.0) {
        out << std::fixed << std::setprecision(1) << (tokens / seconds) << "tok/s";
    } else {
        out << "n/a";
    }
    return out.str();
}

std::string seconds_str(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << 's';
    return out.str();
}

// Compact resolved-sampler summary. temperature <= 0 is the exact-argmax path.
std::string sampler_str(const ninfer::ResolvedSamplingParameters& sampling) {
    if (sampling.temperature <= 0.0f) { return "greedy"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
        << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k;
    if (sampling.min_p > 0.0f) { out << " min_p=" << sampling.min_p; }
    if (sampling.presence_penalty != 0.0f) { out << " pres=" << sampling.presence_penalty; }
    if (sampling.frequency_penalty != 0.0f) { out << " freq=" << sampling.frequency_penalty; }
    out << " seed=" << sampling.seed;
    return out.str();
}

std::string speculative_str(const GenerationMetrics& metrics) {
    if (!metrics.speculative_enabled) { return "off"; }
    std::ostringstream out;
    out << "dflash2 " << std::fixed
        << std::setprecision(2);
    if (metrics.speculative_rounds > 0) {
        const double per_round = 1.0 + static_cast<double>(metrics.speculative_accepted_tokens) /
                                           static_cast<double>(metrics.speculative_rounds);
        out << per_round << "tok/round";
    } else {
        out << "n/a";
    }
    if (metrics.speculative_draft_tokens > 0) {
        const double accept_pct = 100.0 * static_cast<double>(metrics.speculative_accepted_tokens) /
                                  static_cast<double>(metrics.speculative_draft_tokens);
        out << " (" << std::setprecision(1) << accept_pct << "%)";
    }
    return out.str();
}

} // namespace


RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const PreparedRequest& prepared) {
    RequestLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = request.model;
    context.stream                             = request.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = request.max_tokens_set;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.enable_thinking                    = prepared.enable_thinking;
    context.thinking_budget                    = prepared.thinking_budget;
    context.preserve_thinking                  = prepared.preserve_thinking;
    context.preserve_thinking_semantic_change  = prepared.preserve_thinking_semantic_change;
    context.sampling                           = prepared.sampling;
    context.acquisition_seconds                = prepared.acquisition_seconds;
    context.preparation                        = prepared.preparation;
    return context;
}

RequestRejectionLogContext make_request_rejection_log_context(std::uint64_t id,
                                                              std::string protocol,
                                                              const GenerationRequest& request,
                                                              ApiError error) {
    RequestRejectionLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = request.model;
    context.stream                             = request.stream;
    context.message_count                      = request.messages.size();
    context.media_item_count                   = request.media_item_count();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = request.max_tokens_set;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.error                              = std::move(error);
    return context;
}

std::string format_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "[req " << context.id << "] " << context.protocol << ' '
        << (context.stream ? "stream" : "non-stream") << " msgs=" << context.message_count
        << " max_tokens=" << context.requested_output_tokens << ' '
        << (context.requested_output_tokens_client_set ? "(client)" : "(server default)")
        << " tools=" << context.tool_count
        << " tool_choice=" << tool_choice_name(context.tool_choice)
        << " tool_history=" << (context.has_tool_history ? "yes" : "no")
        << " thinking=" << (context.enable_thinking ? "on" : "off")
        << " preserve_thinking=" << (context.preserve_thinking ? "on" : "off")
        << " preserve_change=" << (context.preserve_thinking_semantic_change ? "yes" : "no")
        << " sampler=[" << sampler_str(context.sampling) << ']';
    if (context.thinking_budget) { out << " thinking_budget=" << *context.thinking_budget; }
    if (context.media_item_count != 0) {
        out << " prepare=" << seconds_str(context.preparation.seconds)
            << " acquire=" << seconds_str(context.acquisition_seconds)
            << " media=" << seconds_str(context.preparation.media_preprocess_seconds) << '/'
            << seconds_str(context.preparation.media_preprocess_work_seconds)
            << " tokenize=" << seconds_str(context.preparation.tokenize_seconds)
            << " media_cache=" << context.preparation.media_cache_hits << '/'
            << context.preparation.media_cache_misses << '/'
            << context.preparation.media_singleflight_waits;
    }
    out << " \xE2\x86\x92 submitted";
    return out.str();
}

std::string format_request_rejected(const RequestRejectionLogContext& context) {
    std::ostringstream out;
    out << "[req " << context.id << "] rejected phase=prepare protocol=" << context.protocol << ' '
        << (context.stream ? "stream" : "non-stream") << " msgs=" << context.message_count
        << " media=" << context.media_item_count << " tools=" << context.tool_count
        << " status=" << context.error.status;
    if (!context.error.code.empty()) { out << " code=" << context.error.code; }
    out << " message=" << context.error.message;
    return out.str();
}

std::string format_request_done(const RequestLogContext& context,
                                const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const double ttft_ms             = metrics.ttft_seconds * 1000.0;
    // Prefill emits the first token; the remaining (gen - 1) come from decode.
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;
    const double computed_prefill_tokens = static_cast<double>(
        std::max(0, outcome.prompt_tokens - static_cast<int>(metrics.prefix_cache_hit_tokens)));

    std::ostringstream out;
    out << "[req " << context.id << "] done finish="
        << (outcome.tool_calls.empty() ? finish_reason_name(outcome.finish_reason) : "tool_calls");
    if (!outcome.tool_calls.empty()) { out << " tool_calls=" << outcome.tool_calls.size(); }
    out << " prompt=" << outcome.prompt_tokens << " gen=" << outcome.completion_tokens
        << " cache=" << metrics.prefix_cache_hit_tokens
        << " reuse=" << prefix_reuse_path_name(metrics.prefix_reuse_path) << " ttft=" << std::fixed
        << std::setprecision(0) << ttft_ms << "ms"
        << " prefill=" << rate(computed_prefill_tokens, metrics.prefill_seconds)
        << " decode=" << rate(decode_tokens, metrics.decode_seconds)
        << " wall=" << seconds_str(metrics.total_seconds) << " host=" << std::setprecision(2)
        << request_host_exposed_seconds(metrics.engine_timing) * 1000.0 << "ms";
    if (metrics.engine_timing.decode_rounds == 0) {
        out << " decode-host=n/a wait=n/a";
    } else {
        const double rounds = static_cast<double>(metrics.engine_timing.decode_rounds);
        out << " decode-host=" << std::setprecision(1)
            << metrics.engine_timing.decode_host_exposed_seconds * 1.0e6 / rounds
            << "us/round wait="
            << metrics.engine_timing.decode_device_wait_exposed_seconds * 1.0e6 / rounds
            << "us/round";
    }
    out << " speculative=" << speculative_str(metrics);
    if (outcome.thinking.configured_budget) {
        out << " thinking_budget=" << *outcome.thinking.configured_budget
            << " model_thinking=" << outcome.thinking.model_thinking_tokens
            << " control_tokens=" << outcome.thinking.injected_tokens
            << " control=" << (outcome.thinking.applied ? "applied" : "unused");
    }
    return out.str();
}

std::string format_request_error(const RequestLogContext& context, const std::string& message) {
    std::ostringstream out;
    out << "[req " << context.id << "] error " << message;
    return out.str();
}

std::string format_throughput(const ThroughputReport& report) {
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    std::ostringstream out;
    out << "throughput interval=" << std::fixed << std::setprecision(3) << report.interval_seconds
        << "s prefill=" << std::setprecision(1) << prefill_rate << "tok/s decode=" << decode_rate
        << "tok/s rounds=" << report.decode_rounds
        << " running=" << report.current.running_requests
        << " waiting=" << report.current.waiting_requests << " reused_prompt_tokens="
        << monotonic_delta(report.previous.reused_prompt_tokens,
                           report.current.reused_prompt_tokens);
    return out.str();
}

std::string format_server_start_json(
    const std::string& server_instance_id, std::uint64_t timestamp, const ServeOptions& options,
    const ninfer::EngineOptions& engine_options,
    const ninfer::ModelSamplingDefaults& sampling_defaults, const std::string& public_model_id,
    const ninfer::LoadSummary& load, const ninfer::MemorySummary& memory,
    const ServerLogEnvironment& environment) {
    Json record = event_base(server_instance_id, timestamp, "server_start");

    Json default_thinking_budget = nullptr;
    if (options.default_thinking_budget) {
        default_thinking_budget = *options.default_thinking_budget;
    }

    record["server"]                               = Json{{"host", options.host},
                                                          {"port", options.port},
                                                          {"public_model_id", public_model_id},
                                                          {"api_key_configured", !options.api_key.empty()},
                                                          {"cors_enabled", options.enable_cors},
                                                          {"max_request_bytes", options.max_request_bytes},
                                                          {"media_cache_bytes", options.media_cache_bytes},
                                                          {"media_live_bytes", options.media_live_bytes},
                                                          {"media_preprocess_threads", options.media_preprocess_threads},
                                                          {"request_log_jsonl", options.request_log_jsonl},
                                                          {"default_output_tokens", options.default_max_tokens},
                                                          {"default_thinking", options.enable_thinking},
                                                          {"default_thinking_budget", std::move(default_thinking_budget)},
                                                          {"default_preserve_thinking", options.preserve_thinking}};
    record["model"]                                = Json{{"target", options.model.target_directory.string()},
                                                          {"draft", options.model.draft_directory.string()},
                                                          {"vision", options.model.vision_directory.string()},
                                                          {"identity", load.target},
                                                          {"weights_id", load.weights_id},
                                                          {"bytes_read", load.model_bytes_read},
                                                          {"host_to_device_bytes", load.host_to_device_bytes},
                                                          {"peak_staging_bytes", load.peak_staging_bytes},
                                                          {"load_seconds", load.load_seconds},
                                                          {"upload_seconds", load.upload_seconds}};
    const ninfer::ContextCacheOptions& cache       = engine_options.context_cache;
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(engine_options.max_concurrency) +
        cache.device_state_slots.value();
    record["engine"] = Json{
        {"device", engine_options.device},
        {"max_context", engine_options.max_context},
        {"max_concurrency", engine_options.max_concurrency},
        {"max_pending_requests", engine_options.max_pending_requests},
        {"pending_timeout_ms", engine_options.pending_timeout_ms},
        {"prefill_chunk", engine_options.prefill_chunk},
        {"log_stats_interval_ms", options.log_stats_interval_ms},
        {"vision", engine_options.vision_enabled()},
        {"prefix_reuse", options.allow_prefix_reuse},
        {"speculative", "dflash2-k7"},
        {"context_cache",
         Json{
             {"enabled", cache.enabled},
             {"device_state_slots", cache.device_state_slots.value()},
             {"total_device_state_slots", total_device_state_slots},
             {"host_state_slots", cache.host_state_slots},
             {"host_kv_capacity_bytes", cache.host_kv_capacity_bytes},
             {"max_private_continuations", cache.max_private_continuations.value()},
             {"max_shared_prefixes", cache.max_shared_prefixes.value()},
             {"max_long_anchors_per_continuation", cache.max_long_anchors_per_continuation.value()},
             {"max_cache_markers_per_request", cache.max_cache_markers_per_request.value()}}}};
    record["sampling_defaults"] =
        Json{{"thinking", preset_json(sampling_defaults.thinking)},
             {"non_thinking", preset_json(sampling_defaults.non_thinking)},
             {"server_overrides", overrides_json(options.sampling_overrides)},
             {"omitted_seed", "random"},
             {"greedy", options.greedy}};
    record["memory"] =
        Json{{"weights", arena_json(memory.weights)},
             {"sequence", arena_json(memory.sequence)},
             {"workspace", arena_json(memory.workspace)},
             {"minimum_runtime_reservation_bytes", memory.minimum_runtime_reservation_bytes},
             {"runtime_reservation_bytes", memory.runtime_reservation_bytes},
             {"available_after_weights_bytes", memory.available_after_weights_bytes},
             {"available_after_startup_bytes", memory.available_after_startup_bytes},
             {"planned_slack_bytes", memory.planned_slack_bytes},
             {"cuda_graph_allowance_bytes", memory.cuda_graph_allowance_bytes},
             {"kv_payload_bytes", memory.kv_payload_bytes},
             {"host_state_capacity_slots", memory.host_state_capacity_slots},
             {"host_state_occupied_slots", memory.host_state_occupied_slots},
             {"host_kv_capacity_bytes", memory.host_kv_capacity_bytes},
             {"host_kv_occupied_bytes", memory.host_kv_occupied_bytes}};
    record["environment"] =
        Json{{"device", environment.device},
             {"gpu_name", environment.gpu_name},
             {"gpu_uuid", environment.gpu_uuid},
             {"total_device_memory_bytes", environment.total_device_memory_bytes},
             {"compute_capability_major", environment.compute_capability_major},
             {"compute_capability_minor", environment.compute_capability_minor},
             {"cuda_compile_version", environment.cuda_compile_version},
             {"cuda_runtime_version", environment.cuda_runtime_version},
             {"cuda_driver_version", environment.cuda_driver_version}};
    record["argv"] = options.startup_argv;
    return record.dump();
}

std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context) {
    Json record                   = event_base(server_instance_id, timestamp, "request_start");
    record["request"]             = request_json(context);
    record["preparation_seconds"] = preparation_json(context);
    return record.dump();
}

std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp,
                                         const RequestRejectionLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "request_rejected");
    record["phase"]   = "prepare";
    record["request"] = rejected_request_json(context);
    record["error"]   = error_json(context.error);
    return record.dump();
}

std::string format_request_done_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    Json record       = event_base(server_instance_id, timestamp, "request_done");
    record["request"] = request_json(context);
    record["result"] =
        Json{{"finish_reason", finish_reason_name(outcome.finish_reason)},
             {"prompt_tokens", outcome.prompt_tokens},
             {"completion_tokens", outcome.completion_tokens},
             {"computed_prefill_tokens",
              std::max(0, outcome.prompt_tokens -
                              static_cast<int>(outcome.metrics.prefix_cache_hit_tokens))},
             {"prefix_cache_hit_tokens", outcome.metrics.prefix_cache_hit_tokens},
             {"prefix_reuse_path", prefix_reuse_path_name(outcome.metrics.prefix_reuse_path)},
             {"thinking_budget", outcome.thinking.configured_budget
                                     ? Json(*outcome.thinking.configured_budget)
                                     : Json(nullptr)},
             {"model_thinking_tokens", outcome.thinking.model_thinking_tokens},
             {"thinking_control_tokens", outcome.thinking.injected_tokens},
             {"thinking_control_applied", outcome.thinking.applied},
             {"tool_call_count", outcome.tool_calls.size()}};
    record["timings_seconds"] = Json{
        {"prepare", outcome.metrics.prepare_seconds}, {"ttft", outcome.metrics.ttft_seconds},
        {"vision", outcome.metrics.vision_seconds},   {"prefill", outcome.metrics.prefill_seconds},
        {"decode", outcome.metrics.decode_seconds},   {"total", outcome.metrics.total_seconds}};
    record["engine_timing"]   = request_engine_timing_json(outcome.metrics.engine_timing);
    record["speculative"]     = speculative_json(outcome.metrics);
    record["materialization"] = materialization_json(outcome.metrics.materialization);
    return record.dump();
}

std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context,
                                      const std::string& message) {
    Json record       = event_base(server_instance_id, timestamp, "request_error");
    record["request"] = request_json(context);
    record["error"]   = Json{{"message", message}};
    return record.dump();
}

std::string format_throughput_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                   const ThroughputReport& report) {
    Json record = event_base(server_instance_id, timestamp, "throughput");
    const auto& previous = report.previous;
    const auto& current  = report.current;
    record["interval_seconds"] = report.interval_seconds;
    record["tokens"]           = Json{{"computed_prefill", report.computed_prefill_tokens},
                                      {"committed_decode", report.committed_decode_tokens},
                                      {"visible_model", monotonic_delta(previous.visible_model_tokens,
                                                                        current.visible_model_tokens)},
                                      {"hidden_terminal",
                                       monotonic_delta(previous.hidden_terminal_tokens,
                                                       current.hidden_terminal_tokens)},
                                      {"reused_prompt", monotonic_delta(previous.reused_prompt_tokens,
                                                                        current.reused_prompt_tokens)}};
    record["decode_rounds"]    = report.decode_rounds;
    record["requests"]         = Json{{"running", current.running_requests},
                                      {"waiting", current.waiting_requests}};
    record["prefix_reuse"]     = Json{{"root_selections", monotonic_delta(previous.root_selections,
                                                                          current.root_selections)},
                                      {"shared_stable_prefix_selections",
                                       monotonic_delta(previous.shared_stable_prefix_selections,
                                                       current.shared_stable_prefix_selections)}};
    return record.dump();
}

ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device               = device;
    environment.cuda_compile_version = cuda_version_string(CUDART_VERSION);

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        environment.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name                  = properties.name;
        environment.gpu_uuid                  = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.compute_capability_major  = properties.major;
        environment.compute_capability_minor  = properties.minor;
    }
    return environment;
}

JsonlRequestLog::JsonlRequestLog(const std::string& path) : path_(path) {
    if (path_.empty()) { return; }
    server_instance_id_ = new_server_instance_id();
    output_.open(path_, std::ios::out | std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open request JSONL log for append: " + path_);
    }
}

void JsonlRequestLog::write_server_start(const ServeOptions& options,
                                         const ninfer::EngineOptions& engine_options,
                                         const ninfer::ModelSamplingDefaults& sampling_defaults,
                                         const std::string& public_model_id,
                                         const ninfer::LoadSummary& load,
                                         const ninfer::MemorySummary& memory) {
    if (!enabled()) { return; }
    append(format_server_start_json(server_instance_id_, unix_time_ms(), options, engine_options,
                                    sampling_defaults, public_model_id, load, memory,
                                    query_server_log_environment(options.device)));
}

void JsonlRequestLog::write_request_start(const RequestLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_start_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_rejected(const RequestRejectionLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_rejected_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_done(const RequestLogContext& context,
                                         const GenerationOutcome& outcome) {
    if (!enabled()) { return; }
    append(format_request_done_json(server_instance_id_, unix_time_ms(), context, outcome));
}

void JsonlRequestLog::write_request_error(const RequestLogContext& context,
                                          const std::string& message) {
    if (!enabled()) { return; }
    append(format_request_error_json(server_instance_id_, unix_time_ms(), context, message));
}

void JsonlRequestLog::write_throughput(const ThroughputReport& report) {
    if (!enabled()) { return; }
    append(format_throughput_json(server_instance_id_, unix_time_ms(), report));
}

void JsonlRequestLog::append(std::string record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) { return; }
    output_ << record << '\n';
    output_.flush();
    if (!output_) {
        failed_ = true;
        write_console_log(ConsoleLogLevel::Error, "request JSONL logging failed for " + path_);
    }
}

} // namespace ninfer::serve
