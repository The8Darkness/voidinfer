#include "serve/console_log.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

using namespace voidinfer::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    ServeOptions options;
    options.model.target_directory         = "/models/qwen3.8-27b-exl3";
    options.model.draft_directory          = "/models/qwen3.8-27b-dflash2";
    options.host                           = "127.0.0.1";
    options.port                           = 8123;
    options.api_key                        = "must-not-appear";
    options.model_id_override              = "deployment-alias";
    options.request_log_jsonl              = "requests.jsonl";
    options.max_context                    = 262144;
    options.prefill_chunk                  = 1024;
    options.log_stats_interval_ms          = 2500;
    options.allow_prefix_reuse             = true;
    options.preserve_thinking              = true;
    options.default_thinking_budget        = 512;
    options.sampling_overrides.temperature = 0.6F;
    options.startup_argv = {"voidinfer-serve", "--target", "--api-key", "<redacted>"};

    voidinfer::EngineOptions engine_options;
    engine_options.model                                           = options.model;
    engine_options.device                                          = options.device;
    engine_options.max_context                                     = options.max_context;
    engine_options.max_concurrency                                 = 2;
    engine_options.max_pending_requests                            = options.max_pending_requests;
    engine_options.pending_timeout_ms                              = options.pending_timeout_ms;
    engine_options.prefill_chunk                                   = options.prefill_chunk;
    engine_options.context_cache.device_state_slots                = 2;
    engine_options.context_cache.host_state_slots                  = 3;
    engine_options.context_cache.host_kv_capacity_bytes            = 64ULL << 20;
    engine_options.context_cache.max_private_continuations         = 4;
    engine_options.context_cache.max_shared_prefixes               = 2;
    engine_options.context_cache.max_long_anchors_per_continuation = 2;
    engine_options.context_cache.max_cache_markers_per_request     = 4;

    const voidinfer::ModelSamplingDefaults sampling_defaults{
        .thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F},
        .non_thinking = {.temperature = 0.7F, .top_k = 20, .top_p = 0.8F, .presence_penalty = 1.5F},
    };

    voidinfer::LoadSummary load;
    load.target               = "qwen3.8-27b/exl3";
    load.model_id             = "qwen3.8-27b";
    load.weights_id           = "exl3-6bpw";
    load.load_seconds         = 1.234567890123;
    load.upload_seconds       = 0.345678901234;
    load.model_bytes_read     = 1000;
    load.host_to_device_bytes = 900;
    load.peak_staging_bytes   = 128;

    voidinfer::MemorySummary memory;
    memory.max_context                       = 262144;
    memory.weights.capacity_bytes            = 100;
    memory.sequence.capacity_bytes           = 200;
    memory.workspace.capacity_bytes          = 500;
    memory.minimum_runtime_reservation_bytes = 1300;
    memory.runtime_reservation_bytes         = 1600;
    memory.available_after_weights_bytes     = 1700;
    memory.available_after_startup_bytes     = 180;
    memory.planned_slack_bytes               = 100;
    memory.cuda_graph_allowance_bytes        = 600;
    memory.kv_payload_bytes                  = 400;
    memory.host_state_capacity_slots         = 3;
    memory.host_state_occupied_slots         = 1;
    memory.host_kv_capacity_bytes            = 64ULL << 20;
    memory.host_kv_occupied_bytes            = 8ULL << 20;

    ServerLogEnvironment environment;
    environment.device                    = 0;
    environment.gpu_name                  = "NVIDIA GeForce RTX 5090";
    environment.gpu_uuid                  = "GPU-00000000-0000-0000-0000-000000000000";
    environment.total_device_memory_bytes = 32000000000ULL;
    environment.compute_capability_major  = 12;
    environment.compute_capability_minor  = 0;
    environment.cuda_compile_version      = "13.1";
    environment.cuda_runtime_version      = "13.1";
    environment.cuda_driver_version       = "13.1";

    const Json server = Json::parse(format_server_start_json(
        "serve-test", 1000, options, engine_options, sampling_defaults, "deployment-alias", load,
        memory, environment));
    failures += check(server.at("artifact_type") == kRequestLogArtifactType,
                      "server record artifact type mismatch");
    failures += check(server.at("schema_version") == kRequestLogSchemaVersion,
                      "server record schema mismatch");
    failures += check(server.at("event") == "server_start", "server event mismatch");
    failures += check(server.at("server").at("public_model_id") == "deployment-alias",
                      "resolved public model id missing");
    failures += check(server.at("model").at("identity") == "qwen3.8-27b/exl3" &&
                          server.at("model").at("target") == "/models/qwen3.8-27b-exl3" &&
                          server.at("model").at("draft") == "/models/qwen3.8-27b-dflash2",
                      "server model package missing");
    failures += check(server.at("model").at("weights_id") == "exl3-6bpw",
                      "server weights id missing");
    failures += check(server.at("model").at("bytes_read") == 1000, "model bytes read missing");
    failures += check(server.at("engine").at("max_context") == 262144, "max context missing");
    failures +=
        check(server.at("engine").at("log_stats_interval_ms") == 2500, "stats interval missing");
    failures += check(server.at("server").at("request_log_jsonl") == "requests.jsonl",
                      "request log path missing");
    failures += check(server.at("server").at("default_thinking_budget") == 512,
                      "server thinking budget missing");
    failures += check(server.at("engine").at("vision") == false, "Vision state missing");
    failures += check(server.at("engine").at("speculative") == "dflash2-k7",
                      "speculative profile missing");
    failures += check(server.at("engine").at("prefix_reuse") == true, "prefix-reuse state missing");
    failures += check(
        server.at("engine").at("context_cache").at("device_state_slots") == 2 &&
            server.at("engine").at("context_cache").at("total_device_state_slots") == 4 &&
            server.at("engine").at("context_cache").at("host_state_slots") == 3 &&
            server.at("engine").at("context_cache").at("host_kv_capacity_bytes") == (64ULL << 20) &&
            server.at("engine").at("context_cache").at("max_private_continuations") == 4 &&
            server.at("engine").at("context_cache").at("max_shared_prefixes") == 2,
        "resolved context-cache configuration missing");
    failures += check(server.at("server").at("default_preserve_thinking") == true,
                      "server preserve-thinking default missing");
    failures +=
        check(server.at("sampling_defaults").at("thinking").at("temperature") == 1.0 &&
                  server.at("sampling_defaults").at("non_thinking").at("presence_penalty") == 1.5,
              "registered mode-specific sampling defaults missing");
    failures += check(
        server.at("sampling_defaults").at("server_overrides").at("temperature").get<float>() ==
                0.6F &&
            server.at("sampling_defaults").at("server_overrides").at("top_p").is_null(),
        "server sampling overrides lost omission state");
    failures += check(server.at("environment").at("gpu_name") == "NVIDIA GeForce RTX 5090",
                      "GPU name missing");
    failures += check(server.at("memory").at("workspace").at("capacity_bytes") == 500,
                      "workspace memory missing");
    failures += check(server.at("memory").at("cuda_graph_allowance_bytes") == 600,
                      "CUDA Graph allowance missing");
    failures += check(server.at("memory").at("runtime_reservation_bytes") == 1600 &&
                          server.at("memory").at("available_after_weights_bytes") == 1700 &&
                          server.at("memory").at("available_after_startup_bytes") == 180 &&
                          server.at("memory").at("planned_slack_bytes") == 100,
                      "adaptive KV memory ledger missing");
    failures += check(server.at("memory").at("host_state_capacity_slots") == 3 &&
                          server.at("memory").at("host_state_occupied_slots") == 1 &&
                          server.at("memory").at("host_kv_capacity_bytes") == (64ULL << 20) &&
                          server.at("memory").at("host_kv_occupied_bytes") == (8ULL << 20),
                      "Host context-cache memory ledger missing");
    failures += check(server.dump().find("must-not-appear") == std::string::npos,
                      "server JSON leaked the API key");
    failures += check(server.at("argv").at(3) == "<redacted>",
                      "server argv did not retain the redaction marker");

    GenerationRequest request;
    request.model          = "qwen3.6-27b";
    request.stream         = false;
    request.max_tokens     = 4096;
    request.max_tokens_set = true;
    request.messages.resize(2);
    request.messages.front().content.push_back(ContentPart{.kind = ContentKind::Image});

    PreparedRequest prepared;
    prepared.enable_thinking                           = true;
    prepared.thinking_budget                           = 256;
    prepared.preserve_thinking                         = true;
    prepared.preserve_thinking_semantic_change         = true;
    prepared.sampling.temperature                      = 0.6F;
    prepared.sampling.top_p                            = 0.95F;
    prepared.sampling.top_k                            = 20;
    prepared.sampling.min_p                            = 0.0F;
    prepared.sampling.presence_penalty                 = 1.0F;
    prepared.sampling.frequency_penalty                = 0.0F;
    prepared.sampling.seed                             = 7632647173703958409ULL;
    prepared.acquisition_seconds                       = 0.004;
    prepared.preparation.seconds                       = 0.12;
    prepared.preparation.media_preprocess_seconds      = 0.08;
    prepared.preparation.media_preprocess_work_seconds = 0.31;
    prepared.preparation.tokenize_seconds              = 0.02;
    prepared.preparation.media_items                   = 1;
    prepared.preparation.media_cache_misses            = 1;
    prepared.preparation.built_patch_bytes             = 49152;

    const RequestLogContext context =
        make_request_log_context(7, "openai_chat_completions", request, prepared);
    const Json started = Json::parse(format_request_start_json("serve-test", 2000, context));
    failures +=
        check(started.at("request").at("request_id") == 7, "request id missing from start record");
    failures += check(started.at("request").at("requested_output_tokens") == 4096,
                      "request output budget missing");
    failures += check(started.at("request").at("enable_thinking") == true,
                      "resolved thinking mode missing");
    failures += check(started.at("request").at("thinking_budget") == 256,
                      "resolved thinking budget missing");
    failures += check(started.at("request").at("preserve_thinking") == true &&
                          started.at("request").at("preserve_thinking_semantic_change") == true,
                      "resolved preserve-thinking metadata missing");
    failures += check(started.at("request").at("sampling").at("seed") == 7632647173703958409ULL,
                      "resolved seed missing");
    failures += check(started.at("request").at("media_item_count") == 1 &&
                          started.at("preparation_seconds").at("acquisition") == 0.004 &&
                          started.at("preparation_seconds").at("media_preprocess_work") == 0.31 &&
                          started.at("preparation_seconds").at("tokenize") == 0.02 &&
                          started.at("preparation_seconds").at("cache_misses") == 1,
                      "request-scoped media preparation diagnostics missing");

    ApiError preparation_error;
    preparation_error.status = 400;
    preparation_error.type   = "invalid_request_error";
    preparation_error.param  = "messages";
    preparation_error.code   = "context_length_exceeded";
    preparation_error.message =
        "prepared prompt has 270000 tokens, exceeding Engine max_context 262144";
    const RequestRejectionLogContext rejected_context =
        make_request_rejection_log_context(8, "anthropic_messages", request, preparation_error);
    const Json rejected =
        Json::parse(format_request_rejected_json("serve-test", 2500, rejected_context));
    failures +=
        check(rejected.at("event") == "request_rejected" && rejected.at("phase") == "prepare",
              "preparation rejection event or phase mismatch");
    failures += check(rejected.at("request").at("request_id") == 8 &&
                          rejected.at("request").at("media_item_count") == 1 &&
                          rejected.at("request").at("message_count") == 2,
                      "preparation rejection request shape missing");
    failures += check(rejected.at("error").at("status") == 400 &&
                          rejected.at("error").at("code") == "context_length_exceeded" &&
                          rejected.at("error").at("param") == "messages",
                      "preparation rejection API error missing");
    failures +=
        check(format_request_rejected(rejected_context)
                          .find("rejected phase=prepare protocol=anthropic_messages") !=
                      std::string::npos &&
                  format_request_rejected(rejected_context).find("code=context_length_exceeded") !=
                      std::string::npos,
              "human preparation rejection log is incomplete");

    GenerationOutcome outcome;
    outcome.prompt_tokens                   = 401;
    outcome.completion_tokens               = 1024;
    outcome.finish_reason                   = voidinfer::FinishReason::OutputLimit;
    outcome.metrics.prepare_seconds         = 0.1234567890123;
    outcome.metrics.ttft_seconds            = 0.3580246791357;
    outcome.metrics.vision_seconds          = 0.0;
    outcome.metrics.prefill_seconds         = 0.2345678901234;
    outcome.metrics.decode_seconds          = 5.3456789012345;
    outcome.metrics.total_seconds           = 5.7037035803702;
    outcome.metrics.prefix_cache_hit_tokens = 101;
    outcome.metrics.prefix_reuse_path       = voidinfer::PrefixReusePath::PrivateTurnClosure;
    outcome.metrics.engine_timing           = {
                  .queue_wait_seconds                   = 0.001,
                  .engine_boundary_exposed_seconds      = 0.001,
                  .program_submit_exposed_seconds       = 0.002,
                  .program_post_exposed_seconds         = 0.003,
                  .engine_commit_output_exposed_seconds = 0.004,
                  .engine_maintenance_exposed_seconds   = 0.005,
                  .device_wait_exposed_seconds          = 0.3,
                  .decode_host_exposed_seconds          = 0.01,
                  .decode_device_wait_exposed_seconds   = 0.2,
                  .prefill_units                        = 4,
                  .decode_rounds                        = 2,
                  .control_units                        = 1,
    };
    outcome.metrics.speculative_enabled               = true;
    outcome.metrics.speculative_draft_window          = 3;
    outcome.metrics.speculative_rounds                = 300;
    outcome.metrics.speculative_draft_tokens          = 900;
    outcome.metrics.speculative_accepted_tokens       = 720;
    outcome.metrics.speculative_fallback_steps        = 2;
    outcome.metrics.speculative_accepted_per_position = {290, 240, 190};
    outcome.metrics.speculative_neural_input_rows=1200;
    outcome.metrics.speculative_returned_suffix_rows=900;
    outcome.metrics.speculative_discarded_suffix_rows=300;
    outcome.metrics.speculative_proposed_rows=900;
    outcome.metrics.speculative_verified_rows=1024;
    outcome.metrics.speculative_replayed_rows=124;
    outcome.metrics.speculative_committed_model_rows=1024;
    outcome.metrics.speculative_externally_visible_model_rows=1023;
    outcome.metrics.speculative_hidden_terminal_rows=1;
    outcome.metrics.speculative_diagnostic_rows=17;
    outcome.metrics.speculative_shared_projection_rows=std::nullopt;
    outcome.metrics.materialization                   = {
                          .predicted_now_ns              = 200000,
                          .predicted_future_loss_ns      = 50000,
                          .predicted_total_ns            = 250000,
                          .targets_evaluated             = 7,
                          .projection_work               = 31,
                          .planning_elapsed_ns           = 9000,
                          .search_elapsed_ns             = 6000,
                          .stop_reason                   = voidinfer::MaterializationStopReason::ModelOptimal,
                          .model_optimal                 = true,
                          .budget_exhausted              = false,
                          .best_remaining_lower_bound_ns = 250000,
                          .absolute_bound_gap_ns         = 0,
                          .relative_bound_gap            = 0.0,
                          .selected_degradation_units    = 2,
                          .selected_maximal_fallback     = false,
    };
    outcome.thinking = voidinfer::ThinkingBudgetStats{.configured_budget     = 256,
                                                   .model_thinking_tokens = 256,
                                                   .injected_tokens       = 19,
                                                   .applied               = true};

    const Json done = Json::parse(format_request_done_json("serve-test", 3000, context, outcome));
    failures +=
        check(done.at("result").at("finish_reason") == "output_limit", "finish reason missing");
    failures += check(done.at("result").at("prompt_tokens") == 401, "prompt tokens missing");
    failures += check(done.at("result").at("computed_prefill_tokens") == 300,
                      "computed prefill tokens missing");
    failures += check(done.at("result").at("prefix_reuse_path") == "private_turn_closure",
                      "prefix reuse path missing");
    failures += check(done.at("result").at("thinking_budget") == 256 &&
                          done.at("result").at("model_thinking_tokens") == 256 &&
                          done.at("result").at("thinking_control_tokens") == 19 &&
                          done.at("result").at("thinking_control_applied") == true,
                      "thinking-control result accounting missing");
    outcome.metrics.prefix_reuse_path = voidinfer::PrefixReusePath::PrivateResponseReplay;
    const Json response_restore =
        Json::parse(format_request_done_json("serve-test", 3001, context, outcome));
    failures +=
        check(response_restore.at("result").at("prefix_reuse_path") == "private_response_replay",
              "response checkpoint reuse path missing");
    failures += check(done.at("timings_seconds").at("decode").get<double>() ==
                          outcome.metrics.decode_seconds,
                      "decode time lost precision");
    failures +=
        check(done.at("timings_seconds").at("ttft").get<double>() ==
                          outcome.metrics.ttft_seconds,
              "TTFT missing or lost precision");
    failures += check(done.at("speculative").at("backend") == "dflash2",
                      "speculative backend missing");
    failures +=
        check(done.at("speculative").at("draft_window") == 3, "draft window missing");
    failures += check(done.at("speculative").at("fallback_steps") == 2,
                      "speculative fallback count missing");
    failures +=
        check(done.at("speculative").at("accepted_per_position") == Json::array({290, 240, 190}),
              "speculative position counts missing");
    failures += check(done.at("speculative").at("neural_input_rows")==1200 &&
                          done.at("speculative").at("returned_suffix_rows")==900 &&
                          done.at("speculative").at("discarded_suffix_rows")==300 &&
                          done.at("speculative").at("proposed_rows")==900 &&
                          done.at("speculative").at("verified_rows")==1024 &&
                          done.at("speculative").at("replayed_rows")==124 &&
                          done.at("speculative").at("committed_model_rows")==1024 &&
                          done.at("speculative").at("externally_visible_model_rows")==1023 &&
                          done.at("speculative").at("hidden_terminal_rows")==1 &&
                          done.at("speculative").at("diagnostic_rows")==17 &&
                          done.at("speculative").at("shared_projection_rows").is_null(),
                      "complete-route speculative accounting collapsed or fabricated sharing");
    failures += check(done.at("materialization").at("predicted_total_ns") == 250000 &&
                          done.at("materialization").at("targets_evaluated") == 7 &&
                          done.at("materialization").at("stop_reason") == "model_optimal" &&
                          done.at("materialization").at("model_optimal") == true,
                      "request-owned materialization diagnostics missing");
    failures += check(
        done.at("engine_timing").at("queue_wait_seconds") == 0.001 &&
            std::abs(done.at("engine_timing").at("host_exposed_seconds").at("total").get<double>() -
                     0.015) < 1.0e-15 &&
            done.at("engine_timing").at("decode").at("rounds") == 2 &&
            done.at("engine_timing").at("units").at("prefill") == 4,
        "request Engine timing exposure is incomplete");

    const Json error =
        Json::parse(format_request_error_json("serve-test", 4000, context, "generation failed"));
    failures += check(error.at("event") == "request_error", "request error event mismatch");
    failures += check(error.at("error").at("message") == "generation failed",
                      "request error message missing");

    failures +=
        check(format_request_start(context).find("thinking=on") != std::string::npos &&
                  format_request_start(context).find("thinking_budget=256") != std::string::npos,
              "human request log omits resolved thinking control");
    failures +=
        check(format_request_start(context).find("preserve_thinking=on") != std::string::npos,
              "human request log omits preserve-thinking mode");
    failures += check(format_request_done(context, outcome).find("reuse=private_response_replay") !=
                          std::string::npos,
                      "human request log omits response checkpoint reuse path");
    failures +=
        check(format_request_done(context, outcome).find("host=15.00ms") != std::string::npos &&
                  format_request_done(context, outcome).find("decode-host=5000.0us/round") !=
                      std::string::npos,
              "human request log omits Engine Host exposure");
    failures += check(format_request_start(context).find("submitted") != std::string::npos,
                      "human request log mislabels a submitted request");

    ThroughputReport throughput;
    throughput.interval_seconds              = 2.0;
    throughput.computed_prefill_tokens       = 100;
    throughput.committed_decode_tokens       = 40;
    throughput.decode_rounds                 = 10;
    throughput.previous.root_selections      = 2;
    throughput.previous.reused_prompt_tokens = 1000;
    throughput.current.running_requests      = 2;
    throughput.current.waiting_requests      = 3;
    throughput.current.root_selections       = 3;
    throughput.current.reused_prompt_tokens  = 1512;
    const std::string human_throughput = format_throughput(throughput);
    failures += check(human_throughput.find("prefill=50.0tok/s") != std::string::npos &&
                          human_throughput.find("decode=20.0tok/s") != std::string::npos &&
                          human_throughput.find("rounds=10") != std::string::npos &&
                          human_throughput.find("running=2") != std::string::npos &&
                          human_throughput.find("waiting=3") != std::string::npos &&
                          human_throughput.find("reused_prompt_tokens=512") != std::string::npos,
                      "human throughput report mismatch");
    const Json throughput_json =
        Json::parse(format_throughput_json("serve-test", 5000, throughput));
    failures += check(throughput_json.at("event") == "throughput", "throughput event mismatch");
    failures += check(throughput_json.at("tokens").at("computed_prefill") == 100 &&
                          throughput_json.at("tokens").at("committed_decode") == 40 &&
                          throughput_json.at("tokens").at("reused_prompt") == 512,
                      "throughput token deltas mismatch");
    failures += check(throughput_json.at("decode_rounds") == 10 &&
                          throughput_json.at("requests").at("running") == 2 &&
                          throughput_json.at("requests").at("waiting") == 3 &&
                          throughput_json.at("prefix_reuse").at("root_selections") == 1,
                      "throughput gauges or interval deltas mismatch");

    const std::string console_prefix =
        format_console_log_prefix(std::chrono::system_clock::time_point{}, ConsoleLogLevel::Info);
    failures += check(console_prefix.starts_with('[') &&
                          console_prefix.ends_with("] [info] voidinfer-serve: "),
                      "console log prefix mismatch");

    const std::filesystem::path log_path =
        std::filesystem::temp_directory_path() /
        ("voidinfer-request-log-test-" +
#ifdef _WIN32
         std::to_string(static_cast<long long>(::_getpid())) +
#else
         std::to_string(static_cast<long long>(::getpid())) +
#endif
         ".jsonl");
    std::filesystem::remove(log_path);
    {
        JsonlRequestLog writer(log_path.string());
        writer.write_request_start(context);
    }
    {
        JsonlRequestLog writer(log_path.string());
        writer.write_request_rejected(rejected_context);
        writer.write_request_error(context, "generation failed");
    }
    std::ifstream input(log_path);
    std::string first_line;
    std::string second_line;
    std::string third_line;
    std::string extra_line;
    std::getline(input, first_line);
    std::getline(input, second_line);
    std::getline(input, third_line);
    std::getline(input, extra_line);
    failures += check(!first_line.empty() && !second_line.empty() && !third_line.empty() &&
                          extra_line.empty(),
                      "JSONL writer did not append exactly one flushed line per event");
    if (!first_line.empty() && !second_line.empty() && !third_line.empty()) {
        failures += check(Json::parse(first_line).at("event") == "request_start",
                          "first appended event mismatch");
        failures += check(Json::parse(second_line).at("event") == "request_rejected",
                          "second appended event mismatch");
        failures += check(Json::parse(third_line).at("event") == "request_error",
                          "third appended event mismatch");
    }
    input.close();
    std::filesystem::remove(log_path);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
