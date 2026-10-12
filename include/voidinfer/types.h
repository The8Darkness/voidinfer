#pragma once

#include <chrono>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace voidinfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes    = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes     = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes      = 2ULL << 30;
inline constexpr std::uint32_t kDefaultHostStateSlots    = 8;
inline constexpr std::size_t kDefaultHostKvCapacityBytes = 8ULL << 30;

struct LoadProgress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
};

struct ContextCacheOptions {
    // Engine resolves every optional once at construction. With C=max_concurrency, the enabled
    // defaults are H=C, R=8, Host KV=8 GiB, P=2C, S=C, L=2 and M=4; Engine::options() returns
    // those effective values.
    bool enabled = true;
    // Extra Device checkpoint StateImage slots H. Total Device StateImage capacity is C + H.
    std::optional<std::uint32_t> device_state_slots;
    // Host StateImages and Host KV bytes are independently configured pinned-memory capacities.
    std::uint32_t host_state_slots     = kDefaultHostStateSlots;
    std::size_t host_kv_capacity_bytes = kDefaultHostKvCapacityBytes;
    // Bounded private/shared logical catalogs and per-continuation long-anchor count.
    std::optional<std::uint32_t> max_private_continuations;
    std::optional<std::uint32_t> max_shared_prefixes;
    std::optional<std::uint32_t> max_long_anchors_per_continuation;
    // Input-complexity bound; this does not reserve checkpoint storage.
    std::optional<std::uint32_t> max_cache_markers_per_request;
};

// The model package: an EXL3 target, its DFlash2 draft and, optionally, the
// BF16 Vision tower (`model.visual.*` tensors, e.g. the target's BF16
// checkpoint), which enables image and video input.
struct ModelPackageOptions {
    std::filesystem::path target_directory;
    std::filesystem::path draft_directory;
    std::filesystem::path vision_directory;
    std::filesystem::path verified_dual_manifest;
};

struct EngineOptions {
    ModelPackageOptions model;
    int device                         = 0;
    std::uint32_t max_context          = 32768; // Exact logical ceiling of each request.
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 1024;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    ContextCacheOptions context_cache;
    LoadProgress load_progress;

    [[nodiscard]] bool vision_enabled() const noexcept { return !model.vision_directory.empty(); }
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    Low,
    Medium,
    XHigh,
};

struct ReasoningEffortCapabilities {
    bool low    = false;
    bool medium = false;
    bool xhigh  = false;
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] constexpr bool supports(ReasoningEffort effort) const noexcept {
        switch (effort) {
        case ReasoningEffort::Low:
            return low;
        case ReasoningEffort::Medium:
            return medium;
        case ReasoningEffort::XHigh:
            return xhigh;
        }
        return false;
    }
};

struct PromptCapabilities {
    bool enable_thinking = false;
    ReasoningEffortCapabilities reasoning_effort;
};

struct PromptOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
    bool add_vision_id     = false;
    std::vector<std::string> tool_jsons;
};

enum class CacheRetentionHint : std::uint8_t {
    Default,
    LiveSession,
    Disposable,
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    CacheRetentionHint retention = CacheRetentionHint::Default;
    std::vector<PromptCacheMarker> markers;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
    UnsupportedSampling,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
    // Per-input snapshot when preparation completed, deduplicated within that
    // input. Includes payload objects and patch arrays, excludes allocator and
    // shared_ptr control-block overhead. Do not sum across requests sharing media.
    std::size_t retained_media_payload_bytes = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
    // Optional opaque lifetime retained by an Engine while this delivered
    // message remains owned by the sink. Applications must not interpret it.
    std::shared_ptr<const void> retained_owner;
};

class OutputSink {
public:
    virtual ~OutputSink()                   = default;
    virtual void publish(OutputDelta delta) = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double total_seconds       = 0.0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                   = 0.0;
    double engine_boundary_exposed_seconds      = 0.0;
    double program_submit_exposed_seconds       = 0.0;
    double program_post_exposed_seconds         = 0.0;
    double engine_commit_output_exposed_seconds = 0.0;
    double engine_maintenance_exposed_seconds   = 0.0;
    double device_wait_exposed_seconds          = 0.0;
    double decode_host_exposed_seconds          = 0.0;
    double decode_device_wait_exposed_seconds   = 0.0;
    std::uint64_t prefill_units                 = 0;
    std::uint64_t decode_rounds                 = 0;
    std::uint64_t control_units                 = 0;
};

struct SpeculativeStats {
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    // Optional device-side phase telemetry. These remain zero unless the DFlash phase profiler
    // is explicitly enabled; keeping them on the result makes benchmark and serving records
    // comparable without imposing event overhead on the default decode path.
    double draft_seconds    = 0.0;
    double selector_seconds = 0.0;
    double verifier_seconds = 0.0;
    std::uint64_t verifier_calls = 0;
    // Default-off exact repair checkpoint work. Captured bytes are charged per
    // verifier attempt; reconstructed/fallback rows are target work, not tokens.
    std::uint64_t repair_checkpoint_captured_bytes = 0;
    std::uint64_t repair_checkpoint_restores = 0;
    std::uint64_t repair_checkpoint_reconstructed_rows = 0;
    std::uint64_t repair_checkpoint_fallback_rows = 0;
    std::uint64_t device_seed_handoffs = 0;
    std::uint64_t device_seed_host_fallbacks = 0;
    std::uint64_t draft_local_topk_calls = 0;
    std::uint64_t draft_dense_kmajor_launches = 0;
    // Request-local eager OSCAR continuation-cohort routing. These distinguish
    // actual candidate dispatch from an enabled flag and from exact fallback.
    std::uint64_t oscar_eager_cohort_attempts = 0;
    std::uint64_t oscar_eager_cohort_dispatches = 0;
    std::uint64_t oscar_eager_cohort_latch_misses = 0;
    std::uint64_t oscar_eager_cohort_boundary_fallbacks = 0;
    std::uint64_t oscar_eager_cohort_malformed = 0;
    std::uint64_t staged_b8_verifier_calls = 0;
    std::uint64_t staged_b8_first_half_exits = 0;
    std::uint64_t staged_b8_second_half_calls = 0;
    std::uint64_t staged_b8_skipped_verification_rows = 0;
    // Diagnostic request-local sequence of accepted draft rows for each
    // published DFlash round. This is evidence for staged-verifier economics,
    // not output and not a controller decision by itself.
    std::vector<std::uint8_t> accepted_prefix_per_round;
    // Diagnostic copy of the first request-local proposal. This lets exact
    // target graph oracles replay the real draft-conditioned B8 without
    // changing proposal, verification, or publication behavior.
    std::vector<std::int64_t> first_proposed_tokens;
    // Actual committed target-tap bytes copied device-to-device into draft
    // ingestion. This is transport volume, not saved time or throughput.
    std::uint64_t committed_tap_device_bytes = 0;
    // Committed rows for which all five target-tap planes remained on device.
    // This is an exercised-path counter, not a latency estimate.
    std::uint64_t committed_tap_host_export_rows_avoided = 0;
    std::vector<std::uint64_t> accepted_per_position;
    // Number of rounds that actually proposed each speculative position. This can differ from
    // the selected K near a request budget boundary, so controllers must use it for survival
    // denominators instead of inferring proposals from the selected-K histogram.
    std::vector<std::uint64_t> proposed_per_position;
    std::vector<std::uint64_t> selected_k_histogram;
    // Device verifier time accumulated separately for each selected K. This lets the adaptive
    // controller learn the actual width/cost curve instead of assuming linear scaling.
    std::vector<double> verifier_seconds_by_k;
    // Physical B8 work reported by the EXL3 caller. These count successful
    // neural attempts, not accepted output; other backends leave them zero.
    std::uint64_t neural_input_rows=0,returned_suffix_rows=0,discarded_suffix_rows=0;
    // Complete-route work remains distinct from useful output. Verified rows
    // include original target checks; replayed rows are additional repair work.
    // Per-request shared projection rows are deliberately absent because a C2
    // physical dispatch has no truthful single-request attribution.
    std::uint64_t proposed_rows=0,verified_rows=0,replayed_rows=0;
    std::uint64_t committed_model_rows=0,externally_visible_model_rows=0;
    std::uint64_t hidden_terminal_rows=0,diagnostic_rows=0;
    std::optional<std::uint64_t> shared_projection_rows;
};

enum class VisionPatchStorage : std::uint8_t { BFloat16, Float16 };

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
};

// Per-result semantic token partition. Model tokens before an unpublished
// terminal marker, injected target-control tokens and the hidden marker remain
// distinct. Diagnostic state rows are intentionally outside result-token and
// throughput conservation.
struct GenerationTokenAccounting {
    std::uint64_t visible_model_tokens = 0;
    std::uint64_t injected_control_tokens = 0;
    std::uint64_t hidden_terminal_tokens = 0;
    std::uint64_t diagnostic_state_rows = 0;
    [[nodiscard]] bool conserves_result_tokens(std::size_t result_tokens) const noexcept {
        constexpr auto maximum=std::numeric_limits<std::uint64_t>::max();
        if(visible_model_tokens>maximum-injected_control_tokens)return false;
        const auto visible_and_control=visible_model_tokens+injected_control_tokens;
        if(visible_and_control>maximum-hidden_terminal_tokens)return false;
        return visible_and_control+hidden_terminal_tokens==result_tokens;
    }
};

enum class PrefixReusePath : std::uint8_t {
    Root,
    PrivateEndpoint,
    PrivateTurnClosure,
    PrivateResponseReplay,
    PrivateLongAnchor,
    SharedStablePrefix,
};

// Why pressure planning stopped for the materialization decision committed to one request.
// "ModelOptimal" is relative to the configured target graph, canonical transaction order, and
// numerical cost model; it is not a claim about globally optimal observed TTFT.
enum class MaterializationStopReason : std::uint8_t {
    NoPressure,
    ModelOptimal,
    QueueExhausted,
    TargetBudget,
    ExpansionCapacity,
    TimeBudget,
    ValueOfNextExpansion,
};

[[nodiscard]] inline constexpr const char*
materialization_stop_reason_name(MaterializationStopReason reason) noexcept {
    switch (reason) {
    case MaterializationStopReason::NoPressure:
        return "no_pressure";
    case MaterializationStopReason::ModelOptimal:
        return "model_optimal";
    case MaterializationStopReason::QueueExhausted:
        return "queue_exhausted";
    case MaterializationStopReason::TargetBudget:
        return "target_budget";
    case MaterializationStopReason::ExpansionCapacity:
        return "expansion_capacity";
    case MaterializationStopReason::TimeBudget:
        return "time_budget";
    case MaterializationStopReason::ValueOfNextExpansion:
        return "value_of_next_expansion";
    }
    return "no_pressure";
}

struct MaterializationDiagnostics {
    std::uint64_t predicted_now_ns              = 0;
    std::uint64_t predicted_future_loss_ns      = 0;
    std::uint64_t predicted_total_ns            = 0;
    std::uint32_t targets_evaluated             = 0;
    std::uint64_t projection_work               = 0;
    std::uint64_t planning_elapsed_ns           = 0;
    std::uint64_t search_elapsed_ns             = 0;
    MaterializationStopReason stop_reason       = MaterializationStopReason::NoPressure;
    bool model_optimal                          = true;
    bool budget_exhausted                       = false;
    std::uint64_t best_remaining_lower_bound_ns = 0;
    std::uint64_t absolute_bound_gap_ns         = 0;
    double relative_bound_gap                   = 0.0;
    std::uint32_t selected_degradation_units    = 0;
    bool selected_maximal_fallback              = false;

    [[nodiscard]] friend constexpr bool
    operator==(const MaterializationDiagnostics&,
               const MaterializationDiagnostics&) noexcept = default;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::uint32_t reasoning_tokens     = 0;
    FinishReason finish_reason         = FinishReason::None;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;
    MaterializationDiagnostics materialization;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
    GenerationTokenAccounting token_accounting;
};

enum class GenerationPollState : std::uint8_t { Pending,Completed,Error };
struct GenerationPollResult {
    GenerationPollState state=GenerationPollState::Pending;
    std::optional<GenerationResult> result;
    std::exception_ptr error;
    [[nodiscard]] bool terminal() const noexcept {
        return state!=GenerationPollState::Pending;
    }
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::uint32_t host_state_capacity_slots       = 0;
    std::uint32_t host_state_occupied_slots       = 0;
    std::size_t host_kv_capacity_bytes            = 0;
    std::size_t host_kv_occupied_bytes            = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    // Actual prompt tokens evaluated by prefill; reused checkpoint-prefix tokens are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Partition of committed EXL3 result tokens. Diagnostic state rows are not
    // included in any of these counters or in useful output throughput.
    std::uint64_t visible_model_tokens = 0;
    std::uint64_t hidden_terminal_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t waiting_requests          = 0;

    std::uint64_t root_selections                    = 0;
    std::uint64_t shared_stable_prefix_selections    = 0;
    std::uint64_t reused_prompt_tokens               = 0;
    std::uint32_t last_selected_frontier_tokens      = 0;

    // Completed physical projection batches and independent rendezvous fallbacks;
    // neither counter establishes token/state equivalence or a speedup.
    std::uint64_t shared_target_projection_batches = 0;
    std::uint64_t shared_target_projection_rows = 0;
    std::uint64_t shared_target_projection_failures = 0;
    std::uint64_t shared_target_projection_fallbacks = 0;
    // Compact target reductions remain per request. These count only the bounded
    // C2 result-transport gather and its independent one-row fallbacks.
    std::uint64_t greedy_packet_transport_batches=0;
    std::uint64_t greedy_packet_transport_rows=0;
    std::uint64_t greedy_packet_transport_singles=0;
    std::uint64_t greedy_packet_transport_cancellations=0;
    std::uint64_t greedy_packet_transport_failures=0;
    std::array<std::uint64_t,15> shared_target_projection_family_batches{}; // Q,K,V,draft Q,O,gate,up,down,head,draft K,draft V,draft O,draft down,draft gate,draft up
    std::array<std::uint64_t,15> shared_target_projection_family_fallbacks{}; // eligible offers completed privately, same family order
    std::array<std::uint64_t,64> shared_target_layer_batches{}; // excludes draft layers
    std::uint64_t shared_target_reused_gather_bytes=0;
    std::uint64_t shared_device_prefix_bytes=0;
    std::uint64_t shared_device_prefix_hit_bytes=0,shared_device_prefix_busy_fallbacks=0;
    // Request deltas for the lane-private represented-prefix cache. These are
    // redundant device views; authoritative publication remains exact HostKV.
    std::uint64_t private_device_prefix_hit_bytes=0,private_device_prefix_fill_bytes=0;
    std::uint64_t private_device_prefix_segmented_bytes=0;
    std::uint64_t private_device_prefix_forward_publish_forwards=0;
    std::uint64_t private_device_prefix_partial_hit_bytes=0;
    std::uint64_t private_device_prefix_forward_publish_bytes=0;
    std::uint64_t host_kv_h2d_bytes=0,host_kv_d2h_bytes=0;
    std::uint64_t host_kv_transfer_calls=0,host_kv_copy_submissions=0;
    std::uint64_t host_kv_pinned_slot_waits=0;
    std::uint64_t host_kv_pinned_staging_bytes=0;
    std::uint64_t exact_kv_export_batched_calls=0;
    std::uint64_t exact_kv_export_batched_ranges=0;
    std::uint64_t exact_kv_export_batched_bytes=0;
    std::uint64_t exact_kv_export_batched_fences=0;
    std::uint64_t exact_kv_export_saved_fences=0;
    std::uint64_t prefill_pinned_batch_plane_calls=0;
    std::uint64_t prefill_pinned_batch_page_planes=0;
    std::uint64_t prefill_pinned_batch_bytes=0;
    std::uint64_t host_kv_banked_d2h_forwards=0,host_kv_banked_d2h_planes=0;
    std::uint64_t host_kv_banked_d2h_rows=0,host_kv_banked_d2h_bytes=0;
    std::uint64_t host_kv_banked_d2h_drains=0;
    std::uint64_t registered_kv_upload_bytes=0,registered_kv_upload_fallbacks=0;
    std::uint64_t registered_kv_retired_registrations=0;
    std::uint64_t registered_kv_external_readers=0;
    std::uint64_t registered_kv_external_reader_high_water=0;
    // Same bytes partitioned by the two physical Engine lanes; not additive to total.
    std::array<std::uint64_t,8> registered_kv_upload_lane_bytes{};
    std::uint64_t registered_kv_upload_failures=0;
    std::uint64_t registered_kv_upload_peak_pending=0;
    std::uint64_t shared_device_page_fill_bytes=0,shared_device_page_copy_bytes=0;
    std::uint64_t shared_device_page_attention_bytes=0;
    std::uint64_t shared_device_page_attention_groups=0,shared_device_page_attention_pages=0;
    // Actual query-pair dispatch attempts, including subsequently failed work.
    // Rows count layer-query work, not distinct request or committed tokens.
    std::uint64_t query_pair_launch_attempts=0,query_pair_row_attempts=0;
    // Opted-in rows reaching dispatch; includes fallback and failed attempts.
    // Difference from selected rows is not a physical-load or latency estimate.
    std::uint64_t query_pair_requested_row_attempts=0;
    // Exact six-head-score prefill route selected at the real layer dispatch.
    // Attempts are layer-subbatch launches/rows, not completed requests.
    std::uint64_t gqa_six_query_pair_score_launch_attempts=0;
    std::uint64_t gqa_six_query_pair_score_row_attempts=0;
    std::uint64_t gqa_six_score_k_tile64_launch_attempts=0;
    std::uint64_t gqa_six_score_k_tile64_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_value_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_value_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_row_attempts=0;
    std::uint64_t gqa_six_softmax_six_values_single_load_launch_attempts=0;
    std::uint64_t gqa_six_softmax_six_values_single_load_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_key_pair_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_key_pair_row_attempts=0;
    // Staging upload bytes have been submitted; consumed bytes completed their
    // D2D final-use event. Banks count completed K+V working-cache installs.
    std::uint64_t attention_stage_upload_bytes=0,attention_stage_consumed_bytes=0,attention_stage_banks=0;
    // Staged history represented directly to attention, without a working-plane copy.
    std::uint64_t attention_stage_direct_bytes=0;
    std::uint64_t attention_stage_device_bytes=0,attention_stage_metadata_bytes=0;
    std::uint64_t reconstruction_device_bytes=0,reconstruction_metadata_bytes=0;
    std::uint64_t reconstruction_budget_fallback_lanes=0;
    // Complete projection submission chains; not GPU completions or committed tokens.
    std::uint64_t reconstruction_submissions=0,reconstruction_submitted_rows=0;
    std::uint64_t reconstruction_k6_submissions=0,reconstruction_k6_submitted_rows=0;
    // Actual target-owned eager projection dispatches. These are invocation
    // counters, not useful tokens, GPU durations, or accepted speculative rows.
    std::uint64_t k6_stream_reduction_calls=0,extended_stream_reduction_calls=0;
    std::uint64_t k6_gateup_warpgroup_async_calls=0;
    std::uint64_t k6_gateup_n32_pair_cta_calls=0;
    std::uint64_t k6_gateup_n32_pair_cta_rows=0;
    std::uint64_t k6_fast_decode_calls=0;
    std::uint64_t k6_fast_decode_rows=0;
    std::uint64_t k6_rowpair_n64_calls=0;
    std::uint64_t k6_rowpair_n64_rows=0;
    std::uint64_t k6_down_rowpair_calls=0;
    std::uint64_t k6_down_rowpair_rows=0;
    std::uint64_t shape4_n64_calls=0;
    std::uint64_t shape4_n64_rows=0;
    std::uint64_t reduce_shfl_min_barrier_calls=0;
    std::uint64_t reduce_shfl_min_barrier_rows=0;
    std::uint64_t k7_tiles64_exact_splits_calls=0;
    std::uint64_t target_k8_kv_prefill_async_a_calls=0;
    std::uint64_t target_k8_kv_prefill_async_a_rows=0;
    std::uint64_t target_down_k6_async_a_calls=0;
    std::uint64_t target_gateup_k6_n16_calls=0;
    // Compute-only exact-HostKV GDN segment graph setup/replay. Capture time is
    // model/context startup work; replays are request deltas.
    std::uint64_t host_kv_gdn_segment_graph_captures=0;
    std::uint64_t host_kv_gdn_segment_graph_replays=0;
    double host_kv_gdn_segment_graph_capture_seconds=0.0;
    std::uint64_t host_kv_full_layer_graph_captures=0;
    std::uint64_t host_kv_full_layer_graph_replays=0;
    std::uint64_t host_kv_full_layer_graph_six_softmax_triple_captures=0;
    std::uint64_t host_kv_full_layer_graph_k6_stream_reduction_captures=0;
    std::uint64_t host_kv_full_layer_graph_extended_stream_reduction_captures=0;
    std::uint64_t host_kv_full_layer_graph_target_down_k6_async_a_captures=0;
    std::uint64_t host_kv_full_layer_graph_target_k6_small_m_async_a_captures=0;
    std::uint64_t host_kv_full_layer_graph_target_k7_small_m_async_a_captures=0;
    double host_kv_full_layer_graph_capture_seconds=0.0;
    std::uint64_t host_kv_mlp_tail_graph_captures=0;
    std::uint64_t host_kv_mlp_tail_graph_replays=0;
    double host_kv_mlp_tail_graph_capture_seconds=0.0;
    std::uint64_t host_kv_transaction_checkpoint_graph_captures=0;
    std::uint64_t host_kv_transaction_checkpoint_graph_replays=0;
    double host_kv_transaction_checkpoint_graph_capture_seconds=0.0;
    std::uint64_t host_kv_transaction_recurrent_trace_alias_layers=0;
    std::uint64_t host_kv_transaction_recurrent_trace_copy_bytes_saved=0;
    std::uint64_t recurrent_export_calls=0;
    std::uint64_t recurrent_export_bytes=0;
    std::uint64_t recurrent_export_copy_submissions=0;
    std::uint64_t recurrent_export_batched_copy_calls=0;
    std::uint64_t recurrent_export_batched_copy_ranges=0;
    double recurrent_export_seconds=0.0;
    double recurrent_export_fence_seconds=0.0;
    std::uint64_t target_k6_small_m_async_a_calls=0;
    std::uint64_t target_k7_small_m_async_a_calls=0;
    std::uint64_t target_k5_small_m_batch_calls=0;
    std::uint64_t eager_mlp_gateup_concurrent_calls=0;
    std::uint64_t prefill_qkv_concurrent_calls=0;
    std::uint64_t prefill_qkv_concurrent_rows=0;
    std::uint64_t prefill_projection_graph_captures=0;
    std::uint64_t prefill_projection_graph_replays=0;
    std::uint64_t prefill_projection_graph_binding_fallbacks=0;
    std::uint64_t prefill_projection_chain_graph_captures=0;
    std::uint64_t prefill_projection_chain_graph_replays=0;
    std::uint64_t prefill_projection_chain_graph_fallbacks=0;
    // Target prefill gate/up shared-A candidate. Attempts count eligible-owner
    // consideration; calls/rows count only the exact common-SUH fused route.
    std::uint64_t target_prefill_gate_up_pair_attempts=0;
    std::uint64_t target_prefill_gate_up_pair_calls=0;
    std::uint64_t target_prefill_gate_up_pair_rows=0;
    std::uint64_t gqa_six_softmax_triple_v_tile_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_v_tile_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_full_cta_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_full_cta_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_threads128_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_threads128_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_score_tile_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_score_tile_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_scalar_dim_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_scalar_dim_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_two_query_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_two_query_row_attempts=0;
    std::uint64_t gqa_six_softmax_triple_fused_launch_attempts=0;
    std::uint64_t gqa_six_softmax_triple_fused_row_attempts=0;
    std::uint64_t gqa_six_softmax_tile512_launch_attempts=0;
    std::uint64_t gqa_six_softmax_tile512_row_attempts=0;
    // Eager host submissions, including requests later cancelled/failed. Excludes
    // graph capture/replay; paired count does not establish equal-scale hits.
    std::uint64_t paired_transform_submissions=0,fused_gate_up_submissions=0;
    // Eager host head submission chains only; excludes graph capture/replay.
    std::uint64_t head_submitted_rows=0,head_omitted_rows=0;
    // Logical decisions under explicitly supplied EXL3 horizon costs. These
    // include attempts later aborted; they are not accepted tokens or GPU work.
    std::uint64_t verifier_cost_menu_installations=0;
    std::uint64_t conditional_second_block_calls=0;
    // Successful packed draft dispatches containing one or two child-root lanes.
    std::uint64_t conditional_shared_draft_batches=0,conditional_shared_draft_lanes=0;
    std::array<std::uint64_t,15> conditional_shared_draft_families{};
    // Index0: ordinary + conditional; index1: both conditional.
    std::array<std::uint64_t,2> conditional_shared_draft_peers{};
    std::uint64_t execution_retirement_metadata_bytes=0;
    // Committed startup transactions only; excludes optional prefix caches,
    // shared weights and subsequent dynamic growth.
    std::uint64_t reserved_context_startup_lanes=0;
    std::uint64_t reserved_context_startup_device_bytes=0;
    std::uint64_t reserved_context_fixed_owner_metadata_bytes=0;
    std::uint64_t reserved_continuation_startup_device_bytes=0;
    std::uint64_t reserved_continuation_owner_metadata_bytes=0;
    std::uint64_t reserved_repair_checkpoint_startup_device_bytes=0;
    std::uint64_t reserved_repair_checkpoint_owner_metadata_bytes=0;
    std::uint64_t repair_checkpoint_reservation_fallback_lanes=0;
    // Subset transferred to continuation linear child lifetime on close.
    std::uint64_t continuation_linear_owner_metadata_bytes=0;
    std::uint64_t continuation_generic_owner_metadata_bytes=0;
    // Subset of context allocation metadata transferred to base linear children.
    std::uint64_t context_base_linear_owner_metadata_bytes=0;
    std::uint64_t context_private_generic_owner_metadata_bytes=0;
    std::uint64_t context_shared_generic_owner_metadata_bytes=0;
    std::uint64_t context_shared_control_metadata_bytes=0;
    std::uint64_t context_layer_linear_owner_metadata_bytes=0;
    std::uint64_t reserved_lane_startup_device_bytes=0;
    std::uint64_t reserved_lane_fixed_owner_metadata_bytes=0;
    std::uint64_t reserved_engine_tap_device_bytes=0;
    std::uint64_t reserved_engine_tap_metadata_bytes=0;
    // Retained host wrappers only; excludes opaque CUDA driver allocations.
    std::uint64_t execution_stream_metadata_bytes=0;
    std::uint64_t cancellation_owner_metadata_bytes=0;
    std::uint64_t target_allocation_owner_metadata_bytes=0;
    std::uint64_t draft_weight_owner_metadata_bytes=0;
    std::uint64_t draft_execution_owner_metadata_bytes=0;
    // Subset of execution metadata transferred into linear child lifetimes.
    std::uint64_t draft_linear_owner_metadata_bytes=0;
    std::uint64_t draft_generic_owner_metadata_bytes=0;
    std::uint64_t context_allocation_owner_metadata_bytes=0;
    // Supplied startup observation minus exact live inventory. The remainder is
    // explicitly driver-unknown; it is not classified as a leak or cache.
    std::uint64_t inventoried_device_bytes=0;
    std::uint64_t observed_device_used_bytes=0;
    std::uint64_t driver_unknown_device_bytes=0;
    bool driver_unknown_device_bytes_available=false;
    // Completed packed dispatches, even if later completion bookkeeping fails.
    // Does not estimate work partially submitted by a throwing dispatcher.
    std::uint64_t shared_projection_completed_dispatches=0,shared_projection_completed_rows=0;
    bool shared_projection_retains_claim_owners=false;
    std::uint64_t shared_cost_policy_updates=0;
    // Successful B8 neural attempts, including later-aborted proposals. Input
    // rows include the seed; suffix counts are predictions, not accepted tokens.
    std::uint64_t draft_neural_input_rows=0,draft_returned_suffix_rows=0,draft_discarded_suffix_rows=0;
    std::uint64_t shared_underfilled_cost_accepts=0,shared_underfilled_cost_refusals=0;
    // Live offers rejected before claim due to differing modality/route contracts.
    // Included in total fallbacks; not a dispatched or completed batch count.
    std::uint64_t shared_projection_contract_refusals=0;
    // Preclaim pair refusals, each a subset of fallbacks, not failed GPU dispatches.
    std::uint64_t shared_projection_geometry_refusals=0,shared_projection_authority_refusals=0;
    std::uint64_t shared_projection_stage_refusals=0;
    // Configured wait budgets at actual rendezvous offers, not elapsed time or GPU work.
    std::uint64_t shared_policy_capped_waits=0,shared_policy_capped_wait_budget_us=0;
    std::uint64_t suffix_selection_installations=0,suffix_selection_neural_fallbacks=0;
    std::uint64_t suffix_proposal_calls=0,suffix_proposal_rows=0,suffix_proposal_misses=0;
    std::uint64_t suffix_published_rounds=0,suffix_accepted_rows=0,suffix_repair_rows=0;
    std::uint64_t suffix_skipped_neural_blocks=0;
    std::array<std::uint64_t,9> supplied_verifier_horizon_decisions{};
    std::uint64_t acquired_payload_preservations=0,acquired_full_resets=0;
    std::uint64_t acquired_attachment_cancellations=0;
    std::uint64_t acquired_draft_ring_preservations=0,acquired_draft_ring_restores=0;
    // FIFO head assignments only. Affinity requires an idle lane with an exact
    // resident-root witness; fallback never waits for a busy local lane.
    std::uint64_t request_affinity_assignments=0,request_affinity_fallbacks=0;
    // Decisions only; supplied copy bytes are hints and are never reported as
    // observed transfer or memory. Fairness overrides restore oldest-first.
    std::uint64_t ready_copy_hint_deferrals=0,ready_copy_hint_unknown_fallbacks=0;
    std::uint64_t ready_copy_hint_fairness_overrides=0;
    // Successful preparation returns, including requests that fail later.
    // Joined fills and cache hits are distinct; these are not generated tokens
    // or counts of numerical submissions by producers cancelled during a fill.
    std::uint64_t prefix_preparation_returns=0,concurrent_prefix_preparation_returns=0;
    std::uint64_t prefix_preparation_cache_hits=0;
    // Immutable host prefix-plan completions accepted before this request's
    // first numerical operation.  Peer overlap counts plans completed while a
    // different lane retained pending numerical ownership, not elapsed time.
    std::uint64_t host_preparation_peer_numeric_overlaps=0;
    // Complete selected nondefault-stream command chains, not individual
    // kernels or elapsed overlap. Failures retain graph owners until a drain.
    std::uint64_t execution_dependency_graph_begins=0;
    std::uint64_t execution_dependency_graph_submissions=0;
    std::uint64_t execution_dependency_graph_completions=0;
    std::uint64_t execution_dependency_graph_cancellations=0;
    std::uint64_t execution_dependency_graph_failures=0;
    // Startup-reserved flight objects/token buffers, already in host inventory.
    std::uint64_t prefix_preparation_metadata_bytes=0;
    std::uint64_t fused_residual_norm_submissions=0;
    // Attribution within existing lane persistent inventory, not added charges.
    std::uint64_t exact_attention_score_bytes=0,numeric_attention_scratch_bytes=0;
    // Source allocation delta against the same uncoalesced layout, not free VRAM.
    std::uint64_t coalesced_attention_layers=0,coalesced_attention_bytes_saved=0;
    std::uint64_t shared_device_page_attention_max_pages=0;
    std::uint64_t shared_device_page_fallbacks=0,shared_device_page_failures=0;
    // Engine attachment decisions before cache lookup/fill; descendant/private
    // pages may legitimately fail acquired-root ancestry and use private storage.
    std::uint64_t shared_device_page_ancestor_accepts=0,shared_device_page_ancestor_fallbacks=0;
    std::uint64_t completed_prefix_admissions=0;
    // Requests newly cancelled by an Engine execution failure. Excludes the
    // failing request and prior cancellation. Active means worker-owned until
    // cleanup removes the request, including completion awaiting that cleanup.
    std::uint64_t failure_cancelled_active_requests=0;
    std::uint64_t failure_cancelled_queued_requests=0;
    std::uint64_t shared_device_page_evictions=0;
    // Registry accounting, distinct from cumulative traffic and CUDA free memory.
    std::uint64_t shared_device_page_allocations=0,shared_device_page_resident_bytes=0;
    std::uint64_t shared_device_page_ready_rows=0,shared_device_page_pending_fill_bytes=0;
    // Source backing reachable from cache keys and tracked retired fill handles;
    // overlaps host roots. Ready views prevent physical page retirement.
    std::uint64_t shared_device_page_source_pages=0,shared_device_page_source_host_bytes=0;
    std::uint64_t shared_device_page_in_flight_fill_bytes=0;
    std::uint64_t shared_device_page_retained_readers=0,shared_device_page_metadata_records=0;
    std::uint64_t shared_device_page_failed_allocations=0,shared_device_page_retiring_allocations=0;
    std::uint64_t private_device_prefix_bytes=0;
};

struct LoadSummary {
    std::string target;
    std::string model_id;
    std::string weights_id;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t model_bytes_read     = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
};

} // namespace voidinfer
