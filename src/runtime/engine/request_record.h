#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/generation/generation_budget.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime {

enum class EngineRequestState : std::uint8_t {
    Waiting,
    Prefill,
    DecodeReady,
    ModelFinished,
};

template <class Package>
struct RequestRecord {
    using Clock          = std::chrono::steady_clock;
    using PreparedPrompt = typename Package::PreparedPrompt;
    using OutputSession  = typename Package::OutputSession;
    using BasePlan       = typename Package::RequestBasePlan;
    using SequenceHandle = typename Package::SequenceHandle;

    RequestRecord(std::uint64_t request_identity, PreparedPrompt input,
                  OutputSession output_session, PromptSummary summary, double frontend_seconds,
                  ResolvedRequestOptions request_options, Clock::time_point limit,
                  Clock::time_point submit_time)
        : id(request_identity), prompt(std::move(input)), output(std::move(output_session)),
          prompt_summary(std::move(summary)), prepare_seconds(frontend_seconds),
          options(std::move(request_options)), deadline(limit), submitted(submit_time) {}

    RequestRecord(const RequestRecord&)            = delete;
    RequestRecord& operator=(const RequestRecord&) = delete;

    [[nodiscard]] bool is_waiting() const noexcept {
        return model_state == EngineRequestState::Waiting;
    }

    [[nodiscard]] bool is_prefilling() const noexcept {
        return model_state == EngineRequestState::Prefill;
    }

    [[nodiscard]] bool is_decode_ready() const noexcept {
        return model_state == EngineRequestState::DecodeReady;
    }

    [[nodiscard]] bool is_model_finished() const noexcept {
        return model_state == EngineRequestState::ModelFinished;
    }

    const std::uint64_t id;
    PreparedPrompt prompt;
    OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    Clock::time_point deadline;
    Clock::time_point submitted;
    std::optional<Clock::time_point> first_token;
    std::optional<GenerationBudget> budget;
    std::optional<BeginSummary> begin;
    std::vector<TokenId> generated;
    std::string content;
    std::string reasoning;
    std::optional<LaneId> lane;
    std::optional<SequenceHandle> sequence;
    std::atomic<bool> cancelled{false};
    EngineRequestState model_state = EngineRequestState::Waiting;

    std::optional<BasePlan> base_plan;
    AdmissionResources admission_resources;
    std::uint64_t remaining_service_work = 0;
    std::uint64_t backfill_epoch         = 0;
    BackfillClass backfill_class         = BackfillClass::None;
    GenerationTimings generation_timings;
    SpeculativeStats speculative_stats;

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<OutputDelta> events;
    GenerationResult result;
    std::exception_ptr error;
    bool response_done     = false;
    bool consumer_released = false;
    bool capacity_released = false;
};

} // namespace ninfer::runtime
