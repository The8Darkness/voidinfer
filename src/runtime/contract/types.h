#pragma once

#include "voidinfer/types.h"

#include <cstdint>

namespace voidinfer::runtime {

using ::voidinfer::FinishReason;
using ::voidinfer::OutputChannel;
using ::voidinfer::ResolvedSamplingParameters;
using ::voidinfer::StopPolicy;
using ::voidinfer::StopString;
using ::voidinfer::TokenId;

// Engine has already selected the model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class ContinuationAction : std::uint8_t {
    Decode,
    ApplyTargetControl,
};

struct OutputDecision {
    std::uint32_t accepted_tokens   = 0;
    FinishReason finish_reason      = FinishReason::None;
    ContinuationAction continuation = ContinuationAction::Decode;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Retention preference of a conversation's prepared prefix in the context cache.
enum class RetentionClass : std::uint8_t {
    SharedStable,
    LiveSession,
    RecentPrivate,
    Disposable,
};

} // namespace voidinfer::runtime
