#pragma once

// Adapter between the wire-independent protocol request and the public engine API.

#include "voidinfer/types.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <functional>

namespace voidinfer::serve {

// Media acquisition is a product-layer concern. Translation preserves part order
// and asks the caller to turn each wire source into owning bytes before the
// target frontend sees it.
using MediaAcquirer = std::function<voidinfer::OwnedMedia(const ContentPart&)>;

struct ResolvedPromptSemantics {
    bool enable_thinking = true;
    std::optional<voidinfer::ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
};

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& req,
                                                 const ServeOptions& server,
                                                 const voidinfer::PromptCapabilities& capabilities);

voidinfer::PromptInput to_prompt_input(const GenerationRequest& req,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media);

// Build public request options (output budget, thinking, stop policy, sampler). The
// sampler is resolved from the request's SamplingParams over the server defaults;
// --greedy on the server forces exact argmax regardless of the request. Prefix-reuse
// participation is resolved by GenerationService and supplied explicitly because
// operational requests do not inherit the external-traffic policy.
voidinfer::RequestOptions to_request_options(const GenerationRequest& req, const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse);

// Map an internal finish reason onto the OpenAI wire value. Cancelled maps to
// "stop" (a disconnected client is not an error state on the wire).
const char* finish_reason_wire(voidinfer::FinishReason reason);

} // namespace voidinfer::serve
