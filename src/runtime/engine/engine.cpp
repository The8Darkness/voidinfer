#include <cstdlib>
#include <string_view>
#include "voidinfer/engine.h"

#include "core/device.h"
#include "exl3/exl3_engine_core.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"

#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace voidinfer {
namespace {

EngineOptions normalize_engine_options(EngineOptions options) {
    if(options.model.target_directory.empty() || options.model.draft_directory.empty())
        throw std::invalid_argument("Engine requires target and draft model directories");
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    // L0 OSCAR (INT4 history codes on the device, FP16 history in host memory) admits 262144.
    const char* l0_oscar=std::getenv("VOIDINFER_EXL3_L0_OSCAR");
    const std::uint32_t context_limit=!l0_oscar || std::string_view(l0_oscar)!="0"?262144:32768;
    if(options.max_context<2048 || options.max_context>context_limit)
        throw std::invalid_argument("Engine max_context must be in [2048,262144] (32768 without L0 OSCAR)");

    ContextCacheOptions& cache      = options.context_cache;
    const std::uint32_t concurrency = options.max_concurrency;
    // Terminal roots publish through the host-root cache: it is always enabled.
    if (!cache.enabled) {
        throw std::invalid_argument("Engine requires the host-root context cache");
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes               = cache.max_shared_prefixes.value_or(concurrency);
    cache.max_long_anchors_per_continuation = cache.max_long_anchors_per_continuation.value_or(2U);
    cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    constexpr std::uint64_t exact_kv_bytes_per_token=65536;
    if(cache.host_state_slots==0 || *cache.max_shared_prefixes==0 ||
       cache.host_kv_capacity_bytes<
           static_cast<std::uint64_t>(options.max_context)*exact_kv_bytes_per_token)
        throw std::invalid_argument(
            "Engine requires a host root and host KV capacity for one terminal exact snapshot "
            "(max_context x 64 KiB)");
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.thinking                = options.execution.thinking;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         targets::qwen3_6::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3_6::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationPollResult poll(OutputSink* sink,
                                           const CancellationView& cancellation) = 0;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }
        GenerationPollResult poll(OutputSink* sink,
                                  const CancellationView& cancellation) override {
            return submission_.poll(sink,cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }
    GenerationPollResult poll(OutputSink* sink,const CancellationView& cancellation) {
        return state_->poll(sink,cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

GenerationPollResult GenerationHandle::poll(OutputSink* sink,
                                             const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    if(poll_active_)throw std::logic_error("GenerationHandle poll is not reentrant");
    struct Guard {bool& active;~Guard(){active=false;}} guard{poll_active_};
    poll_active_=true;
    auto result=impl_->poll(sink,cancellation);
    if(result.terminal())impl_.reset();
    return result;
}

class Engine::Impl {
public:
    using Core = exl3::Exl3EngineCore;
    static EngineOptions checked_options(EngineOptions supplied) {
        auto normalized=normalize_engine_options(std::move(supplied));
        Core::require_startup_retirement_available(normalized);
        return normalized;
    }

    explicit Impl(EngineOptions engine_options)
        : options(checked_options(std::move(engine_options))),core(std::make_unique<Core>(options)) {
        load=core->load_summary();
        sampling_defaults=core->sampling_defaults();
    }

    EngineOptions options;
    std::unique_ptr<Core> core;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    template<class Function> decltype(auto) with_frontend(Function&& function) const {
        return function(core->frontend());
    }
};

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return impl_->with_frontend(
        [&](const auto& frontend) -> PreparedPrompt {
            auto prepared      = frontend.prepare(std::move(input), control);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > impl_->options.max_context) {
                throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, sampling_mode, std::move(prepared)));
        });
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->with_frontend(
        [&](const auto& frontend) -> PreparedPrompt {
            if (token_ids.size() > impl_->options.max_context) {
                throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                   context_capacity_error(token_ids.size(), impl_->options.max_context));
            }
            auto prepared      = frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > impl_->options.max_context) {
                throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::Thinking, std::move(prepared)));
        });
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->with_frontend([&](const auto& frontend) {
        return frontend.count_tokens(std::move(input),control);
    });
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->with_frontend([](const auto& frontend) {return frontend.prompt_capabilities();});
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
            GenerationPollResult poll(OutputSink* sink,const CancellationView& cancellation) {
                try {
                    const bool streaming=consumer_mode==OutputConsumerMode::Streaming;
                    if(streaming!=(sink!=nullptr))throw std::invalid_argument(
                        "GenerationHandle poll sink does not match its submitted consumer mode");
                    if(cancellation.requested())result.finish_reason=FinishReason::Cancelled;
                    GenerationPollResult completed;completed.state=GenerationPollState::Completed;
                    completed.result.emplace(std::move(result));return completed;
                } catch(...) {
                    return {GenerationPollState::Error,{},std::current_exception()};
                }
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    auto submission=impl_->core->submit(std::move(prompt.impl_->value),prompt_summary,prepare_seconds,
        std::move(resolved_options),consumer_mode,pending_deadline);
    return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
        impl_,std::move(submission),resolved_sampling));
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode).wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->core->memory_summary();
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->with_frontend([](const auto& frontend) {return frontend.media_cache_summary();});
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->core->runtime_stats();
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ != nullptr) impl_->core->reset_memory_peaks();
}

} // namespace voidinfer
