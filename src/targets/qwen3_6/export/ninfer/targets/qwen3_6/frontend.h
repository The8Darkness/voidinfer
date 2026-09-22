#pragma once
#include <optional>

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::size_t kTokenDomain = 248077;

struct FrontendOptions {
    VisionPatchStorage vision_patch_storage = VisionPatchStorage::BFloat16;
    bool vision_enabled                         = true;
    std::uint32_t max_context                   = 2'048;
    std::size_t media_cache_bytes               = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes                = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads      = 0;
    std::uint32_t max_cache_markers_per_request = 4;
};

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

class PublishedOutput {
public:
    [[nodiscard]] std::size_t retained_text_capacity_bytes() const;
    using iterator       = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&);
    PublishedOutput& operator=(const PublishedOutput&);
    PublishedOutput(PublishedOutput&& other) noexcept;
    PublishedOutput& operator=(PublishedOutput&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }

    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }

    [[nodiscard]] iterator end() noexcept { return values_.begin() + size_; }

    [[nodiscard]] const_iterator end() const noexcept { return values_.begin() + size_; }

    [[nodiscard]] OutputDelta& back() noexcept { return values_[size_ - 1]; }

    [[nodiscard]] const OutputDelta& back() const noexcept { return values_[size_ - 1]; }

    void clear() noexcept;
    void push_back(OutputDelta value);
    void append(OutputChannel channel,std::string_view text);
    void preallocate_text(std::size_t per_channel_limit);
    void copy_reusing_storage(const PublishedOutput& source);
    [[nodiscard]] bool has_fixed_storage(std::size_t limit) const noexcept;

private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
    std::optional<std::size_t> text_limit_;
};

class OutputSession {
public:
    // Private retained extents only; shared tokenizer/control-token backing and
    // allocator overhead are excluded. Snapshot may change during preview.
    [[nodiscard]] std::size_t retained_storage_bytes() const;
    // Prepare persistent decoder buffers before generation; emitted output and
    // per-preview scratch remain separate allocations.
    void preallocate_pending_storage();
    void preallocate_output_storage(std::size_t per_channel_limit);
    void exceed_private_storage_for_test(std::size_t reserved_bytes);
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview_model(std::span<const TokenId> tokens,
                                                        std::uint32_t total_budget_remaining,
                                                        FinishReason limit_reason);
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t effective_output_tokens) const;
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview() noexcept;
    void discard_preview() noexcept;
    // True only after committing a model preview terminated by the target's
    // canonical chat-turn delimiter. Caller stops, limits and discarded
    // previews are not completed conversational frontiers.
    [[nodiscard]] bool completed_chat_turn() const noexcept;
    // Return owned committed buffers after copying; only between previews.
    void recycle_output(PublishedOutput output);
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;

private:
    class Impl;
    explicit OutputSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const noexcept;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    // Combined visible reasoning/content bytes, including terminal UTF-8 repair.
    [[nodiscard]] std::size_t output_text_byte_bound(std::uint32_t token_allowance) const;
    // Initial private session acceptance ceiling, excluding later output growth.
    [[nodiscard]] std::size_t output_session_storage_ceiling(const StopPolicy&,std::size_t output_bytes=0) const;
    [[nodiscard]] OutputSession
    make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                        const OutputOptions& output            = {},
                        const ThinkingControlOptions& thinking = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] std::span<const TokenId> thinking_control_tokens() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace ninfer::targets::qwen3_6
