#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void swap(ResidentPrefixIdentity& other) noexcept;
    void append_generated(std::size_t count, std::int32_t rope_delta);
    std::uint64_t copied_append_allocation_bytes(std::size_t count,std::int32_t rope_delta) const;
    ResidentPrefixIdentity appended_copy(std::size_t count,std::int32_t rope_delta) const;
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;
    [[nodiscard]] bool equals(const ResidentPrefixIdentity& other) const;
    [[nodiscard]] bool prefix_equals(const ResidentPrefixIdentity& other, std::size_t count) const;
    // Retained metadata participates in host residency and physical reservations.
    void visit_allocations(const std::function<void(const void*, std::size_t)>& visitor,
                           bool include_unused_capacity = true) const;

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
    std::vector<std::uint32_t> rewrite_execution_frontiers_;
};

// One rolling digest per token frontier. This is only a content shortlist: exact token and
// ResidentPrefixIdentity comparison remains authoritative for reuse. Keeping it separate from the
// exact identity avoids retaining hash-only state in immutable capture backings.
class PrefixShortlistDigests {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void swap(PrefixShortlistDigests& other) noexcept;
    void append_generated(std::span<const TokenId> tokens, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept {
        return digests_.empty() ? 0 : digests_.size() - 1U;
    }

    [[nodiscard]] std::uint64_t at(std::size_t frontier) const;

private:
    std::vector<std::uint64_t> digests_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  std::span<const TokenId> resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count);

} // namespace ninfer::targets::qwen3_6::detail
