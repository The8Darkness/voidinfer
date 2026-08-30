#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// The tier names are deliberately physical. L0 is the resident speculative representation, L1
// is the frequent verifier representation, L2 is the authoritative host representation, and L3
// is cold persistence. A cold record must never be pulled into the frequent verifier path.
enum class HierarchicalVeriCacheTier : std::uint8_t {
    L0Vram,
    L1PinnedNvfp4,
    L1PinnedFp8,
    L2HostFp16,
    L2HostGdnProtected,
    L3Nvme,
};

enum class HierarchicalVeriCacheEncoding : std::uint8_t {
    Packed2Bit,
    Packed3Bit,
    Nvfp4,
    Fp8,
    BFloat16,
    Fp16,
};

enum class HierarchicalVeriCacheSensitivity : std::uint8_t {
    Normal,
    Recent,
    AttentionSink,
    Pivot,
    Vision,
    SystemAnchor,
    ToolSchema,
    GdnProtected,
};

struct HierarchicalVeriCacheRange {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
    HierarchicalVeriCacheSensitivity sensitivity = HierarchicalVeriCacheSensitivity::Normal;
};

struct HierarchicalVeriCacheSegment {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
    HierarchicalVeriCacheTier tier = HierarchicalVeriCacheTier::L0Vram;
    HierarchicalVeriCacheEncoding encoding = HierarchicalVeriCacheEncoding::Packed2Bit;
    HierarchicalVeriCacheSensitivity sensitivity = HierarchicalVeriCacheSensitivity::Normal;
    std::size_t bytes = 0;
};

[[nodiscard]] constexpr bool
hierarchical_vericache_is_protected(HierarchicalVeriCacheSensitivity sensitivity) noexcept {
    return sensitivity != HierarchicalVeriCacheSensitivity::Normal;
}

[[nodiscard]] constexpr bool hierarchical_vericache_is_low_bit(
    HierarchicalVeriCacheEncoding encoding) noexcept {
    return encoding == HierarchicalVeriCacheEncoding::Packed2Bit ||
           encoding == HierarchicalVeriCacheEncoding::Packed3Bit;
}

[[nodiscard]] constexpr bool hierarchical_vericache_is_host_tier(
    HierarchicalVeriCacheTier tier) noexcept {
    return tier == HierarchicalVeriCacheTier::L1PinnedNvfp4 ||
           tier == HierarchicalVeriCacheTier::L1PinnedFp8 ||
           tier == HierarchicalVeriCacheTier::L2HostFp16 ||
           tier == HierarchicalVeriCacheTier::L2HostGdnProtected ||
           tier == HierarchicalVeriCacheTier::L3Nvme;
}

[[nodiscard]] inline HierarchicalVeriCacheOptions normalize_hierarchical_vericache_options(
    HierarchicalVeriCacheOptions options) {
    const auto validate_window = [](std::uint32_t min_value, std::uint32_t max_value,
                                    std::uint32_t initial, const char* label) {
        if (min_value == 0 || max_value < min_value) {
            throw std::invalid_argument(std::string(label) + " bounds are invalid");
        }
        return std::clamp(initial, min_value, max_value);
    };
    options.l0_to_l1_horizon = validate_window(options.l0_to_l1_min_horizon,
                                               options.l0_to_l1_max_horizon,
                                               options.l0_to_l1_horizon, "L0-to-L1 horizon");
    options.l1_to_l2_horizon = validate_window(options.l1_to_l2_min_horizon,
                                               options.l1_to_l2_max_horizon,
                                               options.l1_to_l2_horizon, "L1-to-L2 horizon");
    if (options.protected_recent_tokens == 0 && options.protected_sink_tokens == 0 &&
        options.protected_pivot_tokens == 0 && options.enabled) {
        throw std::invalid_argument("hierarchical VeriCache must protect at least one token class");
    }
    return options;
}

// A compact segment ledger used by both the future GPU cache route and host persistence. It
// encodes the safety invariant that sensitive tokens cannot silently be assigned 2/3-bit storage.
class HierarchicalVeriCacheLedger {
public:
    explicit HierarchicalVeriCacheLedger(HierarchicalVeriCacheOptions options = {})
        : options_(normalize_hierarchical_vericache_options(std::move(options))) {}

    void clear() noexcept { segments_.clear(); }

    [[nodiscard]] HierarchicalVeriCacheSensitivity classify(
        std::uint32_t begin, std::uint32_t end, std::uint32_t context_end,
        std::span<const HierarchicalVeriCacheRange> explicit_ranges = {}) const {
        if (end < begin) { throw std::invalid_argument("VeriCache range is inverted"); }
        if (begin == end) { return HierarchicalVeriCacheSensitivity::Normal; }

        HierarchicalVeriCacheSensitivity result = HierarchicalVeriCacheSensitivity::Normal;
        const auto choose = [&result](HierarchicalVeriCacheSensitivity candidate) {
            if (static_cast<std::uint8_t>(candidate) > static_cast<std::uint8_t>(result)) {
                result = candidate;
            }
        };
        if (options_.protected_sink_tokens != 0 &&
            begin < options_.protected_sink_tokens && end != 0) {
            choose(HierarchicalVeriCacheSensitivity::AttentionSink);
        }
        const std::uint32_t recent_begin =
            context_end > options_.protected_recent_tokens
                ? context_end - options_.protected_recent_tokens
                : 0U;
        if (options_.protected_recent_tokens != 0 && end > recent_begin && begin < context_end) {
            choose(HierarchicalVeriCacheSensitivity::Recent);
        }
        for (const HierarchicalVeriCacheRange& range : explicit_ranges) {
            if (range.end < range.begin) {
                throw std::invalid_argument("VeriCache protection range is inverted");
            }
            if (range.begin < end && begin < range.end) { choose(range.sensitivity); }
        }
        return result;
    }

    void append(HierarchicalVeriCacheSegment segment) {
        if (segment.end <= segment.begin) {
            throw std::invalid_argument("VeriCache segment must contain tokens");
        }
        if (hierarchical_vericache_is_protected(segment.sensitivity) &&
            hierarchical_vericache_is_low_bit(segment.encoding)) {
            throw std::invalid_argument("protected VeriCache segment cannot use 2/3-bit encoding");
        }
        if (!segments_.empty() && segment.begin < segments_.back().end) {
            throw std::invalid_argument("VeriCache segments overlap");
        }
        segments_.push_back(segment);
    }

    [[nodiscard]] bool can_reencode(std::size_t index,
                                    HierarchicalVeriCacheEncoding encoding) const {
        if (index >= segments_.size()) { throw std::out_of_range("VeriCache segment index"); }
        return !hierarchical_vericache_is_protected(segments_[index].sensitivity) ||
               !hierarchical_vericache_is_low_bit(encoding);
    }

    void reencode(std::size_t index, HierarchicalVeriCacheEncoding encoding,
                  std::size_t bytes) {
        if (!can_reencode(index, encoding)) {
            throw std::invalid_argument("protected VeriCache segment cannot be reencoded low-bit");
        }
        segments_[index].encoding = encoding;
        segments_[index].bytes    = bytes;
    }

    [[nodiscard]] const std::vector<HierarchicalVeriCacheSegment>& segments() const noexcept {
        return segments_;
    }

    [[nodiscard]] std::size_t bytes(HierarchicalVeriCacheTier tier) const noexcept {
        std::size_t total = 0;
        for (const HierarchicalVeriCacheSegment& segment : segments_) {
            if (segment.tier == tier) { total += segment.bytes; }
        }
        return total;
    }

    [[nodiscard]] std::size_t protected_bytes() const noexcept {
        std::size_t total = 0;
        for (const HierarchicalVeriCacheSegment& segment : segments_) {
            if (hierarchical_vericache_is_protected(segment.sensitivity)) { total += segment.bytes; }
        }
        return total;
    }

private:
    HierarchicalVeriCacheOptions options_;
    std::vector<HierarchicalVeriCacheSegment> segments_;
};

enum class HierarchicalVeriCacheTransactionKind : std::uint8_t {
    L0Speculation,
    L0ToL1Verification,
    L1ToL2Verification,
    GdnRecurrentState,
};

struct HierarchicalVeriCacheFrontier {
    std::uint32_t kv  = 0;
    std::uint32_t gdn = 0;
};

struct HierarchicalVeriCacheTransactionToken {
    std::uint64_t id = 0;
    [[nodiscard]] explicit operator bool() const noexcept { return id != 0; }
};

// Fixed-capacity nested transaction stack. The state image source/destination pair is the
// physical snapshot mechanism; this object records and validates the corresponding KV/GDN
// boundaries so a compressed rejection cannot publish a recurrent-state tail accidentally.
class NestedHierarchicalVeriCacheTransaction {
public:
    static constexpr std::size_t kMaxDepth = 4;

    struct Frame {
        HierarchicalVeriCacheTransactionToken token;
        HierarchicalVeriCacheTransactionKind kind =
            HierarchicalVeriCacheTransactionKind::L0Speculation;
        HierarchicalVeriCacheFrontier base;
        HierarchicalVeriCacheFrontier proposed;
    };

    [[nodiscard]] HierarchicalVeriCacheTransactionToken begin(
        HierarchicalVeriCacheTransactionKind kind, HierarchicalVeriCacheFrontier base,
        HierarchicalVeriCacheFrontier proposed) {
        if (depth_ == kMaxDepth) { throw std::logic_error("VeriCache transaction stack is full"); }
        if (proposed.kv < base.kv || proposed.gdn < base.gdn) {
            throw std::invalid_argument("VeriCache transaction proposal precedes its base");
        }
        if (depth_ != 0 &&
            (base.kv < frames_[depth_ - 1].base.kv || base.gdn < frames_[depth_ - 1].base.gdn)) {
            throw std::invalid_argument("nested VeriCache transaction precedes its parent");
        }
        if (++next_id_ == 0) { ++next_id_; }
        frames_[depth_] = Frame{.token = {next_id_},
                                .kind = kind,
                                .base = base,
                                .proposed = proposed};
        ++depth_;
        max_depth_ = std::max(max_depth_, static_cast<std::uint32_t>(depth_));
        return frames_[depth_ - 1].token;
    }

    [[nodiscard]] bool commit(HierarchicalVeriCacheTransactionToken token,
                              HierarchicalVeriCacheFrontier final_frontier) noexcept {
        if (depth_ == 0 || frames_[depth_ - 1].token.id != token.id) { return false; }
        const Frame frame = frames_[depth_ - 1];
        if (final_frontier.kv < frame.base.kv || final_frontier.gdn < frame.base.gdn ||
            final_frontier.kv > frame.proposed.kv || final_frontier.gdn > frame.proposed.gdn) {
            return false;
        }
        if (final_frontier.kv < frame.proposed.kv || final_frontier.gdn < frame.proposed.gdn) {
            ++partial_rollbacks_;
            rolled_back_tokens_ +=
                static_cast<std::uint64_t>(frame.proposed.kv - final_frontier.kv) +
                static_cast<std::uint64_t>(frame.proposed.gdn - final_frontier.gdn);
        }
        --depth_;
        ++commits_;
        return true;
    }

    [[nodiscard]] bool rollback(HierarchicalVeriCacheTransactionToken token) noexcept {
        if (depth_ == 0 || frames_[depth_ - 1].token.id != token.id) { return false; }
        const Frame frame = frames_[depth_ - 1];
        rolled_back_tokens_ +=
            static_cast<std::uint64_t>(frame.proposed.kv - frame.base.kv) +
            static_cast<std::uint64_t>(frame.proposed.gdn - frame.base.gdn);
        --depth_;
        ++rollbacks_;
        return true;
    }

    void rollback_all() noexcept {
        while (depth_ != 0) {
            const Frame frame = frames_[depth_ - 1];
            rolled_back_tokens_ +=
                static_cast<std::uint64_t>(frame.proposed.kv - frame.base.kv) +
                static_cast<std::uint64_t>(frame.proposed.gdn - frame.base.gdn);
            --depth_;
            ++rollbacks_;
        }
    }

    void reset() noexcept {
        depth_ = 0;
        commits_ = 0;
        rollbacks_ = 0;
        partial_rollbacks_ = 0;
        rolled_back_tokens_ = 0;
        max_depth_ = 0;
    }

    [[nodiscard]] bool active() const noexcept { return depth_ != 0; }
    [[nodiscard]] std::size_t depth() const noexcept { return depth_; }
    [[nodiscard]] HierarchicalVeriCacheTransactionToken top_token() const noexcept {
        return depth_ == 0 ? HierarchicalVeriCacheTransactionToken{} : frames_[depth_ - 1].token;
    }
    [[nodiscard]] const Frame* top_frame() const noexcept {
        return depth_ == 0 ? nullptr : &frames_[depth_ - 1];
    }
    [[nodiscard]] std::uint32_t max_depth() const noexcept { return max_depth_; }
    [[nodiscard]] std::uint64_t commits() const noexcept { return commits_; }
    [[nodiscard]] std::uint64_t rollbacks() const noexcept { return rollbacks_; }
    [[nodiscard]] std::uint64_t partial_rollbacks() const noexcept { return partial_rollbacks_; }
    [[nodiscard]] std::uint64_t rolled_back_tokens() const noexcept { return rolled_back_tokens_; }

private:
    std::array<Frame, kMaxDepth> frames_{};
    std::size_t depth_ = 0;
    std::uint64_t next_id_ = 0;
    std::uint64_t commits_ = 0;
    std::uint64_t rollbacks_ = 0;
    std::uint64_t partial_rollbacks_ = 0;
    std::uint64_t rolled_back_tokens_ = 0;
    std::uint32_t max_depth_ = 0;
};

class AdaptiveHierarchicalVeriCacheController {
public:
    explicit AdaptiveHierarchicalVeriCacheController(HierarchicalVeriCacheOptions options = {})
        : options_(normalize_hierarchical_vericache_options(std::move(options))),
          l0_l1_{options_.l0_to_l1_horizon, options_.l0_to_l1_min_horizon,
                 options_.l0_to_l1_max_horizon},
          l1_l2_{options_.l1_to_l2_horizon, options_.l1_to_l2_min_horizon,
                 options_.l1_to_l2_max_horizon} {}

    [[nodiscard]] bool enabled() const noexcept { return options_.enabled; }
    [[nodiscard]] const HierarchicalVeriCacheOptions& options() const noexcept { return options_; }
    [[nodiscard]] std::uint32_t l0_to_l1_horizon() const noexcept { return l0_l1_.current; }
    [[nodiscard]] std::uint32_t l1_to_l2_horizon() const noexcept { return l1_l2_.current; }

    [[nodiscard]] std::uint32_t next_l0_to_l1_boundary(std::uint32_t frontier) const noexcept {
        return saturating_add(frontier, l0_l1_.current);
    }
    [[nodiscard]] std::uint32_t next_l1_to_l2_boundary(std::uint32_t frontier) const noexcept {
        return saturating_add(frontier, l1_l2_.current);
    }

    void observe_l0_to_l1(std::uint32_t proposed, std::uint32_t accepted, bool disagreement,
                          bool rollback, std::uint64_t transfer_bytes = 0,
                          double transfer_seconds = 0.0) noexcept {
        observe_window(l0_l1_, proposed, accepted, disagreement, rollback);
        ++l0_l1_checks_;
        l0_l1_proposed_ += proposed;
        l0_l1_accepted_ += std::min(proposed, accepted);
        l0_l1_disagreements_ += disagreement ? 1U : 0U;
        l0_l1_transfer_bytes_ += transfer_bytes;
        l0_l1_transfer_seconds_ += transfer_seconds;
    }

    void observe_l1_to_l2(std::uint32_t proposed, std::uint32_t accepted, bool disagreement,
                          bool rollback, std::uint64_t transfer_bytes = 0,
                          double transfer_seconds = 0.0) noexcept {
        observe_window(l1_l2_, proposed, accepted, disagreement, rollback);
        ++l1_l2_checks_;
        l1_l2_proposed_ += proposed;
        l1_l2_accepted_ += std::min(proposed, accepted);
        l1_l2_disagreements_ += disagreement ? 1U : 0U;
        l1_l2_transfer_bytes_ += transfer_bytes;
        l1_l2_transfer_seconds_ += transfer_seconds;
    }

    void record_speculative_round() noexcept { ++speculative_rounds_; }
    void record_speculative_rollback() noexcept { ++speculative_rollbacks_; }
    void record_transactions(std::uint64_t commits, std::uint64_t rollbacks,
                             std::uint32_t max_depth) noexcept {
        nested_commits_ += commits;
        nested_rollbacks_ += rollbacks;
        max_nested_depth_ = std::max(max_nested_depth_, max_depth);
    }
    void record_gdn_restore(std::uint64_t bytes, double seconds) noexcept {
        ++gdn_state_restores_;
        gdn_state_restore_bytes_ += bytes;
        gdn_state_restore_seconds_ += seconds;
    }

    void set_tier_bytes(std::size_t l0, std::size_t l1, std::size_t l2, std::size_t l3) noexcept {
        l0_bytes_ = l0;
        l1_bytes_ = l1;
        l2_bytes_ = l2;
        l3_bytes_ = l3;
    }

    void populate(RuntimeStats& stats) const noexcept {
        stats.hierarchical_vericache_enabled = options_.enabled;
        stats.vericache_l0_to_l1_horizon = l0_l1_.current;
        stats.vericache_l1_to_l2_horizon = l1_l2_.current;
        stats.vericache_l0_l1_checks = l0_l1_checks_;
        stats.vericache_l0_l1_proposed_tokens = l0_l1_proposed_;
        stats.vericache_l0_l1_accepted_tokens = l0_l1_accepted_;
        stats.vericache_l0_l1_disagreements = l0_l1_disagreements_;
        stats.vericache_l1_l2_checks = l1_l2_checks_;
        stats.vericache_l1_l2_proposed_tokens = l1_l2_proposed_;
        stats.vericache_l1_l2_accepted_tokens = l1_l2_accepted_;
        stats.vericache_l1_l2_disagreements = l1_l2_disagreements_;
        stats.vericache_speculative_rounds = speculative_rounds_;
        stats.vericache_speculative_rollbacks = speculative_rollbacks_;
        stats.vericache_nested_commits = nested_commits_;
        stats.vericache_nested_rollbacks = nested_rollbacks_;
        stats.vericache_max_nested_depth = max_nested_depth_;
        stats.vericache_gdn_state_restores = gdn_state_restores_;
        stats.vericache_gdn_state_restore_bytes = gdn_state_restore_bytes_;
        stats.vericache_gdn_state_restore_seconds = gdn_state_restore_seconds_;
        stats.vericache_l0_bytes = l0_bytes_;
        stats.vericache_l1_bytes = l1_bytes_;
        stats.vericache_l2_bytes = l2_bytes_;
        stats.vericache_l3_bytes = l3_bytes_;
    }

private:
    struct Window {
        std::uint32_t current;
        std::uint32_t minimum;
        std::uint32_t maximum;
        std::uint32_t clean_observations = 0;
    };

    static void observe_window(Window& window, std::uint32_t proposed, std::uint32_t accepted,
                               bool disagreement, bool rollback) noexcept {
        const bool poor_acceptance = proposed != 0 && accepted * 2U < proposed;
        if (disagreement || rollback || poor_acceptance) {
            window.clean_observations = 0;
            window.current = std::max(window.minimum, (window.current * 3U) / 4U);
            return;
        }
        if (proposed == 0 || accepted < proposed) {
            window.clean_observations = 0;
            return;
        }
        if (++window.clean_observations >= 4) {
            window.clean_observations = 0;
            const std::uint32_t step = std::max(1U, window.current / 4U);
            window.current = std::min(window.maximum, window.current + step);
        }
    }

    [[nodiscard]] static std::uint32_t saturating_add(std::uint32_t value,
                                                      std::uint32_t increment) noexcept {
        const std::uint64_t sum = static_cast<std::uint64_t>(value) + increment;
        return sum > std::numeric_limits<std::uint32_t>::max()
                   ? std::numeric_limits<std::uint32_t>::max()
                   : static_cast<std::uint32_t>(sum);
    }

    HierarchicalVeriCacheOptions options_;
    Window l0_l1_;
    Window l1_l2_;
    std::uint64_t l0_l1_checks_ = 0;
    std::uint64_t l0_l1_proposed_ = 0;
    std::uint64_t l0_l1_accepted_ = 0;
    std::uint64_t l0_l1_disagreements_ = 0;
    std::uint64_t l0_l1_transfer_bytes_ = 0;
    double l0_l1_transfer_seconds_ = 0.0;
    std::uint64_t l1_l2_checks_ = 0;
    std::uint64_t l1_l2_proposed_ = 0;
    std::uint64_t l1_l2_accepted_ = 0;
    std::uint64_t l1_l2_disagreements_ = 0;
    std::uint64_t l1_l2_transfer_bytes_ = 0;
    double l1_l2_transfer_seconds_ = 0.0;
    std::uint64_t speculative_rounds_ = 0;
    std::uint64_t speculative_rollbacks_ = 0;
    std::uint64_t nested_commits_ = 0;
    std::uint64_t nested_rollbacks_ = 0;
    std::uint32_t max_nested_depth_ = 0;
    std::uint64_t gdn_state_restores_ = 0;
    std::uint64_t gdn_state_restore_bytes_ = 0;
    double gdn_state_restore_seconds_ = 0.0;
    std::size_t l0_bytes_ = 0;
    std::size_t l1_bytes_ = 0;
    std::size_t l2_bytes_ = 0;
    std::size_t l3_bytes_ = 0;
};

struct HierarchicalVeriCacheColdRecord {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t bytes = 0;
    HierarchicalVeriCacheEncoding encoding = HierarchicalVeriCacheEncoding::Fp16;
};

// L3 is metadata-only here. The runtime may schedule a background persistence operation, but no
// frequent verifier is allowed to synchronously resolve a cold record through this object.
class HierarchicalVeriCacheColdManifest {
public:
    explicit HierarchicalVeriCacheColdManifest(std::filesystem::path path = {})
        : path_(std::move(path)) {}

    void append(HierarchicalVeriCacheColdRecord record) {
        if (record.end <= record.begin || record.bytes == 0) {
            throw std::invalid_argument("cold VeriCache record is empty");
        }
        if (!records_.empty() && record.begin < records_.back().end) {
            throw std::invalid_argument("cold VeriCache records overlap");
        }
        records_.push_back(record);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] const std::vector<HierarchicalVeriCacheColdRecord>& records() const noexcept {
        return records_;
    }
    [[nodiscard]] std::uint64_t bytes() const noexcept {
        std::uint64_t total = 0;
        for (const auto& record : records_) { total += record.bytes; }
        return total;
    }
    [[nodiscard]] static constexpr bool allowed_in_frequent_verifier() noexcept { return false; }

private:
    std::filesystem::path path_;
    std::vector<HierarchicalVeriCacheColdRecord> records_;
};

} // namespace ninfer::runtime
