#pragma once

#include "core/oscar_mixed_cache_layout.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {

// A page-level identity used by the D2.2c diagnostic. It includes both logical slot identity and
// content hashes, so a successful comparison cannot hide a page/slot remap behind equal bytes.
struct OscarMixedPageFingerprint {
    std::uint32_t model_layer          = 0;
    std::uint32_t page_index           = 0;
    std::uint64_t sequence_id          = 0;
    std::uint32_t logical_token_begin  = 0;
    std::uint32_t logical_token_end    = 0;
    std::uint32_t physical_token_begin = 0;
    std::uint32_t physical_token_end   = 0;
    std::uint16_t occupied_tokens      = 0;
    std::uint16_t capacity_tokens      = 0;
    std::uint16_t layout_version       = 0;
    std::uint16_t group_size           = 0;
    OscarMixedStorageType k_storage    = OscarMixedStorageType::BFloat16;
    OscarMixedStorageType v_storage    = OscarMixedStorageType::BFloat16;
    OscarMixedRegionRole role          = OscarMixedRegionRole::ProtectedPrefix;
    std::uint64_t k_payload_hash       = 0;
    std::uint64_t v_payload_hash       = 0;
    std::uint64_t metadata_hash        = 0;
    std::uint64_t slot_hash            = 0;

    [[nodiscard]] friend bool operator==(const OscarMixedPageFingerprint&,
                                         const OscarMixedPageFingerprint&) = default;
};

struct OscarMixedCacheFingerprint {
    std::uint64_t sequence_id = 0;
    std::uint32_t context_tokens = 0;
    std::uint32_t full_attention_layers = 0;
    std::uint64_t overall_hash = 0;
    std::vector<OscarMixedPageFingerprint> pages;

    [[nodiscard]] friend bool operator==(const OscarMixedCacheFingerprint&,
                                         const OscarMixedCacheFingerprint&) = default;
};

// D2.2c-only transition fixture. It wraps the passing D2.2b bundle and adds branch-local page
// ownership, an immutable checkpoint image, and a deterministic input archive for slow restore.
// It has no attention-facing API and is not a live runtime cache.
class OscarMixedTransitionCache {
public:
    using PageStorage = OscarMixedPageStorage;
    using PageStoragePtr = std::shared_ptr<const PageStorage>;

    OscarMixedTransitionCache(std::uint64_t sequence_id,
                              const OscarMixedAgingAssetContract& asset);

    void append(std::uint32_t logical_token, std::span<const std::uint16_t> k_bf16_by_layer,
                std::span<const std::uint16_t> v_bf16_by_layer);

    // Forking clones logical page metadata and input lineage without re-encoding historical rows.
    // Physical page storage is shared through immutable shared_ptr blocks and is copied on change.
    [[nodiscard]] OscarMixedTransitionCache fork(std::uint64_t child_sequence_id) const;

    // The target must be the unchanged parent lineage of child. Commit adopts the child content
    // atomically and marks the target as committed; it does not append a duplicate token.
    void commit_from(OscarMixedTransitionCache&& child);

    // The state image is a deterministic host snapshot of the branch's original BF16 inputs plus
    // validated asset identity. Restore rebuilds the faithful mixed representation from those
    // inputs; it is intentionally not a Q4 shadow or an attention cache fallback.
    [[nodiscard]] std::vector<std::byte> state_image() const;
    [[nodiscard]] static OscarMixedTransitionCache from_state_image(
        std::span<const std::byte> image, const OscarMixedAgingAssetContract& expected_asset);
    void restore_state_image(std::span<const std::byte> image);

    [[nodiscard]] const OscarMixedAgingCacheBundle& bundle() const noexcept { return bundle_; }
    [[nodiscard]] std::uint64_t sequence_id() const noexcept { return bundle_.sequence_id(); }
    [[nodiscard]] std::uint32_t context_tokens() const noexcept {
        return bundle_.context_tokens();
    }
    [[nodiscard]] const OscarMixedAgingAssetContract& asset() const noexcept {
        return bundle_.asset();
    }
    [[nodiscard]] OscarMixedCacheAccounting accounting() const { return bundle_.accounting(); }
    [[nodiscard]] std::uint32_t aging_conversion_count() const noexcept {
        return bundle_.aging_conversion_count();
    }
    [[nodiscard]] bool committed() const noexcept { return committed_; }

    [[nodiscard]] OscarMixedCacheFingerprint fingerprint(
        bool include_sequence_id = true) const;
    [[nodiscard]] OscarMixedCacheFingerprint content_fingerprint() const {
        return fingerprint(false);
    }
    [[nodiscard]] std::uint64_t state_image_hash() const;
    [[nodiscard]] std::size_t page_count() const noexcept { return page_blocks_.size(); }
    [[nodiscard]] std::size_t shared_page_count_with(
        const OscarMixedTransitionCache& other) const noexcept;
    [[nodiscard]] std::size_t shared_page_refcount(std::size_t page_index) const;
    [[nodiscard]] bool has_logical_token(std::uint32_t logical_token) const noexcept {
        return logical_token < context_tokens();
    }

    // Cross-checks the immutable physical blocks against the typed pages rebuilt by D2.2b.
    void validate() const;

private:
    struct InputToken {
        std::vector<std::uint16_t> k;
        std::vector<std::uint16_t> v;
    };

    OscarMixedAgingCacheBundle bundle_;
    std::vector<InputToken> input_archive_;
    std::vector<PageStoragePtr> page_blocks_;
    std::uint64_t lineage_base_hash_ = 0;
    bool committed_ = false;

    void refresh_page_blocks();
    [[nodiscard]] static bool same_storage(const PageStorage& left,
                                           const PageStorage& right) noexcept;
    [[nodiscard]] OscarMixedCacheFingerprint make_fingerprint(
        bool include_sequence_id) const;
};

} // namespace ninfer
