#pragma once

#include "ops/kv_cache/oscar_int2_g128.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace ninfer {

inline constexpr std::uint32_t kOscarMixedPageTokens     = 64;
inline constexpr std::uint32_t kOscarMixedPrefixTokens   = 64;
inline constexpr std::uint32_t kOscarMixedRecentTokens   = 256;
inline constexpr std::uint32_t kOscarMixedKVHeads        = 4;
inline constexpr std::uint32_t kOscarMixedHeadDim        = 256;
inline constexpr std::uint16_t kOscarMixedLayoutVersion  = 1;
inline constexpr std::uint32_t kOscarMixedTotalLayers    = 64;
inline constexpr std::uint32_t kOscarMixedQueryHeads    = 24;
inline constexpr std::uint32_t kOscarMixedGqaRatio      = 6;
inline constexpr std::uint32_t kOscarMixedRotaryDim     = 64;

inline constexpr char kOscarMixedC4AssetIdentity[] =
    "qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1";
inline constexpr char kOscarMixedC4ModelSha256[] =
    "6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e";
inline constexpr char kOscarMixedC4AssetManifestSha256[] =
    "4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560";
inline constexpr char kOscarMixedC4RotationMode[] = "qqt_sst+r_h_pbr";
inline constexpr char kOscarMixedCal10kAssetIdentity[] =
    "qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal10k-v1";
inline constexpr char kOscarMixedCal10kAssetManifestSha256[] =
    "7426bcd5fd34cd396d5fa2f225d590910e6c213fb3291295e0472478a6f231e9";

inline constexpr std::array<std::uint32_t, 16> kOscarMixedFullAttentionLayers = {
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63};

enum class OscarMixedStorageType : std::uint8_t {
    BFloat16     = 0,
    OscarInt2G128 = 1,
};

enum class OscarMixedRegionRole : std::uint8_t {
    ProtectedPrefix = 0,
    HistoricalBulk  = 1,
    RecentWindow    = 2,
};

[[nodiscard]] constexpr bool is_valid_oscar_mixed_storage_type(
    OscarMixedStorageType type) noexcept {
    return type == OscarMixedStorageType::BFloat16 ||
           type == OscarMixedStorageType::OscarInt2G128;
}

[[nodiscard]] constexpr bool is_valid_oscar_mixed_region_role(
    OscarMixedRegionRole role) noexcept {
    return role == OscarMixedRegionRole::ProtectedPrefix ||
           role == OscarMixedRegionRole::HistoricalBulk ||
           role == OscarMixedRegionRole::RecentWindow;
}

struct OscarMixedPageMetadata {
    std::uint32_t model_layer            = 0;
    std::uint64_t sequence_id            = 0;
    std::uint32_t logical_token_begin    = 0;
    std::uint32_t logical_token_end      = 0;
    std::uint32_t physical_token_begin   = 0;
    std::uint32_t physical_token_end     = 0;
    std::uint32_t physical_page_index    = 0;
    std::uint16_t occupied_tokens        = 0;
    std::uint16_t capacity_tokens        = kOscarMixedPageTokens;
    std::uint16_t layout_version         = kOscarMixedLayoutVersion;
    std::uint16_t group_size             = ninfer::ops::kOscarInt2G128GroupSize;
    OscarMixedStorageType k_storage      = OscarMixedStorageType::BFloat16;
    OscarMixedStorageType v_storage      = OscarMixedStorageType::BFloat16;
    OscarMixedRegionRole role             = OscarMixedRegionRole::ProtectedPrefix;
    std::uint8_t reserved[3]              = {};
};

struct OscarMixedSlotMetadata {
    std::uint32_t model_layer           = 0;
    std::uint64_t sequence_id           = 0;
    std::uint32_t logical_token_begin   = 0;
    std::uint32_t logical_token_end     = 0;
    std::uint32_t physical_token_begin  = 0;
    std::uint32_t physical_token_end    = 0;
    std::uint32_t physical_page_index   = 0;
    std::uint16_t page_offset           = 0;
    std::uint16_t layout_version        = kOscarMixedLayoutVersion;
    std::uint16_t group_size            = ninfer::ops::kOscarInt2G128GroupSize;
    OscarMixedStorageType k_storage     = OscarMixedStorageType::BFloat16;
    OscarMixedStorageType v_storage     = OscarMixedStorageType::BFloat16;
    OscarMixedRegionRole role            = OscarMixedRegionRole::ProtectedPrefix;
    std::uint8_t reserved[3]             = {};
};

// A page owns one physical format. Prefix and recent pages use BF16 code units; historical pages
// use the D2.1 packed payload plus four FP32 values per K/V row. Region-local physical token
// addresses are disambiguated by the explicit role/type metadata and page index.
struct OscarMixedBFloat16PageStorage {
    std::vector<std::uint16_t> k;
    std::vector<std::uint16_t> v;

    OscarMixedBFloat16PageStorage() = default;
    explicit OscarMixedBFloat16PageStorage(std::uint32_t capacity_tokens);

    [[nodiscard]] std::size_t bytes() const noexcept;
};

struct OscarMixedInt2G128PageStorage {
    std::vector<std::uint8_t> k_packed;
    std::vector<std::uint8_t> v_packed;
    std::vector<float> k_scales_zeros;
    std::vector<float> v_scales_zeros;

    OscarMixedInt2G128PageStorage() = default;
    explicit OscarMixedInt2G128PageStorage(std::uint32_t capacity_tokens);

    [[nodiscard]] std::size_t bytes() const noexcept;
};

using OscarMixedPageStorage =
    std::variant<OscarMixedBFloat16PageStorage, OscarMixedInt2G128PageStorage>;

struct OscarMixedPage {
    OscarMixedPageMetadata metadata;
    std::vector<OscarMixedSlotMetadata> slots;
    OscarMixedPageStorage storage;

    [[nodiscard]] std::size_t storage_bytes() const noexcept;
};

struct OscarMixedResolvedSlot {
    std::uint32_t page_index = 0;
    OscarMixedSlotMetadata metadata;
};

struct OscarMixedCacheAccounting {
    std::uint32_t context_tokens        = 0;
    std::uint32_t full_attention_layers = 0;
    std::uint32_t prefix_tokens         = 0;
    std::uint32_t historical_tokens     = 0;
    std::uint32_t recent_tokens         = 0;
    std::uint64_t page_count            = 0;
    std::uint64_t bf16_page_count       = 0;
    std::uint64_t int2_page_count       = 0;
    std::uint64_t logical_bf16_bytes    = 0;
    std::uint64_t logical_int2_payload_bytes  = 0;
    std::uint64_t logical_int2_metadata_bytes = 0;
    std::uint64_t physical_bf16_bytes   = 0;
    std::uint64_t physical_int2_payload_bytes  = 0;
    std::uint64_t physical_int2_metadata_bytes = 0;
    std::uint64_t page_header_bytes     = 0;
    std::uint64_t slot_table_bytes      = 0;
    std::uint64_t mixed_total_bytes     = 0;
    std::uint64_t historical_bulk_total_bytes = 0;
    std::uint64_t logical_value_count   = 0;
    double raw_int2_bytes_per_value     = 0.0;
    double historical_bulk_bytes_per_value = 0.0;
    double mixed_bytes_per_value        = 0.0;
    double historical_bulk_bits_per_value = 0.0;
    double mixed_bits_per_value         = 0.0;
};

class OscarMixedLayerCache {
public:
    OscarMixedLayerCache() = delete;
    OscarMixedLayerCache(std::uint32_t model_layer, std::uint64_t sequence_id,
                         std::uint32_t context_tokens);

    [[nodiscard]] std::uint32_t model_layer() const noexcept { return model_layer_; }
    [[nodiscard]] std::uint64_t sequence_id() const noexcept { return sequence_id_; }
    [[nodiscard]] std::uint32_t context_tokens() const noexcept { return context_tokens_; }
    [[nodiscard]] const std::vector<OscarMixedPage>& pages() const noexcept { return pages_; }

    [[nodiscard]] OscarMixedResolvedSlot resolve(std::uint32_t logical_token) const;
    void validate() const;
    [[nodiscard]] OscarMixedCacheAccounting accounting() const;

private:
    std::uint32_t model_layer_     = 0;
    std::uint64_t sequence_id_     = 0;
    std::uint32_t context_tokens_ = 0;
    std::vector<OscarMixedPage> pages_;
};

class OscarMixedCacheBundle {
public:
    OscarMixedCacheBundle(std::uint64_t sequence_id, std::uint32_t context_tokens,
                          std::span<const std::uint32_t> full_attention_layers);

    [[nodiscard]] std::uint64_t sequence_id() const noexcept { return sequence_id_; }
    [[nodiscard]] std::uint32_t context_tokens() const noexcept { return context_tokens_; }
    [[nodiscard]] const std::vector<OscarMixedLayerCache>& layers() const noexcept {
        return layers_;
    }

    [[nodiscard]] const OscarMixedLayerCache& layer(std::uint32_t model_layer) const;
    void validate() const;
    [[nodiscard]] OscarMixedCacheAccounting accounting() const;

private:
    std::uint64_t sequence_id_     = 0;
    std::uint32_t context_tokens_ = 0;
    std::vector<OscarMixedLayerCache> layers_;
};

// The C4 runtime manifest contract used by the diagnostic aging path. This is an identity and
// topology guard, not a second rotation loader; D2.2b consumes already-rotated K/V rows and does
// not perform attention or rotation fitting.
struct OscarMixedAgingAssetContract {
    std::string asset_identity;
    std::string model_sha256;
    std::string asset_manifest_sha256;
    std::string rotation_mode;
    std::uint32_t total_layers = 0;
    std::uint32_t query_heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t gqa_ratio = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t rotary_dim = 0;
    std::uint32_t group_size = 0;
    bool calibrated = false;
    std::array<std::uint32_t, 16> full_attention_layers{};

    [[nodiscard]] static OscarMixedAgingAssetContract c4_cal30k();
    // Build a validated cache contract from a supported runtime rotation manifest.  This keeps
    // the C4 and already-fitted 10K banks on the same fail-closed topology/codec path.
    [[nodiscard]] static OscarMixedAgingAssetContract from_runtime(
        std::string asset_identity, std::string model_sha256, std::string asset_manifest_sha256,
        std::string rotation_mode);
    void validate() const;
};

// Slow correctness fixture for recent-to-historical aging. Rows supplied to append() are already
// in the selected C4 rotated coordinate system and are represented as BF16 bit units while in the
// prefix/recent tier. A departing row is decoded to FP32, encoded once by the official D2.1
// OscarInt2G128 codec, and rebuilt into a typed historical page. This class deliberately has no
// attention-facing API.
class OscarMixedAgingLayerCache {
public:
    OscarMixedAgingLayerCache() = delete;
    OscarMixedAgingLayerCache(std::uint32_t model_layer, std::uint64_t sequence_id,
                              const OscarMixedAgingAssetContract& asset);

    void append(std::uint32_t logical_token, std::span<const std::uint16_t> k_bf16,
                std::span<const std::uint16_t> v_bf16);
    // Diagnostic guard hook: only the current oldest non-prefix recent token may age. Calling it
    // again for a historical token must throw, proving a logical row cannot be converted twice.
    void age_token(std::uint32_t logical_token);

    [[nodiscard]] std::uint32_t model_layer() const noexcept { return model_layer_; }
    [[nodiscard]] std::uint64_t sequence_id() const noexcept { return sequence_id_; }
    [[nodiscard]] std::uint32_t context_tokens() const noexcept {
        return static_cast<std::uint32_t>(tokens_.size());
    }
    [[nodiscard]] const OscarMixedAgingAssetContract& asset() const noexcept { return asset_; }
    [[nodiscard]] const std::vector<OscarMixedPage>& pages() const noexcept { return pages_; }
    [[nodiscard]] OscarMixedResolvedSlot resolve(std::uint32_t logical_token) const;
    [[nodiscard]] const ops::OscarInt2G128EncodedRow& historical_k(
        std::uint32_t logical_token, std::uint32_t kv_head) const;
    [[nodiscard]] const ops::OscarInt2G128EncodedRow& historical_v(
        std::uint32_t logical_token, std::uint32_t kv_head) const;
    [[nodiscard]] bool has_bf16_payload(std::uint32_t logical_token) const;
    [[nodiscard]] bool was_aged_once(std::uint32_t logical_token) const;
    // Clone the complete logical/physical cache without re-encoding any historical row. The
    // sequence id is branch-local metadata; page payloads and OSCAR metadata are copied exactly.
    [[nodiscard]] OscarMixedAgingLayerCache clone_for_sequence(
        std::uint64_t sequence_id) const;
    [[nodiscard]] std::uint32_t aging_conversion_count() const noexcept {
        return aging_conversion_count_;
    }
    void validate() const;
    [[nodiscard]] OscarMixedCacheAccounting accounting() const;

private:
    struct TokenRecord {
        std::array<std::uint16_t, kOscarMixedKVHeads * kOscarMixedHeadDim> k_bf16{};
        std::array<std::uint16_t, kOscarMixedKVHeads * kOscarMixedHeadDim> v_bf16{};
        std::array<ops::OscarInt2G128EncodedRow, kOscarMixedKVHeads> k_int2{};
        std::array<ops::OscarInt2G128EncodedRow, kOscarMixedKVHeads> v_int2{};
        bool has_bf16 = false;
        bool historical = false;
        bool aged_once = false;
    };

    std::uint32_t model_layer_ = 0;
    std::uint64_t sequence_id_ = 0;
    OscarMixedAgingAssetContract asset_;
    std::vector<TokenRecord> tokens_;
    std::vector<OscarMixedPage> pages_;
    std::uint32_t aging_conversion_count_ = 0;

    void rebuild_pages();
};

// Sixteen-layer forced-input fixture. The flattened row spans are ordered by the exact verified
// full-attention layer list, then KV head, then head dimension.
class OscarMixedAgingCacheBundle {
public:
    OscarMixedAgingCacheBundle(std::uint64_t sequence_id,
                               const OscarMixedAgingAssetContract& asset,
                               std::span<const std::uint32_t> full_attention_layers);

    void append(std::uint32_t logical_token, std::span<const std::uint16_t> k_bf16_by_layer,
                std::span<const std::uint16_t> v_bf16_by_layer);
    [[nodiscard]] std::uint64_t sequence_id() const noexcept { return sequence_id_; }
    [[nodiscard]] std::uint32_t context_tokens() const noexcept {
        return layers_.empty() ? 0U : layers_.front().context_tokens();
    }
    [[nodiscard]] const OscarMixedAgingAssetContract& asset() const noexcept { return asset_; }
    [[nodiscard]] const std::vector<OscarMixedAgingLayerCache>& layers() const noexcept {
        return layers_;
    }
    [[nodiscard]] OscarMixedAgingLayerCache& layer(std::uint32_t model_layer);
    [[nodiscard]] const OscarMixedAgingLayerCache& layer(std::uint32_t model_layer) const;
    // Branch clone used by the D2.2c diagnostic. This preserves encoded rows and conversion
    // counts; it only rewrites sequence identity in page/slot metadata.
    [[nodiscard]] OscarMixedAgingCacheBundle clone_for_sequence(
        std::uint64_t sequence_id) const;
    [[nodiscard]] std::uint32_t aging_conversion_count() const noexcept;
    void validate() const;
    [[nodiscard]] OscarMixedCacheAccounting accounting() const;

private:
    std::uint64_t sequence_id_ = 0;
    OscarMixedAgingAssetContract asset_;
    std::vector<OscarMixedAgingLayerCache> layers_;
};

} // namespace ninfer
