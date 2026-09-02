#include "core/oscar_mixed_cache_layout.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer {
namespace {

constexpr std::uint64_t kBFloat16ValuesPerToken =
    static_cast<std::uint64_t>(kOscarMixedKVHeads) * kOscarMixedHeadDim * 2U;
constexpr std::uint64_t kInt2PayloadBytesPerToken =
    static_cast<std::uint64_t>(kOscarMixedKVHeads) * ninfer::ops::kOscarInt2G128CodeBytes * 2U;
constexpr std::uint64_t kInt2MetadataBytesPerToken =
    static_cast<std::uint64_t>(kOscarMixedKVHeads) *
    ninfer::ops::kOscarInt2G128MetadataItems * sizeof(float) * 2U;

bool is_full_attention_layer(std::uint32_t layer) {
    return std::find(kOscarMixedFullAttentionLayers.begin(),
                     kOscarMixedFullAttentionLayers.end(), layer) !=
           kOscarMixedFullAttentionLayers.end();
}

struct RegionRange {
    OscarMixedRegionRole role;
    OscarMixedStorageType storage;
    std::uint32_t logical_begin;
    std::uint32_t logical_end;
};

std::array<RegionRange, 3> ranges_for(std::uint32_t context_tokens) {
    const std::uint32_t prefix_end = std::min(context_tokens, kOscarMixedPrefixTokens);
    const std::uint32_t recent_begin =
        context_tokens <= kOscarMixedPrefixTokens
            ? context_tokens
            : std::max(kOscarMixedPrefixTokens,
                       context_tokens > kOscarMixedRecentTokens
                           ? context_tokens - kOscarMixedRecentTokens
                           : 0U);
    return {{{OscarMixedRegionRole::ProtectedPrefix, OscarMixedStorageType::BFloat16, 0,
              prefix_end},
             {OscarMixedRegionRole::HistoricalBulk, OscarMixedStorageType::OscarInt2G128,
              prefix_end, recent_begin},
             {OscarMixedRegionRole::RecentWindow, OscarMixedStorageType::BFloat16, recent_begin,
              context_tokens}}};
}

void validate_range(const RegionRange& range, std::uint32_t context_tokens) {
    if (!is_valid_oscar_mixed_region_role(range.role) ||
        !is_valid_oscar_mixed_storage_type(range.storage) ||
        range.logical_begin > range.logical_end || range.logical_end > context_tokens) {
        throw std::invalid_argument("invalid OSCAR mixed-cache region range");
    }
    const bool expected_bf16 = range.role != OscarMixedRegionRole::HistoricalBulk;
    if ((range.storage == OscarMixedStorageType::BFloat16) != expected_bf16) {
        throw std::invalid_argument("OSCAR mixed-cache role/storage mismatch");
    }
}

OscarMixedPage make_page(std::uint32_t model_layer, std::uint64_t sequence_id,
                         std::uint32_t logical_begin, std::uint32_t logical_end,
                         std::uint32_t physical_page_index, std::uint32_t pool_page_index,
                         OscarMixedRegionRole role, OscarMixedStorageType storage) {
    if (logical_begin >= logical_end || logical_end - logical_begin > kOscarMixedPageTokens) {
        throw std::invalid_argument("invalid OSCAR mixed-cache page range");
    }
    const std::uint32_t occupied = logical_end - logical_begin;
    const std::uint32_t physical_begin = pool_page_index * kOscarMixedPageTokens;
    OscarMixedPage page;
    page.metadata = OscarMixedPageMetadata{
        .model_layer          = model_layer,
        .sequence_id          = sequence_id,
        .logical_token_begin  = logical_begin,
        .logical_token_end    = logical_end,
        .physical_token_begin = physical_begin,
        .physical_token_end   = physical_begin + occupied,
        .physical_page_index  = physical_page_index,
        .occupied_tokens      = static_cast<std::uint16_t>(occupied),
        .capacity_tokens      = static_cast<std::uint16_t>(kOscarMixedPageTokens),
        .layout_version       = kOscarMixedLayoutVersion,
        .group_size           = ninfer::ops::kOscarInt2G128GroupSize,
        .k_storage            = storage,
        .v_storage            = storage,
        .role                 = role,
    };
    page.slots.reserve(occupied);
    for (std::uint32_t offset = 0; offset < occupied; ++offset) {
        const std::uint32_t physical_token = physical_begin + offset;
        page.slots.push_back(OscarMixedSlotMetadata{
            .model_layer          = model_layer,
            .sequence_id          = sequence_id,
            .logical_token_begin  = logical_begin + offset,
            .logical_token_end    = logical_begin + offset + 1,
            .physical_token_begin = physical_token,
            .physical_token_end   = physical_token + 1,
            .physical_page_index  = physical_page_index,
            .page_offset          = static_cast<std::uint16_t>(offset),
            .layout_version       = kOscarMixedLayoutVersion,
            .group_size           = ninfer::ops::kOscarInt2G128GroupSize,
            .k_storage            = storage,
            .v_storage            = storage,
            .role                 = role,
        });
    }
    if (storage == OscarMixedStorageType::BFloat16) {
        page.storage = OscarMixedBFloat16PageStorage(kOscarMixedPageTokens);
    } else {
        page.storage = OscarMixedInt2G128PageStorage(kOscarMixedPageTokens);
    }
    return page;
}

template <typename T>
std::size_t vector_bytes(const std::vector<T>& values) noexcept {
    return values.size() * sizeof(T);
}

float bf16_bits_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

bool same_page_metadata(const OscarMixedPageMetadata& left,
                        const OscarMixedPageMetadata& right) noexcept {
    return left.model_layer == right.model_layer && left.sequence_id == right.sequence_id &&
           left.logical_token_begin == right.logical_token_begin &&
           left.logical_token_end == right.logical_token_end &&
           left.physical_token_begin == right.physical_token_begin &&
           left.physical_token_end == right.physical_token_end &&
           left.physical_page_index == right.physical_page_index &&
           left.occupied_tokens == right.occupied_tokens &&
           left.capacity_tokens == right.capacity_tokens &&
           left.layout_version == right.layout_version && left.group_size == right.group_size &&
           left.k_storage == right.k_storage && left.v_storage == right.v_storage &&
           left.role == right.role;
}

bool same_slot_metadata(const OscarMixedSlotMetadata& left,
                        const OscarMixedSlotMetadata& right) noexcept {
    return left.model_layer == right.model_layer && left.sequence_id == right.sequence_id &&
           left.logical_token_begin == right.logical_token_begin &&
           left.logical_token_end == right.logical_token_end &&
           left.physical_token_begin == right.physical_token_begin &&
           left.physical_token_end == right.physical_token_end &&
           left.physical_page_index == right.physical_page_index &&
           left.page_offset == right.page_offset && left.layout_version == right.layout_version &&
           left.group_size == right.group_size && left.k_storage == right.k_storage &&
           left.v_storage == right.v_storage && left.role == right.role;
}

bool same_asset_contract(const OscarMixedAgingAssetContract& left,
                         const OscarMixedAgingAssetContract& right) noexcept {
    return left.asset_identity == right.asset_identity && left.model_sha256 == right.model_sha256 &&
           left.asset_manifest_sha256 == right.asset_manifest_sha256 &&
           left.rotation_mode == right.rotation_mode && left.total_layers == right.total_layers &&
           left.query_heads == right.query_heads && left.kv_heads == right.kv_heads &&
           left.gqa_ratio == right.gqa_ratio && left.head_dim == right.head_dim &&
           left.rotary_dim == right.rotary_dim && left.group_size == right.group_size &&
           left.calibrated == right.calibrated &&
           left.full_attention_layers == right.full_attention_layers;
}

OscarMixedCacheAccounting accounting_for_pages(
    std::uint32_t context_tokens, std::size_t full_attention_layers,
    const std::vector<OscarMixedPage>& pages) {
    OscarMixedCacheAccounting result;
    result.context_tokens = context_tokens;
    result.full_attention_layers = static_cast<std::uint32_t>(full_attention_layers);
    const auto ranges = ranges_for(context_tokens);
    for (const auto& range : ranges) {
        if (range.role == OscarMixedRegionRole::ProtectedPrefix) {
            result.prefix_tokens = range.logical_end - range.logical_begin;
        } else if (range.role == OscarMixedRegionRole::HistoricalBulk) {
            result.historical_tokens = range.logical_end - range.logical_begin;
        } else {
            result.recent_tokens = range.logical_end - range.logical_begin;
        }
    }
    for (const OscarMixedPage& page : pages) {
        ++result.page_count;
        result.page_header_bytes += sizeof(OscarMixedPageMetadata);
        result.slot_table_bytes += page.slots.size() * sizeof(OscarMixedSlotMetadata);
        if (page.metadata.k_storage == OscarMixedStorageType::BFloat16) {
            ++result.bf16_page_count;
            result.physical_bf16_bytes += page.storage_bytes();
        } else {
            ++result.int2_page_count;
            const auto& storage = std::get<OscarMixedInt2G128PageStorage>(page.storage);
            result.physical_int2_payload_bytes += vector_bytes(storage.k_packed) +
                                                  vector_bytes(storage.v_packed);
            result.physical_int2_metadata_bytes += vector_bytes(storage.k_scales_zeros) +
                                                   vector_bytes(storage.v_scales_zeros);
        }
    }
    const std::uint64_t layer_multiplier = static_cast<std::uint64_t>(full_attention_layers);
    result.page_count *= layer_multiplier;
    result.bf16_page_count *= layer_multiplier;
    result.int2_page_count *= layer_multiplier;
    result.physical_bf16_bytes *= layer_multiplier;
    result.physical_int2_payload_bytes *= layer_multiplier;
    result.physical_int2_metadata_bytes *= layer_multiplier;
    result.page_header_bytes *= layer_multiplier;
    result.slot_table_bytes *= layer_multiplier;
    result.logical_bf16_bytes =
        static_cast<std::uint64_t>(result.prefix_tokens + result.recent_tokens) *
        kBFloat16ValuesPerToken * sizeof(std::uint16_t) * full_attention_layers;
    result.logical_int2_payload_bytes =
        static_cast<std::uint64_t>(result.historical_tokens) * kInt2PayloadBytesPerToken *
        full_attention_layers;
    result.logical_int2_metadata_bytes =
        static_cast<std::uint64_t>(result.historical_tokens) * kInt2MetadataBytesPerToken *
        full_attention_layers;
    result.mixed_total_bytes = result.physical_bf16_bytes + result.physical_int2_payload_bytes +
                               result.physical_int2_metadata_bytes + result.page_header_bytes +
                               result.slot_table_bytes;
    result.historical_bulk_total_bytes =
        result.physical_int2_payload_bytes + result.physical_int2_metadata_bytes +
        result.int2_page_count * sizeof(OscarMixedPageMetadata) +
        static_cast<std::uint64_t>(result.historical_tokens) * sizeof(OscarMixedSlotMetadata) *
            full_attention_layers;
    result.logical_value_count = static_cast<std::uint64_t>(context_tokens) *
                                 kBFloat16ValuesPerToken * full_attention_layers;
    result.raw_int2_bytes_per_value =
        static_cast<double>(ninfer::ops::kOscarInt2G128CodeBytes +
                            ninfer::ops::kOscarInt2G128MetadataItems * sizeof(float)) /
        ninfer::ops::kOscarInt2G128HeadDim;
    if (result.historical_tokens != 0) {
        const auto count = static_cast<double>(result.historical_tokens) *
                           kBFloat16ValuesPerToken * full_attention_layers;
        result.historical_bulk_bytes_per_value =
            static_cast<double>(result.historical_bulk_total_bytes) / count;
        result.historical_bulk_bits_per_value = result.historical_bulk_bytes_per_value * 8.0;
    }
    if (result.logical_value_count != 0) {
        result.mixed_bytes_per_value = static_cast<double>(result.mixed_total_bytes) /
                                       static_cast<double>(result.logical_value_count);
        result.mixed_bits_per_value = result.mixed_bytes_per_value * 8.0;
    }
    return result;
}

} // namespace

OscarMixedBFloat16PageStorage::OscarMixedBFloat16PageStorage(std::uint32_t capacity_tokens)
    : k(static_cast<std::size_t>(capacity_tokens) * kOscarMixedKVHeads * kOscarMixedHeadDim),
      v(static_cast<std::size_t>(capacity_tokens) * kOscarMixedKVHeads * kOscarMixedHeadDim) {}

std::size_t OscarMixedBFloat16PageStorage::bytes() const noexcept {
    return vector_bytes(k) + vector_bytes(v);
}

OscarMixedInt2G128PageStorage::OscarMixedInt2G128PageStorage(std::uint32_t capacity_tokens)
    : k_packed(static_cast<std::size_t>(capacity_tokens) * kOscarMixedKVHeads *
               ninfer::ops::kOscarInt2G128CodeBytes),
      v_packed(static_cast<std::size_t>(capacity_tokens) * kOscarMixedKVHeads *
               ninfer::ops::kOscarInt2G128CodeBytes),
      k_scales_zeros(static_cast<std::size_t>(capacity_tokens) * kOscarMixedKVHeads *
                     ninfer::ops::kOscarInt2G128MetadataItems),
      v_scales_zeros(static_cast<std::size_t>(capacity_tokens) * kOscarMixedKVHeads *
                     ninfer::ops::kOscarInt2G128MetadataItems) {}

std::size_t OscarMixedInt2G128PageStorage::bytes() const noexcept {
    return vector_bytes(k_packed) + vector_bytes(v_packed) + vector_bytes(k_scales_zeros) +
           vector_bytes(v_scales_zeros);
}

std::size_t OscarMixedPage::storage_bytes() const noexcept {
    return std::visit([](const auto& value) { return value.bytes(); }, storage);
}

OscarMixedLayerCache::OscarMixedLayerCache(std::uint32_t model_layer,
                                           std::uint64_t sequence_id,
                                           std::uint32_t context_tokens)
    : model_layer_(model_layer), sequence_id_(sequence_id), context_tokens_(context_tokens) {
    if (!is_full_attention_layer(model_layer_)) {
        throw std::invalid_argument("OSCAR mixed cache accepts full-attention layers only");
    }
    if (context_tokens_ == 0) {
        throw std::invalid_argument("OSCAR mixed cache context must be nonzero");
    }

    const auto ranges = ranges_for(context_tokens_);
    std::uint32_t next_physical_page = 0;
    for (const RegionRange& range : ranges) {
        validate_range(range, context_tokens_);
        const std::uint32_t token_count = range.logical_end - range.logical_begin;
        const std::uint32_t page_count =
            (token_count + kOscarMixedPageTokens - 1U) / kOscarMixedPageTokens;
        for (std::uint32_t page = 0; page < page_count; ++page) {
            const std::uint32_t logical_begin = range.logical_begin + page * kOscarMixedPageTokens;
            const std::uint32_t logical_end =
                std::min(range.logical_end, logical_begin + kOscarMixedPageTokens);
            pages_.push_back(make_page(model_layer_, sequence_id_, logical_begin, logical_end,
                                       next_physical_page++, page, range.role, range.storage));
        }
    }
    validate();
}

OscarMixedResolvedSlot OscarMixedLayerCache::resolve(std::uint32_t logical_token) const {
    if (logical_token >= context_tokens_) {
        throw std::out_of_range("OSCAR mixed-cache logical token is out of range");
    }
    for (std::uint32_t page_index = 0; page_index < pages_.size(); ++page_index) {
        const auto& page = pages_[page_index];
        if (logical_token >= page.metadata.logical_token_begin &&
            logical_token < page.metadata.logical_token_end) {
            const std::size_t offset = logical_token - page.metadata.logical_token_begin;
            if (offset >= page.slots.size()) { std::terminate(); }
            return {page_index, page.slots[offset]};
        }
    }
    throw std::logic_error("OSCAR mixed-cache logical token has no physical slot");
}

void OscarMixedLayerCache::validate() const {
    if (!is_full_attention_layer(model_layer_) || context_tokens_ == 0 || pages_.empty()) {
        throw std::invalid_argument("invalid OSCAR mixed-cache layer identity");
    }
    std::uint32_t logical_cursor = 0;
    for (std::uint32_t page_index = 0; page_index < pages_.size(); ++page_index) {
        const OscarMixedPage& page = pages_[page_index];
        const auto& meta = page.metadata;
        if (meta.model_layer != model_layer_ || meta.sequence_id != sequence_id_ ||
            meta.layout_version != kOscarMixedLayoutVersion ||
            meta.group_size != ninfer::ops::kOscarInt2G128GroupSize ||
            meta.k_storage != meta.v_storage || !is_valid_oscar_mixed_storage_type(meta.k_storage) ||
            !is_valid_oscar_mixed_region_role(meta.role) ||
            meta.capacity_tokens != kOscarMixedPageTokens ||
            meta.logical_token_begin != logical_cursor ||
            meta.logical_token_begin >= meta.logical_token_end ||
            meta.logical_token_end > context_tokens_ ||
            meta.occupied_tokens != meta.logical_token_end - meta.logical_token_begin ||
            meta.occupied_tokens > meta.capacity_tokens ||
            meta.physical_token_end < meta.physical_token_begin ||
            meta.physical_token_end - meta.physical_token_begin != meta.occupied_tokens ||
            meta.physical_page_index != page_index || page.slots.size() != meta.occupied_tokens) {
            throw std::invalid_argument("OSCAR mixed-cache page metadata is inconsistent");
        }
        const bool expected_bf16 = meta.role != OscarMixedRegionRole::HistoricalBulk;
        if ((meta.k_storage == OscarMixedStorageType::BFloat16) != expected_bf16) {
            throw std::invalid_argument("OSCAR mixed-cache page role is not typed correctly");
        }
        if (expected_bf16 != std::holds_alternative<OscarMixedBFloat16PageStorage>(page.storage)) {
            throw std::invalid_argument("OSCAR mixed-cache physical storage variant mismatch");
        }
        if (!expected_bf16 &&
            !std::holds_alternative<OscarMixedInt2G128PageStorage>(page.storage)) {
            throw std::invalid_argument("OSCAR mixed-cache INT2 storage variant mismatch");
        }
        for (std::uint32_t offset = 0; offset < page.slots.size(); ++offset) {
            const auto& slot = page.slots[offset];
            if (slot.model_layer != model_layer_ || slot.sequence_id != sequence_id_ ||
                slot.logical_token_begin != meta.logical_token_begin + offset ||
                slot.logical_token_end != slot.logical_token_begin + 1 ||
                slot.physical_token_begin != meta.physical_token_begin + offset ||
                slot.physical_token_end != slot.physical_token_begin + 1 ||
                slot.physical_page_index != page_index || slot.page_offset != offset ||
                slot.layout_version != meta.layout_version || slot.group_size != meta.group_size ||
                slot.k_storage != meta.k_storage || slot.v_storage != meta.v_storage ||
                slot.role != meta.role) {
                throw std::invalid_argument("OSCAR mixed-cache slot metadata is inconsistent");
            }
        }
        logical_cursor = meta.logical_token_end;
    }
    if (logical_cursor != context_tokens_) {
        throw std::invalid_argument("OSCAR mixed-cache has a logical hole or overlap");
    }
    const auto ranges = ranges_for(context_tokens_);
    for (const auto& range : ranges) { validate_range(range, context_tokens_); }
    for (const OscarMixedPage& page : pages_) {
        const auto it = std::find_if(ranges.begin(), ranges.end(), [&](const RegionRange& range) {
            return page.metadata.logical_token_begin >= range.logical_begin &&
                   page.metadata.logical_token_end <= range.logical_end &&
                   page.metadata.role == range.role;
        });
        if (it == ranges.end() || page.metadata.k_storage != it->storage) {
            throw std::invalid_argument("OSCAR mixed-cache page crosses or changes a region");
        }
    }
}

OscarMixedCacheAccounting OscarMixedLayerCache::accounting() const {
    validate();
    OscarMixedCacheAccounting result;
    result.context_tokens = context_tokens_;
    result.full_attention_layers = 1;
    const auto ranges = ranges_for(context_tokens_);
    for (const auto& range : ranges) {
        if (range.role == OscarMixedRegionRole::ProtectedPrefix) {
            result.prefix_tokens = range.logical_end - range.logical_begin;
        } else if (range.role == OscarMixedRegionRole::HistoricalBulk) {
            result.historical_tokens = range.logical_end - range.logical_begin;
        } else {
            result.recent_tokens = range.logical_end - range.logical_begin;
        }
    }
    for (const OscarMixedPage& page : pages_) {
        ++result.page_count;
        result.page_header_bytes += sizeof(OscarMixedPageMetadata);
        result.slot_table_bytes += page.slots.size() * sizeof(OscarMixedSlotMetadata);
        if (page.metadata.k_storage == OscarMixedStorageType::BFloat16) {
            ++result.bf16_page_count;
            result.physical_bf16_bytes += page.storage_bytes();
        } else {
            ++result.int2_page_count;
            const auto& storage = std::get<OscarMixedInt2G128PageStorage>(page.storage);
            result.physical_int2_payload_bytes += vector_bytes(storage.k_packed) +
                                                  vector_bytes(storage.v_packed);
            result.physical_int2_metadata_bytes += vector_bytes(storage.k_scales_zeros) +
                                                   vector_bytes(storage.v_scales_zeros);
        }
    }
    result.logical_bf16_bytes =
        static_cast<std::uint64_t>(result.prefix_tokens + result.recent_tokens) *
        kBFloat16ValuesPerToken * sizeof(std::uint16_t);
    result.logical_int2_payload_bytes =
        static_cast<std::uint64_t>(result.historical_tokens) * kInt2PayloadBytesPerToken;
    result.logical_int2_metadata_bytes =
        static_cast<std::uint64_t>(result.historical_tokens) * kInt2MetadataBytesPerToken;
    result.mixed_total_bytes = result.physical_bf16_bytes + result.physical_int2_payload_bytes +
                               result.physical_int2_metadata_bytes + result.page_header_bytes +
                               result.slot_table_bytes;
    result.historical_bulk_total_bytes =
        result.physical_int2_payload_bytes + result.physical_int2_metadata_bytes +
        result.int2_page_count * sizeof(OscarMixedPageMetadata) +
        static_cast<std::uint64_t>(result.historical_tokens) * sizeof(OscarMixedSlotMetadata);
    result.logical_value_count = static_cast<std::uint64_t>(context_tokens_) *
                                 kBFloat16ValuesPerToken;
    result.raw_int2_bytes_per_value =
        static_cast<double>(ninfer::ops::kOscarInt2G128CodeBytes +
                            ninfer::ops::kOscarInt2G128MetadataItems * sizeof(float)) /
        ninfer::ops::kOscarInt2G128HeadDim;
    if (result.historical_tokens != 0) {
        const auto count = static_cast<double>(result.historical_tokens) * kBFloat16ValuesPerToken;
        result.historical_bulk_bytes_per_value =
            static_cast<double>(result.historical_bulk_total_bytes) / count;
        result.historical_bulk_bits_per_value = result.historical_bulk_bytes_per_value * 8.0;
    }
    if (result.logical_value_count != 0) {
        result.mixed_bytes_per_value = static_cast<double>(result.mixed_total_bytes) /
                                       static_cast<double>(result.logical_value_count);
        result.mixed_bits_per_value = result.mixed_bytes_per_value * 8.0;
    }
    return result;
}

OscarMixedCacheBundle::OscarMixedCacheBundle(std::uint64_t sequence_id,
                                             std::uint32_t context_tokens,
                                             std::span<const std::uint32_t> full_attention_layers)
    : sequence_id_(sequence_id), context_tokens_(context_tokens) {
    if (context_tokens_ == 0 || full_attention_layers.size() != kOscarMixedFullAttentionLayers.size()) {
        throw std::invalid_argument("OSCAR mixed-cache bundle topology is incomplete");
    }
    for (std::size_t index = 0; index < full_attention_layers.size(); ++index) {
        if (full_attention_layers[index] != kOscarMixedFullAttentionLayers[index]) {
            throw std::invalid_argument("OSCAR mixed-cache bundle contains non-full-attention layer");
        }
        layers_.emplace_back(full_attention_layers[index], sequence_id_, context_tokens_);
    }
    validate();
}

const OscarMixedLayerCache& OscarMixedCacheBundle::layer(std::uint32_t model_layer) const {
    for (const auto& value : layers_) {
        if (value.model_layer() == model_layer) { return value; }
    }
    throw std::out_of_range("OSCAR mixed-cache layer is not full-attention");
}

void OscarMixedCacheBundle::validate() const {
    if (layers_.size() != kOscarMixedFullAttentionLayers.size()) {
        throw std::invalid_argument("OSCAR mixed-cache bundle layer count mismatch");
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const auto& value = layers_[index];
        value.validate();
        if (value.model_layer() != kOscarMixedFullAttentionLayers[index] ||
            value.sequence_id() != sequence_id_ || value.context_tokens() != context_tokens_) {
            throw std::invalid_argument("OSCAR mixed-cache bundle layer identity mismatch");
        }
        if (index != 0) {
            const auto& reference_pages = layers_[0].pages();
            const auto& pages = value.pages();
            if (pages.size() != reference_pages.size()) {
                throw std::invalid_argument("OSCAR mixed-cache layers use different page policy");
            }
            for (std::size_t page = 0; page < pages.size(); ++page) {
                const auto& left = reference_pages[page].metadata;
                const auto& right = pages[page].metadata;
                if (left.logical_token_begin != right.logical_token_begin ||
                    left.logical_token_end != right.logical_token_end ||
                    left.occupied_tokens != right.occupied_tokens || left.role != right.role ||
                    left.k_storage != right.k_storage || left.v_storage != right.v_storage ||
                    left.layout_version != right.layout_version || left.group_size != right.group_size) {
                    throw std::invalid_argument("OSCAR mixed-cache layers use different logical policy");
                }
            }
        }
    }
}

OscarMixedCacheAccounting OscarMixedCacheBundle::accounting() const {
    validate();
    OscarMixedCacheAccounting result;
    bool first = true;
    for (const auto& layer : layers_) {
        const OscarMixedCacheAccounting current = layer.accounting();
        if (first) {
            result = current;
            first = false;
        } else {
            result.page_count += current.page_count;
            result.bf16_page_count += current.bf16_page_count;
            result.int2_page_count += current.int2_page_count;
            result.logical_bf16_bytes += current.logical_bf16_bytes;
            result.logical_int2_payload_bytes += current.logical_int2_payload_bytes;
            result.logical_int2_metadata_bytes += current.logical_int2_metadata_bytes;
            result.physical_bf16_bytes += current.physical_bf16_bytes;
            result.physical_int2_payload_bytes += current.physical_int2_payload_bytes;
            result.physical_int2_metadata_bytes += current.physical_int2_metadata_bytes;
            result.page_header_bytes += current.page_header_bytes;
            result.slot_table_bytes += current.slot_table_bytes;
            result.mixed_total_bytes += current.mixed_total_bytes;
            result.historical_bulk_total_bytes += current.historical_bulk_total_bytes;
            result.full_attention_layers += current.full_attention_layers;
        }
    }
    result.logical_value_count = static_cast<std::uint64_t>(context_tokens_) *
                                 kBFloat16ValuesPerToken * result.full_attention_layers;
    if (result.historical_tokens != 0) {
        const auto count = static_cast<double>(result.historical_tokens) * kBFloat16ValuesPerToken *
                           result.full_attention_layers;
        result.historical_bulk_bytes_per_value =
            static_cast<double>(result.historical_bulk_total_bytes) / count;
        result.historical_bulk_bits_per_value = result.historical_bulk_bytes_per_value * 8.0;
    }
    if (result.logical_value_count != 0) {
        result.mixed_bytes_per_value = static_cast<double>(result.mixed_total_bytes) /
                                       static_cast<double>(result.logical_value_count);
        result.mixed_bits_per_value = result.mixed_bytes_per_value * 8.0;
    }
    return result;
}

OscarMixedAgingAssetContract OscarMixedAgingAssetContract::c4_cal30k() {
    OscarMixedAgingAssetContract result;
    result.asset_identity = kOscarMixedC4AssetIdentity;
    result.model_sha256 = kOscarMixedC4ModelSha256;
    result.asset_manifest_sha256 = kOscarMixedC4AssetManifestSha256;
    result.rotation_mode = kOscarMixedC4RotationMode;
    result.total_layers = kOscarMixedTotalLayers;
    result.query_heads = kOscarMixedQueryHeads;
    result.kv_heads = kOscarMixedKVHeads;
    result.gqa_ratio = kOscarMixedGqaRatio;
    result.head_dim = kOscarMixedHeadDim;
    result.rotary_dim = kOscarMixedRotaryDim;
    result.group_size = ninfer::ops::kOscarInt2G128GroupSize;
    result.calibrated = true;
    result.full_attention_layers = kOscarMixedFullAttentionLayers;
    return result;
}

OscarMixedAgingAssetContract OscarMixedAgingAssetContract::from_runtime(
    std::string asset_identity, std::string model_sha256, std::string asset_manifest_sha256,
    std::string rotation_mode) {
    OscarMixedAgingAssetContract result;
    result.asset_identity          = std::move(asset_identity);
    result.model_sha256            = std::move(model_sha256);
    result.asset_manifest_sha256   = std::move(asset_manifest_sha256);
    result.rotation_mode           = std::move(rotation_mode);
    result.total_layers            = kOscarMixedTotalLayers;
    result.query_heads             = kOscarMixedQueryHeads;
    result.kv_heads                = kOscarMixedKVHeads;
    result.gqa_ratio               = kOscarMixedGqaRatio;
    result.head_dim                = kOscarMixedHeadDim;
    result.rotary_dim              = kOscarMixedRotaryDim;
    result.group_size              = ninfer::ops::kOscarInt2G128GroupSize;
    result.calibrated              = true;
    result.full_attention_layers   = kOscarMixedFullAttentionLayers;
    result.validate();
    return result;
}

void OscarMixedAgingAssetContract::validate() const {
    const auto c4 = c4_cal30k();
    const auto cal10k = [&] {
        auto value = c4;
        value.asset_identity        = kOscarMixedCal10kAssetIdentity;
        value.asset_manifest_sha256 = kOscarMixedCal10kAssetManifestSha256;
        return value;
    }();
    if (!same_asset_contract(*this, c4) && !same_asset_contract(*this, cal10k)) {
        throw std::invalid_argument(
            "OSCAR aging requires a validated C4 cal30k or cal10k asset/topology contract");
    }
}

OscarMixedAgingLayerCache::OscarMixedAgingLayerCache(
    std::uint32_t model_layer, std::uint64_t sequence_id,
    const OscarMixedAgingAssetContract& asset)
    : model_layer_(model_layer), sequence_id_(sequence_id), asset_(asset) {
    asset_.validate();
    if (!is_full_attention_layer(model_layer_)) {
        throw std::invalid_argument("OSCAR aging accepts full-attention layers only");
    }
}

void OscarMixedAgingLayerCache::append(std::uint32_t logical_token,
                                       std::span<const std::uint16_t> k_bf16,
                                       std::span<const std::uint16_t> v_bf16) {
    constexpr std::size_t kRowValues =
        static_cast<std::size_t>(kOscarMixedKVHeads) * kOscarMixedHeadDim;
    if (logical_token != tokens_.size()) {
        throw std::invalid_argument("OSCAR aging append must be the next logical token");
    }
    if (k_bf16.size() != kRowValues || v_bf16.size() != kRowValues) {
        throw std::invalid_argument("OSCAR aging append requires four BF16 KV heads of D=256");
    }
    TokenRecord record;
    std::copy(k_bf16.begin(), k_bf16.end(), record.k_bf16.begin());
    std::copy(v_bf16.begin(), v_bf16.end(), record.v_bf16.begin());
    for (const std::uint16_t bits : record.k_bf16) {
        if (!std::isfinite(bf16_bits_to_float(bits))) {
            throw std::invalid_argument("OSCAR aging K input contains NaN or Inf");
        }
    }
    for (const std::uint16_t bits : record.v_bf16) {
        if (!std::isfinite(bf16_bits_to_float(bits))) {
            throw std::invalid_argument("OSCAR aging V input contains NaN or Inf");
        }
    }
    record.has_bf16 = true;
    tokens_.push_back(record);

    const auto ranges = ranges_for(static_cast<std::uint32_t>(tokens_.size()));
    if (ranges[1].logical_end > ranges[1].logical_begin) {
        age_token(ranges[1].logical_end - 1U);
    } else {
        rebuild_pages();
    }
}

void OscarMixedAgingLayerCache::age_token(std::uint32_t logical_token) {
    if (logical_token >= tokens_.size() || logical_token < kOscarMixedPrefixTokens) {
        throw std::invalid_argument("OSCAR aging target is outside the non-prefix cache");
    }
    const auto ranges = ranges_for(static_cast<std::uint32_t>(tokens_.size()));
    if (ranges[1].logical_end == ranges[1].logical_begin ||
        logical_token != ranges[1].logical_end - 1U) {
        throw std::invalid_argument("OSCAR aging target is not the current oldest recent token");
    }
    TokenRecord& record = tokens_[logical_token];
    if (record.historical || record.aged_once || !record.has_bf16) {
        throw std::logic_error("OSCAR aging attempted a second or invalid conversion");
    }

    std::array<float, kOscarMixedHeadDim> k_values{};
    std::array<float, kOscarMixedHeadDim> v_values{};
    for (std::uint32_t head = 0; head < kOscarMixedKVHeads; ++head) {
        const std::size_t begin = static_cast<std::size_t>(head) * kOscarMixedHeadDim;
        for (std::uint32_t dimension = 0; dimension < kOscarMixedHeadDim; ++dimension) {
            k_values[dimension] = bf16_bits_to_float(record.k_bf16[begin + dimension]);
            v_values[dimension] = bf16_bits_to_float(record.v_bf16[begin + dimension]);
        }
        record.k_int2[head] = ninfer::ops::oscar_int2_g128_encode(
            k_values.data(), static_cast<int>(kOscarMixedHeadDim), 0.96F);
        record.v_int2[head] = ninfer::ops::oscar_int2_g128_encode(
            v_values.data(), static_cast<int>(kOscarMixedHeadDim), 0.92F);
    }
    record.k_bf16.fill(0);
    record.v_bf16.fill(0);
    record.has_bf16 = false;
    record.historical = true;
    record.aged_once = true;
    ++aging_conversion_count_;
    rebuild_pages();
}

void OscarMixedAgingLayerCache::rebuild_pages() {
    pages_.clear();
    const std::uint32_t context_tokens = static_cast<std::uint32_t>(tokens_.size());
    if (context_tokens == 0) { return; }

    const auto ranges = ranges_for(context_tokens);
    std::uint32_t next_physical_page = 0;
    std::vector<OscarMixedPage> rebuilt;
    for (const RegionRange& range : ranges) {
        const std::uint32_t token_count = range.logical_end - range.logical_begin;
        const std::uint32_t page_count =
            (token_count + kOscarMixedPageTokens - 1U) / kOscarMixedPageTokens;
        for (std::uint32_t pool_page = 0; pool_page < page_count; ++pool_page) {
            const std::uint32_t logical_begin =
                range.logical_begin + pool_page * kOscarMixedPageTokens;
            const std::uint32_t logical_end =
                std::min(range.logical_end, logical_begin + kOscarMixedPageTokens);
            rebuilt.push_back(make_page(model_layer_, sequence_id_, logical_begin, logical_end,
                                        next_physical_page++, pool_page, range.role,
                                        range.storage));
            OscarMixedPage& page = rebuilt.back();
            for (std::uint32_t logical = logical_begin; logical < logical_end; ++logical) {
                const TokenRecord& record = tokens_[logical];
                const std::uint32_t offset = logical - logical_begin;
                if (range.storage == OscarMixedStorageType::BFloat16) {
                    if (!record.has_bf16 || record.historical) {
                        throw std::logic_error("OSCAR aging BF16 page received historical token");
                    }
                    auto& storage = std::get<OscarMixedBFloat16PageStorage>(page.storage);
                    const std::size_t row_offset =
                        static_cast<std::size_t>(offset) * kOscarMixedKVHeads * kOscarMixedHeadDim;
                    std::copy(record.k_bf16.begin(), record.k_bf16.end(),
                              storage.k.begin() + row_offset);
                    std::copy(record.v_bf16.begin(), record.v_bf16.end(),
                              storage.v.begin() + row_offset);
                } else {
                    if (!record.historical || record.has_bf16) {
                        throw std::logic_error("OSCAR aging INT2 page received BF16 token");
                    }
                    auto& storage = std::get<OscarMixedInt2G128PageStorage>(page.storage);
                    for (std::uint32_t head = 0; head < kOscarMixedKVHeads; ++head) {
                        const auto& k_encoded = record.k_int2[head];
                        const auto& v_encoded = record.v_int2[head];
                        const std::size_t row =
                            static_cast<std::size_t>(offset) * kOscarMixedKVHeads + head;
                        std::copy(k_encoded.packed.begin(), k_encoded.packed.end(),
                                  storage.k_packed.begin() + row *
                                      ninfer::ops::kOscarInt2G128CodeBytes);
                        std::copy(v_encoded.packed.begin(), v_encoded.packed.end(),
                                  storage.v_packed.begin() + row *
                                      ninfer::ops::kOscarInt2G128CodeBytes);
                        const std::size_t metadata = row * ninfer::ops::kOscarInt2G128MetadataItems;
                        std::copy(k_encoded.scales_zeros.begin(), k_encoded.scales_zeros.end(),
                                  storage.k_scales_zeros.begin() + metadata);
                        std::copy(v_encoded.scales_zeros.begin(), v_encoded.scales_zeros.end(),
                                  storage.v_scales_zeros.begin() + metadata);
                    }
                }
            }
        }
    }
    pages_ = std::move(rebuilt);
}

OscarMixedResolvedSlot OscarMixedAgingLayerCache::resolve(std::uint32_t logical_token) const {
    if (logical_token >= tokens_.size()) {
        throw std::out_of_range("OSCAR aging logical token is out of range");
    }
    for (std::uint32_t page_index = 0; page_index < pages_.size(); ++page_index) {
        const auto& page = pages_[page_index];
        if (logical_token >= page.metadata.logical_token_begin &&
            logical_token < page.metadata.logical_token_end) {
            const std::size_t offset = logical_token - page.metadata.logical_token_begin;
            return {page_index, page.slots.at(offset)};
        }
    }
    throw std::logic_error("OSCAR aging logical token has no physical slot");
}

const ops::OscarInt2G128EncodedRow& OscarMixedAgingLayerCache::historical_k(
    std::uint32_t logical_token, std::uint32_t kv_head) const {
    if (kv_head >= kOscarMixedKVHeads || logical_token >= tokens_.size() ||
        !tokens_[logical_token].historical) {
        throw std::invalid_argument("OSCAR historical K row is unavailable");
    }
    return tokens_[logical_token].k_int2[kv_head];
}

const ops::OscarInt2G128EncodedRow& OscarMixedAgingLayerCache::historical_v(
    std::uint32_t logical_token, std::uint32_t kv_head) const {
    if (kv_head >= kOscarMixedKVHeads || logical_token >= tokens_.size() ||
        !tokens_[logical_token].historical) {
        throw std::invalid_argument("OSCAR historical V row is unavailable");
    }
    return tokens_[logical_token].v_int2[kv_head];
}

bool OscarMixedAgingLayerCache::has_bf16_payload(std::uint32_t logical_token) const {
    if (logical_token >= tokens_.size()) {
        throw std::out_of_range("OSCAR aging logical token is out of range");
    }
    return tokens_[logical_token].has_bf16;
}

bool OscarMixedAgingLayerCache::was_aged_once(std::uint32_t logical_token) const {
    if (logical_token >= tokens_.size()) {
        throw std::out_of_range("OSCAR aging logical token is out of range");
    }
    return tokens_[logical_token].aged_once;
}

OscarMixedAgingLayerCache OscarMixedAgingLayerCache::clone_for_sequence(
    std::uint64_t sequence_id) const {
    validate();
    OscarMixedAgingLayerCache result = *this;
    result.sequence_id_ = sequence_id;
    for (auto& page : result.pages_) {
        page.metadata.sequence_id = sequence_id;
        for (auto& slot : page.slots) { slot.sequence_id = sequence_id; }
    }
    result.validate();
    if (result.aging_conversion_count_ != aging_conversion_count_) {
        throw std::logic_error("OSCAR aging clone changed conversion count");
    }
    return result;
}

void OscarMixedAgingLayerCache::validate() const {
    asset_.validate();
    if (!is_full_attention_layer(model_layer_)) {
        throw std::invalid_argument("OSCAR aging layer is not full-attention");
    }
    const std::uint32_t context_tokens = static_cast<std::uint32_t>(tokens_.size());
    if (context_tokens == 0) {
        if (!pages_.empty() || aging_conversion_count_ != 0) {
            throw std::invalid_argument("empty OSCAR aging cache has physical state");
        }
        return;
    }
    const auto ranges = ranges_for(context_tokens);
    std::uint32_t aged_count = 0;
    for (std::uint32_t logical = 0; logical < context_tokens; ++logical) {
        const TokenRecord& record = tokens_[logical];
        const bool historical = logical >= ranges[1].logical_begin &&
                                logical < ranges[1].logical_end;
        if (record.historical != historical || record.has_bf16 == historical ||
            (logical < kOscarMixedPrefixTokens && record.aged_once) ||
            (historical && !record.aged_once)) {
            throw std::invalid_argument("OSCAR aging tier state violates the logical policy");
        }
        if (record.historical) {
            ++aged_count;
            for (std::uint32_t head = 0; head < kOscarMixedKVHeads; ++head) {
                for (const float value : record.k_int2[head].scales_zeros) {
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("OSCAR aging K metadata is not finite");
                    }
                }
                for (const float value : record.v_int2[head].scales_zeros) {
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("OSCAR aging V metadata is not finite");
                    }
                }
                for (const std::uint8_t symbol : record.k_int2[head].symbols) {
                    if (symbol > ninfer::ops::kOscarInt2G128Levels) {
                        throw std::invalid_argument("OSCAR aging K symbol is out of range");
                    }
                }
                for (const std::uint8_t symbol : record.v_int2[head].symbols) {
                    if (symbol > ninfer::ops::kOscarInt2G128Levels) {
                        throw std::invalid_argument("OSCAR aging V symbol is out of range");
                    }
                }
            }
        }
    }
    if (aged_count != aging_conversion_count_) {
        throw std::invalid_argument("OSCAR aging conversion count does not match historical rows");
    }

    OscarMixedLayerCache expected(model_layer_, sequence_id_, context_tokens);
    if (pages_.size() != expected.pages().size()) {
        throw std::invalid_argument("OSCAR aging physical page count mismatch");
    }
    for (std::size_t page_index = 0; page_index < pages_.size(); ++page_index) {
        const auto& actual = pages_[page_index];
        const auto& shape = expected.pages()[page_index];
        if (!same_page_metadata(actual.metadata, shape.metadata) ||
            actual.slots.size() != shape.slots.size()) {
            throw std::invalid_argument("OSCAR aging page metadata mismatch");
        }
        for (std::size_t slot = 0; slot < actual.slots.size(); ++slot) {
            if (!same_slot_metadata(actual.slots[slot], shape.slots[slot])) {
                throw std::invalid_argument("OSCAR aging slot metadata mismatch");
            }
        }
    }
    for (std::uint32_t logical = 0; logical < context_tokens; ++logical) {
        const auto resolved = resolve(logical);
        const auto& page = pages_[resolved.page_index];
        const std::size_t offset = logical - page.metadata.logical_token_begin;
        const TokenRecord& record = tokens_[logical];
        if (!record.historical) {
            if (!std::holds_alternative<OscarMixedBFloat16PageStorage>(page.storage)) {
                throw std::invalid_argument("OSCAR aging BF16 row has non-BF16 storage");
            }
            const auto& storage = std::get<OscarMixedBFloat16PageStorage>(page.storage);
            const std::size_t row_offset =
                offset * kOscarMixedKVHeads * kOscarMixedHeadDim;
            if (!std::equal(record.k_bf16.begin(), record.k_bf16.end(),
                            storage.k.begin() + row_offset) ||
                !std::equal(record.v_bf16.begin(), record.v_bf16.end(),
                            storage.v.begin() + row_offset)) {
                throw std::invalid_argument("OSCAR aging BF16 storage does not match token record");
            }
        } else {
            if (!std::holds_alternative<OscarMixedInt2G128PageStorage>(page.storage)) {
                throw std::invalid_argument("OSCAR aging historical row has non-INT2 storage");
            }
            const auto& storage = std::get<OscarMixedInt2G128PageStorage>(page.storage);
            for (std::uint32_t head = 0; head < kOscarMixedKVHeads; ++head) {
                const std::size_t row = offset * kOscarMixedKVHeads + head;
                const auto& k_encoded = record.k_int2[head];
                const auto& v_encoded = record.v_int2[head];
                if (!std::equal(k_encoded.packed.begin(), k_encoded.packed.end(),
                                storage.k_packed.begin() +
                                    row * ninfer::ops::kOscarInt2G128CodeBytes) ||
                    !std::equal(v_encoded.packed.begin(), v_encoded.packed.end(),
                                storage.v_packed.begin() +
                                    row * ninfer::ops::kOscarInt2G128CodeBytes)) {
                    throw std::invalid_argument("OSCAR aging packed row mismatch");
                }
                const std::size_t metadata = row * ninfer::ops::kOscarInt2G128MetadataItems;
                if (!std::equal(k_encoded.scales_zeros.begin(), k_encoded.scales_zeros.end(),
                                storage.k_scales_zeros.begin() + metadata) ||
                    !std::equal(v_encoded.scales_zeros.begin(), v_encoded.scales_zeros.end(),
                                storage.v_scales_zeros.begin() + metadata)) {
                    throw std::invalid_argument("OSCAR aging FP32 metadata mismatch");
                }
            }
        }
    }
}

OscarMixedCacheAccounting OscarMixedAgingLayerCache::accounting() const {
    validate();
    return accounting_for_pages(static_cast<std::uint32_t>(tokens_.size()), 1, pages_);
}

OscarMixedAgingCacheBundle::OscarMixedAgingCacheBundle(
    std::uint64_t sequence_id, const OscarMixedAgingAssetContract& asset,
    std::span<const std::uint32_t> full_attention_layers)
    : sequence_id_(sequence_id), asset_(asset) {
    asset_.validate();
    if (full_attention_layers.size() != kOscarMixedFullAttentionLayers.size()) {
        throw std::invalid_argument("OSCAR aging bundle topology is incomplete");
    }
    for (std::size_t index = 0; index < full_attention_layers.size(); ++index) {
        if (full_attention_layers[index] != kOscarMixedFullAttentionLayers[index]) {
            throw std::invalid_argument("OSCAR aging bundle contains a non-full-attention layer");
        }
        layers_.emplace_back(full_attention_layers[index], sequence_id_, asset_);
    }
    validate();
}

void OscarMixedAgingCacheBundle::append(std::uint32_t logical_token,
                                        std::span<const std::uint16_t> k_bf16_by_layer,
                                        std::span<const std::uint16_t> v_bf16_by_layer) {
    constexpr std::size_t kRowValues =
        static_cast<std::size_t>(kOscarMixedKVHeads) * kOscarMixedHeadDim;
    const std::size_t expected = layers_.size() * kRowValues;
    if (k_bf16_by_layer.size() != expected || v_bf16_by_layer.size() != expected) {
        throw std::invalid_argument("OSCAR aging bundle row span has the wrong layer extent");
    }
    if (logical_token != context_tokens()) {
        throw std::invalid_argument("OSCAR aging bundle append token is not contiguous");
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const std::size_t offset = index * kRowValues;
        layers_[index].append(logical_token,
                              k_bf16_by_layer.subspan(offset, kRowValues),
                              v_bf16_by_layer.subspan(offset, kRowValues));
    }
}

OscarMixedAgingLayerCache& OscarMixedAgingCacheBundle::layer(std::uint32_t model_layer) {
    for (auto& value : layers_) {
        if (value.model_layer() == model_layer) { return value; }
    }
    throw std::out_of_range("OSCAR aging layer is not full-attention");
}

const OscarMixedAgingLayerCache& OscarMixedAgingCacheBundle::layer(
    std::uint32_t model_layer) const {
    for (const auto& value : layers_) {
        if (value.model_layer() == model_layer) { return value; }
    }
    throw std::out_of_range("OSCAR aging layer is not full-attention");
}

OscarMixedAgingCacheBundle OscarMixedAgingCacheBundle::clone_for_sequence(
    std::uint64_t sequence_id) const {
    validate();
    OscarMixedAgingCacheBundle result(sequence_id, asset_, kOscarMixedFullAttentionLayers);
    result.layers_.clear();
    result.layers_.reserve(layers_.size());
    for (const auto& layer : layers_) {
        result.layers_.push_back(layer.clone_for_sequence(sequence_id));
    }
    result.validate();
    return result;
}

std::uint32_t OscarMixedAgingCacheBundle::aging_conversion_count() const noexcept {
    std::uint32_t result = 0;
    for (const auto& layer : layers_) { result += layer.aging_conversion_count(); }
    return result;
}

void OscarMixedAgingCacheBundle::validate() const {
    asset_.validate();
    if (layers_.size() != kOscarMixedFullAttentionLayers.size()) {
        throw std::invalid_argument("OSCAR aging bundle layer count mismatch");
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const auto& value = layers_[index];
        value.validate();
        if (value.model_layer() != kOscarMixedFullAttentionLayers[index] ||
            value.sequence_id() != sequence_id_ || !same_asset_contract(value.asset(), asset_)) {
            throw std::invalid_argument("OSCAR aging bundle layer identity mismatch");
        }
        if (index != 0) {
            const auto& reference = layers_[0].pages();
            const auto& pages = value.pages();
            if (pages.size() != reference.size()) {
                throw std::invalid_argument("OSCAR aging layers use different logical policy");
            }
            for (std::size_t page = 0; page < pages.size(); ++page) {
                const auto& left = pages[page].metadata;
                const auto& right = reference[page].metadata;
                if (left.logical_token_begin != right.logical_token_begin ||
                    left.logical_token_end != right.logical_token_end || left.role != right.role ||
                    left.k_storage != right.k_storage || left.v_storage != right.v_storage) {
                    throw std::invalid_argument("OSCAR aging layers use different tier policy");
                }
            }
        }
    }
}

OscarMixedCacheAccounting OscarMixedAgingCacheBundle::accounting() const {
    validate();
    if (layers_.empty()) { return {}; }
    return accounting_for_pages(context_tokens(), layers_.size(), layers_.front().pages());
}

} // namespace ninfer
