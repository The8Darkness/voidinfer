#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
#include <limits>
#include <stdexcept>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::size_t kPreparedVisionPatchFeatures = 3ULL * 2ULL * 16ULL * 16ULL;
inline constexpr std::uint64_t kRawPatchesPerVisionToken  = 4;
// Aggregate prompt capacity and one-item execution capacity are intentionally distinct. Multiple
// media items are retained by one prepared prompt but pass through the Vision tower sequentially.
inline constexpr std::uint64_t kMaximumPromptVisionTokens = 32'768;
inline constexpr std::uint64_t kMaximumPromptVisionRawPatches =
    kMaximumPromptVisionTokens * kRawPatchesPerVisionToken;
inline constexpr std::uint64_t kMaximumVisionItemTokens = 16'384;
inline constexpr std::uint64_t kMaximumVisionItemRawPatches =
    kMaximumVisionItemTokens * kRawPatchesPerVisionToken;

// Exact effective settings that can change prepared patch rows. Resource-only
// prompt quotas are intentionally excluded: they decide admission but do not
// change an already admitted item's representation.
struct MediaPreprocessIdentity {
    std::uint32_t schema_revision = 0;
    std::uint64_t minimum_pixels = 0;
    std::uint64_t maximum_pixels = 0;
    std::uint32_t spatial_patch = 0;
    std::uint32_t temporal_patch = 0;
    std::uint32_t merge = 0;
    std::uint32_t normalization_revision = 0;
    double video_fps = 0.0;
    std::int32_t video_min_frames = 0;
    std::int32_t video_max_frames = 0;

    [[nodiscard]] bool valid(bool video) const noexcept {
        if(schema_revision!=1 || !minimum_pixels || maximum_pixels<minimum_pixels ||
           spatial_patch!=16 || temporal_patch!=2 || merge!=2 ||
           normalization_revision!=1)return false;
        if(video)
            return video_fps>0.0 && video_min_frames>0 &&
                video_max_frames>=video_min_frames;
        return video_fps==0.0 && video_min_frames==0 && video_max_frames==0;
    }

    [[nodiscard]] friend bool operator==(const MediaPreprocessIdentity&,
                                         const MediaPreprocessIdentity&) noexcept = default;
};

struct PreparedMediaPayload {
    // Exact row-major input consumed by the selected Vision patch projection.
    VisionPatchStorage storage = VisionPatchStorage::BFloat16;
    MediaPreprocessIdentity preprocess;
    std::unique_ptr<std::uint16_t[]> patches;
    std::size_t patch_elements = 0;

    [[nodiscard]] std::span<std::uint16_t> mutable_span() noexcept {
        return {patches.get(), patch_elements};
    }

    [[nodiscard]] std::span<const std::uint16_t> span() const noexcept {
        return {patches.get(), patch_elements};
    }
};

enum class PromptModality : std::uint8_t {
    Image = 1,
    Video = 2,
};

struct VisionGrid {
    std::int32_t temporal = 0;
    std::int32_t height   = 0;
    std::int32_t width    = 0;
};

struct TokenSpan {
    std::size_t begin = 0;
    std::size_t count = 0;
};

struct VisionItem {
    PromptModality modality = PromptModality::Image;
    VisionGrid grid;
    std::size_t patch_begin = 0;
    std::size_t patch_count = 0;
    // SHA-256 of the owned encoded media bytes. Grid/modality/span identity is carried
    // separately so this digest binds the content without retaining the request payload.
    std::array<std::uint8_t, 32> content_digest{};
    MediaPreprocessIdentity preprocess;
    std::vector<double> timestamps;
    std::vector<TokenSpan> token_spans;
    // Retained after the owning encoder payload is retired. Identical encoded
    // bytes represented as BF16 and FP16 are different encoder inputs.
    VisionPatchStorage patch_storage = VisionPatchStorage::BFloat16;
};

enum class RewriteCheckpointKind : std::uint8_t {
    TurnClosure,
    ResponseReplay,
};

struct RewriteCheckpointSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct PromptIdentity {
    bool reusable = true;
    std::optional<RewriteCheckpointSpec> rewrite_checkpoint;
    // Exact token frontiers at which this serialization can agree with a typed rewrite captured
    // by an earlier turn. Prefill splits at these frontiers so resumed and root execution use the
    // same GDN decomposition; they are not capture requests by themselves.
    std::vector<std::uint32_t> rewrite_execution_frontiers;
};

inline constexpr std::size_t kPreparedSessionKeyCapacity = kMaximumContextCacheSessionKeyBytes;

struct PreparedSessionKey {
    std::uint16_t size = 0;
    std::array<char, kPreparedSessionKeyCapacity> bytes{};

    [[nodiscard]] std::string_view view() const noexcept { return {bytes.data(), size}; }

    [[nodiscard]] friend bool operator==(const PreparedSessionKey&,
                                         const PreparedSessionKey&) noexcept = default;
};

struct PreparedCacheOpportunity {
    PromptCacheMarkerKind kind = PromptCacheMarkerKind::SharedStablePrefix;
    std::uint32_t frontier     = 0;
    std::uint32_t input_order  = 0;

    [[nodiscard]] friend bool operator==(PreparedCacheOpportunity,
                                         PreparedCacheOpportunity) noexcept = default;
};

struct PreparedContextCache {
    std::optional<PreparedSessionKey> session_key;
    runtime::RetentionClass retention = runtime::RetentionClass::RecentPrivate;
    std::vector<PreparedCacheOpportunity> opportunities;
    // Controls replacement of a named SessionIndex entry, not anonymous source ownership.
    bool update_session_index = true;
};

struct PrepareStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::uint64_t attention_pairs        = 0; // Informational; not enforced against any budget.
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
    // Exact text-only render/tokenize cache observations for this preparation.
    std::size_t frontend_cache_hits      = 0;
    std::size_t frontend_cache_misses    = 0;
    std::size_t reused_token_bytes       = 0;
    // Snapshot at preparation completion: distinct payload objects plus patch
    // storage reachable from this input, excluding allocator/control-block overhead.
    std::size_t retained_media_payload_bytes = 0;
};

enum class MediaPayloadRetentionMode : std::uint8_t {
    IdentityOnly,
    ReplayRequired,
};

struct MediaPayloadRetention {
    MediaPayloadRetentionMode mode = MediaPayloadRetentionMode::IdentityOnly;
    // Logical prepared items that still retain an encoder input. Multiple items
    // may share one immutable payload allocation.
    std::size_t live_items    = 0;
    // Physical payload-object and patch-array bytes reachable from this prompt,
    // deduplicated by allocation identity within the prompt.
    std::size_t retained_bytes = 0;
};

struct PreparedPromptData {
    std::vector<TokenId> token_ids;
    std::vector<std::uint8_t> token_types;
    std::vector<std::int32_t> positions;
    std::int32_t rope_delta = 0;
    // One immutable payload per Vision item, in the same order as vision_items.
    std::vector<std::shared_ptr<const PreparedMediaPayload>> media_payloads;
    std::vector<VisionItem> vision_items;
    PromptIdentity identity;
    PreparedContextCache context_cache;
    bool starts_in_reasoning = false;
    PrepareStats prepare;

    [[nodiscard]] std::span<const std::int32_t> position_axis(int axis) const;

    [[nodiscard]] bool has_media() const noexcept { return !vision_items.empty(); }

    [[nodiscard]] bool media_payload_identity_valid() const noexcept {
        if(media_payloads.size()!=vision_items.size())return false;
        for(std::size_t i=0;i<vision_items.size();++i) {
            const auto& item=vision_items[i];const auto& payload=media_payloads[i];
            if(!payload || item.patch_storage!=payload->storage ||
               item.preprocess!=payload->preprocess ||
               !item.preprocess.valid(item.modality==PromptModality::Video))return false;
        }
        return true;
    }

    // Physical payload allocations retained by this input, once per payload
    // object. Does not count logical media uses or claim process-wide uniqueness.
    // A caller combining inputs must union the returned allocation addresses.
    template<class Visitor> void visit_media_payload_allocations(Visitor&& visitor) const {
        for(std::size_t i=0;i<media_payloads.size();++i) {
            const auto& payload=media_payloads[i];if(!payload)continue;
            bool seen=false;
            for(std::size_t prior=0;prior<i;++prior)
                if(media_payloads[prior].get()==payload.get()){seen=true;break;}
            if(seen)continue;
            if(payload->patch_elements>std::numeric_limits<std::size_t>::max()/sizeof(std::uint16_t))
                throw std::overflow_error("prepared media patch byte count overflows size_t");
            if(payload->patch_elements && !payload->patches)
                throw std::invalid_argument("prepared media patch storage missing");
            visitor(payload.get(),sizeof(PreparedMediaPayload));
            if(payload->patch_elements)
                visitor(payload->patches.get(),payload->patch_elements*sizeof(std::uint16_t));
        }
    }

    // Payload slots remain indexed one-to-one with vision_items for the lifetime of the prompt.
    // Reachable allocation extents for this one prompt. Shared media payloads
    // are counted once here, but callers must union owners across prompts.
    template<class Visitor> void visit_storage_allocations(Visitor&& visitor) const {
        visitor(this,sizeof(PreparedPromptData));
        const auto vector_storage=[&](const auto& values) {
            if(values.capacity()>std::numeric_limits<std::size_t>::max()/sizeof(*values.data()))
                throw std::overflow_error("prepared vector storage extent overflow");
            if(values.capacity())visitor(values.data(),values.capacity()*sizeof(*values.data()));
        };
        vector_storage(token_ids);vector_storage(token_types);vector_storage(positions);
        vector_storage(media_payloads);vector_storage(vision_items);
        vector_storage(identity.rewrite_execution_frontiers);
        vector_storage(context_cache.opportunities);
        for(const auto& item:vision_items) {
            vector_storage(item.timestamps);vector_storage(item.token_spans);
        }
        visit_media_payload_allocations(visitor);
    }
    [[nodiscard]] std::size_t retained_storage_bytes() const {
        std::size_t total=0;
        visit_storage_allocations([&](const void*,std::size_t bytes) {
            if(bytes>std::numeric_limits<std::size_t>::max()-total)
                throw std::overflow_error("prepared storage byte count overflow");
            total+=bytes;
        });
        return total;
    }

    [[nodiscard]] MediaPayloadRetention media_payload_retention() const {
        MediaPayloadRetention result;
        result.live_items=static_cast<std::size_t>(std::count_if(
            media_payloads.begin(),media_payloads.end(),[](const auto& payload){return bool(payload);}));
        visit_media_payload_allocations([&](const void*,std::size_t bytes) {
            if(bytes>std::numeric_limits<std::size_t>::max()-result.retained_bytes)
                throw std::overflow_error("prepared media live byte count overflows size_t");
            result.retained_bytes+=bytes;
        });
        if(result.live_items)result.mode=MediaPayloadRetentionMode::ReplayRequired;
        return result;
    }

    // Declare the exact prepared items whose encoder inputs can still be
    // revisited by the selected execution suffix. Validation precedes every
    // release, so an invalid plan cannot partially retire replay inputs.
    void retain_media_payloads_for_replay(std::span<const std::uint32_t> item_indices) {
        if(media_payloads.size()!=vision_items.size())
            throw std::invalid_argument("prepared media replay owner shape mismatch");
        for(std::size_t i=0;i<item_indices.size();++i) {
            const auto item=item_indices[i];
            if(item>=media_payloads.size() || !media_payloads[item])
                throw std::invalid_argument("prepared media replay payload missing");
            if(i && item_indices[i-1]>=item)
                throw std::invalid_argument("prepared media replay items are not strictly ordered");
        }
        std::size_t required=0;
        for(std::size_t item=0;item<media_payloads.size();++item) {
            if(required<item_indices.size() && item_indices[required]==item)++required;
            else media_payloads[item].reset();
        }
    }

    // Payload slots remain indexed one-to-one with vision_items for the lifetime of the prompt.
    // Releasing host storage must not destroy that structural identity while a Vision prefill
    // session can still revisit the same item in a later Text chunk.
    void release_all_media_payloads() noexcept {
        for (auto& payload : media_payloads) { payload.reset(); }
    }
};

class PreparedPromptAccess {
public:
    [[nodiscard]] static const PreparedPromptData& view(const PreparedPrompt& prompt);
    [[nodiscard]] static PreparedPromptData take(PreparedPrompt&& prompt);
};

} // namespace ninfer::targets::qwen3_6
