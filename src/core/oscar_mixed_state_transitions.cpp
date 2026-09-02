#include "core/oscar_mixed_state_transitions.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer {
namespace {

constexpr std::uint32_t kStateImageVersion = 1;
constexpr char kStateImageMagic[] = "OSCARC2C";
constexpr std::size_t kStateImageMagicBytes = sizeof(kStateImageMagic) - 1U;
constexpr std::size_t kRowValues =
    static_cast<std::size_t>(kOscarMixedKVHeads) * kOscarMixedHeadDim;
constexpr std::size_t kMaxStateImageString = 4096;
constexpr std::uint32_t kMaxStateImageTokens = 1U << 20U;
constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) noexcept {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
}

template <typename T>
void hash_scalar(std::uint64_t& hash, const T& value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    hash_bytes(hash, &value, sizeof(value));
}

template <typename T>
std::uint64_t hash_vector(const std::vector<T>& values) noexcept {
    std::uint64_t hash = kFnvOffset;
    const std::size_t size = values.size();
    hash_scalar(hash, size);
    if (!values.empty()) { hash_bytes(hash, values.data(), values.size() * sizeof(T)); }
    return hash;
}

void hash_slot(std::uint64_t& hash, const OscarMixedSlotMetadata& slot,
               bool include_sequence_id) noexcept {
    hash_scalar(hash, slot.model_layer);
    if (include_sequence_id) { hash_scalar(hash, slot.sequence_id); }
    hash_scalar(hash, slot.logical_token_begin);
    hash_scalar(hash, slot.logical_token_end);
    hash_scalar(hash, slot.physical_token_begin);
    hash_scalar(hash, slot.physical_token_end);
    hash_scalar(hash, slot.physical_page_index);
    hash_scalar(hash, slot.page_offset);
    hash_scalar(hash, slot.layout_version);
    hash_scalar(hash, slot.group_size);
    hash_scalar(hash, static_cast<std::uint8_t>(slot.k_storage));
    hash_scalar(hash, static_cast<std::uint8_t>(slot.v_storage));
    hash_scalar(hash, static_cast<std::uint8_t>(slot.role));
}

void hash_page_identity(std::uint64_t& hash, const OscarMixedPageMetadata& metadata,
                        bool include_sequence_id) noexcept {
    hash_scalar(hash, metadata.model_layer);
    if (include_sequence_id) { hash_scalar(hash, metadata.sequence_id); }
    hash_scalar(hash, metadata.logical_token_begin);
    hash_scalar(hash, metadata.logical_token_end);
    hash_scalar(hash, metadata.physical_token_begin);
    hash_scalar(hash, metadata.physical_token_end);
    hash_scalar(hash, metadata.physical_page_index);
    hash_scalar(hash, metadata.occupied_tokens);
    hash_scalar(hash, metadata.capacity_tokens);
    hash_scalar(hash, metadata.layout_version);
    hash_scalar(hash, metadata.group_size);
    hash_scalar(hash, static_cast<std::uint8_t>(metadata.k_storage));
    hash_scalar(hash, static_cast<std::uint8_t>(metadata.v_storage));
    hash_scalar(hash, static_cast<std::uint8_t>(metadata.role));
}

std::uint64_t page_slot_hash(const OscarMixedPage& page, bool include_sequence_id) noexcept {
    std::uint64_t hash = kFnvOffset;
    hash_scalar(hash, page.slots.size());
    for (const auto& slot : page.slots) { hash_slot(hash, slot, include_sequence_id); }
    return hash;
}

std::uint64_t page_payload_hash(const OscarMixedPageStorage& storage, bool k_payload) noexcept {
    return std::visit(
        [k_payload](const auto& value) noexcept -> std::uint64_t {
            using Storage = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Storage, OscarMixedBFloat16PageStorage>) {
                return hash_vector(k_payload ? value.k : value.v);
            } else {
                return hash_vector(k_payload ? value.k_packed : value.v_packed);
            }
        },
        storage);
}

std::uint64_t page_metadata_hash(const OscarMixedPageStorage& storage) noexcept {
    return std::visit(
        [](const auto& value) noexcept -> std::uint64_t {
            using Storage = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Storage, OscarMixedBFloat16PageStorage>) {
                return 0;
            } else {
                std::uint64_t hash = kFnvOffset;
                const auto k = hash_vector(value.k_scales_zeros);
                const auto v = hash_vector(value.v_scales_zeros);
                hash_scalar(hash, k);
                hash_scalar(hash, v);
                return hash;
            }
        },
        storage);
}

class ImageWriter {
public:
    template <typename T>
    void write(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const std::size_t old_size = bytes_.size();
        bytes_.resize(old_size + sizeof(T));
        std::memcpy(bytes_.data() + old_size, &value, sizeof(T));
    }

    void write_bytes(const void* data, std::size_t size) {
        const std::size_t old_size = bytes_.size();
        bytes_.resize(old_size + size);
        if (size != 0) { std::memcpy(bytes_.data() + old_size, data, size); }
    }

    void write_string(const std::string& value) {
        if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("OSCAR state image string is too long");
        }
        write(static_cast<std::uint32_t>(value.size()));
        write_bytes(value.data(), value.size());
    }

    [[nodiscard]] std::vector<std::byte> take() && { return std::move(bytes_); }

private:
    std::vector<std::byte> bytes_;
};

class ImageReader {
public:
    explicit ImageReader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <typename T>
    [[nodiscard]] T read() {
        static_assert(std::is_trivially_copyable_v<T>);
        require_available(sizeof(T));
        T value{};
        std::memcpy(&value, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    void read_bytes(void* destination, std::size_t size) {
        require_available(size);
        if (size != 0) { std::memcpy(destination, bytes_.data() + offset_, size); }
        offset_ += size;
    }

    [[nodiscard]] std::string read_string() {
        const auto size = read<std::uint32_t>();
        if (size > kMaxStateImageString) {
            throw std::invalid_argument("OSCAR state image string exceeds the safety limit");
        }
        std::string result(size, '\0');
        read_bytes(result.data(), result.size());
        return result;
    }

    [[nodiscard]] bool empty() const noexcept { return offset_ == bytes_.size(); }

private:
    std::span<const std::byte> bytes_;
    std::size_t offset_ = 0;

    void require_available(std::size_t size) const {
        if (size > bytes_.size() - offset_) {
            throw std::invalid_argument("truncated OSCAR state image");
        }
    }
};

void require_asset_string(const std::string& actual, const std::string& expected,
                          const char* field) {
    if (actual != expected) {
        throw std::invalid_argument(std::string("OSCAR state image ") + field +
                                    " does not match the validated asset");
    }
}

} // namespace

OscarMixedTransitionCache::OscarMixedTransitionCache(
    std::uint64_t sequence_id, const OscarMixedAgingAssetContract& asset)
    : bundle_(sequence_id, asset, kOscarMixedFullAttentionLayers) {
    validate();
}

void OscarMixedTransitionCache::append(
    std::uint32_t logical_token, std::span<const std::uint16_t> k_bf16_by_layer,
    std::span<const std::uint16_t> v_bf16_by_layer) {
    if (logical_token != context_tokens()) {
        throw std::invalid_argument("OSCAR transition append is not contiguous");
    }
    if (k_bf16_by_layer.size() != kOscarMixedFullAttentionLayers.size() * kRowValues ||
        v_bf16_by_layer.size() != kOscarMixedFullAttentionLayers.size() * kRowValues) {
        throw std::invalid_argument("OSCAR transition append has the wrong layer extent");
    }
    bundle_.append(logical_token, k_bf16_by_layer, v_bf16_by_layer);
    input_archive_.push_back(
        InputToken{std::vector<std::uint16_t>(k_bf16_by_layer.begin(), k_bf16_by_layer.end()),
                   std::vector<std::uint16_t>(v_bf16_by_layer.begin(), v_bf16_by_layer.end())});
    refresh_page_blocks();
    validate();
}

OscarMixedTransitionCache OscarMixedTransitionCache::fork(
    std::uint64_t child_sequence_id) const {
    validate();
    if (child_sequence_id == sequence_id()) {
        throw std::invalid_argument("OSCAR transition fork requires a distinct sequence id");
    }
    OscarMixedTransitionCache child(child_sequence_id, asset());
    child.bundle_ = bundle_.clone_for_sequence(child_sequence_id);
    child.input_archive_ = input_archive_;
    child.page_blocks_ = page_blocks_;
    child.lineage_base_hash_ = fingerprint(true).overall_hash;
    child.committed_ = false;
    child.validate();
    if (child.aging_conversion_count() != aging_conversion_count()) {
        throw std::logic_error("OSCAR transition fork re-encoded historical rows");
    }
    return child;
}

void OscarMixedTransitionCache::commit_from(OscarMixedTransitionCache&& child) {
    if (child.lineage_base_hash_ == 0 || child.lineage_base_hash_ != fingerprint(true).overall_hash) {
        throw std::logic_error("OSCAR transition commit source is not this parent lineage");
    }
    child.validate();
    *this = std::move(child);
    committed_ = true;
    lineage_base_hash_ = 0;
    validate();
}

std::vector<std::byte> OscarMixedTransitionCache::state_image() const {
    validate();
    ImageWriter writer;
    writer.write_bytes(kStateImageMagic, kStateImageMagicBytes);
    writer.write(kStateImageVersion);
    writer.write(sequence_id());
    writer.write(static_cast<std::uint32_t>(input_archive_.size()));
    writer.write(static_cast<std::uint32_t>(kOscarMixedFullAttentionLayers.size()));
    writer.write(static_cast<std::uint32_t>(kRowValues));
    writer.write_string(asset().asset_identity);
    writer.write_string(asset().model_sha256);
    writer.write_string(asset().asset_manifest_sha256);
    writer.write_string(asset().rotation_mode);
    for (const auto& token : input_archive_) {
        writer.write(static_cast<std::uint32_t>(token.k.size()));
        writer.write(static_cast<std::uint32_t>(token.v.size()));
        writer.write_bytes(token.k.data(), token.k.size() * sizeof(std::uint16_t));
        writer.write_bytes(token.v.data(), token.v.size() * sizeof(std::uint16_t));
    }
    return std::move(writer).take();
}

OscarMixedTransitionCache OscarMixedTransitionCache::from_state_image(
    std::span<const std::byte> image, const OscarMixedAgingAssetContract& expected_asset) {
    expected_asset.validate();
    ImageReader reader(image);
    std::array<char, kStateImageMagicBytes> magic{};
    reader.read_bytes(magic.data(), magic.size());
    if (!std::equal(magic.begin(), magic.end(), kStateImageMagic)) {
        throw std::invalid_argument("OSCAR state image magic does not match");
    }
    if (reader.read<std::uint32_t>() != kStateImageVersion) {
        throw std::invalid_argument("unsupported OSCAR state image version");
    }
    const auto sequence_id = reader.read<std::uint64_t>();
    const auto token_count = reader.read<std::uint32_t>();
    const auto layer_count = reader.read<std::uint32_t>();
    const auto row_values = reader.read<std::uint32_t>();
    if (token_count > kMaxStateImageTokens || layer_count != kOscarMixedFullAttentionLayers.size() ||
        row_values != kRowValues) {
        throw std::invalid_argument("OSCAR state image topology is unsupported");
    }
    require_asset_string(reader.read_string(), expected_asset.asset_identity, "asset identity");
    require_asset_string(reader.read_string(), expected_asset.model_sha256, "model hash");
    require_asset_string(reader.read_string(), expected_asset.asset_manifest_sha256,
                         "asset manifest");
    require_asset_string(reader.read_string(), expected_asset.rotation_mode, "rotation mode");

    OscarMixedTransitionCache result(sequence_id, expected_asset);
    std::vector<std::uint16_t> k_values;
    std::vector<std::uint16_t> v_values;
    for (std::uint32_t token = 0; token < token_count; ++token) {
        const auto k_size = reader.read<std::uint32_t>();
        const auto v_size = reader.read<std::uint32_t>();
        if (k_size != kOscarMixedFullAttentionLayers.size() * kRowValues ||
            v_size != kOscarMixedFullAttentionLayers.size() * kRowValues) {
            throw std::invalid_argument("OSCAR state image row extent is invalid");
        }
        k_values.resize(k_size);
        v_values.resize(v_size);
        reader.read_bytes(k_values.data(), k_values.size() * sizeof(std::uint16_t));
        reader.read_bytes(v_values.data(), v_values.size() * sizeof(std::uint16_t));
        result.append(token, k_values, v_values);
    }
    if (!reader.empty()) { throw std::invalid_argument("OSCAR state image has trailing bytes"); }
    result.validate();
    return result;
}

void OscarMixedTransitionCache::restore_state_image(std::span<const std::byte> image) {
    OscarMixedTransitionCache restored = from_state_image(image, asset());
    *this = std::move(restored);
    validate();
}

OscarMixedCacheFingerprint OscarMixedTransitionCache::fingerprint(
    bool include_sequence_id) const {
    validate();
    return make_fingerprint(include_sequence_id);
}

std::uint64_t OscarMixedTransitionCache::state_image_hash() const {
    const auto image = state_image();
    std::uint64_t hash = kFnvOffset;
    if (!image.empty()) { hash_bytes(hash, image.data(), image.size()); }
    return hash;
}

std::size_t OscarMixedTransitionCache::shared_page_count_with(
    const OscarMixedTransitionCache& other) const noexcept {
    const std::size_t count = std::min(page_blocks_.size(), other.page_blocks_.size());
    std::size_t shared = 0;
    for (std::size_t index = 0; index < count; ++index) {
        if (page_blocks_[index] && page_blocks_[index] == other.page_blocks_[index]) { ++shared; }
    }
    return shared;
}

std::size_t OscarMixedTransitionCache::shared_page_refcount(std::size_t page_index) const {
    if (page_index >= page_blocks_.size() || !page_blocks_[page_index]) {
        throw std::out_of_range("OSCAR transition page block is out of range");
    }
    return page_blocks_[page_index].use_count();
}

void OscarMixedTransitionCache::validate() const {
    bundle_.validate();
    if (input_archive_.size() != context_tokens()) {
        throw std::logic_error("OSCAR transition input archive/context mismatch");
    }
    for (const auto& token : input_archive_) {
        if (token.k.size() != kOscarMixedFullAttentionLayers.size() * kRowValues ||
            token.v.size() != kOscarMixedFullAttentionLayers.size() * kRowValues) {
            throw std::logic_error("OSCAR transition input archive row extent mismatch");
        }
    }
    const std::size_t pages_per_layer =
        bundle_.layers().empty() ? 0U : bundle_.layers().front().pages().size();
    if (page_blocks_.size() != bundle_.layers().size() * pages_per_layer) {
        throw std::logic_error("OSCAR transition page block count mismatch");
    }
    std::size_t block_index = 0;
    for (const auto& layer : bundle_.layers()) {
        for (const auto& page : layer.pages()) {
            const auto& block = page_blocks_.at(block_index++);
            if (!block || !same_storage(*block, page.storage)) {
                throw std::logic_error("OSCAR transition page block diverges from typed page");
            }
        }
    }
}

void OscarMixedTransitionCache::refresh_page_blocks() {
    std::vector<PageStoragePtr> old_blocks = page_blocks_;
    std::vector<PageStoragePtr> next_blocks;
    for (const auto& layer : bundle_.layers()) {
        for (const auto& page : layer.pages()) {
            const std::size_t index = next_blocks.size();
            if (index < old_blocks.size() && old_blocks[index] &&
                same_storage(*old_blocks[index], page.storage)) {
                next_blocks.push_back(old_blocks[index]);
            } else {
                next_blocks.push_back(std::make_shared<const PageStorage>(page.storage));
            }
        }
    }
    page_blocks_ = std::move(next_blocks);
}

bool OscarMixedTransitionCache::same_storage(const PageStorage& left,
                                             const PageStorage& right) noexcept {
    if (left.index() != right.index()) { return false; }
    if (const auto* left_bf16 = std::get_if<OscarMixedBFloat16PageStorage>(&left)) {
        const auto* right_bf16 = std::get_if<OscarMixedBFloat16PageStorage>(&right);
        return right_bf16 != nullptr && left_bf16->k == right_bf16->k &&
               left_bf16->v == right_bf16->v;
    }
    const auto* left_int2 = std::get_if<OscarMixedInt2G128PageStorage>(&left);
    const auto* right_int2 = std::get_if<OscarMixedInt2G128PageStorage>(&right);
    return left_int2 != nullptr && right_int2 != nullptr &&
           left_int2->k_packed == right_int2->k_packed &&
           left_int2->v_packed == right_int2->v_packed &&
           left_int2->k_scales_zeros == right_int2->k_scales_zeros &&
           left_int2->v_scales_zeros == right_int2->v_scales_zeros;
}

OscarMixedCacheFingerprint OscarMixedTransitionCache::make_fingerprint(
    bool include_sequence_id) const {
    OscarMixedCacheFingerprint result;
    result.sequence_id = include_sequence_id ? sequence_id() : 0;
    result.context_tokens = context_tokens();
    result.full_attention_layers = static_cast<std::uint32_t>(bundle_.layers().size());
    result.overall_hash = kFnvOffset;
    hash_scalar(result.overall_hash, result.sequence_id);
    hash_scalar(result.overall_hash, result.context_tokens);
    hash_scalar(result.overall_hash, result.full_attention_layers);

    for (const auto& layer : bundle_.layers()) {
        for (std::uint32_t page_index = 0; page_index < layer.pages().size(); ++page_index) {
            const auto& page = layer.pages()[page_index];
            OscarMixedPageFingerprint fingerprint;
            fingerprint.model_layer = page.metadata.model_layer;
            fingerprint.page_index = page_index;
            fingerprint.sequence_id = include_sequence_id ? page.metadata.sequence_id : 0;
            fingerprint.logical_token_begin = page.metadata.logical_token_begin;
            fingerprint.logical_token_end = page.metadata.logical_token_end;
            fingerprint.physical_token_begin = page.metadata.physical_token_begin;
            fingerprint.physical_token_end = page.metadata.physical_token_end;
            fingerprint.occupied_tokens = page.metadata.occupied_tokens;
            fingerprint.capacity_tokens = page.metadata.capacity_tokens;
            fingerprint.layout_version = page.metadata.layout_version;
            fingerprint.group_size = page.metadata.group_size;
            fingerprint.k_storage = page.metadata.k_storage;
            fingerprint.v_storage = page.metadata.v_storage;
            fingerprint.role = page.metadata.role;
            fingerprint.k_payload_hash = page_payload_hash(page.storage, true);
            fingerprint.v_payload_hash = page_payload_hash(page.storage, false);
            fingerprint.metadata_hash = page_metadata_hash(page.storage);
            fingerprint.slot_hash = page_slot_hash(page, include_sequence_id);
            result.pages.push_back(fingerprint);

            hash_page_identity(result.overall_hash, page.metadata, include_sequence_id);
            hash_scalar(result.overall_hash, page_index);
            hash_scalar(result.overall_hash, fingerprint.k_payload_hash);
            hash_scalar(result.overall_hash, fingerprint.v_payload_hash);
            hash_scalar(result.overall_hash, fingerprint.metadata_hash);
            hash_scalar(result.overall_hash, fingerprint.slot_hash);
        }
    }
    return result;
}

} // namespace ninfer
