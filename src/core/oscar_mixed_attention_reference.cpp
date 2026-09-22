#include "core/oscar_mixed_attention_reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ninfer {
namespace {

constexpr std::size_t kMatrixValues =
    static_cast<std::size_t>(kOscarMixedHeadDim) * kOscarMixedHeadDim;
constexpr std::size_t kQueryValues =
    static_cast<std::size_t>(kOscarMixedQueryHeads) * kOscarMixedHeadDim;
constexpr float kAttentionScale = 1.0F / 16.0F; // 1/sqrt(256)

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

void require_finite(std::span<const float> values, const char* message) {
    for (const float value : values) {
        if (!std::isfinite(value)) { throw std::invalid_argument(message); }
    }
}

void rotate_row(const float* input, const float* matrix, bool transpose, float* output) {
    for (std::uint32_t column = 0; column < kOscarMixedHeadDim; ++column) {
        float sum = 0.0F;
        for (std::uint32_t dimension = 0; dimension < kOscarMixedHeadDim; ++dimension) {
            const float matrix_value = transpose
                ? matrix[static_cast<std::size_t>(column) * kOscarMixedHeadDim + dimension]
                : matrix[static_cast<std::size_t>(dimension) * kOscarMixedHeadDim + column];
            sum += input[dimension] * matrix_value;
        }
        output[column] = sum;
    }
}

OscarMixedReadTier tier_for_page(const OscarMixedPage& page) {
    if (page.metadata.role == OscarMixedRegionRole::ProtectedPrefix &&
        page.metadata.k_storage == OscarMixedStorageType::BFloat16 &&
        page.metadata.v_storage == OscarMixedStorageType::BFloat16) {
        return OscarMixedReadTier::ProtectedPrefixBFloat16;
    }
    if (page.metadata.role == OscarMixedRegionRole::HistoricalBulk &&
        page.metadata.k_storage == OscarMixedStorageType::OscarInt2G128 &&
        page.metadata.v_storage == OscarMixedStorageType::OscarInt2G128) {
        return OscarMixedReadTier::HistoricalOscarInt2G128;
    }
    if (page.metadata.role == OscarMixedRegionRole::RecentWindow &&
        page.metadata.k_storage == OscarMixedStorageType::BFloat16 &&
        page.metadata.v_storage == OscarMixedStorageType::BFloat16) {
        return OscarMixedReadTier::RecentBFloat16;
    }
    throw std::logic_error("OSCAR mixed attention encountered invalid page tier/type");
}

void decode_page_row(const OscarMixedPage& page, std::uint32_t page_offset,
                     std::uint32_t kv_head, bool key, float* output) {
    if (page_offset >= page.metadata.occupied_tokens || kv_head >= kOscarMixedKVHeads) {
        throw std::out_of_range("OSCAR mixed attention row is outside its page");
    }
    if (page.metadata.k_storage == OscarMixedStorageType::BFloat16 &&
        page.metadata.v_storage == OscarMixedStorageType::BFloat16) {
        const auto& storage = std::get<OscarMixedBFloat16PageStorage>(page.storage);
        const std::size_t row = static_cast<std::size_t>(page_offset) * kOscarMixedKVHeads + kv_head;
        const std::size_t begin = row * kOscarMixedHeadDim;
        const auto& source = key ? storage.k : storage.v;
        for (std::uint32_t dimension = 0; dimension < kOscarMixedHeadDim; ++dimension) {
            output[dimension] = bf16_to_float(source[begin + dimension]);
        }
        return;
    }
    if (page.metadata.k_storage != OscarMixedStorageType::OscarInt2G128 ||
        page.metadata.v_storage != OscarMixedStorageType::OscarInt2G128) {
        throw std::logic_error("OSCAR mixed attention K/V storage types disagree");
    }
    const auto& storage = std::get<OscarMixedInt2G128PageStorage>(page.storage);
    const std::size_t row = static_cast<std::size_t>(page_offset) * kOscarMixedKVHeads + kv_head;
    ops::OscarInt2G128EncodedRow encoded;
    const auto& packed = key ? storage.k_packed : storage.v_packed;
    const auto& metadata = key ? storage.k_scales_zeros : storage.v_scales_zeros;
    const std::size_t packed_begin = row * ops::kOscarInt2G128CodeBytes;
    const std::size_t metadata_begin = row * ops::kOscarInt2G128MetadataItems;
    std::copy_n(packed.begin() + packed_begin, ops::kOscarInt2G128CodeBytes,
                encoded.packed.begin());
    std::copy_n(metadata.begin() + metadata_begin, ops::kOscarInt2G128MetadataItems,
                encoded.scales_zeros.begin());
    for (std::uint32_t byte = 0; byte < ops::kOscarInt2G128CodeBytes; ++byte) {
        const std::uint8_t packed_value = encoded.packed[byte];
        encoded.symbols[byte] = packed_value & 0x3U;
        encoded.symbols[byte + 64U] = (packed_value >> 2U) & 0x3U;
        encoded.symbols[byte + 128U] = (packed_value >> 4U) & 0x3U;
        encoded.symbols[byte + 192U] = (packed_value >> 6U) & 0x3U;
    }
    ops::oscar_int2_g128_decode(encoded, output, kOscarMixedHeadDim);
}

} // namespace

OscarMixedAttentionReader::OscarMixedAttentionReader(
    const OscarMixedAgingLayerCache& cache, std::span<const float> q_original,
    std::span<const float> r_k, std::span<const float> r_v)
    : cache_(&cache), q_original_(q_original), r_k_(r_k), r_v_(r_v) {
    validate_inputs();
}

void OscarMixedAttentionReader::validate_inputs() const {
    if (cache_ == nullptr) { throw std::invalid_argument("OSCAR mixed attention cache is null"); }
    cache_->validate();
    if (q_original_.size() != kQueryValues || r_k_.size() != kMatrixValues ||
        r_v_.size() != kMatrixValues) {
        throw std::invalid_argument("OSCAR mixed attention input shape mismatch");
    }
    require_finite(q_original_, "OSCAR mixed attention Q is not finite");
    require_finite(r_k_, "OSCAR mixed attention K rotation is not finite");
    require_finite(r_v_, "OSCAR mixed attention V rotation is not finite");
}

void OscarMixedAttentionReader::read_row(std::uint32_t logical_token, std::uint32_t kv_head,
                                         bool key, float* output) const {
    const auto resolved = cache_->resolve(logical_token);
    const auto& page = cache_->pages().at(resolved.page_index);
    decode_page_row(page, resolved.metadata.page_offset, kv_head, key, output);
    require_finite(std::span<const float>(output, kOscarMixedHeadDim),
                   "OSCAR mixed attention decoded row is not finite");
}

OscarMixedAttentionTrace OscarMixedAttentionReader::read(std::uint32_t query_token) const {
    using Clock = std::chrono::steady_clock;
    const bool profiling = [] {
        const char* value = std::getenv("NINFER_OSCAR_D4_1_PROFILE");
        return value != nullptr && value[0] == '1';
    }();
    const auto total_start = profiling ? Clock::now() : Clock::time_point{};
    const auto elapsed_us = [](Clock::time_point start, Clock::time_point end) {
        return std::chrono::duration<double, std::micro>(end - start).count();
    };
    validate_inputs();
    if (query_token >= cache_->context_tokens()) {
        throw std::out_of_range("OSCAR mixed attention query token is out of range");
    }
    const std::uint32_t token_count = query_token + 1U;
    OscarMixedAttentionTrace result;
    result.model_layer = cache_->model_layer();
    result.query_token = query_token;
    result.logical_positions.resize(token_count);
    result.tiers.reserve(token_count);
    result.rotated_q.resize(kQueryValues);
    result.score_logits.resize(static_cast<std::size_t>(kOscarMixedQueryHeads) * token_count);
    result.softmax.resize(result.score_logits.size());
    result.rotated_av.resize(kQueryValues);
    result.recovered_output.resize(kQueryValues);
    for (std::uint32_t logical = 0; logical < token_count; ++logical) {
        result.logical_positions[logical] = logical;
        const auto slot = cache_->resolve(logical);
        const auto& page = cache_->pages().at(slot.page_index);
        const auto tier = tier_for_page(page);
        result.tiers.push_back(tier);
        if (tier == OscarMixedReadTier::ProtectedPrefixBFloat16) {
            ++result.prefix_count;
        } else if (tier == OscarMixedReadTier::HistoricalOscarInt2G128) {
            ++result.historical_count;
        } else {
            ++result.recent_count;
        }
    }
    const auto q_rotation_start = profiling ? Clock::now() : Clock::time_point{};
    for (std::uint32_t q_head = 0; q_head < kOscarMixedQueryHeads; ++q_head) {
        const float* q = q_original_.data() + static_cast<std::size_t>(q_head) * kOscarMixedHeadDim;
        float* rotated_q = result.rotated_q.data() + static_cast<std::size_t>(q_head) * kOscarMixedHeadDim;
        rotate_row(q, r_k_.data(), false, rotated_q);
    }
    if (profiling) { result.q_rotation_us = elapsed_us(q_rotation_start, Clock::now()); }
    require_finite(result.rotated_q, "OSCAR mixed attention rotated Q is not finite");

    std::vector<float> keys(static_cast<std::size_t>(token_count) * kOscarMixedKVHeads *
                            kOscarMixedHeadDim);
    std::vector<float> values(keys.size());
    for (std::uint32_t logical = 0; logical < token_count; ++logical) {
        const bool historical =
            result.tiers[logical] == OscarMixedReadTier::HistoricalOscarInt2G128;
        for (std::uint32_t kv_head = 0; kv_head < kOscarMixedKVHeads; ++kv_head) {
            const std::size_t row = (static_cast<std::size_t>(logical) * kOscarMixedKVHeads +
                                     kv_head) * kOscarMixedHeadDim;
            if (profiling && historical) {
                const auto start = Clock::now();
                read_row(logical, kv_head, true, keys.data() + row);
                result.int2_k_decode_us += elapsed_us(start, Clock::now());
            } else {
                read_row(logical, kv_head, true, keys.data() + row);
            }
        }
    }
    for (std::uint32_t logical = 0; logical < token_count; ++logical) {
        const bool historical =
            result.tiers[logical] == OscarMixedReadTier::HistoricalOscarInt2G128;
        for (std::uint32_t kv_head = 0; kv_head < kOscarMixedKVHeads; ++kv_head) {
            const std::size_t row = (static_cast<std::size_t>(logical) * kOscarMixedKVHeads +
                                     kv_head) * kOscarMixedHeadDim;
            if (profiling && historical) {
                const auto start = Clock::now();
                read_row(logical, kv_head, false, values.data() + row);
                result.int2_v_decode_us += elapsed_us(start, Clock::now());
            } else {
                read_row(logical, kv_head, false, values.data() + row);
            }
        }
    }

    std::vector<float> row_key(kOscarMixedHeadDim);
    std::vector<float> row_value(kOscarMixedHeadDim);
    const auto qk_start = profiling ? Clock::now() : Clock::time_point{};
    for (std::uint32_t q_head = 0; q_head < kOscarMixedQueryHeads; ++q_head) {
        const std::uint32_t kv_head = q_head / kOscarMixedGqaRatio;
        const float* q = result.rotated_q.data() + static_cast<std::size_t>(q_head) * kOscarMixedHeadDim;
        float* scores = result.score_logits.data() + static_cast<std::size_t>(q_head) * token_count;
        for (std::uint32_t logical = 0; logical < token_count; ++logical) {
            const std::size_t row = (static_cast<std::size_t>(logical) * kOscarMixedKVHeads +
                                     kv_head) * kOscarMixedHeadDim;
            std::copy_n(keys.data() + row, kOscarMixedHeadDim, row_key.data());
            float dot = 0.0F;
            for (std::uint32_t dimension = 0; dimension < kOscarMixedHeadDim; ++dimension) {
                dot += q[dimension] * row_key[dimension];
            }
            scores[logical] = dot * kAttentionScale;
        }
    }
    if (profiling) { result.qk_us = elapsed_us(qk_start, Clock::now()); }

    const auto softmax_start = profiling ? Clock::now() : Clock::time_point{};
    for (std::uint32_t q_head = 0; q_head < kOscarMixedQueryHeads; ++q_head) {
        float* scores = result.score_logits.data() + static_cast<std::size_t>(q_head) * token_count;
        float* probabilities = result.softmax.data() + static_cast<std::size_t>(q_head) * token_count;
        const float maximum = *std::max_element(scores, scores + token_count);
        float sum = 0.0F;
        for (std::uint32_t logical = 0; logical < token_count; ++logical) {
            probabilities[logical] = std::exp(scores[logical] - maximum);
            sum += probabilities[logical];
        }
        if (!std::isfinite(sum) || sum <= 0.0F) {
            throw std::runtime_error("OSCAR mixed attention softmax sum is invalid");
        }
        for (std::uint32_t logical = 0; logical < token_count; ++logical) {
            probabilities[logical] /= sum;
        }
    }
    if (profiling) { result.softmax_us = elapsed_us(softmax_start, Clock::now()); }

    const auto av_start = profiling ? Clock::now() : Clock::time_point{};
    for (std::uint32_t q_head = 0; q_head < kOscarMixedQueryHeads; ++q_head) {
        const std::uint32_t kv_head = q_head / kOscarMixedGqaRatio;
        const float* probabilities =
            result.softmax.data() + static_cast<std::size_t>(q_head) * token_count;
        float* output = result.rotated_av.data() +
                        static_cast<std::size_t>(q_head) * kOscarMixedHeadDim;
        std::fill(output, output + kOscarMixedHeadDim, 0.0F);
        for (std::uint32_t logical = 0; logical < token_count; ++logical) {
            const std::size_t row = (static_cast<std::size_t>(logical) * kOscarMixedKVHeads +
                                     kv_head) * kOscarMixedHeadDim;
            std::copy_n(values.data() + row, kOscarMixedHeadDim, row_value.data());
            for (std::uint32_t dimension = 0; dimension < kOscarMixedHeadDim; ++dimension) {
                output[dimension] += probabilities[logical] * row_value[dimension];
            }
        }
    }
    if (profiling) { result.av_us = elapsed_us(av_start, Clock::now()); }
    require_finite(result.score_logits, "OSCAR mixed attention score is not finite");
    require_finite(result.softmax, "OSCAR mixed attention softmax is not finite");
    require_finite(result.rotated_av, "OSCAR mixed attention rotated AV is not finite");
    const auto inverse_start = profiling ? Clock::now() : Clock::time_point{};
    for (std::uint32_t q_head = 0; q_head < kOscarMixedQueryHeads; ++q_head) {
        const float* input = result.rotated_av.data() + static_cast<std::size_t>(q_head) * kOscarMixedHeadDim;
        float* output = result.recovered_output.data() + static_cast<std::size_t>(q_head) * kOscarMixedHeadDim;
        rotate_row(input, r_v_.data(), true, output);
    }
    if (profiling) {
        result.rv_inverse_us = elapsed_us(inverse_start, Clock::now());
        result.total_us = elapsed_us(total_start, Clock::now());
    }
    require_finite(result.recovered_output, "OSCAR mixed attention recovered output is not finite");
    return result;
}

} // namespace ninfer
