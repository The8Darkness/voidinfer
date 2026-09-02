#include "core/oscar_mixed_cache_layout.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

constexpr std::size_t kRowValues =
    static_cast<std::size_t>(ninfer::kOscarMixedKVHeads) * ninfer::kOscarMixedHeadDim;
constexpr std::size_t kLayerValues = kRowValues;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t round = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + round) >> 16U);
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

std::uint32_t mix(std::uint32_t value) noexcept {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

float deterministic_value(std::size_t layer_index, std::uint32_t logical_token,
                          std::uint32_t kv_head, std::uint32_t dimension,
                          bool value_stream) noexcept {
    std::uint32_t seed = 0xD2B20001U;
    seed ^= static_cast<std::uint32_t>(layer_index + 1U) * 0x9e3779b9U;
    seed ^= (logical_token + 17U) * 0x85ebca6bU;
    seed ^= (kv_head + 3U) * 0xc2b2ae35U;
    seed ^= (dimension + 5U) * 0x27d4eb2fU;
    if (value_stream) { seed ^= 0xA5A5A5A5U; }
    const std::int32_t centered = static_cast<std::int32_t>(mix(seed) % 4096U) - 2048;
    const float base = static_cast<float>(centered) / (value_stream ? 149.0F : 113.0F);
    const float offset = static_cast<float>(static_cast<int>(dimension % 19U) - 9) * 0.003F;
    return base + offset + (value_stream ? 0.125F : -0.25F);
}

void make_rows(std::size_t layer_index, std::uint32_t logical_token,
               std::vector<std::uint16_t>& k_rows, std::vector<std::uint16_t>& v_rows) {
    k_rows.resize(kLayerValues);
    v_rows.resize(kLayerValues);
    for (std::uint32_t head = 0; head < ninfer::kOscarMixedKVHeads; ++head) {
        for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim; ++dimension) {
            const std::size_t offset = static_cast<std::size_t>(head) * ninfer::kOscarMixedHeadDim +
                                        dimension;
            k_rows[offset] = float_to_bf16(
                deterministic_value(layer_index, logical_token, head, dimension, false));
            v_rows[offset] = float_to_bf16(
                deterministic_value(layer_index, logical_token, head, dimension, true));
        }
    }
}

std::uint32_t recent_begin(std::uint32_t context_tokens) noexcept {
    if (context_tokens <= ninfer::kOscarMixedPrefixTokens) { return context_tokens; }
    return std::max(ninfer::kOscarMixedPrefixTokens,
                    context_tokens > ninfer::kOscarMixedRecentTokens
                        ? context_tokens - ninfer::kOscarMixedRecentTokens
                        : 0U);
}

void check_roles(const ninfer::OscarMixedAgingCacheBundle& bundle,
                 std::uint32_t expected_context) {
    require(bundle.context_tokens() == expected_context, "bundle context length mismatch");
    const std::uint32_t oldest_recent = recent_begin(expected_context);
    const std::uint32_t expected_aged =
        expected_context > ninfer::kOscarMixedPrefixTokens + ninfer::kOscarMixedRecentTokens
            ? expected_context -
                  (ninfer::kOscarMixedPrefixTokens + ninfer::kOscarMixedRecentTokens)
            : 0U;
    for (const auto& layer : bundle.layers()) {
        require(layer.context_tokens() == expected_context, "layer context length mismatch");
        require(layer.aging_conversion_count() == expected_aged,
                "aging conversion count is not the policy count");
        for (std::uint32_t token = 0; token < expected_context; ++token) {
            const bool historical = token >= ninfer::kOscarMixedPrefixTokens &&
                                    token < oldest_recent;
            const auto resolved = layer.resolve(token);
            const auto& metadata = resolved.metadata;
            require(metadata.k_storage == metadata.v_storage,
                    "K/V representation differs for a logical token");
            require(layer.has_bf16_payload(token) == !historical,
                    "logical token is present in the wrong tier");
            require(layer.was_aged_once(token) == historical,
                    "aging marker disagrees with the logical tier");
            if (token < ninfer::kOscarMixedPrefixTokens) {
                require(metadata.role == ninfer::OscarMixedRegionRole::ProtectedPrefix,
                        "protected prefix changed role");
                require(!layer.was_aged_once(token), "protected prefix was aged");
            } else if (historical) {
                require(metadata.role == ninfer::OscarMixedRegionRole::HistoricalBulk &&
                            metadata.k_storage == ninfer::OscarMixedStorageType::OscarInt2G128,
                        "historical token is not official OSCAR INT2");
            } else {
                require(metadata.role == ninfer::OscarMixedRegionRole::RecentWindow &&
                            metadata.k_storage == ninfer::OscarMixedStorageType::BFloat16,
                        "recent token is not BF16");
            }
        }
    }
}

void check_parity(const ninfer::OscarMixedAgingCacheBundle& bundle, std::size_t layer_index,
                  std::uint32_t logical_token) {
    const std::uint32_t layer_number = ninfer::kOscarMixedFullAttentionLayers[layer_index];
    const auto& layer = bundle.layer(layer_number);
    std::vector<std::uint16_t> k_row;
    std::vector<std::uint16_t> v_row;
    make_rows(layer_index, logical_token, k_row, v_row);
    for (std::uint32_t head = 0; head < ninfer::kOscarMixedKVHeads; ++head) {
        std::array<float, ninfer::kOscarMixedHeadDim> k_values{};
        std::array<float, ninfer::kOscarMixedHeadDim> v_values{};
        const std::size_t begin = static_cast<std::size_t>(head) * ninfer::kOscarMixedHeadDim;
        for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim; ++dimension) {
            k_values[dimension] = bf16_to_float(k_row[begin + dimension]);
            v_values[dimension] = bf16_to_float(v_row[begin + dimension]);
        }
        const auto expected_k = ninfer::ops::oscar_int2_g128_encode(
            k_values.data(), static_cast<int>(ninfer::kOscarMixedHeadDim), 0.96F);
        const auto expected_v = ninfer::ops::oscar_int2_g128_encode(
            v_values.data(), static_cast<int>(ninfer::kOscarMixedHeadDim), 0.92F);
        const auto& actual_k = layer.historical_k(logical_token, head);
        const auto& actual_v = layer.historical_v(logical_token, head);
        require(actual_k.clipped == expected_k.clipped, "K clipped values differ from reference");
        require(actual_v.clipped == expected_v.clipped, "V clipped values differ from reference");
        require(actual_k.scales_zeros == expected_k.scales_zeros,
                "K FP32 metadata differs from reference");
        require(actual_v.scales_zeros == expected_v.scales_zeros,
                "V FP32 metadata differs from reference");
        require(actual_k.symbols == expected_k.symbols, "K symbols differ from reference");
        require(actual_v.symbols == expected_v.symbols, "V symbols differ from reference");
        require(actual_k.packed == expected_k.packed, "K packed bytes differ from reference");
        require(actual_v.packed == expected_v.packed, "V packed bytes differ from reference");
    }
}

std::uint64_t region_storage_bytes(const ninfer::OscarMixedAgingCacheBundle& bundle,
                                   ninfer::OscarMixedRegionRole role) {
    std::uint64_t total = 0;
    for (const auto& layer : bundle.layers()) {
        for (const auto& page : layer.pages()) {
            if (page.metadata.role == role) { total += page.storage_bytes(); }
        }
    }
    return total;
}

struct AgingTransition {
    std::uint32_t append_token = 0;
    std::uint32_t aged_token = 0;
    ninfer::OscarMixedCacheAccounting accounting;
    std::uint64_t bf16_prefix_bytes = 0;
    std::uint64_t bf16_recent_bytes = 0;
};

void write_accounting(std::ofstream& report, const ninfer::OscarMixedCacheAccounting& accounting,
                      std::uint32_t conversion_count, std::uint64_t bf16_prefix_bytes,
                      std::uint64_t bf16_recent_bytes) {
    report << "{\"context_tokens\":" << accounting.context_tokens
           << ",\"prefix_tokens\":" << accounting.prefix_tokens
           << ",\"historical_tokens\":" << accounting.historical_tokens
           << ",\"recent_tokens\":" << accounting.recent_tokens
           << ",\"bf16_prefix_bytes\":" << bf16_prefix_bytes
           << ",\"bf16_recent_bytes\":" << bf16_recent_bytes
           << ",\"int2_payload_bytes\":" << accounting.physical_int2_payload_bytes
           << ",\"int2_metadata_bytes\":" << accounting.physical_int2_metadata_bytes
           << ",\"page_header_bytes\":" << accounting.page_header_bytes
           << ",\"slot_table_bytes\":" << accounting.slot_table_bytes
           << ",\"mixed_total_bytes\":" << accounting.mixed_total_bytes
           << ",\"conversion_count\":" << conversion_count << '}';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string report_path;
        if (argc == 3 && std::string(argv[1]) == "--report") {
            report_path = argv[2];
        } else if (argc != 1) {
            std::cerr << "usage: ninfer_oscar_mixed_cache_aging_test [--report <path>]\n";
            return 2;
        }

        const auto asset = ninfer::OscarMixedAgingAssetContract::c4_cal30k();
        asset.validate();
        auto bad_asset = asset;
        bad_asset.asset_identity = "legacy-fixed-hadamard-q2";
        bool rejected_bad_asset = false;
        try {
            ninfer::OscarMixedAgingCacheBundle rejected(42, bad_asset,
                                                        ninfer::kOscarMixedFullAttentionLayers);
        } catch (const std::invalid_argument&) {
            rejected_bad_asset = true;
        }
        require(rejected_bad_asset, "invalid C4 asset identity was not rejected");

        std::array<std::uint32_t, 1> gdn_layer = {0};
        bool rejected_gdn = false;
        try {
            ninfer::OscarMixedAgingCacheBundle rejected(42, asset, gdn_layer);
        } catch (const std::invalid_argument&) {
            rejected_gdn = true;
        }
        require(rejected_gdn, "GDN layer was accepted into aging bundle");

        ninfer::OscarMixedAgingCacheBundle bundle(
            42, asset, ninfer::kOscarMixedFullAttentionLayers);
        std::vector<std::uint16_t> k_rows;
        std::vector<std::uint16_t> v_rows;
        std::vector<AgingTransition> transitions;

        for (std::uint32_t token = 0; token < 324; ++token) {
            k_rows.assign(ninfer::kOscarMixedFullAttentionLayers.size() * kLayerValues, 0);
            v_rows.assign(ninfer::kOscarMixedFullAttentionLayers.size() * kLayerValues, 0);
            for (std::size_t layer_index = 0;
                 layer_index < ninfer::kOscarMixedFullAttentionLayers.size(); ++layer_index) {
                std::vector<std::uint16_t> k_row;
                std::vector<std::uint16_t> v_row;
                make_rows(layer_index, token, k_row, v_row);
                const std::size_t offset = layer_index * kLayerValues;
                std::copy(k_row.begin(), k_row.end(), k_rows.begin() + offset);
                std::copy(v_row.begin(), v_row.end(), v_rows.begin() + offset);
            }
            bundle.append(token, k_rows, v_rows);
            const std::uint32_t context = token + 1U;
            check_roles(bundle, context);
            if (context == 64U || context == 320U || context >= 321U) {
                bundle.validate();
            }
            if (context >= 321U) {
                const std::uint32_t aged_token = context - 257U;
                if (aged_token <= 67U) {
                    transitions.push_back({token, aged_token, bundle.accounting(),
                                           region_storage_bytes(
                                               bundle, ninfer::OscarMixedRegionRole::ProtectedPrefix),
                                           region_storage_bytes(
                                               bundle, ninfer::OscarMixedRegionRole::RecentWindow)});
                    for (const std::size_t layer_index : {std::size_t{0}, std::size_t{8},
                                                           std::size_t{15}}) {
                        check_parity(bundle, layer_index, aged_token);
                    }
                }
            }
        }

        require(bundle.layer(3).resolve(63).metadata.role ==
                    ninfer::OscarMixedRegionRole::ProtectedPrefix,
                "token 63 is not protected after aging");
        require(bundle.layer(3).has_bf16_payload(63), "protected prefix lost BF16 payload");
        require(!bundle.layer(3).has_bf16_payload(64),
                "historical token 64 still has BF16 payload");
        require(bundle.layer(3).resolve(64).metadata.k_storage ==
                    ninfer::OscarMixedStorageType::OscarInt2G128,
                "historical token 64 is not INT2");
        bool rejected_double_conversion = false;
        try {
            bundle.layer(3).age_token(67);
        } catch (const std::logic_error&) {
            rejected_double_conversion = true;
        }
        require(rejected_double_conversion, "second conversion of token 67 was not rejected");
        bool rejected_prefix_aging = false;
        try {
            bundle.layer(3).age_token(63);
        } catch (const std::invalid_argument&) {
            rejected_prefix_aging = true;
        }
        require(rejected_prefix_aging, "protected prefix aging was not rejected");

        const auto final_accounting = bundle.accounting();
        require(final_accounting.prefix_tokens == 64 && final_accounting.historical_tokens == 4 &&
                    final_accounting.recent_tokens == 256,
                "final tier accounting is incorrect");
        require(final_accounting.physical_int2_payload_bytes ==
                    16ULL * 64ULL * 4ULL * 64ULL * 2ULL,
                "final INT2 payload accounting is incorrect");
        require(final_accounting.physical_int2_metadata_bytes ==
                    16ULL * 64ULL * 4ULL * 4ULL * 4ULL * 2ULL,
                "final INT2 metadata accounting is incorrect");
        require(bundle.layer(3).aging_conversion_count() == 4 &&
                    bundle.layer(35).aging_conversion_count() == 4 &&
                    bundle.layer(63).aging_conversion_count() == 4,
                "representative layer aging coverage is incomplete");

        std::cout << "OscarMixedCache aging: PASS\n"
                  << "  asset=" << asset.asset_identity << " calibrated=true\n"
                  << "  boundary: token 63 prefix, token 64 recent, append token 320 ages 64\n"
                  << "  aged tokens: 64,65,66,67; all 16 full-attention layers\n"
                  << "  exact K/V codec parity: layers 3,35,63; all four KV heads\n"
                  << "  final tiers: prefix=" << final_accounting.prefix_tokens
                  << " historical=" << final_accounting.historical_tokens
                  << " recent=" << final_accounting.recent_tokens << "\n"
                  << "  final bytes: bf16_prefix="
                  << region_storage_bytes(bundle, ninfer::OscarMixedRegionRole::ProtectedPrefix)
                  << " bf16_recent="
                  << region_storage_bytes(bundle, ninfer::OscarMixedRegionRole::RecentWindow)
                  << " int2_payload=" << final_accounting.physical_int2_payload_bytes
                  << " int2_metadata=" << final_accounting.physical_int2_metadata_bytes << '\n';

        if (!report_path.empty()) {
            std::ofstream report(report_path, std::ios::trunc);
            require(static_cast<bool>(report), "cannot create aging report");
            report << std::setprecision(12)
                   << "{\n  \"schema\":\"oscar-d2-2b-aging-v1\",\n"
                   << "  \"status\":\"PASS\",\n"
                   << "  \"asset_identity\":\"" << asset.asset_identity << "\",\n"
                   << "  \"model_sha256\":\"" << asset.model_sha256 << "\",\n"
                   << "  \"asset_manifest_sha256\":\"" << asset.asset_manifest_sha256
                   << "\",\n  \"rotation_mode\":\"" << asset.rotation_mode
                   << "\",\n  \"calibrated\":true,\n  \"full_attention_layers\":16,\n"
                   << "  \"layers\":[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63],\n"
                   << "  \"boundary_sequence\":{\"prefix_last\":63,\"first_recent\":64,\n"
                   << "    \"first_aging_append_token\":320,\"first_aged_token\":64,\n"
                   << "    \"subsequent_aged_tokens\":[65,66,67]},\n"
                   << "  \"codec_parity\":{\"layers\":[3,35,63],\"kv_heads\":4,\n"
                   << "    \"clips\":{\"k\":0.96,\"v\":0.92},\"symbols\":\"exact\",\n"
                   << "    \"packed_bytes\":\"exact\",\"fp32_metadata\":\"exact\"},\n"
                   << "  \"transitions\":[\n";
            for (std::size_t index = 0; index < transitions.size(); ++index) {
                const auto& transition = transitions[index];
                report << "    {\"append_token\":" << transition.append_token
                       << ",\"aged_token\":" << transition.aged_token << ",\"accounting\":";
                write_accounting(report, transition.accounting, transition.aged_token - 63U,
                                 transition.bf16_prefix_bytes, transition.bf16_recent_bytes);
                report << '}' << (index + 1U == transitions.size() ? '\n' : ',');
            }
            report << "  ],\n  \"final_accounting\":";
            write_accounting(
                report, final_accounting, 4,
                region_storage_bytes(bundle, ninfer::OscarMixedRegionRole::ProtectedPrefix),
                region_storage_bytes(bundle, ninfer::OscarMixedRegionRole::RecentWindow));
            report << ",\n  \"guards\":{\"bad_asset_rejected\":true,\"gdn_rejected\":true,\n"
                      "    \"double_conversion_rejected\":true,\"prefix_aging_rejected\":true,\n"
                      "    \"no_bf16_historical_overlap\":true,\"no_holes_or_overlap\":true}\n}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OscarMixedCache aging: FAIL: " << error.what() << '\n';
        return 1;
    }
}
