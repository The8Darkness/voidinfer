#include "core/oscar_mixed_state_transitions.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace {

constexpr std::size_t kRowValues =
    static_cast<std::size_t>(ninfer::kOscarMixedKVHeads) * ninfer::kOscarMixedHeadDim;
constexpr std::size_t kLayerValues =
    kRowValues * ninfer::kOscarMixedFullAttentionLayers.size();
constexpr std::uint32_t kBaseContext = 324;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t round = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + round) >> 16U);
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
                          std::uint32_t branch_salt, bool value_stream) noexcept {
    std::uint32_t seed = 0xD2C20001U ^ branch_salt;
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

void make_rows(std::uint32_t logical_token, std::uint32_t branch_salt,
               std::vector<std::uint16_t>& k_rows, std::vector<std::uint16_t>& v_rows) {
    k_rows.resize(kLayerValues);
    v_rows.resize(kLayerValues);
    for (std::size_t layer = 0; layer < ninfer::kOscarMixedFullAttentionLayers.size(); ++layer) {
        const std::size_t layer_offset = layer * kRowValues;
        for (std::uint32_t head = 0; head < ninfer::kOscarMixedKVHeads; ++head) {
            for (std::uint32_t dimension = 0; dimension < ninfer::kOscarMixedHeadDim;
                 ++dimension) {
                const std::size_t offset = layer_offset +
                    static_cast<std::size_t>(head) * ninfer::kOscarMixedHeadDim + dimension;
                k_rows[offset] = float_to_bf16(deterministic_value(
                    layer, logical_token, head, dimension, branch_salt, false));
                v_rows[offset] = float_to_bf16(deterministic_value(
                    layer, logical_token, head, dimension, branch_salt, true));
            }
        }
    }
}

std::string hex_hash(std::uint64_t value) {
    std::ostringstream output;
    output << "0x" << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

void write_json_string(std::ofstream& output, const std::string& value) {
    output << '"';
    for (const char character : value) {
        if (character == '"' || character == '\\') { output << '\\'; }
        output << character;
    }
    output << '"';
}

void write_page_fingerprint(std::ofstream& output,
                            const ninfer::OscarMixedPageFingerprint& page) {
    output << "{\"layer\":" << page.model_layer << ",\"page\":" << page.page_index
           << ",\"sequence_id\":" << page.sequence_id
           << ",\"logical_begin\":" << page.logical_token_begin
           << ",\"logical_end\":" << page.logical_token_end
           << ",\"physical_begin\":" << page.physical_token_begin
           << ",\"physical_end\":" << page.physical_token_end
           << ",\"occupied\":" << page.occupied_tokens
           << ",\"capacity\":" << page.capacity_tokens
           << ",\"layout_version\":" << page.layout_version
           << ",\"group_size\":" << page.group_size
           << ",\"k_storage\":" << static_cast<unsigned>(page.k_storage)
           << ",\"v_storage\":" << static_cast<unsigned>(page.v_storage)
           << ",\"role\":" << static_cast<unsigned>(page.role);
    output << ",\"k_payload_hash\":";
    write_json_string(output, hex_hash(page.k_payload_hash));
    output << ",\"v_payload_hash\":";
    write_json_string(output, hex_hash(page.v_payload_hash));
    output << ",\"metadata_hash\":";
    write_json_string(output, hex_hash(page.metadata_hash));
    output << ",\"slot_hash\":";
    write_json_string(output, hex_hash(page.slot_hash));
    output << '}';
}

void write_fingerprint(std::ofstream& output, const ninfer::OscarMixedCacheFingerprint& fingerprint) {
    output << "{\"sequence_id\":" << fingerprint.sequence_id
           << ",\"context_tokens\":" << fingerprint.context_tokens
           << ",\"full_attention_layers\":" << fingerprint.full_attention_layers
           << ",\"overall_hash\":";
    write_json_string(output, hex_hash(fingerprint.overall_hash));
    output << ",\"pages\":[";
    for (std::size_t index = 0; index < fingerprint.pages.size(); ++index) {
        if (index != 0) { output << ','; }
        write_page_fingerprint(output, fingerprint.pages[index]);
    }
    output << "]}";
}

void write_accounting(std::ofstream& output, const ninfer::OscarMixedCacheAccounting& accounting) {
    output << "{\"context_tokens\":" << accounting.context_tokens
           << ",\"prefix_tokens\":" << accounting.prefix_tokens
           << ",\"historical_tokens\":" << accounting.historical_tokens
           << ",\"recent_tokens\":" << accounting.recent_tokens
           << ",\"page_count\":" << accounting.page_count
           << ",\"bf16_bytes\":" << accounting.physical_bf16_bytes
           << ",\"int2_payload_bytes\":" << accounting.physical_int2_payload_bytes
           << ",\"int2_metadata_bytes\":" << accounting.physical_int2_metadata_bytes
           << ",\"headers_bytes\":" << accounting.page_header_bytes
           << ",\"slots_bytes\":" << accounting.slot_table_bytes
           << ",\"total_bytes\":" << accounting.mixed_total_bytes << '}';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string report_path;
        for (int index = 1; index + 1 < argc; ++index) {
            if (std::string(argv[index]) == "--report") { report_path = argv[index + 1]; }
        }

        const auto asset = ninfer::OscarMixedAgingAssetContract::c4_cal30k();
        ninfer::OscarMixedTransitionCache base(1001, asset);
        std::vector<std::uint16_t> k_rows;
        std::vector<std::uint16_t> v_rows;
        for (std::uint32_t token = 0; token < kBaseContext; ++token) {
            make_rows(token, 0U, k_rows, v_rows);
            base.append(token, k_rows, v_rows);
        }
        base.validate();
        const auto base_identity = base.fingerprint(true);
        const auto base_content = base.content_fingerprint();
        require(base.context_tokens() == kBaseContext, "base context length");
        require(base.accounting().prefix_tokens == 64 && base.accounting().historical_tokens == 4 &&
                    base.accounting().recent_tokens == 256,
                "base fixture does not contain all three tiers");
        require(base.page_count() == 96, "base page count is not 16 layers times six pages");

        // Fork immediately: content is identical, sequence identity is branch-local, and every
        // immutable physical block is shared without another historical conversion.
        auto immediate_child = base.fork(1002);
        require(immediate_child.content_fingerprint() == base_content,
                "fork changed logical/cache content");
        require(immediate_child.fingerprint(true).sequence_id == 1002,
                "fork did not publish child sequence identity");
        const std::size_t initial_shared_pages = base.shared_page_count_with(immediate_child);
        require(initial_shared_pages == base.page_count(), "fork did not share every page block");
        const std::size_t initial_page_refcount = base.shared_page_refcount(0);
        require(initial_page_refcount == 2 && immediate_child.shared_page_refcount(0) == 2,
                "fork page refcount is not shared");
        require(immediate_child.aging_conversion_count() == base.aging_conversion_count(),
                "fork performed unnecessary historical conversions");
        const auto fork_base_hash = base_identity.overall_hash;

        // Divergent forced append: parent and child receive different rows. Existing immutable
        // prefix blocks remain shared; changed blocks are replaced, so no branch can mutate the
        // other branch's storage.
        auto parent_branch = base.fork(1003);
        auto divergent_child = base.fork(1004);
        for (std::uint32_t token = kBaseContext; token < kBaseContext + 4U; ++token) {
            make_rows(token, 0xA001U, k_rows, v_rows);
            parent_branch.append(token, k_rows, v_rows);
            make_rows(token, 0xB002U, k_rows, v_rows);
            divergent_child.append(token, k_rows, v_rows);
        }
        parent_branch.validate();
        divergent_child.validate();
        require(parent_branch.context_tokens() == 328 && divergent_child.context_tokens() == 328,
                "divergent branch context length");
        require(parent_branch.content_fingerprint() != divergent_child.content_fingerprint(),
                "different forced branches unexpectedly have equal content");
        require(base.context_tokens() == kBaseContext && !base.has_logical_token(327),
                "branch append corrupted the base sequence");
        require(parent_branch.has_logical_token(327) && divergent_child.has_logical_token(327),
                "branch append lost logical positions");
        require(parent_branch.shared_page_count_with(base) >=
                    ninfer::kOscarMixedFullAttentionLayers.size(),
                "copy-on-write did not preserve immutable prefix pages");
        require(divergent_child.shared_page_count_with(base) >=
                    ninfer::kOscarMixedFullAttentionLayers.size(),
                "child copy-on-write did not preserve immutable prefix pages");
        require(parent_branch.accounting().historical_tokens == 8 &&
                    parent_branch.accounting().recent_tokens == 256,
                "parent tier policy after aging is wrong");
        require(divergent_child.accounting().historical_tokens == 8 &&
                    divergent_child.accounting().recent_tokens == 256,
                "child tier policy after aging is wrong");

        // Rollback: snapshot a branch before speculative appends, cross four aging transitions,
        // then restore into the same logical branch and require the complete fingerprint back.
        auto rollback_branch = base.fork(1005);
        const auto rollback_before = rollback_branch.fingerprint(true);
        const auto rollback_image = rollback_branch.state_image();
        const auto rollback_image_hash = rollback_branch.state_image_hash();
        for (std::uint32_t token = kBaseContext; token < kBaseContext + 4U; ++token) {
            make_rows(token, 0xC003U, k_rows, v_rows);
            rollback_branch.append(token, k_rows, v_rows);
        }
        require(rollback_branch.context_tokens() == 328 &&
                    rollback_branch.accounting().historical_tokens == 8,
                "rollback speculation did not cross the aging boundary");
        rollback_branch.restore_state_image(rollback_image);
        require(rollback_branch.fingerprint(true) == rollback_before,
                "rollback did not restore the complete cache fingerprint");
        require(rollback_branch.state_image_hash() == rollback_image_hash,
                "rollback state image hash changed");
        require(!rollback_branch.has_logical_token(324) && rollback_branch.context_tokens() == 324,
                "stale speculative logical position remains addressable");
        require(rollback_branch.accounting().historical_tokens == 4 &&
                    rollback_branch.accounting().recent_tokens == 256,
                "rollback restored the wrong tier counts");
        require(rollback_branch.aging_conversion_count() == base.aging_conversion_count(),
                "rollback restored the wrong conversion count");

        // Commit: the child is from the target's exact lineage; adopting it changes identity and
        // content atomically, without replaying or duplicating the committed suffix.
        auto commit_parent = base.fork(1006);
        auto commit_child = commit_parent.fork(1007);
        for (std::uint32_t token = kBaseContext; token < kBaseContext + 4U; ++token) {
            make_rows(token, 0xD004U, k_rows, v_rows);
            commit_child.append(token, k_rows, v_rows);
        }
        commit_parent.commit_from(std::move(commit_child));
        require(commit_parent.committed() && commit_parent.sequence_id() == 1007,
                "commit did not adopt the child branch");
        require(commit_parent.context_tokens() == 328 &&
                    commit_parent.accounting().historical_tokens == 8 &&
                    commit_parent.accounting().recent_tokens == 256,
                "commit produced incorrect logical tiers");
        require(commit_parent.has_logical_token(327) && !commit_parent.has_logical_token(328),
                "commit duplicated or omitted logical positions");

        // StateImage/restore: serialize a branch with all three tiers and restore it into a fresh
        // context. The binary image includes the C4 identity and all original BF16 input rows;
        // physical INT2 pages and metadata are re-derived by the validated aging path.
        auto state_source = base.fork(1008);
        for (std::uint32_t token = kBaseContext; token < kBaseContext + 8U; ++token) {
            make_rows(token, 0xE005U, k_rows, v_rows);
            state_source.append(token, k_rows, v_rows);
        }
        const auto state_source_fp = state_source.fingerprint(true);
        const auto state_image = state_source.state_image();
        const auto state_image_hash = state_source.state_image_hash();
        auto restored = ninfer::OscarMixedTransitionCache::from_state_image(state_image, asset);
        restored.validate();
        require(restored.fingerprint(true) == state_source_fp,
                "StateImage restore changed the complete cache fingerprint");
        require(restored.state_image_hash() == state_image_hash,
                "StateImage restore is not deterministic");
        require(restored.context_tokens() == 332 && restored.accounting().historical_tokens == 12 &&
                    restored.accounting().recent_tokens == 256,
                "StateImage restore produced incorrect tiers");

        // Full layer coverage is explicit: the transition bundle accepts only the verified 16
        // full-attention layers, so no GDN recurrent state can be interpreted as OSCAR KV.
        require(base.bundle().layers().size() == ninfer::kOscarMixedFullAttentionLayers.size(),
                "transition bundle does not cover all full-attention layers");
        for (std::size_t index = 0; index < base.bundle().layers().size(); ++index) {
            require(base.bundle().layers()[index].model_layer() ==
                        ninfer::kOscarMixedFullAttentionLayers[index],
                    "transition bundle contains a non-full-attention layer");
        }

        if (!report_path.empty()) {
            std::ofstream report(report_path, std::ios::binary);
            if (!report) { throw std::runtime_error("could not open transition report"); }
            report << "{\"schema\":\"oscar-d2-2c-state-transitions-v1\",\"status\":\"PASS\""
                   << ",\"asset_identity\":\"" << asset.asset_identity << "\""
                   << ",\"base_context\":" << kBaseContext
                   << ",\"base_page_count\":" << base.page_count()
                   << ",\"full_attention_layers\":"
                   << ninfer::kOscarMixedFullAttentionLayers.size()
                   << ",\"gdn_state_in_oscar_cache\":false"
                   << ",\"initial_shared_pages\":" << initial_shared_pages
                   << ",\"fork_page_refcount\":" << initial_page_refcount
                   << ",\"fork_no_historical_reencode\":true"
                   << ",\"fork_base_hash\":";
            write_json_string(report, hex_hash(fork_base_hash));
            report << ",\"base_fingerprint\":";
            write_fingerprint(report, base_identity);
            report << ",\"divergent_parent\":{\"context\":"
                   << parent_branch.context_tokens() << ",\"historical\":"
                   << parent_branch.accounting().historical_tokens << ",\"recent\":"
                   << parent_branch.accounting().recent_tokens << ",\"shared_pages_with_base\":"
                   << parent_branch.shared_page_count_with(base) << "}"
                   << ",\"divergent_child\":{\"context\":"
                   << divergent_child.context_tokens() << ",\"historical\":"
                   << divergent_child.accounting().historical_tokens << ",\"recent\":"
                   << divergent_child.accounting().recent_tokens << ",\"shared_pages_with_base\":"
                   << divergent_child.shared_page_count_with(base) << "}"
                   << ",\"rollback\":{\"exact_fingerprint\":true,\"stale_positions_addressable\":false"
                   << ",\"pre_spec_image_bytes\":" << rollback_image.size()
                   << ",\"pre_spec_image_hash\":";
            write_json_string(report, hex_hash(rollback_image_hash));
            report << ",\"restored_context\":" << rollback_branch.context_tokens()
                   << ",\"restored_historical\":"
                   << rollback_branch.accounting().historical_tokens << '}'
                   << ",\"commit\":{\"committed\":true,\"sequence_id\":"
                   << commit_parent.sequence_id() << ",\"context\":"
                   << commit_parent.context_tokens() << ",\"historical\":"
                   << commit_parent.accounting().historical_tokens << ",\"recent\":"
                   << commit_parent.accounting().recent_tokens << '}'
                   << ",\"state_image\":{\"bytes\":" << state_image.size()
                   << ",\"hash\":";
            write_json_string(report, hex_hash(state_image_hash));
            report << ",\"fingerprint_exact\":true,\"restored_fingerprint\":";
            write_fingerprint(report, restored.fingerprint(true));
            report << ",\"accounting\":";
            write_accounting(report, restored.accounting());
            report << "}}\n";
        }

        std::cout << "OscarMixed state transitions: PASS\n"
                  << "  asset=" << asset.asset_identity << "\n"
                  << "  base context=" << base.context_tokens()
                  << " tiers=64/4/256 pages=" << base.page_count() << "\n"
                  << "  fork shared pages=" << initial_shared_pages
                  << " initial refcount(page0)=" << initial_page_refcount << "\n"
                  << "  divergent branches: parent/child context=328 historical=8 recent=256\n"
                  << "  rollback: exact fingerprint restored, stale positions addressable=no\n"
                  << "  commit: sequence=1007 context=328 historical=8 recent=256\n"
                  << "  StateImage: bytes=" << state_image.size()
                  << " exact fingerprint restore=yes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OscarMixed state transitions: FAIL: " << error.what() << '\n';
        return 1;
    }
}
