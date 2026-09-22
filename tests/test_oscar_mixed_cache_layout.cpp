#include "core/oscar_mixed_cache_layout.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void write_accounting_json(std::ofstream& out, const ninfer::OscarMixedCacheAccounting& value) {
    out << "{\n"
        << "    \"context_tokens\": " << value.context_tokens << ",\n"
        << "    \"full_attention_layers\": " << value.full_attention_layers << ",\n"
        << "    \"prefix_tokens\": " << value.prefix_tokens << ",\n"
        << "    \"historical_tokens\": " << value.historical_tokens << ",\n"
        << "    \"recent_tokens\": " << value.recent_tokens << ",\n"
        << "    \"page_count\": " << value.page_count << ",\n"
        << "    \"bf16_page_count\": " << value.bf16_page_count << ",\n"
        << "    \"int2_page_count\": " << value.int2_page_count << ",\n"
        << "    \"logical_bf16_bytes\": " << value.logical_bf16_bytes << ",\n"
        << "    \"logical_int2_payload_bytes\": " << value.logical_int2_payload_bytes << ",\n"
        << "    \"logical_int2_metadata_bytes\": " << value.logical_int2_metadata_bytes << ",\n"
        << "    \"physical_bf16_bytes\": " << value.physical_bf16_bytes << ",\n"
        << "    \"physical_int2_payload_bytes\": " << value.physical_int2_payload_bytes << ",\n"
        << "    \"physical_int2_metadata_bytes\": " << value.physical_int2_metadata_bytes << ",\n"
        << "    \"page_header_bytes\": " << value.page_header_bytes << ",\n"
        << "    \"slot_table_bytes\": " << value.slot_table_bytes << ",\n"
        << "    \"mixed_total_bytes\": " << value.mixed_total_bytes << ",\n"
        << "    \"historical_bulk_total_bytes\": " << value.historical_bulk_total_bytes << ",\n"
        << "    \"logical_value_count\": " << value.logical_value_count << ",\n"
        << std::setprecision(12)
        << "    \"raw_int2_bytes_per_value\": " << value.raw_int2_bytes_per_value << ",\n"
        << "    \"historical_bulk_bytes_per_value\": " << value.historical_bulk_bytes_per_value << ",\n"
        << "    \"mixed_bytes_per_value\": " << value.mixed_bytes_per_value << ",\n"
        << "    \"historical_bulk_bits_per_value\": " << value.historical_bulk_bits_per_value << ",\n"
        << "    \"mixed_bits_per_value\": " << value.mixed_bits_per_value << "\n"
        << "  }";
}

void print_accounting(const ninfer::OscarMixedCacheAccounting& value) {
    std::cout << "context=" << value.context_tokens << " layers=" << value.full_attention_layers
              << " prefix=" << value.prefix_tokens << " bulk=" << value.historical_tokens
              << " recent=" << value.recent_tokens << " pages=" << value.page_count
              << " (bf16=" << value.bf16_page_count << ",int2=" << value.int2_page_count << ")\n"
              << "  logical_bf16=" << value.logical_bf16_bytes
              << " logical_int2_payload=" << value.logical_int2_payload_bytes
              << " logical_int2_metadata=" << value.logical_int2_metadata_bytes << "\n"
              << "  physical_bf16=" << value.physical_bf16_bytes
              << " physical_int2_payload=" << value.physical_int2_payload_bytes
              << " physical_int2_metadata=" << value.physical_int2_metadata_bytes
              << " page_headers=" << value.page_header_bytes
              << " slot_table=" << value.slot_table_bytes
              << " total=" << value.mixed_total_bytes << "\n"
              << std::setprecision(8)
              << "  raw_int2_B/value=" << value.raw_int2_bytes_per_value
              << " historical_bulk_B/value=" << value.historical_bulk_bytes_per_value
              << " mixed_B/value=" << value.mixed_bytes_per_value
              << " mixed_bits/value=" << value.mixed_bits_per_value << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string report_path;
        if (argc == 3 && std::string(argv[1]) == "--report") { report_path = argv[2]; }
        else if (argc != 1) {
            std::cerr << "usage: ninfer_oscar_mixed_cache_layout_test [--report <path>]\n";
            return 2;
        }

        const std::uint64_t sequence_id = 42;
        constexpr std::uint32_t context_tokens = 500;
        ninfer::OscarMixedCacheBundle bundle(
            sequence_id, context_tokens, ninfer::kOscarMixedFullAttentionLayers);
        bundle.validate();
        require(bundle.layers().size() == 16, "full-attention layer count mismatch");

        const auto& layer3 = bundle.layer(3);
        require(layer3.pages().size() == 8, "layer 3 page count mismatch");
        require(layer3.resolve(63).metadata.role == ninfer::OscarMixedRegionRole::ProtectedPrefix,
                "position 63 is not protected BF16");
        require(layer3.resolve(64).metadata.role == ninfer::OscarMixedRegionRole::HistoricalBulk &&
                    layer3.resolve(64).metadata.k_storage == ninfer::OscarMixedStorageType::OscarInt2G128,
                "position 64 is not INT2 historical bulk");
        require(layer3.resolve(243).metadata.role == ninfer::OscarMixedRegionRole::HistoricalBulk,
                "position 243 is not historical bulk");
        require(layer3.resolve(244).metadata.role == ninfer::OscarMixedRegionRole::RecentWindow &&
                    layer3.resolve(244).metadata.k_storage == ninfer::OscarMixedStorageType::BFloat16,
                "position 244 is not recent BF16");
        require(layer3.resolve(499).metadata.role == ninfer::OscarMixedRegionRole::RecentWindow,
                "final position is not recent BF16");

        std::vector<bool> seen(context_tokens, false);
        for (std::uint32_t token = 0; token < context_tokens; ++token) {
            const auto resolved = layer3.resolve(token);
            require(!seen[token], "logical position overlaps");
            seen[token] = true;
            require(resolved.metadata.logical_token_begin == token &&
                        resolved.metadata.logical_token_end == token + 1,
                    "logical position is not exact one-slot mapping");
            require(resolved.metadata.k_storage == resolved.metadata.v_storage,
                    "K/V storage types disagree");
            require(resolved.metadata.sequence_id == sequence_id &&
                        resolved.metadata.model_layer == 3,
                    "slot identity metadata mismatch");
        }
        for (bool value : seen) { require(value, "logical position hole"); }

        for (const auto& layer : bundle.layers()) {
            require(layer.model_layer() != 0 && layer.model_layer() != 1 &&
                        layer.model_layer() != 2,
                    "GDN layer was inserted into the mixed KV bundle");
            require(layer.pages().size() == layer3.pages().size(),
                    "full-attention layers do not share page policy");
        }

        const auto accounting = bundle.accounting();
        require(accounting.prefix_tokens == 64 && accounting.historical_tokens == 180 &&
                    accounting.recent_tokens == 256,
                "mixed region token counts are incorrect");
        require(accounting.bf16_page_count == 80 && accounting.int2_page_count == 48,
                "mixed page type counts are incorrect");
        require(accounting.raw_int2_bytes_per_value == 0.3125,
                "raw INT2 record accounting is incorrect");
        require(sizeof(ninfer::OscarMixedPageMetadata) == 56,
                "unexpected page metadata size");
        require(sizeof(ninfer::OscarMixedSlotMetadata) == 48,
                "unexpected slot metadata size");

        std::cout << "OscarMixedCache static construction: PASS\n"
                  << "  logical positions: 0..499 exactly once; boundaries 63/64 and 243/244 pass\n"
                  << "  K/V type agreement: pass; all 16 full-attention layers share policy\n"
                  << "  GDN state: absent from typed KV bundle\n"
                  << "  metadata sizes: page=" << sizeof(ninfer::OscarMixedPageMetadata)
                  << " slot=" << sizeof(ninfer::OscarMixedSlotMetadata) << " bytes\n";
        print_accounting(accounting);

        const std::vector<std::uint32_t> accounting_contexts = {
            64U, 65U, 320U, 321U, 384U, 500U, 512U, 1024U, 4096U};
        std::vector<ninfer::OscarMixedCacheAccounting> accounting_cases;
        accounting_cases.reserve(accounting_contexts.size());
        for (const std::uint32_t tokens : accounting_contexts) {
            const ninfer::OscarMixedCacheBundle case_bundle(
                sequence_id, tokens, ninfer::kOscarMixedFullAttentionLayers);
            const auto case_accounting = case_bundle.accounting();
            accounting_cases.push_back(case_accounting);
            print_accounting(case_accounting);
        }

        if (!report_path.empty()) {
            std::ofstream report(report_path, std::ios::trunc);
            require(static_cast<bool>(report), "cannot create accounting report");
            report << "{\n  \"schema\": \"oscar-d2-2a-mixed-layout-v1\",\n"
                   << "  \"static_test\": \"PASS\",\n"
                   << "  \"boundary_context_tokens\": 500,\n"
                   << "  \"boundary_prefix_tokens\": 64,\n"
                   << "  \"boundary_historical_tokens\": 180,\n"
                   << "  \"boundary_recent_tokens\": 256,\n"
                   << "  \"boundary_page_count_per_layer\": 8,\n"
                   << "  \"boundary_positions\": [63,64,243,244,499],\n"
                   << "  \"accounting\": ";
            write_accounting_json(report, accounting);
            report << ",\n  \"accounting_contexts\": [\n";
            for (std::size_t index = 0; index < accounting_cases.size(); ++index) {
                report << "    ";
                write_accounting_json(report, accounting_cases[index]);
                if (index + 1 != accounting_cases.size()) { report << ','; }
                report << '\n';
            }
            report << "  ]\n}\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OscarMixedCache static construction: FAIL: " << error.what() << '\n';
        return 1;
    }
}
