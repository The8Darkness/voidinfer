#include "exl3_frontend_resources.h"

#include "targets/qwen3_6/impl/frontend/digest.h"

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace voidinfer::exl3 {
namespace {

using Resources = targets::qwen3_6::FrontendResources;

struct PinnedFile {
    const char* name;
    std::size_t bytes;
    const char* sha256;
    std::string Resources::* destination;
    // Weights-specific metadata of the SC_5.00bpw_H6_V6 revision (nullptr:
    // byte-identical across the pinned revisions).
    const char* sha256_5bpw = nullptr;
};

// Exact byte counts bound allocation and are part of this one pinned identity.
constexpr std::array<PinnedFile, 8> files{{
    {"config.json", 4647,
     "08f73b0d24188776fa278306031a93b863bc6d2b0ac837e24c6ed792b30b3081", nullptr,
     "061923b83b7b3289f3f647a823c75df549fff05d9bf35b84c753a9e4c01d4d84"},
    {"model.safetensors.index.json", 295620,
     "d5bc6446c36d53525403d22002bd67aa735bfc2d0eae7cab1f498a78d406b110", nullptr,
     "62e4c2fcd6d1a4c08f4eef1780c40f0e4370664e2a095a357dc56741c64a5dc1"},
    {"tokenizer.json", 12809320,
     "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
     &Resources::tokenizer_json},
    {"tokenizer_config.json", 17928,
     "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
     &Resources::tokenizer_config_json},
    {"chat_template.jinja", 8952,
     "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
     &Resources::chat_template_jinja},
    {"generation_config.json", 202,
     "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
     &Resources::generation_config_json},
    {"preprocessor_config.json", 390,
     "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
     &Resources::preprocessor_config_json},
    {"video_preprocessor_config.json", 385,
     "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
     &Resources::video_preprocessor_config_json},
}};

// Revision: 0 = SC_6.00bpw_H6_V6, 1 = SC_5.00bpw_H6_V6, -1 = undecided.
std::string read_pinned(const std::filesystem::path& directory, const PinnedFile& pin,
                        int& revision) {
    const auto path = directory / pin.name;
    const auto fail = [&](std::string_view reason) -> void {
        throw std::runtime_error("Pinned EXL3 frontend " + std::string(pin.name) + ": " +
                                 std::string(reason));
    };
    if (!std::filesystem::is_regular_file(path)) { fail("missing regular file"); }
    if (std::filesystem::file_size(path) != pin.bytes) { fail("byte count differs"); }
    std::ifstream input(path, std::ios::binary);
    if (!input) { fail("open failed"); }
    std::string bytes(pin.bytes, '\0');
    input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (input.gcount() != static_cast<std::streamsize>(bytes.size()) || !input) {
        fail("short or failed read");
    }
    // Detect a file that grew after the size check without unbounded allocation.
    if (input.peek() != std::char_traits<char>::eof() || input.bad()) {
        fail("trailing bytes or read failure");
    }
    namespace digest = targets::qwen3_6::frontend_internal;
    const auto hex = digest::sha256_hex(digest::sha256(std::string_view(bytes)));
    if (pin.sha256_5bpw == nullptr) {
        if (hex != pin.sha256) { fail("SHA256 differs"); }
    } else {
        const int matched = hex == pin.sha256 ? 0 : hex == pin.sha256_5bpw ? 1 : -1;
        if (matched < 0) { fail("SHA256 differs"); }
        if (revision >= 0 && revision != matched) { fail("mixed pinned revisions"); }
        revision = matched;
    }
    return bytes;
}

} // namespace

Resources load_pinned_frontend_resources(const std::filesystem::path& target_directory) {
    if (!std::filesystem::is_directory(target_directory)) {
        throw std::invalid_argument("Pinned EXL3 frontend requires a target directory");
    }
    Resources resources;
    int revision = -1;
    for (const auto& pin : files) {
        auto bytes = read_pinned(target_directory, pin, revision);
        if (pin.destination != nullptr) { resources.*(pin.destination) = std::move(bytes); }
    }
    return resources;
}

const char* pinned_weights_id(const std::filesystem::path& target_directory) {
    int revision = -1;
    (void)read_pinned(target_directory, files[0], revision);
    return revision == 1 ? "SC_5.00bpw_H6_V6" : "SC_6.00bpw_H6_V6";
}

} // namespace voidinfer::exl3
