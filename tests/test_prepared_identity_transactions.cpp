#include "exl3/prepared_identity.h"
#include <limits>
#include <iostream>

namespace family = ninfer::targets::qwen3_6;
using family::detail::ResidentPrefixIdentity;
using family::detail::PrefixShortlistDigests;

static void need(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> static void overflow(F&& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::overflow_error&) { rejected = true; }
    need(rejected, "expected whole-append overflow refusal");
}

int main() {
    try {
        family::PreparedPromptData prompt;
        prompt.token_ids = {7, 8};
        prompt.token_types = {0, 0};
        prompt.positions = {0, 1, 0, 1, 0, 1};
        ResidentPrefixIdentity identity;
        PrefixShortlistDigests shortlist;
        identity.assign(prompt);
        shortlist.assign(prompt);
        const auto original = identity;
        const auto digest = shortlist.at(2);
        const std::array<ninfer::TokenId, 2> suffix{9, 10};
        // The first new position fits, the second overflows. Neither operation
        // may retain the first position/token when the complete append fails.
        const auto delta = std::numeric_limits<std::int32_t>::max() - 2;
        overflow([&] { identity.append_generated(2, delta); });
        overflow([&] { shortlist.append_generated(suffix, delta); });
        need(identity.equals(original), "failed append changed identity");
        need(shortlist.size() == 2 && shortlist.at(2) == digest,
             "failed append changed shortlist");
        identity.append_generated(2, 0);
        shortlist.append_generated(suffix, 0);
        auto expected = prompt;
        expected.token_ids = {7, 8, 9, 10};
        expected.token_types = {0, 0, 0, 0};
        expected.positions = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
        PrefixShortlistDigests rebuilt;
        rebuilt.assign(expected);
        need(identity.matches(expected, 4) && shortlist.at(4) == rebuilt.at(4),
             "retry after rejected append differs from fresh construction");

        family::VisionItem item;
        item.grid = {1, 2, 2};
        item.patch_count = 4;
        item.token_spans = {{0, 2}};
        item.timestamps = {0.0};
        prompt.vision_items = {item};
        ninfer::exl3::Exl3PreparedIdentity bf16(prompt);
        prompt.vision_items[0].patch_storage = ninfer::VisionPatchStorage::Float16;
        ninfer::exl3::Exl3PreparedIdentity fp16(prompt);
        need(!bf16.equals(fp16), "different encoder representations alias");
        need(!bf16.append_text(1)->equals(*fp16.append_text(1)),
             "generated append lost encoder representation");
        need(!fp16.prefix_equals(fp16, 1) && fp16.prefix_equals(fp16, 2),
             "media frontier must remain complete");
        std::size_t allocations = 0;
        fp16.visit_allocations([&](const void* address, std::size_t bytes) {
            need(address && bytes, "invalid retained allocation");
            ++allocations;
        });
        need(allocations >= 8, "nested spans/timestamps missing from accounting");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
