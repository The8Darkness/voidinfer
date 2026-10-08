#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::exl3 {

// Per-round DFlash2 verification tree: a draft chain prefix plus sibling
// leaves (sibling_rows.cuh), chosen to maximize the expected accepted length
// per verifier round time: each verifier row costs about kExl3TreeRowCost of
// the fixed round (0.23 ms of ~16.3 ms on RTX 5090, 64 target layers), so
// the tree may use fewer rows than the budget when leaves are unlikely.
//
// Node acceptance is estimated from the draft's unary top-16 softmax: a chain
// token's conditional hit rate by its unary probability decile, and a sibling
// leaf's by its unary rank (1..3, unary order without the chain token) and
// decile. The tables are empirical greedy-target hit rates measured over
// code/prose/translation/structured DFlash2 round logs (Qwen3.8-27B EXL3 6bpw,
// October 2026).
struct Exl3VerificationTree {
    int chain_drafts = 0;                  // chain rows 1..chain_drafts
    std::vector<std::int64_t> sibling_tokens;
    std::vector<int> sibling_offsets;      // logical offset (depth) of each leaf
};

inline constexpr float kExl3TreeRowCost = 0.0141f;
inline constexpr std::array<float, 10> kExl3TreeChainHit{
    0.32f, 0.38f, 0.35f, 0.44f, 0.42f, 0.49f, 0.57f, 0.67f, 0.78f, 0.97f};
inline constexpr std::array<std::array<float, 10>, 3> kExl3TreeSiblingHit{{
    {0.04f, 0.13f, 0.18f, 0.23f, 0.26f, 0.27f, 0.13f, 0.09f, 0.06f, 0.07f},
    {0.03f, 0.10f, 0.15f, 0.09f, 0.12f, 0.12f, 0.12f, 0.12f, 0.12f, 0.12f},
    {0.03f, 0.10f, 0.15f, 0.15f, 0.15f, 0.15f, 0.15f, 0.15f, 0.15f, 0.15f},
}};

// drafts: the chain proposals at depths 1..D; candidate_ids / candidate_unary:
// [D][16] in unary order. budget: maximum verifier rows minus the anchor (>= 1).
inline Exl3VerificationTree exl3_build_verification_tree(
    std::span<const std::int64_t> drafts, std::span<const std::int64_t> candidate_ids,
    std::span<const float> candidate_unary, int budget, int max_siblings) {
    constexpr int kTop = 16, kRanks = 3;
    const int depth = std::min<int>(static_cast<int>(drafts.size()), budget);
    struct Leaf { float value; int offset; std::int64_t token; };
    std::vector<float> chain_hit(static_cast<std::size_t>(depth));
    std::vector<std::array<Leaf, kRanks>> leaves(static_cast<std::size_t>(depth));
    const auto decile = [](float p) { return std::clamp(static_cast<int>(p * 10.0f), 0, 9); };
    for (int d = 0; d < depth; ++d) {
        const auto ids = candidate_ids.subspan(static_cast<std::size_t>(d) * kTop, kTop);
        const auto unary = candidate_unary.subspan(static_cast<std::size_t>(d) * kTop, kTop);
        const float top = *std::max_element(unary.begin(), unary.end());
        std::array<float, kTop> p{};
        float sum = 0.0f;
        for (int c = 0; c < kTop; ++c) sum += p[c] = std::exp(unary[c] - top);
        std::array<int, kTop> order{};
        for (int c = 0; c < kTop; ++c) order[c] = c;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return unary[a] > unary[b]; });
        float chain_p = 0.0f;
        for (int c = 0; c < kTop; ++c)
            if (ids[c] == drafts[static_cast<std::size_t>(d)]) chain_p = p[c] / sum;
        chain_hit[static_cast<std::size_t>(d)] = kExl3TreeChainHit[decile(chain_p)];
        int rank = 0;
        for (int i = 0; i < kTop && rank < kRanks; ++i) {
            const int c = order[i];
            if (ids[c] == drafts[static_cast<std::size_t>(d)]) continue;
            leaves[static_cast<std::size_t>(d)][rank] = {
                kExl3TreeSiblingHit[rank][decile(p[c] / sum)], d + 1, ids[c]};
            ++rank;
        }
        for (; rank < kRanks; ++rank) leaves[static_cast<std::size_t>(d)][rank] = {0.0f, d + 1, -1};
    }
    Exl3VerificationTree best;
    float best_rate = -1.0f;
    for (int nodes = 1; nodes <= budget; ++nodes)
    for (int chain = std::max(1, nodes - std::max(0, max_siblings));
         chain <= std::min(depth, nodes); ++chain) {
        // Expected accepted length: chain prefix survival plus leaves under it.
        float survive = 1.0f, value = 0.0f;
        std::vector<Leaf> pool;
        for (int d = 0; d < chain; ++d) {
            for (const auto& leaf : leaves[static_cast<std::size_t>(d)])
                if (leaf.token >= 0) pool.push_back({survive * leaf.value, leaf.offset, leaf.token});
            survive *= chain_hit[static_cast<std::size_t>(d)];
            value += survive;
        }
        const int count = nodes - chain;
        if (count > static_cast<int>(pool.size())) continue;
        std::partial_sort(pool.begin(), pool.begin() + count, pool.end(),
                          [](const Leaf& a, const Leaf& b) { return a.value > b.value; });
        for (int i = 0; i < count; ++i) value += pool[static_cast<std::size_t>(i)].value;
        // Expected committed tokens (accepted + correction/bonus) per round time.
        const float rate = (1.0f + value) / (1.0f + kExl3TreeRowCost * static_cast<float>(nodes));
        if (rate > best_rate) {
            best_rate = rate;
            best.chain_drafts = chain;
            best.sibling_tokens.clear();
            best.sibling_offsets.clear();
            for (int i = 0; i < count; ++i) {
                best.sibling_tokens.push_back(pool[static_cast<std::size_t>(i)].token);
                best.sibling_offsets.push_back(pool[static_cast<std::size_t>(i)].offset);
            }
        }
    }
    return best;
}

} // namespace ninfer::exl3
