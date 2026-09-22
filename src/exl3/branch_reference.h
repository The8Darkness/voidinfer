#pragma once

#include "exl3/text_model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ninfer::exl3 {

// Eager oscar_fast reference, not vericache_exact. Parent -1 names the
// committed context. Position is root.position()+depth-1; equal token IDs
// never identify a state. Every node includes its own token's KV/GDN/conv.
struct Exl3BranchNode { int parent; std::int64_t token; };
struct Exl3BranchReferenceResult {
    int root_position = 0;
    int committed_position = 0; // exclusive end of target-verified consumed KV
    std::int64_t pending = -1; // target-authorized, not yet consumed into state
    std::vector<std::int64_t> committed_tokens;
    std::vector<std::int64_t> next_tokens;
    std::size_t executed_rows = 0; // includes ancestors and final commit replay
};

inline std::int64_t exl3_branch_greedy(Exl3TextContext& context,
                                      cudaStream_t stream = nullptr) {
    if(exl3_device_greedy_enabled()) return context.greedy_packet(false,stream).decisions[0].token;
    const auto logits = context.logits_host(stream);
    if (logits.empty() || std::any_of(logits.begin(), logits.end(),
            [](float x) { return !std::isfinite(x); }))
        throw std::runtime_error("branch reference invalid logits");
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

// Caller exclusively owns context and its prepared transaction on the default
// stream. No draft/sampler state is touched (greedy only); draft ingestion and
// output publication may occur only after successful return. The observer is
// read-only, cannot reenter, and sees tentative node state. Throwing rolls back.
// One root checkpoint is reused; no full-state allocation per node. This is a
// deliberately slow oracle for compact verification, with all replay counted.
template<class Observe>
Exl3BranchReferenceResult verify_exl3_branch_reference(
    Exl3TextContext& context, std::span<const Exl3BranchNode> nodes,
    int max_context, Observe&& observe) {
    if (!context.transaction_prepared() || context.transaction_active() ||
        context.graph_active() || !context.oscar_enabled() || context.position() <= 0 ||
        nodes.empty() || nodes.size() > 32)
        throw std::invalid_argument("branch reference requires eager root and 1..32 nodes");
    std::vector<std::vector<int>> paths(nodes.size());
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const auto& node = nodes[i];
        if (node.parent < -1 || node.parent >= i || node.token < 0 || node.token >= 248320)
            throw std::invalid_argument("branch reference invalid parent/token");
        for (int j = 0; j < i; ++j)
            if (nodes[j].parent == node.parent && nodes[j].token == node.token)
                throw std::invalid_argument("branch reference duplicate sibling");
        if (node.parent >= 0) paths[i] = paths[node.parent];
        paths[i].push_back(i);
        if (static_cast<std::size_t>(context.position()) + paths[i].size() >
            static_cast<std::size_t>(std::max(0, max_context)))
            throw std::invalid_argument("branch reference context extent");
    }
    Exl3BranchReferenceResult result;
    result.root_position = context.position();
    result.committed_tokens.reserve(nodes.size());
    result.next_tokens.resize(nodes.size());
    result.pending = exl3_branch_greedy(context);
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        context.begin_transaction();
        try {
            for (int ancestor : paths[i]) {
                context.decode(nodes[ancestor].token);
                ++result.executed_rows;
            }
            result.next_tokens[i] = exl3_branch_greedy(context);
            observe(i, context);
            context.rollback_transaction();
        } catch (...) {
            if (context.transaction_active()) context.rollback_transaction();
            throw;
        }
    }
    int parent = -1;
    for (;;) {
        int selected = -1;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
            if (nodes[i].parent == parent && nodes[i].token == result.pending) { selected = i; break; }
        if (selected < 0) break;
        result.committed_tokens.push_back(result.pending);
        result.pending = result.next_tokens[selected];
        parent = selected;
    }
    if (!result.committed_tokens.empty()) {
        context.begin_transaction();
        try {
            for (auto token : result.committed_tokens) {
                // Check again at publication boundary, including replay determinism.
                if (exl3_branch_greedy(context) != token)
                    throw std::runtime_error("branch reference commit replay mismatch");
                context.decode(token);
                ++result.executed_rows;
            }
            if (exl3_branch_greedy(context) != result.pending)
                throw std::runtime_error("branch reference pending replay mismatch");
            context.commit_transaction();
        } catch (...) {
            if (context.transaction_active()) context.rollback_transaction();
            throw;
        }
    }
    result.committed_position = context.position();
    return result;
}

} // namespace ninfer::exl3
