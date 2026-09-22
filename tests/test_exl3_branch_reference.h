#pragma once
#include "exl3/branch_reference.h"

void run_branch_reference_qualification(Exl3TextModel& target,
                                       const std::vector<std::int64_t>& source) {
    using ninfer::exl3::Exl3BranchNode;
    using ninfer::exl3::verify_exl3_branch_reference;
    require(source.size() >= 600 && target.max_context() == 1024,
            "branchref requires real fixture and maxctx1024");
    auto candidate = target.create_context(true);
    auto reference = target.create_context(true);
    require(candidate->try_enable_oscar_from_environment() &&
            reference->try_enable_oscar_from_environment(), "branchref OSCAR");
    candidate->prepare_transaction();
    reference->prepare_transaction();
    int cases = 0, observed = 0;
    std::size_t executed = 0;
    for (const int prefix : {63, 319, 575}) {
        candidate->reset(); reference->reset();
        const std::vector<std::int64_t> prompt(source.begin(), source.begin() + prefix);
        ingest_prefix(*candidate, prompt, CommitSink{});
        ingest_prefix(*reference, prompt, CommitSink{});
        const auto root = prefix_retention_snapshot(*candidate);
        require(root == prefix_retention_snapshot(*reference), "branchref root equality");
        const auto first = sample_target(*reference);
        reference->begin_transaction();
        reference->decode(first);
        const auto second = sample_target(*reference);
        reference->decode(second);
        const auto third = sample_target(*reference);
        reference->rollback_transaction();
        const auto wrong = (first + 1) % kVocab;
        // Equal child tokens under different ancestors must remain different states.
        const std::vector<Exl3BranchNode> nodes{
            {-1, wrong}, {-1, first}, {0, second}, {1, second},
            {2, third}, {3, third}, {0, (second + 1) % kVocab}};
        std::vector<std::byte> wrong_recurrent;
        const auto observer = [&](int node, Exl3TextContext& ctx) {
            require(ctx.transaction_active(), "branchref observer must be tentative");
            // Independent oracle resets and reingests the entire prefix. It does
            // not share the candidate checkpoint or cached branch state.
            reference->reset();
            ingest_prefix(*reference, prompt, CommitSink{});
            std::vector<int> chain;
            for (int p = node; p >= 0; p = nodes[p].parent) chain.push_back(p);
            for (auto it = chain.rbegin(); it != chain.rend(); ++it)
                reference->decode(nodes[*it].token);
            const auto actual = prefix_retention_snapshot(ctx);
            require(actual == prefix_retention_snapshot(*reference),
                    "branchref intermediate complete state mismatch node=" + std::to_string(node));
            if (node == 2) wrong_recurrent = actual.recurrent;
            if (node == 3) require(wrong_recurrent != actual.recurrent,
                    "branchref equal tokens under unequal ancestors collapsed");
            ++observed;
        };
        const auto result = verify_exl3_branch_reference(*candidate, nodes, 1024, observer);
        require(result.committed_tokens == std::vector<std::int64_t>{first, second, third},
                "branchref target-authorized selection");
        require(result.root_position == prefix && result.committed_position == prefix + 3 &&
                result.executed_rows == 17 && !candidate->transaction_active(),
                "branchref accounting/commit boundary");
        reference->reset(); ingest_prefix(*reference, prompt, CommitSink{});
        for (auto token : result.committed_tokens) reference->decode(token);
        require(prefix_retention_snapshot(*candidate) == prefix_retention_snapshot(*reference),
                "branchref accepted-path state");
        require(result.pending == sample_target(*reference), "branchref pending anchor");
        candidate->decode(result.pending); reference->decode(result.pending);
        require(prefix_retention_snapshot(*candidate) == prefix_retention_snapshot(*reference),
                "branchref post-commit continuation");
        executed += result.executed_rows;
        ++cases;

        const auto stable = prefix_retention_snapshot(*candidate);
        const auto other = prefix_retention_snapshot(*reference);
        bool injected = false;
        try {
            verify_exl3_branch_reference(*candidate, nodes, 1024,
                [&](int node, Exl3TextContext&) {
                    require(prefix_retention_snapshot(*reference) == other,
                            "branchref request isolation");
                    if (node == 4) { injected = true; throw std::runtime_error("injected cancellation"); }
                });
        } catch (const std::exception&) { require(injected, "branchref unexpected cancellation failure"); }
        require(injected && !candidate->transaction_active() &&
                prefix_retention_snapshot(*candidate) == stable,
                "branchref cancellation failed to restore complete root");
        ++cases;
        const std::vector<Exl3BranchNode> rejected{{-1, (sample_target(*candidate) + 1) % kVocab}};
        const auto none = verify_exl3_branch_reference(*candidate, rejected, 1024,
                [](int, Exl3TextContext&) {});
        require(none.committed_tokens.empty() && none.executed_rows == 1 &&
                prefix_retention_snapshot(*candidate) == stable,
                "branchref total rejection mutated root");
        ++cases;
        for (const std::vector<Exl3BranchNode> invalid : {
                std::vector<Exl3BranchNode>{{0, first}},
                std::vector<Exl3BranchNode>{{-1, first}, {-1, first}},
                std::vector<Exl3BranchNode>{{-1, kVocab}}}) {
            prefix_retention_reject_unchanged(*candidate, [&] {
                verify_exl3_branch_reference(*candidate, invalid, 1024,
                        [](int, Exl3TextContext&) {});
            }, "branchref invalid topology");
            ++cases;
        }
    }
    std::cout << "BRANCH_REFERENCE PASS cases=" << cases << " observed_nodes=" << observed
              << " selection_executed_rows=" << executed
              << " transaction_bytes=" << candidate->transaction_bytes()
              << " identity=oscar_fast greedy=1 full_state_per_node=0\n";
}
