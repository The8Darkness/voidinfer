#pragma once

// Donor-aligned speculative round. The anchor was emitted previously but is
// still pending. Process it first, process only matching proposals, and leave
// the replacement/full-accept bonus emitted but pending.
struct PendingRoundResult {
    std::vector<std::int64_t> emitted;
    std::int64_t pending = -1;
    int accepted = 0;
    int processed = 0;
    int rejection_index = -1;
};

template <typename SampleTarget, typename DecodeAndCommit>
PendingRoundResult verify_pending_round(
    std::int64_t pending_anchor,
    const std::vector<std::int64_t>& proposals,
    SampleTarget&& sample_target,
    DecodeAndCommit&& decode_and_commit) {
    if (pending_anchor < 0 || proposals.empty())
        throw std::invalid_argument("pending verification requires an anchor and proposals");
    PendingRoundResult result;
    decode_and_commit(pending_anchor);
    result.processed = 1;
    for (std::size_t index = 0; index < proposals.size(); ++index) {
        const std::int64_t target = sample_target();
        if (proposals[index] != target) {
            result.emitted.push_back(target);
            result.pending = target;
            result.rejection_index = static_cast<int>(index);
            return result;
        }
        result.emitted.push_back(target);
        ++result.accepted;
        decode_and_commit(target);
        ++result.processed;
    }
    result.pending = sample_target();
    result.emitted.push_back(result.pending);
    return result;
}
