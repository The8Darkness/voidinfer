#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <optional>
#include <cmath>

namespace ninfer::exl3 {
// Bounded token-history proposal index. These tokens are never authority.
// Index the final token into 256 buckets, then compare the full four-token key;
// capped chain visits bound even adversarial repeated-token histories.
class Exl3SuffixProposer {
public:
    static constexpr std::size_t capacity=4096;
    struct Proposal {std::array<std::int64_t,8> tokens{};std::size_t rows=0;};
    struct Outcome {unsigned proposed_suffix=0,accepted_suffix=0,replay_rows=0;};
    struct OutcomeTotals {std::uint64_t rounds=0,proposed=0,accepted=0,replay=0,skipped_neural_blocks=0;};
    struct SelectionLimits {
        unsigned minimum_rounds=0,retry_after_neural_publications=0;
        double minimum_useful_rows=0,maximum_replay_rows=0;
        void validate() const {
            if(minimum_rounds<1 || minimum_rounds>16 || !retry_after_neural_publications ||
                !std::isfinite(minimum_useful_rows) || minimum_useful_rows<0 || minimum_useful_rows>7 ||
                !std::isfinite(maximum_replay_rows) || maximum_replay_rows<0)
                throw std::invalid_argument("suffix selection limits");
        }
    };
    void set_selection_limits(SelectionLimits value) {
        if(selection_limits_)throw std::invalid_argument("suffix selection limits already bound for request");
        value.validate();
        selection_limits_=value;
    }
    bool has_selection_limits() const noexcept {return selection_limits_.has_value();}
    bool eligible() const noexcept {
        if(!selection_limits_)return true;
        const auto totals=outcome_totals();const auto& limits=*selection_limits_;
        if(totals.rounds<limits.minimum_rounds)return true; // preserved startup path
        if(neural_since_suffix_>=limits.retry_after_neural_publications)return true;
        return double(totals.accepted)/totals.rounds>=limits.minimum_useful_rows &&
            double(totals.replay)/totals.rounds<=limits.maximum_replay_rows;
    }
    void record_neural_publication() noexcept {
        if(neural_since_suffix_<std::numeric_limits<unsigned>::max())++neural_since_suffix_;
    }
    // Fixed recent publication window; lookup hits and aborted attempts do not
    // enter it. Each record represents one matched suffix route skipping B8.
    void record_published_outcome(Outcome outcome) noexcept {
        neural_since_suffix_=0;
        outcomes_[outcome_next_]=outcome;
        outcome_next_=(outcome_next_+1)%outcomes_.size();
        outcome_count_=std::min(outcome_count_+1,outcomes_.size());
    }
    OutcomeTotals outcome_totals() const noexcept {
        OutcomeTotals result;result.rounds=outcome_count_;result.skipped_neural_blocks=outcome_count_;
        for(std::size_t i=0;i<outcome_count_;++i) {
            result.proposed+=outcomes_[i].proposed_suffix;result.accepted+=outcomes_[i].accepted_suffix;
            result.replay+=outcomes_[i].replay_rows;
        }
        return result;
    }
    explicit Exl3SuffixProposer(unsigned key_tokens=4):key_tokens_(key_tokens){
        if(key_tokens_<2 || key_tokens_>4)
            throw std::invalid_argument("suffix key tokens must be2..4");
        reset();
    }
    unsigned key_tokens() const noexcept {return key_tokens_;}
    void reset() noexcept {
        selection_limits_.reset();neural_since_suffix_=0;
        outcome_next_=outcome_count_=0;
        end_=0;heads_.fill(-1);
        for(auto& entry:entries_) entry={-1,-1};
    }
    void append(std::span<const std::int64_t> tokens) {
        if(tokens.size()>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()-end_))
            throw std::overflow_error("suffix history position");
        for(auto token:tokens) if(token<0 || token>=248320)
            throw std::invalid_argument("suffix history token");
        // No allocations or exceptions after preflight, so published history
        // can advance without a second fallible state transaction.
        for(auto token:tokens) {
            const auto position=end_++;
            history_[position%capacity]=token;
            if(position>=static_cast<std::int64_t>(key_tokens_-1)) {
                const auto bucket=static_cast<std::size_t>(token)&255;
                entries_[position%capacity]={position,heads_[bucket]};
                heads_[bucket]=position;
            }
        }
    }
    Proposal propose(std::int64_t authoritative_seed,int maximum_rows) const {
        if(authoritative_seed<0 || authoritative_seed>=248320 || maximum_rows<1 || maximum_rows>8)
            throw std::invalid_argument("suffix proposal seed/extent");
        Proposal result;
        if(end_<static_cast<std::int64_t>(key_tokens_-1) || maximum_rows==1)
            return result;
        const auto oldest=std::max<std::int64_t>(0,end_-static_cast<std::int64_t>(capacity));
        auto cursor=heads_[static_cast<std::size_t>(authoritative_seed)&255];
        for(int visited=0;
            cursor>=oldest+static_cast<std::int64_t>(key_tokens_-1) && visited<64;
            ++visited) {
            const auto& entry=entries_[cursor%capacity];
            if(entry.end!=cursor) break; // overwritten/aged chain; safe neural fallback
            bool matches=cursor+1<end_ && at(cursor)==authoritative_seed;
            for(unsigned back=1;back<key_tokens_ && matches;++back)
                matches=at(cursor-static_cast<std::int64_t>(back))==
                    at(end_-static_cast<std::int64_t>(back));
            if(matches) {
                const auto available=static_cast<std::size_t>(std::min<std::int64_t>(maximum_rows,end_-cursor));
                if(available>result.rows) {
                    result.tokens[0]=authoritative_seed;result.rows=1;
                    for(auto next=cursor+1;next<end_ && result.rows<available;++next)
                        result.tokens[result.rows++]=at(next);
                    if(result.rows==static_cast<std::size_t>(maximum_rows)) return result;
                }
            }
            cursor=entry.previous;
        }
        return result;
    }
private:
    unsigned key_tokens_=4;
    std::optional<SelectionLimits> selection_limits_;
    unsigned neural_since_suffix_=0;
    std::array<Outcome,16> outcomes_{};
    std::size_t outcome_next_=0,outcome_count_=0;
    struct Entry {std::int64_t end=-1,previous=-1;};
    std::array<std::int64_t,capacity> history_{};
    std::array<Entry,capacity> entries_{};
    std::array<std::int64_t,256> heads_{};
    std::int64_t end_=0;
    std::int64_t at(std::int64_t p) const noexcept {return history_[p%capacity];}
};
} // namespace ninfer::exl3
