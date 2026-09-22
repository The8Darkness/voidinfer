#pragma once

#include "exl3/text_model.h"
#include "exl3/branch_reference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::exl3 {

struct Exl3TurboAngleL1Decision {
    std::vector<std::int64_t> authorized_tokens;
    std::size_t accepted = 0;
    std::size_t executed_rows = 0;
    std::size_t replay_rows = 0;
    bool rejected = false;
};

struct Exl3TurboAngleL1Stats {
    double construction_ms = 0.0;
    double restore_ms = 0.0;
    double execution_ms = 0.0;
    double rollback_ms = 0.0;
    std::uint64_t construction_count = 0;
    std::uint64_t restore_count = 0;
    std::uint64_t delta_rebase_count = 0;
    std::uint64_t executed_rows = 0;
    std::uint64_t replay_rows = 0;
};

// Independent approximate verifier for one request. The immutable authoritative
// prefix is represented by full-history TurboAngle pages in DRAM. The supplied
// OSCAR-only context owns all mutable attention, FP32 GDN and BF16 convolution
// state. A short suffix checkpoint contains only lineage and accepted tokens;
// restore always reconstructs from the trusted root before replay.
class Exl3TurboAngleL1Context {
public:
    static constexpr std::size_t max_pending_rows = 32;
    struct Checkpoint {
        std::shared_ptr<const Exl3ExactHostState> root;
        std::shared_ptr<const Exl3TurboAngleWarmPages> pages;
        std::vector<std::int64_t> suffix;
    };

    Exl3TurboAngleL1Context(std::unique_ptr<Exl3TextContext> context,
                           std::shared_ptr<const Exl3ExactHostState> root)
        : context_(std::move(context)) {
        if (!context_ || !context_->oscar_only_context() || !context_->oscar_enabled())
            throw std::invalid_argument("TurboAngle L1 requires private eager OSCAR-only context");
        context_->prepare_continuation(8);
        context_->prepare_transaction();
        rebase(std::move(root));
    }

    int position() const noexcept { return context_->position(); }
    std::size_t suffix_rows() const noexcept { return suffix_.size(); }
    std::size_t page_bytes() const noexcept { return pages_ ? pages_->payload_bytes() : 0; }
    std::size_t context_bytes() const noexcept { return context_->persistent_bytes(); }
    int history_rows() const noexcept { return pages_ ? pages_->rows() : 0; }
    const Exl3TurboAngleL1Stats& stats() const noexcept { return stats_; }

    std::int64_t greedy() {
        return exl3_branch_greedy(*context_);
    }

    void advance(std::int64_t token) {
        if(token<0 || token>=248320) throw std::invalid_argument("TurboAngle L1 token extent");
        if(suffix_.size()>=max_pending_rows)
            throw std::logic_error("TurboAngle L1 suffix checkpoint extent");
        const auto start=std::chrono::steady_clock::now();
        context_->decode(token);
        stats_.execution_ms+=elapsed_ms(start);
        ++stats_.executed_rows;
        suffix_.push_back(token);
    }

    Checkpoint checkpoint() const { return {root_,pages_,suffix_}; }

    void restore(const Checkpoint& checkpoint) {
        if(!checkpoint.root || !checkpoint.pages || !checkpoint.pages->full_history() ||
           checkpoint.root->model_identity()!=context_->model_identity() ||
           checkpoint.suffix.size()>max_pending_rows)
            throw std::invalid_argument("TurboAngle L1 checkpoint identity/extent");
        rollback_pending();
        root_=checkpoint.root;pages_=checkpoint.pages;suffix_.clear();
        restore_root();
        for(auto token:checkpoint.suffix) {
            context_->decode(token);
            suffix_.push_back(token);
            ++stats_.replay_rows;
        }
    }

    void rebase(std::shared_ptr<const Exl3ExactHostState> root) {
        if(!root || root->model_identity()!=context_->model_identity() ||
           root->position()<=0 || root->position()>context_->max_context())
            throw std::invalid_argument("TurboAngle L1 rebase identity/extent");
        rollback_pending();
        const auto start=std::chrono::steady_clock::now();
        auto pages=Exl3TextContext::make_turboangle_l1_pages(root);
        stats_.construction_ms+=elapsed_ms(start);++stats_.construction_count;
        root_=std::move(root);pages_=std::move(pages);suffix_.clear();
        restore_root();
    }

    void rebase_delta(std::shared_ptr<const Exl3ExactHostState> root) {
        if(!screen_pending_ || !root || !root_->is_prefix_of(*root) ||
           root->position()<=root_->position() ||
           root->position()-root_->position()>static_cast<int>(max_pending_rows))
            throw std::invalid_argument("TurboAngle L1 delta rebase lineage/state");
        const auto build_start=std::chrono::steady_clock::now();
        auto pages=Exl3TextContext::extend_turboangle_l1_pages(root,pages_);
        stats_.construction_ms+=elapsed_ms(build_start);++stats_.construction_count;
        const auto rollback_start=std::chrono::steady_clock::now();
        context_->rollback_transaction();
        stats_.rollback_ms+=elapsed_ms(rollback_start);screen_pending_=false;
        const auto restore_start=std::chrono::steady_clock::now();
        context_->rebase_oscar_host_state_delta(*root_,*root,*pages);
        stats_.restore_ms+=elapsed_ms(restore_start);++stats_.restore_count;++stats_.delta_rebase_count;
        root_=std::move(root);pages_=std::move(pages);suffix_.clear();
    }

    Exl3TurboAngleL1Decision screen_one(std::int64_t candidate) {
        if(candidate<0 || candidate>=248320 ||
           suffix_.size()>=max_pending_rows)
            throw std::invalid_argument("TurboAngle L1 scalar candidate extent");
        Exl3TurboAngleL1Decision result;
        const auto authorized=greedy();
        if(!screen_pending_) {context_->begin_transaction();screen_pending_=true;}
        result.authorized_tokens.push_back(authorized);
        result.accepted=authorized==candidate;
        result.rejected=!result.accepted;
        advance(authorized);
        result.executed_rows=1;
        return result;
    }

    // Screen one causal block using the native 2..8-row continuation path. If
    // a mismatch occurs, discard the full teacher-forced window and replay the
    // prior screened suffix plus the valid current prefix and L1 correction
    // from the unchanged private L2-root transaction.
    Exl3TurboAngleL1Decision screen(std::span<const std::int64_t> candidates) {
        if(candidates.size()<2 || candidates.size()>8 ||
           suffix_.size()+candidates.size()>max_pending_rows)
            throw std::invalid_argument("TurboAngle L1 screen requires 2..8 rows");
        for(auto token:candidates) if(token<0 || token>=248320)
            throw std::invalid_argument("TurboAngle L1 candidate extent");
        const auto saved=checkpoint();
        Exl3TurboAngleL1Decision result;
        result.authorized_tokens.reserve(candidates.size());
        std::vector<std::int64_t> path(candidates.size());
        path[0]=greedy();
        if(!screen_pending_) {context_->begin_transaction();screen_pending_=true;}
        const auto start=std::chrono::steady_clock::now();
        context_->continue_rows(candidates);
        if(exl3_device_greedy_enabled()) {
            const auto packet=context_->greedy_packet(true);
            packet.validate(static_cast<std::uint32_t>(candidates.size()),
                context_->request_generation(),context_->position(),packet.serial);
            for(std::size_t row=1;row<path.size();++row)
                path[row]=packet.decisions[row-1].token;
        } else {
            const auto logits=context_->continuation_logits_host();
            if(logits.size()!=candidates.size()*248320 ||
               std::any_of(logits.begin(),logits.end(),
                   [](float value){return !std::isfinite(value);}))
                throw std::runtime_error("TurboAngle L1 continuation logits invalid");
            for(std::size_t row=1;row<path.size();++row) {
                const auto first=logits.begin()+(row-1)*248320;
                path[row]=std::max_element(first,first+248320)-first;
            }
        }
        stats_.execution_ms+=elapsed_ms(start);
        result.executed_rows=candidates.size();
        stats_.executed_rows+=candidates.size();
        for(std::size_t row=0;row<candidates.size();++row) {
            result.authorized_tokens.push_back(path[row]);
            if(path[row]==candidates[row]) ++result.accepted;
            else {result.rejected=true;break;}
        }
        if(!result.rejected) {
            suffix_.insert(suffix_.end(),candidates.begin(),candidates.end());
            return result;
        }
        const auto rollback_start=std::chrono::steady_clock::now();
        context_->rollback_transaction();
        stats_.rollback_ms+=elapsed_ms(rollback_start);screen_pending_=false;
        context_->begin_transaction();screen_pending_=true;
        suffix_.clear();
        for(auto token:saved.suffix) context_->decode(token);
        suffix_=saved.suffix;
        for(std::size_t row=0;row<result.accepted;++row) context_->decode(candidates[row]);
        context_->decode(result.authorized_tokens.back());
        result.replay_rows=saved.suffix.size()+result.accepted+1;
        stats_.replay_rows+=result.replay_rows;
        suffix_.insert(suffix_.end(),candidates.begin(),candidates.begin()+result.accepted);
        suffix_.push_back(result.authorized_tokens.back());
        return result;
    }

private:
    static double elapsed_ms(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    }
    void restore_root() {
        const auto start=std::chrono::steady_clock::now();
        context_->restore_oscar_host_state(*root_,pages_.get());
        stats_.restore_ms+=elapsed_ms(start);++stats_.restore_count;
    }
    void rollback_pending() {
        if(!screen_pending_) return;
        const auto start=std::chrono::steady_clock::now();
        context_->rollback_transaction();
        stats_.rollback_ms+=elapsed_ms(start);screen_pending_=false;
    }

    std::unique_ptr<Exl3TextContext> context_;
    std::shared_ptr<const Exl3ExactHostState> root_;
    std::shared_ptr<const Exl3TurboAngleWarmPages> pages_;
    std::vector<std::int64_t> suffix_;
    Exl3TurboAngleL1Stats stats_;
    bool screen_pending_=false;
};

} // namespace ninfer::exl3
