#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::exl3 {

struct Exl3HierarchyCompletion {
    std::uint64_t request = 0;
    std::uint64_t root_version = 0;
    std::uint64_t epoch = 0;
    int first = 0;
    int rows = 0;
};

struct Exl3HierarchyPublication {
    int first = 0;
    int rows = 0;
    std::uint64_t root_version = 0;
    std::uint64_t epoch = 0;
};

struct Exl3HierarchyRunAhead {
    Exl3HierarchyCompletion dependency;
    int first = 0;
    int rows = 0;
};

// Per-request authority ledger. P/A/S/D are absolute token positions. Every
// completion carries request, root and branch epoch; stale work cannot advance
// a current frontier. L2 settlement invalidates all dependent tentative work.
class Exl3HierarchyFrontiers {
public:
    Exl3HierarchyFrontiers(std::uint64_t request,int position)
        : request_(request),published_(position),authoritative_(position),
          screened_(position),drafted_(position) {
        if(request==0 || position<0) throw std::invalid_argument("hierarchy frontier identity/position");
    }

    int published() const noexcept {return published_;}
    int authoritative() const noexcept {return authoritative_;}
    int screened() const noexcept {return screened_;}
    int drafted() const noexcept {return drafted_;}
    std::uint64_t root_version() const noexcept {return root_version_;}
    std::uint64_t epoch() const noexcept {return epoch_;}

    Exl3HierarchyCompletion begin_l0(int rows) {
        if(rows<1 || rows>8 || screened_!=drafted_)
            throw std::logic_error("hierarchy L0 lead extent/state");
        Exl3HierarchyCompletion completion=current(drafted_,rows);
        drafted_+=rows;check();return completion;
    }

    void complete_l1(const Exl3HierarchyCompletion& completion,int authorized_rows,bool repaired) {
        validate(completion);
        if(completion.first!=screened_ || authorized_rows<1 || authorized_rows>completion.rows)
            throw std::logic_error("hierarchy L1 completion range");
        screened_+=authorized_rows;
        if(repaired) drafted_=screened_;
        if(screened_>drafted_) throw std::logic_error("hierarchy L1 exceeds draft frontier");
        check();
    }

    Exl3HierarchyCompletion begin_l2() const {
        const int rows=screened_-authoritative_;
        if(rows<1 || rows>32) throw std::logic_error("hierarchy L2 range");
        return current(authoritative_,rows);
    }

    Exl3HierarchyRunAhead begin_run_ahead(
        const Exl3HierarchyCompletion& dependency,int rows) const {
        validate(dependency);
        if(dependency.first!=authoritative_ || dependency.rows!=screened_-authoritative_ ||
           rows<1 || rows>8)
            throw std::logic_error("hierarchy run-ahead dependency/range");
        return {dependency,dependency.first+dependency.rows,rows};
    }

    bool can_adopt_run_ahead(const Exl3HierarchyRunAhead& work,int committed_rows) const {
        return current_completion(work.dependency) && work.dependency.first==authoritative_ &&
            work.dependency.rows==screened_-authoritative_ &&
            committed_rows==work.dependency.rows && work.first==screened_ &&
            work.rows>=1 && work.rows<=8;
    }

    void complete_l2(const Exl3HierarchyCompletion& completion,int committed_rows) {
        validate(completion);
        if(completion.first!=authoritative_ || committed_rows<1 || committed_rows>completion.rows)
            throw std::logic_error("hierarchy L2 completion range");
        authoritative_+=committed_rows;
        screened_=authoritative_;drafted_=authoritative_;
        check();
    }

    Exl3HierarchyPublication publish_authorized() {
        if(published_>=authoritative_) throw std::logic_error("hierarchy has no authorized publication");
        Exl3HierarchyPublication publication{published_,authoritative_-published_,root_version_,epoch_};
        published_=authoritative_;check();return publication;
    }

    void rebase() {
        if(published_!=authoritative_ || authoritative_!=screened_ || screened_!=drafted_)
            throw std::logic_error("hierarchy rebase requires settled frontiers");
        ++root_version_;++epoch_;
    }

    void cancel_tentative() {
        if(published_!=authoritative_)
            throw std::logic_error("hierarchy cancel requires published authority");
        screened_=authoritative_;drafted_=authoritative_;++epoch_;check();
    }

    bool current_completion(const Exl3HierarchyCompletion& completion) const noexcept {
        return completion.request==request_ && completion.root_version==root_version_ &&
            completion.epoch==epoch_;
    }

private:
    Exl3HierarchyCompletion current(int first,int rows) const noexcept {
        return {request_,root_version_,epoch_,first,rows};
    }
    void validate(const Exl3HierarchyCompletion& completion) const {
        if(!current_completion(completion)) throw std::logic_error("hierarchy stale completion");
    }
    void check() const {
        if(!(published_<=authoritative_ && authoritative_<=screened_ && screened_<=drafted_))
            throw std::logic_error("hierarchy P/A/S/D invariant");
    }

    std::uint64_t request_=0,root_version_=1,epoch_=1;
    int published_=0,authoritative_=0,screened_=0,drafted_=0;
};

enum class Exl3HierarchyMode { window_exact, cascade_async };

struct Exl3AdaptiveSample {
    Exl3HierarchyMode mode=Exl3HierarchyMode::window_exact;
    double wall_ms=0;
    int published_rows=0;
    double pending_age_ms=0;
    int rollbacks=0;
    bool target_rejected=false;
    bool memory_ok=true;
};

// Deliberately small per-request selector. It begins on the exact window path,
// admits only a warmed cascade with two recent wins and setup headroom, and
// retains a short dwell/cooldown to avoid oscillation. Safety signals bypass
// dwell and discard private work before the caller changes mode.
class Exl3AdaptiveHierarchyPolicy {
public:
    Exl3HierarchyMode mode() const noexcept {return mode_;}
    int cooldown_rounds() const noexcept {return cooldown_rounds_;}
    int fallback_count() const noexcept {return fallback_count_;}
    const std::string& reason() const noexcept {return reason_;}

    void observe(const Exl3AdaptiveSample& sample) {
        if(!std::isfinite(sample.wall_ms) || sample.wall_ms<=0 ||
           sample.published_rows<1 || sample.published_rows>8 ||
           !std::isfinite(sample.pending_age_ms) || sample.pending_age_ms<0 ||
           sample.rollbacks<0)
            throw std::invalid_argument("adaptive hierarchy sample extent");
        const double block_ms=sample.wall_ms*4.0/sample.published_rows;
        if(sample.mode==Exl3HierarchyMode::window_exact) {
            w_block_ms_=w_block_ms_==0?block_ms:0.75*w_block_ms_+0.25*block_ms;
            if(mode_==Exl3HierarchyMode::window_exact && cooldown_rounds_>0) --cooldown_rounds_;
        } else {
            async_block_ms_=async_block_ms_==0?block_ms:0.75*async_block_ms_+0.25*block_ms;
            if(w_block_ms_>0 && block_ms<=w_block_ms_*0.97) ++consecutive_async_wins_;
            else consecutive_async_wins_=0;
        }
        if(mode_!=Exl3HierarchyMode::cascade_async) return;
        ++rounds_in_mode_;
        const bool safety=!sample.memory_ok || sample.target_rejected || sample.rollbacks>=2 ||
            (w_block_ms_>0 && sample.pending_age_ms>2*w_block_ms_);
        const bool economic=rounds_in_mode_>=4 && w_block_ms_>0 && block_ms>w_block_ms_*1.03;
        if(safety || economic) fallback(safety?"safety fallback":"economic fallback");
    }

    bool select(bool l1_ready,bool memory_ok,int remaining_rounds,double setup_ms) {
        if(!std::isfinite(setup_ms) || setup_ms<0 || remaining_rounds<0)
            throw std::invalid_argument("adaptive hierarchy selection extent");
        if(mode_==Exl3HierarchyMode::cascade_async) return true;
        if(!memory_ok) {reason_="memory pressure";return false;}
        if(!l1_ready) {reason_="L1 cold";return false;}
        if(cooldown_rounds_>0) {reason_="cooldown";return false;}
        if(remaining_rounds<4) {reason_="insufficient dwell";return false;}
        if(w_block_ms_<=0 || consecutive_async_wins_<2) {reason_="no measured cascade win";return false;}
        if(setup_ms>w_block_ms_*remaining_rounds*0.03) {reason_="setup exceeds guardrail";return false;}
        mode_=Exl3HierarchyMode::cascade_async;rounds_in_mode_=0;
        reason_="two measured wins";return true;
    }

private:
    void fallback(const char* reason) {
        mode_=Exl3HierarchyMode::window_exact;cooldown_rounds_=8;
        rounds_in_mode_=0;consecutive_async_wins_=0;++fallback_count_;reason_=reason;
    }
    Exl3HierarchyMode mode_=Exl3HierarchyMode::window_exact;
    double w_block_ms_=0,async_block_ms_=0;
    int consecutive_async_wins_=0,rounds_in_mode_=0,cooldown_rounds_=0,fallback_count_=0;
    std::string reason_="conservative start";
};

} // namespace ninfer::exl3
