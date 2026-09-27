#pragma once

#include <fstream>
#include "exl3/dflash2_draft.h"
#include "exl3/exact_outer_reference.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::exl3 {

// One physical C1 greedy request. The caller owns the five 16-row staging
// slabs and the exclusively borrowed draft. Their lifetimes must cover finish
// or reset. The context owner is retained so no tap or seed can outlive it.
// Prefix preparation is performed by the caller; begin_prefilled validates its
// completed target/draft frontier before any output work is allowed.
class Exl3FastDeviceRound {
public:
    static constexpr std::array<int,5> tap_layers{5,19,33,47,61};
    static constexpr std::size_t tap_hidden=5120;
    static constexpr std::uint64_t coherent_policy=0x434f484552454e54ULL;
    using Clock=std::chrono::steady_clock;
    using BeforeVerify=std::function<void(Exl3TextContext&)>;
    using AfterVerify=std::function<void(Exl3TextContext&)>;
    using ChooseWidth=std::function<int(Clock::time_point,int)>;

    struct Step {
        int root_position=0,width=0;
        std::array<std::int64_t,8> proposal{};
        std::vector<std::int64_t> committed_tokens;
        Exl3OuterReferenceResult verification;
        std::size_t staged_bytes=0;
        std::size_t authorized_replay_rows=0;
        bool terminal=false,single_row=false;
        double proposal_ms=0,target_seed_ms=0,draft_api_ms=0;
        double verifier_ms=0,ring_commit_ms=0;
        Clock::time_point seed_ready{};
    };
    struct Totals {
        std::uint64_t rounds=0,verified_rows=0,replayed_rows=0;
        std::uint64_t staged_bytes=0,reused_seeds=0,seed_fallbacks=0;
        double proposal_ms=0,verifier_ms=0,ring_commit_ms=0;
    };
    struct Prepared {
        std::uint64_t ticket=0;
        Step candidate;
    };
    // Called after the ring is complete. The caller may establish its own
    // atomic resident/publication authority here. A throwing callback must
    // leave that external authority unchanged so the local rollback is valid.
    using Publish=std::function<void(const Step&)>;
    // A device pending round calls this while both numerical undo boundaries
    // are live. It may reject the round, but must not mutate external resident
    // authority, even on success. Actual Engine publication is a separate step.
    using BeforeDeviceCommit=std::function<void(const Step&)>;

    Exl3FastDeviceRound(std::shared_ptr<Exl3TextContext> context,
        Exl3Dflash2DraftModel& draft,std::array<std::uint16_t*,5> staging,
        std::uint64_t acquisition,std::uint64_t execution,
        bool reuse_seed=false,cudaStream_t stream=nullptr)
        :context_(std::move(context)),draft_(draft),staging_(staging),
         acquisition_(acquisition),execution_(execution),
         reuse_seed_(reuse_seed),stream_(stream) {
        if(!context_ || !acquisition_ || !execution_)
            throw std::invalid_argument("fast device round owner identity");
        for(auto* plane:staging_)if(!plane)
            throw std::invalid_argument("fast device round staging plane");
    }
    Exl3FastDeviceRound(const Exl3FastDeviceRound&)=delete;
    Exl3FastDeviceRound& operator=(const Exl3FastDeviceRound&)=delete;

    static constexpr int retained_start_for(int total) {
        int first=0;
        for(int forward=0;forward<total;) {
            const int forward_rows=forward==0?16:
                std::min(1024,total-forward);
            for(int row=0;row<forward_rows;row+=16) {
                const int count=std::min(16,forward_rows-row);
                if(forward+row+count<=total-2048)
                    first=forward+row+count;
            }
            forward+=forward_rows;
        }
        return first;
    }

    // Fresh 16 + 1024/tail target computation. The old rows still execute on
    // the target. Only the draft's proven-discarded whole calls omit tap work.
    // This is the same path the harness and guarded Engine must use.
    void begin_fresh(std::span<const std::int64_t> input,bool layer_major) {
        if(phase_!=Phase::unstarted)
            throw std::logic_error("fast device fresh request already began");
        if(context_->position()!=0 || input.size()<16 ||
           input.size()>static_cast<std::size_t>(context_->max_context()) ||
           (layer_major && input.size()<=1040))
            throw std::invalid_argument("fast device fresh input extent");
        phase_=Phase::executing;
        try {
            draft_.reset(stream_);
            draft_.begin_fresh_prefill(0,static_cast<long long>(input.size()),stream_);
            context_->prefill(input.first(16),stream_);
            if(layer_major && input.size()>=2064)
                draft_.skip_fresh_prefill_block(16,0,stream_);
            else commit_captured(16,0);
            if(layer_major)prefill_layer_major(input);
            else {
                for(std::size_t first=16;first<input.size();) {
                    const int rows=static_cast<int>(std::min<std::size_t>(
                        1024,input.size()-first));
                    context_->append_prefill_wide(input.subspan(first,rows),stream_);
                    commit_captured(rows,static_cast<int>(first));
                    first+=rows;
                }
            }
            draft_.finish_fresh_prefill(stream_);
            check(cudaStreamSynchronize(stream_),
                "fast device fresh target/draft completion");
            phase_=Phase::unstarted;
        }catch(...) {phase_=Phase::poisoned;throw;}
    }

    void begin_prefilled() {
        if(phase_!=Phase::unstarted)
            throw std::logic_error("fast device round already began");
        try {
            frontier_=context_->position();
            if(frontier_<=0 || draft_.ring_count()<=0 ||
               draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
                throw std::invalid_argument("fast device round prefill frontier");
            context_->prepare_continuation(8);
            context_->prepare_transaction();
            phase_=Phase::ready;
        }catch(...) {phase_=Phase::poisoned;throw;}
    }

    // Prepare only numerical work. The target holds the candidate frontier,
    // while the draft ring still ends at root_position. The caller must settle
    // or cancel this ticket before another round, finish, or owner destruction.
    // Snapshots are deliberately explicit and potentially expensive; Engine
    // integration must account for their host residency before dispatch.
    Prepared prepare_pending(int width,std::span<const std::int64_t> terminal={},
        Exl3OuterDeviceStageTimeline* timeline=nullptr,
        const BeforeVerify& before_verify={},
        const AfterVerify& after_verify={},
        const ChooseWidth& choose_width={}) {
        require_ready_width(width);
        if(pending_ticket_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("fast device pending ticket exhausted");
        phase_=Phase::executing;
        try {
            pending_root_=context_->export_exact_host_state(stream_);
            pending_ring_=draft_.export_host_ring(stream_);
            if(!pending_root_ || pending_root_->position()!=frontier_ ||
               !pending_ring_ || pending_ring_->position()!=frontier_)
                throw std::logic_error("fast device pending root snapshot frontier");
            Step value;
            value.root_position=frontier_;
            value.width=width;
            value.proposal.fill(248070);
            if(width==1)run_single(value,terminal,true);
            else run_batch(value,terminal,timeline,before_verify,after_verify,
                choose_width,true);
            if(value.committed_tokens.empty() ||
               context_->position()!=frontier_+
                   static_cast<int>(value.committed_tokens.size()) ||
               draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
                throw std::logic_error("fast device pending candidate frontier");
            pending_terminal_.assign(terminal.begin(),terminal.end());
            pending_=std::move(value);
            ++pending_ticket_;
            phase_=Phase::pending;
            return {pending_ticket_,*pending_};
        }catch(...) {
            const auto failure=std::current_exception();
            try {
                if(!pending_root_ || !pending_ring_)
                    throw std::logic_error("fast device pending snapshot incomplete");
                restore_pending_root();
                clear_pending();
                phase_=Phase::ready;
            }catch(...) {phase_=Phase::poisoned;}
            std::rethrow_exception(failure);
        }
    }

    // `retained` is the frontend-authorized prefix of candidate.committed_tokens.
    // A shortened prefix is rechecked and replayed from the exact saved root;
    // its taps replace the speculative stage before the sole draft-ring commit.
    // The returned Step keeps verification as attempted-work diagnostics, while
    // committed_tokens and terminal describe only the settled prefix.
    Step settle_pending(std::uint64_t ticket,std::size_t retained,
        const Publish& publish={}) {
        require_pending_ticket(ticket);
        if(!retained || retained>pending_->committed_tokens.size())
            throw std::invalid_argument("fast device pending retained extent");
        phase_=Phase::executing;
        try {
            Step value=*pending_;
            if(retained<value.committed_tokens.size()) {
                restore_pending_root();
                for(std::size_t row=0;row<retained;++row) {
                    const auto token=value.committed_tokens[row];
                    if(exl3_branch_greedy(*context_,stream_)!=token)
                        throw std::runtime_error("fast device pending prefix replay diverged");
                    context_->decode(token,stream_);
                    for(std::size_t tap=0;tap<5;++tap)
                        context_->copy_tap_row_to_device(tap_layers[tap],
                            staging_[tap]+row*tap_hidden,stream_);
                }
                value.committed_tokens.resize(retained);
                value.staged_bytes=retained*5*tap_hidden*sizeof(std::uint16_t);
                value.authorized_replay_rows=retained;
            }
            const auto ring_start=Clock::now();
            std::array<const std::uint16_t*,5> sources{};
            for(std::size_t tap=0;tap<5;++tap)sources[tap]=staging_[tap];
            draft_.commit_prefill_block(sources.data(),static_cast<int>(retained),
                value.root_position,stream_);
            check(cudaStreamSynchronize(stream_),"fast device pending ring completion");
            value.ring_commit_ms=elapsed(ring_start);
            if(context_->position()!=value.root_position+static_cast<int>(retained) ||
               draft_.ring_base_abs()+draft_.ring_count()!=context_->position())
                throw std::logic_error("fast device pending settlement frontier");
            value.terminal=!pending_terminal_.empty() &&
                exl3_terminal_token(value.committed_tokens.back(),pending_terminal_);
            if(publish)publish(value);
            frontier_=context_->position();
            record_totals(value);
            clear_pending();
            phase_=value.terminal?Phase::closed:Phase::ready;
            return value;
        }catch(...) {
            const auto failure=std::current_exception();
            try {
                restore_pending_root();
                clear_pending();
                phase_=Phase::ready;
            }catch(...) {phase_=Phase::poisoned;}
            std::rethrow_exception(failure);
        }
    }

    void cancel_pending(std::uint64_t ticket) {
        require_pending_ticket(ticket);
        phase_=Phase::executing;
        try {
            restore_pending_root();
            clear_pending();
            phase_=Phase::ready;
        }catch(...) {phase_=Phase::poisoned;throw;}
    }

    // Default-unused numerical pending route. Unlike prepare_pending this
    // captures no target/ring host state. The target transaction remains live
    // after verification; the saturated draft ring remains at the root.
    Prepared prepare_device_pending(int width,
        std::span<const std::int64_t> terminal={},
        Exl3OuterDeviceStageTimeline* timeline=nullptr,
        const BeforeVerify& before_verify={},
        const AfterVerify& after_verify={},
        const ChooseWidth& choose_width={}) {
        require_ready_width(width);
        if(draft_.ring_count()!=Exl3Dflash2DraftModel::ring_keep() ||
           !context_->transaction_prepared() || context_->transaction_active() ||
           !exl3_device_greedy_enabled())
            throw std::logic_error("fast device pending requires saturated ring, prepared transaction and device greedy");
        if(pending_ticket_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("fast device pending ticket exhausted");
        phase_=Phase::executing;
        try {
            Step value;
            value.root_position=frontier_;
            value.width=width;
            value.proposal.fill(248070);
            if(width==1) {
                context_->begin_transaction(stream_);
                run_single(value,terminal,true);
            } else run_batch(value,terminal,timeline,before_verify,after_verify,
                choose_width,true,Exl3OuterDeviceSettlement::DeferTargetCommit);
            if(value.committed_tokens.empty() ||
               value.committed_tokens.size()>static_cast<std::size_t>(value.width) ||
               !context_->transaction_active() ||
               context_->position()!=frontier_+
                   static_cast<int>(value.committed_tokens.size()) ||
               draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
                throw std::logic_error("fast device pending candidate frontier");
            device_pending_terminal_.assign(terminal.begin(),terminal.end());
            device_pending_=std::move(value);
            ++pending_ticket_;
            phase_=Phase::device_pending;
            return {pending_ticket_,*device_pending_};
        } catch(...) {
            const auto failure=std::current_exception();
            try {
                restore_device_pending_root(false);
                clear_device_pending();
                phase_=Phase::ready;
            } catch(...) {
                clear_device_pending();
                phase_=Phase::poisoned;
            }
            std::rethrow_exception(failure);
        }
    }

    // `retained` must be the exact nonempty OutputSession-approved prefix.
    // Use cancel_device_pending for an approved zero-token boundary. A shorter
    // rejected candidate is rolled back and independently rechecked/replayed,
    // because the verifier already consumed its one-shot retained-prefix source.
    // `before_commit` is validation only; this method does not publish a
    // VeriCache child or make numeric settlement atomic with Engine output.
    Step settle_device_pending(std::uint64_t ticket,std::size_t retained,
        const BeforeDeviceCommit& before_commit={}) {
        require_device_pending_ticket(ticket);
        if(!retained || retained>device_pending_->committed_tokens.size())
            throw std::invalid_argument("fast device pending retained extent");
        phase_=Phase::executing;
        bool ring_begin_attempted=false;
        bool ring_undo_armed=false;
        bool irreversible=false;
        try {
            Step value=*device_pending_;
            if(retained<value.committed_tokens.size()) {
                context_->rollback_transaction(stream_);
                context_->begin_transaction(stream_);
                for(std::size_t row=0;row<retained;++row) {
                    const auto token=value.committed_tokens[row];
                    const auto replay_greedy=exl3_branch_greedy(*context_,stream_);
                    if(replay_greedy!=token)
                        throw std::runtime_error(
                            "fast device pending prefix replay diverged at position "+
                            std::to_string(value.root_position+static_cast<int>(row))+
                            " row "+std::to_string(row)+" retained "+
                            std::to_string(retained)+" candidate "+
                            std::to_string(value.committed_tokens.size())+
                            " verifier "+std::to_string(token)+" scalar "+
                            std::to_string(replay_greedy));
                    context_->decode(token,stream_);
                    for(std::size_t tap=0;tap<5;++tap)
                        context_->copy_tap_row_to_device(tap_layers[tap],
                            staging_[tap]+row*tap_hidden,stream_);
                }
                value.committed_tokens.resize(retained);
                value.staged_bytes=retained*5*tap_hidden*sizeof(std::uint16_t);
                value.authorized_replay_rows=retained;
                value.verification.fast_mia_parity_pending.reset();
                value.verification.fast_mia_parity_w1_bonus_rows=0;
            }
            value.terminal=!device_pending_terminal_.empty() &&
                exl3_terminal_token(value.committed_tokens.back(),
                    device_pending_terminal_);
            if(!context_->transaction_active() ||
               context_->position()!=value.root_position+static_cast<int>(retained) ||
               draft_.ring_base_abs()+draft_.ring_count()!=value.root_position)
                throw std::logic_error("fast device pending selected frontier");
            const auto ring_start=Clock::now();
            std::array<const std::uint16_t*,5> sources{};
            for(std::size_t tap=0;tap<5;++tap)sources[tap]=staging_[tap];
            ring_begin_attempted=true;
            draft_.begin_prefill_ring_undo(static_cast<int>(retained),
                value.root_position,stream_);
            ring_undo_armed=true;
            draft_.commit_prefill_block(sources.data(),static_cast<int>(retained),
                value.root_position,stream_);
            check(cudaStreamSynchronize(stream_),
                "fast device pending draft completion");
            value.ring_commit_ms=elapsed(ring_start);
            if(draft_.ring_base_abs()+draft_.ring_count()!=context_->position())
                throw std::logic_error("fast device pending ring frontier");
            if(before_commit)before_commit(value);
            // accept/target commit/continuation finishing can still fail. No
            // external authority may change in before_commit; once acceptance
            // starts, an error poisons both borrowed numerical owners.
            irreversible=true;
            draft_.accept_prefill_ring_undo(stream_);
            ring_undo_armed=false;
            context_->commit_transaction();
            if(context_->continuation_rows()>0)
                context_->finish_exact_continuation(stream_);
            check(cudaStreamSynchronize(stream_),
                "fast device pending target completion");
            frontier_=context_->position();
            record_totals(value);
            clear_device_pending();
            phase_=value.terminal?Phase::closed:Phase::ready;
            return value;
        } catch(...) {
            const auto failure=std::current_exception();
            try {
                if(irreversible || (ring_begin_attempted && !ring_undo_armed))
                    throw std::logic_error("fast device pending cannot prove rollback after ring admission");
                restore_device_pending_root(ring_undo_armed);
                clear_device_pending();
                phase_=Phase::ready;
            } catch(...) {
                clear_device_pending();
                phase_=Phase::poisoned;
            }
            std::rethrow_exception(failure);
        }
    }

    void cancel_device_pending(std::uint64_t ticket) {
        require_device_pending_ticket(ticket);
        phase_=Phase::executing;
        try {
            restore_device_pending_root(false);
            clear_device_pending();
            phase_=Phase::ready;
        } catch(...) {
            clear_device_pending();
            phase_=Phase::poisoned;
            throw;
        }
    }

    Step step(int width,std::span<const std::int64_t> terminal={},
        Exl3OuterDeviceStageTimeline* timeline=nullptr,
        const BeforeVerify& before_verify={},
        const AfterVerify& after_verify={},
        const ChooseWidth& choose_width={}) {
        require_ready_width(width);
        phase_=Phase::executing;
        try {
            Step value;
            value.root_position=frontier_;
            value.width=width;
            value.proposal.fill(248070);
            if(width==1)run_single(value,terminal,false);
            else run_batch(value,terminal,timeline,before_verify,after_verify,
                choose_width,false);
            if(value.committed_tokens.empty() ||
               context_->position()!=value.root_position+
                   static_cast<int>(value.committed_tokens.size()) ||
               draft_.ring_base_abs()+draft_.ring_count()!=context_->position())
                throw std::logic_error("fast device round settlement frontier");
            frontier_=context_->position();
            record_totals(value);
            value.terminal=value.terminal ||
                (!terminal.empty() && exl3_terminal_token(
                    value.committed_tokens.back(),terminal));
            phase_=value.terminal?Phase::closed:Phase::ready;
            return value;
        }catch(...) {phase_=Phase::poisoned;throw;}
    }

    // Completes all promised target and draft work, including the last ring
    // submission. An incomplete/failed request cannot claim a timed drain.
    void finish() {
        if(phase_!=Phase::ready && phase_!=Phase::closed)
            throw std::logic_error("fast device round cannot finish");
        try {
            check(cudaStreamSynchronize(stream_),"fast device round final drain");
            if(context_->position()!=frontier_ ||
               draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
                throw std::logic_error("fast device round final frontier");
            phase_=Phase::closed;
        }catch(...) {phase_=Phase::poisoned;throw;}
    }
    // Explicit request cancellation/reset. If either reset fails, both owners
    // remain poisoned and the caller must retire them instead of reusing them.
    void reset() {
        if(phase_==Phase::executing)
            throw std::logic_error("fast device round executing reset");
        phase_=Phase::poisoned;
        std::exception_ptr first;
        try {context_->reset(stream_);}catch(...) {first=std::current_exception();}
        try {draft_.reset(stream_);}catch(...) {if(!first)first=std::current_exception();}
        if(first)std::rethrow_exception(first);
        clear_pending();clear_device_pending();frontier_=0;totals_={};phase_=Phase::unstarted;
    }
    bool poisoned() const noexcept {return phase_==Phase::poisoned;}
    bool has_pending() const noexcept {
        return phase_==Phase::pending || phase_==Phase::device_pending;
    }
    int frontier() const noexcept {return frontier_;}
    const Totals& totals() const noexcept {return totals_;}

private:
    enum class Phase {unstarted,ready,executing,pending,device_pending,poisoned,closed};
    void require_ready_width(int width) const {
        if(phase_!=Phase::ready)
            throw std::logic_error("fast device round not ready");
        if(width<1 || width>8 ||
           width>context_->max_context()-context_->position())
            throw std::invalid_argument("fast device round width/capacity");
        if(context_->position()!=frontier_ ||
           draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
            throw std::logic_error("fast device round frontier changed");
    }
    void require_pending_ticket(std::uint64_t ticket) const {
        if(phase_!=Phase::pending || !pending_ || !ticket ||
           ticket!=pending_ticket_ || !pending_root_ || !pending_ring_)
            throw std::logic_error("fast device pending ticket unavailable");
    }
    void require_device_pending_ticket(std::uint64_t ticket) const {
        if(phase_!=Phase::device_pending || !device_pending_ || !ticket ||
           ticket!=pending_ticket_ || !context_->transaction_active())
            throw std::logic_error("fast device numerical pending ticket unavailable");
    }
    void restore_device_pending_root(bool ring_undo_armed) {
        if(ring_undo_armed)draft_.rollback_prefill_ring_undo(stream_);
        if(context_->transaction_active())context_->rollback_transaction(stream_);
        check(cudaStreamSynchronize(stream_),
            "fast device numerical pending rollback completion");
        if(context_->position()!=frontier_ ||
           draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
            throw std::logic_error("fast device numerical pending rollback frontier");
    }
    void clear_device_pending() noexcept {
        device_pending_.reset();device_pending_terminal_.clear();
    }
    void restore_pending_root() {
        if(!pending_root_ || !pending_ring_)
            throw std::logic_error("fast device pending root unavailable");
        context_->restore_exact_host_state(*pending_root_,stream_);
        draft_.restore_host_ring(pending_ring_,stream_);
        check(cudaStreamSynchronize(stream_),"fast device pending rollback completion");
        if(context_->position()!=frontier_ ||
           draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
            throw std::logic_error("fast device pending rollback frontier");
    }
    void clear_pending() noexcept {
        pending_.reset();pending_root_.reset();pending_ring_.reset();
        pending_terminal_.clear();
    }
    void record_totals(const Step& value) noexcept {
        totals_.proposal_ms+=value.proposal_ms;
        totals_.ring_commit_ms+=value.ring_commit_ms;
        totals_.staged_bytes+=value.staged_bytes;
        if(value.single_row){++totals_.replayed_rows;return;}
        totals_.verifier_ms+=value.verifier_ms;
        totals_.verified_rows+=value.verification.verification_rows;
        totals_.replayed_rows+=value.verification.replay_rows+
            value.authorized_replay_rows;
        totals_.reused_seeds+=value.verification.device_seed_reused;
        totals_.seed_fallbacks+=value.verification.device_seed_fallback;
        ++totals_.rounds;
    }
    static void check(cudaError_t error,const char* operation) {
        if(error!=cudaSuccess)
            throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
    }
    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    }
    void commit_captured(int rows,int first_abs) {
        if(rows<1 || rows>1024)
            throw std::invalid_argument("fast device captured forward extent");
        for(int first=0;first<rows;first+=16) {
            const int count=std::min(16,rows-first);
            for(std::size_t tap=0;tap<5;++tap)
                context_->copy_tap_rows_to_device(tap_layers[tap],first,
                    staging_[tap],count,stream_);
            std::array<const std::uint16_t*,5> sources{};
            for(std::size_t tap=0;tap<5;++tap)sources[tap]=staging_[tap];
            draft_.commit_prefill_block(sources.data(),count,
                first_abs+first,stream_);
        }
    }
    void prefill_layer_major(std::span<const std::int64_t> input) {
        const int total=static_cast<int>(input.size());
        // Near the minimum admitted context the initial 16 rows still belong
        // to the draft ring. They were committed from the ordinary tap capture.
        const int first=std::max(16,retained_start_for(total));
        const int rows=total-first;
        if(rows<2032 || rows>2063)
            throw std::logic_error("fast device retained tap partition halo");
        const auto bytes=static_cast<std::size_t>(rows)*5*tap_hidden*
            sizeof(std::uint16_t);
        struct Arena {
            std::uint16_t* ptr=nullptr;
            ~Arena(){if(ptr)cudaFree(ptr);}
        } arena;
        check(cudaMalloc(reinterpret_cast<void**>(&arena.ptr),bytes),
            "fast device retained tap arena");
        const Exl3TextContext::RetainedTapTail retained{
            arena.ptr,bytes,first,rows};
        try {
            context_->append_prefill_layer_major(input.subspan(16),stream_,
                &retained);
            for(int forward=16;forward<total;) {
                const int forward_rows=std::min(1024,total-forward);
                for(int row=0;row<forward_rows;row+=16) {
                    const int count=std::min(16,forward_rows-row);
                    const int absolute=forward+row;
                    if(absolute+count<=total-2048) {
                        draft_.skip_fresh_prefill_block(count,absolute,stream_);
                        continue;
                    }
                    if(absolute<first || absolute+count>total)
                        throw std::logic_error("fast device retained tap call outside arena");
                    std::array<const std::uint16_t*,5> sources{};
                    for(std::size_t tap=0;tap<5;++tap)
                        sources[tap]=arena.ptr+
                            (tap*static_cast<std::size_t>(rows)+absolute-first)*
                                tap_hidden;
                    draft_.commit_prefill_block(sources.data(),count,
                        absolute,stream_);
                }
                forward+=forward_rows;
            }
            check(cudaStreamSynchronize(stream_),
                "fast device retained tap completion before arena release");
        }catch(...) {
            cudaStreamSynchronize(stream_);
            throw;
        }
    }
    void run_single(Step& value,std::span<const std::int64_t> terminal,
        bool deferred) {
        const auto start=Clock::now();
        const auto token=exl3_branch_greedy(*context_,stream_);
        value.proposal[0]=token;
        value.seed_ready=Clock::now();
        context_->decode(token,stream_);
        value.proposal_ms=elapsed(start);
        const auto ring_start=Clock::now();
        for(std::size_t tap=0;tap<5;++tap)
            context_->copy_tap_row_to_device(tap_layers[tap],staging_[tap],stream_);
        if(!deferred) {
            std::array<const std::uint16_t*,5> sources{};
            for(std::size_t tap=0;tap<5;++tap)sources[tap]=staging_[tap];
            draft_.commit_prefill_block(sources.data(),1,value.root_position,stream_);
            value.ring_commit_ms=elapsed(ring_start);
        }
        value.staged_bytes=5*tap_hidden*sizeof(std::uint16_t);
        value.committed_tokens.push_back(token);
        value.single_row=true;
        value.terminal=exl3_terminal_token(token,terminal);
    }
    void run_batch(Step& value,std::span<const std::int64_t> terminal,
        Exl3OuterDeviceStageTimeline* timeline,
        const BeforeVerify& before_verify,const AfterVerify& after_verify,
        const ChooseWidth& choose_width,bool deferred,
        Exl3OuterDeviceSettlement settlement=Exl3OuterDeviceSettlement::Eager) {
        const auto revision=std::make_shared<const std::uint64_t>(totals_.rounds+1);
        const Exl3CommittedTapBinding binding{
            std::shared_ptr<const void>(context_,context_.get()),revision,
            value.root_position,acquisition_,execution_};
        const auto proposal_start=Clock::now();
        value.proposal[0]=exl3_branch_greedy(*context_,stream_);
        std::optional<Exl3OuterDeviceSeedPacket> seed;
        if(reuse_seed_)seed=bind_exl3_outer_device_seed(*context_,binding,
            value.proposal[0],coherent_policy,stream_);
        value.seed_ready=Clock::now();
        if(choose_width) {
            const int selected=choose_width(value.seed_ready,value.width);
            if(selected<2 || selected>value.width)
                throw std::invalid_argument("fast device selected horizon");
            value.width=selected;
        }
        const auto draft_start=Clock::now();
        value.target_seed_ms=std::chrono::duration<double,std::milli>(
            draft_start-proposal_start).count();
        auto proposed=draft_.propose_cached_view(
            std::span<const std::int64_t>(value.proposal).first(value.width),
            value.root_position,context_->target_embedding(),
            context_->target_lm_head_weights(),
            context_->target_lm_head_metadata(),248070,stream_);
        if(proposed.size()!=static_cast<std::size_t>(value.width-1))
            throw std::runtime_error("fast device round draft proposal extent");
        value.draft_api_ms=elapsed(draft_start);
        std::copy(proposed.begin(),proposed.end(),value.proposal.begin()+1);
        value.proposal_ms=elapsed(proposal_start);
        std::size_t staged=0;
        const Exl3CommittedTapConsumer consumer=[&](
            const Exl3CommittedTapSegment& segment,cudaStream_t copy_stream) {
            segment.validate();
            if(segment.owner!=binding.context_owner ||
               segment.root_revision!=binding.root_revision ||
               segment.root_position!=value.root_position)
                throw std::logic_error("fast device round committed tap owner");
            for(std::size_t tap=0;tap<5;++tap)
                check(cudaMemcpyAsync(staging_[tap]+
                        static_cast<std::size_t>(segment.destination_first)*tap_hidden,
                    segment.planes[tap]+
                        static_cast<std::size_t>(segment.source_first)*tap_hidden,
                    static_cast<std::size_t>(segment.rows)*tap_hidden*
                        sizeof(std::uint16_t),cudaMemcpyDeviceToDevice,copy_stream),
                    "fast device round committed tap D2D");
            staged+=static_cast<std::size_t>(segment.rows)*5*tap_hidden*
                sizeof(std::uint16_t);
        };
        if(before_verify)before_verify(*context_);
        const auto verifier_start=Clock::now();
        value.verification=verify_exl3_outer_device_resident_reference(
            *context_,std::span<const std::int64_t>(value.proposal).first(value.width),
            terminal,stream_,&binding,&consumer,timeline,
            seed?&*seed:nullptr,reuse_seed_?coherent_policy:0,settlement);
        value.verifier_ms=elapsed(verifier_start);
        if(after_verify)after_verify(*context_);
        value.committed_tokens=value.verification.committed_tokens;
        if(const char* log_path=std::getenv("NINFER_DFLASH2_ROUND_LOG")) {
            // Diagnostic-only round record: root position, proposal chain,
            // committed tokens and the draft's per-position top-16 candidates.
            const int rows=value.width-1;
            std::vector<std::int64_t> candidates(static_cast<std::size_t>(rows)*16);
            if(rows>0 && draft_.last_topk_ids_device_for_test())
                check(cudaMemcpy(candidates.data(),draft_.last_topk_ids_device_for_test(),
                        candidates.size()*sizeof(std::int64_t),cudaMemcpyDeviceToHost),
                    "round log top-16 candidates");
            std::ofstream log(log_path,std::ios::app);
            log<<value.root_position<<';';
            for(int i=0;i<value.width;++i)log<<value.proposal[static_cast<std::size_t>(i)]<<(i+1<value.width?' ':';');
            for(std::size_t i=0;i<value.committed_tokens.size();++i)
                log<<value.committed_tokens[i]<<(i+1<value.committed_tokens.size()?' ':';');
            for(std::size_t i=0;i<candidates.size();++i)
                log<<candidates[i]<<(i+1<candidates.size()?' ':'\n');
            if(candidates.empty())log<<'\n';
        }
        const auto useful=value.committed_tokens.size();
        if(!useful || useful>static_cast<std::size_t>(value.width) ||
           staged!=useful*5*tap_hidden*sizeof(std::uint16_t) ||
           context_->position()!=value.root_position+static_cast<int>(useful))
            throw std::logic_error("fast device round verified tap extent");
        if(!deferred) {
            const auto ring_start=Clock::now();
            std::array<const std::uint16_t*,5> sources{};
            for(std::size_t tap=0;tap<5;++tap)sources[tap]=staging_[tap];
            draft_.commit_prefill_block(sources.data(),static_cast<int>(useful),
                value.root_position,stream_);
            value.ring_commit_ms=elapsed(ring_start);
        }
        value.staged_bytes=staged;
        value.terminal=value.verification.stopped;
    }

    std::shared_ptr<Exl3TextContext> context_;
    Exl3Dflash2DraftModel& draft_;
    std::array<std::uint16_t*,5> staging_{};
    std::uint64_t acquisition_=0,execution_=0;
    bool reuse_seed_=false;
    cudaStream_t stream_=nullptr;
    Phase phase_=Phase::unstarted;
    int frontier_=0;
    Totals totals_{};
    std::uint64_t pending_ticket_=0;
    std::optional<Step> pending_;
    std::shared_ptr<const Exl3ExactHostState> pending_root_;
    std::shared_ptr<const Exl3DraftHostRing> pending_ring_;
    std::vector<std::int64_t> pending_terminal_;
    std::optional<Step> device_pending_;
    std::vector<std::int64_t> device_pending_terminal_;
};

} // namespace ninfer::exl3

static_assert(ninfer::exl3::Exl3FastDeviceRound::retained_start_for(4096)==2048);
static_assert(ninfer::exl3::Exl3FastDeviceRound::retained_start_for(16384)==14336);
static_assert(ninfer::exl3::Exl3FastDeviceRound::retained_start_for(4097)==2048);
