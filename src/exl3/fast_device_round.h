#pragma once

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <thread>
#include <sstream>
#include "exl3/native_context_extent.h"
#include "exl3/l0_history_mark.h"
#include "core/poll_wait.h"
#include "exl3/dflash2_draft.h"
#include "exl3/exact_outer_reference.h"
#include "exl3/verification_tree.h"
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

namespace voidinfer::exl3 {

// VeriCache (VOIDINFER_EXL3_VERICACHE): L0 OSCAR drafts blocks of tokens that
// are re-run in one pass against the exact FP16-L2 history before
// publication; the first drafted token the exact model would not have
// produced is replaced and the rest of the block is redrafted. The block
// length is chosen per request (engine DraftHorizon, VeriCache paper) up to
// VOIDINFER_EXL3_VERICACHE_BLOCK. A drafted token is kept when its exact logit
// trails the exact choice by at most VOIDINFER_EXL3_VERICACHE_DELTA (sampled
// requests: the coupled perturbed values, sampled_select.cuh). "1"/"exact"
// (default): 0.0625, twice the arithmetic floor measured between the decode
// and verifier kernels on identical exact history (max 0.03 over 2K
// tokens); "tolerance": 1.0 (512 hot rows: raw drafts trail by up to ~1.5
// at 32K). The prompt is ingested with exact history as well.
struct Exl3VeriCacheConfig {
    bool enabled=false;
    float delta=1.0f;
    int block=1024;   // longest verify block
    static const Exl3VeriCacheConfig& get() {
        static const Exl3VeriCacheConfig value=[] {
            Exl3VeriCacheConfig c;
            // Exact verification of L0 OSCAR output is the default whenever L0
            // is (VOIDINFER_EXL3_VERICACHE=0 opts out, =tolerance relaxes it).
            const char* mode=std::getenv("VOIDINFER_EXL3_VERICACHE");
            const std::string m=mode?mode:(exl3_l0_oscar_enabled()?"1":"0");
            if(m=="0"||m.empty()) return c;
            if(m!="1" && m!="tolerance" && m!="exact")
                throw std::invalid_argument("VOIDINFER_EXL3_VERICACHE must be 0, 1, tolerance or exact");
            c.enabled=true;
            if(m!="tolerance") c.delta=0.0625f;
            if(const char* d=std::getenv("VOIDINFER_EXL3_VERICACHE_DELTA")) c.delta=std::strtof(d,nullptr);
            if(const char* b=std::getenv("VOIDINFER_EXL3_VERICACHE_BLOCK")) c.block=std::atoi(b);
            if(c.block<1 || c.block>1024 || !(c.delta>=0.0f))
                throw std::invalid_argument("VOIDINFER_EXL3_VERICACHE_BLOCK must be 1..1024 and DELTA >= 0");
            return c;
        }();
        return value;
    }
};

// The Engine's exact VeriCache line, shared by all lanes. Line work (passes,
// fixes, syncs) runs one at a time across lanes in the ordered passes anyway,
// so a line per lane bought no parallelism for ~530 MiB of device memory
// each. A lane's line state lives in its slots (mapped host memory with
// concurrent lanes); a Lease binds the line to the lane that uses it,
// restoring that lane's root (and its window rows) after another lane.
//
// The line's live state is tracked against its owner lane: at the owner's
// root, after the owner's last pass (not saved: a later lane parks it in the
// owner's pending slot first), or stale (a correction discarded that pass).
class Exl3ExactLinePool {
public:
    struct Lane {
        Exl3ExactLineSlot root;      // line state at the lane's line frontier
        Exl3ExactLineSlot pending;   // after its last pass, parked by another lane
        bool pending_parked=false;
    };
    explicit Exl3ExactLinePool(std::shared_ptr<Exl3TextContext> line) : line_(std::move(line)) {
        if(!line_ || !line_->exact_line())throw std::invalid_argument("exact line pool needs an exact line");
        if(cudaEventCreateWithFlags(&released_,cudaEventDisableTiming)!=cudaSuccess)
            throw std::runtime_error("exact line pool event");
    }
    ~Exl3ExactLinePool() {if(released_)cudaEventDestroy(released_);}
    Exl3ExactLinePool(const Exl3ExactLinePool&)=delete;
    Exl3ExactLinePool& operator=(const Exl3ExactLinePool&)=delete;
    // The line itself (folded VeriCache, one lane: no leases).
    Exl3TextContext& line() noexcept {return *line_;}

    // Exclusive use of the line, stream-ordered after the previous lease.
    class Lease {
    public:
        Lease(Exl3ExactLinePool& pool,cudaStream_t stream) : pool_(pool),lock_(pool.mutex_),stream_(stream) {
            if(cudaStreamWaitEvent(stream_,pool_.released_,0)!=cudaSuccess)
                throw std::runtime_error("exact line lease wait");
        }
        ~Lease() {(void)cudaEventRecord(pool_.released_,stream_);}
        Lease(const Lease&)=delete;
        Lease& operator=(const Lease&)=delete;
        // The line bound to `drafting` (lane `lane`), state not restored: the
        // caller syncs it from the drafting context.
        Exl3TextContext& bind(Lane& lane,Exl3TextContext& drafting,const Exl3DeviceSampling& sampling) {
            take(lane,drafting,sampling);
            pool_.holds_=Holds::other;
            pool_.fresh_outputs_=false;
            return *pool_.line_;
        }
        // The line at `lane`'s root.
        Exl3TextContext& at_root(Lane& lane,Exl3TextContext& drafting,const Exl3DeviceSampling& sampling) {
            const bool moved=take(lane,drafting,sampling);
            if(moved || pool_.holds_!=Holds::root) {
                pool_.line_->restore_exact_line_root(lane.root,moved,stream_);
                pool_.holds_=Holds::root;
                pool_.fresh_outputs_=false;
            }
            return *pool_.line_;
        }
        // The line ran rows for its owner past the root (a pass), or now holds
        // the owner's root again (saved there, or a correction restored it).
        void ran_pass() {pool_.holds_=Holds::pass;pool_.fresh_outputs_=true;}
        void ran_to_root() {pool_.holds_=Holds::root;pool_.fresh_outputs_=true;}
        void synced_to_root() {pool_.holds_=Holds::root;}
        // The line still holds `lane`'s last pass (its logits are that pass's).
        bool holds_pass_of(const Lane& lane) const noexcept {
            return pool_.owner_==&lane && pool_.holds_==Holds::pass;
        }
        // The line's last logits/taps are its current state's own rows.
        bool fresh_outputs() const noexcept {return pool_.fresh_outputs_;}
        // Accepted block: `lane`'s pass becomes its root, saved from the line
        // when it still holds that pass, else from its parked pending slot.
        void accept_pass(Lane& lane) {
            if(holds_pass_of(lane)) {
                pool_.line_->save_exact_line_root(lane.root,stream_);
                pool_.holds_=Holds::root;
            } else {
                if(!lane.pending_parked)throw std::logic_error("exact line pass neither held nor parked");
                std::swap(lane.root,lane.pending);
            }
            lane.pending_parked=false;
        }
        // Corrected block: its pass is discarded (the root stays).
        void discard_pass(Lane& lane) {
            if(holds_pass_of(lane))pool_.holds_=Holds::stale;
            lane.pending_parked=false;
        }
    private:
        // Rebinds the line to `lane`, parking the previous owner's unsaved pass.
        bool take(Lane& lane,Exl3TextContext& drafting,const Exl3DeviceSampling& sampling) {
            auto& line=*pool_.line_;
            const bool moved=pool_.owner_!=&lane;
            if(moved) {
                if(pool_.owner_ && pool_.holds_==Holds::pass) {
                    line.save_exact_line_root(pool_.owner_->pending,stream_);
                    pool_.owner_->pending_parked=true;
                }
                pool_.owner_=&lane;
                pool_.holds_=Holds::other;
                pool_.fresh_outputs_=false;
            }
            // Pointers only: the lane's current planes and M-RoPE map.
            line.bind_exact_line(drafting);
            // The line verifies this request's draws with the same coupled
            // sampler (a greedy line "corrected" every sampled token).
            line.set_device_sampling(sampling);
            return moved;
        }
        Exl3ExactLinePool& pool_;
        std::unique_lock<std::mutex> lock_;
        cudaStream_t stream_;
    };
    // A lane's request ended: the line no longer belongs to it.
    void release_lane(const Lane& lane) {
        const std::lock_guard lock(mutex_);
        if(owner_==&lane){owner_=nullptr;holds_=Holds::other;fresh_outputs_=false;}
    }
private:
    enum class Holds {other,root,pass,stale};
    std::mutex mutex_;
    std::shared_ptr<Exl3TextContext> line_;
    cudaEvent_t released_=nullptr;
    Lane* owner_=nullptr;
    Holds holds_=Holds::other;
    bool fresh_outputs_=false;
};

// One physical C1 greedy request. The caller owns the five 16-row staging
// slabs and the exclusively borrowed draft. Their lifetimes must cover finish
// or reset. The context owner is retained so no tap or seed can outlive it.
// Prefix preparation is performed by the caller; begin_prefilled validates its
// completed target/draft frontier before any output work is allowed.
class Exl3FastDeviceRound {
public:
    // Several lanes: the asynchronous line's passes and the drafting
    // context's fixes use the model's prompt-scale pool like the other lanes'
    // prompt passes; the Engine orders them all (enter/leave bracket one
    // pass's enqueue on `stream`). Null: the lane owns the pool.
    struct PassOrder {
        virtual ~PassOrder()=default;
        virtual void enter(cudaStream_t stream)=0;
        virtual void leave(cudaStream_t stream)=0;
    };
    class OrderedPass {
    public:
        OrderedPass(PassOrder* order,cudaStream_t stream) : order_(order),stream_(stream) {if(order_)order_->enter(stream_);}
        ~OrderedPass() {if(order_)order_->leave(stream_);}
        OrderedPass(const OrderedPass&)=delete;
        OrderedPass& operator=(const OrderedPass&)=delete;
    private:
        PassOrder* order_;
        cudaStream_t stream_;
    };
    void set_pass_order(PassOrder* order) noexcept {pass_order_=order;}
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
        double target_seed_ms=0,draft_api_ms=0;
        std::uint64_t verified_blocks=0,corrected_blocks=0;
        // Shared line: last rows decided by their exact choice because another
        // lane took the line before the next drafted token arrived.
        std::uint64_t line_rows_unrescored=0;
        std::uint64_t accepted_checked=0,accepted_off_greedy=0;
        // VOIDINFER_EXL3_VERICACHE_STATS: synchronized step wall times.
        double restore_ms=0,forward_ms=0,scores_ms=0,fix_ms=0,commit_ms=0;
        double verify_ms=0,accepted_gap_sum=0;
        float max_accepted_gap=0;
        // Accepted rows whose exact logit trails the exact top by more than
        // kGapEdges[i] (research: tolerance sweeps from one DELTA=100 run).
        static constexpr std::array<float,6> kGapEdges{0.0f,0.125f,0.25f,0.5f,1.0f,2.0f};
        std::array<std::uint64_t,6> accepted_gap_over{};
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
        if(context_->position()!=0 || input.empty() ||
           input.size()>static_cast<std::size_t>(context_->max_context()))
            throw std::invalid_argument("fast device fresh input extent");
        // The layer-major schedule keeps a 2032..2063-row retained draft
        // window (prefill_layer_major); shorter inputs run row-wise.
        layer_major=layer_major && input.size()>=2048;
        phase_=Phase::executing;
        try {
            draft_.reset(stream_);
            draft_.begin_fresh_prefill(0,static_cast<long long>(input.size()),stream_);
            const int initial=layer_major?Exl3TextContext::layer_major_initial_rows():
                static_cast<int>(std::min<std::size_t>(16,input.size()));
            if(initial) {
                context_->prefill(input.first(initial),stream_);
                if(layer_major && input.size()>=2064)
                    draft_.skip_fresh_prefill_block(initial,0,stream_);
                else commit_captured(initial,0);
            }
            if(layer_major)prefill_layer_major(input);
            else {
                static const bool trace=std::getenv("VOIDINFER_EXL3_PREFILL_TRACE")!=nullptr;
                double target_ms=0,commit_ms=0;
                for(std::size_t first=static_cast<std::size_t>(initial);first<input.size();) {
                    const int rows=static_cast<int>(std::min<std::size_t>(
                        1024,input.size()-first));
                    auto mark=Clock::now();
                    context_->append_prefill_wide(input.subspan(first,rows),stream_);
                    if(trace){(void)cudaStreamSynchronize(stream_);target_ms+=elapsed(mark);mark=Clock::now();}
                    commit_captured(rows,static_cast<int>(first),
                        static_cast<int>(input.size())-2048);
                    if(trace){(void)cudaStreamSynchronize(stream_);commit_ms+=elapsed(mark);}
                    first+=rows;
                }
                if(trace)std::fprintf(stderr,"FRESH_PREFILL rows=%zu target_ms=%.1f commit_ms=%.1f\n",
                    input.size(),target_ms,commit_ms);
            }
            draft_.finish_fresh_prefill(stream_);
            check(poll_then_sync(stream_),
                "fast device fresh target/draft completion");
            phase_=Phase::unstarted;
        }catch(...) {phase_=Phase::poisoned;throw;}
    }

    // Prompt suffix after a restored root. A suffix that replaces the whole
    // draft ring (L0 OSCAR, at least kLayerMajorSuffixRows rows) is ingested
    // layer-major like a fresh prompt: the draft restarts its ring on the
    // suffix and commits only the retained tail. Shorter suffixes run wide
    // target chunks with every block's taps committed to the (contiguous) ring.
    static constexpr int kLayerMajorSuffixRows=4096;
    void ingest_suffix(std::span<const std::int64_t> input,int first) {
        if(phase_!=Phase::unstarted || first<=0 || context_->position()!=first ||
           static_cast<std::size_t>(first)>=input.size() ||
           draft_.ring_base_abs()+draft_.ring_count()!=first)
            throw std::invalid_argument("fast device suffix ingestion frontier");
        if(Exl3NativeContextExtent::l0_oscar_enabled() &&
           input.size()-static_cast<std::size_t>(first)>=static_cast<std::size_t>(kLayerMajorSuffixRows)) {
            ingest_suffix_layer_major(input,first);
            return;
        }
        phase_=Phase::executing;
        try {
            for(std::size_t at=static_cast<std::size_t>(first);at<input.size();) {
                const int rows=static_cast<int>(std::min<std::size_t>(1024,input.size()-at));
                // Up to 8 rows (the turn-closure tail: an assistant header) run
                // the decode-row route, 7 rows 40 -> 17 ms at 3.8K context.
                if(rows<=8)context_->append_prefill(input.subspan(at,rows),stream_);
                else context_->append_prefill_wide(input.subspan(at,rows),stream_);
                commit_captured(rows,static_cast<int>(at));
                at+=static_cast<std::size_t>(rows);
            }
            check(poll_then_sync(stream_),"fast device suffix ingestion completion");
            phase_=Phase::unstarted;
        }catch(...) {phase_=Phase::poisoned;throw;}
    }

    // Layer-major suffix: the draft's 16-row calls start at `first`; calls
    // ending at or before total-2048 are skipped and the rest (the retained
    // 2048..2063-row tail) commit in one bulk pass, as prefill_layer_major does.
    void ingest_suffix_layer_major(std::span<const std::int64_t> input,int first) {
        const int total=static_cast<int>(input.size());
        int tail=first;
        for(int at=first;at<total;at+=16) {
            const int count=std::min(16,total-at);
            if(at+count<=total-2048)tail=at+count;
        }
        const int rows=total-tail;
        if(rows<2032 || rows>2063)
            throw std::logic_error("fast device suffix retained tap partition halo");
        phase_=Phase::executing;
        try {
            const auto bytes=static_cast<std::size_t>(rows)*5*tap_hidden*sizeof(std::uint16_t);
            struct Arena {
                std::uint16_t* ptr=nullptr;
                ~Arena(){if(ptr)cudaFree(ptr);}
            } arena;
            check(cudaMalloc(reinterpret_cast<void**>(&arena.ptr),bytes),"fast device suffix tap arena");
            const Exl3TextContext::RetainedTapTail retained{arena.ptr,bytes,tail,rows};
            draft_.reset(stream_);
            draft_.begin_fresh_prefill(first,total,stream_);
            try {
                context_->append_prefill_layer_major(input.subspan(static_cast<std::size_t>(first)),stream_,&retained);
                for(int at=first;at<tail;at+=16)
                    draft_.skip_fresh_prefill_block(std::min(16,tail-at),at,stream_);
                std::array<const std::uint16_t*,5> sources{};
                for(std::size_t tap=0;tap<5;++tap)
                    sources[tap]=arena.ptr+tap*static_cast<std::size_t>(rows)*tap_hidden;
                draft_.commit_prefill_bulk(sources.data(),rows,tail,stream_);
                draft_.finish_fresh_prefill(stream_);
                check(poll_then_sync(stream_),"fast device suffix layer-major completion");
            }catch(...) {
                poll_then_sync(stream_);
                throw;
            }
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
            check(poll_then_sync(stream_),"fast device pending ring completion");
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
            check(poll_then_sync(stream_),
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
            check(poll_then_sync(stream_),
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
            check(poll_then_sync(stream_),"fast device round final drain");
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
    // ===== VeriCache block verification (L0 OSCAR contexts) =====
    // Tokens drafted after the verified root are re-run in one exact pass
    // (history from the FP16 L2 planes) and checked row by row. delta == 0:
    // the first token that differs from the exact greedy token is replaced;
    // delta > 0: only tokens whose exact logit trails the exact top by more
    // than delta are. The rest of the block is discarded on a replacement.
    struct VerifiedBlock {
        std::vector<std::int64_t> tokens;
        bool corrected=false;
        float max_gap=0.0f;
        double verify_ms=0.0;
    };
    // Saves the verified root (target device checkpoint + draft ring slots for
    // the next `rows` commits) at the current frontier.
    void begin_verification(int rows) {
        if(phase_!=Phase::ready || rows<1 || rows>2047)
            throw std::logic_error("fast device verification begin");
        context_->save_verified_root(stream_);
        draft_.save_ring_checkpoint(rows,stream_);
        verified_frontier_=frontier_;
        verify_rows_=rows;
    }
    int verified_frontier() const noexcept {return verified_frontier_;}
    // Verifies frontier_ - verified_frontier tokens (`tokens`, in order). On
    // return target and draft hold verified_frontier + result.tokens.size()
    // rows of exact state; the root stays until commit_verification.
    // A block ending in a terminal token is verified on a closed round; the
    // phase is preserved.
    VerifiedBlock verify(std::span<const std::int64_t> tokens,float delta) {
        const auto settled_phase=phase_;
        if((phase_!=Phase::ready && phase_!=Phase::closed) || tokens.empty() ||
           static_cast<int>(tokens.size())!=frontier_-verified_frontier_)
            throw std::logic_error("fast device verification extent: tokens "+std::to_string(tokens.size())+
                " frontier "+std::to_string(frontier_)+" verified "+std::to_string(verified_frontier_));
        phase_=Phase::executing;
        try {
            const auto start=Clock::now();
            static const bool timed=std::getenv("VOIDINFER_EXL3_VERICACHE_STATS")!=nullptr;
            auto mark=start;
            const auto step=[&](double& total) {
                if(!timed)return;
                check(poll_then_sync(stream_),"vericache step timing");
                const auto now=Clock::now();
                total+=std::chrono::duration<double,std::milli>(now-mark).count();
                mark=now;
            };
            VerifiedBlock result;
            context_->restore_verified_root(stream_);
            step(totals_.restore_ms);
            exact_forward(tokens);
            step(totals_.forward_ms);
            std::vector<std::int64_t> next(tokens.begin()+1,tokens.end());
            next.push_back(-1);
            const auto scores=context_->exact_row_scores(next,stream_,delta);
            step(totals_.scores_ms);
            std::size_t reject=tokens.size();
            std::int64_t correction=-1;
            for(std::size_t i=0;i+1<tokens.size();++i) {
                const bool bad=delta>0?scores[i].gap>delta:scores[i].greedy!=tokens[i+1];
                if(bad) {
                    reject=i+1;correction=scores[i].greedy;
                    if(timed)std::fprintf(stderr,"VERICACHE_CORRECTION block_rows=%zu reject_at=%zu gap=%.3f drafted=%lld exact=%lld round[%s] context=%p\n",
                        tokens.size(),reject,static_cast<double>(scores[i].gap),
                        static_cast<long long>(tokens[i+1]),static_cast<long long>(correction),
                        context_->round_trace(verified_frontier_+static_cast<int>(reject)).c_str(),
                        static_cast<const void*>(context_.get()));
                    break;
                }
                // Accepted rows: how far each kept token trails the exact top.
                result.max_gap=std::max(result.max_gap,scores[i].gap);
                totals_.accepted_gap_sum+=scores[i].gap;
                totals_.accepted_off_greedy+=scores[i].gap>0.0f?1:0;
                for(std::size_t e=0;e<Totals::kGapEdges.size();++e)
                    totals_.accepted_gap_over[e]+=scores[i].gap>Totals::kGapEdges[e]?1:0;
                ++totals_.accepted_checked;
            }
            totals_.max_accepted_gap=std::max(totals_.max_accepted_gap,result.max_gap);
            result.tokens.assign(tokens.begin(),tokens.begin()+static_cast<std::ptrdiff_t>(reject));
            if(correction>=0) {
                result.tokens.push_back(correction);
                result.corrected=true;
                exact_forward_from_root(result.tokens);
                draft_.restore_ring_checkpoint(stream_);
                commit_captured(static_cast<int>(result.tokens.size()),verified_frontier_);
                step(totals_.fix_ms);
            }
            check(poll_then_sync(stream_),"fast device verification completion");
            frontier_=context_->position();
            if(draft_.ring_base_abs()+draft_.ring_count()!=frontier_ ||
               frontier_!=verified_frontier_+static_cast<int>(result.tokens.size()))
                throw std::logic_error("fast device verification frontier");
            result.verify_ms=elapsed(start);
            ++totals_.verified_blocks;
            totals_.corrected_blocks+=result.corrected?1:0;
            totals_.verify_ms+=result.verify_ms;
            phase_=settled_phase;
            return result;
        } catch(...) {
            context_->set_l0_exact_history(false);
            phase_=Phase::poisoned;
            throw;
        }
    }
    // ===== Folded exact VeriCache (one lane) =====
    // `line` (an exact line of this context, Exl3TextContext::make_exact_line)
    // verifies each drafted block inside the next round's batched target
    // forward instead of a separate exact pass: the block's rows share the
    // round's projections (one weight read). The drafting context keeps L0
    // history between corrections; the line is exact. Roots: the line's own
    // checkpoint and the drafting context's root + draft ring checkpoint at
    // the verified frontier, plus pending ones where the carried block ends.
    void begin_fold_verification(Exl3TextContext& line,int ring_rows) {
        if(phase_!=Phase::ready || ring_rows<1 || ring_rows>2047 || !line.exact_line())
            throw std::logic_error("folded verification begin");
        line_=&line;
        // The line verifies this request's draws: it selects with the same
        // coupled sampler (a greedy line "corrected" every sampled token).
        line.set_device_sampling(context_->device_sampling());
        const OrderedPass ordered(pass_order_,stream_);
        line.sync_exact_line_from(*context_,stream_);
        line.save_verified_root(stream_);
        context_->save_verified_root(stream_);
        draft_.save_ring_checkpoint(ring_rows,stream_);
        verified_frontier_=frontier_;
        verify_rows_=ring_rows;
        fold_armed_=false;
    }
    bool fold_active() const noexcept {return line_!=nullptr;}
    // Asynchronous exact VeriCache on the Engine's shared line (`lane`: this
    // lane's slots in it).
    void begin_async_verification(Exl3ExactLinePool& pool,Exl3ExactLinePool::Lane& lane,int ring_rows) {
        if(phase_!=Phase::ready || ring_rows<1 || ring_rows>2047)
            throw std::logic_error("asynchronous verification begin");
        pool_=&pool;
        lane_=&lane;
        {
            const OrderedPass ordered(pass_order_,stream_);
            Exl3ExactLinePool::Lease lease(pool,stream_);
            auto& line=lease.bind(lane,*context_,context_->device_sampling());
            line.sync_exact_line_from(*context_,stream_);
            line.save_exact_line_root(lane.root,stream_);
            lease.synced_to_root();
        }
        context_->save_verified_root(stream_);
        draft_.save_ring_checkpoint(ring_rows,stream_);
        verified_frontier_=frontier_;
        verify_rows_=ring_rows;
        fold_armed_=false;
        line_head_.clear();
        if(!async_)async_=std::make_unique<AsyncLine>();
    }
    // The next round carries `tokens` = [verified frontier, frontier) on the line.
    void arm_fold(std::span<const std::int64_t> tokens) {
        if(phase_!=Phase::ready || !line_ || fold_armed_ || tokens.size()<2 || tokens.size()>24 ||
           static_cast<int>(tokens.size())!=frontier_-verified_frontier_)
            throw std::logic_error("folded verification arm extent");
        context_->save_verified_root_pending(stream_);
        draft_.save_ring_checkpoint(verify_rows_,stream_,true);
        context_->set_fold_peer(line_,tokens,stream_);
        fold_tokens_.assign(tokens.begin(),tokens.end());
        fold_frontier_=frontier_;
        fold_armed_=true;
    }
    // After the round that carried the block settled (round_first: its first
    // committed token, predicted by the block's last row). A round that did
    // not carry the block (single-row) verifies it on the line alone.
    VerifiedBlock finish_fold(std::int64_t round_first,float delta) {
        if(phase_!=Phase::ready || !fold_armed_)throw std::logic_error("folded verification finish");
        fold_armed_=false;
        const bool carried=!context_->fold_pending();
        if(!carried)context_->set_fold_peer(nullptr,{},nullptr);
        return settle_fold(round_first,delta,carried,true);
    }
    // Verifies the remaining drafted tokens on the line alone (no further
    // round: terminal, budget or cancellation).
    VerifiedBlock flush_fold(std::span<const std::int64_t> tokens,float delta) {
        if((phase_!=Phase::ready && phase_!=Phase::closed) || !line_ || fold_armed_ ||
           static_cast<int>(tokens.size())!=frontier_-verified_frontier_)
            throw std::logic_error("folded verification flush");
        if(tokens.empty())return {};
        fold_tokens_.assign(tokens.begin(),tokens.end());
        fold_frontier_=frontier_;
        return settle_fold(-1,delta,false,false);
    }
    // End of request: the drafting context takes the line's exact state at the
    // verified frontier (drafted tokens past it are dropped).
    void end_fold() {
        if(pool_) {
            if(phase_!=Phase::ready)throw std::logic_error("asynchronous verification end");
            end_async();
            return;
        }
        if(!line_)return;
        if(phase_!=Phase::ready)throw std::logic_error("folded verification end");
        if(fold_armed_){context_->set_fold_peer(nullptr,{},nullptr);fold_armed_=false;}
        if(frontier_!=verified_frontier_) {
            context_->restore_verified_root(stream_);
            draft_.restore_ring_checkpoint(stream_);
        }
        context_->sync_exact_line_from(*line_,stream_,true);
        check(poll_then_sync(stream_),"folded verification end");
        frontier_=context_->position();
        if(draft_.ring_base_abs()+draft_.ring_count()!=frontier_ || frontier_!=verified_frontier_)
            throw std::logic_error("folded verification end frontier");
        line_=nullptr;
    }
    // ===== Asynchronous exact VeriCache (the Engine's shared line) =====
    // The line verifies the drafted block [verified frontier, frontier) on its
    // own stream (a worker thread) while drafting continues past it: at long
    // context the exact pass is bound by the PCIe stream of the FP16 history,
    // which overlaps the HBM-bound drafting rounds; with several lanes the
    // lanes keep drafting in the batched rounds instead of leaving them for
    // exclusive verify passes. The line's root is this lane's slot in the
    // pool (begin_async_verification); the drafting root and draft ring
    // checkpoint where the block ends are the pending ones.
    //
    // A correction does not re-run the history on the line: the line returns
    // to its root, the kept tokens and the exact correction (published) become
    // the line head that the next block forwards first, sharing that block's
    // history stream; the drafting context re-runs them on its L0 history.
    // Roots stay at the line frontier until a block is accepted.
    // A block ending in a terminal token is launched, settled and committed on
    // a closed round, like the synchronous verify/commit_verification.
    void launch_async(std::span<const std::int64_t> tokens,float delta) {
        if((phase_!=Phase::ready && phase_!=Phase::closed) || !pool_ || async_inflight_ || fold_commit_.pending ||
           tokens.empty() || static_cast<int>(tokens.size())!=frontier_-verified_frontier_ ||
           lane_->root.position()!=line_frontier())
            throw std::logic_error("asynchronous verification launch extent");
        context_->save_verified_root_pending(stream_);
        draft_.save_ring_checkpoint(verify_rows_,stream_,true);
        fold_tokens_=line_head_;
        fold_tokens_.insert(fold_tokens_.end(),tokens.begin(),tokens.end());
        async_head_=line_head_.size();
        async_root_=line_frontier();
        fold_frontier_=frontier_;
        async_->launch(*pool_,*lane_,*context_,fold_tokens_,async_root_,delta,stream_,pass_order_);
        async_inflight_=true;
        async_start_=Clock::now();
    }
    bool async_pending() const noexcept {return async_inflight_;}
    bool async_ready() const {return async_inflight_ && async_->done();}
    // Unpublished (drafted) tokens of the block in flight.
    std::size_t async_rows() const noexcept {return async_inflight_?fold_tokens_.size()-async_head_:0;}
    // Drafted rows past the line (bounded by the draft ring checkpoint).
    int async_lead() const noexcept {return frontier_-line_frontier();}
    // Waits for the block in flight and checks it; `next_token` is the token
    // drafted after it (< 0: none, the last row goes unchecked).
    VerifiedBlock settle_async(std::int64_t next_token,float delta) {
        const auto settled_phase=phase_;
        if((phase_!=Phase::ready && phase_!=Phase::closed) || !async_inflight_)
            throw std::logic_error("asynchronous verification settle");
        phase_=Phase::executing;
        try {
            const auto wait_start=Clock::now();
            auto scores=async_->wait();
            async_inflight_=false;
            const OrderedPass ordered(pass_order_,stream_);
            // Async: scores_ms = line pass time (launch to scored).
            totals_.scores_ms+=std::chrono::duration<double,std::milli>(async_->finished_-async_start_).count();
            totals_.forward_ms+=elapsed(wait_start);
            const std::span<const std::int64_t> tokens(fold_tokens_);
            const std::size_t head=async_head_;
            std::vector<std::int64_t> next(tokens.begin()+1,tokens.end());
            next.push_back(next_token);
            // The worker scored the last row against a placeholder: rescore
            // when the drafted next token is not the exact choice, on the
            // line's logits of that pass while it still holds them. Once
            // another lane took the line, the exact choice replaces the
            // drafted token (no tolerance on that one row; still exact).
            if(next_token>=0 && scores.back().greedy!=next_token) {
                Exl3ExactLinePool::Lease lease(*pool_,stream_);
                if(lease.holds_pass_of(*lane_)) {
                    scores=pool_->line().exact_row_scores(next,stream_,delta);
                } else {
                    scores.back().gap=std::numeric_limits<float>::infinity();
                    ++totals_.line_rows_unrescored;
                }
            } else if(next_token>=0) {
                scores.back().gap=0.0f;
            }
            // Rows predicting published head tokens were checked before.
            VerifiedBlock result;
            std::int64_t correction=-1;
            std::size_t reject=tokens.size();
            for(std::size_t i=head?head-1:0;i<tokens.size();++i) {
                if(next[i]<0)break;
                const bool bad=delta>0?scores[i].gap>delta:scores[i].greedy!=next[i];
                if(bad) {reject=i+1;correction=scores[i].greedy;break;}
                result.max_gap=std::max(result.max_gap,scores[i].gap);
                totals_.accepted_gap_sum+=scores[i].gap;
                totals_.accepted_off_greedy+=scores[i].gap>0.0f?1:0;
                for(std::size_t e=0;e<Totals::kGapEdges.size();++e)
                    totals_.accepted_gap_over[e]+=scores[i].gap>Totals::kGapEdges[e]?1:0;
                ++totals_.accepted_checked;
            }
            totals_.max_accepted_gap=std::max(totals_.max_accepted_gap,result.max_gap);
            result.tokens.assign(tokens.begin()+static_cast<std::ptrdiff_t>(head),
                tokens.begin()+static_cast<std::ptrdiff_t>(reject));
            const auto fix_start=Clock::now();
            if(correction>=0) {
                result.tokens.push_back(correction);
                result.corrected=true;
                std::vector<std::int64_t> kept(tokens.begin(),tokens.begin()+static_cast<std::ptrdiff_t>(reject));
                kept.push_back(correction);
                rewind_async(kept);
                line_head_=std::move(kept);
                totals_.fix_ms+=elapsed(fix_start);
            }
            fold_commit_=FoldCommit{true,result.corrected,true,result.tokens};
            ++totals_.verified_blocks;
            totals_.corrected_blocks+=result.corrected?1:0;
            result.verify_ms=elapsed(async_start_);
            totals_.verify_ms+=elapsed(wait_start);
            phase_=settled_phase;
            return result;
        } catch(...) {
            phase_=Phase::poisoned;
            throw;
        }
    }

    // Keeps the first `retained` tokens of the settled block (all, or the
    // prefix the output session accepted).
    void commit_async(std::size_t retained) {
        if((phase_!=Phase::ready && phase_!=Phase::closed) || !fold_commit_.pending ||
           retained>fold_commit_.tokens.size() || async_inflight_)
            throw std::logic_error("asynchronous verification commit");
        auto commit=std::move(fold_commit_);
        fold_commit_={};
        phase_=Phase::executing;
        try {
            if(retained<commit.tokens.size()) {
                // The session stopped inside the block: the line runs its head
                // and the kept prefix exactly, both lines take its state.
                const OrderedPass ordered(pass_order_,stream_);
                Exl3ExactLinePool::Lease lease(*pool_,stream_);
                std::vector<std::int64_t> kept(fold_tokens_.begin(),
                    fold_tokens_.begin()+static_cast<std::ptrdiff_t>(async_head_));
                kept.insert(kept.end(),commit.tokens.begin(),
                    commit.tokens.begin()+static_cast<std::ptrdiff_t>(retained));
                const int root=async_root_;
                if(!commit.corrected)lease.discard_pass(*lane_);
                auto& line=lease.at_root(*lane_,*context_,context_->device_sampling());
                if(!kept.empty()){forward_rows(line,kept,stream_);lease.ran_to_root();}
                context_->restore_verified_root(stream_);
                context_->sync_exact_line_from(line,stream_,lease.fresh_outputs());
                draft_.restore_ring_checkpoint(stream_);
                if(!kept.empty())commit_captured(static_cast<int>(kept.size()),root);
                check(poll_then_sync(stream_),"asynchronous verification partial commit");
                frontier_=context_->position();
                if(line.position()!=frontier_ || draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
                    throw std::logic_error("asynchronous verification partial commit frontier");
                line.save_exact_line_root(lane_->root,stream_);
                lease.synced_to_root();
                context_->save_verified_root(stream_);
                draft_.save_ring_checkpoint(verify_rows_,stream_);
                line_head_.clear();
                verified_frontier_=frontier_;
            } else if(commit.corrected) {
                // Roots stay at the line frontier; the drafting context holds
                // the head (rewind_async).
                verified_frontier_=frontier_;
            } else {
                {
                    Exl3ExactLinePool::Lease lease(*pool_,stream_);
                    lease.accept_pass(*lane_);
                }
                context_->promote_verified_root_pending();
                draft_.promote_ring_checkpoint();
                line_head_.clear();
                verified_frontier_+=static_cast<int>(commit.tokens.size());
            }
            phase_=Phase::ready;
        } catch(...) {phase_=Phase::poisoned;throw;}
    }
    bool async_active() const noexcept {return pool_!=nullptr;}

    // Keeps the first `retained` verified tokens (all, or the prefix the output
    // session accepted) and moves the verified root there.
    void commit_verification(std::span<const std::int64_t> retained) {
        const auto settled_phase=phase_;
        const auto commit_start=Clock::now();
        if((phase_!=Phase::ready && phase_!=Phase::closed) ||
           static_cast<int>(retained.size())>frontier_-verified_frontier_)
            throw std::logic_error("fast device verification commit extent");
        phase_=Phase::executing;
        try {
            if(static_cast<int>(retained.size())<frontier_-verified_frontier_) {
                if(retained.empty()) {
                    context_->restore_verified_root(stream_);
                } else {
                    exact_forward_from_root(retained);
                }
                draft_.restore_ring_checkpoint(stream_);
                if(!retained.empty())
                    commit_captured(static_cast<int>(retained.size()),verified_frontier_);
                check(poll_then_sync(stream_),"fast device verification rewind");
                frontier_=context_->position();
            }
            context_->save_verified_root(stream_);
            draft_.save_ring_checkpoint(verify_rows_,stream_);
            verified_frontier_=frontier_;
            if(std::getenv("VOIDINFER_EXL3_VERICACHE_STATS")) {
                check(poll_then_sync(stream_),"vericache commit timing");
                totals_.commit_ms+=elapsed(commit_start);
            }
            // The caller finishes on a published terminal; otherwise (e.g. the
            // terminal token was corrected away) drafting continues.
            (void)settled_phase;
            phase_=Phase::ready;
        } catch(...) {phase_=Phase::poisoned;throw;}
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
        check(poll_then_sync(stream_),
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
        check(poll_then_sync(stream_),"fast device pending rollback completion");
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
        totals_.target_seed_ms+=value.target_seed_ms;
        totals_.draft_api_ms+=value.draft_api_ms;
        totals_.verified_rows+=value.verification.verification_rows;
        totals_.replayed_rows+=value.verification.replay_rows+
            value.authorized_replay_rows;
        totals_.reused_seeds+=value.verification.device_seed_reused;
        totals_.seed_fallbacks+=value.verification.device_seed_fallback;
        ++totals_.rounds;
    }
    // Restores the verified root and runs `tokens` with exact history.
    static void forward_rows(Exl3TextContext& context,std::span<const std::int64_t> tokens,
                             cudaStream_t stream) {
        for(std::size_t at=0;at<tokens.size();) {
            const auto take=std::min<std::size_t>(tokens.size()-at,1024);
            const auto part=tokens.subspan(at,take);
            if(take>8)context.append_prefill_wide(part,stream);
            else context.append_prefill(part,stream);
            at+=take;
        }
    }
    // Scores the carried (or line-forwarded) block; a correction re-runs the
    // kept tokens from the roots (rewind_fold). Roots move in commit_fold.
    VerifiedBlock settle_fold(std::int64_t round_first,float delta,bool carried,bool round_followed) {
        phase_=Phase::executing;
        try {
            const auto start=Clock::now();
            static const bool timed=std::getenv("VOIDINFER_EXL3_VERICACHE_STATS")!=nullptr;
            auto mark=start;
            const auto step=[&](double& total) {
                if(!timed)return;
                check(poll_then_sync(stream_),"vericache fold step timing");
                const auto now=Clock::now();
                total+=std::chrono::duration<double,std::milli>(now-mark).count();
                mark=now;
            };
            const std::span<const std::int64_t> tokens(fold_tokens_);
            if(!carried)forward_rows(*line_,tokens,stream_);
            if(line_->position()!=verified_frontier_+static_cast<int>(tokens.size()))
                throw std::logic_error("folded verification line extent");
            step(totals_.forward_ms);
            std::vector<std::int64_t> next(tokens.begin()+1,tokens.end());
            next.push_back(round_first);
            const auto scores=line_->exact_row_scores(next,stream_,delta);
            step(totals_.scores_ms);
            auto result=judge_fold(tokens,next,scores,delta,round_followed,step);
            result.verify_ms=elapsed(start);
            totals_.verify_ms+=result.verify_ms;
            phase_=Phase::ready;
            return result;
        } catch(...) {
            phase_=Phase::poisoned;
            throw;
        }
    }
    // Checks each row's exact score against the next drafted token (a row whose
    // next token is unknown, < 0, ends the check); a correction rewinds both
    // lines (rewind_fold). The commit stays pending for commit_fold.
    template<class Step>
    VerifiedBlock judge_fold(std::span<const std::int64_t> tokens,const std::vector<std::int64_t>& next,
                             const std::vector<Exl3TextContext::RowScore>& scores,float delta,
                             bool round_followed,const Step& step) {
        VerifiedBlock result;
        std::int64_t correction=-1;
        std::size_t reject=tokens.size();
        for(std::size_t i=0;i<tokens.size();++i) {
            if(next[i]<0)break;
            const bool bad=delta>0?scores[i].gap>delta:scores[i].greedy!=next[i];
            if(bad) {reject=i+1;correction=scores[i].greedy;break;}
            result.max_gap=std::max(result.max_gap,scores[i].gap);
            totals_.accepted_gap_sum+=scores[i].gap;
            totals_.accepted_off_greedy+=scores[i].gap>0.0f?1:0;
            for(std::size_t e=0;e<Totals::kGapEdges.size();++e)
                totals_.accepted_gap_over[e]+=scores[i].gap>Totals::kGapEdges[e]?1:0;
            ++totals_.accepted_checked;
        }
        totals_.max_accepted_gap=std::max(totals_.max_accepted_gap,result.max_gap);
        result.tokens.assign(tokens.begin(),tokens.begin()+static_cast<std::ptrdiff_t>(reject));
        if(correction>=0) {
            result.tokens.push_back(correction);
            result.corrected=true;
            rewind_fold(result.tokens);
            step(totals_.fix_ms);
        }
        fold_commit_=FoldCommit{true,result.corrected,round_followed,result.tokens};
        check(poll_then_sync(stream_),"folded verification completion");
        ++totals_.verified_blocks;
        totals_.corrected_blocks+=result.corrected?1:0;
        return result;
    }
    // Line frontier: the published frontier minus the head the line has not
    // run yet.
    int line_frontier() const noexcept {return verified_frontier_-static_cast<int>(line_head_.size());}
    // Correction: the line returns to its root; the drafting context re-runs
    // `tokens` (the head: kept tokens and the correction) from its root on
    // its own L0 history and re-commits them to the draft ring.
    void rewind_async(std::span<const std::int64_t> tokens) {
        const int root=async_root_;
        {
            Exl3ExactLinePool::Lease lease(*pool_,stream_);
            lease.discard_pass(*lane_);
        }
        context_->restore_verified_root(stream_);
        forward_rows(*context_,tokens,stream_);
        draft_.restore_ring_checkpoint(stream_);
        commit_captured(static_cast<int>(tokens.size()),root);
        check(poll_then_sync(stream_),"asynchronous verification rewind");
        frontier_=context_->position();
        if(lane_->root.position()!=root || frontier_!=root+static_cast<int>(tokens.size()) ||
           draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
            throw std::logic_error("asynchronous verification rewind frontier");
    }
    // End of request: the line runs any head exactly and the drafting context
    // takes its state at the published frontier.
    // The drafting context takes the line's logits and taps only when they are
    // its state's own rows (a line restored from the slot holds another pass's,
    // like a restored verified root after an exclusive pass).
    void end_async() {
        if(async_inflight_) {
            // A block still in flight (cancellation): drop it.
            (void)async_->wait();
            async_inflight_=false;
            Exl3ExactLinePool::Lease lease(*pool_,stream_);
            lease.discard_pass(*lane_);
        }
        {
            const OrderedPass ordered(pass_order_,stream_);
            Exl3ExactLinePool::Lease lease(*pool_,stream_);
            const int root=line_frontier();
            auto& line=lease.at_root(*lane_,*context_,context_->device_sampling());
            if(!line_head_.empty()){forward_rows(line,line_head_,stream_);lease.ran_to_root();}
            if(frontier_!=root)context_->restore_verified_root(stream_);
            context_->sync_exact_line_from(line,stream_,lease.fresh_outputs());
            if(frontier_!=root)draft_.restore_ring_checkpoint(stream_);
            if(!line_head_.empty())commit_captured(static_cast<int>(line_head_.size()),root);
            check(poll_then_sync(stream_),"asynchronous verification end");
            frontier_=context_->position();
            if(draft_.ring_base_abs()+draft_.ring_count()!=frontier_ || frontier_!=verified_frontier_ ||
               line.position()!=frontier_)
                throw std::logic_error("asynchronous verification end frontier");
        }
        pool_->release_lane(*lane_);
        line_head_.clear();
        pool_=nullptr;
        lane_=nullptr;
    }
    // Both lines back to the roots at the verified frontier, then `tokens`
    // exactly on the line; the drafting context takes the line's state (its
    // L0 aging rewound) and the draft ring re-commits the rows.
    void rewind_fold(std::span<const std::int64_t> tokens) {
        line_->restore_verified_root(stream_);
        forward_rows(*line_,tokens,stream_);
        context_->restore_verified_root(stream_);
        context_->sync_exact_line_from(*line_,stream_,true);
        draft_.restore_ring_checkpoint(stream_);
        commit_captured(static_cast<int>(tokens.size()),verified_frontier_);
        check(poll_then_sync(stream_),"folded verification rewind");
        frontier_=context_->position();
        if(line_->position()!=frontier_ || draft_.ring_base_abs()+draft_.ring_count()!=frontier_)
            throw std::logic_error("folded verification rewind frontier");
    }
public:
    // Keeps the first `retained` tokens of the last folded block (all, or the
    // prefix the output session accepted) and moves every root there.
    void commit_fold(std::size_t retained) {
        if(!fold_commit_.pending && !retained && phase_==Phase::ready)return;
        if(phase_!=Phase::ready || !fold_commit_.pending || retained>fold_commit_.tokens.size())
            throw std::logic_error("folded verification commit");
        auto commit=std::move(fold_commit_);
        fold_commit_={};
        phase_=Phase::executing;
        try {
            if(retained<commit.tokens.size()) {
                // Only the accepted prefix stays (the session stopped inside
                // the block): re-run it from the roots.
                if(!commit.corrected && commit.round_followed)
                    context_->set_fold_peer(nullptr,{},nullptr);
                rewind_fold(std::span<const std::int64_t>(commit.tokens).first(retained));
                line_->save_verified_root(stream_);
                context_->save_verified_root(stream_);
                draft_.save_ring_checkpoint(verify_rows_,stream_);
                verified_frontier_=frontier_;
            } else if(commit.corrected) {
                line_->save_verified_root(stream_);
                context_->save_verified_root(stream_);
                draft_.save_ring_checkpoint(verify_rows_,stream_);
                verified_frontier_=frontier_;
            } else {
                line_->save_verified_root(stream_);
                if(commit.round_followed) {
                    context_->promote_verified_root_pending();
                    draft_.promote_ring_checkpoint();
                } else {
                    context_->save_verified_root(stream_);
                    draft_.save_ring_checkpoint(verify_rows_,stream_);
                }
                verified_frontier_+=static_cast<int>(commit.tokens.size());
            }
            phase_=Phase::ready;
        } catch(...) {phase_=Phase::poisoned;throw;}
    }
private:
    void exact_forward_from_root(std::span<const std::int64_t> tokens) {
        context_->restore_verified_root(stream_);
        exact_forward(tokens);
    }
    void exact_forward(std::span<const std::int64_t> tokens) {
        context_->set_l0_exact_history(true);
        // Row-major wide prefill keeps full-precision projections (the
        // layer-major route's block-quantized ones would make the verifier
        // noisier than the draft it checks).
        if(tokens.size()>8)context_->append_prefill_wide(tokens,stream_);
        else context_->append_prefill(tokens,stream_);
        context_->set_l0_exact_history(false);
    }
    static cudaError_t poll_then_sync(cudaStream_t stream) {
        const auto error=voidinfer::cuda_poll_stream(stream);
        return error!=cudaSuccess?error:cudaStreamSynchronize(stream);
    }
    static void check(cudaError_t error,const char* operation) {
        if(error!=cudaSuccess)
            throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
    }
    static double elapsed(Clock::time_point start) {
        return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    }
    // Draft blocks wholly before `ring_begin` (absolute) fall outside the
    // draft ring and are skipped, as the layer-major schedule does.
    void commit_captured(int rows,int first_abs,int ring_begin=0) {
        if(rows<1 || rows>1024)
            throw std::invalid_argument("fast device captured forward extent");
        for(int first=0;first<rows;first+=16) {
            const int count=std::min(16,rows-first);
            if(first_abs+first+count<=ring_begin) {
                draft_.skip_fresh_prefill_block(count,first_abs+first,stream_);
                continue;
            }
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
        // Both schedules place draft blocks on the same 16-aligned absolute
        // partition; only the target's initial forward differs.
        const int initial=Exl3TextContext::layer_major_initial_rows();
        const int first=std::max(initial,retained_start_for(total));
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
            context_->append_prefill_layer_major(input.subspan(initial),stream_,
                &retained);
            // Discarded calls advance the fresh scope; every committed call
            // lies in the arena [first,total) and commits as one bulk pass.
            for(int forward=initial;forward<total;) {
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
                }
                forward+=forward_rows;
            }
            std::array<const std::uint16_t*,5> sources{};
            for(std::size_t tap=0;tap<5;++tap)
                sources[tap]=arena.ptr+tap*static_cast<std::size_t>(rows)*tap_hidden;
            draft_.commit_prefill_bulk(sources.data(),rows,first,stream_);
            check(poll_then_sync(stream_),
                "fast device retained tap completion before arena release");
        }catch(...) {
            poll_then_sync(stream_);
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
        // The target checkpoint only reads settled target state, which the
        // draft never touches: begin the transaction first so its copies
        // overlap the draft (the verifier then reuses it).
        if(deferred && !context_->transaction_active())
            context_->begin_transaction(stream_);
        // Verification tree (verification_tree.h): the draft proposes the
        // full chain, then the round keeps a chain prefix and fills the
        // remaining verifier rows with up to VOIDINFER_DFLASH2_SIBLINGS sibling
        // leaves at the depths where they add the most expected acceptance.
        const bool tree=deferred && sibling_count()>0;
        draft_.set_proposal_candidates(tree);
        const auto draft_start=Clock::now();
        value.target_seed_ms=std::chrono::duration<double,std::milli>(
            draft_start-proposal_start).count();
        auto proposed=draft_.propose_cached_view(
            std::span<const std::int64_t>(value.proposal).first(value.width),
            value.root_position,context_->target_embedding(),
            context_->target_lm_head_weights(),
            context_->target_lm_head_metadata(),248070,stream_);
        if(proposed.size()!=static_cast<std::size_t>(value.width-1) ||
           (tree && draft_.last_candidate_ids().size()!=proposed.size()*16))
            throw std::runtime_error("fast device round draft proposal extent");
        value.draft_api_ms=elapsed(draft_start);
        std::copy(proposed.begin(),proposed.end(),value.proposal.begin()+1);
        std::vector<int> sibling_offsets;
        if(tree) {
            const auto shape=exl3_build_verification_tree(proposed,
                draft_.last_candidate_ids(),draft_.last_candidate_unary(),
                value.width-1,sibling_count(),
                draft_.batched_rounds()?0.0f:kExl3TreeRowCost);
            std::copy(shape.sibling_tokens.begin(),shape.sibling_tokens.end(),
                      value.proposal.begin()+1+shape.chain_drafts);
            sibling_offsets=shape.sibling_offsets;
            value.width=1+shape.chain_drafts+static_cast<int>(sibling_offsets.size());
        }
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
        // Coupled sampling keys each verifier row by the token position it
        // predicts: chain rows by their index, sibling leaves by their depth.
        std::vector<int> row_depths;
        if(context_->device_sampling().active() && !sibling_offsets.empty()) {
            for(int row=0;row<value.width-static_cast<int>(sibling_offsets.size());++row)
                row_depths.push_back(row);
            row_depths.insert(row_depths.end(),sibling_offsets.begin(),sibling_offsets.end());
        }
        context_->set_sampling_row_depths(row_depths);
        // Presence penalty: each row's drafted history ([seed, chain, siblings]).
        if(context_->device_sampling().presence_penalty!=0.0f)
            context_->set_sampling_row_inputs(std::span<const std::int64_t>(value.proposal).first(value.width));
        struct RowDepthReset {
            Exl3TextContext& context;
            ~RowDepthReset() {context.set_sampling_row_depths({});context.set_sampling_row_inputs({});}
        } row_depth_reset{*context_};
        if(before_verify)before_verify(*context_);
        const auto verifier_start=Clock::now();
        value.verification=verify_exl3_outer_device_resident_reference(
            *context_,std::span<const std::int64_t>(value.proposal).first(value.width),
            terminal,stream_,&binding,&consumer,timeline,
            seed?&*seed:nullptr,reuse_seed_?coherent_policy:0,settlement,sibling_offsets);
        value.verifier_ms=elapsed(verifier_start);
        if(after_verify)after_verify(*context_);
        value.committed_tokens=value.verification.committed_tokens;
        if(const char* log_path=std::getenv("VOIDINFER_DFLASH2_ROUND_LOG")) {
            // Diagnostic-only round record: root position, proposal chain,
            // committed tokens, the draft's per-position top-16 candidates and
            // their unary scores (all drafted positions).
            const int rows=static_cast<int>(proposed.size());
            std::vector<std::int64_t> candidates(static_cast<std::size_t>(rows)*16);
            if(rows>0 && draft_.last_topk_ids_device_for_test())
                check(cudaMemcpy(candidates.data(),draft_.last_topk_ids_device_for_test(),
                        candidates.size()*sizeof(std::int64_t),cudaMemcpyDeviceToHost),
                    "round log top-16 candidates");
            // One write per record (concurrent lanes append to one file):
            // context pointer, sibling offsets and promotions lead the record.
            std::ostringstream log;
            log<<static_cast<const void*>(context_.get())<<" sib=";
            for(const auto offset:sibling_offsets)log<<offset<<',';
            log<<" promo="<<value.verification.sibling_promotions<<' ';
            log<<value.root_position<<';';
            for(int i=0;i<value.width;++i)log<<value.proposal[static_cast<std::size_t>(i)]<<(i+1<value.width?' ':';');
            for(std::size_t i=0;i<value.committed_tokens.size();++i)
                log<<value.committed_tokens[i]<<(i+1<value.committed_tokens.size()?' ':';');
            std::vector<float> unary(candidates.size());
            if(rows>0 && draft_.last_topk_values_device_for_test())
                check(cudaMemcpy(unary.data(),draft_.last_topk_values_device_for_test(),
                        unary.size()*sizeof(float),cudaMemcpyDeviceToHost),
                    "round log top-16 unary scores");
            for(std::size_t i=0;i<candidates.size();++i)
                log<<candidates[i]<<(i+1<candidates.size()?' ':';');
            for(std::size_t i=0;i<unary.size();++i)
                log<<unary[i]<<(i+1<unary.size()?' ':'\n');
            if(candidates.empty())log<<'\n';
            static std::mutex log_mutex;
            const std::lock_guard log_lock(log_mutex);
            std::ofstream(log_path,std::ios::app)<<log.str();
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

    // VOIDINFER_DFLASH2_SIBLINGS: maximum sibling leaves per verification tree
    // (0: plain chain verification).
    static int sibling_count() {
        static const int count=[] {
            const char* value=std::getenv("VOIDINFER_DFLASH2_SIBLINGS");
            const int parsed=value?std::atoi(value):6;
            if(parsed<0 || parsed>6)
                throw std::invalid_argument("VOIDINFER_DFLASH2_SIBLINGS must be 0..6");
            return parsed;
        }();
        return count;
    }

    std::shared_ptr<Exl3TextContext> context_;
    Exl3Dflash2DraftModel& draft_;
    std::array<std::uint16_t*,5> staging_{};
    std::uint64_t acquisition_=0,execution_=0;
    bool reuse_seed_=false;
    cudaStream_t stream_=nullptr;
    Phase phase_=Phase::unstarted;
    int frontier_=0,verified_frontier_=0,verify_rows_=0;
    Exl3TextContext* line_=nullptr;
    // Worker thread and stream of the asynchronous line pass: forwards the
    // block's rows, then scores them (the last row against a placeholder).
    class AsyncLine {
    public:
        AsyncLine() {
            check(cudaGetDevice(&device_),"asynchronous verification device");
            // Lowest priority: the pass is bound by the PCIe history stream, so
            // its kernels fill SMs the decode lane leaves idle.
            // VOIDINFER_EXL3_VERICACHE_ASYNC_PRIORITY=1: highest instead.
            int least=0,greatest=0;
            check(cudaDeviceGetStreamPriorityRange(&least,&greatest),"asynchronous verification priority");
            const char* priority=std::getenv("VOIDINFER_EXL3_VERICACHE_ASYNC_PRIORITY");
            check(cudaStreamCreateWithPriority(&stream_,cudaStreamNonBlocking,
                priority && std::string_view(priority)=="1"?greatest:least),"asynchronous verification stream");
            check(cudaEventCreateWithFlags(&fork_,cudaEventDisableTiming),"asynchronous verification event");
            worker_=std::thread([this] {run();});
        }
        ~AsyncLine() {
            {const std::lock_guard lock(mutex_);quit_=true;}
            cv_.notify_all();
            if(worker_.joinable())worker_.join();
            if(stream_){(void)poll_then_sync(stream_);cudaStreamDestroy(stream_);}
            if(fork_)cudaEventDestroy(fork_);
        }
        AsyncLine(const AsyncLine&)=delete;
        AsyncLine& operator=(const AsyncLine&)=delete;
        void launch(Exl3ExactLinePool& pool,Exl3ExactLinePool::Lane& lane,Exl3TextContext& drafting,
                    std::span<const std::int64_t> tokens,int root,float delta,cudaStream_t after,PassOrder* order) {
            check(cudaEventRecord(fork_,after),"asynchronous verification fork");
            {
                const std::lock_guard lock(mutex_);
                pool_=&pool;
                lane_=&lane;
                drafting_=&drafting;
                root_=root;
                order_=order;
                tokens_.assign(tokens.begin(),tokens.end());
                delta_=delta;
                job_=true;done_=false;error_=nullptr;
            }
            cv_.notify_all();
        }
        bool done() {const std::lock_guard lock(mutex_);return done_;}
        std::vector<Exl3TextContext::RowScore> wait() {
            std::unique_lock lock(mutex_);
            cv_.wait(lock,[&] {return done_;});
            done_=false;
            if(error_)std::rethrow_exception(error_);
            return std::move(scores_);
        }
    private:
        void run() {
            (void)cudaSetDevice(device_);
            l0_exact_host_throttle()=true;
            std::unique_lock lock(mutex_);
            for(;;) {
                cv_.wait(lock,[&] {return job_ || quit_;});
                if(quit_)return;
                job_=false;
                Exl3ExactLinePool& pool=*pool_;
                Exl3ExactLinePool::Lane& lane=*lane_;
                Exl3TextContext& drafting=*drafting_;
                const int root=root_;
                const std::vector<std::int64_t> tokens=tokens_;
                const float delta=delta_;
                PassOrder* const order=order_;
                lock.unlock();
                std::vector<Exl3TextContext::RowScore> scores;
                std::exception_ptr error;
                try {
                    check(cudaStreamWaitEvent(stream_,fork_,0),"asynchronous verification fork");
                    const OrderedPass ordered(order,stream_);
                    Exl3ExactLinePool::Lease lease(pool,stream_);
                    auto& line=lease.at_root(lane,drafting,drafting.device_sampling());
                    if(line.position()!=root)throw std::logic_error("asynchronous verification line root");
                    forward_rows(line,tokens,stream_);
                    std::vector<std::int64_t> next(tokens.begin()+1,tokens.end());
                    next.push_back(tokens.front());
                    scores=line.exact_row_scores(next,stream_,delta);
                    lease.ran_pass();
                } catch(...) {error=std::current_exception();}
                lock.lock();
                finished_=Clock::now();
                scores_=std::move(scores);
                error_=error;
                done_=true;
                cv_.notify_all();
            }
        }
    public:
        Clock::time_point finished_{};
    private:
        int device_=0;
        cudaStream_t stream_=nullptr;
        cudaEvent_t fork_=nullptr;
        std::mutex mutex_;
        std::condition_variable cv_;
        bool job_=false,done_=false,quit_=false;
        Exl3ExactLinePool* pool_=nullptr;
        Exl3ExactLinePool::Lane* lane_=nullptr;
        Exl3TextContext* drafting_=nullptr;
        int root_=0;
        PassOrder* order_=nullptr;
        std::vector<std::int64_t> tokens_;
        float delta_=0.0f;
        std::vector<Exl3TextContext::RowScore> scores_;
        std::exception_ptr error_;
        std::thread worker_;
    };
    std::unique_ptr<AsyncLine> async_;
    Exl3ExactLinePool* pool_=nullptr;
    Exl3ExactLinePool::Lane* lane_=nullptr;
    PassOrder* pass_order_=nullptr;
    bool async_inflight_=false;
    std::vector<std::int64_t> line_head_;   // published tokens the line has not run
    std::size_t async_head_=0;              // head rows of the block in flight
    int async_root_=0;                      // line frontier the block starts at
    Clock::time_point async_start_{};
    struct FoldCommit {
        bool pending=false,corrected=false,round_followed=false;
        std::vector<std::int64_t> tokens;
    } fold_commit_;
    bool fold_armed_=false;
    int fold_frontier_=0;
    std::vector<std::int64_t> fold_tokens_;
    Totals totals_{};
    std::uint64_t pending_ticket_=0;
    std::optional<Step> pending_;
    std::shared_ptr<const Exl3ExactHostState> pending_root_;
    std::shared_ptr<const Exl3DraftHostRing> pending_ring_;
    std::vector<std::int64_t> pending_terminal_;
    std::optional<Step> device_pending_;
    std::vector<std::int64_t> device_pending_terminal_;
};

} // namespace voidinfer::exl3

static_assert(voidinfer::exl3::Exl3FastDeviceRound::retained_start_for(4096)==2048);
static_assert(voidinfer::exl3::Exl3FastDeviceRound::retained_start_for(16384)==14336);
static_assert(voidinfer::exl3::Exl3FastDeviceRound::retained_start_for(4097)==2048);
