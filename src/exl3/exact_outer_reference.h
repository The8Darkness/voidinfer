#pragma once
#include "exl3/branch_reference.h"
#include <chrono>
#include <exception>

namespace ninfer::exl3 {
inline bool exl3_fast_mia_parity_w1_enabled() {
    const char* requested=std::getenv("NINFER_EXL3_FAST_MIA_PARITY_W1");
    if(requested && std::string_view(requested)!="0" &&
       std::string_view(requested)!="1")
        throw std::invalid_argument("FAST_MIA_PARITY W+1 must be 0 or 1");
    return requested && std::string_view(requested)=="1";
}
inline void exl3_outer_greedy_path(Exl3TextContext& exact,std::int64_t seed,
    std::span<std::int64_t> path,cudaStream_t stream,
    std::int64_t* deferred_bonus=nullptr) {
    if(exl3_device_greedy_enabled()) {
        const auto packet=exact.greedy_packet(true,stream);
        exl3_greedy_path(seed,packet,path);
        if(deferred_bonus) *deferred_bonus=packet.decisions[path.size()-1].token;
        return;
    }
    const auto logits=exact.continuation_logits_host(stream);
    if(logits.size()!=path.size()*248320 ||
        std::any_of(logits.begin(),logits.end(),[](float x){return !std::isfinite(x);}))
        throw std::runtime_error("outer nonfinite/invalid logits");
    path[0]=seed;
    for(std::size_t row=1;row<path.size();++row) {
        const auto first=logits.begin()+(row-1)*248320;
        path[row]=std::max_element(first,first+248320)-first;
    }
    if(deferred_bonus) {
        const auto first=logits.begin()+(path.size()-1)*248320;
        *deferred_bonus=std::max_element(first,first+248320)-first;
    }
}

inline void exl3_stage_committed_taps(Exl3TextContext& exact,
    const std::array<std::uint16_t*,5>& staging,int rows,cudaStream_t stream) {
    constexpr std::array<int,5> layers{5,19,33,47,61};
    for(auto* p:staging) if(!p) throw std::invalid_argument("committed tap destination");
    for(int tap=0;tap<5;++tap)
        exact.copy_tap_rows_to_device(layers[tap],0,staging[tap],rows,stream);
}
struct Exl3OuterDecision {
    std::vector<std::int64_t> committed_tokens;
    std::size_t accepted=0;
    bool rejected=false,stopped=false;
};

struct Exl3CommittedTapSegmentMetadata {
    int source_position=0,source_first=0,rows=0;
    int logical_first=0,destination_first=0;
    int attempt_first=0,attempt_rows=0;
    int source_partition_first=0,source_partition_rows=0;
    bool repair=false,correction=false;
    bool operator==(const Exl3CommittedTapSegmentMetadata&) const = default;
};
using Exl3CommittedTapConsumer=
    std::function<void(const Exl3CommittedTapSegment&,cudaStream_t)>;

inline bool exl3_terminal_token(std::int64_t token,std::span<const std::int64_t> terminal) {
    return std::find(terminal.begin(),terminal.end(),token)!=terminal.end();
}

// Target predictions after the first mismatch are under a wrong ancestor and
// cannot authorize output. Terminal tokens also truncate the consumed path.
inline Exl3OuterDecision decide_exl3_outer_prefix(std::span<const std::int64_t> target_path,
    std::span<const std::int64_t> tentative,std::span<const std::int64_t> terminal={}) {
    if(target_path.empty() || target_path.size()!=tentative.size())
        throw std::invalid_argument("outer decision extent");
    Exl3OuterDecision result;
    result.committed_tokens.reserve(tentative.size());
    for(std::size_t row=0;row<tentative.size();++row) {
        const auto authorized=target_path[row];
        result.committed_tokens.push_back(authorized);
        if(tentative[row]==authorized) ++result.accepted;
        else result.rejected=true;
        result.stopped=exl3_terminal_token(authorized,terminal);
        if(result.rejected || result.stopped) break;
    }
    return result;
}

struct Exl3OuterReferenceResult {
    std::shared_ptr<const Exl3ExactHostState> committed_state;
    std::vector<std::int64_t> committed_tokens;
    std::size_t accepted = 0;
    std::size_t executed_rows = 0;
    std::size_t verification_rows = 0;
    std::size_t replay_rows = 0;
    std::size_t native_invocations = 0;
    std::size_t root_restores = 0;
    std::size_t checkpoint_captured_bytes = 0;
    std::size_t checkpoint_restores = 0;
    std::size_t checkpoint_reconstructed_rows = 0;
    std::size_t checkpoint_fallback_rows = 0;
    std::size_t sibling_promotions = 0;
    bool device_seed_reused = false;
    bool device_seed_fallback = false;
    std::size_t committed_tap_d2d_bytes = 0;
    // Committed rows whose five captured tap planes stayed device-resident.
    // The D2D consumer still stages every row into the authoritative draft
    // ingestion buffers; this counts only the avoided host materialization.
    std::size_t committed_tap_host_export_rows_avoided = 0;
    std::size_t discarded_tentative_tap_segments = 0;
    std::uint64_t committed_tap_ready_generation = 0;
    std::uintptr_t committed_tap_ready_event = 0;
    std::array<std::uint64_t,17> native_batch_hist{};
    // Default-off FAST_MIA_PARITY settlement. This is a target-authoritative
    // W+1 row computed by the same continuation forward, retained as the next
    // round's seed rather than published early. That keeps output-budget and
    // stop-token accounting at the existing publication boundary.
    std::size_t fast_mia_parity_w1_bonus_rows = 0;
    std::optional<std::int64_t> fast_mia_parity_pending;
    bool rejected = false;
    bool stopped = false;
    std::array<std::vector<std::uint16_t>,5> committed_taps;
    std::vector<Exl3CommittedTapSegmentMetadata> committed_tap_segments;
};

inline void exl3_emit_committed_tap_segment(Exl3OuterReferenceResult& result,
    Exl3TextContext& exact,const Exl3CommittedTapBinding* binding,
    const Exl3CommittedTapConsumer* consumer,int source_first,int rows,
    int logical_first,int destination_first,int attempt_first,int attempt_rows,
    int source_partition_first,int source_partition_rows,bool repair,
    bool correction,cudaStream_t stream) {
    if(rows<1)return;
    result.committed_tap_segments.push_back({exact.position()-exact.captured_tap_rows(),
        source_first,rows,logical_first,destination_first,attempt_first,attempt_rows,
        source_partition_first,source_partition_rows,repair,correction});
    if(!consumer)return;
    if(!binding || !*binding)
        throw std::invalid_argument("committed tap consumer missing source binding");
    auto segment=exact.committed_tap_segment(*binding,source_first,rows,
        logical_first,destination_first,attempt_first,attempt_rows,
        source_partition_first,source_partition_rows,repair,correction);
    (*consumer)(segment,stream);
}

// Default-off bounded repair for an ordinary-FP16 B2..B8 verifier.  The one
// complete root checkpoint is captured before teacher forcing.  A rejected
// proposal row is never retained: accepted ancestor rows are reconstructed
// from their immediate continuation scratch, then the authoritative correction
// is executed separately.  First-row rejection has no valid nonempty retained
// boundary and therefore rolls back to the root for the scalar correction.
inline Exl3OuterReferenceResult verify_exl3_outer_checkpointed_reference(
    Exl3TextContext& exact, const Exl3ExactHostState& root,
    std::span<const std::int64_t> tentative, bool capture_taps = false,
    std::span<const std::int64_t> terminal = {}, cudaStream_t stream = nullptr,
    bool reuse_resident_root = false,
    const Exl3CommittedTapBinding* tap_binding=nullptr,
    const Exl3CommittedTapConsumer* tap_consumer=nullptr) {
    if(tentative.size()<2 || tentative.size()>8 ||
       exact.continuation_capacity()<tentative.size() ||
       !exact.transaction_prepared())
        throw std::invalid_argument(
            "outer checkpointed reference requires prepared B2..B8 transaction");
    for(auto token:tentative) if(token<0 || token>=248320)
        throw std::invalid_argument("outer checkpointed token extent");
    for(auto token:terminal) if(token<0 || token>=248320)
        throw std::invalid_argument("outer checkpointed terminal token extent");
    if(tap_consumer && (!tap_binding || !*tap_binding))
        throw std::invalid_argument("outer checkpointed tap binding");
    const bool fast_mia_parity_w1=exl3_fast_mia_parity_w1_enabled();

    Exl3OuterReferenceResult result;
    result.committed_tap_segments.reserve(tentative.size());
    if(reuse_resident_root)
        result.root_restores=exact.restore_exact_host_state_if_needed(root,stream)?1:0;
    else {
        exact.restore_exact_host_state(root,stream);
        result.root_restores=1;
    }
    try {
        exact.begin_transaction(stream);
        result.checkpoint_captured_bytes=exact.transaction_bytes();
        const auto authorized=exl3_branch_greedy(exact,stream);
        exact.continue_rows(tentative,stream);
        result.verification_rows=tentative.size();
        result.native_invocations=1;
        ++result.native_batch_hist[tentative.size()];
        std::array<std::int64_t,8> path_storage{};
        auto path=std::span<std::int64_t>(path_storage).first(tentative.size());
        std::int64_t deferred_bonus=-1;
        exl3_outer_greedy_path(exact,authorized,path,stream,
            fast_mia_parity_w1?&deferred_bonus:nullptr);
        auto decision=decide_exl3_outer_prefix(path,tentative,terminal);
        result.committed_tokens=std::move(decision.committed_tokens);
        result.accepted=decision.accepted;
        result.rejected=decision.rejected;
        result.stopped=decision.stopped;
        const bool truncated=result.committed_tokens.size()<tentative.size();
        if(fast_mia_parity_w1 && !result.rejected && !result.stopped && !truncated) {
            if(deferred_bonus<0 || deferred_bonus>=248320)
                throw std::runtime_error("FAST_MIA_PARITY W+1 bonus extent");
            result.fast_mia_parity_pending=deferred_bonus;
            result.fast_mia_parity_w1_bonus_rows=1;
        }
        if(capture_taps)
            for(auto& taps:result.committed_taps)
                taps.reserve(result.committed_tokens.size()*5120);

        const auto append_host_taps=[&] {
            if(!capture_taps)return;
            auto rows=exact.exact_tap_rows_host(stream);
            for(int tap=0;tap<5;++tap)
                result.committed_taps[tap].insert(
                    result.committed_taps[tap].end(),rows[tap].begin(),rows[tap].end());
        };
        const auto stage_taps=[&](int source_first,int rows,int destination_first,
                                  bool repair,bool correction=false) {
            if(!capture_taps && !tap_consumer)return;
            exl3_emit_committed_tap_segment(result,exact,tap_binding,tap_consumer,
                source_first,rows,root.position()+destination_first,destination_first,
                root.position(),static_cast<int>(tentative.size()),
                exact.position()-exact.captured_tap_rows(),exact.captured_tap_rows(),
                repair,correction,stream);
        };

        if(result.rejected) {
            const int accepted=static_cast<int>(result.accepted);
            const auto correction=result.committed_tokens.back();
            if(accepted>0) {
                exact.retain_transaction_prefix(accepted,stream);
                ++result.checkpoint_restores;
                result.checkpoint_reconstructed_rows=accepted;
                append_host_taps();
                stage_taps(0,accepted,0,true);
            } else {
                exact.rollback_transaction(stream);
                ++result.checkpoint_fallback_rows;
            }
            exact.decode(correction,stream);
            ++result.replay_rows;
            ++result.native_invocations;
            ++result.native_batch_hist[1];
            append_host_taps();
            stage_taps(0,1,accepted,true,true);
            if(accepted>0) exact.commit_transaction();
        } else {
            const int retained=static_cast<int>(result.committed_tokens.size());
            if(truncated) {
                exact.retain_transaction_prefix(retained,stream);
                ++result.checkpoint_restores;
                result.checkpoint_reconstructed_rows=retained;
            }
            append_host_taps();
            stage_taps(0,retained,0,truncated);
            exact.commit_transaction();
            if(exact.continuation_rows()>0) exact.finish_exact_continuation(stream);
        }
        result.executed_rows=result.verification_rows+result.replay_rows;
        result.committed_state=exact.export_exact_host_state(stream);
    } catch(...) {
        const auto failure=std::current_exception();
        try {
            if(exact.transaction_active()) exact.rollback_transaction(stream);
            else exact.restore_exact_host_state(root,stream);
        } catch(...) {}
        std::rethrow_exception(failure);
    }
    return result;
}

// Guarded ordinary-device-KV transaction. The caller owns the live context and
// its frontier; this function does not restore or export a full host state.
// Decisions are returned as a compact host packet and the committed target
// state remains on the device. A caller needing an oracle may export after the
// timed call. It must discard the context on a post-commit failure.
struct Exl3OuterDeviceStageTimeline {
    double begin_ms=0, seed_ms=0, submit_ms=0, decision_ms=0, settlement_ms=0;
};
// A host-ready target seed can be carried across draft proposal without a
// second greedy read. The owner retains the context, while root_revision is
// replaced after every committed frontier. The numerical policy tag is owned
// by the request route, rather than inferred from the tentative token values.
struct Exl3OuterDeviceSeedPacket {
    Exl3CommittedTapBinding binding;
    std::uint64_t request_generation=0, numerical_policy=0;
    std::int64_t token=-1;
    cudaStream_t completed_stream=nullptr;
    bool device_greedy=false;
};
inline Exl3OuterDeviceSeedPacket bind_exl3_outer_device_seed(
    const Exl3TextContext& exact,const Exl3CommittedTapBinding& binding,
    std::int64_t token,std::uint64_t numerical_policy,
    cudaStream_t completed_stream=nullptr) {
    if(!binding || binding.context_owner.get()!=&exact ||
       binding.root_position!=exact.position() ||
       token<0 || token>=248320 || !numerical_policy)
        throw std::invalid_argument("device seed packet root/policy extent");
    return {binding,exact.request_generation(),numerical_policy,token,
        completed_stream,
        exl3_device_greedy_enabled()};
}
inline bool exl3_outer_device_seed_matches(
    const Exl3TextContext& exact,const Exl3CommittedTapBinding* tap_binding,
    const Exl3OuterDeviceSeedPacket* ready_seed,
    std::uint64_t numerical_policy,std::int64_t proposed_root,
    cudaStream_t stream=nullptr) {
    return ready_seed && tap_binding && *tap_binding &&
        tap_binding->context_owner.get()==&exact &&
        ready_seed->binding.context_owner==tap_binding->context_owner &&
        ready_seed->binding.root_revision==tap_binding->root_revision &&
        ready_seed->binding.root_position==exact.position() &&
        ready_seed->binding.acquisition==tap_binding->acquisition &&
        ready_seed->binding.execution==tap_binding->execution &&
        ready_seed->request_generation==exact.request_generation() &&
        ready_seed->numerical_policy &&
        ready_seed->numerical_policy==numerical_policy &&
        ready_seed->completed_stream==stream &&
        ready_seed->device_greedy==exl3_device_greedy_enabled() &&
        ready_seed->token==proposed_root &&
        ready_seed->token>=0 && ready_seed->token<248320;
}
// A rejected round's correction row is deferred to the next round's verifier
// (row 0) instead of being decoded by a separate M1 pass. Measured default;
// NINFER_EXL3_FOLD_CORRECTION=0 restores the separate correction pass.
inline bool exl3_fold_correction_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("NINFER_EXL3_FOLD_CORRECTION");
        if(!value)return true;
        if(std::string_view(value)=="0")return false;
        if(std::string_view(value)=="1")return true;
        throw std::invalid_argument("NINFER_EXL3_FOLD_CORRECTION must be 0 or 1");
    }();
    return enabled;
}

enum class Exl3OuterDeviceSettlement : std::uint8_t {
    Eager,
    DeferTargetCommit,
};

inline Exl3OuterReferenceResult verify_exl3_outer_device_resident_reference(
    Exl3TextContext& exact,std::span<const std::int64_t> tentative,
    std::span<const std::int64_t> terminal={},cudaStream_t stream=nullptr,
    const Exl3CommittedTapBinding* tap_binding=nullptr,
    const Exl3CommittedTapConsumer* tap_consumer=nullptr,
    Exl3OuterDeviceStageTimeline* timeline=nullptr,
    const Exl3OuterDeviceSeedPacket* ready_seed=nullptr,
    std::uint64_t numerical_policy=0,
    Exl3OuterDeviceSettlement settlement=Exl3OuterDeviceSettlement::Eager,
    std::span<const int> sibling_offsets={}) {
    if(tentative.size()<2 || tentative.size()>8 ||
       exact.continuation_capacity()<tentative.size() ||
       !exact.transaction_prepared())
        throw std::invalid_argument(
            "device-resident verifier requires prepared B2..B8 transaction");
    // tentative = [seed, chain drafts, sibling leaves]: sibling j is an
    // alternative to chain row sibling_offsets[j] (sibling_rows.cuh).
    const int siblings=static_cast<int>(sibling_offsets.size());
    const int chain_rows=static_cast<int>(tentative.size())-siblings;
    if(siblings<0 || chain_rows<2 ||
       (siblings && (!exl3_device_greedy_enabled() || !exl3_fold_correction_enabled())))
        throw std::invalid_argument("device-resident verifier sibling layout");
    for(auto token:tentative) if(token<0 || token>=248320)
        throw std::invalid_argument("device-resident tentative token extent");
    for(auto token:terminal) if(token<0 || token>=248320)
        throw std::invalid_argument("device-resident terminal token extent");
    if(tap_consumer && (!tap_binding || !*tap_binding))
        throw std::invalid_argument("device-resident tap binding");
    const auto* device_kv=std::getenv("NINFER_EXL3_FAST_DEVICE_KV_TRANSACTION");
    if(!device_kv || std::string_view(device_kv)!="1")
        throw std::invalid_argument("device-resident verifier requires its guarded device-KV policy");

    Exl3OuterReferenceResult result;
    result.committed_tap_segments.reserve(tentative.size());
    const int root_position=exact.position();
    const bool fast_w1=exl3_fast_mia_parity_w1_enabled();
    const bool reuse_seed=exl3_outer_device_seed_matches(
        exact,tap_binding,ready_seed,numerical_policy,tentative.front(),stream);
    result.device_seed_reused=reuse_seed;
    result.device_seed_fallback=ready_seed && !reuse_seed;
    using StageClock=std::chrono::steady_clock;
    auto stage_start=timeline?StageClock::now():StageClock::time_point{};
    const auto stage_end=[&](double& ms) {
        if(!timeline)return;
        const auto now=StageClock::now();
        ms=std::chrono::duration<double,std::milli>(now-stage_start).count();
        stage_start=now;
    };
    try {
        // A caller may begin the transaction before drafting so its
        // checkpoint overlaps the draft.
        if(!exact.transaction_active())exact.begin_transaction(stream);
        result.checkpoint_captured_bytes=exact.transaction_bytes();
        if(timeline)stage_end(timeline->begin_ms);
        // An invalid or stale packet takes the ordinary checked reduction.
        // The valid host token was already synchronized before draft proposal.
        const auto seed=reuse_seed?ready_seed->token:exl3_branch_greedy(exact,stream);
        if(timeline)stage_end(timeline->seed_ms);
        if(siblings)exact.set_verifier_siblings(sibling_offsets);
        exact.continue_rows(tentative,stream);
        if(timeline)stage_end(timeline->submit_ms);
        result.verification_rows=tentative.size();
        result.native_invocations=1;
        ++result.native_batch_hist[tentative.size()];
        std::array<std::int64_t,8> path_storage{};
        auto path=std::span<std::int64_t>(path_storage).first(tentative.size());
        std::int64_t deferred_bonus=-1;
        int promoted_sibling=-1;
        Exl3OuterDecision decision;
        if(siblings) {
            const auto packet=exact.greedy_packet(true,stream);
            auto chain_path=path.first(static_cast<std::size_t>(chain_rows));
            chain_path[0]=seed;
            for(int row=1;row<chain_rows;++row)
                chain_path[static_cast<std::size_t>(row)]=packet.decisions[row-1].token;
            decision=decide_exl3_outer_prefix(chain_path,
                tentative.first(static_cast<std::size_t>(chain_rows)),terminal);
            // A rejected chain row d whose target token equals a sibling at
            // offset d continues through that sibling's row.
            const int depth=static_cast<int>(decision.accepted);
            for(int j=0;j<siblings && decision.rejected && !decision.stopped;++j)
                if(sibling_offsets[static_cast<std::size_t>(j)]==depth &&
                   tentative[static_cast<std::size_t>(chain_rows+j)]==
                       decision.committed_tokens.back()) {
                    promoted_sibling=chain_rows+j;
                    break;
                }
            if(promoted_sibling>=0) {
                const auto next=packet.decisions[promoted_sibling].token;
                decision.committed_tokens.push_back(next);
                decision.accepted=static_cast<std::size_t>(depth)+1;
                decision.stopped=exl3_terminal_token(next,terminal);
                ++result.sibling_promotions;
            }
        } else {
        exl3_outer_greedy_path(exact,seed,path,stream,
            fast_w1?&deferred_bonus:nullptr);
        decision=decide_exl3_outer_prefix(path,tentative,terminal);
        }
        result.committed_tokens=std::move(decision.committed_tokens);
        result.accepted=decision.accepted;
        result.rejected=decision.rejected;
        result.stopped=decision.stopped;
        if(timeline)stage_end(timeline->decision_ms);
        const bool truncated=result.committed_tokens.size()<tentative.size();
        if(fast_w1 && !result.rejected && !result.stopped && !truncated) {
            if(deferred_bonus<0 || deferred_bonus>=248320)
                throw std::runtime_error("device-resident W+1 bonus extent");
            result.fast_mia_parity_pending=deferred_bonus;
            result.fast_mia_parity_w1_bonus_rows=1;
        }
        const auto stage_taps=[&](int source_first,int rows,
                                  int destination_first,bool repair,
                                  bool correction=false) {
            if(!tap_consumer)return;
            exl3_emit_committed_tap_segment(result,exact,tap_binding,
                tap_consumer,source_first,rows,
                root_position+destination_first,destination_first,
                root_position,static_cast<int>(tentative.size()),
                exact.position()-exact.captured_tap_rows(),
                exact.captured_tap_rows(),repair,correction,stream);
        };
        if(result.rejected && result.accepted>0 && exl3_fold_correction_enabled()) {
            // Folded correction: keep only the verified prefix. Retention
            // restores the last retained row's logits, so the next round's
            // seed is exactly this correction and its verifier consumes it as
            // row 0. The correction is published only when consumed.
            const int accepted=static_cast<int>(result.accepted);
            if(promoted_sibling>=0)
                exact.promote_sibling_row(promoted_sibling,accepted-1,stream);
            exact.retain_transaction_prefix(accepted,stream);
            ++result.checkpoint_restores;
            result.checkpoint_reconstructed_rows=accepted;
            stage_taps(0,accepted,0,true);
            result.committed_tokens.pop_back();
            result.stopped=exl3_terminal_token(result.committed_tokens.back(),terminal);
            if(settlement==Exl3OuterDeviceSettlement::Eager)
                exact.commit_transaction();
        } else if(result.rejected) {
            const int accepted=static_cast<int>(result.accepted);
            const auto correction=result.committed_tokens.back();
            if(accepted>0) {
                exact.retain_transaction_prefix(accepted,stream);
                ++result.checkpoint_restores;
                result.checkpoint_reconstructed_rows=accepted;
                stage_taps(0,accepted,0,true);
            } else {
                exact.rollback_transaction(stream);
                ++result.checkpoint_fallback_rows;
                // A pending Engine preview still needs a rollback boundary for
                // the correction row. Capture the restored root before replay.
                if(settlement==Exl3OuterDeviceSettlement::DeferTargetCommit)
                    exact.begin_transaction(stream);
            }
            exact.set_target_projection_timing_phase(Exl3TargetProjectionPhase::replay);
            exact.decode(correction,stream);
            ++result.replay_rows;
            ++result.native_invocations;
            ++result.native_batch_hist[1];
            stage_taps(0,1,accepted,true,true);
            if(accepted>0 && settlement==Exl3OuterDeviceSettlement::Eager)
                exact.commit_transaction();
        } else {
            const int retained=static_cast<int>(result.committed_tokens.size());
            if(truncated) {
                exact.retain_transaction_prefix(retained,stream);
                ++result.checkpoint_restores;
                result.checkpoint_reconstructed_rows=retained;
            }
            stage_taps(0,retained,0,truncated);
            if(settlement==Exl3OuterDeviceSettlement::Eager) {
                exact.commit_transaction();
                if(exact.continuation_rows()>0)
                    exact.finish_exact_continuation(stream);
            }
        }
        result.executed_rows=result.verification_rows+result.replay_rows;
        if(timeline)stage_end(timeline->settlement_ms);
        if(settlement==Exl3OuterDeviceSettlement::DeferTargetCommit &&
           !exact.transaction_active())
            throw std::logic_error("device-resident pending verifier lost target rollback boundary");
        // No full host-state export is performed here; the live context owns
        // the committed device KV, GDN state and captured tap frontier.
    } catch(...) {
        const auto failure=std::current_exception();
        try {
            if(exact.transaction_active()) exact.rollback_transaction(stream);
            else exact.reset(stream);
        } catch(...) {}
        std::rethrow_exception(failure);
    }
    return result;
}

// The inner OSCAR/DFlash2 stream is only a proposal. Greedy verification runs
// against an ordinary full-FP16 context restored from authoritative L2. A
// returned token has been consumed into authoritative KV/recurrent state.
// No callbacks/publication during tentative work. On failure restore root;
// caller must invalidate/rebuild its inner target, ring and pending anchor.
inline Exl3OuterReferenceResult verify_exl3_outer_reference(
    Exl3TextContext& exact, const Exl3ExactHostState& root,
    std::span<const std::int64_t> tentative, bool capture_taps = false,
    std::span<const std::int64_t> terminal = {}, cudaStream_t stream = nullptr,
    bool reuse_resident_root = false,
    const Exl3CommittedTapBinding* tap_binding=nullptr,
    const Exl3CommittedTapConsumer* tap_consumer=nullptr) {
    if (tentative.empty() || tentative.size() > 32)
        throw std::invalid_argument("outer reference requires 1..32 tentative tokens");
    for (auto token : tentative)
        if (token < 0 || token >= 248320) throw std::invalid_argument("outer reference token extent");
    for(auto token:terminal) if(token<0 || token>=248320) throw std::invalid_argument("outer terminal token extent");
    if(tap_consumer && (!tap_binding || !*tap_binding))
        throw std::invalid_argument("outer reference tap consumer missing source binding");
    Exl3OuterReferenceResult result;
    result.committed_tap_segments.reserve(tentative.size());
    result.committed_tokens.reserve(tentative.size());
    if(reuse_resident_root)result.root_restores=exact.restore_exact_host_state_if_needed(root,stream)?1:0;
    else {
        exact.restore_exact_host_state(root,stream);result.root_restores=1;
    }
    try {
        for (auto proposal : tentative) {
            const auto authorized = exl3_branch_greedy(exact,stream);
            exact.decode(authorized,stream);
            if(capture_taps) {
                auto taps=exact.exact_tap_rows_host(stream);
                if(tentative.size()==1)result.committed_taps=std::move(taps);
                else for(int layer=0;layer<5;++layer)
                    result.committed_taps[layer].insert(result.committed_taps[layer].end(),
                        taps[layer].begin(),taps[layer].end());
            }
            if(capture_taps || tap_consumer)
                exl3_emit_committed_tap_segment(result,exact,tap_binding,tap_consumer,
                    0,1,root.position()+static_cast<int>(result.committed_tokens.size()),
                    static_cast<int>(result.committed_tokens.size()),
                    root.position()+static_cast<int>(result.committed_tokens.size()),1,
                    exact.position()-1,1,false,false,stream);
            ++result.executed_rows;
            ++result.verification_rows;
            ++result.native_invocations;
            ++result.native_batch_hist[1];
            result.committed_tokens.push_back(authorized);
            if (proposal != authorized) result.rejected = true;
            else ++result.accepted;
            result.stopped=exl3_terminal_token(authorized,terminal);
            if(result.rejected || result.stopped) break;
        }
        result.committed_state = exact.export_exact_host_state(stream);
    } catch (...) {
        const auto failure=std::current_exception();
        try{exact.restore_exact_host_state(root,stream);}catch(...){}
        std::rethrow_exception(failure);
    }
    return result;
}

// Packed ordinary-FP16 target verification. The full attempted suffix is
// tentative. Rejection restores L2 and replays only accepted tokens plus the
// authoritative replacement. Width is a separate outer budget (2..8 rows),
// independent of the inner DFlash2 B including its pending anchor.
inline Exl3OuterReferenceResult verify_exl3_outer_batched_reference(
    Exl3TextContext& exact, const Exl3ExactHostState& root,
    std::span<const std::int64_t> tentative, bool capture_taps = false,
    std::span<const std::int64_t> terminal = {}, cudaStream_t stream = nullptr,
    bool reuse_resident_root = false,
    const Exl3CommittedTapBinding* tap_binding=nullptr,
    const Exl3CommittedTapConsumer* tap_consumer=nullptr) {
    if(tentative.size()<2 || tentative.size()>8 || exact.continuation_capacity()<tentative.size())
        throw std::invalid_argument("outer batched reference requires prepared2..8rows");
    for(auto token:tentative)
        if(token<0 || token>=248320) throw std::invalid_argument("outer batched token extent");
    for(auto token:terminal) if(token<0 || token>=248320) throw std::invalid_argument("outer terminal token extent");
    if(tap_consumer && (!tap_binding || !*tap_binding))
        throw std::invalid_argument("outer batched tap consumer missing source binding");
    const bool fast_mia_parity_w1=exl3_fast_mia_parity_w1_enabled();
    Exl3OuterReferenceResult result;
    result.committed_tap_segments.reserve(tentative.size());
    // The decision below transfers its complete vector; an initial reservation
    // here would coexist with it and then be discarded by move assignment.
    if(reuse_resident_root)result.root_restores=exact.restore_exact_host_state_if_needed(root,stream)?1:0;
    else {
        exact.restore_exact_host_state(root,stream);result.root_restores=1;
    }
    try {
        auto authorized=exl3_branch_greedy(exact,stream);
        exact.continue_rows(tentative,stream);
        result.executed_rows=tentative.size();
        result.verification_rows=tentative.size();
        result.native_invocations=1;
        ++result.native_batch_hist[tentative.size()];
        std::array<std::int64_t,8> path_storage{};
        auto path=std::span<std::int64_t>(path_storage).first(tentative.size());
        std::int64_t deferred_bonus=-1;
        exl3_outer_greedy_path(exact,authorized,path,stream,
            fast_mia_parity_w1?&deferred_bonus:nullptr);
        auto decision=decide_exl3_outer_prefix(path,tentative,terminal);
        result.committed_tokens=std::move(decision.committed_tokens);
        result.accepted=decision.accepted;result.rejected=decision.rejected;result.stopped=decision.stopped;
        const bool repair=result.rejected || result.committed_tokens.size()<tentative.size();
        if(fast_mia_parity_w1 && !result.rejected && !result.stopped && !repair) {
            if(deferred_bonus<0 || deferred_bonus>=248320)
                throw std::runtime_error("FAST_MIA_PARITY W+1 bonus extent");
            result.fast_mia_parity_pending=deferred_bonus;
            result.fast_mia_parity_w1_bonus_rows=1;
        }
        if(repair) {
            exact.restore_exact_host_state(root,stream);
            ++result.root_restores;
            if(result.committed_tokens.size()==1) exact.decode(result.committed_tokens[0],stream);
            else exact.continue_rows(result.committed_tokens,stream);
            result.executed_rows+=result.committed_tokens.size();
            result.replay_rows+=result.committed_tokens.size();
            ++result.native_invocations;
            ++result.native_batch_hist[result.committed_tokens.size()];
        }
        if(capture_taps) result.committed_taps=exact.exact_tap_rows_host(stream);
        // Consume only the final authoritative rows after repair, before
        // canonicalization overwrites row0 with the final row.
        if(capture_taps || tap_consumer)
            exl3_emit_committed_tap_segment(result,exact,tap_binding,tap_consumer,
                0,static_cast<int>(result.committed_tokens.size()),root.position(),0,
                root.position(),static_cast<int>(tentative.size()),root.position(),
                static_cast<int>(result.committed_tokens.size()),repair,false,stream);
        if(!repair || result.committed_tokens.size()>1) exact.finish_exact_continuation(stream);
        result.committed_state=exact.export_exact_host_state(stream);
    } catch(...) {
        const auto failure=std::current_exception();
        try{exact.restore_exact_host_state(root,stream);}catch(...){}
        std::rethrow_exception(failure);
    }
    return result;
}

// One authoritative L2 window. A context prepared with explicit native16
// executes up to sixteen rows per layer-stack pass; existing contexts
// retain up to eight. Neither case is a native64 kernel. Invocation
// widths and replayed rows remain explicit. No state or token becomes authoritative until the complete
// surviving window has been checked. On rejection/terminal, discard the
// teacher-forced suffix and replay only the accepted prefix plus correction.
inline Exl3OuterReferenceResult verify_exl3_outer_windowed_reference(
    Exl3TextContext& exact,const Exl3ExactHostState& root,
    std::span<const std::int64_t> tentative,bool capture_taps=false,
    std::span<const std::int64_t> terminal={},cudaStream_t stream=nullptr,
    bool reuse_resident_root=false,
    const Exl3CommittedTapBinding* tap_binding=nullptr,
    const Exl3CommittedTapConsumer* tap_consumer=nullptr) {
    if(tentative.size()<2 || tentative.size()>64 || exact.continuation_capacity()<8)
        throw std::invalid_argument("outer windowed reference requires prepared2..64rows");
    const std::size_t native_width = exact.continuation_capacity()==16 ? 16 : 8;
    for(auto token:tentative) if(token<0 || token>=248320)
        throw std::invalid_argument("outer windowed token extent");
    for(auto token:terminal) if(token<0 || token>=248320)
        throw std::invalid_argument("outer windowed terminal token extent");
    if(tap_consumer && (!tap_binding || !*tap_binding))
        throw std::invalid_argument("outer windowed tap consumer missing source binding");
    Exl3OuterReferenceResult result;
    result.committed_tap_segments.reserve(tentative.size());
    result.committed_tokens.reserve(tentative.size());
    if(reuse_resident_root)result.root_restores=exact.restore_exact_host_state_if_needed(root,stream)?1:0;
    else {
        exact.restore_exact_host_state(root,stream);result.root_restores=1;
    }
    bool repair=false;
    try {
        for(std::size_t offset=0;offset<tentative.size() && !repair && !result.stopped;) {
            const std::size_t rows=std::min<std::size_t>(native_width,tentative.size()-offset);
            std::array<std::int64_t,16> path_storage{};
            auto path=std::span<std::int64_t>(path_storage).first(rows);
            path[0]=exl3_branch_greedy(exact,stream);
            if(rows==1) exact.decode(path[0],stream);
            else {
                exact.continue_rows(tentative.subspan(offset,rows),stream);
                exl3_outer_greedy_path(exact,path[0],path,stream);
            }
            result.verification_rows+=rows;++result.native_invocations;
            ++result.native_batch_hist[rows];
            const auto decision=decide_exl3_outer_prefix(path,tentative.subspan(offset,rows),terminal);
            result.accepted+=decision.accepted;
            result.committed_tokens.insert(result.committed_tokens.end(),
                decision.committed_tokens.begin(),decision.committed_tokens.end());
            result.rejected=decision.rejected;result.stopped=decision.stopped;
            repair=decision.rejected || decision.committed_tokens.size()<rows;
            if(!repair) {
                if(capture_taps) {
                    const auto taps=exact.exact_tap_rows_host(stream);
                    for(int layer=0;layer<5;++layer)
                        result.committed_taps[layer].insert(result.committed_taps[layer].end(),
                            taps[layer].begin(),taps[layer].end());
                }
                if(capture_taps || tap_consumer)
                    exl3_emit_committed_tap_segment(result,exact,tap_binding,tap_consumer,
                        0,static_cast<int>(rows),root.position()+static_cast<int>(offset),
                        static_cast<int>(offset),root.position()+static_cast<int>(offset),
                        static_cast<int>(rows),root.position()+static_cast<int>(offset),
                        static_cast<int>(rows),false,false,stream);
                if(rows>1) exact.finish_exact_continuation(stream);
                offset+=rows;
            }
        }
        if(repair) {
            exact.restore_exact_host_state(root,stream);++result.root_restores;
            for(auto& taps:result.committed_taps) taps.clear();
            result.discarded_tentative_tap_segments=result.committed_tap_segments.size();
            result.committed_tap_segments.clear();
            for(std::size_t offset=0;offset<result.committed_tokens.size();) {
                const std::size_t rows=std::min<std::size_t>(native_width,result.committed_tokens.size()-offset);
                const auto span=std::span<const std::int64_t>(result.committed_tokens).subspan(offset,rows);
                if(rows==1) exact.decode(span[0],stream);
                else exact.continue_rows(span,stream);
                if(capture_taps) {
                    const auto taps=exact.exact_tap_rows_host(stream);
                    for(int layer=0;layer<5;++layer)
                        result.committed_taps[layer].insert(result.committed_taps[layer].end(),
                            taps[layer].begin(),taps[layer].end());
                }
                const auto attempt_offset=(offset/native_width)*native_width;
                const auto attempt_rows=std::min<std::size_t>(native_width,
                    tentative.size()-attempt_offset);
                if(capture_taps || tap_consumer)
                    exl3_emit_committed_tap_segment(result,exact,tap_binding,tap_consumer,
                        0,static_cast<int>(rows),root.position()+static_cast<int>(offset),
                        static_cast<int>(offset),root.position()+static_cast<int>(attempt_offset),
                        static_cast<int>(attempt_rows),root.position()+static_cast<int>(offset),
                        static_cast<int>(rows),true,false,stream);
                if(rows>1) exact.finish_exact_continuation(stream);
                result.replay_rows+=rows;++result.native_invocations;offset+=rows;
                ++result.native_batch_hist[rows];
            }
        }
        result.executed_rows=result.verification_rows+result.replay_rows;
        result.committed_state=exact.export_exact_host_state(stream);
    } catch(...) {
        const auto failure=std::current_exception();
        try{exact.restore_exact_host_state(root,stream);}catch(...){}
        std::rethrow_exception(failure);
    }
    return result;
}
} // namespace ninfer::exl3
