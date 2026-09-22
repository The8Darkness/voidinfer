#pragma once

// Exact 4K real-caller differential plus request-reset/lifetime boundary for
// the retained-kernel attention-only graph. Prepared for the focused CUDA gate;
// this helper intentionally does no timing or promotion work.
void run_prefill_attention_chain_graph(
    ninfer::exl3::Exl3TextModel& target,
    ninfer::exl3::Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& fixture) {
    using ninfer::exl3::exl3_prefill_attention_chain_global_snapshot;
    require(fixture.size()>=4096,"prefill attention-chain fixture extent");
    const auto saved=env("NINFER_EXL3_PREFILL_ATTENTION_CHAIN_GRAPH");
    struct Result {
        std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> request;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> prefill_state;
        std::vector<float> prefill_logits;
        std::int64_t next_token=-1;
        std::array<std::uint64_t,5> draft_ring_digest{};
        std::vector<std::int64_t> first_draft_proposal;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> decoded_state;
    };
    std::array<std::unique_ptr<DeviceBuffer>,5> tap_stage;
    std::array<std::uint16_t*,5> tap_stage_pointers{};
    for(int tap=0;tap<5;++tap) {
        tap_stage[tap]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);
        tap_stage_pointers[tap]=static_cast<std::uint16_t*>(tap_stage[tap]->get());
    }
    const auto run_4k=[&](ninfer::exl3::Exl3TextContext& context) {
        const auto request=ninfer::exl3::Exl3VeriCacheRequest::initialize(
            context,std::span<const std::int64_t>(fixture.data(),4096),1024);
        require(request && context.position()==4096,
            "prefill attention-chain Engine 4K request extent");
        cuda_check(cudaDeviceSynchronize(),"prefill attention-chain completion");
        auto state=context.export_exact_host_state();
        auto logits=context.logits_host();
        require(!logits.empty() && std::none_of(logits.begin(),logits.end(),
                [](float value){return !std::isfinite(value);}),
            "prefill attention-chain finite final logits");
        const auto token=static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
        // Exercise the first actual DFlash2 consumer of prompt-time target taps.
        // Final target state/logits do not retain every chunk's tap payload, so
        // they cannot alone establish Engine proposal authority.
        request->restore_draft(draft,tap_stage_pointers);
        const auto ring_digest=draft.ring_digest();
        std::vector<std::int64_t> block(8,kMaskToken);
        block[0]=token;
        auto proposal=draft.propose_cached(block,4096,context.target_embedding(),
            context.target_lm_head_weights(),context.target_lm_head_metadata(),
            kMaskToken);
        require(proposal.size()==7,
            "prefill attention-chain first Engine draft proposal extent");
        context.decode(token);
        cuda_check(cudaDeviceSynchronize(),
            "prefill attention-chain authoritative next-token decode");
        return Result{request,std::move(state),std::move(logits),token,
            ring_digest,std::move(proposal),
            context.export_exact_host_state()};
    };

    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_CHAIN_GRAPH","0");
    auto control=target.create_context(true);
    control->prepare_continuation(8);
    const auto control_result=run_4k(*control);

    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_CHAIN_GRAPH","1");
    auto candidate=target.create_context(true);
    candidate->prepare_continuation(8);
    const auto before=exl3_prefill_attention_chain_global_snapshot();
    const auto first_result=run_4k(*candidate);
    const auto first=exl3_prefill_attention_chain_global_snapshot();
    require(control_result.prefill_logits==first_result.prefill_logits &&
            control_result.next_token==first_result.next_token &&
            control_result.request->same_taps(*first_result.request) &&
            control_result.draft_ring_digest==first_result.draft_ring_digest &&
            control_result.first_draft_proposal==first_result.first_draft_proposal &&
            control_result.prefill_state->same_payload(*first_result.prefill_state) &&
            control_result.decoded_state->same_payload(*first_result.decoded_state),
        "prefill attention-chain first-request tap/ring/proposal/logits/token/state authority");
    require(first.captures>before.captures &&
            first.capture_setups-before.capture_setups==
                first.captures-before.captures &&
            first.replays-before.replays==first.captures-before.captures &&
            first.binding_mismatches==before.binding_mismatches,
        "prefill attention-chain capture/setup/dispatch accounting");

    candidate->reset();
    const auto second_result=run_4k(*candidate);
    const auto second=exl3_prefill_attention_chain_global_snapshot();
    require(control_result.prefill_logits==second_result.prefill_logits &&
            control_result.next_token==second_result.next_token &&
            control_result.request->same_taps(*second_result.request) &&
            control_result.draft_ring_digest==second_result.draft_ring_digest &&
            control_result.first_draft_proposal==second_result.first_draft_proposal &&
            control_result.prefill_state->same_payload(*second_result.prefill_state) &&
            control_result.decoded_state->same_payload(*second_result.decoded_state),
        "prefill attention-chain replay-after-reset tap/ring/proposal/logits/token/state authority");
    require(second.captures==first.captures &&
            second.capture_setups==first.capture_setups &&
            second.replays-first.replays==first.captures-before.captures &&
            second.binding_mismatches==first.binding_mismatches &&
            second.quarantines==first.quarantines,
        "prefill attention-chain replay/reset/lifetime accounting");

    // Reproduce the complete fixed-128 Engine numerical schedule, not merely
    // the raw draft proposal: compact the request, attach it through the serving
    // coordinator, run the selected verifier (including staged-B8, conditional
    // B8 and repair options), and publish each authoritative window. The known
    // fixtures reach the requested output limit without a frontend stop/control
    // boundary, so min(8,remaining) is the Engine's exact useful-token policy.
    using Root=std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>;
    struct CompleteSchedule {
        std::vector<std::int64_t> output;
        std::vector<int> partitions;
        std::array<std::uint64_t,5> draft_ring_digest{};
        Root root;
        std::uint64_t proposal_calls=0,neural_rows=0,proposed_suffix_rows=0;
        std::uint64_t verification_rows=0,replay_rows=0,accepted_draft_rows=0;
    };
    ninfer::exl3::Exl3VeriCacheServingIdentity identity{
        "prefill-attention-chain-128","SC_6.00bpw_H6_V6",
        "pinned-tokens","exact-B8-L2","text"};
    ninfer::exl3::Exl3VeriCacheServingPrefixCache cache(
        {2,64,2ULL<<30,8ULL<<30},identity);
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    require(GlobalMemoryStatusEx(&memory),
        "prefill attention-chain 128-token memory policy");
    ninfer::exl3::Exl3VeriCacheServingCoordinator coordinator(cache,
        {1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
    // Engine seals physical residency authority and its coordinator metadata
    // backing before any request admission. Merely installing the lane callback
    // is insufficient: reserve_snapshot_metadata intentionally refuses an
    // unbound coordinator even for a current lease.
    coordinator.bind_physical_resources({},
        ninfer::exl3::Exl3ResourceInventory::unlimited());
    coordinator.reserve_metadata_startup();
    const auto conditional_value=env("NINFER_EXL3_ENGINE_CONDITIONAL_B8");
    require(conditional_value.empty() || conditional_value=="0" ||
            conditional_value=="1",
        "prefill attention-chain conditional-B8 option");
    const bool conditional_b8=conditional_value=="1";
    const auto run_complete_schedule=[&](const Result& source) {
        auto compact=source.request->compact_draft(
            draft,tap_stage_pointers);
        auto lane=std::make_unique<ninfer::exl3::Exl3Dflash2Execution>(
            target.create_context(true),draft,identity.contract());
        // Engine installs this lane-scoped reservation at startup. The focused
        // lane must reproduce that authority before using verify_owned; an
        // ordinary raw context intentionally has no request metadata credit.
        lane->context().set_request_metadata_reservation(
            [&](std::uint64_t bytes) {
                if(!lane->execution_active())
                    throw std::logic_error(
                        "prefill attention-chain request metadata lane inactive");
                const auto current=lane->lease();
                return coordinator.reserve_snapshot_metadata(
                    &current,{},bytes);
            });
        coordinator.admit(compact);
        const auto lease=coordinator.acquire();
        require(lease.has_value() && lane->acquire(*lease),
            "prefill attention-chain 128-token acquire");
        CompleteSchedule result;
        constexpr std::size_t useful_tokens=128;
        while(result.output.size()<useful_tokens) {
            const auto remaining=useful_tokens-result.output.size();
            auto proposed=lane->propose(
                static_cast<int>(std::min<std::size_t>(8,remaining)));
            auto pending=lane->verify_owned(proposed);
            if(conditional_b8 && remaining>=16 &&
               pending->proposed_rows==8 && pending->costs.neural_rows==8 &&
               pending->verification.accepted==8 &&
               !pending->verification.rejected &&
               !pending->verification.stopped &&
               pending->verification.committed_tokens.size()==8) {
                const auto second=lane->try_propose_conditional_second(
                    *pending,16,coordinator);
                if(second)
                    pending=lane->verify_conditional_second_owned(
                        *pending,*second);
            }
            require(!pending->verification.committed_tokens.empty() &&
                    pending->verification.committed_tokens.size()<=remaining,
                "prefill attention-chain useful publication extent");
            result.verification_rows+=pending->verification.verification_rows;
            result.replay_rows+=pending->verification.replay_rows;
            result.accepted_draft_rows+=pending->accepted_draft_rows();
            auto publication=lane->publish(coordinator,*pending);
            require(publication.tokens==pending->verification.committed_tokens,
                "prefill attention-chain authoritative publication tokens");
            result.partitions.push_back(
                static_cast<int>(publication.tokens.size()));
            result.output.insert(result.output.end(),publication.tokens.begin(),
                publication.tokens.end());
            pending.reset();
        }
        require(result.output.size()==useful_tokens,
            "prefill attention-chain complete useful output count");
        result.root=lane->lease().root;
        result.draft_ring_digest=draft.ring_digest();
        const auto stats=lane->stats();
        result.proposal_calls=stats.proposal_calls;
        result.neural_rows=stats.neural_input_rows;
        result.proposed_suffix_rows=stats.proposed_rows;
        const auto completed=lane->lease();
        lane->release();
        coordinator.complete(completed);
        return result;
    };
    const auto schedule_graph_before=
        exl3_prefill_attention_chain_global_snapshot();
    const auto control_schedule=run_complete_schedule(control_result);
    const auto first_schedule=run_complete_schedule(first_result);
    const auto second_schedule=run_complete_schedule(second_result);
    const auto schedule_graph_after=
        exl3_prefill_attention_chain_global_snapshot();
    const auto same_schedule=[](const CompleteSchedule& expected,
                                const CompleteSchedule& actual) {
        return expected.output==actual.output &&
            expected.partitions==actual.partitions &&
            expected.proposal_calls==actual.proposal_calls &&
            expected.neural_rows==actual.neural_rows &&
            expected.proposed_suffix_rows==actual.proposed_suffix_rows &&
            expected.accepted_draft_rows==actual.accepted_draft_rows &&
            expected.verification_rows==actual.verification_rows &&
            expected.replay_rows==actual.replay_rows &&
            expected.draft_ring_digest==actual.draft_ring_digest &&
            expected.root->same_projected_conditioning_for_test(*actual.root) &&
            expected.root->state()->same_payload(*actual.root->state());
    };
    require(same_schedule(control_schedule,first_schedule),
        "prefill attention-chain first-capture complete Engine schedule");
    require(same_schedule(control_schedule,second_schedule),
        "prefill attention-chain reset-replay complete Engine schedule");
    require(schedule_graph_after.capture_setups==schedule_graph_before.capture_setups &&
            schedule_graph_after.captures==schedule_graph_before.captures &&
            schedule_graph_after.replays==schedule_graph_before.replays &&
            schedule_graph_after.eager_fallbacks==schedule_graph_before.eager_fallbacks &&
            schedule_graph_after.binding_mismatches==schedule_graph_before.binding_mismatches &&
            schedule_graph_after.nested_captures==schedule_graph_before.nested_captures &&
            schedule_graph_after.quarantines==schedule_graph_before.quarantines,
        "prefill attention-chain decode schedule changed prefill graph counters");
    coordinator.close();
    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_CHAIN_GRAPH",saved.c_str());
    std::cout<<"PREFILL_ATTENTION_CHAIN_GRAPH PASS captures="
             <<first.captures-before.captures<<" reset_replays="
             <<second.replays-first.replays
             <<" exact_4k_taps_ring_first_proposal_logits_next_token_state=1"
             <<" exact_128_engine_schedule=1 verifier_rows="
             <<control_schedule.verification_rows<<" replay_rows="
             <<control_schedule.replay_rows<<" neural_rows="
             <<control_schedule.neural_rows<<" lifecycle=1\n";
}
