#include "exl3/ready_work_descriptor.h"

#include <algorithm>
#include <chrono>
#include <array>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

namespace {
int check(bool value,const char* message) {
    if(value)return 0;
    std::cerr<<message<<'\n';return 1;
}
}

int main() {
    using namespace ninfer::exl3;
    using Clock=Exl3ReadyWorkDescriptor::Clock;
    int failures=0;
    const auto model=std::make_shared<int>(7);
    const auto submitted=Clock::time_point(std::chrono::seconds(10));
    const Exl3ReadyWorkResources resources{
        .logical_host_bytes=4096,
        .request_local_device_bytes=0,
        .physical_lane_slots=1,
        .requires_preallocated_context=true};
    const Exl3ReadyExecutionSignature signature{
        .model_owner=model,
        .device=0,
        .stage=Exl3ReadyNumericalStage::HostPreparation,
        .arithmetic=Exl3ReadyArithmetic::OrdinaryFp16B8GreedyText,
        .modality=Exl3ReadyModality::Text};
    Exl3ReadyWorkDescriptor work(17,submitted,{11,12,13,14},991,resources,signature);

    const auto queued=work.assess({17,false,false,false,false});
    failures+=check(queued.logically_admitted && !queued.physically_executable &&
        queued.blocked==Exl3ReadyWorkBlock::BlockedDependency,
        "logical queue admission was confused with physical executability");
    const auto stale=work.assess({18,true,true,true,false});
    failures+=check(!stale.physically_executable &&
        stale.blocked==Exl3ReadyWorkBlock::StaleAcquisition,
        "stale acquisition generation executed immutable work");
    const auto no_lane=work.assess({17,true,false,true,false});
    failures+=check(no_lane.blocked==Exl3ReadyWorkBlock::MissingPhysicalLane,
        "missing physical lane was not distinguished from logical admission");
    const auto no_resource=work.assess({17,true,true,false,false});
    failures+=check(no_resource.blocked==Exl3ReadyWorkBlock::MissingResource,
        "missing required resource remained executable");
    const auto cancelled=work.assess({18,false,false,false,true});
    failures+=check(cancelled.blocked==Exl3ReadyWorkBlock::Cancelled,
        "queued cancellation did not dominate stale/dependency state");
    const auto ready=work.assess({17,true,true,true,false});
    failures+=check(ready.logically_admitted && ready.physically_executable &&
        ready.blocked==Exl3ReadyWorkBlock::None,
        "complete current physical snapshot did not make work executable");

    const auto observation=work.observation();
    failures+=check(observation.generation==17 && observation.input_tokens==4 &&
        observation.input_fingerprint==991 && observation.resources==resources &&
        work.input_tokens()[0]==11 && work.input_tokens()[3]==14 &&
        work.age(submitted-std::chrono::seconds(1)).count()==0 &&
        work.age(submitted+std::chrono::microseconds(25)).count()==25,
        "owned input, resource or age facts changed after publication");

    Exl3ReadyWorkDescriptor compatible(19,submitted,{31},123,resources,signature);
    auto other_signature=signature;
    other_signature.model_owner=std::make_shared<int>(7);
    Exl3ReadyWorkDescriptor wrong_model(20,submitted,{31},124,resources,other_signature);
    other_signature=signature;
    other_signature.stage=Exl3ReadyNumericalStage::PrivateDraftProjection;
    Exl3ReadyWorkDescriptor wrong_stage(21,submitted,{31},125,resources,other_signature);
    failures+=check(work.compatible_with(compatible) && !work.compatible_with(wrong_model) &&
        !work.compatible_with(wrong_stage),
        "ready-work compatibility ignored model ownership or numerical stage");

    Exl3ReadyWorkSnapshotAuthority authority;
    const Exl3ReadyWorkState executable{17,true,true,true,false};
    auto changed_queue=authority.snapshot(work.observation(),executable,0);
    const auto old_revision=changed_queue.revision;
    authority.revise();
    failures+=check(authority.revision()!=old_revision &&
        !authority.consume(changed_queue,work.observation(),executable),
        "queue revision did not invalidate an unconsumed snapshot");
    auto changed_policy=authority.snapshot(work.observation(),executable,0);
    authority.revise();
    failures+=check(!authority.consume(changed_policy,work.observation(),executable),
        "reservation policy revision did not invalidate an unconsumed snapshot");
    auto blocked_snapshot=authority.snapshot(work.observation(),
        Exl3ReadyWorkState{17,false,true,true,false},0);
    failures+=check(!blocked_snapshot.assessment.physically_executable &&
        !authority.consume(blocked_snapshot,work.observation(),executable),
        "blocked dependency assessment became executable during consumption");
    auto cancellation_race=authority.snapshot(work.observation(),executable,0);
    failures+=check(!authority.consume(cancellation_race,work.observation(),
        Exl3ReadyWorkState{17,true,true,true,true}),
        "simultaneous cancellation did not invalidate a ready snapshot");
    auto stale_assessment=authority.snapshot(work.observation(),executable,0);
    auto newer_generation=work.observation();
    ++newer_generation.generation;
    failures+=check(!authority.consume(stale_assessment,newer_generation,
        Exl3ReadyWorkState{newer_generation.generation,true,true,true,false}),
        "changed descriptor generation consumed a stale assessment");
    auto changed_resource=authority.snapshot(work.observation(),executable,0);
    auto changed_observation=work.observation();
    ++changed_observation.resources.logical_host_bytes;
    failures+=check(!authority.consume(changed_resource,changed_observation,executable),
        "changed resource requirement consumed an older snapshot");
    auto current=authority.snapshot(work.observation(),executable,0);
    const auto current_nonce=current.nonce;
    const auto grant=authority.consume(current,work.observation(),executable);
    failures+=check(grant && grant->nonce==current_nonce && grant->lane==0 &&
        grant->observation.generation==17 &&
        !authority.consume(current,work.observation(),executable),
        "current snapshot was not consumed exactly once");

    const std::array<Exl3ReadyLaneLocality,2> hard_gate{{
        {.lane=0,.physical_resources_feasible=true,.identity_compatible=false,
            .reusable_tokens=4096,.cost={1,1}},
        {.lane=1,.physical_resources_feasible=true,.identity_compatible=true,
            .reusable_tokens=0,.cost={}},
    }};
    failures+=check(exl3_select_ready_lane(hard_gate)==1,
        "warm incompatible lane overrode cold feasible lane");
    const std::array<Exl3ReadyLaneLocality,2> missing_estimates{{
        {.lane=0,.physical_resources_feasible=true,.identity_compatible=true,
            .reusable_tokens=128,.cost={50,25}},
        {.lane=1,.physical_resources_feasible=true,.identity_compatible=true,
            .reusable_tokens=256,.cost={}},
    }};
    failures+=check(exl3_select_ready_lane(missing_estimates)==1,
        "unknown locality estimate was treated as optimistic or infeasible");
    const std::array<Exl3ReadyLaneLocality,3> measured_locality{{
        {.lane=0,.physical_resources_feasible=true,.identity_compatible=true,
            .reusable_tokens=256,.cost={40,40}},
        {.lane=1,.physical_resources_feasible=true,.identity_compatible=true,
            .reusable_tokens=128,.cost={10,20}},
        {.lane=2,.physical_resources_feasible=false,.identity_compatible=true,
            .reusable_tokens=512,.cost={1,1}},
    }};
    failures+=check(exl3_select_ready_lane(measured_locality)==1,
        "complete optional locality costs ignored hard resources or lower observed cost");
    auto tied=missing_estimates;
    tied[0].reusable_tokens=256;
    failures+=check(exl3_select_ready_lane(tied)==0,
        "missing-cost fallback did not preserve stable lane order");

    const auto age_origin=Clock::time_point(std::chrono::seconds(100));
    const std::array<Exl3ReadyRequestCandidate,4> repeated_short{{
        {.queue_index=0,.submitted=age_origin+std::chrono::seconds(3),.physically_eligible=true},
        {.queue_index=1,.submitted=age_origin+std::chrono::seconds(4),.physically_eligible=true},
        {.queue_index=2,.submitted=age_origin,.physically_eligible=true},
        {.queue_index=3,.submitted=age_origin+std::chrono::seconds(5),.physically_eligible=true},
    }};
    failures+=check(exl3_select_fair_ready_request(repeated_short)==2,
        "repeated newer short arrivals overtook older eligible work");
    const std::array<Exl3ReadyRequestCandidate,3> blocked_head{{
        {.queue_index=0,.submitted=age_origin,.physically_eligible=false},
        {.queue_index=1,.submitted=age_origin+std::chrono::seconds(1),.physically_eligible=true},
        {.queue_index=2,.submitted=age_origin-std::chrono::seconds(1),
            .physically_eligible=true,.cancelled=true},
    }};
    failures+=check(exl3_select_fair_ready_request(blocked_head)==1,
        "resource-ineligible head or cancelled high-age work blocked a feasible peer");
    auto stable_age=blocked_head;
    stable_age[0].physically_eligible=true;
    stable_age[1].submitted=stable_age[0].submitted;
    failures+=check(exl3_select_fair_ready_request(stable_age)==0,
        "equal-age request choice lost stable queue ordering");

    const auto history_a=std::make_shared<int>(41);
    const auto history_b=std::make_shared<int>(42);
    const Exl3ReadyCopyDemand copy_a{history_a,4096,std::chrono::microseconds(50)};
    const Exl3ReadyCopyDemand copy_b{history_b,2048,std::chrono::microseconds(50)};
    const auto copy_now=age_origin+std::chrono::microseconds(20);
    const std::array<Exl3ReadyCopyCandidate,2> duplicate_history{{
        {{.queue_index=0,.submitted=age_origin,.physically_eligible=true},copy_a,true},
        {{.queue_index=1,.submitted=age_origin+std::chrono::microseconds(1),
            .physically_eligible=true},copy_b,false},
    }};
    failures+=check(exl3_select_copy_aware_ready_request(duplicate_history,copy_now)==1,
        "known duplicate history transfer was launched before an equivalent feasible ordering");
    auto unknown_copy=duplicate_history;
    unknown_copy[1].demand.transfer_bytes.reset();
    unknown_copy[1].demand.maximum_deferral={};
    failures+=check(exl3_select_copy_aware_ready_request(unknown_copy,copy_now)==0,
        "unknown transfer demand was treated as a free nonduplicate alternative");
    auto incompatible_copy=duplicate_history;
    incompatible_copy[1].request.physically_eligible=false;
    failures+=check(exl3_select_copy_aware_ready_request(incompatible_copy,copy_now)==0,
        "copy hint overrode hard physical compatibility or blocked sole progress");
    failures+=check(exl3_select_copy_aware_ready_request(duplicate_history,
        age_origin+copy_a.maximum_deferral)==0,
        "bounded copy deferral overrode oldest-request fairness at its deadline");
    const Exl3ReadyCopyDemand same_value_foreign{std::make_shared<int>(41),4096,
        std::chrono::microseconds(50)};
    failures+=check(copy_a.actionable() && !copy_a.same_history(same_value_foreign),
        "copy history identity was inferred from equal payload rather than strong ownership");
    Exl3ReadyWorkDescriptor hinted(22,submitted,{41,42},126,resources,signature,copy_a);
    auto changed_copy_ticket=authority.snapshot(hinted.observation(),
        Exl3ReadyWorkState{22,true,true,true,false},0);
    auto changed_copy=hinted.observation();changed_copy.copy_demand=copy_b;
    failures+=check(!authority.consume(changed_copy_ticket,changed_copy,
        Exl3ReadyWorkState{22,true,true,true,false}),
        "changed copy-demand identity consumed an older ready snapshot");

    bool invalid=false;
    try {
        Exl3ReadyWorkDescriptor missing(0,submitted,{1},1,resources,signature);
        (void)missing;
    } catch(const std::invalid_argument&) {invalid=true;}
    failures+=check(invalid,"zero ready-work generation was accepted");
    invalid=false;
    try {
        Exl3ReadyWorkDescriptor unowned_copy(23,submitted,{1},2,resources,signature,
            Exl3ReadyCopyDemand{{},1,std::chrono::microseconds(1)});
        (void)unowned_copy;
    } catch(const std::invalid_argument&) {invalid=true;}
    failures+=check(invalid,"positive unowned copy demand was accepted");

    // Deterministic scheduling traces use fixed time points and hard-coded
    // expected identities. They exercise the production selectors, but do not
    // derive the oracle from the selected candidate or from elapsed wall time.
    struct TraceEntry {
        std::size_t id=0;
        Clock::time_point submitted{};
        bool eligible=false,cancelled=false;
        std::shared_ptr<const void> owner;
    };
    const auto trace_select=[](std::span<const TraceEntry> entries) {
        std::vector<Exl3ReadyRequestCandidate> candidates;
        candidates.reserve(entries.size());
        for(const auto& entry:entries)candidates.push_back({
            .queue_index=entry.id,.submitted=entry.submitted,
            .physically_eligible=entry.eligible,.cancelled=entry.cancelled});
        return exl3_select_fair_ready_request(candidates);
    };
    const auto trace_origin=Clock::time_point(std::chrono::seconds(500));
    std::vector<TraceEntry> starvation;
    std::array<std::weak_ptr<const void>,5> starvation_owners;
    for(std::size_t id=0;id<starvation_owners.size();++id) {
        auto owner=std::make_shared<std::size_t>(id);
        starvation_owners[id]=owner;
        starvation.push_back({id,trace_origin+std::chrono::microseconds(id),id!=0,false,
            std::move(owner)});
    }
    constexpr std::array<std::size_t,5> expected_starvation_order{1,0,2,3,4};
    std::array<std::size_t,5> observed_starvation_order{};
    for(std::size_t step=0;step<expected_starvation_order.size();++step) {
        const auto selected=trace_select(starvation);
        failures+=check(selected.has_value(),"starvation trace found no feasible request");
        if(!selected)break;
        observed_starvation_order[step]=*selected;
        const auto position=std::find_if(starvation.begin(),starvation.end(),
            [&](const auto& entry){return entry.id==*selected;});
        failures+=check(position!=starvation.end(),"starvation trace selected unknown identity");
        if(position==starvation.end())break;
        auto active_owner=std::move(position->owner);
        starvation.erase(position);
        failures+=check(!starvation_owners[*selected].expired(),
            "selected trace owner retired before active ownership");
        active_owner.reset();
        failures+=check(starvation_owners[*selected].expired(),
            "completed trace owner survived its final active reference");
        if(step==0)for(auto& entry:starvation)if(entry.id==0)entry.eligible=true;
    }
    failures+=check(observed_starvation_order==expected_starvation_order,
        "newer arrivals starved a newly feasible older request");

    std::array<TraceEntry,3> admission_pressure{{
        {10,trace_origin,false,true,std::make_shared<int>(10)},
        {11,trace_origin+std::chrono::microseconds(1),false,false,std::make_shared<int>(11)},
        {12,trace_origin+std::chrono::microseconds(2),true,false,std::make_shared<int>(12)},
    }};
    failures+=check(trace_select(admission_pressure)==12,
        "cancelled or resource-blocked admission prevented feasible progress");
    admission_pressure[1].eligible=true;
    failures+=check(trace_select(admission_pressure)==11,
        "released admission pressure did not restore older feasible ownership");

    constexpr std::array<Exl3ReadyWorkBlock,5> expected_stage_blocks{
        Exl3ReadyWorkBlock::BlockedDependency,
        Exl3ReadyWorkBlock::MissingPhysicalLane,
        Exl3ReadyWorkBlock::MissingResource,
        Exl3ReadyWorkBlock::None,
        Exl3ReadyWorkBlock::Cancelled};
    const std::array<Exl3ReadyWorkState,5> staged_states{{
        {17,false,false,false,false},
        {17,true,false,false,false},
        {17,true,true,false,false},
        {17,true,true,true,false},
        {17,true,true,true,true},
    }};
    std::array<Exl3ReadyWorkBlock,5> observed_stage_blocks{};
    for(std::size_t index=0;index<staged_states.size();++index)
        observed_stage_blocks[index]=work.assess(staged_states[index]).blocked;
    failures+=check(observed_stage_blocks==expected_stage_blocks,
        "multi-stage readiness trace changed its independent expected transitions");

    // A fake batch trace groups only exact compatible signatures. The expected
    // groups are fixed below: target peers, private-draft stage, and reloaded
    // model ownership must remain three distinct reservation families.
    const auto batch_model=std::make_shared<int>(70);
    const auto reloaded_batch_model=std::make_shared<int>(71);
    auto target_signature=signature;target_signature.model_owner=batch_model;
    target_signature.stage=Exl3ReadyNumericalStage::TargetProjection;
    auto draft_signature=target_signature;
    draft_signature.stage=Exl3ReadyNumericalStage::PrivateDraftProjection;
    auto reloaded_signature=target_signature;
    reloaded_signature.model_owner=reloaded_batch_model;
    struct BatchTraceEntry {std::size_t id;Exl3ReadyExecutionSignature signature;};
    const std::array<BatchTraceEntry,4> batch_trace{{
        {20,target_signature},{21,target_signature},{22,draft_signature},{23,reloaded_signature}}};
    std::vector<std::vector<std::size_t>> observed_groups;
    std::vector<Exl3ReadyExecutionSignature> group_signatures;
    for(const auto& entry:batch_trace) {
        std::size_t group=0;
        while(group<group_signatures.size() &&
              !group_signatures[group].compatible_with(entry.signature))++group;
        if(group==group_signatures.size()) {
            group_signatures.push_back(entry.signature);observed_groups.emplace_back();
        }
        observed_groups[group].push_back(entry.id);
    }
    const std::vector<std::vector<std::size_t>> expected_groups{{20,21},{22},{23}};
    failures+=check(observed_groups==expected_groups,
        "split-batch trace merged a different stage or reloaded model owner");

    std::weak_ptr<const void> retired_model;
    {
        auto old_model=std::make_shared<int>(80);retired_model=old_model;
        auto old_signature=signature;old_signature.model_owner=old_model;
        Exl3ReadyWorkDescriptor old_work(30,trace_origin,{1,2},130,resources,old_signature);
        old_signature.model_owner.reset();old_model.reset();
        failures+=check(!retired_model.expired(),
            "reload trace lost old model while its ready descriptor remained owned");
        auto reload_observation=old_work.observation();
        reload_observation.signature.model_owner=std::make_shared<int>(81);
        Exl3ReadyWorkSnapshotAuthority reload_authority;
        auto old_ticket=reload_authority.snapshot(old_work.observation(),
            Exl3ReadyWorkState{30,true,true,true,false},0);
        failures+=check(!reload_authority.consume(old_ticket,reload_observation,
            Exl3ReadyWorkState{30,true,true,true,false}),
            "reload trace consumed a snapshot under different model ownership");
    }
    failures+=check(retired_model.expired(),
        "reload trace retained old model after its last descriptor/ticket owner retired");

    Exl3ReadyDecisionTrace decision_trace;
    for(std::size_t index=0;index<Exl3ReadyDecisionTrace::capacity+3;++index)
        decision_trace.append({
            .authority_revision=100+index,
            .generation=1+index,
            .input_fingerprint=1000+index,
            .queue_size=2,
            .eligible=1,
            .eligible_deferred=0,
            .physical_rejected=1,
            .lane=static_cast<std::uint8_t>(index%2),
            .affinity_enabled=true,
            .locality_costs_complete=false,
            .reason=Exl3ReadyDecisionReason::OldestFeasible});
    const auto bounded_trace=decision_trace.snapshot();
    failures+=check(bounded_trace.count==Exl3ReadyDecisionTrace::capacity &&
        bounded_trace.total==Exl3ReadyDecisionTrace::capacity+3 &&
        bounded_trace.records[0].generation==4 &&
        bounded_trace.records[bounded_trace.count-1].generation==
            Exl3ReadyDecisionTrace::capacity+3,
        "ready decision provenance did not retain the newest bounded records in order");
    for(std::size_t index=0;index<bounded_trace.count;++index)
        failures+=check(bounded_trace.records[index].authority_revision &&
            bounded_trace.records[index].input_fingerprint &&
            bounded_trace.records[index].count_conserved() &&
            !bounded_trace.records[index].locality_costs_complete,
            "ready decision provenance lost snapshot identity, missing cost, or count conservation");
    decision_trace.reset();
    const auto reset_trace=decision_trace.snapshot();
    failures+=check(!reset_trace.count && !reset_trace.total,
        "disabled/reset decision provenance retained stale records");

    if(!failures)std::cout<<"ok\n";
    return failures?1:0;
}
