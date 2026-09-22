#include "exl3/sampled_authority_boundary.h"

#include <array>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
struct Distributions {std::array<float,6> q{.5F,.3F,.2F,.2F,.3F,.5F},p=q;};
}

int main() {
    ninfer::runtime::ResolvedSamplingParameters values;
    auto greedy=Exl3SamplingPolicy::normalize(values,3);
    auto request=Exl3SampledAdmissionRequest{
        greedy,Exl3DraftDistributionAvailability::current_dflash2()};
    auto decision=Exl3SampledAuthorityBoundary::decide(request);
    need(decision.greedy() && decision.publicly_enabled(),
        "default greedy request crossed sampled boundary");

    values.temperature=.7F;request.policy=Exl3SamplingPolicy::normalize(values,3);
    decision=Exl3SampledAuthorityBoundary::decide(request);
    need(decision.disposition==
            Exl3SampledAdmissionDisposition::draft_distribution_unavailable &&
            !decision.publicly_enabled(),
        "missing DFlash2 q was not the sampled refusal");

    request.draft_availability.normalized_q_available=true;
    request.draft_availability.complete_vocabulary_support=true;
    request.draft_availability.conditioning=
        Exl3DraftDistributionConditioning::autoregressive_predecessor;
    decision=Exl3SampledAuthorityBoundary::decide(request);
    need(decision.disposition==
            Exl3SampledAdmissionDisposition::public_sampled_disabled,
        "capable descriptor bypassed public greedy-only scope");

    auto root=std::make_shared<const int>(1);
    Exl3RequestSamplingState state(2,11,17,23);
    state.bind_acquisition(29,root,31);
    auto owner=std::make_shared<Distributions>();
    Exl3DraftDistributionAuthority q;
    q.method=Exl3SampledAcceptanceMethod::independent_rejection_with_residual;
    q.conditioning=Exl3DraftDistributionConditioning::autoregressive_predecessor;
    q.owner=owner;q.probabilities=owner->q.data();q.rows=2;q.vocabulary=3;
    q.row_stride=3;q.request_generation=11;q.root_revision=37;q.first_position=31;
    q.normalized=true;q.complete_vocabulary_support=true;q.ready=true;
    Exl3SampledDistributionAuthority p;
    p.owner=owner;p.probabilities=owner->p.data();p.rows=2;p.vocabulary=3;
    p.row_stride=3;p.generation=11;p.serial=41;p.position=31;p.ready=true;
    request.exposure=Exl3SampledApiExposure::unqualified_source_api;
    request.request_state=&state;request.draft_q=&q;request.target_p=&p;
    request.rows=2;request.vocabulary=3;request.request_generation=11;
    request.root_revision=37;request.target_serial=41;request.first_position=31;
    request.acquisition=29;
    const auto workspace_plan=Exl3SamplingWorkspacePlan::derive(
        request.policy,request.draft_availability,2,64);
    Exl3ResourceInventory::Allocation workspace_allocation{owner,1,
        Exl3ResourceInventory::Domain::device,workspace_plan.device_bytes};
    auto workspace=Exl3SamplingWorkspaceAuthority::from_reserved_allocation(
        workspace_plan,workspace_allocation,owner.get(),11,37,29);
    request.workspace_plan=&workspace_plan;request.workspace=&workspace;
    decision=Exl3SampledAuthorityBoundary::decide(request);
    need(decision.unqualified_source_ready() && !decision.publicly_enabled(),
        "complete source authority was promoted to public support");

    auto stale=q;stale.root_revision=38;request.draft_q=&stale;
    need(Exl3SampledAuthorityBoundary::decide(request).disposition==
            Exl3SampledAdmissionDisposition::draft_distribution_invalid,
        "stale draft authority admitted");
    request.draft_q=&q;request.target_p=nullptr;
    need(Exl3SampledAuthorityBoundary::decide(request).disposition==
            Exl3SampledAdmissionDisposition::target_distribution_unavailable,
        "missing target distribution admitted");
    auto invalid=values;invalid.temperature=std::numeric_limits<float>::quiet_NaN();
    need(refuses([&]{(void)Exl3SamplingPolicy::normalize(invalid,3);}),
        "invalid policy reached sampled authority boundary");
    std::cout<<"exl3_sampled_authority_boundary PASS\n";
}
