#pragma once

#include "exl3/draft_distribution_contract.h"
#include "exl3/greedy_packet.h"
#include "exl3/request_sampling_state.h"
#include "exl3/sampling_policy.h"
#include "exl3/sampling_workspace_reservation.h"

#include <cstdint>
#include <cstddef>
#include <string_view>

namespace ninfer::exl3 {

enum class Exl3SampledApiExposure : std::uint8_t {
    public_greedy_only,
    unqualified_source_api
};

enum class Exl3SampledAdmissionDisposition : std::uint8_t {
    greedy,
    draft_distribution_unavailable,
    public_sampled_disabled,
    request_state_unavailable,
    target_distribution_unavailable,
    draft_distribution_invalid,
    target_distribution_invalid,
    workspace_unavailable,
    unqualified_source_ready
};

struct Exl3SampledAdmissionRequest {
    Exl3NormalizedSamplingPolicy policy;
    Exl3DraftDistributionAvailability draft_availability;
    Exl3SampledApiExposure exposure=Exl3SampledApiExposure::public_greedy_only;
    const Exl3RequestSamplingState* request_state=nullptr;
    const Exl3DraftDistributionAuthority* draft_q=nullptr;
    const Exl3SampledDistributionAuthority* target_p=nullptr;
    const Exl3SamplingWorkspaceAuthority* workspace=nullptr;
    const Exl3SamplingWorkspacePlan* workspace_plan=nullptr;
    std::uint32_t rows=0,vocabulary=0;
    std::uint64_t request_generation=0,root_revision=0,target_serial=0,acquisition=0;
    int first_position=0;
};

struct Exl3SampledAdmissionDecision {
    Exl3SampledAdmissionDisposition disposition=
        Exl3SampledAdmissionDisposition::public_sampled_disabled;
    [[nodiscard]] bool greedy() const noexcept {
        return disposition==Exl3SampledAdmissionDisposition::greedy;
    }
    [[nodiscard]] bool unqualified_source_ready() const noexcept {
        return disposition==Exl3SampledAdmissionDisposition::unqualified_source_ready;
    }
    [[nodiscard]] bool publicly_enabled() const noexcept{return greedy();}
};

class Exl3SampledAuthorityBoundary {
public:
    static Exl3SampledAdmissionDecision decide(
        const Exl3SampledAdmissionRequest& request) noexcept {
        if(!request.policy.requires_sampled_distribution())
            return {Exl3SampledAdmissionDisposition::greedy};
        if(!request.draft_availability.supports(
                Exl3SampledAcceptanceMethod::independent_rejection_with_residual))
            return {Exl3SampledAdmissionDisposition::draft_distribution_unavailable};
        if(request.exposure!=Exl3SampledApiExposure::unqualified_source_api)
            return {Exl3SampledAdmissionDisposition::public_sampled_disabled};
        if(!request.request_state || request.request_state->cancelled() ||
           request.request_state->request_generation()!=request.request_generation ||
           request.request_state->committed_prefix()!=
                static_cast<std::size_t>(request.first_position))
            return {Exl3SampledAdmissionDisposition::request_state_unavailable};
        if(!request.draft_q)
            return {Exl3SampledAdmissionDisposition::draft_distribution_invalid};
        if(!request.target_p)
            return {Exl3SampledAdmissionDisposition::target_distribution_unavailable};
        try {
            request.draft_q->require(
                Exl3SampledAcceptanceMethod::independent_rejection_with_residual,
                request.rows,request.vocabulary,request.request_generation,
                request.root_revision,request.first_position);
        } catch(...) {
            return {Exl3SampledAdmissionDisposition::draft_distribution_invalid};
        }
        try {
            Exl3SelectionAuthorityRequirement::sampled(
                request.rows,request.request_generation,request.first_position,
                request.target_serial,request.vocabulary).validate(*request.target_p);
        } catch(...) {
            return {Exl3SampledAdmissionDisposition::target_distribution_invalid};
        }
        if(!request.workspace || !request.workspace_plan ||
           !request.workspace->matches(*request.workspace_plan,
                request.request_generation,request.root_revision,request.acquisition))
            return {Exl3SampledAdmissionDisposition::workspace_unavailable};
        return {Exl3SampledAdmissionDisposition::unqualified_source_ready};
    }

    static std::string_view message(Exl3SampledAdmissionDisposition value) noexcept {
        switch(value) {
        case Exl3SampledAdmissionDisposition::draft_distribution_unavailable:
            return "EXL3 DFlash2 exposes greedy tokens and internal parallel-mask selector scores, not complete autoregressive q";
        case Exl3SampledAdmissionDisposition::public_sampled_disabled:
            return "EXL3 public sampled speculative execution is disabled";
        case Exl3SampledAdmissionDisposition::request_state_unavailable:
            return "EXL3 sampled request state is missing, cancelled or stale";
        case Exl3SampledAdmissionDisposition::target_distribution_unavailable:
            return "EXL3 target sampled distribution is unavailable";
        case Exl3SampledAdmissionDisposition::draft_distribution_invalid:
            return "EXL3 draft sampled distribution authority is invalid";
        case Exl3SampledAdmissionDisposition::target_distribution_invalid:
            return "EXL3 target sampled distribution authority is invalid";
        case Exl3SampledAdmissionDisposition::workspace_unavailable:
            return "EXL3 request-local sampled workspace is unavailable";
        case Exl3SampledAdmissionDisposition::unqualified_source_ready:
            return "EXL3 sampled authority is source-ready but unqualified";
        case Exl3SampledAdmissionDisposition::greedy:
            return "EXL3 greedy authority";
        }
        return "EXL3 sampled authority disposition invalid";
    }
};

} // namespace ninfer::exl3
