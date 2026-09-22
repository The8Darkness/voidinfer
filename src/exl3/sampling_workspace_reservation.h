#pragma once

#include "exl3/draft_distribution_contract.h"
#include "exl3/resource_inventory.h"
#include "exl3/sampling_policy.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {

struct Exl3SamplingWorkspacePlan {
    bool sampled=false,occurrence_counts=false;
    std::uint32_t rows=0,vocabulary=0;
    std::uint64_t target_probability_bytes=0,draft_probability_bytes=0;
    std::uint64_t residual_bytes=0,occurrence_count_bytes=0;
    std::uint64_t sampler_transient_bytes=0,device_bytes=0;
    std::uint64_t configuration=0;

    [[nodiscard]] static Exl3SamplingWorkspacePlan derive(
        const Exl3NormalizedSamplingPolicy& policy,
        const Exl3DraftDistributionAvailability& availability,
        std::uint32_t rows,std::uint64_t sampler_transient_bytes) {
        if(!policy.requires_sampled_distribution())return {};
        if(!availability.supports(
                Exl3SampledAcceptanceMethod::independent_rejection_with_residual))
            throw std::invalid_argument("sampling workspace requires complete draft q");
        if(!rows || !policy.vocabulary)
            throw std::invalid_argument("sampling workspace shape");
        const auto elements=checked_multiply(rows,policy.vocabulary);
        Exl3SamplingWorkspacePlan result;
        result.sampled=true;result.occurrence_counts=policy.occurrence_counts_required;
        result.rows=rows;result.vocabulary=policy.vocabulary;
        result.target_probability_bytes=checked_multiply(elements,sizeof(float));
        result.draft_probability_bytes=result.target_probability_bytes;
        result.residual_bytes=result.target_probability_bytes;
        result.occurrence_count_bytes=result.occurrence_counts?
            checked_multiply(policy.vocabulary,sizeof(std::int32_t)):0;
        result.sampler_transient_bytes=sampler_transient_bytes;
        result.device_bytes=checked_add(checked_add(
            checked_add(result.target_probability_bytes,result.draft_probability_bytes),
            result.residual_bytes),checked_add(result.occurrence_count_bytes,
            result.sampler_transient_bytes));
        result.configuration=0x53414D50574B0000ULL^
            (static_cast<std::uint64_t>(rows)<<32)^policy.vocabulary^
            (result.occurrence_counts?0x80000000ULL:0);
        return result;
    }

    [[nodiscard]] Exl3ResourceInventory::Requirement requirement() const {
        Exl3ResourceInventory::Requirement result;result.configuration=configuration;
        if(sampled)result.add(Exl3ResourceInventory::Domain::device,1,device_bytes);
        return result;
    }

    [[nodiscard]] bool same_shape(const Exl3SamplingWorkspacePlan& other) const noexcept {
        return sampled==other.sampled && occurrence_counts==other.occurrence_counts &&
            rows==other.rows && vocabulary==other.vocabulary &&
            target_probability_bytes==other.target_probability_bytes &&
            draft_probability_bytes==other.draft_probability_bytes &&
            residual_bytes==other.residual_bytes &&
            occurrence_count_bytes==other.occurrence_count_bytes &&
            sampler_transient_bytes==other.sampler_transient_bytes &&
            device_bytes==other.device_bytes && configuration==other.configuration;
    }

private:
    static std::uint64_t checked_multiply(std::uint64_t left,std::uint64_t right) {
        if(right && left>std::numeric_limits<std::uint64_t>::max()/right)
            throw std::overflow_error("sampling workspace extent overflow");
        return left*right;
    }
    static std::uint64_t checked_add(std::uint64_t left,std::uint64_t right) {
        if(right>std::numeric_limits<std::uint64_t>::max()-left)
            throw std::overflow_error("sampling workspace extent overflow");
        return left+right;
    }
};

class Exl3SamplingWorkspaceAuthority {
public:
    static Exl3SamplingWorkspaceAuthority from_reserved_allocation(
        Exl3SamplingWorkspacePlan plan,
        const Exl3ResourceInventory::Allocation& allocation,void* data,
        std::uint64_t request_generation,std::uint64_t root_revision,
        std::uint64_t acquisition) {
        if(!plan.sampled || !plan.device_bytes || !data || !allocation.owner ||
           allocation.domain!=Exl3ResourceInventory::Domain::device ||
           allocation.units!=plan.device_bytes || !request_generation ||
           !root_revision || !acquisition)
            throw std::invalid_argument("sampling workspace reservation binding");
        return {std::move(plan),allocation.owner,data,request_generation,
            root_revision,acquisition};
    }

    Exl3SamplingWorkspaceAuthority(const Exl3SamplingWorkspaceAuthority&)=delete;
    Exl3SamplingWorkspaceAuthority& operator=(const Exl3SamplingWorkspaceAuthority&)=delete;
    Exl3SamplingWorkspaceAuthority(Exl3SamplingWorkspaceAuthority&&)=default;
    Exl3SamplingWorkspaceAuthority& operator=(Exl3SamplingWorkspaceAuthority&&)=default;

    [[nodiscard]] bool matches(const Exl3SamplingWorkspacePlan& expected,
        std::uint64_t request_generation,std::uint64_t root_revision,
        std::uint64_t acquisition) const noexcept {
        return owner_ && data_ && plan_.same_shape(expected) &&
            request_generation_==request_generation && root_revision_==root_revision &&
            acquisition_==acquisition;
    }
    [[nodiscard]] const std::shared_ptr<const void>& owner() const noexcept{return owner_;}
    [[nodiscard]] const Exl3SamplingWorkspacePlan& plan() const noexcept{return plan_;}

private:
    Exl3SamplingWorkspaceAuthority(Exl3SamplingWorkspacePlan plan,
        std::shared_ptr<const void> owner,void* data,std::uint64_t request_generation,
        std::uint64_t root_revision,std::uint64_t acquisition)
        :plan_(std::move(plan)),owner_(std::move(owner)),data_(data),
         request_generation_(request_generation),root_revision_(root_revision),
         acquisition_(acquisition) {}

    Exl3SamplingWorkspacePlan plan_;
    std::shared_ptr<const void> owner_;
    void* data_=nullptr;
    std::uint64_t request_generation_=0,root_revision_=0,acquisition_=0;
};

} // namespace ninfer::exl3
