#include "exl3/sampling_workspace_reservation.h"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
struct Storage {std::array<std::byte,16> marker{};};
}

int main() {
    ninfer::runtime::ResolvedSamplingParameters values;
    auto policy=Exl3SamplingPolicy::normalize(values,248320);
    const auto current=Exl3DraftDistributionAvailability::current_dflash2();
    const auto greedy=Exl3SamplingWorkspacePlan::derive(policy,current,8,4096);
    need(!greedy.sampled && !greedy.device_bytes &&
            !greedy.requirement().units[static_cast<unsigned>(
                Exl3ResourceInventory::Domain::device)],
        "greedy request reserved probability workspace");

    values.temperature=.8F;policy=Exl3SamplingPolicy::normalize(values,248320);
    need(refuses([&]{(void)Exl3SamplingWorkspacePlan::derive(policy,current,8,4096);}),
        "sampled workspace planned without a draft distribution contract");
    auto capable=current;capable.normalized_q_available=true;
    capable.complete_vocabulary_support=true;
    capable.conditioning=Exl3DraftDistributionConditioning::autoregressive_predecessor;
    const auto plan=Exl3SamplingWorkspacePlan::derive(policy,capable,8,4096);
    need(plan.target_probability_bytes==8ULL*248320*4 &&
            plan.draft_probability_bytes==plan.target_probability_bytes &&
            plan.residual_bytes==plan.target_probability_bytes &&
            !plan.occurrence_count_bytes && plan.device_bytes==
                3*plan.target_probability_bytes+4096,
        "large-vocabulary probability workspace accounting changed");

    auto storage_a=std::make_shared<Storage>(),storage_b=std::make_shared<Storage>();
    Exl3ResourceInventory::Allocation allocation_a{storage_a,0,
        Exl3ResourceInventory::Domain::device,plan.device_bytes};
    Exl3ResourceInventory::Allocation allocation_b{storage_b,0,
        Exl3ResourceInventory::Domain::device,plan.device_bytes};
    auto first=Exl3SamplingWorkspaceAuthority::from_reserved_allocation(
        plan,allocation_a,storage_a.get(),11,13,17);
    auto second=Exl3SamplingWorkspaceAuthority::from_reserved_allocation(
        plan,allocation_b,storage_b.get(),12,19,23);
    need(first.matches(plan,11,13,17) && second.matches(plan,12,19,23) &&
            !first.matches(plan,12,19,23) && first.owner()!=second.owner(),
        "concurrent request workspaces aliased or ignored lineage");

    std::weak_ptr<const void> retained=storage_a;storage_a.reset();
    need(!retained.expired(),"aborted request lost retained workspace owner");
    auto wrong=allocation_b;wrong.units--;
    need(refuses([&]{(void)Exl3SamplingWorkspaceAuthority::from_reserved_allocation(
            plan,wrong,storage_b.get(),12,19,23);}),
        "under-reserved probability workspace admitted");

    values.presence_penalty=.5F;policy=Exl3SamplingPolicy::normalize(values,248320);
    const auto counted=Exl3SamplingWorkspacePlan::derive(policy,capable,8,0);
    need(counted.occurrence_count_bytes==248320ULL*4 &&
            counted.device_bytes==3*counted.target_probability_bytes+
                counted.occurrence_count_bytes,
        "penalty occurrence-count workspace was not independently budgeted");
    std::cout<<"exl3_sampling_workspace_reservation PASS\n";
}
