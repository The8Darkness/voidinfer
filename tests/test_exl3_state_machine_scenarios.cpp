#include "exl3/state_machine_scenarios.h"

#include <iostream>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
const Exl3StateMachineScenario& named(std::string_view name) {
    for(const auto& scenario:exl3_state_machine_scenarios)
        if(scenario.name==name)return scenario;
    throw std::runtime_error("prepared state-machine scenario is missing");
}
}

int main() {
    for(const auto& scenario:exl3_state_machine_scenarios) {
        need(!scenario.name.empty() && scenario.size>0 &&
                scenario.size<=Exl3StateMachineScenario::maximum_steps,
            "prepared scenario extent is invalid");
        for(std::size_t index=0;index<scenario.size;++index)
            need(scenario.steps[index].epoch!=0 &&
                    scenario.steps[index].expected!=Exl3ScenarioInvariant::none,
                "prepared scenario omitted epoch or expected invariant");
    }
    const auto& stale=named("stale_epoch");
    need(stale.steps[1].epoch!=stale.steps[0].epoch &&
            exl3_scenario_has(stale.steps[1].expected,
                Exl3ScenarioInvariant::state_unchanged),
        "stale-epoch fixture does not preserve authoritative state");
    const auto& cancelled=named("cancel_before_claim");
    need(cancelled.steps[4].operation==Exl3ScenarioOperation::claim_publication &&
            exl3_scenario_has(cancelled.steps[4].expected,
                Exl3ScenarioInvariant::refused),
        "unclaimed cancellation fixture permits publication");
    const auto& duplicate=named("double_commit");
    need(duplicate.steps[4].operation==Exl3ScenarioOperation::commit_publication &&
            duplicate.steps[5].operation==Exl3ScenarioOperation::commit_publication &&
            exl3_scenario_has(duplicate.steps[5].expected,
                Exl3ScenarioInvariant::state_unchanged),
        "double-commit fixture lacks idempotence refusal");
    const auto& repair=named("partial_repair");
    need(repair.steps[3].proposed_rows==8 && repair.steps[3].accepted_rows==3 &&
            repair.steps[4].proposed_rows==3 &&
            exl3_scenario_has(repair.steps[3].expected,
                Exl3ScenarioInvariant::prefix_repaired),
        "partial-repair fixture publishes unaccepted rows");
    const auto& final_use=named("final_use_before_retirement");
    need(exl3_scenario_has(final_use.steps[5].expected,
                Exl3ScenarioInvariant::final_use_required) &&
            final_use.steps[6].operation==Exl3ScenarioOperation::final_use &&
            final_use.steps[7].operation==Exl3ScenarioOperation::retire &&
            exl3_scenario_has(final_use.steps[7].expected,
                Exl3ScenarioInvariant::retired),
        "final-use fixture releases ownership out of order");
    std::cout<<"exl3_state_machine_scenarios PASS\n";
}
