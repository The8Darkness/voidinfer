#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ninfer::exl3 {

enum class Exl3ScenarioOperation : std::uint8_t {
    acquire,
    propose,
    verify,
    cancel,
    repair,
    claim_publication,
    commit_publication,
    final_use,
    retire
};

enum class Exl3ScenarioInvariant : std::uint16_t {
    none=0,
    accepted=1U<<0,
    refused=1U<<1,
    state_unchanged=1U<<2,
    owner_retained=1U<<3,
    prefix_repaired=1U<<4,
    exact_publication=1U<<5,
    final_use_required=1U<<6,
    retired=1U<<7
};

constexpr Exl3ScenarioInvariant operator|(Exl3ScenarioInvariant left,
                                           Exl3ScenarioInvariant right) noexcept {
    return static_cast<Exl3ScenarioInvariant>(
        static_cast<std::uint16_t>(left)|static_cast<std::uint16_t>(right));
}

constexpr bool exl3_scenario_has(Exl3ScenarioInvariant value,
                                 Exl3ScenarioInvariant required) noexcept {
    return (static_cast<std::uint16_t>(value)&static_cast<std::uint16_t>(required))==
           static_cast<std::uint16_t>(required);
}

struct Exl3StateMachineScenarioStep {
    Exl3ScenarioOperation operation=Exl3ScenarioOperation::acquire;
    std::uint64_t epoch=0;
    std::uint32_t proposed_rows=0;
    std::uint32_t accepted_rows=0;
    Exl3ScenarioInvariant expected=Exl3ScenarioInvariant::none;
};

struct Exl3StateMachineScenario {
    static constexpr std::size_t maximum_steps=10;
    std::string_view name;
    std::array<Exl3StateMachineScenarioStep,maximum_steps> steps{};
    std::size_t size=0;
};

inline constexpr std::array<Exl3StateMachineScenario,5> exl3_state_machine_scenarios{{
    {
        "stale_epoch",
        {{{Exl3ScenarioOperation::acquire,7,0,0,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::propose,6,8,0,
           Exl3ScenarioInvariant::refused|Exl3ScenarioInvariant::state_unchanged}}},
        2
    },
    {
        "cancel_before_claim",
        {{{Exl3ScenarioOperation::acquire,11,0,0,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::propose,11,8,0,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::owner_retained},
          {Exl3ScenarioOperation::verify,11,8,5,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::owner_retained},
          {Exl3ScenarioOperation::cancel,11,8,5,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::owner_retained},
          {Exl3ScenarioOperation::claim_publication,11,8,5,
           Exl3ScenarioInvariant::refused|Exl3ScenarioInvariant::state_unchanged},
          {Exl3ScenarioOperation::final_use,11,8,5,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::retire,11,8,0,Exl3ScenarioInvariant::retired}}},
        7
    },
    {
        "double_commit",
        {{{Exl3ScenarioOperation::acquire,13,0,0,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::propose,13,4,0,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::verify,13,4,4,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::claim_publication,13,4,4,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::commit_publication,13,4,4,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::exact_publication},
          {Exl3ScenarioOperation::commit_publication,13,4,4,
           Exl3ScenarioInvariant::refused|Exl3ScenarioInvariant::state_unchanged}}},
        6
    },
    {
        "partial_repair",
        {{{Exl3ScenarioOperation::acquire,17,0,0,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::propose,17,8,0,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::owner_retained},
          {Exl3ScenarioOperation::verify,17,8,3,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::owner_retained},
          {Exl3ScenarioOperation::repair,17,8,3,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::prefix_repaired},
          {Exl3ScenarioOperation::claim_publication,17,3,3,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::commit_publication,17,3,3,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::exact_publication}}},
        6
    },
    {
        "final_use_before_retirement",
        {{{Exl3ScenarioOperation::acquire,19,0,0,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::propose,19,8,0,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::owner_retained},
          {Exl3ScenarioOperation::verify,19,8,8,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::owner_retained},
          {Exl3ScenarioOperation::claim_publication,19,8,8,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::commit_publication,19,8,8,
           Exl3ScenarioInvariant::accepted|Exl3ScenarioInvariant::exact_publication},
          {Exl3ScenarioOperation::retire,19,8,8,
           Exl3ScenarioInvariant::refused|Exl3ScenarioInvariant::owner_retained|
               Exl3ScenarioInvariant::final_use_required},
          {Exl3ScenarioOperation::final_use,19,8,8,Exl3ScenarioInvariant::accepted},
          {Exl3ScenarioOperation::retire,19,8,8,Exl3ScenarioInvariant::retired}}},
        8
    }
}};

} // namespace ninfer::exl3
