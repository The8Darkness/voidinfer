#include "exl3/vision_qualification_plan.h"

#include <iostream>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}}

int main() {
    Exl3V6QualificationPlan plan;
    need(plan.receipt(Exl3V6QualificationCategory::numerical).state==
            Exl3V6QualificationState::failed && !plan.qualified(),
        "known V6 numerical failure was promoted by prepared source");

    auto& numerical=plan.receipt(Exl3V6QualificationCategory::numerical);
    numerical.state=Exl3V6QualificationState::passed;
    numerical.identity="independent-boundary-receipt";
    need(!plan.qualified(),"encoder plausibility certified all media categories");
    auto& state=plan.receipt(Exl3V6QualificationCategory::state_and_ring);
    state.state=Exl3V6QualificationState::passed;
    state.identity="one-matching-token-stream";
    need(!plan.qualified(),"one matching token stream certified semantics/lifecycle/client");

    plan.receipt(Exl3V6QualificationCategory::heldout_semantic).state=
        Exl3V6QualificationState::passed;
    plan.receipt(Exl3V6QualificationCategory::heldout_semantic).identity=
        "independent-heldout-review";
    plan.receipt(Exl3V6QualificationCategory::lifecycle).state=
        Exl3V6QualificationState::passed;
    plan.receipt(Exl3V6QualificationCategory::lifecycle).identity=
        "lifecycle-pressure-cancellation";
    plan.receipt(Exl3V6QualificationCategory::client).state=
        Exl3V6QualificationState::passed;
    plan.receipt(Exl3V6QualificationCategory::client).identity=
        "loopback-sse-reload";
    need(plan.qualified(),"five independent synthetic future receipts did not compose");

    plan.receipt(Exl3V6QualificationCategory::client).identity=
        "one-matching-token-stream";
    bool refused=false;try{(void)plan.qualified();}catch(const std::invalid_argument&){refused=true;}
    need(refused,"one receipt substituted for two qualification categories");
    std::cout<<"v6_qualification_plan PREPARED_NOT_RUN\n";
}
