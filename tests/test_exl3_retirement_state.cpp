#include "exl3/retirement_state.h"
#include <stdexcept>
#include <iostream>
using namespace ninfer::exl3;
void need(bool value){if(!value)throw std::runtime_error("retirement transition");}
int main(){try{
    Exl3RetirementState good;need(good.begin(false));need(!good.result().reusable());
    good.complete(0);need(good.result().reusable());need(!good.begin(false));
    good.complete(17);need(good.result().first_drain_error==0);
    good.observe_cleanup_failure(23);
    good.observe_cleanup_failure(29);good.complete(0);
    need(!good.result().reusable() && good.result().first_cleanup_error==23 &&
        good.result().first_drain_error==0 && !good.begin(false));
    Exl3RetirementState bad;need(bad.begin(true));bad.complete(17);
    bad.complete(0);need(!bad.begin(false));
    need(bad.result().phase==Exl3RetirementPhase::quarantined &&
        bad.result().first_drain_error==17 && bad.result().prior_execution_failure);
    bad.observe_cleanup_failure(31);bad.observe_cleanup_failure(0);
    need(bad.result().first_drain_error==17 && bad.result().first_cleanup_error==31 &&
        bad.result().prior_execution_failure && !bad.result().reusable());
    Exl3RetirementState unstarted;unstarted.complete(0);need(!unstarted.result().reusable());
    Exl3FinalUseWitness use;
    const auto first=use.begin(1,1,17);need(!use.matches(1,1,first));
    bool busy=false;try{use.begin(1,1,17);}catch(const std::logic_error&){busy=true;}need(busy);
    need(use.finish(first,0));need(use.matches(1,1,first));
    const auto second=use.begin(2,2,17); // same event address, new acquisition
    need(!use.finish(first,0) && !use.matches(1,1,first));
    need(use.finish(second,31));need(!use.finish(second,0) && use.first_error()==31);
    bool refused=false;try{use.begin(3,3,17);}catch(const std::logic_error&){refused=true;}need(refused);
    Exl3FinalUseWitness alternate;
    const auto planned=alternate.begin(3,4,41);
    need(!alternate.finish_after_alternate_event(planned-1,37,0) &&
        !alternate.finish_after_alternate_event(planned,0,0));
    need(alternate.finish_after_alternate_event(planned,37,0) &&
        !alternate.finish_after_alternate_event(planned,39,0) &&
        alternate.matches(3,4,planned));
    const auto alternate_snapshot=alternate.snapshot();
    need(alternate_snapshot.ready && alternate_snapshot.event==37 &&
        alternate_snapshot.generation==planned);
    // A staging slot reuses one event and one context-local scope. An older
    // completion must not certify the next copy, even with unchanged scope IDs.
    Exl3FinalUseWitness staging;
    const auto upload=staging.begin(1,1,29);need(staging.finish(upload,0));
    const auto download=staging.begin(1,1,29);
    need(download>upload && !staging.finish(upload,0) && !staging.matches(1,1,download));
    // Model a failed event record after submission. Later successful event
    // observation must not overwrite this uncertain transfer outcome.
    need(staging.finish(download,7));need(!staging.finish(download,0));
    bool poisoned=false;try{staging.begin(1,1,29);}catch(const std::logic_error&){poisoned=true;}
    need(poisoned && staging.first_error()==7);
    const auto retained=staging;
    const auto snapshot=retained.snapshot();
    need(snapshot.acquisition==1 && snapshot.execution==1 && snapshot.generation==download &&
        snapshot.event==29 && !snapshot.ready && snapshot.first_error==7);
    staging=Exl3FinalUseWitness{}; // The original wrapper's lifetime has ended.
    need(retained.snapshot().first_error==7 && retained.snapshot().generation==download);
    need(!retained.matches(1,1,download));
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
