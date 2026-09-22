#include "exl3/sampling_control_stop_semantics.h"

#include <iostream>
#include <memory>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
}

int main() {
    auto root=std::make_shared<const int>(1),control_root=std::make_shared<const int>(2);
    Exl3RequestSamplingState state(3,5,7,11);
    state.bind_acquisition(13,root,64);
    Exl3SamplingControlStopSemantics::commit_forced_control_after_resident(
        state,2,control_root);
    need(state.committed_prefix()==66 && state.committed_draws()==0,
        "forced suffix consumed random state or failed to advance root prefix");

    auto terminal_root=std::make_shared<const int>(3);
    Exl3SamplingWindowTransaction terminal(state,66,4);
    (void)terminal.draw_proposal();(void)terminal.draw_acceptance();
    terminal.accept_proposal();terminal.seal_stop_boundary();
    Exl3SamplingControlStopSemantics::claim_model_publication(terminal);
    const auto hidden=Exl3SamplingControlStopSemantics::commit_model_after_resident(
        terminal,1,true,false,terminal_root);
    need(hidden.resident_tokens==1 && hidden.visible_tokens==0 && hidden.terminal &&
            !hidden.terminal_visible && state.committed_prefix()==67 &&
            state.committed_draws()==2,
        "hidden EOS changed resident/random accounting");

    Exl3SamplingWindowTransaction before_correction(state,67,2);
    const auto proposed=before_correction.draw_proposal();
    const auto acceptance=before_correction.draw_acceptance();
    Exl3SamplingControlStopSemantics::abandon_before_correction(before_correction);
    Exl3SamplingWindowTransaction retry(state,67,2);
    need(retry.draw_proposal()==proposed && retry.draw_acceptance()==acceptance,
        "EOS-before-correction consumed an unpublished decision");
    retry.accept_proposal();retry.rollback();

    Exl3SamplingWindowTransaction claimed(state,67,1);
    (void)claimed.draw_proposal();(void)claimed.draw_acceptance();claimed.accept_proposal();
    Exl3SamplingControlStopSemantics::claim_model_publication(claimed);
    state.cancel();
    auto claimed_root=std::make_shared<const int>(4);
    const auto visible=Exl3SamplingControlStopSemantics::commit_model_after_resident(
        claimed,1,false,false,claimed_root);
    need(visible.visible_tokens==1 && state.committed_prefix()==68,
        "claimed resident publication was released by cancellation");

    Exl3RequestSamplingState replacement(3,6,7,11);
    replacement.bind_acquisition(14,root,64);
    Exl3SamplingWindowTransaction isolated(replacement,64,1);
    need(isolated.draw_proposal()!=proposed,
        "replacement request reused cancelled request random stream");
    isolated.rollback();
    need(refuses([&]{state.advance_forced_control(1,root,false);}),
        "uncommitted forced control advanced cancelled sampling state");
    std::cout<<"exl3_sampling_control_stop_semantics PASS\n";
}
