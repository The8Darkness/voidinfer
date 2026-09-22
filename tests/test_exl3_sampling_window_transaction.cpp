#include "exl3/sampling_window_transaction.h"

#include <iostream>
#include <memory>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
}

int main() {
    auto root=std::make_shared<const int>(1),root65=std::make_shared<const int>(2);
    Exl3RequestSamplingState state(3,5,7,11);
    state.bind_acquisition(13,root,64);

    Exl3SamplingWindowTransaction rejected(state,64,8);
    const auto proposal=rejected.draw_proposal();
    const auto acceptance=rejected.draw_acceptance();
    const auto correction=rejected.reject_and_draw_correction();
    need(rejected.sealed() && refuses([&]{(void)rejected.draw_proposal();}),
        "first rejection did not seal dependent speculative decisions");
    rejected.claim_publication();rejected.commit(1,root65);
    need(state.committed_draws()==3 && state.committed_prefix()==65,
        "published correction did not commit proposal/accept/correction draws and prefix");

    Exl3SamplingWindowTransaction aborted(state,65,4);
    const auto retry_proposal=aborted.draw_proposal();
    const auto retry_acceptance=aborted.draw_acceptance();aborted.accept_proposal();
    const auto retry_second_proposal=aborted.draw_proposal();
    const auto retry_second_acceptance=aborted.draw_acceptance();aborted.accept_proposal();
    aborted.rollback();
    Exl3SamplingWindowTransaction aborted_again(state,65,4);
    need(aborted_again.draw_proposal()==retry_proposal &&
            aborted_again.draw_acceptance()==retry_acceptance,
        "repeated abort consumed the first tentative decision draws");
    aborted_again.accept_proposal();
    need(aborted_again.draw_proposal()==retry_second_proposal &&
            aborted_again.draw_acceptance()==retry_second_acceptance,
        "repeated abort changed later tentative decision draws");
    aborted_again.accept_proposal();aborted_again.rollback();

    auto root67=std::make_shared<const int>(3);
    Exl3SamplingWindowTransaction stopped(state,65,4);
    need(stopped.draw_proposal()==retry_proposal &&
            stopped.draw_acceptance()==retry_acceptance,
        "rollback did not reproduce stop-window decision");
    stopped.accept_proposal();
    need(stopped.draw_proposal()==retry_second_proposal &&
            stopped.draw_acceptance()==retry_second_acceptance,
        "stop-window second decision changed");
    stopped.accept_proposal();
    const auto unpublished_proposal=stopped.draw_proposal();
    const auto unpublished_acceptance=stopped.draw_acceptance();stopped.accept_proposal();
    stopped.seal_stop_boundary();stopped.claim_publication();stopped.commit(2,root67);
    need(state.committed_draws()==7 && state.committed_prefix()==67,
        "partial publication committed stop-truncated decision");

    Exl3SamplingWindowTransaction after_stop(state,67,1);
    need(after_stop.draw_proposal()==unpublished_proposal &&
            after_stop.draw_acceptance()==unpublished_acceptance,
        "stop-truncated suffix decision was not replayable");
    after_stop.accept_proposal();after_stop.rollback();
    need(refuses([&]{stopped.commit(0,root67);}),
        "completed sampling window published twice");
    (void)proposal;(void)acceptance;(void)correction;
    std::cout<<"exl3_sampling_window_transaction PASS\n";
}
