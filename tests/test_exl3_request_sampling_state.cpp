#include "exl3/request_sampling_state.h"

#include <iostream>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
bool refuses(auto&& call){try{call();}catch(...){return true;}return false;}
}

int main() {
    auto revision=std::make_shared<const int>(1);
    auto next_revision=std::make_shared<const int>(2);
    Exl3RequestSamplingState request(11,7,0xabc,123);
    request.bind_acquisition(41,revision,64);
    auto first=request.begin(64);
    const auto draw0=request.draw(first),draw1=request.draw(first);
    request.rollback(first);
    auto retry=request.begin(64);
    need(request.draw(retry)==draw0 && request.draw(retry)==draw1,
        "rollback consumed committed request randomness");
    request.claim_publication(retry);request.commit(retry,1,65,next_revision);
    need(request.committed_draws()==1 && request.committed_prefix()==65,
        "partial acceptance committed rejected draw or lost prefix");
    need(refuses([&]{(void)request.begin(64);}),
        "stale accepted prefix opened a new sampling transaction");
    auto suffix=request.begin(65);
    need(request.draw(suffix)==draw1,"partial commit did not replay first rejected draw");
    request.rollback(suffix);

    Exl3RequestSamplingState independent(11,8,0xabc,123);
    independent.bind_acquisition(42,revision,64);
    auto other=independent.begin(64);
    need(independent.draw(other)!=draw0,"independent request shared RNG stream");
    independent.rollback(other);

    Exl3RequestSamplingState reload(12,7,0xabc,123);
    reload.bind_acquisition(1,revision,64);
    auto replay=reload.begin(64);
    need(reload.draw(replay)==draw0 && reload.reproducibility()==
            Exl3SamplingReproducibility::seeded_request_replay_across_reload,
        "declared seeded replay policy changed across clean reload");
    reload.rollback(replay);

    auto cancelled=request.begin(65);request.cancel();
    need(request.cancelled() && refuses([&]{(void)request.draw(cancelled);}) &&
            refuses([&]{request.commit(cancelled,0,65,next_revision);}),
        "cancelled transaction could draw or publish");
    need(refuses([&]{request.bind_acquisition(99,revision,65);}),
        "cancelled request rebound to another physical acquisition");
    std::cout<<"request_sampling_state PASS\n";
}
