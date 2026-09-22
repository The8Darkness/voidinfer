#include "exl3/draft_distribution_contract.h"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
struct OwnedQ { std::array<float,6> values{0.5F,0.3F,0.2F,0.1F,0.4F,0.5F}; };
}

int main() {
    const auto actual=Exl3DraftDistributionAvailability::current_dflash2();
    need(actual.greedy_tokens_available && actual.internal_selector_scores &&
            !actual.normalized_q_available && !actual.complete_vocabulary_support &&
            actual.permits_explicit_greedy_fallback() &&
            !actual.supports(Exl3SampledAcceptanceMethod::independent_rejection_with_residual),
        "current parallel masked proposal was promoted to autoregressive q");

    auto wrong_conditioning=actual;
    wrong_conditioning.normalized_q_available=true;
    wrong_conditioning.complete_vocabulary_support=true;
    need(!wrong_conditioning.supports(
            Exl3SampledAcceptanceMethod::independent_rejection_with_residual),
        "parallel masked scores admitted as predecessor-conditioned q");
    wrong_conditioning.conditioning=Exl3DraftDistributionConditioning::autoregressive_predecessor;
    need(wrong_conditioning.supports(
            Exl3SampledAcceptanceMethod::independent_rejection_with_residual) &&
            !wrong_conditioning.supports(Exl3SampledAcceptanceMethod::unspecified),
        "compatible and incompatible acceptance methods were not distinguished");

    auto owner=std::make_shared<OwnedQ>();
    Exl3DraftDistributionAuthority q;
    q.method=Exl3SampledAcceptanceMethod::independent_rejection_with_residual;
    q.conditioning=Exl3DraftDistributionConditioning::autoregressive_predecessor;
    q.owner=owner;q.probabilities=owner->values.data();q.rows=2;q.vocabulary=3;
    q.row_stride=3;q.request_generation=11;q.root_revision=17;q.first_position=29;
    q.normalized=true;q.complete_vocabulary_support=true;q.ready=true;
    q.require(q.method,2,3,11,17,29);
    auto stale=q;stale.root_revision=18;
    auto missing=q;missing.probabilities=nullptr;
    auto parallel=q;parallel.conditioning=
        Exl3DraftDistributionConditioning::parallel_masked_target_taps;
    need(refuses([&]{stale.require(q.method,2,3,11,17,29);}) &&
            refuses([&]{missing.require(q.method,2,3,11,17,29);}) &&
            refuses([&]{parallel.require(q.method,2,3,11,17,29);}) &&
            refuses([&]{q.require(Exl3SampledAcceptanceMethod::unspecified,2,3,11,17,29);}),
        "unavailable, stale, wrong-conditioning or incompatible q admitted");
    std::cout<<"exl3_draft_distribution_contract PASS\n";
}
