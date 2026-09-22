#include "exl3/sampling_policy.h"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
}

int main() {
    ninfer::runtime::ResolvedSamplingParameters greedy;
    greedy.temperature=0;greedy.top_k=0;greedy.top_p=0;greedy.min_p=1;
    auto policy=Exl3SamplingPolicy::normalize(greedy,7);
    need(policy.greedy && policy.effective_top_k==7 && policy.top_p_enabled &&
            policy.min_p_enabled && !policy.occurrence_counts_required,
        "stable greedy special case changed resolved parameter record");

    auto sampled=greedy;sampled.temperature=0.7F;sampled.top_k=19;
    sampled.top_p=0.8F;sampled.min_p=0.1F;
    policy=Exl3SamplingPolicy::normalize(sampled);
    need(policy.requires_sampled_distribution() && policy.effective_top_k==19 &&
            policy.top_p_enabled && policy.min_p_enabled,
        "supported positive-temperature filters changed");
    sampled.top_k=20;
    need(Exl3SamplingPolicy::normalize(sampled).effective_top_k==20,
        "registered top-k clamp contract changed");

    auto penalized=greedy;penalized.presence_penalty=0.5F;
    need(Exl3SamplingPolicy::normalize(penalized).requires_sampled_distribution(),
        "temperature-zero penalty was silently discarded as greedy");
    auto invalid=sampled;invalid.top_p=1.01F;
    need(refuses([&]{(void)Exl3SamplingPolicy::normalize(invalid);}),
        "invalid top-p admitted");
    invalid=sampled;invalid.temperature=std::numeric_limits<float>::quiet_NaN();
    need(refuses([&]{(void)Exl3SamplingPolicy::normalize(invalid);}),
        "nonfinite temperature admitted");

    policy=Exl3SamplingPolicy::normalize(sampled,3);
    const std::array<float,3> finite{1,2,3};policy.require_logits(finite);
    auto nonfinite=finite;nonfinite[1]=std::numeric_limits<float>::infinity();
    need(refuses([&]{policy.require_logits(nonfinite);}) &&
            refuses([&]{policy.require_logits(std::span<const float>(finite).first(2));}),
        "nonfinite or wrong-extent logits admitted");
    std::cout<<"exl3_sampling_policy PASS\n";
}
