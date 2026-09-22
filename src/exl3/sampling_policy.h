#pragma once

#include "runtime/contract/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::exl3 {

struct Exl3NormalizedSamplingPolicy {
    runtime::ResolvedSamplingParameters requested;
    std::uint32_t vocabulary=0;
    std::uint32_t effective_top_k=0;
    bool greedy=false,top_p_enabled=false,min_p_enabled=false;
    bool occurrence_counts_required=false;

    [[nodiscard]] bool requires_sampled_distribution() const noexcept{return !greedy;}

    void require_logits(std::span<const float> logits) const {
        if(logits.size()!=vocabulary)
            throw std::invalid_argument("EXL3 sampling logits vocabulary extent");
        if(std::any_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}))
            throw std::invalid_argument("EXL3 sampling logits must be finite");
    }
};

class Exl3SamplingPolicy {
public:
    static Exl3NormalizedSamplingPolicy normalize(
        const runtime::ResolvedSamplingParameters& sampling,
        std::uint32_t vocabulary=248320) {
        if(!vocabulary || !std::isfinite(sampling.temperature) ||
           !std::isfinite(sampling.top_p) || !std::isfinite(sampling.min_p) ||
           !std::isfinite(sampling.presence_penalty) ||
           !std::isfinite(sampling.frequency_penalty) || sampling.temperature<0 ||
           sampling.temperature>2 || sampling.top_k<0 || sampling.top_p<0 ||
           sampling.top_p>1 || sampling.min_p<0 || sampling.min_p>1 ||
           sampling.presence_penalty<-2 || sampling.presence_penalty>2 ||
           sampling.frequency_penalty<-2 || sampling.frequency_penalty>2)
            throw std::invalid_argument("EXL3 resolved sampling policy invalid");
        Exl3NormalizedSamplingPolicy result;
        result.requested=sampling;result.vocabulary=vocabulary;
        // Preserve the registered sampler's fixed top-k menu exactly: 0 or
        // values >=20 select min(20,vocabulary); 1..19 retain their value.
        const auto requested_top_k=sampling.top_k<=0 || sampling.top_k>=20?
            20u:static_cast<std::uint32_t>(sampling.top_k);
        result.effective_top_k=std::min(requested_top_k,vocabulary);
        result.top_p_enabled=sampling.top_p<1;
        result.min_p_enabled=sampling.min_p>0;
        result.occurrence_counts_required=sampling.presence_penalty!=0 ||
            sampling.frequency_penalty!=0;
        // EXL3's currently supported special case is stricter than the generic
        // op: penalties are never silently ignored under temperature zero.
        result.greedy=sampling.temperature==0 && !result.occurrence_counts_required;
        return result;
    }
};

} // namespace ninfer::exl3
