#pragma once

#include "exl3/draft_distribution_contract.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::exl3 {

struct Exl3SampledAcceptanceOutcome {
    bool accepted=false;
    bool correction_draw_used=false;
    std::uint32_t token=0;
    double acceptance_probability=0;
    std::vector<double> residual;
};

// Independent FP64 reference for one speculative position. It is deliberately
// not a production sampler. Both p and q must be complete normalized
// predecessor-conditioned distributions over the same vocabulary. The token
// was drawn from q before this call; a zero-q proposed token is impossible.
class Exl3IndependentAcceptanceReference {
    static constexpr double normalization_tolerance=1e-9;

    static void require_distribution(std::span<const double> values) {
        if(values.empty())throw std::invalid_argument("empty sampled distribution");
        double sum=0;
        for(const auto value:values) {
            if(!std::isfinite(value) || value<0 || value>1)
                throw std::invalid_argument("sampled distribution value invalid");
            sum+=value;
        }
        if(!std::isfinite(sum) || std::abs(sum-1)>normalization_tolerance)
            throw std::invalid_argument("sampled distribution is not normalized");
    }

    static void require_uniform(double value) {
        if(!std::isfinite(value) || value<0 || value>=1)
            throw std::invalid_argument("sampling draw must be in [0,1)");
    }

public:
    static Exl3SampledAcceptanceOutcome evaluate(
        std::span<const double> target_p,std::span<const double> draft_q,
        std::uint32_t proposed_token,double acceptance_draw,double correction_draw) {
        require_distribution(target_p);require_distribution(draft_q);
        require_uniform(acceptance_draw);require_uniform(correction_draw);
        if(target_p.size()!=draft_q.size() || proposed_token>=target_p.size())
            throw std::invalid_argument("sampled acceptance distribution extent");
        const auto q=draft_q[proposed_token];
        if(q==0)throw std::invalid_argument("proposed token has zero draft probability");

        Exl3SampledAcceptanceOutcome result;
        result.acceptance_probability=std::min(1.0,target_p[proposed_token]/q);
        if(acceptance_draw<result.acceptance_probability) {
            result.accepted=true;result.token=proposed_token;
            return result;
        }

        result.residual.resize(target_p.size());
        double residual_sum=0;
        for(std::size_t i=0;i<target_p.size();++i) {
            result.residual[i]=std::max(0.0,target_p[i]-draft_q[i]);
            residual_sum+=result.residual[i];
        }
        if(!std::isfinite(residual_sum) || residual_sum<=normalization_tolerance)
            throw std::logic_error("rejection has no positive residual distribution");
        for(auto& probability:result.residual)probability/=residual_sum;

        result.correction_draw_used=true;
        double cumulative=0;
        std::uint32_t last_positive=0;
        for(std::uint32_t token=0;token<result.residual.size();++token) {
            if(result.residual[token]>0)last_positive=token;
            cumulative+=result.residual[token];
            if(correction_draw<cumulative) {
                result.token=token;return result;
            }
        }
        // Deterministic FP64 round-off fallback; never changes the support.
        result.token=last_positive;
        return result;
    }
};

} // namespace ninfer::exl3
