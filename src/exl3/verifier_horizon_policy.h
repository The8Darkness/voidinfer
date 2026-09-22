#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <optional>
#include <cmath>
#include <array>

namespace ninfer::exl3 {
// Uncalibrated, explicitly selected outer-horizon experiment. The default
// keeps physical B8. A separately admitted execution route may run the same
// decision at its native 2..8-row draft extent; neither changes L2 authority.
class Exl3VerifierHorizonPolicy {
public:
    enum class ProposalSource : unsigned { unknown,neural_b8,suffix,seed_only };
    struct PublicationConstraint {
        std::optional<double> maximum_ms;
        // Supplied complete-window estimate, including both neural blocks,
        // intervening target work and publication. Not a measured p95 or SLA.
        bool allows(std::optional<double> complete_estimate_ms) const {
            if((maximum_ms && (!std::isfinite(*maximum_ms) || *maximum_ms<=0)) ||
                (complete_estimate_ms && (!std::isfinite(*complete_estimate_ms) || *complete_estimate_ms<0)))
                throw std::invalid_argument("invalid publication constraint/estimate");
            return !maximum_ms || (complete_estimate_ms && *complete_estimate_ms<=*maximum_ms);
        }
    };
    struct CostMenu {
        struct Estimate {
            double useful_rows=0,complete_ms=0;
            std::uint64_t samples=0;
            bool stable=false;
        };
        // Explicit supplied estimates for2/4/8. complete_ms already includes
        // unchanged neural B8 work; never add overlapping stage observations.
        std::array<std::optional<Estimate>,3> estimates{};
        double fixed_neural_ms=0;
        std::uint64_t minimum_samples=0;
        std::optional<double> maximum_complete_ms;
        void validate() const {
            if(!minimum_samples || !std::isfinite(fixed_neural_ms) || fixed_neural_ms<0 ||
                (maximum_complete_ms && (!std::isfinite(*maximum_complete_ms) || *maximum_complete_ms<=0)))
                throw std::invalid_argument("invalid supplied horizon cost constraints");
            for(unsigned i=0;i<3;++i)if(estimates[i]) {
                const auto& e=*estimates[i];
                if(!std::isfinite(e.useful_rows) || e.useful_rows<=0 || e.useful_rows>((2u<<i)+1) ||
                    !std::isfinite(e.complete_ms) || e.complete_ms<=0 || e.complete_ms<fixed_neural_ms ||
                    !std::isfinite(e.useful_rows/e.complete_ms))
                    throw std::invalid_argument("invalid complete horizon cost estimate");
            }
        }
        unsigned select(unsigned available,unsigned fallback) const noexcept {
            unsigned selected=0;double best=0;
            for(unsigned i=0;i<3;++i) {
                const unsigned rows=2u<<i;if(rows>available)break;
                if(!estimates[i] || !estimates[i]->stable || estimates[i]->samples<minimum_samples)
                    return fallback;
                const auto& e=*estimates[i];
                if(maximum_complete_ms && e.complete_ms>*maximum_complete_ms)continue;
                const double score=e.useful_rows/e.complete_ms;
                if(score>best){best=score;selected=rows;}
            }
            return selected?selected:fallback;
        }
    };
    struct Costs {
        // Independent observations in milliseconds. Stages may overlap; only
        // an explicitly measured complete round can provide complete_ms.
        // Execution verification_ms is inclusive host wall time through repair
        // and the completion fence, not isolated kernel or GPU time.
        std::optional<double> neural_ms,verification_ms,repair_ms,publication_ms,complete_ms;
        std::optional<unsigned> neural_rows;
        void validate() const {
            for(const auto value:{neural_ms,verification_ms,repair_ms,publication_ms,complete_ms})
                if(value && (!std::isfinite(*value) || *value<0))
                    throw std::invalid_argument("policy cost must be finite nonnegative or unknown");
            if(neural_rows && *neural_rows!=0 && (*neural_rows<2 || *neural_rows>8))
                throw std::invalid_argument("policy neural work must be absent or rows2..8");
        }
    };
    struct Observation {
        unsigned proposed=0,accepted=0,replay=0,target_calls=0;
        bool rejected=false,stopped=false;
        Costs costs;
        ProposalSource source=ProposalSource::unknown;
    };
    struct Counters {
        struct Position {std::uint64_t accepted=0,rejected=0,censored=0,unproposed=0;};
        std::array<Position,7> positions{}; // rows1..7; authoritative seed excluded
        std::array<std::array<Position,7>,4> source_positions{};
        std::uint64_t seed_accepted=0,seed_rejected=0,seed_censored=0;
        std::uint64_t published_rounds=0,accepted_suffix=0,rejected_suffix=0;
        std::uint64_t censored_suffix=0,replay_rows=0,target_calls=0;
    };
    unsigned horizon(unsigned available) const noexcept {
        const auto fallback=std::min(available,horizon_);
        return cost_menu_?cost_menu_->select(available,fallback):fallback;
    }
    void set_cost_menu(CostMenu menu) {menu.validate();cost_menu_=menu;}
    bool has_cost_menu() const noexcept {return cost_menu_.has_value();}
    const Counters& counters() const noexcept {return counters_;}
    const Costs& last_costs() const noexcept {return last_costs_;}
    void reset() noexcept {*this=Exl3VerifierHorizonPolicy{};}
    void observe(Observation value) {
        value.costs.validate();
        if(static_cast<unsigned>(value.source)>=4 ||
            (value.source==ProposalSource::seed_only &&
                (value.proposed!=1 || (value.costs.neural_rows && *value.costs.neural_rows!=0))) ||
            (value.source==ProposalSource::neural_b8 &&
                (!value.costs.neural_rows ||
                 (*value.costs.neural_rows!=8 && *value.costs.neural_rows!=value.proposed))) ||
            (value.source==ProposalSource::suffix && value.costs.neural_rows!=0))
            throw std::invalid_argument("policy proposal source mismatch");
        if(value.proposed<1 || value.proposed>8 || value.accepted>value.proposed ||
           (value.rejected && value.accepted>=value.proposed) ||
           (!value.rejected && !value.stopped && value.accepted!=value.proposed))
            throw std::invalid_argument("verifier policy observation extent");
        ++counters_.published_rounds;
        last_costs_=value.costs;
        counters_.replay_rows+=value.replay;counters_.target_calls+=value.target_calls;
        // Row zero is the authoritative seed, not a neural/suffix prediction.
        const unsigned survived=value.accepted?value.accepted-1:0;
        const unsigned failed=value.rejected && value.accepted>0?1:0;
        counters_.accepted_suffix+=survived;counters_.rejected_suffix+=failed;
        counters_.censored_suffix+=value.proposed-1-survived-failed;
        if(value.accepted)++counters_.seed_accepted;
        else if(value.rejected)++counters_.seed_rejected;
        else ++counters_.seed_censored;
        for(unsigned row=1;row<8;++row) {
            for(auto* destination:{&counters_.positions[row-1],
                &counters_.source_positions[static_cast<unsigned>(value.source)][row-1]}) {
            auto& position=*destination;
            if(row>=value.proposed)++position.unproposed;
            else if(row<value.accepted)++position.accepted;
            else if(value.rejected && value.accepted>0 && row==value.accepted)++position.rejected;
            else ++position.censored;
            }
        }
        // A stop or seed mismatch supplies no uncensored horizon comparison.
        // External tail caps also must not teach that a longer block succeeded.
        if(value.stopped || !value.accepted || value.proposed!=horizon_) {
            early_=full_=0;return;
        }
        if(value.rejected && value.accepted<=horizon_/2) {
            full_=0;
            if(++early_==2){horizon_=std::max(2u,horizon_/2);early_=0;}
        } else if(!value.rejected) {
            early_=0;
            if(++full_==3){horizon_=std::min(8u,horizon_*2);full_=0;}
        } else {early_=full_=0;}
    }
private:
    unsigned horizon_=8,early_=0,full_=0;
    Counters counters_;
    Costs last_costs_;
    std::optional<CostMenu> cost_menu_;
};
}
