#include "exl3/verifier_horizon_policy.h"
#include "exl3/device_horizon_cost_policy.h"
#include <iostream>
#include <limits>
using Policy=ninfer::exl3::Exl3VerifierHorizonPolicy;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try{
    {
        Policy::PublicationConstraint constraint;
        require(constraint.allows(std::nullopt),"absent publication budget restricted route");
        constraint.maximum_ms=10;
        require(!constraint.allows(std::nullopt) && !constraint.allows(11) && constraint.allows(10),
            "publication budget admitted unknown/over-budget choice or refused boundary");
        for(double invalid:{-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            bool refused=false;try{(void)constraint.allows(invalid);}catch(const std::invalid_argument&){refused=true;}
            require(refused,"invalid publication estimate admitted");
        }
    }
    Policy p;require(p.horizon(8)==8 && p.horizon(1)==1,"default/tail cap");
    {
        Policy seed;Policy::Observation value{1,1,0,1,false,false};
        value.source=Policy::ProposalSource::seed_only;value.costs.neural_rows=8;
        bool refused=false;try{seed.observe(value);}catch(const std::invalid_argument&){refused=true;}
        require(refused && !seed.counters().published_rounds && !seed.last_costs().neural_rows,
            "seed-only neural work contradiction changed publication evidence");
        value.costs.neural_rows.reset();seed.observe(value);
        require(!seed.last_costs().neural_rows,"unknown seed work was manufactured");
        value.costs.neural_rows=0;seed.observe(value);
        require(seed.counters().published_rounds==2 && seed.counters().seed_accepted==2 &&
            seed.last_costs().neural_rows==0 && seed.horizon(8)==8,
            "seed-only observation changed legal horizon or lost zero-work attribution");
    }
    {
        // Hand-specified position oracle: A accepted, R rejected, C censored,
        // U not proposed. No policy output constructs these expected labels.
        struct Case {unsigned rows,accepted;bool rejected,stopped;const char* positions;char seed;};
        const std::array cases{
            Case{8,0,true,false,"CCCCCCC",'R'},Case{8,1,true,false,"RCCCCCC",'A'},
            Case{8,7,true,false,"AAAAAAR",'A'},Case{8,8,false,false,"AAAAAAA",'A'},
            Case{4,4,false,false,"AAAUUUU",'A'},Case{4,2,false,true,"ACCUUUU",'A'},
            Case{2,1,true,false,"RUUUUUU",'A'},Case{1,1,false,false,"UUUUUUU",'A'}};
        for(const auto& fixture:cases)for(auto source:{Policy::ProposalSource::unknown,
            Policy::ProposalSource::neural_b8,Policy::ProposalSource::suffix}) {
            Policy observed;Policy::Observation value{fixture.rows,fixture.accepted,0,1,fixture.rejected,fixture.stopped};
            value.source=source;
            if(source==Policy::ProposalSource::neural_b8)value.costs.neural_rows=8;
            if(source==Policy::ProposalSource::suffix)value.costs.neural_rows=0;
            observed.observe(value);
            const auto& counts=observed.counters();
            require(counts.seed_accepted==(fixture.seed=='A') && counts.seed_rejected==(fixture.seed=='R'),
                "independent seed oracle mismatch");
            for(unsigned row=0;row<7;++row) {
                const char expected=fixture.positions[row];
                for(const auto* actual:{&counts.positions[row],&counts.source_positions[static_cast<unsigned>(source)][row]})
                    require(actual->accepted==(expected=='A') && actual->rejected==(expected=='R') &&
                        actual->censored==(expected=='C') && actual->unproposed==(expected=='U'),
                        "independent position censoring oracle mismatch");
                for(unsigned other=0;other<4;++other)if(other!=static_cast<unsigned>(source)) {
                    const auto& untouched=counts.source_positions[other][row];
                    require(!untouched.accepted && !untouched.rejected && !untouched.censored && !untouched.unproposed,
                        "position evidence crossed proposal source");
                }
            }
        }
    }
    {
        Policy priced;Policy::CostMenu menu;
        menu.minimum_samples=10;menu.fixed_neural_ms=100;
        menu.estimates={Policy::CostMenu::Estimate{2,101,10,true},
            Policy::CostMenu::Estimate{3,102,10,true},Policy::CostMenu::Estimate{5,104,10,true}};
        priced.set_cost_menu(menu);require(priced.horizon(8)==8,"fixed neural cost ignored");
        menu.maximum_complete_ms=102;priced.set_cost_menu(menu);
        require(priced.horizon(8)==4,"supplied complete-cost constraint ignored");
        menu.estimates[1]->samples=9;priced.set_cost_menu(menu);
        require(priced.horizon(8)==8,"insufficient samples replaced fallback");
        menu.estimates[1]->samples=10;menu.estimates[1]->stable=false;priced.set_cost_menu(menu);
        require(priced.horizon(8)==8 && priced.horizon(1)==1,"unstable history or terminal cap changed fallback");
        priced.reset();require(priced.horizon(8)==8,"cost policy survived request reset");
    }
    {
        Policy separated;Policy::Observation full{8,8,0,1,false,false};
        full.costs.neural_rows=8;full.source=Policy::ProposalSource::neural_b8;
        separated.observe(full);
        full.costs.neural_rows=0;full.source=Policy::ProposalSource::suffix;
        separated.observe(full);
        require(separated.counters().positions[6].accepted==2 &&
            separated.counters().source_positions[1][6].accepted==1 &&
            separated.counters().source_positions[2][6].accepted==1 &&
            separated.counters().source_positions[0][6].accepted==0,
            "proposal families contaminated per-position evidence");
        full.source=Policy::ProposalSource::neural_b8;bool refused=false;
        try{separated.observe(full);}catch(const std::invalid_argument&){refused=true;}
        require(refused && separated.counters().published_rounds==2,"invalid source changed learning");
        separated.reset();require(separated.counters().source_positions[1][6].accepted==0,
            "source position evidence leaked across request reset");
    }
    {
        Policy native;Policy::Observation short_block{4,4,0,1,false,false};
        short_block.source=Policy::ProposalSource::neural_b8;
        short_block.costs.neural_rows=4;native.observe(short_block);
        require(native.last_costs().neural_rows==4 && native.counters().published_rounds==1,
            "native neural horizon physical extent was not retained");
        short_block.costs.neural_rows=3;bool refused=false;
        try{native.observe(short_block);}catch(const std::invalid_argument&){refused=true;}
        require(refused && native.counters().published_rounds==1,
            "mismatched native neural extent trained policy");
    }
    require(!p.last_costs().complete_ms && !p.last_costs().neural_rows,"unknown cost manufactured");
    {
        Policy costs;
        Policy::Observation observed{8,8,0,1,false,false};
        observed.costs.neural_ms=5;observed.costs.verification_ms=7;
        observed.costs.repair_ms=4;observed.costs.complete_ms=9;observed.costs.neural_rows=8;
        costs.observe(observed);
        require(costs.last_costs().complete_ms==9 && !costs.last_costs().publication_ms,
            "overlapping cost stages summed or unknown filled");
        for(double bad:{-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            observed.costs.neural_ms=bad;bool refused=false;
            try{costs.observe(observed);}catch(const std::invalid_argument&){refused=true;}
            require(refused && costs.counters().published_rounds==1,"invalid cost trained policy");
        }
        costs.reset();require(!costs.last_costs().complete_ms,"cost leaked across reset");
    }
    p.observe({8,2,3,2,true,false});p.observe({8,2,3,2,true,false});
    require(p.horizon(8)==4,"two early rejects did not bound outer work");
    require(p.counters().positions[0].accepted==2 && p.counters().positions[1].rejected==2 &&
        p.counters().positions[2].censored==2 && p.counters().seed_accepted==2,
        "per-position mismatch boundary accounting");
    require(p.counters().accepted_suffix==2 && p.counters().rejected_suffix==2 &&
        p.counters().censored_suffix==10,"wrong-ancestor suffix counted as failures");
    p.observe({4,4,0,1,false,false});p.observe({4,4,0,1,false,false});
    p.observe({4,2,2,2,false,true});
    require(p.horizon(8)==4,"stop increased horizon");
    for(int i=0;i<3;++i)p.observe({4,4,0,1,false,false});
    require(p.horizon(8)==8,"full survival recovery");
    p.observe({2,2,0,1,false,false});require(p.horizon(8)==8,"tail cap changed horizon");
    p.reset();p.observe({8,0,1,2,true,false});
    require(p.counters().rejected_suffix==0 && p.counters().censored_suffix==7,"seed counted as proposal failure");
    require(p.counters().seed_rejected==1 && p.counters().positions[0].rejected==0 &&
        p.counters().positions[6].censored==1,"seed mismatch polluted neural positions");
    {
        for(unsigned limit:{1u,2u,4u,8u}) {
            Policy seed_stop;
            Policy::Observation observation{limit,1,0,1,false,true};
            observation.source=limit==1?Policy::ProposalSource::seed_only:Policy::ProposalSource::neural_b8;
            observation.costs.neural_rows=limit==1?0:8;
            seed_stop.observe(observation);
            const auto& counts=seed_stop.counters();
            require(counts.seed_accepted==1 && counts.accepted_suffix==0 && counts.rejected_suffix==0 &&
                counts.censored_suffix==limit-1 && seed_stop.horizon(8)==8,
                "terminal seed trained on unconsumed neural rows");
            for(unsigned row=1;row<8;++row) {
                const auto& actual=counts.positions[row-1];
                const auto& source=counts.source_positions[static_cast<unsigned>(observation.source)][row-1];
                require(actual.accepted==0 && actual.rejected==0 && actual.censored==(row<limit?1u:0u) &&
                    actual.unproposed==(row>=limit?1u:0u) && source.censored==actual.censored &&
                    source.unproposed==actual.unproposed,"terminal seed mixed censored and absent proposal rows");
            }
        }
        Policy stopped;stopped.observe({4,2,0,1,false,true});
        require(stopped.counters().positions[0].accepted==1 &&
            stopped.counters().positions[1].censored==1 &&
            stopped.counters().positions[3].unproposed==1 &&
            stopped.counters().positions[1].rejected==0,
            "terminal truncation or unavailable rows learned as rejection");
    }
    bool refused=false;try{p.observe({8,9,0,1,false,false});}catch(const std::invalid_argument&){refused=true;}
    require(refused && p.counters().published_rounds==1,"invalid observation changed policy");
    p.reset();require(p.counters().published_rounds==0 && p.horizon(8)==8,"request reset leaked policy");
    {
        ninfer::exl3::Exl3DeviceHorizonCostPolicy cost;
        require(cost.select(8)==8 && cost.select(3)==3,
            "device policy initial B8/tail admission");
        for(int i=0;i<4;++i)cost.observe(8,3,200.0);
        require(cost.select(8)==4,"device policy did not calibrate B4");
        for(int i=0;i<4;++i)cost.observe(4,3,160.0);
        require(cost.select(8)==4 && cost.snapshot(4).useful_per_ms>
            cost.snapshot(8).useful_per_ms,
            "device policy ignored measured complete-round useful rate");
        for(int i=0;i<4;++i)cost.observe(4,3,160.0);
        require(cost.select(8)==8,"device policy did not probe opposing arm");
        cost.observe(8,8,100.0);
        require(cost.select(8)==8,"device policy ignored changed B8 evidence");
        bool invalid=false;
        try{cost.observe(8,9,1.0);}catch(const std::invalid_argument&){invalid=true;}
        require(invalid && cost.observed()==13,
            "invalid device cost observation mutated request history");
        ninfer::exl3::Exl3DeviceHorizonCostPolicy fresh;
        require(fresh.select(8)==8 && fresh.observed()==0,
            "device horizon history crossed requests");
    }
    std::cout<<"VERIFIER_HORIZON_POLICY PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
