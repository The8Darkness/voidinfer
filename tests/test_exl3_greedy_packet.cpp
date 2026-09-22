#include "exl3/greedy_packet.h"
#include <iostream>
#include <vector>
using namespace ninfer::exl3;
static void need(bool ok) {if(!ok) throw std::runtime_error("packet assertion");}
template<class F> void refuses(F f) {
    bool rejected=false;try {f();} catch(const std::runtime_error&) {rejected=true;}
    need(rejected);
}
template<class F> void refuses_invalid(F f) {
    bool rejected=false;try {f();} catch(const std::invalid_argument&) {rejected=true;}
    need(rejected);
}
int main() {
    try {
        const auto* saved_option=std::getenv("NINFER_EXL3_DEVICE_GREEDY");
        const std::string saved=saved_option?saved_option:"";
        _putenv_s("NINFER_EXL3_DEVICE_GREEDY","");
        need(exl3_device_greedy_enabled());
        _putenv_s("NINFER_EXL3_DEVICE_GREEDY","0");
        need(!exl3_device_greedy_enabled());
        _putenv_s("NINFER_EXL3_DEVICE_GREEDY","1");
        need(exl3_device_greedy_enabled());
        _putenv_s("NINFER_EXL3_DEVICE_GREEDY","invalid");
        refuses_invalid([]{(void)exl3_device_greedy_enabled();});
        _putenv_s("NINFER_EXL3_DEVICE_GREEDY",saved.c_str());
        for(unsigned rows:{1u,8u,16u}) {
            Exl3GreedyPacket packet;
            packet.rows=rows;packet.generation=19;packet.position=1024;
            packet.serial=7;packet.ready=true;
            for(unsigned i=0;i<rows;++i) packet.decisions[i]={7,static_cast<int>(100+i),0};
            auto validate=[](const auto& p,unsigned n){p.validate(n,19,1024,7);};
            validate(packet,rows);
            const auto greedy=Exl3SelectionAuthorityRequirement::greedy(
                rows,19,1024,7);
            greedy.validate(packet);
            std::vector<std::int64_t> path(rows);
            exl3_greedy_path(42,packet,path);need(path[0]==42);
            for(unsigned i=1;i<rows;++i) need(path[i]==99+i);
            const auto sampled=Exl3SelectionAuthorityRequirement::sampled(
                rows,19,1024,7);
            refuses_invalid([&]{sampled.validate(packet);});
            Exl3SampledDistributionAuthority missing;
            missing.rows=rows;missing.vocabulary=248320;missing.row_stride=248320;
            missing.generation=19;missing.position=1024;missing.serial=7;missing.ready=true;
            refuses_invalid([&]{sampled.validate(missing);});
            std::vector<float> probabilities(static_cast<std::size_t>(rows)*248320);
            auto probability_owner=std::make_shared<int>(1);
            Exl3SampledDistributionAuthority distribution{probability_owner,
                probabilities.data(),rows,248320,248320,19,7,1024,true};
            sampled.validate(distribution);
            refuses_invalid([&]{greedy.validate(distribution);});
            // Even the discarded final score row must be finite and complete.
            auto bad=packet;bad.decisions[rows-1].nonfinite=1;
            refuses([&]{validate(bad,rows);});
            bad=packet;bad.decisions[rows-1].serial=6;refuses([&]{validate(bad,rows);});
            bad=packet;bad.decisions[0].token=248320;refuses([&]{validate(bad,rows);});
            bad=packet;bad.ready=false;refuses([&]{validate(bad,rows);});
            bad=packet;bad.abi=2;refuses([&]{validate(bad,rows);});
            bad=packet;++bad.generation;refuses([&]{validate(bad,rows);});
            bad=packet;++bad.position;refuses([&]{validate(bad,rows);});
            bad=packet;++bad.serial;refuses([&]{validate(bad,rows);});
            bad=packet;bad.rows=0;refuses([&]{validate(bad,rows);});
            bad=packet;bad.rows=17;refuses([&]{validate(bad,rows);});
            bool extent_refused=false;
            try{exl3_greedy_path(42,packet,{});}catch(const std::invalid_argument&){extent_refused=true;}
            need(extent_refused);
            std::array<std::int64_t,17> oversized{};extent_refused=false;
            try{exl3_greedy_path(42,packet,oversized);}catch(const std::invalid_argument&){extent_refused=true;}
            need(extent_refused);
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
