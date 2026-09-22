#include "exl3/candidate_option_compatibility.h"
#include "exl3/candidate_route_contract.h"
#include "exl3/complete_route_counters.h"
#include <iostream>
#include <stdexcept>

namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> void refuses(F&& body,const char* message) {
    bool refused=false;try{body();}catch(const std::invalid_argument&){refused=true;}
    need(refused,message);
}
}

int main() try {
    using namespace ninfer::exl3;
    require_engine_exact_route({"ordinary-host",Exl3NumericalRouteClass::exact,
        "host-kv-fp16","fp16",true,false},"host-kv-fp16","fp16");
    require_engine_exact_route({"NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7",
        Exl3NumericalRouteClass::exact,"host-kv-fp16","fp16",true,false},
        "host-kv-fp16","fp16");
    refuses([]{require_engine_exact_route({"NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7",
        Exl3NumericalRouteClass::exact,"host-kv-fp16","fp16",false,false},
        "host-kv-fp16","fp16");},"unqualified exact K7 route entered exact profile");
    refuses([]{require_engine_exact_route({"tiled-attention",Exl3NumericalRouteClass::numeric,
        "host-kv-fp16","fp16",false,false},"host-kv-fp16","fp16");},
        "numeric route entered exact profile");
    refuses([]{require_engine_exact_route({"altered",Exl3NumericalRouteClass::exact,
        "oscar-int2","fp16",true,false},"host-kv-fp16","fp16");},
        "altered representation entered exact profile");
    refuses([]{require_engine_exact_route({"composed",Exl3NumericalRouteClass::numeric,
        "host-kv-fp16","fp16",false,true},"host-kv-fp16","fp16");},
        "numeric/exact composition was relabelled exact");

    Exl3CandidateOptionCompatibility base;
    base.physical_lanes=2;base.max_context=32768;base.exact_host_kv=true;
    require_candidate_option_compatibility(base);
    auto target=base;target.shared_target_q=true;target.shared_target_kv=true;
    require_candidate_option_compatibility(target);
    auto missing_target=target;missing_target.shared_target_q=false;
    refuses([&]{require_candidate_option_compatibility(missing_target);},
        "dependent target family lacked Q owner");
    auto mixed=base;mixed.device_prefix=true;mixed.shared_prefix=true;mixed.shared_pages=true;
    refuses([&]{require_candidate_option_compatibility(mixed);},
        "mutually exclusive prefix/page routes composed");
    auto native=base;native.native64k=true;native.max_context=65536;
    refuses([&]{require_candidate_option_compatibility(native);},
        "native64K admitted physical C2");
    auto packets=base;packets.batched_greedy=true;
    refuses([&]{require_candidate_option_compatibility(packets);},
        "batched greedy admitted without device packet owner");

    Exl3CompleteRouteCounters counters;
    counters.record_attempt(8,8,0);counters.record_target_work(3,3);
    counters.record_publication(3,2,1);counters.record_diagnostic(5);
    const auto value=counters.snapshot();
    need(value.proposed_rows==8 && value.verified_rows==11 &&
        value.replayed_rows==3 && value.committed_model_rows==3 &&
        value.externally_visible_model_rows==2 && value.hidden_terminal_rows==1 &&
        value.diagnostic_rows==5,"complete-route counters merged distinct work");
    refuses([&]{counters.record_publication(2,2,1);},
        "invalid visible/hidden publication partition accepted");
    std::cout<<"exl3_candidate_contracts PASS\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
