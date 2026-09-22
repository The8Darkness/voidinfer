#pragma once
#include <stdexcept>

namespace ninfer::exl3 {
enum class Exl3HeadConsumer { root_only, verification, sampled_rows, diagnostic_rows };
struct Exl3HeadConsumerPlan {
    int first_row=0,rows=0,root_row=0;
    static Exl3HeadConsumerPlan make(int submitted,Exl3HeadConsumer consumer,
                                    bool omit_intermediate) {
        if(submitted<=0)throw std::invalid_argument("head requires represented rows");
        switch(consumer) {
        case Exl3HeadConsumer::root_only: break;
        case Exl3HeadConsumer::verification:
        case Exl3HeadConsumer::sampled_rows:
        case Exl3HeadConsumer::diagnostic_rows:
            if(omit_intermediate && submitted>1)
                throw std::invalid_argument("head omission would discard consumed rows");
            break;
        default: throw std::invalid_argument("unknown head consumer");
        }
        return {omit_intermediate?submitted-1:0,omit_intermediate?1:submitted,submitted-1};
    }
};
}
