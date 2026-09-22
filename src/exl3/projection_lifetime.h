#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::exl3 {

inline int exl3_down_slice_columns(std::size_t bytes) {
    for(int columns:{256,512,1024,5120})
        if(bytes==std::size_t(17408)*columns*2)return columns;
    throw std::invalid_argument("exact Down decoded slice capacity");
}

// Ordered-stream K6/K7 Down route: transform -> decode -> MMA partials ->
// split reduction/output transform. Touching phases are concurrent consumers.
struct Exl3ProjectionLifetime {
    unsigned first, last;
    std::size_t bytes;
};

struct Exl3ExactDownLifetimePlan {
    std::array<Exl3ProjectionLifetime,4> buffers;

    static Exl3ExactDownLifetimePlan make(int rows,int splits,int k,
                                         bool mcg,bool mul1,bool bias,int decoded_columns=5120,
                                         int input_features=17408,int output_features=5120) {
        if((rows!=256 && rows!=512 && rows!=1024) || splits<1 || splits>5 ||
           (k!=6 && k!=7) || mcg || !mul1 || bias)
            throw std::invalid_argument("exact Down lifetime topology");
        if(input_features<=0 || output_features<=0 || input_features%16 || output_features%16 ||
           decoded_columns<=0 || decoded_columns>output_features || decoded_columns%16)
            throw std::invalid_argument("exact projection lifetime shape");
        Exl3ExactDownLifetimePlan plan;
        plan.buffers={{{0,2,std::size_t(rows)*input_features*2},
                 {1,2,std::size_t(input_features)*decoded_columns*2},
                 {2,3,std::size_t(rows)*output_features*std::size_t(splits)*4},
                 {3,4,std::size_t(rows)*output_features*2}}};
        return plan;
    }

    void require_binding(const std::array<std::uintptr_t,4>& addresses,
                         const std::array<std::size_t,4>& capacities) const {
        for(std::size_t i=0;i<buffers.size();++i) {
            if(!addresses[i] || capacities[i]<buffers[i].bytes ||
               addresses[i]>std::numeric_limits<std::uintptr_t>::max()-buffers[i].bytes)
                throw std::invalid_argument("exact Down lifetime extent");
        }
        for(std::size_t i=0;i<buffers.size();++i)
            for(std::size_t j=i+1;j<buffers.size();++j) {
                const bool live=buffers[i].first<=buffers[j].last &&
                                buffers[j].first<=buffers[i].last;
                const bool overlap=addresses[i]<addresses[j]+buffers[j].bytes &&
                                   addresses[j]<addresses[i]+buffers[i].bytes;
                if(live && overlap)
                    throw std::invalid_argument("exact Down overlapping consumers");
            }
    }
};

} // namespace ninfer::exl3
