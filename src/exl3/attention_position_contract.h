#pragma once
#include <stdexcept>
namespace ninfer::exl3 {
// Captured with the numerical input binding. This is not an ownership token.
struct Exl3AttentionPositionContract {
    const int* mrope=nullptr;
    int offset=0;
    void require_current(const int* current_mrope,int current_offset) const {
        if(mrope!=current_mrope || offset!=current_offset)
            throw std::invalid_argument("segmented attention rotary binding changed");
    }
};
}
