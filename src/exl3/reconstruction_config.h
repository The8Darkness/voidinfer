#pragma once

#include <cstring>
#include <cstddef>
#include <stdexcept>

namespace ninfer::exl3 {

// This is reservation admission, not a timing model. Existing explicit opt-in
// remains the only preference for reconstruction when costs are unavailable.
inline bool exl3_reconstruction_reservation_fits(std::size_t bytes) noexcept {
    constexpr std::size_t column_bytes=17408u*2u;
    return bytes==column_bytes*256u || bytes==column_bytes*512u ||
        bytes==column_bytes*1024u || bytes==column_bytes*5120u;
}

inline bool exl3_reconstruction_budget_fallback(const char* value) {
    if(!value || std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::invalid_argument("reconstruction budget fallback must be 0 or 1");
}

inline int exl3_reconstruction_slice_columns(const char* value) {
    if(!value || std::strcmp(value,"5120")==0)return 5120;
    if(std::strcmp(value,"256")==0)return 256;
    if(std::strcmp(value,"512")==0)return 512;
    if(std::strcmp(value,"1024")==0)return 1024;
    throw std::invalid_argument("reconstruction slice columns must be 256, 512, 1024 or 5120");
}

inline bool exl3_reconstruction_k6_down_option(const char* value,bool slab_enabled) {
    if(!value || std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")!=0)
        throw std::invalid_argument("exact K6 Down reconstruction must be 0 or 1");
    if(!slab_enabled)
        throw std::invalid_argument("exact K6 Down reconstruction requires the reserved exact reconstruction slab");
    return true;
}

inline bool exl3_reconstruction_k6_gate_up_option(const char* value,bool slab_enabled) {
    if(!value || std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")!=0)
        throw std::invalid_argument("exact K6 gate/up reconstruction must be 0 or 1");
    if(!slab_enabled)
        throw std::invalid_argument("exact K6 gate/up reconstruction requires the reserved exact reconstruction slab");
    return true;
}

} // namespace ninfer::exl3
