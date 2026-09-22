#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::exl3 {
enum class Exl3PairedTransformAdmission {
    admitted, disabled, invalid_rows, timing, observer, preserved_multirow
};
// Explain the actual GDN route exclusion without inspecting device data or
// changing the arithmetic route merely because equal-scale reuse is requested.
inline Exl3PairedTransformAdmission exl3_paired_transform_admission(
    bool enabled,bool timing,bool observer,bool preserve_m1,int rows,bool wide_prefill) noexcept {
    if(!enabled)return Exl3PairedTransformAdmission::disabled;
    if(rows<=0)return Exl3PairedTransformAdmission::invalid_rows;
    if(timing)return Exl3PairedTransformAdmission::timing;
    if(observer)return Exl3PairedTransformAdmission::observer;
    if(preserve_m1 && rows!=1 && !wide_prefill)
        return Exl3PairedTransformAdmission::preserved_multirow;
    return Exl3PairedTransformAdmission::admitted;
}
inline bool exl3_equal_transform_reuse_enabled(const char* value) {
    if(!value || std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::invalid_argument("equal input transform reuse must be 0 or 1");
}
// Input, first/second scale, first/second output. Read-only aliases are legal;
// output writes cannot overlap any other live consumer extent.
inline void exl3_require_paired_transform_extents(int rows,int features,
    const std::array<std::uintptr_t,5>& addresses) {
    if(rows<=0 || features<=0 || features%128 || rows>std::numeric_limits<int>::max()/features)
        throw std::invalid_argument("paired transform indexing geometry");
    const auto row_bytes=std::size_t(rows)*features*2;
    const auto scale_bytes=std::size_t(features)*2;
    const std::array<std::size_t,5> sizes{row_bytes,scale_bytes,scale_bytes,row_bytes,row_bytes};
    for(std::size_t i=0;i<5;++i)
        if(!addresses[i] || addresses[i]>std::numeric_limits<std::uintptr_t>::max()-sizes[i])
            throw std::invalid_argument("paired transform address extent");
    for(std::size_t output=3;output<5;++output)
        for(std::size_t other=0;other<output;++other)
            if(addresses[output]<addresses[other]+sizes[other] &&
               addresses[other]<addresses[output]+sizes[output])
                throw std::invalid_argument("paired transform overlapping live buffers");
}
} // namespace ninfer::exl3
