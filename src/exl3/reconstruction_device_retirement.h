#pragma once

namespace ninfer::exl3 {
struct Exl3ReconstructionDrainResult {
    int error=0;
    int restore_error=0;
    bool released=false;
};

// Providers and release must not throw. Release executes on the owner's device,
// only after its drain succeeds. Caller retains the complete owner otherwise.
template<class Query,class Select,class Drain,class Release>
Exl3ReconstructionDrainResult exl3_retire_reconstruction_device(
    int owner_device,Query query,Select select,Drain drain,Release release,
    bool inject_failure=false) noexcept {
    Exl3ReconstructionDrainResult result;
    if(owner_device<0){result.error=-1;return result;}
    int previous=-1;
    result.error=query(&previous);
    if(result.error)return result;
    if(previous<0){result.error=-1;return result;}
    const bool change=previous!=owner_device;
    if(change)result.error=select(owner_device);
    if(!result.error)result.error=drain();
    if(!result.error && inject_failure)result.error=-2;
    if(!result.error){release();result.released=true;}
    if(change)result.restore_error=select(previous);
    return result;
}
} // namespace ninfer::exl3
