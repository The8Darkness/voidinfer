#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::exl3 {
enum class Exl3RetirementPhase { active, draining, retired, quarantined };
struct Exl3RetirementResult {
    Exl3RetirementPhase phase=Exl3RetirementPhase::active;
    int first_drain_error=0;
    bool prior_execution_failure=false;
    // Engine fills this from still-owned nondefault stream handles. A retired
    // drain permits owner teardown; it is not proof that deferred destruction
    // (or every other resource cleanup) has completed.
    std::uint32_t pending_stream_destructions=0;
    int first_cleanup_error=0;
    bool reusable() const noexcept {return phase==Exl3RetirementPhase::retired;}
};
// Sticky outcome; an uncertain drain cannot be overwritten by a later success.
// Actual ownership remains in the Engine bundle, not in this value object.
class Exl3RetirementState {
public:
    bool begin(bool prior_failure) noexcept {
        if(result_.phase!=Exl3RetirementPhase::active)return false;
        result_.phase=Exl3RetirementPhase::draining;
        result_.prior_execution_failure=prior_failure;return true;
    }
    void complete(int error) noexcept {
        if(result_.phase!=Exl3RetirementPhase::draining)return;
        if(error){result_.first_drain_error=error;result_.phase=Exl3RetirementPhase::quarantined;}
        else result_.phase=Exl3RetirementPhase::retired;
    }
    Exl3RetirementResult result() const noexcept {return result_;}
    void observe_cleanup_failure(int error) noexcept {
        if(!error)return;
        if(!result_.first_cleanup_error)result_.first_cleanup_error=error;
        result_.phase=Exl3RetirementPhase::quarantined;
    }
private:
    Exl3RetirementResult result_;
};
// Value witness for an event owned by a retained physical lane. A recycled
// event address alone is insufficient; every record has a distinct generation.
class Exl3FinalUseWitness {
public:
    struct Snapshot {
        std::uint64_t acquisition=0,execution=0,generation=0;
        std::uintptr_t event=0;
        bool ready=false;
        int first_error=0;
    };
    Snapshot snapshot() const noexcept {return {acquisition_,execution_,generation_,event_,ready_,error_};}
    std::uint64_t begin(std::uint64_t acquisition,std::uint64_t execution,
        std::uintptr_t event) {
        if(error_ || (generation_ && !ready_) || !acquisition || !execution || !event || generation_==UINT64_MAX)
            throw std::logic_error("final-use owner/scope unavailable");
        acquisition_=acquisition;execution_=execution;event_=event;ready_=false;
        return ++generation_;
    }
    bool finish(std::uint64_t generation,int error) noexcept {
        if(generation!=generation_ || !generation || ready_ || error_)return false;
        if(error)error_=error;else ready_=true;
        return true;
    }
    // A planned consumer can be abandoned after a different, earlier producer
    // event has been positively witnessed. The owner coordinating both events
    // supplies that exact alternate identity; retain the same generation and
    // scope while recording which event actually established final use.
    bool finish_after_alternate_event(std::uint64_t generation,
        std::uintptr_t witnessed_event,int error) noexcept {
        if(generation!=generation_ || !generation || !witnessed_event || ready_ || error_)return false;
        event_=witnessed_event;
        if(error)error_=error;else ready_=true;
        return true;
    }
    bool matches(std::uint64_t acquisition,std::uint64_t execution,std::uint64_t generation) const noexcept {
        return ready_ && !error_ && acquisition==acquisition_ && execution==execution_ && generation==generation_;
    }
    std::uint64_t generation() const noexcept {return generation_;}
    int first_error() const noexcept {return error_;}
private:
    std::uint64_t acquisition_=0,execution_=0,generation_=0;
    std::uintptr_t event_=0;
    bool ready_=false;
    int error_=0;
};
}
