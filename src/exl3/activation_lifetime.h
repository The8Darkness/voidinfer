#pragma once
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::exl3 {
// Embedded in a retained private layer, never allocated per projection. Odd
// generations identify one immutable normalized activation; even ones are dead.
// Readers must retain the owning lane before consulting this witness.
class Exl3ActivationLifetime {
    std::atomic<std::uint64_t> generation_{0};
public:
    struct Witness {
        const Exl3ActivationLifetime* lifetime=nullptr;
        std::uint64_t generation=0;
        bool current() const noexcept {
            return lifetime && (generation&1) &&
                lifetime->generation_.load(std::memory_order_acquire)==generation;
        }
    };
    class Scope {
        Exl3ActivationLifetime& lifetime_;
        std::uint64_t generation_;
    public:
        explicit Scope(Exl3ActivationLifetime& lifetime):lifetime_(lifetime) {
            auto previous=lifetime_.generation_.load(std::memory_order_acquire);
            if((previous&1) || previous>=std::numeric_limits<std::uint64_t>::max()-1)
                throw std::logic_error("activation lifetime busy/exhausted");
            generation_=previous+1;
            if(!lifetime_.generation_.compare_exchange_strong(previous,generation_,std::memory_order_acq_rel))
                throw std::logic_error("activation lifetime concurrent producer");
        }
        Scope(const Scope&)=delete;
        Scope& operator=(const Scope&)=delete;
        ~Scope(){lifetime_.generation_.store(generation_+1,std::memory_order_release);}
        Witness witness() const noexcept {return {&lifetime_,generation_};}
    };
};
}
