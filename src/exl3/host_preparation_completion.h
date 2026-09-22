#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <mutex>

namespace ninfer::exl3 {

// Request-local completion authority for host work which is allowed to run
// while another lane owns pending numerical state.  The prepared value is
// deliberately only a plan: accepting it does not imply cache admission,
// physical-lane acquisition, or numerical completion.
class Exl3HostPreparationCompletion {
public:
    enum class Phase : std::uint8_t {
        idle,
        pending,
        ready,
        accepted,
        cancelled,
        failed
    };
    struct Ticket {
        std::uint64_t generation=0;
        friend bool operator==(Ticket,Ticket)=default;
    };
    struct Plan {
        std::uint64_t input_fingerprint=0;
        std::size_t input_tokens=0;
        std::size_t cacheable_tokens=0;
        bool declared_frontier=false;
    };
    struct Snapshot {
        Phase phase=Phase::idle;
        std::uint64_t generation=0;
        bool has_plan=false;
    };

    std::optional<Ticket> begin(std::uint64_t generation) {
        std::lock_guard lock(mutex_);
        if(!generation || generation<=generation_ ||
           (phase_!=Phase::idle && phase_!=Phase::accepted &&
            phase_!=Phase::cancelled && phase_!=Phase::failed))
            return std::nullopt;
        generation_=generation;phase_=Phase::pending;plan_.reset();
        return Ticket{generation};
    }
    bool complete(Ticket ticket,Plan plan) {
        std::lock_guard lock(mutex_);
        if(!current(ticket) || phase_!=Phase::pending ||
           !plan.input_fingerprint || plan.cacheable_tokens>plan.input_tokens)
            return false;
        plan_=plan;phase_=Phase::ready;return true;
    }
    std::optional<Plan> accept(Ticket ticket,
        std::uint64_t input_fingerprint,std::size_t input_tokens) {
        std::lock_guard lock(mutex_);
        if(!current(ticket) || phase_!=Phase::ready || !plan_ ||
           plan_->input_fingerprint!=input_fingerprint ||
           plan_->input_tokens!=input_tokens)
            return std::nullopt;
        phase_=Phase::accepted;return plan_;
    }
    bool cancel(std::uint64_t generation) {
        std::lock_guard lock(mutex_);
        if(!generation)return false;
        if(generation<generation_)return false;
        if(generation>generation_) {
            if(phase_==Phase::pending || phase_==Phase::ready)return false;
            generation_=generation;plan_.reset();
        }
        if(phase_==Phase::accepted || phase_==Phase::failed ||
           phase_==Phase::cancelled)return false;
        phase_=Phase::cancelled;plan_.reset();return true;
    }
    bool fail(Ticket ticket) {
        std::lock_guard lock(mutex_);
        if(!current(ticket) || (phase_!=Phase::pending && phase_!=Phase::ready))
            return false;
        phase_=Phase::failed;plan_.reset();return true;
    }
    Snapshot snapshot() const {
        std::lock_guard lock(mutex_);
        return {phase_,generation_,plan_.has_value()};
    }
private:
    bool current(Ticket ticket) const noexcept {
        return ticket.generation && ticket.generation==generation_;
    }
    Phase phase_=Phase::idle;
    std::uint64_t generation_=0;
    std::optional<Plan> plan_;
    mutable std::mutex mutex_;
};

} // namespace ninfer::exl3
