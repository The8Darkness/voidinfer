#pragma once
#include <cstdint>
#include <stdexcept>

namespace ninfer::exl3 {
// Request-local ordering authority. Private CPU preview may be prepared after
// numerical readiness, but no visible range can be queued before the resident
// state transaction commits.
class Exl3ControlPublicationBoundary {
public:
    enum class Phase : std::uint8_t {
        idle,numerical_pending,numerical_ready,output_prepared,
        publication_pending,resident_committed,exposure_staged,failed
    };
    enum class Kind : std::uint8_t { control,model };
    enum class IndependentWork : std::uint8_t {
        cancellation,deadline,capacity_preflight,private_output
    };
    struct Ticket {
        std::uint64_t generation=0,acquisition=0,execution=0;
        Kind kind=Kind::model;
    };
    struct Snapshot {
        Phase phase=Phase::idle;
        Ticket ticket{};
        std::uint64_t independent_observations=0;
    };

    Ticket begin(std::uint64_t acquisition,std::uint64_t execution,Kind kind) {
        if(phase_!=Phase::idle || !acquisition || !execution || generation_==UINT64_MAX)
            throw std::logic_error("publication numerical boundary unavailable");
        current_={++generation_,acquisition,execution,kind};
        phase_=Phase::numerical_pending;independent_observations_=0;return current_;
    }
    bool observe_independent(Ticket ticket,IndependentWork work) noexcept {
        if(!matches(ticket) || phase_==Phase::failed)return false;
        const bool allowed=work!=IndependentWork::private_output ||
            phase_==Phase::numerical_ready || phase_==Phase::output_prepared;
        if(!allowed)return false;
        ++independent_observations_;return true;
    }
    bool numerical_ready(Ticket ticket) noexcept {
        if(!matches(ticket) || phase_!=Phase::numerical_pending)return false;
        phase_=Phase::numerical_ready;return true;
    }
    bool prepare_output(Ticket ticket) noexcept {
        if(!observe_independent(ticket,IndependentWork::private_output))return false;
        phase_=Phase::output_prepared;return true;
    }
    bool resume_numerical(Ticket ticket) noexcept {
        if(!matches(ticket) || (phase_!=Phase::numerical_ready &&
            phase_!=Phase::output_prepared))return false;
        phase_=Phase::numerical_pending;return true;
    }
    bool begin_publication(Ticket ticket) noexcept {
        if(!matches(ticket) || (phase_!=Phase::numerical_ready &&
            phase_!=Phase::output_prepared))return false;
        phase_=Phase::publication_pending;return true;
    }
    bool resident_committed(Ticket ticket) noexcept {
        if(!matches(ticket) || phase_!=Phase::publication_pending)return false;
        phase_=Phase::resident_committed;return true;
    }
    bool stage_exposure(Ticket ticket) noexcept {
        if(!matches(ticket) || phase_!=Phase::resident_committed)return false;
        phase_=Phase::exposure_staged;return true;
    }
    bool finish_exposure(Ticket ticket) noexcept {
        if(!matches(ticket) || phase_!=Phase::exposure_staged)return false;
        phase_=Phase::idle;current_={};return true;
    }
    bool abandon_private(Ticket ticket) noexcept {
        if(!matches(ticket) || phase_==Phase::publication_pending ||
           phase_==Phase::resident_committed || phase_==Phase::exposure_staged ||
           phase_==Phase::failed)return false;
        phase_=Phase::idle;current_={};return true;
    }
    void fail(Ticket ticket) noexcept {
        if(matches(ticket))phase_=Phase::failed;
    }
    void fail_active() noexcept {if(phase_!=Phase::idle)phase_=Phase::failed;}
    bool reset_after_request() noexcept {
        if(phase_!=Phase::idle && phase_!=Phase::failed)return false;
        phase_=Phase::idle;current_={};return true;
    }
    Snapshot snapshot() const noexcept {
        return {phase_,current_,independent_observations_};
    }
private:
    bool matches(Ticket ticket) const noexcept {
        return ticket.generation && ticket.generation==current_.generation &&
            ticket.acquisition==current_.acquisition &&
            ticket.execution==current_.execution && ticket.kind==current_.kind;
    }
    Phase phase_=Phase::idle;
    Ticket current_{};
    std::uint64_t generation_=0,independent_observations_=0;
};
}
