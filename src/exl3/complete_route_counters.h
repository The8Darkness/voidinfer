#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::exl3 {

struct Exl3CompleteRouteCounterSnapshot {
    std::uint64_t proposed_rows=0;
    std::uint64_t verified_rows=0;
    std::uint64_t replayed_rows=0;
    std::uint64_t committed_model_rows=0;
    std::uint64_t externally_visible_model_rows=0;
    std::uint64_t hidden_terminal_rows=0;
    std::uint64_t diagnostic_rows=0;
};

class Exl3CompleteRouteCounters {
public:
    void record_attempt(std::uint64_t proposed,std::uint64_t verified,
        std::uint64_t replayed) {
        add(value_.proposed_rows,proposed);
        add(value_.verified_rows,verified);
        add(value_.replayed_rows,replayed);
    }
    // Target work can be the first verification or an additional repair. The
    // two row classes stay separate even when one caller reports both.
    void record_target_work(std::uint64_t verified,std::uint64_t replayed) {
        add(value_.verified_rows,verified);add(value_.replayed_rows,replayed);
    }
    void record_publication(std::uint64_t committed,std::uint64_t visible,
        std::uint64_t hidden_terminal) {
        if(visible>committed || hidden_terminal>committed-visible ||
           visible+hidden_terminal!=committed)
            throw std::invalid_argument("EXL3 complete-route publication partition");
        add(value_.committed_model_rows,committed);
        add(value_.externally_visible_model_rows,visible);
        add(value_.hidden_terminal_rows,hidden_terminal);
    }
    void record_diagnostic(std::uint64_t rows) {add(value_.diagnostic_rows,rows);}
    [[nodiscard]] const Exl3CompleteRouteCounterSnapshot& snapshot() const noexcept {
        return value_;
    }
private:
    static void add(std::uint64_t& destination,std::uint64_t value) {
        if(value>std::numeric_limits<std::uint64_t>::max()-destination)
            throw std::overflow_error("EXL3 complete-route counter overflow");
        destination+=value;
    }
    Exl3CompleteRouteCounterSnapshot value_;
};

}
