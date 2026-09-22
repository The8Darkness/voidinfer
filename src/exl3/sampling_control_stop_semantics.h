#pragma once

#include "exl3/request_sampling_state.h"
#include "exl3/sampling_window_transaction.h"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {

struct Exl3SamplingPublicationCount {
    std::size_t resident_tokens=0,visible_tokens=0;
    bool terminal=false,terminal_visible=false;
};

class Exl3SamplingControlStopSemantics {
public:
    static void commit_forced_control_after_resident(
        Exl3RequestSamplingState& state,std::size_t tokens,
        std::shared_ptr<const void> new_root_revision) {
        state.advance_forced_control(tokens,std::move(new_root_revision),true);
    }

    static void claim_model_publication(Exl3SamplingWindowTransaction& window) {
        window.claim_publication();
    }

    static Exl3SamplingPublicationCount commit_model_after_resident(
        Exl3SamplingWindowTransaction& window,std::size_t resident_tokens,
        bool terminal,bool publish_terminal,std::shared_ptr<const void> new_root_revision) {
        if(!resident_tokens || (publish_terminal && !terminal))
            throw std::invalid_argument("sampled terminal publication extent");
        window.commit(resident_tokens,std::move(new_root_revision));
        return {resident_tokens,resident_tokens-(terminal&&!publish_terminal?1u:0u),
            terminal,terminal&&publish_terminal};
    }

    static void abandon_before_correction(Exl3SamplingWindowTransaction& window) {
        window.rollback();
    }
};

} // namespace ninfer::exl3
