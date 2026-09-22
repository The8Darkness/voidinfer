#pragma once

#include "exl3/request_sampling_state.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ninfer::exl3 {

// Request-local tentative sampling window. Each prospective output owns a q
// proposal draw and an acceptance draw; the first rejection additionally owns
// one correction draw and seals the speculative window. Commit counts draws
// only through the immutable prefix actually published by the caller.
class Exl3SamplingWindowTransaction {
    struct Decision { std::uint8_t draws=0; };
public:
    Exl3SamplingWindowTransaction(Exl3RequestSamplingState& state,
        std::size_t accepted_prefix,std::size_t maximum_tokens)
        :state_(&state),accepted_prefix_(accepted_prefix),
         maximum_tokens_(maximum_tokens),
         decisions_(prepare_decisions(accepted_prefix,maximum_tokens)) {
        transaction_=state.begin(accepted_prefix);active_=true;
    }

    ~Exl3SamplingWindowTransaction() {
        if(active_ && state_ && !state_->cancelled()) {
            try {state_->rollback(transaction_);}catch(...) {}
        }
    }

    Exl3SamplingWindowTransaction(const Exl3SamplingWindowTransaction&)=delete;
    Exl3SamplingWindowTransaction& operator=(const Exl3SamplingWindowTransaction&)=delete;

    [[nodiscard]] std::uint64_t draw_proposal() {
        require_open();
        if(pending_proposal_ || pending_acceptance_ ||
           decisions_.size()>=maximum_tokens_)
            throw std::logic_error("sampling proposal draw ordering");
        pending_proposal_=true;
        return state_->draw(transaction_);
    }

    [[nodiscard]] std::uint64_t draw_acceptance() {
        require_open();
        if(!pending_proposal_ || pending_acceptance_)
            throw std::logic_error("sampling acceptance draw ordering");
        pending_acceptance_=true;
        return state_->draw(transaction_);
    }

    void accept_proposal() {
        require_pending();
        decisions_.push_back({2});pending_proposal_=false;pending_acceptance_=false;
    }

    [[nodiscard]] std::uint64_t reject_and_draw_correction() {
        require_pending();
        const auto correction=state_->draw(transaction_);
        decisions_.push_back({3});pending_proposal_=false;pending_acceptance_=false;
        sealed_=true;
        return correction;
    }

    void seal_stop_boundary() {
        require_open();
        if(pending_proposal_ || pending_acceptance_)
            throw std::logic_error("stop during incomplete sampling decision");
        sealed_=true;
    }

    void claim_publication() {
        require_active();
        if(pending_proposal_ || pending_acceptance_)
            throw std::logic_error("sampling publication has incomplete decision");
        state_->claim_publication(transaction_);publication_claimed_=true;
    }

    void commit(std::size_t published_tokens,std::shared_ptr<const void> new_root_revision) {
        require_active();
        if(!publication_claimed_ || pending_proposal_ || pending_acceptance_ ||
           published_tokens>decisions_.size())
            throw std::invalid_argument("sampling publication extent");
        std::uint64_t committed_draws=0;
        for(std::size_t i=0;i<published_tokens;++i)committed_draws+=decisions_[i].draws;
        state_->commit(transaction_,committed_draws,
            accepted_prefix_+published_tokens,std::move(new_root_revision));
        active_=false;
    }

    void rollback() {
        require_active();
        pending_proposal_=false;pending_acceptance_=false;
        state_->rollback(transaction_);active_=false;
    }

    [[nodiscard]] std::size_t decided_tokens() const noexcept{return decisions_.size();}
    [[nodiscard]] bool sealed() const noexcept{return sealed_;}

private:
    static std::vector<Decision> prepare_decisions(
        std::size_t accepted_prefix,std::size_t maximum_tokens) {
        if(!maximum_tokens || maximum_tokens>
                std::numeric_limits<std::size_t>::max()-accepted_prefix)
            throw std::invalid_argument("sampling window requires output capacity");
        std::vector<Decision> result;result.reserve(maximum_tokens);return result;
    }
    void require_active() const {
        if(!active_ || !state_)throw std::logic_error("sampling window inactive");
    }
    void require_open() const {
        require_active();
        if(sealed_)throw std::logic_error("sampling window already sealed");
    }
    void require_pending() const {
        require_open();
        if(!pending_acceptance_)throw std::logic_error("sampling acceptance draw missing");
    }

    Exl3RequestSamplingState* state_=nullptr;
    std::size_t accepted_prefix_=0,maximum_tokens_=0;
    std::vector<Decision> decisions_;
    Exl3RequestSamplingState::Transaction transaction_;
    bool active_=false,pending_proposal_=false,pending_acceptance_=false;
    bool sealed_=false,publication_claimed_=false;
};

} // namespace ninfer::exl3
