#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace ninfer::exl3 {

enum class Exl3SamplingReproducibility : std::uint8_t {
    // Explicit seeds, including the API's deterministic default zero, replay
    // from request identity across a clean model reload. Engine/acquisition
    // identity remains an ownership gate but is not mixed into random values.
    seeded_request_replay_across_reload
};

class Exl3RequestSamplingState {
public:
    struct Transaction {
        const Exl3RequestSamplingState* owner=nullptr;
        std::uint64_t nonce=0,acquisition=0,base_counter=0,tentative_counter=0;
        std::size_t accepted_prefix=0;
        std::shared_ptr<const void> root_revision;
        bool publication_claimed=false;
    };

    Exl3RequestSamplingState(std::uint64_t engine_epoch,std::uint64_t request_generation,
        std::uint64_t input_fingerprint,std::uint64_t seed,
        Exl3SamplingReproducibility reproducibility=
            Exl3SamplingReproducibility::seeded_request_replay_across_reload)
        :engine_epoch_(engine_epoch),request_generation_(request_generation),
         input_fingerprint_(input_fingerprint),seed_(seed),reproducibility_(reproducibility) {
        if(!engine_epoch_ || !request_generation_ || !input_fingerprint_)
            throw std::invalid_argument("request sampling identity");
    }

    void bind_acquisition(std::uint64_t acquisition,
        std::shared_ptr<const void> root_revision,std::size_t accepted_prefix) {
        if(cancelled_.load(std::memory_order_acquire) || !acquisition || !root_revision ||
           !root_revision.use_count() || active_nonce_)
            throw std::invalid_argument("sampling acquisition binding");
        acquisition_=acquisition;root_revision_=std::move(root_revision);
        committed_prefix_=accepted_prefix;
    }

    [[nodiscard]] Transaction begin(std::size_t accepted_prefix) {
        if(cancelled_.load(std::memory_order_acquire) || !acquisition_ ||
           !root_revision_ || active_nonce_ || next_nonce_==0 ||
           accepted_prefix!=committed_prefix_)
            throw std::logic_error("sampling transaction unavailable");
        active_nonce_=next_nonce_++;
        return {this,active_nonce_,acquisition_,committed_counter_,committed_counter_,
            accepted_prefix,root_revision_};
    }

    [[nodiscard]] std::uint64_t draw(Transaction& transaction) const {
        require_current(transaction);
        if(transaction.tentative_counter==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("sampling counter exhausted");
        return splitmix(seed_^input_fingerprint_^request_generation_^
            transaction.tentative_counter++);
    }

    void claim_publication(Transaction& transaction) {
        require_current(transaction);
        if(transaction.publication_claimed)
            throw std::logic_error("sampling publication already claimed");
        transaction.publication_claimed=true;
    }

    void commit(Transaction& transaction,std::uint64_t draws_to_commit,
        std::size_t accepted_prefix,std::shared_ptr<const void> new_root_revision) {
        require_current(transaction,true);
        const auto tentative=transaction.tentative_counter-transaction.base_counter;
        if(!transaction.publication_claimed || draws_to_commit>tentative ||
           accepted_prefix<transaction.accepted_prefix ||
           !new_root_revision || !new_root_revision.use_count() ||
           draws_to_commit>std::numeric_limits<std::uint64_t>::max()-committed_counter_)
            throw std::invalid_argument("sampling transaction commit extent");
        committed_counter_+=draws_to_commit;committed_prefix_=accepted_prefix;
        root_revision_=std::move(new_root_revision);
        finish(transaction);
    }

    void rollback(Transaction& transaction) {
        require_current(transaction,transaction.publication_claimed);finish(transaction);
    }

    void advance_forced_control(std::size_t tokens,
        std::shared_ptr<const void> new_root_revision,bool resident_committed) {
        if(!resident_committed || !tokens || !acquisition_ || active_nonce_ ||
           !new_root_revision || !new_root_revision.use_count() ||
           tokens>std::numeric_limits<std::size_t>::max()-committed_prefix_)
            throw std::invalid_argument("forced control sampling-state advancement");
        committed_prefix_+=tokens;root_revision_=std::move(new_root_revision);
    }

    void cancel() noexcept {
        cancelled_.store(true,std::memory_order_release);
    }

    [[nodiscard]] std::uint64_t engine_epoch() const noexcept{return engine_epoch_;}
    [[nodiscard]] std::uint64_t request_generation() const noexcept{return request_generation_;}
    [[nodiscard]] std::uint64_t committed_draws() const noexcept{return committed_counter_;}
    [[nodiscard]] std::size_t committed_prefix() const noexcept{return committed_prefix_;}
    [[nodiscard]] Exl3SamplingReproducibility reproducibility() const noexcept{return reproducibility_;}
    [[nodiscard]] bool cancelled() const noexcept{return cancelled_.load(std::memory_order_acquire);}

private:
    static std::uint64_t splitmix(std::uint64_t value) noexcept {
        value+=0x9e3779b97f4a7c15ULL;
        value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
        value=(value^(value>>27))*0x94d049bb133111ebULL;
        return value^(value>>31);
    }
    void require_current(const Transaction& transaction,
        bool allow_claimed_cancellation=false) const {
        if((cancelled_.load(std::memory_order_acquire) &&
                !(allow_claimed_cancellation && transaction.publication_claimed)) ||
           transaction.owner!=this ||
           !active_nonce_ || transaction.nonce!=active_nonce_ ||
           transaction.acquisition!=acquisition_ ||
           transaction.root_revision.get()!=root_revision_.get() ||
           transaction.root_revision.owner_before(root_revision_) ||
           root_revision_.owner_before(transaction.root_revision))
            throw std::logic_error("stale or foreign sampling transaction");
    }
    void finish(Transaction& transaction) noexcept {
        active_nonce_=0;transaction.owner=nullptr;transaction.nonce=0;
        transaction.publication_claimed=false;
        transaction.root_revision.reset();
    }

    std::uint64_t engine_epoch_=0,request_generation_=0,input_fingerprint_=0,seed_=0;
    Exl3SamplingReproducibility reproducibility_;
    std::uint64_t acquisition_=0,committed_counter_=0,next_nonce_=1,active_nonce_=0;
    std::size_t committed_prefix_=0;
    std::shared_ptr<const void> root_revision_;
    std::atomic<bool> cancelled_{false};
};

} // namespace ninfer::exl3
