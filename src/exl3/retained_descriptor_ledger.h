#pragma once
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <optional>

namespace ninfer::exl3 {
// Registry operations serialize acquisition; a moved plan may retire elsewhere.
// Tickets keep the counter alive without retaining or dereferencing the registry.
template<class DomainTag> class RetainedByteLedger {
public:
    // An allocator may provide a reserved counter/control block. Tickets then
    // retain that backing directly, independently of the containing registry.
    struct State {std::atomic<std::uint64_t> bytes{0};};
private:
    std::shared_ptr<State> state_;
public:
    RetainedByteLedger():state_(std::make_shared<State>()) {}
    explicit RetainedByteLedger(std::shared_ptr<State> state):state_(std::move(state)) {
        if(!state_ || !state_.use_count())throw std::invalid_argument("retained ledger backing owner missing");
    }
    class Ticket {
        friend class RetainedByteLedger;
        std::shared_ptr<State> state_;
        std::uint64_t bytes_=0;
        Ticket(std::shared_ptr<State> state,std::uint64_t bytes):state_(std::move(state)),bytes_(bytes) {}
    public:
        Ticket(const Ticket&)=delete;
        Ticket& operator=(const Ticket&)=delete;
        Ticket& operator=(Ticket&&)=delete;
        Ticket(Ticket&& other) noexcept:state_(std::move(other.state_)),bytes_(std::exchange(other.bytes_,0)) {}
        ~Ticket(){if(state_)state_->bytes.fetch_sub(bytes_,std::memory_order_acq_rel);}
        std::uint64_t bytes() const noexcept {return bytes_;}
        // Transfer part of an existing charge without allocating or changing
        // the ledger total. Serialized owner retirement may split child storage
        // from a shared wrapper whose final weak reference dies separately.
        std::optional<Ticket> split(std::uint64_t bytes) noexcept {
            if(!state_ || !bytes || bytes>bytes_)return std::nullopt;
            bytes_-=bytes;return Ticket(state_,bytes);
        }
    };
    std::uint64_t bytes() const noexcept {return state_->bytes.load(std::memory_order_acquire);}
    bool owns(const Ticket& ticket) const noexcept {return ticket.state_==state_;}
    std::weak_ptr<const void> lifetime_for_test() const noexcept {return state_;}
    Ticket acquire(std::uint64_t bytes) {
        auto old=state_->bytes.load(std::memory_order_acquire);
        for(;;) {
            if(bytes>std::numeric_limits<std::uint64_t>::max()-old)
                throw std::overflow_error("retained descriptor ledger overflow");
            if(state_->bytes.compare_exchange_weak(old,old+bytes,std::memory_order_acq_rel))break;
        }
        return Ticket(state_,bytes);
    }
};
struct RetainedHostMetadataTag;
struct RetainedDeviceBytesTag;
struct RetainedCudaRegistrationTag;
struct RetainedHostAllocationTag;
using RetainedDescriptorLedger=RetainedByteLedger<RetainedHostMetadataTag>;
using RetainedDeviceLedger=RetainedByteLedger<RetainedDeviceBytesTag>;
using RetainedCudaRegistrationLedger=RetainedByteLedger<RetainedCudaRegistrationTag>;
using RetainedHostAllocationLedger=RetainedByteLedger<RetainedHostAllocationTag>;
}
