#pragma once
#include "exl3/vericache_request.h"
#include <optional>

namespace ninfer::exl3 {
// One physical execution lane, bounded independent active requests. A lease
// covers tentative work through repair/publication; cancellation invalidates
// publication without pretending already issued GPU work was free.
class Exl3VeriCacheQueue {
public:
    enum class Status {ready,cancelled,completed};
    struct Lease {
        std::size_t request;
        std::uint64_t generation;
        std::shared_ptr<const Exl3VeriCacheRequest> root;
    };
private:
    struct Slot {
        std::shared_ptr<const Exl3VeriCacheRequest> state;
        std::uint64_t generation=0;
        Status status=Status::ready;
    };
    std::size_t capacity_,cursor_=0;
    std::vector<Slot> slots_;
    std::optional<Lease> busy_;
    void check_lease(const Lease& lease) const {
        if(!busy_ || busy_->request!=lease.request || busy_->generation!=lease.generation || busy_->root!=lease.root)
            throw std::invalid_argument("foreign or expired execution lease");
    }
public:
    explicit Exl3VeriCacheQueue(std::size_t capacity):capacity_(capacity) {
        if(capacity<1 || capacity>64) throw std::invalid_argument("request queue capacity1..64");
        slots_.reserve(capacity);
    }
    std::size_t add(std::shared_ptr<const Exl3VeriCacheRequest> state) {
        if(!state || slots_.size()==capacity_) throw std::invalid_argument("request admission capacity");
        slots_.push_back(Slot{std::move(state)});return slots_.size()-1;
    }
    std::optional<Lease> acquire() {
        if(busy_) throw std::logic_error("physical execution lane is already owned");
        for(std::size_t i=0;i<slots_.size();++i) {
            const auto id=cursor_;cursor_=(cursor_+1)%slots_.size();
            auto& slot=slots_[id];
            if(slot.status==Status::ready) {busy_=Lease{id,slot.generation,slot.state};return busy_;}
        }
        return std::nullopt;
    }
    void cancel(std::size_t id) {
        auto& slot=slots_.at(id);
        if(slot.status==Status::ready) {slot.status=Status::cancelled;++slot.generation;}
    }
    bool publish(const Lease& lease,std::shared_ptr<const Exl3VeriCacheRequest> updated,bool finished=false) {
        check_lease(lease);
        auto& slot=slots_[lease.request];
        if(slot.status!=Status::ready || slot.generation!=lease.generation) {busy_.reset();return false;}
        if(slot.state!=lease.root || !updated || !updated->is_child_of(*lease.root))
            throw std::invalid_argument("request result has a different authoritative parent");
        slot.state=std::move(updated);++slot.generation;
        if(finished) slot.status=Status::completed;
        busy_.reset();return true;
    }
    void release(const Lease& lease) {check_lease(lease);busy_.reset();}
    bool active(std::size_t id) const {return slots_.at(id).status==Status::ready;}
    Status status(std::size_t id) const {return slots_.at(id).status;}
    const std::shared_ptr<const Exl3VeriCacheRequest>& state(std::size_t id) const {return slots_.at(id).state;}
    std::size_t size() const noexcept {return slots_.size();}
};
} // namespace ninfer::exl3
