#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <optional>
#include "exl3/retained_descriptor_ledger.h"

namespace ninfer::exl3 {
class Exl3ResourceReservationExhausted : public std::runtime_error {
public:
    Exl3ResourceReservationExhausted():std::runtime_error("resource allocation reservation exhausted"){}
    Exl3ResourceReservationExhausted(const char* file,int line):std::runtime_error(
        std::string("resource allocation reservation exhausted at ")+file+":"+std::to_string(line)){}
};
// Payload declarations supplied by allocation owners, never sampled device
// memory. Windows page locking is still exclusively owned by HostResidentSet.
class Exl3ResourceInventory {
public:
    enum class Domain : unsigned {
        device,cuda_registered_host,host_metadata,graph_count,event_count,count
    };
    using Totals=std::array<std::uint64_t,static_cast<unsigned>(Domain::count)>;
    struct Requirement {
        Totals units{};
        // Configuration identity is selected before invoking an allocator.
        std::uint64_t configuration=0;
        void add(Domain domain,std::uint64_t count,std::uint64_t bytes) {
            if(domain>=Domain::count || !count || !bytes)
                throw std::invalid_argument("resource requirement unknown extent");
            auto& total=units[static_cast<unsigned>(domain)];
            if(count>(std::numeric_limits<std::uint64_t>::max()-total)/bytes)
                throw std::overflow_error("resource requirement overflow");
            total+=count*bytes;
        }
    };
    struct Allocation {
        std::shared_ptr<const void> owner;
        // One owner may contain several disjoint allocations. A slot identifies
        // an immutable allocation group within that owner's lifetime.
        std::uint64_t slot=0;
        Domain domain=Domain::device;
        std::uint64_t units=0;
        std::weak_ptr<const void> retired_owner;
        using LifetimeRetire=bool(*)(const std::shared_ptr<const void>&,RetainedDescriptorLedger::Ticket) noexcept;
        LifetimeRetire lifetime_retire=nullptr;
        using DeviceLifetimeRetire=bool(*)(const std::shared_ptr<const void>&,RetainedDeviceLedger::Ticket) noexcept;
        DeviceLifetimeRetire device_lifetime_retire=nullptr;
        using RegistrationLifetimeRetire=bool(*)(const std::shared_ptr<const void>&,RetainedCudaRegistrationLedger::Ticket) noexcept;
        RegistrationLifetimeRetire registration_lifetime_retire=nullptr;
        // A tracking allocation can use a registry-owned slot while preserving
        // the allocation slot of the retired physical owner for transition
        // deduplication. Zero is also a valid physical slot, so this field is
        // consulted only when retired_owner is present.
        std::uint64_t retired_slot=0;
    };
    enum class AttributionReason : std::uint8_t {
        live_owner,
        retired_owner_pending
    };
    struct Attribution {
        std::weak_ptr<const void> owner;
        const void* owner_address=nullptr;
        std::uint64_t slot=0;
        Domain domain=Domain::device;
        std::uint64_t units=0;
        AttributionReason reason=AttributionReason::live_owner;
        std::weak_ptr<const void> retired_owner;
    };
    void add(Allocation value,
        std::source_location location=std::source_location::current()) {
        if(!value.owner || !value.owner.use_count())
            throw std::invalid_argument("resource allocation declaration missing owner");
        if(!value.units)
            throw std::invalid_argument("resource allocation declaration zero extent slot="+
                std::to_string(value.slot)+" domain="+
                std::to_string(static_cast<unsigned>(value.domain))+" at "+
                location.file_name()+":"+std::to_string(location.line()));
        if(value.domain>=Domain::count)
            throw std::invalid_argument("resource allocation declaration invalid domain");
        if(value.lifetime_retire && value.domain!=Domain::host_metadata)
            throw std::invalid_argument("resource allocation declaration metadata retire domain");
        if(value.device_lifetime_retire && value.domain!=Domain::device)
            throw std::invalid_argument("resource allocation declaration device retire domain");
        if(value.registration_lifetime_retire && value.domain!=Domain::cuda_registered_host)
            throw std::invalid_argument("resource allocation declaration registration retire domain");
        for(const auto& prior:allocations_) if(same_owner(prior.owner,value.owner) && prior.slot==value.slot) {
            if(prior.domain!=value.domain || prior.units!=value.units || prior.lifetime_retire!=value.lifetime_retire ||
                prior.device_lifetime_retire!=value.device_lifetime_retire ||
                prior.registration_lifetime_retire!=value.registration_lifetime_retire ||
                prior.retired_slot!=value.retired_slot ||
                prior.retired_owner.owner_before(value.retired_owner) || value.retired_owner.owner_before(prior.retired_owner))
                throw std::invalid_argument("resource owner slot changed extent/domain");
            return;
        }
        allocations_.push_back(std::move(value));
    }
    void append(const Exl3ResourceInventory& other) {for(const auto& value:other.allocations_)add(value);}
    std::optional<Allocation> first_lifetime_retirement() const {
        for(const auto& value:allocations_)if(value.lifetime_retire || value.device_lifetime_retire || value.registration_lifetime_retire)return value;
        return {};
    }
    // Prepare a removal without changing authority or claiming physical release.
    // Exact owner/control block, slot, domain and extent must all still match.
    Exl3ResourceInventory without_allocation(const Allocation& expected) const {
        if(!expected.owner || !expected.units || expected.domain>=Domain::count)
            throw std::invalid_argument("resource retirement declaration");
        Exl3ResourceInventory result;bool found=false;
        for(const auto& value:allocations_) {
            if(same_owner(value.owner,expected.owner) && value.slot==expected.slot) {
                if(value.domain!=expected.domain || value.units!=expected.units || value.lifetime_retire!=expected.lifetime_retire ||
                    value.device_lifetime_retire!=expected.device_lifetime_retire ||
                    value.registration_lifetime_retire!=expected.registration_lifetime_retire)
                    throw std::invalid_argument("resource retirement extent/domain mismatch");
                found=true;
            } else result.allocations_.push_back(value);
        }
        if(!found)throw std::invalid_argument("resource retirement allocation missing");
        return result;
    }
    // Transfer an unchanged retired metadata charge to a tracking owner. The
    // tracker must retain a weak reference until the original object expires;
    // this operation itself grants no new allocation credit.
    Exl3ResourceInventory transfer_metadata(const Allocation& expected,const Allocation& tracking) const {
        if(expected.domain!=Domain::host_metadata || tracking.domain!=Domain::host_metadata ||
            expected.units!=tracking.units || !tracking.owner)
            throw std::invalid_argument("metadata tracking transfer extent/domain");
        for(const auto& value:allocations_)if(same_owner(value.owner,expected.owner) &&
            value.domain!=Domain::host_metadata)
            throw std::invalid_argument("metadata transfer precedes physical retirement");
        const std::weak_ptr<const void> empty;
        for(const auto& value:allocations_)if(same_owner(value.owner,expected.owner) && value.slot==expected.slot &&
            (value.retired_owner.owner_before(empty) || empty.owner_before(value.retired_owner)))
            throw std::invalid_argument("metadata tracking identity cannot be rebound");
        auto next=without_allocation(expected);
        auto bound=tracking;bound.retired_owner=expected.owner;bound.retired_slot=expected.slot;
        next.add(std::move(bound));
        if(next.totals()!=totals())throw std::invalid_argument("metadata transfer merged an existing charge");
        return next;
    }
    bool metadata_owner_expired(const Allocation& tracking,const std::weak_ptr<const void>& retired) const {
        const std::weak_ptr<const void> empty;
        for(const auto& value:allocations_)if(same_owner(value.owner,tracking.owner) && value.slot==tracking.slot) {
            if(value.domain!=Domain::host_metadata || tracking.domain!=value.domain || tracking.units!=value.units ||
                (!value.retired_owner.owner_before(empty) && !empty.owner_before(value.retired_owner)) ||
                value.retired_owner.owner_before(retired) || retired.owner_before(value.retired_owner))
                throw std::invalid_argument("metadata collection retired owner mismatch");
            return value.retired_owner.expired();
        }
        throw std::invalid_argument("metadata collection tracking allocation missing");
    }
    Totals totals() const {
        Totals result{};
        for(const auto& value:allocations_) {
            auto& sum=result[static_cast<unsigned>(value.domain)];
            if(value.units>std::numeric_limits<std::uint64_t>::max()-sum)
                throw std::overflow_error("resource inventory extent overflow");
            sum+=value.units;
        }
        return result;
    }
    std::vector<Attribution> attribution_snapshot() const {
        std::vector<Attribution> result;result.reserve(allocations_.size());
        const std::weak_ptr<const void> empty;
        for(const auto& value:allocations_) {
            const bool retired=value.retired_owner.owner_before(empty) ||
                empty.owner_before(value.retired_owner);
            result.push_back({value.owner,value.owner.get(),value.slot,value.domain,
                value.units,retired?AttributionReason::retired_owner_pending:
                    AttributionReason::live_owner,value.retired_owner});
        }
        return result;
    }
    // Transition accounting follows the physical allocation identity. A
    // registry-owned tracking record therefore aliases the still-live retired
    // owner/slot instead of being charged a second time. If that weak owner has
    // expired, the tracking allocation remains charged under its own identity
    // until the caller explicitly removes it.
    static Totals transition_union(const Exl3ResourceInventory& old,
        const Exl3ResourceInventory& next,const Exl3ResourceInventory& pending={},
        const Exl3ResourceInventory& quarantined={}) {
        struct Charge {
            std::shared_ptr<const void> owner;
            std::uint64_t slot=0;
            Domain domain=Domain::device;
            std::uint64_t units=0;
        };
        std::vector<Charge> charges;
        Totals result{};
        const std::weak_ptr<const void> empty;
        const auto append=[&](const Exl3ResourceInventory& inventory) {
            for(const auto& value:inventory.allocations_) {
                const bool tracked=value.retired_owner.owner_before(empty) ||
                    empty.owner_before(value.retired_owner);
                auto physical=tracked?value.retired_owner.lock():std::shared_ptr<const void>{};
                const auto slot=physical?value.retired_slot:value.slot;
                if(!physical)physical=value.owner;
                auto found=std::find_if(charges.begin(),charges.end(),[&](const Charge& prior) {
                    return same_owner(prior.owner,physical) && prior.slot==slot;
                });
                if(found!=charges.end()) {
                    if(found->domain!=value.domain || found->units!=value.units)
                        throw std::invalid_argument("resource transition alias changed extent/domain");
                    continue;
                }
                auto& total=result[static_cast<unsigned>(value.domain)];
                if(value.units>std::numeric_limits<std::uint64_t>::max()-total)
                    throw std::overflow_error("resource transition extent overflow");
                total+=value.units;
                charges.push_back({std::move(physical),slot,value.domain,value.units});
            }
        };
        append(old);append(next);append(pending);append(quarantined);
        return result;
    }
    static Totals peak(const Exl3ResourceInventory& old,const Exl3ResourceInventory& next,
        const Totals& limits) {
        return peak(old,next,{}, {},limits);
    }
    static Totals peak(const Exl3ResourceInventory& old,const Exl3ResourceInventory& next,
        const Exl3ResourceInventory& pending,const Exl3ResourceInventory& quarantined,
        const Totals& limits) {
        const auto result=transition_union(old,next,pending,quarantined);
        for(unsigned i=0;i<result.size();++i)if(result[i]>limits[i])
            throw std::runtime_error("resource transition budget exhausted");
        return result;
    }
    static Totals unlimited() {Totals result;result.fill(std::numeric_limits<std::uint64_t>::max());return result;}
    // Supplied fresh observation; no device query here. Existing owned bytes
    // are already excluded from available. Unknown external/driver allocations
    // must not become allocation credit merely because total capacity is large.
    static std::uint64_t device_limit(std::uint64_t owned,std::uint64_t total,
        std::uint64_t available,std::uint64_t reserve) {
        if(!total || available>total || owned>total-available || reserve>available)
            throw std::invalid_argument("device availability/reserve contradiction");
        return owned+(available-reserve);
    }
private:
    static bool same_owner(const std::shared_ptr<const void>& a,const std::shared_ptr<const void>& b) noexcept {
        return !a.owner_before(b) && !b.owner_before(a);
    }
    std::vector<Allocation> allocations_;
};
}
