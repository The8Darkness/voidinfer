#pragma once
#include "exl3/registered_kv_backing.h"
#include "exl3/bounded_shared_owner.h"
#include <array>
#include <mutex>
#include <span>

namespace ninfer::exl3 {
// Bounded weak lookup, not a pinned-storage pool. Registry inventory
// retains successful physical owners until explicit idle retirement. Transfer
// reader pins prevent eviction from unregistering an in-flight source.
class Exl3KVRegistrationCache {
    static constexpr std::size_t capacity=64;
    std::array<std::weak_ptr<Exl3RegisteredKVBacking>,capacity> entries_{};
    std::size_t next_retirement_=0;
    std::atomic<std::uint64_t> retired_registrations_{0};
    std::atomic<std::uint64_t> external_readers_{0};
    std::atomic<std::uint64_t> external_reader_high_water_{0};
    std::atomic<bool> refuse_read_pairs_for_test_{false};
    const void* authority_=nullptr;
    const bool runtime_retirement_enabled_;
    std::mutex mutex_;
    struct ConstructionKey {};
public:
    void add_external_readers(std::uint64_t count) noexcept {
        const auto current=external_readers_.fetch_add(count,std::memory_order_acq_rel)+count;
        auto observed=external_reader_high_water_.load(std::memory_order_relaxed);
        while(observed<current && !external_reader_high_water_.compare_exchange_weak(
            observed,current,std::memory_order_release,std::memory_order_relaxed)) {}
    }
    class ExternalRead {
        friend class Exl3KVRegistrationCache;
        std::shared_ptr<Exl3KVRegistrationCache> owner_;
        explicit ExternalRead(std::shared_ptr<Exl3KVRegistrationCache> owner) noexcept:owner_(std::move(owner)){}
    public:
        ExternalRead(const ExternalRead&)=delete;
        ExternalRead& operator=(const ExternalRead&)=delete;
        ExternalRead(ExternalRead&& other) noexcept:owner_(std::move(other.owner_)){}
        ~ExternalRead(){if(owner_)owner_->external_readers_.fetch_sub(1,std::memory_order_acq_rel);}
        bool valid() const noexcept {return owner_ && owner_.use_count();}
    };
    static std::optional<ExternalRead> acquire_external_read(std::shared_ptr<Exl3KVRegistrationCache> owner) {
        if(!owner || !owner.use_count())return std::nullopt;
        std::unique_lock lock(owner->mutex_,std::try_to_lock);
        if(!lock.owns_lock() || owner->external_readers_.load()==std::numeric_limits<std::uint64_t>::max())
            return std::nullopt;
        // Serialized with the complete retirement transaction, not only its
        // eligibility check. Releases need no lock or allocation.
        owner->add_external_readers(1);
        return ExternalRead(std::move(owner));
    }
    std::uint64_t external_readers() const noexcept {return external_readers_.load(std::memory_order_acquire);}
    std::uint64_t external_reader_high_water() const noexcept {
        return external_reader_high_water_.load(std::memory_order_acquire);
    }
    // Reserve both plane readers before either producer can submit. Failure
    // leaves the caller free to use its ordinary upload path for the whole bank.
    static std::optional<std::array<ExternalRead,2>> acquire_external_read_pair(
        std::shared_ptr<Exl3KVRegistrationCache> owner) {
        if(!owner || !owner.use_count())return std::nullopt;
        if(owner->refuse_read_pairs_for_test_.load(std::memory_order_relaxed))return std::nullopt;
        std::unique_lock lock(owner->mutex_,std::try_to_lock);
        if(!lock.owns_lock() || owner->external_readers_.load()>
            std::numeric_limits<std::uint64_t>::max()-2)return std::nullopt;
        owner->add_external_readers(2);
        return std::array<ExternalRead,2>{ExternalRead(owner),ExternalRead(std::move(owner))};
    }
    void refuse_read_pairs_for_test(bool value) noexcept {
        refuse_read_pairs_for_test_.store(value,std::memory_order_relaxed);
    }
    std::uint64_t retired_registrations() const noexcept {return retired_registrations_.load(std::memory_order_relaxed);}
    explicit Exl3KVRegistrationCache(ConstructionKey,const void* authority,bool runtime_retirement)
        :authority_(authority),runtime_retirement_enabled_(runtime_retirement){}
    static constexpr std::size_t metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3KVRegistrationCache>();
    }
    template<class Coordinator>
    static std::shared_ptr<Exl3KVRegistrationCache> create_startup(Coordinator& authority,bool runtime_retirement=true) {
        Exl3ResourceInventory::Requirement required;required.configuration=0x4b56434143+(runtime_retirement?1:0);
        required.add(Exl3ResourceInventory::Domain::host_metadata,1,metadata_bytes());
        std::shared_ptr<Exl3KVRegistrationCache> result;
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("registration cache reservation identity");
            result=make_bounded_shared<Exl3KVRegistrationCache>(ConstructionKey{},&authority,runtime_retirement);
            Exl3ResourceInventory actual;
            actual.add({result,0,Exl3ResourceInventory::Domain::host_metadata,metadata_bytes(),{},
                &attach_bounded_retirement_credit<Exl3KVRegistrationCache,const void>});
            return actual;
        },[&]() noexcept {result.reset();});
        return result;
    }
    template<class Coordinator>
    std::shared_ptr<Exl3RegisteredKVBacking> acquire_startup(Coordinator& authority,
        const Exl3ExactKVExtent& extent,Exl3KVRegistrationProvider provider={}) {
        return acquire_impl(&authority,extent,provider,[&] {
            return Exl3RegisteredKVBacking::acquire_startup(authority,extent,provider);
        },[](const auto&) {return false;}); // Startup has no runtime retirement lease.
    }
    template<class Coordinator>
    std::shared_ptr<Exl3RegisteredKVBacking> acquire_runtime(Coordinator& authority,
        const typename Coordinator::Lease& lease,const Exl3ExactKVExtent& extent,
        Exl3KVRegistrationProvider provider={}) {
        if(!authority.compute_leases_current(std::span(&lease,1)))
            throw std::invalid_argument("registration cache stale runtime lease");
        auto owner=acquire_impl(&authority,extent,provider,[&] {
            return Exl3RegisteredKVBacking::acquire_runtime(authority,lease,extent,provider);
        },[&](const std::shared_ptr<Exl3RegisteredKVBacking>& retired) {
            if(!retired->seal_idle())return false;
            using Inventory=Exl3ResourceInventory;
            if(!retired->retired_successfully()) {
                const Inventory::Allocation registration{retired,0,Inventory::Domain::cuda_registered_host,
                    retired->registered_bytes(),{},nullptr,nullptr,&Exl3RegisteredKVBacking::attach_registration_credit};
                if(!authority.retire_runtime_resource(lease,registration,
                    [&](const auto&) noexcept {return retired->retire_sealed();})) {
                    if(Exl3RegisteredKVBacking::quarantined_bytes())
                        throw std::runtime_error("registration cache retirement unresolved");
                    return false;
                }
            }
            const Inventory::Allocation metadata{retired,1,Inventory::Domain::host_metadata,
                Exl3RegisteredKVBacking::metadata_bytes(),{},&Exl3RegisteredKVBacking::attach_retirement_credit};
            return authority.retire_runtime_metadata_to_lifetime(lease,metadata);
        });
        // Hits and fallback decisions may bypass allocation authority. Both
        // require the current request lease before the caller can upload.
        if(!authority.compute_leases_current(std::span(&lease,1)))
            throw std::invalid_argument("registration cache runtime lease changed during acquisition");
        return owner;
    }
private:
    template<class Allocate,class Retire>
    std::shared_ptr<Exl3RegisteredKVBacking> acquire_impl(const void* authority,
        const Exl3ExactKVExtent& extent,Exl3KVRegistrationProvider provider,Allocate&& allocate,Retire&& retire) {
        if(authority_!=authority)throw std::invalid_argument("registration cache authority changed");
        // Reject malformed dependencies before a full-cache miss can retire
        // an otherwise usable registration to make room for the new one.
        if(!provider.acquire || !provider.release)
            throw std::invalid_argument("KV registration provider incomplete");
        if(!extent.registration_eligible())return {};
        std::unique_lock lock(mutex_,std::try_to_lock);
        if(!lock.owns_lock())return {}; // peer uses exact staging; never waits on cache/provider lock
        if(Exl3RegisteredKVBacking::quarantined_bytes())throw std::runtime_error("unresolved KV registration retirement");
        std::size_t empty=capacity;
        for(std::size_t i=0;i<capacity;++i) {
            if(auto owner=entries_[i].lock()) {
                if(owner->covers(extent)) {
                    // Same range with a different provider must not be registered
                    // twice or released through a mismatched callback table.
                    if(!owner->uses(provider))return {};
                    return owner;
                }
                if(owner->overlaps(extent))return {}; // unrelated/stale overlap cannot create another registration
            } else if(empty==capacity)empty=i;
        }
        const auto reclaim_one=[&]() {
            if(!runtime_retirement_enabled_ || external_readers_.load(std::memory_order_acquire))return capacity;
            for(std::size_t step=0;step<capacity;++step) {
                const auto index=(next_retirement_+step)%capacity;
                if(auto owner=entries_[index].lock();owner && retire(owner)) {
                    retired_registrations_.fetch_add(1,std::memory_order_relaxed);
                    entries_[index].reset();next_retirement_=(index+1)%capacity;return index;
                }
            }
            return capacity;
        };
        bool reclaimed=false;
        if(empty==capacity) {
            empty=reclaim_one();reclaimed=empty!=capacity;
            if(!reclaimed)return {}; // all readers active, or no runtime authority
        }
        std::shared_ptr<Exl3RegisteredKVBacking> owner;
        try {owner=std::forward<Allocate>(allocate)();}
        catch(const Exl3ResourceReservationExhausted&) {
            if(reclaimed)return {}; // at most one idle registration retires per miss
            empty=reclaim_one();if(empty==capacity)return {};
            try {owner=std::forward<Allocate>(allocate)();}
            catch(const Exl3ResourceReservationExhausted&) {return {};}
        }
        if(owner)entries_[empty]=owner;
        return owner;
    }
};
}
