#pragma once

#include "exl3/host_resident_set.h"
#include "exl3/vericache_request.h"
#include "exl3/device_page_key.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <optional>
#include <span>
#include <vector>
#include <utility>

namespace ninfer::exl3 {
class Exl3Dflash2Execution;

// Explicit in-process admission/publication owner for already-authoritative
// VeriCache roots. Numerical execution remains caller-owned. In particular,
// this class does not create contexts, streams, threads, or token policy.
class Exl3VeriCacheServingCoordinator {
    struct RuntimeConstructionState {
        Exl3ResourceInventory::Totals ceiling{},issued{};
        RetainedDeviceLedger device;
        RetainedCudaRegistrationLedger registration;
        RetainedDescriptorLedger metadata;
    };
    RuntimeConstructionState* runtime_construction_state_=nullptr;
public:
    struct RegistrationConstructorCredits {
        RetainedCudaRegistrationLedger::Ticket registration;
        RetainedDescriptorLedger::Ticket metadata;
    };
    RegistrationConstructorCredits reserve_registration_constructor_credits(std::uint64_t registration,std::uint64_t metadata) {
        std::lock_guard lock(mutex_);
        if(!runtime_construction_state_ || !allocating_ || !runtime_allocating_ ||
            allocation_thread_!=std::this_thread::get_id() || !registration || !metadata)
            throw std::logic_error("registration constructor credits require active runtime factory");
        auto& state=*runtime_construction_state_;
        const auto r=static_cast<unsigned>(Exl3ResourceInventory::Domain::cuda_registered_host);
        const auto m=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
        if(state.issued[r]>state.ceiling[r] || state.issued[m]>state.ceiling[m] ||
            registration>state.ceiling[r]-state.issued[r] || metadata>state.ceiling[m]-state.issued[m])
            throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
        auto registration_ticket=state.registration.acquire(registration);
        auto metadata_ticket=state.metadata.acquire(metadata);
        state.issued[r]+=registration;state.issued[m]+=metadata;
        return {std::move(registration_ticket),std::move(metadata_ticket)};
    }
    RegistrationConstructorCredits reserve_startup_registration_constructor_credits(std::uint64_t registration,std::uint64_t metadata) {
        std::lock_guard lock(mutex_);
        if(!allocating_ || runtime_allocating_ || constructor_thread_!=std::this_thread::get_id() ||
            !registration || !metadata)throw std::logic_error("registration credits require active startup factory");
        const auto r=static_cast<unsigned>(Exl3ResourceInventory::Domain::cuda_registered_host);
        const auto m=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
        if(constructor_issued_[r]>constructor_ceiling_[r] || constructor_issued_[m]>constructor_ceiling_[m] ||
            registration>constructor_ceiling_[r]-constructor_issued_[r] || metadata>constructor_ceiling_[m]-constructor_issued_[m])
            throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
        auto a=constructor_registration_.acquire(registration);auto b=constructor_metadata_.acquire(metadata);
        constructor_issued_[r]+=registration;constructor_issued_[m]+=metadata;return {std::move(a),std::move(b)};
    }
    struct ConstructorCredits {
        RetainedDeviceLedger::Ticket device;
        RetainedDescriptorLedger::Ticket metadata;
    };
    ConstructorCredits reserve_runtime_constructor_credits(std::uint64_t device,std::uint64_t metadata) {
        std::lock_guard lock(mutex_);
        if(!runtime_construction_state_ || !allocating_ || !runtime_allocating_ ||
            allocation_thread_!=std::this_thread::get_id() || !device || !metadata)
            throw std::logic_error("device constructor credits require active runtime factory");
        auto& state=*runtime_construction_state_;
        const auto d=static_cast<unsigned>(Exl3ResourceInventory::Domain::device);
        const auto m=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
        if(state.issued[d]>state.ceiling[d] || state.issued[m]>state.ceiling[m] ||
            device>state.ceiling[d]-state.issued[d] || metadata>state.ceiling[m]-state.issued[m])
            throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
        auto device_ticket=state.device.acquire(device);
        auto metadata_ticket=state.metadata.acquire(metadata);
        state.issued[d]+=device;state.issued[m]+=metadata;
        return {std::move(device_ticket),std::move(metadata_ticket)};
    }
    // Provisional charges cover construction only. Committed inventory assumes
    // ownership before callers release them; failed cleanup retains their states.
    ConstructorCredits reserve_constructor_credits(std::uint64_t device,std::uint64_t metadata) {
        std::lock_guard lock(mutex_);
        if(!allocating_ || runtime_allocating_ || constructor_thread_!=std::this_thread::get_id() ||
            !device || !metadata)throw std::logic_error("constructor credits require active startup factory");
        using Domain=Exl3ResourceInventory::Domain;
        const auto d=static_cast<unsigned>(Domain::device),m=static_cast<unsigned>(Domain::host_metadata);
        if(constructor_issued_[d]>constructor_ceiling_[d] || constructor_issued_[m]>constructor_ceiling_[m] ||
            device>constructor_ceiling_[d]-constructor_issued_[d] || metadata>constructor_ceiling_[m]-constructor_issued_[m])
            throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
        auto device_ticket=constructor_device_.acquire(device);
        auto metadata_ticket=constructor_metadata_.acquire(metadata);
        constructor_issued_[d]+=device;constructor_issued_[m]+=metadata;
        return {std::move(device_ticket),std::move(metadata_ticket)};
    }
    struct Policy {
        std::size_t logical_capacity = 1;
        std::size_t physical_capacity = 1;
        std::uint64_t maximum_resident_page_bytes = 0;
        std::uint64_t physical_reserve_bytes = 0;
    };
    struct Ticket {
        std::uint64_t request_id = 0;
        std::uint64_t generation = 0;
    };
    struct Lease {
        Ticket ticket;
        std::shared_ptr<const Exl3VeriCacheRequest> root;
        // Acquisition identity is separate from the publication generation:
        // yielding does not invalidate a client's cancellation ticket.
        std::uint64_t acquisition = 0;
    };
    // Logical authority while a caller-owned physical lane keeps the exact
    // target/draft state. The initial host root stays resident until one
    // terminal child is supplied. This handle never owns numerical state.
    struct DeviceLogicalLease {
        Ticket ticket;
        std::shared_ptr<const Exl3VeriCacheRequest> initial_root;
        std::shared_ptr<const void> physical_owner;
        std::uint64_t acquisition = 0;
        int frontier = 0;
    };
    struct DeviceLogicalPublication {
        DeviceLogicalLease lease;
        std::size_t committed_tokens = 0;
        std::uint64_t sequence = 0;
    };
    // Issued only by a physical DFlash lane after its final scoped event has
    // completed and retired. Logical cancellation or lease liveness cannot
    // manufacture this proof.
    class Retirement {
        friend class Exl3Dflash2Execution;
        friend class Exl3VeriCacheServingCoordinator;
        Ticket ticket_{};
        std::uint64_t acquisition_=0,execution_=0;
        std::uint64_t final_use_generation_=0,retirement_generation_=0;
        Retirement(Ticket ticket,std::uint64_t acquisition,
            std::uint64_t execution,std::uint64_t final_use_generation,
            std::uint64_t retirement_generation) noexcept
            :ticket_(ticket),acquisition_(acquisition),execution_(execution),
             final_use_generation_(final_use_generation),
             retirement_generation_(retirement_generation) {}
    public:
        Retirement()=default;
        bool valid() const noexcept {
            return ticket_.request_id && ticket_.generation && acquisition_ &&
                execution_ && final_use_generation_ && retirement_generation_;
        }
        std::uint64_t execution() const noexcept {return execution_;}
        std::uint64_t final_use_generation() const noexcept {
            return final_use_generation_;
        }
        std::uint64_t generation() const noexcept {
            return retirement_generation_;
        }
    };
    struct Publication {
        Lease lease;
        std::int64_t token = -1;
        std::uint64_t sequence = 0;
        Exl3HostResidentSet::Stats residency;
    };
    struct Turnover {
        Ticket completed;
        Ticket admitted;
        Exl3HostResidentSet::Stats residency;
    };
    struct WindowPublication {
        Lease lease;
        std::vector<std::int64_t> tokens;
        std::uint64_t sequence = 0;
        Exl3HostResidentSet::Stats residency;
    };
    struct Stats {
        std::size_t queued = 0;
        std::size_t active = 0;
        std::size_t admitted = 0;
        std::size_t logical_capacity = 0;
        std::size_t physical_capacity = 0;
        std::uint64_t publications = 0;
        std::uint64_t cancellations = 0;
        std::uint64_t completions = 0;
        std::uint64_t resident_updates = 0;
        std::uint64_t resident_new_locked_bytes = 0;
        std::uint64_t resident_unlocked_bytes = 0;
        double resident_inventory_ms = 0.0;
        double resident_registry_ms = 0.0;
        std::uint64_t locked_page_bytes = 0;
        bool closed = false;
        std::optional<Exl3ResourceInventory::Totals> retained_resource_units;
        bool closing=false;
        bool external_retirement_unresolved=false;
    };

    Exl3VeriCacheServingCoordinator(
        Exl3VeriCacheServingPrefixCache& cache, Policy policy)
        : cache_(cache), policy_(policy),
          resident_(policy.maximum_resident_page_bytes,
                    policy.physical_reserve_bytes) {
        if (!policy_.logical_capacity || !policy_.physical_capacity ||
            policy_.physical_capacity > policy_.logical_capacity ||
            !policy_.maximum_resident_page_bytes ||
            policy_.physical_reserve_bytes < (1ULL << 30))
            throw std::invalid_argument("serving coordinator policy extent");
    }
    Exl3VeriCacheServingCoordinator(
        const Exl3VeriCacheServingCoordinator&) = delete;
    Exl3VeriCacheServingCoordinator& operator=(
        const Exl3VeriCacheServingCoordinator&) = delete;

    Ticket admit(std::shared_ptr<const Exl3VeriCacheRequest> root) {
        validate_root(root);
        std::unique_lock lock(mutex_);
        require_open(lock);
        if (entries().size() >= policy_.logical_capacity)
            throw std::runtime_error("serving coordinator logical capacity exhausted");
        const auto identity = root->state()->model_identity();
        if (model_identity_ && model_identity_ != identity)
            throw std::invalid_argument("serving coordinator model identity mismatch");
        const auto id = next_request_id_;
        // Allocate queue/entry storage before publishing a new resident set.
        entries().reserve(entries().size() + 1);
        // Active entries may all yield after this admission. Reserve their
        // return slots before the resident frontier becomes visible too.
        queue().reserve(entries().size() + 1);
        const std::array<std::pair<std::uint64_t,
            std::shared_ptr<const Exl3VeriCacheRequest>>,1> candidate{{{id, root}}};
        const auto residency = replace_resident_locked(candidate, 0, true);
        (void)residency;
        entries().push_back(Entry{id, 1, false, 0, std::move(root)});
        queue().push_back(id);
        ++next_request_id_;
        if (!model_identity_) model_identity_ = std::move(identity);
        high_admitted_ = std::max(high_admitted_, entries().size());
        high_queued_ = std::max(high_queued_, queue().size());
        return Ticket{id, 1};
    }

    // Startup-only binding of real physical owners. Every subsequent root
    // transition retains them under the SAME sealed residency revision.
    // Allocation-before-startup remains a separate, unfinished reservation seam.
    void bind_physical_resources(Exl3ResourceInventory resources,
        Exl3ResourceInventory::Totals limits) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(physical_bound_ || !entries().empty() || resident_updates_)
            throw std::logic_error("physical inventory must precede admission");
        resident_.set_resource_limits(limits);
        Exl3HostResidentSet::Snapshot snapshot;snapshot.resources=resources;
        resident_.replace(std::move(snapshot));
        physical_resources_=std::move(resources);physical_bound_=true;
    }
    std::size_t resident_root_capacity() const {
        const auto cached=cache_.root_capacity();
        if(policy_.logical_capacity>(std::numeric_limits<std::size_t>::max()-cached)/2)
            throw std::overflow_error("coordinator root snapshot capacity overflow");
        return cached+2*policy_.logical_capacity;
    }
    Exl3ResourceInventory::Requirement metadata_requirement() const {
        Exl3ResourceInventory::Requirement required;
        required.configuration=0x434f4f52444d4554ULL;
        using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<MetadataStorage>());
        required.add(Domain::host_metadata,policy_.logical_capacity,sizeof(Entry));
        required.add(Domain::host_metadata,policy_.logical_capacity,sizeof(std::uint64_t));
        required.add(Domain::host_metadata,policy_.physical_capacity,sizeof(Entry*));
        required.add(Domain::host_metadata,policy_.logical_capacity,sizeof(Replacement));
        required.add(Domain::host_metadata,resident_root_capacity(),sizeof(std::shared_ptr<const Exl3VeriCacheRequest>));
        return required;
    }
    // Reserve all logical slots once, before any roots or vector capacity exist.
    // Both coordinator and inventory retain the same backing allocation owner.
    void reserve_metadata_startup(unsigned fault=0) {
        if(fault>4)throw std::invalid_argument("coordinator metadata fault extent");
        std::unique_lock lock(mutex_);require_open(lock);
        if(!physical_bound_ || metadata_ || entries().capacity() || queue().capacity() || resident_updates_)
            throw std::logic_error("coordinator metadata reservation must precede admission");
        const auto required=metadata_requirement();
        auto next=physical_resources_;
        std::shared_ptr<MetadataStorage> prepared;
        allocating_=true;lock.unlock();
        try {
            resident_.allocate_reserved(required,[&](std::uint64_t configuration) {
                if(configuration!=required.configuration)
                    throw std::logic_error("coordinator metadata configuration changed");
                if(fault==4)throw std::logic_error("coordinator metadata credit refusal reached allocator");
                if(fault==1)throw std::bad_alloc();
                prepared=make_bounded_shared<MetadataStorage>();
                prepared->entries.reserve(policy_.logical_capacity);
                if(fault==2)throw std::bad_alloc();
                prepared->queue.reserve(policy_.logical_capacity);
                prepared->selected.reserve(policy_.physical_capacity);
                prepared->replacements.reserve(policy_.logical_capacity);
                prepared->roots.reserve(resident_root_capacity());
                if(prepared->entries.capacity()!=policy_.logical_capacity ||
                   prepared->queue.capacity()!=policy_.logical_capacity ||
                   prepared->selected.capacity()!=policy_.physical_capacity ||
                   prepared->replacements.capacity()!=policy_.logical_capacity ||
                   prepared->roots.capacity()!=resident_root_capacity())
                    throw std::length_error("coordinator metadata capacity mismatch");
                using Domain=Exl3ResourceInventory::Domain;
                Exl3ResourceInventory actual;
                actual.add({prepared,0,Domain::host_metadata,
                    required.units[static_cast<unsigned>(Domain::host_metadata)]-(fault==3?1:0),{},
                    &attach_bounded_retirement_credit<MetadataStorage,const void>});
                next.append(actual);return actual;
            });
            lock.lock();physical_resources_=std::move(next);metadata_=std::move(prepared);allocating_=false;
        } catch(...) {
            next={};prepared.reset();lock.lock();allocating_=false;throw;
        }
    }
    std::weak_ptr<const void> metadata_owner_for_test() const {
        std::unique_lock lock(mutex_);return metadata_;
    }
    static std::size_t metadata_owner_blocks_for_test() noexcept {
        return bounded_shared_live_blocks_for_test<MetadataStorage>();
    }
    bool publication_scratch_empty_for_test() const {
        std::unique_lock lock(mutex_);
        const auto& storage=metadata_?*metadata_:fallback_metadata_;
        return storage.selected.empty() && storage.replacements.empty() && storage.roots.empty();
    }
    struct PublicationScratchSnapshot {
        std::uintptr_t selected_storage=0,replacement_storage=0,root_storage=0;
        std::size_t selected_capacity=0,replacement_capacity=0,root_capacity=0;
        std::uint64_t publication_epoch=0,root_epoch=0;
        bool publication_active=false,root_active=false;
    };
    PublicationScratchSnapshot publication_scratch_snapshot_for_test() const {
        std::unique_lock lock(mutex_);
        const auto& storage=metadata_?*metadata_:fallback_metadata_;
        return {reinterpret_cast<std::uintptr_t>(storage.selected.data()),
            reinterpret_cast<std::uintptr_t>(storage.replacements.data()),
            reinterpret_cast<std::uintptr_t>(storage.roots.data()),
            storage.selected.capacity(),storage.replacements.capacity(),
            storage.roots.capacity(),storage.publication_epoch,storage.root_epoch,
            storage.publication_scratch_active,storage.root_scratch_active};
    }
    enum class ResidencyFault { none, before_commit, after_first_lock, page_budget_before_commit };
    class LogicalHostLease {
        friend class Exl3VeriCacheServingCoordinator;
        const Exl3VeriCacheServingCoordinator* authority_=nullptr;
        Exl3ResourceInventory::Allocation allocation_;
        LogicalHostLease(const Exl3VeriCacheServingCoordinator* authority,Exl3ResourceInventory::Allocation allocation)
            :authority_(authority),allocation_(std::move(allocation)) {}
    public:
        LogicalHostLease(LogicalHostLease&&)=default;
        LogicalHostLease& operator=(LogicalHostLease&&)=delete;
        LogicalHostLease(const LogicalHostLease&)=delete;
        std::uint64_t bytes() const noexcept {return allocation_.units;}
    };
    // Logical request/result storage has no model-state acquisition. It uses
    // the same reservation authority but may allocate only ordinary host data.
    template<class Factory,class Rollback>
    LogicalHostLease allocate_logical_host(std::uint64_t bytes,Factory&& factory,Rollback&& rollback,
        Exl3HostResidentSet::ReservationExtent extent=Exl3HostResidentSet::ReservationExtent::exact) {
        using Inventory=Exl3ResourceInventory;
        Inventory::Requirement required;required.configuration=0x4c4f47484f5354;
        required.add(Inventory::Domain::host_metadata,1,bytes);
        std::unique_lock lock(mutex_);require_open(lock);
        if(!physical_bound_)throw std::logic_error("logical host reservation requires bound authority");
        auto next=physical_resources_;
        Inventory::Allocation allocation;
        allocating_=true;runtime_allocating_=true;allocation_thread_=std::this_thread::get_id();lock.unlock();
        try {
            resident_.allocate_reserved_growing(required,[&](auto configuration,auto&&) {
                if(configuration!=required.configuration)throw std::logic_error("logical host reservation identity");
                allocation=std::forward<Factory>(factory)();
                if(allocation.domain!=Inventory::Domain::host_metadata)
                    throw std::invalid_argument("logical host allocation domain");
                Inventory actual;actual.add(allocation);next.append(actual);return actual;
            },extent);
            lock.lock();physical_resources_=std::move(next);
            allocating_=false;runtime_allocating_=false;allocation_thread_={};allocation_changed_.notify_all();
            return LogicalHostLease(this,std::move(allocation));
        } catch(...) {
            try{std::forward<Rollback>(rollback)();}catch(...){
                resident_.retain_failed_factory(std::move(next),std::move(allocation));
                seal_failed_startup_retirement();
            }
            next={};allocation={};
            lock.lock();allocating_=false;runtime_allocating_=false;allocation_thread_={};allocation_changed_.notify_all();throw;
        }
    }
    template<class Retire> bool retire_logical_host(LogicalHostLease& lease,Retire&& retire) {
        return retire_logical_host_impl<false>(lease,std::forward<Retire>(retire));
    }
    template<class Attach> bool retire_logical_host_to_lifetime(LogicalHostLease& lease,Attach&& attach) {
        return retire_logical_host_impl<true>(lease,std::forward<Attach>(attach));
    }
    template<bool Deferred,class Retire> bool retire_logical_host_impl(LogicalHostLease& lease,Retire&& retire) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(lease.authority_!=this || !lease.allocation_.owner)
            throw std::invalid_argument("logical host retirement stale authority");
        auto next=physical_resources_.without_allocation(lease.allocation_);
        allocating_=true;runtime_allocating_=true;allocation_thread_=std::this_thread::get_id();lock.unlock();
        try {
            const bool retired=[&] {
                if constexpr(Deferred)return resident_.retire_reserved_host_to_lifetime(lease.allocation_,std::forward<Retire>(retire));
                else return resident_.retire_reserved_host(lease.allocation_,std::forward<Retire>(retire));
            }();
            lock.lock();
            if(retired){physical_resources_=std::move(next);lease.allocation_={};lease.authority_=nullptr;}
            allocating_=false;runtime_allocating_=false;allocation_thread_={};allocation_changed_.notify_all();return retired;
        } catch(...) {
            next={};lock.lock();allocating_=false;runtime_allocating_=false;allocation_thread_={};allocation_changed_.notify_all();throw;
        }
    }
    template<class Factory> void allocate_startup_resources(
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory) {
        allocate_startup_resources_growing(requirement,[&](auto configuration,auto&&) {
            return std::forward<Factory>(factory)(configuration);
        });
    }
    template<class Factory> void allocate_startup_resources_growing(
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory) {
        allocate_startup_resources_growing(requirement,std::forward<Factory>(factory),[]() noexcept {});
    }
    template<class Factory,class Rollback,class Observe=std::nullptr_t> void allocate_startup_resources(
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory,Rollback&& rollback,Observe&& observe=nullptr) {
        allocate_startup_resources_growing(requirement,[&](auto configuration,auto&&) {
            return std::forward<Factory>(factory)(configuration);
        },std::forward<Rollback>(rollback),std::forward<Observe>(observe));
    }
    // Synchronous idle construction only. Temporary peak owners must retire
    // before the factory returns; only the declared retained extent commits.
    template<class Factory,class Rollback,class Observe=std::nullptr_t>
    void allocate_startup_peak_resources(const Exl3ResourceInventory::Requirement& peak,
        const Exl3ResourceInventory::Requirement& retained,Factory&& factory,Rollback&& rollback,Observe&& observe=nullptr) {
        if(!peak.configuration || retained.configuration!=peak.configuration ||
            retained.units==Exl3ResourceInventory::Totals{})
            throw std::invalid_argument("startup peak retained identity or extent");
        for(unsigned i=0;i<peak.units.size();++i)
            if(retained.units[i]>peak.units[i])throw std::invalid_argument("startup retained extent exceeds peak");
        const auto expected_retained=retained.units;
        allocate_startup_resources_growing(peak,[&](auto configuration,auto&&) {
            auto actual=std::forward<Factory>(factory)(configuration);
            if(actual.totals()!=expected_retained)
                throw std::invalid_argument("startup peak retained extent mismatch");
            return actual;
        },std::forward<Rollback>(rollback),std::forward<Observe>(observe),
            Exl3HostResidentSet::ReservationExtent::at_most);
    }
    template<class Factory,class Rollback,class Observe=std::nullptr_t> void allocate_startup_resources_growing(
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory,Rollback&& rollback,Observe&& observe=nullptr,
        Exl3HostResidentSet::ReservationExtent extent=Exl3HostResidentSet::ReservationExtent::exact) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(!physical_bound_ || !entries().empty() || resident_updates_)
            throw std::logic_error("startup reservation must precede admission");
        auto next=physical_resources_;
        constructor_ceiling_=requirement.units;constructor_issued_={};
        constructor_thread_=std::this_thread::get_id();
        allocating_=true;lock.unlock();
        Exl3ResourceInventory produced;
        try {
            // No coordinator lock across allocator callbacks. Mutation APIs
            // refuse while this exclusive startup construction is in flight.
            resident_.allocate_reserved_growing(requirement,[&](std::uint64_t configuration,auto&& extend) {
                const auto grow=[&](const Exl3ResourceInventory::Requirement& next_requirement) {
                    extend(next_requirement);
                    std::lock_guard state_lock(mutex_);constructor_ceiling_=next_requirement.units;
                };
                produced=std::forward<Factory>(factory)(configuration,grow);
                next.append(produced); // prepare coordinator ownership before commit
                return produced;
            },extent);
            lock.lock();physical_resources_=std::move(next);allocating_=false;constructor_thread_={};
        } catch(...) {
            // Keep startup admission exclusive until external factory owners
            // retire, including failures in extent checks after factory return.
            try {std::forward<Rollback>(rollback)();}
            catch(...) {
                resident_.retain_failed_factory(std::move(produced));
                seal_failed_startup_retirement();
            }
            next={};produced={};
            // Inventory copies can be the final owners. Observe their completed
            // destruction before reopening startup admission, without replaying rollback.
            if constexpr(!std::is_same_v<std::remove_cvref_t<Observe>,std::nullptr_t>) {
                try {std::forward<Observe>(observe)();}
                catch(...) {seal_failed_startup_retirement();}
            }
            if(constructor_device_.bytes() || constructor_registration_.bytes() || constructor_metadata_.bytes())seal_failed_startup_retirement();
            lock.lock();allocating_=false;constructor_thread_={};throw;
        }
    }
    // Lease-bound creation of idle registration/metadata owners. No GPU copy,
    // event wait or numerical callback is permitted inside the factory.
    template<class Factory> void allocate_runtime_resources(const Lease& lease,
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory) {
        allocate_runtime_resources_impl(&lease,{},requirement,std::forward<Factory>(factory),[]() noexcept {},nullptr);
    }
    template<class Factory,class Rollback,class Observe=std::nullptr_t> void allocate_runtime_resources(const Lease& lease,
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory,Rollback&& rollback,Observe&& observe=nullptr) {
        static_assert(std::is_nothrow_invocable_v<Rollback&>,"runtime rollback must not throw");
        static_assert(std::is_same_v<std::remove_cvref_t<Observe>,std::nullptr_t> ||
            std::is_nothrow_invocable_v<Observe&>,"runtime rollback observer must not throw");
        allocate_runtime_resources_impl(&lease,{},requirement,std::forward<Factory>(factory),
            std::forward<Rollback>(rollback),std::forward<Observe>(observe));
    }
    // Prefix preparation owns a physical context before a model-state lease
    // exists. Only idle allocation/registration work belongs in this callback.
    template<class Factory> void allocate_preparation_resources(const std::shared_ptr<const void>& preparation_owner,
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory) {
        if(!preparation_owner || !preparation_owner.use_count())throw std::invalid_argument("preparation resource owner missing");
        allocate_runtime_resources_impl(nullptr,preparation_owner,requirement,std::forward<Factory>(factory),[]() noexcept {},nullptr);
    }
    template<class Factory,class Rollback,class Observe=std::nullptr_t>
    void allocate_preparation_resources(const std::shared_ptr<const void>& preparation_owner,
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory,Rollback&& rollback,Observe&& observe=nullptr) {
        if(!preparation_owner || !preparation_owner.use_count())throw std::invalid_argument("preparation resource owner missing");
        static_assert(std::is_nothrow_invocable_v<Rollback&>,"preparation rollback must not throw");
        static_assert(std::is_same_v<std::remove_cvref_t<Observe>,std::nullptr_t> ||
            std::is_nothrow_invocable_v<Observe&>,"preparation observer must not throw");
        allocate_runtime_resources_impl(nullptr,preparation_owner,requirement,std::forward<Factory>(factory),
            std::forward<Rollback>(rollback),std::forward<Observe>(observe));
    }
    template<class Factory,class Rollback,class Observe> void allocate_runtime_resources_impl(const Lease* lease,
        const std::shared_ptr<const void>& preparation_owner,const Exl3ResourceInventory::Requirement& requirement,
        Factory&& factory,Rollback&& rollback,Observe&& observe) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(!physical_bound_ || (!lease && (!preparation_owner || !preparation_owner.use_count())))throw std::logic_error("runtime resource authority unavailable");
        if(lease) {
            const auto& entry=find_locked(lease->ticket.request_id);
            if(!entry.active || entry.generation!=lease->ticket.generation || entry.acquisition!=lease->acquisition || entry.root!=lease->root)
                throw std::invalid_argument("runtime reservation stale physical lease");
        }
        auto next=physical_resources_;
        RuntimeConstructionState constructor_state;constructor_state.ceiling=requirement.units;
        runtime_construction_state_=nullptr; // Authority opens only inside the physical factory below.
        allocating_=true;runtime_allocating_=true;allocation_thread_=std::this_thread::get_id();
        lock.unlock();
        try {
            resident_.allocate_reserved(requirement,[&](std::uint64_t configuration) {
                auto actual=[&] {
                    {std::lock_guard scope_lock(mutex_);runtime_construction_state_=&constructor_state;}
                    struct Scope {
                        Exl3VeriCacheServingCoordinator& owner;
                        ~Scope(){std::lock_guard scope_lock(owner.mutex_);owner.runtime_construction_state_=nullptr;}
                    } scope{*this};
                    return std::forward<Factory>(factory)(configuration);
                }();
                next.append(actual);return actual;
            });
            lock.lock();physical_resources_=std::move(next);
            runtime_construction_state_=nullptr;allocating_=false;runtime_allocating_=false;allocation_thread_={};
            allocation_changed_.notify_all();
        } catch(...) {
            // Release caller-held factory results before reopening admission.
            std::forward<Rollback>(rollback)();
            next={};
            if constexpr(!std::is_same_v<std::remove_cvref_t<Observe>,std::nullptr_t>)std::forward<Observe>(observe)();
            lock.lock();runtime_construction_state_=nullptr;allocating_=false;runtime_allocating_=false;allocation_thread_={};
            allocation_changed_.notify_all();throw;
        }
    }
    std::uint64_t owner_lifetime_metadata_bytes_for_test() const {
        std::lock_guard lock(mutex_);return resident_.owner_lifetime_metadata_bytes_for_test();
    }
    static constexpr std::size_t runtime_host_source_metadata_bytes() noexcept {return bounded_shared_allocation_bytes<HostSourceTable>();}
    std::size_t runtime_host_source_storage_bytes() const {
        std::lock_guard lock(mutex_);return host_sources_?runtime_host_source_metadata_bytes():0;
    }
    std::weak_ptr<const void> runtime_host_source_owner_for_test() const {
        std::lock_guard lock(mutex_);return host_sources_;
    }
    void reserve_runtime_host_sources() {
        {
            std::unique_lock lock(mutex_);require_open(lock);
            if(host_sources_)return;
        }
        Exl3ResourceInventory::Requirement required;required.configuration=0x4853524345;
        required.add(Exl3ResourceInventory::Domain::host_metadata,1,runtime_host_source_metadata_bytes());
        std::shared_ptr<HostSourceTable> prepared;
        allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("host source table reservation identity");
            prepared=make_bounded_shared<HostSourceTable>();
            Exl3ResourceInventory actual;
            actual.add({prepared,0,Exl3ResourceInventory::Domain::host_metadata,runtime_host_source_metadata_bytes(),{},
                &attach_bounded_retirement_credit<HostSourceTable,const void>});
            return actual;
        },[&]() noexcept {prepared.reset();});
        std::lock_guard lock(mutex_);host_sources_=std::move(prepared);
    }
    // Device fill cannot begin before these host ranges join the same old/new
    // residency union as request roots. Neither registry nor snapshots own fill.
    void bind_runtime_host_source(const Lease& lease,const std::shared_ptr<const void>& lifetime,
        const Exl3DevicePageKey& key) {
        std::unique_lock lock(mutex_);require_open(lock);
        const auto& entry=find_locked(lease.ticket.request_id);
        if(!physical_bound_ || !host_sources_ || !entry.active || entry.generation!=lease.ticket.generation ||
           entry.acquisition!=lease.acquisition || entry.root!=lease.root ||
           !lifetime || !lifetime.use_count() || !key.current())
            throw std::invalid_argument("runtime host source stale lease/owner");
        HostSourceBinding prepared;prepared.lifetime=lifetime;prepared.source=key.page();
        key.visit_retained_host_allocations([&](const void* data,std::size_t bytes) {
            const auto first=reinterpret_cast<std::uintptr_t>(data);
            if(!first || bytes>std::numeric_limits<std::uintptr_t>::max()-first || prepared.count==prepared.ranges.size())
                throw std::invalid_argument("runtime host source extent");
            prepared.ranges[prepared.count++]={first,first+bytes};
        });
        auto slot=host_sources_->end();
        for(auto it=host_sources_->begin();it!=host_sources_->end();++it) {
            if(!it->lifetime.owner_before(lifetime) && !lifetime.owner_before(it->lifetime))
                throw std::logic_error("runtime host source already bound");
            if(slot==host_sources_->end() && it->lifetime.expired())slot=it;
        }
        if(slot==host_sources_->end())throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
        auto previous=*slot;*slot=std::move(prepared);
        try{replace_resident_locked({},0);}
        catch(...){*slot=std::move(previous);throw;}
    }
    template<class Retire> bool retire_runtime_resource(const Lease& lease,
        const Exl3ResourceInventory::Allocation& allocation,Retire&& retire) {
        std::unique_lock lock(mutex_);require_open(lock);
        const auto& entry=find_locked(lease.ticket.request_id);
        if(!physical_bound_ || !entry.active || entry.generation!=lease.ticket.generation ||
            entry.acquisition!=lease.acquisition || entry.root!=lease.root)
            throw std::invalid_argument("runtime retirement stale physical lease");
        auto next=physical_resources_.without_allocation(allocation);
        allocating_=true;runtime_allocating_=true;allocation_thread_=std::this_thread::get_id();
        lock.unlock();
        try {
            const bool retired=resident_.retire_reserved(allocation,std::forward<Retire>(retire));
            lock.lock();
            if(retired)physical_resources_=std::move(next);
            allocating_=false;runtime_allocating_=false;allocation_thread_={};
            allocation_changed_.notify_all();return retired;
        } catch(...) {
            next={};lock.lock();allocating_=false;runtime_allocating_=false;allocation_thread_={};
            allocation_changed_.notify_all();throw;
        }
    }
    bool retire_runtime_metadata_to_lifetime(const Lease& lease,
        const Exl3ResourceInventory::Allocation& allocation) {
        return retire_runtime_metadata_to_lifetime_impl(&lease,{},allocation);
    }
    bool retire_preparation_metadata_to_lifetime(const std::shared_ptr<const void>& preparation_owner,
        const Exl3ResourceInventory::Allocation& allocation) {
        if(!preparation_owner || !preparation_owner.use_count())throw std::invalid_argument("preparation metadata owner missing");
        return retire_runtime_metadata_to_lifetime_impl(nullptr,preparation_owner,allocation);
    }
    bool retire_runtime_metadata_to_lifetime_impl(const Lease* lease,const std::shared_ptr<const void>& preparation_owner,
        const Exl3ResourceInventory::Allocation& allocation) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(!physical_bound_ || (!lease && (!preparation_owner || !preparation_owner.use_count())))throw std::logic_error("metadata resource authority unavailable");
        if(lease) {
            const auto& entry=find_locked(lease->ticket.request_id);
            if(!entry.active || entry.generation!=lease->ticket.generation || entry.acquisition!=lease->acquisition || entry.root!=lease->root)
                throw std::invalid_argument("metadata lifetime retirement stale physical lease");
        }
        if(!allocation.lifetime_retire || allocation.domain!=Exl3ResourceInventory::Domain::host_metadata)
            throw std::invalid_argument("metadata lifetime retirement hook unavailable");
        auto next=physical_resources_.without_allocation(allocation);
        if(!resident_.retire_reserved_host_to_lifetime(allocation,
            [&](RetainedDescriptorLedger::Ticket credit) noexcept {
                return allocation.lifetime_retire(allocation.owner,std::move(credit));
            }))return false;
        physical_resources_=std::move(next);return true;
    }
    RetainedDescriptorLedger::Ticket reserve_snapshot_metadata(const Lease* lease,
        const std::shared_ptr<const void>& preparation_owner,std::uint64_t bytes,
        bool exhaust_active_headroom_for_test=false) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(!physical_bound_ || (!lease && (!preparation_owner || !preparation_owner.use_count())))
            throw std::logic_error("snapshot metadata reservation authority unavailable");
        if(lease) {
            const auto& entry=find_locked(lease->ticket.request_id);
            if(!entry.active || entry.generation!=lease->ticket.generation ||
                entry.acquisition!=lease->acquisition || entry.root!=lease->root)
                throw std::invalid_argument("snapshot metadata reservation stale physical lease");
        }
        // The diagnostic follows the same exact lease gate as the reservation.
        // Tighten and reserve under one lock so no publication can interleave.
        if(exhaust_active_headroom_for_test) {
            if(!lease)throw std::invalid_argument("active metadata ceiling requires physical lease");
            resident_.tighten_metadata_headroom_for_test(0);
        }
        return resident_.reserve_host_metadata_lifetime(bytes);
    }
    RetainedHostAllocationLedger::Ticket reserve_snapshot_payload(const Lease& lease,std::uint64_t bytes) {
        std::unique_lock lock(mutex_);require_open(lock);
        const auto& entry=find_locked(lease.ticket.request_id);
        if(!physical_bound_ || !entry.active || entry.generation!=lease.ticket.generation ||
            entry.acquisition!=lease.acquisition || entry.root!=lease.root)
            throw std::invalid_argument("snapshot payload reservation stale physical lease");
        return resident_.reserve_host_payload_lifetime(bytes);
    }
    // Media encoding occurs while a physical lane owns preparation but before
    // its root can be admitted/acquired. The immutable cache payload may then
    // outlive that preparation owner, so its ticket retains the resident-ledger
    // backing directly just like a post-acquisition snapshot ticket.
    RetainedHostAllocationLedger::Ticket reserve_preparation_payload(
        const std::shared_ptr<const void>& preparation_owner,std::uint64_t bytes) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(!physical_bound_ || !preparation_owner || !preparation_owner.use_count())
            throw std::logic_error("media preparation payload authority unavailable");
        return resident_.reserve_host_payload_lifetime(bytes);
    }
    // This entry point is for the typed TapBlock observer only. The request
    // construction path supplies the erased owner; arbitrary client pointers
    // must never be passed to the typed identity predicate.
    void track_snapshot_payload(const Lease& lease,const std::shared_ptr<const void>& owner,
        std::size_t plane,const void* data,std::size_t bytes) {
        std::unique_lock lock(mutex_);require_open(lock);
        const auto& entry=find_locked(lease.ticket.request_id);
        if(!physical_bound_ || !entry.active || entry.generation!=lease.ticket.generation ||
            entry.acquisition!=lease.acquisition || entry.root!=lease.root)
            throw std::invalid_argument("snapshot payload tracking stale physical lease");
        resident_.track_host_payload_lifetime(owner,plane,data,bytes,
            [](const std::shared_ptr<const void>& value,const RetainedHostAllocationLedger& ledger,
                std::size_t index,const void* address,std::size_t extent) noexcept {
                return Exl3VeriCacheRequest::reserved_tap_payload_matches(value,ledger,index,address,extent);
            });
    }
    void transfer_retired_metadata(const Lease& lease,const Exl3ResourceInventory::Allocation& expected,
        const Exl3ResourceInventory::Allocation& tracking) {
        std::unique_lock lock(mutex_);require_open(lock);
        const auto& entry=find_locked(lease.ticket.request_id);
        if(!physical_bound_ || !entry.active || entry.generation!=lease.ticket.generation ||
            entry.acquisition!=lease.acquisition || entry.root!=lease.root)
            throw std::invalid_argument("metadata transfer stale physical lease");
        auto next=physical_resources_.transfer_metadata(expected,tracking);
        resident_.transfer_retired_metadata(expected,tracking);
        physical_resources_=std::move(next);
    }
    bool collect_retired_metadata(const Lease& lease,const Exl3ResourceInventory::Allocation& tracking,
        const std::weak_ptr<const void>& retired) {
        std::unique_lock lock(mutex_);require_open(lock);
        const auto& entry=find_locked(lease.ticket.request_id);
        if(!physical_bound_ || !entry.active || entry.generation!=lease.ticket.generation ||
            entry.acquisition!=lease.acquisition || entry.root!=lease.root)
            throw std::invalid_argument("metadata collection stale physical lease");
        auto next=physical_resources_.without_allocation(tracking);
        if(!resident_.collect_retired_metadata(tracking,retired))return false;
        physical_resources_=std::move(next);return true;
    }
    // One-shot source-test seam; no callback can reenter the coordinator mutex.
    // Invalid requests do not consume it; the next real residency transition does.
    void fail_next_residency_for_test(ResidencyFault fault) {
        std::unique_lock lock(mutex_);require_open(lock);residency_fault_=fault;
    }

    std::vector<Ticket> admit_batch(
        std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> roots) {
        if(roots.empty())throw std::invalid_argument("serving coordinator empty admission batch");
        if(roots.size()>policy_.logical_capacity)
            throw std::runtime_error("serving coordinator logical capacity exhausted");
        std::vector<Ticket> tickets(roots.size());
        admit_batch_into(roots,tickets);
        return tickets;
    }
    void admit_batch_into(
        std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> roots,
        std::span<Ticket> tickets) {
        if (roots.empty())
            throw std::invalid_argument("serving coordinator empty admission batch");
        if(roots.size()!=tickets.size())
            throw std::invalid_argument("serving coordinator admission output extent");
        for (const auto& root : roots) validate_root(root);
        std::unique_lock lock(mutex_);
        require_open(lock);
        if (roots.size() > policy_.logical_capacity - entries().size())
            throw std::runtime_error("serving coordinator logical capacity exhausted");
        const auto identity = roots.front()->state()->model_identity();
        if (model_identity_ && model_identity_ != identity)
            throw std::invalid_argument("serving coordinator model identity mismatch");
        for (const auto& root : roots)
            if (root->state()->model_identity() != identity)
                throw std::invalid_argument("serving coordinator batch identity mismatch");
        entries().reserve(entries().size() + roots.size());
        queue().reserve(entries().size() + roots.size());
        auto& storage=metadata_?*metadata_:fallback_metadata_;
        auto& replacements=storage.replacements;
        if(storage.publication_scratch_active)
            throw std::logic_error("serving coordinator publication scratch reentry");
        if(storage.publication_epoch==UINT64_MAX)
            throw std::overflow_error("serving coordinator publication scratch epoch");
        storage.publication_scratch_active=true;++storage.publication_epoch;
        struct ClearScratch {
            MetadataStorage& storage;
            ~ClearScratch(){storage.replacements.clear();storage.publication_scratch_active=false;}
        } clear{storage};
        replacements.reserve(roots.size());
        for (std::size_t index = 0; index < roots.size(); ++index) {
            const auto id = next_request_id_ + index;
            replacements.emplace_back(id, roots[index]);
        }
        const auto residency = replace_resident_locked(replacements, 0);
        (void)residency;
        for (std::size_t index=0;index<replacements.size();++index) {
            const auto& replacement=replacements[index];
            queue().push_back(replacement.first);
            entries().push_back(Entry{replacement.first,1,false,0,replacement.second});
            tickets[index]=Ticket{replacement.first,1};
        }
        next_request_id_ += roots.size();
        if (!model_identity_) model_identity_ = std::move(identity);
        high_admitted_ = std::max(high_admitted_, entries().size());
        high_queued_ = std::max(high_queued_, queue().size());
    }

    std::optional<Lease> acquire() {
        std::unique_lock lock(mutex_);
        require_open(lock);
        if (active_ >= policy_.physical_capacity || queue().empty()) return {};
        const auto id = queue().front();
        auto& entry = find_locked(id);
        if (entry.active) throw std::logic_error("serving coordinator queued active entry");
        if (entry.acquisition == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("serving coordinator acquisition exhausted");
        queue().erase(queue().begin());
        ++entry.acquisition;
        entry.active = true;
        ++active_;
        high_active_ = std::max(high_active_, active_);
        return Lease{Ticket{entry.id, entry.generation}, entry.root, entry.acquisition};
    }

    // The caller must already own the physical lane. Reserve all logical
    // suffix storage before invalidating its ordinary host Lease.
    DeviceLogicalLease enter_device_logical(const Lease& lease,
        std::shared_ptr<const void> physical_owner,int initial_frontier,
        std::size_t output_token_capacity,int max_context) {
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto& entry=require_lease_locked(lease);
        if(!physical_owner || max_context<=0 ||
           !entry.root->state()->native_extent_valid(max_context) ||
           initial_frontier!=entry.root->state()->position() ||
           initial_frontier<0 || initial_frontier>max_context ||
           output_token_capacity==0 ||
           output_token_capacity>static_cast<std::size_t>(
               max_context-initial_frontier))
            throw std::invalid_argument("serving coordinator device logical entry extent/owner");
        if(entry.generation==UINT64_MAX)
            throw std::overflow_error("serving coordinator device logical generation exhausted");
        std::vector<std::int64_t> suffix;
        suffix.reserve(output_token_capacity);
        DeviceLogicalLease result{
            Ticket{entry.id,entry.generation+1},entry.root,
            physical_owner,entry.acquisition,initial_frontier};
        entry.device_suffix=std::move(suffix);
        entry.device_owner=std::move(physical_owner);
        entry.device_initial_frontier=initial_frontier;
        entry.device_capacity=output_token_capacity;
        entry.device_mode=true;
        ++entry.generation;
        return result;
    }

    // The verified numerical operation and any rollback are caller-owned.
    // This critical section only commits its ordered token result. No host
    // child or resident-set update is made for an intermediate window.
    DeviceLogicalPublication publish_device_logical_window(
        const DeviceLogicalLease& lease,std::span<const std::int64_t> tokens) {
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto& entry=require_device_logical_lease_locked(lease);
        if(tokens.empty() || tokens.size()>128 ||
           tokens.size()>entry.device_capacity-entry.device_suffix.size() ||
           std::any_of(tokens.begin(),tokens.end(),
               [](std::int64_t token){return token<0;}))
            throw std::invalid_argument("serving coordinator device logical window tokens/extent");
        if(entry.generation==UINT64_MAX || entry.publications==UINT64_MAX ||
           publications_==UINT64_MAX || next_publication_sequence_==UINT64_MAX)
            throw std::overflow_error("serving coordinator device logical publication exhausted");
        DeviceLogicalPublication result{
            DeviceLogicalLease{Ticket{entry.id,entry.generation+1},
                entry.root,entry.device_owner,entry.acquisition,
                lease.frontier+static_cast<int>(tokens.size())},
            entry.device_suffix.size()+tokens.size(),next_publication_sequence_};
        // Capacity was reserved on entry. Inserting trivial token values does
        // not allocate; all validation and result construction precedes it.
        entry.device_suffix.insert(entry.device_suffix.end(),tokens.begin(),tokens.end());
        ++entry.generation;
        ++entry.publications;
        ++publications_;
        ++next_publication_sequence_;
        return result;
    }

    // The caller supplies a fully prepared immutable exact host child. Only
    // after its lineage and entire accumulated suffix match do we replace the
    // initial resident root once and restore an ordinary host Lease.
    Lease materialize_device_logical(const DeviceLogicalLease& lease,
        std::shared_ptr<const Exl3VeriCacheRequest> child) {
        validate_root(child);
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto& entry=require_device_logical_lease_locked(lease);
        if(entry.generation==UINT64_MAX || entry.publications==UINT64_MAX ||
           publications_==UINT64_MAX || next_publication_sequence_==UINT64_MAX)
            throw std::overflow_error("serving coordinator device logical materialization exhausted");
        if(entry.device_suffix.empty() || !child->is_child_of(*entry.root) ||
           child->state()->model_identity()!=entry.root->state()->model_identity() ||
           child->state()->position()!=lease.frontier ||
           child->token_suffix(entry.device_initial_frontier)!=entry.device_suffix)
            throw std::invalid_argument("serving coordinator device logical child lineage/tokens");
        const std::array<std::pair<std::uint64_t,
            std::shared_ptr<const Exl3VeriCacheRequest>>,1> replacement{{{entry.id,child}}};
        (void)replace_resident_locked(replacement,0);
        entry.root=std::move(child);
        entry.device_owner.reset();
        std::vector<std::int64_t>{}.swap(entry.device_suffix);
        entry.device_initial_frontier=0;
        entry.device_capacity=0;
        entry.device_mode=false;
        ++entry.generation;
        ++entry.publications;
        ++publications_;
        ++next_publication_sequence_;
        return Lease{Ticket{entry.id,entry.generation},entry.root,entry.acquisition};
    }

    // Active logical cancellation is allowed only after the physical worker
    // has finished or rolled back its operation. The initial root is removed
    // from residency by the same one-shot cancellation path as host leases.
    void cancel_device_logical(const DeviceLogicalLease& lease,
        bool worker_complete) {
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto& entry=require_device_logical_lease_locked(lease);
        if(!worker_complete)
            throw std::logic_error("serving coordinator device logical cancellation needs worker boundary");
        const auto id=entry.id;
        (void)replace_resident_locked(std::span<const std::pair<std::uint64_t,
            std::shared_ptr<const Exl3VeriCacheRequest>>>{},id);
        --active_;
        entries().erase(std::find_if(entries().begin(),entries().end(),
            [&](const Entry& candidate){return candidate.id==id;}));
        ++cancellations_;
        if(entries().empty())model_identity_.reset();
    }

    void yield(const Lease& lease) {
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto& entry = require_lease_locked(lease);
        if(queue().size()==queue().capacity())
            throw std::logic_error("serving coordinator yield exceeds reserved queue");
        queue().push_back(entry.id);
        entry.active = false;
        --active_;
        high_queued_ = std::max(high_queued_, queue().size());
    }

    std::vector<Publication> publish_batch(
        std::span<const Lease> leases,
        std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> roots,
        std::span<const std::int64_t> tokens) {
        if(leases.empty() || leases.size()!=roots.size() || leases.size()!=tokens.size())
            throw std::invalid_argument("serving coordinator publication extent");
        if(leases.size()>policy_.physical_capacity)
            throw std::invalid_argument("serving coordinator publication exceeds physical capacity");
        std::vector<Publication> output(leases.size());
        publish_batch_into(leases,roots,tokens,output);
        return output;
    }
    void publish_batch_into(
        std::span<const Lease> leases,
        std::span<const std::shared_ptr<const Exl3VeriCacheRequest>> roots,
        std::span<const std::int64_t> tokens,std::span<Publication> output) {
        if (leases.empty() || leases.size() != roots.size() ||
            leases.size() != tokens.size() || leases.size()!=output.size())
            throw std::invalid_argument("serving coordinator publication extent");
        std::unique_lock lock(mutex_);
        require_open(lock);
        if (leases.size() > policy_.physical_capacity)
            throw std::invalid_argument("serving coordinator publication exceeds physical capacity");
        auto& storage=metadata_?*metadata_:fallback_metadata_;
        auto& selected=storage.selected;
        auto& replacements=storage.replacements;
        if(storage.publication_scratch_active)
            throw std::logic_error("serving coordinator publication scratch reentry");
        if(storage.publication_epoch==UINT64_MAX)
            throw std::overflow_error("serving coordinator publication scratch epoch");
        storage.publication_scratch_active=true;++storage.publication_epoch;
        // The coordinator lock makes this workspace exclusive. Every exit,
        // including validation/residency failure, releases borrowed roots.
        struct ClearScratch {
            MetadataStorage& storage;
            ~ClearScratch(){storage.selected.clear();storage.replacements.clear();
                storage.publication_scratch_active=false;}
        } clear{storage};
        selected.reserve(leases.size());
        replacements.reserve(leases.size());
        for (std::size_t index = 0; index < leases.size(); ++index) {
            auto& entry = require_lease_locked(leases[index]);
            if (std::find(selected.begin(), selected.end(), &entry) != selected.end())
                throw std::invalid_argument("serving coordinator duplicate publication lease");
            validate_child(entry.root, roots[index], tokens[index]);
            selected.push_back(&entry);
            replacements.emplace_back(entry.id, roots[index]);
        }
        const auto resident = replace_resident_locked(replacements, 0);
        for (std::size_t index = 0; index < selected.size(); ++index) {
            auto& entry = *selected[index];
            entry.root = roots[index];
            ++entry.generation;
            ++entry.publications;
            ++publications_;
            output[index].lease = Lease{
                Ticket{entry.id, entry.generation}, entry.root, entry.acquisition};
            output[index].token = tokens[index];
            output[index].sequence = next_publication_sequence_++;
            output[index].residency = resident;
        }
    }

    Publication publish(const Lease& lease,
        std::shared_ptr<const Exl3VeriCacheRequest> root,
        std::int64_t token) {
        const std::array<Lease,1> leases{lease};
        const std::array<std::shared_ptr<const Exl3VeriCacheRequest>,1> roots{
            std::move(root)};
        const std::array<std::int64_t,1> tokens{token};
        std::array<Publication,1> output;
        publish_batch_into(leases,roots,tokens,output);
        return std::move(output.front());
    }

    // Coherent read-only acquisition snapshot for bounded shared compute.
    // Caller still owns the physical lanes; this never authorizes publication.
    bool compute_leases_current(std::span<const Lease> leases) const {
        std::unique_lock lock(mutex_);
        if(closed_ || leases.empty() || leases.size()>policy_.physical_capacity) return false;
        for(std::size_t i=0;i<leases.size();++i) {
            const auto& lease=leases[i];
            for(std::size_t j=0;j<i;++j) if(leases[j].ticket.request_id==lease.ticket.request_id) return false;
            const auto found=std::find_if(entries().begin(),entries().end(),[&](const Entry& e){return e.id==lease.ticket.request_id;});
            if(found==entries().end() || found->device_mode || !found->active || found->generation!=lease.ticket.generation ||
               found->acquisition!=lease.acquisition || found->root!=lease.root) return false;
        }
        return true;
    }

    // One verified causal window is one atomic resident/publication transition.
    // The child already contains the repaired L2 state and matching draft root.
    WindowPublication publish_window(const Lease& lease,
        std::shared_ptr<const Exl3VeriCacheRequest> root,
        std::span<const std::int64_t> tokens) {
        validate_root(root);
        // Model windows are bounded by their verifier; explicit frontend
        // thinking-control sequences can span several native calls.
        if (tokens.empty() || tokens.size() > 128)
            throw std::invalid_argument("serving coordinator window extent");
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto& entry = require_lease_locked(lease);
        if (!root->is_child_of(*entry.root) ||
            root->state()->model_identity() != entry.root->state()->model_identity() ||
            root->state()->position() != entry.root->state()->position() + tokens.size() ||
            root->token_suffix(entry.root->state()->position()) !=
                std::vector<std::int64_t>(tokens.begin(), tokens.end()))
            throw std::invalid_argument("serving coordinator window lineage/tokens");
        WindowPublication result;
        result.tokens.assign(tokens.begin(), tokens.end());
        const std::array<std::pair<std::uint64_t,
            std::shared_ptr<const Exl3VeriCacheRequest>>,1> replacement{{{entry.id, root}}};
        result.residency = replace_resident_locked(replacement, 0);
        entry.root = std::move(root);
        ++entry.generation;
        ++entry.publications;
        ++publications_;
        result.lease = Lease{Ticket{entry.id, entry.generation}, entry.root, entry.acquisition};
        result.sequence = next_publication_sequence_++;
        return result;
    }

    // Active cancellation is legal only after the worker has completed or
    // rolled back its numerical operation. The generation-stamped lease then
    // becomes stale and cannot publish a late result.
    void cancel(Ticket ticket, bool worker_complete = false) {
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto it = std::find_if(entries().begin(), entries().end(),
            [&](const Entry& entry) { return entry.id == ticket.request_id; });
        if (it == entries().end() || it->generation != ticket.generation)
            throw std::invalid_argument("serving coordinator stale cancellation");
        if (it->active && !worker_complete)
            throw std::logic_error("serving coordinator active cancellation needs worker boundary");
        const auto id = it->id;
        (void)replace_resident_locked(std::span<const std::pair<std::uint64_t,
            std::shared_ptr<const Exl3VeriCacheRequest>>>{}, id);
        queue().erase(std::remove(queue().begin(), queue().end(), id), queue().end());
        if (it->active) --active_;
        entries().erase(it);
        ++cancellations_;
        if (entries().empty()) model_identity_.reset();
    }

    void complete(const Lease& lease) {
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto& entry = require_lease_locked(lease);
        const auto id = entry.id;
        (void)replace_resident_locked(std::span<const std::pair<std::uint64_t,
            std::shared_ptr<const Exl3VeriCacheRequest>>>{}, id);
        --active_;
        entries().erase(std::find_if(entries().begin(), entries().end(),
            [&](const Entry& candidate) { return candidate.id == id; }));
        ++completions_;
        if (entries().empty()) model_identity_.reset();
    }

    void complete_retired(const Lease& lease,const Retirement& retirement) {
        require_retirement(lease,retirement);
        complete(lease);
    }

    void cancel_retired(const Lease& lease,const Retirement& retirement) {
        require_retirement(lease,retirement);
        cancel(lease.ticket,true);
    }

    void yield_retired(const Lease& lease,const Retirement& retirement) {
        require_retirement(lease,retirement);
        yield(lease);
    }

    // Replace one worker-complete active request with one prepared request at
    // the FIFO tail. The resident set sees the final old/new union once, so
    // pages required by both roots remain locked throughout the transition.
    Turnover complete_and_admit(const Lease& lease,
        std::shared_ptr<const Exl3VeriCacheRequest> root) {
        validate_root(root);
        std::unique_lock lock(mutex_);
        require_open(lock);
        auto& retired = require_lease_locked(lease);
        if (root->state()->model_identity() != model_identity_)
            throw std::invalid_argument(
                "serving coordinator turnover identity mismatch");

        // Admission already reserved return slots for every active entry.
        // Refuse a broken extent before changing either resident authority
        // or the completed entry; turnover itself never grows the queue.
        if(queue().size()==queue().capacity())
            throw std::logic_error("serving coordinator turnover exceeds reserved queue");
        const Ticket completed{retired.id, retired.generation};
        const Ticket admitted{next_request_id_, 1};
        const std::array<std::pair<std::uint64_t,
            std::shared_ptr<const Exl3VeriCacheRequest>>,1> replacement{{
                {admitted.request_id, root}}};
        const auto residency = replace_resident_locked(
            replacement, retired.id);
        retired = Entry{admitted.request_id, admitted.generation,
                        false, 0, std::move(root)};
        queue().push_back(admitted.request_id);
        ++next_request_id_;
        --active_;
        ++completions_;
        high_queued_ = std::max(high_queued_, queue().size());
        high_admitted_ = std::max(high_admitted_, entries().size());
        return Turnover{completed, admitted, residency};
    }

    // Serialize retention requirements with queue, publication and completion.
    // Callers may add required entries, but cannot make a live logical root
    // evictable. Reconcile the resident union in the same coordinator critical
    // section after a current policy decision. Logical roots removed by the
    // index then stop contributing, while live device-fill/read bindings keep
    // their exact host pages owned and locked until their own lifetime ends.
    Exl3VeriCachePrefixIndex::PolicyTrim trim_prefix_retention(
        std::uint64_t available_physical_bytes,
        std::span<const Exl3VeriCachePrefixIndex::RetentionDecision> supplied) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(supplied.size()>64)
            throw std::invalid_argument("coordinator retention decision bound exceeded");
        std::array<Exl3VeriCachePrefixIndex::RetentionDecision,64> decisions{};
        for(std::size_t i=0;i<supplied.size();++i) {
            decisions[i]=supplied[i];
            for(const auto& entry:entries())
                if(entry.root==decisions[i].root) {decisions[i].required=true;break;}
        }
        auto result=cache_.trim_with_retention_policy(available_physical_bytes,
            std::span<const Exl3VeriCachePrefixIndex::RetentionDecision>(decisions).first(supplied.size()));
        if(result.inputs_current)replace_resident_locked({},0);
        return result;
    }

    // Cache admission only, not logical request admission or a new residency
    // guarantee. Normal publication still acquires its full resident coverage.
    Exl3VeriCachePrefixIndex::Admission admit_prefix_input_with_retention(
        std::shared_ptr<const Exl3VeriCacheRequest> root,
        std::span<const std::int64_t> input,std::uint64_t available_physical_bytes,
        std::span<const Exl3VeriCachePrefixIndex::RetentionDecision> supplied) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(supplied.size()>64)
            throw std::invalid_argument("coordinator retention decision bound exceeded");
        std::array<Exl3VeriCachePrefixIndex::RetentionDecision,64> decisions{};
        for(std::size_t i=0;i<supplied.size();++i) {
            decisions[i]=supplied[i];
            for(const auto& entry:entries())
                if(entry.root==decisions[i].root) {decisions[i].required=true;break;}
        }
        return cache_.admit_input_authority(std::move(root),input,available_physical_bytes,
            std::span<const Exl3VeriCachePrefixIndex::RetentionDecision>(decisions).first(supplied.size()));
    }

    std::uint64_t tighten_metadata_headroom_for_test(std::uint64_t headroom) {
        std::unique_lock lock(mutex_);require_open(lock);
        if(!physical_bound_ || active_ || !entries().empty() || allocating_)
            throw std::logic_error("metadata ceiling requires idle bound coordinator");
        return resident_.tighten_metadata_headroom_for_test(headroom);
    }
    Stats stats() const {
        std::unique_lock lock(mutex_);
        const bool unresolved=startup_retirement_failed_.load(std::memory_order_acquire);
        return Stats{queue().size(), active_, entries().size(),
            policy_.logical_capacity, policy_.physical_capacity,
            publications_, cancellations_, completions_,
            resident_updates_, resident_new_locked_bytes_,
            resident_unlocked_bytes_, resident_inventory_ms_,
            resident_registry_ms_,
            resident_.locked_page_bytes(), closed_,(allocating_ || unresolved)?std::nullopt:
                std::optional<Exl3ResourceInventory::Totals>{resident_.retained_resource_units()},closing_started_ && !closed_,unresolved};
    }
    std::vector<Exl3ResourceInventory::Attribution> resource_attribution_for_test() const {
        std::lock_guard lock(mutex_);return physical_resources_.attribution_snapshot();
    }
    std::size_t high_queued() const {
        std::unique_lock lock(mutex_); return high_queued_;
    }
    std::size_t queue_capacity_for_test() const {
        std::unique_lock lock(mutex_);return queue().capacity();
    }
    std::size_t high_active() const {
        std::unique_lock lock(mutex_); return high_active_;
    }
    std::size_t high_admitted() const {
        std::unique_lock lock(mutex_); return high_admitted_;
    }

    // A failed factory may retain allocations outside the rolled-back inventory.
    // This permanent seal prevents that authority from admitting more work.
    void seal_failed_startup_retirement() noexcept {
        startup_retirement_failed_.store(true,std::memory_order_release);
        resident_.seal_external_retirement_failure();
    }
    void close() {
        std::unique_lock lock(mutex_);
        if (closed_) return;
        require_open(lock,true);
        if (active_ != 0)
            throw std::logic_error("serving coordinator close with active lease");
        closing_started_=true;
        queue().clear();
        entries().clear();
        model_identity_.reset();
        resident_.close();
        metadata_.reset();physical_resources_={};host_sources_.reset();
        closed_ = true;
    }

private:
    struct Entry {
        std::uint64_t id = 0;
        std::uint64_t generation = 0;
        bool active = false;
        std::uint64_t publications = 0;
        std::shared_ptr<const Exl3VeriCacheRequest> root;
        std::uint64_t acquisition = 0;
        bool device_mode = false;
        std::shared_ptr<const void> device_owner;
        int device_initial_frontier = 0;
        std::size_t device_capacity = 0;
        std::vector<std::int64_t> device_suffix;
    };

    static void validate_root(
        const std::shared_ptr<const Exl3VeriCacheRequest>& root) {
        if (!root || !root->state())
            throw std::invalid_argument("serving coordinator exact root missing");
        const auto tokens = root->token_suffix();
        if (tokens.empty() || static_cast<int>(tokens.size()) !=
                root->state()->position())
            throw std::invalid_argument("serving coordinator token/state extent");
    }
    static void validate_child(
        const std::shared_ptr<const Exl3VeriCacheRequest>& parent,
        const std::shared_ptr<const Exl3VeriCacheRequest>& child,
        std::int64_t token) {
        validate_root(child);
        if (!child->is_child_of(*parent) ||
            child->state()->model_identity() != parent->state()->model_identity() ||
            child->state()->position() != parent->state()->position() + 1)
            throw std::invalid_argument("serving coordinator publication lineage");
        const auto suffix = child->token_suffix(parent->state()->position());
        if (suffix.size() != 1 || suffix.front() != token)
            throw std::invalid_argument("serving coordinator publication token");
    }
    void require_open(std::unique_lock<std::mutex>& lock,bool closing=false) const {
        if(runtime_allocating_) {
            if(allocation_thread_==std::this_thread::get_id())
                throw std::logic_error("runtime allocation factory attempted coordinator mutation");
            allocation_changed_.wait(lock,[&]{return !runtime_allocating_;});
        }
        if (closed_) throw std::logic_error("serving coordinator closed");
        if(!closing && closing_started_)throw std::logic_error("serving coordinator close in progress");
        if(allocating_)throw std::logic_error("serving coordinator startup allocation in flight");
        if(!closing && startup_retirement_failed_.load(std::memory_order_acquire))
            throw std::logic_error("serving coordinator sealed after failed startup retirement");
    }
    Entry& find_locked(std::uint64_t id) {
        const auto found = std::find_if(entries().begin(), entries().end(),
            [&](const Entry& entry) { return entry.id == id; });
        if (found == entries().end())
            throw std::invalid_argument("serving coordinator request missing");
        return *found;
    }
    Entry& require_lease_locked(const Lease& lease) {
        auto& entry = find_locked(lease.ticket.request_id);
        if (entry.device_mode || !entry.active || entry.generation != lease.ticket.generation ||
            entry.root != lease.root || entry.acquisition != lease.acquisition)
            throw std::invalid_argument("serving coordinator stale lease");
        return entry;
    }
    Entry& require_device_logical_lease_locked(const DeviceLogicalLease& lease) {
        auto& entry=find_locked(lease.ticket.request_id);
        const auto& owner=lease.physical_owner;
        const auto& held=entry.device_owner;
        if(!entry.device_mode || !entry.active ||
           entry.generation!=lease.ticket.generation ||
           entry.acquisition!=lease.acquisition ||
           entry.root!=lease.initial_root || !owner || !held ||
           owner.get()!=held.get() || owner.owner_before(held) ||
           held.owner_before(owner) ||
           lease.frontier!=entry.device_initial_frontier+
               static_cast<int>(entry.device_suffix.size()))
            throw std::invalid_argument("serving coordinator stale device logical lease");
        return entry;
    }
    static void require_retirement(const Lease& lease,
        const Retirement& retirement) {
        if(!retirement.valid() ||
           retirement.ticket_.request_id!=lease.ticket.request_id ||
           retirement.ticket_.generation!=lease.ticket.generation ||
           retirement.acquisition_!=lease.acquisition)
            throw std::invalid_argument(
                "serving coordinator stale/missing physical retirement");
    }
    Exl3HostResidentSet::Stats replace_resident_locked(
        std::span<const std::pair<std::uint64_t,
            std::shared_ptr<const Exl3VeriCacheRequest>>> replacements,
        std::uint64_t removed_id, bool request_admission = false) {
        const auto inventory_started = std::chrono::steady_clock::now();
        auto& storage=metadata_?*metadata_:fallback_metadata_;
        auto& roots=storage.roots;
        if(storage.root_scratch_active)
            throw std::logic_error("serving coordinator root scratch reentry");
        if(storage.root_epoch==UINT64_MAX)
            throw std::overflow_error("serving coordinator root scratch epoch");
        storage.root_scratch_active=true;++storage.root_epoch;
        struct ClearRoots {
            MetadataStorage& storage;
            ~ClearRoots(){storage.roots.clear();storage.root_scratch_active=false;}
        } clear{storage};
        roots.reserve(resident_root_capacity());
        cache_.copy_roots_into(roots);
        for (const auto& entry : entries()) {
            if (entry.id == removed_id) continue;
            const auto replacement = std::find_if(replacements.begin(),
                replacements.end(), [&](const auto& value) {
                    return value.first == entry.id;
                });
            roots.push_back(replacement == replacements.end() ?
                entry.root : replacement->second);
        }
        for (const auto& replacement : replacements)
            if (std::none_of(entries().begin(), entries().end(),
                    [&](const Entry& entry) {
                        return entry.id == replacement.first;
                    }))
                roots.push_back(replacement.second);
        const auto source_slots=host_sources_?host_sources_->size():0;
        if(source_slots>std::numeric_limits<std::size_t>::max()-roots.size())
            throw std::overflow_error("coordinator snapshot owner capacity overflow");
        // The fixed source table bounds weak bindings, including bindings
        // that expire before collection. No owner append grows this array.
        Exl3VeriCacheRequest::visit_metadata_allocations(roots,
            [&](const Exl3VeriCacheRequest::MetadataAllocation& allocation) {
                resident_.admit_host_metadata_lifetime(allocation.owner,
                    allocation.bytes,allocation.matches,allocation.attach);
            });
        std::size_t range_count=0;
        if(!roots.empty())Exl3VeriCacheRequest::visit_host_allocations(roots,
            [&](const void*,std::size_t bytes) {
                if(!bytes)return;
                if(range_count==std::numeric_limits<std::size_t>::max())
                    throw std::overflow_error("coordinator snapshot range count overflow");
                ++range_count;
            },false);
        // Count all registered slots conservatively: a weak lifetime may end
        // between planning and collection, but cannot add another table slot.
        if(host_sources_)for(const auto& binding:*host_sources_) {
            if(binding.count>binding.ranges.size())
                throw std::logic_error("runtime host source range extent");
            if(binding.count>std::numeric_limits<std::size_t>::max()-range_count)
                throw std::overflow_error("coordinator snapshot range count overflow");
            range_count+=binding.count;
        }
        const auto required=Exl3HostResidentSet::Snapshot::descriptor_requirement(roots.size()+source_slots,range_count);
        auto fault=ResidencyFault::none;
        const auto stats=resident_.replace_reserved_snapshot(required,[&](std::uint64_t configuration) {
        if(configuration!=required.configuration)throw std::logic_error("coordinator snapshot configuration changed");
        Exl3HostResidentSet::Snapshot snapshot;
        snapshot.resources=physical_resources_;
        snapshot.owners.reserve(roots.size()+source_slots);
        snapshot.reserve_regions(range_count);
        if(host_sources_)for(const auto& binding:*host_sources_)if(auto lifetime=binding.lifetime.lock()) {
            auto source=binding.source.lock();
            if(!source)throw std::logic_error("live runtime host source lost backing");
            snapshot.owners.push_back(std::move(source));
            for(std::size_t i=0;i<binding.count;++i) {
                const auto& range=binding.ranges[i];
                snapshot.add(reinterpret_cast<const void*>(range.first),range.second-range.first);
            }
        }
        for (const auto& root : roots) snapshot.owners.push_back(root);
        if (!roots.empty()) Exl3VeriCacheRequest::visit_host_allocations(roots,
            [&](const void* data, std::size_t bytes) {
                snapshot.add(data, bytes);
            }, false);
        resident_inventory_ms_ += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inventory_started).count();
        return snapshot;
        },[&] {
        // The Engine admission seam must not be consumed by preparation-time
        // resource binding or another request's publication/retirement.
        fault=residency_fault_==ResidencyFault::page_budget_before_commit && !request_admission
            ?ResidencyFault::none:std::exchange(residency_fault_,ResidencyFault::none);
        if(fault==ResidencyFault::page_budget_before_commit)
            throw Exl3HostResidentBudgetExhausted();
        if(fault==ResidencyFault::before_commit)
            throw std::runtime_error("injected coordinator before resident commit");
        },[&](std::size_t locks){
            if(fault==ResidencyFault::after_first_lock && locks==1)
                throw std::runtime_error("injected coordinator after resident lock");
        });
        ++resident_updates_;
        resident_new_locked_bytes_ += stats.new_locked_bytes;
        resident_unlocked_bytes_ += stats.unlocked_bytes;
        resident_registry_ms_ += stats.elapsed_ms;
        return stats;
    }

    Exl3VeriCacheServingPrefixCache& cache_;
    Policy policy_;
    mutable std::mutex mutex_;
    mutable std::condition_variable allocation_changed_;
    bool runtime_allocating_=false;
    std::thread::id allocation_thread_{};
    std::thread::id constructor_thread_{};
    Exl3ResourceInventory::Totals constructor_ceiling_{},constructor_issued_{};
    RetainedDeviceLedger constructor_device_;
    RetainedCudaRegistrationLedger constructor_registration_;
    RetainedDescriptorLedger constructor_metadata_;
    Exl3HostResidentSet resident_;
    Exl3ResourceInventory physical_resources_;
    struct HostSourceBinding {
        std::weak_ptr<const void> lifetime;
        std::weak_ptr<const Exl3ExactKVPage> source;
        std::array<Exl3HostResidentSet::Range,33> ranges{};
        std::size_t count=0;
    };
    using HostSourceTable=std::array<HostSourceBinding,64>;
    std::shared_ptr<HostSourceTable> host_sources_;
    bool physical_bound_=false;
    bool allocating_=false;
    std::atomic<bool> startup_retirement_failed_{false};
    ResidencyFault residency_fault_=ResidencyFault::none;
    std::shared_ptr<const void> model_identity_;
    using Replacement=std::pair<std::uint64_t,std::shared_ptr<const Exl3VeriCacheRequest>>;
    struct MetadataStorage {
        std::vector<Entry> entries;
        std::vector<std::uint64_t> queue;
        std::vector<Entry*> selected;
        std::vector<Replacement> replacements;
        std::vector<std::shared_ptr<const Exl3VeriCacheRequest>> roots;
        std::uint64_t publication_epoch=0,root_epoch=0;
        bool publication_scratch_active=false,root_scratch_active=false;
    };
    MetadataStorage fallback_metadata_;
    std::shared_ptr<MetadataStorage> metadata_;
    std::vector<Entry>& entries() noexcept {return metadata_?metadata_->entries:fallback_metadata_.entries;}
    const std::vector<Entry>& entries() const noexcept {return metadata_?metadata_->entries:fallback_metadata_.entries;}
    std::vector<std::uint64_t>& queue() noexcept {return metadata_?metadata_->queue:fallback_metadata_.queue;}
    const std::vector<std::uint64_t>& queue() const noexcept {return metadata_?metadata_->queue:fallback_metadata_.queue;}

    std::uint64_t next_request_id_ = 1;
    std::uint64_t next_publication_sequence_ = 1;
    std::uint64_t publications_ = 0;
    std::uint64_t cancellations_ = 0;
    std::uint64_t completions_ = 0;
    std::uint64_t resident_updates_ = 0;
    std::uint64_t resident_new_locked_bytes_ = 0;
    std::uint64_t resident_unlocked_bytes_ = 0;
    double resident_inventory_ms_ = 0.0;
    double resident_registry_ms_ = 0.0;
    std::size_t active_ = 0;
    std::size_t high_queued_ = 0;
    std::size_t high_active_ = 0;
    std::size_t high_admitted_ = 0;
    bool closed_ = false;
    bool closing_started_ = false;
};

} // namespace ninfer::exl3
