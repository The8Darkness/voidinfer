#pragma once

// Windows DRAM residency for immutable host payloads. One serialized registry
// per process owns VirtualLock and its reversible process working-set quota.
// Owners keep allocations alive through unlock. Windows has no page lock count:
// page unions, not independent per-request VirtualUnlock calls, handle sharing.
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include "exl3/resource_inventory.h"
#include "exl3/merge_resident_ranges.h"
#include "exl3/retained_descriptor_ledger.h"
#include "exl3/bounded_shared_owner.h"
#include "exl3/reserved_host_payload_union.h"
#include <optional>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

namespace ninfer::exl3 {
// Configured page-budget refusal occurs before ownership publication or locks.
// Keep it distinct from OS failures and failed rollback/retirement.
class Exl3HostResidentBudgetExhausted : public std::runtime_error {
public:
    Exl3HostResidentBudgetExhausted()
        :std::runtime_error("resident page budget exhausted including in-flight roots") {}
};
class Exl3HostResidentSet {
public:
    using Range=std::pair<std::uintptr_t,std::uintptr_t>;
    struct Snapshot {
        Exl3ResourceInventory resources;
        std::vector<std::shared_ptr<const void>> owners;
        std::vector<Range> regions;
        bool fixed_regions=false;
        std::size_t region_limit=0;
        struct DescriptorBacking {
            std::vector<std::shared_ptr<const void>> owners;
            std::vector<Range> regions;
        };
        std::shared_ptr<const DescriptorBacking> descriptor_backing;
        static Exl3ResourceInventory::Requirement descriptor_requirement(
            std::size_t owner_capacity,std::size_t range_capacity) {
            Exl3ResourceInventory::Requirement required;
            required.configuration=0x534e415044455343ULL;
            using Domain=Exl3ResourceInventory::Domain;
            required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<DescriptorBacking>());
            if(owner_capacity)required.add(Domain::host_metadata,owner_capacity,sizeof(std::shared_ptr<const void>));
            if(range_capacity)required.add(Domain::host_metadata,range_capacity,sizeof(Range));
            return required;
        }
        const std::vector<std::shared_ptr<const void>>& payload_owners() const noexcept {
            return descriptor_backing?descriptor_backing->owners:owners;
        }
        void retain_descriptors() {
            if(descriptor_backing || (!owners.capacity() && !regions.capacity()))return;
            const auto required=descriptor_requirement(owners.capacity(),regions.capacity());
            using Domain=Exl3ResourceInventory::Domain;
            auto backing=make_bounded_shared<DescriptorBacking>();
            backing->owners=std::move(owners);backing->regions=std::move(regions);
            resources.add({backing,0,Domain::host_metadata,required.units[static_cast<unsigned>(Domain::host_metadata)]});
            descriptor_backing=std::move(backing);
        }
        void reserve_regions(std::size_t count) {
            if(!regions.empty() || fixed_regions || descriptor_backing)
                throw std::logic_error("resident range reservation already started");
            regions.reserve(count);region_limit=count;fixed_regions=true;
        }
        void add(const void* data,std::size_t bytes) {
            if(!bytes) return;
            if(descriptor_backing)throw std::logic_error("resident snapshot descriptors already retained");
            const auto first=reinterpret_cast<std::uintptr_t>(data);
            if(!first || bytes>std::numeric_limits<std::uintptr_t>::max()-first)
                throw std::invalid_argument("resident payload address overflow");
            if(fixed_regions && (regions.size()>=region_limit || regions.size()>=regions.capacity()))
                throw std::length_error("resident range reservation exhausted");
            regions.emplace_back(first,first+bytes);
        }
    };
    struct Stats {
        std::uint64_t page_bytes=0,new_locked_bytes=0,unlocked_bytes=0,lock_calls=0;
        std::uint64_t transition_peak_page_bytes=0,source_revision=0,committed_revision=0;
        double elapsed_ms=0;
        Exl3ResourceInventory::Totals resource_units{},transition_peak_resource_units{};
    };
    // One seal covers the actual page union plus declared retained physical
    // owners. Declarations are payload accounting, not measured physical peaks.
    class Transition {
        friend class Exl3HostResidentSet;
        const Exl3HostResidentSet* authority_=nullptr;
        std::shared_ptr<const int> identity_;
        std::uint64_t revision_=0,peak_=0;
        Exl3ResourceInventory::Totals resource_peak_{};
        // Declared before owned buffers so destruction releases credit last.
        std::optional<RetainedDescriptorLedger::Ticket> descriptor_credit_;
        std::uint64_t scratch_bytes_=0;
        Snapshot next_;
        std::vector<Range> desired_,added_,removed_;
        bool consumed_=false;
        Transition()=default;
    public:
        Transition(const Transition&)=delete;
        Transition& operator=(const Transition&)=delete;
        Transition(Transition&& other) noexcept
            :authority_(std::exchange(other.authority_,nullptr)),identity_(std::move(other.identity_)),revision_(other.revision_),peak_(other.peak_),
             resource_peak_(other.resource_peak_),descriptor_credit_(std::move(other.descriptor_credit_)),
             scratch_bytes_(other.scratch_bytes_),next_(std::move(other.next_)),desired_(std::move(other.desired_)),
             added_(std::move(other.added_)),removed_(std::move(other.removed_)),consumed_(other.consumed_) {}
        std::uint64_t peak_page_bytes() const noexcept {return peak_;}
        std::uint64_t source_revision() const noexcept {return revision_;}
        const Exl3ResourceInventory::Totals& peak_resource_units_for_test() const noexcept {
            return resource_peak_;
        }
        const Range* range_storage_for_test() const noexcept {return desired_.data();}
        std::size_t range_capacity_for_test() const noexcept {return desired_.capacity();}
    };
private:
    struct PayloadAccounting {
        RetainedHostAllocationLedger ledger;
        ReservedHostPayloadTracker<4096> tracker;
        explicit PayloadAccounting(std::shared_ptr<RetainedHostAllocationLedger::State> state)
            :ledger(std::move(state)) {}
    };
    inline static std::atomic_bool process_owned_=false;
    // There can be only one live registry. Failed teardown permanently retains
    // its complete bundle and leaves process_owned_ sealed, so this fixed slot
    // cannot fill and then trigger replacement allocations or forced release.
    struct FailedCloseBundle {
        Snapshot held;
        Exl3ResourceInventory failed_factory;
        Exl3ResourceInventory::Allocation failed_allocation;
        std::vector<Range> pages;
        RetainedDescriptorLedger assessments,retired;
        RetainedDeviceLedger retired_device;
        RetainedCudaRegistrationLedger retired_registration;
        std::shared_ptr<PayloadAccounting> payload_accounting;
        Exl3ResourceInventory::Totals failed_reservation_ceiling{};
        SIZE_T original_min,original_max;
        DWORD original_flags;
        FailedCloseBundle(Snapshot&& snapshot,std::vector<Range>&& ranges,
            Exl3ResourceInventory&& factory,Exl3ResourceInventory::Allocation&& allocation,
            RetainedDescriptorLedger&& pending,RetainedDescriptorLedger&& lifetimes,RetainedDeviceLedger&& devices,
            RetainedCudaRegistrationLedger&& registrations,
            std::shared_ptr<PayloadAccounting>&& payloads,
            SIZE_T minimum,SIZE_T maximum,DWORD flags,Exl3ResourceInventory::Totals failed_ceiling) noexcept
            :held(std::move(snapshot)),failed_factory(std::move(factory)),failed_allocation(std::move(allocation)),pages(std::move(ranges)),
             assessments(std::move(pending)),retired(std::move(lifetimes)),retired_device(std::move(devices)),
             retired_registration(std::move(registrations)),
             payload_accounting(std::move(payloads)),
             failed_reservation_ceiling(failed_ceiling),original_min(minimum),original_max(maximum),original_flags(flags) {}
    };
    static_assert(std::is_nothrow_move_constructible_v<Snapshot>);
    static_assert(std::is_nothrow_move_constructible_v<RetainedDescriptorLedger>);
    // Raw static storage deliberately has no owning destructor: uncertain OS
    // retirement must not release payloads during C++ static destruction either.
    struct alignas(FailedCloseBundle) FailedCloseStorage {
        std::byte bytes[sizeof(FailedCloseBundle)]{};
    };
    inline static FailedCloseStorage failed_close_storage_{};
    inline static std::atomic_bool failed_close_retained_=false;
    static constexpr std::uint64_t chunk_bytes=16ULL<<20,quota_slack=64ULL<<20;
    std::uint64_t budget_,reserve_,page_=0;
    SIZE_T original_min_=0,original_max_=0,current_min_=0,current_max_=0;
    DWORD original_flags_=0,current_flags_=0;
    bool closed_=false,healthy_=true;
    std::atomic_bool external_retirement_failed_=false;
    Exl3ResourceInventory failed_factory_;
    Exl3ResourceInventory::Allocation failed_allocation_;
    bool failed_factory_retained_=false;
    Snapshot held_;
    std::vector<Range> pages_;
    std::uint64_t revision_=0;
    bool transitioning_=false;
    bool reserving_=false;
    // Conservatively retained promise after a factory throws. It is not an
    // exact physical inventory and must never make a failed authority reusable.
    Exl3ResourceInventory::Totals failed_reservation_ceiling_{};
    bool rollback_fault_for_test_=false;
    std::shared_ptr<const int> authority_identity_=std::make_shared<const int>(0);
    Exl3ResourceInventory::Totals resource_limits_=Exl3ResourceInventory::unlimited();
    mutable RetainedDescriptorLedger assessment_descriptors_;
    RetainedDescriptorLedger retired_host_descriptors_;
    RetainedDeviceLedger retired_device_bytes_;
    RetainedCudaRegistrationLedger retired_registration_bytes_;
    // Lazily reserved before construction; immutable allocation tickets remain
    // independent of root membership and of this registry's lifetime.
    std::shared_ptr<PayloadAccounting> payload_accounting_;
    std::uint64_t payload_surcharge(std::span<const Range> old,std::span<const Range> next={}) const {
        return payload_accounting_?payload_accounting_->tracker.additional_bytes(payload_accounting_->ledger,old,next):0;
    }
    void require_payload_budget(std::uint64_t pages,std::uint64_t extra,std::uint64_t requested=0) const {
        if(pages>budget_ || extra>budget_-pages || requested>budget_-pages-extra)
            throw Exl3HostResidentBudgetExhausted{};
    }
    std::uint64_t page_descriptor_bytes() const {
        if(!pages_.capacity())return 0;
        Exl3ResourceInventory::Requirement required;
        required.add(Exl3ResourceInventory::Domain::host_metadata,pages_.capacity(),sizeof(Range));
        return required.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)];
    }
    Exl3ResourceInventory::Totals allocation_limits(std::uint64_t excluded=0) const {
        auto limits=resource_limits_;
        auto& device=limits[static_cast<unsigned>(Exl3ResourceInventory::Domain::device)];
        const auto retired_device=retired_device_bytes_.bytes();
        if(retired_device>device)throw Exl3ResourceReservationExhausted{};
        device-=retired_device;
        auto& registration=limits[static_cast<unsigned>(Exl3ResourceInventory::Domain::cuda_registered_host)];
        const auto retired_registration=retired_registration_bytes_.bytes();
        if(retired_registration>registration)throw Exl3ResourceReservationExhausted{};
        registration-=retired_registration;
        auto& metadata=limits[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)];
        const auto direct=direct_descriptor_bytes();
        if(direct>metadata)throw Exl3ResourceReservationExhausted{};
        metadata-=direct;
        const auto outstanding=assessment_descriptors_.bytes();
        if(excluded>outstanding)throw std::logic_error("resident assessment credit ownership mismatch");
        if(outstanding-excluded>metadata)throw Exl3ResourceReservationExhausted{};
        metadata-=outstanding-excluded;return limits;
    }
    std::uint64_t direct_descriptor_bytes() const {
        Exl3ResourceInventory::Requirement required;
        using Domain=Exl3ResourceInventory::Domain;
        if(pages_.capacity())required.add(Domain::host_metadata,pages_.capacity(),sizeof(Range));
        // Successful snapshots move this array into inventory backing. A
        // poisoned rollback instead retains a combined array directly here.
        if(held_.owners.capacity())required.add(Domain::host_metadata,held_.owners.capacity(),sizeof(std::shared_ptr<const void>));
        const auto retired=retired_host_descriptors_.bytes();
        if(retired)required.add(Domain::host_metadata,1,retired);
        return required.units[static_cast<unsigned>(Domain::host_metadata)];
    }

    static std::runtime_error error(const char* operation,DWORD code=GetLastError()) {
        return std::runtime_error(std::string("host residency ")+operation+" error="+std::to_string(code));
    }
    static std::vector<Range> merge(std::vector<Range> ranges) {
        merge_resident_ranges_in_place(ranges);
        return ranges;
    }
    static std::vector<Range> difference(const std::vector<Range>& a,const std::vector<Range>& b) {
        return resident_difference(a,b);
    }
    static std::uint64_t extent(const std::vector<Range>& ranges) {
        std::uint64_t result=0;
        for(const auto& r:ranges) {
            if(r.second-r.first>std::numeric_limits<std::uint64_t>::max()-result)
                throw std::overflow_error("resident extent overflow");
            result+=r.second-r.first;
        }
        return result;
    }
    template<class Callback> void query(const Range& range,Callback callback) const {
        std::array<PSAPI_WORKING_SET_EX_INFORMATION,4096> records{};
        for(auto first=range.first;first<range.second;) {
            const auto count=static_cast<std::size_t>(std::min<std::uint64_t>(records.size(),(range.second-first)/page_));
            for(std::size_t i=0;i<count;++i) {records[i]={};records[i].VirtualAddress=reinterpret_cast<void*>(first+i*page_);}
            if(!QueryWorkingSetEx(GetCurrentProcess(),records.data(),static_cast<DWORD>(count*sizeof(records[0]))))
                throw error("query");
            for(std::size_t i=0;i<count;++i) callback(first+i*page_,records[i].VirtualAttributes.Valid!=0,
                records[i].VirtualAttributes.Valid && records[i].VirtualAttributes.Locked);
            first+=count*page_;
        }
    }
    void check_reserve(std::uint64_t missing=0) const {
        MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
        if(!GlobalMemoryStatusEx(&memory)) throw error("available memory");
        if(memory.ullAvailPhys<reserve_ || missing>memory.ullAvailPhys-reserve_)
            throw std::runtime_error("host residency physical reserve exhausted");
    }
    void set_quota(SIZE_T minimum,SIZE_T maximum,DWORD flags) {
        if(!SetProcessWorkingSetSizeEx(GetCurrentProcess(),minimum,maximum,flags)) throw error("working set quota");
        current_min_=minimum;current_max_=maximum;current_flags_=flags;
    }
    // Used only on a failed newly attempted lock. Preflight proved these pages
    // had no foreign lock. Do not VirtualUnlock an unlocked page: Windows can
    // otherwise remove it from the working set.
    void release_partial(const Range& range) {
        Range pending{};
        const auto flush=[&] {
            if(pending.second && !VirtualUnlock(reinterpret_cast<void*>(pending.first),pending.second-pending.first))
                throw error("partial rollback");
            pending={};
        };
        query(range,[&](auto address,bool,bool locked) {
            if(locked) {if(!pending.second) pending.first=address;pending.second=address+page_;}
            else flush();
        });
        flush();
    }
public:
    Exl3HostResidentSet(std::uint64_t maximum_page_bytes,std::uint64_t physical_reserve_bytes)
        :budget_(maximum_page_bytes),reserve_(physical_reserve_bytes) {
        bool expected=false;
        if(!process_owned_.compare_exchange_strong(expected,true)) throw std::logic_error("one resident registry per process");
        try {
            SYSTEM_INFO info{};GetSystemInfo(&info);page_=info.dwPageSize;
            if(!page_ || (page_&(page_-1)) || !budget_ || reserve_<(1ULL<<30))
                throw std::invalid_argument("resident budget/page geometry");
            if(!GetProcessWorkingSetSizeEx(GetCurrentProcess(),&original_min_,&original_max_,&original_flags_))
                throw error("original quota");
            current_min_=original_min_;current_max_=original_max_;current_flags_=original_flags_;
            check_reserve();
        } catch(...) {process_owned_=false;throw;}
    }
    Exl3HostResidentSet(const Exl3HostResidentSet&)=delete;
    Exl3HostResidentSet& operator=(const Exl3HostResidentSet&)=delete;
    ~Exl3HostResidentSet() {
        // Preserve allocation-lifetime credits even when the caller omits close.
        // Explicit close remains the error-reporting path; never throw from teardown.
        if(!closed_ && healthy_ && !transitioning_ && !reserving_) {
            try {close();} catch(...) {}
        }
        if(!closed_) {
            // No allocation, OS retry or user retirement callback on this path.
            // Explicit close reports failure; destructor preserves the unresolved
            // bundle until process exit instead of advertising reusable residency.
            std::construct_at(reinterpret_cast<FailedCloseBundle*>(failed_close_storage_.bytes),
                std::move(held_),std::move(pages_),std::move(failed_factory_),std::move(failed_allocation_),std::move(assessment_descriptors_),
                std::move(retired_host_descriptors_),std::move(retired_device_bytes_),std::move(retired_registration_bytes_),std::move(payload_accounting_),original_min_,original_max_,original_flags_,
                failed_reservation_ceiling_);
            failed_close_retained_.store(true,std::memory_order_release);
        }
    }
    static bool retirement_quarantined() noexcept {
        return failed_close_retained_.load(std::memory_order_acquire);
    }
    static bool failed_close_retained_for_test() noexcept {return retirement_quarantined();}
    static std::optional<Exl3ResourceInventory::Totals> failed_close_reservation_ceiling_for_test() noexcept {
        if(!failed_close_retained_.load(std::memory_order_acquire))return {};
        const auto* bundle=std::launder(reinterpret_cast<const FailedCloseBundle*>(failed_close_storage_.bytes));
        return bundle->failed_reservation_ceiling;
    }
    // A factory rollback may leave owners outside this inventory. This seal is
    // monotonic and callable without touching the serialized snapshot state.
    void seal_external_retirement_failure() noexcept {
        external_retirement_failed_.store(true,std::memory_order_release);
    }
    // Exclusive factory rollback only. One failed factory seals further admission;
    // moving its already prepared owners needs no allocation or extent acceptance.
    void retain_failed_factory(Exl3ResourceInventory&& owners,
        Exl3ResourceInventory::Allocation&& allocation={}) noexcept {
        if(failed_factory_retained_)std::terminate(); // Never overwrite uncertain owners.
        failed_factory_retained_=true;
        failed_factory_=std::move(owners);failed_allocation_=std::move(allocation);
        seal_external_retirement_failure();
    }
    std::uint64_t locked_page_bytes() const {return extent(pages_);}
    Exl3ResourceInventory::Totals failed_reservation_ceiling_for_test() const noexcept {return failed_reservation_ceiling_;}
    std::uint64_t revision() const noexcept {return revision_;}
    void fail_next_rollback_for_test() {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_ || rollback_fault_for_test_)
            throw std::logic_error("resident rollback fault unavailable");
        rollback_fault_for_test_=true;
    }
    std::uint64_t retained_page_descriptor_bytes_for_test() const {return page_descriptor_bytes();}
    Exl3ResourceInventory::Totals retained_resource_units() const {
        if(external_retirement_failed_.load(std::memory_order_acquire))
            throw std::logic_error("resident retained resource total unknown after external retirement failure");
        Exl3ResourceInventory::Requirement required;required.units=held_.resources.totals();
        if(const auto bytes=retired_device_bytes_.bytes())required.add(Exl3ResourceInventory::Domain::device,1,bytes);
        if(const auto bytes=retired_registration_bytes_.bytes())required.add(Exl3ResourceInventory::Domain::cuda_registered_host,1,bytes);
        const auto direct=direct_descriptor_bytes();
        if(direct)required.add(Exl3ResourceInventory::Domain::host_metadata,1,direct);
        const auto outstanding=assessment_descriptors_.bytes();
        if(outstanding)required.add(Exl3ResourceInventory::Domain::host_metadata,1,outstanding);
        return required.units;
    }
    void set_resource_limits(Exl3ResourceInventory::Totals limits) {
        if(revision_ || transitioning_ || reserving_ || closed_)throw std::logic_error("resource limits require unused registry");
        resource_limits_=limits;++revision_; // invalidate assessments made under old limits
    }
    // Diagnostic tightening only: never grants additional capacity or discards
    // retained strong/weak-owner charges. Caller serializes registry access.
    std::uint64_t tighten_metadata_headroom_for_test(std::uint64_t headroom) {
        if(closed_ || !healthy_ || transitioning_ || reserving_ ||
            external_retirement_failed_.load(std::memory_order_acquire))
            throw std::logic_error("metadata ceiling requires healthy idle registry");
        const auto domain=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
        const auto retained=retained_resource_units()[domain];
        if(retained>resource_limits_[domain] || headroom>resource_limits_[domain]-retained)
            throw std::invalid_argument("metadata ceiling diagnostic cannot increase capacity");
        if(revision_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("resident revision exhausted");
        resource_limits_[domain]=retained+headroom;++revision_;
        return retained;
    }
    Transition assess(Snapshot next) const {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_) throw std::logic_error("resident assessment unavailable");
        if(next.descriptor_backing)throw std::logic_error("resident snapshot descriptors already retained");
        if(!next.regions.empty() && (next.owners.empty() || std::any_of(next.owners.begin(),next.owners.end(),
            [](const auto& owner){return !owner || !owner.use_count();}))) throw std::invalid_argument("resident payload owner missing");
        for(auto& r:next.regions) {
            if(r.first>=r.second || r.second>std::numeric_limits<std::uintptr_t>::max()-(page_-1))
                throw std::invalid_argument("resident rounded range overflow");
            r.first&=~(page_-1);r.second=(r.second+page_-1)&~(page_-1);
        }
        Transition plan;plan.authority_=this;plan.identity_=authority_identity_;plan.revision_=revision_;plan.next_=std::move(next);
        // The transition exclusively owns the incoming range storage. Reuse it
        // for normalized coverage rather than retaining a second identical copy.
        // Payload owners remain in the immutable descriptor backing below.
        plan.desired_.swap(plan.next_.regions);
        plan.next_.retain_descriptors();
        // A failed factory bundle is normally accompanied by the permanent
        // external-retirement seal. Keep it in the same physical union anyway:
        // no future recovery path may accidentally treat quarantined owners as
        // newly available capacity.
        plan.resource_peak_=Exl3ResourceInventory::peak(held_.resources,plan.next_.resources,
            {},failed_factory_,allocation_limits());
        plan.resource_peak_[static_cast<unsigned>(Exl3ResourceInventory::Domain::device)]+=retired_device_bytes_.bytes();
        plan.resource_peak_[static_cast<unsigned>(Exl3ResourceInventory::Domain::cuda_registered_host)]+=retired_registration_bytes_.bytes();
        plan.resource_peak_[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)]+=direct_descriptor_bytes();
        plan.resource_peak_[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)]+=assessment_descriptors_.bytes();
        const auto reserve_scratch=[&](std::size_t count) {
            if(!count)return;
            Exl3ResourceInventory::Requirement required;
            using Domain=Exl3ResourceInventory::Domain;
            required.add(Domain::host_metadata,count,sizeof(Range));
            const auto domain=static_cast<unsigned>(Domain::host_metadata);
            auto& peak=plan.resource_peak_[domain];
            if(peak>resource_limits_[domain] || required.units[domain]>resource_limits_[domain]-peak)
                throw Exl3ResourceReservationExhausted{};
            peak+=required.units[domain];
            plan.scratch_bytes_+=required.units[domain];
        };
        reserve_scratch(plan.desired_.capacity());
        merge_resident_ranges_in_place(plan.desired_);
        const auto added_count=resident_difference_count(plan.desired_,pages_);
        const auto removed_count=resident_difference_count(pages_,plan.desired_);
        reserve_scratch(added_count);reserve_scratch(removed_count);
        plan.added_=resident_difference_precounted(plan.desired_,pages_,added_count);
        plan.removed_=resident_difference_precounted(pages_,plan.desired_,removed_count);
        if(plan.added_.capacity()!=added_count || plan.removed_.capacity()!=removed_count)
            throw std::length_error("resident difference descriptor capacity mismatch");
        const auto old=extent(pages_),added=extent(plan.added_);
        if(old>budget_ || added>budget_-old)
            throw Exl3HostResidentBudgetExhausted();
        plan.peak_=old+added;
        require_payload_budget(plan.peak_,payload_surcharge(pages_,plan.desired_));
        auto retained=plan.scratch_bytes_;
        if(plan.next_.descriptor_backing) {
            const auto& backing=*plan.next_.descriptor_backing;
            const auto bytes=Snapshot::descriptor_requirement(backing.owners.capacity(),backing.regions.capacity()).units[
                static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)];
            if(bytes>std::numeric_limits<std::uint64_t>::max()-retained)
                throw std::overflow_error("resident assessment retained extent overflow");
            retained+=bytes;
        }
        plan.descriptor_credit_.emplace(assessment_descriptors_.acquire(retained));
        return plan;
    }
    // Callback is optional diagnostic observation after each successful lock;
    // throwing tests cancellation/rollback before any old ownership is released.
    Stats replace(Snapshot next,const std::function<void(std::size_t)>& after_lock={}) {
        return commit(assess(std::move(next)),after_lock);
    }
    // The descriptor promise cannot escape in an outstanding Transition. This
    // serialized transaction constructs, assesses and commits before returning.
    template<class Factory,class BeforeCommit>
    Stats replace_reserved_snapshot(const Exl3ResourceInventory::Requirement& requirement,
        Factory&& factory,BeforeCommit&& before_commit,
        const std::function<void(std::size_t)>& after_lock={}) {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident snapshot reservation unavailable");
        if(requirement.configuration!=Snapshot::descriptor_requirement(0,0).configuration)
            throw std::invalid_argument("resident snapshot reservation configuration");
        const auto metadata=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
        if(requirement.units[metadata]<bounded_shared_allocation_bytes<Snapshot::DescriptorBacking>())
            throw std::invalid_argument("resident snapshot reservation extent");
        for(unsigned i=0;i<requirement.units.size();++i)
            if(i!=metadata && requirement.units[i])
                throw std::invalid_argument("resident snapshot reservation domain");
        if(revision_>=std::numeric_limits<std::uint64_t>::max()-1)
            throw std::overflow_error("resident snapshot reservation revision exhausted");
        const auto old=held_.resources.totals();
        const auto available_limits=allocation_limits();
        for(unsigned i=0;i<old.size();++i)
            if(old[i]>available_limits[i] || requirement.units[i]>available_limits[i]-old[i])
                throw Exl3ResourceReservationExhausted{};
        ++revision_;reserving_=true;
        struct Guard {bool& active;~Guard(){active=false;}} guard{reserving_};
        auto next=std::forward<Factory>(factory)(requirement.configuration);
        if(!next.fixed_regions || next.descriptor_backing ||
            Snapshot::descriptor_requirement(next.owners.capacity(),next.regions.capacity()).units!=requirement.units)
            throw std::invalid_argument("resident snapshot reserved extent mismatch");
        reserving_=false;
        auto transition=assess(std::move(next));
        reserving_=true;
        std::forward<BeforeCommit>(before_commit)();
        reserving_=false;
        return commit(std::move(transition),after_lock);
    }
    Stats commit(Transition&& plan,const std::function<void(std::size_t)>& after_lock={}) {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_ || plan.authority_!=this || plan.identity_!=authority_identity_ ||
           plan.consumed_ || plan.revision_!=revision_)
            throw std::logic_error("resident stale/consumed transition");
        if(revision_==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("resident revision exhausted");
        const auto own_credit=plan.descriptor_credit_?plan.descriptor_credit_->bytes():0;
        const auto outstanding_credit=assessment_descriptors_.bytes();
        if(own_credit>outstanding_credit)
            throw std::logic_error("resident assessment credit ownership mismatch");
        const auto other_credit=outstanding_credit-own_credit;
        Exl3ResourceInventory::Requirement live_peak;
        live_peak.units=Exl3ResourceInventory::peak(held_.resources,plan.next_.resources,
            {},failed_factory_,allocation_limits(own_credit));
        if(const auto bytes=retired_device_bytes_.bytes())live_peak.add(Exl3ResourceInventory::Domain::device,1,bytes);
        if(const auto bytes=retired_registration_bytes_.bytes())live_peak.add(Exl3ResourceInventory::Domain::cuda_registered_host,1,bytes);
        for(const auto bytes:{direct_descriptor_bytes(),other_credit,plan.scratch_bytes_})
            if(bytes)live_peak.add(Exl3ResourceInventory::Domain::host_metadata,1,bytes);
        for(unsigned i=0;i<live_peak.units.size();++i)
            if(live_peak.units[i]>resource_limits_[i])throw Exl3ResourceReservationExhausted{};
        plan.consumed_=true;
        transitioning_=true;
        struct Guard {bool& active;~Guard(){active=false;}} guard{transitioning_};
        // Release only after local scratch and failed/private snapshot storage.
        auto committing_credit=std::move(plan.descriptor_credit_);plan.descriptor_credit_.reset();
        const auto start=std::chrono::steady_clock::now();Stats stats;
        stats.source_revision=revision_;stats.transition_peak_page_bytes=plan.peak_;
        stats.transition_peak_resource_units=live_peak.units;
        stats.resource_units=plan.next_.resources.totals();
        if(const auto bytes=retired_registration_bytes_.bytes()) {
            Exl3ResourceInventory::Requirement retained;retained.units=stats.resource_units;
            retained.add(Exl3ResourceInventory::Domain::cuda_registered_host,1,bytes);stats.resource_units=retained.units;
        }
        if(const auto bytes=retired_device_bytes_.bytes()) {
            Exl3ResourceInventory::Requirement retained;retained.units=stats.resource_units;
            retained.add(Exl3ResourceInventory::Domain::device,1,bytes);stats.resource_units=retained.units;
        }
        if(plan.desired_.capacity()) {
            Exl3ResourceInventory::Requirement retained;retained.units=stats.resource_units;
            retained.add(Exl3ResourceInventory::Domain::host_metadata,plan.desired_.capacity(),sizeof(Range));
            stats.resource_units=retained.units;
        }
        for(const auto bytes:{other_credit,retired_host_descriptors_.bytes()})if(bytes) {
            Exl3ResourceInventory::Requirement retained;retained.units=stats.resource_units;
            retained.add(Exl3ResourceInventory::Domain::host_metadata,1,bytes);
            stats.resource_units=retained.units;
        }
        auto next=std::move(plan.next_);auto desired=std::move(plan.desired_);
        auto added=std::move(plan.added_),removed=std::move(plan.removed_);
        const auto& old_owners=held_.payload_owners();
        const auto& next_owners=next.payload_owners();
        if(desired.size()>std::numeric_limits<std::size_t>::max()-pages_.size() ||
           next_owners.size()>std::numeric_limits<std::size_t>::max()-old_owners.size())
            throw std::overflow_error("resident rollback descriptor count overflow");
        const auto page_slots=pages_.size()+desired.size();
        const auto owner_slots=old_owners.size()+next_owners.size();
        const auto completed_slots=resident_lock_chunk_count(added,chunk_bytes);
        Exl3ResourceInventory::Requirement rollback;
        using Domain=Exl3ResourceInventory::Domain;
        if(page_slots)rollback.add(Domain::host_metadata,page_slots,sizeof(Range));
        if(owner_slots)rollback.add(Domain::host_metadata,owner_slots,sizeof(std::shared_ptr<const void>));
        if(completed_slots)rollback.add(Domain::host_metadata,completed_slots,sizeof(Range));
        // Commit is exclusive until success or rollback. Reserve recovery
        // descriptors before allocating them and before any lock operation.
        for(unsigned i=0;i<rollback.units.size();++i) {
            auto& peak=stats.transition_peak_resource_units[i];
            if(peak>resource_limits_[i] || rollback.units[i]>resource_limits_[i]-peak)
                throw Exl3ResourceReservationExhausted{};
            peak+=rollback.units[i];
        }
        std::vector<Range> combined_pages;combined_pages.reserve(page_slots);
        std::vector<std::shared_ptr<const void>> combined_owners;combined_owners.reserve(owner_slots);
        std::vector<Range> completed;completed.reserve(completed_slots);
        if(combined_pages.capacity()!=page_slots || combined_owners.capacity()!=owner_slots ||
           completed.capacity()!=completed_slots)
            throw std::length_error("resident rollback descriptor capacity mismatch");
        merge_sorted_resident_ranges(pages_,desired,combined_pages);
        combined_owners.insert(combined_owners.end(),old_owners.begin(),old_owners.end());
        combined_owners.insert(combined_owners.end(),next_owners.begin(),next_owners.end());
        auto combined_resources=held_.resources;combined_resources.append(next.resources);
        const auto poison=[&] {
            healthy_=false;pages_=std::move(combined_pages);held_.owners=std::move(combined_owners);
            held_.descriptor_backing.reset();
            held_.resources=std::move(combined_resources);
        };
        const auto added_bytes=extent(added);
        const auto old_bytes=extent(pages_);
        if(old_bytes>budget_ || added_bytes>budget_-old_bytes)
            throw Exl3HostResidentBudgetExhausted();
        require_payload_budget(old_bytes+added_bytes,payload_surcharge(pages_,desired));
        std::uint64_t missing=0;
        for(const auto& r:added) query(r,[&](auto,bool valid,bool locked) {
            if(locked) throw std::invalid_argument("resident page has a foreign lock");
            if(!valid) missing+=page_;
        });
        if(added_bytes) check_reserve(missing);
        const auto previous_min=current_min_,previous_max=current_max_;const auto previous_flags=current_flags_;
        Range attempted{};
        try {
            if(added_bytes) {
                const auto required=old_bytes+added_bytes;
                if(required>std::numeric_limits<SIZE_T>::max()-2*quota_slack) throw std::overflow_error("resident quota overflow");
                const SIZE_T minimum=std::max<SIZE_T>(current_min_,required+quota_slack);
                const SIZE_T maximum=std::max<SIZE_T>(current_max_,minimum+quota_slack);
                set_quota(minimum,maximum,QUOTA_LIMITS_HARDWS_MIN_DISABLE|QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            }
            for(const auto& r:added) for(auto first=r.first;first<r.second;) {
                const auto end=first+std::min<std::uint64_t>(chunk_bytes,r.second-first);
                if(completed.size()==completed.capacity())
                    throw std::logic_error("resident completed lock reservation exhausted");
                attempted={first,end};
                if(!VirtualLock(reinterpret_cast<void*>(first),end-first)) throw error("VirtualLock");
                completed.push_back(attempted);attempted={};
                query(completed.back(),[&](auto,bool valid,bool locked) {
                    if(!valid || !locked) throw std::runtime_error("new resident lock not witnessed");
                });
                check_reserve();++stats.lock_calls;
                if(after_lock) after_lock(completed.size());
                first=end;
            }
        } catch(...) {
            const auto failure=std::current_exception();
            try {
                if(std::exchange(rollback_fault_for_test_,false))
                    throw std::runtime_error("injected resident rollback before unlock");
                if(attempted.second) release_partial(attempted);
                for(auto i=completed.rbegin();i!=completed.rend();++i)
                    if(!VirtualUnlock(reinterpret_cast<void*>(i->first),i->second-i->first)) throw error("rollback unlock");
                set_quota(previous_min,previous_max,previous_flags);
            } catch(...) {poison();throw;}
            std::rethrow_exception(failure);
        }
        // No fallible allocation below. Keep both generations alive until all
        // obsolete locks release. Unexpected OS unlock failure is fatal/poisoned.
        for(const auto& r:removed) if(!VirtualUnlock(reinterpret_cast<void*>(r.first),r.second-r.first)) {
            const auto failure=error("obsolete unlock");poison();throw failure;
        }
        stats.new_locked_bytes=added_bytes;stats.unlocked_bytes=extent(removed);stats.page_bytes=extent(desired);
        pages_.swap(desired);held_=std::move(next);
        stats.committed_revision=++revision_;
        stats.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return stats;
    }
    // Serialized synchronous construction only (startup or coordinator-owned
    // runtime credit): factory must return idle owners,
    // never enqueue GPU consumers. Its temporary owners unwind before the promise
    // is released on failure. No page-lock or CUDA-registration guarantee is
    // inferred from a device/metadata credit. Dynamic asynchronous fills still
    // need a retained transfer transaction after idle construction completes.
    template<class Factory> Exl3ResourceInventory allocate_reserved(
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory) {
        return allocate_reserved_growing(requirement,[&](auto configuration,auto&&) {
            return std::forward<Factory>(factory)(configuration);
        });
    }
    // The initial promise covers planning owners. The synchronous factory may
    // extend it before constructing additional resources. Extension is monotonic
    // and exclusive; the callback must not escape the factory's lifetime.
    enum class ReservationExtent { exact, at_most };
    template<class Factory> Exl3ResourceInventory allocate_reserved_growing(
        const Exl3ResourceInventory::Requirement& requirement,Factory&& factory,
        ReservationExtent extent=ReservationExtent::exact) {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident allocation reservation unavailable");
        if(!requirement.configuration || requirement.units==Exl3ResourceInventory::Totals{})
            throw std::invalid_argument("resource reservation missing configuration/extent");
        if(revision_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("resident revision exhausted");
        const auto old=held_.resources.totals();
        const auto available_limits=allocation_limits();
        for(unsigned i=0;i<old.size();++i)
            if(old[i]>available_limits[i] || requirement.units[i]>available_limits[i]-old[i])
                throw Exl3ResourceReservationExhausted{};
        // Invalidate every previously assessed plan, including on rollback.
        // The in-flight promise is exclusive and cannot be replayed or nested.
        ++revision_;reserving_=true;
        auto promised=requirement;
        failed_reservation_ceiling_={};
        struct Guard {
            bool& active;
            const Exl3ResourceInventory::Requirement& promised;
            Exl3ResourceInventory::Totals& failed;
            bool committed=false;
            ~Guard(){if(!committed)failed=promised.units;active=false;}
        } guard{reserving_,promised,failed_reservation_ceiling_};
        const auto extend=[&](const Exl3ResourceInventory::Requirement& next) {
            if(next.configuration!=promised.configuration)
                throw std::invalid_argument("resource reservation extension configuration mismatch");
            for(unsigned i=0;i<old.size();++i) {
                if(next.units[i]<promised.units[i])
                    throw std::invalid_argument("resource reservation extension cannot shrink");
                if(old[i]>available_limits[i] || next.units[i]>available_limits[i]-old[i])
                    throw Exl3ResourceReservationExhausted{};
            }
            promised=next;
        };
        auto actual=std::forward<Factory>(factory)(requirement.configuration,extend);
        const auto actual_units=actual.totals();
        if(extent==ReservationExtent::exact) {
            if(actual_units!=promised.units)
                throw std::invalid_argument("resource reservation actual extent/domain mismatch");
        } else {
            if(actual_units==Exl3ResourceInventory::Totals{})
                throw std::invalid_argument("bounded resource reservation empty allocation");
            for(unsigned i=0;i<actual_units.size();++i)
                if(actual_units[i]>promised.units[i])
                    throw std::invalid_argument("bounded resource reservation exceeded ceiling/domain");
        }
        auto combined=held_.resources;combined.append(actual);
        const auto total=combined.totals();
        for(unsigned i=0;i<total.size();++i)
            if(total[i]-old[i]!=actual_units[i])
                throw std::invalid_argument("resource credit reused existing allocation owner");
        // All allocations and checks precede publication. The return is a
        // retained inventory, not a reusable allocation credit.
        held_.resources=std::move(combined);
        guard.committed=true;
        return actual;
    }
    void transfer_retired_metadata(const Exl3ResourceInventory::Allocation& expected,
        const Exl3ResourceInventory::Allocation& tracking) {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident metadata transfer unavailable");
        if(revision_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("resident revision exhausted");
        auto next=held_.resources.transfer_metadata(expected,tracking);
        ++revision_;held_.resources=std::move(next);
    }
    bool collect_retired_metadata(const Exl3ResourceInventory::Allocation& tracking,
        const std::weak_ptr<const void>& retired) {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident metadata collection unavailable");
        const std::weak_ptr<const void> empty;
        if(tracking.domain!=Exl3ResourceInventory::Domain::host_metadata ||
            (!retired.owner_before(empty) && !empty.owner_before(retired)))
            throw std::invalid_argument("metadata collection missing retired identity");
        if(!held_.resources.metadata_owner_expired(tracking,retired))return false;
        auto next=held_.resources.without_allocation(tracking);
        if(revision_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("resident revision exhausted");
        ++revision_;held_.resources=std::move(next);return true;
    }
    // Ordinary host ownership uses a zero-argument retirement callback.
    // False/throw preserves the charge; all fallible inventory preparation
    // precedes callback entry. Device retirement has a separate contract below.
    template<class Retire> bool retire_reserved_host(const Exl3ResourceInventory::Allocation& allocation,
        Retire&& retire) {
        static_assert(std::is_invocable_r_v<bool,Retire&>);
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident host retirement unavailable");
        if(allocation.domain!=Exl3ResourceInventory::Domain::host_metadata)
            throw std::invalid_argument("resident host retirement requires metadata allocation");
        if(revision_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("resident revision exhausted");
        auto next=held_.resources.without_allocation(allocation);
        ++revision_;reserving_=true;
        struct Guard {bool& active;~Guard(){active=false;}} guard{reserving_};
        if(!std::forward<Retire>(retire)())return false;
        held_.resources=std::move(next);return true;
    }
    // Attach must either accept the ticket or return false without retaining
    // it. Credit is conserved while inventory ownership becomes lifetime credit.
    // Charge an already constructed immutable owner independently of resident
    // membership. Its ticket survives eviction and external strong/weak owners.
    std::uint64_t owner_lifetime_metadata_bytes_for_test() const noexcept {return retired_host_descriptors_.bytes();}
    RetainedHostAllocationLedger::Ticket reserve_host_payload_lifetime(std::uint64_t bytes) {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident payload lifetime reservation unavailable");
        if(!bytes)throw std::invalid_argument("resident payload lifetime reservation extent");
        require_payload_budget(extent(pages_),payload_surcharge(pages_),bytes);
        if(!payload_accounting_) {
            using Counter=RetainedHostAllocationLedger::State;
            constexpr auto counter_bytes=bounded_shared_allocation_bytes<Counter>();
            constexpr auto accounting_bytes=bounded_shared_allocation_bytes<PayloadAccounting>();
            static_assert(counter_bytes<=UINT64_MAX-accounting_bytes);
            auto metadata=reserve_host_metadata_lifetime(accounting_bytes+counter_bytes);
            auto counter_credit=metadata.split(counter_bytes);
            if(!counter_credit)throw std::logic_error("resident payload counter metadata split");
            auto counter=make_bounded_shared<Counter>();
            if(!attach_bounded_retirement_credit<Counter>(counter,std::move(*counter_credit)))
                throw std::logic_error("resident payload counter metadata attachment");
            auto accounting=make_bounded_shared<PayloadAccounting>(std::move(counter));
            if(!attach_bounded_retirement_credit<PayloadAccounting>(accounting,std::move(metadata)))
                throw std::logic_error("resident payload accounting metadata attachment");
            payload_accounting_=std::move(accounting);
        }
        if(revision_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("resident revision exhausted");
        auto ticket=payload_accounting_->ledger.acquire(bytes);
        ++revision_;return ticket;
    }
    template<class Matches>
    void track_host_payload_lifetime(const std::shared_ptr<const void>& owner,std::size_t plane,
        const void* data,std::size_t bytes,Matches&& matches) {
        static_assert(std::is_nothrow_invocable_r_v<bool,Matches&,const std::shared_ptr<const void>&,
            const RetainedHostAllocationLedger&,std::size_t,const void*,std::size_t>);
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_ || !payload_accounting_)
            throw std::logic_error("resident payload lifetime tracking unavailable");
        if(!owner || !owner.use_count() || !bytes ||
           !matches(owner,payload_accounting_->ledger,plane,data,bytes))
            throw std::invalid_argument("resident payload lifetime owner/credit mismatch");
        if(revision_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("resident revision exhausted");
        payload_accounting_->tracker.track(owner,plane,data,bytes);
        ++revision_;
    }
    RetainedDescriptorLedger::Ticket reserve_host_metadata_lifetime(std::uint64_t bytes) {
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident metadata lifetime reservation unavailable");
        if(!bytes)throw std::invalid_argument("resident metadata lifetime reservation extent");
        const auto domain=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
        const auto limits=allocation_limits(),held=held_.resources.totals();
        if(held[domain]>limits[domain] || bytes>limits[domain]-held[domain])throw Exl3ResourceReservationExhausted{};
        if(revision_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("resident revision exhausted");
        ++revision_;
        return retired_host_descriptors_.acquire(bytes);
    }
    template<class Matches,class Attach>
    void admit_host_metadata_lifetime(const std::shared_ptr<const void>& owner,std::uint64_t bytes,
        Matches&& matches,Attach&& attach) {
        static_assert(std::is_nothrow_invocable_r_v<bool,Matches&,const std::shared_ptr<const void>&,
            const RetainedDescriptorLedger&>);
        static_assert(std::is_nothrow_invocable_r_v<bool,Attach&,const std::shared_ptr<const void>&,
            RetainedDescriptorLedger::Ticket>);
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident metadata lifetime admission unavailable");
        if(!owner || !owner.use_count() || !bytes)throw std::invalid_argument("resident metadata lifetime owner/extent");
        if(std::forward<Matches>(matches)(owner,retired_host_descriptors_))return;
        auto credit=reserve_host_metadata_lifetime(bytes);
        reserving_=true;
        struct Guard {bool& active;~Guard(){active=false;}} guard{reserving_};
        if(!std::forward<Attach>(attach)(owner,std::move(credit)))
            throw std::logic_error("resident metadata lifetime attachment refused");
    }
    template<class Attach> bool retire_reserved_host_to_lifetime(
        const Exl3ResourceInventory::Allocation& allocation,Attach&& attach) {
        static_assert(std::is_nothrow_invocable_r_v<bool,Attach&,RetainedDescriptorLedger::Ticket>);
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident host lifetime retirement unavailable");
        if(allocation.domain!=Exl3ResourceInventory::Domain::host_metadata)
            throw std::invalid_argument("resident host lifetime retirement domain");
        if(revision_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("resident revision exhausted");
        auto next=held_.resources.without_allocation(allocation);
        ++revision_;reserving_=true;
        struct Guard {bool& active;~Guard(){active=false;}} guard{reserving_};
        auto credit=retired_host_descriptors_.acquire(allocation.units);
        if(!std::forward<Attach>(attach)(std::move(credit)))return false;
        held_.resources=std::move(next);return true;
    }
    // Caller has proved device or registered-host readers complete. The
    // callback retires that resource; metadata and Windows locks stay separate.
    template<class Retire> bool retire_reserved(const Exl3ResourceInventory::Allocation& allocation,
        Retire&& retire) {
        static_assert(std::is_nothrow_invocable_r_v<bool,Retire&,const Exl3ResourceInventory::Allocation&>);
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident retirement reservation unavailable");
        if(allocation.domain!=Exl3ResourceInventory::Domain::device &&
           allocation.domain!=Exl3ResourceInventory::Domain::cuda_registered_host)
            throw std::invalid_argument("resident retirement requires device or CUDA registration allocation");
        if(revision_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("resident revision exhausted");
        auto next=held_.resources.without_allocation(allocation);
        ++revision_;reserving_=true;
        struct Guard {bool& active;~Guard(){active=false;}} guard{reserving_};
        if(!retire(allocation))return false;
        held_.resources=std::move(next);return true;
    }
    template<class Attach> bool retire_reserved_device_to_lifetime(
        const Exl3ResourceInventory::Allocation& allocation,Attach&& attach) {
        static_assert(std::is_nothrow_invocable_r_v<bool,Attach&,RetainedDeviceLedger::Ticket>);
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident device lifetime retirement unavailable");
        if(allocation.domain!=Exl3ResourceInventory::Domain::device)
            throw std::invalid_argument("resident device lifetime retirement domain");
        if(revision_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("resident revision exhausted");
        auto next=held_.resources.without_allocation(allocation);
        ++revision_;reserving_=true;
        struct Guard {bool& active;~Guard(){active=false;}} guard{reserving_};
        auto credit=retired_device_bytes_.acquire(allocation.units);
        if(!std::forward<Attach>(attach)(std::move(credit)))return false;
        held_.resources=std::move(next);return true;
    }
    template<class Attach> bool retire_reserved_registration_to_lifetime(
        const Exl3ResourceInventory::Allocation& allocation,Attach&& attach) {
        static_assert(std::is_nothrow_invocable_r_v<bool,Attach&,RetainedCudaRegistrationLedger::Ticket>);
        if(closed_ || !healthy_ || external_retirement_failed_.load(std::memory_order_acquire) || transitioning_ || reserving_)
            throw std::logic_error("resident registration lifetime retirement unavailable");
        if(allocation.domain!=Exl3ResourceInventory::Domain::cuda_registered_host)
            throw std::invalid_argument("resident registration lifetime retirement domain");
        if(revision_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("resident revision exhausted");
        auto next=held_.resources.without_allocation(allocation);
        ++revision_;reserving_=true;
        struct Guard {bool& active;~Guard(){active=false;}} guard{reserving_};
        auto credit=retired_registration_bytes_.acquire(allocation.units);
        if(!std::forward<Attach>(attach)(std::move(credit)))return false;
        held_.resources=std::move(next);return true;
    }
    void close() {
        if(closed_) return;
        if(external_retirement_failed_.load(std::memory_order_acquire))
            throw std::logic_error("resident external allocation retirement unresolved");
        while(auto allocation=held_.resources.first_lifetime_retirement()) {
            if(allocation->registration_lifetime_retire) {
                if(!retire_reserved_registration_to_lifetime(*allocation,
                    [&](RetainedCudaRegistrationLedger::Ticket credit) noexcept {
                        return allocation->registration_lifetime_retire(allocation->owner,std::move(credit));
                    }))throw std::logic_error("resident close registration lifetime transfer refused");
                continue;
            }
            if(allocation->device_lifetime_retire) {
                if(!retire_reserved_device_to_lifetime(*allocation,
                    [&](RetainedDeviceLedger::Ticket credit) noexcept {
                        return allocation->device_lifetime_retire(allocation->owner,std::move(credit));
                    }))throw std::logic_error("resident close device lifetime transfer refused");
                continue;
            }
            if(!retire_reserved_host_to_lifetime(*allocation,
                [&](RetainedDescriptorLedger::Ticket credit) noexcept {
                    return allocation->lifetime_retire(allocation->owner,std::move(credit));
                }))throw std::logic_error("resident close lifetime transfer refused");
        }
        replace({});set_quota(original_min_,original_max_,original_flags_);
        closed_=true;process_owned_=false;
    }
};
} // namespace ninfer::exl3
#endif
