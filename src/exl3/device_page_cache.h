#pragma once
#include "exl3/device_page_storage.h"
#include "exl3/bounded_shared_owner.h"
#include <mutex>

namespace ninfer::exl3 {
struct Exl3DevicePageWeakComponents {
    // One fill, immutable View and storage block for each of 64 ready entries.
    std::array<std::weak_ptr<const void>,192> owners;
    std::array<std::uint64_t,192> bytes{};
    std::size_t count=0;
};
// Fixed lookup capacity and one coordinator authority. Replacement follows
// physical retirement and weak-owner lookup release. Component metadata tickets
// follow actual allocation lifetimes independently of lookup-slot reuse.
class Exl3DevicePageCache : public std::enable_shared_from_this<Exl3DevicePageCache> {
    struct Entry {
        std::optional<Exl3DevicePageKey> key;
        std::shared_ptr<Exl3DevicePageFill> fill;
        bool retiring=false;
        bool reclaimed=false;
        bool metadata_detached=false;
        bool tracking=false;
        std::weak_ptr<const void> retired_owner;
        std::weak_ptr<const Exl3ExactKVPage> retired_source;
        std::uint64_t retired_source_bytes=0;
    };
    std::array<Entry,64> entries_;
    const void* authority_;
    std::mutex mutex_;
    std::size_t victim_cursor_=0;
    unsigned constructor_fault_for_test_=0;
    bool constructor_cleanup_failure_for_test_=false;
    struct ConstructionKey {};
public:
    Exl3DevicePageWeakComponents weak_components_for_test() {
        std::lock_guard<std::mutex> lock(mutex_);Exl3DevicePageWeakComponents result;
        const auto add=[&](const std::shared_ptr<const void>& owner,std::uint64_t bytes) {
            result.owners[result.count]=owner;result.bytes[result.count++]=bytes;
        };
        for(const auto& entry:entries_)if(entry.fill && entry.fill->ready()) {
            auto view=entry.fill->ready_handle();auto storage=entry.fill->storage_owner_for_retirement();
            if(!view || !storage)throw std::logic_error("ready page lost component ownership");
            add(entry.fill,bounded_shared_allocation_bytes<Exl3DevicePageFill>());
            add(view,bounded_shared_allocation_bytes<Exl3DevicePageFill::View>());
            add(storage,bounded_shared_allocation_bytes<Exl3DevicePageStorage>());
        }
        return result;
    }
    bool constructor_fault_armed_for_test() {
        std::lock_guard<std::mutex> lock(mutex_);return constructor_fault_for_test_!=0;
    }
    void fail_next_constructor_for_test(unsigned stage,bool fail_cleanup) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(stage<1 || stage>4 || constructor_fault_for_test_)
            throw std::invalid_argument("page constructor fixture requires unarmed stage1..4");
        constructor_fault_for_test_=stage;constructor_cleanup_failure_for_test_=fail_cleanup;
    }
    explicit Exl3DevicePageCache(ConstructionKey,const void* authority):authority_(authority){}
    static constexpr std::size_t metadata_bytes() noexcept {
        return bounded_shared_allocation_bytes<Exl3DevicePageCache>();
    }
    static bool attach_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(credit.bytes()!=metadata_bytes())return false;
        return attach_bounded_retirement_credit<Exl3DevicePageCache,const void>(owner,std::move(credit));
    }
    struct Accounting {
        std::uint64_t unique_allocations=0,device_bytes=0,ready_rows=0;
        std::uint64_t pending_fill_bytes=0,failed_allocations=0,retiring_allocations=0;
        std::uint64_t retained_metadata_records=0;
        std::uint64_t retained_readers=0;
        std::uint64_t in_flight_fill_bytes=0;
        std::uint64_t retained_source_pages=0,retained_source_host_bytes=0;
    };
    // Physical registry snapshot, not hit counts or physical free-memory telemetry.
    // A failed allocation remains charged until certified retirement. Weak lookup
    // records describe retired fills; detached component tickets can outlive them.
    Accounting accounting() {
        std::lock_guard<std::mutex> lock(mutex_);
        Accounting result;
        std::array<std::shared_ptr<const Exl3ExactKVPage>,64> source_pages{};
        const auto add_source=[&](std::shared_ptr<const Exl3ExactKVPage> source,std::uint64_t bytes) {
            if(!source)return;
            bool seen=false;
            for(std::size_t i=0;i<result.retained_source_pages;++i)if(source_pages[i]==source){seen=true;break;}
            if(!seen) {
                source_pages[result.retained_source_pages++]=std::move(source);
                if(bytes>std::numeric_limits<std::uint64_t>::max()-result.retained_source_host_bytes)
                    throw std::overflow_error("page cache retained host bytes overflow");
                result.retained_source_host_bytes+=bytes;
            }
        };
        for(const auto& entry:entries_) {
            if(entry.tracking){
                ++result.retained_metadata_records;
                // Retained fill handles own their key even after device storage
                // retirement. A weak source alone must not keep that charge alive.
                if(const auto owner=entry.retired_owner.lock())
                    add_source(entry.retired_source.lock(),entry.retired_source_bytes);
                continue;
            }
            if(!entry.fill)continue;
            add_source(entry.key->page(),entry.key->retained_host_bytes());
            result.retained_readers+=entry.fill->retained_readers();
            if(entry.fill->fill_in_flight())result.in_flight_fill_bytes+=Exl3DevicePageStorage::bytes;
            ++result.retained_metadata_records;
            if(entry.reclaimed)continue;
            ++result.unique_allocations;result.device_bytes+=Exl3DevicePageStorage::bytes;
            if(entry.retiring)++result.retiring_allocations;
            if(entry.fill->failed())++result.failed_allocations;
            else if(entry.fill->ready())result.ready_rows+=64;
            else if(!entry.fill->sealed_for_retirement())result.pending_fill_bytes+=Exl3DevicePageStorage::bytes;
        }
        return result;
    }
    bool uncertain_after_join() {
        std::lock_guard<std::mutex> lock(mutex_);
        for(const auto& entry:entries_)if(entry.fill && entry.fill->uncertain_after_join())return true;
        return false;
    }
    // First eviction phase only: stop attaching new consumers. The physical
    // owner/credit stays retained until a separate certified reclamation step.
    // Existing producer/waiter/reader handles remain valid and may complete.
    template<class Coordinator>
    bool retire_lookup(Coordinator& authority,const typename Coordinator::Lease& lease,
        const Exl3DevicePageKey& key) {
        if(authority_!=&authority || !authority.compute_leases_current(std::span(&lease,1)) || !key.current())
            throw std::invalid_argument("page retirement authority/lease/key");
        std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
        if(!lock.owns_lock())return false;
        if(!authority.compute_leases_current(std::span(&lease,1)) || !key.current())
            throw std::invalid_argument("page retirement lease/key changed before mutation");
        for(auto& entry:entries_)if(entry.fill && entry.key->same(key)) {
            if(entry.retiring)return false;
            entry.retiring=true;return true;
        }
        return false;
    }
    struct Acquisition {
        std::shared_ptr<const Exl3DevicePageFill> shared;
        std::optional<Exl3DevicePageStorage::Prepared> producer;
        Acquisition()=default;
        Acquisition(Acquisition&&)=default;
        void abandon() noexcept {
            if(producer && producer->fill)producer->fill->abandon_producer();
        }
        ~Acquisition(){abandon();}
        Acquisition& operator=(Acquisition&& other) noexcept {
            if(this!=&other){abandon();shared=std::move(other.shared);producer=std::move(other.producer);}
            return *this;
        }
        Acquisition(const Acquisition&)=delete;
    };
    template<class Coordinator>
    std::shared_ptr<Exl3DevicePageFill> seal_retired(Coordinator& authority,
        const typename Coordinator::Lease& lease,const Exl3DevicePageKey& key) {
        if(authority_!=&authority || !authority.compute_leases_current(std::span(&lease,1)) || !key.current())
            throw std::invalid_argument("page seal authority/lease/key");
        std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
        if(!lock.owns_lock())return {};
        if(!authority.compute_leases_current(std::span(&lease,1)) || !key.current())
            throw std::invalid_argument("page seal lease/key changed before mutation");
        for(auto& entry:entries_)if(entry.fill && entry.retiring && entry.key->same(key)) {
            if(entry.fill->sealed_for_retirement() || entry.fill->seal_for_retirement())return entry.fill;
            return {};
        }
        return {};
    }
    template<class Coordinator>
    static std::shared_ptr<Exl3DevicePageCache> create_startup(Coordinator& authority,unsigned startup_fault_for_test=0) {
        if(startup_fault_for_test>1)throw std::invalid_argument("page cache startup fault0/1");
        authority.reserve_runtime_host_sources();
        using Inventory=Exl3ResourceInventory;
        Inventory::Requirement required;required.configuration=0x4456504341;
        required.add(Inventory::Domain::host_metadata,1,metadata_bytes());
        std::shared_ptr<Exl3DevicePageCache> result;
        authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
            if(configuration!=required.configuration)throw std::logic_error("device page cache reservation identity");
            result=make_bounded_shared<Exl3DevicePageCache>(ConstructionKey{},&authority);
            if(startup_fault_for_test)throw std::runtime_error("injected page cache container precommit failure");
            Inventory actual;actual.add({result,0,Inventory::Domain::host_metadata,metadata_bytes(),{},&attach_retirement_credit});
            return actual;
        },[&]() noexcept {result.reset();});
        return result;
    }
    template<class Coordinator>
    bool reclaim_retired(Coordinator& authority,const typename Coordinator::Lease& lease,
        const Exl3DevicePageKey& key) {
        if(authority_!=&authority || !authority.compute_leases_current(std::span(&lease,1)) || !key.current())
            throw std::invalid_argument("page reclamation authority/lease/key");
        std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
        if(!lock.owns_lock())return false;
        if(!authority.compute_leases_current(std::span(&lease,1)) || !key.current())
            throw std::invalid_argument("page reclamation lease/key changed before mutation");
        for(auto& entry:entries_)if(entry.fill && entry.retiring && entry.key->same(key)) {
            // Transfer the full component charge while storage and preallocated
            // views still exist. Each ticket then follows its actual final owner.
            if(!entry.metadata_detached) {
                entry.metadata_detached=authority.retire_runtime_metadata_to_lifetime(lease,
                    Exl3DevicePageStorage::metadata_allocation(entry.fill));
                if(!entry.metadata_detached)return false;
            }
            if(!entry.reclaimed) {
                const auto allocation=Exl3DevicePageStorage::device_allocation(entry.fill);
                entry.reclaimed=authority.retire_runtime_resource(lease,allocation,
                    [&](const auto&) noexcept {return entry.fill->retire_sealed_storage();});
                if(!entry.reclaimed)return false;
            }
            const auto source_bytes=entry.key->retained_host_bytes();
            entry.retired_source=entry.key->page();entry.retired_source_bytes=source_bytes;
            entry.retired_owner=entry.fill;entry.tracking=true;
            entry.fill.reset();entry.key.reset();return true;
        }
        return false;
    }
    template<class Coordinator>
    bool evict_one(Coordinator& authority,const typename Coordinator::Lease& lease) {
        if(authority_!=&authority || !authority.compute_leases_current(std::span(&lease,1)))
            throw std::invalid_argument("page victim selection authority/lease");
        if(Exl3DevicePageStorage::quarantined_bytes() || Exl3DevicePageFill::quarantined_records())
            throw std::runtime_error("page victim selection after uncertain retirement");
        std::optional<Exl3DevicePageKey> key;
        std::shared_ptr<Exl3DevicePageFill> selected;
        {
            std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
            if(!lock.owns_lock())return false;
            if(!authority.compute_leases_current(std::span(&lease,1)))
                throw std::invalid_argument("page victim lease changed before selection");
            for(std::size_t offset=0;offset<entries_.size();++offset) {
                const auto index=(victim_cursor_+offset)%entries_.size();auto& entry=entries_[index];
                if(!entry.fill || !entry.key->current())continue;
                const bool abandoned=entry.fill->abandoned_before_submission();
                if((entry.fill->failed() && !abandoned) ||
                    (!entry.fill->ready() && !entry.fill->sealed_for_retirement() && !abandoned))continue;
                key=entry.key;selected=entry.fill;entry.retiring=true;
                victim_cursor_=(index+1)%entries_.size();break;
            }
        }
        if(!selected)return false;
        // Retain selected across phases so another thread cannot recycle its
        // metadata slot and create an ABA match under the same source key.
        if(!seal_retired(authority,lease,*key))return false;
        return reclaim_retired(authority,lease,*key);
    }
    template<class Coordinator>
    Acquisition acquire(Coordinator& authority,const typename Coordinator::Lease& lease,
        const Exl3DevicePageKey& key,Exl3DevicePageStorageProvider provider={}) {
        if(authority_!=&authority || !authority.compute_leases_current(std::span(&lease,1)) || !key.current())
            throw std::invalid_argument("device page cache authority/lease/key");
        std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
        if(!lock.owns_lock())return {}; // bounded host fallback while allocator owns lookup
        Entry* empty=nullptr;
        for(auto& entry:entries_) {
            if(entry.tracking) {
                // Expiration permits lookup-slot reuse only. Detached component
                // tickets continue charging any externally held weak controls.
                if(!entry.retired_owner.expired())continue;
                if(!authority.compute_leases_current(std::span(&lease,1)) || !key.current())
                    throw std::invalid_argument("device page cache lease/key changed before slot reuse");
                entry=Entry{};
            }
            if(!entry.fill){if(!empty)empty=&entry;continue;}
            if(entry.key->shortlist()==key.shortlist() && entry.key->same(key)) {
                if(entry.retiring || entry.fill->failed())return {};
                // Hits bypass allocation authority. Refuse stale delivery without
                // abandoning the existing producer or revoking peer readers.
                if(!authority.compute_leases_current(std::span(&lease,1)) || !key.current())
                    throw std::invalid_argument("device page cache lease/key changed during lookup");
                Acquisition result;result.shared=entry.fill;return result;
            }
        }
        if(!empty)return {};
        Exl3DevicePageStorage::Prepared prepared;
        if(constructor_fault_for_test_ && constructor_cleanup_failure_for_test_)
            provider.release=[](void*) noexcept {return cudaErrorUnknown;};
        try{prepared=Exl3DevicePageStorage::create_runtime(authority,lease,key,provider,0,&constructor_fault_for_test_);}
        catch(const Exl3ResourceReservationExhausted&){return {};}
        empty->key=key;empty->fill=prepared.fill;
        try{authority.bind_runtime_host_source(lease,prepared.fill,key);}
        catch(...) {
            // Allocation stays inventoried and reachable for normal retirement;
            // no producer or numerical source pointer has escaped this call.
            empty->retiring=true;prepared.fill->abandon_producer();throw;
        }
        if(!authority.compute_leases_current(std::span(&lease,1)) || !key.current()) {
            // Keep the inventoried allocation reachable for certified retirement.
            // No producer has escaped and no numerical submission has occurred.
            empty->retiring=true;prepared.fill->abandon_producer();
            throw std::invalid_argument("device page cache lease/key changed during allocation");
        }
        Acquisition result;result.shared=prepared.fill;result.producer=std::move(prepared);return result;
    }
};
}
