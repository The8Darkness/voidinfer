#pragma once
#include "exl3/reconstruction_control_allocator.h"
#include "exl3/resource_inventory.h"
#include <cuda_runtime.h>
#include <Windows.h>
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::exl3 {
// Recoverable only when the partially constructed slab retired successfully.
// Carry the original provider status through Engine's nested cleanup exception.
class Exl3RecurrentEventCreationFailure : public std::runtime_error {
public:
    const cudaError_t status;
    explicit Exl3RecurrentEventCreationFailure(cudaError_t value)
        :std::runtime_error("recurrent completion event creation failed"),status(value){}
};
struct Exl3RecurrentSlabLayout {
    static constexpr std::size_t max_bytes=144ULL<<20;
    std::array<std::size_t,49> offsets{};
    std::size_t bytes=0;
    static std::optional<Exl3RecurrentSlabLayout> derive(const std::array<std::size_t,48>& elements) noexcept {
        Exl3RecurrentSlabLayout result;
        constexpr auto max_elements=max_bytes/sizeof(float);
        for(std::size_t i=0;i<elements.size();++i) {
            if(elements[i]>max_elements-result.offsets[i])return std::nullopt;
            result.offsets[i+1]=result.offsets[i]+elements[i];
        }
        if(!result.offsets.back())return std::nullopt;
        result.bytes=result.offsets.back()*sizeof(float);return result;
    }
};
// Shared by all physical contexts and retained immutable roots in this process.
// Registration is an additional DMA property, never a substitute for VirtualLock.
struct Exl3RecurrentPinBudget {
    static constexpr std::uint64_t cap=1ULL<<30,reserve=8ULL<<30;
    inline static std::mutex mutex;
    inline static std::uint64_t live=0,peak=0,quarantined=0;
    static bool claim(std::size_t bytes){
        std::lock_guard lock(mutex);MEMORYSTATUSEX m{};m.dwLength=sizeof(m);
        if(bytes>cap-live || !GlobalMemoryStatusEx(&m) || m.ullAvailPhys<reserve+bytes)return false;
        live+=bytes;peak=std::max(peak,live);return true;
    }
    static void release(std::size_t bytes){std::lock_guard lock(mutex);live-=bytes;}
    static void quarantine(std::size_t bytes){std::lock_guard lock(mutex);quarantined+=bytes;}
    static std::array<std::uint64_t,3> snapshot(){std::lock_guard lock(mutex);return {live,peak,quarantined};}
};
struct Exl3RecurrentSlab {
    Exl3RecurrentSlab()=default;
    Exl3RecurrentSlab(const Exl3RecurrentSlab&)=delete;
    Exl3RecurrentSlab& operator=(const Exl3RecurrentSlab&)=delete;
    float* data=nullptr;
    std::size_t bytes=0;
    std::array<std::size_t,49> offsets{};
    cudaEvent_t completed=nullptr;
    int device=0;
    bool registered=false,recorded=false,poisoned=false;
    bool budget_claimed=false;
    bool fail_unregister_for_test=false;
    cudaError_t construction_error=cudaSuccess;
    std::optional<RetainedHostAllocationLedger::Ticket> host_credit;
    std::optional<RetainedCudaRegistrationLedger::Ticket> registration_credit;
    std::optional<RetainedDescriptorLedger::Ticket> object_credit;
    Exl3SharedControlCredit* physical_control=nullptr;
    static bool attach_registration_credit(const std::shared_ptr<const void>& owner,RetainedCudaRegistrationLedger::Ticket credit) noexcept {
        auto* slab=const_cast<Exl3RecurrentSlab*>(static_cast<const Exl3RecurrentSlab*>(owner.get()));
        if(!slab || !slab->registered || !slab->data || slab->retirement_attempted || slab->registration_credit || credit.bytes()!=slab->bytes)return false;
        slab->registration_credit.emplace(std::move(credit));return true;
    }
    static bool attach_object_credit(const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* slab=const_cast<Exl3RecurrentSlab*>(static_cast<const Exl3RecurrentSlab*>(owner.get()));
        if(!slab || slab->retirement_attempted || slab->object_credit || credit.bytes()!=sizeof(Exl3RecurrentSlab))return false;
        slab->object_credit.emplace(std::move(credit));return true;
    }
    static bool attach_control_credit(const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* slab=const_cast<Exl3RecurrentSlab*>(static_cast<const Exl3RecurrentSlab*>(owner.get()));
        if(!slab || slab->retirement_attempted || !slab->physical_control || slab->physical_control->ticket ||
            credit.bytes()!=control_metadata_bytes())return false;
        slab->physical_control->ticket.emplace(std::move(credit));return true;
    }
    bool attach_retirement_credits(RetainedHostAllocationLedger::Ticket host,
        RetainedCudaRegistrationLedger::Ticket registration,RetainedDescriptorLedger::Ticket metadata) noexcept {
        if(!data || !bytes || !registered || retirement_attempted || host_credit || registration_credit || object_credit ||
            host.bytes()!=bytes || registration.bytes()!=bytes || metadata.bytes()!=sizeof(Exl3RecurrentSlab))return false;
        host_credit.emplace(std::move(host));registration_credit.emplace(std::move(registration));
        object_credit.emplace(std::move(metadata));return true;
    }
    std::atomic<bool> borrowed{false};
    bool retirement_attempted=false;
    int cleanup_error=0;
    Exl3RecurrentSlab* quarantine_next=nullptr;
    inline static std::atomic<Exl3RecurrentSlab*> quarantine_head{nullptr};
    inline static std::atomic<std::uint64_t> quarantine_count{0};
    struct NativeCleanup {
        int current(int* value) noexcept {return static_cast<int>(cudaGetDevice(value));}
        int drain() noexcept {return static_cast<int>(cudaDeviceSynchronize());}
        int wait(cudaEvent_t event) noexcept {return static_cast<int>(cudaEventSynchronize(event));}
        int unregister(void* pointer) noexcept {return static_cast<int>(cudaHostUnregister(pointer));}
        int destroy(cudaEvent_t event) noexcept {return static_cast<int>(cudaEventDestroy(event));}
        int free(void* pointer) noexcept {return VirtualFree(pointer,0,MEM_RELEASE)?0:-1;}
    };
    // Called once by the final shared owner. Do not switch devices or retry an
    // uncertain release. Preserve the slab itself, including event identity.
    template<class Cleanup> static void retire_with(Exl3RecurrentSlab* slab,Cleanup& cleanup) noexcept {
        if(!slab || slab->retirement_attempted)return;
        slab->retirement_attempted=true;
        slab->physical_control=nullptr; // its credit lives through final weak control release
        int error=0;
        if(slab->registered || slab->completed || slab->recorded || slab->poisoned) {
            int current=-1;error=cleanup.current(&current);
            if(!error && current!=slab->device)error=static_cast<int>(cudaErrorInvalidDevice);
            if(!error && slab->poisoned)error=cleanup.drain();
            if(!error && slab->recorded) {
                error=slab->completed?cleanup.wait(slab->completed):static_cast<int>(cudaErrorInvalidResourceHandle);
                if(error && slab->completed)error=cleanup.drain();
            }
        }
        if(!error && slab->registered) {
            error=slab->fail_unregister_for_test?static_cast<int>(cudaErrorUnknown):cleanup.unregister(slab->data);
            if(!error){slab->registered=false;slab->registration_credit.reset();}
        }
        if(!error && slab->completed) {
            error=cleanup.destroy(slab->completed);
            if(!error){slab->completed=nullptr;slab->recorded=false;}
        }
        if(!error && slab->data) {
            error=cleanup.free(slab->data);
            if(!error){slab->data=nullptr;slab->host_credit.reset();}
        }
        if(error) {
            slab->cleanup_error=error;slab->poisoned=true;
            if(slab->budget_claimed)Exl3RecurrentPinBudget::quarantine(slab->bytes);
            auto* head=quarantine_head.load(std::memory_order_relaxed);
            do{slab->quarantine_next=head;}while(!quarantine_head.compare_exchange_weak(
                head,slab,std::memory_order_release,std::memory_order_relaxed));
            quarantine_count.fetch_add(1,std::memory_order_release);return;
        }
        if(slab->budget_claimed)Exl3RecurrentPinBudget::release(slab->bytes);
        delete slab;
    }
    static void retire(Exl3RecurrentSlab* slab) noexcept {NativeCleanup cleanup;retire_with(slab,cleanup);}
    static std::shared_ptr<Exl3RecurrentSlab> create(std::optional<RetainedDescriptorLedger::Ticket> metadata=std::nullopt) {
        if(quarantine_count.load(std::memory_order_acquire))throw std::runtime_error("unresolved recurrent slab retirement");
        if(metadata && metadata->bytes()!=physical_metadata_bytes())throw std::invalid_argument("recurrent physical constructor metadata mismatch");
        std::optional<RetainedDescriptorLedger::Ticket> object;
        if(metadata) {
            auto part=metadata->split(sizeof(Exl3RecurrentSlab));
            if(!part)throw std::logic_error("recurrent slab object metadata split");
            object.emplace(std::move(*part));
        }
        bool admitted=false;Exl3SharedControlCredit* header=nullptr;
        auto* slab=new Exl3RecurrentSlab;
        if(object)slab->object_credit.emplace(std::move(*object));
        auto owner=std::shared_ptr<Exl3RecurrentSlab>(slab,&Exl3RecurrentSlab::retire,
            Exl3ReconstructionControlAllocator<Exl3RecurrentSlab>(&admitted,&header));
        if(metadata)header->ticket.emplace(std::move(*metadata));
        owner->physical_control=header;return owner;
    }
    void release_constructor_credits_after_commit() noexcept {
        registration_credit.reset();object_credit.reset();
        if(physical_control)physical_control->ticket.reset();
    }
    static constexpr std::size_t control_metadata_bytes() noexcept {return Exl3ReconstructionControlAllocator<Exl3RecurrentSlab>::capacity;}
    static bool attach_borrower_control_credit(const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* control=const_cast<Exl3SharedControlCredit*>(static_cast<const Exl3SharedControlCredit*>(owner.get()));
        if(!control || control->ticket || credit.bytes()!=control_metadata_bytes())return false;
        control->ticket.emplace(std::move(credit));return true;
    }
    static constexpr std::size_t physical_metadata_bytes() noexcept {return sizeof(Exl3RecurrentSlab)+control_metadata_bytes();}
    static Exl3ResourceInventory::Requirement requirement(const Exl3RecurrentSlabLayout& layout) {
        if(!layout.bytes || layout.bytes>Exl3RecurrentSlabLayout::max_bytes ||
            layout.offsets.front()!=0 || layout.offsets.back()!=layout.bytes/sizeof(float) || layout.bytes%sizeof(float) ||
            !std::is_sorted(layout.offsets.begin(),layout.offsets.end()))
            throw std::invalid_argument("invalid recurrent slab requirement layout");
        Exl3ResourceInventory::Requirement result;result.configuration=layout.bytes;
        result.add(Exl3ResourceInventory::Domain::cuda_registered_host,1,layout.bytes);
        result.add(Exl3ResourceInventory::Domain::host_metadata,1,physical_metadata_bytes());return result;
    }
    static Exl3ResourceInventory resources(const std::shared_ptr<Exl3RecurrentSlab>& owner) {
        if(!owner || !owner->data || !owner->registered || !owner->completed || !owner->physical_control || owner->retirement_attempted)
            throw std::invalid_argument("incomplete recurrent slab inventory owner");
        Exl3ResourceInventory result;using Domain=Exl3ResourceInventory::Domain;
        result.add({owner,0,Domain::cuda_registered_host,owner->bytes,{},nullptr,nullptr,&attach_registration_credit});
        result.add({owner,1,Domain::host_metadata,sizeof(Exl3RecurrentSlab),{},&attach_object_credit});
        result.add({owner,2,Domain::host_metadata,control_metadata_bytes(),{},&attach_control_credit});return result;
    }
    // A separate control block represents one published/active borrower flight.
    // Inventory and pool owners may retain the physical slab independently.
    static std::shared_ptr<Exl3RecurrentSlab> borrow(const std::shared_ptr<Exl3RecurrentSlab>& owner,
        std::optional<RetainedDescriptorLedger::Ticket> metadata=std::nullopt,
        Exl3SharedControlCredit** control_out=nullptr) {
        if(control_out)*control_out=nullptr;
        if(metadata && metadata->bytes()!=control_metadata_bytes())
            throw std::invalid_argument("recurrent borrower metadata credit mismatch");
        if(!owner || !owner.use_count() || owner->poisoned || owner->retirement_attempted)return {};
        bool idle=false;
        if(!owner->borrowed.compare_exchange_strong(idle,true,std::memory_order_acq_rel))return {};
        // shared_ptr invokes this deleter if allocating its control block fails,
        // restoring admission without releasing the pool's physical owner.
        bool admitted=false;Exl3SharedControlCredit* header=nullptr;
        auto result=std::shared_ptr<Exl3RecurrentSlab>(owner.get(),[owner](Exl3RecurrentSlab*) noexcept {
            owner->borrowed.store(false,std::memory_order_release);
        },Exl3ReconstructionControlAllocator<Exl3RecurrentSlab>(&admitted,&header));
        if(metadata)header->ticket.emplace(std::move(*metadata));
        if(control_out)*control_out=header;
        return result;
    }
private:
    ~Exl3RecurrentSlab()=default;
public:
    std::span<const float> plane(std::size_t i)const noexcept{return {data+offsets[i],offsets[i+1]-offsets[i]};}
};
struct Exl3RecurrentExportStats {
    std::uint64_t exports=0,recurrent_bytes=0,pinned_exports=0,pool_hits=0,allocations=0,fallbacks=0,poisons=0;
    std::uint64_t event_create_failures=0;
    double total_ms=0,recurrent_ms=0,allocate_ms=0,register_ms=0,submit_ms=0,fence_ms=0;
    std::uint64_t pool_bytes=0,process_live_bytes=0,process_peak_bytes=0,quarantined_bytes=0;
    std::uint64_t planned_copy_ranges=0,copy_submissions=0,coalesced_copy_ranges=0;
    std::uint64_t fp16_copy_submissions=0,fp32_copy_submissions=0;
    std::uint64_t batched_copy_calls=0,batched_copy_ranges=0;
};
class Exl3RecurrentExportPool {
public:
    struct ConstructionCredits {
        std::optional<RetainedCudaRegistrationLedger::Ticket> registration;
        std::optional<RetainedDescriptorLedger::Ticket> metadata;
    };
    using GrowthFactory=std::function<std::shared_ptr<Exl3RecurrentSlab>(ConstructionCredits)>;
    // Synchronous admission only: factory references expire when admission
    // returns and may be invoked at most once, including failed construction.
    using GrowthAdmission=std::function<std::shared_ptr<Exl3RecurrentSlab>(const Exl3RecurrentSlabLayout&,const GrowthFactory&)>;
    using BorrowAdmission=std::function<std::shared_ptr<Exl3RecurrentSlab>(const std::shared_ptr<Exl3RecurrentSlab>&)>;
private:
    std::array<std::shared_ptr<Exl3RecurrentSlab>,3> slots_{};
    std::size_t slot_count_=0;
    GrowthAdmission admission_;
    BorrowAdmission borrow_admission_;
    unsigned constructor_fault_for_test_=0;
    std::shared_ptr<Exl3RecurrentSlab> borrow_ready(const std::shared_ptr<Exl3RecurrentSlab>& owner) {
        try{return borrow_admission_?borrow_admission_(owner):Exl3RecurrentSlab::borrow(owner);}
        catch(const Exl3ResourceReservationExhausted&){return {};}
    }
    using Clock=std::chrono::steady_clock;
    static double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
public:
    Exl3RecurrentExportStats stats;
    void set_growth_admission(GrowthAdmission admission) {admission_=std::move(admission);}
    void set_borrow_admission(BorrowAdmission admission) {borrow_admission_=std::move(admission);}
    // Fresh, serialized owner only. Consumed after registration: 1 retains the
    // failed unregister owner; 2 permits ordinary cleanup of the same exception.
    // 3/4 inject event-create failure with failed/ordinary unregister cleanup.
    void fail_constructor_for_test(unsigned fault) {
        if(slot_count_ || constructor_fault_for_test_ || fault<1 || fault>4)
            throw std::invalid_argument("recurrent constructor fixture requires fresh pool and fault 1..4");
        constructor_fault_for_test_=fault;
    }
    // One serialized physical execution owner. Returned writable storage becomes
    // immutable before publication, and cannot be acquired again until all roots retire.
    std::shared_ptr<Exl3RecurrentSlab> acquire(const std::array<std::size_t,48>& elements){
        const auto layout=Exl3RecurrentSlabLayout::derive(elements);
        if(!layout || Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire)){++stats.fallbacks;return {};}
        for(auto& slot:slots_)if(slot && !slot->borrowed.load(std::memory_order_acquire) && !slot->poisoned &&
            slot->bytes==layout->bytes && slot->offsets==layout->offsets){
            const auto status=slot->recorded?cudaEventQuery(slot->completed):cudaSuccess;
            if(status==cudaSuccess){auto borrowed=borrow_ready(slot);if(borrowed){++stats.pool_hits;return borrowed;}++stats.fallbacks;return {};}
            if(status!=cudaErrorNotReady){slot->poisoned=true;++stats.poisons;}
        }
        if(slot_count_==slots_.size()){++stats.fallbacks;return {};}
        const auto bytes=layout->bytes;
        bool factory_invoked=false;
        const GrowthFactory factory=[&](ConstructionCredits credits) -> std::shared_ptr<Exl3RecurrentSlab> {
            if(std::exchange(factory_invoked,true))
                throw std::logic_error("recurrent growth factory already consumed");
            if(credits.registration.has_value()!=credits.metadata.has_value() ||
                (credits.registration && credits.registration->bytes()!=bytes))
                throw std::invalid_argument("recurrent constructor registration credit mismatch");
            auto slab=Exl3RecurrentSlab::create(std::move(credits.metadata));
            if(!Exl3RecurrentPinBudget::claim(bytes))return {};
            slab->bytes=bytes;slab->budget_claimed=true;
            if(cudaGetDevice(&slab->device)!=cudaSuccess)return {};
            slab->offsets=layout->offsets;
            auto begin=Clock::now();slab->data=static_cast<float*>(VirtualAlloc(nullptr,bytes,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
            stats.allocate_ms+=ms(begin);
            if(!slab->data)return {};
            begin=Clock::now();const auto registration=cudaHostRegister(slab->data,bytes,cudaHostRegisterDefault);
            stats.register_ms+=ms(begin);
            if(registration!=cudaSuccess)return {};
            slab->registered=true;
            if(credits.registration)slab->registration_credit.emplace(std::move(*credits.registration));
            const auto fault=std::exchange(constructor_fault_for_test_,0U);
            if(fault==1 || fault==2) {
                slab->fail_unregister_for_test=fault==1;
                throw std::runtime_error("injected recurrent post-registration constructor failure");
            }
            const auto event_status=fault>=3?cudaErrorMemoryAllocation:
                cudaEventCreateWithFlags(&slab->completed,cudaEventDisableTiming);
            if(event_status!=cudaSuccess) {
                ++stats.event_create_failures;
                slab->construction_error=event_status;
                slab->fail_unregister_for_test=fault==3;
                throw Exl3RecurrentEventCreationFailure(event_status);
            }
            return slab;
        };
        std::shared_ptr<Exl3RecurrentSlab> slab;
        try{slab=admission_?admission_(*layout,factory):factory({});}
        catch(const Exl3RecurrentEventCreationFailure&) {
            // An unresolved registration must not become a pageable fallback.
            // Engine normally wraps it first; direct pool callers fail closed too.
            if(Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire))throw;
            ++stats.fallbacks;return {};
        }
        catch(const Exl3ResourceReservationExhausted&){++stats.fallbacks;return {};}
        if(!slab){++stats.fallbacks;return {};}
        if(slab->bytes!=layout->bytes || slab->offsets!=layout->offsets)
            throw std::logic_error("recurrent growth admission changed requested layout");
        slots_[slot_count_++]=slab;++stats.allocations;stats.pool_bytes+=bytes;
        auto borrowed=borrow_ready(slab);if(!borrowed)++stats.fallbacks;return borrowed;
    }
    Exl3RecurrentExportStats snapshot()const{
        auto result=stats;auto process=Exl3RecurrentPinBudget::snapshot();
        result.process_live_bytes=process[0];result.process_peak_bytes=process[1];result.quarantined_bytes=process[2];return result;
    }
};
}
