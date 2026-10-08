#pragma once
#include "exl3/text_model.h"
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/reconstruction_control_allocator.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>
#include <utility>

namespace ninfer::exl3 {
// Redundant represented KV only. Authoritative pages and publication remain on
// the host; these weak tags must never perturb page COW or retain a host root.
class Exl3DevicePrefixCache : public std::enable_shared_from_this<Exl3DevicePrefixCache> {
    struct RetainedAllocation {
        void* pointer=nullptr;
        std::uint64_t bytes=0;
        int device=-1;
        int error=0;
        std::optional<RetainedDeviceLedger::Ticket> device_credit;
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        RetainedAllocation* next=nullptr;
    };
public:
    // A shared cache is reused only while one complete layer-stack consumer
    // owns it. Contention falls back to authoritative HostKV; no waiter blocks
    // the worker needed by a projection rendezvous. Failed completion permanently
    // disables the cache while retaining its storage in the owning Engine.
    class Use {
        std::shared_ptr<Exl3DevicePrefixCache> cache_;
        std::unique_lock<std::mutex> lock_;
        bool completed_=false;
        bool engaged_=false;
    public:
        explicit Use(std::shared_ptr<Exl3DevicePrefixCache> cache):cache_(std::move(cache)),lock_(cache_->use_mutex_,std::try_to_lock) {
            engaged_=lock_.owns_lock() && cache_->admitted() && !cache_->poisoned_ &&
                !cache_->drained_ && !quarantined_.load();
        }
        Use(Use&& other) noexcept:cache_(std::move(other.cache_)),lock_(std::move(other.lock_)),
            completed_(other.completed_),engaged_(other.engaged_) {}
        Use(const Use&)=delete;
        ~Use(){if(cache_ && engaged_ && lock_.owns_lock() && !completed_)cache_->poisoned_=true;}
        explicit operator bool() const noexcept {return cache_ && engaged_ && lock_.owns_lock() && !completed_ &&
            !cache_->poisoned_ && !cache_->drained_ && !quarantined_.load();}
        void complete() noexcept {completed_=true;}
    };
    Use try_use(){return Use(shared_from_this());}
    static constexpr int tokens=4096, pages=tokens/Exl3ExactKVPage::token_capacity;
    static constexpr std::uint64_t bytes=std::uint64_t(tokens)*1024*2*32;
    static constexpr std::uint64_t cap=1ULL<<30, reserve=2ULL<<30;
    static std::uint64_t allocation_bytes_required(int capacity) {
        if(capacity!=4096&&capacity!=16384)throw std::invalid_argument("device prefix row menu4096/16384");
        return std::uint64_t(capacity)*1024*2*32;
    }
    static std::uint64_t owner_metadata_bytes_required(int capacity) {
        (void)allocation_bytes_required(capacity);
        return sizeof(Exl3DevicePrefixCache)+sizeof(RetainedAllocation)+control_metadata_bytes()+
            std::uint64_t(capacity/64)*(sizeof(std::weak_ptr<const Exl3ExactKVPage>)+
                sizeof(std::uint8_t));
    }
    static constexpr std::uint64_t control_metadata_bytes() noexcept {
        return Exl3ReconstructionControlAllocator<Exl3DevicePrefixCache>::capacity;
    }
    std::uint64_t owner_metadata_bytes() const {
        return owner_metadata_bytes_required(tokens_)-(control_?0:control_metadata_bytes());
    }
    static constexpr std::uint64_t retirement_metadata_bytes() noexcept {return sizeof(RetainedAllocation);}
    static bool attach_device_credit(const std::shared_ptr<const void>& owner,RetainedDeviceLedger::Ticket credit) noexcept {
        auto* cache=const_cast<Exl3DevicePrefixCache*>(static_cast<const Exl3DevicePrefixCache*>(owner.get()));
        if(!cache || !cache->data_ || !cache->retirement_ || cache->retirement_->device_credit || credit.bytes()!=cache->bytes_)
            return false;
        cache->retirement_->device_credit.emplace(std::move(credit));return true;
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* cache=const_cast<Exl3DevicePrefixCache*>(static_cast<const Exl3DevicePrefixCache*>(owner.get()));
        if(!cache || !cache->control_ || cache->control_->ticket || !cache->retirement_ ||
            cache->retirement_->metadata_credit || cache->object_metadata_credit_ ||
            credit.bytes()!=cache->owner_metadata_bytes())return false;
        auto record=credit.split(sizeof(RetainedAllocation));
        if(!record)return false;
        auto control=credit.split(control_metadata_bytes());
        if(!control)return false;
        cache->retirement_->metadata_credit.emplace(std::move(*record));
        cache->control_->ticket.emplace(std::move(*control));
        cache->object_metadata_credit_.emplace(std::move(credit));return true;
    }
    void release_constructor_credits_after_commit() noexcept {
        retirement_->device_credit.reset();retirement_->metadata_credit.reset();object_metadata_credit_.reset();
        if(control_)control_->ticket.reset();
    }
    std::array<std::uint64_t,2> attached_credits_for_test() const noexcept {
        return {retirement_->device_credit?retirement_->device_credit->bytes():0,
            (retirement_->metadata_credit?retirement_->metadata_credit->bytes():0)+
            (object_metadata_credit_?object_metadata_credit_->bytes():0)+
            (control_ && control_->ticket?control_->ticket->bytes():0)};
    }
    static Exl3ResourceInventory resources(const std::shared_ptr<Exl3DevicePrefixCache>& cache) {
        if(!cache || !cache->admitted() || !cache->control_)
            throw std::invalid_argument("prefix inventory requires allocated bounded owner");
        Exl3ResourceInventory result;using Domain=Exl3ResourceInventory::Domain;
        result.add({cache,0,Domain::device,cache->allocation_bytes(),{},nullptr,&attach_device_credit});
        result.add({cache,1,Domain::host_metadata,cache->owner_metadata_bytes(),{},&attach_metadata_credit});
        return result;
    }
    explicit Exl3DevicePrefixCache(int capacity=tokens) {
        if(quarantined_.load())throw std::runtime_error("unresolved device prefix cleanup");
        bytes_=allocation_bytes_required(capacity);
        tokens_=capacity;tag_count_=capacity/64;
        tags_=std::make_unique<std::weak_ptr<const Exl3ExactKVPage>[]>(tag_count_);
        tag_rows_=std::make_unique<std::uint8_t[]>(tag_count_);
        // Serialize optional reservations among contexts, including the free
        // memory query and allocation. Other required allocations remain owned
        // by the normal full-context resource admission.
        std::lock_guard<std::mutex> lock(admission_);
        std::size_t available=0,total=0;
        if(cudaMemGetInfo(&available,&total)!=cudaSuccess)throw std::runtime_error("device prefix memory query failed");
        if(available<bytes_+reserve)return;
        if(cudaGetDevice(&device_)!=cudaSuccess)throw std::runtime_error("device prefix allocation device query failed");
        auto used=budget_.load();
        do {if(used>cap-bytes_)return;} while(!budget_.compare_exchange_weak(used,used+bytes_));
        const auto error=cudaMalloc(&data_,bytes_);
        if(error!=cudaSuccess){budget_.fetch_sub(bytes_);data_=nullptr;if(error==cudaErrorMemoryAllocation){cudaGetLastError();return;}throw std::runtime_error("device prefix allocation failed");}
    }
    // All shared production owners use one physically bounded control block.
    // Optional credits already exist before any physical allocation starts.
    static std::shared_ptr<Exl3DevicePrefixCache> create(int capacity=tokens,
        std::optional<Exl3VeriCacheServingCoordinator::ConstructorCredits> credits=std::nullopt,
        unsigned control_fault_for_test=0) {
        if(control_fault_for_test>2)throw std::invalid_argument("prefix control fault1/2");
        if(credits && (credits->device.bytes()!=allocation_bytes_required(capacity) ||
            credits->metadata.bytes()!=owner_metadata_bytes_required(capacity)))
            throw std::invalid_argument("prefix constructor credit extent mismatch");
        std::optional<RetainedDescriptorLedger::Ticket> control_credit;
        if(credits) {
            auto part=credits->metadata.split(control_metadata_bytes());
            if(!part)throw std::logic_error("prefix control metadata split");
            control_credit.emplace(std::move(*part));
        }
        auto raw=std::make_unique<Exl3DevicePrefixCache>(capacity);
        // The native constructor has no throwing operation after cudaMalloc.
        // Adopt all acquired-storage credits before shared_ptr can allocate.
        if(credits) {
            raw->retirement_->device_credit.emplace(std::move(credits->device));
            auto record=credits->metadata.split(sizeof(RetainedAllocation));
            if(!record)throw std::logic_error("prefix retirement metadata split");
            raw->retirement_->metadata_credit.emplace(std::move(*record));
            raw->object_metadata_credit_.emplace(std::move(credits->metadata));
        }
        const bool inject=control_fault_for_test && raw->admitted();
        if(inject)control_fault_attempts_.fetch_add(1,std::memory_order_relaxed);
        raw->cleanup_failure_for_test_=inject && control_fault_for_test==2;
        bool admitted=inject;Exl3SharedControlCredit* header=nullptr;
        auto owner=std::shared_ptr<Exl3DevicePrefixCache>(raw.release(),
            [](Exl3DevicePrefixCache* value) noexcept {delete value;},
            Exl3ReconstructionControlAllocator<Exl3DevicePrefixCache>(&admitted,&header));
        if(control_credit)header->ticket.emplace(std::move(*control_credit));
        owner->control_=header;return owner;
    }
    // Optional cache admission: reserve before invoking the existing physical
    // allocator. A cache declined by its independent cap/free-memory policy
    // rolls back the credit and leaves authoritative HostKV as the fallback.
    static std::shared_ptr<Exl3DevicePrefixCache> create_reserved(
        Exl3VeriCacheServingCoordinator& authority,int capacity=tokens,unsigned startup_fault_for_test=0) {
        if(startup_fault_for_test>7)throw std::invalid_argument("prefix startup fault stage1..7");
        struct OptionalDeclined {};
        using Inventory=Exl3ResourceInventory;
        Inventory::Requirement required;required.configuration=0x5052454649584B56;
        required.add(Inventory::Domain::device,1,allocation_bytes_required(capacity));
        required.add(Inventory::Domain::host_metadata,1,owner_metadata_bytes_required(capacity));
        std::shared_ptr<Exl3DevicePrefixCache> prepared;
        const auto quarantine_before=quarantined_.load();
        try {
            authority.allocate_startup_resources(required,[&](auto configuration) {
                if(configuration!=required.configuration)throw std::logic_error("prefix startup reservation identity");
                auto credits=authority.reserve_constructor_credits(allocation_bytes_required(capacity),owner_metadata_bytes_required(capacity));
                prepared=create(capacity,std::move(credits),startup_fault_for_test>=6?startup_fault_for_test-5:0);
                if(!prepared->admitted())throw OptionalDeclined{};
                if(startup_fault_for_test==1)throw std::runtime_error("injected prefix startup precommit failure");
                if(startup_fault_for_test==3)prepared->cleanup_failure_for_test_=true;
                prepared->device_query_failure_for_test_=startup_fault_for_test==4;
                prepared->device_mismatch_for_test_=startup_fault_for_test==5;
                Inventory actual;
                actual.add({prepared,0,Inventory::Domain::device,
                    prepared->allocation_bytes()-(startup_fault_for_test>=2?1:0),{},nullptr,&attach_device_credit});
                actual.add({prepared,1,Inventory::Domain::host_metadata,
                    prepared->owner_metadata_bytes(),{},&attach_metadata_credit});
                return actual;
            },[&]() noexcept {
                prepared.reset();
                if(quarantined_.load()!=quarantine_before)
                    authority.seal_failed_startup_retirement();
            });
        } catch(const OptionalDeclined&) {return {};}
          catch(const Exl3ResourceReservationExhausted&) {return {};}
        prepared->release_constructor_credits_after_commit();
        return prepared;
    }
    ~Exl3DevicePrefixCache(){if(data_){
        const auto error=cleanup_device();
        if(error==cudaSuccess){budget_.fetch_sub(bytes_);return;}
        auto* retained=retirement_.release();
        retained->pointer=data_;retained->bytes=bytes_;retained->device=device_;
        retained->error=static_cast<int>(error);
        auto* head=retained_.load(std::memory_order_relaxed);
        do{retained->next=head;}while(!retained_.compare_exchange_weak(
            head,retained,std::memory_order_release,std::memory_order_relaxed));
        quarantined_.fetch_add(bytes_);
    }}
    // Caller owns the sole cache reference and proves no submitted prefix work.
    // Failed free leaves storage/accounting attached until owner retirement.
    bool retire_pristine() {
        std::unique_lock<std::mutex> lock(use_mutex_,std::try_to_lock);
        if(!lock.owns_lock() || !reusable())return false;
        for(int i=0;i<tag_count_;++i)if(!tags_[i].expired())return false;
        const auto error=cleanup_device();
        if(error!=cudaSuccess){cleanup_error_=error;poisoned_=true;return false;}
        data_=nullptr;drained_=true;budget_.fetch_sub(bytes_);retirement_->device_credit.reset();return true;
    }
    void fail_cleanup_for_test() {
        if(!reusable() || cleanup_failure_for_test_)
            throw std::logic_error("prefix cleanup injection requires unarmed reusable cache");
        cleanup_failure_for_test_=true;
    }
    struct RetirementSnapshot {
        std::uintptr_t pointer=0;std::uint64_t bytes=0;int device=-1,error=0;
        std::uint64_t device_credit=0,metadata_credit=0;
    };
    static RetirementSnapshot retirement_snapshot_for_test() noexcept {
        const auto* retained=retained_.load(std::memory_order_acquire);
        return retained?RetirementSnapshot{reinterpret_cast<std::uintptr_t>(retained->pointer),
            retained->bytes,retained->device,retained->error,
            retained->device_credit?retained->device_credit->bytes():0,
            retained->metadata_credit?retained->metadata_credit->bytes():0}:RetirementSnapshot{};
    }
    // Owning Engine has joined every producer and completed its device drain.
    void retire_after_device_drain() noexcept {drained_=true;}
    Exl3DevicePrefixCache(const Exl3DevicePrefixCache&)=delete;
    Exl3DevicePrefixCache& operator=(const Exl3DevicePrefixCache&)=delete;
    bool admitted()const noexcept{return data_!=nullptr;}
    bool reusable()const noexcept{return admitted() && !poisoned_ && !drained_ && !quarantined_.load();}
    int capacity_tokens()const noexcept{return tokens_;}
    std::uint64_t allocation_bytes()const noexcept{return data_?bytes_:0;}
    void* plane(int bank,bool key)const {
        if(!data_ || bank<0 || bank>=16)throw std::invalid_argument("device prefix plane requires allocated bank0..15");
        if(poisoned_ || drained_)throw std::logic_error("device prefix plane is retired or uncertain");
        if(quarantined_.load())throw std::logic_error("device prefix plane blocked by unresolved cleanup");
        return static_cast<std::byte*>(data_)+(bank*2+(key?0:1))*std::uint64_t(tokens_)*1024*2;
    }
    void invalidate()noexcept{for(int i=0;i<tag_count_;++i){tags_[i].reset();tag_rows_[i]=0;}}
    int matched_rows(const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& source,
        int position,bool allow_partial=false)const {
        if(!admitted() || poisoned_ || drained_ || quarantined_.load())return 0;
        int count=0;
        while(count<tag_count_&&count<source.size()&&count*64<position){
            const auto& page=source[count];
            const int rows=std::min(64,position-count*64);
            if(!page||page->first!=count*64||page->rows<rows||
                (rows!=64 && !allow_partial))break;
            const auto cached=tags_[count].lock();
            if(!cached||cached!=page||tag_rows_[count]!=rows)break;
            ++count;
        }
        return std::min(count*64,position);
    }
    // Only after all bank copies and authoritative host publication fences.
    void publish_tags(const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& source,
        int rows,bool allow_partial=false){
        if(!admitted() || poisoned_ || drained_ || quarantined_.load()){invalidate();return;}
        for(int i=0;i<tag_count_;++i){
            const int represented=std::clamp(rows-i*64,0,64);
            if(represented>0 && (represented==64||allow_partial) && i<source.size()&&
                source[i]&&source[i]->first==i*64&&source[i]->rows>=represented) {
                tags_[i]=source[i];tag_rows_[i]=static_cast<std::uint8_t>(represented);
            } else {tags_[i].reset();tag_rows_[i]=0;}
        }
    }
    static std::array<std::uint64_t,2> budget_snapshot(){return {budget_.load(),quarantined_.load()};}
    static std::uint64_t control_fault_attempts_for_test() noexcept {return control_fault_attempts_.load();}
private:
    cudaError_t cleanup_device() noexcept {
        if(cleanup_error_!=cudaSuccess)return cleanup_error_;
        int current=-1;
        auto error=cleanup_failure_for_test_ || (poisoned_ && !drained_)?cudaErrorUnknown:
            (device_query_failure_for_test_?cudaErrorInitializationError:cudaGetDevice(&current));
        if(error==cudaSuccess && (device_mismatch_for_test_ || current!=device_))error=cudaErrorInvalidDevice;
        return error==cudaSuccess?cudaFree(data_):error;
    }
    cudaError_t cleanup_error_=cudaSuccess;
    std::unique_ptr<RetainedAllocation> retirement_=std::make_unique<RetainedAllocation>();
    std::optional<RetainedDescriptorLedger::Ticket> object_metadata_credit_;
    Exl3SharedControlCredit* control_=nullptr;
    inline static std::atomic<RetainedAllocation*> retained_{nullptr};
    int device_=-1;
    std::mutex use_mutex_;
    bool poisoned_=false;
    bool drained_=false;
    bool cleanup_failure_for_test_=false;
    bool device_query_failure_for_test_=false;
    bool device_mismatch_for_test_=false;
    void* data_=nullptr;
    int tokens_=tokens;
    std::uint64_t bytes_=bytes;
    int tag_count_=0;
    std::unique_ptr<std::weak_ptr<const Exl3ExactKVPage>[]> tags_;
    std::unique_ptr<std::uint8_t[]> tag_rows_;
    inline static std::atomic<std::uint64_t> budget_{0},quarantined_{0};
    inline static std::atomic<std::uint64_t> control_fault_attempts_{0};
    inline static std::mutex admission_;
};
}
