#pragma once
#include "exl3/registered_kv_backing.h"
#include "exl3/kv_transfer_lease.h"
#include "exl3/kv_registration_cache.h"
#include "exl3/registered_kv_upload.h"
#include "exl3/device_page_storage.h"
#include "exl3/device_page_cache.h"
#include "exl3/device_page_fill_command.h"
#include "exl3/device_page_copy.h"
#include "exl3/device_page_readers.h"
#include "exl3/exact_page_extension_plan.h"
#include "exl3/device_page_attention.h"
#include "exl3/attention_stage.h"
#include "exl3/attention_stage_storage.h"
#include "exl3/attention_stage_binding.h"
#include "exl3/attention_stage_command.h"
#include "exl3/attention_stage_history.h"
#include "exl3/attention_stage_resources.h"
#include <cstring>
#include <new>
#include <functional>
#include <limits>
struct PreparedDevicePageStorageProvider {
    inline static unsigned allocations=0,releases=0;
    inline static cudaError_t release_error=cudaSuccess;
    static cudaError_t allocate(void** pointer,std::size_t bytes) noexcept {
        ++allocations;*pointer=::operator new(bytes,std::nothrow);
        return *pointer?cudaSuccess:cudaErrorMemoryAllocation;
    }
    static cudaError_t release(void* pointer) noexcept {
        ++releases;if(release_error!=cudaSuccess)return release_error;
        ::operator delete(pointer);return cudaSuccess;
    }
};
struct PreparedAttentionStageEvents {
    inline static unsigned creates=0,destroys=0,fail_at=0;
    inline static cudaError_t destroy_error=cudaSuccess;
    static cudaError_t create(cudaEvent_t* event) noexcept {
        ++creates;
        if(creates==fail_at)return cudaErrorMemoryAllocation;
        *event=reinterpret_cast<cudaEvent_t>(static_cast<std::uintptr_t>(creates));return cudaSuccess;
    }
    static cudaError_t destroy(cudaEvent_t) noexcept {++destroys;return destroy_error;}
};
struct PreparedKVUploadProvider {
    inline static cudaMemcpyKind expected_kind=cudaMemcpyHostToDevice;
    inline static unsigned copies=0,records=0,waits=0;
    inline static unsigned fail_copy_at=0;
    inline static cudaError_t copy_error=cudaSuccess,record_error=cudaSuccess,wait_error=cudaSuccess;
    static cudaError_t copy(void* out,const void* in,std::size_t bytes,cudaMemcpyKind kind,cudaStream_t) noexcept {
        if(kind!=expected_kind)return cudaErrorInvalidValue;
        ++copies;if(copies==fail_copy_at)return cudaErrorUnknown;
        if(copy_error!=cudaSuccess)return copy_error;
        std::memcpy(out,in,bytes);return cudaSuccess;
    }
    static cudaError_t record(cudaEvent_t,cudaStream_t) noexcept {++records;return record_error;}
    static cudaError_t wait(cudaEvent_t) noexcept {++waits;return wait_error;}
};
struct RefusingKVRegistrationAuthority {
    struct RegistrationCredits {
        ninfer::exl3::RetainedCudaRegistrationLedger::Ticket registration;
        ninfer::exl3::RetainedDescriptorLedger::Ticket metadata;
    };
    RegistrationCredits reserve_startup_registration_constructor_credits(std::uint64_t,std::uint64_t) {
        throw std::logic_error("refusing authority cannot grant constructor credits");
    }
    bool requested=false;
    void seal_failed_startup_retirement() noexcept {}
    template<class Factory,class Rollback,class Observe> void allocate_startup_resources(
        const ninfer::exl3::Exl3ResourceInventory::Requirement& required,Factory&& factory,Rollback&& rollback,Observe&& observe) {
        try{allocate_startup_resources(required,std::forward<Factory>(factory));}
        catch(...){rollback();observe();throw;}
    }
    template<class Factory> void allocate_startup_resources(
        const ninfer::exl3::Exl3ResourceInventory::Requirement& required,Factory&&) {
        using Domain=ninfer::exl3::Exl3ResourceInventory::Domain;
        if(required.units[static_cast<unsigned>(Domain::cuda_registered_host)]!=64ULL*1024*2 ||
           !required.units[static_cast<unsigned>(Domain::host_metadata)])
            throw std::logic_error("registration preflight omitted backing/metadata");
        requested=true;throw std::runtime_error("fixture registration budget exhausted");
    }
};
struct PreparedKVRegistrationProvider {
    inline static unsigned acquisitions=0,releases=0;
    inline static cudaError_t acquire_error=cudaSuccess,release_error=cudaSuccess;
    static cudaError_t acquire(void*,std::size_t) noexcept {++acquisitions;return acquire_error;}
    static cudaError_t release(void*) noexcept {++releases;return release_error;}
};
struct PreparedKVRegistrationAuthority {
    using Lease=unsigned;
    unsigned runtime_requests=0;
    unsigned host_source_bindings=0;
    bool host_sources_reserved=false;
    void reserve_runtime_host_sources(){host_sources_reserved=true;}
    bool fail_host_source_binding=false;
    void bind_runtime_host_source(const Lease& lease,const std::shared_ptr<const void>& lifetime,
        const ninfer::exl3::Exl3DevicePageKey& key) {
        if(!host_sources_reserved || lease!=7 || !lifetime || !key.current())throw std::invalid_argument("fixture host source binding");
        if(fail_host_source_binding)throw std::runtime_error("injected host source binding failure");
        ++host_source_bindings;
    }
    mutable unsigned lease_checks=0;
    unsigned expire_after_checks=0;
    bool exhaust=false,fail_factory=false,fail_commit=false;
    unsigned malformed_constructor_credit=0;
    unsigned malformed_constructor_credit_on_call=0,constructor_credit_calls=0;
    bool sealed=false,constructor_scope=false,runtime_scope=false;
    ninfer::exl3::Exl3ResourceInventory::Totals constructor_ceiling{},constructor_issued{};
    ninfer::exl3::RetainedDeviceLedger constructor_device;
    ninfer::exl3::RetainedDescriptorLedger constructor_metadata;
    ninfer::exl3::RetainedCudaRegistrationLedger constructor_registration;
    using RegistrationCredits=RefusingKVRegistrationAuthority::RegistrationCredits;
    RegistrationCredits issue_registration_credits(std::uint64_t registration,std::uint64_t metadata) {
        using Domain=ninfer::exl3::Exl3ResourceInventory::Domain;
        if(!constructor_scope || !registration || !metadata)throw std::logic_error("fixture registration constructor scope");
        const auto r=static_cast<unsigned>(Domain::cuda_registered_host),m=static_cast<unsigned>(Domain::host_metadata);
        if(registration>constructor_ceiling[r]-constructor_issued[r] || metadata>constructor_ceiling[m]-constructor_issued[m])
            throw ninfer::exl3::Exl3ResourceReservationExhausted{};
        const auto registration_grant=malformed_constructor_credit==1?registration-1:malformed_constructor_credit==2?registration+1:registration;
        const auto metadata_grant=malformed_constructor_credit==3?metadata-1:malformed_constructor_credit==4?metadata+1:metadata;
        auto a=constructor_registration.acquire(registration_grant);auto b=constructor_metadata.acquire(metadata_grant);
        constructor_issued[r]+=registration;constructor_issued[m]+=metadata;return {std::move(a),std::move(b)};
    }
    RegistrationCredits reserve_startup_registration_constructor_credits(std::uint64_t registration,std::uint64_t metadata) {
        if(runtime_scope)throw std::logic_error("fixture startup registration scope");
        return issue_registration_credits(registration,metadata);
    }
    RegistrationCredits reserve_registration_constructor_credits(std::uint64_t registration,std::uint64_t metadata) {
        if(!runtime_scope)throw std::logic_error("fixture runtime registration scope");
        return issue_registration_credits(registration,metadata);
    }
    struct ConstructorCredits {
        ninfer::exl3::RetainedDeviceLedger::Ticket device;
        ninfer::exl3::RetainedDescriptorLedger::Ticket metadata;
    };
    ConstructorCredits reserve_runtime_constructor_credits(std::uint64_t device,std::uint64_t metadata) {
        if(!runtime_scope)throw std::logic_error("fixture runtime constructor scope");
        return issue_constructor_credits(device,metadata);
    }
    ConstructorCredits reserve_constructor_credits(std::uint64_t device,std::uint64_t metadata) {
        if(runtime_scope)throw std::logic_error("fixture startup constructor scope");
        return issue_constructor_credits(device,metadata);
    }
    ConstructorCredits issue_constructor_credits(std::uint64_t device,std::uint64_t metadata) {
        using Domain=ninfer::exl3::Exl3ResourceInventory::Domain;
        if(!constructor_scope || !device || !metadata)throw std::logic_error("fixture constructor scope");
        const auto d=static_cast<unsigned>(Domain::device),m=static_cast<unsigned>(Domain::host_metadata);
        if(device>constructor_ceiling[d]-constructor_issued[d] || metadata>constructor_ceiling[m]-constructor_issued[m])
            throw ninfer::exl3::Exl3ResourceReservationExhausted{};
        ++constructor_credit_calls;
        const auto malformed=(!malformed_constructor_credit_on_call ||
            constructor_credit_calls==malformed_constructor_credit_on_call)?malformed_constructor_credit:0;
        const auto device_grant=malformed==1?device-1:malformed==2?device+1:device;
        const auto metadata_grant=malformed==3?metadata-1:malformed==4?metadata+1:metadata;
        auto a=constructor_device.acquire(device_grant);auto b=constructor_metadata.acquire(metadata_grant);
        constructor_issued[d]+=device;constructor_issued[m]+=metadata;return {std::move(a),std::move(b)};
    }
    void seal_failed_startup_retirement() noexcept {sealed=true;}
    std::function<void()> before_factory;
    ninfer::exl3::Exl3ResourceInventory retained;
    ninfer::exl3::RetainedDescriptorLedger retired_metadata;
    std::uint64_t registered_limit=std::numeric_limits<std::uint64_t>::max();
    bool compute_leases_current(std::span<const Lease> leases) const {
        ++lease_checks;
        return (!expire_after_checks || lease_checks<=expire_after_checks) && leases.size()==1 && leases[0]==7;
    }
    template<class Retire> bool retire_runtime_resource(const Lease& lease,
        const ninfer::exl3::Exl3ResourceInventory::Allocation& allocation,Retire&& retire) {
        if(lease!=7)throw std::invalid_argument("fixture retirement lease stale");
        auto next=retained.without_allocation(allocation);
        if(!retire(allocation))return false;
        retained=std::move(next);return true;
    }
    void transfer_retired_metadata(const Lease& lease,
        const ninfer::exl3::Exl3ResourceInventory::Allocation& old,
        const ninfer::exl3::Exl3ResourceInventory::Allocation& tracking) {
        if(lease!=7)throw std::invalid_argument("fixture metadata lease stale");
        retained=retained.transfer_metadata(old,tracking);
    }
    bool collect_retired_metadata(const Lease& lease,
        const ninfer::exl3::Exl3ResourceInventory::Allocation& tracking,const std::weak_ptr<const void>& old) {
        if(lease!=7)throw std::invalid_argument("fixture metadata collection lease stale");
        if(!retained.metadata_owner_expired(tracking,old))return false;
        auto next=retained.without_allocation(tracking);
        retained=std::move(next);return true;
    }
    template<class Factory> void allocate_runtime_resources(const Lease& lease,
        const ninfer::exl3::Exl3ResourceInventory::Requirement& required,Factory&& factory) {
        if(lease!=7)throw std::invalid_argument("fixture runtime lease stale");
        runtime_scope=true;
        struct RuntimeScope {bool& active;~RuntimeScope(){active=false;}} scope{runtime_scope};
        ++runtime_requests;allocate_startup_resources(required,std::forward<Factory>(factory));
    }
    template<class Factory,class Rollback,class Observe> void allocate_runtime_resources(const Lease& lease,
        const ninfer::exl3::Exl3ResourceInventory::Requirement& required,Factory&& factory,
        Rollback&& rollback,Observe&& observe) {
        try{allocate_runtime_resources(lease,required,std::forward<Factory>(factory));}
        catch(...){std::forward<Rollback>(rollback)();std::forward<Observe>(observe)();throw;}
    }
    bool retire_runtime_metadata_to_lifetime(const Lease& lease,
        const ninfer::exl3::Exl3ResourceInventory::Allocation& allocation) {
        if(lease!=7 || !allocation.lifetime_retire)throw std::invalid_argument("fixture metadata lifetime retirement");
        auto next=retained.without_allocation(allocation);
        if(!allocation.lifetime_retire(allocation.owner,retired_metadata.acquire(allocation.units)))return false;
        retained=std::move(next);return true;
    }
    template<class Factory> void allocate_startup_resources(
        const ninfer::exl3::Exl3ResourceInventory::Requirement& required,Factory&& factory) {
        if(sealed)throw std::runtime_error("fixture authority sealed after cleanup failure");
        if(exhaust)throw ninfer::exl3::Exl3ResourceReservationExhausted{};
        const auto domain=static_cast<unsigned>(ninfer::exl3::Exl3ResourceInventory::Domain::cuda_registered_host);
        const auto current=retained.totals()[domain];
        if(current>registered_limit || required.units[domain]>registered_limit-current)
            throw ninfer::exl3::Exl3ResourceReservationExhausted{};
        if(fail_factory)throw std::runtime_error("prepared unrelated allocation failure");
        if(before_factory)before_factory();
        auto actual=[&] {
            constructor_ceiling=required.units;constructor_issued={};constructor_scope=true;
            struct Scope {bool& active;~Scope(){active=false;}} scope{constructor_scope};
            return factory(required.configuration);
        }();
        if(actual.totals()!=required.units)throw std::runtime_error("registration inventory differs from reservation");
        if(fail_commit)throw std::runtime_error("injected post-factory inventory commit failure");
        retained.append(actual);
    }
    template<class Factory,class Rollback> void allocate_startup_resources(
        const ninfer::exl3::Exl3ResourceInventory::Requirement& required,Factory&& factory,Rollback&& rollback) {
        try{allocate_startup_resources(required,std::forward<Factory>(factory));}
        catch(...){std::forward<Rollback>(rollback)();throw;}
    }
    template<class Factory,class Rollback,class Observe> void allocate_startup_resources(
        const ninfer::exl3::Exl3ResourceInventory::Requirement& required,Factory&& factory,Rollback&& rollback,Observe&& observe) {
        try{allocate_startup_resources(required,std::forward<Factory>(factory));}
        catch(...){rollback();observe();throw;}
    }
};
struct PreparedAttentionStageBudget : PreparedKVRegistrationAuthority {
    std::uint64_t device_limit=0,metadata_limit=0;
    void (*after_rollback)(std::size_t)=nullptr;
    std::size_t rollback_expected_blocks=0;
    template<class Factory>
    void allocate_startup_resources(const ninfer::exl3::Exl3ResourceInventory::Requirement& required,Factory&& factory) {
        using Domain=ninfer::exl3::Exl3ResourceInventory::Domain;
        if(required.units[static_cast<unsigned>(Domain::device)]>device_limit ||
            required.units[static_cast<unsigned>(Domain::host_metadata)]>metadata_limit)
            throw ninfer::exl3::Exl3ResourceReservationExhausted{};
        PreparedKVRegistrationAuthority::allocate_startup_resources(required,std::forward<Factory>(factory));
    }
    template<class Factory,class Rollback>
    void allocate_startup_resources(const ninfer::exl3::Exl3ResourceInventory::Requirement& required,
        Factory&& factory,Rollback&& rollback) {
        try{allocate_startup_resources(required,std::forward<Factory>(factory));}
        catch(...){std::forward<Rollback>(rollback)();if(after_rollback)after_rollback(rollback_expected_blocks);throw;}
    }
    template<class Factory,class Rollback,class Observe>
    void allocate_startup_resources(const ninfer::exl3::Exl3ResourceInventory::Requirement& required,
        Factory&& factory,Rollback&& rollback,Observe&& observe) {
        try{allocate_startup_resources(required,std::forward<Factory>(factory));}
        catch(...){
            std::forward<Rollback>(rollback)();
            if(after_rollback)after_rollback(rollback_expected_blocks);
            std::forward<Observe>(observe)();
            throw;
        }
    }
};
inline void prepared_kv_registration_preflight() {
    using namespace ninfer::exl3;
    {
        using Cache=Exl3KVRegistrationCache;
        const auto before=bounded_shared_live_blocks_for_test<Cache>();
        // Observe while the authority still handles the failed transaction.
        // Eventual destruction after create_startup unwinds is insufficient.
        struct Observation {
            static void after_rollback(std::size_t expected) {
                if(bounded_shared_live_blocks_for_test<Cache>()!=expected)
                    throw std::runtime_error("registration cache result survived coordinator rollback");
            }
        };
        PreparedAttentionStageBudget budget;
        budget.metadata_limit=Cache::metadata_bytes();budget.fail_commit=true;
        budget.after_rollback=&Observation::after_rollback;
        budget.rollback_expected_blocks=before;
        bool original=false;
        try{(void)Cache::create_startup(budget);}catch(const std::runtime_error& error) {
            original=std::string_view(error.what())=="injected post-factory inventory commit failure";
        }
        if(!original || bounded_shared_live_blocks_for_test<Cache>()!=before ||
            budget.retained.totals()!=Exl3ResourceInventory::Totals{})
            throw std::runtime_error("registration cache failed commit retained ownership or replaced error");
        budget.fail_commit=false;budget.after_rollback=nullptr;
        auto retry=Cache::create_startup(budget);
        if(!retry || bounded_shared_live_blocks_for_test<Cache>()!=before+1)
            throw std::runtime_error("registration cache retry after failed commit");
    }
    {
        using Owner=Exl3RegisteredKVBacking;using Provider=PreparedKVRegistrationProvider;
        auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;page->k[0].resize(64*1024);
        const auto extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64);
        for(bool startup:{true,false})for(unsigned malformed=1;malformed<=4;++malformed) {
            PreparedKVRegistrationAuthority bad_credit;bad_credit.malformed_constructor_credit=malformed;
            const auto acquisitions=Provider::acquisitions,releases=Provider::releases;
            const auto blocks=bounded_shared_live_blocks_for_test<Owner>();
            bool refused=false;
            try {
                if(startup)(void)Owner::acquire_startup(bad_credit,extent,{Provider::acquire,Provider::release});
                else (void)Owner::acquire_runtime(bad_credit,7,extent,{Provider::acquire,Provider::release});
            } catch(const std::invalid_argument& error){refused=std::string_view(error.what())==
                "KV registration constructor credit extent";}
            if(!refused || Provider::acquisitions!=acquisitions || Provider::releases!=releases ||
                bounded_shared_live_blocks_for_test<Owner>()!=blocks ||
                bad_credit.constructor_registration.bytes() || bad_credit.constructor_metadata.bytes() ||
                bad_credit.retained.totals()!=Exl3ResourceInventory::Totals{})
                throw std::runtime_error("malformed registration grant created storage or provider ownership");
        }
        PreparedKVRegistrationAuthority authority;
        auto owner=Owner::acquire_startup(authority,extent,{Provider::acquire,Provider::release});
        if(!owner)throw std::runtime_error("registration lifetime fixture allocation");
        RetainedCudaRegistrationLedger ledger;const auto bytes=extent.backing().bytes;
        if(Owner::attach_registration_credit(owner,ledger.acquire(bytes-1)) ||
            !Owner::attach_registration_credit(owner,ledger.acquire(bytes)) ||
            Owner::attach_registration_credit(owner,ledger.acquire(bytes)))
            throw std::runtime_error("registration lifetime exact/duplicate ticket contract");
        auto reader=Owner::acquire_reader(owner,extent);
        if(!reader || owner->seal_idle() || ledger.bytes()!=bytes)
            throw std::runtime_error("registration lifetime released before active reader");
        {
            Exl3KVTransferLease transfer;
            auto destination=std::make_shared<int>(1);
            transfer.rebind_registered(extent,destination,std::move(*reader));reader.reset();
            const auto generation=transfer.begin(1,1,37);
            if(!transfer.finish(generation,0))throw std::runtime_error("registered transfer fixture completion");
            auto unrelated=std::make_shared<Exl3ExactKVPage>();unrelated->rows=64;
            unrelated->k[0].resize(64*1024);page->v[0].resize(64*1024);
            const std::array<Exl3ExactKVExtent,2> mismatches{
                Exl3ExactKVExtent::view(unrelated,0,Exl3ExactKVExtent::Plane::key,64),
                Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::value,64)};
            for(const auto& mismatch:mismatches) {
                auto wrong_reader=Owner::acquire_reader(owner,extent);
                if(!wrong_reader)throw std::runtime_error("registered transfer fixture reader acquisition");
                bool refused=false;
                try{transfer.rebind_registered(mismatch,std::make_shared<int>(2),std::move(*wrong_reader));}
                catch(const std::invalid_argument&){refused=true;}
                if(!refused || !transfer.complete() || transfer.source().data()!=extent.data() ||
                    owner->active_readers()!=1 || owner->seal_idle())
                    throw std::runtime_error("registered rebind accepted unrelated source or replaced completed owner");
            }
            auto retry=Owner::acquire_reader(owner,extent);
            if(!retry)throw std::runtime_error("registered transfer retry reader acquisition");
            transfer.rebind_registered(extent,destination,std::move(*retry));
            const auto next=transfer.begin(1,2,37);
            if(next==generation || transfer.finish(generation,0) || !transfer.finish(next,0))
                throw std::runtime_error("registered rebind failed to preserve event generation");
            transfer.clear_completed();
        }
        reader.reset();
        if(!owner->seal_idle() || !owner->retire_sealed() || ledger.bytes())
            throw std::runtime_error("successful unregister did not release registration ticket");
        if(Owner::attach_registration_credit(owner,ledger.acquire(bytes)))
            throw std::runtime_error("retired registration accepted new byte ticket");
    }
    if(const auto* mode=std::getenv("NINFER_TEST_REGISTRATION_COMMIT_CLEANUP");mode && *mode) {
        using Owner=Exl3RegisteredKVBacking;using Provider=PreparedKVRegistrationProvider;
        const std::string_view selected(mode);
        if(selected!="release" && selected!="retain")throw std::invalid_argument("registration commit cleanup selector");
        const bool retain=selected=="retain";
        const bool startup=std::getenv("NINFER_TEST_REGISTRATION_COMMIT_STARTUP") &&
            std::string_view(std::getenv("NINFER_TEST_REGISTRATION_COMMIT_STARTUP"))=="1";
        auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;page->k[0].resize(64*1024);
        const auto extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64);
        PreparedKVRegistrationAuthority authority;authority.fail_commit=true;
        const auto releases=Provider::releases;
        const auto before=Owner::quarantined_bytes();
        Provider::release_error=retain?cudaErrorUnknown:cudaSuccess;
        bool original=false;
        try{
            if(startup)Owner::acquire_startup(authority,extent,{Provider::acquire,Provider::release});
            else Owner::acquire_runtime(authority,7,extent,{Provider::acquire,Provider::release});
        }
        catch(const std::runtime_error& error){original=std::string_view(error.what())=="injected post-factory inventory commit failure";}
        if(!original || authority.sealed!=retain || Provider::releases!=releases+1 ||
            Owner::quarantined_bytes()!=before+(retain?extent.backing().bytes:0) ||
            authority.retained.totals()!=Exl3ResourceInventory::Totals{})
            throw std::runtime_error("registration commit rollback lost cleanup state or original error");
        if(authority.constructor_registration.bytes()!=(retain?extent.backing().bytes:0) ||
            authority.constructor_metadata.bytes()!=(retain?Owner::state_metadata_bytes():0))
            throw std::runtime_error("registration commit cleanup lost exact constructor survivor credits");
        Provider::release_error=cudaSuccess;
        if(!retain) {
            authority.fail_commit=false;
            auto retry=startup?Owner::acquire_startup(authority,extent,{Provider::acquire,Provider::release}):
                Owner::acquire_runtime(authority,7,extent,{Provider::acquire,Provider::release});
            if(!retry)throw std::runtime_error("clean registration commit rollback blocked retry");
            if(authority.constructor_registration.bytes() || authority.constructor_metadata.bytes())
                throw std::runtime_error("registration successful retry retained provisional credits");
        }
        return; // Failed-unregister mode deliberately quarantines; separate process.
    }
    {
        using Plan=Exl3ExactPageExtensionPlan;
        {
            Exl3ExactKVPage tail;tail.first=64;tail.rows=1;
            for(int bank=0;bank<16;++bank) {
                tail.k[bank].reserve(64*1024);tail.v[bank].reserve(64*1024);
                tail.k[bank].assign(1024,0x1234);tail.v[bank].assign(1024,0x4321);
            }
            auto incomplete=tail; // vector copy need not retain spare capacity
            incomplete.v[15].clear();
            if(exl3_exact_tail_preallocated(incomplete))throw std::runtime_error("tail preflight missed final plane");
            bool refused=false;
            try{exl3_extend_preallocated_tail(incomplete,2);}catch(const std::invalid_argument&){refused=true;}
            if(!refused || incomplete.rows!=1 || incomplete.k[0].size()!=1024)
                throw std::runtime_error("tail preflight changed earlier plane before refusing");
            const auto* k=tail.k[0].data();const auto* v=tail.v[15].data();
            for(int rows:{0,65}) {
                refused=false;try{exl3_extend_preallocated_tail(tail,rows);}catch(const std::invalid_argument&){refused=true;}
                if(!refused || tail.rows!=1)throw std::runtime_error("tail extension accepted invalid rows");
            }
            exl3_extend_preallocated_tail(tail,64);
            if(tail.rows!=64 || tail.first!=64 || tail.k[0].data()!=k || tail.v[15].data()!=v)
                throw std::runtime_error("preallocated tail growth changed physical storage");
            for(int bank=0;bank<16;++bank)for(std::size_t row=0;row<64*1024;++row) {
                if(tail.k[bank][row]!=(row<1024?0x1234:0) || tail.v[bank][row]!=(row<1024?0x4321:0))
                    throw std::runtime_error("preallocated tail growth changed prefix or uninitialized suffix");
            }
        }
        for(const auto& item:std::array<std::array<int,4>,7>{{
            {0,0,0,0},{0,1,1,1},{1,1,1,0},{1,64,1,1},{64,65,2,1},{63,129,3,3},{128,256,4,2}}}) {
            const auto plan=Plan::derive(item[0],item[1]);
            if(plan.all!=static_cast<std::size_t>(item[2]) || plan.fresh!=static_cast<std::size_t>(item[3]))
                throw std::runtime_error("exact page descriptor plan boundary mismatch");
        }
        const auto maximum=std::numeric_limits<int>::max();
        const auto plan=Plan::derive(maximum-1,maximum);
        if(plan.all!=static_cast<std::size_t>(maximum)/64+1 || plan.fresh!=1)
            throw std::runtime_error("exact page descriptor plan signed endpoint overflow");
        for(const auto& pair:std::array<std::array<int,2>,3>{{{-1,0},{0,-1},{65,64}}}) {
            bool refused=false;try{Plan::derive(pair[0],pair[1]);}catch(const std::invalid_argument&){refused=true;}
            if(!refused)throw std::runtime_error("exact page descriptor plan accepted invalid lineage");
        }
    }
    {
        using Registration=Exl3RegisteredKVBacking;using IO=PreparedKVUploadProvider;
        using Provider=PreparedKVRegistrationProvider;
        PreparedKVRegistrationAuthority authority;
        const auto make_page=[] {
            auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;page->k[0].resize(64*1024);return page;
        };
        const auto first=Exl3ExactKVExtent::view(make_page(),0,Exl3ExactKVExtent::Plane::key,64);
        const auto second=Exl3ExactKVExtent::view(make_page(),0,Exl3ExactKVExtent::Plane::key,64);
        const Exl3KVRegistrationProvider registration_provider{Provider::acquire,Provider::release};
        auto active=Registration::acquire_startup(authority,first,registration_provider);
        auto retired=Registration::acquire_startup(authority,second,registration_provider);
        if(!active || !retired || !retired->seal_idle() || !retired->retire_sealed())
            throw std::runtime_error("busy upload fallback fixture registration setup");
        auto upload=Exl3RegisteredKVUpload::create_startup(authority);
        auto destination=std::make_shared<std::vector<std::uint16_t>>(64*1024);
        const auto event=reinterpret_cast<cudaEvent_t>(std::uintptr_t{99});
        const Exl3KVUploadProvider io{IO::copy,IO::record,IO::wait};
        const auto generation=upload->submit(first,destination->data(),64,nullptr,event,destination,active,7,1,io);
        const auto copies=IO::copies,records=IO::records,waits=IO::waits;
        bool refused=false;
        try{upload->submit(second,destination->data(),64,nullptr,event,destination,retired,7,2,io);}
        catch(const std::logic_error&){refused=true;}
        if(!refused || !upload->uncertain() || IO::copies!=copies || IO::records!=records || IO::waits!=waits)
            throw std::runtime_error("busy upload returned staged fallback over in-flight destination");
        upload->complete(generation);
        for(bool unowned_alias:{false,true}) {
            const auto bad_owner=unowned_alias?
                std::shared_ptr<const void>(std::shared_ptr<const void>{},destination.get()):std::shared_ptr<const void>{};
            refused=false;
            try{(void)upload->submit(second,destination->data(),64,nullptr,event,bad_owner,retired,7,2,io);}
            catch(const std::invalid_argument&){refused=true;}
            if(!refused || upload->uncertain() || IO::copies!=copies || IO::records!=records || IO::waits!=waits+1)
                throw std::runtime_error("retired registration masked invalid destination ownership or invoked IO");
        }
        if(upload->submit(second,destination->data(),64,nullptr,event,destination,retired,7,2,io)!=0)
            throw std::runtime_error("idle upload lost retirement fallback");
    }
    {
        auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;page->k[0].resize(64*1024);
        std::shared_ptr<const Exl3ExactKVPage> unowned(std::shared_ptr<const Exl3ExactKVPage>{},page.get());
        bool refused=false;
        try{Exl3ExactKVExtent::view(unowned,0,Exl3ExactKVExtent::Plane::key,64);}
        catch(const std::invalid_argument&){refused=true;}
        if(!refused)throw std::runtime_error("extent accepted page address without retention authority");
        auto original=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64);
        auto moved=std::move(original);
        if(original.backing_current() || original.registration_eligible() || !moved.registration_eligible())
            throw std::runtime_error("moved extent kept authority or invalidated destination");
        refused=false;try{original.data();}catch(const std::invalid_argument&){refused=true;}
        if(!refused)throw std::runtime_error("moved extent exposed stale data pointer");
        std::weak_ptr<const Exl3ExactKVPage> retained=page;page.reset();
        if(retained.expired() || !moved.backing_current())throw std::runtime_error("moved extent failed to retain page backing");
    }
    {
        using Upload=Exl3RegisteredKVUpload;
        const auto blocks=bounded_shared_live_blocks_for_test<Upload>();
        {
            PreparedAttentionStageBudget budget;budget.metadata_limit=Upload::metadata_bytes()-1;
            bool refused=false;
            try{(void)Upload::create_startup(budget);}catch(const Exl3ResourceReservationExhausted&){refused=true;}
            if(!refused || bounded_shared_live_blocks_for_test<Upload>()!=blocks)
                throw std::runtime_error("single upload slot bypassed preconstruction metadata admission");
            ++budget.metadata_limit;budget.fail_commit=true;refused=false;
            try{(void)Upload::create_startup(budget);}catch(const std::runtime_error& error){
                refused=std::string_view(error.what())=="injected post-factory inventory commit failure";
            }
            if(!refused || budget.retained.totals()!=Exl3ResourceInventory::Totals{} ||
                bounded_shared_live_blocks_for_test<Upload>()!=blocks)
                throw std::runtime_error("single upload slot commit failure retained factory result");
            budget.fail_commit=false;
            auto retry=Upload::create_startup(budget);
            if(!retry || bounded_shared_live_blocks_for_test<Upload>()!=blocks+1)
                throw std::runtime_error("single upload slot could not retry after commit refusal");
        }
        for(unsigned lanes:{1U,2U}) {
            PreparedAttentionStageBudget budget;
            const auto required=Upload::lane_requirement(lanes);
            budget.metadata_limit=required.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)]-1;
            bool refused=false;
            try{Upload::create_startup_lanes(budget,lanes);}catch(const Exl3ResourceReservationExhausted&){refused=true;}
            if(!refused || bounded_shared_live_blocks_for_test<Upload>()!=blocks)
                throw std::runtime_error("upload pool allocated before aggregate admission");
            ++budget.metadata_limit;
            for(unsigned fault=1;fault<=2*lanes;++fault) {
                refused=false;
                try{Upload::create_startup_lanes(budget,lanes,fault);}
                catch(const std::runtime_error& error){refused=std::string_view(error.what())=="injected KV upload slot construction failure";}
                if(!refused || budget.retained.totals()!=Exl3ResourceInventory::Totals{} ||
                    bounded_shared_live_blocks_for_test<Upload>()!=blocks)
                    throw std::runtime_error("upload slot failure retained partial pool");
            }
            budget.fail_commit=true;refused=false;
            try{Upload::create_startup_lanes(budget,lanes);}
            catch(const std::runtime_error& error){refused=std::string_view(error.what())=="injected post-factory inventory commit failure";}
            if(!refused || bounded_shared_live_blocks_for_test<Upload>()!=blocks)
                throw std::runtime_error("upload pool commit refusal retained factory result");
            budget.fail_commit=false;
            auto pool=Upload::create_startup_lanes(budget,lanes);
            if(budget.retained.totals()!=required.units || bounded_shared_live_blocks_for_test<Upload>()!=blocks+2*lanes)
                throw std::runtime_error("upload pool exact inventory mismatch");
            for(unsigned lane=0;lane<2;++lane)
              for(unsigned slot=0;slot<Upload::maximum_slots_per_lane;++slot)
                if(bool(pool[lane][slot])!=(lane<lanes && slot<2))
                    throw std::runtime_error("upload pool physical slot shape");
        }
        if(bounded_shared_live_blocks_for_test<Upload>()!=blocks)throw std::runtime_error("upload pool clean retirement leak");
    }
    {
        using Readers=Exl3DevicePageReaders<64>;
        const auto copies=bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>();
        const auto attention=bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>();
        for(unsigned lanes:{1U,2U})for(bool direct:{false,true}) {
            PreparedAttentionStageBudget budget;
            const auto required=Readers::requirement(lanes,direct);
            budget.metadata_limit=required.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)]-1;
            bool refused=false;
            try{Readers::create_startup(budget,lanes,direct);}catch(const Exl3ResourceReservationExhausted&){refused=true;}
            if(!refused || bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>()!=copies ||
                bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>()!=attention)
                throw std::runtime_error("aggregate page reader refusal constructed partial pool");
            ++budget.metadata_limit;
            const auto count=lanes*(1+(direct?64:0));
            for(unsigned fault:{1U,count}) {
                refused=false;
                try{Readers::create_startup(budget,lanes,direct,fault);}
                catch(const std::runtime_error& error){refused=std::string_view(error.what())=="injected page reader construction failure";}
                if(!refused || budget.retained.totals()!=Exl3ResourceInventory::Totals{} ||
                    bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>()!=copies ||
                    bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>()!=attention)
                    throw std::runtime_error("page reader factory failure leaked partial pool");
            }
            budget.fail_commit=true;refused=false;
            try{Readers::create_startup(budget,lanes,direct);}
            catch(const std::runtime_error& error){refused=std::string_view(error.what())=="injected post-factory inventory commit failure";}
            if(!refused || bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>()!=copies ||
                bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>()!=attention)
                throw std::runtime_error("page reader post-factory rollback retained result owners");
            budget.fail_commit=false;
            auto result=Readers::create_startup(budget,lanes,direct);
            if(budget.retained.totals()!=required.units ||
                bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>()!=copies+lanes ||
                bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>()!=attention+(direct?64*lanes:0))
                throw std::runtime_error("page reader pool differs from complete reservation");
            for(unsigned lane=0;lane<2;++lane) {
                if(bool(result.copies[lane])!=(lane<lanes))throw std::runtime_error("page reader physical lane mismatch");
                for(const auto& reader:result.attention[lane])if(bool(reader)!=(direct && lane<lanes))
                    throw std::runtime_error("page reader direct-attention capacity mismatch");
            }
        }
        if(bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>()!=copies ||
            bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>()!=attention)
            throw std::runtime_error("page reader aggregate clean retirement leak");
    }
    if(const auto* selected=std::getenv("NINFER_TEST_DEVICE_PAGE_COPY_FAILURE");selected && *selected) {
        using Copy=Exl3DevicePageCopy;using IO=PreparedKVUploadProvider;
        using Fill=Exl3DevicePageFill;
        const std::string_view mode(selected);
        if(mode!="copy" && mode!="record" && mode!="wait")throw std::invalid_argument("page copy failure stage");
        const auto before=Exl3KVTransferLease::quarantined_records();
        RetainedDescriptorLedger metadata;
        std::weak_ptr<Copy> weak_copy;
        std::weak_ptr<const void> weak_page,weak_destination,weak_storage;
        {
            auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;weak_page=page;
            for(int bank=0;bank<16;++bank){page->k[bank].resize(64*1024);page->v[bank].resize(64*1024);}
            Exl3DevicePageKey key(std::make_shared<int>(9),page,{0,false},64);
            auto storage=std::make_shared<std::vector<std::uint16_t>>(Fill::elements);weak_storage=storage;
            Fill fill(key,storage,*storage);
            const auto generation=fill.begin(7,8,101,page);
            for(int bank=0;bank<16;++bank)for(bool key_plane:{false,true})fill.plane_submitted(generation,bank,key_plane);
            if(!fill.finish(generation,0))throw std::runtime_error("page copy fixture readiness");
            PreparedKVRegistrationAuthority authority;
            auto copy=Copy::create_startup(authority);weak_copy=copy;
            if(Copy::attach_retirement_credit(copy,metadata.acquire(Copy::metadata_bytes()-1)) ||
                !Copy::attach_retirement_credit(copy,metadata.acquire(Copy::metadata_bytes())) ||
                Copy::attach_retirement_credit(copy,metadata.acquire(Copy::metadata_bytes())))
                throw std::runtime_error("page copy retirement exact/duplicate extent");
            auto destination=std::make_shared<std::vector<std::uint16_t>>(64*1024);weak_destination=destination;
            const auto extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64);
            IO::expected_kind=cudaMemcpyDeviceToDevice;
            IO::copies=IO::records=IO::waits=IO::fail_copy_at=0;
            IO::copy_error=mode=="copy"?cudaErrorUnknown:cudaSuccess;
            IO::record_error=mode=="record"?cudaErrorUnknown:cudaSuccess;
            IO::wait_error=mode=="wait"?cudaErrorUnknown:cudaSuccess;
            const auto submit=[&] {copy->copy(fill.ready_handle(),extent,destination->data(),64,nullptr,
                reinterpret_cast<cudaEvent_t>(std::uintptr_t{102}),destination,7,9,{IO::copy,IO::record,IO::wait});};
            bool failed=false;try{submit();}catch(const std::runtime_error& error) {
                failed=std::string_view(error.what())=="shared device page copy final use uncertain";
            }
            if(!failed || !copy->uncertain() || fill.retained_readers()!=1 || IO::copies!=1 ||
                IO::records!=(mode=="copy"?0U:1U) || IO::waits!=(mode=="wait"?1U:0U))
                throw std::runtime_error("page copy failure continued submission or released reader");
            const auto failure=copy->failure();
            const auto expected=mode=="copy"?Copy::Stage::copy:(mode=="record"?Copy::Stage::record:Copy::Stage::wait);
            if(failure.stage!=expected || failure.error!=static_cast<int>(cudaErrorUnknown))
                throw std::runtime_error("page copy lost first failure stage/status");
            IO::copy_error=IO::record_error=IO::wait_error=cudaSuccess;
            bool refused=false;try{submit();}catch(const std::logic_error&){refused=true;}
            if(!refused || IO::copies!=1 || fill.retained_readers()!=1 ||
                copy->failure().stage!=failure.stage || copy->failure().error!=failure.error)
                throw std::runtime_error("page copy failure permitted uncertain reader reuse");
        }
        if(!weak_copy.expired() || weak_page.expired() || weak_storage.expired() || weak_destination.expired() ||
            Exl3KVTransferLease::quarantined_records()!=before+1 || metadata.bytes()!=Copy::metadata_bytes())
            throw std::runtime_error("page copy quarantine lost physical owners or metadata");
        weak_copy.reset();
        if(metadata.bytes()!=Exl3KVTransferLease::state_metadata_bytes())
            throw std::runtime_error("page copy final weak release discarded quarantined State charge");
        return; // Intentional quarantine; each stage needs a separate authorized process.
    }
    if(const auto* selected=std::getenv("NINFER_TEST_ATTENTION_STORAGE_CONSTRUCTOR_FAILURE");selected && *selected) {
        using Owner=Exl3AttentionStageStorage;using Events=PreparedAttentionStageEvents;
        using Storage=PreparedDevicePageStorageProvider;
        const std::string_view mode(selected);
        if(mode!="event" && mode!="release")throw std::invalid_argument("attention constructor cleanup mode");
        const auto before=Owner::quarantined_records(),blocks=bounded_shared_live_blocks_for_test<Owner>();
        const auto frees=Storage::releases;
        Events::creates=Events::destroys=0;Events::fail_at=6;
        Events::destroy_error=mode=="event"?cudaErrorUnknown:cudaSuccess;
        Storage::release_error=mode=="release"?cudaErrorUnknown:cudaSuccess;
        PreparedKVRegistrationAuthority authority;
        bool refused=false;
        try {exl3_create_attention_stage_lanes(authority,2,64,
            {Storage::allocate,Storage::release,Events::create,Events::destroy});}
        catch(const std::runtime_error& error) {
            refused=std::string_view(error.what())=="attention stage event allocation failed";
        }
        if(!refused || !authority.sealed || Owner::quarantined_records()!=before+2 ||
            authority.retained.totals()!=Exl3ResourceInventory::Totals{} ||
            authority.constructor_device.bytes()!=2*64*4096 ||
            authority.constructor_metadata.bytes()!=2*Owner::retirement_metadata_bytes() ||
            bounded_shared_live_blocks_for_test<Owner>()!=blocks ||
            Events::destroys!=(mode=="event"?2U:5U) || Storage::releases!=frees+(mode=="event"?0U:2U))
            throw std::runtime_error("C2 failed construction lost earlier lane retirement credits");
        return; // Intentional quarantine, one mode per separately authorized process.
    }
    if(const auto* selected=std::getenv("NINFER_TEST_DEVICE_PAGE_CONSTRUCTOR_FAILURE");selected && *selected) {
        using Storage=Exl3DevicePageStorage;using Provider=PreparedDevicePageStorageProvider;
        const std::string_view mode(selected);
        if(mode!="retain" && mode!="release")throw std::invalid_argument("page constructor failure mode retain/release");
        const bool retain=mode=="retain";
        unsigned fault=1;
        if(const auto* stage=std::getenv("NINFER_TEST_DEVICE_PAGE_CONSTRUCTOR_STAGE")) {
            const std::string_view value(stage);
            if(value.size()!=1 || value[0]<'1' || value[0]>'4')throw std::invalid_argument("page constructor stage1..4");
            fault=value[0]-'0';
        }
        const auto live_blocks=[] {
            return std::array<std::size_t,4>{bounded_shared_live_blocks_for_test<Storage>(),
                bounded_shared_live_blocks_for_test<Exl3DevicePageFill>(),
                bounded_shared_live_blocks_for_test<Exl3DevicePageFill::View>(),
                bounded_shared_live_blocks_for_test<std::atomic<std::uint64_t>>()};
        };
        const auto blocks_before=live_blocks();
        auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;
        for(int bank=0;bank<16;++bank){page->k[bank].resize(64*1024);page->v[bank].resize(64*1024);}
        Exl3DevicePageKey key(std::make_shared<int>(101),page,{0,false},64);
        PreparedKVRegistrationAuthority authority;
        const auto before=Storage::quarantined_bytes();
        const auto allocations=Provider::allocations,releases=Provider::releases;
        Provider::release_error=retain?cudaErrorUnknown:cudaSuccess;
        bool original=false;
        const std::array<std::string_view,4> errors{"injected device page post-allocation failure",
            "injected page fill after-state failure","injected page view after-counter failure",
            "injected page fill after-view failure"};
        try{(void)Storage::create_runtime(authority,7,key,{Provider::allocate,Provider::release},fault);}
        catch(const std::runtime_error& error){original=std::string_view(error.what())==errors[fault-1];}
        if(!original || Provider::allocations!=allocations+1 || Provider::releases!=releases+1 ||
            authority.retained.totals()!=Exl3ResourceInventory::Totals{} || live_blocks()!=blocks_before)
            throw std::runtime_error("page constructor failure did not unwind original allocation boundary");
        if(retain) {
            const auto credits=Storage::retained_constructor_credits_for_test();
            if(!authority.sealed || Storage::quarantined_bytes()!=before+Storage::bytes ||
                credits[0]!=Storage::bytes || credits[1]!=Storage::retirement_metadata_bytes() ||
                authority.constructor_device.bytes()!=credits[0] || authority.constructor_metadata.bytes()!=credits[1])
                throw std::runtime_error("page constructor cleanup lost exact survivor credits or authority seal");
        } else {
            if(authority.sealed || Storage::quarantined_bytes()!=before || authority.constructor_device.bytes() ||
                authority.constructor_metadata.bytes())throw std::runtime_error("clean page constructor unwind retained credits");
            auto retry=Storage::create_runtime(authority,7,key,{Provider::allocate,Provider::release});
            if(authority.constructor_device.bytes() || authority.constructor_metadata.bytes())
                throw std::runtime_error("page constructor commit duplicated provisional credits");
            retry={};authority.retained={};
            if(Provider::releases!=releases+2)throw std::runtime_error("page constructor retry did not retire exactly once");
        }
        return; // Retain mode intentionally leaves the failed provider owner quarantined.
    }
    const auto bounded_startup=[]<class Owner>() {
        const auto blocks=bounded_shared_live_blocks_for_test<Owner>();
        PreparedAttentionStageBudget budget;budget.metadata_limit=Owner::metadata_bytes()-1;
        bool refused=false;
        try{Owner::create_startup(budget);}catch(const Exl3ResourceReservationExhausted&){refused=true;}
        if(!refused || bounded_shared_live_blocks_for_test<Owner>()!=blocks)
            throw std::runtime_error("registered upload metadata allocation preceded full precredit");
        ++budget.metadata_limit;
        struct RollbackObservation {
            static void observe(std::size_t expected) {
                if(bounded_shared_live_blocks_for_test<Owner>()!=expected)
                    throw std::runtime_error("bounded startup owner survived authority rollback");
            }
        };
        budget.fail_commit=true;budget.after_rollback=&RollbackObservation::observe;
        budget.rollback_expected_blocks=blocks;
        bool original=false;
        try{(void)Owner::create_startup(budget);}
        catch(const std::runtime_error& error) {
            original=std::string_view(error.what())=="injected post-factory inventory commit failure";
        }
        if(!original || bounded_shared_live_blocks_for_test<Owner>()!=blocks ||
            budget.retained.totals()!=Exl3ResourceInventory::Totals{})
            throw std::runtime_error("bounded startup commit failure retained ownership or replaced original error");
        budget.fail_commit=false;budget.after_rollback=nullptr;
        auto owner=Owner::create_startup(budget);
        std::weak_ptr<Owner> weak=owner;
        auto allocation=budget.retained.first_lifetime_retirement();
        RetainedDescriptorLedger retired;
        if(!allocation || allocation->units!=Owner::metadata_bytes() ||
           !allocation->lifetime_retire(allocation->owner,retired.acquire(allocation->units)))
            throw std::runtime_error("registered upload bounded ownership has no complete lifetime charge");
        if(retired.bytes()!=Owner::metadata_bytes())
            throw std::runtime_error("registered upload split retirement changed total credit");
        allocation.reset();budget.retained={};owner.reset();
        if(!weak.expired() || retired.bytes()!=bounded_shared_allocation_bytes<Owner>() ||
            bounded_shared_live_blocks_for_test<Owner>()!=blocks+1)
            throw std::runtime_error("registered upload released weak control storage credit early");
        weak.reset();
        if(retired.bytes() || bounded_shared_live_blocks_for_test<Owner>()!=blocks)
            throw std::runtime_error("registered upload final weak retirement leaked metadata credit");
    };
    bounded_startup.template operator()<Exl3KVRegistrationCache>();
    bounded_startup.template operator()<Exl3RegisteredKVUpload>();
    bounded_startup.template operator()<Exl3DevicePageCache>();
    bounded_startup.template operator()<Exl3DevicePageAttention>();
    bounded_startup.template operator()<Exl3AttentionStage>();
    bounded_startup.template operator()<Exl3DevicePageCopy>();
    {
        using Cache=Exl3DevicePageCache;
        const auto blocks=bounded_shared_live_blocks_for_test<Cache>();
        PreparedAttentionStageBudget budget;budget.metadata_limit=Cache::metadata_bytes();
        bool original=false;
        try{(void)Cache::create_startup(budget,1);}
        catch(const std::runtime_error& error) {
            original=std::string_view(error.what())=="injected page cache container precommit failure";
        }
        if(!original || bounded_shared_live_blocks_for_test<Cache>()!=blocks ||
            budget.retained.totals()!=Exl3ResourceInventory::Totals{})
            throw std::runtime_error("page cache failed startup published or retained container allocation");
        struct CommitRollbackObservation {
            static void observe(std::size_t expected) {
                if(bounded_shared_live_blocks_for_test<Cache>()!=expected)
                    throw std::runtime_error("page cache result survived authority rollback");
            }
        };
        budget.fail_commit=true;budget.after_rollback=&CommitRollbackObservation::observe;
        budget.rollback_expected_blocks=blocks;
        original=false;
        try{(void)Cache::create_startup(budget);}
        catch(const std::runtime_error& error) {
            original=std::string_view(error.what())=="injected post-factory inventory commit failure";
        }
        if(!original || bounded_shared_live_blocks_for_test<Cache>()!=blocks ||
            budget.retained.totals()!=Exl3ResourceInventory::Totals{})
            throw std::runtime_error("page cache failed commit retained allocation or replaced original error");
        budget.fail_commit=false;budget.after_rollback=nullptr;
        auto cache=Cache::create_startup(budget);
        RetainedDescriptorLedger metadata;
        if(Cache::attach_retirement_credit(cache,metadata.acquire(Cache::metadata_bytes()-1)) || metadata.bytes())
            throw std::runtime_error("page cache accepted short container credit");
        if(!Cache::attach_retirement_credit(cache,metadata.acquire(Cache::metadata_bytes())) ||
            Cache::attach_retirement_credit(cache,metadata.acquire(Cache::metadata_bytes())) ||
            metadata.bytes()!=Cache::metadata_bytes())
            throw std::runtime_error("page cache exact or duplicate container credit mismatch");
        std::weak_ptr<Cache> weak=cache;budget.retained={};cache.reset();
        if(!weak.expired() || metadata.bytes()!=Cache::metadata_bytes())
            throw std::runtime_error("page cache weak owner lost retained bounded block charge");
        weak.reset();
        if(metadata.bytes() || bounded_shared_live_blocks_for_test<Cache>()!=blocks)
            throw std::runtime_error("page cache retry leaked bounded container metadata");
    }
    for(unsigned scenario:{0U,1U,2U,3U,4U}) {
        using Registration=Exl3RegisteredKVBacking;using Provider=PreparedKVRegistrationProvider;
        PreparedKVRegistrationAuthority authority;
        const bool retirement_enabled=scenario!=0;
        auto cache=Exl3KVRegistrationCache::create_startup(authority,retirement_enabled);
        const std::size_t count=scenario>=2?64:1;
        authority.registered_limit=count*64ULL*1024*2;
        const Exl3KVRegistrationProvider provider{Provider::acquire,Provider::release};
        std::vector<std::optional<Registration::Reader>> readers;
        std::shared_ptr<Registration> oldest;
        for(std::size_t index=0;index<count;++index) {
            auto page=std::make_shared<Exl3ExactKVPage>();page->first=static_cast<int>(index*64);page->rows=64;
            page->k[0].resize(64*1024);
            const auto extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,page->first+64);
            auto owner=cache->acquire_startup(authority,extent,provider);
            if(!owner)throw std::runtime_error("cache turnover fixture did not populate registration capacity");
            if(index==0)oldest=owner;
            readers.push_back(Registration::acquire_reader(owner,extent));
            if(!readers.back())throw std::runtime_error("cache turnover fixture lost initial reader");
        }
        auto page=std::make_shared<Exl3ExactKVPage>();page->first=4096;page->rows=64;page->k[0].resize(64*1024);
        const auto next=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,4160);
        const auto releases=Provider::releases,acquisitions=Provider::acquisitions;
        if(cache->acquire_runtime(authority,7,next,provider) || Provider::releases!=releases ||
            Provider::acquisitions!=acquisitions)
            throw std::runtime_error("registration pressure evicted active readers instead of staging");
        readers[0].reset();
        {
            std::shared_ptr<Exl3KVRegistrationCache> unowned(std::shared_ptr<Exl3KVRegistrationCache>{},cache.get());
            if(Exl3KVRegistrationCache::acquire_external_read(unowned) ||
                Exl3KVRegistrationCache::acquire_external_read_pair(unowned) || cache->external_readers())
                throw std::runtime_error("nonowning registration cache alias acquired external guard");
        }
        auto external=Exl3KVRegistrationCache::acquire_external_read(cache);
        if(!external || cache->external_readers()!=1 ||
            cache->acquire_runtime(authority,7,next,provider) || Provider::releases!=releases ||
            Provider::acquisitions!=acquisitions)
            throw std::runtime_error("external page fill allowed registration eviction");
        const auto retained_before_stale=authority.retained.totals();
        const auto retired_before_stale=cache->retired_registrations();
        authority.expire_after_checks=authority.lease_checks+1;
        bool stale_pressure=false;
        try{(void)cache->acquire_runtime(authority,7,next,provider);}
        catch(const std::invalid_argument& error){stale_pressure=std::string_view(error.what())==
            "registration cache runtime lease changed during acquisition";}
        authority.expire_after_checks=0;
        if(!stale_pressure || cache->external_readers()!=1 || Provider::releases!=releases ||
            Provider::acquisitions!=acquisitions || authority.retained.totals()!=retained_before_stale ||
            cache->retired_registrations()!=retired_before_stale || oldest->retired_successfully())
            throw std::runtime_error("stale pressure fallback escaped or retired protected backing");
        if(cache->acquire_runtime(authority,7,next,provider) || cache->external_readers()!=1)
            throw std::runtime_error("stale pressure refusal consumed surviving external read guard");
        external.reset();
        if(cache->external_readers()!=0)
            throw std::runtime_error("external page fill retained completed read guard");
        cache->refuse_read_pairs_for_test(true);
        if(Exl3KVRegistrationCache::acquire_external_read_pair(cache) || cache->external_readers())
            throw std::runtime_error("refused history pair acquired partial ownership");
        auto independent=Exl3KVRegistrationCache::acquire_external_read(cache);
        if(!independent || cache->external_readers()!=1)
            throw std::runtime_error("history pair refusal disabled independent page reader");
        independent.reset();cache->refuse_read_pairs_for_test(false);
        auto pair=Exl3KVRegistrationCache::acquire_external_read_pair(cache);
        if(!pair || cache->external_readers()!=2 || cache->external_reader_high_water()!=2)
            throw std::runtime_error("history pair did not reserve both readers atomically");
        std::optional<Exl3KVRegistrationCache::ExternalRead> first_read(std::move((*pair)[0]));
        std::optional<Exl3KVRegistrationCache::ExternalRead> second_read(std::move((*pair)[1]));
        pair.reset();first_read.reset();
        if(cache->external_readers()!=1 || cache->acquire_runtime(authority,7,next,provider) ||
            Provider::releases!=releases || Provider::acquisitions!=acquisitions)
            throw std::runtime_error("first plane completion released second plane eviction guard");
        second_read.reset();
        if(cache->external_readers()!=0 || Exl3KVRegistrationCache::acquire_external_read_pair({}))
            throw std::runtime_error("history pair release or empty-cache refusal lost ownership");
        for(unsigned missing=0;missing<2;++missing) {
            auto malformed=provider;
            if(missing==0)malformed.acquire=nullptr;else malformed.release=nullptr;
            const auto retired_before=cache->retired_registrations();
            bool refused=false;
            try{(void)cache->acquire_runtime(authority,7,next,malformed);}
            catch(const std::invalid_argument& error){refused=std::string_view(error.what())=="KV registration provider incomplete";}
            if(!refused || Provider::releases!=releases || Provider::acquisitions!=acquisitions ||
                cache->retired_registrations()!=retired_before || oldest->retired_successfully())
                throw std::runtime_error("incomplete provider evicted usable registration before refusal");
        }
        auto incomplete_page=std::make_shared<Exl3ExactKVPage>();
        incomplete_page->first=8192;incomplete_page->rows=64;incomplete_page->k[0].resize(1024);
        const auto incomplete=Exl3ExactKVExtent::view(incomplete_page,0,Exl3ExactKVExtent::Plane::key,8193);
        if(cache->acquire_runtime(authority,7,incomplete,provider) ||
            Provider::releases!=releases || Provider::acquisitions!=acquisitions || oldest->retired_successfully())
            throw std::runtime_error("incomplete plane evicted idle registration before fallback");
        if(!retirement_enabled) {
            if(cache->acquire_runtime(authority,7,next,provider) || oldest->retired_successfully() ||
                Provider::releases!=releases || Provider::acquisitions!=acquisitions || authority.retired_metadata.bytes())
                throw std::runtime_error("untracked host reader configuration enabled registration turnover");
            continue;
        }
        std::weak_ptr<Registration> weak_oldest=oldest;
        unsigned failed_acquisitions=0;
        if(scenario>=3) {
            const auto retired_before=cache->retired_registrations();
            const auto requests_before=authority.runtime_requests;
            const auto blocks_before=bounded_shared_live_blocks_for_test<Registration>();
            Provider::acquire_error=scenario==3?cudaErrorMemoryAllocation:cudaSuccess;
            authority.exhaust=scenario==4;
            std::shared_ptr<Registration> failed;
            try{failed=cache->acquire_runtime(authority,7,next,provider);}
            catch(...){Provider::acquire_error=cudaSuccess;authority.exhaust=false;throw;}
            Provider::acquire_error=cudaSuccess;authority.exhaust=false;
            failed_acquisitions=scenario==3?1:0;
            const auto domain=static_cast<unsigned>(Exl3ResourceInventory::Domain::cuda_registered_host);
            if(failed || Provider::releases!=releases+1 || Provider::acquisitions!=acquisitions+failed_acquisitions ||
                authority.runtime_requests!=requests_before+1 ||
                bounded_shared_live_blocks_for_test<Registration>()!=blocks_before ||
                authority.constructor_registration.bytes() || authority.constructor_metadata.bytes() ||
                cache->retired_registrations()!=retired_before+1 || !oldest->retired_successfully() ||
                authority.retained.totals()[domain]!=authority.registered_limit-next.backing().bytes ||
                authority.retired_metadata.bytes()!=Registration::metadata_bytes())
                throw std::runtime_error("failed registration replacement lost bounded eviction or survivor accounting");
            for(std::size_t index=1;index<readers.size();++index)
                if(readers[index]->owner()->active_readers()!=1 || readers[index]->owner()->seal_idle())
                    throw std::runtime_error("failed replacement disturbed active registration peer");
        }
        auto replacement=cache->acquire_runtime(authority,7,next,provider);
        const auto registered_domain=static_cast<unsigned>(Exl3ResourceInventory::Domain::cuda_registered_host);
        if(!replacement || replacement==oldest || !oldest->retired_successfully() ||
            Provider::releases!=releases+1 || Provider::acquisitions!=acquisitions+1+failed_acquisitions ||
            authority.retained.totals()[registered_domain]!=authority.registered_limit ||
            authority.retired_metadata.bytes()!=Registration::metadata_bytes())
            throw std::runtime_error("idle cache turnover lost registration or surviving metadata credit");
        for(std::size_t index=1;index<readers.size();++index)
            if(readers[index]->owner()->active_readers()!=1 || readers[index]->owner()->seal_idle())
                throw std::runtime_error("cache turnover invalidated another in-flight registration");
        oldest.reset();
        if(!weak_oldest.expired() || authority.retired_metadata.bytes()!=bounded_shared_allocation_bytes<Registration>())
            throw std::runtime_error("cache turnover released old weak control credit early");
        weak_oldest.reset();
        if(authority.retired_metadata.bytes())throw std::runtime_error("cache turnover leaked retired metadata");
    }
    {
        using Registration=Exl3RegisteredKVBacking;using IO=PreparedKVUploadProvider;
        auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;page->k[0].resize(64*1024,0x3155);
        const auto extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64);
        PreparedKVRegistrationAuthority authority;
        auto owner=Registration::acquire_startup(authority,extent,
            {PreparedKVRegistrationProvider::acquire,PreparedKVRegistrationProvider::release});
        auto first=Exl3RegisteredKVUpload::create_startup(authority);
        auto second=Exl3RegisteredKVUpload::create_startup(authority);
        auto a=std::make_shared<std::vector<std::uint16_t>>(64*1024);
        auto b=std::make_shared<std::vector<std::uint16_t>>(64*1024);
        const Exl3KVUploadProvider provider{IO::copy,IO::record,IO::wait};
        const auto releases=PreparedKVRegistrationProvider::releases;
        auto first_generation=first->submit(extent,a->data(),64,nullptr,
            reinterpret_cast<cudaEvent_t>(std::uintptr_t{31}),a,owner,7,1,provider);
        auto second_generation=second->submit(extent,b->data(),64,nullptr,
            reinterpret_cast<cudaEvent_t>(std::uintptr_t{32}),b,owner,8,1,provider);
        if(owner->active_readers()!=2 || owner->seal_idle() || owner->retire_sealed())
            throw std::runtime_error("registration retirement bypassed two in-flight readers");
        first->complete(first_generation);
        if(owner->active_readers()!=1 || owner->seal_idle())
            throw std::runtime_error("first upload completion released peer registration pin");
        second->complete(second_generation);
        if(*a!=page->k[0] || *b!=page->k[0] || owner->active_readers()!=0 || !owner->seal_idle())
            throw std::runtime_error("last completed reader did not permit idle registration seal");
        const auto copies=IO::copies,records=IO::records,waits=IO::waits;
        if(first->submit(extent,a->data(),64,nullptr,reinterpret_cast<cudaEvent_t>(std::uintptr_t{31}),
                a,owner,7,2,provider)!=0 || first->uncertain() || IO::copies!=copies || IO::records!=records || IO::waits!=waits)
            throw std::runtime_error("sealed registration did not request untouched staged fallback");
        if(!owner->retire_sealed() || !owner->retire_sealed() || !owner->retired_successfully() ||
            owner->covers(extent) || Registration::acquire_reader(owner,extent) ||
            PreparedKVRegistrationProvider::releases!=releases+1)
            throw std::runtime_error("sealed registration retirement repeated release or admitted another reader");
    }
    {
        using IO=PreparedKVUploadProvider;using Provider=PreparedKVRegistrationProvider;
        PreparedKVRegistrationAuthority authority;
        auto cache=Exl3KVRegistrationCache::create_startup(authority);
        const Exl3KVRegistrationProvider registration_io{Provider::acquire,Provider::release};
        const Exl3KVUploadProvider copy_io{IO::copy,IO::record,IO::wait};
        auto page=std::make_shared<Exl3ExactKVPage>();page->first=128;page->rows=48;
        page->k[0].resize(48*1024,0x3141);
        const auto tail=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,176,160);
        const auto acquisitions=Provider::acquisitions;
        if(cache->acquire_runtime(authority,7,tail,registration_io) || Provider::acquisitions!=acquisitions)
            throw std::runtime_error("private partial suffix bypassed staged fallback");
        page->rows=64;page->k[0].resize(64*1024,0x3252);
        const auto first_extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,176,160);
        const auto second_extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,192,176);
        auto registration=cache->acquire_runtime(authority,7,first_extent,registration_io);
        if(tail.backing_current() || !registration ||
            cache->acquire_runtime(authority,7,second_extent,registration_io)!=registration ||
            Provider::acquisitions!=acquisitions+1)
            throw std::runtime_error("completed suffix revived stale tail or duplicated backing registration");
        auto first=Exl3RegisteredKVUpload::create_startup(authority);
        auto second=Exl3RegisteredKVUpload::create_startup(authority);
        auto a=std::make_shared<std::vector<std::uint16_t>>(200*1024,0x7777);
        auto b=std::make_shared<std::vector<std::uint16_t>>(200*1024,0x6666);
        const auto event_a=reinterpret_cast<cudaEvent_t>(std::uintptr_t{71});
        const auto event_b=reinterpret_cast<cudaEvent_t>(std::uintptr_t{72});
        const auto ga=first->submit(first_extent,a->data(),200,nullptr,event_a,a,registration,7,1,copy_io);
        const auto gb=second->submit(second_extent,b->data(),200,nullptr,event_b,b,registration,8,1,copy_io);
        auto fork=std::make_shared<Exl3ExactKVPage>(*page);
        std::fill(fork->k[0].begin()+32*1024,fork->k[0].begin()+48*1024,0x4353);
        const auto fork_extent=Exl3ExactKVExtent::view(fork,0,Exl3ExactKVExtent::Plane::key,176,160);
        auto fork_registration=cache->acquire_runtime(authority,7,fork_extent,registration_io);
        if(!fork_registration || fork_registration==registration || registration->covers(fork_extent) ||
            Provider::acquisitions!=acquisitions+2 || registration->active_readers()!=2)
            throw std::runtime_error("COW suffix reused old registration or disturbed private readers");
        first->complete(ga);
        if(registration->active_readers()!=1 || registration->seal_idle())
            throw std::runtime_error("completed suffix released another lane's backing");
        for(std::size_t index=0;index<a->size();++index) {
            const auto row=index/1024;
            if((*a)[index]!=(row>=160 && row<176?0x3141:0x7777))
                throw std::runtime_error("published suffix read uncommitted rows or changed destination guards");
        }
        const auto fork_generation=first->submit(fork_extent,a->data(),200,nullptr,event_a,a,
            fork_registration,7,2,copy_io);
        second->complete(gb);first->complete(fork_generation);
        for(std::size_t index=0;index<a->size();++index) {
            const auto row=index/1024;
            if((*a)[index]!=(row>=160 && row<176?0x4353:0x7777) ||
                (*b)[index]!=(row>=176 && row<192?0x3252:0x6666))
                throw std::runtime_error("suffix/COW upload changed destination offset or untouched rows");
        }
        if(registration->active_readers() || fork_registration->active_readers() ||
            page->k[0][32*1024]!=0x3141)
            throw std::runtime_error("suffix/COW completion retained readers or changed shared source");
    }
    {
        using Owner=Exl3RegisteredKVBacking;using Provider=PreparedKVRegistrationProvider;
        auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;page->k[0].resize(64*1024);
        const auto extent=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64);
        PreparedAttentionStageBudget budget;budget.metadata_limit=Owner::metadata_bytes()-1;
        const auto acquisitions=Provider::acquisitions,releases=Provider::releases;
        const auto blocks=bounded_shared_live_blocks_for_test<Owner>();
        const Exl3KVRegistrationProvider provider{Provider::acquire,Provider::release};
        bool refused=false;
        try{Owner::acquire_startup(budget,extent,provider);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        if(!refused || Provider::acquisitions!=acquisitions || bounded_shared_live_blocks_for_test<Owner>()!=blocks)
            throw std::runtime_error("registration control storage bypassed metadata precredit");
        ++budget.metadata_limit;
        auto owner=Owner::acquire_startup(budget,extent,provider);
        std::weak_ptr<Owner> weak=owner;RetainedDescriptorLedger retired;
        auto allocation=budget.retained.first_lifetime_retirement();
        if(!owner || !allocation || allocation->units!=Owner::metadata_bytes() ||
            !allocation->lifetime_retire(allocation->owner,retired.acquire(allocation->units)))
            throw std::runtime_error("registration omitted bounded state lifetime credit");
        allocation.reset();budget.retained={};owner.reset();
        if(!weak.expired() || Provider::releases!=releases+1 ||
            retired.bytes()!=bounded_shared_allocation_bytes<Owner>())
            throw std::runtime_error("registration state and weak control retirement were conflated");
        weak.reset();
        if(retired.bytes() || bounded_shared_live_blocks_for_test<Owner>()!=blocks)
            throw std::runtime_error("registration final weak owner retained metadata charge");
    }
    {
        auto source=std::make_shared<Exl3ExactKVPage>();source->rows=64;source->k[0].resize(64*1024);
        const auto extent=Exl3ExactKVExtent::view(source,0,Exl3ExactKVExtent::Plane::key,64);
        PreparedKVRegistrationAuthority authority;
        authority.before_factory=[&]{source->k[0].clear();};
        const auto acquired=PreparedKVRegistrationProvider::acquisitions;
        bool refused=false;
        try{Exl3RegisteredKVBacking::acquire_startup(authority,extent,
            {PreparedKVRegistrationProvider::acquire,PreparedKVRegistrationProvider::release});}
        catch(const std::invalid_argument&){refused=true;}
        if(!refused || PreparedKVRegistrationProvider::acquisitions!=acquired ||
            authority.retained.totals()!=Exl3ResourceInventory::Totals{})
            throw std::runtime_error("registration submitted stale backing after reservation");
    }
    {
        using Storage=PreparedDevicePageStorageProvider;using Events=PreparedAttentionStageEvents;
        Events::creates=Events::destroys=Events::fail_at=0;
        PreparedAttentionStageBudget budget;budget.device_limit=64*4096;
        budget.metadata_limit=Exl3AttentionStageResources::metadata_bytes()-1;
        Exl3AttentionStageStorageProvider provider{Storage::allocate,Storage::release,Events::create,Events::destroy};
        const auto allocations=Storage::allocations;
        bool refused=false;
        try{Exl3AttentionStageResources::create_startup(budget,64,provider);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        if(!refused || Storage::allocations!=allocations || Events::creates)
            throw std::runtime_error("whole stage metadata budget was checked after device allocation");
        ++budget.metadata_limit;
        auto resources=Exl3AttentionStageResources::create_startup(budget,64,provider);
        if(!resources.storage || !resources.stages || !resources.history || Events::creates!=4)
            throw std::runtime_error("exact-fit stage transaction did not bind complete resources");
        PreparedAttentionStageBudget c2;c2.device_limit=2*64*4096;
        c2.metadata_limit=2*Exl3AttentionStageResources::metadata_bytes()-1;
        const auto before_c2=Storage::allocations;
        refused=false;
        try{exl3_create_attention_stage_lanes(c2,2,64,provider);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        if(!refused || Storage::allocations!=before_c2)
            throw std::runtime_error("C2 staging allocated first lane before complete admission");
        ++c2.metadata_limit;
        auto paired=exl3_create_attention_stage_lanes(c2,2,64,provider);
        if(!paired[0].storage || !paired[1].storage ||
            paired[0].storage->plane(0)==paired[1].storage->plane(0) || paired[0].stages==paired[1].stages)
            throw std::runtime_error("C2 staging did not retain private physical owners");
    }
    {
        using Events=PreparedAttentionStageEvents;
        using Storage=PreparedDevicePageStorageProvider;
        Exl3AttentionStageStorageProvider provider{Storage::allocate,Storage::release,Events::create,Events::destroy};
        if(const auto* mode=std::getenv("NINFER_TEST_C2_ATTENTION_GRANT_CLEANUP");mode && *mode) {
            const std::string_view selected(mode);
            if(selected!="event" && selected!="destination")throw std::invalid_argument("C2 grant cleanup selector");
            PreparedKVRegistrationAuthority transaction;
            transaction.malformed_constructor_credit=3;transaction.malformed_constructor_credit_on_call=2;
            Events::creates=Events::destroys=Events::fail_at=0;
            Events::destroy_error=selected=="event"?cudaErrorUnknown:cudaSuccess;
            Storage::release_error=selected=="destination"?cudaErrorUnknown:cudaSuccess;
            const auto allocations=Storage::allocations,releases=Storage::releases;
            const auto quarantined=Exl3AttentionStageStorage::quarantined_records();
            bool original=false;
            try{(void)exl3_create_attention_stage_lanes(transaction,2,64,provider);}
            catch(const std::invalid_argument& error){original=std::string_view(error.what())==
                "attention stage constructor credit extent";}
            const auto credits=Exl3AttentionStageStorage::retained_credits_for_test();
            if(!original || !transaction.sealed || transaction.constructor_credit_calls!=2 ||
                Exl3AttentionStageStorage::quarantined_records()!=quarantined+1 ||
                Storage::allocations!=allocations+1 || Events::creates!=4 ||
                Events::destroys!=(selected=="event"?1U:4U) ||
                Storage::releases!=releases+(selected=="event"?0U:1U) ||
                credits[0]!=64*4096 || credits[1]!=Exl3AttentionStageStorage::retirement_metadata_bytes() ||
                transaction.constructor_device.bytes()!=credits[0] || transaction.constructor_metadata.bytes()!=credits[1])
                throw std::runtime_error("C2 grant failure cleanup lost original error or quarantined credit ownership");
            bool refused=false;
            try{(void)exl3_create_attention_stage_lanes(transaction,2,64,provider);}
            catch(const std::runtime_error&){refused=true;}
            if(!refused || Storage::allocations!=allocations+1)
                throw std::runtime_error("C2 attention retried allocation after cleanup quarantine");
            return; // Permanent retained handles: this mode requires an isolated future process.
        }
        {
            using Owner=Exl3AttentionStageStorage;
            Events::creates=Events::destroys=Events::fail_at=0;
            PreparedAttentionStageBudget budget;budget.device_limit=64*4096;
            budget.metadata_limit=Owner::metadata_bytes()-1;
            const auto calls=Storage::allocations;
            bool refused=false;
            try{Owner::create_startup(budget,64,provider);}catch(const Exl3ResourceReservationExhausted&){refused=true;}
            if(!refused || Storage::allocations!=calls)
                throw std::runtime_error("attention storage control allocation escaped precredit");
            ++budget.metadata_limit;
            auto owner=Owner::create_startup(budget,64,provider);
            std::weak_ptr<Owner> weak=owner;
            RetainedDeviceLedger device;RetainedDescriptorLedger metadata;
            if(Owner::attach_device_credit(owner,device.acquire(64*4096-1)) ||
                !Owner::attach_device_credit(owner,device.acquire(64*4096)) ||
                Owner::attach_device_credit(owner,device.acquire(64*4096)) ||
                Owner::attach_metadata_credit(owner,metadata.acquire(Owner::metadata_bytes()-1)) ||
                !Owner::attach_metadata_credit(owner,metadata.acquire(Owner::metadata_bytes())) ||
                Owner::attach_metadata_credit(owner,metadata.acquire(Owner::metadata_bytes())))
                throw std::runtime_error("attention storage exact/duplicate retirement credits");
            budget.retained={};owner.reset();
            if(!weak.expired() || device.bytes() ||
                metadata.bytes()!=bounded_shared_allocation_bytes<Owner>())
                throw std::runtime_error("attention storage clean retirement lost weak control credit");
            weak.reset();
            if(metadata.bytes())throw std::runtime_error("attention storage final weak credit leak");
        }
        {
            PreparedKVRegistrationAuthority transaction;transaction.fail_commit=true;
            const auto frees=Storage::releases;
            Events::creates=Events::destroys=Events::fail_at=0;
            bool refused=false;
            try{exl3_create_attention_stage_lanes(transaction,2,64,provider);}
            catch(const std::runtime_error& error) {
                refused=std::string_view(error.what())=="injected post-factory inventory commit failure";
            }
            if(!refused || transaction.sealed || transaction.constructor_device.bytes() ||
                transaction.constructor_metadata.bytes() || Storage::releases!=frees+2 || Events::destroys!=8)
                throw std::runtime_error("C2 attention commit rollback did not release provisional owners");
            transaction.fail_commit=false;
            auto lanes=exl3_create_attention_stage_lanes(transaction,2,64,provider);
            if(!lanes[0].storage || !lanes[1].storage || transaction.constructor_device.bytes() ||
                transaction.constructor_metadata.bytes())
                throw std::runtime_error("C2 attention committed credits were not handed to inventory");
        }
        for(unsigned malformed=1;malformed<=4;++malformed) {
            PreparedKVRegistrationAuthority transaction;
            transaction.malformed_constructor_credit=malformed;
            transaction.malformed_constructor_credit_on_call=2;
            Events::creates=Events::destroys=Events::fail_at=0;
            const auto allocations=Storage::allocations,frees=Storage::releases;
            const auto blocks=bounded_shared_live_blocks_for_test<Exl3AttentionStageStorage>();
            bool refused=false;
            try{(void)exl3_create_attention_stage_lanes(transaction,2,64,provider);}
            catch(const std::invalid_argument& error){refused=std::string_view(error.what())==
                "attention stage constructor credit extent";}
            if(!refused || transaction.constructor_credit_calls!=2 ||
                Storage::allocations!=allocations+1 || Storage::releases!=frees+1 ||
                Events::creates!=4 || Events::destroys!=4 || transaction.sealed ||
                bounded_shared_live_blocks_for_test<Exl3AttentionStageStorage>()!=blocks ||
                transaction.constructor_device.bytes() || transaction.constructor_metadata.bytes() ||
                transaction.retained.totals()!=Exl3ResourceInventory::Totals{})
                throw std::runtime_error("second-lane malformed grant failed to unwind deferred first-lane owners");
        }
        for(unsigned fail=1;fail<=4;++fail) {
            PreparedKVRegistrationAuthority reserved;
            Events::creates=Events::destroys=0;Events::fail_at=fail;
            const auto frees=Storage::releases;
            bool refused=false;
            try{Exl3AttentionStageStorage::create_startup(reserved,64,provider);}
            catch(const std::runtime_error&){refused=true;}
            if(!refused || Events::destroys!=fail-1 || Storage::releases!=frees+1 ||
                reserved.constructor_device.bytes() || reserved.constructor_metadata.bytes() || reserved.sealed)
                throw std::runtime_error("partial attention stage event allocation failed to roll back");
        }
        Events::creates=Events::destroys=Events::fail_at=0;
        PreparedKVRegistrationAuthority reserved;reserved.exhaust=true;
        for(unsigned malformed=1;malformed<=4;++malformed) {
            PreparedKVRegistrationAuthority bad_credit;bad_credit.malformed_constructor_credit=malformed;
            const auto allocations_before=Storage::allocations,releases_before=Storage::releases;
            const auto blocks=bounded_shared_live_blocks_for_test<Exl3AttentionStageStorage>();
            bool refused=false;
            try{(void)Exl3AttentionStageStorage::create_startup(bad_credit,64,provider);}
            catch(const std::invalid_argument& error){refused=std::string_view(error.what())==
                "attention stage constructor credit extent";}
            if(!refused || Storage::allocations!=allocations_before || Storage::releases!=releases_before ||
                Events::creates || Events::destroys ||
                bounded_shared_live_blocks_for_test<Exl3AttentionStageStorage>()!=blocks ||
                bad_credit.constructor_device.bytes() || bad_credit.constructor_metadata.bytes() ||
                bad_credit.retained.totals()!=Exl3ResourceInventory::Totals{})
                throw std::runtime_error("malformed attention grant created destination or events");
        }
        const auto allocations=Storage::allocations;
        bool refused=false;
        try{Exl3AttentionStageStorage::create_startup(reserved,64,provider);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        if(!refused || Storage::allocations!=allocations || Events::creates)
            throw std::runtime_error("attention destination allocated before reservation");
        reserved.exhaust=false;
        const auto releases=Storage::releases;
        auto owner=Exl3AttentionStageStorage::create_startup(reserved,64,provider);
        if(reserved.constructor_device.bytes() || reserved.constructor_metadata.bytes())
            throw std::runtime_error("committed attention storage retained provisional credits");
        if(owner->plane(1)-owner->plane(0)!=64*1024 ||
            owner->producer_event(0)==owner->consumer_event(0) ||
            owner->producer_event(1)==owner->producer_event(0))
            throw std::runtime_error("attention destination plane/event layout");
        auto source=std::make_shared<Exl3ExactKVPage>();source->rows=1;source->k[0].resize(1024);
        Exl3AttentionStage stages;
        const auto producer=owner->producer_event(0),consumer=owner->consumer_event(0);
        {
            auto outside=std::make_shared<Exl3ExactKVPage>();outside->first=64;outside->rows=1;outside->k[0].resize(1024);
            bool capacity_refused=false;
            try{exl3_bind_attention_stage(stages,
                Exl3ExactKVExtent::view(outside,0,Exl3ExactKVExtent::Plane::key,65),owner,1,1);}
            catch(const std::invalid_argument&){capacity_refused=true;}
            if(!capacity_refused)throw std::runtime_error("attention stage admitted destination overflow");
        }
        const auto binding=exl3_bind_attention_stage(stages,
            Exl3ExactKVExtent::view(source,0,Exl3ExactKVExtent::Plane::key,1),owner,1,1);
        const auto ticket=binding.ticket;
        if(binding.destination!=owner->plane(0) || binding.producer!=producer || binding.consumer!=consumer)
            throw std::runtime_error("attention stage binding lost physical owner identity");
        const Exl3AttentionStage::Ticket stripped{ticket.slot,ticket.generation};
        const Exl3AttentionStage::Ticket empty;
        if(!ticket.owner_identity || empty.slot || empty.generation || empty.owner_identity ||
            stages.producer_pending(stripped,reinterpret_cast<std::uintptr_t>(producer)) ||
            stages.producer_finished(stripped,reinterpret_cast<std::uintptr_t>(producer),0) ||
            stages.cancel(stripped) || stages.cancel(empty) ||
            !stages.producer_pending(ticket,reinterpret_cast<std::uintptr_t>(producer)))
            throw std::runtime_error("attention binding accepted stripped/default ticket or damaged current producer");
        bool stripped_consume=false;
        try{stages.consume(stripped,0,Exl3ExactKVExtent::Plane::key);}
        catch(const std::invalid_argument&){stripped_consume=true;}
        if(!stripped_consume)throw std::runtime_error("attention binding consumed an ownerless ticket");
        std::weak_ptr<Exl3AttentionStageStorage> weak=owner;
        owner.reset();reserved.retained={};
        if(weak.expired() || Storage::releases!=releases || Events::destroys)
            throw std::runtime_error("active attention stage lost destination/events");
        if(!stages.producer_finished(ticket,reinterpret_cast<std::uintptr_t>(producer),0))
            throw std::runtime_error("owned attention producer completion refused");
        stages.consume(ticket,0,Exl3ExactKVExtent::Plane::key);
        if(!stages.consumer_finished(ticket,reinterpret_cast<std::uintptr_t>(consumer),0) ||
            !weak.expired() || Storage::releases!=releases+1 || Events::destroys!=4)
            throw std::runtime_error("final attention consumer did not release complete backing");
    }
    {
        PreparedKVRegistrationAuthority authority;
        using Events=PreparedAttentionStageEvents;using Storage=PreparedDevicePageStorageProvider;
        Events::creates=Events::destroys=Events::fail_at=0;
        auto storage=Exl3AttentionStageStorage::create_startup(authority,64,
            {Storage::allocate,Storage::release,Events::create,Events::destroy});
        auto stages=Exl3AttentionStage::create_startup(authority);
        auto source=std::make_shared<Exl3ExactKVPage>();source->rows=1;source->k[0].assign(1024,0x1234);
        source->v[0].assign(1024,0x4321);
        using Copy=PreparedKVUploadProvider;
        Copy::copies=Copy::records=Copy::waits=0;
        {
            using Command=Exl3AttentionStageCommand;
            std::shared_ptr<Exl3AttentionStage> unowned_stage(std::shared_ptr<Exl3AttentionStage>{},stages.get());
            std::shared_ptr<Exl3AttentionStageStorage> unowned_storage(std::shared_ptr<Exl3AttentionStageStorage>{},storage.get());
            auto history=std::make_shared<Command::History>();history->push_back(source);
            std::shared_ptr<Command::History> unowned_history(std::shared_ptr<Command::History>{},history.get());
            for(unsigned missing=0;missing<3;++missing) {
                bool refused=false;
                try{Command::submit_history(missing==0?unowned_stage:stages,missing==1?unowned_history:history,
                    0,Exl3ExactKVExtent::Plane::key,1,missing==2?unowned_storage:storage,1,1,nullptr,
                    {Copy::copy,Copy::record,Copy::wait});}
                catch(const std::invalid_argument&){refused=true;}
                if(!refused || stages->uncertain() || Copy::copies || Copy::records || Copy::waits)
                    throw std::runtime_error("history command accepted nonowning dependency or submitted work");
            }
            for(bool missing_stage:{false,true}) {
                bool refused=false;
                try{Command::submit(missing_stage?unowned_stage:stages,
                    Exl3ExactKVExtent::view(source,0,Exl3ExactKVExtent::Plane::key,1),
                    missing_stage?storage:unowned_storage,1,1,nullptr,{Copy::copy,Copy::record,Copy::wait});}
                catch(const std::invalid_argument&){refused=true;}
                if(!refused || stages->uncertain() || Copy::copies || Copy::records || Copy::waits)
                    throw std::runtime_error("single extent command accepted nonowning dependency");
            }
        }
        auto command=Exl3AttentionStageCommand::submit(stages,
            Exl3ExactKVExtent::view(source,0,Exl3ExactKVExtent::Plane::key,1),storage,1,1,nullptr,
            {Copy::copy,Copy::record,Copy::wait});
        if(Copy::copies!=1 || Copy::records!=1 || Copy::waits)
            throw std::runtime_error("attention stage submit waited for producer");
        command.await_producer();
        if(command.consume(0,Exl3ExactKVExtent::Plane::key)!=storage->plane(0) ||
            !std::equal(source->k[0].begin(),source->k[0].end(),storage->plane(0)))
            throw std::runtime_error("attention stage command changed represented extent");
        command.finish_consumer(nullptr);
        const auto waits=Copy::waits;
        bool replay=false;
        try{command.finish_consumer(nullptr);}catch(const std::logic_error&){replay=true;}
        if(!replay || Copy::waits!=waits || waits!=2 || Copy::records!=2)
            throw std::runtime_error("attention stage completion replay reached provider");
        // Reuse the actual storage/events while the completed command survives.
        // Its old generation must not wait on, consume or cancel the new work.
        auto next=Exl3AttentionStageCommand::submit(stages,
            Exl3ExactKVExtent::view(source,0,Exl3ExactKVExtent::Plane::key,1),storage,1,2,nullptr,
            {Copy::copy,Copy::record,Copy::wait});
        const auto records_before=Copy::records,waits_before=Copy::waits,copies_before=Copy::copies;
        bool old_producer=false,old_consumer=false,old_view=false;
        try{command.await_producer();}catch(const std::logic_error&){old_producer=true;}
        try{command.finish_consumer(nullptr);}catch(const std::logic_error&){old_consumer=true;}
        try{command.consume(0,Exl3ExactKVExtent::Plane::key);}catch(const std::logic_error&){old_view=true;}
        if(!old_producer || !old_consumer || !old_view || command.cancel() ||
            Copy::records!=records_before || Copy::waits!=waits_before || Copy::copies!=copies_before)
            throw std::runtime_error("stale stage command reached provider or changed replacement");
        auto moved=std::move(next);
        bool moved_producer=false,moved_consumer=false,moved_view=false;
        try{next.await_producer();}catch(const std::logic_error&){moved_producer=true;}
        try{next.finish_consumer(nullptr);}catch(const std::logic_error&){moved_consumer=true;}
        try{next.consume(0,Exl3ExactKVExtent::Plane::key);}catch(const std::logic_error&){moved_view=true;}
        if(!moved_producer || !moved_consumer || !moved_view || next.cancel() ||
            Copy::records!=records_before || Copy::waits!=waits_before || Copy::copies!=copies_before)
            throw std::runtime_error("moved stage command retained provider access");
        moved.await_producer();
        if(moved.consume(0,Exl3ExactKVExtent::Plane::key)!=storage->plane(0))
            throw std::runtime_error("moved stage command lost replacement destination");
        moved.finish_consumer(nullptr);
        if(Copy::records!=records_before+1 || Copy::waits!=waits_before+2 || stages->uncertain())
            throw std::runtime_error("replacement stage command lost completion ownership");
        const auto ready_records=Copy::records,ready_waits=Copy::waits;
        {
            auto abandoned=Exl3AttentionStageCommand::submit(stages,
                Exl3ExactKVExtent::view(source,0,Exl3ExactKVExtent::Plane::key,1),storage,1,3,nullptr,
                {Copy::copy,Copy::record,Copy::wait});
            abandoned.await_producer();
            // Simulate a peer refusing admission after this producer completed.
            // No consumer has begun, so scope exit needs no new event operation.
        }
        if(stages->uncertain() || Copy::records!=ready_records+1 || Copy::waits!=ready_waits+1)
            throw std::runtime_error("ready abandoned command retained slot or invoked provider");
        auto recovered=Exl3AttentionStageCommand::submit(stages,
            Exl3ExactKVExtent::view(source,0,Exl3ExactKVExtent::Plane::key,1),storage,1,4,nullptr,
            {Copy::copy,Copy::record,Copy::wait});
        recovered.await_producer();
        recovered.consume(0,Exl3ExactKVExtent::Plane::key);recovered.finish_consumer(nullptr);
    }
    {
        PreparedKVRegistrationAuthority authority;
        using Events=PreparedAttentionStageEvents;using Storage=PreparedDevicePageStorageProvider;
        using Copy=PreparedKVUploadProvider;using Command=Exl3AttentionStageCommand;
        Events::creates=Events::destroys=Events::fail_at=0;
        auto storage=Exl3AttentionStageStorage::create_startup(authority,128,
            {Storage::allocate,Storage::release,Events::create,Events::destroy});
        auto stages=Exl3AttentionStage::create_startup(authority);
        auto history=std::make_shared<Command::History>();
        std::array<std::weak_ptr<const Exl3ExactKVPage>,2> pages;
        for(int i=0;i<2;++i) {
            auto page=std::make_shared<Exl3ExactKVPage>();page->first=i*64;page->rows=i?3:64;
            page->k[0].assign(page->rows*1024,static_cast<std::uint16_t>(0x1100+i));
            page->v[0].assign(page->rows*1024,static_cast<std::uint16_t>(0x2200+i));
            pages[i]=page;history->push_back(std::move(page));
        }
        std::fill_n(storage->plane(0),128*1024,0x6d3b);
        Copy::copies=Copy::records=Copy::waits=0;
        auto history_pool=Exl3AttentionStageHistory::create_startup(authority);
        {
            PreparedKVRegistrationAuthority isolated;
            const auto baseline=Exl3SharedControlAccounting::live_bytes.load();
            const auto before_failure=isolated.retained.totals();
            for(int failure=0;failure<2;++failure) {
                bool allocation_refused=false;
                try{Exl3AttentionStageHistory::create_startup(isolated,failure==0,failure==1);}
                catch(const std::bad_alloc&){allocation_refused=true;}
                if(!allocation_refused || isolated.retained.totals()!=before_failure ||
                    Exl3SharedControlAccounting::live_bytes.load()!=baseline)
                    throw std::runtime_error("history control failure retained snapshot or inventory");
            }
            auto owned=Exl3AttentionStageHistory::create_startup(isolated);
            using Pool=Exl3AttentionStageHistory;
            RetainedDescriptorLedger retired;
            auto snapshot=owned->bind(*history,66,0);
            std::weak_ptr<const Pool::Snapshot> weak_snapshot=snapshot;
            if(Pool::attach_retirement_credit(owned,retired.acquire(Pool::metadata_bytes()-1)) ||
                !Pool::attach_retirement_credit(owned,retired.acquire(Pool::metadata_bytes())) ||
                Pool::attach_retirement_credit(owned,retired.acquire(Pool::metadata_bytes())))
                throw std::runtime_error("history retirement short/exact/duplicate credit contract");
            std::weak_ptr<Exl3AttentionStageHistory> weak=owned;
            if(Exl3SharedControlAccounting::live_bytes.load()!=baseline+1024)
                throw std::runtime_error("reserved history omitted one physical control block");
            owned.reset();
            if(weak.expired())throw std::runtime_error("reserved history escaped inventory retention");
            isolated.retained={};
            if(!weak.expired() || Exl3SharedControlAccounting::live_bytes.load()!=baseline+1024 ||
                retired.bytes()!=sizeof(Pool::Snapshot)+1024 || snapshot->size()!=2)
                throw std::runtime_error("history outer retirement lost independently held snapshot credit");
            snapshot.reset();
            if(!weak_snapshot.expired() || retired.bytes()!=1024)
                throw std::runtime_error("history snapshot strong release lost weak control charges");
            weak_snapshot.reset();
            if(retired.bytes()!=512 || Exl3SharedControlAccounting::live_bytes.load()!=baseline+512)
                throw std::runtime_error("reserved history weak survivor retained wrong control extent");
            weak.reset();
            if(retired.bytes() || Exl3SharedControlAccounting::live_bytes.load()!=baseline)
                throw std::runtime_error("reserved history final weak release leaked control bytes");
        }
        auto snapshot=history_pool->bind(*history,66,0);
        auto registrations=Exl3KVRegistrationCache::create_startup(authority);
        auto registration_read=Exl3KVRegistrationCache::acquire_external_read(registrations);
        if(!registration_read)throw std::runtime_error("history fixture missing registration guard");
        {
            auto survivor=std::move(*registration_read);
            if(registration_read->valid() || !survivor.valid() || registrations->external_readers()!=1)
                throw std::runtime_error("moved registration token lost unique reader ownership");
            const auto copies_before=Copy::copies,records_before=Copy::records;
            bool refused=false;
            try{auto invalid=Command::submit_history(stages,snapshot,0,Exl3ExactKVExtent::Plane::key,66,
                storage,1,1,nullptr,{Copy::copy,Copy::record,Copy::wait},0,std::move(registration_read));}
            catch(const std::invalid_argument&){refused=true;}
            if(!refused || Copy::copies!=copies_before || Copy::records!=records_before || registrations->external_readers()!=1)
                throw std::runtime_error("moved registration token submitted work or released surviving reader");
            registration_read.reset();registration_read.emplace(std::move(survivor));
        }
        auto command=Command::submit_history(stages,snapshot,0,Exl3ExactKVExtent::Plane::key,66,
            storage,1,1,nullptr,{Copy::copy,Copy::record,Copy::wait},0,std::move(registration_read));
        snapshot.reset();
        bool rebind_refused=false;
        try{history_pool->bind(*history,66,0);}catch(const std::logic_error&){rebind_refused=true;}
        if(!rebind_refused)throw std::runtime_error("history pool rebound while stage retained snapshot");
        history.reset();
        if(pages[0].expired() || pages[1].expired() || Copy::copies!=2 || Copy::records!=1 || Copy::waits)
            throw std::runtime_error("history staging did not retain complete source or batched readiness");
        command.await_producer();command.consume(0,Exl3ExactKVExtent::Plane::key);
        if(registrations->external_readers()!=1)
            throw std::runtime_error("history producer completion released final-consumer guard");
        for(int row=0;row<128;++row) for(int column=0;column<1024;++column) {
            const auto expected=row<64?0x1100:row<66?0x1101:0x6d3b;
            if(storage->plane(0)[row*1024+column]!=expected)
                throw std::runtime_error("history staging page seam/tail bounds");
        }
        command.finish_consumer(nullptr);
        history_pool->clear();
        if(registrations->external_readers()!=0)
            throw std::runtime_error("history consumer completion retained registration guard");
        if(!pages[0].expired() || !pages[1].expired())
            throw std::runtime_error("completed history stage retained source pages");
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ATTENTION_HISTORY_FAILURE");mode && *mode) {
        using Copy=PreparedKVUploadProvider;using Command=Exl3AttentionStageCommand;
        using Events=PreparedAttentionStageEvents;using Storage=PreparedDevicePageStorageProvider;
        const std::string_view selected(mode);
        if(selected!="1" && selected!="2")throw std::invalid_argument("attention history failure must be 1 or 2");
        Copy::fail_copy_at=selected=="1"?1U:2U;Copy::copies=Copy::records=Copy::waits=0;
        Events::creates=Events::destroys=Events::fail_at=0;
        const auto before=Exl3KVTransferLease::quarantined_records();
        std::array<std::weak_ptr<const Exl3ExactKVPage>,2> pages;
        std::weak_ptr<Exl3AttentionStageStorage> backing;
        std::weak_ptr<Exl3KVRegistrationCache> failed_registrations;
        RetainedDescriptorLedger history_metadata;
        std::weak_ptr<Exl3AttentionStageHistory> weak_pool;
        {
            PreparedKVRegistrationAuthority authority;
            auto storage=Exl3AttentionStageStorage::create_startup(authority,128,
                {Storage::allocate,Storage::release,Events::create,Events::destroy});
            backing=storage;
            auto stages=Exl3AttentionStage::create_startup(authority);
            auto history=std::make_shared<Command::History>();
            for(int i=0;i<2;++i) {
                auto page=std::make_shared<Exl3ExactKVPage>();page->first=i*64;page->rows=64;
                page->k[0].resize(64*1024);page->v[0].resize(64*1024);
                pages[i]=page;history->push_back(std::move(page));
            }
            bool failed=false;
            auto pool=Exl3AttentionStageHistory::create_startup(authority);weak_pool=pool;
            if(!Exl3AttentionStageHistory::attach_retirement_credit(pool,
                history_metadata.acquire(Exl3AttentionStageHistory::metadata_bytes())))
                throw std::runtime_error("failed history fixture missing metadata lifetime credit");
            auto snapshot=pool->bind(*history,128,0);
            history.reset();
            auto registrations=Exl3KVRegistrationCache::create_startup(authority);
            failed_registrations=registrations;
            auto registration_read=Exl3KVRegistrationCache::acquire_external_read(registrations);
            if(!registration_read)throw std::runtime_error("failed history fixture missing registration guard");
            try{Command::submit_history(stages,snapshot,0,Exl3ExactKVExtent::Plane::key,128,storage,1,1,nullptr,
                {Copy::copy,Copy::record,Copy::wait},0,std::move(registration_read));}
            catch(const std::runtime_error&){failed=true;}
            if(!failed || Copy::copies!=Copy::fail_copy_at || Copy::records || Copy::waits)
                throw std::runtime_error("partial history failure submitted later work");
        }
        if(Exl3KVTransferLease::quarantined_records()!=before+1 || backing.expired() ||
            pages[0].expired() || pages[1].expired())
            throw std::runtime_error("partial history failure lost complete source/destination ownership");
        if(!weak_pool.expired() || history_metadata.bytes()!=sizeof(Exl3AttentionStageHistory::Snapshot)+1024)
            throw std::runtime_error("failed history transfer lost independently retained snapshot credit");
        weak_pool.reset();
        if(history_metadata.bytes()!=sizeof(Exl3AttentionStageHistory::Snapshot)+512)
            throw std::runtime_error("failed history outer weak release discarded snapshot credit");
        const auto registrations=failed_registrations.lock();
        if(!registrations || registrations->external_readers()!=1)
            throw std::runtime_error("partial history quarantine lost registration guard");
        return; // Isolated future process; Engine reload refusal follows in caller.
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ATTENTION_STORAGE_FAILURE");mode && *mode) {
        const bool event=std::string_view(mode)=="event";
        if(!event && std::string_view(mode)!="release")throw std::invalid_argument("attention storage failure mode");
        using Events=PreparedAttentionStageEvents;using Storage=PreparedDevicePageStorageProvider;
        Events::creates=Events::destroys=Events::fail_at=0;
        PreparedKVRegistrationAuthority reserved;
        Exl3AttentionStageStorageProvider provider{Storage::allocate,Storage::release,Events::create,Events::destroy};
        auto owner=Exl3AttentionStageStorage::create_startup(reserved,64,provider);
        const auto before=Exl3AttentionStageStorage::quarantined_records();
        const auto frees=Storage::releases;
        RetainedDeviceLedger device_credit;RetainedDescriptorLedger metadata_credit;
        std::weak_ptr<Exl3AttentionStageStorage> weak=owner;
        if(!Exl3AttentionStageStorage::attach_device_credit(owner,device_credit.acquire(64*4096)) ||
            !Exl3AttentionStageStorage::attach_metadata_credit(owner,
                metadata_credit.acquire(Exl3AttentionStageStorage::metadata_bytes())))
            throw std::runtime_error("attention committed retirement credit attachment failed");
        if(event)Events::destroy_error=cudaErrorUnknown;else Storage::release_error=cudaErrorUnknown;
        owner.reset();reserved.retained={};
        if(Exl3AttentionStageStorage::quarantined_records()!=before+1 ||
            Events::destroys!=(event?1U:4U) || Storage::releases!=frees+(event?0U:1U))
            throw std::runtime_error("attention storage failed cleanup was not quarantined at first failure");
        if(!weak.expired() || device_credit.bytes()!=64*4096 ||
            metadata_credit.bytes()!=Exl3AttentionStageStorage::metadata_bytes())
            throw std::runtime_error("attention failed cleanup lost device/state/control charges");
        weak.reset();
        if(metadata_credit.bytes()!=Exl3AttentionStageStorage::retirement_metadata_bytes())
            throw std::runtime_error("attention weak release discarded quarantined State charge");
        const auto allocations=Storage::allocations;
        bool refused=false;
        try{Exl3AttentionStageStorage::create_startup(reserved,64,provider);}catch(const std::runtime_error&){refused=true;}
        if(!refused || Storage::allocations!=allocations)throw std::runtime_error("attention storage retried after quarantine");
        return; // Separate future process; caller checks Engine reload refusal.
    }
    {
        // A direct attention consumer reads both staged planes through final use.
        // Completing K must not retire the common allocation still read by V.
        const auto* mode=std::getenv("NINFER_TEST_ATTENTION_STAGE_SECOND_PLANE_FAILURE");
        const bool fail_second=mode && std::string_view(mode)=="1";
        using Events=PreparedAttentionStageEvents;using Storage=PreparedDevicePageStorageProvider;
        using Copy=PreparedKVUploadProvider;using Command=Exl3AttentionStageCommand;
        Events::creates=Events::destroys=Events::fail_at=0;
        const auto releases=Storage::releases;
        const auto quarantined=Exl3KVTransferLease::quarantined_records();
        std::weak_ptr<Exl3AttentionStageStorage> weak_storage;
        std::weak_ptr<Command::History> weak_history;
        {
            PreparedKVRegistrationAuthority authority;
            auto storage=Exl3AttentionStageStorage::create_startup(authority,64,
                {Storage::allocate,Storage::release,Events::create,Events::destroy});
            auto stages=Exl3AttentionStage::create_startup(authority);
            auto history=std::make_shared<Command::History>();
            auto page=std::make_shared<Exl3ExactKVPage>();page->rows=1;
            page->k[0].assign(1024,0x1234);page->v[0].assign(1024,0x4321);
            history->push_back(page);page.reset();
            weak_storage=storage;weak_history=history;
            {
                // A busy second slot must refuse the pair before an otherwise
                // available first slot submits anything or acquires a lease.
                auto occupied_value=Command::submit_history(stages,history,0,Exl3ExactKVExtent::Plane::value,1,
                    storage,1,1,nullptr,{Copy::copy,Copy::record,Copy::wait});
                occupied_value.await_producer();
                const auto copies=Copy::copies,records=Copy::records,waits=Copy::waits;
                bool busy=false;
                try{Command::require_history_pair(stages,history,0,1,storage,1,2);}
                catch(const std::logic_error&){busy=true;}
                if(!busy || Copy::copies!=copies || Copy::records!=records || Copy::waits!=waits)
                    throw std::runtime_error("busy second stage admitted pair or reached provider");
            }
            if(stages->uncertain())
                throw std::runtime_error("pair preflight changed idle first slot or retained cancelled peer");
            {
                auto registrations=Exl3KVRegistrationCache::create_startup(authority);
                auto reads=Exl3KVRegistrationCache::acquire_external_read_pair(registrations);
                if(!reads)throw std::runtime_error("stage pair fixture missing registration readers");
                auto survivor=std::move((*reads)[1]);
                const auto copies=Copy::copies,records=Copy::records,waits=Copy::waits;
                bool refused=false;
                try{Command::require_history_pair(stages,history,0,1,storage,1,2,reads);}
                catch(const std::invalid_argument&){refused=true;}
                if(!refused || stages->uncertain() || !survivor.valid() ||
                    registrations->external_readers()!=2 || Copy::copies!=copies ||
                    Copy::records!=records || Copy::waits!=waits)
                    throw std::runtime_error("stage pair accepted moved reader or changed retained authority");
            }
            Command::require_history_pair(stages,history,0,1,storage,1,2);
            auto key=Command::submit_history(stages,history,0,Exl3ExactKVExtent::Plane::key,1,
                storage,1,1,nullptr,{Copy::copy,Copy::record,Copy::wait});
            auto value=Command::submit_history(stages,history,0,Exl3ExactKVExtent::Plane::value,1,
                storage,1,1,nullptr,{Copy::copy,Copy::record,Copy::wait},fail_second?5:0);
            key.await_producer();
            const auto records_before=Copy::records,waits_before=Copy::waits;
            bool unready=false;
            try{Command::consume_pair(key,value,0);}catch(const std::logic_error&){unready=true;}
            if(!unready || Copy::records!=records_before || Copy::waits!=waits_before)
                throw std::runtime_error("unready stage pair invoked a consumer provider");
            value.await_producer();
            bool reversed=false,aliased=false;
            try{Command::consume_pair(value,key,0);}catch(const std::logic_error&){reversed=true;}
            try{Command::consume_pair(key,key,0);}catch(const std::invalid_argument&){aliased=true;}
            if(!reversed || !aliased)
                throw std::runtime_error("stage pair accepted reversed or aliased commands");
            const auto planes=Command::consume_pair(key,value,0);
            if(planes[0]!=storage->plane(0) || planes[1]!=storage->plane(1))
                throw std::runtime_error("direct stage pair changed plane address");
            storage.reset();history.reset();authority.retained={};
            key.finish_consumer(nullptr);
            if(weak_storage.expired() || weak_history.expired() || Storage::releases!=releases || Events::destroys)
                throw std::runtime_error("first plane final use released second plane backing");
            bool failed=false;
            try{value.finish_consumer(nullptr);}catch(const std::runtime_error&){failed=true;}
            if(failed!=fail_second)throw std::runtime_error("second plane final-use result differs");
            if(fail_second) {
                const auto waits=Copy::waits,records=Copy::records;
                bool replay=false;
                try{value.finish_consumer(nullptr);}catch(const std::logic_error&){replay=true;}
                if(!replay || value.cancel() || Copy::waits!=waits || Copy::records!=records ||
                    weak_storage.expired() || weak_history.expired())
                    throw std::runtime_error("failed second plane retried or released common backing");
            } else if(!weak_storage.expired() || !weak_history.expired() ||
                Storage::releases!=releases+1 || Events::destroys!=4)
                throw std::runtime_error("second plane final use retained completed backing");
        }
        if(fail_second) {
            if(Exl3KVTransferLease::quarantined_records()!=quarantined+1 || weak_storage.expired() ||
                weak_history.expired() || Storage::releases!=releases || Events::destroys)
                throw std::runtime_error("second plane destruction lost quarantined shared backing");
            return; // Deliberate quarantine; separate future process.
        }
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ATTENTION_STAGE_FAILURE");mode && *mode) {
        const bool producer=std::string_view(mode)=="producer";
        if(!producer && std::string_view(mode)!="consumer")
            throw std::invalid_argument("attention stage failure must be producer or consumer");
        const auto before=Exl3KVTransferLease::quarantined_records();
        std::weak_ptr<const void> retained_page,retained_destination;
        RetainedDescriptorLedger stage_metadata;
        std::weak_ptr<Exl3AttentionStage> weak_stage;
        {
            auto source=std::make_shared<Exl3ExactKVPage>();source->rows=1;source->k[0].resize(1024);
            auto destination=std::make_shared<int>(7);
            retained_page=source;retained_destination=destination;
            PreparedKVRegistrationAuthority authority;
            auto owner=Exl3AttentionStage::create_startup(authority);weak_stage=owner;
            if(Exl3AttentionStage::attach_retirement_credit(owner,
                    stage_metadata.acquire(Exl3AttentionStage::metadata_bytes()-1)) ||
                !Exl3AttentionStage::attach_retirement_credit(owner,
                    stage_metadata.acquire(Exl3AttentionStage::metadata_bytes())) ||
                Exl3AttentionStage::attach_retirement_credit(owner,
                    stage_metadata.acquire(Exl3AttentionStage::metadata_bytes())))
                throw std::runtime_error("stage retirement short/exact/duplicate credits");
            auto& stages=*owner;
            const auto ticket=stages.begin(0,Exl3ExactKVExtent::view(source,0,Exl3ExactKVExtent::Plane::key,1),
                destination,1,1,101,102);
            source.reset();destination.reset();
            if(producer) {
                if(stages.producer_finished(ticket,101,77))throw std::runtime_error("failed stage producer became ready");
                const auto use=stages.final_use_for_test(ticket);
                if(!use || use->event!=101 || use->first_error!=77 || use->ready)
                    throw std::runtime_error("failed stage producer retained planned consumer event identity");
            } else {
                if(!stages.producer_finished(ticket,101,0))throw std::runtime_error("stage producer witness refused");
                stages.consume(ticket,0,Exl3ExactKVExtent::Plane::key);
                if(stages.consumer_finished(ticket,102,77))throw std::runtime_error("failed stage consumer retired");
            }
            if(stages.cancel(ticket) || stages.producer_finished(ticket,101,0) ||
                stages.consumer_finished(ticket,102,0) || retained_page.expired() || retained_destination.expired())
                throw std::runtime_error("failed stage released owners or accepted later success");
        }
        if(Exl3KVTransferLease::quarantined_records()!=before+1 ||
            retained_page.expired() || retained_destination.expired())
            throw std::runtime_error("failed stage destruction did not retain all physical owners");
        if(!weak_stage.expired() || stage_metadata.bytes()!=
            bounded_shared_allocation_bytes<Exl3AttentionStage>()+Exl3KVTransferLease::state_metadata_bytes())
            throw std::runtime_error("failed stage lost State/control credit or retained idle State");
        weak_stage.reset();
        if(stage_metadata.bytes()!=Exl3KVTransferLease::state_metadata_bytes())
            throw std::runtime_error("failed stage final weak release lost quarantined State credit");
        return; // Deliberate quarantine: separate future process per mode.
    }
    auto page=std::make_shared<Exl3ExactKVPage>();page->rows=1;page->k[0].resize(1024);
    RefusingKVRegistrationAuthority authority;
    auto partial=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,1);
    if(Exl3RegisteredKVBacking::acquire_startup(authority,partial) || authority.requested)
        throw std::runtime_error("mutable partial page reached registration allocator");
    page->rows=64;
    const auto truncated=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,1);
    if(truncated.registration_eligible() || Exl3RegisteredKVBacking::acquire_startup(authority,truncated) || authority.requested)
        throw std::runtime_error("full row metadata admitted truncated registered plane");
    page->rows=64;page->k[0].resize(64*1024);
    auto stable=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64);
    bool refused=false;
    try{Exl3RegisteredKVBacking::acquire_startup(authority,stable);}
    catch(const std::runtime_error& error){refused=std::string(error.what())=="fixture registration budget exhausted";}
    if(!refused || !authority.requested)throw std::runtime_error("registration skipped precredit boundary");
    using Provider=PreparedKVRegistrationProvider;
    Provider::acquisitions=0;Provider::releases=0;
    Exl3KVRegistrationProvider callbacks{Provider::acquire,Provider::release};
    PreparedKVRegistrationAuthority credited;
    Provider::acquire_error=cudaErrorMemoryAllocation;
    if(Exl3RegisteredKVBacking::acquire_startup(credited,stable,callbacks) || Provider::releases)
        throw std::runtime_error("registration failure published owner or unregistered absent range");
    Provider::acquire_error=cudaSuccess;
    auto registered=Exl3RegisteredKVBacking::acquire_startup(credited,stable,callbacks);
    if(!registered)throw std::runtime_error("registration owner fixture allocation failed");
    {
        std::shared_ptr<Exl3RegisteredKVBacking> unowned(std::shared_ptr<Exl3RegisteredKVBacking>{},registered.get());
        const auto readers=registered->active_readers();
        if(Exl3RegisteredKVBacking::acquire_reader(unowned,stable) || registered->active_readers()!=readers)
            throw std::runtime_error("nonowning registration alias acquired unretained reader");
    }
    auto alias=Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64,32);
    if(!alias.registration_eligible() || truncated.registration_eligible())
        throw std::runtime_error("completed plane eligibility revived stale view or rejected stable suffix");
    if(!registered || !registered->covers(alias))throw std::runtime_error("registration lost partial alias coverage");
    auto unrelated=std::make_shared<Exl3ExactKVPage>(*page);
    if(registered->covers(Exl3ExactKVExtent::view(unrelated,0,Exl3ExactKVExtent::Plane::key,64)))
        throw std::runtime_error("registration covered unrelated backing");
    {
        PreparedKVRegistrationAuthority upload_credit;
        auto uploader=Exl3RegisteredKVUpload::create_startup(upload_credit);
        PreparedKVUploadProvider::copies=PreparedKVUploadProvider::records=PreparedKVUploadProvider::waits=0;
        PreparedKVUploadProvider::fail_copy_at=0;
        PreparedKVUploadProvider::copy_error=PreparedKVUploadProvider::record_error=
            PreparedKVUploadProvider::wait_error=cudaSuccess;
        PreparedKVUploadProvider::expected_kind=cudaMemcpyHostToDevice;
        auto destination=std::make_shared<std::vector<std::uint16_t>>(64*1024,0x71);
        const auto provider=Exl3KVUploadProvider{PreparedKVUploadProvider::copy,
            PreparedKVUploadProvider::record,PreparedKVUploadProvider::wait};
        const auto event=reinterpret_cast<cudaEvent_t>(std::uintptr_t{123});
        {
            const auto foreign=Exl3ExactKVExtent::view(unrelated,0,Exl3ExactKVExtent::Plane::key,64,32);
            const auto copies=PreparedKVUploadProvider::copies,records=PreparedKVUploadProvider::records,
                waits=PreparedKVUploadProvider::waits;
            bool refused=false;
            try{uploader->submit(foreign,destination->data(),64,nullptr,event,destination,registered,7,1,provider);}
            catch(const std::invalid_argument&){refused=true;}
            if(!refused || uploader->uncertain() || PreparedKVUploadProvider::copies!=copies ||
                PreparedKVUploadProvider::records!=records || PreparedKVUploadProvider::waits!=waits ||
                !registered->covers(alias) || !std::all_of(destination->begin(),destination->end(),[](auto value){return value==0x71;}))
                throw std::runtime_error("foreign copied page bypassed registration identity at upload");
        }
        for(unsigned missing=0;missing<4;++missing) {
            auto incomplete=provider;
            if(missing==0)incomplete.copy=nullptr;
            if(missing==1)incomplete.record=nullptr;
            if(missing==2)incomplete.wait=nullptr;
            const auto copies=PreparedKVUploadProvider::copies,records=PreparedKVUploadProvider::records,
                waits=PreparedKVUploadProvider::waits;
            bool refused=false;
            try{uploader->submit(alias,destination->data(),64,nullptr,event,
                missing==3?std::shared_ptr<const void>{}:destination,registered,7,1,incomplete);}
            catch(const std::invalid_argument&){refused=true;}
            if(!refused || uploader->uncertain() || PreparedKVUploadProvider::copies!=copies ||
                PreparedKVUploadProvider::records!=records || PreparedKVUploadProvider::waits!=waits ||
                !std::all_of(destination->begin(),destination->end(),[](auto value){return value==0x71;}))
                throw std::runtime_error("incomplete upload dependency mutated destination or provider state");
        }
        for(unsigned invalid=0;invalid<3;++invalid) {
            bool refused=false;
            try{Exl3RegisteredKVUpload::require_submission(alias,invalid==0?nullptr:destination->data(),
                invalid==1?63:64,invalid==2?nullptr:event,7,1);}
            catch(const std::invalid_argument&){refused=true;}
            if(!refused)throw std::runtime_error("Engine upload pre-registration admission accepted invalid destination");
        }
        for(unsigned invalid=0;invalid<4;++invalid) {
            const auto copies=PreparedKVUploadProvider::copies,records=PreparedKVUploadProvider::records,
                waits=PreparedKVUploadProvider::waits;
            auto extent=invalid==0?Exl3ExactKVExtent::view(page,0,Exl3ExactKVExtent::Plane::key,64,64):alias;
            void* pointer=destination->data();
            if(invalid==1)pointer=reinterpret_cast<std::byte*>(destination->data())+1;
            if(invalid==2)pointer=reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max()-65535);
            if(invalid==3)pointer=reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max()-131071);
            bool refused=false;
            try{uploader->submit(extent,pointer,64,nullptr,event,destination,registered,7,1,provider);}
            catch(const std::invalid_argument&){refused=invalid<2;}
            catch(const std::overflow_error&){refused=invalid>=2;}
            if(!refused || uploader->uncertain() || PreparedKVUploadProvider::copies!=copies ||
                PreparedKVUploadProvider::records!=records || PreparedKVUploadProvider::waits!=waits ||
                !std::all_of(destination->begin(),destination->end(),[](auto value){return value==0x71;}))
                throw std::runtime_error("invalid upload address/empty extent reached submission or altered destination");
        }
        for(unsigned missing=0;missing<2;++missing) {
            auto invalid_destination=std::make_shared<std::vector<std::uint16_t>>(64*1024);
            std::weak_ptr<std::vector<std::uint16_t>> lifetime=invalid_destination;
            const auto copies_before=PreparedKVUploadProvider::copies;
            bool refused=false;
            try{uploader->submit(alias,invalid_destination->data(),64,nullptr,event,invalid_destination,registered,
                missing?7:0,missing?0:1,provider);}
            catch(const std::invalid_argument&){refused=true;}
            invalid_destination.reset();
            if(!refused || !lifetime.expired() || uploader->uncertain() || PreparedKVUploadProvider::copies!=copies_before)
                throw std::runtime_error("invalid upload scope retained destination or submitted work");
        }
        uploader->copy(alias,destination->data(),64,nullptr,event,destination,registered,7,1,provider);
        for(std::size_t i=0;i<destination->size();++i)
            if((*destination)[i]!=(i<32*1024?0x71:page->k[0][i]))
                throw std::runtime_error("registered upload changed prefix guard or lost alias offset");
        bool capacity=false;
        try{uploader->copy(alias,destination->data(),63,nullptr,event,destination,registered,7,2,provider);}
        catch(const std::invalid_argument&){capacity=true;}
        if(!capacity || PreparedKVUploadProvider::copies!=1 || uploader->uncertain())
            throw std::runtime_error("invalid destination submitted registered transfer");
        uploader->copy(stable,destination->data(),64,nullptr,event,destination,registered,7,3,provider);
        std::weak_ptr<std::vector<std::uint16_t>> released=destination;destination.reset();
        if(!released.expired() || uploader->uncertain() || PreparedKVUploadProvider::copies!=2 ||
            PreparedKVUploadProvider::records!=2 || PreparedKVUploadProvider::waits!=2)
            throw std::runtime_error("completed registered upload retained lane owner or missed final event");
        auto pending_output=std::make_shared<std::vector<std::uint16_t>>(64*1024);
        std::weak_ptr<std::vector<std::uint16_t>> pending_owner=pending_output;
        auto pending_provider=provider;
        const auto generation=uploader->submit(stable,pending_output->data(),64,nullptr,event,
            pending_output,registered,7,4,pending_provider);
        pending_provider.wait=nullptr;pending_output.reset();
        if(pending_owner.expired() || !uploader->uncertain() || PreparedKVUploadProvider::waits!=2)
            throw std::runtime_error("asynchronous submission waited or released destination");
        bool stale_completion=false;
        try{uploader->complete(generation+1);}catch(const std::logic_error&){stale_completion=true;}
        if(!stale_completion || PreparedKVUploadProvider::waits!=2 || pending_owner.expired())
            throw std::runtime_error("stale asynchronous completion reached provider or released owner");
        uploader->complete(generation);
        if(!pending_owner.expired() || uploader->uncertain() || PreparedKVUploadProvider::waits!=3)
            throw std::runtime_error("asynchronous completion lost captured provider or final ownership release");
        bool repeated=false;
        try{uploader->complete(generation);}catch(const std::logic_error&){repeated=true;}
        if(!repeated || PreparedKVUploadProvider::waits!=3)
            throw std::runtime_error("repeated asynchronous completion waited on recycled event");
        auto peer=Exl3RegisteredKVUpload::create_startup(upload_credit);
        auto first_output=std::make_shared<std::vector<std::uint16_t>>(64*1024);
        auto second_output=std::make_shared<std::vector<std::uint16_t>>(64*1024);
        std::weak_ptr<std::vector<std::uint16_t>> first_pending=first_output,second_pending=second_output;
        const auto first_generation=uploader->submit(stable,first_output->data(),64,nullptr,event,
            first_output,registered,7,5,provider);
        const auto second_generation=peer->submit(alias,second_output->data(),64,nullptr,
            reinterpret_cast<cudaEvent_t>(std::uintptr_t{124}),second_output,registered,7,5,provider);
        first_output.reset();second_output.reset();
        if(first_pending.expired() || second_pending.expired() || PreparedKVUploadProvider::waits!=3)
            throw std::runtime_error("two-slot submission serialized waits or lost an owner");
        peer->complete(second_generation);
        if(first_pending.expired() || !second_pending.expired() || !uploader->uncertain())
            throw std::runtime_error("peer slot completion released another transfer");
        uploader->complete(first_generation);
        if(!first_pending.expired() || PreparedKVUploadProvider::waits!=5)
            throw std::runtime_error("two-slot final drain did not release both owners");
    }
    registered.reset();
    if(Provider::releases)throw std::runtime_error("inventory survivor unregistered early");
    credited.retained={};
    if(Provider::releases!=1)throw std::runtime_error("last registration reference did not retire");
    {
        auto cache=Exl3KVRegistrationCache::create_startup(credited);
        const auto calls=Provider::acquisitions;
        auto first=cache->acquire_startup(credited,stable,callbacks);
        auto second=cache->acquire_runtime(credited,7,alias,callbacks);
        if(!first || first!=second || Provider::acquisitions!=calls+1)
            throw std::runtime_error("same backing alias duplicated registration");
        bool stale_runtime=false;
        try{cache->acquire_runtime(credited,6,alias,callbacks);}catch(const std::invalid_argument&){stale_runtime=true;}
        if(!stale_runtime)throw std::runtime_error("cached registration bypassed runtime lease identity");
        credited.expire_after_checks=credited.lease_checks+1;
        const auto before_stale_hit=Provider::acquisitions;
        stale_runtime=false;
        try{cache->acquire_runtime(credited,7,alias,callbacks);}
        catch(const std::invalid_argument&){stale_runtime=true;}
        credited.expire_after_checks=0;
        if(!stale_runtime || Provider::acquisitions!=before_stale_hit || !first->covers(alias) ||
            cache->acquire_runtime(credited,7,alias,callbacks)!=first)
            throw std::runtime_error("changed cached lease escaped or destroyed surviving registration");
        auto independent=cache->acquire_runtime(credited,7,
            Exl3ExactKVExtent::view(unrelated,0,Exl3ExactKVExtent::Plane::key,64),callbacks);
        if(!independent || independent==first || credited.runtime_requests!=1)
            throw std::runtime_error("new backing skipped runtime precredit");
        auto incompatible=callbacks;incompatible.release=nullptr;
        bool incomplete_provider=false;
        try{(void)cache->acquire_startup(credited,alias,incompatible);}
        catch(const std::invalid_argument&){incomplete_provider=true;}
        if(!incomplete_provider)throw std::runtime_error("registration cache accepted incomplete release provider");
        // A valid but different provider selects staging without creating a
        // duplicate registration. That fallback still needs a current lease.
        auto mismatch=callbacks;
        mismatch.release=+[](void*) noexcept -> cudaError_t {return cudaSuccess;};
        const auto before_fallback=Provider::acquisitions;
        credited.expire_after_checks=credited.lease_checks+1;
        bool stale_fallback=false;
        try{(void)cache->acquire_runtime(credited,7,alias,mismatch);}
        catch(const std::invalid_argument&){stale_fallback=true;}
        credited.expire_after_checks=0;
        if(!stale_fallback || Provider::acquisitions!=before_fallback ||
            cache->acquire_runtime(credited,7,alias,mismatch) || !first->covers(alias))
            throw std::runtime_error("stale fallback escaped lease gate or disturbed registered survivor");
        PreparedKVRegistrationAuthority foreign;bool refused_authority=false;
        try{cache->acquire_startup(foreign,stable,callbacks);}
        catch(const std::invalid_argument&){refused_authority=true;}
        if(!refused_authority)throw std::runtime_error("registration credit crossed authority");
        first.reset();second.reset();credited.retained={};
    }
    {
        PreparedKVRegistrationAuthority limited;
        auto cache=Exl3KVRegistrationCache::create_startup(limited);
        const auto calls=Provider::acquisitions;
        limited.exhaust=true;
        if(cache->acquire_runtime(limited,7,stable,callbacks) || Provider::acquisitions!=calls)
            throw std::runtime_error("registration budget refusal reached provider or lost staging fallback");
        limited.exhaust=false;limited.fail_factory=true;
        bool propagated=false;
        try{cache->acquire_runtime(limited,7,stable,callbacks);}
        catch(const std::runtime_error& error){propagated=std::string(error.what())=="prepared unrelated allocation failure";}
        if(!propagated || Provider::acquisitions!=calls)
            throw std::runtime_error("unrelated allocation failure was hidden as staging fallback");
        limited.fail_factory=false;
        if(!cache->acquire_runtime(limited,7,stable,callbacks) || Provider::acquisitions!=calls+1)
            throw std::runtime_error("reservation fallback consumed cache slot or prevented recovery");
    }
    {
        PreparedKVRegistrationAuthority bounded;
        // This case tests the fixed-capacity no-eviction policy. Runtime
        // retirement/pressure recovery is covered by the turnover cases above.
        auto cache=Exl3KVRegistrationCache::create_startup(bounded,false);
        const auto calls=Provider::acquisitions,releases=Provider::releases;
        std::shared_ptr<Exl3ExactKVPage> first_page;
        std::weak_ptr<Exl3RegisteredKVBacking> first_owner;
        for(int index=0;index<65;++index) {
            auto backing=std::make_shared<Exl3ExactKVPage>();
            backing->first=index*64;backing->rows=64;backing->k[0].resize(64*1024);
            const auto extent=Exl3ExactKVExtent::view(backing,0,Exl3ExactKVExtent::Plane::key,backing->first+64);
            auto owner=cache->acquire_runtime(bounded,7,extent,callbacks);
            if(index<64 && !owner)throw std::runtime_error("registration cache exhausted before bounded capacity");
            if(index==64 && owner)throw std::runtime_error("registration cache exceeded fixed metadata capacity");
            if(index==0){first_page=backing;first_owner=owner;}
        }
        if(Provider::acquisitions!=calls+64 || Provider::releases!=releases || first_owner.expired())
            throw std::runtime_error("full registration cache evicted retained backing or called excess provider");
        auto first_again=cache->acquire_runtime(bounded,7,
            Exl3ExactKVExtent::view(first_page,0,Exl3ExactKVExtent::Plane::key,64),callbacks);
        if(first_again!=first_owner.lock() || Provider::acquisitions!=calls+64)
            throw std::runtime_error("full cache rejected existing backing alias");
        first_again.reset();bounded.retained={};
        if(!first_owner.expired() || Provider::releases!=releases+64)
            throw std::runtime_error("idle bounded registration inventory did not retire all owners");
    }
    {
        PreparedDevicePageStorageProvider::allocations=PreparedDevicePageStorageProvider::releases=0;
        PreparedDevicePageStorageProvider::release_error=cudaSuccess;
        auto complete_page=std::make_shared<Exl3ExactKVPage>();complete_page->rows=64;
        for(int bank=0;bank<16;++bank){
            complete_page->k[bank].assign(64*1024,static_cast<std::uint16_t>(bank*2+1));
            complete_page->v[bank].assign(64*1024,static_cast<std::uint16_t>(bank*2+2));
        }
        Exl3DevicePageKey key(std::make_shared<int>(1),complete_page,{0,false},64);
        PreparedKVRegistrationAuthority physical;physical.exhaust=true;
        const Exl3DevicePageStorageProvider provider{PreparedDevicePageStorageProvider::allocate,
            PreparedDevicePageStorageProvider::release};
        for(unsigned malformed=1;malformed<=4;++malformed) {
            PreparedKVRegistrationAuthority bad_credit;bad_credit.malformed_constructor_credit=malformed;
            const auto allocations=PreparedDevicePageStorageProvider::allocations;
            const auto blocks=bounded_shared_live_blocks_for_test<Exl3DevicePageStorage>();
            bool refused=false;
            try{(void)Exl3DevicePageStorage::create_runtime(bad_credit,7,key,provider);}
            catch(const std::invalid_argument& error){refused=std::string_view(error.what())=="device page constructor credit extent";}
            if(!refused || PreparedDevicePageStorageProvider::allocations!=allocations ||
                bounded_shared_live_blocks_for_test<Exl3DevicePageStorage>()!=blocks ||
                bad_credit.constructor_device.bytes() || bad_credit.constructor_metadata.bytes() ||
                bad_credit.retained.totals()!=Exl3ResourceInventory::Totals{})
                throw std::runtime_error("malformed page constructor credits allocated or retained storage");
        }
        {
            auto changed_page=std::make_shared<Exl3ExactKVPage>(*complete_page);
            Exl3DevicePageKey changed_key(std::make_shared<int>(2),changed_page,{0,false},64);
            PreparedKVRegistrationAuthority delayed;
            delayed.before_factory=[&]{changed_page->v[15].clear();};
            const auto before=PreparedDevicePageStorageProvider::allocations;
            bool changed_refused=false;
            try{(void)Exl3DevicePageStorage::create_runtime(delayed,7,changed_key,provider);}
            catch(const std::invalid_argument&){changed_refused=true;}
            if(!changed_refused || PreparedDevicePageStorageProvider::allocations!=before ||
                delayed.retained.totals()!=Exl3ResourceInventory::Totals{})
                throw std::runtime_error("changed page backing reached physical allocation after reservation");
        }
        bool refused=false;
        try{Exl3DevicePageStorage::create_runtime(physical,7,key,provider);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        if(!refused || PreparedDevicePageStorageProvider::allocations)
            throw std::runtime_error("page allocation invoked provider before physical credit");
        physical.exhaust=false;
        auto prepared=Exl3DevicePageStorage::create_runtime(physical,7,key,provider);
        if(!prepared.fill || prepared.fill->ready() || prepared.destination.size()!=Exl3DevicePageFill::elements ||
            PreparedDevicePageStorageProvider::allocations!=1)
            throw std::runtime_error("page allocation failed to publish idle complete extent");
        prepared={};
        if(PreparedDevicePageStorageProvider::releases)
            throw std::runtime_error("physical inventory lost idle page allocation owner");
        physical.retained={};
        if(PreparedDevicePageStorageProvider::releases!=1)
            throw std::runtime_error("idle page allocation did not retire with inventory");
        {
            auto reclaim=Exl3DevicePageStorage::create_runtime(physical,7,key,provider);
            exl3_fill_device_page(reclaim,nullptr,reinterpret_cast<cudaEvent_t>(std::uintptr_t{303}),
                complete_page,7,3,{PreparedKVUploadProvider::copy,PreparedKVUploadProvider::record,PreparedKVUploadProvider::wait});
            const auto allocation=Exl3DevicePageStorage::device_allocation(reclaim.fill);
            const auto retire=[&](const auto&) noexcept {return reclaim.fill->retire_sealed_storage();};
            if(physical.retire_runtime_resource(7,allocation,retire) || PreparedDevicePageStorageProvider::releases!=1)
                throw std::runtime_error("unsealed page released physical storage");
            if(!physical.retire_runtime_metadata_to_lifetime(7,Exl3DevicePageStorage::metadata_allocation(reclaim.fill)) ||
                !reclaim.fill->seal_for_retirement() || !physical.retire_runtime_resource(7,allocation,retire) ||
                PreparedDevicePageStorageProvider::releases!=2 || reclaim.fill->ready_handle())
                throw std::runtime_error("sealed page did not retire physical storage and reader access");
            reclaim={};physical.retained={};
            if(PreparedDevicePageStorageProvider::releases!=2)
                throw std::runtime_error("retired page destructor repeated physical free");
        }
        PreparedKVRegistrationAuthority cached;
        {
            PreparedKVRegistrationAuthority reload_authority;
            auto cache=Exl3DevicePageCache::create_startup(reload_authority);
            auto old_model=cache->acquire(reload_authority,7,key,provider);
            Exl3DevicePageKey reloaded_key(std::make_shared<int>(88),complete_page,{0,false},
                complete_page->first+64);
            auto new_model=cache->acquire(reload_authority,7,reloaded_key,provider);
            if(!old_model.producer || !new_model.producer || old_model.shared==new_model.shared)
                throw std::runtime_error("model reload reused another model's represented device page");
            const auto accounting=cache->accounting();
            if(accounting.unique_allocations!=2 || accounting.device_bytes!=2*Exl3DevicePageStorage::bytes ||
                accounting.retained_source_pages!=1 || accounting.retained_source_host_bytes!=key.retained_host_bytes())
                throw std::runtime_error("model-specific fills duplicated shared host source charge or omitted device storage");
            auto old_again=cache->acquire(reload_authority,7,key,provider);
            auto new_again=cache->acquire(reload_authority,7,reloaded_key,provider);
            if(old_again.shared!=old_model.shared || new_again.shared!=new_model.shared ||
                old_again.producer || new_again.producer)
                throw std::runtime_error("model-specific pending fills lost independent single-flight identity");
        }
        {
            PreparedKVRegistrationAuthority fault_authority;
            auto cache=Exl3DevicePageCache::create_startup(fault_authority);
            std::array<Exl3DevicePageCache::Acquisition,64> pending;
            const auto allocations=PreparedDevicePageStorageProvider::allocations;
            pending[0]=cache->acquire(fault_authority,7,key,provider);
            for(unsigned index=1;index<pending.size();++index) {
                Exl3DevicePageKey distinct(std::make_shared<unsigned>(index),complete_page,{0,false},
                    complete_page->first+64);
                pending[index]=cache->acquire(fault_authority,7,distinct,provider);
            }
            for(const auto& fill:pending)if(!fill.producer || !fill.shared)
                throw std::runtime_error("device page cache exhausted before 64 pending fills");
            const auto occupied=fault_authority.retained.totals();
            const auto requests=fault_authority.runtime_requests;
            const auto releases=PreparedDevicePageStorageProvider::releases;
            Exl3DevicePageKey excess(std::make_shared<int>(99),complete_page,{0,false},complete_page->first+64);
            auto miss=cache->acquire(fault_authority,7,excess,provider);
            auto hit=cache->acquire(fault_authority,7,key,provider);
            if(miss.shared || miss.producer || hit.shared!=pending[0].shared || hit.producer ||
                cache->evict_one(fault_authority,7) || fault_authority.runtime_requests!=requests ||
                PreparedDevicePageStorageProvider::allocations!=allocations+64 ||
                PreparedDevicePageStorageProvider::releases!=releases || fault_authority.retained.totals()!=occupied)
                throw std::runtime_error("full device page cache changed pending fills or allocated on fallback");
            const auto accounting=cache->accounting();
            if(accounting.unique_allocations!=64 || accounting.device_bytes!=64*Exl3DevicePageStorage::bytes ||
                accounting.retained_source_pages!=1 || accounting.retained_source_host_bytes!=key.retained_host_bytes())
                throw std::runtime_error("full pending page cache lost physical or deduplicated host accounting");
            const Exl3KVUploadProvider transfers{PreparedKVUploadProvider::copy,
                PreparedKVUploadProvider::record,PreparedKVUploadProvider::wait};
            for(unsigned index=0;index<2;++index)
                exl3_fill_device_page(*pending[index].producer,nullptr,
                    reinterpret_cast<cudaEvent_t>(std::uintptr_t{710}+index),complete_page,7,80+index,transfers);
            auto busy=pending[0].shared->ready_handle();
            Exl3DevicePageReader busy_reader;
            busy_reader.bind(busy,key,busy->generation(),complete_page);
            const auto reading=busy_reader.begin(7,82,712);
            if(cache->evict_one(fault_authority,7) || PreparedDevicePageStorageProvider::releases!=releases ||
                pending[0].shared->retained_readers()!=1 || busy_reader.plane(0,true).empty())
                throw std::runtime_error("full-cache victim reclaimed active page reader");
            if(!cache->evict_one(fault_authority,7) || PreparedDevicePageStorageProvider::releases!=releases+1 ||
                pending[0].shared->retained_readers()!=1)
                throw std::runtime_error("full-cache victim cursor failed to reach reclaimable peer");
            pending[1]={}; // Final retired fill release makes its lookup slot reusable.
            auto replacement=cache->acquire(fault_authority,7,excess,provider);
            if(!replacement.producer || fault_authority.retained.totals()!=occupied ||
                PreparedDevicePageStorageProvider::allocations!=allocations+65 ||
                cache->accounting().unique_allocations!=64 ||
                cache->accounting().retained_source_host_bytes!=key.retained_host_bytes())
                throw std::runtime_error("full-cache replacement lost bounded storage/source accounting");
            if(!busy_reader.finish(reading,0) || pending[0].shared->retained_readers()!=0)
                throw std::runtime_error("surviving full-cache reader did not complete after peer replacement");
        }
        {
            PreparedKVRegistrationAuthority fault_authority;
            auto cache=Exl3DevicePageCache::create_startup(fault_authority);
            auto first=cache->acquire(fault_authority,7,key,provider);
            cache->fail_next_constructor_for_test(2,false);
            auto hit=cache->acquire(fault_authority,7,key,provider);
            if(hit.shared!=first.shared || hit.producer || !cache->constructor_fault_armed_for_test())
                throw std::runtime_error("page hit consumed constructor failure authority");
            auto other_page=std::make_shared<Exl3ExactKVPage>(*complete_page);other_page->first=64;
            Exl3DevicePageKey other_key(std::make_shared<int>(22),other_page,{0,false},128);
            const auto allocations=PreparedDevicePageStorageProvider::allocations;
            fault_authority.exhaust=true;
            if(cache->acquire(fault_authority,7,other_key,provider).shared ||
                PreparedDevicePageStorageProvider::allocations!=allocations || !cache->constructor_fault_armed_for_test())
                throw std::runtime_error("page budget refusal consumed constructor fault or invoked provider");
            fault_authority.exhaust=false;bool original=false;
            try{(void)cache->acquire(fault_authority,7,other_key,provider);}
            catch(const std::runtime_error& error){original=std::string_view(error.what())=="injected page fill after-state failure";}
            if(!original || cache->constructor_fault_armed_for_test() ||
                PreparedDevicePageStorageProvider::allocations!=allocations+1)
                throw std::runtime_error("page cache miss failed to consume one constructor stage");
            auto retry=cache->acquire(fault_authority,7,other_key,provider);
            if(!retry.producer || retry.shared->failed())throw std::runtime_error("page constructor fault was not one-shot");
        }
        {
            PreparedKVRegistrationAuthority stale_miss;
            auto cache=Exl3DevicePageCache::create_startup(stale_miss);
            stale_miss.expire_after_checks=stale_miss.lease_checks+1;
            const auto allocated=PreparedDevicePageStorageProvider::allocations;
            const auto released=PreparedDevicePageStorageProvider::releases;
            bool refused_miss=false;
            try{(void)cache->acquire(stale_miss,7,key,provider);}
            catch(const std::invalid_argument&){refused_miss=true;}
            stale_miss.expire_after_checks=0;
            if(!refused_miss || cache->acquire(stale_miss,7,key,provider).shared ||
                PreparedDevicePageStorageProvider::allocations!=allocated+1 ||
                PreparedDevicePageStorageProvider::releases!=released ||
                !cache->seal_retired(stale_miss,7,key) || !cache->reclaim_retired(stale_miss,7,key) ||
                PreparedDevicePageStorageProvider::releases!=released+1)
                throw std::runtime_error("stale page miss escaped or failed certified unsubmitted retirement");
            auto retry=cache->acquire(stale_miss,7,key,provider);
            if(!retry.producer || PreparedDevicePageStorageProvider::allocations!=allocated+2)
                throw std::runtime_error("retired stale page miss prevented fresh producer recovery");
            if(cache->evict_one(stale_miss,7))
                throw std::runtime_error("automatic recovery evicted a live pending producer");
            retry={};
            if(!cache->evict_one(stale_miss,7) ||
                PreparedDevicePageStorageProvider::releases!=released+2)
                throw std::runtime_error("automatic recovery retained abandoned unsubmitted storage");
            auto recovered=cache->acquire(stale_miss,7,key,provider);
            if(!recovered.producer || PreparedDevicePageStorageProvider::allocations!=allocated+3)
                throw std::runtime_error("automatic abandoned recovery prevented replacement admission");
        }
        {
            PreparedKVRegistrationAuthority binding_failure;
            auto cache=Exl3DevicePageCache::create_startup(binding_failure);
            binding_failure.fail_host_source_binding=true;
            const auto allocated=PreparedDevicePageStorageProvider::allocations;
            const auto released=PreparedDevicePageStorageProvider::releases;
            const auto copies_before=PreparedKVUploadProvider::copies;
            bool failed=false;
            try{(void)cache->acquire(binding_failure,7,key,provider);}
            catch(const std::runtime_error& error){failed=std::string_view(error.what())==
                "injected host source binding failure";}
            const auto held=cache->accounting();
            if(!failed || binding_failure.host_source_bindings!=0 || held.unique_allocations!=1 ||
                held.retiring_allocations!=1 || held.failed_allocations!=1 || held.in_flight_fill_bytes!=0 ||
                PreparedDevicePageStorageProvider::allocations!=allocated+1 ||
                PreparedDevicePageStorageProvider::releases!=released || PreparedKVUploadProvider::copies!=copies_before)
                throw std::runtime_error("host binding failure escaped producer or lost inventoried allocation");
            if(cache->acquire(binding_failure,7,key,provider).shared ||
                !cache->evict_one(binding_failure,7) || PreparedDevicePageStorageProvider::releases!=released+1)
                throw std::runtime_error("host binding failure admitted late consumer or prevented retirement");
            binding_failure.fail_host_source_binding=false;
            auto retry=cache->acquire(binding_failure,7,key,provider);
            if(!retry.producer || binding_failure.host_source_bindings!=1 ||
                PreparedDevicePageStorageProvider::allocations!=allocated+2 ||
                PreparedKVUploadProvider::copies!=copies_before)
                throw std::runtime_error("host binding failure prevented fresh bound producer retry");
        }
        auto lookup=Exl3DevicePageCache::create_startup(cached);
        const auto allocations=PreparedDevicePageStorageProvider::allocations;
        auto producer=lookup->acquire(cached,7,key,provider);
        auto waiter=lookup->acquire(cached,7,key,provider);
        if(!producer.producer || waiter.producer || producer.shared!=waiter.shared ||
            waiter.shared->ready() || PreparedDevicePageStorageProvider::allocations!=allocations+1)
            throw std::runtime_error("compatible page fill duplicated producer or exposed incomplete page");
        waiter={};
        cached.expire_after_checks=cached.lease_checks+1;
        bool stale_page_hit=false;
        try{(void)lookup->acquire(cached,7,key,provider);}
        catch(const std::invalid_argument&){stale_page_hit=true;}
        cached.expire_after_checks=0;
        auto surviving_waiter=lookup->acquire(cached,7,key,provider);
        if(!stale_page_hit || surviving_waiter.producer ||
            surviving_waiter.shared!=producer.shared || producer.shared->failed() ||
            PreparedDevicePageStorageProvider::allocations!=allocations+1)
            throw std::runtime_error("stale page hit revoked producer or duplicated surviving backing");
        surviving_waiter={};
        const auto pending_accounting=lookup->accounting();
        std::uint64_t expected_source_bytes=sizeof(Exl3ExactKVPage);
        std::uint64_t source_visits=0,visited_source_bytes=0;
        key.visit_retained_host_allocations([&](const void* pointer,std::size_t bytes) {
            if(!pointer || !bytes)throw std::runtime_error("retained source visitor omitted allocation geometry");
            ++source_visits;visited_source_bytes+=bytes;
        });
        for(int bank=0;bank<16;++bank)
            expected_source_bytes+=(key.page()->k[bank].capacity()+key.page()->v[bank].capacity())*sizeof(std::uint16_t);
        if(pending_accounting.retained_source_pages!=1 ||
            pending_accounting.retained_source_host_bytes!=expected_source_bytes ||
            source_visits!=33 || visited_source_bytes!=expected_source_bytes)
            throw std::runtime_error("page cache omitted retained host source or duplicated waiter charge");
        if(pending_accounting.unique_allocations!=1 || pending_accounting.device_bytes!=Exl3DevicePageStorage::bytes ||
            pending_accounting.pending_fill_bytes!=Exl3DevicePageStorage::bytes || pending_accounting.ready_rows!=0 ||
            pending_accounting.in_flight_fill_bytes!=0)
            throw std::runtime_error("aliased fill waiter changed unique allocation accounting");
        if(producer.shared->failed())throw std::runtime_error("cancelled waiter invalidated producer");
        const auto copies=PreparedKVUploadProvider::copies,records=PreparedKVUploadProvider::records,
            waits=PreparedKVUploadProvider::waits;
        Exl3KVUploadProvider transfers{PreparedKVUploadProvider::copy,PreparedKVUploadProvider::record,
            PreparedKVUploadProvider::wait};
        auto malformed=*producer.producer;malformed.destination=malformed.destination.subspan(1);
        for(unsigned invalid=0;invalid<4;++invalid) {
            bool rejected=false;
            try{exl3_fill_device_page(*producer.producer,nullptr,
                invalid==0?nullptr:reinterpret_cast<cudaEvent_t>(std::uintptr_t{301}),
                invalid==3?std::shared_ptr<const void>{}:complete_page,
                invalid==1?0:7,invalid==2?0:1,transfers);}
            catch(const std::invalid_argument&){rejected=true;}
            if(!rejected || producer.shared->failed() || producer.shared->fill_in_flight() ||
                PreparedKVUploadProvider::copies!=copies || PreparedKVUploadProvider::records!=records ||
                PreparedKVUploadProvider::waits!=waits)
                throw std::runtime_error("invalid fill scope mutated producer or reached provider");
        }
        bool wrong_extent=false;
        try{exl3_fill_device_page(malformed,nullptr,reinterpret_cast<cudaEvent_t>(std::uintptr_t{301}),
            complete_page,7,1,transfers);}catch(const std::invalid_argument&){wrong_extent=true;}
        if(!wrong_extent || PreparedKVUploadProvider::copies!=copies)
            throw std::runtime_error("malformed fill destination reached copy provider");
        exl3_fill_device_page(*producer.producer,nullptr,reinterpret_cast<cudaEvent_t>(std::uintptr_t{301}),
            complete_page,7,1,transfers);
        if(PreparedKVUploadProvider::copies!=copies+32 || PreparedKVUploadProvider::records!=records+1 ||
            PreparedKVUploadProvider::waits!=waits+1)
            throw std::runtime_error("all-plane command omitted copies or readiness event");
        const auto filled=producer.shared->view();
        const auto ready_accounting=lookup->accounting();
        if(ready_accounting.unique_allocations!=1 || ready_accounting.pending_fill_bytes!=0 ||
            ready_accounting.ready_rows!=64 || ready_accounting.failed_allocations!=0)
            throw std::runtime_error("ready page accounting duplicated planes or retained pending fill bytes");
        for(int bank=0;bank<16;++bank)for(bool key_plane:{false,true}) {
            const auto plane=filled.plane(bank,key_plane);
            const auto& expected=key_plane?complete_page->k[bank]:complete_page->v[bank];
            if(!std::equal(plane.begin(),plane.end(),expected.begin(),expected.end()))
                throw std::runtime_error("device page fill changed bank/plane contents");
        }
        auto bridge=Exl3DevicePageCopy::create_startup(cached);
        auto attention=Exl3DevicePageAttention::create_startup(cached);
        auto peer_attention=Exl3DevicePageAttention::create_startup(cached);
        const auto attention_copies=PreparedKVUploadProvider::copies;
        {
            std::shared_ptr<const void> nonowning(std::shared_ptr<const void>{},complete_page.get());
            bool refused_owner=false;
            try{(void)attention->begin(producer.shared->ready_handle(),key,15,64,nonowning,
                7,2,reinterpret_cast<cudaEvent_t>(std::uintptr_t{303}));}
            catch(const std::invalid_argument&){refused_owner=true;}
            if(!refused_owner || attention->uncertain() || producer.shared->retained_readers()!=0 ||
                PreparedKVUploadProvider::copies!=attention_copies)
                throw std::runtime_error("attention accepted nonowning event dependency");
        }
        auto prefix=attention->begin(producer.shared->ready_handle(),key,15,64,complete_page,
            7,2,reinterpret_cast<cudaEvent_t>(std::uintptr_t{303}));
        if(prefix.rows!=64 || prefix.k!=filled.plane(15,true).data() ||
            prefix.v!=filled.plane(15,false).data() || !attention->uncertain())
            throw std::runtime_error("attention did not borrow exact shared bank planes");
        if(lookup->accounting().retained_readers!=1)
            throw std::runtime_error("active attention reader absent from physical accounting");
        bool stale_attention=false;
        try{attention->complete(prefix.generation+1,nullptr,transfers);}
        catch(const std::logic_error&){stale_attention=true;}
        if(!stale_attention || !attention->uncertain())
            throw std::runtime_error("stale attention completion released reader");
        const auto singleton_records=PreparedKVUploadProvider::records;
        const auto singleton_waits=PreparedKVUploadProvider::waits;
        const auto source_first=complete_page->first;
        ++complete_page->first;
        bool stale_source=false;
        try{attention->complete(prefix.generation,nullptr,transfers);}
        catch(const std::logic_error&){stale_source=true;}
        complete_page->first=source_first;
        if(!stale_source || !attention->uncertain() || producer.shared->retained_readers()!=1 ||
            PreparedKVUploadProvider::records!=singleton_records || PreparedKVUploadProvider::waits!=singleton_waits)
            throw std::runtime_error("singleton attention accepted stale source or submitted final-use event");
        attention->complete(prefix.generation,nullptr,transfers);
        if(attention->uncertain() || PreparedKVUploadProvider::copies!=attention_copies ||
            lookup->accounting().retained_readers!=0)
            throw std::runtime_error("direct attention copied prefix or retained completed reader");
        {
            const auto event=reinterpret_cast<cudaEvent_t>(std::uintptr_t{410});
            const auto a=attention->begin(producer.shared->ready_handle(),key,0,64,complete_page,7,3,event);
            const auto b=peer_attention->begin(producer.shared->ready_handle(),key,0,64,complete_page,7,3,event);
            using Attention=Exl3DevicePageAttention;
            std::array<Attention::Completion,2> group{{{attention.get(),a.generation},{peer_attention.get(),b.generation}}};
            const auto records=PreparedKVUploadProvider::records,waits=PreparedKVUploadProvider::waits;
            ++group[1].generation;
            bool stale=false;try{Attention::complete_group(group,nullptr,transfers);}catch(const std::logic_error&){stale=true;}
            if(!stale || lookup->accounting().retained_readers!=2 || PreparedKVUploadProvider::records!=records)
                throw std::runtime_error("stale group released peer or submitted event");
            group[1]=group[0];
            bool duplicate=false;try{Attention::complete_group(group,nullptr,transfers);}catch(const std::logic_error&){duplicate=true;}
            if(!duplicate || lookup->accounting().retained_readers!=2 || PreparedKVUploadProvider::records!=records)
                throw std::runtime_error("duplicate group released peer or submitted event");
            group[1]={peer_attention.get(),b.generation};
            if(Attention::require_group(group)!=event || PreparedKVUploadProvider::records!=records ||
                lookup->accounting().retained_readers!=2)
                throw std::runtime_error("precompute group proof submitted event or released ownership");
            Attention::complete_group(group,nullptr,transfers);
            if(lookup->accounting().retained_readers!=0 || PreparedKVUploadProvider::records!=records+1 ||
                PreparedKVUploadProvider::waits!=waits+1)
                throw std::runtime_error("collective attention completion did not use exactly one event/wait");
            for(unsigned mismatch=0;mismatch<5;++mismatch) {
                auto other_owner=std::make_shared<int>(19);
                const auto left=attention->begin(producer.shared->ready_handle(),key,0,64,complete_page,7,4,event);
                const auto right=peer_attention->begin(producer.shared->ready_handle(),key,mismatch==3?1:0,mismatch==4?65:64,
                    mismatch==2?std::shared_ptr<const void>(other_owner):std::shared_ptr<const void>(complete_page),
                    mismatch==0?8:7,mismatch==1?5:4,event);
                group={{{attention.get(),left.generation},{peer_attention.get(),right.generation}}};
                const auto before_records=PreparedKVUploadProvider::records;
                bool rejected=false;try{Attention::require_group(group);}catch(const std::logic_error&){rejected=true;}
                if(!rejected || PreparedKVUploadProvider::records!=before_records || lookup->accounting().retained_readers!=2)
                    throw std::runtime_error("cross-scope group released readers or submitted event");
                attention->complete(left.generation,nullptr,transfers);
                peer_attention->complete(right.generation,nullptr,transfers);
            }
        }
        auto output=std::make_shared<std::vector<std::uint16_t>>(64*1024,0x77);
        auto slice=Exl3ExactKVExtent::view(complete_page,15,Exl3ExactKVExtent::Plane::value,64,32);
        PreparedKVUploadProvider::expected_kind=cudaMemcpyDeviceToDevice;
        {
            const auto before=PreparedKVUploadProvider::copies;
            bool missing_event=false;
            try{bridge->copy(producer.shared->ready_handle(),slice,output->data(),64,nullptr,
                nullptr,output,7,2,transfers);}
            catch(const std::invalid_argument&){missing_event=true;}
            if(!missing_event || bridge->uncertain() || producer.shared->retained_readers()!=0 ||
                PreparedKVUploadProvider::copies!=before)
                throw std::runtime_error("invalid page copy event retained reader or submitted transfer");
        }
        bridge->copy(producer.shared->ready_handle(),slice,output->data(),64,nullptr,
            reinterpret_cast<cudaEvent_t>(std::uintptr_t{302}),output,7,2,transfers);
        PreparedKVUploadProvider::expected_kind=cudaMemcpyHostToDevice;
        for(std::size_t i=0;i<output->size();++i)
            if((*output)[i]!=(i<32*1024?0x77:32))
                throw std::runtime_error("shared page bridge changed private prefix or selected wrong plane");
        std::weak_ptr<std::vector<std::uint16_t>> output_owner=output;output.reset();
        if(!output_owner.expired() || bridge->uncertain())
            throw std::runtime_error("shared page bridge retained completed destination");
        auto ready=lookup->acquire(cached,7,key,provider);
        if(!ready.shared || ready.producer || !ready.shared->ready())
            throw std::runtime_error("cache failed to share completed fill");
        const auto before_retire_allocations=PreparedDevicePageStorageProvider::allocations;
        const auto before_retire_releases=PreparedDevicePageStorageProvider::releases;
        for(unsigned operation=0;operation<4;++operation) {
            cached.expire_after_checks=cached.lease_checks+1;
            bool refused=false;
            try {
                if(operation==0)(void)lookup->retire_lookup(cached,7,key);
                else if(operation==1)(void)lookup->seal_retired(cached,7,key);
                else if(operation==2)(void)lookup->reclaim_retired(cached,7,key);
                else (void)lookup->evict_one(cached,7);
            } catch(const std::invalid_argument&){refused=true;}
            cached.expire_after_checks=0;
            if(!refused || !lookup->acquire(cached,7,key,provider).shared || !ready.shared->ready() ||
                PreparedDevicePageStorageProvider::allocations!=before_retire_allocations ||
                PreparedDevicePageStorageProvider::releases!=before_retire_releases)
                throw std::runtime_error("stale page retirement/seal/reclamation/victim selection mutated live lookup or storage");
        }
        if(!lookup->retire_lookup(cached,7,key) || lookup->retire_lookup(cached,7,key))
            throw std::runtime_error("page logical retirement was missing or replayable");
        if(lookup->seal_retired(cached,7,key))
            throw std::runtime_error("cache sealed while prepared value view still retained storage");
        if(lookup->acquire(cached,7,key,provider).shared || !ready.shared->ready() ||
            !ready.shared->ready_handle() || PreparedDevicePageStorageProvider::allocations!=before_retire_allocations ||
            PreparedDevicePageStorageProvider::releases!=before_retire_releases)
            throw std::runtime_error("retired lookup admitted late reader or reclaimed existing owner");
        auto other_page=std::make_shared<Exl3ExactKVPage>(*complete_page);
        Exl3DevicePageKey other(std::make_shared<int>(2),other_page,{0,false},64);
        auto abandoned=lookup->acquire(cached,7,other,provider);
        if(!abandoned.producer || abandoned.shared==ready.shared)
            throw std::runtime_error("incompatible source shared fill owner");
        if(!lookup->retire_lookup(cached,7,other) || lookup->acquire(cached,7,other,provider).shared ||
            abandoned.shared->failed())
            throw std::runtime_error("retiring pending page cancelled producer or allowed late waiter");
        abandoned={};
        if(lookup->acquire(cached,7,other,provider).shared)
            throw std::runtime_error("abandoned producer remained available to waiters");
        {
            PreparedKVRegistrationAuthority recycling;
            auto cache=Exl3DevicePageCache::create_startup(recycling);
            auto consumer=Exl3DevicePageAttention::create_startup(recycling);
            auto old=cache->acquire(recycling,7,key,provider);
            exl3_fill_device_page(*old.producer,nullptr,reinterpret_cast<cudaEvent_t>(std::uintptr_t{405}),
                complete_page,7,5,transfers);
            std::weak_ptr<const Exl3DevicePageFill::View> weak_view=old.shared->ready_handle();
            std::weak_ptr<const void> weak_storage=old.shared->storage_owner_for_retirement();
            const auto view_control=bounded_shared_allocation_bytes<Exl3DevicePageFill::View>();
            const auto storage_control=bounded_shared_allocation_bytes<Exl3DevicePageStorage>();
            const auto occupied=recycling.retained.totals();
            if(!cache->retire_lookup(recycling,7,key) || !cache->seal_retired(recycling,7,key) ||
                !cache->reclaim_retired(recycling,7,key))throw std::runtime_error("cache retirement sequence failed");
            const auto tracking=recycling.retained.totals();
            const auto metadata=static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata);
            if(tracking[0]!=0 || tracking[metadata]+Exl3DevicePageStorage::metadata_bytes()!=occupied[metadata] ||
                recycling.retired_metadata.bytes()!=Exl3DevicePageFill::retired_fill_metadata_bytes()+view_control+storage_control ||
                !weak_view.expired() || !weak_storage.expired())
                throw std::runtime_error("live retired fill lost metadata charge");
            const auto retired_source=cache->accounting();
            if(retired_source.device_bytes!=0 || retired_source.retained_source_pages!=1 ||
                retired_source.retained_source_host_bytes!=expected_source_bytes)
                throw std::runtime_error("retired fill handle lost host source charge with device retirement");
            std::weak_ptr<const Exl3DevicePageFill> weak=old.shared;old={};
            if(!weak.expired())throw std::runtime_error("tracking registry retained old fill/source owner");
            const auto released_source=cache->accounting();
            if(released_source.retained_source_pages!=0 || released_source.retained_source_host_bytes!=0 ||
                released_source.retained_metadata_records!=1)
                throw std::runtime_error("unrelated source key prolonged retired fill host charge");
            const auto before_stale_allocations=PreparedDevicePageStorageProvider::allocations;
            const auto before_stale_credits=recycling.retired_metadata.bytes();
            recycling.expire_after_checks=recycling.lease_checks+1;
            bool stale_slot_refused=false;
            try{(void)cache->acquire(recycling,7,key,provider);}
            catch(const std::invalid_argument&){stale_slot_refused=true;}
            recycling.expire_after_checks=0;
            if(!stale_slot_refused || cache->accounting().retained_metadata_records!=1 ||
                recycling.retired_metadata.bytes()!=before_stale_credits ||
                PreparedDevicePageStorageProvider::allocations!=before_stale_allocations ||
                recycling.retained.totals()!=tracking)
                throw std::runtime_error("stale lookup recycled retired slot or changed surviving credit");
            auto replacement=cache->acquire(recycling,7,key,provider);
            if(!replacement.producer || recycling.retained.totals()!=occupied)
                throw std::runtime_error("replacement leaked retired metadata or failed fresh physical credit");
            if(recycling.retired_metadata.bytes()!=bounded_shared_allocation_bytes<Exl3DevicePageFill>()+view_control+storage_control)
                throw std::runtime_error("lookup replacement released externally held weak fill control credit");
            weak.reset();
            if(recycling.retired_metadata.bytes()!=view_control+storage_control)
                throw std::runtime_error("final weak fill released independent view/storage controls");
            weak_view.reset();
            if(recycling.retired_metadata.bytes()!=storage_control)
                throw std::runtime_error("final weak view released independent storage control");
            weak_storage.reset();
            if(recycling.retired_metadata.bytes())throw std::runtime_error("final weak fill release retained detached control charge");
            if(cache->accounting().retained_source_host_bytes!=expected_source_bytes)
                throw std::runtime_error("replacement omitted new source backing charge");
            for(unsigned cycle=0;cycle<3;++cycle) {
                if(cache->evict_one(recycling,7))throw std::runtime_error("victim selection reclaimed unfinished fill");
                exl3_fill_device_page(*replacement.producer,nullptr,
                    reinterpret_cast<cudaEvent_t>(std::uintptr_t{406}),complete_page,7,6+cycle,transfers);
                auto use=consumer->begin(replacement.shared->ready_handle(),key,cycle%16,64,complete_page,
                    7,20+cycle,reinterpret_cast<cudaEvent_t>(std::uintptr_t{408}));
                const auto frees=PreparedDevicePageStorageProvider::releases;
                if(cache->evict_one(recycling,7) || PreparedDevicePageStorageProvider::releases!=frees)
                    throw std::runtime_error("busy-reader victim selection freed physical page");
                consumer->complete(use.generation,nullptr,transfers);
                if(!cache->evict_one(recycling,7) || PreparedDevicePageStorageProvider::releases!=frees+1)
                    throw std::runtime_error("completed victim did not reclaim one physical page");
                replacement={};replacement=cache->acquire(recycling,7,key,provider);
                if(!replacement.producer || recycling.retained.totals()!=occupied)
                    throw std::runtime_error("repeated victim replacement accumulated metadata or device credit");
            }
        }
    }
    {
        auto backing=std::make_shared<Exl3ExactKVPage>();backing->first=128;backing->rows=64;
        for(int bank=0;bank<16;++bank) {
            backing->k[bank].resize(64*1024,static_cast<std::uint16_t>(bank+1));
            backing->v[bank].resize(64*1024,static_cast<std::uint16_t>(bank+17));
        }
        Exl3DevicePageKey key(std::make_shared<int>(6),backing,{0,false},192);
        PreparedKVRegistrationAuthority authority;
        auto prepared=Exl3DevicePageStorage::create_runtime(authority,7,key,
            {PreparedDevicePageStorageProvider::allocate,PreparedDevicePageStorageProvider::release});
        const Exl3KVUploadProvider io{PreparedKVUploadProvider::copy,PreparedKVUploadProvider::record,PreparedKVUploadProvider::wait};
        const auto event=reinterpret_cast<cudaEvent_t>(std::uintptr_t{409});
        auto registrations=Exl3KVRegistrationCache::create_startup(authority);
        auto registration_read=Exl3KVRegistrationCache::acquire_external_read(registrations);
        if(!registration_read)throw std::runtime_error("page fill read guard unavailable");
        {
            auto survivor=std::move(*registration_read);
            const auto copies=PreparedKVUploadProvider::copies,records=PreparedKVUploadProvider::records,
                waits=PreparedKVUploadProvider::waits;
            bool refused=false;
            try{exl3_fill_device_page(prepared,nullptr,event,backing,7,11,io,std::move(registration_read));}
            catch(const std::invalid_argument&){refused=true;}
            if(!refused || prepared.fill->failed() || prepared.fill->fill_in_flight() || prepared.fill->ready() ||
                PreparedKVUploadProvider::copies!=copies || PreparedKVUploadProvider::records!=records ||
                PreparedKVUploadProvider::waits!=waits || registrations->external_readers()!=1 || !survivor.valid())
                throw std::runtime_error("moved page-fill reader mutated readiness or submitted transfer");
            registration_read.reset();registration_read.emplace(std::move(survivor));
        }
        exl3_fill_device_page(prepared,nullptr,event,backing,7,11,io,std::move(registration_read));
        if(registrations->external_readers()!=0)
            throw std::runtime_error("completed page fill retained registration guard");
        auto attention=Exl3DevicePageAttention::create_startup(authority);
        bool unpublished=false;
        try{attention->begin(prepared.fill->ready_handle(),key,15,191,backing,7,12,event);}
        catch(const std::invalid_argument&){unpublished=true;}
        if(!unpublished || prepared.fill->retained_readers()!=0)
            throw std::runtime_error("noninitial attention admitted unpublished page end");
        const auto prefix=attention->begin(prepared.fill->ready_handle(),key,15,192,backing,7,12,event);
        if(prefix.first!=128 || prefix.rows!=64 || prefix.k[0]!=16 || prefix.v[0]!=32 ||
            prepared.fill->retained_readers()!=1)
            throw std::runtime_error("noninitial attention lost range or bank identity");
        attention->complete(prefix.generation,nullptr,io);
        if(prepared.fill->retained_readers()!=0 || !prepared.fill->seal_for_retirement())
            throw std::runtime_error("noninitial attention did not retire reader");
    }
    if(const auto* selected=std::getenv("NINFER_TEST_DEVICE_PAGE_FILL_FAILURE");selected && std::string(selected)=="1") {
        const auto quarantined=Exl3DevicePageFill::quarantined_records();
        const auto released=PreparedDevicePageStorageProvider::releases;
        std::weak_ptr<int> event_owner;
        std::weak_ptr<Exl3KVRegistrationCache> retained_registrations;
        {
            auto backing=std::make_shared<Exl3ExactKVPage>();backing->rows=64;
            for(int bank=0;bank<16;++bank){backing->k[bank].resize(64*1024);backing->v[bank].resize(64*1024);}
            Exl3DevicePageKey key(std::make_shared<int>(9),backing,{0,false},64);
            PreparedKVRegistrationAuthority authority;
            auto cache=Exl3DevicePageCache::create_startup(authority);
            auto claim=cache->acquire(authority,7,key,
                {PreparedDevicePageStorageProvider::allocate,PreparedDevicePageStorageProvider::release});
            auto event=std::make_shared<int>(1);event_owner=event;
            auto registrations=Exl3KVRegistrationCache::create_startup(authority);
            retained_registrations=registrations;
            auto read=Exl3KVRegistrationCache::acquire_external_read(registrations);
            if(!read)throw std::runtime_error("failed page fixture missing registration guard");
            const auto generation=claim.producer->fill->begin(7,1,501,event,std::move(read));
            if(!claim.producer->fill->finish(generation,-1))
                throw std::runtime_error("prepared submitted page failure was not recorded");
            event.reset();claim={};
            if(cache->evict_one(authority,7) || !cache->retire_lookup(authority,7,key) ||
                cache->seal_retired(authority,7,key) || cache->reclaim_retired(authority,7,key) ||
                PreparedDevicePageStorageProvider::releases!=released || event_owner.expired())
                throw std::runtime_error("submitted page failure entered unsubmitted retirement path");
            const auto state=cache->accounting();
            if(state.failed_allocations!=1 || state.device_bytes!=Exl3DevicePageStorage::bytes)
                throw std::runtime_error("submitted page failure lost physical accounting");
            if(registrations->external_readers()!=1)
                throw std::runtime_error("failed page fill released registration guard");
        }
        if(Exl3DevicePageFill::quarantined_records()!=quarantined+1 || event_owner.expired() ||
            PreparedDevicePageStorageProvider::releases!=released)
            throw std::runtime_error("submitted page failure lost quarantine ownership");
        const auto registrations=retained_registrations.lock();
        if(!registrations || registrations->external_readers()!=1)
            throw std::runtime_error("page quarantine dropped registration cache ownership");
        return; // Terminal fixture: the intentionally uncertain owner remains quarantined.
    }
    if(const auto* selected=std::getenv("NINFER_TEST_DEVICE_PAGE_ATTENTION_FAILURE");selected && *selected) {
        using Attention=Exl3DevicePageAttention;using IO=PreparedKVUploadProvider;
        const std::string mode(selected);
        if(mode!="record" && mode!="wait")throw std::invalid_argument("attention failure requires record/wait");
        const auto before=Exl3KVTransferLease::quarantined_records();
        std::weak_ptr<const Exl3DevicePageFill::View> retained_view;
        std::weak_ptr<int> retained_event;
        RetainedDescriptorLedger attention_metadata;
        std::weak_ptr<Attention> weak_attention,weak_peer;
        {
            auto backing=std::make_shared<Exl3ExactKVPage>();backing->rows=64;
            for(int bank=0;bank<16;++bank){backing->k[bank].resize(64*1024);backing->v[bank].resize(64*1024);}
            Exl3DevicePageKey key(std::make_shared<int>(4),backing,{0,false},64);
            PreparedKVRegistrationAuthority authority;
            auto prepared=Exl3DevicePageStorage::create_runtime(authority,7,key,
                {PreparedDevicePageStorageProvider::allocate,PreparedDevicePageStorageProvider::release});
            const Exl3KVUploadProvider io{IO::copy,IO::record,IO::wait};
            const auto event=reinterpret_cast<cudaEvent_t>(std::uintptr_t{407});
            exl3_fill_device_page(prepared,nullptr,event,backing,7,8,io);
            auto attention=Attention::create_startup(authority);
            auto peer=Attention::create_startup(authority);
            weak_attention=attention;weak_peer=peer;
            for(const auto& owner:{attention,peer}) {
                if(Attention::attach_retirement_credit(owner,attention_metadata.acquire(Attention::metadata_bytes()-1)) ||
                   !Attention::attach_retirement_credit(owner,attention_metadata.acquire(Attention::metadata_bytes())) ||
                   Attention::attach_retirement_credit(owner,attention_metadata.acquire(Attention::metadata_bytes())))
                    throw std::runtime_error("attention metadata extent/duplicate retirement contract");
            }
            auto event_owner=std::make_shared<int>(5);retained_event=event_owner;
            auto view=prepared.fill->ready_handle();retained_view=view;
            auto prefix=attention->begin(std::move(view),key,0,64,event_owner,7,9,event);
            auto second=peer->begin(prepared.fill->ready_handle(),key,0,64,event_owner,7,9,event);
            std::array<Attention::Completion,2> group{{{attention.get(),prefix.generation},{peer.get(),second.generation}}};
            event_owner.reset();
            if(prepared.fill->seal_for_retirement())throw std::runtime_error("active attention allowed retirement");
            IO::record_error=mode=="record"?cudaErrorUnknown:cudaSuccess;
            IO::wait_error=mode=="wait"?cudaErrorUnknown:cudaSuccess;
            const auto records=IO::records,waits=IO::waits,copies=IO::copies;
            bool failed=false;
            try{Attention::complete_group(group,nullptr,io);}catch(const std::runtime_error&){failed=true;}
            const auto expected=mode=="record"?Attention::Stage::record:Attention::Stage::wait;
            if(!failed || !attention->uncertain() || !peer->uncertain() || peer->failure().stage!=expected ||
                peer->failure().error!=static_cast<int>(cudaErrorUnknown) || attention->failure().stage!=expected ||
                attention->failure().error!=static_cast<int>(cudaErrorUnknown) || IO::records!=records+1 ||
                IO::waits!=waits+(mode=="wait"?1U:0U) || IO::copies!=copies)
                throw std::runtime_error("attention failure lost stage or continued submission");
            IO::record_error=IO::wait_error=cudaSuccess;
            bool replay=false,reuse=false;
            try{attention->complete(prefix.generation,nullptr,io);}catch(const std::logic_error&){replay=true;}
            try{attention->begin(prepared.fill->ready_handle(),key,0,64,backing,7,10,event);}
            catch(const std::logic_error&){reuse=true;}
            bool peer_replay=false;
            try{peer->complete(second.generation,nullptr,io);}catch(const std::logic_error&){peer_replay=true;}
            if(!replay || !reuse || !peer_replay || prepared.fill->seal_for_retirement() || prepared.fill->retained_readers()!=2 ||
                IO::records!=records+1 || IO::waits!=waits+(mode=="wait"?1U:0U))
                throw std::runtime_error("failed attention retried or released final-use ownership");
        }
        if(retained_view.expired() || retained_event.expired() || Exl3KVTransferLease::quarantined_records()!=before+2)
            throw std::runtime_error("attention teardown dropped unresolved page/event owners");
        if(!weak_attention.expired() || !weak_peer.expired() ||
            attention_metadata.bytes()!=2*Attention::metadata_bytes())
            throw std::runtime_error("attention quarantine lost state or weak control charge");
        weak_attention.reset();weak_peer.reset();
        if(attention_metadata.bytes()!=2*Exl3KVTransferLease::state_metadata_bytes())
            throw std::runtime_error("attention final weak release discarded quarantined state credit");
        return; // deliberate quarantine: one stage per future process
    }
    if(const auto* selected=std::getenv("NINFER_TEST_DEVICE_PAGE_FREE_FAILURE");selected && std::string(selected)=="1") {
        using Storage=Exl3DevicePageStorage;using IO=PreparedDevicePageStorageProvider;
        const auto before=Storage::quarantined_bytes(),calls=std::uint64_t(IO::releases);
        {
            auto backing=std::make_shared<Exl3ExactKVPage>();backing->rows=64;
            for(int bank=0;bank<16;++bank){backing->k[bank].resize(64*1024);backing->v[bank].resize(64*1024);}
            Exl3DevicePageKey key(std::make_shared<int>(3),backing,{0,false},64);
            PreparedKVRegistrationAuthority authority;
            auto prepared=Storage::create_runtime(authority,7,key,{IO::allocate,IO::release});
            exl3_fill_device_page(prepared,nullptr,reinterpret_cast<cudaEvent_t>(std::uintptr_t{404}),
                backing,7,4,{PreparedKVUploadProvider::copy,PreparedKVUploadProvider::record,PreparedKVUploadProvider::wait});
            if(!authority.retire_runtime_metadata_to_lifetime(7,Storage::metadata_allocation(prepared.fill)))
                throw std::runtime_error("failed-free fixture did not transfer committed metadata");
            if(!prepared.fill->seal_for_retirement())throw std::runtime_error("failed-free fixture did not seal");
            const auto allocation=Storage::device_allocation(prepared.fill);
            const auto retained=authority.retained.totals();IO::release_error=cudaErrorUnknown;
            const auto retire=[&](const auto&) noexcept {return prepared.fill->retire_sealed_storage();};
            if(authority.retire_runtime_resource(7,allocation,retire) || authority.retained.totals()!=retained ||
                Storage::quarantined_bytes()!=before+Storage::bytes || IO::releases!=calls+1)
                throw std::runtime_error("failed free released device credit or lost quarantine charge");
            IO::release_error=cudaSuccess;
            if(authority.retire_runtime_resource(7,allocation,retire) || IO::releases!=calls+1 ||
                prepared.fill->ready_handle())throw std::runtime_error("failed free retried or reopened page readers");
            RetainedDeviceLedger retired_device;
            auto next=authority.retained.without_allocation(allocation);
            if(!allocation.device_lifetime_retire(allocation.owner,retired_device.acquire(allocation.units)))
                throw std::runtime_error("failed-free committed device lifetime hook refused");
            authority.retained=std::move(next);
        }
        if(IO::releases!=calls+1 || Storage::quarantined_bytes()!=before+Storage::bytes)
            throw std::runtime_error("failed-free destructor retried or double charged quarantine");
        const auto credits=Storage::retained_constructor_credits_for_test();
        if(credits[0]!=Storage::bytes || credits[1]!=Storage::retirement_metadata_bytes())
            throw std::runtime_error("failed committed free lost exact state/device survivor tickets");
        return; // deliberate retained allocation: separate future process
    }
    if(const auto* selected=std::getenv("NINFER_TEST_KV_UPLOAD_FAILURE");selected && *selected) {
        using Upload=Exl3RegisteredKVUpload;using IO=PreparedKVUploadProvider;
        const std::string mode(selected);
        if(mode=="overlap_wait") {
            IO::copy_error=IO::record_error=IO::wait_error=cudaSuccess;
            const auto before=Exl3KVTransferLease::quarantined_records();
            std::weak_ptr<std::vector<std::uint16_t>> failed_output,healthy_output;
            std::weak_ptr<Exl3RegisteredKVBacking> shared_registration;
            {
                PreparedKVRegistrationAuthority credit;
                auto failed=Upload::create_startup(credit),healthy=Upload::create_startup(credit);
                auto registration=Exl3RegisteredKVBacking::acquire_startup(credited,stable,callbacks);
                shared_registration=registration;
                auto a=std::make_shared<std::vector<std::uint16_t>>(64*1024),b=std::make_shared<std::vector<std::uint16_t>>(64*1024);
                failed_output=a;healthy_output=b;
                const Exl3KVUploadProvider io{IO::copy,IO::record,IO::wait};
                const auto ga=failed->submit(stable,a->data(),64,nullptr,reinterpret_cast<cudaEvent_t>(std::uintptr_t{201}),a,registration,7,1,io);
                const auto gb=healthy->submit(stable,b->data(),64,nullptr,reinterpret_cast<cudaEvent_t>(std::uintptr_t{202}),b,registration,8,1,io);
                if(*a!=page->k[0] || *b!=page->k[0])
                    throw std::runtime_error("overlapping registered readers changed represented rows");
                const auto waits_before_stale=IO::waits;
                bool stale_refused=false;try{healthy->complete(gb+1);}catch(const std::logic_error&){stale_refused=true;}
                if(!stale_refused || IO::waits!=waits_before_stale || !healthy->uncertain() || !failed->uncertain())
                    throw std::runtime_error("stale overlap completion reached provider or changed peer");
                a.reset();b.reset();
                if(failed_output.expired() || healthy_output.expired())throw std::runtime_error("overlap lost pending outputs");
                IO::wait_error=cudaErrorUnknown;
                bool refused=false;try{failed->complete(ga);}catch(const std::runtime_error&){refused=true;}
                if(!refused || !failed->uncertain() || !healthy->uncertain() || healthy_output.expired())
                    throw std::runtime_error("failed wait changed pending peer");
                IO::wait_error=cudaSuccess;healthy->complete(gb);
                if(!healthy_output.expired() || failed_output.expired() || !failed->uncertain())
                    throw std::runtime_error("peer completion changed failed final-use ownership");
                registration.reset();credited.retained={};
                if(shared_registration.expired())
                    throw std::runtime_error("failed reader lost registration after healthy peer completion");
            }
            if(failed_output.expired() || !healthy_output.expired() || shared_registration.expired() ||
                Exl3KVTransferLease::quarantined_records()!=before+1)
                throw std::runtime_error("overlapping failure quarantined wrong number of slots");
            return; // terminal process fixture; no later reuse of deliberate quarantine
        }
        if(mode!="copy" && mode!="record" && mode!="wait")
            throw std::invalid_argument("KV upload failure fixture requires copy/record/wait/overlap_wait");
        IO::copies=IO::records=IO::waits=0;
        IO::copy_error=mode=="copy"?cudaErrorUnknown:cudaSuccess;
        IO::record_error=mode=="record"?cudaErrorUnknown:cudaSuccess;
        IO::wait_error=mode=="wait"?cudaErrorUnknown:cudaSuccess;
        const auto expected=mode=="copy"?Upload::Stage::copy:
            mode=="record"?Upload::Stage::record:Upload::Stage::wait;
        const auto before=Exl3KVTransferLease::quarantined_records();
        std::weak_ptr<std::vector<std::uint16_t>> survivor;
        std::weak_ptr<Exl3RegisteredKVBacking> registration_survivor;
        RetainedDescriptorLedger retired_upload_credit;
        std::weak_ptr<Upload> retired_upload_wrapper;
        {
            PreparedKVRegistrationAuthority upload_credit;
            auto uploader=Upload::create_startup(upload_credit);
            retired_upload_wrapper=uploader;
            if(!Upload::attach_retirement_credit(uploader,retired_upload_credit.acquire(Upload::metadata_bytes())))
                throw std::runtime_error("failed upload fixture could not attach full retirement credit");
            auto owner=Exl3RegisteredKVBacking::acquire_startup(credited,stable,callbacks);
            registration_survivor=owner;
            auto output=std::make_shared<std::vector<std::uint16_t>>(64*1024);survivor=output;
            const Exl3KVUploadProvider io{IO::copy,IO::record,IO::wait};
            const auto event=reinterpret_cast<cudaEvent_t>(std::uintptr_t{125});
            bool failed=false;
            try{uploader->copy(stable,output->data(),64,nullptr,event,output,owner,7,9,io);}
            catch(const std::runtime_error&){failed=true;}
            if(!failed || !uploader->uncertain() || uploader->failure().stage!=expected ||
                uploader->failure().error!=static_cast<int>(cudaErrorUnknown) || IO::copies!=1 ||
                IO::records!=(mode=="copy"?0U:1U) || IO::waits!=(mode=="wait"?1U:0U))
                throw std::runtime_error("registered upload failure lost stage or continued submission");
            bool reuse=false;
            try{uploader->copy(stable,output->data(),64,nullptr,event,output,owner,7,10,io);}
            catch(const std::logic_error&){reuse=true;}
            if(!reuse || IO::copies!=1 || uploader->failure().stage!=expected)
                throw std::runtime_error("uncertain registered upload allowed reuse");
            // Another precredited slot may finish independently; it must not
            // clear the failed slot's witness or retain its own completed output.
            auto peer=Upload::create_startup(upload_credit);
            auto peer_output=std::make_shared<std::vector<std::uint16_t>>(64*1024);
            std::weak_ptr<std::vector<std::uint16_t>> peer_lifetime=peer_output;
            IO::copy_error=IO::record_error=IO::wait_error=cudaSuccess;
            const auto copies_before=IO::copies,records_before=IO::records,waits_before=IO::waits;
            peer->copy(stable,peer_output->data(),64,nullptr,
                reinterpret_cast<cudaEvent_t>(std::uintptr_t{126}),peer_output,owner,8,1,io);
            if(*peer_output!=page->k[0])throw std::runtime_error("surviving upload slot changed source rows");
            peer_output.reset();
            if(!peer_lifetime.expired() || peer->uncertain() || !uploader->uncertain() ||
                uploader->failure().stage!=expected || IO::copies!=copies_before+1 ||
                IO::records!=records_before+1 || IO::waits!=waits_before+1)
                throw std::runtime_error("successful peer altered failed upload ownership or final-use state");
            reuse=false;
            try{uploader->copy(stable,output->data(),64,nullptr,event,output,owner,7,11,io);}
            catch(const std::logic_error&){reuse=true;}
            if(!reuse || IO::copies!=copies_before+1)
                throw std::runtime_error("healthy provider revived failed transfer slot");
            owner.reset();output.reset();credited.retained={};
            if(survivor.expired() || registration_survivor.expired())
                throw std::runtime_error("uncertain upload dropped retained owners");
        }
        if(survivor.expired() || registration_survivor.expired() ||
            Exl3KVTransferLease::quarantined_records()!=before+1)
            throw std::runtime_error("failed upload teardown did not quarantine complete record");
        if(!retired_upload_wrapper.expired() || retired_upload_credit.bytes()!=Upload::metadata_bytes())
            throw std::runtime_error("failed upload released quarantined state or weak wrapper credit early");
        retired_upload_wrapper.reset();
        if(retired_upload_credit.bytes()!=Exl3KVTransferLease::state_metadata_bytes())
            throw std::runtime_error("failed upload did not conserve isolated quarantined state credit");
        const auto pinned=registration_survivor.lock();
        if(!pinned || pinned->active_readers()!=1 || pinned->seal_idle())
            throw std::runtime_error("quarantined transfer lost registration reader pin");
        return; // deliberate quarantine: one failure stage per future process
    }
    if(const auto* fail=std::getenv("NINFER_TEST_KV_REGISTRATION_TEARDOWN_FAILURE");fail &&
        (std::string(fail)=="1" || std::string(fail)=="sealed")) {
        const auto before=Exl3RegisteredKVBacking::quarantined_bytes();
        std::weak_ptr<const Exl3ExactKVPage> survivor;
        RetainedDescriptorLedger retired_registration;
        RetainedCudaRegistrationLedger retired_registration_bytes;
        std::weak_ptr<Exl3RegisteredKVBacking> wrapper;
        const auto releases=Provider::releases;
        {
            auto failed_page=std::make_shared<Exl3ExactKVPage>(*page);survivor=failed_page;
            auto failed_extent=Exl3ExactKVExtent::view(failed_page,0,Exl3ExactKVExtent::Plane::key,64);
            auto owner=Exl3RegisteredKVBacking::acquire_startup(credited,failed_extent,callbacks);
            wrapper=owner;
            if(!Exl3RegisteredKVBacking::attach_registration_credit(owner,
                retired_registration_bytes.acquire(failed_extent.backing().bytes)))
                throw std::runtime_error("unregister failure fixture missing registration byte charge");
            if(!Exl3RegisteredKVBacking::attach_retirement_credit(owner,
                retired_registration.acquire(Exl3RegisteredKVBacking::metadata_bytes())))
                throw std::runtime_error("unregister failure fixture missing complete metadata charge");
            Provider::release_error=cudaErrorUnknown;
            if(std::string(fail)=="sealed" && (!owner->seal_idle() || owner->retire_sealed() ||
                owner->retire_sealed() || Provider::releases!=releases+1))
                throw std::runtime_error("explicit registration retirement failed to latch its first release error");
            owner.reset();credited.retained={};
        }
        if(survivor.expired() || Exl3RegisteredKVBacking::quarantined_bytes()!=before+64ULL*1024*2)
            throw std::runtime_error("failed unregistration lost backing or charge");
        if(Provider::releases!=releases+1)
            throw std::runtime_error("registration destruction retried failed explicit retirement");
        if(!wrapper.expired() || retired_registration.bytes()!=Exl3RegisteredKVBacking::metadata_bytes())
            throw std::runtime_error("unregister failure released child or weak metadata credit early");
        wrapper.reset();
        if(retired_registration_bytes.bytes()!=64ULL*1024*2)
            throw std::runtime_error("failed unregister lost registration charge after final weak release");
        if(retired_registration.bytes()!=Exl3RegisteredKVBacking::state_metadata_bytes())
            throw std::runtime_error("unregister failure lost quarantined metadata credit");
        bool blocked=false;
        try{Exl3RegisteredKVBacking::acquire_startup(credited,stable,callbacks);}
        catch(const std::runtime_error&){blocked=true;}
        if(!blocked)throw std::runtime_error("quarantined registration allowed another acquisition");
    }
}
