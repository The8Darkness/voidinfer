#include "exl3/host_resident_set.h"
#include <cstring>
#include <iostream>
#include <cstdlib>

using Set=ninfer::exl3::Exl3HostResidentSet;
void require(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
struct Block {
    unsigned char* data=nullptr;std::size_t size;
    explicit Block(std::size_t bytes,bool commit=true):size(bytes) {
        data=static_cast<unsigned char*>(VirtualAlloc(nullptr,bytes,MEM_RESERVE|(commit?MEM_COMMIT:0),PAGE_READWRITE));
        require(data!=nullptr,"test allocation");if(commit) std::memset(data,37,bytes);
    }
    ~Block(){if(data) VirtualFree(data,0,MEM_RELEASE);}
};
struct Root {std::shared_ptr<Block> block;std::size_t first,bytes;};
struct DeviceLifetimeFixture {
    std::optional<ninfer::exl3::RetainedDeviceLedger::Ticket> ticket;
    unsigned attachments=0;
    static bool attach(const std::shared_ptr<const void>& owner,
        ninfer::exl3::RetainedDeviceLedger::Ticket credit) noexcept {
        auto* fixture=const_cast<DeviceLifetimeFixture*>(static_cast<const DeviceLifetimeFixture*>(owner.get()));
        if(!fixture || fixture->ticket || credit.bytes()!=40)return false;
        fixture->ticket.emplace(std::move(credit));++fixture->attachments;return true;
    }
};
Set::Snapshot snapshot(std::initializer_list<std::shared_ptr<Root>> roots) {
    Set::Snapshot result;
    for(const auto& root:roots) {result.owners.push_back(root);result.add(root->block->data+root->first,root->bytes);}
    return result;
}
std::pair<std::size_t,std::size_t> inspect(const Block& block,std::size_t first,std::size_t bytes) {
    std::size_t resident=0,locked=0;
    for(std::size_t offset=first;offset<first+bytes;offset+=4096) {
        PSAPI_WORKING_SET_EX_INFORMATION page{};page.VirtualAddress=block.data+offset;
        require(QueryWorkingSetEx(GetCurrentProcess(),&page,sizeof(page))!=0,"test query");
        if(page.VirtualAttributes.Valid) {resident+=4096;if(page.VirtualAttributes.Locked) locked+=4096;}
    }
    return {resident,locked};
}
int main() {
    try {
        if(const auto* mode=std::getenv("NINFER_TEST_FAILED_FACTORY_CEILING");mode && std::string(mode)=="1") {
            using Inventory=ninfer::exl3::Exl3ResourceInventory;
            require(!Set::failed_close_reservation_ceiling_for_test(),"failed ceiling fixture requires fresh process");
            Inventory::Requirement planning;planning.configuration=91;
            planning.add(Inventory::Domain::host_metadata,1,64);
            auto grown=planning;grown.add(Inventory::Domain::device,1,100);
            {
                Set registry(40ULL<<20,8ULL<<30);
                bool failed=false;
                try {
                    registry.allocate_reserved_growing(planning,[&](auto,auto&& extend)->Inventory {
                        extend(grown);
                        registry.seal_external_retirement_failure();
                        throw std::runtime_error("prepared uncertain factory cleanup");
                    });
                }catch(const std::runtime_error&){failed=true;}
                require(failed && registry.failed_reservation_ceiling_for_test()==grown.units,
                    "failed factory omitted grown authorized ceiling");
                bool refused=false;
                try{(void)registry.retained_resource_units();}catch(const std::logic_error&){refused=true;}
                require(refused,"conservative factory ceiling presented as exact retained units");
                bool called=false;refused=false;
                try{registry.allocate_reserved_growing(planning,[&](auto,auto&&)->Inventory {called=true;return {};});}
                catch(const std::logic_error&){refused=true;}
                require(refused && !called && registry.failed_reservation_ceiling_for_test()==grown.units,
                    "sealed retry entered factory or overwrote retained ceiling");
            }
            const auto retained=Set::failed_close_reservation_ceiling_for_test();
            require(retained && *retained==grown.units,"resident destruction lost failed factory ceiling");
            bool replacement_refused=false;
            try{Set replacement(40ULL<<20,8ULL<<30);}catch(const std::logic_error&){replacement_refused=true;}
            require(replacement_refused && Set::failed_close_reservation_ceiling_for_test()==retained,
                "replacement registry reused or changed failed reservation backing");
            std::cout<<"FAILED_FACTORY_CEILING_PREPARED_CASE_COMPLETE exact_retained_bytes=unknown\n";return 0;
        }
        SIZE_T original_min=0,original_max=0;DWORD original_flags=0;
        require(GetProcessWorkingSetSizeEx(GetCurrentProcess(),&original_min,&original_max,&original_flags)!=0,"original test quota");
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        {
        Set registry(40ULL<<20,8ULL<<30);
        auto limits=Inventory::unlimited();limits[0]=100;
        limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=128;
        auto pre_limits=registry.assess({});registry.set_resource_limits(limits);
        bool stale_limits=false;try{registry.commit(std::move(pre_limits));}
        catch(const std::logic_error&){stale_limits=true;}
        require(stale_limits,"assessment survived resource policy change");
        {
            Inventory::Requirement planning;planning.configuration=91;
            planning.add(Inventory::Domain::host_metadata,1,sizeof(int));
            for(unsigned fault:{0u,1u,2u}) {
                std::weak_ptr<int> temporary;bool refused=false;
                try{registry.allocate_reserved_growing(planning,[&](auto,auto&& extend) {
                    auto owner=std::make_shared<int>();temporary=owner;
                    auto next=planning;
                    if(fault==0)next.configuration=92;
                    else if(fault==1)next.units[static_cast<unsigned>(Inventory::Domain::host_metadata)]=0;
                    else next.add(Inventory::Domain::device,1,101);
                    extend(next);
                    return Inventory{};
                });}catch(const std::exception&){refused=true;}
                require(refused && temporary.expired(),"invalid extension retained planning owner");
            }
            auto allocated=registry.allocate_reserved_growing(planning,[&](auto configuration,auto&& extend) {
                require(configuration==91,"planning configuration changed");
                auto owner=std::make_shared<int>();
                auto next=planning;next.add(Inventory::Domain::device,1,100);
                extend(next);extend(next); // Equal promises are idempotent.
                for(unsigned invalid:{0u,1u,2u,3u}) {
                    auto rejected=next;
                    if(invalid==0)rejected.configuration=92;
                    else if(invalid==1)rejected.units[static_cast<unsigned>(Inventory::Domain::device)]=99;
                    else if(invalid==2)rejected.units[static_cast<unsigned>(Inventory::Domain::device)]=101;
                    else {
                        rejected.add(Inventory::Domain::cuda_registered_host,1,55);
                        rejected.units[static_cast<unsigned>(Inventory::Domain::host_metadata)]=sizeof(int)-1;
                    }
                    bool refused=false;
                    try{extend(rejected);}
                    catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=invalid==2;}
                    catch(const std::invalid_argument&){refused=invalid!=2;}
                    require(refused,"invalid extension did not report expected refusal");
                    // The failed extension must leave the prior promise usable.
                    extend(next);
                }
                const auto must_refuse=[&](auto&& mutation) {
                    bool refused=false;
                    try{mutation();}catch(const std::logic_error&){refused=true;}
                    require(refused,"nested mutation disturbed extended reservation");
                };
                must_refuse([&]{registry.assess({});});
                must_refuse([&]{registry.replace({});});
                must_refuse([&]{registry.close();});
                must_refuse([&]{registry.set_resource_limits(Inventory::unlimited());});
                bool nested_factory_called=false;
                must_refuse([&]{registry.allocate_reserved(planning,[&](auto) {
                    nested_factory_called=true;return Inventory{};
                });});
                must_refuse([&]{registry.allocate_reserved_growing(planning,[&](auto,auto&&) {
                    nested_factory_called=true;return Inventory{};
                });});
                require(!nested_factory_called,"nested reservation entered allocator");
                Inventory actual;
                actual.add({owner,0,Inventory::Domain::host_metadata,sizeof(int)});
                actual.add({owner,1,Inventory::Domain::device,100});
                return actual;
            });
            require(allocated.totals()[0]==100,"grown reservation did not commit exact device extent");
            allocated={};registry.replace({});
            for(unsigned fault:{0u,1u,2u}) {
                auto before=registry.assess({});
                std::weak_ptr<int> temporary;bool refused=false;
                try{registry.allocate_reserved_growing(planning,[&](auto,auto&& extend) -> Inventory {
                    auto owner=std::make_shared<int>();temporary=owner;
                    auto next=planning;next.add(Inventory::Domain::device,1,100);
                    extend(next);
                    if(fault==0)throw std::runtime_error("prepared post-extension allocation fault");
                    Inventory actual;
                    actual.add({owner,0,Inventory::Domain::host_metadata,sizeof(int)});
                    actual.add({owner,1,Inventory::Domain::device,fault==1?99ULL:101ULL});
                    return actual;
                });}catch(const std::exception&){refused=true;}
                require(refused && temporary.expired(),"post-extension failure retained temporary owner");
                auto failed_promise=planning;failed_promise.add(Inventory::Domain::device,1,100);
                require(registry.failed_reservation_ceiling_for_test()==failed_promise.units,
                    "factory failure lost grown authorized reservation ceiling");
                refused=false;try{registry.commit(std::move(before));}
                catch(const std::logic_error&){refused=true;}
                require(refused,"post-extension rollback revived stale assessment");
                std::weak_ptr<int> retry_owner;
                auto retry=registry.allocate_reserved_growing(planning,[&](auto,auto&& extend) {
                    auto owner=std::make_shared<int>();retry_owner=owner;
                    auto next=planning;next.add(Inventory::Domain::device,1,100);
                    extend(next);
                    Inventory actual;
                    actual.add({owner,0,Inventory::Domain::host_metadata,sizeof(int)});
                    actual.add({owner,1,Inventory::Domain::device,100});
                    return actual;
                });
                retry={};require(!retry_owner.expired(),"grown retry owner not retained by authority");
                require(registry.failed_reservation_ceiling_for_test()==Inventory::Totals{},
                    "successful factory retained obsolete failed reservation ceiling");
                registry.replace({});
                require(retry_owner.expired(),"grown retry owner survived retirement");
            }
        }
        {
            Inventory::Requirement requirement;requirement.configuration=7;
            requirement.add(Inventory::Domain::device,1,100);
            auto stale=registry.assess({});
            std::weak_ptr<int> owner;
            auto allocated=registry.allocate_reserved(requirement,[&](std::uint64_t configuration) {
                require(configuration==7,"reservation changed selected configuration");
                bool nested=false;try{registry.assess({});}catch(const std::logic_error&){nested=true;}
                require(nested,"assessment ignored promised allocation");
                auto allocation=std::make_shared<int>();owner=allocation;
                Inventory result;result.add({allocation,0,Inventory::Domain::device,100});return result;
            });
            bool refused=false;try{registry.commit(std::move(stale));}catch(const std::logic_error&){refused=true;}
            require(refused,"pre-reservation assessment survived");
            bool called=false;refused=false;
            try{registry.allocate_reserved(requirement,[&](auto){called=true;return Inventory{};});}
            catch(const std::runtime_error&){refused=true;}
            require(refused && !called,"over-budget allocator was invoked");
            allocated={};require(!owner.expired(),"allocated owner not bound to authority");
            registry.replace({});require(owner.expired(),"idle reserved owner not retired");
            for(unsigned fault=0;fault<4;++fault) {
                auto before=registry.assess({});std::weak_ptr<int> temporary;
                refused=false;
                try{registry.allocate_reserved(requirement,[&](auto) -> Inventory {
                    auto allocation=std::make_shared<int>();temporary=allocation;
                    if(fault==0)throw std::runtime_error("prepared allocation fault");
                    if(fault==3) {
                        Inventory invalid;
                        invalid.add({std::shared_ptr<const void>(std::shared_ptr<const void>{},allocation.get()),
                            0,Inventory::Domain::device,100});
                        return invalid;
                    }
                    Inventory result;result.add({allocation,0,fault==1?Inventory::Domain::host_metadata:Inventory::Domain::device,
                        fault==1?100ULL:101ULL});return result;
                });}catch(const std::exception&){refused=true;}
                require(refused && temporary.expired(),"failed allocation escaped rollback");
                refused=false;try{registry.commit(std::move(before));}catch(const std::logic_error&){refused=true;}
                require(refused,"rollback restored stale reservation revision");
                std::weak_ptr<int> retry_owner;
                auto retry=registry.allocate_reserved(requirement,[&](auto) {
                    auto value=std::make_shared<int>();retry_owner=value;
                    Inventory result;result.add({value,0,Inventory::Domain::device,100});return result;
                });
                retry={};
                require(!retry_owner.expired(),"rollback retry failed to bind full-capacity owner");
                registry.replace({});
                require(retry_owner.expired(),"rollback retry owner leaked after retirement");
            }
            auto actual=std::make_shared<int>();requirement.units[0]=40;
            auto factory=[&](auto){Inventory result;result.add({actual,0,Inventory::Domain::device,40});return result;};
            registry.allocate_reserved(requirement,factory);
            refused=false;try{registry.allocate_reserved(requirement,factory);}catch(const std::invalid_argument&){refused=true;}
            require(refused,"second credit consumed by existing owner");
            const Inventory::Allocation retiring{actual,0,Inventory::Domain::device,40};
            unsigned retirement_calls=0;
            auto wrong=retiring;++wrong.units;
            refused=false;
            try{registry.retire_reserved(wrong,[&](const auto&) noexcept {++retirement_calls;return true;});}
            catch(const std::invalid_argument&){refused=true;}
            require(refused && retirement_calls==0,"mismatched retirement invoked physical release");
            auto before_retirement=registry.assess({});
            require(!registry.retire_reserved(retiring,[&](const auto&) noexcept {++retirement_calls;return false;}),
                "failed physical retirement reported success");
            refused=false;try{registry.commit(std::move(before_retirement));}catch(const std::logic_error&){refused=true;}
            require(refused,"failed retirement left stale assessment usable");
            Inventory::Requirement incoming;incoming.configuration=8;incoming.add(Inventory::Domain::device,1,70);
            bool incoming_called=false;
            const auto incoming_factory=[&](auto) {
                incoming_called=true;Inventory result;
                result.add({std::make_shared<int>(),0,Inventory::Domain::device,70});return result;
            };
            refused=false;try{registry.allocate_reserved(incoming,incoming_factory);}catch(const std::runtime_error&){refused=true;}
            require(refused && !incoming_called,"failed retirement released device credit");
            require(registry.retire_reserved(retiring,[&](const auto&) noexcept {++retirement_calls;return true;}),
                "confirmed physical retirement did not release charge");
            refused=false;
            try{registry.retire_reserved(retiring,[&](const auto&) noexcept {++retirement_calls;return true;});}
            catch(const std::invalid_argument&){refused=true;}
            require(refused && retirement_calls==2,"retirement replay reached physical release");
            registry.allocate_reserved(incoming,incoming_factory);
            require(incoming_called,"confirmed retirement left device credit unavailable");
            registry.replace({});
        }
        {
            auto owner=std::make_shared<int>();
            const Inventory::Allocation device{owner,0,Inventory::Domain::device,40};
            Set::Snapshot initial;initial.resources.add(device);registry.replace(std::move(initial));
            std::optional<ninfer::exl3::RetainedDeviceLedger::Ticket> lifetime;
            require(!registry.retire_reserved_device_to_lifetime(device,[](auto) noexcept {return false;}),
                "refused device lifetime attachment released allocation");
            require(registry.retained_resource_units()[0]==40,"refused device lifetime changed charge");
            require(registry.retire_reserved_device_to_lifetime(device,[&](auto ticket) noexcept {
                lifetime.emplace(std::move(ticket));return true;
            }),"device lifetime attachment refused");
            owner.reset();
            const auto empty=registry.replace({});
            require(registry.retained_resource_units()[0]==40 && empty.resource_units[0]==40 &&
                empty.transition_peak_resource_units[0]==40,
                "detached device lifetime disappeared from totals or transition peak");
            Inventory::Requirement incoming;incoming.configuration=18;incoming.add(Inventory::Domain::device,1,70);
            bool called=false;
            const auto factory=[&](auto) {called=true;Inventory result;
                result.add({std::make_shared<int>(),0,Inventory::Domain::device,70});return result;};
            bool refused=false;
            try{registry.allocate_reserved(incoming,factory);}catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && !called,"detached device lifetime bypassed admission budget");
            lifetime.reset();
            require(registry.retained_resource_units()[0]==0,"released device lifetime retained charge");
            registry.allocate_reserved(incoming,factory);require(called,"released lifetime did not restore device capacity");
            registry.replace({});
        }
        {
            constexpr auto domain=Inventory::Domain::cuda_registered_host;
            constexpr auto index=static_cast<unsigned>(domain);
            auto owner=std::make_shared<int>();
            const Inventory::Allocation registration{owner,0,domain,40};
            Set::Snapshot initial;initial.resources.add(registration);registry.replace(std::move(initial));
            std::optional<ninfer::exl3::RetainedCudaRegistrationLedger::Ticket> lifetime;
            require(!registry.retire_reserved_registration_to_lifetime(registration,[](auto) noexcept {return false;}),
                "refused registration attachment released charge");
            require(registry.retained_resource_units()[index]==40,"refused registration transfer changed budget");
            require(registry.retire_reserved_registration_to_lifetime(registration,[&](auto ticket) noexcept {
                lifetime.emplace(std::move(ticket));return true;
            }),"registration lifetime transfer refused");
            const auto empty=registry.replace({});
            require(registry.retained_resource_units()[index]==40 && empty.resource_units[index]==40 &&
                empty.transition_peak_resource_units[index]==40,"detached registration omitted from retained/peak totals");
            Inventory::Requirement incoming;incoming.configuration=19;
            incoming.add(domain,1,std::numeric_limits<std::uint64_t>::max()-39);
            bool called=false;
            const auto factory=[&](auto){called=true;Inventory actual;
                actual.add({std::make_shared<int>(),0,domain,incoming.units[index]});return actual;};
            bool refused=false;try{registry.allocate_reserved(incoming,factory);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && !called,"detached registration bypassed admission budget");
            lifetime.reset();require(registry.retained_resource_units()[index]==0,"unregister credit release did not restore capacity");
            registry.allocate_reserved(incoming,factory);require(called,"released registration did not restore admission");
            registry.replace({});
        }
        {
            auto owner=std::make_shared<int>(1),tracker=std::make_shared<int>(2);
            std::shared_ptr<const void> borrower=owner;
            std::weak_ptr<const void> retired=owner,foreign;
            {auto unrelated=std::make_shared<int>(3);foreign=unrelated;}
            Inventory::Allocation payload{owner,0,Inventory::Domain::host_metadata,128};
            const Inventory::Allocation tracking{tracker,1,Inventory::Domain::host_metadata,128};
            Set::Snapshot initial;initial.resources.add(payload);registry.replace(std::move(initial));
            registry.transfer_retired_metadata(payload,tracking);
            {
                Set::Snapshot alias_return;alias_return.resources.add(payload);
                auto alias_transition=registry.assess(std::move(alias_return));
                require(alias_transition.peak_resource_units_for_test()[
                    static_cast<unsigned>(Inventory::Domain::host_metadata)]==128,
                    "resident transition double-counted live alias of pending retirement");
            }
            payload.owner.reset();owner.reset();
            const auto before=registry.revision();
            bool refused=false;try{registry.collect_retired_metadata(tracking,foreign);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused && registry.revision()==before,"foreign expired owner changed metadata authority");
            require(!registry.collect_retired_metadata(tracking,retired),"borrower did not retain metadata charge");
            Inventory::Requirement incoming;incoming.configuration=9;
            incoming.add(Inventory::Domain::host_metadata,1,1);
            bool allocated=false;
            const auto factory=[&](auto) {allocated=true;Inventory result;
                result.add({std::make_shared<int>(4),0,Inventory::Domain::host_metadata,1});return result;};
            refused=false;try{registry.allocate_reserved(incoming,factory);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && !allocated,"live retired metadata released allocation budget");
            borrower.reset();
            require(registry.collect_retired_metadata(tracking,retired),"matching expired metadata not collected");
            registry.allocate_reserved(incoming,factory);
            require(allocated,"collected metadata credit remained unavailable");
            refused=false;try{registry.collect_retired_metadata(tracking,retired);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused,"metadata collection replay succeeded");
            registry.replace({});
        }
        {
            auto physical=std::make_shared<int>();std::weak_ptr<int> retained=physical;
            Set::Snapshot initial;initial.resources.add({physical,0,Inventory::Domain::device,60});
            registry.replace(std::move(initial));physical.reset();
            require(!retained.expired(),"registry did not retain device owner");
            auto incoming=std::make_shared<int>();Set::Snapshot excessive_transition;
            excessive_transition.resources.add({incoming,0,Inventory::Domain::device,60});
            const auto revision=registry.revision();bool refused=false;
            try{registry.replace(std::move(excessive_transition));}catch(const std::runtime_error&){refused=true;}
            require(refused && registry.revision()==revision && !retained.expired(),
                "whole transition dropped old resource on budget failure");
            registry.replace({});require(retained.expired(),"retired resource not released");
        }
        }
        // The tight resource ceilings above belong to the retirement-credit
        // fixture. Page-residency assessments use a fresh registry so its
        // immutable policy includes ordinary bounded descriptor backing.
        Set registry(40ULL<<20,8ULL<<30);
        auto block=std::make_shared<Block>(4*4096);std::weak_ptr<Block> weak_block=block;
        auto left=std::make_shared<Root>(Root{block,7,2*4096-7});
        auto right=std::make_shared<Root>(Root{block,4096+19,2*4096-19});
        std::weak_ptr<Root> weak_left=left;
        auto first=registry.replace(snapshot({left,right}));
        require(first.page_bytes==3*4096 && first.new_locked_bytes==3*4096,"overlap rounded page union");
        {
            // T027: a caller may abandon a sealed assessment rather than retry
            // it. Its private descriptor promise must retire without changing
            // the resident revision or leaving capacity debt.
            const auto metadata=static_cast<unsigned>(Inventory::Domain::host_metadata);
            const auto before_revision=registry.revision();
            const auto before_bytes=registry.retained_resource_units()[metadata];
            {
                auto abandoned=registry.assess(snapshot({left,right}));
                require(abandoned.source_revision()==before_revision &&
                    registry.retained_resource_units()[metadata]>before_bytes,
                    "abandoned assessment did not retain its private descriptor charge");
            }
            require(registry.revision()==before_revision &&
                registry.retained_resource_units()[metadata]==before_bytes,
                "abandoned assessment changed revision or retained capacity debt");
        }
        {
            auto stale=registry.assess(snapshot({right}));
            auto incoming=snapshot({left,right});incoming.regions.reserve(8);
            const auto* storage=incoming.regions.data();const auto capacity=incoming.regions.capacity();
            auto plan=registry.assess(std::move(incoming));
            require(plan.range_storage_for_test()==storage && plan.range_capacity_for_test()==capacity &&
                stale.range_storage_for_test()!=plan.range_storage_for_test(),
                "assessment copied incoming ranges or shared mutable retained-plan storage");
            require(plan.peak_page_bytes()==3*4096,"assessment double counted aliases");
            const auto receipt=registry.commit(std::move(plan));
            require(receipt.committed_revision==receipt.source_revision+1,"transition revision");
            bool rejected=false;try {registry.commit(std::move(stale));}catch(const std::logic_error&){rejected=true;}
            require(rejected,"stale assessed plan committed");
            rejected=false;try {registry.commit(std::move(plan));}catch(const std::logic_error&){rejected=true;}
            require(rejected,"consumed plan committed twice");
        }
        require(EmptyWorkingSet(GetCurrentProcess())!=0,"trim own test process");
        require(inspect(*block,0,3*4096)==std::pair<std::size_t,std::size_t>{3*4096,3*4096},"locked overlap survived trim");
        left.reset();auto second=registry.replace(snapshot({right}));
        require(weak_left.expired() && second.unlocked_bytes==4096 && second.new_locked_bytes==0,"removed root lifetime");
        require(inspect(*block,4096,2*4096).second==2*4096 && inspect(*block,0,4096).second==0,"shared pages remained locked");

        auto big=std::make_shared<Block>(32ULL<<20);auto new_root=std::make_shared<Root>(Root{big,0,big->size});
        bool failed=false;
        try {registry.replace(snapshot({new_root}),[](auto n){if(n==1) throw std::runtime_error("injected after first lock");});}
        catch(const std::runtime_error& e) {failed=std::string(e.what())=="injected after first lock";}
        require(failed && registry.locked_page_bytes()==2*4096,"injected update rollback identity");
        {
            auto plan=registry.assess(snapshot({new_root}));const auto revision=registry.revision();
            require(plan.peak_page_bytes()==big->size+2*4096,"copy-before-release peak omitted old root");
            bool nested_refused=false;
            try {registry.commit(std::move(plan),[&](auto){
                try {registry.replace(snapshot({right}));}catch(const std::logic_error&){nested_refused=true;}
                throw std::runtime_error("cancel sealed transition");
            });}catch(const std::runtime_error&){}
            require(nested_refused && registry.revision()==revision && registry.locked_page_bytes()==2*4096,
                "cancel/reentrant transition changed authority");
        }
        require(inspect(*big,0,big->size).second==0 && inspect(*block,4096,2*4096).second==2*4096,"rollback lock set");

        auto bad=std::make_shared<Block>(4*4096,false);
        require(VirtualAlloc(bad->data,2*4096,MEM_COMMIT,PAGE_READWRITE)==bad->data,"partial commit fixture");
        auto bad_root=std::make_shared<Root>(Root{bad,0,bad->size});failed=false;
        try {registry.replace(snapshot({bad_root}));} catch(const std::runtime_error&) {failed=true;}
        require(failed && inspect(*bad,0,bad->size).second==0 && registry.locked_page_bytes()==2*4096,"OS failure rollback");

        auto excessive=std::make_shared<Block>(48ULL<<20);auto excessive_root=std::make_shared<Root>(Root{excessive,0,excessive->size});failed=false;
        try {registry.replace(snapshot({excessive_root}));} catch(const std::runtime_error& e) {
            failed=std::string(e.what()).find("page budget exhausted")!=std::string::npos;
        }
        require(failed && inspect(*excessive,0,excessive->size).second==0,"budget rejected before lock");
        require(VirtualLock(bad->data,4096)!=0,"foreign test lock");failed=false;
        auto foreign=std::make_shared<Root>(Root{bad,0,4096});
        try {registry.replace(snapshot({foreign}));} catch(const std::invalid_argument&) {failed=true;}
        require(failed && inspect(*bad,0,4096).second==4096,"foreign lock preserved");
        require(VirtualUnlock(bad->data,4096)!=0,"release foreign test lock");

        registry.replace(snapshot({new_root}));right.reset();block.reset();
        require(weak_block.expired(),"obsolete allocation released after unlocking");
        require(EmptyWorkingSet(GetCurrentProcess())!=0,"trim large own allocation");
        require(inspect(*big,0,big->size)==std::pair<std::size_t,std::size_t>{big->size,big->size},"large trim survived");
        auto device_lifetime=std::make_shared<DeviceLifetimeFixture>();
        bool wrong_device_domain=false;
        try{Inventory invalid;invalid.add({device_lifetime,0,Inventory::Domain::host_metadata,40,{},nullptr,
            &DeviceLifetimeFixture::attach});}catch(const std::invalid_argument&){wrong_device_domain=true;}
        require(wrong_device_domain && !device_lifetime->ticket && !device_lifetime->attachments,
            "device lifetime hook accepted host metadata domain");
        Inventory::Requirement device_required;device_required.configuration=20;
        device_required.add(Inventory::Domain::device,1,40);
        registry.allocate_reserved(device_required,[&](auto) {
            Inventory actual;actual.add({device_lifetime,0,Inventory::Domain::device,40,{},nullptr,&DeviceLifetimeFixture::attach});
            return actual;
        });
        registry.close();require(inspect(*big,0,big->size).second==0,"close unlock");
        require(device_lifetime->attachments==1 && device_lifetime->ticket &&
            registry.retained_resource_units()[0]==40,"close lost external device lifetime charge");
        registry.close();require(device_lifetime->attachments==1,"repeated close attached device credit again");
        device_lifetime->ticket.reset();
        require(registry.retained_resource_units()[0]==0,"released post-close device ticket remained charged");
        SIZE_T final_min=0,final_max=0;DWORD final_flags=0;
        require(GetProcessWorkingSetSizeEx(GetCurrentProcess(),&final_min,&final_max,&final_flags)!=0,"final test quota");
        require(final_min==original_min && final_max==original_max && final_flags==original_flags,"quota restoration");
        failed=false;
        try {Set refused(1ULL<<20,1ULL<<50);} catch(const std::runtime_error& e) {
            failed=std::string(e.what()).find("physical reserve exhausted")!=std::string::npos;
        }
        require(failed,"physical reserve refusal");
        std::cout << "HOST_RESIDENT_SET PASS overlap trim replacement lifetime injected_failure OS_failure foreign_lock budget reserve quota_restore\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << "HOST_RESIDENT_SET FAIL " << e.what() << '\n';return 1;}
}
