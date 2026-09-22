#pragma once
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/device_page_fill.h"
#include "test_exl3_host_residency.h"
#include <atomic>
#include <thread>

void run_t84_serving_coordinator(
    Exl3TextModel& target, const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,
    const std::filesystem::path& output) {
    using Request = ninfer::exl3::Exl3VeriCacheRequest;
    using Cache = ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity = ninfer::exl3::Exl3VeriCacheServingIdentity;
    using Coordinator = ninfer::exl3::Exl3VeriCacheServingCoordinator;
    constexpr std::size_t kPrompt = 4096;
    constexpr std::size_t kPrefix = 3968;
    constexpr std::size_t kTail = 128;
    constexpr int kOutputs = 8;
    require(unsigned(env("NINFER_TEST_RESIDENT_CLOSE_QUARANTINE")=="1") +
            unsigned(env("NINFER_TEST_RESIDENT_POISON_METADATA")=="1") +
            unsigned(env("NINFER_TEST_COORDINATOR_ROLLBACK_QUARANTINE")=="1" ||
                     env("NINFER_TEST_COORDINATOR_ROLLBACK_QUARANTINE")=="startup") +
            unsigned(env("NINFER_TEST_COORDINATOR_LATE_RELEASE")=="1") <= 1,
            "T84 terminal quarantine modes require separate processes");
    require(target.max_context() >= 4352 && code.size() >= kPrompt &&
                prose.size() >= 2 * kTail,
            "T84 fixture extent");
    require(!output.empty() && !std::filesystem::exists(output),
            "T84 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib = std::stoull(env("NINFER_T84_RESERVE_GIB"));
    const auto cache_mib = std::stoull(env("NINFER_T84_CACHE_MIB"));
    require(reserve_gib >= 8 && cache_mib >= 1024,
            "T84 memory policy extent");
    const std::uint64_t reserve = reserve_gib * (1ULL << 30);
    const std::uint64_t cache_budget = cache_mib * (1ULL << 20);
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) != 0 && memory.ullAvailPhys > reserve,
            "T84 physical memory reserve");
    const auto available = [] {
        MEMORYSTATUSEX state{};
        state.dwLength = sizeof(state);
        require(GlobalMemoryStatusEx(&state) != 0,
                "T84 physical memory query");
        return state.ullAvailPhys;
    };
    const Identity identity{
        env("NINFER_T84_NAMESPACE"), env("NINFER_T84_ARTIFACT_ID"),
        env("NINFER_T84_TOKENIZER_ID"), env("NINFER_T84_CONFIGURATION_ID"),
        env("NINFER_T84_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY", "0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    Cache cache(Cache::Policy{8, 64, cache_budget, reserve}, identity);
    auto context_a = target.create_context(true);
    auto context_b = target.create_context(true);
    context_a->prepare_continuation(8);
    context_b->prepare_continuation(8);
    Exl3TextContext* contexts[2]{context_a.get(), context_b.get()};
    TargetGraphC2Stream streams[2];

    std::vector<std::int64_t> common(code.begin(), code.begin() + kPrefix);
    std::array<std::vector<std::int64_t>,3> prompts{common, common, common};
    prompts[0].insert(prompts[0].end(), code.begin() + kPrefix,
                      code.begin() + kPrompt);
    prompts[1].insert(prompts[1].end(), prose.begin(), prose.begin() + kTail);
    prompts[2].insert(prompts[2].end(), prose.begin() + kTail,
                      prose.begin() + 2 * kTail);
    auto fill = cache.prepare(*context_a, common, kPrefix, available(), 1024);
    require(!fill.metrics.cache_hit && fill.metrics.admitted,
            "T84 cache fill");
    fill.request.reset();
    std::array<std::shared_ptr<const Request>,3> initial;
    for (int request = 0; request < 3; ++request) {
        auto prepared = cache.prepare(*contexts[request & 1], prompts[request],
                                      kPrefix, available(), 1024);
        require(prepared.metrics.cache_hit &&
                    prepared.metrics.reused_prompt_tokens == kPrefix &&
                    prepared.metrics.executed_prompt_tokens == kTail,
                "T84 cached request preparation");
        initial[request] = std::move(prepared.request);
    }

    {
        std::vector<std::shared_ptr<const Request>> roots;
        ninfer::exl3::Exl3HostResidentSet::Snapshot snapshot;
        snapshot.reserve_regions(2);
        const auto* range_backing=snapshot.regions.data();
        const auto range_capacity=snapshot.regions.capacity();
        snapshot.add(reinterpret_cast<const void*>(16),8);
        snapshot.add(reinterpret_cast<const void*>(32),8);
        snapshot.add(nullptr,0);
        bool range_refused=false;
        try{snapshot.add(reinterpret_cast<const void*>(48),8);}
        catch(const std::length_error&){range_refused=true;}
        require(range_refused && snapshot.regions.size()==2 && snapshot.regions.front().first==16 &&
            snapshot.regions.back().second==40 && snapshot.regions.data()==range_backing &&
            snapshot.regions.capacity()==range_capacity,
            "T024 snapshot range overflow changed planned descriptors or backing");
        bool replan_refused=false;
        try{snapshot.reserve_regions(3);}catch(const std::logic_error&){replan_refused=true;}
        require(replan_refused && snapshot.regions.size()==2,"T024 range replan changed active snapshot");
        auto descriptor_payload=std::make_shared<const int>(7);
        std::weak_ptr<const int> retained_payload=descriptor_payload;
        snapshot.owners.push_back(descriptor_payload);descriptor_payload.reset();
        const auto descriptor_bytes=ninfer::exl3::bounded_shared_allocation_bytes<
            ninfer::exl3::Exl3HostResidentSet::Snapshot::DescriptorBacking>()+
            snapshot.owners.capacity()*sizeof(std::shared_ptr<const void>)+
            snapshot.regions.capacity()*sizeof(ninfer::exl3::Exl3HostResidentSet::Range);
        snapshot.retain_descriptors();
        const auto domain=static_cast<unsigned>(ninfer::exl3::Exl3ResourceInventory::Domain::host_metadata);
        require(snapshot.resources.totals()[domain]==descriptor_bytes &&
            snapshot.descriptor_backing->regions.data()==range_backing && !retained_payload.expired(),
            "T024 descriptor backing charge lost vector allocation or payload ownership");
        auto retained_inventory=snapshot.resources;
        using Snapshot=ninfer::exl3::Exl3HostResidentSet::Snapshot;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        require(Snapshot::descriptor_requirement(snapshot.descriptor_backing->owners.capacity(),
            snapshot.descriptor_backing->regions.capacity()).units[domain]==descriptor_bytes,
            "T024 snapshot planned and retained descriptor accounting diverged");
        bool overflow_refused=false;
        try{(void)Snapshot::descriptor_requirement(std::numeric_limits<std::size_t>::max(),1);}
        catch(const std::overflow_error&){overflow_refused=true;}
        require(overflow_refused,"T024 descriptor requirement accepted owner capacity overflow");
        Snapshot replacement;
        replacement.reserve_regions(2);replacement.add(reinterpret_cast<const void*>(64),8);
        replacement.retain_descriptors();
        auto limits=Inventory::unlimited();limits[domain]=descriptor_bytes;
        require(Inventory::peak(snapshot.resources,retained_inventory,limits)[domain]==descriptor_bytes,
            "T024 descriptor backing alias was charged twice");
        const auto replacement_bytes=replacement.resources.totals()[domain];
        limits[domain]=descriptor_bytes+replacement_bytes-1;
        bool peak_refused=false;
        try{(void)Inventory::peak(snapshot.resources,replacement.resources,limits);}
        catch(const std::runtime_error& error){peak_refused=std::string_view(error.what())==
            "resource transition budget exhausted";}
        require(peak_refused && !retained_payload.expired(),
            "T024 old/new descriptor overlap discarded old ownership or bypassed peak credit");
        limits[domain]+=1;
        require(Inventory::peak(snapshot.resources,replacement.resources,limits)[domain]==limits[domain],
            "T024 exact-fit descriptor overlap was not accounted");
        snapshot={};
        require(!retained_payload.expired(),"T024 inventory did not retain snapshot descriptor payload");
        retained_inventory={};
        require(retained_payload.expired(),"T024 retired descriptor inventory retained payload owner");
        const std::array<std::shared_ptr<const Request>,2> invalid{initial[0],nullptr};
        std::size_t visited=0;bool invalid_refused=false;
        try{Request::visit_host_allocations(invalid,[&](const void*,std::size_t){++visited;},false);}
        catch(const std::invalid_argument& error){invalid_refused=std::string_view(error.what())==
            "request allocation null image";}
        require(invalid_refused && visited==0,"T024 invalid request batch emitted partial payload inventory");
        bool refused=false;
        try{cache.copy_roots_into(roots);}
        catch(const std::length_error& error){refused=std::string_view(error.what())==
            "prefix root snapshot exceeds reserved capacity";}
        require(refused && roots.empty() && roots.capacity()==0,
            "T024 root snapshot grew an unreserved destination");
        roots.reserve(cache.root_capacity());
        const auto capacity=roots.capacity();const auto* backing=roots.data();
        cache.copy_roots_into(roots);
        require(roots==cache.roots() && roots.capacity()==capacity && roots.data()==backing,
            "T024 root snapshot changed order/ownership or reserved backing");
        roots.reserve(cache.root_capacity()+1);
        roots.push_back(initial[0]);
        const auto reused_capacity=roots.capacity();backing=roots.data();
        cache.copy_roots_into(roots);
        require(roots==cache.roots() && roots.capacity()==reused_capacity && roots.data()==backing,
            "T024 root snapshot retained stale output or replaced backing on reuse");
    }
    {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        const auto required=Registry::Snapshot::descriptor_requirement(1,0);
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        auto limits=Inventory::unlimited();limits[domain]=required.units[domain];
        registry.set_resource_limits(limits);
        bool constructed=false,assessed=false,locked=false,refused=false;
        std::weak_ptr<const int> private_payload;
        try{registry.replace_reserved_snapshot(required,[&](auto) {
            constructed=true;Registry::Snapshot result;
            result.owners.reserve(1);result.reserve_regions(0);
            auto owner=std::make_shared<const int>(11);private_payload=owner;
            result.owners.push_back(std::move(owner));return result;
        },[&]{assessed=true;},[&](auto){locked=true;});}
        catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && constructed && assessed && !locked && private_payload.expired(),
            "T024 missing rollback scratch credit reached locks or retained private owner");
        registry.close();
    }
    {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        using Ledger=ninfer::exl3::RetainedDescriptorLedger;
        {
            using PayloadLedger=ninfer::exl3::RetainedHostAllocationLedger;
            struct Payload {
                PayloadLedger::Ticket ticket;
                alignas(4096) std::array<std::byte,4096> data{};
                explicit Payload(PayloadLedger::Ticket value):ticket(std::move(value)) {}
            };
            Registry payload_registry(4096,reserve);
            auto owner=std::make_shared<Payload>(payload_registry.reserve_host_payload_lifetime(4096));
            bool exhausted=false;
            try{(void)payload_registry.reserve_host_payload_lifetime(1);}
            catch(const ninfer::exl3::Exl3HostResidentBudgetExhausted&){exhausted=true;}
            require(exhausted,"unbound host payload promise did not exhaust page budget");
            const auto matches=[](const std::shared_ptr<const void>& value,const PayloadLedger& ledger,
                std::size_t plane,const void* address,std::size_t bytes) noexcept {
                const auto payload=std::static_pointer_cast<const Payload>(value);
                return plane==0 && ledger.owns(payload->ticket) && payload->data.data()==address && bytes==4096;
            };
            payload_registry.track_host_payload_lifetime(owner,0,owner->data.data(),4096,matches);
            Registry::Snapshot snapshot;snapshot.owners.push_back(owner);snapshot.add(owner->data.data(),4096);
            {
                auto plan=payload_registry.assess(std::move(snapshot));
                require(plan.peak_page_bytes()==4096,"resident admission double counted covered payload credit");
                owner.reset();
                exhausted=false;
                try{(void)payload_registry.reserve_host_payload_lifetime(1);}
                catch(const ninfer::exl3::Exl3HostResidentBudgetExhausted&){exhausted=true;}
                require(exhausted,"uncommitted transition lost retained payload credit");
            }
            {
                auto retry=payload_registry.reserve_host_payload_lifetime(4096);
                require(retry.bytes()==4096,"retired payload did not restore reservation headroom");
            }
            payload_registry.close();
        }
        {
            using Page=ninfer::exl3::Exl3ExactKVPage;
            Registry registry(memory.ullTotalPhys-reserve,reserve);
            const auto metadata_domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
            auto page_limits=Inventory::unlimited();page_limits[metadata_domain]=Page::metadata_bytes();
            registry.set_resource_limits(page_limits);
            const auto reserve_page=[&](std::uint64_t count){return registry.reserve_host_metadata_lifetime(count);};
            auto page=Page::create(reserve_page);auto shared=page;
            const auto before=registry.revision();
            registry.admit_host_metadata_lifetime(shared,Page::metadata_bytes(),&Page::metadata_credit_belongs_to,&Page::attach_metadata_credit);
            require(registry.revision()==before,"shared page publication duplicated reservation");
            const auto live=ninfer::exl3::bounded_shared_live_blocks_for_test<Page>();
            bool refused=false;try{auto other=Page::create(reserve_page);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && ninfer::exl3::bounded_shared_live_blocks_for_test<Page>()==live,
                "page ceiling refusal allocated storage");
            std::weak_ptr<Page> weak=page;page.reset();
            require(!weak.expired() && registry.retained_resource_units()[metadata_domain]==Page::metadata_bytes(),
                "shared page owner lost metadata charge");
            shared.reset();
            require(weak.expired() && registry.retained_resource_units()[metadata_domain]==Page::metadata_bytes(),
                "page final strong release lost weak block charge");
            require(registry.tighten_metadata_headroom_for_test(0)==Page::metadata_bytes(),
                "metadata ceiling tightening omitted weak-only page storage");
            refused=false;try{auto other=Page::create(reserve_page);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused,"tightened metadata ceiling admitted over weak-only page storage");
            weak.reset();auto retry=Page::create(reserve_page);retry.reset();
            require(registry.retained_resource_units()[metadata_domain]==0,"page ceiling retry leaked credit");
        }
        {
            using History=ninfer::exl3::Exl3TokenHistory;
            Registry registry(memory.ullTotalPhys-reserve,reserve);
            const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
            auto limits=Inventory::unlimited();limits[domain]=History::node_metadata_bytes();
            registry.set_resource_limits(limits);
            const auto reserve_control=[&](std::uint64_t count){return registry.reserve_host_metadata_lifetime(count);};
            const std::array<std::int64_t,1> token{17};
            auto history=History{}.append(token,reserve_control);auto alias=history;
            std::weak_ptr<const void> weak;
            const auto revision=registry.revision();
            alias.visit_control_owners([&](const auto& node){
                weak=node;
                registry.admit_host_metadata_lifetime(node,History::node_metadata_bytes(),
                    &History::control_credit_belongs_to,&History::attach_control_credit);
            });
            require(registry.revision()==revision,"token control publication duplicated credit");
            const auto live=History::live_node_blocks_for_test();bool refused=false;
            try{auto child=history.append(token,reserve_control);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && History::live_node_blocks_for_test()==live && history.equals(token),
                "token control ceiling refusal allocated node or changed parent");
            history=History{};
            require(!weak.expired() && alias.equals(token),"token control alias lost retained history");
            alias=History{};
            require(weak.expired() && registry.retained_resource_units()[domain]==History::node_metadata_bytes(),
                "token final strong release dropped weak control charge");
            refused=false;try{auto child=History{}.append(token,reserve_control);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused,"token control ceiling ignored weak owner");
            weak.reset();auto retry=History{}.append(token,reserve_control);retry=History{};
            require(registry.retained_resource_units()[domain]==0,"token control retry leaked credit");
        }
        ninfer::exl3::Exl3VeriCacheRequest::exercise_request_metadata_ceiling_for_test(
            [&]{return std::make_unique<Registry>(memory.ullTotalPhys-reserve,reserve);});
        // These component cases each own the process-wide resident authority.
        // Construct the following fixture only after their registries retire.
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        const auto bytes=ninfer::exl3::bounded_shared_allocation_bytes<int>();
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        auto limits=Inventory::unlimited();limits[domain]=bytes;registry.set_resource_limits(limits);
        {
            auto planned=registry.reserve_host_metadata_lifetime(bytes);
            bool refused=false;
            try{(void)registry.reserve_host_metadata_lifetime(1);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && planned.bytes()==bytes && registry.retained_resource_units()[domain]==bytes,
                "preconstruction snapshot promise did not consume available metadata");
        }
        require(registry.retained_resource_units()[domain]==0,
            "abandoned snapshot construction promise retained metadata");
        auto owner=ninfer::exl3::make_bounded_shared<int>(7);
        const auto matches=[](const std::shared_ptr<const void>& value,const Ledger& ledger) noexcept {
            return ninfer::exl3::bounded_split_credit_belongs_to<int>(value,ledger);
        };
        unsigned attachments=0;
        const auto attach=[&](const std::shared_ptr<const void>& value,Ledger::Ticket credit) noexcept {
            ++attachments;
            return ninfer::exl3::attach_bounded_split_retirement_credit<int>(value,std::move(credit),0);
        };
        registry.admit_host_metadata_lifetime(owner,bytes,matches,attach);
        const auto revision=registry.revision();
        registry.admit_host_metadata_lifetime(owner,bytes,matches,attach);
        require(attachments==1 && registry.revision()==revision && registry.retained_resource_units()[domain]==bytes,
            "owner lifetime re-admission duplicated charge or attachment");
        auto other=ninfer::exl3::make_bounded_shared<int>(9);bool refused=false;
        try{registry.admit_host_metadata_lifetime(other,bytes,matches,attach);}
        catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && attachments==1,"owner lifetime exact cap reached refused attachment");
        registry.replace({});
        std::weak_ptr<int> weak=owner;owner.reset();
        require(weak.expired() && registry.retained_resource_units()[domain]==bytes,
            "resident removal or final strong release dropped containing block credit");
        weak.reset();
        require(registry.retained_resource_units()[domain]==0,"final weak release retained owner lifetime credit");
        registry.admit_host_metadata_lifetime(other,bytes,matches,attach);
        require(attachments==2,"released owner lifetime credit did not permit retry");
        other.reset();
        Ledger foreign_ledger;
        auto foreign=ninfer::exl3::make_bounded_shared<int>(11);
        require(ninfer::exl3::attach_bounded_split_retirement_credit<int>(foreign,foreign_ledger.acquire(bytes),0),
            "foreign metadata fixture credit attachment");
        bool foreign_refused=false;
        try{registry.admit_host_metadata_lifetime(foreign,bytes,matches,attach);}
        catch(const std::logic_error& error){foreign_refused=std::string_view(error.what())==
            "resident metadata lifetime attachment refused";}
        require(foreign_refused && foreign_ledger.bytes()==bytes &&
            registry.retained_resource_units()[domain]==0 && *foreign==11,
            "foreign registry admission duplicated or stole owner credit");
        auto retry=ninfer::exl3::make_bounded_shared<int>(13);
        registry.admit_host_metadata_lifetime(retry,bytes,matches,attach);
        require(registry.retained_resource_units()[domain]==bytes,
            "foreign credit refusal left lifetime admission sealed");
        retry.reset();foreign.reset();
        require(foreign_ledger.bytes()==0 && registry.retained_resource_units()[domain]==0,
            "registry-specific metadata credit release diverged");
        registry.close();
    }
    for(const bool fixed:{false,true})for(std::size_t range_capacity:{1U,8U})
        for(unsigned scratch_slots=0;scratch_slots<=2;++scratch_slots) {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        const auto required=Registry::Snapshot::descriptor_requirement(1,range_capacity);
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        auto limits=Inventory::unlimited();
        limits[domain]=required.units[domain]+scratch_slots*sizeof(Registry::Range);
        registry.set_resource_limits(limits);
        Registry::Snapshot snapshot;snapshot.owners.reserve(1);
        if(fixed)snapshot.reserve_regions(range_capacity);else snapshot.regions.reserve(range_capacity);
        auto owner=std::make_shared<const int>(13);std::weak_ptr<const int> retained=owner;
        snapshot.owners.push_back(std::move(owner));
        // Assessment only: no commit or OS access to this synthetic page.
        snapshot.add(reinterpret_cast<const void*>(0x10000),4096);
        {
            Registry::Snapshot unowned;unowned.regions=snapshot.regions;
            unowned.owners.emplace_back(std::shared_ptr<const void>{},snapshot.owners.front().get());
            const auto revision=registry.revision();bool owner_refused=false;
            try{(void)registry.assess(std::move(unowned));}
            catch(const std::invalid_argument&){owner_refused=true;}
            require(owner_refused && registry.revision()==revision &&
                registry.retained_resource_units()==Inventory::Totals{} && !retained.expired(),
                "resident nonowning payload alias acquired coverage or changed authority");
        }
        bool assessed=false,refused=false;
        try{auto plan=registry.assess(std::move(snapshot));assessed=true;
            require(!retained.expired(),"T024 assessment lost snapshot payload");
            const auto charged=registry.retained_resource_units();
            const auto* buffer=plan.range_storage_for_test();
            auto moved=std::move(plan);
            require(moved.range_storage_for_test()==buffer && moved.range_capacity_for_test()==range_capacity &&
                registry.retained_resource_units()==charged &&
                charged[domain]==required.units[domain]+sizeof(Registry::Range),
                "T213 moved assessment lost reused capacity or duplicated credit");
            bool moved_from_refused=false;
            try{registry.commit(std::move(plan));}catch(const std::logic_error&){moved_from_refused=true;}
            require(moved_from_refused && !retained.expired() && registry.retained_resource_units()==charged,
                "T213 moved-from commit consumed surviving assessment");
        }
        catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
        // Incoming capacity is reused; only the added-range vector is extra.
        require((scratch_slots>=1?assessed:refused) && retained.expired() &&
            registry.retained_resource_units()==Inventory::Totals{},
            "T024 assessment scratch credit refused wrong stage or retained abandoned plan");
        registry.close();
    }
    {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        auto limits=Inventory::unlimited();limits[domain]=1ULL<<20;
        registry.set_resource_limits(limits);
        const auto required=Registry::Snapshot::descriptor_requirement(1,1);
        std::weak_ptr<const void> retained_payload;
        const auto stats=registry.replace_reserved_snapshot(required,[&](auto) {
            Registry::Snapshot snapshot;snapshot.owners.reserve(1);snapshot.reserve_regions(1);
            auto payload=std::make_shared<std::array<std::byte,4096>>();retained_payload=payload;
            snapshot.add(payload->data(),payload->size());snapshot.owners.push_back(std::move(payload));
            return snapshot;
        },[]{});
        const auto page_bytes=registry.retained_page_descriptor_bytes_for_test();
        require(registry.retained_resource_units()==stats.resource_units && page_bytes==sizeof(Registry::Range) &&
            stats.resource_units[domain]==Registry::Snapshot::descriptor_requirement(1,0).units[domain]+page_bytes && !retained_payload.expired(),
            "T024 committed page descriptor capacity missing from retained accounting");
        Inventory::Requirement excess;excess.configuration=0x5041474543524544ULL;
        excess.add(Inventory::Domain::host_metadata,1,limits[domain]-stats.resource_units[domain]+1);
        bool called=false,refused=false;
        try{registry.allocate_reserved(excess,[&](auto)->Inventory {
            called=true;throw std::runtime_error("prepared page credit reached allocator");
        });}catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && !called && !retained_payload.expired(),
            "T024 later allocation reused persistent page descriptor credit");
        registry.close();
        require(registry.retained_resource_units()==Inventory::Totals{} &&
            registry.retained_page_descriptor_bytes_for_test()==0 && retained_payload.expired(),
            "T024 close retained page descriptors or payload after unlock");
    }
    for(const bool commit_pressure:{false,true}) {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        const auto required=Registry::Snapshot::descriptor_requirement(1,0);
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        auto limits=Inventory::unlimited();limits[domain]=2*required.units[domain];
        registry.set_resource_limits(limits);
        std::array<std::weak_ptr<const int>,3> owners;
        const auto prepare=[&](unsigned index) {
            Registry::Snapshot snapshot;snapshot.owners.reserve(1);snapshot.reserve_regions(0);
            auto owner=std::make_shared<const int>(index);owners[index]=owner;
            snapshot.owners.push_back(std::move(owner));return snapshot;
        };
        std::optional<Registry::Transition> first,second;
        first.emplace(registry.assess(prepare(0)));second.emplace(registry.assess(prepare(1)));
        bool refused=false;
        try{auto third=registry.assess(prepare(2));}
        catch(const std::runtime_error& error){refused=std::string_view(error.what())==
            "resource transition budget exhausted";}
        require(refused && owners[2].expired() && !owners[0].expired() && !owners[1].expired() &&
            registry.retained_resource_units()[domain]==limits[domain],
            "T024 simultaneous assessments exceeded descriptor credit or lost owners");
        if(commit_pressure) {
            bool refused_commit=false;const auto revision=registry.revision();
            try{registry.commit(std::move(*first));}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused_commit=true;}
            require(refused_commit && registry.revision()==revision && owners[0].expired() &&
                !owners[1].expired() && registry.retained_resource_units()[domain]==required.units[domain],
                "T024 commit ignored later assessment pressure or damaged surviving plan");
        }
        first.reset();
        require(owners[0].expired() && registry.retained_resource_units()[domain]==required.units[domain],
            "T024 abandoned assessment kept or over-released descriptor credit");
        const auto committed=registry.commit(std::move(*second));
        require(registry.retained_resource_units()==committed.resource_units &&
            committed.resource_units[domain]==required.units[domain] && !owners[1].expired(),
            "T024 consumed assessment wrapper retained duplicate descriptor credit");
        registry.close();
        require(owners[1].expired() && registry.retained_resource_units()==Inventory::Totals{},
            "T024 consumed assessment wrapper retained payload after close");
    }
    for(unsigned invalidation=0;invalidation<4;++invalidation) {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        std::optional<Registry::Transition> survivor;
        std::weak_ptr<const int> payload;
        {
            Registry registry(memory.ullTotalPhys-reserve,reserve);
            const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
            const auto required=Registry::Snapshot::descriptor_requirement(1,0);
            auto limits=Inventory::unlimited();limits[domain]=3*required.units[domain];
            registry.set_resource_limits(limits);
            Registry::Snapshot snapshot;snapshot.owners.reserve(1);snapshot.reserve_regions(0);
            auto owner=std::make_shared<const int>(19);payload=owner;
            snapshot.owners.push_back(std::move(owner));survivor.emplace(registry.assess(std::move(snapshot)));
            if(invalidation==0)registry.replace({});
            else if(invalidation==1) {
                Inventory::Requirement demand;demand.configuration=91;
                demand.add(Inventory::Domain::host_metadata,1,1);
                bool failed=false;
                try{registry.allocate_reserved(demand,[](auto)->Inventory{throw std::bad_alloc();});}
                catch(const std::bad_alloc&){failed=true;}
                require(failed,"T024 stale retained-plan reservation fixture did not fail");
            } else if(invalidation==2)registry.close();
            else {
                const auto retained=registry.retained_resource_units()[domain];
                require(registry.tighten_metadata_headroom_for_test(0)==retained,
                    "T220 tightening omitted outstanding assessment metadata");
            }
            bool stale=false;
            try{registry.commit(std::move(*survivor));}
            catch(const std::logic_error& error){stale=std::string_view(error.what())=="resident stale/consumed transition";}
            require(stale && !payload.expired() && registry.retained_resource_units()[domain]==required.units[domain],
                "T024 stale plan lost descriptor credit or retained payload before destruction");
            if(invalidation!=2) {
                survivor.reset();require(payload.expired() && registry.retained_resource_units()==Inventory::Totals{},
                    "T024 stale plan destruction did not release exact ownership");
                registry.close();
            }
        }
        if(invalidation==2)require(!payload.expired(),"T024 registry destruction invalidated caller-owned plan storage");
        survivor.reset();require(payload.expired(),"T024 outliving plan retained payload after final destruction");
    }
    {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        const auto bytes=Coordinator::runtime_host_source_metadata_bytes();
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        {
            Coordinator owner(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
            auto limits=Inventory::unlimited();limits[domain]=bytes-1;
            owner.bind_physical_resources({},limits);
            bool refused=false;try {owner.reserve_runtime_host_sources();}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&) {refused=true;}
            require(refused && owner.runtime_host_source_storage_bytes()==0 &&
                owner.runtime_host_source_owner_for_test().expired(),
                "T024 host source table exceeded full backing reservation");
            owner.close();
        }
        Coordinator owner(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
        auto limits=Inventory::unlimited();limits[domain]=bytes;
        owner.bind_physical_resources({},limits);owner.reserve_runtime_host_sources();
        auto weak=owner.runtime_host_source_owner_for_test();
        owner.reserve_runtime_host_sources();
        require(owner.runtime_host_source_storage_bytes()==bytes &&
            owner.stats().retained_resource_units->at(domain)==bytes,
            "T024 host source table reserve changed extent or duplicated charge");
        owner.close();
        require(weak.expired() && owner.stats().retained_resource_units->at(domain)==bytes,
            "T024 host source weak tail lost bounded backing credit");
        weak.reset();
        require(owner.stats().retained_resource_units==Inventory::Totals{},
            "T024 host source final weak release retained credit");
    }
    if(env("NINFER_TEST_COORDINATOR_LATE_RELEASE")=="1") {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        std::weak_ptr<const void> produced;
        unsigned rollbacks=0,observations=0;bool callback_alive=false,observer_released=false;
        {
            Coordinator owner(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
            owner.bind_physical_resources({},Inventory::unlimited());
            Inventory::Requirement required;required.configuration=24;
            required.add(Inventory::Domain::host_metadata,1,64);
            bool original=false;
            try {owner.allocate_startup_resources(required,[&](auto) {
                auto backing=std::make_shared<std::array<std::byte,65>>();produced=backing;
                Inventory actual;actual.add({backing,0,Inventory::Domain::host_metadata,65});return actual;
            },[&] {++rollbacks;callback_alive=!produced.expired();},[&] {
                ++observations;observer_released=produced.expired();
                throw std::runtime_error("prepared late cleanup failure");
            });}catch(const std::invalid_argument& error) {original=std::string_view(error.what())==
                "resource reservation actual extent/domain mismatch";}
            require(original && rollbacks==1 && observations==1 && callback_alive && observer_released,
                "late cleanup observation order or primary exception changed");
            require(owner.stats().external_retirement_unresolved && !owner.stats().retained_resource_units,
                "late cleanup failure reopened resource authority");
            bool refused=false;try {owner.close();}catch(const std::logic_error&) {refused=true;}
            require(refused && !owner.stats().closed && rollbacks==1 && observations==1,
                "late cleanup failure closed successfully or replayed callbacks");
        }
        require(Registry::retirement_quarantined(),"late cleanup failure lost process seal");
        std::cout << "T84_LATE_RELEASE_COMPLETE full_serving_coverage=0" << std::endl;return;
    }
    if(env("NINFER_TEST_COORDINATOR_ROLLBACK_QUARANTINE")=="1" ||
       env("NINFER_TEST_COORDINATOR_ROLLBACK_QUARANTINE")=="startup") {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        unsigned rollback_calls=0;
        std::weak_ptr<const void> produced;
        {
            Coordinator owner(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
            owner.bind_physical_resources({},Inventory::unlimited());
            bool original=false;
            try {
                const auto make_allocation=[&]()->Inventory::Allocation {
                    auto backing=std::make_shared<std::array<std::byte,65>>();produced=backing;
                    return {backing,0,Inventory::Domain::host_metadata,65};
                };
                const auto rollback=[&] {
                    ++rollback_calls;require(!produced.expired(),"T024 rollback lost produced owner before callback");
                    throw std::runtime_error("prepared rollback failure");
                };
                if(env("NINFER_TEST_COORDINATOR_ROLLBACK_QUARANTINE")=="startup") {
                    Inventory::Requirement required;required.configuration=23;
                    required.add(Inventory::Domain::host_metadata,1,64);
                    owner.allocate_startup_resources(required,[&](auto) {
                        Inventory actual;actual.add(make_allocation());return actual;
                    },rollback);
                } else owner.allocate_logical_host(64,make_allocation,rollback);
            } catch(const std::invalid_argument& error) {
                original=std::string_view(error.what())=="resource reservation actual extent/domain mismatch";
            }
            require(original && rollback_calls==1,"T024 rollback replaced original factory failure");
            require(owner.stats().external_retirement_unresolved && !owner.stats().retained_resource_units,
                "T024 failed factory advertised a complete retained-resource total");
            for(unsigned attempt=0;attempt<2;++attempt) {
                bool refused=false;try {owner.close();}
                catch(const std::logic_error& error) {refused=std::string_view(error.what())==
                    "resident external allocation retirement unresolved";}
                require(refused && owner.stats().closing && !owner.stats().closed && rollback_calls==1 && !produced.expired(),
                    "T024 failed rollback advertised successful close or replayed rollback");
            }
        }
        require(Registry::retirement_quarantined() && !produced.expired(),"T024 failed rollback lost process quarantine or factory owner");
        bool refused=false;try {Registry replacement(memory.ullTotalPhys-reserve,reserve);}
        catch(const std::logic_error&) {refused=true;}
        require(refused,"T024 failed rollback permitted replacement registry");
        std::cout << "T84_ROLLBACK_QUARANTINE_COMPLETE close_refused=2 full_serving_coverage=0" << std::endl;
        return;
    }
    if(env("NINFER_TEST_RESIDENT_CLOSE_QUARANTINE")=="1") {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        struct Backing {mutable unsigned attempts=0;};
        const auto baseline=ninfer::exl3::bounded_shared_live_blocks_for_test<Backing>();
        std::weak_ptr<Backing> weak;
        {
            Registry registry(memory.ullTotalPhys-reserve,reserve);
            const auto bytes=ninfer::exl3::bounded_shared_allocation_bytes<Backing>();
            Inventory::Requirement required;required.configuration=21;
            required.add(Inventory::Domain::host_metadata,1,bytes);
            registry.allocate_reserved(required,[&](auto) {
                auto owner=ninfer::exl3::make_bounded_shared<Backing>();weak=owner;
                Inventory actual;actual.add({owner,0,Inventory::Domain::host_metadata,bytes,{},
                    +[](const std::shared_ptr<const void>& value,
                        ninfer::exl3::RetainedDescriptorLedger::Ticket) noexcept {
                        ++static_cast<const Backing*>(value.get())->attempts;return false;
                    }});return actual;
            });
            bool refused=false;try {registry.close();}
            catch(const std::logic_error& error) {refused=std::string_view(error.what())==
                "resident close lifetime transfer refused";}
            require(refused && registry.retained_resource_units()[
                static_cast<unsigned>(Inventory::Domain::host_metadata)]==bytes,
                "T024 explicit hook refusal lost original charge");
        }
        auto survivor=weak.lock();
        require(survivor && survivor->attempts==2 && Registry::failed_close_retained_for_test() &&
            ninfer::exl3::bounded_shared_live_blocks_for_test<Backing>()==baseline+1,
            "T024 destructor hook refusal released allocation or replayed fallback callback");
        survivor.reset();
        bool sealed=false;try {Registry replacement(memory.ullTotalPhys-reserve,reserve);}
        catch(const std::logic_error& error) {sealed=std::string_view(error.what())==
            "one resident registry per process";}
        require(sealed && !weak.expired(),"T024 refused-close quarantine allowed replacement");
        // Permanent quarantine: run separately from ordinary T84 coverage.
        std::cout << "T84_CLOSE_QUARANTINE_COMPLETE hook_refusal=1 process_sealed=1"
                  << " full_serving_coverage=0" << std::endl;
        return;
    }
    if(env("NINFER_TEST_RESIDENT_POISON_METADATA")=="1") {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        std::array<std::weak_ptr<const void>,2> owners;
        {
            Registry registry(memory.ullTotalPhys-reserve,reserve);
            registry.set_resource_limits(Inventory::unlimited());
            const auto required=Registry::Snapshot::descriptor_requirement(1,1);
            const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
            const auto prepare=[&](unsigned index) {
                void* allocation=VirtualAlloc(nullptr,65536,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
                require(allocation!=nullptr,"T024 poison fixture allocation");
                std::shared_ptr<void> owner(allocation,[](void* pointer){VirtualFree(pointer,0,MEM_RELEASE);});
                owners[index]=owner;
                Registry::Snapshot snapshot;snapshot.owners.reserve(1);snapshot.reserve_regions(1);
                snapshot.add(allocation,65536);snapshot.owners.push_back(std::move(owner));return snapshot;
            };
            registry.replace_reserved_snapshot(required,[&](auto){return prepare(0);},[]{});
            registry.fail_next_rollback_for_test();
            bool duplicate=false;try{registry.fail_next_rollback_for_test();}
            catch(const std::logic_error&){duplicate=true;}
            unsigned locked=0;bool poisoned=false;
            try{registry.replace_reserved_snapshot(required,[&](auto){return prepare(1);},[]{},[&](auto) {
                ++locked;throw std::runtime_error("prepared failure after new lock");
            });}catch(const std::runtime_error& error){poisoned=std::string_view(error.what())==
                "injected resident rollback before unlock";}
            const auto expected=2*Registry::Snapshot::descriptor_requirement(1,0).units[domain]+2*sizeof(Registry::Range)+
                2*sizeof(std::shared_ptr<const void>);
            require(duplicate && poisoned && locked==1 && !owners[0].expired() && !owners[1].expired() &&
                registry.retained_resource_units()[domain]==expected && registry.locked_page_bytes()==2*65536,
                "T024 poisoned rollback lost old/new owners or combined descriptor accounting");
            bool sealed=false;try{auto plan=registry.assess({});}
            catch(const std::logic_error&){sealed=true;}
            bool close_refused=false;try{registry.close();}catch(const std::logic_error&){close_refused=true;}
            require(sealed && close_refused && !owners[0].expired() && !owners[1].expired(),
                "T024 poisoned registry reopened or advertised successful retirement");
        }
        require(!owners[0].expired() && !owners[1].expired() && Registry::failed_close_retained_for_test(),
            "T024 poisoned teardown released uncertain owner bundle");
        for(unsigned attempt=0;attempt<2;++attempt) {
            bool refused=false;
            try {Registry replacement(memory.ullTotalPhys-reserve,reserve);}
            catch(const std::logic_error& error) {refused=std::string_view(error.what())==
                "one resident registry per process";}
            require(refused && !owners[0].expired() && !owners[1].expired(),
                "T024 quarantined process admitted replacement or recycled owners");
        }
        // Terminal isolated-process fault mode; later cases need a healthy registry.
        std::cout << "T84_POISON_QUARANTINE_COMPLETE retained_owners=2 process_sealed=1"
                  << " full_serving_coverage=0" << std::endl;
        return;
    }
    {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        registry.set_resource_limits(Inventory::unlimited());
        Inventory::Requirement requirement;requirement.configuration=0x484f535452455449ULL;
        requirement.add(Inventory::Domain::host_metadata,1,64);
        Inventory::Allocation allocation;std::weak_ptr<const void> owner;
        registry.allocate_reserved(requirement,[&](auto) {
            auto backing=std::make_shared<std::array<std::byte,64>>();owner=backing;
            allocation={backing,0,Inventory::Domain::host_metadata,64};
            Inventory actual;actual.add(allocation);return actual;
        });
        const auto held=registry.retained_resource_units();
        require(!registry.retire_reserved_host(allocation,[]{return false;}) &&
            registry.retained_resource_units()==held && !owner.expired(),
            "T024 refused host retirement released metadata credit");
        bool threw=false;
        try{registry.retire_reserved_host(allocation,[]()->bool {
            throw std::runtime_error("prepared host retirement failure");
        });}catch(const std::runtime_error& error){threw=std::string_view(error.what())==
            "prepared host retirement failure";}
        require(threw && registry.retained_resource_units()==held && !owner.expired(),
            "T024 throwing host retirement released metadata ownership");
        bool wrong_domain=false,called=false;auto invalid=allocation;invalid.domain=Inventory::Domain::device;
        try{registry.retire_reserved_host(invalid,[&]{called=true;return true;});}
        catch(const std::invalid_argument&){wrong_domain=true;}
        invalid={};
        require(wrong_domain && !called,"T024 host retirement accepted device-domain callback");
        require(registry.retire_reserved_host(allocation,[&]{allocation.owner.reset();return true;}) &&
            registry.retained_resource_units()==Inventory::Totals{} && owner.expired(),
            "T024 host retirement retry retained payload or charge");
        registry.close();
    }
    {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        using Backing=std::array<std::byte,64>;
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        const auto bytes=ninfer::exl3::bounded_shared_allocation_bytes<Backing>();
        auto limits=Inventory::unlimited();limits[domain]=bytes;registry.set_resource_limits(limits);
        Inventory::Requirement required;required.configuration=0x5745414b5441494cULL;
        required.add(Inventory::Domain::host_metadata,1,bytes);
        Inventory::Allocation allocation;std::weak_ptr<const void> weak;
        registry.allocate_reserved(required,[&](auto) {
            auto owner=ninfer::exl3::make_bounded_shared<Backing>();weak=owner;
            allocation={owner,0,Inventory::Domain::host_metadata,bytes};
            Inventory actual;actual.add(allocation);return actual;
        });
        require(!registry.retire_reserved_host_to_lifetime(allocation,
            [](ninfer::exl3::RetainedDescriptorLedger::Ticket) noexcept {return false;}) &&
            registry.retained_resource_units()[domain]==bytes && !weak.expired(),
            "T024 refused lifetime transfer changed inventory or duplicated credit");
        require(registry.retire_reserved_host_to_lifetime(allocation,
            [&](ninfer::exl3::RetainedDescriptorLedger::Ticket credit) noexcept {
                return ninfer::exl3::attach_bounded_retirement_credit<Backing>(allocation.owner,std::move(credit));
            }) && registry.retained_resource_units()[domain]==bytes,
            "T024 lifetime transfer lost or duplicated metadata charge");
        allocation.owner.reset();
        require(weak.expired() && registry.retained_resource_units()[domain]==bytes,
            "T024 last strong owner released weak control-tail credit");
        Inventory::Requirement excess;excess.configuration=7;excess.add(Inventory::Domain::host_metadata,1,1);
        bool called=false,refused=false;
        try{registry.allocate_reserved(excess,[&](auto)->Inventory {
            called=true;throw std::runtime_error("prepared weak tail credit reached allocator");
        });}catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && !called,"T024 weak-only control tail credit was reused");
        registry.close();
        require(registry.retained_resource_units()[domain]==bytes,"T024 close discarded live weak-tail charge");
        weak.reset();
        require(registry.retained_resource_units()==Inventory::Totals{},
            "T024 final weak retirement retained lifetime credit");
    }
    {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        using Backing=std::array<std::byte,64>;
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        const auto bytes=ninfer::exl3::bounded_shared_allocation_bytes<Backing>();
        auto limits=Inventory::unlimited();limits[domain]=bytes;registry.set_resource_limits(limits);
        Inventory::Requirement required;required.configuration=19;required.add(Inventory::Domain::host_metadata,1,bytes);
        std::shared_ptr<Backing> survivor;std::weak_ptr<Backing> weak;
        registry.allocate_reserved(required,[&](auto) {
            survivor=ninfer::exl3::make_bounded_shared<Backing>();weak=survivor;
            Inventory actual;actual.add({survivor,0,Inventory::Domain::host_metadata,bytes,{},
                +[](const std::shared_ptr<const void>& owner,ninfer::exl3::RetainedDescriptorLedger::Ticket credit) noexcept {
                    return ninfer::exl3::attach_bounded_retirement_credit<Backing>(owner,std::move(credit));
                }});return actual;
        });
        registry.close();
        require(!weak.expired() && registry.retained_resource_units()[domain]==bytes,
            "T024 close dropped live owner credit before lifetime transfer");
        survivor.reset();
        require(weak.expired() && registry.retained_resource_units()[domain]==bytes,
            "T024 close-transferred credit ignored weak control tail");
        weak.reset();require(registry.retained_resource_units()==Inventory::Totals{},
            "T024 close-transferred credit survived final allocation retirement");
    }
    {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        struct Backing {std::array<std::byte,64> payload;};
        const auto baseline=ninfer::exl3::bounded_shared_live_blocks_for_test<Backing>();
        const auto bytes=ninfer::exl3::bounded_shared_allocation_bytes<Backing>();
        std::shared_ptr<Backing> survivor;std::weak_ptr<Backing> weak;
        {
            ninfer::exl3::Exl3HostResidentSet registry(memory.ullTotalPhys-reserve,reserve);
            Inventory::Requirement required;required.configuration=20;
            required.add(Inventory::Domain::host_metadata,1,bytes);
            registry.allocate_reserved(required,[&](auto) {
                survivor=ninfer::exl3::make_bounded_shared<Backing>();weak=survivor;
                Inventory actual;actual.add({survivor,0,Inventory::Domain::host_metadata,bytes,{},
                    &ninfer::exl3::attach_bounded_retirement_credit<Backing,const void>});return actual;
            });
            require(!ninfer::exl3::bounded_retirement_credit_bytes_for_test<Backing>(survivor),
                "T024 inventory owner acquired retirement credit before teardown");
            // Intentionally omit explicit close: the healthy destructor must transfer.
        }
        require(ninfer::exl3::bounded_retirement_credit_bytes_for_test<Backing>(survivor)==bytes,
            "T024 implicit registry close lost surviving allocation credit");
        survivor.reset();
        require(weak.expired() && ninfer::exl3::bounded_shared_live_blocks_for_test<Backing>()==baseline+1,
            "T024 implicit close released backing before final weak retirement");
        weak.reset();
        require(ninfer::exl3::bounded_shared_live_blocks_for_test<Backing>()==baseline,
            "T024 implicit close leaked bounded allocation");
    }
    {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        struct Backing {mutable unsigned calls=0;mutable bool refuse=false;};
        Coordinator owner(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
        owner.bind_physical_resources({},Inventory::unlimited());
        const auto bytes=ninfer::exl3::bounded_shared_allocation_bytes<Backing>();
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        std::array<std::shared_ptr<Backing>,2> survivors;
        std::array<std::weak_ptr<Backing>,2> weak;
        std::array<std::optional<Coordinator::LogicalHostLease>,2> leases;
        for(unsigned i=0;i<2;++i)leases[i].emplace(owner.allocate_logical_host(bytes,[&] {
            survivors[i]=ninfer::exl3::make_bounded_shared<Backing>();survivors[i]->refuse=i==1;weak[i]=survivors[i];
            return Inventory::Allocation{survivors[i],0,Inventory::Domain::host_metadata,bytes,{},
                +[](const std::shared_ptr<const void>& value,ninfer::exl3::RetainedDescriptorLedger::Ticket credit) noexcept {
                    const auto backing=std::static_pointer_cast<const Backing>(value);++backing->calls;
                    if(backing->refuse)return false;
                    return ninfer::exl3::attach_bounded_retirement_credit<Backing>(value,std::move(credit));
                }};
        },[]{}));
        bool refused=false;try{owner.close();}catch(const std::logic_error& error){refused=std::string_view(error.what())==
            "resident close lifetime transfer refused";}
        const auto partial=owner.stats();
        require(refused && partial.closing && !partial.closed && partial.retained_resource_units &&
            (*partial.retained_resource_units)[domain]==2*bytes && survivors[0]->calls==1 && survivors[1]->calls==1,
            "T024 partial close lost credits or replayed accepted retirement hook");
        bool admission_sealed=false;try{owner.admit(initial[0]);}
        catch(const std::logic_error& error){admission_sealed=std::string_view(error.what())=="serving coordinator close in progress";}
        require(admission_sealed,"T024 partial close reopened serving admission");
        survivors[1]->refuse=false;owner.close();owner.close();
        require(owner.stats().closed && !owner.stats().closing && survivors[0]->calls==1 && survivors[1]->calls==2,
            "T024 close retry replayed transferred hook or did not finish closure");
        for(unsigned i=0;i<2;++i){survivors[i].reset();leases[i].reset();}
        require(weak[0].expired() && weak[1].expired() && (*owner.stats().retained_resource_units)[domain]==2*bytes,
            "T024 partial-close recovery discarded weak-tail credits");
        weak[0].reset();weak[1].reset();
        require(*owner.stats().retained_resource_units==Inventory::Totals{},
            "T024 final weak retirement retained close-retry charges");
    }
    for(const bool enough_credit:{false,true}) {
        using Registry=ninfer::exl3::Exl3HostResidentSet;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Registry registry(memory.ullTotalPhys-reserve,reserve);
        const auto required=Registry::Snapshot::descriptor_requirement(1,0);
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        auto limits=Inventory::unlimited();limits[domain]=enough_credit?
            required.units[domain]+sizeof(std::shared_ptr<const void>):required.units[domain]-1;
        registry.set_resource_limits(limits);
        unsigned factories=0,before_commit=0;
        std::weak_ptr<const int> retained;
        const auto factory=[&](std::uint64_t configuration) {
            ++factories;require(configuration==required.configuration,"T024 snapshot credit configuration");
            Registry::Snapshot result;result.owners.reserve(1);result.reserve_regions(0);
            auto payload=std::make_shared<const int>(9);retained=payload;
            result.owners.push_back(std::move(payload));return result;
        };
        if(!enough_credit) {
            const auto revision=registry.revision();bool refused=false;
            try{registry.replace_reserved_snapshot(required,factory,[&]{++before_commit;});}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && factories==0 && before_commit==0 && registry.revision()==revision,
                "T024 insufficient snapshot credit reached factory or changed authority");
        } else {
            for(unsigned malformed=0;malformed<3;++malformed) {
                auto invalid=required;
                if(malformed==0)invalid.configuration=0;
                if(malformed==1)invalid.units[domain]=0;
                if(malformed==2)invalid.units[static_cast<unsigned>(Inventory::Domain::device)]=1;
                const auto revision=registry.revision();bool refused=false;
                try{registry.replace_reserved_snapshot(invalid,factory,[&]{++before_commit;});}
                catch(const std::invalid_argument&){refused=true;}
                require(refused && factories==0 && before_commit==0 && registry.revision()==revision,
                    "T024 malformed descriptor promise reached construction or changed revision");
            }
            auto stale=registry.assess({});
            bool factory_failed=false;
            try{registry.replace_reserved_snapshot(required,[&](auto)->Registry::Snapshot {
                throw std::bad_alloc();
            },[&]{++before_commit;});}catch(const std::bad_alloc&){factory_failed=true;}
            require(factory_failed && before_commit==0,"T024 failed snapshot factory reached commit");
            bool stale_refused=false;
            try{registry.commit(std::move(stale));}catch(const std::logic_error&){stale_refused=true;}
            require(stale_refused,"T024 snapshot reservation rollback left old assessment usable");
            bool extent_refused=false;
            try{registry.replace_reserved_snapshot(required,[&](auto configuration) {
                auto result=factory(configuration);result.owners.reserve(2);return result;
            },[&]{++before_commit;});}
            catch(const std::invalid_argument& error){extent_refused=std::string_view(error.what())==
                "resident snapshot reserved extent mismatch";}
            require(extent_refused && retained.expired() && before_commit==0,
                "T024 actual descriptor overrun published or retained private ownership");
            bool publication_failed=false;
            try{registry.replace_reserved_snapshot(required,factory,[] {
                throw std::runtime_error("prepared snapshot publication refusal");
            });}catch(const std::runtime_error& error){publication_failed=std::string_view(error.what())==
                "prepared snapshot publication refusal";}
            require(publication_failed && retained.expired(),"T024 failed snapshot publication retained private backing");
            const auto refuse_nested=[&] {
                const auto calls=factories;bool refused=false;
                try{registry.replace_reserved_snapshot(required,factory,[]{});}
                catch(const std::logic_error&){refused=true;}
                require(refused && factories==calls,"T024 nested descriptor promise entered allocator");
            };
            const auto stats=registry.replace_reserved_snapshot(required,[&](auto configuration) {
                refuse_nested();return factory(configuration);
            },[&]{refuse_nested();++before_commit;});
            require(before_commit==1 && stats.resource_units[domain]==required.units[domain] &&
                stats.transition_peak_resource_units[domain]==limits[domain] && !retained.expired(),
                "T024 snapshot reservation retry lost credit or committed backing");
        }
        registry.close();
        require(retained.expired(),"T024 snapshot close retained descriptor payload");
    }
    {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Coordinator capped(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
        auto limits=Inventory::unlimited();
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        limits[domain]=capped.metadata_requirement().units[domain]-1;
        capped.bind_physical_resources({},limits);
        for(int attempt=0;attempt<2;++attempt) {
            bool refused=false;
            try{capped.reserve_metadata_startup(4);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && capped.metadata_owner_for_test().expired() &&
                capped.queue_capacity_for_test()==0 && capped.stats().resident_updates==0,
                "T024 missing credit entered allocator or published metadata");
        }
        capped.close();
    }
    // Admission must cover every active request's eventual FIFO return, not
    // only the requests that happened to be queued during allocation.
    for(const bool batch:{false,true}) {
        Coordinator owner(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
        owner.bind_physical_resources({},ninfer::exl3::Exl3ResourceInventory::unlimited());
        const auto metadata_blocks=Coordinator::metadata_owner_blocks_for_test();
        for(unsigned fault=1;fault<=3;++fault) {
            bool refused=false;
            try{owner.reserve_metadata_startup(fault);}
            catch(const std::bad_alloc&){refused=fault<=2;}
            catch(const std::invalid_argument& error){
                refused=fault==3 && std::string_view(error.what())==
                    "resource reservation actual extent/domain mismatch";
            }
            require(refused && owner.metadata_owner_for_test().expired() &&
                owner.queue_capacity_for_test()==0 && owner.stats().admitted==0 &&
                Coordinator::metadata_owner_blocks_for_test()==metadata_blocks,
                "T024 failed startup published metadata owner or capacity");
        }
        owner.reserve_metadata_startup();
        const auto metadata_domain=static_cast<unsigned>(ninfer::exl3::Exl3ResourceInventory::Domain::host_metadata);
        const auto metadata_bytes=owner.metadata_requirement().units[metadata_domain];
        auto metadata=owner.metadata_owner_for_test();
        require(!metadata.expired() && owner.queue_capacity_for_test()==3 &&
            Coordinator::metadata_owner_blocks_for_test()==metadata_blocks+1,
            "T024 startup failed to retain exact logical metadata slots");
        bool duplicate=false;
        try{owner.reserve_metadata_startup();}catch(const std::logic_error&){duplicate=true;}
        require(duplicate && !metadata.expired(),"T024 duplicate startup replaced metadata");
        const auto first=owner.admit(initial[0]);
        const auto active=owner.acquire();
        require(active && active->ticket.request_id==first.request_id,
            "T024 return-slot fixture acquisition");
        Coordinator::Ticket second;
        if(batch) {
            const std::array<std::shared_ptr<const Request>,2> additions{initial[1],initial[2]};
            std::array<Coordinator::Ticket,2> output{{{991,17},{992,18}}};
            bool bad_extent=false;
            try{owner.admit_batch_into(additions,std::span<Coordinator::Ticket>(output).first(1));}
            catch(const std::invalid_argument&){bad_extent=true;}
            require(bad_extent && output[0].request_id==991 && output[1].request_id==992 &&
                owner.stats().admitted==1,"T024 admission output extent changed tickets or owners");
            const auto before=owner.stats();
            owner.fail_next_residency_for_test(Coordinator::ResidencyFault::before_commit);
            bool failed=false;
            try{owner.admit_batch_into(additions,output);}
            catch(const std::runtime_error& error){failed=std::string_view(error.what())==
                "injected coordinator before resident commit";}
            require(failed && output[0].request_id==991 && output[0].generation==17 &&
                output[1].request_id==992 && output[1].generation==18 &&
                owner.stats().admitted==before.admitted && owner.stats().active==before.active &&
                owner.stats().resident_updates==before.resident_updates &&
                owner.publication_scratch_empty_for_test(),
                "T024 failed batch admission changed output/authority or retained scratch");
            owner.admit_batch_into(additions,output);second=output.front();
            require(second.request_id==first.request_id+1 && output[1].request_id==first.request_id+2 &&
                owner.publication_scratch_empty_for_test(),
                "T024 batch retry consumed identities or retained replacement roots");
        } else {
            second=owner.admit(initial[1]);owner.admit(initial[2]);
        }
        const auto other=owner.acquire();
        require(other && other->ticket.request_id==second.request_id,
            "T024 return-slot second acquisition");
        const auto capacity=owner.queue_capacity_for_test();
        require(capacity>=owner.stats().admitted,"T024 admission omitted active return slots");
        const auto before=owner.stats();
        owner.yield(*active);owner.yield(*other);
        const auto after=owner.stats();
        require(after.queued==3 && after.active==0 &&
            owner.queue_capacity_for_test()==capacity &&
            after.resident_updates==before.resident_updates &&
            after.publications==before.publications,
            "T024 yield grew metadata or advanced resident/publication authority");
        const auto oldest=owner.acquire();
        require(oldest && oldest->root==initial[2],"T024 yielding disturbed existing FIFO head");
        owner.complete(*oldest);
        const auto returned=owner.acquire();
        require(returned && returned->ticket.request_id==first.request_id,
            "T024 yielded request lost FIFO return order");
        owner.complete(*returned);
        const auto last=owner.acquire();
        require(last && last->ticket.request_id==second.request_id,"T024 final return missing");
        owner.complete(*last);
        require(!metadata.expired() && owner.queue_capacity_for_test()==capacity,
            "T024 empty coordinator released retained metadata capacity early");
        owner.close();
        require(metadata.expired() && owner.queue_capacity_for_test()==0,
            "T024 close retained metadata backing after inventory retirement");
        require(Coordinator::metadata_owner_blocks_for_test()==metadata_blocks+1,
            "T024 close freed bounded control storage while a weak diagnostic survived");
        require(owner.stats().retained_resource_units &&
            (*owner.stats().retained_resource_units)[metadata_domain]==metadata_bytes,
            "T024 metadata weak diagnostic lost its startup retirement credit");
        metadata.reset();
        require(Coordinator::metadata_owner_blocks_for_test()==metadata_blocks &&
            *owner.stats().retained_resource_units==ninfer::exl3::Exl3ResourceInventory::Totals{},
            "T024 last weak metadata diagnostic retained bounded owner allocation");
    }
    {
        using Index=ninfer::exl3::Exl3VeriCachePrefixIndex;
        Cache retention_cache(Cache::Policy{2,64,cache_budget,reserve},identity);
        require(retention_cache.admit_input_authority(initial[0],prompts[0],available()).admitted,
            "T230 coordinator retention fixture admission");
        Coordinator retention_owner(retention_cache,Coordinator::Policy{
            2,1,memory.ullTotalPhys-reserve,reserve});
        const auto ticket=retention_owner.admit(initial[0]);
        const auto metadata=retention_cache.retention_metadata().front();
        const std::array<Index::RetentionDecision,1> permission{{
            {metadata.root,metadata.generation,false,0}}};
        const auto assert_required=[&] {
            const auto before=retention_owner.stats();
            const auto trimmed=retention_owner.trim_prefix_retention(reserve,permission);
            const auto replacement=retention_owner.admit_prefix_input_with_retention(
                initial[0],prompts[0],available(),permission);
            require(!replacement.admitted &&
                retention_cache.retention_metadata().front().generation==metadata.generation,
                "T230 live input admission replaced required cache generation");
            const auto after=retention_owner.stats();
            require(trimmed.inputs_current && !trimmed.budget_met && trimmed.evicted==0 &&
                retention_cache.roots()==std::vector<std::shared_ptr<const Request>>{initial[0]} &&
                before.locked_page_bytes==after.locked_page_bytes &&
                before.resident_updates==after.resident_updates &&
                before.resident_unlocked_bytes==after.resident_unlocked_bytes,
                "T230 queued/active retention override changed publication coverage");
        };
        assert_required();
        const auto lease=retention_owner.acquire();
        require(lease && lease->ticket.request_id==ticket.request_id,
            "T230 retention fixture acquisition");
        assert_required();
        retention_owner.complete(*lease);
        // While queued/active, the same root could not be replaced; after
        // completion its generation may be renewed by explicit input admission.
        const auto renewed=retention_owner.admit_prefix_input_with_retention(
            initial[0],prompts[0],available(),permission);
        require(renewed.admitted && renewed.replaced && renewed.generation!=metadata.generation,
            "T230 completed input retention admission did not renew generation");
        const std::array<Index::RetentionDecision,1> renewed_permission{{
            {initial[0],renewed.generation,false,0}}};
        const auto before_drop=retention_owner.stats();
        const auto dropped=retention_owner.trim_prefix_retention(reserve,renewed_permission);
        require(dropped.inputs_current && dropped.budget_met && dropped.evicted==1 &&
            retention_owner.stats().locked_page_bytes==before_drop.locked_page_bytes &&
            metadata.root->matches_tokens(prompts[0]),
            "T230 completed cache eviction released residency or retained reader");
        retention_owner.close();
    }

    // Independent serial reference; these roots are never externally
    // published through the coordinator.
    std::array<std::shared_ptr<const Request>,2> reference_roots;
    std::array<std::vector<std::int64_t>,2> reference_tokens;
    for (int lane = 0; lane < 2; ++lane) {
        contexts[lane]->restore_exact_host_state(
            *initial[lane]->state(), streams[lane].value);
        auto current = initial[lane];
        auto pending = sample_target(*contexts[lane], streams[lane].value);
        for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
            const std::array<std::int64_t,1> tentative{pending};
            auto [updated, verified] = current->verify(
                *contexts[lane], tentative, {}, streams[lane].value);
            require(verified.committed_tokens.size() == 1 &&
                        verified.committed_tokens[0] == pending,
                    "T84 serial reference decision");
            reference_tokens[lane].push_back(pending);
            current = std::move(updated);
            if (ordinal + 1 < kOutputs)
                pending = sample_target(*contexts[lane], streams[lane].value);
        }
        reference_roots[lane] = std::move(current);
    }

    {
        // Real coordinator admission/ledger policy with host-backed stand-ins
        // for represented data; this fixture issues no page CUDA operations.
        using namespace ninfer::exl3;
        using Fill=Exl3DevicePageFill;using Inventory=Exl3ResourceInventory;
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        Coordinator owner(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
        auto limits=Inventory::unlimited();limits[domain]=Fill::metadata_bytes();
        owner.bind_physical_resources({},limits);
        auto page=std::make_shared<Exl3ExactKVPage>();page->rows=64;
        for(int bank=0;bank<16;++bank){page->k[bank].resize(64*1024);page->v[bank].resize(64*1024);}
        auto model_owner=std::make_shared<int>(19),preparation_owner=std::make_shared<int>(20);
        Exl3DevicePageKey key(model_owner,page,{0,false},64);
        Inventory::Requirement required;required.add(Inventory::Domain::host_metadata,1,Fill::metadata_bytes());
        std::shared_ptr<Fill> fill;
        std::weak_ptr<std::vector<std::uint16_t>> retired_storage;
        unsigned allocations=0;
        const auto allocate=[&](auto) {
            ++allocations;
            auto storage=std::make_shared<std::vector<std::uint16_t>>(Fill::elements);retired_storage=storage;
            fill=make_bounded_shared<Fill>(key,storage,*storage);
            Inventory actual;actual.add({fill,0,Inventory::Domain::host_metadata,Fill::metadata_bytes(),{},
                &Fill::attach_own_metadata_credit});return actual;
        };
        owner.allocate_preparation_resources(preparation_owner,required,allocate);
        Inventory::Allocation allocation{fill,0,Inventory::Domain::host_metadata,Fill::metadata_bytes(),{},
            &Fill::attach_own_metadata_credit};
        require(owner.retire_preparation_metadata_to_lifetime(preparation_owner,allocation),
            "page cap fixture failed actual metadata lifetime handoff");
        // Drop the local inventory declaration too: its strong alias must not
        // accidentally stand in for an externally surviving weak control.
        std::weak_ptr<Fill> weak=fill;
        fill.reset();allocation.owner.reset();
        require(weak.expired() && retired_storage.expired() &&
            owner.stats().retained_resource_units->at(domain)==bounded_shared_allocation_bytes<Fill>(),
            "page weak tail lost charge or retained physical stand-in storage");
        bool refused=false;
        try{owner.allocate_preparation_resources(preparation_owner,required,allocate);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && allocations==1 && !fill,
            "page weak control did not prevent over-cap incoming fill before construction");
        weak.reset();
        require(owner.stats().retained_resource_units->at(domain)==0,"page final weak release retained hard-cap charge");
        owner.allocate_preparation_resources(preparation_owner,required,allocate);
        require(allocations==2 && fill && owner.stats().retained_resource_units->at(domain)==Fill::metadata_bytes(),
            "page hard-cap admission failed to recover after final weak release");
        fill.reset();owner.close();
        require(retired_storage.expired() && owner.stats().retained_resource_units->at(domain)==0,
            "page hard-cap retry leaked storage or metadata at close");
    }
    {
        // T012 close must join an in-flight runtime admission before it can
        // clear the physical inventory. The admitted owner is held by both the
        // returned lease and the resident authority until that serialized close.
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Coordinator owner(cache,Coordinator::Policy{3,2,memory.ullTotalPhys-reserve,reserve});
        owner.bind_physical_resources({},Inventory::unlimited());
        std::atomic_bool factory_entered=false,release_factory=false;
        std::atomic_bool closer_started=false,close_returned=false,allocation_returned=false;
        std::weak_ptr<const void> physical_owner;
        std::exception_ptr allocation_error,close_error;
        std::thread allocation([&] {
            try {
                auto lease=owner.allocate_logical_host(sizeof(int),[&] {
                    auto retained=std::make_shared<int>(71);physical_owner=retained;
                    factory_entered.store(true,std::memory_order_release);
                    while(!release_factory.load(std::memory_order_acquire))std::this_thread::yield();
                    return Inventory::Allocation{std::move(retained),0,
                        Inventory::Domain::host_metadata,sizeof(int)};
                },[]() noexcept {});
                require(lease.bytes()==sizeof(int),"close-during-admission changed logical extent");
                allocation_returned.store(true,std::memory_order_release);
            } catch(...) {allocation_error=std::current_exception();}
        });
        while(!factory_entered.load(std::memory_order_acquire))std::this_thread::yield();
        std::thread closer([&] {
            closer_started.store(true,std::memory_order_release);
            try{owner.close();close_returned.store(true,std::memory_order_release);}
            catch(...){close_error=std::current_exception();}
        });
        while(!closer_started.load(std::memory_order_acquire))std::this_thread::yield();
        require(!close_returned.load(std::memory_order_acquire) && !physical_owner.expired(),
            "close crossed an unfinished admission or released its physical owner");
        release_factory.store(true,std::memory_order_release);
        allocation.join();closer.join();
        require(!allocation_error && !close_error && allocation_returned.load(std::memory_order_acquire) &&
            close_returned.load(std::memory_order_acquire) && physical_owner.expired() && owner.stats().closed,
            "close did not serialize after admission or retain/release its exact owner");
    }
    Coordinator coordinator(cache, Coordinator::Policy{
        8, 2, memory.ullTotalPhys - reserve, reserve});
    const bool runtime_credit_fixture=env("NINFER_TEST_KV_RUNTIME_CREDIT")=="1";
    if(runtime_credit_fixture) {
        ninfer::exl3::Exl3ResourceInventory::Totals limits{};limits.fill(1ULL<<30);
        coordinator.bind_physical_resources({},limits);
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Inventory::Requirement preparation;
        preparation.add(Inventory::Domain::cuda_registered_host,1,8);
        preparation.add(Inventory::Domain::host_metadata,1,8);
        bool called=false,missing_refused=false;
        try{coordinator.allocate_preparation_resources({},preparation,[&](auto){called=true;return Inventory{};});}
        catch(const std::invalid_argument&){missing_refused=true;}
        require(missing_refused && !called,"preparation allocation accepted missing request owner");
        auto request_owner=std::make_shared<int>(1);
        {
            std::shared_ptr<const void> unowned(std::shared_ptr<const void>{},request_owner.get());
            bool refused=false;
            try{coordinator.allocate_preparation_resources(unowned,preparation,[&](auto){called=true;return Inventory{};});}
            catch(const std::invalid_argument&){refused=true;}
            require(refused && !called,"preparation allocation accepted nonowning request identity");
            refused=false;
            try{coordinator.retire_preparation_metadata_to_lifetime(unowned,{});}
            catch(const std::invalid_argument&){refused=true;}
            require(refused,"preparation metadata retirement accepted nonowning request identity");
        }
        for(bool after_factory:{false,true}) {
            std::shared_ptr<int> held;std::weak_ptr<int> weak;
            unsigned rollbacks=0,observations=0;bool failed=false,exclusive=false;
            try{coordinator.allocate_preparation_resources(request_owner,preparation,[&](auto)->Inventory {
                held=std::make_shared<int>(4);weak=held;
                if(!after_factory)throw std::runtime_error("preparation result failure");
                Inventory actual;actual.add({held,0,Inventory::Domain::host_metadata,9});return actual;
            },[&]() noexcept {++rollbacks;held.reset();},[&]() noexcept {
                ++observations;
                try{(void)coordinator.acquire();}catch(const std::logic_error&){exclusive=weak.expired();}catch(...){}
            });}catch(const std::exception&){failed=true;}
            require(failed && rollbacks==1 && observations==1 && exclusive && weak.expired() &&
                coordinator.stats().active==0,"preparation rollback lost owner/exclusive prelease boundary");
        }
        for(unsigned attempt=0;attempt<2;++attempt) {
            bool original=false;
            try{coordinator.allocate_preparation_resources(request_owner,preparation,[&](auto)->Inventory {
                called=true;
                auto credits=coordinator.reserve_registration_constructor_credits(8,8);
                require(credits.registration.bytes()==8 && credits.metadata.bytes()==8,
                    "prelease preparation omitted constructor authority");
                throw std::runtime_error("prepared prefix allocation failure");
            });}
            catch(const std::runtime_error& error){original=std::string(error.what())=="prepared prefix allocation failure";}
            require(original && called && coordinator.stats().active==0,
                "preparation reservation required or manufactured a model-state lease");
        }
    }
    const bool reserved_metadata=env("NINFER_TEST_COORDINATOR_METADATA_RESERVED")=="1";
    if(reserved_metadata) {
        if(!runtime_credit_fixture)
            coordinator.bind_physical_resources({},ninfer::exl3::Exl3ResourceInventory::unlimited());
        coordinator.reserve_metadata_startup();
    }
    for(auto fault:{Coordinator::ResidencyFault::before_commit,Coordinator::ResidencyFault::after_first_lock}) {
        coordinator.fail_next_residency_for_test(fault);bool failed=false;
        try{coordinator.admit(initial[0]);}catch(const std::runtime_error& e){
            failed=std::string(e.what()).find("injected coordinator")!=std::string::npos;
        }
        const auto unchanged=coordinator.stats();
        require(failed && unchanged.admitted==0 && unchanged.queued==0 && unchanged.active==0 &&
            unchanged.locked_page_bytes==0 && unchanged.resident_updates==0,
            "failed admission published queue/root or residency");
    }
    std::array<Coordinator::Ticket,3> tickets;
    for (int request = 0; request < 3; ++request)
        tickets[request] = coordinator.admit(initial[request]);
    require(tickets[0].request_id==1,"failed admission consumed request identity");
    require(coordinator.stats().queued == 3 &&
                coordinator.stats().active == 0 &&
                coordinator.high_admitted() == 3,
            "T84 admission state");

    // Deterministic admission/backpressure trace. Extra requests share immutable
    // roots; cancellation must restore the original FIFO before acquisition.
    const auto snapshot_lifetime_bytes=coordinator.owner_lifetime_metadata_bytes_for_test();
    require(snapshot_lifetime_bytes>=initial[0]->state()->snapshot_metadata_bytes()+initial[0]->request_metadata_bytes(),
        "actual coordinator omitted snapshot lifetime metadata");
    std::vector<Coordinator::Ticket> overflow_tickets;
    for (int extra = 0; extra < 5; ++extra)
        overflow_tickets.push_back(coordinator.admit(initial[extra % 3]));
    const auto full = coordinator.stats();
    bool capacity_refused = false;
    try { (void)coordinator.admit(initial[0]); }
    catch (const std::runtime_error&) { capacity_refused = true; }
    require(capacity_refused && coordinator.stats().admitted == full.admitted &&
                coordinator.stats().queued == full.queued &&
                coordinator.stats().locked_page_bytes == full.locked_page_bytes,
            "T84 rejected admission preserves full resident queue");
    for (const auto ticket : overflow_tickets) coordinator.cancel(ticket);
    require(coordinator.owner_lifetime_metadata_bytes_for_test()==snapshot_lifetime_bytes,
        "aliased coordinator admission/cancellation duplicated or dropped snapshot metadata");

    auto first = coordinator.acquire();
    auto second = coordinator.acquire();
    auto blocked = coordinator.acquire();
    require(first && second && !blocked &&
                first->ticket.request_id == tickets[0].request_id &&
                second->ticket.request_id == tickets[1].request_id &&
                coordinator.stats().active == 2 &&
                coordinator.high_active() == 2,
            "T84 FIFO physical admission");
    std::array<Coordinator::Lease,2> leases{*first, *second};
    if(runtime_credit_fixture) {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        {
            const auto outside_refused=[&] {
                bool refused=false;
                try{(void)coordinator.reserve_registration_constructor_credits(128,64);}
                catch(const std::logic_error&){refused=true;}
                require(refused,"registration constructor grant escaped active factory scope");
            };
            outside_refused();
            Inventory::Requirement grant_requirement;
            grant_requirement.add(Inventory::Domain::cuda_registered_host,1,128);
            grant_requirement.add(Inventory::Domain::host_metadata,1,64);
            for(unsigned attempt=0;attempt<2;++attempt) {
                bool original=false;
                try{coordinator.allocate_runtime_resources(*first,grant_requirement,[&](auto)->Inventory {
                    bool wrong_thread=false;
                    std::thread foreign([&] {
                        try{(void)coordinator.reserve_registration_constructor_credits(128,64);}
                        catch(const std::logic_error&){wrong_thread=true;}
                    });foreign.join();
                    require(wrong_thread,"foreign thread consumed runtime registration credit");
                    for(const auto amounts:{std::pair<std::uint64_t,std::uint64_t>{129,64},{128,65}}) {
                        bool refused=false;
                        try{(void)coordinator.reserve_registration_constructor_credits(amounts.first,amounts.second);}
                        catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
                        require(refused,"runtime constructor exceeded one declared domain");
                    }
                    bool wrong_scope=false;
                    try{(void)coordinator.reserve_constructor_credits(1,1);}
                    catch(const std::logic_error&){wrong_scope=true;}
                    require(wrong_scope,"runtime registration scope admitted startup device grant");
                    {
                        auto credits=coordinator.reserve_registration_constructor_credits(128,64);
                        require(credits.registration.bytes()==128 && credits.metadata.bytes()==64,
                            "exact runtime constructor grant changed extent");
                    }
                    bool replay=false;
                    try{(void)coordinator.reserve_registration_constructor_credits(1,1);}
                    catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){replay=true;}
                    require(replay,"released runtime tickets restored already-consumed grant allowance");
                    throw std::runtime_error("prepared registration constructor scope failure");
                });}
                catch(const std::runtime_error& error){original=std::string(error.what())=="prepared registration constructor scope failure";}
                require(original && coordinator.compute_leases_current(leases),"registration constructor scope failed to unwind cleanly");
                outside_refused();
            }
        }
        {
            bool outside=false;
            try{(void)coordinator.reserve_runtime_constructor_credits(1,1);}
            catch(const std::logic_error&){outside=true;}
            require(outside,"runtime device grant escaped its factory");
            Inventory::Requirement mixed;
            mixed.add(Inventory::Domain::device,1,128);
            mixed.add(Inventory::Domain::cuda_registered_host,1,64);
            mixed.add(Inventory::Domain::host_metadata,1,96);
            for(unsigned attempt=0;attempt<2;++attempt) {
                bool original=false;
                try{coordinator.allocate_runtime_resources(*first,mixed,[&](auto)->Inventory {
                    bool foreign_refused=false;
                    std::thread foreign([&] {
                        try{(void)coordinator.reserve_runtime_constructor_credits(128,64);}
                        catch(const std::logic_error&){foreign_refused=true;}
                    });foreign.join();
                    require(foreign_refused,"foreign thread consumed runtime device grant");
                    bool device_limit=false;
                    try{(void)coordinator.reserve_runtime_constructor_credits(129,1);}
                    catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){device_limit=true;}
                    require(device_limit,"runtime device grant exceeded physical declaration");
                    {
                        auto registration=coordinator.reserve_registration_constructor_credits(64,32);
                        bool metadata_limit=false;
                        try{(void)coordinator.reserve_runtime_constructor_credits(128,65);}
                        catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){metadata_limit=true;}
                        require(metadata_limit,"runtime grants reused registration metadata allowance");
                        auto device=coordinator.reserve_runtime_constructor_credits(128,64);
                        require(device.device.bytes()==128 && device.metadata.bytes()==64 &&
                            registration.registration.bytes()==64 && registration.metadata.bytes()==32,
                            "mixed-domain constructor grants changed exact extents");
                    }
                    bool replay=false;
                    try{(void)coordinator.reserve_runtime_constructor_credits(1,1);}
                    catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){replay=true;}
                    require(replay,"released device tickets restored consumed runtime allowance");
                    throw std::runtime_error("prepared mixed constructor failure");
                });}catch(const std::runtime_error& error){original=std::string_view(error.what())=="prepared mixed constructor failure";}
                require(original && coordinator.compute_leases_current(leases),"mixed constructor failure lost live leases");
            }
        }
        Inventory::Requirement required;required.configuration=71;
        required.add(Inventory::Domain::host_metadata,1,sizeof(int));
        bool called=false;auto stale=*first;++stale.acquisition;
        bool refused=false;
        const auto metadata_before=coordinator.stats().retained_resource_units;
        bool stale_payload=false,stale_tracking=false;
        try{(void)coordinator.reserve_snapshot_payload(stale,1);}
        catch(const std::invalid_argument&){stale_payload=true;}
        try{coordinator.track_snapshot_payload(stale,{},0,nullptr,1);}
        catch(const std::invalid_argument&){stale_tracking=true;}
        require(stale_payload && stale_tracking && coordinator.stats().retained_resource_units==metadata_before,
            "stale payload lease allocated tracking metadata or reached owner interpretation");
        // These must reject the stale acquisition before interpreting even an
        // allocation descriptor, transferring credits or invoking release code.
        bool metadata_retirement_refused=false,physical_retirement_refused=false,release_called=false;
        try{(void)coordinator.retire_runtime_metadata_to_lifetime(stale,Inventory::Allocation{});}
        catch(const std::invalid_argument& error){metadata_retirement_refused=std::string_view(error.what())==
            "metadata lifetime retirement stale physical lease";}
        try{(void)coordinator.retire_runtime_resource(stale,Inventory::Allocation{},[&](const auto&) noexcept {
            release_called=true;return true;
        });}
        catch(const std::invalid_argument& error){physical_retirement_refused=std::string_view(error.what())==
            "runtime retirement stale physical lease";}
        require(metadata_retirement_refused && physical_retirement_refused && !release_called &&
            coordinator.stats().retained_resource_units==metadata_before && coordinator.compute_leases_current(leases),
            "stale retirement moved credits, invoked release or disturbed active leases");
        try{(void)coordinator.reserve_snapshot_metadata(&stale,{},1,true);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused && coordinator.stats().retained_resource_units==metadata_before,
            "stale metadata ceiling diagnostic mutated retained resource accounting");
        // A valid reservation must still have headroom: unchanged usage alone
        // would not detect a stale caller silently tightening the ceiling.
        {
            auto credit=coordinator.reserve_snapshot_metadata(&*first,{},1);
            require(credit.bytes()==1,"stale metadata diagnostic consumed valid lease headroom");
        }
        refused=false;
        try{coordinator.allocate_runtime_resources(stale,required,[&](auto){called=true;return Inventory{};});}
        catch(const std::invalid_argument&){refused=true;}
          require(refused && !called,"stale runtime credit invoked allocator");
          std::weak_ptr<int> rolled_back_owner;
          bool factory_failed=false;
          try {
              coordinator.allocate_runtime_resources(*first,required,[&](auto)->Inventory {
                  auto owner=std::make_shared<int>(8);rolled_back_owner=owner;
                  throw std::runtime_error("prepared runtime allocation failure");
              });
          } catch(const std::runtime_error& error) {
              factory_failed=std::string(error.what())=="prepared runtime allocation failure";
          }
          require(factory_failed && rolled_back_owner.expired() && coordinator.compute_leases_current(leases),
              "runtime factory failure retained temporary owner or invalidated leases");
          // Enter a mutator after rollback: an uncleared runtime allocation flag
          // would reject reentry (or strand a peer), instead of restoring service.
          require(!coordinator.acquire(),"runtime rollback changed physical admission capacity");
          for(bool after_factory:{false,true}) {
              std::shared_ptr<int> held;
              std::weak_ptr<int> weak;
              unsigned rollbacks=0,observations=0;bool exclusive=false,observed_release=false,failed=false;
              try {
                  coordinator.allocate_runtime_resources(*first,required,[&](auto)->Inventory {
                      held=std::make_shared<int>(12);weak=held;
                      if(!after_factory)throw std::runtime_error("runtime held factory failure");
                      Inventory actual;actual.add({held,0,Inventory::Domain::host_metadata,sizeof(int)+1});
                      return actual; // Deliberate reservation overrun after returning an owner.
                  },[&]() noexcept {
                      ++rollbacks;
                      try{(void)coordinator.acquire();}catch(const std::logic_error&){exclusive=true;}catch(...){}
                      held.reset();
                  },[&]() noexcept {
                      ++observations;observed_release=weak.expired();
                      try{(void)coordinator.acquire();exclusive=false;}catch(const std::logic_error&){}catch(...){exclusive=false;}
                  });
              } catch(const std::exception&){failed=true;}
              require(failed && rollbacks==1 && observations==1 && observed_release && exclusive && weak.expired() &&
                  coordinator.compute_leases_current(leases),"runtime rollback escaped exclusivity or retained caller owner");
              require(!coordinator.acquire(),"runtime rollback failed to reopen admission");
          }
          std::shared_ptr<int> runtime_owner;
          coordinator.allocate_runtime_resources(*first,required,[&](auto configuration) {
            require(configuration==71 && coordinator.compute_leases_current(leases),"runtime allocator lost read-only lease access");
            bool nested=false;try{coordinator.acquire();}catch(const std::logic_error&){nested=true;}
            require(nested,"runtime allocator allowed reentrant coordinator mutation");
            runtime_owner=std::make_shared<int>(9);
            Inventory actual;actual.add({runtime_owner,0,Inventory::Domain::host_metadata,sizeof(int)});
            return actual;
        });
        require(coordinator.compute_leases_current(leases),"runtime credit replaced active lease");
        std::weak_ptr<const void> retired=runtime_owner,foreign;
        {auto unrelated=std::make_shared<int>(10);foreign=unrelated;}
        auto tracker=std::make_shared<int>(11);
        Inventory::Allocation payload{runtime_owner,0,Inventory::Domain::host_metadata,sizeof(int)};
        const Inventory::Allocation tracking{tracker,1,Inventory::Domain::host_metadata,sizeof(int)};
        coordinator.transfer_retired_metadata(*first,payload,tracking);
        refused=false;try{coordinator.collect_retired_metadata(stale,tracking,retired);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused,"stale lease collected runtime metadata");
        refused=false;try{coordinator.collect_retired_metadata(*first,tracking,foreign);}
        catch(const std::invalid_argument&){refused=true;}
        require(refused && coordinator.compute_leases_current(leases),"foreign expired owner changed coordinator metadata authority");
        require(!coordinator.collect_retired_metadata(*first,tracking,retired),"live runtime borrower was ignored");
        payload.owner.reset();runtime_owner.reset();
        require(coordinator.collect_retired_metadata(*first,tracking,retired),"expired runtime metadata not collected");
        require(coordinator.compute_leases_current(leases),"metadata collection invalidated physical leases");
    }
    for (int lane = 0; lane < 2; ++lane)
        contexts[lane]->restore_exact_host_state(
            *leases[lane].root->state(), streams[lane].value);
    std::array<std::int64_t,2> pending{
        sample_target(*contexts[0], streams[0].value),
        sample_target(*contexts[1], streams[1].value)};
    std::array<std::vector<std::int64_t>,2> published_tokens;
    std::ofstream publications(output / "publications.csv");
    std::ofstream lifecycle(output / "lifecycle.csv");
    std::ofstream memory_out(output / "memory.csv");
    require(publications.good() && lifecycle.good() && memory_out.good(),
            "T84 evidence open");
    publications << "ordinal,lane,token,sequence,generation,resident_page_bytes\n";
    lifecycle << "phase,queued,active,admitted,publications,cancellations,"
                 "high_queued,high_active,high_admitted,pass\n";
    memory_out << "phase,resident_payload_bytes,locked_payload_bytes,"
                  "locked_page_bytes,cache_payload_bytes,cache_identity_bytes,"
                  "device_free_bytes,physical_reserve_bytes\n";
    const auto write_lifecycle = [&](const char* phase, bool pass) {
        const auto stats = coordinator.stats();
        lifecycle << phase << ',' << stats.queued << ',' << stats.active << ','
            << stats.admitted << ',' << stats.publications << ','
            << stats.cancellations << ',' << coordinator.high_queued() << ','
            << coordinator.high_active() << ',' << coordinator.high_admitted()
            << ',' << pass << '\n';
    };
    write_lifecycle("acquired", true);

    bool wrong_token_rejected = false;
    bool wrong_parent_rejected = false;
    bool stale_rejected = false;
    for (int ordinal = 0; ordinal < kOutputs; ++ordinal) {
        std::array<std::shared_ptr<const Request>,2> updated;
        std::array<ninfer::exl3::Exl3OuterReferenceResult,2> verified;
        std::array<std::exception_ptr,2> errors;
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::thread workers[2];
        for (int lane = 0; lane < 2; ++lane)
            workers[lane] = std::thread([&, lane] {
                ready.fetch_add(1, std::memory_order_release);
                while (!go.load(std::memory_order_acquire))
                    std::this_thread::yield();
                try {
                    const std::array<std::int64_t,1> tentative{pending[lane]};
                    auto result = leases[lane].root->verify(
                        *contexts[lane], tentative, {}, streams[lane].value);
                    updated[lane] = std::move(result.first);
                    verified[lane] = std::move(result.second);
                } catch (...) { errors[lane] = std::current_exception(); }
            });
        while (ready.load(std::memory_order_acquire) != 2)
            std::this_thread::yield();
        go.store(true, std::memory_order_release);
        for (auto& worker : workers) worker.join();
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        if (ordinal == 0) {
            const auto before=coordinator.stats();
            const auto scratch_before=coordinator.publication_scratch_snapshot_for_test();
            for(auto fault:{Coordinator::ResidencyFault::before_commit,Coordinator::ResidencyFault::after_first_lock}) {
                coordinator.fail_next_residency_for_test(fault);bool failed=false;
                std::array<Coordinator::Publication,2> output;
                output[0].sequence=991;output[1].sequence=992;
                try{coordinator.publish_batch_into(leases,updated,pending,output);}catch(const std::runtime_error& e){
                    failed=std::string(e.what()).find("injected coordinator")!=std::string::npos;
                }
                const auto after=coordinator.stats();
                require(failed && after.publications==before.publications && after.active==before.active &&
                    after.locked_page_bytes==before.locked_page_bytes && coordinator.compute_leases_current(leases) &&
                    output[0].sequence==991 && output[1].sequence==992 &&
                    coordinator.publication_scratch_empty_for_test(),
                    "failed batch publication changed live authority");
            }
            const auto scratch_after=coordinator.publication_scratch_snapshot_for_test();
            require(!scratch_after.publication_active && !scratch_after.root_active &&
                scratch_after.selected_storage==scratch_before.selected_storage &&
                scratch_after.replacement_storage==scratch_before.replacement_storage &&
                scratch_after.root_storage==scratch_before.root_storage &&
                scratch_after.selected_capacity==scratch_before.selected_capacity &&
                scratch_after.replacement_capacity==scratch_before.replacement_capacity &&
                scratch_after.root_capacity==scratch_before.root_capacity &&
                scratch_after.publication_epoch==scratch_before.publication_epoch+2 &&
                scratch_after.root_epoch==scratch_before.root_epoch+2,
                "failed publication replaced or shared reserved traversal scratch");
            auto wrong = pending;
            wrong[0] = wrong[0] == 248319 ? 0 : wrong[0] + 1;
            try { (void)coordinator.publish_batch(leases, updated, wrong); }
            catch (const std::invalid_argument&) { wrong_token_rejected = true; }
            const std::array<std::shared_ptr<const Request>,2> swapped{
                updated[1], updated[0]};
            try { (void)coordinator.publish_batch(leases, swapped, pending); }
            catch (const std::invalid_argument&) { wrong_parent_rejected = true; }
        }
        const auto old_leases = leases;
        std::array<Coordinator::Publication,2> published;
        published[0].sequence=991;published[1].sequence=992;
        bool output_extent_rejected=false;
        try{coordinator.publish_batch_into(leases,updated,pending,
            std::span<Coordinator::Publication>(published).first(1));}
        catch(const std::invalid_argument&){output_extent_rejected=true;}
        require(output_extent_rejected && published[0].sequence==991 && published[1].sequence==992 &&
            coordinator.compute_leases_current(leases) && coordinator.publication_scratch_empty_for_test(),
            "T024 invalid output span changed publication or retained scratch roots");
        const auto scratch_before_publish=
            coordinator.publication_scratch_snapshot_for_test();
        coordinator.publish_batch_into(leases,updated,pending,published);
        const auto scratch_after_publish=
            coordinator.publication_scratch_snapshot_for_test();
        require(coordinator.publication_scratch_empty_for_test() &&
            !scratch_after_publish.publication_active && !scratch_after_publish.root_active &&
            scratch_after_publish.selected_storage==scratch_before_publish.selected_storage &&
            scratch_after_publish.replacement_storage==scratch_before_publish.replacement_storage &&
            scratch_after_publish.root_storage==scratch_before_publish.root_storage &&
            scratch_after_publish.selected_capacity==scratch_before_publish.selected_capacity &&
            scratch_after_publish.replacement_capacity==scratch_before_publish.replacement_capacity &&
            scratch_after_publish.root_capacity==scratch_before_publish.root_capacity &&
            scratch_after_publish.publication_epoch==
                scratch_before_publish.publication_epoch+1 &&
            scratch_after_publish.root_epoch==scratch_before_publish.root_epoch+1,
            "T024 publication replaced or retained mutable traversal scratch");
        for (int lane = 0; lane < 2; ++lane) {
            require(verified[lane].verification_rows == 1 &&
                        verified[lane].executed_rows == 1 &&
                        verified[lane].replay_rows == 0 &&
                        verified[lane].root_restores == 0,
                    "T84 authoritative work accounting");
            published_tokens[lane].push_back(pending[lane]);
            leases[lane] = published[lane].lease;
            publications << ordinal << ',' << (lane ? 'B' : 'A') << ','
                << published[lane].token << ',' << published[lane].sequence
                << ',' << published[lane].lease.ticket.generation << ','
                << published[lane].residency.page_bytes << '\n';
        }
        if (ordinal == 0) {
            try { (void)coordinator.publish(
                old_leases[0], updated[0], pending[0]); }
            catch (const std::invalid_argument&) { stale_rejected = true; }
        }
        if (ordinal + 1 < kOutputs)
            for (int lane = 0; lane < 2; ++lane)
                pending[lane] = sample_target(
                    *contexts[lane], streams[lane].value);
    }
    for (int lane = 0; lane < 2; ++lane)
        require(published_tokens[lane] == reference_tokens[lane] &&
                    leases[lane].root->token_suffix() ==
                        reference_roots[lane]->token_suffix() &&
                    leases[lane].root->same_taps(*reference_roots[lane]) &&
                    leases[lane].root->state()->same_payload(
                        *reference_roots[lane]->state()),
                "T84 coordinator output/state oracle");
    require(wrong_token_rejected && wrong_parent_rejected && stale_rejected,
            "T84 invalid publication rejection");
    write_lifecycle("published", true);

    coordinator.yield(leases[0]);
    coordinator.yield(leases[1]);
    auto fairness_third = coordinator.acquire();
    require(fairness_third && fairness_third->ticket.request_id ==
                tickets[2].request_id,
            "T84 FIFO yielded fairness third");
    coordinator.yield(*fairness_third);
    auto fairness_first = coordinator.acquire();
    require(fairness_first && fairness_first->ticket.request_id ==
                tickets[0].request_id,
            "T84 FIFO yielded fairness first");
    require(fairness_first->ticket.generation == leases[0].ticket.generation &&
                fairness_first->acquisition != leases[0].acquisition,
            "T84 reacquisition isolates workers without invalidating cancellation ticket");
    bool stale_yield_refused = false;
    bool stale_complete_refused = false;
    try { coordinator.yield(leases[0]); }
    catch (const std::invalid_argument&) { stale_yield_refused = true; }
    try { coordinator.complete(leases[0]); }
    catch (const std::invalid_argument&) { stale_complete_refused = true; }
    require(stale_yield_refused && stale_complete_refused && coordinator.stats().active == 1,
            "T84 old acquisition cannot mutate current active request");
    coordinator.yield(*fairness_first);
    coordinator.cancel(tickets[2]);
    require(coordinator.stats().cancellations == 1 + overflow_tickets.size(),
            "T84 queued cancellation");

    auto cancel_lease = coordinator.acquire();
    require(cancel_lease && cancel_lease->ticket.request_id ==
                tickets[1].request_id,
            "T84 worker-boundary cancellation acquisition");
    contexts[1]->restore_exact_host_state(
        *cancel_lease->root->state(), streams[1].value);
    const auto cancel_pending = sample_target(*contexts[1], streams[1].value);
    const std::array<std::int64_t,1> cancel_token{cancel_pending};
    auto [late_root, late_verified] = cancel_lease->root->verify(
        *contexts[1], cancel_token, {}, streams[1].value);
    require(late_verified.verification_rows == 1,
            "T84 canceled worker verification");
    coordinator.cancel(cancel_lease->ticket, true);
    bool canceled_stale_rejected = false;
    try { (void)coordinator.publish(*cancel_lease, late_root, cancel_pending); }
    catch (const std::invalid_argument&) { canceled_stale_rejected = true; }
    require(canceled_stale_rejected && coordinator.stats().active == 0 &&
                coordinator.stats().cancellations == 2 + overflow_tickets.size(),
            "T84 worker cancellation stale publication");
    write_lifecycle("canceled", true);

    std::array<std::shared_ptr<const Request>,3> visible{
        leases[0].root, leases[1].root, initial[2]};
    auto cache_roots = cache.roots();
    std::vector<std::shared_ptr<const Request>> observed = cache_roots;
    observed.push_back(visible[0]);
    Exl3HostResidencyProbe probe;
    Request::visit_host_allocations(observed,
        [&](const void* data, std::size_t bytes) { probe.add(data, bytes); },
        false);
    const auto resident = probe.measure();
    require(resident.resident_tensor_bytes == resident.allocated_union_bytes &&
                resident.locked_tensor_bytes == resident.allocated_union_bytes,
            "T84 final visible root residency");
    std::size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes),
               "T84 final device memory");
    require(free_bytes >= (1ULL << 30), "T84 device reserve");
    const auto stats = coordinator.stats();
    memory_out << "before_close," << resident.resident_tensor_bytes << ','
        << resident.locked_tensor_bytes << ',' << stats.locked_page_bytes << ','
        << cache.storage_stats().payload_allocated_bytes << ','
        << cache.storage_stats().identity_allocated_bytes << ',' << free_bytes
        << ',' << reserve << '\n';
    coordinator.close();
    require(coordinator.stats().closed && coordinator.stats().active == 0 &&
                coordinator.stats().locked_page_bytes == 0,
            "T84 coordinator close");
    std::uint64_t retained_snapshot_metadata=0;
    std::vector<std::shared_ptr<const int>> retained_revisions;
    std::vector<std::shared_ptr<const void>> retained_nested_owners;
    const auto retain_metadata=[&](const std::shared_ptr<const void>& owner,std::uint64_t bytes) {
        if(std::find(retained_nested_owners.begin(),retained_nested_owners.end(),owner)==retained_nested_owners.end()) {
            retained_nested_owners.push_back(owner);retained_snapshot_metadata+=bytes;
        }
    };
    for(std::size_t i=0;i<initial.size();++i) {
        bool root_seen=false;
        for(std::size_t j=0;j<i;++j)root_seen|=initial[j]==initial[i];
        if(!root_seen)retained_snapshot_metadata+=initial[i]->request_metadata_bytes();
        initial[i]->visit_revision_owners([&](const auto& revision) {
            if(std::find(retained_revisions.begin(),retained_revisions.end(),revision)==retained_revisions.end()) {
                retained_revisions.push_back(revision);
                retained_snapshot_metadata+=ninfer::exl3::bounded_shared_allocation_bytes<int>();
            }
        });
        bool seen=false;
        for(std::size_t j=0;j<i;++j)seen|=initial[j]->state()==initial[i]->state();
        if(!seen)retained_snapshot_metadata+=initial[i]->state()->snapshot_metadata_bytes();
        initial[i]->state()->visit_page_metadata_owners([&](const auto& page){
            retain_metadata(page,ninfer::exl3::Exl3ExactKVPage::metadata_bytes());
        });
        initial[i]->visit_tap_block_owners([&](const auto& block){
            retain_metadata(block,ninfer::exl3::Exl3VeriCacheRequest::tap_block_metadata_bytes());
        });
        initial[i]->visit_token_control_owners([&](const auto& node){
            retain_metadata(node,ninfer::exl3::Exl3TokenHistory::node_metadata_bytes());
        });
        if(const auto& ring=initial[i]->projected_metadata_owner()) {
            retain_metadata(ring,ring->metadata_bytes());
            ring->visit_page_metadata_owners([&](const auto& page){
                retain_metadata(page,ninfer::exl3::Exl3DraftHostRing::page_metadata_bytes());
            });
        }
        if(const auto& identity=initial[i]->prepared_metadata_owner())retain_metadata(identity,identity->metadata_bytes());
    }
    require(coordinator.owner_lifetime_metadata_bytes_for_test()>=retained_snapshot_metadata &&
        coordinator.stats().retained_resource_units &&
        (*coordinator.stats().retained_resource_units)[static_cast<unsigned>(
            ninfer::exl3::Exl3ResourceInventory::Domain::host_metadata)]>=retained_snapshot_metadata,
        "coordinator close dropped request or nested metadata owned by retained real roots");
    write_lifecycle("closed", true);
    publications.flush(); lifecycle.flush(); memory_out.flush();
    require(publications.good() && lifecycle.good() && memory_out.good(),
            "T84 evidence flush");
    std::cout << "T84_SERVING_COORDINATOR PASS admitted_high=8 active_high=2"
              << " publications=16 cancellations=7 exact=1 fifo=1"
              << " stale_rejected=1 wrong_token_rejected=1"
              << " wrong_parent_rejected=1 resident_locked=1"
              << " free_bytes=" << free_bytes
              << " output=" << output.string() << std::endl;
}
