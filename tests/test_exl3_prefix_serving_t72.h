#pragma once

#include <atomic>
#include <thread>
#include <condition_variable>
#include <mutex>
#include "exl3/vericache_serving_coordinator.h"

void run_prefix_serving_t72_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Index=ninfer::exl3::Exl3VeriCachePrefixIndex;
    using Cache=ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity=ninfer::exl3::Exl3VeriCacheServingIdentity;
    const auto same_storage=[](const Index::StorageStats& a,const Index::StorageStats& b) {
        return a.entries==b.entries && a.payload_allocated_bytes==b.payload_allocated_bytes &&
            a.identity_allocated_bytes==b.identity_allocated_bytes && a.accounted_bytes==b.accounted_bytes &&
            a.inline_lookup_metadata_bytes==b.inline_lookup_metadata_bytes;
    };
    constexpr std::size_t prefix_rows=321,tail_rows=64;
    constexpr std::uint64_t reserve=8ULL<<30,unbounded=64ULL<<30;
    require(target.max_context()>=static_cast<int>(prefix_rows+tail_rows) &&
        source.size()>=prefix_rows+tail_rows+8,"T72 serving fixture extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto serving=target.create_context(true);serving->prepare_continuation(8);
    auto fresh=target.create_context(true);fresh->prepare_continuation(8);
    auto builder=std::shared_ptr<Exl3TextContext>(target.create_context(true));
    builder->prepare_continuation(8);
    const Identity identity{"single-user-process-1","SC_6.00bpw_H6_V6+artifact-pin",
        "tokenizer-sha256-pin","qualified-config-sha256-pin","text-only-v1"};
    bool refused=false;
    try {Cache bad(Cache::Policy{4,64,unbounded,reserve},Identity{});} catch(const std::invalid_argument&) {refused=true;}
    require(refused,"T72 empty identity accepted");
    refused=false;
    try {Cache bad(Cache::Policy{4,64,0,reserve},identity);} catch(const std::invalid_argument&) {refused=true;}
    require(refused,"T72 zero byte budget accepted");

    Cache cache(Cache::Policy{4,64,unbounded,reserve},identity);
    const std::vector<std::int64_t> common(source.begin(),source.begin()+prefix_rows);
    std::vector<std::int64_t> tail_a(source.begin()+prefix_rows,
        source.begin()+prefix_rows+tail_rows);
    auto tail_b=tail_a;tail_b.front()=(tail_b.front()+17)%kVocab;
    const auto make_prompt=[&](const std::vector<std::int64_t>& tail) {
        auto prompt=common;prompt.insert(prompt.end(),tail.begin(),tail.end());return prompt;
    };
    const auto prove=[&](const std::vector<std::int64_t>& prompt,bool expected_hit) {
        const auto cached=cache.prepare(*serving,prompt,prefix_rows,unbounded,32);
        const auto oracle=Request::initialize(*fresh,prompt,32);
        require(cached.metrics.cache_hit==expected_hit &&
            cached.metrics.prompt_tokens==prompt.size() &&
            cached.metrics.reused_prompt_tokens+cached.metrics.executed_prompt_tokens==prompt.size() &&
            cached.metrics.reused_prompt_tokens==(expected_hit?prefix_rows:0) &&
            cached.metrics.executed_prompt_tokens==(expected_hit?tail_rows:prefix_rows+tail_rows) &&
            cached.request->token_suffix()==oracle->token_suffix() &&
            cached.request->same_taps(*oracle) &&
            cached.request->state()->same_payload(*oracle->state()),
            "T72 cached serving request differs from fresh authority");
        return cached;
    };
    const auto first=prove(make_prompt(tail_a),false);
    const auto cold_retention=cache.retention_metadata();
    require(cold_retention.size()==1 && cold_retention[0].preparation_microseconds.has_value() &&
        !cold_retention[0].observed_accesses,"cold serving preparation observation missing or invented accesses");
    require(first.metrics.admitted && first.metrics.cache_storage.entries==1 &&
        first.metrics.cache_storage.payload_allocated_bytes>0 &&
        first.metrics.cache_storage.identity_allocated_bytes>0 &&
        first.metrics.cache_storage.accounted_bytes==
            first.metrics.cache_storage.payload_allocated_bytes+
            first.metrics.cache_storage.identity_allocated_bytes,
        "T72 first admission/accounting");
    const auto second=prove(make_prompt(tail_b),true);
    {
        const auto& root=*first.request;
        const auto model=root.state()->model_identity();
        const int rope=root.state()->rope_offset();
        const auto middle=root.state()->shared_kv_page(64);
        require(middle && root.owns_shared_text_page(middle,model,rope) &&
            second.request->owns_shared_text_page(middle,model,rope),
            "shared middle page lost acquired-root ancestry across distinct tails");
        const auto different=std::make_shared<const ninfer::exl3::Exl3ExactKVPage>(*middle);
        require(!root.owns_shared_text_page(different,model,rope),
            "equal page values and offsets substituted for immutable ancestor ownership");
        const std::shared_ptr<const ninfer::exl3::Exl3ExactKVPage> foreign_owner(
            middle.get(),[](const auto*){});
        require(!root.owns_shared_text_page(foreign_owner,model,rope) &&
            !root.owns_shared_text_page(middle,model,rope+1) &&
            !root.owns_shared_text_page(middle,std::make_shared<const int>(0),rope),
            "page attachment accepted false owner, changed positions or model");
        require(!root.state()->shared_kv_page(-1) && !root.state()->shared_kv_page(65) &&
            !root.state()->shared_kv_page(384) &&
            !root.owns_shared_text_page({},model,rope),
            "page attachment accepted nonpage frontier or incomplete tail");
        const auto changed_tail=second.request->state()->shared_kv_page(320);
        require(changed_tail && !root.owns_shared_text_page(changed_tail,model,rope),
            "changed branch tail reused equal-offset page without ancestor proof");
        builder->restore_exact_host_state(*root.state());
        const auto first_tap_binding=root.committed_tap_binding(builder,17,23);
        const auto repeated_tap_binding=root.committed_tap_binding(builder,17,23);
        builder->restore_exact_host_state(*second.request->state());
        const auto sibling_tap_binding=second.request->committed_tap_binding(builder,17,23);
        auto false_owner_tap_binding=first_tap_binding;
        false_owner_tap_binding.root_revision=std::shared_ptr<const void>(
            first_tap_binding.root_revision.get(),[](const void*){});
        require(first_tap_binding.root_position==root.state()->position() &&
            first_tap_binding.root_revision==repeated_tap_binding.root_revision &&
            first_tap_binding.root_revision!=sibling_tap_binding.root_revision &&
            root.owns_committed_tap_root(first_tap_binding) &&
            !root.owns_committed_tap_root(sibling_tap_binding) &&
            !root.owns_committed_tap_root(false_owner_tap_binding),
            "tap attachment lost exact root revision/frontier ancestry");
        builder->restore_exact_host_state(*root.state());
        bool changed_frontier_refused=false;
        builder->decode(root.token_suffix().back());
        try{(void)root.committed_tap_binding(builder,17,23);}
        catch(const std::invalid_argument&){changed_frontier_refused=true;}
        require(changed_frontier_refused,
            "tap attachment accepted a context beyond the required root frontier");
    }
    const auto hit_retention=cache.retention_metadata();
    {
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Cache logical_cache(Cache::Policy{4,64,unbounded,reserve},identity);
        Coordinator authority(logical_cache,{1,1,unbounded,reserve});
        auto limits=Inventory::unlimited();limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=64;
        authority.bind_physical_resources({},limits);
        std::shared_ptr<std::array<std::byte,64>> storage;
        bool invoked=false;
        const auto allocate=[&](std::uint64_t bytes) {
            return authority.allocate_logical_host(bytes,[&] {
                invoked=true;storage=std::make_shared<std::array<std::byte,64>>();
                return Inventory::Allocation{storage,0,Inventory::Domain::host_metadata,64};
            },[&]{storage.reset();});
        };
        bool refused=false;try{auto excess=allocate(65);}catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && !invoked,"logical output credit refusal reached allocator");
        for(unsigned fault=0;fault<3;++fault) {
            std::weak_ptr<const void> failed_owner;
            unsigned rollbacks=0;bool failed=false;
            try {
                auto rejected=authority.allocate_logical_host(64,[&]() -> Inventory::Allocation {
                    storage=std::make_shared<std::array<std::byte,64>>();failed_owner=storage;
                    if(fault==0)throw std::runtime_error("injected logical result construction failure");
                    return Inventory::Allocation{storage,0,
                        fault==2?Inventory::Domain::device:Inventory::Domain::host_metadata,
                        fault==1?63u:64u};
                },[&]{++rollbacks;storage.reset();});
            } catch(const std::runtime_error& error) {
                failed=fault==0 && std::string_view(error.what())=="injected logical result construction failure";
            } catch(const std::invalid_argument& error) {
                failed=std::string_view(error.what())==(fault==1?
                    "resource reservation actual extent/domain mismatch":"logical host allocation domain");
            }
            require(failed && rollbacks==1 && failed_owner.expired(),
                "failed logical output allocation retained an owner or skipped rollback");
            auto retry=allocate(64);std::weak_ptr<const void> retry_owner=storage;
            require(authority.retire_logical_host(retry,[&]{storage.reset();return true;}) && retry_owner.expired(),
                "failed logical output allocation consumed credit or poisoned retry");
        }
        auto lease=allocate(64);std::weak_ptr<const void> retained=storage;
        require(lease.bytes()==64 && !retained.expired(),"logical output reservation lost host ownership");
        require(!authority.retire_logical_host(lease,[]{return false;}) && !retained.expired(),
            "pending logical output retirement released its allocation");
        invoked=false;refused=false;try{auto excess=allocate(1);}catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && !invoked,"pending output retirement released reservation credit");
        auto moved=std::move(lease);
        refused=false;try{authority.retire_logical_host(lease,[]{return true;});}catch(const std::invalid_argument&){refused=true;}
        require(refused,"moved logical host lease retained retirement authority");
        require(authority.retire_logical_host(moved,[&]{storage.reset();return true;}) && retained.expired(),
            "logical output final retirement retained storage");
        auto retry=allocate(64);
        require(authority.retire_logical_host(retry,[&]{storage.reset();return true;}),"logical output credit was not reusable");
        using Extent=ninfer::exl3::Exl3HostResidentSet::ReservationExtent;
        std::shared_ptr<std::array<std::byte,32>> small,peer;
        auto bounded=authority.allocate_logical_host(64,[&] {
            small=std::make_shared<std::array<std::byte,32>>();
            return Inventory::Allocation{small,0,Inventory::Domain::host_metadata,32};
        },[&]{small.reset();},Extent::at_most);
        require(bounded.bytes()==32,"bounded logical output charged unused ceiling");
        auto remainder=authority.allocate_logical_host(32,[&] {
            peer=std::make_shared<std::array<std::byte,32>>();
            return Inventory::Allocation{peer,0,Inventory::Domain::host_metadata,32};
        },[&]{peer.reset();});
        require(authority.retire_logical_host(remainder,[&]{peer.reset();return true;}) &&
            authority.retire_logical_host(bounded,[&]{small.reset();return true;}),
            "bounded logical output failed to return actual extent credit");
        std::weak_ptr<const void> oversized_owner;refused=false;
        try {
            auto oversized=authority.allocate_logical_host(32,[&] {
                storage=std::make_shared<std::array<std::byte,64>>();oversized_owner=storage;
                return Inventory::Allocation{storage,0,Inventory::Domain::host_metadata,64};
            },[&]{storage.reset();},Extent::at_most);
        } catch(const std::invalid_argument& error) {
            refused=std::string_view(error.what())=="bounded resource reservation exceeded ceiling/domain";
        }
        require(refused && oversized_owner.expired(),"bounded logical output accepted or retained excess extent");
        auto full_retry=allocate(64);
        require(authority.retire_logical_host(full_retry,[&]{storage.reset();return true;}),
            "bounded logical refusal retained reservation debt");
        authority.close();
    }
    {
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        std::uint64_t root_budget=0;
        {
            Cache measured(Cache::Policy{4,64,unbounded,reserve},identity);
            Coordinator authority(measured,{4,1,unbounded,reserve});
            authority.admit(first.request);
            root_budget=authority.stats().locked_page_bytes;
            authority.close();
        }
        require(root_budget>0,"resident budget fixture has no authoritative backing");
        Cache bounded(Cache::Policy{4,64,unbounded,reserve},identity);
        Coordinator authority(bounded,{4,1,root_budget,reserve});
        const auto original=authority.admit(first.request);
        auto active=authority.acquire();
        require(active && active->ticket.request_id==original.request_id,
            "resident budget fixture lost original acquisition");
        const auto before=authority.stats();
        for(unsigned kind=0;kind<3;++kind) {
            bool exhausted=false;
            try {
                if(kind==0)authority.admit(second.request);
                else if(kind==1) {
                    const std::array<std::shared_ptr<const Request>,2> roots{first.request,second.request};
                    authority.admit_batch(roots);
                } else authority.complete_and_admit(*active,second.request);
            } catch(const ninfer::exl3::Exl3HostResidentBudgetExhausted& error) {
                exhausted=std::string_view(error.what())==
                    "resident page budget exhausted including in-flight roots";
            }
            const auto after=authority.stats();
            const std::array<Coordinator::Lease,1> current{*active};
            require(exhausted && authority.compute_leases_current(current) &&
                after.admitted==before.admitted && after.queued==before.queued && after.active==before.active &&
                after.completions==before.completions && after.resident_updates==before.resident_updates &&
                after.locked_page_bytes==root_budget,
                "resource-ineligible admission or turnover changed live authority");
        }
        // An alias adds no resident payload. Earlier failed candidates must not
        // leave queue entries or reservation debt that blocks this feasible peer.
        const auto feasible=authority.admit(first.request);
        require(feasible.request_id==original.request_id+1 && authority.stats().locked_page_bytes==root_budget,
            "budget refusal consumed identity or blocked resident alias");
        authority.complete(*active);
        active=authority.acquire();
        require(active && active->ticket.request_id==feasible.request_id,
            "resource-ineligible candidate blocked feasible peer acquisition");
        authority.complete(*active);
        require(authority.stats().locked_page_bytes==0 && authority.stats().completions==2,
            "bounded admission retained payload after final alias completion");
        authority.close();
    }
    {
        // Exercise actual coordinator ordering with authoritative roots, without
        // inferring feasibility from token lengths or modeling numerical time.
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        Cache order_cache(Cache::Policy{4,64,unbounded,reserve},identity);
        Coordinator order(order_cache,{4,1,unbounded,reserve});
        const auto long_ticket=order.admit(first.request);
        auto running=order.acquire();
        require(running && running->ticket.request_id==long_ticket.request_id,
            "FIFO long request lost initial acquisition");
        for(unsigned round=0;round<4;++round) {
            const auto before_refusal=order.stats();
            order.fail_next_residency_for_test(Coordinator::ResidencyFault::before_commit);
            bool admission_refused=false;
            try {
                if(round%2==0)order.admit(second.request);
                else {
                    const std::array<std::shared_ptr<const Request>,2> rejected{first.request,second.request};
                    order.admit_batch(rejected);
                }
            } catch(const std::runtime_error& error) {
                admission_refused=std::string_view(error.what())=="injected coordinator before resident commit";
            }
            const auto after_refusal=order.stats();
            const std::array<Coordinator::Lease,1> current{*running};
            require(admission_refused && order.compute_leases_current(current) &&
                after_refusal.queued==before_refusal.queued && after_refusal.active==before_refusal.active &&
                after_refusal.admitted==before_refusal.admitted &&
                after_refusal.resident_updates==before_refusal.resident_updates &&
                after_refusal.locked_page_bytes==before_refusal.locked_page_bytes,
                "failed admission disturbed ready work or resident authority");
            const auto cancelled_head=order.admit(second.request);
            require(cancelled_head.request_id==long_ticket.request_id+1+3*round,
                "failed single or batch admission consumed request identity");
            const auto earlier=order.admit(second.request);
            const auto previous_acquisition=running->acquisition;
            order.yield(*running);
            const auto later=order.admit(second.request);
            order.cancel(cancelled_head);
            auto selected=order.acquire();
            require(selected && selected->ticket.request_id==earlier.request_id && !order.acquire(),
                "cancelled FIFO head blocked peer or exceeded physical capacity");
            order.complete(*selected);
            running=order.acquire();
            require(running && running->ticket.request_id==long_ticket.request_id &&
                running->acquisition==previous_acquisition+1 && running->root==first.request,
                "later arrival overtook yielded long authority");
            order.yield(*running);
            selected=order.acquire();
            require(selected && selected->ticket.request_id==later.request_id,
                "long request yield starved earlier queued peer");
            order.complete(*selected);
            running=order.acquire();
            require(running && running->ticket.request_id==long_ticket.request_id &&
                running->acquisition==previous_acquisition+2,
                "repeated FIFO round lost long request acquisition");
        }
        order.complete(*running);
        const auto final_order=order.stats();
        require(final_order.queued==0 && final_order.active==0 && final_order.admitted==0 &&
            final_order.cancellations==4 && final_order.completions==9 && !order.acquire(),
            "FIFO repeated-arrival trace lost terminal accounting");
        order.close();
    }
    require(hit_retention.size()==1 && hit_retention[0].generation==cold_retention[0].generation &&
        hit_retention[0].preparation_microseconds==cold_retention[0].preparation_microseconds,
        "warm hit replaced cold preparation cost with attachment timing");
    require(!second.metrics.admitted && second.metrics.attach_ms>0 && second.metrics.lookup_ms>=0,
        "T72 exact hit metrics");

    bool cancelled=false;
    try {
        auto tail_c=tail_b;tail_c.back()=(tail_c.back()+31)%kVocab;
        cache.prepare(*serving,make_prompt(tail_c),prefix_rows,unbounded,16,
            [&](int position){if(position>static_cast<int>(prefix_rows))
                throw std::runtime_error("intentional T72 cancellation");});
    } catch(const std::runtime_error&) {cancelled=true;}
    require(cancelled,"T72 cancellation did not fire");
    const auto after_cancel=cache.prepare(*serving,make_prompt(tail_a),prefix_rows,unbounded,32);
    require(after_cancel.metrics.cache_hit && after_cancel.metrics.reused_prompt_tokens==prefix_rows,
        "T72 cancellation erased or contaminated root");

    // T234: rendezvous through progress callbacks, without timing-based sleeps.
    {
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        Cache isolated(Cache::Policy{4,64,unbounded,reserve},identity);
        Coordinator authority(isolated,Coordinator::Policy{1,1,unbounded,reserve});
        authority.bind_physical_resources({},ninfer::exl3::Exl3ResourceInventory::unlimited());
        authority.reserve_runtime_host_sources();
        const auto cached_input=first.request->token_suffix();
        require(isolated.admit_input_authority(first.request,cached_input,unbounded).admitted,
            "source lifetime fixture did not retain cache root");
        const auto ticket=authority.admit(first.request);auto lease=authority.acquire();
        require(bool(lease),"host source fixture missing acquired root");
        const auto& state=*first.request->state();
        const ninfer::exl3::Exl3DevicePageKey key(state.model_identity(),state.shared_kv_page(64),
            {state.rope_offset(),false},state.position());
        auto lifetime=std::make_shared<const ninfer::exl3::Exl3DevicePageKey>(key);
        const auto initial_locked=authority.stats().locked_page_bytes;
        authority.fail_next_residency_for_test(Coordinator::ResidencyFault::before_commit);
        bool failed=false;
        try{authority.bind_runtime_host_source(*lease,lifetime,key);}
        catch(const std::runtime_error& error){failed=std::string_view(error.what())==
            "injected coordinator before resident commit";}
        require(failed && authority.stats().locked_page_bytes==initial_locked,
            "failed host source binding changed resident range union");
        authority.bind_runtime_host_source(*lease,lifetime,key);
        const auto bound_locked=authority.stats().locked_page_bytes;
        SYSTEM_INFO system_info{};GetSystemInfo(&system_info);
        require(bound_locked>=initial_locked && bound_locked-initial_locked<=2ULL*system_info.dwPageSize,
            "root-owned KV payload was locked and charged twice by source binding");
        bool duplicate=false;
        try{authority.bind_runtime_host_source(*lease,lifetime,key);}
        catch(const std::logic_error& error){duplicate=std::string_view(error.what())=="runtime host source already bound";}
        require(duplicate && authority.stats().locked_page_bytes==bound_locked,
            "duplicate source binding changed locked ownership");
        {
            using Key=ninfer::exl3::Exl3DevicePageKey;
            std::array<std::shared_ptr<const Key>,63> aliases;
            for(auto& alias:aliases) {
                alias=std::make_shared<const Key>(key);
                authority.bind_runtime_host_source(*lease,alias,key);
            }
            const auto full=authority.stats();
            bool exhausted=false;
            try{authority.bind_runtime_host_source(*lease,std::make_shared<const Key>(key),key);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){exhausted=true;}
            require(exhausted && full.locked_page_bytes==bound_locked &&
                authority.stats().resident_updates==full.resident_updates,
                "source binding saturation changed residency or multiplied aliased backing");
            aliases[17].reset();
            aliases[17]=std::make_shared<const Key>(key);
            authority.bind_runtime_host_source(*lease,aliases[17],key);
            require(authority.stats().locked_page_bytes==bound_locked,
                "expired source lifetime did not permit bounded slot reuse");
        }
        const auto stale=*lease;authority.yield(*lease);lease=authority.acquire();
        bool stale_refused=false;
        try{authority.bind_runtime_host_source(stale,std::make_shared<const ninfer::exl3::Exl3DevicePageKey>(key),key);}
        catch(const std::invalid_argument&){stale_refused=true;}
        require(stale_refused && lease && lease->ticket.request_id==ticket.request_id,
            "old acquisition bound source to reacquired request");
        isolated.reset_for_model_reload(Identity{"source-reload","target-pin","tokenizer-pin","new-config","text-only-v2"});
        require(isolated.roots().empty() && key.current() &&
            authority.compute_leases_current(std::span(&*lease,1)) &&
            authority.stats().locked_page_bytes==bound_locked,
            "cache reload invalidated active source owner or prematurely dropped request locks");
        authority.complete(*lease);
        const auto source_only=authority.stats().locked_page_bytes;
        require(source_only>=key.retained_host_bytes() && source_only<bound_locked,
            "request removal lost live source or retained entire request lock set");
        lifetime.reset();
        using Decision=ninfer::exl3::Exl3VeriCachePrefixIndex::RetentionDecision;
        const auto expired=authority.trim_prefix_retention(unbounded,
            std::span<const Decision>{});
        require(expired.inputs_current && expired.budget_met && !expired.evicted &&
            authority.stats().locked_page_bytes==0,
            "expired fill lifetime retained host source after resident reconciliation");

        // T238: the real retention transaction removes only logical lookup
        // ownership. A delayed fill/reader lifetime still owns and charges its
        // exact page until its last reference disappears.
        require(isolated.admit_input_authority(first.request,cached_input,unbounded).admitted,
            "root/source eviction fixture did not restore logical cache root");
        (void)authority.admit(first.request);lease=authority.acquire();
        require(bool(lease),"root/source eviction fixture reacquisition failed");
        auto fill_lifetime=std::make_shared<const ninfer::exl3::Exl3DevicePageKey>(key);
        authority.bind_runtime_host_source(*lease,fill_lifetime,key);
        authority.complete(*lease);
        const auto before_trim=authority.stats();
        const auto metadata=isolated.retention_metadata();
        require(metadata.size()==1 && metadata.front().root==first.request,
            "root/source eviction fixture lookup metadata mismatch");
        const std::array<Decision,1> remove{{
            {metadata.front().root,metadata.front().generation,false,0}}};
        const auto trimmed=authority.trim_prefix_retention(0,remove);
        const auto after_trim=authority.stats();
        require(trimmed.inputs_current && trimmed.budget_met && trimmed.evicted==1 &&
            isolated.roots().empty() && after_trim.resident_updates==before_trim.resident_updates+1 &&
            after_trim.locked_page_bytes>=key.retained_host_bytes() &&
            after_trim.locked_page_bytes<before_trim.locked_page_bytes,
            "logical root eviction dropped live fill source or retained evicted root coverage");
        fill_lifetime.reset();
        const auto released=authority.trim_prefix_retention(unbounded,
            std::span<const Decision>{});
        require(released.inputs_current && released.budget_met && !released.evicted &&
            authority.stats().locked_page_bytes==0,
            "last fill reference did not release independent host source charge");
        authority.close();
    }
    for(bool short_credit:{true,false}) {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        Cache reserved(Cache::Policy{4,64,unbounded,reserve},identity);
        Coordinator authority(reserved,Coordinator::Policy{2,2,unbounded,reserve});
        const auto bytes=Coordinator::runtime_host_source_metadata_bytes();
        auto limits=Inventory::unlimited();
        limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=bytes-(short_credit?1:0);
        authority.bind_physical_resources({},limits);
        bool exhausted=false;
        try{authority.reserve_runtime_host_sources();}
        catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){exhausted=true;}
        require(exhausted==short_credit && authority.runtime_host_source_storage_bytes()==(short_credit?0:bytes),
            "host source table violated short/exact metadata credit");
        if(!short_credit) {
            authority.reserve_runtime_host_sources(); // Idempotent cache startup call.
            Inventory::Requirement extra;extra.configuration=0x485354455354;
            extra.add(Inventory::Domain::host_metadata,1,1);
            bool invoked=false,refused=false;
            try{authority.allocate_startup_resources(extra,[&](auto){invoked=true;return Inventory{};});}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && !invoked && authority.runtime_host_source_storage_bytes()==bytes,
                "host source table reservation was uncharged or duplicated");
        }
        authority.close();
        require(authority.runtime_host_source_storage_bytes()==0,"closed coordinator retained host source table");
    }
    // Only the producer submits prefix work; the joined context restores after
    // publication. Both threads are joined before inspecting any result.
    for(unsigned slots:{1U,2U,8U})for(bool short_credit:{true,false}) {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        Cache reserved(Cache::Policy{4,64,unbounded,reserve},identity);
        Coordinator authority(reserved,Coordinator::Policy{2,2,unbounded,reserve});
        const auto requirement=Cache::preparation_storage_requirement(slots,serving->max_context());
        auto limits=Inventory::unlimited();
        const auto domain=static_cast<unsigned>(Inventory::Domain::host_metadata);
        limits[domain]=requirement.units[domain]-(short_credit?1:0);
        authority.bind_physical_resources({},limits);
        if(!short_credit)for(unsigned fault=1;fault<=slots+1;++fault) {
            bool construction_failed=false;
            std::array<std::weak_ptr<const void>,8> constructed;
            try{reserved.reserve_preparation_storage(authority,slots,serving->max_context(),fault,&constructed);}
            catch(const std::runtime_error& error) {
                construction_failed=std::string_view(error.what())=="injected preparation storage construction failure";
            }
            catch(const std::invalid_argument& error) {
                construction_failed=fault==slots+1 && std::string_view(error.what())==
                    "resource reservation actual extent/domain mismatch";
            }
            const auto rolled_back=reserved.preparation_storage_stats();
            require(construction_failed && rolled_back.slots==0 && rolled_back.metadata_bytes==0,
                "T234 failed startup construction published partial pool");
            const std::weak_ptr<const void> empty;
            for(unsigned i=0;i<constructed.size();++i) {
                const bool observed=constructed[i].owner_before(empty) || empty.owner_before(constructed[i]);
                require(observed==(i<std::min(fault,slots)) && constructed[i].expired(),
                    "T234 startup rollback retained constructed flight owner or missed fault boundary");
            }
        }
        bool exhausted=false;
        try{reserved.reserve_preparation_storage(authority,slots,serving->max_context());}
        catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){exhausted=true;}
        const auto storage=reserved.preparation_storage_stats();
        require(exhausted==short_credit && storage.slots==(short_credit?0U:slots) &&
            storage.metadata_bytes==(short_credit?0:requirement.units[domain]) &&
            storage.participants==0 && storage.retained_roots==0,
            "T234 startup credit boundary published partial preparation storage");
        if(!short_credit) {
            bool duplicate=false;
            try{reserved.reserve_preparation_storage(authority,slots,serving->max_context());}
            catch(const std::logic_error& error) {
                duplicate=std::string_view(error.what())=="preparation storage already reserved";
            }
            require(duplicate && reserved.preparation_storage_stats().metadata_bytes==storage.metadata_bytes,
                "T234 duplicate storage reservation grew pool");
        }
        authority.close();
    }
    enum class PreparationCase {producer_cancel,reload,waiter_cancel,completion_cancel,admission_failure};
    for(bool pooled:{false,true})for(const auto scenario:{PreparationCase::producer_cancel,PreparationCase::reload,
                            PreparationCase::waiter_cancel,PreparationCase::completion_cancel,
                            PreparationCase::admission_failure}) {
        const bool reload_during_fill=scenario==PreparationCase::reload;
        Cache concurrent(Cache::Policy{4,64,unbounded,reserve},identity);
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Coordinator preparation_authority(concurrent,Coordinator::Policy{2,2,unbounded,reserve});
        if(pooled) {
            preparation_authority.bind_physical_resources({},Inventory::unlimited());
            concurrent.reserve_preparation_storage(preparation_authority,1,serving->max_context());
            const auto storage=concurrent.preparation_storage_stats();
            const auto requirement=Cache::preparation_storage_requirement(1,serving->max_context());
            require(storage.slots==1 && storage.participants==0 && storage.retained_roots==0 &&
                storage.metadata_bytes==requirement.units[static_cast<unsigned>(Inventory::Domain::host_metadata)],
                "T234 reserved preparation storage initial attribution");
        }
        if(scenario==PreparationCase::admission_failure)concurrent.fail_next_preparation_admission_for_test();
        std::mutex gate_mutex;std::condition_variable gate;
        bool producer_entered=false,waiter_entered=false,consumer_finished=false;
        bool producer_finished=false;unsigned waiter_callbacks=0;
        std::exception_ptr producer_error,consumer_error;
        std::shared_ptr<const Request> joined_root,producer_root,exhausted_root;
        bool joined=false,joined_hit=false;
        std::jthread producer([&] {
            try {
                auto result=concurrent.prepare_concurrent(*serving,common,prefix_rows,unbounded,32,
                    [&](int) {
                        std::unique_lock lock(gate_mutex);
                        producer_entered=true;gate.notify_all();
                        if(!gate.wait_for(lock,std::chrono::seconds(10),[&] {
                            return waiter_entered || consumer_finished;
                        }))throw std::runtime_error("T234 waiter rendezvous timeout");
                        if(scenario==PreparationCase::producer_cancel)
                            throw std::runtime_error("T234 producer consumer cancelled");
                    });
                producer_root=result.request;
            } catch(...) {producer_error=std::current_exception();}
            std::lock_guard lock(gate_mutex);producer_finished=true;gate.notify_all();
        });
        std::jthread consumer([&] {
            try {
                {
                    std::unique_lock lock(gate_mutex);
                    if(!gate.wait_for(lock,std::chrono::seconds(10),[&]{return producer_entered;}))
                        throw std::runtime_error("T234 producer rendezvous timeout");
                }
                auto result=concurrent.prepare_concurrent(*fresh,common,prefix_rows,unbounded,32,
                    [&](int) {
                        if(reload_during_fill) {
                            auto replacement=identity;
                            replacement.configuration_identity="T234 replacement during fill";
                            concurrent.reset_for_model_reload(replacement);
                        }
                        std::unique_lock lock(gate_mutex);waiter_entered=true;++waiter_callbacks;gate.notify_all();
                        if(scenario==PreparationCase::completion_cancel && waiter_callbacks==1) {
                            if(!gate.wait_for(lock,std::chrono::seconds(10),[&]{return producer_finished;}))
                                throw std::runtime_error("T234 completion rendezvous timeout");
                            if(pooled) {
                                const auto held=concurrent.preparation_storage_stats();
                                require(held.slots==1 && held.participants==1 && held.retained_roots==1,
                                    "T234 completed flight not retained by last waiter");
                                auto different=common;different.front()=(different.front()+3)%kVocab;
                                const auto fallback=concurrent.prepare_concurrent(*builder,different,
                                    prefix_rows,unbounded,32);
                                exhausted_root=fallback.request;
                                const auto after=concurrent.preparation_storage_stats();
                                require(!fallback.metrics.cache_hit && !fallback.metrics.preparation_reused &&
                                    fallback.metrics.admitted && after.slots==held.slots &&
                                    after.participants==held.participants && after.retained_roots==held.retained_roots &&
                                    after.metadata_bytes==held.metadata_bytes,
                                    "T234 exhausted pool recycled live waiter or grew flight storage");
                            }
                            return;
                        }
                        if(scenario==PreparationCase::waiter_cancel || scenario==PreparationCase::completion_cancel)
                            throw std::runtime_error("T234 joined consumer cancelled");
                    });
                joined_root=result.request;joined=result.metrics.preparation_reused;
                joined_hit=result.metrics.cache_hit;
            } catch(...) {consumer_error=std::current_exception();}
            std::lock_guard lock(gate_mutex);consumer_finished=true;gate.notify_all();
        });
        producer.join();consumer.join();
        if(exhausted_root) {
            auto different=common;different.front()=(different.front()+3)%kVocab;
            const auto oracle=Request::initialize(*builder,different,32);
            require(exhausted_root->matches_tokens(different) && exhausted_root->same_taps(*oracle) &&
                exhausted_root->state()->same_payload(*oracle->state()),
                "T234 exhausted pool fallback differs from independent authority");
        }
        if(pooled) {
            const auto storage=concurrent.preparation_storage_stats();
            require(storage.slots==1 && storage.participants==0 && storage.retained_roots==0,
                "T234 reserved flight retained participant or request after completion");
        }
        if(scenario==PreparationCase::admission_failure) {
            const auto allocation_failure=[](const std::exception_ptr& failure) {
                if(!failure)return false;
                try{std::rethrow_exception(failure);}catch(const std::bad_alloc&){return true;}
                catch(...){return false;}
            };
            require(allocation_failure(producer_error) && allocation_failure(consumer_error) &&
                !producer_root && !joined_root && concurrent.storage_stats().entries==0,
                "T234 preparation admission failure did not fan out without publication");
            const auto retry=concurrent.prepare_concurrent(*serving,common,prefix_rows,unbounded,32);
            const auto oracle=Request::initialize(*builder,common,32);
            require(retry.metrics.admitted && !retry.metrics.cache_hit && !retry.metrics.preparation_reused &&
                retry.request->same_taps(*oracle) && retry.request->state()->same_payload(*oracle->state()),
                "T234 failed preparation prevented fresh authority retry");
            concurrent.fail_next_preparation_admission_for_test();
            bool duplicate_refused=false;
            try{concurrent.fail_next_preparation_admission_for_test();}
            catch(const std::logic_error& error) {
                duplicate_refused=std::string_view(error.what())=="preparation admission fault already armed";
            }
            const auto refusal_state=fresh->export_exact_host_state(nullptr,false);
            const auto refusal_metadata=concurrent.retention_metadata();
            unsigned refusal_callbacks=0;
            auto cold_input=common;cold_input.front()=(cold_input.front()+1)%kVocab;
            for(const auto* input:std::array<const std::vector<std::int64_t>*,2>{&common,&cold_input})
                for(int width:{0,7,17,64,-1}) {
                bool refused=false;
                try{(void)concurrent.prepare_concurrent(*fresh,*input,prefix_rows,unbounded,width,
                    [&](int){++refusal_callbacks;});}
                catch(const std::invalid_argument& error) {
                    refused=std::string_view(error.what())=="request prefill width must be8/16/32/128/1024";
                }
                require(refused,"T234 invalid width reached cold/warm preparation");
            }
            std::vector<std::int64_t> oversized(static_cast<std::size_t>(fresh->max_context())+1,common.front());
            bool capacity_refused=false;
            try{(void)concurrent.prepare_concurrent(*fresh,oversized,prefix_rows,unbounded,32,
                [&](int){++refusal_callbacks;});}
            catch(const std::invalid_argument& error) {
                capacity_refused=std::string_view(error.what())=="serving prompt exceeds context capacity";
            }
            const auto after_refusal=concurrent.retention_metadata();
            require(capacity_refused && !refusal_callbacks && after_refusal.size()==refusal_metadata.size() &&
                after_refusal.front().generation==refusal_metadata.front().generation &&
                after_refusal.front().lookup_selections==refusal_metadata.front().lookup_selections &&
                fresh->export_exact_host_state(nullptr,false)->same_payload(*refusal_state),
                "T234 preallocation refusal mutated context/cache or invoked progress");
            const auto warm=concurrent.prepare_concurrent(*fresh,common,prefix_rows,unbounded,32);
            require(duplicate_refused && warm.metrics.cache_hit && warm.request==retry.request,
                "T234 duplicate fault arm or armed warm lookup changed authority");
            auto changed=common;changed.front()=(changed.front()+1)%kVocab;
            bool next_cold_failed=false;
            try{(void)concurrent.prepare_concurrent(*fresh,changed,prefix_rows,unbounded,32);}
            catch(const std::bad_alloc&){next_cold_failed=true;}
            const auto retained=concurrent.roots();
            require(next_cold_failed && retained.size()==1 && retained.front()==retry.request,
                "T234 warm hit consumed fault or cold failure evicted retained authority");
            const auto changed_retry=concurrent.prepare_concurrent(*fresh,changed,prefix_rows,unbounded,32);
            const auto changed_oracle=Request::initialize(*builder,changed,32);
            require(changed_retry.metrics.admitted && changed_retry.request->matches_tokens(changed) &&
                changed_retry.request->same_taps(*changed_oracle) &&
                changed_retry.request->state()->same_payload(*changed_oracle->state()),
                "T234 one-shot admission fault persisted into second cold retry");
            continue;
        }
        if(scenario==PreparationCase::waiter_cancel || scenario==PreparationCase::completion_cancel) {
            require(scenario!=PreparationCase::completion_cancel || waiter_callbacks==2,
                "T234 completion cancellation acceptance callback missing");
            bool waiter_cancelled=false;
            if(consumer_error)try {std::rethrow_exception(consumer_error);}
                catch(const std::runtime_error& error) {
                    waiter_cancelled=std::string(error.what())=="T234 joined consumer cancelled";
                }
            if(producer_error)std::rethrow_exception(producer_error);
            require(waiter_cancelled && !joined_root && producer_root &&
                concurrent.storage_stats().entries==(exhausted_root?2U:1U),"T234 cancelled waiter poisoned producer");
            const auto oracle=Request::initialize(*builder,common,32);
            require(producer_root->token_suffix()==oracle->token_suffix() &&
                producer_root->same_taps(*oracle) && producer_root->state()->same_payload(*oracle->state()),
                "T234 producer after waiter cancellation differs from independent authority");
            const auto retry=concurrent.prepare_concurrent(*fresh,common,prefix_rows,unbounded,32);
            require(retry.metrics.cache_hit && !retry.metrics.preparation_reused &&
                !retry.metrics.admitted && retry.request==producer_root,
                "T234 cancelled waiter could not reuse completed producer authority");
            continue;
        }
        if(reload_during_fill) {
            const auto epoch_failure=[](const std::exception_ptr& failure) {
                if(!failure)return false;
                try {std::rethrow_exception(failure);}
                catch(const std::runtime_error& error) {
                    return std::string(error.what())=="serving prefix preparation epoch changed";
                } catch(...) {return false;}
            };
            require(epoch_failure(producer_error) && epoch_failure(consumer_error) &&
                !joined_root && concurrent.storage_stats().entries==0,
                "T234 reload published or returned old-namespace preparation");
            const auto replacement=concurrent.prepare_concurrent(*serving,common,prefix_rows,unbounded,32);
            const auto oracle=Request::initialize(*builder,common,32);
            require(!replacement.metrics.cache_hit && !replacement.metrics.preparation_reused &&
                replacement.metrics.admitted && replacement.request->same_taps(*oracle) &&
                replacement.request->state()->same_payload(*oracle->state()),
                "T234 replacement namespace failed independent retry");
            continue;
        }
        bool producer_cancelled=false;
        if(producer_error)try {std::rethrow_exception(producer_error);}
            catch(const std::runtime_error& error) {
                producer_cancelled=std::string(error.what())=="T234 producer consumer cancelled";
            }
        if(consumer_error)std::rethrow_exception(consumer_error);
        require(producer_cancelled && joined && !joined_hit && joined_root &&
            concurrent.storage_stats().entries==1,"T234 producer cancellation lost surviving joiner");
        const auto oracle=Request::initialize(*builder,common,32);
        require(joined_root->token_suffix()==oracle->token_suffix() &&
            joined_root->same_taps(*oracle) && joined_root->state()->same_payload(*oracle->state()),
            "T234 surviving consumer differs from independent authority");
        const auto warm=concurrent.prepare_concurrent(*fresh,common,prefix_rows,unbounded,32);
        require(warm.metrics.cache_hit && !warm.metrics.preparation_reused &&
            !warm.metrics.admitted && warm.request==joined_root,
            "T234 completed preparation was not retained as exact cache authority");
    }

    auto reloaded=identity;reloaded.configuration_identity="different-qualified-config";
    cache.reset_for_model_reload(reloaded);
    require(cache.storage_stats().entries==0,"T72 reload retained old publication");
    require(cache.retention_metadata().empty(),"reload retained stale cost observations");
    const auto after_reload=cache.prepare(*serving,make_prompt(tail_b),prefix_rows,unbounded,32);
    require(!after_reload.metrics.cache_hit && after_reload.metrics.admitted,
        "T72 reload identity reused stale root");
    std::size_t pressure_evictions=0;
    const auto pressure=cache.trim(reserve,&pressure_evictions);
    require(pressure.entries==0 && pressure.accounted_bytes==0 && pressure_evictions==1,
        "T72 physical reserve pressure did not evict cache");

    const std::string contract=identity.contract();
    const auto root64=Request::initialize(*builder,
        std::span<const std::int64_t>(source.data(),64),32);
    const auto root129=Request::initialize(*builder,
        std::span<const std::int64_t>(source.data(),129),32);
    const auto root192=Request::initialize(*builder,
        std::span<const std::int64_t>(source.data(),192),32);
    {
        // T026: C2 admission is one old/new resident-owner union. These roots
        // retain the same model authority but have different host extents.
        // Each fits the selected cap alone; the coordinator must not infer that
        // two individually feasible lane transitions also fit together.
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        const auto measure=[&](std::span<const std::shared_ptr<const Request>> roots) {
            Cache measured(Cache::Policy{4,64,unbounded,reserve},identity);
            Coordinator authority(measured,Coordinator::Policy{4,2,unbounded,reserve});
            (void)authority.admit_batch(roots);
            const auto bytes=authority.stats().locked_page_bytes;
            authority.close();return bytes;
        };
        const std::array<std::shared_ptr<const Request>,1> small{root64},large{root192};
        const std::array<std::shared_ptr<const Request>,2> mixed{root64,root192};
        require(root64->state()->model_identity()==root129->state()->model_identity() &&
            root64->state()->model_identity()==root192->state()->model_identity(),
            "T026 mixed C2 roots did not share model authority");
        const auto small_bytes=measure(small),large_bytes=measure(large),mixed_bytes=measure(mixed);
        const auto independent_cap=std::max(small_bytes,large_bytes);
        require(small_bytes && large_bytes && mixed_bytes>independent_cap,
            "T026 fixture did not distinguish individual and combined resident peaks");

        Cache bounded(Cache::Policy{4,64,unbounded,reserve},identity);
        Coordinator authority(bounded,Coordinator::Policy{4,2,independent_cap,reserve});
        const auto empty=authority.stats();bool exhausted=false;
        try{(void)authority.admit_batch(mixed);}
        catch(const ninfer::exl3::Exl3HostResidentBudgetExhausted&){exhausted=true;}
        const auto rejected=authority.stats();
        require(exhausted && rejected.admitted==empty.admitted && rejected.queued==empty.queued &&
            rejected.active==empty.active && rejected.resident_updates==empty.resident_updates &&
            rejected.locked_page_bytes==empty.locked_page_bytes,
            "T026 infeasible C2 batch partially committed lanes or residency");
        const auto survivor_ticket=authority.admit(root64);auto survivor=authority.acquire();
        require(survivor && survivor->ticket.request_id==survivor_ticket.request_id,
            "T026 survivor acquisition missing");
        const auto before_survivor_failure=authority.stats();
        const std::array<std::shared_ptr<const Request>,2> larger_pair{root129,root192};
        exhausted=false;try{(void)authority.admit_batch(larger_pair);}
        catch(const ninfer::exl3::Exl3HostResidentBudgetExhausted&){exhausted=true;}
        const auto after_survivor_failure=authority.stats();
        const std::array<Coordinator::Lease,1> current{*survivor};
        require(exhausted && authority.compute_leases_current(current) &&
            after_survivor_failure.admitted==before_survivor_failure.admitted &&
            after_survivor_failure.queued==before_survivor_failure.queued &&
            after_survivor_failure.active==before_survivor_failure.active &&
            after_survivor_failure.resident_updates==before_survivor_failure.resident_updates &&
            after_survivor_failure.locked_page_bytes==before_survivor_failure.locked_page_bytes,
            "T026 mixed batch budget failure changed active survivor");
        authority.complete(*survivor);
        const auto large_ticket=authority.admit(root192);auto large_lease=authority.acquire();
        require(large_lease && large_lease->ticket.request_id==large_ticket.request_id,
            "T026 individually feasible large root did not fit shared cap");
        authority.complete(*large_lease);authority.close();

        Cache fault_cache(Cache::Policy{4,64,unbounded,reserve},identity);
        Coordinator faulted(fault_cache,Coordinator::Policy{4,2,unbounded,reserve});
        const auto fault_ticket=faulted.admit(root64);auto fault_survivor=faulted.acquire();
        require(fault_survivor && fault_survivor->ticket.request_id==fault_ticket.request_id,
            "T026 fault survivor acquisition missing");
        const auto before_fault=faulted.stats();
        faulted.fail_next_residency_for_test(Coordinator::ResidencyFault::before_commit);
        bool injected=false;try{(void)faulted.admit_batch(larger_pair);}
        catch(const std::runtime_error& error){
            injected=std::string_view(error.what())=="injected coordinator before resident commit";
        }
        const auto after_fault=faulted.stats();
        const std::array<Coordinator::Lease,1> fault_current{*fault_survivor};
        require(injected && faulted.compute_leases_current(fault_current) &&
            after_fault.admitted==before_fault.admitted && after_fault.queued==before_fault.queued &&
            after_fault.active==before_fault.active &&
            after_fault.resident_updates==before_fault.resident_updates &&
            after_fault.locked_page_bytes==before_fault.locked_page_bytes,
            "T026 injected C2 batch failure changed active survivor");
        faulted.complete(*fault_survivor);faulted.close();
    }
    {
        serving->restore_exact_host_state(*root129->state());
        require(serving->restore_exact_host_state_if_needed(*root64->state()),
            "guarded restore skipped a different exact root");
        require(!serving->restore_exact_host_state_if_needed(*root64->state()) &&
            serving->exact_host_state_resident(*root64->state()),
            "guarded restore missed its unchanged exact witness");
        const auto hit_payload=serving->export_exact_host_state(nullptr,false);
        require(hit_payload->same_payload(*root64->state()),
            "guarded restore hit changed represented KV/recurrent/logit/tap state");
        // Export can establish a fresh residency identity; restore the exact
        // fixture root before checking invalidation by the following decode.
        serving->restore_exact_host_state(*root64->state());
        serving->decode(source[64]);
        require(!serving->exact_host_state_resident(*root64->state()) &&
            serving->restore_exact_host_state_if_needed(*root64->state()),
            "guarded restore reused stale witness after token mutation");
        require(!serving->restore_exact_host_state_if_needed(*root64->state()),
            "guarded restore did not establish replacement witness");
        const auto restored_payload=serving->export_exact_host_state(nullptr,false);
        require(restored_payload->same_payload(*root64->state()),
            "guarded restore after mutation failed complete represented-state recovery");
        auto changed_tokens=root64->token_suffix();
        changed_tokens.back()=(changed_tokens.back()+1)%kVocab;
        const auto changed=Request::initialize(*builder,changed_tokens,32);
        serving->restore_exact_host_state(*changed->state());
        require(serving->position()==root64->state()->position() &&
            !serving->exact_host_state_resident(*root64->state()) &&
            serving->restore_exact_host_state_if_needed(*root64->state()),
            "guarded restore confused equal positions with exact content identity");
        require(serving->export_exact_host_state(nullptr,false)->same_payload(*root64->state()),
            "guarded restore failed recovery from changed same-length prefix");
    }
    const auto one_stats=[&](const std::shared_ptr<const Request>& root) {
        Index one(1);const auto admitted=one.admit(root,contract,unbounded,unbounded,reserve);
        require(admitted.admitted,"T72 one-root accounting admission");return admitted.storage;
    };
    const auto bytes64=one_stats(root64).accounted_bytes;
    const auto bytes129=one_stats(root129).accounted_bytes;
    const auto bytes192=one_stats(root192).accounted_bytes;
    require(bytes64<bytes129 && bytes129<bytes192,"T72 root byte accounting is not monotonic");

    Index bounded(4);
    {
        Cache extension(Cache::Policy{4,64,unbounded,reserve},identity);
        require(extension.admit_input_authority(root64,root64->token_suffix(),unbounded).admitted,
            "extension metadata base admission");
        const auto extended=extension.prepare(*serving,root129->token_suffix(),129,unbounded,32);
        require(extended.metrics.cache_hit && extended.metrics.reused_prompt_tokens==64 &&
            extended.metrics.admitted,"partial-prefix extension metadata route NOT_EXERCISED");
        const auto entries=extension.retention_metadata();
        bool found_extension=false;
        for(const auto& entry:entries)if(entry.root==extended.request) {
            found_extension=true;
            require(!entry.preparation_microseconds && !entry.observed_accesses,
                "partial-prefix extension invented full cold preparation cost");
        }
        require(found_extension,"extension retention metadata omitted admitted root");
    }
    {
        Index ordered(3);
        ordered.publish(root64,contract);ordered.publish(root129,contract);ordered.publish(root192,contract);
        const auto middle=ordered.admit(root129,contract,unbounded,unbounded,0);
        const std::vector<std::shared_ptr<const Request>> middle_order{root64,root192,root129};
        require(middle.admitted && middle.replaced && middle.evicted==0 && ordered.roots()==middle_order &&
            same_storage(middle.storage,ordered.storage_stats()),"middle replacement changed retained FIFO order");
        const auto evicted=ordered.admit(root64,contract+"/new",unbounded,unbounded,0);
        const std::vector<std::shared_ptr<const Request>> evicted_order{root192,root129,root64};
        require(evicted.admitted && !evicted.replaced && evicted.evicted==1 && ordered.roots()==evicted_order &&
            same_storage(evicted.storage,ordered.storage_stats()),"capacity admission moved wrong retained suffix");
        require(!ordered.lookup(*serving,root64->token_suffix(),contract) &&
            ordered.lookup(*serving,root64->token_suffix(),contract+"/new")==root64,
            "capacity eviction merged distinct compatibility contracts");
    }
    require(bounded.admit(root64,contract,bytes129,unbounded,reserve).admitted,
        "T72 bounded first admission");
    const auto replacement=bounded.admit(root129,contract,bytes129,unbounded,reserve);
    require(same_storage(replacement.storage,bounded.storage_stats()),
        "eviction admission returned stale storage accounting");
    require(replacement.admitted && replacement.evicted==1 && bounded.size()==1 &&
        !bounded.lookup(*serving,root64->token_suffix(),contract) &&
        bounded.lookup(*serving,root129->token_suffix(),contract)==root129,
        "T72 byte FIFO eviction");
    const auto before_oversize=bounded.storage_stats();
    const auto oversize=bounded.admit(root192,contract,bytes129,unbounded,reserve);
    require(same_storage(oversize.storage,bounded.storage_stats()),
        "refused admission returned candidate storage instead of retained storage");
    require(!oversize.admitted && oversize.evicted==0 &&
        bounded.storage_stats().accounted_bytes==before_oversize.accounted_bytes &&
        bounded.lookup(*serving,root129->token_suffix(),contract)==root129,
        "T72 oversize admission changed retained root");
    const auto reserve_reject=bounded.admit(root64,contract,unbounded,
        reserve+bytes64-1,reserve);
    require(!reserve_reject.admitted && reserve_reject.effective_budget_bytes==bytes64-1,
        "T72 reserve cap admitted an over-budget root");

    Index longest(4);longest.publish(root64,contract);longest.publish(root129,contract);
    {
        Index growth(4);growth.publish(root64,contract);
        const auto before=growth.roots();
        growth.fail_next_publication_growth_for_test();
        bool duplicate=false;
        try{growth.fail_next_publication_growth_for_test();}catch(const std::logic_error&){duplicate=true;}
        require(duplicate,"prefix growth fault overwritten");
        bool allocation_failed=false;
        try{growth.publish(root129,contract);}catch(const std::bad_alloc&){allocation_failed=true;}
        require(allocation_failed && growth.roots()==before &&
            growth.lookup_longest(*serving,root129->token_suffix(),contract,64)==root64,
            "prefix growth failure changed retained roots or shortlist");
        growth.publish(root129,contract);
        require(growth.lookup_longest(*serving,root129->token_suffix(),contract,64)==root129,
            "prefix growth failure prevented retry");
    }
    require(longest.lookup_longest(*serving,
        std::span<const std::int64_t>(source.data(),192),contract,64)==root129,
        "T72 longest exact prefix selection");
    auto changed=std::vector<std::int64_t>(source.begin(),source.begin()+192);
    changed[100]=(changed[100]+1)%kVocab;
    require(longest.lookup_longest(*serving,changed,contract,64)==root64,
        "T72 longest lookup crossed token mismatch");
    // A newer shorter match must not hide an older longer match; a full match
    // can terminate the scan only after exact identity has been established.
    longest.publish(root64,contract);
    const auto query192=std::span<const std::int64_t>(source.data(),192);
    require(longest.lookup_longest(*serving,query192,contract,64)==root129,
        "longest lookup preferred newer short root");
    longest.publish(root192,contract);
    require(longest.lookup_longest(*serving,query192,contract,64)==root192 &&
        longest.lookup_longest(*serving,changed,contract,64)==root64,
        "longest full-match pruning bypassed exact token comparison");
    require(!longest.lookup_longest(*serving,changed,contract,65) &&
        !longest.lookup_longest(*serving,query192,contract,193),
        "longest lookup ignored minimum frontier");
    Index retained_index(1);
    {
        Index exhausted(1);exhausted.publish(root64,contract);exhausted.exhaust_generations_for_test();
        for(bool admission:{false,true}) {
            bool refused_generation=false;
            try {
                if(admission)exhausted.admit(root129,contract,unbounded,unbounded,0);
                else exhausted.publish(root129,contract);
            }catch(const std::overflow_error&){refused_generation=true;}
            require(refused_generation && exhausted.size()==1 &&
                exhausted.lookup(*serving,root64->token_suffix(),contract)==root64,
                "generation exhaustion evicted existing prefix authority");
        }
    }
    {
        Index metadata(3);
        metadata.publish(root64,contract);
        const auto one=metadata.identity_allocated_bytes();
        const auto retention=metadata.retention_metadata();
        require(retention.size()==1 && retention[0].root==root64 && retention[0].reusable_tokens==64 &&
            retention[0].retained_payload_bytes==metadata.storage_stats().payload_allocated_bytes &&
            !retention[0].preparation_microseconds && !retention[0].observed_accesses,
            "prefix retention metadata invented costs or lost owner attribution");
        require(metadata.supply_retention_observation(root64,retention[0].generation,17,0),
            "current retention observation refused");
        const auto observed=metadata.retention_metadata();
        require(observed[0].preparation_microseconds==17 && observed[0].observed_accesses==0,
            "retention observation lost supplied zero or cost");
        metadata.publish(root64,contract);
        require(!metadata.supply_retention_observation(root64,retention[0].generation,99,99) &&
            !metadata.retention_metadata()[0].preparation_microseconds &&
            !metadata.retention_metadata()[0].observed_accesses,
            "stale observation crossed replacement generation");
        metadata.publish(root129,contract);metadata.publish(root192,contract);
        require(metadata.identity_allocated_bytes()==3*one &&
            metadata.storage_stats().identity_allocated_bytes==3*one,
            "prefix index retained token-length-dependent duplicate key storage");
    }
    {
        Cache input_cache(Cache::Policy{4,64,unbounded,reserve},identity);
        Index aliases(2);aliases.publish(root64,contract);aliases.publish(root64,contract+"/alias");
        const auto alias_metadata=aliases.retention_metadata();
        require(alias_metadata.size()==2 && alias_metadata[0].generation!=alias_metadata[1].generation &&
            alias_metadata[0].retained_payload_bytes==alias_metadata[1].retained_payload_bytes &&
            aliases.storage_stats().payload_allocated_bytes==alias_metadata[0].retained_payload_bytes,
            "retention alias metadata confused per-root bytes with unique cache bytes");
        require(aliases.supply_retention_observation(root64,alias_metadata[0].generation,4,3) &&
            !aliases.retention_metadata()[1].preparation_microseconds,
            "retention observation leaked into alias entry");
        aliases.clear();
        require(alias_metadata[0].root->matches_tokens(root64->token_suffix()) &&
            !aliases.supply_retention_observation(root64,alias_metadata[0].generation,5,4),
            "eviction invalidated retained metadata or accepted stale observation");
        const auto exact_input=root129->token_suffix();
        require(root129->token_count()==exact_input.size() && root129->matches_tokens(exact_input),
            "request paged token identity disagrees with exported suffix");
        require(input_cache.admit_input_authority(root129,exact_input,unbounded).admitted,
            "exact input authority refused");
        const auto retained_roots=input_cache.roots();
        for(int mutation=0;mutation<3;++mutation) {
            auto invalid=exact_input;
            if(mutation==0)invalid[64]=(invalid[64]+1)%kVocab;
            else if(mutation==1)invalid.pop_back();
            else invalid.push_back(1);
            bool mismatch_refused=false;
            try{input_cache.admit_input_authority(root129,invalid,unbounded);}
            catch(const std::invalid_argument&){mismatch_refused=true;}
            require(mismatch_refused && input_cache.roots()==retained_roots,
                "input admission mismatch altered retained authority");
        }
    }
    {
        Index selections(3);
        selections.publish(root64,contract);selections.publish(root129,contract);
        const auto before=selections.retention_metadata();
        require(before[0].lookup_selections==0 && before[1].lookup_selections==0,
            "publication invented lookup selections");
        require(selections.lookup(*serving,root64->token_suffix(),contract)==root64 &&
            selections.lookup_longest(*serving,query192,contract,64)==root129,
            "selection counter fixture lost exact or longest root");
        require(!selections.lookup(*serving,query192,contract) &&
            !selections.lookup_longest(*serving,query192,contract+"/missing",64),
            "selection counter fixture miss unexpectedly hit");
        const auto after=selections.retention_metadata();
        require(after[0].lookup_selections==1 && after[1].lookup_selections==1 &&
            !after[0].observed_accesses && !after[1].observed_accesses &&
            before[0].lookup_selections==0 && before[1].lookup_selections==0,
            "lookup selection counted candidates/misses, invented reuse or mutated snapshot");
        require(selections.admit(root192,contract,unbounded,unbounded,0).admitted,
            "selection move fixture admission refused");
        const auto moved=selections.retention_metadata();
        require(moved.size()==3 && moved[0].lookup_selections==1 &&
            moved[1].lookup_selections==1 && moved[2].lookup_selections==0,
            "admission moves lost retained selection counts");
        selections.publish(root64,contract);
        const auto replaced=selections.retention_metadata();
        require(replaced[0].root==root129 && replaced[0].lookup_selections==1 &&
            replaced[2].root==root64 && replaced[2].lookup_selections==0 &&
            replaced[2].generation!=before[0].generation,
            "replacement inherited selections or damaged surviving entry");
    }
    {
        Index concurrent_selections(2);
        concurrent_selections.publish(root64,contract);
        concurrent_selections.publish(root129,contract);
        const auto exact_tokens=root64->token_suffix();
        std::atomic<bool> failed{false};
        const auto reader=[&] {
            try {
                std::uint64_t previous_exact=0,previous_longest=0;
                for(unsigned iteration=0;iteration<32;++iteration) {
                    if(concurrent_selections.lookup(*serving,exact_tokens,contract)!=root64 ||
                        concurrent_selections.lookup_longest(*serving,query192,contract,64)!=root129)
                        failed.store(true,std::memory_order_relaxed);
                    const auto snapshot=concurrent_selections.retention_metadata();
                    // Each entry is monotonic. A snapshot is not an atomic
                    // observation of all counters at one common instant.
                    if(snapshot.size()!=2) {failed.store(true,std::memory_order_relaxed);return;}
                    if(snapshot[0].lookup_selections<previous_exact ||
                        snapshot[1].lookup_selections<previous_longest ||
                        snapshot[0].observed_accesses || snapshot[1].observed_accesses)
                        failed.store(true,std::memory_order_relaxed);
                    previous_exact=snapshot[0].lookup_selections;
                    previous_longest=snapshot[1].lookup_selections;
                }
            } catch(...) {failed.store(true,std::memory_order_relaxed);}
        };
        {
            // Scope-bound joins also protect fixture owners if thread creation throws.
            std::jthread first_reader(reader),second_reader(reader);
            std::jthread third_reader(reader),fourth_reader(reader);
        }
        const auto final_counts=concurrent_selections.retention_metadata();
        require(!failed.load(std::memory_order_relaxed) && final_counts.size()==2 &&
            final_counts[0].lookup_selections==128 && final_counts[1].lookup_selections==128,
            "concurrent prefix lookups lost selections or metadata observations regressed");
    }
    {
        Index replacement_race(1);replacement_race.publish(root64,contract);
        const auto stale=replacement_race.retention_metadata().front();
        replacement_race.publish(root64,contract);
        const auto exact_tokens=root64->token_suffix();
        std::atomic<bool> failed{false};
        const auto publisher=[&] {
            try {
                for(unsigned iteration=0;iteration<64;++iteration)
                    replacement_race.publish(root64,contract);
            } catch(...) {failed.store(true,std::memory_order_relaxed);}
        };
        const auto observer=[&] {
            try {
                std::uint64_t previous_generation=0;
                for(unsigned iteration=0;iteration<64;++iteration) {
                    const auto snapshot=replacement_race.retention_metadata().front();
                    if(snapshot.generation<previous_generation || snapshot.root!=root64 ||
                        !snapshot.root->matches_tokens(exact_tokens) ||
                        replacement_race.lookup(*serving,exact_tokens,contract)!=root64 ||
                        replacement_race.supply_retention_observation(stale.root,stale.generation,99,99))
                        failed.store(true,std::memory_order_relaxed);
                    previous_generation=snapshot.generation;
                    const bool accepted=replacement_race.supply_retention_observation(
                        snapshot.root,snapshot.generation,17,0);
                    const auto after=replacement_race.retention_metadata().front();
                    // Replacement may occur on either side of observation supply.
                    // Only an unchanged generation can inherit this observation.
                    if(accepted && after.generation==snapshot.generation &&
                        (after.preparation_microseconds!=17 || after.observed_accesses!=0))
                        failed.store(true,std::memory_order_relaxed);
                }
            } catch(...) {failed.store(true,std::memory_order_relaxed);}
        };
        {
            std::jthread writer(publisher),reader(observer);
        }
        const auto retired=replacement_race.retention_metadata().front();
        replacement_race.publish(root64,contract);
        const auto fresh_generation=replacement_race.retention_metadata().front();
        require(!failed.load(std::memory_order_relaxed) &&
            fresh_generation.generation>retired.generation &&
            !fresh_generation.preparation_microseconds && !fresh_generation.observed_accesses &&
            fresh_generation.lookup_selections==0 &&
            !replacement_race.supply_retention_observation(retired.root,retired.generation,23,1) &&
            stale.root->matches_tokens(exact_tokens),
            "concurrent replacement crossed observation generation or invalidated retained root");
    }
    retained_index.publish(root129,contract);
    const auto retained_lookup=retained_index.lookup_longest(*serving,query192,contract,64);
    retained_index.publish(root64,contract);
    require(retained_lookup==root129 && retained_lookup->state()->same_payload(*root129->state()) &&
        retained_index.lookup_longest(*serving,query192,contract,64)==root64,
        "prefix eviction invalidated an already returned root");
    {
        // T229 observations describe actual selections, not a calibrated value
        // score. A longer root need not be the more frequently selected root.
        Index skewed(2);skewed.publish(root64,contract);skewed.publish(root192,contract);
        const auto short_tokens=root64->token_suffix();
        for(unsigned reuse=0;reuse<12;++reuse)
            require(skewed.lookup(*serving,short_tokens,contract)==root64,
                "repeated short-prefix observation missed");
        require(skewed.lookup(*serving,query192,contract)==root192,
            "rare long-prefix observation missed");
        const auto observations=skewed.retention_metadata();
        require(observations.size()==2 && observations[0].reusable_tokens==64 &&
            observations[1].reusable_tokens==192 && observations[0].lookup_selections==12 &&
            observations[1].lookup_selections==1 &&
            !observations[0].preparation_microseconds && !observations[1].preparation_microseconds &&
            !observations[0].observed_accesses && !observations[1].observed_accesses,
            "retention metadata conflated root size, selections and measured benefit");
        require(skewed.roots()==std::vector<std::shared_ptr<const Request>>{root64,root192},
            "retention observations silently changed FIFO ordering");
    }
    {
        for(unsigned scenario=0;scenario<4;++scenario) {
            Index policy(2);policy.publish(root64,contract);policy.publish(root129,contract);
            const auto snapshot=policy.retention_metadata();
            std::array<Index::RetentionDecision,2> decisions{{
                {root64,snapshot[0].generation,scenario==0,9},
                {root129,snapshot[1].generation,scenario==0,1}}};
            if(scenario==2)decisions[0].value_per_byte.reset();
            if(scenario==3)decisions[1].value_per_byte=9;
            const auto trimmed=policy.trim_with_retention_policy(bytes129,unbounded,0,decisions);
            require(trimmed.inputs_current && same_storage(trimmed.storage,policy.storage_stats()),
                "retention policy lost input or accounting authority");
            if(scenario==0)
                require(!trimmed.budget_met && trimmed.evicted==0 && policy.size()==2,
                    "all-required policy reclaimed a required entry");
            else
                require(trimmed.budget_met && trimmed.evicted==1 && policy.roots().front()==
                    (scenario==1?root64:root129),
                    "retention policy lost value ordering or deterministic FIFO fallback");
        }
        {
            Index admission_policy(2);admission_policy.publish(root64,contract);
            admission_policy.publish(root129,contract);
            const auto before=admission_policy.retention_metadata();
            std::array<Index::RetentionDecision,2> supplied{{
                {root64,before[0].generation,true,0},
                {root129,before[1].generation,true,1}}};
            const auto all_required=admission_policy.admit(root192,contract,unbounded,unbounded,0,
                std::span<const Index::RetentionDecision>(supplied));
            require(!all_required.admitted && admission_policy.roots()==
                std::vector<std::shared_ptr<const Request>>{root64,root129},
                "policy admission evicted required entries");
            const auto required_replacement=admission_policy.admit(root64,contract,unbounded,unbounded,0,
                std::span<const Index::RetentionDecision>(supplied));
            require(!required_replacement.admitted && admission_policy.retention_metadata()[0].generation==before[0].generation,
                "policy admission replaced required generation");
            supplied[1].required=false;
            const auto accepted=admission_policy.admit(root192,contract,unbounded,unbounded,0,
                std::span<const Index::RetentionDecision>(supplied));
            require(accepted.admitted && accepted.evicted==1 && admission_policy.roots()==
                std::vector<std::shared_ptr<const Request>>{root64,root192} &&
                same_storage(accepted.storage,admission_policy.storage_stats()),
                "policy admission failed atomic protected-root selection");
        }
        for(unsigned scenario=0;scenario<3;++scenario) {
            Index value_admission(2);value_admission.publish(root64,contract);
            value_admission.publish(root129,contract);
            const auto previous=value_admission.retention_metadata();
            std::array<Index::RetentionDecision,2> supplied{{
                {root64,previous[0].generation,false,9},
                {root129,previous[1].generation,false,1}}};
            if(scenario==1)supplied[0].value_per_byte.reset();
            if(scenario==2)supplied[1].value_per_byte=9;
            const auto admission=value_admission.admit(root192,contract,unbounded,unbounded,0,
                std::span<const Index::RetentionDecision>(supplied));
            require(admission.admitted && admission.evicted==1 && value_admission.roots()==
                std::vector<std::shared_ptr<const Request>>{scenario==0?root64:root129,root192},
                "admission ignored supplied value or unknown/equal FIFO fallback");
            const auto stable=value_admission.roots();
            const auto stale=value_admission.admit(root64,contract,unbounded,unbounded,0,
                std::span<const Index::RetentionDecision>(supplied));
            require(!stale.admitted && value_admission.roots()==stable &&
                same_storage(stale.storage,value_admission.storage_stats()),
                "stale policy admission mutated current entries");
        }
        {
            Index pressure(4);pressure.publish(root64,contract);pressure.publish(root129,contract);
            const auto roots_before=pressure.roots();
            const auto stats_before=pressure.storage_stats();
            const auto metadata_before=pressure.retention_metadata();
            std::array<Index::RetentionDecision,2> supplied{{
                {root64,metadata_before[0].generation,true,9},
                {root129,metadata_before[1].generation,false,1}}};
            const auto blocked=pressure.admit(root192,contract,bytes192,unbounded,0,
                std::span<const Index::RetentionDecision>(supplied));
            require(!blocked.admitted && pressure.roots()==roots_before &&
                same_storage(stats_before,pressure.storage_stats()),
                "byte-pressure refusal partially evicted cache");
            supplied[0].required=false;
            pressure.fail_next_policy_commit_for_test();
            bool failed=false;
            try{pressure.admit(root192,contract,bytes192,unbounded,0,
                std::span<const Index::RetentionDecision>(supplied));}
            catch(const std::bad_alloc&){failed=true;}
            require(failed && pressure.roots()==roots_before &&
                same_storage(stats_before,pressure.storage_stats()) &&
                pressure.retention_metadata()[0].generation==metadata_before[0].generation,
                "policy admission allocation failure mutated old roots");
            const auto retry=pressure.admit(root192,contract,bytes192,unbounded,0,
                std::span<const Index::RetentionDecision>(supplied));
            require(retry.admitted && retry.evicted==2 && pressure.roots()==
                std::vector<std::shared_ptr<const Request>>{root192} &&
                retry.storage.accounted_bytes==bytes192,
                "policy admission byte-pressure retry failed");
        }
        Index alias_policy(2);alias_policy.publish(root64,contract);
        const auto single_bytes=alias_policy.storage_stats().accounted_bytes;
        alias_policy.publish(root64,contract+"/alias");
        const auto aliases=alias_policy.retention_metadata();
        const std::array<Index::RetentionDecision,2> alias_decisions{{
            {root64,aliases[0].generation,true,9},
            {root64,aliases[1].generation,false,1}}};
        const auto prior_storage=alias_policy.storage_stats();
        for(unsigned invalid=0;invalid<4;++invalid) {
            auto supplied=alias_decisions;
            if(invalid==1)std::swap(supplied[0],supplied[1]);
            if(invalid==2)supplied[1]=supplied[0];
            if(invalid==3)supplied[1].root=root129;
            auto view=std::span<const Index::RetentionDecision>(supplied);
            if(invalid==0)view=view.first(1);
            const auto refusal=alias_policy.trim_with_retention_policy(1,unbounded,0,view);
            require(!refusal.inputs_current && refusal.evicted==0 &&
                same_storage(prior_storage,alias_policy.storage_stats()) &&
                alias_policy.retention_metadata()[1].generation==aliases[1].generation,
                "invalid retention snapshot changed alias ownership");
        }
        alias_policy.fail_next_policy_commit_for_test();
        bool commit_failed=false;
        try {alias_policy.trim_with_retention_policy(single_bytes,unbounded,0,alias_decisions);}
        catch(const std::bad_alloc&) {commit_failed=true;}
        require(commit_failed && same_storage(prior_storage,alias_policy.storage_stats()) &&
            alias_policy.retention_metadata()[1].generation==aliases[1].generation,
            "policy allocation failure mutated cache before commit");
        const auto alias_trim=alias_policy.trim_with_retention_policy(single_bytes,unbounded,0,alias_decisions);
        require(alias_trim.inputs_current && alias_trim.budget_met && alias_trim.evicted==1 &&
            alias_trim.storage.payload_allocated_bytes==prior_storage.payload_allocated_bytes &&
            alias_trim.storage.accounted_bytes==single_bytes &&
            alias_policy.lookup(*serving,root64->token_suffix(),contract)==root64 &&
            !alias_policy.lookup(*serving,root64->token_suffix(),contract+"/alias"),
            "alias eviction claimed shared payload savings or lost required identity");
        Index stale_policy(1);stale_policy.publish(root64,contract);
        const auto old=stale_policy.retention_metadata().front();
        const std::array<Index::RetentionDecision,1> decision{{{root64,old.generation,false,0}}};
        stale_policy.publish(root64,contract);
        const auto refused=stale_policy.trim_with_retention_policy(1,unbounded,0,decision);
        require(!refused.inputs_current && !refused.evicted && stale_policy.size()==1,
            "stale retention permission evicted replacement root");
    }
    {
        Cache policy_cache(Cache::Policy{2,64,unbounded,reserve},identity);
        require(policy_cache.admit_input_authority(root64,root64->token_suffix(),unbounded).admitted &&
            policy_cache.admit_input_authority(root129,root129->token_suffix(),unbounded).admitted,
            "serving retention policy fixture admission failed");
        const auto observed=policy_cache.retention_metadata();
        std::array<Index::RetentionDecision,2> decisions{{
            {root64,observed[0].generation,true,1},
            {root129,observed[1].generation,false,9}}};
        const auto constrained=policy_cache.trim_with_retention_policy(reserve+bytes129,decisions);
        require(constrained.inputs_current && constrained.budget_met && constrained.evicted==1 &&
            policy_cache.roots()==std::vector<std::shared_ptr<const Request>>{root64} &&
            observed[1].root->matches_tokens(root129->token_suffix()),
            "serving policy ignored required root or invalidated evicted reader");
        const auto current=policy_cache.retention_metadata().front();
        const std::array<Index::RetentionDecision,1> required{{{root64,current.generation,true,1}}};
        const auto pressured=policy_cache.trim_with_retention_policy(reserve,required);
        require(pressured.inputs_current && !pressured.budget_met && !pressured.evicted,
            "changed physical budget dropped required serving root");
        policy_cache.reset_for_model_reload(identity);
        const auto stale=policy_cache.trim_with_retention_policy(reserve,required);
        require(!stale.inputs_current && stale.storage.entries==0,
            "serving reload accepted stale retention permissions");
    }
    Index transitions(Index::maximum_entries);
    require(transitions.storage_stats().inline_lookup_metadata_bytes==64*sizeof(std::size_t),
        "empty prefix index omitted inline shortlist metadata");
    for(unsigned i=0;i<63;++i)transitions.publish(root192,contract+"/other/"+std::to_string(i));
    transitions.publish(root129,contract);
    require(transitions.size()==64 &&
        transitions.lookup_longest(*serving,query192,contract,64)==root129,
        "length shortlist64 selected incompatible full root");
    transitions.publish(root64,contract);
    require(transitions.size()==Index::maximum_entries &&
        transitions.lookup_longest(*serving,query192,contract,64)==root129,
        "bounded length shortlist lost longer prefix after FIFO replacement");
    const auto bounded_counts=transitions.retention_metadata();
    for(const auto& entry:bounded_counts)
        require(entry.lookup_selections==(entry.root==root129?2ULL:0ULL),
            "bounded search counted rejected candidates or lost selected-root count");
    require(transitions.erase(*serving,root192->token_suffix(),contract+"/other/1") &&
        transitions.size()==Index::maximum_entries-1 &&
        transitions.lookup_longest(*serving,query192,contract,64)==root129,
        "bounded erase retained stale shortlist positions");
    require(transitions.admit(root192,contract,unbounded,unbounded,0).admitted &&
        transitions.lookup_longest(*serving,query192,contract,64)==root192,
        "admission did not update longest prefix");
    transitions.publish(root129,contract);
    require(transitions.size()==Index::maximum_entries &&
        transitions.lookup_longest(*serving,query192,contract,64)==root192,
        "replacement reordered longest-prefix precedence");
    std::size_t shortlist_evicted=0;
    const auto trimmed_shortlist=transitions.trim_to_budget(bytes129,unbounded,0,&shortlist_evicted);
    require(shortlist_evicted>0 && transitions.size()<=64 &&
        trimmed_shortlist.accounted_bytes<=bytes129 &&
        transitions.lookup_longest(*serving,query192,contract,64)==root129,
        "budget trim retained stale length shortlist or evicted newest fitting root");
    const auto same_root_admission=transitions.admit(root129,contract,bytes129,unbounded,0);
    require(same_storage(same_root_admission.storage,transitions.storage_stats()),
        "replacement admission returned stale storage accounting");
    require(same_root_admission.admitted && same_root_admission.replaced &&
        transitions.lookup_longest(*serving,query192,contract,64)==root129,
        "admit replacement invalidated shortlist");
    transitions.clear();
    require(transitions.storage_stats().inline_lookup_metadata_bytes==64*sizeof(std::size_t) &&
        transitions.storage_stats().accounted_bytes==0,
        "prefix clear confused inline owner metadata with evictable storage");
    require(!transitions.lookup_longest(*serving,query192,contract,64),"clear retained shortlist entry");
    transitions.publish(root64,contract);
    require(transitions.lookup_longest(*serving,query192,contract,64)==root64,
        "publish after clear used stale shortlist position");

    const auto roots=cache.roots();
    require(roots.empty(),"T72 pressure-cleared root snapshot");
    std::cout<<"T72_PREFIX_SERVING_T0 PASS exact_requests=4 distinct_tails=2"
        <<" first_avoided="<<first.metrics.reused_prompt_tokens
        <<" second_avoided="<<second.metrics.reused_prompt_tokens
        <<" second_executed="<<second.metrics.executed_prompt_tokens
        <<" second_lookup_ms="<<second.metrics.lookup_ms
        <<" second_attach_ms="<<second.metrics.attach_ms
        <<" cache_payload_bytes="<<first.metrics.cache_storage.payload_allocated_bytes
        <<" cache_identity_bytes="<<first.metrics.cache_storage.identity_allocated_bytes
        <<" bytes64="<<bytes64<<" bytes129="<<bytes129<<" bytes192="<<bytes192
        <<" fifo_evicted="<<replacement.evicted
        <<" pressure_evicted="<<pressure_evictions
        <<" cancellation=rollback reload=cleared oversize=unchanged reserve_cap=pass"
        <<std::endl;
}
