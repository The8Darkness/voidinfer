#pragma once

#include "exl3/vericache_serving_coordinator.h"
#include "exl3/text_model.h"
#include "exl3/dflash2_draft.h"
#include "core/arena.h"
#include <array>
#include <stdexcept>
#include <string_view>

inline void run_device_logical_invalid_handle_host_only() {
    using namespace ninfer::exl3;
    using Coordinator=Exl3VeriCacheServingCoordinator;
    Exl3VeriCacheServingIdentity identity{"device-logical-host","weights","tokens","exact","text"};
    Exl3VeriCacheServingPrefixCache cache({1,64,2ULL<<30,8ULL<<30},identity);
    Coordinator authority(cache,{1,1,16ULL<<30,8ULL<<30});
    const auto rejects=[](auto&& action) {
        try {action();} catch(const std::invalid_argument&) {return true;}
        return false;
    };
    const auto owner=std::make_shared<int>(1);
    const Coordinator::Lease host{};
    const Coordinator::DeviceLogicalLease device{};
    const std::array<std::int64_t,1> token{7};
    if(!rejects([&]{(void)authority.enter_device_logical(host,owner,0,1,256);}) ||
       !rejects([&]{(void)authority.publish_device_logical_window(device,token);}) ||
       !rejects([&]{authority.cancel_device_logical(device,true);}) ||
       !rejects([&]{authority.yield(host);}) ||
       !rejects([&]{authority.complete(host);}))
        throw std::runtime_error("device logical missing lease accepted without numerical state");
    authority.close();
}

// Model-backed fixture: prepared for the owner-controlled model/GPU gate.
// All assertions after root construction exercise coordinator metadata only.
inline void run_device_logical_model_contract(const char* target_path,
    const char* draft_path) {
    using namespace ninfer::exl3;
    using Coordinator=Exl3VeriCacheServingCoordinator;
    using Request=Exl3VeriCacheRequest;
    constexpr int max_context=256;
    auto model=Exl3TextModel::load(target_path,max_context);
    auto draft=Exl3Dflash2DraftModel::load(draft_path);
    struct TapBuffer {
        void* p=nullptr;
        explicit TapBuffer(std::size_t bytes) {
            if(cudaMalloc(&p,bytes)!=cudaSuccess)
                throw std::runtime_error("device logical tap staging allocation");
        }
        ~TapBuffer() {if(p)(void)cudaFree(p);}
    };
    std::array<std::unique_ptr<TapBuffer>,5> staging_storage;
    std::array<std::uint16_t*,5> staging{};
    for(std::size_t tap=0;tap<staging.size();++tap) {
        staging_storage[tap]=std::make_unique<TapBuffer>(16ULL*5120*2);
        staging[tap]=static_cast<std::uint16_t*>(staging_storage[tap]->p);
    }
    auto context=model->create_context(true);
    context->prepare_continuation(8);
    const std::array<std::int64_t,16> prompt{
        100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115};
    const std::array<std::int64_t,3> suffix{201,202,203};
    const std::array<std::int64_t,3> wrong_suffix{201,202,204};
    const auto root=Request::initialize(*context,prompt)->compact_draft(*draft,staging);
    const auto child=root->append_prompt_compact(*context,*draft,staging,suffix,8);
    const auto wrong_child=root->append_prompt_compact(*context,*draft,staging,wrong_suffix,8);
    const auto rejects=[](auto&& action) {
        try {action();} catch(const std::invalid_argument&) {return true;}
        return false;
    };
    const auto make_authority=[&](Exl3VeriCacheServingPrefixCache& cache,
        std::shared_ptr<const Request> initial) {
        auto authority=std::make_unique<Coordinator>(cache,Coordinator::Policy{1,1,16ULL<<30,8ULL<<30});
        authority->bind_physical_resources({},Exl3ResourceInventory::unlimited());
        authority->admit(std::move(initial));
        return authority;
    };
    Exl3VeriCacheServingIdentity identity{"device-logical-model","weights","tokens","exact","text"};
    Exl3VeriCacheServingPrefixCache cache({1,64,2ULL<<30,8ULL<<30},identity);
    auto authority=make_authority(cache,root);
    const auto host=authority->acquire();
    if(!host)throw std::runtime_error("device logical fixture did not acquire host root");
    const auto owner=std::make_shared<int>(1);
    if(!rejects([&]{(void)authority->enter_device_logical(*host,{},16,3,max_context);}) ||
       !rejects([&]{(void)authority->enter_device_logical(*host,owner,17,3,max_context);}) ||
       !rejects([&]{(void)authority->enter_device_logical(*host,owner,16,0,max_context);}) ||
       !rejects([&]{(void)authority->enter_device_logical(*host,owner,16,241,max_context);}))
        throw std::runtime_error("device logical entry accepted invalid owner/frontier/capacity");
    const auto baseline=authority->stats();
    auto device=authority->enter_device_logical(*host,owner,16,3,max_context);
    if(authority->compute_leases_current(std::span<const Coordinator::Lease>(&*host,1)) ||
       !rejects([&]{authority->yield(*host);}) ||
       !rejects([&]{authority->complete(*host);}) ||
       !rejects([&]{(void)authority->publish_window(*host,child,suffix);}) ||
       !rejects([&]{(void)authority->complete_and_admit(*host,root);}))
        throw std::runtime_error("device logical entry left host lease usable");
    auto false_owner=device;
    false_owner.physical_owner=std::shared_ptr<const void>(owner.get(),[](const void*){});
    auto false_frontier=device;++false_frontier.frontier;
    const std::array<std::int64_t,2> first{201,202};
    const std::array<std::int64_t,2> excessive{203,204};
    const std::array<std::int64_t,1> invalid{-1};
    if(!rejects([&]{(void)authority->publish_device_logical_window(false_owner,first);}) ||
       !rejects([&]{(void)authority->publish_device_logical_window(false_frontier,first);}) ||
       !rejects([&]{(void)authority->publish_device_logical_window(device,{});}) ||
       !rejects([&]{(void)authority->publish_device_logical_window(device,invalid);}))
        throw std::runtime_error("device logical preflight accepted forged owner/frontier/tokens");
    const auto first_result=authority->publish_device_logical_window(device,first);
    if(first_result.committed_tokens!=2 || first_result.lease.frontier!=18 ||
       authority->stats().resident_updates!=baseline.resident_updates ||
       !rejects([&]{(void)authority->publish_device_logical_window(device,first);}) ||
       !rejects([&]{(void)authority->publish_device_logical_window(first_result.lease,excessive);}))
        throw std::runtime_error("device logical publication lost generation, frontier or capacity");
    const std::array<std::int64_t,1> last{203};
    device=authority->publish_device_logical_window(first_result.lease,last).lease;
    if(authority->stats().resident_updates!=baseline.resident_updates ||
       !rejects([&]{(void)authority->materialize_device_logical(device,root);}) ||
       !rejects([&]{(void)authority->materialize_device_logical(device,wrong_child);}) ||
       authority->stats().resident_updates!=baseline.resident_updates)
        throw std::runtime_error("device logical invalid terminal child changed residency");
    authority->fail_next_residency_for_test(Coordinator::ResidencyFault::before_commit);
    bool injected=false;
    try {(void)authority->materialize_device_logical(device,child);}
    catch(const std::runtime_error& error) {
        injected=std::string_view(error.what())=="injected coordinator before resident commit";
    }
    if(!injected || authority->stats().resident_updates!=baseline.resident_updates)
        throw std::runtime_error("device logical failed terminal replacement changed authority");
    const auto materialized=authority->materialize_device_logical(device,child);
    if(materialized.root!=child || materialized.ticket.generation<=device.ticket.generation ||
       authority->stats().resident_updates!=baseline.resident_updates+1 ||
       !rejects([&]{(void)authority->publish_device_logical_window(device,last);}))
        throw std::runtime_error("device logical terminal materialization lost host authority");
    authority->complete(materialized);
    authority->close();

    // Resident metadata credits belong to the first coordinator even after
    // completion. A second authority needs a newly constructed exact root;
    // re-admitting the earlier root would attempt to attach foreign credits.
    auto cancellation_context=model->create_context(true);
    cancellation_context->prepare_continuation(8);
    const auto cancellation_root=Request::initialize(*cancellation_context,prompt)->
        compact_draft(*draft,staging);
    Exl3VeriCacheServingPrefixCache cancellation_cache({1,64,2ULL<<30,8ULL<<30},identity);
    auto cancellation=make_authority(cancellation_cache,cancellation_root);
    const auto cancel_host=cancellation->acquire();
    if(!cancel_host)throw std::runtime_error("device logical cancellation fixture acquire");
    auto cancel_device=cancellation->enter_device_logical(*cancel_host,owner,16,3,max_context);
    cancel_device=cancellation->publish_device_logical_window(cancel_device,last).lease;
    bool boundary=false;
    try {cancellation->cancel_device_logical(cancel_device,false);}
    catch(const std::logic_error&) {boundary=true;}
    if(!boundary || cancellation->stats().active!=1)
        throw std::runtime_error("device logical cancellation bypassed worker boundary");
    cancellation->cancel_device_logical(cancel_device,true);
    if(cancellation->stats().active!=0 || cancellation->stats().admitted!=0 ||
       !rejects([&]{cancellation->cancel_device_logical(cancel_device,true);}))
        throw std::runtime_error("device logical cancellation retained live authority");
    cancellation->close();
}
