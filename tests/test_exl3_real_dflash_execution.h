#pragma once
#include "exl3/device_prefix_cache.h"
#include "exl3/shared_conditioning_scope.h"
#include "exl3/engine_greedy_packet_batch.h"
#include <thread>
#include <tuple>

void run_real_dflash_execution(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Lane=ninfer::exl3::Exl3Dflash2Execution;
    using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Root=std::shared_ptr<const Request>;
    using Cache=ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity=ninfer::exl3::Exl3VeriCacheServingIdentity;
    using ninfer::exl3::Exl3GreedyBatchSource;
    using ninfer::exl3::Exl3GreedyPacket;
    using ninfer::exl3::Exl3GreedyRow;
    using Clock=std::chrono::steady_clock;
    {
        ninfer::exl3::RetainedDescriptorLedger ledger;
        Lane::Pending source;source.verification.committed_tokens.reserve(16);
        source.verification.committed_tokens={1,2,3};
        source.verification.committed_taps[0].reserve(32);
        source.verification.committed_taps[0]={4,5};
        const auto blocks=ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>();
        bool refused=false;
        try{(void)Lane::copy_pending_reserved(source,[&](std::uint64_t)->ninfer::exl3::RetainedDescriptorLedger::Ticket {
            require(ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>()==blocks,
                "Pending copy allocated before reservation");throw std::bad_alloc();
        });}catch(const std::bad_alloc&){refused=true;}
        require(refused && ledger.bytes()==0,"Pending copy refusal leaked credit");
        std::vector<std::int64_t> larger_tokens(64,7);
        refused=false;
        try{(void)Lane::copy_pending_reserved(source,[&](std::uint64_t bytes){
            source.verification.committed_tokens.swap(larger_tokens);
            return ledger.acquire(bytes);
        });}catch(const std::invalid_argument&){refused=true;}
        require(refused && ledger.bytes()==0 &&
            ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>()==blocks,
            "Pending source growth during reservation reached copy allocation");
        source.verification.committed_tokens.swap(larger_tokens);
        refused=false;
        try{(void)Lane::copy_pending_reserved(source,[&](std::uint64_t bytes){return ledger.acquire(bytes-1);});}
        catch(const std::invalid_argument&){refused=true;}
        require(refused && ledger.bytes()==0 &&
            ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>()==blocks,
            "underfunded Pending copy allocated or retained ownership");
        auto copied=Lane::copy_pending_reserved(source,[&](std::uint64_t bytes){return ledger.acquire(bytes);});
        const auto actual=copied->payload_extent(ninfer::exl3::bounded_shared_allocation_bytes<Lane::Pending>());
        require(actual && ledger.bytes()==*actual && copied->verification.committed_tokens==source.verification.committed_tokens,
            "Pending copy actual capacity charge");
        auto independent=Lane::copy_pending_reserved(*copied,[&](std::uint64_t bytes){return ledger.acquire(bytes);});
        const auto independent_bytes=independent->payload_extent(ninfer::exl3::bounded_shared_allocation_bytes<Lane::Pending>());
        require(independent_bytes && ledger.bytes()==*actual+*independent_bytes &&
            independent->verification.committed_tokens.data()!=copied->verification.committed_tokens.data(),
            "independent Pending copy reused payload allocation or credit");
        auto moved=std::move(independent);
        require(!independent && ledger.bytes()==*actual+*independent_bytes,
            "Pending owner move changed allocation charge");
        moved.reset();require(ledger.bytes()==*actual,"independent Pending retirement changed source charge");
        std::weak_ptr<Lane::Pending> weak=copied;copied.reset();
        require(ledger.bytes()==ninfer::exl3::bounded_shared_allocation_bytes<Lane::Pending>(),
            "Pending copy weak lifetime lost control credit or retained payload credit");
        weak.reset();require(ledger.bytes()==0,"Pending copy final weak credit leak");
    }
    const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
    const int prefix=env("NINFER_REAL_DFLASH_PREFIX").empty()?512:std::stoi(env("NINFER_REAL_DFLASH_PREFIX"));
    const int outputs=env("NINFER_REAL_DFLASH_OUTPUTS").empty()?32:std::stoi(env("NINFER_REAL_DFLASH_OUTPUTS"));
    const int pairs=env("NINFER_REAL_DFLASH_PAIRS").empty()?1:std::stoi(env("NINFER_REAL_DFLASH_PAIRS"));
    const std::filesystem::path output=env("NINFER_REAL_DFLASH_OUT");
    const auto decode_nvtx=env("NINFER_REAL_DFLASH_NVTX");
    require(decode_nvtx.empty() || decode_nvtx=="0" || decode_nvtx=="1",
        "real DFlash NVTX must be 0 or 1");
    const bool decode_nvtx_enabled=decode_nvtx=="1";
    require(!output.empty() && !std::filesystem::exists(output),"real DFlash new receipt");
    require(prefix>=16 && prefix+outputs<=target.max_context() && outputs>=8 && pairs>=1,"real DFlash extents");
    std::filesystem::create_directories(output);
    std::ofstream checks(output/"checks.csv"),runs(output/"runs.csv"),gaps(output/"releases.csv"),workloads(output/"workloads.csv");
    workloads<<"fixture,pair,arm,requests,useful_outputs,wall_ms,context_constructions,final_exact\n";
    checks<<"fixture,case,pass\n";
    runs<<"fixture,pair,arm,wall_ms,construct_ms,acquire_ms,decode_ms,release_ms,teardown_ms,proposal_calls,proposed_rows,verified_rows,replayed_rows,accepted_rows,committed_rows,publications,first_release_ms,proposal_ms,l2_ms,registry_ms,persistent_bytes,h2d,d2h,suffix_calls,suffix_rows,suffix_misses,suffix_metadata_bytes,horizon_policy_enabled,final_horizon,observed_suffix,failed_suffix,censored_suffix,selector_batched_anchor_chain_calls,fast_mia_parity_w1_settlements,gdn_graph_replays,gdn_graph_launch_cpu_ns,full_layer_graph_replays,full_layer_graph_launch_cpu_ns,exact\n";
    gaps<<"fixture,pair,arm,publication,committed,elapsed_ms\n";
    const auto gate=[&](const char* fixture,const char* name,bool pass){checks<<fixture<<','<<name<<','<<pass<<'\n';checks.flush();require(pass,std::string("real DFlash ")+name);};
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    require(GlobalMemoryStatusEx(&memory) && memory.ullAvailPhys>(8ULL<<30),"real DFlash physical reserve");
    Identity identity{"solo-real-dflash","SC_6.00bpw_H6_V6","pinned-token-ids","exact-B8-greedy","text"};
    Cache cache(Cache::Policy{2,64,2ULL<<30,8ULL<<30},identity);
    Coordinator coordinator(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
    const bool conditional_credit_fixture=env("NINFER_TEST_CONDITIONAL_RETENTION_CREDIT")=="1";
    const bool conditional_credit_refusal=env("NINFER_TEST_CONDITIONAL_RETENTION_REFUSAL")=="1";
    require(!conditional_credit_refusal || conditional_credit_fixture,"conditional refusal requires credit fixture");
    if(conditional_credit_fixture) {
        require(env("NINFER_TEST_CONDITIONAL_B8")=="1","conditional retention credit fixture requires conditional mode");
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        auto limits=Inventory::unlimited();
        limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=
            ninfer::exl3::bounded_shared_allocation_bytes<Lane::Pending>()+
            8*sizeof(std::int64_t)+5ULL*8*5120*sizeof(std::uint16_t);
        if(conditional_credit_refusal)--limits[static_cast<unsigned>(Inventory::Domain::host_metadata)];
        // Component credit fixture only; other physical owners are not claimed
        // inventoried by this deliberately empty initial resource snapshot.
        coordinator.bind_physical_resources({},limits);
    }
    std::array<std::unique_ptr<DeviceBuffer>,5> stage;
    std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap){stage[tap]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[tap]=static_cast<std::uint16_t*>(stage[tap]->get());}
    const auto make_lane=[&]{return std::make_unique<Lane>(target.create_context(true),draft,identity.contract());};
    if(env("NINFER_TEST_COMPACT_HOST_PAYLOAD_ADMISSION")=="1") {
        require(!conditional_credit_fixture && env("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING")!="1",
            "compact host payload fixture requires copied taps and independent accounting");
        require(code.size()>=16,"compact host payload fixture prompt extent");
        // Component fixture: bind admission authority, without claiming the
        // empty inventory accounts for model/device allocations in this test.
        coordinator.bind_physical_resources({},ninfer::exl3::Exl3ResourceInventory::unlimited());
        Root root;
        {
            auto context=target.create_context(true);context->prepare_continuation(8);
            root=Request::initialize(*context,std::span<const std::int64_t>(code.data(),16),1024)
                ->compact_draft(draft,staging);
        }
        auto lane=make_lane();coordinator.admit(root);
        auto lease=coordinator.acquire();require(lease.has_value(),"host payload fixture acquisition");
        lane->acquire(*lease,&coordinator);
        unsigned reservations=0,observations=0;bool reject=false;
        struct ClearPayloadHooks {
            Exl3TextContext& context;
            ~ClearPayloadHooks() {
                context.set_request_host_payload_reservation({});
                context.set_request_host_payload_observer({});
            }
        } clear_hooks{lane->context()};
        lane->context().set_request_host_payload_reservation([&](std::uint64_t bytes) {
            ++reservations;
            if(reject)throw ninfer::exl3::Exl3HostResidentBudgetExhausted{};
            return coordinator.reserve_snapshot_payload(lane->lease(),bytes);
        });
        lane->context().set_request_host_payload_observer([&](const std::shared_ptr<const void>& owner,
            std::size_t plane,const void* data,std::size_t bytes) {
            coordinator.track_snapshot_payload(lane->lease(),owner,plane,data,bytes);++observations;
        });
        const auto proposed=lane->propose(8);
        const std::vector<std::int64_t> tokens(proposed.begin(),proposed.end());
        const auto baseline=lane->verify(tokens);
        gate("host_payload","actual_compact_callbacks",reservations==1 && observations==5);
        lane->abort();reject=true;
        bool refused=false;
        try{(void)lane->verify(tokens);}catch(const ninfer::exl3::Exl3HostResidentBudgetExhausted&){refused=true;}
        gate("host_payload","refusal_before_tap_publication",refused && reservations==2 && observations==5 &&
            lane->context().export_exact_host_state(nullptr,true)->same_payload(*root->state()));
        reject=false;
        const auto retry=lane->verify(tokens);
        gate("host_payload","retry_exact_after_refusal",reservations==3 && observations==10 &&
            retry.verification.committed_tokens==baseline.verification.committed_tokens &&
            retry.root->state()->same_payload(*baseline.root->state()) &&
            retry.root->same_projected_conditioning_for_test(*baseline.root));
        lane->abort();const auto released=lane->lease();lane->release();coordinator.cancel(released.ticket,true);
        lane->context().set_request_host_payload_reservation({});
        lane->context().set_request_host_payload_observer({});
        return;
    }
    if(env("NINFER_TEST_HOST_KV_OWNED_STREAM")=="1") {
        const bool quarantine=env("NINFER_TEST_HOST_KV_OWNED_STREAM_RETIREMENT")=="1";
        require(code.size()>=16,"owned HostKV stream prompt extent");
        struct OwnedStream {
            cudaStream_t stream=nullptr;
            OwnedStream(){cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create owned HostKV fixture stream");}
            ~OwnedStream(){if(stream)cudaStreamDestroy(stream);}
        };
        auto owner=std::make_shared<OwnedStream>();
        const std::weak_ptr<OwnedStream> witness=owner;
        auto context=target.create_context(true,false,false);
        context->retain_execution_stream_owner(owner->stream,owner);
        context->retain_execution_stream_owner(owner->stream,owner);
        {
            auto other=std::make_shared<OwnedStream>();
            bool refused=false;
            try{context->retain_execution_stream_owner(other->stream,other);}
            catch(const std::runtime_error& error){refused=std::string(error.what())=="execution stream owner cannot be replaced";}
            gate("owned_stream","replacement_owner_refused",refused);
            const auto before=context->host_kv_stats();
            bool execution_refused=false;
            try{context->prefill(std::span<const std::int64_t>(code.data(),16),other->stream);}
            catch(const std::runtime_error& error){execution_refused=std::string(error.what())==
                "HostKV forward stream differs from retained stream owner";}
            gate("owned_stream","unowned_stream_refused_before_progress",execution_refused &&
                context->position()==0 && context->host_kv_stats().completed_rows==before.completed_rows &&
                context->host_kv_stats().copy_submissions==before.copy_submissions);
        }
        context->prefill(std::span<const std::int64_t>(code.data(),16),owner->stream);
        gate("owned_stream","correct_stream_works_after_admission_refusal",
            context->position()==16 && context->host_kv_stats().completed_rows==16);
        {
            const auto state=context->export_exact_host_state(owner->stream);
            {
                auto other=std::make_shared<OwnedStream>();
                bool refused=false;
                try{context->restore_exact_host_state(*state,other->stream);}
                catch(const std::runtime_error& error){refused=std::string(error.what())==
                    "HostKV restore stream differs from retained stream owner";}
                gate("owned_stream","wrong_restore_stream_preserves_residency",
                    refused && context->exact_host_state_resident(*state) && context->position()==16);
            }
            context->reset();
            gate("owned_stream","default_reset_after_owned_forward",context->position()==0);
            context->restore_exact_host_state(*state,owner->stream);
            gate("owned_stream","owned_restore_after_default_reset_full_state",
                context->export_exact_host_state(owner->stream)->same_payload(*state));
        }
        owner.reset();
        gate("owned_stream","context_retains_actual_stream_owner",!witness.expired());
        if(quarantine)context->fail_host_kv_compute_retirement_for_test();
        context.reset();
        gate("owned_stream","stream_owner_lifetime_matches_retirement",witness.expired()!=quarantine);
        return;
    }
    if(env("NINFER_TEST_HOST_KV_STREAM_SWITCH_CONTROL")=="1") {
        require(code.size()>=18,"HostKV stream switch control prompt extent");
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> expected;
        {
            auto reference=target.create_context(true,false,false);
            reference->prefill(std::span<const std::int64_t>(code.data(),16));
            reference->decode(code[16]);reference->decode(code[17]);
            expected=reference->export_exact_host_state();
        }
        auto context=target.create_context(true,false,false);
        const auto workspace=context->host_kv_workspace_owner_for_test();
        const auto quarantines=Exl3TextContext::host_kv_quarantined_contexts();
        context->prefill(std::span<const std::int64_t>(code.data(),16));
        cudaStream_t alternate=nullptr;
        cuda_check(cudaStreamCreateWithFlags(&alternate,cudaStreamNonBlocking),"create HostKV alternate control stream");
        context->decode(code[16],alternate);
        context->decode(code[17]);
        gate("stream_switch_control","full_state_matches_same_stream_reference",
            context->export_exact_host_state()->same_payload(*expected));
        context.reset(); // Retire the context before destroying borrowed streams.
        cuda_check(cudaStreamDestroy(alternate),"destroy HostKV alternate control stream");
        gate("stream_switch_control","successful_switch_retires_workspace",
            workspace.expired() && Exl3TextContext::host_kv_quarantined_contexts()==quarantines);
        return;
    }
    if(env("NINFER_TEST_HOST_KV_STREAM_SWITCH_FAILURE")=="1") {
        require(code.size()>=17,"HostKV stream switch prompt extent");
        auto context=target.create_context(true,false,false);
        context->prefill(std::span<const std::int64_t>(code.data(),16));
        auto prior_state=context->export_exact_host_state();
        gate("stream_switch","pre_failure_residency_present",context->exact_host_state_resident(*prior_state));
        const auto before=context->host_kv_stats();
        const auto workspace=context->host_kv_workspace_owner_for_test();
        context->fail_host_kv_compute_retirement_for_test();
        cudaStream_t replacement=nullptr;
        cuda_check(cudaStreamCreateWithFlags(&replacement,cudaStreamNonBlocking),"create replacement HostKV fixture stream");
        bool failed=false;
        try{context->decode(code[16],replacement);}
        catch(const std::runtime_error& error){failed=std::string(error.what()).find("retire prior HostKV forward stream")!=std::string::npos;}
        cuda_check(cudaStreamDestroy(replacement),"destroy unused replacement HostKV fixture stream");
        gate("stream_switch","failed_drain_prevents_new_decode",failed && context->position()==16 &&
            context->host_kv_stats().completed_rows==before.completed_rows &&
            context->host_kv_stats().copy_submissions==before.copy_submissions);
        bool refused=false;
        try{context->decode(code[16]);}
        catch(const std::runtime_error& error){refused=std::string(error.what())=="HostKV decode requires intact transfer lineage";}
        gate("stream_switch","failed_switch_poisoned_lineage",refused);
        gate("stream_switch","failed_switch_cannot_report_reusable_residency",
            !context->exact_host_state_resident(*prior_state));
        prior_state.reset();
        context.reset();
        gate("stream_switch","failed_old_stream_retains_workspace",!workspace.expired());
        const auto retained=Exl3TextContext::host_kv_retirement_snapshot();
        gate("stream_switch","original_default_stream_witness_preserved",
            retained.forward_stream_retained && retained.forward_stream_identity==0 && retained.forward_device>=0);
        return;
    }
    if(env("NINFER_TEST_HOST_KV_PREFILL_BOUNDARY_FAILURE")=="1") {
        const bool compute_failure=env("NINFER_TEST_HOST_KV_COMPUTE_RETIREMENT")=="1";
        const bool compute_only=env("NINFER_TEST_HOST_KV_COMPUTE_ONLY")=="1";
        require(!compute_only || compute_failure,"compute-only retirement requires compute failure variant");
        require(code.size()>=16,"HostKV prefill boundary prompt extent");
        for(unsigned fault:{1u,2u}) {
            if(compute_failure && fault!=1)continue; // Quarantine arm is process-isolated.
            auto context=target.create_context(true,false,false);
            const auto workspace=context->host_kv_workspace_owner_for_test();
            const auto quarantines=Exl3TextContext::host_kv_quarantined_contexts();
            context->fail_host_kv_prefill_boundary_for_test(fault);
            bool failed=false;
            try{context->prefill(std::span<const std::int64_t>(code.data(),16));}
            catch(const std::runtime_error& error){failed=std::string(error.what())==
                (fault==1?"injected HostKV pre-layer prefill failure":"injected HostKV post-forward prefill failure");}
            gate("prefill_boundary","injected_boundary_reached",failed);
            gate("prefill_boundary","public_position_not_committed",context->position()==0);
            gate("prefill_boundary","forward_progress_matches_boundary",context->host_kv_stats().completed_rows==(fault==1?0:16));
            bool refused=false;
            try{context->reset();}
            catch(const std::runtime_error& error){refused=std::string(error.what())==
                "context reset cannot reuse failed HostKV lineage";}
            gate("prefill_boundary","boundary_failure_poisoned_lineage",refused);
            bool retry_refused=false;
            try{context->prefill(std::span<const std::int64_t>(code.data(),16));}
            catch(const std::runtime_error& error){retry_refused=std::string(error.what())==
                "HostKV prefill requires intact transfer lineage";}
            gate("prefill_boundary","retry_refused_before_more_work",retry_refused &&
                context->host_kv_stats().completed_rows==(fault==1?0:16));
            if(compute_failure)context->fail_host_kv_compute_retirement_for_test();
            context.reset();
            gate("prefill_boundary","retirement_matches_injected_outcome",
                workspace.expired()!=compute_failure &&
                Exl3TextContext::host_kv_quarantined_contexts()==quarantines+(compute_failure?1:0));
            if(compute_failure) {
                const auto retained=Exl3TextContext::host_kv_retirement_snapshot();
                gate("prefill_boundary","compute_stream_failure_retained",
                    retained.forward_stream_retained && retained.drain_error==static_cast<int>(cudaErrorUnknown));
                if(compute_only)gate("prefill_boundary","compute_retirement_without_copy_stream",
                    !retained.copy_stream_retained && retained.pinned_bytes==0);
            }
        }
        return;
    }
    if(env("NINFER_TEST_DFLASH_QUARANTINE_PUBLICATION")=="1") {
        const bool hostkv=env("NINFER_TEST_DFLASH_HOSTKV_QUARANTINE")=="1";
        const bool generic=env("NINFER_TEST_DFLASH_GENERIC_QUARANTINE")=="1";
        require(!hostkv || !generic,"publication quarantine domains are exclusive");
        require(code.size()>=16,"quarantine publication prompt extent");
        Root root;
        {
            auto context=target.create_context(true);
            context->prepare_continuation(8);
            root=Request::initialize(*context,std::span<const std::int64_t>(code.data(),16),1024)
                ->compact_draft(draft,staging);
        }
        auto idle=make_lane();
        auto active=make_lane();
        coordinator.admit(root);
        const auto lease=coordinator.acquire();
        require(lease.has_value(),"quarantine publication lease");
        const auto before_acquire=active->stats();
        const auto before_epoch=active->execution_epoch();
        for(bool change_acquisition:{false,true}) {
            auto stale=*lease;
            if(change_acquisition)++stale.acquisition;
            else ++stale.ticket.generation;
            bool refused=false;
            try{active->acquire(stale,&coordinator);}
            catch(const std::invalid_argument& error){refused=std::string_view(error.what())==
                "DFlash execution acquire stale coordinator lease";}
            gate("quarantine_publication","stale_acquisition_before_reset",
                refused && !active->execution_active() && !active->lease().root &&
                active->execution_epoch()==before_epoch &&
                active->stats().acquired_payload_preservations==before_acquire.acquired_payload_preservations &&
                active->stats().acquired_full_resets==before_acquire.acquired_full_resets);
        }
        active->acquire(*lease,&coordinator);
        const auto tokens=active->propose(8);
        const auto pending=active->verify(tokens);
        const auto epoch=active->execution_epoch();
        const auto prior_host_quarantines=Exl3TextContext::host_kv_quarantined_contexts();
        const auto prior_lane_quarantines=Lane::quarantined_execution_count();
        const auto prior_generic_quarantines=Exl3TextContext::generic_quarantined_allocations();
        std::weak_ptr<const void> idle_workspace;
        if(hostkv)idle_workspace=idle->context().host_kv_workspace_owner_for_test();
        if(hostkv)idle->context().fail_host_kv_retirement_for_test();
        else if(!generic)idle->fail_destructor_drain_for_test();
        if(!generic)idle.reset();
        if(generic) {
            const std::array<std::byte,32> payload{};
            bool injected=false;
            try{Exl3TextContext::exercise_generic_upload_for_test(payload,2);}
            catch(const std::runtime_error& error){injected=std::string(error.what())==
                "injected generic post-upload construction failure";}
            gate("quarantine_publication","generic_cleanup_fault_exercised",injected);
            const auto idle_epoch=idle->execution_epoch();
            const auto idle_acquisitions=idle->stats().acquisitions;
            for(unsigned attempt=0;attempt<2;++attempt) {
                bool refused=false;
                try{idle->acquire(*lease);}
                catch(const std::runtime_error& error){refused=std::string(error.what())==
                    "unresolved generic cleanup blocks DFlash execution";}
                gate("quarantine_publication","idle_acquisition_refused_before_mutation",
                    refused && !idle->execution_active() && !idle->lease().root &&
                    idle->execution_epoch()==idle_epoch && idle->stats().acquisitions==idle_acquisitions);
            }
            idle.reset();
        }
        gate("quarantine_publication","intended_retirement_domain_exercised",
            Exl3TextContext::host_kv_quarantined_contexts()==prior_host_quarantines+(hostkv?1:0) &&
            Lane::quarantined_execution_count()==prior_lane_quarantines+((hostkv || generic)?0:1) &&
            Exl3TextContext::generic_quarantined_allocations()==prior_generic_quarantines+(generic?1:0));
        if(hostkv)gate("quarantine_publication","idle_context_workspace_retained",!idle_workspace.expired());
        if(hostkv || generic) {
            const auto direct_error=generic?"unresolved generic cleanup blocks context execution":
                "unresolved HostKV retirement blocks context execution";
            const auto prior_position=active->context().position();
            const auto prior_rows=active->context().continuation_rows();
            const auto prior_transfers=active->context().host_kv_stats();
            bool direct_refused=false;
            try{active->context().decode(code[0]);}
            catch(const std::runtime_error& error){direct_refused=std::string(error.what())==
                direct_error;}
            gate("quarantine_publication","direct_context_decode_refused",direct_refused);
            bool reset_refused=false,restore_refused=false;
            try{active->context().reset();}
            catch(const std::runtime_error& error){reset_refused=std::string(error.what())==
                direct_error;}
            try{active->context().restore_exact_host_state(*root->state());}
            catch(const std::runtime_error& error){restore_refused=std::string(error.what())==
                direct_error;}
            gate("quarantine_publication","direct_context_reset_restore_refused",reset_refused && restore_refused);
            bool guarded_restore_refused=false;
            try{(void)active->context().restore_exact_host_state_if_needed(*root->state());}
            catch(const std::runtime_error& error){guarded_restore_refused=std::string(error.what())==direct_error;}
            gate("quarantine_publication","guarded_restore_cannot_bypass_retirement",guarded_restore_refused);
            const auto request_generation=active->context().request_generation();
            bool preserving_reset_refused=false;
            try{(void)active->context().reset_for_request_preserving(
                "quarantine-must-refuse-before-contract",*root->state(),request_generation);}
            catch(const std::runtime_error& error){preserving_reset_refused=std::string(error.what())==direct_error;}
            gate("quarantine_publication","preserving_reset_cannot_bypass_retirement",
                preserving_reset_refused && active->context().request_generation()==request_generation);
            const auto refuses_quarantine=[&](auto&& operation) {
                try{operation();}
                catch(const std::runtime_error& error){return std::string(error.what())==
                    direct_error;}
                return false;
            };
            gate("quarantine_publication","decode_capture_entry_refused",
                refuses_quarantine([&]{(void)active->context().capture_decode_graph();}));
            gate("quarantine_publication","continuation_capture_entry_refused",
                refuses_quarantine([&]{(void)active->context().capture_continuation_graph();}));
            gate("quarantine_publication","continuation_variant_capture_entry_refused",
                refuses_quarantine([&]{(void)active->context().capture_continuation_graph_rows(4);}));
            gate("quarantine_publication","decode_replay_entry_refused",
                refuses_quarantine([&]{active->context().decode_graph(code[0]);}));
            gate("quarantine_publication","continuation_replay_entry_refused",
                refuses_quarantine([&]{active->context().continue_rows_graph(std::span<const std::int64_t>(code.data(),4));}));
            gate("quarantine_publication","media_entry_refused",
                refuses_quarantine([&]{active->context().append_media_embeddings_numeric({},{});}));
            const auto after_transfers=active->context().host_kv_stats();
            gate("quarantine_publication","refused_entries_preserve_pending_state_and_transfer_counters",
                active->context().position()==prior_position && active->context().continuation_rows()==prior_rows &&
                after_transfers.h2d_bytes==prior_transfers.h2d_bytes &&
                after_transfers.d2h_bytes==prior_transfers.d2h_bytes &&
                after_transfers.copy_submissions==prior_transfers.copy_submissions &&
                after_transfers.transfer_calls==prior_transfers.transfer_calls &&
                after_transfers.completed_rows==prior_transfers.completed_rows);
        }
        const auto before_publication=coordinator.stats();
        bool refused=false;
        try{(void)active->publish(coordinator,pending);}
        catch(const std::runtime_error& error){refused=std::string(error.what())==
            (generic?"unresolved generic cleanup blocks DFlash execution":
                (hostkv?"unresolved HostKV retirement blocks DFlash execution":"unresolved DFlash execution retirement"));}
        gate("quarantine_publication","pending_publication_refused_without_lease_mutation",
            refused && active->lease().root==root && active->lease().acquisition==lease->acquisition &&
            active->execution_epoch()==epoch);
        const auto after_publication=coordinator.stats();
        gate("quarantine_publication","coordinator_publication_and_residency_unchanged",
            after_publication.publications==before_publication.publications &&
            after_publication.resident_updates==before_publication.resident_updates &&
            after_publication.locked_page_bytes==before_publication.locked_page_bytes &&
            after_publication.active==before_publication.active &&
            after_publication.admitted==before_publication.admitted);
        active.reset();
        coordinator.cancel(lease->ticket,true);
        return;
    }
    if(env("NINFER_TEST_HOST_KV_ACQUIRE_RESTORE_FAILURE")=="1") {
        const bool quarantine=env("NINFER_TEST_HOST_KV_ACQUIRE_RETIREMENT")=="1";
        require(code.size()>=16,"HostKV acquire failure prompt extent");
        Root root;
        {
            auto context=target.create_context(true,false,false);
            context->prepare_continuation(8);
            root=Request::initialize(*context,std::span<const std::int64_t>(code.data(),16),1024)
                ->compact_draft(draft,staging);
        }
        auto lane=make_lane();
        std::unique_ptr<Lane> existing_peer;
        if(quarantine)existing_peer=make_lane();
        lane->context().fail_next_host_kv_restore_for_test();
        coordinator.admit(root);
        const auto lease=coordinator.acquire();
        require(lease.has_value(),"HostKV acquire fixture lease");
        const auto epoch=lane->execution_epoch();
        bool failed=false;
        try{lane->acquire(*lease);}
        catch(const std::runtime_error& error){failed=std::string(error.what())==
            "injected HostKV failure after restore upload";}
        gate("hostkv_acquire","actual_restore_failure",failed);
        gate("hostkv_acquire","failed_attempt_retains_source_and_epoch",
            !lane->execution_active() && lane->lease().root==root &&
            lane->lease().acquisition==lease->acquisition && lane->execution_epoch()==epoch+1);
        bool refused=false;
        try{lane->acquire(*lease);}
        catch(const std::invalid_argument&){refused=true;}
        gate("hostkv_acquire","poisoned_lane_cannot_reacquire",refused && lane->execution_epoch()==epoch+1);
        if(quarantine)lane->fail_destructor_drain_for_test();
        lane.reset();
        if(quarantine) {
            const auto retained=Lane::latest_quarantine_for_test();
            const std::weak_ptr<const void> expected_owner=root;
            gate("hostkv_acquire","failed_acquire_quarantine_retains_root_and_draft_lock",
                retained && retained->acquisition==lease->acquisition && retained->execution==epoch+1 &&
                retained->draft_lock_retained && !retained->request_owner.expired() &&
                !retained->request_owner.owner_before(expected_owner) && !expected_owner.owner_before(retained->request_owner));
            const auto peer_epoch=existing_peer->execution_epoch();
            bool peer_refused=false;
            try{existing_peer->acquire(*lease);}
            catch(const std::runtime_error& error){peer_refused=std::string(error.what())==
                "unresolved DFlash execution retirement";}
            gate("hostkv_acquire","existing_peer_refused_before_draft_lock",
                peer_refused && existing_peer->execution_epoch()==peer_epoch && !existing_peer->execution_active());
            bool proposal_refused=false;
            try{(void)existing_peer->propose(8);}
            catch(const std::runtime_error& error){proposal_refused=std::string(error.what())==
                "unresolved DFlash execution retirement";}
            gate("hostkv_acquire","proposal_gate_preserves_quarantine_reason",proposal_refused);
        } else {
            auto replacement=make_lane();
            replacement->acquire(*lease);
            gate("hostkv_acquire","successful_retirement_releases_physical_draft",
                replacement->execution_active() && replacement->lease().root==root);
            gate("hostkv_acquire","replacement_restores_full_request_state",
                replacement->context().export_exact_host_state()->same_payload(*root->state()));
            replacement->release();
        }
        coordinator.cancel(lease->ticket,true);
        return;
    }
    if(env("NINFER_TEST_HOST_KV_RESTORE_FAILURE")=="1") {
        const bool fail_retirement=env("NINFER_TEST_HOST_KV_RESTORE_RETIREMENT")=="1";
        const auto expected_storage=env("NINFER_TEST_HOST_KV_RESTORE_STORAGE");
        require(expected_storage.empty() || expected_storage=="registered" || expected_storage=="ordinary",
            "HostKV restore storage expectation must be registered or ordinary");
        require(code.size()>=17,"HostKV restore failure prompt extent");
        auto candidate=target.create_context(true,false,false);
        candidate->prefill(std::span<const std::int64_t>(code.data(),16));
        auto state=candidate->export_exact_host_state();
        const std::weak_ptr<const ninfer::exl3::Exl3ExactHostState> source_witness=state;
        const auto source_metadata=state->owner_metadata_bytes();
        const auto source_registered=state->recurrent_registered_bytes();
        if(!expected_storage.empty())gate("hostkv_restore","requested_storage_route_exercised",
            expected_storage=="registered"?source_registered>0:source_registered==0);
        std::uint64_t source_payload=0;
        ninfer::exl3::Exl3ExactHostState::visit_host_allocations(
            std::span<const std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>>(&state,1),
            [&](const void*,std::size_t bytes){source_payload+=bytes;},true);
        const auto expected_logits=candidate->logits_host();
        candidate->restore_exact_host_state(*state);
        gate("hostkv_restore","successful_restore_publishes_residency",
            candidate->exact_host_state_resident(*state) && candidate->logits_host()==expected_logits);
        {
            const auto restored=candidate->export_exact_host_state();
            gate("hostkv_restore","successful_restore_full_state",restored->same_payload(*state));
        }
        candidate->restore_exact_host_state(*state);
        gate("hostkv_restore","successful_restore_allows_repeat",
            candidate->exact_host_state_resident(*state));
        candidate->reset();
        gate("hostkv_restore","successful_reset_invalidates_residency",!candidate->exact_host_state_resident(*state));
        candidate->restore_exact_host_state(*state);
        gate("hostkv_restore","restore_after_healthy_reset",
            candidate->exact_host_state_resident(*state) && candidate->logits_host()==expected_logits);
        {
            const auto restored=candidate->export_exact_host_state();
            gate("hostkv_restore","reset_restore_full_state",restored->same_payload(*state));
        }
        {
            auto control=target.create_context(true,false,false);
            control->prefill(std::span<const std::int64_t>(code.data(),16));
            control->decode(code[16]);
            candidate->decode(code[16]);
            const auto expected=control->export_exact_host_state();
            const auto actual=candidate->export_exact_host_state();
            gate("hostkv_restore","restored_decode_matches_fresh_prefix_full_state",
                actual->same_payload(*expected));
            gate("hostkv_restore","restored_decode_invalidates_prior_residency",
                !candidate->exact_host_state_resident(*state));
        }
        candidate->restore_exact_host_state(*state);
        candidate->fail_next_host_kv_restore_for_test();
        bool injected=false;
        try{candidate->restore_exact_host_state(*state);}
        catch(const std::runtime_error& error){injected=std::string(error.what())==
            "injected HostKV failure after restore upload";}
        gate("hostkv_restore","post_mutation_failure_exercised",injected);
        gate("hostkv_restore","failed_restore_does_not_publish_residency",!candidate->exact_host_state_resident(*state));
        bool refused=false;
        try{candidate->restore_exact_host_state(*state);}
        catch(const std::runtime_error& error){refused=std::string(error.what())==
            "exact restore cannot reuse failed HostKV lineage";}
        gate("hostkv_restore","partial_restore_cannot_retry",refused);
        refused=false;
        try{candidate->reset();}
        catch(const std::runtime_error& error){refused=std::string(error.what())==
            "context reset cannot reuse failed HostKV lineage";}
        gate("hostkv_restore","partial_restore_cannot_reset",refused);
        state.reset();
        gate("hostkv_restore","failed_restore_retains_actual_source",!source_witness.expired());
        const auto quarantines=Exl3TextContext::host_kv_quarantined_contexts();
        if(fail_retirement)candidate->fail_host_kv_retirement_for_test();
        candidate.reset();
        gate("hostkv_restore","source_lifetime_matches_retirement",
            source_witness.expired()!=fail_retirement);
        gate("hostkv_restore","quarantine_matches_retirement",
            Exl3TextContext::host_kv_quarantined_contexts()==quarantines+(fail_retirement?1:0));
        if(fail_retirement) {
            const auto retained=Exl3TextContext::host_kv_retirement_snapshot();
            gate("hostkv_restore","retained_source_inventory",
                retained.restore_source_payload_capacity_bytes==source_payload &&
                retained.restore_source_object_bytes==sizeof(ninfer::exl3::Exl3ExactHostState) &&
                retained.restore_source_owner_metadata_bytes==source_metadata &&
                retained.restore_source_registered_bytes==source_registered);
        }
        return;
    }
    if(env("NINFER_TEST_HOST_KV_ACTIVE_RETIREMENT")=="1") {
        const bool batch=env("NINFER_TEST_HOST_KV_BATCH_RETIREMENT")=="1";
        const bool multiple_pages=env("NINFER_TEST_HOST_KV_MULTIPAGE_RETIREMENT")=="1";
        gate("hostkv_active","clean_initial_quarantine",Exl3TextContext::host_kv_quarantined_contexts()==0);
        require(code.size()>=16,"HostKV active retirement prompt extent");
        auto candidate=target.create_context(true,false,false);
        gate("hostkv_active","selected_transfer_storage_present",batch?
            candidate->host_kv_stats().pinned_staging_bytes==0 && candidate->host_kv_stats().batch_metadata_bytes>0:
            candidate->host_kv_stats().pinned_staging_bytes>0);
        const auto owner=candidate->host_kv_workspace_owner_for_test();
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> completed_prefix;
        if(multiple_pages) {
            require(code.size()>=65,"HostKV multipage retirement prompt extent");
            candidate->prefill(std::span<const std::int64_t>(code.data(),16));
            for(int row=16;row<64;++row)candidate->decode(code[row]);
            completed_prefix=candidate->export_exact_host_state();
        }
        candidate->fail_next_host_kv_copy_for_test();
        bool injected=false;
        try{
            if(multiple_pages)candidate->decode(code[64]);
            else candidate->prefill(std::span<const std::int64_t>(code.data(),16));
        }
        catch(const std::runtime_error& error){injected=std::string(error.what())==
            "injected HostKV failure after copy submission";}
        gate("hostkv_active","post_submission_failure_exercised",injected);
        gate("hostkv_active","copy_submission_recorded",candidate->host_kv_stats().copy_submissions>0);
        const auto pending_page=candidate->host_kv_pending_page_owner_for_test();
        gate("hostkv_active","actual_pending_page_present",!pending_page.expired());
        std::uint64_t expected_page_capacity=0;
        {
            const auto page=pending_page.lock();
            require(bool(page),"active fault page witness disappeared");
            gate("hostkv_active","pending_page_geometry",
                page->first==(multiple_pages?64:0) && page->rows==(multiple_pages?1:16));
            for(int bank=0;bank<16;++bank) {
                expected_page_capacity+=page->k[bank].capacity()*sizeof(std::uint16_t);
                expected_page_capacity+=page->v[bank].capacity()*sizeof(std::uint16_t);
            }
        } // Release the test's strong owner before exercising destruction.
        bool retry_refused=false;
        try{candidate->prefill(std::span<const std::int64_t>(code.data(),16));}
        catch(const std::runtime_error& error){retry_refused=std::string(error.what())==
            "HostKV prefill requires intact transfer lineage";}
        gate("hostkv_active","failed_lineage_retry_refused",retry_refused);
        bool append_refused=false;
        try{candidate->append_prefill(std::span<const std::int64_t>(code.data(),1));}
        catch(const std::runtime_error& error){append_refused=std::string(error.what())==
            "HostKV append requires intact transfer lineage";}
        gate("hostkv_active","failed_lineage_append_refused_before_capture_query",append_refused);
        bool continuation_refused=false;
        try{candidate->continue_rows(std::span<const std::int64_t>(code.data(),2));}
        catch(const std::runtime_error& error){continuation_refused=std::string(error.what())==
            "HostKV continuation requires intact transfer lineage";}
        gate("hostkv_active","failed_lineage_continuation_refused_before_upload",continuation_refused);
        bool reset_refused=false;
        try{candidate->reset();}
        catch(const std::runtime_error& error){reset_refused=std::string(error.what())==
            "context reset cannot reuse failed HostKV lineage";}
        gate("hostkv_active","failed_reset_preserves_pending_owners",reset_refused && !pending_page.expired());
        bool request_reset_refused=false;
        try{(void)candidate->reset_for_request(identity.contract());}
        catch(const std::runtime_error& error){request_reset_refused=std::string(error.what())==
            "request reset cannot reuse failed HostKV lineage";}
        gate("hostkv_active","failed_request_reset_refused_before_drain",request_reset_refused);
        if(completed_prefix) {
            bool restore_refused=false;
            try{candidate->restore_exact_host_state(*completed_prefix);}
            catch(const std::runtime_error& error){restore_refused=std::string(error.what())==
                "exact restore cannot reuse failed HostKV lineage";}
            gate("hostkv_active","failed_restore_preserves_pending_owners",restore_refused && !pending_page.expired());
            bool guarded_restore_refused=false;
            try{(void)candidate->restore_exact_host_state_if_needed(*completed_prefix);}
            catch(const std::runtime_error& error){guarded_restore_refused=std::string(error.what())==
                "exact restore cannot reuse failed HostKV lineage";}
            gate("hostkv_active","guarded_restore_preserves_failed_owners",
                guarded_restore_refused && !pending_page.expired());
            const auto failed_generation=candidate->request_generation();
            bool preserving_reset_refused=false;
            try{(void)candidate->reset_for_request_preserving(
                "failed-lineage-must-refuse-before-contract",*completed_prefix,failed_generation);}
            catch(const std::runtime_error& error){preserving_reset_refused=std::string(error.what())==
                "request reset cannot reuse failed HostKV lineage";}
            gate("hostkv_active","preserving_reset_cannot_recover_failed_lineage",
                preserving_reset_refused && candidate->request_generation()==failed_generation &&
                !pending_page.expired());
            completed_prefix.reset(); // Do not let the fixture retain prefix pages.
        }
        candidate.reset();
        const auto snapshot=Exl3TextContext::host_kv_retirement_snapshot();
        gate("hostkv_active","actual_pending_page_survives_context_destruction",!pending_page.expired());
        gate("hostkv_active","failed_final_use_witness_survives_quarantine",
            batch?snapshot.batch_descriptors_pending:snapshot.failed_final_use_slots==1);
        if(batch) {
            gate("hostkv_active","retained_batch_command_counts",
                snapshot.batch_table_sizes[0]>0 && snapshot.batch_table_sizes[0]%2==0 &&
                snapshot.batch_table_sizes[0]==snapshot.batch_table_sizes[1] &&
                snapshot.batch_table_sizes[1]==snapshot.batch_table_sizes[2]);
            for(unsigned table=0;table<3;++table)
                gate("hostkv_active","retained_batch_table_backing",
                    snapshot.batch_table_addresses[table]!=0 &&
                    snapshot.batch_table_sizes[table]<=snapshot.batch_table_capacities[table]);
            const auto again=Exl3TextContext::host_kv_retirement_snapshot();
            gate("hostkv_active","quarantine_observation_preserves_batch_storage",
                snapshot.batch_table_addresses==again.batch_table_addresses &&
                snapshot.batch_table_sizes==again.batch_table_sizes &&
                snapshot.batch_table_capacities==again.batch_table_capacities && again.batch_descriptors_pending);
        }
        // Count each physical page once, including the completed prefix in the
        // later-page fault variant and repeated pending plane references.
        gate("hostkv_active","pending_page_metadata_deduplicated",
            snapshot.page_owner_metadata_bytes==(multiple_pages?2:1)*sizeof(ninfer::exl3::Exl3ExactKVPage));
        gate("hostkv_active","retained_page_capacity_matches_actual_storage",
            (multiple_pages?snapshot.page_payload_capacity_bytes>=expected_page_capacity+16ULL*2*64*1024*sizeof(std::uint16_t):
                snapshot.page_payload_capacity_bytes==expected_page_capacity) &&
            expected_page_capacity>=16ULL*2*64*1024*sizeof(std::uint16_t));
        gate("hostkv_active","pending_transfer_and_scatter_retained",
            snapshot.retained && (batch?(snapshot.batch_descriptors_pending && snapshot.transfer_descriptor_metadata_bytes>0):
                (snapshot.pending_slots>0 && snapshot.scatter_pieces>0)) &&
            snapshot.drain_error==static_cast<int>(cudaErrorUnknown) && !owner.expired());
        return; // Isolated process: queued-transfer owners remain quarantined.
    }
    if(env("NINFER_TEST_CONTINUATION_CLEANUP_RETIREMENT")=="1") {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        auto context=target.create_context(true,false,true);
        const auto before_bytes=context->persistent_bytes();
        const auto required=context->continuation_bytes_required(8);
        Coordinator authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        authority.bind_physical_resources({},Inventory::unlimited());
        auto expected=Exl3TextContext::retirement_quarantine_witness();++expected[4];
        bool mismatch=false;
        try{context->prepare_continuation_reserved(authority,8,5);}
        catch(const std::invalid_argument& error){mismatch=std::string(error.what())==
            "resource reservation actual extent/domain mismatch";}
        const auto retained=Exl3TextContext::generic_retirement_snapshot_for_test();
        gate("continuation_cleanup","unattached_workspace_retained",
            mismatch && context->persistent_bytes()==before_bytes &&
            context->continuation_bytes_required(8)==required &&
            Exl3TextContext::retirement_quarantine_witness()==expected &&
            retained.pointer!=0 && retained.bytes==8ULL*5120*2 &&
            retained.error==static_cast<int>(cudaErrorUnknown));
        bool sealed=false;
        try{context->prepare_continuation_reserved(authority,8);}
        catch(const std::logic_error& error){sealed=std::string(error.what())==
            "serving coordinator sealed after failed startup retirement";}
        gate("continuation_cleanup","failed_cleanup_seals_authority",sealed);
        const auto before=context->host_kv_stats();
        for(bool reset:{false,true}) {
            bool refused=false;
            try{
                if(reset)context->reset();
                else context->prefill(std::span<const std::int64_t>(code.data(),std::min<std::size_t>(16,code.size())));
            }catch(const std::runtime_error& error){refused=std::string(error.what())==
                "unresolved generic cleanup blocks context execution";}
            gate("continuation_cleanup",reset?"survivor_reset_refused":"survivor_prefill_refused",refused);
        }
        const auto after=context->host_kv_stats();
        gate("continuation_cleanup","survivor_refusal_precedes_mutation",
            context->position()==0 && context->persistent_bytes()==before_bytes &&
            before.h2d_bytes==after.h2d_bytes && before.completed_rows==after.completed_rows);
        context.reset();authority.close();
        return; // Isolated process: continuation normalized-row allocation retained.
    }
    if(env("NINFER_TEST_GENERIC_UPLOAD_RETIREMENT")=="1") {
        const auto name=env("NINFER_TEST_GENERIC_RETIREMENT_FAULT");
        require(name.empty() || name=="free" || name=="query" || name=="device",
            "generic upload retirement fault must be free/query/device");
        const unsigned failure=name=="query"?3u:(name=="device"?4u:2u);
        const std::array<std::byte,32> payload{};
        auto expected=Exl3TextContext::retirement_quarantine_witness();
        bool injected=false;
        try{Exl3TextContext::exercise_generic_upload_for_test(payload,1);}
        catch(const std::runtime_error& error){injected=std::string(error.what())==
            "injected generic post-upload construction failure";}
        gate("generic_upload","healthy_cleanup_preserves_original_exception",
            injected && Exl3TextContext::retirement_quarantine_witness()==expected);
        Exl3TextContext::exercise_generic_upload_for_test(payload);
        injected=false;
        try{Exl3TextContext::exercise_generic_upload_for_test(payload,failure);}
        catch(const std::runtime_error& error){injected=std::string(error.what())==
            "injected generic post-upload construction failure";}
        ++expected[4];
        const auto retained=Exl3TextContext::generic_retirement_snapshot_for_test();
        gate("generic_upload","failed_constructor_retains_exact_payload",
            injected && Exl3TextContext::retirement_quarantine_witness()==expected &&
            retained.pointer!=0 && retained.bytes==payload.size() && retained.device>=0 &&
            retained.error==static_cast<int>(failure==3?cudaErrorInitializationError:
                (failure==4?cudaErrorInvalidDevice:cudaErrorUnknown)));
        bool refused=false;
        try{Exl3TextContext::exercise_generic_upload_for_test(payload);}
        catch(const std::runtime_error& error){refused=std::string(error.what())=="unresolved generic allocation cleanup";}
        gate("generic_upload","later_upload_refused",refused);
        return; // Isolated process: failed upload cleanup retains its allocation.
    }
    if(env("NINFER_TEST_GENERIC_WEAK_CONTROL_LIFETIME")=="1") {
        for(bool reserved:{false,true}) {
        Coordinator authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        authority.bind_physical_resources({},ninfer::exl3::Exl3ResourceInventory::unlimited());
        const auto baseline=Exl3TextContext::live_shared_control_metadata_bytes();
        std::shared_ptr<Exl3TextContext> context=reserved?target.create_context_reserved(authority):
            std::shared_ptr<Exl3TextContext>(target.create_context(true,false,true));
        auto weak=context->host_kv_workspace_owner_for_test();
        gate("weak_control","workspace_initially_alive",!weak.expired());
        const auto constructed=Exl3TextContext::live_shared_control_metadata_bytes();
        gate("weak_control","two_fixed_kv_control_blocks",constructed==baseline+2*512);
        context.reset();
        if(reserved)gate("weak_control","authority_retains_context_and_both_controls",
            !weak.expired() && Exl3TextContext::live_shared_control_metadata_bytes()==constructed);
        authority.close();
        gate("weak_control","expired_owner_keeps_one_control_block",
            weak.expired() && Exl3TextContext::live_shared_control_metadata_bytes()==baseline+512);
        weak.reset();
        gate("weak_control","last_weak_release_returns_control_bytes",
            Exl3TextContext::live_shared_control_metadata_bytes()==baseline);
        }
        return;
    }
    if(env("NINFER_TEST_GENERIC_CONTROL_RETIREMENT")=="1") {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        std::uint64_t plane_bytes=0;
        for(unsigned fault:{14u,15u}) {
            const auto control_bytes=Exl3TextContext::live_shared_control_metadata_bytes();
            Coordinator authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            authority.bind_physical_resources({},Inventory::unlimited());
            auto expected=Exl3TextContext::retirement_quarantine_witness();
            if(fault==15)++expected[4];
            bool failed=false;
            try{(void)target.create_context_reserved(authority,true,fault);}
            catch(const std::bad_alloc&){failed=true;}
            gate("generic_control","control_allocation_failure_exercised",
                failed && Exl3TextContext::retirement_quarantine_witness()==expected);
            gate("generic_control","failed_control_allocation_charges_no_block",
                Exl3TextContext::live_shared_control_metadata_bytes()==control_bytes);
            if(fault==14) {
                auto retry=target.create_context_reserved(authority);
                plane_bytes=retry->host_kv_stats().layer_workspace_bytes/2;
                gate("generic_control","healthy_cleanup_same_authority_retry",plane_bytes>0);
                gate("generic_control","retry_charges_two_control_blocks",
                    Exl3TextContext::live_shared_control_metadata_bytes()==control_bytes+1024);
                retry.reset();authority.close();
                gate("generic_control","retry_close_releases_control_blocks",
                    Exl3TextContext::live_shared_control_metadata_bytes()==control_bytes);
            } else {
                const auto retained=Exl3TextContext::generic_retirement_snapshot_for_test();
                gate("generic_control","unpublished_workspace_retained",
                    retained.pointer!=0 && retained.bytes==plane_bytes && retained.error==static_cast<int>(cudaErrorUnknown));
                bool sealed=false;
                try{(void)target.create_context_reserved(authority);}
                catch(const std::logic_error& error){sealed=std::string(error.what())==
                    "serving coordinator sealed after failed startup retirement";}
                gate("generic_control","partial_construction_seals_authority",sealed);
                authority.close();
            }
        }
        return; // Isolated process: final control-allocation arm retains workspace.
    }
    if(env("NINFER_TEST_GENERIC_STARTUP_RETIREMENT")=="1") {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        const auto fault_name=env("NINFER_TEST_GENERIC_RETIREMENT_FAULT");
        require(fault_name.empty() || fault_name=="free" || fault_name=="query" || fault_name=="device",
            "generic retirement fault must be free/query/device");
        const unsigned fault=fault_name=="query"?12u:(fault_name=="device"?13u:11u);
        std::uint64_t expected_plane_bytes=0;
        const auto initial=Exl3TextContext::retirement_quarantine_witness();
        {
            auto control=target.create_context(true,false,true);
            const auto workspace=control->host_kv_stats().layer_workspace_bytes;
            gate("generic_retirement","control_has_two_equal_hostkv_planes",workspace>0 && workspace%2==0);
            expected_plane_bytes=workspace/2;
        }
        gate("generic_retirement","healthy_control_retirement",Exl3TextContext::retirement_quarantine_witness()==initial);
        Coordinator authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        authority.bind_physical_resources({},Inventory::unlimited());
        auto expected=Exl3TextContext::retirement_quarantine_witness();++expected[4];
        bool mismatch=false;
        try{(void)target.create_context_reserved(authority,true,fault);}
        catch(const std::invalid_argument& error){mismatch=std::string(error.what())==
            "resource reservation actual extent/domain mismatch";}
        gate("generic_retirement","only_generic_component_retained",
            mismatch && Exl3TextContext::retirement_quarantine_witness()==expected);
        const auto retained=Exl3TextContext::generic_retirement_snapshot_for_test();
        gate("generic_retirement","retained_real_allocation_record",
            retained.pointer!=0 && retained.bytes==expected_plane_bytes && retained.device>=0 &&
            retained.error==static_cast<int>(fault==12?cudaErrorInitializationError:
                (fault==13?cudaErrorInvalidDevice:cudaErrorUnknown)) && retained.metadata_bytes>0);
        bool sealed=false;
        try{(void)target.create_context_reserved(authority);}
        catch(const std::logic_error& error){sealed=std::string(error.what())==
            "serving coordinator sealed after failed startup retirement";}
        gate("generic_retirement","startup_authority_sealed",sealed);
        bool refused=false;
        try{(void)target.create_context(true,false,true);}
        catch(const std::runtime_error& error){refused=std::string(error.what())==
            "unresolved generic allocation cleanup; context creation refused";}
        gate("generic_retirement","unreserved_context_refused_before_planning",
            refused && Exl3TextContext::retirement_quarantine_witness()==expected);
        authority.close();authority.close();
        return; // Isolated process: generic allocation remains retained.
    }
    if(env("NINFER_TEST_HOST_KV_STARTUP_RETIREMENT")=="1") {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        const bool mismatch=env("NINFER_TEST_HOST_KV_MISMATCH_RETIREMENT")=="1";
        const bool wrapped=mismatch || env("NINFER_TEST_HOST_KV_WRAPPED_STARTUP_RETIREMENT")=="1";
        gate("hostkv_unwind","clean_initial_quarantine",Exl3TextContext::host_kv_quarantined_contexts()==0);
        std::size_t expected_partial_device=0,expected_partial_pinned=0;
        std::uint64_t expected_partial_owner_metadata=0;
        for(unsigned fault:{wrapped?8u:6u,mismatch?10u:(wrapped?9u:7u)}) {
            Coordinator authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            authority.bind_physical_resources({},Inventory::unlimited());
            bool original_error=false;
            const auto quarantine_before=Exl3TextContext::retirement_quarantine_witness();
            try{(void)target.create_context_reserved(authority,true,fault);}
            catch(const std::exception& error){original_error=std::string(error.what())==
                (fault==10?"resource reservation actual extent/domain mismatch":
                    (wrapped?"injected context post-reset construction failure":
                        "injected context post-transfer construction failure"));}
            gate("hostkv_unwind","original_construction_error_preserved",original_error);
            if(fault==6 || fault==8) {
                gate("hostkv_unwind","successful_unwind_not_quarantined",Exl3TextContext::host_kv_quarantined_contexts()==0);
                gate("hostkv_unwind","healthy_unwind_preserves_all_quarantine_components",
                    Exl3TextContext::retirement_quarantine_witness()==quarantine_before);
                auto retry=target.create_context_reserved(authority);
                const auto workspace=retry->host_kv_workspace_owner_for_test();
                const auto later_packet=!wrapped && ninfer::exl3::exl3_device_greedy_enabled()?
                    Exl3TextContext::greedy_packet_bytes_required():0;
                gate("hostkv_unwind","control_device_extent",retry->persistent_bytes()>later_packet);
                expected_partial_device=retry->persistent_bytes()-later_packet;
                expected_partial_pinned=retry->host_kv_stats().pinned_staging_bytes;
                expected_partial_owner_metadata=retry->allocation_owner_metadata_bytes()-
                    (later_packet?(Exl3TextContext::allocation_record_metadata_bytes()+
                        Exl3TextContext::greedy_transfer_pool_metadata_bytes_required()):0);
                retry.reset();authority.close();
                gate("hostkv_unwind","same_authority_retry_retires_owner",workspace.expired());
            } else {
                gate("hostkv_unwind","failed_unwind_quarantined",Exl3TextContext::host_kv_quarantined_contexts()==1);
                auto expected_quarantine=quarantine_before;++expected_quarantine[0];
                gate("hostkv_unwind","failed_unwind_changes_only_hostkv_component",
                    Exl3TextContext::retirement_quarantine_witness()==expected_quarantine);
                const auto retained=Exl3TextContext::quarantined_host_kv_workspace_for_test();
                gate("hostkv_unwind","partial_context_workspace_retained",!retained.expired());
                const auto snapshot=Exl3TextContext::host_kv_retirement_snapshot();
                gate("hostkv_unwind","partial_context_error_and_extents",
                    snapshot.retained && snapshot.drain_error==static_cast<int>(cudaErrorUnknown) &&
                    snapshot.device_bytes==expected_partial_device && snapshot.pinned_bytes==expected_partial_pinned);
                gate("hostkv_unwind","retained_inline_metadata",
                    snapshot.inline_metadata_bytes+sizeof(Exl3TextContext)==Exl3TextContext::fixed_owner_metadata_bytes());
                gate("hostkv_unwind","retained_partial_owner_metadata",
                    snapshot.allocation_owner_metadata_bytes==expected_partial_owner_metadata);
                gate("hostkv_unwind","idle_failure_does_not_invent_pending_work",
                    snapshot.pending_slots==0 && snapshot.scatter_pieces==0);
                gate("hostkv_unwind","construction_stage_forward_witness",
                    snapshot.forward_stream_retained==wrapped);
                bool refused=false;
                try{(void)target.create_context_reserved(authority);}
                catch(const std::logic_error& error){refused=std::string(error.what())==
                    "serving coordinator sealed after failed startup retirement";}
                gate("hostkv_unwind","failed_unwind_retry_refused",refused);
                Inventory::Requirement metadata;metadata.configuration=94;
                metadata.add(Inventory::Domain::host_metadata,1,sizeof(int));
                bool called=false;refused=false;
                try{authority.allocate_startup_resources(metadata,[&](auto) {
                    called=true;return Inventory{};
                });}catch(const std::logic_error& error){refused=std::string(error.what())==
                    "serving coordinator sealed after failed startup retirement";}
                gate("hostkv_unwind","sealed_fixed_factory_not_called",refused && !called);
                refused=false;
                try{authority.allocate_startup_resources_growing(metadata,[&](auto,auto&&) {
                    called=true;return Inventory{};
                });}catch(const std::logic_error& error){refused=std::string(error.what())==
                    "serving coordinator sealed after failed startup retirement";}
                gate("hostkv_unwind","sealed_growing_factory_not_called",refused && !called);
                authority.close();
                authority.close();
                gate("hostkv_unwind","idle_close_keeps_quarantined_owner",!retained.expired());
            }
        }
        return; // Isolated process: stage7/9/10 intentionally retains its Impl.
    }
    if(env("NINFER_TEST_HOST_KV_CONTEXT_RETIREMENT")=="1") {
        const bool device_mismatch=env("NINFER_TEST_HOST_KV_DEVICE_MISMATCH")=="1";
        const bool query_failure=env("NINFER_TEST_HOST_KV_DEVICE_QUERY_FAILURE")=="1";
        require(!device_mismatch || !query_failure,"HostKV retirement fault variants are exclusive");
        // Isolated process: failed-drain quarantine intentionally survives exit.
        const auto before=Exl3TextContext::host_kv_quarantined_contexts();
        gate("hostkv_retirement","clean_initial_quarantine",before==0);
        {
            auto control=target.create_context(true,false,true);
            const auto workspace=control->host_kv_workspace_owner_for_test();
            gate("hostkv_retirement","control_workspace_present",!workspace.expired());
            control.reset();
            gate("hostkv_retirement","successful_drain_releases_workspace",workspace.expired());
            gate("hostkv_retirement","successful_drain_does_not_quarantine",
                Exl3TextContext::host_kv_quarantined_contexts()==before);
        }
        auto candidate=target.create_context(true,false,true);
        gate("hostkv_retirement","pinned_storage_exercised",candidate->host_kv_stats().pinned_staging_bytes>0);
        const auto retained_workspace=candidate->host_kv_workspace_owner_for_test();
        const auto retained_device_bytes=candidate->persistent_bytes();
        const auto retained_pinned_bytes=candidate->host_kv_stats().pinned_staging_bytes;
        const auto retained_descriptor_bytes=candidate->host_kv_stats().batch_metadata_bytes;
        const auto retained_owner_metadata=candidate->allocation_owner_metadata_bytes();
        gate("hostkv_retirement","workspace_owner_present",!retained_workspace.expired());
        if(query_failure)candidate->fail_host_kv_device_query_for_test();
        else candidate->fail_host_kv_retirement_for_test(device_mismatch);
        bool repeated=false;
        try{candidate->fail_host_kv_retirement_for_test();}
        catch(const std::runtime_error& error){repeated=std::string(error.what())==
            "HostKV retirement injection requires unarmed transfer stream";}
        gate("hostkv_retirement","duplicate_arm_refused",repeated);
        candidate.reset();
        gate("hostkv_retirement","failed_drain_quarantines_context",
            Exl3TextContext::host_kv_quarantined_contexts()==before+1);
        gate("hostkv_retirement","failed_drain_retains_actual_workspace",!retained_workspace.expired());
        const auto retained_snapshot=Exl3TextContext::host_kv_retirement_snapshot();
        gate("hostkv_retirement","retained_error_and_extents",
            retained_snapshot.retained && retained_snapshot.drain_error==static_cast<int>(
                query_failure?cudaErrorInitializationError:(device_mismatch?cudaErrorInvalidDevice:cudaErrorUnknown)) &&
            retained_snapshot.device_bytes==retained_device_bytes && retained_snapshot.pinned_bytes==retained_pinned_bytes);
        gate("hostkv_retirement","retained_inline_metadata",
            retained_snapshot.inline_metadata_bytes+sizeof(Exl3TextContext)==Exl3TextContext::fixed_owner_metadata_bytes());
        gate("hostkv_retirement","retained_descriptor_metadata",
            retained_snapshot.transfer_descriptor_metadata_bytes==retained_descriptor_bytes);
        gate("hostkv_retirement","retained_allocation_owner_metadata",
            retained_snapshot.allocation_owner_metadata_bytes==retained_owner_metadata);
        gate("hostkv_retirement","idle_failure_does_not_invent_pending_work",
            retained_snapshot.pending_slots==0 && retained_snapshot.scatter_pieces==0);
        bool refused=false;
        try{auto replacement=target.create_context(true,false,true);}
        catch(const std::runtime_error& error){refused=std::string(error.what())==
            "unresolved HostKV context retirement; context creation refused";}
        gate("hostkv_retirement","replacement_context_refused",refused);
        return;
    }
    if(env("NINFER_TEST_LEGACY_PREFIX_INVENTORY")=="1") {
        using Prefix=ninfer::exl3::Exl3DevicePrefixCache;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        require(env("NINFER_EXL3_HOST_KV_DEVICE_PREFIX")=="1" &&
            (env("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS").empty() ||
             env("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS")=="4096"),
            "legacy prefix inventory fixture requires4096-row enabled prefix");
        const auto budget=Prefix::budget_snapshot();
        auto first=target.create_context(true,true,true);
        auto peer=target.create_context(true,true,true);
        gate("legacy_prefix_inventory","both_private_caches_admitted",
            first->host_kv_stats().device_prefix_bytes==Prefix::bytes &&
            peer->host_kv_stats().device_prefix_bytes==Prefix::bytes);
        const auto first_bytes=first->persistent_bytes(),peer_bytes=peer->persistent_bytes();
        const auto first_metadata=first->allocation_owner_metadata_bytes();
        const auto peer_metadata=peer->allocation_owner_metadata_bytes();
        const auto metadata=Prefix::owner_metadata_bytes_required(4096);
        const auto first_identity=first->device_prefix_owner_for_test();
        const auto peer_identity=peer->device_prefix_owner_for_test();
        for(unsigned slot:{0u,1u}) {
            Inventory conflict;
            conflict.add({first_identity.lock(),slot,slot==0?Inventory::Domain::device:
                Inventory::Domain::host_metadata,(slot==0?Prefix::bytes:metadata)-1});
            const auto before=conflict.totals();
            bool refused=false;
            try{(void)first->share_device_prefix_with(*peer,conflict);}
            catch(const std::invalid_argument& error){refused=std::string(error.what())==
                "resource owner slot changed extent/domain";}
            const auto current_first=first->device_prefix_owner_for_test();
            const auto current_peer=peer->device_prefix_owner_for_test();
            gate("legacy_prefix_inventory","conflict_preserves_inventory_and_contexts",
                refused && conflict.totals()==before &&
                first->persistent_bytes()==first_bytes && peer->persistent_bytes()==peer_bytes &&
                first->allocation_owner_metadata_bytes()==first_metadata &&
                peer->allocation_owner_metadata_bytes()==peer_metadata &&
                !current_first.owner_before(first_identity) && !first_identity.owner_before(current_first) &&
                !current_peer.owner_before(peer_identity) && !peer_identity.owner_before(current_peer));
        }
        Inventory shared;
        {
            const auto borrowed=peer_identity.lock();
            const auto before=Prefix::budget_snapshot();
            bool refused=false;
            try{(void)first->share_device_prefix_with(*peer,shared);}
            catch(const std::runtime_error& error){refused=std::string(error.what())==
                "legacy prefix replacement requires sole destination cache ownership";}
            gate("legacy_prefix_inventory","retained_destination_refuses_replacement",
                borrowed && refused && shared.totals()==Inventory::Totals{} &&
                Prefix::budget_snapshot()==before &&
                first->persistent_bytes()==first_bytes && peer->persistent_bytes()==peer_bytes &&
                first->allocation_owner_metadata_bytes()==first_metadata &&
                peer->allocation_owner_metadata_bytes()==peer_metadata);
        }
        if(env("NINFER_TEST_LEGACY_PREFIX_CLEANUP_FAILURE")=="1") {
            const auto before=Prefix::budget_snapshot();
            peer->fail_device_prefix_cleanup_for_test();
            bool refused=false;
            try{(void)first->share_device_prefix_with(*peer,shared);}
            catch(const std::runtime_error& error){refused=std::string(error.what())==
                "legacy prefix destination retirement failed";}
            gate("legacy_prefix_inventory","cleanup_failure_preserves_transfer_state",
                refused && shared.totals()==Inventory::Totals{} && Prefix::budget_snapshot()==before &&
                first->persistent_bytes()==first_bytes && peer->persistent_bytes()==peer_bytes &&
                first->allocation_owner_metadata_bytes()==first_metadata &&
                peer->allocation_owner_metadata_bytes()==peer_metadata && !peer_identity.expired());
            refused=false;
            try{(void)first->share_device_prefix_with(*peer,shared);}
            catch(const std::runtime_error& error){refused=std::string(error.what())==
                "legacy prefix sharing requires reusable caches";}
            gate("legacy_prefix_inventory","failed_cleanup_retry_refused",refused);
            peer.reset();first.reset();
            const auto retained=Prefix::retirement_snapshot_for_test();
            const auto after=Prefix::budget_snapshot();
            gate("legacy_prefix_inventory","failed_destination_retained_after_teardown",
                peer_identity.expired() && retained.pointer!=0 && retained.bytes==Prefix::bytes &&
                retained.error==static_cast<int>(cudaErrorUnknown) &&
                after[0]==budget[0]+Prefix::bytes && after[1]==budget[1]+Prefix::bytes);
            return; // Isolated process: failed destination allocation remains retained.
        }
        auto owner=first->share_device_prefix_with(*peer,shared);
        gate("legacy_prefix_inventory","replaced_destination_owner_released",peer_identity.expired());
        std::weak_ptr<const void> witness=owner;
        gate("legacy_prefix_inventory","device_and_metadata_transferred",
            first->persistent_bytes()==first_bytes-Prefix::bytes &&
            peer->persistent_bytes()==peer_bytes-Prefix::bytes &&
            first->allocation_owner_metadata_bytes()==first_metadata-metadata &&
            peer->allocation_owner_metadata_bytes()==peer_metadata-metadata &&
            shared.totals()[static_cast<unsigned>(Inventory::Domain::device)]==Prefix::bytes &&
            shared.totals()[static_cast<unsigned>(Inventory::Domain::host_metadata)]==metadata);
        {
            const auto duplicate=shared;
            shared.append(duplicate);
            gate("legacy_prefix_inventory","shared_owner_deduplicated",shared.totals()==duplicate.totals());
        }
        owner.reset();first.reset();
        gate("legacy_prefix_inventory","peer_and_inventory_survive_origin",!witness.expired());
        shared={};
        gate("legacy_prefix_inventory","peer_survives_inventory",!witness.expired());
        peer.reset();
        gate("legacy_prefix_inventory","last_owner_releases_cache",
            witness.expired() && Prefix::budget_snapshot()==budget);
        return;
    }
    if(env("NINFER_TEST_PRIVATE_PREFIX_QUARANTINE_FALLBACK")=="1") {
        using Prefix=ninfer::exl3::Exl3DevicePrefixCache;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        const bool shared=env("NINFER_TEST_PREFIX_FALLBACK_SHARED")=="1";
        const bool segmented=env("NINFER_TEST_PREFIX_FALLBACK_SEGMENTED")=="1";
        require(code.size()>=67,"private prefix fallback requires67 prompt tokens");
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> expected;
        {
            auto reference=target.create_context(true,false,true);
            reference->prefill(std::span<const std::int64_t>(code.data(),16));
            for(int i=16;i<67;++i)reference->decode(code[i]);
            expected=reference->export_exact_host_state();
        }
        auto context=target.create_context(true,false,true);
        Coordinator owner(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        owner.bind_physical_resources({},Inventory::unlimited());
        gate("private_prefix_fallback","prefix_admitted",
            context->prepare_device_prefix_reserved(owner,4096,segmented));
        std::unique_ptr<Exl3TextContext> peer;
        if(shared) {
            peer=target.create_context(true,false,true);
            context->share_device_prefix_with_empty(*peer);
        }
        context->prefill(std::span<const std::int64_t>(code.data(),16));
        for(int i=16;i<66;++i)context->decode(code[i]);
        const auto before=context->host_kv_stats();
        gate("private_prefix_fallback","prefix_fill_exercised",before.device_prefix_fill_bytes>0);
        gate("private_prefix_fallback","prefix_hit_exercised",before.device_prefix_hit_bytes>0);
        if(segmented)gate("private_prefix_fallback","segmented_route_exercised",before.device_prefix_segmented_bytes>0);
        auto peer_before=before;
        if(peer) {
            peer->prefill(std::span<const std::int64_t>(code.data(),16));
            for(int i=16;i<66;++i)peer->decode(code[i]);
            peer_before=peer->host_kv_stats();
        }
        Coordinator failing(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        failing.bind_physical_resources({},Inventory::unlimited());
        const auto budget=Prefix::budget_snapshot();
        bool mismatch=false;
        try{(void)Prefix::create_reserved(failing,4096,3);}
        catch(const std::invalid_argument& error){mismatch=std::string(error.what())==
            "resource reservation actual extent/domain mismatch";}
        gate("private_prefix_fallback","separate_cleanup_failure_exercised",
            mismatch && Prefix::budget_snapshot()[1]==budget[1]+Prefix::bytes);
        context->decode(code[66]);
        const auto after=context->host_kv_stats();
        gate("private_prefix_fallback","no_quarantined_prefix_copy",
            after.device_prefix_fill_bytes==before.device_prefix_fill_bytes &&
            after.device_prefix_hit_bytes==before.device_prefix_hit_bytes &&
            after.device_prefix_segmented_bytes==before.device_prefix_segmented_bytes);
        gate("private_prefix_fallback","authoritative_upload_continues",after.h2d_bytes>before.h2d_bytes);
        gate("private_prefix_fallback","full_state_matches_cache_free_reference",
            context->position()==67 && context->export_exact_host_state()->same_payload(*expected));
        if(peer) {
            peer->decode(code[66]);
            const auto peer_after=peer->host_kv_stats();
            gate("private_prefix_fallback","both_shared_consumers_fallback",
                after.shared_prefix_busy_fallbacks==before.shared_prefix_busy_fallbacks+1 &&
                peer_after.shared_prefix_busy_fallbacks==peer_before.shared_prefix_busy_fallbacks+1 &&
                peer_after.device_prefix_hit_bytes==peer_before.device_prefix_hit_bytes &&
                peer_after.device_prefix_fill_bytes==peer_before.device_prefix_fill_bytes &&
                peer_after.device_prefix_segmented_bytes==peer_before.device_prefix_segmented_bytes &&
                peer_after.h2d_bytes>peer_before.h2d_bytes);
            gate("private_prefix_fallback","peer_full_state_matches_reference",
                peer->position()==67 && peer->export_exact_host_state()->same_payload(*expected));
        }
        peer.reset();context.reset();owner.close();failing.close();
        return; // Isolated process: separate failed prefix allocation is retained.
    }
    if(env("NINFER_TEST_PREFIX_STARTUP_ROLLBACK")=="1") {
        using Prefix=ninfer::exl3::Exl3DevicePrefixCache;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        const auto retirement=env("NINFER_TEST_PREFIX_RETIREMENT_FAULT");
        require(retirement.empty() || retirement=="free" || retirement=="query" || retirement=="device",
            "prefix retirement fault must be free/query/device");
        const unsigned final_fault=retirement=="query"?4u:(retirement=="device"?5u:3u);
        for(unsigned fault:{1u,2u,final_fault}) {
            Coordinator authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            authority.bind_physical_resources({},Inventory::unlimited());
            std::shared_ptr<Prefix> survivor;
            std::vector<std::shared_ptr<const ninfer::exl3::Exl3ExactKVPage>> tagged_pages;
            std::weak_ptr<const ninfer::exl3::Exl3ExactKVPage> tagged_owner;
            if(fault>=3) {
                survivor=std::make_shared<Prefix>(4096);
                gate("prefix_rollback","survivor_admitted",survivor->admitted());
                gate("prefix_rollback","survivor_initially_reusable",survivor->reusable());
                auto use=survivor->try_use();
                gate("prefix_rollback","survivor_initial_use",static_cast<bool>(use));
                // Metadata-only tag fixture: no cache contents are consumed.
                auto page=std::make_shared<ninfer::exl3::Exl3ExactKVPage>();
                page->first=0;page->rows=64;tagged_owner=page;
                tagged_pages.push_back(std::move(page));
                survivor->publish_tags(tagged_pages,64);
                gate("prefix_rollback","survivor_tag_initially_visible",
                    survivor->matched_rows(tagged_pages,64)==64);
                use.complete();
            }
            const auto before=Prefix::budget_snapshot();
            bool expected=false;
            auto expected_witness=Exl3TextContext::retirement_quarantine_witness();
            if(fault>=3)expected_witness[3]+=Prefix::bytes;
            try{(void)Prefix::create_reserved(authority,4096,fault);}
            catch(const std::exception& error){expected=std::string(error.what())==
                (fault==1?"injected prefix startup precommit failure":
                    "resource reservation actual extent/domain mismatch");}
            gate("prefix_rollback","fault_exercised",expected);
            gate("prefix_rollback","only_prefix_quarantine_component_changes",
                Exl3TextContext::retirement_quarantine_witness()==expected_witness);
            const auto after=Prefix::budget_snapshot();
            if(fault<3) {
                gate("prefix_rollback","healthy_cleanup_restores_budget",before==after);
                auto retry=Prefix::create_reserved(authority,4096);
                gate("prefix_rollback","same_authority_retry",retry && retry->admitted());
                retry.reset();authority.close();
                gate("prefix_rollback","close_restores_budget",Prefix::budget_snapshot()==before);
            } else {
                gate("prefix_rollback","failed_cleanup_keeps_budget",
                    after[0]==before[0]+Prefix::bytes && after[1]==before[1]+Prefix::bytes);
                const auto retained=Prefix::retirement_snapshot_for_test();
                gate("prefix_rollback","failed_cleanup_retains_allocation_record",
                    retained.pointer!=0 && retained.bytes==Prefix::bytes && retained.device>=0 &&
                    retained.error==static_cast<int>(fault==4?cudaErrorInitializationError:
                        (fault==5?cudaErrorInvalidDevice:cudaErrorUnknown)));
                bool sealed=false;
                try{(void)Prefix::create_reserved(authority,4096);}
                catch(const std::logic_error& error){sealed=std::string(error.what())==
                    "serving coordinator sealed after failed startup retirement";}
                gate("prefix_rollback","failed_cleanup_seals_authority",sealed);
                bool globally_refused=false;
                try{auto retry=std::make_shared<Prefix>(4096);}
                catch(const std::runtime_error& error){globally_refused=std::string(error.what())==
                    "unresolved device prefix cleanup";}
                gate("prefix_rollback","failed_cleanup_blocks_unreserved_allocation",globally_refused);
                gate("prefix_rollback","survivor_private_cache_fallback",!survivor->reusable() && survivor->admitted());
                gate("prefix_rollback","quarantine_hides_preexisting_tag",
                    survivor->matched_rows(tagged_pages,64)==0);
                survivor->publish_tags(tagged_pages,64);
                gate("prefix_rollback","quarantine_refuses_tag_republication",
                    survivor->matched_rows(tagged_pages,64)==0);
                tagged_pages.clear();
                gate("prefix_rollback","tag_refusal_does_not_retain_host_page",tagged_owner.expired());
                {
                    auto use=survivor->try_use();
                    gate("prefix_rollback","survivor_use_refused",!static_cast<bool>(use));
                }
                bool plane_refused=false;
                try{(void)survivor->plane(0,true);}
                catch(const std::logic_error& error){plane_refused=std::string(error.what())==
                    "device prefix plane blocked by unresolved cleanup";}
                gate("prefix_rollback","survivor_plane_refused",plane_refused);
                survivor.reset();
                const auto released=Prefix::budget_snapshot();
                gate("prefix_rollback","refused_use_does_not_poison_survivor",
                    released[0]==after[0]-Prefix::bytes && released[1]==after[1]);
                authority.close();
            }
        }
        return; // Isolated process: final arm deliberately retains device storage.
    }
    if(env("NINFER_TEST_CONTEXT_SHARED_PREFIX_STARTUP")=="1") {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        auto first=target.create_context(true,false,true);
        auto peer=target.create_context(true,false,true);
        const auto first_bytes=first->persistent_bytes(),peer_bytes=peer->persistent_bytes();
        const auto first_metadata=first->allocation_owner_metadata_bytes();
        const auto peer_metadata=peer->allocation_owner_metadata_bytes();
        {
            Coordinator refused(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto exhausted=Inventory::unlimited();
            exhausted[static_cast<unsigned>(Inventory::Domain::device)]=0;
            refused.bind_physical_resources({},exhausted);
            const auto fallbacks=first->host_kv_stats().device_prefix_fallbacks;
            gate("shared_prefix_startup","optional_credit_refusal_is_fallback",
                !first->prepare_device_prefix_reserved(refused,4096,true));
            gate("shared_prefix_startup","fallback_keeps_context_unattached",
                first->persistent_bytes()==first_bytes && first->host_kv_stats().device_prefix_bytes==0 &&
                first->host_kv_stats().device_prefix_fallbacks==fallbacks+1);
            bool missing_refused=false;
            try{(void)first->share_device_prefix_with_empty(*peer);}
            catch(const std::exception&){missing_refused=true;}
            gate("shared_prefix_startup","fallback_has_no_shareable_cache",missing_refused);
            refused.close();
        }
        Coordinator startup(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        auto limits=Inventory::unlimited();
        limits[static_cast<unsigned>(Inventory::Domain::device)]=4096ULL*1024*2*32;
        startup.bind_physical_resources({},limits);
        gate("shared_prefix_startup","physical_admission_exercised",
            first->prepare_device_prefix_reserved(startup,4096,true));
        gate("shared_prefix_startup","private_attachment_extent",
            first->persistent_bytes()==first_bytes+4096ULL*1024*2*32);
        gate("shared_prefix_startup","reserved_metadata_not_double_counted",
            first->allocation_owner_metadata_bytes()==first_metadata);
        bool legacy_refused=false;
        Inventory legacy_resources;
        try{(void)first->share_device_prefix_with(*peer,legacy_resources);}
        catch(const std::runtime_error& error){legacy_refused=std::string(error.what())==
            "legacy prefix sharing cannot replace reserved owners";}
        gate("shared_prefix_startup","legacy_reserved_replacement_refused",
            legacy_refused && legacy_resources.totals()==Inventory::Totals{});
        gate("shared_prefix_startup","legacy_refusal_preserves_both_extents",
            first->persistent_bytes()==first_bytes+4096ULL*1024*2*32 &&
            peer->persistent_bytes()==peer_bytes &&
            first->allocation_owner_metadata_bytes()==first_metadata &&
            peer->allocation_owner_metadata_bytes()==peer_metadata);
        for(bool failed_source:{false,true}) {
            require(code.size()>=16,"shared prefix failed-lineage fixture requires16 tokens");
            Coordinator failed_owner(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            failed_owner.bind_physical_resources({},Inventory::unlimited());
            auto failed=target.create_context(true,false,true);
            if(failed_source)gate("shared_prefix_startup","failed_source_initial_cache",
                failed->prepare_device_prefix_reserved(failed_owner,4096));
            const auto failed_bytes=failed->persistent_bytes();
            const auto failed_metadata=failed->allocation_owner_metadata_bytes();
            failed->fail_host_kv_prefill_boundary_for_test(1);
            bool injected=false;
            try{failed->prefill(std::span<const std::int64_t>(code.data(),16));}
            catch(const std::runtime_error& error){injected=std::string(error.what())==
                "injected HostKV pre-layer prefill failure";}
            gate("shared_prefix_startup","failed_peer_at_zero_position",injected && failed->position()==0);
            bool refused=false;
            try{
                if(failed_source)(void)failed->share_device_prefix_with_empty(*peer);
                else (void)first->share_device_prefix_with_empty(*failed);
            }
            catch(const std::runtime_error& error){refused=std::string(error.what())==
                "shared reserved prefix requires intact HostKV lineages";}
            gate("shared_prefix_startup",failed_source?"failed_source_sharing_refused":"failed_peer_sharing_refused",refused);
            refused=false;
            try{(void)failed->prepare_device_prefix_reserved(startup,4096);}
            catch(const std::runtime_error& error){refused=std::string(error.what())==
                "reserved prefix requires intact HostKV lineage";}
            gate("shared_prefix_startup","failed_peer_allocation_refused",refused);
            gate("shared_prefix_startup","failed_peer_refusal_preserves_owners",
                failed->persistent_bytes()==failed_bytes &&
                failed->allocation_owner_metadata_bytes()==failed_metadata &&
                first->persistent_bytes()==first_bytes+4096ULL*1024*2*32 &&
                first->allocation_owner_metadata_bytes()==first_metadata &&
                peer->persistent_bytes()==peer_bytes &&
                peer->allocation_owner_metadata_bytes()==peer_metadata);
            failed.reset();failed_owner.close();
        }
        bool self_refused=false;
        try{(void)first->share_device_prefix_with_empty(*first);}
        catch(const std::exception&){self_refused=true;}
        gate("shared_prefix_startup","self_attachment_refused",self_refused);
        auto owner=first->share_device_prefix_with_empty(*peer);
        gate("shared_prefix_startup","shared_bytes_excluded_from_both_private_contexts",
            first->persistent_bytes()==first_bytes && peer->persistent_bytes()==peer_bytes);
        gate("shared_prefix_startup","shared_metadata_excluded_from_private_contexts",
            first->allocation_owner_metadata_bytes()==first_metadata &&
            peer->allocation_owner_metadata_bytes()==peer_metadata);
        gate("shared_prefix_startup","peer_represents_same_extent",
            first->host_kv_stats().device_prefix_bytes==4096ULL*1024*2*32 &&
            peer->host_kv_stats().device_prefix_bytes==4096ULL*1024*2*32);
        bool repeated=false;
        try{(void)first->share_device_prefix_with_empty(*peer);}
        catch(const std::exception&){repeated=true;}
        gate("shared_prefix_startup","repeated_attachment_refused",repeated);
        gate("shared_prefix_startup","refusal_preserves_accounting",
            first->persistent_bytes()==first_bytes && peer->persistent_bytes()==peer_bytes);
        std::weak_ptr<const void> retained=owner;
        owner.reset();first.reset();startup.close();
        gate("shared_prefix_startup","peer_outlives_credit_and_origin",!retained.expired());
        peer.reset();
        gate("shared_prefix_startup","last_peer_releases_unused_cache",retained.expired());
        return;
    }
    if(env("NINFER_TEST_LANE_RESERVED_STARTUP")=="1") {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        auto context=std::shared_ptr<Exl3TextContext>(target.create_context(true,false,true));
        context->prepare_continuation(8);
        const auto context_bytes=context->persistent_bytes();
        // The enclosing fixture owns draft until every authority and lane below
        // has closed. This alias is deliberately limited to this component test.
        auto borrowed_draft=std::shared_ptr<Exl3Dflash2DraftModel>(&draft,[](auto*){});
        for(const auto domain:{Inventory::Domain::device,Inventory::Domain::host_metadata}) {
            Coordinator limited(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();
            const auto metadata=sizeof(Lane)+Lane::retirement_metadata_bytes()+
                (env("NINFER_EXL3_SUFFIX_PROPOSALS")=="1"?sizeof(ninfer::exl3::Exl3SuffixProposer):0);
            limits[static_cast<unsigned>(domain)]=(domain==Inventory::Domain::device?5ULL*16*5120*2:metadata)-1;
            limited.bind_physical_resources({},limits);
            bool refused=false;
            try{(void)Lane::create_reserved(limited,context,borrowed_draft,identity.contract(),nullptr,{});}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            gate("lane_startup",domain==Inventory::Domain::device?"device_one_byte_short":"metadata_one_byte_short",refused);
            gate("lane_startup","refusal_preserves_context",context->persistent_bytes()==context_bytes);
            limited.close();
        }
        for(unsigned fault:{1u,2u,3u,4u}) {
            Coordinator startup(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();
            limits[static_cast<unsigned>(Inventory::Domain::device)]=5ULL*16*5120*2;
            startup.bind_physical_resources({},limits);
            bool injected=false;
            try{(void)Lane::create_reserved(startup,context,borrowed_draft,identity.contract(),nullptr,{},fault);}
            catch(const std::exception& error){injected=std::string(error.what())==
                (fault==1?"injected lane startup preconstruction failure":
                 fault==2?"injected lane startup precommit failure":
                 "resource reservation actual extent/domain mismatch");}
            gate("lane_startup","commit_fault_exercised",injected);
            gate("lane_startup","failure_does_not_grow_context",context->persistent_bytes()==context_bytes);
            auto retry=Lane::create_reserved(startup,context,borrowed_draft,identity.contract(),nullptr,{});
            gate("lane_startup","same_authority_exact_staging_retry",
                retry->construction_added_device_bytes()==5ULL*16*5120*2);
            std::weak_ptr<Lane> retained=retry;
            retry.reset();
            gate("lane_startup","inventory_retains_lane",!retained.expired());
            startup.close();
            gate("lane_startup","close_releases_lane",retained.expired());
        }
        if(env("NINFER_TEST_LANE_STARTUP_RETIREMENT")=="1") {
            Coordinator startup(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            startup.bind_physical_resources({},Inventory::unlimited());
            const auto before=Lane::quarantined_execution_count();
            bool mismatch=false;
            try{(void)Lane::create_reserved(startup,context,borrowed_draft,identity.contract(),nullptr,{},5);}
            catch(const std::invalid_argument& error){mismatch=std::string(error.what())==
                "resource reservation actual extent/domain mismatch";}
            gate("lane_startup","failed_retirement_preserves_mismatch",
                mismatch && Lane::quarantined_execution_count()==before+1);
            const auto retained=Lane::latest_quarantine_for_test();
            gate("lane_startup","failed_startup_retains_full_staging",
                retained && retained->staging_device_bytes==5ULL*16*5120*2 &&
                retained->context_device_bytes==context_bytes && retained->acquisition==0);
            Inventory::Requirement metadata;metadata.configuration=96;
            metadata.add(Inventory::Domain::host_metadata,1,sizeof(int));
            bool sealed=false,called=false;
            try{startup.allocate_startup_resources(metadata,[&](auto) {
                called=true;return Inventory{};
            });}catch(const std::logic_error& error){sealed=std::string(error.what())==
                "serving coordinator sealed after failed startup retirement";}
            gate("lane_startup","failed_retirement_seals_factory",sealed && !called);
            startup.close();startup.close();
        }
        return;
    }
    if(env("NINFER_TEST_CONTEXT_RESERVED_STARTUP")=="1") {
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        {
            Coordinator authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();
            limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=2*sizeof(int);
            authority.bind_physical_resources({},limits);
            Inventory::Requirement initial;initial.configuration=93;
            initial.add(Inventory::Domain::host_metadata,1,sizeof(int));
            std::weak_ptr<int> owner;
            std::shared_ptr<int> external;
            bool rollback_called=false,rollback_exclusive=false,rollback_inventory_released=false;
            bool refused=false;
            try{authority.allocate_startup_resources_growing(initial,[&](auto,auto&& extend) {
                auto value=std::make_shared<int>();owner=value;external=value;
                auto next=initial;next.add(Inventory::Domain::host_metadata,1,sizeof(int));extend(next);
                Inventory actual;actual.add({value,0,Inventory::Domain::host_metadata,sizeof(int)});
                return actual;
            },[&] {
                rollback_called=true;
                rollback_inventory_released=external.use_count()==1;
                bool nested_called=false;
                try{authority.allocate_startup_resources(initial,[&](auto) {
                    nested_called=true;return Inventory{};
                });}catch(const std::logic_error&){rollback_exclusive=!nested_called;}
                external.reset();
            });}catch(const std::invalid_argument&){refused=true;}
            gate("context_startup","coordinator_grown_mismatch_releases_owner",refused && owner.expired());
            gate("context_startup","rollback_precedes_admission_reopening",
                rollback_called && rollback_exclusive && rollback_inventory_released);
            authority.allocate_startup_resources_growing(initial,[&](auto,auto&& extend) {
                auto value=std::make_shared<int>();owner=value;
                auto next=initial;next.add(Inventory::Domain::host_metadata,1,sizeof(int));extend(next);
                bool nested_refused=false,nested_called=false;
                try{authority.allocate_startup_resources_growing(initial,[&](auto,auto&&) {
                    nested_called=true;return Inventory{};
                });}catch(const std::logic_error&){nested_refused=true;}
                gate("context_startup","coordinator_growing_reentrancy_refused",nested_refused && !nested_called);
                nested_refused=false;
                try{authority.allocate_startup_resources(initial,[&](auto) {
                    nested_called=true;return Inventory{};
                });}catch(const std::logic_error&){nested_refused=true;}
                gate("context_startup","coordinator_fixed_reentrancy_refused",nested_refused && !nested_called);
                Inventory actual;actual.add({value,0,Inventory::Domain::host_metadata,2*sizeof(int)});
                return actual;
            });
            gate("context_startup","coordinator_grown_retry_retains_owner",!owner.expired());
            authority.close();
            gate("context_startup","coordinator_grown_close_releases_owner",owner.expired());
        }
        {
            Coordinator authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            authority.bind_physical_resources({},Inventory::unlimited());
            Inventory::Requirement required;required.configuration=95;
            required.add(Inventory::Domain::host_metadata,1,sizeof(int));
            unsigned rollback_calls=0;
            bool original_error=false;
            try{authority.allocate_startup_resources(required,[&](auto) {
                return Inventory{}; // Reservation mismatch after factory return.
            },[&] {
                ++rollback_calls;
                throw std::runtime_error("injected rollback callback failure");
            });}catch(const std::invalid_argument& error) {
                original_error=std::string(error.what())==
                    "resource reservation actual extent/domain mismatch";
            }
            gate("context_startup","rollback_throw_preserves_original_error",
                original_error && rollback_calls==1);
            bool sealed=false,called=false;
            try{authority.allocate_startup_resources(required,[&](auto) {
                called=true;return Inventory{};
            });}catch(const std::logic_error& error) {
                sealed=std::string(error.what())==
                    "serving coordinator sealed after failed startup retirement";
            }
            gate("context_startup","rollback_throw_seals_before_next_factory",sealed && !called);
            authority.close();authority.close();
        }
        {
            Coordinator limited(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();
            limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=
                Exl3TextContext::fixed_owner_metadata_bytes()-1;
            limited.bind_physical_resources({},limits);
            for(unsigned attempt=0;attempt<2;++attempt) {
                bool refused=false;
                // Invalid stage5 is a callback-entry sentinel: if planning is
                // entered, its stage guard throws a different error before Impl.
                try{(void)target.create_context_reserved(limited,true,5);}
                catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
                gate("context_startup","initial_metadata_refused_before_planning",refused);
            }
            limited.close();
        }
        {
            // Ordinary ownership isolates explicit packet growth from the
            // startup authority; reserved callers must charge it at construction.
            auto candidate=target.create_context(true,false,true);
            const auto before_device=candidate->persistent_bytes();
            const auto before_metadata=candidate->allocation_owner_metadata_bytes();
            candidate->prepare_greedy_packet();
            const auto after_device=candidate->persistent_bytes();
            const auto after_metadata=candidate->allocation_owner_metadata_bytes();
            if(ninfer::exl3::exl3_device_greedy_enabled()) {
                gate("context_startup","startup_packet_already_accounted",
                    after_device==before_device && after_metadata==before_metadata);
            } else {
                gate("context_startup","explicit_packet_device_extent",
                    after_device-before_device==Exl3TextContext::greedy_packet_bytes_required());
                gate("context_startup","explicit_packet_owner_record",
                    after_metadata-before_metadata==Exl3TextContext::allocation_record_metadata_bytes()+
                        Exl3TextContext::greedy_transfer_pool_metadata_bytes_required());
            }
            candidate->prepare_greedy_packet();
            gate("context_startup","packet_owner_accounting_idempotent",
                candidate->persistent_bytes()==after_device &&
                candidate->allocation_owner_metadata_bytes()==after_metadata);
        }
        if(env("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH")!="1") {
            // Release each context before constructing the next; compare the
            // optional five layer taps plus embedding trace, not peak overlap.
            auto captured=target.create_context(true,false,true);
            const auto captured_metadata=captured->allocation_owner_metadata_bytes();
            captured.reset();
            auto uncaptured=target.create_context(false,false,true);
            const auto uncaptured_metadata=uncaptured->allocation_owner_metadata_bytes();
            gate("context_startup","six_capture_owner_records",
                captured_metadata>=uncaptured_metadata &&
                captured_metadata-uncaptured_metadata==6*Exl3TextContext::allocation_record_metadata_bytes());
        }
        {
            auto candidate=target.create_context(true,false,true);
            const auto base=candidate->persistent_bytes();
            const auto metadata=Exl3TextContext::continuation_owner_metadata_bytes();
            gate("continuation_startup","represented_owner_metadata_nonzero",metadata>0);
            Coordinator limited(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();
            limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=metadata-1;
            limited.bind_physical_resources({},limits);
            bool refused=false;
            try{candidate->prepare_continuation_reserved(limited,8);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            gate("continuation_startup","owner_metadata_one_byte_short",refused);
            gate("continuation_startup","metadata_refusal_before_attachment",
                candidate->persistent_bytes()==base && candidate->continuation_bytes_required(8)>0);
            limited.close();
            Coordinator exact(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=metadata;
            exact.bind_physical_resources({},limits);
            bool mismatch=false;
            try{candidate->prepare_continuation_reserved(exact,8,4);}
            catch(const std::exception& error){mismatch=std::string(error.what())==
                "resource reservation actual extent/domain mismatch";}
            gate("continuation_startup","metadata_actual_mismatch_exercised",mismatch);
            gate("continuation_startup","metadata_mismatch_before_attachment",
                candidate->persistent_bytes()==base && candidate->continuation_bytes_required(8)>0);
            candidate->prepare_continuation_reserved(exact,8);
            gate("continuation_startup","owner_metadata_exact_fit",candidate->continuation_bytes_required(8)==0);
            exact.close();
        }
        Coordinator startup(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        startup.bind_physical_resources({},Inventory::unlimited());
        auto context=target.create_context_reserved(startup);
        gate("context_startup","reserved_owner_and_device_storage",context && context->persistent_bytes()>0);
        gate("context_startup","reserved_generic_metadata",context->allocation_owner_metadata_bytes()>0);
        gate("context_startup","no_private_prefix",context->host_kv_stats().device_prefix_bytes==0);
        const auto host=context->host_kv_stats();
        const auto device_bytes=context->persistent_bytes();
        const auto metadata_bytes=context->allocation_owner_metadata_bytes()+host.batch_metadata_bytes+
            Exl3TextContext::fixed_owner_metadata_bytes();
        const auto pinned_bytes=host.pinned_staging_bytes;
        for(unsigned fault:{1u,2u,3u,4u}) {
            Coordinator rollback(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();
            limits[static_cast<unsigned>(Inventory::Domain::device)]=device_bytes;
            rollback.bind_physical_resources({},limits);
            bool injected=false;
            try{(void)target.create_context_reserved(rollback,true,fault);}
            catch(const std::exception& error){injected=std::string(error.what())==
                (fault==1?"injected context startup preconstruction failure":
                 fault==2?"injected context startup precommit failure":
                 "resource reservation actual extent/domain mismatch");}
            gate("context_startup","reserved_construction_fault_exercised",injected);
            auto retry=target.create_context_reserved(rollback);
            gate("context_startup","same_authority_exact_fit_after_rollback",retry->persistent_bytes()==device_bytes);
            retry.reset();rollback.close();
        }
        std::weak_ptr<Exl3TextContext> retained=context;
        context.reset();
        gate("context_startup","inventory_retains_context",!retained.expired());
        startup.close();
        gate("context_startup","closed_inventory_releases_context",retained.expired());
        for(const auto domain:{Inventory::Domain::device,Inventory::Domain::host_metadata,
            Inventory::Domain::cuda_registered_host}) {
            const auto bytes=domain==Inventory::Domain::device?device_bytes:
                domain==Inventory::Domain::host_metadata?metadata_bytes:pinned_bytes;
            if(!bytes)continue; // Pinned storage exists only in its enabled configuration.
            Coordinator limited(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();limits[static_cast<unsigned>(domain)]=bytes-1;
            limited.bind_physical_resources({},limits);
            bool refused=false;
            try{(void)target.create_context_reserved(limited);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            gate("context_startup",domain==Inventory::Domain::device?"device_one_byte_short":
                domain==Inventory::Domain::host_metadata?"metadata_one_byte_short":"pinned_one_byte_short",refused);
            limited.close();
            Coordinator exact(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            limits[static_cast<unsigned>(domain)]=bytes;
            exact.bind_physical_resources({},limits);
            auto retry=target.create_context_reserved(exact);
            gate("context_startup","exact_fit_retry",retry->persistent_bytes()==device_bytes);
            retry.reset();exact.close();
        }
        {
            for(unsigned fault:{1u,2u,3u,4u}) {
                auto candidate=target.create_context(true,false,true);
                const auto base=candidate->persistent_bytes();
                constexpr auto required=8ull*(5120ull*4+248320ull*22);
                Coordinator rollback(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
                auto limits=Inventory::unlimited();limits[static_cast<unsigned>(Inventory::Domain::device)]=required;
                limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=
                    Exl3TextContext::continuation_owner_metadata_bytes();
                rollback.bind_physical_resources({},limits);
                bool injected=false;
                try{candidate->prepare_continuation_reserved(rollback,8,fault);}
                catch(const std::exception& error){injected=std::string(error.what())==
                    (fault==1?"injected continuation preconstruction failure":fault==2?
                     "injected continuation precommit failure":"resource reservation actual extent/domain mismatch");}
                gate("context_startup","continuation_commit_fault_exercised",injected);
                gate("context_startup","continuation_commit_failure_unattached",
                    candidate->persistent_bytes()==base && candidate->continuation_bytes_required(8)==required);
                candidate->prepare_continuation_reserved(rollback,8);
                gate("context_startup","continuation_same_authority_retry",candidate->persistent_bytes()==base+required);
                candidate.reset();rollback.close();
            }
            auto unprepared=target.create_context(true,false,true);
            const auto base_bytes=unprepared->persistent_bytes();
            constexpr std::size_t required=8ull*(5120ull*4+248320ull*22);
            Coordinator insufficient(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();limits[static_cast<unsigned>(Inventory::Domain::device)]=required-1;
            insufficient.bind_physical_resources({},limits);
            bool refused=false;
            try{unprepared->prepare_continuation_reserved(insufficient,8);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            gate("context_startup","continuation_credit_refusal",refused);
            gate("context_startup","continuation_refusal_no_attachment",
                unprepared->persistent_bytes()==base_bytes && unprepared->continuation_bytes_required(8)==required);
            insufficient.close();
            Coordinator exact_continuation(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            limits[static_cast<unsigned>(Inventory::Domain::device)]=required;
            exact_continuation.bind_physical_resources({},limits);
            unprepared->prepare_continuation_reserved(exact_continuation,8);
            unprepared->prepare_continuation_reserved(exact_continuation,8);
            gate("context_startup","continuation_exact_fit_idempotent",
                unprepared->persistent_bytes()==base_bytes+required && unprepared->continuation_bytes_required(8)==0);
            exact_continuation.close();
            gate("context_startup","context_retains_continuation_after_credit_close",
                unprepared->continuation_bytes_required(8)==0);
        }
        {
            for(unsigned fault:{1u,2u,3u,4u}) {
                auto candidate=target.create_context(true,false,true);
                const auto base=candidate->persistent_bytes();
                const auto required=candidate->transaction_bytes_required();
                const auto metadata=Exl3TextContext::transaction_owner_metadata_bytes();
                Coordinator reserved(cache,Coordinator::Policy{1,1,
                    memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
                auto limits=Inventory::unlimited();
                limits[static_cast<unsigned>(Inventory::Domain::device)]=required;
                limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=metadata;
                reserved.bind_physical_resources({},limits);
                bool injected=false;
                try{candidate->prepare_transaction_reserved(reserved,fault);}
                catch(const std::exception& error) {
                    injected=std::string(error.what())==
                        (fault==1?"injected P2 transaction preconstruction failure":
                         fault==2?"injected P2 transaction precommit failure":
                         "resource reservation actual extent/domain mismatch");
                }
                gate("repair_checkpoint_startup","commit_fault_exercised",injected);
                gate("repair_checkpoint_startup","failure_unattached",
                    candidate->persistent_bytes()==base &&
                    candidate->transaction_bytes_required()==required &&
                    !candidate->transaction_prepared());
                candidate->prepare_transaction_reserved(reserved);
                candidate->prepare_transaction_reserved(reserved);
                gate("repair_checkpoint_startup","same_authority_exact_fit_idempotent",
                    candidate->persistent_bytes()==base+required &&
                    candidate->transaction_bytes_required()==0 &&
                    candidate->transaction_bytes()==required);
                candidate.reset();reserved.close();
            }
            auto candidate=target.create_context(true,false,true);
            const auto base=candidate->persistent_bytes();
            const auto required=candidate->transaction_bytes_required();
            Coordinator limited(cache,Coordinator::Policy{1,1,
                memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            auto limits=Inventory::unlimited();
            limits[static_cast<unsigned>(Inventory::Domain::device)]=required-1;
            limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=
                Exl3TextContext::transaction_owner_metadata_bytes();
            limited.bind_physical_resources({},limits);
            bool refused=false;
            try{candidate->prepare_transaction_reserved(limited);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            gate("repair_checkpoint_startup","device_one_byte_short",refused);
            gate("repair_checkpoint_startup","refusal_precedes_allocation",
                candidate->persistent_bytes()==base && !candidate->transaction_prepared());
            limited.close();
        }
        {
            Coordinator lane_authority(cache,Coordinator::Policy{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
            lane_authority.bind_physical_resources({},Inventory::unlimited());
            auto owned_context=target.create_context_reserved(lane_authority);
            const auto base_context_bytes=owned_context->persistent_bytes();
            constexpr std::size_t continuation_per_row=5120ull*4+248320ull*22;
            for(int rows:{2,3,4,5,6,7,8})
                gate("context_startup","independent_continuation_requirement",
                    owned_context->continuation_bytes_required(rows)==rows*continuation_per_row);
            for(int rows:{0,1,9}) {
                bool refused=false;
                try{(void)owned_context->continuation_bytes_required(rows);}
                catch(const std::exception&){refused=true;}
                gate("context_startup","invalid_continuation_requirement",refused);
            }
            gate("context_startup","continuation_planning_does_not_allocate",
                owned_context->persistent_bytes()==base_context_bytes);
            std::weak_ptr<Exl3TextContext> witness=owned_context;
            for(const auto* option:{"NINFER_EXL3_SUFFIX_PROPOSALS","NINFER_EXL3_BOUNDED_VERIFIER_HORIZON",
                                    "NINFER_EXL3_NATIVE_VERIFIER_HORIZON","NINFER_EXL3_REPAIR_CHECKPOINT",
                                    "NINFER_EXL3_DEVICE_SEED_HANDOFF"}) {
                struct RestoreLaneOption {
                    const char* key;std::string saved;
                    ~RestoreLaneOption(){_putenv_s(key,saved.c_str());}
                } restore{option,env(option)};
                _putenv_s(option,"invalid");
                bool refused=false;
                try{auto invalid=std::make_unique<Lane>(owned_context,draft,identity.contract());}
                catch(const std::invalid_argument& error){refused=std::string(error.what())==
                    (std::string_view(option)=="NINFER_EXL3_SUFFIX_PROPOSALS"?
                     "suffix proposals must be0 or1":
                     std::string_view(option)=="NINFER_EXL3_BOUNDED_VERIFIER_HORIZON"?
                     "bounded verifier horizon must be0 or1":
                     std::string_view(option)=="NINFER_EXL3_NATIVE_VERIFIER_HORIZON"?
                     "native verifier horizon must be0 or1":
                     std::string_view(option)=="NINFER_EXL3_REPAIR_CHECKPOINT"?
                     "repair checkpoint must be0 or1":"device seed handoff must be0 or1");}
                gate("context_startup","invalid_lane_option_refused",refused);
                gate("context_startup","invalid_lane_option_no_context_growth",
                    owned_context->persistent_bytes()==base_context_bytes &&
                    owned_context->continuation_bytes_required(8)==8*continuation_per_row);
            }
            {
                struct RestoreDeviceSeedOptions {
                    std::string greedy=env("NINFER_EXL3_DEVICE_GREEDY");
                    std::string seed=env("NINFER_EXL3_DEVICE_SEED_HANDOFF");
                    ~RestoreDeviceSeedOptions(){
                        _putenv_s("NINFER_EXL3_DEVICE_GREEDY",greedy.c_str());
                        _putenv_s("NINFER_EXL3_DEVICE_SEED_HANDOFF",seed.c_str());
                    }
                } restore;
                _putenv_s("NINFER_EXL3_DEVICE_GREEDY","0");
                _putenv_s("NINFER_EXL3_DEVICE_SEED_HANDOFF","1");
                bool refused=false;
                try{auto invalid=std::make_unique<Lane>(
                    owned_context,draft,identity.contract());}
                catch(const std::invalid_argument& error){refused=
                    std::string(error.what())==
                        "device seed handoff requires device greedy";}
                gate("context_startup","device_seed_prerequisite_refused",refused);
                gate("context_startup","device_seed_prerequisite_no_growth",
                    owned_context->persistent_bytes()==base_context_bytes);
            }
            for(unsigned stage:{1u,2u,3u,4u,5u,6u}) {
                bool injected=false;
                try{auto failed=std::make_unique<Lane>(owned_context,draft,identity.contract(),nullptr,stage);}
                catch(const std::runtime_error& error){injected=std::string(error.what())==
                    (stage==6?"injected lane event allocation failure":"injected lane staging allocation failure");}
                gate("context_startup","lane_partial_allocation_fault",injected);
                gate("context_startup","lane_failure_preserves_base_context",
                    owned_context->persistent_bytes()==base_context_bytes &&
                    owned_context->continuation_bytes_required(8)==8*continuation_per_row);
            }
            auto owned_lane=std::make_unique<Lane>(owned_context,draft,identity.contract());
            const auto checkpoint_bytes=owned_lane->repair_checkpoint_enabled()?
                owned_context->transaction_bytes():0;
            gate("context_startup","independent_continuation_actual_delta",
                owned_context->persistent_bytes()==
                    base_context_bytes+8*continuation_per_row+checkpoint_bytes);
            gate("context_startup","prepared_continuation_zero_requirement",owned_context->continuation_bytes_required(8)==0);
            owned_context->prepare_continuation(8);
            gate("context_startup","prepared_continuation_no_growth",
                owned_context->persistent_bytes()==
                    base_context_bytes+8*continuation_per_row+checkpoint_bytes);
            bool changed_capacity_refused=false;
            try{(void)owned_context->continuation_bytes_required(4);}
            catch(const std::exception&){changed_capacity_refused=true;}
            gate("context_startup","changed_continuation_capacity_refused",changed_capacity_refused);
            gate("context_startup","lane_base_inventory_snapshot",owned_lane->context_entry_device_bytes()==base_context_bytes);
            gate("context_startup","lane_added_inventory_partition",
                owned_lane->construction_added_device_bytes()+base_context_bytes==owned_lane->persistent_bytes());
            gate("context_startup","lane_added_contains_private_taps",
                owned_lane->construction_added_device_bytes()>=5ULL*16*5120*2);
            owned_context.reset();lane_authority.close();
            gate("context_startup","lane_retains_reserved_context",!witness.expired());
            owned_lane.reset();
            gate("context_startup","lane_release_retires_reserved_context",witness.expired());
        }
        return;
    }
    if(env("NINFER_TEST_GREEDY_PENDING_FAILURE")=="1") {
        auto context=target.create_context(false,false,true);
        context->prefill(std::span<const std::int64_t>(code.data(),16));
        const auto before=Exl3TextContext::quarantined_greedy_packet_transfers();
        auto pending=context->submit_greedy_packet(false,61,71);
        Exl3TextContext::fail_pending_greedy_packet_completion_for_test(pending);
        bool failed=false;
        try{(void)context->finish_greedy_packet(std::move(pending),61,71);}
        catch(const std::runtime_error& error){failed=std::string(error.what())==
            "pending greedy packet completion failed";}
        gate("greedy_pending","uncertain_completion_fails_closed",failed &&
            Exl3TextContext::quarantined_greedy_packet_transfers()==before+1);
        return; // Deliberate retained transfer: fresh process only.
    }
    if(env("NINFER_TEST_GREEDY_ALLOCATION")=="1") {
        struct RestoreGreedyOption {
            std::string saved;
            ~RestoreGreedyOption(){_putenv_s("NINFER_EXL3_DEVICE_GREEDY",saved.c_str());}
        } restore{env("NINFER_EXL3_DEVICE_GREEDY")};
        std::size_t without_packet=0;
        {
            _putenv_s("NINFER_EXL3_DEVICE_GREEDY","0");
            auto context=target.create_context(false,false,true);
            without_packet=context->persistent_bytes();
            const auto pool_bytes=Exl3TextContext::greedy_packet_bytes_required();
            gate("allocation","greedy_fixed_packet_requirement",
                pool_bytes==3*16*sizeof(ninfer::exl3::Exl3GreedyRow) &&
                context->greedy_transfer_pool_registered_bytes_required()==
                    2*16*sizeof(ninfer::exl3::Exl3GreedyRow) &&
                context->greedy_transfer_pool_capacity()==2);
            context->prepare_greedy_packet();
            gate("allocation","greedy_first_prepare_delta",context->persistent_bytes()==without_packet+pool_bytes);
            context->prepare_greedy_packet();
            gate("allocation","greedy_repeat_prepare_no_growth",context->persistent_bytes()==without_packet+pool_bytes);
            context->reset();context->prepare_greedy_packet();
            gate("allocation","greedy_reset_keeps_storage",context->persistent_bytes()==without_packet+pool_bytes);
        }
        {
            _putenv_s("NINFER_EXL3_DEVICE_GREEDY","1");
            auto context=target.create_context(false,false,true);
            gate("allocation","greedy_startup_delta",context->persistent_bytes()==
                without_packet+Exl3TextContext::greedy_packet_bytes_required());
                context->prepare_greedy_packet();
            gate("allocation","greedy_startup_prepare_no_growth",context->persistent_bytes()==
                without_packet+Exl3TextContext::greedy_packet_bytes_required());
        }
        {
            _putenv_s("NINFER_EXL3_DEVICE_GREEDY","1");
            auto context=target.create_context(false,false,true);
            context->prefill(std::span<const std::int64_t>(code.data(),16));
            gate("allocation","greedy_transfer_pool_initially_available",
                context->greedy_transfer_pool_available_for_test()==2);
            auto first=context->submit_greedy_packet(false,81,91);
            const auto first_id=Exl3TextContext::greedy_transfer_identity_for_test(first);
            auto first_reader=context->device_greedy_seed(first,81,91);
            auto duplicate_reader=first_reader;
            (void)context->finish_greedy_packet(std::move(first),81,91);
            gate("allocation","retained_diagnostic_holds_first_slot",
                first_id && first_reader.owner==duplicate_reader.owner &&
                context->greedy_transfer_pool_available_for_test()==1);
            auto second=context->submit_greedy_packet(false,81,91);
            const auto second_id=Exl3TextContext::greedy_transfer_identity_for_test(second);
            auto lagging_consumer=context->device_greedy_seed(second,81,91);
            (void)context->finish_greedy_packet(std::move(second),81,91);
            bool exhausted=false;
            try{(void)context->submit_greedy_packet(false,81,91);}
            catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){exhausted=true;}
            gate("allocation","all_retained_readers_refuse_pool_overcommit",
                exhausted && second_id && second_id!=first_id &&
                context->greedy_transfer_pool_available_for_test()==0);
            first_reader={};duplicate_reader={};
            auto reused=context->submit_greedy_packet(false,81,91);
            gate("allocation","released_diagnostic_reuses_exact_slot",
                Exl3TextContext::greedy_transfer_identity_for_test(reused)==first_id);
            (void)context->finish_greedy_packet(std::move(reused),81,91);
            const auto before_quarantine=Exl3TextContext::quarantined_greedy_packet_transfers();
            context.reset();
            gate("allocation","retired_context_preserves_lagging_reader",
                lagging_consumer.owner &&
                Exl3TextContext::quarantined_greedy_packet_transfers()==before_quarantine);
            lagging_consumer={};
            gate("allocation","clean_lagging_reader_retirement_not_quarantined",
                Exl3TextContext::quarantined_greedy_packet_transfers()==before_quarantine);
        }
        {
            constexpr int rows=3,vocabulary=257,stride=260;
            std::vector<std::uint16_t> scores(rows*stride,0xfc00); // -infinity
            scores[0]=0x3c00;scores[256]=0x3c00; // equal maxima across reduction tails
            scores[stride+12]=0x0000;scores[stride+13]=0x8000; // +0/-0 tie
            scores[2*stride+256]=0x4000; // final logical element after one full block
            auto context=target.create_context(false,false,true);
            context->prepare_greedy_packet();
            const auto packet=context->greedy_packet_from_scores_for_test(
                scores,rows,vocabulary,stride);
            gate("greedy_selector","lowest_index_and_partial_block",
                packet.decisions[0].token==0 && packet.decisions[1].token==12 &&
                packet.decisions[2].token==256);
            bool refused=false;
            try{(void)context->greedy_packet_from_scores_for_test(
                scores,rows,vocabulary-1,vocabulary-2);}
            catch(const std::runtime_error&){refused=true;}
            gate("greedy_selector","invalid_stride_refused_before_launch",refused);
        }
        for(const std::uint16_t invalid:{std::uint16_t{0x7e00},std::uint16_t{0x7c00}}) {
            constexpr int rows=3,vocabulary=257,stride=260;
            std::vector<std::uint16_t> scores(rows*stride,0xbc00);
            scores[0]=0x3c00;scores[stride]=0x3c00;scores[2*stride]=0x3c00;
            // Invalid data in a later row must fail the complete packet even
            // if policy would have stopped after row0.
            scores[2*stride+256]=invalid;
            auto context=target.create_context(false,false,true);
            context->prepare_greedy_packet();
            bool refused=false;
            try{(void)context->greedy_packet_from_scores_for_test(
                scores,rows,vocabulary,stride);}
            catch(const std::runtime_error& error){refused=std::string(error.what())==
                "invalid/nonfinite greedy packet row";}
            gate("greedy_selector","late_nonfinite_preserves_full_score_refusal",refused);
        }
        {
            auto context=target.create_context(false,false,true);
            context->prefill(std::span<const std::int64_t>(code.data(),16));
            const auto scores=context->logits_host();
            const auto expected=std::max_element(scores.begin(),scores.end())-scores.begin();
            auto pending=context->submit_greedy_packet(false,41,51);
            gate("greedy_pending","submission_returns_owned_unready_handle",pending.valid());
            const auto packet=context->finish_greedy_packet(std::move(pending),41,51);
            gate("greedy_pending","completion_matches_full_score_authority",
                !pending.valid() && packet.decisions[0].token==expected);
            bool refused=false;
            try{(void)context->finish_greedy_packet(std::move(pending),41,51);}
            catch(const std::runtime_error&){refused=true;}
            gate("greedy_pending","duplicate_result_use_refused",refused);

            auto stale=context->submit_greedy_packet(false,41,52);
            refused=false;
            try{(void)context->finish_greedy_packet(std::move(stale),41,53);}
            catch(const std::runtime_error&){refused=true;}
            gate("greedy_pending","recycled_execution_scope_refused",refused && !stale.valid());
            {
                auto cancelled=context->submit_greedy_packet(false,42,54);
                gate("greedy_pending","cancelled_consumer_retains_transfer_owner",cancelled.valid());
            }
            auto isolated=context->submit_greedy_packet(false,43,55);
            const auto next=context->finish_greedy_packet(std::move(isolated),43,55);
            gate("greedy_pending","abandoned_consumer_drains_before_new_result",
                next.decisions[0].token==expected &&
                Exl3TextContext::quarantined_greedy_packet_transfers()==0);
        }
        return;
    }
    if(env("NINFER_TEST_GREEDY_BATCH")=="1") {
        using Batch=ninfer::exl3::Exl3EngineGreedyPacketBatch;
        Batch batch;batch.set_timeout_for_test(std::chrono::milliseconds(5));
        std::array<std::shared_ptr<Exl3TextContext>,3> contexts{
            target.create_context(false,false,true),target.create_context(false,false,true),
            target.create_context(false,false,true)};
        std::array<std::int64_t,3> expected{};
        for(std::size_t lane=0;lane<contexts.size();++lane) {
            const auto& input=lane==1?prose:code;
            contexts[lane]->prefill(std::span<const std::int64_t>(input.data(),16+lane));
            const auto scores=contexts[lane]->logits_host();
            expected[lane]=std::max_element(scores.begin(),scores.end())-scores.begin();
        }
        // T002: the actual deferred Engine transport must carry the complete
        // physical owner/final-use record, not just reusable handle addresses.
        {
            auto pending=contexts[2]->submit_greedy_packet(
                false,190,290,nullptr,true);
            auto source=contexts[2]->greedy_batch_source(pending,190,290);
            const auto first_event=source.final_use_event;
            const auto first_generation=source.final_use_generation;
            const auto first_transfer=source.owner.get();
            gate("greedy_batch","pending_source_strong_owners_and_charges",
                source.owns_pending_execution() && source.owner &&
                source.model_owner && source.model_identity &&
                !source.stream_owner &&
                source.final_use_event==reinterpret_cast<std::uintptr_t>(
                    source.consumer_done) &&
                source.device_bytes==16*sizeof(Exl3GreedyRow) &&
                source.registered_host_bytes==16*sizeof(Exl3GreedyRow) &&
                source.metadata_bytes>0);
            auto missing_final_event=source;
            missing_final_event.final_use_event=0;
            gate("greedy_batch","missing_final_event_refused",
                !missing_final_event.owns_pending_execution());
            auto aliased_model=source;
            aliased_model.model_owner=std::shared_ptr<const void>(
                source.model_owner,source.model_owner.get());
            gate("greedy_batch","alias_owner_preserves_pending_record",
                aliased_model.owns_pending_execution());
            const auto packet=batch.finish(*contexts[2],std::move(pending),190,290);
            gate("greedy_batch","owner_record_actual_consumer_completion",
                packet.decisions[0].token==expected[2]);
            source={};missing_final_event={};aliased_model={};

            auto reused=contexts[2]->submit_greedy_packet(
                false,191,291,nullptr,true);
            auto reused_source=contexts[2]->greedy_batch_source(reused,191,291);
            gate("greedy_batch","event_address_reuse_requires_new_generation",
                reused_source.owner.get()==first_transfer &&
                reused_source.final_use_event==first_event &&
                reused_source.final_use_generation>first_generation &&
                reused_source.acquisition==191 && reused_source.execution==291);
            (void)batch.finish(*contexts[2],std::move(reused),191,291);
        }
        {
            cudaStream_t raw_stream=nullptr;
            cuda_check(cudaStreamCreateWithFlags(&raw_stream,cudaStreamNonBlocking),
                "greedy batch explicit stream creation");
            auto stream_owner=std::shared_ptr<cudaStream_t>(
                new cudaStream_t(raw_stream),[](cudaStream_t* owned) noexcept {
                    (void)cudaStreamDestroy(*owned);delete owned;
                });
            std::weak_ptr<cudaStream_t> stream_lifetime=stream_owner;
            {
                auto borrowed=target.create_context(false,false,true);
                borrowed->prefill(std::span<const std::int64_t>(prose.data(),18));
                bool refused=false;
                try{(void)borrowed->submit_greedy_packet(
                    false,192,292,raw_stream,true);}
                catch(const std::runtime_error&){refused=true;}
                gate("greedy_batch","ownerless_explicit_stream_refused",refused);
            }
            auto retiring=target.create_context(false,false,true);
            retiring->retain_execution_stream_owner(raw_stream,stream_owner);
            retiring->prefill(std::span<const std::int64_t>(prose.data(),18));
            Exl3GreedyBatchSource detached;
            {
                auto abandoned=retiring->submit_greedy_packet(
                    false,192,292,raw_stream,true);
                detached=retiring->greedy_batch_source(abandoned,192,292);
            }
            stream_owner.reset();
            retiring.reset();
            gate("greedy_batch","wrapper_retirement_keeps_pending_dependencies",
                detached.owns_pending_execution() && detached.owner &&
                detached.model_owner && detached.model_identity &&
                detached.producer_stream==raw_stream &&
                detached.stream_owner && !stream_lifetime.expired());
            detached={};
            gate("greedy_batch","final_owner_retires_explicit_stream",
                stream_lifetime.expired());
        }
        const auto run_pair=[&](bool invalidate_second) {
            std::array<std::optional<Exl3GreedyPacket>,2> results;
            std::array<std::exception_ptr,2> failures{};
            if(invalidate_second)batch.invalidate_next_row_for_test(2);
            std::thread second([&]{try{
                auto pending=contexts[1]->submit_greedy_packet(false,202,302,nullptr,true);
                results[1]=batch.finish(*contexts[1],std::move(pending),202,302);
            }catch(...){failures[1]=std::current_exception();}});
            std::thread first([&]{try{
                auto pending=contexts[0]->submit_greedy_packet(false,201,301,nullptr,true);
                results[0]=batch.finish(*contexts[0],std::move(pending),201,301);
            }catch(...){failures[0]=std::current_exception();}});
            first.join();second.join();return std::pair{std::move(results),failures};
        };
        auto [mapped,mapped_failures]=run_pair(false);
        auto stats=batch.stats();
        gate("greedy_batch","lane_order_maps_independent_rows",
            !mapped_failures[0] && !mapped_failures[1] && mapped[0] && mapped[1] &&
            mapped[0]->decisions[0].token==expected[0] &&
            mapped[1]->decisions[0].token==expected[1] && stats.batches==1 &&
            stats.batched_rows==2 && stats.singles==0);
        auto [mixed,mixed_failures]=run_pair(true);
        std::array<bool,2> invalid{};
        for(unsigned lane=0;lane<2;++lane)try{
            if(mixed_failures[lane])std::rethrow_exception(mixed_failures[lane]);
        }catch(const std::runtime_error& error){invalid[lane]=std::string(error.what())==
            "invalid/nonfinite greedy packet row";}
        gate("greedy_batch","mixed_row_error_isolated",
            invalid[0]!=invalid[1] &&
            (invalid[0] || (mixed[0] && mixed[0]->decisions[0].token==expected[0])) &&
            (invalid[1] || (mixed[1] && mixed[1]->decisions[0].token==expected[1])));

        const auto run_unequal_extents=[&](int left_rows,int right_rows) {
            const std::array<int,2> extents{left_rows,right_rows};
            std::array<std::shared_ptr<Exl3TextContext>,2> wide;
            std::array<std::vector<std::int64_t>,2> controls;
            std::array<std::optional<Exl3GreedyPacket>,2> packets;
            std::array<std::exception_ptr,2> failures{};
            for(unsigned lane=0;lane<2;++lane) {
                wide[lane]=target.create_context(false,false,true);
                wide[lane]->prepare_continuation(16);
                const auto& input=lane?prose:code;
                wide[lane]->prefill(std::span<const std::int64_t>(input.data(),24+lane));
                if(extents[lane]==1) {
                    const auto scores=wide[lane]->logits_host();
                    controls[lane].push_back(
                        std::max_element(scores.begin(),scores.end())-scores.begin());
                } else {
                    wide[lane]->continue_rows(std::span<const std::int64_t>(
                        input.data()+40,extents[lane]));
                    const auto scores=wide[lane]->continuation_logits_host();
                    const auto vocabulary=scores.size()/static_cast<std::size_t>(extents[lane]);
                    controls[lane].reserve(extents[lane]);
                    for(int row=0;row<extents[lane];++row) {
                        const auto first=scores.begin()+static_cast<std::ptrdiff_t>(row*vocabulary);
                        controls[lane].push_back(std::max_element(
                            first,first+static_cast<std::ptrdiff_t>(vocabulary))-first);
                    }
                }
            }
            const auto before=batch.stats();
            const auto submit=[&](unsigned lane) {try{
                auto pending=wide[lane]->submit_greedy_packet(extents[lane]>1,
                    240+lane,340+lane,nullptr,true);
                packets[lane]=batch.finish(*wide[lane],std::move(pending),
                    240+lane,340+lane);
            }catch(...){failures[lane]=std::current_exception();}};
            std::thread right([&]{submit(1);});
            std::thread left([&]{submit(0);});
            left.join();right.join();
            const auto after=batch.stats();
            bool mapped=!failures[0] && !failures[1];
            for(unsigned lane=0;lane<2 && mapped;++lane) {
                mapped=packets[lane] && packets[lane]->rows==extents[lane];
                for(int row=0;row<extents[lane] && mapped;++row)
                    mapped=packets[lane]->decisions[row].token==controls[lane][row];
            }
            for(unsigned lane=0;lane<2;++lane)
                if(extents[lane]>1)wide[lane]->finish_exact_continuation();
            return std::tuple{mapped,before,after};
        };
        auto [m1_m7,m1_m7_before,m1_m7_after]=run_unequal_extents(1,7);
        gate("greedy_batch","M1_plus_M7_explicit_offsets",
            m1_m7 && m1_m7_after.batches==m1_m7_before.batches+1 &&
            m1_m7_after.batched_rows==m1_m7_before.batched_rows+8);
        auto [m8_m3,m8_m3_before,m8_m3_after]=run_unequal_extents(8,3);
        gate("greedy_batch","M8_plus_M3_excludes_padding_rows",
            m8_m3 && m8_m3_after.batches==m8_m3_before.batches+1 &&
            m8_m3_after.batched_rows==m8_m3_before.batched_rows+11);
        auto [m8_m9,m8_m9_before,m8_m9_after]=run_unequal_extents(8,9);
        gate("greedy_batch","row_capacity_overflow_serializes_exact_extents",
            m8_m9 && m8_m9_after.batches==m8_m9_before.batches &&
            m8_m9_after.batched_rows==m8_m9_before.batched_rows &&
            m8_m9_after.singles==m8_m9_before.singles+2);

        std::atomic<bool> cancelled{true};
        std::optional<Exl3GreedyPacket> live;std::exception_ptr cancelled_failure;
        std::thread cancelled_lane([&]{try{
            auto pending=contexts[0]->submit_greedy_packet(false,211,311,nullptr,true);
            (void)batch.finish(*contexts[0],std::move(pending),211,311,&cancelled);
        }catch(...){cancelled_failure=std::current_exception();}});
        std::thread live_lane([&]{
            auto pending=contexts[1]->submit_greedy_packet(false,212,312,nullptr,true);
            live=batch.finish(*contexts[1],std::move(pending),212,312);
        });
        cancelled_lane.join();live_lane.join();
        stats=batch.stats();
        gate("greedy_batch","partial_cancellation_preserves_peer",
            cancelled_failure && live && live->decisions[0].token==expected[1] &&
            stats.cancelled==1 && stats.singles==1);

        const auto before_overflow=batch.stats();
        std::array<std::optional<Exl3GreedyPacket>,3> overflow_results;
        std::atomic<bool> pair_claimed{false},third_entered{false};
        batch.observe_next_pair_for_test([&]{
            pair_claimed.store(true,std::memory_order_release);
            while(!third_entered.load(std::memory_order_acquire))std::this_thread::yield();
        });
        const auto overflow=[&](unsigned lane) {
            auto pending=contexts[lane]->submit_greedy_packet(
                false,220+lane,320+lane,nullptr,true);
            overflow_results[lane]=batch.finish(*contexts[lane],std::move(pending),
                220+lane,320+lane);
        };
        std::thread first_overflow([&]{overflow(0);});
        std::thread second_overflow([&]{overflow(1);});
        while(!pair_claimed.load(std::memory_order_acquire))std::this_thread::yield();
        std::thread third_overflow([&]{
            third_entered.store(true,std::memory_order_release);overflow(2);
        });
        first_overflow.join();second_overflow.join();third_overflow.join();
        const auto after_overflow=batch.stats();
        gate("greedy_batch","capacity_two_overflow_uses_single_transport",
            after_overflow.batches==before_overflow.batches+1 &&
            after_overflow.batched_rows==before_overflow.batched_rows+2 &&
            after_overflow.singles==before_overflow.singles+1 &&
            std::all_of(overflow_results.begin(),overflow_results.end(),
                [](const auto& result){return result.has_value();}));
        return;
    }
    if(env("NINFER_TEST_TAP_SEGMENT_DESCRIPTOR")=="1") {
        auto context=std::shared_ptr<Exl3TextContext>(target.create_context(true,false,true));
        context->prefill(std::span<const std::int64_t>(code.data(),24));
        const int logical=context->position()-1;
        auto root_revision=std::make_shared<const int>(7);
        ninfer::exl3::Exl3CommittedTapBinding binding{context,root_revision,logical,401,501};
        auto segment=context->committed_tap_segment(binding,0,1,logical,0,
            logical,1,logical,1,false,false);
        segment.validate();
        gate("tap_segment","strong_context_and_model_owners",
            segment.owner.get()==context.get() && segment.model_identity==context->model_identity() &&
            segment.root_revision==root_revision && segment.root_position==logical &&
            segment.current() && segment.rows==1 && segment.logical_first==logical &&
            segment.attempt_first==logical && segment.source_partition_first==logical);
        auto missing=segment;missing.planes[3]=nullptr;
        bool refused=false;try{missing.validate();}catch(const std::invalid_argument&){refused=true;}
        gate("tap_segment","missing_plane_refused",refused);
        auto wrong_frontier=segment;wrong_frontier.root_position=logical+1;
        refused=false;try{wrong_frontier.validate();}catch(const std::invalid_argument&){refused=true;}
        gate("tap_segment","root_frontier_after_segment_refused",refused);
        missing.owner.reset();
        auto foreign=std::make_shared<int>(7);
        refused=false;try{(void)context->committed_tap_segment(
            {foreign,root_revision,logical,401,501},0,1,logical,0,logical,1,logical,1);}
        catch(const std::runtime_error&){refused=true;}
        gate("tap_segment","foreign_owner_refused_before_descriptor",refused);
        const auto logits=context->logits_host();
        context->decode(std::max_element(logits.begin(),logits.end())-logits.begin());
        refused=false;try{segment.validate();}catch(const std::invalid_argument&){refused=true;}
        gate("tap_segment","later_writer_stales_old_generation",refused && !segment.current());
        binding.context_owner.reset();
        std::weak_ptr<Exl3TextContext> retained=context;context.reset();
        gate("tap_segment","descriptor_retains_source_after_external_release",!retained.expired());
        segment.owner.reset();
        gate("tap_segment","last_descriptor_owner_releases_source",retained.expired());
        return;
    }
    if(env("NINFER_TEST_DFLASH_DESTRUCTOR_RETIREMENT")=="1") {
        // Deliberate quarantine: separate future process; no subsequent work.
        std::shared_ptr<Exl3Dflash2DraftModel> owned(draft.create_execution());
        auto stream_backing=std::make_shared<int>(1);
        std::weak_ptr<Exl3Dflash2DraftModel> draft_lifetime=owned;
        std::weak_ptr<int> stream_lifetime=stream_backing;
        auto lane=std::make_unique<Lane>(target.create_context(true),owned,identity.contract(),nullptr,stream_backing);
        owned.reset();stream_backing.reset();
        const auto before=Lane::quarantined_execution_count();
        const auto context_bytes=lane->context().persistent_bytes();
        const auto proposal_address=lane->proposal_host_address_for_test();
        lane->fail_destructor_drain_for_test();lane.reset();
        gate("retirement","failed_destructor_retains_owned_backing",
            Lane::quarantined_execution_count()==before+1 && !draft_lifetime.expired() && !stream_lifetime.expired());
        const auto quarantined=Lane::latest_quarantine_for_test();
        gate("retirement","quarantine_preserves_host_proposal_address",quarantined && proposal_address &&
            quarantined->proposal_host_address==proposal_address && quarantined->proposal_host_bytes==8*sizeof(std::int64_t));
        gate("retirement","quarantine_records_retained_device_extents",quarantined &&
            quarantined->context_device_bytes==context_bytes &&
            quarantined->staging_device_bytes==5ULL*16*5120*2);
        gate("retirement","failed_destructor_records_drain_outcome",quarantined &&
            quarantined->retirement.phase==ninfer::exl3::Exl3RetirementPhase::quarantined &&
            quarantined->retirement.first_drain_error==static_cast<int>(cudaErrorUnknown) &&
            !quarantined->retirement.reusable() &&
            quarantined->completion_authority_generation!=0 &&
            quarantined->completion_retirement_generation==0);
        gate("retirement","idle_quarantine_has_no_fabricated_completion",quarantined &&
            quarantined->acquisition==0 && quarantined->final_use.generation==0 &&
            !quarantined->final_use.ready);
        bool refused=false;try{auto replacement=make_lane();}
        catch(const std::runtime_error& error){refused=std::string(error.what()).find("unresolved DFlash execution retirement")!=std::string::npos;}
        gate("retirement","failed_destructor_blocks_lane_reconstruction",refused);
        return;
    }
    struct Result {
        Root root;std::vector<std::int64_t> tokens;std::vector<int> partitions;
        std::array<std::vector<std::uint16_t>,5> taps;
        double wall=0,construct=0,acquire=0,decode=0,release=0,teardown=0,first=0,registry=0;
        Lane::Stats stats;std::uint64_t verified=0,replayed=0,accepted=0,h2d=0,d2h=0;
        std::uint64_t selector_batched_anchor_chain_calls=0,fast_mia_parity_w1_settlements=0;
        std::uint64_t gdn_graph_replays=0,gdn_graph_launch_cpu_ns=0;
        std::uint64_t full_layer_graph_replays=0,full_layer_graph_launch_cpu_ns=0;
        std::size_t bytes=0;
        ninfer::exl3::Exl3VerifierHorizonPolicy::Counters policy;
        unsigned horizon=8;
    };
    // Two requests on the same physical lane, with an unrelated fixture in
    // between, distinguish actual reuse from merely excluding construction.
    std::unique_ptr<Lane> warm;
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const auto name=fixture.first;require(fixture.second->size()>=prefix,"real DFlash fixture size");
        Root initial;std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> oracle;
        std::vector<std::int64_t> expected;std::array<std::vector<std::uint16_t>,5> taps;
        const auto prep=Clock::now();
        {
            auto ctx=target.create_context(true);ctx->prepare_continuation(8);
            initial=Request::initialize(*ctx,std::span<const std::int64_t>(fixture.second->data(),prefix),1024)->compact_draft(draft,staging);
            for(int i=0;i<outputs;++i){const auto token=sample_target(*ctx);expected.push_back(token);ctx->decode(token);const auto row=ctx->exact_tap_rows_host();for(int t=0;t<5;++t)taps[t].insert(taps[t].end(),row[t].begin(),row[t].end());}
            oracle=ctx->export_exact_host_state(nullptr,true);
        }
        std::cout<<"REAL_DFLASH reference fixture="<<name<<" prefill_and_oracle_ms="<<ms(prep,Clock::now())<<std::endl;
        {
            auto ctx=target.create_context(true);ctx->prepare_continuation(8);
            const auto full=Request::initialize(*ctx,std::span<const std::int64_t>(fixture.second->data(),prefix),1024);
            for(int count:{3,17,65}){
                const std::span<const std::int64_t> suffix(fixture.second->data(),count);
                const auto reference=full->append_prompt(*ctx,suffix,1024);
                reference->restore_draft(draft,staging);const auto ring=draft.export_host_ring(nullptr,false);
                Root compact;
                {
                    using History=ninfer::exl3::Exl3TokenHistory;
                    struct Disarm {~Disarm(){History::fail_allocation_for_test(0);}} disarm;
                    unsigned failures=0;
                    const auto initial_fingerprint=initial->token_fingerprint();
                    for(unsigned fault=1;fault<=16 && !compact;++fault) {
                        unsigned progress=0;
                        History::fail_allocation_for_test(fault);
                        try {
                            compact=initial->append_prompt_compact(*ctx,draft,staging,suffix,1024,
                                [&](int){++progress;});
                            History::fail_allocation_for_test(0);
                        } catch(const std::bad_alloc&) {
                            History::fail_allocation_for_test(0);++failures;
                            gate(name,"compact_history_failure_preserves_target_and_draft",
                                progress==0 && initial->token_fingerprint()==initial_fingerprint &&
                                ctx->export_exact_host_state(nullptr,true)->same_payload(*reference->state()) &&
                                ring->same_payload(*draft.export_host_ring(nullptr,false)));
                        }
                    }
                    gate(name,"compact_history_allocation_retry",compact && failures>0);
                    require(bool(compact),"compact history allocation schedule exhausted before retry");
                }
                gate(name,"compact_prompt_suffix_state_ring",compact->state()->same_payload(*reference->state()) && ring->same_payload(*draft.export_host_ring(nullptr,false)));
                Cache concurrent_compact(Cache::Policy{2,1,2ULL<<30,8ULL<<30},identity);
                const auto input=initial->token_suffix();
                require(concurrent_compact.admit_input_authority(initial,input,memory.ullAvailPhys).admitted,
                    "concurrent compact fixture initial admission");
                auto extended_input=input;extended_input.insert(extended_input.end(),suffix.begin(),suffix.end());
                const auto before_refusal=ctx->export_exact_host_state(nullptr,true);
                for(bool missing_draft:{false,true}) {
                    bool refused=false;
                    try {
                        (void)concurrent_compact.prepare_concurrent(*ctx,extended_input,
                            extended_input.size(),memory.ullAvailPhys,1024,{},
                            missing_draft?nullptr:&draft,missing_draft?&staging:nullptr);
                    } catch(const std::invalid_argument& error) {
                        refused=std::string_view(error.what())=="compact cached prefix needs resident drafter";
                    }
                    gate(name,"concurrent_compact_missing_owner_refusal",refused &&
                        ctx->export_exact_host_state(nullptr,true)->same_payload(*before_refusal) &&
                        concurrent_compact.roots().size()==1 && concurrent_compact.roots().front()==initial);
                }
                bool suffix_cancelled=false;
                try {
                    (void)concurrent_compact.prepare_concurrent(*ctx,extended_input,
                        extended_input.size(),memory.ullAvailPhys,8,
                        [](int){throw std::runtime_error("T234 compact extension cancelled");},&draft,&staging);
                } catch(const std::runtime_error& error) {
                    suffix_cancelled=std::string_view(error.what())=="T234 compact extension cancelled";
                }
                gate(name,"concurrent_compact_cancel_retains_prefix",suffix_cancelled &&
                    concurrent_compact.roots().size()==1 && concurrent_compact.roots().front()==initial &&
                    ctx->export_exact_host_state(nullptr,true)->same_payload(*initial->state()));
                const auto prepared=concurrent_compact.prepare_concurrent(*ctx,extended_input,
                    extended_input.size(),memory.ullAvailPhys,1024,{},&draft,&staging);
                gate(name,"concurrent_compact_suffix_authority",
                    prepared.metrics.cache_hit && !prepared.metrics.preparation_reused &&
                    prepared.metrics.reused_prompt_tokens==input.size() &&
                    prepared.metrics.executed_prompt_tokens==suffix.size() &&
                    prepared.request->matches_tokens(extended_input) &&
                    prepared.request->state()->same_payload(*reference->state()) &&
                    prepared.request->matches_detached_conditioning_for_test(*ring));
                // Explicit complete-input lookup may retain a later input root
                // than the semantic minimum frontier, without rerunning suffix.
                const auto complete=concurrent_compact.prepare_concurrent(*ctx,extended_input,1,
                    memory.ullAvailPhys,1024,{},&draft,&staging,true);
                gate(name,"concurrent_compact_complete_input",
                    complete.metrics.cache_hit && !complete.metrics.preparation_reused &&
                    complete.metrics.executed_prompt_tokens==0 &&
                    complete.metrics.reused_prompt_tokens==extended_input.size() &&
                    complete.request==prepared.request);
            }
            bool cancelled=false;
            try{initial->append_prompt_compact(*ctx,draft,staging,std::span<const std::int64_t>(fixture.second->data(),17),8,[](int){throw std::runtime_error("injected suffix cancellation");});}catch(const std::runtime_error&){cancelled=true;}
            gate(name,"compact_suffix_cancel_restore",cancelled && ctx->export_exact_host_state(nullptr,true)->same_payload(*initial->state()));
        }
        const auto equal=[&](const Result& r,bool compare_taps,std::string_view diagnostic={}){
            const bool token_equal=r.tokens==expected;
            const bool state_equal=r.root->state()->same_payload(*oracle);
            const bool taps_equal=!compare_taps || r.taps==taps;
            initial->restore_draft(draft,staging);int offset=0;
            for(int rows:r.partitions){std::array<const std::uint16_t*,5> ptr{};for(int t=0;t<5;++t){cuda_check(cudaMemcpy(staging[t],taps[t].data()+static_cast<std::size_t>(offset)*5120,rows*5120ULL*2,cudaMemcpyHostToDevice),"real DFlash oracle ring");ptr[t]=staging[t];}draft.commit_prefill_block(ptr.data(),rows,prefix+offset);offset+=rows;}
            const auto ring=draft.export_host_ring(nullptr,false);r.root->restore_draft(draft,staging);
            const bool ring_equal=ring->same_payload(*draft.export_host_ring(nullptr,false));
            if(!diagnostic.empty()) {
                const auto record=[&](std::string_view component,bool pass) {
                    checks<<name<<','<<diagnostic<<'_'<<component<<','<<pass<<'\n';
                };
                record("tokens",token_equal);record("state",state_equal);
                record("taps",taps_equal);record("ring",ring_equal);checks.flush();
            }
            return token_equal && state_equal && taps_equal && ring_equal;
        };
        const auto run=[&](bool reuse,bool correct,int pair){
            Result r;std::unique_ptr<Lane> cold;const auto begin=Clock::now();
            std::unique_ptr<ninfer::NvtxRange> request_range;
            if(decode_nvtx_enabled && !correct)request_range=std::make_unique<ninfer::NvtxRange>(
                std::string("realdflash.timed.request.")+name+(reuse?".reused.":".cold.")+
                std::to_string(pair));
            if(reuse){if(!warm)warm=make_lane();}else cold=make_lane();
            auto& lane=reuse?*warm:*cold;r.construct=ms(begin,Clock::now());
            const auto stats0=lane.stats();
            const auto selector_calls0=draft.selector_batched_anchor_chain_calls();
            const auto reg0=coordinator.stats();
            const auto acquire=Clock::now();coordinator.admit(initial);auto lease=coordinator.acquire();require(lease.has_value(),"real DFlash coordinator acquire");lane.acquire(*lease);r.acquire=ms(acquire,Clock::now());
            const auto transfer0=lane.context().host_kv_stats();const auto decode=Clock::now();
            const auto gdn_graph0=lane.context().host_kv_gdn_segment_graph_stats();
            const auto full_graph0=lane.context().host_kv_full_layer_graph_stats();
            if(correct)gate(name,"horizon_request_reset",lane.verifier_horizon_policy().counters().published_rounds==0);
            const auto supplied_costs=env("NINFER_TEST_HORIZON_SUPPLIED_COSTS");
            require(supplied_costs.empty() || (supplied_costs=="1" &&
                env("NINFER_EXL3_BOUNDED_VERIFIER_HORIZON")=="1"),"supplied test costs require explicit horizon policy");
            const bool native_horizon=env("NINFER_EXL3_NATIVE_VERIFIER_HORIZON")=="1";
            if(correct)gate(name,"horizon_cost_menu_request_reset",!lane.verifier_horizon_policy().has_cost_menu());
            std::optional<ninfer::exl3::Exl3VerifierHorizonPolicy::CostMenu> test_menu;
            if(supplied_costs=="1") {
                // Synthetic decision fixture, never a timing receipt/calibration.
                ninfer::exl3::Exl3VerifierHorizonPolicy::CostMenu menu;
                menu.fixed_neural_ms=1;menu.minimum_samples=1;
                menu.estimates={decltype(menu)::Estimate{2,2,1,true},
                    decltype(menu)::Estimate{4,8,1,true},decltype(menu)::Estimate{8,32,1,true}};
                bool refused=false;
                try{lane.set_horizon_cost_menu(menu,lane.execution_epoch()+1,lease->acquisition);}
                catch(const std::invalid_argument&){refused=true;}
                require(refused && !lane.verifier_horizon_policy().has_cost_menu(),"stale epoch installed cost menu");
                refused=false;
                try{lane.set_horizon_cost_menu(menu,lane.execution_epoch(),lease->acquisition+1);}
                catch(const std::invalid_argument&){refused=true;}
                require(refused && !lane.verifier_horizon_policy().has_cost_menu(),"foreign acquisition installed cost menu");
                lane.set_horizon_cost_menu(menu,lane.execution_epoch(),lease->acquisition,2);
                test_menu=menu;
                const auto updates=lane.stats().cost_menu_updates;refused=false;
                try{lane.set_horizon_cost_menu(menu,lane.execution_epoch(),lease->acquisition);}
                catch(const std::invalid_argument&){refused=true;}
                require(refused && lane.stats().cost_menu_updates==updates,
                    "same-frontier cost menu update bypassed hold interval");
            }
            unsigned previous_full_horizon=0;
            const auto suffix_selection=env("NINFER_TEST_SUFFIX_OUTCOME_SELECTION");
            require(suffix_selection.empty() || (suffix_selection=="1" && env("NINFER_EXL3_SUFFIX_PROPOSALS")=="1"),
                "suffix outcome fixture requires explicit suffix proposals");
            if(suffix_selection=="1") {
                lane.set_suffix_selection_limits({1,2,7,0},lane.execution_epoch(),lane.lease().acquisition);
                bool refused=false;
                try{lane.set_suffix_selection_limits({1,2,0,8},lane.execution_epoch(),lane.lease().acquisition);}
                catch(const std::invalid_argument&){refused=true;}
                require(refused,"same-request suffix constraints rebound");
            }
            while(r.tokens.size()<outputs){
                const auto proposal_calls_before=lane.stats().proposal_calls;
                const auto neural_input_before=lane.stats().neural_input_rows;
                const auto neural_returned_before=lane.stats().neural_returned_suffix_rows;
                const auto neural_discarded_before=lane.stats().neural_discarded_suffix_rows;
                const auto suffix_fallbacks_before=lane.stats().suffix_policy_neural_fallbacks;
                const auto decisions_before=lane.stats().horizon_decisions;
                const auto switches_before=lane.stats().full_budget_horizon_switches;
                const bool full_budget=outputs-r.tokens.size()>=8 &&
                    lane.context().max_context()-lane.context().position()>=8;
                const auto expected_horizon=lane.verifier_horizon_policy().horizon(
                    static_cast<unsigned>(std::min<std::size_t>(8,outputs-r.tokens.size())));
                auto proposed=[&]{
                    std::unique_ptr<ninfer::NvtxRange> range;
                    if(decode_nvtx_enabled && !correct)
                        range=std::make_unique<ninfer::NvtxRange>("realdflash.timed.propose");
                    return lane.propose(static_cast<int>(std::min<std::size_t>(8,outputs-r.tokens.size())));
                }();
                const bool neural_executed=lane.stats().proposal_calls>proposal_calls_before;
                const auto physical_neural_rows=neural_executed?
                    static_cast<unsigned>(native_horizon?proposed.size():8u):0u;
                require(lane.stats().neural_input_rows-neural_input_before==physical_neural_rows &&
                    lane.stats().neural_returned_suffix_rows-neural_returned_before==(neural_executed?proposed.size()-1:0u) &&
                    lane.stats().neural_discarded_suffix_rows-neural_discarded_before==
                        (neural_executed?physical_neural_rows-proposed.size():0u),
                    "physical neural extent and unused suffix row accounting");
                if(lane.stats().suffix_policy_neural_fallbacks>suffix_fallbacks_before)
                    require(lane.stats().proposal_calls==proposal_calls_before+1,
                        "suffix outcome fallback failed to execute neural proposal route");
                if(supplied_costs=="1")require(proposed.size()<=expected_horizon &&
                    lane.stats().horizon_decisions==decisions_before+1,"real proposal ignored supplied horizon decision");
                if(supplied_costs=="1") {
                    const bool switched=full_budget && previous_full_horizon && previous_full_horizon!=expected_horizon;
                    require(lane.stats().full_budget_horizon_switches==switches_before+(switched?1:0),
                        "horizon switch counter included cap or missed policy switch");
                    if(full_budget)previous_full_horizon=expected_horizon;
                }
                const auto refuse_outstanding_update=[&] {
                    if(!test_menu)return;
                    const auto updates=lane.stats().cost_menu_updates;bool refused=false;
                    try{lane.set_horizon_cost_menu(*test_menu,lane.execution_epoch(),lane.lease().acquisition,2);}
                    catch(const std::logic_error&){refused=true;}
                    require(refused && lane.stats().cost_menu_updates==updates,
                        "outstanding decision allowed cost menu replacement");
                };
                refuse_outstanding_update();
                auto pending=[&]{
                    std::unique_ptr<ninfer::NvtxRange> range;
                    if(decode_nvtx_enabled && !correct)
                        range=std::make_unique<ninfer::NvtxRange>("realdflash.timed.verify");
                    return lane.verify(proposed);
                }();
                refuse_outstanding_update();
                const auto suffix_before=lane.stats();
                auto published=[&]{
                    std::unique_ptr<ninfer::NvtxRange> range;
                    if(decode_nvtx_enabled && !correct)
                        range=std::make_unique<ninfer::NvtxRange>("realdflash.timed.publish");
                    return lane.publish(coordinator,pending);
                }();
                if(correct) {
                    const bool suffix=pending.proposed_rows>1 && pending.costs.neural_rows==0;
                    gate(name,"suffix_publication_outcome_attribution",
                        lane.stats().suffix_published_rounds==suffix_before.suffix_published_rounds+(suffix?1:0) &&
                        lane.stats().suffix_accepted_rows==suffix_before.suffix_accepted_rows+
                            (suffix && pending.verification.accepted>0?pending.verification.accepted-1:0) &&
                        lane.stats().suffix_repair_rows==suffix_before.suffix_repair_rows+
                            (suffix?pending.verification.replay_rows:0) &&
                        lane.stats().suffix_published_skipped_neural_blocks==
                            suffix_before.suffix_published_skipped_neural_blocks+(suffix?1:0));
                }
                if(test_menu) {
                    const auto rounds=lane.verifier_horizon_policy().counters().published_rounds;
                    const auto updates=lane.stats().cost_menu_updates;bool refused=false;
                    auto next=*test_menu;
                    const bool prefer_four=(rounds/2)%2!=0;
                    next.estimates[0]->complete_ms=prefer_four?8:2;
                    next.estimates[1]->complete_ms=prefer_four?2:8;
                    try{lane.set_horizon_cost_menu(next,lane.execution_epoch(),lane.lease().acquisition,2);}
                    catch(const std::invalid_argument&){refused=true;}
                    require(refused==(rounds%2!=0) && lane.stats().cost_menu_updates==updates+(refused?0:1),
                        "published-round hold interval did not govern updates");
                    if(!refused)test_menu=next;
                }
                if(correct) {
                    gate(name,"proposal_cost_matches_physical_work",pending.costs.neural_rows &&
                        *pending.costs.neural_rows==physical_neural_rows &&
                        pending.costs.neural_ms && std::isfinite(*pending.costs.neural_ms));
                    gate(name,"verification_cost_observed",pending.costs.verification_ms &&
                        std::isfinite(*pending.costs.verification_ms) && *pending.costs.verification_ms>=0);
                    gate(name,"unmeasured_costs_remain_unknown",!pending.costs.repair_ms &&
                        !pending.costs.complete_ms && !pending.costs.publication_ms);
                    if(env("NINFER_EXL3_BOUNDED_VERIFIER_HORIZON")=="1")
                        gate(name,"published_cost_matches_pending",lane.verifier_horizon_policy().last_costs().verification_ms==
                            pending.costs.verification_ms);
                }
                if(correct && env("NINFER_EXL3_BOUNDED_VERIFIER_HORIZON")=="1")
                    gate(name,"horizon_publication_only",lane.verifier_horizon_policy().counters().published_rounds==r.partitions.size()+1);
                r.verified+=pending.verification.verification_rows;r.replayed+=pending.verification.replay_rows;r.accepted+=pending.verification.accepted;
                r.tokens.insert(r.tokens.end(),published.tokens.begin(),published.tokens.end());r.partitions.push_back(static_cast<int>(published.tokens.size()));
                if(correct)for(int t=0;t<5;++t)r.taps[t].insert(r.taps[t].end(),pending.verification.committed_taps[t].begin(),pending.verification.committed_taps[t].end());
                if(!r.first)r.first=ms(begin,Clock::now());
                if(!correct)gaps<<name<<','<<pair<<','<<(reuse?"reused":"cold")<<','<<r.partitions.size()<<','<<published.tokens.size()<<','<<ms(begin,Clock::now())<<'\n';
            }
            r.decode=ms(decode,Clock::now());r.root=lane.lease().root;r.bytes=lane.persistent_bytes();
            r.policy=lane.verifier_horizon_policy().counters();r.horizon=lane.verifier_horizon_policy().horizon(8);
            const auto transfer1=lane.context().host_kv_stats();r.h2d=transfer1.h2d_bytes-transfer0.h2d_bytes;r.d2h=transfer1.d2h_bytes-transfer0.d2h_bytes;
            const auto gdn_graph1=lane.context().host_kv_gdn_segment_graph_stats();
            const auto full_graph1=lane.context().host_kv_full_layer_graph_stats();
            r.gdn_graph_replays=gdn_graph1.replays-gdn_graph0.replays;
            r.gdn_graph_launch_cpu_ns=gdn_graph1.launch_cpu_ns-gdn_graph0.launch_cpu_ns;
            r.full_layer_graph_replays=full_graph1.replays-full_graph0.replays;
            r.full_layer_graph_launch_cpu_ns=full_graph1.launch_cpu_ns-full_graph0.launch_cpu_ns;
            r.stats=lane.stats();r.stats.proposal_calls-=stats0.proposal_calls;r.stats.proposed_rows-=stats0.proposed_rows;r.stats.proposal_ms-=stats0.proposal_ms;r.stats.verification_ms-=stats0.verification_ms;
            r.stats.neural_input_rows-=stats0.neural_input_rows;
            r.stats.neural_returned_suffix_rows-=stats0.neural_returned_suffix_rows;
            r.stats.neural_discarded_suffix_rows-=stats0.neural_discarded_suffix_rows;
            r.stats.suffix_calls-=stats0.suffix_calls;r.stats.suffix_rows-=stats0.suffix_rows;r.stats.suffix_misses-=stats0.suffix_misses;
            r.stats.fast_mia_parity_w1_settlements-=stats0.fast_mia_parity_w1_settlements;
            r.selector_batched_anchor_chain_calls=draft.selector_batched_anchor_chain_calls()-selector_calls0;
            r.fast_mia_parity_w1_settlements=r.stats.fast_mia_parity_w1_settlements;
            if(correct){Exl3HostResidencyProbe probe;const std::array<Root,1> roots{r.root};Request::visit_host_allocations(roots,[&](const void* p,std::size_t n){probe.add(p,n);},false);const auto measured=probe.measure();gate(name,"published_payload_resident_locked",measured.allocated_union_bytes==measured.resident_tensor_bytes && measured.allocated_union_bytes==measured.locked_tensor_bytes);}
            const auto release=Clock::now();auto final_lease=lane.lease();lane.release();coordinator.complete(final_lease);r.release=ms(release,Clock::now());
            const auto reg1=coordinator.stats();r.registry=reg1.resident_registry_ms-reg0.resident_registry_ms;
            const auto teardown=Clock::now();cold.reset();cuda_check(cudaDeviceSynchronize(),"real DFlash teardown");r.teardown=ms(teardown,Clock::now());r.wall=ms(begin,Clock::now());return r;
        };
        for(bool reuse:{false,true}){const auto r=run(reuse,true,-1);gate(name,reuse?"reused_serial_state_taps_ring":"cold_serial_state_taps_ring",equal(r,true,reuse?"reused_serial":"cold_serial") && (r.stats.proposal_calls>0 || r.stats.suffix_calls>0));}
        // Abort before verification and after a real tentative L2 completion;
        // late Pending cannot publish across abort, release, or request reuse.
        for(int phase:{0,1}){
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());auto proposed=warm->propose(8);
            Lane::Pending stale;if(phase)stale=warm->verify(proposed);
            if(phase) {
                auto wrong_event=stale;++wrong_event.completion_generation;bool refused=false;
                try{warm->publish(coordinator,wrong_event);}catch(const std::invalid_argument&){refused=true;}
                gate(name,"same_request_wrong_event_generation_refused",refused);
                auto wrong_policy=stale;++wrong_policy.policy_revision;refused=false;
                try{warm->publish(coordinator,wrong_policy);}catch(const std::invalid_argument&){refused=true;}
                gate(name,"same_request_wrong_policy_revision_refused",refused);
            }
            warm->abort();gate(name,"abort_restores_exact_root",warm->context().export_exact_host_state(nullptr,true)->same_payload(*initial->state()));
            gate(name,"abort_rebinds_completed_draft_lineage",
                initial->draft_lineage_ready(draft,warm->lease().acquisition,warm->execution_epoch(),prefix));
            gate(name,"draft_lineage_rejects_foreign_scope_or_position",
                !initial->draft_lineage_ready(draft,warm->lease().acquisition+1,warm->execution_epoch(),prefix) &&
                !initial->draft_lineage_ready(draft,warm->lease().acquisition,warm->execution_epoch()+1,prefix) &&
                !initial->draft_lineage_ready(draft,warm->lease().acquisition,warm->execution_epoch(),prefix+1));
            gate(name,"aborted_proposals_do_not_train_horizon",warm->verifier_horizon_policy().counters().published_rounds==0);
            if(phase){bool refused=false;try{warm->publish(coordinator,stale);}catch(const std::invalid_argument&){refused=true;}gate(name,"stale_completion_refused",refused);}
            auto retry=warm->verify(proposed);
            gate(name,"abort_clears_proposal_cost_witness",!retry.costs.neural_ms && !retry.costs.neural_rows);
            warm->abort();
            auto lease=warm->lease();const auto retirement=warm->release();
            if(!phase) {
                Coordinator::Retirement missing;
                bool refused=false;
                try{coordinator.cancel_retired(lease,missing);}
                catch(const std::invalid_argument&){refused=true;}
                gate(name,"cancel_recycle_requires_physical_retirement",refused);
                coordinator.cancel_retired(lease,retirement);
            } else {
                coordinator.yield_retired(lease,retirement);
                const auto reacquired=coordinator.acquire();
                gate(name,"same_slot_reacquire_advances_acquisition",
                    reacquired && reacquired->ticket.request_id==lease.ticket.request_id &&
                    reacquired->acquisition>lease.acquisition);
                bool late=false;
                try{coordinator.cancel_retired(*reacquired,retirement);}
                catch(const std::invalid_argument&){late=true;}
                gate(name,"late_retirement_cannot_certify_reacquired_slot",late);
                warm->acquire(*reacquired);
                const auto current=warm->lease();
                const auto current_retirement=warm->release();
                gate(name,"lane_retirement_generations_never_reuse",
                    current_retirement.generation()>retirement.generation() &&
                    current_retirement.final_use_generation()!=0);
                coordinator.cancel_retired(current,current_retirement);
            }
        }
        if(env("NINFER_TEST_CONDITIONAL_B8")=="1") {
            require(prefix+16<=target.max_context() && outputs>=16,"conditional B8 fixture capacity");
            for(bool constrained:{true,false}) {
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                if(constrained) {
                    bool stale=false;try{warm->set_conditional_publication_limit(10,warm->execution_epoch()+1,warm->lease().acquisition);}
                    catch(const std::invalid_argument&){stale=true;}
                    gate(name,"conditional_latency_wrong_epoch_refused",stale);
                    warm->set_conditional_publication_limit(10,warm->execution_epoch(),warm->lease().acquisition);
                    bool rebound=false;try{warm->set_conditional_publication_limit(20,warm->execution_epoch(),warm->lease().acquisition);}
                    catch(const std::invalid_argument&){rebound=true;}
                    gate(name,"conditional_latency_request_binding",rebound);
                }
                auto latency_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                const auto calls=warm->stats().proposal_calls;
                if(constrained)for(auto estimate:{std::optional<double>{},std::optional<double>{11}}) {
                    bool refused=false;try{(void)warm->propose_conditional_second(latency_first,16,nullptr,estimate);}
                    catch(const std::invalid_argument&){refused=true;}
                    gate(name,"conditional_latency_refuses_before_neural",refused && warm->stats().proposal_calls==calls);
                }
                // Synthetic estimates test admission only, never latency claims.
                const auto admitted=warm->propose_conditional_second(latency_first,16,nullptr,
                    constrained?std::optional<double>{10}:std::nullopt);
                gate(name,"conditional_latency_boundary_and_request_reset",admitted.tokens.size()==8 &&
                    warm->stats().proposal_calls==calls+1 &&
                    admitted.maximum_publication_ms==(constrained?std::optional<double>{10}:std::nullopt) &&
                    admitted.estimated_complete_ms==(constrained?std::optional<double>{10}:std::nullopt));
                auto changed_estimate=admitted;changed_estimate.estimated_complete_ms=9;
                bool changed_refused=false;
                try{(void)warm->verify_conditional_second(latency_first,changed_estimate);}
                catch(const std::invalid_argument&){changed_refused=true;}
                gate(name,"conditional_latency_changed_plan_refused",changed_refused);
                warm->abort();auto lease=warm->lease();warm->release();coordinator.cancel(lease.ticket,true);
            }
            if(conditional_credit_fixture && conditional_credit_refusal) {
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                auto credit_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                const auto before=warm->stats();bool refused=false;
                const auto pending_blocks=ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>();
                try{(void)warm->propose_conditional_second(credit_first,16,&coordinator);}
                catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
                gate(name,"conditional_credit_refusal_before_factory",refused &&
                    warm->stats().conditional_retention_factories==before.conditional_retention_factories &&
                    warm->stats().proposal_calls==before.proposal_calls &&
                    ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>()==pending_blocks);
                const auto fallback=warm->try_propose_conditional_second(credit_first,16,coordinator);
                gate(name,"conditional_credit_engine_fallback_preserves_first",!fallback &&
                    !warm->has_conditional_retention() &&
                    warm->stats().proposal_calls==before.proposal_calls &&
                    warm->stats().conditional_retention_factories==before.conditional_retention_factories &&
                    credit_first.root->state()->position()==initial->state()->position()+8);
                warm->publish(coordinator,credit_first);
                auto lease=warm->lease();warm->release();coordinator.complete(lease);
            }
            if(conditional_credit_fixture && !conditional_credit_refusal)for(int action:{0,1,2}) {
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                auto credit_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                const auto blocks=ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>();
                const auto credit_second=warm->propose_conditional_second(credit_first,16,&coordinator);
                auto retained_control=warm->conditional_predecessor_owner_for_test();
                bool premature=false;try{warm->collect_conditional_retention(coordinator);}
                catch(const std::logic_error&){premature=true;}
                gate(name,"conditional_credit_retained_while_pending",premature && credit_second.tokens.size()==8);
                if(action==2) {
                    const auto combined=warm->verify_conditional_second(credit_first,credit_second);
                    warm->publish(coordinator,combined);
                } else if(action==1)warm->publish(coordinator,credit_first);else warm->abort();
                bool release_refused=false;try{warm->release();}
                catch(const std::logic_error&){release_refused=true;}
                gate(name,"conditional_credit_collection_precedes_release",release_refused);
                gate(name,"conditional_credit_matching_owner_collected",warm->collect_conditional_retention(coordinator));
                gate(name,"conditional_pending_weak_control_survives_collection",retained_control.expired() &&
                    ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>()==blocks+1);
                if(action==0) {
                    auto retry_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                    const auto before=warm->stats();bool refused=false;
                    try{(void)warm->propose_conditional_second(retry_first,16,&coordinator);}
                    catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
                    gate(name,"conditional_weak_control_blocks_exact_cap",refused &&
                        warm->stats().conditional_retention_factories==before.conditional_retention_factories &&
                        warm->stats().proposal_calls==before.proposal_calls);
                    retained_control.reset();
                    const auto retry_second=warm->propose_conditional_second(retry_first,16,&coordinator);
                    gate(name,"conditional_final_weak_allows_exact_retry",retry_second.tokens.size()==8 &&
                        warm->stats().conditional_retention_factories==before.conditional_retention_factories+1);
                    warm->abort();
                    gate(name,"conditional_retry_retention_collected",warm->collect_conditional_retention(coordinator));
                }
                retained_control.reset();
                gate(name,"conditional_pending_final_weak_releases_control",
                    ninfer::exl3::bounded_shared_live_blocks_for_test<Lane::Pending>()==blocks);
                auto lease=warm->lease();warm->release();
                if(action)coordinator.complete(lease);else coordinator.cancel(lease.ticket,true);
            }
            if(env("NINFER_EXL3_SUFFIX_PROPOSALS")!="1") {
                struct ConditioningObservation {
                    Lane* lane=nullptr;
                    Root expected;
                    unsigned calls=0;
                    std::uint32_t families=0;
                };
                auto observation=std::make_shared<ConditioningObservation>();
                observation->lane=warm.get();observation->expected=initial;
                // Installation requires an unacquired draft. Expire borrowed
                // lane access even if proposal/verification exits exceptionally.
                struct ExpireObservation {
                    std::shared_ptr<ConditioningObservation> state;
                    ~ExpireObservation(){state->lane=nullptr;state->expected.reset();}
                } expire_observation{observation};
                draft.set_shared_q_executor([observation](const ninfer::exl3::Exl3DraftSharedQContinuation& offer) {
                    if(!observation->lane)return false;
                    const auto root=observation->lane->shared_proposal_root();
                    require(root && root==observation->expected &&
                        offer.matches_conditioning_owner(root->projected_metadata_owner()) &&
                        offer.projection.position==root->state()->position() &&
                        observation->lane->matches_shared_proposal(offer.segment,offer.seed),
                        "conditional callback lost active predecessor conditioning");
                    ++observation->calls;
                    const auto family=static_cast<unsigned>(offer.projection.family);
                    require(family<static_cast<unsigned>(ninfer::exl3::Exl3TargetSharedFamily::count),
                        "draft callback emitted an unknown projection family");
                    observation->families|=std::uint32_t{1}<<family;
                    return false; // Observe the real offer; retain serial arithmetic.
                },true,true,true,true);
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                const auto calls_before=warm->stats().proposal_calls;
                const auto first_tokens=warm->propose(8);
                for(const auto family:{ninfer::exl3::Exl3TargetSharedFamily::draft_q,
                    ninfer::exl3::Exl3TargetSharedFamily::draft_k,ninfer::exl3::Exl3TargetSharedFamily::draft_v,
                    ninfer::exl3::Exl3TargetSharedFamily::draft_o,ninfer::exl3::Exl3TargetSharedFamily::draft_down,
                    ninfer::exl3::Exl3TargetSharedFamily::draft_gate,ninfer::exl3::Exl3TargetSharedFamily::draft_up})
                    gate(name,"private_draft_family_offer_NOT_EXERCISED",
                        (observation->families&(std::uint32_t{1}<<static_cast<unsigned>(family)))!=0);
                gate(name,"conditional_first_proposal_scope_expired",!warm->shared_proposal_active_for_test());
                auto neural_first=warm->verify(first_tokens);
                const auto acquired=warm->lease().root;
                const auto position=neural_first.root->state()->position();
                const Root foreign_control(acquired.get(),[](const Request*){});
                const Root unowned(std::shared_ptr<const Request>{},acquired.get());
                gate(name,"conditional_dispatch_conditioning_scope",
                    ninfer::exl3::shared_conditioning_scope_valid(neural_first.root,acquired,true,position) &&
                    ninfer::exl3::shared_conditioning_scope_valid(acquired,acquired,true,acquired->state()->position()) &&
                    !ninfer::exl3::shared_conditioning_scope_valid(acquired,neural_first.root,true,acquired->state()->position()) &&
                    !ninfer::exl3::shared_conditioning_scope_valid(foreign_control,acquired,true,acquired->state()->position()) &&
                    !ninfer::exl3::shared_conditioning_scope_valid(unowned,acquired,true,acquired->state()->position()) &&
                    !ninfer::exl3::shared_conditioning_scope_valid(neural_first.root,acquired,false,position) &&
                    !ninfer::exl3::shared_conditioning_scope_valid(neural_first.root,acquired,true,position+1));
                gate(name,"conditional_real_first_neural_work",warm->stats().proposal_calls==calls_before+1 &&
                    neural_first.costs.neural_rows==8);
                const bool eligible=neural_first.proposed_rows==8 && neural_first.verification.accepted==8 &&
                    !neural_first.verification.rejected && !neural_first.verification.stopped;
                Lane::Pending final=neural_first;
                if(eligible) {
                    observation->expected=neural_first.root;
                    const auto callbacks_before=observation->calls;
                    const auto neural_second=warm->propose_conditional_second(neural_first,16);
                    gate(name,"conditional_second_predecessor_callback_observed",
                        observation->calls>callbacks_before);
                    gate(name,"conditional_second_proposal_scope_expired",
                        !warm->shared_proposal_active_for_test() && !warm->shared_proposal_root());
                    final=warm->verify_conditional_second(neural_first,neural_second);
                    gate(name,"conditional_real_two_neural_blocks",warm->stats().proposal_calls==calls_before+2 &&
                        final.conditional_observations &&
                        (*final.conditional_observations)[0].costs.neural_rows==8 &&
                        (*final.conditional_observations)[1].costs.neural_rows==8);
                } else {
                    bool refused=false;try{(void)warm->propose_conditional_second(neural_first,16);}
                    catch(const std::invalid_argument&){refused=true;}
                    gate(name,"conditional_real_first_fallback",refused && warm->stats().proposal_calls==calls_before+1);
                }
                const auto publication=warm->publish(coordinator,final);
                gate(name,"conditional_real_neural_publication_matches_reference",
                    !publication.tokens.empty() && publication.tokens.size()<=16 &&
                    std::equal(publication.tokens.begin(),publication.tokens.end(),expected.begin()));
                auto lease=warm->lease();warm->release();coordinator.complete(lease);
                draft.set_shared_q_executor({});
                auto reference=target.create_context(true);reference->restore_exact_host_state(*initial->state());
                for(std::size_t row=0;row<publication.tokens.size();++row)reference->decode(expected[row]);
                gate(name,"conditional_real_neural_full_state",
                    reference->export_exact_host_state(nullptr,true)->same_payload(*final.root->state()));
                // Which branch was reached is explicit; fallback is never
                // reported as evidence that two neural blocks executed.
                std::cout<<"CONDITIONAL_REAL fixture="<<name<<" second_reached="<<eligible<<'\n';
            }
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            if(env("NINFER_TEST_CONDITIONAL_ROLLBACK_RETIREMENT")=="1") {
                auto first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                const auto second=warm->propose_conditional_second(first,16);
                auto retained=warm->conditional_predecessor_owner_for_test();
                gate(name,"rollback_retirement_owner_present",!retained.expired());
                const auto lease=warm->lease();
                warm->fail_next_rollback_for_test();
                const bool preserve_first=env("NINFER_TEST_CONDITIONAL_FIRST_FAILURE")=="1";
                if(preserve_first)warm->fail_next_conditional_verified_completion_for_test();
                bool failed=false;
                try{if(preserve_first)(void)warm->verify_conditional_second(first,second);else warm->abort();}
                catch(const std::runtime_error& error){failed=std::string(error.what())==
                    (preserve_first?"injected conditional failure after verified completion":
                     "injected DFlash rollback completion failure");}
                gate(name,"failed_rollback_retains_conditional_owner",failed && !retained.expired());
                for(unsigned operation:{0u,1u,2u,3u}) {
                    bool refused=false;
                    try {
                        if(operation==0)(void)warm->publish(coordinator,first);
                        else if(operation==1)warm->release();
                        else if(operation==2)warm->acquire(lease);
                        else warm->abort();
                    } catch(const std::logic_error&) {refused=true;}
                    gate(name,"poisoned_lane_refuses_publication_or_recycle",refused);
                    gate(name,"poisoned_refusal_preserves_conditional_owner",!retained.expired());
                    gate(name,"poisoned_refusal_preserves_root",warm->lease().root==lease.root);
                }
                warm.reset();
                gate(name,"drained_lane_releases_conditional_owner",retained.expired());
                coordinator.cancel(lease.ticket,true);
                return;
            }
            warm->fail_next_conditional_retention_for_test();
            auto retention_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
            const auto retention_calls=warm->stats().proposal_calls;
            bool allocation_refused=false;
            try{(void)warm->propose_conditional_second(retention_first,16);}
            catch(const std::bad_alloc&){allocation_refused=true;}
            gate(name,"conditional_retention_failure_before_neural",allocation_refused &&
                warm->stats().proposal_calls==retention_calls && warm->lease().root==initial);
            const auto first_only=warm->publish(coordinator,retention_first);
            gate(name,"conditional_retention_failure_preserves_first_publication",
                first_only.tokens==std::vector<std::int64_t>(expected.begin(),expected.begin()+8));
            auto retention_lease=warm->lease();warm->release();coordinator.complete(retention_lease);
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            warm->fail_next_conditional_verified_completion_for_test();
            auto failed_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
            const auto failed_second=warm->propose_conditional_second(failed_first,16);
            const auto failed_epoch=warm->execution_epoch();
            bool completion_failed=false;
            try{(void)warm->verify_conditional_second(failed_first,failed_second);}
            catch(const std::runtime_error&){completion_failed=true;}
            gate(name,"conditional_post_verification_failure_restores_original_root",completion_failed &&
                warm->execution_epoch()>failed_epoch && warm->lease().root==initial &&
                warm->context().export_exact_host_state(nullptr,true)->same_payload(*initial->state()) &&
                initial->draft_lineage_ready(draft,warm->lease().acquisition,warm->execution_epoch(),prefix) &&
                warm->verifier_horizon_policy().counters().published_rounds==0);
            bool failed_stale=false;
            try{(void)warm->verify_conditional_second(failed_first,failed_second);}
            catch(const std::invalid_argument&){failed_stale=true;}
            gate(name,"conditional_post_failure_dependency_cannot_retry",failed_stale);
            auto failure_lease=warm->lease();warm->release();coordinator.cancel(failure_lease.ticket,true);
            for(auto fault:{Coordinator::ResidencyFault::before_commit,Coordinator::ResidencyFault::after_first_lock})
            for(bool retry_publication:{false,true}) {
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                auto resident_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                const auto resident_second=warm->propose_conditional_second(resident_first,16);
                const auto resident_combined=warm->verify_conditional_second(resident_first,resident_second);
                const auto policy_before=warm->verifier_horizon_policy().counters().published_rounds;
                const auto suffix_before=warm->stats().suffix_published_rounds;
                coordinator.fail_next_residency_for_test(fault);
                bool publication_failed=false;
                try{(void)warm->publish(coordinator,resident_combined);}
                catch(const std::runtime_error&){publication_failed=true;}
                gate(name,"conditional_residency_failure_preserves_publication_authority",publication_failed &&
                    warm->lease().root==initial &&
                    warm->verifier_horizon_policy().counters().published_rounds==policy_before &&
                    warm->stats().suffix_published_rounds==suffix_before);
                if(retry_publication) {
                    const auto published=warm->publish(coordinator,resident_combined);
                    const bool policy_enabled=env("NINFER_EXL3_BOUNDED_VERIFIER_HORIZON")=="1";
                    gate(name,"conditional_residency_retry_publishes_once",
                        published.tokens==resident_combined.verification.committed_tokens &&
                        warm->verifier_horizon_policy().counters().published_rounds==policy_before+(policy_enabled?2:0));
                    bool duplicate_refused=false;
                    try{(void)warm->publish(coordinator,resident_combined);}
                    catch(const std::invalid_argument&){duplicate_refused=true;}
                    gate(name,"conditional_residency_retry_rejects_duplicate",duplicate_refused);
                    auto lease=warm->lease();warm->release();coordinator.complete(lease);
                } else {
                    warm->abort();bool stale_refused=false;
                    try{(void)warm->publish(coordinator,resident_combined);}
                    catch(const std::invalid_argument&){stale_refused=true;}
                    gate(name,"conditional_residency_failure_abort_restores_root",stale_refused &&
                        warm->context().export_exact_host_state(nullptr,true)->same_payload(*initial->state()) &&
                        initial->draft_lineage_ready(draft,warm->lease().acquisition,warm->execution_epoch(),prefix));
                    auto lease=warm->lease();warm->release();coordinator.cancel(lease.ticket,true);
                }
            }
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            // Teacher-forced first block makes the dependency deterministic;
            // block two is an actual pinned neural B8 call, not a native M16.
            auto first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
            auto old_completion=first;
            const auto before=warm->stats();
            auto missing_taps=first;missing_taps.verification.committed_taps[0].clear();
            bool missing_refused=false;
            try{(void)warm->propose_conditional_second(missing_taps,16);}
            catch(const std::invalid_argument&){missing_refused=true;}
            auto changed_state=first;changed_state.verification.committed_state=initial->state();
            bool state_refused=false;
            try{(void)warm->propose_conditional_second(changed_state,16);}
            catch(const std::invalid_argument&){state_refused=true;}
            gate(name,"conditional_invalid_predecessor_before_retention",missing_refused && state_refused &&
                warm->stats().proposal_calls==before.proposal_calls);
            for(int budget:{0,1,2,7,8,15}) {
                bool refused=false;try{(void)warm->propose_conditional_second(first,budget);}
                catch(const std::invalid_argument&){refused=true;}
                gate(name,"conditional_short_budget_skips_second_neural",refused &&
                    warm->stats().proposal_calls==before.proposal_calls && warm->lease().root==initial);
            }
            const auto second=warm->propose_conditional_second(first,16);
            gate(name,"conditional_b8_retains_completed_target_dependency",
                second.tokens.size()==8 && second.position==prefix+8 &&
                second.conditioning_root==first.root && second.epoch==warm->execution_epoch() &&
                second.costs.neural_rows==8 && second.predecessor_costs.verification_ms==first.costs.verification_ms &&
                warm->stats().conditional_second_calls==before.conditional_second_calls+1 &&
                warm->stats().neural_input_rows==before.neural_input_rows+8 && warm->lease().root==initial);
            bool refused=false;try{(void)warm->propose_conditional_second(first,16);}
            catch(const std::invalid_argument&){refused=true;}
            gate(name,"conditional_b8_once_per_predecessor",refused);
            refused=false;try{warm->publish(coordinator,old_completion);}
            catch(const std::invalid_argument&){refused=true;}
            gate(name,"conditional_b8_extends_final_use",refused);
            warm->abort();refused=false;
            try{(void)warm->propose_conditional_second(first,16);}catch(const std::invalid_argument&){refused=true;}
            gate(name,"conditional_b8_abort_invalidates_predecessor",refused && warm->lease().root==initial);
            auto lease=warm->lease();warm->release();coordinator.cancel(lease.ticket,true);
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            auto composed_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
            const auto composed_second=warm->propose_conditional_second(composed_first,16);
            auto changed_second=composed_second;changed_second.tokens.front()=(changed_second.tokens.front()+1)%248320;
            refused=false;try{(void)warm->verify_conditional_second(composed_first,changed_second);}
            catch(const std::invalid_argument&){refused=true;}
            gate(name,"conditional_changed_seed_refused",refused);
            const auto combined=warm->verify_conditional_second(composed_first,composed_second);
            const auto count=combined.verification.committed_tokens.size();
            gate(name,"conditional_combined_tokens_and_native_accounting",count>=9 && count<=16 &&
                std::equal(combined.verification.committed_tokens.begin(),combined.verification.committed_tokens.end(),expected.begin()) &&
                combined.verification.native_batch_hist[8]>=2 && combined.verification.native_batch_hist[16]==0 &&
                combined.root->is_child_of(*initial));
            for(int tap=0;tap<5;++tap)gate(name,"conditional_combined_taps",
                combined.verification.committed_taps[tap].size()==count*5120 &&
                std::equal(combined.verification.committed_taps[tap].begin(),combined.verification.committed_taps[tap].end(),taps[tap].begin()));
            const auto publication=warm->publish(coordinator,combined);
            gate(name,"conditional_atomic_publication",publication.tokens==combined.verification.committed_tokens &&
                publication.lease.root==combined.root);
            lease=warm->lease();warm->release();coordinator.complete(lease);
            {
                auto reference=target.create_context(true);reference->restore_exact_host_state(*initial->state());
                for(std::size_t row=0;row<count;++row)reference->decode(expected[row]);
                gate(name,"conditional_full_state_reference",reference->export_exact_host_state(nullptr,true)->same_payload(*combined.root->state()));
            }
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            auto stop_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
            const auto stop_second=warm->propose_conditional_second(stop_first,16);
            const std::array<std::int64_t,1> earlier_terminal{expected.front()};
            const auto verification_before=warm->stats().verification_ms;
            refused=false;try{(void)warm->verify_conditional_second(stop_first,stop_second,earlier_terminal);}
            catch(const std::invalid_argument&){refused=true;}
            gate(name,"conditional_changed_stop_policy_refused",refused &&
                warm->stats().verification_ms==verification_before && warm->lease().root==initial);
            const std::array<std::int64_t,1> next_terminal{stop_second.tokens.front()};
            const bool occurred_earlier=std::find(expected.begin(),expected.begin()+8,next_terminal.front())!=expected.begin()+8;
            if(!occurred_earlier) {
                const auto stopped=warm->verify_conditional_second(stop_first,stop_second,next_terminal);
                gate(name,"conditional_second_seed_stop_keeps_exact_prefix",stopped.verification.stopped &&
                    stopped.verification.committed_tokens.size()==9 &&
                    std::equal(stopped.verification.committed_tokens.begin(),stopped.verification.committed_tokens.end(),expected.begin()) &&
                    stopped.root->state()->position()==prefix+9);
                warm->abort();
                refused=false;try{warm->publish(coordinator,stopped);}catch(const std::invalid_argument&){refused=true;}
                gate(name,"conditional_stopped_completion_stale_after_abort",refused &&
                    warm->context().export_exact_host_state(nullptr,true)->same_payload(*initial->state()));
            } else {
                refused=false;try{(void)warm->verify_conditional_second(stop_first,stop_second,next_terminal);}
                catch(const std::invalid_argument&){refused=true;}
                gate(name,"conditional_repeated_terminal_requires_earlier_stop",refused);
                warm->abort();
            }
            const std::array<Coordinator::Lease,1> before_control_lease{warm->lease()};
            {
                const auto committed=warm->lease().root;
                const auto unpublished=warm->apply_control(std::span<const std::int64_t>(expected.data(),1));
                gate(name,"partial_control_is_unpublished",unpublished.root!=committed && warm->lease().root==committed);
                warm->abort();
                const auto old_acquisition=warm->lease();
                warm->release();coordinator.yield(old_acquisition);
                const auto reacquired=coordinator.acquire();
                require(reacquired.has_value(),"control abort reacquisition missing");
                const auto resets_before=warm->stats();
                warm->acquire(*reacquired,&coordinator);
                const auto resets_after=warm->stats();
                const auto* preserve_option=std::getenv("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT");
                const bool preserve=preserve_option && std::string_view(preserve_option)=="1";
                gate(name,"partial_control_abort_reacquire_exact_root",
                    reacquired->acquisition!=old_acquisition.acquisition && warm->lease().root==committed &&
                    resets_after.acquired_payload_preservations+resets_after.acquired_full_resets==
                        resets_before.acquired_payload_preservations+resets_before.acquired_full_resets+1 &&
                    resets_after.acquired_draft_ring_preservations+resets_after.acquired_draft_ring_restores==
                        resets_before.acquired_draft_ring_preservations+resets_before.acquired_draft_ring_restores+1 &&
                    (preserve?
                        resets_after.acquired_payload_preservations==resets_before.acquired_payload_preservations+1 &&
                        resets_after.acquired_draft_ring_preservations==resets_before.acquired_draft_ring_preservations+1:
                        resets_after.acquired_full_resets==resets_before.acquired_full_resets+1 &&
                        resets_after.acquired_draft_ring_restores==resets_before.acquired_draft_ring_restores+1) &&
                    warm->context().export_exact_host_state(nullptr,true)->same_payload(*committed->state()));
                const auto reacquired_ring=draft.export_host_ring(nullptr,false);
                gate(name,"partial_control_abort_reacquire_draft_conditioning",
                    reacquired_ring && committed->matches_detached_conditioning_for_test(*reacquired_ring));
                bool stale_control_refused=false;
                try{warm->publish(coordinator,unpublished);}catch(const std::invalid_argument&){stale_control_refused=true;}
                gate(name,"partial_control_cannot_publish_after_reacquisition",stale_control_refused);
            }
            const std::array<Coordinator::Lease,1> current_control_lease{warm->lease()};
            gate(name,"shared_offer_current_before_control",coordinator.compute_leases_current(current_control_lease));
            {
                auto saved=warm->context().request_metadata_reservation();
                struct RestoreControlReservation {
                    ninfer::exl3::Exl3TextContext& context;
                    ninfer::exl3::Exl3TextContext::SnapshotMetadataReservation saved;
                    ~RestoreControlReservation(){context.set_request_metadata_reservation(std::move(saved));}
                } restore{warm->context(),std::move(saved)};
                unsigned calls=0;std::uint64_t requested=0;
                warm->context().set_request_metadata_reservation([&](std::uint64_t bytes)->ninfer::exl3::RetainedDescriptorLedger::Ticket {
                    ++calls;requested=bytes;throw ninfer::exl3::Exl3ResourceReservationExhausted{};
                });
                const auto epoch=warm->execution_epoch();const auto position=warm->context().position();
                bool refused=false;
                try{(void)warm->apply_control_owned(std::span<const std::int64_t>(expected.data(),1));}
                catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
                gate(name,"owned_control_refusal_before_append",refused && calls==1 &&
                    requested==ninfer::exl3::bounded_shared_allocation_bytes<Lane::Pending>()+sizeof(std::int64_t) &&
                    warm->execution_epoch()==epoch && warm->context().position()==position &&
                    warm->lease().root==current_control_lease[0].root &&
                    coordinator.compute_leases_current(current_control_lease));
            }
            const auto control=warm->apply_control(std::span<const std::int64_t>(expected.data(),1));
            gate(name,"unpublished_control_keeps_committed_compute_identity",
                coordinator.compute_leases_current(current_control_lease));
            const auto control_publication=warm->publish(coordinator,control);
            const std::array<Coordinator::Lease,1> after_control_lease{warm->lease()};
            gate(name,"shared_offer_old_root_stale_after_control",
                !coordinator.compute_leases_current(before_control_lease) &&
                !coordinator.compute_leases_current(current_control_lease) &&
                coordinator.compute_leases_current(after_control_lease) &&
                after_control_lease[0].root==control_publication.lease.root &&
                after_control_lease[0].root!=current_control_lease[0].root &&
                after_control_lease[0].acquisition==current_control_lease[0].acquisition);
            refused=false;try{(void)warm->verify_conditional_second(stop_first,stop_second);}
            catch(const std::invalid_argument&){refused=true;}
            gate(name,"conditional_late_result_after_control_refused",refused &&
                warm->lease().root==control_publication.lease.root);
            lease=warm->lease();warm->release();coordinator.complete(lease);
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            const std::array<Coordinator::Lease,1> reacquired_control_lease{warm->lease()};
            gate(name,"shared_offer_reacquisition_rejects_both_prior_roots",
                !coordinator.compute_leases_current(before_control_lease) &&
                !coordinator.compute_leases_current(current_control_lease) &&
                !coordinator.compute_leases_current(after_control_lease) &&
                coordinator.compute_leases_current(reacquired_control_lease));
            refused=false;try{(void)warm->verify_conditional_second(stop_first,stop_second);}
            catch(const std::invalid_argument&){refused=true;}
            gate(name,"conditional_late_result_after_reacquisition_refused",refused && warm->lease().root==initial);
            lease=warm->lease();warm->release();coordinator.cancel(lease.ticket,true);
            for(const std::size_t kept:{1u,8u,9u,16u}) {
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                if(kept==9) {
                    // An armed but unused fault belongs to the aborted pending
                    // operation, never the next request on this physical lane.
                    const auto abandoned=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                    warm->fail_next_repair_join_for_test(abandoned);
                    warm->abort();auto cancelled_lease=warm->lease();warm->release();
                    coordinator.cancel(cancelled_lease.ticket,true);
                    coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                    auto next_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                    auto next_second=warm->propose_conditional_second(next_first,16);
                    warm->substitute_conditional_tokens_for_test(next_first,next_second,
                        std::span<const std::int64_t>(expected.data()+8,8));
                    const auto next=warm->verify_conditional_second(next_first,next_second);
                    const auto repaired_next=warm->repair_verified_prefix(next,9);
                    gate(name,"repair_join_fault_does_not_cross_abort_reacquisition",
                        repaired_next.verification.committed_tokens.size()==9);
                    warm->abort();auto completed_lease=warm->lease();warm->release();
                    coordinator.cancel(completed_lease.ticket,true);
                    coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                }
                auto first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                auto second=warm->propose_conditional_second(first,16);
                warm->substitute_conditional_tokens_for_test(first,second,
                    std::span<const std::int64_t>(expected.data()+8,8));
                const auto original=warm->verify_conditional_second(first,second);
                gate(name,"conditional_acceptance_excludes_both_seeds",
                    original.verification.accepted==16 && original.accepted_draft_rows()==14);
                if(kept>8) {
                    const auto epoch=warm->execution_epoch();
                    const auto work=warm->stats().verification_ms;
                    warm->fail_next_repair_join_for_test(original);
                    bool refused=false;
                    try{(void)warm->repair_verified_prefix(original,kept);}
                    catch(const std::bad_alloc&){refused=true;}
                    gate(name,"repair_join_refusal_before_abort_or_replay",refused &&
                        warm->execution_epoch()==epoch && warm->stats().verification_ms==work);
                    // The same Pending must remain usable for the successful retry.
                }
                const auto native_before_repair=warm->stats().completed_native_verifier_calls;
                {
                    auto saved=warm->context().request_metadata_reservation();
                    struct RestoreReservation {
                        ninfer::exl3::Exl3TextContext& context;
                        ninfer::exl3::Exl3TextContext::SnapshotMetadataReservation saved;
                        ~RestoreReservation(){context.set_request_metadata_reservation(std::move(saved));}
                    } restore{warm->context(),std::move(saved)};
                    unsigned reservations=0;
                    warm->context().set_request_metadata_reservation([&](std::uint64_t)->ninfer::exl3::RetainedDescriptorLedger::Ticket {
                        ++reservations;throw ninfer::exl3::Exl3ResourceReservationExhausted{};
                    });
                    bool refused=false;const auto epoch=warm->execution_epoch();
                    try{(void)warm->repair_verified_prefix_owned(original,kept);}
                    catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
                    gate(name,"owned_repair_peak_refusal_preserves_pending",refused && reservations==1 &&
                        warm->execution_epoch()==epoch &&
                        warm->stats().completed_native_verifier_calls==native_before_repair);
                    bool invalid=false;
                    try{(void)warm->repair_verified_prefix_owned(original,0);}
                    catch(const std::invalid_argument&){invalid=true;}
                    gate(name,"invalid_owned_repair_precedes_reservation",invalid && reservations==1);
                }
                const auto repaired=warm->repair_verified_prefix(original,kept);
                gate(name,"repair_completed_native_calls_counted_once",
                    warm->stats().completed_native_verifier_calls-native_before_repair==repaired.verification.native_invocations &&
                    repaired.verification.native_invocations>0);
                gate(name,"repair_preserves_draft_acceptance_attribution",
                    repaired.accepted_draft_rows()==(kept<=8?kept-1:7+kept-9));
                gate(name,"verified_prefix_repair_exact_tokens",repaired.proposed_rows==0 &&
                    repaired.verification.committed_tokens.size()==kept &&
                    std::equal(repaired.verification.committed_tokens.begin(),
                        repaired.verification.committed_tokens.end(),expected.begin()));
                for(std::size_t tap=0;tap<5;++tap)
                    gate(name,"verified_prefix_repair_exact_tap_prefix",
                        repaired.verification.committed_taps[tap].size()==kept*5120 &&
                        original.verification.committed_taps[tap].size()>=kept*5120 &&
                        std::equal(repaired.verification.committed_taps[tap].begin(),
                            repaired.verification.committed_taps[tap].end(),original.verification.committed_taps[tap].begin()));
                bool stale=false;try{warm->publish(coordinator,original);}
                catch(const std::invalid_argument&){stale=true;}
                gate(name,"verified_prefix_repair_invalidates_original",stale);
                auto reference=target.create_context(true);reference->restore_exact_host_state(*initial->state());
                for(std::size_t i=0;i<kept;++i)reference->decode(expected[i]);
                gate(name,"verified_prefix_repair_exact_state",
                    reference->export_exact_host_state(nullptr,true)->same_payload(*repaired.root->state()));
                const auto published=warm->publish(coordinator,repaired);
                gate(name,"verified_prefix_repair_single_publication",published.tokens.size()==kept);
                auto completed=warm->lease();warm->release();coordinator.complete(completed);
            }
            for(int mismatch=0;mismatch<8;++mismatch) {
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                auto repair_first=warm->verify(std::span<const std::int64_t>(expected.data(),8));
                auto repair_second=warm->propose_conditional_second(repair_first,16);
                std::vector<std::int64_t> forced(expected.begin()+8,expected.begin()+16);
                forced[mismatch]=(forced[mismatch]+1)%248320;
                warm->substitute_conditional_tokens_for_test(repair_first,repair_second,forced);
                const auto repaired=warm->verify_conditional_second(repair_first,repair_second);
                const auto rows=static_cast<std::size_t>(9+mismatch);
                gate(name,"conditional_exact_second_repair_boundary",repaired.verification.rejected &&
                    repaired.verification.accepted==8+static_cast<std::size_t>(mismatch) &&
                    repaired.verification.committed_tokens.size()==rows &&
                    repaired.verification.replay_rows==(env("NINFER_EXL3_REPAIR_CHECKPOINT")=="1"?
                        1:static_cast<std::size_t>(mismatch+1)) &&
                    std::equal(repaired.verification.committed_tokens.begin(),repaired.verification.committed_tokens.end(),expected.begin()) &&
                    repaired.conditional_observations && !(*repaired.conditional_observations)[1].costs.neural_rows);
                for(int tap=0;tap<5;++tap)gate(name,"conditional_second_repair_taps",
                    repaired.verification.committed_taps[tap].size()==rows*5120 &&
                    std::equal(repaired.verification.committed_taps[tap].begin(),repaired.verification.committed_taps[tap].end(),taps[tap].begin()));
                const auto accepted_draft=7u+(mismatch?static_cast<unsigned>(mismatch-1):0u);
                gate(name,"conditional_correction_is_not_accepted_draft",
                    repaired.accepted_draft_rows()==accepted_draft &&
                    !(repaired.accepted_draft_mask() & (1u<<(rows-1))));
                const auto replayed=warm->repair_verified_prefix(repaired,rows);
                const auto replayed_again=warm->repair_verified_prefix(replayed,rows);
                gate(name,"conditional_correction_attribution_survives_repeated_repair",
                    replayed.accepted_draft_mask()==repaired.accepted_draft_mask() &&
                    replayed_again.accepted_draft_mask()==repaired.accepted_draft_mask() &&
                    replayed_again.accepted_draft_rows()==accepted_draft &&
                    replayed_again.root->state()->same_payload(*repaired.root->state()));
                warm->publish(coordinator,replayed_again);
                lease=warm->lease();warm->release();coordinator.complete(lease);
                auto reference=target.create_context(true);reference->restore_exact_host_state(*initial->state());
                for(std::size_t row=0;row<rows;++row)reference->decode(expected[row]);
                gate(name,"conditional_second_repair_full_state",
                    reference->export_exact_host_state(nullptr,true)->same_payload(*repaired.root->state()));
            }
            for(int requested_stop=8;requested_stop<16;++requested_stop) {
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                const std::array<std::int64_t,1> terminal{expected[requested_stop]};
                const auto earliest=std::find(expected.begin(),expected.begin()+16,terminal.front());
                const auto rows=static_cast<std::size_t>(earliest-expected.begin()+1);
                auto terminal_first=warm->verify(std::span<const std::int64_t>(expected.data(),8),terminal);
                Lane::Pending terminal_result=terminal_first;
                if(rows>8) {
                    auto terminal_second=warm->propose_conditional_second(terminal_first,16);
                    warm->substitute_conditional_tokens_for_test(terminal_first,terminal_second,
                        std::span<const std::int64_t>(expected.data()+8,8));
                    terminal_result=warm->verify_conditional_second(terminal_first,terminal_second,terminal);
                } else {
                    const auto calls=warm->stats().proposal_calls;
                    bool refused=false;try{(void)warm->propose_conditional_second(terminal_first,16);}
                    catch(const std::invalid_argument&){refused=true;}
                    gate(name,"conditional_repeated_stop_blocks_second",refused && warm->stats().proposal_calls==calls);
                }
                gate(name,"conditional_earliest_terminal_prefix",terminal_result.verification.stopped &&
                    !terminal_result.verification.rejected && terminal_result.verification.accepted==rows &&
                    terminal_result.verification.committed_tokens.size()==rows &&
                    std::equal(terminal_result.verification.committed_tokens.begin(),terminal_result.verification.committed_tokens.end(),expected.begin()));
                warm->publish(coordinator,terminal_result);
                lease=warm->lease();warm->release();coordinator.complete(lease);
                auto reference=target.create_context(true);reference->restore_exact_host_state(*initial->state());
                for(std::size_t row=0;row<rows;++row)reference->decode(expected[row]);
                gate(name,"conditional_terminal_full_state",reference->export_exact_host_state(nullptr,true)->same_payload(*terminal_result.root->state()));
                std::cout<<"CONDITIONAL_STOP requested_row="<<requested_stop+1<<" earliest_row="<<rows<<'\n';
            }
            for(bool stop:{false,true}) {
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                std::vector<std::int64_t> proposed(expected.begin(),expected.begin()+8),terminal;
                if(stop)terminal.push_back(expected.front());
                else proposed.front()=(proposed.front()+1)%248320;
                auto blocked=warm->verify(proposed,terminal);
                const auto calls=warm->stats().proposal_calls;
                refused=false;try{(void)warm->propose_conditional_second(blocked,16);}
                catch(const std::invalid_argument&){refused=true;}
                gate(name,"conditional_b8_rejection_or_stop_blocks_neural_work",refused &&
                    warm->stats().proposal_calls==calls && warm->lease().root==initial);
                warm->abort();lease=warm->lease();warm->release();coordinator.cancel(lease.ticket,true);
            }
        }
        if(env("NINFER_EXL3_DEVICE_SEED_HANDOFF")=="1") {
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            const auto before=warm->stats();
            const auto proposed=warm->propose(8);
            const auto after=warm->stats();
            gate(name,"device_seed_actual_neural_caller",
                proposed.size()==8 && proposed.front()==expected.front() &&
                after.device_seed_handoffs==before.device_seed_handoffs+1 &&
                after.device_seed_host_fallbacks==before.device_seed_host_fallbacks);
            warm->abort();auto completed=warm->lease();warm->release();
            coordinator.cancel(completed.ticket,true);

            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            auto pending_seed=warm->context().submit_greedy_packet(
                false,warm->lease().acquisition,warm->execution_epoch());
            const auto device_seed=warm->context().device_greedy_seed(
                pending_seed,warm->lease().acquisition,warm->execution_epoch());
            std::array<std::int64_t,8> masked{};masked.fill(248070);
            gate(name,"device_seed_binds_target_projection_owner",
                device_seed.model_owner &&
                device_seed.model_identity==warm->context().model_identity() &&
                device_seed.target_embedding_bf16==warm->context().target_embedding() &&
                device_seed.target_head.trellis==
                    warm->context().target_lm_head_weights().trellis);
            auto wrong_position=device_seed;++wrong_position.position;
            bool refused=false;
            try{(void)draft.propose_cached_device_seed(masked,wrong_position,
                warm->context().position(),warm->context().target_embedding(),
                warm->context().target_lm_head_weights(),
                warm->context().target_lm_head_metadata(),248070);}
            catch(const std::runtime_error&){refused=true;}
            gate(name,"device_seed_wrong_position_preclaim_refusal",refused);
            auto wrong_mask=masked;wrong_mask[3]=17;refused=false;
            try{(void)draft.propose_cached_device_seed(wrong_mask,device_seed,
                warm->context().position(),warm->context().target_embedding(),
                warm->context().target_lm_head_weights(),
                warm->context().target_lm_head_metadata(),248070);}
            catch(const std::runtime_error&){refused=true;}
            gate(name,"device_seed_nonplaceholder_preclaim_refusal",refused);
            auto wrong_projection=device_seed;
            wrong_projection.target_embedding_bf16=nullptr;refused=false;
            try{(void)draft.propose_cached_device_seed(masked,wrong_projection,
                warm->context().position(),warm->context().target_embedding(),
                warm->context().target_lm_head_weights(),
                warm->context().target_lm_head_metadata(),248070);}
            catch(const std::runtime_error&){refused=true;}
            gate(name,"device_seed_foreign_projection_preclaim_refusal",refused);
            auto wrong_head=device_seed;++wrong_head.target_head_metadata.K;refused=false;
            try{(void)draft.propose_cached_device_seed(masked,wrong_head,
                warm->context().position(),warm->context().target_embedding(),
                warm->context().target_lm_head_weights(),
                warm->context().target_lm_head_metadata(),248070);}
            catch(const std::runtime_error&){refused=true;}
            gate(name,"device_seed_foreign_head_preclaim_refusal",refused);
            auto wrong_acquisition=device_seed;++wrong_acquisition.acquisition;refused=false;
            try{(void)draft.propose_cached_device_seed(masked,wrong_acquisition,
                warm->context().position(),warm->context().target_embedding(),
                warm->context().target_lm_head_weights(),
                warm->context().target_lm_head_metadata(),248070);}
            catch(const std::runtime_error&){refused=true;}
            gate(name,"device_seed_foreign_acquisition_preclaim_refusal",refused &&
                !device_seed.consumer_claimed->load(std::memory_order_acquire));
            auto wrong_execution=device_seed;++wrong_execution.execution;refused=false;
            try{(void)draft.propose_cached_device_seed(masked,wrong_execution,
                warm->context().position(),warm->context().target_embedding(),
                warm->context().target_lm_head_weights(),
                warm->context().target_lm_head_metadata(),248070);}
            catch(const std::runtime_error&){refused=true;}
            gate(name,"device_seed_foreign_execution_preclaim_refusal",refused &&
                !device_seed.consumer_claimed->load(std::memory_order_acquire));
            const auto direct=draft.propose_cached_device_seed(masked,device_seed,
                warm->context().position(),warm->context().target_embedding(),
                warm->context().target_lm_head_weights(),
                warm->context().target_lm_head_metadata(),248070);
            const auto packet=warm->context().finish_greedy_packet(
                std::move(pending_seed),warm->lease().acquisition,warm->execution_epoch());
            gate(name,"device_seed_direct_consumption_authority",
                direct.size()==7 && packet.decisions[0].token==expected.front());
            refused=false;
            try{(void)draft.propose_cached_device_seed(masked,device_seed,
                warm->context().position(),warm->context().target_embedding(),
                warm->context().target_lm_head_weights(),
                warm->context().target_lm_head_metadata(),248070);}
            catch(const std::runtime_error&){refused=true;}
            gate(name,"device_seed_duplicate_consumer_refused",refused);
            warm->abort();completed=warm->lease();warm->release();
            coordinator.cancel(completed.ticket,true);
        }
        // Hard caller budgets retain physical B8 by default. The explicit
        // native-horizon route instead makes the same bounded extent physical.
        for(int budget:{1,2,7,8}) {
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            const auto before=warm->stats();
            const auto proposed=warm->propose(budget);
            const auto after=warm->stats();
            const bool neural=after.proposal_calls!=before.proposal_calls;
            const auto physical_neural_rows=neural?static_cast<unsigned>(
                env("NINFER_EXL3_NATIVE_VERIFIER_HORIZON")=="1"?proposed.size():8u):0u;
            gate(name,"caller_budget_caps_proposal",!proposed.empty() &&
                proposed.size()<=static_cast<std::size_t>(budget) && proposed.front()==expected.front());
            gate(name,"caller_budget_accounts_physical_neural_block",
                after.neural_input_rows-before.neural_input_rows==physical_neural_rows &&
                after.neural_returned_suffix_rows-before.neural_returned_suffix_rows==(neural?proposed.size()-1:0u) &&
                after.neural_discarded_suffix_rows-before.neural_discarded_suffix_rows==
                    (neural?physical_neural_rows-proposed.size():0u));
            if(budget==1)gate(name,"one_row_budget_skips_all_proposers",!neural &&
                after.suffix_calls==before.suffix_calls && after.suffix_misses==before.suffix_misses);
            if(budget==1 && env("NINFER_EXL3_DEVICE_SEED_HANDOFF")=="1" &&
               env("NINFER_EXL3_SUFFIX_PROPOSALS")!="1")
                gate(name,"device_seed_scalar_remainder_uses_host_authority",
                    after.device_seed_handoffs==before.device_seed_handoffs &&
                    after.device_seed_host_fallbacks==before.device_seed_host_fallbacks+1);
            warm->abort();
            gate(name,"budget_abort_preserves_root",warm->lease().root==initial &&
                warm->context().export_exact_host_state(nullptr,true)->same_payload(*initial->state()));
            auto lease=warm->lease();warm->release();coordinator.cancel(lease.ticket,true);
        }
        // Exercise the neural caller, not merely the read-only predicate.
        // A suffix hit legitimately skips neural conditioning admission, so
        // run this matrix only on the existing neural-only configuration.
        if(env("NINFER_EXL3_SUFFIX_PROPOSALS")!="1")for(int fault=0;fault<3;++fault) {
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            const auto acquisition=warm->lease().acquisition;
            const auto epoch=warm->execution_epoch();
            std::shared_ptr<const ninfer::exl3::Exl3DraftHostRing> changed_parent;
            if(fault==0)draft.bind_ring_scope(acquisition,epoch); // missing completion witness
            if(fault==1)draft.bind_ring_scope(acquisition,epoch+1); // foreign execution
            if(fault==2)changed_parent=draft.export_host_ring(nullptr,false); // same values, new owner
            gate(name,"invalid_ring_lineage_not_admitted",
                !initial->draft_lineage_ready(draft,acquisition,epoch,prefix));
            const auto calls=warm->stats().proposal_calls;
            const auto physical=warm->stats().neural_input_rows;
            bool refused=false;
            try{(void)warm->propose(8);}catch(const std::invalid_argument&){refused=true;}
            gate(name,"invalid_ring_refused_before_neural_work",refused &&
                warm->stats().proposal_calls==calls && warm->stats().neural_input_rows==physical &&
                warm->lease().root==initial &&
                warm->verifier_horizon_policy().counters().published_rounds==0);
            warm->abort();
            gate(name,"invalid_ring_recovery_restores_authority",
                warm->context().export_exact_host_state(nullptr,true)->same_payload(*initial->state()) &&
                initial->draft_lineage_ready(draft,acquisition,warm->execution_epoch(),prefix));
            auto recovered=warm->propose(8);
            gate(name,"invalid_ring_recovery_proposes",recovered.size()>=2 &&
                warm->stats().proposal_calls==calls+1);
            warm->abort();auto lease=warm->lease();warm->release();coordinator.cancel(lease.ticket,true);
        }
        coordinator.admit(initial);warm->acquire(*coordinator.acquire());
        auto altered=warm->propose(8);altered.back()=(altered.back()+1)%248320;
        auto altered_pending=warm->verify(altered);
        gate(name,"altered_tokens_do_not_inherit_neural_cost",!altered_pending.costs.neural_rows && !altered_pending.costs.neural_ms);
        warm->abort();
        const std::array<std::int64_t,2> forced{expected[0],(expected[1]+1)%248320};auto rejected=warm->verify(forced);
        gate(name,"external_verification_has_no_neural_cost",!rejected.costs.neural_rows && !rejected.costs.neural_ms);
        gate(name,"forced_rejection_repair",rejected.verification.rejected && rejected.verification.committed_tokens==std::vector<std::int64_t>(expected.begin(),expected.begin()+2));
        warm->abort();auto lease=warm->lease();warm->release();coordinator.cancel(lease.ticket,true);
        {
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            std::atomic<bool> held{false},release_hold{false},finished{false};
            std::optional<Lane::Pending> delayed;std::exception_ptr delayed_failure;
            warm->observe_next_tap_fence_for_test([&] {
                held.store(true,std::memory_order_release);
                while(!release_hold.load(std::memory_order_acquire))std::this_thread::yield();
            });
            const std::array<std::int64_t,2> exact_pair{expected[0],expected[1]};
            std::thread verifier([&] {try{delayed=warm->verify(exact_pair);}
                catch(...){delayed_failure=std::current_exception();}
                finished.store(true,std::memory_order_release);});
            while(!held.load(std::memory_order_acquire) &&
                  !finished.load(std::memory_order_acquire))std::this_thread::yield();
            const bool held_before_fence=held.load(std::memory_order_acquire) &&
                warm->tap_waiting_for_fence_for_test() && !delayed.has_value();
            release_hold.store(true,std::memory_order_release);verifier.join();
            gate(name,"delayed_tap_copy_not_ready_before_completion_fence",held_before_fence);
            if(delayed_failure)std::rethrow_exception(delayed_failure);
            gate(name,"released_tap_copy_binds_completion",
                delayed && delayed->verification.committed_tap_ready_generation==
                    delayed->completion_generation);
            const auto retained_host_taps=delayed->verification.committed_taps;
            const auto old_generation=delayed->completion_generation;
            warm->abort();
            const auto replacement=warm->verify(exact_pair);
            gate(name,"cancelled_round_then_slot_reuse_isolates_taps",
                replacement.completion_generation!=old_generation &&
                delayed->verification.committed_taps==retained_host_taps &&
                replacement.verification.committed_tap_ready_generation==
                    replacement.completion_generation);
            warm->abort();auto held_lease=warm->lease();warm->release();
            coordinator.cancel(held_lease.ticket,true);
        }
        // Prepared E1/E2/E3 matrix: exact same B8 proposals and terminal
        // boundaries, every rejection point, host control versus device route.
        // Independent controls remain the scalar oracle above and full ring
        // export, never a packet compared only with itself.
        const auto saved_greedy=env("NINFER_EXL3_DEVICE_GREEDY");
        const auto saved_d2d=env("NINFER_EXL3_COMMITTED_TAP_D2D");
        const auto saved_device_taps_only=env("NINFER_EXL3_COMPACT_DEVICE_TAPS_ONLY");
        const bool device_taps_only_matrix=
            env("NINFER_TEST_DEVICE_TAPS_ONLY_MATRIX")=="1";
        const auto valid_tap_segments=[](const ninfer::exl3::Exl3OuterReferenceResult& value) {
            int destination=0;
            const int logical_base=value.committed_state->position()-
                static_cast<int>(value.committed_tokens.size());
            for(const auto& segment:value.committed_tap_segments) {
                if(segment.rows<1 || segment.destination_first!=destination ||
                   segment.logical_first!=logical_base+destination ||
                   segment.logical_first<segment.attempt_first ||
                   segment.logical_first+segment.rows>segment.attempt_first+segment.attempt_rows ||
                   segment.logical_first!=segment.source_partition_first+segment.source_first ||
                   segment.source_first+segment.rows>segment.source_partition_rows)
                    return false;
                destination+=segment.rows;
            }
            return destination==static_cast<int>(value.committed_tokens.size());
        };
        warm->context().prepare_greedy_packet(); // before next coordinator admission
        for(int mismatch=0;mismatch<=8;++mismatch) for(int stop:{-1,0,3,7}) {
            std::vector<std::int64_t> proposals(expected.begin(),expected.begin()+8);
            if(mismatch<8) proposals[mismatch]=(proposals[mismatch]+1)%248320;
            std::vector<std::int64_t> terminals;
            if(stop>=0) terminals.push_back(expected[stop]);
            std::array<Lane::Pending,2> results;
            std::array<std::shared_ptr<const ninfer::exl3::Exl3DraftHostRing>,2> rings;
            for(int arm=0;arm<4;++arm) {
                const int slot=arm?1:0;
                _putenv_s("NINFER_EXL3_DEVICE_GREEDY",(arm&1)?"1":"0");
                _putenv_s("NINFER_EXL3_COMMITTED_TAP_D2D",(arm&2)?"1":"0");
                _putenv_s("NINFER_EXL3_COMPACT_DEVICE_TAPS_ONLY",
                    device_taps_only_matrix && (arm&2)?"1":"0");
                coordinator.admit(initial);warm->acquire(*coordinator.acquire());
                const auto acquisition=warm->lease().acquisition;
                const auto execution=warm->execution_epoch();
                gate(name,"verification_starts_from_completed_ring_witness",
                    initial->draft_lineage_ready(draft,acquisition,execution,prefix));
                results[slot]=warm->verify(proposals,terminals);
                gate(name,"verification_writer_replaces_old_ring_witness",
                    !initial->draft_lineage_ready(draft,acquisition,execution,prefix) &&
                    results[slot].root->draft_lineage_ready(draft,acquisition,execution,
                        results[slot].root->state()->position()));
                if(env("NINFER_EXL3_DEVICE_SEED_HANDOFF")=="1" && (arm&2))
                    gate(name,"device_seed_then_d2d_taps_publish_one_current_ring_witness",
                        results[slot].verification.committed_tap_d2d_bytes==
                            results[slot].verification.committed_tokens.size()*5*5120*2 &&
                        results[slot].root->draft_lineage_ready(draft,acquisition,execution,
                            results[slot].root->state()->position()));
                gate(name,"committed_taps_bound_to_lane_completion",
                    !results[slot].verification.committed_tap_segments.empty() &&
                    results[slot].verification.committed_tap_ready_generation==
                        results[slot].completion_generation &&
                    results[slot].verification.committed_tap_ready_event!=0);
                if(arm==0 && mismatch==0 && stop==-1) {
                    auto stale_readiness=results[slot];
                    warm->invalidate_tap_readiness_for_test(stale_readiness);
                    bool refused=false;
                    try{(void)warm->repair_verified_prefix(stale_readiness,1);}
                    catch(const std::invalid_argument&){refused=true;}
                    gate(name,"stale_tap_event_generation_refused_before_repair",
                        refused && results[slot].verification.committed_tap_ready_generation==
                            results[slot].completion_generation);
                }
                rings[slot]=draft.export_host_ring(nullptr,false);
                if(env("NINFER_EXL3_REPAIR_CHECKPOINT")=="1") {
                    const auto& checkpointed=results[slot].verification;
                    const std::size_t committed=checkpointed.committed_tokens.size();
                    gate(name,"checkpoint_attempt_captures_complete_arena",
                        checkpointed.checkpoint_captured_bytes==warm->context().transaction_bytes() &&
                        checkpointed.checkpoint_captured_bytes>0);
                    const bool checkpoint_economics_ok=
                        checkpointed.checkpoint_reconstructed_rows==
                            (checkpointed.rejected?checkpointed.accepted:
                             checkpointed.stopped&&committed<8?committed:0) &&
                        checkpointed.checkpoint_fallback_rows==
                            (checkpointed.rejected && checkpointed.accepted==0?1:0) &&
                        checkpointed.replay_rows==(checkpointed.rejected?1:0);
                    if(!checkpoint_economics_ok)
                        std::cerr<<"CHECKPOINT_ECONOMICS mismatch="<<mismatch
                            <<" stop="<<stop<<" arm="<<arm
                            <<" rejected="<<checkpointed.rejected
                            <<" stopped="<<checkpointed.stopped
                            <<" accepted="<<checkpointed.accepted
                            <<" committed="<<committed
                            <<" reconstructed="<<checkpointed.checkpoint_reconstructed_rows
                            <<" fallback="<<checkpointed.checkpoint_fallback_rows
                            <<" replay="<<checkpointed.replay_rows<<std::endl;
                    gate(name,"checkpoint_wrong_ancestor_never_retained",
                        checkpoint_economics_ok);
                    auto scalar=target.create_context(true);
                    scalar->restore_exact_host_state(*initial->state());
                    for(auto token:checkpointed.committed_tokens)scalar->decode(token);
                    gate(name,"checkpoint_repair_matches_independent_scalar_state",
                        scalar->export_exact_host_state(nullptr,true)->same_payload(
                            *checkpointed.committed_state));
                }
                warm->abort();const auto parent=warm->lease();warm->release();
                coordinator.cancel(parent.ticket,true);
                if(arm) {
                    const auto& a=results[0].verification;const auto& b=results[1].verification;
                    std::weak_ptr<const Request> old_candidate=results[1].root;
                    gate(name,"device_decision_repaired_taps",
                        a.committed_tokens==b.committed_tokens && a.accepted==b.accepted &&
                        a.rejected==b.rejected && a.stopped==b.stopped &&
                        ((device_taps_only_matrix && (arm&2))?
                            std::all_of(b.committed_taps.begin(),b.committed_taps.end(),
                                [](const auto& tap){return tap.empty();}):
                            a.committed_taps==b.committed_taps) &&
                        a.committed_tap_segments==b.committed_tap_segments &&
                        valid_tap_segments(a) && valid_tap_segments(b) &&
                        a.committed_state->same_payload(*b.committed_state) && rings[0]->same_payload(*rings[1]) &&
                            b.committed_tap_d2d_bytes==((arm&2)?b.committed_tokens.size()*5*5120*2:0));
                    if(device_taps_only_matrix && (arm&2))
                        gate(name,"device_only_taps_preserve_state_and_ring",
                            a.committed_tokens==b.committed_tokens && a.accepted==b.accepted &&
                            a.rejected==b.rejected && a.stopped==b.stopped &&
                            std::all_of(b.committed_taps.begin(),b.committed_taps.end(),
                                [](const auto& tap){return tap.empty();}) &&
                            b.committed_tap_host_export_rows_avoided==b.committed_tokens.size() &&
                            valid_tap_segments(b) &&
                            a.committed_state->same_payload(*b.committed_state) &&
                            rings[0]->same_payload(*rings[1]));
                    // Keep only control + current candidate full recurrent roots.
                    results[1]={};rings[1].reset();
                    gate(name,"composed_candidate_owner_retires_after_comparison",
                        old_candidate.expired());
                }
            }
        }
        if(env("NINFER_TEST_COMPACT_TRANSPORT_FAILURE")=="1") {
            _putenv_s("NINFER_EXL3_DEVICE_GREEDY","1");
            _putenv_s("NINFER_EXL3_COMMITTED_TAP_D2D","1");
            coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            const auto failed_lease=warm->lease();
            warm->fail_next_tap_copy_for_test(2); // one plane submitted, next fails
            warm->fail_next_rollback_for_test(); // make completion ownership uncertain
            bool copy_failed=false;
            try{(void)warm->verify(std::span<const std::int64_t>(expected.data(),8));}
            catch(const std::runtime_error& error){copy_failed=std::string_view(error.what())==
                "injected committed tap D2D copy failure";}
            gate(name,"composed_partial_tap_copy_poisoned_lane",
                copy_failed && !warm->execution_active());
            warm.reset();coordinator.cancel(failed_lease.ticket,true);

            warm=make_lane();coordinator.admit(initial);warm->acquire(*coordinator.acquire());
            const auto recovered=warm->verify(
                std::span<const std::int64_t>(expected.data(),8));
            auto scalar=target.create_context(true);
            scalar->restore_exact_host_state(*initial->state());
            for(int row=0;row<8;++row)scalar->decode(expected[row]);
            const auto scalar_state=scalar->export_exact_host_state(nullptr,true);
            gate(name,"retired_failed_owner_allows_exact_replacement",
                recovered.verification.committed_state->same_payload(*scalar_state) &&
                recovered.root->draft_lineage_ready(draft,warm->lease().acquisition,
                    warm->execution_epoch(),recovered.root->state()->position()));
            warm->abort();const auto recovered_lease=warm->lease();warm->release();
            coordinator.cancel(recovered_lease.ticket,true);
        }
        _putenv_s("NINFER_EXL3_DEVICE_GREEDY",saved_greedy.c_str());
        _putenv_s("NINFER_EXL3_COMMITTED_TAP_D2D",saved_d2d.c_str());
        _putenv_s("NINFER_EXL3_COMPACT_DEVICE_TAPS_ONLY",saved_device_taps_only.c_str());
        for(int pair=0;pair<pairs;++pair)for(int order=0;order<2;++order){
            const bool reuse=(pair%2)?order==0:order==1;
            warm.reset();cuda_check(cudaDeviceSynchronize(),"real DFlash workload starts without physical contexts");
            const auto workload_begin=Clock::now();
            std::array<Result,3> results;
            for(int request=0;request<3;++request){
                // Retire prior comparison owners inside the actual workload.
                // Keep compact scalar/token telemetry; only final root escapes.
                if(request)results[request-1].root.reset();
                results[request]=run(reuse,false,pair*3+request);
            }
            warm.reset();cuda_check(cudaDeviceSynchronize(),"real DFlash workload final teardown");
            const double workload_wall=ms(workload_begin,Clock::now());
            const bool final_exact=equal(results.back(),false);
            workloads<<name<<','<<pair<<','<<(reuse?"reused":"cold")<<",3,"<<3*outputs<<','<<workload_wall<<','<<(reuse?1:3)<<','<<final_exact<<'\n';workloads.flush();
            for(int request=0;request<3;++request){const auto& r=results[request];const bool exact=r.tokens==expected && final_exact;
            const bool candidate_telemetry_exact=
                (env("NINFER_DFLASH2_FUSED_SELECTOR")=="1" ?
                    r.selector_batched_anchor_chain_calls==0 :
                    (env("NINFER_DFLASH2_FAST_SELECTOR_BATCHED_ANCHOR_CHAIN")=="1" ?
                        r.selector_batched_anchor_chain_calls>0 :
                        r.selector_batched_anchor_chain_calls==0)) &&
                (env("NINFER_EXL3_FAST_MIA_PARITY_W1")=="1" ?
                    r.fast_mia_parity_w1_settlements>0 :
                    r.fast_mia_parity_w1_settlements==0) &&
                (env("NINFER_EXL3_HOST_KV_CPU_PROFILE")!="1" ||
                    (r.gdn_graph_replays>0 && r.gdn_graph_launch_cpu_ns>0 &&
                     (env("NINFER_EXL3_HOST_KV_FULL_LAYER_GRAPHS")!="1" ||
                      (r.full_layer_graph_replays>0 && r.full_layer_graph_launch_cpu_ns>0))));
            const bool exact_with_candidate_telemetry=exact && candidate_telemetry_exact;
            runs<<name<<','<<pair<<','<<(reuse?"reused":"cold")<<','<<r.wall<<','<<r.construct<<','<<r.acquire<<','<<r.decode<<','<<r.release<<','<<r.teardown<<','<<r.stats.proposal_calls<<','<<r.stats.proposed_rows<<','<<r.verified<<','<<r.replayed<<','<<r.accepted<<','<<r.tokens.size()<<','<<r.partitions.size()<<','<<r.first<<','<<r.stats.proposal_ms<<','<<r.stats.verification_ms<<','<<r.registry<<','<<r.bytes<<','<<r.h2d<<','<<r.d2h<<','<<r.stats.suffix_calls<<','<<r.stats.suffix_rows<<','<<r.stats.suffix_misses<<','<<r.stats.suffix_metadata_bytes<<','<<(env("NINFER_EXL3_BOUNDED_VERIFIER_HORIZON")=="1")<<','<<r.horizon<<','<<r.policy.accepted_suffix<<','<<r.policy.rejected_suffix<<','<<r.policy.censored_suffix<<','<<r.selector_batched_anchor_chain_calls<<','<<r.fast_mia_parity_w1_settlements<<','<<r.gdn_graph_replays<<','<<r.gdn_graph_launch_cpu_ns<<','<<r.full_layer_graph_replays<<','<<r.full_layer_graph_launch_cpu_ns<<','<<exact_with_candidate_telemetry<<'\n';runs.flush();gaps.flush();gate(name,"timed_exact",exact_with_candidate_telemetry);
            }
        }
    }
    warm.reset();coordinator.close();std::cout<<"REAL_DFLASH_EXECUTION PASS exact_L2=1 persistent_C1=1 greedy_only=1 suffix_enabled="
        <<(env("NINFER_EXL3_SUFFIX_PROPOSALS")=="1")<<'\n';
}
