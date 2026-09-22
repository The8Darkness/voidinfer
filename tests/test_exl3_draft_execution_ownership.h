#pragma once
#include "exl3/leased_projection.h"
void run_draft_execution_ownership(Exl3TextModel& target,
    std::unique_ptr<Exl3Dflash2DraftModel>& source,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose){
    using namespace ninfer::exl3;
    using Lane=Exl3Dflash2Execution;using Root=std::shared_ptr<const Exl3VeriCacheRequest>;
    using Clock=std::chrono::steady_clock;
    const auto retirement_mode=env("NINFER_TEST_DRAFT_LINEAR_RETIREMENT");
    if(!retirement_mode.empty() && retirement_mode!="0") {
        require(retirement_mode=="1" || retirement_mode=="partial","draft retirement selector");
        // Terminal fixture: quarantine deliberately survives until process exit.
        // No request is launched on these pristine clones.
        require(Exl3CudaLinearWorkspace::quarantined_workspaces()==0,"draft retirement fresh process");
        RetainedDescriptorLedger metadata;
        RetainedDeviceLedger device;
        std::uint64_t expected_device=0;
        for(unsigned index=0;index<7;++index) {
            auto child=std::shared_ptr<Exl3Dflash2DraftModel>(source->create_execution());
            require(Exl3Dflash2DraftModel::attach_linear_metadata_credit(child,
                metadata.acquire(7*Exl3CudaLinearWorkspace::metadata_bytes())),"draft retirement metadata attachment");
            require(Exl3Dflash2DraftModel::attach_linear_device_credit(child,
                device.acquire(Exl3Dflash2DraftModel::linear_workspace_bytes_required())),"draft retirement device attachment");
            require(!child->fail_linear_retirement_for_test(7),"draft retirement invalid child index");
            require(child->fail_linear_retirement_for_test(index,retirement_mode=="partial"),"draft retirement child injection");
            std::weak_ptr<Exl3Dflash2DraftModel> lifetime=child;
            child.reset();
            require(lifetime.expired(),"draft wrapper releases independently of failed child");
            const auto retained=Exl3CudaLinearWorkspace::latest_retirement_for_test();
            require(retained.error!=0 && retained.transform_retained,"draft failed transform retained");
            require(retained.accumulation_retained==(retirement_mode=="1"),"draft partial accumulation release");
            expected_device+=retained.transformed_bytes+retained.accumulation_bytes;
            require(Exl3CudaLinearWorkspace::quarantined_workspaces()==index+1,"one failed child per draft");
            require(device.bytes()==expected_device,"only failed draft device extents charged");
            require(metadata.bytes()==(index+1)*Exl3CudaLinearWorkspace::metadata_bytes(),"only failed draft metadata charged");
            lifetime.reset();
            require(device.bytes()==expected_device,"weak draft release preserves failed device charge");
        }
        std::cout<<"DRAFT_LINEAR_RETIREMENT_COMPLETE numerical_coverage=0\n";
        return;
    }
    const std::filesystem::path output=env("NINFER_REAL_DFLASH_OUT");
    require(!output.empty()&&!std::filesystem::exists(output),"draft ownership receipt");std::filesystem::create_directories(output);
    std::ofstream checks(output/"checks.csv");checks<<"case,pass\n";
    auto gate=[&](const char* name,bool pass){checks<<name<<','<<pass<<'\n';checks.flush();require(pass,std::string("draft execution ownership ")+name);};
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);require(GlobalMemoryStatusEx(&memory)&&memory.ullAvailPhys>(8ULL<<30),"physical host reserve");
    constexpr int prefix=512,outputs=32;
    const auto loaded_private_requirement=source->execution_bytes_required();
    gate("loaded_private_plan_matches_construction",loaded_private_requirement!=0 &&
        loaded_private_requirement==source->execution_bytes());
    const bool simultaneous=env("NINFER_DRAFT_OWNERSHIP_STREAMS")=="1";
    const bool device_prefix=env("NINFER_EXL3_HOST_KV_DEVICE_PREFIX")=="1";
    if(device_prefix)_putenv_s("NINFER_EXL3_HOST_KV_DEVICE_PREFIX","0");
    struct Stream {cudaStream_t value=nullptr;~Stream(){if(value)cudaStreamDestroy(value);}};
    std::array<Stream,2> streams;
    if(simultaneous)for(auto& stream:streams)cuda_check(cudaStreamCreateWithFlags(&stream.value,cudaStreamNonBlocking),"private draft/L2 stream");
    std::array<std::unique_ptr<DeviceBuffer>,5> storage;std::array<std::uint16_t*,5> staging{};
    for(int t=0;t<5;++t){storage[t]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[t]=static_cast<std::uint16_t*>(storage[t]->get());}
    struct Reference{Root initial;std::shared_ptr<const Exl3ExactHostState> final;std::vector<std::int64_t> tokens;std::array<std::vector<std::uint16_t>,5> taps;};
    std::array<Reference,2> ref;
    for(int i=0;i<2;++i){const auto& input=i?prose:code;require(input.size()>=prefix,"ownership fixture extent");
        auto context=target.create_context(true);context->prepare_continuation(8);
        ref[i].initial=Exl3VeriCacheRequest::initialize(*context,std::span<const std::int64_t>(input.data(),prefix),1024)->compact_draft(*source,staging);
        for(int j=0;j<outputs;++j){auto token=sample_target(*context);ref[i].tokens.push_back(token);context->decode(token);auto taps=context->exact_tap_rows_host();for(int t=0;t<5;++t)ref[i].taps[t].insert(ref[i].taps[t].end(),taps[t].begin(),taps[t].end());}
        ref[i].final=context->export_exact_host_state();
    }
    if(device_prefix)_putenv_s("NINFER_EXL3_HOST_KV_DEVICE_PREFIX","1");
    gate("private_plan_survives_reference_ring_work",
        source->execution_bytes_required()==loaded_private_requirement &&
        source->execution_bytes()==loaded_private_requirement);
    Exl3VeriCacheServingIdentity identity{"solo-physical-draft-ownership","SC_6.00bpw_H6_V6","pinned-token-ids","exact-B8-greedy","text"};
    Exl3VeriCacheServingPrefixCache cache({2,64,2ULL<<30,8ULL<<30},identity);
    const bool reserved_clone=env("NINFER_TEST_DRAFT_RESERVED_CLONE")=="1";
    if(reserved_clone) {
        using Inventory=Exl3ResourceInventory;
        Exl3VeriCacheServingCoordinator grants(cache,{2,2,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        grants.bind_physical_resources({},Inventory::unlimited());
        bool refused=false;
        try{(void)grants.reserve_constructor_credits(1,1);}catch(const std::logic_error&){refused=true;}
        gate("constructor_grant_requires_factory_scope",refused);
        Inventory::Requirement initial;initial.configuration=0x435245444954;
        initial.add(Inventory::Domain::device,1,64);initial.add(Inventory::Domain::host_metadata,1,32);
        grants.allocate_startup_resources_growing(initial,[&](auto,auto&& extend) {
            for(const auto request:std::array<std::pair<std::uint64_t,std::uint64_t>,2>{{{0,1},{1,0}}}) {
                bool zero_refused=false;
                try{(void)grants.reserve_constructor_credits(request.first,request.second);}
                catch(const std::logic_error&){zero_refused=true;}
                gate("constructor_grant_requires_both_nonzero_domains",zero_refused);
            }
            for(const auto request:std::array<std::pair<std::uint64_t,std::uint64_t>,2>{{{65,1},{1,33}}}) {
                bool ceiling_refused=false;
                try{(void)grants.reserve_constructor_credits(request.first,request.second);}
                catch(const Exl3ResourceReservationExhausted&){ceiling_refused=true;}
                gate("constructor_grant_domain_ceiling_refused",ceiling_refused);
            }
            {
                auto first=grants.reserve_constructor_credits(64,32);
                gate("constructor_grant_exact_domain_tickets",first.device.bytes()==64 && first.metadata.bytes()==32);
            }
            bool replay_refused=false;
            try{(void)grants.reserve_constructor_credits(1,1);}catch(const Exl3ResourceReservationExhausted&){replay_refused=true;}
            gate("released_constructor_grant_cannot_replay_same_promise",replay_refused);
            auto grown=initial;grown.add(Inventory::Domain::device,1,32);grown.add(Inventory::Domain::host_metadata,1,16);
            extend(grown);
            auto second=grants.reserve_constructor_credits(32,16);
            gate("constructor_grant_uses_only_growth_remainder",second.device.bytes()==32 && second.metadata.bytes()==16);
            auto owner=std::make_shared<int>(0);Inventory actual;
            actual.add({owner,0,Inventory::Domain::device,96});
            actual.add({owner,1,Inventory::Domain::host_metadata,48});return actual;
        });
        refused=false;
        try{(void)grants.reserve_constructor_credits(1,1);}catch(const std::logic_error&){refused=true;}
        gate("constructor_grant_scope_ends_at_commit",refused);
        grants.close();
    }
    if(reserved_clone)for(auto domain:{Exl3ResourceInventory::Domain::device,Exl3ResourceInventory::Domain::host_metadata}) {
        Exl3VeriCacheServingCoordinator limited(cache,{2,2,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        auto limits=Exl3ResourceInventory::unlimited();
        limits[static_cast<unsigned>(domain)]=(domain==Exl3ResourceInventory::Domain::device?
            source->execution_bytes_required():source->execution_owner_metadata_bytes())-1;
        limited.bind_physical_resources({},limits);
        bool refused=false;
        try{(void)source->create_execution_reserved(limited);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        gate(domain==Exl3ResourceInventory::Domain::device?"clone_device_credit_refusal":"clone_metadata_credit_refusal",refused);
        limited.close();
    }
    Exl3VeriCacheServingCoordinator coordinator(cache,{2,2,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
    if(reserved_clone)coordinator.bind_physical_resources({},Exl3ResourceInventory::unlimited());
    cuda_check(cudaDeviceSynchronize(),"ownership before clone");std::size_t before_free=0,total=0,after_free=0;
    cuda_check(cudaMemGetInfo(&before_free,&total),"ownership clone free before");auto start=Clock::now();
    std::shared_ptr<Exl3Dflash2DraftModel> child=reserved_clone?source->create_execution_reserved(coordinator):
        std::shared_ptr<Exl3Dflash2DraftModel>(source->create_execution());
    cuda_check(cudaDeviceSynchronize(),"ownership clone complete");
    const auto clone_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();cuda_check(cudaMemGetInfo(&after_free,&total),"ownership clone free after");
    gate("immutable_weights_shared",source->shares_weights_with(*child)&&source->weight_bytes()==child->weight_bytes());
    gate("private_head_scratch",source->last_head_input_device_for_test()!=child->last_head_input_device_for_test()&&source->last_head_logits_device_for_test()!=child->last_head_logits_device_for_test());
    gate("child_empty_ring",child->ring_count()==0&&child->ring_base_abs()==0);
    gate("complete_execution_accounting",child->execution_bytes()>child->scratch_bytes()+child->kv_bytes()+child->ring_bytes());
    gate("complete_private_preallocation_extent",child->execution_bytes_required()==child->execution_bytes() &&
        source->execution_bytes_required()==source->execution_bytes());
    gate("clone_requirement_matches_source",child->execution_bytes_required()==source->execution_bytes_required());
    gate("private_owner_metadata_plan_matches_construction",
        source->execution_owner_metadata_bytes_required()==source->execution_owner_metadata_bytes() &&
        child->execution_owner_metadata_bytes_required()==child->execution_owner_metadata_bytes());
    gate("private_linear_preallocation_extent",child->execution_bytes()-child->scratch_bytes()-child->kv_bytes()-child->ring_bytes()==
        Exl3Dflash2DraftModel::linear_workspace_bytes_required());
    gate("draft_generic_metadata_is_disjoint_subset",child->generic_owner_metadata_bytes()>0 &&
        child->generic_owner_metadata_bytes()+7*Exl3CudaLinearWorkspace::metadata_bytes()<child->execution_owner_metadata_bytes());
    RetainedDescriptorLedger draft_metadata_credit;
    RetainedDeviceLedger draft_device_credit;
    const auto linear_metadata=7*Exl3CudaLinearWorkspace::metadata_bytes();
    const auto linear_device=Exl3Dflash2DraftModel::linear_workspace_bytes_required();
    for(const auto bytes:{linear_metadata-1,linear_metadata+1})
        gate("draft_linear_metadata_extent_refused",
            !Exl3Dflash2DraftModel::attach_linear_metadata_credit(child,draft_metadata_credit.acquire(bytes)) &&
            draft_metadata_credit.bytes()==0);
    for(const auto bytes:{linear_device-1,linear_device+1})
        gate("draft_linear_device_extent_refused",
            !Exl3Dflash2DraftModel::attach_linear_device_credit(child,draft_device_credit.acquire(bytes)) &&
            draft_device_credit.bytes()==0);
    ref[0].initial->restore_draft(*source,staging);auto ring=source->export_host_ring(nullptr,false);
    ref[0].initial->restore_draft(*child,staging);gate("cross_execution_root_restore",ring->same_payload(*child->export_host_ring(nullptr,false)));
    child->reset();gate("reset_does_not_change_parent",ring->same_payload(*source->export_host_ring(nullptr,false)));
    std::array<std::unique_ptr<Lane>,2> lanes;
    lanes[0]=std::make_unique<Lane>(target.create_context(true),*source,identity.contract(),streams[0].value);
    lanes[1]=std::make_unique<Lane>(target.create_context(true),*child,identity.contract(),streams[1].value);
    const auto scope_fault_mode=env("NINFER_TEST_DRAFT_CALLBACK_SCOPE");
    require(scope_fault_mode.empty() || scope_fault_mode=="0" || scope_fault_mode=="1" || scope_fault_mode=="same_scope",
        "unknown draft callback scope fixture");
    const bool same_scope_fault=scope_fault_mode=="same_scope";
    const bool scope_fault=scope_fault_mode=="1" || same_scope_fault;
    const auto fault_family_name=env("NINFER_TEST_DRAFT_CALLBACK_FAMILY");
    const std::array<std::pair<const char*,Exl3TargetSharedFamily>,7> fault_families{{
        {"q",Exl3TargetSharedFamily::draft_q},{"k",Exl3TargetSharedFamily::draft_k},
        {"v",Exl3TargetSharedFamily::draft_v},{"o",Exl3TargetSharedFamily::draft_o},
        {"down",Exl3TargetSharedFamily::draft_down},{"gate",Exl3TargetSharedFamily::draft_gate},
        {"up",Exl3TargetSharedFamily::draft_up}}};
    auto fault_family=Exl3TargetSharedFamily::draft_q;
    bool family_found=fault_family_name.empty();
    for(const auto& [name,family]:fault_families)if(fault_family_name==name){fault_family=family;family_found=true;}
    require(family_found && (scope_fault || fault_family_name.empty()),"draft callback family requires valid scope fixture");
    const auto fault_layer_text=env("NINFER_TEST_DRAFT_CALLBACK_LAYER");
    require(fault_layer_text.empty() || (scope_fault && fault_layer_text.size()==1 &&
        fault_layer_text[0]>='0' && fault_layer_text[0]<='4'),"draft callback layer requires0..4 and scope fixture");
    const int fault_layer=fault_layer_text.empty()?0:fault_layer_text[0]-'0';
    auto scope_fault_armed=std::make_shared<bool>(scope_fault);
    auto observed_segment=std::make_shared<Exl3DraftPrivateSegment>();
    auto observed_callbacks=std::make_shared<std::uint64_t>(0);
    auto observed_head=std::make_shared<std::array<const void*,4>>();
    auto observed_parent=std::make_shared<std::shared_ptr<const Exl3DraftHostRing>>();
    const bool gateup_fault=fault_family==Exl3TargetSharedFamily::draft_gate || fault_family==Exl3TargetSharedFamily::draft_up;
    auto observed_activation=std::make_shared<Exl3ActivationLifetime::Witness>();
    if(scope_fault)source->set_shared_q_executor([scope_fault_armed,same_scope_fault,fault_family,fault_layer,
        observed_segment,observed_callbacks,observed_head,observed_parent,observed_activation,
        proposal_lane=lanes[0].get(),draft=source.get()](const Exl3DraftSharedQContinuation& offer) {
        // Compare actual offers only within this proposal, before the injected
        // scope change. Abort/retry starts a different execution epoch.
        if(*scope_fault_armed) {
            auto* lane=proposal_lane; // Fixture invokes callbacks only inside this lane's propose.
            require(lane && lane->matches_shared_proposal(offer.segment,offer.seed),
                "actual callback differs from lane active proposal block");
            require(lane->shared_proposal_root()==lane->lease().root &&
                offer.matches_conditioning_owner(lane->shared_proposal_root()->projected_metadata_owner()),
                "ordinary draft callback lost lane proposal root ownership");
            bool reentry_refused=false;
            try{(void)lane->propose(1);}catch(const std::logic_error& error) {
                reentry_refused=std::string_view(error.what())=="draft proposal already active";
            }
            require(reentry_refused && lane->matches_shared_proposal(offer.segment,offer.seed),
                "reentrant proposal touched or replaced active draft conditioning");
            const auto epoch_before=lane->execution_epoch();
            for(unsigned operation:{0u,1u,2u}) {
                bool refused=false;
                try {
                    if(operation==0)lane->abort();
                    else if(operation==1)lane->release();
                    else {
                        const std::array<std::int64_t,1> seed{offer.seed};
                        (void)lane->verify(seed);
                    }
                }catch(const std::logic_error& error) {
                    refused=std::string_view(error.what())=="draft proposal already active";
                }
                require(refused && lane->execution_epoch()==epoch_before &&
                    lane->matches_shared_proposal(offer.segment,offer.seed),
                    "callback lane mutation changed active proposal or execution epoch");
            }
            auto changed_segment=offer.segment;
            changed_segment.tokens[0]=(changed_segment.tokens[0]+1)%248320;
            require(!lane->matches_shared_proposal(changed_segment,changed_segment.tokens[0]),
                "self-consistent wrong seed bypassed lane proposal conditioning");
            changed_segment=offer.segment;changed_segment.tokens[7]=(changed_segment.tokens[7]+1)%248320;
            require(!lane->matches_shared_proposal(changed_segment,offer.seed),
                "changed draft tail input bypassed lane proposal conditioning");
            const auto parent=draft->completed_ring_parent_for_test();
            require(parent && draft->host_ring_lineage(parent,offer.acquisition,offer.execution),
                "actual shared callback lacks completed request-bound ring parent");
            require(offer.matches_conditioning_owner(parent),"draft offer omitted actual conditioning owner");
            auto changed_owner=offer;changed_owner.conditioning_parent.reset();
            require(!changed_owner.matches_conditioning_owner(parent),"draft offer accepted absent conditioning owner");
            auto unrelated=std::make_shared<int>(31);
            changed_owner.conditioning_parent=std::shared_ptr<const Exl3DraftHostRing>(unrelated,parent.get());
            require(!changed_owner.matches_conditioning_owner(parent),
                "draft offer accepted same ring address with unrelated ownership");
            changed_owner.conditioning_parent=std::shared_ptr<const Exl3DraftHostRing>(parent,parent.get());
            require(changed_owner.matches_conditioning_owner(parent),"draft offer rejected same-owner conditioning alias");
            if(*observed_callbacks==0)*observed_parent=parent;
            else require(*observed_parent==parent,"actual shared callback replaced immutable ring parent");
            const auto& head=offer.projection.head_backing;
            require(head[0] && head[0]==offer.projection.head_identity && head[1] && head[2] && head[3],
                "actual draft callback omitted head backing");
            require(offer.projection.supported_draft_head(),"actual draft callback has unsupported head contract");
            require(same_projection_head_backing(offer.projection.head_identity,head,
                offer.projection.head_identity,head),"identical actual head lost pairing eligibility");
            int foreign_backing=0; // identity-only sentinel, never submitted or dereferenced
            for(std::size_t field=0;field<head.size();++field) {
                auto changed=head;changed[field]=&foreign_backing;
                const void* changed_identity=field==0?changed[0]:offer.projection.head_identity;
                require(!same_projection_head_backing(offer.projection.head_identity,head,changed_identity,changed) &&
                    !same_projection_head_backing(changed_identity,changed,offer.projection.head_identity,head),
                    "changed nonnull head backing paired in one arrival order");
            }
            for(int field=0;field<6;++field) {
                auto changed=offer.projection;
                if(field==0)--changed.head_metadata.in_features;
                if(field==1)--changed.head_metadata.out_features;
                if(field==2)changed.head_metadata.K=5;
                if(field==3)changed.head_metadata.mcg=true;
                if(field==4)changed.head_metadata.mul1=false;
                if(field==5)changed.head_metadata.has_bias=true;
                require(!changed.supported_draft_head(),"draft head metadata mutation retained admission");
            }
            for(std::size_t field=0;field<head.size();++field) {
                auto changed=offer.projection;changed.head_backing[field]=nullptr;
                require(!changed.supported_draft_head(),"draft head missing backing retained admission");
            }
            if(*observed_callbacks==0)*observed_head=head;
            else require(*observed_head==head,"actual draft callback changed head backing across operators");
            require(offer.segment.matches(offer.acquisition,offer.execution,offer.ring_base,
                offer.ring_count,offer.projection.position,offer.seed),"actual callback private segment scope");
            require(offer.segment.attention_view_matches(offer.ring_base,offer.ring_count,
                    offer.projection.position,8) &&
                offer.segment.ring_start_slot()==static_cast<int>(offer.ring_base&2047) &&
                offer.segment.block_mask_visible(0,7) && offer.segment.block_mask_visible(7,0) &&
                !offer.segment.block_mask_visible(-1,0) && !offer.segment.block_mask_visible(0,8),
                "actual callback omitted private full-block mask or ring view");
            for(int field=0;field<6;++field) {
                auto changed=offer.segment;
                if(field==0)++changed.acquisition;
                if(field==1)++changed.execution;
                if(field==2)++changed.ring_base;
                if(field==3)--changed.ring_count;
                if(field==4)++changed.positions[0];
                if(field==5)changed.positions[7]=changed.positions[6];
                require(!changed.matches(offer.acquisition,offer.execution,offer.ring_base,
                        offer.ring_count,offer.projection.position,offer.seed) &&
                    !changed.attention_view_matches(offer.ring_base,offer.ring_count,
                        offer.projection.position,8),
                    "swapped or offset private segment retained proposal attention scope");
            }
            auto peer_scope=offer.segment;
            ++peer_scope.acquisition;++peer_scope.execution;
            require(!peer_scope.matches(offer.acquisition,offer.execution,offer.ring_base,
                    offer.ring_count,offer.projection.position,offer.seed),
                "peer private segment aliases current request epoch");
            if(*observed_callbacks==0)*observed_segment=offer.segment;
            else require(observed_segment->matches_tokens(offer.segment.tokens) &&
                observed_segment->positions==offer.segment.positions &&
                observed_segment->acquisition==offer.acquisition && observed_segment->execution==offer.execution &&
                observed_segment->ring_base==offer.ring_base && observed_segment->ring_count==offer.ring_count,
                "actual callback changed private conditioning across family/layer");
            ++*observed_callbacks;
        }
        if(*scope_fault_armed && offer.projection.family==fault_family && offer.projection.layer==fault_layer) {
            if(fault_family==Exl3TargetSharedFamily::draft_gate || fault_family==Exl3TargetSharedFamily::draft_up) {
                require(offer.projection.activation.current(),"draft MLP callback has no live activation owner");
                *observed_activation=offer.projection.activation;
            }
            *scope_fault_armed=false;
            draft->bind_ring_scope(offer.acquisition,offer.execution+(same_scope_fault?0:1));
            require(!draft->completed_ring_parent_for_test(),"scope mutation retained completed ring witness");
        }
        return false;
    },true,true,true,gateup_fault);
    for(int i=0;i<2;++i)coordinator.admit(ref[i].initial);
    for(int i=0;i<2;++i)lanes[i]->acquire(*coordinator.acquire());
    gate("two_actual_contexts_acquired",coordinator.stats().active==2 && &lanes[0]->context()!=&lanes[1]->context());
    if(scope_fault) {
        const std::array<Exl3VeriCacheServingCoordinator::Lease,2> before_fault_leases{lanes[0]->lease(),lanes[1]->lease()};
        const auto peer_epoch=lanes[1]->execution_epoch();
        const auto peer_conditioning=child->export_host_ring(streams[1].value,false);
        bool rejected=false;
        try{lanes[0]->propose(8);}
        catch(const std::runtime_error& e){rejected=std::string(e.what()).find("shared draft proposal conditioning changed")!=std::string::npos;}
        gate("shared_callback_scope_change_refused",rejected && !*scope_fault_armed);
        gate("shared_callback_unwind_expires_lane_proposal",
            !lanes[0]->matches_shared_proposal(*observed_segment,observed_segment->tokens[0]));
        if(gateup_fault)gate("shared_callback_unwind_expires_mlp_activation",
            observed_activation->lifetime && !observed_activation->current());
        gate("shared_callback_actual_private_segment_observed",*observed_callbacks>0 && observed_segment->valid());
        std::cout<<"shared_callback_selected_family="<<(fault_family_name.empty()?"q":fault_family_name)
            <<" layer="<<fault_layer<<'\n';
        lanes[0]->abort();
        gate("shared_callback_scope_root_restored",lanes[0]->context().export_exact_host_state()->same_payload(*lanes[0]->lease().root->state()));
        const auto restored_parent=source->completed_ring_parent_for_test();
        gate("shared_callback_abort_restores_completed_parent",restored_parent &&
            source->host_ring_lineage(restored_parent,lanes[0]->lease().acquisition,lanes[0]->execution_epoch()));
        gate("shared_callback_abort_rejects_old_parent_scope",!source->host_ring_lineage(*observed_parent,
            observed_segment->acquisition,observed_segment->execution));
        gate("shared_callback_abort_preserves_conditioning_payload",restored_parent && *observed_parent &&
            restored_parent->same_payload(**observed_parent));
        gate("shared_callback_parent_observation_is_inert",source->completed_ring_parent_for_test()==restored_parent &&
            source->completed_ring_parent_for_test()==restored_parent);
        gate("shared_callback_preserves_authority",coordinator.compute_leases_current(before_fault_leases));
        gate("shared_callback_peer_epoch_preserved",lanes[1]->execution_epoch()==peer_epoch);
        gate("shared_callback_peer_state_preserved",lanes[1]->context().export_exact_host_state()->same_payload(*before_fault_leases[1].root->state()));
        gate("shared_callback_peer_ring_preserved",peer_conditioning->same_payload(*child->export_host_ring(streams[1].value,false)));
    }
    {
        auto successful_callbacks=std::make_shared<std::uint64_t>(0);
        source->set_shared_q_executor([lane=lanes[0].get(),observed_segment,successful_callbacks](const Exl3DraftSharedQContinuation& offer) {
            require(lane->matches_shared_proposal(offer.segment,offer.seed),
                "successful callback missing active lane proposal scope");
            require(lane->shared_proposal_root() &&
                offer.matches_conditioning_owner(lane->shared_proposal_root()->projected_metadata_owner()),
                "successful callback conditioning differs from lane proposal root");
            *observed_segment=offer.segment;++*successful_callbacks;return false;
        },true,true,true,gateup_fault);
        const std::array<Exl3VeriCacheServingCoordinator::Lease,2> leases{lanes[0]->lease(),lanes[1]->lease()};
        gate("compute_acquisition_snapshot",coordinator.compute_leases_current(leases));
        // Request/output ceilings may be any size even though the optional
        // adaptive horizon menu is restricted to 2/4/8.
        for(int limit=1;limit<=8;++limit) {
            const auto short_before=lanes[0]->stats(),peer_before=lanes[1]->stats();
            const auto callbacks_before=*successful_callbacks;
            const auto short_proposal=lanes[0]->propose(limit);
            gate("successful_proposal_scope_expires_before_return",
                (limit==1?*successful_callbacks==callbacks_before:*successful_callbacks>callbacks_before) &&
                !lanes[0]->matches_shared_proposal(*observed_segment,observed_segment->tokens[0]) &&
                !lanes[0]->shared_proposal_root());
            const auto full_proposal=lanes[1]->propose(8);
            const auto short_after=lanes[0]->stats(),peer_after=lanes[1]->stats();
            const auto completed_calls=short_after.proposal_calls-short_before.proposal_calls;
            gate("mixed_output_limits_conserve_physical_and_returned_rows",
                completed_calls==(limit==1?0u:1u) &&
                short_after.neural_input_rows-short_before.neural_input_rows==8*completed_calls &&
                short_after.neural_returned_suffix_rows-short_before.neural_returned_suffix_rows+
                short_after.neural_discarded_suffix_rows-short_before.neural_discarded_suffix_rows==7*completed_calls);
            gate("mixed_output_limits_keep_private_extents",short_proposal.size()==static_cast<std::size_t>(limit) &&
                full_proposal.size()==8);
            gate("mixed_output_limits_count_physical_draft_work",
                short_after.neural_input_rows-short_before.neural_input_rows==(limit==1?0u:8u) &&
                peer_after.neural_input_rows-peer_before.neural_input_rows==8);
            gate("mixed_output_limits_count_discarded_rows",
                short_after.neural_returned_suffix_rows-short_before.neural_returned_suffix_rows==static_cast<unsigned>(limit-1) &&
                short_after.neural_discarded_suffix_rows-short_before.neural_discarded_suffix_rows==static_cast<unsigned>(limit==1?0:8-limit) &&
                peer_after.neural_returned_suffix_rows-peer_before.neural_returned_suffix_rows==7 &&
                peer_after.neural_discarded_suffix_rows==peer_before.neural_discarded_suffix_rows);
            gate("mixed_output_limits_do_not_publish",coordinator.compute_leases_current(leases));
            const std::array<std::int64_t,1> terminal{short_proposal.front()};
            const auto stopped=lanes[0]->verify(short_proposal,terminal);
            gate("mixed_output_terminal_commits_only_seed",stopped.verification.stopped &&
                stopped.verification.committed_tokens==std::vector<std::int64_t>{terminal[0]} &&
                stopped.root->state()->position()==leases[0].root->state()->position()+1);
            gate("mixed_output_terminal_preserves_physical_proposal_accounting",
                lanes[0]->stats().neural_input_rows==short_after.neural_input_rows &&
                lanes[0]->stats().neural_returned_suffix_rows==short_after.neural_returned_suffix_rows &&
                lanes[0]->stats().neural_discarded_suffix_rows==short_after.neural_discarded_suffix_rows);
            gate("mixed_output_terminal_does_not_consume_peer",
                lanes[1]->stats().neural_input_rows==peer_after.neural_input_rows &&
                lanes[1]->context().export_exact_host_state()->same_payload(*leases[1].root->state()) &&
                coordinator.compute_leases_current(leases));
            lanes[0]->abort();lanes[1]->abort();
            bool stopped_stale=false;
            try{lanes[0]->publish(coordinator,stopped);}catch(const std::invalid_argument&){stopped_stale=true;}
            gate("mixed_output_terminal_cannot_publish_after_abort",stopped_stale &&
                coordinator.compute_leases_current(leases));
            gate("mixed_output_limits_abort_private_states",
                lanes[0]->context().export_exact_host_state()->same_payload(*leases[0].root->state()) &&
                lanes[1]->context().export_exact_host_state()->same_payload(*leases[1].root->state()));
        }
        std::array<Exl3ProjectionRows,2> physical;
        for(int i=0;i<2;++i) {
            auto input=std::make_shared<std::array<std::uint16_t,8>>();
            auto output=std::make_shared<std::array<std::uint16_t,8>>();
            physical[i].execution=1;physical[i].input_owner=input;physical[i].output_owner=output;
            physical[i].contract="text/fp16/descriptor-only";physical[i].position=leases[i].root->state()->position();
            physical[i].rows=1;physical[i].input_columns=physical[i].output_columns=8;
            physical[i].input_stride=physical[i].output_stride=8;
            physical[i].input=input->data();physical[i].output=output->data();
            physical[i].input_storage_elements=physical[i].output_storage_elements=8;
        }
        const auto packed=assemble_exl3_leased_projection(coordinator,leases,physical,8);
        gate("compute_descriptor_binds_authoritative_root",packed.rows()==2 &&
            packed.lanes()[0].source.root==leases[0].root && packed.lanes()[1].source.acquisition==leases[1].acquisition);
        auto bound=physical;
        for(std::size_t i=0;i<bound.size();++i)bound[i]=packed.lanes()[i].source;
        gate("compute_bound_descriptor_preserved",
            assemble_exl3_leased_projection(coordinator,leases,bound,2).rows()==2);
        for(int fault=0;fault<9;++fault) {
            auto bad=bound;
            int capacity=8;
            if(fault==0)bad[0].request=leases[1].ticket.request_id;
            if(fault==1)++bad[0].acquisition;
            if(fault==2)bad[0].root=std::make_shared<int>(1);
            if(fault==3)bad[0].model=std::make_shared<int>(1);
            if(fault==4)++bad[0].position;
            if(fault==5)capacity=1;
            if(fault==6)capacity=17;
            if(fault==7)bad[0].execution=0;
            if(fault==8)bad[0].output_owner.reset();
            bool rejected=false;
            try{assemble_exl3_leased_projection(coordinator,leases,bad,capacity);}
            catch(const std::invalid_argument&){rejected=true;}
            gate("compute_physical_authority_or_capacity_refused",rejected);
            gate("compute_refusal_preserves_leases",coordinator.compute_leases_current(leases));
        }
        bool short_refused=false;
        try{assemble_exl3_leased_projection(coordinator,std::span(leases).first(1),std::span(physical).first(1),8);}
        catch(const std::invalid_argument&){short_refused=true;}
        gate("compute_single_member_refused",short_refused);
        auto stale=leases;++stale[0].acquisition;
        gate("compute_stale_acquisition",!coordinator.compute_leases_current(stale));
        stale=leases;++stale[0].ticket.generation;
        gate("compute_changed_control_generation",!coordinator.compute_leases_current(stale));
        stale=leases;stale[0].root=leases[1].root;
        gate("compute_foreign_root_same_acquisition",!coordinator.compute_leases_current(stale));
        stale=leases;stale[0].root.reset();
        gate("compute_missing_root_same_acquisition",!coordinator.compute_leases_current(stale));
        gate("compute_single_current_offer",coordinator.compute_leases_current(std::span(leases).first(1)));
        {
            auto arriving_pair=leases;
            ++arriving_pair[1].acquisition;
            gate("compute_current_arrival_does_not_authorize_stale_peer",
                coordinator.compute_leases_current(std::span(arriving_pair).first(1)) &&
                !coordinator.compute_leases_current(arriving_pair));
            std::swap(arriving_pair[0],arriving_pair[1]);
            gate("compute_stale_waiter_refused_in_reverse_order",!coordinator.compute_leases_current(arriving_pair));
            auto reversed=leases;std::swap(reversed[0],reversed[1]);
            gate("compute_current_pair_reverse_order",coordinator.compute_leases_current(reversed));
        }
        gate("compute_refused_identity_did_not_mutate_authority",coordinator.compute_leases_current(leases));
        stale=leases;stale[1]=stale[0];
        gate("compute_duplicate_lane",!coordinator.compute_leases_current(stale));
    }
    bool refused=false;try{source->create_execution();}catch(const std::runtime_error&){refused=true;}gate("clone_busy_resource_refused",refused);
    auto peer_root=lanes[1]->lease().root;auto peer_ring=child->completed_ring_parent_for_test();
    gate("peer_read_only_ring_parent_available",bool(peer_ring));
    if(device_prefix){
        const std::array<std::int64_t,2> forced{ref[0].tokens[0],(ref[0].tokens[1]+1)%248320};
        auto rejected=lanes[0]->verify(forced);
        gate("device_prefix_forced_rejection",rejected.verification.rejected&&rejected.verification.committed_tokens==std::vector<std::int64_t>(ref[0].tokens.begin(),ref[0].tokens.begin()+2));
        lanes[0]->abort();gate("device_prefix_abort_full_root",lanes[0]->context().export_exact_host_state()->same_payload(*ref[0].initial->state()));
    }
    auto stale=lanes[0]->verify(lanes[0]->propose(8));lanes[0]->abort();
    const auto peer_ring_after=child->completed_ring_parent_for_test();
    gate("cancel_peer_full_state_unchanged",lanes[1]->context().export_exact_host_state()->same_payload(*peer_root->state())&&
        peer_ring_after && peer_ring->same_payload(*peer_ring_after));
    refused=false;try{lanes[0]->publish(coordinator,stale);}catch(const std::invalid_argument&){refused=true;}gate("cancelled_stale_publication_refused",refused);stale={};peer_root.reset();peer_ring.reset();ring.reset();
    std::array<std::vector<std::int64_t>,2> tokens;std::array<std::vector<int>,2> partitions;
    std::array<std::array<std::vector<std::uint16_t>,5>,2> taps;
    std::array<std::uint64_t,2> verified{},replayed{},calls{};
    std::array<Exl3VeriCacheServingCoordinator::Lease,2> published_leases{lanes[0]->lease(),lanes[1]->lease()};
    const bool profile=env("NINFER_DRAFT_OWNERSHIP_PROFILE")=="1";
    if(profile)cuda_check(cudaProfilerStart(),"begin C2 diagnostic trace");
    start=Clock::now();
    auto round=[&](int i){
        auto pending=lanes[i]->verify(lanes[i]->propose(static_cast<int>(std::min<std::size_t>(8,outputs-tokens[i].size()))));
        auto publication=lanes[i]->publish(coordinator,pending);++calls[i];verified[i]+=pending.verification.verification_rows;replayed[i]+=pending.verification.replay_rows;
        published_leases[i]=publication.lease;
        tokens[i].insert(tokens[i].end(),publication.tokens.begin(),publication.tokens.end());partitions[i].push_back(static_cast<int>(publication.tokens.size()));
        for(int t=0;t<5;++t)taps[i][t].insert(taps[i][t].end(),pending.verification.committed_taps[t].begin(),pending.verification.committed_taps[t].end());
    };
    if(simultaneous){
        // The draft mutex must be acquired/released on its owning CPU thread.
        // Coordinator leases remain active during this local physical handoff.
        std::array<Exl3VeriCacheServingCoordinator::Lease,2> leases{lanes[0]->lease(),lanes[1]->lease()};
        for(auto& lane:lanes)lane->release();
        int device=0;cuda_check(cudaGetDevice(&device),"concurrent device");
        std::latch launch(1),ready(2);std::atomic<bool> failed=false;std::array<std::exception_ptr,2> failures;
        std::array<std::thread,2> workers;bool parallel_stale_refused=false,launch_failed=false;
        try{for(int i=0;i<2;++i)workers[i]=std::thread([&,i]{
            launch.wait();if(launch_failed)return;
            try{cuda_check(cudaSetDevice(device),"worker device");lanes[i]->acquire(leases[i]);}
            catch(...){failures[i]=std::current_exception();failed=true;}
            ready.count_down();ready.wait();
            try{
                if(failed){if(!failures[i]){lanes[i]->release();}else lanes[i].reset();return;}
                if(i==0){auto discarded=lanes[i]->verify(lanes[i]->propose(8));lanes[i]->abort();
                    try{lanes[i]->publish(coordinator,discarded);}catch(const std::invalid_argument&){parallel_stale_refused=true;}}
                while(tokens[i].size()<outputs&&!failed)round(i);
                lanes[i]->release();
            }catch(...){failures[i]=std::current_exception();failed=true;lanes[i].reset();}
        });}catch(...){launch_failed=true;failed=true;launch.count_down();for(auto& worker:workers)if(worker.joinable())worker.join();throw;}
        launch.count_down();for(auto& worker:workers)worker.join();
        for(auto failure:failures)if(failure)std::rethrow_exception(failure);
        gate("simultaneous_abort_stale_refused",parallel_stale_refused);
        // A release clears the local lease, so restore just the final published
        // coordinator frontier below from records captured during each round.
    }else{
        while(tokens[0].size()<outputs||tokens[1].size()<outputs)
            for(int i=0;i<2;++i)if(tokens[i].size()<outputs)round(i);
    }
    const auto interleaved_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    if(profile)cuda_check(cudaProfilerStop(),"end C2 diagnostic trace");
    std::array<Root,2> final{published_leases[0].root,published_leases[1].root};
    for(int i=0;i<2;++i){gate(i?"prose_tokens_state_taps":"code_tokens_state_taps",tokens[i]==ref[i].tokens&&taps[i]==ref[i].taps&&final[i]->state()->same_payload(*ref[i].final)&&lanes[i]->stats().proposal_calls>0);
        Exl3HostResidencyProbe probe;const std::array<Root,1> roots{final[i]};Exl3VeriCacheRequest::visit_host_allocations(roots,[&](const void* p,std::size_t n){probe.add(p,n);},false);auto measured=probe.measure();
        gate(i?"prose_resident_locked":"code_resident_locked",measured.allocated_union_bytes==measured.resident_tensor_bytes&&measured.allocated_union_bytes==measured.locked_tensor_bytes);
        if(!simultaneous)lanes[i]->release();coordinator.complete(published_leases[i]);
    }
    published_leases={};
    const auto lane_bytes=lanes[0]->persistent_bytes()+lanes[1]->persistent_bytes();
    const auto exports0=lanes[0]->context().recurrent_export_stats(),exports1=lanes[1]->context().recurrent_export_stats();
    nlohmann::ordered_json history=nlohmann::ordered_json::array();
    if(device_prefix)for(int i=0;i<2;++i){const auto h=lanes[i]->context().host_kv_stats();
        if(env("NINFER_EXL3_SEGMENTED_DEVICE_PREFIX")=="1")
            gate("segmented_prefix_read_without_staging",h.device_prefix_segmented_bytes>0 &&
                h.device_prefix_segmented_bytes<=h.device_prefix_hit_bytes);
        gate("device_prefix_actual_hit_no_fallback",h.device_prefix_bytes==256ULL*1024*1024&&h.device_prefix_hit_bytes>0&&h.device_prefix_fallbacks==0);
        history.push_back(nlohmann::ordered_json{{"lane",i},{"h2d_bytes",h.h2d_bytes},{"d2h_bytes",h.d2h_bytes},{"hit_bytes",h.device_prefix_hit_bytes},{"fill_bytes",h.device_prefix_fill_bytes},{"allocated_bytes",h.device_prefix_bytes}});
    }
    lanes[0].reset();lanes[1].reset();coordinator.close();
    // Reserved clones receive their tickets from coordinator close. Ordinary
    // clones exercise the same hooks with independently observable ledgers.
    if(!reserved_clone) {
        gate("draft_linear_metadata_exact_attached",
            Exl3Dflash2DraftModel::attach_linear_metadata_credit(child,draft_metadata_credit.acquire(linear_metadata)));
        gate("draft_linear_device_exact_attached",
            Exl3Dflash2DraftModel::attach_linear_device_credit(child,draft_device_credit.acquire(linear_device)));
    }
    gate("draft_linear_metadata_duplicate_refused",
        !Exl3Dflash2DraftModel::attach_linear_metadata_credit(child,draft_metadata_credit.acquire(linear_metadata)) &&
        draft_metadata_credit.bytes()==(reserved_clone?0:linear_metadata));
    gate("draft_linear_device_duplicate_refused",
        !Exl3Dflash2DraftModel::attach_linear_device_credit(child,draft_device_credit.acquire(linear_device)) &&
        draft_device_credit.bytes()==(reserved_clone?0:linear_device));
    for(int i=0;i<2;++i){ref[i].initial->restore_draft(*child,staging);int offset=0;
        for(int rows:partitions[i]){std::array<const std::uint16_t*,5> ptr{};for(int t=0;t<5;++t){cuda_check(cudaMemcpy(staging[t],ref[i].taps[t].data()+std::size_t(offset)*5120,rows*5120ULL*2,cudaMemcpyHostToDevice),"independent child ring replay");ptr[t]=staging[t];}child->commit_prefill_block(ptr.data(),rows,prefix+offset);offset+=rows;}
        auto expected=child->export_host_ring(nullptr,false);final[i]->restore_draft(*child,staging);gate(i?"prose_independent_ring":"code_independent_ring",expected->same_payload(*child->export_host_ring(nullptr,false)));}
    // Parent teardown must not retire weights or immutable roots retained by child.
    ref[0].initial->restore_draft(*source,staging);ref[0].initial->restore_draft(*child,staging);
    std::vector<std::int64_t> block(8,248070);block[0]=ref[0].tokens[0];
    auto probe_context=target.create_context(true);auto expected=source->propose_cached(block,prefix,probe_context->target_embedding(),probe_context->target_lm_head_weights(),probe_context->target_lm_head_metadata(),248070);
    auto unwitnessed_callbacks=std::make_shared<unsigned>(0);
    child->set_shared_q_executor([unwitnessed_callbacks](const Exl3DraftSharedQContinuation&) {
        ++*unwitnessed_callbacks;return false;
    },true,true,true);
    // Scope binding invalidates completion provenance without changing ring data.
    // Raw independent propose remains supported; shared callbacks must be skipped.
    child->bind_ring_scope(71,93);
    gate("raw_draft_scope_has_no_completed_parent",!child->completed_ring_parent_for_test());
    const auto unwitnessed=child->propose_cached(block,prefix,probe_context->target_embedding(),
        probe_context->target_lm_head_weights(),probe_context->target_lm_head_metadata(),248070);
    gate("unwitnessed_ring_uses_independent_proposal",*unwitnessed_callbacks==0 && unwitnessed==expected &&
        !child->completed_ring_parent_for_test());
    child->bind_ring_scope(0,94);
    child->set_shared_q_executor({});
    const auto weight_bytes=source->weight_bytes(),execution_bytes=child->execution_bytes();source.reset();
    auto survived=child->propose_cached(block,prefix,probe_context->target_embedding(),probe_context->target_lm_head_weights(),probe_context->target_lm_head_metadata(),248070);
    gate("parent_retired_child_real_proposals_exact",survived==expected&&survived.size()==7);
    probe_context.reset();
    {
        auto owned_stream=std::make_shared<Stream>();
        cuda_check(cudaStreamCreateWithFlags(&owned_stream->value,cudaStreamNonBlocking),"retained lane stream");
        std::weak_ptr<Stream> stream_lifetime=owned_stream;
        std::weak_ptr<Exl3Dflash2DraftModel> draft_lifetime=child;
        auto retained=std::make_unique<Lane>(target.create_context(true),child,identity.contract(),owned_stream->value,owned_stream);
        child.reset();owned_stream.reset();
        gate("lane_retains_draft_and_stream",!draft_lifetime.expired()&&!stream_lifetime.expired());
        gate("lane_retains_draft_linear_credits",
            draft_metadata_credit.bytes()==(reserved_clone?0:linear_metadata) &&
            draft_device_credit.bytes()==(reserved_clone?0:linear_device));
        Exl3VeriCacheServingCoordinator retained_coordinator(cache,{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        retained_coordinator.admit(ref[0].initial);retained->acquire(*retained_coordinator.acquire());
        auto proposal=retained->propose(8);auto pending=retained->verify(proposal);
        gate("retained_lane_actual_authority",!pending.verification.committed_tokens.empty() &&
            std::equal(pending.verification.committed_tokens.begin(),pending.verification.committed_tokens.end(),ref[0].tokens.begin()));
        retained->abort();const auto lease=retained->lease();retained->release();retained_coordinator.cancel(lease.ticket,true);
        retained_coordinator.close();retained.reset();
        gate("lane_releases_backing_after_completion",draft_lifetime.expired()&&stream_lifetime.expired());
        gate("final_draft_owner_releases_linear_credits",
            draft_metadata_credit.bytes()==0 && draft_device_credit.bytes()==0);
    }
    final={};ref={};
    const auto final_pin=Exl3RecurrentPinBudget::snapshot();
    gate("all_context_and_root_pin_owners_retired",final_pin[0]==0&&final_pin[2]==0);
    std::ofstream result(output/"result.json");result<<nlohmann::ordered_json{{"status","PASS_DRAFT_SHARED_WEIGHTS_PHYSICAL2_OWNERSHIP"},{"weight_bytes_unique",weight_bytes},{"execution_bytes_each",execution_bytes},{"two_l2_lane_bytes",lane_bytes},{"clone_ms",clone_ms},{"clone_cuda_free_before",before_free},{"clone_cuda_free_after",after_free},{"decode_wall_ms",interleaved_ms},{"useful_outputs",64},{"physical_contexts",2},{"logical_admitted",2},{"gpu_stream_overlap","UNMEASURED"},{"simultaneous_submission",simultaneous},{"verified_code",verified[0]},{"verified_prose",verified[1]},{"replayed_code",replayed[0]},{"replayed_prose",replayed[1]},
        {"pin_live_after_retirement",final_pin[0]},{"pin_peak",final_pin[1]},{"pin_quarantined",final_pin[2]},{"device_prefix_history",history},
        {"code_pinned_exports",exports0.pinned_exports},{"code_pageable_fallbacks",exports0.fallbacks},
        {"prose_pinned_exports",exports1.pinned_exports},{"prose_pageable_fallbacks",exports1.fallbacks}}.dump(2);
    std::cout<<"DRAFT_SHARED_WEIGHTS_PHYSICAL2_OWNERSHIP PASS GPU_OVERLAP_UNMEASURED\n";
}


