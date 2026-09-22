#pragma once

void run_draft_host_ring_qualification(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Ring=ninfer::exl3::Exl3DraftHostRing;
    require(source.size()>4200 && target.max_context()>=4200,"draft host fixture extent");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    std::array<std::unique_ptr<DeviceBuffer>,5> stage;
    std::array<std::uint16_t*,5> staging{};std::array<const std::uint16_t*,5> pointers{};
    for(int tap=0;tap<5;++tap) {
        stage[tap]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);
        staging[tap]=static_cast<std::uint16_t*>(stage[tap]->get());pointers[tap]=staging[tap];
    }
    int cases=0;
    for(int prefix:{63,64,2061,4110}) {
        const auto root=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),prefix),1024);
        root->restore_draft(draft,staging);
        const auto parent=draft.export_host_ring();const auto root_digest=draft.ring_digest();
        require(draft.host_ring_resident(parent),"export did not bind resident witness");
        if(const auto* mode=std::getenv("NINFER_TEST_DRAFT_EXPORT_UNCERTAIN");mode && std::string_view(mode)=="1") {
            ninfer::exl3::RetainedDescriptorLedger metadata;bool failed=false;
            const auto before=Ring::uncertain_transfer_count();
            try{draft.export_host_ring(nullptr,false,[&](std::uint64_t bytes){return metadata.acquire(bytes);},true);}
            catch(const std::exception&){failed=true;}
            require(failed && Ring::uncertain_transfer_count()==before+1 && metadata.bytes()>0 &&
                draft.uncertain_source_allocations_for_test()==1 && !draft.host_ring_resident(parent),
                "submitted draft export failure lost destination credit/source marking");
            bool reset_refused=false,digest_refused=false;
            try{draft.reset();}catch(const std::exception&){reset_refused=true;}
            try{draft.ring_digest();}catch(const std::exception&){digest_refused=true;}
            require(reset_refused && digest_refused,"uncertain draft export permitted reset or diagnostic submission");
            std::cout<<"DRAFT_EXPORT_UNCERTAIN_COMPLETE retained=1 source_marked=1\n";
            return; // Isolated mode intentionally leaves quarantine for teardown.
        }
        const auto before=draft.ring_restore_stats();
        require(!draft.restore_host_ring_if_needed(parent),"resident witness failed to skip restore");
        require(draft.ring_restore_stats().skipped==before.skipped+1,"missing skipped restore counter");
        {
            ninfer::exl3::RetainedDescriptorLedger metadata;unsigned reservations=0;
            const unsigned fail_at=prefix>64?3:2;bool refused=false;
            try {draft.export_host_ring(nullptr,false,[&](std::uint64_t bytes){
                if(++reservations==fail_at)throw ninfer::exl3::Exl3ResourceReservationExhausted{};
                return metadata.acquire(bytes);
            });}catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && reservations==fail_at && metadata.bytes()==0 && draft.ring_digest()==root_digest,
                "draft export metadata refusal leaked credit or changed ring");
            require(parent->same_payload(*draft.export_host_ring()),"failed export changed parent/retry payload");
            draft.restore_host_ring(parent);
        }
        draft.bind_ring_scope(8,1);
        require(!draft.host_ring_resident(parent),"scope change retained stale witness");
        require(draft.restore_host_ring_if_needed(parent),"new acquisition skipped required restore");
        draft.bind_ring_scope(8,2);
        require(!draft.host_ring_resident(parent),"execution generation revived witness");
        draft.restore_host_ring(parent);
        auto private_draft=draft.create_execution();
        require(!private_draft->host_ring_resident(parent),"shared weights implied shared ring");
        require(parent->exported_bytes()==static_cast<std::size_t>(std::min(prefix,2047))*20480,
            "initial projected ring transfer extent");
        std::array<std::shared_ptr<const Ring>,2> children;
        for(int branch=0;branch<2;++branch) {
            std::vector<std::int64_t> tail(source.begin()+prefix,source.begin()+prefix+3);
            if(branch) tail[0]=(tail[0]+7)%kVocab;
            const auto child=root->append_prompt(*exact,tail);
            child->restore_draft(draft,staging);
            const auto reference=draft.export_host_ring(nullptr,false);
            const auto expected_digest=draft.ring_digest();
            std::vector<std::int64_t> block(8,kMaskToken);block[0]=sample_target(*exact);
            const auto expected=draft.propose_cached(block,prefix+3,exact->target_embedding(),
                exact->target_lm_head_weights(),exact->target_lm_head_metadata(),kMaskToken);
            const auto expected_retained=expected;
            const auto control_before=draft.host_control_storage_for_test();
            std::vector<std::int64_t> short_block(3,kMaskToken);short_block[0]=block[0];
            const auto short_result=draft.propose_cached(short_block,prefix+3,exact->target_embedding(),
                exact->target_lm_head_weights(),exact->target_lm_head_metadata(),kMaskToken);
            const auto control_after=draft.host_control_storage_for_test();
            require(expected==expected_retained && short_result.size()==2 &&
                    control_before.context_positions==control_after.context_positions &&
                    control_before.block_positions==control_after.block_positions &&
                    control_after.generation==control_before.generation+1 && !control_after.active,
                "draft fixed host control changed retained result or storage identity");
            struct ReentryProbe {Exl3Dflash2DraftModel* draft=nullptr;bool refused=false;};
            ReentryProbe reentry{&draft,false};
            draft.set_projection_observer_for_test([](const auto&,void* user) {
                auto& probe=*static_cast<ReentryProbe*>(user);
                try{probe.draft->require_host_control_idle_for_test();}
                catch(const std::runtime_error&){probe.refused=true;}
            },&reentry);
            const auto observed=draft.propose_cached(short_block,prefix+3,exact->target_embedding(),
                exact->target_lm_head_weights(),exact->target_lm_head_metadata(),kMaskToken);
            draft.set_projection_observer_for_test(nullptr);
            require(reentry.refused && observed==short_result,
                "draft host control reentry guard absent or changed result");
            std::vector<std::int64_t> oversized(control_after.block_capacity+1,kMaskToken);
            oversized[0]=block[0];bool control_refused=false;
            try{(void)draft.propose_cached(oversized,prefix+3,exact->target_embedding(),
                exact->target_lm_head_weights(),exact->target_lm_head_metadata(),kMaskToken);}
            catch(const std::exception&){control_refused=true;}
            const auto control_refusal=draft.host_control_storage_for_test();
            require(control_refused && expected==expected_retained && !control_refusal.active &&
                    control_refusal.context_positions==control_before.context_positions &&
                    control_refusal.block_positions==control_before.block_positions,
                "draft host control overflow grew storage or changed owning result");
            exact->restore_exact_host_state(*root->state());exact->continue_rows(tail);
            const auto taps=exact->exact_tap_rows_host();exact->finish_exact_continuation();
            require(exact->export_exact_host_state()->same_payload(*child->state()),"draft suffix target ancestor mismatch");
            draft.restore_host_ring(parent);
            for(int tap=0;tap<5;++tap)
                cuda_check(cudaMemcpy(staging[tap],taps[tap].data(),taps[tap].size()*2,cudaMemcpyHostToDevice),"draft host confirmed tap stage");
            draft.commit_prefill_block(pointers.data(),3,prefix);
            require(!draft.host_ring_resident(parent),"commit retained old witness");
            children[branch]=draft.export_host_ring();
            require(children[branch]->exported_bytes()==3*20480 && children[branch]->same_payload(*reference),
                "projected suffix/COW differs from original tap replay");
            require(children[branch]->same_payload(*draft.export_host_ring(nullptr,false)),"shared ring concealed GPU mismatch");
            draft.reset();draft.restore_host_ring(children[branch]);
            require(draft.ring_digest()==expected_digest,"projected ring restore changed logical bits");
            const auto actual=draft.propose_cached(block,prefix+3,exact->target_embedding(),
                exact->target_lm_head_weights(),exact->target_lm_head_metadata(),kMaskToken);
            require(actual==expected && draft.export_host_ring()->exported_bytes()==0,"restored proposal changed or mutated ring");
            draft.rewind_to(draft.ring_count()-2);const auto rewound_digest=draft.ring_digest();
            require(!draft.host_ring_resident(children[branch]),"rewind retained witness");
            const auto rewound=draft.export_host_ring();
            require(rewound->exported_bytes()==static_cast<std::size_t>(draft.ring_count())*20480,"rewind retained unproven host lineage");
            draft.reset();draft.restore_host_ring(rewound);require(draft.ring_digest()==rewound_digest,"rewound host restore mismatch");
            ++cases;
        }
        require(!children[0]->same_payload(*children[1]),"different ancestors aliased projected ring");
        draft.restore_host_ring(parent);require(draft.ring_digest()==root_digest,"child COW corrupted parent");
        const std::array<std::shared_ptr<const Ring>,3> family{parent,children[0],children[1]};
        const std::array<std::shared_ptr<const Ring>,1> single{parent};
        const auto bytes=Ring::visit_allocations(family);
        require(bytes<=Ring::visit_allocations(single)+4*64*20480,"projected branches copied full rings");
        bool failed=false;try {draft.commit_prefill_block(nullptr,0,prefix);} catch(const std::exception&) {failed=true;}
        bool refused=false;try {draft.export_host_ring();} catch(const std::exception&) {refused=true;}
        require(failed && refused,"failed commit published host snapshot");
        require(!draft.host_ring_resident(parent),"failed commit retained resident witness");
        draft.restore_host_ring(parent);require(draft.ring_digest()==root_digest,"known host image did not repair failure");
        draft.reset();draft.begin_fresh_prefill(prefix,prefix+3);
        refused=false;try {draft.export_host_ring();} catch(const std::exception&) {refused=true;}
        require(refused,"incomplete fresh prefill published host snapshot");draft.reset();draft.restore_host_ring(parent);
        std::cout << "DRAFT_HOST_RING_CASE prefix=" << prefix << " logical_rows=" << parent->count()
            << " parent_allocated_bytes=" << Ring::visit_allocations(single) << " family_allocated_bytes=" << bytes
            << " suffix_export_bytes=" << children[0]->exported_bytes() << std::endl;
    }
    std::cout << "DRAFT_HOST_RING PASS branches=" << cases << " ring_wraps=2 proposals=exact logical_bits=exact lineage=COW\n";
}
