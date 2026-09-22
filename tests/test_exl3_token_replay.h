#pragma once

void run_token_replay_qualification(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using State=ninfer::exl3::Exl3ExactHostState;
    require(source.size()>2200 && target.max_context()>=2200,"token replay fixture extent");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    std::array<std::unique_ptr<DeviceBuffer>,5> stage;
    std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {
        stage[tap]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);
        staging[tap]=static_cast<std::uint16_t*>(stage[tap]->get());
    }
    int cases=0;
    for(int prefix:{63,64,2061}) {
        const int common=prefix>64?128:16;
        const auto anchor=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),common));
        auto request=anchor->append_prompt(*exact,std::span<const std::int64_t>(source.data()+common,prefix-common),1024);
        std::vector<std::int64_t> greedy;
        for(int i=0;i<3;++i) {const auto token=sample_target(*exact);greedy.push_back(token);exact->decode(token);}
        auto accepted=request->verify(*exact,greedy);
        require(accepted.second.accepted==3 && !accepted.second.rejected,"token ledger acceptance fixture");
        request=accepted.first;
        std::vector<std::int64_t> wrong(8,0);wrong[0]=(sample_target(*exact)+1)%kVocab;
        auto rejected=request->verify(*exact,wrong);
        require(rejected.second.rejected && rejected.second.committed_tokens.size()==1,"token ledger rejection fixture");
        request=rejected.first;
        std::vector<std::int64_t> expected(source.begin(),source.begin()+prefix);
        expected.insert(expected.end(),greedy.begin(),greedy.end());
        expected.push_back(rejected.second.committed_tokens[0]);
        require(request->token_suffix()==expected && anchor->token_suffix()==std::vector<std::int64_t>(source.begin(),source.begin()+common),
            "tentative/rejected token entered immutable ledger or changed prefix");
        request->restore_draft(draft,staging);const auto digest=draft.ring_digest();
        const auto compact=request->compact_draft(draft,staging);
        for(int use_anchor:{0,1}) {
            const auto plan=use_anchor?request->replay_plan(anchor):compact->replay_plan();
            ninfer::exl3::RetainedDescriptorLedger metadata;
            exact->set_request_metadata_reservation([&](std::uint64_t bytes){return metadata.acquire(bytes);});
            const auto start=std::chrono::steady_clock::now();
            auto restored=plan.restore(*exact);
            exact->set_request_metadata_reservation({});
            std::uint64_t replay_tap_metadata=0;
            restored->visit_tap_block_owners([&](const auto& block){
                if(Request::tap_block_metadata_credit_belongs_to(block,metadata))replay_tap_metadata+=Request::tap_block_metadata_bytes();
            });
            restored->state()->visit_page_metadata_owners([&](const auto& page){
                if(ninfer::exl3::Exl3ExactKVPage::metadata_credit_belongs_to(page,metadata))
                    replay_tap_metadata+=ninfer::exl3::Exl3ExactKVPage::metadata_bytes();
            });
            require(Request::request_metadata_credit_belongs_to(restored,metadata) &&
                metadata.bytes()==restored->request_metadata_bytes()+ninfer::exl3::bounded_shared_allocation_bytes<int>()+replay_tap_metadata,
                "replay retained temporary request credit or omitted final builder credit");
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            require(restored->state()->same_payload(*request->state()) && restored->same_taps(*request) && restored->token_suffix()==expected,
                "cold replay full state/taps/tokens mismatch");
            restored->restore_draft(draft,staging);
            require(draft.ring_digest()==digest,"cold replay regrouped draft projection calls");
            std::cout << "TOKEN_REPLAY_CASE prefix=" << prefix << " checkpoint=" << plan.checkpoint_position()
                << " replay_rows=" << plan.position()-plan.checkpoint_position() << " token_bytes=" << plan.token_allocated_bytes()
                << " replay_ms=" << elapsed << std::endl;
            ++cases;
            exact->reset(); // Drop context-held page owners before final-root retirement.
            std::weak_ptr<const Request> weak=restored;restored.reset();
            require(weak.expired() && metadata.bytes()==ninfer::exl3::bounded_shared_allocation_bytes<Request>(),
                "replay request final strong retirement retained descriptor credit");
            weak.reset();require(metadata.bytes()==0,"replay request final weak credit leaked");
        }
        const auto unrelated=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),common));
        bool refused=false;try {request->replay_plan(unrelated);} catch(const std::invalid_argument&) {refused=true;}
        require(refused,"equal tokens substituted an unbound replay checkpoint");
        if(prefix==63) {
            const auto plan=request->replay_plan();bool interrupted=false;
            try {plan.restore(*exact,1024,[](int){throw std::runtime_error("injected replay interruption");});}
            catch(const std::runtime_error&) {interrupted=true;}
            require(interrupted && exact->position()==0,"failed replay left publishable partial state");
            require(plan.restore(*exact)->state()->same_payload(*request->state()),"cold replay retry authority changed");
        }
    }
    auto hot=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),65));
    std::weak_ptr<const State> old_state=hot->state();const auto cold=hot->replay_plan();
    hot.reset();exact->reset();require(old_state.expired(),"cold token plan retained evicted recurrent/KV image");
    const auto recovered=cold.restore(*exact);
    const auto independent=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),65));
    require(recovered->state()->same_payload(*independent->state()),"actual evict/restore reference mismatch");
    std::cout << "TOKEN_REPLAY PASS cases=" << cases << " ring_wrap=1 rejected_suffix=excluded checkpoint=lineage_only hot_image_released=1\n";
}
