#pragma once

void run_vericache_request_qualification(Exl3TextModel& target,
    Exl3Dflash2DraftModel& draft,const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    const int prefix=target.max_context()==4096 ? 2061 : 321;
    require(source.size()>=prefix+350,"VeriCache request fixture extent");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    const bool compact_l0=env("NINFER_TEST_VERICACHE_L0_ONLY")=="1";
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY",compact_l0?"1":"0");
    auto inner=target.create_context(true);
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    const bool host_kv=env("NINFER_TEST_VERICACHE_HOST_KV")=="1";
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV",host_kv?"1":"0");
    auto exact=target.create_context(true);
    exact->prepare_continuation(8);
    require(inner->try_enable_oscar_from_environment(),"VeriCache inner OSCAR");
    inner->prepare_transaction();inner->prepare_continuation(8);
    TapStage stage;
    std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16ULL*kHidden*2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        staging[tap]=static_cast<std::uint16_t*>(stage.bulk.back()->get());
        stage.bulk_ptrs.push_back(staging[tap]);
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    std::array<std::vector<std::int64_t>,2> histories;
    std::array<std::shared_ptr<const Request>,2> requests;
    for(int request=0;request<2;++request) {
        histories[request].assign(source.begin()+request*350,source.begin()+request*350+prefix);
        requests[request]=Request::initialize(*exact,histories[request]);
    }
    // Wrong request's warm pages must fail before changing visible L0 state.
    requests[0]->restore_inner(*inner,draft,staging);
    const auto root_logits=inner->logits_host();
    const auto wrong=ninfer::exl3::Exl3TextContext::make_turboangle_warm_pages(requests[1]->state(),64);
    bool rejected=false;
    try {requests[0]->restore_inner(*inner,draft,staging,wrong.get());}
    catch(const std::exception&) {rejected=true;}
    require(rejected && root_logits==inner->logits_host(),"wrong warm request mutation");
    double total_ms[2]{};
    std::size_t total_committed[2]{},total_outer_accepted[2]{},total_proposed[2]{},physical_rows[2]{};
    int cases=0;
    for(int round=0;round<3;++round) for(int request=0;request<2;++request) {
        const auto root=requests[request];
        std::shared_ptr<const Request> next[2];
        std::vector<std::int64_t> outputs[2];
        for(int order=0;order<2;++order) {
            const int arm=(round&1)?1-order:order;
            const auto start=std::chrono::steady_clock::now();
            const auto warm=arm ? Exl3TextContext::make_turboangle_warm_pages(root->state(),64) : nullptr;
            root->restore_inner(*inner,draft,staging,warm.get());
            const auto seed=sample_target(*inner);
            std::vector<std::int64_t> block(8,kMaskToken);block[0]=seed;
            const auto proposals=draft.propose_cached(block,inner->position(),inner->target_embedding(),
                inner->target_lm_head_weights(),inner->target_lm_head_metadata(),kMaskToken);
            const auto local=verify_pending_round_transactional(*inner,draft,stage,seed,proposals,
                inner->position(),nullptr,-1,0x7e00u,TransactionRepairMode::retain_prefix);
            std::vector<std::int64_t> tentative{seed};
            tentative.insert(tentative.end(),local.emitted.begin(),local.emitted.end());
            // Outer width is independently bounded at8. A ninth inner pending
            // token is discarded and regenerated only after authoritative repair.
            const auto generated=tentative.size();
            if(tentative.size()>8) tentative.resize(8);
            if(round==1) {
                exact->restore_exact_host_state(*root->state());
                tentative[0]=(sample_target(*exact)+1)%kVocab;
            }
            auto [updated,verified]=root->verify(*exact,tentative);
            // Repair both consumers from the newly paired authoritative state.
            const auto repaired_warm=arm ? Exl3TextContext::make_turboangle_warm_pages(updated->state(),64) : nullptr;
            updated->restore_inner(*inner,draft,staging,repaired_warm.get());
            cuda_check(cudaDeviceSynchronize(),"VeriCache completed repair");
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            total_ms[arm]+=ms;total_committed[arm]+=verified.committed_tokens.size();
            total_outer_accepted[arm]+=verified.accepted;total_proposed[arm]+=tentative.size();
            physical_rows[arm]+=verified.executed_rows;
            next[arm]=updated;outputs[arm]=verified.committed_tokens;
            require(!inner->transaction_active() && inner->position()==updated->state()->position(),
                    "VeriCache repaired inner ownership");
            // A fresh full-reference reingestion checks every retained tap as
            // well as all KV/recurrent/conv state; excluded from round timing.
            auto oracle_ids=histories[request];
            oracle_ids.insert(oracle_ids.end(),outputs[arm].begin(),outputs[arm].end());
            const auto oracle=Request::initialize(*exact,oracle_ids);
            require(updated->state()->same_payload(*oracle->state()) && updated->same_taps(*oracle),
                    "VeriCache authoritative state or rejected tap suffix mismatch");
            if(round==1) require(verified.rejected && outputs[arm].size()==1,"forced outer rejection failed");
            std::cout << "VERICACHE_REQUEST_ROUND request=" << request << " round=" << round
                << " l1=" << arm << " injected=" << (round==1) << " inner_accepted=" << local.accepted
                << " generated=" << generated << " outer_proposed=" << tentative.size()
                << " outer_accepted=" << verified.accepted << " committed=" << outputs[arm].size()
                << " outer_rows=" << verified.executed_rows << " complete_ms=" << ms
                << " l1_bytes=" << (warm?warm->payload_bytes():0) << " tap_bytes=" << updated->tap_bytes() << '\n';
            ++cases;
        }
        require(std::equal(outputs[0].begin(),outputs[0].begin()+std::min(outputs[0].size(),outputs[1].size()),
                outputs[1].begin()),"L1 changed authoritative output prefix");
        histories[request].insert(histories[request].end(),outputs[0].begin(),outputs[0].end());
        requests[request]=next[0];
    }
    for(int arm=0;arm<2;++arm) std::cout << "VERICACHE_REQUEST_TOTAL l1=" << arm
        << " committed=" << total_committed[arm] << " outer_accepted=" << total_outer_accepted[arm]
        << " proposed=" << total_proposed[arm] << " outer_rows=" << physical_rows[arm]
        << " complete_ms=" << total_ms[arm] << " committed_tps=" << total_committed[arm]*1000/total_ms[arm] << '\n';
    std::cout << "VERICACHE_REQUEST PASS cases=" << cases << " requests=2 prefix=" << prefix
        << " identity=vericache_exact_fp16_eager full_prefix_repair_passes=0 host_kv=" << host_kv
        << " compact_l0=" << compact_l0 << '\n';
}
