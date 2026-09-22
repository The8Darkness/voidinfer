#pragma once

void run_capacity_qualification(Exl3TextModel& target,Exl3Dflash2DraftModel& draft) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using State=ninfer::exl3::Exl3ExactHostState;
    using Queue=ninfer::exl3::Exl3VeriCacheQueue;
    const int lanes=std::stoi(env("NINFER_CAPACITY_C")),length=std::stoi(env("NINFER_CAPACITY_TOKENS"));
    const int shared=std::stoi(env("NINFER_CAPACITY_SHARED"));
    const bool compact=env("NINFER_CAPACITY_COMPACT_DRAFT")=="1";
    const bool cold_oracle=env("NINFER_CAPACITY_COLD_ORACLE")=="1";
    const bool lock_resident=env("NINFER_CAPACITY_LOCK_RESIDENT")=="1";
    const bool trim_resident=env("NINFER_CAPACITY_TRIM_RESIDENT")=="1";
    const bool defer_post_restore=env("NINFER_COMPACT_DEFER_POST_RESTORE")=="1";
    const bool force_outer_rejection=env("NINFER_CAPACITY_FORCE_OUTER_REJECTION_ONCE")=="1";
    const auto setting=env("NINFER_CAPACITY_RESERVE_GIB");
    const std::uint64_t reserve=(setting.empty()?8:std::stoull(setting))*(1ULL<<30);
    require(lanes>=1 && lanes<=32 && (lanes&(lanes-1))==0 && length>0 && length+64<=target.max_context() &&
        (shared==0 || shared==1) && reserve>=(4ULL<<30),"capacity cohort/budget extent");
    require(!cold_oracle || lanes==1,"cold full-prefix oracle is bounded to C1");
    require(!trim_resident || lock_resident,"adversarial trim requires explicit resident ownership");
    const std::filesystem::path fixtures=env("NINFER_CAPACITY_DIR"),output=env("NINFER_CAPACITY_OUTPUT");
    require(!fixtures.empty() && !output.empty() && !std::filesystem::exists(output/"progress.jsonl"),"capacity evidence paths");
    std::ofstream progress(output/"progress.jsonl");require(progress.good(),"capacity progress file");
    const auto now=[] {return std::chrono::steady_clock::now();};
    const auto ms=[](auto start,auto end){return std::chrono::duration<double,std::milli>(end-start).count();};
    const auto arrival=now();
    const auto available=[&] {
        MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
        require(GlobalMemoryStatusEx(&memory)!=0,"capacity physical host memory");return memory.ullAvailPhys;
    };
    const auto check_budget=[&](std::uint64_t incremental) {
        const auto free=available();
        require(free>=reserve && incremental<=free-reserve,"capacity physical DRAM reserve exhausted");
    };
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");_putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    auto inner=target.create_context(true);require(inner->try_enable_oscar_from_environment(),"capacity original OSCAR");
    inner->prepare_transaction();inner->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");_putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    TapStage stage;std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16ULL*kHidden*2));stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        staging[tap]=static_cast<std::uint16_t*>(stage.bulk.back()->get());stage.bulk_ptrs.push_back(staging[tap]);
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    const auto terminal=load_ids((fixtures/"stop.ids").string());
    std::vector<std::vector<std::int64_t>> inputs(lanes),generated(lanes);
    for(int id=0;id<lanes;++id) {
        inputs[id]=load_ids((fixtures/("request-"+std::to_string(id)+".ids")).string());
        require(inputs[id].size()==length,"capacity actual prompt extent");
        require(!std::filesystem::exists(output/("request-"+std::to_string(id)+".generated.ids")),"preserve capacity output");
    }
    Queue queue(lanes);std::shared_ptr<const Request> common;
    std::unique_ptr<ninfer::exl3::Exl3HostResidentSet> residency;
    if(lock_resident) {
        MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
        require(GlobalMemoryStatusEx(&memory)!=0 && memory.ullTotalPhys>reserve,"resident physical capacity");
        residency=std::make_unique<ninfer::exl3::Exl3HostResidentSet>(memory.ullTotalPhys-reserve,reserve);
    }
    double resident_update_ms=0,resident_close_ms=0;
    std::uint64_t resident_new_bytes=0,resident_unlocked_bytes=0;
    const auto pin_current=[&](std::shared_ptr<const Request> incoming={}) {
        if(!residency) return;
        std::vector<std::shared_ptr<const Request>> roots;
        for(std::size_t id=0;id<queue.size();++id) roots.push_back(queue.state(id));
        if(common) roots.push_back(common);if(incoming) roots.push_back(std::move(incoming));
        ninfer::exl3::Exl3HostResidentSet::Snapshot snapshot;
        for(const auto& root:roots) snapshot.owners.push_back(root);
        Request::visit_host_allocations(roots,[&](const void* data,std::size_t bytes){snapshot.add(data,bytes);},false);
        const auto stats=residency->replace(std::move(snapshot));
        resident_update_ms+=stats.elapsed_ms;resident_new_bytes+=stats.new_locked_bytes;resident_unlocked_bytes+=stats.unlocked_bytes;
    };
    std::vector<std::int64_t> prefix;
    int current_admission=-1;
    const auto observer=[&](int position) {
        check_budget(0);
        progress << "{\"event\":\"prefill\",\"request\":" << current_admission << ",\"position\":" << position
            << ",\"available_host_bytes\":" << available() << ",\"elapsed_ms\":" << ms(arrival,now()) << "}\n" << std::flush;
    };
    if(shared) {
        prefix=load_ids((fixtures/"common.ids").string());
        require(!prefix.empty() && prefix.size()<length && prefix.size()%64==0,"capacity shared prefix extent");
        for(const auto& input:inputs) require(std::equal(prefix.begin(),prefix.end(),input.begin()),"capacity prefix identity mismatch");
        check_budget(prefix.size()*65536+(512ULL<<20));
        common=Request::initialize(*exact,prefix,1024,observer);
        pin_current();
    }
    std::vector<double> ttft(lanes,0);std::vector<bool> oracle_done(lanes,false);
    std::vector<std::chrono::steady_clock::time_point> eligible(lanes,arrival);
    double prefill_ms=ms(arrival,now()),serving_ms=0,oracle_ms=0,max_wait_ms=0;
    std::uint64_t committed=0,visible=0,outer_attempted=0,l1_calls=0,kv_h2d=0,kv_d2h=0;
    std::uint64_t inner_accepted=0,inner_attempted=0,inner_replay=0,inner_reconstruction=0;
    std::uint64_t outer_matched=0,outer_rejections=0,trimmed_inner_tokens=0;
    bool forced_outer_rejection=false;
    const auto serve=[&](const Queue::Lease& lease,int width,bool differential) {
        const int id=static_cast<int>(lease.request);const auto start=now();
        const auto transfers=exact->host_kv_stats();
        max_wait_ms=std::max(max_wait_ms,ms(eligible[id],start));
        require(generated[id].size()<64,"capacity output did not terminate within64 tokens");
        const bool use_l1=differential && id==lanes-1 && !oracle_done[id];
        const auto warm=use_l1?Exl3TextContext::make_turboangle_warm_pages(lease.root->state(),64):nullptr;
        if(warm) ++l1_calls;
        lease.root->restore_inner(*inner,draft,staging,warm.get());
        const auto seed=sample_target(*inner);std::vector<std::int64_t> block(8,kMaskToken);block[0]=seed;
        const auto proposals=draft.propose_cached(block,inner->position(),inner->target_embedding(),
            inner->target_lm_head_weights(),inner->target_lm_head_metadata(),kMaskToken);
        const auto local=verify_pending_round_transactional(*inner,draft,stage,seed,proposals,inner->position(),
            nullptr,-1,0x7e00u,TransactionRepairMode::retain_prefix);
        inner_accepted+=local.accepted;inner_attempted+=local.attempted_rows;inner_replay+=local.replay_rows;
        inner_reconstruction+=local.state_reconstruction_rows;
        std::vector<std::int64_t> tentative{seed};tentative.insert(tentative.end(),local.emitted.begin(),local.emitted.end());
        const auto produced=tentative.size();
        tentative.resize(std::min<std::size_t>({tentative.size(),static_cast<std::size_t>(width),64-generated[id].size()}));
        if(force_outer_rejection && !forced_outer_rejection && id==0) {
            tentative[0]=(tentative[0]+1)%kVocab;
            forced_outer_rejection=true;
        }
        trimmed_inner_tokens+=produced-tentative.size();
        auto [updated,result]=compact?lease.root->verify_compact(*exact,draft,staging,tentative,terminal):
            lease.root->verify(*exact,tentative,terminal);
        if(!defer_post_restore) updated->restore_inner(*inner,draft,staging);
        cuda_check(cudaDeviceSynchronize(),"capacity authoritative publication");
        // Lock the complete incoming state before publishing it; the old lease
        // remains owned/locked through its post-round scalar oracle. Obsolete
        // roots leave the resident registry on the next admission/publication.
        pin_current(updated);
        require(queue.publish(lease,updated,result.stopped),"capacity unexpected publication cancellation");
        generated[id].insert(generated[id].end(),result.committed_tokens.begin(),result.committed_tokens.end());
        committed+=result.committed_tokens.size();outer_attempted+=result.executed_rows;
        outer_matched+=result.accepted;outer_rejections+=result.rejected;
        for(auto token:result.committed_tokens) if(!ninfer::exl3::exl3_terminal_token(token,terminal)) ++visible;
        const auto released=now();serving_ms+=ms(start,released);eligible[id]=released;
        const auto transferred=exact->host_kv_stats();
        kv_h2d+=transferred.h2d_bytes-transfers.h2d_bytes;kv_d2h+=transferred.d2h_bytes-transfers.d2h_bytes;
        if(ttft[id]==0) ttft[id]=ms(arrival,released);
        progress << "{\"event\":\"release\",\"request\":" << id << ",\"position\":" << updated->state()->position()
            << ",\"tokens\":" << result.committed_tokens.size() << ",\"stopped\":" << result.stopped
            << ",\"round_ms\":" << ms(start,released) << "}\n" << std::flush;
        if(differential && !oracle_done[id]) {
            const auto audit=now();
            const auto scalar=ninfer::exl3::verify_exl3_outer_reference(*exact,*lease.root->state(),tentative,true,terminal);
            require(scalar.committed_tokens==result.committed_tokens && scalar.committed_taps==result.committed_taps &&
                scalar.committed_state->same_payload(*result.committed_state),"capacity packed/scalar full-state oracle");
            oracle_ms+=ms(audit,now());oracle_done[id]=true;
        }
    };
    const auto snapshot=[&](const char* phase) {
        std::vector<std::shared_ptr<const Request>> requests;
        std::vector<std::shared_ptr<const State>> states;
        std::uint64_t active=0,cached=0;
        for(std::size_t id=0;id<queue.size();++id) {
            requests.push_back(queue.state(id));states.push_back(queue.state(id)->state());
            (queue.active(id)?active:cached)+=queue.state(id)->state()->position();
        }
        if(common) {requests.push_back(common);states.push_back(common->state());cached+=common->state()->position();}
        const auto storage=State::storage_stats(states);const auto tokens=Request::token_storage_stats(requests);
        Exl3HostResidencyProbe probe;
        Request::visit_host_allocations(requests,[&](const void* p,std::size_t n){probe.add(p,n);},false);
        if(trim_resident) require(EmptyWorkingSet(GetCurrentProcess())!=0,"capacity own-process trim");
        const auto resident=probe.measure();
        progress << "{\"event\":\"residency\",\"phase\":\"" << phase << "\",\"C\":" << queue.size()
            << ",\"expected_bytes\":" << resident.allocated_union_bytes << ",\"resident_bytes\":" << resident.resident_tensor_bytes
            << ",\"locked_bytes\":" << resident.locked_tensor_bytes << "}\n" << std::flush;
        std::cout << "CAPACITY_RESIDENCY phase=" << phase << " C=" << queue.size()
            << " expected=" << resident.allocated_union_bytes << " resident=" << resident.resident_tensor_bytes
            << " locked=" << resident.locked_tensor_bytes << " lock_mode=" << lock_resident << " trim=" << trim_resident << std::endl;
        require(resident.resident_tensor_bytes==resident.allocated_union_bytes,"capacity payload is not fully resident in DRAM");
        require(!lock_resident || resident.locked_tensor_bytes==resident.allocated_union_bytes,"capacity payload lock ownership missing");
        std::size_t gpu_free=0,gpu_total=0;cuda_check(cudaMemGetInfo(&gpu_free,&gpu_total),"capacity GPU accounting");
        require(gpu_free>=(512ULL<<20),"capacity GPU safety reserve exhausted");check_budget(0);
        PROCESS_MEMORY_COUNTERS_EX process{};process.cb=sizeof(process);
        require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&process),sizeof(process))!=0,"capacity working set");
        std::cout << "CAPACITY_SNAPSHOT phase=" << phase << " C=" << queue.size() << " shared=" << shared
            << " active_tokens=" << active << " cached_tokens=" << cached << " longest=" << storage.longest_context
            << " shared_prefix_tokens=" << prefix.size() << " sharing_saved_kv_bytes="
            << (storage.logical_context_tokens-(common?prefix.size():0))*65536-storage.materialized_kv_bytes
            << " unique_kv_rows=" << storage.materialized_kv_token_rows << " token_records=" << tokens.materialized_token_records
            << " kv_payload_bytes=" << storage.materialized_kv_bytes << " kv_allocated_bytes=" << storage.allocated_kv_bytes
            << " state_payload_bytes=" << storage.unique_other_payload_bytes << " tap_allocated_bytes=" << Request::allocated_tap_bytes(requests)
            << " projected_draft_allocated_bytes=" << Request::allocated_projected_bytes(requests)
            << " token_allocated_bytes=" << tokens.allocated_bytes << " resident_payload_bytes=" << resident.resident_tensor_bytes
            << " resident_page_bytes=" << resident.resident_page_bytes << " locked_payload_bytes=" << resident.locked_tensor_bytes
            << " resident_manager_page_bytes=" << (residency?residency->locked_page_bytes():0)
            << " process_working_set=" << process.WorkingSetSize << " private_commit=" << process.PrivateUsage
            << " gpu_used_bytes=" << gpu_total-gpu_free << " host_available_bytes=" << available() << std::endl;
    };
    for(int id=0;id<lanes;++id) {
        current_admission=id;check_budget((length-prefix.size())*65536+(512ULL<<20));
        const auto start=now();
        {
            auto request=common?common->append_prompt(*exact,std::span<const std::int64_t>(inputs[id]).subspan(prefix.size()),1024,observer):
                Request::initialize(*exact,inputs[id],1024,observer);
            if(compact) request=request->compact_draft(draft,staging);
            pin_current(request);
            queue.add(std::move(request));
        }
        prefill_ms+=ms(start,now());
        const int count=id+1;
        if((count&(count-1))==0) {
            snapshot("admitted");
            for(int expected=0;expected<count;++expected) {
                const auto lease=queue.acquire();require(lease && lease->request==expected,"capacity round-robin order");
                serve(*lease,1,false);require(queue.active(expected),"capacity request terminated before active peak");
            }
            snapshot("active_served");
        }
    }
    while(const auto lease=queue.acquire()) serve(*lease,8,true);
    snapshot("completed");
    for(int id=0;id<lanes;++id) {
        require(queue.status(id)==Queue::Status::completed && oracle_done[id],"capacity completion/oracle missing");
        auto history=inputs[id];history.insert(history.end(),generated[id].begin(),generated[id].end());
        require(queue.state(id)->token_suffix()==history,"capacity token ledger isolation mismatch");
        std::ofstream file(output/("request-"+std::to_string(id)+".generated.ids"));
        for(auto token:generated[id]) file<<token<<'\n';file.close();require(file.good(),"capacity output file");
    }
    double cold_replay_ms=0;
    if(cold_oracle) {
        check_budget(static_cast<std::uint64_t>(queue.state(0)->state()->position())*65536+(512ULL<<20));
        const auto start=now();current_admission=0;
        const auto replay=queue.state(0)->replay_plan().restore(*exact,128,observer);
        pin_current(replay);
        require(replay->state()->same_payload(*queue.state(0)->state()) && replay->token_suffix()==queue.state(0)->token_suffix(),
            "long cold replay width128 target/token oracle mismatch");
        queue.state(0)->restore_draft(draft,staging);const auto reference=draft.export_host_ring(nullptr,false);
        replay->restore_draft(draft,staging);require(reference->same_payload(*draft.export_host_ring(nullptr,false)),
            "long cold replay changed draft conditioning");
        cold_replay_ms=ms(start,now());
        std::cout << "CAPACITY_COLD_ORACLE PASS position=" << replay->state()->position() << " prefix_width=128 replay_ms=" << cold_replay_ms << std::endl;
    }
    // Late cancellation of an independent execution lease at the long checkpoint.
    Queue cancelled(1);cancelled.add(queue.state(lanes-1));const auto lease=*cancelled.acquire();
    const auto cancel_start=now();
    const std::array<std::int64_t,1> proposed{0};
    auto attempt=compact?lease.root->verify_compact(*exact,draft,staging,proposed,terminal):lease.root->verify(*exact,proposed,terminal);
    cancelled.cancel(0);require(!cancelled.publish(lease,attempt.first,attempt.second.stopped) && cancelled.state(0)==lease.root,
        "capacity late cancellation contaminated checkpoint");
    const double cancelled_ms=ms(cancel_start,now());
    if(residency) {const auto start=now();residency->close();resident_close_ms=ms(start,now());}
    require(!force_outer_rejection || (forced_outer_rejection && outer_rejections>0),
        "capacity forced outer rejection was not repaired");
    std::cout << "CAPACITY PASS C=" << lanes << " physical_lanes=1 shared=" << shared << " prompt_tokens_each=" << length
        << " compact_draft=" << compact
        << " prefill_ms=" << prefill_ms << " max_ttft_ms=" << *std::max_element(ttft.begin(),ttft.end())
        << " max_queue_wait_ms=" << max_wait_ms << " committed=" << committed << " visible=" << visible
        << " serving_ms=" << serving_ms << " visible_tps=" << visible*1000/serving_ms << " outer_executed_rows=" << outer_attempted
        << " authoritative_layer_kv_h2d=" << kv_h2d << " authoritative_layer_kv_d2h=" << kv_d2h
        << " inner_accepted=" << inner_accepted << " inner_attempted_rows=" << inner_attempted << " inner_replay_rows=" << inner_replay
        << " inner_reconstruction_rows=" << inner_reconstruction << " outer_matched=" << outer_matched
        << " outer_rejections=" << outer_rejections << " trimmed_inner_tokens=" << trimmed_inner_tokens
        << " oracle_ms=" << oracle_ms << " cancelled_ms=" << cancelled_ms << " L1_rounds=" << l1_calls
        << " cold_replay_ms=" << cold_replay_ms
        << " resident_lock=" << lock_resident << " resident_trim=" << trim_resident
        << " resident_update_ms=" << resident_update_ms << " resident_new_bytes=" << resident_new_bytes
        << " resident_unlocked_bytes=" << resident_unlocked_bytes << " resident_close_ms=" << resident_close_ms
        << " wall_ms=" << ms(arrival,now()) << " defer_post_restore=" << defer_post_restore
        << " forced_outer_rejection=" << forced_outer_rejection
        << " admission=bounded_barrier semantic_check=external_tokenizer\n";
}
