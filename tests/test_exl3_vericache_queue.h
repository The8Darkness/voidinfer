#pragma once

void run_vericache_queue_qualification(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using State=ninfer::exl3::Exl3ExactHostState;
    using Queue=ninfer::exl3::Exl3VeriCacheQueue;
    require(target.max_context()==1024 && code.size()>2200 && prose.size()>2200,"queue fixture extent");
    const auto prefill_setting=env("NINFER_TEST_EXACT_PREFILL_ROWS");
    const int prefill_rows=prefill_setting.empty()?8:std::stoi(prefill_setting);
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");_putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    auto inner=target.create_context(true);
    require(inner->try_enable_oscar_from_environment(),"queue compact OSCAR");
    inner->prepare_transaction();inner->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");_putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    TapStage stage;std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16ULL*kHidden*2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        staging[tap]=static_cast<std::uint16_t*>(stage.bulk.back()->get());
        stage.bulk_ptrs.push_back(staging[tap]);
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    int cohorts=0,oracle_states=0,late_cancellations=0;
    const auto now=[] {return std::chrono::steady_clock::now();};
    const auto milliseconds=[](auto a,auto b) {return std::chrono::duration<double,std::milli>(b-a).count();};
    for(int shared:{0,1}) for(int concurrency:{1,2,4,8}) {
        Queue queue(concurrency);
        const auto arrival=now();
        std::shared_ptr<const Request> common;
        if(shared) common=Request::initialize(*exact,std::span<const std::int64_t>(code.data(),128),prefill_rows);
        std::vector<std::vector<std::int64_t>> histories(concurrency);
        for(int id=0;id<concurrency;++id) {
            const auto& input=(id&1)?prose:code;
            if(shared) {
                histories[id].assign(code.begin(),code.begin()+128);
                histories[id].insert(histories[id].end(),input.begin()+id*257,input.begin()+id*257+64);
                queue.add(common->append_prompt(*exact,std::span<const std::int64_t>(input.data()+id*257,64),prefill_rows));
            } else {
                histories[id].assign(input.begin()+id*257,input.begin()+id*257+192);
                queue.add(Request::initialize(*exact,histories[id],prefill_rows));
            }
        }
        const double prefill_ms=milliseconds(arrival,now());
        std::vector<std::shared_ptr<const State>> initial;
        for(int id=0;id<concurrency;++id) initial.push_back(queue.state(id)->state());
        const auto init_stats=State::storage_stats(initial);
        require(init_stats.logical_context_tokens==concurrency*192 &&
            init_stats.materialized_kv_token_rows==(shared?128+concurrency*64:concurrency*192),
            "queue history sharing/materialization mismatch");
        initial.clear();
        std::vector<int> visits(concurrency,0);
        std::vector<double> ttft(concurrency,0);
        std::vector<std::chrono::steady_clock::time_point> eligible(concurrency,now());
        double complete_ms=0,max_wait_ms=0,cancelled_ms=0;
        std::size_t committed=0,discarded_authoritative=0;
        for(int round=0;round<2;++round) for(int expected=0;expected<concurrency;++expected) {
            const auto lease=queue.acquire();
            require(lease && lease->request==expected,"round-robin fairness/order");
            const int id=static_cast<int>(lease->request);
            ++visits[id];
            const auto start=now();
            max_wait_ms=std::max(max_wait_ms,milliseconds(eligible[id],start));
            if(round==0 && id==0) {
                bool busy=false,foreign=false;
                try {queue.acquire();} catch(const std::exception&) {busy=true;}
                try {queue.publish(*lease,lease->root);} catch(const std::exception&) {foreign=true;}
                require(busy && foreign && queue.state(id)==lease->root,"queue ownership/parent guard");
            }
            // Bound L1 to the last request's64 newest rows; all its cost counts.
            const auto warm=id==concurrency-1 ? Exl3TextContext::make_turboangle_warm_pages(lease->root->state(),64) : nullptr;
            lease->root->restore_inner(*inner,draft,staging,warm.get());
            const auto seed=sample_target(*inner);
            std::vector<std::int64_t> block(8,kMaskToken);block[0]=seed;
            const auto proposals=draft.propose_cached(block,inner->position(),inner->target_embedding(),
                inner->target_lm_head_weights(),inner->target_lm_head_metadata(),kMaskToken);
            const auto local=verify_pending_round_transactional(*inner,draft,stage,seed,proposals,inner->position(),
                nullptr,-1,0x7e00u,TransactionRepairMode::retain_prefix);
            std::vector<std::int64_t> tentative{seed};
            tentative.insert(tentative.end(),local.emitted.begin(),local.emitted.end());
            if(tentative.size()>8) tentative.resize(8);
            auto [updated,verified]=lease->root->verify(*exact,tentative);
            updated->restore_inner(*inner,draft,staging);
            cuda_check(cudaDeviceSynchronize(),"queue authoritative release boundary");
            const bool cancel=concurrency>1 && round==1 && id==concurrency-1;
            if(cancel) queue.cancel(id); // completed work still must not publish
            const bool published=queue.publish(*lease,updated);
            require(published!=cancel,"late cancellation publication");
            if(published) {
                histories[id].insert(histories[id].end(),verified.committed_tokens.begin(),verified.committed_tokens.end());
                committed+=verified.committed_tokens.size();
                if(ttft[id]==0) ttft[id]=milliseconds(arrival,now());
            } else {
                require(queue.state(id)==lease->root,"cancelled request checkpoint mutated");
                discarded_authoritative+=verified.committed_tokens.size();++late_cancellations;
            }
            const double elapsed=milliseconds(start,now());
            complete_ms+=elapsed;if(cancel) cancelled_ms+=elapsed;
            eligible[id]=now();
        }
        for(int count:visits) require(count==2,"starved request");
        // Independent oracles run after the serving cohort, so their work does
        // not inflate the queue/TTFT/complete-round measurements.
        for(int id=0;id<concurrency;++id) {
            const auto oracle=Request::initialize(*exact,histories[id]);
            require(queue.state(id)->state()->same_payload(*oracle->state()) && queue.state(id)->same_taps(*oracle),
                "queued request isolation or cancellation oracle mismatch");
            ++oracle_states;
        }
        exact->restore_exact_host_state(*queue.state(0)->state()); // release oracle-only page copies
        std::vector<std::shared_ptr<const State>> images;
        std::vector<std::shared_ptr<const Request>> request_images;
        std::uint64_t active_tokens=0,cached_tokens=0;
        for(int id=0;id<concurrency;++id) {
            images.push_back(queue.state(id)->state());request_images.push_back(queue.state(id));
            (queue.active(id)?active_tokens:cached_tokens)+=queue.state(id)->state()->position();
        }
        if(common) {images.push_back(common->state());request_images.push_back(common);cached_tokens+=common->state()->position();}
        const auto stats=State::storage_stats(images);
        std::size_t free_gpu=0,total_gpu=0;cuda_check(cudaMemGetInfo(&free_gpu,&total_gpu),"queue physical GPU memory");
        PROCESS_MEMORY_COUNTERS_EX memory{};memory.cb=sizeof(memory);
        require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),sizeof(memory))!=0,
            "queue process residency measurement");
        std::cout << "VERICACHE_QUEUE_COHORT C=" << concurrency << " physical_lanes=1 shared=" << shared
            << " prefill_ms=" << prefill_ms << " max_ttft_ms=" << *std::max_element(ttft.begin(),ttft.end())
            << " max_queue_wait_ms=" << max_wait_ms << " committed=" << committed << " complete_ms=" << complete_ms
            << " committed_tps=" << committed*1000/complete_ms << " cancelled_ms=" << cancelled_ms
            << " discarded_after_cancel=" << discarded_authoritative << " active_tokens=" << active_tokens
            << " cached_tokens=" << cached_tokens << " unique_kv_rows=" << stats.materialized_kv_token_rows
            << " kv_allocated_bytes=" << stats.allocated_kv_bytes << " other_state_payload_bytes=" << stats.unique_other_payload_bytes
            << " tap_allocated_bytes=" << Request::allocated_tap_bytes(request_images)
            << " gpu_used_bytes=" << total_gpu-free_gpu << " process_working_set=" << memory.WorkingSetSize
            << " process_private_commit=" << memory.PrivateUsage << std::endl;
        for(int id=0;id<concurrency;++id) queue.cancel(id);
        require(!queue.acquire(),"cancelled queue yielded work");
        ++cohorts;
    }
    std::cout << "VERICACHE_QUEUE PASS cohorts=" << cohorts << " request_oracles=" << oracle_states
        << " late_cancellations=" << late_cancellations << " max_C=8 physical_lanes=1 prefill_rows=" << prefill_rows << '\n';
}
