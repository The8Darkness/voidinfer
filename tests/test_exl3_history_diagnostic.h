#pragma once

void run_history_diagnostic(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using namespace ninfer::exl3;
    using Root=std::shared_ptr<const Exl3VeriCacheRequest>;
    using Lane=Exl3Dflash2Execution;
    using Json=nlohmann::ordered_json;
    using Clock=std::chrono::steady_clock;
    const auto ms=[](auto start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
    const int requested=std::stoi(env("NINFER_REAL_DFLASH_PREFIX"));
    const bool profile=env("NINFER_HISTORY_PROFILE")=="1";
    constexpr int outputs=32;
    require(requested==0||requested==4096,"history diagnostic bounded prefix");
    const std::filesystem::path dir=env("NINFER_REAL_DFLASH_OUT");
    require(!dir.empty()&&!std::filesystem::exists(dir),"preserve history diagnostic receipt");
    std::filesystem::create_directories(dir);
    std::ofstream checks(dir/"checks.csv");checks<<"case,pass\n";
    const auto gate=[&](const char* name,bool pass){checks<<name<<','<<pass<<'\n';checks.flush();require(pass,std::string("history ")+name);};
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    require(GlobalMemoryStatusEx(&memory)&&memory.ullAvailPhys>(8ULL<<30),"history host reserve");
    std::array<std::unique_ptr<DeviceBuffer>,5> storage;std::array<std::uint16_t*,5> staging{};
    for(int t=0;t<5;++t){storage[t]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[t]=static_cast<std::uint16_t*>(storage[t]->get());}
    struct Reference {Root initial;std::shared_ptr<const Exl3ExactHostState> final;int prefix=0;
        std::vector<std::int64_t> tokens;std::array<std::vector<std::uint16_t>,5> taps;};
    std::array<Reference,2> refs;Json prefix_costs=Json::array();
    _putenv_s("NINFER_EXL3_PINNED_RECURRENT_EXPORT","0");
    for(int i=0;i<2;++i){
        auto& ref=refs[i];const auto& input=i?prose:code;
        ref.prefix=requested?requested:static_cast<int>(input.size());
        require(ref.prefix>=4096&&ref.prefix<=16384&&ref.prefix+outputs<=target.max_context()&&input.size()>=ref.prefix,"history actual fixture extent");
        auto start=Clock::now();auto context=target.create_context(true);context->prepare_continuation(8);const auto construct_ms=ms(start);start=Clock::now();
        ref.initial=Exl3VeriCacheRequest::initialize(*context,std::span<const std::int64_t>(input.data(),ref.prefix),1024)->compact_draft(draft,staging);
        const auto prefill_ms=ms(start);start=Clock::now();
        for(int j=0;j<outputs;++j){auto token=sample_target(*context);ref.tokens.push_back(token);context->decode(token);const auto taps=context->exact_tap_rows_host();for(int t=0;t<5;++t)ref.taps[t].insert(ref.taps[t].end(),taps[t].begin(),taps[t].end());}
        ref.final=context->export_exact_host_state();
        prefix_costs.push_back(Json{{"fixture",i?"prose":"code"},{"prefix",ref.prefix},{"context_ms",construct_ms},{"fresh_prefix_and_compact_ms",prefill_ms},{"serial_oracle_ms",ms(start)},{"tokens",ref.tokens}});
        std::cout<<"HISTORY_REFERENCE fixture="<<(i?"prose":"code")<<" prefix="<<ref.prefix<<" prefill_ms="<<prefill_ms<<std::endl;
    }
    std::ofstream(dir/"prefix-costs.json")<<prefix_costs.dump(2);
    gate("pageable_reference_owners",Exl3RecurrentPinBudget::snapshot()[0]==0);
    _putenv_s("NINFER_EXL3_PINNED_RECURRENT_EXPORT","1");
    Exl3VeriCacheServingIdentity identity{"solo-history","SC_6.00bpw_H6_V6","pinned-ids","exact-B8-greedy","text"};
    Exl3VeriCacheServingPrefixCache cache({2,64,2ULL<<30,8ULL<<30},identity);
    Exl3VeriCacheServingCoordinator coordinator(cache,{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
    auto start=Clock::now();auto lane=std::make_unique<Lane>(target.create_context(true),draft,identity.contract());const auto construct_ms=ms(start);
    struct Record {Root root;std::vector<std::int64_t> tokens;std::vector<int> parts;std::array<std::vector<std::uint16_t>,5> taps;};
    std::array<Record,2> records;Json rows=Json::array();
    if(profile)cuda_check(cudaProfilerStart(),"begin current history diagnostic");
    for(int i=0;i<2;++i){
        nvtxRangePushA(i?"history.prose":"history.code");
        const auto request_start=Clock::now();coordinator.admit(refs[i].initial);auto lease=coordinator.acquire();require(lease.has_value(),"history acquire");lane->acquire(*lease);lease.reset();
        const auto acquire_ms=ms(request_start);const auto kv0=lane->context().host_kv_stats();const auto ex0=lane->context().recurrent_export_stats();const auto ls0=lane->stats();
        const auto decode_start=Clock::now();auto& r=records[i];double first=0;std::uint64_t verified=0,replayed=0;
        nvtxRangePushA("history.decode");
        while(r.tokens.size()<outputs){
            auto p=lane->verify(lane->propose(std::min<int>(8,outputs-static_cast<int>(r.tokens.size()))));
            auto pub=lane->publish(coordinator,p);if(!first)first=ms(request_start);
            r.tokens.insert(r.tokens.end(),pub.tokens.begin(),pub.tokens.end());r.parts.push_back(static_cast<int>(pub.tokens.size()));
            verified+=p.verification.verification_rows;replayed+=p.verification.replay_rows;
            for(int t=0;t<5;++t)r.taps[t].insert(r.taps[t].end(),p.verification.committed_taps[t].begin(),p.verification.committed_taps[t].end());
        }
        const auto decode_ms=ms(decode_start);nvtxRangePop();r.root=lane->lease().root;
        const auto audit_start=Clock::now();Exl3HostResidencyProbe probe;const std::array<Root,1> roots{r.root};
        Exl3VeriCacheRequest::visit_host_allocations(roots,[&](const void* p,std::size_t n){probe.add(p,n);},false);const auto locked=probe.measure();
        gate("publication_resident_locked",locked.allocated_union_bytes==locked.resident_tensor_bytes&&locked.allocated_union_bytes==locked.locked_tensor_bytes);const auto audit_ms=ms(audit_start);
        const auto kv1=lane->context().host_kv_stats();const auto ex1=lane->context().recurrent_export_stats();const auto ls1=lane->stats();
        auto completed=lane->lease();lane->release();coordinator.complete(completed);const auto wall=ms(request_start);
        Json kv=Json::object();
#define HISTORY_DELTA(field) kv[#field]=kv1.field-kv0.field
        HISTORY_DELTA(h2d_bytes);HISTORY_DELTA(d2h_bytes);HISTORY_DELTA(transfer_calls);HISTORY_DELTA(copy_submissions);HISTORY_DELTA(completed_rows);
        HISTORY_DELTA(pinned_gather_cpu_ns);HISTORY_DELTA(pinned_wait_cpu_ns);HISTORY_DELTA(pinned_scatter_cpu_ns);HISTORY_DELTA(page_extension_cpu_ns);
        HISTORY_DELTA(page_clone_bytes);HISTORY_DELTA(page_prefix_refs);HISTORY_DELTA(pinned_pending_scan_cpu_ns);
#undef HISTORY_DELTA
        const Json recurrent=Json{{"exports",ex1.exports-ex0.exports},
            {"total_ms",ex1.total_ms-ex0.total_ms},{"recurrent_ms",ex1.recurrent_ms-ex0.recurrent_ms},
            {"submit_ms",ex1.submit_ms-ex0.submit_ms},{"fence_ms",ex1.fence_ms-ex0.fence_ms},
            {"bytes",ex1.recurrent_bytes-ex0.recurrent_bytes},{"pool_hits",ex1.pool_hits-ex0.pool_hits},
            {"planned_copy_ranges",ex1.planned_copy_ranges-ex0.planned_copy_ranges},
            {"coalesced_copy_ranges",ex1.coalesced_copy_ranges-ex0.coalesced_copy_ranges},
            {"copy_submissions",ex1.copy_submissions-ex0.copy_submissions},
            {"fp32_copy_submissions",ex1.fp32_copy_submissions-ex0.fp32_copy_submissions},
            {"fallbacks",ex1.fallbacks-ex0.fallbacks}};
        rows.push_back(Json{{"fixture",i?"prose":"code"},{"prefix",refs[i].prefix},{"executed_diagnostic_rows",r.tokens.size()},{"request_wall_ms",wall},{"acquire_ms",acquire_ms},{"decode_ms",decode_ms},{"first_release_ms",first},
            {"proposal_calls",ls1.proposal_calls-ls0.proposal_calls},{"proposal_ms",ls1.proposal_ms-ls0.proposal_ms},{"verification_ms",ls1.verification_ms-ls0.verification_ms},{"verified_rows",verified},{"replayed_rows",replayed},{"kv",kv},
            {"recurrent",recurrent},{"recurrent_exports",ex1.exports-ex0.exports},{"recurrent_export_ms",ex1.total_ms-ex0.total_ms},{"recurrent_bytes",ex1.recurrent_bytes-ex0.recurrent_bytes},{"fallbacks",ex1.fallbacks-ex0.fallbacks},{"publication_residency_audit_ms",audit_ms},{"resident_locked_tensor_bytes",locked.allocated_union_bytes}});
        nvtxRangePop();
    }
    if(profile)cuda_check(cudaProfilerStop(),"end current history diagnostic");
    for(int i=0;i<2;++i){
        auto& r=records[i];const auto& ref=refs[i];
        gate("M1_tokens_full_state_taps",r.tokens==ref.tokens&&r.taps==ref.taps&&r.root->state()->same_payload(*ref.final));
        ref.initial->restore_draft(draft,staging);int offset=0;
        for(int count:r.parts){std::array<const std::uint16_t*,5> ptr{};for(int t=0;t<5;++t){cuda_check(cudaMemcpy(staging[t],ref.taps[t].data()+std::size_t(offset)*kHidden,std::size_t(count)*kHidden*2,cudaMemcpyHostToDevice),"history independent ring");ptr[t]=staging[t];}draft.commit_prefill_block(ptr.data(),count,ref.prefix+offset);offset+=count;}
        const auto ring=draft.export_host_ring(nullptr,false);r.root->restore_draft(draft,staging);gate("independent_projected_ring",ring->same_payload(*draft.export_host_ring(nullptr,false)));
        rows[i]["exact"]=true;rows[i]["tokens"]=r.tokens;
    }
    const auto persistent=lane->persistent_bytes();for(auto& r:records)r.root.reset();lane.reset();coordinator.close();
    gate("all_pins_retired",Exl3RecurrentPinBudget::snapshot()[0]==0&&Exl3RecurrentPinBudget::snapshot()[2]==0);
    std::ofstream(dir/"result.json")<<Json{{"status","PASS_CURRENT_REAL_DFLASH_HISTORY_DIAGNOSTIC"},{"profiled",profile},{"context_ms",construct_ms},{"persistent_bytes",persistent},{"requests",rows},{"limits","instrumented diagnostic; not paired throughput or broad quality"}}.dump(2);
    std::cout<<"PASS_CURRENT_REAL_DFLASH_HISTORY_DIAGNOSTIC\n";
}
