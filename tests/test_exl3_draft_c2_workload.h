#pragma once

void run_draft_c2_workload(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using namespace ninfer::exl3;
    using Root=std::shared_ptr<const Exl3VeriCacheRequest>;
    using Lane=Exl3Dflash2Execution;
    using Clock=std::chrono::steady_clock;
    using Json=nlohmann::ordered_json;
    const auto ms=[](auto start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
    const auto utc_ms=[](){return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();};
    const int prefix=std::stoi(env("NINFER_REAL_DFLASH_PREFIX")),outputs=std::stoi(env("NINFER_REAL_DFLASH_OUTPUTS")),pairs=std::stoi(env("NINFER_REAL_DFLASH_PAIRS"));
    const bool extended_comparison=env("NINFER_TEST_EXTENDED_STREAM_COHORT")=="1";
    const bool kernel_comparison=extended_comparison||env("NINFER_TEST_K6_STREAM_COHORT")=="1";
    const bool device_comparison=env("NINFER_TEST_DEVICE_PREFIX_COHORT")=="1";
    const bool long_requests=env("NINFER_TEST_HISTORY_LONG_COHORT")=="1";
    const bool capacity_comparison=env("NINFER_TEST_DEVICE_PREFIX_16K_COHORT")=="1";
    require(!capacity_comparison||(device_comparison&&long_requests),"16K device reserve requires long C1 comparison");
    require(!(device_comparison&&kernel_comparison),"separate K6 and device-prefix cohorts");
    require((long_requests?(device_comparison&&prefix==0):(prefix>=512&&prefix<=4096))&&outputs>=32&&outputs<=128&&pairs>=1&&pairs<=3,"C2 bounded extents");
    const std::filesystem::path dir=env("NINFER_REAL_DFLASH_OUT");
    require(!dir.empty()&&!std::filesystem::exists(dir),"preserve C2 workload receipt");std::filesystem::create_directories(dir);
    _putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION",device_comparison||extended_comparison?"1":"0");
    _putenv_s("NINFER_EXL3_EXTENDED_STREAM_REDUCTION","0");
    _putenv_s("NINFER_EXL3_HOST_KV_DEVICE_PREFIX","0");
    if(device_comparison)run_device_prefix_metadata_gate(dir/"device-prefix-metadata.csv");
    if(capacity_comparison)run_device_prefix_16k_metadata_gate(dir/"device-prefix-16k-metadata.csv");
    if(extended_comparison)run_extended_stream_operator_gate(target,draft,code,prose,dir/"operators",prefix);
    else if(kernel_comparison)run_k6_stream_operator_gate(target,draft,code,prose,dir/"operators",prefix);
    std::ofstream checks(dir/"checks.csv"),workloads(dir/"workloads.jsonl");checks<<"case,pass\n";
    auto gate=[&](const char* name,bool pass){checks<<name<<','<<pass<<'\n';checks.flush();require(pass,std::string("C2 workload ")+name);};
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);require(GlobalMemoryStatusEx(&memory)&&memory.ullAvailPhys>(8ULL<<30),"C2 physical reserve");
    std::array<std::unique_ptr<DeviceBuffer>,5> storage;std::array<std::uint16_t*,5> staging{};
    for(int t=0;t<5;++t){storage[t]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[t]=static_cast<std::uint16_t*>(storage[t]->get());}
    struct Reference {Root initial;std::shared_ptr<const Exl3ExactHostState> final;int prefix=0;std::vector<std::int64_t> tokens;std::array<std::vector<std::uint16_t>,5> taps;};
    std::array<Reference,2> refs;Json prefix_costs=Json::array();
    _putenv_s("NINFER_EXL3_PINNED_RECURRENT_EXPORT","0");
    for(int i=0;i<2;++i){
        const auto& input=i?prose:code;require(input.size()>=prefix,"C2 fixture length");auto start=Clock::now();
        refs[i].prefix=long_requests?static_cast<int>(input.size()):prefix;
        require(refs[i].prefix<=16384&&refs[i].prefix+outputs<=target.max_context(),"bounded actual history request length");
        auto context=target.create_context(true);context->prepare_continuation(8);const auto construct_ms=ms(start);start=Clock::now();
        refs[i].initial=Exl3VeriCacheRequest::initialize(*context,std::span<const std::int64_t>(input.data(),refs[i].prefix),1024)->compact_draft(draft,staging);
        const auto prefill_ms=ms(start);start=Clock::now();
        for(int j=0;j<outputs;++j){const auto token=sample_target(*context);refs[i].tokens.push_back(token);context->decode(token);auto taps=context->exact_tap_rows_host();for(int t=0;t<5;++t)refs[i].taps[t].insert(refs[i].taps[t].end(),taps[t].begin(),taps[t].end());if(long_requests&&(token==248044||token==248046))break;}
        refs[i].final=context->export_exact_host_state();
        prefix_costs.push_back(Json{{"fixture",i?"prose":"code"},{"prefix",refs[i].prefix},{"tokens",refs[i].tokens},{"context_ms",construct_ms},{"fresh_prefix_and_compact_draft_ms",prefill_ms},{"serial_oracle_ms",ms(start)}});
    }
    std::ofstream(dir/"prefix-costs.json")<<prefix_costs.dump(2);
    gate("reference_owners_use_no_pinned_pool",Exl3RecurrentPinBudget::snapshot()[0]==0);
    _putenv_s("NINFER_EXL3_PINNED_RECURRENT_EXPORT","1");
    int device=0;cuda_check(cudaGetDevice(&device),"C2 workload device");
    auto run=[&](int physical,int requests,bool ring_audit,int pair,bool candidate=false,bool force_fallback=false){
        _putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION",(device_comparison||extended_comparison||candidate)?"1":"0");
        _putenv_s("NINFER_EXL3_EXTENDED_STREAM_REDUCTION",extended_comparison&&candidate?"1":"0");
        _putenv_s("NINFER_EXL3_HOST_KV_DEVICE_PREFIX",(device_comparison&&(candidate||capacity_comparison))?"1":"0");
        _putenv_s("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS",capacity_comparison&&candidate?"16384":"4096");
        struct Stream {cudaStream_t value=nullptr;~Stream(){if(value)cudaStreamDestroy(value);}};
        struct Record {Json data;Root final;std::vector<int> partitions;};
        std::vector<Record> records(requests);std::array<Stream,2> streams;
        std::unique_ptr<Exl3Dflash2DraftModel> child;std::array<std::unique_ptr<Lane>,2> lanes;
        Exl3VeriCacheServingIdentity identity{"solo-c2-workload","SC_6.00bpw_H6_V6","pinned-token-ids","exact-B8-greedy","text"};
        Exl3VeriCacheServingPrefixCache cache({2,64,2ULL<<30,8ULL<<30},identity);
        Exl3VeriCacheServingCoordinator coordinator(cache,{static_cast<std::size_t>(requests),static_cast<std::size_t>(physical),memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
        std::size_t free_before=0,total=0,free_constructed=0,free_after=0;cuda_check(cudaMemGetInfo(&free_before,&total),"C2 free before");
        const auto began_utc=utc_ms();const auto begin=Clock::now();
        if(physical==2){child=draft.create_execution();for(auto& stream:streams)cuda_check(cudaStreamCreateWithFlags(&stream.value,cudaStreamNonBlocking),"C2 workload stream");}
        for(int i=0;i<physical;++i)lanes[i]=std::make_unique<Lane>(target.create_context(true),i?*child:draft,identity.contract(),streams[i].value);
        const auto construct_ms=ms(begin);std::uint64_t lane_bytes=0;for(int i=0;i<physical;++i)lane_bytes+=lanes[i]->persistent_bytes();
        cuda_check(cudaMemGetInfo(&free_constructed,&total),"C2 free constructed");
        std::vector<Root> inputs;for(int i=0;i<requests;++i)inputs.push_back(refs[i%2].initial);
        const auto admission_start=Clock::now();const auto tickets=coordinator.admit_batch(inputs);const auto admission_ms=ms(admission_start);inputs.clear();
        std::atomic<bool> failed=false;std::array<std::exception_ptr,2> errors;
        auto execute=[&](int lane_index){
            try{
                cuda_check(cudaSetDevice(device),"C2 workload submit device");
                while(!failed){
                    auto lease=coordinator.acquire();if(!lease)break;
                    auto found=std::find_if(tickets.begin(),tickets.end(),[&](const auto& t){return t.request_id==lease->ticket.request_id;});
                    require(found!=tickets.end(),"C2 ticket identity");const int r=static_cast<int>(found-tickets.begin()),fixture=r%2;
                    auto& record=records[r];auto& lane=*lanes[lane_index];const auto before=lane.stats();
                    const auto request_start=Clock::now();const double acquire_begin_ms=ms(begin);lane.acquire(*lease);lease.reset();
                    if((capacity_comparison||extended_comparison)&&candidate&&ring_audit){
                        const std::array<std::int64_t,2> forced{refs[fixture].tokens[0],(refs[fixture].tokens[1]+1)%248320};
                        auto rejected=lane.verify(forced);require(rejected.verification.rejected&&rejected.verification.committed_tokens==std::vector<std::int64_t>(refs[fixture].tokens.begin(),refs[fixture].tokens.begin()+2),"16K reserve forced rejection repair");
                        lane.abort();require(lane.context().export_exact_host_state()->same_payload(*refs[fixture].initial->state()),"16K reserve abort exact root");
                    }
                    const double acquired_ms=ms(begin);std::vector<std::int64_t> tokens;std::array<std::vector<std::uint16_t>,5> taps;
                    std::uint64_t verified=0,replayed=0;double first_release_ms=0,largest_stall_ms=0,last_release_ms=acquired_ms;
                    const int useful_limit=static_cast<int>(refs[fixture].tokens.size());
                    while(tokens.size()<useful_limit){
                        auto pending=lane.verify(lane.propose(std::min<int>(8,useful_limit-static_cast<int>(tokens.size()))));
                        auto pub=lane.publish(coordinator,pending);const double released_ms=ms(begin);
                        if(!first_release_ms)first_release_ms=released_ms;
                        largest_stall_ms=std::max(largest_stall_ms,released_ms-last_release_ms);last_release_ms=released_ms;
                        tokens.insert(tokens.end(),pub.tokens.begin(),pub.tokens.end());if(ring_audit)record.partitions.push_back(static_cast<int>(pub.tokens.size()));
                        verified+=pending.verification.verification_rows;replayed+=pending.verification.replay_rows;
                        for(int t=0;t<5;++t)taps[t].insert(taps[t].end(),pending.verification.committed_taps[t].begin(),pending.verification.committed_taps[t].end());
                    }
                    const auto audit_start=Clock::now();
                    require(tokens==refs[fixture].tokens&&taps==refs[fixture].taps&&lane.lease().root->state()->same_payload(*refs[fixture].final),"C2 request tokens/full state/taps");
                    Exl3HostResidencyProbe probe;const std::array<Root,1> roots{lane.lease().root};
                    Exl3VeriCacheRequest::visit_host_allocations(roots,[&](const void* p,std::size_t n){probe.add(p,n);},false);const auto resident=probe.measure();
                    require(resident.allocated_union_bytes==resident.resident_tensor_bytes&&resident.allocated_union_bytes==resident.locked_tensor_bytes,"C2 publication residency");
                    const auto audit_ms=ms(audit_start);const auto after=lane.stats();if(ring_audit)record.final=lane.lease().root;
                    auto completed=lane.lease();lane.release();coordinator.complete(completed);
                    record.data=Json{{"request",r},{"fixture",fixture?"prose":"code"},{"prefix",refs[fixture].prefix},{"physical_lane",lane_index},{"exact",true},{"useful",tokens.size()},
                        {"acquire_begin_ms",acquire_begin_ms},{"acquired_ms",acquired_ms},{"first_release_ms",first_release_ms},{"last_release_ms",last_release_ms},{"completion_ms",ms(begin)},
                        {"service_ms",ms(request_start)},{"largest_publication_stall_ms",largest_stall_ms},{"full_state_residency_audit_ms",audit_ms},
                        {"proposal_calls",after.proposal_calls-before.proposal_calls},{"proposed_rows",after.proposed_rows-before.proposed_rows},{"verified_rows",verified},{"replayed_rows",replayed},
                        {"reset_ms",after.reset_ms-before.reset_ms},{"restore_ms",after.restore_ms-before.restore_ms},{"proposal_ms",after.proposal_ms-before.proposal_ms},{"verification_ms",after.verification_ms-before.verification_ms},
                        {"resident_locked_tensor_bytes",resident.allocated_union_bytes}};
                }
            }catch(...){errors[lane_index]=std::current_exception();failed=true;lanes[lane_index].reset();}
        };
        if(physical==1)execute(0);
        else{
            std::latch launch(1);bool launch_failed=false;std::array<std::thread,2> threads;
            try{for(int i=0;i<2;++i)threads[i]=std::thread([&,i]{launch.wait();if(!launch_failed)execute(i);});}
            catch(...){launch_failed=true;launch.count_down();for(auto& thread:threads)if(thread.joinable())thread.join();throw;}
            launch.count_down();for(auto& thread:threads)thread.join();
        }
        for(auto error:errors)if(error)std::rethrow_exception(error);
        require(coordinator.stats().completions==requests,"C2 complete workload extent");
        Json pool_stats=Json::array();std::uint64_t kernel_calls=0;for(int i=0;i<physical;++i){kernel_calls+=lanes[i]->context().stream_reduction_calls();const auto s=lanes[i]->context().recurrent_export_stats();pool_stats.push_back(Json{{"lane",i},{"exports",s.exports},{"pinned_exports",s.pinned_exports},{"fallbacks",s.fallbacks},{"pool_bytes",s.pool_bytes},{"recurrent_bytes",s.recurrent_bytes},{"export_ms",s.total_ms},{"allocate_ms",s.allocate_ms},{"register_ms",s.register_ms}});}
        require((candidate||device_comparison||extended_comparison)?kernel_calls>0:kernel_calls==0,"K6 stream candidate dispatch count");
        std::uint64_t extended_calls=0;for(int i=0;i<physical;++i)extended_calls+=lanes[i]->context().stream_reduction_calls(true);
        require(extended_comparison&&candidate?extended_calls>0:extended_calls==0,"separate extended stream dispatch count");
        Json history=Json::array();
        for(int i=0;i<physical;++i){const auto h=lanes[i]->context().host_kv_stats();
            const auto cache_bytes=Exl3DevicePrefixCache::bytes*(capacity_comparison&&candidate?4:1);
            require(force_fallback?(h.device_prefix_bytes==0&&h.device_prefix_hit_bytes==0&&h.device_prefix_fallbacks==1):((device_comparison&&(candidate||capacity_comparison))?(h.device_prefix_bytes==cache_bytes&&(long_requests||h.device_prefix_hit_bytes>0)&&h.device_prefix_fallbacks==0):h.device_prefix_bytes==0),"device prefix actual dispatch and reserve");
            history.push_back(Json{{"lane",i},{"h2d_bytes",h.h2d_bytes},{"d2h_bytes",h.d2h_bytes},{"device_prefix_bytes",h.device_prefix_bytes},{"device_prefix_hit_bytes",h.device_prefix_hit_bytes},{"device_prefix_fill_bytes",h.device_prefix_fill_bytes},{"device_prefix_fallbacks",h.device_prefix_fallbacks}});
        }
        const auto teardown_start=Clock::now();for(auto& lane:lanes)lane.reset();child.reset();coordinator.close();for(auto& stream:streams)if(stream.value){cuda_check(cudaStreamDestroy(stream.value),"C2 stream retirement");stream.value=nullptr;}
        cuda_check(cudaDeviceSynchronize(),"C2 complete workload fence");const auto teardown_ms=ms(teardown_start);const auto wall_ms=ms(begin);const auto ended_utc=utc_ms();
        cuda_check(cudaMemGetInfo(&free_after,&total),"C2 free retired");
        if(ring_audit)for(int r=0;r<requests;++r){
            const int fixture=r%2;refs[fixture].initial->restore_draft(draft,staging);int offset=0;
            for(int rows:records[r].partitions){std::array<const std::uint16_t*,5> ptr{};for(int t=0;t<5;++t){cuda_check(cudaMemcpy(staging[t],refs[fixture].taps[t].data()+std::size_t(offset)*kHidden,rows*std::size_t(kHidden)*2,cudaMemcpyHostToDevice),"C2 independent ring replay");ptr[t]=staging[t];}draft.commit_prefill_block(ptr.data(),rows,refs[fixture].prefix+offset);offset+=rows;}
            auto expected=draft.export_host_ring(nullptr,false);records[r].final->restore_draft(draft,staging);gate("independent_ring",expected->same_payload(*draft.export_host_ring(nullptr,false)));records[r].final.reset();
        }
        const auto pins=Exl3RecurrentPinBudget::snapshot();gate("all_arm_pin_owners_retired",pins[0]==0&&pins[2]==0);
        gate("arm_device_prefix_retired",Exl3DevicePrefixCache::budget_snapshot()[0]==(force_fallback?Exl3DevicePrefixCache::cap:0)&&Exl3DevicePrefixCache::budget_snapshot()[1]==0);
        Json request_rows=Json::array();for(const auto& record:records)request_rows.push_back(record.data);
        std::size_t useful=0;for(int r=0;r<requests;++r)useful+=refs[r%2].tokens.size();
        Json result{{"physical",physical},{"candidate",candidate},{"stream_reduction_calls",kernel_calls},{"extended_stream_reduction_calls",extended_calls},{"extended_stream_comparison",extended_comparison},{"logical_arrivals",requests},{"pair",pair},{"qualification_only",ring_audit},{"prefix",prefix},{"output_budget_each",outputs},{"useful",useful},{"stop_terminated_long_requests",long_requests},{"forced_device_prefix_fallback",force_fallback},
            {"began_unix_ms",began_utc},{"ended_unix_ms",ended_utc},{"wall_ms_including_audits",wall_ms},{"construction_ms",construct_ms},{"admission_ms",admission_ms},{"teardown_ms",teardown_ms},
            {"lane_persistent_bytes",lane_bytes},{"draft_extra_execution_bytes",physical==2?draft.execution_bytes():0},{"cuda_free_before",free_before},{"cuda_free_constructed",free_constructed},{"cuda_free_after",free_after},
            {"pin_peak_process_bytes",pins[1]},{"pin_live_after",pins[0]},{"pin_quarantined",pins[2]},{"pool_stats",pool_stats},{"history",history},{"device_prefix_comparison",device_comparison},{"device_prefix_capacity_comparison",capacity_comparison},{"requests",request_rows}};
        workloads<<result.dump()<<'\n';workloads.flush();std::cout<<"C2_WORKLOAD physical="<<physical<<" pair="<<pair<<" wall_ms="<<wall_ms<<" exact=1\n"<<std::flush;
    };
    if(device_comparison){
        {std::array<std::unique_ptr<Exl3DevicePrefixCache>,4> blockers;for(auto& blocker:blockers){blocker=std::make_unique<Exl3DevicePrefixCache>();gate("native_fallback_reserve_admitted",blocker->admitted());}
        run(1,2,true,-2,true,true);}
        gate("native_fallback_and_blockers_retired",Exl3DevicePrefixCache::budget_snapshot()[0]==0);
    }
    if(kernel_comparison||device_comparison){
        for(int physical=1;physical<=(long_requests?1:2);++physical)for(bool candidate:{false,true})run(physical,2,true,-1,candidate);
        for(int physical=1;physical<=(long_requests?1:2);++physical)for(int pair=0;pair<pairs;++pair)for(int order=0;order<2;++order)run(physical,6,false,pair,(pair%2)?order==0:order==1);
    }else{
        run(1,2,true,-1);run(2,2,true,-1);
        for(int pair=0;pair<pairs;++pair)for(int order=0;order<2;++order)run((pair%2)?2-order:1+order,6,false,pair);
    }
    gate("full_campaign_pin_retirement",Exl3RecurrentPinBudget::snapshot()[0]==0);
    std::ofstream(dir/"result.json")<<Json{{"status","PASS_REAL_DFLASH_C1_C2_WORKLOAD"},{"pairs",pairs},{"prefix",prefix},{"outputs",outputs},{"sampled_decoding",false},{"http",false}}.dump(2);
}
