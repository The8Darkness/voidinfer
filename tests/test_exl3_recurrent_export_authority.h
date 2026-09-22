#pragma once
void run_recurrent_export_authority(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose){
    using namespace ninfer::exl3;
    using Root=std::shared_ptr<const Exl3VeriCacheRequest>;
    using Lane=Exl3Dflash2Execution;
    using Clock=std::chrono::steady_clock;
    const auto ms=[](auto t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();};
    const int prefix=std::stoi(env("NINFER_REAL_DFLASH_PREFIX")),outputs=std::stoi(env("NINFER_REAL_DFLASH_OUTPUTS")),pairs=std::stoi(env("NINFER_REAL_DFLASH_PAIRS"));
    const bool batched_candidate=env("NINFER_TEST_BATCHED_RECURRENT_EXPORT")=="1";
    const auto configure_export=[&](bool pinned) {
        _putenv_s("NINFER_EXL3_PINNED_RECURRENT_EXPORT",pinned?"1":"0");
        _putenv_s("NINFER_EXL3_BATCHED_RECURRENT_EXPORT",
                  pinned && batched_candidate?"1":"0");
    };
    const std::filesystem::path dir=env("NINFER_REAL_DFLASH_OUT");require(!std::filesystem::exists(dir),"preserve export receipt");std::filesystem::create_directories(dir);
    std::ofstream checks(dir/"checks.csv"),timings(dir/"workloads.csv"),phases(dir/"phases.csv");
    checks<<"fixture,case,pass\n";
    timings<<"fixture,pair,arm,requests,useful,wall_ms,first_release_ms,proposal_calls,verified,replayed,pinned_exports,fallbacks,exact\n";
    phases<<"fixture,pair,arm,exports,recurrent_bytes,total_export_ms,recurrent_ms,allocate_ms,register_ms,submit_ms,fence_ms,pool_hits,pool_bytes,process_peak_bytes\n";
    const auto gate=[&](const char* f,const char* c,bool ok){checks<<f<<','<<c<<','<<ok<<'\n';checks.flush();require(ok,std::string("export gate: ")+c);};
    Exl3VeriCacheServingIdentity identity{"solo-export-real-b8","SC_6.00bpw_H6_V6","pinned-tokens","exact-greedy","text"};
    Exl3VeriCacheServingPrefixCache cache({2,64,2ULL<<30,8ULL<<30},identity);
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);require(GlobalMemoryStatusEx(&memory),"export memory");
    Exl3VeriCacheServingCoordinator coordinator(cache,{1,1,memory.ullTotalPhys-(8ULL<<30),8ULL<<30});
    std::array<std::unique_ptr<DeviceBuffer>,5> allocation;std::array<std::uint16_t*,5> staging{};
    for(int i=0;i<5;++i){allocation[i]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[i]=static_cast<std::uint16_t*>(allocation[i]->get());}
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},std::pair{"prose",&prose}}){
        const auto* name=fixture.first;Root initial;std::shared_ptr<const Exl3ExactHostState> oracle;
        std::vector<std::int64_t> expected;std::array<std::vector<std::uint16_t>,5> oracle_taps;
        configure_export(false);
        {
            auto ctx=target.create_context(true);ctx->prepare_continuation(8);
            initial=Exl3VeriCacheRequest::initialize(*ctx,std::span<const std::int64_t>(fixture.second->data(),prefix),1024)->compact_draft(draft,staging);
            for(int n=0;n<outputs;++n){auto token=sample_target(*ctx);expected.push_back(token);ctx->decode(token);auto taps=ctx->exact_tap_rows_host();for(int i=0;i<5;++i)oracle_taps[i].insert(oracle_taps[i].end(),taps[i].begin(),taps[i].end());}
            oracle=ctx->export_exact_host_state();
        }
        // Three retained immutable slabs exhaust the local pool; a fourth export
        // stays exact in pageable storage. Both representation views must compare.
        std::shared_ptr<const Exl3ExactHostState> survivor;
        configure_export(true);
        {
            auto ctx=target.create_context(true);ctx->restore_exact_host_state(*initial->state());
            std::vector<std::shared_ptr<const Exl3ExactHostState>> held;
            for(int n=0;n<4;++n){held.push_back(ctx->export_exact_host_state());gate(name,"immutable_export_exact",held.back()->same_payload(*initial->state()));}
            const auto stats=ctx->recurrent_export_stats();gate(name,"pool_exhaustion_exact_fallback",stats.pinned_exports==3 && stats.fallbacks==1);
            survivor=held[0];
        }
        {
            configure_export(false);auto ctx=target.create_context(true);
            ctx->restore_exact_host_state(*survivor);gate(name,"slab_outlives_context_restore",ctx->export_exact_host_state()->same_payload(*initial->state()));
        }
        survivor.reset();
        struct Result {std::vector<std::int64_t> tokens;std::vector<int> partitions;std::array<std::vector<std::uint16_t>,5> taps;Root final;double wall=0,first=0;std::uint64_t proposals=0,verified=0,replayed=0;Exl3RecurrentExportStats stats;};
        const auto run=[&](bool pinned,int requests,bool audit){
            configure_export(pinned);Result result;
            cudaStream_t execution_stream=nullptr;
            std::shared_ptr<cudaStream_t> execution_stream_owner;
            if(pinned && batched_candidate) {
                auto* owned=new cudaStream_t(nullptr);
                cuda_check(cudaStreamCreateWithFlags(owned,cudaStreamNonBlocking),
                           "batched recurrent export authority stream");
                execution_stream=*owned;
                execution_stream_owner=std::shared_ptr<cudaStream_t>(owned,[](cudaStream_t* value) {
                    if(value && *value)(void)cudaStreamDestroy(*value);
                    delete value;
                });
            }
            const auto begin=Clock::now();
            auto lane=execution_stream?
                std::make_unique<Lane>(target.create_context(true),
                    std::shared_ptr<Exl3Dflash2DraftModel>(&draft,
                        [](Exl3Dflash2DraftModel*){}),identity.contract(),
                    execution_stream,execution_stream_owner):
                std::make_unique<Lane>(target.create_context(true),draft,
                    identity.contract());
            for(int r=0;r<requests;++r){
                result.final.reset();result.tokens.clear();result.partitions.clear();for(auto& tap:result.taps)tap.clear();
                coordinator.admit(initial);lane->acquire(*coordinator.acquire());
                while(result.tokens.size()<outputs){
                    auto proposed=lane->propose(std::min<int>(8,outputs-static_cast<int>(result.tokens.size())));
                    auto pending=lane->verify(proposed);auto pub=lane->publish(coordinator,pending);
                    result.tokens.insert(result.tokens.end(),pub.tokens.begin(),pub.tokens.end());result.partitions.push_back(static_cast<int>(pub.tokens.size()));
                    result.verified+=pending.verification.verification_rows;result.replayed+=pending.verification.replay_rows;
                    if(audit)for(int i=0;i<5;++i)result.taps[i].insert(result.taps[i].end(),pending.verification.committed_taps[i].begin(),pending.verification.committed_taps[i].end());
                    if(!result.first)result.first=ms(begin);
                }
                result.final=lane->lease().root;
                if(audit){Exl3HostResidencyProbe probe;const std::array<Root,1> roots{result.final};Exl3VeriCacheRequest::visit_host_allocations(roots,[&](const void* p,std::size_t n){probe.add(p,n);},false);auto measured=probe.measure();gate(name,"all_published_payload_resident_locked",measured.allocated_union_bytes==measured.locked_tensor_bytes && measured.allocated_union_bytes==measured.resident_tensor_bytes);}
                auto lease=lane->lease();lane->release();coordinator.complete(lease);
                gate(name,"request_tokens",result.tokens==expected);
            }
            result.stats=lane->context().recurrent_export_stats();result.proposals=lane->stats().proposal_calls;
            if(pinned && batched_candidate)
                gate(name,"batched_recurrent_export_dispatch",
                     result.stats.batched_copy_calls>0 &&
                     result.stats.batched_copy_ranges==48*result.stats.batched_copy_calls);
            lane.reset();
            cuda_check(cudaDeviceSynchronize(),"export workload teardown");
            execution_stream_owner.reset();
            result.wall=ms(begin);return result;
        };
        for(bool pinned:{false,true}){
            auto result=run(pinned,1,true);
            gate(name,"serial_full_state_taps",result.final->state()->same_payload(*oracle) && result.taps==oracle_taps);
            initial->restore_draft(draft,staging);int first=0;
            for(auto rows:result.partitions){std::array<const std::uint16_t*,5> pointers{};for(int i=0;i<5;++i){cuda_check(cudaMemcpy(staging[i],result.taps[i].data()+static_cast<std::size_t>(first)*kHidden,static_cast<std::size_t>(rows)*kHidden*2,cudaMemcpyHostToDevice),"export ring oracle");pointers[i]=staging[i];}draft.commit_prefill_block(pointers.data(),rows,prefix+first);first+=rows;}
            const auto ring=draft.export_host_ring(nullptr,false);result.final->restore_draft(draft,staging);gate(name,"independent_ring",ring->same_payload(*draft.export_host_ring(nullptr,false)));
        }
        for(int pair=0;pair<pairs;++pair)for(int order=0;order<2;++order){
            const bool pinned=(pair%2)?order==0:order==1;auto r=run(pinned,3,false);const auto& s=r.stats;
            const bool exact=r.tokens==expected && r.final->state()->same_payload(*oracle);
            timings<<name<<','<<pair<<','<<(pinned?"pinned":"pageable")<<",3,"<<3*outputs<<','<<r.wall<<','<<r.first<<','<<r.proposals<<','<<r.verified<<','<<r.replayed<<','<<s.pinned_exports<<','<<s.fallbacks<<','<<exact<<'\n';timings.flush();
            phases<<name<<','<<pair<<','<<(pinned?"pinned":"pageable")<<','<<s.exports<<','<<s.recurrent_bytes<<','<<s.total_ms<<','<<s.recurrent_ms<<','<<s.allocate_ms<<','<<s.register_ms<<','<<s.submit_ms<<','<<s.fence_ms<<','<<s.pool_hits<<','<<s.pool_bytes<<','<<s.process_peak_bytes<<'\n';phases.flush();gate(name,"timed_full_state",exact);
        }
        std::cout<<"RECURRENT_EXPORT fixture="<<name<<" PASS\n";
    }
    coordinator.close();configure_export(false);
    require(Exl3RecurrentPinBudget::snapshot()[0]==0,"retained recurrent pin leak");
    std::cout<<"PASS_REAL_DFLASH_RECURRENT_EXPORT\n";
}
