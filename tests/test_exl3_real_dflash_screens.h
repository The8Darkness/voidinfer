#pragma once
// Bounded architecture comparator: actual pinned proposals, optional L0/L1
// screens, and the same final L2 authority. No approximate external publication.
void run_real_dflash_screens(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose){
    using namespace ninfer::exl3;
    using Root=std::shared_ptr<const Exl3VeriCacheRequest>;
    using Clock=std::chrono::steady_clock;
    const auto ms=[](auto t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();};
    const int prefix=512,outputs=32;
    const std::filesystem::path dir=env("NINFER_REAL_DFLASH_OUT");require(!std::filesystem::exists(dir),"preserve screens receipt");std::filesystem::create_directories(dir);
    std::ofstream timing(dir/"workloads.csv"),checks(dir/"checks.csv"),requests(dir/"requests.csv");
    timing<<"fixture,pair,arm,wall_ms,setup_ms,teardown_ms,proposals,l0_rows,l0_rejections,l1_rows,l1_replay,l1_rejections,l2_rows,l2_replay,l0_ms,l1_ms,rebase_ms,persistent_bytes,exact\n";
    requests<<"fixture,pair,arm,request,wall_ms,first_release_ms\n";checks<<"fixture,case,pass\n";
    const auto gate=[&](const char* f,const char* name,bool b){checks<<f<<','<<name<<','<<b<<'\n';checks.flush();require(b,name);};
    Exl3VeriCacheServingIdentity identity{"solo-real-screens","SC_6.00bpw_H6_V6","pinned-tokens","exact-B8-L2","text"};
    Exl3VeriCacheServingPrefixCache cache({2,64,2ULL<<30,8ULL<<30},identity);
    MEMORYSTATUSEX m{};m.dwLength=sizeof(m);require(GlobalMemoryStatusEx(&m),"screens memory");
    Exl3VeriCacheServingCoordinator coordinator(cache,{1,1,m.ullTotalPhys-(8ULL<<30),8ULL<<30});
    std::array<std::unique_ptr<DeviceBuffer>,5> allocation;std::array<std::uint16_t*,5> staging{};
    for(int i=0;i<5;++i){allocation[i]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[i]=static_cast<std::uint16_t*>(allocation[i]->get());}
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},std::pair{"prose",&prose}}){
        const char* name=fixture.first;Root initial;std::shared_ptr<const Exl3ExactHostState> oracle;std::vector<std::int64_t> expected;
        std::array<std::vector<std::uint16_t>,5> oracle_taps;
        {
            auto ctx=target.create_context(true);ctx->prepare_continuation(8);
            initial=Exl3VeriCacheRequest::initialize(*ctx,std::span<const std::int64_t>(fixture.second->data(),prefix),1024)->compact_draft(draft,staging);
            for(int n=0;n<outputs;++n){auto t=sample_target(*ctx);expected.push_back(t);ctx->decode(t);auto taps=ctx->exact_tap_rows_host();for(int i=0;i<5;++i)oracle_taps[i].insert(oracle_taps[i].end(),taps[i].begin(),taps[i].end());}
            oracle=ctx->export_exact_host_state();
        }
        for(int pair=0;pair<2;++pair)for(int order=0;order<2;++order){
            const bool screens=(pair%2)?order==0:order==1;
            const auto start=Clock::now();auto lane=std::make_unique<Exl3Dflash2Execution>(target.create_context(true),draft,identity.contract());
            std::unique_ptr<Exl3TextContext> l0;std::unique_ptr<Exl3TurboAngleL1Context> l1;
            if(screens){
                // Constructor-only context flags, restored before any L2 execution.
                std::vector<std::pair<std::string,std::string>> saved;
                const auto set=[&](const char* k,const char* v){saved.emplace_back(k,env(k));_putenv_s(k,v);};
                set("NINFER_EXL3_NATIVE_CONTINUATION16","0");set("NINFER_EXL3_PINNED_RECURRENT_EXPORT","0");
                for(const char* k:{"NINFER_EXL3_WIDE_PREFILL","NINFER_EXL3_PREFILL_STAGED_REDUCTION","NINFER_EXL3_SHARED_ACCUM","NINFER_EXL3_SHARED_TRANSFORM","NINFER_EXL3_SHARED_LAYER_SCRATCH","NINFER_EXL3_GDN_WIDE_SLAB","NINFER_EXL3_EXACT_HOST_KV"})set(k,"0");
                set("NINFER_EXL3_OSCAR_L0_ONLY","1");
                try {
                    l0=target.create_context(true);auto native=target.create_context(true);
                    require(l0->try_enable_oscar_from_environment() && native->try_enable_oscar_from_environment(),"screen OSCAR contexts");
                    l1=std::make_unique<Exl3TurboAngleL1Context>(std::move(native),initial->state());
                }catch(...){for(const auto& [k,v]:saved)_putenv_s(k.c_str(),v.c_str());throw;}
                for(const auto& [k,v]:saved)_putenv_s(k.c_str(),v.c_str());
            }
            const double setup=ms(start);const auto bytes=lane->persistent_bytes()+(screens?l0->persistent_bytes()+l1->context_bytes():0);
            std::uint64_t l0rows=0,l0reject=0,l1rows=0,l1replay=0,l1reject=0,l2rows=0,l2replay=0;double l0ms=0,l1ms=0,rebasems=0;
            bool exact=true;Root final;
            for(int request=0;request<3;++request){
                const auto begin=Clock::now();double first=0;final.reset();std::vector<std::int64_t> tokens;std::array<std::vector<std::uint16_t>,5> taps;std::vector<int> partitions;
                coordinator.admit(initial);lane->acquire(*coordinator.acquire());
                if(screens){auto t=Clock::now();l0->restore_oscar_host_state(*initial->state());if(request)l1->rebase(initial->state());rebasems+=ms(t);}
                while(tokens.size()<outputs){
                    auto proposed=lane->propose(std::min<int>(8,outputs-static_cast<int>(tokens.size())));
                    if(screens){
                        auto t=Clock::now();std::vector<std::int64_t> l0tokens;
                        for(auto candidate:proposed){auto actual=sample_target(*l0);l0->decode(actual);++l0rows;l0tokens.push_back(actual);if(actual!=candidate){++l0reject;break;}}
                        l0ms+=ms(t);t=Clock::now();
                        const auto screened=l0tokens.size()==1?l1->screen_one(l0tokens[0]):l1->screen(l0tokens);
                        proposed=screened.authorized_tokens;l1rows+=screened.executed_rows;l1replay+=screened.replay_rows;l1reject+=screened.rejected;l1ms+=ms(t);
                    }
                    auto pending=lane->verify(proposed);auto published=lane->publish(coordinator,pending);
                    l2rows+=pending.verification.verification_rows;l2replay+=pending.verification.replay_rows;
                    tokens.insert(tokens.end(),published.tokens.begin(),published.tokens.end());partitions.push_back(static_cast<int>(published.tokens.size()));
                    for(int i=0;i<5;++i)taps[i].insert(taps[i].end(),pending.verification.committed_taps[i].begin(),pending.verification.committed_taps[i].end());
                    if(!first)first=ms(begin);
                    if(screens){auto t=Clock::now();l1->rebase_delta(lane->lease().root->state());l0->restore_oscar_host_state(*lane->lease().root->state());rebasems+=ms(t);}
                }
                final=lane->lease().root;
                if(request==0){Exl3HostResidencyProbe probe;const std::array<Root,1> roots{final};Exl3VeriCacheRequest::visit_host_allocations(roots,[&](const void* p,std::size_t n){probe.add(p,n);},false);const auto measured=probe.measure();gate(name,"published_resident_locked",measured.allocated_union_bytes==measured.resident_tensor_bytes && measured.allocated_union_bytes==measured.locked_tensor_bytes);}
                auto lease=lane->lease();lane->release();coordinator.complete(lease);
                requests<<name<<','<<pair<<','<<(screens?"hierarchy":"direct")<<','<<request<<','<<ms(begin)<<','<<first<<'\n';requests.flush();
                // Complete values are checked outside the request interval, but
                // inside total workload. This is a bounded architectural screen.
                exact=exact && tokens==expected && taps==oracle_taps && final->state()->same_payload(*oracle);
                initial->restore_draft(draft,staging);int offset=0;
                for(int rows:partitions){std::array<const std::uint16_t*,5> ptr{};for(int i=0;i<5;++i){cuda_check(cudaMemcpy(staging[i],oracle_taps[i].data()+static_cast<std::size_t>(offset)*5120,rows*5120ULL*2,cudaMemcpyHostToDevice),"screens ring oracle");ptr[i]=staging[i];}draft.commit_prefill_block(ptr.data(),rows,prefix+offset);offset+=rows;}
                auto ring=draft.export_host_ring(nullptr,false);final->restore_draft(draft,staging);exact=exact && ring->same_payload(*draft.export_host_ring(nullptr,false));
                gate(name,"request_full_state_taps",exact);
            }
            const auto proposals=lane->stats().proposal_calls;const auto teardown=Clock::now();l1.reset();l0.reset();lane.reset();cuda_check(cudaDeviceSynchronize(),"screens teardown");const auto teardownms=ms(teardown);
            timing<<name<<','<<pair<<','<<(screens?"hierarchy":"direct")<<','<<ms(start)<<','<<setup<<','<<teardownms<<','<<proposals<<','<<l0rows<<','<<l0reject<<','<<l1rows<<','<<l1replay<<','<<l1reject<<','<<l2rows<<','<<l2replay<<','<<l0ms<<','<<l1ms<<','<<rebasems<<','<<bytes<<','<<exact<<'\n';timing.flush();
        }
    }
    coordinator.close();std::cout<<"PASS_REAL_DFLASH_L0_L1_L2_SCREEN\n";
}

