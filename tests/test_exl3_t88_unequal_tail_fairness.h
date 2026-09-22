#pragma once
#include "test_exl3_host_residency.h"
#include <atomic>
#include <mutex>
#include <thread>

struct T88RequestTrace {
    std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> root;
    std::vector<std::int64_t> published;
    std::vector<double> releases_ms;
    double queue_wait_ms=0,restore_ms=0,p95_interval_ms=0;
    int worker=-1;
    std::uint64_t verification_rows=0,executed_rows=0,replay_rows=0;
    std::uint64_t committed_rows=0,native_invocations=0,root_restores=0;
};

struct T88ArmTrace {
    std::array<T88RequestTrace,8> requests;
    double makespan_ms=0,residency_update_ms=0;
    std::size_t ready_owned_bytes=0,end_owned_bytes=0,min_free_bytes=0;
    Exl3HostResidency ready_residency,end_residency;
};

double t88_p95(std::vector<double> values) {
    require(!values.empty(),"T88 p95 empty");
    std::sort(values.begin(),values.end());
    return values[static_cast<std::size_t>(std::ceil(0.95*values.size()))-1];
}

void run_t88_unequal_tail_fairness(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose,const std::filesystem::path& output) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Cache=ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using Identity=ninfer::exl3::Exl3VeriCacheServingIdentity;
    constexpr std::size_t kPrefix=3840;
    constexpr int kWarmups=1,kMeasured=4;
    constexpr std::array<int,8> tails{32,64,96,128,160,192,224,256};
    constexpr std::array<int,8> budgets{2,4,6,8,2,4,6,8};
    require(target.max_context()>=4352&&code.size()>=4096&&prose.size()>=2048,
            "T88 serving fixture extent");
    require(!output.empty()&&!std::filesystem::exists(output),"T88 output must be new");
    std::filesystem::create_directories(output);
    const auto reserve_gib=std::stoull(env("NINFER_T88_RESERVE_GIB"));
    const auto cache_mib=std::stoull(env("NINFER_T88_CACHE_MIB"));
    require(reserve_gib>=8&&cache_mib>=1024,"T88 memory policy extent");
    const std::uint64_t reserve=reserve_gib*(1ULL<<30);
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    require(GlobalMemoryStatusEx(&memory)!=0&&memory.ullTotalPhys>reserve&&
            memory.ullAvailPhys>reserve,"T88 physical memory reserve");
    const auto available=[] {
        MEMORYSTATUSEX value{};value.dwLength=sizeof(value);
        require(GlobalMemoryStatusEx(&value)!=0,"T88 physical memory query");
        return value.ullAvailPhys;
    };
    const Identity identity{env("NINFER_T88_NAMESPACE"),env("NINFER_T88_ARTIFACT_ID"),
        env("NINFER_T88_TOKENIZER_ID"),env("NINFER_T88_CONFIGURATION_ID"),
        env("NINFER_T88_MODALITY_ID")};
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    Cache cache(Cache::Policy{16,64,cache_mib*(1ULL<<20),reserve},identity);
    ninfer::exl3::Exl3HostResidentSet residency(memory.ullTotalPhys-reserve,reserve);
    auto context_a=target.create_context(true),context_b=target.create_context(true);
    context_a->prepare_continuation(8);context_b->prepare_continuation(8);
    Exl3TextContext* contexts[2]{context_a.get(),context_b.get()};
    TargetGraphC2Stream streams[2];
    const auto owned_bytes=[&] {
        std::size_t result=0;for(auto* context:contexts)
            result+=context->persistent_bytes()+context->continuation_bytes();
        return result;
    };
    std::vector<std::int64_t> common(code.begin(),code.begin()+kPrefix);
    std::array<std::vector<std::int64_t>,8> prompts;
    for(int request=0;request<8;++request) {
        prompts[request]=common;const int rows=tails[request];
        if((request&1)==0) {
            const auto end=code.begin()+4096-(request/2)*256;
            prompts[request].insert(prompts[request].end(),end-rows,end);
        } else {
            const auto begin=prose.begin()+(request/2)*512;
            prompts[request].insert(prompts[request].end(),begin,begin+rows);
        }
        require(prompts[request].size()==kPrefix+static_cast<std::size_t>(rows),
                "T88 prompt extent");
    }
    struct PinResult {ninfer::exl3::Exl3HostResidentSet::Stats update;Exl3HostResidency observed;};
    const auto pin=[&](const std::array<std::shared_ptr<const Request>,8>& active,bool observe) {
        auto roots=cache.roots();for(const auto& root:active)if(root)roots.push_back(root);
        ninfer::exl3::Exl3HostResidentSet::Snapshot snapshot;
        for(const auto& root:roots)snapshot.owners.push_back(root);
        Request::visit_host_allocations(roots,[&](const void* data,std::size_t bytes){
            snapshot.add(data,bytes);},false);
        const auto update=residency.replace(std::move(snapshot));
        Exl3HostResidency observed;
        if(observe) {
            Exl3HostResidencyProbe probe;
            Request::visit_host_allocations(roots,[&](const void* data,std::size_t bytes){
                probe.add(data,bytes);},false);
            observed=probe.measure();
            require(observed.resident_tensor_bytes==observed.allocated_union_bytes&&
                    observed.locked_tensor_bytes==observed.allocated_union_bytes,
                    "T88 authoritative payload not resident and locked");
        }
        return PinResult{update,observed};
    };
    auto fill=cache.prepare(*context_a,common,kPrefix,available(),1024);
    cuda_check(cudaDeviceSynchronize(),"T88 cache fill completion");
    require(!fill.metrics.cache_hit&&fill.metrics.admitted&&
            fill.metrics.executed_prompt_tokens==kPrefix,"T88 cache fill gate");
    fill.request.reset();
    std::array<std::shared_ptr<const Request>,8> initial;
    std::array<double,8> preparation_ms{};
    std::array<Cache::Metrics,8> preparation{};
    for(int request=0;request<8;++request) {
        auto* context=contexts[request&1];const auto started=std::chrono::steady_clock::now();
        auto prepared=cache.prepare(*context,prompts[request],kPrefix,available(),1024);
        cuda_check(cudaDeviceSynchronize(),"T88 request preparation completion");
        preparation_ms[request]=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count();
        preparation[request]=prepared.metrics;initial[request]=std::move(prepared.request);
        require(preparation[request].cache_hit&&
                preparation[request].reused_prompt_tokens==kPrefix&&
                preparation[request].executed_prompt_tokens==static_cast<std::size_t>(tails[request]),
                "T88 cached preparation accounting");
    }
    const auto initial_lock=pin(initial,true);
    require(initial_lock.observed.resident_tensor_bytes<memory.ullTotalPhys-reserve,
            "T88 C8 host reserve");
    std::ofstream batches(output/"batches.csv"),requests_out(output/"requests.csv"),
        preparation_out(output/"preparation.csv"),cancellation(output/"cancellation.csv"),
        memory_out(output/"memory.csv");
    require(batches.good()&&requests_out.good()&&preparation_out.good()&&
            cancellation.good()&&memory_out.good(),"T88 evidence open");
    preparation_out<<"request,tail_rows,output_budget,preparation_ms,reused_prompt_tokens,executed_prompt_tokens,cache_hit\n";
    for(int request=0;request<8;++request) preparation_out<<request<<','<<tails[request]
        <<','<<budgets[request]<<','<<preparation_ms[request]<<','
        <<preparation[request].reused_prompt_tokens<<','
        <<preparation[request].executed_prompt_tokens<<",1\n";
    batches<<"rep,warmup,order,arm,schedule,logical_requests,physical_limit,queue_high_water,total_outputs,makespan_ms,aggregate_output_s,residency_update_ms,queue_p95_ms,final_p95_ms,interval_p95_ms,min_free_bytes,ready_owned_bytes,end_owned_bytes,ready_resident_bytes,ready_locked_bytes,end_resident_bytes,end_locked_bytes,exact\n";
    requests_out<<"rep,warmup,arm,request,tail_rows,output_budget,worker,initial_pair,queue_wait_ms,restore_ms,first_release_ms,final_release_ms,p95_interval_ms,verification_rows,executed_rows,replay_rows,committed_rows,native_invocations,root_restores,published_tokens,exact\n";
    cancellation<<"case,survivor,canceled,verified_rows_discarded,survivor_publications,canceled_publications,survivor_exact,canceled_root_exact,pass\n";
    memory_out<<"phase,cache_payload_bytes,cache_identity_bytes,cache_accounted_bytes,resident_payload_bytes,locked_payload_bytes,resident_page_bytes,new_locked_bytes,unlocked_bytes,physical_reserve_bytes,device_free_bytes\n";
    std::size_t total_device=0,initial_free=0;
    cuda_check(cudaMemGetInfo(&initial_free,&total_device),"T88 initial device memory");
    memory_out<<"active_c8,"<<cache.storage_stats().payload_allocated_bytes<<','
        <<cache.storage_stats().identity_allocated_bytes<<','<<cache.storage_stats().accounted_bytes
        <<','<<initial_lock.observed.resident_tensor_bytes<<','
        <<initial_lock.observed.locked_tensor_bytes<<','<<initial_lock.update.page_bytes<<','
        <<initial_lock.update.new_locked_bytes<<','<<initial_lock.update.unlocked_bytes<<','
        <<reserve<<','<<initial_free<<'\n';
    const auto run_arm=[&](int physical_limit,int rep) {
        T88ArmTrace result;std::array<std::shared_ptr<const Request>,8> current=initial;
        const auto ready=pin(current,true);result.ready_residency=ready.observed;
        result.ready_owned_bytes=owned_bytes();result.min_free_bytes=std::numeric_limits<std::size_t>::max();
        std::array<int,8> schedule{};const int first=(2*rep)&7;
        for(int index=0;index<8;++index)schedule[index]=(first+index)&7;
        const auto batch_started=std::chrono::steady_clock::now();
        std::atomic<int> next_slot{0},ready_workers{0};std::atomic<bool> go{false};
        std::mutex publication_mutex;std::array<std::exception_ptr,2> errors;
        const auto worker=[&](int physical) {
            try {
                ready_workers.fetch_add(1,std::memory_order_release);
                while(!go.load(std::memory_order_acquire))std::this_thread::yield();
                while(true) {
                    const int slot=next_slot.fetch_add(1);if(slot>=8)break;
                    const int request=schedule[slot];auto& trace=result.requests[request];
                    trace.worker=physical;trace.queue_wait_ms=std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-batch_started).count();
                    auto restore_started=std::chrono::steady_clock::now();
                    contexts[physical]->restore_exact_host_state(*current[request]->state(),streams[physical].value);
                    cuda_check(cudaStreamSynchronize(streams[physical].value),"T88 request restore");
                    trace.restore_ms=std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-restore_started).count();
                    auto pending=sample_target(*contexts[physical],streams[physical].value);
                    for(int ordinal=0;ordinal<budgets[request];++ordinal) {
                        const std::array<std::int64_t,1> token{pending};
                        std::shared_ptr<const Request> parent;
                        {std::lock_guard lock(publication_mutex);parent=current[request];}
                        auto [updated,verified]=parent->verify(*contexts[physical],token,{},streams[physical].value);
                        require(verified.committed_tokens.size()==1&&
                                verified.committed_tokens.front()==pending&&verified.accepted==1&&
                                !verified.rejected&&!verified.stopped,"T88 width-one decision");
                        {
                            std::lock_guard lock(publication_mutex);current[request]=std::move(updated);
                            const auto resident_update=pin(current,false);
                            result.residency_update_ms+=resident_update.update.elapsed_ms;
                            const double stamp=std::chrono::duration<double,std::milli>(
                                std::chrono::steady_clock::now()-batch_started).count();
                            trace.published.push_back(pending);trace.releases_ms.push_back(stamp);
                        }
                        trace.verification_rows+=verified.verification_rows;
                        trace.executed_rows+=verified.executed_rows;trace.replay_rows+=verified.replay_rows;
                        trace.committed_rows+=verified.committed_tokens.size();
                        trace.native_invocations+=verified.native_invocations;trace.root_restores+=verified.root_restores;
                        if(ordinal+1<budgets[request])pending=sample_target(*contexts[physical],streams[physical].value);
                    }
                }
            } catch(...) {errors[physical]=std::current_exception();}
        };
        std::thread workers[2];for(int physical=0;physical<physical_limit;++physical)
            workers[physical]=std::thread(worker,physical);
        while(ready_workers.load(std::memory_order_acquire)!=physical_limit)std::this_thread::yield();
        go.store(true,std::memory_order_release);
        for(int physical=0;physical<physical_limit;++physical)workers[physical].join();
        for(int physical=0;physical<physical_limit;++physical)if(errors[physical])std::rethrow_exception(errors[physical]);
        result.makespan_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-batch_started).count();
        for(int request=0;request<8;++request) {
            auto& trace=result.requests[request];trace.root=current[request];
            require(static_cast<int>(trace.published.size())==budgets[request]&&
                    trace.verification_rows==static_cast<std::uint64_t>(budgets[request])&&
                    trace.executed_rows==static_cast<std::uint64_t>(budgets[request])&&
                    trace.replay_rows==0&&trace.committed_rows==static_cast<std::uint64_t>(budgets[request])&&
                    trace.native_invocations==static_cast<std::uint64_t>(budgets[request])&&trace.root_restores==0,
                    "T88 request work accounting");
            std::vector<double> intervals;double previous=trace.queue_wait_ms;
            for(const auto stamp:trace.releases_ms){intervals.push_back(stamp-previous);previous=stamp;}
            trace.p95_interval_ms=t88_p95(std::move(intervals));
        }
        const auto end=pin(current,true);result.end_residency=end.observed;
        result.end_owned_bytes=owned_bytes();std::size_t free_bytes=0;
        cuda_check(cudaMemGetInfo(&free_bytes,&total_device),"T88 end device memory");
        result.min_free_bytes=free_bytes;
        require(result.ready_owned_bytes==result.end_owned_bytes&&free_bytes>=(1ULL<<30),
                "T88 logical/device memory gate");
        return result;
    };
    std::array<int,8> initial_pair_counts{};
    for(int rep=0;rep<kWarmups+kMeasured;++rep) {
        T88ArmTrace c1,c2;const bool c2_first=(rep&1)!=0;
        for(int arm=0;arm<2;++arm) {
            const int physical=(c2_first?arm==0:arm==1)?2:1;
            auto trace=run_arm(physical,rep);if(physical==2)c2=std::move(trace);else c1=std::move(trace);
        }
        for(int request=0;request<8;++request) require(
            c1.requests[request].published==c2.requests[request].published&&
            c1.requests[request].root->token_suffix()==c2.requests[request].root->token_suffix()&&
            c1.requests[request].root->same_taps(*c2.requests[request].root)&&
            c1.requests[request].root->state()->same_payload(*c2.requests[request].root->state()),
            "T88 matched authoritative request");
        std::ostringstream schedule_text;for(int index=0;index<8;++index){if(index)schedule_text<<'|';schedule_text<<(((2*rep)+index)&7);}
        const auto write_arm=[&](const char* name,const T88ArmTrace& trace,int physical) {
            std::vector<double> queue,finals,intervals;int total_outputs=0;
            for(int request=0;request<8;++request){const auto& row=trace.requests[request];
                queue.push_back(row.queue_wait_ms);finals.push_back(row.releases_ms.back());
                intervals.push_back(row.p95_interval_ms);total_outputs+=budgets[request];
                const int relative=(request-((2*rep)&7)+8)&7;
                requests_out<<rep<<','<<(rep<kWarmups)<<','<<name<<','<<request<<','
                    <<tails[request]<<','<<budgets[request]<<','<<row.worker<<','<<(relative<2)<<','
                    <<row.queue_wait_ms<<','<<row.restore_ms<<','<<row.releases_ms.front()<<','
                    <<row.releases_ms.back()<<','<<row.p95_interval_ms<<','<<row.verification_rows<<','
                    <<row.executed_rows<<','<<row.replay_rows<<','<<row.committed_rows<<','
                    <<row.native_invocations<<','<<row.root_restores<<','
                    <<target_graph_c2_verifier_ids(row.published)<<",1\n";
            }
            batches<<rep<<','<<(rep<kWarmups)<<','<<(c2_first?"C2C1":"C1C2")<<','<<name<<','
                <<schedule_text.str()<<",8,"<<physical<<','<<(8-physical)<<','<<total_outputs<<','
                <<trace.makespan_ms<<','<<1000.0*total_outputs/trace.makespan_ms<<','
                <<trace.residency_update_ms<<','<<t88_p95(queue)<<','<<t88_p95(finals)<<','
                <<t88_p95(intervals)<<','<<trace.min_free_bytes<<','<<trace.ready_owned_bytes<<','
                <<trace.end_owned_bytes<<','<<trace.ready_residency.resident_tensor_bytes<<','
                <<trace.ready_residency.locked_tensor_bytes<<','<<trace.end_residency.resident_tensor_bytes<<','
                <<trace.end_residency.locked_tensor_bytes<<",1\n";
        };
        write_arm("C1",c1,1);write_arm("C2",c2,2);
        if(rep>=kWarmups){const int first=(2*rep)&7;++initial_pair_counts[first];++initial_pair_counts[(first+1)&7];}
    }
    require(initial_pair_counts==std::array<int,8>{1,1,1,1,1,1,1,1},
            "T88 initial pair rotation");
    const auto queued_state=initial[7]->state();const auto queued_tokens=initial[7]->token_suffix();
    const bool queued_exact=initial[7]->state()==queued_state&&initial[7]->token_suffix()==queued_tokens;
    cancellation<<"queued,NONE,7,0,0,0,1,"<<queued_exact<<','<<queued_exact<<'\n';
    require(queued_exact,"T88 queued cancellation");
    (void)pin(initial,false);
    contexts[0]->restore_exact_host_state(*initial[0]->state(),streams[0].value);
    const auto control_pending=sample_target(*contexts[0],streams[0].value);
    const std::array<std::int64_t,1> control_token{control_pending};
    auto [control_root,control_verified]=initial[0]->verify(*contexts[0],control_token,{},streams[0].value);
    require(control_verified.committed_tokens.size()==1,"T88 cancellation control");
    for(int physical=0;physical<2;++physical) {
        contexts[physical]->restore_exact_host_state(*initial[physical]->state(),streams[physical].value);
        cuda_check(cudaStreamSynchronize(streams[physical].value),"T88 cancellation restore");
    }
    std::array<std::int64_t,2> pending{sample_target(*contexts[0],streams[0].value),
        sample_target(*contexts[1],streams[1].value)};
    std::array<std::shared_ptr<const Request>,2> updated;
    std::array<ninfer::exl3::Exl3OuterReferenceResult,2> verified;
    std::array<std::exception_ptr,2> errors;std::thread cancellation_workers[2];
    for(int physical=0;physical<2;++physical)cancellation_workers[physical]=std::thread([&,physical]{
        try{const std::array<std::int64_t,1> token{pending[physical]};auto result=initial[physical]->verify(
            *contexts[physical],token,{},streams[physical].value);updated[physical]=std::move(result.first);
            verified[physical]=std::move(result.second);}catch(...){errors[physical]=std::current_exception();}});
    for(auto& worker:cancellation_workers)worker.join();for(const auto& error:errors)if(error)std::rethrow_exception(error);
    contexts[1]->restore_exact_host_state(*initial[1]->state(),streams[1].value);
    cuda_check(cudaStreamSynchronize(streams[1].value),"T88 canceled root restore");
    auto published_roots=initial;published_roots[0]=updated[0];(void)pin(published_roots,false);
    const bool survivor_exact=updated[0]->token_suffix()==control_root->token_suffix()&&
        updated[0]->same_taps(*control_root)&&updated[0]->state()->same_payload(*control_root->state());
    const bool canceled_exact=contexts[1]->exact_host_state_resident(*initial[1]->state());
    const bool cancel_pass=survivor_exact&&canceled_exact&&verified[1].verification_rows==1;
    cancellation<<"prepublication,0,1,"<<verified[1].verification_rows<<",1,0,"
        <<survivor_exact<<','<<canceled_exact<<','<<cancel_pass<<'\n';
    require(cancel_pass,"T88 prepublication cancellation");
    const auto final_lock=pin(initial,true);std::size_t final_free=0;
    cuda_check(cudaMemGetInfo(&final_free,&total_device),"T88 final device memory");
    const auto storage=cache.storage_stats();
    require(final_lock.observed.resident_tensor_bytes==initial_lock.observed.resident_tensor_bytes&&
            final_lock.observed.locked_tensor_bytes==initial_lock.observed.locked_tensor_bytes&&
            storage.entries==1&&storage.accounted_bytes<=cache_mib*(1ULL<<20)&&final_free>=(1ULL<<30),
            "T88 final memory gate");
    memory_out<<"final_c8,"<<storage.payload_allocated_bytes<<','<<storage.identity_allocated_bytes<<','
        <<storage.accounted_bytes<<','<<final_lock.observed.resident_tensor_bytes<<','
        <<final_lock.observed.locked_tensor_bytes<<','<<final_lock.update.page_bytes<<','
        <<final_lock.update.new_locked_bytes<<','<<final_lock.update.unlocked_bytes<<','<<reserve<<','<<final_free<<'\n';
    batches.flush();requests_out.flush();preparation_out.flush();cancellation.flush();memory_out.flush();
    require(batches.good()&&requests_out.good()&&preparation_out.good()&&cancellation.good()&&memory_out.good(),
            "T88 evidence flush");
    residency.close();
    std::cout<<"T88_UNEQUAL_TAIL_FAIRNESS PASS logical_requests=8 physical_limit=2"
        <<" measured_pairs=4 total_outputs_per_arm=40 exact=1 cancellation=2 output="
        <<output.string()<<std::endl;
}
