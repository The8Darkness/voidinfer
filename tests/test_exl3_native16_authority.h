#pragma once

// Cached-input W16 authority; OSCAR L0 greedy proposals, TurboAngle L1 screens.
// DFlash ring is restored/committed, but no DFlash proposal generator is called.
void run_native16_authority(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::array<std::uint16_t*,5>& staging,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Root=std::shared_ptr<const Request>;
    using Frontiers=ninfer::exl3::Exl3HierarchyFrontiers;
    using L1=ninfer::exl3::Exl3TurboAngleL1Context;
    using Registry=ninfer::exl3::Exl3HostResidentSet;
    using Clock=std::chrono::steady_clock;
    const auto ms=[](auto a,auto b){return std::chrono::duration<double,std::milli>(b-a).count();};
    const std::filesystem::path output=env("NINFER_NATIVE16_AUTHORITY_OUT");
    require(!output.empty() && !std::filesystem::exists(output),"native16 authority new directory");
    std::filesystem::create_directories(output);
    std::ofstream rows(output/"runs.csv"),releases(output/"releases.csv"),checks(output/"correctness.csv");
    require(rows.good() && releases.good() && checks.good(),"native16 authority evidence files");
    rows << "fixture,pair,order,arm,wall_ms,tps,setup_ms,l2_ms,registry_ms,sync_ms,teardown_ms,l0_rows,l0_replay,l0_rejections,l1_rows,l1_replay,l1_rejections,l2_windows,l2_rejections,verified,replayed,committed,discarded,l2_state_exports,l2_ring_exports,native_calls,h2d,d2h,persistent_bytes,registry_final_pages,registry_peak_pages,native_hist,survival_hist,exact\n";
    releases << "fixture,pair,arm,release,rows,elapsed_ms,gap_ms\n";
    checks << "fixture,case,pass\n";
    const auto gate=[&](const char* fixture,const std::string& name,bool pass) {
        checks << fixture << ',' << name << ',' << pass << '\n';checks.flush();
        require(checks.good(),"native16 authority evidence flush");require(pass,"native16 authority gate: "+name);
    };
    require(target.max_context()==4352 && code.size()>=4096 && prose.size()>=4096,"native16 authority fixture extent");
    for(auto p:staging) require(p!=nullptr,"native16 authority stage16 pointer");
    struct Environment {
        std::vector<std::pair<std::string,std::string>> saved;
        void set(const char* key,const char* value) {saved.emplace_back(key,env(key));_putenv_s(key,value);}
        ~Environment(){for(auto i=saved.rbegin();i!=saved.rend();++i)_putenv_s(i->first.c_str(),i->second.c_str());}
    } settings;
    settings.set("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7","0");
    settings.set("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC","0");
    settings.set("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC","0");
    settings.set("NINFER_EXL3_NUMERIC_ATTENTION_TILED","0");
    settings.set("NINFER_EXL3_EXACT_HOST_KV","1");settings.set("NINFER_EXL3_OSCAR_L0_ONLY","0");
    settings.set("NINFER_EXL3_NATIVE_CONTINUATION16","0");
    MEMORYSTATUSEX physical{};physical.dwLength=sizeof(physical);
    require(GlobalMemoryStatusEx(&physical)!=0 && physical.ullAvailPhys>(8ULL<<30),"native16 authority host reserve");
    Registry registry(physical.ullTotalPhys-(8ULL<<30),8ULL<<30);
    Root pinned_input;std::uint64_t page_bytes=0,peak_pages=0,request_id=16000;
    const auto pin=[&](const std::vector<Root>& active) {
        std::vector<Root> roots;if(pinned_input)roots.push_back(pinned_input);
        roots.insert(roots.end(),active.begin(),active.end());Registry::Snapshot snapshot;
        for(const auto& r:roots)snapshot.owners.push_back(r);
        Request::visit_host_allocations(roots,[&](const void* p,std::size_t n){snapshot.add(p,n);},false);
        const auto update=registry.replace(std::move(snapshot));page_bytes=update.page_bytes;peak_pages=std::max(peak_pages,page_bytes);
    };
    const auto append=[](auto& a,const auto& b){a.insert(a.end(),b.begin(),b.end());};
    const auto root_bytes=[](const Root& root) {
        std::vector<std::uint8_t> bytes;const std::array<Root,1> roots{root};
        Request::visit_host_allocations(roots,[&](const void* p,std::size_t n) {
            const auto* b=static_cast<const std::uint8_t*>(p);bytes.insert(bytes.end(),b,b+n);
        },false);return bytes;
    };
    struct Result {
        Root final;std::vector<std::int64_t> tokens;std::array<std::vector<std::uint16_t>,5> taps;
        std::vector<int> release_rows;std::vector<double> release_ms;
        std::array<std::uint64_t,17> hist{},survival{};
        double wall=0,setup=0,l2=0,registration=0,sync=0,teardown=0;
        std::uint64_t l0_rows=0,l0_replay=0,l1_rows=0,l1_replay=0,l1_rejections=0,l2_rejections=0,windows=0,verified=0,replayed=0,discarded=0,native=0,h2d=0,d2h=0;
        std::size_t persistent=0;std::uint64_t registry_final=0,registry_peak=0;
        bool shape_ok=true,canceled=false,stale=false;
    };
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const char* name=fixture.first;Root initial;
        std::vector<std::int64_t> oracle_tokens;std::array<std::vector<std::uint16_t>,5> oracle_taps;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> oracle_state;
        {
            auto ctx=target.create_context(true);ctx->prepare_continuation(8);
            auto input=Request::initialize(*ctx,std::span<const std::int64_t>(fixture.second->data(),4096),1024);
            initial=input->compact_draft(draft,staging);pinned_input=initial;pin({});
            for(int i=0;i<128;++i) {
                const auto token=sample_target(*ctx);oracle_tokens.push_back(token);ctx->decode(token);
                const auto t=ctx->exact_tap_rows_host();for(int j=0;j<5;++j)append(oracle_taps[j],t[j]);
            }
            oracle_state=ctx->export_exact_host_state();
        }
        const auto immutable_input=root_bytes(initial);
        initial->restore_draft(draft,staging);const auto initial_ring=draft.export_host_ring(nullptr,false);
        const auto run=[&](bool native16,bool correctness,int cancel_phase=0) {
            // Draft props/ring restoration is preparation outside the paired interval.
            initial->restore_draft(draft,staging);cuda_check(cudaDeviceSynchronize(),"native16 authority reset draft");
            Result result;result.tokens.reserve(128);result.release_rows.reserve(128);result.release_ms.reserve(128);
            peak_pages=page_bytes;const auto started=Clock::now();
            Environment exact_flags;exact_flags.set("NINFER_EXL3_NATIVE_CONTINUATION16",native16?"1":"0");
            auto exact=std::shared_ptr<Exl3TextContext>(target.create_context(true));
            exact->prepare_continuation(native16?16:8);
            std::unique_ptr<Exl3TextContext> l0,l1_native;
            {
                Environment light;
                light.set("NINFER_EXL3_NATIVE_CONTINUATION16","0");
                for(const char* key:{"NINFER_EXL3_WIDE_PREFILL","NINFER_EXL3_PREFILL_STAGED_REDUCTION",
                        "NINFER_EXL3_SHARED_ACCUM","NINFER_EXL3_SHARED_TRANSFORM","NINFER_EXL3_SHARED_LAYER_SCRATCH","NINFER_EXL3_GDN_WIDE_SLAB"})light.set(key,"0");
                light.set("NINFER_EXL3_EXACT_HOST_KV","0");light.set("NINFER_EXL3_OSCAR_L0_ONLY","1");
                l0=target.create_context(true);l1_native=target.create_context(true);
                require(l0->try_enable_oscar_from_environment() && l1_native->try_enable_oscar_from_environment(),"native16 authority OSCAR contexts");
                l0->prepare_continuation(8);
            }
            auto current=initial;const auto before=exact->host_kv_stats();
            exact->restore_exact_host_state(*current->state());l0->restore_oscar_host_state(*current->state());
            auto l1=std::make_unique<L1>(std::move(l1_native),current->state());
            const auto exact_bytes=exact->persistent_bytes(),l0_bytes=l0->persistent_bytes(),l1_bytes=l1->context_bytes();
            result.persistent=exact_bytes+l0_bytes+l1_bytes;result.setup=ms(started,Clock::now());
            Frontiers frontier(++request_id,4096);
            while(result.tokens.size()<128) {
                const auto boundary=std::min<std::size_t>(16,128-result.tokens.size());
                std::vector<std::int64_t> pending;pending.reserve(boundary);
                while(pending.size()<boundary) {
                    const int count=static_cast<int>(std::min<std::size_t>(4,boundary-pending.size()));
                    const auto work=frontier.begin_l0(count);std::vector<std::int64_t> candidates;
                    for(int i=0;i<count;++i){const auto token=sample_target(*l0);candidates.push_back(token);l0->decode(token);++result.l0_rows;}
                    const auto screened=count==1?l1->screen_one(candidates[0]):l1->screen(candidates);
                    result.l1_rows+=screened.executed_rows;result.l1_replay+=screened.replay_rows;result.l1_rejections+=screened.rejected;
                    frontier.complete_l1(work,static_cast<int>(screened.authorized_tokens.size()),screened.rejected);
                    if(screened.rejected) {l0->restore_oscar_host_state(*current->state());for(auto t:pending)l0->decode(t);for(auto t:screened.authorized_tokens)l0->decode(t);result.l0_replay+=pending.size()+screened.authorized_tokens.size();}
                    append(pending,screened.authorized_tokens);
                }
                const auto completion=frontier.begin_l2();
                require(completion.rows==static_cast<int>(pending.size()) && !pending.empty(),"native16 authority L2 dependency");
                if(cancel_phase==1) {
                    frontier.cancel_tentative();result.canceled=true;result.stale=!frontier.current_completion(completion);
                    bool refused=false;try {frontier.complete_l2(completion,1);}catch(const std::logic_error&){refused=true;}
                    result.stale=result.stale && refused;
                    exact->restore_exact_host_state(*current->state());current->restore_draft(draft,staging);
                    result.shape_ok=result.shape_ok && exact->export_exact_host_state()->same_payload(*current->state()) &&
                        initial_ring->same_payload(*draft.export_host_ring(nullptr,false));break;
                }
                const auto verify_start=Clock::now();auto [next,verified]=current->verify_compact(
                    *exact,draft,staging,pending,{},nullptr,false,
                    current->committed_tap_binding(exact,request_id,result.windows+1));
                result.l2+=ms(verify_start,Clock::now());++result.windows;result.verified+=verified.verification_rows;
                result.replayed+=verified.replay_rows;result.native+=verified.native_invocations;
                result.l2_rejections+=verified.rejected;
                result.discarded+=pending.size()-verified.accepted;
                for(std::size_t m=0;m<17;++m)result.hist[m]+=verified.native_batch_hist[m];
                int segment_rows=0;
                for(const auto& segment:verified.committed_tap_segments) {
                    result.shape_ok=result.shape_ok && segment.destination_first==segment_rows &&
                        segment.rows>0 && segment.logical_first==current->state()->position()+segment_rows &&
                        segment.logical_first==segment.source_partition_first+segment.source_first;
                    segment_rows+=segment.rows;
                }
                result.shape_ok=result.shape_ok &&
                    segment_rows==static_cast<int>(verified.committed_tokens.size());
                const auto* d2d=std::getenv("NINFER_EXL3_COMMITTED_TAP_D2D");
                if(d2d && std::string_view(d2d)=="1")
                    result.shape_ok=result.shape_ok && verified.committed_tap_d2d_bytes==
                        verified.committed_tokens.size()*5*5120*2;
                ++result.survival[verified.accepted];
                if(cancel_phase==2) {
                    frontier.cancel_tentative();result.canceled=true;result.stale=!frontier.current_completion(completion);
                    bool refused=false;try {frontier.complete_l2(completion,1);}catch(const std::logic_error&){refused=true;}
                    result.stale=result.stale && refused;
                    exact->restore_exact_host_state(*current->state());current->restore_draft(draft,staging);
                    result.shape_ok=result.shape_ok && exact->export_exact_host_state()->same_payload(*current->state()) &&
                        initial_ring->same_payload(*draft.export_host_ring(nullptr,false));break;
                }
                const auto sync=Clock::now();cuda_check(cudaDeviceSynchronize(),"native16 authority before publication");result.sync+=ms(sync,Clock::now());
                auto reg=Clock::now();pin({current,next});result.registration+=ms(reg,Clock::now());
                frontier.complete_l2(completion,static_cast<int>(verified.committed_tokens.size()));
                const auto publication=frontier.publish_authorized();
                result.shape_ok=result.shape_ok && publication.first==current->state()->position() && publication.rows==static_cast<int>(verified.committed_tokens.size());
                require(publication.rows>0 && publication.rows<=static_cast<int>(boundary),"native16 authority useful progress");
                append(result.tokens,verified.committed_tokens);
                if(correctness)for(int j=0;j<5;++j)append(result.taps[j],verified.committed_taps[j]);
                result.release_rows.push_back(publication.rows);result.release_ms.push_back(ms(started,Clock::now()));
                current=std::move(next);reg=Clock::now();pin({current});result.registration+=ms(reg,Clock::now());
                l1->rebase_delta(current->state());l0->restore_oscar_host_state(*current->state());frontier.rebase();
            }
            std::uint64_t counted_calls=0,counted_rows=0,counted_windows=0;
            for(std::size_t m=0;m<17;++m){counted_calls+=result.hist[m];counted_rows+=m*result.hist[m];counted_windows+=result.survival[m];}
            result.final=current;result.shape_ok=result.shape_ok && exact_bytes==exact->persistent_bytes() && l0_bytes==l0->persistent_bytes() && l1_bytes==l1->context_bytes() &&
                counted_calls==result.native && counted_rows==result.verified+result.replayed && counted_windows==result.windows;
            if(!cancel_phase) result.shape_ok=result.shape_ok && (native16?result.hist[16]>0:result.hist[16]==0);
            const auto after=exact->host_kv_stats();result.h2d=after.h2d_bytes-before.h2d_bytes;result.d2h=after.d2h_bytes-before.d2h_bytes;
            result.registry_final=page_bytes;result.registry_peak=peak_pages;
            const auto teardown=Clock::now();l1.reset();l0.reset();exact.reset();current.reset();pin({});
            cuda_check(cudaDeviceSynchronize(),"native16 authority teardown");result.teardown=ms(teardown,Clock::now());result.wall=ms(started,Clock::now());return result;
        };
        const auto equal=[&](const Result& r,bool full) {
            bool exact=r.tokens==oracle_tokens && r.final->state()->same_payload(*oracle_state) && r.shape_ok;
            if(full)exact=exact && r.taps==oracle_taps;
            // Independent M1 target taps use the SAME release projection partitions.
            initial->restore_draft(draft,staging);int offset=0;
            for(int count:r.release_rows) {
                std::array<const std::uint16_t*,5> pointers{};
                for(int j=0;j<5;++j) {cuda_check(cudaMemcpy(staging[j],oracle_taps[j].data()+static_cast<std::size_t>(offset)*5120,
                    static_cast<std::size_t>(count)*5120*2,cudaMemcpyHostToDevice),"native16 authority oracle tap stage");pointers[j]=staging[j];}
                draft.commit_prefill_block(pointers.data(),count,4096+offset);offset+=count;
            }
            const auto expected_ring=draft.export_host_ring(nullptr,false);r.final->restore_draft(draft,staging);
            exact=exact && expected_ring->same_payload(*draft.export_host_ring(nullptr,false));return exact;
        };
        {
            const auto a=run(false,true);gate(name,"native8_correctness",equal(a,true));
            const auto b=run(true,true);gate(name,"native16_correctness",equal(b,true));
            a.final->restore_draft(draft,staging);const auto a_ring=draft.export_host_ring(nullptr,false);
            b.final->restore_draft(draft,staging);
            gate(name,"paired_release_state_taps_ring",a.tokens==b.tokens && a.taps==b.taps && a.release_rows==b.release_rows &&
                a.final->state()->same_payload(*b.final->state()) && a_ring->same_payload(*draft.export_host_ring(nullptr,false)));
        }
        for(int phase:{1,2}) {const auto r=run(true,true,phase);gate(name,phase==1?"pre_L2_cancel":"post_L2_cancel",
            r.canceled && r.stale && r.shape_ok && r.tokens.empty() && r.final.get()==initial.get() && root_bytes(initial)==immutable_input);}
        for(int pair=0;pair<6;++pair)for(int order=0;order<2;++order) {
            const bool arm=pair%2?order==0:order==1;const auto r=run(arm,false);const bool exact=equal(r,false);
            rows << name << ',' << pair << ',' << (pair%2?"BA":"AB") << ',' << (arm?"native16":"native8") << ','
                << r.wall << ',' << 128000.0/r.wall << ',' << r.setup << ',' << r.l2 << ',' << r.registration << ',' << r.sync << ',' << r.teardown << ','
                << r.l0_rows << ',' << r.l0_replay << ",NA," << r.l1_rows << ',' << r.l1_replay << ',' << r.l1_rejections << ',' << r.windows << ',' << r.l2_rejections << ',' << r.verified << ','
                << r.replayed << ',' << r.tokens.size() << ',' << r.discarded << ',' << r.windows << ',' << r.windows << ',' << r.native << ',' << r.h2d << ',' << r.d2h << ','
                << r.persistent << ',' << r.registry_final << ',' << r.registry_peak << ',';
            for(int i=0;i<17;++i)rows << (i?";":"") << i << ':' << r.hist[i];rows << ',';
            for(int i=0;i<17;++i)rows << (i?";":"") << i << ':' << r.survival[i];rows << ',' << exact << '\n';
            for(std::size_t i=0;i<r.release_ms.size();++i)releases << name << ',' << pair << ',' << (arm?"native16":"native8") << ',' << i+1 << ','
                << r.release_rows[i] << ',' << r.release_ms[i] << ',' << r.release_ms[i]-(i?r.release_ms[i-1]:0) << '\n';
            rows.flush();releases.flush();require(rows.good() && releases.good(),"native16 authority timed evidence");require(exact,"native16 authority timed exact gate");
        }
        gate(name,"immutable_input_after_all_arms",root_bytes(initial)==immutable_input);
        pinned_input.reset();pin({});
    }
    registry.close();std::cout << "NATIVE16_AUTHORITY PASS W=16 L0_block=4 fixtures=2 pairs_each=6 cached_input=1 draft_proposals=0\n";
}
