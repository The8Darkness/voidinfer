#pragma once
struct WidePrefillOperators {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit=Admission::target_wide_prefill;
    static constexpr std::size_t guard=128;
    static constexpr int sample_widths[]={1,2,15,16,17,31,32,33,127,128,129,255,256,257,511,512,513,767,768,769,1023,1024};
    const int width = env("NINFER_EXL3_PREFILL_STAGED_REDUCTION")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE64")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE128")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE256")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE512")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE1024")=="1" ? 1024 : 512) : 256) : 128) : 64) : 32) : 16;
    std::filesystem::path directory;
    std::ofstream csv;
    std::vector<std::string> seen;
    explicit WidePrefillOperators(const std::filesystem::path& path):directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),"wide output must be new");
        std::filesystem::create_directories(path);csv.open(path/"operators.csv");
        csv << "layer,operation,K,input,output,rows,elements,exact,dispatch\n";
    }
    template<class T> void save(const std::string& name,const std::vector<T>& data) {
        std::ofstream f(directory/name,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));require(f.good(),"wide raw evidence write");
    }
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        static_cast<WidePrefillOperators*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        require(x.rows==width,"wide capture needs configured rows");
        const auto tag=std::to_string(x.layer)+"-"+x.operation;
        require(std::find(seen.begin(),seen.end(),tag)==seen.end(),"duplicate wide projection");seen.push_back(tag);
        const int ni=x.metadata.in_features,no=x.metadata.out_features;
        cuda_check(cudaStreamSynchronize(x.stream),"wide input ready");
        std::vector<std::uint16_t> input(static_cast<std::size_t>(width)*ni);
        cuda_check(cudaMemcpy(input.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"wide input copy");
        _putenv_s("NINFER_EXL3_WIDE_PREFILL","0");
        Exl3CudaLinearWorkspace baseline(ni,no,width,false,false,false,false,false,true);
        _putenv_s("NINFER_EXL3_WIDE_PREFILL","1");
        Exl3CudaLinearWorkspace candidate(ni,no,width,false,false,false,false,false,true);
        Exl3CudaLinearWorkspace unowned(ni,no,width);
        Exl3CudaLinearWorkspace small(ni,no,8,false,false,false,false,false,true);
        require(!baseline.target_wide_prefill_candidate(x.metadata,width,admit) && !unowned.target_wide_prefill_candidate(x.metadata,width,admit) && !small.target_wide_prefill_candidate(x.metadata,width,admit),"wide ownership/latched/capacity guard");
        require(!candidate.target_wide_prefill_candidate(x.metadata,width,Admission::ordinary),"wide ordinary admission");
        for(int n=0;n<=width+1;++n)require(candidate.target_wide_prefill_candidate(x.metadata,n,admit)==(n>=2 && n<=width),"wide width/capacity coverage");
        for(int bad=0;bad<6;++bad) {
            auto md=x.metadata;if(bad==0)md.K=4;if(bad==1)md.mcg=true;if(bad==2)md.mul1=false;if(bad==3)md.has_bias=true;if(bad==4)md.in_features+=128;if(bad==5)md.out_features+=128;
            require(!candidate.target_wide_prefill_candidate(md,width,admit),"wide metadata admission");
        }
        if(seen.size()==1 && width>=512) {
            for(int limit:{512,1024}) if(width>=limit) {
                const auto key="NINFER_EXL3_PREFILL_WIDE"+std::to_string(limit);
                for(const char* flag:{"","0","2","true","01"}) {
                    _putenv_s(key.c_str(),flag);
                    Exl3CudaLinearWorkspace off(ni,no,width,false,false,false,false,false,true);
                    require(off.target_wide_prefill_candidate(x.metadata,limit/2,admit) &&
                        !off.target_wide_prefill_candidate(x.metadata,limit/2+1,admit),"wide512/1024 flag guard");
                }
                _putenv_s(key.c_str(),"1");
            }
        }
        if(seen.size()==1 && width>=256) {
            for(const char* flag:{"","0","2","true","01"}) {
                _putenv_s("NINFER_EXL3_PREFILL_WIDE256",flag);
                Exl3CudaLinearWorkspace off(ni,no,width,false,false,false,false,false,true);
                require(off.target_wide_prefill_candidate(x.metadata,128,admit) &&
                    !off.target_wide_prefill_candidate(x.metadata,129,admit),"wide256 exact flag1 required");
            }
            _putenv_s("NINFER_EXL3_PREFILL_WIDE256","1");
        }
        if(seen.size()==1 && width>=128) {
            for(const char* flag:{"","0","2","true","01"}) {
                _putenv_s("NINFER_EXL3_PREFILL_WIDE128",flag);
                Exl3CudaLinearWorkspace off(ni,no,width,false,false,false,false,false,true);
                require(off.target_wide_prefill_candidate(x.metadata,64,admit) &&
                    !off.target_wide_prefill_candidate(x.metadata,65,admit),"wide128 exact flag1 required");
            }
            _putenv_s("NINFER_EXL3_PREFILL_WIDE128","1");
        }
        if(seen.size()==1 && width==64) {
            for(const char* flag:{"","0","2","true","01"}) {
                _putenv_s("NINFER_EXL3_PREFILL_WIDE64",flag);
                Exl3CudaLinearWorkspace off(ni,no,width,false,false,false,false,false,true);
                require(off.target_wide_prefill_candidate(x.metadata,32,admit) &&
                    !off.target_wide_prefill_candidate(x.metadata,33,admit),"wide64 exact flag1 required");
            }
            _putenv_s("NINFER_EXL3_PREFILL_WIDE64","1");
        }
        if(seen.size()==1) {
            for(const char* flag:{"","0","2","true","01"}) {
                _putenv_s("NINFER_EXL3_WIDE_PREFILL",flag);
                Exl3CudaLinearWorkspace off(ni,no,width,false,false,false,false,false,true);
                require(!off.target_wide_prefill_candidate(x.metadata,width,admit),"wide exact flag1 required");
            }
            _putenv_s("NINFER_EXL3_WIDE_PREFILL","1");
        }
        const std::size_t extent=static_cast<std::size_t>(width)*no;
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555),ref(initial),got(initial),repeat(initial);
        DeviceBuffer a(initial.size()*2),b(initial.size()*2);
        auto* pa=static_cast<std::uint16_t*>(a.get())+guard;auto* pb=static_cast<std::uint16_t*>(b.get())+guard;
        cuda_check(cudaMemcpy(a.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"wide reference guard");
        for(int row=0;row<width;++row)baseline.forward(x.weights,x.metadata,x.input+row*ni,pa+row*no,1,x.stream);
        cuda_check(cudaStreamSynchronize(x.stream),"wide M1 reference ready");
        cuda_check(cudaMemcpy(ref.data(),a.get(),ref.size()*2,cudaMemcpyDeviceToHost),"wide reference copy");
        for(int n=1;n<=width;++n) {
            // Widening changes only independent M16 group count. Retain broad
            // tail/group/capacity boundaries without quadratic all-width traffic.
            if(width>256 && n!=width && n!=width-1 &&
               std::find(std::begin(sample_widths),std::end(sample_widths),n)==std::end(sample_widths)) continue;
            cuda_check(cudaMemcpy(b.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"wide candidate guard");
            candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"wide candidate ready");
            cuda_check(cudaMemcpy(got.data(),b.get(),got.size()*2,cudaMemcpyDeviceToHost),"wide candidate copy");
            bool exact=true;
            for(std::size_t i=0;i<got.size();++i) {
                if(i<guard || i>=guard+static_cast<std::size_t>(n)*no)exact=exact && got[i]==0x3555;
                else exact=exact && got[i]==ref[i] && (got[i]&0x7c00)!=0x7c00;
            }
            if(n==width || !exact){save(tag+"-input.bin",input);save(tag+"-reference.bin",ref);save(tag+"-m"+std::to_string(n)+"-candidate.bin",got);}
            require(exact,"wide exact/canary mismatch "+tag+" M"+std::to_string(n));
            candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"wide repeat ready");
            cuda_check(cudaMemcpy(repeat.data(),b.get(),repeat.size()*2,cudaMemcpyDeviceToHost),"wide repeat copy");require(repeat==got,"wide repeat mismatch");
            csv << x.layer << ',' << x.operation << ',' << x.metadata.K << ',' << ni << ',' << no << ',' << n << ',' << n*no << ",1," << candidate.dispatch_name(x.metadata,n,admit) << '\n';csv.flush();require(csv.good(),"wide csv write");
        }
        std::vector<std::uint16_t> after(input.size());cuda_check(cudaMemcpy(after.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"wide immutable input");require(after==input,"wide input mutation");
    }
};

void run_wide_prefill_operators(Exl3TextModel& target) {
    require(env("NINFER_EXL3_WIDE_PREFILL")=="1","wide operator mode requires flag1");
    const int width=env("NINFER_EXL3_PREFILL_STAGED_REDUCTION")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE64")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE128")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE256")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE512")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE1024")=="1" ? 1024 : 512) : 256) : 128) : 64) : 32) : 16;
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));require(ids.size()>=static_cast<std::size_t>(16+width),"wide teacher extent");
    auto ctx=target.create_context(true);require(ctx->try_enable_oscar_from_environment(),"wide OSCAR");
    ctx->prefill(std::span<const std::int64_t>(ids.data(),16));
    WidePrefillOperators check(env("NINFER_E5A4_OUT"));
    ctx->set_target_projection_observer_for_test(WidePrefillOperators::callback,&check,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    ctx->append_prefill_wide(std::span<const std::int64_t>(ids.data()+16,width));
    cuda_check(cudaDeviceSynchronize(),"wide operator sweep ready");ctx->set_target_projection_observer_for_test(nullptr,nullptr);
    require(check.seen.size()==400 && ctx->position()==16+width,"wide all400 projection coverage");
    std::ofstream f(check.directory/"result.txt");f << "PASS operators=400 widths=1.." << width << " cases=" << 400*width << " exact=1\n";require(f.good(),"wide result write");
    std::cout << "WIDE_PREFILL_OPERATORS PASS operators=400 cases=" << 400*width << "\n";
}

// Smallest real-caller differential for the projection graph cache. The live
// context stays eager; one observed 1024-row projection is replayed through an
// independently owned graph workspace so capture cannot hide caller work.
struct PrefillProjectionGraphOperator {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit=Admission::target_wide_prefill;
    static constexpr std::size_t guard=128;
    bool checked=false;
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        static_cast<PrefillProjectionGraphOperator*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if(checked || x.rows!=1024 || x.metadata.K<5 || x.metadata.K>8)return;
        checked=true;
        const int ni=x.metadata.in_features,no=x.metadata.out_features;
        const std::size_t extent=static_cast<std::size_t>(x.rows)*no;
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555),
            reference(initial),first(initial),second(initial),fallback(initial);
        DeviceBuffer a(initial.size()*2),b(initial.size()*2),c(initial.size()*2);
        auto* pa=static_cast<std::uint16_t*>(a.get())+guard;
        auto* pb=static_cast<std::uint16_t*>(b.get())+guard;
        auto* pc=static_cast<std::uint16_t*>(c.get())+guard;
        cuda_check(cudaStreamSynchronize(x.stream),"projection graph observed input ready");
        std::vector<std::uint16_t> input(static_cast<std::size_t>(x.rows)*ni),after(input.size());
        cuda_check(cudaMemcpy(input.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),
                   "projection graph input snapshot");
        const auto saved=env("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS");
        _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS","0");
        Exl3CudaLinearWorkspace eager(ni,no,1024,false,false,false,false,false,true);
        _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS","1");
        Exl3CudaLinearWorkspace graph(ni,no,1024,false,false,false,false,false,true);
        cuda_check(cudaMemcpy(a.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),
                   "projection graph eager poison");
        cuda_check(cudaMemcpy(b.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),
                   "projection graph bound poison");
        cuda_check(cudaMemcpy(c.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),
                   "projection graph fallback poison");
        eager.forward(x.weights,x.metadata,x.input,pa,1024,x.stream,admit);
        graph.forward(x.weights,x.metadata,x.input,pb,1024,x.stream,admit);
        cuda_check(cudaStreamSynchronize(x.stream),"projection graph first replay ready");
        cuda_check(cudaMemcpy(reference.data(),a.get(),reference.size()*2,cudaMemcpyDeviceToHost),
                   "projection graph eager copy");
        cuda_check(cudaMemcpy(first.data(),b.get(),first.size()*2,cudaMemcpyDeviceToHost),
                   "projection graph first copy");
        graph.forward(x.weights,x.metadata,x.input,pb,1024,x.stream,admit);
        graph.forward(x.weights,x.metadata,x.input,pc,1024,x.stream,admit);
        cuda_check(cudaStreamSynchronize(x.stream),"projection graph repeat/fallback ready");
        cuda_check(cudaMemcpy(second.data(),b.get(),second.size()*2,cudaMemcpyDeviceToHost),
                   "projection graph repeat copy");
        cuda_check(cudaMemcpy(fallback.data(),c.get(),fallback.size()*2,cudaMemcpyDeviceToHost),
                   "projection graph fallback copy");
        require(reference==first && first==second && second==fallback,
                "projection graph exact output/canary mismatch");
        require(graph.prefill_projection_graph_captures()==1 &&
                graph.prefill_projection_graph_replays()==2 &&
                graph.prefill_projection_graph_binding_fallbacks()==1,
                "projection graph capture/replay/fallback counters");
        cuda_check(cudaMemcpy(after.data(),x.input,after.size()*2,cudaMemcpyDeviceToHost),
                   "projection graph input after");
        require(after==input,"projection graph input mutation");
        _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS",saved.c_str());
    }
};

void run_prefill_projection_graph_operator(Exl3TextModel& target) {
    for(const char* key:{"NINFER_EXL3_WIDE_PREFILL","NINFER_EXL3_PREFILL_STAGED_REDUCTION",
            "NINFER_EXL3_PREFILL_WIDE1024","NINFER_EXL3_PREFILL_DIRECT_PARTIALS"})
        require(env(key)=="1","projection graph operator requires retained wide-prefill flags");
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size()>=1040,"projection graph operator teacher extent");
    const auto saved=env("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS");
    _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS","0");
    auto ctx=target.create_context(true);
    require(ctx->try_enable_oscar_from_environment(),"projection graph operator OSCAR");
    ctx->prefill(std::span<const std::int64_t>(ids.data(),16));
    PrefillProjectionGraphOperator check;
    ctx->set_target_projection_observer_for_test(
        PrefillProjectionGraphOperator::callback,&check,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    ctx->append_prefill_wide(std::span<const std::int64_t>(ids.data()+16,1024));
    cuda_check(cudaDeviceSynchronize(),"projection graph operator complete");
    ctx->set_target_projection_observer_for_test(nullptr,nullptr);
    _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS",saved.c_str());
    require(check.checked && ctx->position()==1040,
            "projection graph real-caller coverage");
    std::cout<<"PREFILL_PROJECTION_GRAPH PASS captures=1 replays=2 fallbacks=1 exact=1\n";
}

struct PrefillProjectionChainGraphOperator {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    bool checked=false;
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        static_cast<PrefillProjectionChainGraphOperator*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if(checked || x.rows!=1024 || x.metadata.K!=6 ||
           x.metadata.out_features>6144 || x.metadata.in_features==17408)return;
        checked=true;
        constexpr std::size_t guard=128;
        const int ni=x.metadata.in_features,no=x.metadata.out_features;
        const std::size_t input_extent=static_cast<std::size_t>(x.rows)*ni;
        const std::size_t extent=static_cast<std::size_t>(x.rows)*no;
        std::vector<std::uint16_t> input_a(input_extent),input_b;
        cuda_check(cudaMemcpy(input_a.data(),x.input,input_extent*2,cudaMemcpyDeviceToHost),
                   "projection chain changing-input download");
        input_b=input_a;
        for(std::size_t index=0;index<input_b.size();index+=4093)
            input_b[index]^=0x8000u;
        std::vector<std::uint16_t> poison(extent+2*guard,0x3555),
            eager_a(poison),eager_b(poison),graph_a(poison),graph_b(poison);
        DeviceBuffer changing_input(input_extent*2),ea(poison.size()*2),eb(poison.size()*2),
            ga(poison.size()*2),gb(poison.size()*2);
        for(auto* buffer:{ea.get(),eb.get(),ga.get(),gb.get()})
            cuda_check(cudaMemcpy(buffer,poison.data(),poison.size()*2,cudaMemcpyHostToDevice),
                       "projection chain output poison");
        const auto saved=env("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS");
        const auto saved_chain=env("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS");
        _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS","0");
        _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS","1");
        Exl3CudaLinearWorkspace eager_first(ni,no,1024,false,false,false,false,false,true),
            eager_second(ni,no,1024,false,false,false,false,false,true),
            graph_first(ni,no,1024,false,false,false,false,false,true),
            graph_second(ni,no,1024,false,false,false,false,false,true);
        const auto submit=[&](Exl3CudaLinearWorkspace& first,
            Exl3CudaLinearWorkspace& second,std::uint16_t* first_output,
            std::uint16_t* second_output,const std::uint16_t* input,cudaStream_t stream) {
            first.forward(x.weights,x.metadata,input,first_output,1024,stream,
                Admission::target_wide_prefill);
            second.forward(x.weights,x.metadata,input,second_output,1024,stream,
                Admission::target_wide_prefill);
        };
        cuda_check(cudaMemcpyAsync(changing_input.get(),input_a.data(),input_extent*2,
            cudaMemcpyHostToDevice,x.stream),"projection chain first changing-input upload");
        submit(eager_first,eager_second,static_cast<std::uint16_t*>(ea.get())+guard,
            static_cast<std::uint16_t*>(eb.get())+guard,
            static_cast<const std::uint16_t*>(changing_input.get()),x.stream);

        auto context_owner=std::make_shared<int>(1);
        auto model_owner=std::make_shared<int>(2);
        auto scratch_owner=std::make_shared<int>(3);
        const std::array<ninfer::exl3::Exl3GraphBufferIdentity,5> buffers{{
            {changing_input.get(),input_extent*2},
            {ga.get(),poison.size()*2},{gb.get(),poison.size()*2},
            {x.weights.trellis,2},{x.weights.suh,static_cast<std::size_t>(ni)*2}}};
        ninfer::exl3::Exl3GraphCompatibilityFingerprint fingerprint;
        fingerprint.bind(context_owner,model_owner,scratch_owner,buffers,1024,1024,
            1024,static_cast<unsigned>(ni),5,0x51564b36u,
            ninfer::exl3::Exl3GraphPrecision::oscar_int2_fp16,
            ninfer::exl3::Exl3GraphPositionPolicy::oscar_split_class,1,x.stream);
        const std::array<ninfer::exl3::Exl3GraphBoundResource,5> resources{{
            {scratch_owner,changing_input.get(),buffers[0].bytes,1},
            {scratch_owner,ga.get(),buffers[1].bytes,2},
            {scratch_owner,gb.get(),buffers[2].bytes,3},
            {model_owner,x.weights.trellis,buffers[3].bytes,4},
            {model_owner,x.weights.suh,buffers[4].bytes,5}}};
        ninfer::exl3::Exl3GraphCaptureExtent extent_record;
        extent_record.known=true;
        extent_record.retained[static_cast<unsigned>(
            ninfer::exl3::Exl3ResourceInventory::Domain::graph_count)]=2;
        ninfer::exl3::Exl3PrefillProjectionChainGraph chain;
        ninfer::exl3::Exl3PrefillProjectionChainGraph::Request request{
            fingerprint,std::span<const ninfer::exl3::Exl3GraphBoundResource>(resources),
            extent_record,1,0x51564b360001ull,x.stream,true};
        const auto graph_body=[&](cudaStream_t stream) {
            submit(graph_first,graph_second,
                static_cast<std::uint16_t*>(ga.get())+guard,
                static_cast<std::uint16_t*>(gb.get())+guard,
                static_cast<const std::uint16_t*>(changing_input.get()),stream);
        };
        require(chain.execute(request,graph_body),"projection chain first graph dispatch");
        cuda_check(cudaStreamSynchronize(x.stream),"projection chain first replay");
        require(chain.complete_after_drain(x.stream),"projection chain first completion");
        for(const auto& item:{std::pair{ea.get(),&eager_a},std::pair{eb.get(),&eager_b},
                std::pair{ga.get(),&graph_a},std::pair{gb.get(),&graph_b}})
            cuda_check(cudaMemcpy(item.second->data(),item.first,item.second->size()*2,
                cudaMemcpyDeviceToHost),"projection chain first output copy");
        require(eager_a==graph_a && eager_b==graph_b,
                "projection chain first-input exact output/canary equality");
        const auto output_hash=[](const std::vector<std::uint16_t>& values) {
            std::uint64_t hash=14695981039346656037ull;
            for(const auto value:values) {
                hash^=value&255u;hash*=1099511628211ull;
                hash^=value>>8;hash*=1099511628211ull;
            }
            return hash;
        };
        const auto first_hash_a=output_hash(eager_a);
        const auto first_hash_b=output_hash(eager_b);

        for(auto* buffer:{ea.get(),eb.get(),ga.get(),gb.get()})
            cuda_check(cudaMemcpyAsync(buffer,poison.data(),poison.size()*2,
                cudaMemcpyHostToDevice,x.stream),"projection chain changed-input output poison");
        cuda_check(cudaMemcpyAsync(changing_input.get(),input_b.data(),input_extent*2,
            cudaMemcpyHostToDevice,x.stream),"projection chain second changing-input upload");
        submit(eager_first,eager_second,static_cast<std::uint16_t*>(ea.get())+guard,
            static_cast<std::uint16_t*>(eb.get())+guard,
            static_cast<const std::uint16_t*>(changing_input.get()),x.stream);
        require(chain.execute(request,graph_body),"projection chain compatible replay");
        cuda_check(cudaStreamSynchronize(x.stream),"projection chain second replay");
        require(chain.complete_after_drain(x.stream),"projection chain second completion");
        for(const auto& item:{std::pair{ea.get(),&eager_a},std::pair{eb.get(),&eager_b},
                std::pair{ga.get(),&graph_a},std::pair{gb.get(),&graph_b}})
            cuda_check(cudaMemcpy(item.second->data(),item.first,item.second->size()*2,
                cudaMemcpyDeviceToHost),"projection chain changed-input replay copy");
        require(eager_a==graph_a && eager_b==graph_b,
                "projection chain changed-input replay equality");
        require(first_hash_a!=output_hash(eager_a) || first_hash_b!=output_hash(eager_b),
                "projection chain changed input did not discriminate replay bindings");

        for(auto* buffer:{ga.get(),gb.get()})
            cuda_check(cudaMemcpyAsync(buffer,poison.data(),poison.size()*2,
                cudaMemcpyHostToDevice,x.stream),"projection chain fallback output poison");
        auto mismatch=request;mismatch.generation=2;
        mismatch.fingerprint.bind(context_owner,model_owner,scratch_owner,buffers,
            1024,1024,1024,static_cast<unsigned>(ni),5,0x51564b36u,
            ninfer::exl3::Exl3GraphPrecision::oscar_int2_fp16,
            ninfer::exl3::Exl3GraphPositionPolicy::oscar_split_class,2,x.stream);
        require(!chain.execute(mismatch,graph_body),"projection chain mismatch fallback");
        cuda_check(cudaStreamSynchronize(x.stream),"projection chain fallback ready");
        for(const auto& item:{std::pair{ea.get(),&eager_a},std::pair{eb.get(),&eager_b},
                std::pair{ga.get(),&graph_a},std::pair{gb.get(),&graph_b}})
            cuda_check(cudaMemcpy(item.second->data(),item.first,item.second->size()*2,
                cudaMemcpyDeviceToHost),"projection chain output copy");
        require(eager_a==graph_a && eager_b==graph_b,
                "projection chain changed-input exact output/canary equality");
        const auto before_retire=chain.snapshot();
        require(before_retire.captures==1 && before_retire.replays==2 &&
                before_retire.eager_fallbacks==1 &&
                before_retire.binding_mismatches==1 && before_retire.ready,
                "projection chain route counters");
        require(chain.retire_after_drain("test complete"),
                "projection chain retirement");
        _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS",saved.c_str());
        _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS",saved_chain.c_str());
    }
};

void run_prefill_projection_chain_graph_operator(Exl3TextModel& target) {
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size()>=1040,"projection chain teacher extent");
    const auto saved=env("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS");
    _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS","0");
    auto ctx=target.create_context(true);
    require(ctx->try_enable_oscar_from_environment(),"projection chain OSCAR");
    ctx->prefill(std::span<const std::int64_t>(ids.data(),16));
    PrefillProjectionChainGraphOperator check;
    ctx->set_target_projection_observer_for_test(
        PrefillProjectionChainGraphOperator::callback,&check,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    ctx->append_prefill_wide(std::span<const std::int64_t>(ids.data()+16,1024));
    cuda_check(cudaDeviceSynchronize(),"projection chain caller complete");
    ctx->set_target_projection_observer_for_test(nullptr,nullptr);
    _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS",saved.c_str());
    require(check.checked,"projection chain real projection coverage");
    std::cout<<"PREFILL_PROJECTION_CHAIN_GRAPH PASS captures=1 replays=2 mismatch_fallback=1 exact=1\n";
}

// Focused P16 leaf oracle: original wide/direct-async versus row-pair, same real inputs.
struct RowpairPrefillOperators {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit=Admission::target_wide_prefill;
    static constexpr int widths[]={1,16,31,32,33,47,48,63,64,65,1008,1023,1024};
    static constexpr std::size_t guard=128;
    struct Flag {
        std::string saved=env("NINFER_EXL3_PREFILL_ROWPAIR_K6");
        explicit Flag(const char* value){_putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6",value);}
        ~Flag(){_putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6",saved.c_str());}
    };
    std::filesystem::path directory;
    std::ofstream csv;
    int sites=0,cases=0;
    std::vector<std::string> seen;
    explicit RowpairPrefillOperators(std::filesystem::path p):directory(p) {
        require(!p.empty() && !std::filesystem::exists(p),"rowpair output must be new");
        std::filesystem::create_directories(p);csv.open(p/"operators.csv");
        csv<<"layer,operation,K,input,output,rows,elements,exact,repeat,canaries,input_immutable,dispatch,output_fnv64\n";
    }
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user){static_cast<RowpairPrefillOperators*>(user)->check(x);}
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        const int ni=x.metadata.in_features,no=x.metadata.out_features;
        if(x.metadata.K!=6 || (ni==17408 && no==5120))return;
        require(x.rows==1024,"rowpair real capture width");
        const auto site=std::to_string(x.layer)+"-"+x.operation;
        require(std::find(seen.begin(),seen.end(),site)==seen.end(),"rowpair duplicate site");seen.push_back(site);++sites;
        std::ostringstream site_csv;
        cuda_check(cudaStreamSynchronize(x.stream),"rowpair input ready");
        std::vector<std::uint16_t> input(1024u*ni),after(input.size());
        cuda_check(cudaMemcpy(input.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"rowpair input snapshot");
        const std::size_t extent=1024u*no,partial_bytes=extent*5*sizeof(float);
        DeviceBuffer wa(partial_bytes),wb(partial_bytes);
        const ninfer::exl3::Exl3CudaAccumulationView va{static_cast<float*>(wa.get()),partial_bytes},vb{static_cast<float*>(wb.get()),partial_bytes};
        Flag scope("0");
        Exl3CudaLinearWorkspace baseline(ni,no,1024,false,false,false,false,false,true,false,va);
        _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6","1");
        Exl3CudaLinearWorkspace candidate(ni,no,1024,false,false,false,false,false,true,false,vb);
        require(!baseline.target_rowpair_k6_candidate(x.metadata,1024,admit),"rowpair off latch");
        require(!candidate.target_rowpair_k6_candidate(x.metadata,1024,Admission::ordinary),"rowpair ordinary guard");
        for(int n:{0,1,16,31,32,33,1024,1025})require(candidate.target_rowpair_k6_candidate(x.metadata,n,admit)==(n>=32&&n<=1024),"rowpair row capacity");
        if(sites==1){
            for(const char* value:{"","0","01","true","2"}){
                Flag off(value);Exl3CudaLinearWorkspace w(ni,no,32,false,false,false,false,false,true);
                require(!w.target_rowpair_k6_candidate(x.metadata,32,admit),"rowpair exact1 latch");
            }
            Exl3CudaLinearWorkspace unowned(ni,no,32),small(ni,no,31,false,false,false,false,false,true);
            require(!unowned.target_rowpair_k6_candidate(x.metadata,32,admit)&&!small.target_rowpair_k6_candidate(x.metadata,32,admit),"rowpair owner capacity");
            for(int k:{5,7,8}){auto md=x.metadata;md.K=k;require(!candidate.target_rowpair_k6_candidate(md,32,admit),"rowpair K guard");}
            for(int bad=0;bad<5;++bad){auto md=x.metadata;if(bad==0)md.mcg=true;if(bad==1)md.mul1=false;if(bad==2)md.has_bias=true;if(bad==3)md.in_features+=128;if(bad==4)md.out_features+=128;require(!candidate.target_rowpair_k6_candidate(md,32,admit),"rowpair metadata guard");}
        }
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555),ref(initial),got(initial),repeat(initial);
        DeviceBuffer a(initial.size()*2),b(initial.size()*2);
        std::vector<std::uint32_t> partial_ref(extent*5),partial_got(extent*5);
        for(int n:widths){
            auto run=[&](Exl3CudaLinearWorkspace& w,DeviceBuffer& dst,DeviceBuffer& partial,std::vector<std::uint16_t>& host){
                cuda_check(cudaMemsetAsync(partial.get(),0xA5,partial_bytes,x.stream),"rowpair independent partial poison");
                cuda_check(cudaMemcpyAsync(dst.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice,x.stream),"rowpair independent output poison");
                w.forward(x.weights,x.metadata,x.input,static_cast<std::uint16_t*>(dst.get())+guard,n,x.stream,admit);
                cuda_check(cudaStreamSynchronize(x.stream),"rowpair leaf ready");
                cuda_check(cudaMemcpy(host.data(),dst.get(),host.size()*2,cudaMemcpyDeviceToHost),"rowpair result");
            };
            run(baseline,a,wa,ref);
            const std::size_t active_partial_bytes=static_cast<std::size_t>(n)*no*5*sizeof(float);
            if(n>=32)cuda_check(cudaMemcpy(partial_ref.data(),wa.get(),active_partial_bytes,cudaMemcpyDeviceToHost),"rowpair reference partial bytes");
            run(candidate,b,wb,got);
            if(n>=32){
                cuda_check(cudaMemcpy(partial_got.data(),wb.get(),active_partial_bytes,cudaMemcpyDeviceToHost),"rowpair candidate partial bytes");
                if(std::memcmp(partial_ref.data(),partial_got.data(),active_partial_bytes)!=0){
                    for(const auto& item:{std::pair{"partial-reference.bin",&partial_ref},std::pair{"partial-candidate.bin",&partial_got}}){std::ofstream f(directory/item.first,std::ios::binary);f.write(reinterpret_cast<const char*>(item.second->data()),active_partial_bytes);}
                    require(false,"rowpair exact FP32 split partial failure");
                }
            }
            run(candidate,b,wb,repeat);
            for(std::size_t i=guard;i<guard+static_cast<std::size_t>(n)*no;++i)require((ref[i]&0x7c00)!=0x7c00 && (got[i]&0x7c00)!=0x7c00 && (repeat[i]&0x7c00)!=0x7c00,"rowpair nonfinite active F16");
            bool exact=ref==got && got==repeat,canary=true;
            for(std::size_t i=0;i<initial.size();++i)if(i<guard || i>=guard+static_cast<std::size_t>(n)*no)canary=canary && ref[i]==initial[i] && got[i]==initial[i] && repeat[i]==initial[i];
            const bool eligible=n>=32;
            require((std::string(candidate.dispatch_name(x.metadata,n,admit))=="target_wide_prefill_rowpair_k6")==eligible,"rowpair dispatch witness");
            if(!exact || !canary){
                for(const auto& item:{std::pair{"reference.bin",&ref},std::pair{"candidate.bin",&got},std::pair{"repeat.bin",&repeat}}){std::ofstream f(directory/item.first,std::ios::binary);f.write(reinterpret_cast<const char*>(item.second->data()),item.second->size()*2);}
            }
            require(exact && canary,"rowpair exact/canary/repeat failure");
            std::uint64_t hash=14695981039346656037ull;for(std::size_t i=guard;i<guard+static_cast<std::size_t>(n)*no;++i){hash^=got[i]&255;hash*=1099511628211ull;hash^=got[i]>>8;hash*=1099511628211ull;}
            site_csv<<x.layer<<','<<x.operation<<",6,"<<ni<<','<<no<<','<<n<<','<<static_cast<std::size_t>(n)*no<<",1,1,1,1,"<<candidate.dispatch_name(x.metadata,n,admit)<<','<<hash<<'\n';++cases;
        }
        cuda_check(cudaMemcpy(after.data(),x.input,after.size()*2,cudaMemcpyDeviceToHost),"rowpair input after");require(input==after,"rowpair input mutated");
        csv<<site_csv.str();csv.flush();require(csv.good(),"rowpair witness write");
    }
};

void run_rowpair_prefill_operators(Exl3TextModel& target){
    for(const char* key:{"NINFER_EXL3_WIDE_PREFILL","NINFER_EXL3_PREFILL_STAGED_REDUCTION","NINFER_EXL3_PREFILL_WIDE1024","NINFER_EXL3_PREFILL_DIRECT_PARTIALS","NINFER_EXL3_PREFILL_DIRECT_ASYNC_A","NINFER_EXL3_PREFILL_ROWPAIR_K6"})require(env(key)=="1","rowpair qualified wide flags");
    require(env("NINFER_EXL3_GENERIC_SPLITS").empty() || env("NINFER_EXL3_GENERIC_SPLITS")=="5","rowpair operator fixes five splits");
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));require(ids.size()>=1040,"rowpair teacher extent");
    auto ctx=target.create_context(true);require(ctx->try_enable_oscar_from_environment(),"rowpair OSCAR");ctx->prefill(std::span<const std::int64_t>(ids.data(),16));
    RowpairPrefillOperators check(env("NINFER_E5A4_OUT"));
    ctx->set_target_projection_observer_for_test(RowpairPrefillOperators::callback,&check,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    ctx->append_prefill_wide(std::span<const std::int64_t>(ids.data()+16,1024));
    cuda_check(cudaDeviceSynchronize(),"rowpair operators done");ctx->set_target_projection_observer_for_test(nullptr,nullptr);
    require(check.sites==229 && check.seen.size()==229 && check.cases==229*13,"rowpair exact site/case coverage");
    std::ofstream f(check.directory/"result.txt");f<<"PASS sites="<<check.sites<<" cases="<<check.cases<<" exact=1 partial_workspace=exact_active_five_splits\n";require(f.good(),"rowpair result write");
    std::cout<<"ROWPAIR_PREFILL_OPERATORS PASS sites="<<check.sites<<" cases="<<check.cases<<"\n";
}

// K7 extension of the preserved row-pair schedule.  This is intentionally a
// separate oracle so the rejected/default-off K6 experiment remains untouched.
struct RowpairPrefillK7Operators {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit=Admission::target_wide_prefill;
    static constexpr int widths[]={31,32,33,63,64,1008,1023,1024};
    static constexpr std::size_t guard=128;
    struct Flag {
        std::string saved=env("NINFER_EXL3_PREFILL_ROWPAIR_K7");
        explicit Flag(const char* value){_putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K7",value);}
        ~Flag(){_putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K7",saved.c_str());}
    };
    std::filesystem::path directory;
    std::ofstream csv,timing;
    int observed=0,sites=0,cases=0;
    std::vector<std::string> seen;
    explicit RowpairPrefillK7Operators(std::filesystem::path p):directory(p) {
        require(!p.empty() && !std::filesystem::exists(p),"K7 rowpair output must be new");
        std::filesystem::create_directories(p);csv.open(p/"operators.csv");
        csv<<"layer,operation,K,input,output,rows,elements,exact,repeat,canaries,input_immutable,dispatch,output_fnv64\n";
        const auto timing_path=env("NINFER_E5A4_PREFILL_ROWPAIR_K7_TIMING_OUT");
        if(!timing_path.empty()){
            timing.open(timing_path,std::ios::trunc);
            require(timing.good(),"K7 rowpair timing output creation");
            timing<<"site,layer,operation,order,repetitions,control_us,candidate_us,gain_pct\n";
        }
    }
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user){
        static_cast<RowpairPrefillK7Operators*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        ++observed;
        const int ni=x.metadata.in_features,no=x.metadata.out_features;
        if(x.metadata.K!=7 || (ni==17408 && no==5120))return;
        require(x.rows==1024,"K7 rowpair real capture width");
        const auto site=std::to_string(x.layer)+"-"+x.operation;
        require(std::find(seen.begin(),seen.end(),site)==seen.end(),"K7 rowpair duplicate site");
        seen.push_back(site);++sites;
        std::ostringstream site_csv;
        cuda_check(cudaStreamSynchronize(x.stream),"K7 rowpair input ready");
        std::vector<std::uint16_t> input(1024u*ni),after(input.size());
        cuda_check(cudaMemcpy(input.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"K7 rowpair input snapshot");
        const std::size_t extent=1024u*no,partial_bytes=extent*5*sizeof(float);
        DeviceBuffer wa(partial_bytes),wb(partial_bytes);
        const ninfer::exl3::Exl3CudaAccumulationView va{static_cast<float*>(wa.get()),partial_bytes},vb{static_cast<float*>(wb.get()),partial_bytes};
        Flag scope("0");
        Exl3CudaLinearWorkspace baseline(ni,no,1024,false,false,false,false,false,true,false,va);
        _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K7","1");
        Exl3CudaLinearWorkspace candidate(ni,no,1024,false,false,false,false,false,true,false,vb);
        require(!baseline.target_rowpair_k7_candidate(x.metadata,1024,admit),"K7 rowpair off latch");
        require(!candidate.target_rowpair_k7_candidate(x.metadata,1024,Admission::ordinary),"K7 rowpair ordinary guard");
        for(int n:{0,1,16,31,32,33,1024,1025})
            require(candidate.target_rowpair_k7_candidate(x.metadata,n,admit)==(n>=32&&n<=1024),"K7 rowpair row capacity");
        if(sites==1){
            for(const char* malformed:{"2","true","01"}){
                Flag bad(malformed);bool rejected=false;
                try{Exl3CudaLinearWorkspace w(ni,no,32,false,false,false,false,false,true);}
                catch(const std::invalid_argument&){rejected=true;}
                require(rejected,"K7 rowpair accepted malformed opt-in");
            }
            {Flag off("0");Exl3CudaLinearWorkspace w(ni,no,32,false,false,false,false,false,true);
             require(!w.target_rowpair_k7_candidate(x.metadata,32,admit),"K7 rowpair flag0");}
            Exl3CudaLinearWorkspace unowned(ni,no,32),small(ni,no,31,false,false,false,false,false,true);
            require(!unowned.target_rowpair_k7_candidate(x.metadata,32,admit)&&
                    !small.target_rowpair_k7_candidate(x.metadata,32,admit),"K7 rowpair owner capacity");
            for(int k:{5,6,8}){auto md=x.metadata;md.K=k;
                require(!candidate.target_rowpair_k7_candidate(md,32,admit),"K7 rowpair K guard");}
            for(int bad=0;bad<5;++bad){auto md=x.metadata;
                if(bad==0)md.mcg=true;if(bad==1)md.mul1=false;if(bad==2)md.has_bias=true;
                if(bad==3)md.in_features+=128;if(bad==4)md.out_features+=128;
                require(!candidate.target_rowpair_k7_candidate(md,32,admit),"K7 rowpair metadata guard");}
        }
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555),ref(initial),got(initial),repeat(initial);
        DeviceBuffer a(initial.size()*2),b(initial.size()*2);
        std::vector<std::uint32_t> partial_ref(extent*5),partial_got(extent*5);
        for(int n:widths){
            auto run=[&](Exl3CudaLinearWorkspace& w,DeviceBuffer& dst,DeviceBuffer& partial,std::vector<std::uint16_t>& host){
                cuda_check(cudaMemsetAsync(partial.get(),0xA5,partial_bytes,x.stream),"K7 rowpair independent partial poison");
                cuda_check(cudaMemcpyAsync(dst.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice,x.stream),"K7 rowpair independent output poison");
                w.forward(x.weights,x.metadata,x.input,static_cast<std::uint16_t*>(dst.get())+guard,n,x.stream,admit);
                cuda_check(cudaStreamSynchronize(x.stream),"K7 rowpair leaf ready");
                cuda_check(cudaMemcpy(host.data(),dst.get(),host.size()*2,cudaMemcpyDeviceToHost),"K7 rowpair result");
            };
            run(baseline,a,wa,ref);
            const std::size_t active_partial_bytes=static_cast<std::size_t>(n)*no*5*sizeof(float);
            if(n>=32)cuda_check(cudaMemcpy(partial_ref.data(),wa.get(),active_partial_bytes,cudaMemcpyDeviceToHost),"K7 rowpair reference partial bytes");
            run(candidate,b,wb,got);
            if(n>=32){
                cuda_check(cudaMemcpy(partial_got.data(),wb.get(),active_partial_bytes,cudaMemcpyDeviceToHost),"K7 rowpair candidate partial bytes");
                require(std::memcmp(partial_ref.data(),partial_got.data(),active_partial_bytes)==0,"K7 rowpair exact FP32 split partial failure");
            }
            run(candidate,b,wb,repeat);
            for(std::size_t i=guard;i<guard+static_cast<std::size_t>(n)*no;++i)
                require((ref[i]&0x7c00)!=0x7c00 && (got[i]&0x7c00)!=0x7c00 && (repeat[i]&0x7c00)!=0x7c00,"K7 rowpair nonfinite active F16");
            bool exact=ref==got && got==repeat,canary=true;
            for(std::size_t i=0;i<initial.size();++i)
                if(i<guard || i>=guard+static_cast<std::size_t>(n)*no)
                    canary=canary && ref[i]==initial[i] && got[i]==initial[i] && repeat[i]==initial[i];
            const bool eligible=n>=32;
            require((std::string(candidate.dispatch_name(x.metadata,n,admit))=="target_wide_prefill_rowpair_k7")==eligible,"K7 rowpair dispatch witness");
            require(exact && canary,"K7 rowpair exact/canary/repeat failure");
            std::uint64_t hash=14695981039346656037ull;
            for(std::size_t i=guard;i<guard+static_cast<std::size_t>(n)*no;++i){hash^=got[i]&255;hash*=1099511628211ull;hash^=got[i]>>8;hash*=1099511628211ull;}
            site_csv<<x.layer<<','<<x.operation<<",7,"<<ni<<','<<no<<','<<n<<','<<static_cast<std::size_t>(n)*no<<",1,1,1,1,"<<candidate.dispatch_name(x.metadata,n,admit)<<','<<hash<<'\n';++cases;
        }
        cuda_check(cudaMemcpy(after.data(),x.input,after.size()*2,cudaMemcpyDeviceToHost),"K7 rowpair input after");
        require(input==after,"K7 rowpair input mutated");
        if(timing.is_open()){
            constexpr int repetitions=8;
            cudaEvent_t begin=nullptr,end=nullptr;
            cuda_check(cudaEventCreate(&begin),"create K7 rowpair timing begin");
            cuda_check(cudaEventCreate(&end),"create K7 rowpair timing end");
            auto measure=[&](Exl3CudaLinearWorkspace& w,DeviceBuffer& dst){
                cuda_check(cudaEventRecord(begin,x.stream),"record K7 rowpair timing begin");
                for(int repetition=0;repetition<repetitions;++repetition)
                    w.forward(x.weights,x.metadata,x.input,
                              static_cast<std::uint16_t*>(dst.get())+guard,1024,x.stream,admit);
                cuda_check(cudaEventRecord(end,x.stream),"record K7 rowpair timing end");
                cuda_check(cudaEventSynchronize(end),"synchronize K7 rowpair timing end");
                float milliseconds=0.0f;
                cuda_check(cudaEventElapsedTime(&milliseconds,begin,end),"resolve K7 rowpair timing");
                return static_cast<double>(milliseconds)*1000.0/repetitions;
            };
            double control_us=0.0,candidate_us=0.0;
            const bool control_first=(sites%2)==1;
            if(control_first){control_us=measure(baseline,a);candidate_us=measure(candidate,b);}
            else{candidate_us=measure(candidate,b);control_us=measure(baseline,a);}
            cuda_check(cudaEventDestroy(begin),"destroy K7 rowpair timing begin");
            cuda_check(cudaEventDestroy(end),"destroy K7 rowpair timing end");
            timing<<site<<','<<x.layer<<','<<x.operation<<','<<(control_first?"AB":"BA")
                  <<','<<repetitions<<','<<control_us<<','<<candidate_us<<','
                  <<(control_us-candidate_us)*100.0/control_us<<'\n';
            timing.flush();require(timing.good(),"K7 rowpair timing write");
        }
        csv<<site_csv.str();csv.flush();require(csv.good(),"K7 rowpair witness write");
    }
};

void run_rowpair_k7_prefill_operators(Exl3TextModel& target){
    for(const char* key:{"NINFER_EXL3_WIDE_PREFILL","NINFER_EXL3_PREFILL_STAGED_REDUCTION",
        "NINFER_EXL3_PREFILL_WIDE1024","NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
        "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A","NINFER_EXL3_PREFILL_ROWPAIR_K7"})
        require(env(key)=="1","K7 rowpair qualified wide flags");
    require(env("NINFER_EXL3_PREFILL_ROWPAIR_K6")=="0","K7 rowpair keeps rejected K6 route off");
    require(env("NINFER_EXL3_GENERIC_SPLITS").empty() || env("NINFER_EXL3_GENERIC_SPLITS")=="5","K7 rowpair operator fixes five splits");
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));require(ids.size()>=1040,"K7 rowpair teacher extent");
    auto ctx=target.create_context(true);require(ctx->try_enable_oscar_from_environment(),"K7 rowpair OSCAR");
    ctx->prefill(std::span<const std::int64_t>(ids.data(),16));
    RowpairPrefillK7Operators check(env("NINFER_E5A4_OUT"));
    ctx->set_target_projection_observer_for_test(RowpairPrefillK7Operators::callback,&check,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    ctx->append_prefill_wide(std::span<const std::int64_t>(ids.data()+16,1024));
    cuda_check(cudaDeviceSynchronize(),"K7 rowpair operators done");
    ctx->set_target_projection_observer_for_test(nullptr,nullptr);
    require(check.observed==400 && check.sites>0 && check.seen.size()==static_cast<std::size_t>(check.sites) &&
            check.cases==check.sites*static_cast<int>(std::size(RowpairPrefillK7Operators::widths)),
            "K7 rowpair exact site/case coverage");
    std::ofstream f(check.directory/"result.txt");
    f<<"PASS observed="<<check.observed<<" sites="<<check.sites<<" cases="<<check.cases
     <<" exact=1 partial_workspace=exact_active_five_splits\n";
    require(f.good(),"K7 rowpair result write");
    std::cout<<"ROWPAIR_K7_PREFILL_OPERATORS PASS observed="<<check.observed
             <<" sites="<<check.sites<<" cases="<<check.cases<<"\n";
    if(check.timing.is_open())
        std::cout<<"ROWPAIR_K7_PREFILL_TIMING PASS sites="<<check.sites
                 <<" repetitions=8 order=alternating_AB_BA\n";
}

void run_rowpair_k7_prefill_state(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose){
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()==4352,"K7 rowpair state context extent");
    require(code.size()>=prefix&&prose.size()>=prefix,"K7 rowpair state fixture extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A","1");
    _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6","0");
    const auto greedy=[](Exl3TextContext& context){
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),"K7 rowpair state finite logits");
        return static_cast<std::int64_t>(std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    std::ofstream out(env("NINFER_E5A4_OUT"),std::ios::trunc);
    require(out.good(),"K7 rowpair state output creation");
    out<<"fixture,control_prefill_ms,candidate_prefill_ms,wall_reduction_pct,tokens_exact,state_exact\n";
    for(const auto& fixture:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
        std::pair{"code",&code},std::pair{"prose",&prose}}){
        const std::vector<std::int64_t> input(fixture.second->begin(),fixture.second->begin()+prefix);
        _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K7","0");
        auto control=target.create_context(true);control->prepare_continuation(8);
        _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K7","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(8);
        struct Result{double prefill_ms=0;std::vector<std::int64_t> tokens;
            std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;};
        const auto run=[&](Exl3TextContext& context){
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            std::vector<std::int64_t> tokens;
            for(int row=0;row<8;++row){const auto token=greedy(context);tokens.push_back(token);context.decode(token);}
            return Result{prefill_ms,std::move(tokens),context.export_exact_host_state()};
        };
        const auto baseline=run(*control),changed=run(*candidate);
        const bool tokens_exact=baseline.tokens==changed.tokens;
        const bool state_exact=baseline.state->same_payload(*changed.state);
        require(tokens_exact&&state_exact,"K7 rowpair full prefill token/state mismatch");
        out<<fixture.first<<','<<baseline.prefill_ms<<','<<changed.prefill_ms<<','
           <<(baseline.prefill_ms-changed.prefill_ms)*100.0/baseline.prefill_ms
           <<",1,1\n";
        std::cout<<"ROWPAIR_K7_PREFILL_STATE fixture="<<fixture.first
                 <<" prefix=4096 decode_rows=8 control_prefill_ms="<<baseline.prefill_ms
                 <<" candidate_prefill_ms="<<changed.prefill_ms<<" exact=1\n";
    }
    out.flush();require(out.good(),"K7 rowpair state output write");
    const auto lifecycle_path=env("NINFER_E5A4_PREFILL_ROWPAIR_K7_LIFECYCLE_OUT");
    if(!lifecycle_path.empty()){
        std::ofstream lifecycle(lifecycle_path,std::ios::trunc);
        require(lifecycle.good(),"K7 rowpair lifecycle output creation");
        lifecycle<<"pair,launch,arm,free_before,free_live,free_after,live_used_bytes,endpoint_loss_bytes,position,token,exact,gate\n";
        const std::vector<std::int64_t> input(code.begin(),code.begin()+1024);
        struct Life{std::size_t before=0,live=0,after=0;int position=0;
            std::int64_t token=0;std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;};
        auto run_life=[&](bool candidate){
            Life result;std::size_t total_before=0,total_live=0,total_after=0;
            cuda_check(cudaMemGetInfo(&result.before,&total_before),"K7 rowpair lifecycle before");
            {
                _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K7",candidate?"1":"0");
                auto context=target.create_context(true);context->prepare_continuation(8);
                const auto request=Request::initialize(*context,input,1024);
                result.token=greedy(*context);context->decode(result.token);
                cuda_check(cudaDeviceSynchronize(),"K7 rowpair lifecycle live sync");
                result.position=context->position();
                result.state=context->export_exact_host_state();
                cuda_check(cudaMemGetInfo(&result.live,&total_live),"K7 rowpair lifecycle live");
                require(total_live==total_before,"K7 rowpair lifecycle total live identity");
            }
            cuda_check(cudaDeviceSynchronize(),"K7 rowpair lifecycle teardown sync");
            cuda_check(cudaMemGetInfo(&result.after,&total_after),"K7 rowpair lifecycle after");
            require(total_after==total_before,"K7 rowpair lifecycle total endpoint identity");
            return result;
        };
        for(int pair=0;pair<4;++pair){
            std::array<Life,2> arms;
            const bool control_first=(pair%2)==0;
            for(int launch=0;launch<2;++launch){
                const bool candidate=control_first?(launch==1):(launch==0);
                arms[candidate?1:0]=run_life(candidate);
                const auto& value=arms[candidate?1:0];
                const std::size_t live_used=value.before>value.live?value.before-value.live:0;
                const std::size_t loss=value.before>value.after?value.before-value.after:0;
                const bool gate=loss<=1048576u;
                lifecycle<<pair<<','<<launch<<','<<(candidate?"candidate":"control")<<','
                         <<value.before<<','<<value.live<<','<<value.after<<','<<live_used
                         <<','<<loss<<','<<value.position<<','<<value.token<<",1,"<<gate<<'\n';
                require(gate,"K7 rowpair lifecycle exceeded 1 MiB endpoint gate");
            }
            require(arms[0].position==1025&&arms[1].position==1025&&
                    arms[0].token==arms[1].token&&arms[0].state->same_payload(*arms[1].state),
                    "K7 rowpair lifecycle exact state mismatch");
        }
        lifecycle.flush();require(lifecycle.good(),"K7 rowpair lifecycle output write");
        std::cout<<"ROWPAIR_K7_PREFILL_LIFECYCLE PASS pairs=4 cycles=8 prefix=1024 decode_rows=1 endpoint_gate_bytes=1048576 exact_state=1\n";
    }
    std::cout<<"ROWPAIR_K7_PREFILL_STATE PASS fixtures=2 prefix=4096 decode_rows=8 exact_tokens_state=1\n";
}
