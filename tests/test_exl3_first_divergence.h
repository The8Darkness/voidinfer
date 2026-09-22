#pragma once

// Diagnostic only: fixed real tokens, canonical-input operator replay and a
// bounded in-memory first-layer trace. No model dump and no timing claims.
struct FDMetric {
    double error2=0, norm2=0, max_abs=0;
    std::size_t values=0, different=0, first=static_cast<std::size_t>(-1), worst=0;
    void add(double a,double b,bool bits_equal) {
        require(std::isfinite(a)&&std::isfinite(b),"FD nonfinite value");
        const double d=a-b;
        if(!bits_equal) {if(!different) first=values; ++different;}
        if(std::abs(d)>max_abs) {max_abs=std::abs(d);worst=values;}
        error2+=d*d;norm2+=b*b;++values;
    }
    double rel() const {return norm2>0?std::sqrt(error2/norm2):
        (error2>0?std::numeric_limits<double>::infinity():0);}
};

FDMetric fd_metric(std::span<const std::uint16_t> a,std::span<const std::uint16_t> b) {
    require(a.size()==b.size(),"FD metric extent");FDMetric result;
    for(std::size_t i=0;i<a.size();++i)
        result.add(half_to_float(a[i]),half_to_float(b[i]),a[i]==b[i]);
    return result;
}

void fd_write(std::ofstream& out,const std::string& name,int layer,int position,
              int columns,const FDMetric& m) {
    out<<name<<','<<layer<<','<<position<<','<<columns<<','<<m.values<<','
        <<m.different<<','<<m.first<<','<<m.worst<<','<<m.max_abs<<','<<m.rel()<<'\n';
    out.flush();
}

struct FDTrace {
    bool candidate=false;
    int boundary=22;
    std::array<std::vector<std::uint16_t>,128> outputs;
    std::vector<std::pair<std::string,std::vector<std::uint16_t>>> boundary_planes;
    std::vector<float> boundary_state;
    std::vector<std::uint16_t> q,k,v;
    std::ofstream* report=nullptr;
    int first_layer=-1,first_position=-1;
    std::size_t retained_bytes=0;
    static void callback(const ninfer::exl3::Exl3LayerObservation& x,void* user) {
        static_cast<FDTrace*>(user)->observe(x);
    }
    void observe(const ninfer::exl3::Exl3LayerObservation& x) {
        require(x.position==0||x.position==16,"FD fixed block identity");
        const auto copy=[&](const std::uint16_t* ptr,int columns) {
            require(ptr!=nullptr,"FD missing trace plane");
            return target_continue_device_bits(ptr,static_cast<std::size_t>(x.rows)*columns,
                                                "FD borrowed trace copy");
        };
        cuda_check(cudaStreamSynchronize(x.stream),"FD trace lifetime fence");
        auto output=copy(x.output,5120);
        auto& reference=outputs[(x.position?64:0)+x.layer];
        if(!candidate) {retained_bytes+=output.size()*2;reference=std::move(output);}
        else {
            const auto m=fd_metric(output,reference);
            if(m.different&&first_layer<0) {first_layer=x.layer;first_position=x.position;}
            fd_write(*report,"layer_output",x.layer,x.position,5120,m);
        }
        if(x.position!=16||x.layer!=boundary) return;
        std::size_t index=0;
        const auto plane=[&](const char* name,const std::uint16_t* pointer,int columns) {
            auto values=copy(pointer,columns);
            if(!candidate) {retained_bytes+=values.size()*2;
                boundary_planes.emplace_back(name,std::move(values));}
            else {require(index<boundary_planes.size()&&boundary_planes[index].first==name,
                          "FD boundary plane identity");
                fd_write(*report,name,x.layer,x.position,columns,
                    fd_metric(values,boundary_planes[index].second));}
            ++index;
        };
        if((x.layer+1)%4==0) {
            const auto& t=x.attention;
            plane("input",t.layer_input,5120);plane("input_norm",t.input_norm,5120);
            plane("q_projection",t.q_projection,12288);
            plane("k_projection",t.k_projection,1024);plane("v_projection",t.v_projection,1024);
            plane("q_norm",t.q_normed,6144);plane("k_norm",t.k_normed,1024);
            plane("q_rope",t.q_rope,6144);plane("k_rope",t.k_rope,1024);
            // The existing attention trace is after the in-place sigmoid gate.
            plane("attention_gated",t.attention_output,6144);
            plane("o_projection",t.output_projection,5120);
            plane("post_residual",t.post_attention_residual,5120);
            plane("mlp_norm",t.mlp_input,5120);plane("gate",t.gate_projection,17408);
            plane("up",t.up_projection,17408);plane("activation",t.activated_mlp,17408);
            plane("down",t.down_projection,5120);
            if(!candidate) {
                q=copy(t.q_rope,6144);
                k=target_continue_device_bits(x.k_cache,static_cast<std::size_t>(1040)*1024,"FD real K");
                v=target_continue_device_bits(x.v_cache,static_cast<std::size_t>(1040)*1024,"FD real V");
                retained_bytes+=(q.size()+k.size()+v.size())*2;
            }
        } else {
            const auto& t=x.gdn;
            plane("input",t.layer_input,5120);plane("input_norm",t.input_norm,5120);
            plane("qkv_projection",t.qkv_projection,10240);plane("z_projection",t.z_projection,6144);
            plane("o_projection",t.output_projection,5120);
            plane("post_residual",t.post_attention_residual,5120);
            plane("mlp_norm",t.mlp_input,5120);plane("gate",t.gate_projection,17408);
            plane("up",t.up_projection,17408);plane("activation",t.activated_mlp,17408);
            plane("down",t.down_projection,5120);
            std::vector<float> state(48*128*128);
            cuda_check(cudaMemcpy(state.data(),t.state_after,state.size()*4,cudaMemcpyDeviceToHost),
                       "FD recurrent trace");
            if(!candidate) {boundary_state=std::move(state);retained_bytes+=boundary_state.size()*4;}
            else {FDMetric m;for(std::size_t i=0;i<state.size();++i)
                m.add(state[i],boundary_state[i],std::memcmp(&state[i],&boundary_state[i],4)==0);
                fd_write(*report,"recurrent_after",x.layer,x.position,128,m);}
        }
        require(retained_bytes<2ull*1024*1024*1024,"FD bounded host retention");
    }
};

FDMetric fd_typed_kv(const ninfer::exl3::Exl3ExactHostState& a,
                     const ninfer::exl3::Exl3ExactHostState& b,std::ofstream& out) {
    struct Plane {int layer,first,rows;std::span<const std::uint16_t> k,v;};
    std::vector<Plane> reference;
    b.visit_kv_for_test([&](int l,int f,int r,auto k,auto v){reference.push_back({l,f,r,k,v});});
    std::size_t index=0;FDMetric all;
    a.visit_kv_for_test([&](int l,int f,int r,auto k,auto v){
        require(index<reference.size(),"FD KV inventory overflow");const auto& ref=reference[index++];
        require(l==ref.layer&&f==ref.first&&r==ref.rows,"FD KV logical identity");
        fd_write(out,"K",l,f,1024,fd_metric(k,ref.k));
        fd_write(out,"V",l,f,1024,fd_metric(v,ref.v));
        for(std::size_t i=0;i<k.size();++i) all.add(half_to_float(k[i]),half_to_float(ref.k[i]),k[i]==ref.k[i]);
        for(std::size_t i=0;i<v.size();++i) all.add(half_to_float(v[i]),half_to_float(ref.v[i]),v[i]==ref.v[i]);
    });
    require(index==reference.size(),"FD KV inventory short");return all;
}

void fd_projection_replay(T69ProjectionCaptureItem& site,std::ofstream& report) {
    require(site.input&&site.metadata.K==7,"FD real K7 capture");
    const int ni=site.metadata.in_features,no=site.metadata.out_features;
    ninfer::exl3::Exl3CudaLinearWorkspace direct(ni,no,1024,false,false,false,false,false,true);
    ninfer::exl3::Exl3CudaReconstructGemmWorkspace candidate(ni,no,1024);
    DeviceBuffer a(1024ull*no*2),b(1024ull*no*2);
    direct.forward(site.weights,site.metadata,static_cast<const std::uint16_t*>(site.input->get()),
        static_cast<std::uint16_t*>(a.get()),1024,nullptr,
        ninfer::exl3::Exl3CudaLinearAdmission::target_wide_prefill);
    candidate.forward_numeric_candidate(site.weights,site.metadata,
        static_cast<const std::uint16_t*>(site.input->get()),static_cast<std::uint16_t*>(b.get()),1024);
    const auto ah=target_continue_device_bits(static_cast<const std::uint16_t*>(a.get()),1024ull*no,"FD direct");
    const auto bh=target_continue_device_bits(static_cast<const std::uint16_t*>(b.get()),1024ull*no,"FD GEMM");
    fd_write(report,"same_input_projection",site.layer,16,no,fd_metric(bh,ah));
    const auto oracle=t69_check_fp64_oracle(site,ah,bh,1024);
    std::cout<<"FD_PROJECTION_ORACLE layer="<<site.layer<<" values="<<oracle.direct.values
        <<" direct_max="<<oracle.direct.max_abs<<" direct_rel="<<oracle.direct.relative_l2
        <<" candidate_max="<<oracle.candidate.max_abs<<" candidate_rel="<<oracle.candidate.relative_l2<<'\n';
}

void fd_attention_replay(const FDTrace& trace,std::ofstream& report) {
    constexpr int capacity=4352,rows=16;
    std::vector<std::uint16_t> kp(static_cast<std::size_t>(capacity)*1024,0x7e00),vp=kp;
    std::copy(trace.k.begin(),trace.k.end(),kp.begin());std::copy(trace.v.begin(),trace.v.end(),vp.begin());
    DeviceBuffer q(trace.q.size()*2),k(kp.size()*2),v(vp.size()*2),a(rows*6144*2),b(rows*6144*2);
    DeviceBuffer scores(rows*24*capacity*4);
    const auto workspace_bytes=ninfer::exl3::exl3_numeric_attention_splitk_workspace_bytes(rows,capacity);
    DeviceBuffer workspace(workspace_bytes);
    cuda_check(cudaMemcpy(q.get(),trace.q.data(),trace.q.size()*2,cudaMemcpyHostToDevice),"FD Q upload");
    cuda_check(cudaMemcpy(k.get(),kp.data(),kp.size()*2,cudaMemcpyHostToDevice),"FD K upload");
    cuda_check(cudaMemcpy(v.get(),vp.data(),vp.size()*2,cudaMemcpyHostToDevice),"FD V upload");
    for(int first:{0,496,1008}) {
        const int position=16+first;
        const auto* query=static_cast<const std::uint16_t*>(q.get())+static_cast<std::size_t>(first)*6144;
        ninfer::exl3::exl3_exact_attention_for_test(query,static_cast<const std::uint16_t*>(k.get()),
            static_cast<const std::uint16_t*>(v.get()),static_cast<std::uint16_t*>(a.get()),
            static_cast<float*>(scores.get()),rows,position,capacity,true,nullptr,
            true,true,true,true,true,false,true,false,true,false,false,false,false,true);
        ninfer::exl3::exl3_numeric_attention_splitk_for_test(query,static_cast<const std::uint16_t*>(k.get()),
            static_cast<const std::uint16_t*>(v.get()),static_cast<std::uint16_t*>(b.get()),
            static_cast<float*>(workspace.get()),workspace_bytes,rows,position,capacity);
        const auto ah=target_continue_device_bits(static_cast<const std::uint16_t*>(a.get()),rows*6144,"FD exact attention");
        const auto bh=target_continue_device_bits(static_cast<const std::uint16_t*>(b.get()),rows*6144,"FD splitK attention");
        fd_write(report,"same_input_attention",3,position,6144,fd_metric(bh,ah));
        FDMetric am,bm;
        for(int row:{0,7,15}) for(int head=0;head<24;++head) {
            const int count=position+row+1,kh=head/6;
            std::vector<double> weights(count);double maximum=-std::numeric_limits<double>::infinity();
            for(int token=0;token<count;++token) {
                double dot=0;for(int d=0;d<256;++d)
                    dot+=static_cast<double>(half_to_float(trace.q[(static_cast<std::size_t>(first+row)*24+head)*256+d]))*
                        half_to_float(trace.k[(static_cast<std::size_t>(token)*4+kh)*256+d]);
                weights[token]=dot/16.0;maximum=std::max(maximum,weights[token]);
            }
            double denominator=0;for(auto& w:weights) {w=std::exp(w-maximum);denominator+=w;}
            for(int d=0;d<256;++d) {
                double expected=0;for(int token=0;token<count;++token)
                    expected+=weights[token]*half_to_float(trace.v[(static_cast<std::size_t>(token)*4+kh)*256+d]);
                expected/=denominator;const auto i=(row*24+head)*256+d;
                am.add(half_to_float(ah[i]),expected,false);bm.add(half_to_float(bh[i]),expected,false);
            }
        }
        fd_write(report,"attention_parent_fp64",3,position,256,am);
        fd_write(report,"attention_candidate_fp64",3,position,256,bm);
    }
}

void run_first_divergence(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    const std::string lane=env("NINFER_FD_LANE");
    require(lane=="projection"||lane=="attention","FD lane must be separate projection or attention");
    require(source.size()>=1040&&target.max_context()==4352,"FD real fixture extent");
    const std::filesystem::path dir=env("NINFER_FD_OUT");
    require(!dir.empty()&&!std::filesystem::exists(dir),"FD output must be new");
    std::filesystem::create_directories(dir);
    std::ofstream out(dir/"divergence.csv"),kvout(dir/"kv.csv");
    for(auto* file:{&out,&kvout}) *file<<std::setprecision(17)
        <<"operation,layer,position,columns,values,different,first_index,worst_index,max_abs,relative_l2\n";
    // Independent arithmetic half decoder, including subnormals, validates the comparator.
    for(unsigned bits=0;bits<65536;++bits) {
        const unsigned exp=(bits>>10)&31,frac=bits&1023;
        if(exp==31) continue;
        const double value=std::ldexp(static_cast<double>(exp?1024+frac:frac),exp?int(exp)-25:-24)*
            ((bits&32768)?-1:1);
        require(static_cast<double>(half_to_float(static_cast<std::uint16_t>(bits)))==value,"FD half decoder");
    }
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC","0");
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC","0");
    _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","0");
    FDTrace trace;trace.boundary=lane=="projection"?22:3;trace.report=&out;
    T69ProjectionCapture capture;
    auto parent=target.create_context(true);parent->prepare_continuation(8);
    parent->set_layer_observer_for_test(FDTrace::callback,&trace);
    if(lane=="projection") parent->set_target_projection_observer_for_test(T69ProjectionCapture::callback,
        &capture,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    t69b_ingest(*parent,std::span<const std::int64_t>(source.data(),1040));
    parent->set_layer_observer_for_test(nullptr);parent->set_target_projection_observer_for_test(nullptr);
    const auto parent_state=parent->export_exact_host_state();
    if(lane=="projection") fd_projection_replay(capture.items[1],out);
    else fd_attention_replay(trace,out);
    _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC",lane=="projection"?"1":"0");
    _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED",lane=="attention"?"1":"0");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    trace.candidate=true;candidate->set_layer_observer_for_test(FDTrace::callback,&trace);
    t69b_ingest(*candidate,std::span<const std::int64_t>(source.data(),1040));
    candidate->set_layer_observer_for_test(nullptr);
    const auto candidate_state=candidate->export_exact_host_state();
    const auto typed=fd_typed_kv(*candidate_state,*parent_state,kvout);
    const auto legacy=t69b_exact_kv_metric(candidate_state,parent_state);
    fd_write(out,"typed_KV",-1,1040,1024,typed);
    require(typed.values==legacy.values&&typed.max_abs==legacy.max_abs&&
            std::abs(typed.rel()-legacy.relative_l2)<1e-12,"FD legacy versus typed KV comparison");
    const auto logits=t69b_float_metric(candidate->logits_host(),parent->logits_host());
    std::cout<<"FD_COMPLETE lane="<<lane<<" first_layer="<<trace.first_layer
        <<" first_position="<<trace.first_position<<" retained_host_bytes="<<trace.retained_bytes
        <<" kv_max="<<typed.max_abs<<" kv_rel="<<typed.rel()<<" logits_max="<<logits.max_abs
        <<" logits_rel="<<logits.relative_l2<<" comparator_agrees=1 fixed_input_tokens=1040 timing=UNMEASURED\n";
}
