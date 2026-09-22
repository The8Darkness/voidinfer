#pragma once

void capture_extended_stream_inputs(const ninfer::exl3::Exl3TargetProjectionObservation& item,void* user){
    if((item.metadata.K==7 || (item.metadata.K==6 && item.metadata.in_features==17408)) && item.metadata.out_features!=248320)
        capture_target_k6_projection(item,user);
}

void run_extended_stream_operator_gate(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose,const std::filesystem::path& dir,int prefix=512){
    using namespace ninfer::exl3;
    require(!std::filesystem::exists(dir),"preserve stream operator receipt");std::filesystem::create_directories(dir);
    std::ofstream checks(dir/"operators.csv");checks<<"fixture,record,layer,operation,rows,values,exact,repeat,guards,input_unchanged\n";
    std::array<std::unique_ptr<DeviceBuffer>,5> storage;std::array<std::uint16_t*,5> staging{};
    for(int t=0;t<5;++t){storage[t]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[t]=static_cast<std::uint16_t*>(storage[t]->get());}
    std::uint64_t values=0,cases=0;
    _putenv_s("NINFER_EXL3_EXTENDED_STREAM_REDUCTION","0");
    for(int fixture=0;fixture<2;++fixture){
        const auto& input=fixture?prose:code;require(input.size()>=prefix,"operator prefix length");
        auto context=target.create_context(true);context->prepare_continuation(8);
        auto root=Exl3VeriCacheRequest::initialize(*context,std::span<const std::int64_t>(input.data(),prefix),1024)->compact_draft(draft,staging);
        root->restore_draft(draft,staging);std::vector<std::int64_t> block(8,248070);block[0]=sample_target(*context);
        auto proposed=draft.propose_cached(block,prefix,context->target_embedding(),context->target_lm_head_weights(),context->target_lm_head_metadata(),248070);
        std::copy(proposed.begin(),proposed.end(),block.begin()+1);
        std::vector<TargetK6CapturedProjection> captured;
        context->set_target_projection_observer_for_test(capture_extended_stream_inputs,&captured,nullptr,Exl3TargetProjectionObserverSelection::wide_prefill_all);
        try{context->continue_rows(block);}catch(...){context->set_target_projection_observer_for_test(nullptr);throw;}
        context->set_target_projection_observer_for_test(nullptr);require(!captured.empty(),"no real K6 activations");
        context.reset();root.reset();
        for(std::size_t index=0;index<captured.size();++index){
            const auto& item=captured[index];const auto& op=item.operation;
            auto make=[&](bool candidate){_putenv_s("NINFER_EXL3_EXTENDED_STREAM_REDUCTION",candidate?"1":"0");
                return std::make_unique<Exl3CudaLinearWorkspace>(item.metadata.in_features,item.metadata.out_features,8,false,
                    op=="gate"||op=="up",op=="down",op=="o",op=="z",true,false,Exl3CudaAccumulationView{},Exl3CudaTransformView{},op=="qkv",op=="k"||op=="v",op=="q");};
            auto control=make(false),candidate=make(true);require(control->workspace_bytes()==candidate->workspace_bytes(),"K6 stream reserve changed");
            constexpr std::size_t guard=128;constexpr std::uint16_t sentinel=0x3555;
            const auto width=item.metadata.out_features;const std::size_t capacity=8ULL*width;
            DeviceBuffer device_input(item.input.size()*2),old_output((capacity+2*guard)*2),new_output((capacity+2*guard)*2);
            cuda_check(cudaMemcpy(device_input.get(),item.input.data(),item.input.size()*2,cudaMemcpyHostToDevice),"K6 real activation upload");
            auto* old_base=static_cast<std::uint16_t*>(old_output.get());auto* new_base=static_cast<std::uint16_t*>(new_output.get());
            const std::vector<std::uint16_t> initial(capacity+2*guard,sentinel);std::vector<std::uint16_t> a(initial.size()),b(initial.size()),again(initial.size()),input_after(item.input.size());
            for(int rows:{1,2,3,5,7,8}){
                if(std::size_t(rows)*item.metadata.in_features>item.input.size())continue;
                auto admission=rows==1?Exl3CudaLinearAdmission::ordinary:projection_route_serial_admission(op);
                if(std::string(candidate->dispatch_name(item.metadata,rows,admission))!="target_extended_stream_reduction")continue;
                cuda_check(cudaMemcpy(old_base,initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"K6 old guards");
                cuda_check(cudaMemcpy(new_base,initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"K6 new guards");
                control->forward(item.weights,item.metadata,static_cast<const std::uint16_t*>(device_input.get()),old_base+guard,rows,nullptr,admission);
                candidate->forward(item.weights,item.metadata,static_cast<const std::uint16_t*>(device_input.get()),new_base+guard,rows,nullptr,admission);
                cuda_check(cudaMemcpy(a.data(),old_base,a.size()*2,cudaMemcpyDeviceToHost),"K6 old output");
                cuda_check(cudaMemcpy(b.data(),new_base,b.size()*2,cudaMemcpyDeviceToHost),"K6 new output");
                candidate->forward(item.weights,item.metadata,static_cast<const std::uint16_t*>(device_input.get()),new_base+guard,rows,nullptr,admission);
                cuda_check(cudaMemcpy(again.data(),new_base,again.size()*2,cudaMemcpyDeviceToHost),"K6 repeated output");
                cuda_check(cudaMemcpy(input_after.data(),device_input.get(),item.input.size()*2,cudaMemcpyDeviceToHost),"K6 input unchanged");
                const auto active=std::size_t(rows)*width;
                const bool guarded=std::all_of(b.begin(),b.begin()+guard,[&](auto x){return x==sentinel;})&&std::all_of(b.begin()+guard+active,b.end(),[&](auto x){return x==sentinel;});
                const bool exact=a==b,repeat=b==again,unchanged=input_after==item.input;
                checks<<(fixture?"prose":"code")<<','<<index<<','<<item.layer<<','<<op<<','<<rows<<','<<active<<','<<exact<<','<<repeat<<','<<guarded<<','<<unchanged<<'\n';checks.flush();
                require(exact&&repeat&&guarded&&unchanged,"K6 stream operator exactness gate");values+=active;++cases;
            }
            require(candidate->stream_reduction_calls(true)>0,"candidate was not executed");
        }
        _putenv_s("NINFER_EXL3_EXTENDED_STREAM_REDUCTION","0");
    }
    std::ofstream(dir/"result.json")<<nlohmann::ordered_json{{"status","PASS_EXTENDED_STREAM_OPERATORS"},{"cases",cases},{"values",values},{"artifacts","same pinned target/draft"}}.dump(2);
    std::cout<<"EXTENDED_STREAM_OPERATORS PASS cases="<<cases<<" values="<<values<<'\n'<<std::flush;
}

void capture_target_k6_small_m_async_inputs(
    const ninfer::exl3::Exl3TargetProjectionObservation& item,void* user) {
    if((item.metadata.K==6 || item.metadata.K==7) && item.metadata.in_features!=17408 &&
       item.metadata.out_features!=248320)
        capture_target_k6_projection(item,user);
}

class TargetK6SmallMAsyncEnvironmentRestore {
public:
    TargetK6SmallMAsyncEnvironmentRestore()
        : stream_(env("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION")),
          async_(env("NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A")),
          k7_async_(env("NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A")) {}
    ~TargetK6SmallMAsyncEnvironmentRestore() {
        _putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION",stream_.c_str());
        _putenv_s("NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A",async_.c_str());
        _putenv_s("NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A",k7_async_.c_str());
    }
private:
    std::string stream_,async_,k7_async_;
};

void run_target_k6_small_m_async_qualification(
    Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& prompt,std::ostream& out) {
    using namespace ninfer::exl3;
    require(prompt.size()==512,"target K6 small-M async-A requires ctx512");
    TargetK6SmallMAsyncEnvironmentRestore restore;
    TapStage stage;
    for(int tap=0;tap<kTapCount;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(16)*kHidden*sizeof(std::uint16_t)));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(kHidden)*sizeof(std::uint16_t)));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    auto context=target.create_context(true);
    require(context->try_enable_oscar_from_environment(),
        "target K6 small-M async-A could not enable OSCAR");
    draft.reset();
    ingest_prefix(*context,prompt,CommitSink{&draft,&stage});
    require(context->position()==512&&context->device_position_host()==511&&
            draft.ring_count()==512&&draft.ring_base_abs()==0,
        "target K6 small-M async-A prefix state mismatch");
    const auto pending=sample_target(*context);
    std::vector<std::int64_t> block(8,kMaskToken);
    block[0]=pending;
    const auto proposed=draft.propose_cached(block,512,context->target_embedding(),
        context->target_lm_head_weights(),context->target_lm_head_metadata(),kMaskToken);
    require(proposed.size()==7,"target K6 small-M async-A proposal count mismatch");
    std::copy(proposed.begin(),proposed.end(),block.begin()+1);
    context->prepare_transaction();
    context->prepare_continuation(8);
    std::vector<TargetK6CapturedProjection> captured;
    context->set_target_projection_observer_for_test(
        capture_target_k6_small_m_async_inputs,&captured,nullptr,
        Exl3TargetProjectionObserverSelection::wide_prefill_all);
    try { context->begin_transaction();context->continue_rows(block); }
    catch(...) { context->set_target_projection_observer_for_test(nullptr);throw; }
    context->set_target_projection_observer_for_test(nullptr);
    context->rollback_transaction();
    cuda_check(cudaStreamSynchronize(nullptr),
        "target K6 small-M async-A capture rollback synchronize");
    require(!context->transaction_active()&&context->position()==512&&
            context->device_position_host()==511&&sample_target(*context)==pending,
        "target K6 small-M async-A capture did not restore target state");
    require(!captured.empty(),"target K6 small-M async-A captured no real projections");

    out<<"record,layer,operation,rows,values,control_route,candidate_route,exact,repeat,guards,input_unchanged\n";
    constexpr std::size_t guard=128;
    constexpr std::uint16_t sentinel=0x3555;
    std::uint64_t cases=0,values=0,candidate_calls=0,k7_cases=0,k7_candidate_calls=0;
    bool strict_checked=false;
    for(std::size_t index=0;index<captured.size();++index) {
        const auto& item=captured[index];
        const auto& op=item.operation;
        const auto make=[&](bool candidate) {
            _putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION","1");
            _putenv_s("NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A",candidate?"1":"0");
            _putenv_s("NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A",candidate?"1":"0");
            return std::make_unique<Exl3CudaLinearWorkspace>(
                item.metadata.in_features,item.metadata.out_features,8,false,
                op=="gate"||op=="up",op=="down",op=="o",op=="z",true,false,
                Exl3CudaAccumulationView{},Exl3CudaTransformView{},op=="qkv",
                op=="k"||op=="v",op=="q");
        };
        auto control=make(false),candidate=make(true);
        if(!strict_checked) {
            for(const char* invalid:{"2","true","01"}) {
                _putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION","1");
                _putenv_s("NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A",invalid);
                _putenv_s("NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A","0");
                bool rejected=false;
                try {
                    Exl3CudaLinearWorkspace invalid_ws(
                        item.metadata.in_features,item.metadata.out_features,8,false,
                        op=="gate"||op=="up",op=="down",op=="o",op=="z",true,false,
                        Exl3CudaAccumulationView{},Exl3CudaTransformView{},op=="qkv",
                        op=="k"||op=="v",op=="q");
                } catch(const std::invalid_argument&) { rejected=true; }
                require(rejected,"target K6 small-M async-A accepted non-exact flag");
                _putenv_s("NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A","0");
                _putenv_s("NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A",invalid);
                rejected=false;
                try {
                    Exl3CudaLinearWorkspace invalid_ws(
                        item.metadata.in_features,item.metadata.out_features,8,false,
                        op=="gate"||op=="up",op=="down",op=="o",op=="z",true,false,
                        Exl3CudaAccumulationView{},Exl3CudaTransformView{},op=="qkv",
                        op=="k"||op=="v",op=="q");
                } catch(const std::invalid_argument&) { rejected=true; }
                require(rejected,"target K7 small-M async-A accepted non-exact flag");
            }
            strict_checked=true;
        }
        _putenv_s("NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A","1");
        _putenv_s("NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A","1");
        const auto width=static_cast<std::size_t>(item.metadata.out_features);
        const auto capacity=8u*width;
        DeviceBuffer input(item.input.size()*sizeof(std::uint16_t));
        DeviceBuffer old_output((capacity+2*guard)*sizeof(std::uint16_t));
        DeviceBuffer new_output((capacity+2*guard)*sizeof(std::uint16_t));
        cuda_check(cudaMemcpy(input.get(),item.input.data(),item.input.size()*sizeof(std::uint16_t),
            cudaMemcpyHostToDevice),"target K6 small-M async-A input upload");
        const std::vector<std::uint16_t> initial(capacity+2*guard,sentinel);
        std::vector<std::uint16_t> a(initial.size()),b(initial.size()),again(initial.size());
        std::vector<std::uint16_t> input_after(item.input.size());
        auto* old_base=static_cast<std::uint16_t*>(old_output.get());
        auto* new_base=static_cast<std::uint16_t*>(new_output.get());
        for(int rows:{2,3,5,7,8}) {
            if(static_cast<std::size_t>(rows)*item.metadata.in_features>item.input.size())continue;
            const auto admission=projection_route_serial_admission(op);
            const std::string control_route=control->dispatch_name(item.metadata,rows,admission);
            const std::string candidate_route=candidate->dispatch_name(item.metadata,rows,admission);
            const std::string expected_candidate=item.metadata.K==6
                ? "target_k6_small_m_async_a_stream_reduction"
                : "target_k7_small_m_async_a_stream_reduction";
            if(candidate_route!=expected_candidate)continue;
            const std::string expected_control=item.metadata.K==6
                ? "target_k6_small_m_stream_reduction"
                : "target_extended_stream_reduction";
            require(control_route==expected_control,
                "target K6/K7 small-M async-A control route mismatch");
            cuda_check(cudaMemcpy(old_base,initial.data(),initial.size()*sizeof(std::uint16_t),
                cudaMemcpyHostToDevice),"target K6 small-M async-A control guards");
            cuda_check(cudaMemcpy(new_base,initial.data(),initial.size()*sizeof(std::uint16_t),
                cudaMemcpyHostToDevice),"target K6 small-M async-A candidate guards");
            control->forward(item.weights,item.metadata,static_cast<const std::uint16_t*>(input.get()),
                old_base+guard,rows,nullptr,admission);
            candidate->forward(item.weights,item.metadata,static_cast<const std::uint16_t*>(input.get()),
                new_base+guard,rows,nullptr,admission);
            cuda_check(cudaMemcpy(a.data(),old_base,a.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                "target K6 small-M async-A control output");
            cuda_check(cudaMemcpy(b.data(),new_base,b.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                "target K6 small-M async-A candidate output");
            candidate->forward(item.weights,item.metadata,static_cast<const std::uint16_t*>(input.get()),
                new_base+guard,rows,nullptr,admission);
            cuda_check(cudaMemcpy(again.data(),new_base,again.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                "target K6 small-M async-A repeat output");
            cuda_check(cudaMemcpy(input_after.data(),input.get(),item.input.size()*sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost),"target K6 small-M async-A input download");
            const auto active=static_cast<std::size_t>(rows)*width;
            const bool guards=std::all_of(b.begin(),b.begin()+guard,[&](auto v){return v==sentinel;})&&
                std::all_of(b.begin()+guard+active,b.end(),[&](auto v){return v==sentinel;});
            const bool exact=a==b,repeat=b==again,unchanged=input_after==item.input;
            out<<index<<','<<item.layer<<','<<op<<','<<rows<<','<<active<<','
               <<control_route<<','<<candidate_route<<','<<exact<<','<<repeat<<','
               <<guards<<','<<unchanged<<'\n';out.flush();
            require(exact&&repeat&&guards&&unchanged,
                "target K6 small-M async-A exact differential failed");
            values+=active;++cases;if(item.metadata.K==7)++k7_cases;
        }
        require(control->target_k6_small_m_async_a_calls()==0,
            "target K6 small-M async-A control counter nonzero");
        candidate_calls+=candidate->target_k6_small_m_async_a_calls();
        require(control->target_k7_small_m_async_a_calls()==0,
            "target K7 small-M async-A control counter nonzero");
        k7_candidate_calls+=candidate->target_k7_small_m_async_a_calls();
    }
    require(cases>k7_cases&&k7_cases>0&&candidate_calls==2*(cases-k7_cases)&&
            k7_candidate_calls==2*k7_cases,
        "target K6/K7 small-M async-A dispatch counter mismatch");
    std::cout<<"TARGET_K6_SMALL_M_ASYNC_A PASS cases="<<cases
             <<" k7_cases="<<k7_cases<<" values="<<values
             <<" calls="<<candidate_calls<<" k7_calls="<<k7_candidate_calls
             <<" strict_flag=1 exact=1 repeat=1 guards=1 input_unchanged=1\n"<<std::flush;
}
