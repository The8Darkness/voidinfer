#pragma once

void capture_k6_stream_inputs(const ninfer::exl3::Exl3TargetProjectionObservation& item,void* user){
    if(item.metadata.K==6 && item.metadata.in_features!=17408 && item.metadata.out_features!=248320)
        capture_target_k6_projection(item,user);
}

void run_k6_stream_operator_gate(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose,const std::filesystem::path& dir,int prefix=512){
    using namespace ninfer::exl3;
    require(!std::filesystem::exists(dir),"preserve stream operator receipt");std::filesystem::create_directories(dir);
    std::ofstream checks(dir/"operators.csv");checks<<"fixture,record,layer,operation,rows,values,exact,repeat,guards,input_unchanged\n";
    std::array<std::unique_ptr<DeviceBuffer>,5> storage;std::array<std::uint16_t*,5> staging{};
    for(int t=0;t<5;++t){storage[t]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[t]=static_cast<std::uint16_t*>(storage[t]->get());}
    std::uint64_t values=0,cases=0;
    _putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION","0");
    for(int fixture=0;fixture<2;++fixture){
        const auto& input=fixture?prose:code;require(input.size()>=prefix,"operator prefix length");
        auto context=target.create_context(true);context->prepare_continuation(8);
        auto root=Exl3VeriCacheRequest::initialize(*context,std::span<const std::int64_t>(input.data(),prefix),1024)->compact_draft(draft,staging);
        root->restore_draft(draft,staging);std::vector<std::int64_t> block(8,248070);block[0]=sample_target(*context);
        auto proposed=draft.propose_cached(block,prefix,context->target_embedding(),context->target_lm_head_weights(),context->target_lm_head_metadata(),248070);
        std::copy(proposed.begin(),proposed.end(),block.begin()+1);
        std::vector<TargetK6CapturedProjection> captured;
        context->set_target_projection_observer_for_test(capture_k6_stream_inputs,&captured,nullptr,Exl3TargetProjectionObserverSelection::wide_prefill_all);
        try{context->continue_rows(block);}catch(...){context->set_target_projection_observer_for_test(nullptr);throw;}
        context->set_target_projection_observer_for_test(nullptr);require(!captured.empty(),"no real K6 activations");
        context.reset();root.reset();
        for(std::size_t index=0;index<captured.size();++index){
            const auto& item=captured[index];const auto& op=item.operation;
            auto make=[&](bool candidate){_putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION",candidate?"1":"0");
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
                if(std::string(candidate->dispatch_name(item.metadata,rows,admission))!="target_k6_small_m_stream_reduction")continue;
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
            require(candidate->stream_reduction_calls()>0,"candidate was not executed");
        }
        _putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION","0");
    }
    std::ofstream(dir/"result.json")<<nlohmann::ordered_json{{"status","PASS_K6_STREAM_OPERATORS"},{"cases",cases},{"values",values},{"artifacts","same pinned target/draft"}}.dump(2);
    std::cout<<"K6_STREAM_OPERATORS PASS cases="<<cases<<" values="<<values<<'\n'<<std::flush;
}
