#pragma once

struct PrefillShape4N64Qualification {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit=Admission::target_wide_prefill;
    static constexpr int max_rows=1024;
    static constexpr std::size_t guard=128;
    int cases=0;
    std::uint64_t values=0;

    PrefillShape4N64Qualification() {
        for(const char* name:{"NINFER_EXL3_WIDE_PREFILL",
            "NINFER_EXL3_PREFILL_STAGED_REDUCTION",
            "NINFER_EXL3_PREFILL_STAGED_SHAPE4",
            "NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
            "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A",
            "NINFER_EXL3_PREFILL_WIDE64","NINFER_EXL3_PREFILL_WIDE128",
            "NINFER_EXL3_PREFILL_WIDE256","NINFER_EXL3_PREFILL_WIDE512",
            "NINFER_EXL3_PREFILL_WIDE1024"})_putenv_s(name,"1");
        _putenv_s("NINFER_EXL3_PREFILL_SHAPE4_N64","0");
    }
    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        static_cast<PrefillShape4N64Qualification*>(user)->check(x);
    }
    static std::unique_ptr<Exl3CudaLinearWorkspace> workspace(bool candidate) {
        _putenv_s("NINFER_EXL3_PREFILL_SHAPE4_N64",candidate?"1":"0");
        return std::make_unique<Exl3CudaLinearWorkspace>(
            5120,17408,max_rows,false,true,false,false,false,true,false);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if(cases||x.rows<max_rows||x.metadata.K!=5||
           x.metadata.in_features!=5120||x.metadata.out_features!=17408)return;
        cuda_check(cudaStreamSynchronize(x.stream),"shape4 N64 input ready");
        auto control=workspace(false),candidate=workspace(true);
        _putenv_s("NINFER_EXL3_PREFILL_SHAPE4_N64","0");
        require(!candidate->target_shape4_n64_candidate(x.metadata,31,admit)&&
                candidate->target_shape4_n64_candidate(x.metadata,32,admit)&&
                candidate->target_shape4_n64_candidate(x.metadata,max_rows,admit)&&
                !candidate->target_shape4_n64_candidate(
                    x.metadata,max_rows,Admission::ordinary),
                "shape4 N64 admission boundary");
        const std::size_t input_extent=static_cast<std::size_t>(max_rows)*5120;
        std::vector<std::uint16_t> input_before(input_extent),input_after(input_extent);
        cuda_check(cudaMemcpy(input_before.data(),x.input,input_extent*2,
            cudaMemcpyDeviceToHost),"shape4 N64 input snapshot");
        std::uint64_t expected_calls=0,expected_rows=0;
        for(int rows:{32,33,1009,1024}) {
            const std::size_t active=static_cast<std::size_t>(rows)*17408;
            const std::vector<std::uint16_t> initial(active+2*guard,0x3555);
            std::vector<std::uint16_t> a(initial.size()),b(initial.size()),repeat(initial.size());
            DeviceBuffer old_output(initial.size()*2),new_output(initial.size()*2);
            auto* old_base=static_cast<std::uint16_t*>(old_output.get());
            auto* new_base=static_cast<std::uint16_t*>(new_output.get());
            cuda_check(cudaMemcpy(old_base,initial.data(),initial.size()*2,
                cudaMemcpyHostToDevice),"shape4 N64 control guards");
            cuda_check(cudaMemcpy(new_base,initial.data(),initial.size()*2,
                cudaMemcpyHostToDevice),"shape4 N64 candidate guards");
            require(std::string(control->dispatch_name(x.metadata,rows,admit))==
                        "target_wide_prefill_staged"&&
                    std::string(candidate->dispatch_name(x.metadata,rows,admit))==
                        "target_wide_prefill_shape4_n64",
                    "shape4 N64 route selection");
            control->forward(x.weights,x.metadata,x.input,old_base+guard,rows,x.stream,admit);
            candidate->forward(x.weights,x.metadata,x.input,new_base+guard,rows,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"shape4 N64 compare ready");
            cuda_check(cudaMemcpy(a.data(),old_base,a.size()*2,cudaMemcpyDeviceToHost),
                "shape4 N64 control output");
            cuda_check(cudaMemcpy(b.data(),new_base,b.size()*2,cudaMemcpyDeviceToHost),
                "shape4 N64 candidate output");
            candidate->forward(x.weights,x.metadata,x.input,new_base+guard,rows,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"shape4 N64 repeat ready");
            cuda_check(cudaMemcpy(repeat.data(),new_base,repeat.size()*2,cudaMemcpyDeviceToHost),
                "shape4 N64 repeat output");
            cuda_check(cudaMemcpy(input_after.data(),x.input,input_extent*2,
                cudaMemcpyDeviceToHost),"shape4 N64 input after");
            expected_calls+=2;expected_rows+=2*rows;
            require(a==b&&b==repeat&&input_before==input_after&&
                    std::all_of(b.begin(),b.begin()+guard,[](auto v){return v==0x3555;})&&
                    std::all_of(b.end()-guard,b.end(),[](auto v){return v==0x3555;})&&
                    control->shape4_n64_calls()==0&&
                    candidate->shape4_n64_calls()==expected_calls&&
                    candidate->shape4_n64_rows()==expected_rows,
                    "shape4 N64 bit-exact differential");
            ++cases;values+=active;
        }
    }
};

void run_prefill_shape4_n64_qualification(Exl3TextModel& target) {
    for(const char* malformed:{"2","true","01"}) {
        _putenv_s("NINFER_EXL3_PREFILL_SHAPE4_N64",malformed);
        bool rejected=false;
        try {
            Exl3CudaLinearWorkspace invalid(
                5120,17408,PrefillShape4N64Qualification::max_rows,
                false,true,false,false,false,true,false);
        }catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"shape4 N64 accepted malformed opt-in");
    }
    PrefillShape4N64Qualification qualification;
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size()>=16+PrefillShape4N64Qualification::max_rows,
        "shape4 N64 prompt extent");
    auto context=target.create_context(true);
    context->prefill(std::span<const std::int64_t>(ids.data(),16));
    context->set_target_projection_observer_for_test(
        PrefillShape4N64Qualification::callback,&qualification,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(std::span<const std::int64_t>(
        ids.data()+16,PrefillShape4N64Qualification::max_rows));
    context->finish_exact_prefill();
    cuda_check(cudaDeviceSynchronize(),"shape4 N64 qualification complete");
    context->set_target_projection_observer_for_test(nullptr,nullptr);
    require(qualification.cases==4,"shape4 N64 real-caller coverage");
    std::cout<<"PREFILL_SHAPE4_N64 PASS operations=1 cases="
        <<qualification.cases<<" values="<<qualification.values
        <<" exact=1 repeat=1 guards=1 input_unchanged=1 counters=1\n";
}
