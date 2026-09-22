#pragma once

struct PrefillK6RowpairN64Qualification {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit=Admission::target_wide_prefill;
    static constexpr int max_rows=1024;
    static constexpr std::size_t guard=128;
    std::set<std::string> operations;
    int cases=0;
    std::uint64_t values=0;

    PrefillK6RowpairN64Qualification() {
        for(const char* name:{"NINFER_EXL3_WIDE_PREFILL",
            "NINFER_EXL3_PREFILL_STAGED_REDUCTION",
            "NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
            "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A",
            "NINFER_EXL3_PREFILL_DIRECT_TILES64",
            "NINFER_EXL3_PREFILL_WIDE64","NINFER_EXL3_PREFILL_WIDE128",
            "NINFER_EXL3_PREFILL_WIDE256","NINFER_EXL3_PREFILL_WIDE512",
            "NINFER_EXL3_PREFILL_WIDE1024"})_putenv_s(name,"1");
        _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6","0");
        _putenv_s("NINFER_EXL3_PREFILL_K6_GATEUP_WARPGROUP_ASYNC","0");
        _putenv_s("NINFER_EXL3_PREFILL_K6_GATEUP_N32_PAIR_CTA","0");
        _putenv_s("NINFER_EXL3_PREFILL_K6_MOD48_FAST_DECODE","0");
        _putenv_s("NINFER_EXL3_PREFILL_K6_ROWPAIR_N64","0");
    }

    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        static_cast<PrefillK6RowpairN64Qualification*>(user)->check(x);
    }
    static std::unique_ptr<Exl3CudaLinearWorkspace> workspace(bool candidate) {
        _putenv_s("NINFER_EXL3_PREFILL_K6_ROWPAIR_N64",candidate?"1":"0");
        return std::make_unique<Exl3CudaLinearWorkspace>(
            5120,17408,max_rows,false,true,false,false,false,true,false);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        const std::string operation=x.operation;
        if(x.rows<max_rows||x.metadata.K!=6||x.metadata.in_features!=5120||
           x.metadata.out_features!=17408||
           (operation!="gate"&&operation!="up")||
           !operations.emplace(operation).second)return;
        cuda_check(cudaStreamSynchronize(x.stream),"K6 rowpair N64 input ready");
        auto control=workspace(false),candidate=workspace(true);
        _putenv_s("NINFER_EXL3_PREFILL_K6_ROWPAIR_N64","0");
        require(!candidate->target_k6_rowpair_n64_candidate(x.metadata,31,admit)&&
                candidate->target_k6_rowpair_n64_candidate(x.metadata,32,admit)&&
                candidate->target_k6_rowpair_n64_candidate(x.metadata,max_rows,admit)&&
                !candidate->target_k6_rowpair_n64_candidate(
                    x.metadata,max_rows,Admission::ordinary),
                "K6 rowpair N64 admission boundary");
        const std::size_t input_extent=static_cast<std::size_t>(max_rows)*5120;
        std::vector<std::uint16_t> input_before(input_extent),input_after(input_extent);
        cuda_check(cudaMemcpy(input_before.data(),x.input,input_extent*2,
            cudaMemcpyDeviceToHost),"K6 rowpair N64 input snapshot");
        std::uint64_t expected_calls=0,expected_rows=0;
        for(int rows:{32,33,1009,1024}) {
            const std::size_t active=static_cast<std::size_t>(rows)*17408;
            const std::vector<std::uint16_t> initial(active+2*guard,0x3555);
            std::vector<std::uint16_t> a(initial.size()),b(initial.size()),repeat(initial.size());
            DeviceBuffer old_output(initial.size()*2),new_output(initial.size()*2);
            auto* old_base=static_cast<std::uint16_t*>(old_output.get());
            auto* new_base=static_cast<std::uint16_t*>(new_output.get());
            cuda_check(cudaMemcpy(old_base,initial.data(),initial.size()*2,
                cudaMemcpyHostToDevice),"K6 rowpair N64 control guards");
            cuda_check(cudaMemcpy(new_base,initial.data(),initial.size()*2,
                cudaMemcpyHostToDevice),"K6 rowpair N64 candidate guards");
            require(std::string(control->dispatch_name(x.metadata,rows,admit))==
                        "target_wide_prefill_staged"&&
                    std::string(candidate->dispatch_name(x.metadata,rows,admit))==
                        "target_wide_prefill_k6_rowpair_n64",
                    "K6 rowpair N64 route selection");
            control->forward(x.weights,x.metadata,x.input,old_base+guard,rows,x.stream,admit);
            candidate->forward(x.weights,x.metadata,x.input,new_base+guard,rows,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"K6 rowpair N64 compare ready");
            cuda_check(cudaMemcpy(a.data(),old_base,a.size()*2,cudaMemcpyDeviceToHost),
                "K6 rowpair N64 control output");
            cuda_check(cudaMemcpy(b.data(),new_base,b.size()*2,cudaMemcpyDeviceToHost),
                "K6 rowpair N64 candidate output");
            candidate->forward(x.weights,x.metadata,x.input,new_base+guard,rows,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"K6 rowpair N64 repeat ready");
            cuda_check(cudaMemcpy(repeat.data(),new_base,repeat.size()*2,cudaMemcpyDeviceToHost),
                "K6 rowpair N64 repeat output");
            cuda_check(cudaMemcpy(input_after.data(),x.input,input_extent*2,
                cudaMemcpyDeviceToHost),"K6 rowpair N64 input after");
            expected_calls+=2;expected_rows+=2*rows;
            require(a==b&&b==repeat&&input_before==input_after&&
                    std::all_of(b.begin(),b.begin()+guard,[](auto v){return v==0x3555;})&&
                    std::all_of(b.end()-guard,b.end(),[](auto v){return v==0x3555;})&&
                    control->k6_rowpair_n64_calls()==0&&
                    candidate->k6_rowpair_n64_calls()==expected_calls&&
                    candidate->k6_rowpair_n64_rows()==expected_rows,
                    "K6 rowpair N64 bit-exact differential");
            ++cases;values+=active;
        }
    }
};

void run_prefill_k6_rowpair_n64_qualification(Exl3TextModel& target) {
    for(const char* malformed:{"2","true","01"}) {
        _putenv_s("NINFER_EXL3_PREFILL_K6_ROWPAIR_N64",malformed);
        bool rejected=false;
        try {
            Exl3CudaLinearWorkspace invalid(5120,17408,
                PrefillK6RowpairN64Qualification::max_rows,
                false,true,false,false,false,true,false);
        }catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"K6 rowpair N64 accepted malformed opt-in");
    }
    PrefillK6RowpairN64Qualification qualification;
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size()>=16+PrefillK6RowpairN64Qualification::max_rows,
        "K6 rowpair N64 prompt extent");
    auto context=target.create_context(true);
    context->prefill(std::span<const std::int64_t>(ids.data(),16));
    context->set_target_projection_observer_for_test(
        PrefillK6RowpairN64Qualification::callback,&qualification,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(std::span<const std::int64_t>(
        ids.data()+16,PrefillK6RowpairN64Qualification::max_rows));
    context->finish_exact_prefill();
    cuda_check(cudaDeviceSynchronize(),"K6 rowpair N64 qualification complete");
    context->set_target_projection_observer_for_test(nullptr,nullptr);
    require(qualification.operations==std::set<std::string>{"gate","up"}&&
            qualification.cases==8,"K6 rowpair N64 real-caller coverage");
    std::cout<<"PREFILL_K6_ROWPAIR_N64 PASS operations=2 cases="
        <<qualification.cases<<" values="<<qualification.values
        <<" exact=1 repeat=1 guards=1 input_unchanged=1 counters=1\n";
}

struct PrefillK6SharedDecodeDiscriminator {
    bool complete=false;
    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        static_cast<PrefillK6SharedDecodeDiscriminator*>(user)->run(x);
    }
    void run(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if(complete||x.rows!=1024||x.metadata.K!=6||
           x.metadata.in_features!=5120||x.metadata.out_features!=17408||
           std::string(x.operation)!="gate")return;
        constexpr std::size_t guard=128;
        constexpr std::uint16_t sentinel=0x3555;
        constexpr int samples=15,repetitions=4;
        cuda_check(cudaStreamSynchronize(x.stream),
            "K6 shared-decode discriminator input ready");
        auto selected=PrefillK6RowpairN64Qualification::workspace(true);
        auto candidate=PrefillK6RowpairN64Qualification::workspace(true);
        const std::size_t input_values=static_cast<std::size_t>(x.rows)*
            x.metadata.in_features;
        const std::size_t output_values=static_cast<std::size_t>(x.rows)*
            x.metadata.out_features;
        const std::size_t guarded_values=output_values+2*guard;
        std::vector<std::uint16_t> input_before(input_values),input_after(input_values);
        std::vector<std::uint16_t> canary(guarded_values,sentinel),
            reference(guarded_values),actual(guarded_values),repeat(guarded_values);
        DeviceBuffer selected_output(guarded_values*sizeof(std::uint16_t));
        DeviceBuffer candidate_output(guarded_values*sizeof(std::uint16_t));
        auto* selected_base=static_cast<std::uint16_t*>(selected_output.get());
        auto* candidate_base=static_cast<std::uint16_t*>(candidate_output.get());
        cuda_check(cudaMemcpy(input_before.data(),x.input,input_values*2,
            cudaMemcpyDeviceToHost),"K6 shared-decode input snapshot");
        cuda_check(cudaMemcpy(selected_base,canary.data(),guarded_values*2,
            cudaMemcpyHostToDevice),"K6 shared-decode selected guards");
        cuda_check(cudaMemcpy(candidate_base,canary.data(),guarded_values*2,
            cudaMemcpyHostToDevice),"K6 shared-decode candidate guards");
        selected->forward(x.weights,x.metadata,x.input,selected_base+guard,x.rows,
            x.stream,PrefillK6RowpairN64Qualification::admit);
        candidate->forward_k6_rowpair_shared_decode_for_test(
            x.weights,x.metadata,x.input,candidate_base+guard,x.rows,x.stream);
        cuda_check(cudaStreamSynchronize(x.stream),
            "K6 shared-decode compare ready");
        cuda_check(cudaMemcpy(reference.data(),selected_base,guarded_values*2,
            cudaMemcpyDeviceToHost),"K6 shared-decode selected output");
        cuda_check(cudaMemcpy(actual.data(),candidate_base,guarded_values*2,
            cudaMemcpyDeviceToHost),"K6 shared-decode candidate output");
        cuda_check(cudaMemcpy(candidate_base,canary.data(),guarded_values*2,
            cudaMemcpyHostToDevice),"K6 shared-decode repeat guards");
        candidate->forward_k6_rowpair_shared_decode_for_test(
            x.weights,x.metadata,x.input,candidate_base+guard,x.rows,x.stream);
        cuda_check(cudaStreamSynchronize(x.stream),
            "K6 shared-decode repeat ready");
        cuda_check(cudaMemcpy(repeat.data(),candidate_base,guarded_values*2,
            cudaMemcpyDeviceToHost),"K6 shared-decode repeat output");
        cuda_check(cudaMemcpy(input_after.data(),x.input,input_values*2,
            cudaMemcpyDeviceToHost),"K6 shared-decode input after");
        require(reference==actual&&actual==repeat&&input_before==input_after&&
            std::all_of(actual.begin(),actual.begin()+guard,
                [](auto value){return value==sentinel;})&&
            std::all_of(actual.end()-guard,actual.end(),
                [](auto value){return value==sentinel;}),
            "K6 shared-decode discriminator exactness contract");
        cudaEvent_t begin=nullptr,end=nullptr;
        cuda_check(cudaEventCreate(&begin),"create K6 shared-decode begin event");
        cuda_check(cudaEventCreate(&end),"create K6 shared-decode end event");
        auto destroy=[&](){if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);};
        try {
            for(int warmup=0;warmup<2;++warmup) {
                selected->forward(x.weights,x.metadata,x.input,selected_base+guard,
                    x.rows,x.stream,PrefillK6RowpairN64Qualification::admit);
                candidate->forward_k6_rowpair_shared_decode_for_test(
                    x.weights,x.metadata,x.input,candidate_base+guard,x.rows,x.stream);
            }
            const auto measure=[&](bool shared_decode) {
                cuda_check(cudaEventRecord(begin,x.stream),
                    "record K6 shared-decode sample begin");
                for(int repetition=0;repetition<repetitions;++repetition) {
                    if(shared_decode)
                        candidate->forward_k6_rowpair_shared_decode_for_test(
                            x.weights,x.metadata,x.input,candidate_base+guard,x.rows,x.stream);
                    else selected->forward(x.weights,x.metadata,x.input,
                        selected_base+guard,x.rows,x.stream,
                        PrefillK6RowpairN64Qualification::admit);
                }
                cuda_check(cudaEventRecord(end,x.stream),
                    "record K6 shared-decode sample end");
                cuda_check(cudaEventSynchronize(end),
                    "synchronize K6 shared-decode sample");
                float milliseconds=0.0f;
                cuda_check(cudaEventElapsedTime(&milliseconds,begin,end),
                    "resolve K6 shared-decode sample");
                return static_cast<double>(milliseconds)*1000.0/repetitions;
            };
            std::vector<double> selected_samples,shared_samples;
            for(int sample=0;sample<samples;++sample) {
                if((sample&1)==0) {
                    selected_samples.push_back(measure(false));
                    shared_samples.push_back(measure(true));
                } else {
                    shared_samples.push_back(measure(true));
                    selected_samples.push_back(measure(false));
                }
            }
            std::sort(selected_samples.begin(),selected_samples.end());
            std::sort(shared_samples.begin(),shared_samples.end());
            const double selected_us=selected_samples[samples/2];
            const double shared_us=shared_samples[samples/2];
            std::cout<<std::fixed<<std::setprecision(3)
                <<"PREFILL_K6_SHARED_DECODE_DISCRIMINATOR PASS"
                <<" rows="<<x.rows<<" in_features="<<x.metadata.in_features
                <<" out_features="<<x.metadata.out_features
                <<" selected_median_us="<<selected_us
                <<" shared_decode_median_us="<<shared_us
                <<" gain_percent="<<(100.0*(selected_us-shared_us)/selected_us)
                <<" values="<<output_values
                <<" exact=1 repeat=1 guards=1 input_unchanged=1"<<std::endl;
        } catch(...) {destroy();throw;}
        destroy();complete=true;
    }
};

void run_prefill_k6_shared_decode_discriminator(Exl3TextModel& target) {
    PrefillK6RowpairN64Qualification environment;
    _putenv_s("NINFER_EXL3_PREFILL_K6_ROWPAIR_N64","1");
    PrefillK6SharedDecodeDiscriminator discriminator;
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size()>=1040,"K6 shared-decode prompt extent");
    auto context=target.create_context(true);
    context->prefill(std::span<const std::int64_t>(ids.data(),16));
    context->set_target_projection_observer_for_test(
        PrefillK6SharedDecodeDiscriminator::callback,&discriminator,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(std::span<const std::int64_t>(
        ids.data()+16,1024));
    context->finish_exact_prefill();
    cuda_check(cudaDeviceSynchronize(),"K6 shared-decode discriminator complete");
    context->set_target_projection_observer_for_test(nullptr,nullptr);
    require(discriminator.complete,"K6 shared-decode discriminator missed gate owner");
}
