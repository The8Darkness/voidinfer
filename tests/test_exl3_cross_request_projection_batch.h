#pragma once

struct CrossRequestProjectionCapture {
    int wanted_layer=-1;
    std::string wanted_operation;
    int layer=-1;
    std::string operation;
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    int rows=0;
    std::unique_ptr<DeviceBuffer> input;

    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& observation,
        void* user) {
        static_cast<CrossRequestProjectionCapture*>(user)->observe(observation);
    }

    void observe(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if(input || x.rows!=128 || x.metadata.K!=6 ||
           (x.metadata.in_features==17408&&x.metadata.out_features==5120)) return;
        if(wanted_layer>=0 &&
           (x.layer!=wanted_layer || wanted_operation!=x.operation)) return;
        const std::size_t bytes=static_cast<std::size_t>(x.rows)*
            x.metadata.in_features*sizeof(std::uint16_t);
        input=std::make_unique<DeviceBuffer>(bytes);
        cuda_check(cudaMemcpyAsync(input->get(),x.input,bytes,
            cudaMemcpyDeviceToDevice,x.stream),
            "cross-request projection capture D2D");
        cuda_check(cudaStreamSynchronize(x.stream),
            "cross-request projection capture lifetime");
        layer=x.layer;operation=x.operation;weights=x.weights;
        metadata=x.metadata;rows=x.rows;
    }
};

void run_cross_request_projection_batch_t0(
    Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    require(target.max_context()==1024,
        "cross-request projection batch max context");
    require(source.size()>=640,
        "cross-request projection batch source extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");

    std::array<std::unique_ptr<Exl3TextContext>,2> contexts;
    std::array<CrossRequestProjectionCapture,2> captures;
    for(int request=0;request<2;++request) {
        const std::size_t offset=static_cast<std::size_t>(request)*512;
        contexts[request]=target.create_context(false);
        require(contexts[request]->try_enable_oscar_from_environment(),
            "cross-request projection batch OSCAR");
        contexts[request]->prefill(std::span<const std::int64_t>(
            source.data()+offset,16));
        if(request==1) {
            captures[1].wanted_layer=captures[0].layer;
            captures[1].wanted_operation=captures[0].operation;
        }
        contexts[request]->set_target_projection_observer_for_test(
            CrossRequestProjectionCapture::callback,&captures[request],nullptr,
            ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
        contexts[request]->append_prefill_wide(std::span<const std::int64_t>(
            source.data()+offset+16,128));
        contexts[request]->set_target_projection_observer_for_test(nullptr);
        require(captures[request].input!=nullptr,
            "cross-request projection batch capture missing");
        require(contexts[request]->position()==144,
            "cross-request projection batch context position");
    }

    const auto& first=captures[0];const auto& second=captures[1];
    require(first.layer==second.layer&&first.operation==second.operation&&
        first.rows==128&&second.rows==128,
        "cross-request projection batch site identity");
    require(first.metadata.in_features==second.metadata.in_features&&
        first.metadata.out_features==second.metadata.out_features&&
        first.metadata.K==second.metadata.K&&
        first.metadata.mcg==second.metadata.mcg&&
        first.metadata.mul1==second.metadata.mul1&&
        first.metadata.has_bias==second.metadata.has_bias,
        "cross-request projection batch metadata identity");
    require(first.weights.trellis==second.weights.trellis&&
        first.weights.suh==second.weights.suh&&
        first.weights.svh==second.weights.svh&&
        first.weights.mul1==second.weights.mul1,
        "cross-request projection batch immutable weight identity");

    const int ni=first.metadata.in_features,no=first.metadata.out_features;
    const std::size_t input_elements=128ull*ni;
    const std::size_t output_elements=128ull*no;
    std::vector<std::uint16_t> input_first(input_elements),
        input_second(input_elements),input_after(input_elements);
    cuda_check(cudaMemcpy(input_first.data(),first.input->get(),
        input_elements*2,cudaMemcpyDeviceToHost),
        "cross-request first input witness");
    cuda_check(cudaMemcpy(input_second.data(),second.input->get(),
        input_elements*2,cudaMemcpyDeviceToHost),
        "cross-request second input witness");
    require(input_first!=input_second,
        "cross-request inputs must be independent");

    DeviceBuffer joined_input(input_elements*4);
    cuda_check(cudaMemcpy(static_cast<std::uint16_t*>(joined_input.get()),
        first.input->get(),input_elements*2,cudaMemcpyDeviceToDevice),
        "cross-request join first input");
    cuda_check(cudaMemcpy(static_cast<std::uint16_t*>(joined_input.get())+
        input_elements,second.input->get(),input_elements*2,
        cudaMemcpyDeviceToDevice),"cross-request join second input");

    constexpr std::size_t guard=128;
    std::vector<std::uint16_t> poison(output_elements+2*guard,0x3555),
        ref_first(poison),ref_second(poison);
    std::vector<std::uint16_t> batch_poison(2*output_elements+2*guard,0x3555),
        batch(batch_poison);
    DeviceBuffer out_first(poison.size()*2),out_second(poison.size()*2),
        out_batch(batch_poison.size()*2);
    Exl3CudaLinearWorkspace separate(ni,no,128,false,false,false,false,false,true);
    Exl3CudaLinearWorkspace combined(ni,no,256,false,false,false,false,false,true);
    const auto initialize_outputs=[&] {
        cuda_check(cudaMemcpy(out_first.get(),poison.data(),poison.size()*2,
            cudaMemcpyHostToDevice),"cross-request poison first");
        cuda_check(cudaMemcpy(out_second.get(),poison.data(),poison.size()*2,
            cudaMemcpyHostToDevice),"cross-request poison second");
        cuda_check(cudaMemcpy(out_batch.get(),batch_poison.data(),batch_poison.size()*2,
            cudaMemcpyHostToDevice),"cross-request poison batch");
    };
    initialize_outputs();
    separate.forward(first.weights,first.metadata,
        static_cast<const std::uint16_t*>(first.input->get()),
        static_cast<std::uint16_t*>(out_first.get())+guard,128,nullptr,
        Admission::target_wide_prefill);
    separate.forward(second.weights,second.metadata,
        static_cast<const std::uint16_t*>(second.input->get()),
        static_cast<std::uint16_t*>(out_second.get())+guard,128,nullptr,
        Admission::target_wide_prefill);
    combined.forward(first.weights,first.metadata,
        static_cast<const std::uint16_t*>(joined_input.get()),
        static_cast<std::uint16_t*>(out_batch.get())+guard,256,nullptr,
        Admission::target_wide_prefill);
    cuda_check(cudaDeviceSynchronize(),"cross-request projection exact sync");
    cuda_check(cudaMemcpy(ref_first.data(),out_first.get(),ref_first.size()*2,
        cudaMemcpyDeviceToHost),"cross-request first result");
    cuda_check(cudaMemcpy(ref_second.data(),out_second.get(),ref_second.size()*2,
        cudaMemcpyDeviceToHost),"cross-request second result");
    cuda_check(cudaMemcpy(batch.data(),out_batch.get(),batch.size()*2,
        cudaMemcpyDeviceToHost),"cross-request batch result");
    require(std::equal(ref_first.begin()+guard,ref_first.begin()+guard+output_elements,
                       batch.begin()+guard)&&
            std::equal(ref_second.begin()+guard,ref_second.begin()+guard+output_elements,
                       batch.begin()+guard+output_elements),
        "cross-request projection batched output exactness");
    for(std::size_t i=0;i<guard;++i) {
        require(ref_first[i]==0x3555&&ref_second[i]==0x3555&&batch[i]==0x3555,
            "cross-request projection leading canary");
        require(ref_first[guard+output_elements+i]==0x3555&&
                ref_second[guard+output_elements+i]==0x3555&&
                batch[guard+2*output_elements+i]==0x3555,
            "cross-request projection trailing canary");
    }
    cuda_check(cudaMemcpy(input_after.data(),first.input->get(),
        input_elements*2,cudaMemcpyDeviceToHost),
        "cross-request first input after");
    require(input_after==input_first,"cross-request first input mutated");
    cuda_check(cudaMemcpy(input_after.data(),second.input->get(),
        input_elements*2,cudaMemcpyDeviceToHost),
        "cross-request second input after");
    require(input_after==input_second,"cross-request second input mutated");

    std::ofstream timing(env("NINFER_T51_TIMING_CSV"));
    require(timing.good(),"cross-request projection timing output");
    timing<<"rep,order,separate_us,combined_us,gain_pct\n";
    cudaEvent_t begin=nullptr,end=nullptr;
    cuda_check(cudaEventCreate(&begin),"cross-request timing begin create");
    cuda_check(cudaEventCreate(&end),"cross-request timing end create");
    auto measure=[&](bool batched) {
        cuda_check(cudaEventRecord(begin),"cross-request timing begin");
        constexpr int repeats=8;
        for(int repeat=0;repeat<repeats;++repeat) {
            if(batched) combined.forward(first.weights,first.metadata,
                static_cast<const std::uint16_t*>(joined_input.get()),
                static_cast<std::uint16_t*>(out_batch.get())+guard,256,nullptr,
                Admission::target_wide_prefill);
            else {
                separate.forward(first.weights,first.metadata,
                    static_cast<const std::uint16_t*>(first.input->get()),
                    static_cast<std::uint16_t*>(out_first.get())+guard,128,nullptr,
                    Admission::target_wide_prefill);
                separate.forward(second.weights,second.metadata,
                    static_cast<const std::uint16_t*>(second.input->get()),
                    static_cast<std::uint16_t*>(out_second.get())+guard,128,nullptr,
                    Admission::target_wide_prefill);
            }
        }
        cuda_check(cudaEventRecord(end),"cross-request timing end");
        cuda_check(cudaEventSynchronize(end),"cross-request timing sync");
        float milliseconds=0.0f;
        cuda_check(cudaEventElapsedTime(&milliseconds,begin,end),
            "cross-request timing resolve");
        return static_cast<double>(milliseconds)*1000.0/repeats;
    };
    measure(false);measure(true);
    for(int rep=0;rep<8;++rep) {
        double separate_us=0.0,combined_us=0.0;
        if((rep&1)==0){separate_us=measure(false);combined_us=measure(true);}
        else{combined_us=measure(true);separate_us=measure(false);}
        timing<<rep<<','<<((rep&1)==0?"AB":"BA")<<','<<separate_us<<','
            <<combined_us<<','<<(separate_us-combined_us)*100.0/separate_us<<'\n';
    }
    cuda_check(cudaEventDestroy(begin),"cross-request timing begin destroy");
    cuda_check(cudaEventDestroy(end),"cross-request timing end destroy");
    timing.flush();require(timing.good(),"cross-request projection timing flush");
    std::cout<<"CROSS_REQUEST_PROJECTION_BATCH PASS contexts=2 rows_each=128"
        <<" combined_rows=256 layer="<<first.layer<<" operation="<<first.operation
        <<" K="<<first.metadata.K<<" input_independent=1 weights_shared=1"
        <<" output_exact=1 canaries=1 input_immutable=1 measured_pairs=8"
        <<std::endl;
}
