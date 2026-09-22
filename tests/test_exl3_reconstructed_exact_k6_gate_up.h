#pragma once

void run_reconstructed_exact_k6_gate_up(
    Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    const std::filesystem::path directory=env("NINFER_RECON_K6_GATE_UP_OUT");
    require(!directory.empty()&&!std::filesystem::exists(directory),
        "K6 gate/up reconstruction output directory must be new");
    require(target.max_context()>=1040&&source.size()>=1040,
        "K6 gate/up reconstruction source/context extent");
    for(const char* key:{"NINFER_EXL3_WIDE_PREFILL",
                        "NINFER_EXL3_PREFILL_STAGED_REDUCTION",
                        "NINFER_EXL3_PREFILL_WIDE1024",
                        "NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
                        "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A"})
        require(env(key)=="1",std::string("K6 gate/up reconstruction direct flag ")+key);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");

    struct Capture {
        std::array<T69ProjectionCaptureItem,2> items;
        void observe(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
            if(x.rows!=1024||x.metadata.K!=6||x.metadata.in_features!=5120||
               x.metadata.out_features!=17408||x.metadata.mcg||!x.metadata.mul1||
               x.metadata.has_bias||!x.operation)return;
            const auto operation=std::string_view(x.operation);
            const std::size_t index=operation=="gate"?0:operation=="up"?1:items.size();
            if(index==items.size())return;
            auto& item=items[index];
            if(item.input)return;
            const std::size_t bytes=std::size_t(x.rows)*x.metadata.in_features*2;
            item.input=std::make_unique<DeviceBuffer>(bytes);
            cuda_check(cudaMemcpyAsync(item.input->get(),x.input,bytes,
                cudaMemcpyDeviceToDevice,x.stream),"capture K6 gate input");
            cuda_check(cudaStreamSynchronize(x.stream),"retain K6 gate input");
            item.layer=x.layer;item.operation=x.operation;
            item.weights=x.weights;item.metadata=x.metadata;
        }
    } capture;
    auto context=target.create_context(true);
    const auto model_identity=context->model_identity();
    context->prefill(std::span<const std::int64_t>(source.data(),16));
    context->set_target_projection_observer_for_test(
        [](const auto& value,void* user){static_cast<Capture*>(user)->observe(value);},
        &capture,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(
        std::span<const std::int64_t>(source.data()+16,1024));
    cuda_check(cudaDeviceSynchronize(),"K6 gate/up reconstruction capture complete");
    context->set_target_projection_observer_for_test(nullptr);
    context.reset();
    require(capture.items[0].input&&capture.items[1].input&&
        capture.items[0].weights.trellis!=capture.items[1].weights.trellis,
        "K6 gate/up reconstruction requires real gate and up sites");

    constexpr int ni=5120,no=17408,max_rows=1024;
    constexpr std::size_t decoded_bytes=std::size_t(ni)*no*2;
    constexpr std::size_t output_elements=std::size_t(max_rows)*no;
    auto backing=std::make_shared<DeviceBuffer>(decoded_bytes);
    ninfer::exl3::Exl3ReconstructionStream stream;
    stream.bind_model(model_identity,decoded_bytes,false,backing->get(),true);
    Exl3CudaLinearWorkspace workspace(ni,no,max_rows,false,false,false,false,false,true);
    DeviceBuffer direct_output(output_elements*2),candidate_output(output_elements*2);
    ninfer::exl3::Exl3ReconstructedExactStats stats;
    auto invoke=[&](const T69ProjectionCaptureItem& site,bool candidate,int rows) {
        if(candidate) {
            ninfer::exl3::Exl3ReconstructedExactView view;
            view.data=static_cast<std::uint16_t*>(backing->get());
            view.bytes=decoded_bytes;view.stats=&stats;view.allow_k6_gate_up=true;
            view.ordered_stream=&stream;view.backing_owner=backing;
            view.model_owner=model_identity;workspace.set_reconstructed_exact(view);
            require(std::string_view(workspace.dispatch_name(site.metadata,rows,
                Admission::target_wide_prefill))==
                "target_wide_prefill_reconstructed_exact_k6_gate_up",
                "K6 gate/up exact reconstruction dispatch missing");
        } else workspace.set_reconstructed_exact({});
        workspace.forward(site.weights,site.metadata,
            static_cast<const std::uint16_t*>(site.input->get()),
            static_cast<std::uint16_t*>(candidate?candidate_output.get():direct_output.get()),
            rows,nullptr,Admission::target_wide_prefill);
    };

    std::filesystem::create_directories(directory);
    std::ofstream summary(directory/"summary.csv");
    require(summary.good(),"K6 gate/up reconstruction summary creation");
    summary<<"site,layer,rows,elements,mismatches,first_mismatch,direct_oracle_max_abs,direct_oracle_rel_l2,candidate_oracle_max_abs,candidate_oracle_rel_l2,pass\n";
    std::uint64_t expected_calls=0,expected_rows=0;
    for(std::size_t site_index=0;site_index<capture.items.size();++site_index) {
      const auto& site=capture.items[site_index];
      for(int rows:{256,512,1024}) {
        invoke(site,false,rows);invoke(site,true,rows);
        cuda_check(cudaDeviceSynchronize(),"K6 gate/up reconstruction comparison");
        const std::size_t elements=std::size_t(rows)*no;
        std::vector<std::uint16_t> direct(elements),candidate(elements);
        cuda_check(cudaMemcpy(direct.data(),direct_output.get(),elements*2,
            cudaMemcpyDeviceToHost),"K6 gate/up direct readback");
        cuda_check(cudaMemcpy(candidate.data(),candidate_output.get(),elements*2,
            cudaMemcpyDeviceToHost),"K6 gate/up candidate readback");
        std::size_t mismatches=0,first=elements;
        for(std::size_t i=0;i<elements;++i)if(direct[i]!=candidate[i]) {
            ++mismatches;first=std::min(first,i);
        }
        T69OracleMetrics oracle{};
        if(rows==1024)oracle=t69_check_fp64_oracle(site,direct,candidate,rows);
        const bool pass=mismatches==0&&
            (rows!=1024||(oracle.candidate.relative_l2<=0.003&&
                oracle.candidate.max_abs<=0.10));
        summary<<site_index<<','<<site.layer<<','<<rows<<','<<elements<<','
            <<mismatches<<','<<first<<','<<oracle.direct.max_abs<<','
            <<oracle.direct.relative_l2<<','<<oracle.candidate.max_abs<<','
            <<oracle.candidate.relative_l2<<','<<pass<<'\n';
        summary.flush();require(summary.good(),"K6 gate/up reconstruction evidence write");
        require(pass,"K6 gate/up exact reconstruction correctness gate failed");
        ++expected_calls;expected_rows+=rows;
      }
    }
    workspace.set_reconstructed_exact({});
    require(stats.calls==expected_calls&&stats.rows==expected_rows&&
        stats.k6_gate_up_calls==expected_calls&&stats.k6_gate_up_rows==expected_rows&&
        stats.k6_down_calls==0&&stats.k6_down_rows==0,
        "K6 gate/up reconstruction accounting mismatch");
    std::cout<<"RECONSTRUCTED_EXACT_K6_GATE_UP PASS sites=2 calls="
        <<expected_calls<<" rows="<<expected_rows<<" decoded_bytes="
        <<decoded_bytes<<'\n';
}
