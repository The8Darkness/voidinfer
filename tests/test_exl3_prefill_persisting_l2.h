#pragma once

struct PrefillPersistingL2Operator {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    bool checked=false;
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,
                         void* user) {
        static_cast<PrefillPersistingL2Operator*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if(checked || x.rows!=1024 || x.metadata.K!=6 ||
           x.metadata.in_features==17408 || x.metadata.out_features>17408)return;
        checked=true;
        constexpr std::size_t guard=128;
        const int ni=x.metadata.in_features,no=x.metadata.out_features;
        const std::size_t active=static_cast<std::size_t>(x.rows)*no;
        std::vector<std::uint16_t> poison(active+2*guard,0x3555),
            reference(poison),candidate(poison);
        DeviceBuffer reference_device(poison.size()*2),
            candidate_device(poison.size()*2);
        cuda_check(cudaMemcpy(reference_device.get(),poison.data(),poison.size()*2,
            cudaMemcpyHostToDevice),"persisting L2 reference poison");
        cuda_check(cudaMemcpy(candidate_device.get(),poison.data(),poison.size()*2,
            cudaMemcpyHostToDevice),"persisting L2 candidate poison");
        const auto saved=env("NINFER_EXL3_PREFILL_PERSISTING_L2");
        _putenv_s("NINFER_EXL3_PREFILL_PERSISTING_L2","0");
        Exl3CudaLinearWorkspace control(ni,no,1024,false,false,false,false,
            false,true);
        _putenv_s("NINFER_EXL3_PREFILL_PERSISTING_L2","1");
        Exl3CudaLinearWorkspace changed(ni,no,1024,false,false,false,false,
            false,true);
        const auto admission=Admission::target_wide_prefill;
        require(!control.target_prefill_persisting_l2_candidate(
                    x.metadata,1024,admission),"persisting L2 OFF latch");
        require(changed.target_prefill_persisting_l2_candidate(
                    x.metadata,1024,admission),"persisting L2 exact admission");
        require(!changed.target_prefill_persisting_l2_candidate(
                    x.metadata,1023,admission) &&
                !changed.target_prefill_persisting_l2_candidate(
                    x.metadata,1024,Admission::ordinary),
                "persisting L2 extent/admission refusal");
        control.forward(x.weights,x.metadata,x.input,
            static_cast<std::uint16_t*>(reference_device.get())+guard,1024,
            x.stream,admission);
        changed.forward(x.weights,x.metadata,x.input,
            static_cast<std::uint16_t*>(candidate_device.get())+guard,1024,
            x.stream,admission);
        cuda_check(cudaStreamSynchronize(x.stream),"persisting L2 leaf completion");
        cuda_check(cudaMemcpy(reference.data(),reference_device.get(),
            reference.size()*2,cudaMemcpyDeviceToHost),"persisting L2 reference copy");
        cuda_check(cudaMemcpy(candidate.data(),candidate_device.get(),
            candidate.size()*2,cudaMemcpyDeviceToHost),"persisting L2 candidate copy");
        require(reference==candidate,
            "persisting L2 exact output/canary equality");
        const auto snapshot=changed.prefill_persisting_l2_snapshot();
        require(snapshot.eligible==1 &&
                snapshot.applied+snapshot.unsupported==1,
                "persisting L2 strict dispatch accounting");
        if(snapshot.applied)
            require(snapshot.window_bytes>0 && snapshot.set_aside_bytes>0,
                "persisting L2 applied extent");
        _putenv_s("NINFER_EXL3_PREFILL_PERSISTING_L2",saved.c_str());
    }
};

void run_prefill_persisting_l2(Exl3TextModel& target,
    const std::vector<std::int64_t>& fixture) {
    require(fixture.size()>=1040,"persisting L2 fixture extent");
    const std::array<const char*,10> required{{
        "NINFER_EXL3_WIDE_PREFILL","NINFER_EXL3_PREFILL_STAGED_REDUCTION",
        "NINFER_EXL3_PREFILL_WIDE64","NINFER_EXL3_PREFILL_WIDE128",
        "NINFER_EXL3_PREFILL_WIDE256","NINFER_EXL3_PREFILL_WIDE512",
        "NINFER_EXL3_PREFILL_WIDE1024","NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
        "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A","NINFER_EXL3_PREFILL_DIRECT_TILES64"}};
    std::array<std::string,10> saved{};
    for(std::size_t i=0;i<required.size();++i) {
        saved[i]=env(required[i]);_putenv_s(required[i],"1");
    }
    const auto saved_l2=env("NINFER_EXL3_PREFILL_PERSISTING_L2");
    const auto saved_leaf_graph=env("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS");
    const auto saved_chain_graph=env("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS");
    const auto saved_rowpair=env("NINFER_EXL3_PREFILL_ROWPAIR_K6");
    _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS","0");
    _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS","0");
    _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6","0");
    _putenv_s("NINFER_EXL3_PREFILL_PERSISTING_L2","0");
    auto control=target.create_context(true);
    _putenv_s("NINFER_EXL3_PREFILL_PERSISTING_L2","1");
    auto candidate=target.create_context(true);
    control->prefill(std::span<const std::int64_t>(fixture.data(),16));
    control->append_exact_prefill_wide(
        std::span<const std::int64_t>(fixture.data()+16,1024));
    const auto control_state=control->export_exact_host_state();
    PrefillPersistingL2Operator operator_check;
    candidate->set_target_projection_observer_for_test(
        PrefillPersistingL2Operator::callback,&operator_check,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    candidate->prefill(std::span<const std::int64_t>(fixture.data(),16));
    candidate->append_exact_prefill_wide(
        std::span<const std::int64_t>(fixture.data()+16,1024));
    cuda_check(cudaDeviceSynchronize(),"persisting L2 context completion");
    candidate->set_target_projection_observer_for_test(nullptr,nullptr);
    const auto candidate_state=candidate->export_exact_host_state();
    require(operator_check.checked && control_state->same_payload(*candidate_state),
        "persisting L2 full prefill state equality");
    _putenv_s("NINFER_EXL3_PREFILL_PERSISTING_L2",saved_l2.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_GRAPHS",saved_leaf_graph.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS",saved_chain_graph.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6",saved_rowpair.c_str());
    for(std::size_t i=0;i<required.size();++i)
        _putenv_s(required[i],saved[i].c_str());
    std::cout<<"PREFILL_PERSISTING_L2 PASS rows=1024 exact_leaf=1 exact_state=1"
             <<std::endl;
}
