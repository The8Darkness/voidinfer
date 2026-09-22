#pragma once

// Operator differential for the target-owned K8 K/V large-M projection.
// This header is intentionally not wired into the aggregate test binary here;
// the measured-checkout owner can include it and add the prepared mode without
// changing the candidate implementation.
struct PrefillK8KvAsyncAQualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit = Admission::target_wide_prefill;
    static constexpr int max_rows = 1024;
    static constexpr std::size_t guard = 128;

    struct EnvironmentRestore {
        std::array<std::pair<const char*,std::string>,9> values{{
            {"NINFER_EXL3_PREFILL_K8_KV_ASYNC_A",env("NINFER_EXL3_PREFILL_K8_KV_ASYNC_A")},
            {"NINFER_EXL3_WIDE_PREFILL",env("NINFER_EXL3_WIDE_PREFILL")},
            {"NINFER_EXL3_PREFILL_STAGED_REDUCTION",env("NINFER_EXL3_PREFILL_STAGED_REDUCTION")},
            {"NINFER_EXL3_PREFILL_DIRECT_PARTIALS",env("NINFER_EXL3_PREFILL_DIRECT_PARTIALS")},
            {"NINFER_EXL3_PREFILL_DIRECT_ASYNC_A",env("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A")},
            {"NINFER_EXL3_PREFILL_WIDE64",env("NINFER_EXL3_PREFILL_WIDE64")},
            {"NINFER_EXL3_PREFILL_WIDE128",env("NINFER_EXL3_PREFILL_WIDE128")},
            {"NINFER_EXL3_PREFILL_WIDE256",env("NINFER_EXL3_PREFILL_WIDE256")},
            {"NINFER_EXL3_PREFILL_WIDE512",env("NINFER_EXL3_PREFILL_WIDE512")}}};
        std::string wide1024 = env("NINFER_EXL3_PREFILL_WIDE1024");
        ~EnvironmentRestore() {
            for (const auto& value : values) _putenv_s(value.first,value.second.c_str());
            _putenv_s("NINFER_EXL3_PREFILL_WIDE1024",wide1024.c_str());
        }
    } restore;

    std::set<std::string> operations;
    int cases = 0;
    std::uint64_t values = 0;

    PrefillK8KvAsyncAQualification() {
        for (const char* name : {"NINFER_EXL3_WIDE_PREFILL",
                                 "NINFER_EXL3_PREFILL_STAGED_REDUCTION",
                                 "NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
                                 "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A",
                                 "NINFER_EXL3_PREFILL_WIDE64",
                                 "NINFER_EXL3_PREFILL_WIDE128",
                                 "NINFER_EXL3_PREFILL_WIDE256",
                                 "NINFER_EXL3_PREFILL_WIDE512",
                                 "NINFER_EXL3_PREFILL_WIDE1024"})
            _putenv_s(name,"1");
        _putenv_s("NINFER_EXL3_PREFILL_K8_KV_ASYNC_A","0");
    }

    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& x, void* user) {
        static_cast<PrefillK8KvAsyncAQualification*>(user)->check(x);
    }

    static std::unique_ptr<Exl3CudaLinearWorkspace> workspace(bool candidate) {
        _putenv_s("NINFER_EXL3_PREFILL_K8_KV_ASYNC_A",candidate ? "1" : "0");
        return std::make_unique<Exl3CudaLinearWorkspace>(
            5120,1024,max_rows,false,false,false,false,false,true,false,
            ninfer::exl3::Exl3CudaAccumulationView{},
            ninfer::exl3::Exl3CudaTransformView{},false,true,false);
    }

    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if (x.rows < max_rows || x.metadata.K != 8 ||
            x.metadata.in_features != 5120 || x.metadata.out_features != 1024 ||
            (std::string(x.operation) != "k" && std::string(x.operation) != "v") ||
            !operations.emplace(x.operation).second) return;
        cuda_check(cudaStreamSynchronize(x.stream),"K8 K/V async-A input ready");

        auto control = workspace(false);
        auto candidate = workspace(true);
        _putenv_s("NINFER_EXL3_PREFILL_K8_KV_ASYNC_A","0");
        require(!candidate->target_k8_kv_prefill_async_a_candidate(x.metadata,16,admit) &&
                candidate->target_k8_kv_prefill_async_a_candidate(x.metadata,17,admit) &&
                candidate->target_k8_kv_prefill_async_a_candidate(x.metadata,max_rows,admit) &&
                !candidate->target_k8_kv_prefill_async_a_candidate(x.metadata,max_rows,Admission::ordinary),
                "K8 K/V async-A admission boundary");

        const std::size_t input_extent=static_cast<std::size_t>(max_rows)*5120;
        std::vector<std::uint16_t> input_before(input_extent),input_after(input_extent);
        cuda_check(cudaMemcpy(input_before.data(),x.input,input_extent*sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost),"K8 K/V async-A input snapshot");
        std::uint64_t expected_calls=0,expected_rows=0;
        for (int rows : {17,1009,1024}) {
            const std::size_t active=static_cast<std::size_t>(rows)*1024;
            const std::vector<std::uint16_t> initial(active+2*guard,0x3555);
            std::vector<std::uint16_t> a(initial.size()),b(initial.size()),repeat(initial.size());
            DeviceBuffer old_output(initial.size()*sizeof(std::uint16_t));
            DeviceBuffer new_output(initial.size()*sizeof(std::uint16_t));
            auto* old_base=static_cast<std::uint16_t*>(old_output.get());
            auto* new_base=static_cast<std::uint16_t*>(new_output.get());
            cuda_check(cudaMemcpy(old_base,initial.data(),initial.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),"K8 K/V async-A control guards");
            cuda_check(cudaMemcpy(new_base,initial.data(),initial.size()*sizeof(std::uint16_t),cudaMemcpyHostToDevice),"K8 K/V async-A candidate guards");
            require(std::string(control->dispatch_name(x.metadata,rows,admit))=="target_wide_prefill_staged" &&
                    std::string(candidate->dispatch_name(x.metadata,rows,admit))=="target_wide_prefill_k8_kv_async_a",
                    "K8 K/V async-A route selection");
            control->forward(x.weights,x.metadata,x.input,old_base+guard,rows,x.stream,admit);
            candidate->forward(x.weights,x.metadata,x.input,new_base+guard,rows,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"K8 K/V async-A compare ready");
            cuda_check(cudaMemcpy(a.data(),old_base,a.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),"K8 K/V async-A control output");
            cuda_check(cudaMemcpy(b.data(),new_base,b.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),"K8 K/V async-A candidate output");
            candidate->forward(x.weights,x.metadata,x.input,new_base+guard,rows,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"K8 K/V async-A repeat ready");
            cuda_check(cudaMemcpy(repeat.data(),new_base,repeat.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),"K8 K/V async-A repeat output");
            cuda_check(cudaMemcpy(input_after.data(),x.input,input_extent*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),"K8 K/V async-A input after");
            expected_calls+=2; expected_rows+=2*rows;
            require(a==b && b==repeat && input_before==input_after &&
                    std::all_of(b.begin(),b.begin()+guard,[](auto v){return v==0x3555;}) &&
                    std::all_of(b.end()-guard,b.end(),[](auto v){return v==0x3555;}) &&
                    control->target_k8_kv_prefill_async_a_calls()==0 &&
                    candidate->target_k8_kv_prefill_async_a_calls()==expected_calls &&
                    candidate->target_k8_kv_prefill_async_a_rows()==expected_rows,
                    "K8 K/V async-A bit-exact differential");
            ++cases; values+=active;
        }
    }
};

void run_prefill_k8_kv_async_a_qualification(Exl3TextModel& target) {
    for (const char* malformed : {"2","true","01"}) {
        _putenv_s("NINFER_EXL3_PREFILL_K8_KV_ASYNC_A",malformed);
        bool rejected=false;
        try {
            Exl3CudaLinearWorkspace invalid(
                5120,1024,PrefillK8KvAsyncAQualification::max_rows,
                false,false,false,false,false,true,false,
                ninfer::exl3::Exl3CudaAccumulationView{},
                ninfer::exl3::Exl3CudaTransformView{},false,true,false);
        }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"K8 K/V async-A accepted malformed opt-in");
    }
    PrefillK8KvAsyncAQualification qualification;
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size()>=16+PrefillK8KvAsyncAQualification::max_rows,
            "K8 K/V async-A prompt extent");
    auto context=target.create_context(true);
    context->prefill(std::span<const std::int64_t>(ids.data(),16));
    context->set_target_projection_observer_for_test(
        PrefillK8KvAsyncAQualification::callback,&qualification,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(std::span<const std::int64_t>(
        ids.data()+16,PrefillK8KvAsyncAQualification::max_rows));
    context->finish_exact_prefill();
    cuda_check(cudaDeviceSynchronize(),"K8 K/V async-A qualification complete");
    context->set_target_projection_observer_for_test(nullptr,nullptr);
    require(qualification.operations==std::set<std::string>{"k","v"} &&
            qualification.cases==6,"K8 K/V async-A real-caller coverage");
    std::cout << "PREFILL_K8_KV_ASYNC_A PASS operations=2 cases="
              << qualification.cases << " values=" << qualification.values
              << " exact=1 repeat=1 guards=1 input_unchanged=1 counters=1\n";
}
