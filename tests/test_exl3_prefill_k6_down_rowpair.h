#pragma once

struct PrefillK6DownRowpairQualification {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit=Admission::target_wide_prefill;
    static constexpr int max_rows=1024;
    static constexpr std::size_t guard=128;
    int cases=0;
    std::uint64_t values=0;

    PrefillK6DownRowpairQualification() {
        for(const char* name:{"NINFER_EXL3_WIDE_PREFILL",
            "NINFER_EXL3_PREFILL_STAGED_REDUCTION",
            "NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
            "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A",
            "NINFER_EXL3_PREFILL_WIDE64","NINFER_EXL3_PREFILL_WIDE128",
            "NINFER_EXL3_PREFILL_WIDE256","NINFER_EXL3_PREFILL_WIDE512",
            "NINFER_EXL3_PREFILL_WIDE1024"})_putenv_s(name,"1");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K6_DOWN","0");
        _putenv_s("NINFER_EXL3_PREFILL_K6_DOWN_ROWPAIR","0");
    }
    static void callback(
        const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        static_cast<PrefillK6DownRowpairQualification*>(user)->check(x);
    }
    static std::unique_ptr<Exl3CudaLinearWorkspace> workspace(bool candidate) {
        _putenv_s("NINFER_EXL3_PREFILL_K6_DOWN_ROWPAIR",candidate?"1":"0");
        return std::make_unique<Exl3CudaLinearWorkspace>(
            17408,5120,max_rows,false,false,true,false,false,true,false);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        if(cases||x.rows<max_rows||x.metadata.K!=6||
           x.metadata.in_features!=17408||x.metadata.out_features!=5120||
           std::string(x.operation)!="down")return;
        cuda_check(cudaStreamSynchronize(x.stream),"K6 Down rowpair input ready");
        auto control=workspace(false),candidate=workspace(true);
        _putenv_s("NINFER_EXL3_PREFILL_K6_DOWN_ROWPAIR","0");
        require(!candidate->target_k6_down_rowpair_candidate(x.metadata,31,admit)&&
                candidate->target_k6_down_rowpair_candidate(x.metadata,32,admit)&&
                candidate->target_k6_down_rowpair_candidate(x.metadata,max_rows,admit)&&
                !candidate->target_k6_down_rowpair_candidate(
                    x.metadata,max_rows,Admission::ordinary),
                "K6 Down rowpair admission boundary");
        const std::size_t input_extent=static_cast<std::size_t>(max_rows)*17408;
        std::vector<std::uint16_t> input_before(input_extent),input_after(input_extent);
        cuda_check(cudaMemcpy(input_before.data(),x.input,input_extent*2,
            cudaMemcpyDeviceToHost),"K6 Down rowpair input snapshot");
        std::uint64_t expected_calls=0,expected_rows=0;
        for(int rows:{32,33,1009,1024}) {
            const std::size_t active=static_cast<std::size_t>(rows)*5120;
            const std::vector<std::uint16_t> initial(active+2*guard,0x3555);
            std::vector<std::uint16_t> a(initial.size()),b(initial.size()),repeat(initial.size());
            DeviceBuffer old_output(initial.size()*2),new_output(initial.size()*2);
            auto* old_base=static_cast<std::uint16_t*>(old_output.get());
            auto* new_base=static_cast<std::uint16_t*>(new_output.get());
            cuda_check(cudaMemcpy(old_base,initial.data(),initial.size()*2,
                cudaMemcpyHostToDevice),"K6 Down rowpair control guards");
            cuda_check(cudaMemcpy(new_base,initial.data(),initial.size()*2,
                cudaMemcpyHostToDevice),"K6 Down rowpair candidate guards");
            require(std::string(control->dispatch_name(x.metadata,rows,admit))==
                        "target_wide_prefill_staged"&&
                    std::string(candidate->dispatch_name(x.metadata,rows,admit))==
                        "target_wide_prefill_k6_down_rowpair",
                    "K6 Down rowpair route selection");
            control->forward(x.weights,x.metadata,x.input,old_base+guard,rows,x.stream,admit);
            candidate->forward(x.weights,x.metadata,x.input,new_base+guard,rows,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"K6 Down rowpair compare ready");
            cuda_check(cudaMemcpy(a.data(),old_base,a.size()*2,cudaMemcpyDeviceToHost),
                "K6 Down rowpair control output");
            cuda_check(cudaMemcpy(b.data(),new_base,b.size()*2,cudaMemcpyDeviceToHost),
                "K6 Down rowpair candidate output");
            candidate->forward(x.weights,x.metadata,x.input,new_base+guard,rows,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"K6 Down rowpair repeat ready");
            cuda_check(cudaMemcpy(repeat.data(),new_base,repeat.size()*2,cudaMemcpyDeviceToHost),
                "K6 Down rowpair repeat output");
            cuda_check(cudaMemcpy(input_after.data(),x.input,input_extent*2,
                cudaMemcpyDeviceToHost),"K6 Down rowpair input after");
            expected_calls+=2;expected_rows+=2*rows;
            require(a==b&&b==repeat&&input_before==input_after&&
                    std::all_of(b.begin(),b.begin()+guard,[](auto v){return v==0x3555;})&&
                    std::all_of(b.end()-guard,b.end(),[](auto v){return v==0x3555;})&&
                    control->k6_down_rowpair_calls()==0&&
                    candidate->k6_down_rowpair_calls()==expected_calls&&
                    candidate->k6_down_rowpair_rows()==expected_rows,
                    "K6 Down rowpair bit-exact differential");
            ++cases;values+=active;
        }
    }
};

void run_prefill_k6_down_rowpair_qualification(Exl3TextModel& target) {
    for(const char* malformed:{"2","true","01"}) {
        _putenv_s("NINFER_EXL3_PREFILL_K6_DOWN_ROWPAIR",malformed);
        bool rejected=false;
        try {
            Exl3CudaLinearWorkspace invalid(17408,5120,
                PrefillK6DownRowpairQualification::max_rows,
                false,false,true,false,false,true,false);
        }catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"K6 Down rowpair accepted malformed opt-in");
    }
    PrefillK6DownRowpairQualification qualification;
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    require(ids.size()>=16+PrefillK6DownRowpairQualification::max_rows,
        "K6 Down rowpair prompt extent");
    auto context=target.create_context(true);
    context->prefill(std::span<const std::int64_t>(ids.data(),16));
    context->set_target_projection_observer_for_test(
        PrefillK6DownRowpairQualification::callback,&qualification,nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(std::span<const std::int64_t>(
        ids.data()+16,PrefillK6DownRowpairQualification::max_rows));
    context->finish_exact_prefill();
    cuda_check(cudaDeviceSynchronize(),"K6 Down rowpair qualification complete");
    context->set_target_projection_observer_for_test(nullptr,nullptr);
    require(qualification.cases==4,"K6 Down rowpair real-caller coverage");
    std::cout<<"PREFILL_K6_DOWN_ROWPAIR PASS operations=1 cases="
        <<qualification.cases<<" values="<<qualification.values
        <<" exact=1 repeat=1 guards=1 input_unchanged=1 counters=1\n";
}

void run_prefill_k6_down_rowpair_state(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    constexpr int decode_rows=8;
    require(target.max_context()>=4352&&code.size()>=prefix&&prose.size()>=prefix,
        "K6 Down rowpair state fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "K6 Down rowpair finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    for(const auto& fixture:std::array<
        std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        _putenv_s("NINFER_EXL3_PREFILL_K6_DOWN_ROWPAIR","0");
        auto control=target.create_context(true);control->prepare_continuation(decode_rows);
        _putenv_s("NINFER_EXL3_PREFILL_K6_DOWN_ROWPAIR","1");
        auto candidate=target.create_context(true);candidate->prepare_continuation(decode_rows);
        const std::vector<std::int64_t> input(
            fixture.second->begin(),fixture.second->begin()+prefix);
        const auto run=[&](Exl3TextContext& context) {
            const auto request=Request::initialize(context,input,1024);
            std::vector<std::int64_t> tokens;
            for(int row=0;row<decode_rows;++row) {
                const auto token=greedy(context);tokens.push_back(token);context.decode(token);
            }
            return std::tuple{tokens,context.export_exact_host_state(),
                context.k6_down_rowpair_calls(),context.k6_down_rowpair_rows()};
        };
        const auto baseline=run(*control),changed=run(*candidate);
        require(std::get<0>(baseline)==std::get<0>(changed)&&
                std::get<1>(baseline)->same_payload(*std::get<1>(changed)),
            "K6 Down rowpair token/state");
        require(std::get<2>(baseline)==0&&std::get<3>(baseline)==0&&
                std::get<2>(changed)>0&&std::get<3>(changed)>=prefix,
            "K6 Down rowpair state dispatch counters");
        std::cout<<"PREFILL_K6_DOWN_ROWPAIR_STATE fixture="<<fixture.first
            <<" prefix="<<prefix<<" decode_rows="<<decode_rows
            <<" exact_tokens_state=1 candidate_launches="<<std::get<2>(changed)
            <<" candidate_rows="<<std::get<3>(changed)<<std::endl;
    }
    std::cout<<"PREFILL_K6_DOWN_ROWPAIR_STATE PASS fixtures=2 prefix="<<prefix
        <<" decode_rows="<<decode_rows<<" exact_tokens_state=1"<<std::endl;
}
