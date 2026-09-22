#pragma once
#include "test_exl3_target_z_k6_oracle.h"

struct TargetZK6Qualification {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    static constexpr auto admit = Admission::target_continuation_z;
    static constexpr int output = 6144;
    static constexpr std::size_t guard = 128, extent = 8u * output;
    std::filesystem::path directory;
    std::ofstream csv;
    std::vector<int> layers;
    explicit TargetZK6Qualification(const std::filesystem::path& path):directory(path) {
        require(!path.empty() && !std::filesystem::exists(path),"z qualification output must be new");
        std::filesystem::create_directories(path);
        csv.open(path/"operators.csv");csv << "layer,rows,elements,exact,baseline_us,candidate_us\n";
    }
    template<class T> void save(const std::string& name,const std::vector<T>& data) {
        std::ofstream f(directory/name,std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
        require(f.good(),"z raw evidence write");
    }
    static void callback(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        static_cast<TargetZK6Qualification*>(user)->check(x);
    }
    void check(const ninfer::exl3::Exl3TargetProjectionObservation& x) {
        require(x.rows==8 && x.metadata.K==6 && x.metadata.in_features==kHidden && x.metadata.out_features==output && std::string(x.operation)=="z","z capture extent");
        require(std::find(layers.begin(),layers.end(),x.layer)==layers.end(),"duplicate z capture");
        layers.push_back(x.layer);
        cuda_check(cudaStreamSynchronize(x.stream),"z input ready");
        std::vector<std::uint16_t> input(8u*kHidden);
        cuda_check(cudaMemcpy(input.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"z input copy");
        _putenv_s("NINFER_EXL3_TARGET_Z_K6_SMALL_M","0");
        Exl3CudaLinearWorkspace baseline(kHidden,output,8,false,false,false,false,true);
        _putenv_s("NINFER_EXL3_TARGET_Z_K6_SMALL_M","1");
        Exl3CudaLinearWorkspace candidate(kHidden,output,8,false,false,false,false,true);
        Exl3CudaLinearWorkspace unowned(kHidden,output,8);
        Exl3CudaLinearWorkspace short_owner(kHidden,output,4,false,false,false,false,true);
        _putenv_s("NINFER_EXL3_TARGET_Z_K6_SMALL_M","0");
        require(candidate.target_z_k6_small_m_candidate(x.metadata,8,admit) && !baseline.target_z_k6_small_m_candidate(x.metadata,8,admit),"z latched enable");
        require(!unowned.target_z_k6_small_m_candidate(x.metadata,8,admit) && !candidate.target_z_k6_small_m_candidate(x.metadata,8,Admission::ordinary) && !short_owner.target_z_k6_small_m_candidate(x.metadata,8,admit),"z ownership/capacity");
        for(int m=0;m<=17;++m)require(candidate.target_z_k6_small_m_candidate(x.metadata,m,admit)==(m>=2 && m<=8),"z width admission");
        for(int bad=0;bad<6;++bad) {
            auto md=x.metadata;
            if(bad==0)md.K=7;if(bad==1)md.mcg=true;if(bad==2)md.mul1=false;
            if(bad==3)md.has_bias=true;if(bad==4)md.in_features=4096;if(bad==5)md.out_features=5120;
            require(!candidate.target_z_k6_small_m_candidate(md,8,admit),"z metadata admission");
        }
        if(layers.size()==1)for(const char* disabled:{"","0","2","true","01"}) {
            _putenv_s("NINFER_EXL3_TARGET_Z_K6_SMALL_M",disabled);
            Exl3CudaLinearWorkspace off(kHidden,output,8,false,false,false,false,true);
            require(!off.target_z_k6_small_m_candidate(x.metadata,8,admit),"z exact flag1 required");
        }
        _putenv_s("NINFER_EXL3_TARGET_Z_K6_SMALL_M","0");
        require(std::string(candidate.dispatch_name(x.metadata,8,admit))=="target_z_k6_small_m_mma_split" && std::string(candidate.dispatch_name(x.metadata,1,admit))=="generic_mma_split","z dispatch");
        DeviceBuffer a((extent+2*guard)*2),b((extent+2*guard)*2);
        auto* pa=static_cast<std::uint16_t*>(a.get())+guard;
        auto* pb=static_cast<std::uint16_t*>(b.get())+guard;
        std::vector<std::uint16_t> initial(extent+2*guard,0x3555),ref(initial),got(initial),repeat(initial);
        const auto serial=[&](int n){for(int row=0;row<n;++row)baseline.forward(x.weights,x.metadata,x.input+row*kHidden,pa+row*output,1,x.stream);};
        for(int n=1;n<=8;++n) {
            cuda_check(cudaMemcpy(a.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"z reference guards");
            cuda_check(cudaMemcpy(b.get(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice),"z candidate guards");
            serial(n);candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"z exact ready");
            cuda_check(cudaMemcpy(ref.data(),a.get(),ref.size()*2,cudaMemcpyDeviceToHost),"z reference copy");
            cuda_check(cudaMemcpy(got.data(),b.get(),got.size()*2,cudaMemcpyDeviceToHost),"z candidate copy");
            const auto tag=std::to_string(x.layer)+"-m"+std::to_string(n);
            save(tag+"-reference.bin",ref);save(tag+"-candidate.bin",got);
            require(ref==got,"z exact output mismatch "+tag);
            for(std::size_t i=0;i<got.size();++i) {
                if(i<guard || i>=guard+n*output)require(got[i]==0x3555,"z output tail/canary");
                else require((got[i]&0x7c00)!=0x7c00,"z nonfinite output");
            }
            candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);
            cuda_check(cudaStreamSynchronize(x.stream),"z repeat ready");
            cuda_check(cudaMemcpy(repeat.data(),b.get(),repeat.size()*2,cudaMemcpyDeviceToHost),"z repeat copy");
            require(repeat==got,"z repeat mismatch");
            cudaEvent_t begin{},end{};cuda_check(cudaEventCreate(&begin),"z timing begin");cuda_check(cudaEventCreate(&end),"z timing end");
            const auto time=[&](bool enabled) {
                cuda_check(cudaEventRecord(begin,x.stream),"z record begin");
                for(int rep=0;rep<3;++rep)if(enabled)candidate.forward(x.weights,x.metadata,x.input,pb,n,x.stream,admit);else serial(n);
                cuda_check(cudaEventRecord(end,x.stream),"z record end");cuda_check(cudaEventSynchronize(end),"z timing ready");
                float ms=0;cuda_check(cudaEventElapsedTime(&ms,begin,end),"z timing elapsed");return ms*1000.0/3;
            };
            double before=time(false),after=time(true);cudaEventDestroy(begin);cudaEventDestroy(end);
            csv << x.layer << ',' << n << ',' << n*output << ",1," << before << ',' << after << '\n';csv.flush();require(csv.good(),"z csv write");
        }
        save(std::to_string(x.layer)+"-input.bin",input);
        std::vector<std::uint16_t> after(input.size());
        cuda_check(cudaMemcpy(after.data(),x.input,input.size()*2,cudaMemcpyDeviceToHost),"z immutable input");require(after==input,"z input mutation");
        if(layers.size()==1) {
            const auto oracle=target_z_k6_check_oracle_groups(x.weights,x.metadata,input,got,guard,{0,23,47});
            std::ofstream f(directory/"oracle.txt");f << "values=" << oracle.values << " max_ratio=" << oracle.max_ratio << " within_bound=" << oracle.within_bound << '\n';
            require(f.good() && oracle.within_bound && oracle.values==3072,"z FP64 oracle gate");
        }
    }
};

void run_target_z_k6_qualification(Exl3TextModel& target) {
    require(env("NINFER_EXL3_TARGET_Z_K6_SMALL_M")=="0","z capture requires flag0");
    auto ids=load_ids(env("NINFER_E5A4_PROMPT_FILE"));require(ids.size()>=24,"z teacher extent");
    auto ctx=target.create_context(true);require(ctx->try_enable_oscar_from_environment(),"z OSCAR required");
    ctx->prefill(std::span<const std::int64_t>(ids.data(),16));
    TargetZK6Qualification check(env("NINFER_E5A4_OUT"));
    ctx->set_target_projection_observer_for_test(TargetZK6Qualification::callback,&check,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::z_k6);
    ctx->append_prefill(std::span<const std::int64_t>(ids.data()+16,8));
    cuda_check(cudaDeviceSynchronize(),"z sweep ready");ctx->set_target_projection_observer_for_test(nullptr,nullptr);
    require(check.layers.size()==47 && ctx->position()==24,"z full layer coverage");
    std::ofstream f(check.directory/"result.txt");f << "PASS operators=47 widths=1..8 cases=376 oracle_values=3072\n";require(f.good(),"z result write");
    std::cout << "TARGET_Z_K6 PASS operators=47 cases=376\n";
}
