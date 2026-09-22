#pragma once

void run_host_stream_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(target.max_context()==1024 && source.size()>350,"host stream fixture extent");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    auto resident=target.create_context(true);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto streamed=target.create_context(true);
    resident->prepare_continuation(8);streamed->prepare_continuation(8);
    require(streamed->host_kv_stats().enabled && streamed->host_kv_stats().layer_workspace_bytes==1024*4096,
        "bounded layer workspace geometry");
    require(!streamed->capture_decode_graph(),"host streaming admitted graph capture");
    bool wrong_mode=false;
    try {streamed->try_enable_oscar_from_environment();} catch(const std::exception&) {wrong_mode=true;}
    require(wrong_mode,"host streaming admitted approximate authoritative cache");
    int cases=0;
    for(int prefix:{63,321}) {
        const auto reference=Request::initialize(*resident,std::span<const std::int64_t>(source.data(),prefix));
        const auto from_host=Request::initialize(*streamed,std::span<const std::int64_t>(source.data(),prefix));
        require(reference->state()->same_payload(*from_host->state()) && reference->same_taps(*from_host),
            "streamed authoritative initialization mismatch");
        const auto root=reference->state();
        std::vector<std::int64_t> greedy;
        for(int i=0;i<8;++i) {const auto token=sample_target(*resident);greedy.push_back(token);resident->decode(token);}
        for(int width:{2,8}) for(int reject=0;reject<=width;++reject) {
            auto proposal=std::vector<std::int64_t>(greedy.begin(),greedy.begin()+width);
            if(reject<width) proposal[reject]=(proposal[reject]+1)%kVocab;
            const auto before=streamed->host_kv_stats();
            const auto a=ninfer::exl3::verify_exl3_outer_batched_reference(*resident,*root,proposal,true);
            const auto b=ninfer::exl3::verify_exl3_outer_batched_reference(*streamed,*root,proposal,true);
            const auto after=streamed->host_kv_stats();
            require(a.accepted==b.accepted && a.committed_tokens==b.committed_tokens &&
                a.committed_taps==b.committed_taps && a.committed_state->same_payload(*b.committed_state),
                "streamed outer rejection/commit state mismatch");
            const std::uint64_t passes=reject<width?2:1;
            require(after.h2d_bytes-before.h2d_bytes==passes*prefix*65536 &&
                after.d2h_bytes-before.d2h_bytes==b.executed_rows*65536,
                "required attention transfers omitted or miscounted");
            const auto next=sample_target(*resident);
            resident->decode(next);streamed->decode(next);
            require(resident->export_exact_host_state(nullptr,false)->same_payload(*streamed->export_exact_host_state()),
                "streamed rejected suffix contaminates next decode");
            ++cases;
        }
        for(int repeat=0;repeat<3;++repeat) {
            double times[2]{};
            for(int order=0;order<2;++order) {
                const int arm=(repeat&1)?1-order:order;
                const auto start=std::chrono::steady_clock::now();
                const auto result=ninfer::exl3::verify_exl3_outer_batched_reference(
                    arm?*streamed:*resident,*root,greedy);
                cuda_check(cudaDeviceSynchronize(),"host stream complete verifier timing");
                times[arm]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                require(result.accepted==8,"host stream timing greedy path");
            }
            std::cout << "HOST_STREAM_PAIR prefix=" << prefix << " repeat=" << repeat
                << " resident_ms=" << times[0] << " streamed_ms=" << times[1] << " committed=8\n";
        }
    }
    const auto stats=streamed->host_kv_stats();
    std::cout << "HOST_STREAM PASS cases=" << cases << " initializations=2 pairs=6 layer_workspace_bytes="
        << stats.layer_workspace_bytes << " resident_context_bytes=" << resident->persistent_bytes()
        << " streamed_context_bytes=" << streamed->persistent_bytes() << " h2d_bytes=" << stats.h2d_bytes
        << " d2h_bytes=" << stats.d2h_bytes << " calls=" << stats.transfer_calls
        << " identity=vericache_exact_fp16_eager overlap=none\n";
}
