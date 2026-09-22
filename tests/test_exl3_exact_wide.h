#pragma once

void run_exact_wide_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(target.max_context()==4096 && source.size()>1400,"exact wide fixture extent");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    auto reference=target.create_context(true);reference->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto baseline=target.create_context(true),wide=target.create_context(true);
    baseline->prepare_continuation(8);wide->prepare_continuation(8);
    const auto ingest=[&](Exl3TextContext& ctx,std::span<const std::int64_t> ids,bool large) {
        std::array<std::vector<std::uint16_t>,5> taps;
        for(auto& plane:taps) plane.reserve(ids.size()*kHidden);
        for(std::size_t first=0;first<ids.size();) {
            const int rows=static_cast<int>(std::min<std::size_t>(large?1024:8,ids.size()-first));
            if(large && rows>8) ctx.append_exact_prefill_wide(ids.subspan(first,rows));
            else if(rows>1) ctx.continue_rows(ids.subspan(first,rows));
            else ctx.decode(ids[first]);
            const auto captured=ctx.exact_tap_rows_host();
            for(int tap=0;tap<5;++tap) taps[tap].insert(taps[tap].end(),captured[tap].begin(),captured[tap].end());
            if(large && rows>8) ctx.finish_exact_prefill();
            else if(rows>1) ctx.finish_exact_continuation();
            first+=rows;
        }
        return taps;
    };
    const auto compare_taps=[&](const auto& expected,const auto& actual,int prefix,int rows) {
        for(int tap=0;tap<5;++tap) if(expected[tap]!=actual[tap]) {
            const auto mismatch=std::mismatch(expected[tap].begin(),expected[tap].end(),actual[tap].begin(),actual[tap].end());
            throw std::runtime_error("exact wide tap mismatch prefix="+std::to_string(prefix)+" rows="+std::to_string(rows)+
                " tap="+std::to_string(tap)+" element="+std::to_string(mismatch.first-expected[tap].begin()));
        }
    };
    int cases=0,pairs=0;
    for(int prefix:{16,321}) {
        const auto root=Request::initialize(*reference,std::span<const std::int64_t>(source.data(),prefix))->state();
        for(int rows:{16,32,128,1024}) {
            const auto ids=std::span<const std::int64_t>(source.data()+prefix,rows);
            reference->restore_exact_host_state(*root);
            const auto oracle_taps=ingest(*reference,ids,false);
            const auto oracle=reference->export_exact_host_state(nullptr,false);
            wide->restore_exact_host_state(*root);
            const auto before=wide->host_kv_stats();
            const auto actual_taps=ingest(*wide,ids,true);
            const auto after=wide->host_kv_stats();
            compare_taps(oracle_taps,actual_taps,prefix,rows);
            require(oracle->same_payload(*wide->export_exact_host_state()),"exact wide full KV/recurrent/conv state mismatch");
            require(after.h2d_bytes-before.h2d_bytes==static_cast<std::uint64_t>(prefix)*65536 &&
                after.d2h_bytes-before.d2h_bytes==static_cast<std::uint64_t>(rows)*65536,"wide full attention transfer coverage");
            const auto next=sample_target(*reference);reference->decode(next);wide->decode(next);
            require(reference->export_exact_host_state(nullptr,false)->same_payload(*wide->export_exact_host_state()),
                "exact wide subsequent decode state mismatch");
            for(int repeat=0;repeat<3;++repeat) {
                double times[2]{};
                for(int order=0;order<2;++order) {
                    const int arm=(repeat&1)?1-order:order;
                    auto& ctx=arm?*wide:*baseline;
                    const auto start=std::chrono::steady_clock::now();
                    ctx.restore_exact_host_state(*root);
                    const auto captured=ingest(ctx,ids,arm!=0);
                    const auto state=ctx.export_exact_host_state();
                    const auto token=sample_target(ctx);
                    cuda_check(cudaDeviceSynchronize(),"exact wide prefix-ready timing");
                    times[arm]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    compare_taps(oracle_taps,captured,prefix,rows);
                    require(state->same_payload(*oracle) && token==next,"exact wide timed payload mismatch");
                }
                std::cout << "EXACT_WIDE_PAIR prefix=" << prefix << " rows=" << rows << " repeat=" << repeat
                    << " block8_ms=" << times[0] << " wide_ms=" << times[1] << " wide_append_tps=" << rows*1000/times[1] << '\n';
                ++pairs;
            }
            ++cases;
        }
    }
    std::cout << "EXACT_WIDE PASS cases=" << cases << " pairs=" << pairs
        << " max_rows=1024 score_rows=16 identity=vericache_exact_fp16_eager\n";
}
