#pragma once

void run_wide_request_qualification(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(target.max_context()==4096 && source.size()>2100,"wide request fixture extent");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    std::array<std::unique_ptr<DeviceBuffer>,5> stage;
    std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {
        stage[tap]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);
        staging[tap]=static_cast<std::uint16_t*>(stage[tap]->get());
    }
    const auto prompt=std::span<const std::int64_t>(source.data(),2061);
    for(int repeat=0;repeat<3;++repeat) {
        std::shared_ptr<const Request> result[2];double times[2]{};
        for(int order=0;order<2;++order) {
            const int arm=(repeat&1)?1-order:order;
            const auto start=std::chrono::steady_clock::now();
            int last_progress=0;
            result[arm]=Request::initialize(*exact,prompt,arm?1024:8,[&](int position) {
                require(position>last_progress && position<=2061,"wide request progress extent");last_progress=position;
            });
            const auto seed=sample_target(*exact);(void)seed;
            cuda_check(cudaDeviceSynchronize(),"wide request prompt ready");
            times[arm]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            require(last_progress==2061,"wide request incomplete prompt progress");
        }
        require(result[0]->state()->same_payload(*result[1]->state()) && result[0]->same_taps(*result[1]),
            "wide request full state or ring window mismatch");
        std::array<std::uint64_t,5> digests[2];std::vector<std::int64_t> proposals[2];
        for(int arm=0;arm<2;++arm) {
            result[arm]->restore_draft(draft,staging);
            digests[arm]=draft.ring_digest();
            exact->restore_exact_host_state(*result[arm]->state());
            std::vector<std::int64_t> block(8,kMaskToken);block[0]=sample_target(*exact);
            proposals[arm]=draft.propose_cached(block,2061,exact->target_embedding(),
                exact->target_lm_head_weights(),exact->target_lm_head_metadata(),kMaskToken);
        }
        require(digests[0]==digests[1] && proposals[0]==proposals[1],"wide target changed draft conditioning partitions");
        std::cout << "WIDE_REQUEST_PAIR repeat=" << repeat << " prompt_tokens=2061 block8_ready_ms=" << times[0]
            << " wide_ready_ms=" << times[1] << " wide_prefill_tps=" << 2061*1000/times[1] << '\n';
    }
    std::cout << "WIDE_REQUEST PASS pairs=3 prompt=2061 ring_digests=exact proposals=exact\n";
}
