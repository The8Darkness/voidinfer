#pragma once

void run_quality_probe(Exl3TextModel& target,Exl3Dflash2DraftModel& draft) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    const std::filesystem::path fixtures=env("NINFER_QUALITY_DIR"),output=env("NINFER_QUALITY_OUTPUT");
    require(!fixtures.empty() && !output.empty(),"quality fixture/output paths");
    const auto terminal=load_ids((fixtures/"stop.ids").string());
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");_putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    auto inner=target.create_context(true);require(inner->try_enable_oscar_from_environment(),"quality OSCAR");
    inner->prepare_transaction();inner->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");_putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true),reference=target.create_context(true);
    exact->prepare_continuation(8);reference->prepare_continuation(8);
    TapStage stage;std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16ULL*kHidden*2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        staging[tap]=static_cast<std::uint16_t*>(stage.bulk.back()->get());stage.bulk_ptrs.push_back(staging[tap]);
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    int cases=0;
    for(const std::string name:{"arithmetic","structured","code","prose"}) {
        const auto ids=load_ids((fixtures/(name+".ids")).string());
        const auto destination=output/(name+".generated.ids");
        require(!std::filesystem::exists(destination),"preserve quality output evidence");
        const auto arrival=std::chrono::steady_clock::now();
        auto request=Request::initialize(*exact,ids,1024);
        const double prefill_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-arrival).count();
        std::vector<std::int64_t> generated;bool stopped=false;double decode_ms=0,ttft_ms=0;
        while(generated.size()<64 && !stopped) {
            const auto start=std::chrono::steady_clock::now();
            const auto warm=name=="structured" ? Exl3TextContext::make_turboangle_warm_pages(request->state(),
                std::min(64,request->state()->position())) : nullptr;
            request->restore_inner(*inner,draft,staging,warm.get());
            const auto seed=sample_target(*inner);
            std::vector<std::int64_t> block(8,kMaskToken);block[0]=seed;
            const auto proposals=draft.propose_cached(block,inner->position(),inner->target_embedding(),
                inner->target_lm_head_weights(),inner->target_lm_head_metadata(),kMaskToken);
            const auto local=verify_pending_round_transactional(*inner,draft,stage,seed,proposals,inner->position(),
                nullptr,-1,0x7e00u,TransactionRepairMode::retain_prefix);
            std::vector<std::int64_t> tentative{seed};tentative.insert(tentative.end(),local.emitted.begin(),local.emitted.end());
            tentative.resize(std::min<std::size_t>({tentative.size(),8,64-generated.size()}));
            auto [updated,result]=request->verify(*exact,tentative,terminal);
            updated->restore_inner(*inner,draft,staging);
            cuda_check(cudaDeviceSynchronize(),"quality authoritative release");
            request=std::move(updated);stopped=result.stopped;
            generated.insert(generated.end(),result.committed_tokens.begin(),result.committed_tokens.end());
            decode_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(ttft_ms==0) ttft_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-arrival).count();
        }
        auto oracle=Request::initialize(*reference,ids);
        for(std::size_t i=0;i<generated.size();++i) {
            require(sample_target(*reference)==generated[i],"quality changed authoritative greedy output");
            if(ninfer::exl3::exl3_terminal_token(generated[i],terminal)) require(i+1==generated.size(),"quality published post-EOS output");
            oracle=oracle->append_prompt(*reference,std::span<const std::int64_t>(&generated[i],1));
        }
        require(request->state()->same_payload(*oracle->state()) && request->same_taps(*oracle),"quality final authoritative state");
        std::ofstream file(destination);for(auto token:generated) file<<token<<'\n';file.close();require(file.good(),"quality output write");
        const auto visible=std::count_if(generated.begin(),generated.end(),[&](auto token){return !ninfer::exl3::exl3_terminal_token(token,terminal);});
        std::cout << "QUALITY_PROBE name=" << name << " prompt_tokens=" << ids.size() << " committed_tokens=" << generated.size()
            << " visible_tokens=" << visible << " stopped=" << stopped << " prefill_ms=" << prefill_ms
            << " ttft_ms=" << ttft_ms << " decode_ms=" << decode_ms << " visible_tps=" << visible*1000/decode_ms << '\n';
        ++cases;
    }
    std::cout << "QUALITY_MECHANISM PASS cases=" << cases << " semantic_check=external_tokenizer\n";
}
