#pragma once

void run_prefill_chunk_qualification(Exl3TextModel& target, Exl3Dflash2DraftModel& draft) {
    const auto directory=std::filesystem::path(env("NINFER_E5A4_OUT"));
    require(!directory.empty() && !std::filesystem::exists(directory),"chunk qualification output must be new");
    std::filesystem::create_directories(directory);
    std::ofstream out(directory/"boundaries.csv");
    out << "position,rows,taps_exact,full_state_checked,continuation_bytes\n";
    auto source=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    const int total=env_int("NINFER_E5A4_CONTEXTS",512);
    require((total==512 || total==2080) && target.max_context()>=total+16,"chunk qualification extent");
    std::vector<std::int64_t> ids(total+16);
    for(std::size_t i=0;i<ids.size();++i) ids[i]=source[i%source.size()];
    auto reference=target.create_context(true), candidate=target.create_context(true);
    require(reference->try_enable_oscar_from_environment() && candidate->try_enable_oscar_from_environment(),"chunk OSCAR required");
    const auto reject=[&](auto&& operation) {
        bool failed=false;try{operation();}catch(const std::exception&){failed=true;}
        require(failed,"chunk admission did not reject");
    };
    reject([&]{candidate->append_prefill(std::span<const std::int64_t>(ids.data(),8));});
    require(candidate->position()==0 && candidate->last_forward_rows()==0,"empty-prefix rejection mutated metadata");
    reference->prefill(std::span<const std::int64_t>(ids.data(),16));
    candidate->prefill(std::span<const std::int64_t>(ids.data(),16));
    const auto compare_state=[&](Exl3TextContext& a,Exl3TextContext& b) {
        cuda_check(cudaDeviceSynchronize(),"chunk state ready");
        require_target_continue_state_equal(a,b,"chunk exact state",false);
        for(int layer=0;layer<64;++layer) if((layer+1)%4!=0)
            require(a.gdn_physical_conv_host(layer)==b.gdn_physical_conv_host(layer),"chunk physical convolution");
        require(a.oscar_live_state_host_for_test()==b.oscar_live_state_host_for_test(),"chunk live OSCAR");
    };
    compare_state(*candidate,*reference);
    {
        const auto before=prefix_retention_snapshot(*candidate);
        reject([&]{candidate->append_prefill({});});
        reject([&]{candidate->append_prefill(std::span<const std::int64_t>(ids.data(),9));});
        std::int64_t bad=-1;
        reject([&]{candidate->append_prefill(std::span<const std::int64_t>(&bad,1));});bad=kVocab;
        reject([&]{candidate->append_prefill(std::span<const std::int64_t>(&bad,1));});
        require(prefix_retention_snapshot(*candidate)==before,"invalid chunk mutated state");
    }
    candidate->prepare_transaction();candidate->begin_transaction();
    prefix_retention_reject_unchanged(*candidate,[&]{candidate->append_prefill(std::span<const std::int64_t>(ids.data(),8));},"active transaction append");
    candidate->rollback_transaction();
    const auto allocation_before=candidate->persistent_bytes();
    TapStage stage;
    for(int t=0;t<kTapCount;++t) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(16u*kHidden*2));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(kHidden*2));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    std::array<std::vector<std::uint16_t>,5> reference_taps;
    for(int t=0;t<kTapCount;++t) {
        reference_taps[t]=target_continue_tap_bits(*reference,kTapLayers[t],16);
        require(reference_taps[t]==target_continue_tap_bits(*candidate,kTapLayers[t],16),"initial taps");
        reference_taps[t].reserve(static_cast<std::size_t>(total)*kHidden);
    }
    draft.reset();commit_current_rows(*candidate,draft,stage,16,0);
    int next_width=1,boundaries=0,last_state_cut=16;
    for(int position=16;position<total;) {
        const int count=std::min(next_width,total-position);
        if(next_width<8)++next_width;
        std::array<std::vector<std::uint16_t>,5> expected;
        std::vector<std::uint16_t> embeddings;
        for(int row=0;row<count;++row) {
            reference->decode(ids[position+row]);
            for(int t=0;t<kTapCount;++t) {
                auto bits=target_continue_tap_bits(*reference,kTapLayers[t],1);
                expected[t].insert(expected[t].end(),bits.begin(),bits.end());
                reference_taps[t].insert(reference_taps[t].end(),bits.begin(),bits.end());
            }
            auto bits=reference->embedding_bits_host_for_test();
            embeddings.insert(embeddings.end(),bits.begin(),bits.end());
        }
        candidate->append_prefill(std::span<const std::int64_t>(ids.data()+position,count));
        cuda_check(cudaDeviceSynchronize(),"chunk output ready");
        for(int t=0;t<kTapCount;++t)
            require(target_continue_tap_bits(*candidate,kTapLayers[t],count)==expected[t],"chunk all-row tap mismatch");
        require(candidate->embedding_bits_host_for_test()==embeddings,"chunk embedding mismatch");
        require(target_continue_device_bits(candidate->logits_device(),kVocab,"chunk logits")==
                target_continue_device_bits(reference->logits_device(),kVocab,"serial logits"),"chunk final-row logits mismatch");
        commit_current_rows(*candidate,draft,stage,count,position);
        position+=count;
        const bool full_state=boundaries<8 || position==total || position-last_state_cut>=256;
        if(full_state){compare_state(*candidate,*reference);last_state_cut=position;}
        require(candidate->continuation_bytes()==0 && candidate->continuation_rows()==0 &&
                candidate->persistent_bytes()==allocation_before,"chunk allocated continuation/workspace");
        out << position << ',' << count << ",1," << full_state << ",0\n";out.flush();require(out.good(),"chunk boundary output");++boundaries;
    }
    const auto ring_candidate=draft.ring_digest();
    require(draft.ring_count()==std::min(total,2047) && draft.ring_base_abs()==total-std::min(total,2047),"chunk ring positions");
    draft.reset();
    for(int row=0;row<total;++row) {
        for(int t=0;t<kTapCount;++t)
            cuda_check(cudaMemcpyAsync(stage.row_ptrs[t],reference_taps[t].data()+static_cast<std::size_t>(row)*kHidden,
                                       kHidden*2,cudaMemcpyHostToDevice),"replay serial tap row");
        draft.commit_target_block(const_cast<const std::uint16_t**>(stage.row_ptrs.data()),1,row);
    }
    require(draft.ring_digest()==ring_candidate && draft.ring_count()==std::min(total,2047) &&
            draft.ring_base_abs()==total-std::min(total,2047),"chunk ring digest differs from serial ingestion");
    std::ofstream ring(directory/"ring.csv");ring << "layer,digest\n";
    for(int t=0;t<5;++t)ring << t << ',' << ring_candidate[t] << '\n';require(ring.good(),"ring digest output");
    reference.reset();
    auto target_only=target.create_context(false);
    require(target_only->try_enable_oscar_from_environment(),"target-only chunk OSCAR");
    target_only->prefill(std::span<const std::int64_t>(ids.data(),16));
    const auto target_only_bytes=target_only->persistent_bytes();
    for(int position=16;position<total;position+=8)
        target_only->append_prefill(std::span<const std::int64_t>(ids.data()+position,std::min(8,total-position)));
    compare_state(*target_only,*candidate);
    require(target_only->captured_tap_rows()==0 && target_only->continuation_bytes()==0 &&
            target_only->persistent_bytes()==target_only_bytes,"target-only chunk allocated tap/continuation memory");
    for(int i=total;i<total+8;++i){target_only->decode(ids[i]);candidate->decode(ids[i]);}
    compare_state(*target_only,*candidate);
    std::ofstream result(directory/"result.txt");
    result << "PASS total=" << total << " widths=1..8 boundaries=" << boundaries
           << " all_taps_embeddings_logits=exact native_state=exact ring_digest=exact target_only=exact next8=exact extra_workspace=0\n";
    require(result.good(),"chunk result write");
    std::cout << "PREFILL_CHUNK PASS total=" << total << " boundaries=" << boundaries << "\n";
}
