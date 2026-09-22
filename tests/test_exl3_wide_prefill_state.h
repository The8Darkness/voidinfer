#pragma once

void run_wide_prefill_state_qualification(Exl3TextModel& target, Exl3Dflash2DraftModel& draft) {
    const auto directory=std::filesystem::path(env("NINFER_E5A4_OUT"));
    require(!directory.empty() && !std::filesystem::exists(directory),"chunk qualification output must be new");
    std::filesystem::create_directories(directory);
    std::ofstream out(directory/"boundaries.csv");
    out << "position,rows,taps_exact,full_state_checked,continuation_bytes\n";
    require(env("NINFER_EXL3_WIDE_PREFILL")=="1","wide state needs flag1");
    auto source=load_ids(env("NINFER_E5A4_PROMPT_FILE"));
    const int total=env_int("NINFER_E5A4_CONTEXTS",512);
    require((total==512 || total==2080 || total==4160) && target.max_context()>=total+16,"chunk qualification extent");
    std::vector<std::int64_t> ids(total+16);
    for(std::size_t i=0;i<ids.size();++i) ids[i]=source[i%source.size()];
    const auto gdn_slab_setting=env("NINFER_EXL3_GDN_WIDE_SLAB");
    {
        _putenv_s("NINFER_EXL3_WIDE_PREFILL","0");
        _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB","0");
        auto disabled=target.create_context(true);
        _putenv_s("NINFER_EXL3_WIDE_PREFILL","1");
        require(disabled->try_enable_oscar_from_environment(),"disabled wide context OSCAR");
        disabled->prefill(std::span<const std::int64_t>(ids.data(),16));
        const auto before=prefix_retention_snapshot(*disabled);
        bool rejected=false;
        try{disabled->append_prefill_wide(std::span<const std::int64_t>(ids.data()+16,1));}
        catch(const std::exception&){rejected=true;}
        require(rejected && prefix_retention_snapshot(*disabled)==before,"wide context opt-in did not latch or rejection mutated state");
    }
    // The serial reference needs no expanded64 scratch. Its M1 arithmetic is
    // unchanged; keep the qualified32 allocation to avoid two wide contexts
    // exhausting VRAM during the long serial oracle phase.
    const auto wide64_setting=env("NINFER_EXL3_PREFILL_WIDE64");
    _putenv_s("NINFER_EXL3_PREFILL_WIDE64","0");
    const auto shared_layer_setting=env("NINFER_EXL3_SHARED_LAYER_SCRATCH");
    _putenv_s("NINFER_EXL3_SHARED_LAYER_SCRATCH","0");
    const auto direct_async_a_setting=env("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A");
    _putenv_s("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A","0");
    const auto direct_partials_setting=env("NINFER_EXL3_PREFILL_DIRECT_PARTIALS");
    _putenv_s("NINFER_EXL3_PREFILL_DIRECT_PARTIALS","0");
    const auto staged_shape4_setting=env("NINFER_EXL3_PREFILL_STAGED_SHAPE4");
    _putenv_s("NINFER_EXL3_PREFILL_STAGED_SHAPE4","0");
    auto reference=[&] {
        struct ScopedRowpairOff {
            std::string saved=env("NINFER_EXL3_PREFILL_ROWPAIR_K6");
            ScopedRowpairOff(){_putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6","0");}
            ~ScopedRowpairOff(){_putenv_s("NINFER_EXL3_PREFILL_ROWPAIR_K6",saved.c_str());}
        } off;
        return target.create_context(true);
    }();
    _putenv_s("NINFER_EXL3_PREFILL_STAGED_SHAPE4",staged_shape4_setting.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_DIRECT_PARTIALS",direct_partials_setting.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_DIRECT_ASYNC_A",direct_async_a_setting.c_str());
    _putenv_s("NINFER_EXL3_SHARED_LAYER_SCRATCH",shared_layer_setting.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_WIDE64",wide64_setting.c_str());
    _putenv_s("NINFER_EXL3_GDN_WIDE_SLAB",gdn_slab_setting.c_str());
    auto candidate=target.create_context(true);
    {
        struct ScopedSerialOscarEnvironment {
            std::vector<std::pair<std::string,std::string>> saved;
            ScopedSerialOscarEnvironment() {
                for(const char* name : {"NINFER_OSCAR_PREFILL_PACKED_BYTE4", "NINFER_OSCAR_PREFILL_QUERY_PARALLEL",
                    "NINFER_OSCAR_PREFILL_PARALLEL_MERGE", "NINFER_OSCAR_PREFILL_PARALLEL_SCORES",
                    "NINFER_OSCAR_PREFILL_COALESCED_ROTATIONS"}) {
                    saved.emplace_back(name,env(name));
                    _putenv_s(name,"0");
                }
            }
            ~ScopedSerialOscarEnvironment() {
                for(auto i=saved.rbegin();i!=saved.rend();++i)_putenv_s(i->first.c_str(),i->second.c_str());
            }
        } restore;
        require(reference->try_enable_oscar_from_environment(),"reference OSCAR required");
    }
    require(candidate->try_enable_oscar_from_environment(),"candidate OSCAR required");
    const auto reject=[&](auto&& operation) {
        bool failed=false;try{operation();}catch(const std::exception&){failed=true;}
        require(failed,"chunk admission did not reject");
    };
    reject([&]{candidate->append_prefill_wide(std::span<const std::int64_t>(ids.data(),8));});
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
        reject([&]{candidate->append_prefill_wide({});});
        std::vector<std::int64_t> oversized((env("NINFER_EXL3_PREFILL_STAGED_REDUCTION")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE64")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE128")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE256")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE512")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE1024")=="1" ? 1024 : 512) : 256) : 128) : 64) : 32) : 16)+1,0);
        reject([&]{candidate->append_prefill_wide(oversized);});
        reject([&]{candidate->append_prefill(std::span<const std::int64_t>(ids.data(),9));});
        std::int64_t bad=-1;
        reject([&]{candidate->append_prefill_wide(std::span<const std::int64_t>(&bad,1));});bad=kVocab;
        reject([&]{candidate->append_prefill_wide(std::span<const std::int64_t>(&bad,1));});
        require(prefix_retention_snapshot(*candidate)==before,"invalid chunk mutated state");
    }
    candidate->prepare_transaction();candidate->begin_transaction();
    prefix_retention_reject_unchanged(*candidate,[&]{candidate->append_prefill_wide(std::span<const std::int64_t>(ids.data(),8));},"active transaction append");
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
    if (env("NINFER_DFLASH2_PREFILL_BATCH")=="1") {
        const auto digest=draft.ring_digest();
        const auto base=draft.ring_base_abs();const auto size=draft.ring_count();
        const auto taps=const_cast<const std::uint16_t**>(stage.bulk_ptrs.data());
        reject([&]{draft.commit_prefill_block(nullptr,2,16);});
        reject([&]{draft.commit_prefill_block(taps,0,16);});
        reject([&]{draft.commit_prefill_block(taps,17,16);});
        reject([&]{draft.commit_prefill_block(taps,2,-1);});
        reject([&]{draft.commit_prefill_block(taps,1,-1);});
        reject([&]{draft.commit_prefill_block(taps,2,2147483647LL);});
        reject([&]{draft.commit_prefill_block(taps,2,0);});
        std::array<const std::uint16_t*,5> bad{};
        reject([&]{draft.commit_prefill_block(bad.data(),2,16);});
        require(draft.ring_digest()==digest && draft.ring_base_abs()==base &&
                draft.ring_count()==size,"draft prefill rejection mutated ring");
    }
    const int width=env("NINFER_EXL3_PREFILL_STAGED_REDUCTION")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE64")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE128")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE256")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE512")=="1" ? (env("NINFER_EXL3_PREFILL_WIDE1024")=="1" ? 1024 : 512) : 256) : 128) : 64) : 32) : 16;
    int next_width=1,boundaries=0,last_state_cut=16;
    // All128 operator widths are checked separately. Exercise every16-row
    // boundary and tail transition here without doubling the serial context.
    const std::array<int,14> wide_edges={33,47,48,63,64,65,79,80,95,96,111,112,127,128};
    const std::array<int,16> wide256_edges={31,32,63,64,95,96,127,128,159,160,191,192,223,224,255,256};
    const std::array<int,10> large_edges={31,32,127,128,255,256,511,512,1023,1024};
    std::size_t edge_index=0;
    bool wide_rollback_checked=false;
    for(int position=16;position<total;) {
        const int count=std::min(next_width,total-position);
        if(width>=512 && total>512 && next_width>=16) {
            if(edge_index<large_edges.size() && large_edges[edge_index]<=width) next_width=large_edges[edge_index++];
        } else if(width==256 && total>512 && next_width>=16) {
            if(edge_index<wide256_edges.size()) next_width=wide256_edges[edge_index++];
        } else if(width==128 && next_width>=32) {
            if(edge_index<wide_edges.size()) next_width=wide_edges[edge_index++];
        } else if(next_width<width)++next_width;
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
        candidate->append_prefill_wide(std::span<const std::int64_t>(ids.data()+position,count));
        cuda_check(cudaDeviceSynchronize(),"chunk output ready");
        for(int t=0;t<kTapCount;++t)
            require(target_continue_tap_bits(*candidate,kTapLayers[t],count)==expected[t],"chunk all-row tap mismatch");
        require(candidate->embedding_bits_host_for_test()==embeddings,"chunk embedding mismatch");
        require(target_continue_device_bits(candidate->logits_device(),kVocab,"chunk logits")==
                target_continue_device_bits(reference->logits_device(),kVocab,"serial logits"),"chunk final-row logits mismatch");
        if(env("NINFER_EXL3_SHARED_LAYER_SCRATCH")=="1") {
            for(int layer=3;layer<64;layer+=4) {
                const auto actual=candidate->full_attention_qkv_host(layer);
                const auto expected=reference->full_attention_qkv_host(layer);
                require(actual.rows==count && expected.rows==1,"shared scratch QKV rows");
                const auto last_equal=[&](const auto& a,const auto& b,std::size_t features) {
                    return a.size()==static_cast<std::size_t>(count)*features && b.size()==features &&
                        std::equal(a.end()-features,a.end(),b.begin());
                };
                require(last_equal(actual.q_rope,expected.q_rope,6144) &&
                        last_equal(actual.k_rope,expected.k_rope,1024) &&
                        last_equal(actual.v_projection,expected.v_projection,1024),
                        "shared scratch public QKV last-row mismatch");
            }
        }
        commit_current_rows(*candidate,draft,stage,count,position);
        if (count==width && !wide_rollback_checked) {
            wide_rollback_checked=true;
            const auto before=prefix_retention_snapshot(*candidate);
            candidate->begin_transaction();
            candidate->decode(ids[position+count]);
            candidate->rollback_transaction();
            require(prefix_retention_snapshot(*candidate)==before,"wide transaction rollback");
            compare_state(*candidate,*reference);
        }
        position+=count;
        const bool full_state=boundaries<width || position==total || position-last_state_cut>=256;
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
    // Preserve every byte compared by compare_state, then release the wide
    // context before constructing target-only. No second wide context residency.
    const auto native_snapshot=[&](Exl3TextContext& context) {
        cuda_check(cudaDeviceSynchronize(),"wide host snapshot ready");
        std::vector<std::byte> result;
        const auto append=[&](const auto& values) {
            const std::uint64_t count=values.size();
            const auto* size_bytes=reinterpret_cast<const std::byte*>(&count);
            result.insert(result.end(),size_bytes,size_bytes+sizeof(count));
            if(values.empty()) return;
            const auto* first=reinterpret_cast<const std::byte*>(values.data());
            result.insert(result.end(),first,first+values.size()*sizeof(values[0]));
        };
        append(std::array<int,2>{context.position(),context.device_position_host()});
        append(target_continue_device_bits(context.logits_device(),kVocab,"wide snapshot logits"));
        for(int layer=0;layer<64;++layer) if((layer+1)%4!=0) {
            append(context.gdn_state_host(layer));
            append(context.gdn_physical_conv_host(layer));
        }
        append(context.oscar_live_state_host_for_test());
        return result;
    };
    const auto expected_prefill=native_snapshot(*reference);
    require(native_snapshot(*candidate)==expected_prefill,"candidate serial prefill full host state");
    for(int i=total;i<total+8;++i) {
        reference->decode(ids[i]); candidate->decode(ids[i]);
        require(native_snapshot(*candidate)==native_snapshot(*reference),"candidate serial next8 full host state");
    }
    const auto expected_next8=native_snapshot(*reference);
    reference.reset();
    candidate.reset();
    auto target_only=target.create_context(false);
    require(target_only->try_enable_oscar_from_environment(),"target-only chunk OSCAR");
    target_only->prefill(std::span<const std::int64_t>(ids.data(),16));
    const auto target_only_bytes=target_only->persistent_bytes();
    for(int position=16;position<total;position+=width)
        target_only->append_prefill_wide(std::span<const std::int64_t>(ids.data()+position,std::min(width,total-position)));
    require(native_snapshot(*target_only)==expected_prefill,"target-only full host state");
    require(target_only->captured_tap_rows()==0 && target_only->continuation_bytes()==0 &&
            target_only->persistent_bytes()==target_only_bytes,"target-only chunk allocated tap/continuation memory");
    for(int i=total;i<total+8;++i) target_only->decode(ids[i]);
    require(native_snapshot(*target_only)==expected_next8,"target-only next8 full host state");
    std::ofstream result(directory/"result.txt");
    result << "PASS total=" << total << " max_width=" << width << " boundaries=" << boundaries
           << " all_taps_embeddings_logits=exact native_state=exact ring_digest=exact target_only=exact next8=exact extra_workspace=0"
           << " qkv_last_row=" << (env("NINFER_EXL3_SHARED_LAYER_SCRATCH")=="1" ? "exact" : "not_requested") << "\n";
    require(result.good(),"chunk result write");
    std::cout << "WIDE_PREFILL_STATE PASS total=" << total << " boundaries=" << boundaries << "\n";
}
