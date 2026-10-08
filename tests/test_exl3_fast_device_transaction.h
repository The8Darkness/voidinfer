#pragma once

// Bounded real-draft probe for the ordinary device-KV transaction. This is a
// request-owned single-context lane without VeriCache/Engine publication.
// It includes actual draft proposals, target verification, accepted-prefix
// repair, committed tap D2D staging, and draft ring commit in its decode wall.
void run_fast_device_real_dflash(Exl3TextModel& target,
    Exl3Dflash2DraftModel& draft,const std::vector<std::int64_t>& ids) {
    using Clock=std::chrono::steady_clock;
    using ninfer::exl3::Exl3VeriCacheServingPrefixCache;
    using ninfer::exl3::Exl3VeriCacheServingCoordinator;
    using ninfer::exl3::Exl3VeriCacheServingIdentity;
    using ninfer::exl3::Exl3ResourceInventory;
    const auto prefix_option=env("NINFER_E5A4_CONTEXTS");
    const bool custom_prompt=
        env("NINFER_EXL3_TEST_FAST_DEVICE_CUSTOM_PROMPT")=="1";
    const int prefix_rows=std::stoi(prefix_option);
    require(prefix_option=="4096" || prefix_option=="16384" ||
            (custom_prompt && ((prefix_rows>=4096 && prefix_rows<=4224) ||
                (prefix_rows>=16384 && prefix_rows<=16512))),
        "real device DFlash probe requires 4K or 16K fixture");
    const auto output_option=env("NINFER_EXL3_TEST_FAST_DEVICE_REAL_OUTPUTS");
    require(output_option=="128" || output_option=="512",
        "real device DFlash output budget must be 128 or 512");
    const int output_rows=std::stoi(output_option);
    require(ids.size()>=static_cast<std::size_t>(prefix_rows) &&
            (!custom_prompt || ids.size()==static_cast<std::size_t>(prefix_rows)) &&
            target.max_context()>=prefix_rows+output_rows+
                (custom_prompt?0:128),
        "real device DFlash fixture and capacity");
    const bool draft_layer_major=
        env("NINFER_EXL3_FAST_LAYER_MAJOR_PREFILL")=="1";
    require(env("NINFER_EXL3_FAST_DEVICE_KV_TRANSACTION")=="1" &&
            env("NINFER_EXL3_EXACT_HOST_KV")=="0" &&
            env("NINFER_OSCAR_EXL3")=="0" &&
            (!draft_layer_major || env("NINFER_DFLASH2_PREFILL_WINDOW")=="1"),
        "real device DFlash probe numeric and prefill policy");
    if(env("NINFER_EXL3_TEST_FAST_DEVICE_RING_UNDO")=="1") {
        // End four positions before a 2048-slot boundary so the eight-row
        // trial overwrites both sides of the physical ring wrap.
        const int root_position=prefix_rows-4;
        require(root_position>2048 && !draft_layer_major,
            "ring undo fixture needs a chunk-major saturated root");
        TapStage stage;
        for(int tap=0;tap<kTapCount;++tap) {
            stage.bulk.push_back(std::make_unique<DeviceBuffer>(
                16ull*kHidden*sizeof(std::uint16_t)));
            stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(
                stage.bulk.back()->get()));
        }
        std::array<const std::uint16_t*,5> sources{};
        std::copy(stage.bulk_ptrs.begin(),stage.bulk_ptrs.end(),sources.begin());
        auto live=target.create_context(true);
        draft.reset();
        draft.bind_ring_scope(1,1);
        const std::vector<std::int64_t> ring_prefix(
            ids.begin(),ids.begin()+root_position);
        ingest_prefix(*live,ring_prefix,CommitSink{&draft,&stage});
        for(auto* plane:stage.bulk_ptrs)
            cuda_check(cudaMemset(plane,0,8ull*kHidden*sizeof(std::uint16_t)),
                "ring undo zero fixture taps");
        require(live->position()==root_position &&
                draft.ring_count()==draft.ring_keep() &&
                draft.ring_base_abs()+draft.ring_count()==root_position,
            "ring undo saturated root extent");
        const auto root=draft.export_host_ring(nullptr,false);
        const auto physical_root=draft.physical_ring_digest_for_test();
        const auto represented_root=draft.ring_digest();
        for(int rows:{1,8}) {
            draft.begin_prefill_ring_undo(rows,root_position);
            bool refused=false;
            try { draft.begin_prefill_ring_undo(rows,root_position); }
            catch(const std::exception&) { refused=true; }
            require(refused,"ring undo accepted a second active snapshot");
            refused=false;
            try { draft.commit_prefill_block(sources.data(),rows==1?2:1,
                    root_position); }
            catch(const std::exception&) { refused=true; }
            require(refused,"ring undo accepted a mismatched commit");
            draft.rollback_prefill_ring_undo();
            require(draft.physical_ring_digest_for_test()==physical_root &&
                    draft.ring_digest()==represented_root,
                "ring undo armed rollback changed physical root");
            draft.begin_prefill_ring_undo(rows,root_position);
            draft.commit_prefill_block(sources.data(),rows,root_position);
            require(draft.ring_base_abs()+draft.ring_count()==root_position+rows,
                "ring undo applied frontier");
            draft.rollback_prefill_ring_undo();
            auto restored=draft.export_host_ring(nullptr,false);
            require(draft.physical_ring_digest_for_test()==physical_root &&
                    draft.ring_digest()==represented_root &&
                    restored->same_represented_payload_for_test(*root),
                "ring undo rollback changed physical slots or represented payload");
            std::cout<<"FAST_DEVICE_RING_UNDO rollback_rows="<<rows
                     <<" wrap="<<(rows==8)<<" PASS\n";
        }
        draft.begin_prefill_ring_undo(8,root_position);
        draft.commit_prefill_block(sources.data(),8,root_position);
        draft.accept_prefill_ring_undo();
        const auto accepted_physical=draft.physical_ring_digest_for_test();
        const auto accepted_represented=draft.ring_digest();
        draft.restore_host_ring(root);
        draft.commit_prefill_block(sources.data(),8,root_position);
        require(draft.physical_ring_digest_for_test()==accepted_physical &&
                draft.ring_digest()==accepted_represented,
            "ring undo accepted commit differs from ordinary commit");
        draft.reset();
        std::cout<<"FAST_DEVICE_RING_UNDO PASS root="<<root_position
                 <<" rows=1,8 physical_slots=2048 accept=1\n";
        return;
    }
    const auto width_option=env("NINFER_EXL3_TEST_FAST_DEVICE_REAL_WIDTH");
    require(width_option=="4" || width_option=="6" || width_option=="8",
        "real device DFlash width must be 4, 6 or 8");
    const int real_width=std::stoi(width_option);
    const bool adaptive_width=
        env("NINFER_EXL3_TEST_FAST_DEVICE_ADAPTIVE_WIDTH")=="1";
    const auto seed_reuse_option=env("NINFER_EXL3_TEST_FAST_DEVICE_SEED_REUSE");
    require(seed_reuse_option.empty() || seed_reuse_option=="0" ||
            seed_reuse_option=="1", "real device DFlash seed reuse must be 0 or 1");
    const bool seed_reuse=seed_reuse_option=="1";
    const auto shared_round_option=env("NINFER_EXL3_TEST_SHARED_DEVICE_ROUND");
    require(shared_round_option.empty() || shared_round_option=="0" ||
            shared_round_option=="1",
        "shared device round selection must be 0 or 1");
    const bool shared_round=shared_round_option=="1";
    const bool pending_round=
        env("NINFER_EXL3_TEST_FAST_DEVICE_PENDING_ROUND")=="1";
    const bool device_pending_round=
        env("NINFER_EXL3_TEST_FAST_DEVICE_DEVICE_PENDING_ROUND")=="1";
    require(!(pending_round && device_pending_round) &&
            (!(pending_round || device_pending_round) || shared_round),
        "pending device round requires the shared request round");
    const bool coherent_down_k6=env("NINFER_EXL3_COHERENT_DOWN_K6")=="1";
    const auto coherent_calls_before=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k6_calls_for_test();
    const auto coherent_rows_before=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k6_rows_for_test();
    const bool coherent_down_k7=env("NINFER_EXL3_COHERENT_DOWN_K7")=="1";
    const auto coherent_down_k7_calls_before=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k7_calls_for_test();
    const auto coherent_down_k7_rows_before=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k7_rows_for_test();
    const bool coherent_o_k7=env("NINFER_EXL3_COHERENT_O_K7")=="1";
    const auto coherent_o_calls_before=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_o_k7_calls_for_test();
    const auto coherent_o_rows_before=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_o_k7_rows_for_test();
    const bool coherent_wide_k6=
        env("NINFER_EXL3_TARGET_COHERENT_WIDE_K6")=="1";
    std::uint64_t coherent_wide_calls_before[5]{},
        coherent_wide_rows_before[5]{},
        coherent_wide_calls_after_scalar[5]{},
        coherent_wide_rows_after_scalar[5]{};
    for(int operation=0;operation<5;++operation) {
        coherent_wide_calls_before[operation]=
            ninfer::exl3::Exl3CudaLinearWorkspace::
                coherent_wide_k6_calls_for_test(operation);
        coherent_wide_rows_before[operation]=
            ninfer::exl3::Exl3CudaLinearWorkspace::
                coherent_wide_k6_rows_for_test(operation);
    }
    require(!adaptive_width || real_width==8,
        "adaptive real device DFlash requires B8 upper horizon");
    const std::vector<std::int64_t> prefix(ids.begin(),ids.begin()+prefix_rows);
    if(custom_prompt &&
       env("NINFER_EXL3_TEST_FAST_DEVICE_RESERVED_CONTEXT")=="1") {
        auto cold=target.create_context(true);
        if(env("NINFER_EXL3_L0_OSCAR")!="0") {
            // L0 OSCAR contexts prefill row-major in 1024-row chunks.
            const std::span<const std::int64_t> all(prefix);
            cold->prefill(all.first(16));
            for(std::size_t first=16;first<all.size();first+=1024)
                cold->append_prefill_wide(all.subspan(first,std::min<std::size_t>(1024,all.size()-first)));
        } else {
        const auto initial=static_cast<std::size_t>(
            ninfer::exl3::Exl3TextContext::layer_major_initial_rows());
        if(initial)cold->prefill(std::span<const std::int64_t>(prefix).first(initial));
        cold->append_prefill_layer_major(
            std::span<const std::int64_t>(prefix).subspan(initial));
        }
        cuda_check(cudaStreamSynchronize(nullptr),
            "matched cold target-only prefill completion");
        const auto cold_root=cold->export_exact_host_state();
        const auto cold_first=exl3_branch_greedy(*cold);
        cold->decode(cold_first);
        const auto cold_second=exl3_branch_greedy(*cold);
        std::cout<<"FAST_DEVICE_COLD_TARGET_ONLY hash="
                 <<cold_root->represented_payload_hash_for_test()
                 <<" pair="<<cold_first<<','<<cold_second<<'\n';
        cold.reset();
        cuda_check(cudaDeviceSynchronize(),
            "matched cold target-only release");
    }
    std::vector<std::int64_t> oracle;
    oracle.reserve(output_rows);
    auto scalar=target.create_context(true);
    ingest_prefix(*scalar,prefix,CommitSink{});
    const auto initial_hash=scalar->export_exact_host_state()->
        represented_payload_hash_for_test();
    for(int i=0;i<output_rows;++i) {
        const auto token=exl3_branch_greedy(*scalar);
        oracle.push_back(token);
        scalar->decode(token);
    }
    if(const auto path=env("NINFER_EXL3_TEST_FAST_DEVICE_ORACLE_OUT");!path.empty()) {
        std::ofstream out(path,std::ios::binary);
        for(const auto id:oracle)out<<id<<'\n';
        require(out.good(),"real device DFlash oracle token export failed");
    }
    const auto final_hash=scalar->export_exact_host_state()->
        represented_payload_hash_for_test();
    const auto coherent_calls_after_scalar=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k6_calls_for_test();
    const auto coherent_rows_after_scalar=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k6_rows_for_test();
    const auto coherent_down_k7_calls_after_scalar=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k7_calls_for_test();
    const auto coherent_down_k7_rows_after_scalar=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k7_rows_for_test();
    const auto coherent_o_calls_after_scalar=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_o_k7_calls_for_test();
    const auto coherent_o_rows_after_scalar=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_o_k7_rows_for_test();
    for(int operation=0;operation<5;++operation) {
        coherent_wide_calls_after_scalar[operation]=
            ninfer::exl3::Exl3CudaLinearWorkspace::
                coherent_wide_k6_calls_for_test(operation);
        coherent_wide_rows_after_scalar[operation]=
            ninfer::exl3::Exl3CudaLinearWorkspace::
                coherent_wide_k6_rows_for_test(operation);
    }
    scalar.reset();
    cuda_check(cudaDeviceSynchronize(),
        "release real device DFlash scalar oracle before live probe");

    struct DiagnosticStream {
        cudaStream_t value=nullptr;
        ~DiagnosticStream(){if(value)cudaStreamDestroy(value);}
    } diagnostic_stream;
    if(env("NINFER_EXL3_TEST_FAST_DEVICE_OWN_STREAM")=="1") {
        require(shared_round && custom_prompt,
            "diagnostic nondefault stream needs matched shared round");
        cuda_check(cudaStreamCreateWithFlags(&diagnostic_stream.value,
            cudaStreamNonBlocking),"create matched Engine diagnostic stream");
    }

    TapStage stage;
    for(int tap=0;tap<kTapCount;++tap) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(
            16ull*kHidden*sizeof(std::uint16_t)));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(
            stage.bulk.back()->get()));
    }
    if(!shared_round)draft.reset();
    const auto live_request_start=Clock::now();
    const bool engine_context_shape=
        env("NINFER_EXL3_TEST_FAST_DEVICE_ENGINE_CONTEXT_SHAPE")=="1";
    if(engine_context_shape)
        require(shared_round && custom_prompt,
            "Engine context shape diagnostic needs matched shared round");
    const bool reserved_context=
        env("NINFER_EXL3_TEST_FAST_DEVICE_RESERVED_CONTEXT")=="1";
    const bool reserved_lane=
        env("NINFER_EXL3_TEST_FAST_DEVICE_RESERVED_LANE")=="1";
    std::unique_ptr<Exl3VeriCacheServingPrefixCache> diagnostic_cache;
    std::unique_ptr<Exl3VeriCacheServingCoordinator> diagnostic_coordinator;
    std::shared_ptr<Exl3TextContext> live;
    if(reserved_context) {
        require(shared_round && custom_prompt && engine_context_shape,
            "reserved context diagnostic needs Engine shape and matched prompt");
        Exl3VeriCacheServingIdentity identity{"matched-device-round",
            "SC_6.00bpw_H6_V6","pinned-ids","exact-B8-L2","text"};
        diagnostic_cache=std::make_unique<Exl3VeriCacheServingPrefixCache>(
            Exl3VeriCacheServingPrefixCache::Policy{2,64,2ULL<<30,8ULL<<30},
            identity);
        diagnostic_coordinator=std::make_unique<Exl3VeriCacheServingCoordinator>(
            *diagnostic_cache,Exl3VeriCacheServingCoordinator::Policy{
                1,1,16ULL<<30,8ULL<<30});
        diagnostic_coordinator->bind_physical_resources({},
            Exl3ResourceInventory::unlimited());
        live=target.create_context_reserved(*diagnostic_coordinator);
    } else {
        live=target.create_context(true,!engine_context_shape,
            engine_context_shape);
    }
    if(env("NINFER_EXL3_TEST_FAST_DEVICE_PREPARE_BEFORE_PREFIX")=="1") {
        require(shared_round && custom_prompt,
            "early continuation preparation needs matched shared round");
        if(reserved_context) {
            live->prepare_continuation_reserved(*diagnostic_coordinator,8);
            live->prepare_transaction_reserved(*diagnostic_coordinator);
        } else {
            live->prepare_continuation(8);
            live->prepare_transaction();
        }
    }
    std::shared_ptr<ninfer::exl3::Exl3Dflash2Execution> diagnostic_lane;
    if(reserved_lane) {
        require(reserved_context,
            "reserved lane diagnostic needs reserved context");
        std::shared_ptr<Exl3Dflash2DraftModel> retained_draft(&draft,
            [](Exl3Dflash2DraftModel*){});
        diagnostic_lane=ninfer::exl3::Exl3Dflash2Execution::create_reserved(
            *diagnostic_coordinator,live,std::move(retained_draft),
            "matched-device-round-diagnostic",diagnostic_stream.value,{});
    }
    if(reserved_context) {
        std::size_t available=0,total=0;
        cuda_check(cudaMemGetInfo(&available,&total),
            "matched reserved context memory query");
        std::cout<<"FAST_DEVICE_RESERVED_MEMORY free_mib="
                 <<available/(1024*1024)<<'\n';
    }
    if(env("NINFER_EXL3_TEST_FAST_DEVICE_RESET_FOR_REQUEST")=="1") {
        require(shared_round && custom_prompt,
            "request reset diagnostic needs matched shared round");
        if(!reserved_lane)
            live->bind_request_compatibility("matched-device-round-diagnostic");
        live->reset_for_request("matched-device-round-diagnostic");
    }
    std::unique_ptr<ninfer::exl3::Exl3FastDeviceRound> reusable_round;
    if(shared_round) {
        std::array<std::uint16_t*,5> stage_planes{};
        std::copy(stage.bulk_ptrs.begin(),stage.bulk_ptrs.end(),
            stage_planes.begin());
        reusable_round=std::make_unique<ninfer::exl3::Exl3FastDeviceRound>(
            live,draft,stage_planes,1,1,seed_reuse,diagnostic_stream.value);
    }
    const auto live_prefill_start=Clock::now();
    if(reusable_round) {
        reusable_round->begin_fresh(std::span<const std::int64_t>(prefix),
            draft_layer_major);
        diagnose_fast_device_prefill(*live,draft,prefix_rows,true);
        if(env("NINFER_EXL3_TEST_FAST_DEVICE_SPLIT_ROUND")=="1") {
            require(custom_prompt,
                "split prefill and decode rounds need matched Engine prompt");
            reusable_round.reset();
            std::array<std::uint16_t*,5> stage_planes{};
            std::copy(stage.bulk_ptrs.begin(),stage.bulk_ptrs.end(),
                stage_planes.begin());
            reusable_round=std::make_unique<ninfer::exl3::Exl3FastDeviceRound>(
                live,draft,stage_planes,1,1,seed_reuse,diagnostic_stream.value);
        }
    } else ingest_prefix(*live,prefix,CommitSink{&draft,&stage});
    if(reserved_context) {
        const auto scalar_root=live->export_exact_host_state(diagnostic_stream.value);
        const auto first=exl3_branch_greedy(*live,diagnostic_stream.value);
        live->decode(first,diagnostic_stream.value);
        const auto second=exl3_branch_greedy(*live,diagnostic_stream.value);
        const auto host_logits=live->logits_host(diagnostic_stream.value);
        const auto host_second=static_cast<std::int64_t>(
            std::max_element(host_logits.begin(),host_logits.end())-
                host_logits.begin());
        live->restore_exact_host_state(*scalar_root,diagnostic_stream.value);
        std::cout<<"FAST_DEVICE_RESERVED_SCALAR_PAIR "<<first<<','<<second
                 <<" host_second="<<host_second<<'\n';
    }
    const auto live_initial_hash=live->export_exact_host_state()->
        represented_payload_hash_for_test();
    const auto live_prefill_end=Clock::now();
    std::cout<<"FAST_DEVICE_REAL_DFLASH_ROOT scalar_hash="<<initial_hash
             <<" live_hash="<<live_initial_hash
             <<" live_position="<<live->position()
             <<" ring_base="<<draft.ring_base_abs()
             <<" ring_count="<<draft.ring_count()<<'\n';
    require(live->position()==prefix_rows && draft.ring_count()>0 &&
            draft.ring_base_abs()+draft.ring_count()==prefix_rows &&
            // L0 OSCAR: the oracle ingests through the INT2-history route while
            // the live wide prefill attends exactly, so the roots differ by design.
            (live_initial_hash==initial_hash || env("NINFER_EXL3_L0_OSCAR")!="0"),
        "real device DFlash root frontier differs from target-only oracle");
    std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> diagnostic_root;
    if(env("NINFER_EXL3_TEST_FAST_DEVICE_EXPORT_ROOT")=="1") {
        require(shared_round && custom_prompt,
            "device root export diagnostic needs matched shared round");
        diagnostic_root=ninfer::exl3::Exl3VeriCacheRequest::
            initialize_device_root(*live,draft,prefix,diagnostic_stream.value);
    }
    if(shared_round) {
        reusable_round->begin_prefilled();
    } else {
        live->prepare_continuation(8);
        live->prepare_transaction();
    }
    if(!env("NINFER_EXL3_TEST_VERIFY_QUALITY_OUT").empty()) {
        // Verify-path quality diagnostic: teacher-force a canonical
        // continuation through committed 8-row verifier transactions and
        // record, per row, the NLL of the next canonical token and whether the
        // row argmax equals it. The DFlash round is not run in this mode.
        std::ifstream token_file(env("NINFER_EXL3_TEST_VERIFY_QUALITY_TOKENS"));
        std::vector<std::int64_t> tokens;
        for(std::int64_t token=0;token_file>>token;)tokens.push_back(token);
        require(tokens.size()>=9,"verify quality continuation too short");
        std::ofstream quality(env("NINFER_EXL3_TEST_VERIFY_QUALITY_OUT"));
        quality<<"index,target,nll,argmax\n";
        constexpr int kVocabRows=248320;
        double total_nll=0.0;std::size_t scored=0,agree=0;
        // Block length (1..8; default 8) selects the verifier row count, so
        // decode-sized attention/GEMM routes can be gated by the same metric.
        const int block_rows=env("NINFER_EXL3_TEST_VERIFY_QUALITY_ROWS").empty()?8:
            std::stoi(env("NINFER_EXL3_TEST_VERIFY_QUALITY_ROWS"));
        require(block_rows>=1&&block_rows<=8,"verify quality rows");
        for(std::size_t first=0;first+block_rows<tokens.size();first+=block_rows) {
            const std::span<const std::int64_t> block(tokens.data()+first,block_rows);
            live->begin_transaction(diagnostic_stream.value);
            live->continue_rows(block,diagnostic_stream.value);
            const auto logits=live->continuation_logits_host(diagnostic_stream.value);
            require(logits.size()>=static_cast<std::size_t>(block_rows)*kVocabRows,
                "verify quality logits extent");
            live->commit_transaction();
            for(int row=0;row<block_rows;++row) {
                const float* values=logits.data()+static_cast<std::size_t>(row)*kVocabRows;
                const auto target=tokens[first+row+1];
                float maximum=-INFINITY;int arg=0;
                for(int i=0;i<kVocabRows;++i)if(values[i]>maximum){maximum=values[i];arg=i;}
                double sum=0.0;
                for(int i=0;i<kVocabRows;++i)sum+=std::exp(static_cast<double>(values[i])-maximum);
                const double nll=std::log(sum)+maximum-values[target];
                require(std::isfinite(maximum)&&std::isfinite(nll)&&nll>=-1e-6,
                    "verify quality row is not finite");
                total_nll+=nll;++scored;agree+=arg==target;
                quality<<first+row+1<<','<<target<<','<<nll<<','<<arg<<'\n';
            }
        }
        quality.flush();
        require(quality.good()&&scored>0,"verify quality output write");
        std::cout<<"VERIFY_QUALITY PASS rows="<<scored<<" mean_nll="<<total_nll/scored
                 <<" argmax_agree="<<agree<<'\n';
        return;
    }
    if(pending_round || device_pending_round) {
        const std::array<std::int64_t,2> model_stops{248046,248044};
        const std::span<const std::int64_t> pending_terminal=
            env("NINFER_EXL3_TEST_FAST_DEVICE_MODEL_STOPS")=="1"?
                std::span<const std::int64_t>(model_stops):
                std::span<const std::int64_t>{};
        if(device_pending_round)draft.bind_ring_scope(1,1);
        const auto root_state=live->export_exact_host_state();
        const auto root_ring=draft.export_host_ring();
        auto physical_root=device_pending_round?
            draft.physical_ring_digest_for_test():std::array<std::uint64_t,5>{};
        const auto require_root=[&](const char* label) {
            require(live->position()==prefix_rows &&
                live->export_exact_host_state()->represented_payload_hash_for_test()==
                    root_state->represented_payload_hash_for_test() &&
                draft.export_host_ring()->same_represented_payload_for_test(*root_ring) &&
                (!device_pending_round ||
                    draft.physical_ring_digest_for_test()==physical_root),
                label);
        };
        const auto prepare=[&] {
            auto pending=device_pending_round?
                reusable_round->prepare_device_pending(8,pending_terminal):
                reusable_round->prepare_pending(8,pending_terminal);
            require(!pending.candidate.committed_tokens.empty() &&
                pending.candidate.committed_tokens.size()<=8 &&
                std::equal(pending.candidate.committed_tokens.begin(),
                    pending.candidate.committed_tokens.end(),oracle.begin()) &&
                live->position()==prefix_rows+
                    static_cast<int>(pending.candidate.committed_tokens.size()) &&
                draft.export_host_ring()->same_represented_payload_for_test(*root_ring),
                "pending round candidate or unpublished ring differs");
            return pending;
        };
        const auto make_round=[&] {
            std::array<std::uint16_t*,5> planes{};
            std::copy(stage.bulk_ptrs.begin(),stage.bulk_ptrs.end(),planes.begin());
            auto next=std::make_unique<ninfer::exl3::Exl3FastDeviceRound>(
                live,draft,planes,1,1,seed_reuse,diagnostic_stream.value);
            next->begin_prefilled();
            return next;
        };
        const auto restore_root=[&] {
            reusable_round.reset();
            live->restore_exact_host_state(*root_state);
            draft.restore_host_ring(root_ring);
            if(device_pending_round) {
                draft.bind_ring_scope(1,1);
                physical_root=draft.physical_ring_digest_for_test();
            }
            reusable_round=make_round();
            require_root("pending round root restore differs");
        };
        const auto canceled=prepare();
        if(device_pending_round)
            reusable_round->cancel_device_pending(canceled.ticket);
        else reusable_round->cancel_pending(canceled.ticket);
        require_root("pending round cancellation changed root");
        const auto refused=prepare();
        bool callback_threw=false;
        try {
            const auto reject=[](const ninfer::exl3::Exl3FastDeviceRound::Step&) {
                throw std::runtime_error("pending publication fault");
            };
            if(device_pending_round)
                reusable_round->settle_device_pending(refused.ticket,
                    refused.candidate.committed_tokens.size(),reject);
            else reusable_round->settle_pending(refused.ticket,
                    refused.candidate.committed_tokens.size(),reject);
        } catch(const std::runtime_error& error) {
            callback_threw=std::string_view(error.what())==
                "pending publication fault";
        }
        require(callback_threw && !reusable_round->poisoned(),
            "pending round publication failure was not recoverable");
        require_root("pending publication failure changed root");
        const auto full=prepare();
        if(reserved_context) {
            std::cout<<"FAST_DEVICE_RESERVED_FIRST accepted="
                     <<full.candidate.verification.accepted<<" committed=";
            for(const auto token:full.candidate.committed_tokens)
                std::cout<<token<<',';
            std::cout<<" proposal=";
            for(int row=0;row<full.candidate.width;++row)
                std::cout<<full.candidate.proposal[row]<<',';
            std::cout<<'\n';
        }
        const auto full_step=device_pending_round?
            reusable_round->settle_device_pending(full.ticket,
                full.candidate.committed_tokens.size()):
            reusable_round->settle_pending(full.ticket,
                full.candidate.committed_tokens.size());
        const auto full_hash=live->export_exact_host_state()->
            represented_payload_hash_for_test();
        const auto full_ring=draft.export_host_ring()->detached_payload_for_test();
        require(full_step.committed_tokens==full.candidate.committed_tokens &&
            draft.ring_base_abs()+draft.ring_count()==live->position(),
            "pending full settlement frontier or tokens differ");
        restore_root();
        const auto short_candidate=prepare();
        require(short_candidate.candidate.committed_tokens.size()>1,
            "pending short-prefix fixture needs a multi-token candidate");
        const auto short_step=device_pending_round?
            reusable_round->settle_device_pending(short_candidate.ticket,1):
            reusable_round->settle_pending(short_candidate.ticket,1);
        const auto short_hash=live->export_exact_host_state()->
            represented_payload_hash_for_test();
        const auto short_ring=draft.export_host_ring()->detached_payload_for_test();
        require(short_step.committed_tokens.size()==1 &&
            short_step.committed_tokens.front()==oracle.front() &&
            live->position()==prefix_rows+1 &&
            draft.ring_base_abs()+draft.ring_count()==live->position(),
            "pending short settlement frontier or token differs");
        restore_root();
        const auto eager_short=reusable_round->step(1);
        require(eager_short.committed_tokens==short_step.committed_tokens &&
            live->export_exact_host_state()->represented_payload_hash_for_test()==
                short_hash &&
            draft.export_host_ring()->same_represented_payload_for_test(*short_ring),
            "pending short settlement differs from eager single-row reference");
        restore_root();
        const auto eager_full=reusable_round->step(8);
        require(eager_full.committed_tokens==full_step.committed_tokens &&
            live->export_exact_host_state()->represented_payload_hash_for_test()==
                full_hash &&
            draft.export_host_ring()->same_represented_payload_for_test(*full_ring),
            "pending full settlement differs from eager same-policy reference");
        if(device_pending_round) {
            restore_root();
            const auto single=reusable_round->prepare_device_pending(1);
            require(single.candidate.committed_tokens.size()==1 &&
                single.candidate.committed_tokens.front()==oracle.front(),
                "single-row device pending candidate");
            reusable_round->cancel_device_pending(single.ticket);
            require_root("single-row device pending cancellation changed root");
            const auto single_commit=reusable_round->prepare_device_pending(1);
            const auto single_step=reusable_round->settle_device_pending(
                single_commit.ticket,1);
            require(single_step.committed_tokens.size()==1 &&
                live->position()==prefix_rows+1 &&
                draft.ring_base_abs()+draft.ring_count()==live->position(),
                "single-row device pending settlement frontier");
        }
        reusable_round->finish();
        std::cout<<(device_pending_round?"FAST_DEVICE_DEVICE_PENDING PASS prefix=":
                "FAST_DEVICE_PENDING PASS prefix=")<<prefix_rows
                 <<" candidate="<<full_step.committed_tokens.size()
                 <<" short=1 cancel=1 callback_rollback=1"
                 <<" single="<<(device_pending_round?1:0)
                 <<" full_hash="<<full_hash
                 <<" short_hash="<<short_hash<<'\n';
        return;
    }
    const bool projection_diagnostic=
        env("NINFER_EXL3_TEST_FAST_DEVICE_PROJECTION_TIMING")=="1";
    require(!shared_round || !projection_diagnostic,
        "shared device round projection diagnostic uses the legacy isolated gate");
    if(projection_diagnostic)live->prepare_target_projection_timing();
    const auto live_prepare_end=Clock::now();
    std::vector<std::int64_t> committed;
    committed.reserve(output_rows);
    std::array<std::uint64_t,9> accepted_hist{};
    std::array<std::uint64_t,9> useful_hist{};
    const bool round_state_diagnostic=
        env("NINFER_EXL3_TEST_FAST_DEVICE_ROUND_STATE")=="1";
    const bool stage_timeline_diagnostic=
        env("NINFER_EXL3_TEST_FAST_DEVICE_STAGE_TIMELINE")=="1";
    std::vector<std::pair<int,std::uint64_t>> round_state_hashes;
    std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> first_round_state;
    std::array<std::int64_t,8> first_proposal{};
    double proposal_ms=0.0,verifier_ms=0.0,ring_ms=0.0;
    double target_seed_ms=0.0,draft_api_ms=0.0;
    ninfer::exl3::Exl3OuterDeviceStageTimeline verifier_stages{};
    std::size_t verified_rows=0,replayed_rows=0,rounds=0,d2d_bytes=0;
    std::size_t reused_seeds=0,seed_fallbacks=0;
    std::size_t fused_flash_multirow_calls=0;
    ninfer::exl3::Exl3DeviceHorizonCostPolicy horizon_policy;
    struct PendingHorizonCost {
        unsigned width=0,useful=0;
        Clock::time_point start{};
    };
    std::optional<PendingHorizonCost> pending_horizon_cost;
    std::array<std::uint64_t,9> width_hist{};
    const auto settle_horizon_cost=[&](Clock::time_point ready) {
        if(!pending_horizon_cost)return;
        const auto previous=*pending_horizon_cost;
        pending_horizon_cost.reset();
        if(previous.width==4 || previous.width==8)
            horizon_policy.observe(previous.width,previous.useful,
                std::chrono::duration<double,std::milli>(
                    ready-previous.start).count());
    };
    const auto decode_start=Clock::now();
    while(committed.size()<static_cast<std::size_t>(output_rows)) {
        const int root_position=live->position();
        const int remaining=output_rows-static_cast<int>(committed.size());
        if(reusable_round) {
            const auto fused_before=ninfer::exl3::
                exl3_fast_fused_flash_multirow_attention_calls_for_test();
            ninfer::exl3::Exl3OuterDeviceStageTimeline round_timeline{};
            ninfer::exl3::Exl3FastDeviceRound::ChooseWidth choose;
            if(adaptive_width && remaining>1)
                choose=[&](Clock::time_point seed_ready,int maximum) {
                    settle_horizon_cost(seed_ready);
                    return std::min(static_cast<int>(horizon_policy.select(remaining)),
                        maximum);
                };
            auto outcome=reusable_round->step(std::min(real_width,remaining),{},
                stage_timeline_diagnostic?&round_timeline:nullptr,{},{},choose);
            fused_flash_multirow_calls+=ninfer::exl3::
                exl3_fast_fused_flash_multirow_attention_calls_for_test()-
                fused_before;
            if(stage_timeline_diagnostic) {
                target_seed_ms+=outcome.target_seed_ms;
                draft_api_ms+=outcome.draft_api_ms;
                verifier_stages.begin_ms+=round_timeline.begin_ms;
                verifier_stages.seed_ms+=round_timeline.seed_ms;
                verifier_stages.submit_ms+=round_timeline.submit_ms;
                verifier_stages.decision_ms+=round_timeline.decision_ms;
                verifier_stages.settlement_ms+=round_timeline.settlement_ms;
            }
            proposal_ms+=outcome.proposal_ms;
            verifier_ms+=outcome.verifier_ms;
            ring_ms+=outcome.ring_commit_ms;
            d2d_bytes+=outcome.staged_bytes;
            reused_seeds+=outcome.verification.device_seed_reused;
            seed_fallbacks+=outcome.verification.device_seed_fallback;
            if(round_state_diagnostic && rounds==0)
                first_proposal=outcome.proposal;
            committed.insert(committed.end(),outcome.committed_tokens.begin(),
                outcome.committed_tokens.end());
            if(round_state_diagnostic)
                round_state_hashes.emplace_back(live->position(),
                    live->export_exact_host_state()->
                        represented_payload_hash_for_test());
            if(outcome.single_row) {
                ++replayed_rows;
                if(adaptive_width)settle_horizon_cost(Clock::now());
                break;
            }
            if(round_state_diagnostic && rounds==0)
                first_round_state=live->export_exact_host_state()->
                    detached_payload_for_test();
            if(rounds==0)
                std::cout<<"FAST_DEVICE_REAL_DFLASH_FIRST_MULTIROW_CALLS calls="
                         <<fused_flash_multirow_calls<<'\n';
            ++accepted_hist[outcome.verification.accepted];
            ++useful_hist[outcome.committed_tokens.size()];
            verified_rows+=outcome.verification.verification_rows;
            replayed_rows+=outcome.verification.replay_rows;
            ++rounds;
            ++width_hist[outcome.width];
            if(adaptive_width && (outcome.width==4 || outcome.width==8))
                pending_horizon_cost=PendingHorizonCost{
                    static_cast<unsigned>(outcome.width),
                    static_cast<unsigned>(outcome.committed_tokens.size()),
                    outcome.seed_ready};
            std::cout<<"FAST_DEVICE_REAL_DFLASH round="<<rounds
                     <<" width="<<outcome.width
                     <<" accepted="<<outcome.verification.accepted
                     <<" useful="<<outcome.committed_tokens.size()
                     <<" rejected="<<outcome.verification.rejected
                     <<" root="<<root_position<<'\n';
            continue;
        }
        if(remaining==1) {
            const auto token=exl3_branch_greedy(*live);
            if(adaptive_width)settle_horizon_cost(Clock::now());
            live->decode(token);
            const auto ring_start=Clock::now();
            for(int tap=0;tap<kTapCount;++tap)
                live->copy_tap_row_to_device(
                    kTapLayers[static_cast<std::size_t>(tap)],
                    stage.bulk_ptrs[static_cast<std::size_t>(tap)]);
            draft.commit_prefill_block(
                const_cast<const std::uint16_t**>(stage.bulk_ptrs.data()),
                1,root_position);
            ring_ms+=std::chrono::duration<double,std::milli>(
                Clock::now()-ring_start).count();
            d2d_bytes+=static_cast<std::size_t>(kTapCount)*kHidden*2;
            committed.push_back(token);
            ++replayed_rows;
            if(round_state_diagnostic)
                round_state_hashes.emplace_back(live->position(),
                    live->export_exact_host_state()->
                        represented_payload_hash_for_test());
            break;
        }
        std::array<std::int64_t,8> block{};
        block.fill(248070);
        auto revision=std::make_shared<const std::uint64_t>(rounds+1);
        ninfer::exl3::Exl3CommittedTapBinding binding{
            std::shared_ptr<const void>(live,live.get()),revision,
            root_position,1,1};
        const auto proposal_start=Clock::now();
        block[0]=exl3_branch_greedy(*live);
        constexpr std::uint64_t coherent_device_policy=0x434f484552454e54ULL;
        std::optional<ninfer::exl3::Exl3OuterDeviceSeedPacket> ready_seed;
        if(seed_reuse)ready_seed=ninfer::exl3::bind_exl3_outer_device_seed(
            *live,binding,block[0],coherent_device_policy);
        if(seed_reuse && rounds==0) {
            const auto matches=[&](const auto& candidate,const auto& owner,
                                   std::uint64_t policy) {
                return ninfer::exl3::exl3_outer_device_seed_matches(
                    *live,&owner,&candidate,policy,block[0]);
            };
            require(matches(*ready_seed,binding,coherent_device_policy),
                "current device seed packet rejected");
            auto stale=*ready_seed;
            stale.binding.root_revision=std::make_shared<const std::uint64_t>(99);
            require(!matches(stale,binding,coherent_device_policy),
                "stale device seed revision accepted");
            stale=*ready_seed;
            stale.request_generation^=1;
            require(!matches(stale,binding,coherent_device_policy),
                "stale device seed generation accepted");
            stale=*ready_seed;
            stale.token=(stale.token+1)%248320;
            require(!matches(stale,binding,coherent_device_policy),
                "mismatched device seed token accepted");
            auto other_request=binding;
            other_request.context_owner=std::make_shared<const int>(7);
            require(!matches(*ready_seed,other_request,coherent_device_policy) &&
                    !matches(*ready_seed,binding,coherent_device_policy+1),
                "cross-request or cross-policy device seed accepted");
        }
        const auto seed_ready=Clock::now();
        if(adaptive_width)settle_horizon_cost(seed_ready);
        const int width=std::min(adaptive_width?
            static_cast<int>(horizon_policy.select(remaining)):real_width,
            remaining);
        const auto draft_start=Clock::now();
        if(stage_timeline_diagnostic)
            target_seed_ms+=std::chrono::duration<double,std::milli>(
                draft_start-proposal_start).count();
        auto proposed=draft.propose_cached_view(
            std::span<const std::int64_t>(block).first(width),root_position,
            live->target_embedding(),live->target_lm_head_weights(),
            live->target_lm_head_metadata(),248070);
        if(stage_timeline_diagnostic)
            draft_api_ms+=std::chrono::duration<double,std::milli>(
                Clock::now()-draft_start).count();
        require(proposed.size()==static_cast<std::size_t>(width-1),
            "real device DFlash proposal extent");
        std::copy(proposed.begin(),proposed.end(),block.begin()+1);
        if(round_state_diagnostic && rounds==0)first_proposal=block;
        proposal_ms+=std::chrono::duration<double,std::milli>(
            Clock::now()-proposal_start).count();

        std::size_t staged_bytes=0;
        ninfer::exl3::Exl3CommittedTapConsumer consumer=[&](
            const ninfer::exl3::Exl3CommittedTapSegment& segment,
            cudaStream_t stream) {
            segment.validate();
            require(segment.owner==binding.context_owner &&
                    segment.root_revision==binding.root_revision &&
                    segment.root_position==root_position,
                "real device DFlash tap root binding");
            for(int tap=0;tap<kTapCount;++tap) {
                cuda_check(cudaMemcpyAsync(
                    stage.bulk_ptrs[tap]+
                        static_cast<std::size_t>(segment.destination_first)*kHidden,
                    segment.planes[tap]+
                        static_cast<std::size_t>(segment.source_first)*kHidden,
                    static_cast<std::size_t>(segment.rows)*kHidden*2,
                    cudaMemcpyDeviceToDevice,stream),
                    "real device DFlash committed tap D2D");
            }
            staged_bytes+=static_cast<std::size_t>(segment.rows)*
                kTapCount*kHidden*2;
        };
        const auto verifier_start=Clock::now();
        const auto fused_flash_before=ninfer::exl3::
            exl3_fast_fused_flash_multirow_attention_calls_for_test();
        if(projection_diagnostic && rounds==0) {
            live->begin_target_projection_timing_round(1);
            live->set_target_projection_timing_phase(
                ninfer::exl3::Exl3TargetProjectionPhase::attempt);
        }
        ninfer::exl3::Exl3OuterDeviceStageTimeline round_timeline{};
        auto verified=verify_exl3_outer_device_resident_reference(
            *live,std::span<const std::int64_t>(block).first(width),{},nullptr,
            &binding,&consumer,
            stage_timeline_diagnostic?&round_timeline:nullptr,
            ready_seed?&*ready_seed:nullptr,
            seed_reuse?coherent_device_policy:0);
        reused_seeds+=verified.device_seed_reused;
        seed_fallbacks+=verified.device_seed_fallback;
        fused_flash_multirow_calls+=ninfer::exl3::
            exl3_fast_fused_flash_multirow_attention_calls_for_test()-
            fused_flash_before;
        if(rounds==0)
            std::cout<<"FAST_DEVICE_REAL_DFLASH_FIRST_MULTIROW_CALLS calls="
                     <<fused_flash_multirow_calls<<'\n';
        if(stage_timeline_diagnostic) {
            verifier_stages.begin_ms+=round_timeline.begin_ms;
            verifier_stages.seed_ms+=round_timeline.seed_ms;
            verifier_stages.submit_ms+=round_timeline.submit_ms;
            verifier_stages.decision_ms+=round_timeline.decision_ms;
            verifier_stages.settlement_ms+=round_timeline.settlement_ms;
        }
        verifier_ms+=std::chrono::duration<double,std::milli>(
            Clock::now()-verifier_start).count();
        if(projection_diagnostic && rounds==0) {
            cuda_check(cudaStreamSynchronize(nullptr),
                "complete diagnostic real-device verifier projections");
            const auto records=
                live->finish_target_projection_timing_round_after_synchronize();
            using Key=std::tuple<int,int,int,int,int>;
            std::map<Key,std::pair<int,double>> groups;
            for(const auto& record:records) {
                const Key key{static_cast<int>(record.phase),
                    static_cast<int>(record.operation),record.K,
                    static_cast<int>(record.topology),record.rows};
                auto& group=groups[key];
                group.first+=record.calls;
                group.second+=record.microseconds;
            }
            std::cout<<"FAST_DEVICE_REAL_DFLASH_PROJECTION_DIAGNOSTIC"
                <<" round=1 records="<<records.size()
                <<" instrumented=1 throughput_comparison=0\n";
            for(const auto& [key,group]:groups) {
                const auto [phase,operation,bits,topology,rows]=key;
                std::cout<<"FAST_DEVICE_REAL_DFLASH_PROJECTION_GROUP"
                    <<" phase="<<ninfer::exl3::target_projection_phase_name(
                        static_cast<ninfer::exl3::Exl3TargetProjectionPhase>(phase))
                    <<" operation="<<ninfer::exl3::target_projection_operator_name(
                        static_cast<ninfer::exl3::Exl3TargetProjectionOperator>(operation))
                    <<" K="<<bits<<" topology="
                    <<ninfer::exl3::target_projection_topology_name(
                        static_cast<ninfer::exl3::Exl3TargetProjectionTopology>(topology))
                    <<" rows="<<rows<<" calls="<<group.first
                    <<" gpu_us="<<std::fixed<<std::setprecision(3)
                    <<group.second<<'\n';
            }
        }
        const int useful=static_cast<int>(verified.committed_tokens.size());
        require(useful>=1 && useful<=width &&
                staged_bytes==static_cast<std::size_t>(useful)*
                    kTapCount*kHidden*2 &&
                live->position()==root_position+useful,
            "real device DFlash accepted frontier/tap extent");
        const auto ring_start=Clock::now();
        draft.commit_prefill_block(
            const_cast<const std::uint16_t**>(stage.bulk_ptrs.data()),
            useful,root_position);
        committed.insert(committed.end(),verified.committed_tokens.begin(),
                         verified.committed_tokens.end());
        ring_ms+=std::chrono::duration<double,std::milli>(
            Clock::now()-ring_start).count();
        ++accepted_hist[verified.accepted];
        ++useful_hist[useful];
        verified_rows+=verified.verification_rows;
        replayed_rows+=verified.replay_rows;
        d2d_bytes+=staged_bytes;
        ++rounds;
        ++width_hist[width];
        if(adaptive_width && (width==4 || width==8))
            pending_horizon_cost=PendingHorizonCost{
                static_cast<unsigned>(width),static_cast<unsigned>(useful),
                seed_ready};
        if(round_state_diagnostic)
            round_state_hashes.emplace_back(live->position(),
                live->export_exact_host_state()->
                    represented_payload_hash_for_test());
        if(round_state_diagnostic && rounds==1)
            first_round_state=live->export_exact_host_state()->
                detached_payload_for_test();
        std::cout<<"FAST_DEVICE_REAL_DFLASH round="<<rounds
                 <<" width="<<width<<" accepted="<<verified.accepted
                 <<" useful="<<useful<<" rejected="<<verified.rejected
                 <<" root="<<root_position<<'\n';
    }
    if(reusable_round)reusable_round->finish();
    else cuda_check(cudaDeviceSynchronize(),
        "real device DFlash complete decode and ring work");
    if(adaptive_width)settle_horizon_cost(Clock::now());
    const auto decode_ms=std::chrono::duration<double,std::milli>(
        Clock::now()-decode_start).count();
    const auto request_to_decode_ms=std::chrono::duration<double,std::milli>(
        Clock::now()-live_request_start).count();
    const auto context_create_ms=std::chrono::duration<double,std::milli>(
        live_prefill_start-live_request_start).count();
    const auto live_prefill_ms=std::chrono::duration<double,std::milli>(
        live_prefill_end-live_prefill_start).count();
    const auto continuation_prepare_ms=std::chrono::duration<double,std::milli>(
        live_prepare_end-live_prefill_end).count();
    const auto live_final_hash=live->export_exact_host_state()->
        represented_payload_hash_for_test();
    const auto final_ring_digest=draft.ring_digest();
    std::cout<<"FAST_DEVICE_REAL_DFLASH_DRAFT_RING digest=";
    for(std::size_t i=0;i<final_ring_digest.size();++i) {
        if(i)std::cout<<',';
        std::cout<<final_ring_digest[i];
    }
    std::cout<<" dense_kmajor_launches="<<draft.dense_kmajor_launches()<<'\n';
    if(round_state_diagnostic) {
        std::cout<<"FAST_DEVICE_REAL_DFLASH_TOKENS count="<<committed.size()
                 <<" ids=";
        for(std::size_t i=0;i<committed.size();++i) {
            if(i)std::cout<<',';
            std::cout<<committed[i];
        }
        std::cout<<'\n';
    }
    std::size_t first_mismatch=0;
    while(first_mismatch<std::min(committed.size(),oracle.size()) &&
          committed[first_mismatch]==oracle[first_mismatch])
        ++first_mismatch;
    std::cout<<"FAST_DEVICE_REAL_DFLASH_FINAL committed="<<committed.size()
             <<" prefix="<<prefix_rows
             <<" oracle="<<oracle.size()
             <<" first_mismatch="<<first_mismatch
             <<" expected="<<(first_mismatch<oracle.size()?oracle[first_mismatch]:-1)
             <<" actual="<<(first_mismatch<committed.size()?committed[first_mismatch]:-1)
             <<" expected_hash="<<final_hash
             <<" live_hash="<<live_final_hash
             <<" live_position="<<live->position()
             <<" ring_base="<<draft.ring_base_abs()
             <<" ring_count="<<draft.ring_count()<<'\n';
    const bool valid=committed==oracle && live->position()==prefix_rows+output_rows &&
        draft.ring_base_abs()+draft.ring_count()==prefix_rows+output_rows &&
        live_final_hash==final_hash;
    const auto coherent_scalar_calls=coherent_calls_after_scalar-coherent_calls_before;
    const auto coherent_scalar_rows=coherent_rows_after_scalar-coherent_rows_before;
    const auto coherent_live_calls=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k6_calls_for_test()-
        coherent_calls_after_scalar;
    const auto coherent_live_rows=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k6_rows_for_test()-
        coherent_rows_after_scalar;
    std::cout<<"FAST_DEVICE_REAL_DFLASH_COHERENT_DOWN_K6 enabled="<<coherent_down_k6
             <<" scalar_calls="<<coherent_scalar_calls
             <<" scalar_rows="<<coherent_scalar_rows
             <<" live_calls="<<coherent_live_calls
             <<" live_rows="<<coherent_live_rows<<'\n';
    // Graph replay hides host-side dispatch counts, so require execution in
    // both contexts and multi-row verifier calls rather than count ordering.
    require(!coherent_down_k6 ||
            (coherent_scalar_calls>0 && coherent_live_calls>0 &&
             coherent_live_rows>coherent_live_calls),
        "coherent K6 down did not execute in both scalar and verifier contexts");
    const auto coherent_down_k7_scalar_calls=
        coherent_down_k7_calls_after_scalar-coherent_down_k7_calls_before;
    const auto coherent_down_k7_scalar_rows=
        coherent_down_k7_rows_after_scalar-coherent_down_k7_rows_before;
    const auto coherent_down_k7_live_calls=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k7_calls_for_test()-
        coherent_down_k7_calls_after_scalar;
    const auto coherent_down_k7_live_rows=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_down_k7_rows_for_test()-
        coherent_down_k7_rows_after_scalar;
    std::cout<<"FAST_DEVICE_REAL_DFLASH_COHERENT_DOWN_K7 enabled="<<coherent_down_k7
             <<" scalar_calls="<<coherent_down_k7_scalar_calls
             <<" scalar_rows="<<coherent_down_k7_scalar_rows
             <<" live_calls="<<coherent_down_k7_live_calls
             <<" live_rows="<<coherent_down_k7_live_rows<<'\n';
    require(coherent_down_k7 ?
            (coherent_down_k7_scalar_calls>0 && coherent_down_k7_live_calls>0 &&
             coherent_down_k7_live_rows>coherent_down_k7_live_calls) :
            (coherent_down_k7_scalar_calls==0 && coherent_down_k7_scalar_rows==0 &&
             coherent_down_k7_live_calls==0 && coherent_down_k7_live_rows==0),
        "coherent K7 down scalar/verifier dispatch mismatch");
    const auto coherent_o_scalar_calls=
        coherent_o_calls_after_scalar-coherent_o_calls_before;
    const auto coherent_o_scalar_rows=
        coherent_o_rows_after_scalar-coherent_o_rows_before;
    const auto coherent_o_live_calls=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_o_k7_calls_for_test()-
        coherent_o_calls_after_scalar;
    const auto coherent_o_live_rows=
        ninfer::exl3::Exl3CudaLinearWorkspace::process_coherent_o_k7_rows_for_test()-
        coherent_o_rows_after_scalar;
    std::cout<<"FAST_DEVICE_REAL_DFLASH_COHERENT_O_K7 enabled="<<coherent_o_k7
             <<" scalar_calls="<<coherent_o_scalar_calls
             <<" scalar_rows="<<coherent_o_scalar_rows
             <<" live_calls="<<coherent_o_live_calls
             <<" live_rows="<<coherent_o_live_rows<<'\n';
    require(!coherent_o_k7 ||
            (coherent_o_scalar_calls>0 && coherent_o_live_calls>0 &&
             coherent_o_live_rows>coherent_o_live_calls),
        "coherent K7 O did not execute in both scalar and verifier contexts");
    constexpr const char* coherent_wide_names[5]={
        "q","qkv","z","o","gate_up"};
    for(int operation=0;operation<5;++operation) {
        const auto scalar_calls=coherent_wide_calls_after_scalar[operation]-
            coherent_wide_calls_before[operation];
        const auto scalar_rows=coherent_wide_rows_after_scalar[operation]-
            coherent_wide_rows_before[operation];
        const auto live_calls=ninfer::exl3::Exl3CudaLinearWorkspace::
            coherent_wide_k6_calls_for_test(operation)-
            coherent_wide_calls_after_scalar[operation];
        const auto live_rows=ninfer::exl3::Exl3CudaLinearWorkspace::
            coherent_wide_k6_rows_for_test(operation)-
            coherent_wide_rows_after_scalar[operation];
        std::cout<<"FAST_DEVICE_REAL_DFLASH_COHERENT_WIDE_K6 op="
                 <<coherent_wide_names[operation]
                 <<" enabled="<<coherent_wide_k6
                 <<" scalar_calls="<<scalar_calls
                 <<" scalar_rows="<<scalar_rows
                 <<" live_calls="<<live_calls
                 <<" live_rows="<<live_rows<<'\n';
        require(coherent_wide_k6 ?
                scalar_calls>0 && scalar_rows>=scalar_calls &&
                live_calls>0 && live_rows>=live_calls :
                scalar_calls==0 && scalar_rows==0 &&
                live_calls==0 && live_rows==0,
            "coherent wide K6 scalar/verifier dispatch mismatch");
    }
    std::cout<<std::fixed<<std::setprecision(3)
             <<"FAST_DEVICE_REAL_DFLASH_NUMERIC_SCREEN valid="<<valid
             <<" tokens_equal="<<(committed==oracle)
             <<" state_equal="<<(live_final_hash==final_hash)
             <<" ring_frontier_equal="
             <<(draft.ring_base_abs()+draft.ring_count()==prefix_rows+output_rows)
             <<" decode_ms="<<decode_ms
             <<" useful_tok_s="<<(1000.0*output_rows/decode_ms)
             <<" fused_flash_multirow_calls="<<fused_flash_multirow_calls
             <<" unqualified_if_invalid=1\n";
    if(!valid && round_state_diagnostic) {
        live.reset();
        cuda_check(cudaDeviceSynchronize(),
            "release live context before diagnostic scalar replay");
        const auto first_k=[prefix_rows](const auto& state) {
            std::uint16_t bits=0;
            state.visit_kv_for_test([&](int layer,int first,int,
                std::span<const std::uint16_t> k,
                std::span<const std::uint16_t>) {
                if(layer==7 && first==prefix_rows)bits=k[0];
            });
            return bits;
        };
        auto attempt=target.create_context(true);
        ingest_prefix(*attempt,prefix,CommitSink{});
        attempt->prepare_continuation(8);
        attempt->prepare_transaction();
        attempt->begin_transaction();
        require(exl3_branch_greedy(*attempt)==first_proposal[0],
            "diagnostic first proposal seed changed");
        attempt->continue_rows(std::span<const std::int64_t>(first_proposal));
        attempt->commit_transaction();
        attempt->finish_exact_continuation();
        std::cout<<"FAST_DEVICE_REAL_DFLASH_ATTEMPT_FIRST_K bits="
                 <<first_k(*attempt->export_exact_host_state())
                 <<" repaired_bits="<<first_k(*first_round_state)<<'\n';
        attempt.reset();
        cuda_check(cudaDeviceSynchronize(),
            "release diagnostic full attempt before scalar replay");
        auto replay=target.create_context(true);
        ingest_prefix(*replay,prefix,CommitSink{});
        std::size_t boundary=0;
        for(std::size_t row=0;row<oracle.size();++row) {
            replay->decode(oracle[row]);
            if(row==0)
                std::cout<<"FAST_DEVICE_REAL_DFLASH_SCALAR_FIRST_K bits="
                    <<first_k(*replay->export_exact_host_state())<<'\n';
            if(boundary<round_state_hashes.size() &&
               replay->position()==round_state_hashes[boundary].first) {
                const auto scalar_hash=replay->export_exact_host_state()->
                    represented_payload_hash_for_test();
                std::cout<<"FAST_DEVICE_REAL_DFLASH_ROUND_STATE position="
                         <<replay->position()<<" scalar_hash="<<scalar_hash
                         <<" live_hash="<<round_state_hashes[boundary].second
                         <<" equal="
                         <<(scalar_hash==round_state_hashes[boundary].second)
                         <<'\n';
                if(boundary==0 && scalar_hash!=round_state_hashes[boundary].second)
                    std::cout<<"FAST_DEVICE_REAL_DFLASH_FIRST_DIFFERENCE "
                        <<replay->export_exact_host_state()->
                            represented_first_difference_for_test(*first_round_state)
                        <<'\n';
                ++boundary;
            }
        }
        require(boundary==round_state_hashes.size(),
            "real device DFlash diagnostic missed round frontier");
    }
    std::cout<<"FAST_DEVICE_REAL_DFLASH_MULTIROW_CALLS calls="
             <<fused_flash_multirow_calls<<'\n';
    require(valid,
        "real device DFlash differs from target-only stream/state or ring frontier");
    if(seed_reuse)require(reused_seeds==rounds && seed_fallbacks==0,
        "real device DFlash seed packet did not reach every verifier round");
    // Captured multirow full-layer graphs replay the fused multirow attention
    // without host-side dispatch counts; accept their replay as evidence.
    if(env("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_FUSED_FLASH_MULTIROW")=="1")
        require(fused_flash_multirow_calls>0 ||
                (env("NINFER_EXL3_ORDINARY_FULL_LAYER_MULTIROW_GRAPHS")=="1" &&
                 ninfer::exl3::Exl3TextContext::ordinary_graph_process_stats_for_test()
                     .full_layer_replays>0),
            "real device DFlash multirow fused attention did not dispatch");
    if(adaptive_width) {
        const auto b4=horizon_policy.snapshot(4);
        const auto b8=horizon_policy.snapshot(8);
        std::cout<<std::fixed<<std::setprecision(6)
            <<"FAST_DEVICE_REAL_DFLASH_ADAPTIVE_HORIZON"
            <<" b4_rounds="<<width_hist[4]
            <<" b8_rounds="<<width_hist[8]
            <<" b4_observations="<<b4.trials
            <<" b8_observations="<<b8.trials
            <<" b4_recent_useful_per_ms="<<b4.useful_per_ms
            <<" b8_recent_useful_per_ms="<<b8.useful_per_ms
            <<" b4_recent_complete_ms="<<b4.mean_complete_ms
            <<" b8_recent_complete_ms="<<b8.mean_complete_ms
            <<" policy_scope=request_local_no_fixture_identity\n";
    }
    if(stage_timeline_diagnostic)
        std::cout<<std::fixed<<std::setprecision(3)
            <<"FAST_DEVICE_REAL_DFLASH_STAGE_TIMELINE target_seed_ms="
            <<target_seed_ms<<" draft_api_ms="<<draft_api_ms
            <<" verifier_begin_ms="<<verifier_stages.begin_ms
            <<" verifier_seed_ms="<<verifier_stages.seed_ms
            <<" verifier_submit_ms="<<verifier_stages.submit_ms
            <<" verifier_decision_ms="<<verifier_stages.decision_ms
            <<" verifier_settlement_ms="<<verifier_stages.settlement_ms
            <<" host_intervals_include_async_waits=1\n";
    std::cout<<std::fixed<<std::setprecision(3)
        <<"FAST_DEVICE_REAL_DFLASH_REQUEST_WALL"
        <<" prefix="<<prefix_rows<<" committed="<<output_rows
        <<" context_create_ms="<<context_create_ms
        <<" target_draft_prefill_ms="<<live_prefill_ms
        <<" prefill_input_tok_s="<<(1000.0*prefix_rows/live_prefill_ms)
        <<" continuation_prepare_ms="<<continuation_prepare_ms
        <<" decode_ms="<<decode_ms
        <<" root_to_decode_ms="<<request_to_decode_ms
        <<" excludes_model_load_scalar_oracle_engine_publication_http=1\n";
    std::cout<<std::fixed<<std::setprecision(3)
        <<"FAST_DEVICE_REAL_DFLASH PASS committed="<<output_rows<<" prefix="<<prefix_rows
        <<" rounds="<<rounds
        <<" width_policy="<<real_width
        <<" adaptive_width="<<adaptive_width
        <<" verified_rows="<<verified_rows<<" replayed_rows="<<replayed_rows
        <<" d2d_bytes="<<d2d_bytes<<" decode_ms="<<decode_ms
        <<" reused_seeds="<<reused_seeds<<" seed_fallbacks="<<seed_fallbacks
        <<" useful_tok_s="<<(1000.0*output_rows/decode_ms)
        <<" proposal_ms="<<proposal_ms<<" verifier_ms="<<verifier_ms
        <<" ring_commit_ms="<<ring_ms
        <<" fused_flash_multirow_calls="<<fused_flash_multirow_calls
        <<" fused_topk_liveness_calls="<<draft.fused_topk_liveness_calls()
        <<" state_hash="<<final_hash
        <<" excludes_engine_publication=1\n";
}

// Real 4K frontier diagnostic with sequential oracle/context ownership so the
// two device-KV contexts do not compete for the remaining model VRAM. The
// proposals are ideal target-greedy tokens; draft and publication are absent.
void run_fast_device_long_resident(Exl3TextModel& target,
    const std::vector<std::int64_t>& ids) {
    require(target.max_context()>=4352 && ids.size()>=4096,
        "long device-resident verifier requires 4K fixture and context");
    const auto width_option=env("NINFER_EXL3_TEST_FAST_DEVICE_LONG_WIDTH");
    require(width_option.empty() || width_option=="4" || width_option=="8",
        "long device-resident verifier width must be 4 or 8");
    const int width=width_option=="8"?8:4;
    const int rounds=32/width;
    const std::vector<std::int64_t> prefix(ids.begin(),ids.begin()+4096);
    auto scalar=target.create_context(true);
    ingest_prefix(*scalar,prefix,CommitSink{});
    const auto root_hash=scalar->export_exact_host_state()->
        represented_payload_hash_for_test();
    std::array<std::int64_t,32> trajectory{};
    for(auto& token:trajectory) {
        token=exl3_branch_greedy(*scalar);
        scalar->decode(token);
    }
    const auto final_hash=scalar->export_exact_host_state()->
        represented_payload_hash_for_test();
    scalar.reset();
    cuda_check(cudaDeviceSynchronize(),
        "release 4K scalar device context before resident diagnostic");

    auto resident=target.create_context(true);
    ingest_prefix(*resident,prefix,CommitSink{});
    require(resident->export_exact_host_state()->
            represented_payload_hash_for_test()==root_hash,
        "long device-resident initial represented state differs");
    resident->prepare_continuation(8);
    resident->prepare_transaction();
    const auto graph_before=resident->device_transaction_checkpoint_graph_stats();
    const auto packed_before=Exl3CudaLinearWorkspace::
        process_fast_fp16_m2_8_down_calls_for_test();
    const auto fused_before=Exl3CudaLinearWorkspace::
        process_fast_fp16_m2_8_fused_down_calls_for_test();
    double verifier_ms=0.0;
    for(int round=0;round<rounds;++round) {
        const auto tentative=std::span<const std::int64_t>(trajectory).subspan(
            round*width,width);
        const auto started=std::chrono::steady_clock::now();
        auto live=verify_exl3_outer_device_resident_reference(*resident,tentative);
        cuda_check(cudaDeviceSynchronize(),
            "long device-resident verifier completion");
        const double ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count();
        verifier_ms+=ms;
        require(live.accepted==static_cast<std::size_t>(width) &&
                !live.rejected && !live.stopped &&
                live.committed_tokens.size()==static_cast<std::size_t>(width) &&
                !live.committed_state &&
                live.root_restores==0,
            "long device-resident full-accept round failed");
        std::cout<<"FAST_DEVICE_LONG_RESIDENT round="<<round<<
            " width="<<width<<" verifier_only_ms="<<ms<<'\n';
    }
    require(resident->position()==4128 && resident->export_exact_host_state()->
            represented_payload_hash_for_test()==final_hash,
        "long device-resident final state differs from scalar trajectory");
    const auto graph_after=resident->device_transaction_checkpoint_graph_stats();
    const auto packed_calls=Exl3CudaLinearWorkspace::
        process_fast_fp16_m2_8_down_calls_for_test()-packed_before;
    const auto fused_calls=Exl3CudaLinearWorkspace::
        process_fast_fp16_m2_8_fused_down_calls_for_test()-fused_before;
    if(env("NINFER_EXL3_FAST_FP16_M2_8")=="1" ||
       env("NINFER_EXL3_FAST_FP16_M2_8_DOWN_K6")=="1")
        require(packed_calls>0,
            "long device-resident packed M2-8 route did not dispatch");
    if(env("NINFER_EXL3_FAST_FP16_M2_8_FUSED_DOWN_K6")=="1")
        require(fused_calls>0,
            "long device-resident fused-output M2-8 down route did not dispatch");
    if(env("NINFER_EXL3_DEVICE_KV_TRANSACTION_CHECKPOINT_GRAPH")=="1") {
        require(graph_after.captures==graph_before.captures+1 &&
                graph_after.replays==graph_before.replays+rounds-1,
            "long device-resident checkpoint graph did not cover repeated frontiers");
        std::cout<<"FAST_DEVICE_CHECKPOINT_GRAPH PASS captures="<<
            graph_after.captures-graph_before.captures<<" repeated_replays="<<
            graph_after.replays-graph_before.replays<<
            " cold_capture_ms="<<graph_after.capture_ms-
                graph_before.capture_ms<<'\n';
    }
    std::cout<<"FAST_DEVICE_LONG_RESIDENT PASS committed=32 rounds="<<rounds<<
        " width="<<width<<
        " prefix=4096 verifier_only_ms="<<verifier_ms<<
        " verifier_only_tok_s="<<32000.0/verifier_ms<<
        " packed_m2_8_calls="<<packed_calls<<
        " fused_down_m2_8_calls="<<fused_calls<<
        " excludes_draft_proposal_tap_staging_publication=1"
        " export_excluded=1\n";
    std::cout<<"FAST_DEVICE_TRANSACTION PASS\n";
}

void run_fast_device_transaction(Exl3TextModel& target,
    const std::vector<std::int64_t>& ids) {
    if(env("NINFER_EXL3_TEST_FAST_DEVICE_LONG_RESIDENT")=="1") {
        run_fast_device_long_resident(target,ids);
        return;
    }
    require(target.max_context()>=256 && ids.size()>=256,
        "fast device transaction requires a real 256-token fixture");
    require(env("NINFER_EXL3_EXACT_HOST_KV")=="0" &&
            env("NINFER_OSCAR_EXL3")=="0" &&
            env("NINFER_EXL3_FAST_DEVICE_KV_TRANSACTION")=="1",
        "fast device transaction requires its guarded numerical/KV policy");
    auto scalar=target.create_context(true);
    auto batched=target.create_context(true);
    const bool resident_probe=env("NINFER_EXL3_TEST_FAST_DEVICE_RESIDENT")=="1";
    const bool deferred_verifier_probe=
        env("NINFER_EXL3_TEST_FAST_DEVICE_DEFERRED_VERIFIER")=="1";
    require(!deferred_verifier_probe || resident_probe,
        "deferred verifier gate needs the device-resident context");
    const bool checkpoint_graph_probe=
        env("NINFER_EXL3_DEVICE_KV_TRANSACTION_CHECKPOINT_GRAPH")=="1";
    require(!checkpoint_graph_probe || resident_probe,
        "device transaction checkpoint graph probe requires resident verifier");
    std::shared_ptr<Exl3TextContext> resident;
    if(resident_probe)resident=target.create_context(true);
    auto tap_staging=resident?
        std::make_unique<DeviceBuffer>(5ULL*8*5120*sizeof(std::uint16_t)):nullptr;
    ninfer::exl3::Exl3CommittedTapConsumer stage_committed_taps=[&](
        const ninfer::exl3::Exl3CommittedTapSegment& segment,cudaStream_t stream) {
        segment.validate();
        require(segment.destination_first>=0 &&
                segment.destination_first+segment.rows<=8,
            "device-resident tap staging extent");
        for(int tap=0;tap<5;++tap) {
            auto* destination=static_cast<std::uint16_t*>(tap_staging->get())+
                (tap*8+segment.destination_first)*5120;
            cuda_check(cudaMemcpyAsync(destination,
                segment.planes[tap]+segment.source_first*5120,
                segment.rows*5120*sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice,stream),
                "device-resident committed tap staging");
        }
    };
    const auto read_staged_taps=[&](std::size_t rows) {
        std::array<std::vector<std::uint16_t>,5> taps;
        for(int tap=0;tap<5;++tap) {
            taps[tap].resize(rows*5120);
            cuda_check(cudaMemcpy(taps[tap].data(),
                static_cast<const std::uint16_t*>(tap_staging->get())+tap*8*5120,
                rows*5120*sizeof(std::uint16_t),cudaMemcpyDeviceToHost),
                "read device-resident committed taps outside timing");
        }
        return taps;
    };
    const std::vector<std::int64_t> prefix(ids.begin(),ids.begin()+128);
    ingest_prefix(*scalar,prefix,CommitSink{});
    ingest_prefix(*batched,prefix,CommitSink{});
    if(resident)ingest_prefix(*resident,prefix,CommitSink{});
    batched->prepare_continuation(8);
    batched->prepare_transaction();
    if(resident) {
        resident->prepare_continuation(8);
        resident->prepare_transaction();
    }
    auto root=scalar->export_exact_host_state();
    require(root->represented_payload_hash_for_test()==
            batched->export_exact_host_state()->represented_payload_hash_for_test(),
        "fast device transaction paired root differs");
    if(resident)
        require(root->represented_payload_hash_for_test()==
                resident->export_exact_host_state()->represented_payload_hash_for_test(),
            "device-resident transaction paired root differs");
    std::array<std::int64_t,4> predicted{};
    for(auto& token:predicted) {
        token=exl3_branch_greedy(*scalar);
        scalar->decode(token);
    }
    const std::array<std::int64_t,1> stop_token{predicted[1]};
    if(deferred_verifier_probe) {
        for(int scenario=0;scenario<4;++scenario) {
            auto tentative=predicted;
            if(scenario)tentative[scenario-1]=
                (tentative[scenario-1]+1)%248320;
            auto reference=verify_exl3_outer_reference(
                *scalar,*root,tentative,true);
            require(reference.accepted==
                    static_cast<std::size_t>(scenario?scenario-1:4) &&
                    reference.rejected==static_cast<bool>(scenario),
                "deferred verifier fixture did not reach intended rejection");
            resident->restore_exact_host_state(*root);
            auto eager=verify_exl3_outer_device_resident_reference(
                *resident,tentative);
            auto eager_state=resident->export_exact_host_state();
            std::cout<<"FAST_DEVICE_DEFERRED_VERIFIER eager scenario="
                <<scenario<<" hash="<<eager_state->represented_payload_hash_for_test()
                <<" reference_hash="<<reference.committed_state->represented_payload_hash_for_test()
                <<" accepted="<<eager.accepted<<std::endl;
            for(int trial=0;trial<2;++trial) {
                resident->restore_exact_host_state(*root);
                auto pending=verify_exl3_outer_device_resident_reference(
                    *resident,tentative,{},nullptr,nullptr,nullptr,nullptr,
                    nullptr,0,ninfer::exl3::Exl3OuterDeviceSettlement::
                        DeferTargetCommit);
                require(resident->transaction_active() &&
                    pending.committed_tokens==reference.committed_tokens &&
                    pending.accepted==reference.accepted &&
                    pending.rejected==reference.rejected &&
                    resident->position()==root->position()+
                        static_cast<int>(pending.committed_tokens.size()),
                    "deferred verifier candidate differs from scalar decision");
                if(!trial) {
                    resident->rollback_transaction();
                    require(!resident->transaction_active() &&
                        resident->export_exact_host_state()->
                            represented_payload_hash_for_test()==
                        root->represented_payload_hash_for_test(),
                        "deferred verifier rollback changed root state");
                } else {
                    resident->commit_transaction();
                    if(resident->continuation_rows()>0)
                        resident->finish_exact_continuation();
                    const auto settled=resident->export_exact_host_state();
                    std::cout<<"FAST_DEVICE_DEFERRED_VERIFIER settlement scenario="
                        <<scenario<<" accepted="<<pending.accepted
                        <<" rejected="<<pending.rejected
                        <<" continuation_rows="<<resident->continuation_rows()
                        <<" actual_hash="<<settled->represented_payload_hash_for_test()
                        <<" reference_hash="<<reference.committed_state->represented_payload_hash_for_test()
                        <<std::endl;
                    require(settled->represented_payload_hash_for_test()==
                        reference.committed_state->represented_payload_hash_for_test(),
                        "deferred verifier committed state differs from scalar");
                }
            }
            std::cout<<"FAST_DEVICE_DEFERRED_VERIFIER scenario="<<scenario
                     <<" accepted="<<reference.accepted
                     <<" rejected="<<reference.rejected<<" PASS\n";
        }
        std::cout<<"FAST_DEVICE_DEFERRED_VERIFIER PASS scenarios=4 trials=8\n";
        return;
    }
    for(int scenario=0;scenario<(resident?4:3);++scenario) {
        auto tentative=predicted;
        if(scenario==1)tentative[0]=(tentative[0]+1)%248320;
        if(scenario==2)tentative[2]=(tentative[2]+1)%248320;
        const auto terminal=scenario==3?std::span<const std::int64_t>(stop_token):
            std::span<const std::int64_t>{};
        const auto scalar_start=std::chrono::steady_clock::now();
        auto reference=verify_exl3_outer_reference(*scalar,*root,tentative,true,terminal);
        const auto scalar_end=std::chrono::steady_clock::now();
        const auto down_before=Exl3CudaLinearWorkspace::
            process_fast_fp16_m2_8_down_calls_for_test();
        const auto fused_before=Exl3CudaLinearWorkspace::
            process_fast_fp16_m2_8_fused_down_calls_for_test();
        auto result=verify_exl3_outer_checkpointed_reference(
            *batched,*root,tentative,false,terminal);
        const auto batched_end=std::chrono::steady_clock::now();
        const auto down_calls=Exl3CudaLinearWorkspace::
            process_fast_fp16_m2_8_down_calls_for_test()-down_before;
        const auto fused_calls=Exl3CudaLinearWorkspace::
            process_fast_fp16_m2_8_fused_down_calls_for_test()-fused_before;
        std::cout<<"FAST_DEVICE_TRANSACTION scenario="<<scenario<<
            " scalar_accepted="<<reference.accepted<<
            " batched_accepted="<<result.accepted<<
            " small_m_down_calls="<<down_calls<<
            " fused_small_m_down_calls="<<fused_calls<<
            " scalar_ms="<<std::chrono::duration<double,std::milli>(
                scalar_end-scalar_start).count()<<
            " batched_ms="<<std::chrono::duration<double,std::milli>(
                batched_end-scalar_end).count()<<
            " scalar_hash="<<reference.committed_state->represented_payload_hash_for_test()<<
            " batched_hash="<<result.committed_state->represented_payload_hash_for_test()<<'\n';
        if(env("NINFER_EXL3_FAST_FP16_M2_8_DOWN_K6")=="1" ||
           env("NINFER_EXL3_FAST_FP16_M2_8")=="1")
            require(down_calls>0,"packed FP16 M2-8 down did not dispatch");
        if(env("NINFER_EXL3_FAST_FP16_M2_8_FUSED_DOWN_K6")=="1")
            require(fused_calls>0,"fused-output packed FP16 M2-8 down did not dispatch");
        require(reference.committed_tokens==result.committed_tokens &&
                reference.accepted==result.accepted &&
                reference.rejected==result.rejected &&
                reference.stopped==result.stopped,
            "fast device transaction decisions differ from scalar same-policy verifier");
        require(reference.committed_state->represented_payload_hash_for_test()==
                result.committed_state->represented_payload_hash_for_test(),
            "fast device transaction committed state differs from scalar same-policy verifier");
        if(scenario==3)
            require(reference.stopped && !reference.rejected &&
                    reference.committed_tokens.size()==2,
                "fast device terminal scenario did not stop after two rows");
        if(resident) {
            resident->restore_exact_host_state(*root);
            const ninfer::exl3::Exl3CommittedTapBinding binding{
                std::static_pointer_cast<const void>(resident),
                std::static_pointer_cast<const void>(root),root->position(),
                1,static_cast<std::uint64_t>(scenario+1)};
            const auto resident_start=std::chrono::steady_clock::now();
            auto live=verify_exl3_outer_device_resident_reference(
                *resident,tentative,terminal,nullptr,&binding,&stage_committed_taps);
            cuda_check(cudaDeviceSynchronize(),
                "device-resident verifier completion timing");
            const auto resident_end=std::chrono::steady_clock::now();
            auto resident_state=resident->export_exact_host_state();
            auto resident_taps=read_staged_taps(live.committed_tokens.size());
            std::cout<<"FAST_DEVICE_RESIDENT scenario="<<scenario<<
                " accepted="<<live.accepted<<
                " resident_ms="<<std::chrono::duration<double,std::milli>(
                    resident_end-resident_start).count()<<
                " export_excluded=1 completion_synchronized=1 root_restores="<<live.root_restores<<
                " checkpoint_bytes="<<live.checkpoint_captured_bytes<<
                " state_hash="<<resident_state->represented_payload_hash_for_test()<<'\n';
            require(!live.committed_state && live.root_restores==0 &&
                    live.committed_tokens==reference.committed_tokens &&
                    live.accepted==reference.accepted &&
                    live.rejected==reference.rejected &&
                    live.stopped==reference.stopped,
                "device-resident transaction decisions differ from scalar same-policy verifier");
            require(resident_state->represented_payload_hash_for_test()==
                    reference.committed_state->represented_payload_hash_for_test() &&
                    resident_taps==reference.committed_taps &&
                    resident->position()==root->position()+
                        static_cast<int>(live.committed_tokens.size()),
                "device-resident transaction committed state or taps differ from scalar same-policy verifier");
            if(scenario==3) {
                std::array<std::int64_t,4> next{};
                for(auto& token:next) {
                    token=exl3_branch_greedy(*scalar);
                    scalar->decode(token);
                }
                next[1]=(next[1]+1)%248320;
                auto chained_reference=verify_exl3_outer_reference(
                    *scalar,*reference.committed_state,next,true);
                require(chained_reference.accepted==1 && chained_reference.rejected,
                    "fast device chained scenario did not exercise prefix repair");
                const ninfer::exl3::Exl3CommittedTapBinding chained_binding{
                    std::static_pointer_cast<const void>(resident),
                    std::static_pointer_cast<const void>(reference.committed_state),
                    reference.committed_state->position(),1,5};
                const auto chained_start=std::chrono::steady_clock::now();
                auto chained=verify_exl3_outer_device_resident_reference(
                    *resident,next,{},nullptr,&chained_binding,&stage_committed_taps);
                cuda_check(cudaDeviceSynchronize(),
                    "device-resident chained verifier completion timing");
                const auto chained_end=std::chrono::steady_clock::now();
                auto chained_state=resident->export_exact_host_state();
                auto chained_taps=read_staged_taps(chained.committed_tokens.size());
                std::cout<<"FAST_DEVICE_RESIDENT chained_accepted="<<chained.accepted<<
                    " resident_ms="<<std::chrono::duration<double,std::milli>(
                        chained_end-chained_start).count()<<
                    " export_excluded=1 completion_synchronized=1 root_restores="<<chained.root_restores<<
                    " state_hash="<<chained_state->represented_payload_hash_for_test()<<'\n';
                require(!chained.committed_state && chained.root_restores==0 &&
                        chained.committed_tokens==chained_reference.committed_tokens &&
                        chained.accepted==chained_reference.accepted &&
                        chained.rejected==chained_reference.rejected &&
                        chained.stopped==chained_reference.stopped &&
                        chained_taps==chained_reference.committed_taps &&
                        chained_state->represented_payload_hash_for_test()==
                            chained_reference.committed_state->represented_payload_hash_for_test(),
                    "device-resident chained transaction differs from scalar same-policy verifier");
            }
        }
    }
    if(resident) {
        scalar->restore_exact_host_state(*root);
        std::array<std::int64_t,32> trajectory{};
        for(auto& token:trajectory) {
            token=exl3_branch_greedy(*scalar);
            scalar->decode(token);
        }
        auto trajectory_state=scalar->export_exact_host_state();
        resident->restore_exact_host_state(*root);
        cuda_check(cudaDeviceSynchronize(),
            "device-resident repeated-round initial completion");
        const auto graph_before=
            resident->device_transaction_checkpoint_graph_stats();
        double verifier_ms=0.0;
        for(int round=0;round<8;++round) {
            const auto tentative=std::span<const std::int64_t>(trajectory).subspan(
                round*4,4);
            const auto begin=std::chrono::steady_clock::now();
            auto live=verify_exl3_outer_device_resident_reference(
                *resident,tentative);
            cuda_check(cudaDeviceSynchronize(),
                "device-resident repeated verifier completion");
            const auto end=std::chrono::steady_clock::now();
            const double ms=std::chrono::duration<double,std::milli>(
                end-begin).count();
            verifier_ms+=ms;
            require(live.accepted==4 && !live.rejected && !live.stopped &&
                    live.committed_tokens.size()==4 && !live.committed_state &&
                    live.root_restores==0,
                "device-resident repeated full-accept round failed");
            std::cout<<"FAST_DEVICE_RESIDENT_REPEAT round="<<round<<
                " committed=4 verifier_only_ms="<<ms<<'\n';
        }
        auto repeated_state=resident->export_exact_host_state();
        require(repeated_state->represented_payload_hash_for_test()==
                trajectory_state->represented_payload_hash_for_test() &&
                resident->position()==root->position()+32,
            "device-resident repeated rounds differ from scalar same-policy trajectory");
        const auto graph_after=
            resident->device_transaction_checkpoint_graph_stats();
        if(checkpoint_graph_probe) {
            std::cout<<"FAST_DEVICE_CHECKPOINT_GRAPH_STATS captures_before="<<
                graph_before.captures<<" captures_after="<<graph_after.captures<<
                " replays_before="<<graph_before.replays<<
                " replays_after="<<graph_after.replays<<
                " capture_ms="<<graph_after.capture_ms<<'\n';
            // The restored 128-row root takes the eager first checkpoint;
            // subsequent B4 frontiers must reuse the captured B2..B8 graph.
            require(graph_after.captures>0 && graph_after.capture_ms>0.0 &&
                    graph_after.replays==graph_before.replays+7,
                "device-resident checkpoint graph did not replay each B4 frontier");
            std::cout<<"FAST_DEVICE_CHECKPOINT_GRAPH PASS captures="<<
                graph_after.captures<<" repeated_replays="<<
                graph_after.replays-graph_before.replays<<
                " cold_capture_ms="<<graph_after.capture_ms<<'\n';
        }
        std::cout<<"FAST_DEVICE_RESIDENT_REPEAT PASS committed=32 rounds=8"
            " verifier_only_ms="<<verifier_ms<<
            " verifier_only_tok_s="<<32000.0/verifier_ms<<
            " excludes_draft_proposal_and_publication=1"
            " excludes_draft_tap_staging=1 export_excluded=1\n";
    }
    if(resident)std::cout<<"FAST_DEVICE_RESIDENT PASS\n";
    std::cout<<"FAST_DEVICE_TRANSACTION PASS\n";
}
