#pragma once

void run_transaction_retention_qualification(
    Exl3TextModel& target, Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& source_ids, bool qkv_focused = false, bool fresh_prefill = false) {
    require(target.max_context() == (fresh_prefill ? 8192 : 4096) && source_ids.size() >= (fresh_prefill ? 4097u : 4096u),
            "transactionretainqual requires maxctx4096 and real prompt IDs");
    require(env("NINFER_OSCAR_EXL3") == "1",
            "transactionretainqual requires canonical OSCAR");
    if(qkv_focused){
        const char* qkv_k6 = std::getenv("NINFER_EXL3_TARGET_QKV_K6_SMALL_M");
        require(qkv_k6 == nullptr || std::string(qkv_k6) == "1",
            "targetqkvk6state requires flag unset or 1");
    }
    struct ScopedQkvEnvironment {
        std::vector<std::pair<std::string,std::string>> saved;
        ScopedQkvEnvironment(std::initializer_list<std::pair<const char*,const char*>> settings) {
            for(const auto& setting:settings)saved.emplace_back(setting.first,env(setting.first));
            for(const auto& setting:settings)_putenv_s(setting.first,setting.second);
        }
        ~ScopedQkvEnvironment(){for(auto i=saved.rbegin();i!=saved.rend();++i)_putenv_s(i->first.c_str(),i->second.c_str());}
    };
    if(fresh_prefill)require(env("NINFER_DFLASH2_PREFILL_WINDOW")=="1" && env("NINFER_EXL3_TARGET_QKV_K6_SMALL_M")=="1","fresh state requires P9 and fresh scope flags");
    const bool reduced_reference = qkv_focused || fresh_prefill;
    auto snapshot=[&](Exl3TextContext& ctx){return prefix_retention_snapshot(ctx,fresh_prefill?8192:4096);};
    int cases = 0, retain_cases = 0, default_cases = 0, failures = 0;
    auto run_case = [&](int prefix_rows, int block, int rejection, bool retain) {
        const std::string label = "transactionretainqual ctx=" + std::to_string(prefix_rows) +
            " B=" + std::to_string(block) + " rejection=" + std::to_string(rejection) +
            " route=" + (retain ? "retain" : "default");
        const std::vector<std::int64_t> prefix(source_ids.begin(), source_ids.begin() + prefix_rows);
        auto candidate = target.create_context(true);
        auto reference = [&] {
            if(!reduced_reference){
                ScopedQkvEnvironment restore{{"NINFER_EXL3_GDN_WIDE_SLAB","0"}};
                return target.create_context(true);
            }
            // Established exact32 serial oracle: two expanded1024 contexts do not fit.
            ScopedQkvEnvironment restore{{"NINFER_EXL3_PREFILL_WIDE64","0"},
                {"NINFER_EXL3_PREFILL_ROWPAIR_K6","0"},{"NINFER_EXL3_SHARED_LAYER_SCRATCH","0"},{"NINFER_EXL3_PREFILL_DIRECT_ASYNC_A","0"},
                {"NINFER_EXL3_PREFILL_DIRECT_PARTIALS","0"},{"NINFER_EXL3_PREFILL_STAGED_SHAPE4","0"},
                {"NINFER_EXL3_TARGET_QKV_K6_SMALL_M",fresh_prefill?"1":"0"},
                {"NINFER_EXL3_GDN_WIDE_SLAB","0"}};
            return target.create_context(true);
        }();
        if(reduced_reference){
            require(candidate->try_enable_oscar_from_environment(),label+" candidate OSCAR enable");
            ScopedQkvEnvironment restore{{"NINFER_OSCAR_PREFILL_PACKED_BYTE4","0"},
                {"NINFER_OSCAR_PREFILL_QUERY_PARALLEL","0"},
                {"NINFER_OSCAR_PREFILL_PARALLEL_MERGE","0"},{"NINFER_OSCAR_PREFILL_PARALLEL_SCORES","0"},
                {"NINFER_OSCAR_PREFILL_COALESCED_ROTATIONS","0"}};
            require(reference->try_enable_oscar_from_environment(),label+" serial OSCAR enable");
        }else{
            require(candidate->try_enable_oscar_from_environment() &&
                    reference->try_enable_oscar_from_environment(), label + " OSCAR enable");
        }
        candidate->prepare_transaction(); candidate->prepare_continuation(block);
        reference->prepare_transaction();
        draft.reset();
        auto staging = make_transaction_round_stage();
        ingest_prefix(*candidate, prefix, CommitSink{&draft, &staging});
        if(reduced_reference){
            {ScopedQkvEnvironment restore{{"NINFER_EXL3_PREFILL_CHUNK","32"}};
             ingest_prefix(*reference,prefix,CommitSink{});}
            require_prefix_retention_semantic(snapshot(*candidate),snapshot(*reference),label+" exact32 base state");
            for(const auto layer:kTapLayers){
                DeviceBuffer a(kHidden*2),b(kHidden*2);
                candidate->copy_tap_row_to_device(layer,static_cast<std::uint16_t*>(a.get()));
                reference->copy_tap_row_to_device(layer,static_cast<std::uint16_t*>(b.get()));
                require(target_continue_device_bits(static_cast<const std::uint16_t*>(a.get()),kHidden,"qkv candidate latest tap")==target_continue_device_bits(static_cast<const std::uint16_t*>(b.get()),kHidden,"qkv reference latest tap"),label+" latest base tap");
            }
        }else{
            ingest_prefix(*reference, prefix, CommitSink{});
            require(snapshot(*candidate) == snapshot(*reference),
                label + " base state mismatch");
        }
        auto pending = sample_target(*candidate);
        require(pending == sample_target(*reference), label + " initial pending mismatch");
        std::vector<std::int64_t> emitted{pending}, processed;
        std::array<std::array<std::uint64_t, 5>, 2> round_digests{};
        std::array<std::size_t, 2> processed_cuts{};
        int attempted_total = 0, replay_total = 0, reconstruction_total = 0;
        for (int round = 0; round < 2; ++round) {
            // Produce the exact greedy future without changing the reference base.
            reference->begin_transaction();
            reference->decode(pending);
            std::vector<std::int64_t> future;
            for (int row = 0; row < block; ++row) {
                future.push_back(sample_target(*reference));
                if (row + 1 < block) reference->decode(future.back());
            }
            reference->rollback_transaction();
            std::vector<std::int64_t> proposals(future.begin(), future.end() - 1);
            if (rejection >= 0)
                proposals[static_cast<std::size_t>(rejection)] =
                    (proposals[static_cast<std::size_t>(rejection)] + 1) % kVocab;
            const int position = candidate->position();
            if (cases == 0 && round == 0) {
                const auto before = snapshot(*candidate);
                const auto ring_before = draft.ring_digest();
                const int count_before = draft.ring_count();
                const auto base_before = draft.ring_base_abs();
                for (const int layer : {0, 31, 63}) {
                    bool failed = false;
                    try {
                        (void)verify_pending_round_transactional(*candidate, draft, staging,
                            pending, proposals, position, nullptr, -1, 0x7e00u,
                            TransactionRepairMode::retain_prefix, layer);
                    } catch (const std::runtime_error& error) {
                        require(std::string(error.what()) ==
                            "P2 injected retained-prefix failure after model layer " + std::to_string(layer),
                            label + " unexpected injected failure: " + error.what());
                        failed = true;
                    }
                    require(failed && !candidate->transaction_active() &&
                                snapshot(*candidate) == before,
                            label + " caller failure did not restore original state/payload");
                    require(draft.ring_digest() == ring_before && draft.ring_count() == count_before &&
                                draft.ring_base_abs() == base_before,
                            label + " injected reconstruction published ring");
                    ++failures;
                    std::cout << "TRANSACTIONRETAIN_FAILURE PASS after_layer=" << layer
                              << " target_rollback=exact ring=unchanged" << std::endl;
                }
                bool nonfinite_rejected = false;
                try {
                    (void)verify_pending_round_transactional(*candidate, draft, staging,
                        pending, proposals, position, nullptr, 0, 0x7e00u,
                        TransactionRepairMode::retain_prefix);
                } catch (const std::runtime_error& error) {
                    nonfinite_rejected = std::string(error.what()) ==
                        "transaction verifier nonfinite logit row=0 token=0";
                }
                require(nonfinite_rejected && !candidate->transaction_active() &&
                            snapshot(*candidate) == before &&
                            draft.ring_digest() == ring_before && draft.ring_count() == count_before &&
                            draft.ring_base_abs() == base_before,
                        label + " nonfinite attempt was published or failed rollback");
            }
                if(reduced_reference) {
                const auto ring_probe=draft.ring_digest();const auto count_probe=draft.ring_count();const auto base_probe=draft.ring_base_abs();
                std::vector<std::int64_t> attempted{pending};attempted.insert(attempted.end(),proposals.begin(),proposals.end());
                candidate->begin_transaction();reference->begin_transaction();
                candidate->continue_rows(attempted);
                const auto actual_logits=candidate->continuation_logits_bits_host();
                const auto actual_embeddings=candidate->embedding_bits_host_for_test();
                std::array<std::vector<std::uint16_t>,5> actual_taps,reference_taps;
                for(std::size_t tap=0;tap<kTapLayers.size();++tap)actual_taps[tap]=target_continue_tap_bits(*candidate,kTapLayers[tap],block);
                std::vector<std::uint16_t> reference_logits,reference_embeddings;
                for(const auto token:attempted){
                    reference->decode(token);
                    const auto logits=target_continue_device_bits(reference->logits_device(),kVocab,"qkv attempted reference logits");reference_logits.insert(reference_logits.end(),logits.begin(),logits.end());
                    const auto embedding=reference->embedding_bits_host_for_test();reference_embeddings.insert(reference_embeddings.end(),embedding.begin(),embedding.end());
                    for(std::size_t tap=0;tap<kTapLayers.size();++tap){const auto values=target_continue_tap_bits(*reference,kTapLayers[tap],1);reference_taps[tap].insert(reference_taps[tap].end(),values.begin(),values.end());}
                }
                require_target_continue_exact(actual_logits,reference_logits,kVocab,label+" all attempted logits");
                require_target_continue_exact(actual_embeddings,reference_embeddings,kHidden,label+" all attempted embeddings");
                for(std::size_t tap=0;tap<kTapLayers.size();++tap)require_target_continue_exact(actual_taps[tap],reference_taps[tap],kHidden,label+" all attempted taps");
                candidate->rollback_transaction();reference->rollback_transaction();
                require(candidate->position()==position && reference->position()==position && sample_target(*candidate)==pending && sample_target(*reference)==pending && !candidate->transaction_active() && !reference->transaction_active(),label+" probe rollback");
                require(draft.ring_digest()==ring_probe && draft.ring_count()==count_probe && draft.ring_base_abs()==base_probe,label+" probe ring unchanged");
            }
            // Omit the repair selector for the explicit default-compatibility cases.
            const auto result = retain
                ? verify_pending_round_transactional(*candidate, draft, staging, pending,
                    proposals, position, nullptr, -1, 0x7e00u, TransactionRepairMode::retain_prefix)
                : verify_pending_round_transactional(*candidate, draft, staging, pending,
                    proposals, position);
            const int accepted = rejection < 0 ? block - 1 : rejection;
            const int retained = accepted + 1;
            require(result.accepted == accepted && result.rejection_index == rejection &&
                        result.attempted_rows == block && result.retained_rows == retained &&
                        result.replay_rows == (!retain && rejection >= 0 ? retained : 0) &&
                        result.state_reconstruction_rows == (retain && rejection >= 0 ? retained : 0) &&
                        !candidate->transaction_active(), label + " result/accounting mismatch");
            std::vector<std::int64_t> expected_emitted(future.begin(), future.begin() + retained);
            require(result.emitted == expected_emitted && result.pending == future[accepted],
                    label + " emitted/pending mismatch");
            std::vector<std::uint16_t> expected_logits, expected_embedding;
            std::array<std::vector<std::uint16_t>, 5> expected_taps;
            for (int row = 0; row < retained; ++row) {
                const auto token = row == 0 ? pending : future[static_cast<std::size_t>(row - 1)];
                reference->decode(token); processed.push_back(token);
                const auto logits = target_continue_device_bits(reference->logits_device(), kVocab,
                                                                 "transactionretainqual reference logits");
                expected_logits.insert(expected_logits.end(), logits.begin(), logits.end());
                const auto embedding = reference->embedding_bits_host_for_test();
                expected_embedding.insert(expected_embedding.end(), embedding.begin(), embedding.end());
                for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
                    const auto values = target_continue_tap_bits(*reference, kTapLayers[tap], 1);
                    expected_taps[tap].insert(expected_taps[tap].end(), values.begin(), values.end());
                }
            }
            const auto actual = snapshot(*candidate);
            const auto expected = snapshot(*reference);
            require_prefix_retention_semantic(actual, expected, label + " round state");
            require_transaction_round_state_equal(*candidate, *reference, label + " final tap/state");
            if (retain || rejection < 0) {
                require(actual.embedding == expected_embedding && actual.taps == expected_taps &&
                            actual.continuation_logits == expected_logits,
                        label + " retained all-row payload mismatch");
            } else {
                require(actual.embedding == expected.embedding && actual.taps == expected.taps,
                        label + " default replay payload mismatch");
            }
            for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
                const auto staged = transaction_round_device_bits(staging.bulk_ptrs[tap],
                    static_cast<std::size_t>(retained) * kHidden, "transactionretainqual staged taps");
                require(staged == expected_taps[tap], label + " published staging includes rejected/wrong taps");
            }
            emitted.insert(emitted.end(), result.emitted.begin(), result.emitted.end());
            require(processed.size() + 1 == emitted.size() &&
                        std::equal(processed.begin(), processed.end(), emitted.begin()),
                    label + " processed/emitted ledger mismatch");
            pending = result.pending;
            attempted_total += result.attempted_rows;
            replay_total += result.replay_rows;
            reconstruction_total += result.state_reconstruction_rows;
            const auto end = static_cast<long long>(prefix_rows + processed.size());
            const auto count = std::min<long long>(end, Exl3Dflash2DraftModel::ring_keep());
            require(draft.ring_count() == count && draft.ring_base_abs() == end - count,
                    label + " ring cursor mismatch");
            round_digests[round] = draft.ring_digest(); processed_cuts[round] = processed.size();
        }
        std::uint64_t oscar = 0, ordinary = 0;
        candidate->oscar_routing_counts(oscar, ordinary);
        require(oscar > 0 && ordinary == 0 && candidate->oscar_telemetry().gdn_oscar_dispatches == 0,
                label + " escaped canonical OSCAR");
        for (int step = 0; step < 4; ++step) {
            candidate->decode(pending); reference->decode(pending);
            require(snapshot(*candidate) == snapshot(*reference),
                    label + " following M1 state/payload mismatch");
            pending = sample_target(*reference);
            require(pending == sample_target(*candidate), label + " following greedy mismatch");
        }
        candidate.reset(); reference.reset();
        // Independent ordinary M1 publication rebuild checks BOTH round digests.
        draft.reset();
        auto rebuild = target.create_context(true);
        require(rebuild->try_enable_oscar_from_environment(), label + " rebuild OSCAR");
        auto rebuild_staging = make_transaction_round_stage();
        {ScopedQkvEnvironment original{{"NINFER_DFLASH2_PREFILL_WINDOW","0"}};
         ingest_prefix(*rebuild, prefix, CommitSink{&draft, &rebuild_staging});}
        int cut = 0;
        for (std::size_t index = 0; index < processed.size(); ++index) {
            const int position = rebuild->position();
            rebuild->decode(processed[index]);
            commit_latest_row(*rebuild, draft, rebuild_staging, position);
            if (cut < 2 && index + 1 == processed_cuts[cut]) {
                const auto end = static_cast<long long>(prefix_rows + index + 1);
                const auto count = std::min<long long>(end, Exl3Dflash2DraftModel::ring_keep());
                require(draft.ring_digest() == round_digests[cut] && draft.ring_count() == count &&
                            draft.ring_base_abs() == end - count, label + " independently rebuilt ring mismatch");
                ++cut;
            }
        }
        require(cut == 2, label + " missing ring comparison");
        ++cases;
        if (retain) ++retain_cases; else ++default_cases;
        std::cout << "TRANSACTIONRETAIN_CASE PASS ctx=" << prefix_rows << " B=" << block
                  << " rejection=" << rejection << " route=" << (retain ? "retain" : "default")
                  << " rounds=2 continuation=4 attempted=" << attempted_total
                  << " replay=" << replay_total << " reconstruction=" << reconstruction_total
                  << " retained=" << processed.size() << " ring_count=" << draft.ring_count()
                  << " ring_base=" << draft.ring_base_abs() << " ring_digests=2 state=exact payload=exact"
                  << std::endl;
    };
if(fresh_prefill) {
        run_case(32,8,0,true);run_case(2047,8,3,true);run_case(2048,8,-1,true);
        run_case(2049,8,0,true);run_case(2065,8,3,true);
        run_case(4097,8,0,true);run_case(4097,8,3,true);run_case(4097,8,-1,true);
        require(cases==8 && retain_cases==8 && default_cases==0 && failures==3,"fresh state case count");
        std::cout<<"FRESH_DRAFT_PREFILL_STATE PASS cases=8 rounds=2 continuation=4 payload_probes=16 gdn=48 conv_slots=4 ring_digests=2 ring_cursor=exact"<<std::endl;
        return;
    }
    if(qkv_focused) {
        for(const auto item:std::array<std::pair<int,int>,13>{{{2,0},{3,-1},{4,1},{5,0},{6,4},{7,0},{7,1},{7,2},{7,3},{7,4},{7,5},{7,-1},{8,-1}}})run_case(32,item.first,item.second,true);
        run_case(63,7,3,true);run_case(319,7,-1,true);run_case(511,7,0,true);run_case(2047,7,5,true);run_case(2047,7,-1,true);
        run_case(32,7,3,false);run_case(32,7,-1,false);
        require(cases==20 && retain_cases==18 && default_cases==2 && failures==3,"targetqkvk6state matrix count");
        std::cout << "TARGET_QKV_K6_STATE PASS cases=20 retain=18 default=2 failures=3 nonfinite=1 rounds=2 continuation=4 all_attempted_payload=exact gdn=48 conv_slots=4 ring_digests=2 ring_cursor=exact" << std::endl;
        return;
    }
    for (const int block : {2, 4, 8}) {
        for (int rejection = 0; rejection < block - 1; ++rejection)
            run_case(32, block, rejection, true);
        run_case(32, block, -1, true);
    }
    for (const int prefix : {63, 319, 575})
        for (const int rejection : {0, -1}) run_case(prefix, 8, rejection, true);
    for (const int rejection : {0, 6, -1}) run_case(512, 8, rejection, true);
    for (const int rejection : {0, 3, 6, -1}) run_case(2047, 8, rejection, true);
    for (const int rejection : {3, -1}) run_case(32, 8, rejection, false);
    require(cases == 29 && retain_cases == 27 && default_cases == 2 && failures == 3,
            "transactionretainqual matrix count mismatch");
    std::cout << "TRANSACTIONRETAIN PASS cases=29 retain=27 default=2 failures=3 nonfinite=1"
              << " rounds=2 continuation=4 ring_digests=2 gdn=48 conv_slots=4 oscar_live=exact"
              << " payload=exact model_replay_separate=1" << std::endl;
}
