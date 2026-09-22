#pragma once

void run_exact_host_state_qualification(Exl3TextModel& target,
                                       const std::vector<std::int64_t>& source) {
    require(target.max_context() == 1024 && source.size() >= 256, "exacthost fixture extent");
    auto lane = target.create_context(true);
    auto oracle = target.create_context(true);
    const auto ingest = [&](Exl3TextContext& context, int offset, int rows) {
        context.reset();
        context.prefill(std::span<const std::int64_t>(source.data() + offset, 16));
        for (int i = 16; i < rows; ++i) context.decode(source[offset + i]);
    };
    ingest(*lane, 0, 20);
    if(const auto* mode=std::getenv("NINFER_TEST_EXACT_EXPORT_PLANNING_FAILURE");mode && std::string_view(mode)=="1") {
        ninfer::exl3::RetainedDescriptorLedger snapshot_ledger;
        for(unsigned refusal=0;refusal<3;++refusal) {
            const auto blocks=ninfer::exl3::bounded_shared_live_blocks_for_test<ninfer::exl3::Exl3ExactHostState>();
            const auto prior=lane->recurrent_export_stats();
            lane->set_snapshot_metadata_reservation([&](std::uint64_t bytes) {
                if(refusal==0)throw ninfer::exl3::Exl3ResourceReservationExhausted{};
                return snapshot_ledger.acquire(refusal==1?bytes-1:bytes+1);
            });
            bool rejected=false;
            try{(void)lane->export_exact_host_state();}
            catch(const std::runtime_error& error) {
                rejected=refusal==0?dynamic_cast<const ninfer::exl3::Exl3ResourceReservationExhausted*>(&error)!=nullptr:
                    std::string_view(error.what())=="snapshot metadata reservation extent changed";
            }
            const auto after=lane->recurrent_export_stats();
            require(rejected && snapshot_ledger.bytes()==0 && lane->position()==20 &&
                after.allocations==prior.allocations && after.pool_hits==prior.pool_hits &&
                after.pinned_exports==prior.pinned_exports &&
                ninfer::exl3::bounded_shared_live_blocks_for_test<ninfer::exl3::Exl3ExactHostState>()==blocks,
                "snapshot reservation refusal allocated state or reached recurrent export");
        }
        unsigned reservations=0;
        lane->set_snapshot_metadata_reservation([&](std::uint64_t bytes) {
            ++reservations;return snapshot_ledger.acquire(bytes);
        });
        const auto before=lane->recurrent_export_stats();
        bool failed=false;
        try{lane->export_exact_host_state(nullptr,true,true);}
        catch(const std::runtime_error& error) {
            failed=std::string_view(error.what())=="injected exact export preparation failure after recurrent plan";
        }
        const auto abandoned=lane->recurrent_export_stats();
        require(reservations==1 && snapshot_ledger.bytes()==0,
            "actual export failure retained provisional snapshot metadata");
        require(failed && lane->position()==20 && abandoned.allocations==before.allocations+1 &&
            abandoned.pinned_exports==before.pinned_exports && abandoned.poisons==before.poisons,
            "actual export planning fault submitted or poisoned recurrent state");
        const auto retry=lane->export_exact_host_state();
        require(reservations==2 && snapshot_ledger.bytes()==retry->snapshot_metadata_bytes() &&
            ninfer::exl3::Exl3ExactHostState::snapshot_metadata_credit_belongs_to(retry,snapshot_ledger),
            "actual export failed to attach reserved snapshot metadata");
        const auto retried=lane->recurrent_export_stats();
        require(retried.allocations==abandoned.allocations && retried.pool_hits==abandoned.pool_hits+1 &&
            retried.pinned_exports==abandoned.pinned_exports+1,
            "actual export preparation failure prevented planned slab reuse");
        ingest(*oracle,0,20);
        require(retry->same_payload(*oracle->export_exact_host_state()),
            "actual export planning retry changed authoritative state");
        lane->set_snapshot_metadata_reservation({});
    }
    if(const auto* mode=std::getenv("NINFER_TEST_EXACT_EXPORT_COPY_FAILURE");
       mode && std::string_view(mode)=="1") {
        const auto blocks=ninfer::exl3::bounded_shared_live_blocks_for_test<
            ninfer::exl3::Exl3ExactHostState>();
        const auto before=lane->recurrent_export_stats();
        lane->fail_export_copy_for_test(2);
        bool failed=false;
        try{(void)lane->export_exact_host_state();}
        catch(const std::runtime_error&){failed=true;}
        const auto after=lane->recurrent_export_stats();
        require(failed && lane->position()==20 &&
                after.copy_submissions==before.copy_submissions+2 &&
                ninfer::exl3::bounded_shared_live_blocks_for_test<
                    ninfer::exl3::Exl3ExactHostState>()==blocks,
            "partial export copy failure published state or lost submission boundary");
        const auto retry=lane->export_exact_host_state();
        ingest(*oracle,0,20);
        require(retry->same_payload(*oracle->export_exact_host_state()),
            "partial export copy failure changed retry authority");
    }
    const auto shared = lane->export_exact_host_state();
    // Actual immutable prefix object can be referenced by independent requests.
    const auto request_a_prefix = shared, request_b_prefix = shared;
    require(request_a_prefix.get() == request_b_prefix.get(), "exacthost prefix sharing");
    const auto start = std::chrono::steady_clock::now();
    std::array<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>, 2> histories;
    for (int request = 0; request < 2; ++request) {
        lane->reset(); lane->restore_exact_host_state(*shared);
        require(lane->export_exact_host_state()->same_payload(*shared), "exacthost root restore");
        for (int i = 0; i < 4; ++i) lane->decode(source[64 + request * 32 + i]);
        histories[request] = lane->export_exact_host_state();
        ingest(*oracle, 0, 20);
        for (int i = 0; i < 4; ++i) oracle->decode(source[64 + request * 32 + i]);
        require(histories[request]->same_payload(*oracle->export_exact_host_state()),
                "exacthost private tail differs from full replay");
    }
    require(!histories[0]->same_payload(*histories[1]), "exacthost private tails collapsed");
    int compared = 0;
    std::size_t transferred = 0;
    for (int round = 0; round < 3; ++round) {
        for (int request = 0; request < 2; ++request) {
            // One physical execution lane, two independently live DRAM histories.
            // Reset destroys resident history before restoration from L2.
            lane->reset();
            lane->restore_exact_host_state(*histories[request]);
            transferred += histories[request]->payload_bytes();
            ingest(*oracle, 0, 20);
            for (int i = 0; i < 4; ++i) oracle->decode(source[64 + request * 32 + i]);
            for (int step = 0; step < round; ++step) oracle->decode(sample_target(*oracle));
            const auto authorized = sample_target(*oracle);
            require(sample_target(*lane) == authorized, "exacthost next token after eviction");
            lane->decode(authorized); oracle->decode(authorized);
            histories[request] = lane->export_exact_host_state();
            transferred += histories[request]->payload_bytes();
            require(histories[request]->same_payload(*oracle->export_exact_host_state()),
                    "exacthost full state after request switch");
            ++compared;
        }
    }
    // A materially disjoint prefix replaces the resident lane; restore the
    // earlier immutable shared root afterwards to expose stale suffix leakage.
    ingest(*lane, 128, 24);
    const auto disjoint = lane->export_exact_host_state();
    require(!disjoint->same_payload(*histories[0]), "exacthost disjoint history");
    lane->restore_exact_host_state(*shared);
    lane->decode(source[21]);
    ingest(*oracle, 0, 20); oracle->decode(source[21]);
    require(lane->export_exact_host_state()->same_payload(*oracle->export_exact_host_state()),
            "exacthost restored prefix after disjoint suffix");
    auto compressed = target.create_context(true);
    require(compressed->try_enable_oscar_from_environment(), "exacthost OSCAR negative control");
    compressed->prefill(std::span<const std::int64_t>(source.data(), 16));
    bool blocked = false;
    try { compressed->export_exact_host_state(); } catch (const std::exception&) { blocked = true; }
    require(blocked, "exacthost accepted approximate export");
    blocked = false;
    try { compressed->restore_exact_host_state(*shared); } catch (const std::exception&) { blocked = true; }
    require(blocked && compressed->position() == 16, "exacthost accepted compressed restore");
    std::cout << "EXACT_HOST_STATE PASS request_continuations=" << compared
              << " logical_tokens=" << histories[0]->position() + histories[1]->position()
              << " host_payload_bytes=" << shared->payload_bytes() + histories[0]->payload_bytes() + histories[1]->payload_bytes()
              << " kv_bytes_per_token=65536 recurrent_bytes_per_checkpoint=150994944"
              << " counted_round_transfer_bytes=" << transferred
              << " test_wall_ms=" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count()
              << " identity=vericache_exact_fp16_eager\n";
}
