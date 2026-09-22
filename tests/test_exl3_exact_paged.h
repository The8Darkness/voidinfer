#pragma once

void run_exact_paged_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using State=ninfer::exl3::Exl3ExactHostState;
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    require(source.size()>350 && target.max_context()==1024,"paged fixture extent");
    auto exact=target.create_context(true);
    exact->prepare_continuation(8);
    int cases=0;
    for(int prefix:{63,64,65,127,128,321}) {
        const auto initial=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),prefix));
        const auto root=initial->state();
        require(root->kv_export_bytes()==static_cast<std::size_t>(prefix)*65536,"initial KV transfer extent");
        std::array<std::shared_ptr<const State>,3> family{root};
        for(int branch=0;branch<2;++branch) {
            exact->restore_exact_host_state(*root);
            std::vector<std::int64_t> tail(source.begin()+prefix,source.begin()+prefix+3);
            if(branch) tail[0]=(tail[0]+7)%kVocab;
            // Different ancestors, identical later token strings; state is private.
            exact->continue_rows(tail);exact->finish_exact_continuation();
            family[branch+1]=exact->export_exact_host_state();
            require(family[branch+1]->kv_export_bytes()==3*65536,"suffix-only KV transfer");
            const auto actual_gpu=exact->export_exact_host_state(nullptr,false);
            require(family[branch+1]->same_payload(*actual_gpu),"shared image concealed GPU KV mismatch");
            std::vector<std::int64_t> ids(source.begin(),source.begin()+prefix);
            ids.insert(ids.end(),tail.begin(),tail.end());
            const auto oracle=Request::initialize(*exact,ids);
            require(family[branch+1]->same_payload(*oracle->state()),"paged branch differs from fresh full reference");
            ++cases;
        }
        require(!family[1]->same_payload(*family[2]),"different ancestors aliased state");
        const auto stats=State::storage_stats(family);
        const int full=prefix/64,remainder=prefix%64;
        const int expected_pages=full+(remainder?1:0)+2*((remainder+3+63)/64);
        const int expected_rows=full*64+remainder+2*(remainder+3);
        require(stats.logical_context_tokens==3*prefix+6 && stats.unique_images==3 &&
            stats.unique_kv_pages==expected_pages && stats.materialized_kv_token_rows==expected_rows &&
            stats.materialized_kv_bytes==static_cast<std::uint64_t>(expected_rows)*65536 &&
            stats.allocated_kv_bytes==static_cast<std::uint64_t>(expected_pages)*64*65536,
            "paged physical accounting mismatch");
        // No deduplication from equal strings/payloads or unrelated reingestion.
        const auto independent=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),prefix));
        const std::array<std::shared_ptr<const State>,2> unshared{root,independent->state()};
        const auto separate=State::storage_stats(unshared);
        require(root->same_payload(*independent->state()) && separate.materialized_kv_token_rows==2*prefix,
            "unproven prefix identity deduplicated");
        exact->restore_exact_host_state(*root);
        require(root->same_payload(*exact->export_exact_host_state(nullptr,false)),"COW corrupted parent restore");
        require(exact->export_exact_host_state()->kv_export_bytes()==0,"unchanged image reexports KV");
        std::cout << "EXACT_PAGED_CASE prefix=" << prefix << " logical_tokens=" << stats.logical_context_tokens
            << " materialized_token_rows=" << stats.materialized_kv_token_rows << " pages=" << stats.unique_kv_pages
            << " payload_kv_bytes=" << stats.materialized_kv_bytes << " allocated_kv_bytes=" << stats.allocated_kv_bytes
            << " prefix_saved_bytes=" << stats.logical_kv_bytes-stats.materialized_kv_bytes << '\n';
    }
    std::cout << "EXACT_PAGED PASS branch_cases=" << cases
        << " boundary_roots=6 identity=lineage_only page_tokens=64\n";
}
