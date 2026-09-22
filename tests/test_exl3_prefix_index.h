#pragma once

void run_prefix_index_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Index=ninfer::exl3::Exl3VeriCachePrefixIndex;
    {
        bool refused=false;
        try{Index excessive(Index::maximum_entries+1);}
        catch(const std::invalid_argument& error){refused=std::string_view(error.what())==
            "prefix index capacity exceeds bounded range index";}
        require(refused,"prefix index admitted capacity outside its fixed range metadata");
    }
    {
        ninfer::exl3::Exl3TokenHistory empty;
        require(empty.equals({}),"empty token history equality");
        std::vector<std::int64_t> expected(129);
        for(std::size_t i=0;i<expected.size();++i)expected[i]=static_cast<std::int64_t>(i+1);
        for(std::size_t count:{1U,63U,64U,65U,127U,128U,129U}) {
            const auto span=std::span<const std::int64_t>(expected).first(count);
            const auto history=empty.append(span);
            for(std::size_t split=0;split<=count;++split) {
                const auto chunked=empty.append(span.first(split)).append(span.subspan(split));
                require(chunked.equals(span) && chunked.fingerprint()==history.fingerprint(),
                    "token fingerprint depends on append partition");
                require(chunked.same_tokens(history) && history.same_tokens(chunked),
                    "independently built history pages changed exact identity");
            }
            require(history.append({}).fingerprint()==history.fingerprint(),"empty append changed fingerprint");
            require(history.equals(span) && !history.equals(span.first(count-1)),
                "paged token equality boundary");
            auto changed=std::vector<std::int64_t>(span.begin(),span.end());
            for(std::size_t offset=0;offset<count;++offset) {
                ++changed[offset];require(!history.equals(changed),"paged token equality missed mismatch");
                --changed[offset];
            }
            const std::array<std::int64_t,1> a{201},b{202};
            const auto left=history.append(a),right=history.append(b);
            changed.push_back(a[0]);
            require(left.fingerprint()==empty.append(changed).fingerprint() &&
                history.fingerprint()==empty.append(span).fingerprint(),
                "branch append changed parent or incremental fingerprint");
            require(left.equals(changed) && !right.equals(changed) && history.equals(span),
                "paged token comparison confused immutable branches");
            require(left.same_tokens(empty.append(changed)) && !left.same_tokens(right) &&
                !left.same_tokens(history) && !history.same_tokens(left),
                "shared-page equality confused divergent or unequal histories");
            const auto nested=left.append(b);
            auto nested_tokens=changed;nested_tokens.push_back(b[0]);
            require(nested.equals(nested_tokens) && nested.same_tokens(empty.append(nested_tokens)) &&
                left.equals(changed) && history.equals(span) && !nested.same_tokens(right),
                "nested token branch changed ancestor history");
        }
    }
    {
        using History=ninfer::exl3::Exl3TokenHistory;
        struct Disarm {~Disarm(){History::fail_allocation_for_test(0);}} disarm;
        for(unsigned depth=1;depth<=3;++depth) {
            const std::size_t boundary=std::size_t{64}<<(5*(depth-1));
            const std::vector<std::int64_t> prefix(boundary,7);
            const auto parent=History{}.append(prefix);
            const std::array<std::int64_t,1> suffix{8};
            // One new leaf and depth index nodes; no temporary final root.
            History::fail_allocation_for_test(depth+2);
            const auto child=parent.append(suffix);
            History::fail_allocation_for_test(0);
            auto expected=prefix;expected.push_back(8);
            require(child.equals(expected) && parent.equals(prefix),
                "root promotion copied temporary metadata or changed parent");
        }
        {
            History empty;
            History::fail_allocation_for_test(1);
            require(empty.append({}).equals({}),"empty history append changed tokens");
            bool invalid=false,allocation=false;
            try{(void)empty.append(std::array<std::int64_t,1>{248320});}
            catch(const std::invalid_argument&){invalid=true;}
            try{(void)empty.append(std::array<std::int64_t,1>{1});}
            catch(const std::bad_alloc&){allocation=true;}
            require(invalid && allocation && empty.position()==0,
                "prevalidation or empty append consumed pending history allocation fault");
        }
        {
            ninfer::exl3::RetainedDescriptorLedger metadata;
            const auto reserve_node=[&](std::uint64_t bytes){return metadata.acquire(bytes);};
            const std::vector<std::int64_t> tokens(64,17);
            auto parent=History{}.append(tokens,reserve_node);
            const auto parent_bytes=metadata.bytes();
            const auto live=History::live_node_blocks_for_test();
            const std::array<std::int64_t,1> suffix{18};
            unsigned reservations=0;bool refused=false;
            try {auto failed=parent.append(suffix,[&](std::uint64_t bytes){
                if(++reservations==2)throw ninfer::exl3::Exl3ResourceReservationExhausted{};
                return metadata.acquire(bytes);
            });}catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
            require(refused && reservations==2 && metadata.bytes()==parent_bytes &&
                History::live_node_blocks_for_test()==live && parent.equals(tokens),
                "radix promotion refusal retained partial node or changed shared parent");
            auto child=parent.append(suffix,reserve_node);
            auto expected=tokens;expected.push_back(18);
            require(child.equals(expected) && parent.equals(tokens) && metadata.bytes()==3*History::node_metadata_bytes(),
                "credited radix promotion lost sharing or full-node accounting");
            std::vector<std::weak_ptr<const void>> weak_nodes;
            child.visit_control_owners([&](const auto& node){weak_nodes.emplace_back(node);});
            parent=History{};
            require(metadata.bytes()==3*History::node_metadata_bytes(),"child lost inherited token node charge");
            child=History{};
            require(std::all_of(weak_nodes.begin(),weak_nodes.end(),[](const auto& weak){return weak.expired();}) &&
                metadata.bytes()==3*History::node_metadata_bytes(),"radix strong retirement lost bounded weak storage charge");
            weak_nodes.clear();require(metadata.bytes()==0,"radix final weak retirement leaked metadata");
        }
        for(std::size_t count:{63U,64U,2047U,2048U,2049U,65535U,65536U}) {
            std::vector<std::int64_t> tokens(count,17);
            const auto parent=History{}.append(tokens);
            const auto fingerprint=parent.fingerprint();
            const std::array<const History*,1> old{&parent};
            const auto before=History::visit(old);
            const auto fork=parent;
            const std::array<const History*,2> aliases{&parent,&fork};
            const auto alias_stats=History::visit(aliases);
            ninfer::exl3::RetainedDescriptorLedger control_ledger;
            for(const auto* history:aliases)history->visit_control_owners([&](const auto& node){
                if(!History::control_credit_belongs_to(node,control_ledger))
                    require(History::attach_control_credit(node,control_ledger.acquire(History::node_metadata_bytes())),
                        "token control credit attachment failed");
            });
            require(control_ledger.bytes()==before.allocated_bytes+before.control_bytes,
                "token aliases duplicated or omitted bounded node metadata");
            require(alias_stats.unique_nodes==before.unique_nodes &&
                alias_stats.control_bytes==before.control_bytes && before.control_bytes>0 &&
                alias_stats.allocated_bytes==before.allocated_bytes &&
                alias_stats.logical_tokens==count*2,
                "history fork copied persistent index or double charged shared nodes");
            const std::array<std::int64_t,2> suffix{18,19};
            {
                ninfer::exl3::RetainedDescriptorLedger provisional;
                unsigned calls=0;bool refused=false;
                try {auto failed=parent.append(suffix,[&](std::uint64_t bytes){
                    if(++calls==2)throw ninfer::exl3::Exl3ResourceReservationExhausted{};
                    return provisional.acquire(bytes);
                });}catch(const ninfer::exl3::Exl3ResourceReservationExhausted&){refused=true;}
                // A leaf-only append has one allocation; deeper paths must
                // unwind their first node when the second reservation fails.
                require((calls==1 || refused) && provisional.bytes()==0 && parent.fingerprint()==fingerprint &&
                    parent.equals(tokens),"token reservation failure changed parent or retained provisional control credit");
            }
            auto expected=tokens;expected.insert(expected.end(),suffix.begin(),suffix.end());
            bool recovered=false;unsigned failures=0;
            for(unsigned fail=1;fail<=16;++fail) {
                History::fail_allocation_for_test(fail);
                try {
                    const auto child=parent.append(suffix);
                    History::fail_allocation_for_test(0);
                    require(child.equals(expected) && child.same_tokens(History{}.append(expected)) &&
                        child.suffix(static_cast<int>(count))==std::vector<std::int64_t>(suffix.begin(),suffix.end()),
                        "radix growth changed represented tokens or suffix ordering");
                    const auto nested=child.append(suffix);
                    require(nested.suffix(static_cast<int>(count))==std::vector<std::int64_t>({18,19,18,19}) &&
                        child.equals(expected),"nested radix branch mutated parent tail");
                    recovered=true;break;
                } catch(const std::bad_alloc&) {
                    ++failures;
                    require(parent.equals(tokens) && parent.fingerprint()==fingerprint &&
                        History::visit(old).allocated_bytes==before.allocated_bytes,
                        "failed radix append changed retained parent or its charges");
                }
            }
            require(recovered && failures>0,"history allocation schedule did not reach successful retry");
            const auto left=parent.append(std::array<std::int64_t,1>{20});
            const auto right=parent.append(std::array<std::int64_t,1>{21});
            const std::array<const History*,3> branches{&parent,&left,&right};
            std::size_t visited=0;
            const auto shared=History::visit(branches,[&](const void*,std::size_t bytes){visited+=bytes;});
            require(shared.unique_pages==before.unique_pages+2 &&
                shared.unique_nodes<=before.unique_nodes+12 && visited==shared.allocated_bytes &&
                parent.equals(tokens) && !left.same_tokens(right),
                "tiny branch append copied complete history or lost metadata accounting");
        }
        {
            std::vector<std::int64_t> tokens(70003,23);
            auto parent=History{}.append(std::span<const std::int64_t>(tokens).first(63));
            const auto parent_fingerprint=parent.fingerprint();
            const std::array<const History*,1> retained{&parent};
            const auto parent_storage=History::visit(retained);
            for(unsigned failure:{1U,2U,3U,4U}) {
                History::fail_allocation_for_test(failure);
                bool refused=false;
                try{(void)parent.append(std::span<const std::int64_t>(tokens).subspan(63));}
                catch(const std::bad_alloc&){refused=true;}
                History::fail_allocation_for_test(0);
                require(refused && parent.equals(std::span<const std::int64_t>(tokens).first(63)) &&
                    parent.fingerprint()==parent_fingerprint &&
                    History::visit(retained).allocated_bytes==parent_storage.allocated_bytes,
                    "multi-level growth failure changed retained history");
            }
            auto child=parent.append(std::span<const std::int64_t>(tokens).subspan(63));
            const auto cold=History{}.append(tokens);
            parent={};
            require(child.equals(tokens) && child.same_tokens(cold) &&
                child.fingerprint()==cold.fingerprint() &&
                child.suffix(65535)==std::vector<std::int64_t>(tokens.begin()+65535,tokens.end()),
                "bulk append across radix levels lost tokens after ancestor release");
            const std::array<const History*,1> one{&child};
            const auto storage=History::visit(one);
            require(storage.unique_pages==(tokens.size()+63)/64 &&
                storage.unique_nodes<storage.unique_pages*2,
                "bulk history retained redundant intermediate index paths");
            for(std::size_t changed:{0U,63U,64U,2047U,2048U,65535U,65536U,70002U}) {
                ++tokens[changed];
                require(!child.same_tokens(History{}.append(tokens)),
                    "shared-spine equality skipped changed represented page");
                --tokens[changed];
            }
        }
    }
    constexpr int prefix_rows=321,tail_rows=64;
    require(target.max_context()>=prefix_rows+tail_rows &&
        source.size()>=prefix_rows+tail_rows,"prefix index fixture extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto builder=target.create_context(true);builder->prepare_continuation(8);
    auto cached=target.create_context(true);cached->prepare_continuation(8);
    auto fresh=target.create_context(true);fresh->prepare_continuation(8);
    const auto prefix=std::span<const std::int64_t>(source.data(),prefix_rows);
    const std::string contract="target=resident-qwen3.8-27b;tokenizer=0997f410;"
        "position=zero;mask=causal;modality=text;route=exact-host-v1";

    const auto root=Request::initialize(*builder,prefix,128);
    const auto root_state=root->state();
    {
        using History=ninfer::exl3::Exl3TokenHistory;
        struct Disarm {~Disarm(){History::fail_allocation_for_test(0);}} disarm;
        const auto suffix=std::span<const std::int64_t>(source).subspan(prefix_rows,1);
        const auto fingerprint=root->token_fingerprint();
        const auto unrelated=Request::initialize(*cached,prefix.first(128),128);
        for(unsigned fault:{1U,2U}) {
            unsigned progress=0;bool failed=false;
            History::fail_allocation_for_test(fault);
            try{(void)root->append_prompt(*cached,suffix,8,[&](int){++progress;});}
            catch(const std::bad_alloc&){failed=true;}
            History::fail_allocation_for_test(0);
            require(failed && progress==0 && root->matches_tokens(prefix) &&
                root->token_fingerprint()==fingerprint &&
                cached->export_exact_host_state()->same_payload(*unrelated->state()),
                "request history allocation failure changed prior lane or immutable root");
        }
        const auto child=root->append_prompt(*cached,suffix,8);
        auto complete=std::vector<std::int64_t>(prefix.begin(),prefix.end());
        complete.push_back(suffix.front());
        const auto oracle_parent=Request::initialize(*fresh,prefix,128);
        const auto oracle=oracle_parent->append_prompt(*fresh,suffix,8);
        require(child->matches_tokens(complete) && child->same_taps(*oracle) &&
            child->state()->same_payload(*oracle->state()) && root->matches_tokens(prefix),
            "request append retry after history allocation failure changed authority");
    }
    {
        Index default_index(2);
        default_index.publish(root,contract);
        require(default_index.lookup(*cached,prefix,contract)==root,
            "cached history fingerprint differs from query hash");
        require(default_index.erase(*cached,prefix,contract) && default_index.size()==0,
            "default erase hash differs from publication hash");
        require(default_index.admit(root,contract,64ULL<<30,64ULL<<30,0).admitted &&
            default_index.lookup(*cached,prefix,contract)==root,
            "admission fingerprint differs from default lookup");
    }
    Index index(3,[](std::span<const std::int64_t>,std::string_view) {
        return 0x54452d434f4c4c49ULL; // Force every publication into one hash bucket.
    });
    index.publish(root,contract);
    require(index.lookup(*cached,prefix,contract)==root,"prefix index exact hit");

    auto changed_prefix=std::vector<std::int64_t>(prefix.begin(),prefix.end());
    changed_prefix.back()=(changed_prefix.back()+1)%kVocab;
    require(!index.lookup(*cached,changed_prefix,contract),"prefix index token collision accepted");
    require(!index.lookup(*cached,prefix,"incompatible-route"),"prefix index contract mismatch accepted");
    require(!index.lookup(*cached,prefix.first(prefix.size()-1),contract),"prefix index partial match accepted");
    require(!index.lookup(*cached,{},contract) && !index.lookup(*cached,prefix,{}),
        "prefix index empty identity accepted");

    // A distinct colliding root must neither conceal nor alias the first.
    const auto shorter=prefix.first(prefix.size()-1);
    const auto collision=Request::initialize(*builder,shorter,128);
    index.publish(collision,contract);
    require(index.lookup(*cached,prefix,contract)==root &&
        index.lookup(*cached,shorter,contract)==collision,
        "prefix index full collision verification");
    require(index.lookup_longest(*cached,prefix,contract)==root &&
        index.lookup_longest(*cached,changed_prefix,contract)==collision,
        "length shortlist trusted colliding digest or newer shorter entry");
    require(!index.lookup_longest(*cached,changed_prefix,contract,prefix.size()) &&
        !index.lookup_longest(*cached,prefix,"incompatible-route") &&
        !index.lookup_longest(*cached,{},contract),
        "length shortlist bypassed exact frontier or contract");
    auto divergent=changed_prefix;divergent.front()=(divergent.front()+1)%kVocab;
    require(!index.lookup_longest(*cached,divergent,contract),
        "length shortlist accepted divergent colliding branch");

    const auto prove_branch=[&](std::vector<std::int64_t> tail) {
        const auto selected=index.lookup(*cached,prefix,contract);
        require(selected==root,"prefix index attach root selection");
        const auto attach_start=std::chrono::steady_clock::now();
        const auto attached=selected->append_prompt(*cached,tail,32);
        const double attach_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-attach_start).count();
        std::vector<std::int64_t> complete(prefix.begin(),prefix.end());
        complete.insert(complete.end(),tail.begin(),tail.end());
        const auto fresh_start=std::chrono::steady_clock::now();
        const auto rebuilt=Request::initialize(*fresh,complete,32);
        const double fresh_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-fresh_start).count();
        require(attached->state()->same_payload(*rebuilt->state()) &&
            attached->same_taps(*rebuilt) &&
            attached->token_suffix()==rebuilt->token_suffix(),
            "prefix index attached branch differs from fresh prefill");
        return std::tuple{attached,attach_ms,fresh_ms};
    };
    std::vector<std::int64_t> tail_a(source.begin()+prefix_rows,
        source.begin()+prefix_rows+tail_rows);
    auto tail_b=tail_a;tail_b.front()=(tail_b.front()+7)%kVocab;
    const auto branch_a=prove_branch(tail_a);
    const auto branch_b=prove_branch(tail_b);
    require(!std::get<0>(branch_a)->state()->same_payload(*std::get<0>(branch_b)->state()) &&
        root->state()==root_state,"prefix index branch isolation");

    // A caller cancellation after actual suffix work must roll the execution
    // context back to the immutable root and leave the index publication live.
    bool cancelled=false;
    try {
        root->append_prompt(*cached,tail_a,16,[&](int position) {
            if(position>prefix_rows) throw std::runtime_error("intentional attach cancellation");
        });
    } catch(const std::runtime_error&) {cancelled=true;}
    require(cancelled && cached->exact_host_state_resident(*root->state()) &&
        index.lookup(*cached,prefix,contract)==root,
        "prefix index cancelled attach contaminated authority");

    // Concurrent readers see immutable shared ownership; the forced collision
    // makes every successful lookup exercise complete identity verification.
    std::vector<std::future<bool>> readers;
    for(int reader=0;reader<8;++reader) readers.push_back(std::async(std::launch::async,[&] {
        for(int probe=0;probe<512;++probe)
            if(index.lookup(*cached,prefix,contract)!=root ||
                index.lookup(*cached,changed_prefix,contract)) return false;
        return true;
    }));
    for(auto& reader:readers) require(reader.get(),"prefix index concurrent reader mismatch");

    const auto lookup_start=std::chrono::steady_clock::now();
    for(int probe=0;probe<10000;++probe)
        require(index.lookup(*cached,prefix,contract)==root,"prefix index timed lookup");
    const double lookup_ns=std::chrono::duration<double,std::nano>(
        std::chrono::steady_clock::now()-lookup_start).count()/10000.0;

    // FIFO bound is deterministic. A reader-held root stays valid after index
    // eviction; an index-only publication releases on erase/clear.
    const auto third=Request::initialize(*builder,prefix.first(prefix.size()-2),128);
    const auto fourth=Request::initialize(*builder,prefix.first(prefix.size()-3),128);
    index.publish(third,contract);index.publish(fourth,contract);
    require(index.size()==index.capacity() && !index.lookup(*cached,prefix,contract) &&
        index.lookup(*cached,shorter,contract)==collision,
        "prefix index bounded eviction order");
    require(root->state()->same_payload(*root_state),"eviction invalidated reader-held root");
    require(index.erase(*cached,shorter,contract) &&
        !index.lookup(*cached,shorter,contract),"prefix index exact erase");

    Index lifetime(1);
    std::weak_ptr<const Request> weak;
    {
        auto only=Request::initialize(*builder,prefix.first(64),32);
        weak=only;lifetime.publish(only,contract);only.reset();
    }
    require(!weak.expired(),"prefix index failed to own publication");
    const auto identity_bytes=lifetime.identity_allocated_bytes();
    auto retained=lifetime.lookup(*cached,prefix.first(64),contract);
    require(retained && retained->matches_tokens(prefix.first(64)),"retained history lookup missing");
    lifetime.clear();require(!weak.expired() && lifetime.size()==0 && retained->matches_tokens(prefix.first(64)),
        "index clear invalidated returned immutable history");
    retained.reset();require(weak.expired(),"final returned-root release retained token authority");

    // The lookup copies only the selected strong root while holding the shared
    // index lock. A replacement may retire its Entry immediately afterward,
    // but it cannot invalidate or retarget the caller's returned authority.
    Index concurrent_replacement(1,[](std::span<const std::int64_t>,std::string_view) {
        return 0x54452d434f4c4c49ULL;
    });
    auto old_publication=Request::initialize(*builder,prefix.first(96),64);
    auto new_publication=Request::initialize(*fresh,prefix.first(96),64);
    require(old_publication!=new_publication &&
        old_publication->state()->same_payload(*new_publication->state()),
        "concurrent replacement controls lack distinct exact roots");
    const auto* old_address=old_publication.get();
    const auto* new_address=new_publication.get();
    std::weak_ptr<const Request> old_weak=old_publication,new_weak=new_publication;
    concurrent_replacement.publish(old_publication,contract);old_publication.reset();
    std::promise<void> selected_ready,replacement_committed;
    auto replacement_visible=replacement_committed.get_future().share();
    auto concurrent_reader=std::async(std::launch::async,[&] {
        auto selected=concurrent_replacement.lookup(*cached,prefix.first(96),contract);
        const bool retained_old=selected && selected.get()==old_address &&
            selected->matches_tokens(prefix.first(96));
        selected_ready.set_value();replacement_visible.wait();
        const auto current=concurrent_replacement.lookup(*cached,prefix.first(96),contract);
        return retained_old && selected.get()==old_address &&
            selected->matches_tokens(prefix.first(96)) && current && current.get()==new_address &&
            current->matches_tokens(prefix.first(96));
    });
    selected_ready.get_future().wait();
    concurrent_replacement.publish(new_publication,contract);new_publication.reset();
    require(!old_weak.expired() && !new_weak.expired() &&
        concurrent_replacement.lookup(*cached,prefix.first(96),contract).get()==new_address,
        "replacement did not atomically publish new root or dropped retained lookup");
    replacement_committed.set_value();
    require(concurrent_reader.get() && old_weak.expired() && !new_weak.expired(),
        "concurrent replacement changed returned authority or retained retired Entry");
    concurrent_replacement.clear();
    require(new_weak.expired(),"concurrent replacement clear retained final publication");

    std::cout<<"PREFIX_INDEX PASS forced_collision=1 exact_hits=10000 concurrent_readers=8"
        <<" concurrent_replacement=retained branches=2 cancellation=rollback bounded_capacity=3 identity_bytes="<<identity_bytes
        <<" lookup_ns="<<lookup_ns
        <<" branch_a_cached_ms="<<std::get<1>(branch_a)
        <<" branch_a_fresh_ms="<<std::get<2>(branch_a)
        <<" branch_b_cached_ms="<<std::get<1>(branch_b)
        <<" branch_b_fresh_ms="<<std::get<2>(branch_b)
        <<" final_position="<<std::get<0>(branch_a)->state()->position()<<std::endl;
}
