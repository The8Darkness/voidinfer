#pragma once

void run_coalesced_context_parity(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    require(source.size()>=600 && target.max_context()>=600,"coalesced context fixture extent");
    struct Restore {
        std::string value;
        ~Restore(){_putenv_s("NINFER_EXL3_ATTENTION_COALESCE_INPUT_MLP",value.c_str());}
    } restore{std::getenv("NINFER_EXL3_ATTENTION_COALESCE_INPUT_MLP")?
        std::getenv("NINFER_EXL3_ATTENTION_COALESCE_INPUT_MLP"):""};
    _putenv_s("NINFER_EXL3_ATTENTION_COALESCE_INPUT_MLP","0");
    auto control=target.create_context(true);
    _putenv_s("NINFER_EXL3_ATTENTION_COALESCE_INPUT_MLP","1");
    auto candidate=target.create_context(true);
    control->prepare_continuation(8);candidate->prepare_continuation(8);
    const std::string contract="coalesced-context;ordinary-fp16;private-state;canonical-order";
    control->bind_request_compatibility(contract);candidate->bind_request_compatibility(contract);
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    for(int prefix:{63,319,575}) {
        control->reset();candidate->reset();
        const auto prompt=std::span<const std::int64_t>(source.data(),prefix);
        auto a=Request::initialize(*control,prompt,128);
        auto b=Request::initialize(*candidate,prompt,128);
        for(int step=0;step<8;++step) {
            auto expected=control->export_exact_host_state(),actual=candidate->export_exact_host_state();
            require(expected->same_payload(*actual),"coalesced context changed full represented state");
            const auto token=sample_target(*control);
            require(sample_target(*candidate)==token,"coalesced context changed greedy decision");
            control->decode(token);candidate->decode(token);
        }
        require(control->export_exact_host_state()->same_payload(*candidate->export_exact_host_state()),
            "coalesced context final state changed");
    }
    require(candidate->persistent_bytes()<=control->persistent_bytes(),"coalescing increased context workspace");
    require(control->host_kv_stats().coalesced_attention_layers==0 && candidate->host_kv_stats().coalesced_attention_layers==16,
        "coalesced context layout not exercised");
    require(control->persistent_bytes()-candidate->persistent_bytes()==candidate->host_kv_stats().coalesced_attention_bytes_saved,
        "coalesced context allocation attribution differs");
}

void run_context_reuse_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using Clock=std::chrono::steady_clock;
    require(target.max_context()>=1024 && code.size()>=1024 && prose.size()>=1024,
        "context reuse fixture extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    const std::string contract="target=resident-qwen3.8-27b;maxctx=4352;capture_taps=1;"
        "exact_host=1;parallel_attention=1;prefill=wide128;stream=device-barrier";
    auto reusable=target.create_context(true);reusable->prepare_continuation(8);
    reusable->bind_request_compatibility(contract);
    const auto persistent=reusable->persistent_bytes();

    struct Outcome {
        std::vector<std::int64_t> tokens;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
        double total_ms=0,reset_or_create_ms=0;
    };
    const auto execute=[](Exl3TextContext& context,std::span<const std::int64_t> prompt) {
        const auto request=Request::initialize(context,prompt,128);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=sample_target(context);tokens.push_back(token);
            if(row!=7) context.decode(token);
        }
        return Outcome{std::move(tokens),context.export_exact_host_state(),0,0};
    };
    const auto reused_run=[&](std::span<const std::int64_t> prompt) {
        const auto begin=Clock::now();const auto reset=reusable->reset_for_request(contract);
        const auto reset_end=Clock::now();auto outcome=execute(*reusable,prompt);
        outcome.reset_or_create_ms=std::chrono::duration<double,std::milli>(reset_end-begin).count();
        outcome.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
        require(reset.generation==reusable->request_generation() &&
            reset.persistent_bytes==persistent && reusable->persistent_bytes()==persistent,
            "context reuse generation/allocation contract");
        return outcome;
    };
    const auto fresh_run=[&](std::span<const std::int64_t> prompt) {
        const auto begin=Clock::now();auto context=target.create_context(true);
        context->prepare_continuation(8);const auto created=Clock::now();
        auto outcome=execute(*context,prompt);
        outcome.reset_or_create_ms=std::chrono::duration<double,std::milli>(created-begin).count();
        outcome.total_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
        return outcome;
    };

    const std::array<std::tuple<const char*,const std::vector<std::int64_t>*,int>,4> cases{
        std::tuple{"code_long",&code,768},std::tuple{"prose_short",&prose,96},
        std::tuple{"prose_long",&prose,768},std::tuple{"code_short",&code,96}};
    double reused_total=0,fresh_total=0,reused_setup=0,fresh_setup=0;
    int ordinal=0;
    for(const auto& [name,fixture,rows]:cases) {
        const auto prompt=std::span<const std::int64_t>(fixture->data(),rows);
        Outcome reused,fresh;
        if((ordinal++&1)==0) {reused=reused_run(prompt);fresh=fresh_run(prompt);}
        else {fresh=fresh_run(prompt);reused=reused_run(prompt);}
        require(reused.tokens==fresh.tokens && reused.state->same_payload(*fresh.state),
            std::string("context reuse fresh equivalence ")+name);
        reused_total+=reused.total_ms;fresh_total+=fresh.total_ms;
        reused_setup+=reused.reset_or_create_ms;fresh_setup+=fresh.reset_or_create_ms;
        std::cout<<"CONTEXT_REUSE_CASE name="<<name<<" rows="<<rows
            <<" reused_ms="<<reused.total_ms<<" fresh_ms="<<fresh.total_ms
            <<" reset_ms="<<reused.reset_or_create_ms
            <<" create_prepare_ms="<<fresh.reset_or_create_ms
            <<" generation="<<reusable->request_generation()<<std::endl;
    }

    // Cancel after real target work, then immediately reuse for an unrelated
    // short prompt. No partial KV/GDN/conv/tap state may remain visible.
    reusable->reset_for_request(contract);
    reusable->prefill(std::span<const std::int64_t>(code.data(),16));
    reusable->append_exact_prefill_wide(std::span<const std::int64_t>(code.data()+16,32));
    require(reusable->position()==48,"context reuse cancellation setup");
    const auto cancelled_reset=reusable->reset_for_request(contract);
    require(reusable->position()==0 && cancelled_reset.generation==6,
        "context reuse cancellation reset");
    const auto after_cancel=execute(*reusable,std::span<const std::int64_t>(prose.data(),96));
    const auto after_cancel_fresh=fresh_run(std::span<const std::int64_t>(prose.data(),96));
    require(after_cancel.tokens==after_cancel_fresh.tokens &&
        after_cancel.state->same_payload(*after_cancel_fresh.state),
        "context reuse cancellation contaminated next request");

    // Compatibility rejection is pre-mutation. An ordinary request remains
    // available, while the caller can fall back to a fresh differently bound
    // context if its constructor/configuration contract changes.
    const int before_mismatch=reusable->position();bool incompatible=false;
    try {reusable->reset_for_request(contract+";changed=1");}
    catch(const std::invalid_argument&) {incompatible=true;}
    require(incompatible && reusable->position()==before_mismatch,
        "context reuse incompatible contract mutated state");
    auto fallback=target.create_context(true);fallback->prepare_continuation(8);
    fallback->bind_request_compatibility(contract+";changed=1");
    const auto fallback_reset=fallback->reset_for_request(contract+";changed=1");
    require(fallback_reset.generation==1 && fallback->position()==0,
        "context reuse incompatible fallback");

    // A caller-side failure after additional real work exercises recovery
    // through the same explicit discard-or-reset boundary.
    bool failed=false;
    try {
        reusable->append_exact_prefill_wide(
            std::span<const std::int64_t>(code.data()+128,32));
        throw std::runtime_error("intentional request failure after target work");
    }
    catch(const std::exception&) {failed=true;}
    require(failed && reusable->position()>before_mismatch,
        "context reuse failure trigger did not execute target work");
    const auto repaired=reusable->reset_for_request(contract);
    require(repaired.generation==7 && reusable->position()==0 &&
        reusable->persistent_bytes()==persistent,"context reuse failure recovery");

    {
        auto scoped=target.create_context(true);scoped->prepare_continuation(8);
        scoped->bind_request_compatibility(contract);
        const auto resident=Request::initialize(*scoped,
            std::span<const std::int64_t>(code.data(),128),128);
        scoped->restore_exact_host_state(*resident->state());
        const auto generation=scoped->request_generation();
        const auto preserved=scoped->reset_for_request_preserving(contract,*resident->state(),generation);
        require(preserved.exact_payload_preserved && preserved.generation==generation+1 &&
            scoped->exact_host_state_resident(*resident->state()) && scoped->position()==128,
            "scoped request reset did not preserve exact current root");
        require(!scoped->graph_active() && !scoped->continuation_graph_active() &&
            !scoped->transaction_active(),
            "preserving reset retained request execution admission");
        for(bool changed_contract:{false,true}) {
            bool refused=false;
            try{scoped->reset_for_request_preserving(changed_contract?contract+";tenant=other":contract,
                *resident->state(),changed_contract?preserved.generation:generation);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused && scoped->request_generation()==preserved.generation &&
                scoped->exact_host_state_resident(*resident->state()),
                "scoped request reset accepted stale generation or changed contract");
        }
        require(scoped->export_exact_host_state()->same_payload(*resident->state()),
            "scoped request reset changed preserved payload");
        scoped->restore_exact_host_state(*resident->state());scoped->decode(code[128]);
        const auto miss=scoped->reset_for_request_preserving(contract,*resident->state(),preserved.generation);
        require(!miss.exact_payload_preserved && miss.generation==preserved.generation+1 && scoped->position()==0,
            "scoped request reset skipped changed-state full reset");
        require(!scoped->graph_active() && !scoped->continuation_graph_active() &&
            !scoped->transaction_active(),
            "scoped reset fallback retained request execution admission");
        scoped->restore_exact_host_state(*resident->state());
        require(scoped->export_exact_host_state()->same_payload(*resident->state()),
            "scoped reset miss prevented exact restoration");
        auto full_reset=target.create_context(true);full_reset->prepare_continuation(8);
        full_reset->bind_request_compatibility(contract);
        full_reset->restore_exact_host_state(*resident->state());
        const auto baseline_reset=full_reset->reset_for_request(contract);
        full_reset->restore_exact_host_state(*resident->state());
        scoped->restore_exact_host_state(*resident->state());
        const auto candidate_reset=scoped->reset_for_request_preserving(
            contract,*resident->state(),scoped->request_generation());
        require(!baseline_reset.exact_payload_preserved && candidate_reset.exact_payload_preserved,
            "preserving/full reset reference did not exercise distinct routes");
        for(unsigned row=0;row<3;++row) {
            full_reset->decode(code[128+row]);scoped->decode(code[128+row]);
            const auto expected=full_reset->export_exact_host_state();
            const auto actual=scoped->export_exact_host_state();
            require(actual->same_payload(*expected),
                "preserved request reset changed subsequent exact continuation state");
        }
    }
    {
        auto failed_context=target.create_context(true);failed_context->prepare_continuation(8);
        failed_context->bind_request_compatibility(contract);
        const auto root=Request::initialize(*failed_context,
            std::span<const std::int64_t>(code.data(),128),128);
        failed_context->restore_exact_host_state(*root->state());
        const auto failed_workspace=failed_context->host_kv_workspace_owner_for_test();
        require(!failed_workspace.expired(),"reset completion fixture missing workspace owner");
        const auto generation=failed_context->request_generation();
        failed_context->fail_next_request_reset_completion_for_test();
        bool duplicate_refused=false;
        try{failed_context->fail_next_request_reset_completion_for_test();}
        catch(const std::logic_error&){duplicate_refused=true;}
        require(duplicate_refused,"reset completion seam accepted duplicate arm");
        for(bool stale_generation:{false,true}) {
            bool prevalidation_refused=false;
            try{failed_context->reset_for_request_preserving(
                stale_generation?contract:contract+";foreign=1",*root->state(),
                stale_generation?generation+1:generation);}
            catch(const std::invalid_argument&){prevalidation_refused=true;}
            require(prevalidation_refused && failed_context->request_generation()==generation &&
                failed_context->exact_host_state_resident(*root->state()),
                "reset prevalidation mutated state before pending completion fault");
        }
        bool failed=false;
        try{failed_context->reset_for_request_preserving(contract,*root->state(),generation);}
        catch(const std::runtime_error& error){failed=std::string_view(error.what())==
            "injected request reset completion failure";}
        require(failed && failed_context->request_generation()==generation &&
            !failed_context->exact_host_state_resident(*root->state()),
            "failed reset completion published generation or reusable witness");
        bool restore_refused=false,reset_refused=false;
        try{failed_context->restore_exact_host_state_if_needed(*root->state());}
        catch(const std::runtime_error& error){restore_refused=std::string_view(error.what())==
            "exact restore cannot reuse failed HostKV lineage";}
        try{failed_context->reset_for_request(contract);}
        catch(const std::runtime_error& error){reset_refused=std::string_view(error.what())==
            "request reset cannot reuse failed HostKV lineage";}
        require(restore_refused && reset_refused,"failed reset completion admitted reuse");
        require(!failed_workspace.expired(),"failed reset released workspace before retirement");
        failed_context.reset();
        require(failed_workspace.expired(),"certified context destruction retained reset workspace");
        auto replacement=target.create_context(true);replacement->prepare_continuation(8);
        replacement->bind_request_compatibility(contract);
        replacement->restore_exact_host_state(*root->state());
        require(replacement->export_exact_host_state()->same_payload(*root->state()),
            "reset completion failure damaged immutable root used by replacement context");
    }
    {
        auto exhausted=target.create_context(true);exhausted->prepare_continuation(8);
        exhausted->bind_request_compatibility(contract);
        const auto root=Request::initialize(*exhausted,
            std::span<const std::int64_t>(code.data(),128),128);
        exhausted->restore_exact_host_state(*root->state());
        exhausted->exhaust_request_generation_for_test();
        bool refused=false;
        try{(void)exhausted->reset_for_request(contract);}
        catch(const std::overflow_error& error){refused=std::string_view(error.what())==
            "request context generation exhausted";}
        require(refused && exhausted->request_generation()==std::numeric_limits<std::uint64_t>::max() &&
            exhausted->position()==128 && exhausted->exact_host_state_resident(*root->state()) &&
            exhausted->export_exact_host_state()->same_payload(*root->state()),
            "exhausted request generation reset mutated represented state");
    }

    std::cout<<"CONTEXT_REUSE PASS cases=4 exact_tokens_state=1 long_short=1"
        <<" cancelled_reuse=1 incompatible_fallback=1 failure_recovery=1"
        <<" generations="<<reusable->request_generation()
        <<" persistent_bytes="<<persistent
        <<" reused_total_ms="<<reused_total<<" fresh_total_ms="<<fresh_total
        <<" reused_reset_ms="<<reused_setup<<" fresh_create_prepare_ms="<<fresh_setup
        <<std::endl;
}

void run_context_reuse_performance(Exl3TextModel& target,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    using State=ninfer::exl3::Exl3ExactHostState;
    using Clock=std::chrono::steady_clock;
    constexpr int rows=768;
    require(target.max_context()>=1024 && code.size()>=rows && prose.size()>=rows,
        "context reuse T1 fixture extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    const std::string contract="target=resident-qwen3.8-27b;maxctx=4352;capture_taps=1;"
        "exact_host=1;parallel_attention=1;prefill=wide128;stream=device-barrier";
    auto reused=target.create_context(true);reused->prepare_continuation(8);
    reused->bind_request_compatibility(contract);
    struct Outcome {double total_ms=0,setup_ms=0;std::vector<std::int64_t> tokens;
        std::shared_ptr<const State> state;};
    const auto execute=[](Exl3TextContext& context,std::span<const std::int64_t> prompt) {
        const auto request=Request::initialize(context,prompt,128);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=sample_target(context);tokens.push_back(token);
            if(row!=7) context.decode(token);
        }
        return std::pair{std::move(tokens),context.export_exact_host_state()};
    };
    const auto run_reused=[&](std::span<const std::int64_t> prompt) {
        const auto begin=Clock::now();reused->reset_for_request(contract);const auto setup=Clock::now();
        auto [tokens,state]=execute(*reused,prompt);const auto end=Clock::now();
        return Outcome{std::chrono::duration<double,std::milli>(end-begin).count(),
            std::chrono::duration<double,std::milli>(setup-begin).count(),
            std::move(tokens),std::move(state)};
    };
    const auto run_fresh=[&](std::span<const std::int64_t> prompt) {
        const auto begin=Clock::now();auto context=target.create_context(true);
        context->prepare_continuation(8);const auto setup=Clock::now();
        auto [tokens,state]=execute(*context,prompt);const auto end=Clock::now();
        return Outcome{std::chrono::duration<double,std::milli>(end-begin).count(),
            std::chrono::duration<double,std::milli>(setup-begin).count(),
            std::move(tokens),std::move(state)};
    };
    // Exclude one complete request per arm from the decision cohort so global
    // first-use initialization cannot be assigned to whichever arm runs first.
    const auto warm_prompt=std::span<const std::int64_t>(code.data(),96);
    const auto warm_reused=run_reused(warm_prompt);const auto warm_fresh=run_fresh(warm_prompt);
    require(warm_reused.tokens==warm_fresh.tokens &&
        warm_reused.state->same_payload(*warm_fresh.state),"context reuse T1 warm exactness");
    std::vector<double> all_gains;
    for(const auto& [name,fixture]:std::array<std::pair<const char*,const std::vector<std::int64_t>*>,2>{
        std::pair{"code",&code},std::pair{"prose",&prose}}) {
        std::vector<double> gains,reset_ms,create_ms;
        const auto prompt=std::span<const std::int64_t>(fixture->data(),rows);
        for(int rep=0;rep<3;++rep) {
            Outcome a,b;
            const bool fresh_first=rep==1;
            if(fresh_first) {b=run_fresh(prompt);a=run_reused(prompt);}
            else {a=run_reused(prompt);b=run_fresh(prompt);}
            require(a.tokens==b.tokens && a.state->same_payload(*b.state),
                std::string("context reuse T1 exactness ")+name);
            const double gain=(b.total_ms-a.total_ms)*100.0/b.total_ms;
            gains.push_back(gain);all_gains.push_back(gain);
            reset_ms.push_back(a.setup_ms);create_ms.push_back(b.setup_ms);
            std::cout<<"CONTEXT_REUSE_T1_PAIR fixture="<<name<<" rep="<<rep
                <<" order="<<(fresh_first?"FR":"RF")
                <<" reused_ms="<<a.total_ms<<" fresh_ms="<<b.total_ms
                <<" gain_percent="<<gain<<" reset_ms="<<a.setup_ms
                <<" create_prepare_ms="<<b.setup_ms<<std::endl;
        }
        std::cout<<"CONTEXT_REUSE_T1_FIXTURE fixture="<<name
            <<" median_gain_percent="<<median(gains)
            <<" median_reset_ms="<<median(reset_ms)
            <<" median_create_prepare_ms="<<median(create_ms)<<std::endl;
    }
    std::cout<<"CONTEXT_REUSE_T1 PASS fixtures=2 pairs=6 exact_tokens_state=1"
        <<" median_gain_percent="<<median(all_gains)
        <<" positive_pairs="<<std::count_if(all_gains.begin(),all_gains.end(),[](double v){return v>0;})
        <<" warm_reused_ms="<<warm_reused.total_ms
        <<" warm_fresh_ms="<<warm_fresh.total_ms
        <<" persistent_bytes="<<reused->persistent_bytes()<<std::endl;
}
