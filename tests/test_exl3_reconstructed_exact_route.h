#pragma once
#include "exl3/reconstruction_config.h"
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/vericache_request.h"

// Target-only qualification. Full payload equality includes raw FP32 recurrent
// bytes, convolution, KV, embeddings, taps, logits and state metadata.
void run_reconstructed_exact_route(Exl3TextModel& target,
                                   const std::vector<std::int64_t>& source) {
    using State = ninfer::exl3::Exl3ExactHostState;
    const auto reuse_mode=env("NINFER_TEST_EQUAL_TRANSFORM_CONTEXT");
    require(reuse_mode.empty() || reuse_mode=="1","equal transform context option");
    const bool reuse_context=reuse_mode=="1";
    const auto fused_mode=env("NINFER_TEST_FUSED_GATE_UP_CONTEXT");
    require(fused_mode.empty() || (fused_mode=="1" && reuse_context),
        "fused gate/up context requires equal-transform composed context mode");
    const bool fused_context=fused_mode=="1";
    const auto residual_mode=env("NINFER_TEST_FUSED_RESIDUAL_CONTEXT");
    require(residual_mode.empty() || (residual_mode=="1" && reuse_context),
        "residual context requires composed equal-transform context mode");
    const bool residual_context=residual_mode=="1";
    struct RestoreTransformOptions {
        std::string dual,reuse,timing,fused,residual;
        ~RestoreTransformOptions() {
            _putenv_s("NINFER_EXL3_GDN_DUAL_INPUT_TRANSFORM",dual.c_str());
            _putenv_s("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS",reuse.c_str());
            _putenv_s("NINFER_EXL3_TARGET_PROJECTION_TIMING",timing.c_str());
            _putenv_s("NINFER_EXL3_GDN_FUSED_GATE_UP_TRANSFORM",fused.c_str());
            _putenv_s("NINFER_EXL3_GDN_FUSED_RESIDUAL_NORM",residual.c_str());
        }
    } restore_transform_options{env("NINFER_EXL3_GDN_DUAL_INPUT_TRANSFORM"),
        env("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS"),env("NINFER_EXL3_TARGET_PROJECTION_TIMING"),
        env("NINFER_EXL3_GDN_FUSED_GATE_UP_TRANSFORM"),env("NINFER_EXL3_GDN_FUSED_RESIDUAL_NORM")};
    if(reuse_context) {
        for(const char* key:{"NINFER_TEST_RECON_CLEANUP_FAILURE",
            "NINFER_TEST_RECON_STARTUP_RESERVATION","NINFER_TEST_RECON_RETIRE_FAILURE"})
            require(env(key).empty(),"transform context comparison requires fault modes unset");
    }
    const auto family=env("NINFER_TEST_RECON_EXACT_FAMILY");
    require(family.empty() || family=="k7" || family=="k6_down" || family=="k6_gate_up",
        "exact route family");
    const bool include_k6_down=family=="k6_down";
    const bool include_k6_gate_up=family=="k6_gate_up";
    const bool include_k6=include_k6_down || include_k6_gate_up;
    const int slice_columns=ninfer::exl3::exl3_reconstruction_slice_columns(
        std::getenv("NINFER_EXL3_RECONSTRUCTION_SLICE_COLUMNS"));
    const std::size_t expected_slab_bytes=17408ull*slice_columns*2;
    if(include_k6) {
        const auto async_all=env("NINFER_EXL3_PREFILL_DIRECT_ASYNC_ALL");
        require(async_all=="0" || async_all=="1","K6 route requires explicit async/direct selection");
    }
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b-a).count();
    };
    const std::filesystem::path directory = env("NINFER_RECON_EXACT_ROUTE_OUT");
    require(!directory.empty() && !std::filesystem::exists(directory), "exact route new output directory");
    std::filesystem::create_directories(directory);
    std::ofstream checks(directory / "correctness.csv"), timings(directory / "pairs.csv");
    require(checks.good() && timings.good(), "exact route evidence creation");
    checks << "phase,step,pass,logit_values,tap_values,exports,calls,rows,workspace_bytes,persistent_bytes,paired_transform_submissions,fused_gate_up_submissions,fused_residual_norm_submissions\n";
    timings << "pair,order,arm,construction_ms,prefill_ms,seed_ms,ttft_ms,decode_ms,export_ms,teardown_ms,wall_ms,exports,calls,rows,workspace_bytes,persistent_bytes,tokens_exact,state_exact,accounting_pass\n";
    require(target.max_context()==4352 && source.size()>=4096, "exact route fixture extent");
    for (const char* key : {"NINFER_EXL3_WIDE_PREFILL", "NINFER_EXL3_PREFILL_STAGED_REDUCTION",
            "NINFER_EXL3_PREFILL_WIDE1024", "NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
            "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A"})
        require(env(key)=="1", std::string("exact route direct flag ")+key);
    const auto configure = [include_k6_down,include_k6_gate_up,slice_columns,reuse_context,
                            fused_context,residual_context](bool candidate) {
        if(residual_context)_putenv_s("NINFER_EXL3_GDN_FUSED_RESIDUAL_NORM",candidate?"1":"0");
        if(fused_context)_putenv_s("NINFER_EXL3_GDN_FUSED_GATE_UP_TRANSFORM",candidate?"1":"0");
        if(reuse_context) {
            _putenv_s("NINFER_EXL3_TARGET_PROJECTION_TIMING","0");
            _putenv_s("NINFER_EXL3_GDN_DUAL_INPUT_TRANSFORM",candidate?"1":"0");
            _putenv_s("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS",candidate?"1":"0");
        }
        _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC","0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC","0");
        _putenv_s("NINFER_EXL3_NUMERIC_ATTENTION_TILED","0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7",candidate?"1":"0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K6_DOWN",
            candidate && include_k6_down?"1":"0");
        _putenv_s("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K6_GATE_UP",
            candidate && include_k6_gate_up?"1":"0");
        _putenv_s("NINFER_EXL3_RECONSTRUCTION_SLICE_COLUMNS",std::to_string(candidate?slice_columns:5120).c_str());
    };
    if(const auto mode=env("NINFER_TEST_PREFILL_ROW_TAIL_CONTEXT");!mode.empty()) {
        require(mode=="1","prefill row-tail context option");
        configure(false);
        std::ofstream receipt(directory/"row_tail_context.csv");
        receipt << "rows,state_equal,position_equal,classification\n";
        for(int rows:{17,23,31}) {
            std::shared_ptr<const State> reference;
            std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> reference_request;
            {
                auto control=target.create_context(true);control->prepare_continuation(8);
                auto request=ninfer::exl3::Exl3VeriCacheRequest::initialize(*control,
                    std::span<const std::int64_t>(source.data(),16+rows),32);
                reference=request->state();
                reference_request=request;
            }
            auto candidate=target.create_context(true);candidate->prepare_continuation(8);
            candidate->prefill(std::span<const std::int64_t>(source.data(),16));
            const auto before=candidate->export_exact_host_state();
            for(int bad:{0,16,32}) {
                bool refused=false;
                try{candidate->append_exact_prefill_tail(std::span<const std::int64_t>(source.data()+16,bad));}
                catch(const std::exception&){refused=true;}
                require(refused && candidate->position()==16 &&
                    before->same_payload(*candidate->export_exact_host_state()),
                    "invalid row-tail extent mutated context");
            }
            for(const auto invalid:{std::pair{16,0},std::pair{32,0},std::pair{rows,-1},std::pair{rows,256}}) {
                bool refused=false;
                try{candidate->poison_prefill_tail_hidden_for_test(invalid.first,invalid.second);}
                catch(const std::exception&){refused=true;}
                require(refused && before->same_payload(*candidate->export_exact_host_state()),
                    "invalid hidden-tail poison mutated authority");
            }
            candidate->poison_prefill_tail_hidden_for_test(rows,0x7f);
            candidate->append_exact_prefill_tail(std::span<const std::int64_t>(source.data()+16,rows));
            candidate->finish_exact_prefill();
            const bool positions=candidate->position()==16+rows && candidate->device_position_host()==15+rows;
            const bool same=reference->same_payload(*candidate->export_exact_host_state());
            receipt << rows << ',' << same << ',' << positions << ",REQUIRES_NUMERICAL_QUALIFICATION\n";
            receipt.flush();require(receipt.good(),"row-tail comparison receipt");
            require(positions,"actual-row tail skipped state positions");
            require(same,"row-tail changed canonical final state; retain unqualified route");
            const auto direct_tail_state=candidate->export_exact_host_state();
            candidate.reset();
            {
                auto zero=target.create_context(true);zero->prepare_continuation(8);
                zero->prefill(std::span<const std::int64_t>(source.data(),16));
                zero->poison_prefill_tail_hidden_for_test(rows,0);
                zero->append_exact_prefill_tail(std::span<const std::int64_t>(source.data()+16,rows));
                zero->finish_exact_prefill();
                require(direct_tail_state->same_payload(*zero->export_exact_host_state()),
                    "unused hidden-row values changed actual tail state");
            }
            auto request_context=target.create_context(true);request_context->prepare_continuation(8);
            std::vector<int> boundaries;
            const auto actual_request=ninfer::exl3::Exl3VeriCacheRequest::initialize(*request_context,
                std::span<const std::int64_t>(source.data(),16+rows),32,
                [&](int position){boundaries.push_back(position);},true);
            require(boundaries==std::vector<int>({16,16+rows}),
                "actual request tail did not use one represented final chunk");
            require(actual_request->state()->same_payload(*direct_tail_state),
                "request tail differs from direct context tail");
            require(actual_request->same_taps(*reference_request) &&
                actual_request->tap_bytes()==reference_request->tap_bytes() &&
                actual_request->token_suffix()==reference_request->token_suffix(),
                "request tail altered retained conditioning history");
            // Carry the actual request result into the real native B8 caller.
            // Teacher-forced rows isolate target composition from draft policy.
            const auto verification_tokens=std::span<const std::int64_t>(source.data()+16+rows,8);
            request_context->continue_rows(verification_tokens);
            const auto verification_logits=request_context->continuation_logits_bits_host();
            const auto verification_taps=request_context->exact_tap_rows_host();
            request_context->finish_exact_continuation();
            const auto verification_state=request_context->export_exact_host_state();
            request_context.reset();
            for(bool replay:{false,true}) {
                auto verifier=target.create_context(true);verifier->prepare_continuation(8);
                verifier->restore_exact_host_state(*(replay?actual_request->state():reference));
                verifier->continue_rows(verification_tokens);
                require(verifier->continuation_logits_bits_host()==verification_logits &&
                    verifier->exact_tap_rows_host()==verification_taps,
                    "tail-prefill to native B8 changed logits or taps");
                verifier->finish_exact_continuation();
                require(verifier->export_exact_host_state()->same_payload(*verification_state),
                    "tail-prefill to native B8 changed complete state");
                require(actual_request->state()->same_payload(*direct_tail_state) &&
                    reference_request->state()->same_payload(*reference),
                    "native B8 composition mutated retained prefill roots");
            }
        }
        for(const auto shape:{std::pair{8,17},std::pair{8,31},std::pair{16,17},
            std::pair{16,31},std::pair{32,16},std::pair{32,32},std::pair{32,33}}) {
            std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> baseline;
            std::vector<int> baseline_positions;
            for(bool enabled:{false,true}) {
                auto context=target.create_context(true);context->prepare_continuation(8);
                std::vector<int> positions;
                const auto total=16+shape.second;
                std::vector<std::int64_t> guarded(source.begin(),source.begin()+total);
                guarded.resize(total+32,enabled?-1:248320);
                const auto request=ninfer::exl3::Exl3VeriCacheRequest::initialize(*context,
                    std::span<const std::int64_t>(guarded.data(),total),shape.first,
                    [&](int position){positions.push_back(position);},enabled);
                require(std::all_of(guarded.begin()+total,guarded.end(),
                    [&](auto token){return token==(enabled?-1:248320);}),
                    "request modified outside represented prompt span");
                if(!enabled){baseline=request;baseline_positions=positions;}
                else require(positions==baseline_positions && request->state()->same_payload(*baseline->state()) &&
                    request->same_taps(*baseline) && request->token_suffix()==baseline->token_suffix(),
                    "tail opt-in changed excluded request geometry");
            }
        }
        return;
    }
    if(const auto mode=env("NINFER_TEST_HEAD_CONSUMER_CONTEXT");!mode.empty()) {
        require(mode=="1","head consumer context option");
        configure(false);
        for(int rows:{2,7,8}) {
            auto context=target.create_context(true);context->prepare_continuation(8);
            context->prefill(std::span<const std::int64_t>(source.data(),16));
            const auto prefill_head=context->head_work_stats();
            require(prefill_head.submitted_rows==1 && prefill_head.omitted_rows==15,
                "root-only prefill head accounting mismatch");
            context->continue_rows(std::span<const std::int64_t>(source.data()+16,rows));
            const auto verification_head=context->head_work_stats();
            require(verification_head.submitted_rows==prefill_head.submitted_rows+rows &&
                verification_head.omitted_rows==prefill_head.omitted_rows,
                "verification omitted required head work");
            const auto all=context->continuation_logits_bits_host();
            require(all.size()==std::size_t(rows)*kVocab,"head consumer missing verification vocabulary");
            std::vector<std::uint16_t> root(kVocab),after(kVocab);
            cuda_check(cudaMemcpy(root.data(),context->logits_device(),root.size()*2,cudaMemcpyDeviceToHost),
                "head consumer final root readback");
            require(std::equal(root.begin(),root.end(),all.begin()+std::size_t(rows-1)*kVocab),
                "head root differs from final verification row");
            context->finish_exact_continuation();
            require(context->continuation_rows()==0 && context->continuation_logits_device()==nullptr,
                "retired all-row head output remains visible");
            bool refused=false;
            try{context->continuation_logits_bits_host();}catch(const std::exception&){refused=true;}
            require(refused,"retired verification logits read admitted");
            cuda_check(cudaMemcpy(after.data(),context->logits_device(),after.size()*2,cudaMemcpyDeviceToHost),
                "head consumer retained root readback");
            require(after==root && context->position()==16+rows,
                "head output retirement changed required root");
            require(context->head_work_stats().submitted_rows==verification_head.submitted_rows &&
                context->head_work_stats().omitted_rows==verification_head.omitted_rows,
                "head retirement counted numerical work");
        }
        return; // This fixture never enters the historical timing loop.
    }
    if(const auto mode=env("NINFER_TEST_PREFILL_PARTITION_CONTEXT");!mode.empty()) {
        require(mode=="1","prefill partition context option");
        configure(false);
        std::ofstream partitions(directory/"prefill_partitions.csv");
        partitions << "tokens,classification,state_equal,positions_equal,logits_equal,taps_equal\n";
        require(partitions.good(),"partition receipt creation");
        for(int total:{17,49,143,144,1039,1040,1041}) {
            const bool identical=ninfer::exl3::exl3_prefill_partition_relation(128,1024,total-16)==
                ninfer::exl3::Exl3PrefillPartitionRelation::identical;
            std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> control;
            std::vector<int> expected_positions;
            std::vector<std::uint16_t> expected_logits;
            std::array<std::vector<std::uint16_t>,5> expected_taps;
            for(int width:{128,1024}) {
                auto context=target.create_context(true);context->prepare_continuation(8);
                std::vector<int> positions;
                auto request=ninfer::exl3::Exl3VeriCacheRequest::initialize(*context,
                    std::span<const std::int64_t>(source.data(),total),width,
                    [&](int position){positions.push_back(position);});
                std::vector<std::uint16_t> logits(kVocab);
                cuda_check(cudaMemcpy(logits.data(),context->logits_device(),logits.size()*2,cudaMemcpyDeviceToHost),"partition context logits");
                const auto taps=context->exact_tap_rows_host();
                require(context->position()==total && context->device_position_host()==total-1,
                    "partition context final position");
                if(width==128){control=request;expected_positions=positions;expected_logits=logits;expected_taps=taps;}
                else {
                    const bool state_equal=request->state()->same_payload(*control->state());
                    const bool positions_equal=positions==expected_positions;
                    const bool logits_equal=logits==expected_logits,taps_equal=taps==expected_taps;
                    partitions << total << ',' << (identical?"IDENTICAL_PARTITION":"REQUIRES_NUMERICAL_QUALIFICATION")
                        << ',' << state_equal << ',' << positions_equal << ',' << logits_equal << ',' << taps_equal << '\n';
                    partitions.flush();require(partitions.good(),"partition receipt write");
                    require(positions_equal==identical,"actual call boundaries disagree with partition contract");
                    if(identical)require(state_equal && logits_equal && taps_equal,
                        "identical prefill partitions changed state/outputs/history");
                    // Equality at one changed partition never promotes its classification.
                }
            }
        }
        return;
    }
    if(const auto mode=env("NINFER_TEST_RECON_CLEANUP_FAILURE");!mode.empty()) {
        require(mode=="1","reconstruction cleanup fixture option");
        configure(true);
        const auto before=Exl3TextContext::reconstruction_quarantined_allocations();
        auto ctx=target.create_context(true);
        require(ctx->reconstructed_exact_stats().workspace_bytes==expected_slab_bytes,
            "cleanup fixture missing reconstruction allocation");
        ctx->fail_reconstruction_cleanup_for_test();ctx.reset();
        require(Exl3TextContext::reconstruction_quarantined_allocations()==before+1,
            "failed reconstruction cleanup discarded allocation identity");
        bool refused=false;
        try{auto retry=target.create_context(true);}
        catch(const std::exception& error){refused=std::string(error.what()).find("allocation cleanup")!=std::string::npos;}
        require(refused,"failed reconstruction cleanup admitted context reload");
        return;
    }
    if(const auto mode=env("NINFER_TEST_RECON_STARTUP_RESERVATION");!mode.empty()) {
        require(mode=="1","reconstruction startup fixture option");
        configure(true);
        auto a=target.create_context(true,true,true),b=target.create_context(true,true,true);
        const auto a_bytes=a->persistent_bytes(),b_bytes=b->persistent_bytes();
        std::array<Exl3TextContext*,2> contexts{a.get(),b.get()};
        bool deferred_refused=false;
        try{a->prefill(std::span<const std::int64_t>(source.data(),1));}
        catch(const std::exception& error){deferred_refused=std::string(error.what()).find("reservation not installed")!=std::string::npos;}
        require(deferred_refused,"uninstalled reconstruction context executed prefill");
        using Cache=ninfer::exl3::Exl3VeriCacheServingPrefixCache;
        using Coordinator=ninfer::exl3::Exl3VeriCacheServingCoordinator;
        using Inventory=ninfer::exl3::Exl3ResourceInventory;
        Cache cache({2,64,1ULL<<30,1ULL<<30},{"recon-startup","fixture-model","fixture-tokenizer","C2","text"});
        if(const auto fallback=env("NINFER_TEST_RECON_BUDGET_FALLBACK");!fallback.empty()) {
            require(fallback=="1","reconstruction fallback fixture option");
            Coordinator exhausted(cache,{2,2,1ULL<<30,1ULL<<30});
            auto no_slabs=Inventory::unlimited();
            no_slabs[static_cast<unsigned>(Inventory::Domain::device)]=0;
            exhausted.bind_physical_resources({},no_slabs);
            Exl3TextContext::reserve_reconstruction_lanes(exhausted,contexts,true);
            for(auto* context:contexts)require(context->reconstructed_exact_stats().reservation_fallback &&
                context->reconstructed_exact_stats().workspace_bytes==0,"budget fallback allocated reconstruction slab");
            require(a->persistent_bytes()==a_bytes && b->persistent_bytes()==b_bytes,
                "budget fallback changed persistent device allocation");
            configure(false);auto control=target.create_context(true);
            const auto short_prompt=std::span<const std::int64_t>(source.data(),16);
            control->prefill(short_prompt);a->prefill(short_prompt);b->prefill(short_prompt);
            const auto expected=control->export_exact_host_state();
            require(expected->same_payload(*a->export_exact_host_state()) &&
                expected->same_payload(*b->export_exact_host_state()),"budget fallback changed direct-path state");
            return;
        }
        const auto metadata_bytes=Exl3TextContext::reconstruction_backing_metadata_bytes();
        for(bool metadata_short:{false,true}) {
            Coordinator low(cache,{2,2,1ULL<<30,1ULL<<30});
            auto limits=Inventory::unlimited();
            limits[static_cast<unsigned>(Inventory::Domain::device)]=2*expected_slab_bytes-(metadata_short?0:1);
            limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=2*metadata_bytes-(metadata_short?1:0);
            low.bind_physical_resources({},limits);
            bool refused=false;
            try{Exl3TextContext::reserve_reconstruction_lanes(low,contexts);}
            catch(const std::exception&){refused=true;}
            require(refused && a->reconstructed_exact_stats().workspace_bytes==0 &&
                b->reconstructed_exact_stats().workspace_bytes==0 && a->persistent_bytes()==a_bytes &&
                b->persistent_bytes()==b_bytes,"short C2 reservation installed partial backing");
        }
        Coordinator exact(cache,{2,2,1ULL<<30,1ULL<<30});
        auto limits=Inventory::unlimited();
        limits[static_cast<unsigned>(Inventory::Domain::device)]=2*expected_slab_bytes;
        limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=2*metadata_bytes;
        exact.bind_physical_resources({},limits);
        std::array<Exl3TextContext*,2> duplicate{a.get(),a.get()};
        bool duplicate_refused=false;
        try{Exl3TextContext::reserve_reconstruction_lanes(exact,duplicate);}
        catch(const std::exception&){duplicate_refused=true;}
        require(duplicate_refused,"duplicate reconstruction lane admitted");
        std::array<Exl3TextContext*,2> missing{a.get(),nullptr};
        bool missing_refused=false;
        try{Exl3TextContext::reserve_reconstruction_lanes(exact,missing);}
        catch(const std::runtime_error& error){missing_refused=std::string(error.what())==
            "reconstruction reservation requires pristine context";}
        require(missing_refused && a->reconstructed_exact_stats().workspace_bytes==0 &&
            b->reconstructed_exact_stats().workspace_bytes==0 && a->persistent_bytes()==a_bytes &&
            b->persistent_bytes()==b_bytes,"missing peer changed reconstruction ownership");
        if(const auto cleanup=env("NINFER_TEST_RECON_STARTUP_CLEANUP_FAILURE");!cleanup.empty()) {
            require(cleanup=="1","reconstruction rollback cleanup fixture option");
            const auto before=Exl3TextContext::reconstruction_quarantined_allocations();
            auto expected_witness=Exl3TextContext::retirement_quarantine_witness();
            ++expected_witness[2];
            b->fail_reconstruction_allocation_for_test(true);
            bool injected=false;
            try{Exl3TextContext::reserve_reconstruction_lanes(exact,contexts);}
            catch(const std::runtime_error& error){injected=std::string(error.what()).find("injected reconstruction backing allocation failure")!=std::string::npos;}
            require(injected && Exl3TextContext::reconstruction_quarantined_allocations()==before+1 &&
                a->reconstructed_exact_stats().workspace_bytes==0 && b->reconstructed_exact_stats().workspace_bytes==0,
                "rollback cleanup failure lost pending allocation or installed partial contexts");
            require(Exl3TextContext::retirement_quarantine_witness()==expected_witness,
                "reconstruction rollback changed unrelated quarantine components");
            bool refused=false;
            try{Exl3TextContext::reserve_reconstruction_lanes(exact,contexts);}
            catch(const std::exception& error){refused=std::string(error.what()).find("allocation cleanup")!=std::string::npos;}
            require(refused,"failed rollback cleanup admitted another reservation");
            Inventory::Requirement metadata;metadata.configuration=97;
            metadata.add(Inventory::Domain::host_metadata,1,sizeof(int));
            bool factory_called=false,sealed=false;
            try{exact.allocate_startup_resources(metadata,[&](auto) {
                factory_called=true;return Inventory{};
            });}catch(const std::logic_error& error){sealed=std::string(error.what())==
                "serving coordinator sealed after failed startup retirement";}
            require(sealed && !factory_called,"reconstruction cleanup failure left authority reusable");
            return;
        }
        for(auto* failing:contexts) {
            failing->fail_reconstruction_allocation_for_test();
            bool injected=false;
            try{Exl3TextContext::reserve_reconstruction_lanes(exact,contexts,true);}
            catch(const std::runtime_error& error) {
                injected=std::string(error.what()).find("injected reconstruction backing allocation failure")!=std::string::npos;
            }
            require(injected && a->reconstructed_exact_stats().workspace_bytes==0 &&
                b->reconstructed_exact_stats().workspace_bytes==0 && a->persistent_bytes()==a_bytes &&
                b->persistent_bytes()==b_bytes,"factory failure published partial reconstruction lanes");
        }
        Exl3TextContext::reserve_reconstruction_lanes(exact,contexts);
        require(a->reconstructed_exact_stats().workspace_bytes==expected_slab_bytes &&
            b->reconstructed_exact_stats().workspace_bytes==expected_slab_bytes &&
            a->persistent_bytes()==a_bytes+expected_slab_bytes && b->persistent_bytes()==b_bytes+expected_slab_bytes,
            "exact C2 reconstruction reservation did not install both backings");
        bool repeated_refused=false;
        try{Exl3TextContext::reserve_reconstruction_lanes(exact,contexts);}
        catch(const std::runtime_error& error){repeated_refused=std::string(error.what())==
            "reconstruction backing already installed";}
        require(repeated_refused && a->reconstructed_exact_stats().workspace_bytes==expected_slab_bytes &&
            b->reconstructed_exact_stats().workspace_bytes==expected_slab_bytes &&
            a->persistent_bytes()==a_bytes+expected_slab_bytes && b->persistent_bytes()==b_bytes+expected_slab_bytes,
            "repeated reconstruction reservation changed installed ownership or byte totals");
        return;
    }
    const auto prompt=std::span<const std::int64_t>(source.data(),4096);
    std::array<std::shared_ptr<const State>,3> roots;
    std::array<std::vector<std::uint16_t>,129> logits;
    std::array<std::array<std::vector<std::uint16_t>,5>,129> taps;
    std::vector<std::int64_t> tokens;
    const auto bits=[](Exl3TextContext& ctx) {
        return target_continue_device_bits(ctx.logits_device(),kVocab,"exact route logits");
    };
    const auto write_check=[&](const char* phase,int step,bool pass,Exl3TextContext& ctx,
                              int exports,std::size_t lv=0,std::size_t tv=0) {
        const auto s=ctx.reconstructed_exact_stats();
        checks << phase << ',' << step << ',' << pass << ',' << lv << ',' << tv << ','
            << exports << ',' << s.calls << ',' << s.rows << ',' << s.workspace_bytes
            << ',' << ctx.persistent_bytes() << ',' << ctx.paired_transform_submissions()
            << ',' << ctx.fused_gate_up_submissions() << ',' << ctx.fused_residual_norm_submissions() << '\n';
        checks.flush(); require(checks.good(),"exact route evidence write");
        require(pass,std::string("exact route zero-difference gate: ")+phase);
    };
    // Copy retained roots' actual bytes once so shared pages cannot hide mutation.
    const auto payload=[](const std::shared_ptr<const State>& state) {
        std::vector<std::uint8_t> bytes;
        bytes.reserve(state->payload_bytes());
        const std::array<std::shared_ptr<const State>,1> states{state};
        State::visit_host_allocations(states,[&](const void* data,std::size_t size) {
            const auto* p=static_cast<const std::uint8_t*>(data);
            bytes.insert(bytes.end(),p,p+size);
        },false);
        return bytes;
    };
    std::array<std::vector<std::uint8_t>,2> frozen;
    std::size_t control_persistent=0;
    std::uint64_t eligible_calls=0,eligible_rows=0;
    struct Observed {
        bool include_k6_down=false,include_k6_gate_up=false;
        std::uint64_t calls=0,rows=0,k6_calls=0,k6_rows=0;
    } observed;
    observed.include_k6_down=include_k6_down;
    observed.include_k6_gate_up=include_k6_gate_up;
    struct ResidualIntermediate {
        int layer=-1,position=-1,rows=0;
        bool compare=false,matched=false;
        std::vector<std::uint16_t> residual,normalized;
    } residual_intermediate;
    const auto residual_observer=+[](const ninfer::exl3::Exl3LayerObservation& x,void* user) {
        auto& held=*static_cast<ResidualIntermediate*>(user);
        if(x.rows<16 || !x.gdn.post_attention_residual || !x.gdn.mlp_input)return;
        if(held.layer>=0 && (x.layer!=held.layer || x.position!=held.position || x.rows!=held.rows))return;
        if(!held.compare && held.layer>=0)return;
        cuda_check(cudaStreamSynchronize(x.stream),"residual intermediate observation completion");
        constexpr std::size_t count=16ull*5120;
        std::vector<std::uint16_t> residual(count),normalized(count);
        cuda_check(cudaMemcpy(residual.data(),x.gdn.post_attention_residual,count*2,cudaMemcpyDeviceToHost),"captured represented residual");
        cuda_check(cudaMemcpy(normalized.data(),x.gdn.mlp_input,count*2,cudaMemcpyDeviceToHost),"captured normalized MLP input");
        if(held.compare) {
            require(residual==held.residual && normalized==held.normalized,"fused residual changed captured intermediate bits");
            held.matched=true;
        } else {
            held.layer=x.layer;held.position=x.position;held.rows=x.rows;
            held.residual=std::move(residual);held.normalized=std::move(normalized);
        }
    };
    configure(false);
    {
        auto ctx=target.create_context(true);ctx->prepare_continuation(8);
        if(residual_context)ctx->set_layer_observer_for_test(residual_observer,&residual_intermediate);
        ctx->set_target_projection_observer_for_test(
            [](const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
                auto& o=*static_cast<Observed*>(user);
                if((x.rows!=256 && x.rows!=512 && x.rows!=1024) || x.metadata.mcg ||
                   !x.metadata.mul1 || x.metadata.has_bias || !x.operation)return;
                const auto operation=std::string_view(x.operation);
                const bool down=(x.metadata.K==7 || (o.include_k6_down && x.metadata.K==6)) &&
                    x.metadata.in_features==17408 && x.metadata.out_features==5120 && operation=="down";
                const bool gate_up=o.include_k6_gate_up && x.metadata.K==6 &&
                    x.metadata.in_features==5120 && x.metadata.out_features==17408 &&
                    (operation=="gate" || operation=="up");
                if(!down && !gate_up)return;
                ++o.calls;o.rows+=x.rows;
                if(x.metadata.K==6){++o.k6_calls;o.k6_rows+=x.rows;}
            },&observed,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
        t69b_ingest(*ctx,prompt);ctx->set_target_projection_observer_for_test(nullptr);
        if(residual_context) {
            ctx->set_layer_observer_for_test(nullptr);
            require(residual_intermediate.layer>=0,"residual control missed held GDN intermediate");
        }
        if(reuse_context)require(ctx->paired_transform_submissions()==0,
            "separate-transform control submitted paired transforms");
        if(fused_context)require(ctx->fused_gate_up_submissions()==0,
            "unfused control submitted fused gate/up work");
        if(residual_context)require(ctx->fused_residual_norm_submissions()==0,"residual control enabled fusion");
        control_persistent=ctx->persistent_bytes();
        eligible_calls=observed.calls;eligible_rows=observed.rows;
        int exports=0;
        for(int step=0;step<=128;++step) {
            logits[step]=bits(*ctx);taps[step]=ctx->exact_tap_rows_host();
            if(step==0 || step==64 || step==128) {
                const int index=step/64;roots[index]=ctx->export_exact_host_state();++exports;
                if(index<2) frozen[index]=payload(roots[index]);
            }
            if(step<128) {const auto token=sample_target(*ctx);tokens.push_back(token);ctx->decode(token);}
        }
        const auto s=ctx->reconstructed_exact_stats();
        write_check("parent",128,s.calls==0 && s.rows==0 && s.workspace_bytes==0 && eligible_calls>0,*ctx,exports);
        require(s.k6_down_calls==0 && s.k6_down_rows==0 &&
            s.k6_gate_up_calls==0 && s.k6_gate_up_rows==0 &&
            (!include_k6 || observed.k6_calls>0),
            "exact route K6 inactive accounting or missing real K6 sites");
    }
    configure(true);
    if(reuse_context) {
        auto diagnostic=target.create_context(true);
        diagnostic->prepare_continuation(8);
        std::uint64_t observations=0;
        diagnostic->set_target_projection_observer_for_test(
            [](const ninfer::exl3::Exl3TargetProjectionObservation&,void* user) {
                ++*static_cast<std::uint64_t*>(user);
            },&observations,nullptr,ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
        const auto submissions_before=diagnostic->paired_transform_submissions();
        const auto residual_before=diagnostic->fused_residual_norm_submissions();
        const auto fused_before=diagnostic->fused_gate_up_submissions();
        t69b_ingest(*diagnostic,prompt);
        diagnostic->set_target_projection_observer_for_test(nullptr);
        require(observations>0,"paired-transform diagnostic exclusion missed observer callbacks");
        if(residual_context)require(diagnostic->fused_residual_norm_submissions()==residual_before,
            "residual fusion bypassed projection observer");
        require(diagnostic->paired_transform_submissions()==submissions_before,
            "paired-transform candidate bypassed observer exclusion");
        if(fused_context)require(diagnostic->fused_gate_up_submissions()==fused_before,
            "fused gate/up candidate bypassed observer exclusion");
        auto state=diagnostic->export_exact_host_state();
        write_check("transform_observer_payload",0,state->same_payload(*roots[0]),*diagnostic,1);
        write_check("transform_observer_logits_taps",0,
            bits(*diagnostic)==logits[0] && diagnostic->exact_tap_rows_host()==taps[0],*diagnostic,1);
        // The subsequent ordinary candidate exercises the same options with no
        // observer and must record actual paired submissions. No hit-rate claim.
    }
    if(reuse_context) {
        _putenv_s("NINFER_EXL3_TARGET_PROJECTION_TIMING","1");
        auto diagnostic=target.create_context(true);
        _putenv_s("NINFER_EXL3_TARGET_PROJECTION_TIMING","0");
        require(diagnostic->target_projection_timing_enabled(),"transform timing option not latched");
        diagnostic->prepare_continuation(8);
        diagnostic->prepare_target_projection_timing();
        // Attach the real diagnostic owner without starting a measurement round.
        // This checks exclusion and semantics, not timing values or performance.
        const auto before=diagnostic->paired_transform_submissions();
        const auto residual_before=diagnostic->fused_residual_norm_submissions();
        const auto fused_before=diagnostic->fused_gate_up_submissions();
        t69b_ingest(*diagnostic,prompt);
        require(diagnostic->paired_transform_submissions()==before,
            "paired transforms bypassed prepared projection timing owner");
        if(residual_context)require(diagnostic->fused_residual_norm_submissions()==residual_before,
            "residual fusion bypassed projection timing owner");
        if(fused_context)require(diagnostic->fused_gate_up_submissions()==fused_before,
            "fused gate/up bypassed prepared projection timing owner");
        auto state=diagnostic->export_exact_host_state();
        write_check("transform_timing_payload",0,state->same_payload(*roots[0]),*diagnostic,1);
        write_check("transform_timing_logits_taps",0,
            bits(*diagnostic)==logits[0] && diagnostic->exact_tap_rows_host()==taps[0],*diagnostic,1);
    }
    {
        auto ctx=target.create_context(true);ctx->prepare_continuation(8);
        if(residual_context) {
            residual_intermediate.compare=true;
            ctx->set_layer_observer_for_test(residual_observer,&residual_intermediate);
        }
        t69b_ingest(*ctx,prompt);
        if(residual_context) {
            ctx->set_layer_observer_for_test(nullptr);
            require(residual_intermediate.matched,"residual candidate missed matching intermediate provenance");
            std::ofstream provenance(directory/"residual_intermediate_site.txt");
            provenance << "layer=" << residual_intermediate.layer << '\n'
                << "position=" << residual_intermediate.position << '\n'
                << "submitted_rows=" << residual_intermediate.rows << "\ncompared_rows=16\n";
            provenance.flush();require(provenance.good(),"residual intermediate provenance write");
        }
        if(reuse_context)require(ctx->paired_transform_submissions()>0,
            "composed transform context missed actual GDN paired submissions");
        if(fused_context)require(ctx->fused_gate_up_submissions()>0,
            "composed gate/up context missed actual fused submissions");
        if(residual_context)require(ctx->fused_residual_norm_submissions()>0,"candidate missed actual fused residual work");
        int exports=0;
        const auto verify_row=[&](const char* phase,int step) {
            const auto got=bits(*ctx);const auto got_taps=ctx->exact_tap_rows_host();
            std::size_t tap_values=0;for(const auto& t:got_taps) tap_values+=t.size();
            write_check(phase,step,got==logits[step] && got_taps==taps[step],*ctx,exports,got.size(),tap_values);
        };
        for(int step=0;step<=128;++step) {
            verify_row("teacher",step);
            if(step==0 || step==64 || step==128) {
                auto state=ctx->export_exact_host_state();++exports;
                write_check("full_payload",step,state->same_payload(*roots[step/64]),*ctx,exports);
            }
            if(step<128) ctx->decode(tokens[step]);
        }
        const auto s=ctx->reconstructed_exact_stats();
        write_check("accounting",128,s.calls==eligible_calls && s.rows==eligible_rows &&
            s.workspace_bytes==expected_slab_bytes && ctx->persistent_bytes()==control_persistent+expected_slab_bytes,
            *ctx,exports);
        const bool k6_accounting=include_k6_gate_up?
            (s.k6_gate_up_calls==observed.k6_calls && s.k6_gate_up_rows==observed.k6_rows &&
             s.k6_down_calls==0 && s.k6_down_rows==0):
            (s.k6_down_calls==observed.k6_calls && s.k6_down_rows==observed.k6_rows &&
             s.k6_gate_up_calls==0 && s.k6_gate_up_rows==0);
        write_check("k6_accounting",128,k6_accounting,*ctx,exports);
        // Restore the old prompt root after the suffix, then the mid-decode root.
        for(int root_index:{0,1}) {
            ctx->restore_exact_host_state(*roots[root_index]);
            const int start=root_index*64;
            auto restored=ctx->export_exact_host_state();++exports;
            write_check("restore_payload",start,restored->same_payload(*roots[root_index]),*ctx,exports);
            for(int step=start;step<=128;++step) {
                verify_row("restore_replay",step);
                if(step<128) ctx->decode(tokens[step]);
            }
            auto replayed=ctx->export_exact_host_state();++exports;
            write_check("replay_payload",128,replayed->same_payload(*roots[2]),*ctx,exports);
            write_check("roots_immutable",start,payload(roots[0])==frozen[0] &&
                payload(roots[1])==frozen[1],*ctx,exports);
        }
    }
    // The composed transform mode uses the full payload/logit/tap and restored
    // teacher-forced trajectory above. It does not enter the timing/fault arms.
    // Equal-scale hit incidence is deliberately not inferred from state parity.
    if(fused_context) {
        std::vector<std::int64_t> second_prompt(prompt.begin(),prompt.end());
        std::reverse(second_prompt.begin(),second_prompt.end());
        require(!std::equal(second_prompt.begin(),second_prompt.end(),prompt.begin()),
            "second fused request must have distinct prompt provenance");
        std::array<std::vector<std::uint16_t>,9> second_logits;
        std::array<std::array<std::vector<std::uint16_t>,5>,9> second_taps;
        std::vector<std::int64_t> second_tokens;
        std::shared_ptr<const State> second_final;
        configure(false);
        {
            auto control=target.create_context(true);control->prepare_continuation(8);
            t69b_ingest(*control,second_prompt);
            for(int step=0;step<=8;++step) {
                second_logits[step]=bits(*control);second_taps[step]=control->exact_tap_rows_host();
                if(step<8){const auto token=sample_target(*control);second_tokens.push_back(token);control->decode(token);}
            }
            require(control->fused_gate_up_submissions()==0,"second control enabled fusion");
            second_final=control->export_exact_host_state();
        }
        configure(true);
        {
            auto candidate=target.create_context(true);candidate->prepare_continuation(8);
            t69b_ingest(*candidate,second_prompt);
            require(candidate->fused_gate_up_submissions()>0,"second request missed fused GDN path");
            for(int step=0;step<=8;++step) {
                write_check("fused_second_request",step,bits(*candidate)==second_logits[step] &&
                    candidate->exact_tap_rows_host()==second_taps[step],*candidate,0);
                if(step<8)candidate->decode(second_tokens[step]);
            }
            auto state=candidate->export_exact_host_state();
            write_check("fused_second_payload",8,state->same_payload(*second_final),*candidate,1);
            write_check("fused_prior_roots_immutable",8,payload(roots[0])==frozen[0] &&
                payload(roots[1])==frozen[1],*candidate,1);
        }
        std::ofstream second_identity(directory/"fused_second_prompt_tokens.txt");
        for(auto token:second_prompt)second_identity << token << '\n';
        second_identity.flush();require(second_identity.good(),"second fused request provenance write");
    }
    if(reuse_context)return;
    if(include_k6) {
        const auto failure=env("NINFER_TEST_RECON_RETIRE_FAILURE");
        require(failure.empty() || failure=="0" || failure=="1","reconstruction retirement fixture option");
        if(failure=="1") {
            const auto before=Exl3TextContext::reconstruction_quarantined_contexts();
            auto ctx=target.create_context(true);t69b_ingest(*ctx,prompt);
            const auto s=ctx->reconstructed_exact_stats();
            require((include_k6_gate_up?s.k6_gate_up_calls:s.k6_down_calls)>0,
                "retirement fixture missed K6 reconstruction");
            require(!ctx->reconstruction_retirement_uncertain(),"healthy reconstruction context reported uncertain retirement");
            ctx->fail_reconstruction_retirement_for_test();
            require(ctx->reconstruction_retirement_uncertain(),"armed reconstruction failure missing from close admission");
            ctx.reset();
            require(Exl3TextContext::reconstruction_quarantined_contexts()==before+1,
                "failed reconstruction retirement did not retain context");
            bool refused=false;
            try{auto retry=target.create_context(true);}
            catch(const std::exception&){refused=true;}
            require(refused,"unresolved reconstruction retirement admitted new context");
        }
        return; // Newly prepared family covers correctness, not timing.
    }
    // Timings contain no intermediate snapshots, observers, tap/logit comparisons,
    // or teacher forcing. Seed/TTFT ends at first greedy selection (target-only).
    struct Run { double construction=0,prefill=0,seed=0,ttft=0,decode=0,export_ms=0,teardown=0,wall=0;
        std::uint64_t calls=0,rows=0;std::size_t workspace=0,persistent=0;
        bool tokens_exact=false,state_exact=false,accounting=false; };
    auto run=[&](bool candidate) {
        configure(candidate);Run r;std::vector<std::int64_t> generated;generated.reserve(128);
        cuda_check(cudaDeviceSynchronize(),"exact route timing start");const auto begin=Clock::now();
        auto ctx=target.create_context(true);ctx->prepare_continuation(8);
        cuda_check(cudaDeviceSynchronize(),"exact route construction");const auto constructed=Clock::now();
        t69b_ingest(*ctx,prompt);cuda_check(cudaDeviceSynchronize(),"exact route prefill");const auto prefilled=Clock::now();
        auto token=sample_target(*ctx);const auto seeded=Clock::now();
        for(int i=0;i<128;++i) {
            if(i) token=sample_target(*ctx);
            generated.push_back(token);ctx->decode(token);
        }
        cuda_check(cudaDeviceSynchronize(),"exact route decode");const auto decoded=Clock::now();
        auto state=ctx->export_exact_host_state();cuda_check(cudaDeviceSynchronize(),"exact route final export");
        const auto exported=Clock::now();
        const auto s=ctx->reconstructed_exact_stats();r.calls=s.calls;r.rows=s.rows;r.workspace=s.workspace_bytes;
        r.persistent=ctx->persistent_bytes();
        ctx.reset();cuda_check(cudaDeviceSynchronize(),"exact route teardown");const auto ended=Clock::now();
        r.construction=milliseconds(begin,constructed);r.prefill=milliseconds(constructed,prefilled);
        r.seed=milliseconds(prefilled,seeded);r.ttft=milliseconds(begin,seeded);
        r.decode=milliseconds(seeded,decoded);r.export_ms=milliseconds(decoded,exported);
        r.teardown=milliseconds(exported,ended);r.wall=milliseconds(begin,ended);
        r.tokens_exact=generated==tokens;r.state_exact=state->same_payload(*roots[2]);
        r.accounting=candidate?(r.calls==eligible_calls && r.rows==eligible_rows &&
            r.workspace==expected_slab_bytes && r.persistent==control_persistent+expected_slab_bytes):
            (r.calls==0 && r.rows==0 && r.workspace==0 && r.persistent==control_persistent);
        return r;
    };
    for(int pair=0;pair<6;++pair) {
        for(int order=0;order<2;++order) {
            const bool candidate=(pair%2)?order==0:order==1;
            const auto r=run(candidate);
            timings << pair << ',' << (pair%2?"BA":"AB") << ',' << (candidate?"B":"A") << ','
                << r.construction << ',' << r.prefill << ',' << r.seed << ',' << r.ttft << ','
                << r.decode << ',' << r.export_ms << ',' << r.teardown << ',' << r.wall << ",1,"
                << r.calls << ',' << r.rows << ',' << r.workspace << ',' << r.persistent << ','
                << r.tokens_exact << ',' << r.state_exact << ',' << r.accounting << '\n';
            timings.flush();require(timings.good(),"exact route timing evidence write");
            require(r.tokens_exact && r.state_exact && r.accounting,"exact route timed identity/accounting gate");
        }
    }
    const bool immutable=payload(roots[0])==frozen[0] && payload(roots[1])==frozen[1];
    checks << "roots_after_timing,128," << immutable << ",0,0,10,0,0,0,0\n";
    checks.flush();require(checks.good(),"exact route final evidence write");
    require(immutable,"exact route timed work mutated retained roots");
    std::cout << "RECONSTRUCTED_EXACT_ROUTE PASS prompt=4096 decode=128 pairs=6 correctness_exports=10 timed_exports=12 target_only=1 no_performance_gate=1\n";
}
