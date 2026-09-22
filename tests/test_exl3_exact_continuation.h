#pragma once

void run_exact_continuation_qualification(Exl3TextModel& target,
    const std::vector<std::int64_t>& source) {
    require(source.size()>=350 && target.max_context()==1024,"exactcontinue fixture extent");
    auto batch=target.create_context(true),serial=target.create_context(true);
    batch->prepare_continuation(8);
    int payload_cases=0,repair_cases=0,pairs=0;
    for(int prefix:{16,321}) {
        for(auto* ctx:{batch.get(),serial.get()}) {
            ctx->reset(); ctx->prefill(std::span<const std::int64_t>(source.data(),16));
            for(int i=16;i<prefix;++i) ctx->decode(source[i]);
        }
        const auto root=batch->export_exact_host_state(),oracle_root=serial->export_exact_host_state();
        require(root->same_payload(*oracle_root),"exactcontinue independently ingested root");
        for(int width:{2,4,8}) {
            const std::span<const std::int64_t> teacher(source.data()+prefix,width);
            batch->restore_exact_host_state(*root);
            serial->restore_exact_host_state(*oracle_root);
            batch->continue_rows(teacher);
            const auto logits=batch->continuation_logits_bits_host();
            const auto embedding=batch->embedding_bits_host_for_test();
            std::array<std::vector<std::uint16_t>,5> taps;
            for(int tap=0;tap<5;++tap) taps[tap]=target_continue_tap_bits(*batch,kTapLayers[tap],width);
            for(int row=0;row<width;++row) {
                serial->decode(teacher[row]);
                const auto row_logits=target_continue_device_bits(serial->logits_device(),kVocab,"exactcontinue serial logits");
                require(std::equal(row_logits.begin(),row_logits.end(),logits.begin()+row*kVocab),
                        "exactcontinue attempted logits row="+std::to_string(row));
                const auto row_embedding=serial->embedding_bits_host_for_test();
                require(std::equal(row_embedding.begin(),row_embedding.end(),embedding.begin()+row*kHidden),
                        "exactcontinue attempted embedding");
                for(int tap=0;tap<5;++tap) {
                    const auto values=target_continue_tap_bits(*serial,kTapLayers[tap],1);
                    require(std::equal(values.begin(),values.end(),taps[tap].begin()+row*kHidden),
                            "exactcontinue attempted tap row="+std::to_string(row));
                }
            }
            batch->finish_exact_continuation();
            const auto expected=serial->export_exact_host_state();
            require(batch->export_exact_host_state()->same_payload(*expected),"exactcontinue complete state");
            ++payload_cases;
            for(int retain:(width==2?std::vector<int>{1}:std::vector<int>{1,width-1})) {
                batch->restore_exact_host_state(*root);
                if(retain==1) batch->decode(teacher[0]);
                else {batch->continue_rows(teacher.first(retain)); batch->finish_exact_continuation();}
                serial->restore_exact_host_state(*oracle_root);
                for(int i=0;i<retain;++i) serial->decode(teacher[i]);
                require(batch->export_exact_host_state()->same_payload(*serial->export_exact_host_state()),
                        "exactcontinue rejected-suffix repair");
                const auto replacement=(teacher[retain]+1)%kVocab;
                batch->decode(replacement); serial->decode(replacement);
                require(batch->export_exact_host_state()->same_payload(*serial->export_exact_host_state()),
                        "exactcontinue replacement continuation");
                ++repair_cases;
            }
            for(int repeat=0;repeat<3;++repeat) {
                double timings[2]{};
                for(int order=0;order<2;++order) {
                    const int arm=(repeat&1)?1-order:order;
                    auto& ctx=arm?*batch:*serial;
                    ctx.restore_exact_host_state(arm?*root:*oracle_root);
                    cuda_check(cudaDeviceSynchronize(),"exactcontinue timed start");
                    const auto start=std::chrono::steady_clock::now();
                    if(arm) {ctx.continue_rows(teacher); ctx.finish_exact_continuation();}
                    else for(auto token:teacher) ctx.decode(token);
                    cuda_check(cudaDeviceSynchronize(),"exactcontinue timed end");
                    timings[arm]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    require(ctx.export_exact_host_state()->same_payload(*expected),"exactcontinue paired state");
                }
                std::cout << "EXACT_CONTINUATION_PAIR prefix=" << prefix << " rows=" << width
                          << " repeat=" << repeat << " serial_ms=" << timings[0]
                          << " batched_ms=" << timings[1] << '\n';
                ++pairs;
            }
        }
    }
    std::cout << "EXACT_CONTINUATION PASS payload_cases=" << payload_cases << " repair_cases=" << repair_cases
              << " pairs=" << pairs << " identity=vericache_exact_fp16_eager\n";
}
