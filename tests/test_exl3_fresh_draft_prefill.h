#pragma once

// Test-owned full bytes; no new production download surface.
struct FreshDraftObservation {
    std::vector<std::uint16_t> bytes;
    std::vector<int> metadata;
    static void observe(const Exl3Dflash2DraftModel::RingAttentionObservation& o, void* user) {
        auto& x=*static_cast<FreshDraftObservation*>(user);
        require(o.layer>=0 && o.layer<5 && o.layer==static_cast<int>(x.metadata.size()/5) && o.queries==8 && o.block==8 && o.count>=0 && o.count<=2047 && o.start>=0 && o.start<2048,"fresh observer identity/extent");
        x.metadata.insert(x.metadata.end(),{o.layer,o.start,o.count,o.queries,o.block});
        const std::array<const std::uint16_t*,6> ptr{o.q,o.ring_k,o.ring_v,o.k,o.v,o.output};
        const std::array<std::size_t,6> sizes{std::size_t(o.queries)*4096,2048u*1024,2048u*1024,
            std::size_t(o.block)*1024,std::size_t(o.block)*1024,std::size_t(o.queries)*4096};
        cuda_check(cudaStreamSynchronize(o.stream),"fresh draft observation ready");
        for(int i=0;i<6;++i){const auto old=x.bytes.size();x.bytes.resize(old+sizes[i]);
            cuda_check(cudaMemcpy(x.bytes.data()+old,ptr[i],sizes[i]*2,cudaMemcpyDeviceToHost),"fresh draft observation bytes");}
    }
};
struct FreshDraftProposal {
    FreshDraftObservation attention;
    std::vector<std::uint16_t> head,logits;
    std::vector<std::int64_t> topk,proposal;
    std::vector<float> values;
    long long base=0;int count=0;
};
inline void fresh_draft_equal(const FreshDraftProposal& a,const FreshDraftProposal& b,const std::string& label) {
    const bool exact = a.base==b.base && a.count==b.count && a.attention.metadata==b.attention.metadata &&
        a.attention.bytes==b.attention.bytes && a.head==b.head && a.logits==b.logits &&
        a.topk==b.topk && a.proposal==b.proposal && a.values.size()==b.values.size() &&
        std::memcmp(a.values.data(),b.values.data(),a.values.size()*sizeof(float))==0;
    if(!exact) {
        const auto root=std::filesystem::path(env("NINFER_E5A4_OUT"));
        auto save=[&](const std::string& name,const auto& values){std::ofstream f(root/(label+"-"+name+".bin"),std::ios::binary);f.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));require(f.good(),"fresh failure payload write");};
        save("reference-attention",a.attention.bytes);save("candidate-attention",b.attention.bytes);
        save("reference-head",a.head);save("candidate-head",b.head);save("reference-logits",a.logits);save("candidate-logits",b.logits);
        save("reference-topk",a.topk);save("candidate-topk",b.topk);save("reference-values",a.values);save("candidate-values",b.values);
    }
    require(exact,label+" full draft bytes");
}
void run_fresh_draft_prefill_qualification(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
                                         const std::vector<std::int64_t>& ids) {
    require(target.max_context()>=8192 && ids.size()>=4097,"fresh draft oracle requires8192 capacity and4097 real rows");
    require(env("NINFER_EXL3_TARGET_QKV_K6_SMALL_M")=="1","fresh draft oracle P9 flag1");
    const auto directory=std::filesystem::path(env("NINFER_E5A4_OUT"));
    require(!std::filesystem::exists(directory) && std::filesystem::create_directories(directory),"fresh draft new evidence directory");
    std::ofstream report(directory/"cases.csv");report<<"length,start,partition,submitted,encoded,skipped,physical_slots,layers,proposals,observed_bytes_per_proposal,exact\n";
    auto ctx=target.create_context(true);require(ctx->try_enable_oscar_from_environment(),"fresh draft target OSCAR");
    std::array<std::vector<std::uint16_t>,5> host;
    for(auto& x:host)x.resize(4097u*kHidden);
    DeviceBuffer download(1024u*kHidden*2);
    for(int first=0;first<4097;){const int n=first==0?16:std::min(1024,4097-first);
        const auto tokens=std::span<const std::int64_t>(ids.data()+first,n);
        if(first==0)ctx->prefill(tokens);else ctx->append_prefill_wide(tokens);
        for(int t=0;t<5;++t){ctx->copy_tap_rows_to_device(kTapLayers[t],0,static_cast<std::uint16_t*>(download.get()),n);
            cuda_check(cudaMemcpy(host[t].data()+std::size_t(first)*kHidden,download.get(),std::size_t(n)*kHidden*2,cudaMemcpyDeviceToHost),"fresh real taps");}
        first+=n;
    }
    std::array<std::unique_ptr<DeviceBuffer>,5> staging;
    std::array<const std::uint16_t*,5> ptr;
    for(int t=0;t<5;++t){staging[t]=std::make_unique<DeviceBuffer>(16u*kHidden*2);ptr[t]=static_cast<const std::uint16_t*>(staging[t]->get());}
    auto upload=[&](int first,int n){for(int t=0;t<5;++t)cuda_check(cudaMemcpy(staging[t]->get(),host[t].data()+std::size_t(first)*kHidden,std::size_t(n)*kHidden*2,cudaMemcpyHostToDevice),"fresh taps upload");};
    auto preseed=[&]{draft.reset();upload(0,16);for(int p=0;p<2048;p+=16)draft.commit_prefill_block(ptr.data(),16,p);draft.reset();};
    // One real pending token from4097 is reused for positional unit cases.
    // Actual short-prompt pending/proposal ancestry is exercised by assembled tests.
    const auto pending=sample_target(*ctx);
    auto proposal=[&](int pos){FreshDraftProposal x;x.base=draft.ring_base_abs();x.count=draft.ring_count();
        draft.set_ring_attention_observer_for_test(FreshDraftObservation::observe,&x.attention);
        try {std::vector<std::int64_t> block(8,kMaskToken);block[0]=pending;
            x.proposal=draft.propose_cached(block,pos,ctx->target_embedding(),ctx->target_lm_head_weights(),ctx->target_lm_head_metadata(),kMaskToken);
        }catch(...){draft.set_ring_attention_observer_for_test(nullptr);throw;}
        draft.set_ring_attention_observer_for_test(nullptr);
        require(x.attention.metadata.size()==25 && x.proposal.size()==7,"fresh draft observer coverage");
        for(int layer=0;layer<5;++layer){const auto i=layer*5;require(x.attention.metadata[i]==layer && x.attention.metadata[i+1]==(x.base&2047) && x.attention.metadata[i+2]==x.count && x.attention.metadata[i+3]==8 && x.attention.metadata[i+4]==8,"fresh observed logical window");}
        require(x.attention.bytes.size()==5u*(2u*8*4096+2u*2048*1024+2u*8*1024),"fresh observed full byte extent");
        x.head.resize(7u*kHidden);x.logits.resize(7u*kVocab);x.topk.resize(7u*16);x.values.resize(7u*16);
        cuda_check(cudaMemcpy(x.head.data(),draft.last_head_input_device_for_test(),x.head.size()*2,cudaMemcpyDeviceToHost),"fresh head");
        cuda_check(cudaMemcpy(x.logits.data(),draft.last_head_logits_device_for_test(),x.logits.size()*2,cudaMemcpyDeviceToHost),"fresh logits");
        cuda_check(cudaMemcpy(x.topk.data(),draft.last_topk_ids_device_for_test(),x.topk.size()*8,cudaMemcpyDeviceToHost),"fresh topk");
        cuda_check(cudaMemcpy(x.values.data(),draft.last_topk_values_device_for_test(),x.values.size()*4,cudaMemcpyDeviceToHost),"fresh topk values");return x;};
    int cases=0;
    for(int length:{32,2047,2048,2049,2065,4097})for(int variant:{0,1}) {
        const int start=variant?2051:0;const int end=start+length;
        std::vector<int> partition;for(int p=0;p<length;){const int n=std::min(variant?(p%3==0?7:13):16,length-p);partition.push_back(n);p+=n;}
        auto ingest=[&](bool scoped){preseed();if(scoped)draft.begin_fresh_prefill(start,end);int pos=0;
            for(int n:partition){upload(pos,n);draft.commit_prefill_block(ptr.data(),n,start+pos);pos+=n;}if(scoped)draft.finish_fresh_prefill();};
        ingest(false);const auto expected=proposal(end);const auto repeat=proposal(end);fresh_draft_equal(expected,repeat,"baseline repeat");
        upload(0,1);draft.commit_target_block(ptr.data(),1,end);const auto next=proposal(end+1);
        ingest(true);const auto status=draft.fresh_prefill_status();long long skip=0;int pos=0;
        for(int n:partition){pos+=n;if(start+pos<=end-2048)skip+=n;}
        require(!status.active && !status.failed && status.submitted_rows==length && status.skipped_rows==skip && status.encoded_rows==length-skip,"fresh native row coverage");
        const auto got=proposal(end);fresh_draft_equal(expected,got,"scoped first");fresh_draft_equal(got,proposal(end),"scoped repeat");
        upload(0,1);draft.commit_target_block(ptr.data(),1,end);fresh_draft_equal(next,proposal(end+1),"scoped next commit/proposal");
        report<<length<<','<<start<<','<<variant<<','<<length<<','<<status.encoded_rows<<','<<skip<<",2048,5,3,"<<got.attention.bytes.size()*2<<",1\n";++cases;
    }
    // Every invalid active scope is poisoned until reset, then ordinary reuse works.
    int rejected=0;auto reject=[&](auto operation){bool failed=false;try{operation();}catch(const std::exception&){failed=true;}require(failed,"fresh invalid operation accepted");++rejected;};
    upload(0,16);
    for(int bad=0;bad<8;++bad){draft.reset();draft.begin_fresh_prefill(0,32);
        if(bad==0)reject([&]{draft.finish_fresh_prefill();});
        if(bad==1)reject([&]{draft.begin_fresh_prefill(0,32);});
        if(bad==2)reject([&]{draft.commit_target_block(ptr.data(),1,0);});
        if(bad==3)reject([&]{draft.rewind_to(0);});
        if(bad==4)reject([&]{proposal(0);});
        if(bad==5)reject([&]{draft.commit_prefill_block(ptr.data(),16,1);});
        if(bad==6)reject([&]{draft.commit_prefill_block(nullptr,16,0);});
        if(bad==7)reject([&]{draft.commit_prefill_block(ptr.data(),17,0);});
        require(draft.fresh_prefill_status().failed,"fresh invalid scope not poisoned");draft.reset();}
    draft.commit_prefill_block(ptr.data(),16,0);reject([&]{draft.begin_fresh_prefill(16,32);});draft.reset();
    cudaStream_t other=nullptr;cuda_check(cudaStreamCreate(&other),"fresh alternate stream");
    draft.begin_fresh_prefill(0,32);
    reject([&]{draft.commit_prefill_block(ptr.data(),16,0,other);});
    require(draft.fresh_prefill_status().failed,"fresh wrong stream poison");draft.reset();
    cuda_check(cudaStreamBeginCapture(other,cudaStreamCaptureModeThreadLocal),"fresh capture begin");
    reject([&]{draft.begin_fresh_prefill(0,32,other);});
    cudaGraph_t graph=nullptr;cuda_check(cudaStreamEndCapture(other,&graph),"fresh capture end");
    if(graph)cuda_check(cudaGraphDestroy(graph),"fresh capture graph destroy");
    draft.begin_fresh_prefill(0,32);reject([&]{draft.finish_fresh_prefill(other);});
    reject([&]{draft.commit_prefill_block(ptr.data(),16,0);});
    reject([&]{draft.reset(other);});reject([&]{draft.finish_fresh_prefill();});
    require(draft.fresh_prefill_status().failed,"fresh failed scope remained poisoned");draft.reset();
    cuda_check(cudaStreamDestroy(other),"fresh alternate stream destroy");
    draft.reset();draft.begin_fresh_prefill(0,32);draft.commit_prefill_block(ptr.data(),16,0);
    draft.reset();require(!draft.fresh_prefill_status().active && !draft.fresh_prefill_status().failed,"fresh partial reset cancellation");
    reject([&]{draft.begin_fresh_prefill(32,32);});
    reject([&]{draft.begin_fresh_prefill(2147483640LL,2147483648LL);});
    draft.begin_fresh_prefill(0,8);reject([&]{draft.commit_prefill_block(ptr.data(),16,0);});draft.reset();
    // Exercise the one-row commit fallback in a valid32-row proposal context.
    // Both arms retain the exact same16/15/1 arithmetic partition.
    auto one_row_partition=[&](bool scoped){preseed();if(scoped)draft.begin_fresh_prefill(2051,2083);
        upload(0,16);draft.commit_prefill_block(ptr.data(),16,2051);
        upload(16,15);draft.commit_prefill_block(ptr.data(),15,2067);
        upload(31,1);draft.commit_prefill_block(ptr.data(),1,2082);
        if(scoped)draft.finish_fresh_prefill();};
    std::cout<<"FRESH_DRAFT_PREFILL_STAGE one_row_baseline_begin length=32 partition=16,15,1"<<std::endl;
    one_row_partition(false);const auto one_ref=proposal(2083);
    std::cout<<"FRESH_DRAFT_PREFILL_STAGE one_row_candidate_begin length=32 partition=16,15,1"<<std::endl;
    one_row_partition(true);fresh_draft_equal(one_ref,proposal(2083),"one row commit in32 scope");
    require(draft.fresh_prefill_status().submitted_rows==32 && draft.fresh_prefill_status().encoded_rows==32 && draft.fresh_prefill_status().skipped_rows==0,"one row native coverage");draft.reset();
    report.flush();require(report.good() && cases==12 && rejected==18,"fresh oracle completion");
    std::cout<<"FRESH_DRAFT_PREFILL PASS cases=12 invalid=18 physical_slots=2048 layers=5 proposals=3 full_bytes=1 real_tap_rows=4097 one_row=1"<<std::endl;
}
