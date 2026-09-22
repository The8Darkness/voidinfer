#pragma once

void run_outer_stop_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using namespace ninfer::exl3;
    std::vector<std::int64_t> path{10,11,12,13,14,15,16,17};
    int tables=0;
    // Identical head winners do not imply identical per-request stop policy.
    // Reversing request order must preserve each private terminal boundary.
    for(bool reverse:{false,true}) {
        const std::array<std::int64_t,1> early{path[1]},late{path[6]};
        const auto first=decide_exl3_outer_prefix(path,path,reverse?late:early);
        const auto second=decide_exl3_outer_prefix(path,path,reverse?early:late);
        require(first.stopped && second.stopped &&
            first.committed_tokens.size()==(reverse?7:2) &&
            second.committed_tokens.size()==(reverse?2:7),"shared head leaked peer stop boundary");
        const std::array<std::int64_t,1> absent{99};
        const auto continuing=decide_exl3_outer_prefix(path,path,absent);
        require(!continuing.stopped && continuing.committed_tokens==path,
            "peer terminal truncated a continuing request");
    }
    for(int stop=0;stop<=8;++stop) for(int reject=0;reject<=8;++reject) {
        auto proposal=path;if(reject<8) proposal[reject]=99;
        std::vector<std::int64_t> terminal;if(stop<8) terminal.push_back(path[stop]);
        const auto result=decide_exl3_outer_prefix(path,proposal,terminal);
        const int consumed=std::min({8,stop+1,reject+1});
        require(result.committed_tokens==std::vector<std::int64_t>(path.begin(),path.begin()+consumed) &&
            result.accepted==std::min({8,stop+1,reject}) && result.rejected==(reject<8 && reject<=stop) &&
            result.stopped==(stop<8 && stop<=reject),"outer terminal algebraic decision oracle");
        ++tables;
    }
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto scalar=target.create_context(true),packed=target.create_context(true);
    scalar->prepare_continuation(8);packed->prepare_continuation(8);
    int cases=0;
    for(int prefix:{16,321}) {
        const auto root=Exl3VeriCacheRequest::initialize(*scalar,std::span<const std::int64_t>(source.data(),prefix))->state();
        std::vector<std::int64_t> greedy;
        for(int i=0;i<8;++i) {const auto token=sample_target(*scalar);greedy.push_back(token);scalar->decode(token);}
        for(int stop=0;stop<8;++stop) for(int mode=0;mode<3;++mode) {
            const std::array<std::int64_t,1> terminal{greedy[stop]};
            auto proposal=greedy;
            if(mode) {const int index=mode==1?0:stop;proposal[index]=(proposal[index]+1)%kVocab;}
            const auto a=verify_exl3_outer_reference(*scalar,*root,proposal,true,terminal);
            const auto b=verify_exl3_outer_batched_reference(*packed,*root,proposal,true,terminal);
            require(a.committed_tokens==b.committed_tokens && a.accepted==b.accepted && a.rejected==b.rejected &&
                a.stopped==b.stopped && a.committed_taps==b.committed_taps && a.committed_state->same_payload(*b.committed_state),
                "outer terminal packed state differs from scalar stop");
            require(b.committed_state->position()==prefix+b.committed_tokens.size(),"post-terminal suffix retained");
            scalar->decode(source[prefix+9]);packed->decode(source[prefix+9]);
            require(scalar->export_exact_host_state()->same_payload(*packed->export_exact_host_state()),"terminal repair next input state");
            ++cases;
        }
    }
    auto request=Exl3VeriCacheRequest::initialize(*scalar,std::span<const std::int64_t>(source.data(),16));
    Exl3VeriCacheQueue queue(2);queue.add(request);queue.add(request);
    const auto first=*queue.acquire();
    const std::array<std::int64_t,1> terminal{sample_target(*scalar)};
    auto [updated,result]=request->verify(*scalar,terminal,terminal);
    require(result.stopped && queue.publish(first,updated,true) && queue.status(0)==Exl3VeriCacheQueue::Status::completed,
        "terminal completion publication");
    const auto second=*queue.acquire();queue.cancel(1);
    require(!queue.publish(second,updated,true) && queue.status(1)==Exl3VeriCacheQueue::Status::cancelled && !queue.acquire(),
        "completion bypassed cancellation");
    std::cout << "OUTER_STOP PASS table_cases=" << tables << " model_cases=" << cases << " completed_cancelled=distinct\n";
}
