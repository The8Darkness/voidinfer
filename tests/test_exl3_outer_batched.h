#pragma once

void run_outer_batched_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    require(source.size()>=350 && target.max_context()==1024,"outerbatch fixture extent");
    auto scalar=target.create_context(true),batch=target.create_context(true);
    batch->prepare_continuation(8);
    int cases=0;
    for(int prefix:{16,321}) {
        scalar->reset(); scalar->prefill(std::span<const std::int64_t>(source.data(),16));
        for(int i=16;i<prefix;++i) scalar->decode(source[i]);
        const auto root=scalar->export_exact_host_state();
        std::vector<std::int64_t> greedy;
        for(int i=0;i<8;++i) {const auto token=sample_target(*scalar);greedy.push_back(token);scalar->decode(token);}
        for(int width:{2,4,8}) for(int rejection=-1;rejection<width;++rejection) {
            std::vector<std::int64_t> proposals(greedy.begin(),greedy.begin()+width);
            if(rejection>=0) proposals[rejection]=(proposals[rejection]+1)%kVocab;
            const auto expected=ninfer::exl3::verify_exl3_outer_reference(*scalar,*root,proposals);
            const auto actual=ninfer::exl3::verify_exl3_outer_batched_reference(*batch,*root,proposals);
            require(actual.committed_tokens==expected.committed_tokens && actual.accepted==expected.accepted &&
                    actual.rejected==expected.rejected && actual.committed_state->same_payload(*expected.committed_state),
                    "outerbatch scalar mismatch prefix="+std::to_string(prefix)+" width="+std::to_string(width)+
                    " rejection="+std::to_string(rejection));
            require(actual.executed_rows==width+(rejection<0?0:rejection+1),"outerbatch work accounting");
            const auto next=sample_target(*scalar);
            batch->decode(next);scalar->decode(next);
            require(batch->export_exact_host_state()->same_payload(*scalar->export_exact_host_state()),
                    "outerbatch postrepair continuation");
            ++cases;
        }
    }
    std::cout << "OUTER_BATCHED PASS cases=" << cases << " all_rejection_positions=1 full_state_exact=1\n";
}
