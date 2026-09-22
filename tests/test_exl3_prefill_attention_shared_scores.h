#pragma once

// Focused W08 exactness gate for the default-off large-M shared-score route.
// The 3072+1024 case exercises its maximum 4K dynamic-shared footprint and
// compares represented output bits with the selected six-score / six-softmax /
// triple-values route.
void run_prefill_attention_shared_scores_operator() {
    constexpr int rows=1024,position=3072,capacity=4096;
    constexpr std::uint16_t guard=0x6a5d;
    const std::size_t q_elements=static_cast<std::size_t>(rows)*24*256;
    const std::size_t kv_elements=static_cast<std::size_t>(capacity)*4*256;
    const std::size_t score_elements=static_cast<std::size_t>(16)*24*capacity;
    const auto f16=[](float value) {
        return __half_as_ushort(__float2half_rn(value));
    };
    std::vector<std::uint16_t> q(q_elements),k(kv_elements),v(kv_elements);
    for(std::size_t index=0;index<q.size();++index)
        q[index]=f16(static_cast<float>(
            0.29*std::sin(index*0.013)+0.11*std::cos(index*0.031)));
    for(std::size_t index=0;index<k.size();++index) {
        k[index]=f16(static_cast<float>(
            0.37*std::sin(index*0.017)+(index%997==0?0.75:0.0)));
        v[index]=f16(static_cast<float>(
            0.83*std::cos(index*0.023)-0.19*std::sin(index*0.041)));
    }
    DeviceBuffer dq(q.size()*2),dk(k.size()*2),dv(v.size()*2);
    DeviceBuffer reference((q_elements+2)*2),candidate((q_elements+2)*2),
        candidate256((q_elements+2)*2),candidate_parallel((q_elements+2)*2),
        candidate_head_split((q_elements+2)*2),
        candidate_dimension_split((q_elements+2)*2);
    DeviceBuffer scores(score_elements*sizeof(float));
    cuda_check(cudaMemcpy(dq.get(),q.data(),dq.bytes(),cudaMemcpyHostToDevice),
        "shared-score upload Q");
    cuda_check(cudaMemcpy(dk.get(),k.data(),dk.bytes(),cudaMemcpyHostToDevice),
        "shared-score upload K");
    cuda_check(cudaMemcpy(dv.get(),v.data(),dv.bytes(),cudaMemcpyHostToDevice),
        "shared-score upload V");
    std::vector<std::uint16_t> initialized(q_elements+2,guard);
    for(auto* output:{&reference,&candidate,&candidate256,&candidate_parallel,
                      &candidate_head_split,&candidate_dimension_split})
        cuda_check(cudaMemcpy(output->get(),initialized.data(),output->bytes(),
            cudaMemcpyHostToDevice),"shared-score output canary upload");

    ninfer::exl3::exl3_exact_attention_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(reference.get())+1,
        static_cast<float*>(scores.get()),rows,position,capacity,true,nullptr,
        true,true,true,true,true,false,true,false,true,false,false,true);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(candidate.get())+1,
        rows,position,capacity);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(candidate256.get())+1,
        rows,position,capacity,nullptr,true);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(candidate_parallel.get())+1,
        rows,position,capacity,nullptr,false,true);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(candidate_head_split.get())+1,
        rows,position,capacity,nullptr,false,true,true);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(candidate_dimension_split.get())+1,
        rows,position,capacity,nullptr,false,true,false,true);
    cuda_check(cudaDeviceSynchronize(),"shared-score exact differential");

    std::vector<std::uint16_t> reference_bits(q_elements+2),
        candidate_bits(q_elements+2),candidate256_bits(q_elements+2),
        candidate_parallel_bits(q_elements+2),candidate_head_split_bits(q_elements+2),
        candidate_dimension_split_bits(q_elements+2);
    cuda_check(cudaMemcpy(reference_bits.data(),reference.get(),reference.bytes(),
        cudaMemcpyDeviceToHost),"shared-score reference download");
    cuda_check(cudaMemcpy(candidate_bits.data(),candidate.get(),candidate.bytes(),
        cudaMemcpyDeviceToHost),"shared-score candidate download");
    cuda_check(cudaMemcpy(candidate256_bits.data(),candidate256.get(),
        candidate256.bytes(),cudaMemcpyDeviceToHost),
        "shared-score threads256 candidate download");
    cuda_check(cudaMemcpy(candidate_parallel_bits.data(),candidate_parallel.get(),
        candidate_parallel.bytes(),cudaMemcpyDeviceToHost),
        "shared-score parallel-softmax candidate download");
    cuda_check(cudaMemcpy(candidate_head_split_bits.data(),candidate_head_split.get(),
        candidate_head_split.bytes(),cudaMemcpyDeviceToHost),
        "shared-score head-split256 candidate download");
    cuda_check(cudaMemcpy(candidate_dimension_split_bits.data(),
        candidate_dimension_split.get(),candidate_dimension_split.bytes(),
        cudaMemcpyDeviceToHost),
        "shared-score dimension-split256 candidate download");
    require(reference_bits.front()==guard&&reference_bits.back()==guard&&
            candidate_bits.front()==guard&&candidate_bits.back()==guard&&
            candidate256_bits.front()==guard&&candidate256_bits.back()==guard&&
            candidate_parallel_bits.front()==guard&&candidate_parallel_bits.back()==guard&&
            candidate_head_split_bits.front()==guard&&candidate_head_split_bits.back()==guard&&
            candidate_dimension_split_bits.front()==guard&&
            candidate_dimension_split_bits.back()==guard,
        "shared-score output canary changed");
    require(reference_bits==candidate_bits&&reference_bits==candidate256_bits&&
            reference_bits==candidate_parallel_bits&&
            reference_bits==candidate_head_split_bits&&
            reference_bits==candidate_dimension_split_bits,
        "shared-score output differs from selected exact route");

    // Actual final admitted large-M call from the canonical 4096-token
    // 1024-width schedule. The remaining 32/16-row suffix stays on the
    // selected eager route to avoid high-shared-memory under-occupancy.
    constexpr int tail_rows=128,tail_position=3856;
    const std::size_t tail_elements=static_cast<std::size_t>(tail_rows)*24*256;
    DeviceBuffer tail_reference((tail_elements+2)*2),tail_candidate((tail_elements+2)*2),
        tail_candidate256((tail_elements+2)*2),
        tail_candidate_parallel((tail_elements+2)*2),
        tail_candidate_head_split((tail_elements+2)*2),
        tail_candidate_dimension_split((tail_elements+2)*2);
    std::vector<std::uint16_t> tail_initialized(tail_elements+2,guard);
    for(auto* output:{&tail_reference,&tail_candidate,&tail_candidate256,
                      &tail_candidate_parallel,&tail_candidate_head_split,
                      &tail_candidate_dimension_split})
        cuda_check(cudaMemcpy(output->get(),tail_initialized.data(),output->bytes(),
            cudaMemcpyHostToDevice),"shared-score tail output canary upload");
    ninfer::exl3::exl3_exact_attention_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(tail_reference.get())+1,
        static_cast<float*>(scores.get()),tail_rows,tail_position,capacity,true,nullptr,
        true,true,true,true,true,false,true,false,true,false,false,true);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(tail_candidate.get())+1,
        tail_rows,tail_position,capacity);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(tail_candidate256.get())+1,
        tail_rows,tail_position,capacity,nullptr,true);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(tail_candidate_parallel.get())+1,
        tail_rows,tail_position,capacity,nullptr,false,true);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(tail_candidate_head_split.get())+1,
        tail_rows,tail_position,capacity,nullptr,false,true,true);
    ninfer::exl3::exl3_prefill_attention_shared_scores_for_test(
        static_cast<const std::uint16_t*>(dq.get()),
        static_cast<const std::uint16_t*>(dk.get()),
        static_cast<const std::uint16_t*>(dv.get()),
        static_cast<std::uint16_t*>(tail_candidate_dimension_split.get())+1,
        tail_rows,tail_position,capacity,nullptr,false,true,false,true);
    cuda_check(cudaDeviceSynchronize(),"shared-score final-128 exact differential");
    std::vector<std::uint16_t> tail_reference_bits(tail_elements+2),
        tail_candidate_bits(tail_elements+2),tail_candidate256_bits(tail_elements+2),
        tail_candidate_parallel_bits(tail_elements+2),
        tail_candidate_head_split_bits(tail_elements+2),
        tail_candidate_dimension_split_bits(tail_elements+2);
    cuda_check(cudaMemcpy(tail_reference_bits.data(),tail_reference.get(),
        tail_reference.bytes(),cudaMemcpyDeviceToHost),
        "shared-score tail reference download");
    cuda_check(cudaMemcpy(tail_candidate_bits.data(),tail_candidate.get(),
        tail_candidate.bytes(),cudaMemcpyDeviceToHost),
        "shared-score tail candidate download");
    cuda_check(cudaMemcpy(tail_candidate256_bits.data(),tail_candidate256.get(),
        tail_candidate256.bytes(),cudaMemcpyDeviceToHost),
        "shared-score threads256 tail candidate download");
    cuda_check(cudaMemcpy(tail_candidate_parallel_bits.data(),tail_candidate_parallel.get(),
        tail_candidate_parallel.bytes(),cudaMemcpyDeviceToHost),
        "shared-score parallel-softmax tail candidate download");
    cuda_check(cudaMemcpy(tail_candidate_head_split_bits.data(),
        tail_candidate_head_split.get(),tail_candidate_head_split.bytes(),
        cudaMemcpyDeviceToHost),"shared-score head-split256 tail candidate download");
    cuda_check(cudaMemcpy(tail_candidate_dimension_split_bits.data(),
        tail_candidate_dimension_split.get(),tail_candidate_dimension_split.bytes(),
        cudaMemcpyDeviceToHost),
        "shared-score dimension-split256 tail candidate download");
    require(tail_reference_bits.front()==guard&&tail_reference_bits.back()==guard&&
            tail_candidate_bits.front()==guard&&tail_candidate_bits.back()==guard&&
            tail_candidate256_bits.front()==guard&&tail_candidate256_bits.back()==guard&&
            tail_candidate_parallel_bits.front()==guard&&
            tail_candidate_parallel_bits.back()==guard&&
            tail_candidate_head_split_bits.front()==guard&&
            tail_candidate_head_split_bits.back()==guard&&
            tail_candidate_dimension_split_bits.front()==guard&&
            tail_candidate_dimension_split_bits.back()==guard,
        "shared-score final-128 output canary changed");
    require(tail_reference_bits==tail_candidate_bits&&
            tail_reference_bits==tail_candidate256_bits&&
            tail_reference_bits==tail_candidate_parallel_bits&&
            tail_reference_bits==tail_candidate_head_split_bits&&
            tail_reference_bits==tail_candidate_dimension_split_bits,
        "shared-score final-128 output differs from selected exact route");
    std::cout << "PREFILL_ATTENTION_SHARED_SCORES PASS synthetic_rows=1024 "
                 "synthetic_position=3072 final_rows=128 final_position=3856 "
                 "capacity=4096 max_shared_bytes=101376 output_bit_exact=1 "
                 "threads256_output_bit_exact=1 parallel_softmax_output_bit_exact=1 "
                 "head_split256_output_bit_exact=1 "
                 "dimension_split256_output_bit_exact=1 guards=1\n";
}

void run_prefill_attention_shared_scores_state(
    ninfer::exl3::Exl3TextModel& target,
    const std::vector<std::int64_t>& fixture) {
    using ninfer::exl3::exl3_prefill_attention_shared_score_global_snapshot;
    require(fixture.size()>=4096,"shared-score state fixture extent");
    const auto saved=env("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE");
    const auto saved_rows128=
        env("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_ROWS128");
    const auto saved_threads256=
        env("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_THREADS256");
    const auto saved_parallel_softmax=
        env("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_PARALLEL_SOFTMAX");
    const auto saved_head_split256=
        env("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_HEAD_SPLIT256");
    const auto saved_dimension_split256=
        env("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_DIMENSION_SPLIT256");
    struct Result {
        std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> request;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
        std::vector<float> logits;
        ninfer::exl3::Exl3PrefillAttentionSharedScoreSnapshot counters;
    };
    const auto run=[&](ninfer::exl3::Exl3TextContext& context,const char* enabled,
                       const char* rows128,const char* threads256,
                       const char* parallel_softmax,const char* head_split256,
                       const char* dimension_split256) {
        _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE",enabled);
        _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_ROWS128",rows128);
        _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_THREADS256",threads256);
        _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_PARALLEL_SOFTMAX",
            parallel_softmax);
        _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_HEAD_SPLIT256",
            head_split256);
        _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_DIMENSION_SPLIT256",
            dimension_split256);
        const auto before=exl3_prefill_attention_shared_score_global_snapshot();
        auto request=ninfer::exl3::Exl3VeriCacheRequest::initialize(
            context,std::span<const std::int64_t>(fixture.data(),4096),1024);
        cuda_check(cudaDeviceSynchronize(),"shared-score state completion");
        const auto after=exl3_prefill_attention_shared_score_global_snapshot();
        return Result{std::move(request),context.export_exact_host_state(),
            context.logits_host(),
            {after.launch_attempts-before.launch_attempts,
             after.row_attempts-before.row_attempts,
             after.global_score_bytes_eliminated-
                before.global_score_bytes_eliminated,
             after.threads256_launch_attempts-
                before.threads256_launch_attempts,
             after.parallel_softmax_launch_attempts-
                before.parallel_softmax_launch_attempts,
             after.head_split256_launch_attempts-
                before.head_split256_launch_attempts,
             after.dimension_split256_launch_attempts-
                before.dimension_split256_launch_attempts}};
    };

    auto control=target.create_context(true);
    control->prepare_continuation(8);
    const auto expected=run(*control,"0","0","0","0","0","0");
    auto candidate=target.create_context(true);
    candidate->prepare_continuation(8);
    const auto first=run(*candidate,"1","1","0","1","0","1");
    require(expected.request->same_taps(*first.request) &&
            expected.state->same_payload(*first.state) &&
            expected.logits==first.logits,
        "shared-score first-request HostKV/tap/logit state");
    require(expected.counters.launch_attempts==0 &&
            expected.counters.row_attempts==0 &&
            expected.counters.global_score_bytes_eliminated==0 &&
            first.counters.launch_attempts==160 &&
            first.counters.row_attempts==63488 &&
            first.counters.global_score_bytes_eliminated==12192743424ULL &&
            first.counters.threads256_launch_attempts==0 &&
            first.counters.parallel_softmax_launch_attempts==160 &&
            first.counters.head_split256_launch_attempts==0 &&
            first.counters.dimension_split256_launch_attempts==160,
        "shared-score first-request dispatch accounting");

    candidate->reset();
    const auto second=run(*candidate,"1","1","0","1","0","1");
    require(expected.request->same_taps(*second.request) &&
            expected.state->same_payload(*second.state) &&
            expected.logits==second.logits &&
            first.request->same_taps(*second.request) &&
            first.state->same_payload(*second.state) &&
            first.logits==second.logits,
        "shared-score reset/reuse HostKV/tap/logit state");
    require(second.counters.launch_attempts==160 &&
            second.counters.row_attempts==63488 &&
            second.counters.global_score_bytes_eliminated==12192743424ULL &&
            second.counters.threads256_launch_attempts==0 &&
            second.counters.parallel_softmax_launch_attempts==160 &&
            second.counters.head_split256_launch_attempts==0 &&
            second.counters.dimension_split256_launch_attempts==160,
        "shared-score reset/reuse dispatch accounting");
    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE",saved.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_ROWS128",
        saved_rows128.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_THREADS256",
        saved_threads256.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_PARALLEL_SOFTMAX",
        saved_parallel_softmax.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_HEAD_SPLIT256",
        saved_head_split256.c_str());
    _putenv_s("NINFER_EXL3_PREFILL_ATTENTION_SHARED_SCORE_DIMENSION_SPLIT256",
        saved_dimension_split256.c_str());
    std::cout << "PREFILL_ATTENTION_SHARED_SCORE_STATE PASS "
                 "first_launches=160 reset_launches=160 rows=63488 "
                 "global_score_bytes_eliminated=12192743424 "
                 "parallel_softmax_launches=160 dimension_split256_launches=160 "
                 "exact_hostkv_taps_logits=1 "
                 "reset_lifecycle=1\n";
}
