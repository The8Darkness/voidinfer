#pragma once

void run_exact_attention_gqa_six_segmented_prefix_operator() {
    constexpr int position=4096,capacity=4352;
    const auto f16=[](float value){return __half_as_ushort(__float2half_rn(value));};
    int cases=0;
    for(const int rows:{1,8}) {
        const std::size_t q_elements=static_cast<std::size_t>(rows)*24*256;
        const std::size_t kv_elements=static_cast<std::size_t>(capacity)*4*256;
        const std::size_t prefix_elements=static_cast<std::size_t>(position)*4*256;
        const std::size_t score_elements=static_cast<std::size_t>(rows)*24*capacity;
        std::vector<std::uint16_t> q(q_elements),k(kv_elements),v(kv_elements);
        for(std::size_t i=0;i<q.size();++i)
            q[i]=f16(static_cast<float>(0.27*std::sin(i*0.019)+0.08*std::cos(i*0.047)));
        for(std::size_t i=0;i<k.size();++i) {
            k[i]=f16(static_cast<float>(0.41*std::sin(i*0.031)+(i%991==0?0.8:0.0)));
            v[i]=f16(static_cast<float>(1.07*std::cos(i*0.023)-0.33*std::sin(i*0.071)));
        }
        std::vector<std::uint16_t> private_k=k,private_v=v;
        std::fill_n(private_k.begin(),prefix_elements,0x7e00);
        std::fill_n(private_v.begin(),prefix_elements,0x7e00);
        std::vector<std::uint16_t> prefix_k(k.begin(),k.begin()+prefix_elements);
        std::vector<std::uint16_t> prefix_v(v.begin(),v.begin()+prefix_elements);
        DeviceBuffer dq(q.size()*2),dk(k.size()*2),dv(v.size()*2),
            dprivate_k(private_k.size()*2),dprivate_v(private_v.size()*2),
            dprefix_k(prefix_k.size()*2),dprefix_v(prefix_v.size()*2),
            control_output(q.size()*2),segmented_output(q.size()*2),
            control_scores(score_elements*4),segmented_scores(score_elements*4);
        cuda_check(cudaMemcpy(dq.get(),q.data(),q.size()*2,cudaMemcpyHostToDevice),
            "six segmented upload Q");
        cuda_check(cudaMemcpy(dk.get(),k.data(),k.size()*2,cudaMemcpyHostToDevice),
            "six segmented upload K");
        cuda_check(cudaMemcpy(dv.get(),v.data(),v.size()*2,cudaMemcpyHostToDevice),
            "six segmented upload V");
        cuda_check(cudaMemcpy(dprivate_k.get(),private_k.data(),private_k.size()*2,
            cudaMemcpyHostToDevice),"six segmented upload private K");
        cuda_check(cudaMemcpy(dprivate_v.get(),private_v.data(),private_v.size()*2,
            cudaMemcpyHostToDevice),"six segmented upload private V");
        cuda_check(cudaMemcpy(dprefix_k.get(),prefix_k.data(),prefix_k.size()*2,
            cudaMemcpyHostToDevice),"six segmented upload prefix K");
        cuda_check(cudaMemcpy(dprefix_v.get(),prefix_v.data(),prefix_v.size()*2,
            cudaMemcpyHostToDevice),"six segmented upload prefix V");
        cuda_check(cudaMemset(control_scores.get(),0,control_scores.bytes()),
            "six segmented clear control scores");
        cuda_check(cudaMemset(segmented_scores.get(),0,segmented_scores.bytes()),
            "six segmented clear candidate scores");
        const auto run=[&](const std::uint16_t* key,const std::uint16_t* value,
            DeviceBuffer& output,DeviceBuffer& scores,
            ninfer::exl3::Exl3AttentionPageRanges pages={}) {
            ninfer::exl3::exl3_exact_attention_for_test(
                static_cast<const std::uint16_t*>(dq.get()),key,value,
                static_cast<std::uint16_t*>(output.get()),
                static_cast<float*>(scores.get()),rows,position,capacity,true,
                nullptr,true,true,true,true,true,false,true,false,true,false,
                false,true,false,true,false,false,false,false,false,false,false,
                false,pages);
        };
        run(static_cast<const std::uint16_t*>(dk.get()),
            static_cast<const std::uint16_t*>(dv.get()),control_output,control_scores);
        ninfer::exl3::Exl3AttentionPageRanges pages;
        pages.append(static_cast<const std::uint16_t*>(dprefix_k.get()),
            static_cast<const std::uint16_t*>(dprefix_v.get()),0,position,position);
        run(static_cast<const std::uint16_t*>(dprivate_k.get()),
            static_cast<const std::uint16_t*>(dprivate_v.get()),segmented_output,
            segmented_scores,pages);
        cuda_check(cudaDeviceSynchronize(),"six segmented synchronize");
        require(target_continue_device_bits(
                    static_cast<const std::uint16_t*>(segmented_output.get()),q.size(),
                    "six segmented output")==
                target_continue_device_bits(
                    static_cast<const std::uint16_t*>(control_output.get()),q.size(),
                    "six contiguous output"),
            "segmented selected six-softmax/triple-values output mismatch");
        std::vector<std::uint32_t> control_score_bits(score_elements),
            segmented_score_bits(score_elements);
        cuda_check(cudaMemcpy(control_score_bits.data(),control_scores.get(),
            control_scores.bytes(),cudaMemcpyDeviceToHost),
            "six segmented control scores");
        cuda_check(cudaMemcpy(segmented_score_bits.data(),segmented_scores.get(),
            segmented_scores.bytes(),cudaMemcpyDeviceToHost),
            "six segmented candidate scores");
        require(control_score_bits==segmented_score_bits,
            "segmented selected six-softmax score workspace mismatch");
        require(target_continue_device_bits(
                    static_cast<const std::uint16_t*>(dprefix_k.get()),prefix_k.size(),
                    "six segmented immutable K")==prefix_k &&
                target_continue_device_bits(
                    static_cast<const std::uint16_t*>(dprefix_v.get()),prefix_v.size(),
                    "six segmented immutable V")==prefix_v,
            "segmented selected six-softmax modified immutable prefix");
        ++cases;
    }
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SEGMENTED_PREFIX PASS cases="<<cases
        <<" position=4096 rows=1,8 output_bits=exact score_bits=exact"
        <<" prefix_immutable=1 private_prefix_poisoned=1\n";
}

void run_exact_attention_gqa_six_softmax_triple_values_screen(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    const int prefix=target.max_context()>=4352?4096:96;
    require(code.size()>=prefix&&prose.size()>=prefix,
        "GQA-six softmax triple-values fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "GQA-six softmax triple-values finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    for(const auto& fixture:std::array<
        std::pair<const char*,const std::vector<std::int64_t>*>,2>{
            std::pair{"code",&code},std::pair{"prose",&prose}}) {
        const std::vector<std::int64_t> input(
            fixture.second->begin(),fixture.second->begin()+prefix);
        const auto run=[&](Exl3TextContext& context) {
            const auto started=std::chrono::steady_clock::now();
            const auto request=Request::initialize(context,input,1024);
            const double prefill_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            std::vector<std::int64_t> tokens;
            for(int row=0;row<8;++row) {
                const auto token=greedy(context);tokens.push_back(token);context.decode(token);
            }
            return std::tuple{prefill_ms,tokens,context.export_exact_host_state()};
        };
        const auto baseline=run(*control),changed=run(*candidate);
        require(std::get<1>(baseline)==std::get<1>(changed)&&
            std::get<2>(baseline)->same_payload(*std::get<2>(changed)),
            "GQA-six softmax triple-values token/state");
        std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES fixture="
            <<fixture.first<<" prefix="<<prefix<<" decode_rows=8 control_prefill_ms="
            <<std::get<0>(baseline)<<" candidate_prefill_ms="<<std::get<0>(changed)
            <<" control_tps="<<prefix*1000/std::get<0>(baseline)
            <<" candidate_tps="<<prefix*1000/std::get<0>(changed)<<std::endl;
    }
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES PASS fixtures=2 prefix="
        <<prefix<<" decode_rows=8 exact_tokens_state=1 control=duplicated_triple_softmax"
        <<" candidate=one_six_head_softmax_plus_triple_values"<<std::endl;
}

void run_exact_attention_gqa_six_softmax_triple_pair_dimensions_state(
    Exl3TextModel& target,const std::vector<std::int64_t>& code) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()>=4352&&code.size()>=prefix,
        "pair-dimension triple-values fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FULL_CTA","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_THREADS128","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCALAR_DIM","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_TWO_QUERY","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FUSED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "pair-dimension triple-values finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const std::vector<std::int64_t> input(code.begin(),code.begin()+prefix);
    const auto run=[&](Exl3TextContext& context) {
        const auto request=Request::initialize(context,input,1024);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=greedy(context);tokens.push_back(token);context.decode(token);
        }
        return std::tuple{tokens,context.export_exact_host_state(),
            context.gqa_six_softmax_triple_pair_dimensions_launch_attempts(),
            context.gqa_six_softmax_triple_pair_dimensions_row_attempts()};
    };
    const auto baseline=run(*control),changed=run(*candidate);
    require(std::get<0>(baseline)==std::get<0>(changed)&&
            std::get<1>(baseline)->same_payload(*std::get<1>(changed)),
        "pair-dimension triple-values token/state");
    require(std::get<2>(baseline)==0&&std::get<3>(baseline)==0&&
            std::get<2>(changed)>0&&std::get<3>(changed)>=prefix,
        "pair-dimension triple-values dispatch counters");
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_PAIR_DIMENSIONS_STATE PASS"
        <<" prefix="<<prefix<<" decode_rows=8 exact_tokens_state=1"
        <<" candidate_launches="<<std::get<2>(changed)
        <<" candidate_rows="<<std::get<3>(changed)<<std::endl;
}

void run_exact_attention_gqa_six_softmax_six_values_single_load_state(
    Exl3TextModel& target,const std::vector<std::int64_t>& code) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()>=4352&&code.size()>=prefix,
        "single-load six-values fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FULL_CTA","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_THREADS128","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCALAR_DIM","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_TWO_QUERY","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FUSED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "single-load six-values finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const std::vector<std::int64_t> input(code.begin(),code.begin()+prefix);
    const auto run=[&](Exl3TextContext& context) {
        const auto request=Request::initialize(context,input,1024);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=greedy(context);tokens.push_back(token);context.decode(token);
        }
        return std::tuple{tokens,context.export_exact_host_state(),
            context.gqa_six_softmax_six_values_single_load_launch_attempts(),
            context.gqa_six_softmax_six_values_single_load_row_attempts()};
    };
    const auto baseline=run(*control),changed=run(*candidate);
    require(std::get<0>(baseline)==std::get<0>(changed)&&
            std::get<1>(baseline)->same_payload(*std::get<1>(changed)),
        "single-load six-values token/state");
    require(std::get<2>(baseline)==0&&std::get<3>(baseline)==0&&
            std::get<2>(changed)>0&&std::get<3>(changed)>=prefix,
        "single-load six-values dispatch counters");
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD_STATE PASS"
        <<" prefix="<<prefix<<" decode_rows=8 exact_tokens_state=1"
        <<" candidate_launches="<<std::get<2>(changed)
        <<" candidate_rows="<<std::get<3>(changed)<<std::endl;
}

void run_exact_attention_gqa_six_softmax_six_values_scalar_single_load_state(
    Exl3TextModel& target,const std::vector<std::int64_t>& code) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()>=4352&&code.size()>=prefix,
        "scalar single-load six-values fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","1");
    for(const char* option:{
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE64",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FULL_CTA",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_THREADS128",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCORE_TILE",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCALAR_DIM",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_TWO_QUERY",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FUSED",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_WARP_SCORE_BROADCAST",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SCALAR_SINGLE_LOAD"})
        _putenv_s(option,"0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SCALAR_SINGLE_LOAD",
        "1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "scalar single-load six-values finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const std::vector<std::int64_t> input(code.begin(),code.begin()+prefix);
    const auto run=[&](Exl3TextContext& context) {
        const auto request=Request::initialize(context,input,1024);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=greedy(context);tokens.push_back(token);context.decode(token);
        }
        return std::tuple{tokens,context.export_exact_host_state(),
            context.gqa_six_softmax_six_values_single_load_launch_attempts(),
            context.gqa_six_softmax_six_values_single_load_row_attempts()};
    };
    const auto baseline=run(*control),changed=run(*candidate);
    require(std::get<0>(baseline)==std::get<0>(changed)&&
            std::get<1>(baseline)->same_payload(*std::get<1>(changed)),
        "scalar single-load six-values token/state");
    require(std::get<2>(baseline)==0&&std::get<3>(baseline)==0&&
            std::get<2>(changed)>0&&std::get<3>(changed)>=8,
        "scalar single-load six-values dispatch counters");
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SCALAR_SINGLE_LOAD_STATE PASS"
        <<" prefix="<<prefix<<" decode_rows=8 exact_tokens_state=1"
        <<" candidate_launches="<<std::get<2>(changed)
        <<" candidate_rows="<<std::get<3>(changed)<<std::endl;
}

void run_exact_attention_gqa_six_softmax_triple_key_pair_pipeline_state(
    Exl3TextModel& target,const std::vector<std::int64_t>& code) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()>=4352&&code.size()>=prefix,
        "key-pair pipeline fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","1");
    for(const char* option:{
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE64",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FULL_CTA",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_THREADS128",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCORE_TILE",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCALAR_DIM",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_TWO_QUERY",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FUSED",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_WARP_SCORE_BROADCAST",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SCALAR_SINGLE_LOAD",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_KEY_PAIR_PIPELINE"})
        _putenv_s(option,"0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_KEY_PAIR_PIPELINE",
        "1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "key-pair pipeline finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const std::vector<std::int64_t> input(code.begin(),code.begin()+prefix);
    const auto run=[&](Exl3TextContext& context) {
        const auto request=Request::initialize(context,input,1024);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=greedy(context);tokens.push_back(token);context.decode(token);
        }
        return std::tuple{tokens,context.export_exact_host_state(),
            context.gqa_six_softmax_triple_key_pair_launch_attempts(),
            context.gqa_six_softmax_triple_key_pair_row_attempts()};
    };
    const auto baseline=run(*control),changed=run(*candidate);
    require(std::get<0>(baseline)==std::get<0>(changed)&&
            std::get<1>(baseline)->same_payload(*std::get<1>(changed)),
        "key-pair pipeline token/state");
    require(std::get<2>(baseline)==0&&std::get<3>(baseline)==0&&
            std::get<2>(changed)>0&&std::get<3>(changed)>=8,
        "key-pair pipeline dispatch counters");
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_KEY_PAIR_PIPELINE_STATE PASS"
        <<" prefix="<<prefix<<" decode_rows=8 exact_tokens_state=1"
        <<" candidate_launches="<<std::get<2>(changed)
        <<" candidate_rows="<<std::get<3>(changed)<<std::endl;
}

void run_exact_attention_gqa_six_softmax_triple_score_tile_state(
    Exl3TextModel& target,const std::vector<std::int64_t>& code) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()>=4352&&code.size()>=prefix,
        "score-tile triple-values fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE64","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FULL_CTA","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_THREADS128","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCALAR_DIM","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_TWO_QUERY","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FUSED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCORE_TILE","0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCORE_TILE","1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "score-tile triple-values finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const std::vector<std::int64_t> input(code.begin(),code.begin()+prefix);
    const auto run=[&](Exl3TextContext& context) {
        const auto request=Request::initialize(context,input,1024);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=greedy(context);tokens.push_back(token);context.decode(token);
        }
        return std::tuple{tokens,context.export_exact_host_state(),
            context.gqa_six_softmax_triple_score_tile_launch_attempts(),
            context.gqa_six_softmax_triple_score_tile_row_attempts()};
    };
    const auto baseline=run(*control),changed=run(*candidate);
    require(std::get<0>(baseline)==std::get<0>(changed)&&
            std::get<1>(baseline)->same_payload(*std::get<1>(changed)),
        "score-tile triple-values token/state");
    require(std::get<2>(baseline)==0&&std::get<3>(baseline)==0&&
            std::get<2>(changed)>0&&std::get<3>(changed)>=prefix,
        "score-tile triple-values dispatch counters");
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_SCORE_TILE_STATE PASS"
        <<" prefix="<<prefix<<" decode_rows=8 exact_tokens_state=1"
        <<" candidate_launches="<<std::get<2>(changed)
        <<" candidate_rows="<<std::get<3>(changed)<<std::endl;
}

void run_exact_attention_gqa_six_softmax_triple_warp_score_broadcast_state(
    Exl3TextModel& target,const std::vector<std::int64_t>& code) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096;
    require(target.max_context()>=4352&&code.size()>=prefix,
        "warp-score-broadcast triple-values fixture/context extent");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4","0");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES","1");
    for(const char* option:{
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE64",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FULL_CTA",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_THREADS128",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCORE_TILE",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_SCALAR_DIM",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_TWO_QUERY",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_FUSED",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SCALAR_SINGLE_LOAD",
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_WARP_SCORE_BROADCAST"})
        _putenv_s(option,"0");
    auto control=target.create_context(true);control->prepare_continuation(8);
    _putenv_s(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_WARP_SCORE_BROADCAST",
        "1");
    auto candidate=target.create_context(true);candidate->prepare_continuation(8);
    const auto greedy=[](Exl3TextContext& context) {
        const auto logits=context.logits_host();
        require(!logits.empty()&&std::none_of(logits.begin(),logits.end(),
            [](float value){return !std::isfinite(value);}),
            "warp-score-broadcast triple-values finite logits");
        return static_cast<std::int64_t>(
            std::max_element(logits.begin(),logits.end())-logits.begin());
    };
    const std::vector<std::int64_t> input(code.begin(),code.begin()+prefix);
    const auto run=[&](Exl3TextContext& context) {
        const auto request=Request::initialize(context,input,1024);
        std::vector<std::int64_t> tokens;
        for(int row=0;row<8;++row) {
            const auto token=greedy(context);tokens.push_back(token);context.decode(token);
        }
        return std::tuple{tokens,context.export_exact_host_state(),
            context.gqa_six_softmax_triple_warp_score_broadcast_launch_attempts(),
            context.gqa_six_softmax_triple_warp_score_broadcast_row_attempts()};
    };
    const auto baseline=run(*control),changed=run(*candidate);
    require(std::get<0>(baseline)==std::get<0>(changed)&&
            std::get<1>(baseline)->same_payload(*std::get<1>(changed)),
        "warp-score-broadcast triple-values token/state");
    require(std::get<2>(baseline)==0&&std::get<3>(baseline)==0&&
            std::get<2>(changed)>0&&std::get<3>(changed)>=prefix,
        "warp-score-broadcast triple-values dispatch counters");
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_WARP_SCORE_BROADCAST_STATE PASS"
        <<" prefix="<<prefix<<" decode_rows=8 exact_tokens_state=1"
        <<" candidate_launches="<<std::get<2>(changed)
        <<" candidate_rows="<<std::get<3>(changed)<<std::endl;
}
