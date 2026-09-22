#pragma once

void run_exact_attention_gqa_six_softmax_triple_values_v_tile_operator() {
    struct Case { int rows; int position; int capacity; };
    const std::array<Case,10> cases{{
        {1,0,32},       // one represented history row
        {15,1,32},      // causal counts 2..16, including 8 and 15
        {16,1,40},      // causal counts 2..17 and a poisoned tail
        {1,254,1024},    // tile boundary history extent 255
        {1,255,1024},    // tile boundary history extent 256
        {1,256,1024},    // tile boundary history extent 257
        {1,510,1024},    // double-tile boundary history extent 511
        {1,511,1024},    // double-tile boundary history extent 512
        {1,512,1024},    // double-tile boundary history extent 513
        {1,4095,4096},  // physical C1 history extent
    }};
    constexpr std::uint16_t output_guard=0x6d3b;
    constexpr float score_guard=173.25f;
    const auto f16=[](float value) {
        return __half_as_ushort(__float2half_rn(value));
    };
    std::size_t output_values=0,score_values=0;
    for(std::size_t case_index=0;case_index<cases.size();++case_index) {
        const auto item=cases[case_index];
        const int max_count=item.position+item.rows;
        std::vector<std::uint16_t> q(
            static_cast<std::size_t>(item.rows)*24*256);
        std::vector<std::uint16_t> k(
            static_cast<std::size_t>(item.capacity)*4*256);
        std::vector<std::uint16_t> v(k.size());
        for(std::size_t i=0;i<q.size();++i)
            q[i]=f16(static_cast<float>(0.31*std::sin(i*0.019)+
                0.07*std::cos(i*0.047)));
        for(int row=0;row<item.capacity;++row) for(int kv=0;kv<4;++kv)
            for(int d=0;d<256;++d) {
                const auto index=(static_cast<std::size_t>(row)*4+kv)*256+d;
                if(row>=max_count) {
                    // Values beyond every causal count are intentionally loud;
                    // neither route may consume them.
                    k[index]=f16((row+kv+d)&1?96.0f:-96.0f);
                    v[index]=f16((row+kv+d)&1?-72.0f:72.0f);
                } else {
                    k[index]=f16(static_cast<float>(0.43*std::sin(index*0.037)+
                        (index%991==0?0.9:0.0)));
                    v[index]=f16(static_cast<float>(1.1*std::cos(index*0.023)-
                        0.37*std::sin(index*0.071)));
                }
            }
        const std::size_t output_elements=q.size();
        const std::size_t score_elements=
            static_cast<std::size_t>(item.rows)*24*item.capacity;
        DeviceBuffer dq(q.size()*sizeof(std::uint16_t));
        DeviceBuffer dk(k.size()*sizeof(std::uint16_t));
        DeviceBuffer dv(v.size()*sizeof(std::uint16_t));
        DeviceBuffer control_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer candidate_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer full_cta_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer fused_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer tile512_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer scalar_dim_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer two_query_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer warp_broadcast_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer scalar_single_load_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer key_pair_output((output_elements+2)*sizeof(std::uint16_t));
        DeviceBuffer control_scores((score_elements+2)*sizeof(float));
        DeviceBuffer candidate_scores((score_elements+2)*sizeof(float));
        DeviceBuffer full_cta_scores((score_elements+2)*sizeof(float));
        DeviceBuffer fused_scores((score_elements+2)*sizeof(float));
        DeviceBuffer tile512_scores((score_elements+2)*sizeof(float));
        DeviceBuffer scalar_dim_scores((score_elements+2)*sizeof(float));
        DeviceBuffer two_query_scores((score_elements+2)*sizeof(float));
        DeviceBuffer warp_broadcast_scores((score_elements+2)*sizeof(float));
        DeviceBuffer scalar_single_load_scores((score_elements+2)*sizeof(float));
        DeviceBuffer key_pair_scores((score_elements+2)*sizeof(float));
        cuda_check(cudaMemcpy(dq.get(),q.data(),q.size()*sizeof(std::uint16_t),
            cudaMemcpyHostToDevice),"V-tile operator upload Q");
        cuda_check(cudaMemcpy(dk.get(),k.data(),k.size()*sizeof(std::uint16_t),
            cudaMemcpyHostToDevice),"V-tile operator upload K");
        cuda_check(cudaMemcpy(dv.get(),v.data(),v.size()*sizeof(std::uint16_t),
            cudaMemcpyHostToDevice),"V-tile operator upload V");
        const std::vector<std::uint16_t> output_initial(output_elements+2,output_guard);
        const std::vector<float> score_initial(score_elements+2,score_guard);
        for(auto* allocation:{&control_output,&candidate_output,&full_cta_output,&fused_output,
                              &tile512_output,&scalar_dim_output,&two_query_output,
                              &warp_broadcast_output,&scalar_single_load_output,
                              &key_pair_output})
            cuda_check(cudaMemcpy(allocation->get(),output_initial.data(),allocation->bytes(),
                cudaMemcpyHostToDevice),"V-tile operator output canary");
        for(auto* allocation:{&control_scores,&candidate_scores,&full_cta_scores,&fused_scores,
                              &tile512_scores,&scalar_dim_scores,&two_query_scores,
                              &warp_broadcast_scores,&scalar_single_load_scores,
                              &key_pair_scores})
            cuda_check(cudaMemcpy(allocation->get(),score_initial.data(),allocation->bytes(),
                cudaMemcpyHostToDevice),"V-tile operator score canary");
        const auto run=[&](DeviceBuffer& output,DeviceBuffer& scores,bool v_tile,bool full_cta,
                           bool fused=false,bool tile512=false,bool scalar_dim=false,
                           bool two_query=false,bool warp_broadcast=false,
                           bool scalar_single_load=false,
                           bool key_pair_pipeline=false) {
            ninfer::exl3::exl3_exact_attention_for_test(
                static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(dk.get()),
                static_cast<const std::uint16_t*>(dv.get()),
                static_cast<std::uint16_t*>(output.get())+1,
                static_cast<float*>(scores.get())+1,
                item.rows,item.position,item.capacity,true,nullptr,
                true,  // q shared
                true,  // K half2
                true,  // V half2
                true,  // GQA pair dependency
                true,  // GQA triple dependency
                false, // triple values128
                true,  // staged triple softmax dependency
                false, // fused six-head route
                true,  // six score shards
                false, // sharded values
                false, // four-way triple values
                true,  // selected six-softmax/triple-values route
                false, // packed triples
                false, // extent-selected score shards
                false, // query-pair score shards
                false, // legacy V-tile16
                full_cta,
                fused,
                tile512,
                false, // score K-tile64
                scalar_dim,
                two_query,
                {},    // segmented pages
                false, // pair dimensions
                false, // six values single load
                v_tile, // candidate V-tile64
                warp_broadcast,
                scalar_single_load,
                key_pair_pipeline);
        };
        run(control_output,control_scores,false,false);
        run(candidate_output,candidate_scores,true,false);
        run(full_cta_output,full_cta_scores,false,true);
        run(fused_output,fused_scores,false,false,true);
        run(tile512_output,tile512_scores,false,false,false,true);
        run(scalar_dim_output,scalar_dim_scores,false,false,false,false,true);
        run(two_query_output,two_query_scores,false,false,false,false,false,true);
        run(warp_broadcast_output,warp_broadcast_scores,false,false,false,false,
            false,false,true);
        run(scalar_single_load_output,scalar_single_load_scores,false,false,false,false,
            false,false,false,true);
        run(key_pair_output,key_pair_scores,false,false,false,false,
            false,false,false,false,true);
        cuda_check(cudaDeviceSynchronize(),"V-tile operator synchronize");
        std::vector<std::uint16_t> control_bits(output_elements+2),candidate_bits(output_elements+2);
        std::vector<std::uint16_t> full_cta_bits(output_elements+2);
        std::vector<std::uint16_t> fused_bits(output_elements+2);
        std::vector<std::uint16_t> tile512_bits(output_elements+2);
        std::vector<std::uint16_t> scalar_dim_bits(output_elements+2);
        std::vector<std::uint16_t> two_query_bits(output_elements+2);
        std::vector<std::uint16_t> warp_broadcast_bits(output_elements+2);
        std::vector<std::uint16_t> scalar_single_load_bits(output_elements+2);
        std::vector<std::uint16_t> key_pair_bits(output_elements+2);
        std::vector<float> control_score_bits(score_elements+2),candidate_score_bits(score_elements+2);
        std::vector<float> full_cta_score_bits(score_elements+2);
        std::vector<float> fused_score_bits(score_elements+2);
        std::vector<float> tile512_score_bits(score_elements+2);
        std::vector<float> scalar_dim_score_bits(score_elements+2);
        std::vector<float> two_query_score_bits(score_elements+2);
        std::vector<float> warp_broadcast_score_bits(score_elements+2);
        std::vector<float> scalar_single_load_score_bits(score_elements+2);
        std::vector<float> key_pair_score_bits(score_elements+2);
        cuda_check(cudaMemcpy(control_bits.data(),control_output.get(),control_output.bytes(),
            cudaMemcpyDeviceToHost),"V-tile operator control output");
        cuda_check(cudaMemcpy(candidate_bits.data(),candidate_output.get(),candidate_output.bytes(),
            cudaMemcpyDeviceToHost),"V-tile operator candidate output");
        cuda_check(cudaMemcpy(full_cta_bits.data(),full_cta_output.get(),full_cta_output.bytes(),
            cudaMemcpyDeviceToHost),"full-CTA operator candidate output");
        cuda_check(cudaMemcpy(fused_bits.data(),fused_output.get(),fused_output.bytes(),
            cudaMemcpyDeviceToHost),"fused operator candidate output");
        cuda_check(cudaMemcpy(tile512_bits.data(),tile512_output.get(),tile512_output.bytes(),
            cudaMemcpyDeviceToHost),"tile512 operator candidate output");
        cuda_check(cudaMemcpy(scalar_dim_bits.data(),scalar_dim_output.get(),scalar_dim_output.bytes(),
            cudaMemcpyDeviceToHost),"scalar-dim operator candidate output");
        cuda_check(cudaMemcpy(two_query_bits.data(),two_query_output.get(),two_query_output.bytes(),
            cudaMemcpyDeviceToHost),"two-query operator candidate output");
        cuda_check(cudaMemcpy(warp_broadcast_bits.data(),warp_broadcast_output.get(),
            warp_broadcast_output.bytes(),cudaMemcpyDeviceToHost),
            "warp-score-broadcast operator candidate output");
        cuda_check(cudaMemcpy(scalar_single_load_bits.data(),scalar_single_load_output.get(),
            scalar_single_load_output.bytes(),cudaMemcpyDeviceToHost),
            "scalar-single-load operator candidate output");
        cuda_check(cudaMemcpy(key_pair_bits.data(),key_pair_output.get(),
            key_pair_output.bytes(),cudaMemcpyDeviceToHost),
            "key-pair-pipeline operator candidate output");
        cuda_check(cudaMemcpy(control_score_bits.data(),control_scores.get(),control_scores.bytes(),
            cudaMemcpyDeviceToHost),"V-tile operator control scores");
        cuda_check(cudaMemcpy(candidate_score_bits.data(),candidate_scores.get(),candidate_scores.bytes(),
            cudaMemcpyDeviceToHost),"V-tile operator candidate scores");
        cuda_check(cudaMemcpy(full_cta_score_bits.data(),full_cta_scores.get(),full_cta_scores.bytes(),
            cudaMemcpyDeviceToHost),"full-CTA operator candidate scores");
        cuda_check(cudaMemcpy(fused_score_bits.data(),fused_scores.get(),fused_scores.bytes(),
            cudaMemcpyDeviceToHost),"fused operator candidate scores");
        cuda_check(cudaMemcpy(tile512_score_bits.data(),tile512_scores.get(),tile512_scores.bytes(),
            cudaMemcpyDeviceToHost),"tile512 operator candidate scores");
        cuda_check(cudaMemcpy(scalar_dim_score_bits.data(),scalar_dim_scores.get(),scalar_dim_scores.bytes(),
            cudaMemcpyDeviceToHost),"scalar-dim operator candidate scores");
        cuda_check(cudaMemcpy(two_query_score_bits.data(),two_query_scores.get(),two_query_scores.bytes(),
            cudaMemcpyDeviceToHost),"two-query operator candidate scores");
        cuda_check(cudaMemcpy(warp_broadcast_score_bits.data(),warp_broadcast_scores.get(),
            warp_broadcast_scores.bytes(),cudaMemcpyDeviceToHost),
            "warp-score-broadcast operator candidate scores");
        cuda_check(cudaMemcpy(scalar_single_load_score_bits.data(),scalar_single_load_scores.get(),
            scalar_single_load_scores.bytes(),cudaMemcpyDeviceToHost),
            "scalar-single-load operator candidate scores");
        cuda_check(cudaMemcpy(key_pair_score_bits.data(),key_pair_scores.get(),
            key_pair_scores.bytes(),cudaMemcpyDeviceToHost),
            "key-pair-pipeline operator candidate scores");
        require(control_bits.front()==output_guard&&control_bits.back()==output_guard&&
                candidate_bits.front()==output_guard&&candidate_bits.back()==output_guard,
            "V-tile operator output guard case="+std::to_string(case_index));
        require(full_cta_bits.front()==output_guard&&full_cta_bits.back()==output_guard,
            "full-CTA operator output guard case="+std::to_string(case_index));
        require(fused_bits.front()==output_guard&&fused_bits.back()==output_guard,
            "fused operator output guard case="+std::to_string(case_index));
        require(tile512_bits.front()==output_guard&&tile512_bits.back()==output_guard,
            "tile512 operator output guard case="+std::to_string(case_index));
        require(scalar_dim_bits.front()==output_guard&&scalar_dim_bits.back()==output_guard,
            "scalar-dim operator output guard case="+std::to_string(case_index));
        require(two_query_bits.front()==output_guard&&two_query_bits.back()==output_guard,
            "two-query operator output guard case="+std::to_string(case_index));
        require(warp_broadcast_bits.front()==output_guard&&
                warp_broadcast_bits.back()==output_guard,
            "warp-score-broadcast operator output guard case="+
                std::to_string(case_index));
        require(scalar_single_load_bits.front()==output_guard&&
                scalar_single_load_bits.back()==output_guard,
            "scalar-single-load operator output guard case="+
                std::to_string(case_index));
        require(key_pair_bits.front()==output_guard&&key_pair_bits.back()==output_guard,
            "key-pair-pipeline operator output guard case="+
                std::to_string(case_index));
        require(control_score_bits.front()==score_guard&&control_score_bits.back()==score_guard&&
                candidate_score_bits.front()==score_guard&&candidate_score_bits.back()==score_guard,
            "V-tile operator score guard case="+std::to_string(case_index));
        require(full_cta_score_bits.front()==score_guard&&full_cta_score_bits.back()==score_guard,
            "full-CTA operator score guard case="+std::to_string(case_index));
        require(fused_score_bits.front()==score_guard&&fused_score_bits.back()==score_guard,
            "fused operator score guard case="+std::to_string(case_index));
        require(tile512_score_bits.front()==score_guard&&tile512_score_bits.back()==score_guard,
            "tile512 operator score guard case="+std::to_string(case_index));
        require(scalar_dim_score_bits.front()==score_guard&&scalar_dim_score_bits.back()==score_guard,
            "scalar-dim operator score guard case="+std::to_string(case_index));
        require(two_query_score_bits.front()==score_guard&&two_query_score_bits.back()==score_guard,
            "two-query operator score guard case="+std::to_string(case_index));
        require(warp_broadcast_score_bits.front()==score_guard&&
                warp_broadcast_score_bits.back()==score_guard,
            "warp-score-broadcast operator score guard case="+
                std::to_string(case_index));
        require(scalar_single_load_score_bits.front()==score_guard&&
                scalar_single_load_score_bits.back()==score_guard,
            "scalar-single-load operator score guard case="+
                std::to_string(case_index));
        require(key_pair_score_bits.front()==score_guard&&key_pair_score_bits.back()==score_guard,
            "key-pair-pipeline operator score guard case="+
                std::to_string(case_index));
        require(control_bits==candidate_bits,
            "V-tile operator output mismatch case="+std::to_string(case_index));
        require(control_score_bits==candidate_score_bits,
            "V-tile operator normalized-score mismatch case="+std::to_string(case_index));
        require(control_bits==full_cta_bits,
            "full-CTA operator output mismatch case="+std::to_string(case_index));
        require(control_score_bits==full_cta_score_bits,
            "full-CTA operator normalized-score mismatch case="+std::to_string(case_index));
        require(control_bits==fused_bits,
            "fused operator output mismatch case="+std::to_string(case_index));
        require(control_score_bits==fused_score_bits,
            "fused operator normalized-score mismatch case="+std::to_string(case_index));
        require(control_bits==tile512_bits,
            "tile512 operator output mismatch case="+std::to_string(case_index));
        require(control_score_bits==tile512_score_bits,
            "tile512 operator normalized-score mismatch case="+std::to_string(case_index));
        require(control_bits==scalar_dim_bits,
            "scalar-dim operator output mismatch case="+std::to_string(case_index));
        require(control_score_bits==scalar_dim_score_bits,
            "scalar-dim operator normalized-score mismatch case="+std::to_string(case_index));
        require(control_bits==two_query_bits,
            "two-query operator output mismatch case="+std::to_string(case_index));
        require(control_score_bits==two_query_score_bits,
            "two-query operator normalized-score mismatch case="+std::to_string(case_index));
        require(control_bits==warp_broadcast_bits,
            "warp-score-broadcast operator output mismatch case="+
                std::to_string(case_index));
        require(control_score_bits==warp_broadcast_score_bits,
            "warp-score-broadcast operator normalized-score mismatch case="+
                std::to_string(case_index));
        require(control_bits==scalar_single_load_bits,
            "scalar-single-load operator output mismatch case="+
                std::to_string(case_index));
        require(control_score_bits==scalar_single_load_score_bits,
            "scalar-single-load operator normalized-score mismatch case="+
                std::to_string(case_index));
        require(control_bits==key_pair_bits,
            "key-pair-pipeline operator output mismatch case="+
                std::to_string(case_index));
        require(control_score_bits==key_pair_score_bits,
            "key-pair-pipeline operator normalized-score mismatch case="+
                std::to_string(case_index));
        output_values+=output_elements;
        score_values+=score_elements;
    }
    require(ninfer::exl3::Exl3FullAttentionLayer::
                gqa_six_softmax_triple_v_tile_shared_bytes()==8192,
        "V-tile16 operator shared-memory accounting");
    require(ninfer::exl3::Exl3FullAttentionLayer::
                gqa_six_softmax_triple_v_tile_shared_bytes(64)==32768,
        "V-tile64 operator shared-memory accounting");
    std::cout<<"EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_V_TILE_OPERATOR PASS"
        <<" cases="<<cases.size()<<" output_values="<<output_values
        <<" score_values="<<score_values
        <<" output_bit_exact=1 normalized_scores_bit_exact=1 full_cta_bit_exact=1 fused_bit_exact=1 tile512_bit_exact=1 scalar_dim_bit_exact=1 two_query_bit_exact=1 guards=1"
        <<" warp_score_broadcast_bit_exact=1 poisoned_tail=1 tile64_shared_bytes=32768"
        <<" scalar_single_load_bit_exact=1"
        <<" key_pair_pipeline_bit_exact=1"
        <<std::endl;
}
