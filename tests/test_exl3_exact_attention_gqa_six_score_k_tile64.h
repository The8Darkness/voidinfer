#pragma once

void run_exact_attention_gqa_six_score_k_tile64_operator() {
    struct Case { int rows; int position; int capacity; };
    // Exercise every extent-selected shard count, a 1024-row chunk tail, and
    // the physical-C1 4096-history boundary.
    const std::array<Case,9> cases{{
        {1,0,32}, {1,256,512}, {1,512,1024}, {1,768,1024},
        {1,1024,2048}, {1,1280,2048}, {16,1008,2048},
        {16,2032,4096}, {16,4080,4096}}};
    constexpr std::uint16_t output_guard=0x59a7;
    constexpr float score_guard=193.75f;
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
            static_cast<std::size_t>(item.capacity)*4*256),v(k.size());
        for(std::size_t i=0;i<q.size();++i)
            q[i]=f16(static_cast<float>(0.29*std::sin(i*0.017)+
                0.11*std::cos(i*0.043)));
        for(int row=0;row<item.capacity;++row) for(int kv=0;kv<4;++kv)
            for(int d=0;d<256;++d) {
                const auto index=(static_cast<std::size_t>(row)*4+kv)*256+d;
                if(row>=max_count) {
                    k[index]=f16((row+kv+d)&1?88.0f:-88.0f);
                    v[index]=f16((row+kv+d)&1?-64.0f:64.0f);
                } else {
                    k[index]=f16(static_cast<float>(0.41*std::sin(index*0.031)+
                        0.13*std::cos(index*0.007)));
                    v[index]=f16(static_cast<float>(0.97*std::cos(index*0.021)-
                        0.23*std::sin(index*0.067)));
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
        DeviceBuffer control_scores((score_elements+2)*sizeof(float));
        DeviceBuffer candidate_scores((score_elements+2)*sizeof(float));
        cuda_check(cudaMemcpy(dq.get(),q.data(),q.size()*sizeof(std::uint16_t),
            cudaMemcpyHostToDevice),"K-tile64 upload Q");
        cuda_check(cudaMemcpy(dk.get(),k.data(),k.size()*sizeof(std::uint16_t),
            cudaMemcpyHostToDevice),"K-tile64 upload K");
        cuda_check(cudaMemcpy(dv.get(),v.data(),v.size()*sizeof(std::uint16_t),
            cudaMemcpyHostToDevice),"K-tile64 upload V");
        const std::vector<std::uint16_t> output_initial(
            output_elements+2,output_guard);
        const std::vector<float> score_initial(score_elements+2,score_guard);
        for(auto* allocation:{&control_output,&candidate_output})
            cuda_check(cudaMemcpy(allocation->get(),output_initial.data(),
                allocation->bytes(),cudaMemcpyHostToDevice),
                "K-tile64 output canary");
        for(auto* allocation:{&control_scores,&candidate_scores})
            cuda_check(cudaMemcpy(allocation->get(),score_initial.data(),
                allocation->bytes(),cudaMemcpyHostToDevice),
                "K-tile64 score canary");
        const auto run=[&](DeviceBuffer& output,DeviceBuffer& scores,
                           bool candidate) {
            ninfer::exl3::exl3_exact_attention_for_test(
                static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(dk.get()),
                static_cast<const std::uint16_t*>(dv.get()),
                static_cast<std::uint16_t*>(output.get())+1,
                static_cast<float*>(scores.get())+1,
                item.rows,item.position,item.capacity,true,nullptr,
                true,true,true,true,true,false,true,false,true,false,false,
                true,false,true,false,false,false,false,false,candidate);
        };
        run(control_output,control_scores,false);
        run(candidate_output,candidate_scores,true);
        cuda_check(cudaDeviceSynchronize(),"K-tile64 synchronize");
        std::vector<std::uint16_t> control_bits(output_elements+2),
            candidate_bits(output_elements+2);
        std::vector<float> control_score_bits(score_elements+2),
            candidate_score_bits(score_elements+2);
        cuda_check(cudaMemcpy(control_bits.data(),control_output.get(),
            control_output.bytes(),cudaMemcpyDeviceToHost),
            "K-tile64 control output");
        cuda_check(cudaMemcpy(candidate_bits.data(),candidate_output.get(),
            candidate_output.bytes(),cudaMemcpyDeviceToHost),
            "K-tile64 candidate output");
        cuda_check(cudaMemcpy(control_score_bits.data(),control_scores.get(),
            control_scores.bytes(),cudaMemcpyDeviceToHost),
            "K-tile64 control scores");
        cuda_check(cudaMemcpy(candidate_score_bits.data(),candidate_scores.get(),
            candidate_scores.bytes(),cudaMemcpyDeviceToHost),
            "K-tile64 candidate scores");
        require(control_bits==candidate_bits &&
                control_score_bits==candidate_score_bits &&
                control_bits.front()==output_guard &&
                control_bits.back()==output_guard &&
                candidate_bits.front()==output_guard &&
                candidate_bits.back()==output_guard &&
                control_score_bits.front()==score_guard &&
                control_score_bits.back()==score_guard &&
                candidate_score_bits.front()==score_guard &&
                candidate_score_bits.back()==score_guard,
                "K-tile64 exact differential case="+
                    std::to_string(case_index));
        output_values+=output_elements;score_values+=score_elements;
    }
    require(ninfer::exl3::Exl3FullAttentionLayer::
                gqa_six_score_k_tile64_shared_bytes()==38912,
            "K-tile64 shared-memory accounting");
    std::cout << "EXACT_ATTENTION_GQA_SIX_SCORE_K_TILE64 PASS cases="
              << cases.size() << " output_values=" << output_values
              << " score_values=" << score_values
              << " output_bit_exact=1 normalized_scores_bit_exact=1 guards=1"
              << " poisoned_tail=1 shards_1_to_6=1 physical_c1=1 shared_bytes=38912\n";
}
