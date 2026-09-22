#pragma once

void run_numeric_attention_tiled_t71_oracle(bool splitk=false,bool rows2=false) {
    struct Shape { int capacity; int rows; };
    const std::array<Shape,7> shapes{{
        {31,1},{321,1},{321,8},{4096,1},{4096,8},{4096,16},{16384,1}}};
    const auto f16=[](float x){return __half_as_ushort(__float2half_rn(x));};
    double four_k_baseline=0.0,four_k_candidate=0.0;
    double sixteen_k_baseline=0.0,sixteen_k_candidate=0.0;
    double global_max_abs=0.0,global_rel_l2=0.0,global_fp64_max_abs=0.0;
    int bit_mismatches=0,cases=0;

    for(const auto shape:shapes) {
        const int capacity=shape.capacity,rows=shape.rows,position=capacity-rows;
        const std::size_t output_elements=static_cast<std::size_t>(rows)*24*256;
        const std::size_t cache_elements=static_cast<std::size_t>(capacity)*4*256;
        const std::size_t score_elements=static_cast<std::size_t>(std::min(rows,16))*24*capacity;
        std::vector<std::uint16_t> q(output_elements),k(cache_elements),v(cache_elements);
        for(std::size_t i=0;i<q.size();++i)
            q[i]=f16(static_cast<float>(0.3*std::sin(i*0.017)+0.1*std::cos(i*0.043)));
        for(std::size_t i=0;i<k.size();++i) {
            k[i]=f16(static_cast<float>(0.4*std::sin(i*0.031)+(i%997==0?1.0:0.0)));
            v[i]=f16(static_cast<float>(std::cos(i*0.029)*1.3-std::sin(i*0.073)*0.4));
        }
        constexpr std::uint16_t guard=0x5b7d;
        DeviceBuffer dq(q.size()*2),dk(k.size()*2),dv(v.size()*2);
        const std::size_t split_workspace_bytes=splitk?
            ninfer::exl3::exl3_numeric_attention_splitk_workspace_bytes(rows,capacity):0;
        DeviceBuffer baseline((output_elements+2)*2),candidate((output_elements+2)*2),
            repeat((output_elements+2)*2),scores(score_elements*4),
            split_workspace(split_workspace_bytes+2*sizeof(float));
        cuda_check(cudaMemcpy(dq.get(),q.data(),q.size()*2,cudaMemcpyHostToDevice),"T71 upload Q");
        cuda_check(cudaMemcpy(dk.get(),k.data(),k.size()*2,cudaMemcpyHostToDevice),"T71 upload K");
        cuda_check(cudaMemcpy(dv.get(),v.data(),v.size()*2,cudaMemcpyHostToDevice),"T71 upload V");
        std::vector<std::uint16_t> initialized(output_elements+2,guard);
        for(auto* buffer:{&baseline,&candidate,&repeat})
            cuda_check(cudaMemcpy(buffer->get(),initialized.data(),buffer->bytes(),cudaMemcpyHostToDevice),"T71 output canary");
        const float workspace_guard=193.75f;
        cuda_check(cudaMemcpy(split_workspace.get(),&workspace_guard,sizeof(float),cudaMemcpyHostToDevice),"T71 workspace front canary");
        cuda_check(cudaMemcpy(static_cast<float*>(split_workspace.get())+1+split_workspace_bytes/sizeof(float),
            &workspace_guard,sizeof(float),cudaMemcpyHostToDevice),"T71 workspace back canary");

        const auto run_baseline=[&] {
            ninfer::exl3::exl3_exact_attention_for_test(
                static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(dk.get()),
                static_cast<const std::uint16_t*>(dv.get()),
                static_cast<std::uint16_t*>(baseline.get())+1,
                static_cast<float*>(scores.get()),rows,position,capacity,true,
                nullptr,true,true,true,true,true,false,true,false,true,false,false,
                false,false,true);
        };
        const auto run_candidate=[&](DeviceBuffer& destination) {
            if(splitk) ninfer::exl3::exl3_numeric_attention_splitk_for_test(
                static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(dk.get()),
                static_cast<const std::uint16_t*>(dv.get()),
                static_cast<std::uint16_t*>(destination.get())+1,
                static_cast<float*>(split_workspace.get())+1,split_workspace_bytes,
                rows,position,capacity);
            else if(rows2) ninfer::exl3::exl3_numeric_attention_rows2_for_test(
                static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(dk.get()),
                static_cast<const std::uint16_t*>(dv.get()),
                static_cast<std::uint16_t*>(destination.get())+1,
                rows,position,capacity);
            else ninfer::exl3::exl3_numeric_attention_tiled_for_test(
                static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(dk.get()),
                static_cast<const std::uint16_t*>(dv.get()),
                static_cast<std::uint16_t*>(destination.get())+1,
                rows,position,capacity);
        };
        run_baseline();run_candidate(candidate);run_candidate(repeat);
        cuda_check(cudaDeviceSynchronize(),"T71 synchronize");
        std::vector<std::uint16_t> base_bits(output_elements+2),candidate_bits(output_elements+2),
            repeat_bits(output_elements+2),q_after(q.size()),k_after(k.size()),v_after(v.size());
        cuda_check(cudaMemcpy(base_bits.data(),baseline.get(),baseline.bytes(),cudaMemcpyDeviceToHost),"T71 baseline output");
        cuda_check(cudaMemcpy(candidate_bits.data(),candidate.get(),candidate.bytes(),cudaMemcpyDeviceToHost),"T71 candidate output");
        cuda_check(cudaMemcpy(repeat_bits.data(),repeat.get(),repeat.bytes(),cudaMemcpyDeviceToHost),"T71 repeat output");
        cuda_check(cudaMemcpy(q_after.data(),dq.get(),q.size()*2,cudaMemcpyDeviceToHost),"T71 Q immutability");
        cuda_check(cudaMemcpy(k_after.data(),dk.get(),k.size()*2,cudaMemcpyDeviceToHost),"T71 K immutability");
        cuda_check(cudaMemcpy(v_after.data(),dv.get(),v.size()*2,cudaMemcpyDeviceToHost),"T71 V immutability");
        require(base_bits.front()==guard&&base_bits.back()==guard&&
                candidate_bits.front()==guard&&candidate_bits.back()==guard&&
                repeat_bits.front()==guard&&repeat_bits.back()==guard,
                "T71 output canary");
        require(candidate_bits==repeat_bits,"T71 deterministic repeat");
        require(q_after==q&&k_after==k&&v_after==v,"T71 input mutation");
        float workspace_front=0.0f,workspace_back=0.0f;
        cuda_check(cudaMemcpy(&workspace_front,split_workspace.get(),sizeof(float),cudaMemcpyDeviceToHost),"T71 workspace front guard");
        cuda_check(cudaMemcpy(&workspace_back,static_cast<float*>(split_workspace.get())+1+
            split_workspace_bytes/sizeof(float),sizeof(float),cudaMemcpyDeviceToHost),"T71 workspace back guard");
        require(workspace_front==workspace_guard&&workspace_back==workspace_guard,
            "T71 workspace canary");

        double diff_squared=0.0,base_squared=0.0,max_abs=0.0,fp64_max_abs=0.0;
        int local_mismatches=0;
        for(std::size_t i=0;i<output_elements;++i) {
            const double a=half_to_float(base_bits[i+1]);
            const double b=half_to_float(candidate_bits[i+1]);
            require(std::isfinite(b),"T71 non-finite candidate output");
            const double difference=std::abs(a-b);
            max_abs=std::max(max_abs,difference);
            diff_squared+=difference*difference;base_squared+=a*a;
            if(base_bits[i+1]!=candidate_bits[i+1]) ++local_mismatches;
        }
        const double rel_l2=std::sqrt(diff_squared/std::max(base_squared,1.0e-30));

        // Independent FP64 oracle from the represented public inputs. This is
        // intentionally independent of both CUDA reduction organizations.
        for(int row=0;row<rows;++row) for(int head=0;head<24;++head) {
            const int count=position+row+1,kv_head=head/6;
            std::vector<double> weights(count);
            double maximum=-std::numeric_limits<double>::infinity();
            for(int token=0;token<count;++token) {
                double dot=0.0;
                for(int d=0;d<256;++d)
                    dot+=static_cast<double>(half_to_float(q[(row*24+head)*256+d]))*
                        half_to_float(k[(static_cast<std::size_t>(token)*4+kv_head)*256+d]);
                weights[token]=dot/16.0;maximum=std::max(maximum,weights[token]);
            }
            double denominator=0.0;
            for(auto& weight:weights){weight=std::exp(weight-maximum);denominator+=weight;}
            for(int d=0;d<256;++d) {
                double expected=0.0;
                for(int token=0;token<count;++token)
                    expected+=weights[token]/denominator*
                        half_to_float(v[(static_cast<std::size_t>(token)*4+kv_head)*256+d]);
                const double actual=half_to_float(candidate_bits[1+(row*24+head)*256+d]);
                const double error=std::abs(actual-expected);
                fp64_max_abs=std::max(fp64_max_abs,error);
                require(error<=0.001+0.002*std::abs(expected),"T71 FP64 oracle mismatch");
            }
        }
        require(max_abs<=0.00390625,"T71 baseline max-abs gate");
        require(rel_l2<=0.002,"T71 baseline relative-L2 gate");

        const auto elapsed=[&](bool changed) {
            cudaEvent_t start=nullptr,stop=nullptr;cuda_check(cudaEventCreate(&start),"T71 event start");
            cuda_check(cudaEventCreate(&stop),"T71 event stop");
            std::vector<float> samples;
            for(int iteration=0;iteration<5;++iteration) {
                cuda_check(cudaEventRecord(start),"T71 record start");
                if(changed) run_candidate(candidate); else run_baseline();
                cuda_check(cudaEventRecord(stop),"T71 record stop");
                cuda_check(cudaEventSynchronize(stop),"T71 time synchronize");
                float ms=0.0f;cuda_check(cudaEventElapsedTime(&ms,start,stop),"T71 elapsed");
                samples.push_back(ms);
            }
            cuda_check(cudaEventDestroy(start),"T71 destroy start");
            cuda_check(cudaEventDestroy(stop),"T71 destroy stop");
            std::sort(samples.begin(),samples.end());return static_cast<double>(samples[2]);
        };
        double baseline_ms=0.0,candidate_ms=0.0;
        if((capacity==4096&&rows==16)||(capacity==16384&&rows==1)) {
            baseline_ms=elapsed(false);candidate_ms=elapsed(true);
            if(capacity==4096){four_k_baseline=baseline_ms;four_k_candidate=candidate_ms;}
            else {sixteen_k_baseline=baseline_ms;sixteen_k_candidate=candidate_ms;}
        }
        global_max_abs=std::max(global_max_abs,max_abs);
        global_rel_l2=std::max(global_rel_l2,rel_l2);
        global_fp64_max_abs=std::max(global_fp64_max_abs,fp64_max_abs);
        bit_mismatches+=local_mismatches;++cases;
        std::cout<<(rows2?"T73_NUMERIC_ROWS2":"T71_NUMERIC_TILE")<<" case="<<cases<<" capacity="<<capacity
            <<" rows="<<rows<<" max_abs="<<max_abs<<" rel_l2="<<rel_l2
            <<" fp64_max_abs="<<fp64_max_abs<<" bit_mismatches="<<local_mismatches
            <<" baseline_ms="<<baseline_ms<<" candidate_ms="<<candidate_ms
            <<" score_bytes_avoided="<<score_elements*4
            <<" workspace_bytes="<<split_workspace_bytes
            <<" blocks="<<(splitk?rows*4*((capacity+1023)/1024)+rows*4:rows*4)
            <<" threads=256 variant="<<(splitk?"splitk1024":rows2?"rows2":"single_cta")<<"\n";
    }
    require(four_k_candidate<four_k_baseline,"T71 4K operator performance gate");
    require(sixteen_k_candidate<=1.10*sixteen_k_baseline,"T71 16K operator guard");
    std::cout<<(splitk?"T71B_NUMERIC_SPLITK PASS cases=":rows2?"T73_NUMERIC_ROWS2 PASS cases=":"T71_NUMERIC_TILED PASS cases=")<<cases
        <<" max_abs="<<global_max_abs<<" max_rel_l2="<<global_rel_l2
        <<" fp64_max_abs="<<global_fp64_max_abs<<" bit_mismatches="<<bit_mismatches
        <<" four_k_gain_pct="<<(four_k_baseline-four_k_candidate)*100.0/four_k_baseline
        <<" sixteen_k_gain_pct="<<(sixteen_k_baseline-sixteen_k_candidate)*100.0/sixteen_k_baseline
        <<" deterministic=1 inputs_immutable=1 canaries=1 score_workspace_bytes=0 lane=NUMERIC_CANDIDATE\n";
}

void run_numeric_attention_splitk_t71b_oracle() {
    run_numeric_attention_tiled_t71_oracle(true);
}
