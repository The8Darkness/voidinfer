#pragma once

void run_turboangle_host_oracle() {
    using ninfer::exl3::TurboAngle256;
    constexpr double pi=3.1415926535897932384626433832795;
    int cases=0;
    for(int fixture=0;fixture<5;++fixture) for(bool key:{false,true}) {
        std::array<float,256> input{};
        for(int i=0;i<256;++i) {
            if(fixture==1) input[i]=i==73?3.0f:0.0f;
            if(fixture==2) input[i]=static_cast<float>(std::sin(i*0.17)*2.0+std::cos(i*0.031));
            if(fixture==3) input[i]=(i&1)?-0.125f:0.125f;
            if(fixture==4) input[i]=static_cast<float>((i-127)*0.013);
        }
        const auto record=TurboAngle256::encode(input,key);
        const auto actual=TurboAngle256::decode(record,key);
        std::array<double,256> transformed{};
        // Independent dense mathematical H*D oracle, no FWHT butterflies.
        for(int row=0;row<256;++row) for(int col=0;col<256;++col) {
            const double h=(std::popcount(static_cast<unsigned>(row&col))&1)?-0.0625:0.0625;
            transformed[row]+=h*TurboAngle256::sign(col)*input[col];
        }
        float low,scale; std::memcpy(&low,record.data(),4); std::memcpy(&scale,record.data()+4,4);
        const int ab=key?7:6,nb=key?8:4,bins=1<<ab;
        const auto bits=[&](int offset,int count) {
            unsigned value=0;
            for(int j=0;j<count;++j) if(record[(offset+j)/8]&(1U<<((offset+j)%8))) value+=1U<<j;
            return value;
        };
        std::array<double,256> polar{};
        for(int i=0;i<128;++i) {
            const unsigned a=bits(64+i*(ab+nb),ab),n=bits(64+i*(ab+nb)+ab,nb);
            // Quantization is evaluated from the represented full input formula.
            const auto expected_angle=static_cast<int>(std::lround(bins*std::atan2(transformed[2*i+1],transformed[2*i])/(2*pi)))&(bins-1);
            const double input_radius=std::hypot(transformed[2*i],transformed[2*i+1]);
            if(input_radius>1e-12) require(a==expected_angle,"TurboAngle independent angle oracle");
            const double norm_value=key?input_radius:std::log(std::max(input_radius,1e-30));
            if(scale>0 && input_radius>1e-12)
                require(std::abs(norm_value-(low+n*static_cast<double>(scale)))<=scale*0.501+1e-6,
                        "TurboAngle independent norm oracle");
            double radius=low+n*static_cast<double>(scale);
            if(!key) radius=std::exp(radius);
            polar[2*i]=radius*std::cos(2*pi*a/bins); polar[2*i+1]=radius*std::sin(2*pi*a/bins);
        }
        for(int row=0;row<256;++row) {
            double expected=0;
            for(int col=0;col<256;++col) {
                const double h=(std::popcount(static_cast<unsigned>(row&col))&1)?-0.0625:0.0625;
                expected+=h*polar[col];
            }
            expected*=TurboAngle256::sign(row);
            require(std::isfinite(actual[row]) && std::abs(expected-actual[row])<1e-5,
                    "TurboAngle independent dense inverse oracle");
        }
        ++cases;
    }
    std::cout << "TURBOANGLE_HOST_ORACLE PASS cases=" << cases << '\n';
}

void run_warm_tier_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    require(source.size()>=350 && target.max_context()==1024,"warmtier fixture extent");
    run_turboangle_host_oracle();
    auto exact=target.create_context(true);
    auto inner=target.create_context(true);
    require(inner->try_enable_oscar_from_environment(),"warmtier OSCAR");
    exact->prefill(std::span<const std::int64_t>(source.data(),16));
    for(int i=16;i<321;++i) exact->decode(source[i]);
    const auto l2=exact->export_exact_host_state();
    const auto encode_start=std::chrono::steady_clock::now();
    const auto l1=Exl3TextContext::make_turboangle_warm_pages(l2,64);
    const double encode_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-encode_start).count();
    require(l1->payload_bytes()==64ULL*4*16*(248+168),"warmtier physical payload accounting");
    std::array<double,2> restore_total{};
    std::array<std::size_t,2> accepted{};
    std::vector<std::int64_t> oracle_tokens;
    for(int i=0;i<4;++i) {const auto token=sample_target(*exact); oracle_tokens.push_back(token); exact->decode(token);}
    const auto oracle=exact->export_exact_host_state();
    for(int repeat=0;repeat<3;++repeat) for(int order=0;order<2;++order) {
        const int arm=(repeat&1)?1-order:order;
        inner->reset();
        const auto start=std::chrono::steady_clock::now();
        inner->restore_oscar_host_state(*l2,arm?l1.get():nullptr);
        cuda_check(cudaDeviceSynchronize(),"warmtier restored");
        const double restore_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        restore_total[arm]+=restore_ms;
        std::vector<std::int64_t> tentative;
        for(int i=0;i<4;++i) {const auto token=sample_target(*inner); tentative.push_back(token); inner->decode(token);}
        const auto result=ninfer::exl3::verify_exl3_outer_reference(*exact,*l2,tentative);
        accepted[arm]+=result.accepted;
        require(std::equal(result.committed_tokens.begin(),result.committed_tokens.end(),oracle_tokens.begin()),
                "warmtier unauthorized outer output");
        // Repair only from authoritative L2 and check a full continuation after
        // each trial, even if warm approximation shortened the outer prefix.
        for(std::size_t i=result.committed_tokens.size();i<4;++i) exact->decode(oracle_tokens[i]);
        require(exact->export_exact_host_state()->same_payload(*oracle),"warmtier authoritative state drift");
        std::cout << "WARM_TIER_PAIR repeat=" << repeat << " l1=" << arm << " restore_ms=" << restore_ms
                  << " outer_accepted=" << result.accepted << " proposed=4\n";
    }
    // Exact image identity, not equal token strings, guards warm-page reuse.
    exact->restore_exact_host_state(*l2);
    const auto other_image=exact->export_exact_host_state();
    bool rejected=false;
    try {inner->restore_oscar_host_state(*other_image,l1.get());} catch(const std::exception&){rejected=true;}
    require(rejected,"warmtier accepted wrong image identity");
    require(exact->export_exact_host_state()->same_payload(*l2),"warmtier changed authoritative source");
    std::cout << "WARM_TIER PASS pairs=3 l1_payload_bytes=" << l1->payload_bytes()
              << " l2_covered_bytes=" << 64ULL*65536 << " encode_ms=" << encode_ms
              << " l0_l2_restore_mean_ms=" << restore_total[0]/3 << " l0_l1_l2_restore_mean_ms=" << restore_total[1]/3
              << " outer_accepted_l2=" << accepted[0] << " outer_accepted_l1=" << accepted[1]
              << " h2d_kv_bytes_each=" << 321ULL*65536 << " no_transfer_reduction=1\n";
}
