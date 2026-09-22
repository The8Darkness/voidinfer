#pragma once

struct Exl3GdnKnownOracleCase {
    std::string name;
    int rows=0;
    std::vector<std::uint16_t> q,k,v,expected_output;
    std::vector<float> g,beta,initial_state,expected_state;
};

// These expectations follow without evaluating exp/rsqrt or running an
// independent model.  Case one has zero normalized Q/K, so updates and output
// are zero while g=0 preserves the supplied FP32 state.  Case two begins with
// zero state and V=0, so arbitrary represented Q/K/control values still leave
// the complete state and output exactly zero.
std::array<Exl3GdnKnownOracleCase,2> make_gdn_known_oracle_cases() {
    constexpr int rows=2,key_heads=16,key_columns=128,value_heads=48;
    constexpr std::size_t qk=rows*key_heads*key_columns;
    constexpr std::size_t values=rows*value_heads*key_columns;
    constexpr std::size_t controls=rows*value_heads;
    constexpr auto state_elements=Exl3GdnRecurrentLayout::recurrent_elements;
    Exl3GdnKnownOracleCase identity;
    identity.name="zero_qk_state_identity";identity.rows=rows;
    identity.q.assign(qk,0);identity.k.assign(qk,0);
    identity.v.resize(values);identity.g.assign(controls,0.0f);
    identity.beta.assign(controls,1.0f);
    identity.initial_state.resize(state_elements);
    constexpr std::array<float,5> exact_state{0.0f,0.125f,0.25f,0.5f,1.0f};
    constexpr std::array<std::uint16_t,4> represented_values{0x0000,0x3f00,0xbf00,0x3f80};
    for(std::size_t i=0;i<identity.v.size();++i)
        identity.v[i]=represented_values[i%represented_values.size()];
    for(std::size_t i=0;i<identity.initial_state.size();++i)
        identity.initial_state[i]=exact_state[i%exact_state.size()];
    identity.expected_state=identity.initial_state;
    identity.expected_output.assign(values,0);

    Exl3GdnKnownOracleCase no_signal;
    no_signal.name="zero_state_zero_value";no_signal.rows=rows;
    no_signal.q.resize(qk);no_signal.k.resize(qk);no_signal.v.assign(values,0);
    no_signal.g.resize(controls);no_signal.beta.resize(controls);
    constexpr std::array<std::uint16_t,6> represented_qk{
        0x0000,0x3e80,0x3f00,0x3f80,0x4000,0x4080};
    constexpr std::array<float,4> represented_g{-2.0f,-0.5f,0.0f,1.0f};
    constexpr std::array<float,4> represented_beta{0.0f,0.25f,0.5f,1.0f};
    for(std::size_t i=0;i<qk;++i) {
        no_signal.q[i]=represented_qk[i%represented_qk.size()];
        no_signal.k[i]=represented_qk[(i*3+1)%represented_qk.size()];
    }
    for(std::size_t i=0;i<controls;++i) {
        no_signal.g[i]=represented_g[i%represented_g.size()];
        no_signal.beta[i]=represented_beta[i%represented_beta.size()];
    }
    no_signal.initial_state.assign(state_elements,0.0f);
    no_signal.expected_state=no_signal.initial_state;
    no_signal.expected_output.assign(values,0);
    return {std::move(identity),std::move(no_signal)};
}

void run_gdn_known_oracle_cases() {
    constexpr int key_heads=16,key_columns=128,value_heads=48;
    for(const auto& fixture:make_gdn_known_oracle_cases()) {
        const auto q_bytes=fixture.q.size()*sizeof(std::uint16_t);
        const auto value_bytes=fixture.v.size()*sizeof(std::uint16_t);
        const auto control_bytes=fixture.g.size()*sizeof(float);
        const auto state_bytes=fixture.initial_state.size()*sizeof(float);
        auto q=upload({reinterpret_cast<const std::byte*>(fixture.q.data()),q_bytes},"upload known oracle Q");
        auto k=upload({reinterpret_cast<const std::byte*>(fixture.k.data()),q_bytes},"upload known oracle K");
        auto v=upload({reinterpret_cast<const std::byte*>(fixture.v.data()),value_bytes},"upload known oracle V");
        auto g=upload({reinterpret_cast<const std::byte*>(fixture.g.data()),control_bytes},"upload known oracle g");
        auto beta=upload({reinterpret_cast<const std::byte*>(fixture.beta.data()),control_bytes},"upload known oracle beta");
        auto canonical_state=upload({reinterpret_cast<const std::byte*>(fixture.initial_state.data()),state_bytes},"upload known canonical state");
        auto fused_state=upload({reinterpret_cast<const std::byte*>(fixture.initial_state.data()),state_bytes},"upload known fused state");
        auto canonical_output=allocate_device(value_bytes,"allocate known canonical output");
        auto fused_output=allocate_device(value_bytes,"allocate known fused output");
        const auto normalized_bytes=static_cast<std::size_t>(fixture.rows)*key_heads*key_columns*sizeof(float);
        auto normalized_q=allocate_device(normalized_bytes,"allocate known normalized Q");
        auto normalized_k=allocate_device(normalized_bytes,"allocate known normalized K");
        auto alpha=allocate_device(static_cast<std::size_t>(fixture.rows)*value_heads*sizeof(float),"allocate known alpha");
        Exl3GdnStageFusionFixtureView view{
            static_cast<const std::uint16_t*>(q->ptr),static_cast<const std::uint16_t*>(k->ptr),
            static_cast<const std::uint16_t*>(v->ptr),static_cast<const float*>(g->ptr),
            static_cast<const float*>(beta->ptr),static_cast<float*>(canonical_state->ptr),
            static_cast<float*>(fused_state->ptr),static_cast<std::uint16_t*>(canonical_output->ptr),
            static_cast<std::uint16_t*>(fused_output->ptr),static_cast<float*>(normalized_q->ptr),
            static_cast<float*>(normalized_k->ptr),static_cast<float*>(alpha->ptr),
            q_bytes,value_bytes,control_bytes,state_bytes,normalized_bytes,fixture.rows};
        exl3_gdn_stage_fusion_fixture(view);
        cuda_check(cudaDeviceSynchronize(),"synchronize known GDN oracle case");
        const auto expected_state=std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(fixture.expected_state.data()),state_bytes);
        const auto expected_output=std::span<const std::byte>(
            reinterpret_cast<const std::byte*>(fixture.expected_output.data()),value_bytes);
        require(download_bytes(canonical_state->ptr,state_bytes,"download known canonical state")==
                    std::vector<std::byte>(expected_state.begin(),expected_state.end()) &&
                download_bytes(fused_state->ptr,state_bytes,"download known fused state")==
                    std::vector<std::byte>(expected_state.begin(),expected_state.end()),
            fixture.name+" FP32 state differs from closed-form expectation");
        require(download_bytes(canonical_output->ptr,value_bytes,"download known canonical output")==
                    std::vector<std::byte>(expected_output.begin(),expected_output.end()) &&
                download_bytes(fused_output->ptr,value_bytes,"download known fused output")==
                    std::vector<std::byte>(expected_output.begin(),expected_output.end()),
            fixture.name+" BF16 output differs from closed-form expectation");
    }
}

void run_gdn_external_future_oracle(const std::filesystem::path& path) {
    constexpr int rows=2,key_heads=16,key_columns=128,value_heads=48;
    const auto header=inspect_file(path);
    const auto tensor=[&](const char* suffix,const char* dtype,
        std::vector<std::uint64_t> shape) {
        auto value=read_tensor(path,header,std::string("basis2_")+suffix);
        require(value.info.dtype==dtype && value.info.shape==shape,
            std::string("future recurrent oracle tensor contract: ")+suffix);
        return value;
    };
    auto q=tensor("q","BF16",{rows,key_heads,key_columns});
    auto k=tensor("k","BF16",{rows,key_heads,key_columns});
    auto v=tensor("v","BF16",{rows,value_heads,key_columns});
    auto g=tensor("g","F32",{rows,value_heads});
    auto beta=tensor("beta","F32",{rows,value_heads});
    auto before=tensor("state_before","F32",{value_heads,key_columns,key_columns});
    auto state_expected=tensor("state_after","F32",{value_heads,key_columns,key_columns});
    auto output_expected=tensor("output","BF16",{rows,value_heads,key_columns});
    auto q_expected=tensor("normalized_q","F32",{rows,key_heads,key_columns});
    auto k_expected=tensor("normalized_k","F32",{rows,key_heads,key_columns});
    auto alpha_expected=tensor("alpha","F32",{rows,value_heads});
    auto q_device=upload(q.bytes(),"upload future oracle Q");
    auto k_device=upload(k.bytes(),"upload future oracle K");
    auto v_device=upload(v.bytes(),"upload future oracle V");
    auto g_device=upload(g.bytes(),"upload future oracle g");
    auto beta_device=upload(beta.bytes(),"upload future oracle beta");
    auto canonical_state=upload(before.bytes(),"upload future canonical state");
    auto fused_state=upload(before.bytes(),"upload future fused state");
    auto canonical_output=allocate_device(output_expected.bytes().size_bytes(),"allocate future canonical output");
    auto fused_output=allocate_device(output_expected.bytes().size_bytes(),"allocate future fused output");
    auto normalized_q=allocate_device(q_expected.bytes().size_bytes(),"allocate future normalized Q");
    auto normalized_k=allocate_device(k_expected.bytes().size_bytes(),"allocate future normalized K");
    auto alpha=allocate_device(alpha_expected.bytes().size_bytes(),"allocate future alpha");
    Exl3GdnStageFusionFixtureView view{
        static_cast<const std::uint16_t*>(q_device->ptr),static_cast<const std::uint16_t*>(k_device->ptr),
        static_cast<const std::uint16_t*>(v_device->ptr),static_cast<const float*>(g_device->ptr),
        static_cast<const float*>(beta_device->ptr),static_cast<float*>(canonical_state->ptr),
        static_cast<float*>(fused_state->ptr),static_cast<std::uint16_t*>(canonical_output->ptr),
        static_cast<std::uint16_t*>(fused_output->ptr),static_cast<float*>(normalized_q->ptr),
        static_cast<float*>(normalized_k->ptr),static_cast<float*>(alpha->ptr),
        q.bytes().size_bytes(),v.bytes().size_bytes(),g.bytes().size_bytes(),
        before.bytes().size_bytes(),q_expected.bytes().size_bytes(),rows};
    exl3_gdn_stage_fusion_fixture(view);
    cuda_check(cudaDeviceSynchronize(),"synchronize external GDN future oracle");
    const auto exact=[&](const void* actual,const TensorPayload& expected,const char* label) {
        require(download_bytes(actual,expected.bytes().size_bytes(),label)==
                    std::vector<std::byte>(expected.bytes().begin(),expected.bytes().end()),
            std::string("future recurrent oracle exact mismatch: ")+label);
    };
    exact(canonical_state->ptr,state_expected,"canonical FP32 state");
    exact(fused_state->ptr,state_expected,"fused FP32 state");
    exact(canonical_output->ptr,output_expected,"canonical BF16 output");
    exact(fused_output->ptr,output_expected,"fused BF16 output");
    exact(normalized_q->ptr,q_expected,"normalized FP32 Q");
    exact(normalized_k->ptr,k_expected,"normalized FP32 K");
    exact(alpha->ptr,alpha_expected,"FP32 alpha");
}
