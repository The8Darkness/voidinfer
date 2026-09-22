#pragma once

void run_exact_host_kv_small_m_fused_gate_up_transform(
    Exl3TextModel& target,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    constexpr int prefix=4096,continuation_rows=8;
    require(target.max_context()==4352 && code.size()>=prefix &&
            prose.size()>=prefix,
        "small-M fused gate/up fixture/context extent");
    configure_exact_host_kv_pinned_d2h();
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H","1");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV_BANKED_D2H","1");
    _putenv_s("NINFER_EXL3_GDN_DUAL_INPUT_TRANSFORM","0");
    _putenv_s("NINFER_EXL3_GDN_FUSED_GATE_UP_TRANSFORM","0");
    _putenv_s("NINFER_EXL3_EAGER_MLP_GATEUP_CONCURRENT","0");
    _putenv_s("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION","1");
    _putenv_s("NINFER_EXL3_EXTENDED_STREAM_REDUCTION","1");
    _putenv_s("NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M","1");
    _putenv_s("NINFER_EXL3_TARGET_K6_SMALL_M_ASYNC_A","1");
    _putenv_s("NINFER_EXL3_TARGET_K7_SMALL_M_ASYNC_A","1");
    const auto retirement_before=
        Exl3TextContext::retirement_quarantine_witness();
    {
        _putenv_s("NINFER_EXL3_SMALL_M_FUSED_GATE_UP_TRANSFORM","0");
        auto control=target.create_context(true);
        control->prepare_continuation(continuation_rows);
        _putenv_s("NINFER_EXL3_SMALL_M_FUSED_GATE_UP_TRANSFORM","1");
        auto candidate=target.create_context(true);
        candidate->prepare_continuation(continuation_rows);
        require(candidate->persistent_bytes()==control->persistent_bytes() &&
                candidate->base_linear_owner_metadata_bytes()==
                    control->base_linear_owner_metadata_bytes(),
            "small-M fused gate/up changed resource inventory");
        struct Result {
            std::vector<std::uint16_t> logits;
            std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
            std::uint64_t calls=0;
        };
        const auto continuation=[&](Exl3TextContext& context,
                                    const std::vector<std::int64_t>& source) {
            const auto request=Request::initialize(context,
                std::span<const std::int64_t>(source.data(),prefix),1024);
            const auto before=context.fused_gate_up_submissions();
            context.continue_rows(std::span<const std::int64_t>(
                source.data()+prefix-continuation_rows,continuation_rows));
            auto logits=context.continuation_logits_bits_host();
            context.finish_exact_continuation();
            return Result{std::move(logits),context.export_exact_host_state(),
                context.fused_gate_up_submissions()-before};
        };
        for(const auto& fixture:std::array<std::pair<const char*,
            const std::vector<std::int64_t>*>,2>{std::pair{"code",&code},
            std::pair{"prose",&prose}}) {
            const auto baseline=continuation(*control,*fixture.second);
            const auto changed=continuation(*candidate,*fixture.second);
            require(baseline.logits==changed.logits &&
                    baseline.state->same_payload(*changed.state),
                "small-M fused gate/up continuation logits/state mismatch");
            require(baseline.calls==0 && changed.calls==64,
                "small-M fused gate/up continuation counter mismatch");
            std::cout<<"SMALL_M_FUSED_GATEUP fixture="<<fixture.first
                <<" prefix="<<prefix<<" rows="<<continuation_rows
                <<" control_calls="<<baseline.calls
                <<" candidate_calls="<<changed.calls<<std::endl;
        }
        struct DecodeResult {
            std::vector<float> logits;
            std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> state;
            std::uint64_t calls=0;
        };
        const auto decode_one=[&](Exl3TextContext& context) {
            const auto request=Request::initialize(context,
                std::span<const std::int64_t>(code.data(),prefix),1024);
            const auto before=context.fused_gate_up_submissions();
            const auto root=context.logits_host();
            require(!root.empty(),"small-M fused gate/up root logits missing");
            const auto token=static_cast<std::int64_t>(
                std::max_element(root.begin(),root.end())-root.begin());
            context.decode(token);
            auto logits=context.logits_host();
            return DecodeResult{std::move(logits),context.export_exact_host_state(),
                context.fused_gate_up_submissions()-before};
        };
        const auto baseline=decode_one(*control);
        const auto changed=decode_one(*candidate);
        require(baseline.logits==changed.logits &&
                baseline.state->same_payload(*changed.state),
            "small-M fused gate/up M1 logits/state mismatch");
        require(baseline.calls==0 && changed.calls==64,
            "small-M fused gate/up M1 counter mismatch");
        std::cout<<"SMALL_M_FUSED_GATEUP fixture=code_m1 prefix="<<prefix
            <<" rows=1 control_calls="<<baseline.calls
            <<" candidate_calls="<<changed.calls<<std::endl;
    }
    require(Exl3TextContext::retirement_quarantine_witness()==retirement_before,
        "small-M fused gate/up normal destruction changed retirement quarantine");
    for(const char* malformed:{"2","true","01"}) {
        _putenv_s("NINFER_EXL3_SMALL_M_FUSED_GATE_UP_TRANSFORM",malformed);
        bool rejected=false;
        try {auto invalid=target.create_context(true);}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"small-M fused gate/up accepted malformed flag");
    }
    _putenv_s("NINFER_EXL3_SMALL_M_FUSED_GATE_UP_TRANSFORM","0");
    std::cout<<"SMALL_M_FUSED_GATEUP PASS fixtures=2 prefix="<<prefix
        <<" continuation_rows=8 m1_boundary=1 exact_logits_state=1"
        <<" route_counters=1 zero_extra_bytes=1 normal_retirement=1"
        <<" malformed=1"<<std::endl;
}
