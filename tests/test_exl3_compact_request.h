#pragma once

void run_compact_request_qualification(Exl3TextModel& target,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& source) {
    using Request=ninfer::exl3::Exl3VeriCacheRequest;
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    auto exact=target.create_context(true);exact->prepare_continuation(8);
    std::array<std::unique_ptr<DeviceBuffer>,5> stage;std::array<std::uint16_t*,5> staging{};
    for(int tap=0;tap<5;++tap) {stage[tap]=std::make_unique<DeviceBuffer>(16ULL*kHidden*2);staging[tap]=static_cast<std::uint16_t*>(stage[tap]->get());}
    int cases=0;
    for(int prefix:{63,2061}) {
        const auto root=Request::initialize(*exact,std::span<const std::int64_t>(source.data(),prefix),1024);
        const auto compact=root->compact_draft(draft,staging);
        require(compact->compact_draft() && compact->tap_bytes()==0 && compact->state()==root->state(),"compact representation changed authority");
        std::vector<std::int64_t> greedy;
        for(int i=0;i<8;++i) {const auto token=sample_target(*exact);greedy.push_back(token);exact->decode(token);}
        for(int reject:{0,3,8}) for(int stop:{0,3,8}) {
            auto proposal=greedy;if(reject<8) proposal[reject]=(proposal[reject]+1)%kVocab;
            std::vector<std::int64_t> terminal;if(stop<8) terminal.push_back(greedy[stop]);
            const auto normal=root->verify(*exact,proposal,terminal);
            const auto candidate=compact->verify_compact(*exact,draft,staging,proposal,terminal);
            require(normal.second.committed_tokens==candidate.second.committed_tokens && normal.second.accepted==candidate.second.accepted &&
                normal.second.rejected==candidate.second.rejected && normal.second.stopped==candidate.second.stopped &&
                normal.second.committed_taps==candidate.second.committed_taps && normal.first->state()->same_payload(*candidate.first->state()) &&
                normal.first->token_suffix()==candidate.first->token_suffix() && candidate.first->is_child_of(*compact),
                "compact outer authority/state/token mismatch");
            normal.first->restore_draft(draft,staging);const auto reference=draft.export_host_ring(nullptr,false);
            candidate.first->restore_draft(draft,staging);
            require(reference->same_payload(*draft.export_host_ring(nullptr,false)),"compact confirmed projection changed draft state");
            ++cases;
        }
        bool refused=false;try {compact->verify(*exact,greedy);} catch(const std::logic_error&) {refused=true;}
        require(refused,"compact request silently used tap-only verification");
        refused=false;try {compact->append_prompt(*exact,greedy);} catch(const std::logic_error&) {refused=true;}
        require(refused,"compact request silently lost prompt prefix conditioning");
        refused=false;try {compact->same_taps(*root);} catch(const std::logic_error&) {refused=true;}
        require(refused,"compact state pretended discarded target taps still exist");
        refused=false;try {compact->verify_compact(*exact,draft,staging,{});} catch(const std::invalid_argument&) {refused=true;}
        require(refused && exact->export_exact_host_state()->same_payload(*root->state()),"compact failure did not restore authoritative root");
        root->restore_draft(draft,staging);const auto reference=draft.export_host_ring(nullptr,false);
        compact->restore_draft(draft,staging);require(reference->same_payload(*draft.export_host_ring(nullptr,false)),"compact failure changed root conditioning");
    }
    std::cout << "COMPACT_REQUEST PASS cases=" << cases << " rejected_suffix=exact eos=exact projected_state=exact failure_repair=exact\n";
}
