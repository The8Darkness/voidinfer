#pragma once
#include "exl3/reconstruction_stream.h"
#include "exl3/reconstruction_config.h"
#include "test_exl3_residual_norm.h"

// Uses the T69 capture value type and independent sampled FP64 oracle.
void run_reconstructed_exact_projection(
    Exl3TextModel& target, const std::vector<std::int64_t>& source) {
    using Admission = ninfer::exl3::Exl3CudaLinearAdmission;
    if(const auto residual=env("NINFER_TEST_RESIDUAL_NORM_OPERATOR");!residual.empty()) {
        require(residual=="1","residual norm fixture option");
        run_residual_norm_operator_comparison();
    }
    const std::filesystem::path directory = env("NINFER_RECON_EXACT_OUT");
    require(!directory.empty() && !std::filesystem::exists(directory),
            "reconstructed exact output directory must be new");
    std::filesystem::create_directories(directory);
    std::ofstream summary(directory / "summary.csv"), pairs(directory / "pairs.csv");
    require(summary.good() && pairs.good(), "reconstructed exact evidence creation");
    summary << "rows,layer,operation,direct_dispatch,decoded_bytes,retained_scratch_bytes,mismatches,first_mismatch,input_unchanged,canaries,deterministic,finite,oracle_values,direct_oracle_max_abs,direct_oracle_rel_l2,candidate_oracle_max_abs,candidate_oracle_rel_l2,pass\n";
    pairs << "rows,pair,order,repetitions,direct_us,candidate_us\n";
    require(target.max_context() >= 1040 && source.size() >= 1040,
            "reconstructed exact source/context extent");
    const auto offset_text=env("NINFER_TEST_RECON_EXACT_SOURCE_OFFSET");
    std::size_t source_offset=0;
    require(offset_text.size()<=6,"reconstructed exact source offset length");
    for(char c:offset_text) {
        require(c>='0' && c<='9',"reconstructed exact source offset must be decimal");
        source_offset=source_offset*10+static_cast<unsigned>(c-'0');
    }
    require(source_offset<=source.size()-1040,"reconstructed exact source offset extent");
    const auto* prompt_source=source.data()+source_offset;
    for (const char* key : {"NINFER_EXL3_WIDE_PREFILL",
                            "NINFER_EXL3_PREFILL_STAGED_REDUCTION",
                            "NINFER_EXL3_PREFILL_WIDE1024",
                            "NINFER_EXL3_PREFILL_DIRECT_PARTIALS",
                            "NINFER_EXL3_PREFILL_DIRECT_ASYNC_A"})
        require(env(key) == "1", std::string("reconstructed exact direct flag ") + key);
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV", "1");
    T69ProjectionCapture capture;
    const auto family=env("NINFER_TEST_RECON_EXACT_FAMILY");
    require(family.empty() || family=="k7" || family=="k6_down",
            "reconstructed exact fixture family");
    const auto ordinal_text=env("NINFER_TEST_RECON_EXACT_SITE");
    unsigned ordinal=0;
    require(ordinal_text.size()<=2,"reconstructed exact site ordinal length");
    for(char c:ordinal_text) {
        require(c>='0' && c<='9',"reconstructed exact site ordinal must be decimal");
        ordinal=ordinal*10+static_cast<unsigned>(c-'0');
    }
    require(ordinal<64,"reconstructed exact site ordinal bound");
    const auto async_all=env("NINFER_EXL3_PREFILL_DIRECT_ASYNC_ALL");
    if(family=="k6_down")
        require(async_all=="0" || async_all=="1","K6 fixture requires explicit async/direct selection");
    const auto changed_option=env("NINFER_TEST_RECON_CHANGED_WEIGHTS");
    require(changed_option.empty() || changed_option=="0" || changed_option=="1","changed-weight fixture option");
    struct CaptureSelection {
        T69ProjectionCapture* capture;int k;unsigned ordinal,seen=0;
        ninfer::exl3::Exl3CudaLinearWeights next_weights{};
        int next_layer=-1;
    };
    CaptureSelection selection{&capture,family=="k6_down"?6:7,ordinal};
    const auto fused_option=env("NINFER_TEST_FUSED_GATE_UP_OPERATOR");
    require(fused_option.empty() || fused_option=="1","fused gate/up operator option");
    const auto fused_wide=env("NINFER_TEST_FUSED_GATE_UP_WIDE");
    require(fused_wide.empty() || (fused_wide=="1" && fused_option=="1"),
        "fused wide fixture requires held fused operator mode");
    struct GateUpCapture {
        CaptureSelection* selection;
        std::vector<std::uint16_t> gate,up,activation;
    } gate_up_capture{&selection};
    auto context = target.create_context(true);
    const auto retained_metadata=target.metadata_owner();
    const auto context_metadata=context->metadata_owner();
    require(retained_metadata && retained_metadata.get()==context_metadata.get() &&
        !retained_metadata.owner_before(context_metadata) && !context_metadata.owner_before(retained_metadata),
        "context duplicated immutable metadata backing or lost strong model owner");
    if(fused_option=="1")context->set_layer_observer_for_test(
        [](const ninfer::exl3::Exl3LayerObservation& x,void* user) {
            auto& held=*static_cast<GateUpCapture*>(user);
            const auto& site=held.selection->capture->items[1];
            if(!site.input || x.layer!=site.layer || x.rows!=1024 || !held.gate.empty())return;
            require(x.gdn.gate_projection && x.gdn.up_projection && x.gdn.activated_mlp,
                "fused gate/up selected site is not a GDN layer");
            cuda_check(cudaStreamSynchronize(x.stream),"held gate/up producer completion");
            const std::size_t count=256ull*17408;
            auto copy=[&](std::vector<std::uint16_t>& dst,const std::uint16_t* src) {
                dst.resize(count);
                cuda_check(cudaMemcpy(dst.data(),src,count*2,cudaMemcpyDeviceToHost),"held gate/up intermediate");
            };
            copy(held.gate,x.gdn.gate_projection);copy(held.up,x.gdn.up_projection);
            copy(held.activation,x.gdn.activated_mlp);
        },&gate_up_capture);
    const auto model_identity=context->model_identity();
    context->prefill(std::span<const std::int64_t>(prompt_source, 16));
    // Do not inherit T69's const-char-pointer operation comparison.
    context->set_target_projection_observer_for_test(
        [](const ninfer::exl3::Exl3TargetProjectionObservation& x, void* user) {
            auto& selection=*static_cast<CaptureSelection*>(user);
            auto& site = selection.capture->items[1];
            if (x.rows != 1024 || x.metadata.mcg || !x.metadata.mul1 ||
                x.metadata.has_bias || x.metadata.K != selection.k ||
                x.metadata.in_features != 17408 || x.metadata.out_features != 5120 ||
                !x.operation || std::string(x.operation) != "down") return;
            const auto ordinal=selection.seen++;
            if(ordinal==selection.ordinal+1) {
                selection.next_weights=x.weights;selection.next_layer=x.layer;return;
            }
            if(ordinal!=selection.ordinal || site.input)return;
            site.input = std::make_unique<DeviceBuffer>(1024ull * 17408 * 2);
            cuda_check(cudaMemcpyAsync(site.input->get(), x.input, 1024ull * 17408 * 2,
                cudaMemcpyDeviceToDevice, x.stream), "reconstructed exact capture");
            cuda_check(cudaStreamSynchronize(x.stream), "reconstructed exact capture lifetime");
            site.layer = x.layer; site.operation = x.operation;
            site.weights = x.weights; site.metadata = x.metadata;
        }, &selection, nullptr,
        ninfer::exl3::Exl3TargetProjectionObserverSelection::wide_prefill_all);
    context->append_exact_prefill_wide(
        std::span<const std::int64_t>(prompt_source + 16, 1024));
    cuda_check(cudaDeviceSynchronize(), "reconstructed exact capture complete");
    context->set_target_projection_observer_for_test(nullptr);
    context.reset();
    auto& site = capture.items[1];
    require(site.input != nullptr, "reconstructed exact selected Down ordinal absent; no fallback site");
    std::ofstream identity(directory / "site.txt");
    identity << "family=" << (selection.k==6?"k6_down":"k7") << '\n'
             << "ordinal=" << ordinal << '\n' << "layer=" << site.layer << '\n'
             << "async_all=" << async_all << '\n'
             << "source_offset=" << source_offset << '\n';
    identity.flush();require(identity.good(),"reconstructed exact site identity write");
    std::ofstream prompt_identity(directory / "prompt_tokens.txt");
    for(std::size_t i=0;i<1040;++i)prompt_identity << prompt_source[i] << '\n';
    prompt_identity.flush();require(prompt_identity.good(),"reconstructed exact prompt identity write");
    constexpr int ni = 17408, no = 5120;
    constexpr std::size_t guard = 128, maximum = 1024ull * no;
    const int slice_columns=ninfer::exl3::exl3_reconstruction_slice_columns(
        std::getenv("NINFER_EXL3_RECONSTRUCTION_SLICE_COLUMNS"));
    const std::size_t decoded_bytes = static_cast<std::size_t>(ni) * slice_columns * 2;
    Exl3CudaLinearWorkspace direct(ni, no, 1024, false, false, false, false, false, true);
    const std::size_t retained_bytes = decoded_bytes + 4*guard + direct.workspace_bytes() +
        1024ull * ni * 2 + (maximum + 2 * guard) * 4;
    // Device diagnostic scratch; CPU copies below add under 120 MiB.
    const std::size_t fused_fixture_bytes=fused_option=="1"?12ull*256*ni*2+8*guard:0;
    require(retained_bytes + fused_fixture_bytes + 120ull * 1024 * 1024 < 600ull * 1024 * 1024,
            "reconstructed exact diagnostic scratch budget");
    if(fused_wide=="1")require(retained_bytes + (4ull*1024+3ull*256+1)*ni*2 +
        120ull*1024*1024 < 600ull*1024*1024,"fused wide diagnostic scratch budget; select a smaller reconstruction slice");
    struct DecodedBacking {
        DeviceBuffer allocation;
        ninfer::exl3::Exl3ReconstructionStream stream;
        explicit DecodedBacking(std::size_t bytes):allocation(bytes){}
    };
    auto decoded_owner=std::make_shared<DecodedBacking>(decoded_bytes+4*guard);
    if(selection.k==6)decoded_owner->stream.bind_model(model_identity,decoded_bytes,true,
        static_cast<std::uint16_t*>(decoded_owner->allocation.get())+guard);
    auto& decoded=decoded_owner->allocation;
    DeviceBuffer direct_out((maximum + 2 * guard) * 2),
        candidate_out((maximum + 2 * guard) * 2);
    auto* decoded_begin=static_cast<std::uint16_t*>(decoded.get());
    auto* decoded_data=decoded_begin+guard;
    std::vector<std::uint16_t> decoded_guard(guard,0x3555),decoded_guard_read(guard);
    cuda_check(cudaMemcpy(decoded_begin,decoded_guard.data(),guard*2,cudaMemcpyHostToDevice),
        "reconstructed exact leading decode guard");
    cuda_check(cudaMemcpy(decoded_data+decoded_bytes/2,decoded_guard.data(),guard*2,cudaMemcpyHostToDevice),
        "reconstructed exact trailing decode guard");
    std::vector<std::uint16_t> input_before(1024ull * ni);
    cuda_check(cudaMemcpy(input_before.data(), site.input->get(), input_before.size() * 2,
        cudaMemcpyDeviceToHost), "reconstructed exact input baseline");
    std::ofstream activation_identity(directory / "activation_f16.bin",std::ios::binary);
    activation_identity.write(reinterpret_cast<const char*>(input_before.data()),
        static_cast<std::streamsize>(input_before.size()*sizeof(std::uint16_t)));
    activation_identity.flush();require(activation_identity.good(),"reconstructed exact activation identity write");
    const auto* input = static_cast<const std::uint16_t*>(site.input->get());
    // Reservation selection must retain the real direct caller when no exact
    // implemented slab extent is installed. No cost observations are supplied.
    {
        constexpr int rows=256;
        const auto count=std::size_t(rows)*site.metadata.out_features;
        direct.set_reconstructed_exact({});
        const std::string baseline_route=direct.dispatch_name(site.metadata,rows,Admission::target_wide_prefill);
        direct.forward(site.weights,site.metadata,input,
            static_cast<std::uint16_t*>(direct_out.get())+guard,rows,nullptr,Admission::target_wide_prefill);
        std::vector<std::uint16_t> expected(count),actual(count);
        cuda_check(cudaMemcpy(expected.data(),static_cast<std::uint16_t*>(direct_out.get())+guard,
            count*2,cudaMemcpyDeviceToHost),"strategy direct reference");
        for(const auto bytes:{std::size_t(0),decoded_bytes-1,decoded_bytes+1}) {
            ninfer::exl3::Exl3ReconstructedExactView unavailable;
            unavailable.data=decoded_data;unavailable.bytes=bytes;
            unavailable.allow_k6_down=true;
            unavailable.backing_owner=decoded_owner;unavailable.model_owner=model_identity;
            direct.set_reconstructed_exact(unavailable);
            require(std::string(direct.dispatch_name(site.metadata,rows,Admission::target_wide_prefill))==baseline_route,
                "unsupported slab changed direct strategy");
            direct.forward(site.weights,site.metadata,input,
                static_cast<std::uint16_t*>(candidate_out.get())+guard,rows,nullptr,Admission::target_wide_prefill);
            cuda_check(cudaMemcpy(actual.data(),static_cast<std::uint16_t*>(candidate_out.get())+guard,
                count*2,cudaMemcpyDeviceToHost),"strategy fallback result");
            require(actual==expected,"reservation fallback changed direct output");
        }
        direct.set_reconstructed_exact({});
        ninfer::exl3::Exl3ReconstructedExactView restored;
        restored.data=decoded_data;restored.bytes=decoded_bytes;
        restored.allow_k6_down=true;restored.backing_owner=decoded_owner;
        restored.model_owner=model_identity;
        direct.set_reconstructed_exact(restored);
        require(std::string(direct.dispatch_name(site.metadata,rows,Admission::target_wide_prefill))==
            (selection.k==6?"target_wide_prefill_reconstructed_exact_k6_down":
                "target_wide_prefill_reconstructed_exact_k7"),
            "direct fallback poisoned surviving reconstruction reservation");
        direct.set_reconstructed_exact({});
    }
    auto invoke = [&](bool candidate, int rows) {
        if(candidate && selection.k==6) {
            ninfer::exl3::Exl3ReconstructedExactView view;
            view.data=decoded_data;view.bytes=decoded_bytes;
            view.allow_k6_down=true;
            view.ordered_stream=&decoded_owner->stream;
            view.backing_owner=decoded_owner;
            view.model_owner=model_identity;
            direct.set_reconstructed_exact(view);
            try {
                require(std::string(direct.dispatch_name(site.metadata,rows,Admission::target_wide_prefill))==
                    "target_wide_prefill_reconstructed_exact_k6_down","K6 Down actual reconstruction dispatch missing");
                direct.forward(site.weights,site.metadata,input,
                    static_cast<std::uint16_t*>(candidate_out.get())+guard,rows,nullptr,
                    Admission::target_wide_prefill);
            } catch(...) {direct.set_reconstructed_exact({});throw;}
            direct.set_reconstructed_exact({});
            return;
        }
        if (candidate) direct.forward_reconstructed_exact_for_test(site.weights, site.metadata,
            input, static_cast<std::uint16_t*>(candidate_out.get()) + guard, rows,
            decoded_data, decoded_bytes);
        else direct.forward(site.weights, site.metadata, input,
            static_cast<std::uint16_t*>(direct_out.get()) + guard, rows, nullptr,
            Admission::target_wide_prefill);
    };
    for (int rows : {256, 512, 1024}) {
        const std::size_t active = static_cast<std::size_t>(rows) * no;
        std::vector<std::uint16_t> a(maximum + 2 * guard, 0x3555), b(a);
        cuda_check(cudaMemcpy(direct_out.get(), a.data(), a.size() * 2, cudaMemcpyHostToDevice), "reconstructed exact poison direct");
        cuda_check(cudaMemcpy(candidate_out.get(), b.data(), b.size() * 2, cudaMemcpyHostToDevice), "reconstructed exact poison candidate");
        invoke(false, rows); invoke(true, rows);
        cuda_check(cudaMemcpy(a.data(), direct_out.get(), a.size() * 2, cudaMemcpyDeviceToHost), "reconstructed exact direct output");
        cuda_check(cudaMemcpy(b.data(), candidate_out.get(), b.size() * 2, cudaMemcpyDeviceToHost), "reconstructed exact candidate output");
        std::size_t mismatches = 0, first = active;
        bool finite = true, canaries = true;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (i < guard || i >= guard + active)
                canaries = canaries && a[i] == 0x3555 && b[i] == 0x3555;
            else {
                if (a[i] != b[i]) { ++mismatches; first = std::min(first, i - guard); }
                finite = finite && std::isfinite(half_to_float(a[i])) && std::isfinite(half_to_float(b[i]));
            }
        }
        invoke(false, rows); invoke(true, rows);
        std::vector<std::uint16_t> repeated(a.size());
        cuda_check(cudaMemcpy(repeated.data(), direct_out.get(), repeated.size() * 2, cudaMemcpyDeviceToHost), "reconstructed exact repeat direct");
        bool deterministic = repeated == a;
        cuda_check(cudaMemcpy(repeated.data(), candidate_out.get(), repeated.size() * 2, cudaMemcpyDeviceToHost), "reconstructed exact repeat candidate");
        deterministic = deterministic && repeated == b;
        repeated.resize(input_before.size());
        cuda_check(cudaMemcpy(repeated.data(), site.input->get(), repeated.size() * 2, cudaMemcpyDeviceToHost), "reconstructed exact input after");
        const bool unchanged = repeated == input_before;
        repeated.clear(); repeated.shrink_to_fit();
        T69OracleMetrics oracle{};
        bool oracle_pass = true;
        if (rows == 1024) {
            std::vector<std::uint16_t> aa(a.begin() + guard, a.begin() + guard + active),
                bb(b.begin() + guard, b.begin() + guard + active);
            oracle = t69_check_fp64_oracle(site, aa, bb, rows);
            oracle_pass = oracle.candidate.relative_l2 <= 0.003 && oracle.candidate.max_abs <= 0.10 &&
                oracle.candidate.relative_l2 <= oracle.direct.relative_l2 + 0.0005 &&
                oracle.candidate.max_abs <= oracle.direct.max_abs + 0.02;
        }
        for(const auto* edge:{decoded_begin,decoded_data+decoded_bytes/2}) {
            cuda_check(cudaMemcpy(decoded_guard_read.data(),edge,guard*2,cudaMemcpyDeviceToHost),
                "reconstructed exact decode guard readback");
            canaries=canaries && decoded_guard_read==decoded_guard;
        }
        const bool pass = mismatches == 0 && unchanged && canaries && deterministic && finite && oracle_pass;
        summary << rows << ',' << site.layer << ',' << site.operation << ','
            << direct.dispatch_name(site.metadata, rows, Admission::target_wide_prefill) << ','
            << decoded_bytes << ',' << retained_bytes << ',' << mismatches << ',' << first << ','
            << unchanged << ',' << canaries << ',' << deterministic << ',' << finite << ','
            << oracle.candidate.values << ',' << oracle.direct.max_abs << ',' << oracle.direct.relative_l2 << ','
            << oracle.candidate.max_abs << ',' << oracle.candidate.relative_l2 << ',' << pass << '\n';
        summary.flush();
        require(summary.good(), "reconstructed exact evidence write");
        require(pass, "reconstructed exact correctness gate failed; see summary.csv");
    }
    if(const char* alignment=std::getenv("NINFER_TEST_RECON_PACKED_ALIGNMENT")) {
        require(std::string(alignment)=="1","packed alignment fixture option");
        const std::size_t packed_words=std::size_t(ni/16)*(no/16)*(16*selection.k);
        const std::size_t copy_bytes=(packed_words+2*guard+3)*2;
        require(retained_bytes+copy_bytes+120ull*1024*1024<600ull*1024*1024,
            "packed alignment diagnostic budget; select a smaller reconstruction slice");
        DeviceBuffer copied(copy_bytes);
        auto* base=static_cast<std::uint16_t*>(copied.get());
        require((reinterpret_cast<std::uintptr_t>(site.weights.trellis)&7u)==0,
            "packed reference is not vector aligned");
        const auto original=site.weights;
        std::vector<std::uint16_t> expected(256ull*no),actual(expected.size()),edge(guard+3);
        invoke(true,256);
        cuda_check(cudaMemcpy(expected.data(),static_cast<std::uint16_t*>(candidate_out.get())+guard,
            expected.size()*2,cudaMemcpyDeviceToHost),"aligned packed reference");
        std::array<std::uint16_t,512> fragments{},fragment_result{};
        const auto read_fragments=[&](auto& destination) {
            cuda_check(cudaMemcpy(destination.data(),decoded_data,256*2,cudaMemcpyDeviceToHost),
                "first native fragment in final output slice");
            cuda_check(cudaMemcpy(destination.data()+256,decoded_data+decoded_bytes/2-256,256*2,
                cudaMemcpyDeviceToHost),"last native fragment in final output slice");
        };
        read_fragments(fragments);
        std::uint32_t multiplier=0;
        cuda_check(cudaMemcpy(&multiplier,original.mul1,sizeof(multiplier),cudaMemcpyDeviceToHost),
            "native fragment reference multiplier");
        std::vector<std::uint16_t> packed_tile(16*selection.k);
        for(int edge_index=0;edge_index<2;++edge_index) {
            const int tile_k=edge_index==0?0:ni/16-1;
            const int tile_n=edge_index==0?(no-slice_columns)/16:no/16-1;
            const auto packed_offset=(std::size_t(tile_k)*(no/16)+tile_n)*packed_tile.size();
            cuda_check(cudaMemcpy(packed_tile.data(),original.trellis+packed_offset,
                packed_tile.size()*2,cudaMemcpyDeviceToHost),"native fragment reference packed tile");
            for(int encoded=0;encoded<256;++encoded) {
                const auto state=t69_decode_state_bitwise(packed_tile.data(),selection.k,encoded);
                require(fragments[edge_index*256+encoded]==h6_decode_mul1(state,multiplier),
                    "native reconstruction fragment differs from independent bitwise reference");
            }
        }
        for(int offset:{0,1,2,3}) {
        auto* misaligned=base+guard+offset;
        require((reinterpret_cast<std::uintptr_t>(misaligned)&7u)==std::uintptr_t(offset*2),
            "packed copied-view alignment mismatch");
        for(int poison:{0x35,0x6a}) {
            cuda_check(cudaMemset(base,poison,copy_bytes),"packed alignment poison");
            cuda_check(cudaMemcpy(misaligned,original.trellis,packed_words*2,cudaMemcpyDeviceToDevice),
                "copy represented packed extent");
            site.weights.trellis=misaligned;
            try {invoke(true,256);}
            catch(...) {site.weights=original;throw;}
            site.weights=original;
            cuda_check(cudaMemcpy(actual.data(),static_cast<std::uint16_t*>(candidate_out.get())+guard,
                actual.size()*2,cudaMemcpyDeviceToHost),"misaligned packed result");
            require(actual==expected,"packed alignment or boundary poison changed reconstructed output");
            read_fragments(fragment_result);
            require(fragment_result==fragments,"packed alignment changed native decoded boundary fragments");
            auto observed_weights=original;observed_weights.trellis=misaligned;
            for(int tile_k:{0,ni/16-1})for(int tile_n:{0,no/16-1}) {
                cuda_check(cudaMemset(decoded_data,0x5a,288*2),"native fragment output guards");
                ninfer::exl3::exl3_reconstruct_native_tile_for_test(observed_weights,site.metadata,
                    tile_k,tile_n,decoded_data+16);
                std::array<std::uint16_t,288> guarded_fragment{};
                cuda_check(cudaMemcpy(guarded_fragment.data(),decoded_data,288*2,cudaMemcpyDeviceToHost),
                    "native fragment output guard readback");
                require(std::all_of(guarded_fragment.begin(),guarded_fragment.begin()+16,[](auto x){return x==0x5a5a;}) &&
                    std::all_of(guarded_fragment.end()-16,guarded_fragment.end(),[](auto x){return x==0x5a5a;}),
                    "native fragment exceeded output extent");
                cuda_check(cudaMemcpy(fragment_result.data(),decoded_data+16,256*2,cudaMemcpyDeviceToHost),
                    "bounded native corner fragment");
                const auto packed_offset=(std::size_t(tile_k)*(no/16)+tile_n)*packed_tile.size();
                cuda_check(cudaMemcpy(packed_tile.data(),original.trellis+packed_offset,
                    packed_tile.size()*2,cudaMemcpyDeviceToHost),"corner fragment source reference");
                for(int encoded=0;encoded<256;++encoded)
                    require(fragment_result[encoded]==h6_decode_mul1(
                        t69_decode_state_bitwise(packed_tile.data(),selection.k,encoded),multiplier),
                        "first/final native tile differs from independent reference");
            }
            for(int invalid=0;invalid<9;++invalid) {
                auto weights=observed_weights;auto metadata=site.metadata;
                int tile_k=0,tile_n=0;
                if(invalid==0)tile_k=-1;
                if(invalid==1)tile_k=ni/16;
                if(invalid==2)tile_n=-1;
                if(invalid==3)tile_n=no/16;
                if(invalid==4)weights.trellis=nullptr;
                if(invalid==5)weights.mul1=nullptr;
                if(invalid==6)metadata.K=5;
                if(invalid==7)metadata.has_bias=true;
                if(invalid==8)metadata.out_features-=16;
                cuda_check(cudaMemset(decoded_data,0x5a,288*2),"invalid fragment output sentinel");
                bool refused=false;
                try{ninfer::exl3::exl3_reconstruct_native_tile_for_test(weights,metadata,tile_k,tile_n,decoded_data+16);}
                catch(const std::invalid_argument&){refused=true;}
                require(refused,"invalid native fragment observation admitted");
                std::array<std::uint16_t,288> unchanged{};
                cuda_check(cudaMemcpy(unchanged.data(),decoded_data,288*2,cudaMemcpyDeviceToHost),
                    "invalid native fragment sentinel readback");
                require(std::all_of(unchanged.begin(),unchanged.end(),[](auto x){return x==0x5a5a;}),
                    "invalid native fragment observation modified output");
            }
            const auto sentinel=static_cast<std::uint16_t>(poison*257);
            cuda_check(cudaMemcpy(edge.data(),base,(guard+offset)*2,cudaMemcpyDeviceToHost),"packed leading guard");
            require(std::all_of(edge.begin(),edge.begin()+guard+offset,[&](auto x){return x==sentinel;}),
                "packed leading guard modified");
            cuda_check(cudaMemcpy(edge.data(),misaligned+packed_words,guard*2,cudaMemcpyDeviceToHost),"packed final guard");
            require(std::all_of(edge.begin(),edge.begin()+guard,[&](auto x){return x==sentinel;}),
                "packed final guard modified");
        }
        }
    }
    if(changed_option=="1") {
        require(selection.next_layer>=0 && selection.next_weights.trellis!=site.weights.trellis,
            "changed-weight fixture requires another real compatible Down site");
        const auto original=site.weights;
        std::vector<std::uint16_t> first,expected(256ull*no),actual(expected.size());
        for(int step=0;step<3;++step) {
            site.weights=step==1?selection.next_weights:original;
            invoke(false,256);invoke(true,256);
            cuda_check(cudaMemcpy(expected.data(),static_cast<std::uint16_t*>(direct_out.get())+guard,
                expected.size()*2,cudaMemcpyDeviceToHost),"changed-weight direct output");
            cuda_check(cudaMemcpy(actual.data(),static_cast<std::uint16_t*>(candidate_out.get())+guard,
                actual.size()*2,cudaMemcpyDeviceToHost),"changed-weight sliced output");
            require(actual==expected,"reconstruction retained stale decoded weights");
            require(std::all_of(actual.begin(),actual.end(),[](std::uint16_t value){return std::isfinite(half_to_float(value));}),
                "changed-weight output is nonfinite");
            if(step==0)first=actual;
            else require(step==1?actual!=first:actual==first,"changed-weight A-B-A coverage not exercised");
            for(const auto* edge:{decoded_begin,decoded_data+decoded_bytes/2}) {
                cuda_check(cudaMemcpy(decoded_guard_read.data(),edge,guard*2,cudaMemcpyDeviceToHost),
                    "changed-weight decoded guard readback");
                require(decoded_guard_read==decoded_guard,"changed-weight decode crossed slice allocation");
            }
        }
        site.weights=original;
        std::vector<std::uint16_t> input_after(input_before.size());
        cuda_check(cudaMemcpy(input_after.data(),site.input->get(),input_after.size()*2,cudaMemcpyDeviceToHost),
            "changed-weight input preservation readback");
        require(input_after==input_before,"changed-weight reconstruction modified held activation");
        identity << "changed_weight_layer=" << selection.next_layer << '\n';
        identity.flush();require(identity.good(),"changed-weight provenance write");
    }
    if(fused_option=="1") {
        require(!gate_up_capture.gate.empty(),"fused gate/up held intermediates absent");
        const std::size_t capacity=256ull*ni;
        DeviceBuffer gate_buffer(capacity*2),up_buffer(capacity*2),activation_buffer((capacity+2*guard)*2);
        cuda_check(cudaMemcpy(gate_buffer.get(),gate_up_capture.gate.data(),capacity*2,cudaMemcpyHostToDevice),"fused gate upload");
        cuda_check(cudaMemcpy(up_buffer.get(),gate_up_capture.up.data(),capacity*2,cudaMemcpyHostToDevice),"fused up upload");
        for(int rows:{1,7,8,15,16,255,256})for(int order:{0,1,2}) {
            const std::size_t count=std::size_t(rows)*ni;
            std::vector<std::uint16_t> gate_rows(count),up_rows(count),activation_rows(count);
            for(int row=0;row<rows;++row) {
                // Two disjoint captured row groups represent independent producers.
                // Swap or interleave them without changing any row's canonical bits.
                const int half=rows<=16?8:128;
                const int source_row=order==0?row:order==1?(row+half)%(2*half):(row/2+(row%2)*half);
                auto gather=[&](const auto& source,auto& destination) {
                    std::copy_n(source.begin()+std::size_t(source_row)*ni,ni,
                        destination.begin()+std::size_t(row)*ni);
                };
                gather(gate_up_capture.gate,gate_rows);gather(gate_up_capture.up,up_rows);
                gather(gate_up_capture.activation,activation_rows);
            }
            cuda_check(cudaMemcpy(gate_buffer.get(),gate_rows.data(),count*2,cudaMemcpyHostToDevice),"fused reordered gate upload");
            cuda_check(cudaMemcpy(up_buffer.get(),up_rows.data(),count*2,cudaMemcpyHostToDevice),"fused reordered up upload");
            auto* activation=static_cast<std::uint16_t*>(activation_buffer.get())+guard;
            cuda_check(cudaMemcpy(activation,activation_rows.data(),count*2,cudaMemcpyHostToDevice),"canonical activation upload");
            direct.transform_input(site.weights,site.metadata,activation,rows);
            std::vector<std::uint16_t> expected(count),got(count),guarded(count+2*guard);
            cuda_check(cudaMemcpy(expected.data(),direct.transformed_device(),count*2,cudaMemcpyDeviceToHost),"canonical activation transform");
            // Workspace has 1024 rows, so this sentinel is within allocated storage.
            cuda_check(cudaMemset(const_cast<std::uint16_t*>(direct.transformed_device()),0x5a,(count+guard)*2),
                "fused transformed destination sentinel");
            cuda_check(cudaMemset(activation_buffer.get(),0x5a,guarded.size()*2),"fused activation guards");
            direct.transform_gate_up(site.weights,site.metadata,
                static_cast<const std::uint16_t*>(gate_buffer.get()),static_cast<const std::uint16_t*>(up_buffer.get()),activation,rows);
            cuda_check(cudaMemcpy(got.data(),direct.transformed_device(),count*2,cudaMemcpyDeviceToHost),"fused activation transformed result");
            require(got==expected,"fused gate/up changed transformed bits");
            std::array<std::uint16_t,guard> transformed_tail{};
            cuda_check(cudaMemcpy(transformed_tail.data(),direct.transformed_device()+count,guard*2,cudaMemcpyDeviceToHost),
                "fused transformed tail readback");
            require(std::all_of(transformed_tail.begin(),transformed_tail.end(),[](std::uint16_t x){return x==0x5a5a;}),
                "fused gate/up wrote beyond transformed rows");
            cuda_check(cudaMemcpy(guarded.data(),activation_buffer.get(),guarded.size()*2,cudaMemcpyDeviceToHost),"fused activation represented result");
            require(std::equal(activation_rows.begin(),activation_rows.end(),guarded.begin()+guard),
                "fused gate/up changed canonical activation rounding");
            require(std::all_of(guarded.begin(),guarded.begin()+guard,[](std::uint16_t x){return x==0x5a5a;}) &&
                std::all_of(guarded.begin()+guard+count,guarded.end(),[](std::uint16_t x){return x==0x5a5a;}),"fused gate/up activation tail guard");
            cuda_check(cudaMemcpy(got.data(),gate_buffer.get(),count*2,cudaMemcpyDeviceToHost),"fused gate preservation");
            require(got==gate_rows,"fused transform modified gate producer");
            cuda_check(cudaMemcpy(got.data(),up_buffer.get(),count*2,cudaMemcpyDeviceToHost),"fused up preservation");
            require(got==up_rows,"fused transform modified up producer");
        }
        for(int invalid=0;invalid<13;++invalid) {
            auto metadata=site.metadata;
            auto weights=site.weights;
            auto* gate=static_cast<std::uint16_t*>(gate_buffer.get());
            auto* up=static_cast<std::uint16_t*>(up_buffer.get());
            auto* activation=static_cast<std::uint16_t*>(activation_buffer.get())+guard;
            int rows=1;
            if(invalid==0)metadata.mcg=true;
            if(invalid==1)metadata.mul1=false;
            if(invalid==2)metadata.has_bias=true;
            if(invalid==3)metadata.K=4;
            if(invalid==4)metadata.in_features-=128;
            if(invalid==5)rows=0;
            if(invalid==6)rows=1025;
            if(invalid==7)gate=nullptr;
            if(invalid==8)up=nullptr;
            if(invalid==9)activation=gate+1;
            if(invalid==10)activation=up+1;
            if(invalid==11)activation=const_cast<std::uint16_t*>(direct.transformed_device())+1;
            if(invalid==12)weights.suh=nullptr;
            const std::size_t checked=ni+2*guard;
            for(void* buffer:{gate_buffer.get(),up_buffer.get(),activation_buffer.get(),
                static_cast<void*>(const_cast<std::uint16_t*>(direct.transformed_device()))})
                cuda_check(cudaMemset(buffer,0x5a,checked*2),"fused refusal sentinel");
            bool refused=false;
            try{direct.transform_gate_up(weights,metadata,gate,up,activation,rows);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused,"fused gate/up invalid request reached numerical submission");
            std::vector<std::uint16_t> unchanged(checked);
            for(const void* buffer:{static_cast<const void*>(gate_buffer.get()),
                static_cast<const void*>(up_buffer.get()),static_cast<const void*>(activation_buffer.get()),
                static_cast<const void*>(direct.transformed_device())}) {
                cuda_check(cudaMemcpy(unchanged.data(),buffer,checked*2,cudaMemcpyDeviceToHost),"fused refusal preservation");
                require(std::all_of(unchanged.begin(),unchanged.end(),[](std::uint16_t x){return x==0x5a5a;}),
                    "fused refusal modified producer or output");
            }
        }
    }
    if(fused_wide=="1")for(int rows:{512,1024}) {
        // Repeat held canonical rows to cover launch geometry without claiming
        // naturally captured 512/1024-row producer provenance.
        const std::size_t count=std::size_t(rows)*ni;
        DeviceBuffer gates(count*2),ups(count*2),activations(count*2);
        for(int row=0;row<rows;++row) {
            const std::size_t source_row=(row*17)%256;
            auto upload=[&](void* dst,const auto& held) {
                cuda_check(cudaMemcpy(static_cast<std::uint16_t*>(dst)+std::size_t(row)*ni,
                    held.data()+source_row*ni,ni*2,cudaMemcpyHostToDevice),"assembled wide producer row");
            };
            upload(gates.get(),gate_up_capture.gate);upload(ups.get(),gate_up_capture.up);
            upload(activations.get(),gate_up_capture.activation);
        }
        direct.transform_input(site.weights,site.metadata,static_cast<const std::uint16_t*>(activations.get()),rows);
        std::vector<std::uint16_t> expected(count),row_bits(ni);
        cuda_check(cudaMemcpy(expected.data(),direct.transformed_device(),count*2,cudaMemcpyDeviceToHost),"wide canonical transformed bits");
        direct.transform_gate_up(site.weights,site.metadata,static_cast<const std::uint16_t*>(gates.get()),
            static_cast<const std::uint16_t*>(ups.get()),static_cast<std::uint16_t*>(activations.get()),rows);
        for(int row=0;row<rows;++row) {
            const std::size_t offset=std::size_t(row)*ni,source_offset=std::size_t((row*17)%256)*ni;
            cuda_check(cudaMemcpy(row_bits.data(),direct.transformed_device()+offset,ni*2,cudaMemcpyDeviceToHost),"wide fused transformed row");
            require(std::equal(row_bits.begin(),row_bits.end(),expected.begin()+offset),"wide fused transform differs");
            auto check=[&](const void* data,const auto& held) {
                cuda_check(cudaMemcpy(row_bits.data(),static_cast<const std::uint16_t*>(data)+offset,
                    ni*2,cudaMemcpyDeviceToHost),"wide fused represented row");
                require(std::equal(row_bits.begin(),row_bits.end(),held.begin()+source_offset),"wide fused producer or activation differs");
            };
            check(gates.get(),gate_up_capture.gate);check(ups.get(),gate_up_capture.up);
            check(activations.get(),gate_up_capture.activation);
        }
    }
    if(const auto reuse=env("NINFER_TEST_EQUAL_TRANSFORM_REUSE");!reuse.empty()) {
        require(reuse=="1","equal transform reuse fixture option");
        struct RestoreReuse {
            std::string value;
            ~RestoreReuse(){_putenv_s("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS",value.c_str());}
        } restore{env("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS")};
        _putenv_s("NINFER_EXL3_REUSE_EQUAL_INPUT_TRANSFORMS","1");
        DeviceBuffer second_scale(ni*2);
        std::vector<std::uint16_t> scales(ni);
        cuda_check(cudaMemcpy(scales.data(),site.weights.suh,ni*2,cudaMemcpyDeviceToHost),"equal-transform scale baseline");
        for(int variant=0;variant<3;++variant) {
            auto second=site.weights;
            if(variant) {
                if(variant==2)scales[0]^=0x8000;
                cuda_check(cudaMemcpy(second_scale.get(),scales.data(),ni*2,cudaMemcpyHostToDevice),"equal-transform held scales");
                second.suh=static_cast<const std::uint16_t*>(second_scale.get());
            }
            for(int transform_rows:{1,8,15,16}) {
            const std::size_t elements=std::size_t(transform_rows)*ni;
            DeviceBuffer fresh_input(elements*2);
            std::vector<std::uint16_t> activation(elements),original(elements);
            cuda_check(cudaMemcpy(original.data(),input,elements*2,cudaMemcpyDeviceToHost),"equal-transform held activation");
            std::vector<std::uint16_t> restored_first,restored_second;
            for(int generation=0;generation<3;++generation) {
            activation=original;
            if(generation==1)for(auto& value:activation)value^=0x8000;
            cuda_check(cudaMemcpy(fresh_input.get(),activation.data(),elements*2,cudaMemcpyHostToDevice),"equal-transform fresh activation");
            const auto* transform_input=static_cast<const std::uint16_t*>(fresh_input.get());
            std::vector<std::uint16_t> expected_first(elements),expected_second(elements),got(elements+2*guard);
            direct.transform_input(site.weights,site.metadata,transform_input,transform_rows);
            cuda_check(cudaMemcpy(expected_first.data(),direct.transformed_device(),elements*2,cudaMemcpyDeviceToHost),"equal-transform first reference");
            direct.transform_input(second,site.metadata,transform_input,transform_rows);
            cuda_check(cudaMemcpy(expected_second.data(),direct.transformed_device(),elements*2,cudaMemcpyDeviceToHost),"equal-transform second reference");
            cuda_check(cudaMemset(direct_out.get(),0x5a,got.size()*2),"equal-transform first guards");
            cuda_check(cudaMemset(candidate_out.get(),0x5a,got.size()*2),"equal-transform second guards");
            ninfer::exl3::exl3_transform_input_pair(site.weights,site.metadata,second,site.metadata,transform_input,
                static_cast<std::uint16_t*>(direct_out.get())+guard,static_cast<std::uint16_t*>(candidate_out.get())+guard,transform_rows);
            auto check_output=[&](const void* output,const std::vector<std::uint16_t>& expected) {
                cuda_check(cudaMemcpy(got.data(),output,got.size()*2,cudaMemcpyDeviceToHost),"equal-transform guarded result");
                require(std::all_of(got.begin(),got.begin()+guard,[](std::uint16_t value){return value==0x5a5a;}),"equal-transform leading guard overwritten");
                require(std::all_of(got.begin()+guard+elements,got.end(),[](std::uint16_t value){return value==0x5a5a;}),"equal-transform trailing guard overwritten");
                require(std::equal(expected.begin(),expected.end(),got.begin()+guard),"equal-transform represented result differs from independent transform");
            };
            check_output(direct_out.get(),expected_first);
            check_output(candidate_out.get(),expected_second);
            cuda_check(cudaMemcpy(activation.data(),fresh_input.get(),elements*2,cudaMemcpyDeviceToHost),"equal-transform input preservation");
            for(std::size_t i=0;i<elements;++i)
                require(activation[i]==std::uint16_t(original[i]^(generation==1?0x8000:0)),"equal-transform modified input");
            if(generation==0){restored_first=expected_first;restored_second=expected_second;}
            if(generation==1)require(expected_first!=restored_first || expected_second!=restored_second,"equal-transform freshness case did not distinguish changed activation");
            if(generation==2)require(expected_first==restored_first && expected_second==restored_second,"equal-transform restored activation changed reference");
            }
            }
        }
    }
    // New-family preparation is correctness-only; retain the historical K7
    // timing source without making it part of this new fixture's contract.
    if(selection.k==6) {
        // All comparisons above performed blocking readback. Exercise retention
        // only after those consumers have completed, not as a GPU retirement test.
        std::weak_ptr<DecodedBacking> weak=decoded_owner;
        ninfer::exl3::Exl3ReconstructedExactView retained;
        retained.data=decoded_data;retained.bytes=decoded_bytes;
        retained.ordered_stream=&decoded_owner->stream;retained.backing_owner=decoded_owner;
        retained.model_owner=model_identity;
        decoded_owner.reset();
        require(!weak.expired(),"reconstruction view failed to retain allocation and witness");
        retained.ordered_stream->require_ordered(0);
        retained.allow_k6_down=true;
        for(int mismatch=0;mismatch<4;++mismatch) {
            auto changed=retained;
            if(mismatch==0)changed.model_owner=std::make_shared<int>(1);
            if(mismatch==1)changed.model_owner.reset();
            if(mismatch==2)changed.bytes=decoded_bytes==17408ull*256*2?17408ull*512*2:17408ull*256*2;
            if(mismatch==3)changed.data+=16;
            direct.set_reconstructed_exact(changed);
            const std::size_t checked_elements=256ull*ni;
            cuda_check(cudaMemset(const_cast<std::uint16_t*>(direct.transformed_device()),0x5a,checked_elements*2),
                "reconstruction identity transform sentinel");
            for(bool transformed:{false,true}) {
                bool rejected=false;
                try {
                    if(transformed)direct.forward_from_transformed(site.weights,site.metadata,direct.transformed_device(),
                        static_cast<std::uint16_t*>(candidate_out.get())+guard,256,nullptr,Admission::target_wide_prefill);
                    else direct.forward(site.weights,site.metadata,input,
                        static_cast<std::uint16_t*>(candidate_out.get())+guard,256,nullptr,Admission::target_wide_prefill);
                }catch(const std::invalid_argument&){rejected=true;}
                require(rejected,"reconstruction view identity mismatch reached projection");
            }
            std::vector<std::uint16_t> scratch(checked_elements);
            cuda_check(cudaMemcpy(scratch.data(),direct.transformed_device(),checked_elements*2,cudaMemcpyDeviceToHost),
                "reconstruction identity transform readback");
            require(std::all_of(scratch.begin(),scratch.end(),[](std::uint16_t value){return value==0x5a5a;}),
                "reconstruction identity refusal modified transform scratch");
        }
        direct.set_reconstructed_exact(retained);
        const std::size_t transformed_elements=256ull*ni;
        cuda_check(cudaMemset(const_cast<std::uint16_t*>(direct.transformed_device()),0x5a,transformed_elements*2),
            "reconstruction failed-admission transform sentinel");
        retained.ordered_stream->fail();
        bool refused=false;
        try {
            direct.forward(site.weights,site.metadata,input,
                static_cast<std::uint16_t*>(candidate_out.get())+guard,256,nullptr,Admission::target_wide_prefill);
        } catch(const std::runtime_error&){refused=true;}
        require(refused,"failed reconstruction slab admitted another forward");
        cuda_check(cudaMemset(candidate_out.get(),0x5a,transformed_elements*2),"failed fused activation sentinel");
        bool fused_refused=false;
        try {
            direct.transform_gate_up(site.weights,site.metadata,input,input,
                static_cast<std::uint16_t*>(candidate_out.get()),256,nullptr,Admission::target_wide_prefill);
        }catch(const std::runtime_error&){fused_refused=true;}
        require(fused_refused,"failed reconstruction slab admitted fused producer");
        std::vector<std::uint16_t> activation_after(transformed_elements);
        cuda_check(cudaMemcpy(activation_after.data(),candidate_out.get(),transformed_elements*2,
            cudaMemcpyDeviceToHost),"failed fused activation readback");
        require(std::all_of(activation_after.begin(),activation_after.end(),[](std::uint16_t x){return x==0x5a5a;}),
            "failed reconstruction admission modified fused activation");
        std::vector<std::uint16_t> transformed_after(transformed_elements);
        cuda_check(cudaMemcpy(transformed_after.data(),direct.transformed_device(),transformed_elements*2,
            cudaMemcpyDeviceToHost),"reconstruction failed-admission transform readback");
        require(std::all_of(transformed_after.begin(),transformed_after.end(),
            [](std::uint16_t value){return value==0x5a5a;}),
            "failed reconstruction admission modified transform scratch");
        direct.set_reconstructed_exact({});
        retained={};
        require(weak.expired(),"retired reconstruction view retained backing indefinitely");
        return;
    }
    struct Events {
        cudaEvent_t begin = nullptr, end = nullptr;
        ~Events() { if (end) cudaEventDestroy(end); if (begin) cudaEventDestroy(begin); }
    } events;
    cuda_check(cudaEventCreate(&events.begin), "reconstructed exact event begin");
    cuda_check(cudaEventCreate(&events.end), "reconstructed exact event end");
    constexpr int repetitions = 3;
    auto measure = [&](bool candidate) {
        cuda_check(cudaEventRecord(events.begin), "reconstructed exact timing begin");
        for (int r = 0; r < repetitions; ++r) invoke(candidate, 1024);
        cuda_check(cudaEventRecord(events.end), "reconstructed exact timing end");
        cuda_check(cudaEventSynchronize(events.end), "reconstructed exact timing wait");
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, events.begin, events.end), "reconstructed exact timing resolve");
        return static_cast<double>(ms) * 1000 / repetitions;
    };
    for (int pair = 0; pair < 6; ++pair) {
        double a = 0, b = 0;
        if (pair % 2 == 0) { a = measure(false); b = measure(true); }
        else { b = measure(true); a = measure(false); }
        pairs << "1024," << pair << ',' << (pair % 2 == 0 ? "AB" : "BA") << ','
              << repetitions << ',' << a << ',' << b << '\n';
        pairs.flush();
        require(pairs.good(), "reconstructed exact timing evidence write");
    }
    std::cout << "RECONSTRUCTED_EXACT PASS rows=256,512,1024 pairs=6 scope=operator no_performance_gate=1\n";
}
