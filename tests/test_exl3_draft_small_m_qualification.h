// Included by test_exl3_dflash2_accept.cpp inside its anonymous namespace.
// Captures the represented inputs of one real cached B8 proposal, then checks
// the opt-in K5 M2..8 route against the unchanged same-workspace M1 topology.

struct DraftSmallCapturedProjection {
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    std::vector<std::uint16_t> input;
    int layer = -1;
    std::string name;
};

void capture_draft_small_projection(
    const Exl3Dflash2DraftModel::ProjectionObservation& observation, void* user) {
    auto& captured = *static_cast<std::vector<DraftSmallCapturedProjection>*>(user);
    require(observation.rows == 8, "draft-small capture requires a real B8 proposal");
    require(observation.layer >= 0 && observation.layer < 5,
            "draft-small capture layer outside five-layer model");
    require(observation.name != nullptr, "draft-small capture missing projection name");
    DraftSmallCapturedProjection item;
    item.weights = observation.weights;
    item.metadata = observation.metadata;
    item.layer = observation.layer;
    item.name = observation.name;
    item.input.resize(static_cast<std::size_t>(observation.rows) *
                      static_cast<std::size_t>(observation.metadata.in_features));
    cuda_check(cudaMemcpyAsync(item.input.data(), observation.input,
                               item.input.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, observation.stream),
               "draft-small capture input");
    cuda_check(cudaStreamSynchronize(observation.stream),
               "draft-small capture synchronize");
    captured.push_back(std::move(item));
}


struct DraftSmallGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~DraftSmallGraph() {
        if (executable != nullptr) cudaGraphExecDestroy(executable);
        if (graph != nullptr) cudaGraphDestroy(graph);
    }
};

// Two independently conditioned, captured B8 inputs, one explicit M16 operator.
// This checks represented activations only; Engine lifecycle supplies real
// request/ring/publication ownership coverage separately.
void check_draft_shared_projection_m16(const DraftSmallCapturedProjection& a,
    const DraftSmallCapturedProjection& b) {
    using Admission=ninfer::exl3::Exl3CudaLinearAdmission;
    const bool kv=a.name=="k" || a.name=="v";
    const bool is_output=a.name=="o";
    const bool is_down=a.name=="down";
    const bool is_gateup=a.name=="gate" || a.name=="up";
    const int input_columns=is_down?17408:is_output?4096:5120;
    const int output_columns=is_gateup?17408:(is_down || is_output)?5120:kv?1024:4096;
    const auto admission=is_gateup?Admission::draft_shared_gateup_m16:is_down?Admission::draft_shared_down_m16:is_output?Admission::draft_shared_o_m16:kv?Admission::draft_shared_kv_m16:Admission::draft_shared_q_m16;
    require(a.metadata.K==5 && b.metadata.K==5 && a.metadata.in_features==input_columns && b.metadata.in_features==input_columns &&
        a.metadata.out_features==output_columns && b.metadata.out_features==output_columns &&
        a.input.size()==8ULL*input_columns && b.input.size()==8ULL*input_columns,"held draft segment geometry");
    require((a.name=="q" || a.name=="k" || a.name=="v" || a.name=="o" || is_down || is_gateup) && b.name==a.name && a.layer==b.layer && a.metadata.K==5 &&
        a.weights.trellis==b.weights.trellis && a.weights.suh==b.weights.suh &&
        a.weights.svh==b.weights.svh && a.weights.mul1==b.weights.mul1,"draft M16 weight identity");
    const auto previous=env("NINFER_EXL3_DRAFT_SMALL_M");
    _putenv_s("NINFER_EXL3_DRAFT_SMALL_M","1");
    Exl3CudaLinearWorkspace control(input_columns,output_columns,8,!is_gateup),combined(input_columns,output_columns,16,true),
        single_row(input_columns,output_columns,1,!is_gateup);
    _putenv_s("NINFER_EXL3_DRAFT_SMALL_M",previous.c_str());
    if(!is_gateup) {
        require(control.draft_small_m_candidate_for_test(a.metadata,8),"independent draft small-M control not admitted");
        require(!single_row.draft_small_m_candidate_for_test(a.metadata,2),
            "draft small-M candidate exceeded workspace row allocation");
        auto wrong_input=a.metadata;--wrong_input.in_features;
        auto wrong_output=a.metadata;--wrong_output.out_features;
        require(!control.draft_small_m_candidate_for_test(wrong_input,8) &&
            !control.draft_small_m_candidate_for_test(wrong_output,8),
            "draft small-M candidate accepted metadata from a different workspace shape");
    }
    require(combined.draft_shared_m16_candidate(a.metadata,16,admission),
        "draft M16 operator NOT_ADMITTED");
    require(!combined.draft_shared_m16_candidate(a.metadata,8,admission),
        "draft M16 admitted single private segment");
    if(is_gateup) {
        require(std::string(control.dispatch_name(a.metadata,8))=="generic_tile" &&
            std::string(combined.dispatch_name(a.metadata,16,admission))=="draft_shared_gateup_m16_ordered",
            "draft gate/up comparison selected a different arithmetic route");
        for(int invalid_rows:{1,7,8,15,17})
            require(!combined.draft_shared_gateup_m16_candidate(a.metadata,invalid_rows,admission),
                "draft gate/up admitted incomplete private M16 segments");
    }
    // One poison row on either side catches an M16 route that treats the two
    // B8 segments as a padded/temporal wider proposal.
    DeviceBuffer input(18ULL*input_columns*2),output((16ULL*output_columns+256)*2),reference(16ULL*output_columns*2);
    auto* input_base=static_cast<std::uint16_t*>(input.get());
    auto* in=input_base+input_columns;
    auto* out=static_cast<std::uint16_t*>(output.get());
    auto* ref=static_cast<std::uint16_t*>(reference.get());
    std::vector<std::uint16_t> independent_transforms(is_gateup?16ULL*input_columns:0);
    for(bool reverse:{false,true}) {
        const auto& first=reverse?b:a;const auto& second=reverse?a:b;
        cuda_check(cudaMemset(input_base,0x7e,18ULL*input_columns*2),"M16 poison adjacent rows");
        cuda_check(cudaMemcpy(in,first.input.data(),8ULL*input_columns*2,cudaMemcpyHostToDevice),"M16 first private segment");
        cuda_check(cudaMemcpy(in+8*input_columns,second.input.data(),8ULL*input_columns*2,cudaMemcpyHostToDevice),"M16 second private segment");
        cuda_check(cudaMemset(out,0x55,(16ULL*output_columns+256)*2),"M16 destination guards");
        control.forward(a.weights,a.metadata,in,ref,8);
        if(is_gateup)cuda_check(cudaMemcpy(independent_transforms.data(),control.transformed_device(),
            8ULL*input_columns*2,cudaMemcpyDeviceToHost),"draft gate/up first private transform");
        control.forward(a.weights,a.metadata,in+8*input_columns,ref+8*output_columns,8);
        if(is_gateup)cuda_check(cudaMemcpy(independent_transforms.data()+8*input_columns,control.transformed_device(),
            8ULL*input_columns*2,cudaMemcpyDeviceToHost),"draft gate/up second private transform");
        combined.forward(a.weights,a.metadata,in,out+128,16,nullptr,admission);
        cuda_check(cudaDeviceSynchronize(),"M16 represented comparison completion");
        if(is_gateup) {
            std::vector<std::uint16_t> transformed(independent_transforms.size());
            cuda_check(cudaMemcpy(transformed.data(),combined.transformed_device(),transformed.size()*2,
                cudaMemcpyDeviceToHost),"draft gate/up combined transform");
            require(transformed==independent_transforms,"draft gate/up M16 changed canonical input transform");
        }
        std::vector<std::uint16_t> expected(16ULL*output_columns),actual(16ULL*output_columns+256);
        cuda_check(cudaMemcpy(expected.data(),ref,expected.size()*2,cudaMemcpyDeviceToHost),"M16 control output");
        cuda_check(cudaMemcpy(actual.data(),out,actual.size()*2,cudaMemcpyDeviceToHost),"M16 actual output");
        require(std::equal(expected.begin(),expected.end(),actual.begin()+128),"M16 changed private draft activations");
        require(std::all_of(actual.begin(),actual.begin()+128,[](auto value){return value==0x5555;}) &&
            std::all_of(actual.end()-128,actual.end(),[](auto value){return value==0x5555;}),"M16 wrote outside destination");
        std::vector<std::uint16_t> input_guards(2ULL*input_columns);
        cuda_check(cudaMemcpy(input_guards.data(),input_base,input_columns*2,cudaMemcpyDeviceToHost),
            "M16 leading poison row");
        cuda_check(cudaMemcpy(input_guards.data()+input_columns,in+16ULL*input_columns,
            input_columns*2,cudaMemcpyDeviceToHost),"M16 trailing poison row");
        require(std::all_of(input_guards.begin(),input_guards.end(),[](auto value){return value==0x7e7e;}),
            "draft M16 modified adjacent private input rows");
        {
            auto* transformed=const_cast<std::uint16_t*>(combined.transformed_device());
            cuda_check(cudaMemset(transformed,0x6b,16ULL*input_columns*2),"null output transform sentinel");
            bool refused=false;
            try{combined.forward(a.weights,a.metadata,in,nullptr,16,nullptr,admission);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused,"draft M16 accepted missing destination");
            cuda_check(cudaDeviceSynchronize(),"null output refusal observation");
            std::vector<std::uint16_t> unchanged(16ULL*input_columns);
            cuda_check(cudaMemcpy(unchanged.data(),transformed,unchanged.size()*2,cudaMemcpyDeviceToHost),
                "null output transform readback");
            require(std::all_of(unchanged.begin(),unchanged.end(),[](auto value){return value==0x6b6b;}),
                "missing destination changed transform scratch before refusal");
        }
        const auto require_untouched_refusal=[&](const Exl3CudaLinearMetadata& metadata,int rows,Admission requested) {
            cuda_check(cudaMemset(out,0x33,(16ULL*output_columns+256)*2),"M16 refusal destination sentinel");
            auto* transformed=const_cast<std::uint16_t*>(combined.transformed_device());
            cuda_check(cudaMemset(transformed,0x6b,16ULL*input_columns*2),"M16 refusal transform sentinel");
            bool refused=false;
            try{combined.forward(a.weights,metadata,in,out+128,rows,nullptr,requested);}
            catch(const std::invalid_argument&){refused=true;}
            require(refused,"explicit draft M16 invalid admission silently selected another route");
            cuda_check(cudaDeviceSynchronize(),"M16 refusal completion observation");
            std::vector<std::uint16_t> untouched(16ULL*output_columns+256);
            cuda_check(cudaMemcpy(untouched.data(),out,untouched.size()*2,cudaMemcpyDeviceToHost),
                "M16 refusal destination readback");
            require(std::all_of(untouched.begin(),untouched.end(),[](auto value){return value==0x3333;}),
                "rejected draft M16 admission modified destination before throwing");
            std::vector<std::uint16_t> untouched_transform(16ULL*input_columns);
            cuda_check(cudaMemcpy(untouched_transform.data(),transformed,untouched_transform.size()*2,
                cudaMemcpyDeviceToHost),"M16 refusal transform readback");
            require(std::all_of(untouched_transform.begin(),untouched_transform.end(),[](auto value){return value==0x6b6b;}),
                "rejected draft M16 admission modified transform before throwing");
        };
        for(int invalid_rows:{0,1,7,8,15,17})require_untouched_refusal(a.metadata,invalid_rows,admission);
        require_untouched_refusal(a.metadata,16,kv?Admission::draft_shared_q_m16:Admission::draft_shared_kv_m16);
        for(int mutation=0;mutation<6;++mutation) {
            auto invalid=a.metadata;
            switch(mutation) {
                case 0: --invalid.in_features;break;
                case 1: --invalid.out_features;break;
                case 2: invalid.K=6;break;
                case 3: invalid.mcg=true;break;
                case 4: invalid.mul1=false;break;
                case 5: invalid.has_bias=true;break;
            }
            require_untouched_refusal(invalid,16,admission);
        }
        // Reuse this exact workspace after all refused submissions. Recreating
        // it would conceal a refusal that damaged persistent dispatch state.
        cuda_check(cudaMemset(out,0x55,(16ULL*output_columns+256)*2),"M16 recovery destination guards");
        combined.forward(a.weights,a.metadata,in,out+128,16,nullptr,admission);
        cuda_check(cudaDeviceSynchronize(),"M16 recovery completion");
        cuda_check(cudaMemcpy(actual.data(),out,actual.size()*2,cudaMemcpyDeviceToHost),
            "M16 recovery output");
        require(std::equal(expected.begin(),expected.end(),actual.begin()+128),
            "draft M16 refused submission changed subsequent valid private activations");
        require(std::all_of(actual.begin(),actual.begin()+128,[](auto value){return value==0x5555;}) &&
            std::all_of(actual.end()-128,actual.end(),[](auto value){return value==0x5555;}),
            "draft M16 recovery wrote outside destination");
        if(is_gateup) {
            std::vector<std::uint16_t> recovered_transform(independent_transforms.size());
            cuda_check(cudaMemcpy(recovered_transform.data(),combined.transformed_device(),recovered_transform.size()*2,
                cudaMemcpyDeviceToHost),"draft gate/up recovered transform");
            require(recovered_transform==independent_transforms,
                "draft gate/up refusal changed subsequent canonical input transform");
        }
    }
    // Adjacent binary16 values around ordinary rounding boundaries are a
    // represented-arithmetic check, not a token plausibility test. They must
    // preserve the same two independent M8 results for every draft family.
    {
        constexpr std::array<std::uint16_t,12> boundary_bits{
            0x0001,0x03ff,0x0400,0x3554,0x3555,0x3556,
            0x3bff,0x3c00,0x3c01,0x8001,0xbbff,0xbc01};
        std::vector<std::uint16_t> boundary(16ULL*input_columns);
        for(std::size_t i=0;i<boundary.size();++i)
            boundary[i]=boundary_bits[(i+(i/input_columns)*5)%boundary_bits.size()];
        cuda_check(cudaMemset(input_base,0x7e,18ULL*input_columns*2),"M16 boundary poison rows");
        cuda_check(cudaMemcpy(in,boundary.data(),boundary.size()*2,cudaMemcpyHostToDevice),
            "M16 boundary input");
        control.forward(a.weights,a.metadata,in,ref,8);
        if(is_gateup)cuda_check(cudaMemcpy(independent_transforms.data(),control.transformed_device(),
            8ULL*input_columns*2,cudaMemcpyDeviceToHost),"draft boundary first transform");
        control.forward(a.weights,a.metadata,in+8ULL*input_columns,ref+8ULL*output_columns,8);
        if(is_gateup)cuda_check(cudaMemcpy(independent_transforms.data()+8ULL*input_columns,
            control.transformed_device(),8ULL*input_columns*2,cudaMemcpyDeviceToHost),
            "draft boundary second transform");
        cuda_check(cudaMemset(out,0x55,(16ULL*output_columns+256)*2),"M16 boundary output guards");
        combined.forward(a.weights,a.metadata,in,out+128,16,nullptr,admission);
        cuda_check(cudaDeviceSynchronize(),"M16 boundary represented completion");
        std::vector<std::uint16_t> expected(16ULL*output_columns),actual(16ULL*output_columns+256);
        cuda_check(cudaMemcpy(expected.data(),ref,expected.size()*2,cudaMemcpyDeviceToHost),
            "M16 boundary independent output");
        cuda_check(cudaMemcpy(actual.data(),out,actual.size()*2,cudaMemcpyDeviceToHost),
            "M16 boundary combined output");
        require(std::equal(expected.begin(),expected.end(),actual.begin()+128),
            "draft M16 boundary inputs changed represented independent results");
        require(std::all_of(actual.begin(),actual.begin()+128,[](auto value){return value==0x5555;}) &&
            std::all_of(actual.end()-128,actual.end(),[](auto value){return value==0x5555;}),
            "draft M16 boundary case wrote outside destination");
        if(is_gateup) {
            std::vector<std::uint16_t> transformed(independent_transforms.size());
            cuda_check(cudaMemcpy(transformed.data(),combined.transformed_device(),transformed.size()*2,
                cudaMemcpyDeviceToHost),"draft boundary combined transform");
            require(transformed==independent_transforms,
                "draft gate/up boundary case changed canonical input transform");
        }
    }
}

void run_draft_small_m_qualification(Exl3TextModel& target,
                                     Exl3Dflash2DraftModel& draft,
                                     const std::vector<std::int64_t>& prompt,
                                     bool oscar,
                                     std::ostream& out) {
    require(prompt.size() == 512, "draft-small qualification requires real ctx512");
    require(oscar, "draft-small qualification requires canonical OSCAR routing");
    require(env("NINFER_EXL3_DRAFT_SMALL_M") == "0",
            "draft-small real capture must start from explicit flag0 baseline");
    ninfer::test::ScopedEnvironmentRestore restore_environment{
        "NINFER_EXL3_DRAFT_SMALL_M",
        "NINFER_EXL3_GENERIC_SPLITS"};
    _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");
    int device = 0;
    cudaDeviceProp device_properties{};
    cuda_check(cudaGetDevice(&device), "draft-small current device");
    cuda_check(cudaGetDeviceProperties(&device_properties, device),
               "draft-small device properties");
    std::cout << "E5A5K5_DEVICE_CONFIG device=" << device
              << " sm_count=" << device_properties.multiProcessorCount
              << " normal_split_policy=min(5,cooperative_capacity/output_blocks)\n";

    TapStage stage;
    for (int t = 0; t < kTapCount; ++t) {
        stage.bulk.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(16) * kHidden * sizeof(std::uint16_t)));
        stage.rows.push_back(std::make_unique<DeviceBuffer>(
            static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t)));
        stage.bulk_ptrs.push_back(static_cast<std::uint16_t*>(stage.bulk.back()->get()));
        stage.row_ptrs.push_back(static_cast<std::uint16_t*>(stage.rows.back()->get()));
    }
    auto context = target.create_context(true);
    if (oscar)
        require(context->try_enable_oscar_from_environment(),
                "draft-small OSCAR enable failed");
    draft.reset();
    ingest_prefix(*context, prompt, CommitSink{&draft, &stage});
    require(draft.ring_count() == 512 && draft.ring_base_abs() == 0,
            "draft-small ctx512 ring construction mismatch");

    std::vector<DraftSmallCapturedProjection> captured;
    const bool shared_gateup_fixture=env("NINFER_TEST_DRAFT_SHARED_GATEUP_M16")=="1";
    const auto capture_count=shared_gateup_fixture?35U:25U;
    captured.reserve(capture_count);
    draft.set_projection_observer_for_test(capture_draft_small_projection, &captured,shared_gateup_fixture);
    struct ObserverReset {
        Exl3Dflash2DraftModel* model;
        ~ObserverReset() { if (model != nullptr) model->set_projection_observer_for_test(nullptr); }
    } observer_reset{&draft};
    std::vector<std::int64_t> block(8, kMaskToken);
    block[0] = sample_target(*context);  // emitted seed remains pending at ring tail
    const auto baseline_proposals = draft.propose_cached(
        block, 512, context->target_embedding(), context->target_lm_head_weights(),
        context->target_lm_head_metadata(), kMaskToken);
    draft.set_projection_observer_for_test(nullptr);
    observer_reset.model = nullptr;
    require(baseline_proposals.size() == 7,
            "draft-small baseline proposal count mismatch");
    require(captured.size() == capture_count,
            "draft-small observer did not capture selected K5 projections");
    const bool shared_kv_fixture=env("NINFER_TEST_DRAFT_SHARED_KV_M16")=="1";
    const bool shared_o_fixture=env("NINFER_TEST_DRAFT_SHARED_O_M16")=="1";
    const bool shared_down_fixture=env("NINFER_TEST_DRAFT_SHARED_DOWN_M16")=="1";
    if(env("NINFER_TEST_DRAFT_SHARED_Q_M16")=="1" || shared_kv_fixture || shared_o_fixture || shared_down_fixture || shared_gateup_fixture) {
        auto sibling=draft.create_execution();sibling->restore_host_ring(draft.export_host_ring());
        std::vector<DraftSmallCapturedProjection> other;other.reserve(capture_count);
        sibling->set_projection_observer_for_test(capture_draft_small_projection,&other,shared_gateup_fixture);
        ObserverReset sibling_observer{sibling.get()};
        auto private_block=block;private_block[0]=(block[0]+1)%248070;
        const auto proposals=sibling->propose_cached(private_block,512,context->target_embedding(),
            context->target_lm_head_weights(),context->target_lm_head_metadata(),kMaskToken);
        sibling->set_projection_observer_for_test(nullptr);sibling_observer.model=nullptr;
        require(proposals.size()==7 && other.size()==capture_count,"second private B8 capture");
        require(sibling->ring_count()==draft.ring_count() && sibling->ring_base_abs()==draft.ring_base_abs(),
            "proposal mutated committed ring frontier");
        unsigned compared=0;
        for(std::size_t i=0;i<captured.size();++i)
            if(captured[i].name=="q" || (shared_kv_fixture && (captured[i].name=="k" || captured[i].name=="v")) ||
                (shared_o_fixture && captured[i].name=="o") || (shared_down_fixture && captured[i].name=="down") ||
                (shared_gateup_fixture && (captured[i].name=="gate" || captured[i].name=="up"))) {
                check_draft_shared_projection_m16(captured[i],other[i]);++compared;
            }
        require(compared==5U+(shared_kv_fixture?10U:0U)+(shared_o_fixture?5U:0U)+(shared_down_fixture?5U:0U)+(shared_gateup_fixture?10U:0U),"held draft family coverage NOT_EXERCISED");
    }
    // The legacy small-M suite below has a five-family denominator. Gate/up
    // ordered-M16 coverage above is separate from its MMA-versus-M1 cases.
    std::erase_if(captured,[](const auto& item){return item.name=="gate" || item.name=="up";});

    std::array<int, 5> per_layer{};
    std::array<std::array<bool, 5>, 5> seen{};
    auto name_index = [](const std::string& name) {
        if (name == "q") return 0;
        if (name == "k") return 1;
        if (name == "v") return 2;
        if (name == "o") return 3;
        if (name == "down") return 4;
        return -1;
    };
    for (const auto& item : captured) {
        require(item.metadata.K == 5 && item.metadata.mul1 && !item.metadata.mcg &&
                    !item.metadata.has_bias,
                "draft-small captured unsupported metadata");
        const int projection = name_index(item.name);
        require(projection >= 0,
                "draft-small observer captured an unintended projection");
        require(!seen[static_cast<std::size_t>(item.layer)][static_cast<std::size_t>(projection)],
                "draft-small observer captured a duplicate layer/projection pair");
        seen[static_cast<std::size_t>(item.layer)][static_cast<std::size_t>(projection)] = true;
        ++per_layer[static_cast<std::size_t>(item.layer)];
    }
    require(std::all_of(per_layer.begin(), per_layer.end(), [](int n) { return n == 5; }),
            "draft-small observer did not capture five projections per layer");

    const std::filesystem::path input_directory =
        std::filesystem::path(env("NINFER_E5A4_OUT")).parent_path() / "inputs";
    std::filesystem::create_directories(input_directory);
    for (const auto& item : captured) {
        const auto path = input_directory /
            ("layer" + std::to_string(item.layer) + "-" + item.name + "-" +
             std::to_string(item.metadata.in_features) + "x" +
             std::to_string(item.metadata.out_features) + ".f16");
        require(!std::filesystem::exists(path),
                "draft-small refuses to overwrite captured input: " + path.string());
        std::ofstream input_file(path, std::ios::binary);
        require(input_file.good(), "cannot create draft-small input capture: " + path.string());
        input_file.write(reinterpret_cast<const char*>(item.input.data()),
                         static_cast<std::streamsize>(item.input.size() * sizeof(std::uint16_t)));
        require(input_file.good(), "cannot write draft-small input capture: " + path.string());
    }

    constexpr std::size_t guard = 128;
    constexpr std::uint16_t sentinel = 0x3555;
    out << "source,layer,name,M,in_features,out_features,K,dispatch,repeat_equal,guards_ok,"
           "finite,bit_mismatches,generic_max_abs,generic_rel_l2,oracle_values,oracle_max_abs,"
           "oracle_rel_l2,oracle_max_error_over_bound,oracle_max_bound,oracle_fp64_bound,"
           "oracle_worst_group,oracle_worst_row,oracle_worst_column,oracle_within_bound,"
           "input_distinct_rows\n";
    for (std::size_t index = 0; index < captured.size(); ++index) {
        const auto& item = captured[index];
        const int in_features = item.metadata.in_features;
        const int out_features = item.metadata.out_features;
        int input_distinct_rows = 0;
        for (int row = 0; row < 8; ++row) {
            bool first = true;
            for (int prior = 0; prior < row; ++prior) {
                if (std::equal(item.input.begin() + static_cast<std::size_t>(row) * in_features,
                               item.input.begin() + static_cast<std::size_t>(row + 1) * in_features,
                               item.input.begin() + static_cast<std::size_t>(prior) * in_features)) {
                    first = false;
                    break;
                }
            }
            input_distinct_rows += first;
        }
        DeviceBuffer input(item.input.size() * sizeof(std::uint16_t));
        const std::size_t output_elements = 8u * static_cast<std::size_t>(out_features) + 2u * guard;
        DeviceBuffer output(output_elements * sizeof(std::uint16_t));
        DeviceBuffer reference(static_cast<std::size_t>(out_features) * sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* output_base = static_cast<std::uint16_t*>(output.get());
        auto* output_device = output_base + guard;
        auto* reference_device = static_cast<std::uint16_t*>(reference.get());
        cuda_check(cudaMemcpy(input_device, item.input.data(), item.input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "draft-small input upload");

        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "1");
        Exl3CudaLinearWorkspace candidate(in_features, out_features, 8);
        require(std::string(candidate.dispatch_name(item.metadata, 1)) == "generic_mma_split",
                "draft-small candidate changed M1 dispatch");
        const int output_blocks = out_features / 512;
        const bool s5_proven_by_occupancy_lower_bound =
            std::string(candidate.dispatch_name(item.metadata, 8)) ==
                "draft_small_m_mma_split" &&
            device_properties.multiProcessorCount >= 5 * output_blocks;
        std::cout << "E5A5K5_WORKSPACE_CONFIG layer=" << item.layer
                  << " name=" << item.name
                  << " in=" << in_features << " out=" << out_features
                  << " workspace_bytes=" << candidate.workspace_bytes()
                  << " output_blocks=" << output_blocks
                  << " normal_splits=5"
                  << " s5_proven_by_occupancy_lower_bound="
                  << s5_proven_by_occupancy_lower_bound << '\n';
        require(s5_proven_by_occupancy_lower_bound,
                "draft-small host cannot prove the declared normal S5 launch policy");
        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "0");
        Exl3CudaLinearWorkspace generic(in_features, out_features, 8);
        require(std::string(generic.dispatch_name(item.metadata, 8)) == "generic_tile",
                "draft-small explicit flag0 did not select generic M8");
        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "1");
        require(std::string(generic.dispatch_name(item.metadata, 8)) == "generic_tile",
                "draft-small flag0 workspace did not latch construction state");

        std::vector<std::uint16_t> initial(output_elements, sentinel);
        std::vector<std::uint16_t> actual(output_elements), repeated(output_elements);
        std::vector<std::uint16_t> row_reference(static_cast<std::size_t>(out_features));
        for (int m = 2; m <= 8; ++m) {
            require(std::string(candidate.dispatch_name(item.metadata, m)) ==
                        "draft_small_m_mma_split",
                    "draft-small candidate did not dispatch M2..8");
            cuda_check(cudaMemcpy(output_base, initial.data(), output_elements * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice), "draft-small canary upload");
            candidate.forward(item.weights, item.metadata, input_device, output_device, m);
            cuda_check(cudaMemcpy(actual.data(), output_base,
                                  output_elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "draft-small candidate download");
            const bool guards_ok =
                std::all_of(actual.begin(), actual.begin() + guard,
                            [](std::uint16_t value) { return value == sentinel; }) &&
                std::all_of(actual.begin() + guard + static_cast<std::size_t>(m) * out_features,
                            actual.end(), [](std::uint16_t value) { return value == sentinel; });
            bool repeat_equal = true;
            for (int repeat = 0; repeat < 2; ++repeat) {
                candidate.forward(item.weights, item.metadata, input_device, output_device, m);
                cuda_check(cudaMemcpy(repeated.data(), output_base,
                                      output_elements * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost), "draft-small repeat download");
                repeat_equal = repeat_equal && repeated == actual;
            }
            std::size_t bit_mismatches = 0;
            bool finite = true;
            for (int row = 0; row < m; ++row) {
                candidate.forward(item.weights, item.metadata,
                                  input_device + static_cast<std::size_t>(row) * in_features,
                                  reference_device, 1);
                cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                      row_reference.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost), "draft-small M1 row download");
                const std::size_t offset = guard + static_cast<std::size_t>(row) * out_features;
                for (int column = 0; column < out_features; ++column) {
                    bit_mismatches += actual[offset + static_cast<std::size_t>(column)] !=
                                      row_reference[static_cast<std::size_t>(column)];
                    finite = finite && std::isfinite(half_to_float(
                        actual[offset + static_cast<std::size_t>(column)]));
                }
            }
            K5OracleStats oracle;
            if (m == 8) {
                const std::vector<int> groups{
                    0, (out_features / 128) / 2, out_features / 128 - 1};
                oracle = k5_check_oracle_groups(item.weights, item.metadata, item.input,
                                                actual, guard, groups);
                require(oracle.within_bound,
                        "draft-small real M8 exceeded independent packed-code oracle bound");
            }
            out << "real," << item.layer << ',' << item.name << ',' << m << ','
                << in_features << ',' << out_features << ',' << item.metadata.K << ','
                << candidate.dispatch_name(item.metadata, m) << ',' << repeat_equal << ','
                << guards_ok << ',' << finite << ',' << bit_mismatches << ",0,0,"
                << oracle.values << ',' << std::setprecision(12) << oracle.max_abs << ','
                << (oracle.norm_sq > 0.0 ? std::sqrt(oracle.error_sq / oracle.norm_sq)
                                         : std::sqrt(oracle.error_sq)) << ','
                << oracle.max_ratio << ',' << oracle.max_bound << ',' << oracle.max_fp64_bound << ','
                << oracle.worst_group << ',' << oracle.worst_row << ',' << oracle.worst_column << ',';
            if (oracle.values == 0) out << "NA";
            else out << oracle.within_bound;
            out << ',' << input_distinct_rows << '\n';
            require(repeat_equal && guards_ok && finite && bit_mismatches == 0,
                    "draft-small real-input exact M1 qualification failed");
        }

        // The retained scalar route has a different reduction. It is a finite
        // diagnostic comparison, not a bit-exact acceptance oracle.
        cuda_check(cudaMemcpy(output_base, initial.data(), output_elements * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "draft-small generic canary upload");
        generic.forward(item.weights, item.metadata, input_device, output_device, 8);
        cuda_check(cudaMemcpy(repeated.data(), output_base,
                              output_elements * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                   "draft-small generic download");
        const bool generic_guards =
            std::all_of(repeated.begin(), repeated.begin() + guard,
                        [](std::uint16_t value) { return value == sentinel; }) &&
            std::all_of(repeated.begin() + guard + 8u * static_cast<std::size_t>(out_features),
                        repeated.end(), [](std::uint16_t value) { return value == sentinel; });
        generic.forward(item.weights, item.metadata, input_device, output_device, 8);
        std::vector<std::uint16_t> generic_repeated(output_elements);
        cuda_check(cudaMemcpy(generic_repeated.data(), output_base,
                              output_elements * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                   "draft-small generic repeat download");
        const bool generic_repeat = generic_repeated == repeated;
        cuda_check(cudaMemcpy(output_base, initial.data(), output_elements * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "draft-small candidate compare canary upload");
        candidate.forward(item.weights, item.metadata, input_device, output_device, 8);
        cuda_check(cudaMemcpy(actual.data(), output_base,
                              output_elements * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                   "draft-small candidate compare download");
        double max_abs = 0.0, error_sq = 0.0, norm_sq = 0.0;
        bool generic_finite = true;
        std::size_t generic_bit_mismatches = 0;
        for (std::size_t i = 0; i < 8u * static_cast<std::size_t>(out_features); ++i) {
            const double a = half_to_float(repeated[guard + i]);
            const double b = half_to_float(actual[guard + i]);
            generic_finite = generic_finite && std::isfinite(a) && std::isfinite(b);
            generic_bit_mismatches += repeated[guard + i] != actual[guard + i];
            const double delta = a - b;
            max_abs = std::max(max_abs, std::abs(delta));
            error_sq += delta * delta;
            norm_sq += b * b;
        }
        require(generic_finite, "draft-small retained generic M8 produced nonfinite output");
        out << "generic_compare," << item.layer << ',' << item.name << ",8,"
            << in_features << ',' << out_features << ',' << item.metadata.K << ','
            << generic.dispatch_name(item.metadata, 8) << ',' << generic_repeat << ','
            << generic_guards << ',' << generic_finite << ',' << generic_bit_mismatches << ','
            << std::setprecision(12) << max_abs << ','
            << (norm_sq > 0.0 ? std::sqrt(error_sq / norm_sq) : std::sqrt(error_sq))
            << ",0,0,0,0,0,0,-1,-1,-1,NA," << input_distinct_rows << '\n';
        require(generic_guards && generic_repeat,
                "draft-small retained generic M8 repeat/canary check failed");
    }
    std::vector<const DraftSmallCapturedProjection*> representatives;
    for (const auto& item : captured) {
        const auto match = [&](const DraftSmallCapturedProjection* prior) {
            return prior->metadata.in_features == item.metadata.in_features &&
                   prior->metadata.out_features == item.metadata.out_features;
        };
        if (std::find_if(representatives.begin(), representatives.end(), match) ==
            representatives.end()) representatives.push_back(&item);
    }
    require(representatives.size() == 4,
            "draft-small real capture did not contain all four admitted shapes");

    // Dispatch/latching boundaries are checked on every admitted shape. Only a
    // draft-owned workspace requests the qualified unset default; generic and
    // target-like workspaces remain generic unless exact env=1 overrides them.
    for (const auto* item : representatives) {
        const int in_features = item->metadata.in_features;
        const int out_features = item->metadata.out_features;
        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "");
        Exl3CudaLinearWorkspace generic_unset(in_features, out_features, 9);
        Exl3CudaLinearWorkspace target_like_unset(in_features, out_features, 9, false);
        Exl3CudaLinearWorkspace draft_unset(in_features, out_features, 9, true);
        require(std::string(generic_unset.dispatch_name(item->metadata, 8)) == "generic_tile" &&
                    std::string(target_like_unset.dispatch_name(item->metadata, 8)) ==
                        "generic_tile",
                "draft-small generic-owned unset workspace enabled candidate");
        require(std::string(draft_unset.dispatch_name(item->metadata, 8)) ==
                    "draft_small_m_mma_split",
                "draft-small draft-owned unset workspace missed qualified default");
        for (const char* disabled : {"0", "2", "true", "01"}) {
            _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", disabled);
            Exl3CudaLinearWorkspace generic_disabled(in_features, out_features, 9);
            Exl3CudaLinearWorkspace draft_disabled(in_features, out_features, 9, true);
            require(std::string(generic_disabled.dispatch_name(item->metadata, 8)) ==
                        "generic_tile" &&
                        std::string(draft_disabled.dispatch_name(item->metadata, 8)) ==
                            "generic_tile",
                    std::string("draft-small non-exact opt-in enabled candidate: ") + disabled);
        }
        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "0");
        Exl3CudaLinearWorkspace generic_zero_latched(in_features, out_features, 9);
        Exl3CudaLinearWorkspace draft_zero_latched(in_features, out_features, 9, true);
        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "1");
        Exl3CudaLinearWorkspace on_ws(in_features, out_features, 9);
        Exl3CudaLinearWorkspace draft_on_ws(in_features, out_features, 9, true);
        require(std::string(on_ws.dispatch_name(item->metadata, 1)) == "generic_mma_split" &&
                    std::string(on_ws.dispatch_name(item->metadata, 8)) ==
                        "draft_small_m_mma_split" &&
                    std::string(draft_on_ws.dispatch_name(item->metadata, 8)) ==
                        "draft_small_m_mma_split",
                "draft-small opt-in dispatch boundary failed");
        require(std::string(generic_zero_latched.dispatch_name(item->metadata, 8)) ==
                    "generic_tile" &&
                    std::string(draft_zero_latched.dispatch_name(item->metadata, 8)) ==
                        "generic_tile",
                "draft-small explicit-zero workspaces observed post-construction env flip");
        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "0");
        require(std::string(on_ws.dispatch_name(item->metadata, 8)) ==
                    "draft_small_m_mma_split" &&
                    std::string(draft_on_ws.dispatch_name(item->metadata, 8)) ==
                        "draft_small_m_mma_split" &&
                    std::string(draft_unset.dispatch_name(item->metadata, 8)) ==
                        "draft_small_m_mma_split" &&
                    std::string(generic_unset.dispatch_name(item->metadata, 8)) == "generic_tile" &&
                    std::string(target_like_unset.dispatch_name(item->metadata, 8)) ==
                        "generic_tile",
                "draft-small candidate workspace did not latch opt-in");
        require(std::string(on_ws.dispatch_name(item->metadata, 9)) == "generic_tile",
                "draft-small M9 escaped bounded candidate");
        for (int metadata_case = 0; metadata_case < 4; ++metadata_case) {
            auto unrelated = item->metadata;
            if (metadata_case == 0) unrelated.K = 6;
            if (metadata_case == 1) unrelated.mcg = true;
            if (metadata_case == 2) unrelated.mul1 = false;
            if (metadata_case == 3) unrelated.has_bias = true;
            require(std::string(on_ws.dispatch_name(unrelated, 8)) == "generic_tile",
                    "draft-small candidate captured unrelated metadata");
        }

        DeviceBuffer invalid_input(item->input.size() * sizeof(std::uint16_t));
        const std::size_t invalid_elements =
            9u * static_cast<std::size_t>(out_features) + 2u * guard;
        DeviceBuffer invalid_output(invalid_elements * sizeof(std::uint16_t));
        auto* invalid_input_device = static_cast<std::uint16_t*>(invalid_input.get());
        auto* invalid_output_base = static_cast<std::uint16_t*>(invalid_output.get());
        auto* invalid_output_device = invalid_output_base + guard;
        const std::vector<std::uint16_t> invalid_canaries(invalid_elements, sentinel);
        std::vector<std::uint16_t> invalid_after(invalid_elements);
        cuda_check(cudaMemcpy(invalid_input_device, item->input.data(),
                              item->input.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "draft-small invalid-row input");
        on_ws.transform_input(item->weights, item->metadata, invalid_input_device, 1);
        for (int invalid_rows : {-1, 0, 10}) {
            for (int entry = 0; entry < 2; ++entry) {
                cuda_check(cudaMemcpy(invalid_output_base, invalid_canaries.data(),
                                      invalid_elements * sizeof(std::uint16_t),
                                      cudaMemcpyHostToDevice),
                           "draft-small invalid-row canaries");
                bool rejected = false;
                try {
                    if (entry == 0) {
                        on_ws.forward(item->weights, item->metadata, invalid_input_device,
                                      invalid_output_device, invalid_rows);
                    } else {
                        on_ws.forward_from_transformed(
                            item->weights, item->metadata, on_ws.transformed_device(),
                            invalid_output_device, invalid_rows);
                    }
                } catch (const std::invalid_argument&) {
                    rejected = true;
                }
                require(rejected, "draft-small invalid row count was not rejected");
                cuda_check(cudaMemcpy(invalid_after.data(), invalid_output_base,
                                      invalid_elements * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost),
                           "draft-small invalid-row output check");
                require(invalid_after == invalid_canaries,
                        "draft-small invalid row call modified output before rejection");
            }
        }
        std::cout << "E5A5K5_INVALID_ROWS_PASS in=" << in_features
                  << " out=" << out_features
                  << " rows=-1|0|10 entries=forward|forward_from_transformed\n";
    }

    auto synthetic_input = [](int in_features) {
        std::vector<std::uint16_t> values(8u * static_cast<std::size_t>(in_features), 0);
        for (int column = 0; column < in_features; ++column) {
            values[static_cast<std::size_t>(column)] = __half_as_ushort(__float2half_rn(
                std::sin(static_cast<float>(column) * 0.013F)));
            values[static_cast<std::size_t>(in_features + column)] =
                __half_as_ushort(__float2half_rn(
                    std::sin(static_cast<float>(column) * 0.007F + 1.0F)));
            values[static_cast<std::size_t>(2 * in_features + column)] =
                __half_as_ushort(__float2half_rn((column & 1) ? 2.0F : -2.0F));
            values[static_cast<std::size_t>(7 * in_features + column)] =
                __half_as_ushort(__float2half_rn((column & 1) ? -0.25F : 0.5F));
        }
        values[static_cast<std::size_t>(4 * in_features)] =
            __half_as_ushort(__float2half_rn(16.0F));
        values[static_cast<std::size_t>(6 * in_features)] =
            __half_as_ushort(__float2half_rn(3.0F));
        values[static_cast<std::size_t>(6 * in_features + in_features - 1)] =
            __half_as_ushort(__float2half_rn(-5.0F));
        values[static_cast<std::size_t>(6 * in_features - 1)] =
            __half_as_ushort(__float2half_rn(-16.0F));
        return values;
    };

    // One distinct-row synthetic profile per admitted shape covers zero,
    // alternating signs, smooth values, and bounded first/last impulses.
    for (const auto* item : representatives) {
        const int in_features = item->metadata.in_features;
        const int out_features = item->metadata.out_features;
        const auto synthetic = synthetic_input(in_features);
        DeviceBuffer input(synthetic.size() * sizeof(std::uint16_t));
        DeviceBuffer output((8u * static_cast<std::size_t>(out_features) + 2u * guard) *
                            sizeof(std::uint16_t));
        DeviceBuffer reference(static_cast<std::size_t>(out_features) * sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* output_base = static_cast<std::uint16_t*>(output.get());
        auto* output_device = output_base + guard;
        auto* reference_device = static_cast<std::uint16_t*>(reference.get());
        const std::size_t elements = 8u * static_cast<std::size_t>(out_features) + 2u * guard;
        const std::vector<std::uint16_t> initial(elements, sentinel);
        std::vector<std::uint16_t> actual(elements), repeated(elements);
        std::vector<std::uint16_t> row_reference(static_cast<std::size_t>(out_features));
        cuda_check(cudaMemcpy(input_device, synthetic.data(), synthetic.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "draft-small synthetic upload");
        _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");
        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "1");
        Exl3CudaLinearWorkspace workspace(in_features, out_features, 8);
        for (int m = 2; m <= 8; ++m) {
            cuda_check(cudaMemcpy(output_base, initial.data(), elements * sizeof(std::uint16_t),
                                  cudaMemcpyHostToDevice), "draft-small synthetic canaries");
            workspace.forward(item->weights, item->metadata, input_device, output_device, m);
            cuda_check(cudaMemcpy(actual.data(), output_base, elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "draft-small synthetic result");
            workspace.forward(item->weights, item->metadata, input_device, output_device, m);
            cuda_check(cudaMemcpy(repeated.data(), output_base, elements * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "draft-small synthetic repeat");
            const bool guards_ok =
                std::all_of(actual.begin(), actual.begin() + guard,
                            [](std::uint16_t value) { return value == sentinel; }) &&
                std::all_of(actual.begin() + guard + static_cast<std::size_t>(m) * out_features,
                            actual.end(), [](std::uint16_t value) { return value == sentinel; });
            std::size_t mismatches = 0;
            bool finite = true;
            for (int row = 0; row < m; ++row) {
                workspace.forward(item->weights, item->metadata,
                                  input_device + static_cast<std::size_t>(row) * in_features,
                                  reference_device, 1);
                cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                      row_reference.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost), "draft-small synthetic M1");
                const std::size_t offset = guard + static_cast<std::size_t>(row) * out_features;
                for (int column = 0; column < out_features; ++column) {
                    const auto bits = actual[offset + static_cast<std::size_t>(column)];
                    mismatches += bits != row_reference[static_cast<std::size_t>(column)];
                    finite = finite && std::isfinite(half_to_float(bits));
                }
            }
            const bool repeat_equal = actual == repeated;
            out << "synthetic," << item->layer << ',' << item->name << ',' << m << ','
                << in_features << ',' << out_features << ',' << item->metadata.K << ','
                << workspace.dispatch_name(item->metadata, m) << ',' << repeat_equal << ','
                << guards_ok << ',' << finite << ',' << mismatches
                << ",0,0,0,0,0,0,0,0,-1,-1,-1,NA,8\n";
            require(repeat_equal && guards_ok && finite && mismatches == 0,
                    "draft-small synthetic exact qualification failed");
        }

        // Explicit S1 is compared only with its own M1 topology.
        _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "1");
        Exl3CudaLinearWorkspace split1(in_features, out_features, 8);
        cuda_check(cudaMemcpy(output_base, initial.data(), elements * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice), "draft-small S1 canaries");
        split1.forward(item->weights, item->metadata, input_device, output_device, 8);
        cuda_check(cudaMemcpy(actual.data(), output_base, elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost), "draft-small S1 result");
        split1.forward(item->weights, item->metadata, input_device, output_device, 8);
        cuda_check(cudaMemcpy(repeated.data(), output_base, elements * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost), "draft-small S1 repeat");
        const bool split1_repeat = actual == repeated;
        const bool split1_guards =
            std::all_of(actual.begin(), actual.begin() + guard,
                        [](std::uint16_t value) { return value == sentinel; }) &&
            std::all_of(actual.begin() + guard + 8u * static_cast<std::size_t>(out_features),
                        actual.end(), [](std::uint16_t value) { return value == sentinel; });
        std::size_t split1_mismatches = 0;
        bool split1_finite = true;
        for (int row = 0; row < 8; ++row) {
            split1.forward(item->weights, item->metadata,
                           input_device + static_cast<std::size_t>(row) * in_features,
                           reference_device, 1);
            cuda_check(cudaMemcpy(row_reference.data(), reference_device,
                                  row_reference.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost), "draft-small S1 M1");
            const std::size_t offset = guard + static_cast<std::size_t>(row) * out_features;
            for (int column = 0; column < out_features; ++column) {
                split1_mismatches += actual[offset + static_cast<std::size_t>(column)] !=
                                     row_reference[static_cast<std::size_t>(column)];
                split1_finite = split1_finite && std::isfinite(half_to_float(
                    actual[offset + static_cast<std::size_t>(column)]));
            }
        }
        require(split1_repeat && split1_guards && split1_finite && split1_mismatches == 0,
                "draft-small explicit S1 diverged from its same-topology M1 rows");
        out << "synthetic_s1," << item->layer << ',' << item->name << ",8,"
            << in_features << ',' << out_features << ',' << item->metadata.K << ','
            << split1.dispatch_name(item->metadata, 8)
            << ',' << split1_repeat << ',' << split1_guards << ',' << split1_finite << ','
            << split1_mismatches
            << ",0,0,0,0,0,0,0,0,-1,-1,-1,NA,8\n";
        _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");
    }

    // Stable-buffer capture/replay for M7/M8 on each admitted shape. The eager
    // reference owns distinct workspace/output storage; graph output is poisoned
    // before replay and the changed input was never evaluated by captured scratch.
    cudaStream_t graph_stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking),
               "draft-small graph stream create");
    for (const auto* item : representatives) {
        const int in_features = item->metadata.in_features;
        const int out_features = item->metadata.out_features;
        auto changed = synthetic_input(in_features);
        DeviceBuffer input(changed.size() * sizeof(std::uint16_t));
        const std::size_t graph_elements =
            8u * static_cast<std::size_t>(out_features) + 2u * guard;
        DeviceBuffer graph_output(graph_elements * sizeof(std::uint16_t));
        DeviceBuffer eager_output(graph_elements * sizeof(std::uint16_t));
        auto* input_device = static_cast<std::uint16_t*>(input.get());
        auto* graph_base = static_cast<std::uint16_t*>(graph_output.get());
        auto* eager_base = static_cast<std::uint16_t*>(eager_output.get());
        auto* graph_destination = graph_base + guard;
        auto* eager_destination = eager_base + guard;
        const std::vector<std::uint16_t> graph_initial(graph_elements, sentinel);
        _putenv_s("NINFER_EXL3_GENERIC_SPLITS", "");
        _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "1");
        Exl3CudaLinearWorkspace graph_workspace(in_features, out_features, 8);
        Exl3CudaLinearWorkspace eager_workspace(in_features, out_features, 8);
        for (int m : {7, 8}) {
            _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "1");
            std::vector<std::uint16_t> seed(changed.size(), 0);
            cuda_check(cudaMemcpyAsync(input_device, seed.data(), seed.size() * sizeof(std::uint16_t),
                                       cudaMemcpyHostToDevice, graph_stream),
                       "draft-small graph seed upload");
            cuda_check(cudaStreamSynchronize(graph_stream), "draft-small graph seed sync");
            DraftSmallGraph graph;
            cuda_check(cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal),
                       "draft-small graph begin");
            graph_workspace.forward(item->weights, item->metadata, input_device,
                                    graph_destination, m, graph_stream);
            cuda_check(cudaStreamEndCapture(graph_stream, &graph.graph), "draft-small graph end");
            std::size_t graph_nodes = 0;
            cuda_check(cudaGraphGetNodes(graph.graph, nullptr, &graph_nodes),
                       "draft-small graph node count");
            require(graph_nodes > 0, "draft-small captured an empty graph");
            cuda_check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0),
                       "draft-small graph instantiate");
            changed[static_cast<std::size_t>(m - 7) * in_features + 17] =
                __half_as_ushort(__float2half_rn(m == 7 ? 3.25F : -5.5F));
            cuda_check(cudaMemcpyAsync(input_device, changed.data(), changed.size() * sizeof(std::uint16_t),
                                       cudaMemcpyHostToDevice, graph_stream),
                       "draft-small graph changed input");
            cuda_check(cudaMemcpyAsync(eager_base, graph_initial.data(),
                                       graph_elements * sizeof(std::uint16_t),
                                       cudaMemcpyHostToDevice, graph_stream),
                       "draft-small graph eager canaries");
            eager_workspace.forward(item->weights, item->metadata, input_device,
                                    eager_destination, m, graph_stream);
            cuda_check(cudaStreamSynchronize(graph_stream), "draft-small graph eager sync");
            std::vector<std::uint16_t> eager(graph_elements), replay(graph_elements), replay2(graph_elements);
            cuda_check(cudaMemcpy(eager.data(), eager_base,
                                  graph_elements * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                       "draft-small graph eager download");
            _putenv_s("NINFER_EXL3_DRAFT_SMALL_M", "0");
            cuda_check(cudaMemcpyAsync(graph_base, graph_initial.data(),
                                       graph_elements * sizeof(std::uint16_t),
                                       cudaMemcpyHostToDevice, graph_stream),
                       "draft-small graph poison");
            cuda_check(cudaGraphLaunch(graph.executable, graph_stream),
                       "draft-small graph replay");
            cuda_check(cudaStreamSynchronize(graph_stream), "draft-small graph replay sync");
            cuda_check(cudaMemcpy(replay.data(), graph_base,
                                  graph_elements * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                       "draft-small graph replay download");
            cuda_check(cudaGraphLaunch(graph.executable, graph_stream),
                       "draft-small graph repeat replay");
            cuda_check(cudaStreamSynchronize(graph_stream), "draft-small graph repeat sync");
            cuda_check(cudaMemcpy(replay2.data(), graph_base,
                                  graph_elements * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                       "draft-small graph repeat download");
            const bool graph_repeat = replay == replay2;
            const bool graph_guards =
                std::all_of(replay.begin(), replay.begin() + guard,
                            [](std::uint16_t value) { return value == sentinel; }) &&
                std::all_of(replay.begin() + guard + static_cast<std::size_t>(m) * out_features,
                            replay.end(), [](std::uint16_t value) { return value == sentinel; }) &&
                std::all_of(eager.begin(), eager.begin() + guard,
                            [](std::uint16_t value) { return value == sentinel; }) &&
                std::all_of(eager.begin() + guard + static_cast<std::size_t>(m) * out_features,
                            eager.end(), [](std::uint16_t value) { return value == sentinel; });
            bool graph_finite = true;
            std::size_t graph_mismatches = 0;
            for (std::size_t i = 0; i < static_cast<std::size_t>(m) * out_features; ++i) {
                graph_mismatches += replay[guard + i] != eager[guard + i];
                graph_finite = graph_finite &&
                    std::isfinite(half_to_float(replay[guard + i]));
            }
            require(graph_repeat && graph_guards && graph_finite && graph_mismatches == 0,
                    "draft-small changed-input graph replay diverged from eager");
            require(std::string(graph_workspace.dispatch_name(item->metadata, m)) ==
                        "draft_small_m_mma_split",
                    "draft-small graph workspace observed post-construction env flip");
            out << "graph," << item->layer << ',' << item->name << ',' << m << ','
                << in_features << ',' << out_features << ',' << item->metadata.K << ','
                << graph_workspace.dispatch_name(item->metadata, m)
                << ',' << graph_repeat << ',' << graph_guards << ',' << graph_finite << ','
                << graph_mismatches << ",0,0,0,0,0,0,0,0,-1,-1,-1,NA,8\n";
        }
    }
    cuda_check(cudaStreamDestroy(graph_stream), "draft-small graph stream destroy");
    out.flush();
    std::cout << "E5A5K5_CAPTURE_EXACT_PASS calls=" << captured.size()
              << " baseline_proposals=";
    for (std::size_t i = 0; i < baseline_proposals.size(); ++i) {
        if (i != 0) std::cout << '|';
        std::cout << baseline_proposals[i];
    }
    std::cout << '\n';
}
