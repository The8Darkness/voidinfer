#pragma once

void run_media_state_gate(Exl3TextModel& model,const std::vector<std::int64_t>& code,
    const std::vector<std::int64_t>& prose){
    using namespace ninfer::exl3;using Json=nlohmann::ordered_json;
    const std::filesystem::path dir=env("NINFER_REAL_DFLASH_OUT");
    require(!dir.empty()&&!std::filesystem::exists(dir),"preserve media-state receipt");std::filesystem::create_directories(dir);
    std::ofstream checks(dir/"checks.csv");checks<<"case,pass\n";
    auto gate=[&](const char* name,bool pass){checks<<name<<','<<pass<<'\n';checks.flush();require(pass,std::string("media state ")+name);};
    struct Audit {std::vector<std::int32_t> positions;double squared=0,reference=0,maximum=0;std::uint64_t rotated=0,unrotated=0;bool exact_tail=true;};
    const auto observer=[](const Exl3LayerObservation& layer,void* user){
        if(!layer.attention.q_normed)return;
        auto& a=*static_cast<Audit*>(user);require(a.positions.size()==std::size_t(layer.rows)*3,"MRoPE audit position extent");
        auto plane=[&](const std::uint16_t* before,const std::uint16_t* after,int heads){
            std::vector<std::uint16_t> input(std::size_t(layer.rows)*heads*256),output(input.size());
            cuda_check(cudaMemcpyAsync(input.data(),before,input.size()*2,cudaMemcpyDeviceToHost,layer.stream),"MRoPE represented input audit");
            cuda_check(cudaMemcpyAsync(output.data(),after,output.size()*2,cudaMemcpyDeviceToHost,layer.stream),"MRoPE represented output audit");
            cuda_check(cudaStreamSynchronize(layer.stream),"MRoPE audit completion");
            for(int r=0;r<layer.rows;++r)for(int h=0;h<heads;++h)for(int c=0;c<256;++c){
                const auto index=(std::size_t(r)*heads+h)*256+c;
                if(c>=64){a.exact_tail&=input[index]==output[index];++a.unrotated;continue;}
                const int pair=c%32,mate=c<32?c+32:c-32;const double angle=double(a.positions[r*3+pair%3])*std::pow(10000000.,-double(pair)/32.);
                const double x=half_to_float(input[index]),y=half_to_float(input[(std::size_t(r)*heads+h)*256+mate]);
                const double expected=x*std::cos(angle)+(c<32?-y:y)*std::sin(angle),delta=double(half_to_float(output[index]))-expected;
                a.squared+=delta*delta;a.reference+=expected*expected;a.maximum=std::max(a.maximum,std::abs(delta));++a.rotated;
            }
        };
        plane(layer.attention.q_normed,layer.attention.q_rope,24);plane(layer.attention.k_normed,layer.attention.k_rope,4);
    };
    Json audits=Json::array();
    for(int fixture=0;fixture<2;++fixture){
        const auto& ids=fixture?prose:code;require(ids.size()>=80,"media state text fixture length");
        _putenv_s("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH","0");auto plain=model.create_context(true);plain->prepare_continuation(8);
        auto initial=Exl3VeriCacheRequest::initialize(*plain,std::span<const std::int64_t>(ids.data(),64),1024)->state();
        _putenv_s("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH","1");auto media=model.create_context(true);media->prepare_continuation(8);
        gate("bounded_private_media_scratch",media->persistent_bytes()==plain->persistent_bytes()+8ULL*5120*4+8ULL*3*4);
        auto embeddings=[&](int count){
            std::vector<std::uint16_t> raw(std::size_t(count)*5120);std::vector<float> features(raw.size());
            for(int i=0;i<count;++i)cuda_check(cudaMemcpy(raw.data()+std::size_t(i)*5120,plain->target_embedding()+std::size_t(ids[64+i])*5120,5120*2,cudaMemcpyDeviceToHost),"independent BF16 embedding row");
            for(std::size_t i=0;i<raw.size();++i)features[i]=std::bit_cast<float>(std::uint32_t(raw[i])<<16);
            return features;
        };
        for(int rows:{1,3,8}){
            plain->restore_exact_host_state(*initial);media->restore_exact_host_state(*initial);
            std::array<std::vector<std::uint16_t>,5> expected_taps;
            for(int i=0;i<rows;++i){plain->decode(ids[64+i]);auto taps=plain->exact_tap_rows_host();for(int t=0;t<5;++t)expected_taps[t].insert(expected_taps[t].end(),taps[t].begin(),taps[t].end());}
            auto expected=plain->export_exact_host_state();auto features=embeddings(rows);std::vector<std::int32_t> positions(rows*3);
            for(int i=0;i<rows;++i)for(int axis=0;axis<3;++axis)positions[i*3+axis]=64+i;
            media->append_media_embeddings_numeric(features,positions);gate("equal_axes_all_committed_taps",media->exact_tap_rows_host()==expected_taps);media->finish_exact_prefill();
            gate("text_embeddings_equal_axes_full_M1_state",media->rope_offset()==0&&media->export_exact_host_state()->same_payload(*expected));
        }
        media->restore_exact_host_state(*initial);auto features=embeddings(8);Audit audit;audit.positions.resize(24);
        for(int i=0;i<8;++i){audit.positions[i*3]=64+i/4;audit.positions[i*3+1]=64+i%2;audit.positions[i*3+2]=64+(i/2)%2;}
        media->set_layer_observer_for_test(observer,&audit);media->append_media_embeddings_numeric(features,audit.positions);media->set_layer_observer_for_test(nullptr);
        const double relative=std::sqrt(audit.squared/audit.reference);
        audits.push_back(Json{{"fixture",fixture?"prose":"code"},{"relative_rmse",relative},{"maxabs",audit.maximum},{"rotated_values",audit.rotated},{"unrotated_values",audit.unrotated},{"unrotated_exact",audit.exact_tail}});
        std::ofstream(dir/"rotary-audit.json")<<audits.dump(2);
        gate("independent_FP64_interleaved_rotary",audit.rotated>0&&relative<.001&&audit.maximum<.01&&audit.exact_tail);
        media->finish_exact_prefill();auto root=media->export_exact_host_state();
        gate("logical_cursor_separate_rotary_offset",root->position()==72&&root->rope_offset()==-6&&media->rope_offset()==-6);
        gate("changed_axes_change_numerical_state",!root->same_payload(*plain->export_exact_host_state()));
        bool refused=false;try{plain->restore_exact_host_state(*root);}catch(const std::exception&){refused=true;}gate("unprepared_context_refuses_nonzero_offset",refused);
        std::vector<std::int64_t> tokens;std::array<std::vector<std::uint16_t>,5> expected_taps;
        for(int i=0;i<8;++i){auto token=sample_target(*media);tokens.push_back(token);media->decode(token);auto taps=media->exact_tap_rows_host();for(int t=0;t<5;++t)expected_taps[t].insert(expected_taps[t].end(),taps[t].begin(),taps[t].end());}
        auto final=media->export_exact_host_state();auto recycled=model.create_context(true);recycled->prepare_continuation(8);recycled->restore_exact_host_state(*root);
        gate("restored_offset_and_resident_witness",recycled->rope_offset()==-6&&recycled->exact_host_state_resident(*root));
        recycled->continue_rows(tokens);gate("offset_native8_all_taps_match_M1",recycled->exact_tap_rows_host()==expected_taps);recycled->finish_exact_continuation();
        gate("offset_native8_full_state_match_M1",recycled->export_exact_host_state()->same_payload(*final));
        recycled->restore_exact_host_state(*root);gate("retained_media_root_unchanged",recycled->export_exact_host_state()->same_payload(*root));
        for(int bad=0;bad<6;++bad){auto f=features;auto p=audit.positions;if(bad==0)f.pop_back();if(bad==1)f[0]=std::numeric_limits<float>::quiet_NaN();if(bad==2)p.pop_back();if(bad==3)p[0]=-1;if(bad==4)p[0]=recycled->max_context();if(bad==5){f.resize(9*5120);p.resize(27);}
            refused=false;try{recycled->append_media_embeddings_numeric(f,p);}catch(const std::exception&){refused=true;}
            gate("invalid_media_input_before_mutation",refused&&recycled->export_exact_host_state()->same_payload(*root));
        }
        cudaStream_t stream=nullptr;cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"media capture test stream");
        cuda_check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"media capture test begin");refused=false;
        try{recycled->append_media_embeddings_numeric(features,audit.positions,stream);}catch(const std::exception&){refused=true;}
        cudaGraph_t graph=nullptr;const auto ended=cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);cudaStreamDestroy(stream);
        gate("external_capture_refused_before_mutation",refused&&ended==cudaSuccess&&recycled->export_exact_host_state()->same_payload(*root));
        recycled->reset();gate("reset_clears_offset",recycled->rope_offset()==0&&recycled->position()==0);
        recycled->restore_exact_host_state(*initial);plain->restore_exact_host_state(*initial);const auto token=sample_target(*plain);plain->decode(token);recycled->decode(token);
        gate("recycled_scalar_request_exact",recycled->export_exact_host_state()->same_payload(*plain->export_exact_host_state()));
    }
    const std::filesystem::path frozen=env("NINFER_TEST_FROZEN_V6_EMBEDDINGS");
    if(!frozen.empty()){
        require(std::filesystem::file_size(frozen)==64ULL*5120*4,"frozen native V6 embedding extent");
        std::vector<float> features(64*5120);std::ifstream input(frozen,std::ios::binary);
        input.read(reinterpret_cast<char*>(features.data()),features.size()*4);require(bool(input),"frozen V6 read");
        auto serial=model.create_context(true);serial->prepare_continuation(8);
        auto initial=Exl3VeriCacheRequest::initialize(*serial,std::span<const std::int64_t>(code.data(),64),1024)->state();
        auto batch=model.create_context(true);batch->prepare_continuation(8);batch->restore_exact_host_state(*initial);
        std::vector<std::int32_t> xyz(64*3);for(int i=0;i<64;++i){xyz[i*3]=64;xyz[i*3+1]=64+i/8;xyz[i*3+2]=64+i%8;}
        std::array<std::vector<std::uint16_t>,5> serial_taps,batch_taps;
        for(int i=0;i<64;++i){serial->append_media_embeddings_numeric(std::span<const float>(features).subspan(i*5120,5120),std::span<const std::int32_t>(xyz).subspan(i*3,3));
            auto taps=serial->exact_tap_rows_host();for(int t=0;t<5;++t)serial_taps[t].insert(serial_taps[t].end(),taps[t].begin(),taps[t].end());}
        for(int i=0;i<64;i+=8){batch->append_media_embeddings_numeric(std::span<const float>(features).subspan(i*5120,8*5120),std::span<const std::int32_t>(xyz).subspan(i*3,24));
            auto taps=batch->exact_tap_rows_host();for(int t=0;t<5;++t)batch_taps[t].insert(batch_taps[t].end(),taps[t].begin(),taps[t].end());batch->finish_exact_prefill();}
        auto root=batch->export_exact_host_state();
        gate("frozen_native_V6_64_rows_M1_M8_full_state_taps",root->same_payload(*serial->export_exact_host_state())&&serial_taps==batch_taps&&root->position()==128&&root->rope_offset()==-56);
        const auto original=features[0];features[0]=original+1;
        gate("retired_input_buffer_cannot_mutate_root",batch->export_exact_host_state()->same_payload(*root));
        batch->restore_exact_host_state(*initial);batch->append_media_embeddings_numeric(std::span<const float>(features).first(8*5120),std::span<const std::int32_t>(xyz).first(24));batch->finish_exact_prefill();auto changed=batch->export_exact_host_state();
        features[0]=original;batch->restore_exact_host_state(*initial);batch->append_media_embeddings_numeric(std::span<const float>(features).first(8*5120),std::span<const std::int32_t>(xyz).first(24));batch->finish_exact_prefill();
        gate("changed_represented_embedding_changes_state",!changed->same_payload(*batch->export_exact_host_state()));
        batch->restore_exact_host_state(*root);
        for(int i=0;i<8;++i){auto token=sample_target(*serial);gate("native_V6_restored_greedy_token",token==sample_target(*batch));serial->decode(token);batch->decode(token);}
        gate("native_V6_restored_text_continuation_full_state",serial->export_exact_host_state()->same_payload(*batch->export_exact_host_state()));
    }
    gate("all_media_gate_pins_retired",Exl3RecurrentPinBudget::snapshot()[0]==0&&Exl3RecurrentPinBudget::snapshot()[2]==0);
    std::ofstream(dir/"result.json")<<Json{{"status","PASS_MEDIA_STATE_RESEARCH"},{"rotary_audits",audits},{"native_v6_injected",!frozen.empty()},{"full_media_quality","UNQUALIFIED"},{"http_media",false}}.dump(2);
    std::cout<<"PASS_MEDIA_STATE_RESEARCH\n";
}
