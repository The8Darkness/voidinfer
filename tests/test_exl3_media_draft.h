#pragma once
void run_media_draft_gate(Exl3TextModel& model,Exl3Dflash2DraftModel& draft,
    const std::vector<std::int64_t>& code,const std::vector<std::int64_t>& prose){
    using namespace ninfer::exl3;using Json=nlohmann::ordered_json;
    const std::filesystem::path dir=env("NINFER_REAL_DFLASH_OUT"),frozen=env("NINFER_TEST_FROZEN_V6_EMBEDDINGS");
    require(!dir.empty()&&!std::filesystem::exists(dir),"preserve media draft receipt");std::filesystem::create_directories(dir);
    require(std::filesystem::file_size(frozen)==64ULL*5120*4,"frozen V6 extent");
    std::vector<float> features(64*5120);std::ifstream input(frozen,std::ios::binary);input.read(reinterpret_cast<char*>(features.data()),features.size()*4);require(bool(input),"frozen V6 read");
    const std::filesystem::path plan_path=env("NINFER_TEST_PREPARED_MEDIA_PLAN");const bool prepared=!plan_path.empty();
    Json plan;std::vector<std::int64_t> prepared_ids;std::vector<std::int32_t> prepared_positions;int prefix=64;
    if(prepared){std::ifstream file(plan_path);plan=Json::parse(file);prepared_ids=plan.at("token_ids").get<std::vector<std::int64_t>>();prepared_positions=plan.at("positions_axis_major").get<std::vector<std::int32_t>>();
        const auto& items=plan.at("vision_items");require(plan.at("patch_storage")=="fp16"&&items.size()==1&&items[0].at("modality")==1&&items[0].at("grid")==Json::array({1,16,16})&&items[0].at("token_spans").size()==1,"prepared single image geometry");
        prefix=items[0].at("token_spans")[0].at("begin").get<int>();require(prefix>0&&items[0].at("token_spans")[0].at("count")==64&&prepared_ids.size()>std::size_t(prefix+64)&&prepared_positions.size()==prepared_ids.size()*3,"prepared image span/position extent");
        auto types=plan.at("token_types").get<std::vector<int>>();require(types.size()==prepared_ids.size(),"prepared token type extent");
        for(std::size_t i=0;i<types.size();++i){require(types[i]==(i>=prefix&&i<prefix+64?1:0),"prepared image type span");require(prepared_ids[i]>=0&&prepared_ids[i]<248320,"prepared vocabulary extent");if(types[i])require(prepared_ids[i]==248056,"prepared image placeholder identity");}
    }
    std::ofstream checks(dir/"checks.csv");checks<<"case,pass\n";
    auto gate=[&](const char* name,bool pass){checks<<name<<','<<pass<<'\n';checks.flush();require(pass,std::string("media draft ")+name);};
    Json fixtures=Json::array();
    {
        std::array<std::unique_ptr<DeviceBuffer>,5> storage;std::array<std::uint16_t*,5> staging{};
        for(int t=0;t<5;++t){storage[t]=std::make_unique<DeviceBuffer>(16ULL*5120*2);staging[t]=static_cast<std::uint16_t*>(storage[t]->get());}
        struct Block {int first,rows;std::array<std::vector<std::uint16_t>,5> taps;};
        auto commit=[&](Exl3Dflash2DraftModel& owner,const Block& block){std::array<const std::uint16_t*,5> pointers{};
            for(int t=0;t<5;++t){require(block.taps[t].size()==std::size_t(block.rows)*5120,"media conditioning tap extent");cuda_check(cudaMemcpy(staging[t],block.taps[t].data(),block.taps[t].size()*2,cudaMemcpyHostToDevice),"media conditioning upload");pointers[t]=staging[t];}
            owner.commit_prefill_block(pointers.data(),block.rows,block.first);
        };
        auto independent=draft.create_execution();
        for(int fixture=0;fixture<(prepared?1:2);++fixture){
            const auto& ids=prepared?prepared_ids:(fixture?prose:code);require(ids.size()>=prefix,"media draft prefix extent");
            auto exact=model.create_context(true);exact->prepare_continuation(8);
            if(prepared)for(int i=0;i<prefix;++i)for(int axis=0;axis<3;++axis)require(prepared_positions[axis*ids.size()+i]==i,"prepared text prefix scalar rotary identity");
            auto text=Exl3VeriCacheRequest::initialize(*exact,std::span<const std::int64_t>(ids.data(),prefix),1024);text->restore_draft(draft,staging);
            std::vector<std::int32_t> xyz(64*3);for(int i=0;i<64;++i){xyz[i*3]=prefix;xyz[i*3+1]=prefix+i/8;xyz[i*3+2]=prefix+i%8;}
            if(prepared)for(int i=0;i<64;++i)for(int axis=0;axis<3;++axis)xyz[i*3+axis]=prepared_positions[axis*ids.size()+prefix+i];
            std::vector<Block> blocks;
            for(int i=0;i<64;i+=8){exact->append_media_embeddings_numeric(std::span<const float>(features).subspan(i*5120,8*5120),std::span<const std::int32_t>(xyz).subspan(i*3,24));
                blocks.push_back({prefix+i,8,exact->exact_tap_rows_host()});commit(draft,blocks.back());exact->finish_exact_prefill();}
            if(prepared)for(std::size_t i=prefix+64;i<ids.size();){const int rows=static_cast<int>(std::min<std::size_t>(8,ids.size()-i));
                for(int r=0;r<rows;++r)for(int axis=0;axis<3;++axis)require(prepared_positions[axis*ids.size()+i+r]==int(i+r)+exact->rope_offset(),"prepared suffix scalar/offset rotary identity");
                if(rows==1)exact->decode(ids[i]);else exact->continue_rows(std::span<const std::int64_t>(ids).subspan(i,rows));
                blocks.push_back({int(i),rows,exact->exact_tap_rows_host()});commit(draft,blocks.back());if(rows>1)exact->finish_exact_continuation();i+=rows;
            }
            auto initial=exact->export_exact_host_state();auto initial_ring=draft.export_host_ring(nullptr,false);
            const int initial_position=prepared?static_cast<int>(ids.size()):128;
            gate("media_L2_offset_logical_draft_frontier",initial->position()==initial_position&&initial->rope_offset()==-56&&draft.ring_base_abs()+draft.ring_count()==initial_position&&(!prepared||initial->rope_offset()==plan.at("rope_delta").get<int>()));
            text->restore_draft(*independent,staging);for(const auto& block:blocks)commit(*independent,block);
            gate("independent_media_conditioning_ring",initial_ring->same_payload(*independent->export_host_ring(nullptr,false)));
            auto reference=model.create_context(true);reference->prepare_continuation(8);reference->restore_exact_host_state(*initial);
            auto propose=[&](int rows){std::vector<std::int64_t> masked(rows,248070);masked[0]=sample_target(*exact);
                auto proposed=draft.propose_cached(masked,exact->position(),exact->target_embedding(),exact->target_lm_head_weights(),exact->target_lm_head_metadata(),248070);
                require(proposed.size()==std::size_t(rows-1),"actual B8 media proposal extent");masked.resize(1);masked.insert(masked.end(),proposed.begin(),proposed.end());return masked;};
            const auto first_proposals=propose(8);gate("proposal_does_not_commit_media_ring",initial_ring->same_payload(*draft.export_host_ring(nullptr,false)));
            const auto first=sample_target(*reference);reference->decode(first);const auto second=sample_target(*reference);reference->decode(second);
            const std::array<std::int64_t,2> forced{first,(second+1)%248320};auto rejected=verify_exl3_outer_batched_reference(*exact,*initial,forced,true);
            gate("forced_second_row_media_repair",rejected.rejected&&rejected.committed_tokens==std::vector<std::int64_t>{first,second}&&rejected.committed_state->same_payload(*reference->export_exact_host_state()));
            exact->restore_exact_host_state(*initial);draft.restore_host_ring(initial_ring);reference->restore_exact_host_state(*initial);rejected={};
            gate("media_abort_restore_and_proposal_repeat",exact->export_exact_host_state()->same_payload(*initial)&&propose(8)==first_proposals);
            std::vector<std::int64_t> tokens;std::uint64_t calls=0,proposals=0,verified=0,replay=0,accepted=0;auto root=initial;
            const std::array<std::int64_t,2> terminal{248044,248046};
            while(tokens.size()<32){const int rows=static_cast<int>(std::min<std::size_t>(8,32-tokens.size()));
                std::vector<std::int64_t> tentative;
                if(rows>1){tentative=propose(rows);++calls;proposals+=rows-1;}else tentative={sample_target(*exact)};
                auto result=rows>1?verify_exl3_outer_batched_reference(*exact,*root,tentative,true,terminal):verify_exl3_outer_reference(*exact,*root,tentative,true,terminal);
                std::array<std::vector<std::uint16_t>,5> taps;
                for(auto token:result.committed_tokens){gate("media_committed_token_scalar_M1",token==sample_target(*reference));reference->decode(token);auto row=reference->exact_tap_rows_host();for(int t=0;t<5;++t)taps[t].insert(taps[t].end(),row[t].begin(),row[t].end());}
                gate("media_round_full_state_and_all_repaired_taps",result.committed_state->same_payload(*reference->export_exact_host_state())&&taps==result.committed_taps);
                Block committed{root->position(),static_cast<int>(result.committed_tokens.size()),result.committed_taps};commit(draft,committed);
                // Independent scalar taps, same commit partitions; rejected rows never enter either ring.
                committed.taps=std::move(taps);commit(*independent,committed);
                gate("media_round_independent_ring",draft.export_host_ring(nullptr,false)->same_payload(*independent->export_host_ring(nullptr,false)));
                tokens.insert(tokens.end(),result.committed_tokens.begin(),result.committed_tokens.end());verified+=result.verification_rows;replay+=result.replay_rows;accepted+=result.accepted;root=result.committed_state;
                if(result.stopped)break;
            }
            gate("actual_media_draft_calls",calls>0&&proposals>0);
            fixtures.push_back(Json{{"fixture",prepared?"prepared-image-chat":fixture?"prose":"code"},{"prompt_tokens",initial_position},{"generated_including_stop",tokens.size()},{"tokens",tokens},{"proposal_calls",calls},{"proposed_rows",proposals},{"verified_rows",verified},{"replayed_rows",replay},{"accepted_including_seed",accepted},{"final_logical_position",root->position()},{"rope_offset",root->rope_offset()}});
            exact->restore_exact_host_state(*initial);draft.restore_host_ring(initial_ring);gate("retained_media_root_ring_unchanged",exact->export_exact_host_state()->same_payload(*initial)&&initial_ring->same_payload(*draft.export_host_ring(nullptr,false)));
            draft.reset();independent->reset();
        }
    }
    gate("all_media_draft_pins_retired",Exl3RecurrentPinBudget::snapshot()[0]==0&&Exl3RecurrentPinBudget::snapshot()[2]==0);
    std::ofstream(dir/"result.json")<<Json{{"status","PASS_MEDIA_DRAFT_RESEARCH"},{"fixtures",fixtures},{"complete_prepared_image_chat",prepared},{"coordinator_publication",false},{"frontend_media_identity",false},{"encoder_full_numeric_gate","FAILED"},{"quality","UNQUALIFIED"}}.dump(2);
    std::cout<<"PASS_MEDIA_DRAFT_RESEARCH\n";
}
