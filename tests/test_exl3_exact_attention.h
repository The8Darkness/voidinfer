#pragma once
#include "attention_segment_reference.h"

void run_exact_attention_operator_oracle() {
    for(int rows:{1,8,16,1024})
        require(ninfer::exl3::Exl3FullAttentionLayer::shared_scratch_bytes(rows)-
            ninfer::exl3::Exl3FullAttentionLayer::shared_scratch_bytes(rows,true)==
            static_cast<std::size_t>(rows)*5120*2,"coalesced input/MLP scratch extent delta");
    int cases=0;
    std::vector<std::pair<int,int>> shapes{{31,1},{321,1},{321,3},{321,8},{321,15},{321,16},
        {4096,1},{16384,1},{321,17},{321,32},{321,65}};
    const auto* native64k=std::getenv("NINFER_TEST_EXACT_ATTENTION_NATIVE64K");
    require(!native64k || std::string_view(native64k)=="0" || std::string_view(native64k)=="1",
        "native64K attention fixture option must be 0 or 1");
    if(native64k && std::string_view(native64k)=="1")
        for(int rows:{1,8,15,16})shapes.emplace_back(65536,rows);
    for(const auto shape:shapes) {
        const auto [capacity,rows]=shape;
        const int position=capacity-rows;
        std::vector<std::uint16_t> q(rows*24*256),k(static_cast<std::size_t>(capacity)*4*256),v(k.size());
        const auto f16=[](float x){return __half_as_ushort(__float2half_rn(x));};
        for(std::size_t i=0;i<q.size();++i) q[i]=f16(static_cast<float>(0.3*std::sin(i*0.017)+0.1*std::cos(i*0.043)));
        for(std::size_t i=0;i<k.size();++i) {
            k[i]=f16(static_cast<float>(0.4*std::sin(i*0.031)+(i%997==0?1.0:0.0)));
            v[i]=f16(static_cast<float>(std::cos(i*0.029)*1.3-std::sin(i*0.073)*0.4));
        }
        DeviceBuffer dq(q.size()*2),dk(k.size()*2),dv(v.size()*2),out(q.size()*2),qshared(q.size()*2),half2(q.size()*2),vhalf2(q.size()*2),gqapair(q.size()*2),gqatriple(q.size()*2),gqatriple128(q.size()*2),gqatriplestaged(q.size()*2),gqasix(q.size()*2),gqasixscores(q.size()*2),gqasixextent(q.size()*2),gqasixquerypair(q.size()*2),gqasixsharded(q.size()*2),gqatriplevalues4(q.size()*2),gqasixsoftmaxtriple(q.size()*2),gqasixpackedtriples(q.size()*2),ref(q.size()*2);
        DeviceBuffer scores(static_cast<std::size_t>(std::min(rows,16))*24*capacity*4);
        cuda_check(cudaMemcpy(dq.get(),q.data(),q.size()*2,cudaMemcpyHostToDevice),"exactattn upload Q");
        cuda_check(cudaMemcpy(dk.get(),k.data(),k.size()*2,cudaMemcpyHostToDevice),"exactattn upload K");
        cuda_check(cudaMemcpy(dv.get(),v.data(),v.size()*2,cudaMemcpyHostToDevice),"exactattn upload V");
        const auto launch=[&](bool parallel,DeviceBuffer& destination,bool shared=false,bool paired=false,bool value_paired=false,bool gqa_pair=false,bool gqa_triple=false,bool gqa_triple_values128=false,bool gqa_triple_softmax_staged=false,bool gqa_six=false,bool gqa_six_scores=false,bool gqa_six_values_sharded=false,bool gqa_triple_values4=false,bool gqa_six_softmax_triple_values=false,bool gqa_six_packed_triples=false,bool gqa_six_extent_shards=false,bool gqa_six_query_pair_scores=false) {
            ninfer::exl3::exl3_exact_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(dk.get()),static_cast<const std::uint16_t*>(dv.get()),
                static_cast<std::uint16_t*>(destination.get()),static_cast<float*>(scores.get()),rows,position,capacity,parallel,nullptr,shared,paired,value_paired,gqa_pair,gqa_triple,gqa_triple_values128,gqa_triple_softmax_staged,gqa_six,gqa_six_scores,gqa_six_values_sharded,gqa_triple_values4,gqa_six_softmax_triple_values,gqa_six_packed_triples,gqa_six_extent_shards,gqa_six_query_pair_scores);
        };
        launch(true,out);
        launch(true,qshared,true);
        const auto baseline=target_continue_device_bits(static_cast<const std::uint16_t*>(out.get()),q.size(),"exactattn parallel result");
        if(rows<=16) for(int split:{0,1,position/2,position}) {
            if(split>position) continue;
            // Poison prefix rows in the tail allocation. A wrong address
            // selection cannot accidentally pass by reading duplicate bytes.
            auto tail_k=k,tail_v=v;
            std::fill_n(tail_k.begin(),static_cast<std::size_t>(split)*1024,0x7e00);
            std::fill_n(tail_v.begin(),static_cast<std::size_t>(split)*1024,0x7e00);
            DeviceBuffer tk(k.size()*2),tv(v.size()*2),segmented(q.size()*2);
            cuda_check(cudaMemcpy(tk.get(),tail_k.data(),tail_k.size()*2,cudaMemcpyHostToDevice),"segmented poisoned K");
            cuda_check(cudaMemcpy(tv.get(),tail_v.data(),tail_v.size()*2,cudaMemcpyHostToDevice),"segmented poisoned V");
            ninfer::exl3::exl3_exact_segmented_attention_for_test(
                static_cast<const std::uint16_t*>(dq.get()),static_cast<const std::uint16_t*>(dk.get()),
                static_cast<const std::uint16_t*>(dv.get()),split,static_cast<const std::uint16_t*>(tk.get()),
                static_cast<const std::uint16_t*>(tv.get()),static_cast<std::uint16_t*>(segmented.get()),
                static_cast<float*>(scores.get()),rows,position,capacity);
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(segmented.get()),q.size(),
                "segmented exact result")==baseline,"segmented attention represented bits differ");
        }
        if(rows<=16 && position>=128) for(int first:{1,64,position-64}) {
            constexpr int extent=64;
            auto private_k=k,private_v=v;
            std::fill_n(private_k.begin()+static_cast<std::size_t>(first)*1024,extent*1024,0x7e00);
            std::fill_n(private_v.begin()+static_cast<std::size_t>(first)*1024,extent*1024,0x7e00);
            // Compact allocation with guards makes absolute shared indexing visible.
            std::vector<std::uint16_t> shared_k((extent+2)*1024,0x7e00),shared_v(shared_k.size(),0x7e00);
            std::copy_n(k.begin()+static_cast<std::size_t>(first)*1024,extent*1024,shared_k.begin()+1024);
            std::copy_n(v.begin()+static_cast<std::size_t>(first)*1024,extent*1024,shared_v.begin()+1024);
            DeviceBuffer pk(k.size()*2),pv(v.size()*2),sk(shared_k.size()*2),sv(shared_v.size()*2),segmented(q.size()*2);
            cuda_check(cudaMemcpy(pk.get(),private_k.data(),private_k.size()*2,cudaMemcpyHostToDevice),"interval private K");
            cuda_check(cudaMemcpy(pv.get(),private_v.data(),private_v.size()*2,cudaMemcpyHostToDevice),"interval private V");
            cuda_check(cudaMemcpy(sk.get(),shared_k.data(),shared_k.size()*2,cudaMemcpyHostToDevice),"interval shared K");
            cuda_check(cudaMemcpy(sv.get(),shared_v.data(),shared_v.size()*2,cudaMemcpyHostToDevice),"interval shared V");
            ninfer::exl3::exl3_exact_segmented_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(sk.get())+1024,static_cast<const std::uint16_t*>(sv.get())+1024,extent,
                static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                static_cast<std::uint16_t*>(segmented.get()),static_cast<float*>(scores.get()),rows,position,capacity,nullptr,first);
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(segmented.get()),q.size(),
                "nonzero interval result")==baseline,"nonzero interval changed represented attention bits");
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(sk.get()),shared_k.size(),"interval K unchanged")==shared_k &&
                target_continue_device_bits(static_cast<const std::uint16_t*>(sv.get()),shared_v.size(),"interval V unchanged")==shared_v,
                "segmented attention modified immutable shared planes/guards");
        }
        if(rows<=16 && position>=256) {
            auto private_k=k,private_v=v;
            constexpr int page_rows=64,guarded_rows=66;
            const int page_count=std::min(64,position/128);
            std::vector<int> page_first(page_count);
            for(int page=0;page<page_count;++page)
                page_first[page]=static_cast<int>(static_cast<std::int64_t>(page)*(position-page_rows)/(page_count-1));
            std::vector<std::uint16_t> shared_k(static_cast<std::size_t>(page_count)*guarded_rows*1024,0x7e00),shared_v(shared_k.size(),0x7e00);
            for(int page=0;page<page_count;++page) {
                const auto first=static_cast<std::size_t>(page_first[page])*1024;
                const auto packed=(static_cast<std::size_t>(page)*guarded_rows+1)*1024;
                std::copy_n(k.begin()+first,page_rows*1024,shared_k.begin()+packed);
                std::copy_n(v.begin()+first,page_rows*1024,shared_v.begin()+packed);
                std::fill_n(private_k.begin()+first,page_rows*1024,0x7e00);
                std::fill_n(private_v.begin()+first,page_rows*1024,0x7e00);
            }
            // Independent chronological expansion certifies the control inputs
            // from the actual guarded shared storage and poisoned private gaps.
            const auto reference_k=std::make_shared<const attention_reference::Storage>(shared_k);
            const auto reference_v=std::make_shared<const attention_reference::Storage>(shared_v);
            const std::array<attention_reference::Owner,2> registry{reference_k,reference_v};
            std::vector<attention_reference::Segment> k_segments,v_segments;
            for(int page=page_count-1;page>=0;--page) {
                const auto first=static_cast<std::size_t>(page_first[page]);
                const auto packed=static_cast<std::size_t>(page)*guarded_rows+1;
                k_segments.push_back({first,page_rows,packed,0,reference_k});
                v_segments.push_back({first,page_rows,packed,1,reference_v});
            }
            require(attention_reference::expand(private_k,1024,capacity,k_segments,registry)==k &&
                attention_reference::expand(private_v,1024,capacity,v_segments,registry)==v,
                "independent segment reference disagrees with canonical attention inputs");
            for(int query=0;query<rows;++query)for(int key=position;key<capacity;++key)
                require(attention_reference::visible(position,rows,query,key)==(key<=position+query),
                    "independent attention reference future-row mask");
            DeviceBuffer pk(k.size()*2),pv(v.size()*2),sk(shared_k.size()*2),sv(shared_v.size()*2),paged(q.size()*2);
            cuda_check(cudaMemcpy(pk.get(),private_k.data(),private_k.size()*2,cudaMemcpyHostToDevice),"page private K");
            cuda_check(cudaMemcpy(pv.get(),private_v.data(),private_v.size()*2,cudaMemcpyHostToDevice),"page private V");
            cuda_check(cudaMemcpy(sk.get(),shared_k.data(),shared_k.size()*2,cudaMemcpyHostToDevice),"page shared K");
            cuda_check(cudaMemcpy(sv.get(),shared_v.data(),shared_v.size()*2,cudaMemcpyHostToDevice),"page shared V");
            ninfer::exl3::Exl3AttentionPageRanges ranges;
            // Reverse order matches Engine newest-page-first acquisition.
            for(int page=page_count-1;page>=0;--page) {
                const auto offset=(static_cast<std::size_t>(page)*guarded_rows+1)*1024;
                ranges.append(static_cast<const std::uint16_t*>(sk.get())+offset,
                    static_cast<const std::uint16_t*>(sv.get())+offset,page_first[page],page_rows,position);
            }
            ninfer::exl3::exl3_exact_page_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),ranges,
                static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                static_cast<std::uint16_t*>(paged.get()),static_cast<float*>(scores.get()),rows,position,capacity);
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(paged.get()),q.size(),"multi-page output")==baseline,
                "multi-page attention changed represented bits across private gaps");
            auto chronological=ranges;
            std::reverse(chronological.ranges,chronological.ranges+chronological.count);
            ninfer::exl3::exl3_exact_page_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),chronological,
                static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                static_cast<std::uint16_t*>(paged.get()),static_cast<float*>(scores.get()),rows,position,capacity);
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(paged.get()),q.size(),"chronological page output")==baseline,
                "page descriptor order changed chronological numerical result");
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(sk.get()),shared_k.size(),"page immutable K")==shared_k &&
                target_continue_device_bits(static_cast<const std::uint16_t*>(sv.get()),shared_v.size(),"page immutable V")==shared_v &&
                target_continue_device_bits(static_cast<const std::uint16_t*>(pk.get()),private_k.size(),"page private K guards")==private_k &&
                target_continue_device_bits(static_cast<const std::uint16_t*>(pv.get()),private_v.size(),"page private V guards")==private_v,
                "page order comparison mutated represented input or poisoned gaps");
            ninfer::exl3::exl3_exact_page_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),ranges,
                static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                static_cast<std::uint16_t*>(paged.get()),static_cast<float*>(scores.get()),rows,position,capacity,nullptr,true);
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(paged.get()),q.size(),"Q-shared multi-page output")==
                target_continue_device_bits(static_cast<const std::uint16_t*>(qshared.get()),q.size(),"Q-shared contiguous control"),
                "segmented Q-shared profile changed its represented output bits");
            launch(true,gqapair,true,true,true,true);
            ninfer::exl3::exl3_exact_page_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),ranges,
                static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                static_cast<std::uint16_t*>(paged.get()),static_cast<float*>(scores.get()),rows,position,capacity,nullptr,false,true);
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(paged.get()),q.size(),"GQA pair multi-page output")==
                target_continue_device_bits(static_cast<const std::uint16_t*>(gqapair.get()),q.size(),"GQA pair contiguous control"),
                "segmented GQA pair changed represented per-head output");
            ninfer::exl3::exl3_exact_page_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),ranges,
                static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                static_cast<std::uint16_t*>(paged.get()),static_cast<float*>(scores.get()),rows,position,capacity,nullptr,false,false,true);
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(paged.get()),q.size(),"query-pair multi-page output")==
                target_continue_device_bits(static_cast<const std::uint16_t*>(qshared.get()),q.size(),"query-pair canonical control"),
                "query-pair history reuse changed causal row output");
            for(int profile=0;profile<3;++profile) {
                ninfer::exl3::exl3_exact_page_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),chronological,
                    static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                    static_cast<std::uint16_t*>(paged.get()),static_cast<float*>(scores.get()),rows,position,capacity,
                    nullptr,profile==0,profile==1,profile==2);
                const auto* reference=static_cast<const std::uint16_t*>(profile==1?gqapair.get():qshared.get());
                require(target_continue_device_bits(static_cast<const std::uint16_t*>(paged.get()),q.size(),"ordered profile output")==
                    target_continue_device_bits(reference,q.size(),"ordered profile control"),
                    "attention profile changed arithmetic with page descriptor order");
            }
            // Exact-sized output plus surrounding canaries exposes an erroneous
            // write by the absent second query in an odd final pair.
            constexpr std::uint16_t query_guard=0x6d3b;
            std::vector<std::uint16_t> guarded_init(q.size()+2,query_guard);
            DeviceBuffer guarded_query(guarded_init.size()*2);
            cuda_check(cudaMemcpy(guarded_query.get(),guarded_init.data(),guarded_init.size()*2,cudaMemcpyHostToDevice),
                "query-pair output guards");
            ninfer::exl3::exl3_exact_page_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),ranges,
                static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                static_cast<std::uint16_t*>(guarded_query.get())+1,static_cast<float*>(scores.get()),rows,position,capacity,nullptr,false,false,true);
            const auto guarded_result=target_continue_device_bits(static_cast<const std::uint16_t*>(guarded_query.get()),
                guarded_init.size(),"query-pair guarded output");
            require(guarded_result.front()==query_guard && guarded_result.back()==query_guard,
                "query-pair wrote outside valid query rows");
            const auto query_control=target_continue_device_bits(
                static_cast<const std::uint16_t*>(qshared.get()),q.size(),"query-pair poison control");
            // Later queries may legitimately consume poisoned rows. Compare only
            // the chronological prefix whose entire causal history is unchanged.
            // In particular, the first query of each pair must not inherit NaNs
            // from the second query's extra K/V row (including zero * NaN).
            for(int last_valid=0;last_valid<rows-1;++last_valid) {
                auto future_k=private_k,future_v=private_v;
                const auto begin=static_cast<std::size_t>(position+last_valid+1)*1024;
                std::fill(future_k.begin()+begin,future_k.end(),0x7e00);
                std::fill(future_v.begin()+begin,future_v.end(),0x7e00);
                cuda_check(cudaMemcpy(pk.get(),future_k.data(),future_k.size()*2,cudaMemcpyHostToDevice),"query-pair future K poison");
                cuda_check(cudaMemcpy(pv.get(),future_v.data(),future_v.size()*2,cudaMemcpyHostToDevice),"query-pair future V poison");
                ninfer::exl3::exl3_exact_page_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),ranges,
                    static_cast<const std::uint16_t*>(pk.get()),static_cast<const std::uint16_t*>(pv.get()),
                    static_cast<std::uint16_t*>(paged.get()),static_cast<float*>(scores.get()),rows,position,capacity,nullptr,false,false,true);
                const auto prefix=target_continue_device_bits(static_cast<const std::uint16_t*>(paged.get()),
                    static_cast<std::size_t>(last_valid+1)*24*256,"query-pair unaffected causal prefix");
                require(std::equal(prefix.begin(),prefix.end(),query_control.begin()),
                    "query-pair future K/V contaminated an earlier query");
            }
            require(target_continue_device_bits(static_cast<const std::uint16_t*>(sk.get()),shared_k.size(),"multi-page K guards")==shared_k &&
                target_continue_device_bits(static_cast<const std::uint16_t*>(sv.get()),shared_v.size(),"multi-page V guards")==shared_v,
                "multi-page attention mutated shared storage or guards");
        }
        if(rows<=16) for(const auto range:std::array<std::pair<int,int>,4>{{{-1,1},{position+1,0},{position,1},{0,-1}}}) {
            bool refused=false;
            try {
                ninfer::exl3::exl3_exact_segmented_attention_for_test(static_cast<const std::uint16_t*>(dq.get()),
                    static_cast<const std::uint16_t*>(dk.get()),static_cast<const std::uint16_t*>(dv.get()),range.second,
                    static_cast<const std::uint16_t*>(dk.get()),static_cast<const std::uint16_t*>(dv.get()),
                    static_cast<std::uint16_t*>(out.get()),static_cast<float*>(scores.get()),rows,position,capacity,nullptr,range.first);
            } catch(const std::invalid_argument&) {refused=true;}
            require(refused,"segmented attention admitted negative or unpublished interval");
        }
        const auto actual=target_continue_device_bits(static_cast<const std::uint16_t*>(qshared.get()),q.size(),"exactattn Q-shared result");
        require(actual==baseline,"exactattn Q-shared bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,half2,true,true);
        const auto paired=target_continue_device_bits(static_cast<const std::uint16_t*>(half2.get()),q.size(),"exactattn K-half2 result");
        require(paired==baseline,"exactattn K-half2 bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,vhalf2,true,true,true);
        const auto value_paired=target_continue_device_bits(static_cast<const std::uint16_t*>(vhalf2.get()),q.size(),"exactattn V-half2 result");
        require(value_paired==baseline,"exactattn V-half2 bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqapair,true,true,true,true);
        const auto paired_heads=target_continue_device_bits(static_cast<const std::uint16_t*>(gqapair.get()),q.size(),"exactattn GQA-pair result");
        require(paired_heads==baseline,"exactattn GQA-pair bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqatriple,true,true,true,true,true);
        const auto triple_heads=target_continue_device_bits(static_cast<const std::uint16_t*>(gqatriple.get()),q.size(),"exactattn GQA-triple result");
        require(triple_heads==baseline,"exactattn GQA-triple bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqatriple128,true,true,true,true,true,true);
        const auto triple_heads128=target_continue_device_bits(static_cast<const std::uint16_t*>(gqatriple128.get()),q.size(),"exactattn GQA-triple values128 result");
        require(triple_heads128==baseline,"exactattn GQA-triple values128 bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqatriplestaged,true,true,true,true,true,false,true);
        const auto triple_heads_staged=target_continue_device_bits(static_cast<const std::uint16_t*>(gqatriplestaged.get()),q.size(),"exactattn GQA-triple staged-softmax result");
        require(triple_heads_staged==baseline,"exactattn GQA-triple staged-softmax bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqasix,true,true,true,true,true,false,true,true);
        const auto six_heads=target_continue_device_bits(static_cast<const std::uint16_t*>(gqasix.get()),q.size(),"exactattn GQA-six result");
        require(six_heads==baseline,"exactattn GQA-six bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqasixscores,true,true,true,true,true,false,true,false,true);
        const auto six_scores=target_continue_device_bits(static_cast<const std::uint16_t*>(gqasixscores.get()),q.size(),"exactattn GQA-six score-only result");
        require(six_scores==baseline,"exactattn GQA-six score-only bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        std::vector<float> six_score_workspace(static_cast<std::size_t>(std::min(rows,16))*24*capacity);
        cuda_check(cudaMemcpy(six_score_workspace.data(),scores.get(),
            six_score_workspace.size()*sizeof(float),cudaMemcpyDeviceToHost),
            "exactattn GQA-six score workspace");
        launch(true,gqasixextent,true,true,true,true,true,false,true,false,true,false,false,true,false,true);
        const auto six_extent=target_continue_device_bits(static_cast<const std::uint16_t*>(gqasixextent.get()),q.size(),"exactattn GQA-six extent-shards result");
        require(six_extent==six_scores,"exactattn GQA-six extent-shards bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqasixquerypair,true,true,true,true,true,false,true,false,true,false,false,false,false,true,true);
        const auto six_query_pair=target_continue_device_bits(static_cast<const std::uint16_t*>(gqasixquerypair.get()),q.size(),"exactattn GQA-six query-pair scores result");
        require(six_query_pair==six_scores,"exactattn GQA-six query-pair scores bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        std::vector<float> query_pair_score_workspace(six_score_workspace.size());
        cuda_check(cudaMemcpy(query_pair_score_workspace.data(),scores.get(),
            query_pair_score_workspace.size()*sizeof(float),cudaMemcpyDeviceToHost),
            "exactattn GQA-six query-pair score workspace");
        require(query_pair_score_workspace==six_score_workspace,
            "exactattn GQA-six query-pair score workspace mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqasixsharded,true,true,true,true,true,false,true,false,true,true);
        const auto six_sharded=target_continue_device_bits(static_cast<const std::uint16_t*>(gqasixsharded.get()),q.size(),"exactattn GQA-six sharded-values result");
        require(six_sharded==baseline,"exactattn GQA-six sharded-values bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqatriplevalues4,true,true,true,true,true,false,true,false,true,false,true);
        const auto triple_values4=target_continue_device_bits(static_cast<const std::uint16_t*>(gqatriplevalues4.get()),q.size(),"exactattn GQA-triple values4 result");
        require(triple_values4==baseline,"exactattn GQA-triple values4 bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        launch(true,gqasixsoftmaxtriple,true,true,true,true,true,false,true,false,true,false,false,true);
        const auto six_softmax_triple=target_continue_device_bits(static_cast<const std::uint16_t*>(gqasixsoftmaxtriple.get()),q.size(),"exactattn GQA-six softmax triple-values result");
        require(six_softmax_triple==baseline,"exactattn GQA-six softmax triple-values bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        if((rows==1||rows==3||rows==15||rows==16) &&
           (capacity==321||capacity==4096)) {
            constexpr std::uint16_t guard=0x6bd5;
            std::vector<std::uint16_t> initialized(q.size()+2,guard);
            DeviceBuffer pair_dimensions((q.size()+2)*sizeof(std::uint16_t));
            DeviceBuffer pair_dimensions_repeat((q.size()+2)*sizeof(std::uint16_t));
            for(auto* candidate:{&pair_dimensions,&pair_dimensions_repeat})
                cuda_check(cudaMemcpy(candidate->get(),initialized.data(),candidate->bytes(),
                    cudaMemcpyHostToDevice),"pair-dimensions output canary upload");
            const auto run_pair_dimensions=[&](DeviceBuffer& destination) {
                ninfer::exl3::exl3_exact_attention_for_test(
                    static_cast<const std::uint16_t*>(dq.get()),
                    static_cast<const std::uint16_t*>(dk.get()),
                    static_cast<const std::uint16_t*>(dv.get()),
                    static_cast<std::uint16_t*>(destination.get())+1,
                    static_cast<float*>(scores.get()),rows,position,capacity,true,nullptr,
                    true,true,true,true,true,false,true,false,true,false,false,true,
                    false,false,false,false,false,false,false,false,false,false,{},true);
            };
            run_pair_dimensions(pair_dimensions);
            run_pair_dimensions(pair_dimensions_repeat);
            cuda_check(cudaDeviceSynchronize(),"pair-dimensions deterministic synchronize");
            const auto paired=target_continue_device_bits(
                static_cast<const std::uint16_t*>(pair_dimensions.get()),q.size()+2,
                "pair-dimensions guarded result");
            const auto repeated=target_continue_device_bits(
                static_cast<const std::uint16_t*>(pair_dimensions_repeat.get()),q.size()+2,
                "pair-dimensions repeated guarded result");
            require(paired.front()==guard&&paired.back()==guard&&
                    repeated.front()==guard&&repeated.back()==guard,
                "pair-dimensions output canary changed capacity="+
                    std::to_string(capacity)+" rows="+std::to_string(rows));
            require(std::equal(paired.begin()+1,paired.end()-1,baseline.begin()),
                "pair-dimensions represented bits differ capacity="+
                    std::to_string(capacity)+" rows="+std::to_string(rows));
            require(paired==repeated,
                "pair-dimensions nondeterministic capacity="+
                    std::to_string(capacity)+" rows="+std::to_string(rows));
            const auto represented_v=target_continue_device_bits(
                static_cast<const std::uint16_t*>(dv.get()),v.size(),
                "pair-dimensions immutable future-poison V");
            require(represented_v==v,
                "pair-dimensions changed V/future poison capacity="+
                    std::to_string(capacity)+" rows="+std::to_string(rows));
        }
        launch(true,gqasixpackedtriples,true,true,true,true,true,false,true,false,true,false,false,false,true);
        const auto six_packed_triples=target_continue_device_bits(static_cast<const std::uint16_t*>(gqasixpackedtriples.get()),q.size(),"exactattn GQA-six packed-triples result");
        require(six_packed_triples==baseline,"exactattn GQA-six packed-triples bit mismatch capacity="+
            std::to_string(capacity)+" rows="+std::to_string(rows));
        if(capacity<=4096) {
            launch(false,ref);
            const auto original=target_continue_device_bits(static_cast<const std::uint16_t*>(ref.get()),q.size(),"exactattn original");
            require(actual==original,"exactattn original bit mismatch capacity="+std::to_string(capacity)+" rows="+std::to_string(rows));
        }
        // Naive mathematical oracle in FP64 from represented FP16 public inputs.
        // Does not reproduce production casts, FP32 sums or kernel organization.
        for(int row=0;row<rows;++row) for(int head=0;head<24;++head) {
            const int count=position+row+1,kv_head=head/6;
            std::vector<double> weights(count);
            double maximum=-std::numeric_limits<double>::infinity();
            for(int token=0;token<count;++token) {
                double dot=0;
                for(int d=0;d<256;++d) dot+=static_cast<double>(half_to_float(q[(row*24+head)*256+d]))*
                    half_to_float(k[(static_cast<std::size_t>(token)*4+kv_head)*256+d]);
                weights[token]=dot/16.0; maximum=std::max(maximum,weights[token]);
            }
            double denominator=0;
            for(auto& weight:weights) {weight=std::exp(weight-maximum);denominator+=weight;}
            for(int d=0;d<256;++d) {
                double expected=0;
                for(int token=0;token<count;++token) expected+=weights[token]/denominator*
                    half_to_float(v[(static_cast<std::size_t>(token)*4+kv_head)*256+d]);
                const double value=half_to_float(actual[(row*24+head)*256+d]);
                require(std::isfinite(value)&&std::abs(value-expected)<=0.001+0.002*std::abs(expected),
                        "exactattn FP64 mathematical oracle mismatch");
            }
        }
        ++cases;
    }
    int boundary_cases=0;
    for(const int key_count:std::array<int,18>{
        255,256,257,511,512,513,767,768,769,
        1023,1024,1025,1279,1280,1281,1535,1536,1537}) {
        constexpr std::uint16_t output_guard=0x6d3b;
        constexpr float score_guard=173.25f;
        const int rows=(key_count==513||key_count==1281)?16:1;
        const int position=key_count-rows,capacity=key_count;
        std::vector<std::uint16_t> q(static_cast<std::size_t>(rows)*24*256);
        std::vector<std::uint16_t> k(static_cast<std::size_t>(capacity)*4*256),v(k.size());
        const auto f16=[](float x){return __half_as_ushort(__float2half_rn(x));};
        for(std::size_t i=0;i<q.size();++i)
            q[i]=f16(static_cast<float>(0.31*std::sin(i*0.019)+0.07*std::cos(i*0.047)));
        for(std::size_t i=0;i<k.size();++i) {
            k[i]=f16(static_cast<float>(0.43*std::sin(i*0.037)+(i%991==0?0.9:0.0)));
            v[i]=f16(static_cast<float>(1.1*std::cos(i*0.023)-0.37*std::sin(i*0.071)));
        }
        const std::size_t output_elements=q.size(),score_elements=
            static_cast<std::size_t>(rows)*24*capacity;
        DeviceBuffer dq(q.size()*2),dk(k.size()*2),dv(v.size()*2);
        DeviceBuffer fixed_output((output_elements+2)*2),extent_output((output_elements+2)*2);
        DeviceBuffer fixed_scores((score_elements+2)*4),extent_scores((score_elements+2)*4);
        cuda_check(cudaMemcpy(dq.get(),q.data(),q.size()*2,cudaMemcpyHostToDevice),"extent boundary upload Q");
        cuda_check(cudaMemcpy(dk.get(),k.data(),k.size()*2,cudaMemcpyHostToDevice),"extent boundary upload K");
        cuda_check(cudaMemcpy(dv.get(),v.data(),v.size()*2,cudaMemcpyHostToDevice),"extent boundary upload V");
        std::vector<std::uint16_t> output_init(output_elements+2,output_guard);
        std::vector<float> score_init(score_elements+2,score_guard);
        for(auto* allocation:{&fixed_output,&extent_output})
            cuda_check(cudaMemcpy(allocation->get(),output_init.data(),allocation->bytes(),cudaMemcpyHostToDevice),"extent boundary output canary");
        for(auto* allocation:{&fixed_scores,&extent_scores})
            cuda_check(cudaMemcpy(allocation->get(),score_init.data(),allocation->bytes(),cudaMemcpyHostToDevice),"extent boundary score canary");
        const auto run=[&](DeviceBuffer& output,DeviceBuffer& scores,bool extent) {
            ninfer::exl3::exl3_exact_attention_for_test(
                static_cast<const std::uint16_t*>(dq.get()),
                static_cast<const std::uint16_t*>(dk.get()),
                static_cast<const std::uint16_t*>(dv.get()),
                static_cast<std::uint16_t*>(output.get())+1,
                static_cast<float*>(scores.get())+1,rows,position,capacity,true,
                nullptr,true,true,true,true,true,false,true,false,true,false,false,
                true,false,extent);
        };
        run(fixed_output,fixed_scores,false);run(extent_output,extent_scores,true);
        cuda_check(cudaDeviceSynchronize(),"extent boundary synchronize");
        std::vector<std::uint16_t> fixed_bits(output_elements+2),extent_bits(output_elements+2);
        std::vector<float> fixed_score_bits(score_elements+2),extent_score_bits(score_elements+2);
        cuda_check(cudaMemcpy(fixed_bits.data(),fixed_output.get(),fixed_output.bytes(),cudaMemcpyDeviceToHost),"extent boundary fixed output");
        cuda_check(cudaMemcpy(extent_bits.data(),extent_output.get(),extent_output.bytes(),cudaMemcpyDeviceToHost),"extent boundary extent output");
        cuda_check(cudaMemcpy(fixed_score_bits.data(),fixed_scores.get(),fixed_scores.bytes(),cudaMemcpyDeviceToHost),"extent boundary fixed scores");
        cuda_check(cudaMemcpy(extent_score_bits.data(),extent_scores.get(),extent_scores.bytes(),cudaMemcpyDeviceToHost),"extent boundary extent scores");
        require(fixed_bits.front()==output_guard&&fixed_bits.back()==output_guard&&
                extent_bits.front()==output_guard&&extent_bits.back()==output_guard,
                "extent boundary output guard key_count="+std::to_string(key_count));
        require(fixed_score_bits.front()==score_guard&&fixed_score_bits.back()==score_guard&&
                extent_score_bits.front()==score_guard&&extent_score_bits.back()==score_guard,
                "extent boundary score guard key_count="+std::to_string(key_count));
        require(std::equal(fixed_bits.begin()+1,fixed_bits.end()-1,extent_bits.begin()+1),
                "extent boundary output mismatch key_count="+std::to_string(key_count));
        require(std::equal(fixed_score_bits.begin()+1,fixed_score_bits.end()-1,extent_score_bits.begin()+1),
                "extent boundary workspace mismatch key_count="+std::to_string(key_count));
        std::vector<std::uint16_t> q_after(q.size()),k_after(k.size()),v_after(v.size());
        cuda_check(cudaMemcpy(q_after.data(),dq.get(),q.size()*2,cudaMemcpyDeviceToHost),"extent boundary Q immutability");
        cuda_check(cudaMemcpy(k_after.data(),dk.get(),k.size()*2,cudaMemcpyDeviceToHost),"extent boundary K immutability");
        cuda_check(cudaMemcpy(v_after.data(),dv.get(),v.size()*2,cudaMemcpyDeviceToHost),"extent boundary V immutability");
        require(q_after==q&&k_after==k&&v_after==v,
                "extent boundary input mutation key_count="+std::to_string(key_count));
        ++boundary_cases;
    }
    std::cout << "EXACT_ATTENTION_OPERATOR PASS cases=" << cases
              << " max_context=16384 q_shared_bit_exact=1 k_half2_bit_exact=1 v_half2_bit_exact=1 gqa_pair_bit_exact=1 gqa_triple_bit_exact=1 gqa_triple_values128_bit_exact=1 gqa_triple_softmax_staged_bit_exact=1 gqa_six_bit_exact=1 gqa_six_scores_bit_exact=1 gqa_six_query_pair_scores_bit_exact=1 gqa_six_values_sharded_bit_exact=1 gqa_triple_values4_bit_exact=1 gqa_six_softmax_triple_values_bit_exact=1 gqa_six_softmax_triple_pair_dimensions_bit_exact=1 gqa_six_packed_triples_bit_exact=1 oracle=FP64\n";
    std::cout << "EXACT_ATTENTION_EXTENT_SHARDS PASS boundary_cases=" << boundary_cases
              << " boundaries=256,512,768,1024,1280,1536 output_workspace_bit_exact=1 guards=1 inputs_immutable=1\n";
}

void run_exact_attention_extended_cap_oracle(int capacity) {
    require(capacity==32768||capacity==65536||capacity==131072,
        "extended exact-attention capacity must be 32768, 65536, or 131072");
    constexpr int rows=1;
    const int position=capacity-rows;
    constexpr std::uint16_t output_guard=0x5a6d;
    constexpr float score_guard=219.75f;
    const auto f16=[](float x){return __half_as_ushort(__float2half_rn(x));};
    std::vector<std::uint16_t> q(static_cast<std::size_t>(rows)*24*256);
    std::vector<std::uint16_t> k(static_cast<std::size_t>(capacity)*4*256),v(k.size());
    for(std::size_t i=0;i<q.size();++i)
        q[i]=f16(static_cast<float>(0.29*std::sin(i*0.013)+0.11*std::cos(i*0.041)));
    for(std::size_t i=0;i<k.size();++i) {
        k[i]=f16(static_cast<float>(0.39*std::sin(i*0.027)+(i%1009==0?0.95:0.0)));
        v[i]=f16(static_cast<float>(1.17*std::cos(i*0.021)-0.33*std::sin(i*0.067)));
    }
    const std::size_t output_elements=q.size();
    const std::size_t score_elements=static_cast<std::size_t>(rows)*24*capacity;
    DeviceBuffer dq(q.size()*2),dk(k.size()*2),dv(v.size()*2);
    DeviceBuffer baseline_output((output_elements+2)*2),fixed_output((output_elements+2)*2),
        extent_output((output_elements+2)*2),repeat_output((output_elements+2)*2);
    DeviceBuffer baseline_scores((score_elements+2)*4),fixed_scores((score_elements+2)*4),
        extent_scores((score_elements+2)*4),repeat_scores((score_elements+2)*4);
    cuda_check(cudaMemcpy(dq.get(),q.data(),q.size()*2,cudaMemcpyHostToDevice),"native-cap upload Q");
    cuda_check(cudaMemcpy(dk.get(),k.data(),k.size()*2,cudaMemcpyHostToDevice),"native-cap upload K");
    cuda_check(cudaMemcpy(dv.get(),v.data(),v.size()*2,cudaMemcpyHostToDevice),"native-cap upload V");
    std::vector<std::uint16_t> output_init(output_elements+2,output_guard);
    std::vector<float> score_init(score_elements+2,score_guard);
    for(auto* allocation:{&baseline_output,&fixed_output,&extent_output,&repeat_output})
        cuda_check(cudaMemcpy(allocation->get(),output_init.data(),allocation->bytes(),cudaMemcpyHostToDevice),
            "native-cap output canary");
    for(auto* allocation:{&baseline_scores,&fixed_scores,&extent_scores,&repeat_scores})
        cuda_check(cudaMemcpy(allocation->get(),score_init.data(),allocation->bytes(),cudaMemcpyHostToDevice),
            "native-cap score canary");
    const auto run=[&](DeviceBuffer& output,DeviceBuffer& scores,bool fixed_six,bool extent) {
        ninfer::exl3::exl3_exact_attention_for_test(
            static_cast<const std::uint16_t*>(dq.get()),
            static_cast<const std::uint16_t*>(dk.get()),
            static_cast<const std::uint16_t*>(dv.get()),
            static_cast<std::uint16_t*>(output.get())+1,
            static_cast<float*>(scores.get())+1,rows,position,capacity,true,nullptr,
            fixed_six, // Q-shared
            fixed_six, // K-half2
            fixed_six, // V-half2
            fixed_six, // GQA pair
            fixed_six, // GQA triple
            false,     // triple values128
            fixed_six, // staged softmax
            false,     // fused six-head route
            fixed_six, // fixed-six score route
            false,     // sharded values
            false,     // triple values4
            fixed_six, // triple values after six-score softmax
            false,     // packed triples
            extent);   // public-key extent-selected score shards
    };
    run(baseline_output,baseline_scores,false,false);
    run(fixed_output,fixed_scores,true,false);
    run(extent_output,extent_scores,true,true);
    run(repeat_output,repeat_scores,true,true);
    cuda_check(cudaDeviceSynchronize(),"native-cap synchronize");
    const auto download_output=[&](DeviceBuffer& source,const char* label) {
        std::vector<std::uint16_t> bits(output_elements+2);
        cuda_check(cudaMemcpy(bits.data(),source.get(),source.bytes(),cudaMemcpyDeviceToHost),label);
        require(bits.front()==output_guard&&bits.back()==output_guard,
            std::string(label)+" guard");
        return bits;
    };
    const auto download_scores=[&](DeviceBuffer& source,const char* label) {
        std::vector<float> bits(score_elements+2);
        cuda_check(cudaMemcpy(bits.data(),source.get(),source.bytes(),cudaMemcpyDeviceToHost),label);
        require(bits.front()==score_guard&&bits.back()==score_guard,
            std::string(label)+" guard");
        return bits;
    };
    const auto baseline_bits=download_output(baseline_output,"native-cap baseline output");
    const auto fixed_bits=download_output(fixed_output,"native-cap fixed output");
    const auto extent_bits=download_output(extent_output,"native-cap extent output");
    const auto repeat_bits=download_output(repeat_output,"native-cap repeat output");
    require(std::equal(baseline_bits.begin()+1,baseline_bits.end()-1,fixed_bits.begin()+1),
        "native-cap fixed output mismatch");
    require(std::equal(fixed_bits.begin()+1,fixed_bits.end()-1,extent_bits.begin()+1),
        "native-cap extent output mismatch");
    require(extent_bits==repeat_bits,"native-cap repeat output mismatch");
    const auto baseline_score_bits=download_scores(baseline_scores,"native-cap baseline scores");
    const auto fixed_score_bits=download_scores(fixed_scores,"native-cap fixed scores");
    const auto extent_score_bits=download_scores(extent_scores,"native-cap extent scores");
    const auto repeat_score_bits=download_scores(repeat_scores,"native-cap repeat scores");
    require(fixed_score_bits==extent_score_bits,"native-cap extent workspace mismatch");
    require(extent_score_bits==repeat_score_bits,"native-cap repeat workspace mismatch");
    require(baseline_score_bits.front()==score_guard&&baseline_score_bits.back()==score_guard,
        "native-cap baseline workspace guard");
    std::vector<std::uint16_t> q_after(q.size()),k_after(k.size()),v_after(v.size());
    cuda_check(cudaMemcpy(q_after.data(),dq.get(),q.size()*2,cudaMemcpyDeviceToHost),"native-cap Q immutability");
    cuda_check(cudaMemcpy(k_after.data(),dk.get(),k.size()*2,cudaMemcpyDeviceToHost),"native-cap K immutability");
    cuda_check(cudaMemcpy(v_after.data(),dv.get(),v.size()*2,cudaMemcpyDeviceToHost),"native-cap V immutability");
    require(q_after==q&&k_after==k&&v_after==v,"native-cap input mutation");
    double maximum_absolute_error=0.0;
    for(int head=0;head<24;++head) {
        const int kv_head=head/6;
        std::vector<double> weights(capacity);
        double maximum=-std::numeric_limits<double>::infinity();
        for(int token=0;token<capacity;++token) {
            double dot=0.0;
            for(int d=0;d<256;++d)
                dot+=static_cast<double>(half_to_float(q[head*256+d]))*
                    half_to_float(k[(static_cast<std::size_t>(token)*4+kv_head)*256+d]);
            weights[token]=dot/16.0;
            maximum=std::max(maximum,weights[token]);
        }
        double denominator=0.0;
        for(auto& weight:weights) {weight=std::exp(weight-maximum);denominator+=weight;}
        for(int d=0;d<256;++d) {
            double expected=0.0;
            for(int token=0;token<capacity;++token)
                expected+=weights[token]/denominator*
                    half_to_float(v[(static_cast<std::size_t>(token)*4+kv_head)*256+d]);
            const double value=half_to_float(extent_bits[1+head*256+d]);
            const double absolute_error=std::abs(value-expected);
            maximum_absolute_error=std::max(maximum_absolute_error,absolute_error);
            require(std::isfinite(value)&&absolute_error<=0.001+0.002*std::abs(expected),
                "native-cap FP64 mathematical oracle mismatch head="+std::to_string(head)+
                " dimension="+std::to_string(d));
        }
    }
    std::cout << "EXACT_ATTENTION_NATIVE_CAP PASS capacity=" << capacity
              << " rows=" << rows
              << " baseline_fixed_extent_bit_exact=1 workspace_bit_exact=1 deterministic=1 guards=1 inputs_immutable=1"
              << " fp64_max_abs_error=" << maximum_absolute_error << " oracle=FP64\n";
}

void run_extended_context_storage_contract(Exl3TextModel& target,int expected_context) {
    require((expected_context==65536||expected_context==131072)&&
            target.max_context()==expected_context,
            "extended-context contract target extent");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","0");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    bool rejected=false;
    try {
        auto invalid=target.create_context(false);
        (void)invalid;
    } catch(const std::exception& error) {
        rejected=std::string(error.what()).find("extended context requires exact-host KV or OSCAR-only storage")!=
            std::string::npos;
    }
    require(rejected,"extended-context resident GPU-KV route did not fail closed");
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","1");
    auto oscar_only=target.create_context(false);
    require(oscar_only->max_context()==expected_context,"extended-context OSCAR-only extent");
    const auto oscar_bytes=oscar_only->persistent_bytes();
    oscar_only.reset();
    _putenv_s("NINFER_EXL3_OSCAR_L0_ONLY","0");
    _putenv_s("NINFER_EXL3_EXACT_HOST_KV","1");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    auto exact_host=target.create_context(true);
    const auto stats=exact_host->host_kv_stats();
    require(exact_host->max_context()==expected_context&&stats.enabled&&
            stats.layer_workspace_bytes==static_cast<std::uint64_t>(expected_context)*4096,
            "extended-context exact-host storage contract");
    std::cout << "EXTENDED_CONTEXT_STORAGE PASS max_context=" << expected_context
              << " default_gpu_kv_rejected=1"
              << " oscar_only=1 exact_host=1 layer_workspace_bytes=" << stats.layer_workspace_bytes
              << " oscar_persistent_bytes=" << oscar_bytes
              << " exact_host_persistent_bytes=" << exact_host->persistent_bytes() << "\n";
}

void run_exact_attention_model_qualification(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    require(source.size()>=350 && target.max_context()==1024,"exactattn model fixture extent");
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","0");
    auto original=target.create_context(true);
    _putenv_s("NINFER_EXL3_EXACT_ATTENTION_PARALLEL","1");
    auto candidate=target.create_context(true);
    original->prepare_continuation(8);candidate->prepare_continuation(8);
    for(auto* context:{original.get(),candidate.get()}) context->prefill(std::span<const std::int64_t>(source.data(),16));
    int compared=0;
    for(int position=16;position<321;) {
        const int rows=std::min(8,321-position);
        for(auto* context:{original.get(),candidate.get()}) {
            if(rows==1) context->decode(source[position]);
            else {context->continue_rows(std::span<const std::int64_t>(source.data()+position,rows));context->finish_exact_continuation();}
        }
        position+=rows;
        if(position==24||position==64||position==320||position==321) {
            require(original->export_exact_host_state()->same_payload(*candidate->export_exact_host_state()),
                    "exactattn real full state mismatch position="+std::to_string(position));
            ++compared;
        }
    }
    const auto root=original->export_exact_host_state();
    for(int repeat=0;repeat<3;++repeat) {
        double times[2]{};
        for(int order=0;order<2;++order) {
            const int arm=(repeat&1)?1-order:order;
            auto& context=arm?*candidate:*original;
            context.restore_exact_host_state(*root);
            cuda_check(cudaDeviceSynchronize(),"exactattn timing start");
            const auto start=std::chrono::steady_clock::now();
            for(int i=321;i<329;++i) context.decode(source[i]);
            cuda_check(cudaDeviceSynchronize(),"exactattn timing end");
            times[arm]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        require(original->export_exact_host_state()->same_payload(*candidate->export_exact_host_state()),
                "exactattn paired decode full state");
        std::cout << "EXACT_ATTENTION_PAIR repeat=" << repeat << " rows=8 original_ms=" << times[0]
                  << " parallel_ms=" << times[1] << '\n';
    }
    std::cout << "EXACT_ATTENTION_MODEL PASS state_boundaries=" << compared
              << " pairs=3 score_workspace_bytes=" << 16ULL*24*1024*4 << '\n';
}
