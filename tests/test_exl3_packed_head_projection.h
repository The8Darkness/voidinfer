#pragma once
#include "exl3/packed_projection_dispatch.h"
#include "exl3/target_q_continuation.h"
#include "test_exl3_packed_projection_prefix.h"

inline void run_packed_head_projection(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using namespace ninfer::exl3;
    constexpr int hidden=5120,vocab=248320;
    require(source.size()>=1040,"held head needs two real requests");
    struct Capture {std::shared_ptr<DeviceBuffer> input;Exl3CudaLinearWeights weights{};Exl3CudaLinearMetadata metadata{};};
    std::array<Capture,2> captures;
    const bool observe_families=env("NINFER_TEST_HELD_TARGET_FAMILY_OFFERS")=="1";
    for(int lane=0;lane<2;++lane) {
        std::uint32_t offered_families=0;
        auto context=target.create_context(true);
        prefill_packed_projection_prefix(*context,
            std::span<const std::int64_t>(source.data()+lane*520,512));
        context->prepare_continuation(8);
        context->set_target_q_executor([&](const Exl3TargetQContinuation& projection) {
            const auto family=static_cast<unsigned>(projection.family);
            require(family<static_cast<unsigned>(Exl3TargetSharedFamily::count),"held target unknown family offer");
            offered_families|=std::uint32_t{1}<<family;
            if(projection.family!=Exl3TargetSharedFamily::head)return false;
            require(projection.layer==64 && projection.rows==8,"held head actual boundary");
            auto& capture=captures[lane];
            require(!capture.input,"held head captured twice");
            capture.input=std::make_shared<DeviceBuffer>(8ULL*hidden*2);
            cuda_check(cudaMemcpyAsync(capture.input->get(),projection.input,8ULL*hidden*2,
                cudaMemcpyDeviceToDevice,projection.stream),"capture actual final normalized rows");
            cuda_check(cudaStreamSynchronize(projection.stream),"retain held head rows");
            capture.weights=projection.weights;capture.metadata=projection.metadata;
            return false; // actual context completes its independent head and root-logit copy
        },observe_families,observe_families,observe_families,observe_families,true);
        context->continue_rows(std::span<const std::int64_t>(source.data()+lane*520+512,8));
        context->set_target_q_executor({});
        require(bool(captures[lane].input),"held head NOT_EXERCISED");
        if(observe_families)for(const auto family:{Exl3TargetSharedFamily::q,Exl3TargetSharedFamily::k,
            Exl3TargetSharedFamily::v,Exl3TargetSharedFamily::o,Exl3TargetSharedFamily::gate,
            Exl3TargetSharedFamily::up,Exl3TargetSharedFamily::down,Exl3TargetSharedFamily::head})
            require((offered_families&(std::uint32_t{1}<<static_cast<unsigned>(family)))!=0,
                "held real target projection family NOT_EXERCISED");
    }
    const auto& w=captures[0].weights;const auto& m=captures[0].metadata;
    const auto& other=captures[1].weights;
    require(w.trellis==other.trellis && w.suh==other.suh && w.svh==other.svh && w.mul1==other.mul1,
        "held head target backing mismatch");
    Exl3CudaLinearWorkspace control(hidden,vocab,8),combined(hidden,vocab,16);
    combined.set_native_continuation16(true);
    const auto admission=Exl3CudaLinearAdmission::target_continuation_head;
    require(exl3_packed_target_candidate(combined,m,16,admission),"held head M16 NOT_EXERCISED");
    require(!exl3_packed_target_candidate(control,m,16,admission),"head exceeded owned row extent");
    require(!exl3_packed_target_candidate(combined,m,17,admission),"head crossed native tile");
    DeviceBuffer gather(16ULL*hidden*2),scatter(16ULL*vocab*2),reference(8ULL*vocab*2);
    auto model=std::make_shared<int>(1);
    for(auto counts:packed_projection_row_pairs)
    for(bool reverse:{false,true}) {
        std::array<Exl3ProjectionRows,2> lanes;
        for(int lane=0;lane<2;++lane) {
            auto output=std::make_shared<DeviceBuffer>(8ULL*vocab*2);
            cuda_check(cudaMemset(output->get(),0x55,8ULL*vocab*2),"head private tail guard");
            lanes[lane]={static_cast<std::uint64_t>(lane+1),1,1,model,std::make_shared<int>(lane),captures[lane].input,
                output,"text/fp16/H6/head",512+lane,counts[lane],hidden,vocab,hidden,vocab,
                static_cast<const std::uint16_t*>(captures[lane].input->get()),
                static_cast<std::uint16_t*>(output->get()),8ULL*hidden,8ULL*vocab};
        }
        if(reverse)std::swap(lanes[0],lanes[1]);
        const auto live=[](const auto&){return true;};
        auto plan=Exl3PackedProjectionPlan::assemble(lanes,16,live);
        const auto receipt=dispatch_exl3_packed_target_projection(plan,combined,w,m,
            static_cast<std::uint16_t*>(gather.get()),16ULL*hidden,
            static_cast<std::uint16_t*>(scatter.get()),16ULL*vocab,live,nullptr,admission);
        require(receipt.physical_batches==1 && receipt.rows==plan.rows(),"held head shared coverage");
        for(const auto& lane:lanes) {
            for(int row=0;row<lane.rows;++row)
                control.forward(w,m,lane.input+row*hidden,static_cast<std::uint16_t*>(reference.get())+row*vocab,1);
            std::vector<std::uint16_t> expected(lane.rows*vocab),actual(8*vocab);
            cuda_check(cudaMemcpy(expected.data(),reference.get(),expected.size()*2,cudaMemcpyDeviceToHost),"held head independent logits");
            cuda_check(cudaMemcpy(actual.data(),lane.output,actual.size()*2,cudaMemcpyDeviceToHost),"held head private logits");
            require(std::equal(expected.begin(),expected.end(),actual.begin()),"H6 full-vocabulary represented parity");
            require(std::all_of(actual.begin()+expected.size(),actual.end(),[](auto value){return value==0x5555;}),
                "head scattered beyond private requested rows");
        }
        for(const auto& lane:lanes)cuda_check(cudaMemset(lane.output,0x55,8ULL*vocab*2),"canceled head guards");
        int visits=0;bool canceled=false;
        try {
            dispatch_exl3_packed_target_projection(plan,combined,w,m,
                static_cast<std::uint16_t*>(gather.get()),16ULL*hidden,
                static_cast<std::uint16_t*>(scatter.get()),16ULL*vocab,
                [&](const auto&){return ++visits<=2;},nullptr,admission);
        } catch(const std::invalid_argument&){canceled=true;}
        require(canceled,"head canceled before scatter was accepted");
        for(const auto& lane:lanes) {
            std::vector<std::uint16_t> untouched(8*vocab);
            cuda_check(cudaMemcpy(untouched.data(),lane.output,untouched.size()*2,cudaMemcpyDeviceToHost),"canceled private head logits");
            require(std::all_of(untouched.begin(),untouched.end(),[](auto value){return value==0x5555;}),
                "canceled head wrote request logits");
        }
    }
}
