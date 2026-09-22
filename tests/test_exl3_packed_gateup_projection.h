#pragma once
#include "exl3/packed_projection_dispatch.h"
#include "exl3/target_q_continuation.h"
#include "test_exl3_packed_projection_prefix.h"

struct PackedGateupCapture {
    const char* operation="gate";
    int bits=5,layer=-1;
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    std::shared_ptr<DeviceBuffer> input;
    static void observe(const ninfer::exl3::Exl3TargetProjectionObservation& value,void* user) {
        auto& self=*static_cast<PackedGateupCapture*>(user);
        if(self.input || value.rows!=8 || value.metadata.K!=self.bits ||
            std::string(value.operation)!=self.operation || (self.layer>=0 && self.layer!=value.layer))return;
        self.input=std::make_shared<DeviceBuffer>(8ULL*value.metadata.in_features*2);
        cuda_check(cudaMemcpyAsync(self.input->get(),value.input,8ULL*value.metadata.in_features*2,
            cudaMemcpyDeviceToDevice,value.stream),"held gate/up capture");
        cuda_check(cudaStreamSynchronize(value.stream),"held gate/up capture lifetime");
        self.weights=value.weights;self.metadata=value.metadata;self.layer=value.layer;
    }
};

inline void run_packed_gateup_projection(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using namespace ninfer::exl3;
    constexpr auto admission=Exl3CudaLinearAdmission::target_continuation_gate_up;
    require(source.size()>=1040,"held gate/up requires two request inputs");
    for(const char* operation:{"gate","up"})for(int bits:{5,6,7}) {
        std::array<PackedGateupCapture,2> captures;
        for(int lane=0;lane<2;++lane) {
            auto& capture=captures[lane];capture.operation=operation;capture.bits=bits;
            if(lane)capture.layer=captures[0].layer;
            auto context=target.create_context(true);
            prefill_packed_projection_prefix(*context,{source.data()+lane*520,512});
            context->prepare_continuation(8);
            context->set_target_projection_observer_for_test(PackedGateupCapture::observe,&capture,nullptr,
                Exl3TargetProjectionObserverSelection::wide_prefill_all);
            context->continue_rows({source.data()+lane*520+512,8});
            context->set_target_projection_observer_for_test(nullptr);
            require(bool(capture.input),"held gate/up family/bitwidth NOT_EXERCISED");
        }
        const auto& weights=captures[0].weights;const auto& metadata=captures[0].metadata;
        const auto& peer=captures[1].weights;
        require(metadata.in_features==5120 && metadata.out_features==17408 &&
            weights.trellis==peer.trellis && weights.suh==peer.suh &&
            weights.svh==peer.svh && weights.mul1==peer.mul1,
            "held gate/up captures differ in shape or physical weight backing");
        Exl3CudaLinearWorkspace independent(5120,17408,8);
        Exl3CudaLinearWorkspace combined(5120,17408,16,false,true);
        combined.set_native_continuation16(true);
        require(exl3_packed_target_candidate(combined,metadata,16,admission),
            "held gate/up M16 candidate NOT_EXERCISED");
        DeviceBuffer gather(16ULL*5120*2),scatter(16ULL*17408*2),reference(8ULL*17408*2);
        auto model=std::make_shared<int>(1);
        for(auto counts:packed_projection_row_pairs)
        for(bool reverse:{false,true})for(bool padded:{false,true}) {
            const std::size_t stride=17408+(padded?16:0);
            std::array<Exl3ProjectionRows,2> lanes;
            for(int lane=0;lane<2;++lane) {
                auto destination=std::make_shared<DeviceBuffer>(8ULL*stride*2);
                cuda_check(cudaMemset(destination->get(),0x55,8ULL*stride*2),"held gate/up guards");
                lanes[lane]={static_cast<std::uint64_t>(lane+1),1,1,model,std::make_shared<int>(lane),
                    captures[lane].input,destination,"held/gate-up",512+lane,counts[lane],5120,17408,
                    5120,stride,static_cast<const std::uint16_t*>(captures[lane].input->get()),
                    static_cast<std::uint16_t*>(destination->get()),8ULL*5120,8ULL*stride};
            }
            if(reverse)std::swap(lanes[0],lanes[1]);
            const auto live=[](const auto&){return true;};
            auto plan=Exl3PackedProjectionPlan::assemble(lanes,16,live);
            std::vector<std::uint16_t> expected_transforms(plan.rows()*5120);
            std::array<std::vector<std::uint16_t>,2> expected_outputs;
            std::size_t packed_row=0;
            for(std::size_t lane=0;lane<lanes.size();++lane) {
                const auto& row=lanes[lane];
                independent.forward(weights,metadata,row.input,static_cast<std::uint16_t*>(reference.get()),row.rows);
                expected_outputs[lane].resize(static_cast<std::size_t>(row.rows)*17408);
                cuda_check(cudaMemcpy(expected_outputs[lane].data(),reference.get(),
                    expected_outputs[lane].size()*2,cudaMemcpyDeviceToHost),"held gate/up independent output");
                cuda_check(cudaMemcpy(expected_transforms.data()+packed_row*5120,independent.transformed_device(),
                    static_cast<std::size_t>(row.rows)*5120*2,cudaMemcpyDeviceToHost),
                    "held gate/up independent transform");
                packed_row+=row.rows;
            }
            const auto receipt=dispatch_exl3_packed_target_projection(plan,combined,weights,metadata,
                static_cast<std::uint16_t*>(gather.get()),16ULL*5120,
                static_cast<std::uint16_t*>(scatter.get()),16ULL*17408,live,nullptr,admission);
            require(receipt.physical_batches==1 && receipt.rows==plan.rows(),"held gate/up physical row accounting");
            std::vector<std::uint16_t> actual_transforms(expected_transforms.size());
            cuda_check(cudaMemcpy(actual_transforms.data(),combined.transformed_device(),
                actual_transforms.size()*2,cudaMemcpyDeviceToHost),"held gate/up packed transform");
            require(actual_transforms==expected_transforms,
                "held gate/up M16 changed canonical per-request input transforms");
            for(std::size_t lane=0;lane<lanes.size();++lane) {
                const auto& row=lanes[lane];std::vector<std::uint16_t> actual(8*stride);
                cuda_check(cudaMemcpy(actual.data(),row.output,actual.size()*2,cudaMemcpyDeviceToHost),
                    "held gate/up private output");
                for(int index=0;index<8;++index) {
                    const auto begin=actual.begin()+index*stride;
                    if(index<row.rows)require(std::equal(expected_outputs[lane].begin()+index*17408,
                        expected_outputs[lane].begin()+(index+1)*17408,begin),
                        "held gate/up represented output mismatch");
                    const auto guard=begin+(index<row.rows?17408:0);
                    require(std::all_of(guard,begin+stride,[](auto value){return value==0x5555;}),
                        "held gate/up crossed private row extent");
                }
            }
            for(const auto& row:lanes)cuda_check(cudaMemset(row.output,0x55,8*stride*2),
                "held gate/up cancellation guards");
            int visits=0;bool cancelled=false;
            try {dispatch_exl3_packed_target_projection(plan,combined,weights,metadata,
                static_cast<std::uint16_t*>(gather.get()),16ULL*5120,
                static_cast<std::uint16_t*>(scatter.get()),16ULL*17408,
                [&](const auto&){return ++visits<=2;},nullptr,admission);}
            catch(const std::invalid_argument&){cancelled=true;}
            require(cancelled && visits>2,"held gate/up late cancellation boundary not reached");
            for(const auto& row:lanes) {
                std::vector<std::uint16_t> untouched(8*stride);
                cuda_check(cudaMemcpy(untouched.data(),row.output,untouched.size()*2,cudaMemcpyDeviceToHost),
                    "held gate/up cancelled destination");
                require(std::all_of(untouched.begin(),untouched.end(),[](auto value){return value==0x5555;}),
                    "held gate/up cancellation scattered into private destination");
            }
        }
    }
}
