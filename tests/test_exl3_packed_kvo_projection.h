#pragma once
#include "exl3/packed_projection_dispatch.h"
#include "exl3/target_q_continuation.h"
#include "test_exl3_packed_projection_prefix.h"

struct PackedKVOCapture {
    const char* operation="k";
    int bits=7,layer=-1;
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    std::shared_ptr<DeviceBuffer> input;
    static void observe(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        auto& self=*static_cast<PackedKVOCapture*>(user);
        if(self.input || x.rows!=8 || x.metadata.K!=self.bits ||
           std::string(x.operation)!=self.operation || (self.layer>=0 && self.layer!=x.layer))return;
        self.input=std::make_shared<DeviceBuffer>(8ULL*x.metadata.in_features*2);
        cuda_check(cudaMemcpyAsync(self.input->get(),x.input,8ULL*x.metadata.in_features*2,
            cudaMemcpyDeviceToDevice,x.stream),"held KVO capture");
        cuda_check(cudaStreamSynchronize(x.stream),"held KVO capture lifetime");
        self.weights=x.weights;self.metadata=x.metadata;self.layer=x.layer;
    }
};

inline void run_packed_kvo_projection(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using namespace ninfer::exl3;
    require(source.size()>=1040,"held KVO requires two request inputs");
    for(const char* operation:{"k","v","o"})for(int variant=0;variant<2;++variant) {
        const bool output=std::string(operation)=="o";
        const int bits=(output?6:7)+variant;
        std::array<PackedKVOCapture,2> captures;
        for(int lane=0;lane<2;++lane) {
            auto& capture=captures[lane];capture.operation=operation;capture.bits=bits;
            if(lane)capture.layer=captures[0].layer;
            auto context=target.create_context(true);
            prefill_packed_projection_prefix(*context,{source.data()+lane*520,512});
            context->prepare_continuation(8);
            context->set_target_projection_observer_for_test(PackedKVOCapture::observe,&capture,nullptr,
                Exl3TargetProjectionObserverSelection::wide_prefill_all);
            context->continue_rows({source.data()+lane*520+512,8});
            context->set_target_projection_observer_for_test(nullptr);
            require(bool(capture.input),"held KVO family/bitwidth NOT_EXERCISED");
        }
        const auto& w=captures[0].weights;const auto& m=captures[0].metadata;
        const auto& other=captures[1].weights;
        require(w.trellis==other.trellis && w.suh==other.suh && w.svh==other.svh && w.mul1==other.mul1,
            "held KVO captures differ in actual weight backing");
        Exl3CudaLinearWorkspace control(m.in_features,m.out_features,8,false,false,false,output,
            false,false,false,{}, {},false,!output);
        Exl3CudaLinearWorkspace combined(m.in_features,m.out_features,16,false,false,false,output,
            false,false,false,{}, {},false,!output);
        combined.set_native_continuation16(true);
        const auto admission=output?Exl3CudaLinearAdmission::target_continuation_o:Exl3CudaLinearAdmission::target_continuation_kv;
        require(exl3_packed_target_candidate(combined,m,16,admission),"held KVO candidate NOT_EXERCISED");
        DeviceBuffer gather(16ULL*m.in_features*2),scatter(16ULL*m.out_features*2),reference(8ULL*m.out_features*2);
        auto model=std::make_shared<int>(1);
        for(auto counts:packed_projection_row_pairs)for(bool reverse:{false,true})
        for(bool padded:{false,true}) {
            const std::size_t stride=static_cast<std::size_t>(m.out_features)+(padded?16:0);
            std::array<Exl3ProjectionRows,2> lanes;
            for(int lane=0;lane<2;++lane) {
                auto destination=std::make_shared<DeviceBuffer>(8ULL*stride*2);
                cuda_check(cudaMemset(destination->get(),0x55,8ULL*stride*2),"held KVO guards");
                lanes[lane]={static_cast<std::uint64_t>(lane+1),1,1,model,std::make_shared<int>(lane),captures[lane].input,destination,
                    "held/KVO",512+lane,counts[lane],m.in_features,m.out_features,
                    static_cast<std::size_t>(m.in_features),stride,
                    static_cast<const std::uint16_t*>(captures[lane].input->get()),static_cast<std::uint16_t*>(destination->get()),
                    8ULL*m.in_features,8ULL*stride};
            }
            if(reverse)std::swap(lanes[0],lanes[1]);
            const auto live=[](const auto&){return true;};
            auto plan=Exl3PackedProjectionPlan::assemble(lanes,16,live);
            const auto receipt=dispatch_exl3_packed_target_projection(plan,combined,w,m,
                static_cast<std::uint16_t*>(gather.get()),16ULL*m.in_features,
                static_cast<std::uint16_t*>(scatter.get()),16ULL*m.out_features,live,nullptr,admission);
            require(receipt.physical_batches==1 && receipt.rows==plan.rows(),"held KVO physical row accounting");
            for(const auto& lane:lanes) {
                control.forward(w,m,lane.input,static_cast<std::uint16_t*>(reference.get()),lane.rows,nullptr,admission);
                std::vector<std::uint16_t> expected(lane.rows*m.out_features),actual(8*stride);
                cuda_check(cudaMemcpy(expected.data(),reference.get(),expected.size()*2,cudaMemcpyDeviceToHost),"held KVO reference");
                cuda_check(cudaMemcpy(actual.data(),lane.output,actual.size()*2,cudaMemcpyDeviceToHost),"held KVO output");
                for(int row=0;row<8;++row) {
                    const auto begin=actual.begin()+row*stride;
                    if(row<lane.rows)require(std::equal(expected.begin()+row*m.out_features,
                        expected.begin()+(row+1)*m.out_features,begin),"held KVO represented output mismatch");
                    const auto guard=begin+(row<lane.rows?m.out_features:0);
                    require(std::all_of(guard,begin+stride,[](auto value){return value==0x5555;}),
                        "held KVO crossed row padding or private valid extent");
                }
            }
            for(const auto& lane:lanes)cuda_check(cudaMemset(lane.output,0x55,8*stride*2),"held KVO cancellation guards");
            int visits=0;bool cancelled=false;
            try {
                dispatch_exl3_packed_target_projection(plan,combined,w,m,
                    static_cast<std::uint16_t*>(gather.get()),16ULL*m.in_features,
                    static_cast<std::uint16_t*>(scatter.get()),16ULL*m.out_features,
                    [&](const auto&){return ++visits<=2;},nullptr,admission);
            }catch(const std::invalid_argument&){cancelled=true;}
            require(cancelled && visits>2,"held KVO late cancellation boundary not reached");
            for(const auto& lane:lanes) {
                std::vector<std::uint16_t> untouched(8*stride);
                cuda_check(cudaMemcpy(untouched.data(),lane.output,untouched.size()*2,cudaMemcpyDeviceToHost),
                    "held KVO cancelled destination");
                require(std::all_of(untouched.begin(),untouched.end(),[](auto value){return value==0x5555;}),
                    "held KVO cancellation scattered into private destination");
            }
        }
    }
}
