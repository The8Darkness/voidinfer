#pragma once
#include "exl3/packed_projection_dispatch.h"
#include "test_exl3_packed_projection_prefix.h"

// Held post-SiLU inputs from two real requests. This is operator test source,
// separate from the Engine terminal-root test; neither has been executed.
struct PackedDownCapture {
    int bits=6,layer=-1;
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    std::shared_ptr<DeviceBuffer> input;
    static void observe(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        auto& self=*static_cast<PackedDownCapture*>(user);
        if(self.input || x.rows!=8 || x.metadata.K!=self.bits ||
           x.metadata.in_features!=17408 || x.metadata.out_features!=5120 ||
           (self.layer>=0 && x.layer!=self.layer))return;
        self.input=std::make_shared<DeviceBuffer>(8ULL*17408*2);
        cuda_check(cudaMemcpyAsync(self.input->get(),x.input,8ULL*17408*2,
            cudaMemcpyDeviceToDevice,x.stream),"capture private post-SiLU input");
        cuda_check(cudaStreamSynchronize(x.stream),"retain held down input");
        self.weights=x.weights;self.metadata=x.metadata;self.layer=x.layer;
    }
};
inline void run_packed_down_projection(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using namespace ninfer::exl3;
    require(source.size()>=1040,"held down needs two actual requests");
    for(int bits:{6,7}) {
        std::array<PackedDownCapture,2> captures;
        for(int lane=0;lane<2;++lane) {
            captures[lane].bits=bits;
            if(lane)captures[lane].layer=captures[0].layer;
            auto context=target.create_context(true);
            prefill_packed_projection_prefix(*context,
                std::span<const std::int64_t>(source.data()+lane*520,512));
            context->prepare_continuation(8);
            context->set_target_projection_observer_for_test(PackedDownCapture::observe,&captures[lane],nullptr,
                bits==6?Exl3TargetProjectionObserverSelection::down_k6:Exl3TargetProjectionObserverSelection::down_k7);
            context->continue_rows(std::span<const std::int64_t>(source.data()+lane*520+512,8));
            context->set_target_projection_observer_for_test(nullptr);
            require(bool(captures[lane].input),"held down bitwidth NOT_EXERCISED");
        }
        const auto& w=captures[0].weights;const auto& m=captures[0].metadata;
        const auto& other=captures[1].weights;
        require(w.trellis==other.trellis && w.suh==other.suh && w.svh==other.svh &&
            w.mul1==other.mul1,"held down requests must use the same actual layer weights");
        // Caller selects identical topology/splits/async-A for the two routes.
        Exl3CudaLinearWorkspace control(17408,5120,8,false,false,true);
        Exl3CudaLinearWorkspace combined(17408,5120,16,false,false,true);
        combined.set_native_continuation16(true);
        const auto admission=Exl3CudaLinearAdmission::target_continuation_down;
        require(exl3_packed_target_candidate(combined,m,16,admission),"held down route NOT_EXERCISED");
        require(!exl3_packed_target_candidate(combined,m,17,admission),"down crossed physical row capacity");
        require(!exl3_packed_target_candidate(combined,m,16,Exl3CudaLinearAdmission::target_continuation_q),
            "down accepted Q arithmetic admission");
        DeviceBuffer gather(16ULL*17408*2),scatter(16ULL*5120*2),reference(8ULL*5120*2);
        auto model=std::make_shared<int>(1);
        for(auto counts:packed_projection_row_pairs)
        for(bool reverse:{false,true}) for(bool padded:{false,true}) {
            const std::size_t output_stride=padded?5136:5120;
            std::array<Exl3ProjectionRows,2> lanes;
            for(int lane=0;lane<2;++lane) {
                auto output=std::make_shared<DeviceBuffer>(8ULL*output_stride*2);
                cuda_check(cudaMemset(output->get(),0x55,8ULL*output_stride*2),"down output guard");
                lanes[lane]={static_cast<std::uint64_t>(lane+1),1,1,model,std::make_shared<int>(lane),
                    captures[lane].input,output,"text/fp16/held-down",512+lane,counts[lane],17408,5120,17408,output_stride,
                    static_cast<const std::uint16_t*>(captures[lane].input->get()),
                    static_cast<std::uint16_t*>(output->get()),8ULL*17408,8ULL*output_stride};
            }
            if(reverse)std::swap(lanes[0],lanes[1]);
            const auto live=[](const auto&){return true;};
            auto plan=Exl3PackedProjectionPlan::assemble(lanes,16,live);
            bool capacity_refused=false;
            try {
                dispatch_exl3_packed_target_projection(plan,combined,w,m,
                    static_cast<std::uint16_t*>(gather.get()),plan.rows()*17408ULL-1,
                    static_cast<std::uint16_t*>(scatter.get()),16ULL*5120,live,nullptr,admission);
            } catch(const std::invalid_argument&){capacity_refused=true;}
            require(capacity_refused,"down accepted undersized packed input");
            for(const auto& lane:lanes) {
                std::vector<std::uint16_t> untouched(8*output_stride);
                cuda_check(cudaMemcpy(untouched.data(),lane.output,untouched.size()*2,cudaMemcpyDeviceToHost),"refused down destinations");
                require(std::all_of(untouched.begin(),untouched.end(),[](auto value){return value==0x5555;}),
                    "capacity refusal touched down destination");
            }
            auto receipt=dispatch_exl3_packed_target_projection(plan,combined,w,m,
                static_cast<std::uint16_t*>(gather.get()),16ULL*17408,
                static_cast<std::uint16_t*>(scatter.get()),16ULL*5120,live,nullptr,admission);
            require(receipt.physical_batches==1 && receipt.rows==plan.rows(),"held down physical coverage");
            for(const auto& lane:lanes) {
                // Independent old per-row controls: no packed offset in oracle.
                for(int row=0;row<lane.rows;++row)
                    control.forward(w,m,lane.input+row*17408,
                        static_cast<std::uint16_t*>(reference.get())+row*5120,1);
                std::vector<std::uint16_t> expected(lane.rows*5120),actual(8*output_stride);
                cuda_check(cudaMemcpy(expected.data(),reference.get(),expected.size()*2,cudaMemcpyDeviceToHost),"held down control");
                cuda_check(cudaMemcpy(actual.data(),lane.output,actual.size()*2,cudaMemcpyDeviceToHost),"held down scatter");
                for(int row=0;row<8;++row) {
                    const auto first=actual.begin()+row*output_stride;
                    if(row<lane.rows)require(std::equal(expected.begin()+row*5120,expected.begin()+(row+1)*5120,first),
                        "down reduction/cast or private row placement parity");
                    const auto guard=first+(row<lane.rows?5120:0);
                    require(std::all_of(guard,first+output_stride,[](auto value){return value==0x5555;}),
                        "down scatter crossed row padding/private tail");
                }
            }
        }
    }
}
