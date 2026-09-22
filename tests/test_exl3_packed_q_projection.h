#pragma once
#include "exl3/packed_projection_dispatch.h"
#include "exl3/target_q_continuation.h"
#include "test_exl3_packed_projection_prefix.h"
#include "test_exl3_packed_down_projection.h"
#include "test_exl3_packed_head_projection.h"
#include "test_exl3_packed_kvo_projection.h"
#include "test_exl3_packed_gateup_projection.h"

struct PackedQCapture {
    int layer=-1;
    ninfer::exl3::Exl3CudaLinearWeights weights{};
    ninfer::exl3::Exl3CudaLinearMetadata metadata{};
    std::shared_ptr<DeviceBuffer> input;
    static void observe(const ninfer::exl3::Exl3TargetProjectionObservation& x,void* user) {
        auto& self=*static_cast<PackedQCapture*>(user);
        if(self.input || std::string(x.operation)!="q" || x.rows!=8 || x.metadata.K!=6) return;
        self.input=std::make_shared<DeviceBuffer>(8ULL*5120*2);
        cuda_check(cudaMemcpyAsync(self.input->get(),x.input,8ULL*5120*2,cudaMemcpyDeviceToDevice,x.stream),"packed Q capture");
        cuda_check(cudaStreamSynchronize(x.stream),"packed Q capture owner");
        self.weights=x.weights;self.metadata=x.metadata;self.layer=x.layer;
    }
};
void run_packed_q_projection(Exl3TextModel& target,const std::vector<std::int64_t>& source) {
    using namespace ninfer::exl3;
    if(env("NINFER_TEST_PACKED_DOWN")=="1")run_packed_down_projection(target,source);
    if(env("NINFER_TEST_PACKED_HEAD")=="1")run_packed_head_projection(target,source);
    if(env("NINFER_TEST_PACKED_KVO")=="1")run_packed_kvo_projection(target,source);
    if(env("NINFER_TEST_PACKED_GATEUP")=="1")run_packed_gateup_projection(target,source);
    // Independent expected routes, including the private draft O/down families.
    // This checks capability selection only; held arithmetic and Engine coverage
    // remain separate requirements and cannot be inferred from this table.
    using Family=Exl3TargetSharedFamily;
    using Admission=Exl3CudaLinearAdmission;
    struct CapabilityCase {Family family;Exl3CudaLinearMetadata metadata;Admission expected;};
    const std::array capability_cases{
        CapabilityCase{Family::q,{5120,12288,6,false,true,false},Admission::target_continuation_q},
        CapabilityCase{Family::k,{5120,1024,7,false,true,false},Admission::target_continuation_kv},
        CapabilityCase{Family::v,{5120,1024,8,false,true,false},Admission::target_continuation_kv},
        CapabilityCase{Family::o,{6144,5120,6,false,true,false},Admission::target_continuation_o},
        CapabilityCase{Family::gate,{5120,17408,5,false,true,false},Admission::target_continuation_gate_up},
        CapabilityCase{Family::up,{5120,17408,7,false,true,false},Admission::target_continuation_gate_up},
        CapabilityCase{Family::down,{17408,5120,7,false,true,false},Admission::target_continuation_down},
        CapabilityCase{Family::head,{5120,248320,6,false,true,false},Admission::target_continuation_head},
        CapabilityCase{Family::draft_q,{5120,4096,5,false,true,false},Admission::draft_shared_q_m16},
        CapabilityCase{Family::draft_k,{5120,1024,5,false,true,false},Admission::draft_shared_kv_m16},
        CapabilityCase{Family::draft_v,{5120,1024,5,false,true,false},Admission::draft_shared_kv_m16},
        CapabilityCase{Family::draft_o,{4096,5120,5,false,true,false},Admission::draft_shared_o_m16},
        CapabilityCase{Family::draft_down,{17408,5120,5,false,true,false},Admission::draft_shared_down_m16},
        CapabilityCase{Family::draft_gate,{5120,17408,5,false,true,false},Admission::draft_shared_gateup_m16},
        CapabilityCase{Family::draft_up,{5120,17408,5,false,true,false},Admission::draft_shared_gateup_m16}
    };
    static_assert(capability_cases.size()==static_cast<std::size_t>(Family::count),
        "shared-family capability fixture must cover every executable family");
    static_assert(exl3_target_shared_capabilities.size()==static_cast<std::size_t>(Family::count),
        "executable shared-family table must cover every family");
    std::array<unsigned,static_cast<std::size_t>(Family::count)> family_cases{};
    for(const auto& test:capability_cases) {
        const auto index=static_cast<std::size_t>(test.family);
        require(index<family_cases.size(),"capability fixture contains a non-executable family");
        ++family_cases[index];
        const auto* capability=target_shared_capability(test.family,test.metadata);
        const auto admitted=target_shared_admission(test.family,test.metadata);
        require(capability && admitted && capability==&exl3_target_shared_capabilities[index] &&
            capability->family==test.family && capability->admission==test.expected && *admitted==test.expected,
            "shared family selected wrong executable capability or arithmetic admission");
        const auto scope=capability->scope;
        if(scope==Exl3TargetSharedScope::target_layer)
            require(capability->supports_layer(0) && capability->supports_layer(63) &&
                !capability->supports_layer(-1) && !capability->supports_layer(64),
                "target capability accepted a non-target layer");
        else if(scope==Exl3TargetSharedScope::draft_layer)
            require(capability->supports_layer(0) && capability->supports_layer(4) &&
                !capability->supports_layer(-1) && !capability->supports_layer(5),
                "private-draft capability accepted a non-draft layer");
        else require(capability->supports_layer(64) && !capability->supports_layer(63) &&
                !capability->supports_layer(65),"output-head capability accepted a transformer layer");
        require(capability->arithmetic_intent==Exl3TargetSharedArithmeticIntent::exact_batch_invariant,
            "shared family lost explicit batch-invariant arithmetic intent");
        if(scope==Exl3TargetSharedScope::draft_layer)
            require(!capability->supports_combined_rows(15) && capability->supports_combined_rows(16) &&
                !capability->supports_combined_rows(17),"private-draft capability crossed fixed M16 topology");
        else require(!capability->supports_combined_rows(1) && capability->supports_combined_rows(2) &&
                capability->supports_combined_rows(16) && !capability->supports_combined_rows(17),
                "target capability crossed its exact small-M row menu");
        for(int mutation=0;mutation<9;++mutation) {
            auto changed=test.metadata;
            switch(mutation) {
                case 0:changed.mcg=true;break;
                case 1:changed.mul1=false;break;
                case 2:changed.has_bias=true;break;
                case 3:--changed.in_features;break;
                case 4:++changed.in_features;break;
                case 5:--changed.out_features;break;
                case 6:++changed.out_features;break;
                case 7:changed.K=4;break;
                case 8:changed.K=9;break;
            }
            require(!target_shared_admission(test.family,changed),"shared family accepted unsupported layout mutation");
        }
        require(!target_shared_admission(static_cast<Family>(-1),test.metadata) &&
            !target_shared_admission(Family::count,test.metadata) &&
            !target_shared_admission(static_cast<Family>(static_cast<unsigned>(Family::count)+1),test.metadata),
            "unknown shared family inherited valid geometry admission");
    }
    {
        auto classified=exl3_target_shared_capabilities[0];
        classified.arithmetic_intent=Exl3TargetSharedArithmeticIntent::canonical_independent_only;
        require(!classified.supports_combined_rows(16),
            "old canonical equivalence was promoted to shared batch invariance");
        classified.arithmetic_intent=Exl3TargetSharedArithmeticIntent::numeric_research;
        require(!classified.supports_combined_rows(16),
            "numeric research intent was promoted to exact shared execution");
    }
    for(std::size_t index=0;index<family_cases.size();++index) {
        require(family_cases[index]==1,"capability fixture duplicated or omitted a shared family");
        require(static_cast<std::size_t>(exl3_target_shared_capabilities[index].family)==index,
            "executable capability table order diverged from family accounting");
    }
    const Exl3CudaLinearMetadata common_gateup{5120,17408,5,false,true,false};
    for(const auto pair:{std::array{Family::gate,Family::draft_gate},std::array{Family::up,Family::draft_up}}) {
        const auto target_route=target_shared_admission(pair[0],common_gateup);
        const auto draft_route=target_shared_admission(pair[1],common_gateup);
        require(target_route==Admission::target_continuation_gate_up &&
            draft_route==Admission::draft_shared_gateup_m16 && target_route!=draft_route,
            "identical gate/up geometry erased target versus private-draft arithmetic admission");
    }
    for(int bits:{5,6,7,8}) {
        const Exl3CudaLinearMetadata draft_gateup{5120,17408,bits,false,true,false};
        for(auto family:{Family::draft_gate,Family::draft_up})
            require(bool(target_shared_admission(family,draft_gateup))==(bits==5),
                "draft gate/up accepted another family's quantization route");
        const Exl3CudaLinearMetadata q{5120,12288,bits,false,true,false};
        const Exl3CudaLinearMetadata kv{5120,1024,bits,false,true,false};
        for(auto family:{Exl3TargetSharedFamily::draft_k,Exl3TargetSharedFamily::draft_v}) {
            require(bool(target_shared_admission(family,kv))==(bits==5),"draft KV bitwidth admission");
            require(!target_shared_admission(family,q),"draft KV accepted Q shape");
        }
        const Exl3CudaLinearMetadata output{6144,5120,bits,false,true,false};
        const Exl3CudaLinearMetadata draft_output{4096,5120,bits,false,true,false};
        const Exl3CudaLinearMetadata draft_down{17408,5120,bits,false,true,false};
        require(bool(target_shared_admission(Exl3TargetSharedFamily::draft_o,draft_output))==(bits==5),
            "private draft O admitted target arithmetic width");
        require(bool(target_shared_admission(Exl3TargetSharedFamily::draft_down,draft_down))==(bits==5),
            "private draft down admitted target arithmetic width");
        require(!target_shared_admission(Exl3TargetSharedFamily::draft_o,output) &&
            !target_shared_admission(Exl3TargetSharedFamily::o,draft_output),
            "target and draft O interchanged attention output extents");
        const Exl3CudaLinearMetadata draft_q{5120,4096,bits,false,true,false};
        const Exl3CudaLinearMetadata mlp{5120,17408,bits,false,true,false};
        const Exl3CudaLinearMetadata head{5120,248320,bits,false,true,false};
        require(bool(target_shared_admission(Exl3TargetSharedFamily::head,head))==(bits==6),"shared head requires H6");
        auto short_head=head;--short_head.out_features;
        require(!target_shared_admission(Exl3TargetSharedFamily::head,short_head),"shared head accepted truncated vocabulary");
        auto biased_head=head;biased_head.has_bias=true;
        require(!target_shared_admission(Exl3TargetSharedFamily::head,biased_head),"shared head accepted biased layout");
        auto down=q;down.in_features=17408;down.out_features=5120;
        require(bool(target_shared_admission(Exl3TargetSharedFamily::down,down))==(bits==6 || bits==7),"shared down bitwidth table");
        require(!target_shared_admission(Exl3TargetSharedFamily::down,q),"down accepted normalized Q input geometry");
        auto down_bias=down;down_bias.has_bias=true;
        require(!target_shared_admission(Exl3TargetSharedFamily::down,down_bias),"down accepted bias");
        for(auto family:{Exl3TargetSharedFamily::gate,Exl3TargetSharedFamily::up}) {
            require(bool(target_shared_admission(family,mlp))==(bits==5 || bits==6 || bits==7),"shared gate/up bitwidth table");
            require(!target_shared_admission(family,q),"gate/up accepted Q destination geometry");
        }
        require(bool(target_shared_admission(Exl3TargetSharedFamily::o,output))==(bits==6 || bits==7),"shared O bitwidth table");
        require(bool(target_shared_admission(Exl3TargetSharedFamily::draft_q,draft_q))==(bits==5),"shared draft Q bitwidth table");
        require(bool(target_shared_admission(Exl3TargetSharedFamily::q,q))==(bits==6),"shared Q bitwidth table");
        for(auto family:{Exl3TargetSharedFamily::k,Exl3TargetSharedFamily::v}) {
            require(bool(target_shared_admission(family,kv))==(bits==7 || bits==8),"shared KV bitwidth table");
            require(!target_shared_admission(family,q),"shared family shape confusion");
            auto mixed=kv;mixed.mcg=true;require(!target_shared_admission(family,mixed),"shared mixed layout admitted");
            mixed=kv;mixed.has_bias=true;require(!target_shared_admission(family,mixed),"shared biased layout admitted");
        }
    }
    require(source.size()>=1040,"packed Q two real request fixture");
    std::array<PackedQCapture,2> captures;
    for(int lane=0;lane<2;++lane) {
        auto context=target.create_context(true);
        prefill_packed_projection_prefix(*context,
            std::span<const std::int64_t>(source.data()+lane*520,512));
        context->prepare_continuation(8);
        context->set_target_projection_observer_for_test(PackedQCapture::observe,&captures[lane],nullptr,
            Exl3TargetProjectionObserverSelection::wide_prefill_all);
        context->continue_rows(std::span<const std::int64_t>(source.data()+lane*520+512,8));
        context->set_target_projection_observer_for_test(nullptr);
        require(captures[lane].input!=nullptr,"packed Q missing real activation");
    }
    const auto& w=captures[0].weights;const auto& m=captures[0].metadata;
    require(captures[0].layer==captures[1].layer,"packed Q shared layer");
    require(w.trellis==captures[1].weights.trellis && w.suh==captures[1].weights.suh &&
        w.svh==captures[1].weights.svh && w.mul1==captures[1].weights.mul1,"packed Q shared weights");
    Exl3CudaLinearWorkspace control(5120,12288,8,false,false,false,false,false,false,false,{}, {},false,false,true);
    Exl3CudaLinearWorkspace combined(5120,12288,16,false,false,false,false,false,false,false,{}, {},false,false,true);
    combined.set_native_continuation16(true); // packed operator only, private L2 stays B8
    DeviceBuffer gather(16ULL*5120*2),scatter(16ULL*12288*2),reference(8ULL*12288*2);
    auto model=std::make_shared<int>(1);
    for(auto counts:packed_projection_row_pairs) for(bool reverse:{false,true}) {
        std::array<Exl3ProjectionRows,2> rows;
        for(int lane=0;lane<2;++lane) {
            auto output=std::make_shared<DeviceBuffer>(8ULL*12288*2);
            cuda_check(cudaMemset(output->get(),0x55,8ULL*12288*2),"packed Q output canary");
            rows[lane]={static_cast<std::uint64_t>(lane+1),1,1,model,std::make_shared<int>(lane),captures[lane].input,output,
                "text/fp16/layer3/q/K6",512+lane,counts[lane],5120,12288,5120,12288,
                static_cast<const std::uint16_t*>(captures[lane].input->get()),static_cast<std::uint16_t*>(output->get()),8ULL*5120,8ULL*12288};
        }
        if(reverse) std::swap(rows[0],rows[1]);
        const auto live=[](const auto&){return true;};
        auto plan=Exl3PackedProjectionPlan::assemble(rows,16,live);
        for(int missing=0;missing<4;++missing) {
            auto incomplete=w;
            if(missing==0)incomplete.trellis=nullptr;
            if(missing==1)incomplete.suh=nullptr;
            if(missing==2)incomplete.svh=nullptr;
            if(missing==3)incomplete.mul1=nullptr;
            int owner_visits=0;bool refused=false;
            try{dispatch_exl3_packed_q_projection(plan,combined,incomplete,m,
                static_cast<std::uint16_t*>(gather.get()),16ULL*5120,
                static_cast<std::uint16_t*>(scatter.get()),16ULL*12288,
                [&](const auto&){++owner_visits;return true;});}
            catch(const std::invalid_argument& e){refused=std::string(e.what())=="shared projection weight backing missing";}
            require(refused && owner_visits==0,"missing weights reached packed consumer admission");
        }
        const auto receipt=dispatch_exl3_packed_q_projection(plan,combined,w,m,
            static_cast<std::uint16_t*>(gather.get()),16ULL*5120,static_cast<std::uint16_t*>(scatter.get()),16ULL*12288,live);
        require(receipt.physical_batches==1 && receipt.rows==plan.rows(),"actual combined Q dispatch");
        for(const auto& lane:rows) {
            control.forward(w,m,lane.input,static_cast<std::uint16_t*>(reference.get()),lane.rows,
                nullptr,Exl3CudaLinearAdmission::target_continuation_q);
            std::vector<std::uint16_t> expected(lane.rows*12288),actual(expected.size());
            cuda_check(cudaMemcpy(expected.data(),reference.get(),expected.size()*2,cudaMemcpyDeviceToHost),"packed Q control");
            cuda_check(cudaMemcpy(actual.data(),lane.output,actual.size()*2,cudaMemcpyDeviceToHost),"packed Q actual");
            if(expected!=actual) {
                const auto mismatch=std::mismatch(expected.begin(),expected.end(),actual.begin());
                const auto index=static_cast<std::size_t>(mismatch.first-expected.begin());
                throw std::runtime_error("packed Q exact represented output lane="+
                    std::to_string(lane.request)+" rows="+std::to_string(lane.rows)+
                    " total="+std::to_string(plan.rows())+" reverse="+
                    std::to_string(reverse?1:0)+" first="+std::to_string(index)+
                    " expected="+std::to_string(*mismatch.first)+" actual="+
                    std::to_string(actual[index])+" control="+
                    control.dispatch_name(m,lane.rows,
                        Exl3CudaLinearAdmission::target_continuation_q)+" candidate="+
                    combined.dispatch_name(m,plan.rows(),
                        Exl3CudaLinearAdmission::target_continuation_q));
            }
            std::vector<std::uint16_t> guard((8-lane.rows)*12288);
            if(!guard.empty())cuda_check(cudaMemcpy(guard.data(),lane.output+lane.rows*12288,guard.size()*2,cudaMemcpyDeviceToHost),"packed Q tail canary");
            require(std::all_of(guard.begin(),guard.end(),[](auto h){return h==0x5555;}),"packed Q crossed private output span");
        }
        for(const auto& lane:rows) cuda_check(cudaMemset(lane.output,0x55,8ULL*12288*2),"packed Q canceled canary");
        int visits=0;bool canceled=false;
        try {dispatch_exl3_packed_q_projection(plan,combined,w,m,
            static_cast<std::uint16_t*>(gather.get()),16ULL*5120,static_cast<std::uint16_t*>(scatter.get()),16ULL*12288,
            [&](const auto&){return ++visits<=2;});}
        catch(const std::invalid_argument&){canceled=true;}
        require(canceled,"packed Q late cancellation was ignored");
        for(const auto& lane:rows) {
            std::vector<std::uint16_t> actual(8*12288);
            cuda_check(cudaMemcpy(actual.data(),lane.output,actual.size()*2,cudaMemcpyDeviceToHost),"packed Q canceled output");
            require(std::all_of(actual.begin(),actual.end(),[](auto h){return h==0x5555;}),"canceled Q scattered outputs");
        }
    }
}
