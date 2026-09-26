#include <ninfer/engine.h>
#include "core/decode_graph.h"
#include "exl3/graph_capture_requirements.h"
#include "exl3/projection_graph_binding.h"
#include "exl3/bounded_graph_entry.h"
#include "exl3/device_graph_role_table.h"
#include "exl3/control_publication_boundary.h"
#include "exl3/host_preparation_completion.h"
#include "exl3/execution_dependency_graph.h"
#include "exl3/graph_numerical_boundary.h"
#include "exl3/export_copy_plan.h"
#include "exl3/reconstruction_control_allocator.h"
#include "exl3/layer_buffer_retirement.h"
#include "exl3/gdn_layer.h"
#include "exl3/host_resident_set.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "exl3/recurrent_export_pool.h"
#include "exl3/exl3_engine_core.h"
#include "exl3/engine_target_q.h"
#include "exl3/dflash2_execution.h"
#include "exl3/exl3_frontend_resources.h"
#include "exl3/text_model.h"
#include "exl3/full_attention_layer.h"
#include "exl3/native_context_extent.h"
#include "exl3/prefill_attention_chain_graph.h"
#include "exl3/packed_cost_policy.h"
#include "exl3/host_resident_set.h"
#include "exl3/device_prefix_cache.h"
#include "exl3/vericache_request.h"
#include "test_exl3_registered_kv_preflight.h"
#include "test_exl3_future_context_matrix.h"
#include "test_exl3_device_logical_lease.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>

using namespace ninfer;
static void need(bool v,const char* msg){if(!v)throw std::runtime_error(msg);}
static PromptInput prompt(const std::string& text,bool thinking=false){
    PromptInput p;p.options.enable_thinking=thinking;
    p.messages.push_back({ChatRole::User,{{MessagePartKind::Text,text}}});return p;
}
struct CancellingSink : OutputSink {
    std::atomic<bool>& cancel;std::string text;
    explicit CancellingSink(std::atomic<bool>& c):cancel(c){}
    void publish(OutputDelta d)override{text+=d.text;cancel=true;}
};
int main(){try{
    if(const auto* mode=std::getenv("NINFER_TEST_DEVICE_LOGICAL_HOST");
       mode && std::string_view(mode)=="1") {
        run_device_logical_invalid_handle_host_only();
        std::cout<<"DEVICE_LOGICAL_HOST_COMPLETE device_execution=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_NUMERICAL_BOUNDARY");
       mode && std::string_view(mode)=="1") {
        using Boundary=ninfer::exl3::Exl3GraphNumericalBoundary;
        int input=0,output=0,stream=0;
        auto context=std::make_shared<int>(1),in_owner=std::make_shared<int>(2);
        auto out_owner=std::make_shared<int>(3),stream_owner=std::make_shared<int>(4);
        const auto make_with_stream=[&](std::uint32_t captured,std::uint64_t dependency,
            std::uint64_t current,bool fixed,
            std::shared_ptr<const void> retained_stream) {
            Boundary boundary;
            boundary.bind(context,in_owner,out_owner,&input,&output,
                reinterpret_cast<std::uintptr_t>(&stream),
                std::move(retained_stream),dependency,current,fixed);
            boundary.capture_work(captured);
            boundary.outside_work(Boundary::host_allocation|
                Boundary::host_registration|Boundary::host_export|
                Boundary::publication|Boundary::dynamic_callback|
                Boundary::external_transfer);
            return boundary;
        };
        const auto make=[&](std::uint32_t captured,std::uint64_t dependency=7,
            std::uint64_t current=7,bool fixed=true) {
            return make_with_stream(captured,dependency,current,fixed,stream_owner);
        };
        const auto eligible=make(Boundary::numerical);
        need(eligible.assess().eligible() &&
            (eligible.outside_work()&Boundary::host_export) &&
            (eligible.outside_work()&Boundary::publication),
            "host export/publication were not kept outside numerical capture");
        for(const auto illegal:{Boundary::host_allocation,
            Boundary::host_registration,Boundary::host_export,
            Boundary::publication,Boundary::dynamic_callback,
            Boundary::external_transfer,Boundary::dynamic_extent}) {
            need(!make(Boundary::numerical|illegal).assess().eligible(),
                "illegal host/dynamic operation entered graph capture");
        }
        need(make(Boundary::numerical,6,7).assess().reason==
                Boundary::Reason::stale_external_dependency &&
            make_with_stream(Boundary::numerical,7,7,true,{}).assess().reason==
                Boundary::Reason::missing_stream_owner &&
            make(Boundary::numerical,7,7,false).assess().reason==
                Boundary::Reason::unstable_geometry,
            "graph boundary accepted stale, unowned or dynamic input");
        std::cout<<"GRAPH_NUMERICAL_BOUNDARY_COMPLETE device_execution=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_EXECUTION_DEPENDENCY_GRAPH");
       mode && std::string_view(mode)=="1") {
        using Graph=ninfer::exl3::Exl3ExecutionDependencyGraph;
        using Stage=Graph::Stage;
        int stream_address=0,event_address=0,wrong_event=0;
        auto stream=std::make_shared<int>(1),draft=std::make_shared<int>(2);
        auto upload=std::make_shared<int>(3),attention=std::make_shared<int>(4);
        auto exported=std::make_shared<int>(5);
        std::array<std::weak_ptr<int>,5> weak{stream,draft,upload,attention,exported};
        Graph failed;
        const auto ticket=failed.begin(11,21,
            reinterpret_cast<std::uintptr_t>(&stream_address),stream);
        failed.add_node(ticket,Stage::draft,draft);
        failed.add_node(ticket,Stage::upload,upload);
        failed.add_node(ticket,Stage::attention,attention);
        failed.add_node(ticket,Stage::export_state,exported);
        failed.add_edge(ticket,Stage::draft,Stage::upload);
        failed.add_edge(ticket,Stage::upload,Stage::attention);
        bool missing=false;
        try {failed.submit_selected_slice(ticket,
            reinterpret_cast<std::uintptr_t>(&event_address));}
        catch(const std::logic_error&){missing=true;}
        need(missing && failed.snapshot().phase==Graph::Phase::building,
            "dependency graph accepted a missing export edge");
        failed.add_edge(ticket,Stage::attention,Stage::export_state);
        bool cycle=false;
        try {failed.add_edge(ticket,Stage::export_state,Stage::draft);}
        catch(const std::invalid_argument&){cycle=true;}
        need(cycle && failed.snapshot().edges==3,
            "dependency graph accepted a transfer/compute cycle");
        failed.submit_selected_slice(ticket,
            reinterpret_cast<std::uintptr_t>(&event_address));
        stream.reset();draft.reset();upload.reset();attention.reset();exported.reset();
        const auto all_retained=[&] {for(const auto& owner:weak)if(owner.expired())return false;return true;};
        const auto all_released=[&] {for(const auto& owner:weak)if(!owner.expired())return false;return true;};
        need(all_retained() &&
            !failed.complete(ticket,reinterpret_cast<std::uintptr_t>(&wrong_event),0) &&
            !failed.complete(ticket,reinterpret_cast<std::uintptr_t>(&event_address),37) &&
            failed.snapshot().phase==Graph::Phase::failed &&
            failed.snapshot().first_error==37 && !failed.retire(ticket),
            "failed dependency graph released a producer or accepted a stale event");
        need(failed.retire_after_stream_drain(ticket) &&
            all_released(),
            "drained failed dependency graph retained shared producers");

        Graph complete;
        auto owner=std::make_shared<int>(6);std::weak_ptr<int> retained=owner;
        const auto current=complete.begin(12,22,
            reinterpret_cast<std::uintptr_t>(&stream_address),owner);
        for(const auto stage:{Stage::draft,Stage::upload,Stage::attention,Stage::export_state})
            complete.add_node(current,stage,owner);
        complete.add_edge(current,Stage::draft,Stage::upload);
        complete.add_edge(current,Stage::upload,Stage::attention);
        complete.add_edge(current,Stage::attention,Stage::export_state);
        complete.submit_selected_slice(current,
            reinterpret_cast<std::uintptr_t>(&event_address));
        owner.reset();
        need(complete.complete(current,reinterpret_cast<std::uintptr_t>(&event_address),0) &&
            !retained.expired() && complete.retire(current) && retained.expired(),
            "completed dependency graph did not retain through explicit retirement");
        std::cout<<"EXECUTION_DEPENDENCY_GRAPH_COMPLETE device_execution=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_HOST_PREPARATION_COMPLETION");
       mode && std::string_view(mode)=="1") {
        using Completion=ninfer::exl3::Exl3HostPreparationCompletion;
        Completion completion;
        const auto first=completion.begin(1);
        need(first && completion.complete(*first,{101,80,64,true}) &&
            completion.snapshot().phase==Completion::Phase::ready,
            "host preparation did not publish a ready private plan");
        need(completion.cancel(first->generation) &&
            !completion.accept(*first,101,80) &&
            completion.snapshot().phase==Completion::Phase::cancelled,
            "cancelled host preparation remained acceptable");
        const auto second=completion.begin(2);
        need(second && !completion.complete(*first,{101,80,64,true}) &&
            completion.complete(*second,{202,96,64,false}) &&
            !completion.accept(*second,201,96) &&
            completion.snapshot().phase==Completion::Phase::ready,
            "stale or mismatched host completion changed current authority");
        const auto accepted=completion.accept(*second,202,96);
        need(accepted && accepted->cacheable_tokens==64 &&
            completion.snapshot().phase==Completion::Phase::accepted,
            "current host preparation could not be accepted exactly once");
        const auto third=completion.begin(3);
        need(third && completion.fail(*third) &&
            !completion.complete(*third,{303,64,64,false}) &&
            !completion.accept(*third,303,64),
            "failed host preparation exposed a late result");
        std::cout<<"HOST_PREPARATION_COMPLETION_COMPLETE device_execution=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_SHARED_PAIR_GEOMETRY");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        std::array<std::array<std::uint16_t,8>,4> buffers{};
        auto model=std::make_shared<int>(1),root=std::make_shared<int>(2),storage=std::make_shared<int>(3);
        Exl3ProjectionRows a{1,1,1,model,root,storage,storage,"pair-test",0,1,4,4,4,4,
            buffers[0].data(),buffers[1].data(),8,8};
        auto b=a;b.request=2;b.input=buffers[2].data();b.output=buffers[3].data();
        need(Exl3ProjectionRows::valid_pair(a,b,16) && Exl3ProjectionRows::valid_pair(b,a,16),
            "disjoint shared pair refused");
        for(unsigned alias:{0u,1u,2u,3u}) {
            auto invalid=b;
            if(alias==0)invalid.input=a.input+1;
            if(alias==1)invalid.input=a.output+1;
            if(alias==2)invalid.output=const_cast<std::uint16_t*>(a.input)+1;
            if(alias==3)invalid.output=a.output+1;
            need(!Exl3ProjectionRows::valid_pair(a,invalid,16),"shared pair admitted cross-lane overlap");
        }
        auto adjacent=b;adjacent.input=a.input+4;adjacent.input_storage_elements=4;
        need(Exl3ProjectionRows::valid_pair(a,adjacent,16),"shared pair rejected adjacent disjoint extents");
        auto stale=b;stale.execution=0;
        need(!Exl3ProjectionRows::valid_pair(a,stale,16) && !Exl3ProjectionRows::valid_pair(a,b,1),
            "shared pair admitted invalid epoch or insufficient row capacity");
        std::cout<<"SHARED_PAIR_GEOMETRY_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_FINGERPRINT");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        int context_address=0,stream_address=0;
        std::array<std::array<std::uint16_t,16>,3> storage{};
        std::array<Exl3GraphBufferIdentity,3> buffers{{
            {storage[0].data(),storage[0].size()*sizeof(std::uint16_t)},
            {storage[1].data(),storage[1].size()*sizeof(std::uint16_t)},
            {storage[2].data(),storage[2].size()*sizeof(std::uint16_t)}}};
        auto context_control_a=std::make_shared<int>(1);
        auto context_control_b=std::make_shared<int>(2);
        const std::shared_ptr<const void> context_a(context_control_a,&context_address);
        // Deliberately the same raw address under a different control block.
        const std::shared_ptr<const void> context_b(context_control_b,&context_address);
        const std::shared_ptr<const void> model=std::make_shared<int>(3);
        const std::shared_ptr<const void> scratch=std::make_shared<int>(4);
        const auto options=Exl3ProjectionGraphOptions::from_values(
            {"3","4","2","1"});
        const auto stream=reinterpret_cast<cudaStream_t>(&stream_address);
        const auto make=[&](const std::shared_ptr<const void>& context,
            std::span<const Exl3GraphBufferIdentity> identities,
            unsigned native_width,std::uint32_t route,std::uint64_t generation) {
            Exl3GraphCompatibilityFingerprint fingerprint;
            fingerprint.bind(context,model,scratch,identities,4,8,native_width,
                5120,3,route,Exl3GraphPrecision::oscar_int2_fp16,
                Exl3GraphPositionPolicy::oscar_split_class,generation,stream,options);
            need(fingerprint.valid(),"complete graph fingerprint was rejected");
            return fingerprint;
        };
        const auto baseline=make(context_a,buffers,8,3,9);
        need(baseline.matches(make(context_a,buffers,8,3,9)),
            "identical graph fingerprint did not match");
        need(!baseline.matches(make(context_b,buffers,8,3,9)),
            "same graph pointer under a replacement owner was accepted");
        auto moved_buffers=buffers;
        moved_buffers[1].address=storage[1].data()+1;
        need(!baseline.matches(make(context_a,moved_buffers,8,3,9)),
            "changed graph buffer address was accepted");
        need(!baseline.matches(make(context_a,buffers,8,7,9)),
            "changed graph route bit was accepted");
        need(!baseline.matches(make(context_a,buffers,16,3,9)),
            "changed native graph width was accepted");
        need(!baseline.matches(make(context_a,buffers,8,3,8)),
            "stale graph generation was accepted");
        std::cout<<"GRAPH_FINGERPRINT_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_REQUEST_COMPLETION_EVENT_POOL");mode && std::string_view(mode)=="1") {
        using Pool=ninfer::exl3::Exl3RequestCompletionEventPool;
        Pool pool;std::array<int,2> events{};
        pool.configure(std::array<std::uintptr_t,2>{
            reinterpret_cast<std::uintptr_t>(&events[0]),
            reinterpret_cast<std::uintptr_t>(&events[1])},101);
        auto owner0=std::make_shared<int>(1),owner1=std::make_shared<int>(2);
        std::weak_ptr<int> weak0=owner0,weak1=owner1;
        const auto first=pool.acquire(11,21,31,owner0);
        const auto second=pool.acquire(12,22,32,owner1);
        need(first && second && !pool.acquire(13,23,33,std::make_shared<int>(3)),
            "completion pool did not report bounded exhaustion");
        need(pool.submit(*first) && pool.cancel(*first) &&
                pool.snapshot(first->slot).phase==Pool::Phase::pending,
            "submitted cancellation retired before physical completion");
        owner0.reset();
        need(!weak0.expired() && pool.notify(*first,first->event,0) &&
                pool.snapshot(first->slot).phase==Pool::Phase::retired && weak0.expired(),
            "cancelled completion did not retain through notification and retire");
        auto replacement_owner=std::make_shared<int>(3);
        std::weak_ptr<int> replacement_weak=replacement_owner;
        const auto replacement=pool.acquire(13,23,33,replacement_owner);
        need(replacement && replacement->slot==first->slot &&
                replacement->event==first->event &&
                replacement->generation!=first->generation && pool.submit(*replacement),
            "completion pool did not generation-separate repeated acquisition");
        replacement_owner.reset();
        need(!pool.notify(*first,first->event,0) && !replacement_weak.expired(),
            "old completion notification certified a recycled event");
        auto wrong_retirement=*replacement;--wrong_retirement.retirement_generation;
        auto wrong_acquisition=*replacement;--wrong_acquisition.acquisition;
        need(!pool.notify(wrong_retirement,wrong_retirement.event,0) &&
                !pool.notify(wrong_acquisition,wrong_acquisition.event,0) &&
                !replacement_weak.expired(),
            "foreign acquisition or retirement tenure certified pending completion");
        need(pool.notify(*replacement,replacement->event,0) &&
                !pool.notify(*replacement,replacement->event,0) &&
                pool.matches(101,13,23,33,replacement->generation) &&
                !pool.acquire(14,24,34,std::make_shared<int>(4)) &&
                pool.retire(replacement->generation) && replacement_weak.expired() &&
                !pool.retire(replacement->generation),
            "ready completion was recycled early or duplicate notification was accepted");
        owner1.reset();
        need(pool.cancel(*second) && weak1.expired() &&
                pool.snapshot(second->slot).phase==Pool::Phase::retired,
            "unsubmitted cancellation retained an owner or failed retirement");
        auto failed_owner=std::make_shared<int>(5);
        std::weak_ptr<int> failed_weak=failed_owner;
        const auto failed=pool.acquire(14,24,34,failed_owner);
        need(failed && pool.submit(*failed),"completion failure fixture acquisition");
        failed_owner.reset();
        need(!pool.notify(*failed,failed->event,77) &&
                pool.snapshot(failed->slot).phase==Pool::Phase::failed &&
                pool.snapshot(failed->slot).first_error==77 &&
                !pool.cancel(*failed) && !pool.retire(failed->generation) &&
                !failed_weak.expired(),
            "completion error released its owner or became reusable");
        Pool reloaded;
        reloaded.configure(std::array<std::uintptr_t,2>{
            reinterpret_cast<std::uintptr_t>(&events[0]),
            reinterpret_cast<std::uintptr_t>(&events[1])},102);
        auto reload_owner=std::make_shared<int>(6);std::weak_ptr<int> reload_weak=reload_owner;
        const auto reload_ticket=reloaded.acquire(11,21,31,reload_owner);
        need(reload_ticket && reload_ticket->generation==first->generation &&
                reload_ticket->event==first->event && reloaded.submit(*reload_ticket),
            "post-reload fixture did not reproduce native event/logical generation alias");
        reload_owner.reset();
        need(!reloaded.notify(*first,first->event,0) && !reload_weak.expired() &&
                reloaded.notify(*reload_ticket,reload_ticket->event,0) &&
                reloaded.matches(102,11,21,31,reload_ticket->generation),
            "pre-reload callback certified replacement pool authority");
        Pool exhausted;
        exhausted.configure(std::array<std::uintptr_t,1>{
            reinterpret_cast<std::uintptr_t>(&events[0])},103);
        exhausted.exhaust_generation_for_test();
        bool wrapped=false;
        try{(void)exhausted.acquire(15,25,35,std::make_shared<int>(7));}
        catch(const std::overflow_error&){wrapped=true;}
        need(wrapped && exhausted.snapshot(0).phase==Pool::Phase::retired,
            "completion generation exhaustion wrapped into a reusable epoch");
        std::cout<<"REQUEST_COMPLETION_EVENT_POOL_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_CONTROL_PUBLICATION_BOUNDARY");mode && std::string_view(mode)=="1") {
        using Boundary=ninfer::exl3::Exl3ControlPublicationBoundary;
        Boundary boundary;
        const auto control=boundary.begin(11,21,Boundary::Kind::control);
        need(boundary.observe_independent(control,Boundary::IndependentWork::cancellation) &&
                boundary.observe_independent(control,Boundary::IndependentWork::deadline) &&
                boundary.observe_independent(control,Boundary::IndependentWork::capacity_preflight) &&
                !boundary.observe_independent(control,Boundary::IndependentWork::private_output),
            "publication boundary misclassified work while numerical result was pending");
        bool second_control=false;
        try{(void)boundary.begin(11,21,Boundary::Kind::control);}
        catch(const std::logic_error&){second_control=true;}
        need(second_control,"control append entered while prior numerical work was pending");
        need(boundary.numerical_ready(control) && boundary.prepare_output(control) &&
                boundary.begin_publication(control) && !boundary.stage_exposure(control),
            "publication boundary exposed output before resident validation");
        boundary.fail(control);
        need(boundary.snapshot().phase==Boundary::Phase::failed &&
                !boundary.resident_committed(control) && !boundary.stage_exposure(control) &&
                boundary.reset_after_request(),
            "failed publication revived exposure or prevented request retirement");

        const auto model=boundary.begin(12,22,Boundary::Kind::model);
        std::array<unsigned,4> ordering{};std::size_t count=0;
        need(boundary.numerical_ready(model),"model numerical completion ordering");
        ordering[count++]=1;
        need(boundary.prepare_output(model),"model private output ordering");
        ordering[count++]=2;
        need(boundary.resume_numerical(model) && boundary.numerical_ready(model) &&
                boundary.prepare_output(model) && boundary.begin_publication(model),
            "publication boundary rejected followup numerical ordering");
        ordering[count++]=3;
        need(boundary.resident_committed(model) && boundary.stage_exposure(model),
            "publication boundary rejected resident/exposure ordering");
        ordering[count++]=4;
        need(boundary.finish_exposure(model),
            "publication boundary rejected followup numerical/commit/exposure order");
        need(ordering==std::array<unsigned,4>{1,2,3,4} &&
                boundary.snapshot().phase==Boundary::Phase::idle &&
                !boundary.finish_exposure(control),
            "publication callback ordering accepted a stale ticket");
        std::cout<<"CONTROL_PUBLICATION_BOUNDARY_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_DEVICE_GRAPH_ROLE_TABLE");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        std::array<std::int64_t,4> tokens_a{},tokens_b{};
        std::array<std::uint16_t,32> hidden{},trace{};
        auto token_owner_a=std::make_shared<int>(1);
        auto token_owner_b=std::make_shared<int>(2);
        auto hidden_owner=std::make_shared<int>(3);
        auto trace_owner=std::make_shared<int>(4);
        std::weak_ptr<int> old_token_owner=token_owner_a;
        const auto bindings_for=[&](std::int64_t* tokens,
                const std::shared_ptr<int>& token_owner) {
            return std::array<Exl3GraphRoleBinding,3>{{
                {Exl3GraphPointerRole::token_ids,token_owner,tokens,4,
                    sizeof(std::int64_t),false},
                {Exl3GraphPointerRole::hidden_output,hidden_owner,hidden.data(),32,
                    sizeof(std::uint16_t),true},
                {Exl3GraphPointerRole::embedding_trace,trace_owner,trace.data(),32,
                    sizeof(std::uint16_t),true}
            }};
        };
        Exl3DeviceGraphRoleTable table;table.configure(4,8);
        auto bindings=bindings_for(tokens_a.data(),token_owner_a);
        const auto first=table.update(bindings);
        need(first.changed && first.record.generation==1 &&
                first.record.token_ids==tokens_a.data() &&
                first.record.hidden_output==hidden.data() &&
                first.record.embedding_trace==trace.data(),
            "complete graph role generation was not published atomically");
        const auto expect_invalid=[&](std::span<const Exl3GraphRoleBinding> candidate,
                const char* message) {
            bool refused=false;
            try{(void)table.update(candidate);}catch(const std::invalid_argument&){refused=true;}
            need(refused && table.generation()==1,message);
        };
        expect_invalid(std::span<const Exl3GraphRoleBinding>(bindings.data(),2),
            "graph role table admitted a partial update");
        auto reused=bindings;reused[2].role=Exl3GraphPointerRole::hidden_output;
        expect_invalid(reused,"graph role table admitted a reused role slot");
        auto geometry=bindings;geometry[1].elements=31;
        expect_invalid(geometry,"graph role table admitted incompatible pointer geometry");
        auto alias=bindings;alias[2].address=hidden.data();
        expect_invalid(alias,"graph role table admitted aliased writable roles");
        bool mismatch=false;
        try{table.mark_uploaded(2);}catch(const std::logic_error&){mismatch=true;}
        need(mismatch && table.uploaded_generation()==0,
            "graph role table accepted an upload generation mismatch");
        table.mark_uploaded(1);
        mismatch=false;
        try{table.begin_use(2,7);}catch(const std::logic_error&){mismatch=true;}
        need(mismatch && !table.pending_replay(),
            "graph role table accepted a stale replay generation");
        table.begin_use(1,7);
        const auto unchanged=table.update(bindings);
        need(!unchanged.changed && table.pending_replay()==7,
            "identical graph roles disturbed a pending replay");
        token_owner_a.reset();bindings[0].owner.reset();
        need(!old_token_owner.expired(),
            "graph role table did not retain the active token owner");
        auto replacement=bindings_for(tokens_b.data(),token_owner_b);
        bool pending_refused=false;
        try{(void)table.update(replacement);}catch(const std::logic_error&){pending_refused=true;}
        need(pending_refused && table.generation()==1 && !old_token_owner.expired(),
            "pending graph replay admitted a replacement token owner");
        need(!table.complete_use(6) && table.complete_use(7),
            "graph role table lost the exact final-use frontier");
        const auto second=table.update(replacement);
        need(second.changed && second.record.generation==2 &&
                second.record.token_ids==tokens_b.data() && old_token_owner.expired(),
            "completed graph role slot did not replace and release its old owner");
        table.mark_uploaded(2);table.begin_use(2,8);
        need(table.complete_use(8),"replacement graph role generation did not complete");
        std::cout<<"DEVICE_GRAPH_ROLE_TABLE_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_EXPORT_COPY_PLAN");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        std::array<std::byte,64> source{},destination{};
        for(std::size_t i=0;i<source.size();++i)
            source[i]=static_cast<std::byte>((i*37u+11u)&0xffu);
        Exl3ExportCopyPlan<6> plan;
        need(plan.add(destination.data(),source.data(),8,
                Exl3ExportPrecision::fp16) &&
             plan.add(destination.data()+8,source.data()+8,8,
                Exl3ExportPrecision::fp16) &&
             plan.add(destination.data()+16,source.data()+16,8,
                Exl3ExportPrecision::fp32) &&
             plan.add(destination.data()+24,source.data()+24,8,
                Exl3ExportPrecision::fp32,true) &&
             plan.add(destination.data()+40,source.data()+40,8,
                Exl3ExportPrecision::fp16),
            "export copy fixture rejected bounded ranges");
        need(plan.size()==4 && plan.logical_ranges()==5 &&
                plan.coalesced_ranges()==1 && plan.logical_bytes()==40 &&
                plan[0].bytes==16 && plan[0].logical_ranges==2,
            "export copy merged precision, padding or discontiguous ranges");
        unsigned calls=0;
        const auto partial=plan.submit([&](const auto& segment) noexcept {
            if(++calls==2)return 37;
            std::memcpy(segment.destination,segment.source,segment.bytes);return 0;
        });
        need(!partial && partial.attempted==2 && partial.completed==1 &&
                partial.failed_segment==1 && partial.status==37 &&
                std::memcmp(destination.data(),source.data(),16)==0 &&
                destination[16]==std::byte{},
            "export copy partial failure reordered or published later ranges");
        destination.fill(std::byte{});
        const auto complete=plan.submit([&](const auto& segment) noexcept {
            std::memcpy(segment.destination,segment.source,segment.bytes);return 0;
        });
        need(complete && complete.attempted==4 && complete.completed==4 &&
                std::memcmp(destination.data(),source.data(),32)==0 &&
                std::memcmp(destination.data()+40,source.data()+40,8)==0,
            "export copy plan changed represented bits");
        Exl3ExportCopyPlan<1> full;
        need(full.add(destination.data(),source.data(),4,Exl3ExportPrecision::fp32) &&
                !full.add(destination.data()+8,source.data()+8,4,
                    Exl3ExportPrecision::fp32) &&
                full.size()==1 && full.logical_ranges()==1,
            "export copy capacity refusal mutated the accepted plan");
        bool invalid=false;
        try{(void)full.add(destination.data()+1,source.data(),3,
            Exl3ExportPrecision::fp16);}catch(const std::invalid_argument&){invalid=true;}
        need(invalid,"export copy accepted mis-sized precision range");
        std::cout<<"EXPORT_COPY_PLAN_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_BOUNDED_GRAPH_ENTRY");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        int context_address=0,buffer_address=0,stream_address=0;
        const auto stream=reinterpret_cast<cudaStream_t>(&stream_address);
        const auto options=Exl3ProjectionGraphOptions::from_values(
            {"2","4","1","0"});
        Exl3GraphCaptureExtent reservation;reservation.known=true;
        reservation.retained[static_cast<unsigned>(
            Exl3ResourceInventory::Domain::graph_count)]=2;
        reservation.retained[static_cast<unsigned>(
            Exl3ResourceInventory::Domain::host_metadata)]=96;
        reservation.temporary[static_cast<unsigned>(
            Exl3ResourceInventory::Domain::device)]=4096;
        const auto make_fingerprint=[&](const std::shared_ptr<const void>& context,
            std::uint64_t generation) {
            std::array<Exl3GraphBufferIdentity,1> buffers{{
                {&buffer_address,sizeof(buffer_address)}}};
            Exl3GraphCompatibilityFingerprint result;
            result.bind(context,context,{},buffers,8,8,8,5120,3,1,
                Exl3GraphPrecision::oscar_int2_fp16,
                Exl3GraphPositionPolicy::oscar_split_class,generation,stream,options);
            need(result.valid(),"bounded graph fixture fingerprint");return result;
        };
        {
            auto context=std::make_shared<int>(1);
            std::shared_ptr<const void> context_identity(context,&context_address);
            std::weak_ptr<int> weak=context;
            std::array<Exl3GraphBoundResource,1> resources{{
                {context_identity,&buffer_address,sizeof(buffer_address),0}}};
            Exl3BoundedGraphEntry entry;
            entry.bind(make_fingerprint(context_identity,7),resources,7,91,reservation);
            resources={};context_identity.reset();context.reset();
            need(!weak.expired(),"graph entry failed to retain context resource owner");
            const auto replay=entry.begin_replay(7,stream);
            entry.invalidate("cache eviction");
            entry.invalidate("later replacement reason");
            auto pending=entry.snapshot();
            need(pending.pending && pending.replay_serial==replay &&
                    pending.resource_count==1 &&
                    pending.retained[static_cast<unsigned>(
                        Exl3ResourceInventory::Domain::graph_count)]==2 &&
                    pending.temporary[static_cast<unsigned>(
                        Exl3ResourceInventory::Domain::device)]==4096 &&
                    pending.invalidation_reason=="cache eviction" &&
                    !entry.release_after_destroy() && !weak.expired(),
                "pending graph replay released owner or lost peak/reason");
            need(!entry.complete_after_drain(nullptr,replay,0),
                "foreign/default stream completed pending graph replay");
            need(entry.complete_after_drain(stream,replay,0) &&
                    entry.release_after_destroy() && weak.expired(),
                "completed graph destruction retained context resources");
        }
        {
            auto resource=std::make_shared<int>(2);
            std::shared_ptr<const void> identity(resource,&context_address);
            std::weak_ptr<int> weak=resource;
            std::array<Exl3GraphBoundResource,1> resources{{
                {identity,&buffer_address,sizeof(buffer_address),0}}};
            Exl3BoundedGraphEntry failed;
            failed.bind(make_fingerprint(identity,8),resources,8,92,reservation);
            failed.invalidate("cache eviction");
            bool rebound=false;
            try {failed.bind(make_fingerprint(identity,8),resources,8,92,reservation);}
            catch(const std::logic_error&) {rebound=true;}
            need(rebound,"invalidated graph entry rebound before handle destruction");
            failed.quarantine("native graph destruction failed",77);
            resources={};identity.reset();resource.reset();
            const auto snapshot=failed.snapshot();
            need(snapshot.phase==Exl3BoundedGraphEntry::Phase::quarantined &&
                    snapshot.first_error==77 && !failed.release_after_destroy() &&
                    !weak.expired(),
                "failed graph destruction released resources or lost first error");
        }
        {
            auto resource=std::make_shared<int>(3);
            std::shared_ptr<const void> identity(resource,&context_address);
            std::weak_ptr<int> weak=resource;
            std::array<Exl3GraphBoundResource,1> resources{{
                {identity,&buffer_address,sizeof(buffer_address),0}}};
            Exl3BoundedGraphEntry failed_replay;
            failed_replay.bind(make_fingerprint(identity,9),resources,9,93,reservation);
            const auto replay=failed_replay.begin_replay(9,stream);
            resources={};identity.reset();resource.reset();
            need(!failed_replay.complete_after_drain(stream,replay,37),
                "failed replay was reported as a completed final use");
            failed_replay.quarantine("prepared replay failure",37);
            const auto snapshot=failed_replay.snapshot();
            need(snapshot.phase==Exl3BoundedGraphEntry::Phase::quarantined &&
                    snapshot.first_error==37 && snapshot.resource_count==1 &&
                    snapshot.resource_bytes==sizeof(buffer_address) &&
                    !failed_replay.release_after_destroy() && !weak.expired(),
                "failed replay released bound owner or lost resource/error identity");
        }
        std::cout<<"BOUNDED_GRAPH_ENTRY_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_MENU_TRANSACTION");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;using Inventory=Exl3ResourceInventory;
        Exl3VeriCacheServingIdentity identity{"graph-menu","weights","tokens","exact","text"};
        Exl3VeriCacheServingPrefixCache roots({2,64,2ULL<<30,8ULL<<30},identity);
        Exl3VeriCacheServingCoordinator authority(roots,{1,1,16ULL<<30,8ULL<<30});
        auto limits=Inventory::unlimited();limits[static_cast<unsigned>(Inventory::Domain::device)]=80;
        authority.bind_physical_resources({},limits);
        std::array<Exl3GraphCaptureEntry,2> menu{};
        menu[0].rows=4;menu[1].rows=8;
        for(auto& entry:menu) {
            entry.extent.known=true;
            entry.extent.retained[static_cast<unsigned>(Inventory::Domain::device)]=20;
            entry.extent.retained[static_cast<unsigned>(Inventory::Domain::graph_count)]=2;
            entry.extent.temporary[static_cast<unsigned>(Inventory::Domain::device)]=40;
        }
        std::array<std::shared_ptr<int>,2> owners;
        std::array<std::weak_ptr<int>,2> weak;
        unsigned calls=0,rollbacks=0,observations=0;bool mismatch=true;
        const auto factory=[&](const Exl3GraphCaptureEntry& entry) {
            const auto slot=calls++;
            need(slot<2 && entry.rows==(slot?8u:4u),"graph menu changed during factory");
            owners[slot]=std::make_shared<int>(1);weak[slot]=owners[slot];
            Inventory actual;actual.add({owners[slot],0,Inventory::Domain::device,mismatch && slot==1?21u:20u});
            actual.add({owners[slot],1,Inventory::Domain::graph_count,2});
            if(!slot)menu[1].rows=16; // The accepted menu must already be copied.
            return actual;
        };
        const auto rollback=[&]() noexcept {++rollbacks;owners={};};
        bool refused=false;
        try{(void)exl3_allocate_graph_menu_startup(authority,9,menu,factory,rollback,[&] {
            ++observations;need(weak[0].expired() && weak[1].expired(),"graph menu rollback retained partial owners");
        });}catch(const std::invalid_argument& error){refused=std::string_view(error.what())=="graph menu entry retained extent mismatch";}
        need(refused && calls==2 && rollbacks==1 && observations==1,"graph menu partial failure boundary");
        menu[1].rows=8;
        {
            unsigned aliases=0;bool rejected=false;
            try{(void)exl3_allocate_graph_menu_startup(authority,9,menu,[&](const Exl3GraphCaptureEntry&) {
                if(!owners[0])owners[0]=std::make_shared<int>(1);
                ++aliases;weak[0]=owners[0];Inventory actual;
                actual.add({owners[0],0,Inventory::Domain::device,20});
                actual.add({owners[0],1,Inventory::Domain::graph_count,2});return actual;
            },rollback);}catch(const std::invalid_argument& error) {
                rejected=std::string_view(error.what())=="graph menu reused entry allocation owner";
            }
            need(rejected && aliases==2 && rollbacks==2 && weak[0].expired(),
                "graph menu counted duplicate owners as separate entry allocations");
        }
        calls=0;mismatch=false;menu[1].rows=8;
        auto retained=exl3_allocate_graph_menu_startup(authority,9,menu,factory,rollback);
        need(calls==2 && retained.totals()[static_cast<unsigned>(Inventory::Domain::device)]==40,
            "graph menu retry retained temporary peak or omitted entry");
        owners={};retained={};authority.close();
        need(weak[0].expired() && weak[1].expired(),"graph menu close retained owner");
        std::cout<<"GRAPH_MENU_TRANSACTION_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_STARTUP_PEAK_CREDIT");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;using Inventory=Exl3ResourceInventory;
        Exl3VeriCacheServingIdentity identity{"peak-credit","weights","tokens","exact","text"};
        Exl3VeriCacheServingPrefixCache roots({2,64,2ULL<<30,8ULL<<30},identity);
        Exl3VeriCacheServingCoordinator authority(roots,{1,1,16ULL<<30,8ULL<<30});
        auto limits=Inventory::unlimited();limits[static_cast<unsigned>(Inventory::Domain::device)]=100;
        authority.bind_physical_resources({},limits);
        Inventory::Requirement peak,retained;peak.configuration=retained.configuration=7;
        peak.add(Inventory::Domain::device,1,100);retained.add(Inventory::Domain::device,1,50);
        unsigned calls=0,rollbacks=0;std::shared_ptr<int> prepared;
        const auto factory=[&](std::uint64_t configuration) {
            need(configuration==7,"startup peak configuration changed");++calls;
            prepared=std::make_shared<int>(1);Inventory actual;
            actual.add({prepared,0,Inventory::Domain::device,50});return actual;
        };
        const auto rollback=[&]() noexcept {++rollbacks;prepared.reset();};
        for(unsigned invalid:{0u,1u,2u,3u}) {
            auto bad_peak=peak,bad_retained=retained;
            if(invalid==0)bad_peak.configuration=0;
            if(invalid==1)bad_retained.configuration=8;
            if(invalid==2)bad_retained.units={};
            if(invalid==3)bad_retained.units[static_cast<unsigned>(Inventory::Domain::device)]=101;
            bool refused=false;
            try{authority.allocate_startup_peak_resources(bad_peak,bad_retained,factory,rollback);}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && calls==0 && rollbacks==0 && !prepared,
                "invalid peak declaration reached factory or rollback");
        }
        for(unsigned mismatch:{0u,1u,2u}) {
            unsigned failed_calls=0,observations=0;std::weak_ptr<int> weak;
            bool refused=false;
            try {
                authority.allocate_startup_peak_resources(peak,retained,[&](std::uint64_t) {
                    ++failed_calls;prepared=std::make_shared<int>(1);weak=prepared;
                    Inventory actual;
                    actual.add({prepared,0,mismatch==2?Inventory::Domain::host_metadata:Inventory::Domain::device,
                        mismatch==0?49u:mismatch==1?51u:50u});
                    return actual;
                },rollback,[&] {
                    ++observations;
                    need(weak.expired(),"peak mismatch observed before owner rollback");
                });
            } catch(const std::invalid_argument& error) {
                refused=std::string_view(error.what())=="startup peak retained extent mismatch";
            }
            need(refused && failed_calls==1 && observations==1 && weak.expired() && !prepared &&
                rollbacks==mismatch+1,"peak mismatch lost exact retained extent or rollback");
        }
        {
            const auto declared=retained.units;bool refused=false;
            try {
                authority.allocate_startup_peak_resources(peak,retained,[&](std::uint64_t) {
                    retained.units[static_cast<unsigned>(Inventory::Domain::device)]=49;
                    prepared=std::make_shared<int>(1);Inventory actual;
                    actual.add({prepared,0,Inventory::Domain::device,49});return actual;
                },rollback);
            } catch(const std::invalid_argument& error) {
                refused=std::string_view(error.what())=="startup peak retained extent mismatch";
            }
            retained.units=declared;
            need(refused && !prepared && rollbacks==4,"peak factory rewrote its retained contract");
        }
        authority.allocate_startup_peak_resources(peak,retained,factory,rollback);
        bool refused=false;
        try{authority.allocate_startup_peak_resources(peak,retained,factory,rollback);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        need(refused && calls==1,"startup peak ignored retained old owner or invoked refused factory");
        peak.units=retained.units;
        authority.allocate_startup_peak_resources(peak,retained,factory,rollback);
        need(calls==2,"startup peak failed to release temporary reservation after commit");
        prepared.reset();authority.close();
        std::cout<<"STARTUP_PEAK_CREDIT_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_MENU_RESERVATION");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;using Inventory=Exl3ResourceInventory;
        const auto graphs=static_cast<unsigned>(Inventory::Domain::graph_count);
        std::array<Exl3GraphCaptureEntry,3> menu{};
        for(std::size_t i=0;i<menu.size();++i) {
            menu[i].rows=std::array<unsigned,3>{4,6,8}[i];
            menu[i].extent.known=true;menu[i].extent.retained[graphs]=2;
            menu[i].extent.temporary[graphs]=1;
        }
        const auto reserve_with_limit=[&](std::uint64_t limit,
                std::shared_ptr<int> owner) {
            Exl3VeriCacheServingIdentity identity{"graph-reserve","weights","tokens","exact","text"};
            Exl3VeriCacheServingPrefixCache roots({1,64,2ULL<<30,8ULL<<30},identity);
            Exl3VeriCacheServingCoordinator authority(roots,{1,1,16ULL<<30,8ULL<<30});
            auto limits=Inventory::unlimited();limits[graphs]=limit;
            authority.bind_physical_resources({},limits);
            const auto result=exl3_reserve_graph_menu_startup(
                authority,11,menu,owner,40);
            owner.reset();authority.close();return result;
        };
        auto refused_owner=std::make_shared<int>(1);
        std::weak_ptr<int> refused_weak=refused_owner;bool exhausted=false;
        try{(void)reserve_with_limit(6,refused_owner);}
        catch(const Exl3ResourceReservationExhausted&){exhausted=true;}
        refused_owner.reset();
        need(exhausted && refused_weak.expired(),
            "graph menu capture-peak refusal retained owner or reached allocation");
        auto accepted_owner=std::make_shared<int>(2);
        const auto reserved=reserve_with_limit(7,accepted_owner);
        need(reserved[graphs]==7,
            "graph menu did not retain the complete handles-plus-capture peak");
        std::array<Exl3GraphCaptureEntry,2> fallback{{menu[0],{}}};
        fallback[1].rows=1;fallback[1].extent.known=true;
        fallback[1].disposition=Exl3GraphCaptureDisposition::eager_fallback;
        const auto fallback_plan=Exl3GraphCaptureRequirements::derive(12,{},fallback);
        need(fallback_plan.additional_peak.units[graphs]==3 &&
                fallback_plan.retained_after_capture[graphs]==2,
            "explicit eager fallback consumed graph or capture capacity");
        fallback[1].extent.retained[graphs]=1;bool fallback_refused=false;
        try{(void)Exl3GraphCaptureRequirements::derive(12,{},fallback);}
        catch(const std::invalid_argument&){fallback_refused=true;}
        need(fallback_refused,"resource-bearing eager fallback entered graph menu");
        std::array<Exl3GraphCaptureEntry,5> overflow{};
        bool overflow_refused=false;
        try{(void)Exl3GraphCaptureRequirements::derive(13,{},overflow);}
        catch(const std::invalid_argument&){overflow_refused=true;}
        need(overflow_refused,"overflowing graph menu reached capture admission");
        std::cout<<"GRAPH_MENU_RESERVATION_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_CAPTURE_REQUIREMENTS");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        using Inventory=Exl3ResourceInventory;
        const auto device=static_cast<unsigned>(Inventory::Domain::device);
        const auto graphs=static_cast<unsigned>(Inventory::Domain::graph_count);
        Inventory::Totals old{};old[device]=100;old[graphs]=2;
        std::array<Exl3GraphCaptureEntry,2> entries{};
        entries[0].rows=4;entries[1].rows=8;
        for(auto& entry:entries){entry.extent.known=true;entry.extent.retained[graphs]=2;}
        entries[0].extent.retained[device]=20;entries[0].extent.temporary[device]=50;
        entries[1].extent.retained[device]=30;entries[1].extent.temporary[device]=40;
        const auto plan=Exl3GraphCaptureRequirements::derive(7,old,entries);
        need(plan.additional_peak.configuration==7 && plan.additional_peak.units[device]==100 &&
            plan.retained_after_capture[device]==50 && plan.peak_with_old_owners[device]==200 &&
            plan.peak_with_old_owners[graphs]==6,"graph capture peak lost old owners or serialized temporary storage");
        for(unsigned fault:{0u,1u,2u,3u}) {
            auto invalid=entries;
            if(fault==0)invalid[1].extent.known=false;
            if(fault==1)invalid[1].rows=4;
            if(fault==2)invalid[1].rows=16;
            if(fault==3)invalid[1].extent.retained[graphs]=1;
            bool refused=false;try{(void)Exl3GraphCaptureRequirements::derive(7,old,invalid);}
            catch(const std::invalid_argument&){refused=true;}
            need(refused,"graph capture admitted incomplete or unsupported menu");
        }
        old[device]=std::numeric_limits<std::uint64_t>::max();
        bool overflow=false;try{(void)Exl3GraphCaptureRequirements::derive(7,old,entries);}
        catch(const std::overflow_error&){overflow=true;}
        need(overflow,"graph capture old/new peak overflow accepted");
        std::cout<<"GRAPH_CAPTURE_REQUIREMENTS_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_CONTEXT_GRAPH_RETIREMENT");mode && *mode) {
        const std::string_view selector(mode);
        const unsigned failure=selector=="clean"?0:selector=="executable"?1:selector=="definition"?2:
            selector=="query"?3:selector=="device"?4:selector=="drain"?5:6;
        need(failure<=5,"context graph retirement selector");
        ninfer::exl3::Exl3TextContext::exercise_context_graph_retirement_for_test(failure);
        std::cout<<"CONTEXT_GRAPH_RETIREMENT_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_INVALIDATION");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3TextContext::exercise_graph_invalidation_for_test();
        std::cout<<"GRAPH_INVALIDATION_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_PARTIAL_CREATION");mode && std::string_view(mode)=="1") {
        static int graph_storage=0,exec_storage=0;
        static unsigned begins=0,ends=0,graph_destroys=0,instantiations=0,exec_destroys=0;
        static bool fail_end=false,fail_destroy=false,fail_launch=false;
        DecodeGraphDefinition::Provider definition_provider{
            +[](cudaStream_t){++begins;return cudaSuccess;},
            +[](cudaStream_t,cudaGraph_t* graph){++ends;*graph=reinterpret_cast<cudaGraph_t>(&graph_storage);return fail_end?cudaErrorUnknown:cudaSuccess;},
            +[](cudaGraph_t){++graph_destroys;return fail_destroy?cudaErrorUnknown:cudaSuccess;}};
        for(unsigned missing:{0u,1u,2u}) {
            auto incomplete=definition_provider;
            if(missing==0)incomplete.begin=nullptr;
            if(missing==1)incomplete.end=nullptr;
            if(missing==2)incomplete.destroy=nullptr;
            bool refused=false;try{DecodeGraphDefinition invalid(incomplete);}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && begins==0 && ends==0 && graph_destroys==0,
                "incomplete graph provider acquired resources before rejection");
        }
        for(bool destruction_failure:{false,true})for(bool body_failure:{false,true}) {
            begins=ends=graph_destroys=0;fail_destroy=destruction_failure;fail_end=!body_failure;
            {
                DecodeGraphDefinition graph(definition_provider);bool refused=false;
                try{graph.capture(nullptr,[&]{if(body_failure)throw std::runtime_error("prepared capture body failure");});}
                catch(const std::exception& error) {
                    refused=!body_failure || std::string_view(error.what())=="prepared capture body failure";
                }
                need(refused && begins==1 && ends==1 && graph_destroys==1 && graph.ready()==destruction_failure,
                    "partial graph definition lost original failure or returned handle");
                DecodeGraphDefinition moved(std::move(graph));
                need(!graph.ready() && moved.ready()==destruction_failure &&
                    moved.try_reset()==(destruction_failure?cudaErrorUnknown:cudaSuccess) && graph_destroys==1,
                    "moved partial graph lost poison/provider or retried cleanup");
            }
            need(graph_destroys==1,"partial graph destructor retried uncertain handle");
        }
        fail_end=false;fail_destroy=false;
        DecodeGraphDefinition definition(definition_provider);definition.capture(nullptr,[]{});
        DecodeGraphExecutable::Provider executable_provider{
            +[](cudaGraphExec_t* exec,cudaGraph_t){++instantiations;*exec=reinterpret_cast<cudaGraphExec_t>(&exec_storage);return cudaErrorUnknown;},
            +[](cudaGraphExec_t){++exec_destroys;return fail_destroy?cudaErrorUnknown:cudaSuccess;}};
        for(unsigned missing:{0u,1u}) {
            auto incomplete=executable_provider;
            if(missing==0)incomplete.instantiate=nullptr;else incomplete.destroy=nullptr;
            bool refused=false;try{DecodeGraphExecutable invalid(incomplete);}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && instantiations==0 && exec_destroys==0,
                "incomplete executable provider acquired resources before rejection");
        }
        {
            auto replay_provider=executable_provider;
            replay_provider.instantiate=+[](cudaGraphExec_t* exec,cudaGraph_t){
                ++instantiations;
                *exec=reinterpret_cast<cudaGraphExec_t>(&exec_storage);
                return cudaSuccess;
            };
            replay_provider.launch=+[](cudaGraphExec_t,cudaStream_t){
                return fail_launch?cudaErrorUnknown:cudaSuccess;
            };
            DecodeGraphExecutable replay(replay_provider);
            replay.instantiate(definition);
            replay.launch(nullptr);
            fail_launch=true;bool refused=false;
            try{replay.launch(nullptr);}catch(const std::runtime_error&){refused=true;}
            fail_launch=false;
            need(refused && replay.ready(),
                "failed graph replay released executable ownership");
        }
        for(bool destruction_failure:{false,true}) {
            fail_destroy=destruction_failure;instantiations=exec_destroys=0;
            {
                DecodeGraphExecutable executable(executable_provider);bool refused=false;
                try{executable.instantiate(definition);}catch(const std::exception&){refused=true;}
                need(refused && instantiations==1 && exec_destroys==1 && executable.ready()==destruction_failure,
                    "partial executable lost failed-create handle");
                DecodeGraphExecutable moved(std::move(executable));
                need(!executable.ready() && moved.ready()==destruction_failure &&
                    moved.try_reset()==(destruction_failure?cudaErrorUnknown:cudaSuccess) && exec_destroys==1,
                    "moved partial executable retried uncertain destroy");
            }
            need(exec_destroys==1,"partial executable destructor retried uncertain handle");
        }
        fail_destroy=false;
        {
            DecodeGraphDefinition destination(definition_provider),source(definition_provider);
            destination.capture(nullptr,[]{});source.capture(nullptr,[]{});
            const auto before=graph_destroys;fail_destroy=true;
            destination=std::move(source);
            need(destination.ready() && source.ready() && graph_destroys==before+1,
                "failed graph destination retirement consumed source ownership");
            fail_destroy=false;destination=std::move(source);
            need(destination.ready() && source.ready() && graph_destroys==before+1,
                "poisoned graph move assignment retried destination destruction");
            need(source.try_reset()==cudaSuccess && !source.ready(),"graph move source was not independently releasable");
        }
        {
            auto successful=executable_provider;
            successful.instantiate=+[](cudaGraphExec_t* exec,cudaGraph_t){++instantiations;*exec=reinterpret_cast<cudaGraphExec_t>(&exec_storage);return cudaSuccess;};
            DecodeGraphExecutable destination(successful),source(successful);
            destination.instantiate(definition);source.instantiate(definition);
            const auto before=exec_destroys;fail_destroy=true;
            destination=std::move(source);
            need(destination.ready() && source.ready() && exec_destroys==before+1,
                "failed executable destination retirement consumed source ownership");
            fail_destroy=false;destination=std::move(source);
            need(destination.ready() && source.ready() && exec_destroys==before+1,
                "poisoned executable move assignment retried destruction");
            need(source.try_reset()==cudaSuccess && !source.ready(),"executable move source lost independent release");
        }
        need(definition.try_reset()==cudaSuccess,"prepared definition clean release");
        std::cout<<"GRAPH_PARTIAL_CREATION_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_HANDLE_RETIREMENT");mode && std::string_view(mode)=="1") {
        int storage=0;int* failed=&storage;unsigned calls=0;
        const auto failure=ninfer::detail::retire_decode_graph_handle(failed,[&](int* value) noexcept {
            ++calls;return value==&storage?cudaErrorUnknown:cudaErrorInvalidValue;
        });
        need(failure==cudaErrorUnknown && failed==&storage && calls==1,
            "failed graph retirement forgot handle provenance");
        {
            int* uncertain=&storage;cudaError_t first=cudaSuccess;unsigned attempts=0;
            const auto destroy=[&](int*) noexcept {++attempts;return cudaErrorUnknown;};
            need(ninfer::detail::retire_decode_graph_handle_once(uncertain,first,destroy)==cudaErrorUnknown,
                "graph destroy did not retain first failure");
            need(ninfer::detail::retire_decode_graph_handle_once(uncertain,first,[&](int*) noexcept {
                ++attempts;return cudaSuccess;
            })==cudaErrorUnknown && uncertain==&storage && attempts==1,
                "graph cleanup retried uncertain native destruction or replaced first error");
        }
        int* clean=&storage;
        need(ninfer::detail::retire_decode_graph_handle(clean,[&](int*) noexcept {
            ++calls;return cudaSuccess;
        })==cudaSuccess && !clean && calls==2,"successful graph retirement retained handle");
        need(ninfer::detail::retire_decode_graph_handle(clean,[&](int*) noexcept {
            ++calls;return cudaErrorUnknown;
        })==cudaSuccess && calls==2,"empty graph retirement invoked provider");
        struct GraphOwner {
            cudaError_t error;unsigned& order;unsigned expected;unsigned calls=0;bool retained=true;
            cudaError_t try_reset() noexcept {
                ++calls;
                if(++order!=expected)return cudaErrorInvalidValue;
                if(error==cudaSuccess)retained=false;
                return error;
            }
        };
        for(unsigned failure:{0u,1u,2u}) {
            unsigned order=0;
            GraphOwner executable{failure==1?cudaErrorUnknown:cudaSuccess,order,1};
            GraphOwner definition{failure==2?cudaErrorUnknown:cudaSuccess,order,2};
            const auto result=ninfer::detail::retire_decode_graph_pair(executable,definition);
            need(result==(failure?cudaErrorUnknown:cudaSuccess) && executable.calls==1 &&
                definition.calls==(failure==1?0u:1u) && executable.retained==(failure==1) &&
                definition.retained==(failure!=0),
                "graph pair retirement lost ordering, survivor or first-failure stop");
        }
        for(unsigned failure:{0u,1u,2u,3u,4u,5u}) {
            unsigned queries=0,drains=0,retires=0;
            const auto result=ninfer::detail::retire_decode_graph_device(failure==1?-1:3,
                [&](int* device) noexcept {
                    ++queries;*device=failure==3?4:3;
                    return failure==2?cudaErrorInitializationError:cudaSuccess;
                },[&]() noexcept {++drains;return failure==4?cudaErrorUnknown:cudaSuccess;},
                [&]() noexcept {++retires;return failure==5?cudaErrorUnknown:cudaSuccess;});
            const auto expected=failure==1 || failure==3?cudaErrorInvalidDevice:
                failure==2?cudaErrorInitializationError:failure>=4?cudaErrorUnknown:cudaSuccess;
            need(result==expected && queries==(failure==1?0u:1u) &&
                drains==(failure==0 || failure>=4?1u:0u) &&
                retires==(failure==0 || failure==5?1u:0u),
                "graph device retirement bypassed provenance, drain or first error");
        }
        std::cout<<"GRAPH_HANDLE_RETIREMENT_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GRAPH_RETIREMENT_POISON");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3TextContext::exercise_graph_retirement_poison_for_test();
        std::cout<<"GRAPH_RETIREMENT_POISON_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_CONTINUATION_OWNER_METADATA");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3TextContext::exercise_continuation_owner_metadata_for_test();
        std::cout<<"CONTINUATION_OWNER_METADATA_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_DRAFT_RING_UNCERTAIN_METADATA");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3DraftHostRing::exercise_uncertain_metadata_for_test();
        std::cout<<"DRAFT_RING_UNCERTAIN_METADATA_COMPLETE retained=1 device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_REQUEST_METADATA");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3DraftHostRing::exercise_metadata_for_test();
        ninfer::exl3::Exl3VeriCacheRequest::exercise_request_metadata_for_test();
        std::cout<<"REQUEST_METADATA_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_EXACT_PAGE_EXTENSION");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3TextContext::exercise_exact_page_extension_for_test();
        std::cout<<"EXACT_PAGE_EXTENSION_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_RECURRENT_STORAGE_ACCOUNTING");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        RetainedDescriptorLedger metadata;
        auto page=make_bounded_shared<Exl3ExactKVPage>();page->k[0].resize(1024,19);
        const auto bytes=Exl3ExactKVPage::metadata_bytes();
        need(!Exl3ExactKVPage::attach_metadata_credit(page,metadata.acquire(bytes-1)) && metadata.bytes()==0,
            "KV page short metadata credit accepted or leaked");
        need(Exl3ExactKVPage::attach_metadata_credit(page,metadata.acquire(bytes)),"KV page exact credit refused");
        auto child=make_bounded_shared<Exl3ExactKVPage>(*page);
        need(!Exl3ExactKVPage::metadata_credit_belongs_to(child,metadata) && child->k[0]==page->k[0] &&
            child->k[0].data()!=page->k[0].data(),"KV page clone inherited credit or aliased payload");
        std::weak_ptr<Exl3ExactKVPage> weak=page;page.reset();
        need(weak.expired() && metadata.bytes()==bytes && child->k[0][0]==19,
            "KV page strong retirement lost weak charge or clone payload");
        weak.reset();need(metadata.bytes()==0,"KV page final weak metadata leaked");
        ninfer::exl3::Exl3ExactHostState::exercise_recurrent_storage_accounting_for_test();
        std::cout<<"RECURRENT_STORAGE_ACCOUNTING_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_RECURRENT_CONSTRUCTOR_METADATA");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        RetainedDescriptorLedger metadata;
        for(const auto bytes:{Exl3RecurrentSlab::physical_metadata_bytes()-1,Exl3RecurrentSlab::physical_metadata_bytes()+1}) {
            bool refused=false;try{auto invalid=Exl3RecurrentSlab::create(metadata.acquire(bytes));}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && metadata.bytes()==0,"recurrent constructor accepted wrong metadata extent");
        }
        auto owner=Exl3RecurrentSlab::create(metadata.acquire(Exl3RecurrentSlab::physical_metadata_bytes()));
        need(metadata.bytes()==Exl3RecurrentSlab::physical_metadata_bytes(),"recurrent constructor lost provisional metadata");
        std::weak_ptr<Exl3RecurrentSlab> weak=owner;owner.reset();
        need(weak.expired() && metadata.bytes()==Exl3RecurrentSlab::control_metadata_bytes(),
            "constructor metadata did not split object and final-weak control lifetime");
        weak.reset();need(metadata.bytes()==0,"recurrent constructor control credit leaked");
        owner=Exl3RecurrentSlab::create(metadata.acquire(Exl3RecurrentSlab::physical_metadata_bytes()));
        owner->release_constructor_credits_after_commit();
        need(metadata.bytes()==0,"recurrent commit retained provisional metadata charge");
        owner.reset();need(Exl3RecurrentSlab::quarantine_count.load()==0,"empty credited constructor quarantined storage");
        std::cout<<"RECURRENT_CONSTRUCTOR_METADATA_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_RECURRENT_GROWTH_ADMISSION");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        Exl3RecurrentExportPool pool;unsigned admissions=0;
        pool.set_growth_admission([&](const Exl3RecurrentSlabLayout& layout,const Exl3RecurrentExportPool::GrowthFactory&) -> std::shared_ptr<Exl3RecurrentSlab> {
            ++admissions;need(layout.bytes==32 && layout.offsets[1]==3 && layout.offsets.back()==8,
                "pool admission changed physical plane requirement");
            return {}; // Deliberately never invoke native allocation factory.
        });
        std::array<std::size_t,48> sizes{};sizes[0]=3;sizes[47]=5;
        need(!pool.acquire(sizes) && admissions==1 && pool.stats.allocations==0 && pool.stats.pool_bytes==0 && pool.stats.fallbacks==1,
            "refused recurrent admission published physical pool slot");
        pool.set_growth_admission([&](const auto&,const auto&) -> std::shared_ptr<Exl3RecurrentSlab> {
            ++admissions;throw Exl3ResourceReservationExhausted{};
        });
        need(!pool.acquire(sizes) && admissions==2 && pool.stats.allocations==0 && pool.stats.fallbacks==2,
            "reservation refusal did not preserve recurrent fallback");
        sizes[0]=std::numeric_limits<std::size_t>::max();
        need(!pool.acquire(sizes) && admissions==2 && pool.stats.pool_bytes==0,
            "invalid recurrent shape reached admission callback");
        // A malformed first construction consumes this admission's authority;
        // the second call must fail before reaching any native allocation.
        RetainedCudaRegistrationLedger registration;
        pool.set_growth_admission([&](const auto&,const Exl3RecurrentExportPool::GrowthFactory& factory) -> std::shared_ptr<Exl3RecurrentSlab> {
            ++admissions;bool malformed=false,repeated=false;
            try{(void)factory({registration.acquire(32),std::nullopt});}
            catch(const std::invalid_argument&){malformed=true;}
            try{(void)factory({});}
            catch(const std::logic_error& error){repeated=std::string_view(error.what())=="recurrent growth factory already consumed";}
            need(malformed && repeated && registration.bytes()==0,"recurrent admission reused failed factory authority");
            return {};
        });
        sizes[0]=3;
        need(!pool.acquire(sizes) && !pool.acquire(sizes) && admissions==4 && pool.stats.allocations==0,
            "recurrent single-use factory leaked authority across admission attempts");
        std::cout<<"RECURRENT_GROWTH_ADMISSION_COMPLETE native_operations=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_RECURRENT_SLAB_INVENTORY_HOOKS");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        auto owner=Exl3RecurrentSlab::create();
        RetainedDescriptorLedger metadata;RetainedCudaRegistrationLedger registration;
        need(!Exl3RecurrentSlab::attach_registration_credit(owner,registration.acquire(128)) && registration.bytes()==0,
            "unregistered slab accepted registration credit");
        bool refused=false;try{auto invalid=Exl3RecurrentSlab::resources(owner);}catch(const std::invalid_argument&){refused=true;}
        need(refused,"empty slab accepted physical inventory");
        need(!Exl3RecurrentSlab::attach_object_credit(owner,metadata.acquire(sizeof(Exl3RecurrentSlab)-1)) && metadata.bytes()==0,
            "short slab object metadata accepted");
        need(Exl3RecurrentSlab::attach_object_credit(owner,metadata.acquire(sizeof(Exl3RecurrentSlab))) &&
            Exl3RecurrentSlab::attach_control_credit(owner,metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes())),
            "slab metadata hooks rejected exact tickets");
        need(!Exl3RecurrentSlab::attach_control_credit(owner,metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes())) &&
            metadata.bytes()==Exl3RecurrentSlab::physical_metadata_bytes(),"duplicate slab control hook altered metadata ownership");
        std::weak_ptr<Exl3RecurrentSlab> weak=owner;owner.reset();
        need(weak.expired() && metadata.bytes()==Exl3RecurrentSlab::control_metadata_bytes(),
            "slab object/control metadata lifetimes were conflated");
        weak.reset();need(metadata.bytes()==0,"slab final weak control credit leaked");
        std::cout<<"RECURRENT_SLAB_INVENTORY_HOOKS_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_RECURRENT_SLAB_BORROWERS");mode && std::string_view(mode)=="1") {
        using ninfer::exl3::Exl3RecurrentSlab;
        const auto control_before=ninfer::exl3::Exl3SharedControlAccounting::live_bytes.load();
        ninfer::exl3::RetainedDescriptorLedger metadata;
        auto owner=Exl3RecurrentSlab::create();
        auto inventory_owner=owner;
        std::shared_ptr<Exl3RecurrentSlab> unowned(std::shared_ptr<Exl3RecurrentSlab>{},owner.get());
        ninfer::exl3::Exl3SharedControlCredit* refused_header=owner->physical_control;
        const auto owned_controls=ninfer::exl3::Exl3SharedControlAccounting::live_bytes.load();
        need(!Exl3RecurrentSlab::borrow(unowned,metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes()),&refused_header) &&
            !refused_header && !owner->borrowed.load() && metadata.bytes()==0 &&
            ninfer::exl3::Exl3SharedControlAccounting::live_bytes.load()==owned_controls,
            "nonowning slab alias created a borrower or retained credit");
        bool short_refused=false;
        try{auto invalid=Exl3RecurrentSlab::borrow(owner,metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes()-1));}
        catch(const std::invalid_argument&){short_refused=true;}
        need(short_refused && !owner->borrowed.load() && metadata.bytes()==0,"short borrower credit altered admission");
        bool preparation_failed=false;
        try {
            auto planned=Exl3RecurrentSlab::borrow(owner,metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes()));
            need(planned && owner->borrowed.load(),"planned export did not acquire a flight");
            throw std::bad_alloc(); // Later snapshot/page preparation, before producer submission.
        } catch(const std::bad_alloc&){preparation_failed=true;}
        need(preparation_failed && !owner->borrowed.load() && metadata.bytes()==0 &&
            ninfer::exl3::Exl3SharedControlAccounting::live_bytes.load()==owned_controls,
            "abandoned export planning retained borrower or control credit");
        auto first=Exl3RecurrentSlab::borrow(owner,metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes()));
        need(first && owner->borrowed.load() && first.get()==owner.get(),"slab borrower lost physical identity");
        need(!Exl3RecurrentSlab::borrow(owner),"slab admitted simultaneous writable borrowers");
        auto root=first;first.reset();
        need(owner->borrowed.load() && !Exl3RecurrentSlab::borrow(owner),"slab reused before final root release");
        std::weak_ptr<Exl3RecurrentSlab> weak_root=root;root.reset();
        need(weak_root.expired() && !owner->borrowed.load(),"slab final borrower did not release flight");
        need(metadata.bytes()==Exl3RecurrentSlab::control_metadata_bytes(),"borrower metadata retired before final weak owner");
        ninfer::exl3::Exl3SharedControlCredit* second_header=nullptr;
        auto second=Exl3RecurrentSlab::borrow(owner,std::nullopt,&second_header);
        std::shared_ptr<const void> allocation_owner(second,second_header);
        need(!Exl3RecurrentSlab::attach_borrower_control_credit(allocation_owner,
            metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes()-1)),"borrower inventory hook accepted short credit");
        need(Exl3RecurrentSlab::attach_borrower_control_credit(allocation_owner,
            metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes())),"borrower inventory hook rejected exact credit");
        need(!Exl3RecurrentSlab::attach_borrower_control_credit(allocation_owner,
            metadata.acquire(Exl3RecurrentSlab::control_metadata_bytes())) &&
            metadata.bytes()==2*Exl3RecurrentSlab::control_metadata_bytes(),"duplicate borrower hook changed retained credits");
        allocation_owner.reset();
        need(second && second.get()==inventory_owner.get(),"retained inventory owner prevented physical slab reuse");
        owner.reset();inventory_owner.reset();
        need(second->borrowed.load(),"borrower did not retain physical owner after pool retirement");
        auto surviving_owner=second;second.reset();surviving_owner.reset();weak_root.reset();
        need(metadata.bytes()==0 && ninfer::exl3::Exl3SharedControlAccounting::live_bytes.load()==control_before,
            "slab control blocks or final-weak metadata credits leaked");
        need(Exl3RecurrentSlab::quarantine_count.load()==0,"empty slab borrower test quarantined storage");
        std::cout<<"RECURRENT_SLAB_BORROWERS_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_RECURRENT_SLAB_CLEANUP");mode && std::string_view(mode)=="1") {
        using ninfer::exl3::Exl3RecurrentSlab;
        struct Cleanup {
            unsigned fail=0,unregisters=0,destroys=0,frees=0;
            int current(int* device) noexcept {*device=fail==2?4:3;return fail==1?71:0;}
            int drain() noexcept {return fail==3?73:0;}
            int wait(cudaEvent_t) noexcept {return fail==3?74:0;}
            int unregister(void*) noexcept {++unregisters;return fail==4?75:0;}
            int destroy(cudaEvent_t) noexcept {++destroys;return fail==5?76:0;}
            int free(void*) noexcept {++frees;return fail==6?77:0;}
        };
        need(Exl3RecurrentSlab::quarantine_count.load()==0,"slab cleanup fixture requires fresh process");
        int backing=0,event_identity=0;
        for(unsigned fault=0;fault<=6;++fault) {
            ninfer::exl3::RetainedHostAllocationLedger host_credit;
            ninfer::exl3::RetainedCudaRegistrationLedger registration_credit;
            ninfer::exl3::RetainedDescriptorLedger metadata_credit;
            auto* slab=new Exl3RecurrentSlab;
            slab->data=reinterpret_cast<float*>(&backing);slab->bytes=128;slab->device=3;
            slab->completed=reinterpret_cast<cudaEvent_t>(&event_identity);
            slab->registered=true;slab->recorded=true;
            need(!slab->attach_retirement_credits(host_credit.acquire(127),registration_credit.acquire(128),
                metadata_credit.acquire(sizeof(Exl3RecurrentSlab))) && host_credit.bytes()==0 &&
                registration_credit.bytes()==0 && metadata_credit.bytes()==0,"slab partial attachment leaked typed credits");
            need(slab->attach_retirement_credits(host_credit.acquire(128),registration_credit.acquire(128),
                metadata_credit.acquire(sizeof(Exl3RecurrentSlab))),"slab exact typed credit attachment failed");
            need(!slab->attach_retirement_credits(host_credit.acquire(128),registration_credit.acquire(128),
                metadata_credit.acquire(sizeof(Exl3RecurrentSlab))) && host_credit.bytes()==128 && registration_credit.bytes()==128 &&
                metadata_credit.bytes()==sizeof(Exl3RecurrentSlab),"slab duplicate attachment changed original credits");
            Cleanup cleanup;cleanup.fail=fault;
            Exl3RecurrentSlab::retire_with(slab,cleanup);
            if(!fault) {
                need(cleanup.unregisters==1 && cleanup.destroys==1 && cleanup.frees==1,
                    "successful slab cleanup omitted release stage");
                need(host_credit.bytes()==0 && registration_credit.bytes()==0 && metadata_credit.bytes()==0,
                    "successful slab cleanup retained typed credits");
                continue;
            }
            need(Exl3RecurrentSlab::quarantine_head.load()==slab && slab->cleanup_error!=0 && slab->poisoned,
                "failed slab cleanup lost physical owner");
            need(slab->data==reinterpret_cast<float*>(&backing),"failed slab cleanup freed host backing");
            need(cleanup.unregisters==(fault>=4?1u:0u) && cleanup.destroys==(fault>=5?1u:0u) &&
                cleanup.frees==(fault>=6?1u:0u),"slab cleanup continued past first failed stage");
            const auto count=Exl3RecurrentSlab::quarantine_count.load();
            Exl3RecurrentSlab::retire_with(slab,cleanup);
            need(Exl3RecurrentSlab::quarantine_count.load()==count,"slab failed cleanup republished or retried");
            need(slab->registered==(fault<=4) && (slab->completed!=nullptr)==(fault<=5),
                "slab cleanup forgot remaining registration/event obligations");
            need(host_credit.bytes()==128 && registration_credit.bytes()==(fault<=4?128u:0u) &&
                metadata_credit.bytes()==sizeof(Exl3RecurrentSlab),"slab cleanup released the wrong domain credit");
        }
        bool refused=false;try{auto slab=Exl3RecurrentSlab::create();}catch(const std::runtime_error&){refused=true;}
        need(refused,"slab quarantine admitted new physical owner");
        std::cout<<"RECURRENT_SLAB_CLEANUP_COMPLETE device_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_RECURRENT_SLAB_LAYOUT");mode && std::string_view(mode)=="1") {
        using ninfer::exl3::Exl3RecurrentSlabLayout;
        std::array<std::size_t,48> planes{};
        need(!Exl3RecurrentSlabLayout::derive(planes),"empty recurrent slab admitted");
        planes[0]=Exl3RecurrentSlabLayout::max_bytes/sizeof(float);
        const auto exact=Exl3RecurrentSlabLayout::derive(planes);
        need(exact && exact->bytes==Exl3RecurrentSlabLayout::max_bytes && exact->offsets[1]==planes[0] &&
            exact->offsets.back()==planes[0],"exact-fit recurrent layout rejected or truncated");
        const auto requirement=ninfer::exl3::Exl3RecurrentSlab::requirement(*exact);
        need(requirement.units[static_cast<unsigned>(ninfer::exl3::Exl3ResourceInventory::Domain::cuda_registered_host)]==exact->bytes &&
            requirement.units[static_cast<unsigned>(ninfer::exl3::Exl3ResourceInventory::Domain::host_metadata)]==
                ninfer::exl3::Exl3RecurrentSlab::physical_metadata_bytes(),"recurrent requirement omitted registration or physical metadata");
        auto malformed=*exact;malformed.offsets.back()=0;bool malformed_refused=false;
        try{(void)ninfer::exl3::Exl3RecurrentSlab::requirement(malformed);}catch(const std::invalid_argument&){malformed_refused=true;}
        need(malformed_refused,"recurrent requirement accepted inconsistent plane extent");
        planes[47]=1;need(!Exl3RecurrentSlabLayout::derive(planes),"recurrent aggregate cap overflow admitted");
        planes={};planes[0]=std::numeric_limits<std::size_t>::max();
        need(!Exl3RecurrentSlabLayout::derive(planes),"recurrent size overflow admitted");
        planes={};planes[0]=3;planes[47]=5;
        const auto first=Exl3RecurrentSlabLayout::derive(planes);
        planes[0]=4;planes[47]=4;
        const auto second=Exl3RecurrentSlabLayout::derive(planes);
        need(first && second && first->bytes==32 && second->bytes==32 && first->offsets!=second->offsets,
            "same-byte recurrent layouts lost plane identity");
        need(first->offsets[1]==3 && first->offsets[47]==3 && first->offsets[48]==8,
            "recurrent sparse plane offsets incorrect");
        std::cout<<"RECURRENT_SLAB_LAYOUT_COMPLETE pool_execution=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_LAYER_BUFFER_RETIREMENT");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        using Owner=Exl3LayerBufferRetirement;
        static unsigned releases=0;
        static bool fail=false;
        const auto cleanup=+[](void*,int device) noexcept -> int {
            ++releases;return device!=3?99:(fail?17:0);
        };
        RetainedDeviceLedger device;
        RetainedDescriptorLedger metadata;
        int storage=0;
        need(Owner::quarantined()==0,"layer retirement fixture requires fresh process");
        {
            Owner owner(cleanup);owner.adopt(&storage,128,3);
            bool refused=false;
            try{owner.attach(device.acquire(127),metadata.acquire(Owner::record_bytes()));}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && device.bytes()==0 && metadata.bytes()==0,"short layer ticket leaked credits");
            owner.attach(device.acquire(128),metadata.acquire(Owner::record_bytes()));
            refused=false;
            try{owner.attach(device.acquire(128),metadata.acquire(Owner::record_bytes()));}
            catch(const std::invalid_argument&){refused=true;}
            need(refused && device.bytes()==128 && metadata.bytes()==Owner::record_bytes(),"duplicate layer tickets changed ownership");
            owner.retire();owner.retire();
        }
        need(releases==1 && device.bytes()==0 && metadata.bytes()==0 && Owner::quarantined()==0,
            "successful layer retirement leaked or retried");
        fail=true;
        {
            Owner owner(cleanup);owner.adopt(&storage,128,3);
            owner.attach(device.acquire(128),metadata.acquire(Owner::record_bytes()));
            owner.retire();owner.retire();
        }
        const auto* retained=Owner::latest_for_test();
        need(releases==2 && Owner::quarantined()==1 && retained && retained->pointer==&storage &&
            retained->bytes==128 && retained->device==3 && retained->error==17 &&
            device.bytes()==128 && metadata.bytes()==Owner::record_bytes(),"failed layer retirement lost storage or credits");
        bool refused=false;try{Owner retry(cleanup);}catch(const std::runtime_error&){refused=true;}
        need(refused && releases==2,"layer quarantine allowed retry");
        std::cout<<"LAYER_BUFFER_RETIREMENT_COMPLETE engine_integration=pending numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_SHARED_WORKSPACE_UNWIND");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        const auto before=Exl3CudaLinearWorkspace::quarantined_workspaces();
        bool original=false;
        try {
            Exl3EngineTargetQ failed(false,false,false,false,false,false,false,false,false,2,true);
        } catch(const std::runtime_error& error) {
            original=std::string_view(error.what())=="injected shared projection startup allocation failure";
        }
        need(original && Exl3CudaLinearWorkspace::quarantined_workspaces()==before+1,
            "shared constructor unwind lost primary error or failed workspace owner");
        bool refused=false;try {Exl3EngineCore replacement(EngineOptions{});}
        catch(const std::runtime_error& error) {refused=std::string_view(error.what())==
            "EXL3 unresolved linear workspace retirement; reload refused";}
        need(refused,"shared constructor unwind allowed replacement Engine");
        std::cout << "SHARED_WORKSPACE_UNWIND_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_PACKED_STORAGE_RETIREMENT");mode &&
        (std::string_view(mode)=="1" || std::string_view(mode)=="partial" || std::string_view(mode)=="unwind")) {
        using namespace ninfer::exl3;
        need(!Exl3EngineTargetQ::packed_retirement_unresolved(),"packed retirement fixture requires fresh process");
        {
            Exl3EngineTargetQ control;
            const auto requirement=Exl3EngineTargetQ::requirement();
            need(control.host_metadata_bytes()==requirement.units[static_cast<unsigned>(
                    Exl3ResourceInventory::Domain::host_metadata)],
                "packed retirement record missing from startup metadata reservation");
        }
        need(!Exl3EngineTargetQ::packed_retirement_unresolved(),"normal packed retirement quarantined storage");
        {
            RetainedDescriptorLedger metadata;
            RetainedDeviceLedger device_credit;
            auto parent=make_bounded_shared<Exl3EngineTargetQ>();
            const auto device_bytes=parent->bytes();
            need(!Exl3EngineTargetQ::attach_device_retirement_credit(parent,device_credit.acquire(device_bytes-1)) &&
                device_credit.bytes()==0,"short shared device ticket retained partial child charge");
            need(!Exl3EngineTargetQ::attach_device_retirement_credit(parent,device_credit.acquire(device_bytes+1)) &&
                device_credit.bytes()==0,"oversized shared device ticket retained partial child charge");
            need(Exl3EngineTargetQ::attach_device_retirement_credit(parent,device_credit.acquire(parent->bytes())),
                "normal shared device lifetime attachment failed");
            need(!Exl3EngineTargetQ::attach_device_retirement_credit(parent,device_credit.acquire(device_bytes)) &&
                device_credit.bytes()==device_bytes,"duplicate shared device ticket changed original charge");
            const auto bytes=parent->host_metadata_bytes();
            std::weak_ptr<Exl3EngineTargetQ> weak=parent;
            need(!Exl3EngineTargetQ::attach_retirement_credit(parent,metadata.acquire(bytes-1)) &&
                metadata.bytes()==0 && parent->child_metadata_credit_bytes_for_test()==0 &&
                !bounded_retirement_credit_bytes_for_test<Exl3EngineTargetQ>(parent),
                "short parent ticket partially attached child metadata");
            need(!Exl3EngineTargetQ::attach_retirement_credit(parent,metadata.acquire(bytes+1)) &&
                metadata.bytes()==0 && parent->child_metadata_credit_bytes_for_test()==0,
                "oversized parent ticket mutated child credit slots");
            need(Exl3EngineTargetQ::attach_retirement_credit(parent,metadata.acquire(bytes)) &&
                metadata.bytes()==bytes,"exact parent ticket did not conserve metadata");
            const auto children=parent->child_metadata_credit_bytes_for_test();
            need(!Exl3EngineTargetQ::attach_retirement_credit(parent,metadata.acquire(bytes)) &&
                metadata.bytes()==bytes && parent->child_metadata_credit_bytes_for_test()==children,
                "duplicate parent ticket replaced or double charged children");
            parent.reset();
            need(device_credit.bytes()==0,"successful shared cleanup retained device credit");
            need(weak.expired() && metadata.bytes()==bounded_shared_allocation_bytes<Exl3EngineTargetQ>(),
                "successful child retirement retained metadata beyond weak parent control");
            weak.reset();
            need(metadata.bytes()==0,"successful final weak release retained metadata charge");
        }
        const bool partial=std::string_view(mode)=="partial";
        const bool unwind=std::string_view(mode)=="unwind";
        RetainedDescriptorLedger packed_metadata;
        RetainedDeviceLedger packed_device;
        std::weak_ptr<Exl3EngineTargetQ> retired_parent;
        bool original=false;
        try {
            if(unwind) {
                Exl3EngineTargetQ failed(false,false,false,false,false,false,false,false,false,4,false,1);
            } else {
                auto failed=make_bounded_shared<Exl3EngineTargetQ>(false,false,false,false,false,false,false,false,false,
                    0,false,partial?2:1);
                retired_parent=failed;
                need(Exl3EngineTargetQ::attach_device_retirement_credit(failed,packed_device.acquire(failed->bytes())),
                    "packed parent device attachment refused exact charge");
                need(Exl3EngineTargetQ::attach_retirement_credit(failed,packed_metadata.acquire(failed->host_metadata_bytes())),
                    "packed parent metadata attachment refused exact charge");
                failed.reset();
                need(retired_parent.expired() && packed_metadata.bytes()==
                    bounded_shared_allocation_bytes<Exl3EngineTargetQ>()+Exl3EngineTargetQ::packed_retirement_metadata_bytes(),
                    "packed quarantine lost child credit or retained successful workspace credit");
                retired_parent.reset();
                need(packed_metadata.bytes()==Exl3EngineTargetQ::packed_retirement_metadata_bytes(),
                    "final parent weak release dropped packed quarantine metadata");
                need(packed_device.bytes()==16ULL*5120*2+(partial?0:16ULL*12288*2),
                    "packed quarantine device charge did not match remaining allocations");
            }
        } catch(const std::runtime_error& error) {
            original=std::string_view(error.what())=="injected shared projection startup allocation failure";
            if(!original)throw;
        }
        need(original==unwind,"packed cleanup replaced original constructor exception");
        const auto retained=Exl3EngineTargetQ::packed_retirement_snapshot_for_test();
        int device=-1;need(cudaGetDevice(&device)==cudaSuccess,"packed fixture device observation failed");
        need(Exl3EngineTargetQ::packed_retirement_unresolved() && retained.device==device &&
            retained.error==static_cast<int>(cudaErrorUnknown) && retained.restore_error==0 &&
            retained.input_bytes==16ULL*5120*2 && retained.output_bytes==(partial?0:16ULL*12288*2),
            "packed retirement lost remaining storage, device or first cleanup failure");
        bool refused=false;try {Exl3EngineCore replacement(EngineOptions{});}
        catch(const std::runtime_error& error) {refused=std::string_view(error.what())==
            "EXL3 unresolved packed projection retirement; reload refused";}
        need(refused,"packed quarantine permitted replacement Engine admission");
        std::cout << "PACKED_STORAGE_RETIREMENT_COMPLETE numerical_coverage=0 full_engine_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_LINEAR_BORROWED_RETIREMENT");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        struct Backing {
            void* pointer=nullptr;
            explicit Backing(std::size_t bytes) {need(cudaMalloc(&pointer,bytes)==cudaSuccess,"borrowed fixture allocation");}
            ~Backing(){if(pointer)cudaFree(pointer);}
        };
        const auto required=Exl3LinearWorkspaceRequirements::derive(5120,12288,16);
        Backing transform(required.transformed_bytes),accumulation(required.accumulation_bytes);
        const Exl3CudaTransformView transform_view{static_cast<std::uint16_t*>(transform.pointer),required.transformed_bytes};
        const Exl3CudaAccumulationView accumulation_view{static_cast<float*>(accumulation.pointer),required.accumulation_bytes};
        const auto before=Exl3CudaLinearWorkspace::quarantine_footprint();
        need(before.has_value(),"borrowed fixture footprint unavailable");
        const auto quarantined=Exl3CudaLinearWorkspace::quarantined_workspaces();
        RetainedDeviceLedger device;
        {
            Exl3CudaLinearWorkspace::Owner borrowed=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16,
                false,false,false,false,false,false,false,accumulation_view,transform_view);
            need(borrowed->workspace_bytes()==0 && !borrowed->attach_device_credit(device.acquire(0)) && !device.bytes(),
                "fully borrowed workspace accepted owned device charge");
        }
        for(bool borrow_transform:{false,true}) {
            auto workspace=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16,
                false,false,false,false,false,false,false,
                borrow_transform?Exl3CudaAccumulationView{}:accumulation_view,
                borrow_transform?transform_view:Exl3CudaTransformView{});
            const auto owned=borrow_transform?required.accumulation_bytes:required.transformed_bytes;
            need(workspace->workspace_bytes()==owned,"mixed workspace charged borrowed allocation");
            need(!workspace->attach_device_credit(device.acquire(required.owned_bytes)),
                "mixed workspace accepted combined owned and borrowed charge");
            need(workspace->attach_device_credit(device.acquire(owned)),"mixed workspace refused exact owned charge");
            workspace->fail_owned_retirement_for_test();
            Exl3CudaLinearWorkspace::retire_owned(std::move(workspace));
            const auto retained=Exl3CudaLinearWorkspace::latest_retirement_for_test();
            need(retained.transform_retained==!borrow_transform && retained.accumulation_retained==borrow_transform,
                "mixed workspace quarantined borrowed allocation");
        }
        const auto after=Exl3CudaLinearWorkspace::quarantine_footprint();
        need(after && after->device_bytes==before->device_bytes+required.owned_bytes &&
            device.bytes()==required.owned_bytes && Exl3CudaLinearWorkspace::quarantined_workspaces()==quarantined+2,
            "mixed workspace lifetime conservation included borrowed bytes");
        need(cudaMemset(transform.pointer,0x31,required.transformed_bytes)==cudaSuccess &&
            cudaMemset(accumulation.pointer,0x52,required.accumulation_bytes)==cudaSuccess &&
            cudaDeviceSynchronize()==cudaSuccess,"workspace retirement invalidated external borrowed backing");
        return 0; // Deliberate quarantine: separate future process.
    }
    if(const auto* mode=std::getenv("NINFER_TEST_LINEAR_CONSTRUCTOR_RETIREMENT");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        const auto before=Exl3CudaLinearWorkspace::quarantined_workspaces();
        const auto footprint=Exl3CudaLinearWorkspace::quarantine_footprint();
        need(footprint.has_value(),"constructor retirement initial footprint unavailable");
        const auto required=Exl3LinearWorkspaceRequirements::derive(5120,12288,16);
        RetainedDeviceLedger constructor_device;
        RetainedDescriptorLedger constructor_metadata;
        for(bool fail_cleanup:{false,true}) {
            Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(fail_cleanup);
            bool failed=false;
            try{auto workspace=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16,
                false,false,false,false,false,false,false,Exl3CudaAccumulationView{},Exl3CudaTransformView{},
                false,false,false,constructor_device.acquire(required.owned_bytes),
                constructor_metadata.acquire(Exl3CudaLinearWorkspace::metadata_bytes()));}
            catch(const std::runtime_error&){failed=true;}
            need(failed && Exl3CudaLinearWorkspace::quarantined_workspaces()==before+(fail_cleanup?1:0),
                "constructor allocation failure lost cleanup disposition");
            need(constructor_device.bytes()==(fail_cleanup?required.transformed_bytes:0) &&
                constructor_metadata.bytes()==(fail_cleanup?Exl3CudaLinearWorkspace::storage_retirement_metadata_bytes():0),
                "linear constructor failed to retain only allocated transform and retirement-record credits");
        }
        const auto retained=Exl3CudaLinearWorkspace::latest_storage_retirement_for_test();
        const auto after=Exl3CudaLinearWorkspace::quarantine_footprint();
        need(retained.error==static_cast<int>(cudaErrorUnknown) && retained.restore_error==0 &&
            retained.transform_retained && !retained.accumulation_retained &&
            retained.transformed_bytes==required.transformed_bytes && !retained.accumulation_bytes &&
            after && after->device_bytes==footprint->device_bytes+required.transformed_bytes &&
            after->host_object_bytes==footprint->host_object_bytes+
                Exl3CudaLinearWorkspace::storage_retirement_metadata_bytes(),
            "failed constructor cleanup lost remaining storage record or byte charge");
        {
            auto raw=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16);
            raw->fail_owned_retirement_for_test(true);
        }
        const auto raw=Exl3CudaLinearWorkspace::latest_storage_retirement_for_test();
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==before+2 &&
            raw.transform_retained && !raw.accumulation_retained && raw.transformed_bytes==required.transformed_bytes,
            "raw workspace destructor lost partial-free retention");
        bool refused=false;
        try{Exl3EngineCore replacement(EngineOptions{});}catch(const std::runtime_error& error){
            refused=std::string_view(error.what())=="EXL3 unresolved linear workspace retirement; reload refused";
        }
        need(refused,"constructor storage quarantine allowed Engine admission");
        return 0; // Deliberate quarantine: separate future process.
    }
    if(const auto* mode=std::getenv("NINFER_TEST_DRAFT_GENERIC_RETIREMENT");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3Dflash2DraftModel::exercise_generic_retirement_for_test();
        std::cout<<"DRAFT_GENERIC_RETIREMENT_COMPLETE constructor_grants=pending\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_DRAFT_UNCERTAIN_SOURCE_RETIREMENT");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3Dflash2DraftModel::exercise_generic_retirement_for_test(true);
        std::cout<<"DRAFT_UNCERTAIN_SOURCE_RETIREMENT_COMPLETE retained=1\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_SHARED_CONSTRUCTOR_CREDITS");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3TextContext::exercise_shared_constructor_credits_for_test(1);
        ninfer::exl3::Exl3TextContext::exercise_shared_constructor_credits_for_test(2);
        std::cout<<"SHARED_CONSTRUCTOR_CREDITS_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_GENERIC_CONSTRUCTOR_CREDITS");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3TextContext::exercise_generic_constructor_credits_for_test(1);
        ninfer::exl3::Exl3TextContext::exercise_generic_constructor_credits_for_test(2);
        std::cout<<"GENERIC_CONSTRUCTOR_CREDITS_COMPLETE production_reservation_transfer=pending\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_LINEAR_OWNED_RETIREMENT");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        Exl3TextContext::exercise_generic_lifetime_credits_for_test();
        {
            using Allocator=Exl3ReconstructionControlAllocator<std::byte>;
            bool admitted=false;Exl3SharedControlCredit* credit=nullptr;
            Allocator allocator(&admitted,&credit);
            Exl3ReconstructionControlAllocator<std::uint64_t> rebound(allocator);
            need(allocator==rebound,"control allocator rebind lost admission/header identity");
            const auto baseline=Exl3SharedControlAccounting::live_bytes.load();
            for(const auto count:{std::size_t(0),Allocator::payload_capacity+1}) {
                bool refused=false;
                try{auto* unexpected=allocator.allocate(count);allocator.deallocate(unexpected,count);}
                catch(const std::bad_alloc&){refused=true;}
                need(refused && !admitted && !credit && Exl3SharedControlAccounting::live_bytes.load()==baseline,
                    "invalid control payload changed admission or allocated backing");
            }
            auto* payload=allocator.allocate(Allocator::payload_capacity);
            need(admitted && credit && reinterpret_cast<std::uintptr_t>(payload)%Allocator::alignment==0,
                "exact-capacity control payload misaligned or unpublished");
            payload[0]=std::byte{0x12};payload[Allocator::payload_capacity-1]=std::byte{0x34};
            need(!credit->ticket,"control payload overwrote credit header");
            bool duplicate_refused=false;
            try{auto* unexpected=rebound.allocate(1);rebound.deallocate(unexpected,1);}
            catch(const std::bad_alloc&){duplicate_refused=true;}
            need(duplicate_refused,"rebound allocator bypassed single-block admission");
            Allocator copy(rebound);copy.deallocate(payload,Allocator::payload_capacity);
            need(Exl3SharedControlAccounting::live_bytes.load()==baseline,"rebound deallocation lost backing balance");
        }
        {
            RetainedDescriptorLedger ledger;
            Exl3SharedControlCredit* credit=nullptr;
            const auto baseline=Exl3SharedControlAccounting::live_bytes.load();
            auto owner=[&] {
                bool admitted=false;
                auto result=std::shared_ptr<int>(new int(7),std::default_delete<int>(),
                    Exl3ReconstructionControlAllocator<std::byte>(&admitted,&credit));
                need(admitted,"control allocator omitted admission");return result;
            }(); // Construction admission storage is already dead at both release edges.
            need(credit && !credit->ticket,"control allocator omitted fresh credit record");
            credit->ticket.emplace(ledger.acquire(Exl3ReconstructionControlAllocator<std::byte>::capacity));
            std::weak_ptr<int> weak=owner;owner.reset();
            need(weak.expired() && ledger.bytes()==Exl3ReconstructionControlAllocator<std::byte>::capacity,
                "control metadata released before final weak owner");
            weak.reset();
            need(ledger.bytes()==0 && Exl3SharedControlAccounting::live_bytes.load()==baseline,
                "final weak owner failed to release control metadata");
        }
        const auto context_witness_before=Exl3TextContext::retirement_quarantine_witness();
        const auto before=Exl3CudaLinearWorkspace::quarantined_workspaces();
        const auto footprint_before=Exl3CudaLinearWorkspace::quarantine_footprint();
        need(footprint_before.has_value(),"initial linear quarantine footprint overflow");
        const auto expected=Exl3LinearWorkspaceRequirements::derive(5120,12288,16);
        RetainedDescriptorLedger workspace_metadata;
        RetainedDeviceLedger workspace_device;
        {
            Exl3CudaLinearWorkspace::Owner converted=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16);
            need(converted->attach_metadata_credit(workspace_metadata.acquire(Exl3CudaLinearWorkspace::metadata_bytes())),
                "workspace metadata credit attachment refused exact charge");
            converted=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16);
            need(workspace_metadata.bytes()==0,"successful workspace replacement retained metadata credit");
            auto moved=std::move(converted);
            need(!converted && bool(moved),"retained workspace factory conversion lost unique ownership");
        }
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==before,
            "retained workspace replacement/destruction failed normal cleanup");
        need(Exl3TextContext::retirement_quarantine_witness()==context_witness_before,
            "successful linear cleanup changed context startup quarantine witness");
        auto control=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16);
        auto* control_slot=control.release();
        Exl3CudaLinearWorkspace::retire_slot(control_slot);
        need(!control && !control_slot && Exl3CudaLinearWorkspace::quarantined_workspaces()==before,
            "idle workspace control failed normal retirement");
        auto failed=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16);
        failed->fail_owned_retirement_for_test();
        need(failed->attach_device_credit(workspace_device.acquire(expected.owned_bytes)),
            "failed workspace fixture missing device lifetime charge");
        need(failed->attach_metadata_credit(workspace_metadata.acquire(Exl3CudaLinearWorkspace::metadata_bytes())),
            "failed workspace fixture missing metadata charge");
        auto* failed_slot=failed.release();
        Exl3CudaLinearWorkspace::retire_slot(failed_slot);
        need(!failed && !failed_slot && Exl3CudaLinearWorkspace::quarantined_workspaces()==before+1,
            "failed owned workspace was not retained exactly once");
        auto expected_context_witness=context_witness_before;
        ++expected_context_witness.back();
        need(Exl3TextContext::retirement_quarantine_witness()==expected_context_witness,
            "failed linear cleanup omitted or corrupted context startup quarantine witness");
        need(workspace_metadata.bytes()==Exl3CudaLinearWorkspace::metadata_bytes(),
            "quarantined workspace released child metadata credit");
        int current_device=-1;
        need(cudaGetDevice(&current_device)==cudaSuccess,"linear retirement fixture device observation failed");
        const auto retained=Exl3CudaLinearWorkspace::latest_retirement_for_test();
        need(retained.device==current_device && retained.error==static_cast<int>(cudaErrorUnknown) &&
            retained.restore_error==0 && retained.transform_retained && retained.accumulation_retained,
            "linear retirement lost device/error or outstanding allocation identity");
        need(retained.transformed_bytes==expected.transformed_bytes &&
            retained.accumulation_bytes==expected.accumulation_bytes,
            "linear retirement lost retained allocation extents");
        Exl3CudaLinearWorkspace::retire_slot(failed_slot);
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==before+1,
            "empty workspace retirement duplicated quarantine");
        auto partial=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16);
        need(partial->attach_device_credit(workspace_device.acquire(expected.owned_bytes)),
            "partial workspace fixture missing device lifetime charge");
        partial->fail_owned_retirement_for_test(true);
        Exl3CudaLinearWorkspace::retire_owned(std::move(partial));
        const auto partially_retained=Exl3CudaLinearWorkspace::latest_retirement_for_test();
        need(workspace_device.bytes()==expected.owned_bytes+expected.transformed_bytes,
            "partial workspace cleanup released retained device charge or retained freed charge");
        need(partially_retained.transform_retained && !partially_retained.accumulation_retained &&
            partially_retained.transformed_bytes==expected.transformed_bytes &&
            partially_retained.accumulation_bytes==0,
            "partial linear retirement retained or charged an already freed accumulation");
        const auto footprint=Exl3CudaLinearWorkspace::quarantine_footprint();
        need(footprint && footprint->device_bytes==footprint_before->device_bytes+
                expected.owned_bytes+expected.transformed_bytes &&
            footprint->host_object_bytes==footprint_before->host_object_bytes+
                2*Exl3CudaLinearWorkspace::metadata_bytes() &&
            Exl3CudaLinearWorkspace::quarantined_workspaces()==before+2,
            "linear quarantine footprint did not conserve remaining physical allocations");
        bool unwound=false;
        try {
            Exl3CudaLinearWorkspace::Owner pending=std::make_unique<Exl3CudaLinearWorkspace>(5120,12288,16);
            pending->fail_owned_retirement_for_test();
            throw std::runtime_error("prepared enclosing-owner allocation failure");
        } catch(const std::runtime_error& error) {
            unwound=std::string_view(error.what())=="prepared enclosing-owner allocation failure";
        }
        const auto unwound_footprint=Exl3CudaLinearWorkspace::quarantine_footprint();
        need(unwound && Exl3CudaLinearWorkspace::quarantined_workspaces()==before+3 &&
            unwound_footprint && unwound_footprint->device_bytes==footprint->device_bytes+expected.owned_bytes &&
            unwound_footprint->host_object_bytes==footprint->host_object_bytes+Exl3CudaLinearWorkspace::metadata_bytes(),
            "enclosing-owner unwind lost failed workspace or double-counted retention");
        bool refused=false;try {Exl3EngineCore replacement(EngineOptions{});}
        catch(const std::runtime_error& error) {refused=std::string_view(error.what())==
            "EXL3 unresolved linear workspace retirement; reload refused";}
        need(refused,"quarantined linear workspace permitted Engine startup");
        std::cout << "LINEAR_OWNED_RETIREMENT_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_LINEAR_TRANSFORMED_PREFLIGHT");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        // Construction configures CUDA, but invalid forward inputs must fail
        // before any launch/dereference of these host-only sentinel addresses.
        alignas(16) std::array<std::byte,3072> storage{};
        auto* base=reinterpret_cast<std::uint16_t*>(storage.data());
        auto* accum=reinterpret_cast<float*>(storage.data()+512);
        Exl3CudaLinearWorkspace workspace(128,128,1,false,false,false,false,false,false,false,
            {accum,2560},{base,256});
        Exl3CudaLinearMetadata metadata{128,128,5,false,true,false};
        std::int32_t multiplier=1;
        Exl3CudaLinearWeights weights{base,base,base,&multiplier};
        for(unsigned fault=0;fault<3;++fault) {
            const auto* input=fault==0?base+1:fault==1?reinterpret_cast<std::uint16_t*>(accum):
                reinterpret_cast<std::uint16_t*>(std::numeric_limits<std::uintptr_t>::max()-127);
            bool refused=false;try {workspace.forward_from_transformed(weights,metadata,input,base,1,nullptr);}
            catch(const std::exception& error) {refused=std::string_view(error.what())==
                (fault==0?"EXL3 borrowed workspace storage alignment":fault==1?
                 "EXL3 borrowed transform/accumulation overlap":"EXL3 borrowed workspace address overflow");}
            need(refused,"pretransformed dispatch bypassed workspace storage admission");
        }
        std::cout << "LINEAR_TRANSFORMED_PREFLIGHT_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_LINEAR_BORROWED_PREFLIGHT");mode && std::string_view(mode)=="1") {
        alignas(16) std::array<std::byte,3072> storage{};
        for(unsigned fault=0;fault<6;++fault) {
            auto* transformed=storage.data();auto* accumulation=storage.data()+512;
            if(fault==0)accumulation=storage.data();
            if(fault==1)transformed=storage.data()+1;
            if(fault==2)accumulation=storage.data()+513;
            if(fault==3)transformed=reinterpret_cast<std::byte*>(std::numeric_limits<std::uintptr_t>::max()-127);
            if(fault==4)transformed=storage.data()+2;
            if(fault==5)transformed=storage.data()+8;
            bool refused=false;
            try {
                ninfer::exl3::Exl3CudaLinearWorkspace invalid(128,128,1,false,false,false,false,false,false,false,
                    {reinterpret_cast<float*>(accumulation),2560},{reinterpret_cast<std::uint16_t*>(transformed),256});
            } catch(const std::exception& error) {
                const std::string_view expected=fault==0?"EXL3 borrowed transform/accumulation overlap":
                    fault==3?"EXL3 borrowed workspace address overflow":"EXL3 borrowed workspace storage alignment";
                refused=std::string_view(error.what())==expected;
            }
            need(refused,"real linear constructor did not refuse borrowed geometry before CUDA setup");
        }
        std::cout << "LINEAR_BORROWED_PREFLIGHT_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_HOST_QUARANTINE");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        using Inventory=Exl3ResourceInventory;
        struct Backing {mutable unsigned calls=0;};
        std::weak_ptr<Backing> weak;
        {
            Exl3HostResidentSet registry(1,1ULL<<30);
            Inventory::Requirement required;required.configuration=22;
            const auto bytes=bounded_shared_allocation_bytes<Backing>();
            required.add(Inventory::Domain::host_metadata,1,bytes);
            registry.allocate_reserved(required,[&](auto) {
                auto owner=make_bounded_shared<Backing>();weak=owner;
                Inventory actual;actual.add({owner,0,Inventory::Domain::host_metadata,bytes,{},
                    +[](const std::shared_ptr<const void>& value,RetainedDescriptorLedger::Ticket) noexcept {
                        ++static_cast<const Backing*>(value.get())->calls;return false;
                    }});return actual;
            });
        }
        need(Exl3HostResidentSet::retirement_quarantined() && !weak.expired(),
            "host quarantine fixture failed to retain owner");
        for(unsigned attempt=0;attempt<2;++attempt) {
            bool refused=false;
            // Empty options are intentional: the seal must reject before Impl,
            // model/package setup or any device initialization is attempted.
            try {Exl3EngineCore replacement(EngineOptions{});}
            catch(const std::runtime_error& error) {refused=std::string_view(error.what())==
                "EXL3 unresolved host residency retirement; reload refused";}
            auto survivor=weak.lock();
            need(refused && survivor && survivor->calls==1,
                "Engine startup bypassed host quarantine or replayed retirement hook");
        }
        std::cout << "ENGINE_HOST_QUARANTINE_COMPLETE reload_refused=2 full_engine_coverage=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_KV_REGISTRATION_PREFLIGHT");mode && std::string(mode)=="1") {
        prepared_kv_registration_preflight();
        if(ninfer::exl3::Exl3RegisteredKVBacking::quarantined_bytes() ||
            ninfer::exl3::Exl3KVTransferLease::quarantined_records() ||
            ninfer::exl3::Exl3DevicePageStorage::quarantined_bytes() ||
            ninfer::exl3::Exl3DevicePageFill::quarantined_records() ||
            ninfer::exl3::Exl3AttentionStageStorage::quarantined_records()) {
            bool refused=false;
            try{ninfer::exl3::Exl3EngineCore replacement(EngineOptions{});}
            catch(const std::runtime_error& error){refused=std::string(error.what()).find("unresolved KV retirement")!=std::string::npos;}
            need(refused,"quarantined registration allowed Engine startup");
        }
        return 0;
    }
    const auto* target=std::getenv("NINFER_EXL3_TARGET_PATH");const auto* draft=std::getenv("NINFER_EXL3_DFLASH2_PATH");
    if(!target || !draft)return 77;
    if(const auto* mode=std::getenv("NINFER_TEST_DEVICE_LOGICAL_MODEL");
       mode && std::string_view(mode)=="1") {
        run_device_logical_model_contract(target,draft);
        std::cout<<"DEVICE_LOGICAL_MODEL_COMPLETE\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_PREFIX_CONSTRUCTOR_CREDITS");mode && *mode) {
        using namespace ninfer::exl3;using Cache=Exl3DevicePrefixCache;
        const std::string_view selector(mode);
        need(selector.size()==1 && selector[0]>='1' && selector[0]<='7',"prefix constructor fault selector1..7");
        const unsigned fault=selector[0]-'0';const bool retain=(fault>=3 && fault<=5) || fault==7;
        const auto before=Cache::budget_snapshot();need(before[1]==0,"prefix credit fixture requires fresh process");
        const auto control_attempts=Cache::control_fault_attempts_for_test();
        Exl3VeriCacheServingIdentity identity{"prefix-credit-failure","weights","tokens","exact","text"};
        Exl3VeriCacheServingPrefixCache roots({2,64,2ULL<<30,8ULL<<30},identity);
        bool original=false;
        {
            Exl3VeriCacheServingCoordinator authority(roots,{1,1,16ULL<<30,8ULL<<30});
            authority.bind_physical_resources({},Exl3ResourceInventory::unlimited());
            const auto control_bytes=Exl3SharedControlAccounting::live_bytes.load();
            try{(void)Cache::create_reserved(authority,4096,fault);}
            catch(const std::bad_alloc&){original=fault>=6;}
            catch(const std::exception& error) {
                original=std::string_view(error.what())==(fault==1?"injected prefix startup precommit failure":
                    "resource reservation actual extent/domain mismatch");
            }
            need(original,"prefix construction failed to reach intended commit boundary or replaced original error");
            need(Cache::control_fault_attempts_for_test()==control_attempts+(fault>=6?1:0),
                "prefix control fixture did not reach intended allocator failure boundary");
            need(Exl3SharedControlAccounting::live_bytes.load()==control_bytes,
                "prefix failed startup retained a nonexistent or unreferenced control block");
            authority.close();
        }
        const auto after=Cache::budget_snapshot();
        if(retain) {
            const auto retained=Cache::retirement_snapshot_for_test();
            const auto extent=Cache::allocation_bytes_required(4096);
            need(retained.pointer && retained.bytes==extent && retained.device_credit==extent &&
                retained.metadata_credit==Cache::retirement_metadata_bytes(),
                "prefix failed cleanup lost exact device or surviving record metadata credits");
            const int expected=(fault==3 || fault==7)?cudaErrorUnknown:fault==4?cudaErrorInitializationError:cudaErrorInvalidDevice;
            need(retained.error==expected && after[0]==before[0]+extent && after[1]==before[1]+extent,
                "prefix failed cleanup lost provider status or physical budget");
            bool refused=false;try{auto reload=Cache::create();}
            catch(const std::runtime_error& error){refused=std::string_view(error.what())=="unresolved device prefix cleanup";}
            need(refused,"prefix constructor bypassed retained failure");
        } else need(after==before,"clean prefix startup unwind retained physical charge");
        std::cout<<"PREFIX_CONSTRUCTOR_CREDITS_COMPLETE fault="<<fault<<'\n';return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_PREFIX_CONTROL_CREDITS");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;using Cache=Exl3DevicePrefixCache;
        const auto controls=Exl3SharedControlAccounting::live_bytes.load();
        const auto budget=Cache::budget_snapshot();RetainedDescriptorLedger metadata;
        auto owner=Cache::create();need(owner->admitted(),"prefix control fixture NOT_EXERCISED: physical admission declined");
        const auto required=Cache::owner_metadata_bytes_required(4096);
        need(owner->owner_metadata_bytes()==required &&
            Exl3SharedControlAccounting::live_bytes.load()==controls+Cache::control_metadata_bytes(),
            "prefix control requirement differs from physical block");
        need(!Cache::attach_metadata_credit(owner,metadata.acquire(required-1)) && metadata.bytes()==0,
            "short prefix control credit accepted");
        need(Cache::attach_metadata_credit(owner,metadata.acquire(required)),"prefix exact metadata credit refused");
        need(!Cache::attach_metadata_credit(owner,metadata.acquire(required)) && metadata.bytes()==required,
            "duplicate prefix metadata changed retained charge");
        std::weak_ptr<Cache> weak=owner;
        {auto use=owner->try_use();need(bool(use),"bounded prefix control lost shared_from_this identity");use.complete();}
        need(owner->retire_pristine(),"prefix control fixture pristine retirement failed");
        owner.reset();
        need(weak.expired() && metadata.bytes()==Cache::control_metadata_bytes() && Cache::budget_snapshot()==budget,
            "prefix object lifetime released final-weak control credit or retained freed device");
        need(Exl3SharedControlAccounting::live_bytes.load()==controls+Cache::control_metadata_bytes(),
            "prefix weak survivor lost physical control block");
        weak.reset();need(metadata.bytes()==0 && Exl3SharedControlAccounting::live_bytes.load()==controls,
            "prefix final weak release leaked control metadata");
        std::cout<<"PREFIX_CONTROL_CREDITS_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_PREFIX_RESERVED_STARTUP");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        using Inventory=Exl3ResourceInventory;
        using Cache=Exl3DevicePrefixCache;
        Exl3VeriCacheServingIdentity identity{"prefix-credit","weights","tokens","exact","text"};
        Exl3VeriCacheServingPrefixCache roots({2,64,2ULL<<30,8ULL<<30},identity);
        const auto before=Cache::budget_snapshot();
        for(const auto domain:{Inventory::Domain::device,Inventory::Domain::host_metadata}) {
            Exl3VeriCacheServingCoordinator authority(roots,{1,1,16ULL<<30,8ULL<<30});
            auto limits=Inventory::unlimited();
            limits[static_cast<unsigned>(domain)]=0;
            authority.bind_physical_resources({},limits);
            need(!Cache::create_reserved(authority,4096),"optional prefix ignored exhausted domain");
            need(!Cache::create_reserved(authority,4096),"optional prefix refusal changed retry behavior");
            need(Cache::budget_snapshot()==before,"credit refusal reached physical prefix allocation");
            authority.close();
        }
        Exl3VeriCacheServingCoordinator authority(roots,{1,1,16ULL<<30,8ULL<<30});
        auto limits=Inventory::unlimited();
        limits[static_cast<unsigned>(Inventory::Domain::device)]=4096ULL*1024*2*32;
        limits[static_cast<unsigned>(Inventory::Domain::host_metadata)]=Cache::owner_metadata_bytes_required(4096);
        authority.bind_physical_resources({},limits);
        auto cache=Cache::create_reserved(authority,4096);
        need(bool(cache),"prefix exact-fit ownership case NOT_EXERCISED: physical admission declined");
        need(cache->allocation_bytes()==4096ULL*1024*2*32,"prefix actual allocation extent mismatch");
        need(cache->attached_credits_for_test()==std::array<std::uint64_t,2>{0,0},
            "committed prefix retained provisional constructor credits");
        for(int bank:{-1,16,2147483647}) {
            bool refused=false;
            try{(void)cache->plane(bank,true);}
            catch(const std::invalid_argument&){refused=true;}
            need(refused,"prefix plane accepted invalid bank");
        }
        const auto first_plane=reinterpret_cast<std::uintptr_t>(cache->plane(0,true));
        const auto last_plane=reinterpret_cast<std::uintptr_t>(cache->plane(15,false));
        need(last_plane-first_plane==31ULL*4096*1024*2,"prefix plane layout crossed allocation extent");
        const std::vector<std::shared_ptr<const Exl3ExactKVPage>> missing_pages(64);
        cache->publish_tags(missing_pages,4096);
        need(cache->matched_rows(missing_pages,4096)==0,"missing prefix pages became represented hits");
        cache->publish_tags({},4096);
        cache->invalidate();
        need(cache->matched_rows(missing_pages,4096)==0,"empty publication or invalidation retained missing tags");
        {
            // Tag identity fixture only: no plane copy or numerical cache hit is
            // claimed by these page shells.
            const auto page=[](int first,int rows) {
                auto result=std::make_shared<Exl3ExactKVPage>();
                result->first=first;result->rows=rows;
                return std::shared_ptr<const Exl3ExactKVPage>(std::move(result));
            };
            std::vector<std::shared_ptr<const Exl3ExactKVPage>> history{page(0,64),page(64,64),page(128,31)};
            cache->publish_tags(history,159);
            need(cache->matched_rows(history,159)==128,"prefix tag match admitted partial page or lost full pages");
            need(cache->matched_rows(history,127)==64,"prefix tag match crossed published position");
            auto branch=history;branch[1]=page(64,64);
            need(cache->matched_rows(branch,159)==64,"equal geometry replaced page identity proof");
            cache->publish_tags(history,64);
            need(cache->matched_rows(history,159)==64,"shortened publication retained stale suffix tags");
            cache->publish_tags(history,159);
            std::weak_ptr<const Exl3ExactKVPage> weak=history[0];
            branch.clear();history.clear();
            need(weak.expired(),"prefix tags retained authoritative page owner");
            std::vector<std::shared_ptr<const Exl3ExactKVPage>> replacement{page(0,64)};
            need(cache->matched_rows(replacement,64)==0,"expired tag matched replacement page");
            cache->publish_tags(replacement,64);
            cache->invalidate();
            need(cache->matched_rows(replacement,64)==0,"invalidation retained positive page tag");
        }
        std::weak_ptr<Cache> retained=cache;
        {
            auto use=cache->try_use();
            need(bool(use),"reserved cache refused initial lease");
            use.complete();
            need(!bool(use),"completed prefix lease still authorized use");
        }
        // This fixture has submitted no work using the cache planes.
        cache->retire_after_device_drain();
        {
            auto use=cache->try_use();
            need(!bool(use),"retired prefix cache granted lease");
            use.complete();
        }
        bool retired_plane=false;
        try{(void)cache->plane(0,true);}
        catch(const std::logic_error&){retired_plane=true;}
        need(retired_plane,"retired prefix cache exposed plane storage");
        cache.reset();
        need(!retained.expired(),"prefix credit did not retain cache owner");
        authority.close();
        need(retained.expired(),"closed prefix credit retained unused cache");
        need(Cache::budget_snapshot()==before,"closed prefix credit leaked physical budget");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_SHARED_PREFIX_LEASE");mode && std::string(mode)=="1") {
        using Cache=ninfer::exl3::Exl3DevicePrefixCache;
        const auto before=Cache::budget_snapshot();
        auto cache=Cache::create();std::weak_ptr<Cache> weak=cache;
        {
            auto reader=cache->try_use();need(bool(reader),"first prefix reader refused");
            bool peer_refused=false;
            std::thread peer([&]{auto contender=cache->try_use();peer_refused=!bool(contender);});peer.join();
            need(peer_refused,"shared prefix waited or granted simultaneous mutable fill");
            cache.reset();need(!weak.expired(),"prefix reader did not retain allocation owner");
            reader.complete();
        }
        need(weak.expired(),"completed reader retained unused cache");
        cache=Cache::create();
        auto tag=std::make_shared<ninfer::exl3::Exl3ExactKVPage>();tag->first=0;tag->rows=64;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3ExactKVPage>> tagged{tag};
        cache->publish_tags(tagged,64);
        need(cache->matched_rows(tagged,64)==64,"uncertain-cache fixture lacked initial tag hit");
        {auto failed=cache->try_use();need(bool(failed),"new prefix cache unavailable");}
        {auto retry=cache->try_use();need(!bool(retry),"uncertain prefix consumer became reusable");retry.complete();}
        need(cache->matched_rows(tagged,64)==0,"uncertain cache exposed stale prefix hit");
        cache->publish_tags(tagged,64);
        need(cache->matched_rows(tagged,64)==0,"tag publication revived uncertain cache");
        bool uncertain_plane=false;
        try{(void)cache->plane(0,true);}
        catch(const std::logic_error&){uncertain_plane=true;}
        need(uncertain_plane,"uncertain cache exposed device plane");
        need(cudaDeviceSynchronize()==cudaSuccess,"prepared prefix owner drain");
        cache->retire_after_device_drain();cache.reset();
        need(Cache::budget_snapshot()==before,"retired prefix owner credit not returned");
        return 0;
    }
    EngineOptions opt;opt.exl3_package=PinnedExl3PackageOptions{target,draft,{}};
    // This qualification binary binds the current exact-host wide-prefill
    // workspace, whose admitted construction-time row capacity is 32.
    opt.prefill_chunk=32;
    const bool device_prefix=std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX")&&std::string(std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX"))=="1";
    const bool prefix16k=std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS")&&std::string(std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS"))=="16384";
    opt.max_context=prefix16k?16640:device_prefix?4352:1024;opt.kv_capacity=KvCapacityPolicy::explicit_capacity(opt.max_context);
    opt.use_cuda_graph=false;opt.speculative={SpeculativeBackend::DFlash2,7,ProposalHead::Full};
    opt.max_pending_requests=2;opt.context_cache.max_shared_prefixes=4;
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_REQUEST_AFFINITY");
       mode && std::string_view(mode)=="1") {
        const auto* affinity=std::getenv("NINFER_EXL3_REQUEST_AFFINITY");
        const auto* preserve=std::getenv("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT");
        need(affinity && std::string_view(affinity)=="1" &&
            preserve && std::string_view(preserve)=="1",
            "request affinity fixture requires affinity and preservation routes");
        opt.max_concurrency=2;opt.context_cache.enabled=true;
        opt.max_pending_requests=2;opt.context_cache.max_shared_prefixes=4;
        ninfer::exl3::Exl3EngineCore owner(opt);
        std::mutex assignment_mutex;
        std::vector<std::pair<std::size_t,bool>> assignments;
        owner.observe_lane_assignment_for_test([&](std::size_t lane,bool affine) {
            std::lock_guard lock(assignment_mutex);assignments.emplace_back(lane,affine);
        });
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=0;request.execution.allow_prefix_reuse=true;
        request.stop=owner.frontend().default_stop_policy();
        const auto submit_tokens=[&](const std::vector<TokenId>& tokens) {
            auto prepared=owner.frontend().prepare_tokens(tokens);const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        const auto finish=[&](auto submission) {
            const auto result=submission.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            need(result.finish_reason==FinishReason::OutputLimit && result.generated_token_ids.empty(),
                "request affinity zero-output control changed result");
        };

        std::vector<TokenId> growing(96,198);
        finish(submit_tokens(growing));
        need(assignments.size()==1 && !assignments[0].second,
            "cold request invented lane affinity");
        const auto seeded_lane=assignments[0].first;
        growing.push_back(12050);
        finish(submit_tokens(growing));
        need(assignments.size()==2 && assignments[1]==std::pair{seeded_lane,true},
            "free exact-resident lane was not selected for growing input");

        owner.stale_lane_affinity_for_test(seeded_lane);
        growing.push_back(13);
        finish(submit_tokens(growing));
        need(assignments.size()==3 && !assignments[2].second,
            "stale context generation authorized affinity");
        const auto current_lane=assignments[2].first;

        struct Gate {
            std::mutex mutex;std::condition_variable changed;
            bool local_entered=false,peer_assigned=false,release=false;
            std::size_t local_lane=2,peer_lane=2;bool peer_affine=true;
        } gate;
        owner.observe_lane_assignment_for_test([&](std::size_t lane,bool affine) {
            std::unique_lock lock(gate.mutex);
            if(!gate.local_entered) {
                gate.local_entered=true;gate.local_lane=lane;
                need(affine,"fairness control did not select its idle exact-resident lane");
                gate.changed.notify_all();
                need(gate.changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate.release;}),
                    "request affinity fairness release timeout");
            } else {
                gate.peer_assigned=true;gate.peer_lane=lane;gate.peer_affine=affine;
                gate.changed.notify_all();
            }
        });
        growing.push_back(198);
        auto local=submit_tokens(growing);
        {
            std::unique_lock lock(gate.mutex);
            need(gate.changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate.local_entered;}),
                "request affinity local assignment timeout");
        }
        auto peer=submit_tokens(growing);
        {
            std::unique_lock lock(gate.mutex);
            need(gate.changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate.peer_assigned;}),
                "request affinity free-lane assignment timeout");
            need(gate.local_lane==current_lane && gate.peer_lane!=gate.local_lane && !gate.peer_affine,
                "busy local lane overrode FIFO progress on a free lane");
            gate.release=true;gate.changed.notify_all();
        }
        finish(std::move(peer));finish(std::move(local));

        owner.observe_lane_assignment_for_test([&](std::size_t lane,bool affine) {
            std::lock_guard lock(assignment_mutex);assignments.emplace_back(lane,affine);
        });
        auto incompatible=growing;incompatible.front()=12050;
        finish(submit_tokens(incompatible));
        need(assignments.size()==4 && !assignments.back().second,
            "incompatible token root authorized lane affinity");
        const auto stats=owner.runtime_stats();
        need(stats.request_affinity_assignments>=2 && stats.request_affinity_fallbacks>=3 &&
            stats.acquired_payload_preservations>=1 && stats.acquired_draft_ring_preservations>=1,
            "request affinity accounting or exact target/draft preservation path not exercised");
        owner.observe_lane_assignment_for_test({});
        need(owner.close().reusable(),"request affinity Engine retirement");
        std::cout<<"ENGINE_REQUEST_AFFINITY_COMPLETE exact_local=1 stale_refused=1"
            <<" busy_fallback=1 incompatible_refused=1 assignments="
            <<stats.request_affinity_assignments<<" fallbacks="<<stats.request_affinity_fallbacks<<'\n';
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_MIXED_ARRIVALS");
       mode && std::string_view(mode)=="1") {
        using Clock=std::chrono::steady_clock;
        using Root=std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>;
        // The production EXL3 constructor supports physical C1/C2 only. This
        // scenario uses physical C2 and eight bounded logical queue positions;
        // it does not relabel logical depth as physical C4/C8 execution.
        auto physical_c4=opt;physical_c4.max_concurrency=4;
        bool physical_c4_refused=false;
        try {ninfer::exl3::Exl3EngineCore unsupported(physical_c4);}
        catch(const std::invalid_argument& error) {
            physical_c4_refused=std::string_view(error.what()).find("C1 or explicit C2")!=
                std::string_view::npos;
        }
        need(physical_c4_refused,"mixed-arrival source treated logical depth as physical C4");
        opt.max_concurrency=2;opt.max_pending_requests=8;opt.context_cache.enabled=true;
        ninfer::exl3::Exl3EngineCore owner(opt);
        need(owner.request_queue_capacity_for_test()==10,
            "mixed-arrival queue did not reserve physical C2 plus logical C8");
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=16;
        request.execution.allow_prefix_reuse=false;
        request.stop=owner.frontend().default_stop_policy();
        std::array<std::vector<TokenId>,2> inputs{
            std::vector<TokenId>(72,198),std::vector<TokenId>(72,198)};
        inputs[1].front()=12050;
        std::mutex roots_mutex;
        std::vector<Root> roots;
        owner.observe_terminal_roots_for_test([&](auto root) {
            std::lock_guard lock(roots_mutex);roots.push_back(std::move(root));
        });
        const auto submit=[&](std::size_t family,OutputConsumerMode consumer) {
            auto prepared=owner.frontend().prepare_tokens(inputs[family]);
            const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,consumer,
                Clock::now()+std::chrono::minutes(5));
        };
        std::array<GenerationResult,2> authority;
        std::array<Root,2> authority_roots;
        for(std::size_t family=0;family<authority.size();++family) {
            auto baseline=submit(family,OutputConsumerMode::Aggregate);
            authority[family]=baseline.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            std::lock_guard lock(roots_mutex);
            need(roots.size()==family+1,"mixed-arrival baseline root missing");
            authority_roots[family]=roots.back();
            need(!authority[family].generated_token_ids.empty() && authority_roots[family],
                "mixed-arrival baseline did not establish output/state authority");
        }
        const auto baseline_root_count=roots.size();
        owner.enable_ready_decision_trace_for_test(true);

        struct StartGate {
            std::mutex mutex;std::condition_variable changed;
            std::size_t entered=0;bool release=false;
        } gate;
        struct ReleaseGate {
            StartGate& gate;
            ~ReleaseGate(){std::lock_guard lock(gate.mutex);gate.release=true;gate.changed.notify_all();}
        } release_gate{gate};
        owner.observe_request_start_for_test([&] {
            std::unique_lock lock(gate.mutex);
            ++gate.entered;gate.changed.notify_all();
            if(gate.entered<=2)need(gate.changed.wait_for(lock,std::chrono::seconds(30),
                [&]{return gate.release;}),"mixed-arrival held worker release timeout");
        });

        constexpr std::size_t request_count=10,cancel_index=5;
        std::array<std::size_t,request_count> families{};
        std::array<bool,request_count> streaming{};
        std::array<ninfer::exl3::Exl3ReadyWorkObservation,request_count> submitted_work{};
        std::vector<std::optional<ninfer::exl3::Exl3EngineCore::Submission>> handles;
        handles.reserve(request_count);
        for(std::size_t index=0;index<request_count;++index) {
            families[index]=index%2;streaming[index]=(index%3)==0;
            // Keep the deterministic queued cancellation aggregate-only so it
            // cannot acquire a sink lease before the held workers are released.
            if(index==cancel_index)streaming[index]=false;
            handles.emplace_back(submit(families[index],streaming[index]?
                OutputConsumerMode::Streaming:OutputConsumerMode::Aggregate));
            submitted_work[index]=handles.back()->ready_work_for_test();
            if(index<2) {
                std::unique_lock lock(gate.mutex);
                need(gate.changed.wait_for(lock,std::chrono::seconds(30),
                    [&]{return gate.entered>=index+1;}),
                    "mixed-arrival worker did not reach deterministic hold");
            }
        }
        const auto queued_at=Clock::now();
        const auto saturated=owner.runtime_stats();
        need(saturated.running_requests==2 && saturated.waiting_requests==8,
            "mixed-arrival source did not establish physical C2/logical C8 pressure");
        auto cancel_queued=handles[cancel_index]->cancellation_callback_for_test();
        cancel_queued();
        need(owner.runtime_stats().running_requests==2 &&
            owner.runtime_stats().waiting_requests==7,
            "mixed-arrival queued cancellation did not release one logical slot");
        const auto released_at=Clock::now();
        {std::lock_guard lock(gate.mutex);gate.release=true;gate.changed.notify_all();}

        struct CaptureSink final:OutputSink {
            std::string reasoning,content;
            std::uint64_t calls=0,bytes=0;
            void publish(OutputDelta delta) override {
                ++calls;bytes+=delta.text.size();
                (delta.channel==OutputChannel::Reasoning?reasoning:content)+=delta.text;
            }
        };
        std::array<CaptureSink,request_count> sinks;
        std::array<std::optional<GenerationResult>,request_count> results;
        std::uint64_t streaming_tokens=0,aggregate_tokens=0;
        std::uint64_t streaming_requests=0,aggregate_requests=0;
        for(std::size_t index=0;index<request_count;++index) {
            results[index].emplace(handles[index]->wait(
                streaming[index]?static_cast<OutputSink*>(&sinks[index]):nullptr,{}));
            owner.set_host_kv_routes_for_test(false,false);
            if(index==cancel_index) {
                need(results[index]->finish_reason==FinishReason::Cancelled &&
                    results[index]->generated_token_ids.empty() && !sinks[index].calls,
                    "mixed-arrival queued cancellation exposed output");
                continue;
            }
            const auto& expected=authority[families[index]];
            need(results[index]->generated_token_ids==expected.generated_token_ids &&
                results[index]->finish_reason==expected.finish_reason &&
                results[index]->reasoning==expected.reasoning &&
                results[index]->content==expected.content,
                "mixed-arrival survivor changed complete aggregate authority");
            if(streaming[index]) {
                need(sinks[index].reasoning==expected.reasoning &&
                    sinks[index].content==expected.content && sinks[index].calls,
                    "mixed-arrival streaming delivery changed complete output");
                ++streaming_requests;streaming_tokens+=results[index]->generated_token_ids.size();
            } else {
                need(!sinks[index].calls,"aggregate mixed-arrival request published to a sink");
                ++aggregate_requests;aggregate_tokens+=results[index]->generated_token_ids.size();
            }
        }
        const auto completed_at=Clock::now();
        owner.observe_request_start_for_test({});
        owner.observe_terminal_roots_for_test({});
        std::vector<Root> mixed_roots;
        {
            std::lock_guard lock(roots_mutex);
            need(roots.size()==baseline_root_count+request_count-1,
                "mixed-arrival cancellation lost or fabricated terminal roots");
            mixed_roots.assign(roots.begin()+static_cast<std::ptrdiff_t>(baseline_root_count),roots.end());
        }
        std::array<std::vector<TokenId>,2> full_authority=inputs;
        for(std::size_t family=0;family<full_authority.size();++family)
            full_authority[family].insert(full_authority[family].end(),
                authority[family].generated_token_ids.begin(),authority[family].generated_token_ids.end());
        for(const auto& root:mixed_roots) {
            bool matched=false;
            for(std::size_t family=0;family<authority_roots.size();++family) {
                const std::vector<std::int64_t> widened(
                    full_authority[family].begin(),full_authority[family].end());
                if(root->matches_tokens(widened)) {
                    need(root->state()->same_payload(*authority_roots[family]->state()) &&
                        root->same_projected_conditioning_for_test(*authority_roots[family]),
                        "mixed-arrival terminal root changed exact state or private conditioning");
                    matched=true;break;
                }
            }
            need(matched,"mixed-arrival terminal root has no independent input/output authority");
        }
        const auto settled=owner.runtime_stats();
        need(settled.running_requests==0 && settled.waiting_requests==0,
            "mixed-arrival completion retained logical or physical request ownership");
        const auto decisions=owner.ready_decision_trace_for_test();
        need(decisions.total==request_count-1 && decisions.count==request_count-1,
            "mixed-arrival provenance did not count each actually selected survivor once");
        std::array<bool,request_count> decision_seen{};
        for(std::size_t record_index=0;record_index<decisions.count;++record_index) {
            const auto& record=decisions.records[record_index];
            need(record.authority_revision && record.generation && record.input_fingerprint &&
                record.lane<2 && record.count_conserved(),
                "mixed-arrival provenance lost snapshot identity or count conservation");
            bool matched=false;
            for(std::size_t request_index=0;request_index<request_count;++request_index)
                if(record.generation==submitted_work[request_index].generation &&
                   record.input_fingerprint==submitted_work[request_index].input_fingerprint) {
                    need(request_index!=cancel_index && !decision_seen[request_index],
                        "mixed-arrival provenance selected a cancelled or duplicate request identity");
                    decision_seen[request_index]=true;matched=true;break;
                }
            need(matched,"mixed-arrival provenance does not name a submitted descriptor snapshot");
        }
        for(std::size_t request_index=0;request_index<request_count;++request_index)
            need(decision_seen[request_index]==(request_index!=cancel_index),
                "mixed-arrival provenance omitted a survivor or included queued cancellation");
        const auto release_gap_us=std::chrono::duration_cast<std::chrono::microseconds>(
            released_at-queued_at).count();
        const auto elapsed_us=std::max<std::int64_t>(1,
            std::chrono::duration_cast<std::chrono::microseconds>(completed_at-released_at).count());
        std::cout<<"ENGINE_MIXED_ARRIVALS_COMPLETE physical_c1_supported=1 physical_c=2 physical_c4_supported=0"
            <<" logical_queue=8 submitted="<<request_count<<" cancelled=1"
            <<" streaming_requests="<<streaming_requests<<" aggregate_requests="<<aggregate_requests
            <<" streaming_tokens="<<streaming_tokens<<" aggregate_tokens="<<aggregate_tokens
            <<" streaming_tokens_per_second="<<(streaming_tokens*1000000ULL/static_cast<std::uint64_t>(elapsed_us))
            <<" aggregate_tokens_per_second="<<(aggregate_tokens*1000000ULL/static_cast<std::uint64_t>(elapsed_us))
            <<" release_gap_us="<<release_gap_us<<" elapsed_us="<<elapsed_us<<'\n';
        owner.enable_ready_decision_trace_for_test(false);
        need(owner.close().reusable(),"mixed-arrival Engine retirement");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_STATE_WORKSPACE_COMPOSITION");mode) {
        const std::string_view selector(mode);
        need(selector=="1" || selector=="2","state/workspace composition selector must be 1 or 2");
        const bool c2=selector=="2";
        const auto* pinned=std::getenv("NINFER_EXL3_PINNED_RECURRENT_EXPORT");
        need(pinned && std::string_view(pinned)=="1",
            "state/workspace composition requires recurrent slab ownership");
        const auto* shared_pages_option=std::getenv("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES");
        const bool shared_pages=shared_pages_option && std::string_view(shared_pages_option)=="1";
        if(shared_pages)need(c2,"shared page composition requires physical C2");
        opt.max_concurrency=c2?2:1;opt.context_cache.enabled=true;

        using Root=std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>;
        using State=std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>;
        using Ring=std::shared_ptr<const ninfer::exl3::Exl3DraftHostRing>;
        const auto quarantine_snapshot=[] {
            const auto pins=ninfer::exl3::Exl3RecurrentPinBudget::snapshot();
            return std::array<std::uint64_t,7>{
                ninfer::exl3::Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire),
                pins[0],pins[2],
                ninfer::exl3::Exl3DevicePageStorage::quarantined_bytes(),
                ninfer::exl3::Exl3DevicePageFill::quarantined_records(),
                ninfer::exl3::Exl3EngineCore::execution_stream_quarantined_handles_for_test(),
                ninfer::exl3::Exl3LayerBufferRetirement::quarantined()};
        };
        const auto initial_ownership=quarantine_snapshot();
        std::vector<TokenId> reload_tokens;
        State reload_state;Ring reload_ring;
        std::string input="Preserve this bridge inspection record exactly, then give a concise maintenance summary: ";
        for(unsigned i=0;i<10;++i)
            input+="pier stable; drainage clear; reinforcement recorded; settlement unchanged. ";
        const auto primary_prompt=prompt(input);
        const auto peer_prompt=prompt("Explain in one sentence why a bridge expansion joint needs inspection.");

        struct RetainingSink final:OutputSink {
            std::vector<OutputDelta> held;
            std::string content,reasoning;
            void publish(OutputDelta delta) override {held.push_back(std::move(delta));}
            void release_all() {
                while(!held.empty()) {
                    auto delta=std::move(held.front());held.erase(held.begin());
                    (delta.channel==OutputChannel::Reasoning?reasoning:content)+=delta.text;
                }
            }
        };

        for(unsigned reload=0;reload<2;++reload) {
            need(quarantine_snapshot()==initial_ownership,
                "prior composition Engine lifetime retained ownership after destruction");
            std::mutex roots_mutex;
            std::vector<Root> roots;
            ninfer::exl3::Exl3EngineCore owner(opt);
            const auto startup_inventory=owner.resource_attribution_for_test();
            need(!startup_inventory.empty(),
                "state/workspace composition startup inventory is empty");
            owner.observe_terminal_roots_for_test([&](auto root) {
                std::lock_guard lock(roots_mutex);roots.push_back(std::move(root));
            });
            runtime::ResolvedRequestOptions request;
            request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=48;
            request.execution.allow_prefix_reuse=true;
            request.stop=owner.frontend().default_stop_policy();
            const auto submit=[&](const PromptInput& value,OutputConsumerMode consumer) {
                auto prepared=owner.frontend().prepare(value);const auto summary=prepared.summary();
                return owner.submit(std::move(prepared),summary,0,request,consumer,
                    std::chrono::steady_clock::now()+std::chrono::minutes(5));
            };
            const auto wait_aggregate=[&](const PromptInput& value) {
                auto handle=submit(value,OutputConsumerMode::Aggregate);
                auto result=handle.wait(nullptr,{});owner.set_host_kv_routes_for_test(false,shared_pages);
                return result;
            };

            const auto admissions_before_baseline=
                owner.runtime_stats().completed_prefix_admissions;
            const auto baseline=wait_aggregate(primary_prompt);
            const auto admissions_after_baseline=
                owner.runtime_stats().completed_prefix_admissions;
            need(!baseline.generated_token_ids.empty() &&
                baseline.content.size()+baseline.reasoning.size()>8,
                "state/workspace composition baseline lacks bounded output");
            need(baseline.finish_reason!=FinishReason::OutputLimit ||
                    admissions_after_baseline==admissions_before_baseline,
                "output-limited composition baseline was admitted as a completed chat turn");
            const auto cached=wait_aggregate(primary_prompt);
            need(cached.generated_token_ids==baseline.generated_token_ids &&
                cached.finish_reason==baseline.finish_reason &&
                cached.reused_prompt_tokens==baseline.prompt.prompt_tokens,
                "composition prefix root changed exact output");
            const auto live_pins=ninfer::exl3::Exl3RecurrentPinBudget::snapshot();
            need(live_pins[0]>initial_ownership[1] && live_pins[2]==initial_ownership[2],
                "actual Engine request did not retain a live nonquarantined recurrent slab");
            const auto grown_inventory=owner.resource_attribution_for_test();
            const auto same_owner=[](const std::weak_ptr<const void>& left,
                const std::weak_ptr<const void>& right) {
                const auto a=left.lock(),b=right.lock();
                return a && b && !a.owner_before(b) && !b.owner_before(a);
            };
            for(const auto& startup:startup_inventory) {
                const auto found=std::find_if(grown_inventory.begin(),grown_inventory.end(),
                    [&](const auto& current) {
                        return current.slot==startup.slot && current.domain==startup.domain &&
                            current.units==startup.units && same_owner(current.owner,startup.owner);
                    });
                need(found!=grown_inventory.end(),
                    "runtime growth replaced or resized a fixed startup inventory owner");
            }
            need(grown_inventory.size()>startup_inventory.size() &&
                std::any_of(grown_inventory.begin(),grown_inventory.end(),[&](const auto& current) {
                    if(current.domain!=ninfer::exl3::Exl3ResourceInventory::Domain::cuda_registered_host)
                        return false;
                    return std::none_of(startup_inventory.begin(),startup_inventory.end(),
                        [&](const auto& startup) {
                            return startup.slot==current.slot && startup.domain==current.domain &&
                                startup.units==current.units && same_owner(current.owner,startup.owner);
                        });
                }),
                "runtime recurrent growth did not join the retained startup inventory");

            RetainingSink sink;
            std::size_t roots_before_backpressure=0;
            {
                std::lock_guard lock(roots_mutex);roots_before_backpressure=roots.size();
            }
            auto backpressured=submit(primary_prompt,OutputConsumerMode::Streaming);
            backpressured.set_delivery_byte_limit_for_test(4);
            GenerationPollResult delivery;
            const auto delivery_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(sink.held.size()<2 && std::chrono::steady_clock::now()<delivery_deadline) {
                delivery=backpressured.poll(&sink,{});
                need(delivery.state==GenerationPollState::Pending,
                    "composition backpressure completed before retaining two owners");
                if(sink.held.size()<2)std::this_thread::yield();
            }
            need(sink.held.size()==2 && backpressured.outstanding_delivery_batches_for_test()==2,
                "composition did not fill both retained delivery slots");
            const auto terminal_publication_deadline=
                std::chrono::steady_clock::now()+std::chrono::seconds(30);
            bool terminal_published=false;
            while(std::chrono::steady_clock::now()<terminal_publication_deadline) {
                {
                    std::lock_guard lock(roots_mutex);
                    terminal_published=roots.size()>roots_before_backpressure;
                }
                if(terminal_published)break;
                std::this_thread::yield();
            }
            need(terminal_published,
                "composition did not publish a terminal root under retained delivery backpressure");
            const auto blocked=backpressured.poll(&sink,CancellationView([]{return true;}));
            need(blocked.state==GenerationPollState::Pending && sink.held.size()==2 &&
                backpressured.outstanding_delivery_batches_for_test()==2,
                "cancelled waiter consumed or released a retained delivery owner");
            GenerationPollResult terminal;
            const auto drain_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            while(std::chrono::steady_clock::now()<drain_deadline) {
                sink.release_all();terminal=backpressured.poll(&sink,{});
                if(terminal.state!=GenerationPollState::Pending)break;
                std::this_thread::yield();
            }
            sink.release_all();
            need(terminal.state==GenerationPollState::Completed && terminal.result,
                "composition backpressure did not reach terminal result");
            need(terminal.result->generated_token_ids==baseline.generated_token_ids,
                "composition backpressure changed generated tokens");
            need(terminal.result->finish_reason==baseline.finish_reason,
                "composition backpressure changed finish reason");
            if(sink.content!=baseline.content || sink.reasoning!=baseline.reasoning)
                throw std::runtime_error("composition backpressure changed channel content baseline_content="+
                    std::to_string(baseline.content.size())+" streamed_content="+
                    std::to_string(sink.content.size())+" baseline_reasoning="+
                    std::to_string(baseline.reasoning.size())+" streamed_reasoning="+
                    std::to_string(sink.reasoning.size()));
            owner.set_host_kv_routes_for_test(false,shared_pages);

            std::atomic<bool> cancel=false;CancellingSink cancelling(cancel);
            auto cancelled=submit(primary_prompt,OutputConsumerMode::Streaming);
            const auto partial=cancelled.wait(&cancelling,CancellationView([&]{return cancel.load();}));
            owner.set_host_kv_routes_for_test(false,shared_pages);
            need(partial.finish_reason==FinishReason::Cancelled && !cancelling.text.empty(),
                "composition active cancellation was not observed after publication");
            const auto restored=wait_aggregate(primary_prompt);
            need(restored.generated_token_ids==baseline.generated_token_ids &&
                restored.finish_reason==baseline.finish_reason,
                "composition cancellation contaminated the next request");
            Root retained;
            {
                std::lock_guard lock(roots_mutex);
                need(!roots.empty(),"composition restored request published no terminal root");
                retained=roots.back();
            }

            if(c2) {
                const auto peer_expected=wait_aggregate(peer_prompt);
                auto first=submit(primary_prompt,OutputConsumerMode::Aggregate);
                auto second=submit(peer_prompt,OutputConsumerMode::Aggregate);
                const auto first_result=first.wait(nullptr,{});
                const auto second_result=second.wait(nullptr,{});
                owner.set_host_kv_routes_for_test(false,shared_pages);
                need(first_result.generated_token_ids==baseline.generated_token_ids &&
                    second_result.generated_token_ids==peer_expected.generated_token_ids,
                    "C2 composition changed request-local exact output");
            }

            const auto stats=owner.runtime_stats();
            const bool dependency_graph_conserved=
                stats.execution_dependency_graph_begins==
                    stats.execution_dependency_graph_completions+
                    stats.execution_dependency_graph_cancellations+
                    stats.execution_dependency_graph_failures &&
                stats.execution_dependency_graph_failures==0;
            // C1 intentionally owns the CUDA default stream. The selected
            // transfer/compute graph exists only for retained nondefault
            // streams, which physical C2 constructs independently per lane.
            if(!(dependency_graph_conserved &&
                (c2?stats.execution_dependency_graph_begins>0:
                    stats.execution_dependency_graph_begins==0)))
                throw std::runtime_error(
                    "composition dependency graph lost terminal conservation begins="+
                    std::to_string(stats.execution_dependency_graph_begins)+
                    " submissions="+
                    std::to_string(stats.execution_dependency_graph_submissions)+
                    " completions="+std::to_string(stats.execution_dependency_graph_completions)+
                    " cancellations="+std::to_string(stats.execution_dependency_graph_cancellations)+
                    " failures="+std::to_string(stats.execution_dependency_graph_failures));
            const auto composed_retention=owner.prefix_retention_metadata();
            if(composed_retention.empty())
                throw std::runtime_error(
                    "composition prefix retention/admission mismatch admissions="+
                    std::to_string(stats.completed_prefix_admissions)+
                    " resident="+std::to_string(composed_retention.size())+
                    " baseline_finish="+std::to_string(static_cast<unsigned>(baseline.finish_reason))+
                    " baseline_tokens="+std::to_string(baseline.generated_token_ids.size()));
            if(shared_pages) {
                need(stats.shared_device_page_allocations>0 &&
                    stats.shared_device_page_source_pages>0 &&
                    stats.shared_device_page_fill_bytes>0 &&
                    stats.shared_device_page_retained_readers==0 &&
                    stats.shared_device_page_pending_fill_bytes==0 &&
                    stats.shared_device_page_in_flight_fill_bytes==0 &&
                    stats.shared_device_page_failed_allocations==0 &&
                    stats.shared_device_page_failures==0,
                    "C2 composition shared prefix pages were absent or not finally owned");
            } else need(stats.shared_device_page_fill_bytes==0 &&
                stats.shared_device_page_copy_bytes==0 &&
                stats.shared_device_page_attention_bytes==0 &&
                stats.shared_device_page_failures==0,
                "default composition unexpectedly entered the shared-page route");
            const auto* shared_projection=std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_Q");
            if(!shared_projection || std::string_view(shared_projection)!="1")
                need(stats.shared_target_projection_batches==0,
                    "default composition unexpectedly entered shared numerical projection");

            {
                std::lock_guard lock(roots_mutex);
                need(roots.size()>=(c2?8U:5U),"composition lost terminal root publication");
            }
            auto final_state=retained->state()->detached_payload_for_test();
            auto final_ring=retained->detached_projected_conditioning_for_test();
            need(final_state && final_ring,"composition terminal root omitted exact state or conditioning");
            if(reload)need(restored.generated_token_ids==reload_tokens && reload_state && reload_ring &&
                final_state->same_represented_payload_for_test(*reload_state) &&
                final_ring->same_represented_payload_for_test(*reload_ring),
                "replacement Engine changed tokens, exact state or draft conditioning");
            else {reload_tokens=restored.generated_token_ids;reload_state=final_state;reload_ring=final_ring;}
            const auto resident=owner.prefix_retention_metadata();
            std::vector<ninfer::exl3::Exl3VeriCachePrefixIndex::RetentionDecision> decisions;
            decisions.reserve(resident.size());
            for(const auto& entry:resident)
                decisions.push_back({entry.root,entry.generation,false,0});
            const auto evicted=owner.trim_prefix_retention(0,decisions);
            need(evicted.inputs_current && evicted.budget_met && evicted.evicted==resident.size() &&
                owner.prefix_retention_metadata().empty() &&
                retained->state()->same_represented_payload_for_test(*final_state) &&
                retained->matches_detached_conditioning_for_test(*final_ring),
                "cache eviction changed externally retained terminal authority");
            owner.observe_terminal_roots_for_test({});
            {
                std::lock_guard lock(roots_mutex);roots.clear();
            }
            need(owner.close().reusable(),"composition Engine final retirement was not reusable");
            need(retained->state()->same_represented_payload_for_test(*final_state) &&
                retained->matches_detached_conditioning_for_test(*final_ring),
                "Engine close invalidated externally retained terminal authority");
            retained.reset();final_state.reset();final_ring.reset();
            const auto final_ownership=quarantine_snapshot();
            const bool quarantine_clean=final_ownership[0]==initial_ownership[0] &&
                final_ownership[2]==initial_ownership[2] &&
                final_ownership[3]==initial_ownership[3] &&
                final_ownership[4]==initial_ownership[4] &&
                final_ownership[5]==initial_ownership[5] &&
                final_ownership[6]==initial_ownership[6];
            // close() proves reusable retirement but the closed owner remains
            // alive until this iteration ends, so its ordinary live recurrent
            // registration may remain charged here. Full equality is checked
            // at the next post-destruction boundary and after the loop.
            if(!(quarantine_clean && final_ownership[1]>=initial_ownership[1])) {
                std::string detail;
                for(std::size_t index=0;index<final_ownership.size();++index)
                    detail+=" ["+std::to_string(index)+":"+
                        std::to_string(initial_ownership[index])+"->"+
                        std::to_string(final_ownership[index])+"]";
                throw std::runtime_error(
                    "composition close quarantined slab/page/stream ownership"+detail);
            }
        }
        reload_state.reset();reload_ring.reset();
        need(quarantine_snapshot()==initial_ownership,
            "composition reload oracle retained final or quarantined ownership");
        std::cout<<"ENGINE_STATE_WORKSPACE_COMPOSITION C"<<(c2?2:1)<<" COMPLETE\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_HOST_PREPARATION_OVERLAP");
       mode && std::string_view(mode)=="1") {
        const auto* shared=std::getenv("NINFER_EXL3_ENGINE_SHARED_PREPARATION");
        need(shared && std::string_view(shared)=="1",
            "host preparation overlap requires concurrent prefix preparation");
        opt.max_concurrency=2;opt.context_cache.enabled=true;
        ninfer::exl3::Exl3EngineCore owner(opt);
        std::mutex gate_mutex;std::condition_variable gate;
        bool numerical_pending=false,host_completed=false;
        bool release_numerical=false,release_host=false;
        std::size_t numerical_lane=2,host_lane=2;
        owner.observe_prefix_preparation_for_test([&](std::size_t lane,int) {
            std::unique_lock lock(gate_mutex);
            if(numerical_pending)return;
            numerical_pending=true;numerical_lane=lane;gate.notify_all();
            if(!gate.wait_for(lock,std::chrono::seconds(30),[&]{return release_numerical;}))
                throw std::runtime_error("host overlap numerical gate timeout");
        });
        owner.observe_host_preparation_for_test(
            [&](std::size_t lane,std::uint64_t generation,bool peer_numerical) {
                if(!generation || !peer_numerical)return;
                std::unique_lock lock(gate_mutex);
                host_completed=true;host_lane=lane;gate.notify_all();
                if(!gate.wait_for(lock,std::chrono::seconds(30),[&]{return release_host;}))
                    throw std::runtime_error("host overlap completion gate timeout");
            });
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=8;
        request.execution.allow_prefix_reuse=true;
        request.stop=owner.frontend().default_stop_policy();
        std::string text="Order these bridge inspection notes without changing their sequence: ";
        for(unsigned i=0;i<20;++i)
            text+="check soil, inspect drainage, verify reinforcement, record settlement; ";
        const auto submit=[&] {
            auto prepared=owner.frontend().prepare(prompt(text));
            const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,
                OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        auto first=submit();
        {
            std::unique_lock lock(gate_mutex);
            need(gate.wait_for(lock,std::chrono::seconds(30),[&]{return numerical_pending;}),
                "first request did not retain pending numerical work");
        }
        need(first.poll(nullptr,{}).state==GenerationPollState::Pending,
            "pending numerical request reported terminal before host overlap");
        auto second=submit();
        {
            std::unique_lock lock(gate_mutex);
            need(gate.wait_for(lock,std::chrono::seconds(30),[&]{return host_completed;}),
                "peer host plan did not complete during numerical work");
        }
        need(numerical_lane<2 && host_lane<2 && numerical_lane!=host_lane &&
            first.poll(nullptr,{}).state==GenerationPollState::Pending,
            "host completion consumed or replaced pending peer authority");
        const auto* cancel_option=std::getenv(
            "NINFER_TEST_ENGINE_HOST_PREPARATION_CANCEL");
        const bool cancel=cancel_option && std::string_view(cancel_option)=="1";
        if(cancel)need(second.poll(nullptr,CancellationView([]{return true;})).state==
                GenerationPollState::Pending,
            "blocked host preparation did not expose cancellable pending state");
        {
            std::lock_guard lock(gate_mutex);
            release_host=true;release_numerical=true;
        }
        gate.notify_all();
        const auto peer=second.wait(nullptr,{});
        const auto primary=first.wait(nullptr,{});
        owner.observe_prefix_preparation_for_test({});
        owner.observe_host_preparation_for_test({});
        const auto stats=owner.runtime_stats();
        need(stats.execution_dependency_graph_submissions==
                stats.execution_dependency_graph_completions &&
            stats.execution_dependency_graph_completions>=(cancel?1U:2U) &&
            stats.execution_dependency_graph_failures==0,
            "real overlap requests did not complete their event dependency chains");
        if(cancel) {
            need(peer.finish_reason==FinishReason::Cancelled &&
                peer.generated_token_ids.empty() &&
                stats.host_preparation_returns==1 &&
                stats.host_preparation_peer_numeric_overlaps==0 &&
                stats.host_preparation_cancelled_before_accept==1,
                "cancelled host plan advanced numerical or output state");
        } else {
            need(!primary.generated_token_ids.empty() &&
                primary.generated_token_ids==peer.generated_token_ids &&
                primary.finish_reason==peer.finish_reason &&
                stats.host_preparation_returns==2 &&
                stats.host_preparation_peer_numeric_overlaps==1 &&
                stats.host_preparation_cancelled_before_accept==0,
                "accepted host overlap changed request-local output order");
        }
        need(owner.close().reusable(),"host preparation overlap retirement");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SELECTION_AUTHORITY");
       mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3EngineCore owner(opt);
        runtime::ResolvedRequestOptions sampled;
        sampled.execution.sampling.temperature=0.7F;
        sampled.execution.requested_output_tokens=8;
        sampled.execution.allow_prefix_reuse=false;
        sampled.stop.include_model_defaults=false;
        auto sampled_prompt=owner.frontend().prepare(prompt("Explain bridge foundations."));
        const auto sampled_summary=sampled_prompt.summary();
        const auto before=owner.runtime_stats();bool refused=false;
        try {
            (void)owner.submit(std::move(sampled_prompt),sampled_summary,0,sampled,
                OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        } catch(const RequestError& error) {
            refused=error.kind()==RequestErrorKind::UnsupportedSampling;
        }
        const auto rejected=owner.runtime_stats();
        need(refused && rejected.running_requests==before.running_requests &&
            rejected.waiting_requests==before.waiting_requests &&
            rejected.greedy_packet_transport_batches==before.greedy_packet_transport_batches &&
            rejected.greedy_packet_transport_singles==before.greedy_packet_transport_singles,
            "sampled request reached compact greedy authority or Engine ownership");
        auto greedy=sampled;greedy.execution.sampling.temperature=0;
        auto greedy_prompt=owner.frontend().prepare(prompt("Explain bridge foundations."));
        const auto greedy_summary=greedy_prompt.summary();
        const auto result=owner.submit(std::move(greedy_prompt),greedy_summary,0,greedy,
            OutputConsumerMode::Aggregate,
            std::chrono::steady_clock::now()+std::chrono::minutes(5)).wait(nullptr,{});
        need(!result.generated_token_ids.empty(),"greedy authority replay produced no result");
        need(owner.close().reusable(),"selection authority fixture retirement");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_AVAILABILITY_PROVIDER");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string_view selector(mode);
        need(selector=="failed" || selector=="stale" || selector=="contradictory" ||
            selector=="final_failed" || selector=="attribution",
            "Engine availability provider fixture selector");
        std::atomic<unsigned> calls=0;
        const auto provider=[&] {
            const auto call=++calls;
            if(selector=="failed" || (selector=="final_failed" && call==2))
                return Exl3DeviceAvailability{1,1,Exl3DeviceAvailability::Clock::now(),37};
            if(selector=="stale")
                return Exl3DeviceAvailability{std::numeric_limits<std::uint64_t>::max(),
                    std::numeric_limits<std::uint64_t>::max()/2,
                    Exl3DeviceAvailability::Clock::now()-std::chrono::seconds(2),0};
            if(selector=="contradictory")
                return Exl3DeviceAvailability{64,65,Exl3DeviceAvailability::Clock::now(),0};
            return Exl3DeviceAvailability{std::numeric_limits<std::uint64_t>::max(),
                std::numeric_limits<std::uint64_t>::max()/2,Exl3DeviceAvailability::Clock::now(),0};
        };
        bool refused=false;
        try{
            Exl3EngineCore owner(opt,0,0,0,0,provider);
            if(selector=="attribution") {
                const auto stats=owner.runtime_stats();
                const auto attributed=owner.resource_attribution_for_test();
                std::uint64_t device=0;
                for(std::size_t i=0;i<attributed.size();++i) {
                    const auto& entry=attributed[i];
                    need(!entry.owner.expired() && entry.owner_address && entry.units &&
                        entry.reason==Exl3ResourceInventory::AttributionReason::live_owner,
                        "Engine resource attribution emitted an ownerless or unexplained record");
                    if(entry.domain==Exl3ResourceInventory::Domain::device) {
                        need(entry.units<=UINT64_MAX-device,"Engine device attribution overflow");
                        device+=entry.units;
                    }
                    for(std::size_t j=0;j<i;++j) {
                        const auto left=entry.owner.lock(),right=attributed[j].owner.lock();
                        const bool same=left && right && !left.owner_before(right) && !right.owner_before(left);
                        need(!same || entry.slot!=attributed[j].slot,
                            "Engine resource attribution duplicated an owner slot alias");
                    }
                }
                need(!attributed.empty() && stats.driver_unknown_device_bytes_available &&
                    stats.inventoried_device_bytes==device &&
                    stats.observed_device_used_bytes==stats.inventoried_device_bytes+
                        stats.driver_unknown_device_bytes,
                    "Engine device attribution lost exact inventory or driver-unknown remainder");
                need(owner.close().reusable(),"Engine attribution fixture retirement");
                refused=true;
            }
        }
        catch(const std::runtime_error& error) {
            const std::string_view message(error.what());
            refused=selector=="failed" || selector=="final_failed"?
                message=="device availability provider failed":
                message=="device availability missing/stale observation" ||
                    message=="device availability/reserve contradiction";
        }
        need(refused,"Engine availability provider failure was ignored or replaced");
        need(calls==(selector=="failed" || selector=="stale" ||
            selector=="contradictory"?1U:2U),
            "Engine availability provider was skipped or unexpectedly reread");
        if(selector=="final_failed") {
            // The second observation occurs after all retained startup owners
            // are prepared but before workers become externally reachable.
            // A fresh CUDA-backed construction must remain possible in a
            // separate invocation of this fixture.
            need(Exl3EngineCore::execution_stream_retirement_slots_for_test()==0,
                "failed final availability observation retained stream ownership");
        }
        std::cout<<"ENGINE_AVAILABILITY_PROVIDER_COMPLETE selector="<<selector<<'\n';return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_METADATA_FLOOR");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        struct SavedEnvironment {
            const char* name;std::string value;
            explicit SavedEnvironment(const char* key):name(key),value(std::getenv(key)?std::getenv(key):""){}
            ~SavedEnvironment(){_putenv_s(name,value.c_str());}
        } saved_cap("NINFER_EXL3_ENGINE_HOST_METADATA_LIMIT_BYTES"),
          saved_staging("NINFER_EXL3_ENGINE_ATTENTION_STAGING");
        const auto accepted_floor=[&]() -> std::uint64_t {
            const auto cancellations=Exl3EngineCore::cancellation_owner_blocks_for_test();
            const auto streams=Exl3EngineCore::execution_stream_owner_blocks_for_test();
            try{Exl3EngineCore owner(opt,0,0,0,128);}
            catch(const std::runtime_error& error) {
                need(Exl3EngineCore::cancellation_owner_blocks_for_test()==cancellations &&
                    Exl3EngineCore::execution_stream_owner_blocks_for_test()==streams,
                    "Engine early metadata gate retained cancellation or stream owners");
                const std::string_view message(error.what());
                constexpr std::string_view prefix="Engine mandatory metadata floor accepted:";
                if(message.substr(0,prefix.size())==prefix)
                    return std::stoull(std::string(message.substr(prefix.size())));
                throw;
            }
            throw std::runtime_error("Engine floor diagnostic stop NOT_EXERCISED");
        };
        for(unsigned concurrency:{1u,2u}) {
            opt.max_concurrency=concurrency;
            _putenv_s(saved_cap.name,"");_putenv_s(saved_staging.name,"0");
            const auto ordinary=accepted_floor();
            const auto lane_controls=Exl3EngineCore::execution_stream_owner_bytes_for_test()+
                Exl3TextContext::fixed_owner_metadata_bytes()+Exl3TextContext::continuation_owner_metadata_bytes();
            need(ordinary>=concurrency*lane_controls,
                "Engine mandatory floor smaller than required stream/context/continuation controls");
            _putenv_s(saved_staging.name,"1");
            const auto staged=accepted_floor();
            need(staged==ordinary+concurrency*Exl3AttentionStageResources::metadata_bytes(),
                "Engine floor omitted or duplicated lane attention metadata");
            for(unsigned staging:{0u,1u}) {
                _putenv_s(saved_staging.name,staging?"1":"0");
                const auto floor=staging?staged:ordinary;
                need(floor>0,"Engine mandatory floor unexpectedly empty");
                _putenv_s(saved_cap.name,std::to_string(floor-1).c_str());
                bool refused=false;
                try{(void)accepted_floor();}
                catch(const Exl3ResourceReservationExhausted&){refused=true;}
                need(refused,"Engine accepted below-floor host metadata ceiling");
                _putenv_s(saved_cap.name,std::to_string(floor).c_str());
                need(accepted_floor()==floor,"Engine rejected exact known floor before later admission");
            }
        }
        std::cout<<"ENGINE_METADATA_FLOOR_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_METADATA_CAP_ZERO");mode && std::string_view(mode)=="1") {
        const auto* cap=std::getenv("NINFER_EXL3_ENGINE_HOST_METADATA_LIMIT_BYTES");
        need(cap && std::string_view(cap)=="0","zero metadata cap fixture requires explicit zero-byte configuration");
        bool refused=false;
        try{ninfer::exl3::Exl3EngineCore owner(opt);}
        catch(const std::runtime_error& error) {
            const std::string_view message(error.what());
            refused=message=="resource transition budget exhausted" || message=="resource allocation reservation exhausted";
        }
        need(refused,"Engine ignored explicit zero metadata cap or failed outside budget admission");
        std::cout<<"ENGINE_METADATA_CAP_ZERO_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_UPLOAD_POOL");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string_view selector(mode);need(selector=="1" || selector=="2","Engine upload lane selector");
        opt.max_concurrency=selector=="2"?2:1;
        const auto blocks=bounded_shared_live_blocks_for_test<Exl3RegisteredKVUpload>();
        bool original=false;
        try{Exl3EngineCore owner(opt,0,0,0,127);}
        catch(const std::runtime_error& error){original=std::string_view(error.what())=="injected KV upload slot construction failure";}
        need(original,"Engine upload fixture did not reach final-slot failure");
        need(bounded_shared_live_blocks_for_test<Exl3RegisteredKVUpload>()==blocks,"Engine upload failure retained partial pool");
        {Exl3EngineCore reload(opt);need(reload.close().reusable(),"upload pool failure prevented fresh Engine reload");}
        need(bounded_shared_live_blocks_for_test<Exl3RegisteredKVUpload>()==blocks,"Engine upload shutdown retained slots");
        std::cout<<"ENGINE_UPLOAD_POOL_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_PAGE_READERS");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string_view selector(mode);need(selector=="1" || selector=="2","Engine page reader lane selector");
        opt.max_concurrency=selector=="2"?2:1;
        const auto copies=bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>();
        const auto readers=bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>();
        bool original=false;
        try{Exl3EngineCore owner(opt,0,0,0,126);}
        catch(const std::runtime_error& error){original=std::string_view(error.what())=="injected page reader construction failure";}
        need(original,"Engine page reader fixture did not reach last-reader failure");
        need(bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>()==copies &&
            bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>()==readers,
            "Engine reader startup unwind retained partial pool");
        {Exl3EngineCore reload(opt);need(reload.close().reusable(),"reader construction failure prevented Engine reload");}
        need(bounded_shared_live_blocks_for_test<Exl3DevicePageCopy>()==copies &&
            bounded_shared_live_blocks_for_test<Exl3DevicePageAttention>()==readers,
            "Engine reader pool shutdown retained control allocations");
        std::cout<<"ENGINE_PAGE_READERS_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_PAGE_CONTAINER");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;using Cache=Exl3DevicePageCache;
        const auto blocks=bounded_shared_live_blocks_for_test<Cache>();bool original=false;
        try{Exl3EngineCore owner(opt,0,0,0,125);}
        catch(const std::runtime_error& error) {
            original=std::string_view(error.what())=="injected page cache container precommit failure";
        }
        need(original,"Engine page container fixture did not reach shared-pages startup failure");
        need(bounded_shared_live_blocks_for_test<Cache>()==blocks,
            "Engine startup unwind retained empty page container allocation");
        {Exl3EngineCore reload(opt);need(reload.close().reusable(),"page container failure prevented fresh Engine startup");}
        need(bounded_shared_live_blocks_for_test<Cache>()==blocks,"Engine page container successful shutdown leaked control storage");
        std::cout<<"ENGINE_PAGE_CONTAINER_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_PREFIX_CONTROL");mode && *mode) {
        using namespace ninfer::exl3;using Cache=Exl3DevicePrefixCache;
        const std::string_view selector(mode);need(selector=="release" || selector=="retain","Engine prefix control selector");
        const bool retain=selector=="retain";
        need(device_prefix,"Engine prefix control fixture requires represented prefix enabled");
        if(const auto* lanes=std::getenv("NINFER_TEST_ENGINE_PREFIX_CONTROL_LANES")) {
            need(std::string_view(lanes)=="1" || std::string_view(lanes)=="2","Engine prefix control physical lanes1/2");
            opt.max_concurrency=std::string_view(lanes)=="2"?2:1;
        }
        const auto before=Cache::budget_snapshot();need(before[1]==0,"Engine prefix control requires fresh process");
        const auto attempts=Cache::control_fault_attempts_for_test();bool original=false;
        try{Exl3EngineCore owner(opt,0,0,0,retain?124:123);}
        catch(const std::bad_alloc&){original=true;}
        need(original && Cache::control_fault_attempts_for_test()==attempts+1,
            "Engine prefix control fixture did not reach allocator failure exactly once");
        if(retain) {
            const auto held=Cache::retirement_snapshot_for_test();
            const auto extent=Cache::allocation_bytes_required(prefix16k?16384:4096);
            need(held.pointer && held.bytes==extent && held.device_credit==extent &&
                held.metadata_credit==Cache::retirement_metadata_bytes() && held.error==cudaErrorUnknown,
                "Engine prefix control failure lost acquired storage or exact survivor credits");
            bool refused=false;try{Exl3EngineCore reload(opt);}
            catch(const std::runtime_error& error) {
                refused=std::string_view(error.what())=="EXL3 unresolved device prefix retirement; reload refused";
            }
            need(refused,"Engine reload bypassed failed prefix control retirement");
        } else {
            need(Cache::budget_snapshot()==before,"Engine prefix control clean unwind retained physical charge");
            Exl3EngineCore reload(opt);need(reload.close().reusable(),"clean prefix control failure blocked fresh Engine");
        }
        std::cout<<"ENGINE_PREFIX_CONTROL_COMPLETE mode="<<selector<<" lanes="<<opt.max_concurrency<<'\n';return 0;
    }
    // Isolated process: this mode intentionally retains uncertain transfer owners.
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_DRAFT_EXPORT_UNCERTAIN");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        need(opt.max_concurrency==1,"Engine uncertain draft export fixture requires C1");
        const auto before=Exl3DraftHostRing::uncertain_transfer_count();
        std::weak_ptr<const void> uncertain_source;
        {
            Exl3EngineCore owner(opt);owner.fail_next_draft_export_completion_for_test();
            auto prepared=owner.frontend().prepare_tokens(std::vector<TokenId>{1,2,3,4},true);
            const auto summary=prepared.summary();
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=1;request.execution.allow_prefix_reuse=true;
            request.stop.include_model_defaults=false;
            bool failed=false;
            try {
                owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(10)).wait(nullptr,{});
            } catch(const std::exception& error) {
                const auto inspect=[&](auto&& self,const std::exception& current)->void {
                    failed|=std::string_view(current.what()).find("draft host suffix complete")!=std::string_view::npos;
                    try{std::rethrow_if_nested(current);}catch(const std::exception& inner){self(self,inner);}
                };
                inspect(inspect,error);
            }
            uncertain_source=owner.uncertain_draft_source_owner_for_test();
            need(!uncertain_source.expired(),"Engine uncertain export lacks retained source allocation");
            const auto first=owner.close(),again=owner.close();
            need(failed && Exl3DraftHostRing::uncertain_transfer_count()==before+1,
                "Engine fixture missed submitted draft export completion failure");
            need(first.phase==Exl3RetirementPhase::quarantined && !again.reusable() &&
                again.first_drain_error==first.first_drain_error,
                "Engine recycled uncertain draft export after owning drain");
        }
        need(Exl3DraftHostRing::uncertain_transfer_count()==before+1,
            "Engine destruction released unresolved draft destination");
        need(!uncertain_source.expired(),"Engine destruction released unresolved draft source owner");
        bool reload_refused=false;
        try{Exl3EngineCore::require_startup_retirement_available(opt);}
        catch(const std::runtime_error&){reload_refused=true;}
        need(reload_refused,"uncertain Engine draft export allowed reload");
        std::cout<<"ENGINE_DRAFT_EXPORT_UNCERTAIN_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_METADATA_CEILING");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        need(opt.max_concurrency==1,"aggregate metadata ceiling fixture requires C1");
        const auto initial_blocks=Exl3EngineCore::result_owner_blocks_for_test();
        {
            Exl3EngineCore owner(opt);
            const auto startup=owner.retained_host_metadata_for_test();
            std::shared_ptr<const Exl3VeriCacheRequest> retained_root;
            owner.observe_terminal_roots_for_test([&](auto root){retained_root=std::move(root);});
            auto submit=[&] {
                auto prepared=owner.frontend().prepare_tokens(std::vector<TokenId>{1,2,3,4},true);
                const auto summary=prepared.summary();
                runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
                request.execution.requested_output_tokens=1;request.execution.allow_prefix_reuse=true;
                request.stop.include_model_defaults=false;
                return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(10));
            };
            {
                auto warm=submit();const auto completed=warm.wait(nullptr,{});
                // Wait for terminal ownership cleanup as well as result delivery.
                const auto retained=owner.tighten_metadata_headroom_for_test(0);
                need(retained_root && !completed.generated_token_ids.empty() && retained>startup,
                    "aggregate metadata fixture lacks completed root/result ownership");
                const auto root_identity=retained_root.get();
                const auto result_blocks=Exl3EngineCore::result_owner_blocks_for_test();
                const auto root_blocks=bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>();
                bool refused=false;
                try {auto rejected=submit();}
                catch(const RequestError& error) {
                    refused=error.kind()==RequestErrorKind::Overloaded && std::string_view(error.what())==
                        "EXL3 result reservation exceeds available host budget";
                }
                need(refused && owner.retained_host_metadata_for_test()==retained &&
                    Exl3EngineCore::result_owner_blocks_for_test()==result_blocks &&
                    bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==root_blocks &&
                    retained_root.get()==root_identity,
                    "aggregate ceiling refusal allocated storage or disturbed retained ownership");
                bool increase_refused=false;
                try {owner.tighten_metadata_headroom_for_test(1);}
                catch(const std::invalid_argument&){increase_refused=true;}
                need(increase_refused,"metadata diagnostic increased the installed ceiling");
            }
            owner.observe_terminal_roots_for_test({});retained_root.reset();
            need(owner.close().reusable(),"aggregate metadata ceiling refusal poisoned Engine retirement");
        }
        need(Exl3EngineCore::result_owner_blocks_for_test()==initial_blocks,
            "aggregate metadata fixture retained result storage after teardown");
        std::cout<<"ENGINE_METADATA_CEILING_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_REQUEST_METADATA_FAILURE");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string_view selector(mode);
        need(opt.max_concurrency==1 && (selector=="preparation" || selector=="active" || selector=="active_ceiling"),
            "request metadata failure fixture requires C1 and preparation, active or active_ceiling");
        const unsigned phase=selector=="preparation"?1:selector=="active"?2:3;
        const auto blocks=bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>();
        {
            Exl3EngineCore owner(opt);owner.fail_request_metadata_for_test(phase);
            auto prepared=owner.frontend().prepare_tokens(std::vector<TokenId>{1,2,3,4},true);
            const auto summary=prepared.summary();
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=1;request.execution.allow_prefix_reuse=true;
            request.stop.include_model_defaults=false;
            bool refused=false,completed=false;
            try {
                owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(10)).wait(nullptr,{});
                completed=true;
            } catch(const std::exception& error) {
                const auto inspect=[&](auto&& self,const std::exception& current)->void {
                    if(phase==3)refused|=dynamic_cast<const Exl3ResourceReservationExhausted*>(&current)!=nullptr;
                    else refused|=std::string_view(current.what())=="request metadata admission fixture exhausted";
                    try{std::rethrow_if_nested(current);}catch(const std::exception& inner){self(self,inner);}
                };
                inspect(inspect,error);
            }
            need(owner.request_metadata_fault_hit_for_test()==phase && refused && !completed,
                "request metadata refusal missed phase or completed request");
            (void)owner.close();
        }
        need(bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==blocks,
            "request metadata refusal retained request storage");
        Exl3EngineCore reload(opt);need(reload.close().reusable(),"request metadata refusal blocked fresh Engine");
        std::cout<<"ENGINE_REQUEST_METADATA_FAILURE_COMPLETE mode="<<selector<<'\n';return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_HOST_PAYLOAD_FAILURE");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string_view selector(mode);
        need(opt.max_concurrency==1 && (selector=="reservation" || selector=="tracking"),
            "host payload failure fixture requires C1 and reservation or tracking");
        const auto* direct=std::getenv("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING");
        need(!direct || std::string_view(direct)=="0","host payload failure fixture requires copied taps");
        const unsigned phase=selector=="reservation"?1:2;
        const auto blocks=bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>();
        const auto run=[&](Exl3EngineCore& owner) {
            auto prepared=owner.frontend().prepare_tokens(std::vector<TokenId>{1,2,3,4},true);
            const auto summary=prepared.summary();
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=8;request.execution.allow_prefix_reuse=false;
            request.stop.include_model_defaults=false;
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(10)).wait(nullptr,{});
        };
        std::vector<TokenId> expected;
        std::shared_ptr<const Exl3ExactHostState> baseline_state;
        std::shared_ptr<const Exl3DraftHostRing> baseline_ring;
        {
            Exl3EngineCore baseline(opt);
            baseline.observe_terminal_roots_for_test([&](auto root){
                baseline_state=root->state()->detached_payload_for_test();
                baseline_ring=root->detached_projected_conditioning_for_test();
            });
            expected=run(baseline).generated_token_ids;
            need(!expected.empty() && baseline_state && baseline_ring,"host payload baseline did not reach decode");
            baseline.observe_terminal_roots_for_test({});
            need(baseline.close().reusable(),"host payload baseline retirement failed");
        }
        {
            Exl3EngineCore owner(opt);owner.fail_request_host_payload_for_test(phase);
            bool refused=false,completed=false;
            try{(void)run(owner);completed=true;}catch(const std::exception& error) {
                const auto inspect=[&](auto&& self,const std::exception& current)->void {
                    refused|=std::string_view(current.what())==(phase==1?
                        "request host payload reservation fixture failure":"request host payload tracking fixture failure");
                    try{std::rethrow_if_nested(current);}catch(const std::exception& inner){self(self,inner);}
                };
                inspect(inspect,error);
            }
            need(owner.request_host_payload_fault_hit_for_test()==phase && refused && !completed,
                "host payload failure missed actual reservation/tracking or completed request");
            need(owner.close().reusable(),"host payload failure retained uncertain Engine ownership");
        }
        {
            Exl3EngineCore retry(opt);
            std::shared_ptr<const Exl3VeriCacheRequest> retry_root;
            retry.observe_terminal_roots_for_test([&](auto root){retry_root=std::move(root);});
            const auto result=run(retry);
            need(result.generated_token_ids==expected && retry_root &&
                retry_root->state()->same_represented_payload_for_test(*baseline_state) &&
                retry_root->matches_detached_conditioning_for_test(*baseline_ring),
                "host payload failure changed fresh Engine exact retry");
            retry.observe_terminal_roots_for_test({});retry_root.reset();
            need(retry.close().reusable(),"host payload retry retirement failed");
        }
        baseline_state.reset();baseline_ring.reset();
        need(bounded_shared_live_blocks_for_test<Exl3VeriCacheRequest>()==blocks,
            "host payload failure/retry retained request storage");
        std::cout<<"ENGINE_HOST_PAYLOAD_FAILURE_COMPLETE mode="<<selector<<'\n';return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SNAPSHOT_CANCEL");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        need(opt.max_concurrency==1,"snapshot cancellation fixture requires C1");
        Exl3EngineCore owner(opt);owner.cancel_next_active_snapshot_for_test();
        auto prepared=owner.frontend().prepare_tokens(std::vector<TokenId>{1,2,3,4},true);
        const auto summary=prepared.summary();
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=8;request.execution.allow_prefix_reuse=true;
        request.stop.include_model_defaults=false;
        const auto result=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
            std::chrono::steady_clock::now()+std::chrono::minutes(10)).wait(nullptr,{});
        need(result.finish_reason==FinishReason::Cancelled && result.generated_token_ids.empty(),
            "active snapshot cancellation failed or published discarded verification");
        need(owner.close().reusable(),"active snapshot cancellation poisoned Engine retirement");
        std::cout<<"ENGINE_SNAPSHOT_CANCEL_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_RECURRENT_PREPARATION_COMMIT");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string_view selector(mode);
        need(selector=="slab" || selector=="borrower" || selector=="active_slab" || selector=="active_borrower" ||
             selector=="cancel_slab" || selector=="active_cancel_slab",
            "recurrent commit selector");
        const bool cancellation=selector=="cancel_slab" || selector=="active_cancel_slab";
        const unsigned phase=selector=="active_cancel_slab"?2:1;
        const unsigned fault=selector=="slab"?1:selector=="borrower"?2:selector=="active_slab"?3:
            selector=="active_borrower"?4:0;
        const auto* enabled=std::getenv("NINFER_EXL3_PINNED_RECURRENT_EXPORT");
        need(enabled && std::string_view(enabled)=="1" && opt.max_concurrency==1,
            "recurrent preparation commit fixture requires pinned export and C1");
        need(Exl3RecurrentSlab::quarantine_count.load()==0,"recurrent commit fixture requires fresh process");
        const auto budget=Exl3RecurrentPinBudget::snapshot();
        const auto control_bytes=Exl3SharedControlAccounting::live_bytes.load();
        {
            Exl3EngineCore owner(opt);
            if(cancellation)owner.cancel_recurrent_growth_after_factory_for_test(phase);
            else owner.fail_recurrent_commit_for_test(fault);
            auto prepared=owner.frontend().prepare_tokens(std::vector<TokenId>{1,2,3,4},true);
            const auto summary=prepared.summary();
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=1;request.execution.allow_prefix_reuse=true;
            request.stop.include_model_defaults=false;
            bool mismatch=false,completed=false,cancelled=false;
            try {
                const auto result=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(10)).wait(nullptr,{});
                cancelled=result.finish_reason==FinishReason::Cancelled && result.generated_token_ids.empty();
                completed=true;
            } catch(const std::exception& error) {
                const auto inspect=[&](auto&& self,const std::exception& current)->void {
                    mismatch|=std::string_view(current.what())=="resource reservation actual extent/domain mismatch" ||
                        std::string_view(current.what())=="bounded resource reservation exceeded ceiling/domain";
                    try{std::rethrow_if_nested(current);}catch(const std::exception& inner){self(self,inner);}
                };
                inspect(inspect,error);
            }
            if(cancellation) {
                need(owner.recurrent_growth_cancel_hit_for_test()==phase,
                    "post-factory cancellation did not reach selected actual Engine lease phase");
                need(completed && cancelled && !mismatch,
                    "post-factory cancellation escaped as allocation failure instead of bounded fallback");
                need(owner.close().reusable(),"recurrent growth cancellation poisoned Engine retirement");
            } else {
                need(owner.recurrent_commit_fault_hit_for_test()==fault,
                    "post-factory refusal did not reach selected actual Engine callback and lease phase");
                need(!completed && mismatch,"post-factory reservation failure was swallowed or replaced");
                (void)owner.close(); // Request failure may poison Engine even when resource cleanup succeeds.
            }
        }
        const auto after=Exl3RecurrentPinBudget::snapshot();
        need(Exl3RecurrentSlab::quarantine_count.load()==0 && after[0]==budget[0] && after[2]==budget[2],
            "post-factory rollback retained physical recurrent budget");
        need(Exl3SharedControlAccounting::live_bytes.load()==control_bytes,
            "post-factory rollback retained physical or borrower control storage");
        Exl3EngineCore reload(opt);need(reload.close().reusable(),"clean commit rollback blocked fresh Engine");
        std::cout<<"ENGINE_RECURRENT_PREPARATION_COMMIT_COMPLETE mode="<<selector<<'\n';return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_RECURRENT_CONSTRUCTOR");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string_view selector(mode);
        need(selector=="retain" || selector=="release" || selector=="event_retain" || selector=="event_release",
            "recurrent constructor selector");
        const bool event=selector=="event_retain" || selector=="event_release";
        const bool retain=selector=="retain" || selector=="event_retain";
        const auto* enabled=std::getenv("NINFER_EXL3_PINNED_RECURRENT_EXPORT");
        need(enabled && std::string_view(enabled)=="1" && opt.max_concurrency==1,
            "recurrent constructor fixture requires pinned export and C1");
        need(Exl3RecurrentSlab::quarantine_count.load()==0,"recurrent fixture requires fresh process");
        const auto budget=Exl3RecurrentPinBudget::snapshot();
        bool original=false,nested_cleanup=false,completed=false;
        {
            Exl3EngineCore owner(opt);owner.fail_recurrent_constructor_for_test((event?2:0)+(retain?1:2));
            auto prepared=owner.frontend().prepare_tokens(std::vector<TokenId>{1,2,3,4},true);
            const auto summary=prepared.summary();
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=1;request.execution.allow_prefix_reuse=true;
            request.stop.include_model_defaults=false;
            try {
                owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(10)).wait(nullptr,{});
                completed=true;
            } catch(const std::exception& error) {
                nested_cleanup=std::string_view(error.what())=="recurrent growth cleanup unresolved";
                const auto inspect=[&](auto&& self,const std::exception& current)->void {
                    if(event) {
                        const auto* failure=dynamic_cast<const Exl3RecurrentEventCreationFailure*>(&current);
                        original|=failure && failure->status==cudaErrorMemoryAllocation;
                    } else original|=std::string_view(current.what())=="injected recurrent post-registration constructor failure";
                    try{std::rethrow_if_nested(current);}catch(const std::exception& inner){self(self,inner);}
                };
                inspect(inspect,error);
            }
            need(owner.recurrent_event_create_failures_for_test()==(event?1:0),
                "recurrent event fault did not reach actual Engine prefix export exactly once");
            if(event && !retain) {
                need(completed && !original && !nested_cleanup,"clean event-create failure did not fall back");
                need(owner.close().reusable(),"clean event-create fallback poisoned Engine");
            } else {
                need(!completed && original && nested_cleanup==retain,"recurrent constructor lost original or cleanup error");
                (void)owner.close(); // Nonrecoverable request failure poisons Engine despite successful cleanup.
            }
        }
        if(retain) {
            const auto* slab=Exl3RecurrentSlab::quarantine_head.load(std::memory_order_acquire);
            need(Exl3RecurrentSlab::quarantine_count.load()==1 && slab && slab->data && slab->registered &&
                !slab->completed && slab->retirement_attempted && slab->cleanup_error==cudaErrorUnknown,
                "failed constructor did not retain exact pre-event registered slab");
            need(slab->construction_error==(event?cudaErrorMemoryAllocation:cudaSuccess),
                "recurrent retirement lost original event provider status");
            need(slab->registration_credit && slab->registration_credit->bytes()==slab->bytes &&
                slab->object_credit && slab->object_credit->bytes()==sizeof(Exl3RecurrentSlab),
                "failed constructor lost registration or object metadata credits");
            const auto after=Exl3RecurrentPinBudget::snapshot();
            need(after[0]==budget[0]+slab->bytes && after[2]==budget[2]+slab->bytes,
                "failed constructor lost retained host budget");
            bool refused=false;
            try{Exl3EngineCore reload(opt);}catch(const std::runtime_error& error) {
                refused=std::string_view(error.what())=="EXL3 unresolved recurrent slab retirement; reload refused";
            }
            need(refused,"Engine reload bypassed recurrent quarantine");
        } else {
            const auto after=Exl3RecurrentPinBudget::snapshot();
            need(Exl3RecurrentSlab::quarantine_count.load()==0 && after[0]==budget[0] && after[2]==budget[2],
                "clean recurrent constructor unwind retained physical budget");
            Exl3EngineCore reload(opt);need(reload.close().reusable(),"clean recurrent unwind blocked fresh Engine");
        }
        std::cout<<"ENGINE_RECURRENT_CONSTRUCTOR_COMPLETE mode="<<selector<<'\n';return 0;
    }
    if(const auto* measure=std::getenv("NINFER_TEST_ENGINE_MEASURE");measure && *measure) {
        if(const auto* configured=std::getenv("NINFER_EXL3_PREFILL_CHUNK");configured && *configured) {
            std::uint32_t value=0;
            const auto end=configured+std::strlen(configured);
            const auto parsed=std::from_chars(configured,end,value);
            need(parsed.ec==std::errc{} && parsed.ptr==end && value,
                "Engine measurement prefill chunk must be a complete positive integer");
            opt.prefill_chunk=value;
        }
        const auto* prompt_path=std::getenv("NINFER_TEST_ENGINE_MEASURE_PROMPT_FILE");
        need(prompt_path && *prompt_path,"Engine measurement requires prompt file");
        std::ifstream input(prompt_path);need(input.good(),"Engine measurement prompt open");
        std::vector<TokenId> tokens;TokenId token=0;while(input>>token)tokens.push_back(token);
        const auto hash=[](std::span<const TokenId> values) {
            std::uint64_t result=1469598103934665603ULL;
            for(const auto value:values){result^=static_cast<std::uint64_t>(value);result*=1099511628211ULL;}
            return result;
        };
        const auto submit=[&](ninfer::exl3::Exl3EngineCore& owner,std::span<const TokenId> prompt_tokens,
            std::uint32_t output_tokens,bool reuse) {
            std::vector<TokenId> owned(prompt_tokens.begin(),prompt_tokens.end());
            auto prepared=owner.frontend().prepare_tokens(std::move(owned),reuse);const auto summary=prepared.summary();
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=output_tokens;request.execution.allow_prefix_reuse=reuse;
            request.stop.include_model_defaults=false;
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(10));
        };
        std::cout<<std::setprecision(17);
        if(std::string_view(measure)=="attention_staging_prefix_state") {
            need(tokens.size()>=4096,"attention staging-prefix state prompt extent");
            using Root=std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>;
            opt.prefill_chunk=1024;opt.max_context=4352;
            opt.kv_capacity=KvCapacityPolicy::explicit_capacity(opt.max_context);
            opt.max_concurrency=1;opt.context_cache.enabled=true;
            ninfer::exl3::Exl3EngineCore owner(opt);
            std::array<std::vector<TokenId>,2> outputs;
            std::array<Root,2> roots;
            std::array<FinishReason,2> reasons{};
            std::array<std::uint64_t,2> staged_banks{},stage_upload{},stage_direct{},
                prefix_hits{},triple_launches{};
            unsigned arm=0;
            owner.observe_terminal_roots_for_test([&](auto root){roots[arm]=std::move(root);});
            for(arm=0;arm<2;++arm) {
                owner.set_attention_staging_for_test(arm!=0);
                const auto before=owner.runtime_stats();
                const auto result=submit(owner,std::span<const TokenId>(tokens.data(),4096),8,false)
                    .wait(nullptr,{});
                owner.set_host_kv_routes_for_test(false,false);
                const auto after=owner.runtime_stats();
                outputs[arm]=result.generated_token_ids;reasons[arm]=result.finish_reason;
                staged_banks[arm]=after.attention_stage_banks-before.attention_stage_banks;
                stage_upload[arm]=after.attention_stage_upload_bytes-before.attention_stage_upload_bytes;
                stage_direct[arm]=after.attention_stage_direct_bytes-before.attention_stage_direct_bytes;
                prefix_hits[arm]=after.private_device_prefix_hit_bytes-before.private_device_prefix_hit_bytes;
                triple_launches[arm]=after.gqa_six_softmax_triple_value_launch_attempts-
                    before.gqa_six_softmax_triple_value_launch_attempts;
                need(outputs[arm].size()==8 && reasons[arm]==FinishReason::OutputLimit && roots[arm],
                    "attention staging-prefix state request incomplete");
            }
            need(outputs[1]==outputs[0] && reasons[1]==reasons[0] && roots[0] && roots[1] &&
                    roots[1]->state()->same_payload(*roots[0]->state()) &&
                    roots[1]->same_projected_conditioning_for_test(*roots[0]),
                "attention staging-prefix changed output, exact state or conditioning");
            need(staged_banks[0]==0 && stage_upload[0]==0 && stage_direct[0]==0 &&
                    staged_banks[1]>0 && stage_upload[1]>0 && stage_direct[1]==stage_upload[1],
                "attention staging-prefix direct route NOT_EXERCISED");
            need(prefix_hits[0]>0 && prefix_hits[1]>0 &&
                    triple_launches[0]>0 && triple_launches[1]==triple_launches[0],
                "attention staging-prefix lost private-prefix or retained six-softmax dispatch");
            owner.observe_terminal_roots_for_test({});roots={};
            owner.set_attention_staging_for_test(false);
            need(owner.close().reusable(),"attention staging-prefix retirement");
            std::cout<<"ATTENTION_STAGING_PREFIX_STATE PASS banks="<<staged_banks[1]
                <<" staged_bytes="<<stage_direct[1]<<" prefix_hits="<<prefix_hits[1]
                <<" triple_launches="<<triple_launches[1]<<'\n';
            return 0;
        }
        if(std::string_view(measure)=="prefill_pinned_batch_state") {
            need(tokens.size()>=4096,"prefill pinned-batch state prompt extent");
            const auto* saved=std::getenv("NINFER_EXL3_PREFILL_PINNED_BATCH_OVER_REGISTERED");
            const auto* saved_registered=std::getenv("NINFER_EXL3_ENGINE_REGISTERED_KV_UPLOAD");
            struct RestorePrefillPinnedBatch {
                std::string value,registered;
                ~RestorePrefillPinnedBatch(){_putenv_s(
                    "NINFER_EXL3_PREFILL_PINNED_BATCH_OVER_REGISTERED",value.c_str());
                    _putenv_s("NINFER_EXL3_ENGINE_REGISTERED_KV_UPLOAD",registered.c_str());}
            } restore{saved?saved:"",saved_registered?saved_registered:""};
            using Root=std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>;
            std::array<std::vector<TokenId>,3> outputs;
            std::array<Root,3> roots;
            std::array<FinishReason,3> reasons{};
            std::array<std::uint64_t,3> plane_calls{},page_planes{},batch_bytes{},
                copy_submissions{},transfer_calls{},h2d_bytes{};
            opt.prefill_chunk=1024;opt.max_context=4352;
            opt.kv_capacity=KvCapacityPolicy::explicit_capacity(opt.max_context);
            opt.max_concurrency=1;opt.context_cache.enabled=true;
            for(unsigned enabled=0;enabled<3;++enabled) {
                _putenv_s("NINFER_EXL3_ENGINE_REGISTERED_KV_UPLOAD",enabled?"1":"0");
                _putenv_s("NINFER_EXL3_PREFILL_PINNED_BATCH_OVER_REGISTERED",
                    enabled==2?"1":"0");
                ninfer::exl3::Exl3EngineCore owner(opt);
                owner.observe_terminal_roots_for_test([&](auto root){roots[enabled]=std::move(root);});
                const auto before=owner.runtime_stats();
                const auto result=submit(owner,std::span<const TokenId>(tokens.data(),4096),8,false).wait(nullptr,{});
                owner.set_host_kv_routes_for_test(false,false);
                const auto after=owner.runtime_stats();
                outputs[enabled]=result.generated_token_ids;reasons[enabled]=result.finish_reason;
                plane_calls[enabled]=after.prefill_pinned_batch_plane_calls-
                    before.prefill_pinned_batch_plane_calls;
                page_planes[enabled]=after.prefill_pinned_batch_page_planes-
                    before.prefill_pinned_batch_page_planes;
                batch_bytes[enabled]=after.prefill_pinned_batch_bytes-
                    before.prefill_pinned_batch_bytes;
                copy_submissions[enabled]=after.host_kv_copy_submissions-
                    before.host_kv_copy_submissions;
                transfer_calls[enabled]=after.host_kv_transfer_calls-
                    before.host_kv_transfer_calls;
                h2d_bytes[enabled]=after.host_kv_h2d_bytes-before.host_kv_h2d_bytes;
                need(outputs[enabled].size()==8 && reasons[enabled]==FinishReason::OutputLimit &&
                        roots[enabled],"prefill pinned-batch state request incomplete");
                need(owner.close().reusable(),"prefill pinned-batch state retirement");
            }
            need(plane_calls[0]==0 && plane_calls[1]==0 &&
                    page_planes[0]==0 && page_planes[1]==0 &&
                    batch_bytes[0]==0 && batch_bytes[1]==0 &&
                    plane_calls[2]>0 && page_planes[2]>0 && batch_bytes[2]>0,
                "prefill pinned-batch state route NOT_EXERCISED");
            need(outputs[1]==outputs[0] && reasons[1]==reasons[0] && roots[0] && roots[1] &&
                    roots[1]->state()->same_payload(*roots[0]->state()),
                "registered-page control changed output or exact terminal state");
            need(outputs[2]==outputs[1] && reasons[2]==reasons[1],
                "prefill pinned-batch changed visible output");
            need(roots[2] && roots[2]->state()->same_payload(*roots[1]->state()),
                "prefill pinned-batch changed exact terminal state");
            need(roots[2]->same_projected_conditioning_for_test(*roots[1]),
                "prefill pinned-batch changed projected conditioning");
            need(transfer_calls[2]==transfer_calls[1] && h2d_bytes[2]==h2d_bytes[1] &&
                    copy_submissions[2]<copy_submissions[1],
                "prefill pinned-batch did not preserve logical transfers while reducing submissions");
            std::cout<<"PREFILL_PINNED_BATCH_STATE PASS plane_calls="<<plane_calls[2]
                <<" page_planes="<<page_planes[2]<<" batch_bytes="<<batch_bytes[2]
                <<" copies_registered="<<copy_submissions[1]<<" copies_on="<<copy_submissions[2]
                <<" transfers="<<transfer_calls[2]<<" h2d_bytes="<<h2d_bytes[2]<<'\n';
            return 0;
        }
        if(std::string_view(measure)=="baseline_matrix") {
            const auto* concurrency_text=
                std::getenv("NINFER_TEST_ENGINE_MEASURE_CONCURRENCY");
            need(concurrency_text &&
                    (std::string_view(concurrency_text)=="1" ||
                     std::string_view(concurrency_text)=="2"),
                "Engine baseline matrix concurrency must be 1 or 2");
            const unsigned concurrency=static_cast<unsigned>(*concurrency_text-'0');
            // The production context ceiling is 32 Ki tokens, so reserve the final
            // 16 positions for this measurement's fixed decode budget.
            constexpr std::array<std::size_t,4> extents{512,4096,16384,32752};
            std::size_t maximum_extent=extents.back();
            if(const auto* selected=std::getenv("NINFER_TEST_ENGINE_MEASURE_MAX_EXTENT");
                    selected && *selected) {
                const auto match=std::find_if(extents.begin(),extents.end(),[&](std::size_t value) {
                    return std::to_string(value)==selected;
                });
                need(match!=extents.end(),"Engine baseline matrix maximum extent unsupported");
                maximum_extent=*match;
            }
            unsigned repetitions=3;
            if(const auto* selected=std::getenv("NINFER_TEST_ENGINE_MEASURE_REPETITIONS");
                    selected && *selected) {
                need(std::string_view(selected)=="1" || std::string_view(selected)=="3",
                    "Engine baseline matrix repetitions must be 1 or 3");
                repetitions=static_cast<unsigned>(*selected-'0');
            }
            unsigned output_tokens=16;
            if(const auto* selected=std::getenv("NINFER_TEST_ENGINE_MEASURE_OUTPUT_TOKENS");
                    selected && *selected) {
                need(std::string_view(selected)=="1" || std::string_view(selected)=="16",
                    "Engine baseline matrix output tokens must be 1 or 16");
                output_tokens=std::string_view(selected)=="1"?1u:16u;
            }
            need(!tokens.empty(),"Engine baseline matrix source tokens are empty");
            opt.prefill_chunk=32;
            opt.max_concurrency=concurrency;opt.context_cache.enabled=true;
            const auto make_input=[&](std::size_t extent,unsigned family) {
                std::vector<TokenId> result;result.reserve(extent);
                for(std::size_t index=0;index<extent;++index)
                    result.push_back(tokens[index%tokens.size()]);
                if(family)result.front()=result.front()==12050?198:12050;
                return result;
            };
            std::size_t completed_cells=0;
            for(const auto extent:extents)if(extent<=maximum_extent) {
                opt.max_context=static_cast<std::uint32_t>(extent+output_tokens);
                opt.kv_capacity=KvCapacityPolicy::explicit_capacity(opt.max_context);
                ninfer::exl3::Exl3EngineCore owner(opt);
                const auto clear_retention=[&] {
                    const auto resident=owner.prefix_retention_metadata();
                    std::vector<ninfer::exl3::Exl3VeriCachePrefixIndex::RetentionDecision>
                        decisions;
                    decisions.reserve(resident.size());
                    for(const auto& entry:resident)
                        decisions.push_back({entry.root,entry.generation,false,0});
                    const auto trimmed=owner.trim_prefix_retention(0,decisions);
                    need(trimmed.inputs_current && trimmed.budget_met &&
                            trimmed.evicted==resident.size() &&
                            owner.prefix_retention_metadata().empty(),
                        "Engine baseline matrix could not reset retained prefixes");
                };
                auto warmup=submit(owner,std::span<const TokenId>(tokens.data(),
                    std::min<std::size_t>(tokens.size(),128)),1,false).wait(nullptr,{});
                owner.set_host_kv_routes_for_test(false,false);
                need(warmup.generated_token_ids.size()==1,
                    "Engine baseline matrix warmup output");
                clear_retention();
                for(unsigned repetition=1;repetition<=repetitions;++repetition) {
                std::array<std::vector<TokenId>,2> inputs{
                    make_input(extent,0),make_input(extent,1)};
                const auto before=owner.runtime_stats();
                const auto cold_started=std::chrono::steady_clock::now();
                std::array<std::optional<ninfer::exl3::Exl3EngineCore::Submission>,2> cold;
                for(unsigned lane=0;lane<concurrency;++lane)
                    cold[lane].emplace(submit(owner,inputs[lane],output_tokens,true));
                std::array<GenerationResult,2> cold_results;
                for(unsigned lane=0;lane<concurrency;++lane)
                    cold_results[lane]=cold[lane]->wait(nullptr,{});
                owner.set_host_kv_routes_for_test(false,false);
                const auto cold_wall=std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-cold_started).count();
                const auto after_cold=owner.runtime_stats();

                const auto warm_started=std::chrono::steady_clock::now();
                std::array<std::optional<ninfer::exl3::Exl3EngineCore::Submission>,2> warm;
                for(unsigned lane=0;lane<concurrency;++lane)
                    warm[lane].emplace(submit(owner,inputs[lane],output_tokens,true));
                std::array<GenerationResult,2> warm_results;
                for(unsigned lane=0;lane<concurrency;++lane)
                    warm_results[lane]=warm[lane]->wait(nullptr,{});
                owner.set_host_kv_routes_for_test(false,false);
                const auto warm_wall=std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-warm_started).count();
                const auto after_warm=owner.runtime_stats();
                const auto memory=owner.memory_summary();
                for(unsigned lane=0;lane<concurrency;++lane) {
                    const auto& cold_result=cold_results[lane];
                    const auto& warm_result=warm_results[lane];
                    need(cold_result.generated_token_ids.size()==output_tokens &&
                            warm_result.generated_token_ids==cold_result.generated_token_ids &&
                            cold_result.finish_reason==FinishReason::OutputLimit &&
                            warm_result.finish_reason==cold_result.finish_reason,
                        "Engine baseline matrix changed cold/warm exact output");
                    std::cout<<"BASELINE_MATRIX concurrency="<<concurrency
                        <<" prompt_tokens="<<extent<<" repetition="<<repetition
                        <<" lane="<<lane<<" generated_tokens="<<output_tokens
                        <<" cold_wall_seconds="<<cold_wall
                        <<" warm_wall_seconds="<<warm_wall
                        <<" cold_prefill_seconds="<<cold_result.timings.prefill_seconds
                        <<" cold_decode_seconds="<<cold_result.timings.decode_seconds
                        <<" warm_prefill_seconds="<<warm_result.timings.prefill_seconds
                        <<" warm_decode_seconds="<<warm_result.timings.decode_seconds
                        <<" warm_reused_prompt_tokens="<<warm_result.reused_prompt_tokens
                        <<" output_hash="<<hash(cold_result.generated_token_ids)
                        <<" cold_computed_prefill_tokens="
                        <<after_cold.computed_prefill_tokens-before.computed_prefill_tokens
                        <<" warm_computed_prefill_tokens="
                        <<after_warm.computed_prefill_tokens-after_cold.computed_prefill_tokens
                        <<" runtime_reservation_bytes="<<memory.runtime_reservation_bytes
                        <<" workspace_capacity_bytes="<<memory.workspace.capacity_bytes
                        <<'\n';
                }
                clear_retention();
                ++completed_cells;
                }
                need(owner.close().reusable(),"Engine baseline matrix retirement");
            }
            std::cout<<"BASELINE_MATRIX_COMPLETE concurrency="<<concurrency
                <<" cells="<<completed_cells<<'\n';
            return 0;
        }
        if(std::string_view(measure)=="single") {
            need(tokens.size()>=4096,"Engine single measurement prompt extent");
            std::uint32_t measurement_max_context=4352;
            if(const auto* requested=std::getenv("NINFER_TEST_ENGINE_MEASURE_MAXCTX");
               requested && *requested) {
                const std::string_view extent(requested);
                need(extent=="4352" || extent=="16640",
                    "Engine single measurement max context menu4352/16640");
                measurement_max_context=extent=="16640"?16640U:4352U;
            }
            opt.max_context=measurement_max_context;
            opt.kv_capacity=KvCapacityPolicy::explicit_capacity(measurement_max_context);
            opt.max_concurrency=1;opt.context_cache.enabled=true;
            ninfer::exl3::Exl3EngineCore owner(opt);
            const bool skip_warmup=std::getenv("NINFER_TEST_ENGINE_MEASURE_SKIP_WARMUP") &&
                std::string_view(std::getenv("NINFER_TEST_ENGINE_MEASURE_SKIP_WARMUP"))=="1";
            if(!skip_warmup) {
                auto warm=submit(owner,std::span<const TokenId>(tokens.data(),128),16,false).wait(nullptr,{});
                owner.set_host_kv_routes_for_test(false,false);
                need(warm.generated_token_ids.size()==16,"Engine single measurement warmup output");
            }
            const auto before=owner.runtime_stats();
            const auto attention_graph_before=
                ninfer::exl3::exl3_prefill_attention_chain_global_snapshot();
            const auto shared_score_before=
                ninfer::exl3::exl3_prefill_attention_shared_score_global_snapshot();
            const bool profile=std::getenv("NINFER_TEST_ENGINE_MEASURE_PROFILE") &&
                std::string_view(std::getenv("NINFER_TEST_ENGINE_MEASURE_PROFILE"))=="1";
            if(profile)need(cudaProfilerStart()==cudaSuccess,"Engine measurement profiler start");
            GenerationResult result;
            try {
                result=submit(owner,std::span<const TokenId>(tokens.data(),4096),128,false).wait(nullptr,{});
            } catch(...) {
                if(profile)cudaProfilerStop();
                throw;
            }
            if(profile)need(cudaProfilerStop()==cudaSuccess,"Engine measurement profiler stop");
            owner.set_host_kv_routes_for_test(false,false);
            const auto after=owner.runtime_stats();const auto memory=owner.memory_summary();
            const auto attention_graph_after=
                ninfer::exl3::exl3_prefill_attention_chain_global_snapshot();
            const auto shared_score_after=
                ninfer::exl3::exl3_prefill_attention_shared_score_global_snapshot();
            need(result.generated_token_ids.size()==128 && result.finish_reason==FinishReason::OutputLimit,
                "Engine single measurement did not reach output limit");
            std::ostringstream accepted_prefix_trace;
            for(std::size_t index=0;
                index<result.speculative.accepted_prefix_per_round.size();++index) {
                if(index) accepted_prefix_trace << ',';
                accepted_prefix_trace << static_cast<unsigned>(
                    result.speculative.accepted_prefix_per_round[index]);
            }
            std::ostringstream first_proposal_trace;
            for(std::size_t index=0;
                index<result.speculative.first_proposed_tokens.size();++index) {
                if(index) first_proposal_trace << ',';
                first_proposal_trace << result.speculative.first_proposed_tokens[index];
            }
            std::cout<<"MEASURE mode=single prompt_tokens="<<result.prompt.prompt_tokens
                <<" generated_tokens="<<result.generated_token_ids.size()
                <<" prefill_chunk="<<opt.prefill_chunk
                <<" engine_first_token_seconds="<<result.timings.first_token_seconds
                <<" prefill_seconds="<<result.timings.prefill_seconds
                <<" decode_seconds="<<result.timings.decode_seconds
                <<" total_seconds="<<result.timings.total_seconds
                <<" output_hash="<<hash(result.generated_token_ids)
                <<" computed_prefill_tokens="<<after.computed_prefill_tokens-before.computed_prefill_tokens
                <<" committed_decode_tokens="<<after.committed_decode_tokens-before.committed_decode_tokens
                <<" visible_model_tokens="<<result.token_accounting.visible_model_tokens
                <<" injected_control_tokens="<<result.token_accounting.injected_control_tokens
                <<" hidden_terminal_tokens="<<result.token_accounting.hidden_terminal_tokens
                <<" diagnostic_state_rows="<<result.token_accounting.diagnostic_state_rows
                <<" proposed_rows="<<result.speculative.proposed_rows
                <<" verified_rows="<<result.speculative.verified_rows
                <<" replayed_rows="<<result.speculative.replayed_rows
                <<" repair_checkpoint_captured_bytes="<<result.speculative.repair_checkpoint_captured_bytes
                <<" repair_checkpoint_restores="<<result.speculative.repair_checkpoint_restores
                <<" repair_checkpoint_reconstructed_rows="<<result.speculative.repair_checkpoint_reconstructed_rows
                <<" repair_checkpoint_fallback_rows="<<result.speculative.repair_checkpoint_fallback_rows
                <<" committed_model_rows="<<result.speculative.committed_model_rows
                <<" externally_visible_model_rows="<<result.speculative.externally_visible_model_rows
                <<" hidden_terminal_rows="<<result.speculative.hidden_terminal_rows
                <<" diagnostic_rows="<<result.speculative.diagnostic_rows
                <<" neural_input_rows="<<result.speculative.neural_input_rows
                <<" returned_suffix_rows="<<result.speculative.returned_suffix_rows
                <<" discarded_suffix_rows="<<result.speculative.discarded_suffix_rows
                <<" suffix_proposal_calls="<<after.suffix_proposal_calls-before.suffix_proposal_calls
                <<" suffix_proposal_rows="<<after.suffix_proposal_rows-before.suffix_proposal_rows
                <<" suffix_proposal_misses="<<after.suffix_proposal_misses-before.suffix_proposal_misses
                <<" suffix_published_rounds="<<after.suffix_published_rounds-before.suffix_published_rounds
                <<" suffix_accepted_rows="<<after.suffix_accepted_rows-before.suffix_accepted_rows
                <<" suffix_repair_rows="<<after.suffix_repair_rows-before.suffix_repair_rows
                <<" suffix_skipped_neural_blocks="<<after.suffix_skipped_neural_blocks-before.suffix_skipped_neural_blocks
                <<" device_seed_handoffs="<<result.speculative.device_seed_handoffs
                <<" device_seed_host_fallbacks="<<result.speculative.device_seed_host_fallbacks
                <<" draft_local_topk_calls="<<result.speculative.draft_local_topk_calls
                <<" draft_dense_kmajor_launches="<<result.speculative.draft_dense_kmajor_launches
                <<" oscar_eager_cohort_attempts="<<result.speculative.oscar_eager_cohort_attempts
                <<" oscar_eager_cohort_dispatches="<<result.speculative.oscar_eager_cohort_dispatches
                <<" oscar_eager_cohort_latch_misses="<<result.speculative.oscar_eager_cohort_latch_misses
                <<" oscar_eager_cohort_boundary_fallbacks="
                    <<result.speculative.oscar_eager_cohort_boundary_fallbacks
                <<" oscar_eager_cohort_malformed="<<result.speculative.oscar_eager_cohort_malformed
                <<" staged_b8_verifier_calls="<<result.speculative.staged_b8_verifier_calls
                <<" staged_b8_first_half_exits="<<result.speculative.staged_b8_first_half_exits
                <<" staged_b8_second_half_calls="<<result.speculative.staged_b8_second_half_calls
                <<" staged_b8_skipped_verification_rows="
                    <<result.speculative.staged_b8_skipped_verification_rows
                <<" accepted_prefix_per_round="<<accepted_prefix_trace.str()
                <<" first_proposed_tokens="<<first_proposal_trace.str()
                <<" committed_tap_device_bytes="<<result.speculative.committed_tap_device_bytes
                <<" committed_tap_host_export_rows_avoided="
                    <<result.speculative.committed_tap_host_export_rows_avoided
                <<" k6_stream_reduction_calls="<<after.k6_stream_reduction_calls-before.k6_stream_reduction_calls
                <<" extended_stream_reduction_calls="<<after.extended_stream_reduction_calls-before.extended_stream_reduction_calls
                <<" k6_gateup_warpgroup_async_calls="<<after.k6_gateup_warpgroup_async_calls-before.k6_gateup_warpgroup_async_calls
                <<" k6_gateup_n32_pair_cta_calls="<<after.k6_gateup_n32_pair_cta_calls-before.k6_gateup_n32_pair_cta_calls
                <<" k6_fast_decode_calls="<<after.k6_fast_decode_calls-before.k6_fast_decode_calls
                <<" k6_fast_decode_rows="<<after.k6_fast_decode_rows-before.k6_fast_decode_rows
                <<" k6_rowpair_n64_calls="<<after.k6_rowpair_n64_calls-before.k6_rowpair_n64_calls
                <<" k6_rowpair_n64_rows="<<after.k6_rowpair_n64_rows-before.k6_rowpair_n64_rows
                <<" k6_down_rowpair_calls="<<after.k6_down_rowpair_calls-before.k6_down_rowpair_calls
                <<" k6_down_rowpair_rows="<<after.k6_down_rowpair_rows-before.k6_down_rowpair_rows
                <<" shape4_n64_calls="<<after.shape4_n64_calls-before.shape4_n64_calls
                <<" shape4_n64_rows="<<after.shape4_n64_rows-before.shape4_n64_rows
                <<" reduce_shfl_min_barrier_calls="<<after.reduce_shfl_min_barrier_calls-before.reduce_shfl_min_barrier_calls
                <<" reduce_shfl_min_barrier_rows="<<after.reduce_shfl_min_barrier_rows-before.reduce_shfl_min_barrier_rows
                <<" k6_gateup_n32_pair_cta_rows="<<after.k6_gateup_n32_pair_cta_rows-before.k6_gateup_n32_pair_cta_rows
                <<" k7_tiles64_exact_splits_calls="<<after.k7_tiles64_exact_splits_calls-before.k7_tiles64_exact_splits_calls
                <<" target_k8_kv_prefill_async_a_calls="<<after.target_k8_kv_prefill_async_a_calls-before.target_k8_kv_prefill_async_a_calls
                <<" target_k8_kv_prefill_async_a_rows="<<after.target_k8_kv_prefill_async_a_rows-before.target_k8_kv_prefill_async_a_rows
                <<" target_down_k6_async_a_calls="<<after.target_down_k6_async_a_calls-before.target_down_k6_async_a_calls
                <<" target_gateup_k6_n16_calls="<<after.target_gateup_k6_n16_calls-before.target_gateup_k6_n16_calls
                <<" host_kv_gdn_segment_graph_captures_total="<<after.host_kv_gdn_segment_graph_captures
                <<" host_kv_gdn_segment_graph_replays="<<after.host_kv_gdn_segment_graph_replays-before.host_kv_gdn_segment_graph_replays
                <<" host_kv_gdn_segment_graph_capture_seconds_total="<<after.host_kv_gdn_segment_graph_capture_seconds
                <<" host_kv_full_layer_graph_captures_total="<<after.host_kv_full_layer_graph_captures
                <<" host_kv_full_layer_graph_replays="<<after.host_kv_full_layer_graph_replays-before.host_kv_full_layer_graph_replays
                <<" host_kv_full_layer_graph_six_softmax_triple_captures_total="<<after.host_kv_full_layer_graph_six_softmax_triple_captures
                <<" host_kv_full_layer_graph_k6_stream_reduction_captures_total="<<after.host_kv_full_layer_graph_k6_stream_reduction_captures
                <<" host_kv_full_layer_graph_extended_stream_reduction_captures_total="<<after.host_kv_full_layer_graph_extended_stream_reduction_captures
                <<" host_kv_full_layer_graph_target_down_k6_async_a_captures_total="<<after.host_kv_full_layer_graph_target_down_k6_async_a_captures
                <<" host_kv_full_layer_graph_target_k6_small_m_async_a_captures_total="<<after.host_kv_full_layer_graph_target_k6_small_m_async_a_captures
                <<" host_kv_full_layer_graph_target_k7_small_m_async_a_captures_total="<<after.host_kv_full_layer_graph_target_k7_small_m_async_a_captures
                <<" host_kv_full_layer_graph_capture_seconds_total="<<after.host_kv_full_layer_graph_capture_seconds
                <<" host_kv_mlp_tail_graph_captures_total="<<after.host_kv_mlp_tail_graph_captures
                <<" host_kv_mlp_tail_graph_replays="<<after.host_kv_mlp_tail_graph_replays-before.host_kv_mlp_tail_graph_replays
                <<" host_kv_mlp_tail_graph_capture_seconds_total="<<after.host_kv_mlp_tail_graph_capture_seconds
                <<" host_kv_transaction_checkpoint_graph_captures="
                <<after.host_kv_transaction_checkpoint_graph_captures-
                    before.host_kv_transaction_checkpoint_graph_captures
                <<" host_kv_transaction_checkpoint_graph_replays="
                <<after.host_kv_transaction_checkpoint_graph_replays-
                    before.host_kv_transaction_checkpoint_graph_replays
                <<" host_kv_transaction_checkpoint_graph_capture_seconds="
                <<after.host_kv_transaction_checkpoint_graph_capture_seconds-
                    before.host_kv_transaction_checkpoint_graph_capture_seconds
                <<" host_kv_transaction_recurrent_trace_alias_layers="
                <<after.host_kv_transaction_recurrent_trace_alias_layers-
                    before.host_kv_transaction_recurrent_trace_alias_layers
                <<" host_kv_transaction_recurrent_trace_copy_bytes_saved="
                <<after.host_kv_transaction_recurrent_trace_copy_bytes_saved-
                    before.host_kv_transaction_recurrent_trace_copy_bytes_saved
                <<" recurrent_export_calls="
                    <<after.recurrent_export_calls-before.recurrent_export_calls
                <<" recurrent_export_bytes="
                    <<after.recurrent_export_bytes-before.recurrent_export_bytes
                <<" recurrent_export_copy_submissions="
                    <<after.recurrent_export_copy_submissions-
                        before.recurrent_export_copy_submissions
                <<" recurrent_export_batched_copy_calls="
                    <<after.recurrent_export_batched_copy_calls-
                        before.recurrent_export_batched_copy_calls
                <<" recurrent_export_batched_copy_ranges="
                    <<after.recurrent_export_batched_copy_ranges-
                        before.recurrent_export_batched_copy_ranges
                <<" recurrent_export_seconds="
                    <<after.recurrent_export_seconds-before.recurrent_export_seconds
                <<" recurrent_export_fence_seconds="
                    <<after.recurrent_export_fence_seconds-
                        before.recurrent_export_fence_seconds
                <<" target_k6_small_m_async_a_calls="<<after.target_k6_small_m_async_a_calls-before.target_k6_small_m_async_a_calls
                <<" target_k7_small_m_async_a_calls="<<after.target_k7_small_m_async_a_calls-before.target_k7_small_m_async_a_calls
                <<" target_k5_small_m_batch_calls="<<after.target_k5_small_m_batch_calls-before.target_k5_small_m_batch_calls
                <<" eager_mlp_gateup_concurrent_calls="<<after.eager_mlp_gateup_concurrent_calls-before.eager_mlp_gateup_concurrent_calls
                <<" prefill_qkv_concurrent_calls="<<after.prefill_qkv_concurrent_calls-before.prefill_qkv_concurrent_calls
                <<" prefill_qkv_concurrent_rows="<<after.prefill_qkv_concurrent_rows-before.prefill_qkv_concurrent_rows
                <<" prefill_projection_graph_captures="<<after.prefill_projection_graph_captures-before.prefill_projection_graph_captures
                <<" prefill_projection_graph_replays="<<after.prefill_projection_graph_replays-before.prefill_projection_graph_replays
                <<" prefill_projection_graph_binding_fallbacks="<<after.prefill_projection_graph_binding_fallbacks-before.prefill_projection_graph_binding_fallbacks
                <<" prefill_projection_chain_graph_captures="<<after.prefill_projection_chain_graph_captures-before.prefill_projection_chain_graph_captures
                <<" prefill_projection_chain_graph_replays="<<after.prefill_projection_chain_graph_replays-before.prefill_projection_chain_graph_replays
                <<" prefill_projection_chain_graph_fallbacks="<<after.prefill_projection_chain_graph_fallbacks-before.prefill_projection_chain_graph_fallbacks
                <<" prefill_attention_chain_graph_capture_setups="
                    <<attention_graph_after.capture_setups-attention_graph_before.capture_setups
                <<" prefill_attention_chain_graph_captures="
                    <<attention_graph_after.captures-attention_graph_before.captures
                <<" prefill_attention_chain_graph_replays="
                    <<attention_graph_after.replays-attention_graph_before.replays
                <<" prefill_attention_chain_graph_fallbacks="
                    <<attention_graph_after.eager_fallbacks-attention_graph_before.eager_fallbacks
                <<" prefill_attention_chain_graph_binding_mismatches="
                    <<attention_graph_after.binding_mismatches-
                        attention_graph_before.binding_mismatches
                <<" prefill_attention_shared_score_launch_attempts="
                    <<shared_score_after.launch_attempts-shared_score_before.launch_attempts
                <<" prefill_attention_shared_score_row_attempts="
                    <<shared_score_after.row_attempts-shared_score_before.row_attempts
                <<" prefill_attention_shared_score_global_bytes_eliminated="
                    <<shared_score_after.global_score_bytes_eliminated-
                        shared_score_before.global_score_bytes_eliminated
                <<" prefill_attention_shared_score_threads256_launch_attempts="
                    <<shared_score_after.threads256_launch_attempts-
                        shared_score_before.threads256_launch_attempts
                <<" prefill_attention_shared_score_parallel_softmax_launch_attempts="
                    <<shared_score_after.parallel_softmax_launch_attempts-
                        shared_score_before.parallel_softmax_launch_attempts
                <<" prefill_attention_shared_score_head_split256_launch_attempts="
                    <<shared_score_after.head_split256_launch_attempts-
                        shared_score_before.head_split256_launch_attempts
                <<" prefill_attention_shared_score_dimension_split256_launch_attempts="
                    <<shared_score_after.dimension_split256_launch_attempts-
                        shared_score_before.dimension_split256_launch_attempts
                <<" target_prefill_gate_up_pair_attempts="<<after.target_prefill_gate_up_pair_attempts-before.target_prefill_gate_up_pair_attempts
                <<" target_prefill_gate_up_pair_calls="<<after.target_prefill_gate_up_pair_calls-before.target_prefill_gate_up_pair_calls
                <<" target_prefill_gate_up_pair_rows="<<after.target_prefill_gate_up_pair_rows-before.target_prefill_gate_up_pair_rows
                <<" gqa_six_softmax_triple_v_tile_launch_attempts="<<after.gqa_six_softmax_triple_v_tile_launch_attempts-before.gqa_six_softmax_triple_v_tile_launch_attempts
                <<" gqa_six_softmax_triple_v_tile_row_attempts="<<after.gqa_six_softmax_triple_v_tile_row_attempts-before.gqa_six_softmax_triple_v_tile_row_attempts
                <<" gqa_six_softmax_triple_full_cta_launch_attempts="<<after.gqa_six_softmax_triple_full_cta_launch_attempts-before.gqa_six_softmax_triple_full_cta_launch_attempts
                <<" gqa_six_softmax_triple_full_cta_row_attempts="<<after.gqa_six_softmax_triple_full_cta_row_attempts-before.gqa_six_softmax_triple_full_cta_row_attempts
                <<" gqa_six_softmax_triple_threads128_launch_attempts="<<after.gqa_six_softmax_triple_threads128_launch_attempts-before.gqa_six_softmax_triple_threads128_launch_attempts
                <<" gqa_six_softmax_triple_threads128_row_attempts="<<after.gqa_six_softmax_triple_threads128_row_attempts-before.gqa_six_softmax_triple_threads128_row_attempts
                <<" gqa_six_softmax_triple_score_tile_launch_attempts="<<after.gqa_six_softmax_triple_score_tile_launch_attempts-before.gqa_six_softmax_triple_score_tile_launch_attempts
                <<" gqa_six_softmax_triple_score_tile_row_attempts="<<after.gqa_six_softmax_triple_score_tile_row_attempts-before.gqa_six_softmax_triple_score_tile_row_attempts
                <<" gqa_six_softmax_triple_scalar_dim_launch_attempts="<<after.gqa_six_softmax_triple_scalar_dim_launch_attempts-before.gqa_six_softmax_triple_scalar_dim_launch_attempts
                <<" gqa_six_softmax_triple_scalar_dim_row_attempts="<<after.gqa_six_softmax_triple_scalar_dim_row_attempts-before.gqa_six_softmax_triple_scalar_dim_row_attempts
                <<" gqa_six_softmax_triple_two_query_launch_attempts="<<after.gqa_six_softmax_triple_two_query_launch_attempts-before.gqa_six_softmax_triple_two_query_launch_attempts
                <<" gqa_six_softmax_triple_two_query_row_attempts="<<after.gqa_six_softmax_triple_two_query_row_attempts-before.gqa_six_softmax_triple_two_query_row_attempts
                <<" conditional_second_block_calls="<<after.conditional_second_block_calls-before.conditional_second_block_calls
                <<" gqa_six_softmax_triple_fused_launch_attempts="<<after.gqa_six_softmax_triple_fused_launch_attempts-before.gqa_six_softmax_triple_fused_launch_attempts
                <<" gqa_six_softmax_triple_fused_row_attempts="<<after.gqa_six_softmax_triple_fused_row_attempts-before.gqa_six_softmax_triple_fused_row_attempts
                <<" gqa_six_softmax_tile512_launch_attempts="<<after.gqa_six_softmax_tile512_launch_attempts-before.gqa_six_softmax_tile512_launch_attempts
                <<" gqa_six_softmax_tile512_row_attempts="<<after.gqa_six_softmax_tile512_row_attempts-before.gqa_six_softmax_tile512_row_attempts
                <<" reconstruction_submissions="<<after.reconstruction_submissions-before.reconstruction_submissions
                <<" reconstruction_submitted_rows="<<after.reconstruction_submitted_rows-before.reconstruction_submitted_rows
                <<" reconstruction_k6_submissions="<<after.reconstruction_k6_submissions-before.reconstruction_k6_submissions
                <<" reconstruction_k6_submitted_rows="<<after.reconstruction_k6_submitted_rows-before.reconstruction_k6_submitted_rows
                <<" fused_gate_up_submissions="<<after.fused_gate_up_submissions-before.fused_gate_up_submissions
                <<" gqa_six_query_pair_score_launch_attempts="
                <<after.gqa_six_query_pair_score_launch_attempts-before.gqa_six_query_pair_score_launch_attempts
                <<" gqa_six_query_pair_score_row_attempts="
                <<after.gqa_six_query_pair_score_row_attempts-before.gqa_six_query_pair_score_row_attempts
                <<" gqa_six_score_k_tile64_launch_attempts="
                <<after.gqa_six_score_k_tile64_launch_attempts-before.gqa_six_score_k_tile64_launch_attempts
                <<" gqa_six_score_k_tile64_row_attempts="
                <<after.gqa_six_score_k_tile64_row_attempts-before.gqa_six_score_k_tile64_row_attempts
                <<" gqa_six_softmax_triple_value_launch_attempts="
                <<after.gqa_six_softmax_triple_value_launch_attempts-before.gqa_six_softmax_triple_value_launch_attempts
                <<" gqa_six_softmax_triple_value_row_attempts="
                <<after.gqa_six_softmax_triple_value_row_attempts-before.gqa_six_softmax_triple_value_row_attempts
                <<" gqa_six_softmax_triple_pair_dimensions_launch_attempts="
                <<after.gqa_six_softmax_triple_pair_dimensions_launch_attempts-
                    before.gqa_six_softmax_triple_pair_dimensions_launch_attempts
                <<" gqa_six_softmax_triple_pair_dimensions_row_attempts="
                <<after.gqa_six_softmax_triple_pair_dimensions_row_attempts-
                    before.gqa_six_softmax_triple_pair_dimensions_row_attempts
                <<" gqa_six_softmax_six_values_single_load_launch_attempts="
                <<after.gqa_six_softmax_six_values_single_load_launch_attempts-
                    before.gqa_six_softmax_six_values_single_load_launch_attempts
                <<" gqa_six_softmax_six_values_single_load_row_attempts="
                <<after.gqa_six_softmax_six_values_single_load_row_attempts-
                    before.gqa_six_softmax_six_values_single_load_row_attempts
                <<" gqa_six_softmax_triple_key_pair_launch_attempts="
                <<after.gqa_six_softmax_triple_key_pair_launch_attempts-
                    before.gqa_six_softmax_triple_key_pair_launch_attempts
                <<" gqa_six_softmax_triple_key_pair_row_attempts="
                <<after.gqa_six_softmax_triple_key_pair_row_attempts-
                    before.gqa_six_softmax_triple_key_pair_row_attempts
                <<" attention_stage_banks="<<after.attention_stage_banks-before.attention_stage_banks
                <<" attention_stage_upload_bytes="<<after.attention_stage_upload_bytes-before.attention_stage_upload_bytes
                <<" attention_stage_consumed_bytes="<<after.attention_stage_consumed_bytes-before.attention_stage_consumed_bytes
                <<" attention_stage_direct_bytes="<<after.attention_stage_direct_bytes-before.attention_stage_direct_bytes
                <<" private_device_prefix_hit_bytes="<<after.private_device_prefix_hit_bytes-before.private_device_prefix_hit_bytes
                <<" private_device_prefix_fill_bytes="<<after.private_device_prefix_fill_bytes-before.private_device_prefix_fill_bytes
                <<" private_device_prefix_segmented_bytes="<<after.private_device_prefix_segmented_bytes-before.private_device_prefix_segmented_bytes
                <<" private_device_prefix_forward_publish_forwards="
                <<after.private_device_prefix_forward_publish_forwards-
                    before.private_device_prefix_forward_publish_forwards
                <<" private_device_prefix_partial_hit_bytes="
                <<after.private_device_prefix_partial_hit_bytes-
                    before.private_device_prefix_partial_hit_bytes
                <<" private_device_prefix_forward_publish_bytes="
                <<after.private_device_prefix_forward_publish_bytes-
                    before.private_device_prefix_forward_publish_bytes
                <<" host_kv_h2d_bytes="<<after.host_kv_h2d_bytes-before.host_kv_h2d_bytes
                <<" host_kv_d2h_bytes="<<after.host_kv_d2h_bytes-before.host_kv_d2h_bytes
                <<" host_kv_transfer_calls="<<after.host_kv_transfer_calls-before.host_kv_transfer_calls
                <<" host_kv_copy_submissions="<<after.host_kv_copy_submissions-before.host_kv_copy_submissions
                <<" host_kv_pinned_slot_waits="
                    <<after.host_kv_pinned_slot_waits-before.host_kv_pinned_slot_waits
                <<" host_kv_pinned_staging_bytes="
                    <<after.host_kv_pinned_staging_bytes
                <<" registered_kv_upload_peak_pending="
                    <<after.registered_kv_upload_peak_pending
                <<" exact_kv_export_batched_calls="<<after.exact_kv_export_batched_calls-before.exact_kv_export_batched_calls
                <<" exact_kv_export_batched_ranges="<<after.exact_kv_export_batched_ranges-before.exact_kv_export_batched_ranges
                <<" exact_kv_export_batched_bytes="<<after.exact_kv_export_batched_bytes-before.exact_kv_export_batched_bytes
                <<" exact_kv_export_batched_fences="<<after.exact_kv_export_batched_fences-before.exact_kv_export_batched_fences
                <<" exact_kv_export_saved_fences="<<after.exact_kv_export_saved_fences-before.exact_kv_export_saved_fences
                <<" prefill_pinned_batch_plane_calls="<<after.prefill_pinned_batch_plane_calls-before.prefill_pinned_batch_plane_calls
                <<" prefill_pinned_batch_page_planes="<<after.prefill_pinned_batch_page_planes-before.prefill_pinned_batch_page_planes
                <<" prefill_pinned_batch_bytes="<<after.prefill_pinned_batch_bytes-before.prefill_pinned_batch_bytes
                <<" host_kv_banked_d2h_forwards="<<after.host_kv_banked_d2h_forwards-before.host_kv_banked_d2h_forwards
                <<" host_kv_banked_d2h_planes="<<after.host_kv_banked_d2h_planes-before.host_kv_banked_d2h_planes
                <<" host_kv_banked_d2h_rows="<<after.host_kv_banked_d2h_rows-before.host_kv_banked_d2h_rows
                <<" host_kv_banked_d2h_bytes="<<after.host_kv_banked_d2h_bytes-before.host_kv_banked_d2h_bytes
                <<" host_kv_banked_d2h_drains="<<after.host_kv_banked_d2h_drains-before.host_kv_banked_d2h_drains
                <<" queue_wait_seconds="<<result.engine_timing.queue_wait_seconds
                <<" engine_boundary_exposed_seconds="<<result.engine_timing.engine_boundary_exposed_seconds
                <<" program_submit_exposed_seconds="<<result.engine_timing.program_submit_exposed_seconds
                <<" program_post_exposed_seconds="<<result.engine_timing.program_post_exposed_seconds
                <<" engine_commit_output_exposed_seconds="<<result.engine_timing.engine_commit_output_exposed_seconds
                <<" engine_maintenance_exposed_seconds="<<result.engine_timing.engine_maintenance_exposed_seconds
                <<" device_wait_exposed_seconds="<<result.engine_timing.device_wait_exposed_seconds
                <<" decode_host_exposed_seconds="<<result.engine_timing.decode_host_exposed_seconds
                <<" decode_device_wait_exposed_seconds="<<result.engine_timing.decode_device_wait_exposed_seconds
                <<" prefill_units="<<result.engine_timing.prefill_units
                <<" decode_rounds="<<result.engine_timing.decode_rounds
                <<" control_units="<<result.engine_timing.control_units
                <<" reused_prompt_tokens="<<result.reused_prompt_tokens
                <<" runtime_reservation_bytes="<<memory.runtime_reservation_bytes
                <<" workspace_capacity_bytes="<<memory.workspace.capacity_bytes
                <<" prefix_preparation_metadata_bytes="<<after.prefix_preparation_metadata_bytes
                <<'\n';
            need(owner.close().reusable(),"Engine single measurement retirement");return 0;
        }
        if(std::string_view(measure)=="shared_pair") {
            need(tokens.size()>=768,"Engine shared measurement prompt extent");
            opt.max_context=1024;opt.kv_capacity=KvCapacityPolicy::explicit_capacity(1024);
            opt.max_concurrency=2;opt.context_cache.enabled=true;
            ninfer::exl3::Exl3EngineCore owner(opt);
            auto warm=submit(owner,std::span<const TokenId>(tokens.data(),128),1,false).wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            need(warm.generated_token_ids.size()==1,"Engine shared measurement warmup output");
            std::mutex gate_mutex;std::condition_variable gate;std::size_t producer_lane=2;bool waiter_seen=false;
            const bool shared=std::getenv("NINFER_EXL3_ENGINE_SHARED_PREPARATION") &&
                std::string_view(std::getenv("NINFER_EXL3_ENGINE_SHARED_PREPARATION"))=="1";
            if(shared)owner.observe_prefix_preparation_for_test([&](std::size_t lane,int) {
                std::unique_lock lock(gate_mutex);
                if(producer_lane==2){producer_lane=lane;gate.notify_all();
                    if(!gate.wait_for(lock,std::chrono::seconds(30),[&]{return waiter_seen;}))
                        throw std::runtime_error("Engine shared measurement join timeout");}
                else if(lane!=producer_lane){waiter_seen=true;gate.notify_all();}
            });
            const auto before=owner.runtime_stats();const auto started=std::chrono::steady_clock::now();
            auto first=submit(owner,std::span<const TokenId>(tokens.data(),768),1,true);
            if(shared){std::unique_lock lock(gate_mutex);need(gate.wait_for(lock,std::chrono::seconds(30),
                [&]{return producer_lane<2;}),"Engine shared measurement producer start");}
            auto second=submit(owner,std::span<const TokenId>(tokens.data(),768),1,true);
            const auto a=first.wait(nullptr,{}),b=second.wait(nullptr,{});
            const auto wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            owner.set_host_kv_routes_for_test(false,false);if(shared)owner.observe_prefix_preparation_for_test({});
            const auto after=owner.runtime_stats();const auto memory=owner.memory_summary();
            need(a.generated_token_ids==b.generated_token_ids && a.generated_token_ids.size()==1,
                "Engine shared measurement output mismatch");
            std::cout<<"MEASURE mode=shared_pair prompt_tokens="<<a.prompt.prompt_tokens
                <<" generated_tokens_per_request="<<a.generated_token_ids.size()
                <<" wall_seconds="<<wall<<" prefill_a_seconds="<<a.timings.prefill_seconds
                <<" prefill_b_seconds="<<b.timings.prefill_seconds
                <<" output_hash="<<hash(a.generated_token_ids)
                <<" preparation_returns="<<after.prefix_preparation_returns-before.prefix_preparation_returns
                <<" concurrent_returns="<<after.concurrent_prefix_preparation_returns-before.concurrent_prefix_preparation_returns
                <<" joins="<<after.prefix_preparation_joins-before.prefix_preparation_joins
                <<" cache_hits="<<after.prefix_preparation_cache_hits-before.prefix_preparation_cache_hits
                <<" computed_prefill_tokens="<<after.computed_prefill_tokens-before.computed_prefill_tokens
                <<" runtime_reservation_bytes="<<memory.runtime_reservation_bytes
                <<" workspace_capacity_bytes="<<memory.workspace.capacity_bytes
                <<" prefix_preparation_metadata_bytes="<<after.prefix_preparation_metadata_bytes
                <<'\n';
            need(owner.close().reusable(),"Engine shared measurement retirement");return 0;
        }
        throw std::invalid_argument("unknown Engine measurement mode");
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_RETIREMENT_METADATA");mode && std::string_view(mode)=="1") {
        std::uint64_t per_lane=0,per_stream=0,target_records=0,draft_records=0,context_records=0,private_draft_records=0;
        for(unsigned lanes:{1u,2u}) {
            opt.max_concurrency=lanes;
            ninfer::exl3::Exl3EngineCore owner(opt);
            const auto metadata=owner.runtime_stats().execution_retirement_metadata_bytes;
            const auto startup=owner.runtime_stats();
            const auto enabled=[](const char* name){const auto* value=std::getenv(name);return value && std::string_view(value)=="1";};
            const auto represented_device=startup.reserved_context_startup_device_bytes+
                startup.reserved_continuation_startup_device_bytes+startup.reserved_lane_startup_device_bytes+
                startup.reserved_repair_checkpoint_startup_device_bytes+
                startup.reserved_engine_tap_device_bytes+
                startup.reconstruction_device_bytes+startup.attention_stage_device_bytes+
                startup.private_device_prefix_bytes+startup.shared_device_prefix_bytes;
            need(owner.memory_summary().runtime_reservation_bytes>=represented_device,
                "Engine workspace summary omits represented reserved startup groups");
            {
                need(startup.reserved_context_startup_lanes==lanes,"Engine bypassed reserved context startup");
                need(startup.reserved_context_startup_device_bytes>0,"reserved context storage omitted");
                need(startup.reserved_context_fixed_owner_metadata_bytes==
                    lanes*ninfer::exl3::Exl3TextContext::fixed_owner_metadata_bytes(),
                    "Engine omitted fixed context owner metadata");
                need(startup.reserved_continuation_startup_device_bytes==lanes*8ULL*(5120*4+248320*22),
                    "Engine continuation credit differs from independent B8 extent");
                need(startup.reserved_continuation_owner_metadata_bytes==
                    lanes*ninfer::exl3::Exl3TextContext::continuation_owner_metadata_bytes(),
                    "Engine omitted committed continuation owner metadata");
                need(enabled("NINFER_EXL3_REPAIR_CHECKPOINT")?
                        startup.reserved_repair_checkpoint_startup_device_bytes>0 &&
                            startup.reserved_repair_checkpoint_startup_device_bytes%lanes==0:
                        startup.reserved_repair_checkpoint_startup_device_bytes==0,
                    "Engine repair checkpoint startup option/accounting mismatch");
                need(startup.reserved_repair_checkpoint_owner_metadata_bytes==
                    (enabled("NINFER_EXL3_REPAIR_CHECKPOINT")?
                        lanes*ninfer::exl3::Exl3TextContext::transaction_owner_metadata_bytes():0),
                    "Engine omitted repair checkpoint owner metadata");
                need(startup.continuation_linear_owner_metadata_bytes==
                    lanes*ninfer::exl3::Exl3CudaLinearWorkspace::metadata_bytes(),
                    "Engine omitted continuation linear lifetime metadata subset");
                need(startup.continuation_generic_owner_metadata_bytes==
                    2*lanes*ninfer::exl3::Exl3TextContext::allocation_record_metadata_bytes(),
                    "Engine omitted continuation normalized-row/logit lifetime metadata");
                need(startup.context_private_generic_owner_metadata_bytes>0 &&
                    startup.context_private_generic_owner_metadata_bytes%
                        ninfer::exl3::Exl3TextContext::allocation_record_metadata_bytes()==0 &&
                    startup.context_private_generic_owner_metadata_bytes<startup.context_allocation_owner_metadata_bytes,
                    "Engine private generic lifetime metadata missing or exceeds context total");
                need(startup.context_shared_generic_owner_metadata_bytes%
                        ninfer::exl3::Exl3TextContext::allocation_record_metadata_bytes()==0 &&
                    startup.context_shared_generic_owner_metadata_bytes+startup.context_private_generic_owner_metadata_bytes<
                        startup.context_allocation_owner_metadata_bytes,
                    "Engine shared KV lifetime subset overlaps private or parent allocation metadata");
                need(startup.context_base_linear_owner_metadata_bytes>=
                    lanes*ninfer::exl3::Exl3CudaLinearWorkspace::metadata_bytes() &&
                    startup.context_base_linear_owner_metadata_bytes<=
                    2*lanes*ninfer::exl3::Exl3CudaLinearWorkspace::metadata_bytes(),
                    "Engine base linear metadata must include head and optional auxiliary workspace");
                need(startup.context_layer_linear_owner_metadata_bytes>0 &&
                    startup.context_layer_linear_owner_metadata_bytes%ninfer::exl3::Exl3CudaLinearWorkspace::metadata_bytes()==0 &&
                    startup.context_layer_linear_owner_metadata_bytes+startup.context_base_linear_owner_metadata_bytes<
                        startup.context_allocation_owner_metadata_bytes,
                    "Engine layer lifetime subset missing or overlaps parent metadata");
                need(startup.reserved_lane_startup_device_bytes==lanes*5ULL*16*5120*2,
                    "Engine lane credit includes context or omits staging");
                need(startup.reserved_lane_fixed_owner_metadata_bytes>0,
                    "Engine omitted committed lane object metadata");
                need(startup.reserved_engine_tap_device_bytes==lanes*5ULL*16*5120*2,
                    "Engine omitted or duplicated committed external tap storage");
                need(startup.reserved_engine_tap_metadata_bytes==lanes*5*sizeof(std::uint16_t*),
                    "Engine external tap owner metadata extent mismatch");
                if(enabled("NINFER_EXL3_ENGINE_SHARED_DEVICE_PREFIX") && lanes==2) {
                    need(startup.private_device_prefix_bytes==0 &&
                        (startup.shared_device_prefix_bytes==0 || startup.shared_device_prefix_bytes==4096ULL*1024*2*32),
                        "reserved shared prefix allocated private or duplicated cache storage");
                } else if(device_prefix && !enabled("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES") &&
                    !enabled("NINFER_EXL3_ENGINE_ATTENTION_STAGING")) {
                    const auto prefix_extent=(prefix16k?16384ULL:4096ULL)*1024*2*32;
                    need(startup.private_device_prefix_bytes<=lanes*prefix_extent &&
                        startup.private_device_prefix_bytes%prefix_extent==0,
                        "reserved optional prefix reported partial or duplicated physical storage");
                } else need(startup.private_device_prefix_bytes==0,
                    "prefix-excluded reserved startup allocated private prefix storage");
            }
            const auto stream_metadata=owner.runtime_stats().execution_stream_metadata_bytes;
            const auto model_metadata=owner.runtime_stats().target_allocation_owner_metadata_bytes;
            const auto context_metadata=owner.runtime_stats().context_allocation_owner_metadata_bytes;
            if(lanes==1){need(context_metadata>0,"Engine omitted private context allocation records");context_records=context_metadata;}
            else need(context_metadata==2*context_records,"Engine private context metadata did not scale by lane");
            const auto draft_metadata=owner.runtime_stats().draft_weight_owner_metadata_bytes;
            const auto private_draft_metadata=owner.runtime_stats().draft_execution_owner_metadata_bytes;
            need(owner.runtime_stats().draft_linear_owner_metadata_bytes==
                lanes*7*ninfer::exl3::Exl3CudaLinearWorkspace::metadata_bytes(),
                "draft linear lifetime metadata did not match physical lane count");
            if(lanes==1){need(private_draft_metadata>7*ninfer::exl3::Exl3CudaLinearWorkspace::metadata_bytes(),
                "Engine omitted private draft workspace objects or scratch allocation records");private_draft_records=private_draft_metadata;}
            else need(private_draft_metadata==2*private_draft_records,"private draft record metadata did not scale by clone");
            if(lanes==1){need(draft_metadata>0,"Engine omitted draft weight owner records");draft_records=draft_metadata;}
            else need(draft_metadata==draft_records,"shared draft metadata counted per execution clone");
            if(lanes==1){need(model_metadata>0,"Engine omitted target allocation owner records");target_records=model_metadata;}
            else need(model_metadata==target_records,"shared target allocation metadata scaled with private lanes");
            if(lanes==1){need(stream_metadata>0,"Engine omitted default-stream wrapper ownership");per_stream=stream_metadata;}
            else need(stream_metadata==2*per_stream,"Engine omitted private stream wrapper metadata");
            if(lanes==1){need(metadata>0,"Engine omitted retirement allocation metadata");per_lane=metadata;}
            else need(metadata==2*per_lane,"Engine did not account both private retirement owners");
            need(owner.close().reusable(),"retirement metadata lifecycle drain failed");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_LINEAR_CONSTRUCTOR_CREDITS");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string selector(mode);
        need(selector=="1" || selector=="base" || selector=="attention" || selector=="gdn","Engine linear constructor selector");
        const bool base=selector=="base";
        const bool layer=selector=="attention" || selector=="gdn";
        need(opt.max_concurrency==1 && Exl3CudaLinearWorkspace::quarantined_workspaces()==0,
            "Engine linear constructor fixture requires fresh C1 process");
        bool original=false;
        try{Exl3EngineCore owner(opt,0,0,0,selector=="attention"?91:(selector=="gdn"?92:(base?90:89)));}
        catch(const std::exception& error){original=std::string(error.what()).find("injected EXL3 accumulation allocation failure")!=std::string::npos;}
        need(original,"Engine linear construction replaced injected allocation failure");
        const auto retained=Exl3CudaLinearWorkspace::latest_storage_retirement_for_test();
        const auto credits=Exl3CudaLinearWorkspace::latest_storage_credits_for_test();
        const auto plan=Exl3LinearWorkspaceRequirements::derive(5120,248320,base?1:8);
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==1 && retained.transform_retained &&
            !retained.accumulation_retained && retained.transformed_bytes>0 &&
            (layer || retained.transformed_bytes==plan.transformed_bytes) &&
            credits[0]==Exl3CudaLinearWorkspace::storage_retirement_metadata_bytes() && credits[1]==retained.transformed_bytes,
            "actual continuation constructor lost exact transform/record tickets");
        need(Exl3TextContext::generic_quarantined_allocations()==0,
            "linear constructor failure prevented generic sibling cleanup");
        const auto ceiling=Exl3HostResidentSet::failed_close_reservation_ceiling_for_test();
        need(ceiling && (*ceiling)[0]>=credits[1],"linear constructor failure omitted sealed reservation ceiling");
        std::cout<<"ENGINE_LINEAR_CONSTRUCTOR_CREDITS_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_LATE_STARTUP_RETIREMENT");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string selector(mode);
        need(selector=="context" || selector=="continuation" || selector=="private","late startup retirement selector");
        need(opt.max_concurrency==1 && Exl3TextContext::generic_quarantined_allocations()==0 &&
            !Exl3HostResidentSet::failed_close_retained_for_test(),"late startup retirement requires fresh C1 process");
        bool original_failure=false;
        try{Exl3EngineCore owner(opt,0,0,0,selector=="private"?88:(selector=="context"?86:87));}
        catch(const std::exception& error){original_failure=std::string(error.what())=="resource reservation actual extent/domain mismatch";}
        need(original_failure,"late cleanup replaced original allocation extent failure");
        need(Exl3TextContext::generic_quarantined_allocations()==1,"late startup inventory release omitted failed allocation");
        const auto retained=Exl3TextContext::generic_retirement_snapshot_for_test();
        need(retained.pointer && retained.bytes>0 && retained.error==static_cast<int>(cudaErrorUnknown),
            "late startup cleanup lost allocation/error identity");
        const auto ceiling=Exl3HostResidentSet::failed_close_reservation_ceiling_for_test();
        need(ceiling && (*ceiling)[0]>=retained.bytes &&
            (*ceiling)[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)]>=retained.metadata_bytes,
            "post-inventory observer failed to retain sealed authority and reservation ceiling");
        need(retained.device_credit_bytes==retained.bytes &&
            retained.metadata_credit_bytes==retained.metadata_bytes,
            "actual continuation startup failure lost preallocation device/record tickets");
        std::cout<<"ENGINE_LATE_STARTUP_RETIREMENT_COMPLETE generic_constructor_credits=covered\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_CONTEXT_STARTUP");mode && std::string_view(mode)=="1") {
        const std::array<const char*,11> errors{
            "injected context startup preconstruction failure",
            "injected context startup precommit failure",
            "resource reservation actual extent/domain mismatch",
            "resource reservation actual extent/domain mismatch",
            "injected continuation preconstruction failure",
            "injected continuation precommit failure",
            "resource reservation actual extent/domain mismatch",
            "injected lane startup preconstruction failure",
            "injected lane startup precommit failure",
            "resource reservation actual extent/domain mismatch",
            "resource reservation actual extent/domain mismatch"};
        for(unsigned lanes:{1u,2u}) {
            opt.max_concurrency=lanes;
            for(unsigned fault=1;fault<=11*lanes;++fault) {
                const auto quarantine_before=ninfer::exl3::Exl3TextContext::retirement_quarantine_witness();
                const auto continuation_blocks=ninfer::exl3::Exl3TextContext::continuation_owner_blocks_for_test();
                bool injected=false;
                try{ninfer::exl3::Exl3EngineCore owner(opt,0,0,0,fault);}
                catch(const std::exception& error){injected=std::string(error.what())==errors[(fault-1)%11];}
                need(injected,"Engine context/continuation/lane startup fault NOT_EXERCISED");
                need(ninfer::exl3::Exl3TextContext::continuation_owner_blocks_for_test()==continuation_blocks,
                    "Engine startup rollback retained continuation control allocation");
                need(ninfer::exl3::Exl3TextContext::retirement_quarantine_witness()==quarantine_before,
                    "normally retired startup failure changed post-inventory quarantine witness");
                // A lane1 failure follows lane0's committed transactions. Retry
                // must reconstruct both lanes without retaining occupied state.
                ninfer::exl3::Exl3EngineCore retry(opt);
                const auto counters=retry.runtime_stats();
                need(ninfer::exl3::Exl3TextContext::continuation_owner_blocks_for_test()==continuation_blocks+lanes,
                    "Engine retry omitted bounded continuation control owners");
                need(counters.reserved_context_startup_lanes==lanes && counters.running_requests==0,
                    "Engine startup retry omitted reserved lanes or started requests");
                need(counters.reserved_lane_startup_device_bytes==lanes*5ULL*16*5120*2,
                    "Engine startup retry double-counted lane storage");
                need(retry.close().reusable(),"Engine startup fault prevented reusable close");
                need(ninfer::exl3::Exl3TextContext::continuation_owner_blocks_for_test()==continuation_blocks,
                    "Engine reusable close retained continuation control allocation");
            }
            for(unsigned invalid_case:{0u,1u,2u,3u}) {
                bool refused=false;
                try {
                    ninfer::exl3::Exl3EngineCore invalid(opt,
                        invalid_case==1?1:0,invalid_case==2?1:0,invalid_case==3?1:0,
                        invalid_case==0?29:1);
                } catch(const std::invalid_argument& error) {
                    refused=std::string(error.what())==
                        "unsupported context startup fault stage or combination";
                }
                need(refused,"Engine accepted out-of-range or mixed startup faults");
            }
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_TAP_STARTUP");mode && std::string_view(mode)=="1") {
        for(unsigned lanes:{1u,2u}) {
            opt.max_concurrency=lanes;
            std::vector<unsigned> faults;
            for(unsigned fault=41;fault<41+5*lanes;++fault)faults.push_back(fault);
            faults.insert(faults.end(),{51u,52u,53u});
            for(unsigned fault:faults) {
                bool injected=false;
                try{ninfer::exl3::Exl3EngineCore failed(opt,0,0,0,fault);}
                catch(const std::exception& error){injected=std::string(error.what())==
                    (fault<=50?"injected Engine tap staging allocation failure":
                     fault==51?"injected Engine tap staging precommit failure":
                     "resource reservation actual extent/domain mismatch");}
                need(injected,"Engine tap allocation fault NOT_EXERCISED");
                ninfer::exl3::Exl3EngineCore retry(opt);
                need(retry.runtime_stats().reserved_context_startup_lanes==lanes &&
                    retry.runtime_stats().running_requests==0,"tap rollback prevented idle complete Engine retry");
                need(retry.close().reusable(),"tap allocation retry failed retirement");
            }
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_STREAM_UNWIND");mode && *mode) {
        using Core=ninfer::exl3::Exl3EngineCore;
        const std::string_view selected(mode);
        const bool after_return=selected=="extent0" || selected=="extent1";
        need(selected=="0" || selected=="1" || after_return,
            "stream unwind mode must select 0, 1, extent0 or extent1");
        const unsigned lane=(selected=="1" || selected=="extent1")?1:0;
        opt.max_concurrency=2;
        bool original=false;
        try {Core failed(opt,0,0,0,(after_return?81:79)+lane);}
        catch(const std::invalid_argument& error) {original=after_return && std::string_view(error.what())==
            "resource reservation actual extent/domain mismatch";}
        catch(const std::runtime_error& error) {original=std::string_view(error.what())==
            "injected Engine stream unwind failure" && !after_return;}
        need(original && Core::execution_stream_retirement_error_for_test()==static_cast<int>(cudaErrorUnknown) &&
            Core::execution_stream_retirement_slots_for_test()>=1,
            "stream constructor unwind lost primary error or failed handle capacity");
        need(ninfer::exl3::Exl3HostResidentSet::retirement_quarantined(),
            "failed stream rollback released residency authority");
        need(Core::execution_stream_quarantined_handles_for_test()==1,
            "stream rollback lost or duplicated failed handle");
        if(after_return)need(Core::execution_stream_owner_blocks_for_test()>=1,
            "post-return rollback lost retained inventory backing");
        for(unsigned attempt=0;attempt<2;++attempt) {
            bool refused=false;try {
                if(attempt==0) {Core replacement(EngineOptions{});}
                else {Engine replacement(opt);}
            }
            catch(const std::runtime_error& error) {refused=std::string_view(error.what())==
                "EXL3 unresolved execution stream retirement; reload refused";}
            need(refused,"stream constructor unwind permitted reload");
        }
        std::cout << "ENGINE_STREAM_UNWIND_COMPLETE full_engine_coverage=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_STREAM_CAPACITY");mode && std::string_view(mode)=="1") {
        using Core=ninfer::exl3::Exl3EngineCore;
        opt.max_concurrency=2;
        {
            Core owner(opt);const auto closed=owner.close();
            need(closed.reusable() && closed.pending_stream_destructions==2 &&
                owner.close().pending_stream_destructions==2,
                "stream capacity close omitted pending final-owner destruction");
            need(Core::execution_stream_retirement_slots_for_test()==2,
                "live closed Engine released stream retirement slots prematurely");
            for(unsigned attempt=0;attempt<2;++attempt) {
                bool refused=false;try {
                    if(attempt==0) {Core replacement(opt);}
                    else {Engine replacement(opt);}
                }
                catch(const std::runtime_error& error) {refused=std::string_view(error.what())==
                    "execution stream retirement capacity exhausted";}
                need(refused && Core::execution_stream_retirement_slots_for_test()==2 &&
                    Core::execution_stream_retirement_error_for_test()==0,
                    "stream capacity refusal changed live slots or fabricated failure");
            }
        }
        need(Core::execution_stream_retirement_slots_for_test()==0,"successful stream destruction leaked slots");
        Core retry(opt);need(retry.close().reusable(),"stream capacity did not recover after final owner release");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_STREAM_RETIREMENT");mode && std::string_view(mode)=="1") {
        using Core=ninfer::exl3::Exl3EngineCore;
        opt.max_concurrency=2;
        {
            Core owner(opt);
            owner.fail_execution_stream_destroy_for_test(0,static_cast<int>(cudaErrorUnknown));
            owner.fail_execution_stream_destroy_for_test(1,static_cast<int>(cudaErrorUnknown));
            // close certifies the drain; last-owner stream destruction happens
            // after this scope. It must not erase a later destroy failure.
            const auto closed=owner.close();
            need(closed.reusable() && closed.pending_stream_destructions==2,
                "stream destroy fixture failed drain or hid pending destruction");
            need(Core::execution_stream_retirement_error_for_test()==0,
                "stream destroy failure reported before destruction");
        }
        need(Core::execution_stream_retirement_error_for_test()==static_cast<int>(cudaErrorUnknown),
            "stream destroy failure was discarded");
        need(Core::execution_stream_retirement_slots_for_test()==2 && Core::execution_stream_quarantined_handles_for_test()==2,
            "failed stream destruction recycled capacity or lost a handle");
        for(unsigned attempt=0;attempt<2;++attempt) {
            bool refused=false;try {
                if(attempt==0) {Core replacement(EngineOptions{});}
                else {Engine replacement(opt);}
            }
            catch(const std::runtime_error& error) {refused=std::string_view(error.what())==
                "EXL3 unresolved execution stream retirement; reload refused";}
            need(refused,"stream destroy failure permitted Engine reload");
        }
        std::cout << "ENGINE_STREAM_RETIREMENT_COMPLETE reload_refused=2 full_engine_coverage=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_STREAM_STARTUP");mode && std::string_view(mode)=="1") {
        const std::array<const char*,3> expected{
            "injected Engine stream preconstruction failure",
            "injected Engine stream precommit failure",
            "resource reservation actual extent/domain mismatch"};
        for(unsigned lanes:{1u,2u}) {
            opt.max_concurrency=lanes;
            for(unsigned fault=23;fault<23+3*lanes;++fault) {
                const auto blocks=ninfer::exl3::Exl3EngineCore::execution_stream_owner_blocks_for_test();
                const auto slots=ninfer::exl3::Exl3EngineCore::execution_stream_retirement_slots_for_test();
                bool injected=false;
                try{ninfer::exl3::Exl3EngineCore failed(opt,0,0,0,fault);}
                catch(const std::exception& error){injected=std::string(error.what())==expected[(fault-23)%3];}
                need(injected,"Engine stream startup fault NOT_EXERCISED");
                need(ninfer::exl3::Exl3EngineCore::execution_stream_owner_blocks_for_test()==blocks,
                    "stream startup refusal retained bounded backing");
                need(ninfer::exl3::Exl3EngineCore::execution_stream_retirement_slots_for_test()==slots,
                    "stream startup refusal retained retirement capacity");
                ninfer::exl3::Exl3EngineCore retry(opt);
                need(retry.runtime_stats().reserved_context_startup_lanes==lanes,
                    "stream rollback prevented complete lane startup");
                need(retry.runtime_stats().execution_stream_metadata_bytes==
                    lanes*ninfer::exl3::Exl3EngineCore::execution_stream_owner_bytes_for_test(),
                    "stream retry omitted committed metadata");
                const auto closed=retry.close();
                need(closed.reusable() && closed.pending_stream_destructions==(lanes==2?2u:0u),
                    "stream startup close failed drain or misreported nondefault handles");
            }
            bool credit_refused=false;
            try{ninfer::exl3::Exl3EngineCore failed(opt,0,0,0,31);}
            catch(const std::exception& error){credit_refused=std::string(error.what())==
                "resource allocation reservation exhausted";}
            need(credit_refused,"stream metadata credit did not refuse before allocation");
            ninfer::exl3::Exl3EngineCore retry(opt);
            need(retry.runtime_stats().reserved_context_startup_lanes==lanes,
                "stream credit refusal prevented fresh complete startup");
            need(retry.close().reusable(),"stream credit refusal retry failed retirement");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_DRAFT_GENERIC_CREDITS");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string selector(mode);
        need(selector=="first" || selector=="clone","draft generic credit selector");
        const bool clone=selector=="clone";opt.max_concurrency=clone?2:1;
        need(Exl3Dflash2DraftModel::generic_quarantined_allocations()==0 &&
            Exl3CudaLinearWorkspace::quarantined_workspaces()==0,"draft generic fixture requires fresh process");
        bool original=false;
        try{Exl3EngineCore owner(opt,0,clone?8:0,clone?0:8);}
        catch(const std::exception& error){original=std::string(error.what()).find("resource reservation actual extent/domain mismatch")!=std::string::npos;}
        need(original,"draft late cleanup replaced original inventory failure");
        const auto retained=Exl3Dflash2DraftModel::latest_generic_retirement_for_test();
        need(Exl3Dflash2DraftModel::generic_quarantined_allocations()==1 && retained.pointer_retained &&
            retained.device>=0 && retained.error==static_cast<int>(cudaErrorUnknown) && retained.bytes>0 &&
            retained.device_credit==retained.bytes && retained.metadata_credit==retained.record_bytes &&
            retained.record_bytes>0,"draft late cleanup lost exact allocation/record credits");
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==0,"draft generic failure quarantined unrelated linear workspace");
        const auto ceiling=Exl3HostResidentSet::failed_close_reservation_ceiling_for_test();
        need(ceiling && (*ceiling)[0]>=retained.bytes,"draft late cleanup lost sealed reservation ceiling");
        bool refused=false;
        try{Exl3EngineCore retry(opt);}
        catch(const std::exception& error){refused=std::string(error.what()).find("unresolved draft allocation retirement")!=std::string::npos;}
        need(refused && Exl3Dflash2DraftModel::generic_quarantined_allocations()==1,
            "draft late cleanup allowed reload or retried failed storage");
        const auto after=Exl3Dflash2DraftModel::latest_generic_retirement_for_test();
        need(after.device_credit==retained.device_credit && after.metadata_credit==retained.metadata_credit,
            "draft refused reload changed retained credits");
        std::cout<<"ENGINE_DRAFT_GENERIC_CREDITS_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_DRAFT_CONSTRUCTOR_CREDITS");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string selector(mode);
        need(selector=="first" || selector=="clone","draft constructor credit selector");
        const bool clone=selector=="clone";opt.max_concurrency=clone?2:1;
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==0 &&
            Exl3Dflash2DraftModel::generic_quarantined_allocations()==0,"draft constructor fixture requires fresh process");
        bool original=false;
        try{Exl3EngineCore owner(opt,0,clone?7:0,clone?0:7);}
        catch(const std::exception& error){original=std::string(error.what()).find("injected EXL3 accumulation allocation failure")!=std::string::npos;}
        need(original,"draft materialization replaced original constructor failure");
        const auto retained=Exl3CudaLinearWorkspace::latest_storage_retirement_for_test();
        const auto credits=Exl3CudaLinearWorkspace::latest_storage_credits_for_test();
        const auto plan=Exl3LinearWorkspaceRequirements::derive(5*5120,5120,16);
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==1 && retained.transform_retained &&
            !retained.accumulation_retained && retained.transformed_bytes==plan.transformed_bytes &&
            credits[1]==plan.transformed_bytes && credits[0]==Exl3CudaLinearWorkspace::storage_retirement_metadata_bytes(),
            "actual draft FC constructor lost exact transform/record credits");
        need(Exl3Dflash2DraftModel::generic_quarantined_allocations()==0,"draft FC failure unexpectedly quarantined generic storage");
        const auto ceiling=Exl3HostResidentSet::failed_close_reservation_ceiling_for_test();
        need(ceiling && (*ceiling)[0]>=credits[1],"draft constructor failure failed to preserve sealed reservation");
        std::cout<<"ENGINE_DRAFT_CONSTRUCTOR_CREDITS_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_DRAFT_CLONE_ALLOCATION");mode && std::string_view(mode)=="1") {
        opt.max_concurrency=2;
        for(unsigned stage:{1u,2u,3u,4u,5u,6u}) {
            bool injected=false;
            try{ninfer::exl3::Exl3EngineCore owner(opt,0,stage);}
            catch(const std::exception& error){injected=std::string(error.what())==
                (stage==1?"injected Engine draft clone preconstruction failure":
                 stage==2?"injected Engine draft clone precommit failure":
                 stage<=4?"resource reservation actual extent/domain mismatch":
                 "resource allocation reservation exhausted");}
            need(injected,"Engine reserved draft clone fault NOT_EXERCISED");
            need(ninfer::exl3::Exl3Dflash2DraftModel::generic_quarantined_allocations()==0 &&
                ninfer::exl3::Exl3CudaLinearWorkspace::quarantined_workspaces()==0,
                "ordinary draft clone constructor rollback retained failed allocations");
            ninfer::exl3::Exl3EngineCore retry(opt);
            const auto counters=retry.runtime_stats();
            need(counters.draft_execution_owner_metadata_bytes>0 && counters.running_requests==0,
                "draft clone retry omitted owners or started request work");
            need(retry.close().reusable(),"draft clone allocation rollback prevented reusable Engine startup");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_FIRST_DRAFT_ALLOCATION");mode && std::string_view(mode)=="1") {
        for(unsigned concurrency:{1u,2u})for(unsigned stage:{1u,2u,3u,4u,5u,6u}) {
            opt.max_concurrency=concurrency;
            bool injected=false;
            try{ninfer::exl3::Exl3EngineCore owner(opt,0,0,stage);}
            catch(const std::exception& error){injected=std::string(error.what())==
                (stage==1?"injected Engine first draft preconstruction failure":
                 stage==2?"injected Engine first draft precommit failure":
                 stage<=4?"resource reservation actual extent/domain mismatch":
                 "resource allocation reservation exhausted");}
            need(injected,"Engine reserved first draft fault NOT_EXERCISED");
            need(ninfer::exl3::Exl3Dflash2DraftModel::generic_quarantined_allocations()==0 &&
                ninfer::exl3::Exl3CudaLinearWorkspace::quarantined_workspaces()==0,
                "ordinary first draft constructor rollback retained failed allocations");
            ninfer::exl3::Exl3EngineCore retry(opt);
            const auto counters=retry.runtime_stats();
            need(counters.draft_execution_owner_metadata_bytes>0 &&
                counters.running_requests==0 && counters.waiting_requests==0,
                "first draft retry omitted owners or published request work");
            need(retry.close().reusable(),"first draft rollback prevented reusable startup");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_CONSTRUCTOR_CREDITS");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string selector(mode);
        need(selector=="retain" || selector=="release","shared constructor credit selector");
        unsigned family=1;
        if(const auto* selected=std::getenv("NINFER_TEST_ENGINE_SHARED_CONSTRUCTOR_FAMILY");selected && *selected) {
            const std::string value(selected);std::size_t parsed=0;
            const auto number=std::stoul(value,&parsed);
            need(parsed==value.size() && number>=1 && number<=11,"shared constructor family selector");
            family=static_cast<unsigned>(number);
        }
        // Physical families, independent of which optional predecessors are enabled.
        const std::array<std::pair<int,int>,11> shapes{{{5120,12288},{5120,17408},{5120,4096},
            {5120,1024},{4096,5120},{6144,5120},{17408,5120},{17408,5120},
            {5120,17408},{5120,248320},{5120,1024}}};
        const auto* enabled=std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_Q");
        need(enabled && std::string_view(enabled)=="1","shared constructor fixture requires real Engine shared Q route");
        opt.max_concurrency=2;
        const auto before=Exl3TextContext::retirement_quarantine_witness();
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==0 && !Exl3EngineTargetQ::packed_retirement_unresolved(),
            "shared constructor credit fixture requires fresh process");
        bool original=false;
        const unsigned fault=family==1?(selector=="retain"?101:102):103+2*(family-2)+(selector=="release"?1:0);
        try{Exl3EngineCore failed(opt,0,0,0,fault);}
        catch(const std::exception& error){original=std::string(error.what()).find("injected EXL3 accumulation allocation failure")!=std::string::npos;}
        need(original,"shared Q constructor replaced original accumulation failure");
        need(!Exl3EngineTargetQ::packed_retirement_unresolved(),"shared Q constructor unexpectedly retained packed buffers");
        if(selector=="release") {
            need(Exl3TextContext::retirement_quarantine_witness()==before,"shared constructor successful cleanup quarantined storage");
            need(!Exl3HostResidentSet::failed_close_reservation_ceiling_for_test(),"shared constructor release sealed residency");
            {Exl3EngineCore retry(opt);need(retry.close().reusable(),"shared constructor cleanup prevented Engine retry");}
            need(Exl3TextContext::retirement_quarantine_witness()==before,"shared retry destruction changed quarantine");
        } else {
            const auto retained=Exl3CudaLinearWorkspace::latest_storage_retirement_for_test();
            const auto credits=Exl3CudaLinearWorkspace::latest_storage_credits_for_test();
            const auto plan=Exl3LinearWorkspaceRequirements::derive(shapes[family-1].first,shapes[family-1].second,16);
            need(Exl3CudaLinearWorkspace::quarantined_workspaces()==1 && retained.transform_retained &&
                !retained.accumulation_retained && retained.transformed_bytes==plan.transformed_bytes &&
                credits[1]==plan.transformed_bytes && credits[0]==Exl3CudaLinearWorkspace::storage_retirement_metadata_bytes(),
                "shared Q constructor did not retain exact transform/record credits");
            const auto ceiling=Exl3HostResidentSet::failed_close_reservation_ceiling_for_test();
            need(ceiling && (*ceiling)[0]>=credits[1],"shared constructor failure lost reservation ceiling");
            bool refused=false;try{Exl3EngineCore retry(opt);}
            catch(const std::exception& error){refused=std::string(error.what()).find("unresolved linear workspace retirement")!=std::string::npos;}
            need(refused && Exl3CudaLinearWorkspace::quarantined_workspaces()==1,"shared constructor reload retried failed cleanup");
        }
        std::cout<<"ENGINE_SHARED_CONSTRUCTOR_CREDITS_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_UNWIND");mode &&
        (std::string_view(mode)=="1" || std::string_view(mode)=="extent" || std::string_view(mode)=="packed_extent")) {
        using Core=ninfer::exl3::Exl3EngineCore;
        const bool packed=std::string_view(mode)=="packed_extent";
        const bool after_return=std::string_view(mode)=="extent" || packed;
        opt.max_concurrency=2;
        const auto before=ninfer::exl3::Exl3CudaLinearWorkspace::quarantined_workspaces();
        bool original=false;
        try {Core failed(opt,after_return?0:2,0,0,packed?85:after_return?84:83);}
        catch(const std::invalid_argument& error) {original=after_return && std::string_view(error.what())==
            "resource reservation actual extent/domain mismatch";}
        catch(const std::runtime_error& error) {original=std::string_view(error.what())==
            "injected shared projection startup allocation failure" && !after_return;}
        need(original && ninfer::exl3::Exl3CudaLinearWorkspace::quarantined_workspaces()==before+(packed?0:1),
            "Engine shared unwind lost primary error or failed workspace");
        if(packed) {
            const auto retained=ninfer::exl3::Exl3EngineTargetQ::packed_retirement_snapshot_for_test();
            need(ninfer::exl3::Exl3EngineTargetQ::packed_retirement_unresolved() &&
                retained.input_bytes>0 && retained.output_bytes>0 && retained.device>=0 &&
                retained.error==static_cast<int>(cudaErrorUnknown) && retained.restore_error==0,
                "Engine final inventory release lost packed storage cleanup failure");
            need(retained.input_credit==retained.input_bytes && retained.output_credit==retained.output_bytes &&
                retained.metadata_credit==ninfer::exl3::Exl3EngineTargetQ::packed_retirement_metadata_bytes(),
                "Engine packed startup lost exact constructor credits after inventory unwind");
        } else {
            const auto retained=ninfer::exl3::Exl3CudaLinearWorkspace::latest_retirement_for_test();
            const auto credits=ninfer::exl3::Exl3CudaLinearWorkspace::latest_retirement_credits_for_test();
            const auto plan=ninfer::exl3::Exl3LinearWorkspaceRequirements::derive(5120,12288,16);
            need(retained.transformed_bytes==plan.transformed_bytes && retained.accumulation_bytes==plan.accumulation_bytes &&
                credits[1]==plan.owned_bytes && credits[0]==ninfer::exl3::Exl3CudaLinearWorkspace::metadata_bytes(),
                "Engine shared Q unwind lost exact constructor device/object/record credits");
        }
        need(ninfer::exl3::Exl3HostResidentSet::retirement_quarantined(),
            "Engine shared cleanup failure advertised clean residency close");
        bool refused=false;try {Engine replacement(opt);}
        catch(const std::runtime_error& error) {refused=std::string_view(error.what())==(packed?
            "EXL3 unresolved packed projection retirement; reload refused":
            "EXL3 unresolved linear workspace retirement; reload refused");}
        need(refused,"public Engine bypassed shared startup cleanup failure");
        std::cout << "ENGINE_SHARED_UNWIND_COMPLETE full_engine_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_ALLOCATION");mode && std::string(mode)=="1") {
        opt.max_concurrency=2;
        const auto enabled=[](const char* name){const auto* value=std::getenv(name);return value && std::string(value)=="1";};
        const auto stages=4U+unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_TARGET_KV"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_TARGET_O"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_DRAFT_Q_M16"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_TARGET_GATEUP"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_TARGET_DOWN"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_HEAD"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_DRAFT_KV_M16"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_DRAFT_O_M16"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_DRAFT_DOWN_M16"))+
            unsigned(enabled("NINFER_EXL3_ENGINE_SHARED_DRAFT_GATEUP_M16"));
        bool outside_menu=false;
        try{ninfer::exl3::Exl3EngineCore owner(opt,stages+1);}
        catch(const std::invalid_argument& error){
            outside_menu=std::string(error.what())=="shared allocation fault outside selected startup menu";
        }
        need(outside_menu,"shared allocation fault beyond final cleanup stage was admitted");
        for(unsigned stage=1;stage<=stages;++stage) {
            const auto blocks=ninfer::exl3::Exl3EngineCore::shared_projection_owner_blocks_for_test();
            bool injected=false;
            try{ninfer::exl3::Exl3EngineCore owner(opt,stage);}
            catch(const std::runtime_error& e){injected=std::string(e.what()).find("injected shared projection startup allocation failure")!=std::string::npos;}
            need(injected,"selected shared allocation fault was not reached");
            need(ninfer::exl3::Exl3EngineCore::shared_projection_owner_blocks_for_test()==blocks,
                "shared startup failure leaked bounded rendezvous backing");
            // Constructor never starts workers or publishes a partial Engine.
            // Reacquiring the actual process authority detects leaked residency.
            ninfer::exl3::Exl3HostResidentSet registry(1ULL<<30,8ULL<<30);registry.close();
        }
        {
            ninfer::exl3::Exl3EngineCore retry(opt);
            const auto expected_metadata=ninfer::exl3::bounded_shared_allocation_bytes<ninfer::exl3::Exl3EngineTargetQ>()+
                ninfer::exl3::Exl3EngineTargetQ::packed_retirement_metadata_bytes()+
                (stages-3)*ninfer::exl3::Exl3CudaLinearWorkspace::metadata_bytes();
            need(retry.shared_projection_metadata_bytes_for_test()==expected_metadata,
                "selected shared family omitted workspace host objects from metadata");
            need(retry.close().reusable(),"startup rollback prevented clean retry");
            need(retry.shared_projection_retirement_credit_for_test()==expected_metadata,
                "shared workspace object credit disappeared before final owner release");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_NONBLOCKING_COMPLETION");mode && std::string(mode)=="1") {
        opt.max_concurrency=2;opt.max_pending_requests=4;
        struct Gate {std::mutex mutex;std::condition_variable changed;unsigned starts=0;bool release=false;};
        auto gate=std::make_shared<Gate>();
        ninfer::exl3::Exl3EngineCore owner(opt);
        owner.observe_request_start_for_test([gate] {
            std::unique_lock lock(gate->mutex);
            if(++gate->starts==1) {
                gate->changed.notify_all();
                gate->changed.wait(lock,[&]{return gate->release;});
            }
        });
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;request.execution.requested_output_tokens=8;
        request.execution.allow_prefix_reuse=false;request.stop=owner.frontend().default_stop_policy();
        const auto submit=[&](OutputConsumerMode consumer) {
            auto prepared=owner.frontend().prepare(prompt("Explain bridge foundations."));
            const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,consumer,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        auto held=submit(OutputConsumerMode::Aggregate);
        {
            std::unique_lock lock(gate->mutex);
            need(gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->starts==1;}),
                "nonblocking fixture did not hold first worker");
        }
        auto peer=submit(OutputConsumerMode::Aggregate);
        need(held.poll(nullptr,{}).state==GenerationPollState::Pending,
            "held request did not expose pending completion");
        GenerationPollResult peer_poll;
        const auto peer_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        do {
            peer_poll=peer.poll(nullptr,{});
            if(peer_poll.state==GenerationPollState::Pending)std::this_thread::yield();
        } while(peer_poll.state==GenerationPollState::Pending &&
            std::chrono::steady_clock::now()<peer_deadline);
        need(peer_poll.state==GenerationPollState::Completed && peer_poll.result &&
                !peer_poll.result->generated_token_ids.empty(),
            "unrelated ready request did not complete while peer remained pending");
        {
            std::lock_guard lock(gate->mutex);gate->release=true;gate->changed.notify_all();
        }
        GenerationPollResult held_poll;
        const auto held_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        do {
            held_poll=held.poll(nullptr,{});
            if(held_poll.state==GenerationPollState::Pending)std::this_thread::yield();
        } while(held_poll.state==GenerationPollState::Pending &&
            std::chrono::steady_clock::now()<held_deadline);
        need(held_poll.state==GenerationPollState::Completed && held_poll.result,
            "released pending request did not complete");
        owner.set_host_kv_routes_for_test(false,false);
        owner.observe_request_start_for_test({});

        struct ReentrantSink final:OutputSink {
            ninfer::exl3::Exl3EngineCore& owner;unsigned calls=0;
            explicit ReentrantSink(ninfer::exl3::Exl3EngineCore& value):owner(value){}
            void publish(OutputDelta) override {(void)owner.runtime_stats();++calls;}
        } sink{owner};
        auto streaming=submit(OutputConsumerMode::Streaming);
        GenerationPollResult streaming_poll;
        const auto streaming_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        do {
            streaming_poll=streaming.poll(&sink,{});
            if(streaming_poll.state==GenerationPollState::Pending)std::this_thread::yield();
        } while(streaming_poll.state==GenerationPollState::Pending &&
            std::chrono::steady_clock::now()<streaming_deadline);
        need(streaming_poll.state==GenerationPollState::Completed && sink.calls,
            "poll callback reentrancy deadlocked or lost streaming completion");
        owner.set_host_kv_routes_for_test(false,false);

        auto cancelled=submit(OutputConsumerMode::Aggregate);
        GenerationPollResult cancel_poll=cancelled.poll(nullptr,CancellationView([]{return true;}));
        const auto cancel_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(cancel_poll.state==GenerationPollState::Pending &&
              std::chrono::steady_clock::now()<cancel_deadline) {
            std::this_thread::yield();cancel_poll=cancelled.poll(nullptr,{});
        }
        need(cancel_poll.state==GenerationPollState::Completed && cancel_poll.result &&
                cancel_poll.result->finish_reason==FinishReason::Cancelled,
            "poll cancellation did not preserve terminal completion");
        owner.set_host_kv_routes_for_test(false,false);

        owner.fail_next_window_publication_for_test();
        auto failed=submit(OutputConsumerMode::Streaming);
        ReentrantSink failed_sink{owner};GenerationPollResult failure_poll;
        const auto failure_deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        do {
            failure_poll=failed.poll(&failed_sink,{});
            if(failure_poll.state==GenerationPollState::Pending)std::this_thread::yield();
        } while(failure_poll.state==GenerationPollState::Pending &&
            std::chrono::steady_clock::now()<failure_deadline);
        bool expected_error=false;
        if(failure_poll.state==GenerationPollState::Error && failure_poll.error)try {
            std::rethrow_exception(failure_poll.error);
        } catch(const std::exception&) {expected_error=true;}
        need(expected_error && failed_sink.calls==0,
            "poll error completion was exposed as ready output");
        need(owner.close().reusable(),"nonblocking completion retirement");
        std::cout<<"ENGINE_NONBLOCKING_COMPLETION_COMPLETE\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_STREAM_BATCH_DELIVERY");mode && std::string(mode)=="1") {
        struct Sink:OutputSink {
            std::string content,reasoning;unsigned calls=0;
            std::size_t largest=0;
            std::vector<OutputChannel> transitions;
            void publish(OutputDelta delta) override {
                ++calls;
                largest=std::max(largest,delta.text.size());
                if(transitions.empty() || transitions.back()!=delta.channel)transitions.push_back(delta.channel);
                if(delta.channel==OutputChannel::Reasoning)reasoning+=delta.text;else content+=delta.text;
            }
        };
        std::mutex terminal_mutex;std::condition_variable terminal_changed;bool terminal=false;
        ninfer::exl3::Exl3EngineCore owner(opt);
        for(bool thinking:{false,true}) {
        Sink sink;
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;request.execution.requested_output_tokens=64;
        if(thinking)request.execution.thinking.budget=1;
        request.execution.allow_prefix_reuse=false;request.stop=owner.frontend().default_stop_policy();
        const auto submit=[&](OutputConsumerMode mode) {
            auto prepared=owner.frontend().prepare(prompt("Describe bridge foundations, drainage and inspection in detail.",thinking));
            const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,mode,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        {
            auto resources=ninfer::exl3::load_pinned_frontend_resources(target);
            auto larger=ninfer::targets::qwen3_6::make_frontend(resources,
                {.vision_enabled=false,.max_context=opt.max_context+1});
            auto oversized=larger.prepare_tokens(std::vector<TokenId>(static_cast<std::size_t>(opt.max_context)+1,0));
            auto misleading=oversized.summary();misleading.prompt_tokens=1;
            const auto before=owner.runtime_stats();bool refused=false;
            try{auto bad=owner.submit(std::move(oversized),misleading,0,request,OutputConsumerMode::Aggregate,{});}
            catch(const RequestError& error){refused=error.kind()==RequestErrorKind::ContextLengthExceeded;}
            need(refused && owner.runtime_stats().waiting_requests==before.waiting_requests &&
                owner.runtime_stats().running_requests==before.running_requests,
                "oversized prepared input reached queue or trusted caller summary");
        }
        auto aggregate=submit(OutputConsumerMode::Aggregate);const auto expected=aggregate.wait(nullptr,{});
        need(expected.content.size()+expected.reasoning.size()<=owner.frontend().output_text_byte_bound(64),
            "visible output exceeds artifact-derived allowance");
        owner.set_host_kv_routes_for_test(false,false);
        if(thinking)need(!expected.reasoning.empty() && !expected.content.empty(),
            "reasoning-to-content stream range fixture NOT_EXERCISED");
        auto streaming=submit(OutputConsumerMode::Streaming);const auto actual=streaming.wait(&sink,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(sink.calls>0,"stream batch delivery NOT_EXERCISED");
        need(actual.generated_token_ids==expected.generated_token_ids && actual.finish_reason==expected.finish_reason &&
            actual.content==expected.content && actual.reasoning==expected.reasoning &&
            sink.content==actual.content && sink.reasoning==actual.reasoning,
            "stream batch ownership transfer dropped, duplicated or reordered visible output");
        owner.observe_terminal_roots_for_test([&](auto) {
            std::lock_guard lock(terminal_mutex);terminal=true;terminal_changed.notify_all();
        });
        auto delayed=submit(OutputConsumerMode::Streaming);
        const auto delayed_record=delayed.result_owner_for_test();
        const auto delayed_reserved_bytes=delayed.result_reserved_bytes_for_test();
        const auto delayed_fixed_bytes=delayed.result_fixed_bytes_for_test();
        const auto delayed_option_bytes=delayed.option_storage_bytes_for_test();
        const auto delayed_prompt_bytes=delayed.prompt_storage_bytes_for_test();
        const auto delayed_session_bytes=delayed.initial_session_bytes_for_test();
        need(delayed_fixed_bytes>sizeof(GenerationResult),"result charge omitted Request backing");
        {
            std::unique_lock lock(terminal_mutex);
            need(terminal_changed.wait_for(lock,std::chrono::seconds(30),[&]{return terminal;}),
                "delayed stream did not reach terminal authority without consumer");
        }
        // Leave the completed streaming result unconsumed while another request
        // uses the Engine. Pending output must not retain a physical worker.
        auto peer=submit(OutputConsumerMode::Aggregate);
        const auto peer_result=peer.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(!delayed_record.expired(),"unconsumed result lost reserved record owner");
        need(delayed.observed_session_bytes_for_test()>0 &&
            delayed.observed_session_bytes_for_test()<=delayed_session_bytes,
            "actual Engine preview/recycle did not stay within reserved private session extent");
        need(peer_result.generated_token_ids==expected.generated_token_ids &&
            peer_result.content==expected.content && peer_result.reasoning==expected.reasoning &&
            peer_result.finish_reason==expected.finish_reason && owner.runtime_stats().running_requests==0,
            "unconsumed stream blocked peer or changed its authoritative output");
        Sink delayed_sink;
        const auto delayed_result=delayed.wait(&delayed_sink,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(delayed_record.expired(),"consumed result retained logical record reservation owner");
        const auto reserved_text_bound=owner.frontend().output_text_byte_bound(64);
        need(delayed_result.reasoning.capacity()>=reserved_text_bound &&
            delayed_result.content.capacity()>=reserved_text_bound,
            "stream result lost preallocated text storage during publication or transfer");
        need(delayed_prompt_bytes>0 && delayed_session_bytes>0 &&
            delayed_reserved_bytes==delayed_fixed_bytes+delayed_option_bytes+delayed_prompt_bytes+delayed_session_bytes+
            delayed_result.generated_token_ids.capacity()*sizeof(decltype(GenerationResult::generated_token_ids)::value_type)+
            delayed_result.reasoning.capacity()+delayed_result.content.capacity()+2,
            "result reservation did not cover actual transferred token capacity");
        need(delayed_result.generated_token_ids==expected.generated_token_ids &&
            delayed_result.finish_reason==expected.finish_reason && delayed_sink.content==expected.content &&
            delayed_sink.reasoning==expected.reasoning && delayed_sink.transitions==sink.transitions && delayed_sink.calls>0 &&
            delayed_sink.calls<expected.generated_token_ids.size(),
            "delayed pending output failed to coalesce or changed visible sequence");
        if(thinking)need(sink.transitions.size()>=2 && sink.transitions.front()==OutputChannel::Reasoning &&
            sink.transitions.back()==OutputChannel::Content,"committed range queue changed channel ordering");
        {std::lock_guard lock(terminal_mutex);terminal=false;}
        auto bounded=submit(OutputConsumerMode::Streaming);
        bool refused=false;
        try{bounded.set_delivery_byte_limit_for_test(3);}catch(const std::invalid_argument&){refused=true;}
        need(refused,"delivery budget unable to fit UTF-8 scalar accepted");
        bounded.set_delivery_byte_limit_for_test(4);
        {
            std::unique_lock lock(terminal_mutex);
            need(terminal_changed.wait_for(lock,std::chrono::seconds(30),[&]{return terminal;}),
                "bounded terminal drain fixture did not finish production");
        }
        Sink bounded_sink;
        const auto bounded_result=bounded.wait(&bounded_sink,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(bounded_sink.largest<=4 && bounded_sink.calls>delayed_sink.calls &&
            bounded_sink.content==expected.content && bounded_sink.reasoning==expected.reasoning &&
            bounded_sink.transitions==sink.transitions && bounded_result.generated_token_ids==expected.generated_token_ids &&
            bounded_result.finish_reason==expected.finish_reason,
            "bounded final drain lost ranges, channels or output credit");
        if(!thinking) {
            struct RetainingSink final:OutputSink {
                std::vector<OutputDelta> held;
                std::string content,reasoning;
                void publish(OutputDelta delta) override {held.push_back(std::move(delta));}
                void release_front() {
                    auto delta=std::move(held.front());held.erase(held.begin());
                    (delta.channel==OutputChannel::Reasoning?reasoning:content)+=delta.text;
                }
                void release_all() {while(!held.empty())release_front();}
            } retaining;
            need(expected.content.size()+expected.reasoning.size()>12,
                "retained sink backpressure fixture output too short");
            {std::lock_guard lock(terminal_mutex);terminal=false;}
            auto backpressured=submit(OutputConsumerMode::Streaming);
            const auto backpressure_record=backpressured.result_owner_for_test();
            backpressured.set_delivery_byte_limit_for_test(4);
            {
                std::unique_lock lock(terminal_mutex);
                need(terminal_changed.wait_for(lock,std::chrono::seconds(30),[&]{return terminal;}),
                    "retained sink fixture did not reach terminal production");
            }
            GenerationPollResult pending;
            for(unsigned attempt=0;attempt<2;++attempt) {
                pending=backpressured.poll(&retaining,{});
                need(pending.state==GenerationPollState::Pending && retaining.held.size()==attempt+1 &&
                    backpressured.outstanding_delivery_batches_for_test()==attempt+1,
                    "retained sink did not acquire one bounded delivery owner");
            }
            const auto blocked=backpressured.poll(&retaining,CancellationView([]{return true;}));
            need(blocked.state==GenerationPollState::Pending && retaining.held.size()==2 &&
                backpressured.outstanding_delivery_batches_for_test()==2 && !backpressure_record.expired(),
                "retained sink cancellation exceeded two slots or released request credit");
            retaining.release_front();
            need(backpressured.outstanding_delivery_batches_for_test()==1,
                "released sink batch did not return its request slot");
            pending=backpressured.poll(&retaining,{});
            need(pending.state==GenerationPollState::Pending && retaining.held.size()==2 &&
                backpressured.outstanding_delivery_batches_for_test()==2,
                "returned delivery slot did not resume ordered drain");
            GenerationPollResult completed;
            const auto poll_bound=expected.content.size()+expected.reasoning.size()+2;
            for(std::size_t attempt=0;attempt<poll_bound;++attempt) {
                retaining.release_all();
                completed=backpressured.poll(&retaining,{});
                if(completed.state!=GenerationPollState::Pending)break;
            }
            retaining.release_all();
            need(completed.state==GenerationPollState::Completed && completed.result &&
                completed.result->generated_token_ids==expected.generated_token_ids &&
                completed.result->finish_reason==expected.finish_reason &&
                retaining.content==expected.content && retaining.reasoning==expected.reasoning &&
                backpressure_record.expired(),
                "retained sink backpressure lost ordering, result authority or reservation retirement");
        }
        owner.observe_terminal_roots_for_test({});
        {std::lock_guard lock(terminal_mutex);terminal=false;}
        }
        need(owner.close().reusable(),"stream batch delivery retirement");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_QUEUE_STARTUP");mode && std::string(mode)=="1") {
        const std::array<const char*,4> expected{
            "injected request queue preconstruction failure",
            "injected request queue precommit failure",
            "resource reservation actual extent/domain mismatch",
            "resource allocation reservation exhausted"};
        for(unsigned concurrency:{1u,2u})for(unsigned fault=71;fault<=74;++fault) {
            opt.max_concurrency=concurrency;opt.max_pending_requests=3;
            bool refused=false;
            try{ninfer::exl3::Exl3EngineCore failed(opt,0,0,0,fault);}
            catch(const std::exception& error){refused=std::string_view(error.what())==expected[fault-71];}
            need(refused,"request queue startup fault NOT_EXERCISED");
            ninfer::exl3::Exl3EngineCore retry(opt);
            need(retry.request_queue_capacity_for_test()==3+concurrency &&
                retry.runtime_stats().waiting_requests==0 && retry.runtime_stats().running_requests==0,
                "queue startup rollback prevented exact-capacity idle retry");
            need(retry.close().reusable(),"queue startup retry retirement");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_COORDINATOR_METADATA_STARTUP");mode && std::string(mode)=="1") {
        for(unsigned concurrency:{1u,2u})for(unsigned fault=75;fault<=78;++fault) {
            opt.max_concurrency=concurrency;opt.max_pending_requests=3;
            bool refused=false;
            try{ninfer::exl3::Exl3EngineCore failed(opt,0,0,0,fault);}
            catch(const std::bad_alloc&){refused=fault<=76;}
            catch(const std::exception& error){
                refused=(fault==77 && std::string_view(error.what())==
                    "resource reservation actual extent/domain mismatch") ||
                    (fault==78 && std::string_view(error.what())=="resource allocation reservation exhausted");
            }
            need(refused,"coordinator metadata startup fault NOT_EXERCISED");
            ninfer::exl3::Exl3EngineCore retry(opt);
            need(retry.runtime_stats().waiting_requests==0 && retry.runtime_stats().running_requests==0,
                "coordinator metadata startup retry not idle");
            need(retry.close().reusable(),"coordinator metadata startup retry retirement");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_CANCELLATION_GUARD_STARTUP");mode && std::string(mode)=="1") {
        const std::array<const char*,4> expected{
            "injected cancellation owner preconstruction failure",
            "injected cancellation owner precommit failure",
            "resource reservation actual extent/domain mismatch",
            "resource allocation reservation exhausted"};
        for(unsigned concurrency:{1u,2u})for(unsigned fault=61;fault<=64;++fault) {
            opt.max_concurrency=concurrency;
            const auto blocks=ninfer::exl3::Exl3EngineCore::cancellation_owner_blocks_for_test();
            bool refused=false;
            try{ninfer::exl3::Exl3EngineCore failed(opt,0,0,0,fault);}
            catch(const std::exception& error){refused=std::string_view(error.what())==expected[fault-61];}
            need(refused && ninfer::exl3::Exl3EngineCore::cancellation_owner_blocks_for_test()==blocks,
                "reserved cancellation guard startup fault unexercised or retained bounded block");
            std::weak_ptr<const void> guard;
            {
                ninfer::exl3::Exl3EngineCore retry(opt);
                guard=retry.cancellation_owner_for_test();
                const auto stats=retry.runtime_stats();
                need(!guard.expired() && stats.cancellation_owner_metadata_bytes>=256 &&
                    ninfer::exl3::Exl3EngineCore::cancellation_owner_blocks_for_test()==blocks+1 &&
                    stats.running_requests==0 && stats.waiting_requests==0,
                    "cancellation guard rollback prevented complete idle retry");
                need(retry.close().reusable(),"cancellation guard startup retry retirement");
                need(retry.retained_host_metadata_for_test()==stats.cancellation_owner_metadata_bytes+
                    retry.request_queue_metadata_bytes_for_test()+stats.execution_stream_metadata_bytes+
                    retry.shared_projection_metadata_bytes_for_test()+stats.draft_linear_owner_metadata_bytes+stats.draft_generic_owner_metadata_bytes+
                    stats.continuation_linear_owner_metadata_bytes+stats.context_base_linear_owner_metadata_bytes+
                    stats.context_layer_linear_owner_metadata_bytes+stats.continuation_generic_owner_metadata_bytes+
                    stats.context_private_generic_owner_metadata_bytes+stats.context_shared_generic_owner_metadata_bytes+
                    stats.context_shared_control_metadata_bytes,
                    "closed Engine dropped live startup guard/queue lifetime credit");
            }
            need(guard.expired(),"startup retry retained cancellation owner after destruction");
            need(ninfer::exl3::Exl3EngineCore::cancellation_owner_blocks_for_test()==blocks+1,
                "cancellation control backing disappeared before weak diagnostic retirement");
            guard.reset();
            need(ninfer::exl3::Exl3EngineCore::cancellation_owner_blocks_for_test()==blocks,
                "cancellation control backing survived final weak retirement");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_KV_BORROWER_LIFETIME");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        const auto controls_before=Exl3TextContext::live_shared_control_metadata_bytes();
        const auto quarantine_before=Exl3TextContext::retirement_quarantine_witness();
        std::shared_ptr<const void> borrower;
        std::array<std::uint64_t,3> credits{};
        {
            Exl3EngineCore owner(opt);
            borrower=owner.retain_host_kv_workspace_for_test();
            need(Exl3TextContext::shared_kv_credits_for_test(borrower)==credits,
                "KV credits attached before inventory retirement");
            need(owner.close().reusable(),"borrowed KV prevented idle Engine close");
            credits=Exl3TextContext::shared_kv_credits_for_test(borrower);
            need(credits[0]>0 && credits[1]==Exl3TextContext::allocation_record_metadata_bytes() &&
                credits[2]==Exl3ReconstructionControlAllocator<std::byte>::capacity,
                "Engine close omitted borrowed KV allocation/control credits");
        }
        need(Exl3TextContext::shared_kv_credits_for_test(borrower)==credits,
            "Engine destruction released independent KV borrower credits");
        const auto* failure_mode=std::getenv("NINFER_TEST_ENGINE_KV_BORROWER_FAULT");
        const std::string failure=failure_mode?failure_mode:"0";
        need(failure=="0" || failure=="1" || failure=="2" || failure=="3","KV borrower failure selector");
        const unsigned fault=static_cast<unsigned>(failure[0]-'0');
        if(fault)need(Exl3TextContext::fail_shared_kv_retirement_for_test(borrower,fault),
            "KV borrower failure injection refused joined owner");
        std::weak_ptr<const void> weak=borrower;borrower.reset();
        auto expected_quarantine=quarantine_before;
        if(fault)++expected_quarantine[4];
        need(weak.expired() && Exl3TextContext::retirement_quarantine_witness()==expected_quarantine,
            "final KV borrower retirement changed unexpected quarantine families");
        if(fault) {
            const auto retained=Exl3TextContext::generic_retirement_snapshot_for_test();
            const int expected_error=fault==1?static_cast<int>(cudaErrorUnknown):
                fault==2?static_cast<int>(cudaErrorInitializationError):static_cast<int>(cudaErrorInvalidDevice);
            need(retained.pointer && retained.bytes==credits[0] && retained.device_credit_bytes==credits[0] &&
                retained.metadata_credit_bytes==retained.metadata_bytes && retained.metadata_bytes<credits[1] &&
                retained.error==expected_error,"failed KV borrower lost exact allocation credits or cause");
        }
        need(Exl3TextContext::live_shared_control_metadata_bytes()==controls_before+
            Exl3ReconstructionControlAllocator<std::byte>::capacity,
            "KV weak observer lost control backing or retained unrelated controls");
        weak.reset();
        need(Exl3TextContext::live_shared_control_metadata_bytes()==controls_before,
            "final KV weak observer retained control backing");
        if(fault) {
            const auto retained=Exl3TextContext::generic_retirement_snapshot_for_test();
            need(retained.device_credit_bytes==credits[0] && retained.metadata_credit_bytes==retained.metadata_bytes &&
                Exl3TextContext::retirement_quarantine_witness()==expected_quarantine,
                "weak KV release dropped quarantine credits or retried failed cleanup");
        }
        std::cout<<"ENGINE_KV_BORROWER_LIFETIME_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_CONTINUATION_ALLOCATION_RETIREMENT");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string selector(mode);
        need(selector=="0" || selector=="1","continuation allocation slot selector");
        const unsigned slot=selector=="1"?1:0;
        const auto* fault_mode=std::getenv("NINFER_TEST_ENGINE_CONTINUATION_ALLOCATION_FAULT");
        const std::string fault_selector=fault_mode?fault_mode:"1";
        need(fault_selector=="1" || fault_selector=="2" || fault_selector=="3","continuation allocation fault selector");
        const unsigned fault=static_cast<unsigned>(fault_selector[0]-'0');
        need(Exl3TextContext::generic_quarantined_allocations()==0,"generic retirement requires fresh process");
        {
            Exl3EngineCore owner(opt);
            owner.fail_continuation_allocation_retirement_for_test(slot,fault);
            need(owner.close().reusable(),"continuation allocation close failed before final destruction");
            need(Exl3TextContext::generic_quarantined_allocations()==0,"continuation allocation retired before owner destruction");
        }
        need(Exl3TextContext::generic_quarantined_allocations()==1,"continuation failure did not retain exactly one allocation");
        const auto retained=Exl3TextContext::generic_retirement_snapshot_for_test();
        const int expected_error=fault==1?static_cast<int>(cudaErrorUnknown):
            fault==2?static_cast<int>(cudaErrorInitializationError):static_cast<int>(cudaErrorInvalidDevice);
        need(retained.pointer && retained.bytes>0 && retained.error==expected_error,
            "continuation retirement lost allocation identity or failure cause");
        need(retained.device_credit_bytes==retained.bytes &&
            retained.metadata_credit_bytes==retained.metadata_bytes &&
            retained.metadata_bytes<Exl3TextContext::allocation_record_metadata_bytes(),
            "continuation retirement lost device credit or retained destroyed wrapper metadata");
        std::cout<<"ENGINE_CONTINUATION_ALLOCATION_RETIREMENT_COMPLETE numerical_coverage=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_LAYER_BUFFER_CONSTRUCTOR_RELEASE");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        opt.max_concurrency=1;
        const auto before=Exl3TextContext::retirement_quarantine_witness();
        need(Exl3LayerBufferRetirement::quarantined()==0,"constructor release fixture requires fresh process");
        for(unsigned fault:{98u,99u,100u}) {
            bool original=false;
            try{Exl3EngineCore failed(opt,0,0,0,fault);}
            catch(const std::exception& error){original=std::string(error.what()).find(fault==100?
                "injected GDN arena wrapper allocation failure":"injected layer buffer postallocation failure")!=std::string::npos;}
            need(original,"successful cleanup replaced original layer constructor exception");
            need(Exl3TextContext::retirement_quarantine_witness()==before,
                "successful layer constructor cleanup changed quarantine witness");
            need(!Exl3HostResidentSet::failed_close_reservation_ceiling_for_test(),
                "released constructor credits left permanent failed reservation");
            {
                Exl3EngineCore retry(opt);
                need(retry.close().reusable(),"layer constructor cleanup prevented normal Engine retry");
            }
            need(Exl3TextContext::retirement_quarantine_witness()==before,
                "successful retry final destruction leaked a layer owner");
        }
        std::cout<<"ENGINE_LAYER_BUFFER_CONSTRUCTOR_RELEASE_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_LAYER_BUFFER_STARTUP_CREDITS");mode && *mode) {
        using namespace ninfer::exl3;
        using Owner=Exl3LayerBufferRetirement;
        const std::string selector(mode);
        const bool attention_constructor=selector=="attention-constructor";
        const bool gdn_constructor=selector=="gdn-constructor";
        const bool arena_constructor=selector=="arena-constructor";
        const bool constructor=attention_constructor || gdn_constructor || arena_constructor;
        need(selector=="attention" || selector=="gdn" || constructor,"startup buffer family selector");
        need(Owner::quarantined()==0 && Exl3CudaLinearWorkspace::quarantined_workspaces()==0,
            "startup buffer fixture requires fresh process");
        opt.max_concurrency=1;bool original=false;
        const unsigned fault=attention_constructor?95:(gdn_constructor?96:(arena_constructor?97:(selector=="gdn"?94:93)));
        const char* expected_error=arena_constructor?"injected GDN arena wrapper allocation failure":
            (constructor?"injected layer buffer postallocation failure":"resource reservation actual extent/domain mismatch");
        try{Exl3EngineCore engine(opt,0,0,0,fault);}
        catch(const std::exception& error){original=std::string(error.what()).find(expected_error)!=std::string::npos;}
        need(original,"layer buffer cleanup replaced startup inventory failure");
        const auto* retained=Owner::latest_for_test();
        need(Owner::quarantined()==1 && retained && retained->pointer && retained->bytes>0 && retained->device>=0 &&
            retained->error==static_cast<int>(cudaErrorUnknown),"startup buffer failure lost allocation identity");
        need(retained->device_credit && retained->device_credit->bytes()==retained->bytes &&
            retained->metadata_credit && retained->metadata_credit->bytes()==Owner::record_bytes(),
            "startup buffer unwind lost exact constructor credits");
        if(arena_constructor)need(retained->bytes==(1u<<20),"arena constructor retained wrong allocation extent");
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==0,"startup buffer cleanup retained unrelated linear storage");
        const auto ceiling=Exl3HostResidentSet::failed_close_reservation_ceiling_for_test();
        need(ceiling && (*ceiling)[0]>=retained->bytes,"startup buffer failure lost sealed reservation ceiling");
        bool refused=false;try{Exl3EngineCore retry(opt);}
        catch(const std::exception& error){refused=std::string(error.what()).find("unresolved layer buffer retirement")!=std::string::npos;}
        need(refused && Owner::quarantined()==1 && Owner::latest_for_test()==retained,
            "startup buffer reload retried failed cleanup");
        std::cout<<"ENGINE_LAYER_BUFFER_STARTUP_CREDITS_COMPLETE numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_ATTENTION_BUFFER_RETIREMENT");
        (mode && *mode) || ((mode=std::getenv("NINFER_TEST_ENGINE_GDN_BUFFER_RETIREMENT")) && *mode)) {
        using namespace ninfer::exl3;
        using Owner=Exl3LayerBufferRetirement;
        const auto* gdn_mode=std::getenv("NINFER_TEST_ENGINE_GDN_BUFFER_RETIREMENT");
        const bool gdn=gdn_mode && *gdn_mode;
        const auto* attention_mode=std::getenv("NINFER_TEST_ENGINE_ATTENTION_BUFFER_RETIREMENT");
        need(!(gdn && attention_mode && *attention_mode),"select only one layer buffer family per fresh process");
        const std::string selector(mode);std::size_t parsed=0;
        const auto colon=selector.find(':');const bool physical=colon!=std::string::npos;
        const auto index=std::stoul(selector.substr(0,colon),&parsed);
        need(parsed==(physical?colon:selector.size()) && index<=1024,"layer buffer index selector");
        unsigned slot=0;
        if(physical) {
            const auto suffix=selector.substr(colon+1);std::size_t slot_parsed=0;
            const auto value=std::stoul(suffix,&slot_parsed);
            need(slot_parsed==suffix.size() && value<(gdn?34u:18u),"physical layer buffer slot selector");
            slot=static_cast<unsigned>(value);
        }
        need(Owner::quarantined()==0 && Exl3CudaLinearWorkspace::quarantined_workspaces()==0,
            "attention buffer retirement requires fresh process");
        opt.max_concurrency=1;
        std::uint64_t expected_bytes=0;
        {
            Exl3EngineCore engine(opt);
            if(physical) {
                bool invalid_refused=false;
                try{engine.fail_layer_buffer_retirement_for_test(gdn,1024,slot);}
                catch(const std::invalid_argument&){invalid_refused=true;}
                need(invalid_refused && Owner::quarantined()==0,"invalid physical selection mutated retirement");
                expected_bytes=engine.fail_layer_buffer_retirement_for_test(gdn,static_cast<unsigned>(index),slot);
                if(gdn && (slot==29 || slot==30))need(expected_bytes==Exl3GdnLayer::kStateBytes,"GDN recurrent selected extent");
                if(gdn && slot==31)need(expected_bytes==Exl3GdnLayer::kConvStateStorageBytes,"GDN convolution storage selected extent");
                if(gdn && slot==32)need(expected_bytes==Exl3GdnLayer::kConvStateBytes,"GDN convolution trace selected extent");
                if(gdn && slot==33)need(expected_bytes==(1u<<20),"GDN operation arena selected extent");
            }
            else if(gdn)engine.fail_gdn_buffer_retirement_for_test(static_cast<unsigned>(index));
            else engine.fail_attention_buffer_retirement_for_test(static_cast<unsigned>(index));
            need(engine.close().reusable(),"attention buffer close failed before owner destruction");
            need(Owner::quarantined()==0,"attention buffer retired before Engine destruction");
        }
        const auto* retained=Owner::latest_for_test();
        if(physical)need(retained && retained->bytes==expected_bytes,"physical layer retained the wrong buffer extent");
        need(Owner::quarantined()==1 && retained && retained->pointer && retained->bytes>0 &&
            retained->device>=0 && retained->error==static_cast<int>(cudaErrorUnknown),
            "actual attention cleanup lost allocation identity");
        need(retained->device_credit && retained->device_credit->bytes()==retained->bytes &&
            retained->metadata_credit && retained->metadata_credit->bytes()==Owner::record_bytes(),
            "actual attention inventory lost exact final-owner credits");
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==0,"attention buffer fault retained unrelated linear workspace");
        bool refused=false;try{Exl3EngineCore retry(opt);}
        catch(const std::exception& error){refused=std::string(error.what()).find("unresolved layer buffer retirement")!=std::string::npos;}
        need(refused && Owner::quarantined()==1 && Owner::latest_for_test()==retained &&
            retained->device_credit->bytes()==retained->bytes && retained->metadata_credit->bytes()==Owner::record_bytes(),
            "attention reload retried cleanup or changed credits");
        std::cout<<(gdn?"ENGINE_GDN_BUFFER_RETIREMENT_COMPLETE":"ENGINE_ATTENTION_BUFFER_RETIREMENT_COMPLETE")
            <<" numerical_coverage=0\n";return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_CONTEXT_LINEAR_RETIREMENT");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string selector(mode);
        std::size_t parsed=0;const auto slot=std::stoul(selector,&parsed);
        need(parsed==selector.size() && slot<=1024,"context linear retirement slot selector");
        const auto* partial_mode=std::getenv("NINFER_TEST_ENGINE_CONTEXT_LINEAR_PARTIAL");
        const bool partial=partial_mode && std::string_view(partial_mode)=="1";
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==0,"context linear retirement requires fresh process");
        {
            Exl3EngineCore owner(opt);
            owner.fail_context_linear_retirement_for_test(static_cast<unsigned>(slot),partial);
            need(owner.close().reusable(),"idle context close failed before child destruction");
            need(Exl3CudaLinearWorkspace::quarantined_workspaces()==0,"context child retired before enclosing Engine destruction");
        }
        need(Exl3CudaLinearWorkspace::quarantined_workspaces()==1,"context destruction did not retain exactly one failed child");
        const auto retained=Exl3CudaLinearWorkspace::latest_retirement_for_test();
        const auto credits=Exl3CudaLinearWorkspace::latest_retirement_credits_for_test();
        need(retained.error!=0 && retained.transformed_bytes+retained.accumulation_bytes>0,
            "context failed child lost retained device extent");
        if(partial)need(!retained.accumulation_retained,"partial context cleanup retained released accumulation");
        need(credits[0]==Exl3CudaLinearWorkspace::metadata_bytes() &&
            credits[1]==retained.transformed_bytes+retained.accumulation_bytes,
            "production context inventory failed to preserve exact child lifetime charges");
        std::cout<<"ENGINE_CONTEXT_LINEAR_RETIREMENT_COMPLETE numerical_coverage=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SUBMISSION_OUTLIVES_OWNER");mode && std::string(mode)=="1") {
        std::optional<ninfer::exl3::Exl3EngineCore::Submission> dropped,waited;
        std::weak_ptr<const void> cancellation_guard;
        std::weak_ptr<const void> dropped_record,waited_record;
        std::uint64_t waited_reservation=0;
        std::uint64_t waited_fixed_bytes=0;
        std::uint64_t waited_option_bytes=0;
        std::uint64_t waited_prompt_bytes=0;
        std::uint64_t waited_session_bytes=0;
        {
            ninfer::exl3::Exl3EngineCore owner(opt);
            cancellation_guard=owner.cancellation_owner_for_test();
            need(!cancellation_guard.expired(),"live Engine missing cancellation guard owner");
            need(owner.runtime_stats().cancellation_owner_metadata_bytes>0,
                "cancellation lifetime guard omitted reserved metadata");
            runtime::ResolvedRequestOptions request;
            request.execution.sampling.temperature=0;request.execution.requested_output_tokens=8;
            request.execution.allow_prefix_reuse=false;request.stop=owner.frontend().default_stop_policy();
            const auto submit=[&] {
                auto prepared=owner.frontend().prepare(prompt("Explain bridge foundations."));
                const auto summary=prepared.summary();
                return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(5));
            };
            dropped.emplace(submit());waited.emplace(submit());
            dropped_record=dropped->result_owner_for_test();
            waited_record=waited->result_owner_for_test();
            waited_reservation=waited->result_reserved_bytes_for_test();
            waited_fixed_bytes=waited->result_fixed_bytes_for_test();
            waited_option_bytes=waited->option_storage_bytes_for_test();
            waited_prompt_bytes=waited->prompt_storage_bytes_for_test();
            waited_session_bytes=waited->initial_session_bytes_for_test();
            const auto detached_bytes=dropped->result_reserved_bytes_for_test()+waited_reservation+
                owner.runtime_stats().cancellation_owner_metadata_bytes+owner.request_queue_metadata_bytes_for_test()+
                owner.runtime_stats().execution_stream_metadata_bytes+owner.shared_projection_metadata_bytes_for_test()+
                owner.runtime_stats().draft_linear_owner_metadata_bytes+owner.runtime_stats().draft_generic_owner_metadata_bytes+
                owner.runtime_stats().continuation_linear_owner_metadata_bytes+
                owner.runtime_stats().context_base_linear_owner_metadata_bytes+
                owner.runtime_stats().context_layer_linear_owner_metadata_bytes+
                owner.runtime_stats().continuation_generic_owner_metadata_bytes+
                owner.runtime_stats().context_private_generic_owner_metadata_bytes+
                owner.runtime_stats().context_shared_generic_owner_metadata_bytes+
                owner.runtime_stats().context_shared_control_metadata_bytes;
            need(owner.close().reusable(),"outliving submissions prevented owner retirement");
            need(!dropped_record.expired() && !waited_record.expired() &&
                owner.retained_host_metadata_for_test()==detached_bytes,
                "Engine close released unconsumed result storage");
        }
        need(cancellation_guard.expired(),"detached submissions retained cancellation guard payload");
        need(!dropped_record.expired() && !waited_record.expired(),
            "Engine destruction released detached result storage");
        dropped.reset(); // Calls cancellation after Impl destruction.
        need(dropped_record.expired() && !waited_record.expired(),
            "discarding detached result retained its record or released a peer");
        unsigned polls=0;
        const auto result=waited->wait(nullptr,CancellationView([&]{++polls;return true;}));
        need(polls==1 && result.generated_token_ids.size()<=8,
            "completed outliving submission lost terminal delivery");
        need(waited_record.expired() && waited_fixed_bytes>sizeof(GenerationResult) && waited_prompt_bytes>0 &&
            waited_session_bytes>0 && waited_reservation==waited_fixed_bytes+waited_option_bytes+waited_prompt_bytes+waited_session_bytes+
            result.generated_token_ids.capacity()*sizeof(decltype(GenerationResult::generated_token_ids)::value_type)+
            result.reasoning.capacity()+result.content.capacity()+2,
            "detached result failed to transfer reserved token capacity and release record");
        waited.reset();
        auto replacement=std::make_unique<ninfer::exl3::Exl3EngineCore>(opt);
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;request.execution.requested_output_tokens=8;
        request.execution.allow_prefix_reuse=false;request.stop=replacement->frontend().default_stop_policy();
        auto prepared=replacement->frontend().prepare(prompt("Explain bridge foundations."));
        const auto summary=prepared.summary();
        auto racing=replacement->submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
            std::chrono::steady_clock::now()+std::chrono::minutes(5));
        auto racing_guard=replacement->cancellation_owner_for_test();
        const auto racing_record=racing.result_owner_for_test();
        need(replacement->close().reusable(),"detached cancellation guard prevented replacement Engine");
        // Workers have retired. Race only the submission's guard pin/detach
        // protocol against Impl destruction; either lock ordering is valid.
        std::mutex start_mutex;std::condition_variable start_changed;bool start=false;
        std::exception_ptr cancellation_failure;
        unsigned racing_polls=0;
        std::thread canceller([&,submission=std::move(racing)]() mutable {
            try {
                {std::unique_lock lock(start_mutex);start_changed.wait(lock,[&]{return start;});}
                const auto terminal=submission.wait(
                    nullptr,CancellationView([&]{++racing_polls;return true;}));
                need(terminal.generated_token_ids.size()<=8,"concurrent detached submission lost output bound");
            } catch(...) {cancellation_failure=std::current_exception();}
        });
        {std::lock_guard lock(start_mutex);start=true;start_changed.notify_all();}
        replacement.reset();
        canceller.join();
        if(cancellation_failure)std::rethrow_exception(cancellation_failure);
        need(racing_polls==1 && racing_guard.expired() && racing_record.expired(),
            "concurrent cancellation retained guard/result or lost terminal delivery");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_READY_WORK");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        opt.max_concurrency=1;opt.max_pending_requests=1;
        struct Gate {std::mutex mutex;std::condition_variable changed;bool entered=false,released=false;};
        auto gate=std::make_shared<Gate>();
        Exl3EngineCore owner(opt);
        struct Release {
            std::shared_ptr<Gate> gate;
            ~Release(){std::lock_guard lock(gate->mutex);gate->released=true;gate->changed.notify_all();}
        } release{gate};
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;request.execution.requested_output_tokens=8;
        request.execution.allow_prefix_reuse=false;request.stop=owner.frontend().default_stop_policy();
        const auto submit=[&](std::string text) {
            auto prepared=owner.frontend().prepare(prompt(std::move(text)));
            const auto summary=prepared.summary();
            auto submission=owner.submit(std::move(prepared),summary,0,request,
                OutputConsumerMode::Aggregate,std::chrono::steady_clock::now()+std::chrono::minutes(5));
            return std::pair{std::move(submission),summary};
        };
        owner.observe_request_start_for_test([gate] {
            std::unique_lock lock(gate->mutex);gate->entered=true;gate->changed.notify_all();
            need(gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->released;}),
                "ready-work fixture worker release timeout");
        });
        auto [active,active_summary]=submit("Explain why bridges need strong foundations.");
        const auto active_work=active.ready_work_for_test();
        {
            std::unique_lock lock(gate->mutex);
            need(gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->entered;}),
                "ready-work fixture did not hold assigned worker");
        }
        auto [queued,queued_summary]=submit("Explain why bridge cables need careful tensioning.");
        const auto queued_work=queued.ready_work_for_test();
        const auto queued_state=queued_work.assess({queued_work.generation,false,false,false,false});
        const auto missing_resource=queued_work.assess({queued_work.generation,true,true,false,false});
        const auto stale=queued_work.assess({active_work.generation,true,true,true,false});
        need(active_work.generation && queued_work.generation>active_work.generation &&
            active_work.input_tokens==active_summary.prompt_tokens &&
            queued_work.input_tokens==queued_summary.prompt_tokens &&
            active_work.resources.logical_host_bytes==active.result_reserved_bytes_for_test() &&
            queued_work.resources.logical_host_bytes==queued.result_reserved_bytes_for_test() &&
            active_work.signature.compatible_with(queued_work.signature) &&
            active_work.signature.stage==Exl3ReadyNumericalStage::HostPreparation &&
            active_work.signature.arithmetic==Exl3ReadyArithmetic::OrdinaryFp16B8GreedyText &&
            queued_state.logically_admitted && !queued_state.physically_executable &&
            queued_state.blocked==Exl3ReadyWorkBlock::BlockedDependency &&
            missing_resource.blocked==Exl3ReadyWorkBlock::MissingResource &&
            stale.blocked==Exl3ReadyWorkBlock::StaleAcquisition,
            "Engine queue did not publish complete immutable ready-work facts");
        unsigned cancellation_polls=0;
        const auto cancelled=queued.wait(
            nullptr,CancellationView([&]{++cancellation_polls;return true;}));
        const auto cancelled_state=queued_work.assess({active_work.generation,false,false,false,true});
        need(cancellation_polls==1 && cancelled.finish_reason==FinishReason::Cancelled &&
            cancelled.generated_token_ids.empty() &&
            cancelled_state.blocked==Exl3ReadyWorkBlock::Cancelled &&
            owner.runtime_stats().waiting_requests==0,
            "queued cancellation changed immutable work or retained logical capacity");
        {std::lock_guard lock(gate->mutex);gate->released=true;gate->changed.notify_all();}
        const auto completed=active.wait(nullptr,{});
        need(completed.generated_token_ids.size()<=8 && owner.runtime_stats().waiting_requests==0 &&
            owner.runtime_stats().running_requests==0,
            "ready-work integration lost active completion or queue accounting");
        need(owner.close().reusable(),"ready-work integration retirement");
        std::cout<<"ENGINE_READY_WORK_COMPLETE stale=1 dependency=1 cancellation=1 missing_resource=1 numerical_coverage=0\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_CANCELLED_QUEUE_CAPACITY");mode && std::string(mode)=="1") {
        opt.max_concurrency=1;opt.max_pending_requests=1;
        struct Gate {std::mutex mutex;std::condition_variable changed;bool entered=false,released=false;unsigned starts=0;};
        auto gate=std::make_shared<Gate>();
        ninfer::exl3::Exl3EngineCore owner(opt);
        struct Release {
            std::shared_ptr<Gate> gate;
            ~Release(){std::lock_guard lock(gate->mutex);gate->released=true;gate->changed.notify_all();}
        } release{gate};
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;request.execution.requested_output_tokens=8;
        request.execution.allow_prefix_reuse=false;request.stop=owner.frontend().default_stop_policy();
        const auto submit=[&] {
            auto prepared=owner.frontend().prepare(prompt("Explain why bridges need strong foundations."));
            const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        auto baseline=submit();const auto expected=baseline.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        owner.observe_request_start_for_test([gate] {
            std::unique_lock lock(gate->mutex);++gate->starts;gate->entered=true;gate->changed.notify_all();
            need(gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->released;}),
                "cancelled queue fixture worker release timeout");
        });
        auto active=submit();
        {
            std::unique_lock lock(gate->mutex);
            need(gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->entered;}),
                "cancelled queue fixture did not hold worker");
        }
        {auto abandoned=submit();} // Destructor cancels while the only worker is held.
        need(owner.runtime_stats().waiting_requests==0,"abandoned queue entry required another arrival");
        {
            auto mismatched=submit();
            std::atomic<bool> sink_cancel{false};
            CancellingSink sink(sink_cancel);
            bool rejected=false;
            try{mismatched.wait(&sink,{});}
            catch(const std::invalid_argument& error){rejected=std::string_view(error.what())=="EXL3 consumer mode mismatch";}
            need(rejected && sink.text.empty() && !sink_cancel.load() && owner.runtime_stats().waiting_requests==0,
                "consumer mismatch retained queued request or published output");
            {std::lock_guard lock(gate->mutex);need(gate->starts==1 && !gate->released,
                "consumer mismatch needed worker progress for cleanup");}
        }
        auto replacement=submit(); // Must reclaim cancelled logical capacity.
        std::optional<ninfer::exl3::Exl3EngineCore::Submission> survivor;
        unsigned cancellation_polls=0;
        const auto cancelled=replacement.wait(nullptr,CancellationView([&] {
            ++cancellation_polls;
            return true;
        }));
        need(cancellation_polls==1 && !survivor.has_value() && owner.runtime_stats().waiting_requests==0 &&
            cancelled.finish_reason==FinishReason::Cancelled && cancelled.generated_token_ids.empty(),
            "queued cancellation was not delivered to waiting consumer");
        {std::lock_guard lock(gate->mutex);need(!gate->released && gate->starts==1,
            "queued cancellation required active worker release");}
        survivor.emplace(submit());
        const auto held=owner.runtime_stats();
        need(held.running_requests==1 && held.waiting_requests==1,
            "cancelled queued request retained capacity or changed active ownership");
        bool overloaded=false;
        try{auto excess=submit();}catch(const RequestError& error){overloaded=error.kind()==RequestErrorKind::Overloaded;}
        need(overloaded,"queue cleanup discarded a live replacement");
        {std::lock_guard lock(gate->mutex);gate->released=true;gate->changed.notify_all();}
        const auto actual=active.wait(nullptr,{}),replaced=survivor->wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(actual.generated_token_ids==expected.generated_token_ids && replaced.generated_token_ids==expected.generated_token_ids &&
            actual.finish_reason==expected.finish_reason && replaced.finish_reason==expected.finish_reason,
            "cancelled queue cleanup changed live request output");
        {std::lock_guard lock(gate->mutex);need(gate->starts==2,"cancelled queued request acquired physical worker");}
        need(owner.runtime_stats().waiting_requests==0 && owner.runtime_stats().running_requests==0,
            "cancelled queue completion lost logical accounting");
        {std::lock_guard lock(gate->mutex);gate->entered=false;gate->released=false;}
        auto claimed=submit();
        {
            std::unique_lock lock(gate->mutex);
            need(gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->entered;}),
                "active cancellation fixture did not cross dequeue boundary");
        }
        auto queued_peer=submit();
        unsigned active_polls=0;
        const auto active_cancelled=claimed.wait(nullptr,CancellationView([&] {
            if(++active_polls==2) {
                // First poll already cancelled. Active credit must remain until
                // the held worker is released and performs its own completion.
                const auto retained=owner.runtime_stats();
                need(retained.running_requests==1 && retained.waiting_requests==1,
                    "direct cancellation retired already-claimed worker");
                std::lock_guard lock(gate->mutex);gate->released=true;gate->changed.notify_all();
            }
            return true;
        }));
        need(active_polls>=2 && active_cancelled.finish_reason==FinishReason::Cancelled &&
            active_cancelled.generated_token_ids.empty(),"active cancellation lost worker completion boundary");
        const auto peer_result=queued_peer.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(peer_result.generated_token_ids==expected.generated_token_ids && peer_result.finish_reason==expected.finish_reason,
            "active cancellation changed queued peer output");
        {std::lock_guard lock(gate->mutex);need(gate->starts==4,"dequeue cancellation changed worker count");}
        need(owner.runtime_stats().waiting_requests==0 && owner.runtime_stats().running_requests==0,
            "active cancellation released logical credit incorrectly");
        need(owner.close().reusable(),"cancelled queue capacity fixture retirement");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_OUTPUT_EXTENT_FAILURE");mode && std::string(mode)=="1") {
        opt.max_concurrency=1;
        std::atomic<unsigned> roots{0};
        ninfer::exl3::Exl3EngineCore owner(opt);
        owner.observe_terminal_roots_for_test([&](auto){++roots;});
        owner.fail_next_output_extent_for_test();
        bool duplicate_refused=false;
        try{owner.fail_next_output_extent_for_test();}catch(const std::logic_error&){duplicate_refused=true;}
        need(duplicate_refused,"output extent fault allowed duplicate arming");
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;request.execution.requested_output_tokens=8;
        request.execution.allow_prefix_reuse=false;request.stop=owner.frontend().default_stop_policy();
        const auto submit=[&] {
            auto prepared=owner.frontend().prepare(prompt("Explain bridge foundations."));
            const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Streaming,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        struct Sink final:OutputSink {unsigned calls=0;void publish(OutputDelta) override{++calls;}} sink;
        auto handle=submit();const auto record=handle.result_owner_for_test();
        bool failed=false;
        try{handle.wait(&sink,{});}catch(const std::logic_error& error) {
            failed=std::string_view(error.what())=="Engine output session exceeds reserved private extent";
        }
        need(failed && sink.calls==0 && roots.load()==0,
            "session extent violation escaped guard or published output/root");
        bool unavailable=false;
        try{auto rejected=submit();}catch(const RequestError& error){unavailable=error.kind()==RequestErrorKind::Unavailable;}
        need(unavailable,"session invariant failure did not seal Engine admission");
        need(owner.close().reusable(),"host session extent failure prevented clean Engine retirement");
        need(record.expired() && owner.runtime_stats().committed_decode_tokens==0,
            "failed session retained request backing or credited uncommitted decode");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_ADMISSION_BUDGET");mode && std::string(mode)=="1") {
        opt.max_concurrency=2;
        std::mutex roots_mutex;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>> roots;
        ninfer::exl3::Exl3EngineCore owner(opt);
        owner.observe_terminal_roots_for_test([&](auto root) {
            std::lock_guard lock(roots_mutex);roots.push_back(std::move(root));
        });
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=8;
        request.execution.allow_prefix_reuse=false;
        request.stop=owner.frontend().default_stop_policy();
        const auto submit=[&] {
            auto prepared=owner.frontend().prepare(prompt("Explain why bridges need strong foundations."));
            const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        auto baseline=submit();const auto expected=baseline.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        const auto before=owner.runtime_stats();
        owner.fail_next_admission_budget_for_test();
        auto first=submit(),second=submit();
        unsigned refusals=0,completed=0;
        for(auto* handle:{&first,&second}) {
            try {
                const auto result=handle->wait(nullptr,{});
                need(result.generated_token_ids==expected.generated_token_ids &&
                    result.finish_reason==expected.finish_reason,"admission refusal changed peer output");
                ++completed;
            } catch(const RequestError& error) {
                need(error.kind()==RequestErrorKind::Overloaded && std::string_view(error.what())==
                    "EXL3 request exceeds available resident page budget","admission budget lost request-local error");
                ++refusals;
            }
        }
        owner.set_host_kv_routes_for_test(false,false);
        const auto after=owner.runtime_stats();
        need(refusals==1 && completed==1 && roots.size()==2 &&
            after.failure_cancelled_active_requests==before.failure_cancelled_active_requests &&
            after.failure_cancelled_queued_requests==before.failure_cancelled_queued_requests,
            "admission budget refusal poisoned Engine or cancelled peer");
        auto retry=submit();const auto retried=retry.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(retried.generated_token_ids==expected.generated_token_ids && retried.finish_reason==expected.finish_reason &&
            roots.size()==3 && roots[0]->state()->same_payload(*roots[1]->state()) &&
            roots[0]->state()->same_payload(*roots[2]->state()),
            "admission budget retry lost authoritative state");
        {
            using Root=ninfer::exl3::Exl3VeriCacheRequest;
            const std::array<std::shared_ptr<const Root>,1> one{roots[0]};
            const std::array<std::shared_ptr<const Root>,2> alias{roots[0],roots[0]};
            const auto one_metadata=Root::metadata_allocation_stats(one);
            const auto alias_metadata=Root::metadata_allocation_stats(alias);
            std::uint64_t payload_bytes=0;std::size_t payload_ranges=0;
            Root::visit_host_allocations(alias,[&](const void*,std::size_t bytes) {
                need(bytes<=UINT64_MAX-payload_bytes,"root payload visitor sum overflow");
                payload_bytes+=bytes;++payload_ranges;
            },false);
            const auto kind=[](Root::MetadataOwnerKind value) {
                return static_cast<std::size_t>(value);
            };
            need(one_metadata.logical_bytes>0 && one_metadata.unique_owners>0 &&
                alias_metadata.logical_bytes==one_metadata.logical_bytes &&
                alias_metadata.unique_owners==one_metadata.unique_owners &&
                alias_metadata.bytes_by_kind==one_metadata.bytes_by_kind &&
                one_metadata.bytes_by_kind[kind(Root::MetadataOwnerKind::request)]>0 &&
                one_metadata.bytes_by_kind[kind(Root::MetadataOwnerKind::revision)]>0 &&
                one_metadata.bytes_by_kind[kind(Root::MetadataOwnerKind::draft_ring)]>0 &&
                one_metadata.bytes_by_kind[kind(Root::MetadataOwnerKind::draft_page)]>0 &&
                one_metadata.bytes_by_kind[kind(Root::MetadataOwnerKind::exact_page)]>0 &&
                one_metadata.bytes_by_kind[kind(Root::MetadataOwnerKind::token_node)]>0 &&
                one_metadata.bytes_by_kind[kind(Root::MetadataOwnerKind::exact_state)]>0 &&
                payload_bytes>0 && payload_ranges>0,
                "root allocation visitor duplicated aliases or mixed logical metadata with physical payload");
            bool overflow=false;
            try{(void)Root::checked_metadata_sum_for_test(UINT64_MAX,1);}
            catch(const std::overflow_error&){overflow=true;}
            need(overflow && Root::checked_metadata_sum_for_test(UINT64_MAX-1,1)==UINT64_MAX,
                "root metadata visitor lost checked overflow boundary");
        }
        {
            auto invalid_options=request;
            invalid_options.execution.requested_output_tokens=2;
            invalid_options.execution.thinking.budget=1;
            auto invalid_prompt=owner.frontend().prepare(prompt("Explain bridge foundations.",true));
            const auto invalid_summary=invalid_prompt.summary();
            bool refused=false;
            try {
                auto rejected=owner.submit(std::move(invalid_prompt),invalid_summary,0,invalid_options,
                    OutputConsumerMode::Aggregate,std::chrono::steady_clock::now()+std::chrono::minutes(5));
            } catch(const std::invalid_argument& error) {
                refused=std::string_view(error.what())==
                    "effective output capacity after the thinking budget must fit the complete control suffix and one post-close model token";
            }
            need(refused && roots.size()==3 && owner.runtime_stats().running_requests==0 &&
                owner.runtime_stats().waiting_requests==0,
                "invalid output-session capacity reached a worker or published a terminal root");
        }
        for(unsigned stage=1;stage<=9;++stage) {
        const auto result_blocks=ninfer::exl3::Exl3EngineCore::result_owner_blocks_for_test();
        for(const unsigned invalid:{0u,10u}) {
            bool invalid_refused=false;
            try{owner.fail_next_result_allocation_for_test(invalid);}catch(const std::invalid_argument&){invalid_refused=true;}
            need(invalid_refused,"invalid result allocation stage armed a fault");
        }
        owner.fail_next_result_allocation_for_test(stage);
        bool double_arm_refused=false;
        try{owner.fail_next_result_allocation_for_test();}catch(const std::logic_error&){double_arm_refused=true;}
        bool allocation_refused=false;
        try{auto rejected=submit();}catch(const RequestError& error) {
            allocation_refused=error.kind()==RequestErrorKind::Overloaded &&
                std::string_view(error.what())=="EXL3 result storage allocation failed";
        }
        const auto refused_stats=owner.runtime_stats();
        need(double_arm_refused && allocation_refused && roots.size()==2+stage &&
            ninfer::exl3::Exl3EngineCore::result_owner_blocks_for_test()==result_blocks &&
            refused_stats.running_requests==0 && refused_stats.waiting_requests==0 &&
            refused_stats.failure_cancelled_active_requests==after.failure_cancelled_active_requests &&
            refused_stats.failure_cancelled_queued_requests==after.failure_cancelled_queued_requests,
            "result allocation refusal published a request or poisoned Engine");
        auto allocation_retry=submit();const auto recovered=allocation_retry.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(recovered.generated_token_ids==expected.generated_token_ids && recovered.finish_reason==expected.finish_reason &&
            recovered.content==expected.content && recovered.reasoning==expected.reasoning &&
            roots.size()==3+stage && roots[0]->state()->same_payload(*roots.back()->state()),
            "result allocation rollback prevented authoritative retry");
        }
        const auto removed_root=roots.front();
        const std::array<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>,1>
            removed_roots{removed_root};
        const auto removed_before=
            ninfer::exl3::Exl3VeriCacheRequest::metadata_allocation_stats(removed_roots);
        owner.observe_terminal_roots_for_test({});
        roots.clear();
        need(owner.close().reusable(),"admission budget fixture retirement");
        const auto removed_after=
            ninfer::exl3::Exl3VeriCacheRequest::metadata_allocation_stats(removed_roots);
        need(removed_after.logical_bytes==removed_before.logical_bytes &&
            removed_after.unique_owners==removed_before.unique_owners &&
            removed_after.bytes_by_kind==removed_before.bytes_by_kind,
            "removed but externally retained root lost nested allocation ownership");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_COST_INSTALL");mode && std::string(mode)=="1") {
        using Policy=ninfer::exl3::Exl3PackedCostPolicy;
        opt.max_concurrency=2;
        ninfer::exl3::Exl3EngineCore owner(opt);
        const std::array<Policy::Entry,1> entries{{{0,5120,12288,6,3,5,100,40,30}}};
        const Policy supplied(entries);
        const auto before_policy=owner.runtime_stats();
        owner.set_shared_cost_policy(supplied);
        bool refused=false;
        try{owner.set_shared_cost_policy(Policy::unqualified_test_permit_underfilled());}
        catch(const std::invalid_argument&){refused=true;}
        need(refused,"production shared policy admitted test bypass");
        need(owner.runtime_stats().shared_cost_policy_updates==before_policy.shared_cost_policy_updates+1,
            "failed policy installation changed update count");
        owner.set_shared_cost_policy(Policy{});
        const auto cleared_policy=owner.runtime_stats();
        need(cleared_policy.shared_cost_policy_updates==before_policy.shared_cost_policy_updates+2 &&
            cleared_policy.shared_underfilled_cost_accepts==before_policy.shared_underfilled_cost_accepts &&
            cleared_policy.shared_underfilled_cost_refusals==before_policy.shared_underfilled_cost_refusals,
            "idle cost policy update fabricated a physical membership decision");
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>> policy_roots;
        std::mutex policy_roots_mutex;
        unsigned busy_refusals=0;
        owner.observe_terminal_roots_for_test([&](auto root) {
            std::lock_guard lock(policy_roots_mutex);
            policy_roots.push_back(std::move(root));
            try{owner.set_shared_cost_policy(Policy{});}
            catch(const std::logic_error&){++busy_refusals;}
        });
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=16;request.execution.allow_prefix_reuse=false;
        request.stop=owner.frontend().default_stop_policy();
        const auto run=[&] {
            auto prepared=owner.frontend().prepare(prompt("Explain why a bridge needs strong foundations."));
            const auto summary=prepared.summary();
            auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            auto result=handle.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            return result.generated_token_ids;
        };
        owner.set_host_kv_routes_for_test(false,false);
        const auto baseline=run();
        owner.set_shared_cost_policy(supplied);const auto selected=run();
        owner.set_shared_cost_policy(Policy{});const auto restored=run();
        const auto after_requests=owner.runtime_stats();
        need(busy_refusals==3 && after_requests.shared_cost_policy_updates==before_policy.shared_cost_policy_updates+4,
            "busy request mutated shared policy");
        need(baseline==selected && baseline==restored && policy_roots.size()==3,
            "single-request shared policy changed output");
        need(policy_roots[0]->state()->same_payload(*policy_roots[1]->state()) &&
            policy_roots[0]->state()->same_payload(*policy_roots[2]->state()),
            "single-request shared policy changed authoritative state");
        need(after_requests.shared_underfilled_cost_accepts==before_policy.shared_underfilled_cost_accepts &&
            after_requests.shared_underfilled_cost_refusals==before_policy.shared_underfilled_cost_refusals,
            "single request fabricated compatible peer decisions");
        // Synthetic supplied costs cover every ordered underfilled Q pair;
        // they are not calibrated and never enable the test bypass.
        std::array<Policy::Entry,63> pair_entries;
        std::size_t entry_count=0;
        for(int first=1;first<=8;++first)for(int second=1;second<=8;++second)
            if(first+second<16)pair_entries[entry_count++]={0,5120,12288,6,first,second,1000,1,50};
        owner.set_shared_cost_policy(Policy(pair_entries));
        owner.claim_next_shared_pair_at_age_for_test(std::chrono::microseconds(1));
        const auto before_pair=owner.runtime_stats();
        const auto submit_pair_member=[&] {
            auto prepared=owner.frontend().prepare(prompt("Explain why a bridge needs strong foundations."));
            const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        auto first_pair=submit_pair_member();auto second_pair=submit_pair_member();
        const auto first_result=first_pair.wait(nullptr,{}),second_result=second_pair.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        const auto after_pair=owner.runtime_stats();
        if(after_pair.shared_underfilled_cost_accepts==before_pair.shared_underfilled_cost_accepts)
            std::cerr<<"SHARED_COST_PAIR accepts="
                <<after_pair.shared_underfilled_cost_accepts-before_pair.shared_underfilled_cost_accepts
                <<" refusals="<<after_pair.shared_underfilled_cost_refusals-before_pair.shared_underfilled_cost_refusals
                <<" batches="<<after_pair.shared_target_projection_batches-before_pair.shared_target_projection_batches
                <<" rows="<<after_pair.shared_target_projection_rows-before_pair.shared_target_projection_rows
                <<" fallbacks="<<after_pair.shared_target_projection_fallbacks-before_pair.shared_target_projection_fallbacks
                <<" capped_waits="<<after_pair.shared_policy_capped_waits-before_pair.shared_policy_capped_waits
                <<" capped_budget_us="<<after_pair.shared_policy_capped_wait_budget_us-before_pair.shared_policy_capped_wait_budget_us
                <<'\n';
        need(after_pair.shared_underfilled_cost_accepts>before_pair.shared_underfilled_cost_accepts,
            "supplied underfilled Engine pair NOT_EXERCISED");
        need(after_pair.shared_target_projection_batches>before_pair.shared_target_projection_batches &&
            after_pair.shared_target_projection_failures==before_pair.shared_target_projection_failures,
            "supplied decision did not complete shared work");
        need(first_result.generated_token_ids==baseline && second_result.generated_token_ids==baseline &&
            policy_roots.size()==5 && busy_refusals==5,"supplied pair changed output or allowed busy mutation");
        for(std::size_t i=3;i<5;++i)need(policy_roots[0]->state()->same_payload(*policy_roots[i]->state()),
            "supplied pair changed authoritative state");
        owner.set_shared_cost_policy(Policy{});
        const auto before_cleared_pair=owner.runtime_stats();
        auto cleared_first=submit_pair_member();auto cleared_second=submit_pair_member();
        const auto cleared_first_result=cleared_first.wait(nullptr,{}),cleared_second_result=cleared_second.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        const auto after_cleared_pair=owner.runtime_stats();
        need(after_cleared_pair.shared_underfilled_cost_accepts==before_cleared_pair.shared_underfilled_cost_accepts,
            "cleared table retained underfilled admission");
        need(cleared_first_result.generated_token_ids==baseline && cleared_second_result.generated_token_ids==baseline &&
            policy_roots.size()==7 && busy_refusals==7,"cleared paired policy changed output or accepted busy update");
        for(std::size_t i=5;i<7;++i)need(policy_roots[0]->state()->same_payload(*policy_roots[i]->state()),
            "cleared paired policy changed authoritative state");
        // Exercise the actual offer boundary without asserting scheduler timing.
        // Every eligible underfilled first member has exactly a 1us budget.
        for(auto& entry:pair_entries)entry.max_age_us=1;
        owner.set_shared_cost_policy(Policy(pair_entries));
        const auto before_bounded=owner.runtime_stats();
        const auto bounded=run();
        const auto after_bounded=owner.runtime_stats();
        const auto capped=after_bounded.shared_policy_capped_waits-before_bounded.shared_policy_capped_waits;
        need(capped>0,"bounded supplied Engine wait NOT_EXERCISED");
        need(after_bounded.shared_policy_capped_wait_budget_us-before_bounded.shared_policy_capped_wait_budget_us==capped,
            "actual Engine offer ignored 1us supplied wait budget");
        need(bounded==baseline && policy_roots.size()==8 && busy_refusals==8 &&
            policy_roots[0]->state()->same_payload(*policy_roots[7]->state()),
            "bounded wait changed tokens, state or busy policy isolation");
        need(after_bounded.shared_underfilled_cost_accepts==before_bounded.shared_underfilled_cost_accepts &&
            after_bounded.shared_underfilled_cost_refusals==before_bounded.shared_underfilled_cost_refusals,
            "single bounded offer fabricated a peer decision");
        // Repeated isolated arrivals must receive a fresh bounded decision;
        // earlier timeouts cannot leave a peer or accumulated wait credit.
        for(unsigned repetition=0;repetition<2;++repetition) {
            const auto before_repeat=owner.runtime_stats();
            const auto repeated=run();
            const auto after_repeat=owner.runtime_stats();
            const auto repeat_caps=after_repeat.shared_policy_capped_waits-before_repeat.shared_policy_capped_waits;
            need(repeat_caps>0 &&
                after_repeat.shared_policy_capped_wait_budget_us-before_repeat.shared_policy_capped_wait_budget_us==repeat_caps,
                "repeated isolated arrival lost its bounded wait policy");
            need(after_repeat.shared_underfilled_cost_accepts==before_repeat.shared_underfilled_cost_accepts &&
                after_repeat.shared_underfilled_cost_refusals==before_repeat.shared_underfilled_cost_refusals &&
                after_repeat.shared_target_projection_batches==before_repeat.shared_target_projection_batches,
                "timed-out isolated offer survived as a shared peer");
            need(repeated==baseline && policy_roots.size()==9+repetition && busy_refusals==9+repetition &&
                policy_roots[0]->state()->same_payload(*policy_roots.back()->state()),
                "repeated bounded arrival changed output, state or policy isolation");
        }
        owner.set_shared_cost_policy(Policy{});
        const auto before_unbounded=owner.runtime_stats();
        const auto cleared_bounded=run();
        const auto after_unbounded=owner.runtime_stats();
        need(after_unbounded.shared_policy_capped_waits==before_unbounded.shared_policy_capped_waits &&
            after_unbounded.shared_policy_capped_wait_budget_us==before_unbounded.shared_policy_capped_wait_budget_us,
            "cleared policy retained capped wait accounting");
        need(cleared_bounded==baseline && policy_roots.size()==11 && busy_refusals==11 &&
            policy_roots[0]->state()->same_payload(*policy_roots[10]->state()),
            "cleared bounded policy changed authoritative output");
        // Two requested tokens cross exactly one continuation projection after
        // the first prefill-logit token, reaching the final output allowance
        // without relying on a peer arriving or wall-clock assertions.
        request.execution.requested_output_tokens=2;
        owner.set_shared_cost_policy(Policy(pair_entries));
        const auto before_short=owner.runtime_stats();
        const auto short_capped=run();
        const auto after_short=owner.runtime_stats();
        const auto short_caps=after_short.shared_policy_capped_waits-before_short.shared_policy_capped_waits;
        need(short_caps>0 &&
            after_short.shared_policy_capped_wait_budget_us-before_short.shared_policy_capped_wait_budget_us==short_caps,
            "final-output request did not retain its bounded offer policy");
        need(after_short.shared_target_projection_batches==before_short.shared_target_projection_batches &&
            after_short.shared_underfilled_cost_accepts==before_short.shared_underfilled_cost_accepts &&
            after_short.shared_underfilled_cost_refusals==before_short.shared_underfilled_cost_refusals,
            "final-output isolated request fabricated a peer");
        owner.set_shared_cost_policy(Policy{});
        const auto short_independent=run();
        const auto after_short_clear=owner.runtime_stats();
        need(short_capped.size()==2 && short_capped==short_independent &&
            policy_roots.size()==13 && busy_refusals==13 &&
            policy_roots[11]->state()->same_payload(*policy_roots[12]->state()),
            "final-output bounded wait changed tokens, state or policy isolation");
        need(after_short_clear.shared_policy_capped_waits==after_short.shared_policy_capped_waits &&
            after_short_clear.shared_policy_capped_wait_budget_us==after_short.shared_policy_capped_wait_budget_us,
            "final-output request retained cleared wait policy");
        request.execution.requested_output_tokens=16;
        owner.set_shared_cost_policy_for_test(Policy::unqualified_test_permit_underfilled());
        owner.set_shared_rendezvous_timeout_for_test(std::chrono::milliseconds(5));
        const auto expired_before=owner.expired_shared_pair_refusals_for_test();
        const auto before_late=owner.runtime_stats();
        owner.expire_next_shared_pair_for_test();
        bool rearm_refused=false;
        try{owner.expire_next_shared_pair_for_test();}
        catch(const std::logic_error&){rearm_refused=true;}
        need(rearm_refused,"pending expired-pair seam was silently replaced");
        auto late_first=submit_pair_member(),late_second=submit_pair_member();
        const auto late_a=late_first.wait(nullptr,{}),late_b=late_second.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(owner.expired_shared_pair_refusals_for_test()==expired_before+1,
            "actual expired compatible pair rejection NOT_EXERCISED");
        need(late_a.generated_token_ids==baseline && late_b.generated_token_ids==baseline &&
            policy_roots.size()==15 && busy_refusals==15 &&
            policy_roots[0]->state()->same_payload(*policy_roots[13]->state()) &&
            policy_roots[0]->state()->same_payload(*policy_roots[14]->state()),
            "expired pair fallback changed private output or state");
        need(owner.runtime_stats().shared_target_projection_failures==before_late.shared_target_projection_failures,
            "expired pair rejection poisoned shared owner");
        for(const auto& retained:owner.shared_claim_owners_for_test())
            need(retained.expired(),"expired-pair completion retained claim authority");
        owner.split_shared_contracts_for_test(true);
        const auto before_split=owner.runtime_stats();
        auto split_first=submit_pair_member(),split_second=submit_pair_member();
        const auto split_a=split_first.wait(nullptr,{}),split_b=split_second.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        const auto after_split=owner.runtime_stats();
        need(after_split.shared_projection_contract_refusals>before_split.shared_projection_contract_refusals,
            "C2 diagnostic contract rejection NOT_EXERCISED");
        need(after_split.shared_target_projection_batches==before_split.shared_target_projection_batches &&
            after_split.shared_target_projection_failures==before_split.shared_target_projection_failures,
            "mixed contracts dispatched together or poisoned shared owner");
        need(split_a.generated_token_ids==baseline && split_b.generated_token_ids==baseline && policy_roots.size()==17,
            "contract split changed private tokens or omitted terminal roots");
        for(std::size_t i=15;i<17;++i)need(policy_roots[0]->state()->same_payload(*policy_roots[i]->state()),
            "contract split changed authoritative state");
        for(const auto& retained:owner.shared_claim_owners_for_test())
            need(retained.expired(),"contract refusal retained claim ownership");
        owner.split_shared_contracts_for_test(false);
        auto resumed_first=submit_pair_member(),resumed_second=submit_pair_member();
        const auto resumed_a=resumed_first.wait(nullptr,{}),resumed_b=resumed_second.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        const auto after_resumed=owner.runtime_stats();
        need(after_resumed.shared_target_projection_batches>after_split.shared_target_projection_batches &&
            after_resumed.shared_projection_contract_refusals==after_split.shared_projection_contract_refusals,
            "cleared contract split did not recover same-contract sharing");
        need(resumed_a.generated_token_ids==baseline && resumed_b.generated_token_ids==baseline && policy_roots.size()==19,
            "recovered contract sharing changed private tokens");
        for(std::size_t i=17;i<19;++i)need(policy_roots[0]->state()->same_payload(*policy_roots[i]->state()),
            "recovered contract sharing changed authoritative state");
        bool invalid_preclaim=false;
        try{owner.set_shared_preclaim_fault_for_test(4);}catch(const std::invalid_argument&){invalid_preclaim=true;}
        need(invalid_preclaim,"Engine accepted unknown preclaim diagnostic");
        for(unsigned fault:{1u,2u,3u}) {
            owner.set_shared_preclaim_fault_for_test(fault);
            const auto before=owner.runtime_stats();const auto root_count=policy_roots.size();
            auto first=submit_pair_member(),second=submit_pair_member();
            const auto a=first.wait(nullptr,{}),b=second.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            const auto after=owner.runtime_stats();
            need(fault==1?after.shared_projection_geometry_refusals>before.shared_projection_geometry_refusals:
                fault==2?after.shared_projection_authority_refusals>before.shared_projection_authority_refusals:
                after.shared_projection_stage_refusals>before.shared_projection_stage_refusals,
                "Engine preclaim refusal NOT_EXERCISED");
            need(fault==1?after.shared_projection_authority_refusals==before.shared_projection_authority_refusals:
                after.shared_projection_geometry_refusals==before.shared_projection_geometry_refusals,
                "Engine preclaim refusal attributed to wrong gate");
            if(fault==3)need(after.shared_projection_authority_refusals==before.shared_projection_authority_refusals,
                "opposite-stage refusal reached request authority gate");
            need(after.shared_target_projection_batches==before.shared_target_projection_batches &&
                after.shared_target_projection_failures==before.shared_target_projection_failures &&
                after.shared_projection_completed_dispatches==before.shared_projection_completed_dispatches,
                "Engine preclaim refusal dispatched or poisoned owner");
            need(a.generated_token_ids==baseline && b.generated_token_ids==baseline && policy_roots.size()==root_count+2,
                "Engine preclaim fallback changed private tokens");
            for(auto i=root_count;i<policy_roots.size();++i)
                need(policy_roots[0]->state()->same_payload(*policy_roots[i]->state()),"Engine preclaim fallback changed private state");
            for(const auto& retained:owner.shared_claim_owners_for_test())
                need(retained.expired(),"Engine preclaim refusal retained claim ownership");
        }
        owner.set_shared_preclaim_fault_for_test(0);
        const auto before_recovery=owner.runtime_stats();
        const auto recovery_roots=policy_roots.size();
        auto recovered_first=submit_pair_member(),recovered_second=submit_pair_member();
        const auto recovered_a=recovered_first.wait(nullptr,{}),recovered_b=recovered_second.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(owner.runtime_stats().shared_target_projection_batches>before_recovery.shared_target_projection_batches &&
            recovered_a.generated_token_ids==baseline && recovered_b.generated_token_ids==baseline,
            "Engine sharing did not recover after clearing preclaim fault");
        need(policy_roots.size()==recovery_roots+2,"Engine preclaim recovery omitted terminal roots");
        for(auto i=recovery_roots;i<policy_roots.size();++i)
            need(policy_roots[0]->state()->same_payload(*policy_roots[i]->state()),"Engine preclaim recovery changed private state");
        // Arm a one-shot observer at the actual unclaimed rendezvous boundary.
        // It publishes while the rendezvous mutex is still held, so cancellation
        // notification must synchronize with the following condition wait rather
        // than relying on a sleep or on a guessed worker arrival interval.
        std::mutex wait_mutex;std::condition_variable wait_changed;bool wait_registered=false;
        owner.observe_next_shared_wait_for_test([&] {
            std::lock_guard lock(wait_mutex);wait_registered=true;wait_changed.notify_all();
        });
        const auto before_wait_cancel=owner.runtime_stats();
        const auto wait_cancel_roots=policy_roots.size();
        auto active_waiter=submit_pair_member();
        const auto cancel_active_waiter=active_waiter.cancellation_callback_for_test();
        {
            std::unique_lock lock(wait_mutex);
            need(wait_changed.wait_for(lock,std::chrono::seconds(30),[&]{return wait_registered;}),
                "active shared waiter did not register at rendezvous");
        }
        cancel_active_waiter();
        const auto cancelled_waiter=active_waiter.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        const auto after_wait_cancel=owner.runtime_stats();
        need(cancelled_waiter.finish_reason==FinishReason::Cancelled && cancelled_waiter.generated_token_ids.empty(),
            "active rendezvous cancellation published output or lost cancellation");
        need(after_wait_cancel.shared_target_projection_batches==before_wait_cancel.shared_target_projection_batches &&
            after_wait_cancel.shared_target_projection_failures==before_wait_cancel.shared_target_projection_failures &&
            after_wait_cancel.shared_projection_completed_dispatches==before_wait_cancel.shared_projection_completed_dispatches &&
            after_wait_cancel.shared_target_projection_fallbacks>before_wait_cancel.shared_target_projection_fallbacks &&
            after_wait_cancel.running_requests==0 && after_wait_cancel.waiting_requests==0,
            "active rendezvous cancellation claimed work, poisoned owner or retained request credit");
        for(const auto& retained:owner.shared_claim_owners_for_test())
            need(retained.expired(),"cancelled unclaimed waiter retained physical claim ownership");
        const auto before_wait_recovery=owner.runtime_stats();
        auto wait_recovery_first=submit_pair_member(),wait_recovery_second=submit_pair_member();
        const auto wait_recovery_a=wait_recovery_first.wait(nullptr,{});
        const auto wait_recovery_b=wait_recovery_second.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(owner.runtime_stats().shared_target_projection_batches>before_wait_recovery.shared_target_projection_batches &&
            wait_recovery_a.generated_token_ids==baseline && wait_recovery_b.generated_token_ids==baseline,
            "cancelled active waiter prevented subsequent shared progress");
        need(policy_roots.size()>=wait_cancel_roots+2,"active-wait cancellation recovery omitted terminal roots");
        for(std::size_t i=policy_roots.size()-2;i<policy_roots.size();++i)
            need(policy_roots[0]->state()->same_payload(*policy_roots[i]->state()),
                "active-wait cancellation recovery changed private state");
        // Completed submissions may still expose a cancellation callback. Its
        // rendezvous notification must be harmless and cannot cancel a later
        // acquisition or make completed output disappear.
        const auto cancel_completed=recovered_first.cancellation_callback_for_test();
        const auto before_late_cancel=owner.runtime_stats();
        cancel_completed();cancel_completed();
        const auto after_late_cancel=owner.runtime_stats();
        need(after_late_cancel.shared_target_projection_batches==before_late_cancel.shared_target_projection_batches &&
            after_late_cancel.shared_target_projection_failures==before_late_cancel.shared_target_projection_failures &&
            after_late_cancel.shared_projection_completed_dispatches==before_late_cancel.shared_projection_completed_dispatches &&
            after_late_cancel.running_requests==0 && after_late_cancel.waiting_requests==0,
            "completed cancellation changed idle shared work or request ownership");
        const auto late_roots=policy_roots.size();
        auto next_first=submit_pair_member(),next_second=submit_pair_member();
        const auto next_a=next_first.wait(nullptr,{}),next_b=next_second.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(next_a.generated_token_ids==baseline && next_b.generated_token_ids==baseline &&
            policy_roots.size()==late_roots+2,
            "completed cancellation contaminated subsequent shared acquisitions");
        for(auto i=late_roots;i<policy_roots.size();++i)
            need(policy_roots[0]->state()->same_payload(*policy_roots[i]->state()),
                "late cancellation changed subsequent private state");
        owner.set_shared_cost_policy(Policy{});
        owner.set_shared_rendezvous_timeout_for_test(std::chrono::microseconds(50));
        need(owner.close().reusable(),"shared supplied policy retirement");
        refused=false;try{owner.set_shared_cost_policy(supplied);}
        catch(const std::logic_error&){refused=true;}
        need(refused,"closed Engine accepted supplied shared policy");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_CONFIG");mode && std::string(mode)=="1") {
        opt.max_concurrency=2;
        struct Restore {
            std::vector<std::pair<std::string,std::string>> values;
            ~Restore(){for(const auto& [name,value]:values)_putenv_s(name.c_str(),value.c_str());}
        } restore;
        const std::array<const char*,28> names{
            "NINFER_EXL3_ENGINE_SHARED_TARGET_Q","NINFER_EXL3_ENGINE_SHARED_TARGET_KV",
            "NINFER_EXL3_ENGINE_SHARED_TARGET_O","NINFER_EXL3_ENGINE_SHARED_DRAFT_Q_M16",
            "NINFER_EXL3_ENGINE_SHARED_DEVICE_PREFIX","NINFER_EXL3_TARGET_Q_K6_SMALL_M",
            "NINFER_EXL3_TARGET_KV_SMALL_M","NINFER_EXL3_TARGET_O_K6_SMALL_M",
            "NINFER_EXL3_HOST_KV_DEVICE_PREFIX","NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION",
            "NINFER_EXL3_EXTENDED_STREAM_REDUCTION","NINFER_EXL3_ENGINE_SHARED_TARGET_GATEUP",
            "NINFER_EXL3_ENGINE_SHARED_TARGET_DOWN","NINFER_EXL3_ENGINE_SHARED_GATHER_REUSE",
            "NINFER_EXL3_ENGINE_SHARED_HEAD","NINFER_EXL3_H6_SMALL_M","NINFER_EXL3_ENGINE_SHARED_DRAFT_KV_M16",
            "NINFER_EXL3_ENGINE_SHARED_DRAFT_O_M16","NINFER_EXL3_ENGINE_SHARED_DRAFT_DOWN_M16",
            "NINFER_EXL3_ENGINE_SHARED_DRAFT_GATEUP_M16",
            "NINFER_EXL3_ENGINE_REGISTERED_KV_UPLOAD","NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES",
            "NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGE_ATTENTION","NINFER_EXL3_ENGINE_ATTENTION_STAGING",
            "NINFER_EXL3_DRAFT_SMALL_M","NINFER_EXL3_ENGINE_BATCHED_GREEDY_PACKET",
            "NINFER_EXL3_DEVICE_GREEDY","NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS"};
        for(const auto* name:names){const auto* value=std::getenv(name);restore.values.emplace_back(name,value?value:"");}
        const auto baseline=[&]{for(const auto* name:names)_putenv_s(name,"0");};
        const auto refused=[&](const char* expected) {
            bool rejected=false;
            try{ninfer::exl3::Exl3EngineCore invalid(opt);}
            catch(const std::invalid_argument& e){rejected=std::string(e.what()).find(expected)!=std::string::npos;}
            need(rejected,"shared configuration was not refused at its declared boundary");
        };
        for(unsigned i=0;i<5;++i){baseline();_putenv_s(names[i],"2");refused("must be0 or1");}
        baseline();_putenv_s("NINFER_EXL3_ENGINE_SHARED_DRAFT_GATEUP_M16","2");
        refused("gate/up M16 option requires 0 or 1");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_SHARED_DRAFT_GATEUP_M16","1");
        refused("gate/up requires C2 shared draft Q M16");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_REGISTERED_KV_UPLOAD","2");refused("must be0 or1");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES","2");refused("must be0 or1");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGE_ATTENTION","2");refused("must be 0 or 1");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_ATTENTION_STAGING","2");refused("must be 0 or 1");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_BATCHED_GREEDY_PACKET","2");refused("must be0 or1");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_BATCHED_GREEDY_PACKET","1");refused("requires device greedy");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_BATCHED_GREEDY_PACKET","1");
        _putenv_s("NINFER_EXL3_DEVICE_GREEDY","1");opt.max_concurrency=1;
        refused("requires physical C2");opt.max_concurrency=2;
        baseline();_putenv_s("NINFER_EXL3_ENGINE_ATTENTION_STAGING","1");
        _putenv_s("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES","1");refused("staging and shared cache routes are exclusive");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGE_ATTENTION","1");refused("requires shared device pages");
        baseline();_putenv_s("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES","1");
        _putenv_s("NINFER_EXL3_ENGINE_SHARED_DEVICE_PREFIX","1");
        _putenv_s("NINFER_EXL3_HOST_KV_DEVICE_PREFIX","1");
        refused("exclusive Engine routes");
        baseline();_putenv_s("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS","4096x");
        refused("require an unsigned integer");
        baseline();_putenv_s(names[11],"2");refused("must be0 or1");
        baseline();_putenv_s(names[11],"1");refused("shared gate/up requires");
        baseline();_putenv_s(names[12],"2");refused("must be0 or1");
        baseline();_putenv_s(names[12],"1");refused("shared down requires");
        baseline();_putenv_s(names[13],"2");refused("must be0 or1");
        baseline();_putenv_s(names[13],"1");refused("gather reuse requires");
        baseline();_putenv_s(names[14],"2");refused("must be0 or1");
        baseline();_putenv_s(names[14],"1");refused("shared head requires");
        baseline();_putenv_s(names[16],"2");refused("must be0 or1");
        baseline();_putenv_s(names[16],"1");refused("draft KV requires");
        baseline();_putenv_s(names[17],"2");refused("must be0 or1");
        baseline();_putenv_s(names[17],"1");refused("draft O requires");
        baseline();_putenv_s(names[18],"2");refused("must be0 or1");
        baseline();_putenv_s(names[18],"1");refused("draft down requires");
        baseline();_putenv_s(names[0],"1");_putenv_s(names[5],"1");_putenv_s(names[14],"1");
        refused("shared head requires"); // explicit H6 opt-out cannot be overridden
        baseline();_putenv_s(names[0],"1");refused("explicit Q/K6");
        baseline();_putenv_s(names[1],"1");refused("shared KV requires");
        baseline();_putenv_s(names[2],"1");refused("shared O requires");
        baseline();_putenv_s(names[3],"1");refused("enabled draft small-M route");
        baseline();_putenv_s(names[3],"1");_putenv_s(names[24],"1");
        {
            ninfer::exl3::Exl3EngineCore isolated_draft(opt);
            need(isolated_draft.shared_projection_metadata_bytes_for_test()>0,
                "isolated draft did not construct the common projection owner");
            need(isolated_draft.close().reusable(),"isolated draft owner retirement");
        }
        baseline();_putenv_s(names[4],"1");refused("shared prefix requires");
        for(unsigned i:{9U,10U}) {
            baseline();_putenv_s(names[0],"1");_putenv_s(names[5],"1");_putenv_s(names[i],"1");
            refused("shared target route conflict");
        }
        baseline();
        {
            ninfer::exl3::Exl3EngineCore disabled(opt);
            using Policy=ninfer::exl3::Exl3PackedCostPolicy;
            const std::array<Policy::Entry,1> estimates{{{0,5120,12288,6,3,5,100,40,30}}};
            for(const auto& policy:{Policy{},Policy(estimates)}) {
                bool rejected=false;try{disabled.set_shared_cost_policy(policy);}
                catch(const std::logic_error&){rejected=true;}
                need(rejected,"supplied policy implicitly enabled disabled sharing");
            }
            const auto counts=disabled.runtime_stats();
            need(counts.shared_cost_policy_updates==0 && counts.shared_underfilled_cost_accepts==0 &&
                counts.shared_target_projection_batches==0,"disabled policy refusal changed sharing state");
            need(disabled.close().reusable(),"disabled shared policy owner retirement");
        }
        // Exercise the smallest supported mixed-family owner.  Larger optional
        // families (notably head/down) must not be required to mask a packed
        // Q/O buffer-accounting error.
        baseline();
        _putenv_s("NINFER_EXL3_ENGINE_SHARED_TARGET_Q","1");
        _putenv_s("NINFER_EXL3_TARGET_Q_K6_SMALL_M","1");
        _putenv_s("NINFER_EXL3_ENGINE_SHARED_TARGET_O","1");
        _putenv_s("NINFER_EXL3_TARGET_O_K6_SMALL_M","1");
        {
            ninfer::exl3::Exl3EngineCore shared_q_o(opt);
            need(shared_q_o.close().reusable(),"minimal shared Q/O owner retirement");
        }
        std::cout<<"engine_shared_q_o_minimal_config PASS\n";
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_NATIVE64K");mode && std::string_view(mode)=="1") {
        using Extent=ninfer::exl3::Exl3NativeContextExtent;
        ninfer::exl3::test::check_future_context_qualification_matrix_source();
        const auto candidate=Extent::candidate128k_plan(131072,false,true,true,false,1,8ULL<<30,false);
        need(candidate.maximum_context==131072 && candidate.last_position==131071 &&
            candidate.exact_kv_pages==2048 && candidate.exact_kv_payload_bytes==(8ULL<<30) &&
            candidate.minimum_host_reservation_bytes==(8ULL<<30),
            "128K static integer/page/reserve candidate contract");
        bool candidate_short=false,candidate_cache=false,candidate_c2=false;
        bool candidate_million=false,candidate_execution=false;
        try{(void)Extent::candidate128k_plan(131072,false,true,true,false,1,(8ULL<<30)-1,false);}
        catch(const std::invalid_argument&){candidate_short=true;}
        try{(void)Extent::candidate128k_plan(131072,false,true,true,false,1,8ULL<<30,true);}
        catch(const std::invalid_argument&){candidate_cache=true;}
        try{(void)Extent::candidate128k_plan(131072,false,true,true,false,2,8ULL<<30,false);}
        catch(const std::invalid_argument&){candidate_c2=true;}
        try{(void)Extent::checked_max_context(1000000,false,true);}
        catch(const std::invalid_argument&){candidate_million=true;}
        try{Extent::refuse_candidate128k_execution(131072,false,true,true,false,1,8ULL<<30,false);}
        catch(const std::invalid_argument&){candidate_execution=true;}
        need(candidate_short && candidate_cache && candidate_c2 && candidate_million && candidate_execution,
            "128K candidate admitted an under-reserved/cache/C2/million/execution configuration");
        need(Extent::configuration_limit(true,false)==65536 &&
            Extent::checked_max_context(65536,true,false)==65536 &&
            Extent::checked_input_end(65536,65536)==65536 &&
            Extent::checked_position(65535,65536)==65535 &&
            Extent::exact_kv_payload_bytes(65536)==(4ULL<<30),
            "native64K configuration/input/position/KV extent contract");
        bool above=false,short_budget=false,mixed_control=false;
        try{(void)Extent::checked_input_end(65537,65536);}catch(const std::invalid_argument&){above=true;}
        try{Extent::require_native64k_engine(65536,true,false,true,false,1,(4ULL<<30)-1);}
        catch(const std::invalid_argument&){short_budget=true;}
        try{(void)Extent::configuration_limit(true,true);}catch(const std::invalid_argument&){mixed_control=true;}
        need(above && short_budget && mixed_control,
            "native64K boundary or one-request resource refusal contract");
        Extent::require_native64k_engine(65536,true,false,true,false,1,4ULL<<30);
        opt.max_concurrency=1;opt.max_context=65536;
        opt.kv_capacity=KvCapacityPolicy::explicit_capacity(65536);
        ninfer::exl3::Exl3EngineCore owner(opt);
        const auto* exact_option=std::getenv("NINFER_TEST_ENGINE_NATIVE64K_EXACT_INPUT");
        need(!exact_option || std::string_view(exact_option)=="0" ||
            std::string_view(exact_option)=="1","native64K exact-input option must be0or1");
        if(exact_option && std::string_view(exact_option)=="1") {
            std::vector<TokenId> tokens(65536,1);
            auto prepared=owner.frontend().prepare_tokens(std::move(tokens),true);
            const auto summary=prepared.summary();
            need(summary.prompt_tokens==65536,"native64K exact input changed prepared length");
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=16;request.execution.allow_prefix_reuse=false;
            request.stop=owner.frontend().default_stop_policy();
            const auto output=owner.submit(std::move(prepared),summary,0,request,
                OutputConsumerMode::Aggregate,std::chrono::steady_clock::now()+
                std::chrono::minutes(60)).wait(nullptr,{});
            const auto stats=owner.runtime_stats();
            need(output.finish_reason==FinishReason::ContextCapacity &&
                output.generated_token_ids.empty() && stats.computed_prefill_tokens==65536 &&
                stats.committed_decode_tokens==0 &&
                output.token_accounting.conserves_result_tokens(0) &&
                output.token_accounting.diagnostic_state_rows==0,
                "native64K exact input did not stop at typed context boundary");
            need(owner.close().reusable(),"native64K exact-input request did not retire");
            return 0;
        }
        std::mutex terminal_mutex;
        std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> terminal;
        owner.observe_terminal_roots_for_test([&](auto root) {
            std::lock_guard lock(terminal_mutex);
            need(!terminal,"native64K produced more than one terminal authority");
            terminal=std::move(root);
        });
        // Bounded tokenizer-only fixture construction when this source is later
        // authorized. Select one genuine near-64K prompt, never aggregate roots.
        std::optional<targets::qwen3_6::PreparedPrompt> selected;
        unsigned low=1,high=65536;
        for(unsigned attempt=0;attempt<17 && low<=high;++attempt) {
            const auto repeats=low+(high-low)/2;
            std::string text;text.reserve(static_cast<std::size_t>(repeats)*13+128);
            text="The first marker is COBALT.\n";
            for(unsigned i=0;i<repeats;++i)text+=" river bridge";
            text+="\nState the first marker in one word.";
            const auto count=owner.frontend().count_tokens(prompt(text));
            if(count>=65504 && count<=65520) {
                selected.emplace(owner.frontend().prepare(prompt(std::move(text))));break;
            }
            if(count<65504)low=repeats+1;else high=repeats-1;
        }
        need(selected.has_value(),"native64K fixture could not form bounded real prompt");
        const auto summary=selected->summary();
        unsigned attachment_cancel=0;
        if(const auto* value=std::getenv("NINFER_TEST_ENGINE_NATIVE64K_CANCEL_ATTACHMENT");
            value && *value) {
            const std::string_view selected_phase(value);
            attachment_cancel=selected_phase=="1"?1:selected_phase=="2"?2:selected_phase=="3"?3:0;
            need(attachment_cancel,"native64K attachment cancellation requires phase1..3");
            owner.cancel_next_attachment_for_test(attachment_cancel);
        }
        unsigned restore_cancel_layer=0;
        if(const auto* value=std::getenv("NINFER_TEST_ENGINE_NATIVE64K_CANCEL_RESTORE_LAYER");
            value && *value) {
            need(std::string_view(value)=="0" || std::string_view(value)=="24",
                "native64K mid-restore cancellation requires layer24 or 0");
            if(std::string_view(value)=="24") {
                restore_cancel_layer=24;
                owner.cancel_next_attachment_during_restore_for_test(restore_cancel_layer);
            }
        }
        need(!(attachment_cancel && restore_cancel_layer),
            "native64K attachment cancellation fixtures are mutually exclusive");
        const auto attachment_quarantine_before=
            ninfer::exl3::Exl3TextContext::retirement_quarantine_witness();
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=16;request.execution.allow_prefix_reuse=false;
        request.stop=owner.frontend().default_stop_policy();
        auto handle=owner.submit(std::move(*selected),summary,0,request,OutputConsumerMode::Aggregate,
            std::chrono::steady_clock::now()+std::chrono::minutes(60));
        const auto output=handle.wait(nullptr,{});
        need(output.token_accounting.conserves_result_tokens(output.generated_token_ids.size()) &&
            output.token_accounting.injected_control_tokens==0 &&
            output.token_accounting.diagnostic_state_rows==0,
            "native64K result conflated semantic/control/diagnostic rows");
        if(attachment_cancel || restore_cancel_layer) {
            std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> cancelled_terminal;
            {std::lock_guard lock(terminal_mutex);cancelled_terminal=terminal;}
            const auto cancelled_stats=owner.runtime_stats();
            need(output.finish_reason==FinishReason::Cancelled &&
                output.generated_token_ids.empty() && !cancelled_terminal &&
                owner.attachment_cancel_hit_for_test()==attachment_cancel &&
                owner.attachment_restore_cancel_hit_for_test()==
                    (restore_cancel_layer?restore_cancel_layer:0) &&
                cancelled_stats.acquired_attachment_cancellations==1 &&
                ninfer::exl3::Exl3TextContext::retirement_quarantine_witness()==
                    attachment_quarantine_before,
                "native64K cancelled attachment published or lost its safe boundary");
            auto isolated=owner.frontend().prepare_tokens(std::vector<TokenId>{1,2,3,4},true);
            const auto isolated_summary=isolated.summary();
            auto isolated_request=request;isolated_request.execution.requested_output_tokens=1;
            const auto isolated_output=owner.submit(std::move(isolated),isolated_summary,0,
                isolated_request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5)).wait(nullptr,{});
            {std::lock_guard lock(terminal_mutex);cancelled_terminal=terminal;}
            need(!isolated_output.generated_token_ids.empty() && cancelled_terminal &&
                cancelled_terminal->state()->position()==
                    static_cast<int>(isolated_summary.prompt_tokens+
                        isolated_output.generated_token_ids.size()),
                "request after native64K attachment cancellation inherited cancelled state");
            need(owner.close().reusable(),
                "native64K attachment cancellation prevented Engine retirement");
            return 0;
        }
        need(output.generated_token_ids.size()<=16 && summary.prompt_tokens+output.generated_token_ids.size()<=65536,
            "native64K output/context budget exceeded");
        need(output.finish_reason==FinishReason::OutputLimit || output.finish_reason==FinishReason::StopToken ||
            output.finish_reason==FinishReason::StopString,"native64K request did not finish at an explicit output/stop boundary");
        std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> terminal_root;
        {std::lock_guard lock(terminal_mutex);terminal_root=terminal;}
        need(terminal_root && terminal_root->compact_draft(),
            "native64K request omitted terminal target/draft authority");
        auto detached_state=terminal_root->state()->detached_payload_for_test();
        auto detached_conditioning=terminal_root->detached_projected_conditioning_for_test();
        need(detached_state && terminal_root->state()->same_represented_payload_for_test(*detached_state) &&
            detached_state->position()==terminal_root->state()->position() &&
            detached_state->rope_offset()==terminal_root->state()->rope_offset(),
            "native64K terminal state detach changed represented authority");
        need(detached_conditioning && terminal_root->matches_detached_conditioning_for_test(*detached_conditioning),
            "native64K terminal private conditioning detach changed represented authority");
        int published_end=0;std::size_t published_extents=0;
        terminal_root->state()->visit_kv_for_test([&](int,const int first,const int rows,auto key,auto value) {
            need(first>=0 && rows>0 && first<=terminal_root->state()->position() &&
                rows<=terminal_root->state()->position()-first &&
                key.size()==static_cast<std::size_t>(rows)*1024 && value.size()==key.size(),
                "native64K terminal KV extent/plane mismatch");
            published_end=std::max(published_end,first+rows);++published_extents;
        });
        need(published_extents && published_end==terminal_root->state()->position(),
            "native64K terminal KV history does not cover its authoritative position");
        need(owner.close().reusable(),"native64K request did not retire");
        const auto stats=owner.runtime_stats();
        need(stats.exact_attention_score_bytes==16ULL*24*65536*4 && stats.numeric_attention_scratch_bytes==0 &&
            stats.committed_decode_tokens==stats.visible_model_tokens+
                stats.injected_control_tokens+stats.hidden_terminal_tokens,
            "native64K exact attention scratch envelope differs");
        if(stats.attention_stage_device_bytes)
            need(stats.attention_stage_device_bytes==65536ULL*4096 && stats.attention_stage_banks>0 &&
                stats.attention_stage_upload_bytes==stats.attention_stage_consumed_bytes+stats.attention_stage_direct_bytes,
                "native64K stage envelope or completion differs");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_RECON_RETIRE_FAILURE");mode && *mode) {
        need(std::string_view(mode)=="1","reconstruction retirement fixture requires1");
        opt.max_concurrency=1;opt.max_context=4352;opt.prefill_chunk=1024;
        opt.kv_capacity=KvCapacityPolicy::explicit_capacity(4352);
        {
            ninfer::exl3::Exl3EngineCore owner(opt);
            std::cout<<"RECON_ENGINE_GATE constructed"<<std::endl;
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=2;request.execution.allow_prefix_reuse=false;
            request.stop=owner.frontend().default_stop_policy();
            std::string context;
            for(int i=0;i<128;++i)context+="The record describes a river, a stone bridge, and a quiet village. ";
            auto prepared=owner.frontend().prepare(prompt(context+"Describe the setting."));
            const auto summary=prepared.summary();
            need(summary.prompt_tokens>=1024 && summary.prompt_tokens<=4096,"reconstruction retirement prompt extent");
            std::cout<<"RECON_ENGINE_GATE prepared prompt_tokens="<<summary.prompt_tokens<<std::endl;
            auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            std::cout<<"RECON_ENGINE_GATE submitted"<<std::endl;
            handle.wait(nullptr,{});
            std::cout<<"RECON_ENGINE_GATE request_complete"<<std::endl;
            owner.fail_reconstruction_retirement_for_test(); // idle and exercised slab required
            std::cout<<"RECON_ENGINE_GATE retirement_failure_armed"<<std::endl;
            const auto reconstruction=owner.runtime_stats();
            if(const char* head=std::getenv("NINFER_TEST_ENGINE_HEAD_ACCOUNTING")) {
                need(std::string(head)=="1","Engine head accounting fixture option");
                need(reconstruction.head_submitted_rows>0 && reconstruction.head_omitted_rows>0,
                    "Engine did not publish eager head submission/omission counters");
            }
            need(reconstruction.reconstruction_device_bytes>0 &&
                reconstruction.reconstruction_metadata_bytes==ninfer::exl3::Exl3TextContext::reconstruction_backing_metadata_bytes() &&
                reconstruction.reconstruction_budget_fallback_lanes==0 && reconstruction.reconstruction_submissions>0 &&
                reconstruction.reconstruction_submitted_rows>=256*reconstruction.reconstruction_submissions &&
                reconstruction.reconstruction_submitted_rows<=1024*reconstruction.reconstruction_submissions,
                "Engine reconstruction reservation/submission attribution missing");
            if(const char* accounting=std::getenv("NINFER_TEST_ENGINE_FUSED_ACCOUNTING")) {
                need(std::string(accounting)=="1","Engine fused accounting fixture option");
                need(reconstruction.fused_gate_up_submissions>0 && reconstruction.paired_transform_submissions>0,
                    "Engine missed actual paired/fused transform submissions");
            }
            const auto retired=owner.close();
            std::cout<<"RECON_ENGINE_GATE close_phase="<<static_cast<int>(retired.phase)<<std::endl;
            if(const char* residual=std::getenv("NINFER_TEST_ENGINE_RESIDUAL_ACCOUNTING")) {
                need(std::string(residual)=="1","Engine residual accounting fixture option");
                need(reconstruction.fused_residual_norm_submissions>0,
                    "Engine missed actual fused residual submissions");
            }
            need(retired.phase==ninfer::exl3::Exl3RetirementPhase::quarantined && !retired.reusable(),
                "Engine recycled unresolved reconstruction owners");
            need(owner.close().phase==retired.phase,"reconstruction close outcome was not sticky");
        }
        std::cout<<"RECON_ENGINE_GATE owner_destroyed"<<std::endl;
        bool refused=false;
        try{ninfer::exl3::Exl3EngineCore replacement(opt);}
        catch(const std::runtime_error& error){refused=std::string(error.what()).find("unresolved Engine retirement")!=std::string::npos;}
        need(refused,"reconstruction retirement admitted Engine reload");
        std::cout<<"RECON_ENGINE_GATE replacement_refused"<<std::endl;
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_ATTENTION_STAGING_FAILURE");mode && *mode) {
        need(mode[0]>='1' && mode[0]<='5' && mode[1]=='\0',"staging failure requires stage1..5");
        const unsigned stage=static_cast<unsigned>(mode[0]-'0');opt.max_concurrency=1;
        const auto* plane_option=std::getenv("NINFER_TEST_ENGINE_ATTENTION_STAGING_SECOND_PLANE");
        const bool second_plane=plane_option && std::string_view(plane_option)=="1";
        {
            ninfer::exl3::Exl3EngineCore owner(opt);
            owner.fail_attention_staging_for_test(stage+(second_plane?5:0));
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=24;request.execution.allow_prefix_reuse=false;
            request.stop=owner.frontend().default_stop_policy();
            std::string context;
            for(int i=0;i<12;++i)context+="The record describes a river, a stone bridge, and a quiet village. ";
            auto prepared=owner.frontend().prepare(prompt(context+"Describe the setting."));
            const auto summary=prepared.summary();
            auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            bool failed=false;
            try{handle.wait(nullptr,{});}catch(const std::runtime_error& error) {
                const std::string message(error.what());
                failed=message.find(stage==1?"attention stage history copy failed":
                    stage==2?"attention stage history record failed":stage==3?
                    "attention stage producer completion failed":"attention stage consumer completion failed")!=std::string::npos;
            }
            need(failed,"actual Engine staging failure NOT_EXERCISED");
            if(const auto* coalesced=std::getenv("NINFER_EXL3_ATTENTION_COALESCE_INPUT_MLP");
                coalesced && std::string_view(coalesced)=="1")
                need(owner.runtime_stats().coalesced_attention_layers==16,
                    "staging failure did not retain actual coalesced layer layout");
            const auto retired=owner.close(); // joins workers before reading final deltas
            need(!retired.reusable() && retired.phase==ninfer::exl3::Exl3RetirementPhase::quarantined,
                "uncertain staging command released Engine working cache/streams");
            const auto stats=owner.runtime_stats();
            const auto completed_planes=stats.attention_stage_consumed_bytes+stats.attention_stage_direct_bytes;
            need(stats.attention_stage_device_bytes>0 && stats.attention_stage_metadata_bytes>0 &&
                stats.attention_stage_banks==0 && completed_planes<=stats.attention_stage_upload_bytes,
                "failed staging lost reservation or credited incomplete consumer");
            if(!second_plane || stage<=2)need(completed_planes==0,"failed first consumer credited plane completion");
            if(second_plane && stage>=4)need(completed_planes>0 &&
                completed_planes*2==stats.attention_stage_upload_bytes,
                "second-plane failure lost first-plane completion or credited failed plane");
            if(stage>=3)need(stats.attention_stage_upload_bytes>0,"staging wait failure lost submitted uploads");
        }
        bool reload=false;
        try{ninfer::exl3::Exl3EngineCore replacement(opt);}
        catch(const std::runtime_error& error){reload=std::string(error.what()).find("unresolved Engine retirement")!=std::string::npos;}
        need(reload,"uncertain staging failure permitted reload");
        return 0; // One terminal failure per separately authorized future process.
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_PAGE_WEAK_COMPONENTS");mode && *mode) {
        using namespace ninfer::exl3;
        const std::string_view lanes(mode);need(lanes=="1" || lanes=="2","page weak component physical lanes1/2");
        opt.max_concurrency=lanes=="2"?2:1;
        Exl3EngineCore owner(opt);
        {
            std::string context;
            for(int i=0;i<12;++i)context+="The record describes a river, a stone bridge, and a quiet village. ";
            const auto submit=[&](const char* ending) {
                auto prepared=owner.frontend().prepare(prompt(context+ending));const auto summary=prepared.summary();
                runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
                request.execution.requested_output_tokens=24;request.execution.allow_prefix_reuse=false;
                request.stop=owner.frontend().default_stop_policy();
                return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(5));
            };
            auto first=submit("Describe the setting.");
            std::optional<Exl3EngineCore::Submission> second;
            if(opt.max_concurrency==2)second.emplace(submit("Describe the bridge."));
            need(!first.wait(nullptr,{}).generated_token_ids.empty(),"page weak fixture primary request produced no output");
            if(second)need(!second->wait(nullptr,{}).generated_token_ids.empty(),"page weak fixture peer produced no output");
        }
        auto components=owner.shared_page_weak_components_for_test();
        need(components.count && components.count%3==0,"Engine ready page components NOT_EXERCISED");
        need(owner.close().reusable(),"page weak component Engine close failed");
        for(std::size_t i=0;i<components.count;++i) {
            need(components.owners[i].expired(),"Engine close retained strong page component owner");
            const auto before=owner.retained_host_metadata_for_test();
            components.owners[i].reset();
            const auto after=owner.retained_host_metadata_for_test();
            need(before>=after && before-after==components.bytes[i],
                "Engine final weak component release changed wrong metadata extent");
        }
        std::cout<<"ENGINE_PAGE_WEAK_COMPONENTS_COMPLETE lanes="<<lanes<<" components="<<components.count<<'\n';return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_PAGE_CONSTRUCTOR");mode && *mode) {
        using namespace ninfer::exl3;using Storage=Exl3DevicePageStorage;
        const std::string_view selected(mode);
        need(selected=="retain" || selected=="release","Engine page constructor mode retain/release");
        const bool retain=selected=="retain";unsigned stage=1;opt.max_concurrency=1;
        std::string_view peer_mode="none";
        if(const auto* peer=std::getenv("NINFER_TEST_ENGINE_PAGE_CONSTRUCTOR_PEER"))peer_mode=peer;
        need(peer_mode=="none" || peer_mode=="queued" || peer_mode=="active","page constructor peer none/queued/active");
        const bool active_peer=peer_mode=="active",with_peer=peer_mode!="none";
        if(active_peer)opt.max_concurrency=2;
        if(const auto* raw=std::getenv("NINFER_TEST_DEVICE_PAGE_CONSTRUCTOR_STAGE")) {
            const std::string_view value(raw);need(value.size()==1 && value[0]>='1' && value[0]<='4',"Engine page constructor stage1..4");
            stage=value[0]-'0';
        }
        const std::array<std::string_view,4> errors{"injected device page post-allocation failure",
            "injected page fill after-state failure","injected page view after-counter failure","injected page fill after-view failure"};
        const auto before=Storage::quarantined_bytes();need(before==0,"Engine page constructor fixture requires fresh process");
        {
            Exl3EngineCore owner(opt);owner.fail_next_page_constructor_for_test(stage,retain);
            struct ConstructorPeerGate {
                std::mutex mutex;std::condition_variable changed;unsigned starts=0;
                bool submitted=false,released=false;
                void release(){std::lock_guard lock(mutex);released=true;changed.notify_all();}
            };
            auto gate=std::make_shared<ConstructorPeerGate>();
            std::optional<Exl3EngineCore::Submission> primary,peer;
            struct ReleaseConstructorPeer {
                std::shared_ptr<ConstructorPeerGate> gate;
                ~ReleaseConstructorPeer(){gate->release();}
            } release_peer{gate};
            if(with_peer)owner.observe_request_start_for_test([gate,active_peer] {
                std::unique_lock lock(gate->mutex);const auto ordinal=++gate->starts;gate->changed.notify_all();
                if(ordinal==1) {
                    if(!gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{
                        return gate->released || (active_peer?gate->starts>=2:gate->submitted);
                    }))throw std::runtime_error("constructor peer admission timed out");
                } else if(active_peer && !gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->released;}))
                    throw std::runtime_error("constructor active peer release timed out");
            });
            bool duplicate=false;try{owner.fail_next_page_constructor_for_test(stage,retain);}
            catch(const std::invalid_argument&){duplicate=true;}
            need(duplicate,"Engine page constructor replaced pending one-shot");
            std::string context;
            for(int i=0;i<12;++i)context+="The record describes a river, a stone bridge, and a quiet village. ";
            auto prepared=owner.frontend().prepare(prompt(context+"Describe the setting."));
            const auto summary=prepared.summary();runtime::ResolvedRequestOptions request;
            request.execution.sampling.temperature=0;request.execution.requested_output_tokens=24;
            request.execution.allow_prefix_reuse=false;request.stop=owner.frontend().default_stop_policy();
            primary.emplace(owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5)));
            if(with_peer) {
                {
                    std::unique_lock lock(gate->mutex);
                    need(gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->starts==1;}),
                        "constructor primary did not enter deterministic start gate");
                }
                auto pending=owner.frontend().prepare(prompt(context+"Describe the bridge."));const auto pending_summary=pending.summary();
                peer.emplace(owner.submit(std::move(pending),pending_summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(5)));
                std::lock_guard lock(gate->mutex);gate->submitted=true;gate->changed.notify_all();
            }
            bool original=false;
            try{primary->wait(nullptr,{});}
            catch(const std::runtime_error& error){original=std::string_view(error.what())==errors[stage-1];}
            gate->release();
            need(original,"Engine page constructor boundary NOT_EXERCISED or original error replaced");
            if(peer) {
                const auto cancelled=peer->wait(nullptr,{});
                need(cancelled.finish_reason==FinishReason::Cancelled && cancelled.generated_token_ids.empty() &&
                    cancelled.speculative.neural_input_rows==0,"constructor failure peer decoded or lost cancellation");
                const auto stats=owner.runtime_stats();
                need(stats.failure_cancelled_active_requests==(active_peer?1U:0U) &&
                    stats.failure_cancelled_queued_requests==(active_peer?0U:1U),"constructor peer cancellation attribution mismatch");
            }
            (void)owner.close();
        }
        if(retain) {
            const auto credits=Storage::retained_constructor_credits_for_test();
            need(Storage::quarantined_bytes()==before+Storage::bytes && credits[0]==Storage::bytes &&
                credits[1]==Storage::retirement_metadata_bytes(),"Engine page constructor lost exact survivor credits");
            bool refused=false;try{Exl3EngineCore reload(opt);}
            catch(const std::runtime_error& error){refused=std::string_view(error.what())=="EXL3 unresolved device page retirement; reload refused";}
            need(refused,"Engine reload bypassed page constructor quarantine");
        } else {
            need(Storage::quarantined_bytes()==before,"clean Engine page constructor cleanup quarantined storage");
            Exl3EngineCore reload(opt);need(reload.close().reusable(),"clean page constructor failure blocked fresh Engine");
        }
        std::cout<<"ENGINE_PAGE_CONSTRUCTOR_COMPLETE stage="<<stage<<" mode="<<selected<<" peer="<<peer_mode<<'\n';return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_PAGE_FAILURE");mode && *mode) {
        need(mode[0]>='1' && mode[0]<='8' && mode[1]=='\0',"shared page failure requires stage1..8");
        const unsigned stage=static_cast<unsigned>(mode[0]-'0');opt.max_concurrency=1;
        const auto* queued_mode=std::getenv("NINFER_TEST_ENGINE_PAGE_FAILURE_QUEUED");
        const bool queued_peer=queued_mode && std::string_view(queued_mode)=="1";
        const auto* active_mode=std::getenv("NINFER_TEST_ENGINE_PAGE_FAILURE_ACTIVE");
        const bool active_peer=active_mode && std::string_view(active_mode)=="1";
        need(!(queued_peer && active_peer),"choose queued or active page failure peer");
        if(active_peer)opt.max_concurrency=2;
        if(queued_peer && opt.max_pending_requests<2)opt.max_pending_requests=2;
        {
            unsigned fault_callbacks=0;
            std::optional<ninfer::exl3::Exl3EngineCore::Submission> queued;
            ninfer::exl3::Exl3EngineCore owner(opt);
            struct PeerGate {
                std::mutex mutex;std::condition_variable changed;
                unsigned starts=0;bool entered=false,released=false;
                void release(){std::lock_guard lock(mutex);released=true;changed.notify_all();}
            };
            auto gate=std::make_shared<PeerGate>();
            struct ReleasePeer {std::shared_ptr<PeerGate> gate;~ReleasePeer(){gate->release();}} release_peer{gate};
            if(active_peer)owner.observe_request_start_for_test([gate]{
                std::unique_lock lock(gate->mutex);
                if(++gate->starts==1)return;
                gate->entered=true;gate->changed.notify_all();
                if(!gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->released;}))
                    throw std::runtime_error("active page-failure peer release timed out");
            });
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=24;request.execution.allow_prefix_reuse=false;
            request.stop=owner.frontend().default_stop_policy();
            owner.fail_next_shared_page_for_test(stage,[&,request]{
                ++fault_callbacks;
                bool observer_busy=false;
                try{owner.observe_request_start_for_test({});}catch(const std::logic_error&){observer_busy=true;}
                need(observer_busy,"active fault callback replaced worker observer");
                if(!queued_peer && !active_peer)return;
                auto peer=owner.frontend().prepare(prompt("Name one primary color."));
                const auto summary=peer.summary();
                queued.emplace(owner.submit(std::move(peer),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(5)));
                if(active_peer) {
                    std::unique_lock lock(gate->mutex);
                    need(gate->changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate->entered;}),
                        "page failure active peer did not reach worker boundary");
                    return;
                }
                {
                    auto dropped_prompt=owner.frontend().prepare(prompt("Name one secondary color."));
                    const auto dropped_summary=dropped_prompt.summary();
                    auto dropped=owner.submit(std::move(dropped_prompt),dropped_summary,0,request,
                        OutputConsumerMode::Aggregate,std::chrono::steady_clock::now()+std::chrono::minutes(5));
                    // C1 is still inside this callback: destruction cancels the
                    // second queued request before the Engine sees its failure.
                }
                const auto admitted=owner.runtime_stats();
                need(admitted.waiting_requests==2 && admitted.running_requests==1 &&
                    admitted.failure_cancelled_queued_requests==0,
                    "fault-boundary admission snapshot omitted queued peers or anticipated failure");
            });
            bool duplicate=false;
            try{owner.fail_next_shared_page_for_test(stage==1?6:1);}catch(const std::logic_error&){duplicate=true;}
            need(duplicate,"second page failure arm replaced first fault");
            std::string context;
            for(int i=0;i<12;++i)context+="The record describes a river, a stone bridge, and a quiet village. ";
            auto prepared=owner.frontend().prepare(prompt(context+"Describe the setting."));
            const auto summary=prepared.summary();
            auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            bool failed=false;
            try{handle.wait(nullptr,{});}catch(const std::runtime_error& error) {
                failed=std::string(error.what()).find(stage<=3?"device page fill transfer/readiness failed":
                    stage<=6?"shared device page copy final use uncertain":"shared attention final use uncertain")!=std::string::npos;
            }
            gate->release();
            need(failed,"Engine page failure boundary NOT_EXERCISED");
            need(fault_callbacks==1,"page fault callback did not run exactly once at provider consumption");
            if(queued_peer || active_peer) {
                need(queued.has_value(),"page failure did not enqueue deterministic peer");
                const auto cancelled=queued->wait(nullptr,{});
                need(cancelled.finish_reason==FinishReason::Cancelled && cancelled.generated_token_ids.empty() &&
                    cancelled.speculative.neural_input_rows==0,
                    "queued page-failure peer performed decoding or lost cancellation");
            }
            const auto failure_boundary=owner.runtime_stats();
            need(failure_boundary.failure_cancelled_active_requests==(active_peer?1u:0u) &&
                failure_boundary.failure_cancelled_queued_requests==(queued_peer?1u:0u),
                "page failure queued cancellation attribution mismatch");
            // An uncertain page failure closes this Engine's admission boundary.
            // A later independent request must not reuse its retained lane/page.
            auto peer_prompt=owner.frontend().prepare(prompt("Name one primary color."));
            const auto peer_summary=peer_prompt.summary();
            bool peer_refused=false;
            try{auto peer=owner.submit(std::move(peer_prompt),peer_summary,0,request,
                OutputConsumerMode::Aggregate,std::chrono::steady_clock::now()+std::chrono::minutes(5));}
            catch(const RequestError& error){peer_refused=error.kind()==RequestErrorKind::Unavailable;}
            need(peer_refused,"failed page Engine admitted a later peer request");
            bool rearm_refused=false;
            try{owner.fail_next_shared_page_for_test(stage);}catch(const std::logic_error&){rearm_refused=true;}
            need(rearm_refused,"failed page Engine accepted a replacement fault arm");
            bool start_observer_refused=false,root_observer_refused=false;
            try{owner.observe_request_start_for_test({});}catch(const std::logic_error&){start_observer_refused=true;}
            try{owner.observe_terminal_roots_for_test({});}catch(const std::logic_error&){root_observer_refused=true;}
            need(start_observer_refused && root_observer_refused,
                "failed page Engine allowed observer lifetime mutation");
            const auto stats=owner.runtime_stats();
            need(stats.failure_cancelled_active_requests==(active_peer?1u:0u) && stats.failure_cancelled_queued_requests==(queued_peer?1u:0u),
                "rejected post-failure admission was counted as cancelled queued work");
            if(stage>=7)need(stats.shared_device_page_retained_readers>=1,
                "failed attention reader missing from Engine physical accounting");
            need(stats.shared_device_page_failures==1 && (stage>=7?stats.shared_device_page_attention_bytes==0:
                stats.shared_device_page_copy_bytes==0),
                "failed page command lost attribution or credited incomplete copy");
            need((stage<=3 && stats.shared_device_page_fill_bytes==0) ||
                (stage>3 && stage<=6 && stats.shared_device_page_fill_bytes==ninfer::exl3::Exl3DevicePageStorage::bytes) ||
                (stage>=7 && stats.shared_device_page_fill_bytes>=ninfer::exl3::Exl3DevicePageStorage::bytes),
                "page failure lost completed-only fill accounting");
            const auto retired=owner.close();
            need(!retired.reusable() && retired.phase==ninfer::exl3::Exl3RetirementPhase::quarantined,
                "uncertain page failure released Engine storage");
            const auto joined=owner.runtime_stats();
            need(joined.running_requests==0 && joined.waiting_requests==0 &&
                joined.failure_cancelled_queued_requests==(queued_peer?1u:0u),
                "page failure teardown retained queued work or recounted prior cancellation");
        }
        bool reload=false;
        try{ninfer::exl3::Exl3EngineCore replacement(opt);}
        catch(const std::runtime_error& error){reload=std::string(error.what()).find("unresolved Engine retirement")!=std::string::npos;}
        need(reload,"uncertain page failure allowed Engine reload");
        return 0; // separate future process for each deliberate quarantine stage
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_CONDITIONAL_B8");mode && std::string_view(mode)=="1") {
        const auto* saved=std::getenv("NINFER_EXL3_ENGINE_CONDITIONAL_B8");
        struct RestoreConditional {
            std::string value;
            ~RestoreConditional(){_putenv_s("NINFER_EXL3_ENGINE_CONDITIONAL_B8",value.c_str());}
        } restore{saved?saved:""};
        using Root=std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>;
        constexpr std::array<std::uint32_t,5> budgets{1,8,9,16,32};
        std::array<std::vector<TokenId>,5> baselines;
        std::array<Root,5> baseline_roots;
        std::array<FinishReason,5> baseline_reasons{};
        std::array<std::vector<TokenId>,4> stopped_baselines;
        std::array<Root,4> stopped_roots;
        std::vector<TokenId> control_baseline;
        Root control_root;
        std::uint64_t control_injected=0;
        FinishReason control_reason=FinishReason::None;
        for(bool enabled:{false,true}) {
            _putenv_s("NINFER_EXL3_ENGINE_CONDITIONAL_B8",enabled?"1":"0");
            const auto pending_blocks=ninfer::exl3::bounded_shared_live_blocks_for_test<ninfer::exl3::Exl3Dflash2Execution::Pending>();
            ninfer::exl3::Exl3EngineCore owner(opt);
            Root terminal;
            owner.observe_terminal_roots_for_test([&](auto root){terminal=std::move(root);});
            for(std::size_t index=0;index<budgets.size();++index) {
            terminal.reset();
            const auto before=owner.runtime_stats();
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=budgets[index];request.execution.allow_prefix_reuse=false;
            request.stop=owner.frontend().default_stop_policy();
            auto prepared=owner.frontend().prepare(prompt("Explain how a river flows from mountains to the sea."));
            const auto summary=prepared.summary();
            auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            const auto result=handle.wait(nullptr,{});
            need(terminal && !result.generated_token_ids.empty(),"conditional Engine missing terminal result");
            const auto counters=owner.runtime_stats();
            const auto second_calls=counters.conditional_second_block_calls-before.conditional_second_block_calls;
            need(result.generated_token_ids.size()<=budgets[index],"conditional Engine exceeded output budget");
            if(!enabled || budgets[index]<16)
                need(second_calls==0,"disabled or short-budget Engine executed second block");
            if(!enabled){baselines[index]=result.generated_token_ids;baseline_roots[index]=terminal;
                baseline_reasons[index]=result.finish_reason;}
            else {
                need(result.generated_token_ids==baselines[index] && result.finish_reason==baseline_reasons[index] &&
                    terminal->state()->same_payload(*baseline_roots[index]->state()) &&
                    terminal->same_projected_conditioning_for_test(*baseline_roots[index]),
                    "conditional Engine differs from independent B8 baseline");
                std::cout<<"ENGINE_CONDITIONAL budget="<<budgets[index]<<" second_calls="<<second_calls
                    <<" reached="<<(second_calls!=0)<<'\n';
            }
            }
            constexpr std::array<std::size_t,4> stop_rows{0,7,8,15};
            for(std::size_t index=0;index<stop_rows.size();++index) {
                const auto row=stop_rows[index];
                if(baselines.back().size()<=row) {
                    std::cout<<"ENGINE_CONDITIONAL_STOP row="<<row+1<<" reached=0 short_reference=1\n";
                    continue;
                }
                const auto stop_token=baselines.back()[row];
                const auto first=std::find(baselines.back().begin(),baselines.back().end(),stop_token);
                const auto first_row=static_cast<std::size_t>(first-baselines.back().begin());
                terminal.reset();const auto before=owner.runtime_stats();
                runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
                request.execution.requested_output_tokens=32;request.execution.allow_prefix_reuse=false;
                request.stop=owner.frontend().default_stop_policy();request.stop.token_ids.push_back(stop_token);
                auto prepared=owner.frontend().prepare(prompt("Explain how a river flows from mountains to the sea."));
                const auto summary=prepared.summary();
                auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(5));
                const auto result=handle.wait(nullptr,{});
                need(terminal && result.finish_reason==FinishReason::StopToken,
                    "conditional custom stop did not terminate request");
                const auto calls=owner.runtime_stats().conditional_second_block_calls-before.conditional_second_block_calls;
                if(!enabled || first_row<8)need(calls==0,"first-block stop allowed dependent neural work");
                if(!enabled){stopped_baselines[index]=result.generated_token_ids;stopped_roots[index]=terminal;}
                else need(result.generated_token_ids==stopped_baselines[index] && stopped_roots[index] &&
                    terminal->state()->same_payload(*stopped_roots[index]->state()) &&
                    terminal->same_projected_conditioning_for_test(*stopped_roots[index]),
                    "conditional stop changed publication state or conditioning");
                std::cout<<"ENGINE_CONDITIONAL_STOP requested_row="<<row+1<<" actual_first_row="<<first_row+1
                    <<" second_calls="<<calls<<'\n';
            }
            {
                terminal.reset();
                runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
                request.execution.requested_output_tokens=96;request.execution.allow_prefix_reuse=false;
                request.execution.thinking.budget=1;
                request.stop=owner.frontend().default_stop_policy();
                auto prepared=owner.frontend().prepare(prompt("Explain why two plus two equals four.",true));
                const auto summary=prepared.summary();
                auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(5));
                const auto result=handle.wait(nullptr,{});
                need(terminal && result.thinking.applied && result.thinking.injected_tokens>0 &&
                    result.thinking.model_thinking_tokens==1 &&
                    result.token_accounting.injected_control_tokens==result.thinking.injected_tokens &&
                    result.token_accounting.conserves_result_tokens(result.generated_token_ids.size()),
                    "conditional Engine control/accounting NOT_EXERCISED");
                if(!enabled) {
                    control_baseline=result.generated_token_ids;control_root=terminal;
                    control_injected=result.thinking.injected_tokens;control_reason=result.finish_reason;
                } else need(result.generated_token_ids==control_baseline && result.finish_reason==control_reason &&
                    result.thinking.injected_tokens==control_injected && control_root &&
                    terminal->state()->same_payload(*control_root->state()) &&
                    terminal->same_projected_conditioning_for_test(*control_root),
                    "conditional Engine control changed output, state or private conditioning");
            }
            Root acquired_root;
            std::array<bool,3> cancellation_reached{};
            std::array<std::uint64_t,3> cancellation_verifier{},cancellation_neural{};
            owner.observe_acquired_roots_for_test([&](auto root){acquired_root=std::move(root);});
            if(enabled && opt.max_concurrency==1)for(unsigned stage=1;stage<=3;++stage) {
                owner.cancel_conditional_stage_for_test(stage);
                const auto run_cancel_case=[&](std::uint32_t budget=32) {
                    terminal.reset();acquired_root.reset();
                    runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
                    request.execution.requested_output_tokens=budget;request.execution.allow_prefix_reuse=false;
                    request.stop=owner.frontend().default_stop_policy();
                    auto prepared=owner.frontend().prepare(prompt("Explain how a river flows from mountains to the sea."));
                    const auto summary=prepared.summary();
                    auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                        std::chrono::steady_clock::now()+std::chrono::minutes(5));
                    return handle.wait(nullptr,{});
                };
                const auto cancelled=run_cancel_case();
                const auto hit=owner.conditional_cancel_hit_for_test();
                if(hit) {
                    cancellation_reached[stage-1]=true;
                    cancellation_verifier[stage-1]=cancelled.speculative.verifier_calls;
                    cancellation_neural[stage-1]=cancelled.speculative.neural_input_rows;
                    need(cancelled.speculative.verifier_calls>0,
                        "conditional cancellation lost completed first verifier work");
                    need(hit==stage && cancelled.finish_reason==FinishReason::Cancelled && terminal &&
                        cancelled.generated_token_ids.size()<=baselines.back().size() &&
                        std::equal(cancelled.generated_token_ids.begin(),cancelled.generated_token_ids.end(),baselines.back().begin()),
                        "conditional cancellation published invalid prefix");
                }
                std::cout<<"ENGINE_CONDITIONAL_CANCEL stage="<<stage<<" reached="<<(hit!=0)<<'\n';
                const auto cancelled_root=terminal;
                if(hit && cancelled.generated_token_ids.empty())
                    need(cancelled_root && acquired_root &&
                        cancelled_root->state()->same_payload(*acquired_root->state()) &&
                        cancelled_root->same_projected_conditioning_for_test(*acquired_root),
                        "empty conditional cancellation changed acquired state or conditioning");
                owner.cancel_conditional_stage_for_test(0);
                if(hit && !cancelled.generated_token_ids.empty()) {
                    const auto replay=run_cancel_case(static_cast<std::uint32_t>(cancelled.generated_token_ids.size()));
                    need(terminal && cancelled_root && replay.generated_token_ids==cancelled.generated_token_ids &&
                        terminal->state()->same_payload(*cancelled_root->state()) &&
                        terminal->same_projected_conditioning_for_test(*cancelled_root),
                        "conditional cancellation retained unpublished state or draft conditioning");
                }
                std::cout<<"ENGINE_CONDITIONAL_CANCEL_STATE stage="<<stage
                    <<" prefix_replay_prepared="<<(hit && !cancelled.generated_token_ids.empty())<<'\n';
                const auto retry=run_cancel_case();
                need(terminal && retry.generated_token_ids==baselines.back() &&
                    terminal->state()->same_payload(*baseline_roots.back()->state()) &&
                    terminal->same_projected_conditioning_for_test(*baseline_roots.back()),
                    "conditional cancellation poisoned subsequent request");
            }
            if(cancellation_reached[0] && cancellation_reached[1])
                need(cancellation_verifier[1]==cancellation_verifier[0] &&
                    cancellation_neural[1]==cancellation_neural[0]+8,
                    "post-proposal cancellation misattributed neural or verifier work");
            if(cancellation_reached[1] && cancellation_reached[2])
                need(cancellation_verifier[2]>cancellation_verifier[1] &&
                    cancellation_neural[2]==cancellation_neural[1],
                    "post-verification cancellation lost second verifier work or fabricated neural work");
            owner.observe_acquired_roots_for_test({});
            need(owner.close().reusable(),"conditional Engine retained pending credit at close");
            need(ninfer::exl3::bounded_shared_live_blocks_for_test<ninfer::exl3::Exl3Dflash2Execution::Pending>()==pending_blocks,
                "Engine model-round Pending owners survived retirement");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SUFFIX_SELECTION");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3EngineCore owner(opt);
        const ninfer::exl3::Exl3SuffixProposer::SelectionLimits limits{1,2,7,0};
        const auto* enabled=std::getenv("NINFER_EXL3_SUFFIX_PROPOSALS");
        if(!enabled || std::string_view(enabled)!="1") {
            bool refused=false;try{owner.set_suffix_selection_limits(limits);}catch(const std::invalid_argument&){refused=true;}
            need(refused,"Engine suffix limits enabled inactive proposer");
            owner.set_suffix_selection_limits(std::nullopt);need(owner.close().reusable(),"suffix disabled retirement");return 0;
        }
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>> suffix_roots;
        unsigned suffix_busy_refusals=0;
        owner.observe_terminal_roots_for_test([&](auto root){
            suffix_roots.push_back(std::move(root));
            try{owner.set_suffix_selection_limits(std::nullopt);}
            catch(const std::logic_error&){++suffix_busy_refusals;}
        });
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=24;request.execution.allow_prefix_reuse=false;
        request.stop=owner.frontend().default_stop_policy();
        const auto run=[&] {
            auto prepared=owner.frontend().prepare(prompt("Repeat this phrase: the river flows past the bridge. Explain its meaning."));
            const auto summary=prepared.summary();
            auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            auto result=handle.wait(nullptr,{});owner.set_host_kv_routes_for_test(false,false);
            return result.generated_token_ids;
        };
        owner.set_host_kv_routes_for_test(false,false);const auto baseline=run();
        const auto before=owner.runtime_stats();
        for(unsigned invalid=0;invalid<4;++invalid) {
            auto bad=limits;
            if(invalid==0)bad.minimum_rounds=0;
            if(invalid==1)bad.retry_after_neural_publications=0;
            if(invalid==2)bad.minimum_useful_rows=8;
            if(invalid==3)bad.maximum_replay_rows=-1;
            bool refused=false;try{owner.set_suffix_selection_limits(bad);}catch(const std::invalid_argument&){refused=true;}
            need(refused,"Engine suffix invalid configuration admitted");
        }
        owner.set_suffix_selection_limits(limits);
        const auto selected=run();const auto after=owner.runtime_stats();
        need(after.suffix_selection_installations==before.suffix_selection_installations+1,
            "Engine suffix limits not installed on actual acquisition");
        owner.set_suffix_selection_limits(std::nullopt);const auto restored=run();
        need(owner.runtime_stats().suffix_selection_installations==after.suffix_selection_installations &&
            selected==baseline && restored==baseline,"Engine suffix policy token/reset parity");
        need(suffix_roots.size()==3 && suffix_busy_refusals==3 &&
            suffix_roots[0]->state()->same_payload(*suffix_roots[1]->state()) &&
            suffix_roots[0]->state()->same_payload(*suffix_roots[2]->state()),
            "Engine suffix full-state parity or busy setter isolation");
        std::cout<<"SUFFIX_SELECTION neural_fallbacks="<<after.suffix_selection_neural_fallbacks-before.suffix_selection_neural_fallbacks<<'\n';
        need(owner.close().reusable(),"Engine suffix policy retirement");
        bool closed_refused=false;try{owner.set_suffix_selection_limits(limits);}catch(const std::logic_error&){closed_refused=true;}
        need(closed_refused,"closed Engine accepted suffix limits");return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_COMPLETED_PREFIX_CANCEL");mode && std::string_view(mode)=="1") {
        opt.context_cache.enabled=true;opt.max_concurrency=1;
        std::mutex terminal_mutex;std::condition_variable terminal_changed;
        bool terminal_seen=false;unsigned polls=0;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>> states;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3DraftHostRing>> rings;
        const auto capture=[&](auto root) {
            states.push_back(root->state()->detached_payload_for_test());
            rings.push_back(root->detached_projected_conditioning_for_test());
        };
        ninfer::exl3::Exl3EngineCore owner(opt);
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=128;request.execution.allow_prefix_reuse=true;
        request.stop=owner.frontend().default_stop_policy();
        std::string input="Read these records: ";
        for(int i=0;i<12;++i)input+="The stone bridge crosses the river. ";
        input+="Reply with only the single word bridge, then end your response.";
        const auto submit=[&] {
            auto prepared=owner.frontend().prepare(prompt(input));const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        owner.observe_terminal_roots_for_test(capture);
        auto first=submit();const auto baseline=first.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        const auto before=owner.runtime_stats();
        need(baseline.finish_reason==FinishReason::StopToken && !baseline.generated_token_ids.empty() &&
            baseline.generated_token_ids.back()==248046 && before.completed_prefix_admissions==1,
            "completed-prefix cancellation control NOT_EXERCISED");
        need(baseline.token_accounting.hidden_terminal_tokens==1 &&
            baseline.token_accounting.injected_control_tokens==0 &&
            baseline.token_accounting.visible_model_tokens+1==baseline.generated_token_ids.size() &&
            baseline.token_accounting.diagnostic_state_rows==0 &&
            baseline.token_accounting.conserves_result_tokens(baseline.generated_token_ids.size()) &&
            before.committed_decode_tokens==before.visible_model_tokens+
                before.injected_control_tokens+before.hidden_terminal_tokens,
            "stopped answer conflated visible tokens with hidden terminal/state diagnostics");
        auto fixed_diagnostic=baseline.token_accounting;
        fixed_diagnostic.diagnostic_state_rows=17;
        need(fixed_diagnostic.visible_model_tokens==baseline.token_accounting.visible_model_tokens &&
            fixed_diagnostic.conserves_result_tokens(baseline.generated_token_ids.size()),
            "fixed-row state diagnostic changed semantic token conservation");
        need(owner.trim_prefix_retention(0,{}).budget_met,"completed cancellation fixture retained baseline cache");
        owner.observe_terminal_roots_for_test([&](auto root) {
            capture(root);
            std::unique_lock lock(terminal_mutex);terminal_seen=true;terminal_changed.notify_all();
            if(!terminal_changed.wait_for(lock,std::chrono::seconds(30),[&]{return polls>=2;}))
                throw std::runtime_error("late cancellation did not reach request waiter");
        });
        auto second=submit();
        const auto cancelled=second.wait(nullptr,CancellationView([&] {
            std::lock_guard lock(terminal_mutex);
            if(!terminal_seen)return false;
            ++polls;terminal_changed.notify_all();return true;
        }));
        owner.set_host_kv_routes_for_test(false,false);
        owner.observe_terminal_roots_for_test({});
        need(polls>=2 && owner.runtime_stats().completed_prefix_admissions==before.completed_prefix_admissions &&
            owner.prefix_retention_metadata().size()==1,
            "late-cancelled closed turn was admitted beyond its explicit input root");
        need(cancelled.generated_token_ids==baseline.generated_token_ids && states.size()==2 && rings.size()==2 &&
            states[0]->same_represented_payload_for_test(*states[1]) &&
            rings[0]->same_represented_payload_for_test(*rings[1]),
            "late admission cancellation changed already completed authoritative output");
        need(baseline.generated_token_ids.size()>1 && baseline.generated_token_ids.front()!=248046,
            "partial-stop control lacks a nonterminal generated token");
        owner.observe_terminal_roots_for_test(capture);
        for(bool custom_stop:{false,true}) {
            need(owner.trim_prefix_retention(0,{}).budget_met,"partial-stop fixture retained earlier roots");
            request.execution.requested_output_tokens=custom_stop?128:1;
            request.stop=owner.frontend().default_stop_policy();
            if(custom_stop)request.stop.token_ids.push_back(baseline.generated_token_ids.front());
            auto handle=submit();const auto partial=handle.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            need(partial.finish_reason==(custom_stop?FinishReason::StopToken:FinishReason::OutputLimit) &&
                partial.generated_token_ids==std::vector<TokenId>{baseline.generated_token_ids.front()},
                "partial-stop fixture did not terminate at the intended first token");
            need(partial.token_accounting.conserves_result_tokens(1) &&
                partial.token_accounting.diagnostic_state_rows==0 &&
                partial.token_accounting.hidden_terminal_tokens==(custom_stop?1u:0u) &&
                partial.token_accounting.visible_model_tokens==(custom_stop?0u:1u),
                "partial stop/output-limit semantic accounting differs");
            need(owner.runtime_stats().completed_prefix_admissions==before.completed_prefix_admissions &&
                owner.prefix_retention_metadata().size()==1,
                "partial assistant output was admitted as a completed turn");
        }
        owner.observe_terminal_roots_for_test({});
        need(states.size()==4 && rings.size()==4 && states[2]->same_represented_payload_for_test(*states[3]) &&
            rings[2]->same_represented_payload_for_test(*rings[3]),
            "output limit and custom stop changed the same accepted prefix state");
        need(owner.close().reusable(),"late cancellation fixture retirement");return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_DECLARED_PREFIX");mode && std::string_view(mode)=="1") {
        const auto* previous=std::getenv("NINFER_EXL3_ENGINE_DECLARED_PREFIX");
        struct Restore {std::string value;~Restore(){_putenv_s("NINFER_EXL3_ENGINE_DECLARED_PREFIX",value.c_str());}} restore{previous?previous:""};
        for(const char* invalid:{"2","true","-1","01"," 1"}) {
            _putenv_s("NINFER_EXL3_ENGINE_DECLARED_PREFIX",invalid);bool refused=false;
            try{ninfer::exl3::Exl3EngineCore invalid_owner(opt);}
            catch(const std::invalid_argument& error){refused=std::string_view(error.what())==
                "NINFER_EXL3_ENGINE_DECLARED_PREFIX must be 0 or 1";}
            need(refused,"declared-prefix option accepted malformed value");
        }
        _putenv_s("NINFER_EXL3_ENGINE_DECLARED_PREFIX","1");
        opt.context_cache.enabled=true;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>> states;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3DraftHostRing>> rings;
        ninfer::exl3::Exl3EngineCore owner(opt);
        owner.observe_terminal_roots_for_test([&](auto root){
            states.push_back(root->state()->detached_payload_for_test());
            rings.push_back(root->detached_projected_conditioning_for_test());
        });
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=8;request.execution.allow_prefix_reuse=true;
        request.stop=owner.frontend().default_stop_policy();
        std::string stable;
        for(int i=0;i<20;++i)stable+="Use the supplied bridge records exactly. Preserve names and measurements. ";
        for(bool tool_boundary:{false,true}) {
            owner.set_host_kv_routes_for_test(false,false);
            need(owner.trim_prefix_retention(0,{}).budget_met,"declared-prefix fixture failed to clear prior roots");
            states.clear();rings.clear();
            const auto make_input=[&](bool second,bool changed_boundary=false) {
                auto input=prompt(second?"List the bridge measurements.":"Summarize the bridge records.");
                input.messages.insert(input.messages.begin(),{ChatRole::System,{{MessagePartKind::Text,stable}}});
                if(tool_boundary) {
                    input.options.tool_jsons={
                        R"({"type":"function","function":{"name":"inspect_bridge","parameters":{"type":"object"}}})",
                        second?R"({"type":"function","function":{"name":"measure_bridge","parameters":{"type":"object"}}})":
                               R"({"type":"function","function":{"name":"list_bridges","parameters":{"type":"object"}}})"};
                    input.context_cache.markers.push_back({.kind=PromptCacheMarkerKind::SharedStablePrefix,
                        .location=PromptCacheMarkerLocation::ToolBoundary,.after_tool_count=1});
                } else input.context_cache.markers.push_back({.after_message_count=1,
                    .kind=PromptCacheMarkerKind::SharedStablePrefix});
                if(changed_boundary) {
                    if(tool_boundary)input.options.tool_jsons.front()=
                        R"({"type":"function","function":{"name":"archive_bridge","parameters":{"type":"object"}}})";
                    else input.messages.front().parts.front().text="Ignore obsolete records. "+stable;
                }
                return input;
            };
            const auto run=[&](bool second,std::size_t& frontier,bool changed_boundary=false) {
                auto prepared=owner.frontend().prepare(make_input(second,changed_boundary));
                const auto& data=ninfer::targets::qwen3_6::PreparedPromptAccess::view(prepared);
                frontier=ninfer::exl3::exl3_declared_prefix_frontier(data,data.token_ids.size());
                need(frontier>=64 && frontier<data.token_ids.size(),"declared fixture lacks eligible inner frontier");
                const auto summary=prepared.summary();
                auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                    std::chrono::steady_clock::now()+std::chrono::minutes(5));
                auto result=handle.wait(nullptr,{});owner.set_host_kv_routes_for_test(false,false);return result;
            };
            std::size_t first_frontier=0,second_frontier=0;
            {
                auto interrupted=make_input(false);
                if(tool_boundary)interrupted.options.tool_jsons.clear();
                else interrupted.messages.erase(interrupted.messages.begin());
                // The original message marker now refers to the user message,
                // so explicitly preserve its unavailable original second boundary.
                if(!tool_boundary)interrupted.context_cache.markers.front().after_message_count=2;
                bool refused=false;
                try{(void)owner.frontend().prepare(std::move(interrupted));}
                catch(const std::invalid_argument&){refused=true;}
                need(refused && owner.prefix_retention_metadata().empty(),
                    "interrupted input invented missing declared boundary or admitted a root");
            }
            const auto first=run(false,first_frontier),warm=run(true,second_frontier);
            need(first.reused_prompt_tokens==0 && first_frontier==second_frontier &&
                warm.reused_prompt_tokens==second_frontier,"different tail failed to reuse exact declared boundary");
            std::size_t repeated_frontier=0;
            const auto repeated=run(true,repeated_frontier);
            need(repeated_frontier==second_frontier &&
                repeated.reused_prompt_tokens>=repeated_frontier &&
                repeated.generated_token_ids==warm.generated_token_ids &&
                repeated.finish_reason==warm.finish_reason && states.size()==3 && rings.size()==3 &&
                states[1]->same_represented_payload_for_test(*states[2]) &&
                rings[1]->same_represented_payload_for_test(*rings[2]),
                "exact repeated declared turn changed frontier, target state or private conditioning");
            need(owner.trim_prefix_retention(0,{}).budget_met,"declared cold control retained cache roots");
            const auto cold=run(true,second_frontier);
            need(cold.reused_prompt_tokens==0 && cold.generated_token_ids==warm.generated_token_ids &&
                cold.finish_reason==warm.finish_reason && states.size()==4 && rings.size()==4 &&
                states[1]->same_represented_payload_for_test(*states[3]) &&
                rings[1]->same_represented_payload_for_test(*rings[3]),
                "declared boundary hit changed cold target state or private conditioning");
            const auto changed=run(true,second_frontier,true);
            need(changed.reused_prompt_tokens==0,
                "changed content before declared boundary reused prior tool/message root");
            need(owner.trim_prefix_retention(0,{}).budget_met,"changed-boundary cold control retained roots");
            const auto changed_cold=run(true,second_frontier,true);
            need(changed_cold.reused_prompt_tokens==0 &&
                changed.generated_token_ids==changed_cold.generated_token_ids &&
                changed.finish_reason==changed_cold.finish_reason && states.size()==6 && rings.size()==6 &&
                states[4]->same_represented_payload_for_test(*states[5]) &&
                rings[4]->same_represented_payload_for_test(*rings[5]),
                "changed-boundary miss differed from independent cold authority");
        }
        owner.observe_terminal_roots_for_test({});
        need(owner.close().reusable(),"declared-prefix fixture retirement");return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_PREPARATION_VERIFY_INTERLEAVE");
       mode && std::string_view(mode)=="1") {
        const auto* enabled=std::getenv("NINFER_EXL3_ENGINE_SHARED_PREPARATION");
        need(enabled && std::string_view(enabled)=="1",
            "preparation/verification interleave requires reserved concurrent preparation");
        const auto* cancel_option=std::getenv("NINFER_TEST_ENGINE_PREPARATION_VERIFY_CANCEL");
        const bool cancel=cancel_option && std::string_view(cancel_option)=="1";
        opt.max_concurrency=2;opt.context_cache.enabled=true;
        ninfer::exl3::Exl3EngineCore owner(opt);
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=8;request.execution.allow_prefix_reuse=true;
        request.stop=owner.frontend().default_stop_policy();
        std::string short_text="Use this stable bridge record and answer in one sentence: ";
        for(unsigned row=0;row<10;++row)
            short_text+="Pier A is granite, span B is steel, and inspection interval C is annual. ";
        std::string long_text="Prepare the complete independent archive before answering: ";
        for(unsigned row=0;row<48;++row)
            long_text+="Archive row "+std::to_string(row)+" preserves a distinct load, owner, and inspection note. ";
        const auto submit=[&](const std::string& text) {
            auto prepared=owner.frontend().prepare(prompt(text));const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        // Establish the short request's complete serialized authority before
        // installing a callback that intentionally holds only the long lane.
        const auto reference=submit(short_text).wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        const auto before=owner.runtime_stats();
        std::mutex gate_mutex;std::condition_variable gate;
        std::size_t preparation_lane=2;bool preparation_held=false,release_preparation=false;
        owner.observe_prefix_preparation_for_test([&](std::size_t lane,int position) {
            if(position<=0)return;
            std::unique_lock lock(gate_mutex);
            if(preparation_lane==2)preparation_lane=lane;
            if(lane!=preparation_lane)return;
            preparation_held=true;gate.notify_all();
            if(!gate.wait_for(lock,std::chrono::seconds(30),[&]{return release_preparation;}))
                throw std::runtime_error("preparation/verification release timeout");
        });
        auto preparation=submit(long_text);
        {
            std::unique_lock lock(gate_mutex);
            need(gate.wait_for(lock,std::chrono::seconds(30),[&]{return preparation_held;}),
                "large prefix preparation did not reach held numerical work");
        }
        if(cancel)need(preparation.poll(nullptr,CancellationView([]{return true;})).state==
                GenerationPollState::Pending,
            "held preparation cancellation consumed unfinished authority");
        // Completion here proves the other lane traversed its ordinary private
        // proposal, verification and publication while the large prefix worker
        // remained inside its preparation primitive.
        auto verification=submit(short_text);
        const auto verified=verification.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(verified.generated_token_ids==reference.generated_token_ids &&
            verified.finish_reason==reference.finish_reason &&
            preparation.poll(nullptr,{}).state==GenerationPollState::Pending,
            "small verification failed to preserve serialized authority during peer preparation");
        {
            std::lock_guard lock(gate_mutex);release_preparation=true;
        }
        gate.notify_all();
        const auto prepared=preparation.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        owner.observe_prefix_preparation_for_test({});
        const auto after=owner.runtime_stats();
        need(preparation_lane<2 &&
            after.concurrent_prefix_preparation_returns-before.concurrent_prefix_preparation_returns==
                (cancel?1U:2U),
            "preparation/verification route lost bounded preparation ownership accounting");
        if(cancel)need(prepared.finish_reason==FinishReason::Cancelled &&
                prepared.generated_token_ids.empty(),
            "cancelled held preparation exposed tentative output");
        else need(!prepared.generated_token_ids.empty(),
            "released large preparation failed to resume canonical request order");
        need(owner.close().reusable(),"preparation/verification interleave retirement");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_PREPARATION_EXTENSION");
       mode && std::string_view(mode)=="1") {
        const auto* enabled=std::getenv("NINFER_EXL3_ENGINE_SHARED_PREPARATION");
        need(enabled && std::string_view(enabled)=="1",
            "shared extension fixture requires concurrent prefix preparation");
        opt.max_concurrency=2;opt.context_cache.enabled=true;
        ninfer::exl3::Exl3EngineCore owner(opt);
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=0;request.execution.allow_prefix_reuse=true;
        request.stop=owner.frontend().default_stop_policy();
        std::mutex roots_mutex;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>> roots;
        owner.observe_terminal_roots_for_test([&](auto root) {
            std::lock_guard lock(roots_mutex);roots.push_back(std::move(root));
        });
        const auto submit_tokens=[&](const std::vector<TokenId>& tokens) {
            auto prepared=owner.frontend().prepare_tokens(tokens);const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        std::vector<TokenId> prefix(96,198);
        auto seed=submit_tokens(prefix).wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        need(seed.finish_reason==FinishReason::OutputLimit && seed.generated_token_ids.empty(),
            "shared extension seed changed zero-output result");
        const auto seeded_stats=owner.runtime_stats();
        {
            std::lock_guard lock(roots_mutex);roots.clear();
        }
        std::vector<TokenId> extended=prefix;
        for(unsigned i=0;i<64;++i)extended.push_back(i&1?13:12050);
        std::mutex gate_mutex;std::condition_variable gate;
        std::size_t producer_lane=2,waiter_lane=2;bool waiter_seen=false;
        owner.observe_prefix_preparation_for_test([&](std::size_t lane,int position) {
            std::unique_lock lock(gate_mutex);
            if(position==0 && producer_lane<2 && lane!=producer_lane) {
                waiter_seen=true;waiter_lane=lane;gate.notify_all();return;
            }
            producer_lane=lane;gate.notify_all();
            if(!gate.wait_for(lock,std::chrono::seconds(30),[&]{return waiter_seen;}))
                throw std::runtime_error("shared extension join timeout");
        });
        auto producer=submit_tokens(extended);
        {
            std::unique_lock lock(gate_mutex);
            need(gate.wait_for(lock,std::chrono::seconds(30),[&]{return producer_lane<2;}),
                "shared extension producer did not reach cached-root append");
        }
        auto waiter=submit_tokens(extended);
        const auto producer_result=producer.wait(nullptr,{});
        const auto waiter_result=waiter.wait(nullptr,{});
        owner.set_host_kv_routes_for_test(false,false);
        owner.observe_prefix_preparation_for_test({});
        const auto completed=owner.runtime_stats();
        const auto retention=owner.prefix_retention_metadata();
        need(waiter_seen && producer_lane<2 && waiter_lane<2 && producer_lane!=waiter_lane &&
            completed.prefix_preparation_returns-seeded_stats.prefix_preparation_returns==2 &&
            completed.concurrent_prefix_preparation_returns-seeded_stats.concurrent_prefix_preparation_returns==2 &&
            completed.prefix_preparation_joins-seeded_stats.prefix_preparation_joins==1 &&
            completed.prefix_preparation_cache_hits-seeded_stats.prefix_preparation_cache_hits==2,
            "cached-root extension did not execute one joined bounded preparation");
        need(producer_result.reused_prompt_tokens==prefix.size() &&
            waiter_result.reused_prompt_tokens==extended.size() &&
            producer_result.finish_reason==FinishReason::OutputLimit &&
            waiter_result.finish_reason==FinishReason::OutputLimit &&
            producer_result.generated_token_ids.empty() && waiter_result.generated_token_ids.empty(),
            "shared extension reuse attribution or zero-output result changed");
        const std::vector<std::int64_t> widened_extended(extended.begin(),extended.end());
        {
            std::lock_guard lock(roots_mutex);
            need(roots.size()==2 && roots[0]->matches_tokens(widened_extended) &&
                roots[1]->matches_tokens(widened_extended) &&
                roots[0]->state()->same_payload(*roots[1]->state()) &&
                roots[0]->same_projected_conditioning_for_test(*roots[1]),
                "shared extension changed exact target or private conditioning authority");
        }
        need(std::any_of(retention.begin(),retention.end(),[&](const auto& entry) {
                return entry.reusable_tokens==extended.size() &&
                    entry.root->matches_tokens(widened_extended);
            }),"shared extension did not admit the completed full-prefix authority");
        owner.observe_terminal_roots_for_test({});
        need(owner.close().reusable(),"shared extension fixture retirement");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_PREPARATION");mode && std::string_view(mode)=="1") {
        const auto* enabled=std::getenv("NINFER_EXL3_ENGINE_SHARED_PREPARATION");
        need(enabled && std::string_view(enabled)=="1","shared preparation fixture requires explicit option");
        const auto* cancel_option=std::getenv("NINFER_TEST_ENGINE_PREPARATION_CANCEL_PRODUCER");
        const bool cancel_producer=cancel_option && std::string_view(cancel_option)=="1";
        const auto* cancel_waiter_option=std::getenv("NINFER_TEST_ENGINE_PREPARATION_CANCEL_WAITER");
        const bool cancel_waiter=cancel_waiter_option && std::string_view(cancel_waiter_option)=="1";
        need(!(cancel_producer && cancel_waiter),"preparation cancellation variants are exclusive");
        const bool cancel_one=cancel_producer || cancel_waiter;
        const auto* failure_option=std::getenv("NINFER_TEST_ENGINE_PREPARATION_ADMISSION_FAILURE");
        const bool admission_failure=failure_option && std::string_view(failure_option)=="1";
        need(!admission_failure || !cancel_one,"preparation failure and cancellation variants are exclusive");
        opt.max_concurrency=2;opt.context_cache.enabled=true;
        std::mutex gate_mutex;std::condition_variable gate;
        bool waiter_seen=false;std::size_t producer_lane=2,waiter_lane=2;
        unsigned cancellation_polls=0;
        std::atomic<unsigned> busy_observer_refusals{0};
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>> roots;
        // Owner is destroyed before callback captures on every exception path.
        auto owner_storage=std::make_unique<ninfer::exl3::Exl3EngineCore>(opt);
        auto& owner=*owner_storage;
        const auto preparation_requirement=ninfer::exl3::Exl3VeriCacheServingPrefixCache::
            preparation_storage_requirement(opt.max_concurrency,opt.max_context);
        need(owner.runtime_stats().prefix_preparation_metadata_bytes==
            preparation_requirement.units[static_cast<unsigned>(ninfer::exl3::Exl3ResourceInventory::Domain::host_metadata)],
            "Engine preparation metadata reservation attribution");
        if(admission_failure)owner.fail_next_prefix_preparation_admission_for_test();
        owner.observe_prefix_preparation_for_test([&](std::size_t lane,int position) {
            bool refused=false;
            try{owner.observe_prefix_preparation_for_test({});}
            catch(const std::logic_error& error) {
                refused=std::string_view(error.what())==
                    "prefix preparation observer requires idle enabled live Engine";
            }
            if(!refused)throw std::runtime_error("busy Engine replaced preparation observer");
            ++busy_observer_refusals;
            std::unique_lock lock(gate_mutex);
            // Producers may report their initial position as zero. Once the
            // producer lane is established, zero from the other lane is the
            // joined waiter's explicit progress/cancellation poll.
            if(position==0 && producer_lane<2 && lane!=producer_lane) {
                waiter_seen=true;waiter_lane=lane;gate.notify_all();
                if(cancel_waiter && !gate.wait_for(lock,std::chrono::seconds(30),[&]{return cancellation_polls>=2;}))
                    throw std::runtime_error("Engine waiter cancellation poll timeout");
                return;
            }
            producer_lane=lane;gate.notify_all();
            if(!gate.wait_for(lock,std::chrono::seconds(30),[&]{
                return waiter_seen && (!cancel_one || cancellation_polls>=2);
            }))
                throw std::runtime_error("Engine shared preparation join timeout");
        });
        owner.observe_terminal_roots_for_test([&](auto root) {
            std::lock_guard lock(gate_mutex);roots.push_back(std::move(root));
        });
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=8;request.execution.allow_prefix_reuse=true;
        request.stop=owner.frontend().default_stop_policy();
        std::string text="Summarize the following bridge design notes: ";
        for(unsigned row=0;row<20;++row)
            text+="The foundation carries loads into stable ground. Inspect soil and drainage before construction. ";
        const auto submit=[&] {
            auto prepared=owner.frontend().prepare(prompt(text));const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        auto first=submit();
        {
            std::unique_lock lock(gate_mutex);
            need(gate.wait_for(lock,std::chrono::seconds(30),[&]{return producer_lane<2;}),
                "Engine first request did not enter producer progress");
        }
        auto second=submit();
        if(admission_failure) {
            bool first_failed=false,second_failed=false,second_cancelled=false;
            try{(void)first.wait(nullptr,{});}catch(const std::bad_alloc&){first_failed=true;}
            try {
                const auto peer=second.wait(nullptr,{});
                second_cancelled=peer.finish_reason==FinishReason::Cancelled && peer.generated_token_ids.empty();
            } catch(const std::bad_alloc&){second_failed=true;}
            // Engine failure propagation can cancel a peer while its first
            // progress callback is returning, before it observes flight failure.
            // Both outcomes forbid successful authority; T72 separately requires
            // direct shared exception fanout without Engine-wide cancellation.
            need(owner.close().reusable(),"Engine preparation admission failure retirement");
            const auto failed_stats=owner.runtime_stats();
            need(first_failed && (second_failed || second_cancelled) && waiter_seen && roots.empty() &&
                (!second_cancelled || failed_stats.failure_cancelled_active_requests>=1) &&
                failed_stats.prefix_preparation_returns==0 &&
                failed_stats.concurrent_prefix_preparation_returns==0 &&
                failed_stats.prefix_preparation_joins==0,
                "Engine preparation admission failure lost fanout or reported successful work");
            bool failed_observer_refused=false,failed_arm_refused=false;
            try{owner.observe_prefix_preparation_for_test({});}
            catch(const std::logic_error&){failed_observer_refused=true;}
            try{owner.fail_next_prefix_preparation_admission_for_test();}
            catch(const std::logic_error&){failed_arm_refused=true;}
            need(failed_observer_refused && failed_arm_refused,
                "retired failed Engine accepted preparation observer or fault");
            return 0;
        }
        const CancellationView cancellation([&] {
            std::lock_guard lock(gate_mutex);
            if(!cancel_one || !waiter_seen)return false;
            ++cancellation_polls;gate.notify_all();return true;
        });
        GenerationResult a,b;
        if(cancel_waiter) {b=second.wait(nullptr,cancellation);a=first.wait(nullptr,{});}
        else {a=first.wait(nullptr,cancellation);b=second.wait(nullptr,{});}
        const auto& survivor=cancel_waiter?a:b;
        owner.set_host_kv_routes_for_test(false,false);
        owner.observe_prefix_preparation_for_test({});
        owner.observe_terminal_roots_for_test({});
        const auto stats=owner.runtime_stats();
        // prepare_shared_text_prefix rejects recursive same-thread joins. Its
        // joined return/counter is therefore the authoritative distinct-worker
        // witness; progress position zero is not a stable physical-lane label.
        const bool shared_preparation_counts=waiter_seen && busy_observer_refusals.load()>=2 &&
            producer_lane<2 && waiter_lane<2 &&
            stats.prefix_preparation_returns==(cancel_one?1U:2U) &&
            stats.concurrent_prefix_preparation_returns==(cancel_one?1U:2U) &&
            stats.prefix_preparation_joins==(cancel_waiter?0U:1U) && stats.prefix_preparation_cache_hits==0;
        if(!shared_preparation_counts)std::cerr<<"SHARED_PREPARATION_COUNTS waiter="<<waiter_seen
            <<" observer_refusals="<<busy_observer_refusals.load()<<" producer_lane="<<producer_lane
            <<" waiter_lane="<<waiter_lane<<" returns="<<stats.prefix_preparation_returns
            <<" concurrent_returns="<<stats.concurrent_prefix_preparation_returns
            <<" joins="<<stats.prefix_preparation_joins<<" cache_hits="<<stats.prefix_preparation_cache_hits
            <<" cancel_producer="<<cancel_producer<<" cancel_waiter="<<cancel_waiter<<'\n';
        need(shared_preparation_counts,"Engine shared preparation did not execute exactly one joined fill");
        if(cancel_producer) {
            need(cancellation_polls>=2 && a.finish_reason==FinishReason::Cancelled &&
                a.generated_token_ids.empty() && !b.generated_token_ids.empty() && roots.size()==1,
                "Engine cancelled producer lost surviving preparation consumer");
        } else if(cancel_waiter) {
            need(cancellation_polls>=2 && b.finish_reason==FinishReason::Cancelled &&
                b.generated_token_ids.empty() && !a.generated_token_ids.empty() && roots.size()==1,
                "Engine cancelled waiter poisoned preparation producer");
        } else need(!a.generated_token_ids.empty() && a.generated_token_ids==b.generated_token_ids &&
            a.finish_reason==b.finish_reason && roots.size()==2 &&
            roots[0]->state()->same_payload(*roots[1]->state()) &&
            roots[0]->same_projected_conditioning_for_test(*roots[1]),
            "Engine shared preparation changed private target/draft authority");
        const auto reference_state=roots[0]->state()->detached_payload_for_test();
        const auto reference_conditioning=roots[0]->detached_projected_conditioning_for_test();
        need(reference_conditioning!=nullptr,"Engine shared preparation missing conditioning snapshot");
        need(owner.close().reusable(),"Engine shared preparation retirement");
        bool closed_observer_refused=false;
        try{owner.observe_prefix_preparation_for_test({});}
        catch(const std::logic_error& error) {
            closed_observer_refused=std::string_view(error.what())==
                "prefix preparation observer requires idle enabled live Engine";
        }
        need(closed_observer_refused,"closed Engine accepted preparation observer");
        roots.clear();owner_storage.reset();
        // Construct a distinct serialized Engine only after releasing the C2
        // reservation. Detached comparisons do not retain its model/workspace.
        struct RestorePreparationOption {
            ~RestorePreparationOption(){_putenv_s("NINFER_EXL3_ENGINE_SHARED_PREPARATION","1");}
        } restore_preparation;
        _putenv_s("NINFER_EXL3_ENGINE_SHARED_PREPARATION","0");
        std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> baseline_root;
        ninfer::exl3::Exl3EngineCore baseline(opt);
        bool disabled_observer_refused=false;
        try{baseline.observe_prefix_preparation_for_test([](std::size_t,int){});}
        catch(const std::logic_error& error) {
            disabled_observer_refused=std::string_view(error.what())==
                "prefix preparation observer requires idle enabled live Engine";
        }
        need(disabled_observer_refused,"disabled Engine accepted preparation observer");
        baseline.observe_terminal_roots_for_test([&](auto root){baseline_root=std::move(root);});
        auto baseline_request=request;baseline_request.stop=baseline.frontend().default_stop_policy();
        auto baseline_input=baseline.frontend().prepare(prompt(text));
        const auto baseline_summary=baseline_input.summary();
        auto baseline_handle=baseline.submit(std::move(baseline_input),baseline_summary,0,baseline_request,
            OutputConsumerMode::Aggregate,std::chrono::steady_clock::now()+std::chrono::minutes(5));
        const auto reference=baseline_handle.wait(nullptr,{});
        baseline.set_host_kv_routes_for_test(false,false);
        baseline.observe_terminal_roots_for_test({});
        const auto baseline_stats=baseline.runtime_stats();
        need(baseline_stats.prefix_preparation_metadata_bytes==0,
            "disabled Engine allocated shared preparation storage");
        need(baseline_root && reference.generated_token_ids==survivor.generated_token_ids &&
            reference.finish_reason==survivor.finish_reason &&
            baseline_root->state()->same_represented_payload_for_test(*reference_state) &&
            baseline_root->matches_detached_conditioning_for_test(*reference_conditioning),
            "Engine shared preparation differs from independent serialized authority");
        need(baseline_stats.prefix_preparation_returns==1 &&
            baseline_stats.concurrent_prefix_preparation_returns==0 &&
            baseline_stats.prefix_preparation_joins==0 && baseline_stats.prefix_preparation_cache_hits==0,
            "Engine serialized baseline did not execute independent cold preparation");
        need(baseline.close().reusable(),"Engine serialized preparation baseline retirement");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_PREFIX_RETENTION");mode && std::string_view(mode)=="1") {
        std::cout<<"PREFIX_RETENTION phase=malformed-options\n"<<std::flush;
        {
            const auto* original=std::getenv("NINFER_EXL3_ENGINE_SHARED_PREPARATION");
            struct RestoreSharedPreparation {
                std::string value;
                ~RestoreSharedPreparation(){_putenv_s("NINFER_EXL3_ENGINE_SHARED_PREPARATION",value.c_str());}
            } restore{original?original:""};
            for(const char* invalid:{"2","true","-1","01"," 1"}) {
                _putenv_s("NINFER_EXL3_ENGINE_SHARED_PREPARATION",invalid);
                bool refused=false;
                try{ninfer::exl3::Exl3EngineCore invalid_owner(opt);}
                catch(const std::invalid_argument& error) {
                    refused=std::string_view(error.what())==
                        "NINFER_EXL3_ENGINE_SHARED_PREPARATION must be 0 or 1";
                }
                need(refused,"Engine accepted malformed shared preparation option");
            }
        }
        {
            const auto* original=std::getenv("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT");
            const std::string saved=original?original:"";
            for(const char* invalid:{"2","true","-1","01"}) {
                _putenv_s("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT",invalid);
                bool refused=false;
                try{ninfer::exl3::Exl3EngineCore invalid_owner(opt);}
                catch(const std::invalid_argument& error){refused=std::string_view(error.what())==
                    "preserve acquired root must be0 or1";}
                catch(...) {_putenv_s("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT",saved.c_str());throw;}
                _putenv_s("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT",saved.c_str());
                need(refused,"Engine accepted malformed root-preservation option");
            }
        }
        std::cout<<"PREFIX_RETENTION phase=disabled-cache\n"<<std::flush;
        {
            auto disabled_options=opt;disabled_options.context_cache.enabled=false;
            ninfer::exl3::Exl3EngineCore disabled(disabled_options);
            bool refused=false;
            try{(void)disabled.admit_prefix_input_with_retention({}, {},0,{});}
            catch(const std::logic_error& error) {
                refused=std::string_view(error.what()).find("enabled context cache")!=std::string_view::npos;
            }
            need(refused && disabled.prefix_retention_metadata().empty(),
                "explicit admission bypassed disabled Engine cache");
            need(disabled.close().reusable(),"disabled retention fixture retirement");
        }
        std::cout<<"PREFIX_RETENTION phase=active-cache\n"<<std::flush;
        opt.context_cache.enabled=true;
        auto owner=std::make_unique<ninfer::exl3::Exl3EngineCore>(opt);
        need(owner->prefix_retention_metadata().empty(),"new Engine inherited retention metadata");
        const auto empty=owner->trim_prefix_retention(0,{});
        need(empty.inputs_current && empty.budget_met && empty.evicted==0,
            "empty Engine retention transaction failed");
        unsigned metadata_busy=0,trim_busy=0,admission_busy=0;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3ExactHostState>> terminal_states;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3DraftHostRing>> terminal_conditioning;
        owner->observe_terminal_roots_for_test([&](auto root){
            terminal_states.push_back(root->state()->detached_payload_for_test());
            terminal_conditioning.push_back(root->detached_projected_conditioning_for_test());
            try{(void)owner->prefix_retention_metadata();}catch(const std::logic_error&){++metadata_busy;}
            try{(void)owner->trim_prefix_retention(0,{});}catch(const std::logic_error&){++trim_busy;}
            try{(void)owner->admit_prefix_input_with_retention({}, {},0,{});}
            catch(const std::logic_error&){++admission_busy;}
        });
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=8;request.execution.allow_prefix_reuse=true;
        request.stop=owner->frontend().default_stop_policy();
        std::string retention_prompt="Summarize the following bridge design notes: ";
        for(unsigned line=0;line<20;++line)
            retention_prompt+="The foundation carries loads into stable ground. Inspect soil and drainage before construction. ";
        const auto run=[&] {
            auto prepared=owner->frontend().prepare(prompt(retention_prompt));
            const auto summary=prepared.summary();
            auto handle=owner->submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            auto result=handle.wait(nullptr,{});
            owner->set_host_kv_routes_for_test(false,false); // worker cleanup barrier
            return result;
        };
        const auto acquisition_before=owner->runtime_stats();
        const auto cold=run(),warm=run();
        const auto preparation_after=owner->runtime_stats();
        const auto* preparation_option=std::getenv("NINFER_EXL3_ENGINE_SHARED_PREPARATION");
        const bool concurrent_preparation=preparation_option && std::string_view(preparation_option)=="1";
        need(preparation_after.prefix_preparation_returns-acquisition_before.prefix_preparation_returns==2 &&
            preparation_after.concurrent_prefix_preparation_returns-acquisition_before.concurrent_prefix_preparation_returns==
                (concurrent_preparation?2U:0U) &&
            preparation_after.prefix_preparation_joins==acquisition_before.prefix_preparation_joins &&
            preparation_after.prefix_preparation_cache_hits-acquisition_before.prefix_preparation_cache_hits==1,
            "Engine cold/warm preparation attribution or option routing");
        const auto expected_computed=static_cast<std::uint64_t>(cold.prompt.prompt_tokens)+
            warm.prompt.prompt_tokens-warm.reused_prompt_tokens;
        need(cold.reused_prompt_tokens==0 && cold.prefix_reuse_path==PrefixReusePath::Root &&
            warm.prefix_reuse_path==PrefixReusePath::SharedStablePrefix &&
            preparation_after.computed_prefill_tokens-acquisition_before.computed_prefill_tokens==expected_computed &&
            preparation_after.reused_prompt_tokens-acquisition_before.reused_prompt_tokens==warm.reused_prompt_tokens &&
            preparation_after.root_selections-acquisition_before.root_selections==1 &&
            preparation_after.shared_stable_prefix_selections-
                acquisition_before.shared_stable_prefix_selections==1 &&
            preparation_after.last_selected_frontier_tokens==warm.reused_prompt_tokens,
            "Engine cold/warm selection or avoided-prefill token accounting diverged");
        const auto resident=owner->prefix_retention_metadata();
        for(const auto& entry:resident)
            std::cout<<"PREFIX_RETENTION entry generation="<<entry.generation
                <<" tokens="<<entry.reusable_tokens
                <<" prep_us="<<(entry.preparation_microseconds?std::to_string(*entry.preparation_microseconds):"null")
                <<" observed="<<(entry.observed_accesses?std::to_string(*entry.observed_accesses):"null")
                <<" selections="<<entry.lookup_selections<<'\n';
        need(!resident.empty() && warm.reused_prompt_tokens>=64,
            "Engine retention fixture did not populate and reuse actual cache");
        need(std::any_of(resident.begin(),resident.end(),[](const auto& entry) {
                return entry.preparation_microseconds.has_value() && entry.lookup_selections>0;
            }),"Engine compact-root renewal erased cold preparation or lookup observations");
        std::vector<ninfer::exl3::Exl3VeriCachePrefixIndex::RetentionDecision> decisions;
        for(const auto& entry:resident)decisions.push_back({entry.root,entry.generation,false,0});
        const auto dropped=owner->trim_prefix_retention(0,decisions);
        need(dropped.inputs_current && dropped.budget_met && dropped.evicted==resident.size() &&
            owner->prefix_retention_metadata().empty(),"Engine retention did not evict idle prefixes");
        std::cout<<"PREFIX_RETENTION phase=rebuild\n"<<std::flush;
        const auto rebuilt=run();
        const auto rebuilt_metadata=owner->prefix_retention_metadata();
        need(!rebuilt_metadata.empty(),"Engine regeneration did not republish cache roots");
        const auto stale_trim=owner->trim_prefix_retention(0,decisions);
        const auto after_stale=owner->prefix_retention_metadata();
        need(!stale_trim.inputs_current && stale_trim.evicted==0 &&
            after_stale.size()==rebuilt_metadata.size(),
            "Engine accepted retired retention decisions after regeneration");
        for(std::size_t i=0;i<after_stale.size();++i)
            need(after_stale[i].root==rebuilt_metadata[i].root &&
                after_stale[i].generation==rebuilt_metadata[i].generation &&
                after_stale[i].lookup_selections==rebuilt_metadata[i].lookup_selections,
                "stale Engine trim mutated regenerated entry");
        need(rebuilt.reused_prompt_tokens==0 && cold.generated_token_ids==warm.generated_token_ids &&
            cold.generated_token_ids==rebuilt.generated_token_ids && cold.finish_reason==rebuilt.finish_reason &&
            terminal_states.size()==3 && terminal_states[0]->same_represented_payload_for_test(*terminal_states[1]) &&
            terminal_states[0]->same_represented_payload_for_test(*terminal_states[2]),
            "Engine eviction/rebuild changed represented output or terminal state");
        need(terminal_conditioning.size()==3 && terminal_conditioning[0] && terminal_conditioning[1] &&
            terminal_conditioning[2] && terminal_conditioning[0]->same_represented_payload_for_test(*terminal_conditioning[1]) &&
            terminal_conditioning[0]->same_represented_payload_for_test(*terminal_conditioning[2]),
            "Engine eviction/rebuild changed private projected conditioning");
        const auto reference_conditioning=terminal_conditioning[0];
        need(static_cast<bool>(reference_conditioning),
            "preservation fixture failed to retain independent conditioning reference");
        const auto acquisition_after=owner->runtime_stats();
        const auto expected_rebuild_computed=expected_computed+rebuilt.prompt.prompt_tokens;
        need(rebuilt.prefix_reuse_path==PrefixReusePath::Root && rebuilt.reused_prompt_tokens==0 &&
            acquisition_after.computed_prefill_tokens-acquisition_before.computed_prefill_tokens==
                expected_rebuild_computed &&
            acquisition_after.reused_prompt_tokens-acquisition_before.reused_prompt_tokens==
                warm.reused_prompt_tokens &&
            acquisition_after.root_selections-acquisition_before.root_selections==2 &&
            acquisition_after.shared_stable_prefix_selections-
                acquisition_before.shared_stable_prefix_selections==1 &&
            acquisition_after.last_selected_frontier_tokens==0,
            "Engine eviction/rebuild economics counters invented reuse or lost computed work");
        const auto preserved=acquisition_after.acquired_payload_preservations-acquisition_before.acquired_payload_preservations;
        const auto cleared=acquisition_after.acquired_full_resets-acquisition_before.acquired_full_resets;
        const auto draft_preserved=acquisition_after.acquired_draft_ring_preservations-
            acquisition_before.acquired_draft_ring_preservations;
        const auto draft_restored=acquisition_after.acquired_draft_ring_restores-
            acquisition_before.acquired_draft_ring_restores;
        const auto* preserve_option=std::getenv("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT");
        const bool preservation_enabled=preserve_option && std::string_view(preserve_option)=="1";
        need(preserved+cleared==3 && draft_preserved+draft_restored==3 &&
            (preservation_enabled?preserved>0 && draft_preserved>0:preserved==0 && draft_preserved==0),
            "Engine acquisition counters missed target/draft work or failed option isolation");
        need(!rebuilt.generated_token_ids.empty() && metadata_busy==3 && trim_busy==3 && admission_busy==3,
            "Engine retention API accepted outstanding request or omitted observer");
        bool null_authority_refused=false;
        try{(void)owner->admit_prefix_input_with_retention({}, {},0,{});}
        catch(const std::invalid_argument&){null_authority_refused=true;}
        need(null_authority_refused,"Engine retention admitted missing model authority");
        const auto admission_root=rebuilt_metadata.front().root;
        const auto admission_input=admission_root->token_suffix();
        std::vector<ninfer::exl3::Exl3VeriCachePrefixIndex::RetentionDecision> admission_decisions;
        for(const auto& entry:rebuilt_metadata)
            admission_decisions.push_back({entry.root,entry.generation,false,1});
        auto mismatched_input=admission_input;mismatched_input.pop_back();
        bool mismatch_refused=false;
        // Synthetic supplied budget for policy plumbing, not an availability measurement.
        constexpr std::uint64_t supplied_available=std::numeric_limits<std::uint64_t>::max();
        try{(void)owner->admit_prefix_input_with_retention(admission_root,mismatched_input,
            supplied_available,admission_decisions);}
        catch(const std::invalid_argument&){mismatch_refused=true;}
        need(mismatch_refused,"Engine explicit retention admission accepted token mismatch");
        const auto admitted=owner->admit_prefix_input_with_retention(admission_root,admission_input,
            supplied_available,admission_decisions);
        need(admitted.admitted && admitted.replaced && admitted.generation!=rebuilt_metadata.front().generation,
            "Engine explicit retention admission did not renew exact input authority");
        const auto stale_admission=owner->admit_prefix_input_with_retention(admission_root,admission_input,
            supplied_available,admission_decisions);
        need(!stale_admission.admitted,"Engine explicit admission accepted stale generation decisions");
        std::cout<<"PREFIX_RETENTION phase=close-first\n"<<std::flush;
        owner->observe_terminal_roots_for_test({});
        need(owner->close().reusable(),"Engine retention fixture retirement");
        bool metadata_closed=false,trim_closed=false,admission_closed=false;
        try{(void)owner->prefix_retention_metadata();}catch(const std::logic_error&){metadata_closed=true;}
        try{(void)owner->trim_prefix_retention(0,{});}catch(const std::logic_error&){trim_closed=true;}
        try{(void)owner->admit_prefix_input_with_retention({}, {},0,{});}
        catch(const std::logic_error&){admission_closed=true;}
        need(metadata_closed && trim_closed && admission_closed,"closed Engine accepted retention API");
        owner.reset(); // release the first Engine reservation before constructing its replacement
        std::cout<<"PREFIX_RETENTION phase=replacement\n"<<std::flush;
        {
            const auto* prior_option=std::getenv("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT");
            struct RestorePreservationOption {
                std::string value;
                ~RestorePreservationOption(){_putenv_s("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT",value.c_str());}
            } restore_option{prior_option?prior_option:""};
            _putenv_s("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT",preservation_enabled?"0":"1");
            ninfer::exl3::Exl3EngineCore replacement(opt);
            bool foreign_refused=false;
            try{(void)replacement.admit_prefix_input_with_retention(
                admission_root,admission_input,supplied_available,{});}
            catch(const std::invalid_argument& error) {
                foreign_refused=std::string_view(error.what()).find("current Engine model authority")!=std::string_view::npos;
            }
            need(foreign_refused && replacement.prefix_retention_metadata().empty() &&
                admission_root->matches_tokens(admission_input),
                "replacement Engine accepted foreign model or invalidated retained input");
            std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest> opposite_root;
            replacement.observe_terminal_roots_for_test([&](auto root){
                need(!opposite_root,"opposite preservation fixture observed duplicate terminal root");
                opposite_root=std::move(root);
            });
            auto opposite_input=replacement.frontend().prepare(prompt(retention_prompt));
            const auto opposite_summary=opposite_input.summary();
            auto opposite_request=request;opposite_request.stop=replacement.frontend().default_stop_policy();
            auto opposite_handle=replacement.submit(std::move(opposite_input),opposite_summary,0,
                opposite_request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            const auto opposite=opposite_handle.wait(nullptr,{});
            replacement.set_host_kv_routes_for_test(false,false);
            const auto opposite_stats=replacement.runtime_stats();
            need(opposite_root && opposite.generated_token_ids==cold.generated_token_ids &&
                opposite.finish_reason==cold.finish_reason &&
                opposite_root->state()->same_represented_payload_for_test(*terminal_states[0]),
                "Engine preservation on/off changed tokens or represented target state");
            need(opposite_root->matches_detached_conditioning_for_test(*reference_conditioning),
                "Engine preservation on/off changed represented draft conditioning");
            need(opposite_stats.acquired_payload_preservations+opposite_stats.acquired_full_resets==1 &&
                opposite_stats.acquired_draft_ring_preservations+opposite_stats.acquired_draft_ring_restores==1 &&
                (preservation_enabled?opposite_stats.acquired_payload_preservations==0:
                    opposite_stats.acquired_payload_preservations==1) &&
                (preservation_enabled?opposite_stats.acquired_draft_ring_preservations==0:
                    opposite_stats.acquired_draft_ring_preservations==1),
                "opposite Engine preservation route not exercised");
            replacement.observe_terminal_roots_for_test({});
            need(replacement.close().reusable(),"foreign retention refusal prevented replacement retirement");
        }
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_VERIFIER_COST_MENU");mode && std::string_view(mode)=="1") {
        ninfer::exl3::Exl3EngineCore owner(opt);
        ninfer::exl3::Exl3VerifierHorizonPolicy::CostMenu menu;
        menu.minimum_samples=1;menu.fixed_neural_ms=1;
        menu.estimates={decltype(menu)::Estimate{2,2,1,true},decltype(menu)::Estimate{4,8,1,true},decltype(menu)::Estimate{8,32,1,true}};
        const auto* enabled=std::getenv("NINFER_EXL3_BOUNDED_VERIFIER_HORIZON");
        if(!enabled || std::string_view(enabled)!="1") {
            bool refused=false;try{owner.set_verifier_cost_menu(menu);}catch(const std::invalid_argument&){refused=true;}
            need(refused,"Engine cost menu enabled inactive horizon route");
            owner.set_verifier_cost_menu(std::nullopt);need(owner.close().reusable(),"cost menu disabled retirement");return 0;
        }
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>> roots;
        unsigned busy_refusals=0;
        owner.observe_terminal_roots_for_test([&](auto root){
            roots.push_back(std::move(root));
            try{owner.set_verifier_cost_menu(std::nullopt);}
            catch(const std::logic_error&){++busy_refusals;}
        });
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=16;request.execution.allow_prefix_reuse=false;
        request.stop=owner.frontend().default_stop_policy();
        const auto run=[&] {
            auto prepared=owner.frontend().prepare(prompt("Explain why a bridge needs strong foundations."));
            const auto summary=prepared.summary();
            auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            auto result=handle.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false); // wait through worker cleanup
            return result.generated_token_ids;
        };
        owner.set_host_kv_routes_for_test(false,false);
        const auto baseline=run();const auto baseline_stats=owner.runtime_stats();
        auto invalid_menu=menu;invalid_menu.minimum_samples=0;
        bool invalid_refused=false;try{owner.set_verifier_cost_menu(invalid_menu);}
        catch(const std::invalid_argument&){invalid_refused=true;}
        need(invalid_refused,"Engine invalid supplied menu admitted");
        owner.set_verifier_cost_menu(menu);const auto candidate=run();const auto menu_stats=owner.runtime_stats();
        need(menu_stats.verifier_cost_menu_installations==baseline_stats.verifier_cost_menu_installations+1 &&
            menu_stats.supplied_verifier_horizon_decisions[2]>baseline_stats.supplied_verifier_horizon_decisions[2],
            "Engine supplied menu not installed or selected in actual proposals");
        for(unsigned row=3;row<9;++row)need(menu_stats.supplied_verifier_horizon_decisions[row]==baseline_stats.supplied_verifier_horizon_decisions[row],
            "Engine supplied two-row menu selected larger horizon");
        owner.set_verifier_cost_menu(std::nullopt);const auto restored=run();const auto restored_stats=owner.runtime_stats();
        need(restored_stats.verifier_cost_menu_installations==menu_stats.verifier_cost_menu_installations &&
            restored_stats.supplied_verifier_horizon_decisions==menu_stats.supplied_verifier_horizon_decisions && busy_refusals==3,
            "cleared menu remained active or busy mutation was admitted");
        need(baseline==candidate && baseline==restored && roots.size()==3,"Engine supplied horizon token/reset parity");
        need(roots[0]->state()->same_payload(*roots[1]->state()) && roots[0]->state()->same_payload(*roots[2]->state()),
            "Engine supplied horizon full state parity");
        need(owner.close().reusable(),"Engine supplied horizon retirement");return 0;
    }
    const auto* page_parity_mode=std::getenv("NINFER_TEST_ENGINE_SHARED_PAGES_PARITY");
    const bool page_parity=page_parity_mode && std::string(page_parity_mode)=="1";
    const auto* staging_mode=std::getenv("NINFER_TEST_ENGINE_ATTENTION_STAGING_PARITY");
    const bool staging_parity=staging_mode && std::string_view(staging_mode)=="1";
    const auto* staging_turnover_option=std::getenv("NINFER_TEST_ENGINE_ATTENTION_STAGING_TURNOVER");
    const bool staging_turnover=staging_turnover_option && std::string_view(staging_turnover_option)=="1";
    need(!staging_turnover || staging_parity,"staging turnover requires staging parity");
    need(!(staging_parity && page_parity),"staging/page parity scenarios are separate");
    if(const auto* prefix=std::getenv("NINFER_TEST_ENGINE_REGISTERED_PREFIX_TURNOVER");
        prefix && std::string_view(prefix)=="1") {
        const auto* registered=std::getenv("NINFER_TEST_ENGINE_REGISTERED_KV_PARITY");
        need(!staging_parity && !page_parity && registered && std::string_view(registered)=="1",
            "prefix turnover requires registered-upload parity route");
    }
    const auto* page_c2_option=std::getenv("NINFER_TEST_ENGINE_SHARED_PAGES_C2");
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_REGISTERED_PRESSURE");mode && std::string_view(mode)=="1") {
        using namespace ninfer::exl3;
        need(opt.max_concurrency==1,"registered pressure fixture requires C1");
        Exl3EngineCore owner(opt);
        std::vector<std::shared_ptr<const Exl3ExactHostState>> states;
        std::vector<std::shared_ptr<const Exl3DraftHostRing>> rings;
        owner.observe_terminal_roots_for_test([&](auto root) {
            states.push_back(root->state()->detached_payload_for_test());
            rings.push_back(root->detached_projected_conditioning_for_test());
        });
        std::string text;
        for(unsigned i=0;i<64;++i)text+="A river passes the stone bridge beside a quiet village. ";
        text+="Summarize this setting in one sentence.";
        const auto run=[&] {
            auto prepared=owner.frontend().prepare(prompt(text));const auto summary=prepared.summary();
            runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=24;request.execution.allow_prefix_reuse=false;
            request.stop=owner.frontend().default_stop_policy();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5)).wait(nullptr,{});
        };
        owner.set_host_kv_routes_for_test(false,false);
        const auto baseline=run();
        owner.set_host_kv_routes_for_test(true,false); // wait for worker accounting
        const auto before=owner.runtime_stats();
        owner.hold_registration_reader_for_test(true);
        const auto protected_result=run();
        owner.set_host_kv_routes_for_test(true,false);
        const auto protected_stats=owner.runtime_stats();
        need(protected_result.generated_token_ids==baseline.generated_token_ids &&
            protected_stats.registered_kv_external_readers==1 &&
            protected_stats.registered_kv_retired_registrations==before.registered_kv_retired_registrations &&
            protected_stats.registered_kv_upload_bytes>before.registered_kv_upload_bytes &&
            protected_stats.registered_kv_upload_fallbacks>before.registered_kv_upload_fallbacks &&
            protected_stats.registered_kv_upload_failures==before.registered_kv_upload_failures,
            "protected Engine pressure failed parity or registration/fallback coverage NOT_EXERCISED");
        owner.hold_registration_reader_for_test(false);
        const auto recovered=run();
        owner.set_host_kv_routes_for_test(false,false);
        const auto after=owner.runtime_stats();
        need(recovered.generated_token_ids==baseline.generated_token_ids &&
            after.registered_kv_external_readers==0 &&
            after.registered_kv_retired_registrations>protected_stats.registered_kv_retired_registrations &&
            after.registered_kv_upload_bytes>protected_stats.registered_kv_upload_bytes &&
            after.registered_kv_upload_failures==before.registered_kv_upload_failures,
            "released Engine pressure guard did not restore registered turnover NOT_EXERCISED");
        need(states.size()==3 && rings.size()==3,"registered pressure terminal state capture");
        for(unsigned i=1;i<3;++i)need(states[i]->same_represented_payload_for_test(*states[0]) &&
            rings[i] && rings[0] && rings[i]->same_represented_payload_for_test(*rings[0]),
            "registered pressure changed canonical state or private draft conditioning");
        owner.observe_terminal_roots_for_test({});states.clear();rings.clear();
        need(owner.close().reusable(),"registered pressure retained uncertain Engine ownership");
        std::cout<<"ENGINE_REGISTERED_PRESSURE_COMPLETE\n";return 0;
    }
    const bool page_c2=page_parity && page_c2_option && std::string(page_c2_option)=="1";
    const auto* staging_c2_option=std::getenv("NINFER_TEST_ENGINE_ATTENTION_STAGING_C2");
    const bool staging_c2=staging_parity && staging_c2_option && std::string_view(staging_c2_option)=="1";
    if(page_c2 || staging_c2)opt.max_concurrency=2;
    const auto* page_root_eviction_option=std::getenv(
        "NINFER_TEST_ENGINE_SHARED_PAGES_ROOT_EVICTION");
    const bool page_root_eviction=page_root_eviction_option &&
        std::string_view(page_root_eviction_option)=="1";
    need(!page_root_eviction || page_c2,
        "shared-page root eviction requires shared-page C2 parity");
    if(page_root_eviction) {
        opt.context_cache.enabled=true;
        opt.context_cache.max_shared_prefixes=1;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_REGISTERED_KV_PARITY");staging_parity || page_parity || (mode && std::string(mode)=="1")) {
        const auto* paired_option=std::getenv("NINFER_TEST_ENGINE_REGISTERED_KV_C2");
        const bool registered_c2=paired_option && std::string_view(paired_option)=="1";
        need(!registered_c2 || (!staging_parity && !page_parity),"registered C2 requires registered-upload parity route");
        if(registered_c2)opt.max_concurrency=2;
        ninfer::exl3::Exl3EngineCore owner(opt);
        std::mutex captured_mutex;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>> captured;
        owner.observe_terminal_roots_for_test([&](auto root) {
            std::lock_guard lock(captured_mutex);captured.push_back(std::move(root));
        });
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=24;
        request.execution.allow_prefix_reuse=page_root_eviction;
        request.stop=owner.frontend().default_stop_policy();
        std::string context;
        for(int i=0;i<32;++i)context+="The record describes a river, a stone bridge, and a quiet village. ";
        const std::array<std::string,2> prompts{context+"Summarize the setting in one sentence.",
            context+"Write a Python function returning the number of bridges, which is one."};
        const auto submit=[&](unsigned index) {
            auto prepared=owner.frontend().prepare(prompt(prompts[index]));const auto summary=prepared.summary();
            return owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        owner.set_host_kv_routes_for_test(false,false);
        if(staging_parity)owner.set_attention_staging_for_test(false);
        std::array<std::vector<TokenId>,2> expected;
        for(unsigned i=0;i<2;++i){auto handle=submit(i);expected[i]=handle.wait(nullptr,{}).generated_token_ids;}
        owner.set_host_kv_routes_for_test(false,false); // counters publish during worker cleanup
        const auto staged=owner.runtime_stats();
        need(staged.registered_kv_upload_bytes==0,"staged parity control used registered copies");
        need(staged.shared_device_page_copy_bytes==0 && staged.shared_device_page_fill_bytes==0 &&
            staged.shared_device_page_attention_bytes==0,
            "staged parity control used shared device pages");
        owner.set_host_kv_routes_for_test(staging_turnover || (!page_parity&&!staging_parity),page_parity);
        if(staging_parity)owner.set_attention_staging_for_test(true);
        const auto* page_reader_overlap_option=std::getenv("NINFER_TEST_ENGINE_SHARED_PAGES_READER_OVERLAP");
        const bool page_reader_overlap=page_reader_overlap_option && std::string_view(page_reader_overlap_option)=="1";
        need(!page_reader_overlap || page_c2,"registration reader overlap requires shared-page C2 parity");
        need(!page_root_eviction || page_reader_overlap,
            "shared-page root eviction requires the deterministic fill hold");
        if(page_reader_overlap) {
            struct FillGate {
                std::mutex mutex;std::condition_variable changed;bool entered=false,released=false;
                ~FillGate(){std::lock_guard lock(mutex);released=true;changed.notify_all();}
            } gate;
            owner.observe_next_shared_page_fill_for_test([&] {
                std::unique_lock lock(gate.mutex);gate.entered=true;gate.changed.notify_all();
                need(gate.changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate.released;}),
                    "held page fill did not receive release");
            });
            auto first=submit(0);
            {
                std::unique_lock lock(gate.mutex);
                need(gate.changed.wait_for(lock,std::chrono::seconds(30),[&]{return gate.entered;}),
                    "shared page fill reader did not enter hold boundary");
            }
            auto second=submit(1);
            const auto second_result=second.wait(nullptr,{});
            need(second_result.generated_token_ids==expected[1],
                "peer fallback during retained registration read changed output");
            const auto overlap=owner.runtime_stats();
            need(overlap.registered_kv_external_reader_high_water>=2,
                "C2 shared-page route did not overlap independent registration readers");
            if(page_root_eviction)need(overlap.shared_device_page_source_pages>0 &&
                overlap.shared_device_page_source_host_bytes>0,
                "peer cache replacement dropped delayed shared-page fill source");
            {std::lock_guard lock(gate.mutex);gate.released=true;gate.changed.notify_all();}
            need(first.wait(nullptr,{}).generated_token_ids==expected[0],
                "held registration reader changed original lane output");
        } else for(unsigned i=0;i<2;++i) {
            auto handle=submit(i);
            need(handle.wait(nullptr,{}).generated_token_ids==expected[i],"registered Engine tokens differ from staging");
        }
        owner.set_host_kv_routes_for_test(false,false); // wait for final worker cleanup
        const auto registered=owner.runtime_stats();
        need(registered.registered_kv_external_readers==0,
            "completed history parity retained external registration readers");
        if(page_reader_overlap)need(registered.registered_kv_external_reader_high_water>=2,
            "drained C2 route lost overlapping registration-reader evidence");
        const auto* query_expect=std::getenv("NINFER_TEST_ENGINE_QUERY_PAIR_EXPECT");
        const auto* query_option=std::getenv("NINFER_EXL3_EXACT_ATTENTION_QUERY_PAIR");
        const bool query_opted_in=query_option && std::string_view(query_option)=="1";
        need(staged.query_pair_requested_row_attempts>=staged.query_pair_row_attempts &&
            registered.query_pair_requested_row_attempts>=registered.query_pair_row_attempts &&
            registered.query_pair_requested_row_attempts-staged.query_pair_requested_row_attempts>=
                registered.query_pair_row_attempts-staged.query_pair_row_attempts,
            "query-pair selected rows exceed opted-in dispatch rows");
        if(query_expect) {
            const std::string_view expected_route(query_expect);
            need(expected_route=="active" || expected_route=="inactive","invalid query-pair route expectation");
            if(expected_route=="active") {
                need(query_opted_in,"active query-pair fixture omitted explicit opt-in");
                const auto launches=registered.query_pair_launch_attempts-staged.query_pair_launch_attempts;
                const auto rows=registered.query_pair_row_attempts-staged.query_pair_row_attempts;
                need(staged.query_pair_launch_attempts>0 && launches>0,
                    "query-pair did not execute in both private and selected history routes");
                need(staged.query_pair_row_attempts>=staged.query_pair_launch_attempts &&
                    staged.query_pair_row_attempts<=16*staged.query_pair_launch_attempts &&
                    rows>=launches && rows<=16*launches,
                    "query-pair bounded layer-row attempt accounting diverged");
            } else need(registered.query_pair_launch_attempts==0 && registered.query_pair_row_attempts==0,
                "excluded query-pair profile dispatched candidate work");
        }
        if(!query_opted_in)
            need(registered.query_pair_launch_attempts==0 && registered.query_pair_row_attempts==0 &&
                registered.query_pair_requested_row_attempts==0,
                "query-pair opt-out changed actual Engine dispatch");
        if(staging_parity) {
            if(staging_turnover) {
                need(registered.registered_kv_upload_bytes>staged.registered_kv_upload_bytes &&
                    registered.registered_kv_retired_registrations>staged.registered_kv_retired_registrations,
                    "staging with registered upload and turnover NOT_EXERCISED");
            }
            need(staged.attention_stage_upload_bytes==0 && staged.attention_stage_consumed_bytes==0 && staged.attention_stage_direct_bytes==0 &&
                staged.attention_stage_banks==0,"staging-disabled control submitted staging work");
            need(registered.attention_stage_banks>0 && registered.attention_stage_upload_bytes>0 &&
                registered.attention_stage_consumed_bytes+registered.attention_stage_direct_bytes==registered.attention_stage_upload_bytes,
                "staging parity not exercised or left unconsumed uploads");
            if(const auto* expected=std::getenv("NINFER_TEST_ENGINE_STAGE_DIRECT_EXPECT")) {
                const std::string_view route(expected);
                need(route=="direct" || route=="copy","staged history route expectation");
                if(route=="direct")need(registered.attention_stage_direct_bytes>0 &&
                    registered.attention_stage_consumed_bytes==0,"staged direct history route not exercised");
                else need(registered.attention_stage_direct_bytes==0 && registered.attention_stage_consumed_bytes>0,
                    "staged fallback copy route not exercised");
            }
            need(registered.attention_stage_device_bytes==static_cast<std::uint64_t>(opt.max_context)*4096*opt.max_concurrency &&
                registered.attention_stage_metadata_bytes>0,"staging reserved resource attribution");
            need(registered.attention_stage_device_bytes==staged.attention_stage_device_bytes &&
                registered.attention_stage_metadata_bytes==staged.attention_stage_metadata_bytes,
                "staging toggle reallocated reserved owners");
        } else if(page_parity) {
            need(registered.shared_device_page_allocations>0 &&
                registered.shared_device_page_ancestor_accepts>0 &&
                registered.shared_device_page_source_pages>0 &&
                registered.shared_device_page_source_host_bytes>=registered.shared_device_page_source_pages*
                    ninfer::exl3::Exl3DevicePageStorage::bytes &&
                registered.shared_device_page_resident_bytes==registered.shared_device_page_allocations*
                    ninfer::exl3::Exl3DevicePageStorage::bytes,
                "page registry allocation/byte conservation");
            need(registered.shared_device_page_retained_readers==0 &&
                registered.shared_device_page_pending_fill_bytes==0 && registered.shared_device_page_failed_allocations==0,
                "idle page parity retained pending readers/fills or failed allocation");
            need(registered.private_device_prefix_bytes==0,"shared page parity retained duplicate private prefix allocation");
            need(registered.shared_device_page_fill_bytes>0 &&
                registered.shared_device_page_copy_bytes+registered.shared_device_page_attention_bytes>
                    registered.shared_device_page_fill_bytes,
                "shared page fill/reuse parity NOT_EXERCISED");
            need(registered.shared_device_page_failures==0 && registered.registered_kv_upload_bytes==0,
                "shared page parity failed or used registered route");
            if(page_root_eviction) {
                const auto metadata=owner.prefix_retention_metadata();
                need(metadata.size()==1,
                    "bounded peer cache replacement retained more than one logical root");
                const std::array<ninfer::exl3::Exl3VeriCachePrefixIndex::RetentionDecision,1> remove{{
                    {metadata.front().root,metadata.front().generation,false,0}}};
                const auto source_before=owner.runtime_stats();
                const auto evicted=owner.trim_prefix_retention(0,remove);
                const auto source_after=owner.runtime_stats();
                need(evicted.inputs_current && evicted.budget_met && evicted.evicted==1 &&
                    owner.prefix_retention_metadata().empty() &&
                    source_after.shared_device_page_source_pages==source_before.shared_device_page_source_pages &&
                    source_after.shared_device_page_source_host_bytes==source_before.shared_device_page_source_host_bytes,
                    "logical Engine root eviction revoked or recharged live shared-page sources");
            }
        } else {
            need(registered.registered_kv_upload_bytes>0,"registered parity route NOT_EXERCISED");
            need(registered.registered_kv_upload_peak_pending==2,"registered two-slot pipeline NOT_EXERCISED");
            need(registered.shared_device_page_copy_bytes==0,"registered parity used shared page route");
            if(const auto* prefix=std::getenv("NINFER_TEST_ENGINE_REGISTERED_PREFIX_TURNOVER");
                prefix && std::string_view(prefix)=="1") {
                need(registered.private_device_prefix_bytes>0 || registered.shared_device_prefix_bytes>0,
                    "registered prefix turnover missing represented prefix owner");
                need(registered.registered_kv_retired_registrations>staged.registered_kv_retired_registrations,
                    "registered prefix turnover NOT_EXERCISED");
                if(registered.shared_device_prefix_bytes)
                    need(registered.shared_device_prefix_hit_bytes>staged.shared_device_prefix_hit_bytes,
                        "registered shared prefix reuse NOT_EXERCISED");
            }
            if(const auto* turnover=std::getenv("NINFER_TEST_ENGINE_REGISTERED_KV_TURNOVER");
                turnover && std::string_view(turnover)=="1")
                need(registered.registered_kv_retired_registrations>0,
                    "ordinary pinned registered cache turnover NOT_EXERCISED");
        }
        need(registered.registered_kv_upload_failures==0,"registered parity concealed upload failure");
        if(const auto* direct=std::getenv("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGE_ATTENTION");
            direct && std::string_view(direct)=="1")
        {
            std::cout<<"shared_page_direct groups="<<registered.shared_device_page_attention_groups
                <<" pages="<<registered.shared_device_page_attention_pages
                <<" max_pages="<<registered.shared_device_page_attention_max_pages
                <<" fill_bytes="<<registered.shared_device_page_fill_bytes
                <<" copy_bytes="<<registered.shared_device_page_copy_bytes
                <<" attention_bytes="<<registered.shared_device_page_attention_bytes<<'\n';
            need(registered.shared_device_page_attention_max_pages>=2 && registered.shared_device_page_attention_groups>0,
                "page parity never consumed multiple shared pages in one layer");
            need(registered.shared_device_page_attention_bytes==registered.shared_device_page_attention_pages*64ULL*1024*2*2,
                "completed attention page/byte accounting diverged");
        }
        else need(registered.shared_device_page_attention_bytes==0,"disabled shared attention route executed");
        if(registered_c2) {
            owner.set_host_kv_routes_for_test(true,false);
            const auto before_pair=owner.runtime_stats();
            auto first=submit(0);auto second=submit(1);
            const auto first_result=first.wait(nullptr,{}),second_result=second.wait(nullptr,{});
            need(first_result.generated_token_ids==expected[0] &&
                second_result.generated_token_ids==expected[1],"registered C2 changed private outputs");
            owner.set_host_kv_routes_for_test(false,false);
            const auto after_pair=owner.runtime_stats();
            need(after_pair.registered_kv_upload_bytes>before_pair.registered_kv_upload_bytes &&
                after_pair.registered_kv_upload_failures==before_pair.registered_kv_upload_failures,
                "registered C2 upload not exercised or failed");
            const auto lane0=after_pair.registered_kv_upload_lane_bytes[0]-before_pair.registered_kv_upload_lane_bytes[0];
            const auto lane1=after_pair.registered_kv_upload_lane_bytes[1]-before_pair.registered_kv_upload_lane_bytes[1];
            need(lane0>0 && lane1>0,"registered C2 physical lane coverage NOT_EXERCISED");
            need(lane0+lane1==after_pair.registered_kv_upload_bytes-before_pair.registered_kv_upload_bytes,
                "registered C2 lane bytes do not partition total work");
            std::lock_guard lock(captured_mutex);
            need(captured.size()==6,"registered C2 terminal root count");
            std::array<bool,2> matched{};
            for(std::size_t result=4;result<6;++result) {
                bool found=false;
                for(std::size_t baseline=0;baseline<2;++baseline)if(!matched[baseline] &&
                    captured[baseline]->state()->same_payload(*captured[result]->state()) &&
                    captured[baseline]->same_projected_conditioning_for_test(*captured[result])) {
                    matched[baseline]=true;found=true;break;
                }
                need(found,"registered C2 terminal state/conditioning lost private baseline identity");
            }
            captured.resize(4);
        }
        std::array<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>,2> cancellation_baselines;
        {
            std::lock_guard lock(captured_mutex);need(captured.size()==4,"registered parity terminal root count");
            cancellation_baselines={captured[0],captured[1]};
            std::array<bool,2> matched{};
            for(std::size_t result=2;result<4;++result) {
                bool found=false;
                for(std::size_t baseline=0;baseline<2;++baseline)if(!matched[baseline] &&
                    captured[baseline]->state()->same_payload(*captured[result]->state()) &&
                    captured[baseline]->same_projected_conditioning_for_test(*captured[result])) {
                    matched[baseline]=true;found=true;break;
                }
                need(found,"host-route terminal state/conditioning lost private baseline identity");
            }
            captured.clear();
        }
        if(staging_turnover) {
            owner.refuse_attention_read_pairs_for_test(true);
            owner.set_host_kv_routes_for_test(true,false);
            const auto before=owner.runtime_stats();
            for(unsigned i=0;i<2;++i) {
                auto handle=submit(i);
                need(handle.wait(nullptr,{}).generated_token_ids==expected[i],
                    "attention admission fallback changed tokens");
            }
            owner.set_host_kv_routes_for_test(false,false);
            const auto after=owner.runtime_stats();
            need(after.attention_stage_upload_bytes==before.attention_stage_upload_bytes &&
                after.attention_stage_banks==before.attention_stage_banks &&
                after.registered_kv_upload_bytes>before.registered_kv_upload_bytes &&
                after.registered_kv_upload_failures==before.registered_kv_upload_failures &&
                after.registered_kv_external_readers==0,
                "attention admission refusal submitted staging or lost ordinary fallback");
            {
                std::lock_guard lock(captured_mutex);
                need(captured.size()==2,"attention fallback terminal roots missing");
                for(unsigned i=0;i<2;++i)
                    need(captured[i]->state()->same_payload(*cancellation_baselines[i]->state()) &&
                        captured[i]->same_projected_conditioning_for_test(*cancellation_baselines[i]),
                        "attention admission fallback changed full state/conditioning");
                captured.clear();
            }
            owner.refuse_attention_read_pairs_for_test(false);
        }
        if(page_parity) {
            owner.use_foreign_page_ancestor_for_test(true);
            owner.set_host_kv_routes_for_test(false,true);
            const auto before=owner.runtime_stats();
            for(unsigned i=0;i<2;++i) {
                auto handle=submit(i);
                need(handle.wait(nullptr,{}).generated_token_ids==expected[i],
                    "foreign page ancestry fallback changed output tokens");
            }
            owner.set_host_kv_routes_for_test(false,false);
            owner.use_foreign_page_ancestor_for_test(false);
            const auto after=owner.runtime_stats();
            need(after.shared_device_page_ancestor_fallbacks>before.shared_device_page_ancestor_fallbacks &&
                after.shared_device_page_ancestor_accepts==before.shared_device_page_ancestor_accepts &&
                after.shared_device_page_fill_bytes==before.shared_device_page_fill_bytes &&
                after.shared_device_page_copy_bytes==before.shared_device_page_copy_bytes &&
                after.shared_device_page_attention_bytes==before.shared_device_page_attention_bytes &&
                after.shared_device_page_failures==before.shared_device_page_failures,
                "foreign ancestry used shared storage or failed instead of private fallback");
            {
                std::lock_guard lock(captured_mutex);
                need(captured.size()==2,"foreign ancestry fallback terminal roots missing");
                for(unsigned i=0;i<2;++i)
                    need(captured[i]->state()->same_payload(*cancellation_baselines[i]->state()) &&
                        captured[i]->same_projected_conditioning_for_test(*cancellation_baselines[i]),
                        "foreign ancestry fallback changed target state or draft conditioning");
                captured.clear();
            }
        }
        if(page_parity || staging_parity || registered_c2) {
            owner.set_host_kv_routes_for_test(registered_c2 || staging_turnover,page_parity);
            if(staging_parity)owner.set_attention_staging_for_test(true);
            const auto before_cancel=owner.runtime_stats();
            std::atomic<bool> cancel=false;CancellingSink sink(cancel);
            auto prepared=owner.frontend().prepare(prompt(prompts[0]));const auto summary=prepared.summary();
            auto streaming=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Streaming,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            std::optional<ninfer::exl3::Exl3EngineCore::Submission> peer;
            if(page_c2 || staging_c2 || registered_c2)peer.emplace(submit(1));
            const auto cancelled=streaming.wait(&sink,CancellationView([&]{return cancel.load();}));
            need(cancelled.finish_reason==FinishReason::Cancelled && !sink.text.empty(),"page parity cancellation not exercised");
            if(peer)need(peer->wait(nullptr,{}).generated_token_ids==expected[1],"page cancellation contaminated concurrent peer");
            if(peer) {
                owner.set_host_kv_routes_for_test(registered_c2 || staging_turnover,page_parity);
                std::lock_guard lock(captured_mutex);
                bool peer_state=false;
                for(const auto& root:captured)if(root->state()->same_payload(*cancellation_baselines[1]->state()) &&
                    root->same_projected_conditioning_for_test(*cancellation_baselines[1]))peer_state=true;
                need(peer_state,"cancelled request changed surviving peer state/conditioning");
                captured.clear();
            }
            {auto abandoned=submit(1);}
            auto reused=submit(0);
            need(reused.wait(nullptr,{}).generated_token_ids==expected[0],"page cancellation/drop changed subsequent tokens");
            owner.set_host_kv_routes_for_test(false,false); // wait for all worker ownership cleanup
            const auto drained=owner.runtime_stats();
            {
                std::lock_guard lock(captured_mutex);
                bool reused_state=false;
                for(const auto& root:captured)if(root->state()->same_payload(*cancellation_baselines[0]->state()) &&
                    root->same_projected_conditioning_for_test(*cancellation_baselines[0]))reused_state=true;
                need(reused_state,"post-cancellation reuse changed full state/conditioning");
            }
            if(registered_c2 || staging_turnover) {
                need(drained.registered_kv_upload_bytes>before_cancel.registered_kv_upload_bytes &&
                    drained.registered_kv_upload_failures==before_cancel.registered_kv_upload_failures,
                    "registered cancellation/reuse upload not exercised or failed");
                const auto first=drained.registered_kv_upload_lane_bytes[0]-before_cancel.registered_kv_upload_lane_bytes[0];
                const auto second=drained.registered_kv_upload_lane_bytes[1]-before_cancel.registered_kv_upload_lane_bytes[1];
                need(first+second==drained.registered_kv_upload_bytes-before_cancel.registered_kv_upload_bytes,
                    "registered cancellation/reuse lost lane attribution");
                if(staging_turnover && staging_c2)
                    need(first>0 && second>0,"combined staging cancellation missed a physical registered lane");
                need(drained.registered_kv_external_readers==0,
                    "registered cancellation/reuse retained external history readers");
            }
            if(staging_parity) {
                owner.set_attention_staging_for_test(false); // refuses any unfinished stage
                need(drained.attention_stage_banks>registered.attention_stage_banks &&
                    drained.attention_stage_upload_bytes==drained.attention_stage_consumed_bytes+drained.attention_stage_direct_bytes,
                    "cancel/drop left unfinished staging traffic or never exercised reuse");
                need(drained.attention_stage_device_bytes==registered.attention_stage_device_bytes &&
                    drained.attention_stage_metadata_bytes==registered.attention_stage_metadata_bytes,
                    "cancel/drop changed staging reservation ownership");
            }
            need(drained.shared_device_page_retained_readers==0 && drained.shared_device_page_pending_fill_bytes==0 &&
                drained.shared_device_page_in_flight_fill_bytes==0 && drained.shared_device_page_failed_allocations==0 &&
                drained.shared_device_page_failures==0,"cancel/drop retained unfinished page ownership");
            need(drained.shared_device_page_resident_bytes==drained.shared_device_page_allocations*
                ninfer::exl3::Exl3DevicePageStorage::bytes,"cancel/drop lost unique page allocation credit");
            {std::lock_guard lock(captured_mutex);captured.clear();}
        }
        need(owner.close().reusable(),"registered parity did not retire Engine");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_STATE");mode && std::string(mode)=="1") {
        const std::string shared_matrix=std::getenv("NINFER_TEST_ENGINE_SHARED_MATRIX")?
            std::getenv("NINFER_TEST_ENGINE_SHARED_MATRIX"):"";
        need(shared_matrix.empty() || shared_matrix=="draft" || shared_matrix=="target" ||
            shared_matrix=="composed" || shared_matrix=="disabled",
            "shared Engine matrix requires draft, target, composed or disabled");
        const auto route_enabled=[](const char* name) {
            const auto* value=std::getenv(name);return value && std::string_view(value)=="1";
        };
        const bool target_q_enabled=route_enabled("NINFER_EXL3_ENGINE_SHARED_TARGET_Q");
        const bool draft_q_enabled=route_enabled("NINFER_EXL3_ENGINE_SHARED_DRAFT_Q_M16");
        const bool packet_batch_enabled=route_enabled("NINFER_EXL3_ENGINE_BATCHED_GREEDY_PACKET");
        const bool device_seed_enabled=route_enabled("NINFER_EXL3_DEVICE_SEED_HANDOFF");
        const bool tap_d2d_enabled=route_enabled("NINFER_EXL3_COMMITTED_TAP_D2D");
        if(shared_matrix=="draft")need(!target_q_enabled && draft_q_enabled,
            "isolated shared-draft matrix enabled target Q or omitted draft Q");
        if(shared_matrix=="target")need(target_q_enabled && !draft_q_enabled,
            "isolated shared-target matrix enabled draft Q or omitted target Q");
        if(shared_matrix=="composed")need(target_q_enabled && draft_q_enabled,
            "composed shared matrix requires target Q and draft Q");
        if(shared_matrix=="disabled")need(!target_q_enabled && !draft_q_enabled,
            "disabled shared matrix enabled a projection owner");
        const auto* conditional_mode=std::getenv("NINFER_TEST_ENGINE_SHARED_CONDITIONAL");
        const bool conditional_pair=conditional_mode && std::string_view(conditional_mode)=="1";
        const auto* mixed_option=std::getenv("NINFER_TEST_ENGINE_SHARED_CONDITIONAL_MIXED");
        const bool mixed_conditional=mixed_option && std::string_view(mixed_option)=="1";
        need(!mixed_option || !*mixed_option || std::string_view(mixed_option)=="0" || mixed_conditional,
            "conditional mixed option requires0 or1");
        need(!mixed_conditional || conditional_pair,"mixed conditional fixture requires conditional composition");
        if(conditional_pair) {
            const auto* enabled=std::getenv("NINFER_EXL3_ENGINE_CONDITIONAL_B8");
            const auto* shared=std::getenv("NINFER_EXL3_ENGINE_SHARED_DRAFT_Q_M16");
            need(enabled && std::string_view(enabled)=="1" && shared && std::string_view(shared)=="1",
                "conditional shared fixture requires actual Engine conditional and draft Q routes");
        }
        const auto* control_option=std::getenv("NINFER_TEST_ENGINE_SHARED_CONTROL");
        const bool control_case=control_option && std::string(control_option)=="1";
        need(!control_option || !*control_option || std::string(control_option)=="0" || control_case,
            "shared control option requires0 or1");
        unsigned first_limit=32;
        if(const auto* limit=std::getenv("NINFER_TEST_ENGINE_SHARED_FIRST_LIMIT");limit && *limit) {
            need(!control_case,"shared control owns output96; omit shared first limit");
            bool found=false;
            for(unsigned candidate:{1U,2U,3U,4U,5U,6U,7U,8U,9U,16U,32U})if(std::string(limit)==std::to_string(candidate)) {
                first_limit=candidate;found=true;
            }
            need(found,"shared first output limit requires1..9,16 or32");
        }
        if(control_case) {
            const auto* terminal=std::getenv("NINFER_TEST_ENGINE_SHARED_TERMINAL");
            need(!terminal || std::string(terminal)!="1",
                "shared control and token-terminal fixtures require separate invocations");
            first_limit=96;
        }
        if(mixed_conditional)need(first_limit==9 && !control_case,
            "mixed conditional fixture requires first output9 without thinking control");
        opt.max_concurrency=2;
        ninfer::exl3::Exl3EngineCore owner(opt);
        std::mutex captured_mutex;
        std::vector<std::shared_ptr<const ninfer::exl3::Exl3VeriCacheRequest>> captured;
        owner.observe_terminal_roots_for_test([&](auto root){std::lock_guard lock(captured_mutex);captured.push_back(std::move(root));});
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=32;request.execution.allow_prefix_reuse=false;
        request.stop=owner.frontend().default_stop_policy();
        const std::array<std::string,2> prompts{"Write a Python function that adds two integers. Return only code.",
            "Explain why leaves change color in autumn in two short sentences."};
        // Completion order is arbitrary. Consume each serial authority exactly
        // once; token equality alone may select the wrong private state.
        const auto require_paired_authorities=[&] {
            need(captured.size()==4,"terminal root observation count");
            std::array<bool,2> matched{};
            for(unsigned i=2;i<4;++i) {
                bool found=false;
                for(unsigned reference=0;reference<2;++reference) {
                    if(matched[reference] ||
                        captured[i]->token_suffix()!=captured[reference]->token_suffix() ||
                        !captured[i]->state()->same_payload(*captured[reference]->state()) ||
                        !captured[i]->same_projected_conditioning_for_test(*captured[reference]))continue;
                    matched[reference]=true;found=true;break;
                }
                need(found,"shared Engine lost or duplicated a private terminal authority");
            }
            need(matched[0] && matched[1],"shared Engine omitted a serial authority");
        };
        std::optional<TokenId> first_terminal;
        const auto submit=[&](unsigned index) {
            auto prepared=owner.frontend().prepare(prompt(control_case && index==0?"What is 2 plus 2?":prompts[index],
                control_case && index==0));const auto summary=prepared.summary();
            auto limited=request;limited.execution.requested_output_tokens=index?32:first_limit;
            if(control_case && index==0)limited.execution.thinking.budget=1;
            if(index==0 && first_terminal)limited.stop.token_ids.push_back(*first_terminal);
            return owner.submit(std::move(prepared),summary,0,limited,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
        };
        auto serial_a=submit(0);const auto serial_result_a=serial_a.wait(nullptr,{});
        const auto expected_a=serial_result_a.generated_token_ids;
        if(control_case)need(serial_result_a.thinking.applied && serial_result_a.thinking.injected_tokens>0 &&
            serial_result_a.thinking.model_thinking_tokens==1,"serial thinking control NOT_EXERCISED");
        const auto require_control_parity=[&](const auto& result) {
            if(!control_case)return;
            need(result.thinking.applied==serial_result_a.thinking.applied &&
                result.thinking.injected_tokens==serial_result_a.thinking.injected_tokens &&
                result.thinking.model_thinking_tokens==serial_result_a.thinking.model_thinking_tokens &&
                result.finish_reason==serial_result_a.finish_reason,
                "shared thinking control metadata or finish reason changed");
        };
        owner.set_host_kv_routes_for_test(false,false);
        const auto short_work=owner.runtime_stats();
        if(conditional_pair && first_limit<16)
            need(short_work.conditional_second_block_calls==0,
                "short serial baseline started conditional second block");
        if(first_limit==1)need(short_work.draft_neural_input_rows==0 &&
            short_work.draft_returned_suffix_rows==0 && short_work.draft_discarded_suffix_rows==0,
            "seed-only request executed or fabricated neural B8");
        auto serial_b=submit(1);const auto serial_result_b=serial_b.wait(nullptr,{});
        const auto expected_b=serial_result_b.generated_token_ids;
        const auto require_private_neural_work=[&](const auto& result,const auto& serial) {
            // Conservation alone cannot detect counters copied from the peer:
            // both requests may independently satisfy the seven-suffix rule.
            // Sharing changes physical projection grouping, not which private
            // fixed-B8 proposal calls each request performs in this parity case.
            const auto& actual=result.speculative;
            const auto& expected=serial.speculative;
            need(actual.neural_input_rows==expected.neural_input_rows &&
                actual.returned_suffix_rows==expected.returned_suffix_rows &&
                actual.discarded_suffix_rows==expected.discarded_suffix_rows,
                "shared request inherited peer work or changed private B8 proposal accounting");
            need(actual.committed_tap_device_bytes==expected.committed_tap_device_bytes,
                "shared request changed private committed-tap transport volume");
            need(actual.device_seed_handoffs==expected.device_seed_handoffs &&
                actual.device_seed_host_fallbacks==expected.device_seed_host_fallbacks,
                "shared request changed private seed authority route");
        };
        const auto require_peer_control_parity=[&](const auto& result) {
            need(result.thinking.applied==serial_result_b.thinking.applied &&
                result.thinking.injected_tokens==serial_result_b.thinking.injected_tokens &&
                result.thinking.model_thinking_tokens==serial_result_b.thinking.model_thinking_tokens &&
                result.finish_reason==serial_result_b.finish_reason,
                "shared peer inherited control metadata or changed finish reason");
        };
        owner.set_host_kv_routes_for_test(false,false); // complete serial worker cleanup before policy mutation
        if(packet_batch_enabled)
            owner.set_greedy_packet_batch_timeout_for_test(std::chrono::milliseconds(5));
        need(expected_a.size()<=first_limit && expected_b.size()<=32,"serial mixed output budget exceeded");
        if(shared_matrix=="disabled") {
            auto disabled_a=submit(0),disabled_b=submit(1);
            const auto result_a=disabled_a.wait(nullptr,{}),result_b=disabled_b.wait(nullptr,{});
            need(result_a.generated_token_ids==expected_a && result_b.generated_token_ids==expected_b,
                "both-disabled C2 changed serial request output");
            owner.set_host_kv_routes_for_test(false,false);
            {
                std::lock_guard lock(captured_mutex);require_paired_authorities();captured.clear();
            }
            const auto work=owner.runtime_stats();
            need(work.shared_target_projection_batches==0 &&
                work.shared_target_projection_fallbacks==0 &&
                work.shared_projection_completed_dispatches==0 &&
                !work.shared_projection_retains_claim_owners,
                "both-disabled C2 constructed or dispatched a shared projection route");
            if(packet_batch_enabled)need(work.greedy_packet_transport_batches>0 &&
                work.greedy_packet_transport_rows==2*work.greedy_packet_transport_batches,
                "both-projection-disabled C2 did not exercise independent packet transport");
            need(owner.close().reusable(),"both-disabled shared matrix did not retire Engine");
            std::cout<<"ENGINE_SHARED_MATRIX_DISABLED_COMPLETE\n";return 0;
        }
        owner.set_shared_cost_policy_for_test(
            ninfer::exl3::Exl3PackedCostPolicy::unqualified_test_permit_underfilled());
        owner.set_shared_rendezvous_timeout_for_test(std::chrono::milliseconds(5));
        const auto before_forward_pair=owner.runtime_stats();
        auto paired_a=submit(0),paired_b=submit(1);
        const auto result_a=paired_a.wait(nullptr,{}),result_b=paired_b.wait(nullptr,{});
        need(result_a.generated_token_ids==expected_a,"shared state fixture first tokens");
        need(result_b.generated_token_ids==expected_b,"shared state fixture second tokens");
        require_private_neural_work(result_a,serial_result_a);
        require_private_neural_work(result_b,serial_result_b);
        require_control_parity(result_a);
        require_peer_control_parity(result_b);
        if(control_case)need(result_a.thinking.applied && result_a.thinking.injected_tokens>0 &&
            result_a.thinking.model_thinking_tokens==1,"paired thinking control NOT_EXERCISED");
        for(const auto* result:{&result_a,&result_b}) {
            const auto& work=result->speculative;
            need(work.neural_input_rows%8==0 && work.returned_suffix_rows+work.discarded_suffix_rows==
                (work.neural_input_rows/8)*7,"per-request B8 prediction accounting mismatch");
        }
        if(first_limit==1)need(result_a.speculative.neural_input_rows==0,
            "seed-only paired request inherited peer neural work");
        for(const auto* result:{&result_a,&result_b}) {
            if(tap_d2d_enabled)
                need(result->speculative.committed_tap_device_bytes>0,
                    "actual Engine request omitted selected committed-tap D2D transport");
            else need(result->speculative.committed_tap_device_bytes==0,
                "option-off Engine request reported committed-tap D2D transport");
        }
        if(device_seed_enabled)need(result_b.speculative.device_seed_handoffs>0,
            "actual Engine request omitted selected device-seed handoff");
        if(first_limit>1 && first_limit<8)need(result_a.speculative.neural_input_rows>0 &&
            result_a.speculative.discarded_suffix_rows>0,"short neural discarded-tail route NOT_EXERCISED");
        owner.set_host_kv_routes_for_test(false,false);
        {
            std::lock_guard lock(captured_mutex);require_paired_authorities();
            captured.resize(2); // Retain both independent serial authorities for reversed arrival.
        }
        const auto after_forward_pair=owner.runtime_stats();
        auto reversed_b=submit(1),reversed_a=submit(0);
        const auto reversed_result_b=reversed_b.wait(nullptr,{}),reversed_result_a=reversed_a.wait(nullptr,{});
        need(reversed_result_a.generated_token_ids==expected_a &&
            reversed_result_b.generated_token_ids==expected_b,"reversed arrival changed private output");
        require_private_neural_work(reversed_result_a,serial_result_a);
        require_private_neural_work(reversed_result_b,serial_result_b);
        require_control_parity(reversed_result_a);
        require_peer_control_parity(reversed_result_b);
        if(control_case)need(reversed_result_a.thinking.applied && reversed_result_a.thinking.injected_tokens>0 &&
            reversed_result_a.thinking.model_thinking_tokens==1,"reversed thinking control NOT_EXERCISED");
        for(const auto* result:{&reversed_result_a,&reversed_result_b}) {
            const auto& work=result->speculative;
            need(work.neural_input_rows%8==0 && work.returned_suffix_rows+work.discarded_suffix_rows==
                (work.neural_input_rows/8)*7,"reversed arrival lost private B8 accounting");
        }
        if(first_limit==1)need(reversed_result_a.speculative.neural_input_rows==0,
            "reversed seed-only request inherited peer neural work");
        owner.set_host_kv_routes_for_test(false,false);
        {
            std::lock_guard lock(captured_mutex);require_paired_authorities();captured.clear();
        }
        const auto sharing=owner.runtime_stats();
        if(packet_batch_enabled) {
            need(after_forward_pair.greedy_packet_transport_batches>
                    before_forward_pair.greedy_packet_transport_batches &&
                sharing.greedy_packet_transport_batches>
                    after_forward_pair.greedy_packet_transport_batches,
                "batched greedy transport not exercised in both arrival orders");
            need(sharing.greedy_packet_transport_rows==
                    2*sharing.greedy_packet_transport_batches &&
                sharing.greedy_packet_transport_failures==0,
                "batched greedy transport row conservation or completion failure");
        } else need(sharing.greedy_packet_transport_batches==0 &&
            sharing.greedy_packet_transport_rows==0,
            "disabled greedy packet transport published batch work");
        if(packet_batch_enabled || device_seed_enabled || tap_d2d_enabled)
            std::cout<<"ENGINE_TRANSPORT_ROUTE forward_decode_s="
                <<result_a.timings.decode_seconds+result_b.timings.decode_seconds
                <<" reverse_decode_s="
                <<reversed_result_a.timings.decode_seconds+reversed_result_b.timings.decode_seconds
                <<" packet_batches="<<sharing.greedy_packet_transport_batches
                <<" seed_handoffs="
                <<result_a.speculative.device_seed_handoffs+result_b.speculative.device_seed_handoffs+
                    reversed_result_a.speculative.device_seed_handoffs+reversed_result_b.speculative.device_seed_handoffs
                <<" tap_d2d_bytes="
                <<result_a.speculative.committed_tap_device_bytes+result_b.speculative.committed_tap_device_bytes+
                    reversed_result_a.speculative.committed_tap_device_bytes+
                    reversed_result_b.speculative.committed_tap_device_bytes<<'\n';
        if(conditional_pair) {
            const auto require_composition=[&](const auto& before,const auto& after) {
                need(after.conditional_second_block_calls>before.conditional_second_block_calls,
                    "conditional C2 second block NOT_EXERCISED");
                need(after.shared_target_projection_family_batches[3]>before.shared_target_projection_family_batches[3],
                    "conditional C2 shared draft Q NOT_EXERCISED");
                const auto batches=after.conditional_shared_draft_batches-before.conditional_shared_draft_batches;
                const auto lanes=after.conditional_shared_draft_lanes-before.conditional_shared_draft_lanes;
                need(batches>0 && lanes>=batches && lanes<=2*batches,
                    "actual conditional child-root packed draft dispatch NOT_EXERCISED");
                const auto mixed=after.conditional_shared_draft_peers[0]-before.conditional_shared_draft_peers[0];
                const auto both=after.conditional_shared_draft_peers[1]-before.conditional_shared_draft_peers[1];
                need(mixed+both==batches && mixed+2*both==lanes,
                    "conditional packed peer attribution differs from completed work");
                if(mixed_conditional)need(mixed>0 && both==0,
                    "asymmetric conditional mixed peer NOT_EXERCISED or short peer speculated twice");
                std::uint64_t total=0;
                for(std::size_t family=0;family<15;++family) {
                    const auto count=after.conditional_shared_draft_families[family]-before.conditional_shared_draft_families[family];
                    need(count<=after.shared_target_projection_family_batches[family]-before.shared_target_projection_family_batches[family],
                        "conditional family count exceeds actual packed family work");
                    total+=count;
                }
                need(total==batches && after.conditional_shared_draft_families[3]>before.conditional_shared_draft_families[3],
                    "conditional packed draft Q family NOT_EXERCISED");
                std::cout<<"CONDITIONAL_PACKED mixed="<<mixed<<" both="<<both<<'\n';
            };
            require_composition(before_forward_pair,after_forward_pair);
            require_composition(after_forward_pair,sharing);
            // Child-root counters are charged only after successful packed
            // completion, separately from proposal reach and ordinary draft Q.
        }
        need(sharing.draft_neural_input_rows%8==0 &&
            sharing.draft_returned_suffix_rows+sharing.draft_discarded_suffix_rows==
                (sharing.draft_neural_input_rows/8)*7,
            "mixed output budgets lost physical B8 suffix accounting");
        if(control_case) {
            need(after_forward_pair.shared_projection_contract_refusals>
                    before_forward_pair.shared_projection_contract_refusals &&
                sharing.shared_projection_contract_refusals>
                    after_forward_pair.shared_projection_contract_refusals,
                "thinking-control peers did not reach exact preclaim contract refusal in both arrival orders");
            need(after_forward_pair.shared_target_projection_batches==
                    before_forward_pair.shared_target_projection_batches &&
                sharing.shared_target_projection_batches==after_forward_pair.shared_target_projection_batches,
                "different retained control spans entered shared physical work");
        }
        if(const auto* gateup=std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_GATEUP");!control_case && first_limit>1 && gateup && std::string(gateup)=="1") {
            std::uint64_t gdn=0;for(unsigned layer=0;layer<64;++layer)if(layer%4!=3)gdn+=sharing.shared_target_layer_batches[layer];
            need(gdn>0,"shared state GDN MLP NOT_EXERCISED");
        }
        if(first_limit>1 && !control_case && target_q_enabled)
            need(sharing.shared_target_projection_family_batches[0]>0,"shared state Q fixture NOT_EXERCISED");
        const auto* gather=std::getenv("NINFER_EXL3_ENGINE_SHARED_GATHER_REUSE");
        if(gather && std::string(gather)=="1" && first_limit>1 && !control_case) {
            need(after_forward_pair.shared_target_reused_gather_bytes>
                before_forward_pair.shared_target_reused_gather_bytes,
                "forward arrival gather reuse NOT_EXERCISED");
            need(sharing.shared_target_reused_gather_bytes>
                after_forward_pair.shared_target_reused_gather_bytes,
                "reversed arrival gather reuse NOT_EXERCISED");
        }
        else need(sharing.shared_target_reused_gather_bytes==0,"default gather unexpectedly reused");
        const std::array<std::pair<const char*,unsigned>,14> enabled_families{{
            {"NINFER_EXL3_ENGINE_SHARED_TARGET_KV",1},{"NINFER_EXL3_ENGINE_SHARED_TARGET_KV",2},
            {"NINFER_EXL3_ENGINE_SHARED_DRAFT_Q_M16",3},{"NINFER_EXL3_ENGINE_SHARED_TARGET_O",4},
            {"NINFER_EXL3_ENGINE_SHARED_TARGET_GATEUP",5},{"NINFER_EXL3_ENGINE_SHARED_TARGET_GATEUP",6},
            {"NINFER_EXL3_ENGINE_SHARED_TARGET_DOWN",7},{"NINFER_EXL3_ENGINE_SHARED_HEAD",8},
            {"NINFER_EXL3_ENGINE_SHARED_DRAFT_KV_M16",9},{"NINFER_EXL3_ENGINE_SHARED_DRAFT_KV_M16",10},
            {"NINFER_EXL3_ENGINE_SHARED_DRAFT_O_M16",11},{"NINFER_EXL3_ENGINE_SHARED_DRAFT_DOWN_M16",12},
            {"NINFER_EXL3_ENGINE_SHARED_DRAFT_GATEUP_M16",13},{"NINFER_EXL3_ENGINE_SHARED_DRAFT_GATEUP_M16",14}}};
        for(const auto& [flag,index]:enabled_families) {
            const auto* selected=std::getenv(flag);
            const bool enabled=selected && std::string(selected)=="1";
            if(first_limit>1 && enabled && !control_case) {
                need(after_forward_pair.shared_target_projection_family_batches[index]>
                    before_forward_pair.shared_target_projection_family_batches[index],
                    "forward arrival enabled family NOT_EXERCISED");
                need(sharing.shared_target_projection_family_batches[index]>
                    after_forward_pair.shared_target_projection_family_batches[index],
                    "reversed arrival enabled family NOT_EXERCISED");
            }
            if(control_case)need(after_forward_pair.shared_target_projection_family_batches[index]==
                    before_forward_pair.shared_target_projection_family_batches[index] &&
                sharing.shared_target_projection_family_batches[index]==
                    after_forward_pair.shared_target_projection_family_batches[index],
                "control-contract refusal dispatched an optional projection family");
            if(first_limit>1 && enabled)
                need(sharing.shared_target_projection_family_fallbacks[index]>0,
                    "eligible projection family never recorded its private fallback path");
            if(!enabled)need(sharing.shared_target_projection_family_batches[index]==0,
                "shared state dispatched a disabled projection family");
        }
        if(first_limit>1 && !control_case && target_q_enabled) {
            need(after_forward_pair.shared_target_projection_family_batches[0]>
                before_forward_pair.shared_target_projection_family_batches[0],
                "forward arrival shared Q NOT_EXERCISED");
            need(sharing.shared_target_projection_family_batches[0]>
                after_forward_pair.shared_target_projection_family_batches[0],
                "reversed arrival shared Q NOT_EXERCISED");
        }
        if(target_q_enabled)need(sharing.shared_target_projection_family_fallbacks[0]>0,
            "shared Q workload omitted eligible serial/private fallback attribution");
        else need(sharing.shared_target_projection_family_batches[0]==0 &&
            sharing.shared_target_projection_family_fallbacks[0]==0,
            "draft-only workload installed target Q execution");
        {
            std::uint64_t family_fallbacks=0;
            for(const auto value:sharing.shared_target_projection_family_fallbacks)family_fallbacks+=value;
            need(family_fallbacks==sharing.shared_target_projection_fallbacks,
                "per-family fallback accounting does not conserve global eligible offers");
        }
        if(first_limit==1)for(unsigned family:{3U,9U,10U,11U,12U,13U,14U})
            need(sharing.shared_target_projection_family_batches[family]==0,"seed-only request fabricated paired neural M16");
        if(const auto* terminal_case=std::getenv("NINFER_TEST_ENGINE_SHARED_TERMINAL");terminal_case && std::string(terminal_case)=="1") {
            need(!expected_a.empty(),"terminal source reference empty");
            first_terminal=expected_a[std::min<std::size_t>(3,expected_a.size()-1)];
            auto terminal_serial=submit(0);const auto terminal_reference=terminal_serial.wait(nullptr,{});
            auto peer_serial=submit(1);const auto peer_reference=peer_serial.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            need(terminal_reference.finish_reason==FinishReason::StopToken,"selected terminal NOT_EXERCISED");
            auto terminal_pair=submit(0),peer_pair=submit(1);
            const auto terminal_result=terminal_pair.wait(nullptr,{}),peer_result=peer_pair.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            need(terminal_result.finish_reason==FinishReason::StopToken &&
                terminal_result.generated_token_ids==terminal_reference.generated_token_ids &&
                peer_result.generated_token_ids==peer_reference.generated_token_ids &&
                peer_result.finish_reason==peer_reference.finish_reason,"paired terminal changed release or peer output");
            const auto require_terminal_accounting=[&](const auto& result) {
                const auto& work=result.speculative;
                need(work.neural_input_rows%8==0 && work.returned_suffix_rows+work.discarded_suffix_rows==
                    (work.neural_input_rows/8)*7,"terminal handling changed completed B8 suffix accounting");
            };
            require_terminal_accounting(terminal_result);require_terminal_accounting(peer_result);
            {
                std::lock_guard lock(captured_mutex);
                require_paired_authorities();captured.resize(2);
            }
            auto terminal_reverse_peer=submit(1),terminal_reverse_first=submit(0);
            const auto reverse_peer=terminal_reverse_peer.wait(nullptr,{});
            const auto reverse_terminal=terminal_reverse_first.wait(nullptr,{});
            owner.set_host_kv_routes_for_test(false,false);
            need(reverse_terminal.finish_reason==FinishReason::StopToken &&
                reverse_terminal.generated_token_ids==terminal_reference.generated_token_ids &&
                reverse_peer.generated_token_ids==peer_reference.generated_token_ids &&
                reverse_peer.finish_reason==peer_reference.finish_reason,
                "reversed terminal arrival changed stop boundary or peer output");
            require_terminal_accounting(reverse_terminal);require_terminal_accounting(reverse_peer);
            {
                std::lock_guard lock(captured_mutex);
                require_paired_authorities();captured.clear();
            }
            for(const auto& retained:owner.shared_claim_owners_for_test())
                need(retained.expired(),"terminal pair retained completed physical claim owners");
        }
        need(owner.close().reusable(),"shared state fixture retirement");return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_INVALID_SHARED_OFFER");mode && std::string(mode)=="1") {
        opt.max_concurrency=2;
        ninfer::exl3::Exl3EngineCore owner(opt);
        owner.invalidate_next_shared_offer_for_test();
        bool duplicate_refused=false;
        try{owner.invalidate_next_shared_offer_for_test();}
        catch(const std::logic_error&){duplicate_refused=true;}
        need(duplicate_refused,"invalid offer seam replaced outstanding injection");
        runtime::ResolvedRequestOptions request;
        request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=16;
        request.execution.allow_prefix_reuse=false;
        request.stop=owner.frontend().default_stop_policy();
        auto prepared=owner.frontend().prepare(prompt("Write a Python function that adds two integers. Return only code."));
        const auto summary=prepared.summary();
        auto handle=owner.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
            std::chrono::steady_clock::now()+std::chrono::minutes(5));
        bool rejected=false;
        try{handle.wait(nullptr,{});}
        catch(const std::exception& e) {
            rejected=std::string(e.what()).find("shared projection offer has invalid ownership or geometry")!=std::string::npos;
        }
        need(rejected,"actual invalid offer admission NOT_EXERCISED");
        need(owner.close().reusable(),"invalid offer prevented certified Engine retirement");
        const auto work=owner.runtime_stats();
        need(work.shared_target_projection_batches==0 && work.shared_projection_completed_dispatches==0 &&
            work.shared_target_projection_failures==0 && !work.shared_projection_retains_claim_owners,
            "invalid offer entered physical shared dispatch or retained claims");
        bool closed_refused=false;
        try{owner.invalidate_next_shared_offer_for_test();}
        catch(const std::logic_error&){closed_refused=true;}
        need(closed_refused,"closed Engine accepted invalid-offer injection");
        return 0;
    }
    if(const auto* mode=std::getenv("NINFER_TEST_ENGINE_SHARED_CALLBACK_RETIREMENT");mode && std::string_view(mode)=="1") {
        using Core=ninfer::exl3::Exl3EngineCore;
        opt.max_concurrency=2;
        const auto blocks=Core::shared_projection_owner_blocks_for_test();
        std::weak_ptr<const void> backing;
        {
        // Observation storage must outlive owner destruction if an assertion
        // throws before explicit close releases the captured final deleter.
        std::atomic<unsigned> calls{0};
        std::atomic<bool> retirement_observed_empty{false};
        Core owner(opt);
        backing=owner.shared_projection_owner_for_test();
        const auto bytes=owner.shared_projection_metadata_bytes_for_test();
        need(bytes>0 && !backing.expired() && !owner.shared_projection_retirement_credit_for_test(),
            "shared rendezvous metadata fixture missing owner or already retired");
        auto capture=std::shared_ptr<int>(new int(42),[&](int* value) noexcept {
            delete value;
            try {
                const auto claims=owner.shared_claim_owners_for_test();
                retirement_observed_empty=std::all_of(claims.begin(),claims.end(),
                    [](const auto& claim){return claim.expired();});
            } catch(...) {retirement_observed_empty=false;}
        });
        std::weak_ptr<int> weak=capture;
        owner.fail_next_shared_q_completion_for_test(false,[capture,&calls]{++calls;});
        capture.reset();need(!weak.expired(),"armed completion callback failed to retain capture");
        need(owner.close().reusable(),"unused completion callback blocked certified drain");
        need(weak.expired() && calls.load()==0,
            "shared retirement retained unused callback or invoked it without dispatch");
        need(retirement_observed_empty.load(),
            "retirement deleter could not observe detached shared claim owners");
        need(owner.close().reusable() && calls.load()==0,"repeated close replayed unused callback");
        need(owner.shared_projection_retirement_credit_for_test()==bytes,
            "shared rendezvous close lost full bounded host credit");
        }
        need(backing.expired() && Core::shared_projection_owner_blocks_for_test()==blocks+1,
            "shared rendezvous weak tail lost bounded backing");
        backing.reset();need(Core::shared_projection_owner_blocks_for_test()==blocks,
            "shared rendezvous final weak release leaked backing");
        return 0;
    }
    if(const auto* fault=std::getenv("NINFER_TEST_ENGINE_SHARED_Q_FAILURE");fault &&
        (std::string(fault)=="1" || std::string(fault)=="producer" || std::string(fault)=="cancel_completion" ||
         std::string(fault)=="observer_failure" || std::string(fault)=="partial_scatter")) {
        const bool producer_fault=std::string(fault)=="producer";
        const bool partial_scatter=std::string(fault)=="partial_scatter";
        unsigned fault_family=15;
        if(const auto* selected=std::getenv("NINFER_TEST_ENGINE_SHARED_FAILURE_DRAFT_FAMILY");selected && *selected) {
            need(partial_scatter,"draft failure family selector requires partial_scatter mode");
            using Family=ninfer::exl3::Exl3TargetSharedFamily;
            const std::array<std::pair<std::string_view,Family>,7> families{{
                {"q",Family::draft_q},{"k",Family::draft_k},{"v",Family::draft_v},
                {"o",Family::draft_o},{"down",Family::draft_down},
                {"gate",Family::draft_gate},{"up",Family::draft_up}}};
            bool found=false;
            for(const auto& [name,family]:families)if(name==selected) {
                fault_family=static_cast<unsigned>(family);found=true;break;
            }
            need(found,"draft failure family must be q/k/v/o/down/gate/up");
        }
        const bool cancel_completion=std::string(fault)=="cancel_completion";
        const bool observer_failure=std::string(fault)=="observer_failure";
        const std::string expected_failure=partial_scatter?
            "injected shared projection partial scatter failure":producer_fault?
            "injected shared Q producer drain failure":observer_failure?
            "injected shared completion observer failure":"injected shared Q completion failure";
        const auto* quarantine_value=std::getenv("NINFER_TEST_ENGINE_SHARED_FAILURE_QUARANTINE");
        const bool quarantine=quarantine_value && std::string(quarantine_value)=="1";
        std::array<std::weak_ptr<const void>,8> claim_owners;
        std::mutex recovery_state_mutex;
        std::shared_ptr<const ninfer::exl3::Exl3ExactHostState> recovery_state,observed_recovery_state;
        std::shared_ptr<const ninfer::exl3::Exl3DraftHostRing> recovery_ring,observed_recovery_ring;
        std::weak_ptr<const ninfer::exl3::Exl3VeriCacheRequest> recovery_source_root;
        std::weak_ptr<const ninfer::exl3::Exl3ExactHostState> recovery_source_state;
        const auto observe_recovery=[&](auto root) {
            auto detached=root->state()->detached_payload_for_test();
            need(root->state()->same_represented_payload_for_test(*detached),
                "detached target copy changed represented source payload");
            need(!root->state()->same_payload(*detached),
                "detached target copy bypassed production identity comparison");
            struct KVReference {
                int layer,first,rows;
                std::span<const std::uint16_t> key,value;
            };
            std::vector<KVReference> kv_reference;
            root->state()->visit_kv_for_test([&](int layer,int first,int rows,auto key,auto value) {
                kv_reference.push_back({layer,first,rows,key,value});
            });
            std::size_t kv_index=0;
            detached->visit_kv_for_test([&](int layer,int first,int rows,auto key,auto value) {
                need(kv_index<kv_reference.size(),"detached KV added a plane");
                const auto& expected=kv_reference[kv_index++];
                need(layer==expected.layer && first==expected.first && rows==expected.rows &&
                    std::equal(key.begin(),key.end(),expected.key.begin(),expected.key.end()) &&
                    std::equal(value.begin(),value.end(),expected.value.begin(),expected.value.end()),
                    "detached KV changed original represented values or geometry");
            });
            need(kv_index==kv_reference.size() && kv_index>0,"detached KV omitted source planes");
            need(detached->position()==root->state()->position() &&
                detached->rope_offset()==root->state()->rope_offset(),"detached position changed");
            using State=ninfer::exl3::Exl3ExactHostState;
            std::vector<std::pair<std::uintptr_t,std::size_t>> source_allocations;
            const std::array<std::shared_ptr<const State>,1> source_states{root->state()},copies{detached};
            State::visit_host_allocations(source_states,[&](const void* address,std::size_t bytes) {
                source_allocations.emplace_back(reinterpret_cast<std::uintptr_t>(address),bytes);
            },false);
            need(!source_allocations.empty(),"detached recovery allocation source NOT_EXERCISED");
            std::size_t copied_allocations=0;
            State::visit_host_allocations(copies,[&](const void* address,std::size_t bytes) {
                const auto copy_address=reinterpret_cast<std::uintptr_t>(address);
                for(const auto& [source_address,source_bytes]:source_allocations) {
                    // Subtract ordered addresses instead of overflowing end pointers.
                    const bool overlaps=copy_address>=source_address
                        ?copy_address-source_address<source_bytes
                        :source_address-copy_address<bytes;
                    need(!overlaps,"detached recovery aliases original payload storage");
                }
                ++copied_allocations;
            },false);
            need(copied_allocations>0,"detached recovery payload missing");
            auto ring=root->detached_projected_conditioning_for_test();
            need(bool(ring),"recovery projected conditioning NOT_EXERCISED");
            need(root->matches_detached_conditioning_for_test(*ring),
                "detached draft copy changed represented source conditioning");
            using Root=ninfer::exl3::Exl3VeriCacheRequest;
            using Ring=ninfer::exl3::Exl3DraftHostRing;
            const std::array<std::shared_ptr<const Root>,1> roots{root};
            source_allocations.clear();
            Root::visit_host_allocations(roots,[&](const void* address,std::size_t bytes) {
                source_allocations.emplace_back(reinterpret_cast<std::uintptr_t>(address),bytes);
            },false);
            const std::array<std::shared_ptr<const Ring>,1> rings{ring};
            const auto ring_bytes=Ring::visit_allocations(rings,[&](const void* address,std::size_t bytes) {
                const auto copy_address=reinterpret_cast<std::uintptr_t>(address);
                for(const auto& [source_address,source_bytes]:source_allocations) {
                    const bool overlaps=copy_address>=source_address
                        ?copy_address-source_address<source_bytes
                        :source_address-copy_address<bytes;
                    need(!overlaps,"detached draft ring aliases original request storage");
                }
            },false);
            need(ring_bytes>0,"detached draft ring payload NOT_EXERCISED");
            need(!detached->model_identity(),"detached recovery retained model identity");
            std::lock_guard lock(recovery_state_mutex);
            need(!observed_recovery_state,"duplicate recovery terminal observation");
            recovery_source_root=root;recovery_source_state=root->state();
            observed_recovery_state=std::move(detached);
            observed_recovery_ring=std::move(ring);
        };
        const auto recovery_reference_request=[&](auto& engine) {
            runtime::ResolvedRequestOptions request;
            request.execution.sampling.temperature=0;
            request.execution.requested_output_tokens=8;
            request.execution.allow_prefix_reuse=false;
            request.stop=engine.frontend().default_stop_policy();
            auto prepared=engine.frontend().prepare(prompt("Describe a quiet autumn morning."));
            const auto summary=prepared.summary();
            auto handle=engine.submit(std::move(prepared),summary,0,request,OutputConsumerMode::Aggregate,
                std::chrono::steady_clock::now()+std::chrono::minutes(5));
            return handle.wait(nullptr,{});
        };
        std::vector<TokenId> recovery_tokens;
        FinishReason recovery_finish{};
        opt.max_concurrency=2;
        {
        ninfer::exl3::Exl3EngineCore owner(opt);
        if(!quarantine) {
            owner.observe_terminal_roots_for_test(observe_recovery);
            const auto reference=recovery_reference_request(owner);
            recovery_tokens=reference.generated_token_ids;recovery_finish=reference.finish_reason;
            need(!recovery_tokens.empty() && recovery_finish!=FinishReason::Cancelled,
                "pre-fault recovery reference missing");
            owner.set_host_kv_routes_for_test(false,false);
            owner.observe_terminal_roots_for_test({});
            std::lock_guard lock(recovery_state_mutex);
            need(bool(observed_recovery_state),"pre-fault recovery state missing");
            recovery_state=std::move(observed_recovery_state);
            recovery_ring=std::move(observed_recovery_ring);
        }
        owner.set_shared_cost_policy_for_test(
            ninfer::exl3::Exl3PackedCostPolicy::unqualified_test_permit_underfilled());
        owner.set_shared_rendezvous_timeout_for_test(std::chrono::milliseconds(5));
        std::function<void()> cancel_claimed;
        std::atomic<unsigned> completion_callbacks{0};
        auto observer_capture=std::make_shared<int>(42);std::weak_ptr<int> observer_weak=observer_capture;
        bool invalid_family_refused=false;
        try{owner.fail_next_shared_q_completion_for_test(false,{},true,16);}
        catch(const std::invalid_argument&){invalid_family_refused=true;}
        need(invalid_family_refused,"shared failure accepted an unknown family");
        owner.fail_next_shared_q_completion_for_test(producer_fault,(cancel_completion || observer_failure)?std::function<void()>([&,observer_capture] {
            ++completion_callbacks;
            if(observer_failure)throw std::runtime_error("injected shared completion observer failure");
            cancel_claimed();
        }):std::function<void()>{},partial_scatter,fault_family);
        observer_capture.reset();
        bool duplicate_fault_refused=false;
        try{owner.fail_next_shared_q_completion_for_test(!producer_fault);}
        catch(const std::logic_error&){duplicate_fault_refused=true;}
        need(duplicate_fault_refused,"shared fault arm replaced an outstanding failure stage");
        runtime::ResolvedRequestOptions request;request.execution.sampling.temperature=0;
        request.execution.requested_output_tokens=64;request.stop=owner.frontend().default_stop_policy();
        auto first=owner.frontend().prepare(prompt("Write a Python function that adds two integers. Return only code."));
        auto second=owner.frontend().prepare(prompt("Write a Python function that multiplies two integers. Return only code."));
        auto queued=owner.frontend().prepare(prompt("Describe a quiet autumn morning."));
        const auto queued_summary=queued.summary();
        const auto first_summary=first.summary(),second_summary=second.summary();
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::minutes(5);
        auto a=owner.submit(std::move(first),first_summary,0,request,OutputConsumerMode::Aggregate,deadline);
        // Installed before the second request can join the pair. Its queue
        // publication synchronizes this callback with the dispatching worker.
        if(cancel_completion)cancel_claimed=a.cancellation_callback_for_test();
        auto b=owner.submit(std::move(second),second_summary,0,request,OutputConsumerMode::Aggregate,deadline);
        auto c=owner.submit(std::move(queued),queued_summary,0,request,OutputConsumerMode::Aggregate,deadline);
        bool first_failed=false,second_failed=false;
        std::string first_error="completed without failure",second_error="completed without failure";
        try{a.wait(nullptr,{});}catch(const std::runtime_error& e){first_error=e.what();first_failed=first_error.find(expected_failure)!=std::string::npos;}
        try{b.wait(nullptr,{});}catch(const std::runtime_error& e){second_error=e.what();second_failed=second_error.find(expected_failure)!=std::string::npos;}
        if(!first_failed || !second_failed)throw std::runtime_error(
            "two-consumer shared failure NOT_EXERCISED or first error replaced: first='"+
            first_error+"' second='"+second_error+"'");
        need(c.wait(nullptr,{}).finish_reason==FinishReason::Cancelled,
            "shared failure did not cancel queued independent request");
        const auto failed_work=owner.runtime_stats();
        need(completion_callbacks.load()==((cancel_completion || observer_failure)?1u:0u) && observer_weak.expired(),
            "post-scatter cancellation callback was skipped or replayed");
        need(failed_work.shared_projection_retains_claim_owners,
            "shared failure released claimed owners before final drain");
        claim_owners=owner.shared_claim_owners_for_test();
        for(const auto& retained:claim_owners)need(!retained.expired(),"failed claim lost physical backing");
        need(claim_owners[1].owner_before(claim_owners[5]) || claim_owners[5].owner_before(claim_owners[1]),
            "shared failure fixture did not retain independent rollback roots");
        need(failed_work.shared_target_projection_failures==1,"shared fault counted per consumer");
        if(producer_fault || partial_scatter)need(failed_work.shared_projection_completed_dispatches==failed_work.shared_target_projection_batches &&
            failed_work.shared_projection_completed_rows==failed_work.shared_target_projection_rows,
            "incomplete producer/scatter failure fabricated completed packed work");
        else need(failed_work.shared_projection_completed_dispatches==failed_work.shared_target_projection_batches+1 &&
            failed_work.shared_projection_completed_rows>=failed_work.shared_target_projection_rows+2,
            "completed shared numerical work disappeared after completion fault");
        bool failed_metadata_refused=false,failed_trim_refused=false;
        try{(void)owner.prefix_retention_metadata();}
        catch(const std::logic_error&){failed_metadata_refused=true;}
        try{(void)owner.trim_prefix_retention(0,{});}
        catch(const std::logic_error&){failed_trim_refused=true;}
        need(failed_metadata_refused && failed_trim_refused,
            "failed Engine accepted prefix retention access before retirement");
        need(owner.runtime_stats().shared_projection_retains_claim_owners,
            "refused retention call released failed shared claims");
        if(quarantine) {
            owner.fail_retirement_drain_for_test(static_cast<int>(cudaErrorUnknown));
            const auto closed=owner.close();
            need(!closed.reusable() && closed.phase==ninfer::exl3::Exl3RetirementPhase::quarantined,
                "failed shared final drain did not quarantine Engine");
            need(owner.runtime_stats().shared_projection_retains_claim_owners,
                "failed final drain released shared claims");
        } else {
        need(owner.close().reusable(),"successful final drain did not retire poisoned shared consumers");
        need(!owner.runtime_stats().shared_projection_retains_claim_owners,
            "certified retirement retained failed claim owners");
        bool closed_fault_refused=false;
        try{owner.fail_next_shared_q_completion_for_test(producer_fault);}
        catch(const std::logic_error&){closed_fault_refused=true;}
        need(closed_fault_refused,"closed Engine accepted a shared failure arm");
        }
        }
        if(quarantine) {
            for(const auto& retained:claim_owners)need(!retained.expired(),
                "Engine destructor released quarantined shared backing");
            bool reload_refused=false;
            try{ninfer::exl3::Exl3EngineCore replacement(opt);}
            catch(const std::runtime_error& e){reload_refused=std::string(e.what()).find("reload refused")!=std::string::npos;}
            need(reload_refused,"shared quarantine admitted replacement Engine");
        } else {
            for(const auto& retired:claim_owners)need(retired.expired(),
                "successfully retired Engine retained failed shared claim backing");
            {
                std::lock_guard lock(recovery_state_mutex);
                need(recovery_source_root.expired() && recovery_source_state.expired(),
                    "detached recovery reference retained original terminal authority");
                need(recovery_state && recovery_ring,
                    "detached recovery values did not survive original authority");
            }
            ninfer::exl3::Exl3EngineCore replacement(opt);
            replacement.observe_terminal_roots_for_test(observe_recovery);
            const auto recovered=recovery_reference_request(replacement);
            need(recovered.generated_token_ids==recovery_tokens && recovered.finish_reason==recovery_finish,
                "replacement request differs from pre-fault serial reference");
            need(!recovered.generated_token_ids.empty() && recovered.generated_token_ids.size()<=8 &&
                recovered.finish_reason!=FinishReason::Cancelled,
                "replacement request failed after certified shared retirement");
            replacement.set_host_kv_routes_for_test(false,false);
            replacement.observe_terminal_roots_for_test({});
            {
                std::lock_guard lock(recovery_state_mutex);
                need(recovery_state && observed_recovery_state &&
                    recovery_state->same_payload(*observed_recovery_state),
                    "replacement recovery changed represented KV/recurrent/logit/tap state");
                need(recovery_ring && observed_recovery_ring &&
                    recovery_ring->same_payload(*observed_recovery_ring),
                    "replacement recovery changed private projected draft conditioning");
            }
            const auto recovery_work=replacement.runtime_stats();
            need(recovery_work.shared_target_projection_failures==0 &&
                !recovery_work.shared_projection_retains_claim_owners,
                "replacement request inherited poisoned shared state");
            need(replacement.close().reusable(),
                "successful shared-failure retirement prevented replacement lifecycle");
        }
        return 0;
    }
    // Dedicated terminal process fixture: quarantine intentionally survives to
    // process exit. Never combine with subsequent model tests in this process.
    const auto* retirement_case=std::getenv("NINFER_TEST_ENGINE_RETIREMENT_CASE");
    if(retirement_case && std::string(retirement_case)=="success") {
        {ninfer::exl3::Exl3EngineCore owner(opt);need(owner.close().reusable() && owner.close().reusable(),"successful Engine close not idempotent");}
        {ninfer::exl3::Exl3EngineCore replacement(opt);need(replacement.close().reusable(),"successful Engine retirement blocked reload");}
        return 0;
    }
    if(retirement_case && std::string(retirement_case)=="failure") {
        {
            ninfer::exl3::Exl3EngineCore owner(opt);
            owner.fail_retirement_drain_for_test(static_cast<int>(cudaErrorUnknown));
            const auto first=owner.close(),again=owner.close();
            need(first.phase==ninfer::exl3::Exl3RetirementPhase::quarantined &&
                again.first_drain_error==first.first_drain_error && !again.reusable(),"failed Engine drain became reusable");
        }
        bool registry_retained=false;
        try{ninfer::exl3::Exl3HostResidentSet replacement_registry(1ULL<<30,8ULL<<30);}
        catch(const std::logic_error& e){registry_retained=std::string(e.what()).find("one resident registry")!=std::string::npos;}
        need(registry_retained,"failed Engine destroyed physical residency owner");
        bool refused=false;try{ninfer::exl3::Exl3EngineCore replacement(opt);}
        catch(const std::runtime_error& e){refused=std::string(e.what()).find("reload refused")!=std::string::npos;}
        need(refused,"quarantined Engine admitted replacement");return 0;
    }
    if(retirement_case)throw std::invalid_argument("unknown retirement test case");
    _putenv_s("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH","1");
    bool media_refused=false;try{Engine unqualified(opt);}catch(const std::invalid_argument& e){media_refused=std::string(e.what()).find("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH")!=std::string::npos;}
    _putenv_s("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH","0");
    need(media_refused,"Engine admitted unqualified media research profile");
    std::cout<<"media_research_profile_refused PASS\n";
    if(prefix16k){auto excess=opt;excess.max_concurrency=2;bool refused=false;try{Engine invalid(excess);}catch(const std::invalid_argument& e){refused=std::string(e.what()).find("physicalC1")!=std::string::npos;}need(refused,"16K device reserve admitted unqualified C2");std::cout<<"device_prefix16k_C2_refused PASS\n";}
    const bool physical2=std::getenv("NINFER_TEST_ENGINE_CONCURRENCY") && std::string(std::getenv("NINFER_TEST_ENGINE_CONCURRENCY"))=="2";
    RequestOptions request;request.execution.sampling.temperature=0;request.execution.requested_output_tokens=32;
    std::vector<TokenId> canonical;
    std::string system;
    for(int i=0;i<12;++i)system+="You are a precise programming assistant. Preserve exact variable names and whitespace in code. ";
    auto p=prompt("Write a Python function that adds two integers. Return only code.");
    p.messages.insert(p.messages.begin(),{ChatRole::System,{{MessagePartKind::Text,system}}});
    const auto prose=prompt("Explain why leaves change color in autumn in two short sentences.");
    auto thinking=request;thinking.execution.thinking.budget=1;thinking.execution.requested_output_tokens=96;
    std::vector<TokenId> prose_canonical,thinking_canonical;
    if(physical2){
        {Engine reference(opt);canonical=reference.generate(reference.prepare(p),request).generated_token_ids;
            prose_canonical=reference.generate(reference.prepare(prose),request).generated_token_ids;
            thinking_canonical=reference.generate(reference.prepare(prompt("What is 2 plus 2?",true)),thinking).generated_token_ids;}
        need(ninfer::exl3::Exl3RecurrentPinBudget::snapshot()[0]==0,"C1 oracle retirement before C2");
        opt.max_concurrency=2;
    }
    for(int reload=0;reload<2;++reload){
        {
            Engine engine(opt);
            auto prepared=engine.prepare(p);need(engine.count_tokens(p)==prepared.summary().prompt_tokens,"count mismatch");
            auto first=engine.generate(std::move(prepared),request);
            need(first.speculative.drafted_tokens>0 && !first.content.empty(),"real draft output missing");
            const auto require_complete_route=[](const GenerationResult& value) {
                // The complete proposal includes one target seed per published
                // round; drafted_tokens counts only the generated continuation.
                need(value.speculative.proposed_rows==
                        value.speculative.drafted_tokens+value.speculative.rounds &&
                    value.speculative.verified_rows>=value.speculative.committed_model_rows &&
                    value.speculative.replayed_rows<=value.speculative.verified_rows &&
                    value.speculative.committed_model_rows==
                        value.speculative.externally_visible_model_rows+
                        value.speculative.hidden_terminal_rows &&
                    value.speculative.externally_visible_model_rows==
                        value.token_accounting.visible_model_tokens &&
                    value.speculative.hidden_terminal_rows==
                        value.token_accounting.hidden_terminal_tokens &&
                    value.speculative.diagnostic_rows==
                        value.token_accounting.diagnostic_state_rows &&
                    !value.speculative.shared_projection_rows.has_value(),
                    "complete-route work/output counters lost separation");
            };
            require_complete_route(first);
            if(reload || physical2)need(first.generated_token_ids==canonical && first.reused_prompt_tokens==0,"reload/C1 oracle leaked roots or changed output");
            canonical=first.generated_token_ids;
            auto again=engine.generate(engine.prepare(p),request);
            need(again.generated_token_ids==canonical && again.reused_prompt_tokens==first.prompt.prompt_tokens,"exact input reuse failed");
            auto branch=p;branch.messages.push_back({ChatRole::Assistant,{{MessagePartKind::Text,again.content}}});
            branch.messages.push_back({ChatRole::User,{{MessagePartKind::Text,"Now change the function to multiply those two integers."}}});
            auto cached=engine.generate(engine.prepare(branch),request);
            request.execution.allow_prefix_reuse=false;
            auto uncached=engine.generate(engine.prepare(branch),request);
            request.execution.allow_prefix_reuse=true;
            need(cached.reused_prompt_tokens>64 && cached.generated_token_ids==uncached.generated_token_ids,"branched cache changed tokens");
            // Cancellation after a committed streaming delta, followed by exact context recycling.
            std::atomic<bool> cancel=false;CancellingSink sink(cancel);
            auto cancelled=engine.generate(engine.prepare(p),request,&sink,CancellationView([&]{return cancel.load();}));
            need(cancelled.finish_reason==FinishReason::Cancelled && !sink.text.empty(),"stream cancellation not observed");
            auto restored=engine.generate(engine.prepare(p),request);
            need(restored.generated_token_ids==canonical,"stream cancellation contaminated next request");
            // An abandoned handle must cancel while the next FIFO request remains usable.
            {auto abandoned=engine.submit(engine.prepare(p),request);}
            auto after_drop=engine.generate(engine.prepare(p),request);
            need(after_drop.generated_token_ids==canonical,"dropped handle contaminated context");
            auto queued=engine.submit(engine.prepare(p),request);
            auto deadline=engine.submit(engine.prepare(p),request,OutputConsumerMode::Aggregate,std::chrono::steady_clock::now()-std::chrono::milliseconds(1));
            need(queued.wait().generated_token_ids==canonical,"FIFO first output changed");
            bool expired=false;try{deadline.wait();}catch(const RequestError& e){expired=e.kind()==RequestErrorKind::QueueTimeout;}
            need(expired,"queue deadline not enforced");
            auto thought=engine.generate(engine.prepare(prompt("What is 2 plus 2?",true)),thinking);
            need(thought.thinking.applied && thought.thinking.injected_tokens>0 && thought.thinking.model_thinking_tokens==1,"thinking control failed");
            require_complete_route(thought);
            if(physical2){
                need(thought.generated_token_ids==thinking_canonical,"C2 control differs from C1");
                // All handles are submitted before waiting. Distinguishable
                // roots catch admission/acquisition routing swaps; repeat with
                // reversed membership to exercise both private resources.
                for(int round=0;round<3;++round){
                    auto a=engine.submit(engine.prepare(round%2?prose:p),request);
                    auto b=engine.submit(engine.prepare(round%2?p:prose),request);
                    auto c=engine.submit(engine.prepare(prompt("What is 2 plus 2?",true)),thinking);
                    need(a.wait().generated_token_ids==(round%2?prose_canonical:canonical),"C2 first routed result");
                    need(b.wait().generated_token_ids==(round%2?canonical:prose_canonical),"C2 second routed result");
                    need(c.wait().generated_token_ids==thinking_canonical,"C2 concurrent control result");
                }
                std::atomic<bool> peer_cancel=false;CancellingSink peer_sink(peer_cancel);
                auto cancel_handle=engine.submit(engine.prepare(p),request,OutputConsumerMode::Streaming);
                auto peer=engine.submit(engine.prepare(prose),request);
                auto dropped=engine.submit(engine.prepare(p),request);dropped={};
                auto cancelled_peer=cancel_handle.wait(&peer_sink,CancellationView([&]{return peer_cancel.load();}));
                need(cancelled_peer.finish_reason==FinishReason::Cancelled,"C2 cancel not observed");
                need(peer.wait().generated_token_ids==prose_canonical,"C2 cancellation contaminated peer");
                need(engine.generate(engine.prepare(p),request).generated_token_ids==canonical,"C2 post-cancel recycling");
                std::vector<PreparedPrompt> queued_prompts;for(int i=0;i<5;++i)queued_prompts.push_back(engine.prepare(prose));
                std::vector<GenerationHandle> held;for(int i=0;i<4;++i)held.push_back(engine.submit(std::move(queued_prompts[i]),request));
                bool overloaded=false;try{auto excess=engine.submit(std::move(queued_prompts[4]),request);}catch(const RequestError& e){overloaded=e.kind()==RequestErrorKind::Overloaded;}
                need(overloaded,"C2 pending limit not enforced");
                for(auto& handle:held)need(handle.wait().generated_token_ids==prose_canonical,"C2 bounded queue result");
                std::cout<<"physical2_distinct_requests_control_peer_cancel PASS\n";
                const bool shared_q=std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_Q") &&
                    std::string(std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_Q"))=="1";
                const auto routes=engine.runtime_stats();
                const auto* shared_pages=std::getenv("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES");
                if(shared_pages && std::string(shared_pages)=="1") {
                    need(routes.private_device_prefix_bytes==0 && routes.shared_device_prefix_bytes==0,
                        "shared page route allocated a redundant represented prefix cache");
                    need(routes.shared_device_page_fill_bytes>0 &&
                        routes.shared_device_page_copy_bytes+routes.shared_device_page_attention_bytes>0,
                        "shared device pages Engine route NOT_EXERCISED");
                    need(routes.shared_device_page_fill_bytes%ninfer::exl3::Exl3DevicePageStorage::bytes==0,
                        "shared device page fill accounting lost complete allocation extent");
                    need(routes.shared_device_page_failures==0,"successful lifecycle concealed shared page failure");
                } else need(routes.shared_device_page_fill_bytes==0 && routes.shared_device_page_copy_bytes==0 &&
                    routes.shared_device_page_fallbacks==0 && routes.shared_device_page_failures==0,
                    "default Engine entered shared device page route");
                const auto* registered_upload=std::getenv("NINFER_EXL3_ENGINE_REGISTERED_KV_UPLOAD");
                need(routes.registered_kv_upload_failures==0,"successful lifecycle concealed registered upload failure");
                if(registered_upload && std::string(registered_upload)=="1") {
                    need(routes.registered_kv_upload_bytes>0,"registered Engine upload NOT_EXERCISED");
                    need(routes.registered_kv_upload_bytes%(1024*2)==0,"registered upload lost row byte accounting");
                } else need(routes.registered_kv_upload_bytes==0 && routes.registered_kv_upload_fallbacks==0,
                    "default Engine entered registered upload route");
                const auto* shared_prefix=std::getenv("NINFER_EXL3_ENGINE_SHARED_DEVICE_PREFIX");
                if(shared_prefix && std::string(shared_prefix)=="1") {
                    need(routes.shared_device_prefix_bytes==4096ULL*1024*2*32,"shared prefix owner not charged once");
                    need(routes.shared_device_prefix_hit_bytes>0,"shared prefix reuse NOT_EXERCISED");
                } else need(routes.shared_device_prefix_bytes==0,"default prefix unexpectedly shared");
                if(shared_q) {
                    // A token-equal run that never paired is NOT an integration
                    // result. This gate must fail instead of silently passing.
                    need(routes.shared_target_projection_batches>0,"Engine shared Q NOT_EXERCISED");
                    need(routes.shared_target_projection_fallbacks>0,"Engine lone-lane Q fallback NOT_EXERCISED");
                    need(routes.shared_target_projection_rows>=2*routes.shared_target_projection_batches &&
                        routes.shared_target_projection_rows<=16*routes.shared_target_projection_batches,
                        "Engine physical row accounting");
                    need(routes.shared_target_projection_failures==0,"successful lifecycle concealed shared failure");
                    need(routes.shared_projection_contract_refusals==0,
                        "ordinary-text lifecycle produced mixed projection contracts");
                    need(routes.shared_projection_geometry_refusals==0 && routes.shared_projection_authority_refusals==0,
                        "ordinary-text lifecycle produced invalid pair geometry or stale preclaim authority");
                    need(!routes.shared_projection_retains_claim_owners,
                        "successful idle lifecycle retained a completed claim");
                    need(routes.shared_projection_completed_dispatches==routes.shared_target_projection_batches &&
                        routes.shared_projection_completed_rows==routes.shared_target_projection_rows,
                        "successful shared dispatch conservation");
                    const auto families=routes.shared_target_projection_family_batches;
                    std::uint64_t family_total=0;for(const auto count:families)family_total+=count;
                    need(family_total==routes.shared_target_projection_batches,
                        "shared target family accounting");
                    const auto* kv=std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_KV");
                    if(kv && std::string(kv)=="1")need(families[1]>0 && families[2]>0,"shared K/V NOT_EXERCISED");
                    else need(families[1]==0 && families[2]==0,"Q-only route unexpectedly shared K/V");
                    const auto* draft_q=std::getenv("NINFER_EXL3_ENGINE_SHARED_DRAFT_Q_M16");
                    if(draft_q && std::string(draft_q)=="1")need(families[3]>0,"private draft M16 Q NOT_EXERCISED");
                    else need(families[3]==0,"default draft unexpectedly shared M16");
                    const auto* output=std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_O");
                    if(output && std::string(output)=="1")need(families[4]>0,"shared attention output NOT_EXERCISED");
                    else need(families[4]==0,"default O unexpectedly shared");
                    const auto* gateup=std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_GATEUP");
                    if(gateup && std::string(gateup)=="1") {
                        need(families[5]>0 && families[6]>0,"shared gate/up NOT_EXERCISED");
                        std::uint64_t gdn=0;for(unsigned layer=0;layer<64;++layer)if(layer%4!=3)gdn+=routes.shared_target_layer_batches[layer];
                        need(gdn>0,"GDN gate/up NOT_EXERCISED");
                    }
                    else need(families[5]==0 && families[6]==0,"default gate/up unexpectedly shared");
                    const auto* down=std::getenv("NINFER_EXL3_ENGINE_SHARED_TARGET_DOWN");
                    if(down && std::string(down)=="1")need(families[7]>0,"shared large-down NOT_EXERCISED");
                    else need(families[7]==0,"default down unexpectedly shared");
                    const auto* head=std::getenv("NINFER_EXL3_ENGINE_SHARED_HEAD");
                    if(head && std::string(head)=="1")need(families[8]>0,"shared H6 head NOT_EXERCISED");
                    else need(families[8]==0,"default head unexpectedly shared");
                    const auto* draft_kv=std::getenv("NINFER_EXL3_ENGINE_SHARED_DRAFT_KV_M16");
                    if(draft_kv && std::string(draft_kv)=="1")need(families[9]>0 && families[10]>0,"private draft KV NOT_EXERCISED");
                    else need(families[9]==0 && families[10]==0,"default draft KV unexpectedly shared");
                    const auto* draft_o=std::getenv("NINFER_EXL3_ENGINE_SHARED_DRAFT_O_M16");
                    if(draft_o && std::string(draft_o)=="1")need(families[11]>0,"private draft O NOT_EXERCISED");
                    else need(families[11]==0,"default draft O unexpectedly shared");
                    const auto* draft_down=std::getenv("NINFER_EXL3_ENGINE_SHARED_DRAFT_DOWN_M16");
                    if(draft_down && std::string(draft_down)=="1")need(families[12]>0,"private draft down NOT_EXERCISED");
                    else need(families[12]==0,"default draft down unexpectedly shared");
                    const auto* draft_gateup=std::getenv("NINFER_EXL3_ENGINE_SHARED_DRAFT_GATEUP_M16");
                    if(draft_gateup && std::string(draft_gateup)=="1")
                        need(families[13]>0 && families[14]>0,"private draft gate/up NOT_EXERCISED");
                    else need(families[13]==0 && families[14]==0,"default draft gate/up unexpectedly shared");
                } else need(routes.shared_target_projection_batches==0,"default Engine unexpectedly shared Q");
                std::cout<<"engine_shared_q_batches="<<routes.shared_target_projection_batches
                    <<" fallbacks="<<routes.shared_target_projection_fallbacks
                    <<" contract_refusals="<<routes.shared_projection_contract_refusals
                    <<" geometry_refusals="<<routes.shared_projection_geometry_refusals
                    <<" authority_refusals="<<routes.shared_projection_authority_refusals<<'\n';
            }
            std::cout<<"reload="<<reload<<" cached="<<again.reused_prompt_tokens<<" branch="<<cached.reused_prompt_tokens
                <<" canonical_tokens="<<canonical.size()<<" thinking_tokens="<<thought.generated_token_ids.size()<<" PASS\n";
        }
        need(ninfer::exl3::Exl3EngineCore::execution_stream_retirement_error_for_test()==0 &&
            ninfer::exl3::Exl3EngineCore::execution_stream_retirement_slots_for_test()==0 &&
            ninfer::exl3::Exl3EngineCore::execution_stream_owner_blocks_for_test()==0,
            "public Engine teardown retained stream owners or unresolved retirement before external drain");
        cudaDeviceSynchronize();std::size_t free=0,total=0;cudaMemGetInfo(&free,&total);
        const auto pin=ninfer::exl3::Exl3RecurrentPinBudget::snapshot();
        need(pin[0]==0 && pin[2]==0,"recurrent slab survives complete Engine destruction");
        std::cout<<"graceful_reload="<<reload<<" free_device_bytes="<<free<<" total="<<total<<"\n";
        std::cout<<"recurrent_pin_live="<<pin[0]<<" peak="<<pin[1]<<" quarantined="<<pin[2]<<"\n";
    }
    std::cout<<"PASS_EXL3_ENGINE_LIFECYCLE\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
