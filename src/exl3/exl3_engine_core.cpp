#include "exl3/exl3_engine_core.h"
#include "exl3/environment_options.h"
#include "exl3/layer_buffer_retirement.h"
#include "exl3/reconstruction_control_allocator.h"
#include "exl3/output_delivery.h"
#include "exl3/engine_round_tokens.h"
#include "exl3/reserved_request_queue.h"
#include "exl3/bounded_shared_owner.h"
#include "exl3/embedded_shared_control.h"
#include "exl3/request_option_storage.h"
#include "exl3/candidate_option_compatibility.h"
#include "exl3/candidate_route_contract.h"
#include "exl3/complete_route_counters.h"
#include "exl3/reconstruction_config.h"
#include "exl3/exl3_frontend_resources.h"
#include "exl3/dflash2_execution.h"
#include "exl3/fast_device_round.h"
#include "exl3/batched_verify_coordinator.h"
#include "exl3/engine_scratch_requirements.h"
#include "exl3/resource_availability.h"
#include "exl3/engine_target_q.h"
#include "exl3/engine_greedy_packet_batch.h"
#include "exl3/registered_kv_backing.h"
#include "exl3/kv_transfer_lease.h"
#include "exl3/kv_registration_cache.h"
#include "exl3/registered_kv_upload.h"
#include "exl3/device_page_cache.h"
#include "exl3/device_prefix_cache.h"
#include "exl3/device_page_fill_command.h"
#include "exl3/device_page_copy.h"
#include "exl3/attention_stage_storage.h"
#include "exl3/device_page_readers.h"
#include "exl3/attention_stage_resources.h"
#include "exl3/native_context_extent.h"
#include "exl3/control_publication_boundary.h"
#include "exl3/host_preparation_completion.h"
#include "exl3/public_media_qualification.h"
#include "exl3/vision_model.h"
#include "exl3/request_sampling_state.h"
#include "exl3/sampling_policy.h"
#include "exl3/sampled_authority_boundary.h"
#include "exl3/sampling_control_stop_semantics.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <nlohmann/json.hpp>
#include <atomic>
#include <charconv>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <thread>
#include <shared_mutex>
#include <Windows.h>

namespace ninfer::exl3 {
namespace {
using Clock=std::chrono::steady_clock;
using Root=std::shared_ptr<const Exl3VeriCacheRequest>;
using Family=targets::qwen3_6::Frontend;
// The existing process-exclusive host registry permits only one live Engine
// bundle. After uncertain shutdown, keep that entire bundle (including registry,
// streams, roots, draft/model owners and readback storage) until process exit.
// Never run its destructors during static teardown. No retry or replacement loop.
std::atomic<void*> quarantined_engine{nullptr};
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct DraftTapStaging {
    std::array<std::uint16_t*,Exl3EngineTapRequirements::layers> pointers{};
    ~DraftTapStaging(){for(auto p:pointers)if(p)cudaFree(p);}
};
struct ExecutionStream {
    inline static std::array<std::atomic<cudaStream_t>,2> unresolved{};
    inline static std::array<std::atomic<bool>,2> occupied{};
    inline static std::atomic<int> first_destroy_error{0};
    cudaStream_t value=nullptr;
    unsigned retirement_slot=2;
    int destroy_fault_for_test=0;
    static std::size_t occupied_count() noexcept {
        std::size_t count=0;for(const auto& slot:occupied)if(slot.load())++count;return count;
    }
    explicit ExecutionStream(bool nondefault) {
        if(!nondefault)return;
        for(unsigned i=0;i<occupied.size();++i) {
            bool free=false;
            if(!occupied[i].compare_exchange_strong(free,true))continue;
            if(first_destroy_error.load()) {
                occupied[i].store(false);
                throw std::runtime_error("EXL3 unresolved execution stream retirement; reload refused");
            }
            retirement_slot=i;return;
        }
        throw std::runtime_error("execution stream retirement capacity exhausted");
    }
    ExecutionStream(const ExecutionStream&)=delete;
    ExecutionStream& operator=(const ExecutionStream&)=delete;
    // Only constructor rollback may retire explicitly: these streams have never
    // been submitted to a lane. Published streams retire at final-owner release.
    cudaError_t retire_idle() noexcept {
        if(!value){
            if(retirement_slot<occupied.size())occupied[retirement_slot].store(false);
            retirement_slot=2;return cudaSuccess;
        }
        const auto error=destroy_fault_for_test?static_cast<cudaError_t>(destroy_fault_for_test):cudaStreamDestroy(value);
        if(error==cudaSuccess){occupied[retirement_slot].store(false);retirement_slot=2;value=nullptr;return error;}
        int expected=0;first_destroy_error.compare_exchange_strong(expected,static_cast<int>(error));
        // This slot was reserved before CUDA stream creation and is never reused
        // after failure. Teardown cannot exhaust capacity or allocate storage.
        unresolved[retirement_slot].store(value);
        retirement_slot=2;value=nullptr;return error;
    }
    ~ExecutionStream(){retire_idle();}
};
std::uint64_t available(){MEMORYSTATUSEX m{};m.dwLength=sizeof(m);if(!GlobalMemoryStatusEx(&m))throw std::runtime_error("EXL3 physical memory query");return m.ullAvailPhys;}
std::uint64_t total_memory(){MEMORYSTATUSEX m{};m.dwLength=sizeof(m);if(!GlobalMemoryStatusEx(&m))throw std::runtime_error("EXL3 physical memory query");return m.ullTotalPhys;}
std::uint64_t resident_budget(const EngineOptions& options) {
    constexpr std::uint64_t reserve=8ULL<<30;
    const auto total=total_memory();
    if(total<=reserve)throw std::invalid_argument("EXL3 physical memory is below the resident reserve");
    auto result=total-reserve;
    // For the genuine native64K configuration, the declared HostKV capacity is
    // also a hard resident ceiling. Smaller configurations preserve their
    // established policy while the wider configuration fails closed.
    if(options.max_context==Exl3NativeContextExtent::native64k_tokens &&
        options.context_cache.host_kv_capacity_bytes)
        result=std::min<std::uint64_t>(result,options.context_cache.host_kv_capacity_bytes);
    return result;
}
std::uint64_t host_plan_fingerprint(std::span<const std::int64_t> tokens) noexcept {
    std::uint64_t value=1469598103934665603ULL;
    for(const auto token:tokens) {
        const auto bits=static_cast<std::uint64_t>(token);
        for(unsigned byte=0;byte<8;++byte) {
            value^=static_cast<std::uint8_t>(bits>>(byte*8));
            value*=1099511628211ULL;
        }
    }
    value^=static_cast<std::uint64_t>(tokens.size());value*=1099511628211ULL;
    return value?value:1;
}
std::optional<std::uint64_t> measured_microseconds(double milliseconds) noexcept {
    constexpr auto maximum=static_cast<double>(std::numeric_limits<std::uint64_t>::max())/1000.0;
    if(!(milliseconds>=0.0) || milliseconds>maximum)return std::nullopt;
    return static_cast<std::uint64_t>(milliseconds*1000.0);
}
void require_exact_profile(){
    const auto flag=[](const char* name){const auto* p=std::getenv(name);return p?std::string(p):std::string();};
    if(flag("NINFER_EXL3_EXACT_HOST_KV")!="1" || flag("NINFER_EXL3_OSCAR_L0_ONLY")=="1")
        throw std::invalid_argument("EXL3 Engine requires explicit ordinary FP16 host profile");
    if(!flag("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION").empty() &&
        flag("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION")!="0" && flag("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION")!="1")
        throw std::invalid_argument("K6 small-M stream reduction must be0 or1");
    if(!flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX").empty() &&
        flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX")!="0" && flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX")!="1")
        throw std::invalid_argument("represented device prefix must be0 or1");
    if(!flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS").empty()&&flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS")!="4096"&&flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS")!="16384")
        throw std::invalid_argument("EXL3 Engine device prefix row menu4096/16384");
    if(!flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_FORWARD_PUBLISH").empty() &&
        flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_FORWARD_PUBLISH")!="0" &&
        flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_FORWARD_PUBLISH")!="1")
        throw std::invalid_argument("EXL3 Engine device prefix forward publication must be0 or1");
    if(flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_FORWARD_PUBLISH")=="1" &&
        (flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX")!="1" ||
         flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS")!="16384" ||
         flag("NINFER_EXL3_SEGMENTED_DEVICE_PREFIX")=="1" ||
         flag("NINFER_EXL3_ENGINE_SHARED_DEVICE_PREFIX")=="1"))
        throw std::invalid_argument(
            "EXL3 Engine device prefix forward publication requires private unsegmented 16K prefix storage");
    if(!flag("NINFER_EXL3_EXTENDED_STREAM_REDUCTION").empty()&&flag("NINFER_EXL3_EXTENDED_STREAM_REDUCTION")!="0"&&flag("NINFER_EXL3_EXTENDED_STREAM_REDUCTION")!="1")
        throw std::invalid_argument("EXL3 Engine extended stream reduction must be0 or1");
    Exl3PublicMediaQualification::require_research_profile_disabled(
        flag("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH"));
    require_engine_exact_route({"ordinary-host-kv",Exl3NumericalRouteClass::exact,
        "host-kv-fp16","fp16",true,false},"host-kv-fp16","fp16");
    for(const auto* name:{"NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC","NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC","NINFER_EXL3_NUMERIC_ATTENTION_TILED","NINFER_EXL3_NATIVE_CONTINUATION16"})
        if(!flag(name).empty() && flag(name)!="0")require_engine_exact_route({name,
            std::string_view(name).find("NUMERIC")!=std::string_view::npos ||
                std::string_view(name).find("NATIVE_CONTINUATION16")!=std::string_view::npos
                ?Exl3NumericalRouteClass::numeric:Exl3NumericalRouteClass::exact,
            "host-kv-fp16","fp16",false,false},"host-kv-fp16","fp16");
    if(!flag("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7").empty() &&
        flag("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7")!="0")
        require_engine_exact_route({"NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7",
            Exl3NumericalRouteClass::exact,"host-kv-fp16","fp16",true,false},
            "host-kv-fp16","fp16");
}
void require_coherent_device_profile(const EngineOptions& options) {
    if((options.max_concurrency!=1 && options.max_concurrency!=2) || options.enable_vision || options.use_cuda_graph ||
       options.max_context<2048 || options.max_context>(Exl3NativeContextExtent::l0_oscar_enabled()?
           static_cast<int>(Exl3NativeContextExtent::l0_oscar_tokens):32768))
        throw std::invalid_argument(
            "coherent-device EXL3 requires C1 greedy text and context 2048..32768 (262144 with L0 OSCAR)");
    const auto equals=[](const char* name,std::string_view expected) {
        const auto* actual=std::getenv(name);
        if(!actual || std::string_view(actual)!=expected)
            throw std::invalid_argument(std::string("coherent-device EXL3 requires ")+
                name+"="+std::string(expected));
    };
    equals("NINFER_EXL3_EXACT_HOST_KV","0");
    equals("NINFER_OSCAR_EXL3","0");
    equals("NINFER_EXL3_FAST_DEVICE_KV_TRANSACTION","1");
    equals("NINFER_EXL3_DEVICE_GREEDY","1");
    equals("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL","1");
    equals("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_TILED","1");
    equals("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL_WMMA32","1");
    equals("NINFER_EXL3_FAST_LAYER_MAJOR_PREFILL","1");
    // K5 layer reconstruction (FP32-accumulated cuBLAS at live context >= 8192)
    // is a quality-gated numerics policy; both settings are admitted.
    {
        const auto* k5=std::getenv("NINFER_EXL3_FAST_LAYER_MAJOR_K5_RECONSTRUCT");
        if(!k5 || (std::string_view(k5)!="0" && std::string_view(k5)!="1"))
            throw std::invalid_argument(
                "coherent-device EXL3 requires NINFER_EXL3_FAST_LAYER_MAJOR_K5_RECONSTRUCT=0|1");
    }
    equals("NINFER_DFLASH2_PREFILL_WINDOW","1");
    equals("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SCORES","1");
    equals("NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_FUSED_FLASH_MULTIROW","1");
    equals("NINFER_EXL3_COHERENT_DOWN_K6","1");
}
Exl3PublicMediaModality public_media_modality(
    const targets::qwen3_6::PreparedPromptData& prepared) noexcept {
    bool image=false,video=false;
    for(const auto& item:prepared.vision_items) {
        if(item.modality==targets::qwen3_6::PromptModality::Image)image=true;
        else if(item.modality==targets::qwen3_6::PromptModality::Video)video=true;
        else return Exl3PublicMediaModality::unsupported;
    }
    if(image&&video)return Exl3PublicMediaModality::image_and_video;
    if(image)return Exl3PublicMediaModality::image;
    if(video)return Exl3PublicMediaModality::video;
    return Exl3PublicMediaModality::text;
}
}
struct Exl3EngineCore::Request {
    struct Storage;
    struct DeliveryLease {
        std::shared_ptr<Request> request;
        explicit DeliveryLease(std::shared_ptr<Request> value);
        ~DeliveryLease();
    };
    struct CancellationOwner {std::mutex mutex;Impl* owner=nullptr;std::mutex output_mutex;Impl* output_owner=nullptr;};
    std::weak_ptr<CancellationOwner> cancellation_owner;
    std::shared_ptr<GenerationResult> result_owner;
    std::optional<Exl3VeriCacheServingCoordinator::LogicalHostLease> result_lease;
    explicit Request(std::shared_ptr<GenerationResult> owner,runtime::ResolvedRequestOptions& request_options,
        targets::qwen3_6::PreparedPrompt& request_prompt,targets::qwen3_6::OutputSession& request_output,
        const Exl3ReadyWorkDescriptor& request_ready,std::span<const std::int64_t> request_control,
        std::uint64_t engine_epoch)
        :result_owner(std::move(owner)),prompt(request_prompt),options(request_options),output(request_output),
         ready_work(request_ready),control_ids(request_control),submitted(request_ready.submitted()),result(*result_owner) {
        if(options.execution.sampling.temperature>0 ||
           options.execution.sampling.presence_penalty!=0 ||
           options.execution.sampling.frequency_penalty!=0)
            sampling.emplace(engine_epoch,ready_work.generation(),
                ready_work.input_fingerprint(),options.execution.sampling.seed);
    }
    ~Request();
    void cancel() noexcept;
    targets::qwen3_6::PreparedPrompt& prompt;
    runtime::ResolvedRequestOptions& options;
    targets::qwen3_6::OutputSession& output;
    const Exl3ReadyWorkDescriptor& ready_work;
    std::span<const std::int64_t> control_ids;
    OutputConsumerMode consumer;
    Clock::time_point submitted,deadline;
    std::atomic<bool> cancelled{false};
    std::optional<Exl3RequestSamplingState> sampling;
    std::uint64_t host_preparation_generation=0;
    Exl3HostPreparationCompletion host_preparation;
    std::mutex mutex;std::condition_variable changed;
    struct PendingDelta {OutputChannel channel;std::size_t offset,bytes;};
    // Qwen frontend transitions once from reasoning to content. Pending output
    // therefore needs at most two inline ranges, even for an absent consumer.
    std::array<PendingDelta,2> deltas{};
    std::size_t pending_deltas=0;
    bool emitted_content=false;
    std::size_t result_text_limit=0;
    std::size_t result_token_limit=0;
    EngineRoundTokenStorage round_tokens;
    std::size_t initial_session_bytes=0;
    std::atomic<std::size_t> observed_session_bytes{0};
    // A sink may retain delivered OutputDelta values after publish returns.
    // Each such value owns one lease back to this request. The fixed two-slot
    // ceiling applies backpressure before another string is materialized.
    std::atomic<std::size_t> outstanding_delivery_batches{0};
    Exl3ControlPublicationBoundary publication_boundary;
    bool output_extent_fault=false;
    std::atomic<bool> window_publication_fault{false};
    void require_output_storage() {
        if(std::exchange(output_extent_fault,false))output.exceed_private_storage_for_test(initial_session_bytes);
        const auto retained=output.retained_storage_bytes();
        if(retained>initial_session_bytes)
            throw std::logic_error("Engine output session exceeds reserved private extent");
        auto previous=observed_session_bytes.load();
        while(previous<retained && !observed_session_bytes.compare_exchange_weak(previous,retained)){}
    }
    void require_result_slots(std::size_t count) const {
        const auto used=result.generated_token_ids.size();
        if(!output_result_slots_available(used,result_token_limit,result.generated_token_ids.capacity(),count))
            throw std::logic_error("Engine output exceeds reserved token result slots");
    }
    std::size_t delivery_byte_limit=output_delivery_byte_limit;
    GenerationResult& result;
    bool done=false;
    std::exception_ptr failure;
};
// The inventory retains raw backing, never the live Request shared_ptr. Its
// explicit deleter ends Request lifetime even while the inventory owns storage.
struct Exl3EngineCore::Request::Storage {
    GenerationResult result;
    runtime::ResolvedRequestOptions options;
    targets::qwen3_6::PreparedPrompt prompt;
    targets::qwen3_6::OutputSession output;
    std::optional<Exl3ReadyWorkDescriptor> ready_work;
    std::vector<std::int64_t> control_ids;
    alignas(Request) std::byte request_bytes[sizeof(Request)];
    EmbeddedSharedControl request_control;
};
struct Exl3EngineCore::Impl {
    std::shared_ptr<Request::CancellationOwner> cancellation_owner;
    bool coherent_device=false;
    // Admission of the ordinary graph families (each still selected by its
    // own switch) into the coherent-device context.  Qualified for GDN-segment
    // and full-layer graphs; NINFER_EXL3_COHERENT_ORDINARY_GRAPHS=0 selects the
    // eager fallback.
    bool coherent_ordinary_graphs=false;
    EngineOptions options;
    bool public_media_enabled=false;
    std::optional<Exl3VerifierHorizonPolicy::CostMenu> verifier_cost_menu;
    std::optional<Exl3SuffixProposer::SelectionLimits> suffix_selection_limits;
    bool conditional_b8=false;
    // Package-owned pinned resources, immutable target weights, then physical
    // draft/context ownership. Cached roots and response records outlive turns.
    targets::qwen3_6::FrontendResources resources;
    Family frontend;
    ModelSamplingDefaults defaults;
    std::shared_ptr<Exl3TextModel> target;
    std::shared_ptr<Exl3VisionModel> vision_model;
    std::shared_ptr<Exl3VisionContext> vision;
    std::array<std::shared_ptr<ExecutionStream>,2> streams;
    std::array<std::shared_ptr<Exl3Dflash2DraftModel>,2> drafts;
    std::array<std::shared_ptr<Exl3Dflash2Execution>,2> lanes;
    std::array<std::shared_ptr<DraftTapStaging>,2> staging_owners;
    std::shared_ptr<Exl3EngineTargetQ> shared_q;
    std::shared_ptr<Exl3EngineGreedyPacketBatch> greedy_packet_batch;
    std::shared_ptr<Exl3KVRegistrationCache> kv_registrations;
    std::optional<Exl3KVRegistrationCache::ExternalRead> registration_reader_for_test;
    Exl3RegisteredKVUpload::LanePool kv_uploads;
    std::atomic<bool> registered_upload_enabled{true};
    std::shared_ptr<Exl3DevicePageCache> shared_pages;
    std::atomic<bool> shared_pages_enabled{true};
    std::shared_ptr<const void> foreign_page_ancestor_for_test;
    std::atomic<unsigned> shared_page_fault{0};
    std::function<void()> shared_page_before_failure;
    std::atomic<bool> shared_page_fill_observer_armed{false};
    std::function<void()> shared_page_fill_observer;
    Exl3KVUploadProvider attention_provider_for_test() {
        Exl3KVUploadProvider provider;
        auto stage=shared_page_fault.load();
        if(stage<7 || stage>8 || !shared_page_fault.compare_exchange_strong(stage,0))return provider;
        if(auto callback=std::move(shared_page_before_failure))callback();
        if(stage==7)provider.record=[](cudaEvent_t,cudaStream_t) noexcept {return cudaErrorUnknown;};
        else provider.wait=[](cudaEvent_t) noexcept {return cudaErrorUnknown;};
        return provider;
    }
    Exl3KVUploadProvider page_provider_for_test(bool fill) {
        Exl3KVUploadProvider provider;
        std::function<void()> fill_observer;
        if(fill && shared_page_fill_observer_armed.load(std::memory_order_acquire)) {
            std::lock_guard lock(mutex);
            if(shared_page_fill_observer_armed.exchange(false,std::memory_order_acq_rel))
                fill_observer=std::move(shared_page_fill_observer);
        }
        if(fill_observer)fill_observer();
        auto stage=shared_page_fault.load();
        if((fill && stage>=1 && stage<=3) || (!fill && stage>=4 && stage<=6)) {
            if(!shared_page_fault.compare_exchange_strong(stage,0))stage=0;
        } else stage=0;
        if(stage)if(auto callback=std::move(shared_page_before_failure))callback();
        if(stage>3)stage-=3;
        if(stage==1)provider.copy=[](void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t) noexcept {return cudaErrorUnknown;};
        if(stage==2)provider.record=[](cudaEvent_t,cudaStream_t) noexcept {return cudaErrorUnknown;};
        if(stage==3)provider.wait=[](cudaEvent_t) noexcept {return cudaErrorUnknown;};
        return provider;
    }
    std::array<std::shared_ptr<Exl3DevicePageCopy>,2> page_copies;
    std::array<std::array<std::shared_ptr<Exl3DevicePageAttention>,Exl3AttentionPageRanges::capacity>,2> page_attention;
    Exl3VeriCacheServingIdentity identity;
    Exl3VeriCacheServingPrefixCache cache;
    Exl3VeriCacheServingCoordinator coordinator;
    LoadSummary load;
    mutable std::mutex mutex;
    std::mutex admission_mutex;
    std::condition_variable changed;
    ReservedRequestQueue<Request> queue;
    std::array<std::shared_ptr<Request>,2> active;
    Exl3ReadyWorkSnapshotAuthority ready_work_authority;
    struct LaneFrontier {
        std::weak_ptr<const Exl3VeriCacheRequest> root;
        std::uint64_t request_generation=0;
        Exl3ReadyLocalityCost locality;
    };
    std::array<LaneFrontier,2> lane_frontiers;
    // True from the first request-local prefill operation through the last
    // numerical/transfer cleanup in execute().  It is an overlap witness only;
    // it never grants another request access to this lane's state.
    std::array<std::atomic<bool>,2> numerical_active{};
    std::uint64_t next_host_preparation_generation=1;
    bool stopping=false,failed=false;
    unsigned result_allocation_fault=0;
    std::atomic<unsigned> recurrent_commit_fault{0};
    std::atomic<unsigned> recurrent_commit_fault_hit{0};
    std::atomic<unsigned> recurrent_cancel_after_factory{0};
    std::atomic<unsigned> recurrent_cancel_after_factory_hit{0};
    std::atomic<bool> cancel_active_snapshot_for_test{false};
    std::atomic<unsigned> conditional_cancel_stage{0},conditional_cancel_hit{0};
    std::atomic<unsigned> request_metadata_fault{0},request_metadata_fault_hit{0};
    std::atomic<unsigned> request_host_payload_fault{0},request_host_payload_fault_hit{0};
    std::atomic<bool> window_publication_fault{false};
    bool output_extent_fault=false;
    bool concurrent_prefix_preparation=false;
      bool declared_prefix_preparation=false;
      bool request_affinity=false;
      Exl3EngineCore::ReadyCopyDemandProvider copy_demand_provider;
      bool ready_decision_trace_enabled=false;
      Exl3ReadyDecisionTrace ready_decision_trace;
    std::size_t outstanding=0;
    RuntimeStats stats;
    MemorySummary memory;
    std::array<std::thread,2> workers;
    // Batched multi-agent rounds of the two coherent-device lanes
    // (NINFER_EXL3_BATCHED_ROUNDS, default on for C2).
    std::unique_ptr<Exl3BatchedVerifyCoordinator> batched_verify;
    // Concurrent coherent-device lanes: prompt ingestion runs alone on the
    // device (exclusive; process-wide prefill scratch, and DFlash2 rounds are
    // not qualified to overlap another lane's ingestion), decode rounds of all
    // lanes run together (shared).
    std::shared_mutex coherent_prefill_mutex;
    Exl3RetirementState retirement;
    int retirement_fault=0;
    std::exception_ptr first_execution_failure;
    std::function<void(Root)> terminal_root_observer;
    std::function<void(Root)> acquired_root_observer;
    std::function<void()> request_start_observer;
    std::function<void(std::size_t,bool)> lane_assignment_observer;
    std::function<void(std::size_t,int)> prefix_preparation_observer;
    std::function<void(std::size_t,std::uint64_t,bool)> host_preparation_observer;
    std::atomic<unsigned> attachment_cancel_phase{0},attachment_cancel_hit{0};
    // One-based encoded target layer; zero means no deterministic mid-restore cancellation.
    std::atomic<unsigned> attachment_restore_cancel_layer{0},attachment_restore_cancel_hit{0};

    explicit Impl(const EngineOptions& value,unsigned shared_allocation_fault,unsigned draft_clone_fault,unsigned first_draft_fault,unsigned context_startup_fault,
        Exl3DeviceAvailability::Provider supplied_availability):
        coherent_device(value.exl3_package->round_implementation==
            Exl3RoundImplementation::CoherentDevice),
        coherent_ordinary_graphs(coherent_device && [] {
            const char* value=std::getenv("NINFER_EXL3_COHERENT_ORDINARY_GRAPHS");
            if(value && std::string_view(value)!="0" && std::string_view(value)!="1")
                throw std::invalid_argument("coherent ordinary graphs must be 0 or 1");
            return !value || std::string_view(value)=="1";
        }()),options(value),
        public_media_enabled(
            !coherent_device && Exl3PublicMediaQualification::evaluate_current(
                Exl3PublicMediaModality::image).media_allowed() &&
            Exl3PublicMediaQualification::evaluate_current(
                Exl3PublicMediaModality::video).media_allowed()),
        resources(load_pinned_frontend_resources(value.exl3_package->target_directory)),
        frontend(targets::qwen3_6::make_frontend(resources,targets::qwen3_6::FrontendOptions{
            .vision_enabled=public_media_enabled,.max_context=value.max_context,
            .max_cache_markers_per_request=value.context_cache.max_cache_markers_per_request.value_or(4)})),
        identity{"exl3-engine-epoch-1","SC_6.00bpw_H6_V6","0997f410-c3cf9e34",
            coherent_device?"coherent-device-P1-K6-B8-greedy":"ordinary-FP16-B8-greedy",
            public_media_enabled?"text+qualified-v6-media":"text"},
        cache(Exl3VeriCacheServingPrefixCache::Policy{std::max(1U,value.context_cache.max_shared_prefixes.value_or(1)),64,std::max<std::size_t>(1,value.context_cache.host_kv_capacity_bytes),8ULL<<30},identity),
        coordinator(cache,{value.max_concurrency,value.max_concurrency,resident_budget(value),8ULL<<30}) {
        if(coherent_device)require_coherent_device_profile(value);
        else require_exact_profile();
        const auto flag=[](const char* name) {
            const auto* value=std::getenv(name);
            if(value && std::string_view(value)!="0" && std::string_view(value)!="1")
                throw std::invalid_argument(std::string(name)+" must be0 or1");
            return value && std::string_view(value)=="1";
        };
        const bool native64k=flag("NINFER_EXL3_EXTENDED_CONTEXT_64K");
        const bool candidate128k=flag("NINFER_EXL3_EXTENDED_CONTEXT_128K");
        const bool exact_host=flag("NINFER_EXL3_EXACT_HOST_KV");
        const bool oscar_only=flag("NINFER_EXL3_OSCAR_L0_ONLY");
        const bool device_prefix=flag("NINFER_EXL3_HOST_KV_DEVICE_PREFIX");
        Exl3NativeContextExtent::refuse_candidate128k_execution(value.max_context,
            native64k,candidate128k,exact_host,oscar_only,value.max_concurrency,
            value.context_cache.host_kv_capacity_bytes,device_prefix);
        Exl3NativeContextExtent::require_native64k_engine(value.max_context,
            native64k,candidate128k,exact_host,oscar_only,
            value.max_concurrency,value.context_cache.host_kv_capacity_bytes);
        conditional_b8 = read_binary_option("NINFER_EXL3_ENGINE_CONDITIONAL_B8",
            "NINFER_EXL3_ENGINE_CONDITIONAL_B8 must be 0 or 1");
        const bool device_taps_only=flag("NINFER_EXL3_COMPACT_DEVICE_TAPS_ONLY");
        if(device_taps_only &&
           (!flag("NINFER_EXL3_COMMITTED_TAP_D2D") ||
            !flag("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING")))
            throw std::invalid_argument(
                "device-only committed taps require direct D2D tap staging");
        if(device_taps_only && conditional_b8)
            throw std::invalid_argument(
                "device-only committed taps are incompatible with conditional B8");
        if(device_taps_only && flag("NINFER_EXL3_STAGED_B8_VERIFIER"))
            throw std::invalid_argument(
                "device-only committed taps are incompatible with staged B8");
        const bool reconstruction_fallback=exl3_reconstruction_budget_fallback(
            std::getenv("NINFER_EXL3_RECONSTRUCTION_BUDGET_FALLBACK"));
        concurrent_prefix_preparation = read_binary_option("NINFER_EXL3_ENGINE_SHARED_PREPARATION",
            "NINFER_EXL3_ENGINE_SHARED_PREPARATION must be 0 or 1");
        declared_prefix_preparation = read_binary_option("NINFER_EXL3_ENGINE_DECLARED_PREFIX",
            "NINFER_EXL3_ENGINE_DECLARED_PREFIX must be 0 or 1");
        request_affinity = read_binary_option("NINFER_EXL3_REQUEST_AFFINITY",
            "NINFER_EXL3_REQUEST_AFFINITY must be 0 or 1");
        const auto* preserve_option=std::getenv("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT");
        if(request_affinity && (!preserve_option || std::string_view(preserve_option)!="1"))
            throw std::invalid_argument("request affinity requires acquired-root preservation");
        const bool share_prefix = read_binary_option("NINFER_EXL3_ENGINE_SHARED_DEVICE_PREFIX",
            "Engine shared device prefix must be0 or1") && value.max_concurrency==2;
        if(share_prefix) {
            const auto* cache=std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX");
            if(!cache || std::string_view(cache)!="1")
                throw std::invalid_argument("Engine shared prefix requires represented device prefix option");
        }
        const bool share_q = read_binary_option("NINFER_EXL3_ENGINE_SHARED_TARGET_Q",
            "Engine shared target Q must be0 or1") && value.max_concurrency==2;
        const bool batch_greedy = read_binary_option("NINFER_EXL3_ENGINE_BATCHED_GREEDY_PACKET",
            "Engine batched greedy packet must be0 or1");
        const bool device_greedy=exl3_device_greedy_enabled();
        if(batch_greedy && value.max_concurrency!=2)
            throw std::invalid_argument("Engine batched greedy packet requires physical C2");
        if(batch_greedy) {
            if(!device_greedy)
                throw std::invalid_argument("Engine batched greedy packet requires device greedy");
        }
        const bool share_kv = read_binary_option("NINFER_EXL3_ENGINE_SHARED_TARGET_KV",
            "Engine shared target KV must be0 or1") && value.max_concurrency==2;
        const bool share_o = read_binary_option("NINFER_EXL3_ENGINE_SHARED_TARGET_O",
            "Engine shared target O must be0 or1") && value.max_concurrency==2;
        const bool share_gateup = read_binary_option("NINFER_EXL3_ENGINE_SHARED_TARGET_GATEUP",
            "Engine shared gate/up must be0 or1") && value.max_concurrency==2;
        if(share_gateup) {
            const auto* k5=std::getenv("NINFER_EXL3_TARGET_GATEUP_K5_SMALL_M");
            const auto* k6=std::getenv("NINFER_EXL3_TARGET_GATEUP_SMALL_M");
            const auto* k7=std::getenv("NINFER_EXL3_TARGET_GATEUP_K7_SMALL_M");
            if(!share_q || ((!k5 || std::string_view(k5)!="1") &&
                (!k6 || std::string_view(k6)!="1") && (!k7 || std::string_view(k7)!="1")))
                throw std::invalid_argument("Engine shared gate/up requires shared Q and explicit gate/up small-M route");
        }
        const bool share_down = read_binary_option("NINFER_EXL3_ENGINE_SHARED_TARGET_DOWN",
            "Engine shared down must be0 or1") && value.max_concurrency==2;
        if(share_down) {
            const auto* k6=std::getenv("NINFER_EXL3_TARGET_DOWN_SMALL_M");
            const auto* k7=std::getenv("NINFER_EXL3_TARGET_DOWN_K7_SMALL_M");
            if(!share_q || ((!k6 || std::string_view(k6)!="1") && (!k7 || std::string_view(k7)!="1")))
                throw std::invalid_argument("Engine shared down requires shared Q and explicit down small-M route");
            // Family-specific 16-k6/16-k7 selections fall back for the other K.
            // The workspace retains the existing split and async-A policy.
        }
        const bool share_head = read_binary_option("NINFER_EXL3_ENGINE_SHARED_HEAD",
            "Engine shared head must be0 or1") && value.max_concurrency==2;
        const auto* h6=std::getenv("NINFER_EXL3_H6_SMALL_M");
        if(share_head && (!share_q || (h6 && std::string_view(h6)=="0")))
            throw std::invalid_argument("Engine shared head requires shared Q and enabled H6 small-M");
        const bool share_gather = read_binary_option("NINFER_EXL3_ENGINE_SHARED_GATHER_REUSE",
            "Engine gather reuse must be0 or1") && value.max_concurrency==2;
        if(share_o) {
            const auto* k6=std::getenv("NINFER_EXL3_TARGET_O_K6_SMALL_M");
            const auto* k7=std::getenv("NINFER_EXL3_TARGET_O_K7_SMALL_M");
            if(!share_q || ((!k6 || std::string_view(k6)!="1") && (!k7 || std::string_view(k7)!="1")))
                throw std::invalid_argument("Engine shared O requires shared Q owner and explicit O small-M route");
        }
        const bool share_draft = read_binary_option("NINFER_EXL3_ENGINE_SHARED_DRAFT_Q_M16",
            "Engine shared draft Q M16 must be0 or1") && value.max_concurrency==2;
        if(share_draft) {
            const auto* route=std::getenv("NINFER_EXL3_DRAFT_SMALL_M");
            if(route && std::string_view(route)!="1")
                throw std::invalid_argument("Engine shared draft requires enabled draft small-M route");
        }
        // Target and private-draft families share one bounded rendezvous owner,
        // but neither family is a prerequisite for executing the other.
        const bool share_projection_owner=share_q || share_draft;
        const bool share_draft_kv = read_binary_option("NINFER_EXL3_ENGINE_SHARED_DRAFT_KV_M16",
            "Engine draft KV must be0 or1") && value.max_concurrency==2;
        if(share_draft_kv && !share_draft)throw std::invalid_argument("Engine draft KV requires shared draft Q M16");
        const bool share_draft_o = read_binary_option("NINFER_EXL3_ENGINE_SHARED_DRAFT_O_M16",
            "Engine draft O must be0 or1") && value.max_concurrency==2;
        if(share_draft_o && !share_draft)throw std::invalid_argument("Engine draft O requires shared draft Q M16");
        const bool share_draft_down = read_binary_option("NINFER_EXL3_ENGINE_SHARED_DRAFT_DOWN_M16",
            "Engine draft down must be0 or1") && value.max_concurrency==2;
        if(share_draft_down && !share_draft)throw std::invalid_argument("Engine draft down requires shared draft Q M16");
        const bool share_draft_gateup = read_binary_option("NINFER_EXL3_ENGINE_SHARED_DRAFT_GATEUP_M16",
            "Engine shared draft gate/up M16 option requires 0 or 1");
        if(share_draft_gateup && !share_draft)
            throw std::invalid_argument("Engine draft gate/up requires C2 shared draft Q M16");
        if(share_gather && !share_gateup && !share_draft_gateup)
            throw std::invalid_argument("Engine gather reuse requires shared gate/up");
        if(share_kv) {
            const auto* route=std::getenv("NINFER_EXL3_TARGET_KV_SMALL_M");
            if(!share_q || !route || std::string_view(route)!="1")
                throw std::invalid_argument("Engine shared KV requires shared Q and explicit KV small-M route");
        }
        if(share_q) {
            for(const auto* name:{"NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION","NINFER_EXL3_EXTENDED_STREAM_REDUCTION"}) {
                const auto* selected=std::getenv(name);
                if(selected && std::string_view(selected)!="0")
                    throw std::invalid_argument(std::string("Engine shared target route conflict: ")+name);
            }
            const auto* route=std::getenv("NINFER_EXL3_TARGET_Q_K6_SMALL_M");
            if(!route || std::string_view(route)!="1")
                throw std::invalid_argument("Engine shared Q requires explicit Q/K6 small-M route");
        }
        if(shared_allocation_fault && (!share_projection_owner || shared_allocation_fault>4U+unsigned(share_kv)+unsigned(share_draft)+unsigned(share_o)+unsigned(share_gateup)+unsigned(share_down)+unsigned(share_head)+unsigned(share_draft_kv)+unsigned(share_draft_o)+unsigned(share_draft_down)+unsigned(share_draft_gateup)))
            throw std::invalid_argument("shared allocation fault outside selected startup menu");
        std::uint64_t device_reserve=0;
        const bool registered_upload = read_binary_option("NINFER_EXL3_ENGINE_REGISTERED_KV_UPLOAD",
            "Engine registered KV upload must be0 or1");
        const auto* registered_upload_slots_option=
            std::getenv("NINFER_EXL3_EXACT_HOST_KV_PINNED_H2D_SLOTS");
        if(registered_upload_slots_option &&
           std::string_view(registered_upload_slots_option)!="2" &&
           std::string_view(registered_upload_slots_option)!="32")
            throw std::invalid_argument(
                "exact HostKV pinned H2D slots must be 2 or 32");
        const unsigned registered_upload_slots=
            registered_upload_slots_option &&
            std::string_view(registered_upload_slots_option)=="32"?32:2;
        const bool share_pages = read_binary_option("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGES",
            "Engine shared device pages must be0 or1");
        const bool share_attention = read_binary_option("NINFER_EXL3_ENGINE_SHARED_DEVICE_PAGE_ATTENTION",
            "shared page attention option must be 0 or 1");
        const bool attention_staging = read_binary_option("NINFER_EXL3_ENGINE_ATTENTION_STAGING",
            "attention staging option must be 0 or 1");
        if(attention_staging && (share_pages || share_prefix))
            throw std::invalid_argument("attention staging and shared cache routes are exclusive");
        if(share_attention && !share_pages)throw std::invalid_argument("shared page attention requires shared device pages");
        if(share_pages && share_prefix)
            throw std::invalid_argument("shared pages and whole-prefix sharing are exclusive Engine routes");
        if(const auto* requested=std::getenv("NINFER_EXL3_ENGINE_DEVICE_RESERVE_BYTES");requested && *requested) {
            const std::string_view text(requested);
            const auto parsed=std::from_chars(text.data(),text.data()+text.size(),device_reserve);
            if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
                throw std::invalid_argument("Engine device reserve requires unsigned bytes");
        }
        const auto* prefix_rows=std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS");
        std::uint32_t parsed_prefix_rows=0;
        if(prefix_rows && *prefix_rows) {
            const std::string_view text(prefix_rows);
            const auto parsed=std::from_chars(text.data(),text.data()+text.size(),
                parsed_prefix_rows);
            if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
                throw std::invalid_argument(
                    "EXL3 represented prefix rows require an unsigned integer");
        }
        const auto selected=[](const char* name) {
            const auto* value=std::getenv(name);return value && std::string_view(value)=="1";
        };
        if(coherent_device) {
            if(native64k || candidate128k ||
               selected("NINFER_EXL3_NATIVE_CONTINUATION16") || device_prefix ||
               share_prefix || registered_upload || share_pages || share_attention ||
               attention_staging || share_projection_owner || share_kv || share_o ||
               share_gateup || share_down || share_head || share_gather ||
               share_draft_kv || share_draft_o || share_draft_down ||
               share_draft_gateup || batch_greedy || conditional_b8 ||
               concurrent_prefix_preparation || declared_prefix_preparation ||
               request_affinity)
                throw std::invalid_argument(
                    "coherent-device EXL3 cannot compose with host-KV Engine candidates");
        } else {
            require_candidate_option_compatibility({
                value.max_concurrency,value.max_context,
                parsed_prefix_rows,
                exact_host,oscar_only,native64k,candidate128k,
                selected("NINFER_EXL3_NATIVE_CONTINUATION16"),device_prefix,share_prefix,
                registered_upload,share_pages,share_attention,attention_staging,
                share_q,share_kv,share_o,share_gateup,share_down,share_head,share_gather,
                selected("NINFER_EXL3_K6_SMALL_M_STREAM_REDUCTION") ||
                    selected("NINFER_EXL3_EXTENDED_STREAM_REDUCTION"),
                share_draft,share_draft_kv,share_draft_o,share_draft_down,
                share_draft_gateup,device_greedy,batch_greedy});
        }
        const auto host_metadata_limit=exl3_host_metadata_limit(std::getenv("NINFER_EXL3_ENGINE_HOST_METADATA_LIMIT_BYTES"));
        if(!share_pages && prefix_rows&&std::string(prefix_rows)=="16384"&&value.max_concurrency!=1)
            throw std::invalid_argument("EXL3 Engine device prefix rows16384 require physicalC1");
        if((value.max_concurrency!=1 && value.max_concurrency!=2) || value.enable_vision || value.use_cuda_graph || value.hierarchical_vericache.enabled ||
            !value.max_pending_requests || !value.pending_timeout_ms || value.max_context<16 ||
            value.speculative.backend!=SpeculativeBackend::DFlash2 || value.speculative.draft_tokens!=7)
            throw std::invalid_argument("EXL3 text requires C1 or explicit C2/eager/DFlash2-K7 and disabled Vision");
        const auto* private_prefix_option=std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX");
        // Attention staging has independent reserved storage.  The staged
        // producer may seed that storage from this lane-private prefix and
        // upload only the uncached suffix; both candidates remain default-off.
        const bool private_prefix_requested=!share_pages && private_prefix_option &&
            std::string_view(private_prefix_option)=="1";
        if(private_prefix_requested && prefix_rows && std::string_view(prefix_rows)!="4096" &&
            std::string_view(prefix_rows)!="16384")
            throw std::invalid_argument("device prefix row menu4096/16384");
        const int private_prefix_capacity=prefix_rows && std::string_view(prefix_rows)=="16384"?16384:4096;
        const auto* segmented_prefix_option=std::getenv("NINFER_EXL3_SEGMENTED_DEVICE_PREFIX");
        const bool segmented_prefix_requested=segmented_prefix_option && std::string_view(segmented_prefix_option)=="1";
        const bool shared_workspace_unwind_fault=share_projection_owner && context_startup_fault==83 && shared_allocation_fault==2;
        const bool valid_startup_fault=context_startup_fault==128 || context_startup_fault==31 || context_startup_fault<=11*value.max_concurrency ||
            (share_pages && (context_startup_fault==125 || context_startup_fault==126)) ||
            (registered_upload && context_startup_fault==127) ||
            (private_prefix_requested && (context_startup_fault==123 || context_startup_fault==124)) ||
            (value.max_concurrency==1 && (context_startup_fault>=86 && context_startup_fault<=100)) ||
            (value.max_concurrency==2 && context_startup_fault>=79 && context_startup_fault<=82) ||
            shared_workspace_unwind_fault ||
            (share_projection_owner && !shared_allocation_fault && (context_startup_fault==101 || context_startup_fault==102)) ||
            (share_projection_owner && !shared_allocation_fault && context_startup_fault>=103 && context_startup_fault<=122) ||
            (share_projection_owner && !shared_allocation_fault && (context_startup_fault==84 || context_startup_fault==85)) ||
            (context_startup_fault>=71 && context_startup_fault<=78) ||
            (context_startup_fault>=61 && context_startup_fault<=64) ||
            (context_startup_fault>=51 && context_startup_fault<=53) ||
            (context_startup_fault>=41 && context_startup_fault<41+5*value.max_concurrency) ||
            (context_startup_fault>=23 && context_startup_fault<23+3*value.max_concurrency);
        if(context_startup_fault && (!valid_startup_fault ||
            (shared_allocation_fault && !shared_workspace_unwind_fault) || draft_clone_fault || first_draft_fault))
            throw std::invalid_argument("unsupported context startup fault stage or combination");
        const auto begin=Clock::now();
        if(first_draft_fault && (first_draft_fault>8 || draft_clone_fault || shared_allocation_fault))
            throw std::invalid_argument("first draft fault requires isolated stage1..8");
        if(draft_clone_fault && (value.max_concurrency!=2 || draft_clone_fault>8 || shared_allocation_fault))
            throw std::invalid_argument("draft clone fault requires isolated C2 stage1..8");
        using Domain=Exl3ResourceInventory::Domain;
        using Queue=ReservedRequestQueue<Request>;
        const auto queue_slots=static_cast<std::uint64_t>(value.max_pending_requests)+value.max_concurrency;
        Exl3ResourceInventory::Requirement queue_required;queue_required.configuration=0x52514555455545;
        queue_required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Queue::Storage>());
        queue_required.add(Domain::host_metadata,queue_slots,sizeof(Queue::Entry));
        const auto queue_bytes=queue_required.units[static_cast<unsigned>(Domain::host_metadata)];
        const auto coordinator_metadata_bytes=coordinator.metadata_requirement().units[static_cast<unsigned>(Domain::host_metadata)];
        Exl3ResourceInventory::Requirement cancellation_required;
        cancellation_required.configuration=0x43414e43454c;
        cancellation_required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Request::CancellationOwner>());
        Exl3ResourceInventory::Requirement stream_required;
        stream_required.configuration=0x53545245414D4F57;
        stream_required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<ExecutionStream>());
        auto mandatory_metadata=queue_required;
        mandatory_metadata.add(Domain::host_metadata,1,coordinator_metadata_bytes);
        mandatory_metadata.add(Domain::host_metadata,1,
            cancellation_required.units[static_cast<unsigned>(Domain::host_metadata)]);
        mandatory_metadata.add(Domain::host_metadata,value.max_concurrency,
            stream_required.units[static_cast<unsigned>(Domain::host_metadata)]);
        // Every fresh eager lane constructs these owners before serving requests.
        // Their size declarations are also used by the reserved constructors.
        mandatory_metadata.add(Domain::host_metadata,value.max_concurrency,
            Exl3TextContext::fixed_owner_metadata_bytes());
        mandatory_metadata.add(Domain::host_metadata,value.max_concurrency,
            Exl3TextContext::continuation_owner_metadata_bytes());
        if(attention_staging)
            mandatory_metadata.add(Domain::host_metadata,value.max_concurrency,
                Exl3AttentionStageResources::metadata_bytes());
        if(mandatory_metadata.units[static_cast<unsigned>(Domain::host_metadata)]>host_metadata_limit)
            throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
        // Diagnostic stop before model/lane allocation, after the real budget gate.
        if(context_startup_fault==128)
            throw std::runtime_error("Engine mandatory metadata floor accepted:"+
                std::to_string(mandatory_metadata.units[static_cast<unsigned>(Domain::host_metadata)]));
        // Reuse these exact declarations below. This is only a known mandatory
        // floor, including mandatory opted-in attention metadata; weights,
        // other lane resources and runtime owners still need full admission.
        const auto cfg=nlohmann::json::parse(resources.generation_config_json);
        SamplingPreset preset;
        preset.temperature=cfg.value("temperature",1.0F);preset.top_k=cfg.value("top_k",20);preset.top_p=cfg.value("top_p",0.95F);
        defaults={preset,preset};
        Exl3LoadOptions loading;loading.verified_dual_manifest=value.exl3_package->verified_dual_manifest;
        // Deliberate pre-authority construction boundary: target/draft immutable
        // weights (and qualified vision weights) are allocated by their artifact
        // loaders before an Engine coordinator exists. Their authoritative byte
        // extents come from the completed loader objects and are bound exactly
        // once below; this caller must not guess a reservation from file size or
        // duplicate loader arithmetic. Every allocation after these immutable
        // roots is planned and transacted through `coordinator`. A future loader
        // preflight API must return the same owner/domain/extent manifest before
        // this boundary before weight construction can join startup reservation.
        target=Exl3TextModel::load(value.exl3_package->target_directory,value.max_context,loading);
        if(public_media_enabled) {
            vision_model=make_bounded_shared<Exl3VisionModel>(
                value.exl3_package->target_directory);
            vision=make_bounded_shared<Exl3VisionContext>(*vision_model,1024,
                Exl3EncodedMediaCacheLimits{});
        }
        drafts[0]=Exl3Dflash2DraftModel::load_impl(value.exl3_package->draft_directory,true);
        // Contexts and optional caches join after the single initial bind.
        std::size_t scratch=vision?vision->device_bytes():0;
        for(std::uint32_t i=0;i<value.max_concurrency;++i){
            if(i)drafts[i]=drafts[0]->create_execution_impl(true);
        }
        load.target="qwen3.8-27b/exl3";load.model_id="SC_6.00bpw_H6_V6";load.weights_id="SC_6.00bpw_H6_V6";
        load.load_seconds=seconds(begin);const auto& transfer=target->load_stats();
        load.artifact_bytes_read=transfer.source_bytes[0]+transfer.source_bytes[1];load.peak_staging_bytes=transfer.staging_host_bytes;
        memory.device=value.device;memory.max_context=value.max_context;
        memory.kv_cache=value.kv_cache;
        const auto shared_weight_bytes=target->model_bytes()+drafts[0]->weight_bytes()+
            (vision_model?vision_model->device_bytes():0);
        memory.weights={shared_weight_bytes,shared_weight_bytes,shared_weight_bytes};
        memory.workspace={scratch,scratch,scratch};memory.runtime_reservation_bytes=scratch;
        // Actual startup allocation groups, retained through every admission and
        // publication. Shared weights are charged once; each lane remains private.
        // This does not retroactively reserve the loader boundary above. Future
        // dynamic state/graph growth is separately reserved at its real caller.
        Exl3ResourceInventory inventory;
        inventory.add({target,0,Domain::device,target->model_bytes()});
        if(vision_model) {
            inventory.add({vision_model,0,Domain::device,vision_model->device_bytes()});
            inventory.add({vision_model,1,Domain::host_metadata,
                bounded_shared_allocation_bytes<Exl3VisionModel>()});
        }
        if(vision) {
            inventory.add({vision,0,Domain::device,vision->device_bytes()});
            inventory.add({vision,1,Domain::host_metadata,
                bounded_shared_allocation_bytes<Exl3VisionContext>()});
        }
        stats.target_allocation_owner_metadata_bytes=target->allocation_owner_metadata_bytes();
        if(stats.target_allocation_owner_metadata_bytes)
            inventory.add({target,1,Domain::host_metadata,stats.target_allocation_owner_metadata_bytes});
        inventory.add({drafts[0],0,Domain::device,drafts[0]->weight_bytes()});
        stats.draft_weight_owner_metadata_bytes=drafts[0]->weight_owner_metadata_bytes();
        inventory.add({drafts[0],2,Domain::host_metadata,stats.draft_weight_owner_metadata_bytes});
        for(std::uint32_t i=0;i<value.max_concurrency;++i) {
            if(drafts[i]->execution_bytes())inventory.append(drafts[i]->execution_resources(drafts[i]));
            const auto draft_records=drafts[i]->execution_owner_metadata_bytes();
            stats.draft_execution_owner_metadata_bytes+=draft_records;
            if(drafts[i]->execution_bytes())stats.draft_linear_owner_metadata_bytes+=7*Exl3CudaLinearWorkspace::metadata_bytes();
            stats.draft_generic_owner_metadata_bytes+=drafts[i]->generic_owner_metadata_bytes();
        }
        auto limits=Exl3ResourceInventory::unlimited();
        limits[static_cast<unsigned>(Domain::host_metadata)]=host_metadata_limit;
        // The provider is a capacity observation only.  Tests and future hosts
        // may supply another source, but it cannot allocate, mutate the retained
        // inventory, or turn an unavailable observation into allocation credit.
        // Production retains the existing CUDA observation when none is supplied.
        const Exl3DeviceAvailability::Provider availability=supplied_availability?
            std::move(supplied_availability):Exl3DeviceAvailability::Provider{[] {
                std::size_t free=0,total=0;
                const auto error=cudaMemGetInfo(&free,&total);
                return Exl3DeviceAvailability{total,free,Clock::now(),static_cast<int>(error)};
            }};
        const auto observation=Exl3DeviceAvailability::read(availability);
        limits[static_cast<unsigned>(Domain::device)]=observation.limit(
            inventory.totals()[static_cast<unsigned>(Domain::device)],device_reserve,
            Clock::now(),std::chrono::seconds(1));
        if(draft_clone_fault==5 || draft_clone_fault==6) {
            const auto domain=draft_clone_fault==5?Domain::device:Domain::host_metadata;
            const auto bytes=draft_clone_fault==5?drafts[0]->execution_bytes_required():drafts[0]->execution_owner_metadata_bytes_required();
            Exl3ResourceInventory::Requirement capped;capped.units=inventory.totals();
            capped.add(domain,1,bytes); // First draft must fit; refuse the clone.
            capped.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Request::CancellationOwner>());
            capped.add(Domain::host_metadata,1,queue_bytes);
            capped.add(Domain::host_metadata,1,coordinator_metadata_bytes);
            capped.add(domain,1,bytes-1);
            const auto index=static_cast<unsigned>(domain);
            limits[index]=std::min(limits[index],capped.units[index]);
        }
        if(first_draft_fault==5 || first_draft_fault==6) {
            const auto domain=first_draft_fault==5?Domain::device:Domain::host_metadata;
            const auto bytes=first_draft_fault==5?drafts[0]->execution_bytes_required():drafts[0]->execution_owner_metadata_bytes_required();
            Exl3ResourceInventory::Requirement capped;capped.units=inventory.totals();
            capped.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Request::CancellationOwner>());
            capped.add(Domain::host_metadata,1,queue_bytes);
            capped.add(Domain::host_metadata,1,coordinator_metadata_bytes);
            capped.add(domain,1,bytes-1);
            const auto index=static_cast<unsigned>(domain);
            limits[index]=std::min(limits[index],capped.units[index]);
        }
        if(context_startup_fault==31) {
            Exl3ResourceInventory::Requirement capped;capped.units=inventory.totals();
            capped.add(Domain::host_metadata,value.max_concurrency,drafts[0]->execution_owner_metadata_bytes_required());
            capped.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<ExecutionStream>()-1);
            capped.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Request::CancellationOwner>());
            capped.add(Domain::host_metadata,1,queue_bytes);
            capped.add(Domain::host_metadata,1,coordinator_metadata_bytes);
            const auto index=static_cast<unsigned>(Domain::host_metadata);
            limits[index]=std::min(limits[index],capped.units[index]);
        }
        if(context_startup_fault==64) {
            Exl3ResourceInventory::Requirement capped;capped.units=inventory.totals();
            capped.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Request::CancellationOwner>()-1);
            const auto index=static_cast<unsigned>(Domain::host_metadata);
            limits[index]=std::min(limits[index],capped.units[index]);
        }
        if(context_startup_fault==74) {
            Exl3ResourceInventory::Requirement capped;capped.units=inventory.totals();
            capped.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Request::CancellationOwner>());
            capped.add(Domain::host_metadata,1,queue_bytes-1);
            const auto index=static_cast<unsigned>(Domain::host_metadata);
            limits[index]=std::min(limits[index],capped.units[index]);
        }
        if(context_startup_fault==78) {
            Exl3ResourceInventory::Requirement capped;capped.units=inventory.totals();
            capped.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Request::CancellationOwner>());
            capped.add(Domain::host_metadata,1,queue_bytes);
            capped.add(Domain::host_metadata,1,coordinator_metadata_bytes-1);
            const auto index=static_cast<unsigned>(Domain::host_metadata);
            limits[index]=std::min(limits[index],capped.units[index]);
        }
        coordinator.bind_physical_resources(std::move(inventory),limits);
        {
            const auto& required=cancellation_required;
            std::shared_ptr<Request::CancellationOwner> prepared;
            coordinator.allocate_startup_resources(required,[&](auto configuration) {
                if(configuration!=required.configuration)throw std::logic_error("cancellation owner reservation identity");
                if(context_startup_fault==64)throw std::logic_error("cancellation owner credit refusal reached allocator");
                if(context_startup_fault==61)throw std::runtime_error("injected cancellation owner preconstruction failure");
                prepared=make_bounded_shared<Request::CancellationOwner>();
                if(context_startup_fault==62)throw std::runtime_error("injected cancellation owner precommit failure");
                Exl3ResourceInventory actual;
                actual.add({prepared,0,Domain::host_metadata,bounded_shared_allocation_bytes<Request::CancellationOwner>()-(context_startup_fault==63?1:0),{},
                    &attach_bounded_retirement_credit<Request::CancellationOwner,const void>});
                return actual;
            },[&]{prepared.reset();});
            cancellation_owner=std::move(prepared);cancellation_owner->owner=this;cancellation_owner->output_owner=this;
            stats.cancellation_owner_metadata_bytes=bounded_shared_allocation_bytes<Request::CancellationOwner>();
        }
        {
            std::shared_ptr<Queue::Storage> prepared;
            coordinator.allocate_startup_resources(queue_required,[&](auto configuration) {
                if(configuration!=queue_required.configuration)throw std::logic_error("request queue reservation identity");
                if(context_startup_fault==74)throw std::logic_error("request queue credit refusal reached allocator");
                if(context_startup_fault==71)throw std::runtime_error("injected request queue preconstruction failure");
                prepared=make_bounded_shared<Queue::Storage>();
                prepared->reserve(static_cast<std::size_t>(queue_slots));
                if(context_startup_fault==72)throw std::runtime_error("injected request queue precommit failure");
                if(prepared->capacity()!=queue_slots)throw std::logic_error("request queue actual capacity mismatch");
                Exl3ResourceInventory actual;
                actual.add({prepared,0,Domain::host_metadata,queue_bytes-(context_startup_fault==73?1:0),{},
                    &attach_bounded_retirement_credit<Queue::Storage,const void>});
                return actual;
            },[&]{prepared.reset();});
            queue.bind(std::move(prepared));
        }
        coordinator.reserve_metadata_startup(context_startup_fault>=75 && context_startup_fault<=78?
            context_startup_fault-74:0);
        if(concurrent_prefix_preparation)
            cache.reserve_preparation_storage(coordinator,value.max_concurrency,value.max_context);
        stats.prefix_preparation_metadata_bytes=cache.preparation_storage_stats().metadata_bytes;
        {
            Exl3ResourceInventory::Requirement required;required.configuration=0x445246544631;
            required.add(Domain::device,1,drafts[0]->execution_bytes_required());
            required.add(Domain::host_metadata,1,drafts[0]->execution_owner_metadata_bytes_required());
            coordinator.allocate_startup_resources(required,[&](std::uint64_t configuration){
                if(configuration!=required.configuration)throw std::logic_error("Engine first draft credit identity");
                if(first_draft_fault==5 || first_draft_fault==6)throw std::logic_error("first draft cap refusal reached allocator");
                if(first_draft_fault==1)throw std::runtime_error("injected Engine first draft preconstruction failure");
                if(first_draft_fault==7)Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(true);
                drafts[0]->materialize_private_execution(&coordinator);
                if(first_draft_fault==2)throw std::runtime_error("injected Engine first draft precommit failure");
                if(first_draft_fault==8 && !drafts[0]->fail_generic_retirement_for_test())throw std::logic_error("draft generic failure owner missing");
                auto actual=drafts[0]->execution_resources(drafts[0],first_draft_fault==3 || first_draft_fault==8,first_draft_fault==4);
                return actual;
            },[&]() noexcept {drafts[0].reset();},[&]() noexcept {
                if(Exl3Dflash2DraftModel::generic_quarantined_allocations() || Exl3CudaLinearWorkspace::quarantined_workspaces())
                    coordinator.seal_failed_startup_retirement();
            });
            drafts[0]->finish_constructor_credits();
            scratch+=drafts[0]->execution_bytes();
            memory.workspace={scratch,scratch,scratch};memory.runtime_reservation_bytes=scratch;
            stats.draft_execution_owner_metadata_bytes+=drafts[0]->execution_owner_metadata_bytes();
            stats.draft_linear_owner_metadata_bytes+=7*Exl3CudaLinearWorkspace::metadata_bytes();
            stats.draft_generic_owner_metadata_bytes+=drafts[0]->generic_owner_metadata_bytes();
        }
        if(value.max_concurrency==2) {
            Exl3ResourceInventory::Requirement required;required.configuration=0x44524654434C;
            required.add(Domain::device,1,drafts[0]->execution_bytes_required());
            required.add(Domain::host_metadata,1,drafts[0]->execution_owner_metadata_bytes());
            coordinator.allocate_startup_resources(required,[&](std::uint64_t configuration){
                if(configuration!=required.configuration)throw std::logic_error("Engine draft clone credit identity");
                if(draft_clone_fault==5 || draft_clone_fault==6)throw std::logic_error("draft clone cap refusal reached allocator");
                if(draft_clone_fault==1)throw std::runtime_error("injected Engine draft clone preconstruction failure");
                if(draft_clone_fault==7)Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(true);
                drafts[1]->materialize_private_execution(&coordinator);
                if(draft_clone_fault==2)throw std::runtime_error("injected Engine draft clone precommit failure");
                if(draft_clone_fault==8 && !drafts[1]->fail_generic_retirement_for_test())throw std::logic_error("draft generic failure owner missing");
                auto actual=drafts[1]->execution_resources(drafts[1],draft_clone_fault==3 || draft_clone_fault==8,draft_clone_fault==4);
                return actual;
            },[&]() noexcept {drafts[1].reset();},[&]() noexcept {
                if(Exl3Dflash2DraftModel::generic_quarantined_allocations() || Exl3CudaLinearWorkspace::quarantined_workspaces())
                    coordinator.seal_failed_startup_retirement();
            });
            drafts[1]->finish_constructor_credits();
            scratch+=drafts[1]->execution_bytes();
            memory.workspace={scratch,scratch,scratch};memory.runtime_reservation_bytes=scratch;
            stats.draft_execution_owner_metadata_bytes+=drafts[1]->execution_owner_metadata_bytes();
            stats.draft_linear_owner_metadata_bytes+=7*Exl3CudaLinearWorkspace::metadata_bytes();
            stats.draft_generic_owner_metadata_bytes+=drafts[1]->generic_owner_metadata_bytes();
        }
        {
            for(std::uint32_t i=0;i<value.max_concurrency;++i) {
                const unsigned fault=context_startup_fault && context_startup_fault<=22 && (context_startup_fault-1)/11==i?
                    (context_startup_fault-1)%11+1:0;
                const unsigned stream_fault=context_startup_fault>=23 && (context_startup_fault-23)/3==i?
                    (context_startup_fault-23)%3+1:0;
                const auto& stream_requirement=stream_required;
                std::shared_ptr<ExecutionStream> stream_owner;
                coordinator.allocate_startup_resources(stream_requirement,[&](auto configuration) {
                    if(configuration!=stream_requirement.configuration)
                        throw std::logic_error("Engine stream startup reservation identity");
                    if(context_startup_fault==31)throw std::logic_error("stream credit refusal reached allocator");
                    if(stream_fault==1)throw std::runtime_error("injected Engine stream preconstruction failure");
                    stream_owner=make_bounded_shared<ExecutionStream>(value.max_concurrency==2);
                    if(value.max_concurrency==2)
                        check(cudaStreamCreateWithFlags(&stream_owner->value,cudaStreamNonBlocking));
                    if(value.max_concurrency==2 && context_startup_fault==79+i) {
                        stream_owner->destroy_fault_for_test=static_cast<int>(cudaErrorUnknown);
                        throw std::runtime_error("injected Engine stream unwind failure");
                    }
                    if(stream_fault==2)throw std::runtime_error("injected Engine stream precommit failure");
                    const bool failed_extent_retirement=value.max_concurrency==2 && context_startup_fault==81+i;
                    if(failed_extent_retirement)stream_owner->destroy_fault_for_test=static_cast<int>(cudaErrorUnknown);
                    Exl3ResourceInventory actual;
                    actual.add({stream_owner,0,Domain::host_metadata,bounded_shared_allocation_bytes<ExecutionStream>()-
                        ((stream_fault==3 || failed_extent_retirement)?1:0),{},
                    &attach_bounded_retirement_credit<ExecutionStream,const void>});
                    return actual;
                },[&] {
                    // Release the caller's private owner while startup credit is
                    // still exclusive. A destructor failure seals the authority;
                    // the coordinator preserves the original factory exception.
                    if(stream_owner)stream_owner->retire_idle();
                    stream_owner.reset();
                    if(ExecutionStream::first_destroy_error.load())
                        throw std::runtime_error("execution stream rollback retirement unresolved");
                });
                streams[i]=std::move(stream_owner);
                stats.execution_stream_metadata_bytes+=bounded_shared_allocation_bytes<ExecutionStream>();
                // The coherent route admits the ordinary graph families unless
                // the eager fallback is selected.  The families themselves are
                // selected by their own switches and captured at context creation.
                auto context=target->create_context_reserved(coordinator,true,
                    context_startup_fault>=90 && context_startup_fault<=100?context_startup_fault-73:
                    (context_startup_fault==88?16:(context_startup_fault==86?11:(fault<=4?fault:0))),
                    public_media_enabled,!coherent_device || coherent_ordinary_graphs);
                const auto base_context_bytes=context->persistent_bytes();
                context->prepare_continuation_reserved(coordinator,8,
                    context_startup_fault==89?6:(context_startup_fault==87?5:(fault>=5 && fault<=7?fault-4:0)));
                const auto continuation_bytes=context->persistent_bytes()-base_context_bytes;
                const auto before_repair_checkpoint=context->persistent_bytes();
                const auto* repair_checkpoint=std::getenv("NINFER_EXL3_REPAIR_CHECKPOINT");
                if(repair_checkpoint && std::string_view(repair_checkpoint)!="0" &&
                    std::string_view(repair_checkpoint)!="1")
                    throw std::invalid_argument("repair checkpoint must be0 or1");
                if(repair_checkpoint && std::string_view(repair_checkpoint)=="1") {
                    try { context->prepare_transaction_reserved(coordinator); }
                    catch(const Exl3ResourceReservationExhausted&) {
                        ++stats.repair_checkpoint_reservation_fallback_lanes;
                    }
                }
                const auto repair_checkpoint_bytes=
                    context->persistent_bytes()-before_repair_checkpoint;
                lanes[i]=Exl3Dflash2Execution::create_reserved(coordinator,std::move(context),
                    drafts[i],identity.contract(),streams[i]->value,streams[i],fault>=8?fault-7:0);
                ++stats.reserved_context_startup_lanes;
                stats.reserved_context_startup_device_bytes+=base_context_bytes;
                stats.reserved_context_fixed_owner_metadata_bytes+=Exl3TextContext::fixed_owner_metadata_bytes();
                stats.reserved_continuation_startup_device_bytes+=continuation_bytes;
                stats.reserved_continuation_owner_metadata_bytes+=Exl3TextContext::continuation_owner_metadata_bytes();
                stats.reserved_repair_checkpoint_startup_device_bytes+=repair_checkpoint_bytes;
                if(repair_checkpoint_bytes)
                    stats.reserved_repair_checkpoint_owner_metadata_bytes+=
                        Exl3TextContext::transaction_owner_metadata_bytes();
                stats.continuation_linear_owner_metadata_bytes+=Exl3CudaLinearWorkspace::metadata_bytes();
                stats.continuation_generic_owner_metadata_bytes+=2*Exl3TextContext::allocation_record_metadata_bytes();
                stats.reserved_lane_startup_device_bytes+=lanes[i]->construction_added_device_bytes();
                stats.reserved_lane_fixed_owner_metadata_bytes+=sizeof(Exl3Dflash2Execution);
                // Context, continuation and lane staging have distinct retained
                // owners in the single coordinator; do not add a bundled lane
                // allocation again after those credits have committed.
                scratch+=lanes[i]->persistent_bytes();
                stats.context_allocation_owner_metadata_bytes+=lanes[i]->context().allocation_owner_metadata_bytes();
                stats.context_base_linear_owner_metadata_bytes+=lanes[i]->context().base_linear_owner_metadata_bytes();
                stats.context_private_generic_owner_metadata_bytes+=lanes[i]->context().private_generic_owner_metadata_bytes();
                stats.context_shared_generic_owner_metadata_bytes+=lanes[i]->context().shared_generic_owner_metadata_bytes();
                stats.context_shared_control_metadata_bytes+=lanes[i]->context().shared_generic_owner_metadata_bytes()/
                    Exl3TextContext::allocation_record_metadata_bytes()*Exl3ReconstructionControlAllocator<std::byte>::capacity;
                stats.context_layer_linear_owner_metadata_bytes+=lanes[i]->context().layer_linear_owner_metadata_bytes();
                const auto attention_memory=lanes[i]->context().host_kv_stats();
                stats.exact_attention_score_bytes+=attention_memory.exact_attention_score_bytes;
                stats.numeric_attention_scratch_bytes+=attention_memory.numeric_attention_scratch_bytes;
                stats.coalesced_attention_layers+=attention_memory.coalesced_attention_layers;
                stats.coalesced_attention_bytes_saved+=attention_memory.coalesced_attention_bytes_saved;
                stats.execution_retirement_metadata_bytes+=Exl3Dflash2Execution::retirement_metadata_bytes();
            }
            memory.workspace={scratch,scratch,scratch};memory.runtime_reservation_bytes=scratch;
        }
        {
            std::array<Exl3TextContext*,2> contexts{};
            for(std::uint32_t i=0;i<value.max_concurrency;++i)contexts[i]=&lanes[i]->context();
            Exl3TextContext::reserve_reconstruction_lanes(coordinator,
                std::span(contexts.data(),value.max_concurrency),reconstruction_fallback);
            for(std::uint32_t i=0;i<value.max_concurrency;++i) {
                const auto reconstruction=contexts[i]->reconstructed_exact_stats();
                stats.reconstruction_device_bytes+=reconstruction.workspace_bytes;
                scratch+=reconstruction.workspace_bytes;
                if(reconstruction.workspace_bytes)
                    stats.reconstruction_metadata_bytes+=Exl3TextContext::reconstruction_backing_metadata_bytes();
                stats.reconstruction_budget_fallback_lanes+=reconstruction.reservation_fallback?1:0;
            }
        }
        if(attention_staging) {
          auto stage_lanes=exl3_create_attention_stage_lanes(coordinator,value.max_concurrency,value.max_context);
          for(std::uint32_t i=0;i<value.max_concurrency;++i) {
            auto& resources=stage_lanes[i];
            lanes[i]->context().set_attention_staging(std::move(resources.storage),std::move(resources.stages),std::move(resources.history));
            stats.attention_stage_device_bytes+=static_cast<std::uint64_t>(value.max_context)*4096;
            scratch+=static_cast<std::uint64_t>(value.max_context)*4096;
            stats.attention_stage_metadata_bytes+=Exl3AttentionStageStorage::metadata_bytes()+
                Exl3AttentionStage::metadata_bytes()+Exl3AttentionStageHistory::metadata_bytes();
          }
        }
        if(share_pages) {
            shared_pages=Exl3DevicePageCache::create_startup(coordinator,context_startup_fault==125?1:0);
            auto readers=Exl3DevicePageReaders<Exl3AttentionPageRanges::capacity>::create_startup(
                coordinator,value.max_concurrency,share_attention,context_startup_fault==126?
                    value.max_concurrency*(1+(share_attention?Exl3AttentionPageRanges::capacity:0)):0);
            page_copies=std::move(readers.copies);page_attention=std::move(readers.attention);
            for(std::uint32_t i=0;i<value.max_concurrency;++i) {
                if(share_attention) {
                    lanes[i]->context().set_shared_page_attention([this,i](std::shared_ptr<const Exl3ExactKVPage> page,
                        int bank,int position,cudaStream_t stream,cudaEvent_t event)->Exl3SharedAttentionPrefix {
                        auto& lane=lanes[i];
                        if(!lane->execution_active() || !shared_pages_enabled.load())return {};
                        try {
                            unsigned slot=0;
                            while(slot<page_attention[i].size() && page_attention[i][slot]->uncertain())++slot;
                            if(slot==page_attention[i].size())return {};
                            const auto lease=lane->lease();const auto epoch=lane->execution_epoch();
                            const bool ancestor=lease.root && lease.root->owns_shared_text_page(page,
                                foreign_page_ancestor_for_test?foreign_page_ancestor_for_test:lane->context().model_identity(),
                                lane->context().rope_offset());
                            {
                                std::lock_guard lock(mutex);
                                if(ancestor)++stats.shared_device_page_ancestor_accepts;
                                else ++stats.shared_device_page_ancestor_fallbacks;
                            }
                            if(!ancestor)return {};
                            const auto require_scope=[&]{
                                if(!lane->execution_active() || lane->execution_epoch()!=epoch ||
                                    !coordinator.compute_leases_current(std::span(&lease,1)))
                                    throw std::invalid_argument("shared page attention execution changed");
                            };
                            Exl3DevicePageKey key(lane->context().model_identity(),std::move(page),
                                {lane->context().rope_offset(),false},position);
                            Exl3DevicePageAttention::require_admission(key,bank,position,lane,
                                lease.acquisition,epoch,event);
                            auto registration_read=kv_registrations
                                ? Exl3KVRegistrationCache::acquire_external_read(kv_registrations)
                                : std::optional<Exl3KVRegistrationCache::ExternalRead>{};
                            if(kv_registrations && !registration_read)return {};
                            auto acquired=shared_pages->acquire(coordinator,lease,key);
                            if(!acquired.shared && shared_pages->evict_one(coordinator,lease)) {
                                {std::lock_guard lock(mutex);++stats.shared_device_page_evictions;}
                                acquired=shared_pages->acquire(coordinator,lease,key);
                            }
                            if(!acquired.shared)return {};
                            require_scope();
                            if(acquired.producer) {
                                exl3_fill_device_page(*acquired.producer,stream,event,lane,
                                    lease.acquisition,epoch,page_provider_for_test(true),std::move(registration_read));
                                std::lock_guard lock(mutex);stats.shared_device_page_fill_bytes+=Exl3DevicePageStorage::bytes;
                            }
                            auto view=acquired.shared->ready_handle();if(!view)return {};
                            require_scope();
                            auto result=page_attention[i][slot]->begin(std::move(view),key,bank,position,lane,
                                lease.acquisition,epoch,event);
                            result.slot=slot;return result;
                        } catch(...) {lane->poison_shared_completion();throw;}
                    },[this,i](std::span<const Exl3SharedAttentionPrefix> pages,cudaStream_t stream,Exl3TextContext::SharedPageAttentionPhase phase) {
                        try {
                            if(pages.empty() || pages.size()>page_attention[i].size())throw std::invalid_argument("attention completion count");
                            std::array<Exl3DevicePageAttention::Completion,Exl3AttentionPageRanges::capacity> group{};
                            for(std::size_t p=0;p<pages.size();++p) {
                                if(pages[p].slot>=page_attention[i].size())throw std::invalid_argument("attention completion slot");
                                group[p]={page_attention[i][pages[p].slot].get(),pages[p].generation};
                            }
                            const auto members=std::span(group.data(),pages.size());
                            if(phase==Exl3TextContext::SharedPageAttentionPhase::before_compute)
                                Exl3DevicePageAttention::require_group(members);
                            else Exl3DevicePageAttention::complete_group(members,stream,attention_provider_for_test());
                        }
                        catch(...) {lanes[i]->poison_shared_completion();throw;}
                    });
                }
                lanes[i]->context().set_shared_page_copier([this,i](const Exl3ExactKVExtent& extent,
                    void* destination,int capacity,cudaStream_t stream,cudaEvent_t event) {
                    auto& lane=lanes[i];
                    if(!lane->execution_active() || !shared_pages_enabled.load() ||
                        extent.first()+extent.rows()!=extent.owner()->first+64)return false;
                    try {
                        const auto lease=lane->lease();const auto epoch=lane->execution_epoch();
                        const auto require_scope=[&]{
                            if(!lane->execution_active() || lane->execution_epoch()!=epoch ||
                                !coordinator.compute_leases_current(std::span(&lease,1)))
                                throw std::invalid_argument("shared page copy execution changed");
                        };
                        Exl3RegisteredKVUpload::require_submission(extent,destination,capacity,event,
                            lease.acquisition,epoch);
                        const bool ancestor=lease.root && lease.root->owns_shared_text_page(extent.owner(),
                            foreign_page_ancestor_for_test?foreign_page_ancestor_for_test:lane->context().model_identity(),
                            lane->context().rope_offset());
                        {
                            std::lock_guard lock(mutex);
                            if(ancestor)++stats.shared_device_page_ancestor_accepts;
                            else ++stats.shared_device_page_ancestor_fallbacks;
                        }
                        if(!ancestor)return false;
                        Exl3DevicePageKey key(lane->context().model_identity(),extent.owner(),
                            {lane->context().rope_offset(),false},extent.first()+extent.rows());
                        auto registration_read=kv_registrations
                            ? Exl3KVRegistrationCache::acquire_external_read(kv_registrations)
                            : std::optional<Exl3KVRegistrationCache::ExternalRead>{};
                        if(kv_registrations && !registration_read)return false;
                        auto acquired=shared_pages->acquire(coordinator,lease,key);
                        if(!acquired.shared && shared_pages->evict_one(coordinator,lease)) {
                            {std::lock_guard lock(mutex);++stats.shared_device_page_evictions;}
                            acquired=shared_pages->acquire(coordinator,lease,key);
                        }
                        if(!acquired.shared)return false;
                        require_scope();
                        if(acquired.producer) {
                            exl3_fill_device_page(*acquired.producer,stream,event,lane,
                                lease.acquisition,epoch,page_provider_for_test(true),std::move(registration_read));
                            std::lock_guard lock(mutex);stats.shared_device_page_fill_bytes+=Exl3DevicePageStorage::bytes;
                        }
                        auto view=acquired.shared->ready_handle();
                        if(!view)return false; // compatible peer still filling; use existing host route
                        require_scope();
                        page_copies[i]->copy(std::move(view),extent,destination,capacity,stream,event,lane,
                            lease.acquisition,epoch,page_provider_for_test(false));
                        return true;
                    } catch(...) {lane->poison_shared_completion();throw;}
                });
            }
        }
        if(registered_upload) {
            // Registered upload requires fixed pinned chunks: fallback copies
            // gather into separate buffers rather than reading registered pages
            // on the device. Shared-page and attention-history fills retain
            // external read guards. Represented prefix fills copy from device
            // layer staging, so they introduce no additional registered host read.
            kv_registrations=Exl3KVRegistrationCache::create_startup(coordinator);
            kv_uploads=Exl3RegisteredKVUpload::create_startup_lanes(coordinator,value.max_concurrency,
                context_startup_fault==127?2*value.max_concurrency:0,
                registered_upload_slots);
            if(attention_staging)for(const auto& lane:lanes)if(lane)
                lane->context().set_attention_registration_cache(kv_registrations);
            for(std::uint32_t i=0;i<value.max_concurrency;++i) {
                lanes[i]->context().set_registered_kv_uploader([this,i,registered_upload_slots](const Exl3ExactKVExtent& extent,
                    void* destination,int capacity,cudaStream_t stream,cudaEvent_t event,int slot)->std::uint64_t {
                    auto& lane=lanes[i];
                    if(!lane->execution_active() || !registered_upload_enabled.load())return false;
                    try {
                        if(slot<0 || static_cast<unsigned>(slot)>=registered_upload_slots)
                            throw std::invalid_argument("registered KV upload slot");
                        const auto lease=lane->lease();const auto epoch=lane->execution_epoch();
                        Exl3RegisteredKVUpload::require_submission(extent,destination,capacity,event,lease.acquisition,epoch);
                        auto registration=kv_registrations->acquire_runtime(coordinator,lease,extent);
                        if(lane->execution_epoch()!=epoch || !coordinator.compute_leases_current(std::span(&lease,1)))
                            throw std::invalid_argument("registered KV upload execution changed before submission");
                        if(!registration)return false;
                        return kv_uploads[i][slot]->submit(extent,destination,capacity,stream,event,lane,
                            std::move(registration),lease.acquisition,epoch);
                    } catch(...) {lane->poison_shared_completion();throw;}
                },[this,i,registered_upload_slots](int slot,std::uint64_t generation) {
                    try {
                        if(slot<0 || static_cast<unsigned>(slot)>=registered_upload_slots)
                            throw std::invalid_argument("registered KV completion slot");
                        kv_uploads[i][slot]->complete(generation);
                    } catch(...) {lanes[i]->poison_shared_completion();throw;}
                });
            }
        }
        const auto taps=Exl3EngineTapRequirements::for_lanes(value.max_concurrency,sizeof(DraftTapStaging));
        std::array<std::shared_ptr<DraftTapStaging>,2> prepared_staging;
        coordinator.allocate_startup_resources(taps,[&](std::uint64_t configuration) {
            if(configuration!=value.max_concurrency)
                throw std::logic_error("Engine tap allocation configuration changed");
            Exl3ResourceInventory actual;
            // Keep partial allocations private until the complete C1/C2 batch
            // has been acquired. This factory enqueues no consumers.
            std::array<std::shared_ptr<DraftTapStaging>,2> pending;
            unsigned allocated_planes=0;
            for(std::uint32_t i=0;i<value.max_concurrency;++i) {
                pending[i]=std::make_shared<DraftTapStaging>();
                for(auto& p:pending[i]->pointers) {
                    check(cudaMalloc(reinterpret_cast<void**>(&p),Exl3EngineTapRequirements::plane_bytes));
                    if(context_startup_fault==40+(++allocated_planes))
                        throw std::runtime_error("injected Engine tap staging allocation failure");
                }
                actual.add({pending[i],0,Domain::device,
                    Exl3EngineTapRequirements::layers*Exl3EngineTapRequirements::plane_bytes-
                        (context_startup_fault==52 && i==0?1:0)});
                actual.add({pending[i],1,Domain::host_metadata,sizeof(DraftTapStaging)-
                    (context_startup_fault==53 && i==0?1:0)});
            }
            if(context_startup_fault==51)throw std::runtime_error("injected Engine tap staging precommit failure");
            prepared_staging=std::move(pending);
            return actual;
        });
        staging_owners=std::move(prepared_staging);
        stats.reserved_engine_tap_device_bytes=taps.units[static_cast<unsigned>(Domain::device)];
        stats.reserved_engine_tap_metadata_bytes=taps.units[static_cast<unsigned>(Domain::host_metadata)];
        scratch+=stats.reserved_engine_tap_device_bytes;
        const bool skip_context_hooks=[] {
            const auto* skip=std::getenv("NINFER_EXL3_TEST_ENGINE_SKIP_CONTEXT_HOOKS");
            if(!skip || std::string_view(skip)!="1")return false;
            const auto* root=std::getenv("NINFER_EXL3_TEST_ENGINE_ROOT_HASH");
            if(!root || std::string_view(root)!="1")
                throw std::invalid_argument("skip context hooks requires diagnostic root probe");
            return true;
        }();
        if(!skip_context_hooks) {
        for(std::uint32_t i=0;i<value.max_concurrency;++i)
            lanes[i]->context().set_snapshot_metadata_reservation([this,i](std::uint64_t bytes) {
                const auto request=active[i];
                // Cancellation is observed by the request loop after this
                // synchronous export. Keep its still-live authority so abort
                // can discard the prepared root without poisoning the lane.
                if(!request)throw std::logic_error("snapshot metadata request unavailable");
                auto& lane=lanes[i];
                if(lane->execution_active()) {
                    if(cancel_active_snapshot_for_test.exchange(false))request->cancelled=true;
                    const auto lease=lane->lease();
                    return coordinator.reserve_snapshot_metadata(&lease,{},bytes);
                }
                return coordinator.reserve_snapshot_metadata(nullptr,request,bytes);
            });
        for(std::uint32_t i=0;i<value.max_concurrency;++i)
            lanes[i]->context().set_request_metadata_reservation([this,i](std::uint64_t bytes) {
                const auto request=active[i];
                if(!request)throw std::logic_error("request metadata authority unavailable");
                auto& lane=lanes[i];
                const unsigned phase=lane->execution_active()?2:1;
                auto expected=phase;
                if(request_metadata_fault.compare_exchange_strong(expected,0)) {
                    request_metadata_fault_hit.store(phase);
                    throw std::runtime_error("request metadata admission fixture exhausted");
                }
                if(lane->execution_active()) {
                    const auto lease=lane->lease();
                    expected=3;
                    const bool exhaust=request_metadata_fault.compare_exchange_strong(expected,0);
                    if(exhaust)request_metadata_fault_hit.store(3);
                    return coordinator.reserve_snapshot_metadata(&lease,{},bytes,exhaust);
                }
                return coordinator.reserve_snapshot_metadata(nullptr,request,bytes);
            });
        for(std::uint32_t i=0;i<value.max_concurrency;++i) {
            lanes[i]->context().set_request_host_payload_reservation([this,i](std::uint64_t bytes) {
                const auto request=active[i];
                auto& lane=lanes[i];
                if(!request || !lane->execution_active())
                    throw std::logic_error("request host payload authority unavailable");
                // A cancellation flag does not revoke the still-current lease.
                // The synchronous append must retain credit through cleanup.
                auto credit=coordinator.reserve_snapshot_payload(lane->lease(),bytes);
                unsigned expected=1;
                if(request_host_payload_fault.compare_exchange_strong(expected,0)) {
                    request_host_payload_fault_hit.store(1);
                    throw std::runtime_error("request host payload reservation fixture failure");
                }
                return credit;
            });
            lanes[i]->context().set_request_host_payload_observer([this,i](
                const std::shared_ptr<const void>& owner,std::size_t plane,const void* data,std::size_t bytes) {
                const auto request=active[i];
                auto& lane=lanes[i];
                if(!request || !lane->execution_active())
                    throw std::logic_error("request host payload tracking authority unavailable");
                coordinator.track_snapshot_payload(lane->lease(),owner,plane,data,bytes);
                unsigned expected=2;
                if(plane==2 && request_host_payload_fault.compare_exchange_strong(expected,0)) {
                    request_host_payload_fault_hit.store(2);
                    throw std::runtime_error("request host payload tracking fixture failure");
                }
            });
        }
        for(std::uint32_t i=0;i<value.max_concurrency;++i)
            lanes[i]->context().set_recurrent_growth_admission([this,i](const Exl3RecurrentSlabLayout& layout,
                const Exl3RecurrentExportPool::GrowthFactory& factory) -> std::shared_ptr<Exl3RecurrentSlab> {
                auto& lane=lanes[i];
                const auto request=active[i];if(!request || request->cancelled)return {};
                const bool executing=lane->execution_active();const auto epoch=lane->execution_epoch();
                std::optional<Exl3VeriCacheServingCoordinator::Lease> lease;if(executing)lease=lane->lease();
                const auto before=Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire);
                const auto requirement=Exl3RecurrentSlab::requirement(layout);
                std::shared_ptr<Exl3RecurrentSlab> prepared;
                try {
                    const auto allocate=[&](std::uint64_t configuration) {
                        if(configuration!=requirement.configuration || lane->execution_active()!=executing || lane->execution_epoch()!=epoch || active[i]!=request || request->cancelled)
                            throw std::invalid_argument("recurrent growth execution changed");
                        auto credits=coordinator.reserve_registration_constructor_credits(layout.bytes,Exl3RecurrentSlab::physical_metadata_bytes());
                        prepared=factory({std::move(credits.registration),std::move(credits.metadata)});
                        if(!prepared)throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
                        unsigned cancel_phase=executing?2:1;
                        if(recurrent_cancel_after_factory.compare_exchange_strong(cancel_phase,0)) {
                            recurrent_cancel_after_factory_hit.store(executing?2:1);
                            request->cancelled.store(true,std::memory_order_release);
                        }
                        // Construction may cross a concurrent cancellation or
                        // lane transition. Refuse publication while the new slab
                        // is still owned solely by the reservation transaction.
                        if(lane->execution_active()!=executing || lane->execution_epoch()!=epoch ||
                           active[i]!=request || request->cancelled.load(std::memory_order_acquire))
                            throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
                        if(prepared->bytes!=layout.bytes || prepared->offsets!=layout.offsets)
                            throw std::logic_error("recurrent allocation changed reserved layout");
                        auto actual=Exl3RecurrentSlab::resources(prepared);
                        unsigned expected=executing?3:1;
                        if(recurrent_commit_fault.compare_exchange_strong(expected,0)) {
                            recurrent_commit_fault_hit.store(executing?3:1);
                            actual.add({prepared,99,Domain::host_metadata,1});
                        }
                        return actual;
                    };
                    const auto rollback=[&]() noexcept {prepared.reset();};
                    const auto observe=[&]() noexcept {
                        if(Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire)!=before)
                            coordinator.seal_failed_startup_retirement();
                    };
                    if(lease)coordinator.allocate_runtime_resources(*lease,requirement,allocate,rollback,observe);
                    else coordinator.allocate_preparation_resources(request,requirement,allocate,rollback,observe);
                    prepared->release_constructor_credits_after_commit();
                    return prepared;
                } catch(...) {
                    prepared.reset();
                    if(Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire)!=before) {
                        coordinator.seal_failed_startup_retirement();
                        lane->poison_shared_completion();
                        std::throw_with_nested(std::runtime_error("recurrent growth cleanup unresolved"));
                    }
                    throw;
                }
            });
        for(std::uint32_t i=0;i<value.max_concurrency;++i)
            lanes[i]->context().set_recurrent_borrow_admission([this,i](const std::shared_ptr<Exl3RecurrentSlab>& owner) -> std::shared_ptr<Exl3RecurrentSlab> {
                auto& lane=lanes[i];const auto request=active[i];if(!request || request->cancelled)return {};
                const bool executing=lane->execution_active();const auto epoch=lane->execution_epoch();
                std::optional<Exl3VeriCacheServingCoordinator::Lease> lease;if(executing)lease=lane->lease();
                Exl3ResourceInventory::Requirement requirement;requirement.configuration=0x524543424f5252;
                requirement.add(Domain::host_metadata,1,Exl3RecurrentSlab::control_metadata_bytes());
                std::shared_ptr<Exl3RecurrentSlab> borrowed;
                Exl3ResourceInventory::Allocation allocation;
                const auto allocate=[&](std::uint64_t configuration) {
                    if(configuration!=requirement.configuration || lane->execution_active()!=executing || lane->execution_epoch()!=epoch || active[i]!=request || request->cancelled)
                        throw std::invalid_argument("recurrent borrower execution changed");
                    Exl3SharedControlCredit* header=nullptr;
                    borrowed=Exl3RecurrentSlab::borrow(owner,std::nullopt,&header);
                    if(!borrowed)throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
                    allocation={std::shared_ptr<const void>(borrowed,header),0,Domain::host_metadata,
                        Exl3RecurrentSlab::control_metadata_bytes(),{},
                        &Exl3RecurrentSlab::attach_borrower_control_credit};
                    Exl3ResourceInventory actual;actual.add(allocation);
                    unsigned expected=executing?4:2;
                    if(recurrent_commit_fault.compare_exchange_strong(expected,0)) {
                        recurrent_commit_fault_hit.store(executing?4:2);
                        actual.add({borrowed,99,Domain::host_metadata,1});
                    }
                    return actual;
                };
                const auto rollback=[&]() noexcept {allocation={};borrowed.reset();};
                if(lease)coordinator.allocate_runtime_resources(*lease,requirement,allocate,rollback);
                else coordinator.allocate_preparation_resources(request,requirement,allocate,rollback);
                if(!(lease?coordinator.retire_runtime_metadata_to_lifetime(*lease,allocation):
                    coordinator.retire_preparation_metadata_to_lifetime(request,allocation)))
                    throw std::logic_error("recurrent borrower lifetime transfer refused");
                return borrowed;
            });
        }
        if(share_projection_owner) {
            const auto requirement=Exl3EngineTargetQ::requirement(share_kv,share_draft,share_o,share_gateup,share_down,share_head,share_draft_kv,share_draft_o,share_draft_down,share_draft_gateup);
            std::shared_ptr<Exl3EngineTargetQ> prepared_shared;
            coordinator.allocate_startup_resources(requirement,[&](std::uint64_t configuration) {
                if(configuration!=requirement.configuration)
                    throw std::logic_error("Engine shared Q configuration changed");
                if(context_startup_fault==101 || context_startup_fault==102)
                    Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(context_startup_fault==101);
                const bool family_fault=context_startup_fault>=103 && context_startup_fault<=122;
                auto owner=make_bounded_shared<Exl3EngineTargetQ>(share_kv,share_draft,share_o,share_gateup,share_down,share_head,share_draft_kv,share_draft_o,share_draft_down,shared_allocation_fault,context_startup_fault==83 || context_startup_fault==84,context_startup_fault==85?1:0,share_draft_gateup,&coordinator,
                    family_fault?2+(context_startup_fault-103)/2:0,family_fault?(context_startup_fault-103)%2==0:true);
                Exl3ResourceInventory actual;actual.add({owner,0,Domain::device,owner->bytes(),{},nullptr,
                    &Exl3EngineTargetQ::attach_device_retirement_credit});
                actual.add({owner,1,Domain::host_metadata,owner->host_metadata_bytes()-((context_startup_fault==84 || context_startup_fault==85)?1:0),{},
                    &Exl3EngineTargetQ::attach_retirement_credit});
                owner->enable_gather_reuse(share_gather);
                prepared_shared=std::move(owner);return actual;
            },[&] {
                prepared_shared.reset();
                if(Exl3CudaLinearWorkspace::quarantined_workspaces() || Exl3EngineTargetQ::packed_retirement_unresolved())
                    throw std::runtime_error("shared workspace startup retirement unresolved");
            },[] {
                if(Exl3CudaLinearWorkspace::quarantined_workspaces() || Exl3EngineTargetQ::packed_retirement_unresolved())
                    throw std::runtime_error("shared workspace final rollback release unresolved");
            });
            shared_q=std::move(prepared_shared);
            shared_q->release_constructor_credits_after_commit();
            scratch+=shared_q->bytes();memory.workspace={scratch,scratch,scratch};
            memory.runtime_reservation_bytes=scratch;
            for(std::uint32_t i=0;i<value.max_concurrency;++i) {
                if(share_q)lanes[i]->context().set_target_q_executor([this,i](const Exl3TargetQContinuation& q) {
                    auto& lane=lanes[i];
                    if(!lane->execution_active())return false; // prefill/idle fallback
                    try {
                        // This worker owns active[i] until execute returns;
                        // cancellation mutates only its atomic request flag. Its
                        // retained control span is the exact per-request sharing
                        // contract for this synchronous callback.
                        return shared_q->execute(q,lane->lease(),lane->execution_epoch(),lane,target,coordinator,
                            active[i]?active[i]->control_ids:std::span<const std::int64_t>{},
                            active[i]?&active[i]->cancelled:nullptr);
                    } catch(...) {lane->poison_shared_completion();throw;}
                },share_kv,share_o,share_gateup,share_down,share_head);
                if(share_draft)drafts[i]->set_shared_q_executor([this,i](const Exl3DraftSharedQContinuation& q) {
                    auto& lane=lanes[i];
                    if(!lane->execution_active())return false;
                    const auto conditioning=lane->shared_proposal_root();
                    if(q.acquisition!=lane->lease().acquisition || q.execution!=lane->execution_epoch() ||
                       !lane->matches_shared_proposal(q.segment,q.seed) ||
                       !conditioning || !q.matches_conditioning_owner(conditioning->projected_metadata_owner()) ||
                       q.projection.position!=conditioning->state()->position() ||
                       q.ring_base<0 || q.ring_count<0 || q.ring_count>2047 ||
                       q.ring_base>q.projection.position || q.projection.position-q.ring_base!=q.ring_count ||
                       q.seed<0 || q.seed>=248320 || q.projection.rows!=8 ||
                       !q.segment.matches(q.acquisition,q.execution,q.ring_base,q.ring_count,q.projection.position,q.seed))
                        throw std::invalid_argument("Engine draft Q private conditioning scope mismatch");
                    try {
                        return shared_q->execute(q.projection,lane->lease(),lane->execution_epoch(),lane,target,coordinator,
                            active[i]?active[i]->control_ids:std::span<const std::int64_t>{},
                            active[i]?&active[i]->cancelled:nullptr,conditioning);
                    } catch(...) {lane->poison_shared_completion();throw;}
                },share_draft_kv,share_draft_o,share_draft_down,share_draft_gateup);
            }
        }
        if(batch_greedy) {
            const auto requirement=Exl3EngineGreedyPacketBatch::requirement();
            std::shared_ptr<Exl3EngineGreedyPacketBatch> prepared_batch;
            coordinator.allocate_startup_resources(requirement,[&](std::uint64_t configuration) {
                if(configuration!=requirement.configuration)
                    throw std::logic_error("Engine greedy packet batch configuration changed");
                auto owner=make_bounded_shared<Exl3EngineGreedyPacketBatch>();
                Exl3ResourceInventory actual;
                actual.add({owner,0,Domain::device,
                    Exl3EngineGreedyPacketBatch::device_bytes()});
                actual.add({owner,1,Domain::cuda_registered_host,
                    Exl3EngineGreedyPacketBatch::registered_host_bytes()});
                actual.add({owner,2,Domain::host_metadata,
                    Exl3EngineGreedyPacketBatch::host_metadata_bytes()});
                prepared_batch=std::move(owner);return actual;
            },[&] {
                prepared_batch.reset();
                if(Exl3EngineGreedyPacketBatch::retirement_unresolved())
                    throw std::runtime_error("greedy packet batch startup retirement unresolved");
            },[] {
                if(Exl3EngineGreedyPacketBatch::retirement_unresolved())
                    throw std::runtime_error("greedy packet batch final rollback unresolved");
            });
            greedy_packet_batch=std::move(prepared_batch);
            scratch+=Exl3EngineGreedyPacketBatch::device_bytes();
            memory.workspace={scratch,scratch,scratch};
            memory.runtime_reservation_bytes=scratch;
            for(std::uint32_t i=0;i<value.max_concurrency;++i)
                lanes[i]->set_greedy_packet_batch_finish([this,i](Exl3TextContext& context,
                    Exl3PendingGreedyPacket&& pending,std::uint64_t acquisition,
                    std::uint64_t execution,cudaStream_t) {
                    return greedy_packet_batch->finish(context,std::move(pending),
                        acquisition,execution,active[i]?&active[i]->cancelled:nullptr);
                });
        }
        // Optional represented caches must not consume credit needed by any
        // mandatory lane, reconstruction, upload or shared-projection owner.
        if(private_prefix_requested) {
            for(std::uint32_t i=0;i<(share_prefix?1U:value.max_concurrency);++i) {
                lanes[i]->context().prepare_device_prefix_reserved(coordinator,
                    private_prefix_capacity,segmented_prefix_requested,
                    i==0 && (context_startup_fault==123 || context_startup_fault==124)?context_startup_fault-117:0);
                const auto bytes=lanes[i]->context().host_kv_stats().device_prefix_bytes;
                if(share_prefix && bytes) {
                    lanes[i]->context().share_device_prefix_with_empty(lanes[1]->context());
                    stats.shared_device_prefix_bytes=bytes;
                } else if(!share_prefix)stats.private_device_prefix_bytes+=bytes;
                scratch+=bytes;
            }
            memory.workspace={scratch,scratch,scratch};memory.runtime_reservation_bytes=scratch;
        }
        for(const auto& lane:lanes)if(lane) {
            const auto graph=lane->context().host_kv_gdn_segment_graph_stats();
            stats.host_kv_gdn_segment_graph_captures+=graph.captures;
            stats.host_kv_gdn_segment_graph_capture_seconds+=graph.capture_ms/1000.0;
            const auto full=lane->context().host_kv_full_layer_graph_stats();
            stats.host_kv_full_layer_graph_captures+=full.captures;
            stats.host_kv_full_layer_graph_six_softmax_triple_captures+=
                full.six_softmax_triple_captures;
            stats.host_kv_full_layer_graph_k6_stream_reduction_captures+=
                full.k6_stream_reduction_captures;
            stats.host_kv_full_layer_graph_extended_stream_reduction_captures+=
                full.extended_stream_reduction_captures;
            stats.host_kv_full_layer_graph_target_down_k6_async_a_captures+=
                full.target_down_k6_async_a_captures;
            stats.host_kv_full_layer_graph_target_k6_small_m_async_a_captures+=
                full.target_k6_small_m_async_a_captures;
            stats.host_kv_full_layer_graph_target_k7_small_m_async_a_captures+=
                full.target_k7_small_m_async_a_captures;
            stats.host_kv_full_layer_graph_capture_seconds+=full.capture_ms/1000.0;
            const auto mlp_tail=lane->context().host_kv_mlp_tail_graph_stats();
            stats.host_kv_mlp_tail_graph_captures+=mlp_tail.captures;
            stats.host_kv_mlp_tail_graph_capture_seconds+=
                mlp_tail.capture_ms/1000.0;
        }
        memory.workspace={scratch,scratch,scratch};memory.runtime_reservation_bytes=scratch;
        load.load_seconds=seconds(begin);
        const auto final_availability=Exl3DeviceAvailability::read(availability);
        memory.available_after_startup_bytes=final_availability.available;
        if(const auto retained=coordinator.stats().retained_resource_units) {
            const auto device=static_cast<unsigned>(Domain::device);
            const auto attribution=final_availability.attribution(
                (*retained)[device],Clock::now(),std::chrono::seconds(1));
            stats.inventoried_device_bytes=attribution.inventoried_bytes;
            stats.observed_device_used_bytes=attribution.observed_used_bytes;
            stats.driver_unknown_device_bytes=attribution.driver_unknown_bytes;
            stats.driver_unknown_device_bytes_available=true;
        }
        if(coherent_device && value.max_concurrency==2) {
            const char* batched=std::getenv("NINFER_EXL3_BATCHED_ROUNDS");
            if(!batched || std::string_view(batched)!="0") {
                const char* wait=std::getenv("NINFER_EXL3_BATCHED_ROUND_WAIT_US");
                batched_verify=std::make_unique<Exl3BatchedVerifyCoordinator>(
                    std::chrono::microseconds(wait?std::atoi(wait):5000));
                auto first=lanes[0]->context_owner_for_device_round();
                auto second=lanes[1]->context_owner_for_device_round();
                // Each lane may lead a batched forward: one graph per direction.
                first->prepare_batched_continuation_graph(*second,8,8);
                second->prepare_batched_continuation_graph(*first,8,8);
                first->set_batched_verify(batched_verify.get());
                second->set_batched_verify(batched_verify.get());
            }
        }
        try{for(std::uint32_t i=0;i<value.max_concurrency;++i)workers[i]=std::thread([this,i]{loop(i);});}
        catch(...){
            {std::lock_guard lock(mutex);stopping=true;}
            changed.notify_all();for(auto& worker:workers)if(worker.joinable())worker.join();throw;
        }
    }
    Exl3RetirementResult close() noexcept {
        const auto workspace_cleanup_failed=[]() noexcept {
            return Exl3DevicePageStorage::quarantined_bytes()!=0 || Exl3DevicePageFill::quarantined_records()!=0 ||
                Exl3DevicePrefixCache::budget_snapshot()[1]!=0 ||
                Exl3CudaLinearWorkspace::quarantined_workspaces()!=0 ||
                Exl3EngineGreedyPacketBatch::retirement_unresolved() ||
                Exl3Dflash2DraftModel::generic_quarantined_allocations()!=0 ||
                Exl3LayerBufferRetirement::quarantined()!=0 ||
                Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire)!=0 ||
                Exl3EngineTargetQ::packed_retirement_unresolved();
        };
        if(retirement.result().phase!=Exl3RetirementPhase::active) {
            if(workspace_cleanup_failed())retirement.observe_cleanup_failure(-2);
            return retirement.result();
        }
        try {
            {std::lock_guard lock(mutex);stopping=true;for(auto& r:active)if(r)r->cancelled=true;for(auto& r:queue)r->cancelled=true;}
            changed.notify_all();for(auto& worker:workers)if(worker.joinable())worker.join();
            retirement.begin(first_execution_failure!=nullptr);
            auto error=retirement_fault?static_cast<cudaError_t>(std::exchange(retirement_fault,0)):cudaSetDevice(options.device);
            if(error==cudaSuccess)error=cudaDeviceSynchronize();
            if(error!=cudaSuccess){retirement.complete(static_cast<int>(error));return retirement.result();}
            if(workspace_cleanup_failed()) {
                retirement.observe_cleanup_failure(-2);return retirement.result();
            }
            // A later successful drain does not erase a reconstruction failure
            // latch or an independently quarantined context. Retain the Engine
            // bundle before recycling lane/coordinator owners.
            if(Exl3TextContext::reconstruction_quarantined_contexts() || Exl3TextContext::reconstruction_quarantined_allocations()) {
                retirement.complete(-2);return retirement.result();
            }
            for(const auto& lane:lanes)if(lane && lane->context().reconstruction_retirement_uncertain()) {
                retirement.complete(-2);return retirement.result();
            }
            // No producer can enqueue another consumer after this certificate.
            for(auto& lane:lanes)if(lane) {
                lane->retire_after_device_drain();
                lane->context().retire_device_prefix_after_drain();
                const auto& lease=lane->lease();
                if(lease.root && coordinator.compute_leases_current(std::span(&lease,1)))
                    coordinator.cancel(lease.ticket,true);
            }
            if(shared_q)shared_q->retire_after_device_drain();
            if(greedy_packet_batch)greedy_packet_batch->retire_after_device_drain();
            if(Exl3EngineGreedyPacketBatch::retirement_unresolved()) {
                retirement.complete(-2);return retirement.result();
            }
            if(Exl3KVTransferLease::quarantined_records() || Exl3RegisteredKVBacking::quarantined_bytes()) {
                retirement.complete(-2);return retirement.result();
            }
            for(const auto& slots:kv_uploads)for(const auto& upload:slots)if(upload && upload->uncertain()) {
                retirement.complete(-2);return retirement.result();
            }
            for(const auto& lane:lanes)if(lane && lane->context().attention_staging_uncertain()) {
                retirement.complete(-2);return retirement.result();
            }
            if(shared_pages && shared_pages->uncertain_after_join()) {
                retirement.complete(-2);return retirement.result();
            }
            for(const auto& copy:page_copies)if(copy && copy->uncertain()) {
                retirement.complete(-2);return retirement.result();
            }
            for(const auto& readers:page_attention)for(const auto& attention:readers)if(attention && attention->uncertain()) {
                retirement.complete(-2);return retirement.result();
            }
            {std::lock_guard lock(mutex);shared_pages.reset();}
            coordinator.close();
            if(workspace_cleanup_failed()) {
                retirement.observe_cleanup_failure(-2);return retirement.result();
            }
            // Last inventory references may attempt unregistration during close.
            if(Exl3RegisteredKVBacking::quarantined_bytes() || Exl3DevicePageStorage::quarantined_bytes() ||
                Exl3DevicePageFill::quarantined_records()) {
                retirement.complete(-2);return retirement.result();
            }
            retirement.complete(0);
        } catch(...) {
            retirement.begin(first_execution_failure!=nullptr);
            retirement.complete(-1); // host join/residency cleanup failure, not a CUDA code
        }
        return retirement.result();
    }
    void emit(Request& request,const targets::qwen3_6::PublishedOutput& output){
        std::lock_guard lock(request.mutex);
        for(auto& delta:output){
            const auto content=request.result.content.size(),reasoning=request.result.reasoning.size();
            if(content>request.result_text_limit || reasoning>request.result_text_limit-content ||
                delta.text.size()>request.result_text_limit-content-reasoning)
                throw std::logic_error("Engine output exceeds tokenizer-derived text bound");
            if(delta.channel==OutputChannel::Reasoning && request.emitted_content)
                throw std::logic_error("Engine output returned to reasoning after content");
            const bool coalesce=request.pending_deltas &&
                request.deltas[request.pending_deltas-1].channel==delta.channel;
            if(request.consumer==OutputConsumerMode::Streaming && !coalesce && request.pending_deltas==request.deltas.size())
                throw std::logic_error("Engine output exceeded frontend channel bound");
            auto& text=delta.channel==OutputChannel::Reasoning?request.result.reasoning:request.result.content;
            const auto offset=text.size();
            if(!output_result_slots_available(offset,request.result_text_limit,text.capacity(),delta.text.size()))
                throw std::logic_error("Engine output exceeds preallocated text capacity");
            text+=delta.text;
            if(delta.channel==OutputChannel::Content)request.emitted_content=true;
            if(request.consumer==OutputConsumerMode::Streaming){
                // Offsets survive string growth; no borrowed pointer or duplicate
                // pending text allocation is retained by the queue.
                if(coalesce)request.deltas[request.pending_deltas-1].bytes+=delta.text.size();
                else request.deltas[request.pending_deltas++]={delta.channel,offset,delta.text.size()};
            }
        }
        request.changed.notify_all();
    }
    struct PreferredLane {
        std::size_t lane=0;
        bool affine=false,locality_costs_complete=false;
    };
    std::optional<PreferredLane> preferred_lane_locked(const Request& request) {
        if(!request_affinity || options.max_concurrency<2)return std::nullopt;
        std::array<Exl3ReadyLaneLocality,2> candidates{};
        const auto input=request.ready_work.input_tokens();
        for(std::size_t index=0;index<options.max_concurrency;++index) {
            auto& candidate=candidates[index];candidate.lane=index;
            candidate.physical_resources_feasible=!active[index] && lanes[index] && drafts[index] &&
                request.result_lease.has_value();
            if(!candidate.physical_resources_feasible)continue;
            const auto& signature=request.ready_work.signature();
            candidate.identity_compatible=signature.device==options.device &&
                Exl3ReadyExecutionSignature::same_owner(signature.model_owner,target);
            if(!candidate.identity_compatible)continue;
            auto& frontier=lane_frontiers[index];
            auto root=frontier.root.lock();
            if(!root) {frontier={};continue;}
            if(!lanes[index] || frontier.request_generation!=lanes[index]->context().request_generation()) {
                frontier={};continue;
            }
            const auto tokens=root->token_count();
            const bool root_compatible=tokens && tokens<=input.size() && !root->prepared_identity() &&
                root->state()->model_identity()==lanes[index]->context().model_identity() &&
                root->state()->position()<lanes[index]->context().max_context() &&
                lanes[index]->context().exact_host_state_resident(*root->state()) &&
                root->matches_tokens(input.first(tokens));
            if(!root_compatible)continue; // Lane remains a feasible cold fallback.
            candidate.reusable_tokens=tokens;candidate.cost=frontier.locality;
        }
        const auto selected=exl3_select_ready_lane(
            std::span<const Exl3ReadyLaneLocality>(candidates).first(options.max_concurrency));
        if(!selected)return std::nullopt;
        bool locality_costs_complete=true;
        for(const auto& candidate:std::span<const Exl3ReadyLaneLocality>(candidates).first(
                options.max_concurrency))
            if(candidate.physical_resources_feasible && candidate.identity_compatible &&
               !candidate.cost.total_microseconds())locality_costs_complete=false;
        return PreferredLane{*selected,candidates[*selected].reusable_tokens>0,
            locality_costs_complete};
    }
    struct ReadyRequestSelection {
        std::size_t index=0;
        bool affine=false,copy_deferred=false,copy_unknown=false,copy_fairness=false;
        bool locality_costs_complete=false;
        std::uint32_t queue_size=0,eligible=0,eligible_deferred=0;
        std::uint32_t cancelled_rejected=0,physical_rejected=0;
        std::uint32_t identity_rejected=0,affinity_deferred=0;
    };
    std::optional<ReadyRequestSelection> ready_request_locked(std::size_t lane_index) {
        std::optional<ReadyRequestSelection> selected;
        std::optional<Exl3ReadyRequestCandidate> selected_age;
        std::optional<ReadyRequestSelection> nonduplicate;
        std::optional<Exl3ReadyRequestCandidate> nonduplicate_age;
        Exl3ReadyCopyDemand selected_demand;
        bool selected_duplicate=false,all_copy_known=true;
        std::uint32_t eligible_count=0,cancelled_rejected=0,physical_rejected=0;
        std::uint32_t identity_rejected=0,affinity_deferred=0;
        const auto increment=[](std::uint32_t& value) noexcept {
            if(value!=std::numeric_limits<std::uint32_t>::max())++value;
        };
        const Exl3ReadyCopyDemand* active_copy=nullptr;
        if(copy_demand_provider && options.max_concurrency==2) {
            const auto& peer=active[1-lane_index];
            if(peer && peer->ready_work.copy_demand().actionable())
                active_copy=&peer->ready_work.copy_demand();
        }
        std::size_t index=0;
        for(const auto& request:queue) {
            bool eligible=false,affine=false,locality_costs_complete=false;
            const bool cancelled=request->cancelled.load(std::memory_order_acquire);
            const auto& signature=request->ready_work.signature();
            const bool identity_compatible=signature.device==options.device &&
                Exl3ReadyExecutionSignature::same_owner(signature.model_owner,target);
            const bool physical_resources=!active[lane_index] && lanes[lane_index] &&
                drafts[lane_index] && request->result_lease.has_value();
            if(cancelled)increment(cancelled_rejected);
            else if(!identity_compatible)increment(identity_rejected);
            else if(!physical_resources)increment(physical_rejected);
            else if(request_affinity) {
                const auto preferred=preferred_lane_locked(*request);
                eligible=preferred && preferred->lane==lane_index;
                affine=eligible && preferred->affine;
                locality_costs_complete=eligible && preferred->locality_costs_complete;
                if(!eligible)increment(affinity_deferred);
            } else {
                eligible=true;
            }
            if(eligible)increment(eligible_count);
            const Exl3ReadyRequestCandidate age{index,request->ready_work.submitted(),eligible,
                cancelled};
            const auto& demand=request->ready_work.copy_demand();
            const bool duplicate=active_copy && demand.actionable() &&
                demand.same_history(*active_copy);
            if(age.physically_eligible && !age.cancelled && active_copy) {
                if(!demand.transfer_bytes)all_copy_known=false;
                if(!duplicate && (!nonduplicate_age ||
                    exl3_ready_request_precedes(age,*nonduplicate_age))) {
                    nonduplicate=ReadyRequestSelection{index,affine};
                    nonduplicate->locality_costs_complete=locality_costs_complete;
                    nonduplicate_age=age;
                }
            }
            if(age.physically_eligible && !age.cancelled &&
               (!selected_age || exl3_ready_request_precedes(age,*selected_age))) {
                selected=ReadyRequestSelection{index,affine};
                selected->locality_costs_complete=locality_costs_complete;selected_age=age;
                selected_demand=demand;selected_duplicate=duplicate;
            }
            ++index;
        }
        if(selected && active_copy && selected_duplicate && selected_demand.actionable()) {
            const auto now=Clock::now();
            const auto age=now<=selected_age->submitted?std::chrono::microseconds::zero():
                std::chrono::duration_cast<std::chrono::microseconds>(now-selected_age->submitted);
            if(!all_copy_known)selected->copy_unknown=true;
            else if(age>=selected_demand.maximum_deferral)selected->copy_fairness=true;
            else if(nonduplicate) {
                nonduplicate->copy_deferred=true;selected=std::move(nonduplicate);
            }
        } else if(selected && active_copy && !all_copy_known)selected->copy_unknown=true;
        if(selected) {
            selected->queue_size=queue.size()>std::numeric_limits<std::uint32_t>::max()?
                std::numeric_limits<std::uint32_t>::max():static_cast<std::uint32_t>(queue.size());
            selected->eligible=eligible_count;
            selected->eligible_deferred=eligible_count?eligible_count-1:0;
            selected->cancelled_rejected=cancelled_rejected;
            selected->physical_rejected=physical_rejected;
            selected->identity_rejected=identity_rejected;
            selected->affinity_deferred=affinity_deferred;
        }
        return selected;
    }
    void publish_preview(Request& request,std::optional<
        Exl3ControlPublicationBoundary::Ticket> boundary={}) {
        if(boundary && !request.publication_boundary.stage_exposure(*boundary))
            throw std::logic_error("Engine output exposure precedes resident commit");
        if(!boundary && request.publication_boundary.snapshot().phase!=
                Exl3ControlPublicationBoundary::Phase::idle)
            throw std::logic_error("Engine unscoped output exposure during publication");
        try {
            request.require_output_storage();
            auto published=request.output.commit_preview();
            emit(request,published);
            request.output.recycle_output(std::move(published));
            request.require_output_storage();
            if(boundary && !request.publication_boundary.finish_exposure(*boundary))
                throw std::logic_error("Engine output exposure completion identity");
        } catch(...) {
            if(boundary)request.publication_boundary.fail(*boundary);
            throw;
        }
    }
    void execute_coherent_device(Request& request,std::size_t lane_index) {
        auto& lane=lanes[lane_index];
        auto& draft=*drafts[lane_index];
        auto& result=request.result;
        auto& output=request.output;
        const auto& prepared=targets::qwen3_6::PreparedPromptAccess::view(request.prompt);
        const auto ids=request.ready_work.input_tokens();
        if(prepared.has_media() || ids.size()<2048 ||
           ids.size()>options.max_context || request.sampling ||
           !request.options.stop.strings.empty() ||
           request.options.execution.thinking.budget.has_value() ||
           request.options.execution.sampling.temperature!=0 ||
           request.options.execution.sampling.presence_penalty!=0 ||
           request.options.execution.sampling.frequency_penalty!=0)
            throw std::invalid_argument(
                "coherent-device EXL3 requires C1 greedy text with saturated draft ring");
        if(request.cancelled) {result.finish_reason=FinishReason::Cancelled;return;}
        if(Clock::now()>=request.deadline)
            throw RequestError(RequestErrorKind::QueueTimeout,
                "EXL3 request queue deadline expired");
        auto context=lane->context_owner_for_device_round();
        const auto stream=streams[lane_index]->value;
        const auto prefill=Clock::now();
        std::optional<Exl3VeriCacheServingCoordinator::Lease> host_lease;
        std::optional<Exl3VeriCacheServingCoordinator::DeviceLogicalLease> device_lease;
        std::unique_ptr<Exl3FastDeviceRound> round;
        std::optional<std::uint64_t> pending_ticket;
        std::vector<std::int64_t> committed;
        auto root=Root{};
        std::vector<std::int64_t> terminal_tokens;
        if(request.options.stop.include_model_defaults)
            for(const auto token:frontend.default_stop_policy().token_ids)
                terminal_tokens.push_back(token);
        for(const auto token:request.options.stop.token_ids)
            terminal_tokens.push_back(token);
        if(const auto* value=std::getenv("NINFER_EXL3_TEST_ENGINE_ROOT_HASH");
           value && std::string_view(value)=="1") {
            std::fprintf(stderr,"COHERENT_DEVICE_ENGINE_TERMINAL ids=");
            for(const auto token:terminal_tokens)
                std::fprintf(stderr,"%lld,",static_cast<long long>(token));
            std::fprintf(stderr,"\n");
        }
        const auto& vericache=Exl3VeriCacheConfig::get();
        std::unique_lock prefill_lock(coherent_prefill_mutex);
        // Rounds already submitted by other lanes finish before ingestion.
        if(options.max_concurrency>1)check(cudaDeviceSynchronize());
        try {
            (void)context->reset_for_request(identity.contract());
            const auto epoch=context->request_generation();
            // VeriCache verifies against exact history: ingest the prompt exactly too.
            if(vericache.enabled)context->set_l0_exact_history(true);
            // Conversation reuse (NINFER_EXL3_COHERENT_ROOT_REUSE=1): the longest
            // retained device root whose tokens prefix this input is restored
            // (target exact state + draft ring) and only the suffix is ingested.
            static const bool root_reuse=[] {
                const auto* value=std::getenv("NINFER_EXL3_COHERENT_ROOT_REUSE");
                if(!value || !*value || std::string_view(value)=="0")return false;
                if(std::string_view(value)!="1")
                    throw std::invalid_argument("NINFER_EXL3_COHERENT_ROOT_REUSE must be 0 or 1");
                return true;
            }();
            const bool reuse_allowed=root_reuse && options.context_cache.enabled &&
                request.options.execution.allow_prefix_reuse && prepared.identity.reusable;
            std::shared_ptr<const Exl3VeriCacheRequest> reused;
            if(reuse_allowed)
                for(const auto& candidate:cache.roots())
                    if(candidate && candidate->compact_draft() && !candidate->prepared_identity() &&
                       candidate->token_count()>=64 && candidate->token_count()<ids.size() &&
                       (!reused || candidate->token_count()>reused->token_count()) &&
                       candidate->matches_tokens(std::span<const std::int64_t>(ids).first(candidate->token_count())))
                        reused=candidate;
            // The turn-closure frontier is the rendered prefix the next turn of
            // this conversation agrees with: it gets its own retained root.
            std::size_t stable=ids.size();
            if(prepared.identity.rewrite_checkpoint)
                stable=std::min<std::size_t>(stable,prepared.identity.rewrite_checkpoint->frontier);
            const bool split=reuse_allowed && stable>=64 && stable<ids.size() &&
                (!reused || stable>reused->token_count());
            const auto span_ids=std::span<const std::int64_t>(ids);
            const auto extend_device=[&](const std::shared_ptr<const Exl3VeriCacheRequest>& from,std::size_t end,
                                         bool resident) {
                const auto first=from->token_count();
                return from->append_device_prompt(*context,draft,span_ids.subspan(first,end-first),[&] {
                    Exl3FastDeviceRound ingest(context,draft,staging_owners[lane_index]->pointers,
                        1,epoch,false,stream);
                    ingest.ingest_suffix(span_ids.first(end),static_cast<int>(first));
                },stream,resident);
            };
            std::shared_ptr<const Exl3VeriCacheRequest> base=reused;
            bool base_resident=false;
            if(split) {
                if(base) base=extend_device(base,stable,false);
                else {
                    {
                        Exl3FastDeviceRound fresh(context,draft,staging_owners[lane_index]->pointers,
                            1,epoch,false,stream);
                        fresh.begin_fresh(span_ids.first(stable),stable>1040);
                    }
                    base=Exl3VeriCacheRequest::initialize_device_root(*context,draft,
                        span_ids.first(stable),stream);
                }
                base_resident=true;
                (void)cache.admit_input_authority(base,span_ids.first(stable),available());
            }
            if(base) {
                root=extend_device(base,ids.size(),base_resident);
            } else {
                Exl3FastDeviceRound fresh(context,draft,staging_owners[lane_index]->pointers,
                    1,epoch,false,stream);
                fresh.begin_fresh(ids,ids.size()>1040);
            }
            const bool root_diagnostic=[] {
                const auto* value=std::getenv("NINFER_EXL3_TEST_ENGINE_ROOT_HASH");
                return value && std::string_view(value)=="1";
            }();
            if(root_diagnostic) {
                const auto diagnostic_root=context->export_exact_host_state(stream);
                const auto first=exl3_branch_greedy(*context,stream);
                context->decode(first,stream);
                const auto second=exl3_branch_greedy(*context,stream);
                const auto host_logits=context->logits_host(stream);
                const auto host_second=static_cast<std::int64_t>(
                    std::max_element(host_logits.begin(),host_logits.end())-
                        host_logits.begin());
                context->restore_exact_host_state(*diagnostic_root,stream);
                const auto restored_first=exl3_branch_greedy(*context,stream);
                context->decode(restored_first,stream);
                const auto restored_second=exl3_branch_greedy(*context,stream);
                context->restore_exact_host_state(*diagnostic_root,stream);
                const auto restored_hash=context->export_exact_host_state(stream)->
                    represented_payload_hash_for_test();
                if(restored_hash!=diagnostic_root->represented_payload_hash_for_test())
                    throw std::logic_error("coherent device diagnostic root restore");
                std::fprintf(stderr,
                    "COHERENT_DEVICE_ENGINE_ROOT prompt=%zu hash=%llu transaction_prepared=%d scalar_pair=%lld,%lld host_second=%lld restored_pair=%lld,%lld\n",
                    ids.size(),static_cast<unsigned long long>(restored_hash),
                    static_cast<int>(context->transaction_prepared()),
                    static_cast<long long>(first),static_cast<long long>(second),
                    static_cast<long long>(host_second),
                    static_cast<long long>(restored_first),
                    static_cast<long long>(restored_second));
            }
            {
                const std::size_t reused_tokens=reused?reused->token_count():0;
                std::lock_guard lock(mutex);
                stats.computed_prefill_tokens+=ids.size()-reused_tokens;
                stats.reused_prompt_tokens+=reused_tokens;
                stats.last_selected_frontier_tokens=reused_tokens;
                if(reused) {++stats.shared_stable_prefix_selections;++stats.prefix_preparation_cache_hits;}
                else ++stats.root_selections;
                ++stats.prefix_preparation_returns;
            }
            result.reused_prompt_tokens=reused?static_cast<std::uint32_t>(reused->token_count()):0;
            result.prefix_reuse_path=reused?PrefixReusePath::SharedStablePrefix:PrefixReusePath::Root;
            result.timings.prefill_seconds=seconds(prefill);
            if(request.cancelled) {
                result.finish_reason=FinishReason::Cancelled;
                output.preview_terminal(FinishReason::Cancelled);
                publish_preview(request);
                return;
            }
            if(ids.size()==options.max_context) {
                result.finish_reason=FinishReason::ContextCapacity;
                output.preview_terminal(result.finish_reason);
                publish_preview(request);
                return;
            }
            const auto allowance=std::min<std::uint32_t>(
                request.options.execution.requested_output_tokens,
                options.max_context-static_cast<std::uint32_t>(ids.size()));
            const auto limit=allowance<request.options.execution.requested_output_tokens?
                FinishReason::ContextCapacity:FinishReason::OutputLimit;
            output.validate_generation_capacity(allowance);
            committed.reserve(allowance);
            if(!base)root=Exl3VeriCacheRequest::initialize_device_root(*context,draft,ids,stream);
            if(reuse_allowed && !split && ids.size()>=64) {
                // The rendered input is a stable prefix of the next turn even
                // when the generated reply is later re-rendered differently.
                const auto admitted=cache.admit_input_authority(root,ids,available());
                if(admitted.admitted) {std::lock_guard lock(mutex);++stats.completed_prefix_admissions;}
            }
            {
                std::lock_guard admission(admission_mutex);
                const auto ticket=coordinator.admit(root);
                host_lease=coordinator.acquire();
                if(!host_lease || host_lease->ticket.request_id!=ticket.request_id)
                    throw std::logic_error("coherent-device physical/coordinator acquisition");
            }
            draft.bind_ring_scope(host_lease->acquisition,epoch);
            // This lane's DFlash2 rounds may meet the other lane's at verification.
            struct BatchedLane {
                Exl3BatchedVerifyCoordinator* coordinator;
                explicit BatchedLane(Exl3BatchedVerifyCoordinator* value) : coordinator(value) {
                    if(coordinator)coordinator->set_active(true);
                }
                ~BatchedLane() {
                    if(!coordinator)return;
                    coordinator->set_active(false);
                    if(std::getenv("NINFER_EXL3_BATCHED_STATS"))std::fprintf(stderr,"BATCHED_ROUNDS batched=%llu solo=%llu\n",
                        static_cast<unsigned long long>(coordinator->batched_rounds()),static_cast<unsigned long long>(coordinator->solo_rounds()));
                }
            } batched_lane(batched_verify.get());
            round=std::make_unique<Exl3FastDeviceRound>(context,draft,
                staging_owners[lane_index]->pointers,host_lease->acquisition,
                epoch,false,stream);
            if(vericache.enabled)context->set_l0_exact_history(false);
            round->begin_prefilled();
            check(cudaStreamSynchronize(stream));
            prefill_lock.unlock();
            device_lease=coordinator.enter_device_logical(*host_lease,
                std::static_pointer_cast<const void>(lane),context->position(),
                allowance,options.max_context);
            host_lease.reset();
            if(acquired_root_observer)acquired_root_observer(root);
            result.speculative.enabled=true;
            result.speculative.backend=SpeculativeBackend::DFlash2;
            result.speculative.draft_window=7;
            const auto decode=Clock::now();
            std::uint32_t remaining=allowance;
            Exl3CompleteRouteCounters route;
            // Test-only host phase attribution (prepare / decide / settle /
            // publish), accumulated per request and printed after it.
            const bool round_phases=[] {
                const auto* value=std::getenv("NINFER_EXL3_TEST_ENGINE_ROUND_PHASES");
                return value && std::string_view(value)=="1";
            }();
            std::array<double,4> phase_ms{};
            auto phase_mark=Clock::now();
            const auto phase=[&](std::size_t index) {
                if(!round_phases)return;
                const auto now=Clock::now();
                phase_ms[index]+=std::chrono::duration<double,std::milli>(now-phase_mark).count();
                phase_mark=now;
            };
            if(vericache.enabled) {
                // Draft numerically ahead with L0 rounds, verify each block, then
                // publish only verified tokens (in <= 16-token output previews).
                round->begin_verification(vericache.block);
                std::vector<std::int64_t> unverified;
                bool terminal_seen=false;
                while(!request.cancelled && remaining) {
                    std::shared_lock round_lock(coherent_prefill_mutex);
                    if(!output.pending_control_tokens().empty())
                        throw std::invalid_argument(
                            "coherent-device EXL3 does not admit injected control tokens");
                    const auto budget=output.model_token_budget_remaining(remaining);
                    if(!budget)throw std::logic_error(
                        "coherent-device output has no model budget or control path");
                    const std::size_t room=budget>unverified.size()?budget-unverified.size():0;
                    const std::size_t block=static_cast<std::size_t>(vericache.block);
                    if(!terminal_seen && room && unverified.size()<block) {
                        const auto width=static_cast<int>(std::min<std::size_t>(
                            {8,room,block-unverified.size()}));
                        const auto prepared_round=round->prepare_device_pending(width,terminal_tokens);
                        pending_ticket=prepared_round.ticket;
                        const auto& candidate=prepared_round.candidate;
                        route.record_attempt(candidate.width,0,0);
                        route.record_target_work(candidate.verification.verification_rows,
                            candidate.verification.replay_rows);
                        if(result.speculative.first_proposed_tokens.empty())
                            for(int row=0;row<candidate.width;++row)
                                result.speculative.first_proposed_tokens.push_back(
                                    static_cast<TokenId>(candidate.proposal[row]));
                        Exl3FastDeviceRound::Step settled;
                        try {
                            settled=round->settle_device_pending(*pending_ticket,
                                candidate.committed_tokens.size());
                        } catch(...) {pending_ticket.reset();throw;}
                        pending_ticket.reset();
                        unverified.insert(unverified.end(),settled.committed_tokens.begin(),
                            settled.committed_tokens.end());
                        terminal_seen=settled.terminal;
                        ++result.speculative.rounds;
                        result.speculative.accepted_tokens+=settled.verification.accepted;
                        result.speculative.accepted_prefix_per_round.push_back(
                            static_cast<std::uint8_t>(settled.verification.accepted));
                        continue;
                    }
                    const auto verified=round->verify(unverified,vericache.delta);
                    unverified.clear();
                    terminal_seen=false;
                    std::vector<std::int64_t> retained;
                    bool finished=false;
                    for(std::size_t at=0;at<verified.tokens.size();) {
                        const auto chunk=std::span<const std::int64_t>(verified.tokens).subspan(
                            at,std::min<std::size_t>(16,verified.tokens.size()-at));
                        const auto boundary=request.publication_boundary.begin(
                            device_lease->acquisition,epoch,
                            Exl3ControlPublicationBoundary::Kind::model);
                        if(!request.publication_boundary.numerical_ready(boundary))
                            throw std::logic_error("coherent-device verified readiness ordering");
                        auto tokens=request.round_tokens.assign(chunk);
                        const auto decision=output.preview_model(tokens.span(),remaining,limit);
                        if(!request.publication_boundary.prepare_output(boundary))
                            throw std::logic_error("coherent-device verified preview ordering");
                        request.require_output_storage();
                        if(!decision.accepted_tokens) {
                            if(!request.publication_boundary.abandon_private(boundary))
                                throw std::logic_error("coherent-device verified empty preview ordering");
                            publish_preview(request);
                            if(decision.finished()) {result.finish_reason=decision.finish_reason;finished=true;}
                            break;
                        }
                        if(decision.accepted_tokens!=tokens.size())
                            tokens=request.round_tokens.truncate(decision.accepted_tokens);
                        const auto selected=request.round_tokens.repair().span();
                        if(selected.size()>remaining)
                            throw std::logic_error("coherent-device selected output exceeds allowance");
                        request.require_result_slots(selected.size());
                        if(!request.publication_boundary.resume_numerical(boundary) ||
                           !request.publication_boundary.numerical_ready(boundary) ||
                           !request.publication_boundary.prepare_output(boundary) ||
                           !request.publication_boundary.begin_publication(boundary))
                            throw std::logic_error("coherent-device verified publication ordering");
                        const auto publication=coordinator.publish_device_logical_window(
                            *device_lease,selected);
                        device_lease=publication.lease;
                        if(!request.publication_boundary.resident_committed(boundary))
                            throw std::logic_error("coherent-device verified logical commit ordering");
                        committed.insert(committed.end(),selected.begin(),selected.end());
                        retained.insert(retained.end(),selected.begin(),selected.end());
                        result.generated_token_ids.insert(
                            result.generated_token_ids.end(),tokens.begin(),tokens.end());
                        remaining-=static_cast<std::uint32_t>(selected.size());
                        const bool hidden_terminal=decision.finish_reason==FinishReason::StopToken &&
                            !request.options.stop.publish_stop_token;
                        if(hidden_terminal)++result.token_accounting.hidden_terminal_tokens;
                        result.token_accounting.visible_model_tokens+=
                            selected.size()-(hidden_terminal?1u:0u);
                        route.record_publication(selected.size(),
                            selected.size()-(hidden_terminal?1u:0u),hidden_terminal?1u:0u);
                        if(result.timings.first_token_seconds==0)
                            result.timings.first_token_seconds=
                                seconds(request.submitted)+result.timings.prepare_seconds;
                        publish_preview(request,boundary);
                        at+=selected.size();
                        if(decision.finished()) {result.finish_reason=decision.finish_reason;finished=true;break;}
                        if(selected.size()<chunk.size())break;
                    }
                    round->commit_verification(retained);
                    if(finished)break;
                }
                // Unverified drafted tokens (cancellation) are never published: rewind.
                if(round->frontier()!=round->verified_frontier())round->commit_verification({});
                if(std::getenv("NINFER_EXL3_VERICACHE_STATS")) {
                    const auto& t=round->totals();
                    std::fprintf(stderr,"VERICACHE blocks=%llu corrected=%llu verify_ms=%.1f block=%d delta=%.3f "
                        "checked=%llu off_greedy=%llu mean_gap=%.5f max_gap=%.4f "
                        "restore_ms=%.1f forward_ms=%.1f scores_ms=%.1f fix_ms=%.1f commit_ms=%.1f\n",
                        static_cast<unsigned long long>(t.verified_blocks),
                        static_cast<unsigned long long>(t.corrected_blocks),t.verify_ms,
                        vericache.block,vericache.delta,
                        static_cast<unsigned long long>(t.accepted_checked),
                        static_cast<unsigned long long>(t.accepted_off_greedy),
                        t.accepted_checked?t.accepted_gap_sum/static_cast<double>(t.accepted_checked):0.0,
                        static_cast<double>(t.max_accepted_gap),t.restore_ms,t.forward_ms,t.scores_ms,
                        t.fix_ms,t.commit_ms);
                }
            } else
            while(remaining && !request.cancelled) {
                std::shared_lock round_lock(coherent_prefill_mutex);
                if(round_phases)phase_mark=Clock::now();
                if(!output.pending_control_tokens().empty())
                    throw std::invalid_argument(
                        "coherent-device EXL3 does not admit injected control tokens");
                const auto budget=output.model_token_budget_remaining(remaining);
                if(!budget)throw std::logic_error(
                    "coherent-device output has no model budget or control path");
                const auto width=static_cast<int>(std::min<std::uint32_t>(8,budget));
                const auto boundary=request.publication_boundary.begin(
                    device_lease->acquisition,epoch,
                    Exl3ControlPublicationBoundary::Kind::model);
                const auto prepared_round=round->prepare_device_pending(width,
                    terminal_tokens);
                phase(0);
                pending_ticket=prepared_round.ticket;
                const auto& candidate=prepared_round.candidate;
                if(root_diagnostic && result.speculative.rounds==0) {
                    std::fprintf(stderr,"COHERENT_DEVICE_ENGINE_FIRST accepted=%zu committed=",
                        candidate.verification.accepted);
                    for(const auto token:candidate.committed_tokens)
                        std::fprintf(stderr,"%lld,",static_cast<long long>(token));
                    std::fprintf(stderr," proposal=");
                    for(int row=0;row<candidate.width;++row)
                        std::fprintf(stderr,"%lld,",
                            static_cast<long long>(candidate.proposal[row]));
                    std::fprintf(stderr,"\n");
                }
                route.record_attempt(candidate.width,0,0);
                route.record_target_work(candidate.verification.verification_rows,
                    candidate.verification.replay_rows);
                if(result.speculative.first_proposed_tokens.empty())
                    for(int row=0;row<candidate.width;++row)
                        result.speculative.first_proposed_tokens.push_back(
                            static_cast<TokenId>(candidate.proposal[row]));
                if(!request.publication_boundary.numerical_ready(boundary))
                    throw std::logic_error("coherent-device numerical readiness ordering");
                if(request.cancelled) {
                    round->cancel_device_pending(*pending_ticket);
                    pending_ticket.reset();
                    if(!request.publication_boundary.abandon_private(boundary))
                        throw std::logic_error("coherent-device cancellation ordering");
                    break;
                }
                auto tokens=request.round_tokens.assign(candidate.committed_tokens);
                const auto decision=output.preview_model(tokens.span(),remaining,limit);
                if(!request.publication_boundary.prepare_output(boundary))
                    throw std::logic_error("coherent-device output preview ordering");
                request.require_output_storage();
                if(!decision.accepted_tokens || request.cancelled) {
                    round->cancel_device_pending(*pending_ticket);
                    pending_ticket.reset();
                    if(!request.publication_boundary.abandon_private(boundary))
                        throw std::logic_error("coherent-device empty preview ordering");
                    if(request.cancelled) {output.discard_preview();break;}
                    publish_preview(request);
                    ++result.speculative.rounds;
                    if(decision.finished()) {
                        result.finish_reason=decision.finish_reason;
                        break;
                    }
                    continue;
                }
                if(decision.accepted_tokens!=tokens.size())
                    tokens=request.round_tokens.truncate(decision.accepted_tokens);
                const auto selected=request.round_tokens.repair().span();
                if(selected.size()>remaining)
                    throw std::logic_error("coherent-device selected output exceeds allowance");
                request.require_result_slots(selected.size());
                if(!request.publication_boundary.resume_numerical(boundary))
                    throw std::logic_error("coherent-device settlement ordering");
                phase(1);
                Exl3FastDeviceRound::Step settled;
                try {
                    settled=round->settle_device_pending(*pending_ticket,
                        selected.size(),[&](const Exl3FastDeviceRound::Step&) {
                            if(request.cancelled)throw RequestError(
                                RequestErrorKind::Cancelled,
                                "coherent-device request cancelled before numerical commit");
                        });
                } catch(...) {
                    pending_ticket.reset();
                    throw;
                }
                pending_ticket.reset();
                phase(2);
                if(!request.publication_boundary.numerical_ready(boundary) ||
                   !request.publication_boundary.prepare_output(boundary) ||
                   !request.publication_boundary.begin_publication(boundary))
                    throw std::logic_error("coherent-device resident publication ordering");
                const auto publication=coordinator.publish_device_logical_window(
                    *device_lease,selected);
                device_lease=publication.lease;
                if(!request.publication_boundary.resident_committed(boundary))
                    throw std::logic_error("coherent-device logical commit ordering");
                committed.insert(committed.end(),selected.begin(),selected.end());
                result.generated_token_ids.insert(
                    result.generated_token_ids.end(),tokens.begin(),tokens.end());
                remaining-=static_cast<std::uint32_t>(selected.size());
                const bool hidden_terminal=decision.finish_reason==FinishReason::StopToken &&
                    !request.options.stop.publish_stop_token;
                if(hidden_terminal)++result.token_accounting.hidden_terminal_tokens;
                result.token_accounting.visible_model_tokens+=
                    selected.size()-(hidden_terminal?1u:0u);
                route.record_publication(selected.size(),
                    selected.size()-(hidden_terminal?1u:0u),hidden_terminal?1u:0u);
                if(result.timings.first_token_seconds==0)
                    result.timings.first_token_seconds=
                        seconds(request.submitted)+result.timings.prepare_seconds;
                publish_preview(request,boundary);
                ++result.speculative.rounds;
                result.speculative.accepted_tokens+=settled.verification.accepted;
                result.speculative.accepted_prefix_per_round.push_back(
                    static_cast<std::uint8_t>(settled.verification.accepted));
                phase(3);
                if(decision.finished()) {
                    result.finish_reason=decision.finish_reason;
                    break;
                }
            }
            if(request.cancelled && result.finish_reason==FinishReason::None) {
                result.finish_reason=FinishReason::Cancelled;
                output.preview_terminal(FinishReason::Cancelled);
                publish_preview(request);
            } else if(result.finish_reason==FinishReason::None) {
                result.finish_reason=limit;
                output.preview_terminal(limit);
                publish_preview(request);
            }
            round->finish();
            if(round_phases)
                std::fprintf(stderr,
                    "COHERENT_DEVICE_ENGINE_ROUND_PHASES rounds=%zu prepare_ms=%.3f decide_ms=%.3f settle_ms=%.3f publish_ms=%.3f\n",
                    static_cast<std::size_t>(result.speculative.rounds),
                    phase_ms[0],phase_ms[1],phase_ms[2],phase_ms[3]);
            if(!result.token_accounting.conserves_result_tokens(
                result.generated_token_ids.size()))
                throw std::logic_error("coherent-device result token accounting");
            result.timings.decode_seconds=seconds(decode);
            result.reasoning_tokens=output.reasoning_tokens();
            result.thinking=output.thinking_stats();
            const auto totals=round->totals();
            const auto route_snapshot=route.snapshot();
            result.speculative.drafted_tokens=
                route_snapshot.proposed_rows-result.speculative.rounds;
            result.speculative.verifier_calls=totals.rounds;
            result.speculative.proposed_rows=route_snapshot.proposed_rows;
            result.speculative.verified_rows=route_snapshot.verified_rows;
            result.speculative.replayed_rows=route_snapshot.replayed_rows;
            result.speculative.committed_model_rows=route_snapshot.committed_model_rows;
            result.speculative.externally_visible_model_rows=
                route_snapshot.externally_visible_model_rows;
            result.speculative.hidden_terminal_rows=route_snapshot.hidden_terminal_rows;
            if(!committed.empty()) {
                auto child=root->append_device_terminal(*context,draft,committed,stream);
                host_lease=coordinator.materialize_device_logical(*device_lease,
                    std::move(child));
                device_lease.reset();
                if(terminal_root_observer)terminal_root_observer(host_lease->root);
                if(options.context_cache.enabled &&
                   request.options.execution.allow_prefix_reuse &&
                   !request.cancelled && prepared.identity.reusable &&
                   output.completed_chat_turn() &&
                   host_lease->root->state()->position()>=64) {
                    const auto admitted=cache.admit_completed_authority(
                        host_lease->root,available());
                    if(admitted.admitted) {
                        std::lock_guard lock(mutex);
                        ++stats.completed_prefix_admissions;
                    }
                }
                coordinator.complete(*host_lease);
                host_lease.reset();
            } else {
                coordinator.cancel_device_logical(*device_lease,true);
                device_lease.reset();
            }
            // Test-only semantic final-state witness, observed after all timed
            // work and publication so graph and eager policies can be compared.
            if(const auto* value=std::getenv("NINFER_EXL3_TEST_ENGINE_FINAL_STATE_HASH");
               value && std::string_view(value)=="1")
                std::fprintf(stderr,"COHERENT_DEVICE_ENGINE_FINAL_STATE output=%zu hash=%llu\n",
                    result.generated_token_ids.size(),static_cast<unsigned long long>(
                        context->export_exact_host_state(stream)->represented_payload_hash_for_test()));
            std::lock_guard lock(mutex);
            stats.committed_decode_tokens+=result.generated_token_ids.size();
            stats.visible_model_tokens+=result.token_accounting.visible_model_tokens;
            stats.hidden_terminal_tokens+=result.token_accounting.hidden_terminal_tokens;
            stats.decode_rounds+=result.speculative.rounds;
        } catch(...) {
            const bool uncertain_publication=
                request.publication_boundary.snapshot().phase==
                    Exl3ControlPublicationBoundary::Phase::publication_pending;
            request.publication_boundary.fail_active();
            bool numerical_safe=!round || !round->poisoned();
            if(pending_ticket && round) {
                try {round->cancel_device_pending(*pending_ticket);}
                catch(...) {numerical_safe=false;}
            }
            if(numerical_safe) {
                try {
                    if(device_lease)coordinator.cancel_device_logical(*device_lease,true);
                    else if(host_lease)coordinator.cancel(host_lease->ticket,true);
                } catch(...) {numerical_safe=false;}
            }
            if(!numerical_safe || uncertain_publication) {
                lane->poison_shared_completion();
                std::lock_guard lock(mutex);
                failed=true;
            }
            throw;
        }
    }
    void execute(Request& request,std::size_t lane_index){
        if(coherent_device) {
            execute_coherent_device(request,lane_index);
            return;
        }
        auto& lane=lanes[lane_index];auto& draft=drafts[lane_index];auto& staging=staging_owners[lane_index]->pointers;
        auto& result=request.result;
        if(request.cancelled){result.finish_reason=FinishReason::Cancelled;return;}
        if(Clock::now()>=request.deadline)throw RequestError(RequestErrorKind::QueueTimeout,"EXL3 request queue deadline expired");
        auto& output=request.output;
        const auto& prepared=targets::qwen3_6::PreparedPromptAccess::view(request.prompt);
        Exl3PublicMediaQualification::require_current_request(
            public_media_modality(prepared));
        const auto ids=request.ready_work.input_tokens();
        const auto fingerprint=request.ready_work.input_fingerprint();
        const auto host_ticket=request.host_preparation.begin(
            request.host_preparation_generation);
        if(!host_ticket) {
            if(request.cancelled)
                throw RequestError(RequestErrorKind::Cancelled,
                    "EXL3 host preparation cancelled");
            throw std::logic_error("Engine host preparation generation unavailable");
        }
        std::size_t cacheable=0;bool declared_frontier=false;
        try {
            if(options.context_cache.enabled && request.options.execution.allow_prefix_reuse &&
               prepared.identity.reusable)
                cacheable=ids.size(); // Supplied input only; never generated text.
            if(cacheable && prepared.identity.rewrite_checkpoint)
                cacheable=std::min<std::size_t>(cacheable,
                    prepared.identity.rewrite_checkpoint->frontier);
            if(cacheable<64)cacheable=0;
            if(cacheable && declared_prefix_preparation) {
                const auto frontier=exl3_declared_prefix_frontier(prepared,cacheable);
                if(frontier){cacheable=frontier;declared_frontier=true;}
            }
            const Exl3HostPreparationCompletion::Plan plan{
                fingerprint,ids.size(),cacheable,declared_frontier};
            if(!request.host_preparation.complete(*host_ticket,plan)) {
                if(request.cancelled)
                    throw RequestError(RequestErrorKind::Cancelled,
                        "EXL3 host preparation cancelled");
                throw std::logic_error("Engine host preparation completion identity");
            }
            const bool peer_numerical=options.max_concurrency==2 &&
                numerical_active[1-lane_index].load(std::memory_order_acquire);
            if(host_preparation_observer)
                host_preparation_observer(lane_index,host_ticket->generation,
                    peer_numerical);
            if(request.cancelled) {
                request.host_preparation.cancel(host_ticket->generation);
                std::lock_guard lock(mutex);
                ++stats.host_preparation_cancelled_before_accept;
                throw RequestError(RequestErrorKind::Cancelled,
                    "EXL3 host preparation cancelled");
            }
            const auto accepted=request.host_preparation.accept(*host_ticket,
                fingerprint,ids.size());
            if(!accepted)throw std::logic_error(
                "Engine host preparation acceptance identity");
            cacheable=accepted->cacheable_tokens;
            std::lock_guard lock(mutex);
            ++stats.host_preparation_returns;
            if(peer_numerical)++stats.host_preparation_peer_numeric_overlaps;
        } catch(...) {
            const auto failure=std::current_exception();
            try {request.host_preparation.fail(*host_ticket);}catch(...) {}
            std::rethrow_exception(failure);
        }
        struct NumericalOwner {
            std::atomic<bool>& active;
            explicit NumericalOwner(std::atomic<bool>& value):active(value) {
                if(active.exchange(true,std::memory_order_acq_rel))
                    throw std::logic_error("Engine lane numerical ownership overlap");
            }
            ~NumericalOwner(){active.store(false,std::memory_order_release);}
        } numerical_owner(numerical_active[lane_index]);
        const auto prefill=Clock::now();
        const auto preparation_progress=[&](int position){
            if(prefix_preparation_observer)prefix_preparation_observer(lane_index,position);
            if(request.cancelled)
                throw RequestError(RequestErrorKind::Cancelled,"EXL3 prefill cancelled");
        };
        const bool media=prepared.has_media();
        auto root=media?cache.prepare_media(lane->context(),*vision,request.prompt,
                cacheable==ids.size(),available(),[&]{return request.cancelled.load();},
                [&](std::uint64_t replay_bytes,std::uint64_t output_bytes) {
                    Exl3EncodedMediaRetentionCredits credits;
                    credits.replay.emplace(coordinator.reserve_preparation_payload(
                        request.result_owner,replay_bytes));
                    credits.output.emplace(coordinator.reserve_preparation_payload(
                        request.result_owner,output_bytes));
                    return credits;
                }):
            (concurrent_prefix_preparation?
                cache.prepare_concurrent(lane->context(),ids,cacheable,available(),options.prefill_chunk,
                    preparation_progress,draft.get(),&staging,true):
                cache.prepare(lane->context(),ids,cacheable,available(),options.prefill_chunk,
                    preparation_progress,draft.get(),&staging,true));
        {
            // Preparation has released cache locks. Publish only successful
            // returns here; later cancellation does not erase completed work.
            std::lock_guard lock(mutex);
            ++stats.prefix_preparation_returns;
            if(concurrent_prefix_preparation)++stats.concurrent_prefix_preparation_returns;
            if(root.metrics.preparation_reused)++stats.prefix_preparation_joins;
            if(root.metrics.cache_hit)++stats.prefix_preparation_cache_hits;
        }
        result.reused_prompt_tokens=static_cast<std::uint32_t>(root.metrics.reused_prompt_tokens);
        result.prefix_reuse_path=(root.metrics.cache_hit || root.metrics.preparation_reused)?
            PrefixReusePath::SharedStablePrefix:PrefixReusePath::Root;
        {
            // Preparation has already evaluated every non-reused prompt token.
            // Attribute that work and the selected frontier here so a later
            // cancellation, output failure or max-context return cannot erase
            // work that actually happened. This is a count, never a latency or
            // avoided-time estimate.
            std::lock_guard lock(mutex);
            stats.computed_prefill_tokens+=ids.size()-result.reused_prompt_tokens;
            stats.reused_prompt_tokens+=result.reused_prompt_tokens;
            stats.last_selected_frontier_tokens=result.reused_prompt_tokens;
            if(result.prefix_reuse_path==PrefixReusePath::SharedStablePrefix)
                ++stats.shared_stable_prefix_selections;
            else
                ++stats.root_selections;
        }
        // wait() may set cancellation as preparation returns (including an
        // immediate cache hit with no progress callback). Retain shared cache
        // authority, but do not compact or admit this cancelled consumer.
        if(request.cancelled)
            throw RequestError(RequestErrorKind::Cancelled,"EXL3 prefill cancelled");
        auto compact=root.request->compact_draft(*draft,staging,lane->context().request_metadata_reservation());root.request.reset();
        if(!media && cacheable && ids.size()>=64) {
            std::optional<std::uint64_t> cold_preparation;
            if(!root.metrics.cache_hit && !root.metrics.preparation_reused) {
                const auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-prefill).count();
                if(elapsed>=0)cold_preparation=static_cast<std::uint64_t>(elapsed);
            }
            cache.admit_input_authority(compact,ids,available(),std::nullopt,cold_preparation);
        }
        result.timings.prefill_seconds=seconds(prefill);
        if(ids.size()==options.max_context){result.finish_reason=FinishReason::ContextCapacity;output.preview_terminal(result.finish_reason);publish_preview(request);return;}
        std::optional<Exl3VeriCacheServingCoordinator::Lease> lease;
        {
            // Each frontend request already owns one physical worker. Keep its
            // admission/acquisition atomic so another worker cannot take its root.
            std::lock_guard admission(admission_mutex);
            Exl3VeriCacheServingCoordinator::Ticket ticket;
            try {ticket=coordinator.admit(compact);}
            catch(const Exl3HostResidentBudgetExhausted&) {
                // No lease or physical lane attachment exists yet. A declared
                // residency limit rejects this request without poisoning peers.
                throw RequestError(RequestErrorKind::Overloaded,"EXL3 request exceeds available resident page budget");
            }
            lease=coordinator.acquire();
            if(!lease || lease->ticket.request_id!=ticket.request_id)
                throw std::logic_error("EXL3 physical request/coordinator binding");
        }
        bool acquired=false;
        const auto collect_conditional=[&] {
            if(lane->has_conditional_retention() && !lane->collect_conditional_retention(coordinator))
                throw std::logic_error("Engine conditional retention still has live owners");
        };
        try {
            const auto cancel_attachment=[&](unsigned phase) {
                auto expected=phase;
                if(phase && attachment_cancel_phase.compare_exchange_strong(expected,0)) {
                    attachment_cancel_hit.store(phase,std::memory_order_release);
                    request.cancelled.store(true,std::memory_order_release);
                }
                return request.cancelled.load(std::memory_order_acquire);
            };
            const auto restore_cancel_layer=
                attachment_restore_cancel_layer.exchange(0,std::memory_order_acq_rel);
            Exl3Dflash2Execution::AttachmentRestoreProgress restore_progress;
            if(restore_cancel_layer)restore_progress=[&](int completed_layer) {
                if(static_cast<unsigned>(completed_layer+1)==restore_cancel_layer) {
                    attachment_restore_cancel_hit.store(
                        static_cast<unsigned>(completed_layer),std::memory_order_release);
                    request.cancelled.store(true,std::memory_order_release);
                }
            };
            if(!lane->acquire(*lease,&coordinator,cancel_attachment,restore_progress))
                throw RequestError(RequestErrorKind::Cancelled,
                    "EXL3 request cancelled during physical prefix attachment");
            acquired=true;compact.reset();
            if(request.sampling)request.sampling->bind_acquisition(
                lease->acquisition,lane->lease().root->revision_owner(),
                static_cast<std::size_t>(lane->lease().root->state()->position()));
            if(acquired_root_observer)acquired_root_observer(lane->lease().root);
            if(verifier_cost_menu)lane->set_horizon_cost_menu(*verifier_cost_menu,lane->execution_epoch(),lease->acquisition);
            if(suffix_selection_limits)lane->set_suffix_selection_limits(*suffix_selection_limits,lane->execution_epoch(),lease->acquisition);
            auto remaining=std::min<std::uint32_t>(request.options.execution.requested_output_tokens,options.max_context-static_cast<std::uint32_t>(ids.size()));
            const auto limit=remaining<request.options.execution.requested_output_tokens?FinishReason::ContextCapacity:FinishReason::OutputLimit;
            output.validate_generation_capacity(remaining);
            const auto decode=Clock::now();const auto stats0=lane->stats();
            const auto oscar0=lane->context().oscar_telemetry();
            Exl3CompleteRouteCounters complete_route;
            result.speculative.enabled=true;result.speculative.backend=SpeculativeBackend::DFlash2;result.speculative.draft_window=7;
            const auto abort_round=[&](Exl3ControlPublicationBoundary::Ticket boundary) {
                if(request.publication_boundary.snapshot().phase!=
                        Exl3ControlPublicationBoundary::Phase::numerical_pending &&
                   !request.publication_boundary.resume_numerical(boundary))
                    throw std::logic_error("Engine rollback numerical ordering");
                lane->abort();
                if(!request.publication_boundary.numerical_ready(boundary) ||
                   !request.publication_boundary.abandon_private(boundary))
                    throw std::logic_error("Engine rollback completion ordering");
            };
            const auto arm_publication_fault=[&] {
                if(request.window_publication_fault.exchange(false,std::memory_order_acq_rel))
                    coordinator.fail_next_residency_for_test(
                        Exl3VeriCacheServingCoordinator::ResidencyFault::before_commit);
            };
            while(remaining && !request.cancelled){
                collect_conditional();
                const auto control=output.pending_control_tokens();
                if(!control.empty()) {
                    request.require_result_slots(control.size());
                    if(control.size()>remaining)throw std::logic_error("Engine control exceeds remaining output allowance");
                    const auto tokens=request.control_ids;
                    if(control.size()!=tokens.size() || !std::equal(control.begin(),control.end(),tokens.begin()))
                        throw std::logic_error("Engine pending control differs from reserved suffix");
                    const auto boundary=request.publication_boundary.begin(
                        lease->acquisition,lane->execution_epoch(),
                        Exl3ControlPublicationBoundary::Kind::control);
                    auto pending=lane->apply_control_owned(tokens);
                    if(!request.publication_boundary.numerical_ready(boundary))
                        throw std::logic_error("Engine control completion ordering");
                    output.preview_control(control,remaining);
                    if(!request.publication_boundary.prepare_output(boundary))
                        throw std::logic_error("Engine control output preparation ordering");
                    request.require_output_storage();
                    if(request.cancelled){
                        abort_round(boundary);output.discard_preview();
                        break;
                    }
                    if(!request.publication_boundary.begin_publication(boundary))
                        throw std::logic_error("Engine control publication ordering");
                    arm_publication_fault();
                    lane->publish(coordinator,*pending);remaining-=static_cast<std::uint32_t>(tokens.size());
                    if(!request.publication_boundary.resident_committed(boundary))
                        throw std::logic_error("Engine control resident commit ordering");
                    if(request.sampling)
                        Exl3SamplingControlStopSemantics::commit_forced_control_after_resident(
                            *request.sampling,tokens.size(),lane->lease().root->revision_owner());
                    result.generated_token_ids.insert(result.generated_token_ids.end(),control.begin(),control.end());
                    result.token_accounting.injected_control_tokens+=control.size();
                    publish_preview(request,boundary);continue;
                }
                const auto boundary=request.publication_boundary.begin(
                    lease->acquisition,lane->execution_epoch(),
                    Exl3ControlPublicationBoundary::Kind::model);
                auto proposed=lane->propose(std::min<std::uint32_t>(8,output.model_token_budget_remaining(remaining)));
                if(result.speculative.first_proposed_tokens.empty())
                    result.speculative.first_proposed_tokens.assign(
                        proposed.begin(),proposed.end());
                complete_route.record_attempt(proposed.size(),0,0);
                if(!request.publication_boundary.numerical_ready(boundary))
                    throw std::logic_error("Engine proposal completion ordering");
                if(request.cancelled){abort_round(boundary);break;}
                if(!request.publication_boundary.resume_numerical(boundary))
                    throw std::logic_error("Engine verification numerical ordering");
                auto pending=lane->verify_owned(proposed);
                complete_route.record_target_work(
                    pending->verification.verification_rows,pending->verification.replay_rows);
                if(!request.publication_boundary.numerical_ready(boundary))
                    throw std::logic_error("Engine model completion ordering");
                if(request.cancelled){
                    abort_round(boundary);
                    break;
                }
                auto tokens=request.round_tokens.assign(pending->verification.committed_tokens);
                auto decision=output.preview_model(tokens.span(),remaining,limit);
                if(!request.publication_boundary.prepare_output(boundary))
                    throw std::logic_error("Engine model output preparation ordering");
                if(conditional_b8 && output.model_token_budget_remaining(remaining)>=16 &&
                    pending->proposed_rows==8 && pending->costs.neural_rows==8 &&
                    pending->verification.accepted==8 && !pending->verification.rejected &&
                    !pending->verification.stopped && tokens.size()==8 &&
                    decision.accepted_tokens==8 && !decision.finished() &&
                    decision.continuation==runtime::ContinuationAction::Decode) {
                    // First preview proves no stop/control boundary before
                    // the dependent block; nothing has been published yet.
                    output.discard_preview();
                    if(!request.publication_boundary.resume_numerical(boundary))
                        throw std::logic_error("Engine conditional numerical ordering");
                    const auto cancel_at=[&](unsigned stage) {
                        auto expected=stage;
                        if(conditional_cancel_stage.compare_exchange_strong(expected,0)) {
                            conditional_cancel_hit.store(stage);request.cancelled=true;
                        }
                    };
                    cancel_at(1);
                    if(request.cancelled){
                        abort_round(boundary);
                        break;
                    }
                    const auto second=lane->try_propose_conditional_second(*pending,16,coordinator);
                    if(second)complete_route.record_attempt(second->tokens.size(),0,0);
                    if(second)cancel_at(2);
                    if(request.cancelled){
                        abort_round(boundary);
                        break;
                    }
                    if(second) {
                        const auto verified=pending->verification.verification_rows;
                        const auto replayed=pending->verification.replay_rows;
                        pending=lane->verify_conditional_second_owned(*pending,*second);
                        if(pending->verification.verification_rows<verified ||
                           pending->verification.replay_rows<replayed)
                            throw std::logic_error("Engine conditional route counters regressed");
                        complete_route.record_target_work(
                            pending->verification.verification_rows-verified,
                            pending->verification.replay_rows-replayed);
                    }
                    if(second)cancel_at(3);
                    if(!request.publication_boundary.numerical_ready(boundary))
                        throw std::logic_error("Engine conditional completion ordering");
                    if(request.cancelled){
                        abort_round(boundary);
                        break;
                    }
                    tokens=request.round_tokens.assign(pending->verification.committed_tokens);
                    decision=output.preview_model(tokens.span(),remaining,limit);
                    if(!request.publication_boundary.prepare_output(boundary))
                        throw std::logic_error("Engine conditional output preparation ordering");
                }
                request.require_output_storage();
                if(decision.accepted_tokens!=tokens.size()){
                    tokens=request.round_tokens.truncate(decision.accepted_tokens);
                    if(!tokens.empty()) {
                        if(!request.publication_boundary.resume_numerical(boundary))
                            throw std::logic_error("Engine repair numerical ordering");
                        pending=lane->repair_verified_prefix_owned(*pending,tokens.size());
                        complete_route.record_target_work(pending->verification.verification_rows,
                            pending->verification.replay_rows);
                        if(!request.publication_boundary.numerical_ready(boundary) ||
                           !request.publication_boundary.prepare_output(boundary))
                            throw std::logic_error("Engine repair completion ordering");
                    } else {
                        abort_round(boundary);
                    }
                }
                if(!tokens.empty()){
                    request.require_result_slots(tokens.size());
                    if(tokens.size()>remaining)throw std::logic_error("Engine verification exceeds remaining output allowance");
                    if(request.cancelled){
                        abort_round(boundary);output.discard_preview();
                        break;
                    }
                    if(!request.publication_boundary.begin_publication(boundary))
                        throw std::logic_error("Engine model publication ordering");
                    arm_publication_fault();
                    lane->publish(coordinator,*pending);
                    if(!request.publication_boundary.resident_committed(boundary))
                        throw std::logic_error("Engine model resident commit ordering");
                    result.generated_token_ids.insert(result.generated_token_ids.end(),tokens.begin(),tokens.end());remaining-=static_cast<std::uint32_t>(tokens.size());
                    const bool hidden_terminal=decision.finish_reason==FinishReason::StopToken &&
                        !request.options.stop.publish_stop_token;
                    if(hidden_terminal)++result.token_accounting.hidden_terminal_tokens;
                    result.token_accounting.visible_model_tokens+=tokens.size()-(hidden_terminal?1u:0u);
                    complete_route.record_publication(tokens.size(),
                        tokens.size()-(hidden_terminal?1u:0u),hidden_terminal?1u:0u);
                    if(result.timings.first_token_seconds==0)result.timings.first_token_seconds=seconds(request.submitted)+result.timings.prepare_seconds;
                }
                // No sink sees data until L2 repair + resident ownership commit.
                if(tokens.empty())publish_preview(request);
                else publish_preview(request,boundary);
                ++result.speculative.rounds;
                result.speculative.accepted_tokens+=tokens.empty()?0:pending->accepted_draft_rows();
                result.speculative.accepted_prefix_per_round.push_back(
                    static_cast<std::uint8_t>(pending->accepted_draft_rows()));
                if(decision.finished()){result.finish_reason=decision.finish_reason;break;}
            }
            if(request.cancelled && result.finish_reason==FinishReason::None){result.finish_reason=FinishReason::Cancelled;output.preview_terminal(FinishReason::Cancelled);publish_preview(request);}
            else if(result.finish_reason==FinishReason::None){result.finish_reason=limit;output.preview_terminal(limit);publish_preview(request);}
            if(!result.token_accounting.conserves_result_tokens(result.generated_token_ids.size()))
                throw std::logic_error("Engine semantic/result token accounting mismatch");
            result.timings.decode_seconds=seconds(decode);result.reasoning_tokens=output.reasoning_tokens();result.thinking=output.thinking_stats();
            result.speculative.drafted_tokens=lane->stats().proposed_rows-stats0.proposed_rows;
            // Includes completed unpublished/cancelled work and frontend repair.
            result.speculative.verifier_calls=lane->stats().completed_native_verifier_calls-stats0.completed_native_verifier_calls;
            result.speculative.repair_checkpoint_captured_bytes=
                lane->stats().repair_checkpoint_captured_bytes-stats0.repair_checkpoint_captured_bytes;
            result.speculative.repair_checkpoint_restores=
                lane->stats().repair_checkpoint_restores-stats0.repair_checkpoint_restores;
            result.speculative.repair_checkpoint_reconstructed_rows=
                lane->stats().repair_checkpoint_reconstructed_rows-stats0.repair_checkpoint_reconstructed_rows;
            result.speculative.repair_checkpoint_fallback_rows=
                lane->stats().repair_checkpoint_fallback_rows-stats0.repair_checkpoint_fallback_rows;
            result.speculative.device_seed_handoffs=
                lane->stats().device_seed_handoffs-stats0.device_seed_handoffs;
            result.speculative.device_seed_host_fallbacks=
                lane->stats().device_seed_host_fallbacks-stats0.device_seed_host_fallbacks;
            result.speculative.draft_local_topk_calls=
                lane->stats().draft_local_topk_calls-stats0.draft_local_topk_calls;
            result.speculative.draft_dense_kmajor_launches=
                lane->stats().draft_dense_kmajor_launches-stats0.draft_dense_kmajor_launches;
            const auto oscar1=lane->context().oscar_telemetry();
            result.speculative.oscar_eager_cohort_attempts=
                oscar1.continuation_cohort_eager_attempts-
                    oscar0.continuation_cohort_eager_attempts;
            result.speculative.oscar_eager_cohort_dispatches=
                oscar1.continuation_cohort_eager_dispatches-
                    oscar0.continuation_cohort_eager_dispatches;
            result.speculative.oscar_eager_cohort_latch_misses=
                oscar1.continuation_cohort_eager_latch_misses-
                    oscar0.continuation_cohort_eager_latch_misses;
            result.speculative.oscar_eager_cohort_boundary_fallbacks=
                oscar1.continuation_cohort_eager_boundary_fallbacks-
                    oscar0.continuation_cohort_eager_boundary_fallbacks;
            result.speculative.oscar_eager_cohort_malformed=
                oscar1.continuation_cohort_eager_malformed-
                    oscar0.continuation_cohort_eager_malformed;
            result.speculative.staged_b8_verifier_calls=
                lane->stats().staged_b8_verifier_calls-stats0.staged_b8_verifier_calls;
            result.speculative.staged_b8_first_half_exits=
                lane->stats().staged_b8_first_half_exits-stats0.staged_b8_first_half_exits;
            result.speculative.staged_b8_second_half_calls=
                lane->stats().staged_b8_second_half_calls-stats0.staged_b8_second_half_calls;
            result.speculative.staged_b8_skipped_verification_rows=
                lane->stats().staged_b8_skipped_verification_rows-
                    stats0.staged_b8_skipped_verification_rows;
            result.speculative.committed_tap_device_bytes=
                lane->stats().committed_tap_device_bytes-stats0.committed_tap_device_bytes;
            result.speculative.committed_tap_host_export_rows_avoided=
                lane->stats().committed_tap_host_export_rows_avoided-
                    stats0.committed_tap_host_export_rows_avoided;
            result.speculative.neural_input_rows=lane->stats().neural_input_rows-stats0.neural_input_rows;
            result.speculative.returned_suffix_rows=lane->stats().neural_returned_suffix_rows-stats0.neural_returned_suffix_rows;
            result.speculative.discarded_suffix_rows=lane->stats().neural_discarded_suffix_rows-stats0.neural_discarded_suffix_rows;
            const auto route=complete_route.snapshot();
            result.speculative.proposed_rows=route.proposed_rows;
            result.speculative.verified_rows=route.verified_rows;
            result.speculative.replayed_rows=route.replayed_rows;
            result.speculative.committed_model_rows=route.committed_model_rows;
            result.speculative.externally_visible_model_rows=route.externally_visible_model_rows;
            result.speculative.hidden_terminal_rows=route.hidden_terminal_rows;
            result.speculative.diagnostic_rows=result.token_accounting.diagnostic_state_rows;
            result.speculative.shared_projection_rows=std::nullopt;
            const auto final=lane->lease();
            // Diagnostic-only immutable host snapshot, outside coordinator and
            // queue locks. Disabled Engine execution pays no snapshot copy.
            if(terminal_root_observer)terminal_root_observer(final.root);
            // Consume the committed frontend semantic boundary. Custom stop
            // strings/tokens, discarded previews and output limits do not close
            // a conversational turn even if they end generation.
            if(options.context_cache.enabled && request.options.execution.allow_prefix_reuse &&
                !request.cancelled && prepared.identity.reusable && output.completed_chat_turn() &&
                final.root->state()->position()>=64) {
                const auto admission=cache.admit_completed_authority(final.root,available());
                if(admission.admitted) {
                    std::lock_guard lock(mutex);++stats.completed_prefix_admissions;
                }
            }
            collect_conditional();
            const auto retired=lane->release();
            coordinator.complete_retired(final,retired);acquired=false;
            if(request_affinity) {
                std::lock_guard lock(mutex);
                Exl3ReadyLocalityCost locality;
                locality.root_attachment_microseconds=measured_microseconds(root.metrics.attach_ms);
                if(root.metrics.cache_hit || root.metrics.preparation_reused)
                    locality.history_restore_microseconds=measured_microseconds(
                        root.metrics.prefill_ms-root.metrics.attach_ms);
                lane_frontiers[lane_index]={final.root,lane->context().request_generation(),locality};
            }
        }catch(...){
            const auto failure=std::current_exception();
            request.publication_boundary.fail_active();
            try {
                if(acquired && lane->execution_active()) {
                    lane->abort();collect_conditional();auto final=lane->lease();
                    const auto retired=lane->release();
                    coordinator.cancel_retired(final,retired);acquired=false;
                } else if(!acquired)coordinator.cancel(lease->ticket,true);
                // A poisoned active lease stays retained until Engine drains.
            } catch(...) {lane->poison_shared_completion();}
            std::rethrow_exception(failure);
        }
        std::lock_guard lock(mutex);
        stats.committed_decode_tokens+=result.generated_token_ids.size();
        stats.visible_model_tokens+=result.token_accounting.visible_model_tokens;
        stats.injected_control_tokens+=result.token_accounting.injected_control_tokens;
        stats.hidden_terminal_tokens+=result.token_accounting.hidden_terminal_tokens;
        stats.decode_rounds+=result.speculative.rounds;
    }
    void retire_result(Request& request) {
        if(request.result_lease && coordinator.retire_logical_host_to_lifetime(*request.result_lease,
            [&](RetainedDescriptorLedger::Ticket credit) noexcept {
                return attach_bounded_retirement_credit<Request::Storage>(request.result_owner,std::move(credit));
            }))request.result_owner.reset();
    }
    void cancel_queued(Request& request) {
        // Request cancellation has already published its atomic flag. Do this
        // before taking the queue mutex; rendezvous admission has its own lock.
        if(shared_q)shared_q->notify_cancellation();
        if(greedy_packet_batch)greedy_packet_batch->notify_cancellation();
        std::lock_guard lock(mutex);
        const auto found=std::find_if(queue.begin(),queue.end(),[&](const auto& queued){return queued.get()==&request;});
        if(found==queue.end())return; // Active/completed ownership stays with its worker.
        {
            std::lock_guard request_lock(request.mutex);
            request.result.finish_reason=FinishReason::Cancelled;
            request.result.timings.total_seconds=seconds(request.submitted)+request.result.timings.prepare_seconds;
            request.done=true;
        }
        queue.erase(found);--outstanding;
        ready_work_authority.revise();
        stats.waiting_requests=static_cast<std::uint32_t>(queue.size());
        request.changed.notify_all();changed.notify_all();
    }
    void loop(std::size_t lane_index){
        for(;;){
            std::shared_ptr<Request> request;
            std::optional<Exl3ReadyWorkSnapshotAuthority::Grant> work_grant;
            std::exception_ptr selection_failure;
            bool affine=false,copy_deferred=false,copy_unknown=false,copy_fairness=false;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock,[&] {
                    if(stopping)return true;
                    if(queue.empty())return false;
                    return ready_request_locked(lane_index).has_value();
                });
                if(queue.empty() && stopping)break;
                auto selected=ready_request_locked(lane_index);
                if(!selected && stopping)selected=ReadyRequestSelection{0,false};
                if(!selected)continue;
                affine=selected->affine;
                copy_deferred=selected->copy_deferred;
                copy_unknown=selected->copy_unknown;
                copy_fairness=selected->copy_fairness;
                auto position=queue.begin()+static_cast<std::ptrdiff_t>(selected->index);
                request=*position;
                ready_work_authority.revise(); // selected queue membership changes below
                const auto observation=request->ready_work.observation();
                const auto state=[&] {return Exl3ReadyWorkState{
                    request->host_preparation_generation,true,true,
                    request->result_lease.has_value() && lanes[lane_index] && drafts[lane_index],
                    request->cancelled.load(std::memory_order_acquire)};};
                try {
                    auto ticket=ready_work_authority.snapshot(observation,state(),lane_index);
                    work_grant=ready_work_authority.consume(ticket,observation,state());
                    if(!work_grant && !request->cancelled.load(std::memory_order_acquire))
                        selection_failure=std::make_exception_ptr(
                            std::logic_error("EXL3 ready-work snapshot changed during selection"));
                } catch(...) {selection_failure=std::current_exception();}
                if(ready_decision_trace_enabled && work_grant) {
                    const auto reason=selected->copy_deferred?
                        Exl3ReadyDecisionReason::CopyNonduplicate:
                        selected->copy_unknown?Exl3ReadyDecisionReason::UnknownCopyFallback:
                        selected->copy_fairness?Exl3ReadyDecisionReason::CopyFairnessOverride:
                        selected->affine?Exl3ReadyDecisionReason::ExactLocality:
                        Exl3ReadyDecisionReason::OldestFeasible;
                    ready_decision_trace.append({
                        .authority_revision=work_grant->revision,
                        .generation=observation.generation,
                        .input_fingerprint=observation.input_fingerprint,
                        .queue_size=selected->queue_size,
                        .eligible=selected->eligible,
                        .eligible_deferred=selected->eligible_deferred,
                        .cancelled_rejected=selected->cancelled_rejected,
                        .physical_rejected=selected->physical_rejected,
                        .identity_rejected=selected->identity_rejected,
                        .affinity_deferred=selected->affinity_deferred,
                        .lane=static_cast<std::uint8_t>(lane_index),
                        .affinity_enabled=request_affinity,
                        .locality_costs_complete=selected->locality_costs_complete,
                        .reason=reason});
                }
                queue.erase(position);active[lane_index]=request;
                lane_frontiers[lane_index]={};
                if(request_affinity) {
                    if(affine)++stats.request_affinity_assignments;
                    else ++stats.request_affinity_fallbacks;
                }
                if(copy_deferred)++stats.ready_copy_hint_deferrals;
                if(copy_unknown)++stats.ready_copy_hint_unknown_fallbacks;
                if(copy_fairness)++stats.ready_copy_hint_fairness_overrides;
                stats.waiting_requests=static_cast<std::uint32_t>(queue.size());++stats.running_requests;
            }
            changed.notify_all(); // A different idle lane may now own the next oldest feasible request.
            const auto host_kv_before=lanes[lane_index]->context().host_kv_stats();
            const auto recurrent_export_before=
                lanes[lane_index]->context().recurrent_export_stats();
            const auto reconstruction_before=lanes[lane_index]->context().reconstructed_exact_stats();
            const auto paired_before=lanes[lane_index]->context().paired_transform_submissions();
            const auto k6_stream_before=lanes[lane_index]->context().stream_reduction_calls(false);
            const auto extended_stream_before=lanes[lane_index]->context().stream_reduction_calls(true);
            const auto k6_gateup_warpgroup_before=
                lanes[lane_index]->context().k6_gateup_warpgroup_async_calls();
            const auto k6_gateup_n32_pair_cta_calls_before=
                lanes[lane_index]->context().k6_gateup_n32_pair_cta_calls();
            const auto k6_gateup_n32_pair_cta_rows_before=
                lanes[lane_index]->context().k6_gateup_n32_pair_cta_rows();
            const auto k6_fast_decode_calls_before=
                lanes[lane_index]->context().k6_fast_decode_calls();
            const auto k6_fast_decode_rows_before=
                lanes[lane_index]->context().k6_fast_decode_rows();
            const auto k6_rowpair_n64_calls_before=
                lanes[lane_index]->context().k6_rowpair_n64_calls();
            const auto k6_rowpair_n64_rows_before=
                lanes[lane_index]->context().k6_rowpair_n64_rows();
            const auto k6_down_rowpair_calls_before=
                lanes[lane_index]->context().k6_down_rowpair_calls();
            const auto k6_down_rowpair_rows_before=
                lanes[lane_index]->context().k6_down_rowpair_rows();
            const auto shape4_n64_calls_before=
                lanes[lane_index]->context().shape4_n64_calls();
            const auto shape4_n64_rows_before=
                lanes[lane_index]->context().shape4_n64_rows();
            const auto reduce_shfl_min_barrier_calls_before=
                lanes[lane_index]->context().reduce_shfl_min_barrier_calls();
            const auto reduce_shfl_min_barrier_rows_before=
                lanes[lane_index]->context().reduce_shfl_min_barrier_rows();
            const auto k7_tiles64_exact_splits_before=
                lanes[lane_index]->context().k7_tiles64_exact_splits_calls();
            const auto target_k8_kv_prefill_async_a_calls_before=
                lanes[lane_index]->context().target_k8_kv_prefill_async_a_calls();
            const auto target_k8_kv_prefill_async_a_rows_before=
                lanes[lane_index]->context().target_k8_kv_prefill_async_a_rows();
            const auto target_down_k6_async_a_before=
                lanes[lane_index]->context().target_down_k6_async_a_calls();
            const auto target_gateup_k6_n16_before=
                lanes[lane_index]->context().target_gateup_k6_n16_calls();
            const auto host_kv_gdn_segment_graph_before=
                lanes[lane_index]->context().host_kv_gdn_segment_graph_stats();
            const auto host_kv_full_layer_graph_before=
                lanes[lane_index]->context().host_kv_full_layer_graph_stats();
            const auto host_kv_mlp_tail_graph_before=
                lanes[lane_index]->context().host_kv_mlp_tail_graph_stats();
            const auto target_k6_small_m_async_a_before=
                lanes[lane_index]->context().target_k6_small_m_async_a_calls();
            const auto target_k7_small_m_async_a_before=
                lanes[lane_index]->context().target_k7_small_m_async_a_calls();
            const auto target_k5_small_m_batch_before=
                lanes[lane_index]->context().target_k5_small_m_batch_calls();
            const auto eager_mlp_gateup_concurrent_before=
                lanes[lane_index]->context().eager_mlp_gateup_concurrent_calls();
            const auto prefill_qkv_concurrent_calls_before=
                lanes[lane_index]->context().prefill_qkv_concurrent_calls();
            const auto prefill_qkv_concurrent_rows_before=
                lanes[lane_index]->context().prefill_qkv_concurrent_rows();
            const auto prefill_projection_graph_captures_before=
                lanes[lane_index]->context().prefill_projection_graph_captures();
            const auto prefill_projection_graph_replays_before=
                lanes[lane_index]->context().prefill_projection_graph_replays();
            const auto prefill_projection_graph_binding_fallbacks_before=
                lanes[lane_index]->context().prefill_projection_graph_binding_fallbacks();
            const auto prefill_projection_chain_graph_captures_before=
                lanes[lane_index]->context().prefill_projection_chain_graph_captures();
            const auto prefill_projection_chain_graph_replays_before=
                lanes[lane_index]->context().prefill_projection_chain_graph_replays();
            const auto prefill_projection_chain_graph_fallbacks_before=
                lanes[lane_index]->context().prefill_projection_chain_graph_fallbacks();
            const auto target_prefill_gate_up_pair_attempts_before=
                lanes[lane_index]->context().target_prefill_gate_up_pair_attempts();
            const auto target_prefill_gate_up_pair_calls_before=
                lanes[lane_index]->context().target_prefill_gate_up_pair_calls();
            const auto target_prefill_gate_up_pair_rows_before=
                lanes[lane_index]->context().target_prefill_gate_up_pair_rows();
            const auto gqa_six_v_tile_launch_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_v_tile_launch_attempts();
            const auto gqa_six_v_tile_row_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_v_tile_row_attempts();
            const auto gqa_six_full_cta_launch_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_full_cta_launch_attempts();
            const auto gqa_six_full_cta_row_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_full_cta_row_attempts();
            const auto gqa_six_threads128_launch_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_threads128_launch_attempts();
            const auto gqa_six_threads128_row_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_threads128_row_attempts();
            const auto gqa_six_score_tile_launch_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_score_tile_launch_attempts();
            const auto gqa_six_score_tile_row_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_score_tile_row_attempts();
            const auto gqa_six_scalar_dim_launch_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_scalar_dim_launch_attempts();
            const auto gqa_six_scalar_dim_row_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_scalar_dim_row_attempts();
            const auto gqa_six_two_query_launch_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_two_query_launch_attempts();
            const auto gqa_six_two_query_row_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_two_query_row_attempts();
            const auto gqa_six_fused_launch_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_fused_launch_attempts();
            const auto gqa_six_fused_row_before=
                lanes[lane_index]->context().gqa_six_softmax_triple_fused_row_attempts();
            const auto gqa_six_tile512_launch_before=
                lanes[lane_index]->context().gqa_six_softmax_tile512_launch_attempts();
            const auto gqa_six_tile512_row_before=
                lanes[lane_index]->context().gqa_six_softmax_tile512_row_attempts();
            const auto head_before=lanes[lane_index]->context().head_work_stats();
            const auto policy_before=lanes[lane_index]->stats();
            const auto fused_before=lanes[lane_index]->context().fused_gate_up_submissions();
            const auto residual_before=lanes[lane_index]->context().fused_residual_norm_submissions();
            try {
                if(selection_failure)std::rethrow_exception(selection_failure);
                if(lane_assignment_observer)lane_assignment_observer(lane_index,affine);
                if(request_start_observer)request_start_observer();
                const auto readiness=request->ready_work.assess({
                    request->host_preparation_generation,true,true,
                    request->result_lease.has_value() && lanes[lane_index] && drafts[lane_index],
                    request->cancelled.load(std::memory_order_acquire)});
                if(!work_grant || readiness.blocked==Exl3ReadyWorkBlock::Cancelled)
                    request->result.finish_reason=FinishReason::Cancelled;
                else {
                    if(!readiness.physically_executable)
                        throw std::logic_error("assigned EXL3 ready work is not physically executable");
                    if(work_grant->observation.generation!=request->ready_work.generation() ||
                       work_grant->lane!=lane_index)
                        throw std::logic_error("consumed EXL3 ready-work grant identity");
                    check(cudaSetDevice(options.device));
                    execute(*request,lane_index);
                }
            } catch(const RequestError& error) {
                if(error.kind()==RequestErrorKind::Cancelled)request->result.finish_reason=FinishReason::Cancelled;
                else request->failure=std::current_exception();
            } catch(...) {
                request->failure=std::current_exception();
                std::lock_guard lock(mutex);
                if(!first_execution_failure)first_execution_failure=request->failure;
                failed=true;
                // Stop logical work, never retire buffers here. Peers observe
                // cancellation before publication and keep their own cleanup
                // obligations until the owning shutdown drain completes.
                for(auto& peer:active)if(peer && peer!=request && !peer->cancelled.exchange(true))
                    ++stats.failure_cancelled_active_requests;
                for(auto& queued:queue)if(!queued->cancelled.exchange(true))
                    ++stats.failure_cancelled_queued_requests;
            }
            // Transfer work happened even if the request was cancelled or later
            // failed. Publish once, before notifying completion, on every exit.
            const auto host_kv_after=lanes[lane_index]->context().host_kv_stats();
            const auto recurrent_export_after=
                lanes[lane_index]->context().recurrent_export_stats();
            const auto reconstruction_after=lanes[lane_index]->context().reconstructed_exact_stats();
            const auto paired_after=lanes[lane_index]->context().paired_transform_submissions();
            const auto k6_stream_after=lanes[lane_index]->context().stream_reduction_calls(false);
            const auto extended_stream_after=lanes[lane_index]->context().stream_reduction_calls(true);
            const auto k6_gateup_warpgroup_after=
                lanes[lane_index]->context().k6_gateup_warpgroup_async_calls();
            const auto k6_gateup_n32_pair_cta_calls_after=
                lanes[lane_index]->context().k6_gateup_n32_pair_cta_calls();
            const auto k6_gateup_n32_pair_cta_rows_after=
                lanes[lane_index]->context().k6_gateup_n32_pair_cta_rows();
            const auto k6_fast_decode_calls_after=
                lanes[lane_index]->context().k6_fast_decode_calls();
            const auto k6_fast_decode_rows_after=
                lanes[lane_index]->context().k6_fast_decode_rows();
            const auto k6_rowpair_n64_calls_after=
                lanes[lane_index]->context().k6_rowpair_n64_calls();
            const auto k6_rowpair_n64_rows_after=
                lanes[lane_index]->context().k6_rowpair_n64_rows();
            const auto k6_down_rowpair_calls_after=
                lanes[lane_index]->context().k6_down_rowpair_calls();
            const auto k6_down_rowpair_rows_after=
                lanes[lane_index]->context().k6_down_rowpair_rows();
            const auto shape4_n64_calls_after=
                lanes[lane_index]->context().shape4_n64_calls();
            const auto shape4_n64_rows_after=
                lanes[lane_index]->context().shape4_n64_rows();
            const auto reduce_shfl_min_barrier_calls_after=
                lanes[lane_index]->context().reduce_shfl_min_barrier_calls();
            const auto reduce_shfl_min_barrier_rows_after=
                lanes[lane_index]->context().reduce_shfl_min_barrier_rows();
            const auto k7_tiles64_exact_splits_after=
                lanes[lane_index]->context().k7_tiles64_exact_splits_calls();
            const auto target_k8_kv_prefill_async_a_calls_after=
                lanes[lane_index]->context().target_k8_kv_prefill_async_a_calls();
            const auto target_k8_kv_prefill_async_a_rows_after=
                lanes[lane_index]->context().target_k8_kv_prefill_async_a_rows();
            const auto target_down_k6_async_a_after=
                lanes[lane_index]->context().target_down_k6_async_a_calls();
            const auto target_gateup_k6_n16_after=
                lanes[lane_index]->context().target_gateup_k6_n16_calls();
            const auto host_kv_gdn_segment_graph_after=
                lanes[lane_index]->context().host_kv_gdn_segment_graph_stats();
            const auto host_kv_full_layer_graph_after=
                lanes[lane_index]->context().host_kv_full_layer_graph_stats();
            const auto host_kv_mlp_tail_graph_after=
                lanes[lane_index]->context().host_kv_mlp_tail_graph_stats();
            const auto target_k6_small_m_async_a_after=
                lanes[lane_index]->context().target_k6_small_m_async_a_calls();
            const auto target_k7_small_m_async_a_after=
                lanes[lane_index]->context().target_k7_small_m_async_a_calls();
            const auto target_k5_small_m_batch_after=
                lanes[lane_index]->context().target_k5_small_m_batch_calls();
            const auto eager_mlp_gateup_concurrent_after=
                lanes[lane_index]->context().eager_mlp_gateup_concurrent_calls();
            const auto prefill_qkv_concurrent_calls_after=
                lanes[lane_index]->context().prefill_qkv_concurrent_calls();
            const auto prefill_qkv_concurrent_rows_after=
                lanes[lane_index]->context().prefill_qkv_concurrent_rows();
            const auto prefill_projection_graph_captures_after=
                lanes[lane_index]->context().prefill_projection_graph_captures();
            const auto prefill_projection_graph_replays_after=
                lanes[lane_index]->context().prefill_projection_graph_replays();
            const auto prefill_projection_graph_binding_fallbacks_after=
                lanes[lane_index]->context().prefill_projection_graph_binding_fallbacks();
            const auto prefill_projection_chain_graph_captures_after=
                lanes[lane_index]->context().prefill_projection_chain_graph_captures();
            const auto prefill_projection_chain_graph_replays_after=
                lanes[lane_index]->context().prefill_projection_chain_graph_replays();
            const auto prefill_projection_chain_graph_fallbacks_after=
                lanes[lane_index]->context().prefill_projection_chain_graph_fallbacks();
            const auto target_prefill_gate_up_pair_attempts_after=
                lanes[lane_index]->context().target_prefill_gate_up_pair_attempts();
            const auto target_prefill_gate_up_pair_calls_after=
                lanes[lane_index]->context().target_prefill_gate_up_pair_calls();
            const auto target_prefill_gate_up_pair_rows_after=
                lanes[lane_index]->context().target_prefill_gate_up_pair_rows();
            const auto gqa_six_v_tile_launch_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_v_tile_launch_attempts();
            const auto gqa_six_v_tile_row_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_v_tile_row_attempts();
            const auto gqa_six_full_cta_launch_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_full_cta_launch_attempts();
            const auto gqa_six_full_cta_row_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_full_cta_row_attempts();
            const auto gqa_six_threads128_launch_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_threads128_launch_attempts();
            const auto gqa_six_threads128_row_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_threads128_row_attempts();
            const auto gqa_six_score_tile_launch_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_score_tile_launch_attempts();
            const auto gqa_six_score_tile_row_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_score_tile_row_attempts();
            const auto gqa_six_scalar_dim_launch_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_scalar_dim_launch_attempts();
            const auto gqa_six_scalar_dim_row_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_scalar_dim_row_attempts();
            const auto gqa_six_two_query_launch_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_two_query_launch_attempts();
            const auto gqa_six_two_query_row_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_two_query_row_attempts();
            const auto gqa_six_fused_launch_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_fused_launch_attempts();
            const auto gqa_six_fused_row_after=
                lanes[lane_index]->context().gqa_six_softmax_triple_fused_row_attempts();
            const auto gqa_six_tile512_launch_after=
                lanes[lane_index]->context().gqa_six_softmax_tile512_launch_attempts();
            const auto gqa_six_tile512_row_after=
                lanes[lane_index]->context().gqa_six_softmax_tile512_row_attempts();
            const auto head_after=lanes[lane_index]->context().head_work_stats();
            const auto fused_after=lanes[lane_index]->context().fused_gate_up_submissions();
            const auto residual_after=lanes[lane_index]->context().fused_residual_norm_submissions();
            {
                std::lock_guard lock(mutex);
                stats.shared_device_prefix_hit_bytes+=host_kv_after.shared_prefix_hit_bytes-host_kv_before.shared_prefix_hit_bytes;
                stats.shared_device_prefix_busy_fallbacks+=host_kv_after.shared_prefix_busy_fallbacks-host_kv_before.shared_prefix_busy_fallbacks;
                stats.private_device_prefix_hit_bytes+=
                    host_kv_after.device_prefix_hit_bytes-host_kv_before.device_prefix_hit_bytes;
                stats.private_device_prefix_fill_bytes+=
                    host_kv_after.device_prefix_fill_bytes-host_kv_before.device_prefix_fill_bytes;
                stats.private_device_prefix_segmented_bytes+=
                    host_kv_after.device_prefix_segmented_bytes-
                    host_kv_before.device_prefix_segmented_bytes;
                stats.private_device_prefix_forward_publish_forwards+=
                    host_kv_after.device_prefix_forward_publish_forwards-
                    host_kv_before.device_prefix_forward_publish_forwards;
                stats.private_device_prefix_partial_hit_bytes+=
                    host_kv_after.device_prefix_partial_hit_bytes-
                    host_kv_before.device_prefix_partial_hit_bytes;
                stats.private_device_prefix_forward_publish_bytes+=
                    host_kv_after.device_prefix_forward_publish_bytes-
                    host_kv_before.device_prefix_forward_publish_bytes;
                stats.host_kv_h2d_bytes+=host_kv_after.h2d_bytes-host_kv_before.h2d_bytes;
                stats.host_kv_d2h_bytes+=host_kv_after.d2h_bytes-host_kv_before.d2h_bytes;
                stats.host_kv_transfer_calls+=
                    host_kv_after.transfer_calls-host_kv_before.transfer_calls;
                stats.host_kv_copy_submissions+=
                    host_kv_after.copy_submissions-host_kv_before.copy_submissions;
                stats.host_kv_pinned_slot_waits+=
                    host_kv_after.pinned_slot_waits-host_kv_before.pinned_slot_waits;
                stats.host_kv_pinned_staging_bytes=std::max(
                    stats.host_kv_pinned_staging_bytes,
                    host_kv_after.pinned_staging_bytes);
                stats.exact_kv_export_batched_calls+=
                    host_kv_after.exact_kv_export_batched_calls-
                    host_kv_before.exact_kv_export_batched_calls;
                stats.exact_kv_export_batched_ranges+=
                    host_kv_after.exact_kv_export_batched_ranges-
                    host_kv_before.exact_kv_export_batched_ranges;
                stats.exact_kv_export_batched_bytes+=
                    host_kv_after.exact_kv_export_batched_bytes-
                    host_kv_before.exact_kv_export_batched_bytes;
                stats.exact_kv_export_batched_fences+=
                    host_kv_after.exact_kv_export_batched_fences-
                    host_kv_before.exact_kv_export_batched_fences;
                stats.exact_kv_export_saved_fences+=
                    host_kv_after.exact_kv_export_saved_fences-
                    host_kv_before.exact_kv_export_saved_fences;
                stats.prefill_pinned_batch_plane_calls+=
                    host_kv_after.prefill_pinned_batch_plane_calls-
                    host_kv_before.prefill_pinned_batch_plane_calls;
                stats.prefill_pinned_batch_page_planes+=
                    host_kv_after.prefill_pinned_batch_page_planes-
                    host_kv_before.prefill_pinned_batch_page_planes;
                stats.prefill_pinned_batch_bytes+=
                    host_kv_after.prefill_pinned_batch_bytes-
                    host_kv_before.prefill_pinned_batch_bytes;
                stats.host_kv_banked_d2h_forwards+=
                    host_kv_after.banked_d2h_forwards-host_kv_before.banked_d2h_forwards;
                stats.host_kv_banked_d2h_planes+=
                    host_kv_after.banked_d2h_planes-host_kv_before.banked_d2h_planes;
                stats.host_kv_banked_d2h_rows+=
                    host_kv_after.banked_d2h_rows-host_kv_before.banked_d2h_rows;
                stats.host_kv_banked_d2h_bytes+=
                    host_kv_after.banked_d2h_bytes-host_kv_before.banked_d2h_bytes;
                stats.host_kv_banked_d2h_drains+=
                    host_kv_after.banked_d2h_drains-host_kv_before.banked_d2h_drains;
                stats.registered_kv_upload_bytes+=host_kv_after.registered_upload_bytes-host_kv_before.registered_upload_bytes;
                stats.registered_kv_upload_lane_bytes[lane_index]+=
                    host_kv_after.registered_upload_bytes-host_kv_before.registered_upload_bytes;
                stats.registered_kv_upload_fallbacks+=host_kv_after.registered_upload_fallbacks-host_kv_before.registered_upload_fallbacks;
                stats.registered_kv_upload_failures+=host_kv_after.registered_upload_failures-host_kv_before.registered_upload_failures;
                stats.shared_device_page_copy_bytes+=host_kv_after.shared_page_copy_bytes-host_kv_before.shared_page_copy_bytes;
                stats.shared_device_page_attention_bytes+=host_kv_after.shared_page_attention_bytes-host_kv_before.shared_page_attention_bytes;
                stats.shared_device_page_attention_groups+=host_kv_after.shared_page_attention_groups-host_kv_before.shared_page_attention_groups;
                stats.query_pair_launch_attempts+=host_kv_after.query_pair_launch_attempts-host_kv_before.query_pair_launch_attempts;
                stats.gqa_six_query_pair_score_launch_attempts+=
                    host_kv_after.gqa_six_query_pair_score_launch_attempts-
                    host_kv_before.gqa_six_query_pair_score_launch_attempts;
                stats.gqa_six_query_pair_score_row_attempts+=
                    host_kv_after.gqa_six_query_pair_score_row_attempts-
                    host_kv_before.gqa_six_query_pair_score_row_attempts;
                stats.gqa_six_score_k_tile64_launch_attempts+=
                    host_kv_after.gqa_six_score_k_tile64_launch_attempts-
                    host_kv_before.gqa_six_score_k_tile64_launch_attempts;
                stats.gqa_six_score_k_tile64_row_attempts+=
                    host_kv_after.gqa_six_score_k_tile64_row_attempts-
                    host_kv_before.gqa_six_score_k_tile64_row_attempts;
                stats.gqa_six_softmax_triple_value_launch_attempts+=
                    host_kv_after.gqa_six_softmax_triple_value_launch_attempts-
                    host_kv_before.gqa_six_softmax_triple_value_launch_attempts;
                stats.gqa_six_softmax_triple_value_row_attempts+=
                    host_kv_after.gqa_six_softmax_triple_value_row_attempts-
                    host_kv_before.gqa_six_softmax_triple_value_row_attempts;
                stats.gqa_six_softmax_triple_pair_dimensions_launch_attempts+=
                    host_kv_after.gqa_six_softmax_triple_pair_dimensions_launch_attempts-
                    host_kv_before.gqa_six_softmax_triple_pair_dimensions_launch_attempts;
                stats.gqa_six_softmax_triple_pair_dimensions_row_attempts+=
                    host_kv_after.gqa_six_softmax_triple_pair_dimensions_row_attempts-
                    host_kv_before.gqa_six_softmax_triple_pair_dimensions_row_attempts;
                stats.gqa_six_softmax_six_values_single_load_launch_attempts+=
                    host_kv_after.gqa_six_softmax_six_values_single_load_launch_attempts-
                    host_kv_before.gqa_six_softmax_six_values_single_load_launch_attempts;
                stats.gqa_six_softmax_six_values_single_load_row_attempts+=
                    host_kv_after.gqa_six_softmax_six_values_single_load_row_attempts-
                    host_kv_before.gqa_six_softmax_six_values_single_load_row_attempts;
                stats.gqa_six_softmax_triple_key_pair_launch_attempts+=
                    host_kv_after.gqa_six_softmax_triple_key_pair_launch_attempts-
                    host_kv_before.gqa_six_softmax_triple_key_pair_launch_attempts;
                stats.gqa_six_softmax_triple_key_pair_row_attempts+=
                    host_kv_after.gqa_six_softmax_triple_key_pair_row_attempts-
                    host_kv_before.gqa_six_softmax_triple_key_pair_row_attempts;
                stats.host_kv_transaction_checkpoint_graph_captures+=
                    host_kv_after.transaction_checkpoint_graph_captures-
                    host_kv_before.transaction_checkpoint_graph_captures;
                stats.host_kv_transaction_checkpoint_graph_replays+=
                    host_kv_after.transaction_checkpoint_graph_replays-
                    host_kv_before.transaction_checkpoint_graph_replays;
                stats.host_kv_transaction_checkpoint_graph_capture_seconds+=
                    (host_kv_after.transaction_checkpoint_graph_capture_ms-
                     host_kv_before.transaction_checkpoint_graph_capture_ms)/1000.0;
                stats.host_kv_transaction_recurrent_trace_alias_layers+=
                    host_kv_after.transaction_recurrent_trace_alias_layers-
                    host_kv_before.transaction_recurrent_trace_alias_layers;
                stats.host_kv_transaction_recurrent_trace_copy_bytes_saved+=
                    host_kv_after.transaction_recurrent_trace_copy_bytes_saved-
                    host_kv_before.transaction_recurrent_trace_copy_bytes_saved;
                stats.recurrent_export_calls+=
                    recurrent_export_after.exports-recurrent_export_before.exports;
                stats.recurrent_export_bytes+=
                    recurrent_export_after.recurrent_bytes-
                    recurrent_export_before.recurrent_bytes;
                stats.recurrent_export_copy_submissions+=
                    recurrent_export_after.copy_submissions-
                    recurrent_export_before.copy_submissions;
                stats.recurrent_export_batched_copy_calls+=
                    recurrent_export_after.batched_copy_calls-
                    recurrent_export_before.batched_copy_calls;
                stats.recurrent_export_batched_copy_ranges+=
                    recurrent_export_after.batched_copy_ranges-
                    recurrent_export_before.batched_copy_ranges;
                stats.recurrent_export_seconds+=
                    (recurrent_export_after.recurrent_ms-
                     recurrent_export_before.recurrent_ms)/1000.0;
                stats.recurrent_export_fence_seconds+=
                    (recurrent_export_after.fence_ms-
                     recurrent_export_before.fence_ms)/1000.0;
                stats.reconstruction_submissions+=reconstruction_after.calls-reconstruction_before.calls;
                stats.paired_transform_submissions+=paired_after-paired_before;
                stats.k6_stream_reduction_calls+=k6_stream_after-k6_stream_before;
                stats.extended_stream_reduction_calls+=extended_stream_after-extended_stream_before;
                stats.k6_gateup_warpgroup_async_calls+=
                    k6_gateup_warpgroup_after-k6_gateup_warpgroup_before;
                stats.k6_gateup_n32_pair_cta_calls+=
                    k6_gateup_n32_pair_cta_calls_after-
                    k6_gateup_n32_pair_cta_calls_before;
                stats.k6_gateup_n32_pair_cta_rows+=
                    k6_gateup_n32_pair_cta_rows_after-
                    k6_gateup_n32_pair_cta_rows_before;
                stats.k6_fast_decode_calls+=
                    k6_fast_decode_calls_after-k6_fast_decode_calls_before;
                stats.k6_fast_decode_rows+=
                    k6_fast_decode_rows_after-k6_fast_decode_rows_before;
                stats.k6_rowpair_n64_calls+=
                    k6_rowpair_n64_calls_after-k6_rowpair_n64_calls_before;
                stats.k6_rowpair_n64_rows+=
                    k6_rowpair_n64_rows_after-k6_rowpair_n64_rows_before;
                stats.k6_down_rowpair_calls+=
                    k6_down_rowpair_calls_after-k6_down_rowpair_calls_before;
                stats.k6_down_rowpair_rows+=
                    k6_down_rowpair_rows_after-k6_down_rowpair_rows_before;
                stats.shape4_n64_calls+=
                    shape4_n64_calls_after-shape4_n64_calls_before;
                stats.shape4_n64_rows+=
                    shape4_n64_rows_after-shape4_n64_rows_before;
                stats.reduce_shfl_min_barrier_calls+=
                    reduce_shfl_min_barrier_calls_after-
                    reduce_shfl_min_barrier_calls_before;
                stats.reduce_shfl_min_barrier_rows+=
                    reduce_shfl_min_barrier_rows_after-
                    reduce_shfl_min_barrier_rows_before;
                stats.k7_tiles64_exact_splits_calls+=
                    k7_tiles64_exact_splits_after-k7_tiles64_exact_splits_before;
                stats.target_k8_kv_prefill_async_a_calls+=
                    target_k8_kv_prefill_async_a_calls_after-
                    target_k8_kv_prefill_async_a_calls_before;
                stats.target_k8_kv_prefill_async_a_rows+=
                    target_k8_kv_prefill_async_a_rows_after-
                    target_k8_kv_prefill_async_a_rows_before;
                stats.target_down_k6_async_a_calls+=
                    target_down_k6_async_a_after-target_down_k6_async_a_before;
                stats.target_gateup_k6_n16_calls+=
                    target_gateup_k6_n16_after-target_gateup_k6_n16_before;
                stats.host_kv_gdn_segment_graph_replays+=
                    host_kv_gdn_segment_graph_after.replays-
                    host_kv_gdn_segment_graph_before.replays;
                stats.host_kv_full_layer_graph_replays+=
                    host_kv_full_layer_graph_after.replays-
                    host_kv_full_layer_graph_before.replays;
                stats.host_kv_mlp_tail_graph_replays+=
                    host_kv_mlp_tail_graph_after.replays-
                    host_kv_mlp_tail_graph_before.replays;
                stats.target_k6_small_m_async_a_calls+=
                    target_k6_small_m_async_a_after-target_k6_small_m_async_a_before;
                stats.target_k7_small_m_async_a_calls+=
                    target_k7_small_m_async_a_after-target_k7_small_m_async_a_before;
                stats.target_k5_small_m_batch_calls+=
                    target_k5_small_m_batch_after-target_k5_small_m_batch_before;
                stats.eager_mlp_gateup_concurrent_calls+=
                    eager_mlp_gateup_concurrent_after-
                    eager_mlp_gateup_concurrent_before;
                stats.prefill_qkv_concurrent_calls+=
                    prefill_qkv_concurrent_calls_after-
                    prefill_qkv_concurrent_calls_before;
                stats.prefill_qkv_concurrent_rows+=
                    prefill_qkv_concurrent_rows_after-
                    prefill_qkv_concurrent_rows_before;
                stats.prefill_projection_graph_captures+=
                    prefill_projection_graph_captures_after-
                    prefill_projection_graph_captures_before;
                stats.prefill_projection_graph_replays+=
                    prefill_projection_graph_replays_after-
                    prefill_projection_graph_replays_before;
                stats.prefill_projection_graph_binding_fallbacks+=
                    prefill_projection_graph_binding_fallbacks_after-
                    prefill_projection_graph_binding_fallbacks_before;
                stats.prefill_projection_chain_graph_captures+=
                    prefill_projection_chain_graph_captures_after-
                    prefill_projection_chain_graph_captures_before;
                stats.prefill_projection_chain_graph_replays+=
                    prefill_projection_chain_graph_replays_after-
                    prefill_projection_chain_graph_replays_before;
                stats.prefill_projection_chain_graph_fallbacks+=
                    prefill_projection_chain_graph_fallbacks_after-
                    prefill_projection_chain_graph_fallbacks_before;
                stats.target_prefill_gate_up_pair_attempts+=
                    target_prefill_gate_up_pair_attempts_after-
                    target_prefill_gate_up_pair_attempts_before;
                stats.target_prefill_gate_up_pair_calls+=
                    target_prefill_gate_up_pair_calls_after-
                    target_prefill_gate_up_pair_calls_before;
                stats.target_prefill_gate_up_pair_rows+=
                    target_prefill_gate_up_pair_rows_after-
                    target_prefill_gate_up_pair_rows_before;
                stats.gqa_six_softmax_triple_v_tile_launch_attempts+=
                    gqa_six_v_tile_launch_after-gqa_six_v_tile_launch_before;
                stats.gqa_six_softmax_triple_v_tile_row_attempts+=
                    gqa_six_v_tile_row_after-gqa_six_v_tile_row_before;
                stats.gqa_six_softmax_triple_full_cta_launch_attempts+=
                    gqa_six_full_cta_launch_after-gqa_six_full_cta_launch_before;
                stats.gqa_six_softmax_triple_full_cta_row_attempts+=
                    gqa_six_full_cta_row_after-gqa_six_full_cta_row_before;
                stats.gqa_six_softmax_triple_threads128_launch_attempts+=
                    gqa_six_threads128_launch_after-gqa_six_threads128_launch_before;
                stats.gqa_six_softmax_triple_threads128_row_attempts+=
                    gqa_six_threads128_row_after-gqa_six_threads128_row_before;
                stats.gqa_six_softmax_triple_score_tile_launch_attempts+=
                    gqa_six_score_tile_launch_after-gqa_six_score_tile_launch_before;
                stats.gqa_six_softmax_triple_score_tile_row_attempts+=
                    gqa_six_score_tile_row_after-gqa_six_score_tile_row_before;
                stats.gqa_six_softmax_triple_scalar_dim_launch_attempts+=
                    gqa_six_scalar_dim_launch_after-gqa_six_scalar_dim_launch_before;
                stats.gqa_six_softmax_triple_scalar_dim_row_attempts+=
                    gqa_six_scalar_dim_row_after-gqa_six_scalar_dim_row_before;
                stats.gqa_six_softmax_triple_two_query_launch_attempts+=
                    gqa_six_two_query_launch_after-gqa_six_two_query_launch_before;
                stats.gqa_six_softmax_triple_two_query_row_attempts+=
                    gqa_six_two_query_row_after-gqa_six_two_query_row_before;
                stats.gqa_six_softmax_triple_fused_launch_attempts+=
                    gqa_six_fused_launch_after-gqa_six_fused_launch_before;
                stats.gqa_six_softmax_triple_fused_row_attempts+=
                    gqa_six_fused_row_after-gqa_six_fused_row_before;
                stats.gqa_six_softmax_tile512_launch_attempts+=
                    gqa_six_tile512_launch_after-gqa_six_tile512_launch_before;
                stats.gqa_six_softmax_tile512_row_attempts+=
                    gqa_six_tile512_row_after-gqa_six_tile512_row_before;
                stats.head_submitted_rows+=head_after.submitted_rows-head_before.submitted_rows;
                stats.head_omitted_rows+=head_after.omitted_rows-head_before.omitted_rows;
                const auto& policy_after=lanes[lane_index]->stats();
                stats.draft_neural_input_rows+=policy_after.neural_input_rows-policy_before.neural_input_rows;
                stats.draft_returned_suffix_rows+=policy_after.neural_returned_suffix_rows-policy_before.neural_returned_suffix_rows;
                stats.draft_discarded_suffix_rows+=policy_after.neural_discarded_suffix_rows-policy_before.neural_discarded_suffix_rows;
                stats.verifier_cost_menu_installations+=policy_after.cost_menu_updates-policy_before.cost_menu_updates;
                stats.conditional_second_block_calls+=policy_after.conditional_second_calls-policy_before.conditional_second_calls;
                stats.suffix_selection_installations+=policy_after.suffix_limit_installations-policy_before.suffix_limit_installations;
                stats.acquired_payload_preservations+=policy_after.acquired_payload_preservations-policy_before.acquired_payload_preservations;
                stats.acquired_full_resets+=policy_after.acquired_full_resets-policy_before.acquired_full_resets;
                stats.acquired_attachment_cancellations+=policy_after.acquired_attachment_cancellations-
                    policy_before.acquired_attachment_cancellations;
                stats.acquired_draft_ring_preservations+=policy_after.acquired_draft_ring_preservations-
                    policy_before.acquired_draft_ring_preservations;
                stats.acquired_draft_ring_restores+=policy_after.acquired_draft_ring_restores-
                    policy_before.acquired_draft_ring_restores;
                stats.execution_dependency_graph_begins+=policy_after.dependency_graph_begins-
                    policy_before.dependency_graph_begins;
                stats.execution_dependency_graph_submissions+=policy_after.dependency_graph_submissions-
                    policy_before.dependency_graph_submissions;
                stats.execution_dependency_graph_completions+=policy_after.dependency_graph_completions-
                    policy_before.dependency_graph_completions;
                stats.execution_dependency_graph_cancellations+=policy_after.dependency_graph_cancellations-
                    policy_before.dependency_graph_cancellations;
                stats.execution_dependency_graph_failures+=policy_after.dependency_graph_failures-
                    policy_before.dependency_graph_failures;
                stats.suffix_selection_neural_fallbacks+=policy_after.suffix_policy_neural_fallbacks-policy_before.suffix_policy_neural_fallbacks;
                stats.suffix_proposal_calls+=policy_after.suffix_calls-policy_before.suffix_calls;
                stats.suffix_proposal_rows+=policy_after.suffix_rows-policy_before.suffix_rows;
                stats.suffix_proposal_misses+=policy_after.suffix_misses-policy_before.suffix_misses;
                stats.suffix_published_rounds+=policy_after.suffix_published_rounds-
                    policy_before.suffix_published_rounds;
                stats.suffix_accepted_rows+=policy_after.suffix_accepted_rows-
                    policy_before.suffix_accepted_rows;
                stats.suffix_repair_rows+=policy_after.suffix_repair_rows-
                    policy_before.suffix_repair_rows;
                stats.suffix_skipped_neural_blocks+=
                    policy_after.suffix_published_skipped_neural_blocks-
                    policy_before.suffix_published_skipped_neural_blocks;
                for(std::size_t row=0;row<stats.supplied_verifier_horizon_decisions.size();++row)
                    stats.supplied_verifier_horizon_decisions[row]+=policy_after.supplied_horizon_decisions[row]-policy_before.supplied_horizon_decisions[row];
                stats.fused_gate_up_submissions+=fused_after-fused_before;
                stats.fused_residual_norm_submissions+=residual_after-residual_before;
                stats.reconstruction_submitted_rows+=reconstruction_after.rows-reconstruction_before.rows;
                stats.reconstruction_k6_submissions+=
                    reconstruction_after.k6_down_calls-reconstruction_before.k6_down_calls+
                    reconstruction_after.k6_gate_up_calls-reconstruction_before.k6_gate_up_calls;
                stats.reconstruction_k6_submitted_rows+=
                    reconstruction_after.k6_down_rows-reconstruction_before.k6_down_rows+
                    reconstruction_after.k6_gate_up_rows-reconstruction_before.k6_gate_up_rows;
                stats.attention_stage_upload_bytes+=host_kv_after.attention_stage_upload_bytes-host_kv_before.attention_stage_upload_bytes;
                stats.attention_stage_consumed_bytes+=host_kv_after.attention_stage_consumed_bytes-host_kv_before.attention_stage_consumed_bytes;
                stats.attention_stage_direct_bytes+=host_kv_after.attention_stage_direct_bytes-host_kv_before.attention_stage_direct_bytes;
                stats.attention_stage_banks+=host_kv_after.attention_stage_banks-host_kv_before.attention_stage_banks;
                stats.query_pair_row_attempts+=host_kv_after.query_pair_row_attempts-host_kv_before.query_pair_row_attempts;
                stats.query_pair_requested_row_attempts+=host_kv_after.query_pair_requested_row_attempts-host_kv_before.query_pair_requested_row_attempts;
                stats.shared_device_page_attention_pages+=host_kv_after.shared_page_attention_pages-host_kv_before.shared_page_attention_pages;
                stats.shared_device_page_attention_max_pages=std::max(stats.shared_device_page_attention_max_pages,
                    host_kv_after.shared_page_attention_max_pages);
                stats.shared_device_page_fallbacks+=host_kv_after.shared_page_fallbacks-host_kv_before.shared_page_fallbacks;
                stats.shared_device_page_failures+=host_kv_after.shared_page_failures-host_kv_before.shared_page_failures;
                stats.registered_kv_upload_peak_pending=std::max(stats.registered_kv_upload_peak_pending,
                    host_kv_after.registered_upload_peak_pending);
            }
            request->result.timings.total_seconds=seconds(request->submitted)+request->result.timings.prepare_seconds;
            {std::lock_guard lock(request->mutex);request->done=true;}request->changed.notify_all();
            {std::lock_guard lock(mutex);active[lane_index].reset();--outstanding;
                ready_work_authority.revise();--stats.running_requests;}
            changed.notify_all();
        }
    }
};
Exl3EngineCore::Request::~Request() {
    try {
        if(auto guard=cancellation_owner.lock()) {
            // No Engine queue lock is acquired here: a final queued reference
            // may itself be released while that lock is held.
            std::lock_guard lock(guard->output_mutex);
            if(guard->output_owner)guard->output_owner->retire_result(*this);
        }
    } catch(...) {} // Inventory conservatively retains failed retirement until close.
}
Exl3EngineCore::Request::DeliveryLease::DeliveryLease(
    std::shared_ptr<Request> value):request(std::move(value)) {
    if(!request)throw std::invalid_argument("output delivery lease request missing");
    auto current=request->outstanding_delivery_batches.load(std::memory_order_acquire);
    for(;;) {
        if(current>=output_delivery_outstanding_limit)
            throw std::logic_error("Engine output delivery slots exhausted");
        if(request->outstanding_delivery_batches.compare_exchange_weak(current,current+1,
                std::memory_order_acq_rel,std::memory_order_acquire))break;
    }
}
Exl3EngineCore::Request::DeliveryLease::~DeliveryLease() {
    if(!request)return;
    const auto previous=request->outstanding_delivery_batches.fetch_sub(1,
        std::memory_order_acq_rel);
    if(!previous)std::terminate();
    request->changed.notify_all();
}
void Exl3EngineCore::Request::cancel() noexcept {
    cancelled=true;
    if(sampling)sampling->cancel();
    try {host_preparation.cancel(host_preparation_generation);}catch(...) {}
    // Detached submissions retain no guard payload. A cancellation temporarily
    // pins it; destruction detaches its pointer under the same mutex.
    try {
        if(auto guard=cancellation_owner.lock()) {
            std::lock_guard lock(guard->mutex);
            if(guard->owner)guard->owner->cancel_queued(*this);
        }
    } catch(...) {} // Atomic cancellation remains visible to the worker.
}
void Exl3EngineCore::require_startup_retirement_available(const EngineOptions& options) {
    if(Exl3DevicePageStorage::quarantined_bytes() || Exl3DevicePageFill::quarantined_records())
        throw std::runtime_error("EXL3 unresolved device page retirement; reload refused");
    if(Exl3DevicePrefixCache::budget_snapshot()[1])
        throw std::runtime_error("EXL3 unresolved device prefix retirement; reload refused");
    if(Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire))
        throw std::runtime_error("EXL3 unresolved recurrent slab retirement; reload refused");
    if(Exl3LayerBufferRetirement::quarantined())
        throw std::runtime_error("EXL3 unresolved layer buffer retirement; reload refused");
    if(Exl3EngineTargetQ::packed_retirement_unresolved())
        throw std::runtime_error("EXL3 unresolved packed projection retirement; reload refused");
    if(Exl3EngineGreedyPacketBatch::retirement_unresolved())
        throw std::runtime_error("EXL3 unresolved greedy packet batch retirement; reload refused");
    if(Exl3CudaLinearWorkspace::quarantined_workspaces())
        throw std::runtime_error("EXL3 unresolved linear workspace retirement; reload refused");
    if(Exl3Dflash2DraftModel::generic_quarantined_allocations())
        throw std::runtime_error("EXL3 unresolved draft allocation retirement; reload refused");
    if(ExecutionStream::first_destroy_error.load())
        throw std::runtime_error("EXL3 unresolved execution stream retirement; reload refused");
    // Fast refusal before Impl/model setup. Each stream still atomically claims
    // its own slot, so concurrent construction cannot rely on this observation.
    if(options.max_concurrency==2 && ExecutionStream::occupied_count()!=0)
        throw std::runtime_error("execution stream retirement capacity exhausted");
    if(Exl3HostResidentSet::retirement_quarantined())
        throw std::runtime_error("EXL3 unresolved host residency retirement; reload refused");
    if(quarantined_engine.load())throw std::runtime_error("EXL3 unresolved Engine retirement; reload refused");
    if(Exl3Dflash2Execution::quarantined_execution_count())
        throw std::runtime_error("EXL3 unresolved DFlash execution retirement; reload refused");
    if(Exl3TextContext::reconstruction_quarantined_contexts() || Exl3TextContext::reconstruction_quarantined_allocations())
        throw std::runtime_error("EXL3 unresolved reconstruction retirement; reload refused");
    if(Exl3RegisteredKVBacking::quarantined_bytes() || Exl3KVTransferLease::quarantined_records() ||
        Exl3DevicePageStorage::quarantined_bytes() || Exl3DevicePageFill::quarantined_records() ||
        Exl3AttentionStageStorage::quarantined_records())
        throw std::runtime_error("EXL3 unresolved KV retirement; reload refused");
}
Exl3EngineCore::Exl3EngineCore(const EngineOptions& options,unsigned shared_allocation_fault_for_test,unsigned draft_clone_fault_for_test,unsigned first_draft_fault_for_test,unsigned context_startup_fault_for_test,
    Exl3DeviceAvailability::Provider device_availability_provider_for_test) {
    if(!options.prefill_chunk)
        throw std::invalid_argument("EXL3 Engine prefill_chunk must be nonzero");
    require_startup_retirement_available(options);
    impl_=std::make_unique<Impl>(options,shared_allocation_fault_for_test,draft_clone_fault_for_test,first_draft_fault_for_test,context_startup_fault_for_test,
        std::move(device_availability_provider_for_test));
}
Exl3RetirementResult Exl3EngineCore::close() noexcept {
    auto result=impl_->close();
    for(const auto& stream:impl_->streams)if(stream && stream->value)++result.pending_stream_destructions;
    return result;
}
std::weak_ptr<const void> Exl3EngineCore::cancellation_owner_for_test() const {
    return impl_->cancellation_owner;
}
std::size_t Exl3EngineCore::cancellation_owner_blocks_for_test() noexcept {
    return bounded_shared_live_blocks_for_test<Request::CancellationOwner>();
}
std::size_t Exl3EngineCore::result_owner_blocks_for_test() noexcept {
    return bounded_shared_live_blocks_for_test<Request::Storage>();
}
std::size_t Exl3EngineCore::execution_stream_owner_bytes_for_test() noexcept {
    return bounded_shared_allocation_bytes<ExecutionStream>();
}
std::size_t Exl3EngineCore::execution_stream_owner_blocks_for_test() noexcept {
    return bounded_shared_live_blocks_for_test<ExecutionStream>();
}
std::size_t Exl3EngineCore::shared_projection_owner_blocks_for_test() noexcept {
    return bounded_shared_live_blocks_for_test<Exl3EngineTargetQ>();
}
std::uint64_t Exl3EngineCore::shared_projection_metadata_bytes_for_test() const {
    return impl_->shared_q?impl_->shared_q->host_metadata_bytes():0;
}
std::weak_ptr<const void> Exl3EngineCore::shared_projection_owner_for_test() const noexcept {
    return impl_->shared_q;
}
std::optional<std::uint64_t> Exl3EngineCore::shared_projection_retirement_credit_for_test() const noexcept {
    auto bytes=bounded_retirement_credit_bytes_for_test<Exl3EngineTargetQ>(impl_->shared_q);
    if(bytes && impl_->shared_q)*bytes+=impl_->shared_q->child_metadata_credit_bytes_for_test();
    return bytes;
}
int Exl3EngineCore::execution_stream_retirement_error_for_test() noexcept {
    return ExecutionStream::first_destroy_error.load();
}
std::size_t Exl3EngineCore::execution_stream_retirement_slots_for_test() noexcept {
    return ExecutionStream::occupied_count();
}
std::size_t Exl3EngineCore::execution_stream_quarantined_handles_for_test() noexcept {
    std::size_t count=0;for(const auto& handle:ExecutionStream::unresolved)if(handle.load())++count;return count;
}
void Exl3EngineCore::fail_execution_stream_destroy_for_test(unsigned lane,int error) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || lane>=impl_->streams.size() ||
        !impl_->streams[lane] || !impl_->streams[lane]->value || error<=0 ||
        impl_->streams[lane]->destroy_fault_for_test)
        throw std::logic_error("stream destroy fault requires idle live nondefault stream");
    impl_->streams[lane]->destroy_fault_for_test=error;
}
std::uint64_t Exl3EngineCore::request_queue_metadata_bytes_for_test() const {
    using Queue=ReservedRequestQueue<Request>;
    Exl3ResourceInventory::Requirement required;
    required.add(Exl3ResourceInventory::Domain::host_metadata,1,bounded_shared_allocation_bytes<Queue::Storage>());
    const auto capacity=impl_->queue.capacity();
    if(capacity)required.add(Exl3ResourceInventory::Domain::host_metadata,capacity,sizeof(Queue::Entry));
    return required.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)];
}
std::uint64_t Exl3EngineCore::retained_host_metadata_for_test() const {
    const auto stats=impl_->coordinator.stats();
    if(!stats.retained_resource_units)throw std::logic_error("Engine metadata snapshot unavailable during allocation or unresolved retirement");
    return (*stats.retained_resource_units)[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)];
}
std::vector<Exl3ResourceInventory::Attribution>
Exl3EngineCore::resource_attribution_for_test() const {
    return impl_->coordinator.resource_attribution_for_test();
}
std::uint64_t Exl3EngineCore::tighten_metadata_headroom_for_test(std::uint64_t headroom) {
    std::unique_lock lock(impl_->mutex);
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0;});
    if(impl_->stopping || impl_->failed)
        throw std::logic_error("metadata ceiling requires idle live Engine");
    const auto result=impl_->coordinator.tighten_metadata_headroom_for_test(headroom);
    impl_->ready_work_authority.revise();return result;
}
std::size_t Exl3EngineCore::request_queue_capacity_for_test() const {
    return impl_->queue.capacity();
}
void Exl3EngineCore::fail_next_admission_budget_for_test() {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("admission budget fault requires idle live Engine");
    impl_->coordinator.fail_next_residency_for_test(
        Exl3VeriCacheServingCoordinator::ResidencyFault::page_budget_before_commit);
}
void Exl3EngineCore::fail_next_result_allocation_for_test(unsigned stage) {
    if(stage<1 || stage>9)throw std::invalid_argument("result allocation fault stage outside 1..9");
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || impl_->result_allocation_fault)
        throw std::logic_error("result allocation fault requires idle live unarmed Engine");
    impl_->result_allocation_fault=stage;
}
void Exl3EngineCore::fail_next_output_extent_for_test() {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || impl_->output_extent_fault)
        throw std::logic_error("output extent fault requires idle live unarmed Engine");
    impl_->output_extent_fault=true;
}
void Exl3EngineCore::fail_next_window_publication_for_test() {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed ||
       impl_->window_publication_fault.load(std::memory_order_acquire))
        throw std::logic_error(
            "window publication fault requires idle live unarmed Engine");
    impl_->window_publication_fault.store(true,std::memory_order_release);
}
void Exl3EngineCore::fail_next_shared_q_completion_for_test(bool before_producer_drain,std::function<void()> observer,
    bool after_first_scatter,unsigned family) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("shared Q fault requires idle live Engine");
    if(!impl_->shared_q)throw std::logic_error("shared Q fault requires enabled C2 owner");
    impl_->shared_q->fail_next_completion_for_test(before_producer_drain,std::move(observer),after_first_scatter,family);
}
std::function<void()> Exl3EngineCore::Submission::cancellation_callback_for_test() const {
    return [request=std::weak_ptr<Request>(request_)]() noexcept {
        if(auto owner=request.lock())owner->cancel();
    };
}
void Exl3EngineCore::invalidate_next_shared_offer_for_test() {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->shared_q)
        throw std::logic_error("invalid shared offer requires idle live C2 owner");
    impl_->shared_q->invalidate_next_offer_for_test();
}
void Exl3EngineCore::set_shared_rendezvous_timeout_for_test(std::chrono::microseconds timeout) {
    if(!impl_->shared_q)throw std::logic_error("shared rendezvous owner is disabled");
    impl_->shared_q->set_rendezvous_timeout_for_test(timeout);
}
void Exl3EngineCore::set_greedy_packet_batch_timeout_for_test(
    std::chrono::microseconds timeout) {
    if(!impl_->greedy_packet_batch)
        throw std::logic_error("greedy packet batch owner is disabled");
    impl_->greedy_packet_batch->set_timeout_for_test(timeout);
}
void Exl3EngineCore::observe_next_shared_wait_for_test(std::function<void()> observer) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->shared_q || !impl_->lanes[1])
        throw std::logic_error("shared wait observer requires idle live shared C2 Engine");
    impl_->shared_q->observe_next_wait_for_test(std::move(observer));
}
void Exl3EngineCore::expire_next_shared_pair_for_test() {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->shared_q)
        throw std::logic_error("expired shared pair requires idle live C2 owner");
    impl_->shared_q->expire_next_pair_for_test();
}
void Exl3EngineCore::claim_next_shared_pair_at_age_for_test(std::chrono::microseconds age) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->shared_q)
        throw std::logic_error("claim-age shared pair requires idle live C2 owner");
    impl_->shared_q->claim_next_pair_at_age_for_test(age);
}
std::uint64_t Exl3EngineCore::expired_shared_pair_refusals_for_test() {
    if(!impl_->shared_q)throw std::logic_error("expired pair observation requires enabled C2 owner");
    return impl_->shared_q->expired_pair_refusals_for_test();
}
std::array<std::weak_ptr<const void>,8> Exl3EngineCore::shared_claim_owners_for_test() {
    if(!impl_->shared_q)throw std::logic_error("shared owner observation requires enabled C2 owner");
    return impl_->shared_q->claimed_owners_for_test();
}
void Exl3EngineCore::fail_next_shared_page_for_test(unsigned stage,std::function<void()> before_failure) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->shared_pages || impl_->outstanding || impl_->failed || impl_->stopping || stage<1 || stage>8 ||
        impl_->shared_pages->constructor_fault_armed_for_test() ||
        (stage>=7 && !impl_->page_attention[0][0]))
        throw std::invalid_argument("shared page fault requires idle prepared Engine and stage1..8; attention stages require prepared reader");
    unsigned empty=0;
    if(!impl_->shared_page_fault.compare_exchange_strong(empty,stage))throw std::logic_error("shared page fault already armed");
    impl_->shared_page_before_failure=std::move(before_failure);
}
void Exl3EngineCore::observe_next_shared_page_fill_for_test(std::function<void()> observer) {
    std::lock_guard lock(impl_->mutex);
    if(!observer)throw std::invalid_argument("shared page fill observer is empty");
    if(!impl_->shared_pages || !impl_->kv_registrations || impl_->outstanding || impl_->failed || impl_->stopping ||
       impl_->shared_page_fill_observer_armed.load(std::memory_order_acquire) || impl_->shared_page_fault.load())
        throw std::logic_error("shared page fill observer requires idle registered shared-page Engine");
    impl_->shared_page_fill_observer=std::move(observer);
    impl_->shared_page_fill_observer_armed.store(true,std::memory_order_release);
}
void Exl3EngineCore::fail_next_page_constructor_for_test(unsigned stage,bool fail_cleanup) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->shared_pages || impl_->outstanding || impl_->failed || impl_->stopping || impl_->shared_page_fault.load())
        throw std::logic_error("page constructor fixture requires idle live shared-page Engine");
    impl_->shared_pages->fail_next_constructor_for_test(stage,fail_cleanup);
}
Exl3DevicePageWeakComponents Exl3EngineCore::shared_page_weak_components_for_test() {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->shared_pages || impl_->failed || impl_->stopping)
        throw std::logic_error("page component snapshot requires idle live shared-page Engine");
    // Submission completion precedes the worker's outstanding-count update.
    impl_->changed.wait(lock,[&]{return !impl_->outstanding || impl_->failed || impl_->stopping;});
    if(impl_->failed || impl_->stopping)throw std::logic_error("page component snapshot interrupted by Engine retirement");
    return impl_->shared_pages->weak_components_for_test();
}

void Exl3EngineCore::set_attention_staging_for_test(bool enabled) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->stats.attention_stage_device_bytes)throw std::logic_error("staging test requires prepared owners");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)throw std::logic_error("staging test on closed/failed Engine");
    for(const auto& lane:impl_->lanes)if(lane && lane->context().attention_staging_uncertain())
        throw std::logic_error("staging test on uncertain context");
    for(const auto& lane:impl_->lanes)if(lane)lane->context().set_attention_staging_enabled_for_test(enabled);
    impl_->ready_work_authority.revise();
}
void Exl3EngineCore::fail_attention_staging_for_test(unsigned stage) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->stats.attention_stage_device_bytes || !impl_->lanes[0] || impl_->lanes[1])
        throw std::logic_error("staging failure fixture requires prepared C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)throw std::logic_error("staging failure fixture on closed/failed Engine");
    impl_->lanes[0]->context().fail_attention_staging_for_test(stage);
}
void Exl3EngineCore::refuse_attention_read_pairs_for_test(bool enabled) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->kv_registrations || !impl_->stats.attention_stage_device_bytes)
        throw std::logic_error("attention pair refusal requires registered staging owners");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping || impl_->kv_registrations->external_readers())
        throw std::logic_error("attention pair refusal requires intact drained Engine");
    impl_->kv_registrations->refuse_read_pairs_for_test(enabled);
    impl_->ready_work_authority.revise();
}
void Exl3EngineCore::fail_reconstruction_retirement_for_test() {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1])
        throw std::logic_error("reconstruction retirement fixture requires C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)
        throw std::logic_error("reconstruction retirement fixture on closed/failed Engine");
    impl_->lanes[0]->context().fail_reconstruction_retirement_for_test();
}
void Exl3EngineCore::fail_gdn_buffer_retirement_for_test(unsigned owned_index) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1])throw std::logic_error("GDN buffer fixture requires C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)throw std::logic_error("GDN buffer fixture requires live drained Engine");
    if(!impl_->lanes[0]->context().fail_gdn_buffer_retirement_for_test(owned_index))
        throw std::invalid_argument("GDN buffer owned index out of range");
}
std::uint64_t Exl3EngineCore::fail_layer_buffer_retirement_for_test(bool gdn,unsigned layer,unsigned slot) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1])throw std::logic_error("physical buffer fixture requires C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)throw std::logic_error("physical buffer fixture requires live drained Engine");
    const auto bytes=impl_->lanes[0]->context().fail_layer_buffer_retirement_for_test(gdn,layer,slot);
    if(!bytes)throw std::invalid_argument("physical layer buffer is absent, borrowed or out of range");
    return bytes;
}
void Exl3EngineCore::fail_recurrent_constructor_for_test(unsigned fault) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1] || impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("recurrent constructor fixture requires idle live C1 Engine");
    impl_->lanes[0]->context().fail_recurrent_constructor_for_test(fault);
}
void Exl3EngineCore::fail_recurrent_commit_for_test(unsigned fault) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1] || impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("recurrent commit fixture requires idle live C1 Engine");
    if(fault<1 || fault>4)throw std::invalid_argument("recurrent commit fault selector");
    impl_->recurrent_commit_fault_hit.store(0);
    impl_->recurrent_commit_fault.store(fault);
}
void Exl3EngineCore::cancel_recurrent_growth_after_factory_for_test(unsigned phase) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1] || impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("recurrent growth cancellation fixture requires idle live C1 Engine");
    if(phase<1 || phase>2)throw std::invalid_argument("recurrent growth cancellation phase1..2");
    impl_->recurrent_cancel_after_factory_hit.store(0);
    impl_->recurrent_cancel_after_factory.store(phase);
}
void Exl3EngineCore::cancel_next_active_snapshot_for_test() {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1] || impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("snapshot cancellation fixture requires idle live C1 Engine");
    impl_->cancel_active_snapshot_for_test.store(true);
}
void Exl3EngineCore::cancel_conditional_stage_for_test(unsigned stage) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->conditional_b8 || !impl_->lanes[0] || impl_->lanes[1] ||
       impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("conditional cancellation requires idle live conditional C1 Engine");
    if(stage>3)throw std::invalid_argument("conditional cancellation stage0..3");
    impl_->conditional_cancel_hit.store(0);impl_->conditional_cancel_stage.store(stage);
}
unsigned Exl3EngineCore::conditional_cancel_hit_for_test() const {
    return impl_->conditional_cancel_hit.load();
}
void Exl3EngineCore::cancel_next_attachment_for_test(unsigned phase) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1] || impl_->outstanding ||
        impl_->failed || impl_->stopping)
        throw std::logic_error("attachment cancellation requires idle live C1 Engine");
    if(phase<1 || phase>3)throw std::invalid_argument("attachment cancellation phase1..3");
    impl_->attachment_cancel_hit.store(0,std::memory_order_release);
    impl_->attachment_cancel_phase.store(phase,std::memory_order_release);
}
unsigned Exl3EngineCore::attachment_cancel_hit_for_test() const {
    return impl_->attachment_cancel_hit.load(std::memory_order_acquire);
}
void Exl3EngineCore::cancel_next_attachment_during_restore_for_test(
    unsigned completed_layer) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("mid-restore cancellation requires idle live Engine");
    const auto* preserve=std::getenv("NINFER_EXL3_PRESERVE_ACQUIRED_ROOT");
    if(completed_layer>=64 || (preserve && std::string_view(preserve)=="1"))
        throw std::invalid_argument("mid-restore cancellation requires layer0..63 and full restore");
    impl_->attachment_restore_cancel_hit.store(64,std::memory_order_release);
    impl_->attachment_restore_cancel_layer.store(completed_layer+1,std::memory_order_release);
}
unsigned Exl3EngineCore::attachment_restore_cancel_hit_for_test() const {
    return impl_->attachment_restore_cancel_hit.load(std::memory_order_acquire);
}
void Exl3EngineCore::fail_request_metadata_for_test(unsigned phase) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1] || impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("request metadata fixture requires idle live C1 Engine");
    if(phase<1 || phase>3)throw std::invalid_argument("request metadata fault phase");
    impl_->request_metadata_fault_hit.store(0);impl_->request_metadata_fault.store(phase);
}
void Exl3EngineCore::fail_request_host_payload_for_test(unsigned phase) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1] || impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("host payload fixture requires idle live C1 Engine");
    if(phase<1 || phase>2)throw std::invalid_argument("host payload fault phase");
    const auto* direct=std::getenv("NINFER_EXL3_COMPACT_DIRECT_TAP_STAGING");
    if(direct && std::string_view(direct)=="1")throw std::logic_error("host payload fixture requires copied taps");
    impl_->request_host_payload_fault_hit.store(0);impl_->request_host_payload_fault.store(phase);
}
unsigned Exl3EngineCore::request_host_payload_fault_hit_for_test() const {
    std::unique_lock lock(impl_->mutex);
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0;});
    return impl_->request_host_payload_fault_hit.load();
}
void Exl3EngineCore::fail_next_draft_export_completion_for_test() {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->drafts[0] || impl_->drafts[1] || impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("draft export completion fixture requires idle live C1 Engine");
    impl_->drafts[0]->fail_next_host_export_completion_for_test();
}
std::weak_ptr<const void> Exl3EngineCore::uncertain_draft_source_owner_for_test() const {
    std::unique_lock lock(impl_->mutex);
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0;});
    if(!impl_->drafts[0] || impl_->drafts[1])
        throw std::logic_error("draft source observer requires retained C1 Engine");
    return impl_->drafts[0]->uncertain_source_owner_for_test();
}
unsigned Exl3EngineCore::request_metadata_fault_hit_for_test() const {
    std::unique_lock lock(impl_->mutex);
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0;});
    return impl_->request_metadata_fault_hit.load();
}
unsigned Exl3EngineCore::recurrent_commit_fault_hit_for_test() const {
    std::unique_lock lock(impl_->mutex);
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0;});
    return impl_->recurrent_commit_fault_hit.load();
}
unsigned Exl3EngineCore::recurrent_growth_cancel_hit_for_test() const {
    std::unique_lock lock(impl_->mutex);
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0;});
    return impl_->recurrent_cancel_after_factory_hit.load();
}
std::uint64_t Exl3EngineCore::recurrent_event_create_failures_for_test() const {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1])throw std::logic_error("recurrent event fixture requires C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0;});
    return impl_->lanes[0]->context().recurrent_export_stats().event_create_failures;
}
void Exl3EngineCore::fail_attention_buffer_retirement_for_test(unsigned owned_index) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1])throw std::logic_error("attention buffer fixture requires C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)throw std::logic_error("attention buffer fixture requires live drained Engine");
    if(!impl_->lanes[0]->context().fail_attention_buffer_retirement_for_test(owned_index))
        throw std::invalid_argument("attention buffer owned index out of range");
}
void Exl3EngineCore::fail_context_linear_retirement_for_test(unsigned slot,bool partial) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1])
        throw std::logic_error("context linear retirement fixture requires C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)
        throw std::logic_error("context linear retirement fixture requires live drained Engine");
    if(!impl_->lanes[0]->context().fail_linear_retirement_for_test(slot,partial))
        throw std::invalid_argument("context linear retirement slot has no owned device workspace");
}
void Exl3EngineCore::fail_continuation_allocation_retirement_for_test(unsigned slot,unsigned fault) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1])
        throw std::logic_error("continuation allocation retirement fixture requires C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)
        throw std::logic_error("continuation allocation retirement fixture requires live drained Engine");
    if(!impl_->lanes[0]->context().fail_continuation_allocation_retirement_for_test(slot,fault))
        throw std::invalid_argument("continuation allocation retirement slot/fault unavailable");
}
std::shared_ptr<const void> Exl3EngineCore::retain_host_kv_workspace_for_test() {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->lanes[0] || impl_->lanes[1])
        throw std::logic_error("shared KV borrower fixture requires C1 Engine");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)
        throw std::logic_error("shared KV borrower requires live drained Engine");
    auto owner=impl_->lanes[0]->context().host_kv_workspace_owner_for_test().lock();
    if(!owner)throw std::logic_error("shared KV borrower requires HostKV workspace");
    return owner;
}
void Exl3EngineCore::use_foreign_page_ancestor_for_test(bool enabled) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->failed || impl_->stopping || !impl_->shared_pages)
        throw std::logic_error("page ancestry fixture requires idle live Engine with shared pages");
    impl_->foreign_page_ancestor_for_test=enabled?std::make_shared<const int>(0):nullptr;
    impl_->ready_work_authority.revise();
}

void Exl3EngineCore::set_host_kv_routes_for_test(bool registered_upload,bool shared_pages) {
    std::unique_lock lock(impl_->mutex);
    if((registered_upload && !impl_->kv_registrations) || (shared_pages && !impl_->shared_pages))
        throw std::logic_error("HostKV route test requires prepared startup owners");
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)throw std::logic_error("HostKV route test on closed/failed Engine");
    impl_->registered_upload_enabled=registered_upload;impl_->shared_pages_enabled=shared_pages;
    impl_->ready_work_authority.revise();
}

void Exl3EngineCore::set_registered_upload_for_test(bool enabled) {
    std::unique_lock lock(impl_->mutex);
    if(!impl_->kv_registrations)throw std::logic_error("registered upload test requires prepared startup owners");
    // Tests serialize this boundary against submissions. Wait for worker cleanup
    // as well as result notification so one model owner serves both routes.
    impl_->changed.wait(lock,[&]{return impl_->outstanding==0 || impl_->failed;});
    if(impl_->failed || impl_->stopping)throw std::logic_error("registered upload test on closed/failed Engine");
    impl_->registered_upload_enabled=enabled;
    impl_->ready_work_authority.revise();
}
void Exl3EngineCore::hold_registration_reader_for_test(bool hold) {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->kv_registrations || !impl_->lanes[0] || impl_->lanes[1] ||
        impl_->outstanding || impl_->failed || impl_->stopping)
        throw std::logic_error("registration reader diagnostic requires idle live registered C1 Engine");
    if(hold) {
        if(impl_->registration_reader_for_test)throw std::logic_error("registration diagnostic reader already held");
        auto reader=Exl3KVRegistrationCache::acquire_external_read(impl_->kv_registrations);
        if(!reader)throw std::runtime_error("registration diagnostic reader unavailable");
        impl_->registration_reader_for_test.emplace(std::move(*reader));
    } else impl_->registration_reader_for_test.reset();
    impl_->ready_work_authority.revise();
}

void Exl3EngineCore::observe_request_start_for_test(std::function<void()> observer) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("request start observer requires idle live Engine");
    impl_->request_start_observer=std::move(observer);
}
void Exl3EngineCore::observe_lane_assignment_for_test(
    std::function<void(std::size_t,bool)> observer) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed ||
       impl_->options.max_concurrency!=2 || !impl_->request_affinity)
        throw std::logic_error("lane assignment observer requires idle affinity-enabled C2 Engine");
    impl_->lane_assignment_observer=std::move(observer);
}
void Exl3EngineCore::stale_lane_affinity_for_test(std::size_t lane) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed ||
       impl_->options.max_concurrency!=2 || !impl_->request_affinity || lane>=2 ||
       impl_->lane_frontiers[lane].root.expired())
        throw std::logic_error("stale lane affinity requires an idle retained C2 frontier");
    if(impl_->lane_frontiers[lane].request_generation==std::numeric_limits<std::uint64_t>::max())
        impl_->lane_frontiers[lane].request_generation=0;
    else ++impl_->lane_frontiers[lane].request_generation;
}
void Exl3EngineCore::observe_prefix_preparation_for_test(std::function<void(std::size_t,int)> observer) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->concurrent_prefix_preparation)
        throw std::logic_error("prefix preparation observer requires idle enabled live Engine");
    impl_->prefix_preparation_observer=std::move(observer);
}
void Exl3EngineCore::observe_host_preparation_for_test(
    std::function<void(std::size_t,std::uint64_t,bool)> observer) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed ||
       impl_->options.max_concurrency!=2)
        throw std::logic_error(
            "host preparation observer requires idle live C2 Engine");
    impl_->host_preparation_observer=std::move(observer);
}
void Exl3EngineCore::fail_next_prefix_preparation_admission_for_test() {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->concurrent_prefix_preparation)
        throw std::logic_error("prefix preparation fault requires idle enabled live Engine");
    impl_->cache.fail_next_preparation_admission_for_test();
}
void Exl3EngineCore::observe_terminal_roots_for_test(std::function<void(Root)> observer) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("terminal root observer requires idle live Engine");
    impl_->terminal_root_observer=std::move(observer);
}
void Exl3EngineCore::observe_acquired_roots_for_test(std::function<void(Root)> observer) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("acquired root observer requires idle live Engine");
    impl_->acquired_root_observer=std::move(observer);
}
void Exl3EngineCore::set_shared_cost_policy_for_test(const Exl3PackedCostPolicy& policy) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || !impl_->shared_q)
        throw std::logic_error("shared cost policy requires idle enabled Engine");
    impl_->shared_q->set_cost_policy(policy);
    impl_->ready_work_authority.revise();
}
void Exl3EngineCore::split_shared_contracts_for_test(bool enabled) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->shared_q || !impl_->lanes[1])
        throw std::logic_error("contract split requires idle live shared C2 Engine");
    impl_->shared_q->split_contracts_for_test(enabled);
}
void Exl3EngineCore::set_shared_preclaim_fault_for_test(unsigned fault) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->shared_q || !impl_->lanes[1])
        throw std::logic_error("preclaim fault requires idle live shared C2 Engine");
    impl_->shared_q->set_preclaim_fault_for_test(fault);
}
void Exl3EngineCore::set_shared_cost_policy(const Exl3PackedCostPolicy& policy) {
    if(policy.has_test_bypass())throw std::invalid_argument("shared cost policy refuses test bypass");
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed || !impl_->shared_q)
        throw std::logic_error("shared cost policy requires idle live enabled Engine");
    impl_->shared_q->set_cost_policy(policy);
    impl_->ready_work_authority.revise();
    ++impl_->stats.shared_cost_policy_updates;
}
void Exl3EngineCore::set_verifier_cost_menu(std::optional<Exl3VerifierHorizonPolicy::CostMenu> menu) {
    if(menu)menu->validate();
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("verifier cost menu requires idle live Engine");
    if(menu)for(const auto& lane:impl_->lanes)if(lane && !lane->verifier_horizon_enabled())
        throw std::invalid_argument("verifier cost menu requires bounded horizon opt-in");
    impl_->verifier_cost_menu=std::move(menu);
    impl_->ready_work_authority.revise();
}
std::vector<Exl3VeriCachePrefixIndex::RetentionMetadata> Exl3EngineCore::prefix_retention_metadata() {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("prefix retention metadata requires idle live Engine");
    return impl_->cache.retention_metadata();
}
Exl3VeriCachePrefixIndex::PolicyTrim Exl3EngineCore::trim_prefix_retention(
    std::uint64_t available_physical_bytes,
    std::span<const Exl3VeriCachePrefixIndex::RetentionDecision> decisions) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("prefix retention policy requires idle live Engine");
    auto result=impl_->coordinator.trim_prefix_retention(available_physical_bytes,decisions);
    impl_->ready_work_authority.revise();return result;
}
Exl3VeriCachePrefixIndex::Admission Exl3EngineCore::admit_prefix_input_with_retention(
    std::shared_ptr<const Exl3VeriCacheRequest> root,std::span<const std::int64_t> input,
    std::uint64_t available_physical_bytes,
    std::span<const Exl3VeriCachePrefixIndex::RetentionDecision> decisions) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("prefix retention admission requires idle live Engine");
    if(!impl_->options.context_cache.enabled)
        throw std::logic_error("prefix retention admission requires enabled context cache");
    if(!root || !root->state() || impl_->lanes.empty() || !impl_->lanes.front() ||
        root->state()->model_identity()!=impl_->lanes.front()->context().model_identity())
        throw std::invalid_argument("prefix retention admission requires current Engine model authority");
    auto result=impl_->coordinator.admit_prefix_input_with_retention(
        std::move(root),input,available_physical_bytes,decisions);
    impl_->ready_work_authority.revise();return result;
}
void Exl3EngineCore::set_suffix_selection_limits(std::optional<Exl3SuffixProposer::SelectionLimits> limits) {
    if(limits)limits->validate();
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("suffix selection limits require idle live Engine");
    if(limits)for(const auto& lane:impl_->lanes)if(lane && !lane->suffix_proposals_enabled())
        throw std::invalid_argument("suffix selection limits require suffix proposal opt-in");
    impl_->suffix_selection_limits=limits;
    impl_->ready_work_authority.revise();
}
void Exl3EngineCore::set_ready_copy_demand_provider(ReadyCopyDemandProvider provider) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("ready copy demand provider requires idle live Engine");
    impl_->copy_demand_provider=std::move(provider);
    impl_->ready_work_authority.revise();
}
void Exl3EngineCore::enable_ready_decision_trace_for_test(bool enabled) {
    std::lock_guard lock(impl_->mutex);
    if(impl_->outstanding || impl_->stopping || impl_->failed)
        throw std::logic_error("ready decision trace update requires idle live Engine");
    impl_->ready_decision_trace.reset();
    impl_->ready_decision_trace_enabled=enabled;
}
Exl3ReadyDecisionTrace::Snapshot Exl3EngineCore::ready_decision_trace_for_test() const {
    std::lock_guard lock(impl_->mutex);
    if(!impl_->ready_decision_trace_enabled)
        throw std::logic_error("ready decision trace is disabled");
    return impl_->ready_decision_trace.snapshot();
}
void Exl3EngineCore::fail_retirement_drain_for_test(int error) {
    if(error<=0 || impl_->retirement.result().phase!=Exl3RetirementPhase::active)
        throw std::invalid_argument("retirement fault requires active Engine and positive error");
    impl_->retirement_fault=error;
}
Exl3EngineCore::~Exl3EngineCore(){
    {std::lock_guard lock(impl_->cancellation_owner->mutex);impl_->cancellation_owner->owner=nullptr;}
    {std::lock_guard lock(impl_->cancellation_owner->output_mutex);impl_->cancellation_owner->output_owner=nullptr;}
    if(!close().reusable()) {
        // No allocation is needed on this failure edge. Registry exclusivity
        // prevents a second live bundle while this one remains retained.
        quarantined_engine.store(impl_.release());
    }
}
Family& Exl3EngineCore::frontend(){return impl_->frontend;}
ModelSamplingDefaults Exl3EngineCore::sampling_defaults()const{return impl_->defaults;}
LoadSummary Exl3EngineCore::load_summary()const{return impl_->load;}
MemorySummary Exl3EngineCore::memory_summary()const{std::lock_guard lock(impl_->mutex);return impl_->memory;}
RuntimeStats Exl3EngineCore::runtime_stats()const{
    RuntimeStats result;
    {
        std::lock_guard lock(impl_->mutex);result=impl_->stats;
        if(impl_->kv_registrations) {
            result.registered_kv_retired_registrations=impl_->kv_registrations->retired_registrations();
            result.registered_kv_external_readers=impl_->kv_registrations->external_readers();
            result.registered_kv_external_reader_high_water=impl_->kv_registrations->external_reader_high_water();
        }
        // Cache operations release their registry lock before publishing Engine
        // traffic counters. Keep close from removing the registry mid-snapshot.
        if(impl_->shared_pages) {
            const auto pages=impl_->shared_pages->accounting();
            result.shared_device_page_allocations=pages.unique_allocations;
            result.shared_device_page_resident_bytes=pages.device_bytes;
            result.shared_device_page_ready_rows=pages.ready_rows;
            result.shared_device_page_source_pages=pages.retained_source_pages;
            result.shared_device_page_source_host_bytes=pages.retained_source_host_bytes;
            result.shared_device_page_pending_fill_bytes=pages.pending_fill_bytes;
            result.shared_device_page_in_flight_fill_bytes=pages.in_flight_fill_bytes;
            result.shared_device_page_retained_readers=pages.retained_readers;
            result.shared_device_page_metadata_records=pages.retained_metadata_records;
            result.shared_device_page_failed_allocations=pages.failed_allocations;
            result.shared_device_page_retiring_allocations=pages.retiring_allocations;
        }
    }
    // Never acquire the rendezvous mutex beneath the Engine queue mutex.
    if(impl_->shared_q) {
        const auto shared=impl_->shared_q->stats();
        result.shared_target_projection_batches=shared.batches;
        result.shared_target_projection_rows=shared.rows;
        result.shared_target_projection_failures=shared.failures;
        result.shared_projection_completed_dispatches=shared.completed_dispatches;
        result.shared_projection_completed_rows=shared.completed_dispatch_rows;
        result.shared_projection_retains_claim_owners=shared.retains_claim_owners;
        result.shared_underfilled_cost_accepts=shared.underfilled_cost_accepts;
        result.shared_underfilled_cost_refusals=shared.underfilled_cost_refusals;
        result.shared_projection_contract_refusals=shared.contract_refusals;
        result.shared_projection_geometry_refusals=shared.geometry_refusals;
        result.shared_projection_stage_refusals=shared.stage_refusals;
        result.shared_projection_authority_refusals=shared.authority_refusals;
        result.shared_policy_capped_waits=shared.policy_capped_waits;
        result.shared_policy_capped_wait_budget_us=shared.policy_capped_wait_budget_us;
        result.shared_target_projection_fallbacks=shared.fallbacks;
        result.shared_target_projection_family_batches=shared.families;
        result.shared_target_projection_family_fallbacks=shared.family_fallbacks;
        result.conditional_shared_draft_batches=shared.conditional_draft_batches;
        result.conditional_shared_draft_lanes=shared.conditional_draft_lanes;
        result.conditional_shared_draft_families=shared.conditional_families;
        result.conditional_shared_draft_peers=shared.conditional_peers;
        result.shared_target_reused_gather_bytes=shared.reused_gather_bytes;
        result.shared_target_layer_batches=shared.layers;
    }
    if(impl_->greedy_packet_batch) {
        const auto packet=impl_->greedy_packet_batch->stats();
        result.greedy_packet_transport_batches=packet.batches;
        result.greedy_packet_transport_rows=packet.batched_rows;
        result.greedy_packet_transport_singles=packet.singles;
        result.greedy_packet_transport_cancellations=packet.cancelled;
        result.greedy_packet_transport_failures=packet.failures;
    }
    return result;
}
void Exl3EngineCore::reset_memory_peaks()noexcept{}
Exl3EngineCore::Submission Exl3EngineCore::submit(targets::qwen3_6::PreparedPrompt prompt,PromptSummary summary,double prepare,runtime::ResolvedRequestOptions options,OutputConsumerMode consumer,Clock::time_point deadline){
    const auto sampling_policy=Exl3SamplingPolicy::normalize(options.execution.sampling);
    const auto sampled_admission=Exl3SampledAuthorityBoundary::decide({
        sampling_policy,impl_->drafts[0]->sampled_distribution_availability()});
    if(!sampled_admission.greedy())
        throw RequestError(RequestErrorKind::UnsupportedSampling,
            std::string(Exl3SampledAuthorityBoundary::message(
                sampled_admission.disposition)));
    const auto prompt_tokens=targets::qwen3_6::PreparedPromptAccess::view(prompt).token_ids.size();
    Exl3NativeContextExtent::Position prompt_end=0;
    try {
        prompt_end=Exl3NativeContextExtent::checked_input_end(prompt_tokens,
            static_cast<Exl3NativeContextExtent::Position>(impl_->options.max_context));
    } catch(const std::invalid_argument&) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,"EXL3 prepared input exceeds Engine context");
    }
    const auto output_allowance=std::min<std::uint32_t>(options.execution.requested_output_tokens,
        impl_->options.max_context-static_cast<std::uint32_t>(prompt_end));
    const auto text_limit=impl_->frontend.output_text_byte_bound(output_allowance);
    unsigned result_allocation_fault=0;
    bool output_extent_fault=false;
    std::uint64_t ready_generation=0;
    ReadyCopyDemandProvider copy_demand_provider;
    const auto submitted=Clock::now();
    {
        std::lock_guard lock(impl_->mutex);
        if(impl_->stopping || impl_->failed)throw RequestError(RequestErrorKind::Unavailable,"EXL3 engine unavailable");
        if(impl_->next_host_preparation_generation==std::numeric_limits<std::uint64_t>::max())
            throw RequestError(RequestErrorKind::Unavailable,
                "EXL3 host preparation generation exhausted");
        ready_generation=impl_->next_host_preparation_generation++;
        result_allocation_fault=std::exchange(impl_->result_allocation_fault,0u);
        output_extent_fault=std::exchange(impl_->output_extent_fault,false);
        copy_demand_provider=impl_->copy_demand_provider;
    }
    Exl3ReadyCopyDemand copy_demand;
    if(copy_demand_provider) {
        copy_demand=copy_demand_provider(
            targets::qwen3_6::PreparedPromptAccess::view(prompt).token_ids);
        if(!copy_demand.valid())
            throw std::invalid_argument("EXL3 ready copy demand must be bounded and owned");
    }
    std::shared_ptr<GenerationResult> result_owner;
    std::shared_ptr<Request::Storage> result_storage;
    const auto text_capacity_ceiling=output_text_capacity_ceiling(text_limit);
    const auto option_bytes=request_stop_storage_bytes(options.stop);
    const auto prompt_bytes=targets::qwen3_6::PreparedPromptAccess::view(prompt).retained_storage_bytes();
    const auto session_ceiling=impl_->frontend.output_session_storage_ceiling(options.stop,text_limit);
    const auto control_tokens=impl_->frontend.thinking_control_tokens();
    std::size_t initial_session_bytes=0;
    const auto delivery_bytes=consumer==OutputConsumerMode::Streaming?
        output_delivery_outstanding_limit*(output_delivery_slot_capacity_ceiling()+
            bounded_shared_allocation_bytes<Request::DeliveryLease>()):0;
    const auto result_bytes=[option_bytes,prompt_bytes,prompt_tokens,
        control_count=control_tokens.size(),delivery_bytes](std::size_t capacity,std::size_t reasoning,std::size_t content,
        std::size_t session_bytes) {
        Exl3ResourceInventory::Requirement required;
        required.add(Exl3ResourceInventory::Domain::host_metadata,1,bounded_shared_allocation_bytes<Request::Storage>());
        if(option_bytes)required.add(Exl3ResourceInventory::Domain::host_metadata,1,option_bytes);
        required.add(Exl3ResourceInventory::Domain::host_metadata,1,prompt_bytes);
        if(prompt_tokens)required.add(Exl3ResourceInventory::Domain::host_metadata,prompt_tokens,sizeof(std::int64_t));
        if(control_count)required.add(Exl3ResourceInventory::Domain::host_metadata,control_count,sizeof(std::int64_t));
        if(session_bytes)required.add(Exl3ResourceInventory::Domain::host_metadata,1,session_bytes);
        using Token=decltype(GenerationResult::generated_token_ids)::value_type;
        if(capacity)required.add(Exl3ResourceInventory::Domain::host_metadata,capacity,sizeof(Token));
        // Include both terminators. Inline string storage is conservatively
        // charged again. Storage's bounded control arena is included above;
        // Request's embedded control arena is part of Storage too. General
        // heap bookkeeping remains excluded.
        if(reasoning)required.add(Exl3ResourceInventory::Domain::host_metadata,reasoning,1);
        if(content)required.add(Exl3ResourceInventory::Domain::host_metadata,content,1);
        if(delivery_bytes)required.add(Exl3ResourceInventory::Domain::host_metadata,1,delivery_bytes);
        required.add(Exl3ResourceInventory::Domain::host_metadata,2,1);
        return required.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)];
    };
    auto result_lease=[&] {
      try {return impl_->coordinator.allocate_logical_host(
        result_bytes(output_allowance,text_capacity_ceiling,text_capacity_ceiling,session_ceiling),[&] {
        result_storage=make_bounded_shared<Request::Storage>();
        result_storage->options=std::move(options);
        result_storage->prompt=std::move(prompt);
        if(request_stop_storage_bytes(result_storage->options.stop)!=option_bytes)
            throw std::logic_error("request option move changed reserved capacity");
        result_owner=std::shared_ptr<GenerationResult>(result_storage,&result_storage->result);
        if(result_allocation_fault==1)throw std::bad_alloc{};
        // Reserve final token storage before any worker owns numerical state.
        result_owner->generated_token_ids.reserve(output_allowance);
        if(result_allocation_fault==2)throw std::bad_alloc{};
        result_owner->reasoning.reserve(text_limit);
        if(result_allocation_fault==3)throw std::bad_alloc{};
        result_owner->content.reserve(text_limit);
        if(result_allocation_fault==4)throw std::bad_alloc{};
        if(!output_result_capacity_accepted(output_allowance,result_owner->generated_token_ids.capacity(),
            text_limit,text_capacity_ceiling,result_owner->reasoning.capacity(),result_owner->content.capacity()))
            throw RequestError(RequestErrorKind::Overloaded,"EXL3 result allocator exceeds capacity policy");
        const auto& input=targets::qwen3_6::PreparedPromptAccess::view(result_storage->prompt).token_ids;
        std::vector<std::int64_t> ready_input;
        ready_input.reserve(input.size());
        ready_input.assign(input.begin(),input.end());
        if(ready_input.capacity()!=prompt_tokens)
            throw RequestError(RequestErrorKind::Overloaded,"EXL3 input conversion exceeds reserved capacity");
        if(result_allocation_fault==8)throw std::bad_alloc{};
        const auto input_fingerprint=host_plan_fingerprint(ready_input);
        result_storage->control_ids.reserve(control_tokens.size());
        result_storage->control_ids.assign(control_tokens.begin(),control_tokens.end());
        if(result_storage->control_ids.capacity()!=control_tokens.size())
            throw RequestError(RequestErrorKind::Overloaded,"EXL3 control conversion exceeds reserved capacity");
        if(result_allocation_fault==9)throw std::bad_alloc{};
        result_storage->output=impl_->frontend.make_output_session(result_storage->prompt,
            result_storage->options.stop,result_storage->options.output,result_storage->options.execution.thinking);
        result_storage->output.preallocate_pending_storage();
        result_storage->output.preallocate_output_storage(text_limit);
        result_storage->output.validate_generation_capacity(output_allowance);
        initial_session_bytes=result_storage->output.retained_storage_bytes();
        if(initial_session_bytes>session_ceiling)
            throw RequestError(RequestErrorKind::Overloaded,"EXL3 output session exceeds initial capacity policy");
        if(result_allocation_fault==7)throw std::bad_alloc{};
        const auto logical_bytes=result_bytes(result_owner->generated_token_ids.capacity(),
            result_owner->reasoning.capacity(),result_owner->content.capacity(),initial_session_bytes);
        result_storage->ready_work.emplace(ready_generation,submitted,std::move(ready_input),
            input_fingerprint,Exl3ReadyWorkResources{
                .logical_host_bytes=logical_bytes,
                .request_local_device_bytes=0,
                .physical_lane_slots=1,
                .requires_preallocated_context=true},
            Exl3ReadyExecutionSignature{
                .model_owner=impl_->target,
                .device=impl_->options.device,
                .stage=Exl3ReadyNumericalStage::HostPreparation,
                .arithmetic=Exl3ReadyArithmetic::OrdinaryFp16B8GreedyText,
                .modality=Exl3ReadyModality::Text},std::move(copy_demand));
        return Exl3ResourceInventory::Allocation{result_owner,0,Exl3ResourceInventory::Domain::host_metadata,
            logical_bytes,{},
            +[](const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
                return attach_bounded_retirement_credit<Request::Storage>(owner,std::move(credit));
            }};
      },[&]{result_owner.reset();result_storage.reset();},Exl3HostResidentSet::ReservationExtent::at_most);}
      catch(const Exl3ResourceReservationExhausted&) {
        throw RequestError(RequestErrorKind::Overloaded,"EXL3 result reservation exceeds available host budget");
      } catch(const std::bad_alloc&) {
        throw RequestError(RequestErrorKind::Overloaded,"EXL3 result storage allocation failed");
      }
    }();
    std::shared_ptr<Request> r;
    try{
        if(result_allocation_fault==5)throw std::bad_alloc{};
        auto* request=std::construct_at(reinterpret_cast<Request*>(result_storage->request_bytes),result_owner,
            result_storage->options,result_storage->prompt,result_storage->output,
            *result_storage->ready_work,result_storage->control_ids,
            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(impl_.get())));
        r=std::shared_ptr<Request>(request,[storage=result_storage](Request* value) mutable noexcept {
            std::destroy_at(value);
            storage.reset();
        },EmbeddedControlAllocator<Request>(result_storage,result_storage->request_control,result_allocation_fault==6));
    }
    catch(const std::bad_alloc&) {
        impl_->coordinator.retire_logical_host(result_lease,[&]{result_owner.reset();result_storage.reset();return true;});
        throw RequestError(RequestErrorKind::Overloaded,"EXL3 result storage allocation failed");
    }
    catch(...){impl_->coordinator.retire_logical_host(result_lease,[&]{result_owner.reset();result_storage.reset();return true;});throw;}
    r->cancellation_owner=impl_->cancellation_owner;
    r->result_lease.emplace(std::move(result_lease));
    r->result.prompt=summary;r->result.timings.prepare_seconds=prepare;r->consumer=consumer;
    r->result_text_limit=text_limit;
    r->result_token_limit=output_allowance;
    r->initial_session_bytes=initial_session_bytes;
    r->output_extent_fault=output_extent_fault;
    r->deadline=deadline==Clock::time_point{}?r->submitted+std::chrono::milliseconds(impl_->options.pending_timeout_ms):deadline;
    {
        std::lock_guard lock(impl_->mutex);
        if(impl_->stopping || impl_->failed)throw RequestError(RequestErrorKind::Unavailable,"EXL3 engine unavailable");
        // Queue membership is protected by this mutex. These requests have
        // never acquired a worker or any physical readers; cancellation can
        // release their logical capacity without waiting for a busy lane.
        for(auto pending=impl_->queue.begin();pending!=impl_->queue.end();) {
            const auto& queued=*pending;
            if(!queued->cancelled.load()){++pending;continue;}
            {
                std::lock_guard request_lock(queued->mutex);
                queued->result.finish_reason=FinishReason::Cancelled;
                queued->result.timings.total_seconds=seconds(queued->submitted)+queued->result.timings.prepare_seconds;
                queued->done=true;
            }
            queued->changed.notify_all();
            pending=impl_->queue.erase(pending);
            impl_->ready_work_authority.revise();
            --impl_->outstanding;
        }
        impl_->stats.waiting_requests=static_cast<std::uint32_t>(impl_->queue.size());
        if(impl_->outstanding>=impl_->queue.capacity())
            throw RequestError(RequestErrorKind::Overloaded,"EXL3 request queue full");
        r->host_preparation_generation=ready_generation;
        impl_->queue.push_back(r);++impl_->outstanding;
        r->window_publication_fault.store(
            impl_->window_publication_fault.exchange(false,std::memory_order_acq_rel),
            std::memory_order_release);
        impl_->ready_work_authority.revise();
        impl_->stats.waiting_requests=static_cast<std::uint32_t>(impl_->queue.size());
    }
    if(impl_->request_affinity)impl_->changed.notify_all();
    else impl_->changed.notify_one();
    return Submission(std::move(r));
}
Exl3EngineCore::Submission::Submission(std::shared_ptr<Request> r):request_(std::move(r)){}
Exl3EngineCore::Submission::Submission(Submission&&)noexcept=default;
Exl3EngineCore::Submission& Exl3EngineCore::Submission::operator=(Submission&& other)noexcept{if(this!=&other){if(request_)request_->cancel();request_=std::move(other.request_);}return *this;}
Exl3EngineCore::Submission::~Submission(){if(request_)request_->cancel();}
std::weak_ptr<const void> Exl3EngineCore::Submission::result_owner_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return request_->result_owner;
}
std::uint64_t Exl3EngineCore::Submission::result_reserved_bytes_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return request_->result_lease->bytes();
}
std::uint64_t Exl3EngineCore::Submission::result_fixed_bytes_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return bounded_shared_allocation_bytes<Request::Storage>()+
        (request_->consumer==OutputConsumerMode::Streaming?
            output_delivery_outstanding_limit*(output_delivery_slot_capacity_ceiling()+
                bounded_shared_allocation_bytes<Request::DeliveryLease>()):0);
}
std::uint64_t Exl3EngineCore::Submission::option_storage_bytes_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return request_stop_storage_bytes(request_->options.stop)+request_->control_ids.size_bytes();
}
std::uint64_t Exl3EngineCore::Submission::prompt_storage_bytes_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return targets::qwen3_6::PreparedPromptAccess::view(request_->prompt).retained_storage_bytes()+
        request_->ready_work.input_tokens().size_bytes();
}
std::uint64_t Exl3EngineCore::Submission::initial_session_bytes_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return request_->initial_session_bytes;
}
std::uint64_t Exl3EngineCore::Submission::observed_session_bytes_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return request_->observed_session_bytes.load();
}
std::uint64_t Exl3EngineCore::Submission::outstanding_delivery_batches_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return request_->outstanding_delivery_batches.load(std::memory_order_acquire);
}
Exl3ReadyWorkObservation Exl3EngineCore::Submission::ready_work_for_test() const {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    return request_->ready_work.observation();
}
void Exl3EngineCore::Submission::set_delivery_byte_limit_for_test(std::size_t bytes) {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    if(bytes<4 || bytes>output_delivery_byte_limit)
        throw std::invalid_argument("delivery test byte limit outside 4..65536");
    std::lock_guard lock(request_->mutex);request_->delivery_byte_limit=bytes;
}
GenerationResult Exl3EngineCore::Submission::wait(OutputSink* sink,const CancellationView& cancellation){
    if(!request_)throw std::logic_error("empty EXL3 submission");
    if((request_->consumer==OutputConsumerMode::Streaming)!=(sink!=nullptr)){
        request_->cancel();throw std::invalid_argument("EXL3 consumer mode mismatch");
    }
    for(;;) {
        auto result=poll(sink,cancellation);
        if(result.state==GenerationPollState::Completed)return std::move(*result.result);
        if(result.state==GenerationPollState::Error)std::rethrow_exception(result.error);
        auto r=request_;
        std::unique_lock lock(r->mutex);
        r->changed.wait_for(lock,std::chrono::milliseconds(10),[&] {
            return (r->pending_deltas && r->outstanding_delivery_batches.load(
                    std::memory_order_acquire)<output_delivery_outstanding_limit) ||
                (r->done && !r->pending_deltas);
        });
    }
}
GenerationPollResult Exl3EngineCore::Submission::poll(
    OutputSink* sink,const CancellationView& cancellation) {
    if(!request_)throw std::logic_error("empty EXL3 submission");
    auto r=request_;
    if((r->consumer==OutputConsumerMode::Streaming)!=(sink!=nullptr))
        throw std::invalid_argument("EXL3 consumer mode mismatch");
    try {
        if(cancellation.requested())r->cancel();
    } catch(...) {
        r->cancel();request_.reset();
        return {GenerationPollState::Error,{},std::current_exception()};
    }
    try {
        std::array<OutputDelta,2> deltas{};std::size_t delivery_count=0;bool done;
        std::exception_ptr failure;
        {
            std::lock_guard lock(r->mutex);
            // Materialize owned sink messages while source strings are protected.
            // The queue itself retains ranges only; no view escapes this lock.
            std::size_t credit=r->delivery_byte_limit,consumed=0,partial=0;
            const auto available=output_delivery_outstanding_limit-
                std::min(output_delivery_outstanding_limit,
                    r->outstanding_delivery_batches.load(std::memory_order_acquire));
            for(const auto& pending:std::span(r->deltas).first(r->pending_deltas)) {
                if(delivery_count==deltas.size() || delivery_count==available)break;
                const auto& text=pending.channel==OutputChannel::Reasoning?r->result.reasoning:r->result.content;
                if(pending.offset>text.size() || pending.bytes>text.size()-pending.offset)
                    throw std::logic_error("pending output range exceeds committed text");
                const auto bytes=output_delivery_prefix(std::string_view(text).substr(pending.offset,pending.bytes),credit);
                if(!bytes && pending.bytes)break;
                auto delivered=text.substr(pending.offset,bytes);
                if(delivered.capacity()>output_delivery_slot_capacity_ceiling())
                    throw std::logic_error("Engine output delivery allocation exceeds reserved slot");
                auto lease=make_bounded_shared<Request::DeliveryLease>(r);
                deltas[delivery_count++]={pending.channel,std::move(delivered),std::move(lease)};
                credit-=bytes;
                if(bytes==pending.bytes)++consumed;
                else {partial=bytes;break;}
            }
            // Commit queue consumption only after every owned copy succeeds.
            for(std::size_t i=consumed;i<r->pending_deltas;++i)r->deltas[i-consumed]=r->deltas[i];
            r->pending_deltas-=consumed;
            if(partial){r->deltas.front().offset+=partial;r->deltas.front().bytes-=partial;}
            done=r->done && r->pending_deltas==0;
            if(done)failure=r->failure;
        }
        // User callbacks are outside request, Engine queue and coordinator locks.
        if(sink)for(auto& d:std::span(deltas).first(delivery_count))
            sink->publish(std::move(d));
        if(!done)return {};
        request_.reset();
        if(failure)return {GenerationPollState::Error,{},failure};
        GenerationPollResult result;result.state=GenerationPollState::Completed;
        result.result.emplace(std::move(r->result));return result;
    } catch(...) {
        r->cancel();request_.reset();
        return {GenerationPollState::Error,{},std::current_exception()};
    }
}
}
