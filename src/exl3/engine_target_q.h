#pragma once
#include "exl3/packed_projection_dispatch.h"
#include "exl3/target_q_continuation.h"
#include "exl3/packed_cost_policy.h"
#include "exl3/packed_gather_signature.h"
#include "exl3/reconstruction_device_retirement.h"
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/shared_conditioning_scope.h"
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <atomic>

namespace ninfer::exl3 {
static_assert(Exl3PackedCostPolicy::family_count==
    static_cast<unsigned>(Exl3TargetSharedFamily::count),
    "shared projection families require matching cost policy admission");
// One bounded rendezvous for the existing two Engine workers. It schedules no
// requests. The first suspended layer waits at most 50us for an unclaimed peer;
// once claimed, both lanes retain exclusive scratch until numerical completion.
// No Engine/coordinator lock may be held when entering this boundary.
class Exl3EngineTargetQ {
    using Coordinator=Exl3VeriCacheServingCoordinator;
    struct PackedStorage {
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        std::optional<RetainedDeviceLedger::Ticket> input_credit,output_credit;
        std::uint16_t *input=nullptr,*output=nullptr;
        std::size_t input_bytes=0,output_bytes=0;
        int device=-1,error=0,restore_error=0;
        unsigned failure_stage_for_test=0;
        PackedStorage* next=nullptr;
    };
    struct DeletePackedStorage {
        void operator()(PackedStorage* storage) const noexcept {
            if(!storage)return;
            auto credit=std::move(storage->metadata_credit);
            delete storage;
        }
    };
    inline static std::atomic<PackedStorage*> packed_quarantine_{nullptr};
    // Created before CUDA acquisition; failure retirement never allocates.
    std::unique_ptr<PackedStorage,DeletePackedStorage> packed_{new PackedStorage};
    struct Offer {
        // Published readiness facts are immutable. Mutable fields below describe
        // this rendezvous, not a changed numerical stage or resource identity.
        const Exl3ProjectionRows rows;
        const Coordinator::Lease lease;
        const std::shared_ptr<const Exl3VeriCacheRequest> projection_root;
        // Borrowed from the active Request whose strong owner remains published
        // in Engine::active until this synchronous offer returns.
        const std::span<const std::int64_t> control_contract;
        const Exl3CudaLinearWeights weights;
        const Exl3CudaLinearMetadata metadata;
        const cudaStream_t stream;
        const int layer;
        const Exl3TargetSharedFamily family;
        const void* const head_identity;
        const std::atomic<bool>* const cancelled;
        const Exl3ActivationLifetime::Witness activation;
        const std::array<const void*,4> head_backing;
        const std::chrono::microseconds wait_budget;
        bool claimed=false,done=false,shared=false;
        std::exception_ptr failure;
        const std::chrono::steady_clock::time_point offered=std::chrono::steady_clock::now();
    };
    std::unique_ptr<Exl3CudaLinearWorkspace> workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> kv_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> draft_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> draft_kv_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> draft_o_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> draft_down_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> draft_gateup_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> o_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> gateup_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> down_workspace_;
    std::unique_ptr<Exl3CudaLinearWorkspace> head_workspace_;
    int input_columns_=5120;
    int output_columns_=12288;
    std::mutex mutex_;
    std::condition_variable changed_;
    Offer* waiting_=nullptr;
    bool busy_=false,failed_=false,completion_fault_=false;
    bool producer_drain_fault_=false,partial_scatter_fault_=false;
    unsigned fault_family_for_test_=Exl3PackedCostPolicy::family_count;
    std::function<void()> completion_observer_for_test_;
    std::exception_ptr first_failure_;
    // On failure retain all roots/model/private destinations with the Engine
    // bundle. Logical cancellation never destroys this physical owner record.
    std::optional<Exl3PackedProjectionPlan> retained_;
    // Fixed storage protects the interval before plan assembly (including a
    // failed producer drain). Shared-pointer copies cannot allocate here.
    std::array<std::shared_ptr<const void>,8> claimed_owners_{};
    std::uint64_t shared_batches_=0,shared_rows_=0,failures_=0,fallbacks_=0;
    std::uint64_t completed_dispatches_=0,completed_dispatch_rows_=0;
    std::uint64_t contract_refusals_=0;
    std::uint64_t geometry_refusals_=0,authority_refusals_=0;
    std::uint64_t stage_refusals_=0;
    std::uint64_t underfilled_cost_accepts_=0,underfilled_cost_refusals_=0;
    std::uint64_t policy_capped_waits_=0,policy_capped_wait_budget_us_=0;
    std::array<std::uint64_t,15> family_batches_{};
    std::array<std::uint64_t,15> family_fallbacks_{};
    std::uint64_t conditional_draft_batches_=0,conditional_draft_lanes_=0;
    std::array<std::uint64_t,15> conditional_draft_families_{};
    std::array<std::uint64_t,2> conditional_peer_counts_{};
    std::array<std::uint64_t,64> layer_batches_{};
    Exl3PackedCostPolicy cost_policy_;
    Exl3PackedGatherSignature gather_signature_;
    bool gather_reuse_enabled_=false;
    std::uint64_t reused_gather_bytes_=0;
    std::chrono::microseconds rendezvous_timeout_{50};
    bool invalidate_next_offer_for_test_=false;
    bool split_contracts_for_test_=false;
    unsigned preclaim_fault_for_test_=0;
    std::function<void()> next_wait_observer_for_test_;
    bool expire_next_pair_for_test_=false;
    std::optional<std::chrono::microseconds> claim_age_for_test_;
    std::uint64_t expired_pair_refusals_for_test_=0;
    static void check(cudaError_t error) {
        if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
    }
    bool fallback(Exl3TargetSharedFamily family) noexcept {
        ++fallbacks_;
        const auto index=static_cast<std::size_t>(family);
        if(index<family_fallbacks_.size())++family_fallbacks_[index];
        return false;
    }
    static bool same_execution_contract(const Offer& a,const Offer& b) {
        return a.rows.contract==b.rows.contract && a.projection_root && b.projection_root &&
            a.projection_root->same_input_identity(*b.projection_root) &&
            std::equal(a.control_contract.begin(),a.control_contract.end(),
                b.control_contract.begin(),b.control_contract.end());
    }
    static bool compatible(const Offer& a,const Offer& b) {
        return a.layer==b.layer && a.family==b.family && same_execution_contract(a,b) &&
            same_projection_head_backing(a.head_identity,a.head_backing,b.head_identity,b.head_backing) &&
            Exl3ProjectionRows::same_owner(a.rows.model,b.rows.model) && a.rows.request!=b.rows.request &&
            a.rows.rows+b.rows.rows<=16 && a.weights.trellis==b.weights.trellis &&
            a.weights.suh==b.weights.suh && a.weights.svh==b.weights.svh && a.weights.mul1==b.weights.mul1 &&
            a.metadata.K==b.metadata.K && a.metadata.mul1==b.metadata.mul1 &&
            a.metadata.mcg==b.metadata.mcg && a.metadata.has_bias==b.metadata.has_bias &&
            a.metadata.in_features==b.metadata.in_features && a.metadata.out_features==b.metadata.out_features;
    }
public:
    static constexpr std::size_t packed_retirement_metadata_bytes() noexcept {return sizeof(PackedStorage);}
    static bool packed_retirement_unresolved() noexcept {
        return packed_quarantine_.load(std::memory_order_acquire)!=nullptr;
    }
    struct PackedRetirementSnapshot {
        int device=-1,error=0,restore_error=0;
        std::size_t input_bytes=0,output_bytes=0;
        std::uint64_t input_credit=0,output_credit=0,metadata_credit=0;
    };
    static PackedRetirementSnapshot packed_retirement_snapshot_for_test() noexcept {
        const auto* retained=packed_quarantine_.load(std::memory_order_acquire);
        return retained?PackedRetirementSnapshot{retained->device,retained->error,retained->restore_error,
            retained->input?retained->input_bytes:0,retained->output?retained->output_bytes:0,
            retained->input_credit?retained->input_credit->bytes():0,
            retained->output_credit?retained->output_credit->bytes():0,
            retained->metadata_credit?retained->metadata_credit->bytes():0}:
            PackedRetirementSnapshot{};
    }
    static Exl3ResourceInventory::Requirement requirement(bool kv=false,bool draft=false,bool output=false,bool gateup=false,bool down=false,bool head=false,bool draft_kv=false,bool draft_o=false,bool draft_down=false,bool draft_gateup=false) {
        const auto workspace=Exl3LinearWorkspaceRequirements::derive(5120,12288,16);
        Exl3ResourceInventory::Requirement result;result.configuration=0x51363136+(kv?1:0)+(draft?2:0)+(output?4:0)+(gateup?8:0)+(down?16:0)+(head?32:0)+(draft_kv?64:0)+(draft_o?128:0)+(draft_down?256:0)+(draft_gateup?512:0);
        result.add(Exl3ResourceInventory::Domain::device,1,workspace.owned_bytes);
        result.add(Exl3ResourceInventory::Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3EngineTargetQ>());
        result.add(Exl3ResourceInventory::Domain::host_metadata,1,sizeof(PackedStorage));
        const auto workspace_count=1u+unsigned(kv)+unsigned(draft)+unsigned(output)+unsigned(gateup)+
            unsigned(down)+unsigned(head)+unsigned(draft_kv)+unsigned(draft_o)+unsigned(draft_down)+unsigned(draft_gateup);
        result.add(Exl3ResourceInventory::Domain::host_metadata,workspace_count,Exl3CudaLinearWorkspace::metadata_bytes());
        result.add(Exl3ResourceInventory::Domain::device,16,2ULL*(((down || draft_down)?17408:output?6144:5120)+(head?248320:(gateup || draft_gateup)?17408:12288)));
        if(kv)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(5120,1024,16).owned_bytes);
        if(draft)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(5120,4096,16).owned_bytes);
        if(output)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(6144,5120,16).owned_bytes);
        if(gateup)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(5120,17408,16).owned_bytes);
        if(down)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(17408,5120,16).owned_bytes);
        if(head)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(5120,248320,16).owned_bytes);
        if(draft_kv)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(5120,1024,16).owned_bytes);
        if(draft_o)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(4096,5120,16).owned_bytes);
        if(draft_down)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(17408,5120,16).owned_bytes);
        if(draft_gateup)result.add(Exl3ResourceInventory::Domain::device,1,
            Exl3LinearWorkspaceRequirements::derive(5120,17408,16).owned_bytes);
        return result;
    }
    explicit Exl3EngineTargetQ(bool kv=false,bool draft=false,bool output=false,bool gateup=false,bool down=false,bool head=false,bool draft_kv=false,bool draft_o=false,bool draft_down=false,unsigned allocation_fault=0,bool retirement_fault_for_test=false,unsigned packed_retirement_fault_for_test=0,bool draft_gateup=false,Coordinator* constructor_authority=nullptr,unsigned constructor_family_for_test=0,bool constructor_cleanup_failure_for_test=true) {
        if(packed_retirement_fault_for_test>2)throw std::invalid_argument("packed retirement fault outside cleanup stages");
        const std::array enabled_families{true,true,gateup,draft,draft_kv,draft_o,output,down,draft_down,draft_gateup,head,kv};
        if(constructor_family_for_test>=enabled_families.size() || !enabled_families[constructor_family_for_test])
            throw std::invalid_argument("shared constructor fault family disabled or invalid");
        packed_->failure_stage_for_test=packed_retirement_fault_for_test;
        unsigned allocation_stage=0;
        const auto before_allocation=[&] {
            if(++allocation_stage==allocation_fault)
                throw std::runtime_error("injected shared projection startup allocation failure");
        };
        const auto make_linear=[&](unsigned family,int input,int output,bool small=false,bool gate=false,
            bool down_owner=false,bool output_owner=false,bool kv_owner=false,bool q_owner=false) {
            std::optional<RetainedDeviceLedger::Ticket> device;
            std::optional<RetainedDescriptorLedger::Ticket> metadata;
            if(constructor_authority) {
                const auto plan=Exl3LinearWorkspaceRequirements::derive(input,output,16);
                auto credits=constructor_authority->reserve_constructor_credits(plan.owned_bytes,Exl3CudaLinearWorkspace::metadata_bytes());
                device.emplace(std::move(credits.device));metadata.emplace(std::move(credits.metadata));
            }
            if(constructor_family_for_test==family)
                Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(constructor_cleanup_failure_for_test);
            return std::make_unique<Exl3CudaLinearWorkspace>(input,output,16,small,gate,down_owner,output_owner,
                false,false,false,Exl3CudaAccumulationView{},Exl3CudaTransformView{},false,kv_owner,q_owner,
                std::move(device),std::move(metadata));
        };
        try {
        before_allocation();
        workspace_=make_linear(1,5120,12288,false,false,false,false,false,true);
        // Operator batch width only. Each actual target transaction remains B8.
        workspace_->set_native_continuation16(true);
        if(retirement_fault_for_test)workspace_->fail_owned_retirement_for_test();
        if(gateup) {
            before_allocation();output_columns_=17408;
            gateup_workspace_=make_linear(2,5120,17408,false,true);
            gateup_workspace_->set_native_continuation16(true);
        }
        if(draft){before_allocation();draft_workspace_=make_linear(3,5120,4096,true);}
        if(draft_kv){before_allocation();draft_kv_workspace_=make_linear(4,5120,1024,true);}
        if(draft_o){before_allocation();draft_o_workspace_=make_linear(5,4096,5120,true);}
        if(output) {
            before_allocation();
            input_columns_=6144;
            o_workspace_=make_linear(6,6144,5120,false,false,false,true);
            o_workspace_->set_native_continuation16(true);
        }
        if(down) {
            before_allocation();input_columns_=17408;
            down_workspace_=make_linear(7,17408,5120,false,false,true);
            down_workspace_->set_native_continuation16(true);
        }
        if(draft_down) {
            before_allocation();input_columns_=17408;
            draft_down_workspace_=make_linear(8,17408,5120,true);
        }
        if(draft_gateup) {
            before_allocation();output_columns_=17408;
            draft_gateup_workspace_=make_linear(9,5120,17408,true);
        }
        if(head) {
            before_allocation();output_columns_=248320;
            head_workspace_=make_linear(10,5120,248320);
            head_workspace_->set_native_continuation16(true);
        }
        if(kv) {
            before_allocation();
            kv_workspace_=make_linear(11,5120,1024,false,false,false,false,true);
            kv_workspace_->set_native_continuation16(true);
        }
            before_allocation();
            check(cudaGetDevice(&packed_->device));
            packed_->input_bytes=16ULL*input_columns_*2;
            packed_->output_bytes=16ULL*output_columns_*2;
            if(constructor_authority) {
                auto credits=constructor_authority->reserve_constructor_credits(
                    packed_->input_bytes+packed_->output_bytes,sizeof(PackedStorage));
                auto input=credits.device.split(packed_->input_bytes);
                auto output=credits.device.split(packed_->output_bytes);
                if(!input || !output)throw std::logic_error("packed constructor credit split failed");
                packed_->input_credit.emplace(std::move(*input));
                packed_->output_credit.emplace(std::move(*output));
                packed_->metadata_credit.emplace(std::move(credits.metadata));
            }
            check(cudaMalloc(reinterpret_cast<void**>(&packed_->input),packed_->input_bytes));
            before_allocation();
            check(cudaMalloc(reinterpret_cast<void**>(&packed_->output),packed_->output_bytes));
            // Final construction boundary exercises rollback with both packed
            // buffers present, before startup inventory can publish this owner.
            before_allocation();
        } catch(...) {
            // A grant for an allocation that never happened must not survive
            // beside a failed cleanup of the other packed buffer.
            if(!packed_->input)packed_->input_credit.reset();
            if(!packed_->output)packed_->output_credit.reset();
            retire_packed();
            retire_workspaces();throw;
        }
    }
    // Engine joins producers and drains before releasing this inventoried owner.
    void release_constructor_credits_after_commit() noexcept {
        const std::array workspaces{workspace_.get(),kv_workspace_.get(),draft_workspace_.get(),draft_kv_workspace_.get(),
            draft_o_workspace_.get(),draft_down_workspace_.get(),draft_gateup_workspace_.get(),o_workspace_.get(),
            gateup_workspace_.get(),down_workspace_.get(),head_workspace_.get()};
        for(auto* workspace:workspaces)if(workspace)workspace->release_constructor_credits_after_commit();
        if(packed_) {
            packed_->input_credit.reset();packed_->output_credit.reset();packed_->metadata_credit.reset();
        }
    }
    ~Exl3EngineTargetQ(){
        retire_packed();
        retire_workspaces();
    }
private:
    void retire_packed() noexcept {
        if(!packed_ || (!packed_->input && !packed_->output))return;
        const auto result=exl3_retire_reconstruction_device(packed_->device,
            [](int* device) noexcept {return static_cast<int>(cudaGetDevice(device));},
            [](int device) noexcept {return static_cast<int>(cudaSetDevice(device));},
            [&]() noexcept {
                cudaError_t error=packed_->failure_stage_for_test==1?cudaErrorUnknown:cudaSuccess;
                if(error==cudaSuccess && packed_->output) {
                    error=cudaFree(packed_->output);
                    if(error==cudaSuccess){packed_->output=nullptr;packed_->output_credit.reset();}
                }
                if(error==cudaSuccess && packed_->failure_stage_for_test==2)error=cudaErrorUnknown;
                if(error==cudaSuccess && packed_->input) {
                    error=cudaFree(packed_->input);
                    if(error==cudaSuccess){packed_->input=nullptr;packed_->input_credit.reset();}
                }
                return static_cast<int>(error);
            },[]() noexcept {});
        if(result.released && !result.restore_error)return;
        packed_->error=result.error;packed_->restore_error=result.restore_error;
        auto* retained=packed_.release();
        auto* head=packed_quarantine_.load(std::memory_order_relaxed);
        do {retained->next=head;}while(!packed_quarantine_.compare_exchange_weak(
            head,retained,std::memory_order_release,std::memory_order_relaxed));
    }
    void retire_workspaces() noexcept {
        Exl3CudaLinearWorkspace::retire_owned(std::move(workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(kv_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(draft_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(draft_kv_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(draft_o_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(draft_down_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(draft_gateup_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(o_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(gateup_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(down_workspace_));
        Exl3CudaLinearWorkspace::retire_owned(std::move(head_workspace_));
    }
public:
    std::uint64_t host_metadata_bytes() const {
        const auto count=unsigned(bool(workspace_))+unsigned(bool(kv_workspace_))+unsigned(bool(draft_workspace_))+
            unsigned(bool(o_workspace_))+unsigned(bool(gateup_workspace_))+unsigned(bool(down_workspace_))+
            unsigned(bool(head_workspace_))+unsigned(bool(draft_kv_workspace_))+unsigned(bool(draft_o_workspace_))+
            unsigned(bool(draft_down_workspace_))+unsigned(bool(draft_gateup_workspace_));
        Exl3ResourceInventory::Requirement actual;
        actual.add(Exl3ResourceInventory::Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3EngineTargetQ>());
        if(packed_)actual.add(Exl3ResourceInventory::Domain::host_metadata,1,sizeof(PackedStorage));
        if(count)actual.add(Exl3ResourceInventory::Domain::host_metadata,count,Exl3CudaLinearWorkspace::metadata_bytes());
        return actual.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)];
    }
    static bool attach_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || !can_attach_bounded_retirement_credit<Exl3EngineTargetQ>(owner))return false;
        auto* shared=const_cast<Exl3EngineTargetQ*>(static_cast<const Exl3EngineTargetQ*>(owner.get()));
        if(shared->packed_ && shared->packed_->metadata_credit)return false;
        const std::array workspaces{shared->workspace_.get(),shared->kv_workspace_.get(),shared->draft_workspace_.get(),
            shared->draft_kv_workspace_.get(),shared->draft_o_workspace_.get(),shared->draft_down_workspace_.get(),
            shared->draft_gateup_workspace_.get(),shared->o_workspace_.get(),shared->gateup_workspace_.get(),
            shared->down_workspace_.get(),shared->head_workspace_.get()};
        std::uint64_t expected=bounded_shared_allocation_bytes<Exl3EngineTargetQ>()+
            (shared->packed_?sizeof(PackedStorage):0);
        for(const auto* workspace:workspaces)if(workspace) {
            if(!workspace->can_attach_metadata_credit())return false;
            expected+=Exl3CudaLinearWorkspace::metadata_bytes();
        }
        if(credit.bytes()!=expected)return false;
        if(shared->packed_) {
            auto packed_credit=credit.split(sizeof(PackedStorage));
            if(!packed_credit)return false;
            shared->packed_->metadata_credit.emplace(std::move(*packed_credit));
        }
        for(auto* workspace:workspaces)if(workspace) {
            auto child=credit.split(Exl3CudaLinearWorkspace::metadata_bytes());
            if(!child || !workspace->attach_metadata_credit(std::move(*child)))return false;
        }
        return attach_bounded_retirement_credit<Exl3EngineTargetQ>(owner,std::move(credit));
    }
    std::uint64_t child_metadata_credit_bytes_for_test() const noexcept {
        std::uint64_t bytes=packed_ && packed_->metadata_credit?packed_->metadata_credit->bytes():0;
        for(const auto* workspace:{workspace_.get(),kv_workspace_.get(),draft_workspace_.get(),
            draft_kv_workspace_.get(),draft_o_workspace_.get(),draft_down_workspace_.get(),
            draft_gateup_workspace_.get(),o_workspace_.get(),gateup_workspace_.get(),down_workspace_.get(),head_workspace_.get()})
            if(workspace)bytes+=workspace->metadata_credit_bytes_for_test();
        return bytes;
    }
    static bool attach_device_retirement_credit(const std::shared_ptr<const void>& owner,
        RetainedDeviceLedger::Ticket credit) noexcept {
        if(!owner)return false;
        auto* shared=const_cast<Exl3EngineTargetQ*>(static_cast<const Exl3EngineTargetQ*>(owner.get()));
        if(!shared->packed_ || !shared->packed_->input || !shared->packed_->output ||
            shared->packed_->input_credit || shared->packed_->output_credit || credit.bytes()!=shared->bytes())return false;
        const std::array workspaces{shared->workspace_.get(),shared->kv_workspace_.get(),shared->draft_workspace_.get(),
            shared->draft_kv_workspace_.get(),shared->draft_o_workspace_.get(),shared->draft_down_workspace_.get(),
            shared->draft_gateup_workspace_.get(),shared->o_workspace_.get(),shared->gateup_workspace_.get(),
            shared->down_workspace_.get(),shared->head_workspace_.get()};
        for(const auto* workspace:workspaces)if(workspace && !workspace->can_attach_device_credit())return false;
        for(auto* workspace:workspaces)if(workspace && workspace->workspace_bytes()) {
            auto part=credit.split(workspace->workspace_bytes());
            if(!part || !workspace->attach_device_credit(std::move(*part)))return false;
        }
        auto input=credit.split(shared->packed_->input_bytes);
        auto output=credit.split(shared->packed_->output_bytes);
        if(!input || !output)return false;
        shared->packed_->input_credit.emplace(std::move(*input));
        shared->packed_->output_credit.emplace(std::move(*output));return true;
    }
    std::size_t bytes() const noexcept {return workspace_->workspace_bytes()+
        (kv_workspace_?kv_workspace_->workspace_bytes():0)+
        (draft_workspace_?draft_workspace_->workspace_bytes():0)+
        (draft_kv_workspace_?draft_kv_workspace_->workspace_bytes():0)+
        (draft_o_workspace_?draft_o_workspace_->workspace_bytes():0)+
        (draft_down_workspace_?draft_down_workspace_->workspace_bytes():0)+
        (draft_gateup_workspace_?draft_gateup_workspace_->workspace_bytes():0)+
        (o_workspace_?o_workspace_->workspace_bytes():0)+
        (gateup_workspace_?gateup_workspace_->workspace_bytes():0)+
        (down_workspace_?down_workspace_->workspace_bytes():0)+
        (head_workspace_?head_workspace_->workspace_bytes():0)+16ULL*(input_columns_+output_columns_)*2;}
    std::uint64_t shared_batches() {std::lock_guard lock(mutex_);return shared_batches_;}
    void invalidate_next_offer_for_test() {
        std::lock_guard lock(mutex_);
        if(busy_ || waiting_ || failed_ || invalidate_next_offer_for_test_)
            throw std::logic_error("invalid offer seam requires idle unarmed owner");
        invalidate_next_offer_for_test_=true;
    }
    void split_contracts_for_test(bool enabled) {
        std::lock_guard lock(mutex_);
        if(busy_ || waiting_ || failed_)throw std::logic_error("contract split requires idle owner");
        gather_signature_.clear();split_contracts_for_test_=enabled;
    }
    void set_preclaim_fault_for_test(unsigned fault) {
        std::lock_guard lock(mutex_);
        if(busy_ || waiting_ || failed_)throw std::logic_error("preclaim fault requires idle owner");
        if(fault>3)throw std::invalid_argument("preclaim fault menu0..3");
        gather_signature_.clear();preclaim_fault_for_test_=fault;
    }
    std::uint64_t shared_rows() {std::lock_guard lock(mutex_);return shared_rows_;}
    void expire_next_pair_for_test() {
        std::lock_guard lock(mutex_);
        if(busy_ || waiting_ || failed_ || expire_next_pair_for_test_ || claim_age_for_test_)
            throw std::logic_error("expired pair seam requires idle unarmed owner");
        expire_next_pair_for_test_=true;
    }
    void claim_next_pair_at_age_for_test(std::chrono::microseconds age) {
        std::lock_guard lock(mutex_);
        if(busy_ || waiting_ || failed_ || expire_next_pair_for_test_ || claim_age_for_test_)
            throw std::logic_error("claim-age seam requires idle unarmed owner");
        if(age.count()<0 || age>std::chrono::microseconds(50))
            throw std::invalid_argument("claim-age seam outside 0us..50us");
        claim_age_for_test_=age;
    }
    std::uint64_t expired_pair_refusals_for_test() {
        std::lock_guard lock(mutex_);return expired_pair_refusals_for_test_;
    }
    std::uint64_t failures() {std::lock_guard lock(mutex_);return failures_;}
    std::uint64_t fallbacks() {std::lock_guard lock(mutex_);return fallbacks_;}
    struct Stats {std::uint64_t batches,rows,failures,fallbacks;std::array<std::uint64_t,15> families,family_fallbacks;std::array<std::uint64_t,64> layers;std::uint64_t reused_gather_bytes,completed_dispatches,completed_dispatch_rows;bool retains_claim_owners;std::uint64_t underfilled_cost_accepts,underfilled_cost_refusals,policy_capped_waits,policy_capped_wait_budget_us,contract_refusals,geometry_refusals,authority_refusals,conditional_draft_batches,conditional_draft_lanes;std::array<std::uint64_t,15> conditional_families;std::array<std::uint64_t,2> conditional_peers;std::uint64_t stage_refusals;};
    Stats stats() {std::lock_guard lock(mutex_);return {shared_batches_,shared_rows_,failures_,fallbacks_,family_batches_,family_fallbacks_,layer_batches_,reused_gather_bytes_,completed_dispatches_,completed_dispatch_rows_,bool(claimed_owners_[0]),underfilled_cost_accepts_,underfilled_cost_refusals_,policy_capped_waits_,policy_capped_wait_budget_us_,contract_refusals_,geometry_refusals_,authority_refusals_,conditional_draft_batches_,conditional_draft_lanes_,conditional_draft_families_,conditional_peer_counts_,stage_refusals_};}
    void enable_gather_reuse(bool enabled) {
        Exl3PackedGatherSignature retired_gather;
        {
            std::lock_guard lock(mutex_);
            if(busy_ || waiting_ || failed_)throw std::logic_error("gather policy requires idle owner");
            retired_gather=std::exchange(gather_signature_,{});
            gather_reuse_enabled_=enabled;
        }
    }
    std::array<std::weak_ptr<const void>,8> claimed_owners_for_test() {
        std::lock_guard lock(mutex_);
        if(busy_ || waiting_)throw std::logic_error("owner observation requires joined shared consumers");
        std::array<std::weak_ptr<const void>,8> result;
        for(std::size_t i=0;i<result.size();++i)result[i]=claimed_owners_[i];
        return result;
    }
    // Owning Engine has joined both workers and certified its device drain.
    void retire_after_device_drain() {
        std::function<void()> retired_observer;
        Exl3PackedGatherSignature retired_gather;
        std::optional<Exl3PackedProjectionPlan> retired_plan;
        std::array<std::shared_ptr<const void>,8> retired_owners{};
        {
            std::lock_guard lock(mutex_);
            if(busy_ || waiting_)throw std::logic_error("shared retirement requires joined producers");
            // Detach every externally owned reference before running its final
            // deleter. A deleter may observe the now-empty rendezvous again.
            retired_gather=std::exchange(gather_signature_,{});
            retired_plan=std::exchange(retained_,std::nullopt);
            retired_owners=std::exchange(claimed_owners_,{});
            retired_observer=std::exchange(completion_observer_for_test_,{});
            completion_fault_=false;producer_drain_fault_=false;partial_scatter_fault_=false;
            fault_family_for_test_=Exl3PackedCostPolicy::family_count;
        }
    }
    void fail_next_completion_for_test(bool before_producer_drain=false,std::function<void()> observer={},
        bool after_first_scatter=false,unsigned family=Exl3PackedCostPolicy::family_count) {
        std::lock_guard lock(mutex_);
        if(family>Exl3PackedCostPolicy::family_count)
            throw std::invalid_argument("shared Q fault family out of range");
        if(before_producer_drain && after_first_scatter)
            throw std::invalid_argument("shared Q fault requires one failure stage");
        if(busy_ || waiting_ || failed_ || completion_fault_ || producer_drain_fault_ || partial_scatter_fault_)
            throw std::logic_error("shared Q fault seam requires idle owner");
        producer_drain_fault_=before_producer_drain;
        partial_scatter_fault_=after_first_scatter;
        fault_family_for_test_=family;
        completion_fault_=!before_producer_drain && !after_first_scatter;
        completion_observer_for_test_=std::move(observer);
    }
    void set_cost_policy(Exl3PackedCostPolicy policy) {
        std::lock_guard lock(mutex_);
        if(busy_ || waiting_ || failed_)throw std::logic_error("packed cost update requires idle owner");
        cost_policy_=std::move(policy);
    }
    void notify_cancellation() {
        // Synchronize with predicate evaluation and wait registration: notifying
        // without this mutex can lose the wake between those two operations.
        // Never alter a claimed offer or release its physical owners here.
        std::lock_guard lock(mutex_);
        if(waiting_ && waiting_->cancelled && waiting_->cancelled->load())
            changed_.notify_all();
    }
    void observe_next_wait_for_test(std::function<void()> observer) {
        std::lock_guard lock(mutex_);
        if(!observer)throw std::invalid_argument("shared wait observer is empty");
        if(busy_ || waiting_ || failed_ || next_wait_observer_for_test_)
            throw std::logic_error("shared wait observer requires idle unarmed owner");
        next_wait_observer_for_test_=std::move(observer);
    }
    void set_rendezvous_timeout_for_test(std::chrono::microseconds timeout) {
        std::lock_guard lock(mutex_);
        if(busy_ || waiting_ || failed_)throw std::logic_error("rendezvous update requires idle owner");
        if(timeout<std::chrono::microseconds(50) || timeout>std::chrono::milliseconds(5))
            throw std::invalid_argument("test rendezvous timeout outside 50us..5ms");
        rendezvous_timeout_=timeout;
    }

    bool execute(const Exl3TargetQContinuation& q,const Coordinator::Lease& lease,
        std::uint64_t epoch,std::shared_ptr<const void> lane_owner,
        std::shared_ptr<const void> model_owner,Coordinator& coordinator,
        std::span<const std::int64_t> control_contract={},
        const std::atomic<bool>* cancelled=nullptr,
        std::shared_ptr<const Exl3VeriCacheRequest> conditioning_root={}) {
        std::unique_lock lock(mutex_);
        if(failed_)std::rethrow_exception(first_failure_);
        // A control publication or recycled acquisition replaces the bound root.
        // Reject stale callbacks before offering them to a peer or touching any
        // producer stream. Dispatch still repeats the coherent two-lease check.
        const std::array<Coordinator::Lease,1> offered_lease{lease};
        if(!coordinator.compute_leases_current(offered_lease))
            throw std::invalid_argument("shared projection offer has stale request authority");
        const auto* capability=target_shared_capability(q.family,q.metadata);
        const auto admission=capability?std::optional{capability->admission}:std::nullopt;
        const bool draft_kv=q.family==Exl3TargetSharedFamily::draft_k || q.family==Exl3TargetSharedFamily::draft_v;
        const bool draft_o=q.family==Exl3TargetSharedFamily::draft_o;
        const bool draft_down=q.family==Exl3TargetSharedFamily::draft_down;
        const bool draft_gateup=q.family==Exl3TargetSharedFamily::draft_gate || q.family==Exl3TargetSharedFamily::draft_up;
        const bool draft=capability && capability->scope==Exl3TargetSharedScope::draft_layer;
        if(conditioning_root && !shared_conditioning_scope_valid(conditioning_root,lease.root,draft,q.position))
            throw std::invalid_argument("shared draft conditioning root scope mismatch");
        const auto& projection_root=conditioning_root?conditioning_root:lease.root;
        const bool output=q.family==Exl3TargetSharedFamily::o;
        const bool gateup=q.family==Exl3TargetSharedFamily::gate || q.family==Exl3TargetSharedFamily::up;
        const bool down=q.family==Exl3TargetSharedFamily::down;
        const bool head=q.family==Exl3TargetSharedFamily::head;
        auto* workspace=draft_gateup?draft_gateup_workspace_.get():draft_down?draft_down_workspace_.get():draft_o?draft_o_workspace_.get():draft_kv?draft_kv_workspace_.get():head?head_workspace_.get():down?down_workspace_.get():gateup?gateup_workspace_.get():output?o_workspace_.get():draft?draft_workspace_.get():q.family==Exl3TargetSharedFamily::q?workspace_.get():kv_workspace_.get();
        if(busy_ || (cancelled && cancelled->load()) || q.rows<1 || q.rows>8 ||
           !capability || !capability->supports_layer(q.layer) || !admission || !workspace ||
           (draft && (q.rows!=8 || !q.supported_draft_head())) ||
           !(head?workspace->target_head_small_m_candidate(q.metadata,2,*admission):down?workspace->target_down_small_m_candidate(q.metadata,2,*admission):gateup?(workspace->target_gateup_small_m_candidate(q.metadata,2,*admission) || workspace->target_gateup_k5_small_m_candidate(q.metadata,2,*admission)):output?(workspace->target_o_k6_small_m_candidate(q.metadata,2,*admission) ||
                     workspace->target_o_k7_small_m_candidate(q.metadata,2,*admission)):
             draft?workspace->draft_shared_m16_candidate(q.metadata,16,*admission):q.family==Exl3TargetSharedFamily::q
             ?workspace->target_q_k6_small_m_candidate(q.metadata,2,*admission)
             :workspace->target_kv_small_m_candidate(q.metadata,2,*admission))) return fallback(q.family);
        const auto offered_epoch=std::exchange(invalidate_next_offer_for_test_,false)?0:epoch;
        const auto policy_timeout=std::chrono::microseconds(cost_policy_.wait_budget_us(
            static_cast<unsigned>(q.family),q.metadata.in_features,q.metadata.out_features,q.metadata.K,q.rows));
        const auto timeout=policy_timeout<rendezvous_timeout_?policy_timeout:rendezvous_timeout_;
        Offer offer{{lease.ticket.request_id,lease.acquisition,offered_epoch,std::move(model_owner),projection_root,
            lane_owner,std::move(lane_owner),split_contracts_for_test_ && (lease.ticket.request_id&1)
                ?"Engine/diagnostic-split/fp16/shared-target/M16":projection_root->prepared_metadata_owner()
                ?"Engine/prepared-media/fp16/shared-target/M16":"Engine/text/fp16/shared-target/M16",
            q.position,q.rows,q.metadata.in_features,q.metadata.out_features,
            static_cast<std::size_t>(q.metadata.in_features),static_cast<std::size_t>(q.metadata.out_features),q.input,q.output,
            static_cast<std::size_t>(q.rows)*q.metadata.in_features,static_cast<std::size_t>(q.rows)*q.metadata.out_features},
            lease,projection_root,control_contract,q.weights,q.metadata,q.stream,q.layer,q.family,q.head_identity,cancelled,q.activation,q.head_backing,timeout};
        if(!offer.rows.valid_admission(8))
            throw std::invalid_argument("shared projection offer has invalid ownership or geometry");
        if(!waiting_) {
            if(!cost_policy_.may_wait(static_cast<unsigned>(q.family),q.metadata.in_features,
                q.metadata.out_features,q.metadata.K,q.rows)) return fallback(q.family);
            if(policy_timeout<rendezvous_timeout_) {
                ++policy_capped_waits_;
                policy_capped_wait_budget_us_+=static_cast<std::uint64_t>(timeout.count());
            }
            waiting_=&offer;
            // This one-shot source fixture runs after the offer becomes the
            // unclaimed waiter and while its mutex still excludes cancellation
            // notification. Once it returns, wait_until releases that mutex;
            // Request::cancel can then publish the wake without a lost edge.
            if(auto observer=std::exchange(next_wait_observer_for_test_,{})) {
                try {observer();}
                catch(...) {waiting_=nullptr;throw;}
            }
            // The claim-age seam is a correctness fixture, not a production
            // budget. Give the peer worker bounded host scheduling time while
            // the real gate below still evaluates the injected policy age.
            const auto scheduling_wait=claim_age_for_test_?std::chrono::seconds(1):offer.wait_budget;
            changed_.wait_until(lock,offer.offered+scheduling_wait,[&]{
                return offer.claimed || (offer.cancelled && offer.cancelled->load());
            });
            if(!offer.claimed) {waiting_=nullptr;claim_age_for_test_.reset();return fallback(q.family);}
            changed_.wait(lock,[&]{return offer.done;});
            if(offer.failure)std::rethrow_exception(offer.failure);
            return offer.shared;
        }
        // This observation is the membership boundary. Cancellation after claim
        // may suppress publication, but must not revoke either physical owner.
        if((waiting_->cancelled && waiting_->cancelled->load()) || (cancelled && cancelled->load())) {
            return fallback(q.family);
        }
        if(!same_execution_contract(*waiting_,offer)) {
            ++contract_refusals_;return fallback(q.family);
        }
        // Opposite target/draft stages must fall back before claiming either
        // destination. The diagnostic substitutes only the candidate family;
        // the immutable offer and real callback retain their original stage.
        const auto candidate_family=preclaim_fault_for_test_==3
            ? (offer.family==Exl3TargetSharedFamily::draft_q
                ? Exl3TargetSharedFamily::q:Exl3TargetSharedFamily::draft_q)
            : offer.family;
        if(waiting_->family!=candidate_family || waiting_->layer!=offer.layer) {
            ++stage_refusals_;return fallback(q.family);
        }
        if(!compatible(*waiting_,offer) || !capability->supports_combined_rows(waiting_->rows.rows+q.rows) ||
           !exl3_packed_target_candidate(*workspace,q.metadata,waiting_->rows.rows+q.rows,*admission)) {
            return fallback(q.family);
        }
        const bool geometry_valid=[&] {
            if(preclaim_fault_for_test_!=1)return Exl3ProjectionRows::valid_pair(waiting_->rows,offer.rows,16);
            auto diagnostic=offer.rows;diagnostic.input=waiting_->rows.input;
            return Exl3ProjectionRows::valid_pair(waiting_->rows,diagnostic,16);
        }();
        if(!geometry_valid) {
            ++geometry_refusals_;return fallback(q.family);
        }
        // Recheck both members together at claim, not only at initial offer.
        // A waiting peer can lose authority while this host lane is arriving.
        // Refuse before consuming fault seams, claiming destinations or draining
        // streams; dispatch still repeats this same snapshot after the drains.
        std::array<Coordinator::Lease,2> leases{waiting_->lease,offer.lease};
        if(preclaim_fault_for_test_==2)leases[0].acquisition=0;
        if(!coordinator.compute_leases_current(leases)) {++authority_refusals_;return fallback(q.family);}
        // Only the explicit diagnostic seam substitutes time. Normal callers
        // always use the current clock, and both paths use the same claim gate.
        const bool forced_expiry=std::exchange(expire_next_pair_for_test_,false);
        // Full 8+8 pairs already have their standing production permission;
        // preserve the one-shot seam until an underfilled decision is reached.
        const bool underfilled_pair=waiting_->rows.rows+q.rows<16;
        const auto forced_claim_age=underfilled_pair?
            std::exchange(claim_age_for_test_,std::nullopt):std::optional<std::chrono::microseconds>{};
        const auto claim_time=forced_expiry?waiting_->offered+waiting_->wait_budget:
            forced_claim_age?waiting_->offered+*forced_claim_age:std::chrono::steady_clock::now();
        const auto age=std::chrono::duration_cast<std::chrono::microseconds>(claim_time-waiting_->offered).count();
        const bool cost_permitted=Exl3PackedCostPolicy::claim_before_deadline(
            waiting_->offered,claim_time,waiting_->wait_budget) && age>=0 &&
            cost_policy_.permits(static_cast<unsigned>(q.family),q.metadata.in_features,
            q.metadata.out_features,q.metadata.K,waiting_->rows.rows,q.rows,static_cast<std::uint64_t>(age));
        // Count decisions only for compatible underfilled peers at this actual
        // membership boundary. Bypass/full pairs and successful GPU work differ.
        if(underfilled_pair && !cost_policy_.has_test_bypass()) {
            if(cost_permitted)++underfilled_cost_accepts_;else ++underfilled_cost_refusals_;
        }
        if(!cost_permitted) {
            if(forced_expiry)++expired_pair_refusals_for_test_;
            return fallback(q.family);
        }
        auto* peer=waiting_;waiting_=nullptr;busy_=true;peer->claimed=true;
        claimed_owners_={peer->rows.model,peer->rows.root,peer->rows.input_owner,peer->rows.output_owner,
            offer.rows.model,offer.rows.root,offer.rows.input_owner,offer.rows.output_owner};
        // Unrelated target/draft pairs cannot consume a family-specific fault.
        // Selection occurs only after a real pair has acquired both destinations.
        const bool fault_family_matches=fault_family_for_test_==Exl3PackedCostPolicy::family_count ||
            fault_family_for_test_==static_cast<unsigned>(offer.family);
        const bool completion_fault=fault_family_matches?std::exchange(completion_fault_,false):false;
        const bool producer_drain_fault=fault_family_matches?std::exchange(producer_drain_fault_,false):false;
        const bool partial_scatter_fault=fault_family_matches?std::exchange(partial_scatter_fault_,false):false;
        auto completion_observer=fault_family_matches?std::exchange(completion_observer_for_test_,{}):std::function<void()>{};
        changed_.notify_all();lock.unlock();
        std::exception_ptr failure;
        std::uint64_t reused_bytes=0;
        bool dispatch_completed=false;
        try {
            if(producer_drain_fault)throw std::runtime_error("injected shared Q producer drain failure");
            // Source producers run on independent streams. Both complete before
            // gather uses the second lane's stream; neither host lane can resume.
            check(cudaStreamSynchronize(peer->stream));check(cudaStreamSynchronize(offer.stream));
            const auto live=[&](const Exl3ProjectionRows&) {return coordinator.compute_leases_current(leases);};
            std::array<Exl3ProjectionRows,2> rows{peer->rows,offer.rows};
            const auto signature=[](const Offer& member) {
                const auto& s=member.rows;
                return Exl3PackedGatherSignature::Lane{s.request,s.acquisition,s.execution,s.model,s.root,
                    s.input_owner,s.input,member.layer,s.position,s.rows,s.input_columns,s.input_stride,member.activation,s.contract};
            };
            const Exl3PackedGatherSignature::Pair keys{signature(*peer),signature(offer)};
            const int gather_order=gather_reuse_enabled_ &&
                (offer.family==Exl3TargetSharedFamily::up || offer.family==Exl3TargetSharedFamily::draft_up)
                ? gather_signature_.consume_order(keys,packed_->input,16ULL*input_columns_) : 0;
            if(gather_order<0)std::swap(rows[0],rows[1]);
            const bool reuse=gather_order!=0;
            gather_signature_.clear(); // every other physical dispatch overwrites the cache
            retained_.emplace(Exl3PackedProjectionPlan::assemble(rows,16,live));
            dispatch_exl3_packed_target_projection(*retained_,*workspace,offer.weights,offer.metadata,
                packed_->input,16ULL*input_columns_,packed_->output,16ULL*output_columns_,live,offer.stream,*admission,reuse,partial_scatter_fault);
            dispatch_completed=true;
            if(completion_observer)completion_observer();
            if(completion_fault)throw std::runtime_error("injected shared Q completion failure");
            if(reuse)reused_bytes=static_cast<std::uint64_t>(retained_->rows())*offer.metadata.in_features*2;
            if(gather_reuse_enabled_ &&
                (offer.family==Exl3TargetSharedFamily::gate || offer.family==Exl3TargetSharedFamily::draft_gate))
                gather_signature_.remember(keys,packed_->input,16ULL*input_columns_);
            // Dispatch synchronizes scatter before any destination is resumed.
            retained_.reset();
        } catch(...) {failure=std::current_exception();}
        completion_observer={}; // Captured-owner destruction must not hold mutex_.
        lock.lock();
        if(dispatch_completed){++completed_dispatches_;completed_dispatch_rows_+=peer->rows.rows+offer.rows.rows;}
        if(failure){first_failure_=failure;failed_=true;++failures_;}
        else {claimed_owners_={};++shared_batches_;shared_rows_+=peer->rows.rows+offer.rows.rows;reused_gather_bytes_+=reused_bytes;
            ++family_batches_[static_cast<unsigned>(offer.family)];
            if(draft) {
                const unsigned conditional_lanes=
                    unsigned(!Exl3ProjectionRows::same_owner(peer->rows.root,peer->lease.root))+
                    unsigned(!Exl3ProjectionRows::same_owner(offer.rows.root,offer.lease.root));
                if(conditional_lanes){
                    ++conditional_draft_batches_;conditional_draft_lanes_+=conditional_lanes;
                    ++conditional_draft_families_[static_cast<unsigned>(offer.family)];
                    ++conditional_peer_counts_[conditional_lanes-1];
                }
            }
            if(!draft && !head)++layer_batches_[offer.layer];}
        busy_=false;peer->failure=failure;peer->shared=!failure;peer->done=true;
        changed_.notify_all();
        if(failure)std::rethrow_exception(failure);
        return true;
    }
};
}
