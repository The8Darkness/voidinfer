#pragma once
#include "exl3/gaming_optimizations.h"
#include "exl3/target_q_continuation.h"
#include "exl3/draft_distribution_contract.h"

#include "exl3/linear_cuda.h"
#include "exl3/resource_inventory.h"
#include "exl3/bounded_shared_owner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>
#include <span>
#include <functional>
#include <span>
#include <mutex>
#include <limits>
#include <atomic>

namespace ninfer::exl3 {

struct Exl3DeviceGreedySeed;

class Exl3DraftHostRing {
    struct ConstructionKey {};
    struct Page {
        long long first=0;
        int rows=0;
        std::array<std::vector<std::uint16_t>,5> k,v;
    };
    std::shared_ptr<const int> model_identity_;
    long long base_=0;
    int count_=0;
    std::size_t exported_bytes_=0;
    std::vector<std::shared_ptr<const Page>> pages_;
    // Installed only when a submitted transfer cannot prove completion.
    // Self-retention requires no failure-path allocation and preserves credits.
    std::shared_ptr<Exl3DraftHostRing> uncertain_transfer_owner_;
    inline static std::atomic<std::uint64_t> uncertain_transfer_count_{0};
    static void retain_uncertain_transfer(const std::shared_ptr<Exl3DraftHostRing>& owner) noexcept {
        if(!owner->uncertain_transfer_owner_) {
            owner->uncertain_transfer_owner_=owner;
            uncertain_transfer_count_.fetch_add(1,std::memory_order_release);
        }
    }
    Exl3DraftHostRing()=default;
    static std::shared_ptr<Exl3DraftHostRing> create() {
        return make_bounded_shared<Exl3DraftHostRing>(ConstructionKey{});
    }
    static std::shared_ptr<Page> create_page(
        const std::function<RetainedDescriptorLedger::Ticket(std::uint64_t)>& reserve={}) {
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {
            credit.emplace(reserve(page_metadata_bytes()));
            if(credit->bytes()!=page_metadata_bytes())throw std::invalid_argument("draft page metadata extent");
        }
        auto page=make_bounded_shared<Page>();
        if(credit && !attach_page_metadata_credit(page,std::move(*credit)))
            throw std::logic_error("draft page metadata attachment");
        return page;
    }
    friend class Exl3Dflash2DraftModel;
public:
    static std::uint64_t uncertain_transfer_count() noexcept {
        return uncertain_transfer_count_.load(std::memory_order_acquire);
    }
    static void require_transfer_healthy() {
        if(uncertain_transfer_count())throw std::runtime_error("unresolved draft transfer forbids execution");
    }
    static void exercise_uncertain_metadata_for_test() {
        RetainedDescriptorLedger ledger;
        const MetadataReservation reserve=[&](std::uint64_t bytes){return ledger.acquire(bytes);};
        auto ring=create_planned(1,reserve);ring->pages_.push_back(create_page(reserve));
        const auto bytes=ledger.bytes();const auto before=uncertain_transfer_count();
        std::weak_ptr<Exl3DraftHostRing> weak=ring;
        retain_uncertain_transfer(ring);retain_uncertain_transfer(ring);ring.reset();
        if(weak.expired() || ledger.bytes()!=bytes || uncertain_transfer_count()!=before+1)
            throw std::runtime_error("uncertain draft transfer lost destination/credit or duplicated retention");
        bool refused=false;try{require_transfer_healthy();}catch(const std::runtime_error&){refused=true;}
        if(!refused)throw std::runtime_error("uncertain draft transfer did not close execution gate");
    }
    using MetadataReservation=std::function<RetainedDescriptorLedger::Ticket(std::uint64_t)>;
    static std::size_t planned_page_capacity(long long base,int count) {
        if(base<0 || count<1 || base>std::numeric_limits<long long>::max()-count)
            throw std::invalid_argument("draft ring metadata geometry");
        return static_cast<std::size_t>((static_cast<std::uint64_t>(base%64)+count+63)/64);
    }
    static std::shared_ptr<Exl3DraftHostRing> create_planned(std::size_t capacity,const MetadataReservation& reserve={}) {
        Exl3ResourceInventory::Requirement required;using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3DraftHostRing>());
        if(capacity)required.add(Domain::host_metadata,capacity,sizeof(std::shared_ptr<const Page>));
        const auto bytes=required.units[static_cast<unsigned>(Domain::host_metadata)];
        std::optional<RetainedDescriptorLedger::Ticket> credit;
        if(reserve) {credit.emplace(reserve(bytes));if(credit->bytes()!=bytes)throw std::invalid_argument("draft ring metadata extent");}
        auto result=create();result->pages_.reserve(capacity);
        if(credit && credit->bytes()!=result->metadata_bytes())throw std::logic_error("draft ring allocation extent");
        if(credit && !attach_metadata_credit(result,std::move(*credit)))throw std::logic_error("draft ring metadata attachment");
        return result;
    }
    explicit Exl3DraftHostRing(ConstructionKey):Exl3DraftHostRing() {}
    std::uint64_t metadata_bytes() const {
        Exl3ResourceInventory::Requirement required;using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3DraftHostRing>());
        if(pages_.capacity())required.add(Domain::host_metadata,pages_.capacity(),sizeof(pages_[0]));
        return required.units[static_cast<unsigned>(Domain::host_metadata)];
    }
    static bool metadata_credit_belongs_to(const std::shared_ptr<const void>& owner,
        const RetainedDescriptorLedger& ledger) noexcept {
        return bounded_split_credit_belongs_to<Exl3DraftHostRing>(owner,ledger);
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!owner || !owner.use_count())return false;
        const auto* ring=static_cast<const Exl3DraftHostRing*>(owner.get());
        return attach_bounded_split_retirement_credit<Exl3DraftHostRing>(owner,std::move(credit),
            ring->pages_.capacity()*sizeof(ring->pages_[0]));
    }
    long long base() const noexcept {return base_;}
    static constexpr std::size_t page_metadata_bytes() noexcept {return bounded_shared_allocation_bytes<Page>();}
    template<class Visitor> void visit_page_metadata_owners(Visitor&& visitor) const {
        for(const auto& page:pages_)visitor(std::shared_ptr<const void>(page));
    }
    static bool page_metadata_credit_belongs_to(const std::shared_ptr<const void>& owner,
        const RetainedDescriptorLedger& ledger) noexcept {
        return bounded_split_credit_belongs_to<Page>(owner,ledger);
    }
    static bool attach_page_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        return attach_bounded_split_retirement_credit<Page>(owner,std::move(credit),0);
    }
    static void exercise_metadata_for_test() {
        const auto need=[](bool value,const char* message){if(!value)throw std::runtime_error(message);};
        RetainedDescriptorLedger ledger;auto ring=create();ring->pages_.reserve(7);
        need(planned_page_capacity(0,2048)==32 && planned_page_capacity(63,2048)==33 &&
            planned_page_capacity(64,1)==1,"draft ring descriptor planning boundary");
        {
            auto planned=create_planned(33,[&](std::uint64_t bytes){return ledger.acquire(bytes);});
            need(ledger.bytes()==planned->metadata_bytes(),"planned draft ring metadata missing");
        }
        need(ledger.bytes()==0,"planned draft ring metadata leaked");
        {
            auto retained=create_page();retained->rows=1;retained->k[0].resize(1024,41);
            const auto live=bounded_shared_live_blocks_for_test<Page>();
            unsigned calls=0;bool refused=false;
            const MetadataReservation reserve=[&](std::uint64_t bytes){
                if(++calls==3)throw Exl3ResourceReservationExhausted(__FILE__,__LINE__);
                return ledger.acquire(bytes);
            };
            try {
                auto partial=create_planned(3,reserve);
                partial->pages_.push_back(retained);
                partial->pages_.push_back(create_page(reserve));
                partial->pages_.push_back(create_page(reserve));
            }catch(const Exl3ResourceReservationExhausted&){refused=true;}
            need(refused && calls==3 && ledger.bytes()==0 && bounded_shared_live_blocks_for_test<Page>()==live &&
                retained->rows==1 && retained->k[0][0]==41,"draft partial metadata refusal damaged parent or leaked allocation");
        }
        const auto bytes=ring->metadata_bytes();
        need(!attach_metadata_credit(ring,ledger.acquire(bytes-1)) && ledger.bytes()==0,"draft ring short credit accepted");
        need(attach_metadata_credit(ring,ledger.acquire(bytes)),"draft ring exact credit refused");
        need(!attach_metadata_credit(ring,ledger.acquire(bytes)) && ledger.bytes()==bytes,"draft ring duplicate charge retained");
        std::weak_ptr<Exl3DraftHostRing> weak=ring;ring.reset();
        need(weak.expired() && ledger.bytes()==bounded_shared_allocation_bytes<Exl3DraftHostRing>(),
            "draft ring final strong retirement retained descriptors or dropped block");
        weak.reset();need(ledger.bytes()==0,"draft ring final weak credit leaked");
        auto page=make_bounded_shared<Page>();page->k[0].resize(1024,9);
        const auto page_bytes=page_metadata_bytes();
        need(attach_page_metadata_credit(page,ledger.acquire(page_bytes)),"draft ring page credit refused");
        auto parent=create();parent->pages_.push_back(page);
        auto child=create();child->pages_.push_back(page);
        std::weak_ptr<Page> page_weak=page;page.reset();parent.reset();
        need(child->pages_[0]->k[0][0]==9 && ledger.bytes()==page_bytes,"draft child lost shared page or charge");
        child.reset();
        need(page_weak.expired() && ledger.bytes()==page_bytes,"draft page weak block charge released early");
        page_weak.reset();need(ledger.bytes()==0,"draft page final weak metadata leaked");
    }
    int count() const noexcept {return count_;}
    long long position() const noexcept {return base_+count_;}
    std::size_t exported_bytes() const noexcept {return exported_bytes_;}
    bool same_payload(const Exl3DraftHostRing& other) const;
    // Diagnostic only; production equality also requires model identity.
    bool same_represented_payload_for_test(const Exl3DraftHostRing& other) const;
    // Diagnostic value copy; excludes model identity and shared page ownership.
    std::shared_ptr<const Exl3DraftHostRing> detached_payload_for_test() const;
    // Exact chronological K/V bytes of every usable logical row. Diagnostic only.
    std::size_t write_represented_payload_for_test(const std::filesystem::path& path) const;
    static std::uint64_t visit_allocations(std::span<const std::shared_ptr<const Exl3DraftHostRing>> states,
        const std::function<void(const void*,std::size_t)>& visitor={},bool include_unused_capacity=true);
};

// Qualification-only selector entry point. Device buffers must cover rows*vocab
// F16 inputs and rows*16 outputs. Rows0 is a no-op; supported rows are1..7,
// with vocab16..248576 to exercise the fixed248320 vocabulary and nearby tails.
void dflash2_topk16_for_test(const std::uint16_t* logits, int rows, int vocab,
                           std::int64_t* ids, float* values, bool parallel,
                           bool local_merge = false,
                           cudaStream_t stream = nullptr,
                           unsigned int* nonfinite_flag = nullptr);

// Qualification entry: Q[queries,32,128], ring K/V[2048,8,128], block
// K/V[block,8,128], output[queries,32,128]. Count0..2047, queries0..8,
// block1..8, start0..2047. No input mutation. Queries0 is a no-op.
void dflash2_ring_attention_for_test(const std::uint16_t* q,
    const std::uint16_t* ring_k, const std::uint16_t* ring_v, int start, int count,
    const std::uint16_t* k, const std::uint16_t* v, std::uint16_t* out,
    int queries, int block, float scale, bool parallel, cudaStream_t stream = nullptr);
// True when the parallel ring attention at this context length takes the
// split-K tensor-core route (FP16 probabilities; not bitwise to the serial
// kernel, checked against the independent FP64 bound only).
bool dflash2_ring_attention_split_for_test(int count);

// Qualification entry for the draft F16 output-major dense projection and its
// exact K-major storage adapter. The two routes retain chronological FP32
// accumulation and differ only in represented-weight addressing.
void dflash2_dense_t_for_test(const std::uint16_t* input,
    const std::uint16_t* weights, std::uint16_t* output, int rows, int k, int n,
    bool weights_kmajor, cudaStream_t stream = nullptr);

// E5A2 minimal native DFlash2 draft adapter for the EXL3 Qwen3.8-27B target.
//
// The drafter (Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) is a separate 5-layer
// model: it owns no embeddings and no LM head. It consumes the TARGET hidden
// taps at layers {5,19,33,47,61} and the TARGET-side mask/block embedding, and it
// is verified with the TARGET H6 LM head. Under forced rejection every proposal is
// discarded, so DFlash2-enabled decode reduces to ordinary target decode.
//
// All draft kernels run natively on the target GPU. EXL3-quantized projections
// (fc/q/k/v/o/gate/up/down) reuse the qualified Exl3CudaLinearWorkspace family.
// NINFER_DFLASH2_PROJECTION_TIMING=1 is an eager-only diagnostic sampled at
// load: it allocates persistent CUDA events and prints per-call/aggregate GPU
// stream elapsed attribution after propose()'s existing final synchronization.
// Intervals include host-feed gaps. The disabled model creates, records and
// resolves no timing events and emits no timing output. Internal wall timing
// excludes entry validation and CSV output; use the harness for full API cost.
// NINFER_DFLASH2_FUSED_SELECTOR=1 is a separate default-off physical-C1
// proposal-chain candidate. It preserves the candidate order, FP32 reduction
// order, strict-greater tie rule, device-seed status slot, and proposal rows,
// but evaluates the bounded autoregressive chain in one 256-thread kernel.
// It falls back for position-confidence/timing diagnostics and does not alter
// acceptance counts, ring/W+1 semantics, or target/draft ordering.

class Exl3Dflash2DraftModel {
public:
    struct DraftPositionConfidence {
        float selected_edge_score = 0.0F;
        float runner_up_edge_score = 0.0F;
        float edge_margin = 0.0F;
        float selected_unary_score = 0.0F;
        float best_unary_score = 0.0F;
        float runner_up_unary_score = 0.0F;
        float unary_margin = 0.0F;
        std::int32_t selected_candidate_rank = -1;
    };
    struct ProjectionObservation {
        Exl3CudaLinearWeights weights{};
        Exl3CudaLinearMetadata metadata{};
        const std::uint16_t* input = nullptr;
        int rows = 0;
        int layer = -1;
        const char* name = nullptr;
        cudaStream_t stream = nullptr;
    };
    using ProjectionObserver = void (*)(const ProjectionObservation&, void* user);
    struct RingAttentionObservation {
        const std::uint16_t *q, *ring_k, *ring_v, *k, *v, *output;
        int start, count, queries, block, layer;
        float scale;
        cudaStream_t stream;
    };
    using RingAttentionObserver = void (*)(const RingAttentionObservation&, void*);

    // Strict native loader: config + quantization metadata + all 189 tensors
    // (names/shapes/dtypes) are validated and uploaded. Fail-closed on mismatch.
    static std::unique_ptr<Exl3Dflash2DraftModel> load(
        const std::filesystem::path& model_directory);

    // A new empty execution resource sharing only immutable uploaded weights
    // and their ring compatibility identity. KV/ring/scratch/graphs/observers
    // are never copied or shared. Caller serializes construction with raw APIs.
    // The new resource can outlive this object; no second artifact load/upload.
    std::unique_ptr<Exl3Dflash2DraftModel> create_execution() const;
    bool shares_weights_with(const Exl3Dflash2DraftModel& other) const noexcept;
    std::size_t execution_bytes() const noexcept; // scratch + workspaces + KV/ring

    GoptSubmissions gaming_submissions() const noexcept;
    ~Exl3Dflash2DraftModel();
    Exl3Dflash2DraftModel(const Exl3Dflash2DraftModel&) = delete;
    Exl3Dflash2DraftModel& operator=(const Exl3Dflash2DraftModel&) = delete;

    int block_capacity() const noexcept;   // max block length (mask token included)
    int spec_capacity() const noexcept;    // max proposals per block (block_capacity-1)
    int draft_layers() const noexcept;     // 5
    std::size_t weight_bytes() const noexcept;
    std::uint64_t dense_kmajor_launches() const noexcept;
    // Number of launches of the opt-in batched selector anchor-chain kernel.
    // The default/reference selector path does not increment this counter.
    std::uint64_t selector_batched_anchor_chain_calls() const noexcept;
    // Shared weight allocation records/vector only, excluding private workspaces.
    std::uint64_t weight_owner_metadata_bytes() const;
    // Private scratch/KV/ring records, vector capacity and linear workspace objects.
    std::uint64_t execution_owner_metadata_bytes() const;
    Exl3ResourceInventory execution_resources(const std::shared_ptr<Exl3Dflash2DraftModel>& owner,
        unsigned device_shortfall_for_test=0,unsigned metadata_shortfall_for_test=0) const;
    static bool attach_linear_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept;
    // Pristine/joined test owners only; indices follow fc,q,kv,o,mlp,down,head.
    bool fail_linear_retirement_for_test(unsigned index,bool after_accumulation=false) noexcept;
    static bool attach_linear_device_credit(const std::shared_ptr<const void>& owner,
        RetainedDeviceLedger::Ticket credit) noexcept;
    // Owner slots and records for the configured private menu, before construction.
    std::uint64_t execution_owner_metadata_bytes_required() const;
    // Fixed private seven-workspace menu; excludes scratch, KV and ring buffers.
    static std::size_t linear_workspace_bytes_required();
    // Derived from the configured allocation menu before private construction.
    std::size_t execution_bytes_required() const noexcept;
    template<class Coordinator>
    std::shared_ptr<Exl3Dflash2DraftModel> create_execution_reserved(Coordinator& authority) const {
        using Inventory=Exl3ResourceInventory;
        Inventory::Requirement requirement;requirement.configuration=0x44524654434C;
        requirement.add(Inventory::Domain::device,1,execution_bytes_required());
        requirement.add(Inventory::Domain::host_metadata,1,execution_owner_metadata_bytes());
        std::shared_ptr<Exl3Dflash2DraftModel> result;
        authority.allocate_startup_resources(requirement,[&](std::uint64_t configuration){
            if(configuration!=requirement.configuration)throw std::logic_error("draft clone reservation identity");
            auto prepared=std::shared_ptr<Exl3Dflash2DraftModel>(create_execution_impl(true));
            prepared->materialize_private_execution(&authority);
            auto actual=prepared->execution_resources(prepared);
            result=std::move(prepared);return actual;
        },[&]() noexcept {result.reset();},[&]() noexcept {
            if(generic_quarantined_allocations() || Exl3CudaLinearWorkspace::quarantined_workspaces())
                authority.seal_failed_startup_retirement();
        });
        result->finish_constructor_credits();
        return result;
    }
    std::size_t kv_bytes() const noexcept; // E5A2 minimal draft KV allocation
    static std::uint64_t generic_quarantined_allocations() noexcept;
    std::size_t uncertain_source_allocations_for_test() const noexcept;
    // Non-owning observation of the first source marked by an uncertain export.
    // Inspect only after the caller has joined/finished model execution.
    std::weak_ptr<const void> uncertain_source_owner_for_test() const noexcept;
    bool fail_generic_retirement_for_test() noexcept;
    struct GenericRetirementSnapshot {
        std::uint64_t bytes=0,device_credit=0,metadata_credit=0,record_bytes=0;
        int device=-1,error=0;
        bool pointer_retained=false;
    };
    static GenericRetirementSnapshot latest_generic_retirement_for_test() noexcept;
    std::uint64_t generic_owner_metadata_bytes() const noexcept;
    static void exercise_generic_retirement_for_test(bool uncertain_transfer=false);
    std::size_t scratch_bytes() const noexcept;

    // Diagnostic view of [last proposal count][5120] F16 H6 inputs. Valid only
    // after a successful proposal, until the next proposal/reset/destruction.
    // Caller must synchronize the proposal stream before a host copy.
    const std::uint16_t* last_head_input_device_for_test() const noexcept;
    // Same lifetime/synchronization contract; selector arrays contain rows*16.
    const std::uint16_t* last_head_logits_device_for_test() const noexcept;
    const std::int64_t* last_topk_ids_device_for_test() const noexcept;
    const float* last_topk_values_device_for_test() const noexcept;
    std::uint64_t local_topk_calls() const noexcept;
    std::uint64_t fused_topk_liveness_calls() const noexcept;
    // Test-only host telemetry for the most recent proposal. Empty unless the
    // model was loaded with NINFER_DFLASH2_POSITION_CONFIDENCE=1. Each entry
    // corresponds to one proposed position and is copied at the selector's
    // existing per-position synchronization; no post-verification label is
    // stored here.
    const std::vector<DraftPositionConfidence>&
    last_position_confidence_for_test() const noexcept;
    struct HostControlStorageSnapshot {
        std::uintptr_t context_positions=0,block_positions=0;
        std::size_t context_capacity=0,block_capacity=0;
        std::uint64_t generation=0;
        bool active=false;
    };
    HostControlStorageSnapshot host_control_storage_for_test() const noexcept;
    void require_host_control_idle_for_test() const;

    // Test-only eager observation of the 25 K5 q/k/v/o/down inputs in a proposal.
    // The callback runs immediately before the corresponding forward on its stream.
    // Weight pointers remain valid only for this model's lifetime; the input view is
    // valid only during the callback, so any stream-ordered copy must own its target
    // and complete before draft scratch is reused. Observer installation is mutable
    // and serialized with proposal execution. The callback must not mutate/reenter
    // the model or its buffers. Observation is incompatible with CUDA capture and
    // projection timing and may synchronize: it is qualification, not performance.
    // With no observer installed, propose owns no observation staging/CUDA resources.
    void set_projection_observer_for_test(ProjectionObserver observer,
                                          void* user = nullptr,bool include_gateup=false) noexcept;
    // Engine installs this before acquiring the private draft execution. Only
    // cached-ring B8 Q projections may suspend; all conditioning stays private.
    void set_shared_q_executor(Exl3DraftSharedQExecutor executor,bool block_kv=false,bool block_o=false,bool block_down=false,bool block_gateup=false);
    // Same eager-only non-mutating callback contract, after ring attention.
    // Views are valid only in the callback; synchronize/copy on its stream.
    void set_ring_attention_observer_for_test(RingAttentionObserver observer,
                                             void* user = nullptr) noexcept;

    // Runs one DFlash2 propose cycle on `stream`:
    //  block_ids      - [block_length] host token ids (last real token then mask ids)
    //  block_pos0     - absolute position of block_ids[0]
    //  layer_taps     - [5] device arrays, each [context_rows][5120] F16 post-layer rows
    //  context_rows   - rows actually present in every layer_taps array (<= window)
    //  ctx_pos0       - absolute position of layer_taps[i][0]
    //  window_rows    - number of most-recent rows of each array to use (<= context_rows)
    //  target_embedding_bf16 - target embed_tokens table BF16 [248320][5120] (device)
    //  target_head           - target H6 LM head weights
    //  target_head_metadata  - target H6 LM head metadata
    //  mask_token_id         - draft mask id (248070), embedded target-side
    // Returns the greedy proposal ids (one per speculative position), or an empty
    // vector if block_length < 2.
    [[nodiscard]] Exl3DraftDistributionAvailability
    sampled_distribution_availability() const noexcept {
        return Exl3DraftDistributionAvailability::current_dflash2();
    }

    std::vector<std::int64_t> propose(
        const std::vector<std::int64_t>& block_ids,
        int block_pos0,
        const std::uint16_t* const* layer_taps,
        int context_rows,
        int ctx_pos0,
        int window_rows,
        const std::uint16_t* target_embedding_bf16,
        const Exl3CudaLinearWeights& target_head,
        const Exl3CudaLinearMetadata& target_head_metadata,
        std::int64_t mask_token_id,
        cudaStream_t stream = nullptr);

    // Resets draft KV/position scratch. Does not touch the target.
    void reset(cudaStream_t stream = nullptr);

    // ===== E5A3 bounded draft KV ring (sliding window 2048, keep 2047) =====
    // The reference (RotatingKVCache max_size=sliding_window-1) proves draft KV
    // storage stays bounded independently of target context: per layer, K is cached
    // POST-k_norm and POST-RoPE at absolute positions (never re-rotated) and V raw.
    // Positions are implicit: committed tokens occupy the contiguous absolute span
    // [ring_base_abs, ring_base_abs + ring_count); slot(p) = p mod 2048.
    // Conv needs no persistent state (block-local, zero-padded first row, exactly
    // as the reference); the selector is stateless. propose() never mutates the ring.
    static constexpr int ring_capacity() noexcept { return 2048; }
    static constexpr int ring_keep() noexcept { return 2047; }  // sliding_window - 1
    // Ingests NEW target tap rows into the ring (bulk for prefill, 1/cycle for
    // decode). taps[t] points at rows fresh [5120] rows for tap layer t; abs_pos0
    // is the absolute position of the first new row. Must extend the committed span
    // contiguously (or start an empty ring). Keeps only the newest 2047 rows.
    void commit_target_block(const std::uint16_t* const* taps, int rows,
                               long long abs_pos0, cudaStream_t stream = nullptr);
    // Explicit bounded prefill path. Decode commits retain commit_target_block.
    //1..16 contiguous new rows; opt-in batches native projections at most8 rows.
    void commit_prefill_block(const std::uint16_t* const* taps, int rows,
                              long long abs_pos0, cudaStream_t stream = nullptr);
    // Optional request-scoped undo for one saturated-ring prefill commit of 1..8
    // rows. begin snapshots the affected physical slots on the same eager stream;
    // exactly one matching commit_prefill_block may follow. accept keeps it;
    // rollback restores every slot and the pre-commit ring/witness metadata.
    // The caller serializes this execution resource and uses the same stream for
    // all three calls. A failed rollback poisons the ring until reset.
    void begin_prefill_ring_undo(int rows, long long abs_pos0,
                                 cudaStream_t stream = nullptr);
    void accept_prefill_ring_undo(cudaStream_t stream = nullptr);
    void rollback_prefill_ring_undo(cudaStream_t stream = nullptr);
    // Explicit eager fresh initialization on an empty logical ring. Submit the
    // original ordered1..16-row calls on the declared stream through exact end.
    // Only whole calls ending<=end-2048 may be omitted; crossing calls preserve
    // their original internal partition. Finish before propose/commit/rewind.
    // An incomplete/failed scope cannot be reused; reset on its stream cancels it.
    struct FreshPrefillStatus {
        bool active = false, failed = false;
        long long start = 0, end = 0, submitted_cursor = 0;
        long long submitted_rows = 0, encoded_rows = 0, skipped_rows = 0;
    };
    void begin_fresh_prefill(long long start, long long end, cudaStream_t stream = nullptr);
    // Submit one original <=16-row partition that lies wholly before the
    // fresh window. Advances logical coverage without requiring discarded taps.
    void skip_fresh_prefill_block(int rows, long long abs_pos0,
                                  cudaStream_t stream = nullptr);
    void finish_fresh_prefill(cudaStream_t stream = nullptr);
    FreshPrefillStatus fresh_prefill_status() const noexcept;
    // propose() reading committed ring K/V instead of recomputing the window.
    // Requires block_pos0 == ring_base_abs() + ring_count() (contiguous tail).
    std::vector<std::int64_t> propose_cached(
        const std::vector<std::int64_t>& block_ids, int block_pos0,
        const std::uint16_t* target_embedding_bf16,
        const Exl3CudaLinearWeights& target_head,
        const Exl3CudaLinearMetadata& target_head_metadata,
        std::int64_t mask_token_id, cudaStream_t stream = nullptr);
    long long ring_base_abs() const noexcept;
    // Borrowed host input: retain through successful call completion, or through
    // stream retirement if the call fails after enqueue. Engine uses lane storage.
    // Proposals also report each position's top-16 candidate IDs and unary
    // scores ([position][16], unary order of the selector) for verification
    // tree construction.
    void set_proposal_candidates(bool enabled);
    const std::vector<std::int64_t>& last_candidate_ids() const noexcept;
    const std::vector<float>& last_candidate_unary() const noexcept;
    std::vector<std::int64_t> propose_cached_view(
        std::span<const std::int64_t> block_ids,int block_pos0,
        const std::uint16_t* target_embedding_bf16,
        const Exl3CudaLinearWeights& target_head,
        const Exl3CudaLinearMetadata& target_head_metadata,
        std::int64_t mask_token_id,cudaStream_t stream=nullptr);
    // The host block contains only mask placeholders. Row0 is copied from the
    // event-bound target packet and consumed once on the supplied draft stream.
    std::vector<std::int64_t> propose_cached_device_seed(
        std::span<const std::int64_t> masked_block,const Exl3DeviceGreedySeed& seed,
        int block_pos0,const std::uint16_t* target_embedding_bf16,
        const Exl3CudaLinearWeights& target_head,
        const Exl3CudaLinearMetadata& target_head_metadata,
        std::int64_t mask_token_id,cudaStream_t stream=nullptr);
    int ring_count() const noexcept;
    std::size_t ring_bytes() const noexcept;
    // Immutable FP16 projected K/V, bound to this resident drafter. Only proven
    // append lineage shares pages; false downloads all logical rows for audits.
    // Caller serializes commits and these eager operations on the same stream.
    std::shared_ptr<const Exl3DraftHostRing> export_host_ring(cudaStream_t stream=nullptr,bool share_prefix=true,
        const Exl3DraftHostRing::MetadataReservation& reserve={},bool fail_completion_for_test=false);
    // One shot at an actual submitted export download, including Engine callers.
    void fail_next_host_export_completion_for_test();
    void restore_host_ring(std::shared_ptr<const Exl3DraftHostRing> state,cudaStream_t stream=nullptr);
    // Scope changes normally invalidate the device witness. Rebinding may keep
    // it only when the exact immutable ring is still resident under a completed
    // nonzero prior acquisition/execution witness.
    void bind_ring_scope(std::uint64_t acquisition,std::uint64_t execution);
    bool rebind_ring_scope_if_resident(const std::shared_ptr<const Exl3DraftHostRing>& state,
                                       std::uint64_t acquisition,std::uint64_t execution);
    bool host_ring_resident(const std::shared_ptr<const Exl3DraftHostRing>& state) const noexcept;
    // Read-only observation; never exports, rebinds or submits stream work.
    std::shared_ptr<const Exl3DraftHostRing> completed_ring_parent_for_test() const noexcept;
    bool host_ring_lineage(const std::shared_ptr<const Exl3DraftHostRing>& state,
                           std::uint64_t acquisition,std::uint64_t execution) const noexcept;
    bool restore_host_ring_if_needed(std::shared_ptr<const Exl3DraftHostRing> state,
                                    cudaStream_t stream=nullptr);
    struct RingRestoreStats {std::uint64_t required=0,skipped=0;};
    RingRestoreStats ring_restore_stats() const noexcept;
    // FNV-1a digest of committed ring K/V per layer (test/equivalence use).
    std::array<std::uint64_t, 5> ring_digest(cudaStream_t stream = nullptr);
    // Diagnostic digest of all 2048 physical slots, including the spare slot.
    std::array<std::uint64_t, 5> physical_ring_digest_for_test(
        cudaStream_t stream = nullptr);
    // Logical rollback: rewinds the committed length (no copies).
    // require(count <= ring_count()).
    void rewind_to(int count, cudaStream_t stream = nullptr);

private:
    // Physical execution lanes may share immutable weights through distinct
    // create_execution() resources, but must serialize this object's scratch/ring.
    // A lane holds the lock only while it
    // owns an active coordinator request; raw draft APIs remain caller-serialized.
    friend class Exl3Dflash2Execution;
    friend class Exl3EngineCore;
    static std::unique_ptr<Exl3Dflash2DraftModel> load_impl(
        const std::filesystem::path& model_directory,bool defer_private);
    std::unique_ptr<Exl3Dflash2DraftModel> create_execution_impl(bool defer_private) const;
    void materialize_private_execution(class Exl3VeriCacheServingCoordinator* authority=nullptr);
    void finish_constructor_credits() noexcept;
    mutable std::mutex execution_mutex_;
    void require_no_fresh_prefill();
    void commit_target_block_internal(const std::uint16_t* const* taps, int rows,
                                     long long abs_pos0, cudaStream_t stream);
    void commit_prefill_block_internal(const std::uint16_t* const* taps, int rows,
                                      long long abs_pos0, cudaStream_t stream);
    struct Impl;
    explicit Exl3Dflash2DraftModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    // E5A3: shared forward implementation. use_ring=false is the qualified E5A2
    // path (recompute window from taps); use_ring=true reads committed ring K/V.
    // Branches are confined to context sourcing, the attention call, and E5A2
    // one-shot diagnostics (skipped on the ring path).
    std::vector<std::int64_t> propose_internal(
        std::span<const std::int64_t> block_ids, int block_pos0,
        const std::uint16_t* const* layer_taps, int context_rows, int ctx_pos0,
        int window_rows, const std::uint16_t* target_embedding_bf16,
        const Exl3CudaLinearWeights& target_head,
        const Exl3CudaLinearMetadata& target_head_metadata,
        std::int64_t mask_token_id, cudaStream_t stream, bool use_ring,
        const Exl3DeviceGreedySeed* device_seed=nullptr);
};

// E5A2 diagnostic tap-history container bound to a target Exl3TextContext.
// It accumulates the real post-layer tap rows produced by prefill/decode so a
// draft propose can read a recent context window. Host/device copies only; it
// never mutates target state.
class Exl3Dflash2TapHistory {
public:
    // capacity rows retained per tap layer.
    explicit Exl3Dflash2TapHistory(int capacity);
    ~Exl3Dflash2TapHistory();
    Exl3Dflash2TapHistory(const Exl3Dflash2TapHistory&) = delete;
    Exl3Dflash2TapHistory& operator=(const Exl3Dflash2TapHistory&) = delete;

    // After ctx->prefill(token_ids): copies the captured prefill rows
    // (positions [position-rows, position)) into history.
    void capture_prefill(const class Exl3TextContext& ctx, int rows,
                         cudaStream_t stream = nullptr);
    // After ctx->decode(token): copies the just-decoded row
    // (absolute position ctx.position()-1) into history.
    void capture_decode(const class Exl3TextContext& ctx, cudaStream_t stream = nullptr);

    void reset();
    int total_rows() const noexcept { return total_rows_; }
    int capacity() const noexcept { return capacity_; }
    // Reads window_rows of the most recent rows into dst (5 consecutive 5120-wide
    // F16 segments per row). Returns the absolute position of the first returned row.
    int copy_window(const class Exl3TextContext& ctx, int window_rows,
                    std::uint16_t* const* dst, cudaStream_t stream = nullptr) const;
    // Raw device row pointer for the most recent captured row of one tap layer
    // (used by the tap-equivalence tests). total_rows()>0 required.
    const std::uint16_t* latest_row_device(int layer_index,
                                           cudaStream_t stream = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int total_rows_ = 0;
    int capacity_ = 0;
};

} // namespace ninfer::exl3
