#pragma once

#include <cuda_runtime_api.h>
#include "core/decode_graph.h"
#include "exl3/linear_workspace_requirements.h"
#include "exl3/retained_descriptor_ledger.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <array>
#include <atomic>
#include <optional>
#include <limits>

namespace ninfer::exl3 {

// E2B intentionally exposes only device views. The packed EXL3 representation remains
// the source of truth; callers own the immutable model allocations and the output buffer.
struct Exl3CudaLinearMetadata {
    int in_features = 0;
    int out_features = 0;
    int K = 0;
    bool mcg = false;
    bool mul1 = false;
    bool has_bias = false;
};

struct Exl3CudaLinearWeights {
    const std::uint16_t* trellis = nullptr; // [320, 1088, 80], device I16 bits
    const std::uint16_t* suh = nullptr;     // [5120], device F16 bits
    const std::uint16_t* svh = nullptr;     // [17408], device F16 bits
    const std::int32_t* mul1 = nullptr;     // scalar, device I32
};

// Test-only upper-bound discriminator for replacing scalar MUL1 decoding with
// one model-owned exact state lookup.  No production dispatch consumes this
// result and table construction is reported separately from steady-state work.
struct Exl3Mul1LookupDiscriminator {
    double arithmetic_median_us = 0.0;
    double lookup_median_us = 0.0;
    double table_setup_us = 0.0;
    std::size_t state_pairs = 0;
    std::size_t table_bytes = 0;
    std::uint32_t multiplier = 0;
    std::uint64_t mismatches = 0;
};

Exl3Mul1LookupDiscriminator exl3_mul1_lookup_discriminator_for_test(
    std::uint32_t multiplier);

// Test-only real-shape discriminator for issuing the next split-plane load
// before the unchanged chronological FP32 accumulation and selected output
// Hadamard. No production dispatch consumes this result.
struct Exl3SplitPlanePrefetchDiscriminator {
    double selected_median_us = 0.0;
    double prefetched_median_us = 0.0;
    std::size_t rows = 0;
    std::size_t output_features = 0;
    std::size_t split_count = 0;
    std::size_t accumulation_bytes = 0;
    std::uint64_t mismatches = 0;
};

Exl3SplitPlanePrefetchDiscriminator
exl3_split_plane_prefetch_discriminator_for_test();

// Test-only bounded observation of the production native reconstruction kernel.
// Output contains exactly256 half-bit values in encoded fragment order.
void exl3_reconstruct_native_tile_for_test(const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,int input_tile,int output_tile,
    std::uint16_t* output,cudaStream_t stream=nullptr);

// Test-only full-owner materialization in the exact fragment order consumed by
// the tensor-core B operand. The packed weights remain authoritative.
void exl3_reconstruct_native_weights_for_test(
    const Exl3CudaLinearWeights& weights,
    const Exl3CudaLinearMetadata& metadata,std::uint16_t* output,
    std::size_t output_bytes,cudaStream_t stream=nullptr);

// A small-M candidate must be admitted by its owning target call site as well
// as enabled in the construction-latched workspace policy.  Ordinary callers
// always use ordinary, so an environment setting cannot widen dispatch alone.
enum class Exl3CudaLinearAdmission {
    ordinary,
    target_continuation_gate_up,
    target_prefill_gate_up,
    target_continuation_down,
    target_continuation_o,
    target_continuation_z,
    target_wide_prefill,
    // Explicit native Qwen3.8 MTP owner.  This is intentionally separate
    // from target_wide_prefill so ordinary/default callers cannot select the
    // K4 wide route merely by changing workspace flags.
    native_mtp_wide_prefill,
    // One-row native MTP bridge.  This is a separate admission so the
    // qualified wide-row bound above remains intact while the first real
    // DFlash2 -> MTP step can use the same production K4 operator.
    native_mtp_one_step,
    target_initial16,
    draft_prefill_fc,
    target_continuation_qkv,
    target_continuation_kv,
    target_continuation_q,
    draft_shared_q_m16,
    draft_shared_kv_m16,
    draft_shared_o_m16,
    draft_shared_down_m16,
    draft_shared_gateup_m16,
    target_continuation_head,
};

// Context-owned transient arena for sequential target layers. Borrowed temporary
// trace pointers remain valid only until another sharing layer uses the arena.
// Retained repair inputs, public context QKV exports and persistent state stay
// private. Standalone layers with an empty view retain private trace storage.
struct Exl3CudaLayerScratchView {
    void* data = nullptr;
    std::size_t bytes = 0;
};

// Borrowed input transform storage. Every consumer must finish before any
// workspace sharing this view transforms another input. Owner outlives queued use.
// Base must be 16-byte aligned for the admitted async-A readers.
struct Exl3CudaTransformView {
    std::uint16_t* data = nullptr;
    std::size_t bytes = 0;
};

// Borrowed accumulation storage for projections ordered on one stream (or
// explicitly synchronized streams). Owner must outlive every workspace and
// queued use. It never stores transformed inputs or observable layer outputs.
struct Exl3CudaAccumulationView {
    float* data = nullptr;
    std::size_t bytes = 0;
};

struct Exl3ReconstructedExactStats {
    bool reservation_fallback = false;
    std::uint64_t calls = 0;
    std::uint64_t rows = 0;
    std::uint64_t k6_down_calls = 0;
    std::uint64_t k6_down_rows = 0;
    std::uint64_t k6_gate_up_calls = 0;
    std::uint64_t k6_gate_up_rows = 0;
    std::size_t workspace_bytes = 0;
};

// One context-owned, transient Down weight slab, overwritten for every projection.
// All borrowers execute on the context's ordered stream. No pointer into it is
// retained by a host root, continuation, or model. The owner outlives queued work.
class Exl3ReconstructionStream;
struct Exl3ReconstructedExactView {
    std::uint16_t* data = nullptr;
    std::size_t bytes = 0;
    Exl3ReconstructedExactStats* stats = nullptr;
    bool allow_k6_down = false;
    bool allow_k6_gate_up = false;
    Exl3ReconstructionStream* ordered_stream = nullptr;
    // Retains data, statistics and ordering witness as one backing group.
    std::shared_ptr<void> backing_owner;
    std::shared_ptr<const void> model_owner;
};

// Numeric telemetry for one complete transformed-basis reconstruct + cuBLAS
// GEMM projection. The qualification controls remain test-only; the separate
// FAST_SAME_WEIGHTS_FP16KV_PREFILL opt-in admits the same operator for the
// explicitly labeled ordinary FP16 device-KV fast backend.
struct Exl3ReconstructGemmPhaseTiming {
    double input_transform_us = 0.0;
    double reconstruct_us = 0.0;
    double gemm_us = 0.0;
    double output_transform_us = 0.0;
    double total_us = 0.0;
};

struct Exl3ReconstructGemmStats {
    std::uint64_t calls = 0;
    std::uint64_t k5_calls = 0;
    std::uint64_t k5_lt_calls = 0;
    std::uint64_t large_lt_calls = 0;
    std::uint64_t mxfp8_calls = 0;
    std::uint64_t mxfp8_rows = 0;
    std::uint64_t nvfp4_calls = 0;
    std::uint64_t nvfp4_rows = 0;
    std::uint64_t fused_mlp_calls = 0;
    std::uint64_t k6_calls = 0;
    std::uint64_t k7_calls = 0;
    std::uint64_t rows = 0;
    std::uint64_t fused_original_calls = 0;
    std::uint64_t fused_original_rows = 0;
    std::uint64_t fused_gate_up_down_calls = 0;
    std::uint64_t fused_gate_up_down_rows = 0;
    std::uint64_t fused_down_residual_calls = 0;
    std::uint64_t fused_down_residual_rows = 0;
    // Exploratory, separately gated backend telemetry.  This is distinct
    // from the existing FP32-compute numeric route and remains default-off.
    std::uint64_t fp16_compute_calls = 0;
    std::uint64_t fp16_compute_rows = 0;
    // Actual FP16 slab writes from reconstruct-then-GEMM calls. Packed-direct
    // and persistent leaves do not contribute to this count.
    std::uint64_t reconstructed_weight_bytes = 0;
    std::uint64_t reconstructed_weight_calls = 0;
    std::uint64_t reused_weight_calls = 0;
    std::uint64_t reused_weight_bytes = 0;
    std::uint64_t prefetched_weight_submissions = 0;
    std::uint64_t prefetched_weight_hits = 0;
    std::size_t cached_weight_capacity_bytes = 0;
    // Separately labeled same-weight FP16-KV packed K6 backend. This is direct
    // packed-weight execution, not reconstruction or re-quantizing.
    std::uint64_t packed_direct_k6_calls = 0;
    std::uint64_t packed_direct_k6_rows = 0;
    // Target GDN bulk MLP packed K5 gate/up/down. The FP16 GEMM destination
    // boundary is preserved before the output transform.
    std::uint64_t packed_direct_k5_gate_up_calls = 0;
    std::uint64_t packed_direct_k5_down_calls = 0;
    std::uint64_t packed_direct_k5_rows = 0;
    // Separately labeled native cooperative prefill leaf. It consumes the
    // packed EXL3 weights directly and is never folded into packed_direct_k6.
    std::uint64_t persistent_prefill_calls = 0;
    std::uint64_t persistent_prefill_rows = 0;
    // Separately labeled native Mia-shaped cooperative FP16-output prefill.
    // This is not the FP32 persistent adapter above and remains default-off.
    std::uint64_t mia_prefill_fp16_calls = 0;
    std::uint64_t mia_prefill_fp16_rows = 0;
    std::size_t workspace_bytes = 0;
};

class Exl3CudaReconstructGemmWorkspace {
public:
    Exl3CudaReconstructGemmWorkspace(int in_features, int out_features,
                                     int max_rows,
                                     bool accept_transpose_shape = false,
                                     bool accept_all_model_shapes = false);
    ~Exl3CudaReconstructGemmWorkspace();

    Exl3CudaReconstructGemmWorkspace(
        const Exl3CudaReconstructGemmWorkspace&) = delete;
    Exl3CudaReconstructGemmWorkspace& operator=(
        const Exl3CudaReconstructGemmWorkspace&) = delete;

    void forward_numeric_candidate(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,
        const std::uint16_t* input,
        std::uint16_t* output,
        int rows,
        cudaStream_t stream = nullptr,
        Exl3ReconstructGemmPhaseTiming* timing = nullptr,
        const std::uint16_t* up = nullptr,
        std::uint16_t* activation = nullptr,
        const std::uint16_t* residual = nullptr,
        std::uint16_t* down_trace = nullptr,
        int trace_row_base = 0);

    // Produce the represented SiLU(gate)*up activation and the down input
    // transform in one tile-owned launch. The remaining projection path is
    // identical to forward_numeric_candidate.
    void forward_numeric_gate_up_down(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,
        const std::uint16_t* gate,
        const std::uint16_t* up,
        std::uint16_t* activation,
        std::uint16_t* output,
        int rows,
        cudaStream_t stream = nullptr);
    bool supports_fused_gate_up_down() const noexcept;
    // Prefill MLP on the block-quantized route: gate/up keep their raw GEMM
    // outputs and the down projection's quantized input is produced in one
    // pass (gate/up output transforms, SiLU*up, down input transform and
    // block quantization). Returns false without work when not admitted.
    // Gate/up buffers then hold untransformed values; activation is unused.
    bool forward_numeric_mlp(
        const Exl3CudaLinearWeights& gate, const Exl3CudaLinearMetadata& gate_metadata,
        const Exl3CudaLinearWeights& up, const Exl3CudaLinearMetadata& up_metadata,
        const Exl3CudaLinearWeights& down, const Exl3CudaLinearMetadata& down_metadata,
        const std::uint16_t* input, std::uint16_t* gate_output, std::uint16_t* up_output,
        std::uint16_t* activation, std::uint16_t* output, int rows,
        cudaStream_t stream = nullptr);

    // Reuse only the current layer's reconstructed projections while a
    // layer-major prompt suffix is processed in causal row chunks. The caller
    // owns the stream ordering and closes the scope before the next layer.
    void begin_layer_reuse(std::size_t max_cached_bytes, bool allow_k5 = false);
    void end_layer_reuse() noexcept;
    // Model layer of the active layer-major reuse scope (-1 outside one); the
    // NVFP4 mode 3 policy keeps layers >= 56 on MXFP8.
    void set_prefill_layer(int layer) noexcept;
    // Schedule one immutable K5 slab in the current layer's bounded reuse
    // slot. The next numeric forward joins the preparation before GEMM.
    bool prefetch_numeric_weight(const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,int rows,cudaStream_t stream);

    // Isolated V6 bring-up: donor-style GEMM destination precision BEFORE the
    // output Hadamard. Exactly one destination is required. No text dispatch
    // or T69 policy selects this method; full-model vision qualification pending.
    void forward_v6_numeric(const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata, const std::uint16_t* input,
        std::uint16_t* half_output, float* float_output, int rows,
        cudaStream_t stream = nullptr);

    std::size_t workspace_bytes() const noexcept;
    static std::size_t workspace_bytes_required(int in_features,int out_features,int rows,
        bool accept_transpose_shape);
    Exl3ReconstructGemmStats stats() const noexcept;
    bool supports(const Exl3CudaLinearMetadata& metadata,
                  int rows) const noexcept;
    bool accepts_all_model_shapes() const noexcept;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// The first supported unset-default arm matches the qualified wide/staged
// shared-accumulation layouts at final capacities 128 and 1024. Width and
// accumulation selectors remain explicit; callers resolve their own optional
// unset/0/1 environment latch with this common predicate.
inline bool exl3_prefill_qualified_default_group(
    int final_capacity, bool wide_prefill_owner,
    bool staged_prefill_enabled, bool shared_accumulation_enabled) noexcept {
    return wide_prefill_owner && staged_prefill_enabled &&
        shared_accumulation_enabled &&
        (final_capacity == 128 || final_capacity == 1024);
}

// Apply two independent SUH-qualified input transforms to one activation.
// The butterfly and F16 rounding order of each result matches transform_input;
// only the common activation load and launch are shared. Destinations must be
// disjoint and sized rows * in_features.
void exl3_transform_input_pair(
    const Exl3CudaLinearWeights& first_weights,
    const Exl3CudaLinearMetadata& first_metadata,
    const Exl3CudaLinearWeights& second_weights,
    const Exl3CudaLinearMetadata& second_metadata,
    const std::uint16_t* input,
    std::uint16_t* first_transformed,
    std::uint16_t* second_transformed,
    int rows,
    cudaStream_t stream = nullptr);

// Persistent temporary storage for the isolated EXL3 leaf. It is allocated once and
// reused across calls; no CUDA allocation occurs in forward().
// GDN control projection operands for a merged single-row qkv/z launch.
struct Exl3GdnControlSide {
    const std::uint16_t* a_weight = nullptr;
    const std::uint16_t* b_weight = nullptr;
    const float* a_log = nullptr;
    const float* dt_bias = nullptr;
    float* a_output = nullptr;
    float* b_output = nullptr;
    float* beta_trace = nullptr;
    float* g_trace = nullptr;
    int heads = 0;
};

class Exl3CudaLinearWorkspace {
public:
    explicit Exl3CudaLinearWorkspace(int max_rows = 16);
    Exl3CudaLinearWorkspace(int in_features, int out_features, int max_rows = 16,
                            bool default_enable_draft_small_m = false,
                            bool target_gateup_owner = false,
                            bool target_down_owner = false,
                            bool target_o_k7_owner = false,
                            bool target_z_k6_owner = false,
                            bool target_wide_prefill_owner = false,
                            bool draft_prefill_fc_owner = false,
                            Exl3CudaAccumulationView accumulation = {},
                            Exl3CudaTransformView transformed = {},
                            bool target_qkv_k6_owner = false,
                            bool target_kv_owner = false,
                            bool target_q_k6_owner = false,
                            std::optional<RetainedDeviceLedger::Ticket> constructor_device_credit = {},
                            std::optional<RetainedDescriptorLedger::Ticket> constructor_metadata_credit = {});
    void release_constructor_credits_after_commit() noexcept {
        storage_retirement_->transform_credit.reset();storage_retirement_->accumulation_credit.reset();
        storage_retirement_->metadata_credit.reset();constructor_object_credit_.reset();
    }
    ~Exl3CudaLinearWorkspace();
    static constexpr std::size_t metadata_bytes() noexcept {
        return sizeof(Exl3CudaLinearWorkspace)+sizeof(StorageRetirement);
    }
    static constexpr std::size_t storage_retirement_metadata_bytes() noexcept {return sizeof(StorageRetirement);}
    bool can_attach_metadata_credit() const noexcept {
        return storage_retirement_ && !storage_retirement_->metadata_credit;
    }
    bool can_attach_device_credit() const noexcept {
        return storage_retirement_ && !storage_retirement_->transform_credit &&
            !storage_retirement_->accumulation_credit &&
            (!transformed_owned_ || transformed_) && (!accum_owned_ || accum_);
    }
    bool attach_device_credit(RetainedDeviceLedger::Ticket credit) noexcept {
        if(!workspace_bytes_ || !can_attach_device_credit() || credit.bytes()!=workspace_bytes_)return false;
        if(transformed_owned_) {
            auto part=credit.split(transformed_capacity_bytes_);if(!part)return false;
            storage_retirement_->transform_credit.emplace(std::move(*part));
        }
        if(accum_owned_) {
            auto part=credit.split(accumulation_capacity_bytes_);if(!part)return false;
            storage_retirement_->accumulation_credit.emplace(std::move(*part));
        }
        return true;
    }
    bool attach_metadata_credit(RetainedDescriptorLedger::Ticket credit) noexcept {
        if(!can_attach_metadata_credit() || credit.bytes()!=metadata_bytes())return false;
        storage_retirement_->metadata_credit.emplace(std::move(credit));return true;
    }
    std::uint64_t metadata_credit_bytes_for_test() const noexcept {
        return storage_retirement_ && storage_retirement_->metadata_credit?
            storage_retirement_->metadata_credit->bytes():0;
    }
    // Inventory aliases must point at this workspace and retain its enclosing owner.
    static bool attach_owner_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* workspace=const_cast<Exl3CudaLinearWorkspace*>(static_cast<const Exl3CudaLinearWorkspace*>(owner.get()));
        return workspace && workspace->attach_metadata_credit(std::move(credit));
    }
    static bool attach_owner_device_credit(const std::shared_ptr<const void>& owner,
        RetainedDeviceLedger::Ticket credit) noexcept {
        auto* workspace=const_cast<Exl3CudaLinearWorkspace*>(static_cast<const Exl3CudaLinearWorkspace*>(owner.get()));
        return workspace && workspace->attach_device_credit(std::move(credit));
    }
    static void fail_constructor_cleanup_for_test(bool cleanup_failure) noexcept {
        constructor_failure_for_test_=cleanup_failure?2:1;
    }
    // Joined/pristine owning callers transfer their complete object here.
    // A failed free retains the object and remaining pointers without allocation.
    static void retire_owned(std::unique_ptr<Exl3CudaLinearWorkspace> owner) noexcept;
    static void retire_slot(Exl3CudaLinearWorkspace*& slot) noexcept {
        std::unique_ptr<Exl3CudaLinearWorkspace> owner(slot);
        slot=nullptr; // Unpublish before cleanup can retain the complete owner.
        retire_owned(std::move(owner));
    }
    struct RetainedDeleter {
        RetainedDeleter() noexcept=default;
        RetainedDeleter(std::default_delete<Exl3CudaLinearWorkspace>) noexcept {}
        RetainedDeleter& operator=(std::default_delete<Exl3CudaLinearWorkspace>) noexcept {return *this;}
        void operator()(Exl3CudaLinearWorkspace* workspace) const noexcept {
            retire_owned(std::unique_ptr<Exl3CudaLinearWorkspace>(workspace));
        }
    };
    // Accept existing make_unique factories, but retain failed cleanup at every
    // reset/destruction edge without changing enclosing member destruction order.
    using Owner=std::unique_ptr<Exl3CudaLinearWorkspace,RetainedDeleter>;
    static std::size_t quarantined_workspaces() noexcept {return quarantine_count_.load();}
    static std::uint64_t process_target_m1_k6_n16_calls_for_test() noexcept {
        return process_target_m1_k6_n16_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_target_m1_k7_three_word_calls_for_test() noexcept {
        return process_target_m1_k7_three_word_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_fp16_accum_calls_for_test() noexcept {
        return process_fast_same_weights_fp16_accum_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_fp16_m1_calls_for_test() noexcept {
        return process_fast_same_weights_fp16_m1_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_fp16_m2_8_down_calls_for_test() noexcept {
        return process_fast_fp16_m2_8_down_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_fp16_m2_8_fused_down_calls_for_test() noexcept {
        return process_fast_fp16_m2_8_fused_down_calls_.load(std::memory_order_relaxed);
    }
    // Operation indices: Q, QKV, Z, O, gate/up. Gate and up share the
    // registered shape/admission; each invocation is counted separately.
    static std::uint64_t coherent_wide_k6_calls_for_test(int operation) noexcept {
        return operation >= 0 && operation < 5
            ? coherent_wide_k6_calls_[operation].load(std::memory_order_relaxed) : 0;
    }
    static std::uint64_t coherent_wide_k6_rows_for_test(int operation) noexcept {
        return operation >= 0 && operation < 5
            ? coherent_wide_k6_rows_[operation].load(std::memory_order_relaxed) : 0;
    }
    static std::uint64_t coherent_wide_k6_split10_calls_for_test(int operation) noexcept {
        return operation >= 0 && operation < 5
            ? coherent_wide_k6_split10_calls_[operation].load(std::memory_order_relaxed) : 0;
    }
    static std::size_t coherent_wide_k6_shared_bytes_for_test() noexcept;
    int coherent_wide_k6_resident_capacity_for_test() const noexcept {
        return coherent_wide_k6_resident_capacity_;
    }
    int coherent_wide_k6_registers_per_thread_for_test() const noexcept {
        return coherent_wide_k6_registers_per_thread_;
    }
    int coherent_wide_k6_operation_for_test() const noexcept {
        return coherent_wide_k6_operation_;
    }
    static std::uint64_t process_coherent_down_k6_calls_for_test() noexcept {
        return process_coherent_down_k6_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_coherent_down_k6_rows_for_test() noexcept {
        return process_coherent_down_k6_rows_.load(std::memory_order_relaxed);
    }
    int coherent_down_k6_resident_capacity_for_test() const noexcept {
        return coherent_down_k6_resident_capacity_;
    }
    static std::uint64_t process_coherent_down_k7_calls_for_test() noexcept {
        return process_coherent_down_k7_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_coherent_down_k7_rows_for_test() noexcept {
        return process_coherent_down_k7_rows_.load(std::memory_order_relaxed);
    }
    int coherent_down_k7_resident_capacity_for_test() const noexcept {
        return coherent_down_k7_resident_capacity_;
    }
    static std::size_t coherent_down_k7_shared_bytes_for_test() noexcept;
    static std::uint64_t process_coherent_o_k7_calls_for_test() noexcept {
        return process_coherent_o_k7_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_coherent_o_k7_rows_for_test() noexcept {
        return process_coherent_o_k7_rows_.load(std::memory_order_relaxed);
    }
    int coherent_o_k7_resident_capacity_for_test() const noexcept {
        return coherent_o_k7_resident_capacity_;
    }
    static std::uint64_t process_fast_same_weights_fp16_m1_wide_n32_calls_for_test() noexcept {
        return process_fast_same_weights_fp16_m1_wide_n32_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_fp16_m1_n16_calls_for_test() noexcept {
        return process_fast_same_weights_fp16_m1_n16_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_native_persistent_m1_calls_for_test() noexcept {
        return process_fast_native_persistent_m1_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_native_mia_m1_fp16_calls_for_test() noexcept {
        return process_fast_native_mia_m1_fp16_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_native_mia_target_prefill_fp16_calls_for_test() noexcept {
        return process_fast_native_mia_target_prefill_fp16_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_fp16kv_m1_gate_up_pair_calls_for_test() noexcept {
        return process_fast_same_weights_fp16kv_m1_gate_up_pair_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_fp16kv_m1_kv_pair_calls_for_test() noexcept {
        return process_fast_same_weights_fp16kv_m1_kv_pair_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_fp16_m1_n64_calls_for_test() noexcept {
        return process_fast_same_weights_fp16_m1_n64_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_fp16_m1_n64_k5_calls_for_test() noexcept {
        return process_fast_same_weights_fp16_m1_n64_k5_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_int8_gemv_calls_for_test() noexcept {
        return process_fast_same_weights_int8_gemv_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_int8_gemv_k7_calls_for_test() noexcept {
        return process_fast_same_weights_int8_gemv_k7_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_int8_gemv_down_k6_calls_for_test() noexcept {
        return process_fast_same_weights_int8_gemv_down_k6_calls_.load(std::memory_order_relaxed);
    }
    static std::uint64_t process_fast_same_weights_int8_gemv_down_k7_calls_for_test() noexcept {
        return process_fast_same_weights_int8_gemv_down_k7_calls_.load(std::memory_order_relaxed);
    }
    struct RetirementSnapshot {
        int device=-1,error=0,restore_error=0;
        bool transform_retained=false,accumulation_retained=false;
        std::size_t transformed_bytes=0,accumulation_bytes=0;
    };
    static RetirementSnapshot latest_retirement_for_test() noexcept {
        const auto* owner=quarantine_.load(std::memory_order_acquire);
        return owner?RetirementSnapshot{owner->owner_device_,owner->retirement_error_,owner->retirement_restore_error_,
            owner->transformed_owned_ && owner->transformed_,owner->accum_owned_ && owner->accum_,
            owner->transformed_owned_ && owner->transformed_?owner->transformed_capacity_bytes_:0,
            owner->accum_owned_ && owner->accum_?owner->accumulation_capacity_bytes_:0}:RetirementSnapshot{};
    }
    static RetirementSnapshot latest_storage_retirement_for_test() noexcept {
        const auto* storage=storage_quarantine_.load(std::memory_order_acquire);
        return storage?RetirementSnapshot{storage->device,storage->error,storage->restore_error,
            storage->transformed!=nullptr,storage->accumulation!=nullptr,
            storage->transformed_bytes,storage->accumulation_bytes}:RetirementSnapshot{};
    }
    static std::array<std::uint64_t,2> latest_retirement_credits_for_test() noexcept {
        const auto* owner=quarantine_.load(std::memory_order_acquire);
        if(!owner || !owner->storage_retirement_)return {};
        const auto& storage=*owner->storage_retirement_;
        return {(storage.metadata_credit?storage.metadata_credit->bytes():0)+
            (owner->constructor_object_credit_?owner->constructor_object_credit_->bytes():0),
            (storage.transform_credit?storage.transform_credit->bytes():0)+
            (storage.accumulation_credit?storage.accumulation_credit->bytes():0)};
    }
    static std::array<std::uint64_t,2> latest_storage_credits_for_test() noexcept {
        const auto* storage=storage_quarantine_.load(std::memory_order_acquire);
        if(!storage)return {};
        return {storage->metadata_credit?storage->metadata_credit->bytes():0,
            (storage->transform_credit?storage->transform_credit->bytes():0)+
            (storage->accumulation_credit?storage->accumulation_credit->bytes():0)};
    }
    struct QuarantineFootprint {std::size_t device_bytes=0,host_object_bytes=0;};
    // Published quarantine nodes are immutable and never reclaimed. This is an
    // allocation-free snapshot of one acquired list head, excluding borrowed
    // storage and successfully freed pointers. Allocator overhead is not known.
    // This reports physical retention; it does not transfer residency credits.
    static std::optional<QuarantineFootprint> quarantine_footprint() noexcept {
        QuarantineFootprint result;
        const auto add=[](std::size_t& total,std::size_t bytes) noexcept {
            if(bytes>std::numeric_limits<std::size_t>::max()-total)return false;
            total+=bytes;return true;
        };
        for(auto* owner=quarantine_.load(std::memory_order_acquire);owner;owner=owner->quarantine_next_) {
            if(!add(result.host_object_bytes,metadata_bytes()) ||
                !add(result.device_bytes,owner->transformed_owned_ && owner->transformed_?
                    owner->transformed_capacity_bytes_:0) ||
                !add(result.device_bytes,owner->accum_owned_ && owner->accum_?
                    owner->accumulation_capacity_bytes_:0))return std::nullopt;
        }
        for(auto* storage=storage_quarantine_.load(std::memory_order_acquire);storage;storage=storage->next) {
            if(!add(result.host_object_bytes,sizeof(StorageRetirement)) ||
                !add(result.device_bytes,storage->transformed_bytes) ||
                !add(result.device_bytes,storage->accumulation_bytes))return std::nullopt;
        }
        return result;
    }
    void fail_owned_retirement_for_test(bool after_accumulation=false) noexcept {
        retirement_failure_stage_for_test_=after_accumulation?2:1;
    }

    Exl3CudaLinearWorkspace(const Exl3CudaLinearWorkspace&) = delete;
    Exl3CudaLinearWorkspace& operator=(const Exl3CudaLinearWorkspace&) = delete;

    void forward(const Exl3CudaLinearWeights& weights,
                 const Exl3CudaLinearMetadata& metadata,
                 const std::uint16_t* input,
                 std::uint16_t* output,
                 int rows,
                 cudaStream_t stream = nullptr,
                 Exl3CudaLinearAdmission admission =
                     Exl3CudaLinearAdmission::ordinary);

    // Opt-in operator qualification: decode one K7 projection into caller-owned
    // FP16 fragment order, then retain the canonical M16 MMA and split reduction.
    // Scratch is overwritten each call and must be private to this ordered stream.
    void forward_reconstructed_exact_for_test(const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata, const std::uint16_t* input,
        std::uint16_t* output, int rows, std::uint16_t* decoded,
        std::size_t decoded_bytes, cudaStream_t stream = nullptr);

    // Test-only actual-owner gate for persistent decoded-B economics. It keeps
    // the selected N16 async-A MMA, split reduction and output transform.
    void forward_predecoded_m1_k6_for_test(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
        std::uint16_t* output,int rows,const std::uint16_t* decoded,
        std::size_t decoded_bytes,cudaStream_t stream=nullptr);

    // Test-only rows=1 K6/N16 discriminator. The selected async-A topology,
    // split boundaries and MMA/reduction order are unchanged; only the proven
    // exact two-word K6 lane-window extraction replaces generic div/mod decode.
    void forward_fast_decode_m1_k6_for_test(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
        std::uint16_t* output,int rows,cudaStream_t stream=nullptr);

    // Test-only native-MTP prefill discriminator. K4 remains globally
    // unadmitted for wide production callers; this reuses each decoded tile
    // across sixteen rows while preserving generic_tile's per-output FP32
    // accumulation order and the existing input/output transforms.
    void forward_k4_prefill_for_test(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
        std::uint16_t* output,int rows,cudaStream_t stream=nullptr);

    // Test-only draft-model discriminator for K4 wide prefill. This path is
    // unreachable from production admissions: its represented dequantization,
    // determinism, numerical error and economics must be qualified first.
    void forward_k4_prefill_mma_for_test(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
        std::uint16_t* output,int rows,cudaStream_t stream=nullptr);

    // Test-only wide-prefill discriminator. It retains the rejected row-pair
    // ownership but decodes each staged K6 B fragment once per CTA.
    void forward_k6_rowpair_shared_decode_for_test(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,const std::uint16_t* input,
        std::uint16_t* output,int rows,cudaStream_t stream=nullptr);

    // Admission by an ordinary exact-host context prepared for native M16.
    // Draft head pruning: the H6 single-split head computes only output
    // columns [0, features) (a multiple of 128; 0 = all). Columns beyond keep
    // their previous contents; the row stride stays out_features.
    void set_active_output_features(int features) noexcept { active_output_features_ = features; }
    void set_native_continuation16(bool enabled) noexcept {
        native_continuation_rows_ = enabled ? 16 : 8;
    }

    void set_reconstructed_exact(Exl3ReconstructedExactView view) noexcept {
        reconstructed_exact_ = view;
    }

    // Explicit two-phase form used by layer code when several projections consume
    // the same activation. Private storage survives until this workspace transforms
    // again; borrowed storage survives only until ANY sharing workspace transforms.
    void transform_input(const Exl3CudaLinearWeights& weights,
                         const Exl3CudaLinearMetadata& metadata,
                         const std::uint16_t* input,
                         int rows,
                         cudaStream_t stream = nullptr);

    // Retains the canonical FP16 activation and feeds it directly into Hadamard.
    void transform_gate_up(const Exl3CudaLinearWeights& weights,
                           const Exl3CudaLinearMetadata& metadata,
                           const std::uint16_t* gate,const std::uint16_t* up,
                           std::uint16_t* activation,int rows,
                           cudaStream_t stream = nullptr,
                           Exl3CudaLinearAdmission admission = Exl3CudaLinearAdmission::ordinary);

    void forward_from_transformed(const Exl3CudaLinearWeights& weights,
                                  const Exl3CudaLinearMetadata& metadata,
                                  const std::uint16_t* transformed_input,
                                  std::uint16_t* output,
                                  int rows,
                                  cudaStream_t stream = nullptr,
                                  Exl3CudaLinearAdmission admission =
                                      Exl3CudaLinearAdmission::ordinary,
                                  const std::uint16_t* raw_input = nullptr);

    // The next coherent down/O projection (1..8 rows) also writes
    // residual_out = half(left + projection) in its reduction kernel. The
    // caller checks take_residual_applied() and skips its residual launch.
    void arm_residual(const std::uint16_t* left,std::uint16_t* residual_out) noexcept {
        pending_residual_left_=left; pending_residual_out_=residual_out; residual_applied_=false;
    }
    bool take_residual_applied() noexcept {
        const bool applied=residual_applied_;
        residual_applied_=false; pending_residual_left_=nullptr; pending_residual_out_=nullptr;
        return applied;
    }

    // Two same-input projections of 1..8 rows (e.g. GDN qkv and z) in one
    // fused-input coherent producer launch and one reduction launch; false
    // (nothing submitted) when ineligible. Optional GDN control projections
    // (rows*heads extra CTAs, the control_fused_staged_kernel arithmetic) and
    // a BF16 copy of the first output in the GDN convolution-input layout
    // ([feature*rows+row]) ride on the same launches.
    bool forward_merged_pair(Exl3CudaLinearWorkspace& second_workspace,
                             const Exl3CudaLinearWeights& first_weights,
                             const Exl3CudaLinearMetadata& first_metadata,
                             std::uint16_t* first_output,
                             const Exl3CudaLinearWeights& second_weights,
                             const Exl3CudaLinearMetadata& second_metadata,
                             std::uint16_t* second_output,const std::uint16_t* input,
                             int rows,cudaStream_t stream,
                             const Exl3GdnControlSide* control = nullptr,
                             std::uint16_t* first_bf16 = nullptr);
    // MLP gate/up/SiLU of 1..8 rows in one fused-input coherent producer
    // launch plus one reduction/activation launch, which can also emit the
    // down projection's transformed input. Returns false (nothing submitted)
    // when the pair is not on the fused-input coherent wide route.
    bool forward_merged_gate_up_silu(Exl3CudaLinearWorkspace& up_workspace,
                                     const Exl3CudaLinearWeights& gate_weights,
                                     const Exl3CudaLinearMetadata& gate_metadata,
                                     const Exl3CudaLinearWeights& up_weights,
                                     const Exl3CudaLinearMetadata& up_metadata,
                                     const std::uint16_t* input,std::uint16_t* gate_output,
                                     std::uint16_t* up_output,std::uint16_t* activation,
                                     int rows,cudaStream_t stream,
                                     const Exl3CudaLinearWeights* down_weights = nullptr,
                                     Exl3CudaLinearWorkspace* down_workspace = nullptr);

    // Exact target-prefill fast path for a gate/up pair. Matrix-specific SUH
    // transforms share one launch and remain separate through the paired MMA.
    // Returns false without mutation when the candidate is disabled or either
    // projection falls outside the fixed K6 wide-prefill contract.
    bool forward_target_prefill_gate_up_pair(
        Exl3CudaLinearWorkspace& up_workspace,
        const Exl3CudaLinearWeights& gate_weights,
        const Exl3CudaLinearMetadata& gate_metadata,
        const Exl3CudaLinearWeights& up_weights,
        const Exl3CudaLinearMetadata& up_metadata,
        const std::uint16_t* input,std::uint16_t* gate_output,
        std::uint16_t* up_output,int rows,cudaStream_t stream=nullptr);

    // Separately labeled native same-weight FP16-KV target M=1 gate/up pair.
    // Both projections retain their packed trellises, native FP32 partials and
    // output Hadamard reductions while one cooperative launch owns the two
    // independent projection bodies. The exact fallback remains available.
    void forward_target_m1_gate_up_pair(
        Exl3CudaLinearWorkspace& up_workspace,
        const Exl3CudaLinearWeights& gate_weights,
        const Exl3CudaLinearMetadata& gate_metadata,
        const Exl3CudaLinearWeights& up_weights,
        const Exl3CudaLinearMetadata& up_metadata,
        const std::uint16_t* input,std::uint16_t* gate_output,
        std::uint16_t* up_output,cudaStream_t stream=nullptr);

    // Test-only spelling retained for existing qualification callers.
    void forward_target_m1_gate_up_pair_for_test(
        Exl3CudaLinearWorkspace& up_workspace,
        const Exl3CudaLinearWeights& gate_weights,
        const Exl3CudaLinearMetadata& gate_metadata,
        const Exl3CudaLinearWeights& up_weights,
        const Exl3CudaLinearMetadata& up_metadata,
        const std::uint16_t* input,std::uint16_t* gate_output,
        std::uint16_t* up_output,cudaStream_t stream=nullptr);

    // Test-only discriminator for one target-owned rows=1 K6 K/V pair. The
    // two projections retain the native packed-MMA split/reduction order but
    // share one CTA launch and use independent accumulation planes.
    void forward_target_m1_kv_pair_for_test(
        Exl3CudaLinearWorkspace& second_workspace,
        const Exl3CudaLinearWeights& first_weights,
        const Exl3CudaLinearMetadata& first_metadata,
        const Exl3CudaLinearWeights& second_weights,
        const Exl3CudaLinearMetadata& second_metadata,
        const std::uint16_t* input,std::uint16_t* first_output,
        std::uint16_t* second_output,cudaStream_t stream=nullptr);

    // Test-only discriminator for one target-owned rows=1 K6/K7 K/V pair
    // using one 512-thread CTA per projection in a z=2 launch. Each CTA
    // splits its K range across two 256-thread subgroups and reduces into the
    // existing five FP32 partial planes; the packed weights remain native.
    void forward_target_m1_kv_wide_pair_for_test(
        Exl3CudaLinearWorkspace& second_workspace,
        const Exl3CudaLinearWeights& first_weights,
        const Exl3CudaLinearMetadata& first_metadata,
        const Exl3CudaLinearWeights& second_weights,
        const Exl3CudaLinearMetadata& second_metadata,
        const std::uint16_t* input,std::uint16_t* first_output,
        std::uint16_t* second_output,cudaStream_t stream=nullptr);

    // Diagnostic only: names the kernel path forward_from_transformed() would
    // take for these dimensions/rows. Mirrors the dispatch conditions exactly;
    // no numerical behavior change.
    const char* dispatch_name(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission =
            Exl3CudaLinearAdmission::ordinary) const noexcept;

    // Actual-dispatch telemetry for the separately labeled FP16-KV wide
    // prefill GEMM candidate. These remain zero unless it really launches.
    std::uint64_t fast_wide_prefill_gemm_calls() const noexcept {
        return fast_wide_prefill_gemm_calls_;
    }
    std::uint64_t fast_wide_prefill_gemm_rows() const noexcept {
        return fast_wide_prefill_gemm_rows_;
    }

    bool draft_prefill_kv_candidate(const Exl3CudaLinearMetadata& metadata, int rows) const noexcept {
        return rows <= max_rows_ && in_features_ == 5120 && out_features_ == 1024 &&
            metadata.in_features == in_features_ && metadata.out_features == out_features_ &&
            draft_small_m_candidate(metadata, rows) && generic_capacity_[0] >= 2;
    }
    bool draft_prefill_fc_candidate(const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_rowpair_k6_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows, Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k6_fast_decode_candidate(
        const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k6_rowpair_n64_candidate(
        const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k6_down_rowpair_candidate(
        const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_shape4_n64_candidate(
        const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_rowpair_k7_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k6_gateup_warpgroup_async_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k6_gateup_n32_pair_cta_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k7_tiles64_exact_splits_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k8_kv_prefill_async_a_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_staged_prefill_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_initial16_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_gateup_m16_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_gateup_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_gateup_k6_n16_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_gateup_k5_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k5_small_m_batch_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k6_m1_simt_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_m1_k6_n16_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_fast_same_weights_fp16_m1_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool fast_fp16_m2_8_down_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows,Exl3CudaLinearAdmission admission) const noexcept;
    bool fast_fp16_m2_8_fused_down_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows,Exl3CudaLinearAdmission admission) const noexcept;
    bool coherent_wide_k6_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows,Exl3CudaLinearAdmission admission) const noexcept;
    int coherent_wide_k6_split_count(int rows) const noexcept;
    int coherent_split_override(const char* name, int rows) const;
    bool coherent_down_k6_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows,Exl3CudaLinearAdmission admission) const noexcept;
    bool coherent_down_k7_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows,Exl3CudaLinearAdmission admission) const noexcept;
    bool coherent_o_k7_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows,Exl3CudaLinearAdmission admission) const noexcept;
    bool fast_fp16_m2_8_all_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows,Exl3CudaLinearAdmission admission) const noexcept;
    bool fast_native_persistent_prefill_candidate(
        const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool fast_native_mia_target_prefill_fp16_candidate(
        const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool fast_native_persistent_m1_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool fast_native_mia_m1_fp16_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_fast_same_weights_fp16_m1_n64_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_fast_same_weights_fp16_m1_n64_k5_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_m1_k7_int8_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_m1_int8_down_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_down_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_down_k6_async_a_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k6_small_m_async_a_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool native_k6_critical_path_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_k7_small_m_async_a_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_o_k7_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_kv_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_o_k6_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_z_k6_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_qkv_k6_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool draft_shared_kv_m16_candidate(const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept {
        const auto capacity=draft_k5_async_a_enabled_?draft_k5_async_a_capacity_:generic_capacity_[0];
        return admission==Exl3CudaLinearAdmission::draft_shared_kv_m16 && draft_small_m_enabled_ &&
            rows==16 && rows<=max_rows_ && metadata.K==5 && metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
            in_features_==5120 && out_features_==1024 && metadata.in_features==in_features_ &&
            metadata.out_features==out_features_ && capacity>=2 && generic_capacity_[0]>=2;
    }
    bool draft_shared_m16_candidate(const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept {
        return draft_shared_q_m16_candidate(metadata,rows,admission) ||
            draft_shared_kv_m16_candidate(metadata,rows,admission) ||
            draft_shared_o_m16_candidate(metadata,rows,admission) ||
            draft_shared_down_m16_candidate(metadata,rows,admission) ||
            draft_shared_gateup_m16_candidate(metadata,rows,admission);
    }
    bool draft_shared_gateup_m16_candidate(const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept {
        // Independent draft MLP uses generic_tile, not the draft MMA route.
        return admission==Exl3CudaLinearAdmission::draft_shared_gateup_m16 && draft_small_m_enabled_ &&
            rows==16 && rows<=max_rows_ && metadata.K==5 && metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
            in_features_==5120 && out_features_==17408 && metadata.in_features==in_features_ &&
            metadata.out_features==out_features_;
    }
    bool draft_shared_down_m16_candidate(const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept {
        const auto capacity=draft_k5_async_a_enabled_?draft_k5_async_a_capacity_:generic_capacity_[0];
        return admission==Exl3CudaLinearAdmission::draft_shared_down_m16 && draft_small_m_enabled_ &&
            rows==16 && rows<=max_rows_ && metadata.K==5 && metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
            in_features_==17408 && out_features_==5120 && metadata.in_features==in_features_ &&
            metadata.out_features==out_features_ && capacity>=10 && generic_capacity_[0]>=10;
    }
    bool draft_shared_o_m16_candidate(const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept {
        const auto capacity=draft_k5_async_a_enabled_?draft_k5_async_a_capacity_:generic_capacity_[0];
        return admission==Exl3CudaLinearAdmission::draft_shared_o_m16 && draft_small_m_enabled_ &&
            rows==16 && rows<=max_rows_ && metadata.K==5 && metadata.mul1 && !metadata.mcg && !metadata.has_bias &&
            in_features_==4096 && out_features_==5120 && metadata.in_features==in_features_ &&
            metadata.out_features==out_features_ && capacity>=10 && generic_capacity_[0]>=10;
    }
    bool target_head_small_m_candidate(const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept {
        return admission==Exl3CudaLinearAdmission::target_continuation_head && h6_small_m_enabled_ &&
            rows>=2 && rows<=native_continuation_rows_ && rows<=max_rows_ &&
            in_features_==5120 && out_features_==248320 && metadata.in_features==in_features_ &&
            metadata.out_features==out_features_ && metadata.K==6 && metadata.mul1 &&
            !metadata.mcg && !metadata.has_bias;
    }
    bool target_q_k6_small_m_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool draft_shared_q_m16_candidate(const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool target_wide_prefill_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool fast_wide_prefill_gemm_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool native_mtp_wide_prefill_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool native_mtp_one_step_candidate(
        const Exl3CudaLinearMetadata& metadata, int rows,
        Exl3CudaLinearAdmission admission) const noexcept;

    const std::uint16_t* transformed_device() const noexcept { return transformed_; }

    bool draft_small_m_candidate_for_test(
        const Exl3CudaLinearMetadata& metadata,int rows) const noexcept {
        return draft_small_m_candidate(metadata,rows);
    }

    int max_rows() const noexcept { return max_rows_; }
    std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
    std::uint64_t stream_reduction_calls(bool extended=false) const noexcept { return extended?extended_stream_reduction_calls_:stream_reduction_calls_; }
    std::uint64_t k6_gateup_warpgroup_async_calls() const noexcept {
        return k6_gateup_warpgroup_async_calls_;
    }
    std::uint64_t k6_gateup_n32_pair_cta_calls() const noexcept {
        return k6_gateup_n32_pair_cta_calls_;
    }
    std::uint64_t k6_gateup_n32_pair_cta_rows() const noexcept {
        return k6_gateup_n32_pair_cta_rows_;
    }
    std::uint64_t k6_fast_decode_calls() const noexcept {
        return k6_fast_decode_calls_;
    }
    std::uint64_t k6_fast_decode_rows() const noexcept {
        return k6_fast_decode_rows_;
    }
    std::uint64_t k6_rowpair_n64_calls() const noexcept {
        return k6_rowpair_n64_calls_;
    }
    std::uint64_t k6_rowpair_n64_rows() const noexcept {
        return k6_rowpair_n64_rows_;
    }
    std::uint64_t k6_down_rowpair_calls() const noexcept {
        return k6_down_rowpair_calls_;
    }
    std::uint64_t k6_down_rowpair_rows() const noexcept {
        return k6_down_rowpair_rows_;
    }
    std::uint64_t shape4_n64_calls() const noexcept {return shape4_n64_calls_;}
    std::uint64_t shape4_n64_rows() const noexcept {return shape4_n64_rows_;}
    std::uint64_t reduce_shfl_min_barrier_calls() const noexcept {
        return reduce_shfl_min_barrier_calls_;
    }
    std::uint64_t reduce_shfl_min_barrier_rows() const noexcept {
        return reduce_shfl_min_barrier_rows_;
    }
    std::uint64_t k7_tiles64_exact_splits_calls() const noexcept {
        return k7_tiles64_exact_splits_calls_;
    }
    std::uint64_t target_k8_kv_prefill_async_a_calls() const noexcept {
        return target_k8_kv_prefill_async_a_calls_;
    }
    std::uint64_t target_k8_kv_prefill_async_a_rows() const noexcept {
        return target_k8_kv_prefill_async_a_rows_;
    }
    std::uint64_t target_down_k6_async_a_calls() const noexcept {
        return target_down_k6_async_a_calls_;
    }
    std::uint64_t target_gateup_k6_n16_calls() const noexcept {
        return target_gateup_k6_n16_calls_;
    }
    std::uint64_t target_k6_small_m_async_a_calls() const noexcept {
        return target_k6_small_m_async_a_calls_;
    }
    std::uint64_t target_k7_small_m_async_a_calls() const noexcept {
        return target_k7_small_m_async_a_calls_;
    }
    std::uint64_t target_k5_small_m_batch_calls() const noexcept {
        return target_k5_small_m_batch_calls_;
    }
    std::uint64_t target_k6_m1_simt_calls() const noexcept {
        return target_k6_m1_simt_calls_;
    }
    // Separately labeled default-off R49 target-only native K6 critical path.
    // This counts actual packed/MMA kernel dispatches, not eligibility checks.
    std::uint64_t native_k6_critical_path_calls() const noexcept {
        return native_k6_critical_path_calls_;
    }
    // Separately labeled default-off K6 register-fragment pipeline
    // differential.  This counts actual candidate kernel dispatches.
    std::uint64_t native_k6_register_pipeline_calls() const noexcept {
        return native_k6_register_pipeline_calls_;
    }
    std::uint64_t target_prefill_gate_up_pair_attempts() const noexcept {
        return target_prefill_gate_up_pair_attempts_;
    }
    std::uint64_t target_prefill_gate_up_pair_calls() const noexcept {
        return target_prefill_gate_up_pair_calls_;
    }
    std::uint64_t target_prefill_gate_up_pair_rows() const noexcept {
        return target_prefill_gate_up_pair_rows_;
    }
    const char* target_prefill_gate_up_pair_route() const noexcept {
        return target_prefill_gate_up_pair_
            ? "target_prefill_k6_gate_up_pair_shared_a"
            : "separate_gate_up";
    }
    std::uint64_t prefill_projection_graph_captures() const noexcept {
        return prefill_projection_graph_captures_;
    }
    std::uint64_t prefill_projection_graph_replays() const noexcept {
        return prefill_projection_graph_replays_;
    }
    std::uint64_t prefill_projection_graph_binding_fallbacks() const noexcept {
        return prefill_projection_graph_binding_fallbacks_;
    }
    struct PrefillPersistingL2Snapshot {
        std::uint64_t eligible=0,applied=0,unsupported=0;
        std::size_t window_bytes=0,set_aside_bytes=0;
    };
    PrefillPersistingL2Snapshot prefill_persisting_l2_snapshot() const noexcept {
        return {prefill_persisting_l2_eligible_,prefill_persisting_l2_applied_,
            prefill_persisting_l2_unsupported_,prefill_persisting_l2_window_bytes_,
            prefill_persisting_l2_set_aside_bytes_};
    }
    bool target_prefill_persisting_l2_candidate(
        const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;

private:
    struct StorageRetirement {
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        std::optional<RetainedDeviceLedger::Ticket> transform_credit,accumulation_credit;
        void* transformed=nullptr;
        void* accumulation=nullptr;
        std::size_t transformed_bytes=0,accumulation_bytes=0;
        int device=-1,error=0,restore_error=0;
        StorageRetirement* next=nullptr;
    };
    std::unique_ptr<StorageRetirement> storage_retirement_;
    std::optional<RetainedDescriptorLedger::Ticket> constructor_object_credit_;
    inline static std::atomic<StorageRetirement*> storage_quarantine_{nullptr};
    inline static std::atomic<std::uint64_t> process_target_m1_k6_n16_calls_{0};
    inline static std::atomic<std::uint64_t> process_target_m1_k7_three_word_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_fp16_accum_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_fp16_m1_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_fp16_m2_8_down_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_fp16_m2_8_fused_down_calls_{0};
    inline static std::array<std::atomic<std::uint64_t>,5> coherent_wide_k6_calls_{};
    inline static std::array<std::atomic<std::uint64_t>,5> coherent_wide_k6_rows_{};
    inline static std::array<std::atomic<std::uint64_t>,5> coherent_wide_k6_split10_calls_{};
    inline static std::atomic<std::uint64_t> process_coherent_down_k6_calls_{0};
    inline static std::atomic<std::uint64_t> process_coherent_down_k6_rows_{0};
    inline static std::atomic<std::uint64_t> process_coherent_down_k7_calls_{0};
    inline static std::atomic<std::uint64_t> process_coherent_down_k7_rows_{0};
    inline static std::atomic<std::uint64_t> process_coherent_o_k7_calls_{0};
    inline static std::atomic<std::uint64_t> process_coherent_o_k7_rows_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_fp16_m1_wide_n32_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_fp16_m1_n16_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_native_persistent_m1_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_native_mia_m1_fp16_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_native_mia_target_prefill_fp16_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_fp16kv_m1_gate_up_pair_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_fp16kv_m1_kv_pair_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_fp16_m1_n64_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_fp16_m1_n64_k5_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_int8_gemv_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_int8_gemv_k7_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_int8_gemv_down_k6_calls_{0};
    inline static std::atomic<std::uint64_t> process_fast_same_weights_int8_gemv_down_k7_calls_{0};
    inline static thread_local unsigned constructor_failure_for_test_=0;
    void retire_storage_fallback() noexcept;
    bool k6_stream_reduction_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows, Exl3CudaLinearAdmission admission) const noexcept;
    bool k6_stream_reduction_enabled_=false;
    std::uint64_t stream_reduction_calls_=0;
    bool extended_stream_reduction_enabled_=false;
    // Only the exact-HostKV compute-only GDN segment graph admits these two
    // fixed ordinary launches during capture. Other graph routes retain their
    // existing fail-closed disposition.
    bool host_kv_gdn_segment_graph_capture_=false;
    bool kv_k7_stream_reduction_enabled_=false;
    std::uint64_t extended_stream_reduction_calls_=0;
    bool extended_stream_reduction_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows, Exl3CudaLinearAdmission admission) const noexcept;
    bool reconstructed_exact_candidate(const Exl3CudaLinearMetadata& metadata,
        int rows, Exl3CudaLinearAdmission admission) const noexcept;
    bool prefill_projection_graph_candidate(
        const Exl3CudaLinearMetadata& metadata,int rows,
        Exl3CudaLinearAdmission admission) const noexcept;
    bool apply_prefill_persisting_l2(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,cudaStream_t stream);
    void forward_reconstructed_exact_from_transformed(
        const Exl3CudaLinearWeights& weights, const Exl3CudaLinearMetadata& metadata,
        const std::uint16_t* transformed_input, std::uint16_t* output, int rows,
        std::uint16_t* decoded, std::size_t decoded_bytes, cudaStream_t stream);
    bool try_fast_wide_prefill_gemm_from_transformed(
        const Exl3CudaLinearWeights& weights,
        const Exl3CudaLinearMetadata& metadata,
        const std::uint16_t* transformed_input,
        std::uint16_t* output,
        int rows,
        cudaStream_t stream);
    struct FastWideGemmState;
    std::unique_ptr<FastWideGemmState> fast_wide_gemm_state_;
    inline static std::atomic<Exl3CudaLinearWorkspace*> quarantine_{nullptr};
    inline static std::atomic<std::size_t> quarantine_count_{0};
    Exl3CudaLinearWorkspace* quarantine_next_=nullptr;
    int retirement_error_=0;
    int retirement_restore_error_=0;
    int owner_device_=-1;
    unsigned retirement_failure_stage_for_test_=0;
    int native_continuation_rows_ = 8;
    Exl3ReconstructedExactView reconstructed_exact_{};
    std::size_t accumulation_capacity_bytes_ = 0;
    std::size_t transformed_capacity_bytes_ = 0;
    bool draft_small_m_candidate(const Exl3CudaLinearMetadata& metadata,
                                 int rows) const noexcept;
    int in_features_ = 0;
    int out_features_ = 0;
    int max_rows_ = 0;
    int cooperative_grid_ = 0;
    int generic_capacity_[3] = {};
    int target_gateup_k6_async_a_capacity_ = 0;
    int draft_k5_async_a_capacity_ = 0;
    int target_down_k6_async_a_capacity_ = 0;
    int target_down_k7_async_a_capacity_ = 0;
    int target_m1_k6_n32_async_a_capacity_ = 0;
    int target_m1_k7_n32_async_a_capacity_ = 0;
    int target_k6_small_m_async_a_capacity_ = 0;
    int target_k7_small_m_async_a_capacity_ = 0;
    int large_down_candidate_capacity_[2] = {};
    int large_down_candidate8_capacity_[2] = {};
    int fast_same_weights_fp16_m1_capacity_[3] = {};
    int coherent_wide_k6_operation_ = -1;
    int coherent_wide_k6_resident_capacity_ = 0;
    int coherent_wide_k6_registers_per_thread_ = 0;
    int coherent_down_k6_resident_capacity_ = 0;
    int coherent_down_k7_resident_capacity_ = 0;
    int coherent_o_k7_resident_capacity_ = 0;
    int fast_same_weights_fp16_m1_n16_capacity_[3] = {};
    int fast_same_weights_fp16_m1_n64_capacity_[3] = {};
    int fast_native_persistent_m1_grid_[2] = {};
    int fast_native_mia_m1_fp16_fused_input_grid_[2] = {};
    int fast_native_persistent_prefill_capacity_[3] = {};
    int fast_native_mia_target_prefill_fp16_capacity_ = 0;
    bool specialized_shape_ = false;
    bool allow_generic_variants_ = true;
    bool h6_small_m_enabled_ = true;
    int active_output_features_ = 0;
    bool draft_small_m_enabled_ = false;
    bool target_gateup_small_m_enabled_ = false;
    bool target_gateup_k6_async_a_enabled_ = false;
    bool target_gateup_k6_n16_enabled_ = false;
    bool draft_k5_async_a_enabled_ = false;
    bool target_gateup_k7_small_m_enabled_ = false;
    bool target_gateup_k5_small_m_enabled_ = false;
    bool target_k5_small_m_batch_enabled_ = false;
    bool target_k6_m1_simt_enabled_ = false;
    bool fast_same_weights_fp16_accum_enabled_ = false;
    bool fast_same_weights_fp16_m1_enabled_ = false;
    bool fast_fp16_m2_8_down_enabled_ = false;
    bool fast_fp16_m2_8_fused_down_enabled_ = false;
    bool coherent_wide_k6_enabled_ = false;
    bool coherent_wide_k6_split10_enabled_ = false;
    bool coherent_down_k6_enabled_ = false;
    bool coherent_down_k7_enabled_ = false;
    bool coherent_o_k7_enabled_ = false;
    bool fast_fp16_m2_8_all_enabled_ = false;
    bool fast_fp16_m2_8_async_a_enabled_ = false;
    bool fast_same_weights_fp16_m1_n16_enabled_ = false;
    bool fast_native_persistent_m1_enabled_ = false;
    bool fast_native_mia_m1_fp16_enabled_ = false;
    bool fast_native_mia_m1_fp16_fused_input_enabled_ = false;
    bool fast_native_mia_target_prefill_fp16_enabled_ = false;
    int fast_native_persistent_m1_only_ = 0;
    bool fast_native_persistent_prefill_enabled_ = false;
    bool fast_same_weights_fp16kv_m1_mgemm_pair_enabled_ = false;
    int fast_same_weights_fp16kv_m1_mgemm_pair_grid_[3] = {};
    bool fast_same_weights_fp16kv_m1_mgemm_policy_enabled_ = false;
    int fast_same_weights_fp16kv_m1_mgemm_policy_grid_[2] = {};
    bool fast_same_weights_fp16_m1_wide_enabled_ = false;
    bool fast_same_weights_fp16_m1_wide_n32_enabled_ = false;
    bool fast_same_weights_fp16_m1_wide_k5_enabled_ = false;
    bool fast_same_weights_fp16_m1_n64_enabled_ = false;
    bool fast_same_weights_fp16_m1_n64_k5_enabled_ = false;
    bool fast_same_weights_fp16_m1_k6_register_pipeline_enabled_ = false;
    int fast_same_weights_fp16_m1_k6_register_pipeline_capacity_ = 0;
    bool fast_same_weights_int8_gemv_enabled_ = false;
    bool fast_same_weights_int8_gemv_k7_enabled_ = false;
    bool fast_same_weights_int8_gemv_down_k6_enabled_ = false;
    bool fast_same_weights_int8_gemv_down_k7_enabled_ = false;
    // Separately labeled dispatch differential mirroring the pinned Mia
    // Blackwell policy: only wide K6 M=1 projections may use the INT8
    // activation lane; narrow K6 and every K7 projection stay on the native
    // packed MGEMM path.  Default-off so the established route is unchanged.
    bool fast_same_weights_int8_mia_policy_enabled_ = false;
    bool fast_same_weights_int8_gemv_occupancy_grid_enabled_ = false;
    bool target_m1_k6_n16_enabled_ = false;
    bool target_m1_k7_three_word_enabled_ = false;
    bool target_gateup_m16_enabled_ = false;
    bool target_down_small_m_enabled_ = false;
    bool target_down_k7_small_m_enabled_ = false;
    bool target_down_k6_async_a_enabled_ = false;
    bool target_down_k7_async_a_enabled_ = false;
    bool target_m1_k6_n32_async_a_enabled_ = false;
    bool target_m1_k7_n32_async_a_enabled_ = false;
    bool target_k6_small_m_async_a_enabled_ = false;
    bool target_k7_small_m_async_a_enabled_ = false;
    bool native_k6_critical_path_enabled_ = false;
    int native_k6_critical_path_capacity_ = 0;
    bool native_k6_global_slices_enabled_ = false;
    int native_k6_global_slices_capacity_ = 0;
    bool native_k6_register_pipeline_enabled_ = false;
    int native_k6_register_pipeline_capacity_ = 0;
    // Separate shape-4 selector for the Mia-shaped four-stage/three-fragment
    // K6 differential.  This remains independent from the broader register
    // pipeline opt-in so the existing route and its receipts are preserved.
    bool native_k6_shape4_register_pipeline_enabled_ = false;
    // Opt-in N16 form of the same-weight FastK6/register-fragment
    // differential.  Keep its launch capacity separate: the smaller output
    // tile changes both shared storage and register occupancy.
    bool native_k6_register_pipeline_n16_enabled_ = false;
    int native_k6_register_pipeline_n16_capacity_ = 0;
    bool target_o_k7_small_m_enabled_ = false;
    bool target_kv_small_m_enabled_ = false;
    bool target_o_k6_small_m_enabled_ = false;
    bool target_z_k6_small_m_enabled_ = false;
    bool target_qkv_k6_small_m_enabled_ = false;
    bool target_q_k6_small_m_enabled_ = false;
    bool target_wide_prefill_enabled_ = false;
    bool fast_wide_prefill_gemm_enabled_ = false;
    std::uint64_t fast_wide_prefill_gemm_calls_ = 0;
    std::uint64_t fast_wide_prefill_gemm_rows_ = 0;
    bool target_initial16_enabled_ = false;
    bool target_direct_async_a_ = false;
    bool target_direct_async_all_ = false;
    bool target_direct_tiles64_ = false;
    bool target_k6_fast_decode_ = false;
    std::uint64_t k6_fast_decode_calls_ = 0;
    std::uint64_t k6_fast_decode_rows_ = 0;
    bool target_k6_rowpair_n64_ = false;
    std::uint64_t k6_rowpair_n64_calls_ = 0;
    std::uint64_t k6_rowpair_n64_rows_ = 0;
    bool target_k6_down_rowpair_ = false;
    std::uint64_t k6_down_rowpair_calls_ = 0;
    std::uint64_t k6_down_rowpair_rows_ = 0;
    bool target_shape4_n64_ = false;
    std::uint64_t shape4_n64_calls_ = 0;
    std::uint64_t shape4_n64_rows_ = 0;
    bool target_k6_gateup_warpgroup_async_ = false;
    std::uint64_t k6_gateup_warpgroup_async_calls_ = 0;
    bool target_k6_gateup_n32_pair_cta_ = false;
    std::uint64_t k6_gateup_n32_pair_cta_calls_ = 0;
    std::uint64_t k6_gateup_n32_pair_cta_rows_ = 0;
    bool target_prefill_gate_up_pair_ = false;
    std::uint64_t target_prefill_gate_up_pair_attempts_ = 0;
    std::uint64_t target_prefill_gate_up_pair_calls_ = 0;
    std::uint64_t target_prefill_gate_up_pair_rows_ = 0;
    bool target_k7_tiles64_exact_splits_ = false;
    std::uint64_t k7_tiles64_exact_splits_calls_ = 0;
    bool target_k8_kv_prefill_async_a_ = false;
    std::uint64_t target_k8_kv_prefill_async_a_calls_ = 0;
    std::uint64_t target_k8_kv_prefill_async_a_rows_ = 0;
    std::uint64_t target_down_k6_async_a_calls_ = 0;
    std::uint64_t target_gateup_k6_n16_calls_ = 0;
    std::uint64_t target_k6_small_m_async_a_calls_ = 0;
    std::uint64_t target_k7_small_m_async_a_calls_ = 0;
    std::uint64_t target_k5_small_m_batch_calls_ = 0;
    std::uint64_t target_k6_m1_simt_calls_ = 0;
    std::uint64_t native_k6_critical_path_calls_ = 0;
    std::uint64_t native_k6_register_pipeline_calls_ = 0;
    bool target_reduce_shfl_ = false;
    bool target_reduce_shfl_min_barriers_ = false;
    std::uint64_t reduce_shfl_min_barrier_calls_ = 0;
    std::uint64_t reduce_shfl_min_barrier_rows_ = 0;
    bool target_rowpair_k6_ = false;
    bool target_rowpair_k7_ = false;
    bool target_direct_partials_ = false;
    bool target_staged_shape4_ = false;
    bool target_staged_prefill_enabled_ = false;
    bool target_wide64_enabled_ = false;
    bool target_wide128_enabled_ = false;
    bool target_wide256_enabled_ = false;
    bool target_wide512_enabled_ = false;
    bool target_wide1024_enabled_ = false;
    struct PrefillProjectionGraphBinding {
        Exl3CudaLinearWeights weights{};
        Exl3CudaLinearMetadata metadata{};
        const std::uint16_t* input = nullptr;
        std::uint16_t* output = nullptr;
        cudaStream_t stream = nullptr;
        int rows = 0;
        Exl3CudaLinearAdmission admission = Exl3CudaLinearAdmission::ordinary;
    };
    bool prefill_projection_graph_enabled_ = false;
    ninfer::DecodeGraphDefinition prefill_projection_graph_definition_{};
    ninfer::DecodeGraphExecutable prefill_projection_graph_executable_{};
    std::optional<PrefillProjectionGraphBinding> prefill_projection_graph_binding_;
    std::uint64_t prefill_projection_graph_captures_ = 0;
    std::uint64_t prefill_projection_graph_replays_ = 0;
    std::uint64_t prefill_projection_graph_binding_fallbacks_ = 0;
    bool prefill_persisting_l2_enabled_ = false;
    std::uint64_t prefill_persisting_l2_eligible_ = 0;
    std::uint64_t prefill_persisting_l2_applied_ = 0;
    std::uint64_t prefill_persisting_l2_unsupported_ = 0;
    std::size_t prefill_persisting_l2_window_bytes_ = 0;
    std::size_t prefill_persisting_l2_set_aside_bytes_ = 0;
    bool draft_prefill_fc_enabled_ = false;
    std::uint16_t* transformed_ = nullptr;
    float* accum_ = nullptr;
    const std::uint16_t* pending_residual_left_ = nullptr;
    std::uint16_t* pending_residual_out_ = nullptr;
    bool residual_applied_ = false;
    bool accum_owned_ = true;
    bool transformed_owned_ = true;
    std::size_t workspace_bytes_ = 0;
};

// Context-owned resources for bounded eager overlap of the independent MLP
// gate and up projections.  The view borrows one private up-projection
// workspace and one auxiliary stream; the owning text context keeps every
// resource alive until all layer objects have retired.
struct Exl3MlpGateUpConcurrencyView {
    cudaStream_t up_stream = nullptr;
    cudaEvent_t fork = nullptr;
    cudaEvent_t up_done = nullptr;
    Exl3CudaLinearWorkspace* up_workspace = nullptr;

    bool complete() const noexcept {
        return up_stream && fork && up_done && up_workspace;
    }
};

} // namespace ninfer::exl3
