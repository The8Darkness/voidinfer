#pragma once
#include "exl3/gaming_optimizations.h"
#include "exl3/sibling_rows.h"

#include "exl3/linear_cuda.h"
#include "exl3/layer_buffer_retirement.h"
#include "exl3/target_projection_timing.h"
#include "exl3/target_projection_observer.h"
#include "exl3/target_q_continuation.h"
#include "exl3/bounded_shared_owner.h"
#include "exl3/prefill_projection_chain_graph.h"
#include "core/arena.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace ninfer::exl3 {

class Exl3GdnLayer;
class Exl3TextContext;

// Canonical layout of one actual EXL3 GDN layer. Recurrent values are FP32 and
// ordered [value_head][key][value]. Convolution owns four physical BF16 slots
// per channel while the public history trace exposes the oldest three. Context
// position is the exclusive token frontier represented by both planes.
struct Exl3GdnRecurrentLayout {
    static constexpr std::size_t model_layers=64;
    static constexpr std::size_t value_heads=48;
    static constexpr std::size_t key_heads=16;
    static constexpr std::size_t key_columns=128;
    static constexpr std::size_t value_columns=128;
    static constexpr std::size_t convolution_channels=10240;
    static constexpr std::size_t convolution_history=3;
    static constexpr std::size_t convolution_physical_slots=4;
    static constexpr std::size_t recurrent_elements=
        value_heads*key_columns*value_columns;
    static constexpr std::size_t recurrent_bytes=recurrent_elements*sizeof(float);
    static constexpr std::size_t convolution_history_elements=
        convolution_channels*convolution_history;
    static constexpr std::size_t convolution_storage_elements=
        convolution_channels*convolution_physical_slots;
    static constexpr std::size_t convolution_history_bytes=
        convolution_history_elements*sizeof(std::uint16_t);
    static constexpr std::size_t convolution_storage_bytes=
        convolution_storage_elements*sizeof(std::uint16_t);

    static bool valid_layer(int model_layer) noexcept {
        return model_layer>=0 &&
            static_cast<std::size_t>(model_layer)<model_layers &&
            model_layer%4!=3;
    }
};

// Host-side launch contract for the optional paired-column load/store path.
// It changes only memory transactions; each column retains the scalar update
// and reduction order. Unsafe views must use the one-column resident kernel.
struct Exl3GdnRecurrentVectorAccess {
    const float* state = nullptr;
    std::size_t state_capacity_bytes = 0;
    const std::uint16_t* values = nullptr;
    std::size_t values_capacity_bytes = 0;
    std::uint16_t* output = nullptr;
    std::size_t output_capacity_bytes = 0;
    std::size_t rows = 0;
    std::size_t active_value_columns = Exl3GdnRecurrentLayout::value_columns;

    bool pair_columns_supported() const noexcept {
        if(!state || !values || !output || !rows ||
           active_value_columns!=Exl3GdnRecurrentLayout::value_columns ||
           reinterpret_cast<std::uintptr_t>(state)%alignof(float2)!=0 ||
           reinterpret_cast<std::uintptr_t>(values)%alignof(std::uint32_t)!=0 ||
           reinterpret_cast<std::uintptr_t>(output)%alignof(std::uint32_t)!=0)
            return false;
        constexpr auto maximum=std::numeric_limits<std::size_t>::max();
        if(rows>maximum/Exl3GdnRecurrentLayout::value_heads ||
           rows*Exl3GdnRecurrentLayout::value_heads>
               maximum/active_value_columns)
            return false;
        const auto row_elements=rows*Exl3GdnRecurrentLayout::value_heads*
            active_value_columns;
        if(row_elements>maximum/sizeof(std::uint16_t))return false;
        const auto row_bytes=row_elements*sizeof(std::uint16_t);
        if(state_capacity_bytes<Exl3GdnRecurrentLayout::recurrent_bytes ||
           values_capacity_bytes<row_bytes || output_capacity_bytes<row_bytes)
            return false;
        const auto disjoint=[](const void* left,std::size_t left_bytes,
                               const void* right,std::size_t right_bytes) noexcept {
            const auto l=reinterpret_cast<std::uintptr_t>(left);
            const auto r=reinterpret_cast<std::uintptr_t>(right);
            if(left_bytes>std::numeric_limits<std::uintptr_t>::max()-l ||
               right_bytes>std::numeric_limits<std::uintptr_t>::max()-r)
                return false;
            return l+left_bytes<=r || r+right_bytes<=l;
        };
        return disjoint(state,Exl3GdnRecurrentLayout::recurrent_bytes,
                        values,row_bytes) &&
            disjoint(state,Exl3GdnRecurrentLayout::recurrent_bytes,
                     output,row_bytes) &&
            disjoint(values,row_bytes,output,row_bytes);
    }
};

// The resident prefill route materializes normalized Q/K and exp(g) once, at
// the same FP32 boundaries used inside the canonical recurrence kernel, then
// consumes those represented values in the unchanged row-chronological
// recurrence. It is an exact topology only for an explicitly wide,
// preserve-M1 call with resident storage. Every other call uses the canonical
// one-kernel path.
struct Exl3GdnStageFusionContract {
    std::size_t rows = 0;
    bool resident_storage = false;
    bool wide_prefill = false;
    bool preserve_m1_topology = false;

    bool exact_route_supported() const noexcept {
        return rows>1 && rows<=1024 && resident_storage && wide_prefill &&
            preserve_m1_topology;
    }
};

// Test-only operator view for comparing the canonical recurrent stage with the
// materialized-normalization/resident-update topology. The caller initializes
// both state planes identically and owns every range through stream completion.
// This entry point performs no allocation and is never called by Engine code.
struct Exl3GdnStageFusionFixtureView {
    const std::uint16_t* q = nullptr;
    const std::uint16_t* k = nullptr;
    const std::uint16_t* v = nullptr;
    const float* g = nullptr;
    const float* beta = nullptr;
    float* canonical_state = nullptr;
    float* fused_state = nullptr;
    std::uint16_t* canonical_output = nullptr;
    std::uint16_t* fused_output = nullptr;
    float* normalized_q = nullptr;
    float* normalized_k = nullptr;
    float* alpha = nullptr;
    std::size_t qk_capacity_bytes = 0;
    std::size_t value_capacity_bytes = 0;
    std::size_t control_capacity_bytes = 0;
    std::size_t state_capacity_bytes = 0;
    std::size_t normalized_capacity_bytes = 0;
    int rows = 0;

    bool valid() const noexcept {
        if(rows<=1 || rows>1024 || !q || !k || !v || !g || !beta ||
           !canonical_state || !fused_state || canonical_state==fused_state ||
           !canonical_output || !fused_output ||
           canonical_output==fused_output || !normalized_q || !normalized_k ||
           normalized_q==normalized_k || !alpha)
            return false;
        const auto active_rows=static_cast<std::size_t>(rows);
        const auto qk_bytes=active_rows*Exl3GdnRecurrentLayout::key_heads*
            Exl3GdnRecurrentLayout::key_columns*sizeof(std::uint16_t);
        const auto value_bytes=active_rows*Exl3GdnRecurrentLayout::value_heads*
            Exl3GdnRecurrentLayout::value_columns*sizeof(std::uint16_t);
        const auto control_bytes=active_rows*Exl3GdnRecurrentLayout::value_heads*
            sizeof(float);
        const auto normalized_bytes=active_rows*
            Exl3GdnRecurrentLayout::key_heads*
            Exl3GdnRecurrentLayout::key_columns*sizeof(float);
        return qk_capacity_bytes>=qk_bytes &&
            value_capacity_bytes>=value_bytes &&
            control_capacity_bytes>=control_bytes &&
            state_capacity_bytes>=Exl3GdnRecurrentLayout::recurrent_bytes &&
            normalized_capacity_bytes>=normalized_bytes;
    }
};

void exl3_gdn_stage_fusion_fixture(
    const Exl3GdnStageFusionFixtureView& view,
    cudaStream_t stream=nullptr,bool pair_columns=false);
// Verifier-sized (1..8 rows) recurrence route of the decode/verify forward:
// q,k BF16 [rows][16][128] (unnormalized), v BF16 [rows][48][128], g (log
// decay) and beta FP32 [rows][48], state FP32 [48][128][128] updated in place,
// output BF16 [rows][48][128]. chain_rows (device int, may be null) selects
// the sibling-leaf layout of sibling_rows.cuh.
void exl3_gdn_verifier_recurrence_fixture(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,const float* g,const float* beta,float* state,
    std::uint16_t* output,int rows,const int* chain_rows,cudaStream_t stream=nullptr);

// Owning read-only slice of the most recent supported M1-topology recurrent
// history. `storage_owner` must keep the context/layer allocations alive;
// `model_owner` independently binds coefficient identity. Any later layer
// mutation changes the generation authority, so a cached view fails closed.
struct Exl3GdnContinuationHistoryView {
    std::shared_ptr<const void> storage_owner;
    std::shared_ptr<const void> model_owner;
    const std::uint64_t* generation_authority = nullptr;
    std::uint64_t generation = 0;
    int model_layer = -1;
    int base_position = -1;
    int source_rows = 0;
    int first_row = 0;
    int rows = 0;
    const std::uint16_t* conv_input = nullptr;
    const std::uint16_t* q = nullptr;
    const std::uint16_t* k = nullptr;
    const std::uint16_t* v = nullptr;
    const float* g = nullptr;
    const float* beta = nullptr;
    const std::uint16_t* recurrent_output = nullptr;

    bool current() const noexcept {
        return storage_owner && model_owner && generation_authority &&
            generation!=0 && *generation_authority==generation &&
            Exl3GdnRecurrentLayout::valid_layer(model_layer) &&
            base_position>=0 && source_rows>0 && first_row>=0 && rows>0 &&
            first_row<=source_rows && rows<=source_rows-first_row &&
            conv_input && q && k && v && g && beta && recurrent_output;
    }
};

enum class Exl3GdnHistoryStorage : std::uint8_t {
    refused,
    private_retained,
    shared_wide,
};

// Shared transient scratch is stream-ordered across layers. The five planes
// needed to reconstruct rejected B2/B4/B8 work remain private; the shared wide
// slab is admitted only when no graph/rejection history can survive the call.
struct Exl3GdnScratchReuseContract {
    static constexpr std::size_t private_history_rows=16;
    static constexpr std::size_t wide_features_per_row=26624;
    std::size_t row_capacity = 0;
    std::size_t private_row_capacity = 0;
    std::size_t shared_transient_bytes = 0;
    std::size_t shared_wide_bytes = 0;

    static bool verifier_rows_supported(std::size_t rows) noexcept {
        return (rows>=1 && rows<=8) || rows==16;
    }
    static bool retained_rows_supported(std::size_t rows) noexcept {
        return rows>=2 && rows<=8;
    }
    static bool policy_horizon(std::size_t rows) noexcept {
        return rows==2 || rows==4 || rows==8;
    }
    static std::size_t wide_bytes_required(std::size_t rows) noexcept {
        constexpr auto per_row=wide_features_per_row*sizeof(std::uint16_t);
        return rows>std::numeric_limits<std::size_t>::max()/per_row
            ? std::numeric_limits<std::size_t>::max() : rows*per_row;
    }
    bool valid() const noexcept {
        return row_capacity>0 && private_row_capacity>0 &&
            private_row_capacity<=row_capacity &&
            (private_row_capacity==row_capacity ||
             (private_row_capacity==private_history_rows &&
              shared_wide_bytes>=wide_bytes_required(row_capacity)));
    }
    Exl3GdnHistoryStorage history_storage(
        std::size_t rows,bool retain_for_rejection,bool graph_consumer) const noexcept {
        if(!valid() || !rows || rows>row_capacity ||
           (retain_for_rejection && !retained_rows_supported(rows)) ||
           (graph_consumer && !verifier_rows_supported(rows)))
            return Exl3GdnHistoryStorage::refused;
        if(rows<=private_row_capacity)
            return Exl3GdnHistoryStorage::private_retained;
        if(retain_for_rejection || graph_consumer)
            return Exl3GdnHistoryStorage::refused;
        return shared_wide_bytes>=wide_bytes_required(row_capacity)
            ? Exl3GdnHistoryStorage::shared_wide
            : Exl3GdnHistoryStorage::refused;
    }
};

// Exact represented residual is retained in both routes. Caller owns all extents
// through stream completion; output storage must not alias either input/weights.
void exl3_gdn_residual_norm(const std::uint16_t* left,const std::uint16_t* right,
    const std::uint16_t* weight,std::uint16_t* residual,std::uint16_t* normalized,
    int rows,bool fused,cudaStream_t stream=nullptr);

struct Exl3GdnLayerWeights {
    Exl3CudaLinearWeights qkv{};
    Exl3CudaLinearWeights z{};
    Exl3CudaLinearWeights o{};
    Exl3CudaLinearWeights gate{};
    Exl3CudaLinearWeights up{};
    Exl3CudaLinearWeights down{};
    Exl3CudaLinearMetadata qkv_metadata{};
    Exl3CudaLinearMetadata z_metadata{};
    Exl3CudaLinearMetadata o_metadata{};
    Exl3CudaLinearMetadata gate_metadata{};
    Exl3CudaLinearMetadata up_metadata{};
    Exl3CudaLinearMetadata down_metadata{};

    const std::uint16_t* input_norm = nullptr;
    const std::uint16_t* gdn_norm = nullptr;
    const std::uint16_t* post_attention_norm = nullptr;
    const std::uint16_t* conv_weight = nullptr; // [10240, 4] BF16
    const std::uint16_t* a_weight = nullptr;    // [48, 5120] F16 checkpoint layout
    const std::uint16_t* b_weight = nullptr;    // [48, 5120] F16 checkpoint layout
    const float* a_log = nullptr;               // [48] FP32
    const float* dt_bias = nullptr;             // [48] FP32, represented BF16 values
};

// One immutable descriptor per model GDN layer. Context lanes alias this fixed
// record and its model owner; mutable recurrent/convolution planes never enter it.
struct Exl3GdnImmutableCoefficients {
    std::shared_ptr<const void> model_owner;
    int model_layer = -1;
    Exl3GdnLayerWeights weights{};

    static bool same_linear(const Exl3CudaLinearWeights& left,
                            const Exl3CudaLinearWeights& right) noexcept {
        return left.trellis==right.trellis && left.suh==right.suh &&
            left.svh==right.svh && left.mul1==right.mul1;
    }
    static bool same_metadata(const Exl3CudaLinearMetadata& left,
                              const Exl3CudaLinearMetadata& right) noexcept {
        return left.in_features==right.in_features &&
            left.out_features==right.out_features && left.K==right.K &&
            left.mcg==right.mcg && left.mul1==right.mul1 &&
            left.has_bias==right.has_bias;
    }
    bool same_representation(const Exl3GdnLayerWeights& other) const noexcept {
        return same_linear(weights.qkv,other.qkv) &&
            same_linear(weights.z,other.z) && same_linear(weights.o,other.o) &&
            same_linear(weights.gate,other.gate) &&
            same_linear(weights.up,other.up) &&
            same_linear(weights.down,other.down) &&
            same_metadata(weights.qkv_metadata,other.qkv_metadata) &&
            same_metadata(weights.z_metadata,other.z_metadata) &&
            same_metadata(weights.o_metadata,other.o_metadata) &&
            same_metadata(weights.gate_metadata,other.gate_metadata) &&
            same_metadata(weights.up_metadata,other.up_metadata) &&
            same_metadata(weights.down_metadata,other.down_metadata) &&
            weights.input_norm==other.input_norm &&
            weights.gdn_norm==other.gdn_norm &&
            weights.post_attention_norm==other.post_attention_norm &&
            weights.conv_weight==other.conv_weight &&
            weights.a_weight==other.a_weight && weights.b_weight==other.b_weight &&
            weights.a_log==other.a_log && weights.dt_bias==other.dt_bias;
    }
    bool matches(const std::shared_ptr<const void>& expected_model,
                 int expected_layer,const Exl3GdnLayerWeights& expected) const noexcept {
        return model_owner && expected_model && model_owner==expected_model &&
            model_layer==expected_layer &&
            Exl3GdnRecurrentLayout::valid_layer(model_layer) &&
            same_representation(expected);
    }
};

struct Exl3GdnLayerTrace {
    const std::uint16_t* layer_input = nullptr;
    const std::uint16_t* input_norm = nullptr;
    const std::uint16_t* qkv_projection = nullptr;
    const std::uint16_t* z_projection = nullptr;
    const float* b_projection = nullptr;
    const float* a_projection = nullptr;
    const std::uint16_t* conv_input = nullptr;
    const std::uint16_t* conv_output = nullptr;
    const float* beta = nullptr;
    const float* g = nullptr;
    const float* state_before = nullptr;
    const float* state_after = nullptr;
    const std::uint16_t* linear_attention_output = nullptr;
    const std::uint16_t* gdn_norm_input = nullptr;
    const std::uint16_t* gdn_norm = nullptr;
    const std::uint16_t* output_projection_input = nullptr;
    const std::uint16_t* output_projection = nullptr;
    const std::uint16_t* post_attention_residual = nullptr;
    const std::uint16_t* mlp_input = nullptr;
    const std::uint16_t* gate_projection = nullptr;
    const std::uint16_t* up_projection = nullptr;
    const std::uint16_t* activated_mlp = nullptr;
    const std::uint16_t* down_projection = nullptr;
    const std::uint16_t* layer_output = nullptr;
};

struct Exl3GdnLayerTimings {
    std::array<double, 12> microseconds{};
    double total_microseconds = 0.0;
};

// Caller-owned device storage for one layer transaction. The caller keeps both
// buffers alive until the stream used by save_checkpoint/restore_checkpoint has
// completed the operation. The two ranges must not overlap each other or any
// layer-owned state/scratch, except for the explicitly marked layer-owned
// recurrent trace alias. Restore requires a valid saved snapshot that remains
// unmodified until restore completes. Order forward/reset/save/restore on one
// stream or establish explicit dependencies between streams.
struct Exl3GdnLayerCheckpoint {
    void* recurrent_state_device = nullptr;
    std::size_t recurrent_state_capacity_bytes = 0;
    void* conv_state_device = nullptr;
    std::size_t conv_state_capacity_bytes = 0;
    // Written only by a successful eager save_checkpoint(). These fields bind
    // retained-prefix reconstruction to the layer and exact saved generation.
    // A captured save clears them; restore never rearms them.
    mutable const Exl3GdnLayer* owner = nullptr;
    mutable std::uint64_t generation = 0;
    // Freshness authority belongs to this descriptor's backing ranges, not to
    // the layer globally: independent base/replacement checkpoints may coexist.
    // Copies borrow the source descriptor's authority and therefore become
    // stale when that source overwrites the same backing ranges.
    mutable std::uint64_t backing_generation = 0;
    mutable const std::uint64_t* generation_authority = nullptr;
    mutable const void* saved_recurrent_state_device = nullptr;
    mutable const void* saved_conv_state_device = nullptr;
    std::shared_ptr<const void> model_owner;
    int model_layer = -1;
    mutable int position = -1;
    std::size_t recurrent_layer_stride_bytes = 0;
    std::size_t convolution_layer_stride_bytes = 0;
    // Exact HostKV verifier-only optimization: save the root recurrent image in
    // the layer's existing state-before trace. The immediately following
    // preserve_m1_topology forward must keep that trace immutable instead of
    // redundantly copying the same root into it again.
    bool recurrent_trace_alias = false;

    static bool ranges_disjoint(const void* left,std::size_t left_bytes,
                                const void* right,std::size_t right_bytes) noexcept {
        if(!left || !right)return false;
        const auto l=reinterpret_cast<std::uintptr_t>(left);
        const auto r=reinterpret_cast<std::uintptr_t>(right);
        if(left_bytes>std::numeric_limits<std::uintptr_t>::max()-l ||
           right_bytes>std::numeric_limits<std::uintptr_t>::max()-r)
            return false;
        return l+left_bytes<=r || r+right_bytes<=l;
    }

    bool layout_valid() const noexcept {
        return model_owner && Exl3GdnRecurrentLayout::valid_layer(model_layer) &&
            recurrent_state_device && conv_state_device &&
            recurrent_state_capacity_bytes>=Exl3GdnRecurrentLayout::recurrent_bytes &&
            conv_state_capacity_bytes>=Exl3GdnRecurrentLayout::convolution_storage_bytes &&
            recurrent_layer_stride_bytes==Exl3GdnRecurrentLayout::recurrent_bytes &&
            convolution_layer_stride_bytes==Exl3GdnRecurrentLayout::convolution_storage_bytes &&
            ranges_disjoint(recurrent_state_device,
                            Exl3GdnRecurrentLayout::recurrent_bytes,
                            conv_state_device,
                            Exl3GdnRecurrentLayout::convolution_storage_bytes);
    }

    bool saved_for(const std::shared_ptr<const void>& expected_model,
                   int expected_layer,int expected_position,
                   std::uint64_t expected_generation,
                   const Exl3GdnLayer* expected_owner) const noexcept {
        return layout_valid() && position>=0 && position==expected_position &&
            expected_model &&
            model_owner==expected_model && model_layer==expected_layer &&
            owner==expected_owner && generation!=0 &&
            generation==expected_generation &&
            generation_authority && *generation_authority==generation &&
            saved_recurrent_state_device==recurrent_state_device &&
            saved_conv_state_device==conv_state_device;
    }
};

struct Exl3GdnStageOracleTelemetry {
    std::uint32_t staged_pairs = 0;
    std::uint32_t projection_phases = 0;
    std::uint32_t serial_b8_projection_calls = 0;
    std::uint32_t fallback_calls = 0;
    std::uint32_t published_lanes = 0;
};

// Context-owned resources for the experimental fixed-B8 continuation graph.
// The auxiliary stream, events, and private Z workspace must be created before
// graph capture and must outlive the captured graph executable. The private
// workspace deliberately borrows neither the context accumulation arena nor
// its transformed-input storage.
struct Exl3GdnGraphQkvzConcurrencyView {
    cudaStream_t z_stream = nullptr;
    cudaEvent_t fork = nullptr;
    cudaEvent_t z_done = nullptr;
    Exl3CudaLinearWorkspace* z_workspace = nullptr;

    bool complete() const noexcept {
        return z_stream && fork && z_done && z_workspace;
    }
};

// Optional context-owned storage for rows that exceed the private retention
// capacity.  The owner keeps this slab alive for the context lifetime and
// orders all sharing layers on one stream.  It is never shared across
// contexts and never used by retained-prefix reconstruction.
struct Exl3GdnWideScratchView {
    void* data = nullptr;
    std::size_t bytes = 0;
};

class Exl3GdnLayer {
public:
    GoptSubmissions gaming_submissions() const noexcept { return gaming_submissions_; }
    // Successful host submissions outside graph capture; not replay or GPU completion counts.
    std::uint64_t paired_transform_submissions() const noexcept { return paired_transform_submissions_; }
    std::uint64_t fused_gate_up_submissions() const noexcept { return fused_gate_up_submissions_; }
    std::uint64_t fused_residual_norm_submissions() const noexcept { return fused_residual_norm_submissions_; }
    // Default-off same-weight FP16-KV decode staging candidate. This counts
    // host submissions of the fused F16->BF16 convolution/packing kernel, not
    // completion events or generated tokens.
    std::uint64_t fast_same_weights_fp16kv_gdn_decode_conv_calls() const noexcept {
        return fast_same_weights_fp16kv_gdn_decode_conv_calls_;
    }
    // Actual host submissions of the separately labeled native same-weight
    // FP16-KV GDN M=1 gate/up pair.  Graph capture contributes one submission
    // per captured layer; graph replay does not fabricate additional calls.
    std::uint64_t fast_same_weights_fp16kv_gdn_m1_gate_up_pair_submissions() const noexcept {
        return fast_same_weights_fp16kv_gdn_m1_gate_up_pair_submissions_;
    }
    Exl3PrefillProjectionChainGraph::Snapshot prefill_projection_chain_graph_stats() const noexcept {
        return prefill_projection_chain_graph_.snapshot();
    }
    void complete_prefill_projection_chain_graph_after_drain(cudaStream_t stream);
    static std::size_t shared_scratch_bytes(int rows);
    static std::size_t wide_scratch_bytes(int rows);
    Exl3GdnLayer(const Exl3GdnLayerWeights& weights, int max_rows = 16,
                 Exl3CudaAccumulationView accumulation = {}, Exl3CudaTransformView transformed = {},
                 Exl3CudaLayerScratchView scratch = {},
                 Exl3GdnWideScratchView wide_scratch = {},
                 Exl3CudaReconstructGemmWorkspace* reconstruct_gemm = nullptr,
                 class Exl3VeriCacheServingCoordinator* constructor_authority=nullptr,
                 unsigned buffer_fault_for_test=0,
                 std::shared_ptr<const Exl3GdnImmutableCoefficients> immutable_coefficients={},
                 Exl3CudaAccumulationView paired_up_accumulation = {},
                 Exl3CudaTransformView paired_up_transformed = {});
    ~Exl3GdnLayer();

    Exl3GdnLayer(const Exl3GdnLayer&) = delete;
    Exl3GdnLayer& operator=(const Exl3GdnLayer&) = delete;

    void reset(cudaStream_t stream = nullptr);
    struct BulkPrefillBuffers {
        std::uint16_t* h = nullptr;
        std::uint16_t* qkv = nullptr;
        std::uint16_t* z = nullptr;
        int rows = 0;
    };
    struct DeferredMlpBuffers {
        std::uint16_t* post = nullptr;
        std::uint16_t* mlp_input = nullptr;
        int rows = 0;
    };
    bool supports_bulk_prefill() const noexcept;
    void prepare_bulk_prefill(const std::uint16_t* input,
                             BulkPrefillBuffers buffers,
                             cudaStream_t stream);
    bool supports_bulk_mlp(int rows) const noexcept;
    void forward_before_bulk_mlp(const std::uint16_t* input,
                                 DeferredMlpBuffers buffers,
                                 cudaStream_t stream);
    void finish_bulk_mlp(DeferredMlpBuffers buffers,
                         std::uint16_t* gate,std::uint16_t* up,
                         std::uint16_t* act,std::uint16_t* down,
                         std::uint16_t* output,
                         cudaStream_t stream,bool preserve_trace=false);
    void forward(const std::uint16_t* input, std::uint16_t* output, int rows,
                 cudaStream_t stream = nullptr, bool profile = false,
                 // Explicit target-verifier mode; nonlinear staging remains
                 // batched while every packed projection retains its M1 path.
                 bool preserve_m1_topology = false,
                 bool wide_prefill = false,
                 const BulkPrefillBuffers* prepared = nullptr,
                 const DeferredMlpBuffers* deferred_mlp = nullptr);
    // Qualification-only T0a topology oracle. One host thread submits a fixed
    // projection-by-projection schedule for two independent real layer
    // instances. Each lane retains the authoritative fixed-B8 projection
    // routing and private mutable state; no cross-lane arithmetic is used.
    // The ordinary forward path does not call this entry point.
    static void forward_pair_staged_serial_for_test(
        Exl3GdnLayer& first, const std::uint16_t* first_input,
        std::uint16_t* first_output, Exl3GdnLayer& second,
        const std::uint16_t* second_input, std::uint16_t* second_output,
        cudaStream_t stream = nullptr, int fail_after_published_lane = -1,
        Exl3GdnStageOracleTelemetry* telemetry = nullptr);
    // Graph-local only: install one context-owned private Z workspace and this
    // layer's fork/join events. Ordinary eager and prefill forwards remain on
    // their original single-stream path.
    void prepare_continuation_graph_qkvz_concurrency(
        Exl3GdnGraphQkvzConcurrencyView view);
    // Ordinary eager continuation only. The owning text context serializes
    // reuse of this private up workspace across layers with the supplied
    // events; graph capture, prefill and shared projection paths are excluded.
    void prepare_eager_mlp_gateup_concurrency(
        Exl3MlpGateUpConcurrencyView view);
    void set_capture_active(bool active) noexcept { capture_active_ = active; }
    // Verifier row layout (sibling_rows.cuh); nullptr keeps every row a chain row.
    void set_chain_rows_device(const int* chain_rows) noexcept { chain_rows_device_ = chain_rows; }
    // Production contexts skip trace-only work: the per-forward recurrent
    // state-before copy (3 MB per layer) and the packed conv-output/head copies;
    // only diagnostics and tests read those trace() fields.
    void set_skip_state_trace(bool skip) noexcept { skip_state_trace_ = skip; }
    // The next GDN layer of the stack: this layer's decode down reduction also
    // writes the successor's input RMS norm into the successor's input-norm
    // buffer, and the successor skips its own norm for exactly that input.
    void set_successor(Exl3GdnLayer* successor) noexcept { successor_ = successor; }
    // Weights of the next layer's first projection, prefetched into L2 on a
    // side branch before this layer's down projection (M1 decode).
    void set_next_layer_prefetch(const std::uint16_t* trellis) noexcept { next_layer_prefetch_ = trellis; }
    const std::uint16_t* first_projection_trellis() const noexcept { return weights_.qkv.trellis; }
    // Per-row continuation traces (convolution input, q/k/v, gates) of the
    // last `rows`-row forward that an accepted sibling must carry into
    // `destination` before retained-prefix repair.
    void append_sibling_row_copies(Exl3SiblingRowCopies& out,int source,int destination,
                                   int rows) const;
    void set_shared_gateup_executor(Exl3TargetQExecutor executor,int layer,bool gateup=true,bool down=false) {
        shared_gateup_executor_=std::move(executor);model_layer_=layer;
        shared_gateup_enabled_=gateup;shared_down_enabled_=down;
    }
    // These operations are asynchronous and ordered only by the supplied stream.
    // They allocate no memory and copy the complete mutable recurrent and
    // physical four-slot convolution state.
    // Restore refreshes state diagnostics; other trace fields and timings still
    // describe the last executed forward, not the checkpoint's forward.
    void bind_recurrent_layout(std::shared_ptr<const void> model_owner,int model_layer);
    void validate_checkpoint_storage(const Exl3GdnLayerCheckpoint& checkpoint) const;
    void validate_saved_checkpoint(const Exl3GdnLayerCheckpoint& checkpoint,
                                   int expected_position) const;
    void save_checkpoint(const Exl3GdnLayerCheckpoint& checkpoint,int position,
                         cudaStream_t stream = nullptr);
    // Stable-pointer HostKV transaction graph support. Capture records only
    // the two D2D copies. Each replay invalidates the previous capability
    // before graph launch, then publishes the new generation after submission.
    void record_checkpoint_copy_nodes(const Exl3GdnLayerCheckpoint& checkpoint,
                                      cudaStream_t stream) const;
    std::uint64_t begin_checkpoint_graph_replay(
        const Exl3GdnLayerCheckpoint& checkpoint);
    void publish_checkpoint_graph_replay(
        const Exl3GdnLayerCheckpoint& checkpoint,int position,
        std::uint64_t generation,cudaStream_t stream);
    void restore_checkpoint(const Exl3GdnLayerCheckpoint& checkpoint, cudaStream_t stream = nullptr);
    // Synchronous host checkpoint restore, preserving FP32 recurrent and all
    // four physical BF16 convolution slots. Invalidates retention capabilities.
    void restore_host_state(std::span<const float> recurrent,
                            std::span<const std::uint16_t> convolution,
                            cudaStream_t stream = nullptr);
    // Reconstruct the state after retained_rows from a fresh eager save and
    // scratch produced by the immediately following successful
    // preserve_m1_topology B2..8 forward. Restore never creates this capability:
    // after restoring a reusable base, save it eagerly again before the forward.
    // The operation is asynchronous, allocates nothing, consumes the capability,
    // and requires the same stream. The checkpoint remains reusable for restore.
    void reconstruct_retained_prefix(const Exl3GdnLayerCheckpoint& checkpoint,
                                     int retained_rows,
                                     cudaStream_t stream = nullptr);
    // Split form for a captured multi-layer repair graph: the host part runs
    // every provenance check and state transition of reconstruct_retained_prefix
    // without GPU work; the enqueue part submits exactly its device work and
    // may be recorded into a graph (fixed buffers, no host state access).
    // Returns the attempted row count the enqueue must use.
    int reconstruct_retained_prefix_host(const Exl3GdnLayerCheckpoint& checkpoint,
                                         int retained_rows, cudaStream_t stream);
    void enqueue_retained_prefix_reconstruct(const Exl3GdnLayerCheckpoint& checkpoint,
                                             int retained_rows, int attempted_rows,
                                             cudaStream_t stream) const;
    // Host-side completion for a successfully replayed fixed-width graph.
    // Binds stable forward scratch to the eager checkpoint taken immediately
    // before replay; it performs no device work or state publication.
    void arm_captured_retained_prefix(const Exl3GdnLayerCheckpoint& checkpoint,
                                      int attempted_rows,cudaStream_t stream);
    void set_projection_timing(Exl3TargetProjectionTiming* timing,
                               int model_layer) noexcept {
        projection_timing_ = timing;
        model_layer_ = model_layer;
    }
    void set_projection_observer(Exl3TargetProjectionObserver observer,
                                 void* user, int model_layer,
                                 Exl3TargetProjectionObserverSelection selection) noexcept {
        projection_observer_ = observer;
        projection_observer_user_ = observer ? user : nullptr;
        projection_observer_selection_ = selection;
        model_layer_ = model_layer;
    }

    Exl3GdnLayerTrace trace() const noexcept { return trace_; }
    std::uint64_t stream_reduction_calls(bool extended=false) const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->stream_reduction_calls(extended);return result;
    }
    std::uint64_t k6_gateup_warpgroup_async_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_gateup_warpgroup_async_calls();return result;
    }
    std::uint64_t k6_gateup_n32_pair_cta_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_gateup_n32_pair_cta_calls();return result;
    }
    std::uint64_t k6_gateup_n32_pair_cta_rows() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_gateup_n32_pair_cta_rows();return result;
    }
    std::uint64_t k6_fast_decode_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_fast_decode_calls();return result;
    }
    std::uint64_t k6_rowpair_n64_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_rowpair_n64_calls();return result;
    }
    std::uint64_t k6_rowpair_n64_rows() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_rowpair_n64_rows();return result;
    }
    std::uint64_t k6_down_rowpair_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_down_rowpair_calls();return result;
    }
    std::uint64_t k6_down_rowpair_rows() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_down_rowpair_rows();return result;
    }
    std::uint64_t shape4_n64_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->shape4_n64_calls();return result;
    }
    std::uint64_t shape4_n64_rows() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->shape4_n64_rows();return result;
    }
    std::uint64_t reduce_shfl_min_barrier_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->reduce_shfl_min_barrier_calls();return result;
    }
    std::uint64_t reduce_shfl_min_barrier_rows() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->reduce_shfl_min_barrier_rows();return result;
    }
    std::uint64_t k6_fast_decode_rows() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k6_fast_decode_rows();return result;
    }
    std::uint64_t k7_tiles64_exact_splits_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k7_tiles64_exact_splits_calls();return result;
    }
    std::uint64_t target_down_k6_async_a_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->target_down_k6_async_a_calls();return result;
    }
    std::uint64_t target_gateup_k6_n16_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->target_gateup_k6_n16_calls();return result;
    }
    std::uint64_t target_k6_small_m_async_a_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->target_k6_small_m_async_a_calls();return result;
    }
    std::uint64_t target_k7_small_m_async_a_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->target_k7_small_m_async_a_calls();return result;
    }
    std::uint64_t target_k5_small_m_batch_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->target_k5_small_m_batch_calls();return result;
    }
    std::uint64_t fast_wide_prefill_gemm_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->fast_wide_prefill_gemm_calls();return result;
    }
    std::uint64_t fast_wide_prefill_gemm_rows() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->fast_wide_prefill_gemm_rows();return result;
    }
    std::uint64_t native_k6_critical_path_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->native_k6_critical_path_calls();return result;
    }
    std::uint64_t native_k6_register_pipeline_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->native_k6_register_pipeline_calls();return result;
    }
    std::uint64_t prefill_projection_graph_captures() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->prefill_projection_graph_captures();return result;
    }
    std::uint64_t prefill_projection_graph_replays() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->prefill_projection_graph_replays();return result;
    }
    std::uint64_t prefill_projection_graph_binding_fallbacks() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->prefill_projection_graph_binding_fallbacks();return result;
    }
    std::uint64_t eager_mlp_gateup_concurrent_calls() const noexcept {
        return eager_mlp_gateup_concurrent_calls_;
    }
    void set_native_continuation16(bool enabled) noexcept {
        for (auto* workspace : linear_workspaces_)
            if (workspace) workspace->set_native_continuation16(enabled);
    }
    void set_reconstructed_exact(Exl3ReconstructedExactView view) noexcept {
        for (auto* workspace : linear_workspaces_)
            if (workspace) workspace->set_reconstructed_exact(view);
    }
    const Exl3GdnLayerTimings& last_timings() const noexcept { return timings_; }
    const float* recurrent_state_device() const noexcept { return recurrent_state_; }
    const std::uint16_t* conv_state_device() const noexcept { return conv_state_trace_; }
    const std::uint16_t* physical_conv_state_device() const noexcept { return conv_state_; }
    std::shared_ptr<const void> immutable_coefficient_owner() const noexcept {
        return immutable_coefficients_;
    }
    std::size_t recurrent_state_bytes() const noexcept { return kStateBytes; }
    std::size_t conv_state_bytes() const noexcept { return kConvStateBytes; }
    std::size_t physical_conv_state_bytes() const noexcept { return kConvStateStorageBytes; }
    std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
    static std::size_t fixed_owner_metadata_required() noexcept;
    std::size_t fixed_owner_metadata_bytes() const noexcept;
    // Batched multi-agent rounds: while set, forward() runs its row-independent
    // work over own_rows + every peer's rows, and each peer layer (another
    // agent's context, same model layer) runs its own stateful core on its rows.
    // stream: the peer's own (transaction) stream; launches use the owner's.
    struct BatchPeer { Exl3GdnLayer* layer = nullptr; int rows = 0; cudaStream_t stream = nullptr; };
    struct Batch {
        int own_rows = 0;
        int peer_count = 0;
        std::array<BatchPeer, 1> peers{};
    };
    void set_batch(const Batch* batch) noexcept { batch_ = batch; }
    // Borrowed child pointers for enclosing-context lifetime inventory only.
    std::array<Exl3CudaLinearWorkspace*,6> linear_workspace_owners() const noexcept {return linear_workspaces_;}
    std::array<Exl3LayerBufferRetirement*,34> buffer_retirement_owners() noexcept {
        std::array<Exl3LayerBufferRetirement*,34> result{};
        for(std::size_t i=0;i<result.size();++i)if(buffer_retirements_[i] && buffer_retirements_[i]->bytes())result[i]=&*buffer_retirements_[i];
        return result;
    }
    static std::size_t workspace_bytes_required(int rows,bool borrowed_accumulation,
        bool borrowed_transform,bool borrowed_scratch,bool split_wide,bool resident);
    static constexpr std::size_t kStateElements = Exl3GdnRecurrentLayout::recurrent_elements;
    static constexpr std::size_t kStateBytes = Exl3GdnRecurrentLayout::recurrent_bytes;
    static constexpr std::size_t kConvStateElements = Exl3GdnRecurrentLayout::convolution_history_elements;
    static constexpr std::size_t kConvStateStorageElements = Exl3GdnRecurrentLayout::convolution_storage_elements;
    static constexpr std::size_t kConvStateBytes = Exl3GdnRecurrentLayout::convolution_history_bytes;
    static constexpr std::size_t kConvStateStorageBytes = Exl3GdnRecurrentLayout::convolution_storage_bytes;

private:
    // Arguments of gdn_middle(): the stateful core between the input
    // projections and the output projection (see gdn_layer.cu).
    struct GdnMiddle {
        std::uint16_t *qkv,*conv_input,*q,*k,*v,*conv_output,*z,*z_bf16,*core,*gdn_norm,*head_trace,*o_input;
        float *g_trace,*beta_trace;
        int rows;
        cudaStream_t stream;
        bool wide_prefill,preserve_m1_topology,merged_qkvz_side,eligible_retained_prefix;
        const void* base_checkpoint_recurrent;
        bool collect_stage_events;
        cudaEvent_t* starts;
        cudaEvent_t* ends;
    };
    void gdn_middle(const GdnMiddle& m);
    struct SegmentState {
        std::uint64_t base_generation = 0;
        const void* base_recurrent = nullptr;
        const void* base_conv = nullptr;
        bool eligible = false;
        Exl3GdnHistoryStorage storage{};
    };
    SegmentState begin_segment(int rows, cudaStream_t stream, bool preserve_m1_topology,
                               cudaStream_t state_stream = nullptr);
    void end_segment(const SegmentState& segment, int rows, cudaStream_t stream,
                     bool preserve_m1_topology);
    struct PeerSegmentSource {
        const std::uint16_t *h,*qkv,*conv_input,*z;
        const float *a,*b,*g_trace,*beta_trace;
        std::uint16_t* o_input;
        int rows;
        bool merged_qkvz_side;
    };
    void run_peer_segment(const PeerSegmentSource& source, cudaStream_t stream,
                          cudaStream_t state_stream, bool preserve_m1_topology);
    const Batch* batch_ = nullptr;
    friend class Exl3TextContext;
    // The TextContext passes the bounded shared owner for this layer. Every
    // exposed history buffer is layer-owned, while model_owner separately
    // retains the immutable coefficients. Internal repair uses the same row
    // contract synchronously.
    Exl3GdnContinuationHistoryView continuation_history_view(
        std::shared_ptr<const void> storage_owner,int base_position,
        int first_row,int rows) const;
    Exl3GdnLayerWeights weights_{};
    int max_rows_ = 0;
    Exl3GamingOptions gaming_ = Exl3GamingOptions::from_environment();
    GoptSubmissions gaming_submissions_{};
    bool prefill_resident_ = false;
    bool prefill_resident_pair_columns_ = false;
    bool prefill_resident_pair_vector_io_ = false;
    bool prefill_resident_quad_columns_ = false;
    bool dual_input_transform_ = false;
    bool fused_gate_up_transform_ = false;
    bool small_m_fused_gate_up_transform_ = false;
    bool bulk_mlp_fused_down_ = false;
    bool bulk_mlp_fused_residual_ = false;
    bool bulk_mlp_weight_prefetch_ = false;
    bool fused_residual_norm_ = false;
    bool fast_same_weights_fp16kv_gdn_decode_conv_ = false;
    bool fast_same_weights_fp16kv_gdn_m1_gate_up_pair_ = false;
    std::uint64_t fused_residual_norm_submissions_ = 0;
    std::uint64_t fused_gate_up_submissions_ = 0;
    std::uint64_t paired_transform_submissions_ = 0;
    std::uint64_t fast_same_weights_fp16kv_gdn_decode_conv_calls_ = 0;
    std::uint64_t fast_same_weights_fp16kv_gdn_m1_gate_up_pair_submissions_ = 0;
    float* prefill_normalized_ = nullptr;
    bool prefill_normalized_owned_ = false;
    std::size_t workspace_bytes_ = 0;
    std::array<std::uint16_t*, 22> half_buffers_{};
    std::array<std::uint16_t*, 5> wide_half_buffers_{};
    std::array<float*, 6> float_buffers_{};
    std::array<std::size_t,22> half_buffer_bytes_{};
    std::array<std::size_t,5> wide_half_buffer_bytes_{};
    std::array<std::size_t,6> float_buffer_bytes_{};
    std::size_t prefill_normalized_bytes_ = 0;
    std::array<std::optional<Exl3LayerBufferRetirement>,34> buffer_retirements_{};
    std::array<bool, 22> half_buffer_owned_{};
    std::array<bool, 6> float_buffer_owned_{};
    bool split_wide_storage_ = false;
    int private_row_capacity_ = 0;
    Exl3GdnScratchReuseContract scratch_reuse_{};
    float* recurrent_state_ = nullptr;
    float* recurrent_state_before_ = nullptr;
    bool skip_state_trace_ = false;
    Exl3GdnLayer* successor_ = nullptr;
    const std::uint16_t* next_layer_prefetch_ = nullptr;
    const std::uint16_t* prenormalized_input_ = nullptr;
    int prenormalized_rows_ = 0;
    std::uint16_t* conv_state_ = nullptr;
    std::uint16_t* conv_state_trace_ = nullptr;
    DeviceArena* op_workspace_ = nullptr;
    std::array<Exl3CudaLinearWorkspace*, 6> linear_workspaces_{};
    Exl3CudaReconstructGemmWorkspace* reconstruct_gemm_ = nullptr;
    Exl3TargetProjectionTiming* projection_timing_ = nullptr;
    Exl3TargetProjectionObserver projection_observer_ = nullptr;
    void* projection_observer_user_ = nullptr;
    Exl3TargetProjectionObserverSelection projection_observer_selection_ =
        Exl3TargetProjectionObserverSelection::gate_up_k6;
    int model_layer_ = -1;
    std::shared_ptr<const void> recurrent_model_owner_;
    std::shared_ptr<const Exl3GdnImmutableCoefficients> immutable_coefficients_;
    Exl3TargetQExecutor shared_gateup_executor_;
    Exl3ActivationLifetime mlp_activation_lifetime_;
    bool shared_gateup_enabled_=false,shared_down_enabled_=false;
    const int* chain_rows_device_ = nullptr;
    bool capture_active_ = false;
    Exl3GdnGraphQkvzConcurrencyView graph_qkvz_concurrency_{};
    std::shared_ptr<const void> prefill_projection_chain_context_owner_ =
        std::make_shared<int>(1);
    std::shared_ptr<const void> prefill_projection_chain_model_owner_ =
        std::make_shared<int>(2);
    std::shared_ptr<const void> prefill_projection_chain_scratch_owner_ =
        std::make_shared<int>(3);
    Exl3PrefillProjectionChainGraph prefill_projection_chain_graph_;
    Exl3MlpGateUpConcurrencyView eager_mlp_gateup_concurrency_{};
    std::uint64_t eager_mlp_gateup_concurrent_calls_ = 0;
    std::uint64_t checkpoint_generation_ = 0;
    std::uint64_t current_checkpoint_generation_ = 0;
    cudaStream_t current_checkpoint_stream_ = nullptr;
    const void* current_checkpoint_recurrent_ = nullptr;
    const void* current_checkpoint_conv_ = nullptr;
    bool retained_prefix_available_ = false;
    int retained_prefix_rows_ = 0;
    cudaStream_t retained_prefix_stream_ = nullptr;
    std::uint64_t retained_prefix_checkpoint_generation_ = 0;
    const void* retained_prefix_checkpoint_recurrent_ = nullptr;
    const void* retained_prefix_checkpoint_conv_ = nullptr;
    std::uint64_t continuation_history_generation_ = 0;
    int continuation_history_rows_ = 0;
    Exl3GdnLayerTrace trace_{};
    Exl3GdnLayerTimings timings_{};

    void invalidate_continuation_history() noexcept {
        if(++continuation_history_generation_==0)
            ++continuation_history_generation_;
        continuation_history_rows_=0;
    }

};

} // namespace ninfer::exl3
