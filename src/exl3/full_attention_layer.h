#pragma once

#include "exl3/linear_cuda.h"
#include "exl3/sibling_rows.h"
#include "exl3/layer_buffer_retirement.h"
#include "exl3/attention_page_ranges.h"
#include "exl3/attention_position_contract.h"
#include "exl3/attention_input_view.h"
#include "exl3/attention_causal_rows.h"
#include "exl3/attention_profile.h"
#include "exl3/oscar_runtime.h"
#include "exl3/target_projection_timing.h"
#include "exl3/target_projection_observer.h"
#include "exl3/target_q_continuation.h"
#include "exl3/prefill_projection_chain_graph.h"
#include "exl3/prefill_attention_chain_graph.h"
#include "core/decode_graph.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::exl3 {
// NINFER_EXL3_FA2_PREFILL (default 1): FA2-style prefill attention route.
bool exl3_fa2_prefill_enabled();

struct Exl3PrefillQkvConcurrencyView {
    cudaStream_t k_stream = nullptr;
    cudaStream_t v_stream = nullptr;
    cudaEvent_t fork = nullptr;
    cudaEvent_t k_done = nullptr;
    cudaEvent_t v_done = nullptr;
    Exl3CudaLinearWorkspace* k_workspace = nullptr;
    Exl3CudaLinearWorkspace* v_workspace = nullptr;

    bool complete() const noexcept {
        return k_stream && v_stream && fork && k_done && v_done &&
            k_workspace && v_workspace;
    }
};

void exl3_exact_page_attention_for_test(const std::uint16_t* q,
    const Exl3AttentionPageRanges& ranges,const std::uint16_t* private_k,const std::uint16_t* private_v,
    std::uint16_t* output,float* scores,int rows,int position,int capacity,cudaStream_t stream=nullptr,bool q_shared=false,bool gqa_pair=false,bool query_pair=false);
// Research entry: immutable interval [prefix_first,prefix_first+prefix_rows),
// private absolute-indexed rows elsewhere. FP32 arithmetic/order is unchanged.
void exl3_exact_segmented_attention_for_test(const std::uint16_t* q,
    const std::uint16_t* prefix_k,const std::uint16_t* prefix_v,int prefix_rows,
    const std::uint16_t* tail_k,const std::uint16_t* tail_v,std::uint16_t* output,
    float* scores,int rows,int position,int capacity,cudaStream_t stream=nullptr,int prefix_first=0);

// Qualification surface for represented FP16 Q/K/V, causal GQA24:4,d256.
// scores covers rows*24*capacity FP32 elements; parallel=false is the original
// serial arithmetic kernel (short contexts only, dynamic shared score limit).
void exl3_exact_attention_for_test(const std::uint16_t* q,const std::uint16_t* k,
    const std::uint16_t* v,std::uint16_t* output,float* scores,int rows,
    int position,int capacity,bool parallel,cudaStream_t stream=nullptr,
    bool q_shared=false,bool k_half2=false,bool v_half2=false,bool gqa_pair=false,bool gqa_triple=false,
    bool gqa_triple_values128=false,bool gqa_triple_softmax_staged=false,
    bool gqa_six=false,bool gqa_six_scores=false,
    bool gqa_six_values_sharded=false,bool gqa_triple_values4=false,
    bool gqa_six_softmax_triple_values=false,
    bool gqa_six_packed_triples=false,
    bool gqa_six_extent_shards=false,
    bool gqa_six_query_pair_scores=false,
    bool gqa_six_softmax_triple_values_v_tile=false,
    bool gqa_six_softmax_triple_values_full_cta=false,
    bool gqa_six_softmax_triple_values_fused=false,
    bool gqa_six_softmax_tile512=false,
    bool gqa_six_score_k_tile64=false,
    bool gqa_six_softmax_triple_values_scalar_dim=false,
    bool gqa_six_softmax_triple_values_two_query=false,
    Exl3AttentionPageRanges segmented_pages={},
    bool gqa_six_softmax_triple_values_pair_dimensions=false,
    bool gqa_six_softmax_six_values_single_load=false,
    bool gqa_six_softmax_triple_values_v_tile64=false,
    bool gqa_six_softmax_triple_values_warp_score_broadcast=false,
    bool gqa_six_softmax_six_values_scalar_single_load=false,
    bool gqa_six_softmax_triple_values_key_pair_pipeline=false,
    bool gqa_six_softmax_fused_scalar_values=false);

// Test-only real-shape gate for a coordinated score-layout rewrite. The
// candidate interleaves each chronological three-head tuple; conversion/setup
// is excluded from steady-state timings and no production route uses it.
struct Exl3TripleScoreLayoutDiscriminator {
    double head_major_median_us=0.0;
    double interleaved_median_us=0.0;
    std::size_t score_bytes=0;
    std::size_t history_rows=0;
    std::uint64_t mismatches=0;
};

Exl3TripleScoreLayoutDiscriminator
exl3_triple_score_layout_discriminator_for_test();

struct Exl3SoftmaxValueFusionDiscriminator {
    double selected_median_us=0.0;
    double fused_median_us=0.0;
    std::size_t score_bytes=0;
    std::size_t history_rows=0;
    std::uint64_t output_mismatches=0;
    std::uint64_t score_mismatches=0;
};

Exl3SoftmaxValueFusionDiscriminator
exl3_softmax_value_fusion_discriminator_for_test();

struct Exl3DeferredNormalizationDiscriminator {
    double selected_median_us=0.0;
    double deferred_median_us=0.0;
    std::size_t score_bytes=0;
    std::size_t denominator_bytes=0;
    std::size_t history_rows=0;
    std::uint64_t output_mismatches=0;
    std::uint64_t normalized_score_mismatches=0;
};

Exl3DeferredNormalizationDiscriminator
exl3_deferred_normalization_discriminator_for_test();

struct Exl3AttentionGateFusionDiscriminator {
    double selected_median_us=0.0;
    double fused_median_us=0.0;
    std::size_t history_rows=0;
    std::size_t output_bytes=0;
    std::uint64_t mismatches=0;
};

Exl3AttentionGateFusionDiscriminator
exl3_attention_gate_fusion_discriminator_for_test();

// T71 research-only numerical surface. This consumes the same represented
// FP16 Q/K/V but uses tiled online softmax, so it is not bit-exact.
void exl3_numeric_attention_tiled_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream=nullptr);

void exl3_numeric_attention_rows4_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream=nullptr);

void exl3_numeric_attention_rows2_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream=nullptr);

// W08 exact large-M experiment: retain the complete six-head score plane in
// one CTA's dynamic shared memory, then execute the selected chronological
// softmax and value chains without a global score workspace.
void exl3_prefill_attention_shared_scores_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    int rows,int position,int capacity,cudaStream_t stream=nullptr,
    bool threads256=false,bool parallel_softmax=false,
    bool head_split256=false,bool dimension_split256=false);

struct Exl3PrefillAttentionSharedScoreSnapshot {
    std::uint64_t launch_attempts=0;
    std::uint64_t row_attempts=0;
    std::uint64_t global_score_bytes_eliminated=0;
    std::uint64_t threads256_launch_attempts=0;
    std::uint64_t parallel_softmax_launch_attempts=0;
    std::uint64_t head_split256_launch_attempts=0;
    std::uint64_t dimension_split256_launch_attempts=0;
};

// Process-wide real-caller dispatch seam. Attempts are recorded only after a
// valid launch submission; they do not by themselves establish completion.
Exl3PrefillAttentionSharedScoreSnapshot
exl3_prefill_attention_shared_score_global_snapshot() noexcept;

// Process-wide real-caller dispatch seam for the separately gated native
// FP16-KV fused-flash candidate.  The counter records submitted launches and
// does not by itself establish numerical or completion validity.
std::uint64_t exl3_fast_fused_flash_attention_calls_for_test() noexcept;
std::uint64_t exl3_fast_fused_flash_multirow_attention_calls_for_test() noexcept;

// Process-wide real-caller dispatch seam for the separately gated native
// whole-context FP16-KV fused attention candidate.  The counter records one
// submitted launch per layer call and does not establish numerical validity.
std::uint64_t exl3_fast_whole_context_fused_attention_calls_for_test() noexcept;

// Process-wide real-caller dispatch seam for the separately gated native
// FP16-KV online-tiled prefill candidate. The counter records submitted
// prefill launches and does not establish numerical or completion validity.
std::uint64_t exl3_fast_prefill_tiled_attention_calls_for_test() noexcept;

// Process-wide real-caller dispatch seam for the separately gated native
// FP16-KV four-query-per-CTA prefill candidate. The counter records submitted
// launches and does not by itself establish numerical or completion validity.
std::uint64_t exl3_fast_prefill_rows4_attention_calls_for_test() noexcept;
std::uint64_t exl3_fast_prefill_rows8_attention_calls_for_test() noexcept;
std::uint64_t exl3_fast_prefill_wmma_attention_calls_for_test() noexcept;
std::uint64_t exl3_fast_prefill_wmma32_attention_calls_for_test() noexcept;
std::uint64_t exl3_fast_prefill_rows2_attention_calls_for_test() noexcept;
// Process-wide real-caller dispatch seam for the separately gated native
// FP16-KV online-softmax decode candidate. The counter records submitted
// ordinary eager decode launches and does not establish numerical validity.
std::uint64_t exl3_fast_online_decode_attention_calls_for_test() noexcept;
// Process-wide real-caller dispatch seam for the separately gated native
// FP16-KV cuBLAS GQA decode candidate. The counter records completed route
// submissions only after both GEMMs and the softmax launch are enqueued.
std::uint64_t exl3_fast_cublas_attention_calls_for_test() noexcept;

std::size_t exl3_numeric_attention_splitk_workspace_bytes(int rows,int capacity);
void exl3_numeric_attention_splitk_for_test(const std::uint16_t* q,
    const std::uint16_t* k,const std::uint16_t* v,std::uint16_t* output,
    float* workspace,std::size_t workspace_bytes,int rows,int position,
    int capacity,cudaStream_t stream=nullptr);

struct Exl3FullAttentionLayerWeights {
    Exl3CudaLinearWeights q{};
    Exl3CudaLinearWeights k{};
    Exl3CudaLinearWeights v{};
    Exl3CudaLinearWeights o{};
    Exl3CudaLinearWeights gate{};
    Exl3CudaLinearWeights up{};
    Exl3CudaLinearWeights down{};

    Exl3CudaLinearMetadata q_metadata{};
    Exl3CudaLinearMetadata k_metadata{};
    Exl3CudaLinearMetadata v_metadata{};
    Exl3CudaLinearMetadata o_metadata{};
    Exl3CudaLinearMetadata gate_metadata{};
    Exl3CudaLinearMetadata up_metadata{};
    Exl3CudaLinearMetadata down_metadata{};

    const std::uint16_t* input_norm = nullptr;
    const std::uint16_t* q_norm = nullptr;
    const std::uint16_t* k_norm = nullptr;
    const std::uint16_t* post_attention_norm = nullptr;
};

struct Exl3FullAttentionLayerTrace {
    const std::uint16_t* layer_input = nullptr;
    const std::uint16_t* input_norm = nullptr;
    const std::uint16_t* q_projection = nullptr;
    const std::uint16_t* k_projection = nullptr;
    const std::uint16_t* v_projection = nullptr;
    const std::uint16_t* q_normed = nullptr;
    const std::uint16_t* k_normed = nullptr;
    const std::uint16_t* q_rope = nullptr;
    const std::uint16_t* k_rope = nullptr;
    const std::uint16_t* attention_output = nullptr;
    const std::uint16_t* output_projection = nullptr;
    const std::uint16_t* post_attention_residual = nullptr;
    const std::uint16_t* mlp_input = nullptr;
    const std::uint16_t* gate_projection = nullptr;
    const std::uint16_t* up_projection = nullptr;
    const std::uint16_t* activated_mlp = nullptr;
    const std::uint16_t* down_projection = nullptr;
    const std::uint16_t* layer_output = nullptr;
};

struct Exl3FullAttentionLayerTimings {
    std::array<double, 13> microseconds{};
    double total_microseconds = 0.0;
};

class Exl3FullAttentionLayer {
public:
    static std::size_t shared_scratch_bytes(int rows,bool coalesce_input_mlp=false);
    Exl3FullAttentionLayer(const Exl3FullAttentionLayerWeights& weights,
                            int max_rows = 16, Exl3CudaAccumulationView accumulation = {}, Exl3CudaTransformView transformed = {},
                 Exl3CudaLayerScratchView scratch = {},
                 Exl3CudaReconstructGemmWorkspace* reconstruct_gemm = nullptr,bool coalesce_input_mlp=false,
                 class Exl3VeriCacheServingCoordinator* constructor_authority=nullptr,unsigned buffer_fault_for_test=0,
                 Exl3CudaAccumulationView paired_up_accumulation = {},
                 Exl3CudaTransformView paired_up_transformed = {});
    ~Exl3FullAttentionLayer();

    Exl3FullAttentionLayer(const Exl3FullAttentionLayer&) = delete;
    Exl3FullAttentionLayer& operator=(const Exl3FullAttentionLayer&) = delete;

    void forward(const std::uint16_t* input,
                 std::uint16_t* output,
                 int rows,
                 int position = 0,
                 cudaStream_t stream = nullptr,
                 bool profile = false,
                 // Explicit target-verifier mode: batch rowwise staging, but
                 // preserve M1 projection reductions and chronological OSCAR.
                 bool preserve_m1_topology = false,
                 bool wide_prefill = false,
                 // Optional bounded continuation adapter. The executable is
                 // captured for this layer/row shape/output address and covers
                 // only post-attention norm plus the MLP tail. Attention,
                 // HostKV ownership and publication remain eager.
                 DecodeGraphExecutable* mlp_tail_graph = nullptr);

    // Capture surface for the bounded continuation adapter above. The caller
    // owns the graph and proves that output is the same stable address used at
    // replay. buffers_[10] already names this layer's fixed post-attention
    // residual, so no HostKV or position-dependent state enters the graph.
    void capture_mlp_tail_graph(std::uint16_t* output,int rows,
        cudaStream_t stream);

    // Context-owned retained-prefix repair. A successful eager OSCAR
    // preserve_m1_topology B2..8 forward publishes one same-stream capability
    // over that forward's q/kr/v scratch. Validation is mutation-free; append
    // consumes the capability and reappends rows chronologically without
    // projections, RoPE, attention decode, or allocation.
    void validate_retained_prefix_reappend(int retained_rows,
                                           int attempted_rows,
                                           int base_position,
                                           cudaStream_t stream) const;
    void reappend_retained_prefix(int retained_rows, int attempted_rows,
                                  int base_position, cudaStream_t stream);
    // Host-side completion for a successfully replayed continuation graph.
    // The graph produced the same stable q/kr/v scratch as eager forward;
    // publication remains transaction-owned and outside the graph.
    void arm_captured_retained_prefix(int attempted_rows,int base_position,
                                      cudaStream_t stream);
    void note_captured_replay(int rows) noexcept { oscar_count_ += rows; }
    void invalidate_retained_prefix() noexcept {
        retained_prefix_available_ = false;
    }

    // Optional persistent ordinary-attention cache used by the E4A model
    // context.  The existing E3A call surface remains uncached when this is
    // not set.  Cache layout is [capacity, 4 KV heads, 256 head dimensions]
    // in F16 for both K and V.
    void set_kv_cache(std::uint16_t* k_cache,
                      std::uint16_t* v_cache,
                      int capacity) noexcept;
    // Context-owned shared scratch, reused sequentially across attention layers.
    void set_exact_attention_scores(float* scores,int row_capacity) noexcept {
        exact_scores_=scores; exact_score_rows_=row_capacity;
    }
    bool supports_segmented_exact_prefix() const noexcept {
        const bool supported=Exl3AttentionProfile{exact_scores_!=nullptr,bool(oscar_),capture_active_,
            numeric_attention_splitk_,exact_attention_k_half2_,exact_attention_v_half2_,
            exact_attention_gqa_pair_,exact_attention_gqa_triple_,exact_attention_gqa_six_,
            exact_attention_gqa_six_scores_,exact_attention_gqa_six_values_sharded_,
            exact_attention_gqa_six_packed_triples_,
            exact_attention_gqa_six_softmax_triple_values_ ||
                exact_attention_gqa_six_softmax_fused_scalar_values_}.segmented();
        const bool selected_six=exact_attention_gqa_six_scores_ &&
            (exact_attention_gqa_six_softmax_triple_values_ ||
             exact_attention_gqa_six_softmax_fused_scalar_values_);
        if (fast_fused_flash_attention_ || fast_prefill_tiled_attention_) return false;
        const bool unsupported_six_variant=selected_six &&
            (exact_attention_gqa_six_query_pair_scores_ ||
             exact_attention_gqa_six_score_k_tile64_ ||
             exact_attention_gqa_six_softmax_triple_values_v_tile_ ||
             exact_attention_gqa_six_softmax_triple_values_v_tile64_ ||
             exact_attention_gqa_six_softmax_triple_values_full_cta_ ||
             exact_attention_gqa_six_softmax_triple_values_threads128_ ||
             exact_attention_gqa_six_softmax_triple_values_score_tile_ ||
             exact_attention_gqa_six_softmax_triple_values_key_pair_pipeline_ ||
             exact_attention_gqa_six_softmax_triple_values_pair_dimensions_ ||
             exact_attention_gqa_six_softmax_six_values_single_load_ ||
             exact_attention_gqa_six_softmax_six_values_scalar_single_load_ ||
             exact_attention_gqa_six_softmax_triple_values_scalar_dim_ ||
             exact_attention_gqa_six_softmax_triple_values_two_query_ ||
             exact_attention_gqa_six_softmax_triple_values_fused_ ||
             exact_attention_gqa_six_softmax_tile512_ ||
             exact_attention_gqa_six_softmax_fused_scalar_values_ ||
             exact_attention_gqa_six_decode_fused_);
        return supported && !unsupported_six_variant;
    }
    // Complete consumer admission precedes staged-plane ownership transitions.
    bool supports_direct_staged_history(int rows,int position) const noexcept {
        // A contiguous staged image preserves the selected attention kernels;
        // unlike segmented pages it does not require an alternate reader.
        const bool contiguous_profile=exact_scores_ && !oscar_ && !capture_active_ &&
            !numeric_attention_splitk_;
        return exl3_direct_staged_attention_supported(contiguous_profile,
            rows,position,cache_capacity_,exact_score_rows_);
    }
    // Borrowed until the context records the stage consumer edge.  The staged
    // image contains [0,position); forward appends the current rows to both the
    // authoritative cache and this private consumer image before attention.
    void set_direct_staged_history(std::uint16_t* k,std::uint16_t* v,int rows) {
        if(rows<0 || rows>cache_capacity_ || (rows && (!k || !v)) ||
            (!rows && (k || v)))
            throw std::invalid_argument("direct staged history admission");
        direct_staged_k_=k;direct_staged_v_=v;direct_staged_rows_=rows;
    }
    // Borrowed through this layer's completed forward. Context owns the cache
    // and the copy/compute event chain, and clears the binding every invocation.
    void set_segmented_exact_prefix(const std::uint16_t* k,const std::uint16_t* v,int rows,int first=0) {
        if(rows<0 || first<0 || first>cache_capacity_ || rows>cache_capacity_-first ||
            (rows && (!k || !v || !supports_segmented_exact_prefix())))
            throw std::invalid_argument("segmented exact prefix admission");
        exact_prefix_k_=k;exact_prefix_v_=v;exact_prefix_rows_=rows;
        exact_prefix_first_=first;
        exact_page_ranges_={};
        exact_position_contract_={mrope_positions_,rope_offset_};
    }
    void set_segmented_exact_pages(const Exl3AttentionPageRanges& ranges,int position) {
        if(!supports_segmented_exact_prefix() || position<0 || position>cache_capacity_ ||
           ranges.count<0 || ranges.count>Exl3AttentionPageRanges::capacity)
            throw std::invalid_argument("segmented page range admission");
        Exl3AttentionPageRanges checked;
        for(int i=0;i<ranges.count;++i) {
            const auto& r=ranges.ranges[i];checked.append(r.k,r.v,r.first,r.rows,position);
        }
        exact_prefix_rows_=0;exact_prefix_first_=0;exact_prefix_k_=exact_prefix_v_=nullptr;
        exact_page_ranges_=checked;
        exact_position_contract_={mrope_positions_,rope_offset_};
    }
    Exl3AttentionPageRanges segmented_pages_for_test() const noexcept {return exact_page_ranges_;}
    void set_exact_attention_q_shared(bool enabled) noexcept {
        exact_attention_q_shared_=enabled;
    }
    void set_exact_attention_query_pair(bool enabled) noexcept {exact_attention_query_pair_=enabled;}
    // Lane-owned counters: read only while the context is idle or on its worker.
    // Attempts include launches that later fail; no GPU completion is implied.
    std::uint64_t query_pair_launch_attempts() const noexcept {return query_pair_launch_attempts_;}
    std::uint64_t query_pair_row_attempts() const noexcept {return query_pair_row_attempts_;}
    std::uint64_t query_pair_requested_row_attempts() const noexcept {return query_pair_requested_row_attempts_;}
    std::uint64_t gqa_six_query_pair_score_launch_attempts() const noexcept {
        return gqa_six_query_pair_score_launch_attempts_;
    }
    std::uint64_t gqa_six_query_pair_score_row_attempts() const noexcept {
        return gqa_six_query_pair_score_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_value_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_value_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_value_row_attempts() const noexcept {
        return gqa_six_softmax_triple_value_row_attempts_;
    }
    std::uint64_t prefill_shared_score_launch_attempts() const noexcept {
        return prefill_shared_score_launch_attempts_;
    }
    std::uint64_t prefill_shared_score_row_attempts() const noexcept {
        return prefill_shared_score_row_attempts_;
    }
    std::uint64_t prefill_shared_score_global_bytes_eliminated() const noexcept {
        return prefill_shared_score_global_bytes_eliminated_;
    }
    std::uint64_t gqa_six_softmax_triple_v_tile_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_v_tile_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_v_tile_row_attempts() const noexcept {
        return gqa_six_softmax_triple_v_tile_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_full_cta_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_full_cta_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_full_cta_row_attempts() const noexcept {
        return gqa_six_softmax_triple_full_cta_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_threads128_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_threads128_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_threads128_row_attempts() const noexcept {
        return gqa_six_softmax_triple_threads128_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_score_tile_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_score_tile_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_score_tile_row_attempts() const noexcept {
        return gqa_six_softmax_triple_score_tile_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_key_pair_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_key_pair_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_key_pair_row_attempts() const noexcept {
        return gqa_six_softmax_triple_key_pair_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_warp_score_broadcast_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_warp_score_broadcast_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_warp_score_broadcast_row_attempts() const noexcept {
        return gqa_six_softmax_triple_warp_score_broadcast_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_pair_dimensions_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_row_attempts() const noexcept {
        return gqa_six_softmax_triple_pair_dimensions_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_six_values_single_load_launch_attempts() const noexcept {
        return gqa_six_softmax_six_values_single_load_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_six_values_single_load_row_attempts() const noexcept {
        return gqa_six_softmax_six_values_single_load_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_scalar_dim_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_scalar_dim_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_scalar_dim_row_attempts() const noexcept {
        return gqa_six_softmax_triple_scalar_dim_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_two_query_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_two_query_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_two_query_row_attempts() const noexcept {
        return gqa_six_softmax_triple_two_query_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_fused_launch_attempts() const noexcept {
        return gqa_six_softmax_triple_fused_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_triple_fused_row_attempts() const noexcept {
        return gqa_six_softmax_triple_fused_row_attempts_;
    }
    std::uint64_t gqa_six_softmax_fused_scalar_values_launch_attempts() const noexcept {
        return gqa_six_softmax_fused_scalar_values_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_fused_scalar_values_row_attempts() const noexcept {
        return gqa_six_softmax_fused_scalar_values_row_attempts_;
    }
    std::uint64_t exact_attention_gqa_six_decode_fused_calls() const noexcept {
        return exact_attention_gqa_six_decode_fused_calls_;
    }
    std::uint64_t gqa_six_softmax_tile512_launch_attempts() const noexcept {
        return gqa_six_softmax_tile512_launch_attempts_;
    }
    std::uint64_t gqa_six_softmax_tile512_row_attempts() const noexcept {
        return gqa_six_softmax_tile512_row_attempts_;
    }
    std::uint64_t gqa_six_score_k_tile64_launch_attempts() const noexcept {
        return gqa_six_score_k_tile64_launch_attempts_;
    }
    std::uint64_t gqa_six_score_k_tile64_row_attempts() const noexcept {
        return gqa_six_score_k_tile64_row_attempts_;
    }
    const char* gqa_six_score_route() const noexcept {
        if (!exact_attention_gqa_six_scores_) return "not_selected";
        if (exact_attention_gqa_six_score_k_tile64_ &&
            exact_attention_gqa_six_softmax_triple_values_)
            return "gqa_six_score_k_tile64";
        if (exact_attention_gqa_six_query_pair_scores_)
            return "gqa_six_query_pair_scores";
        return "gqa_six_score_shards";
    }
    static constexpr std::size_t gqa_six_score_k_tile64_shared_bytes() noexcept {
        return (6u*256u*sizeof(float))+(8u*8u*256u*sizeof(std::uint16_t));
    }
    const char* gqa_six_softmax_route() const noexcept {
        if (exact_attention_gqa_six_softmax_fused_scalar_values_)
            return "gqa_six_softmax_fused_scalar_values";
        if (!exact_attention_gqa_six_softmax_triple_values_)
            return "not_selected";
        if (exact_attention_gqa_six_softmax_triple_values_fused_)
            return "gqa_six_softmax_triple_values_fused";
        return exact_attention_gqa_six_softmax_tile512_
            ? "gqa_six_softmax_tile512"
            : "gqa_six_softmax_tile256";
    }
    const char* gqa_six_softmax_triple_value_route() const noexcept {
        if (exact_attention_gqa_six_softmax_fused_scalar_values_)
            return "gqa_six_softmax_fused_scalar_values";
        if (!exact_attention_gqa_six_softmax_triple_values_) return "not_selected";
        if (exact_attention_gqa_six_softmax_triple_values_v_tile64_)
            return "gqa_six_softmax_triple_values_v_tile64";
        if (exact_attention_gqa_six_softmax_triple_values_v_tile_)
            return "gqa_six_softmax_triple_values_v_tile16";
        if (exact_attention_gqa_six_softmax_triple_values_full_cta_)
            return "gqa_six_softmax_triple_values_full_cta";
        if (exact_attention_gqa_six_softmax_triple_values_threads128_)
            return "gqa_six_softmax_triple_values_threads128";
        if (exact_attention_gqa_six_softmax_triple_values_score_tile_)
            return "gqa_six_softmax_triple_values_score_tile";
        if (exact_attention_gqa_six_softmax_triple_values_key_pair_pipeline_)
            return "gqa_six_softmax_triple_values_key_pair_pipeline";
        if (exact_attention_gqa_six_softmax_triple_values_warp_score_broadcast_)
            return "gqa_six_softmax_triple_values_warp_score_broadcast";
        if (exact_attention_gqa_six_softmax_triple_values_pair_dimensions_)
            return "gqa_six_softmax_triple_values_pair_dimensions";
        if (exact_attention_gqa_six_softmax_six_values_single_load_)
            return "gqa_six_softmax_six_values_single_load";
        if (exact_attention_gqa_six_softmax_six_values_scalar_single_load_)
            return "gqa_six_softmax_six_values_scalar_single_load";
        if (exact_attention_gqa_six_softmax_triple_values_scalar_dim_)
            return "gqa_six_softmax_triple_values_scalar_dim";
        if (exact_attention_gqa_six_softmax_triple_values_two_query_)
            return "gqa_six_softmax_triple_values_two_query";
        if (exact_attention_gqa_six_softmax_triple_values_fused_)
            return "gqa_six_softmax_triple_values_fused";
        return "gqa_six_softmax_triple_values";
    }
    static constexpr std::size_t gqa_six_softmax_triple_v_tile_shared_bytes(
        std::size_t tile_keys=16u) noexcept {
        return tile_keys * 128u * sizeof(std::uint32_t);
    }
    void set_exact_attention_k_half2(bool enabled) noexcept {
        exact_attention_k_half2_=enabled;
    }
    void set_exact_attention_v_half2(bool enabled) noexcept {
        exact_attention_v_half2_=enabled;
    }
    void set_exact_attention_gqa_pair(bool enabled) noexcept {
        exact_attention_gqa_pair_=enabled;
    }
    void set_exact_attention_gqa_triple(bool enabled) noexcept {
        exact_attention_gqa_triple_=enabled;
    }
    void set_exact_attention_gqa_triple_values128(bool enabled) noexcept {
        exact_attention_gqa_triple_values128_=enabled;
    }
    void set_exact_attention_gqa_triple_softmax_staged(bool enabled) noexcept {
        exact_attention_gqa_triple_softmax_staged_=enabled;
    }
    void set_exact_attention_gqa_six(bool enabled) noexcept {
        exact_attention_gqa_six_=enabled;
    }
    void set_exact_attention_gqa_six_scores(bool enabled) noexcept {
        exact_attention_gqa_six_scores_=enabled;
    }
    void set_exact_attention_gqa_six_extent_shards(bool enabled) noexcept {
        exact_attention_gqa_six_extent_shards_=enabled;
    }
    void set_exact_attention_gqa_six_values_sharded(bool enabled) noexcept {
        exact_attention_gqa_six_values_sharded_=enabled;
    }
    void set_exact_attention_gqa_triple_values4(bool enabled) noexcept {
        exact_attention_gqa_triple_values4_=enabled;
    }
    void set_exact_attention_gqa_six_softmax_triple_values(bool enabled) noexcept {
        exact_attention_gqa_six_softmax_triple_values_=enabled;
    }
    void set_exact_attention_gqa_six_softmax_triple_values_pair_dimensions(
        bool enabled) noexcept {
        exact_attention_gqa_six_softmax_triple_values_pair_dimensions_=enabled;
    }
    void set_exact_attention_gqa_six_softmax_six_values_single_load(
        bool enabled) noexcept {
        exact_attention_gqa_six_softmax_six_values_single_load_=enabled;
    }
    void set_exact_attention_gqa_six_softmax_six_values_scalar_single_load(
        bool enabled) noexcept {
        exact_attention_gqa_six_softmax_six_values_scalar_single_load_=enabled;
    }
    void set_exact_attention_gqa_six_softmax_fused_scalar_values(
        bool enabled) noexcept {
        exact_attention_gqa_six_softmax_fused_scalar_values_=enabled;
    }
    void set_exact_attention_gqa_six_packed_triples(bool enabled) noexcept {
        exact_attention_gqa_six_packed_triples_=enabled;
    }
    void set_numeric_attention_splitk(float* workspace,
                                      std::size_t workspace_bytes,
                                      bool enabled) noexcept {
        numeric_attention_splitk_workspace_=workspace;
        numeric_attention_splitk_workspace_bytes_=workspace_bytes;
        numeric_attention_splitk_=enabled;
    }
    // capacity_splits: FP32 partial planes allocated (the FA2 prefill route
    // needs four); split_count: the WMMA32 route's split (0 when unselected).
    void set_fast_wmma32_split2_workspace(float* output,float* stats,
                                           int capacity_rows,int split_count,
                                           int capacity_splits) noexcept {
        fast_wmma32_split2_output_=output;
        fast_wmma32_split2_stats_=stats;
        fast_wmma32_split2_capacity_rows_=capacity_rows;
        fast_wmma32_split_count_=split_count;
        fast_wmma32_split2_capacity_splits_=capacity_splits;
    }

    // Decode-only graph path: kernels read the current position through this
    // stable device pointer instead of a captured host scalar.
    void set_position_device(const int* position_device) noexcept;
    // K/V cache rows an accepted sibling carries into its chain slot.
    void append_sibling_row_copies(Exl3SiblingRowCopies& out,int source,
                                   int destination) const;
    // Research media phases only. Causal/cache positions keep their original
    // logical cursor; these coordinates affect the rotary phase exclusively.
    void set_mrope_positions(const int* rows_xyz,int offset) noexcept {
        mrope_positions_=rows_xyz;rope_offset_=offset;
    }

    // E4C1 OSCAR attach: when set, this layer routes attention through the
    // resident OSCAR cache instead of the ordinary F16 path. Null disables.
    void set_oscar(Exl3OscarContext* oscar, int model_layer) noexcept;
    // E4B2 graph capture freezes launch parameters; the layer sizes cached
    // decode scratch by full capacity while this is set. Eager uses actual
    // need. Set by the owning text context around capture only.
    void set_capture_active(bool active) noexcept { capture_active_ = active; }
    // Ordinary physical-C1 full-layer graph capture may admit the native
    // segmented fused-flash attention route. Its launch grid is frozen from
    // the capture frontier, while the kernels read the live position device
    // scalar at replay. This marker is never enabled for HostKV/reference
    // graphs.
    void set_ordinary_full_layer_graph_capture(bool active) noexcept {
        ordinary_full_layer_graph_capture_ = active;
    }
    // Ordinary eager continuation only: install a private up workspace and
    // context-owned fork/join resources. Capture, prefill, instrumentation and
    // shared projection execution retain their existing paths.
    void prepare_eager_mlp_gateup_concurrency(
        Exl3MlpGateUpConcurrencyView view);
    // Ordinary eager wide prefill only. K and V borrow private workspaces and
    // auxiliary streams; the caller stream executes Q and joins both siblings.
    void prepare_prefill_qkv_concurrency(Exl3PrefillQkvConcurrencyView view);
    void set_target_q_executor(Exl3TargetQExecutor executor,int layer,bool kv=false,bool output=false,bool gateup=false,bool down=false) {
        target_q_executor_=std::move(executor);model_layer_=layer;target_kv_executor_enabled_=kv;
        target_o_executor_enabled_=output;
        target_gateup_executor_enabled_=gateup;
        target_down_executor_enabled_=down;
    }
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
    void dispatch_counts(std::uint64_t& oscar, std::uint64_t& ordinary) const noexcept {
        oscar = oscar_count_;
        ordinary = ordinary_count_;
    }

    Exl3FullAttentionLayerTrace trace() const {
        if(coalesce_input_mlp_)throw std::logic_error("coalesced attention scratch excludes intermediate trace");
        return trace_;
    }
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
    std::uint64_t target_prefill_gate_up_pair_attempts() const noexcept {
        return linear_workspaces_[4]
            ? linear_workspaces_[4]->target_prefill_gate_up_pair_attempts() : 0;
    }
    std::uint64_t target_prefill_gate_up_pair_calls() const noexcept {
        return linear_workspaces_[4]
            ? linear_workspaces_[4]->target_prefill_gate_up_pair_calls() : 0;
    }
    std::uint64_t target_prefill_gate_up_pair_rows() const noexcept {
        return linear_workspaces_[4]
            ? linear_workspaces_[4]->target_prefill_gate_up_pair_rows() : 0;
    }
    std::uint64_t k7_tiles64_exact_splits_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->k7_tiles64_exact_splits_calls();return result;
    }
    std::uint64_t target_k8_kv_prefill_async_a_calls() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->target_k8_kv_prefill_async_a_calls();return result;
    }
    std::uint64_t target_k8_kv_prefill_async_a_rows() const noexcept {
        std::uint64_t result=0;for(auto* workspace:linear_workspaces_)if(workspace)result+=workspace->target_k8_kv_prefill_async_a_rows();return result;
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
    Exl3PrefillProjectionChainGraph::Snapshot prefill_projection_chain_graph_stats() const noexcept {
        return prefill_projection_chain_graph_.snapshot();
    }
    Exl3PrefillAttentionChainGraph::Snapshot prefill_attention_chain_graph_stats() const noexcept {
        return prefill_attention_chain_graph_.snapshot();
    }
    void complete_prefill_projection_chain_graph_after_drain(cudaStream_t stream);
    std::uint64_t eager_mlp_gateup_concurrent_calls() const noexcept {
        return eager_mlp_gateup_concurrent_calls_;
    }
    std::uint64_t prefill_qkv_concurrent_calls() const noexcept {
        return prefill_qkv_concurrent_calls_;
    }
    std::uint64_t prefill_qkv_concurrent_rows() const noexcept {
        return prefill_qkv_concurrent_rows_;
    }
    std::uint64_t fused_gate_up_submissions() const noexcept {
        return fused_gate_up_submissions_;
    }
    std::uint64_t fast_same_weights_fp16kv_m1_gate_up_pair_submissions() const noexcept {
        return fast_same_weights_fp16kv_m1_gate_up_pair_submissions_;
    }
    std::uint64_t fast_same_weights_fp16kv_m1_kv_pair_submissions() const noexcept {
        return fast_same_weights_fp16kv_m1_kv_pair_submissions_;
    }
    void set_native_continuation16(bool enabled) noexcept {
        for (auto* workspace : linear_workspaces_)
            if (workspace) workspace->set_native_continuation16(enabled);
    }
    void set_reconstructed_exact(Exl3ReconstructedExactView view) noexcept {
        for (auto* workspace : linear_workspaces_)
            if (workspace) workspace->set_reconstructed_exact(view);
    }
    const Exl3FullAttentionLayerTimings& last_timings() const noexcept { return timings_; }
    std::size_t workspace_bytes() const noexcept { return workspace_bytes_; }
    static std::size_t fixed_owner_metadata_required() noexcept;
    std::size_t fixed_owner_metadata_bytes() const noexcept;
    // Borrowed child pointers for enclosing-context lifetime inventory only.
    std::array<Exl3CudaLinearWorkspace*,7> linear_workspace_owners() const noexcept {return linear_workspaces_;}
    std::array<Exl3LayerBufferRetirement*,18> buffer_retirement_owners() noexcept {
        std::array<Exl3LayerBufferRetirement*,18> result{};
        for(std::size_t i=0;i<result.size();++i)if(buffer_retirements_[i] && buffer_retirements_[i]->bytes())result[i]=&*buffer_retirements_[i];
        return result;
    }
    // Configuration-only owned device bytes; excludes supplied borrowed storage.
    static std::size_t workspace_bytes_required(int rows,bool borrowed_accumulation,
        bool borrowed_transform,bool borrowed_scratch,bool coalesce_input_mlp);

private:
    void forward_mlp_tail(std::uint16_t* output,int rows,int position,
        cudaStream_t stream,bool profile,bool preserve_m1_topology,
        bool wide_prefill,cudaEvent_t* starts,cudaEvent_t* ends);
    void ensure_fast_cublas_attention_resources(int capacity);
    Exl3FullAttentionLayerWeights weights_{};
    int max_rows_ = 0;
    std::size_t workspace_bytes_ = 0;
    std::array<std::uint16_t*, 18> buffers_{};
    std::array<bool, 18> buffer_owned_{};
    std::array<std::optional<Exl3LayerBufferRetirement>,18> buffer_retirements_{};
    std::array<Exl3CudaLinearWorkspace*, 7> linear_workspaces_{};
    Exl3CudaReconstructGemmWorkspace* reconstruct_gemm_ = nullptr;
    std::uint16_t* k_cache_ = nullptr;
    std::uint16_t* v_cache_ = nullptr;
    int cache_capacity_ = 0;
    std::uint16_t* direct_staged_k_=nullptr;
    std::uint16_t* direct_staged_v_=nullptr;
    int direct_staged_rows_=0;
    const std::uint16_t* exact_prefix_k_=nullptr;
    const std::uint16_t* exact_prefix_v_=nullptr;
    int exact_prefix_rows_=0;
    int exact_prefix_first_=0;
    Exl3AttentionPageRanges exact_page_ranges_;
    Exl3AttentionPositionContract exact_position_contract_;
    float* exact_scores_ = nullptr;
    int exact_score_rows_ = 0;
    bool exact_attention_q_shared_ = false;
    bool exact_attention_query_pair_ = false;
    bool coalesce_input_mlp_=false;
    std::uint64_t query_pair_launch_attempts_=0,query_pair_row_attempts_=0;
    std::uint64_t query_pair_requested_row_attempts_=0;
    bool exact_attention_k_half2_ = false;
    bool exact_attention_v_half2_ = false;
    bool exact_attention_gqa_pair_ = false;
    bool exact_attention_gqa_triple_ = false;
    bool exact_attention_gqa_triple_values128_ = false;
    bool exact_attention_gqa_triple_softmax_staged_ = false;
    bool exact_attention_gqa_six_ = false;
    bool exact_attention_gqa_six_scores_ = false;
    bool exact_attention_gqa_six_extent_shards_ = false;
    bool exact_attention_gqa_six_values_sharded_ = false;
    bool exact_attention_gqa_triple_values4_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_v_tile_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_v_tile64_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_full_cta_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_threads128_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_score_tile_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_key_pair_pipeline_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_warp_score_broadcast_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_pair_dimensions_ = false;
    bool exact_attention_gqa_six_softmax_six_values_single_load_ = false;
    bool exact_attention_gqa_six_softmax_six_values_scalar_single_load_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_scalar_dim_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_two_query_ = false;
    bool exact_attention_gqa_six_softmax_triple_values_fused_ = false;
    bool exact_attention_gqa_six_softmax_tile512_ = false;
    bool exact_attention_gqa_six_softmax_fused_scalar_values_ = false;
    bool exact_attention_gqa_six_score_k_tile64_ = false;
    bool exact_attention_gqa_six_packed_triples_ = false;
    bool exact_attention_gqa_six_query_pair_scores_ = false;
    std::uint64_t gqa_six_query_pair_score_launch_attempts_=0;
    std::uint64_t gqa_six_query_pair_score_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_value_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_value_row_attempts_=0;
    std::uint64_t prefill_shared_score_launch_attempts_=0;
    std::uint64_t prefill_shared_score_row_attempts_=0;
    std::uint64_t prefill_shared_score_global_bytes_eliminated_=0;
    std::uint64_t gqa_six_softmax_triple_v_tile_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_v_tile_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_full_cta_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_full_cta_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_threads128_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_threads128_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_score_tile_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_score_tile_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_key_pair_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_key_pair_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_warp_score_broadcast_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_warp_score_broadcast_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_pair_dimensions_row_attempts_=0;
    std::uint64_t gqa_six_softmax_six_values_single_load_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_six_values_single_load_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_scalar_dim_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_scalar_dim_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_two_query_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_two_query_row_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_fused_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_triple_fused_row_attempts_=0;
    std::uint64_t gqa_six_softmax_tile512_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_tile512_row_attempts_=0;
    std::uint64_t gqa_six_softmax_fused_scalar_values_launch_attempts_=0;
    std::uint64_t gqa_six_softmax_fused_scalar_values_row_attempts_=0;
    std::uint64_t gqa_six_score_k_tile64_launch_attempts_=0;
    std::uint64_t gqa_six_score_k_tile64_row_attempts_=0;
    bool fast_fused_flash_attention_ = false;
    bool fast_fused_flash_attention_keys256_ = false;
    bool fast_fused_flash_multirow_ = false;
    bool fast_whole_context_fused_attention_ = false;
    bool fast_prefill_tiled_attention_ = false;
    bool fast_prefill_rows4_attention_ = false;
    bool fast_prefill_rows8_attention_ = false;
    bool fast_prefill_rows8_threads128_ = false;
    bool fast_prefill_wmma_attention_ = false;
    bool fast_prefill_wmma32_attention_ = false;
    bool fast_prefill_wmma64_attention_ = false;
    bool fast_prefill_wmma64_register_attention_ = false;
    bool fast_prefill_wmma64_register_keys64_attention_ = false;
    bool fast_prefill_wmma64_shared_heads_attention_ = false;
    bool fast_prefill_wmma32_padded_attention_ = false;
    bool fast_prefill_rows2_attention_ = false;
    bool fast_online_decode_attention_ = false;
    bool fast_online_decode_attention_warp_heads_ = false;
    bool fast_cublas_attention_ = false;
    void* fast_cublas_handle_ = nullptr;
    float* fast_cublas_scores_ = nullptr;
    std::uint16_t* fast_cublas_probs_ = nullptr;
    int fast_cublas_capacity_ = 0;
    bool fast_same_weights_fp16kv_m1_gate_up_pair_ = false;
    bool fast_same_weights_fp16kv_m1_kv_pair_ = false;
    bool fast_same_weights_fp16kv_m1_kv_wide_pair_ = false;
    bool fast_same_weights_fp16kv_m1_kv_pair_graph_ = false;
    bool exact_attention_gqa_six_decode_fused_ = false;
    std::uint64_t exact_attention_gqa_six_decode_fused_calls_ = 0;
    float* numeric_attention_splitk_workspace_ = nullptr;
    std::size_t numeric_attention_splitk_workspace_bytes_ = 0;
    bool numeric_attention_splitk_ = false;
    float* fast_wmma32_split2_output_ = nullptr;
    float* fast_wmma32_split2_stats_ = nullptr;
    int fast_wmma32_split2_capacity_rows_ = 0;
    int fast_wmma32_split_count_ = 0;
    int fast_wmma32_split2_capacity_splits_ = 0;
    const int* position_device_ = nullptr;
    const int* mrope_positions_ = nullptr;
    int rope_offset_ = 0;
    Exl3OscarContext* oscar_ = nullptr;
    bool capture_active_ = false;
    bool ordinary_full_layer_graph_capture_ = false;
    bool small_m_fused_gate_up_transform_ = false;
    std::uint64_t fused_gate_up_submissions_ = 0;
    std::uint64_t fast_same_weights_fp16kv_m1_gate_up_pair_submissions_ = 0;
    std::uint64_t fast_same_weights_fp16kv_m1_kv_pair_submissions_ = 0;
    Exl3MlpGateUpConcurrencyView eager_mlp_gateup_concurrency_{};
    std::uint64_t eager_mlp_gateup_concurrent_calls_ = 0;
    Exl3PrefillQkvConcurrencyView prefill_qkv_concurrency_{};
    std::uint64_t prefill_qkv_concurrent_calls_ = 0;
    std::uint64_t prefill_qkv_concurrent_rows_ = 0;
    std::shared_ptr<const void> prefill_projection_chain_context_owner_ =
        std::make_shared<int>(1);
    std::shared_ptr<const void> prefill_projection_chain_model_owner_ =
        std::make_shared<int>(2);
    std::shared_ptr<const void> prefill_projection_chain_scratch_owner_ =
        std::make_shared<int>(3);
    Exl3PrefillProjectionChainGraph prefill_projection_chain_graph_;
    // Declared after the owner tokens so graph handles retire before their
    // bound context/model/scratch lifetime witnesses.
    Exl3PrefillAttentionChainGraph prefill_attention_chain_graph_;
    Exl3TargetProjectionTiming* projection_timing_ = nullptr;
    Exl3TargetProjectionObserver projection_observer_ = nullptr;
    void* projection_observer_user_ = nullptr;
    Exl3TargetProjectionObserverSelection projection_observer_selection_ =
        Exl3TargetProjectionObserverSelection::gate_up_k6;
    int model_layer_ = -1;
    Exl3TargetQExecutor target_q_executor_;
    bool target_kv_executor_enabled_=false;
    bool target_o_executor_enabled_=false;
    bool target_gateup_executor_enabled_=false;
    Exl3ActivationLifetime mlp_activation_lifetime_;
    bool target_down_executor_enabled_=false;
    int oscar_layer_ = -1;
    std::uint64_t oscar_count_ = 0;
    std::uint64_t ordinary_count_ = 0;
    bool retained_prefix_available_ = false;
    int retained_prefix_rows_ = 0;
    int retained_prefix_position_ = 0;
    cudaStream_t retained_prefix_stream_ = nullptr;
    Exl3FullAttentionLayerTrace trace_{};
    Exl3FullAttentionLayerTimings timings_{};
};

} // namespace ninfer::exl3
