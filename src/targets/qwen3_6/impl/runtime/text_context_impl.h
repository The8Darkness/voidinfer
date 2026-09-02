#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/arena.h"
#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/visual_scatter.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

bool oscar_d4_profile_enabled() noexcept {
    const char* value = std::getenv("NINFER_OSCAR_D4_1_PROFILE");
    return value != nullptr && value[0] == '1';
}

double oscar_d4_elapsed_us(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) noexcept {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

bool fp8_greedy_argmax_enabled() {
    const char* value = std::getenv("NINFER_FP8_GREEDY_ARGMAX");
    return value == nullptr || value[0] != '0';
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

std::uint16_t runtime_float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Match CUDA's __float2bfloat16 round-to-nearest-even conversion used by the qualified
    // rotation path, rather than truncating the FP32 mantissa.
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

float runtime_bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features.data == nullptr;
    const bool batch   = batch_features.data != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features.data != nullptr ? &batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features.data != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features.slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features.data != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                         qwen3_6::PagedKVCacheView kv, LinearAttentionStatePool& state,
                         qwen3_6::RoundState& io, Tensor& prefill_hidden,
                         std::uint32_t prefill_chunk, std::uint32_t text_kv_base,
                         qwen3_6::PagedKVCacheView mtp_kv,
                         const qwen3_6::PagedKVCache* batch_text_kv,
                         const qwen3_6::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(mtp_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base),
      batch_text_kv_(batch_text_kv), batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    oscar_rotations_ = oscar_internal::rotation_set_from_environment();
    if (oscar_internal::matched_fp32_mode_enabled()) {
        if (batch_text_kv_ == nullptr) {
            throw std::logic_error("OSCAR matched FP32 mode requires a text KV owner");
        }
        matched_fp32_cache_ = oscar_internal::matched_fp32_cache_for(
            batch_text_kv_, static_cast<std::uint32_t>(batch_text_kv_->max_context()));
    }
    if (oscar_internal::live_int2_reference_mode_enabled() ||
        oscar_internal::live_int2_gpu_mode_enabled()) {
        if (batch_text_kv_ == nullptr || !oscar_rotations_) {
            throw std::logic_error(
                "OSCAR live INT2 mode requires a text KV owner and rotation assets");
        }
        live_mixed_cache_ = oscar_internal::live_mixed_cache_for(
            batch_text_kv_, static_cast<std::uint32_t>(batch_text_kv_->max_context()),
            oscar_rotations_);
    }
    set_linear_state_slots(0, 0);
    bind();
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::bind() {
    using TargetBindings = LoadedModelData;
    using TargetMlp      = MlpWeights;
    const auto bind_mlp  = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;
    if (weights_.optimized_proposal) {
        const auto& proposal = *weights_.optimized_proposal;
        set_proposal_head(&proposal.head, static_cast<const std::int32_t*>(proposal.token_ids.data),
                          proposal.head.n);
    }

    if (mtp_enabled()) {
        if (!weights_.mtp) {
            throw std::invalid_argument("MTP state was enabled without materialized MTP weights");
        }
        const auto& source = *weights_.mtp;
        mtp_               = MtpW{&source,
                    &source.input_projection,
                    &source.embedding_norm,
                    &source.hidden_norm,
                    &source.input_norm,
                    &source.query_norm,
                    &source.key_norm,
                    &source.output,
                    &source.post_attention_norm,
                    &source.final_norm};
    }

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            FullLayerW& out = full_[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            const auto& source =
                weights_.full_layers[static_cast<std::size_t>(ModelConfig::full_idx(layer))];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const std::size_t gidx = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
            GdnLayerW& out         = gdn_[gidx];
            const auto& source     = weights_.gdn_layers[gidx];
            out.input_norm         = &source.input_norm;
            out.projection         = &source.projection;
            out.conv1d             = &source.convolution;
            out.gdn_norm           = &source.norm;
            out.out_proj           = &source.output;
            out.post_attn_norm     = &source.post_attention_norm;
            out.mlp                = bind_mlp(source.post_mixer);
        }
    }
}

const MtpW& TextContext::mtp_weights() const {
    if (!mtp_enabled()) { throw std::runtime_error("MTP draft weights are not enabled"); }
    return mtp_;
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({kCfg.hidden, T});

    auto roots = workspace_recipe::mtp_stem<TextConfig>(work_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 || input_embeddings->ne[0] != kCfg.hidden ||
            input_embeddings->numel() != static_cast<std::int64_t>(kCfg.hidden) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({kCfg.hidden, T});
    } else {
        emb = roots.embedding;
        ops::embedding(flat_ids, *embed_, emb, s);
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, *mtp_.pre_fc_norm_embedding, kCfg.rms_eps, true, e, s);
    ops::rmsnorm(flat_hidden, *mtp_.pre_fc_norm_hidden, kCfg.rms_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    ops::linear(fc_in, *mtp_.fc, x, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, *mtp_.input_norm, kCfg.rms_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace_recipe::mtp_attention_projection<TextConfig>(work_, T);
    Tensor q              = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k              = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor gate           = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor v              = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat         = q.view({kCfg.q_size, T});
    Tensor gate_flat      = gate.view({kCfg.q_size, T});
    Tensor k_flat         = k.view({kCfg.kv_size, T});
    Tensor v_flat         = v.view({kCfg.kv_size, T});
    Variant::mtp_attention_projection(ah, mtp_.payload->attention, q_flat, gate_flat, k_flat,
                                      v_flat, work_, s);

    const auto results = workspace_recipe::mtp_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);

    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor k_batch        = kn.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor v_batch        = v.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
        Tensor a_batch        = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
            *active_backend_kv_table_rows_, {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::causal_softmax_attention(qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row,
                                      {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
                                      batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace_recipe::mtp_post_attention<TextConfig>(work_, T);
    Tensor o        = post.output;
    ops::linear(a.view({kCfg.q_size, T}), *mtp_.o_proj, o, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x, work_, s);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({kCfg.hidden, T});
    ops::rmsnorm(x, *mtp_.norm, kCfg.rms_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {kCfg.hidden, 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ah_last = work_.alloc(DType::BF16, {kCfg.hidden, 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Tensor v_flat = work_.alloc(DType::BF16, {kCfg.kv_size, T});
        Variant::mtp_kv_projection(ah, mtp_.payload->attention, k_flat, v_flat, work_, s);
        Tensor k  = k_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor v  = v_flat.view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor kn = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_kv, T});
        ops::rmsnorm(k, *mtp_.k_norm, kCfg.rms_eps, true, kn, s);
        ops::rope(rope_positions, kCfg.rotary_dim, kCfg.rope_theta, kn, s);
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(kCfg.hidden) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat    = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Tensor gate_flat = work_.alloc(DType::BF16, {kCfg.q_size, 1});
        Variant::mtp_q_gate_projection(ah_last, mtp_.payload->attention, q_flat, gate_flat, work_,
                                       s);
        Tensor q    = q_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor gate = gate_flat.view({kCfg.head_dim, kCfg.n_q, 1});
        Tensor qn   = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::rmsnorm(q, *mtp_.q_norm, kCfg.rms_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        ops::rope(last_rope_position, kCfg.rotary_dim, kCfg.rope_theta, qn, s);

        Tensor a = work_.alloc(DType::BF16, {kCfg.head_dim, kCfg.n_q, 1});
        ops::causal_softmax_attention_cached(qn, last_position,
                                             {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
                                             mtp_kv_.layer_view(0), envelope, work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::linear(a.view({kCfg.q_size, 1}), *mtp_.o_proj, o, s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {kCfg.hidden, 1});
        ops::rmsnorm(x_last, *mtp_.post_attn_norm, kCfg.rms_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            Variant::mtp_post_mixer(mh, mtp_.payload->post_mixer, x_last, work_, s);
        }
        ops::rmsnorm(x_last, *mtp_.norm, kCfg.rms_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    const int T = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, kCfg.vocab, T, "proposal logits");
    if (proposal_head_ != nullptr) {
        Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
        ops::linear(hidden, *proposal_head_, proposal_logits, ctx_.stream);
        ops::argmax(proposal_logits, proposal_tokens, proposal_head_n_, ctx_.stream);
        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                      ctx_.stream);
    } else {
        Tensor output_logits = matrix_window(logits, T);
        ops::linear(hidden, *lm_head_, output_logits, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, T}, "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {kCfg.vocab, 1}, "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, 1}, "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, stream);
        ops::linear(hidden, *lm_head_, logits, stream);
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           bool greedy_target_head, Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {kCfg.hidden, columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({kCfg.hidden, columns});
        Tensor flat_logits = logits.view({kCfg.vocab, columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, flat_hidden, stream);
        const bool use_greedy_fp8_head =
            greedy_target_head && fp8_greedy_argmax_enabled() && columns <= 48 &&
            lm_head_->qtype == QType::FP8_E4M3FN_ROW_BF16S && lm_head_->n == kCfg.vocab &&
            lm_head_->k == kCfg.hidden;
        if (use_greedy_fp8_head) {
            ops::linear_argmax(flat_hidden, *lm_head_, flat_tokens, kCfg.token_domain, flat_logits,
                               stream);
        } else {
            ops::linear(flat_hidden, *lm_head_, flat_logits, stream);
            ops::argmax(flat_logits, flat_tokens, kCfg.token_domain, stream);
        }
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      bool greedy_target_head) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             greedy_target_head, tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink, bool greedy_target_head) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             greedy_target_head, sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {kCfg.hidden, width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void dump_runtime_attention_diagnostic(std::int32_t layer, const char* label,
                                       const Tensor& value, cudaStream_t stream);
void dump_runtime_layer_stage_diagnostic(std::int32_t layer, const char* label,
                                         const Tensor& value, cudaStream_t stream);
void dump_runtime_position_diagnostic(std::int32_t layer, const Tensor& value,
                                      cudaStream_t stream);
void dump_runtime_matched_cache_diagnostic(
    std::int32_t layer, std::int32_t full_layer, const oscar_internal::OscarMatchedFP32Cache& cache,
    const Tensor& positions, cudaStream_t stream);

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int layer, int fidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    if (ph == Phase::Prefill && oscar_internal::matched_fp32_mode_enabled()) {
        // D1.3 uses one deterministic prefill dump followed by step-tagged ordinary decode
        // dumps. The tag is thread-local because TextContext executes on the runtime worker.
        oscar_internal::set_matched_diagnostic_step(0);
    }
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const auto projection = workspace_recipe::text_attention_projection<TextConfig>(work_, T);
    Tensor h              = projection.hidden;
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q         = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor gate      = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor k         = projection.key.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor v         = projection.value.view({kCfg.head_dim, kCfg.n_kv, T});
    Tensor q_flat    = q.view({kCfg.q_size, T});
    Tensor gate_flat = gate.view({kCfg.q_size, T});
    Tensor k_flat    = k.view({kCfg.kv_size, T});
    Tensor v_flat    = v.view({kCfg.kv_size, T});
    Variant::attention_projection(h, *w.projection, q_flat, gate_flat, k_flat, v_flat, ph, work_,
                                  s);

    const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
    Tensor qn          = results.normalized_query.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor kn          = results.normalized_key.view({kCfg.head_dim, kCfg.n_kv, T});
    ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    ops::rope(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s);
    dump_runtime_attention_diagnostic(layer, "q", qn, s);
    dump_runtime_attention_diagnostic(layer, "k", kn, s);
    dump_runtime_attention_diagnostic(layer, "v", v, s);
    const char* capture_armed = std::getenv("NINFER_OSCAR_QKV_CAPTURE_ARMED");
    if (ph == Phase::Prefill && oscar_capture_ == nullptr && capture_armed != nullptr &&
        capture_armed[0] == '1') {
        oscar_capture_ = oscar_internal::qkv_capture_from_environment();
    }
    if (ph == Phase::Prefill && oscar_capture_ != nullptr) {
        oscar_capture_->capture(layer, fidx, qn, kn, v, s);
    }

    const bool live_gpu = live_mixed_cache_ != nullptr &&
                          oscar_internal::live_int2_gpu_mode_enabled();
    const bool live_gpu_resident = live_mixed_cache_ != nullptr &&
                                   oscar_internal::live_int2_gpu_resident_mode_enabled();
    const bool live_reference = live_mixed_cache_ != nullptr && !live_gpu;
    const bool matched_fp32 = matched_fp32_cache_ != nullptr;
    const bool d4_profile = oscar_d4_profile_enabled();
    const auto live_full_attention_start = d4_profile && (live_reference || live_gpu)
                                               ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
    Tensor a = results.attention.view({kCfg.head_dim, kCfg.n_q, T});
    Tensor attention_for_output = a;
    if (live_gpu) {
        if ((active_sequence_batch_ != 0 &&
             (active_sequence_batch_ != 1 || active_sequence_width_ != 1)) ||
            (active_sequence_batch_ == 0 && T <= 0)) {
            throw std::logic_error(
                "OSCAR live INT2 GPU path currently supports one sequence only");
        }
        if (cache_positions.dtype != DType::I32 || !cache_positions.is_contiguous() ||
            cache_positions.ne[0] != T || cache_positions.ne[1] != 1 || cache_positions.ne[2] != 1 ||
            cache_positions.ne[3] != 1) {
            throw std::logic_error("OSCAR live INT2 GPU positions must be [T]");
        }
        if (!oscar_rotations_) {
            throw std::logic_error("OSCAR live INT2 GPU path has no rotation assets");
        }

        std::vector<std::int32_t> positions(static_cast<std::size_t>(T));
        std::vector<float> rotated_k;
        std::vector<float> rotated_v;
        if (!live_gpu_resident) {
            rotated_k.resize(static_cast<std::size_t>(kCfg.kv_size) * T);
            rotated_v.resize(static_cast<std::size_t>(kCfg.kv_size) * T);
        }
        // The normal work arena is sized for the production BF16 path and deliberately
        // rejects these diagnostic FP32 temporaries. Keep this first live integration
        // self-contained in temporary device allocations; D4.4 can replace them with a
        // persistent per-request scratch plan after the correctness gate.
        ::ninfer::DeviceBuffer fp32_q_storage(
            static_cast<std::size_t>(kCfg.q_size) * T * sizeof(float));
        ::ninfer::DeviceBuffer fp32_k_storage(
            static_cast<std::size_t>(kCfg.kv_size) * T * sizeof(float));
        ::ninfer::DeviceBuffer fp32_v_storage(
            static_cast<std::size_t>(kCfg.kv_size) * T * sizeof(float));
        Tensor fp32_q(fp32_q_storage.p, DType::FP32, {kCfg.q_size, T});
        Tensor fp32_k(fp32_k_storage.p, DType::FP32, {kCfg.kv_size, T});
        Tensor fp32_v(fp32_v_storage.p, DType::FP32, {kCfg.kv_size, T});
        fp32_q = fp32_q.view({kCfg.head_dim, kCfg.n_q, T});
        fp32_k = fp32_k.view({kCfg.head_dim, kCfg.n_kv, T});
        fp32_v = fp32_v.view({kCfg.head_dim, kCfg.n_kv, T});
        const auto qkv_rotation_start = d4_profile ? std::chrono::steady_clock::now()
                                                    : std::chrono::steady_clock::time_point{};
        oscar_rotations_->rotate_qkv_fp32(layer, qn, kn, v, fp32_q, fp32_k, fp32_v, s);
        CUDA_CHECK(cudaMemcpyAsync(positions.data(), cache_positions.data,
                                   positions.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                   s));
        if (!live_gpu_resident) {
            CUDA_CHECK(cudaMemcpyAsync(rotated_k.data(), fp32_k.data,
                                       rotated_k.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                       s));
            CUDA_CHECK(cudaMemcpyAsync(rotated_v.data(), fp32_v.data,
                                       rotated_v.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                       s));
        }
        CUDA_CHECK(cudaStreamSynchronize(s));
        for (int token = 1; token < T; ++token) {
            if (positions[token] != positions[0] + token) {
                throw std::logic_error("OSCAR live GPU positions are not a contiguous append");
            }
        }
        if (live_gpu_resident) {
            live_mixed_cache_->record_gpu_incremental_host_device_bytes(
                positions.size() * sizeof(std::int32_t));
        }
        if (d4_profile) {
            live_mixed_cache_->record_qkv_rotation_us(
                oscar_d4_elapsed_us(qkv_rotation_start, std::chrono::steady_clock::now()));
        }
        if (live_gpu_resident) {
            live_mixed_cache_->append_gpu(layer, static_cast<std::uint32_t>(positions[0]), fp32_k,
                                          fp32_v, s);
        } else {
            for (const float value : rotated_k) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("OSCAR live GPU rotated K is NaN/Inf");
                }
            }
            for (const float value : rotated_v) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error("OSCAR live GPU rotated V is NaN/Inf");
                }
            }
            std::vector<std::uint16_t> k_row(static_cast<std::size_t>(kCfg.n_kv) * kCfg.head_dim);
            std::vector<std::uint16_t> v_row(k_row.size());
            for (int token = 0; token < T; ++token) {
                for (int head = 0; head < kCfg.n_kv; ++head) {
                    for (int dimension = 0; dimension < kCfg.head_dim; ++dimension) {
                        const std::size_t source = static_cast<std::size_t>(dimension) +
                                                   static_cast<std::size_t>(kCfg.head_dim) *
                                                       (static_cast<std::size_t>(head) +
                                                        static_cast<std::size_t>(kCfg.n_kv) * token);
                        const std::size_t row = static_cast<std::size_t>(head) * kCfg.head_dim +
                                                dimension;
                        k_row[row] = runtime_float_to_bf16(rotated_k[source]);
                        v_row[row] = runtime_float_to_bf16(rotated_v[source]);
                    }
                }
                live_mixed_cache_->append(layer, static_cast<std::uint32_t>(positions[token]), k_row,
                                          v_row);
            }
            const auto staging_start = d4_profile ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
            live_mixed_cache_->prepare_gpu(layer, s);
            if (d4_profile) {
                // prepare_gpu records its own cache-copy timing; this local marker is intentionally
                // retained only as a synchronization boundary for the kernel timing below.
                (void)staging_start;
            }
        }
        ::ninfer::DeviceBuffer fp32_attention_storage(
            static_cast<std::size_t>(kCfg.q_size) * T * sizeof(float));
        Tensor fp32_attention(fp32_attention_storage.p, DType::FP32, {kCfg.q_size, T});
        fp32_attention = fp32_attention.view({kCfg.head_dim, kCfg.n_q, T});
        const auto gpu_start = d4_profile ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
        if (live_gpu_resident && ph == Phase::Prefill && T > 1) {
            const std::uint32_t query_block =
                oscar_internal::live_gpu_prefill_query_block_size();
            for (std::uint32_t token = 0; token < static_cast<std::uint32_t>(T);) {
                const std::uint32_t count = std::min(
                    query_block, static_cast<std::uint32_t>(T) - token);
                live_mixed_cache_->attention_gpu_batch(
                    layer, static_cast<std::uint32_t>(positions[token]),
                    fp32_q.slice(2, static_cast<std::int32_t>(token),
                                 static_cast<std::int32_t>(count)),
                    fp32_attention.slice(2, static_cast<std::int32_t>(token),
                                         static_cast<std::int32_t>(count)),
                    count, s);
                live_mixed_cache_->record_gpu_prefill_batch(count);
                token += count;
            }
        } else {
            for (int token = 0; token < T; ++token) {
                live_mixed_cache_->attention_gpu(
                    layer, static_cast<std::uint32_t>(positions[token]),
                    fp32_q.slice(2, token, 1), fp32_attention.slice(2, token, 1), s);
                if (live_gpu_resident) {
                    if (ph == Phase::Prefill) {
                        live_mixed_cache_->record_gpu_prefill_batch(1);
                    } else {
                        live_mixed_cache_->record_gpu_decode_batch(1);
                    }
                }
            }
        }
        if (d4_profile) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            live_mixed_cache_->record_gpu_mixed_kernel_us(
                oscar_d4_elapsed_us(gpu_start, std::chrono::steady_clock::now()));
        }

        ::ninfer::DeviceBuffer fp32_recovered_storage(
            static_cast<std::size_t>(kCfg.q_size) * T * sizeof(float));
        Tensor fp32_recovered(fp32_recovered_storage.p, DType::FP32, {kCfg.q_size, T});
        fp32_recovered = fp32_recovered.view({kCfg.head_dim, kCfg.n_q, T});
        const auto recovery_start = d4_profile ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point{};
        oscar_rotations_->inverse_value_fp32(layer, fp32_attention, fp32_recovered, s);
        if (d4_profile) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            live_mixed_cache_->record_gpu_recovery_us(
                oscar_d4_elapsed_us(recovery_start, std::chrono::steady_clock::now()));
        }

        const char* validate_reference_d43 = std::getenv("NINFER_OSCAR_D4_3_VALIDATE_REFERENCE");
        const char* validate_reference_d44 = std::getenv("NINFER_OSCAR_D4_4_VALIDATE_REFERENCE");
        const char* validate_reference_d45 = std::getenv("NINFER_OSCAR_D4_5_VALIDATE_REFERENCE");
        const char* validate_reference_d46 = std::getenv("NINFER_OSCAR_D4_6_VALIDATE_REFERENCE");
        const bool d45_reference = validate_reference_d45 != nullptr &&
                                   validate_reference_d45[0] == '1';
        const bool d46_reference = validate_reference_d46 != nullptr &&
                                   validate_reference_d46[0] == '1';
        if (((validate_reference_d43 != nullptr && validate_reference_d43[0] == '1') ||
             (validate_reference_d44 != nullptr && validate_reference_d44[0] == '1') ||
             d45_reference || d46_reference) &&
            (layer == 3 || layer == 35 || layer == 63)) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            const auto metrics = [](std::span<const float> expected,
                                    std::span<const float> actual) {
                if (expected.size() != actual.size() || expected.empty()) {
                    throw std::logic_error("OSCAR D4.5 validation shape mismatch");
                }
                double abs_sum = 0.0;
                double diff2 = 0.0;
                double ref2 = 0.0;
                float max_abs = 0.0F;
                for (std::size_t index = 0; index < expected.size(); ++index) {
                    const float diff = std::abs(expected[index] - actual[index]);
                    max_abs = std::max(max_abs, diff);
                    abs_sum += diff;
                    diff2 += static_cast<double>(diff) * diff;
                    ref2 += static_cast<double>(expected[index]) * expected[index];
                }
                return std::array<double, 3>{max_abs, abs_sum / expected.size(),
                    std::sqrt(diff2 / std::max(ref2, std::numeric_limits<double>::min()))};
            };
            std::vector<int> validation_tokens{T - 1};
            if (d45_reference || d46_reference) {
                constexpr std::array<std::uint32_t, 13> kD45BoundaryQueries{
                    63U, 64U, 68U, 319U, 320U, 321U, 322U, 323U, 324U, 325U, 326U, 327U,
                    331U};
                for (int candidate = 0; candidate < T; ++candidate) {
                    if (std::find(kD45BoundaryQueries.begin(), kD45BoundaryQueries.end(),
                                  static_cast<std::uint32_t>(positions[candidate])) !=
                        kD45BoundaryQueries.end()) {
                        validation_tokens.push_back(candidate);
                    }
                }
                std::sort(validation_tokens.begin(), validation_tokens.end());
                validation_tokens.erase(
                    std::unique(validation_tokens.begin(), validation_tokens.end()),
                    validation_tokens.end());
            }
            for (const int token : validation_tokens) {
                std::vector<std::uint16_t> q_bits(static_cast<std::size_t>(kCfg.q_size));
                std::vector<float> gpu_rotated(static_cast<std::size_t>(kCfg.q_size));
                std::vector<float> gpu_recovered(static_cast<std::size_t>(kCfg.q_size));
                const Tensor q_current = qn.slice(2, token, 1);
                const Tensor raw_current = fp32_attention.slice(2, token, 1);
                const Tensor recovered_current = fp32_recovered.slice(2, token, 1);
                CUDA_CHECK(cudaMemcpy(q_bits.data(), q_current.data,
                                      q_bits.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(gpu_rotated.data(), raw_current.data,
                                      gpu_rotated.size() * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(gpu_recovered.data(), recovered_current.data,
                                      gpu_recovered.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                std::vector<float> q_original(q_bits.size());
                for (std::size_t index = 0; index < q_bits.size(); ++index) {
                    q_original[index] = runtime_bf16_to_float(q_bits[index]);
                }
                const auto trace = live_mixed_cache_->attention(
                    layer, static_cast<std::uint32_t>(positions[token]), q_original);
                const auto av_error = metrics(trace.rotated_av, gpu_rotated);
                const auto recovered_error = metrics(trace.recovered_output, gpu_recovered);
                std::cerr << (d46_reference ? "OSCAR D4.6" :
                              (d45_reference ? "OSCAR D4.5" : "OSCAR D4.3"))
                          << " live/reference layer=" << layer
                          << " query=" << positions[token]
                          << " rotated_av_max_abs=" << av_error[0]
                          << " rotated_av_mean_abs=" << av_error[1]
                          << " rotated_av_rel_l2=" << av_error[2]
                          << " recovered_max_abs=" << recovered_error[0]
                          << " recovered_mean_abs=" << recovered_error[1]
                          << " recovered_rel_l2=" << recovered_error[2]
                          << " verdict="
                          << ((av_error[2] <= 1.0e-4 && recovered_error[2] <= 1.0e-4) ? "PASS"
                                                                                         : "FAIL")
                          << '\n';
                if (av_error[2] > 1.0e-4 || recovered_error[2] > 1.0e-4 ||
                    av_error[0] > 1.0e-3 || recovered_error[0] > 1.0e-3) {
                    throw std::logic_error("OSCAR batched live/reference attention mismatch");
                }
            }
        }
        dump_runtime_attention_diagnostic(layer, "attention_gpu_rotated_fp32", fp32_attention, s);
        dump_runtime_attention_diagnostic(layer, "attention_gpu_recovered_fp32", fp32_recovered, s);
        oscar_rotations_->fp32_to_bf16(fp32_recovered, a, s);
        attention_for_output = a;
        if (d4_profile) {
            const double full_attention_us =
                oscar_d4_elapsed_us(live_full_attention_start, std::chrono::steady_clock::now());
            live_mixed_cache_->record_full_attention_us(full_attention_us);
            if (live_gpu_resident) {
                live_mixed_cache_->record_gpu_phase_full_attention_us(
                    ph == Phase::Prefill, full_attention_us);
            }
        }
        if (layer == 63 && d4_profile) {
            const auto counts = live_mixed_cache_->accounting();
            const auto& profile = live_mixed_cache_->profile_totals();
            std::cerr << "OSCAR GPU telemetry: oscar_calibrated=true"
                      << " asset_identity=" << oscar_rotations_->asset_identity()
                      << " asset_hash=" << oscar_rotations_->asset_hash()
                      << " group_size=128 k_clip=0.96 v_clip=0.92 prefix_length=64 recent_length=256"
                      << " prefix_token_count=" << counts.prefix_tokens
                      << " historical_token_count=" << counts.historical_tokens
                      << " recent_token_count=" << counts.recent_tokens
                      << " int2_payload_bytes=" << counts.physical_int2_payload_bytes
                      << " int2_metadata_bytes=" << counts.physical_int2_metadata_bytes
                      << " gpu_cache_staging_us=" << profile.gpu_cache_staging_us
                      << " gpu_cache_staging_bytes=" << profile.gpu_cache_staging_bytes
                      << " gpu_qkv_rotation_us=" << profile.qkv_rotation_us
                      << " gpu_mixed_kernel_us=" << profile.gpu_mixed_kernel_us
                      << " gpu_fused_kernel_us=" << profile.gpu_fused_kernel_us
                      << " gpu_recovery_us=" << profile.gpu_recovery_us
                      << " gpu_full_attention_us=" << profile.full_attention_us
                      << " gpu_prefill_full_attention_us="
                      << profile.gpu_prefill_full_attention_us
                      << " gpu_decode_full_attention_us="
                      << profile.gpu_decode_full_attention_us
                      << " gpu_attention_calls=" << profile.gpu_attention_calls
                      << " gpu_attention_batches=" << profile.gpu_attention_batches
                      << " gpu_attention_kernel_launches="
                      << profile.gpu_attention_kernel_launches
                      << " gpu_prefill_queries=" << profile.gpu_prefill_queries
                      << " gpu_prefill_batches=" << profile.gpu_prefill_batches
                      << " gpu_decode_queries=" << profile.gpu_decode_queries
                      << " gpu_decode_batches=" << profile.gpu_decode_batches
                      << " gpu_fused_query_tiles=" << profile.gpu_fused_query_tiles
                      << " gpu_fused_kv_tiles=" << profile.gpu_fused_kv_tiles
                      << " gpu_prefill_query_block="
                      << oscar_internal::live_gpu_prefill_query_block_size()
                      << " gpu_resident_publish_us=" << profile.gpu_resident_publish_us
                      << " gpu_resident_aging_us=" << profile.gpu_resident_aging_us
                      << " gpu_resident_append_calls=" << profile.gpu_resident_append_calls
                      << " gpu_resident_aging_events=" << profile.gpu_resident_aging_events
                      << " gpu_resident_codec_parity_checks="
                      << profile.gpu_resident_codec_parity_checks
                      << " gpu_incremental_host_device_bytes="
                      << profile.gpu_incremental_host_device_bytes
                      << " gpu_resident_cache_bytes=" << profile.gpu_resident_cache_bytes
                      << " gpu_resident_workspace_bytes="
                      << profile.gpu_resident_workspace_bytes
                      << " legacy_q2_dispatched=false bf16_historical_shadow=false fallback=false"
                      << " selected_layout=mixed-bf16-prefix-oscar-int2-g128-bf16-recent"
                      << " selected_attention_implementation="
                      << (live_gpu_resident
                              ? (oscar_internal::live_int2_gpu_fused_mode_enabled()
                                     ? "oscar-mixed-gpu-d4-6-fused-resident"
                                     : "oscar-mixed-gpu-d4-5-three-stage-resident")
                              : "oscar-mixed-gpu-d4-2b")
                      << '\n';
        }
    } else if (live_reference) {
        if ((active_sequence_batch_ != 0 &&
             (active_sequence_batch_ != 1 || active_sequence_width_ != 1)) ||
            (active_sequence_batch_ == 0 && T <= 0)) {
            throw std::logic_error(
                "OSCAR live INT2 reference path currently supports one sequence only");
        }
        if (cache_positions.dtype != DType::I32 || !cache_positions.is_contiguous() ||
            cache_positions.ne[0] != T || cache_positions.ne[1] != 1 || cache_positions.ne[2] != 1 ||
            cache_positions.ne[3] != 1) {
            throw std::logic_error("OSCAR live INT2 reference positions must be [T]");
        }
        std::vector<std::int32_t> positions(static_cast<std::size_t>(T));
        std::vector<std::uint16_t> q_bf16(static_cast<std::size_t>(kCfg.q_size) * T);
        std::vector<float> rotated_k(static_cast<std::size_t>(kCfg.kv_size) * T);
        std::vector<float> rotated_v(static_cast<std::size_t>(kCfg.kv_size) * T);
        Tensor fp32_q = work_.alloc(DType::FP32, {kCfg.q_size, T})
                            .view({kCfg.head_dim, kCfg.n_q, T});
        Tensor fp32_k = work_.alloc(DType::FP32, {kCfg.kv_size, T})
                            .view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor fp32_v = work_.alloc(DType::FP32, {kCfg.kv_size, T})
                            .view({kCfg.head_dim, kCfg.n_kv, T});
        if (!oscar_rotations_) {
            throw std::logic_error("OSCAR live INT2 reference path has no rotation assets");
        }
        const auto qkv_rotation_start = d4_profile ? std::chrono::steady_clock::now()
                                                    : std::chrono::steady_clock::time_point{};
        oscar_rotations_->rotate_qkv_fp32(layer, qn, kn, v, fp32_q, fp32_k, fp32_v, s);
        CUDA_CHECK(cudaMemcpyAsync(positions.data(), cache_positions.data,
                                   positions.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                   s));
        CUDA_CHECK(cudaMemcpyAsync(q_bf16.data(), qn.data, q_bf16.size() * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(rotated_k.data(), fp32_k.data, rotated_k.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(rotated_v.data(), fp32_v.data, rotated_v.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        if (d4_profile) {
            live_mixed_cache_->record_qkv_rotation_us(
                oscar_d4_elapsed_us(qkv_rotation_start, std::chrono::steady_clock::now()));
        }
        for (const float value : rotated_k) {
            if (!std::isfinite(value)) { throw std::runtime_error("OSCAR live rotated K is NaN/Inf"); }
        }
        for (const float value : rotated_v) {
            if (!std::isfinite(value)) { throw std::runtime_error("OSCAR live rotated V is NaN/Inf"); }
        }
        std::vector<std::uint16_t> k_row(static_cast<std::size_t>(kCfg.n_kv) * kCfg.head_dim);
        std::vector<std::uint16_t> v_row(k_row.size());
        for (int token = 0; token < T; ++token) {
            for (int head = 0; head < kCfg.n_kv; ++head) {
                for (int dimension = 0; dimension < kCfg.head_dim; ++dimension) {
                    const std::size_t source = static_cast<std::size_t>(dimension) +
                                               static_cast<std::size_t>(kCfg.head_dim) *
                                                   (static_cast<std::size_t>(head) +
                                                    static_cast<std::size_t>(kCfg.n_kv) * token);
                    const std::size_t row = static_cast<std::size_t>(head) * kCfg.head_dim +
                                            dimension;
                    k_row[row] = runtime_float_to_bf16(rotated_k[source]);
                    v_row[row] = runtime_float_to_bf16(rotated_v[source]);
                }
            }
            live_mixed_cache_->append(layer, static_cast<std::uint32_t>(positions[token]), k_row,
                                      v_row);
        }
        std::vector<std::uint16_t> output_bf16(static_cast<std::size_t>(kCfg.q_size) * T);
        std::vector<float> q_original(static_cast<std::size_t>(kCfg.q_size));
        for (int token = 0; token < T; ++token) {
            for (int head = 0; head < kCfg.n_q; ++head) {
                for (int dimension = 0; dimension < kCfg.head_dim; ++dimension) {
                    const std::size_t tensor_index = static_cast<std::size_t>(dimension) +
                                                     static_cast<std::size_t>(kCfg.head_dim) *
                                                         (static_cast<std::size_t>(head) +
                                                          static_cast<std::size_t>(kCfg.n_q) * token);
                    q_original[static_cast<std::size_t>(head) * kCfg.head_dim + dimension] =
                        runtime_bf16_to_float(q_bf16[tensor_index]);
                }
            }
            const auto trace = live_mixed_cache_->attention(
                layer, static_cast<std::uint32_t>(positions[token]), q_original);
            if (trace.recovered_output.size() != static_cast<std::size_t>(kCfg.q_size)) {
                throw std::logic_error("OSCAR live attention output shape mismatch");
            }
            for (int head = 0; head < kCfg.n_q; ++head) {
                for (int dimension = 0; dimension < kCfg.head_dim; ++dimension) {
                    const std::size_t tensor_index = static_cast<std::size_t>(dimension) +
                                                     static_cast<std::size_t>(kCfg.head_dim) *
                                                         (static_cast<std::size_t>(head) +
                                                          static_cast<std::size_t>(kCfg.n_q) * token);
                    const float recovered = trace.recovered_output[
                        static_cast<std::size_t>(head) * kCfg.head_dim + dimension];
                    if (!std::isfinite(recovered)) {
                        throw std::runtime_error("OSCAR live recovered attention is NaN/Inf");
                    }
                    output_bf16[tensor_index] = runtime_float_to_bf16(recovered);
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(a.data, output_bf16.data(), output_bf16.size() * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        dump_runtime_attention_diagnostic(layer, "attention_live_recovered", a, s);
        attention_for_output = a;
        if (d4_profile) {
            const double full_attention_us =
                oscar_d4_elapsed_us(live_full_attention_start, std::chrono::steady_clock::now());
            live_mixed_cache_->record_full_attention_us(full_attention_us);
            if (live_gpu_resident) {
                live_mixed_cache_->record_gpu_phase_full_attention_us(
                    ph == Phase::Prefill, full_attention_us);
            }
        }
    } else if (matched_fp32) {
        if ((active_sequence_batch_ != 0 &&
             (active_sequence_batch_ != 1 || active_sequence_width_ != 1)) ||
            (active_sequence_batch_ == 0 && T <= 0)) {
            throw std::logic_error(
                "OSCAR matched FP32 diagnostic path currently supports one sequence only");
        }
        Tensor fp32_q = work_.alloc(DType::FP32, {kCfg.q_size, T})
                            .view({kCfg.head_dim, kCfg.n_q, T});
        Tensor fp32_k = work_.alloc(DType::FP32, {kCfg.kv_size, T})
                            .view({kCfg.head_dim, kCfg.n_kv, T});
        Tensor fp32_v = work_.alloc(DType::FP32, {kCfg.kv_size, T})
                            .view({kCfg.head_dim, kCfg.n_kv, T});
        if (oscar_rotations_) {
            oscar_rotations_->rotate_qkv_fp32(layer, qn, kn, v, fp32_q, fp32_k, fp32_v, s);
        } else {
            oscar_internal::bf16_to_fp32(qn, fp32_q, s);
            oscar_internal::bf16_to_fp32(kn, fp32_k, s);
            oscar_internal::bf16_to_fp32(v, fp32_v, s);
        }
        dump_runtime_attention_diagnostic(layer, "q_fp32", fp32_q, s);
        dump_runtime_attention_diagnostic(layer, "k_fp32", fp32_k, s);
        dump_runtime_attention_diagnostic(layer, "v_fp32", fp32_v, s);

        // The diagnostic cache is indexed by absolute token position and survives the stack
        // lifetime of this TextContext, so the same FP32 representation is used by prefill and
        // the subsequent ordinary decode card.
        Tensor matched_positions = cache_positions.view({T});
        dump_runtime_position_diagnostic(layer, matched_positions, s);
        matched_fp32_cache_->append(fidx, fp32_k, fp32_v, matched_positions, s);
        if (layer == 3) {
            dump_runtime_matched_cache_diagnostic(layer, fidx, *matched_fp32_cache_,
                                                  matched_positions, s);
        }
        Tensor fp32_attention = work_.alloc(DType::FP32, {kCfg.q_size, T})
                                    .view({kCfg.head_dim, kCfg.n_q, T});
        matched_fp32_cache_->attention(fidx, fp32_q, matched_positions, kAttnScale,
                                       fp32_attention, s);
        dump_runtime_attention_diagnostic(layer, "attention_fp32", fp32_attention, s);

        if (oscar_rotations_) {
            Tensor fp32_recovered = work_.alloc(DType::FP32, {kCfg.q_size, T})
                                        .view({kCfg.head_dim, kCfg.n_q, T});
            oscar_rotations_->inverse_value_fp32(layer, fp32_attention, fp32_recovered, s);
            dump_runtime_attention_diagnostic(layer, "attention_recovered_fp32", fp32_recovered,
                                              s);
            oscar_internal::fp32_to_bf16(fp32_recovered, a, s);
        } else {
            // Both matched paths cross the same single final BF16 boundary before gate/output.
            oscar_internal::fp32_to_bf16(fp32_attention, a, s);
        }
        dump_runtime_attention_diagnostic(layer, "attention_recovered", a, s);
        attention_for_output = a;
    } else {
        Tensor q_for_attention = qn;
        Tensor k_for_attention = kn;
        Tensor v_for_attention = v;
        Tensor rotated_q;
        Tensor rotated_k;
        Tensor rotated_v;
        Tensor fp32_q;
        Tensor fp32_k;
        Tensor fp32_v;
        Tensor fp32_attention;
        Tensor fp32_recovered;
        const auto rotation_precision = oscar_rotations_
                                            ? oscar_internal::rotation_precision_mode()
                                            : oscar_internal::RotationPrecision::Bf16Materialized;
        const bool fp32_rotation =
            rotation_precision == oscar_internal::RotationPrecision::Fp32Rotation ||
            rotation_precision == oscar_internal::RotationPrecision::Fp32RotationAndInverse;
        const bool fp32_inverse =
            rotation_precision == oscar_internal::RotationPrecision::Fp32Inverse ||
            rotation_precision == oscar_internal::RotationPrecision::Fp32RotationAndInverse;
        const bool use_fp32_reference_attention =
            oscar_rotations_ && fp32_rotation && ph == Phase::Prefill &&
            active_sequence_batch_ == 0 && T <= 64;
        if (oscar_rotations_) {
            rotated_q = work_.alloc(DType::BF16, {kCfg.q_size, T});
            rotated_k = work_.alloc(DType::BF16, {kCfg.kv_size, T});
            rotated_v = work_.alloc(DType::BF16, {kCfg.kv_size, T});
            rotated_q = rotated_q.view({kCfg.head_dim, kCfg.n_q, T});
            rotated_k = rotated_k.view({kCfg.head_dim, kCfg.n_kv, T});
            rotated_v = rotated_v.view({kCfg.head_dim, kCfg.n_kv, T});
            oscar_rotations_->rotate_qkv(layer, qn, kn, v, rotated_q, rotated_k, rotated_v, s);
            dump_runtime_attention_diagnostic(layer, "q_rot", rotated_q, s);
            dump_runtime_attention_diagnostic(layer, "k_rot", rotated_k, s);
            dump_runtime_attention_diagnostic(layer, "v_rot", rotated_v, s);
            if (fp32_rotation) {
                fp32_q = work_.alloc(DType::FP32, {kCfg.q_size, T})
                             .view({kCfg.head_dim, kCfg.n_q, T});
                fp32_k = work_.alloc(DType::FP32, {kCfg.kv_size, T})
                             .view({kCfg.head_dim, kCfg.n_kv, T});
                fp32_v = work_.alloc(DType::FP32, {kCfg.kv_size, T})
                             .view({kCfg.head_dim, kCfg.n_kv, T});
                oscar_rotations_->rotate_qkv_fp32(layer, qn, kn, v, fp32_q, fp32_k, fp32_v, s);
                dump_runtime_attention_diagnostic(layer, "q_fp32", fp32_q, s);
                dump_runtime_attention_diagnostic(layer, "k_fp32", fp32_k, s);
                dump_runtime_attention_diagnostic(layer, "v_fp32", fp32_v, s);
            }
            if (fp32_inverse) {
                fp32_recovered = work_.alloc(DType::FP32, {kCfg.q_size, T})
                                     .view({kCfg.head_dim, kCfg.n_q, T});
            }
            q_for_attention = rotated_q;
            k_for_attention = rotated_k;
            v_for_attention = rotated_v;
        }

        const Tensor& kv_table_rows =
            active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
        if (active_sequence_batch_ != 0) {
            const std::int32_t width = active_sequence_width_;
            if (width <= 0 || width * active_sequence_batch_ != T) {
                throw std::logic_error("Text sequence batch binding does not match aggregate columns");
            }
            Tensor q_batch =
                q_for_attention.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
            Tensor k_batch =
                k_for_attention.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
            Tensor v_batch =
                v_for_attention.view({kCfg.head_dim, kCfg.n_kv, width, active_sequence_batch_});
            Tensor a_batch = a.view({kCfg.head_dim, kCfg.n_q, width, active_sequence_batch_});
            Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
            const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
            ops::causal_softmax_attention(q_batch, k_batch, v_batch, position_batch, valid,
                                          kv_table_rows, {kCfg.head_dim, kCfg.n_q, kCfg.n_kv},
                                          kAttnScale, batch_text_kv_->batch_layer_view(fidx),
                                          *active_causal_attention_envelope_, work_, a_batch, s);
        } else {
            ops::causal_softmax_attention(q_for_attention, k_for_attention, v_for_attention,
                                          cache_positions, Tensor{}, kv_table_rows,
                                          {kCfg.head_dim, kCfg.n_q, kCfg.n_kv}, kAttnScale,
                                          batch_text_kv_->batch_layer_view(fidx),
                                          *active_causal_attention_envelope_, work_, a, s);
        }
        dump_runtime_attention_diagnostic(layer, "attention_raw", a, s);
        if (use_fp32_reference_attention) {
            fp32_attention = work_.alloc(DType::FP32, {kCfg.q_size, T})
                                 .view({kCfg.head_dim, kCfg.n_q, T});
            oscar_rotations_->reference_attention_fp32(fp32_q, fp32_k, fp32_v, kAttnScale,
                                                       fp32_attention, s);
            dump_runtime_attention_diagnostic(layer, "attention_fp32", fp32_attention, s);
        }
        bool recovered_in_attention_output = false;
        if (oscar_rotations_) {
            rotated_q = rotated_q.view({kCfg.head_dim, kCfg.n_q, T});
            if (use_fp32_reference_attention) {
                if (fp32_inverse) {
                    oscar_rotations_->inverse_value_fp32(layer, fp32_attention, fp32_recovered, s);
                    dump_runtime_attention_diagnostic(layer, "attention_recovered_fp32",
                                                      fp32_recovered, s);
                    oscar_rotations_->fp32_to_bf16(fp32_recovered, rotated_q, s);
                } else {
                    oscar_rotations_->fp32_to_bf16(fp32_attention, rotated_q, s);
                    oscar_rotations_->inverse_value(layer, rotated_q, a, s);
                    attention_for_output = a;
                    recovered_in_attention_output = true;
                }
            } else if (fp32_inverse) {
                oscar_rotations_->inverse_value_fp32(layer, a, fp32_recovered, s);
                dump_runtime_attention_diagnostic(layer, "attention_recovered_fp32", fp32_recovered,
                                                  s);
                oscar_rotations_->fp32_to_bf16(fp32_recovered, rotated_q, s);
            } else {
                oscar_rotations_->inverse_value(layer, a, rotated_q, s);
            }
            dump_runtime_attention_diagnostic(layer, "attention_recovered",
                                              recovered_in_attention_output ? a : rotated_q, s);
            if (!recovered_in_attention_output) { attention_for_output = rotated_q; }
        }
    }
    ops::sigmoid_mul(gate, attention_for_output, s);

    Variant::attention_output_projection(attention_for_output.view({kCfg.q_size, T}), *w.o_proj, x,
                                         ph, work_, s);
    // linear_add writes the attention output projection directly into the residual stream.
    // This tap therefore records the exact post-attention residual boundary, not a separate
    // unfused projection tensor.
    dump_runtime_layer_stage_diagnostic(layer, "post_attention_linear_add", x, s);
}

void dump_runtime_diagnostic(const Tensor& hidden, const Tensor& logits, cudaStream_t stream) {
    const char* prefix = std::getenv("NINFER_OSCAR_RUNTIME_DIAGNOSTIC_PREFIX");
    if (prefix == nullptr || *prefix == '\0') { return; }
    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, 1},
                         "OSCAR runtime diagnostic hidden");
    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, 1},
                         "OSCAR runtime diagnostic logits");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto write = [](const std::string& path, const Tensor& tensor) {
        std::vector<std::uint8_t> bytes(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(bytes.data(), tensor.data, bytes.size(), cudaMemcpyDeviceToHost));
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot open OSCAR runtime diagnostic " + path); }
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        if (!output) { throw std::runtime_error("cannot write OSCAR runtime diagnostic " + path); }
    };
    write(std::string(prefix) + ".hidden.bf16", hidden);
    write(std::string(prefix) + ".logits.bf16", logits);
}

void dump_runtime_layer_diagnostic(std::int32_t layer, const Tensor& value,
                                   cudaStream_t stream) {
    if (oscar_d4_profile_enabled()) { return; }
    const char* prefix = std::getenv("NINFER_OSCAR_RUNTIME_LAYER_DIAGNOSTIC_PREFIX");
    if (prefix == nullptr || *prefix == '\0') { return; }
    require_tensor_window(value, DType::BF16, kCfg.hidden, value.ne[1],
                          "OSCAR runtime diagnostic layer value");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint8_t> bytes(value.bytes());
    CUDA_CHECK(cudaMemcpy(bytes.data(), value.data, bytes.size(), cudaMemcpyDeviceToHost));
    const std::string path = std::string(prefix) + ".tokens_" + std::to_string(value.ne[1]) +
                             oscar_internal::matched_diagnostic_step_suffix() + ".layer_" +
                             std::to_string(layer) + ".bf16";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot open OSCAR layer diagnostic " + path); }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) { throw std::runtime_error("cannot write OSCAR layer diagnostic " + path); }
}

void dump_runtime_layer_stage_diagnostic(std::int32_t layer, const char* label,
                                         const Tensor& value, cudaStream_t stream) {
    if (oscar_d4_profile_enabled()) { return; }
    const char* prefix = std::getenv("NINFER_OSCAR_RUNTIME_LAYER_DIAGNOSTIC_PREFIX");
    if (prefix == nullptr || *prefix == '\0') { return; }
    require_tensor_window(value, DType::BF16, kCfg.hidden, value.ne[1],
                          "OSCAR runtime diagnostic layer stage value");
    if (label == nullptr || *label == '\0') {
        throw std::invalid_argument("OSCAR runtime diagnostic layer stage label is empty");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint8_t> bytes(value.bytes());
    CUDA_CHECK(cudaMemcpy(bytes.data(), value.data, bytes.size(), cudaMemcpyDeviceToHost));
    const std::string path = std::string(prefix) + ".tokens_" + std::to_string(value.ne[1]) +
                             oscar_internal::matched_diagnostic_step_suffix() + ".layer_" +
                             std::to_string(layer) + "." + label + ".bf16";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot open OSCAR layer stage diagnostic " + path); }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("cannot write OSCAR layer stage diagnostic " + path);
    }
}

void dump_runtime_attention_diagnostic(std::int32_t layer, const char* label,
                                       const Tensor& value, cudaStream_t stream) {
    if (oscar_d4_profile_enabled()) { return; }
    const char* prefix = std::getenv("NINFER_OSCAR_RUNTIME_ATTENTION_DIAGNOSTIC_PREFIX");
    if (prefix == nullptr || *prefix == '\0') { return; }
    if ((value.dtype != DType::BF16 && value.dtype != DType::FP32) || !value.is_contiguous() ||
        value.data == nullptr || value.ne[0] <= 0 || value.ne[1] <= 0 || value.ne[2] <= 0 ||
        value.ne[3] != 1) {
        throw std::invalid_argument("OSCAR runtime diagnostic attention value has invalid shape/dtype");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint8_t> bytes(value.bytes());
    CUDA_CHECK(cudaMemcpy(bytes.data(), value.data, bytes.size(), cudaMemcpyDeviceToHost));
    const std::string path = std::string(prefix) + ".tokens_" + std::to_string(value.ne[2]) +
                             oscar_internal::matched_diagnostic_step_suffix() + ".layer_" +
                             std::to_string(layer) + "." + label +
                             (value.dtype == DType::BF16 ? ".bf16" : ".fp32");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot open OSCAR attention diagnostic " + path); }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) { throw std::runtime_error("cannot write OSCAR attention diagnostic " + path); }
}

void dump_runtime_position_diagnostic(std::int32_t layer, const Tensor& value,
                                      cudaStream_t stream) {
    if (oscar_d4_profile_enabled()) { return; }
    const char* prefix = std::getenv("NINFER_OSCAR_RUNTIME_ATTENTION_DIAGNOSTIC_PREFIX");
    if (prefix == nullptr || *prefix == '\0') { return; }
    if (value.dtype != DType::I32 || !value.is_contiguous() || value.data == nullptr ||
        value.ne[0] <= 0 || value.ne[1] != 1 || value.ne[2] != 1 || value.ne[3] != 1) {
        throw std::invalid_argument("OSCAR runtime diagnostic positions have invalid shape/dtype");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::int32_t> positions(static_cast<std::size_t>(value.ne[0]));
    CUDA_CHECK(cudaMemcpy(positions.data(), value.data, positions.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    const std::string path = std::string(prefix) + ".tokens_" + std::to_string(value.ne[0]) +
                             oscar_internal::matched_diagnostic_step_suffix() + ".layer_" +
                             std::to_string(layer) + ".positions.i32";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) { throw std::runtime_error("cannot open OSCAR position diagnostic " + path); }
    output.write(reinterpret_cast<const char*>(positions.data()),
                 static_cast<std::streamsize>(positions.size() * sizeof(std::int32_t)));
    if (!output) { throw std::runtime_error("cannot write OSCAR position diagnostic " + path); }
}

void dump_runtime_matched_cache_diagnostic(
    std::int32_t layer, std::int32_t full_layer,
    const oscar_internal::OscarMatchedFP32Cache& cache,
    const Tensor& positions, cudaStream_t stream) {
    if (oscar_d4_profile_enabled()) { return; }
    const char* prefix = std::getenv("NINFER_OSCAR_RUNTIME_ATTENTION_DIAGNOSTIC_PREFIX");
    if (prefix == nullptr || *prefix == '\0') { return; }
    if (positions.dtype != DType::I32 || !positions.is_contiguous() || positions.data == nullptr ||
        positions.ne[0] <= 0 || positions.ne[1] != 1 || positions.ne[2] != 1 ||
        positions.ne[3] != 1) {
        throw std::invalid_argument("OSCAR runtime diagnostic cache positions are invalid");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::int32_t last_position = -1;
    CUDA_CHECK(cudaMemcpy(&last_position,
                          static_cast<const std::uint8_t*>(positions.data) +
                              static_cast<std::size_t>(positions.ne[0] - 1) * sizeof(std::int32_t),
                          sizeof(last_position), cudaMemcpyDeviceToHost));
    if (last_position < 0 || static_cast<std::uint64_t>(last_position) + 1U > cache.max_context()) {
        throw std::invalid_argument("OSCAR runtime diagnostic cache position is out of range");
    }
    const std::uint32_t token_count = static_cast<std::uint32_t>(last_position) + 1U;
    std::vector<float> k(static_cast<std::size_t>(token_count) * 4U * kCfg.head_dim);
    std::vector<float> v(k.size());
    cache.copy_layer_prefix(full_layer, token_count, k.data(), v.data(), stream);
    const std::string suffix = oscar_internal::matched_diagnostic_step_suffix();
    const auto write = [&](const char* label, const std::vector<float>& values) {
        const std::string path = std::string(prefix) + ".cache_tokens_" +
                                 std::to_string(token_count) + suffix + ".layer_" +
                                 std::to_string(layer) + "." + label + ".fp32";
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot open OSCAR cache diagnostic " + path); }
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!output) { throw std::runtime_error("cannot write OSCAR cache diagnostic " + path); }
    };
    write("cache_k_fp32", k);
    write("cache_v_fp32", v);
}

void TextContext::gdn_mix(const GdnLayerW& w, Tensor& x, int gidx, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace_recipe::gdn_control<TextConfig>(work_, T);
    Tensor h           = control.hidden;
    Tensor g           = control.g;
    Tensor beta        = control.beta;
    Variant::gdn_norm_control_projection(x, *w.input_norm, kCfg.rms_eps, *w.projection, h, g, beta,
                                         work_, s);

    const auto projection = workspace_recipe::gdn_projection<TextConfig>(work_, T);
    Tensor z              = projection.output_gate.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor qc             = projection.query;
    Tensor kc             = projection.key;
    Tensor vc             = projection.value;
    if (ph == Phase::Verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        Tensor projection_input = h.view({kCfg.hidden, width, active_sequence_batch_});
        Tensor query_output     = qc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor key_output       = kc.view({kCfg.key_dim, width, active_sequence_batch_});
        Tensor value_output     = vc.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor gate_output      = z.view({kCfg.value_dim, width, active_sequence_batch_});
        Tensor conv_states      = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            Variant::gdn_input_projection_record(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_source_slots_, records.conv, query_output, key_output,
                value_output, gate_output, ph, work_, s);
        } else {
            Variant::gdn_input_projection_snapshot(
                projection_input, *w.projection, *w.conv1d, conv_states, valid,
                *active_linear_state_source_slots_, *active_linear_state_destination_slots_,
                query_output, key_output, value_output, gate_output, ph, work_, s);
        }
    } else {
        Tensor qkv = workspace_recipe::gdn_prefill_conv<TextConfig>(work_, T);
        Variant::gdn_input_projection(h, *w.projection, qkv, z, ph, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, *w.conv1d, conv_state_in, conv_state_out, qc, kc, vc, s);
    }

    Tensor q_recurrent = qc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});
    Tensor k_recurrent = kc.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, T});

    Tensor vv = vc.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    Tensor o  = workspace_recipe::gdn_recurrent_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    if (ph == Phase::Verify) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch =
            q_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor k_batch =
            k_recurrent.view({kCfg.gdn_k_dim, kCfg.gdn_k_heads, width, active_sequence_batch_});
        Tensor v_batch = vv.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor g_batch = g.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor beta_batch = beta.view({kCfg.gdn_v_heads, width, active_sequence_batch_});
        Tensor out_batch =
            o.view({kCfg.gdn_v_dim, kCfg.gdn_v_heads, width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(q_batch, k_batch, v_batch, g_batch, beta_batch,
                                               kGdnScale, recurrent_states, valid,
                                               *active_linear_state_source_slots_, records.key,
                                               records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch, kGdnScale,
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale,
                             /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out,
                             o, s);
    }

    Tensor on = workspace_recipe::gdn_normalized_output<TextConfig>(work_, T).view(
        {kCfg.gdn_v_dim, kCfg.gdn_v_heads, T});
    ops::gated_rmsnorm(o, *w.gdn_norm, z, kCfg.rms_eps, on, s);

    Variant::gdn_output_projection(on.view({kCfg.value_dim, T}), *w.out_proj, x, ph, work_, s);
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);

    Variant::post_mixer(h, *m.payload, x, ph, work_, s);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (ModelConfig::is_full(layer)) {
            const int fidx         = ModelConfig::full_idx(layer);
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(full, x, layer, fidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                dump_runtime_layer_diagnostic(layer, x, ctx_.stream);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        } else {
            const int gidx       = ModelConfig::gdn_idx(layer);
            const GdnLayerW& gdn = gdn_.at(static_cast<std::size_t>(gidx));
            nvtx::ScopedRange layer_range(prefill ? nvtx::Name::PrefillLayerGdn
                                                  : nvtx::Name::VerifyLayerGdn,
                                          nvtx::Category::Gdn, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn, nvtx::Category::Gdn,
                    static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                gdn_mix(gdn, x, gidx, ph);
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(gdn.post_attn_norm, gdn.mlp, x, ph);
                dump_runtime_layer_diagnostic(layer, x, ctx_.stream);
                if constexpr (Tap::enabled) { tap.capture_layer(layer, x, ctx_.stream); }
            }
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    if (matched_fp32_cache_ && kv_.valid() && base == 0) { matched_fp32_cache_->reset(s); }
    if (live_mixed_cache_ && kv_.valid() && base == 0) { live_mixed_cache_->reset(); }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s);
                dump_runtime_diagnostic(last_xf, logits, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_, io_.pos,
                                ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token, kCfg.token_domain, s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_6::MtpAlignmentWindow mtp_window = qwen3_6::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings = work_.alloc(DType::BF16, {kCfg.hidden, len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_6::MtpVisualOverlap overlap = qwen3_6::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace_recipe::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_6::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token     = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token     = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden    = work_.alloc(DType::BF16, {kCfg.hidden, 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {kCfg.hidden, 1}, "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_6::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
