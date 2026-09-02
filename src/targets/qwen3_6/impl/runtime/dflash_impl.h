#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/dflash2_dynamic_conv.h"
#include "ninfer/ops/dflash2_qkv_proj.h"
#include "ninfer/ops/dflash2_predecessor_ids.h"
#include "ninfer/ops/dflash2_selector_lattice.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = dflash_state(state).pending_features.slice(1, 0, width),
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash && !V::DFlashConfig::is_v2) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        ops::linear(features.view({Config::feature_rows, columns}),
                    state.execution.model.dflash->feature_projection, projected,
                    state.execution.device.stream);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected, state.execution.model.dflash->context_norm, Config::rms_epsilon,
                     false, context, state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            const bool local_layer  = layer < Config::local_layers;
            const int layer_width   = local_layer ? local_width : width;
            const int layer_columns = layer_width * batch;
            Tensor layer_context    = local_layer && replace_local_window
                                          ? context.slice(1, local_offset, local_width)
                                          : context;
            Tensor layer_positions  = local_layer && replace_local_window
                                          ? positions.slice(0, local_offset, local_width)
                                          : positions;
            Tensor key;
            Tensor value;
            if constexpr (Config::is_v2) {
                // DFlash2 injects committed target features through the drafter's
                // QKV projection. Only K/V are retained in the local cache; Q is
                // materialized in the shared attention roots for the attention pass.
                auto attn_roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, layer_columns);
                Tensor query_raw =
                    attn_roots.query_raw.view({Config::head_dim, Config::query_heads, layer_columns});
                Tensor key_raw =
                    attn_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
                value = attn_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
                Tensor query_flat = query_raw.view({Config::query_size, layer_columns});
                Tensor key_flat = key_raw.view({Config::kv_size, layer_columns});
                Tensor value_flat = value.view({Config::kv_size, layer_columns});
                ops::dflash2_qkv_proj(layer_context, weight.query_key_value, query_flat, key_flat,
                                      value_flat, state.execution.device.stream);
                key = attn_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(layer_positions.view({layer_columns}), Config::head_dim, Config::rope_theta,
                          key, state.execution.device.stream);
            } else {
                auto layer_roots =
                    workspace_recipe::dflash_context_layer<Config>(state.execution.work, layer_columns);
                Tensor key_raw =
                    layer_roots.key_raw.view({Config::head_dim, Config::kv_heads, layer_columns});
                value =
                    layer_roots.value.view({Config::head_dim, Config::kv_heads, layer_columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, layer_columns});
                Tensor value_flat = value.view({Config::kv_size, layer_columns});
                ops::linear_pair(layer_context, weight.context_key, weight.context_value, key_flat,
                                 value_flat, state.execution.device.stream);
                key = layer_roots.key.view({Config::head_dim, Config::kv_heads, layer_columns});
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(layer_positions.view({layer_columns}), Config::head_dim, Config::rope_theta,
                          key, state.execution.device.stream);
            }
            Tensor key_batch = key.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor value_batch =
                value.view({Config::head_dim, Config::kv_heads, layer_width, batch});
            Tensor position_batch = layer_positions.view({layer_width, batch});
            if (local_layer) {
                // DFlash2 uses the BF16 local drafter cache. Keep the legacy dual-write branch
                // available for DFlash1 configurations that still request a quantized local L0.
                if (dflash_state(state).local_q4_shadow != nullptr) {
                    ops::kv_cache_append_prefix_oscar_dual(
                        key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                        dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                        dflash_state(state).local_q4_layer(static_cast<std::uint32_t>(layer)),
                        state.execution.device.stream);
                } else {
                    ops::kv_cache_append_prefix(
                        key_batch, value_batch, position_batch, local_counts, lanes, local_envelope,
                        dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                        state.execution.device.stream);
                }
            } else {
                ops::kv_cache_append_prefix(
                    key_batch, value_batch, position_batch, commit_counts, table_rows, envelope,
                    dflash_state(state).full_batch_layer(0), state.execution.device.stream);
            }
        }
    }
}

template <class V>
requires(V::DFlashConfig::is_v2)
void propose_batch_v2_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                           std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes,
                           bool use_q4_verifier_cache);

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes,
                        bool use_q4_verifier_cache = false) {
    if constexpr (V::DFlashConfig::is_v2) {
        propose_batch_v2_impl<V>(state, frame, batch_size, k, envelopes,
                                 use_q4_verifier_cache);
    } else if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        Tensor anchors             = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers           = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor state_destinations  = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor full_rows           = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor ids                 = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions           = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts              = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);
        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));
            {
                auto attention_scope = state.execution.work.scope();
                auto roots =
                    workspace_recipe::dflash_attention<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);
                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw = roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value   = roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat   = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::attn_input_proj(roots.hidden, weight.query_key_value, query_flat, key_flat,
                                     value_flat, state.execution.device.stream);
                Tensor query = roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key   = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.stream);
                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                if (layer < Config::local_layers) {
                    ops::sliding_window_attention(
                        query_batch, key_batch, value_batch, positions, valid_columns,
                        state_destinations,
                        {Config::head_dim, Config::query_heads, Config::kv_heads},
                        Config::local_capacity, Config::attention_scale,
                        use_q4_verifier_cache
                            ? dflash_state(state).local_q4_layer(static_cast<std::uint32_t>(layer))
                            : dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                        envelopes.local, state.execution.work, attention_batch,
                        state.execution.device.stream);
                } else {
                    ops::context_softmax_attention(
                        query_batch, key_batch, value_batch, frontiers, valid_columns, full_rows,
                        {Config::head_dim, Config::query_heads, Config::kv_heads},
                        Config::attention_scale, dflash_state(state).full_batch_layer(0),
                        envelopes.full, state.execution.work, attention_batch,
                        state.execution.device.stream);
                }
                ops::linear_add(roots.attention.view({Config::query_size, columns}),
                                weight.attention_output, residual, state.execution.work,
                                state.execution.device.stream);
            }
            {
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                ops::linear_swiglu(roots.hidden, weight.gate_up, roots.intermediate,
                                   state.execution.work, state.execution.device.stream);
                ops::linear_add(roots.intermediate, weight.down, residual, state.execution.work,
                                state.execution.device.stream);
            }
        }

        Tensor packed = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        const std::size_t element_bytes = dtype_size(DType::BF16);
        const std::size_t row_bytes =
            static_cast<std::size_t>(Config::hidden) * static_cast<std::size_t>(k) * element_bytes;
        const std::size_t source_pitch =
            static_cast<std::size_t>(Config::hidden) * width * element_bytes;
        const auto* source = static_cast<const std::byte*>(residual.data) +
                             static_cast<std::size_t>(Config::hidden) * element_bytes;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                     static_cast<std::size_t>(batch_size), cudaMemcpyDeviceToDevice,
                                     state.execution.device.stream));
        Tensor proposal_hidden = state.execution.work.alloc(
            DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
        ops::rmsnorm(packed, state.execution.model.dflash->final_norm, Config::rms_epsilon, false,
                     proposal_hidden, state.execution.device.stream);
        Tensor flat_drafts = drafts.view({static_cast<std::int32_t>(k) * batch_size});
        if (state.execution.proposal_head == ProposalHead::Full) {
            Tensor logits = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
            ops::argmax(logits, flat_drafts, TextConfig::token_domain,
                        state.execution.device.stream);
        } else {
            if (!state.execution.model.optimized_proposal.has_value()) {
                throw std::logic_error("optimized DFlash proposal head is unavailable");
            }
            const auto& proposal = *state.execution.model.optimized_proposal;
            Tensor logits        = state.execution.work.alloc(
                DType::BF16, {V::draft_head_rows, static_cast<std::int32_t>(k) * batch_size});
            ops::linear(proposal_hidden, proposal.head, logits, state.execution.device.stream);
            ops::argmax(logits, flat_drafts, V::draft_head_rows, state.execution.device.stream);
            ops::proposal_remap_token_ids(flat_drafts,
                                          static_cast<const std::int32_t*>(proposal.token_ids.data),
                                          V::draft_head_rows, state.execution.device.stream);
        }
        state.execution.work.reset();
    }
}

template <class V>
requires(V::DFlashConfig::is_v2)
void propose_batch_v2_impl(DFlashBatchContext& state, qwen3_6::DFlashDecodeState& frame,
                           std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes,
                           bool use_q4_verifier_cache) {
    using Config = typename V::DFlashConfig;
        static_assert(Config::block_size >= 2);
        static_assert(Config::selector_rank == ops::kDFlash2SelectorRank);
        static_assert(Config::selector_top_k == ops::kDFlash2SelectorTopK);

        const std::int32_t width   = static_cast<std::int32_t>(k) + 1;
        const std::int32_t columns = width * batch_size;
        if (k != static_cast<std::uint32_t>(Config::block_size - 1)) {
            throw std::logic_error("DFlash2 requires its complete configured draft block");
        }

        Tensor anchors             = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers           = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor valid_columns       = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor state_destinations  = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor ids                 = frame.proposal_ids.slice(1, 0, batch_size);
        Tensor positions           = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor drafts              = frame.draft_tokens.slice(1, 0, batch_size);

        state.execution.work.reset();
        ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                                  positions, state.execution.device.stream);

        Tensor residual = state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        // Keep the embedding directly in the residual buffer. Each DFlash2
        // sublayer updates that buffer in place, so the proposal path does
        // not need an intermediate embedding copy or a second residual
        // buffer for the attention branch.
        ops::embedding(ids.view({columns}), state.execution.model.token_embedding, residual,
                       state.execution.device.stream);

        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope = state.execution.work.scope();
            const auto& weight =
                state.execution.model.dflash->layers.at(static_cast<std::size_t>(layer));

            {
                auto attention_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_attention<Config>(state.execution.work,
                                                                         columns);
                ops::rmsnorm(residual, weight.input_norm, Config::rms_epsilon, false, roots.hidden,
                             state.execution.device.stream);

                Tensor attention_dynamic = state.execution.work.alloc(
                    DType::BF16, {Config::conv_projection_rows, columns});
                ops::linear(roots.hidden, weight.attention_conv_projection, attention_dynamic,
                            state.execution.device.stream);
                Tensor noise_conv =
                    state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
                ops::dflash2_dynamic_conv(roots.hidden, attention_dynamic,
                                          weight.attention_conv_base, 0, Config::block_size,
                                          noise_conv, state.execution.device.stream);

                Tensor query_raw =
                    roots.query_raw.view({Config::head_dim, Config::query_heads, columns});
                Tensor key_raw =
                    roots.key_raw.view({Config::head_dim, Config::kv_heads, columns});
                Tensor value =
                    roots.value.view({Config::head_dim, Config::kv_heads, columns});
                Tensor query_flat = query_raw.view({Config::query_size, columns});
                Tensor key_flat = key_raw.view({Config::kv_size, columns});
                Tensor value_flat = value.view({Config::kv_size, columns});
                ops::dflash2_qkv_proj(
                    noise_conv, weight.query_key_value, query_flat, key_flat, value_flat,
                    state.execution.device.stream);

                Tensor query =
                    roots.query.view({Config::head_dim, Config::query_heads, columns});
                Tensor key = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                ops::rmsnorm(query_raw, weight.query_norm, Config::rms_epsilon, false, query,
                             state.execution.device.stream);
                ops::rmsnorm(key_raw, weight.key_norm, Config::rms_epsilon, false, key,
                             state.execution.device.stream);
                ops::rope(positions.view({columns}), Config::head_dim, Config::rope_theta, query,
                          key, state.execution.device.stream);

                Tensor query_batch =
                    query.view({Config::head_dim, Config::query_heads, width, batch_size});
                Tensor key_batch =
                    key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor value_batch =
                    value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                Tensor attention_batch = roots.attention.view(
                    {Config::head_dim, Config::query_heads, width, batch_size});
                ops::sliding_window_attention(
                    query_batch, key_batch, value_batch, positions, valid_columns,
                    state_destinations,
                    {Config::head_dim, Config::query_heads, Config::kv_heads},
                    Config::local_capacity, Config::attention_scale,
                    use_q4_verifier_cache
                        ? dflash_state(state).local_q4_layer(static_cast<std::uint32_t>(layer))
                        : dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                    envelopes.local, state.execution.work, attention_batch,
                    state.execution.device.stream);

                Tensor attention_out =
                    state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
                ops::linear(roots.attention.view({Config::query_size, columns}),
                            weight.attention_output, attention_out,
                            state.execution.device.stream);
                Tensor attention_out_conv =
                    state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
                ops::dflash2_dynamic_conv(attention_out, attention_dynamic,
                                          weight.attention_conv_base, 1, Config::block_size,
                                          attention_out_conv, state.execution.device.stream);
                ops::residual_add(attention_out_conv, residual, state.execution.device.stream);
            }

            {
                auto mlp_scope = state.execution.work.scope();
                auto roots = workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                ops::rmsnorm(residual, weight.post_attention_norm, Config::rms_epsilon, false,
                             roots.hidden, state.execution.device.stream);
                Tensor mlp_dynamic = state.execution.work.alloc(
                    DType::BF16, {Config::conv_projection_rows, columns});
                ops::linear(roots.hidden, weight.mlp_conv_projection, mlp_dynamic,
                            state.execution.device.stream);
                Tensor ffn_conv =
                    state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
                ops::dflash2_dynamic_conv(roots.hidden, mlp_dynamic, weight.mlp_conv_base, 0,
                                          Config::block_size, ffn_conv,
                                          state.execution.device.stream);
                // DFlash2's W8 gate/up projection now uses the Qwen3.8 fused SwiGLU route. This
                // keeps the gate/up accumulators in registers and avoids materializing the
                // 34,816 x columns BF16 intermediate before the nonlinearity.
                ops::linear_swiglu(ffn_conv, weight.gate_up, roots.intermediate,
                                   state.execution.work, state.execution.device.stream);
                Tensor mlp_out =
                    state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
                ops::linear(roots.intermediate, weight.down, mlp_out,
                            state.execution.device.stream);
                Tensor mlp_out_conv =
                    state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
                ops::dflash2_dynamic_conv(mlp_out, mlp_dynamic, weight.mlp_conv_base, 1,
                                          Config::block_size, mlp_out_conv,
                                          state.execution.device.stream);
                ops::residual_add(mlp_out_conv, residual, state.execution.device.stream);
            }
        }

        Tensor proposal_hidden =
            state.execution.work.alloc(DType::BF16, {Config::hidden, columns});
        ops::rmsnorm(residual, state.execution.model.dflash->final_norm, Config::rms_epsilon,
                     false, proposal_hidden, state.execution.device.stream);
        // DFlash2's selector only consumes the top-16 proposal candidates. When the caller
        // selected the optimized proposal profile, reuse the artifact's quantized 131K-row
        // draft head instead of streaming the full 248K-row target head. The candidate IDs are
        // remapped to the target vocabulary before selector codebook lookup. The default path
        // rescored those 16 rows from the FP8 target head; an opt-in wider shortlist is retained
        // for workload research through NINFER_DFLASH2_WIDE_SHORTLIST=1. Target verification
        // remains authoritative for every accepted token.
        const OptimizedProposalWeights* private_proposal =
            state.execution.proposal_head == ProposalHead::Optimized &&
                    state.execution.model.optimized_proposal.has_value()
                ? &*state.execution.model.optimized_proposal
                : nullptr;
        const std::int32_t proposal_vocab =
            private_proposal == nullptr ? TextConfig::output_rows : private_proposal->head.n;
        state.dflash.phase_timing.begin_selector();
        Tensor logits =
            state.execution.work.alloc(DType::BF16, {proposal_vocab, columns});
        if (private_proposal == nullptr) {
            ops::linear(proposal_hidden, state.execution.model.output_head, logits,
                        state.execution.device.stream);
        } else {
            ops::linear(proposal_hidden, private_proposal->head, logits,
                        state.execution.device.stream);
        }

        Tensor candidates =
            state.execution.work.alloc(DType::I32, {Config::selector_top_k, columns});
        Tensor unary =
            state.execution.work.alloc(DType::FP32, {Config::selector_top_k, columns});
        const char* wide_shortlist_env = std::getenv("NINFER_DFLASH2_WIDE_SHORTLIST");
        const bool use_wide_shortlist = wide_shortlist_env != nullptr &&
                                        wide_shortlist_env[0] == '1';
        if (private_proposal == nullptr || !use_wide_shortlist) {
            ops::dflash2_select_candidates(logits, candidates, unary,
                                           state.execution.device.stream);
            if (private_proposal != nullptr) {
                Tensor candidates_flat = candidates.view({Config::selector_top_k * columns});
                ops::proposal_remap_token_ids(
                    candidates_flat,
                    static_cast<const std::int32_t*>(private_proposal->token_ids.data),
                    private_proposal->head.n, state.execution.device.stream);
                if (state.execution.model.output_head.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
                    ops::dflash2_score_fp8_candidates(proposal_hidden,
                                                      state.execution.model.output_head, candidates,
                                                      unary, state.execution.device.stream);
                }
            }
        } else {
            Tensor shortlist = state.execution.work.alloc(
                DType::I32, {ops::kDFlash2ProposalShortlistTopK, columns});
            Tensor shortlist_scores = state.execution.work.alloc(
                DType::FP32, {ops::kDFlash2ProposalShortlistTopK, columns});
            Tensor shortlist_scratch =
                state.execution.work.alloc(DType::I32, {ops::kDFlash2SelectorTopK, columns * 2});
            Tensor shortlist_scratch_scores = state.execution.work.alloc(
                DType::FP32, {ops::kDFlash2SelectorTopK, columns * 2});
            ops::dflash2_select_candidates_wide(logits, shortlist, shortlist_scores,
                                                shortlist_scratch, shortlist_scratch_scores,
                                                state.execution.device.stream);
            Tensor shortlist_flat = shortlist.view({ops::kDFlash2ProposalShortlistTopK * columns});
            ops::proposal_remap_token_ids(
                shortlist_flat, static_cast<const std::int32_t*>(private_proposal->token_ids.data),
                private_proposal->head.n, state.execution.device.stream);
            if (state.execution.model.output_head.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
                ops::dflash2_score_fp8_candidates(proposal_hidden,
                                                  state.execution.model.output_head, shortlist,
                                                  shortlist_scores, state.execution.device.stream);
            }
            ops::dflash2_reduce_scored_candidates(shortlist, shortlist_scores, candidates, unary,
                                                  state.execution.device.stream);
        }

        Tensor selector_hidden =
            state.execution.work.alloc(DType::BF16, {Config::selector_rank, columns});
        ops::linear(proposal_hidden, state.execution.model.dflash->selector_hidden_projection,
                    selector_hidden, state.execution.device.stream);

        const std::int32_t top_k_columns = Config::selector_top_k * columns;
        Tensor candidates_flat = candidates.view({top_k_columns});
        Tensor successor =
            state.execution.work.alloc(DType::BF16, {Config::selector_rank, top_k_columns});
        ops::embedding(candidates_flat,
                       state.execution.model.dflash->selector_successor_codebook, successor,
                       state.execution.device.stream);

        Tensor predecessor_ids =
            state.execution.work.alloc(DType::I32, {Config::selector_top_k, columns});
        ops::dflash2_predecessor_ids(candidates, anchors, Config::block_size, predecessor_ids,
                                     state.execution.device.stream);
        Tensor predecessor =
            state.execution.work.alloc(DType::BF16, {Config::selector_rank, top_k_columns});
        ops::embedding(predecessor_ids.view({top_k_columns}),
                       state.execution.model.dflash->selector_predecessor_codebook, predecessor,
                       state.execution.device.stream);

        Tensor lattice = state.execution.work.alloc(
            DType::FP32, {ops::kDFlash2SelectorPackedWidth, columns});
        ops::dflash2_selector_lattice(
            selector_hidden, successor.view({Config::selector_rank, Config::selector_top_k,
                                              columns}),
            predecessor.view({Config::selector_rank, Config::selector_top_k, columns}),
            candidates, unary, ops::kDFlash2SelectorPackedWidth, Config::block_size, lattice,
            state.execution.device.stream);
        ops::dflash2_trace_path(lattice, Config::block_size, drafts,
                                state.execution.device.stream);
        state.dflash.phase_timing.end_selector();
    state.execution.work.reset();
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DFlashEnvelopes envelopes,
                              ops::CausalAttentionExecutionEnvelope target_envelope,
                              bool allow_greedy_target_head) {
    return [&state, batch_size, k, envelopes, target_envelope, allow_greedy_target_head] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        if constexpr (Variant::DFlashConfig::is_v2) {
            state.dflash.phase_timing.begin_round();
        }
        qwen3_6::DFlashDecodeState& frame = state.frame;
        const std::int32_t width          = static_cast<std::int32_t>(k) + 1;
        std::uint32_t proposal_k          = k;
        if constexpr (Variant::DFlashConfig::is_v2) {
            // DFlash2's learned proposal lattice is fixed at its trained block size. Adaptive K
            // therefore keeps proposal generation at K7 and narrows only target verification,
            // while honoring a caller-configured draft window smaller than the trained block.
            proposal_k = std::min<std::uint32_t>(
                Variant::DFlashConfig::block_size - 1U,
                static_cast<std::uint32_t>(frame.draft_tokens.ne[0]));
        }
        const std::int32_t proposal_width = static_cast<std::int32_t>(proposal_k) + 1;
        bool all_greedy_target_rows       = allow_greedy_target_head;
        for (std::int32_t row = 0; row < batch_size; ++row) {
            all_greedy_target_rows =
                all_greedy_target_rows && state.host_ingress.sampling[row].temperature <= 0.0F;
        }
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        Tensor anchors            = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers          = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts     = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents            = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows          = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows        = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor active_lanes       = frame.active_lanes.slice(0, 0, batch_size);
        Tensor state_sources      = frame.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = frame.state_destination_slots.slice(0, 0, batch_size);
        Tensor append_positions   = frame.append_positions.slice(1, 0, batch_size);
        Tensor append_counts      = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts             = frame.draft_tokens.slice(1, 0, batch_size);
        Tensor q2_l1_accepted     = frame.q2_l1_accepted.slice(0, 0, batch_size);
        Tensor verify_ids         = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions   = frame.proposal_positions.slice(1, 0, batch_size);
        Tensor target_tokens      = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits      = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden      = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden    = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens    = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts    = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted           = frame.accepted_drafts.slice(0, 0, batch_size);
        const bool live_q4_verifier = dflash_state(state).local_q4_shadow != nullptr;
        const bool q4_primary       = live_q4_verifier && dflash_state(state).q4_primary;

        state.execution.work.reset();
        Tensor compact_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, proposal_width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, active_lanes,
                                   context_starts, frontiers, compact_features, append_positions,
                                   append_counts, state.execution.device.stream);
        append_context_impl<Variant>(state, compact_features, append_positions, append_counts,
                                     state_destinations, dflash_rows,
                                     {0, static_cast<std::uint32_t>(proposal_width)});

        if (q4_primary) {
            // Q4 is the live verifier/serving proposal. Keep Q2 resident and independently
            // dual-written for fallback, but do not spend another full DFlash2 pass generating a
            // proposal that the higher-fidelity Q4 pass would immediately replace.
            propose_batch_impl<Variant>(state, frame, batch_size, proposal_k, envelopes, true);
            CUDA_CHECK(cudaMemcpyAsync(q2_l1_accepted.data, extents.data,
                                       static_cast<std::size_t>(batch_size) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
        } else {
            propose_batch_impl<Variant>(state, frame, batch_size, proposal_k, envelopes);
        }
        if (live_q4_verifier && !q4_primary) {
            // Preserve the Q2 proposal in a dedicated contiguous round-state slot while the Q4
            // verifier reruns the DFlash attention stack over the independently written Q4 cache.
            // append_positions has width k+1 and cannot be sliced on dim 0 without leaving a
            // width-sized batch stride.
            Tensor q2_drafts = frame.q2_drafts.slice(1, 0, batch_size);
            CUDA_CHECK(cudaMemcpyAsync(q2_drafts.data, drafts.data, drafts.bytes(),
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
            propose_batch_impl<Variant>(state, frame, batch_size, proposal_k, envelopes, true);
            // The Q4 pass is a refinement, not merely a veto. Record the common Q2->Q4 prefix for
            // acceptance telemetry, but send the complete Q4 proposal to the authoritative target
            // so a Q2 disagreement can be replaced by the higher-fidelity candidate.
            ops::speculative_common_prefix(q2_drafts, drafts, extents, q2_l1_accepted,
                                           append_counts, state.execution.device.stream);
        }
        Tensor verify_drafts              = drafts;
        Tensor verify_ids_for_target      = verify_ids;
        Tensor target_positions_for_target = target_positions;
        Tensor target_tokens_for_target   = target_tokens;
        Tensor target_logits_for_target   = target_logits;
        Tensor target_hidden_for_target   = target_hidden;
        Tensor licensed_tokens_for_target = licensed_tokens;
        if (k != proposal_k) {
            verify_drafts = Tensor(frame.adaptive_drafts.data, DType::I32,
                                   {static_cast<std::int32_t>(k), batch_size});
            CUDA_CHECK(cudaMemcpy2DAsync(
                verify_drafts.data, static_cast<std::size_t>(k) * sizeof(TokenId), drafts.data,
                static_cast<std::size_t>(proposal_k) * sizeof(TokenId),
                static_cast<std::size_t>(k) * sizeof(TokenId), static_cast<std::size_t>(batch_size),
                cudaMemcpyDeviceToDevice, state.execution.device.stream));
            verify_ids_for_target =
                Tensor(frame.adaptive_verify_ids.data, DType::I32, {width, batch_size});
            target_positions_for_target =
                Tensor(frame.adaptive_target_positions.data, DType::I32, {width, batch_size});
            target_tokens_for_target =
                Tensor(frame.adaptive_target_tokens.data, DType::I32, {width, batch_size});
            target_logits_for_target = Tensor(frame.adaptive_target_logits.data, DType::BF16,
                                              {TextConfig::output_rows, width, batch_size});
            target_hidden_for_target = Tensor(frame.adaptive_target_hidden.data, DType::BF16,
                                              {TextConfig::hidden, width, batch_size});
            licensed_tokens_for_target =
                Tensor(frame.adaptive_licensed_tokens.data, DType::I32, {width, batch_size});
        }
        const GdnReplayRecords* replay_records_for_target = state.execution.replay_records;
        if constexpr (Variant::DFlashConfig::is_v2) {
            if (state.dflash.adaptive_replay_layout.has_value()) {
                // The global replay plane is graph-shaped at the configured width. Adaptive K
                // uses a dense per-width scratch plane so GDN's record-producing kernels retain
                // their contiguous tensor contract; the matching fold plan is used at commit.
                state.dflash.adaptive_replay_records =
                    state.dflash.make_adaptive_replay_records(width);
                state.dflash.adaptive_replay_fold.emplace(
                    state.dflash.adaptive_replay_records,
                    state.execution.linear_attention.all_layers_view());
                replay_records_for_target = &state.dflash.adaptive_replay_records;
            }
        }
        ops::speculative_prepare_verify_inputs(
            anchors, verify_drafts, frontiers, extents, verify_ids_for_target,
            target_positions_for_target, state.execution.device.stream);

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, active_lanes, valid_columns, width, batch_size);
        if constexpr (Variant::DFlashConfig::is_v2) {
            state.dflash.phase_timing.begin_verifier();
        }
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids                     = verify_ids_for_target,
                                 .cache_positions         = target_positions_for_target,
                                 .rope_positions          = target_positions_for_target,
                                 .valid_columns           = valid_columns,
                                 .kv_table_rows           = text_rows,
                                 .state_source_slots      = state_sources,
                                 .state_destination_slots = state_destinations,
                                 .target_hidden           = target_hidden_for_target,
                                 .target_logits           = target_logits_for_target,
                                 .target_tokens           = target_tokens_for_target,
                                 .greedy_target_head      = all_greedy_target_rows,
                                 .drafts                  = verify_drafts,
                                 .current_extents         = extents,
                                 .frontiers               = frontiers,
                                 .anchors                 = anchors,
                                 .licensed_tokens         = licensed_tokens_for_target,
                                 .licensed_counts         = licensed_counts,
                                 .accepted_drafts         = accepted,
                                 .selected_hidden         = selected_hidden,
                                 .replay_records          = replay_records_for_target,
                                 .sampling                = frame.sampling,
                                 .feature_sink            = &sink,
                             },
                             target_envelope);
        if constexpr (Variant::DFlashConfig::is_v2) {
            state.dflash.phase_timing.end_verifier();
        }
        if (k != proposal_k) {
            // Pending-resolution correction uses the graph-shaped frame tensor after this call.
            // Preserve the packed adaptive hidden columns there before the host egress copy.
            const std::size_t element_bytes = dtype_size(DType::BF16);
            const std::size_t column_bytes =
                static_cast<std::size_t>(TextConfig::hidden) * element_bytes;
            const std::size_t source_pitch = column_bytes * static_cast<std::size_t>(width);
            const std::size_t target_pitch = column_bytes * static_cast<std::size_t>(proposal_width);
            auto* target_base = static_cast<std::byte*>(frame.target_hidden.data);
            const auto* source_base =
                static_cast<const std::byte*>(target_hidden_for_target.data);
            for (std::int32_t row = 0; row < batch_size; ++row) {
                CUDA_CHECK(cudaMemcpy2DAsync(
                    target_base + static_cast<std::size_t>(row) * target_pitch, target_pitch,
                    source_base + static_cast<std::size_t>(row) * source_pitch, source_pitch,
                    column_bytes, static_cast<std::size_t>(width), cudaMemcpyDeviceToDevice,
                    state.execution.device.stream));
            }
            CUDA_CHECK(cudaMemcpy2DAsync(
                frame.licensed_tokens.data,
                static_cast<std::size_t>(proposal_width) * sizeof(TokenId),
                licensed_tokens_for_target.data, static_cast<std::size_t>(width) * sizeof(TokenId),
                static_cast<std::size_t>(width) * sizeof(TokenId), static_cast<std::size_t>(batch_size),
                cudaMemcpyDeviceToDevice, state.execution.device.stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::DFlashDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
        if (live_q4_verifier) {
            CUDA_CHECK(cudaMemcpyAsync(state.host_egress.l0_l1_accepted.data(),
                                       q2_l1_accepted.data,
                                       static_cast<std::size_t>(batch_size) * sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, state.execution.device.stream));
        }
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, DFlashEnvelopes envelopes,
                                 ops::CausalAttentionExecutionEnvelope target_envelope,
                                 bool greedy_target_head,
                                 DecodeGraphDefinition& definition) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope,
                                         greedy_target_head);
    capture_graph(state, definition, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         DFlashEnvelopes envelopes,
                         ops::CausalAttentionExecutionEnvelope target_envelope,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, envelopes, target_envelope,
                                         executable == nullptr);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
