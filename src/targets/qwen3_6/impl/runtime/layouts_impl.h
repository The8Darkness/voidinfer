#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/dflash2_selector_lattice.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"

#include "ops/kv_cache/d256_profile.h"

#include <algorithm>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

bool dflash_full_bf16_enabled() noexcept {
    const char* value = std::getenv("NINFER_DFLASH_FULL_BF16");
    // This is an explicit comparison mode for the DFlash full-attention cache. The serving
    // hierarchy remains OSCAR-Q2 by default; setting the switch restores the BF16 target cache
    // while leaving the local speculative cache at its configured L0 format.
    return value != nullptr && value[0] == '1';
}

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

struct TargetKVCacheProfile {
    DType dtype;
    std::int32_t quant_group;
};

TargetKVCacheProfile target_kv_cache_profile(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {DType::BF16, 0};
    case KvCacheStorage::Int8Group64:
        return {DType::I8, qwen3_6::kKvInt8QuantGroup};
    case KvCacheStorage::Fp8E4M3Row256:
        return {DType::FP8_E4M3FN, qwen3_6::kKvFp8QuantGroup};
    case KvCacheStorage::Nvfp4:
        return {DType::U8, qwen3_6::kKvNvfp4QuantGroup};
    case KvCacheStorage::VeriCacheNvfp4:
        // The hierarchical serving profile is deliberately Q2 on the device.  L1/L2 are
        // host-side representations; no exact BF16 target KV is reserved in VRAM.
        return {DType::U8, ops::kD256OscarQuantGroup};
    }
    throw std::invalid_argument("unknown KV-cache storage profile");
}

TargetKVCacheProfile mtp_kv_cache_profile(KvCacheStorage storage) {
    if (storage == KvCacheStorage::VeriCacheNvfp4) {
        return {DType::U8, ops::kD256OscarQuantGroup};
    }
    return target_kv_cache_profile(storage);
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen3.6 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots,
        "Qwen3.6 StateImage slot count exceeds int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
                     .full_attention_layers     = TextConfig::full_attention_layers(),
                     .mtp_layers                = TextConfig::mtp_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = TextConfig::kv_heads,
                     .attention_head_dim        = TextConfig::head_dim,
                     .kv_dtype                  = plan.kv_dtype,
                     .kv_quant_group            = plan.kv_quant_group,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                     .mtp_kv_dtype               = plan.mtp_kv_dtype,
                     .mtp_kv_quant_group         = plan.mtp_kv_quant_group,
                     .mtp_kv_profile_explicit    = plan.mtp_kv_profile_explicit,
                 });
    qwen3_6::StateImageSpec state_image_spec{
        .linear =
            {
                .layers         = TextConfig::gdn_layers(),
                .conv_channels  = TextConfig::convolution_dim,
                .conv_width     = TextConfig::gdn_conv_state_width,
                .value_heads    = TextConfig::gdn_value_heads,
                .value_head_dim = TextConfig::gdn_value_head_dim,
                .key_head_dim   = TextConfig::gdn_key_head_dim,
                .slot_count     = state_image_slots,
                .conv_dtype     = DType::BF16,
            },
        .hidden = TextConfig::hidden,
    };
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            const bool dflash_nvfp4 = plan.kv_storage == KvCacheStorage::VeriCacheNvfp4;
            std::uint32_t protected_recent = 0;
            std::uint32_t protected_anchor = 0;
            if (dflash_nvfp4 && plan.hierarchical_vericache.enabled &&
                (plan.hierarchical_vericache.protected_recent_tokens != 0 ||
                 plan.hierarchical_vericache.protected_sink_tokens != 0 ||
                 plan.hierarchical_vericache.protected_pivot_tokens != 0)) {
                const std::uint64_t anchor_request =
                    static_cast<std::uint64_t>(plan.hierarchical_vericache.protected_sink_tokens) +
                    plan.hierarchical_vericache.protected_pivot_tokens;
                protected_anchor = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                    DFlashConfig::local_capacity, anchor_request));
                const std::uint32_t recent_capacity_limit =
                    static_cast<std::uint32_t>(DFlashConfig::local_capacity) - protected_anchor;
                const std::uint32_t requested = std::min(
                    plan.hierarchical_vericache.protected_recent_tokens, recent_capacity_limit);
                // The cyclic sidecar uses a mask for zero-overhead slot selection.  Keep the
                // largest supported power of two so a user-supplied non-power-of-two request is
                // conservative rather than allocating more state than requested.
                std::uint32_t power_of_two = 1;
                if (requested != 0) {
                    while (power_of_two <= requested / 2U) { power_of_two <<= 1U; }
                    protected_recent = power_of_two;
                }
            }
            // The hierarchical DFlash route is the OSCAR experiment: L0 stays in VRAM at Q2 by
            // default (the option still permits a Q4 storage control), while its independent host
            // mirror is always a true OSCAR-Q4 layout.  The legacy non-hierarchical route keeps
            // the existing NVFP4/qgroup16 contract.
            const bool use_oscar = plan.hierarchical_vericache.enabled;
            const std::uint8_t l0_bits =
                use_oscar ? plan.hierarchical_vericache.l0_bits : static_cast<std::uint8_t>(4);
            state_image_spec.dflash_local = qwen3_6::DFlashLocalStateSpec{
                .layers             = DFlashConfig::local_layers,
                .capacity           = DFlashConfig::local_capacity,
                .protected_capacity = protected_recent,
                .protected_anchor_capacity = protected_anchor,
                .kv_heads            = DFlashConfig::kv_heads,
                .head_dim            = DFlashConfig::head_dim,
                .dtype               = dflash_nvfp4 ? DType::U8 : DType::BF16,
                .quant_group         = dflash_nvfp4
                                           ? (use_oscar ? 128 : qwen3_6::kKvNvfp4QuantGroup)
                                           : 0,
                .quant_bits          = dflash_nvfp4 ? l0_bits : static_cast<std::uint8_t>(0),
                .quantization        = dflash_nvfp4
                                           ? (use_oscar
                                                  ? CyclicKVCacheQuantization::OscarAffine
                                                  : CyclicKVCacheQuantization::Nvfp4)
                                           : CyclicKVCacheQuantization::Auto,
            };
            if (dflash_nvfp4) {
                // L1 is a separate OSCAR-Q4 host mirror when hierarchy is enabled. The current
                // shadow-free path promotes the resident L0 Q2 representation through bounded
                // StateImage conversion; L2 remains the FP16 authoritative host state.
                auto host_local = *state_image_spec.dflash_local;
                host_local.quant_bits = 4;
                host_local.quant_group = use_oscar ? 128 : qwen3_6::kKvNvfp4QuantGroup;
                host_local.quantization = use_oscar
                                               ? CyclicKVCacheQuantization::OscarAffine
                                               : CyclicKVCacheQuantization::Nvfp4;
                state_image_spec.dflash_host_local = host_local;
            }
        }
    }
    out.state_images = qwen3_6::plan_state_image_device_pool(builder, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder, GdnReplayRecordSpec{
                         .layers          = TextConfig::gdn_layers(),
                         .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                         .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                         .conv_channels   = TextConfig::convolution_dim,
                         .qk_heads        = TextConfig::gdn_key_heads,
                         .value_heads     = TextConfig::gdn_value_heads,
                         .key_dim         = TextConfig::gdn_key_head_dim,
                         .value_dim       = TextConfig::gdn_value_head_dim,
                     });
    }
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            constexpr bool dflash_v2 = DFlashConfig::is_v2;
            const bool full_bf16 = dflash_v2 && dflash_full_bf16_enabled();
            const bool full_oscar = dflash_v2 && !full_bf16;
            const DType full_code_dtype = full_oscar ? DType::U8 : DType::BF16;
            const std::int32_t full_code_extent =
                full_oscar ? (DFlashConfig::head_dim * 2 + 7) / 8 : DFlashConfig::head_dim;
            const std::int32_t full_scale_extent =
                full_oscar ? ops::kD256OscarScaleExtent : 0;
            const DType full_scale_dtype = DType::BF16;
            KVPageGeometry full_geometry{
                .page_tokens        = kPagedKVPageSize,
                .device_plane_order = PagedKVPlaneOrder::HeadMajor,
                .planes = [&] {
                    std::vector<KVPlaneGeometry> planes;
                    planes.reserve(full_oscar ? 4U : 2U);
                    planes.push_back(
                        {full_code_dtype, full_code_extent, DFlashConfig::kv_heads, 256});
                    planes.push_back(
                        {full_code_dtype, full_code_extent, DFlashConfig::kv_heads, 256});
                    if (full_oscar) {
                        planes.push_back(
                            {full_scale_dtype, full_scale_extent, DFlashConfig::kv_heads, 256});
                        planes.push_back(
                            {full_scale_dtype, full_scale_extent, DFlashConfig::kv_heads, 256});
                    }
                    return planes;
                }(),
            };
            dflash.full = qwen3_6::PagedKVCacheLayout{
                .pages = plan_device_kv_page_pool(
                    builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages,
                                                  .geometry         = std::move(full_geometry)}),
                .execution_tables = plan_kv_execution_tables(
                    builder,
                    KVExecutionTableSpec{
                        .logical_page_capacity = logical_pages,
                        .table_rows            = static_cast<std::int32_t>(plan.max_concurrency),
                    }),
                .layers      = 1,
                .max_context = plan.capacity,
                .kv_heads    = DFlashConfig::kv_heads,
                .head_dim    = DFlashConfig::head_dim,
                .dtype       = full_code_dtype,
                .quant_group = full_oscar ? ops::kD256OscarQuantGroup : 0,
            };
            dflash.prefill_features = add_tensor(
                builder, DType::BF16, {DFlashConfig::feature_rows, effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(builder, DType::BF16,
                                                  {DFlashConfig::feature_rows,
                                                   static_cast<std::int32_t>(plan.draft_window + 1U),
                                                   static_cast<std::int32_t>(plan.max_concurrency)},
                                                  "DFlash pending target features");
        }
    }

    out.round = qwen3_6::begin_round_state_layout(
        builder, qwen3_6::RoundStateSpec{.hidden         = TextConfig::hidden,
                                         .output_rows    = TextConfig::output_rows,
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .enable_mtp     = plan.features.mtp(),
                                         .enable_dflash  = plan.features.dflash()});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, effective_prefill_chunk}, "step prefill hidden");
    qwen3_6::complete_round_state_layout(builder, out.round);
    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts =
        add_tensor(builder, DType::I32,
                   {TextConfig::token_domain, static_cast<std::int32_t>(plan.max_concurrency)},
                   "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "sampling config");
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::CausalAttentionExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace_recipe::text_prefill_roots<TextConfig>(
            layout, tokens, plan.features.vision ? 3 : 0, plan.features.vision ? tokens : 0);
    };
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, qwen3_6::TextPhase phase,
                                     std::int32_t batch_size, std::int32_t min_width,
                                     std::int32_t max_width,
                                     ops::CausalAttentionExecutionEnvelope envelope) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(plan.weights_profile,
                                                                               phase, first, last));
        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {TextConfig::head_dim, TextConfig::query_heads, TextConfig::kv_heads},
                            plan.kv_dtype, envelope, batch_size, min_width, max_width));
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                               std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width) {
        auto stage = layout.scope();
        (void)workspace_recipe::gdn_control<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_norm_control_projection_workspace_capacity_bytes(first, last));
        (void)workspace_recipe::gdn_projection<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else if (path == GdnWorkspacePath::ReplayRecord) {
            scratch(layout, Variant::gdn_input_projection_record_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else {
            (void)workspace_recipe::gdn_prefill_conv<TextConfig>(layout, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        }
        (void)workspace_recipe::gdn_recurrent_output<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Prefill) {
            scratch(layout,
                    ops::gated_delta_net_workspace_capacity_bytes(
                        TextConfig::gdn_key_heads, TextConfig::gdn_value_heads, true, first, last));
        }
        (void)workspace_recipe::gdn_normalized_output<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3_6::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width,
                                 ops::CausalAttentionExecutionEnvelope envelope) {
        attention_stage(layout, first, last, phase, batch_size, min_width, max_width, envelope);
        gdn_stage(layout, first, last, phase, path, batch_size, min_width, max_width);
        post_mixer_stage(layout, first, last, phase);
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem<TextConfig>(layout, tokens, !preembedded);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
        (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {TextConfig::head_dim, TextConfig::query_heads, TextConfig::kv_heads},
                            plan.mtp_kv_dtype, envelope, 1, tokens, tokens));
        (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            scratch(layout, Variant::mtp_kv_projection_workspace_capacity_bytes(first, last));
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
        }
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, Variant::mtp_q_gate_projection_workspace_capacity_bytes(1, 1));
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {TextConfig::head_dim, TextConfig::query_heads, TextConfig::kv_heads},
                            plan.mtp_kv_dtype, text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(1, 1));
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
         ++batch) {
        WorkspaceLayoutBuilder ordinary;
        matrix(ordinary, DType::BF16, TextConfig::hidden, batch);
        target_body(ordinary, batch, batch, qwen3_6::TextPhase::Verify, GdnWorkspacePath::Snapshot,
                    batch, 1, 1, text_envelope);
        scratch(ordinary,
                ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, batch, batch));
        out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            TextConfig::token_domain, drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, TextConfig::hidden, aggregate);
            target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
                scratch(layout,
                        Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
                (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
                scratch(layout,
                        ops::causal_softmax_attention_workspace_capacity_bytes(
                            {TextConfig::head_dim, TextConfig::query_heads, TextConfig::kv_heads},
                            plan.mtp_kv_dtype, text_envelope, batch, width, width));
                (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
                scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    TextConfig::token_domain, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.dflash()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const auto dflash_context_capacity = [&](std::int32_t tokens, bool compact_input) {
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlashConfig::feature_rows, tokens);
                }
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                if constexpr (DFlashConfig::is_v2) {
                    for (std::int32_t layer_index = 0;
                         layer_index < DFlashConfig::layers; ++layer_index) {
                        auto layer = layout.scope();
                        (void)layer_index;
                        (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                    }
                } else {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                if constexpr (DFlashConfig::is_v2) {
                    matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                    matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                    for (std::int32_t layer_index = 0;
                         layer_index < DFlashConfig::layers; ++layer_index) {
                        auto layer = layout.scope();
                        (void)layer_index;
                        matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                        {
                            auto attention = layout.scope();
                            (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                            matrix(layout, DType::BF16, DFlashConfig::conv_projection_rows,
                                   tokens);
                            matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                            scratch(layout, ops::sliding_window_attention_workspace_capacity_bytes(
                                                 {DFlashConfig::head_dim, DFlashConfig::query_heads,
                                                  DFlashConfig::kv_heads},
                                                 DFlashConfig::local_capacity,
                                                 {0, plan.capacity}, width, width, batch));
                            matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                            matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                        }
                        {
                            auto mlp = layout.scope();
                            (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                            matrix(layout, DType::BF16, DFlashConfig::conv_projection_rows,
                                   tokens);
                            matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                            matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                            matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                        }
                    }
                    matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                    const std::int32_t proposal_rows =
                        plan.proposal_head == ProposalHead::Optimized ? Variant::draft_head_rows
                                                                        : TextConfig::output_rows;
                    matrix(layout, DType::BF16, proposal_rows, tokens);
                    if (plan.proposal_head == ProposalHead::Optimized) {
                        matrix(layout, DType::I32, ops::kDFlash2ProposalShortlistTopK, tokens);
                        matrix(layout, DType::FP32, ops::kDFlash2ProposalShortlistTopK, tokens);
                        matrix(layout, DType::I32, ops::kDFlash2SelectorTopK, tokens * 2);
                        matrix(layout, DType::FP32, ops::kDFlash2SelectorTopK, tokens * 2);
                    }
                    matrix(layout, DType::I32, DFlashConfig::selector_top_k, tokens);
                    matrix(layout, DType::FP32, DFlashConfig::selector_top_k, tokens);
                    matrix(layout, DType::BF16, DFlashConfig::selector_rank, tokens);
                    matrix(layout, DType::BF16, DFlashConfig::selector_rank,
                           DFlashConfig::selector_top_k * tokens);
                    matrix(layout, DType::I32, DFlashConfig::selector_top_k, tokens);
                    matrix(layout, DType::BF16, DFlashConfig::selector_rank,
                           DFlashConfig::selector_top_k * tokens);
                    matrix(layout, DType::FP32, ops::kDFlash2SelectorPackedWidth, tokens);
                } else {
                    matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                    {
                        auto attention = layout.scope();
                        (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                        scratch(layout,
                                std::max(ops::sliding_window_attention_workspace_capacity_bytes(
                                             {DFlashConfig::head_dim, DFlashConfig::query_heads,
                                              DFlashConfig::kv_heads},
                                             DFlashConfig::local_capacity, {0, plan.capacity}, width,
                                             width, batch),
                                         ops::context_softmax_attention_workspace_capacity_bytes(
                                             {DFlashConfig::head_dim, DFlashConfig::query_heads,
                                              DFlashConfig::kv_heads},
                                             {0, plan.capacity}, width, width, batch)));
                        scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                            QType::W8G32_F16S, DFlashConfig::hidden,
                                            DFlashConfig::query_size, tokens, tokens));
                    }
                    {
                        auto mlp = layout.scope();
                        (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                        scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                            QType::W8G32_F16S, 2 * DFlashConfig::intermediate,
                                            DFlashConfig::hidden, tokens, tokens));
                        scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                            QType::W8G32_F16S, DFlashConfig::hidden,
                                            DFlashConfig::intermediate, tokens, tokens));
                    }
                    matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                    matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                    if (plan.proposal_head == ProposalHead::Optimized) {
                        matrix(layout, DType::BF16, Variant::draft_head_rows, drafts * batch);
                    } else {
                        matrix(layout, DType::BF16, TextConfig::output_rows, drafts * batch);
                    }
                }
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, TextConfig::hidden, aggregate);
                target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                        TextConfig::token_domain, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                out.dflash_round           = std::max({out.dflash_round, finish(target), accept,
                                                       dflash_context_capacity(aggregate, true), proposal});
            }
        }
    }

    out.general_capacity = std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill,
                                     out.mtp_round, out.dflash_context, out.dflash_round});
    out.capacity         = out.general_capacity;
    if (plan.features.vision) {
        const std::uint32_t merged = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(plan.capacity, kMaximumVisionItemTokens));
        out.vision   = schedule::VisionContext::plan_workspace(merged, out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    if (options.max_context == 0 || options.max_context > Variant::maximum_context) {
        throw std::invalid_argument("max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash:
        if (kMaximumDFlashDraftTokens == 0) {
            throw std::invalid_argument("DFlash is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumDFlashDraftTokens) {
            throw std::invalid_argument("DFlash draft window must be in [1,15]");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("DFlash and Vision cannot be enabled together");
        }
        break;
    }
    if (options.kv_cache == KvCacheStorage::VeriCacheNvfp4) {
        if (options.speculative.backend != SpeculativeBackend::Mtp &&
            options.speculative.backend != SpeculativeBackend::DFlash) {
            throw std::invalid_argument(
                "vericache-nvfp4 requires --spec mtp or --spec dflash; the exact target cache and compressed "
                "draft tier are separate");
        }
        if (TextConfig::head_dim != 256) {
            throw std::invalid_argument("vericache-nvfp4 requires a 256-wide attention head");
        }
    }
    if (device.sm() != 120) {
        throw std::invalid_argument("Qwen3.6 family runtime requires compute capability 12.0");
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->weights_profile     = inputs.weights_profile;
    impl->capacity            = inputs.capacity;
    impl->main_page_groups    = main_page_groups;
    impl->kv_capacity         = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency     = inputs.max_concurrency;
    impl->prefill_chunk       = inputs.prefill_chunk;
    impl->draft_window        = inputs.draft_window;
    impl->speculative_backend = inputs.speculative_backend;
    impl->kv_storage          = inputs.kv_storage;
    impl->kv_dtype            = inputs.kv_dtype;
    impl->kv_quant_group      = inputs.kv_quant_group;
    impl->mtp_kv_dtype        = inputs.mtp_kv_profile_explicit ? inputs.mtp_kv_dtype
                                                                 : inputs.kv_dtype;
    impl->mtp_kv_quant_group  = inputs.mtp_kv_profile_explicit ? inputs.mtp_kv_quant_group
                                                                : inputs.kv_quant_group;
    impl->mtp_kv_profile_explicit = inputs.mtp_kv_profile_explicit;
    impl->proposal_head       = inputs.proposal_head;
    impl->features            = inputs.features;
    impl->use_cuda_graph      = inputs.use_cuda_graph;
    if constexpr (Variant::DFlashConfig::is_v2) {
        if (impl->speculative_backend == SpeculativeBackend::DFlash) {
            // DFlash2's selector trace is now a deterministic device-side
            // single-lane operation, so the complete proposal/trace/verify
            // transaction can be captured in the existing graph profiles.
            impl->draft_window   = Variant::DFlashConfig::block_size - 1U;
        }
    }
    impl->device              = inputs.device;
    impl->context_cache       = inputs.context_cache;
    impl->hierarchical_vericache = inputs.hierarchical_vericache;
    impl->persistent          = persistent_layout(*impl);
    impl->workspace           = build_workspace_plan(*impl);
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        if (impl->speculative_backend == SpeculativeBackend::None) {
            impl->graph_allowance_bytes = checked_mul(12ULL * kMiB, impl->max_concurrency,
                                                      "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles = mtp_graph_profiles(impl->capacity, impl->draft_window);
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) {
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + 2ULL * impl->draft_window);
                    return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
        } else {
            const auto class_allowance = [&](std::uint32_t batch_size) {
                const auto profiles =
                    dflash_graph_profiles(impl->capacity, impl->draft_window, batch_size);
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible = std::min<std::uint64_t>(
                            impl->capacity,
                            static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL);
                        return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
                    },
                    "DFlash graph allowance");
            };
            for (std::uint32_t batch_size = 1; batch_size <= impl->max_concurrency; ++batch_size) {
                impl->graph_allowance_bytes =
                    checked_add(impl->graph_allowance_bytes, class_allowance(batch_size),
                                "DFlash exact-b graph allowance");
            }
            // The default C1 greedy route has a second graph family because fused target-head
            // argmax changes the captured node sequence. Reserve only one additional C1 family;
            // C2/C4/C8 continue to share the existing full-logit graph set.
            impl->graph_allowance_bytes =
                checked_add(impl->graph_allowance_bytes, class_allowance(1U),
                            "DFlash greedy C1 graph allowance");
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile) {
    validate_target_options(device, options);
    const TargetKVCacheProfile kv_profile = target_kv_cache_profile(options.kv_cache);
    const TargetKVCacheProfile mtp_profile = mtp_kv_cache_profile(options.kv_cache);

    SequencePlanningInputs inputs{
        .weights_profile     = weights_profile,
        .capacity            = options.max_context,
        .max_concurrency     = options.max_concurrency,
        .prefill_chunk       = std::min(options.prefill_chunk, options.max_context),
        .draft_window        = options.speculative.draft_tokens,
        .speculative_backend = options.speculative.backend,
        .kv_dtype            = kv_profile.dtype,
        .kv_quant_group      = kv_profile.quant_group,
        .proposal_head       = options.speculative.proposal_head,
        .features            = qwen3_6::startup_features(options),
        .use_cuda_graph      = options.use_cuda_graph,
        .device              = options.device,
        .context_cache       = options.context_cache,
        .hierarchical_vericache = options.hierarchical_vericache,
        .kv_storage          = options.kv_cache,
        .mtp_kv_dtype        = mtp_profile.dtype,
        .mtp_kv_quant_group  = mtp_profile.quant_group,
        .mtp_kv_profile_explicit = options.kv_cache == KvCacheStorage::VeriCacheNvfp4,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_6::detail::SequencePlannerImpl<Variant>>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.6 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.6 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.6 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
