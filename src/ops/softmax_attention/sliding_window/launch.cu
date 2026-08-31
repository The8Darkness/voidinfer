#include "ops/softmax_attention/sliding_window/launch.h"

#include "core/device.h"
#include "ops/softmax_attention/sliding_window/lowbit.cuh"
#include "ops/softmax_attention/sliding_window/nvfp4.cuh"
#include "ops/softmax_attention/sliding_window/oscar.cuh"
#include "ops/softmax_attention/sliding_window/kernel.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

int nvfp4_pair_mode() noexcept {
    const char* value = std::getenv("NINFER_NVFP4_PAIR");
    if (value == nullptr) return 2;
    if (value[0] == '2') return 2;
    return value[0] == '1' ? 1 : 0;
}

bool use_oscar_q2_grouped() noexcept {
    const char* value = std::getenv("NINFER_OSCAR_Q2_GROUPED");
    // Grouped Q2 is retained as an experiment. At short serving windows the
    // four barriers per row cost more than the duplicated scalar work, so the
    // natural one-warp route remains the default until a context-aware policy
    // selects it for a measured win.
    return value != nullptr && value[0] != '0';
}

template <int Tokens, class Launch>
void dispatch_token_case(Launch&& launch) {
    constexpr int Warps = (Tokens + 3) / 4;
    launch.template operator()<Tokens, Warps>();
}

template <class Launch>
void dispatch_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
#define NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(TOKENS)                                         \
    case TOKENS:                                                                                   \
        dispatch_token_case<TOKENS>(launch);                                                       \
        return
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(1);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(2);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(3);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(4);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(5);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(6);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(7);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(8);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(9);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(10);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(11);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(12);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(13);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(14);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(15);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(16);
#undef NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE
    default:
        throw std::invalid_argument("sliding_window_attention: unsupported T");
    }
}

} // namespace

SlidingWindowAttentionPlan
sliding_window_attention_resolve_plan(std::int32_t tokens,
                                      SlidingWindowAttentionExecutionEnvelope envelope,
                                      std::int32_t window) {
    if (window < 2 || (window & (window - 1)) != 0) {
        throw std::invalid_argument("sliding_window_attention plan: window must be a power of two");
    }
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("sliding_window_attention plan: T must be 1..16");
    }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("sliding_window_attention plan: invalid envelope");
    }
    // Graph envelopes whose longest context fits three key tiles avoid the second kernel and
    // workspace round trip. At four tiles, split-KV is already faster for every qualified T.
    constexpr std::uint32_t direct_context_limit = 96;
    const bool direct                            = envelope.max_context <= direct_context_limit;
    constexpr std::int32_t key_block             = 32;
    const std::uint32_t context_rows = std::min(
        envelope.max_context, static_cast<std::uint32_t>(window - 1));
    const std::int32_t context_tiles =
        static_cast<std::int32_t>((context_rows + key_block - 1u) / key_block);
    constexpr std::int32_t split_limit = 32;
    return {
        .route =
            direct ? SlidingWindowAttentionRoute::Direct : SlidingWindowAttentionRoute::SplitKv,
        .tokens         = tokens,
        .warps          = (tokens + 3) / 4,
        .split_capacity = direct ? 1 : std::min(split_limit, std::max(1, context_tiles)),
        .max_context    = static_cast<std::int32_t>(envelope.max_context),
        .window         = window,
    };
}

const char* sliding_window_attention_route_name(SlidingWindowAttentionRoute route) {
    switch (route) {
    case SlidingWindowAttentionRoute::Direct:
        return "direct";
    case SlidingWindowAttentionRoute::SplitKv:
        return "split_kv";
    }
    return "unknown";
}

void sliding_window_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& lanes, float scale,
                                     const CyclicKVCacheLayerView& context,
                                     const SlidingWindowAttentionPlan& plan, Tensor& partial_acc,
                                     Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                     cudaStream_t stream) {
    dispatch_tokens(q.ne[2], [&]<int Tokens, int Warps>() {
        const bool direct = plan.route == SlidingWindowAttentionRoute::Direct;
        if (plan.warps != Warps || plan.split_capacity < 1 ||
            plan.split_capacity > kSlidingWindowMaxCandidateSplit ||
            (direct && plan.split_capacity != 1)) {
            throw std::invalid_argument("sliding_window_attention: inconsistent plan");
        }
        constexpr int KeyBlock = 32;
        constexpr std::size_t SmemBytes =
            2u * KeyBlock * kContextQueryHeadDim * sizeof(__nv_bfloat16);
        if (direct) {
            const dim3 direct_grid(kContextQueryKVHeads, 1, q.ne[3]);
            sliding_window_attention_split_partial_kernel<Tokens, Warps, KeyBlock, true>
                <<<direct_grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(lanes.data),
                    static_cast<const __nv_bfloat16*>(context.k.data),
                    static_cast<const __nv_bfloat16*>(context.v.data),
                    static_cast<int>(context.padded_capacity), plan.max_context, plan.window, 1,
                    scale,
                    static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        const dim3 partial_grid(kContextQueryKVHeads, plan.split_capacity, q.ne[3]);
        sliding_window_attention_split_partial_kernel<Tokens, Warps, KeyBlock, false>
            <<<partial_grid, Warps * 32, SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(query_k.data),
                static_cast<const __nv_bfloat16*>(query_v.data),
                static_cast<const std::int32_t*>(positions.data),
                static_cast<const std::int32_t*>(valid_columns.data),
                static_cast<const std::int32_t*>(lanes.data),
                static_cast<const __nv_bfloat16*>(context.k.data),
                static_cast<const __nv_bfloat16*>(context.v.data),
                static_cast<int>(context.padded_capacity), plan.max_context, plan.window,
                plan.split_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());

        constexpr int ReduceWarps = 1;
        constexpr int ReduceRows  = kContextQueryQHeads * Tokens;
        const dim3 reduce_grid((ReduceRows + ReduceWarps - 1) / ReduceWarps, 1, q.ne[3]);
        sliding_window_attention_reduce_kernel<Tokens, KeyBlock, ReduceWarps>
            <<<reduce_grid, ReduceWarps * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(positions.data),
                static_cast<const std::int32_t*>(valid_columns.data), plan.max_context, plan.window,
                plan.split_capacity, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    });
}

void sliding_window_attention_nvfp4_launch(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& lanes, float scale,
    const CyclicKVCacheLayerView& context, std::int32_t max_context, std::int32_t window,
    Tensor& out, cudaStream_t stream) {
    if (max_context < 0 || window < 2 || (window & (window - 1)) != 0) {
        throw std::invalid_argument("sliding_window_attention NVFP4 launch has invalid bounds");
    }
    // The NVFP4 path groups the four query heads that share one KV head so their packed K/V row
    // is decoded once per block instead of once per query-head warp.
    const dim3 grid(static_cast<unsigned>(q.ne[2] * kCyclicKVCacheNvfp4KVHeads),
                    static_cast<unsigned>(q.ne[3]), 1);
    const auto launch = [&]<bool HasProtected>() {
        const int pair_mode = nvfp4_pair_mode();
        if (pair_mode != 0) {
            const dim3 pair_grid(static_cast<unsigned>(q.ne[2] * kCyclicKVCacheNvfp4KVHeads * 4),
                                 static_cast<unsigned>(q.ne[3]), 1);
            if (pair_mode == 2) {
                sliding_window_attention_nvfp4_kernel<HasProtected, true, true>
                    <<<pair_grid, 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(lanes.data),
                    static_cast<const std::uint8_t*>(context.k.data),
                    static_cast<const std::uint8_t*>(context.v.data),
                    static_cast<const std::uint8_t*>(context.k_scale.data),
                    static_cast<const std::uint8_t*>(context.v_scale.data),
                    static_cast<const __nv_bfloat16*>(context.protected_k.data),
                    static_cast<const __nv_bfloat16*>(context.protected_v.data),
                    static_cast<int>(context.padded_capacity), max_context, window,
                    static_cast<int>(context.protected_capacity),
                    static_cast<int>(context.protected_anchor_capacity),
                    static_cast<int>(context.protected_padded_capacity), q.ne[2], scale,
                    static_cast<__nv_bfloat16*>(out.data));
            } else {
                sliding_window_attention_nvfp4_kernel<HasProtected, true, false>
                    <<<pair_grid, 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(lanes.data),
                    static_cast<const std::uint8_t*>(context.k.data),
                    static_cast<const std::uint8_t*>(context.v.data),
                    static_cast<const std::uint8_t*>(context.k_scale.data),
                    static_cast<const std::uint8_t*>(context.v_scale.data),
                    static_cast<const __nv_bfloat16*>(context.protected_k.data),
                    static_cast<const __nv_bfloat16*>(context.protected_v.data),
                    static_cast<int>(context.padded_capacity), max_context, window,
                    static_cast<int>(context.protected_capacity),
                    static_cast<int>(context.protected_anchor_capacity),
                    static_cast<int>(context.protected_padded_capacity), q.ne[2], scale,
                    static_cast<__nv_bfloat16*>(out.data));
            }
        } else {
            sliding_window_attention_nvfp4_grouped_kernel<HasProtected><<<grid, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(query_k.data),
            static_cast<const __nv_bfloat16*>(query_v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(lanes.data),
            static_cast<const std::uint8_t*>(context.k.data),
            static_cast<const std::uint8_t*>(context.v.data),
            static_cast<const std::uint8_t*>(context.k_scale.data),
            static_cast<const std::uint8_t*>(context.v_scale.data),
            static_cast<const __nv_bfloat16*>(context.protected_k.data),
            static_cast<const __nv_bfloat16*>(context.protected_v.data),
            static_cast<int>(context.padded_capacity), max_context, window,
            static_cast<int>(context.protected_capacity),
            static_cast<int>(context.protected_anchor_capacity),
            static_cast<int>(context.protected_padded_capacity), q.ne[2], scale,
            static_cast<__nv_bfloat16*>(out.data));
        }
    };
    if (context.protected_capacity != 0 || context.protected_anchor_capacity != 0) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

void sliding_window_attention_lowbit_launch(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& lanes, float scale,
    const CyclicKVCacheLayerView& context, std::int32_t max_context, std::int32_t window,
    Tensor& out, cudaStream_t stream) {
    if (max_context < 0 || window < 2 || (window & (window - 1)) != 0) {
        throw std::invalid_argument("sliding_window_attention low-bit launch has invalid bounds");
    }
    const dim3 grid(static_cast<unsigned>(q.ne[2] * kCyclicKVCacheNvfp4KVHeads * 4),
                    static_cast<unsigned>(q.ne[3]), 1);
    const auto launch = [&]<int Bits>() {
        sliding_window_attention_lowbit_kernel<Bits><<<grid, 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(query_k.data),
            static_cast<const __nv_bfloat16*>(query_v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(lanes.data),
            static_cast<const std::uint8_t*>(context.k.data),
            static_cast<const std::uint8_t*>(context.v.data),
            static_cast<const std::uint8_t*>(context.k_scale.data),
            static_cast<const std::uint8_t*>(context.v_scale.data),
            static_cast<const __nv_bfloat16*>(context.protected_k.data),
            static_cast<const __nv_bfloat16*>(context.protected_v.data),
            static_cast<int>(context.padded_capacity), max_context, window,
            static_cast<int>(context.protected_capacity),
            static_cast<int>(context.protected_anchor_capacity),
            static_cast<int>(context.protected_padded_capacity), q.ne[2], scale,
            static_cast<__nv_bfloat16*>(out.data));
    };
    if (context.quant_bits == 2) {
        launch.template operator()<2>();
    } else if (context.quant_bits == 3) {
        launch.template operator()<3>();
    } else {
        throw std::invalid_argument("sliding_window_attention low-bit launch requires 2 or 3 bits");
    }
    CUDA_CHECK(cudaGetLastError());
}

void sliding_window_attention_oscar_launch(
    const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& lanes, float scale,
    const CyclicKVCacheLayerView& context, std::int32_t max_context, std::int32_t window,
    Tensor& out, cudaStream_t stream) {
    if (max_context < 0 || window < 2 || (window & (window - 1)) != 0) {
        throw std::invalid_argument("sliding_window_attention OSCAR launch has invalid bounds");
    }
    if (context.quant_bits == 2 && use_oscar_q2_grouped()) {
        const dim3 grouped_grid(static_cast<unsigned>(q.ne[2] * kCyclicKVCacheOscarKVHeads),
                                static_cast<unsigned>(q.ne[3]), 1u);
        const auto launch_grouped = [&]<bool HasProtected>() {
            sliding_window_attention_oscar_q2_grouped_kernel<HasProtected>
                <<<grouped_grid, 128, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(lanes.data),
                    static_cast<const std::uint8_t*>(context.k.data),
                    static_cast<const std::uint8_t*>(context.v.data),
                    static_cast<const __nv_bfloat16*>(context.k_scale.data),
                    static_cast<const __nv_bfloat16*>(context.v_scale.data),
                    static_cast<const __nv_bfloat16*>(context.protected_k.data),
                    static_cast<const __nv_bfloat16*>(context.protected_v.data),
                    static_cast<int>(context.padded_capacity), max_context, window,
                    static_cast<int>(context.protected_capacity),
                    static_cast<int>(context.protected_anchor_capacity),
                    static_cast<int>(context.protected_padded_capacity), q.ne[2], scale,
                    static_cast<__nv_bfloat16*>(out.data));
        };
        if (context.protected_capacity != 0 || context.protected_anchor_capacity != 0) {
            launch_grouped.template operator()<true>();
        } else {
            launch_grouped.template operator()<false>();
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 grid(static_cast<unsigned>(q.ne[2] * 32), static_cast<unsigned>(q.ne[3]), 1);
    const auto launch = [&]<int Bits>() {
        sliding_window_attention_oscar_kernel<Bits><<<grid, 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const __nv_bfloat16*>(query_k.data),
            static_cast<const __nv_bfloat16*>(query_v.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(lanes.data),
            static_cast<const std::uint8_t*>(context.k.data),
            static_cast<const std::uint8_t*>(context.v.data),
            static_cast<const __nv_bfloat16*>(context.k_scale.data),
            static_cast<const __nv_bfloat16*>(context.v_scale.data),
            static_cast<const __nv_bfloat16*>(context.protected_k.data),
            static_cast<const __nv_bfloat16*>(context.protected_v.data),
            static_cast<int>(context.padded_capacity), max_context, window,
            static_cast<int>(context.protected_capacity),
            static_cast<int>(context.protected_anchor_capacity),
            static_cast<int>(context.protected_padded_capacity), q.ne[2], scale,
            static_cast<__nv_bfloat16*>(out.data));
    };
    if (context.quant_bits == 2) {
        launch.template operator()<2>();
    } else if (context.quant_bits == 3) {
        launch.template operator()<3>();
    } else if (context.quant_bits == 4) {
        launch.template operator()<4>();
    } else {
        throw std::invalid_argument("OSCAR attention requires 2, 3, or 4 bits");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
