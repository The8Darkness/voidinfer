#pragma once
#include <cstddef>

namespace ninfer::exl3 {
// Ordered stages in one full-attention forward, on the same stream. Shared
// projection callbacks must finish private scatter before returning to the layer.
enum class Exl3AttentionScratchStep : unsigned {
    input_norm, qkv_projection, attention, output_projection, residual_norm,
    mlp_projection, activation, down_projection, residual_output, diagnostic_trace
};
struct Exl3AttentionScratchLifetime {
    Exl3AttentionScratchStep first,last;
    std::size_t bytes=0;
};
inline constexpr bool exl3_attention_scratch_can_alias(Exl3AttentionScratchLifetime a,
    Exl3AttentionScratchLifetime b,bool same_stream,bool diagnostic_trace,bool pending_consumer) noexcept {
    return same_stream && !diagnostic_trace && !pending_consumer && a.bytes && b.bytes &&
        a.first<=a.last && b.first<=b.last && (a.last<b.first || b.last<a.first);
}
// input_norm (buffer0) ends after Q/K/V input projections; mlp_in (buffer11)
// starts at post-attention residual normalization. The existing trace exposes
// both, extending their lifetimes and forbidding this alias in diagnostic mode.
}
