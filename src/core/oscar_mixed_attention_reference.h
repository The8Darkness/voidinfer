#pragma once

#include "core/oscar_mixed_cache_layout.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

enum class OscarMixedReadTier : std::uint8_t {
    ProtectedPrefixBFloat16 = 0,
    HistoricalOscarInt2G128 = 1,
    RecentBFloat16 = 2,
};

struct OscarMixedAttentionTrace {
    std::uint32_t model_layer = 0;
    std::uint32_t query_token = 0;
    std::vector<std::uint32_t> logical_positions;
    std::vector<OscarMixedReadTier> tiers;
    std::uint32_t prefix_count = 0;
    std::uint32_t historical_count = 0;
    std::uint32_t recent_count = 0;
    std::vector<float> rotated_q;
    std::vector<float> score_logits;
    std::vector<float> softmax;
    std::vector<float> rotated_av;
    std::vector<float> recovered_output;
    // Populated only when NINFER_OSCAR_D4_1_PROFILE=1. Values are wall-clock microseconds for
    // this scalar reference call and are intentionally diagnostic rather than a performance API.
    double q_rotation_us = 0.0;
    double int2_k_decode_us = 0.0;
    double qk_us = 0.0;
    double softmax_us = 0.0;
    double int2_v_decode_us = 0.0;
    double av_us = 0.0;
    double rv_inverse_us = 0.0;
    double total_us = 0.0;
};

// Deliberately slow attention reader over the validated D2.2 mixed representation. It is a
// reference/diagnostic path only: rows are decoded into a temporary per-query buffer, never into
// a persistent BF16 shadow, and no serving or CUDA dispatch is exposed.
class OscarMixedAttentionReader {
public:
    OscarMixedAttentionReader(const OscarMixedAgingLayerCache& cache,
                              std::span<const float> q_original,
                              std::span<const float> r_k,
                              std::span<const float> r_v);

    [[nodiscard]] OscarMixedAttentionTrace read(std::uint32_t query_token) const;

private:
    const OscarMixedAgingLayerCache* cache_ = nullptr;
    std::span<const float> q_original_;
    std::span<const float> r_k_;
    std::span<const float> r_v_;

    void validate_inputs() const;
    void read_row(std::uint32_t logical_token, std::uint32_t kv_head, bool key,
                  float* output) const;
};

} // namespace ninfer
