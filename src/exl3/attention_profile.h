#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cstddef>
#include <limits>

namespace ninfer::exl3 {
enum class Exl3SegmentedAttentionRoute {inherited,scalar,q_shared,gqa_pair,query_pair};
inline constexpr Exl3SegmentedAttentionRoute exl3_segmented_attention_route(
    bool supported,bool ordinary_fp16,int rows,int position,int offset,int capacity,
    int query_heads,int kv_heads,int dimension,bool query_pair,bool gqa_pair,bool q_shared) noexcept {
    if(!supported || !ordinary_fp16 || query_heads!=24 || kv_heads!=4 || dimension!=256 ||
        rows<1 || rows>16 || position<0 || offset<0 || capacity<rows ||
        position>capacity-rows || offset>capacity-rows-position)
        return Exl3SegmentedAttentionRoute::inherited;
    if(query_pair)return Exl3SegmentedAttentionRoute::query_pair;
    if(gqa_pair)return Exl3SegmentedAttentionRoute::gqa_pair;
    if(q_shared)return Exl3SegmentedAttentionRoute::q_shared;
    return Exl3SegmentedAttentionRoute::scalar;
}
// L0 OSCAR contexts keep only the sink and the recent window exact: verifier
// segments cover at most this many keys past the history (l0_oscar.cuh).
constexpr int kExl3L0ExactWindowKeys=2048;
inline bool exl3_l0_oscar_enabled() {
    const char* value=std::getenv("NINFER_EXL3_L0_OSCAR");
    return value && value[0]=='1' && value[1]==0;
}
inline int exl3_exact_attention_capacity(int capacity) {
    return exl3_l0_oscar_enabled() && capacity>kExl3L0ExactWindowKeys+64?
        kExl3L0ExactWindowKeys+64:capacity;
}
inline std::size_t exl3_exact_attention_score_bytes(int rows,int capacity) {
    capacity=exl3_exact_attention_capacity(capacity);
    if(rows<1 || rows>16 || capacity<rows)
        throw std::invalid_argument("exact attention scratch row/capacity geometry");
    // 100 floats per row and key: the 24 query-head score plane, or the
    // 64-key flash-segment slots (6 heads x (256 values + 2 stats) x 4 KV
    // heads / 64 keys = 96.75) so verifier rows keep 64-key segments.
    const auto row_bytes=static_cast<std::size_t>(rows)*100*sizeof(float);
    if(static_cast<std::size_t>(capacity)>std::numeric_limits<std::size_t>::max()/row_bytes)
        throw std::overflow_error("exact attention scratch byte extent");
    return row_bytes*static_cast<std::size_t>(capacity);
}
// Admission for the complete staged consumer, before either plane is consumed.
// Each score batch is within the native row menu and the last causal endpoint
// stays inside the working cache; the layer still selects its precise kernel.
inline constexpr bool exl3_direct_staged_attention_supported(bool profile,int rows,
    int position,int capacity,int score_rows) noexcept {
    return profile && rows>=1 && rows<=16 && score_rows>=1 && score_rows<=16 &&
        position>0 && capacity>=rows && position<=capacity-rows;
}
inline bool exl3_attention_query_pair_option(const char* value,bool scores_available) {
    if(value && std::strcmp(value,"0")!=0 && std::strcmp(value,"1")!=0)
        throw std::invalid_argument("exact attention query-pair must be 0 or 1");
    const bool enabled=value && std::strcmp(value,"1")==0;
    if(enabled && !scores_available)
        throw std::invalid_argument("query-pair requires parallel exact attention");
    return enabled;
}
// Numerical profile eligibility only; ownership, geometry and readiness are
// admitted separately by the actual layer/context binding.
struct Exl3AttentionProfile {
    bool scores=false,oscar=false,capture=false,numeric_splitk=false;
    bool k_half2=false,v_half2=false,gqa_pair=false,gqa_triple=false;
    bool gqa_six=false,six_scores=false,six_sharded=false,six_packed=false,six_softmax=false;
    constexpr bool segmented() const noexcept {
        const bool inherited= ((!k_half2 && !v_half2) || gqa_pair) &&
            !gqa_triple && !gqa_six && !six_scores && !six_sharded &&
            !six_packed && !six_softmax;
        // The selected six-score/six-softmax/triple-values route can preserve
        // its exact arithmetic while resolving immutable prefix rows through
        // the same bounded range descriptor as the inherited readers.
        // k_half2/v_half2/GQA-pair/GQA-triple are prerequisites that remain
        // enabled underneath the selected six-score dispatch.  They do not
        // describe competing consumers once six_scores is selected.
        const bool six_softmax_triple=!gqa_six && six_scores && !six_sharded &&
            !six_packed && six_softmax;
        return scores && !oscar && !capture && !numeric_splitk &&
            (inherited || six_softmax_triple);
    }
};
}
