#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace ninfer::exl3 {

enum class Exl3SampledAcceptanceMethod : std::uint8_t {
    unspecified,
    independent_rejection_with_residual
};

enum class Exl3DraftDistributionConditioning : std::uint8_t {
    parallel_masked_target_taps,
    autoregressive_predecessor
};

// Describes what the real proposal generator can publish. Internal selector
// scores are not an authority: exact sampled rejection requires a complete,
// normalized q for every vocabulary item under each actual predecessor.
struct Exl3DraftDistributionAvailability {
    bool greedy_tokens_available=false;
    bool internal_selector_scores=false;
    bool normalized_q_available=false;
    bool complete_vocabulary_support=false;
    Exl3DraftDistributionConditioning conditioning=
        Exl3DraftDistributionConditioning::parallel_masked_target_taps;

    [[nodiscard]] static Exl3DraftDistributionAvailability current_dflash2() noexcept {
        return {true,true,false,false,
            Exl3DraftDistributionConditioning::parallel_masked_target_taps};
    }

    [[nodiscard]] bool supports(Exl3SampledAcceptanceMethod method) const noexcept {
        return method==Exl3SampledAcceptanceMethod::independent_rejection_with_residual &&
            normalized_q_available && complete_vocabulary_support &&
            conditioning==Exl3DraftDistributionConditioning::autoregressive_predecessor;
    }

    [[nodiscard]] bool permits_explicit_greedy_fallback() const noexcept {
        return greedy_tokens_available;
    }
};

// One produced distribution authority. This is deliberately separate from the
// capability above: a capable implementation must still bind every q row to
// immutable request/root/position ownership before exact acceptance may use it.
struct Exl3DraftDistributionAuthority {
    Exl3SampledAcceptanceMethod method=Exl3SampledAcceptanceMethod::unspecified;
    Exl3DraftDistributionConditioning conditioning=
        Exl3DraftDistributionConditioning::parallel_masked_target_taps;
    std::shared_ptr<const void> owner;
    const float* probabilities=nullptr;
    std::uint32_t rows=0,vocabulary=0,row_stride=0;
    std::uint64_t request_generation=0,root_revision=0;
    int first_position=0;
    bool normalized=false,complete_vocabulary_support=false,ready=false;

    void require(Exl3SampledAcceptanceMethod expected_method,
        std::uint32_t expected_rows,std::uint32_t expected_vocabulary,
        std::uint64_t expected_request_generation,std::uint64_t expected_root_revision,
        int expected_first_position) const {
        if(expected_method!=Exl3SampledAcceptanceMethod::independent_rejection_with_residual ||
           method!=expected_method)
            throw std::invalid_argument("unsupported sampled acceptance method");
        if(conditioning!=Exl3DraftDistributionConditioning::autoregressive_predecessor)
            throw std::invalid_argument("draft q has incompatible predecessor conditioning");
        if(!owner || !probabilities || !ready || !normalized ||
           !complete_vocabulary_support || !expected_rows || !expected_vocabulary ||
           rows!=expected_rows || vocabulary!=expected_vocabulary ||
           row_stride<expected_vocabulary || request_generation!=expected_request_generation ||
           root_revision!=expected_root_revision || first_position!=expected_first_position)
            throw std::invalid_argument("missing or stale draft distribution authority");
    }
};

} // namespace ninfer::exl3
