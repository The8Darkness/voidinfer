#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

namespace ninfer::exl3 {

enum class Exl3HeldoutMediaKind : std::uint8_t { image,video };
enum class Exl3HeldoutInputKind : std::uint8_t { deterministic_ppm,external_artifact };
enum class Exl3HeldoutGate : std::uint32_t {
    preprocessing=1U<<0,encoder_boundaries=1U<<1,l2_state=1U<<2,
    continuation=1U<<3,semantic_review=1U<<4
};
constexpr std::uint32_t operator|(Exl3HeldoutGate left,Exl3HeldoutGate right) noexcept {
    return static_cast<std::uint32_t>(left)|static_cast<std::uint32_t>(right);
}

struct Exl3HeldoutMediaFixture {
    std::string_view id;
    Exl3HeldoutMediaKind media_kind;
    Exl3HeldoutInputKind input_kind;
    std::string_view input_locator;
    std::string_view prompt;
    std::uint32_t required_gates=0;
    std::optional<std::string_view> input_sha256;
    std::optional<std::string_view> independent_embedding_sha256;
    std::optional<std::string_view> expected_semantic_answer;
};

struct Exl3HeldoutEvaluationCriteria {
    bool exact_preprocessing_identity=true;
    double maximum_boundary_relative_rmse=0.005;
    double minimum_boundary_cosine=0.9999;
    bool require_finite_boundaries=true;
    bool require_exact_l2_state_and_taps=true;
    bool require_same_input_frozen_continuation=true;
    bool require_independent_semantic_review=true;
    // No pass is possible until the independent artifacts and review exist.
    bool references_resolved=false;
};
inline constexpr Exl3HeldoutEvaluationCriteria kExl3HeldoutEvaluationCriteria{};

inline constexpr std::uint32_t kExl3HeldoutRequiredGates=
    static_cast<std::uint32_t>(Exl3HeldoutGate::preprocessing)|
    static_cast<std::uint32_t>(Exl3HeldoutGate::encoder_boundaries)|
    static_cast<std::uint32_t>(Exl3HeldoutGate::l2_state)|
    static_cast<std::uint32_t>(Exl3HeldoutGate::continuation)|
    static_cast<std::uint32_t>(Exl3HeldoutGate::semantic_review);

inline constexpr std::array<Exl3HeldoutMediaFixture,4> kExl3HeldoutMediaFixtures{{
    {"image-gradient-landscape",Exl3HeldoutMediaKind::image,Exl3HeldoutInputKind::deterministic_ppm,
        "gradient-128x64-phase-53","Describe the spatial pattern without guessing text.",
        kExl3HeldoutRequiredGates,std::nullopt,std::nullopt,std::nullopt},
    {"image-block-portrait",Exl3HeldoutMediaKind::image,Exl3HeldoutInputKind::deterministic_ppm,
        "block-64x128-rgb-31-127-223","Describe the dominant regions and their arrangement.",
        kExl3HeldoutRequiredGates,std::nullopt,std::nullopt,std::nullopt},
    {"video-two-segment-motion",Exl3HeldoutMediaKind::video,Exl3HeldoutInputKind::external_artifact,
        "fixtures/v6/two-segment-motion.mp4","Describe changes over time in order.",
        kExl3HeldoutRequiredGates,std::nullopt,std::nullopt,std::nullopt},
    {"video-odd-frame-padding",Exl3HeldoutMediaKind::video,Exl3HeldoutInputKind::external_artifact,
        "fixtures/v6/odd-frame-padding.mp4","Describe the sequence without inferring audio.",
        kExl3HeldoutRequiredGates,std::nullopt,std::nullopt,std::nullopt}
}};

inline void validate_exl3_heldout_fixture_catalogue(
    std::span<const Exl3HeldoutMediaFixture> fixtures=kExl3HeldoutMediaFixtures) {
    bool image=false,video=false;
    for(std::size_t i=0;i<fixtures.size();++i) {
        const auto& fixture=fixtures[i];
        if(fixture.id.empty() || fixture.input_locator.empty() || fixture.prompt.empty() ||
           fixture.required_gates!=kExl3HeldoutRequiredGates ||
           fixture.independent_embedding_sha256 || fixture.expected_semantic_answer)
            throw std::invalid_argument("V6 heldout fixture incomplete or fabricated");
        if(fixture.input_kind==Exl3HeldoutInputKind::external_artifact && fixture.input_sha256)
            throw std::invalid_argument("unresolved V6 artifact carries an unverified hash");
        for(std::size_t prior=0;prior<i;++prior)if(fixtures[prior].id==fixture.id)
            throw std::invalid_argument("duplicate V6 heldout fixture identity");
        image|=fixture.media_kind==Exl3HeldoutMediaKind::image;
        video|=fixture.media_kind==Exl3HeldoutMediaKind::video;
    }
    if(!image || !video)throw std::invalid_argument("V6 heldout modality coverage incomplete");
}

} // namespace ninfer::exl3
