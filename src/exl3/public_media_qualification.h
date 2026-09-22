#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::exl3 {

enum class Exl3PublicMediaModality : std::uint8_t {
    text,
    image,
    video,
    image_and_video,
    unsupported
};

enum class Exl3PublicMediaGateState : std::uint8_t { missing,failed,passed };

enum class Exl3PublicMediaGate : std::uint8_t {
    independent_encoder,
    preprocessing,
    encoder_boundaries,
    target_l2_and_taps,
    draft_conditioning,
    lifecycle_and_cancellation,
    heldout_semantics,
    client_path,
    count
};

struct Exl3PublicMediaEvidence {
    std::uint32_t schema_revision=1;
    std::array<Exl3PublicMediaGateState,
        static_cast<std::size_t>(Exl3PublicMediaGate::count)> gates{};
    // These are separate authorities. One old research receipt cannot stand in
    // for an independent encoder, the complete media path, held-out review and
    // the public client boundary.
    std::string independent_encoder_identity;
    std::string full_media_path_identity;
    std::string heldout_review_identity;
    std::string client_path_identity;

    [[nodiscard]] Exl3PublicMediaGateState gate(Exl3PublicMediaGate value) const noexcept {
        return gates[static_cast<std::size_t>(value)];
    }
    void set(Exl3PublicMediaGate value,Exl3PublicMediaGateState state) noexcept {
        gates[static_cast<std::size_t>(value)]=state;
    }
};

struct Exl3PublicMediaPolicy {
    bool explicit_enablement=false;
    // The implementation campaign cannot remove this latch. A later,
    // separately authorized qualification must construct a new policy only
    // after binding every independent receipt above.
    bool public_enablement_forbidden=true;
};

enum class Exl3PublicMediaDisposition : std::uint8_t {
    text_unaffected,
    explicitly_disabled,
    unsupported_modality,
    failed_gate,
    missing_gate,
    incomplete_evidence_identity,
    qualification_forbidden,
    eligible
};

struct Exl3PublicMediaDecision {
    Exl3PublicMediaDisposition disposition=Exl3PublicMediaDisposition::explicitly_disabled;
    Exl3PublicMediaGate gate=Exl3PublicMediaGate::count;

    [[nodiscard]] bool text_allowed() const noexcept {
        return disposition==Exl3PublicMediaDisposition::text_unaffected;
    }
    [[nodiscard]] bool media_allowed() const noexcept {
        return disposition==Exl3PublicMediaDisposition::eligible;
    }
};

class Exl3PublicMediaQualification {
    static bool supported(Exl3PublicMediaModality value) noexcept {
        return value==Exl3PublicMediaModality::image ||
            value==Exl3PublicMediaModality::video ||
            value==Exl3PublicMediaModality::image_and_video;
    }
public:
    [[nodiscard]] static Exl3PublicMediaDecision evaluate(
        Exl3PublicMediaModality modality,const Exl3PublicMediaEvidence& evidence,
        Exl3PublicMediaPolicy policy) noexcept {
        if(modality==Exl3PublicMediaModality::text)
            return {Exl3PublicMediaDisposition::text_unaffected};
        if(!supported(modality))
            return {Exl3PublicMediaDisposition::unsupported_modality};
        if(!policy.explicit_enablement)
            return {Exl3PublicMediaDisposition::explicitly_disabled};
        for(std::size_t i=0;i<evidence.gates.size();++i)
            if(evidence.gates[i]==Exl3PublicMediaGateState::failed)
                return {Exl3PublicMediaDisposition::failed_gate,
                    static_cast<Exl3PublicMediaGate>(i)};
        for(std::size_t i=0;i<evidence.gates.size();++i)
            if(evidence.gates[i]!=Exl3PublicMediaGateState::passed)
                return {Exl3PublicMediaDisposition::missing_gate,
                    static_cast<Exl3PublicMediaGate>(i)};
        if(evidence.schema_revision!=1 || evidence.independent_encoder_identity.empty() ||
           evidence.full_media_path_identity.empty() || evidence.heldout_review_identity.empty() ||
           evidence.client_path_identity.empty())
            return {Exl3PublicMediaDisposition::incomplete_evidence_identity};
        if(policy.public_enablement_forbidden)
            return {Exl3PublicMediaDisposition::qualification_forbidden};
        return {Exl3PublicMediaDisposition::eligible};
    }

    [[nodiscard]] static Exl3PublicMediaEvidence current_evidence() noexcept {
        Exl3PublicMediaEvidence evidence;
        // T281 retains a known failed late encoder boundary. All other public
        // receipts are absent, including independent held-out/client evidence.
        evidence.set(Exl3PublicMediaGate::encoder_boundaries,
            Exl3PublicMediaGateState::failed);
        return evidence;
    }

    [[nodiscard]] static Exl3PublicMediaDecision evaluate_current(
        Exl3PublicMediaModality modality) noexcept {
        return evaluate(modality,current_evidence(),{true,true});
    }

    [[nodiscard]] static std::string_view disposition_name(
        Exl3PublicMediaDisposition value) noexcept {
        switch(value) {
        case Exl3PublicMediaDisposition::text_unaffected:return "text unaffected";
        case Exl3PublicMediaDisposition::explicitly_disabled:return "explicitly disabled";
        case Exl3PublicMediaDisposition::unsupported_modality:return "unsupported modality";
        case Exl3PublicMediaDisposition::failed_gate:return "failed qualification gate";
        case Exl3PublicMediaDisposition::missing_gate:return "missing qualification gate";
        case Exl3PublicMediaDisposition::incomplete_evidence_identity:return "incomplete evidence identity";
        case Exl3PublicMediaDisposition::qualification_forbidden:return "qualification forbidden";
        case Exl3PublicMediaDisposition::eligible:return "eligible";
        }
        return "unknown";
    }

    static void require_current_request(Exl3PublicMediaModality modality) {
        const auto decision=evaluate_current(modality);
        if(decision.text_allowed() || decision.media_allowed())return;
        throw std::invalid_argument(std::string("EXL3 public media qualification gate: ")+
            std::string(disposition_name(decision.disposition)));
    }

    static void require_research_profile_disabled(std::string_view value) {
        if(value.empty() || value=="0")return;
        if(value!="1")
            throw std::invalid_argument("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH must be0 or1");
        const auto decision=evaluate_current(Exl3PublicMediaModality::image);
        throw std::invalid_argument(std::string("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH cannot enable public media: ")+
            std::string(disposition_name(decision.disposition)));
    }
};

} // namespace ninfer::exl3
