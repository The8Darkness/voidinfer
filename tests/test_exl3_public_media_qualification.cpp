#include "exl3/public_media_qualification.h"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace ninfer::exl3;

namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}

Exl3PublicMediaEvidence complete_evidence() {
    Exl3PublicMediaEvidence evidence;
    for(std::size_t i=0;i<evidence.gates.size();++i)
        evidence.gates[i]=Exl3PublicMediaGateState::passed;
    evidence.independent_encoder_identity="independent-v6-reference-receipt";
    evidence.full_media_path_identity="exact-l2-draft-lifecycle-receipt";
    evidence.heldout_review_identity="heldout-image-video-review-receipt";
    evidence.client_path_identity="loopback-stream-cache-cancel-receipt";
    return evidence;
}
}

int main() {
    const auto current=Exl3PublicMediaQualification::evaluate_current(
        Exl3PublicMediaModality::image);
    need(current.disposition==Exl3PublicMediaDisposition::failed_gate &&
        current.gate==Exl3PublicMediaGate::encoder_boundaries,
        "known failed V6 boundary did not keep public media closed");

    auto evidence=complete_evidence();
    evidence.set(Exl3PublicMediaGate::independent_encoder,
        Exl3PublicMediaGateState::missing);
    auto decision=Exl3PublicMediaQualification::evaluate(
        Exl3PublicMediaModality::image,evidence,{true,false});
    need(decision.disposition==Exl3PublicMediaDisposition::missing_gate &&
        decision.gate==Exl3PublicMediaGate::independent_encoder,
        "missing independent encoder evidence admitted");

    evidence=complete_evidence();
    evidence.set(Exl3PublicMediaGate::lifecycle_and_cancellation,
        Exl3PublicMediaGateState::failed);
    decision=Exl3PublicMediaQualification::evaluate(
        Exl3PublicMediaModality::video,evidence,{true,false});
    need(decision.disposition==Exl3PublicMediaDisposition::failed_gate &&
        decision.gate==Exl3PublicMediaGate::lifecycle_and_cancellation,
        "failed lifecycle evidence admitted");

    evidence=complete_evidence();
    decision=Exl3PublicMediaQualification::evaluate(
        Exl3PublicMediaModality::unsupported,evidence,{true,false});
    need(decision.disposition==Exl3PublicMediaDisposition::unsupported_modality,
        "unsupported modality admitted");
    decision=Exl3PublicMediaQualification::evaluate(
        Exl3PublicMediaModality::image,evidence,{false,false});
    need(decision.disposition==Exl3PublicMediaDisposition::explicitly_disabled,
        "implicit media enablement admitted");

    auto legacy=evidence;
    legacy.independent_encoder_identity.clear();
    legacy.full_media_path_identity.clear();
    legacy.heldout_review_identity.clear();
    legacy.client_path_identity="old-research-receipt";
    decision=Exl3PublicMediaQualification::evaluate(
        Exl3PublicMediaModality::image,legacy,{true,false});
    need(decision.disposition==Exl3PublicMediaDisposition::incomplete_evidence_identity,
        "one legacy receipt substituted for independent public gates");

    for(const auto modality:{Exl3PublicMediaModality::image,
        Exl3PublicMediaModality::video,Exl3PublicMediaModality::image_and_video}) {
        decision=Exl3PublicMediaQualification::evaluate(modality,evidence,{true,true});
        need(decision.disposition==Exl3PublicMediaDisposition::qualification_forbidden,
            "campaign source removed the public enablement latch");
        decision=Exl3PublicMediaQualification::evaluate(modality,evidence,{true,false});
        need(decision.media_allowed(),"complete explicit future evidence remained ineligible");
    }

    evidence.set(Exl3PublicMediaGate::encoder_boundaries,
        Exl3PublicMediaGateState::failed);
    decision=Exl3PublicMediaQualification::evaluate(
        Exl3PublicMediaModality::text,evidence,{true,true});
    need(decision.text_allowed() && !decision.media_allowed(),
        "unqualified media gates changed greedy text eligibility");

    bool research_refused=false;
    try{Exl3PublicMediaQualification::require_research_profile_disabled("1");}
    catch(const std::invalid_argument& error){
        research_refused=std::string(error.what()).find("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH")!=std::string::npos;
    }
    need(research_refused,"research option bypassed current public media gate");
    Exl3PublicMediaQualification::require_research_profile_disabled("0");

    std::cout<<"public_media_qualification_source PREPARED_NOT_RUN\n";
}
