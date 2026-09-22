#include "exl3/vision_heldout_fixtures.h"
#include <iostream>

using namespace ninfer::exl3;
template<class F> bool refuses(F&& f){try{f();return false;}catch(const std::exception&){return true;}}
void need(bool value,const char* label){if(!value)throw std::runtime_error(label);}
int main(){try{
    validate_exl3_heldout_fixture_catalogue();
    std::size_t images=0,videos=0,unresolved=0;
    for(const auto& fixture:kExl3HeldoutMediaFixtures) {
        images+=fixture.media_kind==Exl3HeldoutMediaKind::image;
        videos+=fixture.media_kind==Exl3HeldoutMediaKind::video;
        unresolved+=!fixture.independent_embedding_sha256 && !fixture.expected_semantic_answer;
    }
    need(images==2 && videos==2 && unresolved==kExl3HeldoutMediaFixtures.size(),
        "heldout fixture breadth or unresolved authority");
    need(kExl3HeldoutEvaluationCriteria.exact_preprocessing_identity &&
            kExl3HeldoutEvaluationCriteria.maximum_boundary_relative_rmse==0.005 &&
            kExl3HeldoutEvaluationCriteria.minimum_boundary_cosine==0.9999 &&
            kExl3HeldoutEvaluationCriteria.require_finite_boundaries &&
            kExl3HeldoutEvaluationCriteria.require_exact_l2_state_and_taps &&
            kExl3HeldoutEvaluationCriteria.require_same_input_frozen_continuation &&
            kExl3HeldoutEvaluationCriteria.require_independent_semantic_review &&
            !kExl3HeldoutEvaluationCriteria.references_resolved,
        "heldout evaluation criteria promoted unresolved evidence");
    auto fabricated=kExl3HeldoutMediaFixtures;
    fabricated[0].expected_semantic_answer="fabricated answer";
    need(refuses([&]{validate_exl3_heldout_fixture_catalogue(fabricated);}),
        "fabricated semantic answer accepted");
    auto missing_gate=kExl3HeldoutMediaFixtures;
    missing_gate[1].required_gates&=~static_cast<std::uint32_t>(Exl3HeldoutGate::l2_state);
    need(refuses([&]{validate_exl3_heldout_fixture_catalogue(missing_gate);}),
        "incomplete heldout gate family accepted");
    auto unresolved_hash=kExl3HeldoutMediaFixtures;
    unresolved_hash[2].input_sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    need(refuses([&]{validate_exl3_heldout_fixture_catalogue(unresolved_hash);}),
        "unverified external video hash accepted");
    std::cout<<"PASS heldout V6 fixture definitions\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
