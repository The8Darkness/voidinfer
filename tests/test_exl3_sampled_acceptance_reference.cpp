#include "exl3/sampled_acceptance_reference.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ninfer::exl3;
namespace {
void need(bool value,const char* message){if(!value)throw std::runtime_error(message);}
template<class F> bool refuses(F&& call){try{call();}catch(...){return true;}return false;}
}

int main() {
    const std::array<double,2> p_accept{0.6,0.4},q_accept{0.5,0.5};
    auto result=Exl3IndependentAcceptanceReference::evaluate(
        p_accept,q_accept,0,0.999,0.25);
    need(result.accepted && !result.correction_draw_used && result.token==0 &&
            result.acceptance_probability==1 && result.residual.empty(),
        "p-over-q cap-one acceptance changed");

    const std::array<double,2> p_reject{0.2,0.8},q_reject{0.5,0.5};
    result=Exl3IndependentAcceptanceReference::evaluate(
        p_reject,q_reject,0,0.5,0.75);
    need(!result.accepted && result.correction_draw_used && result.token==1 &&
            std::abs(result.acceptance_probability-0.4)<1e-12 &&
            result.residual[0]==0 && result.residual[1]==1,
        "first rejection residual was not normalized exactly to its support");

    const std::array<double,2> p_gap{0.25,0.75},q_gap{0,1};
    result=Exl3IndependentAcceptanceReference::evaluate(p_gap,q_gap,1,0.9,0.0);
    need(!result.accepted && result.token==0 && result.residual[0]==1 &&
            result.residual[1]==0,
        "target-only support gap was lost from correction distribution");

    const std::array<double,3> p_zero{0,0.5,0.5},q_zero{0,0.25,0.75};
    need(refuses([&]{(void)Exl3IndependentAcceptanceReference::evaluate(
            p_zero,q_zero,0,0.5,0.5);}),
        "zero-q proposal was treated as a possible draft draw");
    const std::array<double,2> identical{0.5,0.5};
    result=Exl3IndependentAcceptanceReference::evaluate(
        identical,identical,1,0.999999,0.5);
    need(result.accepted && !result.correction_draw_used,
        "identical distributions manufactured a residual correction");

    auto malformed=p_accept;malformed[0]=std::numeric_limits<double>::infinity();
    const std::array<double,2> not_normalized{0.4,0.4};
    need(refuses([&]{(void)Exl3IndependentAcceptanceReference::evaluate(
            malformed,q_accept,0,0.1,0.1);}) &&
            refuses([&]{(void)Exl3IndependentAcceptanceReference::evaluate(
            not_normalized,q_accept,0,0.1,0.1);}) &&
            refuses([&]{(void)Exl3IndependentAcceptanceReference::evaluate(
            p_accept,q_accept,0,1.0,0.1);}),
        "invalid probability or draw admitted");
    std::cout<<"exl3_sampled_acceptance_reference PASS\n";
}
