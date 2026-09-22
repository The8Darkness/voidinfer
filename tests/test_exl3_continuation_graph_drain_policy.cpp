#include "exl3/continuation_graph_drain_policy.h"
#include <array>
#include <iostream>
#include <stdexcept>

using Policy=ninfer::exl3::Exl3ContinuationGraphDrainPolicy;
static void need(bool value) {if(!value)throw std::runtime_error("graph drain policy assertion");}

int main() {try {
    auto owner=std::make_shared<int>(1);
    const std::array<Policy::Pending,3> one{{{false,0},{false,0},{true,41}}};
    need(Policy::select(false,owner,41,one)==Policy::Mode::device);
    need(Policy::select(true,{},41,one)==Policy::Mode::device);
    need(Policy::select(true,owner,0,one)==Policy::Mode::device);
    need(Policy::select(true,owner,41,one)==Policy::Mode::owned_stream);

    // Two private graph entries on the same retained stream need one fence.
    const std::array<Policy::Pending,3> private_pair{{
        {true,41},{true,41},{false,0}}};
    need(Policy::select(true,owner,41,private_pair)==Policy::Mode::owned_stream);

    // Cross-stream transfer, missing final stream identity, and a default-
    // stream control suffix all preserve the conservative device drain.
    auto cross=private_pair;cross[1].stream=43;
    need(Policy::select(true,owner,41,cross)==Policy::Mode::device);
    auto missing=private_pair;missing[1].stream=0;
    need(Policy::select(true,owner,41,missing)==Policy::Mode::device);
    const std::array<Policy::Pending,1> control{{{true,0}}};
    need(Policy::select(true,owner,41,control)==Policy::Mode::device);

    // An alias of the retained token remains the same strong lifetime.
    std::shared_ptr<const void> alias(owner,owner.get());
    need(Policy::select(true,alias,41,one)==Policy::Mode::owned_stream);
    std::cout<<"CONTINUATION_GRAPH_DRAIN_POLICY_COMPLETE device_execution=0\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}}
