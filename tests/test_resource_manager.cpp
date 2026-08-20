#include "runtime/engine/resource_manager.h"

#include <array>
#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using ninfer::runtime::AdmissionResources;
    using ninfer::runtime::LaneId;
    using ninfer::runtime::LogicalLaneState;
    using ninfer::runtime::ResidentResourceDescriptor;
    using ninfer::runtime::ResourceCandidateDescriptor;
    using ninfer::runtime::ResourceLedger;

    int failures = 0;
    const AdmissionResources selection_capacity{2, 100, 100};
    const AdmissionResources used{0, 60, 60};
    const std::array<ResidentResourceDescriptor, 3> residents{
        ResidentResourceDescriptor{LaneId{0}, {0, 20, 20}},
        ResidentResourceDescriptor{LaneId{1}, {0, 20, 20}},
        ResidentResourceDescriptor{LaneId{2}, {0, 20, 20}},
    };

    const std::array<ResourceCandidateDescriptor, 2> no_eviction_priority{
        ResourceCandidateDescriptor{LaneId{0}, {1, 50, 50}, {0, 20, 20}, 5},
        ResourceCandidateDescriptor{LaneId{1}, {1, 90, 90}, {0, 20, 20}, 100},
    };
    const auto first = ninfer::runtime::select_resource_candidate(no_eviction_priority, residents,
                                                                  used, selection_capacity);
    failures += check(first.found && first.candidate_index == 0 && first.eviction_count == 0,
                      "a higher-reuse eviction candidate displaced a no-eviction candidate");

    const std::array<ResourceCandidateDescriptor, 3> reuse_and_tie{
        ResourceCandidateDescriptor{LaneId{2}, {1, 55, 55}, {0, 20, 20}, 8},
        ResourceCandidateDescriptor{LaneId{1}, {1, 55, 55}, {0, 20, 20}, 12},
        ResourceCandidateDescriptor{LaneId{0}, {1, 55, 55}, {0, 20, 20}, 12},
    };
    const auto best = ninfer::runtime::select_resource_candidate(reuse_and_tie, residents, used,
                                                                 selection_capacity);
    failures += check(best.found && best.candidate_index == 2 && best.eviction_count == 0,
                      "candidate selection did not maximize reuse with a lane-order tie break");

    const std::array<ResourceCandidateDescriptor, 1> needs_two_evictions{
        ResourceCandidateDescriptor{LaneId{0}, {1, 90, 90}, {0, 20, 20}, 20},
    };
    const auto eviction = ninfer::runtime::select_resource_candidate(needs_two_evictions, residents,
                                                                     used, selection_capacity);
    failures +=
        check(eviction.found && eviction.eviction_count == 2 &&
                  eviction.evictions[0] == LaneId{1} && eviction.evictions[1] == LaneId{2},
              "candidate selection did not choose the shortest lane-ordered eviction prefix");

    ResourceLedger ledger({2, 100, 100}, 2);
    const AdmissionResources active0{1, 60, 40};
    const AdmissionResources resident0{0, 40, 20};
    const AdmissionResources active1{1, 60, 70};
    failures += check(ledger.adopt_active(LaneId{0}, active0) &&
                          ledger.lane(LaneId{0}).state == LogicalLaneState::Active,
                      "Free to Active transition failed");
    failures += check(ledger.retain(LaneId{0}, active0, resident0) &&
                          ledger.lane(LaneId{0}).state == LogicalLaneState::Resident,
                      "Active to Resident transition failed");
    failures += check(ledger.adopt_active(LaneId{1}, active1) &&
                          ledger.used() == AdmissionResources{1, 100, 90},
                      "resident and active entitlements were not combined dimension-wise");
    failures += check(!ledger.release_active(LaneId{1}, AdmissionResources{1, 59, 70}),
                      "an incorrect active release acknowledgement was accepted");
    failures += check(ledger.release_active(LaneId{1}, active1) &&
                          ledger.release_resident(LaneId{0}, resident0) &&
                          ledger.used() == AdmissionResources{},
                      "Active/Resident to Free transitions did not close the ledger");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
