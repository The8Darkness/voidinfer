#include "runtime/engine/admission_policy.h"
#include "runtime/engine/scheduler.h"

#include <array>
#include <iostream>
#include <utility>

namespace {

struct SchedulerRequest {
    using SequenceHandle = std::uint64_t;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using ninfer::runtime::ActiveAdmissionSnapshot;
    using ninfer::runtime::AdmissionResources;
    using ninfer::runtime::BackfillClass;

    int failures = 0;
    const AdmissionResources capacity{
        .active_lanes     = 4,
        .main_kv_pages    = 160,
        .backend_kv_pages = 128,
    };
    const AdmissionResources head{
        .active_lanes     = 1,
        .main_kv_pages    = 64,
        .backend_kv_pages = 48,
    };
    std::array<ActiveAdmissionSnapshot, 2> incumbents{
        ActiveAdmissionSnapshot{
            .request_id            = 1,
            .resources             = {1, 64, 32},
            .remaining_work_quanta = 100,
        },
        ActiveAdmissionSnapshot{
            .request_id            = 2,
            .resources             = {1, 48, 64},
            .remaining_work_quanta = 20,
        },
    };

    const auto protection = ninfer::runtime::make_admission_protection(
        7, 10, head, std::span<const ActiveAdmissionSnapshot>(incumbents), capacity);
    failures += check(protection.donor_count == 1 && protection.donor_ids[0] == 2 &&
                          protection.temporal_credit == 20,
                      "release frontier did not select the earliest sufficient incumbent");
    failures += check(ninfer::runtime::protection_frontier_distance(protection, incumbents) == 20,
                      "frontier distance did not follow the frozen donor");

    const AdmissionResources persistent_candidate{1, 24, 40};
    failures += check(ninfer::runtime::persistent_backfill_is_safe(protection, incumbents,
                                                                   persistent_candidate, capacity),
                      "future resource surplus rejected a persistent-safe backfill");
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(
                          protection, incumbents, AdmissionResources{1, 40, 60}, capacity),
                      "persistent backfill borrowed protected future capacity");

    std::array<ActiveAdmissionSnapshot, 3> with_persistent{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id            = 3,
            .resources             = persistent_candidate,
            .remaining_work_quanta = 50,
            .backfill_epoch        = 7,
            .backfill_class        = BackfillClass::Persistent,
        },
    };
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(
                          protection, with_persistent, AdmissionResources{1, 9, 9}, capacity),
                      "persistent ledger failed to accumulate earlier backfills");

    std::array<ActiveAdmissionSnapshot, 2> after_donor{
        incumbents[0],
        ActiveAdmissionSnapshot{
            .request_id            = 4,
            .resources             = {1, 32, 64},
            .remaining_work_quanta = 8,
            .backfill_epoch        = 7,
            .backfill_class        = BackfillClass::Temporal,
        },
    };
    failures += check(ninfer::runtime::protection_frontier_distance(protection, after_donor) == 0,
                      "later temporal work changed the frozen frontier");
    failures += check(
        ninfer::runtime::protected_head_safe_without_temporal(protection, after_donor, capacity),
        "released frontier did not mature behind a temporal borrower");

    failures += check(
        !ninfer::runtime::admission_resources_fit(AdmissionResources{1, 161, 1}, capacity) &&
            !ninfer::runtime::admission_resources_fit(AdmissionResources{1, 1, 129}, capacity),
        "independent KV pools were incorrectly treated as interchangeable capacity");

    using Scheduler = ninfer::runtime::Scheduler<SchedulerRequest>;
    Scheduler scheduler;
    scheduler.observe_fifo_head(10);
    failures += check(scheduler.protect_blocked_head(10, head, incumbents, capacity),
                      "blocked FIFO head did not open a protection epoch");

    const AdmissionResources persistent_safe_candidate{1, 24, 24};
    auto persistent =
        scheduler.qualify_backfill(11, persistent_safe_candidate, 50, incumbents, capacity);
    failures += check(persistent && persistent->backfill_class() == BackfillClass::Persistent &&
                          scheduler.validate_grant(*persistent),
                      "persistent-safe candidate did not receive a valid Scheduler grant");
    if (!persistent) { return 1; }
    const std::uint64_t protection_epoch = persistent->protection_epoch();
    scheduler.commit_admission(std::move(*persistent));

    std::array<ActiveAdmissionSnapshot, 3> active_with_persistent{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id            = 11,
            .resources             = persistent_safe_candidate,
            .remaining_work_quanta = 50,
            .backfill_epoch        = protection_epoch,
            .backfill_class        = BackfillClass::Persistent,
        },
    };
    const AdmissionResources temporal_candidate{1, 16, 8};
    auto temporal =
        scheduler.qualify_backfill(12, temporal_candidate, 8, active_with_persistent, capacity);
    failures += check(temporal && temporal->backfill_class() == BackfillClass::Temporal &&
                          scheduler.validate_grant(*temporal),
                      "bounded temporal candidate did not receive a valid Scheduler grant");
    if (!temporal) { return 1; }
    scheduler.commit_admission(std::move(*temporal));
    failures += check(
        !scheduler.qualify_backfill(13, temporal_candidate, 13, active_with_persistent, capacity),
        "committed temporal work did not consume protected credit");

    std::array<ActiveAdmissionSnapshot, 3> matured{
        incumbents[0],
        active_with_persistent[2],
        ActiveAdmissionSnapshot{
            .request_id            = 12,
            .resources             = temporal_candidate,
            .remaining_work_quanta = 8,
            .backfill_epoch        = protection_epoch,
            .backfill_class        = BackfillClass::Temporal,
        },
    };
    failures += check(!scheduler.protect_blocked_head(10, head, matured, capacity),
                      "matured protected opportunity did not enter drain");
    scheduler.on_waiting_removed(10);
    scheduler.observe_fifo_head(14);
    auto head_grant = scheduler.grant_head(14, 1);
    failures += check(scheduler.validate_grant(head_grant),
                      "observed FIFO head did not receive a valid admission grant");
    scheduler.commit_admission(std::move(head_grant));

    using BoundaryAction = Scheduler::BoundaryAction;
    failures +=
        check(scheduler.choose_boundary(true, false, false) == BoundaryAction::AttemptAdmission &&
                  scheduler.choose_boundary(false, true, false) == BoundaryAction::Decode,
              "idle boundary ordering did not preserve admission and decode priority");
    scheduler.set_prefill_lane(0);
    failures += check(scheduler.choose_boundary(true, true, false) == BoundaryAction::Decode &&
                          scheduler.choose_boundary(true, true, true) == BoundaryAction::Prefill,
                      "prefill/decode alternation did not follow the completed GPU unit");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
