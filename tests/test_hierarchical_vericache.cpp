#include "runtime/engine/hierarchical_vericache.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::runtime;

void test_adaptive_windows() {
    HierarchicalVeriCacheOptions options;
    options.enabled = true;
    options.l0_to_l1_horizon = 8;
    options.l0_to_l1_min_horizon = 4;
    options.l0_to_l1_max_horizon = 16;
    options.l1_to_l2_horizon = 512;
    AdaptiveHierarchicalVeriCacheController controller(options);

    assert(controller.l0_to_l1_horizon() == 8);
    controller.observe_l0_to_l1(8, 0, true, true);
    assert(controller.l0_to_l1_horizon() == 6);
    for (int i = 0; i < 4; ++i) { controller.observe_l0_to_l1(6, 6, false, false); }
    assert(controller.l0_to_l1_horizon() == 7);
    assert(controller.next_l0_to_l1_boundary(100) == 107);
    assert(controller.next_l1_to_l2_boundary(100) == 612);

    RuntimeStats stats;
    controller.populate(stats);
    assert(stats.hierarchical_vericache_enabled);
    assert(stats.vericache_l0_l1_checks == 5);
    assert(stats.vericache_l0_l1_disagreements == 1);
}

void test_protection_ledger() {
    HierarchicalVeriCacheOptions options;
    options.enabled = true;
    options.protected_recent_tokens = 8;
    options.protected_sink_tokens = 2;
    HierarchicalVeriCacheLedger ledger(options);

    const std::vector<HierarchicalVeriCacheRange> explicit_ranges{
        {20, 24, HierarchicalVeriCacheSensitivity::ToolSchema},
        {30, 32, HierarchicalVeriCacheSensitivity::GdnProtected},
    };
    assert(ledger.classify(0, 1, 40) == HierarchicalVeriCacheSensitivity::AttentionSink);
    assert(ledger.classify(33, 40, 40) == HierarchicalVeriCacheSensitivity::Recent);
    assert(ledger.classify(20, 21, 40, explicit_ranges) ==
           HierarchicalVeriCacheSensitivity::ToolSchema);
    assert(ledger.classify(30, 31, 40, explicit_ranges) ==
           HierarchicalVeriCacheSensitivity::GdnProtected);

    ledger.append({0, 2, HierarchicalVeriCacheTier::L0Vram,
                   HierarchicalVeriCacheEncoding::Nvfp4,
                   HierarchicalVeriCacheSensitivity::AttentionSink, 64});
    ledger.append({2, 20, HierarchicalVeriCacheTier::L0Vram,
                   HierarchicalVeriCacheEncoding::Packed2Bit,
                   HierarchicalVeriCacheSensitivity::Normal, 128});
    assert(!ledger.can_reencode(0, HierarchicalVeriCacheEncoding::Packed3Bit));
    assert(ledger.can_reencode(1, HierarchicalVeriCacheEncoding::Packed3Bit));
    ledger.reencode(1, HierarchicalVeriCacheEncoding::Packed3Bit, 96);
    assert(ledger.bytes(HierarchicalVeriCacheTier::L0Vram) == 160);
    assert(ledger.protected_bytes() == 64);
}

void test_nested_transactions() {
    NestedHierarchicalVeriCacheTransaction transaction;
    const auto parent = transaction.begin(
        HierarchicalVeriCacheTransactionKind::L0Speculation, {100, 100}, {164, 164});
    const auto child = transaction.begin(
        HierarchicalVeriCacheTransactionKind::GdnRecurrentState, {100, 100}, {164, 164});
    assert(transaction.active());
    assert(transaction.depth() == 2);
    assert(transaction.commit(child, {132, 132}));
    assert(transaction.commit(parent, {132, 132}));
    assert(!transaction.active());
    assert(transaction.commits() == 2);
    assert(transaction.partial_rollbacks() == 2);
    assert(transaction.rolled_back_tokens() == 128);
    assert(transaction.max_depth() == 2);

    const auto rollback_parent = transaction.begin(
        HierarchicalVeriCacheTransactionKind::L1ToL2Verification, {200, 200}, {240, 240});
    const auto rollback_child = transaction.begin(
        HierarchicalVeriCacheTransactionKind::GdnRecurrentState, {200, 200}, {240, 240});
    assert(transaction.rollback(rollback_child));
    assert(transaction.rollback(rollback_parent));
    assert(transaction.rollbacks() == 2);
}

void test_cold_manifest() {
    HierarchicalVeriCacheColdManifest manifest("vericache.l3");
    manifest.append({0, 256, 0, 1024, HierarchicalVeriCacheEncoding::Fp16});
    manifest.append({256, 512, 1024, 768, HierarchicalVeriCacheEncoding::Fp8});
    assert(manifest.records().size() == 2);
    assert(manifest.bytes() == 1792);
    assert(!manifest.allowed_in_frequent_verifier());
}

} // namespace

int main() {
    test_adaptive_windows();
    test_protection_ledger();
    test_nested_transactions();
    test_cold_manifest();
    std::cout << "ok\n";
    return 0;
}
