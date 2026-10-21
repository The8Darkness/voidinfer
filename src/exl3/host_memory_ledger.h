#pragma once
// Process-wide byte counts of the large pinned host holders, printed with
// host memory refusals and evictions (VOIDINFER_EXL3_VERICACHE_STATS): the
// commit charge they hold is invisible in the working set.
#include "exl3/host_memory_available.h"

#include <atomic>
#include <cstdint>
#include <cstdio>

namespace voidinfer::exl3 {

struct Exl3HostMemoryLedger {
    std::atomic<std::int64_t> plane_bytes{0};       // host-backed growable regions, mapped
    std::atomic<std::int64_t> plane_pool_bytes{0};  // pooled host chunks
    std::atomic<std::int64_t> code_slab_bytes{0};   // parked L0 code slabs
    std::atomic<std::int64_t> code_free_bytes{0};   // free blocks in those slabs
    std::atomic<std::int64_t> suspended_bytes{0};   // per-context L0 code suspension stages
    static Exl3HostMemoryLedger& get() {
        static Exl3HostMemoryLedger value;
        return value;
    }
    static void print(const char* where) {
        auto& l = get();
        const auto mib = [](const std::atomic<std::int64_t>& v) { return static_cast<long long>(v.load() >> 20); };
        std::fprintf(stderr,
            "HOST_LEDGER %s available_mib=%llu planes_mib=%lld plane_pool_mib=%lld code_slabs_mib=%lld "
            "code_free_mib=%lld suspended_mib=%lld\n",
            where, static_cast<unsigned long long>(exl3_available_host_bytes().value_or(0) >> 20),
            mib(l.plane_bytes), mib(l.plane_pool_bytes), mib(l.code_slab_bytes), mib(l.code_free_bytes),
            mib(l.suspended_bytes));
    }
};

}  // namespace voidinfer::exl3
