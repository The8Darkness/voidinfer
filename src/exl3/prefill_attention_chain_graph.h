#pragma once

#include "exl3/bounded_graph_entry.h"
#include "core/decode_graph.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace ninfer::exl3 {

// Retains the exact selected score/softmax/value launch chain for one of the
// three history-bearing 1024-row chunks of a 4096-token prefill.  The body is
// numerical work only: KV append, cursor publication, and request retirement
// remain eager and transaction-owned.
class Exl3PrefillAttentionChainGraph {
public:
    static constexpr int slot_count=4;
    struct Request {
        Exl3GraphCompatibilityFingerprint fingerprint;
        std::span<const Exl3GraphBoundResource> resources;
        Exl3GraphCaptureExtent reserved_peak;
        std::uint64_t generation=0;
        std::uint64_t reservation_configuration=0;
        cudaStream_t stream=nullptr;
        int slot=-1;
        bool eligible=false;
    };
    struct Snapshot {
        std::uint64_t capture_setups=0,captures=0,replays=0,eager_fallbacks=0;
        std::uint64_t binding_mismatches=0,nested_captures=0,quarantines=0;
        std::uint64_t ready_slots=0,pending_slots=0;
    };
    using Body=std::function<void(cudaStream_t)>;

    Exl3PrefillAttentionChainGraph();
    ~Exl3PrefillAttentionChainGraph();
    Exl3PrefillAttentionChainGraph(const Exl3PrefillAttentionChainGraph&)=delete;
    Exl3PrefillAttentionChainGraph& operator=(const Exl3PrefillAttentionChainGraph&)=delete;

    // True means capture/replay submitted the body. False means the identical
    // eager body was submitted on request.stream.
    bool execute(const Request& request,const Body& body);
    bool complete_after_drain(cudaStream_t stream,int error=0) noexcept;
    bool retire_after_drain(const char* reason="owner retirement") noexcept;
    Snapshot snapshot() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Process-wide attempt counters are a test/real-caller dispatch seam only.
// They do not establish completion or performance.
Exl3PrefillAttentionChainGraph::Snapshot
exl3_prefill_attention_chain_global_snapshot() noexcept;

} // namespace ninfer::exl3
