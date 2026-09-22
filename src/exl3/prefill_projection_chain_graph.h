#pragma once

#include "exl3/bounded_graph_entry.h"
#include "core/decode_graph.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace ninfer::exl3 {

// One retained same-stream prefill projection chain. The body must enqueue the
// identical selected kernels on the supplied stream; it must not allocate,
// publish state, mutate HostKV, or synchronize. Dynamic state belongs outside
// this numerical slice.
class Exl3PrefillProjectionChainGraph {
public:
    struct Request {
        Exl3GraphCompatibilityFingerprint fingerprint;
        std::span<const Exl3GraphBoundResource> resources;
        Exl3GraphCaptureExtent reserved_peak;
        std::uint64_t generation=0;
        std::uint64_t reservation_configuration=0;
        cudaStream_t stream=nullptr;
        bool eligible=false;
    };
    struct Snapshot {
        std::uint64_t captures=0,replays=0,eager_fallbacks=0;
        std::uint64_t binding_mismatches=0,nested_captures=0;
        std::uint64_t quarantines=0;
        bool ready=false,pending=false;
    };
    using Body=std::function<void(cudaStream_t)>;

    Exl3PrefillProjectionChainGraph();
    ~Exl3PrefillProjectionChainGraph();
    Exl3PrefillProjectionChainGraph(const Exl3PrefillProjectionChainGraph&)=delete;
    Exl3PrefillProjectionChainGraph& operator=(const Exl3PrefillProjectionChainGraph&)=delete;

    // Returns true only when a graph was captured or replayed. False means the
    // exact eager body was submitted on request.stream.
    bool execute(const Request& request,const Body& body);
    // The owner calls this only after the replay stream is drained. It clears
    // the final-use marker without evicting a compatible executable.
    bool complete_after_drain(cudaStream_t stream,int error=0) noexcept;
    // Explicit final retirement. Failure retains the complete entry in a
    // process-lifetime quarantine instead of releasing bound owners.
    bool retire_after_drain(const char* reason="owner retirement") noexcept;
    Snapshot snapshot() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::exl3
