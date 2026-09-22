#include "exl3/prefill_projection_chain_graph.h"

#include "core/device.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::exl3 {

struct Exl3PrefillProjectionChainGraph::Impl {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    Exl3BoundedGraphEntry entry;
    Snapshot counters;
    bool enabled=false;
    std::uint64_t pending_serial=0;
    Impl* quarantine_next=nullptr;
    inline static std::atomic<Impl*> quarantine{nullptr};

    void retain_quarantined() noexcept {
        ++counters.quarantines;
        auto* head=quarantine.load(std::memory_order_relaxed);
        do {quarantine_next=head;} while(!quarantine.compare_exchange_weak(
            head,this,std::memory_order_release,std::memory_order_relaxed));
    }
};

Exl3PrefillProjectionChainGraph::Exl3PrefillProjectionChainGraph()
    : impl_(std::make_unique<Impl>()) {
    const char* value=std::getenv("NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS");
    if(value && std::strcmp(value,"0")!=0 && std::strcmp(value,"1")!=0)
        throw std::invalid_argument(
            "NINFER_EXL3_PREFILL_PROJECTION_CHAIN_GRAPHS must be 0 or 1");
    impl_->enabled=value && std::strcmp(value,"1")==0;
}

Exl3PrefillProjectionChainGraph::~Exl3PrefillProjectionChainGraph() {
    if(!impl_)return;
    if(impl_->entry.pending() || !retire_after_drain("destructor")) {
        impl_->entry.quarantine("pending or failed graph retirement",cudaErrorUnknown);
        impl_->retain_quarantined();
        (void)impl_.release();
    }
}

bool Exl3PrefillProjectionChainGraph::execute(const Request& request,const Body& body) {
    if(!body)throw std::invalid_argument("prefill projection chain body");
    if(!impl_->enabled || !request.eligible || !request.fingerprint.valid() || !request.generation ||
       !request.reservation_configuration || request.resources.empty() ||
       !request.reserved_peak.known) {
        ++impl_->counters.eager_fallbacks;body(request.stream);return false;
    }
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(request.stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone) {
        ++impl_->counters.nested_captures;body(request.stream);return false;
    }
    // The physical-C1 caller currently uses CUDA's legacy/default stream,
    // which cannot begin stream capture.  Keep this default-off experiment
    // recoverable for explicit-stream callers without crashing its real
    // caller; replacing the stream changes the selected ownership contract.
    if(!request.stream) {
        ++impl_->counters.eager_fallbacks;body(request.stream);return false;
    }
    if(impl_->executable.ready()) {
        if(!impl_->entry.admits(request.fingerprint,request.generation)) {
            ++impl_->counters.binding_mismatches;
            ++impl_->counters.eager_fallbacks;body(request.stream);return false;
        }
        impl_->pending_serial=impl_->entry.begin_replay(
            request.generation,request.stream);
        impl_->executable.launch(request.stream);
        ++impl_->counters.replays;return true;
    }

    // The compatibility fingerprint and final-use record bind the caller's
    // stream. Capture the selected kernels on that same stream as well: a
    // substitute nonblocking stream can silently change stream-qualified
    // workspace/dispatch behavior while still replaying into the caller.
    impl_->definition.capture(request.stream,[&] {body(request.stream);});
    impl_->executable.instantiate(impl_->definition);
    impl_->executable.upload(request.stream);
    impl_->entry.bind(request.fingerprint,request.resources,request.generation,
        request.reservation_configuration,request.reserved_peak);
    ++impl_->counters.captures;
    impl_->pending_serial=impl_->entry.begin_replay(
        request.generation,request.stream);
    impl_->executable.launch(request.stream);
    ++impl_->counters.replays;return true;
}

bool Exl3PrefillProjectionChainGraph::complete_after_drain(
    cudaStream_t stream,int error) noexcept {
    if(!impl_ || !impl_->pending_serial)return false;
    const bool result=impl_->entry.complete_after_drain(
        stream,impl_->pending_serial,error);
    if(!impl_->entry.pending())impl_->pending_serial=0;
    return result;
}

bool Exl3PrefillProjectionChainGraph::retire_after_drain(const char* reason) noexcept {
    if(!impl_ || impl_->entry.pending())return false;
    if(!impl_->executable.ready() && !impl_->definition.ready())return true;
    impl_->entry.invalidate(reason?reason:"owner retirement");
    const auto executable_error=impl_->executable.try_reset();
    const auto definition_error=executable_error==cudaSuccess
        ?impl_->definition.try_reset():executable_error;
    if(executable_error!=cudaSuccess || definition_error!=cudaSuccess) {
        impl_->entry.quarantine("CUDA graph handle retirement failure",
            static_cast<int>(executable_error!=cudaSuccess?executable_error:definition_error));
        return false;
    }
    return impl_->entry.release_after_destroy();
}

Exl3PrefillProjectionChainGraph::Snapshot
Exl3PrefillProjectionChainGraph::snapshot() const noexcept {
    if(!impl_)return {};
    auto result=impl_->counters;
    result.ready=impl_->executable.ready();
    result.pending=impl_->entry.pending();
    return result;
}

} // namespace ninfer::exl3
