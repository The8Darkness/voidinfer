#include "exl3/prefill_attention_chain_graph.h"

#include "core/device.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::exl3 {
namespace {
struct GlobalCounters {
    std::atomic<std::uint64_t> capture_setups{0},captures{0},replays{0};
    std::atomic<std::uint64_t> eager_fallbacks{0},binding_mismatches{0};
    std::atomic<std::uint64_t> nested_captures{0},quarantines{0};
};
GlobalCounters global;
}

struct Exl3PrefillAttentionChainGraph::Impl {
    struct Slot {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        Exl3BoundedGraphEntry entry;
        std::uint64_t pending_serial=0;
    };
    std::array<Slot,slot_count> slots;
    Snapshot counters;
    bool enabled=false;
    Impl* quarantine_next=nullptr;
    inline static std::atomic<Impl*> quarantine{nullptr};

    void retain_quarantined() noexcept {
        ++counters.quarantines;++global.quarantines;
        auto* head=quarantine.load(std::memory_order_relaxed);
        do {quarantine_next=head;} while(!quarantine.compare_exchange_weak(
            head,this,std::memory_order_release,std::memory_order_relaxed));
    }
};

Exl3PrefillAttentionChainGraph::Exl3PrefillAttentionChainGraph()
    : impl_(std::make_unique<Impl>()) {
    const char* value=std::getenv("NINFER_EXL3_PREFILL_ATTENTION_CHAIN_GRAPH");
    if(value && std::strcmp(value,"0")!=0 && std::strcmp(value,"1")!=0)
        throw std::invalid_argument(
            "NINFER_EXL3_PREFILL_ATTENTION_CHAIN_GRAPH must be 0 or 1");
    impl_->enabled=value && std::strcmp(value,"1")==0;
}

Exl3PrefillAttentionChainGraph::~Exl3PrefillAttentionChainGraph() {
    if(!impl_)return;
    bool pending=false;
    for(const auto& slot:impl_->slots)pending=pending||slot.entry.pending();
    if(pending || !retire_after_drain("destructor")) {
        for(auto& slot:impl_->slots)
            slot.entry.quarantine("pending or failed graph retirement",cudaErrorUnknown);
        impl_->retain_quarantined();
        (void)impl_.release();
    }
}

bool Exl3PrefillAttentionChainGraph::execute(
    const Request& request,const Body& body) {
    if(!body)throw std::invalid_argument("prefill attention chain body");
    if(!impl_->enabled || !request.eligible || request.slot<0 ||
       request.slot>=slot_count || !request.fingerprint.valid() ||
       !request.generation || !request.reservation_configuration ||
       request.resources.empty() || !request.reserved_peak.known) {
        ++impl_->counters.eager_fallbacks;++global.eager_fallbacks;
        body(request.stream);return false;
    }
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(request.stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone) {
        ++impl_->counters.nested_captures;++global.nested_captures;
        body(request.stream);return false;
    }
    auto& slot=impl_->slots[static_cast<std::size_t>(request.slot)];
    if(slot.executable.ready()) {
        if(!slot.entry.admits(request.fingerprint,request.generation)) {
            ++impl_->counters.binding_mismatches;++global.binding_mismatches;
            ++impl_->counters.eager_fallbacks;++global.eager_fallbacks;
            body(request.stream);return false;
        }
        slot.pending_serial=slot.entry.begin_replay(
            request.generation,request.stream);
        slot.executable.launch(request.stream);
        ++impl_->counters.replays;++global.replays;return true;
    }

    ++impl_->counters.capture_setups;++global.capture_setups;
    cudaStream_t capture_stream=nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking));
    try {
        slot.definition.capture(capture_stream,[&] {body(capture_stream);});
    } catch(...) {
        (void)cudaStreamDestroy(capture_stream);throw;
    }
    CUDA_CHECK(cudaStreamDestroy(capture_stream));
    slot.executable.instantiate(slot.definition);
    slot.executable.upload(request.stream);
    slot.entry.bind(request.fingerprint,request.resources,request.generation,
        request.reservation_configuration,request.reserved_peak);
    ++impl_->counters.captures;++global.captures;
    slot.pending_serial=slot.entry.begin_replay(
        request.generation,request.stream);
    slot.executable.launch(request.stream);
    ++impl_->counters.replays;++global.replays;return true;
}

bool Exl3PrefillAttentionChainGraph::complete_after_drain(
    cudaStream_t stream,int error) noexcept {
    if(!impl_)return false;
    bool had_pending=false,ok=true;
    for(auto& slot:impl_->slots)if(slot.pending_serial) {
        had_pending=true;
        ok=slot.entry.complete_after_drain(
            stream,slot.pending_serial,error)&&ok;
        if(!slot.entry.pending())slot.pending_serial=0;
    }
    return had_pending&&ok;
}

bool Exl3PrefillAttentionChainGraph::retire_after_drain(
    const char* reason) noexcept {
    if(!impl_)return true;
    for(const auto& slot:impl_->slots)if(slot.entry.pending())return false;
    bool ok=true;
    for(auto& slot:impl_->slots) {
        if(!slot.executable.ready() && !slot.definition.ready())continue;
        slot.entry.invalidate(reason?reason:"owner retirement");
        const auto executable_error=slot.executable.try_reset();
        const auto definition_error=executable_error==cudaSuccess
            ?slot.definition.try_reset():executable_error;
        if(executable_error!=cudaSuccess || definition_error!=cudaSuccess) {
            slot.entry.quarantine("CUDA graph handle retirement failure",
                static_cast<int>(executable_error!=cudaSuccess?
                    executable_error:definition_error));
            ok=false;continue;
        }
        ok=slot.entry.release_after_destroy()&&ok;
    }
    return ok;
}

Exl3PrefillAttentionChainGraph::Snapshot
Exl3PrefillAttentionChainGraph::snapshot() const noexcept {
    if(!impl_)return {};
    auto result=impl_->counters;
    for(const auto& slot:impl_->slots) {
        result.ready_slots+=slot.executable.ready()?1:0;
        result.pending_slots+=slot.entry.pending()?1:0;
    }
    return result;
}

Exl3PrefillAttentionChainGraph::Snapshot
exl3_prefill_attention_chain_global_snapshot() noexcept {
    Exl3PrefillAttentionChainGraph::Snapshot result;
    result.capture_setups=global.capture_setups.load(std::memory_order_relaxed);
    result.captures=global.captures.load(std::memory_order_relaxed);
    result.replays=global.replays.load(std::memory_order_relaxed);
    result.eager_fallbacks=global.eager_fallbacks.load(std::memory_order_relaxed);
    result.binding_mismatches=global.binding_mismatches.load(std::memory_order_relaxed);
    result.nested_captures=global.nested_captures.load(std::memory_order_relaxed);
    result.quarantines=global.quarantines.load(std::memory_order_relaxed);
    return result;
}

} // namespace ninfer::exl3
