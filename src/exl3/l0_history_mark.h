#pragma once
// "History rows written" mark for the L0 exact-history staging copies
// (l0_exact_history_staged). A multi-row forward pass over all layers records,
// at its start, an event on its stream after which rows [0, limit) of every
// attention layer's L2 planes hold their final values for the pass: limit is
// the pass position (layer-major: the block's first row). The staging copy
// stream copies those rows under that event, overlapping the layers before
// the attention call; newer rows wait for the call itself. Without a live mark
// (other callers, graph capture) every copy waits for the call.
#include <cuda_runtime.h>

namespace ninfer::exl3 {

struct L0HistoryWritten {
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    int limit = -1;
};

inline L0HistoryWritten& l0_history_written() {
    thread_local L0HistoryWritten value;
    return value;
}

class L0HistoryWrittenScope {
public:
    L0HistoryWrittenScope(int limit, cudaStream_t stream) : saved_(l0_history_written()) {
        cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
        if (limit <= 0 || cudaStreamIsCapturing(stream, &capturing) != cudaSuccess ||
            capturing != cudaStreamCaptureStatusNone ||
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) != cudaSuccess) {
            event_ = nullptr;
            l0_history_written() = {};
            return;
        }
        if (cudaEventRecord(event_, stream) != cudaSuccess) {
            cudaEventDestroy(event_);
            event_ = nullptr;
            l0_history_written() = {};
            return;
        }
        l0_history_written() = {stream, event_, limit};
    }
    ~L0HistoryWrittenScope() {
        l0_history_written() = saved_;
        if (event_) cudaEventDestroy(event_);
    }
    L0HistoryWrittenScope(const L0HistoryWrittenScope&) = delete;
    L0HistoryWrittenScope& operator=(const L0HistoryWrittenScope&) = delete;

private:
    L0HistoryWritten saved_;
    cudaEvent_t event_ = nullptr;
};

}  // namespace ninfer::exl3
