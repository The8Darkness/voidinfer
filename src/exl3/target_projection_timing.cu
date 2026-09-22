#include "exl3/target_projection_timing.h"

#include <stdexcept>
#include <string>

namespace ninfer::exl3 {
namespace {

void timing_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

} // namespace

Exl3TargetProjectionTiming::Exl3TargetProjectionTiming() {
    try {
        for (auto& slot : slots_) {
            timing_check(cudaEventCreate(&slot.start),
                         "create target projection start event");
            timing_check(cudaEventCreate(&slot.end),
                         "create target projection end event");
        }
    } catch (...) {
        for (auto& slot : slots_) {
            if (slot.start) cudaEventDestroy(slot.start);
            if (slot.end) cudaEventDestroy(slot.end);
            slot.start = nullptr;
            slot.end = nullptr;
        }
        throw;
    }
}

Exl3TargetProjectionTiming::~Exl3TargetProjectionTiming() {
    for (auto& slot : slots_) {
        if (slot.start) cudaEventDestroy(slot.start);
        if (slot.end) cudaEventDestroy(slot.end);
    }
}

void Exl3TargetProjectionTiming::begin_round(int round, cudaStream_t stream) {
    if (active_) throw std::runtime_error("target projection timing round is already active");
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    timing_check(cudaStreamIsCapturing(stream, &capture_status),
                 "query target projection timing stream capture");
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::runtime_error(
            "target projection timing is unavailable during stream capture");
    }
    round_ = round;
    used_ = 0;
    phase_ = Exl3TargetProjectionPhase::attempt;
    stream_ = stream;
    active_ = true;
}

int Exl3TargetProjectionTiming::begin(
    int layer, Exl3TargetProjectionOperator operation, int rows, int K,
    int in_features, int out_features, Exl3TargetProjectionTopology topology,
    int calls, cudaStream_t stream) {
    if (!active_) return -1;
    if (used_ >= kMaxSlots) {
        throw std::runtime_error("target projection timing slot capacity exceeded");
    }
    if (stream != stream_) {
        throw std::runtime_error("target projection timing stream mismatch");
    }
    const int index = used_++;
    auto& slot = slots_[static_cast<std::size_t>(index)];
    slot.ended = false;
    slot.metadata = {round_, phase_, layer, operation, rows, K, in_features,
                     out_features, topology, calls, 0.0};
    timing_check(cudaEventRecord(slot.start, stream),
                 "record target projection start event");
    return index;
}

void Exl3TargetProjectionTiming::end(int slot, cudaStream_t stream) {
    if (slot < 0) return;
    if (!active_ || slot >= used_) {
        throw std::runtime_error("target projection timing slot is invalid");
    }
    if (stream != stream_) {
        throw std::runtime_error("target projection timing stream mismatch");
    }
    auto& record = slots_[static_cast<std::size_t>(slot)];
    timing_check(cudaEventRecord(record.end, stream),
                 "record target projection end event");
    record.ended = true;
}

void Exl3TargetProjectionTiming::validate_execution_stream(cudaStream_t stream) const {
    if (!active_) return;
    if (stream != stream_) {
        throw std::runtime_error("target projection timing stream mismatch");
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    timing_check(cudaStreamIsCapturing(stream, &capture_status),
                 "query active target projection timing stream capture");
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::runtime_error(
            "active target projection timing is unavailable during stream capture");
    }
}

std::vector<Exl3TargetProjectionTimingRecord>
Exl3TargetProjectionTiming::finish_after_synchronize() {
    if (!active_) throw std::runtime_error("target projection timing round is not active");
    active_ = false;
    std::vector<Exl3TargetProjectionTimingRecord> result;
    result.reserve(static_cast<std::size_t>(used_));
    for (int index = 0; index < used_; ++index) {
        auto& slot = slots_[static_cast<std::size_t>(index)];
        if (!slot.ended) {
            throw std::runtime_error("target projection timing slot has no end event");
        }
        const cudaError_t ready = cudaEventQuery(slot.end);
        if (ready == cudaErrorNotReady) {
            throw std::runtime_error(
                "target projection timing resolved before round synchronization");
        }
        timing_check(ready, "query target projection timing end event");
        float milliseconds = 0.0f;
        timing_check(cudaEventElapsedTime(&milliseconds, slot.start, slot.end),
                     "resolve target projection timing event");
        slot.metadata.microseconds = static_cast<double>(milliseconds) * 1000.0;
        result.push_back(slot.metadata);
    }
    return result;
}

const char* target_projection_phase_name(Exl3TargetProjectionPhase phase) noexcept {
    return phase == Exl3TargetProjectionPhase::attempt ? "attempt" : "replay";
}

const char* target_projection_operator_name(Exl3TargetProjectionOperator operation) noexcept {
    switch (operation) {
    case Exl3TargetProjectionOperator::q: return "q";
    case Exl3TargetProjectionOperator::k: return "k";
    case Exl3TargetProjectionOperator::v: return "v";
    case Exl3TargetProjectionOperator::o: return "o";
    case Exl3TargetProjectionOperator::gate: return "gate";
    case Exl3TargetProjectionOperator::up: return "up";
    case Exl3TargetProjectionOperator::down: return "down";
    case Exl3TargetProjectionOperator::qkv: return "qkv";
    case Exl3TargetProjectionOperator::z: return "z";
    case Exl3TargetProjectionOperator::lm_head: return "lm_head";
    }
    return "unknown";
}

const char* target_projection_topology_name(Exl3TargetProjectionTopology topology) noexcept {
    switch (topology) {
    case Exl3TargetProjectionTopology::m1: return "m1";
    case Exl3TargetProjectionTopology::m1_per_row: return "m1_per_row";
    case Exl3TargetProjectionTopology::batched: return "batched";
    case Exl3TargetProjectionTopology::small_m_single_split:
        return "small_m_single_split";
    case Exl3TargetProjectionTopology::small_m_mma_split:
        return "small_m_mma_split";
    }
    return "unknown";
}

} // namespace ninfer::exl3
