#include "artifact/materializer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::artifact {
namespace {

constexpr std::size_t kSlotBytes        = 64ULL * 1024ULL * 1024ULL;
constexpr std::size_t kMaximumSlotCount = 4;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) { throw ArtifactError(label); }
    return a + b;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value / alignment * alignment;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, const char* label) {
    return checked_add(value, alignment - 1, label) / alignment * alignment;
}

class Slot {
public:
    explicit Slot(std::size_t bytes) : buffer(bytes) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    ~Slot() {
        if (pending) { (void)cudaEventSynchronize(event); }
        if (event != nullptr) { (void)cudaEventDestroy(event); }
    }

    void wait() {
        if (pending) {
            CUDA_CHECK(cudaEventSynchronize(event));
            pending = false;
        }
    }

    PinnedHostBuffer buffer;
    cudaEvent_t event = nullptr;
    bool pending      = false;
};

struct CopyRange {
    std::uint64_t source_begin = 0;
    std::uint64_t source_end   = 0;
    std::byte* destination     = nullptr;
};

struct ReadSpan {
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

struct H2DEventPairs {
    std::vector<cudaEvent_t> starts;
    std::vector<cudaEvent_t> stops;

    ~H2DEventPairs() {
        for (cudaEvent_t event : starts) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
        for (cudaEvent_t event : stops) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
    }

    void initialize(std::size_t count) {
        starts.resize(count);
        stops.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            CUDA_CHECK(cudaEventCreate(&starts[i]));
            CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
    }
};

struct DualReadTask {
    std::size_t index       = 0;
    std::uint64_t source   = 0;
    std::size_t request    = 0;
    std::uint64_t required = 0;
    std::size_t source_id  = 0;
};

struct DualReadAck {
    std::mutex mutex;
    std::condition_variable condition;
    bool released = false;

    void release() {
        {
            std::lock_guard lock(mutex);
            released = true;
        }
        condition.notify_one();
    }

    void wait() {
        std::unique_lock lock(mutex);
        condition.wait(lock, [this] { return released; });
    }
};

struct DualReadResult {
    DualReadTask task;
    Slot* slot = nullptr;
    std::size_t bytes_read = 0;
    std::shared_ptr<DualReadAck> ack;
};

class DualReadResults {
public:
    void push(DualReadResult result) {
        {
            std::lock_guard lock(mutex_);
            results_.push_back(std::move(result));
        }
        condition_.notify_one();
    }

    void fail(std::exception_ptr error) {
        {
            std::lock_guard lock(mutex_);
            if (error_ == nullptr) { error_ = error; }
            for (DualReadResult& result : results_) { result.ack->release(); }
        }
        condition_.notify_all();
    }

    bool failed() {
        std::lock_guard lock(mutex_);
        return error_ != nullptr;
    }

    DualReadResult pop() {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [this] { return !results_.empty() || error_ != nullptr; });
        if (error_ != nullptr) { std::rethrow_exception(error_); }
        DualReadResult result = std::move(results_.front());
        results_.pop_front();
        return result;
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<DualReadResult> results_;
    std::exception_ptr error_;
};

} // namespace

void* MaterializedArtifact::device_data(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].device == nullptr) {
        throw ArtifactError("object handle does not name a materialized tensor");
    }
    return objects_[handle.index].device;
}

std::span<const std::byte> MaterializedArtifact::resource_bytes(ObjectHandle handle) const {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    return objects_[handle.index].resource;
}

std::vector<std::byte> MaterializedArtifact::take_resource_bytes(ObjectHandle handle) {
    if (handle.index >= objects_.size() || objects_[handle.index].resource.empty()) {
        throw ArtifactError("object handle does not name a materialized resource");
    }
    auto& resource = objects_[handle.index].resource;
    stats_.retained_resource_bytes -= resource.size();
    return std::move(resource);
}

DeviceArena& MaterializedArtifact::device_arena() {
    if (!device_arena_) { throw ArtifactError("artifact has no device tensor backing"); }
    return *device_arena_;
}

MaterializedArtifact materialize(const Reader& reader, const MaterializationPlan& plan,
                                 DeviceContext& device, LoadProgress* progress) {
    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    const auto device_allocation_start = std::chrono::steady_clock::now();
    out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
    out.stats_.device_allocation_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      device_allocation_start)
            .count();
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    const auto host_copy_start = std::chrono::steady_clock::now();
    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }
    out.stats_.host_resource_copy_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - host_copy_start).count();

    std::vector<CopyRange> ranges;
    ranges.reserve(plan.device_objects.size());
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        const PayloadSpan payload = reader.payload(reader.objects().at(placement.object.index));
        DeviceSpan storage =
            out.device_arena_->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                           static_cast<std::size_t>(placement.alignment));
        const auto actual_offset =
            static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                       static_cast<std::byte*>(out.device_arena_->base()));
        if (actual_offset != placement.offset || payload.data.size() != placement.bytes) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device = storage.data;
        ranges.push_back(CopyRange{
            .source_begin = payload.absolute_offset,
            .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                        "artifact tensor source range overflows u64"),
            .destination  = static_cast<std::byte*>(storage.data),
        });
        total = checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                   "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t slot_bytes =
        static_cast<std::size_t>(std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    const std::size_t slot_count = static_cast<std::size_t>(
        std::min<std::uint64_t>(kMaximumSlotCount, 1 + (aligned_read_bytes - 1) / slot_bytes));
    std::size_t chunk_count = 0;
    for (const ReadSpan& span : read_spans) {
        chunk_count += static_cast<std::size_t>((span.end - span.begin + slot_bytes - 1) /
                                                slot_bytes);
    }
    const auto staging_allocation_start = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<Slot>> slots;
    slots.reserve(slot_count);
    for (std::size_t i = 0; i < slot_count; ++i) {
        slots.push_back(std::make_unique<Slot>(slot_bytes));
    }
    out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(slot_bytes) * slot_count;
    out.stats_.host_staging_allocation_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      staging_allocation_start)
            .count();

    std::size_t next_slot  = 0;
    std::size_t next_range = 0;
    std::size_t chunk_index = 0;
    double synchronization_seconds = 0.0;
    const auto start       = std::chrono::steady_clock::now();
    CudaEventTimer h2d_timer(device, device.transfer_stream);
    H2DEventPairs h2d_active_events;
    h2d_active_events.initialize(chunk_count);
    h2d_timer.start();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            Slot& slot = *slots[next_slot++ % slot_count];
            const auto wait_start = std::chrono::steady_clock::now();
            slot.wait();
            synchronization_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wait_start)
                    .count();

            const std::uint64_t remaining = span.end - source;
            const std::size_t request     = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            auto destination =
                std::span<std::byte>(static_cast<std::byte*>(slot.buffer.data()), request);
            const std::size_t bytes_read = reader.read_direct(source, destination);
            const std::uint64_t required = std::min<std::uint64_t>(request, remaining);
            if (bytes_read < required) {
                throw ArtifactError("direct artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes =
                checked_add(out.stats_.file_bytes, bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end =
                checked_add(source, bytes_read, "artifact direct I/O result overflows u64");

            CUDA_CHECK(cudaEventRecord(h2d_active_events.starts[chunk_index],
                                       device.transfer_stream));
            while (next_range < ranges.size() && ranges[next_range].source_end <= source) {
                ++next_range;
            }
            std::size_t range_index = next_range;
            while (range_index < ranges.size() && ranges[range_index].source_begin < chunk_end) {
                const CopyRange& range         = ranges[range_index];
                const std::uint64_t copy_begin = std::max(source, range.source_begin);
                const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                if (copy_begin < copy_end) {
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        static_cast<std::byte*>(slot.buffer.data()) +
                            static_cast<std::size_t>(copy_begin - source),
                        amount, cudaMemcpyHostToDevice, device.transfer_stream));
                    copied =
                        checked_add(copied, amount, "artifact copied byte count overflows u64");
                }
                if (range.source_end <= chunk_end) {
                    ++range_index;
                } else {
                    break;
                }
            }
            next_range = range_index;
            CUDA_CHECK(cudaEventRecord(h2d_active_events.stops[chunk_index],
                                       device.transfer_stream));
            ++chunk_index;
            CUDA_CHECK(cudaEventRecord(slot.event, device.transfer_stream));
            slot.pending = true;

            if (progress != nullptr && progress->callback && copied != last_published &&
                copied < total) {
                last_published = copied;
                progress->callback("weights", copied, total);
            }
        }
    }
    h2d_timer.record_stop();
    const auto final_sync_start = std::chrono::steady_clock::now();
    for (const auto& slot : slots) { slot->wait(); }
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    synchronization_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - final_sync_start)
            .count();
    if (copied != total || next_range != ranges.size()) {
        throw ArtifactError("direct materialization did not cover every tensor byte");
    }
    out.stats_.h2d_bytes = copied;
    out.stats_.h2d_stream_seconds = static_cast<double>(h2d_timer.elapsed_ms()) / 1000.0;
    double h2d_active_seconds = 0.0;
    for (std::size_t i = 0; i < chunk_index; ++i) {
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, h2d_active_events.starts[i],
                                        h2d_active_events.stops[i]));
        h2d_active_seconds += static_cast<double>(elapsed_ms) / 1000.0;
    }
    out.stats_.h2d_active_seconds = h2d_active_seconds;
    out.stats_.synchronization_seconds = synchronization_seconds;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

MaterializedArtifact materialize_dual(const Reader& primary, const Reader& secondary,
                                      const MaterializationPlan& plan, DeviceContext& device,
                                      ArtifactReadMode mode, LoadProgress* progress) {
    if (mode == ArtifactReadMode::Single) {
        throw ArtifactError("dual materialization requires a dual artifact read mode");
    }

    MaterializedArtifact out;
    out.objects_.resize(plan.object_count);
    const std::uint64_t capacity = plan.device_capacity_bytes;
    if (capacity == 0 || capacity > static_cast<std::uint64_t>(SIZE_MAX)) {
        throw ArtifactError("artifact tensor backing size is invalid");
    }
    const auto device_allocation_start = std::chrono::steady_clock::now();
    out.device_arena_ = std::make_unique<DeviceArena>(static_cast<std::size_t>(capacity));
    out.stats_.device_allocation_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      device_allocation_start)
            .count();
    out.stats_.device_capacity_bytes = capacity;
    out.stats_.tensor_count          = plan.device_objects.size();
    out.stats_.resource_count        = plan.host_objects.size();

    const auto host_copy_start = std::chrono::steady_clock::now();
    for (const HostMaterialization& placement : plan.host_objects) {
        auto& resource            = out.objects_.at(placement.object.index).resource;
        const PayloadSpan payload = primary.payload(primary.objects().at(placement.object.index));
        resource.assign(payload.data.begin(), payload.data.end());
        out.stats_.retained_resource_bytes += resource.size();
        out.stats_.file_bytes =
            checked_add(out.stats_.file_bytes, resource.size(), "artifact read bytes overflow u64");
    }
    out.stats_.host_resource_copy_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - host_copy_start).count();

    std::vector<CopyRange> ranges;
    ranges.reserve(plan.device_objects.size());
    std::uint64_t copied         = 0;
    std::uint64_t last_published = 0;
    std::uint64_t total          = 0;
    for (const DeviceMaterialization& placement : plan.device_objects) {
        const PayloadSpan payload = primary.payload(primary.objects().at(placement.object.index));
        DeviceSpan storage =
            out.device_arena_->alloc_bytes(static_cast<std::size_t>(placement.bytes),
                                           static_cast<std::size_t>(placement.alignment));
        const auto actual_offset =
            static_cast<std::uint64_t>(static_cast<std::byte*>(storage.data) -
                                       static_cast<std::byte*>(out.device_arena_->base()));
        if (actual_offset != placement.offset || payload.data.size() != placement.bytes) {
            throw ArtifactError("materialization plan does not match artifact payload");
        }
        out.objects_.at(placement.object.index).device = storage.data;
        ranges.push_back(CopyRange{
            .source_begin = payload.absolute_offset,
            .source_end   = checked_add(payload.absolute_offset, placement.bytes,
                                        "artifact tensor source range overflows u64"),
            .destination  = static_cast<std::byte*>(storage.data),
        });
        total = checked_add(total, placement.bytes, "artifact tensor byte count overflows u64");
    }
    if (ranges.empty()) { throw ArtifactError("materialization plan has no device tensors"); }
    std::sort(ranges.begin(), ranges.end(), [](const CopyRange& a, const CopyRange& b) {
        return a.source_begin < b.source_begin;
    });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].source_begin < ranges[i - 1].source_end) {
            throw ArtifactError("materialization source ranges overlap");
        }
    }

    constexpr std::uint64_t alignment = Reader::direct_io_alignment;
    std::vector<ReadSpan> read_spans;
    read_spans.reserve(ranges.size());
    std::uint64_t aligned_read_bytes = 0;
    for (const CopyRange& range : ranges) {
        const std::uint64_t begin = align_down(range.source_begin, alignment);
        if (read_spans.empty() || begin > align_up(read_spans.back().end, alignment,
                                                    "artifact direct I/O span overflows u64")) {
            read_spans.push_back(ReadSpan{begin, range.source_end});
        } else {
            read_spans.back().end = std::max(read_spans.back().end, range.source_end);
        }
    }
    for (const ReadSpan& span : read_spans) {
        aligned_read_bytes = checked_add(
            aligned_read_bytes,
            align_up(span.end - span.begin, alignment, "artifact direct I/O span overflows u64"),
            "artifact direct I/O byte count overflows u64");
    }
    const std::size_t slot_bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(kSlotBytes, aligned_read_bytes));
    constexpr std::size_t kDualSlotsPerDrive = 2;
    std::vector<std::vector<std::unique_ptr<Slot>>> slots(2);
    const auto staging_allocation_start = std::chrono::steady_clock::now();
    for (auto& drive_slots : slots) {
        drive_slots.reserve(kDualSlotsPerDrive);
        for (std::size_t i = 0; i < kDualSlotsPerDrive; ++i) {
            drive_slots.push_back(std::make_unique<Slot>(slot_bytes));
        }
    }
    out.stats_.peak_staging_bytes = static_cast<std::uint64_t>(slot_bytes) * 2 * kDualSlotsPerDrive;
    out.stats_.host_staging_allocation_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      staging_allocation_start)
            .count();

    std::vector<DualReadTask> tasks;
    for (const ReadSpan& span : read_spans) {
        for (std::uint64_t source = span.begin; source < span.end; source += slot_bytes) {
            const std::uint64_t remaining = span.end - source;
            const std::size_t request = static_cast<std::size_t>(std::min<std::uint64_t>(
                slot_bytes,
                align_up(remaining, alignment, "artifact direct I/O request overflows u64")));
            tasks.push_back(DualReadTask{
                .index       = tasks.size(),
                .source      = source,
                .request     = request,
                .required    = std::min<std::uint64_t>(request, remaining),
                .source_id   = mode == ArtifactReadMode::DualStaticAlternating
                                   ? tasks.size() % 2
                                   : 0,
            });
        }
    }
    if (tasks.empty()) { throw ArtifactError("dual materialization has no read tasks"); }

    const auto start = std::chrono::steady_clock::now();
    DualReadResults results;
    std::atomic<std::size_t> next_task{0};
    std::atomic<std::uint32_t> active_reads{0};
    std::atomic<std::uint32_t> max_active_reads{0};
    std::atomic<std::int64_t> first_read_ns{0};
    std::atomic<std::int64_t> last_read_ns{0};
    std::vector<std::thread> workers;
    workers.reserve(2);
    for (std::size_t drive = 0; drive < 2; ++drive) {
        workers.emplace_back([&, drive] {
            try {
                const Reader& reader = drive == 0 ? primary : secondary;
                std::size_t slot_index = 0;
                std::size_t static_cursor = drive;
                for (;;) {
                    if (results.failed()) { return; }
                    std::size_t task_index = 0;
                    if (mode == ArtifactReadMode::DualStaticAlternating) {
                        task_index = static_cursor;
                        static_cursor += 2;
                        if (task_index >= tasks.size()) { return; }
                    } else {
                        task_index = next_task.fetch_add(1, std::memory_order_relaxed);
                        if (task_index >= tasks.size()) { return; }
                    }
                    Slot& slot = *slots[drive][slot_index++ % kDualSlotsPerDrive];
                    slot.wait();
                    const DualReadTask& task = tasks[task_index];
                    auto destination = std::span<std::byte>(
                        static_cast<std::byte*>(slot.buffer.data()), task.request);
                    const std::uint32_t active = active_reads.fetch_add(
                                                       1, std::memory_order_acq_rel) +
                                                 1;
                    auto observed = max_active_reads.load(std::memory_order_relaxed);
                    while (observed < active &&
                           !max_active_reads.compare_exchange_weak(
                               observed, active, std::memory_order_release,
                               std::memory_order_relaxed)) {}
                    const auto read_start = std::chrono::steady_clock::now();
                    const auto read_start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   read_start.time_since_epoch())
                                                   .count();
                    std::int64_t unset = 0;
                    first_read_ns.compare_exchange_strong(
                        unset, read_start_ns, std::memory_order_release,
                        std::memory_order_relaxed);
                    std::size_t bytes_read = 0;
                    try {
                        bytes_read = reader.read_direct(task.source, destination);
                    } catch (...) {
                        active_reads.fetch_sub(1, std::memory_order_acq_rel);
                        throw;
                    }
                    const auto read_end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  std::chrono::steady_clock::now().time_since_epoch())
                                                  .count();
                    auto last_observed = last_read_ns.load(std::memory_order_relaxed);
                    while (last_observed < read_end_ns &&
                           !last_read_ns.compare_exchange_weak(
                               last_observed, read_end_ns, std::memory_order_release,
                               std::memory_order_relaxed)) {}
                    active_reads.fetch_sub(1, std::memory_order_acq_rel);
                    auto ack = std::make_shared<DualReadAck>();
                    results.push(DualReadResult{
                        .task       = task,
                        .slot       = &slot,
                        .bytes_read = bytes_read,
                        .ack        = ack,
                    });
                    // The consumer records the H2D copy and then releases this slot for reuse.
                    ack->wait();
                }
            } catch (...) {
                results.fail(std::current_exception());
            }
        });
    }

    CudaEventTimer h2d_timer(device, device.transfer_stream);
    H2DEventPairs h2d_active_events;
    h2d_active_events.initialize(tasks.size());
    h2d_timer.start();
    if (progress != nullptr && progress->callback) { progress->callback("weights", 0, total); }

    std::optional<DualReadResult> active_result;
    try {
        for (std::size_t completed = 0; completed < tasks.size(); ++completed) {
            active_result = results.pop();
            DualReadResult& result = *active_result;
            const DualReadTask& task = result.task;
            if (result.bytes_read < task.required) {
                throw ArtifactError("dual artifact read ended before the planned tensor range");
            }
            out.stats_.file_bytes = checked_add(
                out.stats_.file_bytes, result.bytes_read, "artifact read bytes overflow u64");
            const std::uint64_t chunk_end = checked_add(
                task.source, result.bytes_read, "artifact direct I/O result overflows u64");
            CUDA_CHECK(
                cudaEventRecord(h2d_active_events.starts[task.index], device.transfer_stream));
            for (const CopyRange& range : ranges) {
                const std::uint64_t copy_begin = std::max(task.source, range.source_begin);
                const std::uint64_t copy_end   = std::min(chunk_end, range.source_end);
                if (copy_begin < copy_end) {
                    const auto amount = static_cast<std::size_t>(copy_end - copy_begin);
                    CUDA_CHECK(cudaMemcpyAsync(
                        range.destination +
                            static_cast<std::size_t>(copy_begin - range.source_begin),
                        static_cast<std::byte*>(result.slot->buffer.data()) +
                            static_cast<std::size_t>(copy_begin - task.source),
                        amount, cudaMemcpyHostToDevice, device.transfer_stream));
                    copied = checked_add(copied, amount,
                                         "artifact copied byte count overflows u64");
                }
            }
            CUDA_CHECK(cudaEventRecord(h2d_active_events.stops[task.index],
                                       device.transfer_stream));
            CUDA_CHECK(cudaEventRecord(result.slot->event, device.transfer_stream));
            result.slot->pending = true;
            result.ack->release();
            active_result.reset();
            if (progress != nullptr && progress->callback && copied != last_published &&
                copied < total) {
                last_published = copied;
                progress->callback("weights", copied, total);
            }
        }
    } catch (...) {
        if (active_result.has_value()) { active_result->ack->release(); }
        results.fail(std::current_exception());
        for (auto& worker : workers) { worker.join(); }
        throw;
    }
    h2d_timer.record_stop();
    const auto final_sync_start = std::chrono::steady_clock::now();
    for (auto& drive_slots : slots) {
        for (auto& slot : drive_slots) { slot->wait(); }
    }
    CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
    const double synchronization_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - final_sync_start).count();
    for (auto& worker : workers) { worker.join(); }
    out.stats_.dual_max_parallel_reads = max_active_reads.load(std::memory_order_acquire);
    const auto first_read = first_read_ns.load(std::memory_order_acquire);
    const auto last_read = last_read_ns.load(std::memory_order_acquire);
    if (first_read != 0 && last_read >= first_read) {
        out.stats_.dual_direct_read_wall_seconds =
            static_cast<double>(last_read - first_read) / 1'000'000'000.0;
    }
    if (copied != total) { throw ArtifactError("dual materialization did not cover every tensor byte"); }
    out.stats_.h2d_bytes               = copied;
    out.stats_.h2d_stream_seconds      = static_cast<double>(h2d_timer.elapsed_ms()) / 1000.0;
    double h2d_active_seconds          = 0.0;
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, h2d_active_events.starts[i],
                                        h2d_active_events.stops[i]));
        h2d_active_seconds += static_cast<double>(elapsed_ms) / 1000.0;
    }
    out.stats_.h2d_active_seconds       = h2d_active_seconds;
    out.stats_.synchronization_seconds  = synchronization_seconds;
    out.stats_.upload_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (progress != nullptr && progress->callback) { progress->callback("weights", copied, total); }
    return out;
}

} // namespace ninfer::artifact
