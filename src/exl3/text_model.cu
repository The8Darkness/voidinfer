#include "exl3/text_model.h"
#include "exl3/fixed_allocation_owners.h"
#include "exl3/host_kv_transfer_requirements.h"
#include "exl3/exact_page_extension_plan.h"
#include "exl3/linear_workspace_requirements.h"
#include "exl3/reconstruction_config.h"
#include "exl3/quant_descriptor.h"
#include "exl3/borrowed_descriptor.h"
#include "exl3/reconstruction_stream.h"
#include "exl3/reconstruction_device_retirement.h"
#include "exl3/reconstruction_control_allocator.h"
#include "exl3/vericache_serving_coordinator.h"
#include "exl3/greedy_packet.cuh"
#include "exl3/transfer_descriptor_storage.h"
#include "exl3/scatter_piece_storage.h"
#include "exl3/device_prefix_cache.h"
#include "exl3/native_context_extent.h"
#include "exl3/retirement_state.h"
#include "exl3/continuation_graph_drain_policy.h"
#include "exl3/turboangle_host.h"
#include "ops/softmax_attention/oscar_mixed/launch.h"
#include "core/nvtx_range.h"
#include "core/decode_graph.h"
#include "exl3/projection_graph_binding.h"
#include "exl3/export_copy_plan.h"
#include "exl3/bounded_graph_entry.h"
#include "exl3/device_graph_role_table.h"
#include "exl3/graph_numerical_boundary.h"
#include "exl3/head_consumer_plan.h"
#include "exl3/bounded_shared_owner.h"
#include "core/arena.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <unordered_set>

namespace ninfer::exl3 {
namespace {

using Json = nlohmann::json;
constexpr int kHidden = 5120;
constexpr int kVocab = 248320;
constexpr int kLayers = 64;
constexpr int kMaxRows = 16;
constexpr int kQHeads = 24;
constexpr int kKVHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kKVProjection = kKVHeads*kHeadDim;
constexpr int kIntermediate = 17408;
constexpr int kGdnQkv = 10240;
constexpr int kGdnZ = 6144;
constexpr std::array<int, 5> kTapLayers = {5, 19, 33, 47, 61};
std::atomic<std::uint64_t> next_exact_residency_id{1};

bool fast_device_kv_transaction_enabled() {
    const char* value=std::getenv("NINFER_EXL3_FAST_DEVICE_KV_TRANSACTION");
    if(value && std::strcmp(value,"0")!=0 && std::strcmp(value,"1")!=0)
        throw std::invalid_argument("fast device-KV transaction must be 0 or 1");
    return value && std::strcmp(value,"1")==0;
}
std::atomic<std::size_t> hostkv_quarantined_contexts{0};
// Process-wide ordinary device-KV graph witnesses.  Engine-owned contexts are
// not reachable from linked qualification tests, so capture/replay counts are
// mirrored here to prove which graph families actually executed.
struct OrdinaryGraphProcessCounters {
    std::atomic<std::uint64_t> gdn_segment_captures{0},gdn_segment_replays{0};
    std::atomic<std::uint64_t> full_layer_captures{0},full_layer_replays{0};
    std::atomic<std::uint64_t> mlp_tail_captures{0},mlp_tail_replays{0};
} ordinary_graph_process_counters;
void count_ordinary_graph(std::atomic<std::uint64_t>& counter) noexcept {
    counter.fetch_add(1,std::memory_order_relaxed);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

// Planning checks with literal diagnostics must not allocate before the complete
// resource promise is extended. Construct the exception only on failure; retain
// the string overload for callers that already own a composed diagnostic.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct ExactPageExtension {
    std::vector<std::shared_ptr<const Exl3ExactKVPage>> all;
    std::vector<std::shared_ptr<Exl3ExactKVPage>> fresh;
};

ExactPageExtension extend_exact_pages(
    const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& prefix,int old_position,int new_position,
    bool reuse_unique_tail=false,Exl3HostKVStats* stats=nullptr,unsigned fault_for_test=0,
    const Exl3TextContext::SnapshotMetadataReservation& reserve_metadata={}) {
    require(fault_for_test<=2,"exact page extension fault index");
    const auto started=stats?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    if(stats) ++stats->page_extension_calls;
    const auto plan=Exl3ExactPageExtensionPlan::derive(old_position,new_position);
    require(prefix.size()==Exl3ExactPageExtensionPlan::derive(old_position,old_position).all,
        "exact page extension missing prefix descriptors");
    ExactPageExtension result;
    const auto create_page=[&] {
        return Exl3ExactKVPage::create(reserve_metadata);
    };
    // Descriptor allocation must finish before a unique private tail is changed.
    result.all.reserve(plan.all);result.fresh.reserve(plan.fresh);
    std::shared_ptr<Exl3ExactKVPage> deferred_tail;
    int deferred_rows=0;
    for(std::size_t index=0;index<plan.all;++index) {
        const int first=static_cast<int>(index*Exl3ExactKVPage::token_capacity);
        const int rows=std::min(Exl3ExactKVPage::token_capacity,new_position-first);
        if(index<prefix.size() && first+rows<=old_position) {
            if(stats) ++stats->page_prefix_refs;
            const auto ref_started=stats?std::chrono::steady_clock::now():
                std::chrono::steady_clock::time_point{};
            result.all.push_back(prefix[index]);
            if(stats) stats->page_prefix_ref_cpu_ns+=static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-ref_started).count());
            continue;
        }
        const auto prepare_started=stats?std::chrono::steady_clock::now():
            std::chrono::steady_clock::time_point{};
        std::shared_ptr<Exl3ExactKVPage> page;
        if(index<prefix.size()) {
            std::uint64_t payload_bytes=0;
            for(int bank=0;bank<16;++bank)
                payload_bytes+=(prefix[index]->k[bank].size()+
                    prefix[index]->v[bank].size())*sizeof(std::uint16_t);
            // Pages are created mutable and exposed only as shared_ptr<const>.
            // A last-page strong count of one proves the context owns the only
            // reference: retained/exported roots necessarily raise the count.
            // Reuse is therefore copy-on-write equivalent without cloning the
            // private partial tail. Context mutation is single-threaded.
            if(reuse_unique_tail && index+1==prefix.size() &&
                    prefix[index].use_count()==1 && exl3_exact_tail_preallocated(*prefix[index])) {
                page=std::const_pointer_cast<Exl3ExactKVPage>(prefix[index]);
                deferred_tail=page;deferred_rows=rows;
                if(fault_for_test==1)throw std::bad_alloc();
                if(stats) {
                    ++stats->page_unique_tail_reuses;
                    stats->page_unique_tail_reuse_bytes+=payload_bytes;
                }
            } else {
                page=create_page();
                if(stats) stats->page_clone_bytes+=payload_bytes;
                const auto clone_started=stats?std::chrono::steady_clock::now():
                    std::chrono::steady_clock::time_point{};
                *page=*prefix[index];
                if(stats) stats->page_clone_cpu_ns+=static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-clone_started).count());
            }
        } else page=create_page();
        if(page!=deferred_tail) {
          page->first=first;page->rows=rows;
          for(int bank=0;bank<16;++bank) {
            page->k[bank].reserve(Exl3ExactKVPage::token_capacity*1024);
            page->v[bank].reserve(Exl3ExactKVPage::token_capacity*1024);
            page->k[bank].resize(rows*1024);page->v[bank].resize(rows*1024);
          }
        }
        result.all.push_back(page);result.fresh.push_back(std::move(page));
        if(stats) stats->page_payload_prepare_cpu_ns+=static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-prepare_started).count());
    }
    // All allocating work is finished; only now change the unique old tail.
    if(fault_for_test==2)throw std::bad_alloc();
    if(deferred_tail)exl3_extend_preallocated_tail(*deferred_tail,deferred_rows);
    if(stats) stats->page_extension_cpu_ns+=static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()-started).count());
    return result;
}

__host__ __device__ float half_to_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__host__ __device__ float bf16_to_float(std::uint16_t bits) {
    union { std::uint32_t u; float f; } value{static_cast<std::uint32_t>(bits) << 16u};
    return value.f;
}

__host__ __device__ std::uint16_t float_to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

TensorPayload load_tensor_single(const IndexedSafetensors& collection, const std::string& name) {
    for (const auto& shard : collection.shards) {
        if (shard.header.find(name) != nullptr) return read_tensor(shard.path, shard.header, name);
    }
    throw std::runtime_error("canonical EXL3 tensor is absent: " + name);
}

struct DeviceAllocation {
    struct Retained {
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        std::optional<RetainedDeviceLedger::Ticket> device_credit;
        void* pointer=nullptr;std::size_t bytes=0;int device=-1,error=0;Retained* next=nullptr;
    };
    inline static std::atomic<Retained*> quarantine{nullptr};
    inline static std::atomic<std::uint64_t> quarantined_count{0};
    std::unique_ptr<Retained> retirement=std::make_unique<Retained>();
    std::optional<RetainedDescriptorLedger::Ticket> owner_metadata_credit;
    int device=-1;
    bool cleanup_failure_for_test=false;
    bool device_query_failure_for_test=false;
    bool device_mismatch_for_test=false;
    bool shared_control_admitted=false;
    Exl3SharedControlCredit* shared_control_credit=nullptr;
    static constexpr std::size_t shared_control_bytes=Exl3ReconstructionControlAllocator<std::byte>::capacity;
    void* ptr = nullptr;
    std::size_t bytes = 0;
    static constexpr std::size_t owner_metadata_bytes() {return sizeof(DeviceAllocation)+sizeof(Retained);}
    static bool attach_control_credit(const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* allocation=static_cast<const DeviceAllocation*>(owner.get());
        if(!allocation || !allocation->shared_control_credit || allocation->shared_control_credit->ticket ||
            credit.bytes()!=shared_control_bytes)return false;
        allocation->shared_control_credit->ticket.emplace(std::move(credit));return true;
    }
    static bool attach_device_credit(const std::shared_ptr<const void>& owner,RetainedDeviceLedger::Ticket credit) noexcept {
        auto* allocation=const_cast<DeviceAllocation*>(static_cast<const DeviceAllocation*>(owner.get()));
        if(!allocation || !allocation->ptr || !allocation->retirement ||
            allocation->retirement->device_credit || credit.bytes()!=allocation->bytes)return false;
        allocation->retirement->device_credit.emplace(std::move(credit));return true;
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* allocation=const_cast<DeviceAllocation*>(static_cast<const DeviceAllocation*>(owner.get()));
        if(!allocation || !allocation->retirement || allocation->owner_metadata_credit ||
            allocation->retirement->metadata_credit || credit.bytes()!=owner_metadata_bytes())return false;
        auto record=credit.split(sizeof(Retained));if(!record)return false;
        allocation->retirement->metadata_credit.emplace(std::move(*record));
        allocation->owner_metadata_credit.emplace(std::move(credit));return true;
    }
    static std::shared_ptr<DeviceAllocation> create_shared(std::size_t bytes,const char* label,unsigned control_fault=0,
        std::optional<RetainedDeviceLedger::Ticket> device_credit={},
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit={}) {
        require(control_fault<=2,"generic shared control fault stage0..2");
        require(bool(device_credit)==bool(metadata_credit),"shared constructor requires both credit domains");
        std::optional<RetainedDescriptorLedger::Ticket> control_credit;
        if(metadata_credit) {
            require(metadata_credit->bytes()==owner_metadata_bytes()+shared_control_bytes,
                "shared constructor metadata extent mismatch");
            auto part=metadata_credit->split(shared_control_bytes);
            require(bool(part),"shared constructor control split failed");
            control_credit.emplace(std::move(*part));
        }
        auto* owner=new DeviceAllocation(bytes,label,std::move(device_credit),std::move(metadata_credit));
        owner->shared_control_admitted=control_fault!=0;
        owner->cleanup_failure_for_test=control_fault==2;
        auto result=std::shared_ptr<DeviceAllocation>(owner,[](DeviceAllocation* value) noexcept {delete value;},
            Exl3ReconstructionControlAllocator<std::byte>(&owner->shared_control_admitted,&owner->shared_control_credit));
        if(control_credit)owner->shared_control_credit->ticket.emplace(std::move(*control_credit));
        return result;
    }
    void prepare_device() {
        require(quarantined_count.load()==0,"unresolved generic allocation cleanup");
        cuda_check(cudaGetDevice(&device),"generic allocation device");
    }
    void release() noexcept {
        if(!ptr)return;
        int current=-1;auto error=cleanup_failure_for_test?cudaErrorUnknown:
            (device_query_failure_for_test?cudaErrorInitializationError:cudaGetDevice(&current));
        if(error==cudaSuccess && (device_mismatch_for_test || current!=device))error=cudaErrorInvalidDevice;
        if(error==cudaSuccess)error=cudaFree(ptr);
        if(error!=cudaSuccess) {
            auto* record=retirement.release();
            record->pointer=ptr;record->bytes=bytes;record->device=device;record->error=static_cast<int>(error);
            auto* head=quarantine.load(std::memory_order_relaxed);
            do{record->next=head;}while(!quarantine.compare_exchange_weak(
                head,record,std::memory_order_release,std::memory_order_relaxed));
            quarantined_count.fetch_add(1,std::memory_order_release);
        }
        if(error==cudaSuccess && retirement)retirement->device_credit.reset();
        ptr=nullptr;
    }

    void adopt_constructor_credits(std::optional<RetainedDeviceLedger::Ticket> device_credit,
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit) {
        require(bool(device_credit)==bool(metadata_credit),"generic constructor requires both credit domains");
        if(!device_credit)return;
        require(bytes && device_credit->bytes()==bytes && metadata_credit->bytes()==owner_metadata_bytes(),
            "generic constructor credit extent mismatch");
        auto record=metadata_credit->split(sizeof(Retained));
        require(bool(record),"generic constructor metadata split failed");
        retirement->metadata_credit.emplace(std::move(*record));
        owner_metadata_credit.emplace(std::move(*metadata_credit));
        retirement->device_credit.emplace(std::move(*device_credit));
    }
    void release_constructor_credits_after_commit() noexcept {
        // The committed inventory now owns this live allocation's reservation.
        retirement->device_credit.reset();retirement->metadata_credit.reset();owner_metadata_credit.reset();
        if(shared_control_credit)shared_control_credit->ticket.reset();
    }
    DeviceAllocation(std::span<const std::byte> source, const char* label,unsigned upload_fault_for_test=0,
        std::optional<RetainedDeviceLedger::Ticket> device_credit={},
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit={}) : bytes(source.size_bytes()) {
        require(upload_fault_for_test<=4,"generic upload fault stage0..4");
        adopt_constructor_credits(std::move(device_credit),std::move(metadata_credit));
        prepare_device();
        cuda_check(cudaMalloc(&ptr, bytes), label);
        try {
            cuda_check(cudaMemcpy(ptr, source.data(), bytes, cudaMemcpyHostToDevice),
                       "upload canonical EXL3 tensor");
            if(upload_fault_for_test) {
                cleanup_failure_for_test=upload_fault_for_test==2;
                device_query_failure_for_test=upload_fault_for_test==3;
                device_mismatch_for_test=upload_fault_for_test==4;
                throw std::runtime_error("injected generic post-upload construction failure");
            }
        } catch (...) {
            release();
            throw;
        }
    }
    explicit DeviceAllocation(std::size_t size, const char* label,
        std::optional<RetainedDeviceLedger::Ticket> device_credit={},
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit={}) : bytes(size) {
        adopt_constructor_credits(std::move(device_credit),std::move(metadata_credit));
        prepare_device();
        cuda_check(cudaMalloc(&ptr, bytes), label);
    }
    ~DeviceAllocation() {release();}
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;
};

struct ReconstructionBacking {
    inline static std::atomic<ReconstructionBacking*> quarantine{nullptr};
    inline static std::atomic<std::size_t> quarantined_allocations{0};
    ReconstructionBacking* quarantine_next=nullptr;
    bool cleanup_failure_for_test=false;
    bool control_allocation_admitted=false;
    static constexpr std::size_t control_bytes=Exl3ReconstructionControlAllocator<std::byte>::capacity;
    const int device=[] {
        int value=-1;cuda_check(cudaGetDevice(&value),"reconstruction allocation device");return value;
    }();
    DeviceAllocation allocation;
    Exl3ReconstructedExactStats stats;
    Exl3ReconstructionStream stream;
    explicit ReconstructionBacking(std::size_t bytes)
        : allocation(bytes,"allocate shared exact reconstruction slab") {
        stats.workspace_bytes=bytes;
    }
    static std::shared_ptr<ReconstructionBacking> create(std::size_t bytes) {
        require(quarantined_allocations.load(std::memory_order_acquire)==0,
            "unresolved reconstruction allocation cleanup; allocation refused");
        auto* backing=new ReconstructionBacking(bytes);
        return std::shared_ptr<ReconstructionBacking>(backing,
            [](ReconstructionBacking* value) noexcept {
                const auto released=exl3_retire_reconstruction_device(value->device,
                    [](int* device) noexcept {return static_cast<int>(cudaGetDevice(device));},
                    [](int device) noexcept {return static_cast<int>(cudaSetDevice(device));},
                    [&]() noexcept {return value->cleanup_failure_for_test ? static_cast<int>(cudaErrorUnknown) :
                        static_cast<int>(cudaFree(value->allocation.ptr));},
                    [&]() noexcept {value->allocation.ptr=nullptr;delete value;});
                if(released.released)return;
                value->stream.fail();
                auto* head=quarantine.load(std::memory_order_relaxed);
                do{value->quarantine_next=head;}
                while(!quarantine.compare_exchange_weak(head,value,std::memory_order_release,std::memory_order_relaxed));
                quarantined_allocations.fetch_add(1,std::memory_order_release);
            },Exl3ReconstructionControlAllocator<std::byte>(&backing->control_allocation_admitted));
    }
};
std::atomic<std::size_t> reconstruction_quarantined_contexts{0};

class ParallelTensorStager {
    static constexpr std::size_t kBytes = 16u * 1024u * 1024u;
    struct Slot {
        std::byte* data = nullptr;
        std::string name;
        std::uint64_t offset = 0;
        std::size_t bytes = 0;
        bool pending = false;
        bool ready = false;
        std::exception_ptr error;
    };
    const VerifiedDualTensorFiles& files_;
    std::array<Slot, 2> slots_;
    std::array<std::thread, 2> workers_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    void shutdown() noexcept {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        cv_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
        for (auto& slot : slots_) if (slot.data) { cudaFreeHost(slot.data); slot.data = nullptr; }
    }
    void worker(int source) {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            auto& slot = slots_[source];
            cv_.wait(lock, [&] { return stop_ || slot.pending; });
            if (stop_) return;
            const auto name = slot.name;
            const auto offset = slot.offset;
            const auto bytes = slot.bytes;
            slot.pending = false;
            lock.unlock();
            std::exception_ptr error;
            try { files_.read_into(name, offset, {slot.data, bytes}, source); }
            catch (...) { error = std::current_exception(); }
            lock.lock();
            slot.error = error;
            slot.ready = true;
            lock.unlock();
            cv_.notify_all();
        }
    }
public:
    explicit ParallelTensorStager(const VerifiedDualTensorFiles& files) : files_(files) {
        try {
            for (auto& slot : slots_)
                cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.data), kBytes), "allocate bounded loader staging");
            for (int source = 0; source < 2; ++source)
                workers_[source] = std::thread([this, source] { worker(source); });
        } catch (...) { shutdown(); throw; }
    }
    ~ParallelTensorStager() { shutdown(); }
    static constexpr std::size_t host_bytes() { return 2 * kBytes; }
    void upload(const TensorInfo& tensor, void* destination, std::span<const std::byte> audit = {}) {
        require(audit.empty() || audit.size() == tensor.bytes(), "loader audit size mismatch");
        std::uint64_t next = 0;
        auto assign = [&](int source) {
            auto& slot = slots_[source];
            slot.name = tensor.name;
            slot.offset = next;
            slot.bytes = static_cast<std::size_t>(std::min<std::uint64_t>(kBytes, tensor.bytes() - next));
            next += slot.bytes;
            slot.error = {};
            slot.pending = true;
        };
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int source = 0; source < 2; ++source) {
                require(!slots_[source].pending && !slots_[source].ready, "loader staging slot is not idle");
                if (next < tensor.bytes()) assign(source);
            }
        }
        cv_.notify_all();
        std::uint64_t copied = 0;
        while (copied < tensor.bytes()) {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return slots_[0].ready || slots_[1].ready; });
            const int source = slots_[0].ready ? 0 : 1;
            auto& slot = slots_[source];
            const auto offset = slot.offset;
            const auto bytes = slot.bytes;
            const auto error = slot.error;
            slot.ready = false;
            lock.unlock();
            if (error) std::rethrow_exception(error);
            if (!audit.empty()) require(std::memcmp(slot.data, audit.data() + offset, bytes) == 0,
                                        "parallel loader chunk differs from original reader: " + tensor.name);
            // All CUDA work remains on the loading thread. A slot cannot be reused
            // until this synchronous upload completes; its peer can read meanwhile.
            cuda_check(cudaMemcpy(static_cast<std::byte*>(destination) + offset, slot.data, bytes,
                                  cudaMemcpyHostToDevice), "upload parallel loader chunk");
            copied += bytes;
            lock.lock();
            if (next < tensor.bytes()) assign(source);
            lock.unlock();
            cv_.notify_all();
        }
    }
};

std::span<const std::byte> as_bytes(std::span<const std::uint16_t> values) {
    return {reinterpret_cast<const std::byte*>(values.data()), values.size_bytes()};
}

std::span<const std::byte> as_bytes(std::span<const float> values) {
    return {reinterpret_cast<const std::byte*>(values.data()), values.size_bytes()};
}

__global__ void embedding_lookup_kernel(const std::int64_t* ids,
                                         const std::uint16_t* bf16_weight,
                                         std::uint16_t* output,
                                         int rows) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * kHidden;
    if (index >= total) return;
    const int row = index / kHidden;
    const int column = index % kHidden;
    const std::int64_t token = ids[row];
    output[index] = float_to_half(bf16_to_float(bf16_weight[token * kHidden + column]));
}

// The graph captures only the stable record address. Request-visible pointer
// roles are published as one stream-ordered generation before replay.
__global__ void graph_role_embedding_kernel(
        const Exl3DeviceGraphRoleRecord* roles,
        const std::uint16_t* bf16_weight,int captured_rows) {
    const int index=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;
    const int total=captured_rows*kHidden;
    if(index>=total)return;
    const int row=index/kHidden;
    const int column=index%kHidden;
    const std::int64_t token=roles->token_ids[row];
    const auto value=float_to_half(bf16_to_float(
        bf16_weight[token*kHidden+column]));
    roles->hidden_output[index]=value;
    roles->embedding_trace[index]=value;
}

__global__ void media_embedding_cast_kernel(const float* input,std::uint16_t* output,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=float_to_half(input[i]);
}

__global__ void final_rms_norm_kernel(const std::uint16_t* input,
                                      const std::uint16_t* weight,
                                      std::uint16_t* output) {
    __shared__ float partial[512];
    const int lane = static_cast<int>(threadIdx.x);
    float sum = 0.0f;
    for (int i = lane; i < kHidden; i += blockDim.x) {
        const float value = half_to_float(input[i]);
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(partial[0] / static_cast<float>(kHidden) + 1.0e-6f);
    for (int i = lane; i < kHidden; i += blockDim.x) {
        const float value = half_to_float(input[i]) * inv * (half_to_float(weight[i]) + 1.0f);
        output[i] = float_to_half(value);
    }
}

} // namespace

struct Exl3GreedyPacketTransfer {
    struct Retained {
        std::unique_ptr<DeviceAllocation> device;
        std::shared_ptr<const void> model_owner,stream_owner;
        Exl3GreedyRow* host=nullptr;
        cudaEvent_t event=nullptr,device_event=nullptr,consumer_event=nullptr;
        int first_error=0;
        std::optional<RetainedCudaRegistrationLedger::Ticket> registration_credit;
        std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;
        Retained* next=nullptr;
    };
    inline static std::atomic<Retained*> quarantine{nullptr};
    inline static std::atomic<std::uint64_t> quarantined{0};
    std::unique_ptr<DeviceAllocation> device;
    std::unique_ptr<Retained> retirement=std::make_unique<Retained>();
    std::shared_ptr<const void> model_owner,stream_owner;
    Exl3GreedyRow* host=nullptr;
    cudaEvent_t event=nullptr,device_event=nullptr,consumer_event=nullptr;
    Exl3FinalUseWitness final_use;
    std::uint64_t completion_generation=0;
    cudaStream_t stream=nullptr;
    cudaStream_t consumer_stream=nullptr;
    std::atomic<bool> consumer_claimed{false},consumer_recorded{false};
    bool submitted=false,event_recorded=false,host_readback_submitted=false,completed=false;
    bool completion_failure_for_test=false;

    static constexpr std::size_t row_bytes=16*sizeof(Exl3GreedyRow);
    static constexpr std::size_t metadata_bytes_required() noexcept {
        return bounded_shared_allocation_bytes<Exl3GreedyPacketTransfer>()+
            DeviceAllocation::owner_metadata_bytes();
    }
    std::optional<RetainedCudaRegistrationLedger::Ticket> registration_credit;
    std::optional<RetainedDescriptorLedger::Ticket> metadata_credit;

    static bool attach_device_credit(const std::shared_ptr<const void>& owner,
        RetainedDeviceLedger::Ticket credit) noexcept {
        auto* value=const_cast<Exl3GreedyPacketTransfer*>(
            static_cast<const Exl3GreedyPacketTransfer*>(owner.get()));
        return value && value->device &&
            DeviceAllocation::attach_device_credit(
                std::shared_ptr<const void>(owner,value->device.get()),std::move(credit));
    }
    static bool attach_registration_credit(const std::shared_ptr<const void>& owner,
        RetainedCudaRegistrationLedger::Ticket credit) noexcept {
        auto* value=const_cast<Exl3GreedyPacketTransfer*>(
            static_cast<const Exl3GreedyPacketTransfer*>(owner.get()));
        if(!value || !value->host || value->registration_credit ||
           credit.bytes()!=row_bytes)return false;
        value->registration_credit.emplace(std::move(credit));return true;
    }
    static bool attach_metadata_credit(const std::shared_ptr<const void>& owner,
        RetainedDescriptorLedger::Ticket credit) noexcept {
        auto* value=const_cast<Exl3GreedyPacketTransfer*>(
            static_cast<const Exl3GreedyPacketTransfer*>(owner.get()));
        if(!value || value->metadata_credit ||
           credit.bytes()!=metadata_bytes_required())return false;
        value->metadata_credit.emplace(std::move(credit));return true;
    }

    Exl3GreedyPacketTransfer()
        :device(std::make_unique<DeviceAllocation>(
            row_bytes,"allocate pending greedy packet")) {
        try {
            cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&host),
                row_bytes,cudaHostAllocPortable),
                "allocate pending greedy packet host destination");
            cuda_check(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),
                "allocate pending greedy packet event");
            cuda_check(cudaEventCreateWithFlags(&device_event,cudaEventDisableTiming),
                "allocate pending greedy packet device-ready event");
            cuda_check(cudaEventCreateWithFlags(&consumer_event,cudaEventDisableTiming),
                "allocate pending greedy packet consumer event");
        } catch(...) {
            if(consumer_event)cudaEventDestroy(consumer_event);
            if(device_event)cudaEventDestroy(device_event);
            if(event)cudaEventDestroy(event);
            if(host)cudaFreeHost(host);
            throw;
        }
    }
    int wait() noexcept {
        if(completed)return final_use.first_error();
        cudaError_t error=cudaSuccess;
        if(completion_failure_for_test) {
            error=cudaErrorUnknown;
            final_use.finish(completion_generation,static_cast<int>(error));
        }
        else if(submitted && consumer_claimed.load(std::memory_order_acquire)) {
            // Once claimed, only the recorded consumer event can prove final
            // use. A stream address or successful producer wait is not a
            // substitute for the missing final event.
            error=consumer_recorded.load(std::memory_order_acquire)?
                cudaEventSynchronize(consumer_event):cudaErrorUnknown;
            final_use.finish(completion_generation,static_cast<int>(error));
        } else if(submitted && event_recorded) {
            error=cudaEventSynchronize(event);
            final_use.finish(completion_generation,static_cast<int>(error));
        } else if(submitted) {
            // An unclaimed deferred consumer has no consumer event. Witness the
            // exact producer event before abandoning the planned handoff.
            error=cudaEventSynchronize(device_event);
            final_use.finish_after_alternate_event(completion_generation,
                reinterpret_cast<std::uintptr_t>(device_event),
                static_cast<int>(error));
        }
        completed=true;
        return static_cast<int>(error);
    }
    int recycle() noexcept {
        int error=submitted?wait():0;
        if(error)return error;
        stream=nullptr;consumer_stream=nullptr;submitted=false;event_recorded=false;
        host_readback_submitted=false;completed=false;completion_failure_for_test=false;
        completion_generation=0;
        model_owner.reset();stream_owner.reset();
        consumer_claimed.store(false,std::memory_order_release);
        consumer_recorded.store(false,std::memory_order_release);
        return 0;
    }
    ~Exl3GreedyPacketTransfer() {
        auto error=wait();
        if(!error && consumer_event)
            error=static_cast<int>(cudaEventDestroy(consumer_event));
        if(!error)consumer_event=nullptr;
        if(!error && device_event)
            error=static_cast<int>(cudaEventDestroy(device_event));
        if(!error)device_event=nullptr;
        if(!error && event)error=static_cast<int>(cudaEventDestroy(event));
        if(!error){event=nullptr;if(host)error=static_cast<int>(cudaFreeHost(host));}
        if(!error){host=nullptr;registration_credit.reset();metadata_credit.reset();return;}
        auto* retained=retirement.release();
        retained->device=std::move(device);retained->host=host;
        retained->model_owner=std::move(model_owner);
        retained->stream_owner=std::move(stream_owner);
        retained->event=event;retained->device_event=device_event;
        retained->consumer_event=consumer_event;
        retained->first_error=error;
        if(registration_credit)
            retained->registration_credit.emplace(std::move(*registration_credit));
        if(metadata_credit)
            retained->metadata_credit.emplace(std::move(*metadata_credit));
        auto* head=quarantine.load(std::memory_order_relaxed);
        do{retained->next=head;}while(!quarantine.compare_exchange_weak(
            head,retained,std::memory_order_release,std::memory_order_relaxed));
        quarantined.fetch_add(1,std::memory_order_release);
        host=nullptr;event=nullptr;device_event=nullptr;consumer_event=nullptr;
    }
};

struct Exl3TextModel::Impl {
    // Lifetime-stable process identity for exact host images; prevents address
    // reuse from making an old model's checkpoint compatible with a new model.
    std::shared_ptr<const int> host_state_identity = std::make_shared<const int>(0);
    struct LayerBinding {
        bool full_attention = false;
        Exl3FullAttentionLayerWeights full{};
        Exl3GdnLayerWeights gdn{};
    };

    std::filesystem::path directory;
    Exl3LoadOptions load_options;
    IndexedSafetensors collection;
    std::unique_ptr<VerifiedDualTensorFiles> dual_files;
    std::unique_ptr<ParallelTensorStager> tensor_stager;
    Exl3ModelLoadStats load_stats;
    bool audit_parallel = false;
    std::vector<std::unique_ptr<DeviceAllocation>> allocations;
    std::array<LayerBinding, kLayers> layers{};
    std::array<std::optional<Exl3GdnImmutableCoefficients>,kLayers>
        gdn_coefficients{};
    Exl3CudaLinearWeights lm_head{};
    Exl3CudaLinearMetadata lm_head_metadata{};
    const std::uint16_t* embedding = nullptr;
    const std::uint16_t* final_norm = nullptr;
    std::size_t model_bytes = 0;

    TensorPayload load_tensor(const IndexedSafetensors& input, const std::string& name) {
        const auto start = std::chrono::steady_clock::now();
        auto tensor = load_tensor_single(input, name);
        if (audit_parallel) {
            const auto reference = load_tensor_single(input, name);
            require(tensor.bytes().size() == reference.bytes().size() &&
                    std::equal(tensor.bytes().begin(), tensor.bytes().end(), reference.bytes().begin()),
                    "loaded tensor differs from canonical single reader: " + name);
            ++load_stats.audited_tensors;
            load_stats.audited_bytes += tensor.bytes().size();
        }
        load_stats.payload_read_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return tensor;
    }

    void* upload(std::span<const std::byte> bytes, const char* label) {
        const auto start = std::chrono::steady_clock::now();
        void* pointer=nullptr;
        Exl3LinearWorkspaceRequirements::allocate_owned(allocations,model_bytes,&pointer,bytes.size_bytes(),
            [&]{return std::make_unique<DeviceAllocation>(bytes,label);});
        load_stats.upload_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return pointer;
    }

    void* load_raw_tensor(const std::string& name, const std::string& label, std::string_view dtype) {
        const TensorInfo* info = nullptr;
        for (const auto& shard : collection.shards) if ((info = shard.header.find(name)) != nullptr) break;
        require(info != nullptr, name + " raw tensor metadata absent");
        return load_raw_tensor(Exl3BorrowedDescriptor<TensorInfo>(&collection,*info),label,dtype);
    }

    // Descriptor is borrowed from this model's retained immutable collection.
    // Loading consumes it synchronously; it never outlives the strong model owner.
    void* load_raw_tensor(const Exl3BorrowedDescriptor<TensorInfo>& descriptor, const std::string& label, std::string_view dtype) {
        const TensorInfo* info=&descriptor.get(&collection);
        const auto& name=info->name;
        require(info->dtype == dtype, name + " raw tensor metadata mismatch");
        if (!tensor_stager || info->bytes() < 8u * 1024u * 1024u)
            return upload(load_tensor(collection, name).bytes(), label.c_str());
        const auto start = std::chrono::steady_clock::now();
        if(info->bytes()>std::numeric_limits<std::size_t>::max())
            throw std::overflow_error("staged tensor allocation extent overflow");
        void* pointer=nullptr;
        Exl3LinearWorkspaceRequirements::allocate_owned(allocations,model_bytes,&pointer,
            static_cast<std::size_t>(info->bytes()),[&]{
                auto allocation=std::make_unique<DeviceAllocation>(static_cast<std::size_t>(info->bytes()),label.c_str());
                if(audit_parallel) {
                    const auto reference=load_tensor_single(collection,name);
                    tensor_stager->upload(*info,allocation->ptr,reference.bytes());
                    ++load_stats.audited_tensors;load_stats.audited_bytes+=info->bytes();
                } else tensor_stager->upload(*info,allocation->ptr);
                return allocation;
            });
        load_stats.staged_transfer_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return pointer;
    }

    const std::uint16_t* load_f16(const TensorPayload& tensor, const std::string& label) {
        require(tensor.info.dtype == "F16" || tensor.info.dtype == "BF16",
                tensor.info.name + " must be F16 or BF16");
        if (tensor.info.dtype == "F16") {
            return static_cast<const std::uint16_t*>(upload(tensor.bytes(), label.c_str()));
        }
        const auto source = tensor.typed<std::uint16_t>("BF16");
        std::vector<std::uint16_t> converted;
        converted.reserve(source.size());
        for (const auto value : source) converted.push_back(float_to_half(bf16_to_float(value)));
        return static_cast<const std::uint16_t*>(upload(as_bytes(converted), label.c_str()));
    }

    const std::uint16_t* load_raw16(const TensorPayload& tensor, const std::string& label,
                                    std::string_view dtype) {
        require(tensor.info.dtype == dtype, tensor.info.name + " has unexpected dtype");
        return static_cast<const std::uint16_t*>(upload(tensor.bytes(), label.c_str()));
    }

    const float* load_f32(const TensorPayload& tensor, const std::string& label) {
        std::vector<float> converted;
        if (tensor.info.dtype == "F32") {
            const auto source = tensor.typed<float>("F32");
            converted.assign(source.begin(), source.end());
        } else if (tensor.info.dtype == "BF16") {
            const auto source = tensor.typed<std::uint16_t>("BF16");
            converted.reserve(source.size());
            for (const auto value : source) converted.push_back(bf16_to_float(value));
        } else {
            throw std::runtime_error(tensor.info.name + " must be F32 or BF16");
        }
        return static_cast<const float*>(upload(as_bytes(converted), label.c_str()));
    }

    const TensorInfo& tensor_info(const std::string& name) const {
        for (const auto& shard : collection.shards) if (const auto* info = shard.header.find(name)) return *info;
        throw std::runtime_error("canonical EXL3 tensor is absent: " + name);
    }

    Exl3CudaLinearWeights load_linear(const std::string& prefix, int in_features,
                                      int out_features, Exl3CudaLinearMetadata& metadata) {
        const auto& trellis = tensor_info(prefix + ".trellis");
        const auto& suh = tensor_info(prefix + ".suh");
        const auto& svh = tensor_info(prefix + ".svh");
        const auto& mul1 = tensor_info(prefix + ".mul1");
        require(trellis.dtype == "I16" && trellis.shape.size() == 3,
                prefix + " trellis metadata mismatch");
        require(trellis.shape[0] == static_cast<std::uint64_t>(in_features / 16) &&
                trellis.shape[1] == static_cast<std::uint64_t>(out_features / 16) &&
                trellis.shape[2] % 16 == 0, prefix + " EXL3 dimensions mismatch");
        require(suh.dtype == "F16" && suh.shape.size()==1 && suh.shape[0]==static_cast<std::uint64_t>(in_features),
                prefix + " suh metadata mismatch");
        require(svh.dtype == "F16" && svh.shape.size()==1 && svh.shape[0]==static_cast<std::uint64_t>(out_features),
                prefix + " svh metadata mismatch");
        require(mul1.dtype == "I32" && mul1.shape.empty(), prefix + " mul1 metadata mismatch");
        const int bits = exl3_native_bits_from_tile_extent(trellis.shape[2]);
        Exl3CudaLinearMetadata result{in_features, out_features, bits, false, true, false};
        metadata = result;
        const Exl3CudaLinearWeights loaded{
            static_cast<const std::uint16_t*>(load_raw_tensor(Exl3BorrowedDescriptor<TensorInfo>(&collection,trellis), prefix + " trellis", "I16")),
            static_cast<const std::uint16_t*>(load_raw_tensor(Exl3BorrowedDescriptor<TensorInfo>(&collection,suh), prefix + " suh", "F16")),
            static_cast<const std::uint16_t*>(load_raw_tensor(Exl3BorrowedDescriptor<TensorInfo>(&collection,svh), prefix + " svh", "F16")),
            static_cast<const std::int32_t*>(load_raw_tensor(Exl3BorrowedDescriptor<TensorInfo>(&collection,mul1), prefix + " mul1", "I32"))};
        load_stats.linear_descriptor_reuses+=4;
        return loaded;
    }

    const std::uint16_t* load_norm(const std::string& name, int features) {
        const auto tensor = load_tensor(collection, name);
        require(tensor.info.shape == std::vector<std::uint64_t>{static_cast<std::uint64_t>(features)},
                name + " shape mismatch");
        return load_f16(tensor, name);
    }

    void load_from_disk() {
        const auto config_path = directory / "config.json";
        std::ifstream config_file(config_path, std::ios::binary);
        require(static_cast<bool>(config_file), "cannot open canonical EXL3 config.json");
        Json config = Json::parse(std::string(std::istreambuf_iterator<char>(config_file), {}));
        require(config.at("architectures").at(0).get<std::string>() == "Qwen3_5ForConditionalGeneration",
                "E4A requires Qwen3_5ForConditionalGeneration");
        const auto& text = config.at("text_config");
        require(text.at("model_type").get<std::string>() == "qwen3_5_text" &&
                text.at("hidden_size").get<int>() == kHidden &&
                text.at("num_hidden_layers").get<int>() == kLayers &&
                text.at("intermediate_size").get<int>() == kIntermediate &&
                text.at("vocab_size").get<int>() == kVocab &&
                text.at("head_dim").get<int>() == kHeadDim,
                "canonical EXL3 text dimensions do not match E4A");
        require(config.at("quantization_config").at("quant_method").get<std::string>() == "exl3" &&
                config.at("quantization_config").at("head_bits").get<int>() == 6 &&
                config.at("quantization_config").at("codebook").get<std::string>() == "mul1" &&
                config.at("quantization_config").at("out_scales").get<std::string>() == "always",
                "unsupported canonical EXL3 quantization metadata");
        const auto layer_types = text.at("layer_types").get<std::vector<std::string>>();
        require(layer_types.size() == kLayers, "canonical EXL3 layer schedule is not 64 layers");
        int full_count = 0;
        for (int i = 0; i < kLayers; ++i) {
            const bool full = layer_types[i] == "full_attention";
            const bool gdn = layer_types[i] == "linear_attention";
            require(full || gdn, "unsupported Qwen3.5 layer type at index " + std::to_string(i));
            require(full == (i % 4 == 3), "canonical Qwen3.5 full-attention schedule mismatch at layer " + std::to_string(i));
            layers[i].full_attention = full;
            full_count += full ? 1 : 0;
        }
        require(full_count == 16, "canonical EXL3 full-attention layer count mismatch");

        collection = inspect_indexed_directory(directory);
        const char* mirror_override = std::getenv("NINFER_EXL3_DUAL_READ_MANIFEST");
        const char* dual_mode = std::getenv("NINFER_EXL3_DUAL_LOAD");
        require(!dual_mode || !*dual_mode || std::strcmp(dual_mode, "0") == 0 ||
                std::strcmp(dual_mode, "1") == 0, "NINFER_EXL3_DUAL_LOAD must be0 or1");
        const bool disabled = load_options.disable_dual_artifact_loading ||
                              (dual_mode && std::strcmp(dual_mode, "0") == 0);
        std::filesystem::path mirror;
        if (!disabled) {
            mirror = load_options.verified_dual_manifest;
            if (mirror.empty() && mirror_override && *mirror_override) mirror = mirror_override;
            if (mirror.empty()) mirror = find_verified_dual_manifest(directory);
            require(!(dual_mode && std::strcmp(dual_mode, "1") == 0) || !mirror.empty(),
                    "forced dual loading requires a verified mirror manifest");
        }
        const char* audit = std::getenv("NINFER_EXL3_DUAL_READ_AUDIT");
        audit_parallel = audit && std::strcmp(audit, "1") == 0;
        require(!audit_parallel || !mirror.empty(), "parallel audit requires a verified mirror manifest");
        if (!mirror.empty()) {
            const auto start = std::chrono::steady_clock::now();
            dual_files = std::make_unique<VerifiedDualTensorFiles>(collection, mirror);
            tensor_stager = std::make_unique<ParallelTensorStager>(*dual_files);
            load_stats.staging_host_bytes = ParallelTensorStager::host_bytes();
            load_stats.staging_prepare_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            load_stats.parallel_staged = true;
        }
        embedding = static_cast<const std::uint16_t*>(load_raw_tensor(
            "model.language_model.embed_tokens.weight", "embedding", "BF16"));
        final_norm = load_norm("model.language_model.norm.weight", kHidden);

        for (int i = 0; i < kLayers; ++i) {
            const std::string base = "model.language_model.layers." + std::to_string(i);
            if (layers[i].full_attention) {
                auto& w = layers[i].full;
                w.q = load_linear(base + ".self_attn.q_proj", kHidden, 12288, w.q_metadata);
                w.k = load_linear(base + ".self_attn.k_proj", kHidden, 1024, w.k_metadata);
                w.v = load_linear(base + ".self_attn.v_proj", kHidden, 1024, w.v_metadata);
                w.o = load_linear(base + ".self_attn.o_proj", 6144, kHidden, w.o_metadata);
                w.gate = load_linear(base + ".mlp.gate_proj", kHidden, kIntermediate, w.gate_metadata);
                w.up = load_linear(base + ".mlp.up_proj", kHidden, kIntermediate, w.up_metadata);
                w.down = load_linear(base + ".mlp.down_proj", kIntermediate, kHidden, w.down_metadata);
                w.input_norm = load_norm(base + ".input_layernorm.weight", kHidden);
                w.q_norm = load_norm(base + ".self_attn.q_norm.weight", kHeadDim);
                w.k_norm = load_norm(base + ".self_attn.k_norm.weight", kHeadDim);
                w.post_attention_norm = load_norm(base + ".post_attention_layernorm.weight", kHidden);
            } else {
                auto& w = layers[i].gdn;
                w.qkv = load_linear(base + ".linear_attn.in_proj_qkv", kHidden, kGdnQkv, w.qkv_metadata);
                w.z = load_linear(base + ".linear_attn.in_proj_z", kHidden, kGdnZ, w.z_metadata);
                w.o = load_linear(base + ".linear_attn.out_proj", kGdnZ, kHidden, w.o_metadata);
                w.gate = load_linear(base + ".mlp.gate_proj", kHidden, kIntermediate, w.gate_metadata);
                w.up = load_linear(base + ".mlp.up_proj", kHidden, kIntermediate, w.up_metadata);
                w.down = load_linear(base + ".mlp.down_proj", kIntermediate, kHidden, w.down_metadata);
                w.input_norm = load_norm(base + ".input_layernorm.weight", kHidden);
                // The gated RMSNorm launcher consumes the canonical BF16 weight
                // representation.  Do not round it through the F16 loader.
                const auto gdn_norm = load_tensor(collection, base + ".linear_attn.norm.weight");
                require(gdn_norm.info.dtype == "BF16" &&
                        gdn_norm.info.shape == std::vector<std::uint64_t>{128},
                        base + " GDN norm metadata mismatch");
                w.gdn_norm = load_raw16(gdn_norm, base + " GDN norm", "BF16");
                w.post_attention_norm = load_norm(base + ".post_attention_layernorm.weight", kHidden);
                const auto conv = load_tensor(collection, base + ".linear_attn.conv1d.weight");
                require(conv.info.dtype == "BF16" && conv.info.shape == std::vector<std::uint64_t>{10240, 1, 4},
                        base + " convolution metadata mismatch");
                w.conv_weight = load_raw16(conv, base + " convolution", "BF16");
                w.a_weight = load_f16(load_tensor(collection, base + ".linear_attn.in_proj_a.weight"), base + " a weight");
                w.b_weight = load_f16(load_tensor(collection, base + ".linear_attn.in_proj_b.weight"), base + " b weight");
                w.a_log = load_f32(load_tensor(collection, base + ".linear_attn.A_log"), base + " A_log");
                w.dt_bias = load_f32(load_tensor(collection, base + ".linear_attn.dt_bias"), base + " dt_bias");
                gdn_coefficients[i].emplace(Exl3GdnImmutableCoefficients{
                    host_state_identity,i,w});
            }
        }
        lm_head = load_linear("lm_head", kHidden, kVocab, lm_head_metadata);
        require(lm_head_metadata.K == 6, "E4A requires the canonical H6 LM head");
        // Every upload above is synchronous; no device operation retains host views.
        const auto release_start = std::chrono::steady_clock::now();
        tensor_stager.reset();
        if (dual_files) for (int source = 0; source < 2; ++source)
            load_stats.source_bytes[source] = dual_files->source_bytes(source);
        dual_files.reset();
        load_stats.staging_release_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - release_start).count();
    }
};

struct Exl3TextContext::Impl {
    Exl3GamingOptions gaming = Exl3GamingOptions::from_environment();
    GoptSubmissions gaming_submissions{};
    // Declared first so it outlives every device/graph member during teardown.
    std::shared_ptr<const void> execution_stream_owner;
    cudaStream_t owned_execution_stream=nullptr;
    struct Transaction {
        std::unique_ptr<DeviceAllocation> gdn_recurrent;
        std::unique_ptr<DeviceAllocation> gdn_conv;
        std::unique_ptr<DeviceAllocation> oscar_cache;
        std::unique_ptr<DeviceAllocation> logits;
        std::unique_ptr<DeviceAllocation> taps;
        std::unique_ptr<DeviceAllocation> embedding;
        std::unique_ptr<DeviceAllocation> position_device;
        std::array<Exl3GdnLayerCheckpoint, kLayers> gdn_checkpoints{};
        Exl3OscarCheckpoint oscar_checkpoint{};
        std::size_t bytes = 0;
        int position = 0;
        int tap_rows = 0;
        int embedding_rows = 0;
        int last_rows = 0;
        cudaStream_t stream = nullptr;
        bool fresh_snapshot = false;
        bool prefix_available = false;
        bool rollback_required = false;
        int attempt_base_position = 0;
        int attempted_rows = 0;
        int continuation_graph_rows = 0;
        bool continuation_graph_attempt = false;
        DecodeGraphDefinition checkpoint_graph_definition;
        DecodeGraphExecutable checkpoint_graph_executable;
        cudaStream_t checkpoint_graph_stream = nullptr;
        bool checkpoint_graph_active = false;
        bool active = false;
        bool host_kv = false;
        bool device_kv = false;
        std::size_t host_kv_page_count = 0;
        std::shared_ptr<const Exl3ExactKVPage> host_kv_tail;
    };

    struct Continuation {
        struct AdditionalGraph {
            Exl3ProjectionGraphBinding projection_binding;
            Exl3GraphCompatibilityFingerprint compatibility;
            Exl3BoundedGraphEntry lifecycle;
            Exl3DeviceGraphRoleTable roles;
            int rows = 0;
            int split_class = 0;
            std::uint64_t active_generation = 0;
            std::uint64_t capture_count = 0;
            std::uint64_t compatible_reuse_count = 0;
            cudaStream_t origin_stream = nullptr;
            std::uint32_t route_bits = 0;
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            bool active = false;
            std::string reason = "not captured";
        };
        std::unique_ptr<DeviceAllocation> final_norm;
        std::unique_ptr<DeviceAllocation> logits;
        Exl3CudaLinearWorkspace::Owner head_workspace;
        std::size_t bytes = 0;
        int capacity = 0;
        int rows = 0;
        int graph_split_class = 0;
        Exl3ProjectionGraphBinding projection_binding;
        Exl3GraphCompatibilityFingerprint graph_compatibility;
        Exl3BoundedGraphEntry graph_lifecycle;
        Exl3DeviceGraphRoleTable graph_roles;
        std::uint64_t graph_generation_counter = 0;
        std::uint64_t graph_active_generation = 0;
        std::uint64_t graph_capture_count = 0;
        std::uint64_t graph_compatible_reuse_count = 0;
        cudaStream_t graph_origin_stream = nullptr;
        std::uint32_t graph_route_bits = 0;
        DecodeGraphDefinition graph_definition;
        DecodeGraphExecutable graph_executable;
        std::array<AdditionalGraph, 2> additional_graphs{};
        std::array<Exl3GraphCaptureEntry,3> graph_menu{};
        Exl3ResourceInventory::Totals graph_menu_reservation{};
        std::uint64_t graph_menu_configuration=0;
        bool graph_menu_required=false;
        bool graph_menu_reserved=false;
        std::string graph_menu_reason="not required";
        Exl3ContinuationGraphStats graph_stats;
        bool graph_active = false;
        std::string graph_reason = "not captured";
    };

    // Optional exact-HostKV compute-only graphs. Each entry captures the three
    // consecutive GDN layers between two full-attention layers for one fixed
    // row count. HostKV upload/download, attention, transaction publication,
    // repair, and export remain outside the captured region.
    struct HostKVGdnSegmentGraph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        int rows=0;
        int first_layer=0;
    };
    static constexpr int host_kv_gdn_segment_count=kLayers/4;
    static constexpr int host_kv_gdn_graph_row_shapes=8;
    std::array<HostKVGdnSegmentGraph,
        host_kv_gdn_segment_count*host_kv_gdn_graph_row_shapes>
        host_kv_gdn_segment_graphs{};
    bool host_kv_gdn_segment_graphs_enabled=false;
    // Separately labeled ordinary device-KV experiment. It reuses the same
    // bounded three-GDN-layer graph storage, but never enables itself for the
    // exact HostKV lane or for the frozen reference path.
    bool ordinary_gdn_segment_graphs_enabled=false;
    std::uint64_t host_kv_gdn_segment_graph_captures=0;
    std::uint64_t host_kv_gdn_segment_graph_replays=0;
    double host_kv_gdn_segment_graph_capture_ms=0.0;
    std::uint64_t host_kv_gdn_segment_graph_launch_cpu_ns=0;

    // Optional exact-HostKV per-attention-layer compute graphs. Dynamic HostKV
    // upload/download, ownership, and publication remain in process_rows(); the
    // graph sees only stable layer buffers, cache planes, and position_device.
    struct HostKVFullLayerGraph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        int rows=0;
        int layer=0;
    };
    static constexpr int host_kv_full_layer_count=kLayers/4;
    static constexpr int host_kv_full_graph_row_shapes=8;
    std::array<HostKVFullLayerGraph,
        host_kv_full_layer_count*host_kv_full_graph_row_shapes>
        host_kv_full_layer_graphs{};
    bool host_kv_full_layer_graphs_enabled=false;
    std::uint64_t host_kv_full_layer_graph_captures=0;
    std::uint64_t host_kv_full_layer_graph_replays=0;
    std::uint64_t host_kv_full_layer_graph_six_softmax_triple_captures=0;
    std::uint64_t host_kv_full_layer_graph_k6_stream_reduction_captures=0;
    std::uint64_t host_kv_full_layer_graph_extended_stream_reduction_captures=0;
    std::uint64_t host_kv_full_layer_graph_target_down_k6_async_a_captures=0;
    std::uint64_t host_kv_full_layer_graph_target_k6_small_m_async_a_captures=0;
    std::uint64_t host_kv_full_layer_graph_target_k7_small_m_async_a_captures=0;
    double host_kv_full_layer_graph_capture_ms=0.0;
    std::uint64_t host_kv_full_layer_graph_launch_cpu_ns=0;
    // Separately owned ordinary device-KV candidate.  Only the physical-C1
    // single-row shape is captured: the graph contains the complete full
    // attention layer, including its cache append, attention, output
    // projection, residual, norm, and MLP.  It never aliases HostKV graph
    // storage or the frozen/reference path.
    std::array<HostKVFullLayerGraph,
        host_kv_full_layer_count*host_kv_full_graph_row_shapes>
        ordinary_full_layer_graphs{};
    bool ordinary_full_layer_graphs_enabled=false;
    bool ordinary_full_layer_graph_extended_replay=false;
    // Continuation (verifier) rows 2..8 replay captured full-layer graphs too;
    // the retained-prefix capability is re-armed on the host after replay.
    bool ordinary_full_layer_multirow_graphs_enabled=false;
    // Device-KV retained-prefix GDN repair: one captured graph per
    // (retained, attempted) row pair records every layer's device work.
    struct GdnRepairGraph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        bool ready=false;
        // Checkpoint buffers the graph recorded; a mismatch forces recapture.
        std::array<const void*,kLayers> recorded_checkpoints{};
    };
    std::array<GdnRepairGraph,64> gdn_repair_graphs{};
    // Overlapped retained-prefix GDN repair: the repair graph runs on a side
    // stream forked from the target stream; every later operation on target
    // state joins it first (join_repair / drain_repair).
    cudaStream_t repair_stream=nullptr;
    cudaEvent_t repair_fork=nullptr;
    cudaEvent_t repair_join=nullptr;
    bool repair_pending=false;
    void join_repair(cudaStream_t stream) {
        if(!repair_pending) return;
        cuda_check(cudaStreamWaitEvent(stream,repair_join,0),"join overlapped GDN repair");
        repair_pending=false;
    }
    void drain_repair() {
        if(!repair_pending) return;
        cuda_check(cudaEventSynchronize(repair_join),"drain overlapped GDN repair");
        repair_pending=false;
    }
    int ordinary_full_layer_graph_capture_position=0;
    std::uint64_t ordinary_full_layer_graph_captures=0;
    std::uint64_t ordinary_full_layer_graph_replays=0;
    double ordinary_full_layer_graph_capture_ms=0.0;
    // Bounded adapter for the position-independent post-attention norm/MLP
    // tail. Attention, HostKV transfers, repair and publication stay eager.
    struct HostKVMlpTailGraph {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        int rows=0;
        int layer=0;
    };
    std::array<HostKVMlpTailGraph,
        host_kv_full_layer_count*host_kv_full_graph_row_shapes>
        host_kv_mlp_tail_graphs{};
    bool host_kv_mlp_tail_graphs_enabled=false;
    // Separately labeled ordinary device-KV experiment. It reuses the same
    // bounded per-layer storage and capture body, but never enables itself for
    // exact HostKV or for the frozen reference lane.
    bool ordinary_mlp_tail_graphs_enabled=false;
    std::uint64_t host_kv_mlp_tail_graph_captures=0;
    std::uint64_t host_kv_mlp_tail_graph_replays=0;
    double host_kv_mlp_tail_graph_capture_ms=0.0;
    bool host_kv_transaction_checkpoint_graph_enabled=false;
    bool device_transaction_checkpoint_graph_enabled=false;
    std::uint64_t device_transaction_checkpoint_graph_captures=0;
    std::uint64_t device_transaction_checkpoint_graph_replays=0;
    double device_transaction_checkpoint_graph_capture_ms=0.0;
    bool host_kv_transaction_recurrent_trace_enabled=false;

    std::shared_ptr<const Exl3TextModel::Impl> model;
    int max_context = 0;
    // Proven by successful ordinary restore/export, invalidated by reset.
    // Retain only immutable KV pages, never an ancestor's full recurrent image.
    int exact_prefix_position = 0;
    int rope_offset = 0;
    std::unique_ptr<DeviceAllocation> media_features,media_positions;
    std::vector<std::shared_ptr<const Exl3ExactKVPage>> exact_prefix_pages;
    std::uint64_t resident_exact_state_id = 0;
    Exl3HostKVStats host_kv;
    bool host_kv_batch_sync = false;
    bool host_kv_batch_copy = false;
    Exl3TransferDescriptorStorage<void*> host_kv_batch_destinations;
    Exl3TransferDescriptorUse host_kv_batch_use;
    Exl3TransferDescriptorStorage<const void*> host_kv_batch_sources;
    Exl3TransferDescriptorStorage<std::size_t> host_kv_batch_sizes;
    bool host_kv_pinned_chunks = false;
    bool host_kv_prefill_pinned_batch = false;
    Exl3TextContext::RegisteredKVUploader registered_kv_uploader;
    Exl3TextContext::SharedPageCopier shared_page_copier;
    Exl3TextContext::SharedPageAttention shared_page_attention;
    Exl3TextContext::SharedPageAttentionCompletion shared_page_attention_completion;
    Exl3TextContext::RegisteredKVCompletion registered_kv_completion;
    static constexpr int host_kv_pinned_max_slots=32;
    int host_kv_pinned_slot_count=2;
    std::array<std::uint64_t,host_kv_pinned_max_slots> registered_kv_pending{},
        registered_kv_pending_bytes{};
    bool host_kv_pinned_d2h = false;
    bool host_kv_banked_d2h = false;
    bool oscar_only = false;
    bool host_kv_failed = false;
    bool fast_prefill_failed = false;
    bool gdn_bulk_prefill_enabled = false;
    bool gdn_bulk_mlp_enabled = false;
    bool gdn_bulk_mlp_short_k5_enabled = false;
    bool fast_wmma32_split2_enabled = false;
    int fast_wmma32_split_count = 0;
    int fast_wmma32_split_capacity = 0;
    int gdn_bulk_capacity = 0;
    std::shared_ptr<DeviceAllocation> host_layer_k, host_layer_v;
    std::shared_ptr<Exl3AttentionStageStorage> attention_stage_storage;
    std::shared_ptr<Exl3AttentionStage> attention_stages;
    std::shared_ptr<Exl3KVRegistrationCache> attention_registration_cache;
    std::shared_ptr<Exl3AttentionStageHistory> attention_stage_history;
    std::uint64_t attention_stage_epoch=0;
    bool attention_staging_enabled=true;
    bool coalesce_attention_input_mlp=false;
    unsigned attention_staging_failure_for_test=0;
    std::shared_ptr<Exl3DevicePrefixCache> device_prefix;
    bool device_prefix_shared=false;
    bool device_prefix_metadata_external=false;
    bool segmented_device_prefix=false;
    bool device_prefix_forward_publish=false;
    cudaStream_t host_kv_copy_stream = nullptr;
    int host_kv_copy_device = -1;
    bool host_kv_device_mismatch_for_test = false;
    bool host_kv_device_query_failure_for_test = false;
    bool host_kv_retirement_failure_for_test=false;
    bool host_kv_copy_failure_for_test=false;
    bool host_kv_restore_failure_for_test=false;
    std::function<void(int)> exact_restore_layer_observer_for_test;
    unsigned host_kv_prefill_fault_for_test=0;
    std::shared_ptr<const Exl3ExactHostState> host_kv_restore_source;
    cudaStream_t host_kv_restore_stream=nullptr;
    int host_kv_restore_device=-1;
    cudaStream_t host_kv_forward_stream=nullptr;
    int host_kv_forward_device=-1;
    bool host_kv_forward_used=false;
    bool host_kv_compute_retirement_failure_for_test=false;
    int host_kv_retirement_error=0;
    cudaEvent_t host_kv_h2d_ready = nullptr, host_kv_compute_ready = nullptr;
    static constexpr std::size_t host_kv_pinned_chunk_bytes = 1U<<20;
    static constexpr int host_kv_banked_d2h_rows = 8;
    static constexpr int host_kv_banked_d2h_banks = 16;
    static constexpr std::size_t host_kv_row_bytes = 1024*sizeof(std::uint16_t);
    static constexpr std::size_t host_kv_banked_d2h_bytes =
        host_kv_banked_d2h_banks*2*host_kv_banked_d2h_rows*host_kv_row_bytes;
    struct HostKVPinnedScatterPiece {
        std::shared_ptr<Exl3ExactKVPage> page;
        std::size_t destination_byte_offset = 0;
        std::size_t staging_offset = 0;
        std::size_t bytes = 0;
        int bank = 0;
        bool key_plane = false;
    };
    std::array<std::unique_ptr<PinnedHostBuffer>,host_kv_pinned_max_slots>
        host_kv_pinned_staging;
    std::unique_ptr<PinnedHostBuffer> host_kv_banked_d2h_staging;
    std::array<cudaEvent_t,host_kv_pinned_max_slots> host_kv_pinned_ready{};
    std::array<bool,host_kv_pinned_max_slots> host_kv_pinned_in_flight{};
    std::array<Exl3FinalUseWitness,host_kv_pinned_max_slots>
        host_kv_pinned_final_use{};
    // Each nonempty piece contains at least one complete 2048-byte KV row.
    // Inline slots are included in the initial sizeof(Impl) metadata credit.
    std::array<Exl3ScatterPieceStorage<HostKVPinnedScatterPiece,
        host_kv_pinned_chunk_bytes/(1024*sizeof(std::uint16_t))>,2> host_kv_pinned_scatter;
    Exl3ScatterPieceStorage<HostKVPinnedScatterPiece,
        host_kv_banked_d2h_banks*2*2> host_kv_banked_d2h_scatter;
    bool host_kv_deferred_scatter = false;
    bool host_kv_cpu_profile = false;
    bool host_kv_unique_tail_reuse = false;
    bool capture_taps = true;
    bool continuation_graph_b8_enabled = false;
    bool continuation_graph_gdn_qkvz_concurrent = false;
    cudaStream_t gdn_qkvz_z_stream = nullptr;
    std::array<cudaEvent_t, kLayers> gdn_qkvz_fork{};
    std::array<cudaEvent_t, kLayers> gdn_qkvz_z_done{};
    Exl3CudaLinearWorkspace::Owner gdn_qkvz_z_workspace;
    bool eager_mlp_gateup_concurrent = false;
    cudaStream_t eager_mlp_gateup_stream = nullptr;
    cudaEvent_t eager_mlp_gateup_fork = nullptr;
    cudaEvent_t eager_mlp_gateup_up_done = nullptr;
    Exl3CudaLinearWorkspace::Owner eager_mlp_gateup_up_workspace;
    std::size_t eager_mlp_gateup_private_workspace_bytes = 0;
    int eager_mlp_gateup_layer_count = 0;
    bool prefill_qkv_concurrent = false;
    cudaStream_t prefill_qkv_k_stream = nullptr;
    cudaStream_t prefill_qkv_v_stream = nullptr;
    cudaEvent_t prefill_qkv_fork = nullptr;
    cudaEvent_t prefill_qkv_k_done = nullptr;
    cudaEvent_t prefill_qkv_v_done = nullptr;
    Exl3CudaLinearWorkspace::Owner prefill_qkv_k_workspace;
    Exl3CudaLinearWorkspace::Owner prefill_qkv_v_workspace;
    std::size_t prefill_qkv_private_workspace_bytes = 0;
    int prefill_qkv_layer_count = 0;
    std::unique_ptr<Exl3CudaReconstructGemmWorkspace>
        numeric_prefill_projection_workspace;
    bool fast_same_weights_fp16kv_prefill_enabled = false;
    Exl3TextContext::FastSameWeightsFp16KvPrefillStats
        fast_same_weights_fp16kv_prefill_stats;
    std::shared_ptr<ReconstructionBacking> reconstruction_backing;
    Exl3TextContext::HeadWorkStats head_work;
    std::size_t deferred_reconstruction_bytes=0;
    bool deferred_reconstruction_k6=false;
    bool deferred_reconstruction_k6_gate_up=false;
    bool reconstruction_budget_fallback=false;
    bool reconstruction_allocation_failure_for_test=false;
    bool reconstruction_rollback_cleanup_failure_for_test=false;
    Impl* reconstruction_quarantine_next=nullptr;
    bool reconstruction_retirement_failure_for_test=false;
    std::size_t gdn_qkvz_private_workspace_bytes = 0;
    int gdn_qkvz_layer_count = 0;
    bool wide_prefill_enabled = false;
    int prefill_capacity = kMaxRows;
    bool native_continuation16_enabled = false;
    Exl3FixedAllocationOwners<DeviceAllocation,32> allocations;
    Exl3VeriCacheServingCoordinator* constructor_authority=nullptr;
    std::unique_ptr<DeviceAllocation> make_private_allocation(std::size_t bytes,const char* label) {
        if(!constructor_authority)return std::make_unique<DeviceAllocation>(bytes,label);
        auto credits=constructor_authority->reserve_constructor_credits(bytes,DeviceAllocation::owner_metadata_bytes());
        return std::make_unique<DeviceAllocation>(bytes,label,std::move(credits.device),std::move(credits.metadata));
    }
    void finish_private_constructor_credits() noexcept {
        constructor_authority=nullptr;
        allocations.visit_owners([](auto* child){if(child)child->release_constructor_credits_after_commit();});
        for(const auto& child:taps)if(child)child->release_constructor_credits_after_commit();
        for(auto* child:{embedding_trace.get(),media_features.get(),media_positions.get(),greedy_rows.get()})
            if(child)child->release_constructor_credits_after_commit();
        for(const auto& child:cache_k)if(child)child->release_constructor_credits_after_commit();
        for(const auto& child:cache_v)if(child)child->release_constructor_credits_after_commit();
        if(host_layer_k)host_layer_k->release_constructor_credits_after_commit();
        if(host_layer_v)host_layer_v->release_constructor_credits_after_commit();
        if(head_workspace)head_workspace->release_constructor_credits_after_commit();
        if(gdn_qkvz_z_workspace)gdn_qkvz_z_workspace->release_constructor_credits_after_commit();
        if(eager_mlp_gateup_up_workspace)
            eager_mlp_gateup_up_workspace->release_constructor_credits_after_commit();
        if(prefill_qkv_k_workspace)prefill_qkv_k_workspace->release_constructor_credits_after_commit();
        if(prefill_qkv_v_workspace)prefill_qkv_v_workspace->release_constructor_credits_after_commit();
        for(const auto& layer:full_layers)if(layer)
            for(auto* child:layer->linear_workspace_owners())if(child)child->release_constructor_credits_after_commit();
        for(const auto& layer:full_layers)if(layer)
            for(auto* child:layer->buffer_retirement_owners())if(child)child->release_constructor_credits_after_commit();
        for(const auto& layer:gdn_layers)if(layer)
            for(auto* child:layer->linear_workspace_owners())if(child)child->release_constructor_credits_after_commit();
        for(const auto& layer:gdn_layers)if(layer)
            for(auto* child:layer->buffer_retirement_owners())if(child)child->release_constructor_credits_after_commit();
    }
    std::unique_ptr<Exl3CudaLinearWorkspace> make_base_linear(
        int input,int output,int rows,bool z_owner=false,
        bool gateup_owner=false,bool kv_owner=false,bool q_owner=false,
        bool wide_prefill_owner=false) {
        std::optional<RetainedDeviceLedger::Ticket> device;
        std::optional<RetainedDescriptorLedger::Ticket> metadata;
        if(constructor_authority) {
            const auto plan=Exl3LinearWorkspaceRequirements::derive(input,output,rows);
            auto credits=constructor_authority->reserve_constructor_credits(plan.owned_bytes,Exl3CudaLinearWorkspace::metadata_bytes());
            device.emplace(std::move(credits.device));metadata.emplace(std::move(credits.metadata));
        }
        return std::make_unique<Exl3CudaLinearWorkspace>(input,output,rows,
            false,gateup_owner,false,false,z_owner,
            gateup_owner||wide_prefill_owner,false,
            Exl3CudaAccumulationView{},Exl3CudaTransformView{},
            false,kv_owner,q_owner,std::move(device),std::move(metadata));
    }
    bool target_projection_timing_opt_in = false;
    Exl3TargetQExecutor shared_head_executor;
    std::unique_ptr<Exl3TargetProjectionTiming> target_projection_timing;
    Exl3TargetProjectionObserver target_projection_observer = nullptr;
    void* target_projection_observer_user = nullptr;
    Exl3LayerObserver layer_observer = nullptr;
    void* layer_observer_user = nullptr;
    std::array<std::unique_ptr<Exl3FullAttentionLayer>, kLayers> full_layers{};
    // Continuation-history views may outlive the enclosing context. The view
    // retains its layer directly; every buffer exposed by that view is one of
    // the layer-owned non-transient history buffers.
    std::array<std::shared_ptr<Exl3GdnLayer>, kLayers> gdn_layers{};
    std::array<std::shared_ptr<DeviceAllocation>, kLayers> cache_k{};
    std::array<std::shared_ptr<DeviceAllocation>, kLayers> cache_v{};
    std::array<std::unique_ptr<DeviceAllocation>, kTapLayers.size()> taps{};
    std::uint16_t* hidden_a = nullptr;
    std::uint16_t* hidden_b = nullptr;
    std::uint16_t* gdn_bulk_h = nullptr;
    std::uint16_t* gdn_bulk_qkv = nullptr;
    std::uint16_t* gdn_bulk_z = nullptr;
    std::uint16_t* gdn_mlp_post = nullptr;
    std::uint16_t* gdn_mlp_input = nullptr;
    std::uint16_t* gdn_mlp_gate = nullptr;
    std::uint16_t* gdn_mlp_up = nullptr;
    float* fast_wmma32_split2_output = nullptr;
    float* fast_wmma32_split2_stats = nullptr;
    // Source of the most recent target final-normalized hidden row(s), before
    // the H6 head. It is only a transient source; native MTP callers must use
    // target_hidden_handoff(), which copies it into owned storage. The row
    // metadata is separate from last_rows because root-only head consumers
    // retain only the final row's normalized hidden.
    const std::uint16_t* last_hidden_source = nullptr;
    int last_hidden_source_rows = 0;
    int last_hidden_source_first_row = 0;
    int last_hidden_source_position = 0;
    std::uint64_t last_hidden_generation = 0;
    // Optional native-MTP source prepared before inference.  Unlike
    // last_hidden_source, this storage is position-indexed and survives
    // ping-pong reuse across ordinary chunked forwards.  It is intentionally
    // separate from target KV/authority and is only populated when the
    // caller explicitly arms the capture.
    std::unique_ptr<DeviceAllocation> native_mtp_hidden_capture;
    std::shared_ptr<const void> native_mtp_hidden_capture_model_identity;
    int native_mtp_hidden_capture_capacity = 0;
    int native_mtp_hidden_capture_first_position = -1;
    int native_mtp_hidden_capture_valid_rows = 0;
    std::uint64_t native_mtp_hidden_capture_generation = 0;
    std::uint16_t* final_norm = nullptr;
    std::uint16_t* logits = nullptr;
    Exl3CudaLinearWorkspace::Owner head_workspace;
    std::unique_ptr<DeviceAllocation> greedy_rows;
    std::array<std::shared_ptr<Exl3GreedyPacketTransfer>,2> greedy_transfers;
    std::uint64_t greedy_serial = 0;
    std::array<Exl3GreedyRow,16> greedy_host_rows{};
    bool greedy_readback_failed=false;
    GreedyReadbackStats greedy_readback;
    std::int64_t* token_ids = nullptr;
    std::size_t persistent_bytes = 0;
    int tap_rows = 0;
    int embedding_rows = 0;
    int last_rows = 0;
    std::atomic<std::uint64_t> tap_generation{0};
    bool qkv_trace_valid = false;
    std::string request_compatibility_contract;
    std::uint64_t request_generation = 0;
    bool fail_request_reset_completion_for_test=false;
    std::unique_ptr<DeviceAllocation> embedding_trace;
    int* position_device = nullptr;
    std::int64_t* draft_token_id = nullptr;
    std::unique_ptr<Exl3OscarContext> oscar;
    Exl3OscarTelemetry oscar_telemetry{};
    DecodeGraphDefinition graph_definition;
    DecodeGraphExecutable graph_executable;
    bool graph_active = false;
    bool graph_capture_active = false;
    std::string graph_reason = "not captured";
    std::shared_ptr<Transaction> transaction;
    std::shared_ptr<Continuation> continuation;
    Exl3RecurrentExportPool recurrent_export;
    unsigned export_copy_fault_for_test=0;
    Exl3TextContext::SnapshotMetadataReservation snapshot_metadata_reservation;
    Exl3TextContext::SnapshotMetadataReservation request_metadata_reservation;
    Exl3TextContext::RequestHostPayloadReservation request_host_payload_reservation;
    Exl3TextContext::RequestHostPayloadObserver request_host_payload_observer;
    bool pinned_recurrent_export = []{const auto* p=std::getenv("NINFER_EXL3_PINNED_RECURRENT_EXPORT");return p && std::string_view(p)=="1";}();
    bool batched_recurrent_export = false;

    static constexpr std::size_t continuation_graph_role_record_count=3;
    static constexpr std::size_t continuation_graph_role_bytes(int capacity) noexcept {
        return capacity==8
            ? continuation_graph_role_record_count*sizeof(Exl3DeviceGraphRoleRecord)
            : 0;
    }
    Exl3DeviceGraphRoleTable* continuation_graph_roles(unsigned rows) noexcept {
        if(!continuation || continuation->capacity!=8)return nullptr;
        if(rows==8)return &continuation->graph_roles;
        if(rows==4 || rows==6)
            return &continuation->additional_graphs[rows==4?0:1].roles;
        return nullptr;
    }
    const Exl3DeviceGraphRoleTable* continuation_graph_roles(
            unsigned rows) const noexcept {
        return const_cast<Impl*>(this)->continuation_graph_roles(rows);
    }
    const Exl3GraphCaptureExtent* continuation_graph_extent(
            unsigned rows) const noexcept {
        if(!continuation)return nullptr;
        for(const auto& entry:continuation->graph_menu)
            if(entry.rows==rows && entry.extent.known)return &entry.extent;
        return nullptr;
    }
    bool continuation_graph_menu_admits(unsigned rows) const noexcept {
        const auto* extent=continuation_graph_extent(rows);
        if(!extent)return false;
        if(!continuation->graph_menu_required)return true;
        if(!continuation->graph_menu_reserved ||
           !continuation->graph_menu_configuration)return false;
        const auto graphs=static_cast<unsigned>(
            Exl3ResourceInventory::Domain::graph_count);
        return continuation->graph_menu_reservation[graphs]>=7 &&
            extent->retained[graphs]==2 && extent->temporary[graphs]==1;
    }
    Exl3ContinuationGraphStats continuation_graph_stats_snapshot() const noexcept {
        if(!continuation)return {};
        auto result=continuation->graph_stats;
        result.live_definitions=0;
        result.live_executables=0;
        result.bound_resource_owners_by_shape={};
        result.bound_external_bytes_by_shape={};
        result.retained_units_by_shape={};
        result.quarantined_entries=0;
        result.reserved_peak_units=continuation->graph_menu_reservation;
        result.driver_memory_bytes_known=false;
        result.driver_memory_bytes=0;
        const auto add=[&](std::size_t shape,const auto& definition,
                           const auto& executable,
                           const Exl3BoundedGraphEntry& lifecycle) noexcept {
            if(definition.ready())++result.live_definitions;
            if(executable.ready())++result.live_executables;
            const auto snapshot=lifecycle.snapshot();
            result.bound_resource_owners_by_shape[shape]=snapshot.resource_count;
            result.bound_external_bytes_by_shape[shape]=snapshot.resource_bytes;
            result.retained_units_by_shape[shape]=snapshot.retained;
            if(snapshot.phase==Exl3BoundedGraphEntry::Phase::quarantined)
                ++result.quarantined_entries;
        };
        add(2,continuation->graph_definition,continuation->graph_executable,
            continuation->graph_lifecycle);
        for(std::size_t shape=0;shape<continuation->additional_graphs.size();++shape) {
            const auto& graph=continuation->additional_graphs[shape];
            add(shape,graph.definition,graph.executable,graph.lifecycle);
        }
        return result;
    }
    Exl3DeviceGraphRoleRecord* continuation_graph_role_record(
            unsigned rows) const noexcept {
        if(!continuation || continuation->capacity!=8 ||
           !continuation->final_norm || !continuation->final_norm->ptr)
            return nullptr;
        const std::size_t slot=rows==4?0:rows==6?1:rows==8?2:
            continuation_graph_role_record_count;
        if(slot==continuation_graph_role_record_count)return nullptr;
        auto* base=static_cast<std::byte*>(continuation->final_norm->ptr)+
            static_cast<std::size_t>(continuation->capacity)*kHidden*
                sizeof(std::uint16_t);
        return reinterpret_cast<Exl3DeviceGraphRoleRecord*>(
            base+slot*sizeof(Exl3DeviceGraphRoleRecord));
    }
    void publish_continuation_graph_roles(unsigned rows,cudaStream_t stream,
            bool synchronize) {
        auto* table=continuation_graph_roles(rows);
        auto* device_record=continuation_graph_role_record(rows);
        require(table && device_record,"continuation graph role table is unavailable");
        const auto context_identity=graph_context_identity();
        const std::array<Exl3GraphRoleBinding,3> bindings{{
            {Exl3GraphPointerRole::token_ids,context_identity,token_ids,rows,
                sizeof(std::int64_t),false},
            {Exl3GraphPointerRole::hidden_output,context_identity,hidden_a,
                static_cast<std::size_t>(rows)*kHidden,
                sizeof(std::uint16_t),true},
            {Exl3GraphPointerRole::embedding_trace,context_identity,
                embedding_trace?embedding_trace->ptr:nullptr,
                static_cast<std::size_t>(rows)*kHidden,
                sizeof(std::uint16_t),true}
        }};
        const auto update=table->update(bindings);
        if(update.changed || table->uploaded_generation()!=table->generation()) {
            cuda_check(cudaMemcpyAsync(device_record,&table->record(),
                sizeof(Exl3DeviceGraphRoleRecord),cudaMemcpyHostToDevice,stream),
                "publish continuation graph pointer roles");
            if(synchronize)
                cuda_check(cudaStreamSynchronize(stream),
                    "complete continuation graph pointer-role publication");
            table->mark_uploaded(table->generation());
        }
    }

    struct ContinuationGraphBoundary {
        std::array<Exl3GraphBufferIdentity,
            Exl3GraphCompatibilityFingerprint::buffer_capacity> buffers{};
        std::size_t count=0;
    };
    ContinuationGraphBoundary continuation_graph_boundary(
            unsigned rows) const noexcept {
        ContinuationGraphBoundary result;
        if(!continuation)return result;
        const auto row_storage=static_cast<std::size_t>(
            (host_kv.enabled || oscar_only) ? prefill_capacity : max_context);
        const auto hidden_bytes=row_storage*static_cast<std::size_t>(kHidden)*
            sizeof(std::uint16_t);
        const auto add=[&](const void* address,std::size_t bytes) noexcept {
            result.buffers[result.count++]={address,bytes};
        };
        add(position_device,sizeof(int));
        add(hidden_a,hidden_bytes);
        add(hidden_b,hidden_bytes);
        add(embedding_trace?embedding_trace->ptr:nullptr,hidden_bytes);
        for(const auto& tap:taps)add(tap?tap->ptr:nullptr,hidden_bytes);
        add(continuation->final_norm?continuation->final_norm->ptr:nullptr,
            continuation->final_norm?continuation->final_norm->bytes:0);
        add(continuation->logits?continuation->logits->ptr:nullptr,
            continuation->logits?continuation->logits->bytes:0);
        add(continuation_graph_role_record(rows),
            sizeof(Exl3DeviceGraphRoleRecord));
        return result;
    }
    std::shared_ptr<const void> graph_context_identity() const noexcept {
        return {model,static_cast<const void*>(this)};
    }
    std::uint32_t continuation_graph_route(bool fixed_b8) const noexcept {
        std::uint32_t result=fixed_b8
            ? Exl3ContinuationGraphRouteFixedB8
            : Exl3ContinuationGraphRouteVariableRows;
        if(oscar && oscar->continuation_cohort_b8_enabled())
            result|=Exl3ContinuationGraphRouteChronologicalOscar;
        if(continuation_graph_gdn_qkvz_concurrent)
            result|=Exl3ContinuationGraphRouteGdnQkvzConcurrent;
        return result;
    }
    Exl3GraphNumericalBoundary continuation_numerical_boundary(
        unsigned rows,cudaStream_t stream) const noexcept {
        Exl3GraphNumericalBoundary result;
        const auto context=graph_context_identity();
        const auto* input=continuation_graph_role_record(rows);
        const auto* output=continuation && continuation->logits?
            continuation->logits->ptr:nullptr;
        result.bind(context,continuation,continuation,input,output,
            reinterpret_cast<std::uintptr_t>(stream),
            stream?execution_stream_owner:std::shared_ptr<const void>{},
            request_generation,request_generation,
            rows>=1 && rows<=8 && continuation && continuation->capacity==8 &&
                input && output && (!stream ||
                    (execution_stream_owner && stream==owned_execution_stream)));
        result.capture_work(Exl3GraphNumericalBoundary::numerical);
        if(host_kv.enabled)result.capture_work(
            Exl3GraphNumericalBoundary::host_registration|
            Exl3GraphNumericalBoundary::external_transfer);
        if(target_projection_observer || target_projection_timing || layer_observer)
            result.capture_work(Exl3GraphNumericalBoundary::dynamic_callback);
        // Pointer-role upload, capture allocation/instantiation, exact host
        // export, resident publication and public callbacks remain outside the
        // graph even though the complete request performs them.
        result.outside_work(Exl3GraphNumericalBoundary::host_allocation|
            Exl3GraphNumericalBoundary::host_export|
            Exl3GraphNumericalBoundary::publication|
            Exl3GraphNumericalBoundary::dynamic_callback|
            Exl3GraphNumericalBoundary::external_transfer);
        return result;
    }
    Exl3GraphCompatibilityFingerprint continuation_graph_fingerprint(
        unsigned rows,int split_class,std::uint32_t route_bits,
        std::uint64_t generation,cudaStream_t stream,
        Exl3ProjectionGraphOptions options=Exl3ProjectionGraphOptions::current()) const noexcept {
        Exl3GraphCompatibilityFingerprint result;
        if(!continuation)return result;
        const auto boundary=continuation_graph_boundary(rows);
        // The role record is the stable captured input boundary. Its generation
        // can publish a new retained token source without changing this key.
        result.bind(graph_context_identity(),model,reconstruction_backing,
            std::span<const Exl3GraphBufferIdentity>(boundary.buffers.data(),boundary.count),
            rows,static_cast<unsigned>(continuation->capacity),
            static_cast<unsigned>(continuation->capacity),
            static_cast<unsigned>(kHidden),split_class,route_bits,
            Exl3GraphPrecision::oscar_int2_fp16,
            Exl3GraphPositionPolicy::oscar_split_class,generation,stream,options);
        return result;
    }
    void bind_continuation_graph_lifecycle(Exl3BoundedGraphEntry& entry,
        const Exl3GraphCompatibilityFingerprint& fingerprint,unsigned rows,
        std::uint64_t generation) {
        const auto boundary=continuation_graph_boundary(rows);
        std::array<Exl3GraphBoundResource,
            Exl3BoundedGraphEntry::resource_capacity> resources{};
        std::size_t count=0;
        const auto context_identity=graph_context_identity();
        for(std::size_t i=0;i<boundary.count;++i)
            resources[count++]={context_identity,boundary.buffers[i].address,
                boundary.buffers[i].bytes,static_cast<std::uint64_t>(i)};
        resources[count++]={model,model.get(),model->model_bytes,100};
        if(reconstruction_backing)
            resources[count++]={reconstruction_backing,reconstruction_backing.get(),
                reconstruction_backing->allocation.bytes,101};
        if(execution_stream_owner)
            resources[count++]={execution_stream_owner,
                reinterpret_cast<const void*>(owned_execution_stream),
                sizeof(cudaStream_t),102};
        const auto* reserved=continuation_graph_extent(rows);
        if(!reserved)throw std::logic_error(
            "continuation graph shape is outside the reserved menu");
        const auto reservation=*reserved;
        const std::uint64_t configuration=0x4752415000000000ULL |
            (static_cast<std::uint64_t>(rows)<<32) |
            generation;
        entry.bind(fingerprint,std::span<const Exl3GraphBoundResource>(
            resources.data(),count),generation,configuration,reservation);
    }
    bool complete_continuation_graph_use(int rows,cudaStream_t stream,
        int error=0) noexcept {
        if(!continuation)return false;
        // Only reserved 4/6/8-row shapes can own a captured continuation
        // graph. Other supported continuation widths execute eagerly and have
        // no graph final-use authority to complete.
        if(rows!=4 && rows!=6 && rows!=8)return true;
        auto& entry=rows==8?continuation->graph_lifecycle:
            continuation->additional_graphs[rows==4?0:1].lifecycle;
        auto* roles=continuation_graph_roles(static_cast<unsigned>(rows));
        if(!entry.pending())return roles && !roles->pending_replay();
        ++continuation->graph_stats.final_use_observations;
        const auto serial=entry.replay_serial();
        const bool lifecycle_completed=entry.complete_after_drain(stream,serial,error);
        const bool roles_completed=roles && roles->complete_use(serial);
        const bool completed=lifecycle_completed && roles_completed;
        if(error)entry.quarantine("continuation graph final use failed",error);
        else if(!completed)entry.quarantine(
            "continuation graph final-use identity mismatch",
            static_cast<int>(cudaErrorInvalidResourceHandle));
        if(error || !completed)poison_graph_retirement(static_cast<cudaError_t>(
            error?error:static_cast<int>(cudaErrorInvalidResourceHandle)));
        return completed;
    }
    void drain_continuation_graph_uses() {
        if(!continuation)return;
        const bool pending=continuation->graph_lifecycle.pending() ||
            continuation->additional_graphs[0].lifecycle.pending() ||
            continuation->additional_graphs[1].lifecycle.pending();
        if(!pending)return;
        const auto* option=std::getenv("NINFER_EXL3_NARROW_GRAPH_DRAIN");
        if(option && std::string_view(option)!="0" &&
           std::string_view(option)!="1")
            throw std::invalid_argument(
                "narrow continuation graph drain must be 0 or 1");
        const std::array<Exl3ContinuationGraphDrainPolicy::Pending,3> uses{{
            {continuation->additional_graphs[0].lifecycle.pending(),
             reinterpret_cast<std::uintptr_t>(
                 continuation->additional_graphs[0].origin_stream)},
            {continuation->additional_graphs[1].lifecycle.pending(),
             reinterpret_cast<std::uintptr_t>(
                 continuation->additional_graphs[1].origin_stream)},
            {continuation->graph_lifecycle.pending(),
             reinterpret_cast<std::uintptr_t>(continuation->graph_origin_stream)}}};
        const auto mode=Exl3ContinuationGraphDrainPolicy::select(
            option && std::string_view(option)=="1",execution_stream_owner,
            reinterpret_cast<std::uintptr_t>(owned_execution_stream),uses);
        if(mode==Exl3ContinuationGraphDrainPolicy::Mode::owned_stream) {
            cuda_check(cudaStreamSynchronize(owned_execution_stream),
                "drain owned-stream continuation graph uses before context mutation");
            ++continuation->graph_stats.narrow_final_use_drains;
        } else {
            cuda_check(cudaDeviceSynchronize(),
                "drain continuation graph uses before context mutation");
            ++continuation->graph_stats.device_final_use_drains;
        }
        for(const int rows:{4,6,8}) {
            auto& lifecycle=rows==8?continuation->graph_lifecycle:
                continuation->additional_graphs[rows==4?0:1].lifecycle;
            if(lifecycle.pending())require(complete_continuation_graph_use(
                rows,rows==8?continuation->graph_origin_stream:
                    continuation->additional_graphs[rows==4?0:1].origin_stream),
                "continuation graph mutation lost final-use ownership");
        }
    }
    void invalidate_continuation_graphs(std::string_view reason) {
        if(!continuation)return;
        continuation->graph_active=false;
        continuation->projection_binding.invalidate();
        continuation->graph_compatibility.invalidate();
        continuation->graph_lifecycle.invalidate(reason);
        continuation->graph_reason.assign(reason);
        for(auto& graph:continuation->additional_graphs) {
            graph.active=false;
            graph.projection_binding.invalidate();
            graph.compatibility.invalidate();
            graph.lifecycle.invalidate(reason);
            graph.reason.assign(reason);
        }
    }

    static inline std::atomic<Impl*> host_kv_quarantine{nullptr};
    int graph_retirement_error=0;
    int graph_owner_device=-1;
    void bind_graph_device() {
        int device=-1;auto error=cudaGetDevice(&device);
        if(error==cudaSuccess && graph_owner_device>=0 && graph_owner_device!=device)
            error=cudaErrorInvalidDevice;
        if(error!=cudaSuccess) {
            poison_graph_retirement(error);
            cuda_check(error,"query or match graph allocation device");
        }
        graph_owner_device=device;
    }
    bool has_graph_handles() const noexcept {
        if(graph_definition.ready() || graph_executable.ready())return true;
        for(const auto& graph:host_kv_gdn_segment_graphs)
            if(graph.definition.ready() || graph.executable.ready())return true;
        for(const auto& graph:host_kv_full_layer_graphs)
            if(graph.definition.ready() || graph.executable.ready())return true;
        for(const auto& graph:ordinary_full_layer_graphs)
            if(graph.definition.ready() || graph.executable.ready())return true;
        for(const auto& graph:host_kv_mlp_tail_graphs)
            if(graph.definition.ready() || graph.executable.ready())return true;
        if(transaction && (transaction->checkpoint_graph_definition.ready() ||
                           transaction->checkpoint_graph_executable.ready()))
            return true;
        if(!continuation)return false;
        if(continuation->graph_definition.ready() || continuation->graph_executable.ready())return true;
        for(const auto& graph:continuation->additional_graphs)
            if(graph.definition.ready() || graph.executable.ready())return true;
        return false;
    }
    template<class Query,class Drain>
    cudaError_t retire_graph_handles_with(Query&& query,Drain&& drain) noexcept {
        if(graph_retirement_error)return static_cast<cudaError_t>(graph_retirement_error);
        if(!has_graph_handles())return cudaSuccess;
        const auto error=ninfer::detail::retire_decode_graph_device(graph_owner_device,
            query,drain,
            [&]() noexcept {
                auto result=ninfer::detail::retire_decode_graph_pair(graph_executable,graph_definition);
                if(result!=cudaSuccess)return result;
                for(auto& graph:host_kv_gdn_segment_graphs) {
                    result=ninfer::detail::retire_decode_graph_pair(
                        graph.executable,graph.definition);
                    if(result!=cudaSuccess)return result;
                }
                for(auto& graph:host_kv_full_layer_graphs) {
                    result=ninfer::detail::retire_decode_graph_pair(
                        graph.executable,graph.definition);
                    if(result!=cudaSuccess)return result;
                }
                for(auto& graph:ordinary_full_layer_graphs) {
                    result=ninfer::detail::retire_decode_graph_pair(
                        graph.executable,graph.definition);
                    if(result!=cudaSuccess)return result;
                }
                for(auto& graph:host_kv_mlp_tail_graphs) {
                    result=ninfer::detail::retire_decode_graph_pair(
                        graph.executable,graph.definition);
                    if(result!=cudaSuccess)return result;
                }
                if(transaction) {
                    result=ninfer::detail::retire_decode_graph_pair(
                        transaction->checkpoint_graph_executable,
                        transaction->checkpoint_graph_definition);
                    if(result!=cudaSuccess)return result;
                }
                if(continuation) {
                    const auto retire_entry=[&](auto& executable,auto& definition,
                        Exl3BoundedGraphEntry& lifecycle,
                        Exl3DeviceGraphRoleTable& roles,
                        cudaStream_t stream) noexcept {
                        const bool had_handles=executable.ready() || definition.ready();
                        if(had_handles)++continuation->graph_stats.destruction_attempts;
                        const auto phase=lifecycle.snapshot().phase;
                        if(phase==Exl3BoundedGraphEntry::Phase::quarantined) {
                            if(had_handles)++continuation->graph_stats.destruction_failures;
                            return cudaErrorUnknown;
                        }
                        if(lifecycle.pending()) {
                            const auto serial=lifecycle.replay_serial();
                            if(!lifecycle.complete_after_drain(stream,serial,0) ||
                               !roles.complete_use(serial)) {
                            lifecycle.quarantine("context graph final-use identity mismatch",
                                static_cast<int>(cudaErrorInvalidResourceHandle));
                            if(had_handles)++continuation->graph_stats.destruction_failures;
                            return cudaErrorInvalidResourceHandle;
                            }
                        } else if(roles.pending_replay()) {
                            lifecycle.quarantine("context graph role final-use mismatch",
                                static_cast<int>(cudaErrorInvalidResourceHandle));
                            if(had_handles)++continuation->graph_stats.destruction_failures;
                            return cudaErrorInvalidResourceHandle;
                        }
                        if(phase!=Exl3BoundedGraphEntry::Phase::empty)
                            lifecycle.invalidate("context graph retirement");
                        const auto retired=ninfer::detail::retire_decode_graph_pair(
                            executable,definition);
                        if(retired!=cudaSuccess) {
                            if(had_handles)++continuation->graph_stats.destruction_failures;
                            lifecycle.quarantine("context graph destruction failed",retired);
                            return retired;
                        }
                        if(lifecycle.snapshot().phase==
                                Exl3BoundedGraphEntry::Phase::invalidated &&
                           !lifecycle.release_after_destroy()) {
                            lifecycle.quarantine("context graph resource release refused",
                                static_cast<int>(cudaErrorInvalidResourceHandle));
                            if(had_handles)++continuation->graph_stats.destruction_failures;
                            return cudaErrorInvalidResourceHandle;
                        }
                        if(had_handles)++continuation->graph_stats.destruction_successes;
                        return cudaSuccess;
                    };
                    result=retire_entry(continuation->graph_executable,
                        continuation->graph_definition,
                        continuation->graph_lifecycle,
                        continuation->graph_roles,
                        continuation->graph_origin_stream);
                    if(result!=cudaSuccess)return result;
                    for(auto& graph:continuation->additional_graphs) {
                        result=retire_entry(graph.executable,graph.definition,
                            graph.lifecycle,graph.roles,graph.origin_stream);
                        if(result!=cudaSuccess)return result;
                    }
                }
                return cudaSuccess;
            });
        if(error!=cudaSuccess)poison_graph_retirement(error);
        else {
            graph_active=false;
            if(transaction)transaction->checkpoint_graph_active=false;
            if(continuation) {
                continuation->graph_active=false;
                continuation->projection_binding.invalidate();
                continuation->graph_compatibility.invalidate();
                for(auto& graph:continuation->additional_graphs) {
                    graph.active=false;
                    graph.projection_binding.invalidate();
                    graph.compatibility.invalidate();
                }
            }
        }
        return error;
    }
    cudaError_t retire_graph_handles() noexcept {
        return retire_graph_handles_with(
            [](int* device) noexcept {return cudaGetDevice(device);},
            []() noexcept {return cudaDeviceSynchronize();});
    }
    void poison_graph_retirement(cudaError_t error) noexcept {
        if(error==cudaSuccess || graph_retirement_error)return;
        graph_retirement_error=static_cast<int>(error);
        // Fence execution immediately, before the owner reaches destruction.
        hostkv_quarantined_contexts.fetch_add(1,std::memory_order_release);
    }
    std::uint64_t allocation_owner_metadata_bytes() const;
    static void require_host_kv_retirement_admission() {
        require(hostkv_quarantined_contexts.load(std::memory_order_acquire)==0,
            "unresolved HostKV retirement blocks context execution");
        require(DeviceAllocation::quarantined_count.load(std::memory_order_acquire)==0,
            "unresolved generic cleanup blocks context execution");
    }
    static bool retain_failed_host_kv_drain(std::unique_ptr<Impl>& owner) noexcept {
        if(owner && !owner->graph_retirement_error)(void)owner->retire_graph_handles();
        if(owner && owner->graph_retirement_error) {
            owner->host_kv_retirement_error=owner->graph_retirement_error;
            auto* retained=owner.release();
            auto* head=host_kv_quarantine.load(std::memory_order_relaxed);
            do {retained->reconstruction_quarantine_next=head;}
            while(!host_kv_quarantine.compare_exchange_weak(head,retained,std::memory_order_release,std::memory_order_relaxed));
            // poison_graph_retirement already recorded this unresolved owner.
            return true;
        }
        if(!owner || (!owner->host_kv_copy_stream && !owner->host_kv_restore_source && !owner->host_kv_forward_used))return false;
        int current_device=-1;
        auto error=owner->host_kv_device_query_failure_for_test?cudaErrorInitializationError:
            cudaGetDevice(&current_device);
        // Refuse destruction on an unproven device. The whole bundle must
        // remain alive, including events and allocations released by ~Impl.
        if(error==cudaSuccess && ((owner->host_kv_copy_stream && current_device!=owner->host_kv_copy_device) ||
                (owner->host_kv_restore_source && current_device!=owner->host_kv_restore_device) ||
                (owner->host_kv_forward_used && current_device!=owner->host_kv_forward_device) ||
                owner->host_kv_device_mismatch_for_test))
            error=cudaErrorInvalidDevice;
        if(error==cudaSuccess && owner->host_kv_restore_source)
            error=cudaStreamSynchronize(owner->host_kv_restore_stream);
        if(error==cudaSuccess && owner->host_kv_forward_used)
            error=owner->host_kv_compute_retirement_failure_for_test?cudaErrorUnknown:
                cudaStreamSynchronize(owner->host_kv_forward_stream);
        if(error==cudaSuccess)
            error=owner->host_kv_retirement_failure_for_test?cudaErrorUnknown:
                (owner->host_kv_copy_stream?cudaStreamSynchronize(owner->host_kv_copy_stream):cudaSuccess);
        if(error==cudaSuccess)return false;
        owner->host_kv_retirement_error=static_cast<int>(error);
        auto* retained=owner.release();
        auto* head=host_kv_quarantine.load(std::memory_order_relaxed);
        do {retained->reconstruction_quarantine_next=head;}
        while(!host_kv_quarantine.compare_exchange_weak(head,retained,std::memory_order_release,std::memory_order_relaxed));
        hostkv_quarantined_contexts.fetch_add(1,std::memory_order_release);
        return true;
    }
    ~Impl() {
        if(repair_stream) {
            (void)cudaStreamSynchronize(repair_stream);
            (void)cudaEventDestroy(repair_fork);
            (void)cudaEventDestroy(repair_join);
            (void)cudaStreamDestroy(repair_stream);
            repair_stream=nullptr;
        }
        // Captured nodes retain the auxiliary stream/events/workspace. Destroy
        // the executable and definition before releasing any such resource.
        if(transaction) {
            transaction->checkpoint_graph_executable.reset();
            transaction->checkpoint_graph_definition.reset();
        }
        if (continuation) {
            for (auto& graph : continuation->additional_graphs) {
                graph.executable.reset();
                graph.definition.reset();
            }
            continuation->graph_executable.reset();
            continuation->graph_definition.reset();
        }
        for (auto event : gdn_qkvz_fork) if (event) cudaEventDestroy(event);
        for (auto event : gdn_qkvz_z_done) if (event) cudaEventDestroy(event);
        if (gdn_qkvz_z_stream) cudaStreamDestroy(gdn_qkvz_z_stream);
        if (eager_mlp_gateup_fork) cudaEventDestroy(eager_mlp_gateup_fork);
        if (eager_mlp_gateup_up_done)
            cudaEventDestroy(eager_mlp_gateup_up_done);
        if (eager_mlp_gateup_stream)
            cudaStreamDestroy(eager_mlp_gateup_stream);
        if(prefill_qkv_fork)cudaEventDestroy(prefill_qkv_fork);
        if(prefill_qkv_k_done)cudaEventDestroy(prefill_qkv_k_done);
        if(prefill_qkv_v_done)cudaEventDestroy(prefill_qkv_v_done);
        if(prefill_qkv_k_stream)cudaStreamDestroy(prefill_qkv_k_stream);
        if(prefill_qkv_v_stream)cudaStreamDestroy(prefill_qkv_v_stream);
        if(host_kv_copy_stream) cudaStreamSynchronize(host_kv_copy_stream);
        for(auto& scatter:host_kv_pinned_scatter) scatter.clear();
        host_kv_banked_d2h_scatter.clear();
        for(auto event:host_kv_pinned_ready) if(event) cudaEventDestroy(event);
        if(host_kv_h2d_ready) cudaEventDestroy(host_kv_h2d_ready);
        if(host_kv_compute_ready) cudaEventDestroy(host_kv_compute_ready);
        if(host_kv_copy_stream) cudaStreamDestroy(host_kv_copy_stream);
    }

    void acquire_host_kv_pinned_slot(int index,const char* label) {
        require(index>=0 && index<host_kv_pinned_slot_count &&
                host_kv_pinned_staging[index],
            "invalid exact host KV pinned slot acquire");
        if(registered_kv_pending[index]) {
            try {registered_kv_completion(index,registered_kv_pending[index]);}
            catch(...) {++host_kv.registered_upload_failures;throw;}
            host_kv.registered_upload_bytes+=registered_kv_pending_bytes[index];
            registered_kv_pending[index]=0;registered_kv_pending_bytes[index]=0;
        }
        if(host_kv_pinned_in_flight[index]) {
            require(!host_kv_pinned_final_use[index].first_error(),
                "exact host KV pinned slot has unresolved transfer failure");
            const auto wait_started=host_kv_cpu_profile?
                std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
            const auto completion=cudaEventSynchronize(host_kv_pinned_ready[index]);
            auto& witness=host_kv_pinned_final_use[index];
            require(witness.finish(witness.generation(),static_cast<int>(completion)),
                "exact host KV pinned completion generation unavailable");
            cuda_check(completion,label);
            if(host_kv_cpu_profile) host_kv.pinned_wait_cpu_ns+=
                static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-wait_started).count());
            host_kv_pinned_in_flight[index]=false;
            ++host_kv.pinned_slot_waits;
        }
        if(index>=static_cast<int>(host_kv_pinned_scatter.size()))return;
        auto& pieces=host_kv_pinned_scatter[static_cast<std::size_t>(index)];
        if(pieces.empty()) return;
        require(host_kv_pinned_staging[index] &&
            host_kv_pinned_staging[index]->size()==host_kv_pinned_chunk_bytes,
            "deferred exact host KV scatter missing staging owner");
        const auto scatter_started=host_kv_cpu_profile?
            std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        exl3_host_kv_scatter_batch(pieces,[&](const auto& piece) {
            require(piece.page && piece.bank>=0 && piece.bank<16 && piece.bytes>0,
                "invalid deferred exact host KV scatter descriptor");
            auto& plane=piece.key_plane?piece.page->k[piece.bank]:piece.page->v[piece.bank];
            require(exl3_host_kv_scatter_fits(plane.size()*2,piece.destination_byte_offset,
                    host_kv_pinned_chunk_bytes,piece.staging_offset,piece.bytes),
                "deferred exact host KV scatter descriptor exceeds storage");
        },[&](const auto& piece) noexcept {
            auto& plane=piece.key_plane?piece.page->k[piece.bank]:piece.page->v[piece.bank];
            std::memcpy(reinterpret_cast<std::byte*>(plane.data())+
                    piece.destination_byte_offset,
                static_cast<const std::byte*>(host_kv_pinned_staging[index]->data())+
                    piece.staging_offset,piece.bytes);
            host_kv.pinned_scatter_bytes+=piece.bytes;
        });
        pieces.clear();
        if(host_kv_cpu_profile) host_kv.pinned_scatter_cpu_ns+=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-scatter_started).count());
        ++host_kv.pinned_deferred_drains;
    }

    void begin_host_kv_batch_metadata(std::size_t pages) {
        require(pages<=host_kv_batch_destinations.capacity()/2 &&
            pages<=host_kv_batch_sources.capacity()/2 && pages<=host_kv_batch_sizes.capacity()/2,
            "host KV metadata exceeds prepared context capacity");
        if(host_kv_batch_use.retire_before_reuse([&] {
            // A stream wait orders device work, not host mutation of these
            // borrowed descriptor arrays. Retain them through batch completion.
            cuda_check(cudaStreamSynchronize(host_kv_copy_stream),
                "complete host KV batch before descriptor reuse");
        })) {
            ++host_kv.batch_metadata_reuse_drains;
        }
        host_kv_batch_destinations.clear();host_kv_batch_sources.clear();host_kv_batch_sizes.clear();
        ++host_kv.batch_metadata_reuses;
    }
    void submit_host_kv_batch(Exl3TransferDescriptorStorage<void*>& destinations,
                              Exl3TransferDescriptorStorage<const void*>& sources,
                              Exl3TransferDescriptorStorage<std::size_t>& sizes,
                              cudaMemcpySrcAccessOrder source_order,
                              const char* label) {
        require(host_kv_batch_copy && host_kv_copy_stream && !destinations.empty() &&
            !host_kv_batch_use.pending() && destinations.size()==sources.size() && sources.size()==sizes.size(),
            "invalid exact host KV batch-copy submission");
        cudaMemcpyAttributes attributes{};
        attributes.srcAccessOrder=source_order;
        // A failed enqueue may still leave uncertain work; do not permit
        // descriptor mutation until the owning stream has been drained.
        host_kv_batch_use.begin_submission();
        cuda_check(cudaMemcpyBatchAsync(destinations.data(),sources.data(),sizes.data(),
            sizes.size(),attributes,host_kv_copy_stream),label);
        ++host_kv.copy_submissions;
        if(std::exchange(host_kv_copy_failure_for_test,false)) {
            host_kv_retirement_failure_for_test=true;
            throw std::runtime_error("injected HostKV failure after copy submission");
        }
    }

    void submit_host_kv_pinned_copy(int slot,void* destination,const void* source,
        std::size_t bytes,cudaMemcpyKind direction,const char* label) {
        require(slot>=0 && slot<host_kv_pinned_slot_count &&
                !host_kv_pinned_in_flight[slot],
            "exact host KV pinned submission requires an acquired slot");
        auto& witness=host_kv_pinned_final_use[slot];
        // These witnesses are private to the context; the monotonically growing
        // generation distinguishes each reuse of its fixed event address.
        const auto generation=witness.begin(1,1,
            reinterpret_cast<std::uintptr_t>(host_kv_pinned_ready[slot]));
        host_kv_pinned_in_flight[slot]=true;
        const auto copy=cudaMemcpyAsync(destination,source,bytes,direction,host_kv_copy_stream);
        if(copy!=cudaSuccess) {
            witness.finish(generation,static_cast<int>(copy));
            cuda_check(copy,label);
        }
        const auto recorded=cudaEventRecord(host_kv_pinned_ready[slot],host_kv_copy_stream);
        if(recorded!=cudaSuccess) {
            witness.finish(generation,static_cast<int>(recorded));
            cuda_check(recorded,"record exact host KV pinned chunk completion");
        }
        ++host_kv.copy_submissions;
        if(std::exchange(host_kv_copy_failure_for_test,false)) {
            witness.finish(generation,static_cast<int>(cudaErrorUnknown));
            host_kv_retirement_failure_for_test=true;
            throw std::runtime_error("injected HostKV failure after copy submission");
        }
    }

    void submit_host_kv_pinned_plane(
        const std::vector<std::shared_ptr<const Exl3ExactKVPage>>& pages,
        int bank,void* destination,bool key_plane,int position,const char* label,int first_row=0,
        const Exl3AttentionPageRanges* shared_pages=nullptr,
        bool prefill_pinned_batch=false) {
        require(host_kv_pinned_chunks && host_kv_copy_stream && !pages.empty() &&
            bank>=0 && bank<16 && destination,"invalid exact host KV pinned-chunk upload");
        require(first_row>=0 && first_row<=position,"attention cached prefix outside published history");
        if(first_row==position)return;
        int slot=0;std::size_t buffered=0,logical_offset=0,buffer_start=0;
        bool prefill_pinned_batch_used=false;
        const auto acquire=[&] {
            acquire_host_kv_pinned_slot(slot,
                "reuse exact host KV pinned chunk");
        };
        const auto flush=[&] {
            if(!buffered) return;
            submit_host_kv_pinned_copy(slot,static_cast<std::byte*>(destination)+buffer_start,
                host_kv_pinned_staging[slot]->data(),buffered,cudaMemcpyHostToDevice,label);
            slot=(slot+1)%host_kv_pinned_slot_count;buffered=0;
        };
        acquire();
        for(const auto& page:pages) {
            require(static_cast<std::size_t>(page->first)*1024*2==logical_offset,
                "exact host KV pinned pages must be contiguous");
            if(shared_pages && shared_pages->contains_page(page->first,page->rows)) {
                require(page->rows==64 && position>=page->first+64,"shared attention upload gap requires complete published page");
                flush();logical_offset+=64ULL*1024*2;continue;
            }
            const auto extent=Exl3ExactKVExtent::view(page,bank,key_plane?
                Exl3ExactKVExtent::Plane::key:Exl3ExactKVExtent::Plane::value,position,first_row);
            const auto* source=reinterpret_cast<const std::byte*>(extent.data());
            // A unique private tail may already be resized for this append.
            // Upload only rows belonging to the previously published prefix.
            const int valid_rows=std::clamp(position-page->first,0,page->rows);
            const int skipped_rows=std::clamp(first_row-page->first,0,valid_rows);
            const std::size_t skipped_bytes=static_cast<std::size_t>(skipped_rows)*1024*2;
            logical_offset+=skipped_bytes;
            const std::size_t plane_bytes=extent.bytes();
            if(!plane_bytes)continue;
            ++host_kv.transfer_calls;
            if(shared_page_copier && page->rows==64 && extent.first()+extent.rows()==page->first+64) {
                flush();acquire();
                bool copied=false;
                try{copied=shared_page_copier(extent,destination,max_context,host_kv_copy_stream,host_kv_pinned_ready[slot]);}
                catch(...){++host_kv.shared_page_failures;throw;}
                if(copied) {
                    logical_offset+=plane_bytes;++host_kv.copy_submissions;
                    host_kv.shared_page_copy_bytes+=plane_bytes;continue;
                }
                ++host_kv.shared_page_fallbacks;
            }
            host_kv.h2d_bytes+=plane_bytes;
            if(!prefill_pinned_batch && registered_kv_uploader &&
               page->rows==Exl3ExactKVPage::token_capacity) {
                flush();acquire();
                std::uint64_t uploaded=0;
                try {
                    uploaded=registered_kv_uploader(extent,destination,max_context,
                        host_kv_copy_stream,host_kv_pinned_ready[slot],slot);
                } catch(...) {++host_kv.registered_upload_failures;throw;}
                if(uploaded) {
                    logical_offset+=plane_bytes;++host_kv.copy_submissions;
                    registered_kv_pending[slot]=uploaded;registered_kv_pending_bytes[slot]=plane_bytes;
                    std::uint64_t pending=0;
                    for(int index=0;index<host_kv_pinned_slot_count;++index)
                        pending+=std::uint64_t(bool(registered_kv_pending[index]));
                    host_kv.registered_upload_peak_pending=
                        std::max(host_kv.registered_upload_peak_pending,pending);
                    slot=(slot+1)%host_kv_pinned_slot_count;
                    continue;
                }
                ++host_kv.registered_upload_fallbacks;
            }
            if(prefill_pinned_batch) {
                prefill_pinned_batch_used=true;
                ++host_kv.prefill_pinned_batch_page_planes;
                host_kv.prefill_pinned_batch_bytes+=plane_bytes;
            }
            for(std::size_t source_offset=0;source_offset<plane_bytes;) {
                if(!buffered) {acquire();buffer_start=logical_offset;}
                const std::size_t amount=std::min(host_kv_pinned_chunk_bytes-buffered,
                    plane_bytes-source_offset);
                const auto gather_started=host_kv_cpu_profile?
                    std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
                std::memcpy(static_cast<std::byte*>(host_kv_pinned_staging[slot]->data())+buffered,
                    source+source_offset,amount);
                if(host_kv_cpu_profile) host_kv.pinned_gather_cpu_ns+=
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-gather_started).count());
                buffered+=amount;source_offset+=amount;logical_offset+=amount;
                if(buffered==host_kv_pinned_chunk_bytes) flush();
            }
        }
        flush();
        if(prefill_pinned_batch_used)
            ++host_kv.prefill_pinned_batch_plane_calls;
        // Drain the bounded tail before returning to layer compute/publication.
        // Earlier extents overlapped submission; staged slots retain their old
        // asynchronous behavior and are not forced through an extra wait here.
        // The qualified two-slot route retains its per-plane completion
        // boundary.  The default-off deep ring intentionally carries exact
        // registered owners across planes/layers until the existing complete
        // forward drain, removing repeated host synchronization from the
        // target submission path without changing copy or compute order.
        if(host_kv_pinned_slot_count==2)
            for(int index=0;index<2;++index)if(registered_kv_pending[index])
                acquire_host_kv_pinned_slot(index,
                    "complete registered host KV plane tail");
    }

    void receive_host_kv_pinned_plane(
        const std::vector<std::shared_ptr<Exl3ExactKVPage>>& pages,
        int bank,const void* source,bool key_plane,int position,const char* label) {
        require(host_kv_pinned_d2h && host_kv_pinned_chunks && host_kv_copy_stream &&
            !pages.empty() && bank>=0 && bank<16 && source,
            "invalid exact host KV pinned-chunk download");
        require(position>=0 && position<=max_context,"pinned download position outside context");
        int planned_end=-1;
        for(const auto& page:pages) {
            require(page && page->first>=0 && page->first<=max_context &&
                page->rows>0 && page->rows<=Exl3ExactKVPage::token_capacity &&
                page->rows<=max_context-page->first,
                "pinned download page extent outside context");
            const auto& plane=key_plane?page->k[bank]:page->v[bank];
            require(plane.size()>=static_cast<std::size_t>(page->rows)*1024,
                "pinned download page plane is incomplete");
            require(exl3_host_kv_append_download_interval(page->first,page->rows,position,max_context,planned_end),
                "pinned download page sequence is not contiguous");
        }
        int slot=0;
        std::size_t buffered=0,buffer_start=0;
        const auto acquire=[&](int index) {
            acquire_host_kv_pinned_slot(index,
                "complete exact host KV pinned download before scatter/reuse");
        };
        const auto flush=[&] {
            if(!buffered) return;
            submit_host_kv_pinned_copy(slot,host_kv_pinned_staging[slot]->data(),
                static_cast<const std::byte*>(source)+buffer_start,buffered,
                cudaMemcpyDeviceToHost,label);
            slot=1-slot;buffered=0;
        };
        for(const auto& page:pages) {
            const int skip=std::clamp(position-page->first,0,page->rows);
            auto& plane=key_plane?page->k[bank]:page->v[bank];
            std::size_t destination_byte_offset=
                static_cast<std::size_t>(skip)*1024*2;
            std::size_t remaining=static_cast<std::size_t>(page->rows-skip)*1024*2;
            if(!remaining)continue;
            std::size_t source_offset=static_cast<std::size_t>(page->first+skip)*1024*2;
            host_kv.d2h_bytes+=remaining;++host_kv.transfer_calls;
            while(remaining) {
                if(!buffered) {acquire(slot);buffer_start=source_offset;}
                require(source_offset==buffer_start+buffered,
                    "exact host KV pinned download source must be contiguous");
                const std::size_t amount=std::min(host_kv_pinned_chunk_bytes-buffered,
                    remaining);
                require(amount && amount%(1024*sizeof(std::uint16_t))==0,
                    "pinned scatter piece must contain complete KV rows");
                host_kv_pinned_scatter[slot].push_back(
                    {page,destination_byte_offset,buffered,amount,bank,key_plane});
                destination_byte_offset+=amount;source_offset+=amount;
                remaining-=amount;buffered+=amount;
                const auto scan_started=host_kv_cpu_profile?
                    std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
                std::uint64_t pending_bytes=0;
                for(const auto& slot_pieces:host_kv_pinned_scatter)
                    for(const auto& piece:slot_pieces) pending_bytes+=piece.bytes;
                if(host_kv_cpu_profile) host_kv.pinned_pending_scan_cpu_ns+=
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-scan_started).count());
                host_kv.pinned_max_pending_bytes=
                    std::max(host_kv.pinned_max_pending_bytes,pending_bytes);
                if(buffered==host_kv_pinned_chunk_bytes) flush();
            }
        }
        flush();
        // Authoritative page vectors are publishable only after the device copy
        // and CPU scatter complete. This also proves both fixed slots reusable.
        if(!host_kv_deferred_scatter) {acquire(0);acquire(1);}
        else ++host_kv.pinned_plane_tail_deferrals;
    }

    void submit_host_kv_banked_d2h_plane(
        const std::vector<std::shared_ptr<Exl3ExactKVPage>>& pages,
        int bank,const void* source,bool key_plane,int position,int rows,
        const char* label) {
        require(host_kv_banked_d2h && host_kv_banked_d2h_staging &&
            host_kv_copy_stream && !pages.empty() && bank>=0 &&
            bank<host_kv_banked_d2h_banks && source && rows>0 &&
            rows<=host_kv_banked_d2h_rows && position>=0 &&
            position<=max_context-rows,
            "invalid exact host KV banked D2H submission");
        const auto region=static_cast<std::size_t>(host_kv_banked_d2h_rows)*
            host_kv_row_bytes;
        const auto plane_index=static_cast<std::size_t>(bank)*2+
            (key_plane?0U:1U);
        const auto staging_base=plane_index*region;
        const auto copy_bytes=static_cast<std::size_t>(rows)*host_kv_row_bytes;
        require(staging_base<=host_kv_banked_d2h_staging->size() &&
            copy_bytes<=host_kv_banked_d2h_staging->size()-staging_base,
            "exact host KV banked D2H slice exceeds staging owner");
        std::size_t new_pieces=0;
        int planned_end=-1;
        for(const auto& page:pages) {
            require(page && page->first>=0 && page->first<=max_context &&
                page->rows>0 && page->rows<=Exl3ExactKVPage::token_capacity &&
                page->rows<=max_context-page->first,
                "banked D2H page extent outside context");
            const auto& plane=key_plane?page->k[bank]:page->v[bank];
            require(plane.size()>=static_cast<std::size_t>(page->rows)*1024,
                "banked D2H page plane is incomplete");
            require(exl3_host_kv_append_download_interval(page->first,page->rows,
                    position,max_context,planned_end),
                "banked D2H page sequence is not contiguous");
            if(page->first+page->rows>position)++new_pieces;
        }
        require(planned_end==position+rows &&
            new_pieces<=host_kv_banked_d2h_scatter.capacity()-
                host_kv_banked_d2h_scatter.size(),
            "banked D2H descriptors omit suffix or exceed fixed capacity");
        for(const auto& page:pages) {
            const int skip=std::clamp(position-page->first,0,page->rows);
            const int copied_rows=page->rows-skip;
            if(!copied_rows)continue;
            auto& plane=key_plane?page->k[bank]:page->v[bank];
            const auto destination_byte_offset=static_cast<std::size_t>(skip)*
                host_kv_row_bytes;
            const auto bytes=static_cast<std::size_t>(copied_rows)*host_kv_row_bytes;
            const auto row_offset=page->first+skip-position;
            require(row_offset>=0 && row_offset+copied_rows<=rows,
                "banked D2H page lies outside submitted suffix");
            host_kv_banked_d2h_scatter.push_back({page,destination_byte_offset,
                staging_base+static_cast<std::size_t>(row_offset)*host_kv_row_bytes,
                bytes,bank,key_plane});
            host_kv.d2h_bytes+=bytes;++host_kv.transfer_calls;
        }
        const auto copy=cudaMemcpyAsync(
            static_cast<std::byte*>(host_kv_banked_d2h_staging->data())+staging_base,
            static_cast<const std::byte*>(source)+
                static_cast<std::size_t>(position)*host_kv_row_bytes,
            copy_bytes,cudaMemcpyDeviceToHost,host_kv_copy_stream);
        cuda_check(copy,label);
        ++host_kv.copy_submissions;++host_kv.banked_d2h_planes;
        host_kv.banked_d2h_bytes+=copy_bytes;
        if(std::exchange(host_kv_copy_failure_for_test,false)) {
            host_kv_retirement_failure_for_test=true;
            throw std::runtime_error("injected HostKV failure after copy submission");
        }
    }

    void drain_host_kv_banked_d2h() {
        if(host_kv_banked_d2h_scatter.empty())return;
        require(host_kv_banked_d2h_staging &&
            host_kv_banked_d2h_staging->size()==host_kv_banked_d2h_bytes,
            "banked D2H drain missing fixed staging owner");
        const auto scatter_started=host_kv_cpu_profile?
            std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        std::uint64_t pending_bytes=0;
        for(const auto& piece:host_kv_banked_d2h_scatter)
            pending_bytes+=piece.bytes;
        host_kv.pinned_max_pending_bytes=std::max(
            host_kv.pinned_max_pending_bytes,pending_bytes);
        exl3_host_kv_scatter_batch(host_kv_banked_d2h_scatter,[&](const auto& piece) {
            require(piece.page && piece.bank>=0 &&
                piece.bank<host_kv_banked_d2h_banks && piece.bytes>0,
                "invalid banked D2H scatter descriptor");
            auto& plane=piece.key_plane?piece.page->k[piece.bank]:piece.page->v[piece.bank];
            require(exl3_host_kv_scatter_fits(plane.size()*sizeof(std::uint16_t),
                    piece.destination_byte_offset,host_kv_banked_d2h_staging->size(),
                    piece.staging_offset,piece.bytes),
                "banked D2H scatter descriptor exceeds storage");
        },[&](const auto& piece) noexcept {
            auto& plane=piece.key_plane?piece.page->k[piece.bank]:piece.page->v[piece.bank];
            std::memcpy(reinterpret_cast<std::byte*>(plane.data())+
                    piece.destination_byte_offset,
                static_cast<const std::byte*>(host_kv_banked_d2h_staging->data())+
                    piece.staging_offset,piece.bytes);
            host_kv.pinned_scatter_bytes+=piece.bytes;
        });
        host_kv_banked_d2h_scatter.clear();
        if(host_kv_cpu_profile)host_kv.pinned_scatter_cpu_ns+=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now()-scatter_started).count());
        ++host_kv.banked_d2h_drains;
    }

    void allocate(std::size_t bytes, void** pointer, const char* label) {
        Exl3LinearWorkspaceRequirements::allocate_owned(allocations,persistent_bytes,pointer,bytes,
            [&]{return make_private_allocation(bytes,label);});
    }

    bool native_mtp_hidden_capture_active() const noexcept {
        return native_mtp_hidden_capture &&
            native_mtp_hidden_capture->ptr != nullptr &&
            native_mtp_hidden_capture_capacity > 0 &&
            native_mtp_hidden_capture_model_identity != nullptr;
    }

    void invalidate_native_mtp_hidden_capture() {
        native_mtp_hidden_capture_first_position = -1;
        native_mtp_hidden_capture_valid_rows = 0;
        if (++native_mtp_hidden_capture_generation == 0)
            throw std::overflow_error("native MTP hidden capture generation exhausted");
    }

    void validate_native_mtp_hidden_capture_extent(int position, int rows) const {
        if (!native_mtp_hidden_capture_active()) return;
        require(!graph_capture_active && !graph_active,
            "native MTP hidden capture is unavailable during graph execution");
        require(position >= 0 && rows > 0 &&
                    position <= std::numeric_limits<int>::max() - rows,
                "native MTP hidden capture position extent is invalid");
        if (native_mtp_hidden_capture_valid_rows == 0) {
            require(native_mtp_hidden_capture_first_position == -1,
                "native MTP hidden capture empty-range metadata is invalid");
        } else {
            require(native_mtp_hidden_capture_first_position >= 0 &&
                        position == native_mtp_hidden_capture_first_position +
                            native_mtp_hidden_capture_valid_rows,
                    "native MTP hidden capture requires a contiguous forward range");
        }
        require(rows <= native_mtp_hidden_capture_capacity -
                    native_mtp_hidden_capture_valid_rows,
            "native MTP hidden capture capacity exhausted");
    }

    void note_native_mtp_hidden_capture_rows(int position, int rows) {
        require(native_mtp_hidden_capture_active(),
            "native MTP hidden capture is not prepared");
        require(position >= 0 && rows > 0 &&
                    rows <= native_mtp_hidden_capture_capacity -
                        native_mtp_hidden_capture_valid_rows,
            "native MTP hidden capture publication extent is invalid");
        if (native_mtp_hidden_capture_valid_rows == 0)
            native_mtp_hidden_capture_first_position = position;
        else require(position == native_mtp_hidden_capture_first_position +
                          native_mtp_hidden_capture_valid_rows,
                      "native MTP hidden capture publication is not contiguous");
        native_mtp_hidden_capture_valid_rows += rows;
        if (++native_mtp_hidden_capture_generation == 0)
            throw std::overflow_error("native MTP hidden capture generation exhausted");
    }

    void copy_native_mtp_hidden_capture_rows(const std::uint16_t* source,
                                             int position, int rows,
                                             cudaStream_t stream) {
        require(source != nullptr, "native MTP hidden capture source is null");
        validate_native_mtp_hidden_capture_extent(position, rows);
        const std::size_t row_bytes = static_cast<std::size_t>(kHidden) *
            sizeof(std::uint16_t);
        auto* destination = static_cast<std::uint16_t*>(
            native_mtp_hidden_capture->ptr) +
            static_cast<std::size_t>(native_mtp_hidden_capture_valid_rows) * kHidden;
        cuda_check(cudaMemcpyAsync(destination, source,
            static_cast<std::size_t>(rows) * row_bytes,
            cudaMemcpyDeviceToDevice, stream),
            "capture native MTP target hidden rows");
        note_native_mtp_hidden_capture_rows(position, rows);
    }

    int tap_index(int layer) const {
        for (std::size_t i = 0; i < kTapLayers.size(); ++i) if (kTapLayers[i] == layer) return static_cast<int>(i);
        return -1;
    }

    struct ProfileEvents {
        cudaEvent_t total_start = nullptr;
        cudaEvent_t total_end = nullptr;
        cudaEvent_t token_end = nullptr;
        cudaEvent_t embedding_start = nullptr;
        cudaEvent_t embedding_end = nullptr;
        cudaEvent_t layer_stack_start = nullptr;
        cudaEvent_t layer_stack_end = nullptr;
        cudaEvent_t final_norm_start = nullptr;
        cudaEvent_t final_norm_end = nullptr;
        cudaEvent_t lm_head_start = nullptr;
        cudaEvent_t lm_head_end = nullptr;
        std::array<cudaEvent_t, kLayers> layer_start{};
        std::array<cudaEvent_t, kLayers> layer_end{};

        void create() {
            auto make = [](cudaEvent_t* event) { cuda_check(cudaEventCreate(event), "create E4B1 profile event"); };
            make(&total_start); make(&total_end); make(&token_end);
            make(&embedding_start); make(&embedding_end);
            make(&layer_stack_start); make(&layer_stack_end);
            make(&final_norm_start); make(&final_norm_end);
            make(&lm_head_start); make(&lm_head_end);
            for (int i = 0; i < kLayers; ++i) { make(&layer_start[i]); make(&layer_end[i]); }
        }

        void destroy() noexcept {
            auto drop = [](cudaEvent_t& event) { if (event) cudaEventDestroy(event); event = nullptr; };
            drop(total_start); drop(total_end); drop(token_end);
            drop(embedding_start); drop(embedding_end);
            drop(layer_stack_start); drop(layer_stack_end);
            drop(final_norm_start); drop(final_norm_end);
            drop(lm_head_start); drop(lm_head_end);
            for (int i = 0; i < kLayers; ++i) { drop(layer_start[i]); drop(layer_end[i]); }
        }
    };

    static void record(cudaEvent_t event, cudaStream_t stream, const char* label) {
        cuda_check(cudaEventRecord(event, stream), label);
    }

    void validate_target_projection_execution(cudaStream_t stream) const {
        if (target_projection_timing && target_projection_timing->active())
            target_projection_timing->validate_execution_stream(stream);
    }

    void retain_host_kv_forward_stream(cudaStream_t stream) {
        if(host_kv.enabled) {
            require(!execution_stream_owner || !stream || stream==owned_execution_stream,
                "HostKV forward stream differs from retained stream owner");
            int device=-1;
            cuda_check(cudaGetDevice(&device),"HostKV forward allocation device");
            require((!host_kv_forward_used || device==host_kv_forward_device) &&
                (!host_kv_copy_stream || device==host_kv_copy_device),
                "HostKV forward device differs from retained owners");
            // A stream change must retire earlier work before replacing its
            // witness. Keep the last stream through post-forward caller copies.
            if(host_kv_forward_used && host_kv_forward_stream!=stream) {
                const auto error=host_kv_compute_retirement_failure_for_test?cudaErrorUnknown:
                    cudaStreamSynchronize(host_kv_forward_stream);
                if(error!=cudaSuccess)host_kv_failed=true;
                cuda_check(error,"retire prior HostKV forward stream");
            }
            host_kv_forward_stream=stream;
            host_kv_forward_used=true;
            host_kv_forward_device=device;
        }
    }

    HostKVGdnSegmentGraph& host_kv_gdn_segment_graph(int rows,int segment) {
        require(rows>=1 && rows<=host_kv_gdn_graph_row_shapes &&
                segment>=0 && segment<host_kv_gdn_segment_count,
            "HostKV GDN segment graph index");
        return host_kv_gdn_segment_graphs[
            static_cast<std::size_t>(rows-1)*host_kv_gdn_segment_count+segment];
    }

    const HostKVGdnSegmentGraph& host_kv_gdn_segment_graph(
        int rows,int segment) const {
        return const_cast<Impl*>(this)->host_kv_gdn_segment_graph(rows,segment);
    }

    void capture_host_kv_gdn_segment_graphs() {
        if(!host_kv_gdn_segment_graphs_enabled &&
           !ordinary_gdn_segment_graphs_enabled)return;
        const bool ordinary=ordinary_gdn_segment_graphs_enabled;
        require((ordinary && !host_kv.enabled) ||
                (!ordinary && host_kv.enabled),
            "GDN segment graph KV ownership mode");
        require(!oscar && !oscar_only && capture_taps &&
                !graph_active && !graph_capture_active &&
                !target_projection_timing && !target_projection_observer,
            "GDN segment graph requires pristine target context");
        bind_graph_device();
        cudaStream_t capture_stream=nullptr;
        cuda_check(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking),
            "create HostKV GDN segment graph capture stream");
        const auto started=std::chrono::steady_clock::now();
        try {
            for(int rows=1;rows<=host_kv_gdn_graph_row_shapes;++rows) {
                const bool preserve_m1_topology=rows>1;
                for(int segment=0;segment<host_kv_gdn_segment_count;++segment) {
                    const int first=segment*4;
                    auto& entry=host_kv_gdn_segment_graph(rows,segment);
                    require(gdn_layers[first] && gdn_layers[first+1] &&
                            gdn_layers[first+2] && full_layers[first+3],
                        "HostKV GDN segment graph layer topology");
                    entry.rows=rows;entry.first_layer=first;
                    for(int layer=first;layer<first+3;++layer)
                        gdn_layers[layer]->set_capture_active(true);
                    try {
                        entry.definition.capture(capture_stream,[&] {
                            std::uint16_t* current=hidden_a;
                            for(int layer=first;layer<first+3;++layer) {
                                auto* next=current==hidden_a?hidden_b:hidden_a;
                                gdn_layers[layer]->forward(current,next,rows,
                                    capture_stream,false,preserve_m1_topology,false);
                                const int tap=tap_index(layer);
                                if(tap>=0)cuda_check(cudaMemcpyAsync(taps[tap]->ptr,next,
                                    static_cast<std::size_t>(rows)*kHidden*sizeof(std::uint16_t),
                                    cudaMemcpyDeviceToDevice,capture_stream),
                                    "capture HostKV GDN segment hidden tap");
                                current=next;
                            }
                        });
                    } catch(...) {
                        for(int layer=first;layer<first+3;++layer)
                            gdn_layers[layer]->set_capture_active(false);
                        throw;
                    }
                    for(int layer=first;layer<first+3;++layer)
                        gdn_layers[layer]->set_capture_active(false);
                    entry.executable.instantiate(entry.definition);
                    entry.executable.upload(capture_stream);
                    ++host_kv_gdn_segment_graph_captures;
                    if(ordinary)count_ordinary_graph(
                        ordinary_graph_process_counters.gdn_segment_captures);
                }
            }
            cuda_check(cudaStreamSynchronize(capture_stream),
                "complete HostKV GDN segment graph preparation");
        } catch(...) {
            for(auto& layer:gdn_layers)if(layer)layer->set_capture_active(false);
            (void)cudaStreamSynchronize(capture_stream);
            (void)cudaStreamDestroy(capture_stream);
            throw;
        }
        cuda_check(cudaStreamDestroy(capture_stream),
            "destroy HostKV GDN segment graph capture stream");
        host_kv_gdn_segment_graph_capture_ms=
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
    }

    HostKVFullLayerGraph& host_kv_full_layer_graph(int rows,int segment) {
        require(rows>=1 && rows<=host_kv_full_graph_row_shapes &&
                segment>=0 && segment<host_kv_full_layer_count,
            "HostKV full-layer graph index");
        return host_kv_full_layer_graphs[
            static_cast<std::size_t>(rows-1)*host_kv_full_layer_count+segment];
    }

    void capture_host_kv_full_layer_graphs() {
        if(!host_kv_full_layer_graphs_enabled)return;
        require(host_kv.enabled && host_kv_gdn_segment_graphs_enabled &&
                !oscar && !oscar_only && capture_taps && !graph_active &&
                !graph_capture_active && !target_projection_timing &&
                !target_projection_observer && !coalesce_attention_input_mlp,
            "HostKV full-layer graphs require pristine exact target context");
        bind_graph_device();
        cudaStream_t capture_stream=nullptr;
        cuda_check(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking),
            "create HostKV full-layer graph capture stream");
        const auto started=std::chrono::steady_clock::now();
        try {
            for(int rows=1;rows<=host_kv_full_graph_row_shapes;++rows) {
                const bool preserve_m1_topology=rows>1;
                for(int segment=0;segment<host_kv_full_layer_count;++segment) {
                    const int layer=segment*4+3;
                    auto& entry=host_kv_full_layer_graph(rows,segment);
                    require(static_cast<bool>(full_layers[layer]),
                        "HostKV full-layer graph topology");
                    entry.rows=rows;entry.layer=layer;
                    full_layers[layer]->set_segmented_exact_prefix(nullptr,nullptr,0);
                    full_layers[layer]->set_mrope_positions(nullptr,rope_offset);
                    full_layers[layer]->set_capture_active(true);
                    const auto six_before=full_layers[layer]->
                        gqa_six_softmax_triple_value_launch_attempts();
                    const auto k6_stream_before=
                        full_layers[layer]->stream_reduction_calls();
                    const auto extended_stream_before=
                        full_layers[layer]->stream_reduction_calls(true);
                    const auto down_k6_before=
                        full_layers[layer]->target_down_k6_async_a_calls();
                    const auto small_k6_before=
                        full_layers[layer]->target_k6_small_m_async_a_calls();
                    const auto small_k7_before=
                        full_layers[layer]->target_k7_small_m_async_a_calls();
                    try {
                        entry.definition.capture(capture_stream,[&] {
                            full_layers[layer]->forward(hidden_b,hidden_a,rows,0,
                                capture_stream,false,preserve_m1_topology,false);
                        });
                    } catch(...) {
                        full_layers[layer]->set_capture_active(false);
                        throw;
                    }
                    full_layers[layer]->set_capture_active(false);
                    const auto accumulate=[&](std::uint64_t before,
                        std::uint64_t after,std::uint64_t& total,
                        const char* label) {
                        require(after>=before,
                            std::string("HostKV full-layer graph counter regressed: ")+label);
                        total+=after-before;
                    };
                    accumulate(six_before,full_layers[layer]->
                            gqa_six_softmax_triple_value_launch_attempts(),
                        host_kv_full_layer_graph_six_softmax_triple_captures,
                        "six-softmax/triple-values");
                    accumulate(k6_stream_before,
                        full_layers[layer]->stream_reduction_calls(),
                        host_kv_full_layer_graph_k6_stream_reduction_captures,
                        "K6 stream reduction");
                    accumulate(extended_stream_before,
                        full_layers[layer]->stream_reduction_calls(true),
                        host_kv_full_layer_graph_extended_stream_reduction_captures,
                        "extended stream reduction");
                    accumulate(down_k6_before,
                        full_layers[layer]->target_down_k6_async_a_calls(),
                        host_kv_full_layer_graph_target_down_k6_async_a_captures,
                        "down K6 async-A");
                    accumulate(small_k6_before,
                        full_layers[layer]->target_k6_small_m_async_a_calls(),
                        host_kv_full_layer_graph_target_k6_small_m_async_a_captures,
                        "small-M K6 async-A");
                    accumulate(small_k7_before,
                        full_layers[layer]->target_k7_small_m_async_a_calls(),
                        host_kv_full_layer_graph_target_k7_small_m_async_a_captures,
                        "small-M K7 async-A");
                    entry.executable.instantiate(entry.definition);
                    entry.executable.upload(capture_stream);
                    ++host_kv_full_layer_graph_captures;
                }
            }
            cuda_check(cudaStreamSynchronize(capture_stream),
                "complete HostKV full-layer graph preparation");
            require(host_kv_full_layer_graph_six_softmax_triple_captures==
                        host_kv_full_layer_graph_captures &&
                    host_kv_full_layer_graph_k6_stream_reduction_captures>0 &&
                    host_kv_full_layer_graph_extended_stream_reduction_captures>0 &&
                    host_kv_full_layer_graph_target_down_k6_async_a_captures>0 &&
                    host_kv_full_layer_graph_target_k6_small_m_async_a_captures>0 &&
                    host_kv_full_layer_graph_target_k7_small_m_async_a_captures>0,
                "HostKV full-layer graph did not preserve the retained fast dispatches: "
                "captures="+std::to_string(host_kv_full_layer_graph_captures)+
                " six="+std::to_string(host_kv_full_layer_graph_six_softmax_triple_captures)+
                " k6_stream="+std::to_string(host_kv_full_layer_graph_k6_stream_reduction_captures)+
                " extended="+std::to_string(host_kv_full_layer_graph_extended_stream_reduction_captures)+
                " down_k6="+std::to_string(host_kv_full_layer_graph_target_down_k6_async_a_captures)+
                " small_k6="+std::to_string(host_kv_full_layer_graph_target_k6_small_m_async_a_captures)+
                " small_k7="+std::to_string(host_kv_full_layer_graph_target_k7_small_m_async_a_captures));
        } catch(...) {
            for(auto& layer:full_layers)if(layer)layer->set_capture_active(false);
            (void)cudaStreamSynchronize(capture_stream);
            (void)cudaStreamDestroy(capture_stream);
            throw;
        }
        cuda_check(cudaStreamDestroy(capture_stream),
            "destroy HostKV full-layer graph capture stream");
        host_kv_full_layer_graph_capture_ms=
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
    }

    HostKVFullLayerGraph& ordinary_full_layer_graph(int rows,int segment) {
        require(rows>=1 && rows<=host_kv_full_graph_row_shapes &&
                segment>=0 && segment<host_kv_full_layer_count,
            "ordinary full-layer graph index");
        return ordinary_full_layer_graphs[
            static_cast<std::size_t>(rows-1)*host_kv_full_layer_count+segment];
    }

    void capture_ordinary_full_layer_graphs() {
        if(!ordinary_full_layer_graphs_enabled)return;
        require(!host_kv.enabled && !host_kv_full_layer_graphs_enabled &&
                !ordinary_mlp_tail_graphs_enabled && !oscar && !oscar_only &&
                !continuation_graph_b8_enabled && capture_taps &&
                !graph_active && !graph_capture_active &&
                !target_projection_timing && !target_projection_observer &&
                !coalesce_attention_input_mlp && !eager_mlp_gateup_concurrent,
            "ordinary full-layer graphs require pristine physical C1 context");
        bind_graph_device();
        cudaStream_t capture_stream=nullptr;
        cuda_check(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking),
            "create ordinary full-layer graph capture stream");
        const auto started=std::chrono::steady_clock::now();
        try {
            // One graph per full-attention layer removes the complete
            // attention-layer launch chain while position_device_ keeps the
            // live cache frontier dynamic at replay. The C1 shape is always
            // captured; verifier continuation rows 2..8 are captured when the
            // multirow family is admitted.
            // The segmented fused attention grid is frozen during capture.
            // Size it for this context's full capacity; the merge reads only
            // live segments from position_device_ during replay.
            ordinary_full_layer_graph_capture_position=max_context-1;
            require(ordinary_full_layer_graph_capture_position>0,
                "ordinary full-layer graph capture frontier");
            const int max_rows=ordinary_full_layer_multirow_graphs_enabled?
                host_kv_full_graph_row_shapes:1;
            for(int rows=1;rows<=max_rows;++rows)
            for(int segment=0;segment<host_kv_full_layer_count;++segment) {
                const int layer=segment*4+3;
                const int capture_position=max_context-rows;
                auto& entry=ordinary_full_layer_graph(rows,segment);
                require(static_cast<bool>(full_layers[layer]),
                    "ordinary full-layer graph topology");
                entry.rows=rows;entry.layer=layer;
                full_layers[layer]->set_segmented_exact_prefix(nullptr,nullptr,0);
                full_layers[layer]->set_mrope_positions(nullptr,rope_offset);
                full_layers[layer]->set_capture_active(true);
                full_layers[layer]->set_ordinary_full_layer_graph_capture(true);
                try {
                    entry.definition.capture(capture_stream,[&] {
                        full_layers[layer]->forward(hidden_b,hidden_a,rows,
                            capture_position,capture_stream,false,rows>1,false,nullptr);
                    });
                } catch(...) {
                    full_layers[layer]->set_capture_active(false);
                    full_layers[layer]->set_ordinary_full_layer_graph_capture(false);
                    throw;
                }
                full_layers[layer]->set_capture_active(false);
                full_layers[layer]->set_ordinary_full_layer_graph_capture(false);
                entry.executable.instantiate(entry.definition);
                entry.executable.upload(capture_stream);
                ++ordinary_full_layer_graph_captures;
                count_ordinary_graph(ordinary_graph_process_counters.full_layer_captures);
            }
            cuda_check(cudaStreamSynchronize(capture_stream),
                "complete ordinary full-layer graph preparation");
        } catch(...) {
            for(auto& layer:full_layers)if(layer) {
                layer->set_capture_active(false);
                layer->set_ordinary_full_layer_graph_capture(false);
            }
            (void)cudaStreamSynchronize(capture_stream);
            (void)cudaStreamDestroy(capture_stream);
            throw;
        }
        cuda_check(cudaStreamDestroy(capture_stream),
            "destroy ordinary full-layer graph capture stream");
        ordinary_full_layer_graph_capture_ms=
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
    }

    HostKVMlpTailGraph& host_kv_mlp_tail_graph(int rows,int segment) {
        require(rows>=1 && rows<=host_kv_full_graph_row_shapes &&
                segment>=0 && segment<host_kv_full_layer_count,
            "HostKV MLP-tail graph index");
        return host_kv_mlp_tail_graphs[
            static_cast<std::size_t>(rows-1)*host_kv_full_layer_count+segment];
    }

    void capture_host_kv_mlp_tail_graphs() {
        if(!host_kv_mlp_tail_graphs_enabled &&
           !ordinary_mlp_tail_graphs_enabled)return;
        const bool ordinary=ordinary_mlp_tail_graphs_enabled;
        require((ordinary && !host_kv.enabled) ||
                (!ordinary && host_kv.enabled),
            "MLP-tail graph KV ownership mode");
        require(!host_kv_full_layer_graphs_enabled &&
                !oscar && !oscar_only &&
                capture_taps && !graph_active && !graph_capture_active &&
                !target_projection_timing && !target_projection_observer &&
                !coalesce_attention_input_mlp && !eager_mlp_gateup_concurrent,
            "MLP-tail graphs require pristine exact target context");
        bind_graph_device();
        cudaStream_t capture_stream=nullptr;
        cuda_check(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking),
            "create HostKV MLP-tail graph capture stream");
        const auto started=std::chrono::steady_clock::now();
        try {
            for(int rows=1;rows<=host_kv_full_graph_row_shapes;++rows) {
                for(int segment=0;segment<host_kv_full_layer_count;++segment) {
                    const int layer=segment*4+3;
                    auto& entry=host_kv_mlp_tail_graph(rows,segment);
                    require(static_cast<bool>(full_layers[layer]),
                        "HostKV MLP-tail graph topology");
                    entry.rows=rows;entry.layer=layer;
                    full_layers[layer]->set_capture_active(true);
                    try {
                        entry.definition.capture(capture_stream,[&] {
                            full_layers[layer]->capture_mlp_tail_graph(
                                hidden_a,rows,capture_stream);
                        });
                    } catch(...) {
                        full_layers[layer]->set_capture_active(false);
                        throw;
                    }
                    full_layers[layer]->set_capture_active(false);
                    entry.executable.instantiate(entry.definition);
                    entry.executable.upload(capture_stream);
                    ++host_kv_mlp_tail_graph_captures;
                    if(ordinary)count_ordinary_graph(
                        ordinary_graph_process_counters.mlp_tail_captures);
                }
            }
            cuda_check(cudaStreamSynchronize(capture_stream),
                "complete HostKV MLP-tail graph preparation");
        } catch(...) {
            for(auto& layer:full_layers)if(layer)layer->set_capture_active(false);
            (void)cudaStreamSynchronize(capture_stream);
            (void)cudaStreamDestroy(capture_stream);
            throw;
        }
        cuda_check(cudaStreamDestroy(capture_stream),
            "destroy HostKV MLP-tail graph capture stream");
        host_kv_mlp_tail_graph_capture_ms=
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
    }
    struct HostKVForwardFailureScope {
        Impl& owner;
        int exceptions=std::uncaught_exceptions();
        ~HostKVForwardFailureScope() noexcept {
            if(owner.host_kv.enabled && std::uncaught_exceptions()>exceptions)
                owner.host_kv_failed=true;
        }
    };
    // Keep route admission separate from the workspace's own reconstruction
    // counters.  This is intentionally host-side telemetry: it proves which
    // wide prefill batches entered the same-weight candidate and how many
    // numeric projection calls actually returned from that workspace, while
    // the layer remains the exact/reference fallback for every other shape.
    struct FastSameWeightsPrefillTelemetryScope {
        Impl& owner;
        Exl3ReconstructGemmStats before{};
        bool active=false;
        FastSameWeightsPrefillTelemetryScope(Impl& value,bool candidate) noexcept
            : owner(value),active(candidate && value.numeric_prefill_projection_workspace!=nullptr) {
            if(active) {
                before=value.numeric_prefill_projection_workspace->stats();
                ++owner.fast_same_weights_fp16kv_prefill_stats.wide_prefill_forwards;
            }
        }
        ~FastSameWeightsPrefillTelemetryScope() noexcept {
            if(!active)return;
            const auto after=owner.numeric_prefill_projection_workspace->stats();
            if(after.calls<before.calls || after.rows<before.rows)return;
            owner.fast_same_weights_fp16kv_prefill_stats.numeric_dispatch_calls+=
                after.calls-before.calls;
            owner.fast_same_weights_fp16kv_prefill_stats.numeric_dispatch_rows+=
                after.rows-before.rows;
        }
    };
    void process_rows(const std::int64_t* ids, int rows, int position, cudaStream_t stream,
                      ProfileEvents* events = nullptr, bool include_embedding = true,
                      bool continuation_reference = false,
                      bool projection_timing_prevalidated = false,
                       bool final_row_only = false,
                       bool wide_prefill = false,const int* positions_xyz=nullptr,
                       int layer_begin=0,int layer_end=kLayers,
                       std::uint16_t* initial_hidden=nullptr,
                       std::uint16_t* alternate_hidden=nullptr,
                       bool skip_head=false,
                       const Exl3GdnLayer::BulkPrefillBuffers* gdn_prepared=nullptr) {
        require(!fast_prefill_failed,
                "failed layer-major prefill requires context reset");
        require(layer_begin>=0 && layer_begin<=layer_end && layer_end<=kLayers &&
                (!alternate_hidden || initial_hidden),
                "layer-major forward range and buffer contract");
        require(!gdn_prepared ||
                (layer_end==layer_begin+1 && gdn_layers[layer_begin] &&
                 gdn_prepared->rows==rows && !include_embedding),
                "bulk GDN prepared input requires one layer and matching rows");
        retain_host_kv_forward_stream(stream);
        const bool fast_same_weights_wide_candidate =
            fast_same_weights_fp16kv_prefill_enabled && wide_prefill && rows>=256 &&
            !graph_capture_active && !graph_active &&
            layer_begin==0 && layer_end==kLayers;
        FastSameWeightsPrefillTelemetryScope fast_same_weights_telemetry(
            *this,fast_same_weights_wide_candidate);
        if(fast_same_weights_telemetry.active)
            fast_same_weights_telemetry.owner.fast_same_weights_fp16kv_prefill_stats.wide_prefill_rows+=
                static_cast<std::uint64_t>(rows);
        const bool all_head_rows=continuation_reference && !final_row_only;
        const bool capture_native_mtp_rows =
            native_mtp_hidden_capture_active() && !graph_capture_active && !graph_active;
        if (capture_native_mtp_rows)
            validate_native_mtp_hidden_capture_extent(position, rows);
        const auto head_plan=Exl3HeadConsumerPlan::make(rows,
            all_head_rows?Exl3HeadConsumer::verification:Exl3HeadConsumer::root_only,!all_head_rows);
        require(!all_head_rows || (continuation && continuation->capacity>=head_plan.rows),
            "all-row head consumer lacks prepared output capacity");
        require(!oscar_only || oscar!=nullptr,"OSCAR-only context needs explicit OSCAR initialization");
        resident_exact_state_id=0;
        ExactPageExtension pending_host;
        bool use_banked_host_kv_d2h=false;
        int cached_rows=0,cacheable_rows=0;
        bool forward_publish_device_prefix=false;
        std::optional<Exl3DevicePrefixCache::Use> shared_prefix_use;
        if(device_prefix_shared && device_prefix)shared_prefix_use.emplace(device_prefix->try_use());
        const bool cache_usable=device_prefix && device_prefix->reusable() &&
            (!device_prefix_shared || (shared_prefix_use && static_cast<bool>(*shared_prefix_use)));
        if(device_prefix_shared && !cache_usable)++host_kv.shared_prefix_busy_fallbacks;
        if(host_kv.enabled) {
            require(!host_kv_failed && !oscar && !graph_active && !graph_capture_active &&
                position==exact_prefix_position,"host KV requires intact ordinary eager lineage");
            require(position>=0 && position<=max_context && rows>0 && rows<=max_context-position,
                "host KV forward interval exceeds prepared context");
            require(!registered_kv_uploader || host_kv_pinned_chunks,
                "registered HostKV requires pinned fallback before history submission");
            host_kv_failed=true;
            pending_host=extend_exact_pages(exact_prefix_pages,position,position+rows,
                host_kv_unique_tail_reuse,host_kv_cpu_profile?&host_kv:nullptr,0,request_metadata_reservation);
            if(cache_usable){
                cacheable_rows=std::min(position,device_prefix->capacity_tokens())/64*64;
                forward_publish_device_prefix=device_prefix_forward_publish &&
                    !device_prefix_shared && !segmented_device_prefix &&
                    position+rows<=device_prefix->capacity_tokens();
                cached_rows=device_prefix->matched_rows(exact_prefix_pages,position,
                    forward_publish_device_prefix);
                // Invalidate BEFORE any bank refill. A failed partial overwrite
                // cannot leave an old identity pointing at new/unfinished data.
                if(cached_rows<cacheable_rows)device_prefix->invalidate();
            }
            use_banked_host_kv_d2h=host_kv_banked_d2h &&
                rows<=host_kv_banked_d2h_rows && !pending_host.fresh.empty();
            if(use_banked_host_kv_d2h) {
                require(host_kv_banked_d2h_scatter.empty(),
                    "banked D2H descriptors survived prior publication");
                ++host_kv.banked_d2h_forwards;
                host_kv.banked_d2h_rows+=rows;
            }
        }
        const bool arm_captured_transaction_prefix=
            transaction && transaction->active && transaction->fresh_snapshot &&
            transaction->stream==stream;
        if (transaction && transaction->active) {
            transaction->fresh_snapshot = false;
            transaction->prefix_available = false;
            for (auto& layer : full_layers)
                if (layer) layer->invalidate_retained_prefix();
        }
        if (!projection_timing_prevalidated)
            validate_target_projection_execution(stream);
        if (include_embedding) {
            if (events) record(events->embedding_start, stream, "record E4B1 embedding start");
            std::unique_ptr<NvtxRange> embedding_range;
            if (events) embedding_range = std::make_unique<NvtxRange>("exl3.embedding");
            embedding_lookup_kernel<<<(rows * kHidden + 255) / 256, 256, 0, stream>>>(
                ids, model->embedding, hidden_a, rows);
            cuda_check(cudaGetLastError(), "launch E4A embedding lookup");
            if (events) record(events->embedding_end, stream, "record E4B1 embedding end");
            embedding_range.reset();
            if (capture_taps) {
                cuda_check(cudaMemcpyAsync(embedding_trace->ptr, hidden_a,
                    static_cast<std::size_t>(rows) * kHidden * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice, stream), "capture E4A embedding output");
                embedding_rows = rows;
                tap_rows = rows;
            }
        }
        std::uint16_t* current = initial_hidden ? initial_hidden : hidden_a;
        if (events) record(events->layer_stack_start, stream, "record E4B1 layer stack start");
        int host_bank=0;
        const bool stage_history=Exl3AttentionStageRoute{attention_staging_enabled,
            bool(attention_stage_storage),host_kv.enabled,bool(host_kv_copy_stream),
            bool(device_prefix),bool(shared_page_attention),bool(shared_page_copier),bool(positions_xyz),
            wide_prefill,bool(events),bool(layer_observer),graph_capture_active,graph_active,bool(oscar),
            position,rows,max_context,exact_prefix_pages.size()}.eligible();
        std::array<std::optional<Exl3AttentionStageCommand>,2> staged_bank;
        if(stage_history) {
            require(attention_stage_epoch!=UINT64_MAX,"attention stage epoch exhausted");
            ++attention_stage_epoch;
        }
        const bool gdn_segment_graphs_enabled=
            ((host_kv_gdn_segment_graphs_enabled && host_kv.enabled) ||
             (ordinary_gdn_segment_graphs_enabled && !host_kv.enabled));
        const bool use_host_kv_gdn_segment_graphs=
            gdn_segment_graphs_enabled && !oscar &&
            !graph_active && !graph_capture_active && !events && !layer_observer &&
            !target_projection_timing && !target_projection_observer &&
            !wide_prefill && !positions_xyz && !eager_mlp_gateup_concurrent &&
            rows>=1 && rows<=host_kv_gdn_graph_row_shapes &&
            (rows==1 || continuation_reference);
        const bool use_host_kv_full_layer_graphs=
            host_kv_full_layer_graphs_enabled &&
            use_host_kv_gdn_segment_graphs &&
            rows<=host_kv_full_graph_row_shapes;
        const bool use_ordinary_full_layer_graphs=
            ordinary_full_layer_graphs_enabled && !host_kv.enabled && !oscar &&
            !oscar_only && !graph_active && !graph_capture_active && !events &&
            !layer_observer && !target_projection_timing &&
            !target_projection_observer && !wide_prefill && !positions_xyz &&
            !eager_mlp_gateup_concurrent && !native_mtp_hidden_capture_active() &&
            ((!continuation_reference && rows==1 &&
              (position<=ordinary_full_layer_graph_capture_position ||
               (ordinary_full_layer_graph_extended_replay &&
                position+rows<=max_context))) ||
             (ordinary_full_layer_multirow_graphs_enabled &&
              continuation_reference && rows>=2 &&
              rows<=host_kv_full_graph_row_shapes &&
              position+rows<=max_context));
        if (rows==1 && std::getenv("NINFER_EXL3_ORDINARY_FULL_LAYER_GRAPH_DIAGNOSTIC"))
            std::fprintf(stderr,
                "ordinary_full_layer_graph_route position=%d max=%d enabled=%d extended=%d use=%d wide=%d graph=%d capture=%d events=%d observer=%d timing=%d projection=%d eager_mlp=%d continuation=%d native_mtp=%d\\n",
                position,max_context,ordinary_full_layer_graphs_enabled ? 1 : 0,
                ordinary_full_layer_graph_extended_replay ? 1 : 0,
                use_ordinary_full_layer_graphs ? 1 : 0,wide_prefill ? 1 : 0,
                graph_active ? 1 : 0,graph_capture_active ? 1 : 0,events ? 1 : 0,
                layer_observer ? 1 : 0,target_projection_timing ? 1 : 0,
                target_projection_observer ? 1 : 0,eager_mlp_gateup_concurrent ? 1 : 0,
                continuation_reference ? 1 : 0,native_mtp_hidden_capture_active() ? 1 : 0);
        const bool use_host_kv_mlp_tail_graphs=
            ((host_kv_mlp_tail_graphs_enabled && host_kv.enabled) ||
             (ordinary_mlp_tail_graphs_enabled && !host_kv.enabled)) && !oscar &&
            !graph_active && !graph_capture_active && !events &&
            !layer_observer && !target_projection_timing &&
            !target_projection_observer && !wide_prefill && !positions_xyz &&
            !eager_mlp_gateup_concurrent &&
            rows>=1 && rows<=host_kv_full_graph_row_shapes &&
            (rows==1 || continuation_reference);
        for (int layer = layer_begin; layer < layer_end; ++layer) {
            if(use_host_kv_gdn_segment_graphs && layer%4==0) {
                const int segment=layer/4;
                auto& graph=host_kv_gdn_segment_graph(rows,segment);
                require(current==hidden_a && graph.rows==rows &&
                        graph.first_layer==layer && graph.definition.ready() &&
                        graph.executable.ready(),
                    "HostKV GDN segment graph compatibility");
                const auto launch_started=host_kv_cpu_profile?
                    std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
                graph.executable.launch(stream);
                if(host_kv_cpu_profile)host_kv_gdn_segment_graph_launch_cpu_ns+=
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()-launch_started).count());
                ++host_kv_gdn_segment_graph_replays;
                if(!host_kv.enabled)count_ordinary_graph(
                    ordinary_graph_process_counters.gdn_segment_replays);
                const bool arm_retained=rows>1 && continuation_reference &&
                    arm_captured_transaction_prefix;
                if(arm_retained)for(int captured=layer;captured<layer+3;++captured)
                    gdn_layers[captured]->arm_captured_retained_prefix(
                        transaction->gdn_checkpoints[captured],rows,stream);
                current=hidden_b;
                layer+=2;
                continue;
            }
            std::uint16_t* next = alternate_hidden
                ? alternate_hidden : (current == hidden_a ? hidden_b : hidden_a);
            std::unique_ptr<NvtxRange> layer_range;
            if (events) {
                record(events->layer_start[layer], stream, "record E4B1 layer start");
                layer_range = std::make_unique<NvtxRange>(
                    std::string("exl3.layer.") + std::to_string(layer));
            }
            if (model->layers[layer].full_attention) {
                std::array<Exl3SharedAttentionPrefix,Exl3AttentionPageRanges::capacity> shared_attention{};
                Exl3AttentionPageRanges shared_ranges;
                std::optional<Exl3OwnedAttentionInputView> owned_attention;
                bool direct_staged_history=false;
                if(!host_kv.enabled && !oscar_only) {
                    require(cache_k[layer] && cache_v[layer],"ordinary attention missing local cache owners");
                    owned_attention.emplace(static_cast<const std::uint16_t*>(cache_k[layer]->ptr),
                        static_cast<const std::uint16_t*>(cache_v[layer]->ptr),max_context,cache_k[layer],cache_v[layer]);
                }
                full_layers[layer]->set_capture_active(graph_capture_active);
                full_layers[layer]->set_segmented_exact_prefix(nullptr,nullptr,0);
                full_layers[layer]->set_mrope_positions(positions_xyz,rope_offset);
                if(host_kv.enabled) {
                    owned_attention.emplace(static_cast<const std::uint16_t*>(host_layer_k->ptr),
                        static_cast<const std::uint16_t*>(host_layer_v->ptr),max_context,host_layer_k,host_layer_v);
                    // Prove both planes before choosing pinned, batch, ordinary
                    // copies or shared-page consumption; empty history is legal
                    // only at position zero. No transfer may repair a missing row.
                    exl3_require_attention_history(exact_prefix_pages,position,host_bank);
                    if(stage_history && (host_bank%2)==1 && (staged_bank[0] || staged_bank[1])) {
                        require(staged_bank[0] && staged_bank[1],"next-bank stage missing producer");
                        direct_staged_history=full_layers[layer]->supports_direct_staged_history(rows,position);
                        for(auto& command:staged_bank)command->await_producer();
                        const auto staged_planes=Exl3AttentionStageCommand::consume_pair(
                            *staged_bank[0],*staged_bank[1],host_bank);
                        for(unsigned plane=0;plane<2;++plane) {
                            auto& command=*staged_bank[plane];
                            auto* source=staged_planes[plane];
                            if(direct_staged_history)continue;
                            cuda_check(cudaMemcpyAsync(plane?host_layer_v->ptr:host_layer_k->ptr,source,
                                static_cast<std::size_t>(position)*2048,cudaMemcpyDeviceToDevice,stream),
                                "consume staged next-bank history");
                            command.finish_consumer(stream);staged_bank[plane].reset();
                            host_kv.attention_stage_consumed_bytes+=static_cast<std::uint64_t>(position)*2048;
                        }
                        if(direct_staged_history) {
                            full_layers[layer]->set_direct_staged_history(
                                staged_planes[0],staged_planes[1],position);
                            owned_attention->append({staged_planes[0],staged_planes[1],0,position},attention_stage_storage,position);
                        } else {
                            ++host_kv.attention_stage_banks;
                            attention_stage_history->clear();
                        }
                    } else if(host_kv_pinned_chunks && !exact_prefix_pages.empty()) {
                        int upload_first=cached_rows;
                        if(shared_page_attention && !cached_rows && !positions_xyz && !wide_prefill &&
                           !events && !graph_capture_active && position>=64 &&
                           full_layers[layer]->supports_segmented_exact_prefix()) {
                            try {
                                for(auto it=exact_prefix_pages.rbegin();it!=exact_prefix_pages.rend();++it) {
                                    if(shared_ranges.count==Exl3AttentionPageRanges::capacity)break;
                                    if((*it)->rows!=64 || (*it)->first>position-64)continue;
                                    auto acquired=shared_page_attention(*it,host_bank,position,host_kv_copy_stream,host_kv_compute_ready);
                                    if(!acquired.rows)continue;
                                    shared_attention[shared_ranges.count]=acquired;
                                    shared_ranges.append(acquired.k,acquired.v,acquired.first,acquired.rows,position);
                                }
                            }
                            catch(...) {++host_kv.shared_page_failures;throw;}
                            if(shared_ranges.count) {
                                full_layers[layer]->set_segmented_exact_pages(shared_ranges,position);
                            }
                        }
                        if(cached_rows){
                            const auto bytes=std::size_t(cached_rows)*1024*2;
                            if(forward_publish_device_prefix && cached_rows>cacheable_rows)
                                host_kv.device_prefix_partial_hit_bytes+=
                                    std::uint64_t(cached_rows-cacheable_rows)*1024*2*2;
                            if(device_prefix_shared)host_kv.shared_prefix_hit_bytes+=bytes*2;
                            if(segmented_device_prefix && full_layers[layer]->supports_segmented_exact_prefix()) {
                                full_layers[layer]->set_segmented_exact_prefix(
                                    static_cast<const std::uint16_t*>(device_prefix->plane(host_bank,true)),
                                    static_cast<const std::uint16_t*>(device_prefix->plane(host_bank,false)),cached_rows);
                                owned_attention->append({static_cast<const std::uint16_t*>(device_prefix->plane(host_bank,true)),
                                    static_cast<const std::uint16_t*>(device_prefix->plane(host_bank,false)),0,cached_rows},
                                    device_prefix,position);
                                host_kv.device_prefix_segmented_bytes+=bytes*2;
                            } else {
                                cuda_check(cudaMemcpyAsync(host_layer_k->ptr,device_prefix->plane(host_bank,true),bytes,cudaMemcpyDeviceToDevice,host_kv_copy_stream),"reuse represented prefix K");
                                cuda_check(cudaMemcpyAsync(host_layer_v->ptr,device_prefix->plane(host_bank,false),bytes,cudaMemcpyDeviceToDevice,host_kv_copy_stream),"reuse represented prefix V");
                            }
                            host_kv.device_prefix_hit_bytes+=bytes*2;
                        }
                        submit_host_kv_pinned_plane(exact_prefix_pages,host_bank,
                            host_layer_k->ptr,true,position,"upload pinned exact host K chunks",upload_first,
                            &shared_ranges,wide_prefill && host_kv_prefill_pinned_batch &&
                                bool(registered_kv_uploader));
                        submit_host_kv_pinned_plane(exact_prefix_pages,host_bank,
                            host_layer_v->ptr,false,position,"upload pinned exact host V chunks",upload_first,
                            &shared_ranges,wide_prefill && host_kv_prefill_pinned_batch &&
                                bool(registered_kv_uploader));
                        if(cacheable_rows>cached_rows){
                            const auto offset=std::size_t(cached_rows)*1024*2,bytes=std::size_t(cacheable_rows-cached_rows)*1024*2;
                            cuda_check(cudaMemcpyAsync(static_cast<std::byte*>(device_prefix->plane(host_bank,true))+offset,static_cast<std::byte*>(host_layer_k->ptr)+offset,bytes,cudaMemcpyDeviceToDevice,host_kv_copy_stream),"fill represented prefix K");
                            cuda_check(cudaMemcpyAsync(static_cast<std::byte*>(device_prefix->plane(host_bank,false))+offset,static_cast<std::byte*>(host_layer_v->ptr)+offset,bytes,cudaMemcpyDeviceToDevice,host_kv_copy_stream),"fill represented prefix V");
                            host_kv.device_prefix_fill_bytes+=bytes*2;
                        }
                        cuda_check(cudaEventRecord(host_kv_h2d_ready,host_kv_copy_stream),
                            "record exact host KV pinned H2D completion");
                        cuda_check(cudaStreamWaitEvent(stream,host_kv_h2d_ready,0),
                            "wait exact host KV pinned H2D completion");
                    } else if(host_kv_batch_copy && !exact_prefix_pages.empty()) {
                        begin_host_kv_batch_metadata(exact_prefix_pages.size());
                        auto& destinations=host_kv_batch_destinations;
                        auto& sources=host_kv_batch_sources;auto& sizes=host_kv_batch_sizes;
                        for(const auto& page:exact_prefix_pages) {
                            const auto append=[&](void* destination,Exl3ExactKVExtent::Plane plane) {
                                const auto extent=Exl3ExactKVExtent::view(page,host_bank,plane,position);
                                if(!extent.rows())return;
                                const auto bytes=extent.bytes();
                                destinations.push_back(static_cast<std::uint16_t*>(destination)+
                                    extent.destination_offset(max_context));
                                sources.push_back(extent.data());sizes.push_back(bytes);
                                host_kv.h2d_bytes+=bytes;++host_kv.transfer_calls;
                            };
                            append(host_layer_k->ptr,Exl3ExactKVExtent::Plane::key);
                            append(host_layer_v->ptr,Exl3ExactKVExtent::Plane::value);
                        }
                        submit_host_kv_batch(destinations,sources,sizes,
                            cudaMemcpySrcAccessOrderAny,"batch full required layer KV");
                        cuda_check(cudaEventRecord(host_kv_h2d_ready,host_kv_copy_stream),
                            "record exact host KV batch H2D completion");
                        cuda_check(cudaStreamWaitEvent(stream,host_kv_h2d_ready,0),
                            "wait exact host KV batch H2D completion");
                    } else for(const auto& page:exact_prefix_pages) {
                        const auto upload=[&](void* destination,Exl3ExactKVExtent::Plane plane) {
                            const auto extent=Exl3ExactKVExtent::view(page,host_bank,plane,position);
                            if(!extent.rows())return;
                            const auto bytes=extent.bytes();
                            cuda_check(cudaMemcpyAsync(static_cast<std::uint16_t*>(destination)+
                                extent.destination_offset(max_context),extent.data(),bytes,
                                cudaMemcpyHostToDevice,stream),"stream full required layer KV");
                            ++host_kv.copy_submissions;
                            if(!host_kv_batch_sync)
                                cuda_check(cudaStreamSynchronize(stream),"complete required layer KV transfer");
                            host_kv.h2d_bytes+=bytes;++host_kv.transfer_calls;
                        };
                        upload(host_layer_k->ptr,Exl3ExactKVExtent::Plane::key);
                        upload(host_layer_v->ptr,Exl3ExactKVExtent::Plane::value);
                    }
                }
                full_layers[layer]->set_mrope_positions(positions_xyz,rope_offset);
                if(owned_attention) {
                    for(int index=0;index<shared_ranges.count;++index)
                        owned_attention->append(shared_ranges.ranges[index],shared_attention[index].owner.lock(),position);
                    // The upload/copy branches above skip precisely these shared
                    // ranges. Bind their local complement before any attention
                    // kernel can observe the representation.
                    owned_attention->append_local_complement(position);
                    owned_attention->require_geometry(position,rows);
                }
                if(shared_ranges.count) {
                    try {shared_page_attention_completion(std::span(shared_attention.data(),shared_ranges.count),stream,
                        Exl3TextContext::SharedPageAttentionPhase::before_compute);}
                    catch(...) {++host_kv.shared_page_failures;throw;}
                }
                if(stage_history && (host_bank%2)==0 && host_bank+1<16) {
                    auto registration_reads=attention_registration_cache
                        ? Exl3KVRegistrationCache::acquire_external_read_pair(attention_registration_cache)
                        : std::optional<std::array<Exl3KVRegistrationCache::ExternalRead,2>>{};
                    // No producer or history binding exists on contention. The
                    // next odd bank therefore takes the ordinary upload branch.
                    if(!attention_registration_cache || registration_reads) {
                    auto history=attention_stage_history->bind(exact_prefix_pages,position,host_bank+1);
                    Exl3AttentionStageCommand::require_history_pair(attention_stages,history,host_bank+1,position,
                        attention_stage_storage,attention_stage_epoch,attention_stage_epoch,registration_reads);
                    for(unsigned plane=0;plane<2;++plane) {
                        unsigned failure=0;
                        const auto selected=attention_staging_failure_for_test;
                        if(selected && plane==(selected-1)/5) {
                            failure=1+(selected-1)%5;
                            attention_staging_failure_for_test=0;
                        }
                        auto registration_read=registration_reads
                            ? std::optional<Exl3KVRegistrationCache::ExternalRead>(std::move((*registration_reads)[plane]))
                            : std::optional<Exl3KVRegistrationCache::ExternalRead>{};
                        const auto prefix_bytes=static_cast<std::size_t>(cached_rows)*2048;
                        if(prefix_bytes) {
                            cuda_check(cudaMemcpyAsync(attention_stage_storage->plane(plane),
                                device_prefix->plane(host_bank+1,plane==0),prefix_bytes,
                                cudaMemcpyDeviceToDevice,host_kv_copy_stream),
                                "seed staged next-bank device prefix");
                            host_kv.device_prefix_hit_bytes+=prefix_bytes;
                        }
                        staged_bank[plane].emplace(Exl3AttentionStageCommand::submit_history(attention_stages,history,
                            host_bank+1,plane?Exl3ExactKVExtent::Plane::value:Exl3ExactKVExtent::Plane::key,position,
                            attention_stage_storage,attention_stage_epoch,attention_stage_epoch,host_kv_copy_stream,{},failure,
                            std::move(registration_read),cached_rows));
                        const auto bytes=static_cast<std::uint64_t>(position)*2048;
                        std::uint64_t suffix_bytes=0,suffix_submissions=0;
                        for(const auto& page:*history) {
                            const auto extent=Exl3ExactKVExtent::view(page,host_bank+1,
                                plane?Exl3ExactKVExtent::Plane::value:Exl3ExactKVExtent::Plane::key,
                                position,cached_rows);
                            if(extent.rows()) {suffix_bytes+=extent.bytes();++suffix_submissions;}
                        }
                        host_kv.attention_stage_upload_bytes+=bytes;
                        host_kv.h2d_bytes+=suffix_bytes;
                        host_kv.copy_submissions+=suffix_submissions;
                        host_kv.transfer_calls+=suffix_submissions;
                    }
                    }
                }
                const bool replay_ordinary_full_layer_graph=
                    use_ordinary_full_layer_graphs && !direct_staged_history &&
                    shared_ranges.count==0;
                const bool replay_full_layer_graph=
                    use_host_kv_full_layer_graphs && !direct_staged_history &&
                    shared_ranges.count==0;
                if(replay_ordinary_full_layer_graph) {
                    const int segment=layer/4;
                    auto& graph=ordinary_full_layer_graph(rows,segment);
                    require(current==hidden_b && next==hidden_a &&
                            graph.rows==rows && graph.layer==layer &&
                            graph.definition.ready() && graph.executable.ready(),
                        "ordinary full-layer graph compatibility");
                    // The captured full forward consumes any retained-prefix
                    // capability just as the eager forward does.  Keep the
                    // ordinary physical-KV graph out of transaction/reference
                    // continuation state by construction.
                    full_layers[layer]->invalidate_retained_prefix();
                    graph.executable.launch(stream);
                    ++ordinary_full_layer_graph_replays;
                    count_ordinary_graph(ordinary_graph_process_counters.full_layer_replays);
                    // Replay bypasses the eager host-side arming of the
                    // retained-prefix capability; reproduce it exactly as the
                    // HostKV continuation graphs do.
                    if(rows>1 && continuation_reference &&
                       arm_captured_transaction_prefix)
                        full_layers[layer]->arm_captured_retained_prefix(
                            rows,position,stream);
                    if (rows==1 &&
                        std::getenv("NINFER_EXL3_ORDINARY_FULL_LAYER_GRAPH_DIAGNOSTIC"))
                        std::fprintf(stderr,
                            "ordinary_full_layer_graph_replay position=%d layer=%d count=%llu\\n",
                            position,layer,
                            static_cast<unsigned long long>(ordinary_full_layer_graph_replays));
                } else if(replay_full_layer_graph) {
                    const int segment=layer/4;
                    auto& graph=host_kv_full_layer_graph(rows,segment);
                    require(current==hidden_b && next==hidden_a &&
                            graph.rows==rows && graph.layer==layer &&
                            graph.definition.ready() && graph.executable.ready(),
                        "HostKV full-layer graph compatibility");
                    // forward() always consumes any previous retained-prefix
                    // capability before launching this layer.  CUDA replay
                    // bypasses that host-side transition, so reproduce it
                    // explicitly before optionally arming the new transaction.
                    full_layers[layer]->invalidate_retained_prefix();
                    const auto launch_started=host_kv_cpu_profile?
                        std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
                    graph.executable.launch(stream);
                    if(host_kv_cpu_profile)host_kv_full_layer_graph_launch_cpu_ns+=
                        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now()-launch_started).count());
                    ++host_kv_full_layer_graph_replays;
                    if(rows>1 && continuation_reference &&
                       arm_captured_transaction_prefix)
                        full_layers[layer]->arm_captured_retained_prefix(
                            rows,position,stream);
                } else {
                    DecodeGraphExecutable* mlp_tail_graph=nullptr;
                    if(use_host_kv_mlp_tail_graphs) {
                        const int segment=layer/4;
                        auto& graph=host_kv_mlp_tail_graph(rows,segment);
                        require(current==hidden_b && next==hidden_a &&
                                graph.rows==rows && graph.layer==layer &&
                                graph.definition.ready() &&
                                graph.executable.ready(),
                            "HostKV MLP-tail graph compatibility");
                        mlp_tail_graph=&graph.executable;
                    }
                    full_layers[layer]->forward(
                        current,next,rows,position,stream,false,
                        continuation_reference,wide_prefill,mlp_tail_graph);
                    if(mlp_tail_graph) {
                        ++host_kv_mlp_tail_graph_replays;
                        if(!host_kv.enabled)count_ordinary_graph(
                            ordinary_graph_process_counters.mlp_tail_replays);
                    }
                }
                if(direct_staged_history) {
                    // Both planes remain leased through the complete attention
                    // consumer, not merely through its first score/head launch.
                    for(auto& command:staged_bank) {
                        command->finish_consumer(stream);command.reset();
                        host_kv.attention_stage_direct_bytes+=static_cast<std::uint64_t>(position)*2048;
                    }
                    full_layers[layer]->set_direct_staged_history(nullptr,nullptr,0);
                    ++host_kv.attention_stage_banks;
                    attention_stage_history->clear();
                }
                if(shared_ranges.count) {
                    try {shared_page_attention_completion(std::span(shared_attention.data(),shared_ranges.count),stream,
                        Exl3TextContext::SharedPageAttentionPhase::final_use);}
                    catch(...) {++host_kv.shared_page_failures;throw;}
                    ++host_kv.shared_page_attention_groups;
                    host_kv.shared_page_attention_pages+=shared_ranges.count;
                    host_kv.shared_page_attention_max_pages=std::max(host_kv.shared_page_attention_max_pages,
                        static_cast<std::uint64_t>(shared_ranges.count));
                    for(int index=0;index<shared_ranges.count;++index)
                        host_kv.shared_page_attention_bytes+=std::uint64_t(shared_attention[index].rows)*1024*2*2;
                    full_layers[layer]->set_segmented_exact_prefix(nullptr,nullptr,0);
                }
                if(host_kv.enabled) {
                    int download_end=position;
                    for(const auto& page:pending_host.fresh) {
                        require(page && exl3_host_kv_append_download_page(page->first,page->rows,
                            position,max_context,page->k[host_bank].size(),page->v[host_bank].size(),download_end),
                            "HostKV download page geometry/planes invalid");
                    }
                    require(download_end==position+rows,"HostKV download pages omit or exceed new suffix");
                    if(forward_publish_device_prefix) {
                        // A matching partial tag proves [page_first,position)
                        // already resident.  A miss republishes the complete
                        // current page so no stale bytes can be tagged valid.
                        const int copy_first=cached_rows==position?position:(position/64)*64;
                        const auto copy_bytes=static_cast<std::size_t>(
                            position+rows-copy_first)*1024*2;
                        cuda_check(cudaEventRecord(host_kv_compute_ready,stream),
                            "record exact HostKV rows for represented forward publication");
                        cuda_check(cudaStreamWaitEvent(host_kv_copy_stream,
                            host_kv_compute_ready,0),
                            "wait exact HostKV rows for represented forward publication");
                        const auto copy=[&](bool key) {
                            cuda_check(cudaMemcpyAsync(
                                static_cast<std::byte*>(device_prefix->plane(host_bank,key))+
                                    static_cast<std::size_t>(copy_first)*1024*2,
                                static_cast<const std::byte*>(key?host_layer_k->ptr:host_layer_v->ptr)+
                                    static_cast<std::size_t>(copy_first)*1024*2,
                                copy_bytes,cudaMemcpyDeviceToDevice,host_kv_copy_stream),
                                key?"publish exact HostKV forward K rows":
                                    "publish exact HostKV forward V rows");
                        };
                        copy(true);copy(false);
                        host_kv.device_prefix_forward_publish_bytes+=copy_bytes*2;
                    }
                    if(host_kv_pinned_d2h && !pending_host.fresh.empty()) {
                        cuda_check(cudaEventRecord(host_kv_compute_ready,stream),
                            "record exact host KV layer completion");
                        cuda_check(cudaStreamWaitEvent(host_kv_copy_stream,host_kv_compute_ready,0),
                            "wait exact host KV layer completion");
                        if(use_banked_host_kv_d2h) {
                            submit_host_kv_banked_d2h_plane(pending_host.fresh,host_bank,
                                host_layer_k->ptr,true,position,rows,
                                "bank authoritative new K suffix");
                            submit_host_kv_banked_d2h_plane(pending_host.fresh,host_bank,
                                host_layer_v->ptr,false,position,rows,
                                "bank authoritative new V suffix");
                        } else {
                            receive_host_kv_pinned_plane(pending_host.fresh,host_bank,
                                host_layer_k->ptr,true,position,
                                "download pinned authoritative new K chunks");
                            receive_host_kv_pinned_plane(pending_host.fresh,host_bank,
                                host_layer_v->ptr,false,position,
                                "download pinned authoritative new V chunks");
                        }
                    } else if(host_kv_batch_copy && !pending_host.fresh.empty()) {
                        cuda_check(cudaEventRecord(host_kv_compute_ready,stream),
                            "record exact host KV layer completion");
                        cuda_check(cudaStreamWaitEvent(host_kv_copy_stream,host_kv_compute_ready,0),
                            "wait exact host KV layer completion");
                        begin_host_kv_batch_metadata(pending_host.fresh.size());
                        auto& destinations=host_kv_batch_destinations;
                        auto& sources=host_kv_batch_sources;auto& sizes=host_kv_batch_sizes;
                        for(const auto& page:pending_host.fresh) {
                            const int skip=std::clamp(position-page->first,0,page->rows);
                            const std::size_t bytes=static_cast<std::size_t>(page->rows-skip)*1024*2;
                            if(!bytes)continue;
                            const auto append=[&](auto& plane,const void* source) {
                                destinations.push_back(plane.data()+skip*1024);
                                sources.push_back(static_cast<const std::uint16_t*>(source)+
                                    static_cast<std::size_t>(page->first+skip)*1024);
                                sizes.push_back(bytes);host_kv.d2h_bytes+=bytes;++host_kv.transfer_calls;
                            };
                            append(page->k[host_bank],host_layer_k->ptr);
                            append(page->v[host_bank],host_layer_v->ptr);
                        }
                        if(!sizes.empty())submit_host_kv_batch(destinations,sources,sizes,
                            cudaMemcpySrcAccessOrderStream,"batch authoritative new KV");
                    } else for(const auto& page:pending_host.fresh) {
                            const int skip=std::clamp(position-page->first,0,page->rows);
                            const std::size_t bytes=static_cast<std::size_t>(page->rows-skip)*1024*2;
                            if(!bytes)continue;
                            const auto download=[&](auto& plane,const void* source) {
                                cuda_check(cudaMemcpyAsync(plane.data()+skip*1024,
                                    static_cast<const std::uint16_t*>(source)+static_cast<std::size_t>(page->first+skip)*1024,
                                    bytes,cudaMemcpyDeviceToHost,stream),"stream authoritative new KV");
                                ++host_kv.copy_submissions;
                                if(!host_kv_batch_sync)
                                    cuda_check(cudaStreamSynchronize(stream),"complete new authoritative KV transfer");
                                host_kv.d2h_bytes+=bytes;++host_kv.transfer_calls;
                            };
                        download(page->k[host_bank],host_layer_k->ptr);
                        download(page->v[host_bank],host_layer_v->ptr);
                        }
                    ++host_bank;
                }
            } else {
                gdn_layers[layer]->set_capture_active(graph_capture_active);
                gdn_layers[layer]->forward(current, next, rows, stream, false,
                                            continuation_reference, wide_prefill,
                                            gdn_prepared);
            }
            if (layer_observer) {
                require(!graph_capture_active && !graph_active,
                        "layer observer requires eager execution");
                Exl3LayerObservation observation;
                observation.layer = layer; observation.position = position;
                observation.rows = rows; observation.output = next;
                observation.stream = stream;
                if (full_layers[layer]) {
                    observation.attention = full_layers[layer]->trace();
                    observation.k_cache = host_kv.enabled
                        ? static_cast<const std::uint16_t*>(host_layer_k->ptr) : nullptr;
                    observation.v_cache = host_kv.enabled
                        ? static_cast<const std::uint16_t*>(host_layer_v->ptr) : nullptr;
                } else observation.gdn = gdn_layers[layer]->trace();
                layer_observer(observation, layer_observer_user);
            }
            if (events) {
                record(events->layer_end[layer], stream, "record E4B1 layer end");
                layer_range.reset();
            }
            const int tap = tap_index(layer);
            if (capture_taps && tap >= 0) {
                cuda_check(cudaMemcpyAsync(taps[tap]->ptr, next,
                    static_cast<std::size_t>(rows) * kHidden * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice, stream), "capture E4A hidden tap");
            }
            current = next;
        }
        if (events) record(events->layer_stack_end, stream, "record E4B1 layer stack end");
        if (skip_head) {
            require(!host_kv.enabled && !forward_publish_device_prefix &&
                    !events && !graph_active && !graph_capture_active,
                    "layer-major partial forward requires ordinary eager ownership");
            return;
        }
        if(forward_publish_device_prefix) {
            // Join the copy stream once, after all 16 layer publications.  The
            // existing final forward synchronize then proves represented bytes
            // complete before weak identities are published.
            cuda_check(cudaEventRecord(host_kv_h2d_ready,host_kv_copy_stream),
                "record exact HostKV represented forward publication completion");
            cuda_check(cudaStreamWaitEvent(stream,host_kv_h2d_ready,0),
                "wait exact HostKV represented forward publication completion");
        }
        if (events) record(events->final_norm_start, stream, "record E4B1 final norm start");
        if (all_head_rows) {
            auto& scratch = *continuation;
            for (int row = 0; row < rows; ++row) {
                final_rms_norm_kernel<<<1, 512, 0, stream>>>(
                    current + static_cast<std::size_t>(row) * kHidden,
                    model->final_norm,
                    static_cast<std::uint16_t*>(scratch.final_norm->ptr) +
                        static_cast<std::size_t>(row) * kHidden);
                cuda_check(cudaGetLastError(), "launch P2 continuation final RMSNorm");
            }
            if (capture_native_mtp_rows)
                copy_native_mtp_hidden_capture_rows(
                    static_cast<const std::uint16_t*>(scratch.final_norm->ptr),
                    position, rows, stream);
        } else if (capture_native_mtp_rows) {
            auto* captured = static_cast<std::uint16_t*>(
                native_mtp_hidden_capture->ptr) +
                static_cast<std::size_t>(native_mtp_hidden_capture_valid_rows) * kHidden;
            for (int row = 0; row < rows; ++row) {
                final_rms_norm_kernel<<<1, 512, 0, stream>>>(
                    current + static_cast<std::size_t>(row) * kHidden,
                    model->final_norm,
                    captured + static_cast<std::size_t>(row) * kHidden);
                cuda_check(cudaGetLastError(), "launch native MTP capture final RMSNorm");
            }
            cuda_check(cudaMemcpyAsync(
                           final_norm,
                           captured + static_cast<std::size_t>(head_plan.first_row) * kHidden,
                           static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToDevice, stream),
                       "retain native MTP capture root final norm");
            note_native_mtp_hidden_capture_rows(position, rows);
        } else {
            final_rms_norm_kernel<<<1, 512, 0, stream>>>(
                current + static_cast<std::size_t>(head_plan.first_row) * kHidden,
                model->final_norm, final_norm);
            cuda_check(cudaGetLastError(), "launch E4A final RMSNorm");
        }
        // Native MTP consumes the target's final-normalized hidden h_t, not
        // the residual immediately before the target final norm. Keep only a
        // transient source identity here; target_hidden_handoff() copies the
        // row(s) into independently owned storage before publishing them.
        if (!graph_capture_active) {
            if (all_head_rows) {
                last_hidden_source = static_cast<const std::uint16_t*>(
                    continuation->final_norm->ptr);
                last_hidden_source_rows = rows;
                last_hidden_source_first_row = 0;
            } else {
                last_hidden_source = final_norm;
                last_hidden_source_rows = 1;
                last_hidden_source_first_row = head_plan.first_row;
            }
            last_hidden_source_position = position;
            if (++last_hidden_generation == 0)
                throw std::overflow_error("native MTP hidden generation exhausted");
        }
        if (events) record(events->final_norm_end, stream, "record E4B1 final norm end");
        if (events) record(events->lm_head_start, stream, "record E4B1 LM head start");
        if (all_head_rows) {
            auto& scratch = *continuation;
            auto* norm = static_cast<const std::uint16_t*>(scratch.final_norm->ptr);
            auto* all_logits = static_cast<std::uint16_t*>(scratch.logits->ptr);
            const char* dispatch = scratch.head_workspace->dispatch_name(
                model->lm_head_metadata, rows);
            const bool small_m =
                std::strcmp(dispatch, "h6_small_m_single_split") == 0;
            const int timing_slot = target_projection_timing
                ? target_projection_timing->begin(
                      -1, Exl3TargetProjectionOperator::lm_head, rows,
                      model->lm_head_metadata.K, model->lm_head_metadata.in_features,
                      model->lm_head_metadata.out_features,
                      small_m ? Exl3TargetProjectionTopology::small_m_single_split
                              : Exl3TargetProjectionTopology::m1_per_row,
                      small_m ? 1 : rows, stream)
                : -1;
            const bool shared_head=small_m && shared_head_executor && !events && !target_projection_timing &&
                !graph_active && !graph_capture_active && !oscar && rows>=1 && rows<=8 &&
                target_shared_admission(Exl3TargetSharedFamily::head,model->lm_head_metadata).has_value() &&
                shared_head_executor(Exl3TargetQContinuation{model->lm_head,model->lm_head_metadata,
                    norm,all_logits,rows,0,64,stream,Exl3TargetSharedFamily::head});
            if (!shared_head && small_m) {
                scratch.head_workspace->forward(model->lm_head, model->lm_head_metadata,
                                                norm, all_logits, rows, stream);
            } else if(!shared_head) {
                // Explicit H6 opt-out retains the safe, qualified M1 head route.
                for (int row = 0; row < rows; ++row) {
                    head_workspace->forward(model->lm_head, model->lm_head_metadata,
                                            norm + static_cast<std::size_t>(row) * kHidden,
                                            all_logits + static_cast<std::size_t>(row) * kVocab,
                                            1, stream);
                }
            }
            if (target_projection_timing)
                target_projection_timing->end(timing_slot, stream);
            cuda_check(cudaGetLastError(), "launch P2 continuation H6 LM head");
            cuda_check(cudaMemcpyAsync(logits,
                all_logits + static_cast<std::size_t>(head_plan.root_row) * kVocab,
                kVocab * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice, stream),
                "retain P2 continuation final-row logits");
        } else {
            const int timing_slot = target_projection_timing
                ? target_projection_timing->begin(
                      -1, Exl3TargetProjectionOperator::lm_head, 1,
                      model->lm_head_metadata.K, model->lm_head_metadata.in_features,
                      model->lm_head_metadata.out_features,
                      Exl3TargetProjectionTopology::m1, 1, stream)
                : -1;
            head_workspace->forward(model->lm_head, model->lm_head_metadata, final_norm, logits, 1, stream);
            if (target_projection_timing)
                target_projection_timing->end(timing_slot, stream);
            cuda_check(cudaGetLastError(), "launch E4A H6 LM head");
        }
        if(!graph_capture_active) {
            head_work.submitted_rows+=static_cast<std::uint64_t>(head_plan.rows);
            head_work.omitted_rows+=static_cast<std::uint64_t>(rows-head_plan.rows);
        }
        if (events) record(events->lm_head_end, stream, "record E4B1 LM head end");
        last_rows = rows;
        if(capture_taps && !graph_capture_active) {
            const auto generation=tap_generation.fetch_add(1,std::memory_order_release)+1;
            require(generation,"authoritative tap generation exhausted");
        }
        qkv_trace_valid = !graph_capture_active;
        if(host_kv.enabled) {
            cuda_check(cudaStreamSynchronize(stream),"complete authoritative host KV forward");
            for(auto& layer:full_layers)
                if(layer)layer->complete_prefill_projection_chain_graph_after_drain(stream);
            for(auto& layer:gdn_layers)
                if(layer)layer->complete_prefill_projection_chain_graph_after_drain(stream);
            if(host_kv_batch_copy) {
                const auto finish_copies=[&] {
                    cuda_check(cudaStreamSynchronize(host_kv_copy_stream),
                        "complete authoritative host KV batch copies");
                };
                if(!host_kv_batch_use.retire_before_reuse(finish_copies))finish_copies();
                if(host_kv_pinned_chunks && host_kv_pinned_slot_count>2) {
                    // The stream drain proves every deep-ring transfer complete.
                    // Complete slots 2..N here; slots 0/1 retain the established
                    // D2H publication boundary below (or are completed here when
                    // pinned D2H is explicitly disabled).
                    const int first=host_kv_pinned_d2h?2:0;
                    for(int slot=first;slot<host_kv_pinned_slot_count;++slot)
                        if(host_kv_pinned_in_flight[slot] ||
                           registered_kv_pending[slot])
                            acquire_host_kv_pinned_slot(slot,
                                "retire deep exact HostKV H2D slot after forward drain");
                }
            }
            if(host_kv_pinned_d2h) {
                drain_host_kv_banked_d2h();
                const bool deferred_publication =
                    !host_kv_pinned_scatter[0].empty() ||
                    !host_kv_pinned_scatter[1].empty();
                acquire_host_kv_pinned_slot(0,
                    "complete deferred exact host KV scatter slot0 before publication");
                acquire_host_kv_pinned_slot(1,
                    "complete deferred exact host KV scatter slot1 before publication");
                if(deferred_publication)
                    ++host_kv.pinned_publication_drains;
            }
            if(cache_usable) {
                if(forward_publish_device_prefix) {
                    device_prefix->publish_tags(pending_host.all,position+rows,true);
                    ++host_kv.device_prefix_forward_publish_forwards;
                } else device_prefix->publish_tags(exact_prefix_pages,cacheable_rows);
            }
            exact_prefix_pages=std::move(pending_host.all);
            exact_prefix_position=position+rows;
            host_kv.completed_rows+=rows;
            host_kv_failed=false;
            if(shared_prefix_use)shared_prefix_use->complete();
        }
    }
};

Exl3TextModel::Exl3TextModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)), max_context_(0), model_bytes_(impl_->model_bytes) {}

Exl3TextModel::~Exl3TextModel() = default;

const Exl3ModelLoadStats& Exl3TextModel::load_stats() const noexcept { return impl_->load_stats; }

std::unique_ptr<Exl3TextModel> Exl3TextModel::load(const std::filesystem::path& model_directory,
                                                   int max_context, const Exl3LoadOptions& options) {
    const char* extended_context=std::getenv("NINFER_EXL3_EXTENDED_CONTEXT_64K");
    require(!extended_context || std::strcmp(extended_context,"0")==0 ||
            std::strcmp(extended_context,"1")==0,
            "extended 64K context must be 0 or 1");
    const bool extended_context_enabled=extended_context &&
        std::strcmp(extended_context,"1")==0;
    const char* extended_context_128=std::getenv("NINFER_EXL3_EXTENDED_CONTEXT_128K");
    require(!extended_context_128 || std::strcmp(extended_context_128,"0")==0 ||
            std::strcmp(extended_context_128,"1")==0,
            "extended 128K context must be 0 or 1");
    const bool extended_context_128_enabled=extended_context_128 &&
        std::strcmp(extended_context_128,"1")==0;
    const int context_limit=static_cast<int>(Exl3NativeContextExtent::configuration_limit(
        extended_context_enabled,extended_context_128_enabled));
    require(max_context > 0 && max_context <= context_limit,
            extended_context_128_enabled ? "E4C1 extended max_context must be 1..131072" :
            (extended_context_enabled ? "E4C1 extended max_context must be 1..65536" :
                                        "E4C1 max_context must be 1..32768"));
    require(!extended_context_128_enabled,
        "128K is a static configuration candidate only; model execution is unsupported");
    auto impl = std::make_unique<Impl>();
    impl->directory = model_directory;
    impl->load_options = options;
    impl->load_from_disk();
    auto result = std::unique_ptr<Exl3TextModel>(new Exl3TextModel(std::move(impl)));
    result->max_context_ = Exl3NativeContextExtent::checked_max_context(
        static_cast<std::uint64_t>(max_context),extended_context_enabled,
        extended_context_128_enabled);
    result->model_bytes_ = result->impl_->model_bytes;
    return result;
}

std::uint64_t Exl3TextModel::allocation_owner_metadata_bytes() const {
    Exl3ResourceInventory::Requirement requirement;
    using Domain=Exl3ResourceInventory::Domain;
    if(impl_->allocations.capacity())requirement.add(Domain::host_metadata,
        impl_->allocations.capacity(),sizeof(std::unique_ptr<DeviceAllocation>));
    if(impl_->allocations.size())requirement.add(Domain::host_metadata,
        impl_->allocations.size(),DeviceAllocation::owner_metadata_bytes());
    return requirement.units[static_cast<unsigned>(Domain::host_metadata)];
}

std::unique_ptr<Exl3TextContext> Exl3TextModel::create_context(bool capture_taps,bool allocate_device_prefix,bool defer_reconstruction) const {
    return create_context_impl(capture_taps,allocate_device_prefix,defer_reconstruction,nullptr).exclusive;
}
std::shared_ptr<Exl3TextContext> Exl3TextModel::create_context_reserved(
    Exl3VeriCacheServingCoordinator& authority,bool capture_taps,
    unsigned startup_fault_for_test,bool enable_qualified_media,
    bool allow_ordinary_graphs) const {
    Exl3ResourceInventory::Requirement initial;initial.configuration=0x4354585354415254;
    initial.add(Exl3ResourceInventory::Domain::host_metadata,1,Exl3TextContext::fixed_owner_metadata_bytes());
    ContextConstruction prepared;
    const auto planning_quarantine_before=Exl3TextContext::retirement_quarantine_witness();
    authority.allocate_startup_resources_growing(initial,[&](auto configuration,auto&& extend) {
        require(configuration==initial.configuration,"context planning reservation identity");
        Exl3ResourceInventory actual;
        prepared=create_context_impl(capture_taps,false,true,&authority,startup_fault_for_test,
            extend,&actual,enable_qualified_media,allow_ordinary_graphs);
        return actual;
    },[&]() noexcept {
        const auto before=Exl3TextContext::retirement_quarantine_witness();
        prepared.shared.reset();
        prepared.exclusive.reset();
        if(Exl3TextContext::retirement_quarantine_witness()!=before)
            authority.seal_failed_startup_retirement();
    },[&]() noexcept {
        if(Exl3TextContext::retirement_quarantine_witness()!=planning_quarantine_before)
            authority.seal_failed_startup_retirement();
    });
    prepared.shared->impl_->finish_private_constructor_credits();
    return std::move(prepared.shared);
}
Exl3TextModel::ContextConstruction Exl3TextModel::create_context_impl(
    bool capture_taps,bool allocate_device_prefix,bool defer_reconstruction,
    Exl3VeriCacheServingCoordinator* authority,unsigned startup_fault,
    const std::function<void(const Exl3ResourceInventory::Requirement&)>& extend,
    Exl3ResourceInventory* actual_result,bool enable_qualified_media,
    bool allow_ordinary_graphs) const {
    require((startup_fault<=4 || (startup_fault>=6 && startup_fault<=27)) && (!startup_fault || authority),
        "context startup fault requires reserved stage1..4 or6..19");
    require(hostkv_quarantined_contexts.load(std::memory_order_acquire)==0,
        "unresolved HostKV context retirement; context creation refused");
    require(DeviceAllocation::quarantined_count.load(std::memory_order_acquire)==0,
        "unresolved generic allocation cleanup; context creation refused");
    require(reconstruction_quarantined_contexts.load(std::memory_order_acquire)==0,
        "unresolved reconstruction context retirement; context creation refused");
    require(ReconstructionBacking::quarantined_allocations.load(std::memory_order_acquire)==0,
        "unresolved reconstruction allocation cleanup; context creation refused");
    auto impl = std::make_unique<Exl3TextContext::Impl>();
    struct ConstructionRetirement {
        std::unique_ptr<Exl3TextContext::Impl>& owner;
        Exl3VeriCacheServingCoordinator* authority;
        const std::array<std::uint64_t,8> before=Exl3TextContext::retirement_quarantine_witness();
        ~ConstructionRetirement() noexcept {
            if(owner && !Exl3TextContext::Impl::retain_failed_host_kv_drain(owner))
                owner.reset();
            if(authority && Exl3TextContext::retirement_quarantine_witness()!=before)
                authority->seal_failed_startup_retirement();
        }
    } construction_retirement{impl,authority};
    impl->model = impl_;
    impl->max_context = max_context_;
    impl->capture_taps = capture_taps;
    const char* continuation_graph_b8=
        std::getenv("NINFER_E5A4_CONTINUATION_GRAPH_B8");
    require(!continuation_graph_b8 ||
                std::strcmp(continuation_graph_b8,"0")==0 ||
                std::strcmp(continuation_graph_b8,"1")==0,
            "fixed-B8 continuation graph must be 0 or 1");
    impl->continuation_graph_b8_enabled=continuation_graph_b8 &&
        std::strcmp(continuation_graph_b8,"1")==0;
    const char* graph_gdn_qkvz =
        std::getenv("NINFER_EXL3_CONTINUATION_GRAPH_GDN_QKVZ_CONCURRENT");
    require(!graph_gdn_qkvz || std::strcmp(graph_gdn_qkvz,"0")==0 ||
                std::strcmp(graph_gdn_qkvz,"1")==0,
            "fixed-B8 graph GDN QKV/Z concurrency must be 0 or 1");
    impl->continuation_graph_gdn_qkvz_concurrent =
        impl->continuation_graph_b8_enabled && graph_gdn_qkvz &&
        std::strcmp(graph_gdn_qkvz,"1")==0;
    const char* host_kv=std::getenv("NINFER_EXL3_EXACT_HOST_KV");
    impl->host_kv.enabled=host_kv && std::strcmp(host_kv,"1")==0;
    const char* host_kv_gdn_segment_graphs=
        std::getenv("NINFER_EXL3_HOST_KV_GDN_SEGMENT_GRAPHS");
    require(!host_kv_gdn_segment_graphs ||
            std::strcmp(host_kv_gdn_segment_graphs,"0")==0 ||
            std::strcmp(host_kv_gdn_segment_graphs,"1")==0,
        "HostKV GDN segment graphs must be 0 or 1");
    impl->host_kv_gdn_segment_graphs_enabled=impl->host_kv.enabled &&
        host_kv_gdn_segment_graphs &&
        std::strcmp(host_kv_gdn_segment_graphs,"1")==0;
    require(!impl->host_kv_gdn_segment_graphs_enabled ||
            !impl->continuation_graph_b8_enabled,
        "HostKV GDN segment graphs exclude OSCAR continuation graphs");
    const char* host_kv_full_layer_graphs=
        std::getenv("NINFER_EXL3_HOST_KV_FULL_LAYER_GRAPHS");
    require(!host_kv_full_layer_graphs ||
            std::strcmp(host_kv_full_layer_graphs,"0")==0 ||
            std::strcmp(host_kv_full_layer_graphs,"1")==0,
        "HostKV full-layer graphs must be 0 or 1");
    impl->host_kv_full_layer_graphs_enabled=
        host_kv_full_layer_graphs &&
        std::strcmp(host_kv_full_layer_graphs,"1")==0;
    require(!impl->host_kv_full_layer_graphs_enabled ||
            (impl->host_kv_gdn_segment_graphs_enabled &&
             !impl->continuation_graph_b8_enabled),
        "HostKV full-layer graphs require compatible GDN segment graphs");
    const char* ordinary_gdn_segment_graphs=
        std::getenv("NINFER_EXL3_ORDINARY_GDN_SEGMENT_GRAPHS");
    require(!ordinary_gdn_segment_graphs ||
            std::strcmp(ordinary_gdn_segment_graphs,"0")==0 ||
            std::strcmp(ordinary_gdn_segment_graphs,"1")==0,
        "ordinary device-KV GDN segment graphs must be 0 or 1");
    impl->ordinary_gdn_segment_graphs_enabled=allow_ordinary_graphs &&
        ordinary_gdn_segment_graphs &&
        std::strcmp(ordinary_gdn_segment_graphs,"1")==0;
    require(!impl->ordinary_gdn_segment_graphs_enabled ||
            (!impl->host_kv.enabled && !impl->host_kv_full_layer_graphs_enabled &&
             !impl->continuation_graph_b8_enabled),
        "ordinary device-KV GDN segment graphs require physical device KV");
    const char* host_kv_mlp_tail_graphs=
        std::getenv("NINFER_EXL3_HOST_KV_MLP_TAIL_GRAPHS");
    require(!host_kv_mlp_tail_graphs ||
            std::strcmp(host_kv_mlp_tail_graphs,"0")==0 ||
            std::strcmp(host_kv_mlp_tail_graphs,"1")==0,
        "HostKV MLP-tail graphs must be 0 or 1");
    impl->host_kv_mlp_tail_graphs_enabled=host_kv_mlp_tail_graphs &&
        std::strcmp(host_kv_mlp_tail_graphs,"1")==0;
    require(!impl->host_kv_mlp_tail_graphs_enabled ||
            (impl->host_kv.enabled && !impl->host_kv_full_layer_graphs_enabled &&
             !impl->continuation_graph_b8_enabled),
        "HostKV MLP-tail graphs require ordinary exact HostKV");
    const char* ordinary_mlp_tail_graphs=
        std::getenv("NINFER_EXL3_ORDINARY_MLP_TAIL_GRAPHS");
    require(!ordinary_mlp_tail_graphs ||
            std::strcmp(ordinary_mlp_tail_graphs,"0")==0 ||
            std::strcmp(ordinary_mlp_tail_graphs,"1")==0,
        "ordinary device-KV MLP-tail graphs must be 0 or 1");
    impl->ordinary_mlp_tail_graphs_enabled=allow_ordinary_graphs &&
        ordinary_mlp_tail_graphs &&
        std::strcmp(ordinary_mlp_tail_graphs,"1")==0;
    require(!impl->ordinary_mlp_tail_graphs_enabled ||
            (!impl->host_kv.enabled && !impl->host_kv_full_layer_graphs_enabled &&
             !impl->continuation_graph_b8_enabled),
        "ordinary device-KV MLP-tail graphs require physical device KV");
    const char* ordinary_full_layer_graphs=
        std::getenv("NINFER_EXL3_ORDINARY_FULL_LAYER_GRAPHS");
    require(!ordinary_full_layer_graphs ||
            std::strcmp(ordinary_full_layer_graphs,"0")==0 ||
            std::strcmp(ordinary_full_layer_graphs,"1")==0,
        "ordinary device-KV full-layer graphs must be 0 or 1");
    impl->ordinary_full_layer_graphs_enabled=allow_ordinary_graphs &&
        ordinary_full_layer_graphs &&
        std::strcmp(ordinary_full_layer_graphs,"1")==0;
    require(!impl->ordinary_full_layer_graphs_enabled ||
            (!impl->host_kv.enabled && !impl->host_kv_full_layer_graphs_enabled &&
             !impl->ordinary_mlp_tail_graphs_enabled &&
             !impl->continuation_graph_b8_enabled),
        "ordinary device-KV full-layer graphs require physical C1 and exclude MLP-tail graphs");
    const char* ordinary_full_layer_graph_extended_replay=std::getenv(
        "NINFER_EXL3_ORDINARY_FULL_LAYER_GRAPH_EXTENDED_REPLAY");
    require(!ordinary_full_layer_graph_extended_replay ||
            std::strcmp(ordinary_full_layer_graph_extended_replay,"0")==0 ||
            std::strcmp(ordinary_full_layer_graph_extended_replay,"1")==0,
        "ordinary device-KV full-layer graph extended replay must be 0 or 1");
    impl->ordinary_full_layer_graph_extended_replay=allow_ordinary_graphs &&
        ordinary_full_layer_graph_extended_replay &&
        std::strcmp(ordinary_full_layer_graph_extended_replay,"1")==0;
    require(!impl->ordinary_full_layer_graph_extended_replay ||
            impl->ordinary_full_layer_graphs_enabled,
        "ordinary full-layer graph extended replay requires ordinary full-layer graphs");
    const char* ordinary_full_layer_multirow=std::getenv(
        "NINFER_EXL3_ORDINARY_FULL_LAYER_MULTIROW_GRAPHS");
    require(!ordinary_full_layer_multirow ||
            std::strcmp(ordinary_full_layer_multirow,"0")==0 ||
            std::strcmp(ordinary_full_layer_multirow,"1")==0,
        "ordinary full-layer multirow graphs must be 0 or 1");
    // Default on with extended full-layer replay (measured); "0" keeps the
    // verifier's multi-row full-attention layers eager.
    impl->ordinary_full_layer_multirow_graphs_enabled=
        impl->ordinary_full_layer_graph_extended_replay &&
        (!ordinary_full_layer_multirow ||
         std::strcmp(ordinary_full_layer_multirow,"1")==0);
    const char* host_kv_transaction_checkpoint_graph=std::getenv(
        "NINFER_EXL3_HOST_KV_TRANSACTION_CHECKPOINT_GRAPH");
    require(!host_kv_transaction_checkpoint_graph ||
            std::strcmp(host_kv_transaction_checkpoint_graph,"0")==0 ||
            std::strcmp(host_kv_transaction_checkpoint_graph,"1")==0,
        "HostKV transaction checkpoint graph must be 0 or 1");
    impl->host_kv_transaction_checkpoint_graph_enabled=
        host_kv_transaction_checkpoint_graph &&
        std::strcmp(host_kv_transaction_checkpoint_graph,"1")==0;
    require(!impl->host_kv_transaction_checkpoint_graph_enabled ||
            (impl->host_kv.enabled && !impl->continuation_graph_b8_enabled),
        "HostKV transaction checkpoint graph requires exact HostKV ownership");
    const char* device_transaction_checkpoint_graph=std::getenv(
        "NINFER_EXL3_DEVICE_KV_TRANSACTION_CHECKPOINT_GRAPH");
    require(!device_transaction_checkpoint_graph ||
            std::strcmp(device_transaction_checkpoint_graph,"0")==0 ||
            std::strcmp(device_transaction_checkpoint_graph,"1")==0,
        "device-KV transaction checkpoint graph must be 0 or 1");
    // Default on for the guarded device-KV transaction (measured); "0" keeps
    // the eager per-layer checkpoint copies.
    impl->device_transaction_checkpoint_graph_enabled=
        device_transaction_checkpoint_graph ?
            std::strcmp(device_transaction_checkpoint_graph,"1")==0 :
            (!impl->host_kv.enabled && !impl->oscar &&
             fast_device_kv_transaction_enabled());
    require(!impl->device_transaction_checkpoint_graph_enabled ||
            (!impl->host_kv.enabled && !impl->oscar &&
             fast_device_kv_transaction_enabled()),
        "device-KV transaction checkpoint graph requires guarded device KV");
    const char* host_kv_transaction_recurrent_trace=std::getenv(
        "NINFER_EXL3_HOST_KV_TRANSACTION_RECURRENT_TRACE");
    require(!host_kv_transaction_recurrent_trace ||
            std::strcmp(host_kv_transaction_recurrent_trace,"0")==0 ||
            std::strcmp(host_kv_transaction_recurrent_trace,"1")==0,
        "HostKV transaction recurrent trace must be 0 or 1");
    impl->host_kv_transaction_recurrent_trace_enabled=
        host_kv_transaction_recurrent_trace &&
        std::strcmp(host_kv_transaction_recurrent_trace,"1")==0;
    require(!impl->host_kv_transaction_recurrent_trace_enabled ||
            impl->host_kv.enabled,
        "HostKV transaction recurrent trace requires exact HostKV ownership");
    const char* batched_recurrent_export=std::getenv(
        "NINFER_EXL3_BATCHED_RECURRENT_EXPORT");
    require(!batched_recurrent_export ||
            std::strcmp(batched_recurrent_export,"0")==0 ||
            std::strcmp(batched_recurrent_export,"1")==0,
        "batched recurrent export must be 0 or 1");
    impl->batched_recurrent_export=batched_recurrent_export &&
        std::strcmp(batched_recurrent_export,"1")==0;
    require(!impl->batched_recurrent_export || impl->pinned_recurrent_export,
        "batched recurrent export requires pinned recurrent export");
    require(!impl->batched_recurrent_export,
        "batched recurrent export is rejected on this caller: the CUDA batch "
        "faulted at the immutable recurrent-state fence");
    const char* native16 = std::getenv("NINFER_EXL3_NATIVE_CONTINUATION16");
    require(!native16 || std::strcmp(native16,"0")==0 || std::strcmp(native16,"1")==0,
        "native continuation16 must be 0 or 1");
    impl->native_continuation16_enabled = native16 && std::strcmp(native16,"1")==0;
    require(!impl->native_continuation16_enabled ||
        (impl->host_kv.enabled && !impl->oscar_only && !impl->continuation_graph_b8_enabled),
        "native continuation16 requires ordinary eager exact-host ownership");
    const char* host_kv_cpu_profile=std::getenv("NINFER_EXL3_HOST_KV_CPU_PROFILE");
    require(!host_kv_cpu_profile || std::strcmp(host_kv_cpu_profile,"0")==0 ||
        std::strcmp(host_kv_cpu_profile,"1")==0,
        "exact host KV CPU profile must be 0 or 1");
    impl->host_kv_cpu_profile=impl->host_kv.enabled && host_kv_cpu_profile &&
        std::strcmp(host_kv_cpu_profile,"1")==0;
    const char* host_kv_unique_tail_reuse=
        std::getenv("NINFER_EXL3_EXACT_HOST_KV_UNIQUE_TAIL_REUSE");
    require(!host_kv_unique_tail_reuse ||
            std::strcmp(host_kv_unique_tail_reuse,"0")==0 ||
            std::strcmp(host_kv_unique_tail_reuse,"1")==0,
        "exact host KV unique-tail reuse must be 0 or 1");
    impl->host_kv_unique_tail_reuse=impl->host_kv.enabled &&
        host_kv_unique_tail_reuse &&
        std::strcmp(host_kv_unique_tail_reuse,"1")==0;
    const char* host_kv_batch_sync=std::getenv("NINFER_EXL3_EXACT_HOST_KV_BATCH_SYNC");
    require(!host_kv_batch_sync || std::strcmp(host_kv_batch_sync,"0")==0 ||
        std::strcmp(host_kv_batch_sync,"1")==0,
        "exact host KV batch synchronization must be 0 or 1");
    // Qualified default: one authoritative stream barrier after the complete
    // forward. Set =0 to retain the former per-page barriers for diagnosis.
    impl->host_kv_batch_sync=impl->host_kv.enabled &&
        (!host_kv_batch_sync || std::strcmp(host_kv_batch_sync,"0")!=0);
    require(!impl->host_kv_batch_sync || impl->host_kv.enabled,
        "batched host KV synchronization requires exact host KV");
    const char* host_kv_batch_copy=std::getenv("NINFER_EXL3_EXACT_HOST_KV_BATCH_COPY");
    require(!host_kv_batch_copy || std::strcmp(host_kv_batch_copy,"0")==0 ||
        std::strcmp(host_kv_batch_copy,"1")==0,
        "exact host KV batch copy must be 0 or 1");
    // Qualified CUDA 13 path. Set =0 to retain individual submissions for
    // compatibility diagnosis without changing logical page transfers.
    impl->host_kv_batch_copy=impl->host_kv.enabled && impl->host_kv_batch_sync &&
        (!host_kv_batch_copy || std::strcmp(host_kv_batch_copy,"0")!=0);
    require(!impl->host_kv_batch_copy || (impl->host_kv.enabled && impl->host_kv_batch_sync),
        "exact host KV batch copy requires final-barrier host KV");
    const char* host_kv_pinned_chunks=std::getenv("NINFER_EXL3_EXACT_HOST_KV_PINNED_CHUNKS");
    require(!host_kv_pinned_chunks || std::strcmp(host_kv_pinned_chunks,"0")==0 ||
        std::strcmp(host_kv_pinned_chunks,"1")==0,
        "exact host KV pinned chunks must be 0 or 1");
    // Qualified bounded default: two fixed 1 MiB pinned buffers replace
    // pageable batch-source staging. Explicit zero retains page batching.
    impl->host_kv_pinned_chunks=impl->host_kv_batch_copy &&
        (!host_kv_pinned_chunks || std::strcmp(host_kv_pinned_chunks,"0")!=0);
    const char* host_kv_pinned_h2d_slots=
        std::getenv("NINFER_EXL3_EXACT_HOST_KV_PINNED_H2D_SLOTS");
    require(!host_kv_pinned_h2d_slots ||
            std::strcmp(host_kv_pinned_h2d_slots,"2")==0 ||
            std::strcmp(host_kv_pinned_h2d_slots,"32")==0,
        "exact HostKV pinned H2D slots must be 2 or 32");
    impl->host_kv_pinned_slot_count=impl->host_kv_pinned_chunks &&
        host_kv_pinned_h2d_slots &&
        std::strcmp(host_kv_pinned_h2d_slots,"32")==0?32:2;
    const char* host_kv_prefill_pinned_batch=
        std::getenv("NINFER_EXL3_PREFILL_PINNED_BATCH_OVER_REGISTERED");
    require(!host_kv_prefill_pinned_batch ||
            std::strcmp(host_kv_prefill_pinned_batch,"0")==0 ||
            std::strcmp(host_kv_prefill_pinned_batch,"1")==0,
        "exact host KV prefill pinned batch must be 0 or 1");
    impl->host_kv_prefill_pinned_batch=impl->host_kv_pinned_chunks &&
        host_kv_prefill_pinned_batch &&
        std::strcmp(host_kv_prefill_pinned_batch,"1")==0;
    require(!impl->host_kv_prefill_pinned_batch || impl->host_kv_pinned_chunks,
        "exact host KV prefill pinned batch requires pinned chunks");
    const char* host_kv_pinned_d2h=std::getenv("NINFER_EXL3_EXACT_HOST_KV_PINNED_D2H");
    require(!host_kv_pinned_d2h || std::strcmp(host_kv_pinned_d2h,"0")==0 ||
        std::strcmp(host_kv_pinned_d2h,"1")==0,
        "exact host KV pinned D2H must be 0 or 1");
    // Qualified bounded default. Reuse the fixed staging allocation; never
    // publish a page until event completion and CPU scatter. Explicit zero
    // retains the prior pageable D2H batch for matched diagnosis.
    impl->host_kv_pinned_d2h=impl->host_kv_pinned_chunks &&
        (!host_kv_pinned_d2h || std::strcmp(host_kv_pinned_d2h,"0")!=0);
    require(!impl->host_kv_pinned_d2h || impl->host_kv_pinned_chunks,
        "exact host KV pinned D2H requires pinned chunks");
    const char* host_kv_banked_d2h=
        std::getenv("NINFER_EXL3_EXACT_HOST_KV_BANKED_D2H");
    require(!host_kv_banked_d2h || std::strcmp(host_kv_banked_d2h,"0")==0 ||
            std::strcmp(host_kv_banked_d2h,"1")==0,
        "exact host KV banked D2H must be 0 or 1");
    impl->host_kv_banked_d2h=impl->host_kv_pinned_d2h &&
        host_kv_banked_d2h && std::strcmp(host_kv_banked_d2h,"1")==0;
    require(!impl->host_kv_banked_d2h || impl->host_kv_pinned_d2h,
        "exact host KV banked D2H requires pinned D2H");
    const char* host_kv_deferred_scatter=
        std::getenv("NINFER_EXL3_EXACT_HOST_KV_DEFERRED_SCATTER");
    require(!host_kv_deferred_scatter ||
            std::strcmp(host_kv_deferred_scatter,"0")==0 ||
            std::strcmp(host_kv_deferred_scatter,"1")==0,
        "exact host KV deferred scatter must be 0 or 1");
    impl->host_kv_deferred_scatter=impl->host_kv_pinned_d2h &&
        host_kv_deferred_scatter &&
        std::strcmp(host_kv_deferred_scatter,"1")==0;
    const auto transfer_plan=Exl3HostKVTransferRequirements::derive(max_context_,impl->host_kv_batch_copy,
        impl->host_kv_pinned_chunks,impl->host_kv_pinned_d2h,
        impl->host_kv_pinned_chunk_bytes,impl->host_kv_banked_d2h,
        impl->host_kv_banked_d2h_bytes,
        static_cast<std::size_t>(impl->host_kv_pinned_slot_count));
    require(transfer_plan.pinned_owners==
        (impl->host_kv_pinned_chunks?
            static_cast<std::size_t>(impl->host_kv_pinned_slot_count):0)+
            (impl->host_kv_banked_d2h?1:0),
        "host KV pinned owner plan differs from staging layout");
    require(transfer_plan.descriptors<=impl->host_kv_batch_destinations.max_size() &&
        transfer_plan.descriptors<=impl->host_kv_batch_sources.max_size() &&
        transfer_plan.descriptors<=impl->host_kv_batch_sizes.max_size(),
        "host KV descriptor plan exceeds container capacity");
    const auto materialize_transfer=[&] {
    if(impl->host_kv_batch_copy) {
        const auto& transfer_requirement=transfer_plan.resources;
        if(!impl->host_kv_pinned_chunks || !impl->host_kv_pinned_d2h) {
            const auto moves=transfer_plan.descriptors;
            impl->host_kv_batch_destinations.reserve(moves);
            impl->host_kv_batch_sources.reserve(moves);impl->host_kv_batch_sizes.reserve(moves);
            Exl3ResourceInventory::Requirement actual_metadata;
            actual_metadata.add(Exl3ResourceInventory::Domain::host_metadata,
                impl->host_kv_batch_destinations.capacity(),sizeof(void*));
            actual_metadata.add(Exl3ResourceInventory::Domain::host_metadata,
                impl->host_kv_batch_sources.capacity(),sizeof(const void*));
            actual_metadata.add(Exl3ResourceInventory::Domain::host_metadata,
                impl->host_kv_batch_sizes.capacity(),sizeof(std::size_t));
            impl->host_kv.batch_metadata_bytes=actual_metadata.units[
                static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)];
        }
        cuda_check(cudaGetDevice(&impl->host_kv_copy_device),"HostKV transfer allocation device");
        cuda_check(cudaStreamCreateWithFlags(&impl->host_kv_copy_stream,cudaStreamNonBlocking),
            "create exact host KV batch-copy stream");
        cuda_check(cudaEventCreateWithFlags(&impl->host_kv_h2d_ready,cudaEventDisableTiming),
            "create exact host KV H2D event");
        cuda_check(cudaEventCreateWithFlags(&impl->host_kv_compute_ready,cudaEventDisableTiming),
            "create exact host KV compute event");
        if(impl->host_kv_pinned_chunks) {
            std::size_t actual_pinned_bytes=0,actual_pinned_owners=0;
            for(int slot=0;slot<impl->host_kv_pinned_slot_count;++slot) {
                impl->host_kv_pinned_staging[slot]=std::make_unique<PinnedHostBuffer>(
                    impl->host_kv_pinned_chunk_bytes);
                require(impl->host_kv_pinned_staging[slot]->size()==impl->host_kv_pinned_chunk_bytes,
                    "host KV pinned chunk allocation extent mismatch");
                ++actual_pinned_owners;
                actual_pinned_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(actual_pinned_bytes,
                    impl->host_kv_pinned_staging[slot]->size());
                cuda_check(cudaEventCreateWithFlags(&impl->host_kv_pinned_ready[slot],
                    cudaEventDisableTiming),"create exact host KV pinned chunk event");
            }
            if(impl->host_kv_banked_d2h) {
                impl->host_kv_banked_d2h_staging=
                    std::make_unique<PinnedHostBuffer>(impl->host_kv_banked_d2h_bytes);
                require(impl->host_kv_banked_d2h_staging->size()==
                        impl->host_kv_banked_d2h_bytes,
                    "host KV banked D2H allocation extent mismatch");
                ++actual_pinned_owners;
                actual_pinned_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(
                    actual_pinned_bytes,impl->host_kv_banked_d2h_staging->size());
            }
            require(actual_pinned_owners==transfer_plan.pinned_owners,
                "host KV pinned owner construction count mismatch");
            require(actual_pinned_bytes==transfer_requirement.units[
                static_cast<unsigned>(Exl3ResourceInventory::Domain::cuda_registered_host)],
                "host KV complete pinned allocation requirement mismatch");
            impl->host_kv.pinned_staging_bytes=actual_pinned_bytes;
        }
    }
    };
    const char* oscar_only=std::getenv("NINFER_EXL3_OSCAR_L0_ONLY");
    impl->oscar_only=oscar_only && std::strcmp(oscar_only,"1")==0;
    require(!impl->host_kv.enabled || !impl->oscar_only,"exact host KV and OSCAR-only modes are separate");
    const char* eager_mlp_gateup=
        std::getenv("NINFER_EXL3_EAGER_MLP_GATEUP_CONCURRENT");
    require(!eager_mlp_gateup || std::strcmp(eager_mlp_gateup,"0")==0 ||
            std::strcmp(eager_mlp_gateup,"1")==0,
        "eager MLP gate/up concurrency must be 0 or 1");
    impl->eager_mlp_gateup_concurrent=eager_mlp_gateup &&
        std::strcmp(eager_mlp_gateup,"1")==0;
    require(!impl->eager_mlp_gateup_concurrent ||
            (impl->host_kv.enabled && !impl->oscar_only &&
             !impl->continuation_graph_b8_enabled),
        "eager MLP gate/up concurrency requires ordinary exact HostKV");
    require(max_context_<=32768 || impl->host_kv.enabled || impl->oscar_only,
            "extended context requires exact-host KV or OSCAR-only storage");
    const char* wide_prefill = std::getenv("NINFER_EXL3_WIDE_PREFILL");
    impl->wide_prefill_enabled = wide_prefill && std::strcmp(wide_prefill, "1") == 0;
    const char* prefill_qkv=
        std::getenv("NINFER_EXL3_PREFILL_QKV_CONCURRENT");
    require(!prefill_qkv || std::strcmp(prefill_qkv,"0")==0 ||
            std::strcmp(prefill_qkv,"1")==0,
        "wide-prefill QKV concurrency must be 0 or 1");
    impl->prefill_qkv_concurrent=prefill_qkv &&
        std::strcmp(prefill_qkv,"1")==0;
    require(!impl->prefill_qkv_concurrent ||
            (impl->wide_prefill_enabled && impl->host_kv.enabled &&
             !impl->oscar_only && !impl->continuation_graph_b8_enabled),
        "wide-prefill QKV concurrency requires ordinary exact HostKV wide prefill");
    const char* staged = std::getenv("NINFER_EXL3_PREFILL_STAGED_REDUCTION");
    const bool staged_prefill_enabled = staged && std::strcmp(staged, "1") == 0;
    const char* shared_accum = std::getenv("NINFER_EXL3_SHARED_ACCUM");
    const bool shared_accumulation_enabled =
        shared_accum && std::strcmp(shared_accum, "1") == 0;
    const char* wide64 = std::getenv("NINFER_EXL3_PREFILL_WIDE64");
    const bool wide64_enabled = wide64 && std::strcmp(wide64, "1") == 0;
    const char* wide128 = std::getenv("NINFER_EXL3_PREFILL_WIDE128");
    const bool wide128_enabled = wide128 && std::strcmp(wide128, "1") == 0;
    const char* wide256 = std::getenv("NINFER_EXL3_PREFILL_WIDE256");
    const bool wide256_enabled = wide256 && std::strcmp(wide256, "1") == 0;
    const char* wide512 = std::getenv("NINFER_EXL3_PREFILL_WIDE512");
    const bool wide512_enabled = wide512 && std::strcmp(wide512, "1") == 0;
    const char* wide1024 = std::getenv("NINFER_EXL3_PREFILL_WIDE1024");
    const bool wide1024_enabled = wide1024 && std::strcmp(wide1024, "1") == 0;
    // Resolve the requested width ladder before validating transform/scratch
    // prerequisites so unset optional defaults can be admitted at final 128/1024.
    int requested_prefill_capacity = kMaxRows;
    if (impl->wide_prefill_enabled && staged_prefill_enabled)
        requested_prefill_capacity = 32;
    if (requested_prefill_capacity == 32 && wide64_enabled)
        requested_prefill_capacity = 64;
    if (requested_prefill_capacity == 64 && wide128_enabled)
        requested_prefill_capacity = 128;
    if (requested_prefill_capacity == 128 && wide256_enabled)
        requested_prefill_capacity = 256;
    if (requested_prefill_capacity == 256 && wide512_enabled)
        requested_prefill_capacity = 512;
    if (requested_prefill_capacity == 512 && wide1024_enabled)
        requested_prefill_capacity = 1024;
    const char* shared_layer = std::getenv("NINFER_EXL3_SHARED_LAYER_SCRATCH");
    const bool qualified_default_group = exl3_prefill_qualified_default_group(
        requested_prefill_capacity, impl->wide_prefill_enabled,
        staged_prefill_enabled, shared_accumulation_enabled);
    const bool shared_layer_enabled = shared_layer
        ? std::strcmp(shared_layer, "1") == 0 : qualified_default_group;
    const char* shared_transform = std::getenv("NINFER_EXL3_SHARED_TRANSFORM");
    const bool shared_transform_enabled = shared_transform
        ? std::strcmp(shared_transform, "1") == 0 : qualified_default_group;
    const char* gdn_wide_slab = std::getenv("NINFER_EXL3_GDN_WIDE_SLAB");
    if (impl->wide_prefill_enabled && staged_prefill_enabled)
        impl->prefill_capacity = 32;
    if (impl->prefill_capacity == 32 && wide64_enabled)
        impl->prefill_capacity = 64;
    if (impl->prefill_capacity == 64 && wide128_enabled) {
        require(shared_accumulation_enabled, "wide128 requires shared accumulation");
        impl->prefill_capacity = 128;
    }
    if (impl->prefill_capacity == 128 && wide256_enabled) {
        require(shared_transform_enabled, "wide256 requires shared input transform");
        impl->prefill_capacity = 256;
    }
    if (impl->prefill_capacity == 256 && wide512_enabled) {
        require(shared_layer_enabled, "wide512 requires shared layer scratch");
        impl->prefill_capacity=512;
    }
    if (impl->prefill_capacity == 512 && wide1024_enabled)
        impl->prefill_capacity=1024;
    const char* numeric_prefill_projection = std::getenv(
        "NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_NUMERIC");
    require(!numeric_prefill_projection ||
            std::strcmp(numeric_prefill_projection, "0") == 0 ||
            std::strcmp(numeric_prefill_projection, "1") == 0,
        "numeric reconstruct GEMM prefill must be 0 or 1");
    const bool numeric_prefill_projection_enabled =
        numeric_prefill_projection &&
        std::strcmp(numeric_prefill_projection, "1") == 0;
    const char* numeric_prefill_k7 = std::getenv(
        "NINFER_EXL3_PREFILL_RECONSTRUCT_GEMM_K7_NUMERIC");
    require(!numeric_prefill_k7 || std::strcmp(numeric_prefill_k7, "0") == 0 ||
            std::strcmp(numeric_prefill_k7, "1") == 0,
        "numeric K7 reconstruct GEMM prefill must be 0 or 1");
    const bool numeric_prefill_k7_enabled = numeric_prefill_k7 &&
        std::strcmp(numeric_prefill_k7, "1") == 0;
    const char* fast_same_weights_fp16kv_prefill = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_PREFILL");
    require(!fast_same_weights_fp16kv_prefill ||
            std::strcmp(fast_same_weights_fp16kv_prefill, "0") == 0 ||
            std::strcmp(fast_same_weights_fp16kv_prefill, "1") == 0,
        "FAST same-weight FP16-KV prefill must be 0 or 1");
    const bool fast_same_weights_fp16kv_prefill_enabled =
        fast_same_weights_fp16kv_prefill &&
        std::strcmp(fast_same_weights_fp16kv_prefill, "1") == 0;
    const char* fast_same_weights_fp16kv_decode = std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_DECODE");
    require(!fast_same_weights_fp16kv_decode ||
            std::strcmp(fast_same_weights_fp16kv_decode, "0") == 0 ||
            std::strcmp(fast_same_weights_fp16kv_decode, "1") == 0,
        "FAST same-weight FP16-KV decode must be 0 or 1");
    const bool fast_same_weights_fp16kv_decode_enabled =
        fast_same_weights_fp16kv_decode &&
        std::strcmp(fast_same_weights_fp16kv_decode, "1") == 0;
    require(!(numeric_prefill_projection_enabled && numeric_prefill_k7_enabled),
        "numeric broad and K7-only reconstruct GEMM are separate controls");
    require(!fast_same_weights_fp16kv_prefill_enabled ||
            (!numeric_prefill_projection_enabled && !numeric_prefill_k7_enabled),
        "FAST same-weight FP16-KV prefill is separate from numeric qualification controls");
    const bool numeric_prefill_workspace_enabled =
        numeric_prefill_projection_enabled || numeric_prefill_k7_enabled ||
        fast_same_weights_fp16kv_prefill_enabled ||
        fast_same_weights_fp16kv_decode_enabled;
    require(!impl->native_continuation16_enabled ||
        !numeric_prefill_workspace_enabled,
        "native continuation16 excludes numeric projection candidates");
    const char* oscar_requested = std::getenv("NINFER_OSCAR_EXL3");
    const bool fast_same_weights_fp16kv_scope =
        fast_same_weights_fp16kv_prefill_enabled && !impl->host_kv.enabled &&
        !impl->oscar_only && !impl->continuation_graph_b8_enabled &&
        (!oscar_requested || std::strcmp(oscar_requested, "0") == 0) &&
        impl->wide_prefill_enabled && impl->prefill_capacity == 1024;
    // The cuBLAS wide-prefill differential has its own same-weight scope.  It
    // reconstructs the packed trellis exactly into the ordinary FP16 slab and
    // never changes the resident KV representation, so it may share the
    // ordinary device-KV construction path while remaining default-off.
    const char* fast_wide_prefill_gemm = std::getenv(
        "NINFER_EXL3_FAST_MIA_PARITY_WIDE_PREFILL_GEMM");
    const bool fast_wide_prefill_gemm_scope =
        fast_wide_prefill_gemm &&
        std::strcmp(fast_wide_prefill_gemm, "1") == 0 &&
        !impl->host_kv.enabled && !impl->oscar_only &&
        !impl->continuation_graph_b8_enabled &&
        (!oscar_requested || std::strcmp(oscar_requested, "0") == 0) &&
        impl->wide_prefill_enabled && staged_prefill_enabled &&
        impl->prefill_capacity == 1024 && !numeric_prefill_workspace_enabled;
    const bool fast_same_weights_fp16kv_decode_scope =
        fast_same_weights_fp16kv_decode_enabled && !impl->host_kv.enabled &&
        !impl->oscar_only && !impl->continuation_graph_b8_enabled &&
        (!oscar_requested || std::strcmp(oscar_requested, "0") == 0);
    impl->fast_same_weights_fp16kv_prefill_enabled = fast_same_weights_fp16kv_scope;
    const char* gdn_bulk=std::getenv("NINFER_EXL3_FAST_GDN_BULK_PREFILL");
    require(!gdn_bulk || std::strcmp(gdn_bulk,"0")==0 ||
            std::strcmp(gdn_bulk,"1")==0,
            "GDN bulk prefill must be 0 or 1");
    impl->gdn_bulk_prefill_enabled=fast_same_weights_fp16kv_scope &&
        gdn_bulk && std::strcmp(gdn_bulk,"1")==0;
    const char* gdn_bulk_mlp=std::getenv("NINFER_EXL3_FAST_GDN_BULK_MLP");
    require(!gdn_bulk_mlp || std::strcmp(gdn_bulk_mlp,"0")==0 ||
            std::strcmp(gdn_bulk_mlp,"1")==0,
            "GDN bulk MLP must be 0 or 1");
    impl->gdn_bulk_mlp_enabled=fast_same_weights_fp16kv_scope &&
        gdn_bulk_mlp && std::strcmp(gdn_bulk_mlp,"1")==0;
    const char* gdn_bulk_mlp_short_k5=std::getenv(
        "NINFER_EXL3_FAST_GDN_BULK_MLP_SHORT_K5");
    require(!gdn_bulk_mlp_short_k5 ||
            std::strcmp(gdn_bulk_mlp_short_k5,"0")==0 ||
            std::strcmp(gdn_bulk_mlp_short_k5,"1")==0,
            "GDN bulk MLP short K5 must be 0 or 1");
    impl->gdn_bulk_mlp_short_k5_enabled=impl->gdn_bulk_mlp_enabled &&
        gdn_bulk_mlp_short_k5 &&
        std::strcmp(gdn_bulk_mlp_short_k5,"1")==0;
    require(!gdn_bulk_mlp_short_k5 ||
            std::strcmp(gdn_bulk_mlp_short_k5,"1")!=0 ||
            impl->gdn_bulk_mlp_short_k5_enabled,
            "GDN bulk MLP short K5 requires Fast90 bulk MLP admission");
    const char* gdn_bulk_mlp_rows=std::getenv("NINFER_EXL3_FAST_GDN_BULK_MLP_ROWS");
    require(!gdn_bulk_mlp_rows || std::strcmp(gdn_bulk_mlp_rows,"4096")==0 ||
            std::strcmp(gdn_bulk_mlp_rows,"8192")==0,
            "GDN bulk MLP rows must be 4096 or 8192");
    const char* wmma_split2=std::getenv("NINFER_EXL3_FAST_WMMA32_SPLIT2");
    require(!wmma_split2 || std::strcmp(wmma_split2,"0")==0 ||
            std::strcmp(wmma_split2,"1")==0,
            "WMMA32 split2 option must be 0 or 1");
    impl->fast_wmma32_split2_enabled=fast_same_weights_fp16kv_scope &&
        wmma_split2 && std::strcmp(wmma_split2,"1")==0;
    const char* wmma_split4=std::getenv("NINFER_EXL3_FAST_WMMA32_SPLIT4");
    require(!wmma_split4 || std::strcmp(wmma_split4,"0")==0 ||
            std::strcmp(wmma_split4,"1")==0,
            "WMMA32 split4 option must be 0 or 1");
    const bool split4_enabled=fast_same_weights_fp16kv_scope &&
        wmma_split4 && std::strcmp(wmma_split4,"1")==0;
    require(!split4_enabled || !impl->fast_wmma32_split2_enabled,
            "WMMA32 split2 and split4 are exclusive");
    impl->fast_wmma32_split_count=split4_enabled?4:
        (impl->fast_wmma32_split2_enabled?2:0);
    // The FA2 prefill route (same fast scope) uses four FP32 partial planes.
    impl->fast_wmma32_split_capacity=
        fast_same_weights_fp16kv_scope && exl3_fa2_prefill_enabled() ? 4 :
        impl->fast_wmma32_split_count;
    require(!impl->gdn_bulk_prefill_enabled || !impl->gdn_bulk_mlp_enabled,
            "GDN bulk prefix and MLP are isolated candidates");
    impl->gdn_bulk_capacity=impl->gdn_bulk_mlp_enabled
        ? std::min(gdn_bulk_mlp_rows && std::strcmp(gdn_bulk_mlp_rows,"8192")==0
                       ? 8192 : 4096,max_context_)
        : (impl->gdn_bulk_prefill_enabled ? std::min(4096,max_context_) : 0);
    impl->fast_same_weights_fp16kv_prefill_stats.enabled = fast_same_weights_fp16kv_scope;
    require(!numeric_prefill_workspace_enabled ||
            ((numeric_prefill_projection_enabled || numeric_prefill_k7_enabled) &&
             impl->host_kv.enabled && impl->wide_prefill_enabled &&
             impl->prefill_capacity == 1024) ||
            fast_same_weights_fp16kv_scope ||
            fast_same_weights_fp16kv_decode_scope,
        "numeric reconstruct GEMM requires target-only exact-host wide1024 or an admitted same-weight FP16-KV scope");
    const char* reconstructed_exact = std::getenv(
        "NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K7");
    require(!reconstructed_exact || std::strcmp(reconstructed_exact, "0") == 0 ||
            std::strcmp(reconstructed_exact, "1") == 0,
        "exact K7 reconstruction must be 0 or 1");
    const bool reconstructed_exact_enabled = reconstructed_exact &&
        std::strcmp(reconstructed_exact, "1") == 0;
    const int reconstruction_columns=exl3_reconstruction_slice_columns(
        std::getenv("NINFER_EXL3_RECONSTRUCTION_SLICE_COLUMNS"));
    require(reconstructed_exact_enabled || reconstruction_columns==5120,
        "bounded reconstruction slices require exact reconstruction enabled");
    const char* reconstructed_k6=std::getenv("NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K6_DOWN");
    const bool reconstructed_k6_enabled=exl3_reconstruction_k6_down_option(
        reconstructed_k6,reconstructed_exact_enabled);
    const char* reconstructed_k6_gate_up=std::getenv(
        "NINFER_EXL3_PREFILL_RECONSTRUCT_EXACT_K6_GATE_UP");
    const bool reconstructed_k6_gate_up_enabled=
        exl3_reconstruction_k6_gate_up_option(
            reconstructed_k6_gate_up,reconstructed_exact_enabled);
    require(!reconstructed_exact_enabled ||
        (impl->host_kv.enabled && !impl->oscar_only && impl->wide_prefill_enabled &&
         staged_prefill_enabled && impl->prefill_capacity == 1024 &&
         !numeric_prefill_workspace_enabled) || fast_wide_prefill_gemm_scope,
        "exact K7 reconstruction requires ordinary exact-host wide1024 or the admitted same-weight wide GEMM scope");
    const char* direct_partials = std::getenv("NINFER_EXL3_PREFILL_DIRECT_PARTIALS");
    require(!reconstructed_exact_enabled ||
        ((!direct_partials && qualified_default_group) ||
         (direct_partials && std::strcmp(direct_partials, "1") == 0)),
        "exact K7 reconstruction requires canonical direct partials");
    // Default to the qualified slab only for contexts with its required scratch paths.
    // An explicit 0 retains the original allocation; explicit 1 validates prerequisites below.
    const bool gdn_wide_slab_enabled = gdn_wide_slab
        ? std::strcmp(gdn_wide_slab, "1") == 0
        : impl->wide_prefill_enabled && shared_layer_enabled && impl->prefill_capacity > 16;
    if (gdn_wide_slab_enabled) {
        require(impl->wide_prefill_enabled && shared_layer_enabled,
                "GDN wide slab requires wide prefill and shared layer scratch");
    }
    const char* target_timing = std::getenv("NINFER_EXL3_TARGET_PROJECTION_TIMING");
    impl->target_projection_timing_opt_in =
        target_timing != nullptr && std::strcmp(target_timing, "1") == 0;
    const int row_storage=(impl->host_kv.enabled || impl->oscar_only) ? impl->prefill_capacity : max_context_;
    require(row_storage>0,"target context storage rows must be positive");
    const auto checked_extent=[&](std::size_t bytes_per_row) {
        Exl3ResourceInventory::Requirement required;
        required.add(Exl3ResourceInventory::Domain::device,static_cast<std::size_t>(row_storage),bytes_per_row);
        const auto bytes=required.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::device)];
        require(bytes<=std::numeric_limits<std::size_t>::max(),"target context storage extent overflow");
        return static_cast<std::size_t>(bytes);
    };
    const std::size_t hidden_bytes=checked_extent(kHidden*sizeof(std::uint16_t));
    const auto token_bytes=checked_extent(sizeof(std::int64_t));
    const auto visit_base_allocations=[&](auto&& allocate) {
        allocate(hidden_bytes,reinterpret_cast<void**>(&impl->hidden_a),"allocate E4A hidden A");
        allocate(hidden_bytes,reinterpret_cast<void**>(&impl->hidden_b),"allocate E4A hidden B");
        if (impl->gdn_bulk_prefill_enabled) {
            const auto bulk_rows=static_cast<std::size_t>(impl->gdn_bulk_capacity);
            allocate(bulk_rows*kHidden*sizeof(std::uint16_t),
                reinterpret_cast<void**>(&impl->gdn_bulk_h),"allocate GDN bulk normalized input");
            allocate(bulk_rows*10240*sizeof(std::uint16_t),
                reinterpret_cast<void**>(&impl->gdn_bulk_qkv),"allocate GDN bulk QKV");
            allocate(bulk_rows*6144*sizeof(std::uint16_t),
                reinterpret_cast<void**>(&impl->gdn_bulk_z),"allocate GDN bulk Z");
        }
        if (impl->gdn_bulk_mlp_enabled) {
            const auto bulk_rows=static_cast<std::size_t>(impl->gdn_bulk_capacity);
            allocate(bulk_rows*kHidden*sizeof(std::uint16_t),
                reinterpret_cast<void**>(&impl->gdn_mlp_post),"allocate GDN bulk post residual");
            allocate(bulk_rows*kHidden*sizeof(std::uint16_t),
                reinterpret_cast<void**>(&impl->gdn_mlp_input),"allocate GDN bulk MLP input");
            allocate(bulk_rows*17408*sizeof(std::uint16_t),
                reinterpret_cast<void**>(&impl->gdn_mlp_gate),"allocate GDN bulk gate");
            allocate(bulk_rows*17408*sizeof(std::uint16_t),
                reinterpret_cast<void**>(&impl->gdn_mlp_up),"allocate GDN bulk up");
        }
        if (impl->fast_wmma32_split_capacity) {
            constexpr std::size_t rows=1024;
            allocate(static_cast<std::size_t>(impl->fast_wmma32_split_capacity)*
                rows*kQHeads*kHeadDim*sizeof(float),
                reinterpret_cast<void**>(&impl->fast_wmma32_split2_output),
                "allocate WMMA32 split FP32 partial output");
            allocate(static_cast<std::size_t>(impl->fast_wmma32_split_capacity)*
                rows*kQHeads*2*sizeof(float),
                reinterpret_cast<void**>(&impl->fast_wmma32_split2_stats),
                "allocate WMMA32 split FP32 max and sum");
        }
        allocate(kHidden*sizeof(std::uint16_t),reinterpret_cast<void**>(&impl->final_norm),"allocate E4A final norm");
        allocate(kVocab*sizeof(std::uint16_t),reinterpret_cast<void**>(&impl->logits),"allocate E4A logits");
        allocate(token_bytes,reinterpret_cast<void**>(&impl->token_ids),"allocate E4A token ids");
        allocate(sizeof(int),reinterpret_cast<void**>(&impl->position_device),"allocate E4B2 position parameter");
        allocate(sizeof(std::int64_t),reinterpret_cast<void**>(&impl->draft_token_id),"allocate E5A2 draft token id");
    };
    std::size_t base_bytes=0,base_owners=0;
    visit_base_allocations([&](std::size_t bytes,void**,const char*) {
        base_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(base_bytes,bytes);++base_owners;
    });
    const auto materialize_base=[&] {
    const auto expected_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(impl->persistent_bytes,base_bytes);
    require(base_owners<=impl->allocations.max_size()-impl->allocations.size(),"target base owner capacity overflow");
    const auto expected_owners=impl->allocations.size()+base_owners;
    impl->allocations.reserve(expected_owners);
    visit_base_allocations([&](std::size_t bytes,void** pointer,const char* label){impl->allocate(bytes,pointer,label);});
    require(impl->persistent_bytes==expected_persistent && impl->allocations.size()==expected_owners,
        "target context base allocation requirement mismatch");
    };
    float* exact_scores=nullptr;
    float* numeric_attention_splitk_workspace=nullptr;
    std::size_t numeric_attention_splitk_workspace_bytes=0;
    const char* parallel_exact=std::getenv("NINFER_EXL3_EXACT_ATTENTION_PARALLEL");
    const char* numeric_attention_splitk=
        std::getenv("NINFER_EXL3_NUMERIC_ATTENTION_TILED");
    require(!numeric_attention_splitk || std::strcmp(numeric_attention_splitk,"0")==0 ||
        std::strcmp(numeric_attention_splitk,"1")==0,
        "numeric tiled attention must be 0 or 1");
    const bool numeric_attention_splitk_enabled=numeric_attention_splitk &&
        std::strcmp(numeric_attention_splitk,"1")==0;
    require(!reconstructed_exact_enabled || !numeric_attention_splitk_enabled,
        "exact K7 reconstruction and numeric attention require separate qualification");
    require(!impl->native_continuation16_enabled || !numeric_attention_splitk_enabled,
        "native continuation16 excludes numeric attention");
    require(!numeric_attention_splitk_enabled ||
        (impl->host_kv.enabled && !impl->oscar_only && parallel_exact &&
         std::strcmp(parallel_exact,"1")==0),
        "numeric tiled attention requires exact-host parallel attention");
    const bool exact_scores_enabled=!impl->oscar_only && parallel_exact &&
        std::strcmp(parallel_exact,"1")==0 && !numeric_attention_splitk_enabled;
    const auto exact_score_requirement=exact_scores_enabled?exl3_exact_attention_score_bytes(16,max_context_):0;
    if(numeric_attention_splitk_enabled) {
        numeric_attention_splitk_workspace_bytes=
            exl3_numeric_attention_splitk_workspace_bytes(16,max_context_);
        require(numeric_attention_splitk_workspace_bytes>0,
            "numeric tiled attention workspace extent");
    }
    const auto materialize_attention=[&] {
      if(exact_score_requirement) {
        impl->allocate(exact_score_requirement,reinterpret_cast<void**>(&exact_scores),"allocate shared exact attention scores");
        impl->host_kv.exact_attention_score_bytes=exact_score_requirement;
      }
      if(numeric_attention_splitk_enabled) {
        impl->allocate(numeric_attention_splitk_workspace_bytes,
            reinterpret_cast<void**>(&numeric_attention_splitk_workspace),
            "allocate shared numeric split-K attention workspace");
        impl->host_kv.numeric_attention_scratch_bytes=numeric_attention_splitk_workspace_bytes;
      }
    };
    const char* exact_q_shared=std::getenv("NINFER_EXL3_EXACT_ATTENTION_Q_SHARED");
    const char* query_pair=std::getenv("NINFER_EXL3_EXACT_ATTENTION_QUERY_PAIR");
    const bool query_pair_enabled=exl3_attention_query_pair_option(query_pair,exact_scores_enabled);
    require(!exact_q_shared || std::strcmp(exact_q_shared,"0")==0 ||
        std::strcmp(exact_q_shared,"1")==0,
        "exact attention Q-shared must be 0 or 1");
    // Qualified exact-host default. Explicit =0 retains the scalar-Q-load
    // parallel kernel; explicit =1 also permits focused non-host diagnostics.
    const bool exact_q_shared_enabled=exact_scores_enabled &&
        ((impl->host_kv.enabled && (!exact_q_shared || std::strcmp(exact_q_shared,"0")!=0)) ||
         (exact_q_shared && std::strcmp(exact_q_shared,"1")==0));
    require(!exact_q_shared_enabled || exact_scores_enabled,
        "exact attention Q-shared requires parallel exact attention");
    const char* exact_k_half2=std::getenv("NINFER_EXL3_EXACT_ATTENTION_K_HALF2");
    require(!exact_k_half2 || std::strcmp(exact_k_half2,"0")==0 ||
        std::strcmp(exact_k_half2,"1")==0,
        "exact attention K-half2 must be 0 or 1");
    // The half2 load changes only instruction width; the paired FP32 FMAs
    // retain d=0..255 order. It is qualified together with Q-shared.
    const bool exact_k_half2_enabled=exact_q_shared_enabled &&
        (!exact_k_half2 || std::strcmp(exact_k_half2,"0")!=0);
    require(!exact_k_half2_enabled || exact_q_shared_enabled,
        "exact attention K-half2 requires Q-shared");
    const char* exact_v_half2=std::getenv("NINFER_EXL3_EXACT_ATTENTION_V_HALF2");
    require(!exact_v_half2 || std::strcmp(exact_v_half2,"0")==0 ||
        std::strcmp(exact_v_half2,"1")==0,
        "exact attention V-half2 must be 0 or 1");
    // Qualified exact-host default. The paired load changes only how two
    // adjacent FP16 values reach registers; each dimension retains its own
    // key=0..count-1 scalar FP32 accumulation sequence.
    const bool exact_v_half2_enabled=exact_k_half2_enabled &&
        ((impl->host_kv.enabled && (!exact_v_half2 || std::strcmp(exact_v_half2,"0")!=0)) ||
         (exact_v_half2 && std::strcmp(exact_v_half2,"1")==0));
    require(!exact_v_half2_enabled || exact_k_half2_enabled,
        "exact attention V-half2 requires K-half2");
    const char* exact_gqa_pair=std::getenv("NINFER_EXL3_EXACT_ATTENTION_GQA_PAIR");
    require(!exact_gqa_pair || std::strcmp(exact_gqa_pair,"0")==0 ||
        std::strcmp(exact_gqa_pair,"1")==0,
        "exact attention GQA pair must be 0 or 1");
    // Qualified exact-host default. Explicit =0 retains the single-head
    // V-half2 kernel; explicit =1 also permits focused non-host diagnostics.
    const bool exact_gqa_pair_enabled=exact_v_half2_enabled &&
        ((impl->host_kv.enabled && (!exact_gqa_pair || std::strcmp(exact_gqa_pair,"0")!=0)) ||
         (exact_gqa_pair && std::strcmp(exact_gqa_pair,"1")==0));
    require(!exact_gqa_pair_enabled || exact_v_half2_enabled,
        "exact attention GQA pair requires V-half2");
    const char* exact_gqa_triple=std::getenv("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE");
    require(!exact_gqa_triple || std::strcmp(exact_gqa_triple,"0")==0 ||
        std::strcmp(exact_gqa_triple,"1")==0,
        "exact attention GQA triple must be 0 or 1");
    // Qualified exact-host default. Explicit =0 retains the promoted two-head
    // schedule; explicit =1 permits focused non-host diagnostics.
    const bool exact_gqa_triple_enabled=exact_gqa_pair_enabled &&
        ((impl->host_kv.enabled && (!exact_gqa_triple || std::strcmp(exact_gqa_triple,"0")!=0)) ||
         (exact_gqa_triple && std::strcmp(exact_gqa_triple,"1")==0));
    require(!exact_gqa_triple_enabled || exact_gqa_pair_enabled,
        "exact attention GQA triple requires GQA pair");
    const char* exact_gqa_triple_values128=std::getenv("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES128");
    require(!exact_gqa_triple_values128 || std::strcmp(exact_gqa_triple_values128,"0")==0 ||
        std::strcmp(exact_gqa_triple_values128,"1")==0,
        "exact attention GQA triple values128 must be 0 or 1");
    const bool exact_gqa_triple_values128_enabled=exact_gqa_triple_enabled &&
        exact_gqa_triple_values128 && std::strcmp(exact_gqa_triple_values128,"1")==0;
    require(!exact_gqa_triple_values128_enabled || exact_gqa_triple_enabled,
        "exact attention GQA triple values128 requires GQA triple");
    const char* exact_gqa_triple_softmax_staged=std::getenv("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_SOFTMAX_STAGED");
    require(!exact_gqa_triple_softmax_staged || std::strcmp(exact_gqa_triple_softmax_staged,"0")==0 ||
        std::strcmp(exact_gqa_triple_softmax_staged,"1")==0,
        "exact attention GQA triple staged softmax must be 0 or 1");
    // Qualified exact-host default. Cooperative loads stage each score tile,
    // while the three head accumulators retain their original serial key order.
    const bool exact_gqa_triple_softmax_staged_enabled=exact_gqa_triple_enabled &&
        ((impl->host_kv.enabled && (!exact_gqa_triple_softmax_staged ||
            std::strcmp(exact_gqa_triple_softmax_staged,"0")!=0)) ||
         (exact_gqa_triple_softmax_staged && std::strcmp(exact_gqa_triple_softmax_staged,"1")==0));
    const char* exact_gqa_six=std::getenv("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX");
    require(!exact_gqa_six || std::strcmp(exact_gqa_six,"0")==0 ||
        std::strcmp(exact_gqa_six,"1")==0,
        "exact attention GQA six must be 0 or 1");
    const bool exact_gqa_six_enabled=exact_gqa_triple_softmax_staged_enabled &&
        exact_gqa_six && std::strcmp(exact_gqa_six,"1")==0;
    require(!exact_gqa_six_enabled || exact_gqa_triple_softmax_staged_enabled,
        "exact attention GQA six requires staged GQA triple");
    const char* exact_gqa_six_scores=
        std::getenv("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SCORES");
    require(!exact_gqa_six_scores || std::strcmp(exact_gqa_six_scores,"0")==0 ||
        std::strcmp(exact_gqa_six_scores,"1")==0,
        "exact attention GQA six scores must be 0 or 1");
    require(!(exact_gqa_six_enabled && exact_gqa_six_scores &&
        std::strcmp(exact_gqa_six_scores,"1")==0),
        "full GQA six and score-only GQA six are separate controls");
    // Qualified exact-host default: reuse each represented K load across all
    // six Q heads, but retain the promoted triple staged-value concurrency.
    // Explicit zero keeps the prior triple-score kernel. Explicit full GQA-six
    // suppresses this default and remains a separate diagnostic.
    const bool exact_gqa_six_scores_enabled=exact_gqa_triple_softmax_staged_enabled &&
        !exact_gqa_six_enabled &&
        ((impl->host_kv.enabled && (!exact_gqa_six_scores ||
            std::strcmp(exact_gqa_six_scores,"0")!=0)) ||
         (exact_gqa_six_scores && std::strcmp(exact_gqa_six_scores,"1")==0));
    const char* exact_gqa_six_extent_shards=std::getenv(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_EXTENT_SHARDS");
    require(!exact_gqa_six_extent_shards ||
        std::strcmp(exact_gqa_six_extent_shards,"0")==0 ||
        std::strcmp(exact_gqa_six_extent_shards,"1")==0,
        "exact attention GQA six extent shards must be 0 or 1");
    // Qualified exact-host default: avoid score CTAs whose first 256-key tile
    // lies beyond the public eager extent. Explicit zero retains the fixed-six
    // fallback; captured graphs still launch six shards in the layer.
    const bool exact_gqa_six_extent_shards_enabled=exact_gqa_six_scores_enabled &&
        ((impl->host_kv.enabled && (!exact_gqa_six_extent_shards ||
            std::strcmp(exact_gqa_six_extent_shards,"0")!=0)) ||
         (exact_gqa_six_extent_shards &&
            std::strcmp(exact_gqa_six_extent_shards,"1")==0));
    require(!exact_gqa_six_extent_shards_enabled || exact_gqa_six_scores_enabled,
        "exact attention GQA six extent shards require six-head scores");
    const char* exact_gqa_six_values_sharded=
        std::getenv("NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_VALUES_SHARDED");
    require(!exact_gqa_six_values_sharded ||
        std::strcmp(exact_gqa_six_values_sharded,"0")==0 ||
        std::strcmp(exact_gqa_six_values_sharded,"1")==0,
        "exact attention GQA six sharded values must be 0 or 1");
    const bool exact_gqa_six_values_sharded_enabled=exact_gqa_six_scores_enabled &&
        exact_gqa_six_values_sharded &&
        std::strcmp(exact_gqa_six_values_sharded,"1")==0;
    require(!(exact_gqa_six_enabled && exact_gqa_six_values_sharded_enabled),
        "full GQA six and sharded-value GQA six are separate controls");
    const char* exact_gqa_six_softmax_triple_values=std::getenv(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES");
    require(!exact_gqa_six_softmax_triple_values ||
        std::strcmp(exact_gqa_six_softmax_triple_values,"0")==0 ||
        std::strcmp(exact_gqa_six_softmax_triple_values,"1")==0,
        "exact attention GQA six softmax triple values must be 0 or 1");
    // Qualified exact-host default. Normalize the six represented query heads
    // once, then retain the independently ordered three-head value accumulators.
    // Explicit zero preserves the former duplicated-softmax route; explicit one
    // also permits focused non-host diagnostics.
    const bool exact_gqa_six_softmax_triple_values_enabled=
        exact_gqa_six_scores_enabled && !exact_gqa_six_values_sharded_enabled &&
        ((impl->host_kv.enabled && (!exact_gqa_six_softmax_triple_values ||
            std::strcmp(exact_gqa_six_softmax_triple_values,"0")!=0)) ||
         (exact_gqa_six_softmax_triple_values &&
            std::strcmp(exact_gqa_six_softmax_triple_values,"1")==0));
    const char* exact_gqa_six_softmax_triple_values_pair_dimensions=std::getenv(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_TRIPLE_VALUES_PAIR_DIMENSIONS");
    require(!exact_gqa_six_softmax_triple_values_pair_dimensions ||
        std::strcmp(exact_gqa_six_softmax_triple_values_pair_dimensions,"0")==0 ||
        std::strcmp(exact_gqa_six_softmax_triple_values_pair_dimensions,"1")==0,
        "exact attention GQA six softmax triple values pair dimensions must be 0 or 1");
    const bool exact_gqa_six_softmax_triple_values_pair_dimensions_enabled=
        exact_gqa_six_softmax_triple_values_pair_dimensions &&
        std::strcmp(exact_gqa_six_softmax_triple_values_pair_dimensions,"1")==0;
    require(!exact_gqa_six_softmax_triple_values_pair_dimensions_enabled ||
        exact_gqa_six_softmax_triple_values_enabled,
        "pair-dimension triple values require selected six-head softmax triple values");
    const char* exact_gqa_six_softmax_six_values_single_load=std::getenv(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_SOFTMAX_SIX_VALUES_SINGLE_LOAD");
    require(!exact_gqa_six_softmax_six_values_single_load ||
        std::strcmp(exact_gqa_six_softmax_six_values_single_load,"0")==0 ||
        std::strcmp(exact_gqa_six_softmax_six_values_single_load,"1")==0,
        "exact attention GQA six softmax six values single load must be 0 or 1");
    const bool exact_gqa_six_softmax_six_values_single_load_enabled=
        exact_gqa_six_softmax_six_values_single_load &&
        std::strcmp(exact_gqa_six_softmax_six_values_single_load,"1")==0;
    require(!exact_gqa_six_softmax_six_values_single_load_enabled ||
        exact_gqa_six_softmax_triple_values_enabled,
        "single-load six values require selected six-head softmax triple values");
    const char* exact_gqa_six_packed_triples=std::getenv(
        "NINFER_EXL3_EXACT_ATTENTION_GQA_SIX_PACKED_TRIPLES");
    require(!exact_gqa_six_packed_triples ||
        std::strcmp(exact_gqa_six_packed_triples,"0")==0 ||
        std::strcmp(exact_gqa_six_packed_triples,"1")==0,
        "exact attention GQA six packed triples must be 0 or 1");
    const bool exact_gqa_six_packed_triples_enabled=
        exact_gqa_six_scores_enabled && !exact_gqa_six_values_sharded_enabled &&
        exact_gqa_six_packed_triples &&
        std::strcmp(exact_gqa_six_packed_triples,"1")==0;
    const char* exact_gqa_triple_values4=
        std::getenv("NINFER_EXL3_EXACT_ATTENTION_GQA_TRIPLE_VALUES4");
    require(!exact_gqa_triple_values4 ||
        std::strcmp(exact_gqa_triple_values4,"0")==0 ||
        std::strcmp(exact_gqa_triple_values4,"1")==0,
        "exact attention GQA triple values4 must be 0 or 1");
    const bool exact_gqa_triple_values4_enabled=
        exact_gqa_triple_softmax_staged_enabled && !exact_gqa_six_enabled &&
        !exact_gqa_six_values_sharded_enabled && exact_gqa_triple_values4 &&
        std::strcmp(exact_gqa_triple_values4,"1")==0;
    require(!(exact_gqa_triple_values128_enabled&&exact_gqa_triple_values4_enabled),
        "GQA triple values128 and values4 are separate controls");
    require(!(exact_gqa_six_softmax_triple_values_enabled&&
        exact_gqa_triple_values4_enabled),
        "six-head standalone softmax and values4 are separate controls");
    require(!(exact_gqa_six_packed_triples_enabled&&
        (exact_gqa_six_softmax_triple_values_enabled||exact_gqa_triple_values4_enabled)),
        "packed triples and prior value experiments are separate controls");
    if(impl->host_kv.enabled || impl->oscar_only) {
        require(impl->oscar_only || exact_scores_enabled ||
            numeric_attention_splitk_enabled,
            "host KV requires exact or numeric parallel attention");
    }
    const bool streamed_kv=impl->host_kv.enabled || impl->oscar_only;
    const auto kv_rows=impl->oscar_only?256:max_context_;
    require(kv_rows>0,"target KV storage rows must be positive");
    Exl3ResourceInventory::Requirement kv_plane_requirement;
    kv_plane_requirement.add(Exl3ResourceInventory::Domain::device,
        static_cast<std::size_t>(kv_rows),kKVHeads*kHeadDim*sizeof(std::uint16_t));
    const auto kv_plane_extent=kv_plane_requirement.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::device)];
    require(kv_plane_extent<=std::numeric_limits<std::size_t>::max(),"target KV plane extent overflow");
    const auto kv_plane_bytes=static_cast<std::size_t>(kv_plane_extent);
    const auto visit_kv_allocations=[&](auto&& allocate) {
        if(streamed_kv) {
            allocate(impl->host_layer_k,"allocate streamed layer K");
            allocate(impl->host_layer_v,"allocate streamed layer V");
        } else for(int layer=0;layer<kLayers;++layer) {
            if(!impl->model->layers[layer].full_attention)continue;
            allocate(impl->cache_k[layer],"allocate E4A K cache");
            allocate(impl->cache_v[layer],"allocate E4A V cache");
        }
    };
    std::size_t required_kv_bytes=0,required_kv_owners=0;
    visit_kv_allocations([&](auto&,const char*) {
        ++required_kv_owners;
        required_kv_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(required_kv_bytes,kv_plane_bytes);
    });
    const auto materialize_kv=[&] {
    const auto expected_kv_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(impl->persistent_bytes,required_kv_bytes);
    unsigned allocation_index=0;
    visit_kv_allocations([&](auto& owner,const char* label) {
        const auto fault=allocation_index++==0 && (startup_fault==14 || startup_fault==15)?startup_fault-13:0;
        auto prepared=[&] {
            if(!authority)return DeviceAllocation::create_shared(kv_plane_bytes,label,fault);
            auto credits=authority->reserve_constructor_credits(kv_plane_bytes,
                DeviceAllocation::owner_metadata_bytes()+DeviceAllocation::shared_control_bytes);
            return DeviceAllocation::create_shared(kv_plane_bytes,label,fault,
                std::move(credits.device),std::move(credits.metadata));
        }();
        owner=std::move(prepared);
        impl->persistent_bytes+=kv_plane_bytes;
    });
    require(impl->persistent_bytes==expected_kv_persistent,"target KV allocation requirement mismatch");
    if(impl->host_kv.enabled)impl->host_kv.layer_workspace_bytes=required_kv_bytes;
    };
    Exl3ResourceInventory::Requirement tap_requirement;
    if(capture_taps)tap_requirement.add(Exl3ResourceInventory::Domain::device,kTapLayers.size()+1,hidden_bytes);
    const auto tap_extent=tap_requirement.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::device)];
    require(tap_extent<=std::numeric_limits<std::size_t>::max(),"target tap extent overflow");
    const auto materialize_taps=[&] {
      if(capture_taps) {
        const std::size_t tap_bytes=hidden_bytes;
        const auto expected_tap_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            impl->persistent_bytes,static_cast<std::size_t>(tap_extent));
        for (std::size_t i = 0; i < kTapLayers.size(); ++i) {
            auto tap = impl->make_private_allocation(tap_bytes, "allocate E4A hidden tap");
            impl->persistent_bytes += tap_bytes;
            impl->taps[i] = std::move(tap);
        }
        impl->embedding_trace = impl->make_private_allocation(hidden_bytes, "allocate E4A embedding trace");
        impl->persistent_bytes += hidden_bytes;
        require(impl->persistent_bytes==expected_tap_persistent,"target tap allocation requirement mismatch");
      }
    };
    Exl3CudaAccumulationView accumulation;
    Exl3CudaAccumulationView paired_up_accumulation;
    Exl3CudaTransformView paired_up_transformed;
    Exl3LinearWorkspaceRequirements shared_linear_requirement;
    if(impl->wide_prefill_enabled && (shared_accumulation_enabled || shared_transform_enabled))
        shared_linear_requirement=Exl3LinearWorkspaceRequirements::derive(17408,17408,impl->prefill_capacity);
    if (impl->wide_prefill_enabled && shared_accumulation_enabled) {
        // All target layer projections are ordered. Completed projection partials
        // are reused. allocations outlives both layer arrays in Impl.
        accumulation.bytes = shared_linear_requirement.accumulation_bytes;
    }
    const auto* prefill_gate_up_pair=std::getenv(
        "NINFER_EXL3_TARGET_PREFILL_GATE_UP_PAIR");
    require(!prefill_gate_up_pair || std::strcmp(prefill_gate_up_pair,"0")==0 ||
        std::strcmp(prefill_gate_up_pair,"1")==0,
        "target prefill gate/up pair must be 0 or 1");
    const auto* fast_m1_gate_up_pair=std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_GATE_UP_PAIR");
    require(!fast_m1_gate_up_pair || std::strcmp(fast_m1_gate_up_pair,"0")==0 ||
        std::strcmp(fast_m1_gate_up_pair,"1")==0,
        "FAST same-weight FP16-KV M1 gate/up pair must be 0 or 1");
    const auto* fast_gdn_m1_gate_up_pair=std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_GDN_M1_GATE_UP_PAIR");
    require(!fast_gdn_m1_gate_up_pair ||
        std::strcmp(fast_gdn_m1_gate_up_pair,"0")==0 ||
        std::strcmp(fast_gdn_m1_gate_up_pair,"1")==0,
        "FAST same-weight FP16-KV GDN M1 gate/up pair must be 0 or 1");
    const auto* fast_m1_kv_pair=std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_KV_PAIR");
    require(!fast_m1_kv_pair || std::strcmp(fast_m1_kv_pair,"0")==0 ||
        std::strcmp(fast_m1_kv_pair,"1")==0,
        "FAST same-weight FP16-KV M1 K/V pair must be 0 or 1");
    const auto* fast_m1_kv_wide_pair=std::getenv(
        "NINFER_EXL3_FAST_SAME_WEIGHTS_FP16KV_M1_KV_WIDE_PAIR");
    require(!fast_m1_kv_wide_pair || std::strcmp(fast_m1_kv_wide_pair,"0")==0 ||
        std::strcmp(fast_m1_kv_wide_pair,"1")==0,
        "FAST same-weight FP16-KV M1 wide K/V pair must be 0 or 1");
    if(impl->wide_prefill_enabled && shared_accumulation_enabled &&
       ((prefill_gate_up_pair && std::strcmp(prefill_gate_up_pair,"1")==0) ||
        (fast_m1_gate_up_pair && std::strcmp(fast_m1_gate_up_pair,"1")==0) ||
        (fast_m1_kv_pair && std::strcmp(fast_m1_kv_pair,"1")==0) ||
        (fast_m1_kv_wide_pair && std::strcmp(fast_m1_kv_wide_pair,"1")==0) ||
        (fast_gdn_m1_gate_up_pair && std::strcmp(fast_gdn_m1_gate_up_pair,"1")==0)))
        paired_up_accumulation.bytes=shared_linear_requirement.accumulation_bytes;
    Exl3CudaTransformView transformed;
    if (impl->wide_prefill_enabled && shared_transform_enabled) {
        // Layer projections call forward() sequentially on the context stream;
        // no layer retains a transformed input after that projection completes.
        transformed.bytes = shared_linear_requirement.transformed_bytes;
    }
    if(impl->wide_prefill_enabled && shared_transform_enabled &&
       ((fast_m1_gate_up_pair && std::strcmp(fast_m1_gate_up_pair,"1")==0) ||
        (fast_m1_kv_pair && std::strcmp(fast_m1_kv_pair,"1")==0) ||
        (fast_m1_kv_wide_pair && std::strcmp(fast_m1_kv_wide_pair,"1")==0) ||
        (fast_gdn_m1_gate_up_pair && std::strcmp(fast_gdn_m1_gate_up_pair,"1")==0)))
        paired_up_transformed.bytes=shared_linear_requirement.transformed_bytes;
    Exl3CudaLayerScratchView scratch;
    const auto* coalesce_scratch=std::getenv("NINFER_EXL3_ATTENTION_COALESCE_INPUT_MLP");
    require(!coalesce_scratch || std::strcmp(coalesce_scratch,"0")==0 || std::strcmp(coalesce_scratch,"1")==0,
        "attention scratch coalescing must be 0 or 1");
    impl->coalesce_attention_input_mlp=coalesce_scratch && std::strcmp(coalesce_scratch,"1")==0;
    require(!impl->coalesce_attention_input_mlp || (!impl->oscar && !impl->oscar_only),
        "attention scratch coalescing requires ordinary context");
    Exl3GdnWideScratchView gdn_wide_scratch;
    if (impl->wide_prefill_enabled && shared_layer_enabled) {
        scratch.bytes=std::max(Exl3GdnLayer::shared_scratch_bytes(impl->prefill_capacity),
                              Exl3FullAttentionLayer::shared_scratch_bytes(impl->prefill_capacity,impl->coalesce_attention_input_mlp));
        if(impl->coalesce_attention_input_mlp)
            impl->host_kv.coalesced_attention_bytes_saved=std::max(Exl3GdnLayer::shared_scratch_bytes(impl->prefill_capacity),
                Exl3FullAttentionLayer::shared_scratch_bytes(impl->prefill_capacity))-scratch.bytes;
    }
    if (gdn_wide_slab_enabled) {
        gdn_wide_scratch.bytes = Exl3GdnLayer::wide_scratch_bytes(impl->prefill_capacity);
    }
    const auto visit_shared_allocations=[&](auto&& allocate) {
        if(accumulation.bytes)allocate(accumulation.bytes,reinterpret_cast<void**>(&accumulation.data),"allocate shared target accumulation");
        if(paired_up_accumulation.bytes)allocate(paired_up_accumulation.bytes,
            reinterpret_cast<void**>(&paired_up_accumulation.data),
            "allocate paired target up accumulation");
        if(transformed.bytes)allocate(transformed.bytes,reinterpret_cast<void**>(&transformed.data),"allocate shared target input transform");
        if(paired_up_transformed.bytes)allocate(paired_up_transformed.bytes,
            reinterpret_cast<void**>(&paired_up_transformed.data),
            "allocate paired GDN up input transform");
        if(scratch.bytes)allocate(scratch.bytes,&scratch.data,"allocate shared target layer scratch");
        if(gdn_wide_scratch.bytes)allocate(gdn_wide_scratch.bytes,&gdn_wide_scratch.data,"allocate shared GDN wide slab");
    };
    std::size_t shared_bytes=0,shared_owners=0;
    visit_shared_allocations([&](std::size_t bytes,void**,const char*) {
        shared_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(shared_bytes,bytes);++shared_owners;
    });
    const auto materialize_shared=[&] {
    if(shared_owners) {
        const auto expected_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(impl->persistent_bytes,shared_bytes);
        require(shared_owners<=impl->allocations.max_size()-impl->allocations.size(),"target shared owner capacity overflow");
        const auto expected_owners=impl->allocations.size()+shared_owners;
        impl->allocations.reserve(expected_owners);
        visit_shared_allocations([&](std::size_t bytes,void** pointer,const char* label){impl->allocate(bytes,pointer,label);});
        require(impl->persistent_bytes==expected_bytes && impl->allocations.size()==expected_owners,
            "target shared allocation requirement mismatch");
    }
    };
    const bool fast_same_weights_all_model_shapes =
        fast_same_weights_fp16kv_scope ||
        fast_same_weights_fp16kv_decode_scope;
    const int numeric_workspace_in = fast_same_weights_all_model_shapes
        ? 17408 : (numeric_prefill_k7_enabled ? 17408 : kHidden);
    const int numeric_workspace_out = fast_same_weights_all_model_shapes
        ? 17408 : (numeric_prefill_k7_enabled ? kHidden : 17408);
    const bool numeric_workspace_transpose =
        fast_same_weights_all_model_shapes || !numeric_prefill_k7_enabled;
    const auto numeric_requirement=numeric_prefill_workspace_enabled?
        Exl3CudaReconstructGemmWorkspace::workspace_bytes_required(
            numeric_workspace_in,numeric_workspace_out,
            impl->gdn_bulk_capacity ? impl->gdn_bulk_capacity : impl->prefill_capacity,
            numeric_workspace_transpose):0;
    const auto materialize_numeric=[&] {
      if(numeric_requirement) {
        const auto expected_numeric_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            impl->persistent_bytes,numeric_requirement);
        impl->numeric_prefill_projection_workspace =
            std::make_unique<Exl3CudaReconstructGemmWorkspace>(
                numeric_workspace_in,numeric_workspace_out,
                impl->gdn_bulk_capacity ? impl->gdn_bulk_capacity : impl->prefill_capacity,
                numeric_workspace_transpose,
                fast_same_weights_all_model_shapes);
        require(impl->numeric_prefill_projection_workspace->workspace_bytes()==numeric_requirement,
            "target numeric projection requirement mismatch");
        impl->persistent_bytes=expected_numeric_persistent;
      }
    };
    Exl3ReconstructedExactView reconstructed_exact_view;
    if(reconstructed_exact_enabled && defer_reconstruction) {
        impl->deferred_reconstruction_bytes=17408ull*reconstruction_columns*sizeof(std::uint16_t);
        impl->deferred_reconstruction_k6=reconstructed_k6_enabled;
        impl->deferred_reconstruction_k6_gate_up=reconstructed_k6_gate_up_enabled;
    }
    const auto reconstruction_requirement=(reconstructed_exact_enabled && !defer_reconstruction)?
        17408ull*reconstruction_columns*sizeof(std::uint16_t):0;
    const auto materialize_reconstruction=[&] {
      if(reconstruction_requirement) {
        const auto expected_reconstruction_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            impl->persistent_bytes,reconstruction_requirement);
        reconstructed_exact_view.bytes = reconstruction_requirement;
        impl->reconstruction_backing=ReconstructionBacking::create(reconstructed_exact_view.bytes);
        impl->reconstruction_backing->stream.bind_model(impl->model,
            reconstructed_exact_view.bytes,reconstructed_k6_enabled,
            impl->reconstruction_backing->allocation.ptr,
            reconstructed_k6_gate_up_enabled);
        impl->persistent_bytes=expected_reconstruction_persistent;
        reconstructed_exact_view.data=static_cast<std::uint16_t*>(impl->reconstruction_backing->allocation.ptr);
        reconstructed_exact_view.backing_owner=impl->reconstruction_backing;
        reconstructed_exact_view.model_owner=impl->model;
        reconstructed_exact_view.stats = &impl->reconstruction_backing->stats;
        reconstructed_exact_view.allow_k6_down = reconstructed_k6_enabled;
        reconstructed_exact_view.allow_k6_gate_up = reconstructed_k6_gate_up_enabled;
        reconstructed_exact_view.ordered_stream = &impl->reconstruction_backing->stream;
      }
    };
    const auto head_requirement=Exl3LinearWorkspaceRequirements::derive(kHidden,kVocab,1);
    const auto* resident_option=std::getenv("NINFER_EXL3_PREFILL_GDN_RESIDENT");
    const bool resident_required=impl->gaming[Gopt::GdnSmallResident] ||
        (resident_option?std::strcmp(resident_option,"1")==0:
        (qualified_default_group && scratch.bytes!=0));
    std::array<std::size_t,kLayers> layer_requirements{};
    auto required_layer_group=head_requirement.owned_bytes;
    for(int layer=0;layer<kLayers;++layer) {
        layer_requirements[layer]=impl->model->layers[layer].full_attention?
            Exl3FullAttentionLayer::workspace_bytes_required(impl->prefill_capacity,
                accumulation.bytes!=0,transformed.bytes!=0,scratch.bytes!=0,
                impl->coalesce_attention_input_mlp):
            Exl3GdnLayer::workspace_bytes_required(impl->prefill_capacity,
                accumulation.bytes!=0,transformed.bytes!=0,scratch.bytes!=0,
                gdn_wide_scratch.bytes!=0,resident_required);
        required_layer_group=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            required_layer_group,layer_requirements[layer]);
    }
    const char* media=std::getenv("NINFER_EXL3_MEDIA_EXECUTION_RESEARCH");
    require(!media||std::strcmp(media,"0")==0||std::strcmp(media,"1")==0,"media research option must be0 or1");
    const bool media_enabled=enable_qualified_media || (media && std::strcmp(media,"1")==0);
    require(!media_enabled || (impl->host_kv.enabled&&!impl->oscar_only&&!impl->continuation_graph_b8_enabled),
        "media research requires ordinary eager host context");
    const char* device_prefix=std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX");
    const char* segmented_prefix=std::getenv("NINFER_EXL3_SEGMENTED_DEVICE_PREFIX");
    const char* forward_publish_prefix=std::getenv(
        "NINFER_EXL3_HOST_KV_DEVICE_PREFIX_FORWARD_PUBLISH");
    require(!segmented_prefix || std::strcmp(segmented_prefix,"0")==0 || std::strcmp(segmented_prefix,"1")==0,
        "segmented device prefix must be0 or1");
    impl->segmented_device_prefix=allocate_device_prefix && segmented_prefix && std::strcmp(segmented_prefix,"1")==0;
    require(!forward_publish_prefix || std::strcmp(forward_publish_prefix,"0")==0 ||
        std::strcmp(forward_publish_prefix,"1")==0,
        "device prefix forward publication must be0 or1");
    impl->device_prefix_forward_publish=forward_publish_prefix &&
        std::strcmp(forward_publish_prefix,"1")==0;
    require(!device_prefix||std::strcmp(device_prefix,"0")==0||std::strcmp(device_prefix,"1")==0,"device prefix option must be0 or1");
    int cache_rows=0;
    if(allocate_device_prefix && device_prefix && std::strcmp(device_prefix,"1")==0) {
        const char* row_option=std::getenv("NINFER_EXL3_HOST_KV_DEVICE_PREFIX_ROWS");
        require(!row_option||std::strcmp(row_option,"4096")==0||std::strcmp(row_option,"16384")==0,"device prefix row menu4096/16384");
        cache_rows=row_option?std::atoi(row_option):4096;
        require(!impl->device_prefix_forward_publish ||
            (cache_rows==16384 && !impl->segmented_device_prefix),
            "device prefix forward publication requires unsegmented 16K storage");
        require(impl->host_kv.enabled&&impl->host_kv_pinned_chunks&&!impl->oscar_only&&!impl->continuation_graph_b8_enabled&&max_context_>=cache_rows,
            "device prefix requires ordinary eager pinned HostKV context covering reserved rows");
    }
    const auto* greedy=std::getenv("NINFER_EXL3_DEVICE_GREEDY");
    require(!greedy || std::strcmp(greedy,"0")==0 || std::strcmp(greedy,"1")==0,"device greedy must be0 or1");
    const bool greedy_enabled=!greedy || std::strcmp(greedy,"1")==0;
    // Check the combined delayed groups before creating any of their storage.
    auto required_delayed_groups=Exl3LinearWorkspaceRequirements::append_owned_bytes(base_bytes,required_kv_bytes);
    required_delayed_groups=Exl3LinearWorkspaceRequirements::append_owned_bytes(required_delayed_groups,required_layer_group);
    if(tap_extent)required_delayed_groups=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        required_delayed_groups,static_cast<std::size_t>(tap_extent));
    if(shared_bytes)required_delayed_groups=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        required_delayed_groups,shared_bytes);
    if(numeric_requirement)required_delayed_groups=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        required_delayed_groups,numeric_requirement);
    if(reconstruction_requirement)required_delayed_groups=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        required_delayed_groups,reconstruction_requirement);
    if(exact_score_requirement)required_delayed_groups=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        required_delayed_groups,exact_score_requirement);
    if(numeric_attention_splitk_workspace_bytes)required_delayed_groups=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        required_delayed_groups,numeric_attention_splitk_workspace_bytes);
    const auto expected_delayed_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        impl->persistent_bytes,required_delayed_groups);
    const auto graph_z_bytes=impl->continuation_graph_gdn_qkvz_concurrent?
        Exl3LinearWorkspaceRequirements::derive(kHidden,6144,8).owned_bytes:0;
    const auto eager_mlp_gateup_bytes=impl->eager_mlp_gateup_concurrent?
        Exl3LinearWorkspaceRequirements::derive(
            kHidden,kIntermediate,8).owned_bytes:0;
    std::size_t prefill_qkv_bytes=0;
    if(impl->prefill_qkv_concurrent) {
        const auto kv_bytes=Exl3LinearWorkspaceRequirements::derive(
            kHidden,kKVProjection,impl->prefill_capacity).owned_bytes;
        prefill_qkv_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            kv_bytes,kv_bytes);
    }
    if(impl->continuation_graph_gdn_qkvz_concurrent) {
        unsigned gdn_count=0;
        for(int layer=0;layer<kLayers;++layer)if(!impl->model->layers[layer].full_attention)++gdn_count;
        require(gdn_count==48,"fixed-B8 GDN QKV/Z concurrency requires all 48 GDN layers");
    }
    constexpr std::size_t media_feature_bytes=8ULL*kHidden*sizeof(float);
    constexpr std::size_t media_position_bytes=8ULL*3*sizeof(std::int32_t);
    auto expected_with_optional=expected_delayed_persistent;
    if(graph_z_bytes)expected_with_optional=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        expected_with_optional,graph_z_bytes);
    if(eager_mlp_gateup_bytes)
        expected_with_optional=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            expected_with_optional,eager_mlp_gateup_bytes);
    if(prefill_qkv_bytes)
        expected_with_optional=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            expected_with_optional,prefill_qkv_bytes);
    if(media_enabled) {
        expected_with_optional=Exl3LinearWorkspaceRequirements::append_owned_bytes(expected_with_optional,media_feature_bytes);
        expected_with_optional=Exl3LinearWorkspaceRequirements::append_owned_bytes(expected_with_optional,media_position_bytes);
    }
    const auto prefix_requirement=cache_rows?Exl3DevicePrefixCache::allocation_bytes_required(cache_rows):0;
    const auto expected_with_prefix=prefix_requirement?Exl3LinearWorkspaceRequirements::append_owned_bytes(
        expected_with_optional,prefix_requirement):expected_with_optional;
    const auto greedy_requirement=greedy_enabled?Exl3TextContext::greedy_packet_bytes_required():0;
    const auto expected_startup_peak=greedy_requirement?Exl3LinearWorkspaceRequirements::append_owned_bytes(
        expected_with_prefix,greedy_requirement):expected_with_prefix;
    const std::size_t generic_owner_count=base_owners+shared_owners+
        (exact_score_requirement?1:0)+(numeric_attention_splitk_workspace_bytes?1:0);
    require(generic_owner_count<=impl->allocations.max_size()-impl->allocations.size(),
        "target complete generic owner capacity overflow");
    const auto expected_generic_owners=impl->allocations.size()+generic_owner_count;
    impl->allocations.reserve(expected_generic_owners);
    const auto reserved_generic_capacity=impl->allocations.capacity();
    Exl3ResourceInventory::Requirement generic_metadata_requirement;
    if(required_kv_owners)generic_metadata_requirement.add(Exl3ResourceInventory::Domain::host_metadata,
        required_kv_owners,DeviceAllocation::shared_control_bytes);
    if(cache_rows)generic_metadata_requirement.add(Exl3ResourceInventory::Domain::host_metadata,1,
        Exl3DevicePrefixCache::owner_metadata_bytes_required(cache_rows));
    // Fixed owner slots are already included in sizeof(Impl), credited before
    // planning. Only separately allocated DeviceAllocation records join here.
    generic_metadata_requirement.add(Exl3ResourceInventory::Domain::host_metadata,
        expected_generic_owners+required_kv_owners+(capture_taps?kTapLayers.size()+1:0)+
            (media_enabled?2:0)+(greedy_enabled?1:0),
        DeviceAllocation::owner_metadata_bytes());
    if(greedy_enabled)generic_metadata_requirement.add(
        Exl3ResourceInventory::Domain::host_metadata,1,
        Exl3TextContext::greedy_transfer_pool_metadata_bytes_required());
    generic_metadata_requirement.add(Exl3ResourceInventory::Domain::host_metadata,
        1+(impl->continuation_graph_gdn_qkvz_concurrent?1:0)+
            (impl->eager_mlp_gateup_concurrent?1:0)+
            (impl->prefill_qkv_concurrent?2:0),
        Exl3CudaLinearWorkspace::metadata_bytes());
    if(numeric_requirement)generic_metadata_requirement.add(
        Exl3ResourceInventory::Domain::host_metadata,1,sizeof(Exl3CudaReconstructGemmWorkspace));
    if(transfer_plan.pinned_owners)generic_metadata_requirement.add(
        Exl3ResourceInventory::Domain::host_metadata,transfer_plan.pinned_owners,sizeof(PinnedHostBuffer));
    for(int layer=0;layer<kLayers;++layer)
        generic_metadata_requirement.add(Exl3ResourceInventory::Domain::host_metadata,1,
            impl->model->layers[layer].full_attention?Exl3FullAttentionLayer::fixed_owner_metadata_required():
                Exl3GdnLayer::fixed_owner_metadata_required());
    Exl3ResourceInventory::Requirement context_requirement=transfer_plan.resources;
    context_requirement.add(Exl3ResourceInventory::Domain::device,1,expected_startup_peak);
    if(greedy_enabled)context_requirement.add(
        Exl3ResourceInventory::Domain::cuda_registered_host,1,
        Exl3TextContext::greedy_transfer_pool_registered_bytes_required());
    context_requirement.add(Exl3ResourceInventory::Domain::host_metadata,1,
        generic_metadata_requirement.units[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)]);
    context_requirement.add(Exl3ResourceInventory::Domain::host_metadata,1,
        Exl3TextContext::fixed_owner_metadata_bytes());
    // The synchronous construction closure is the ownership boundary for startup
    // admission. Planning and owner-vector preparation above allocate no device
    // buffers; the closure retains all partial construction in Impl on failure.
    const auto construct=[&]() -> std::unique_ptr<Exl3TextContext> {
    impl->constructor_authority=authority;
    materialize_base();
    materialize_kv();
    materialize_taps();
    materialize_shared();
    materialize_numeric();
    materialize_reconstruction();
    materialize_attention();
    require(impl->allocations.size()==expected_generic_owners &&
        impl->allocations.capacity()==reserved_generic_capacity,
        "target generic owner plan changed during device construction");
    const auto expected_layer_group=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        impl->persistent_bytes,required_layer_group);
    const auto expected_head_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(
        impl->persistent_bytes,head_requirement.owned_bytes);
    if(startup_fault==17)Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(true);
    impl->head_workspace=impl->make_base_linear(kHidden,kVocab,1);
    require(impl->head_workspace->workspace_bytes()==head_requirement.owned_bytes,
        "target head allocation requirement mismatch");
    impl->persistent_bytes=expected_head_persistent;
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl->model->layers[layer].full_attention) {
            const auto layer_requirement=layer_requirements[layer];
            const auto expected_layer_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(
                impl->persistent_bytes,layer_requirement);
            if(startup_fault==18) {
                require(!transformed.data,"attention constructor failure fixture requires owned transform");
                Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(true);
            }
            impl->full_layers[layer] = std::make_unique<Exl3FullAttentionLayer>(
                impl->model->layers[layer].full, impl->prefill_capacity,
                accumulation, transformed, scratch,
                impl->numeric_prefill_projection_workspace.get(),impl->coalesce_attention_input_mlp,authority,
                startup_fault==22?1:(startup_fault==25?3:0),paired_up_accumulation,
                paired_up_transformed);
            require(impl->full_layers[layer]->workspace_bytes()==layer_requirement,
                "target full-attention layer requirement mismatch");
            require(impl->full_layers[layer]->fixed_owner_metadata_bytes()==
                Exl3FullAttentionLayer::fixed_owner_metadata_required(),
                "target full-attention owner metadata requirement mismatch");
            impl->persistent_bytes=expected_layer_persistent;
            if(impl->coalesce_attention_input_mlp) {
                ++impl->host_kv.coalesced_attention_layers;
                if(!scratch.data)impl->host_kv.coalesced_attention_bytes_saved+=
                    static_cast<std::uint64_t>(impl->prefill_capacity)*kHidden*sizeof(std::uint16_t);
            }
            impl->full_layers[layer]->set_reconstructed_exact(reconstructed_exact_view);
            impl->full_layers[layer]->set_kv_cache(
                impl->oscar_only?nullptr:static_cast<std::uint16_t*>(impl->host_kv.enabled?impl->host_layer_k->ptr:impl->cache_k[layer]->ptr),
                impl->oscar_only?nullptr:static_cast<std::uint16_t*>(impl->host_kv.enabled?impl->host_layer_v->ptr:impl->cache_v[layer]->ptr), max_context_);
            impl->full_layers[layer]->set_position_device(impl->position_device);
            impl->full_layers[layer]->set_exact_attention_scores(exact_scores,16);
            impl->full_layers[layer]->set_exact_attention_q_shared(exact_q_shared_enabled);
            impl->full_layers[layer]->set_exact_attention_query_pair(query_pair_enabled);
            impl->full_layers[layer]->set_exact_attention_k_half2(exact_k_half2_enabled);
            impl->full_layers[layer]->set_exact_attention_v_half2(exact_v_half2_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_pair(exact_gqa_pair_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_triple(exact_gqa_triple_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_triple_values128(exact_gqa_triple_values128_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_triple_softmax_staged(exact_gqa_triple_softmax_staged_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_six(exact_gqa_six_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_six_scores(exact_gqa_six_scores_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_six_extent_shards(
                exact_gqa_six_extent_shards_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_six_values_sharded(
                exact_gqa_six_values_sharded_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_triple_values4(
                exact_gqa_triple_values4_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_six_softmax_triple_values(
                exact_gqa_six_softmax_triple_values_enabled);
            impl->full_layers[layer]->
                set_exact_attention_gqa_six_softmax_triple_values_pair_dimensions(
                    exact_gqa_six_softmax_triple_values_pair_dimensions_enabled);
            impl->full_layers[layer]->
                set_exact_attention_gqa_six_softmax_six_values_single_load(
                    exact_gqa_six_softmax_six_values_single_load_enabled);
            impl->full_layers[layer]->set_exact_attention_gqa_six_packed_triples(
                exact_gqa_six_packed_triples_enabled);
            impl->full_layers[layer]->set_numeric_attention_splitk(
                numeric_attention_splitk_workspace,
                numeric_attention_splitk_workspace_bytes,
                numeric_attention_splitk_enabled);
            if(impl->fast_wmma32_split_capacity)
                impl->full_layers[layer]->set_fast_wmma32_split2_workspace(
                    impl->fast_wmma32_split2_output,
                    impl->fast_wmma32_split2_stats,1024,
                    impl->fast_wmma32_split_count,impl->fast_wmma32_split_capacity);
        } else {
            const auto layer_requirement=layer_requirements[layer];
            const auto expected_layer_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(
                impl->persistent_bytes,layer_requirement);
            if(startup_fault==19) {
                require(!transformed.data,"GDN constructor failure fixture requires owned transform");
                Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(true);
            }
            require(impl->model->gdn_coefficients[layer].has_value(),
                    "target GDN immutable coefficient binding absent");
            auto coefficient_owner=
                std::shared_ptr<const Exl3GdnImmutableCoefficients>(
                    impl->model,&*impl->model->gdn_coefficients[layer]);
            impl->gdn_layers[layer] = make_bounded_shared<Exl3GdnLayer>(
                impl->model->layers[layer].gdn, impl->prefill_capacity, accumulation, transformed,
                scratch, gdn_wide_scratch,
                impl->numeric_prefill_projection_workspace.get(),authority,
                startup_fault==23?1:(startup_fault==24?2:(startup_fault==26?3:(startup_fault==27?4:0))),
                std::move(coefficient_owner),paired_up_accumulation,
                paired_up_transformed);
            impl->gdn_layers[layer]->bind_recurrent_layout(
                impl->model->host_state_identity,layer);
            require(impl->gdn_layers[layer]->workspace_bytes()==layer_requirement,
                "target GDN layer requirement mismatch");
            require(impl->gdn_layers[layer]->fixed_owner_metadata_bytes()==
                Exl3GdnLayer::fixed_owner_metadata_required(),
                "target GDN owner metadata requirement mismatch");
            impl->persistent_bytes=expected_layer_persistent;
            impl->gdn_layers[layer]->set_reconstructed_exact(reconstructed_exact_view);
        }
    }
    require(impl->persistent_bytes==expected_layer_group,"target complete layer group requirement mismatch");
    require(impl->persistent_bytes==expected_delayed_persistent,"target delayed construction requirement mismatch");
    if (impl->continuation_graph_gdn_qkvz_concurrent) {
        const auto expected_graph_z_persistent=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            impl->persistent_bytes,graph_z_bytes);
        impl->gdn_qkvz_z_workspace=impl->make_base_linear(kHidden,6144,8,true);
        impl->gdn_qkvz_private_workspace_bytes =
            impl->gdn_qkvz_z_workspace->workspace_bytes();
        require(impl->gdn_qkvz_private_workspace_bytes==graph_z_bytes,
            "target graph Z allocation requirement mismatch");
        impl->persistent_bytes=expected_graph_z_persistent;
        cuda_check(cudaStreamCreateWithFlags(&impl->gdn_qkvz_z_stream,
                                              cudaStreamNonBlocking),
                   "create fixed-B8 GDN Z projection stream");
        // Profiler-only metadata: this does not add graph capture/replay work.
        nvtxNameCudaStreamA(impl->gdn_qkvz_z_stream,
                            "voidinfer.fixed_b8.gdn_z_aux");
        for (int layer = 0; layer < kLayers; ++layer) {
            if (!impl->gdn_layers[layer]) continue;
            cuda_check(cudaEventCreateWithFlags(&impl->gdn_qkvz_fork[layer],
                                                cudaEventDisableTiming),
                       "create fixed-B8 GDN QKV/Z fork event");
            cuda_check(cudaEventCreateWithFlags(&impl->gdn_qkvz_z_done[layer],
                                                cudaEventDisableTiming),
                       "create fixed-B8 GDN Z completion event");
            impl->gdn_layers[layer]->prepare_continuation_graph_qkvz_concurrency({
                impl->gdn_qkvz_z_stream, impl->gdn_qkvz_fork[layer],
                impl->gdn_qkvz_z_done[layer], impl->gdn_qkvz_z_workspace.get()});
            ++impl->gdn_qkvz_layer_count;
        }
        require(impl->gdn_qkvz_layer_count == 48,
                "fixed-B8 GDN QKV/Z concurrency requires all 48 GDN layers");
    }
    if (impl->eager_mlp_gateup_concurrent) {
        const auto expected_gateup_persistent=
            Exl3LinearWorkspaceRequirements::append_owned_bytes(
                impl->persistent_bytes,eager_mlp_gateup_bytes);
        impl->eager_mlp_gateup_up_workspace=impl->make_base_linear(
            kHidden,kIntermediate,8,false,true);
        impl->eager_mlp_gateup_private_workspace_bytes=
            impl->eager_mlp_gateup_up_workspace->workspace_bytes();
        require(impl->eager_mlp_gateup_private_workspace_bytes==
                    eager_mlp_gateup_bytes,
            "eager MLP gate/up allocation requirement mismatch");
        impl->persistent_bytes=expected_gateup_persistent;
        cuda_check(cudaStreamCreateWithFlags(
                       &impl->eager_mlp_gateup_stream,cudaStreamNonBlocking),
                   "create eager MLP up projection stream");
        nvtxNameCudaStreamA(impl->eager_mlp_gateup_stream,
                            "voidinfer.eager.mlp_up_aux");
        cuda_check(cudaEventCreateWithFlags(
                       &impl->eager_mlp_gateup_fork,cudaEventDisableTiming),
                   "create eager MLP gate/up fork event");
        cuda_check(cudaEventCreateWithFlags(
                       &impl->eager_mlp_gateup_up_done,cudaEventDisableTiming),
                   "create eager MLP up completion event");
        const Exl3MlpGateUpConcurrencyView view{
            impl->eager_mlp_gateup_stream,impl->eager_mlp_gateup_fork,
            impl->eager_mlp_gateup_up_done,
            impl->eager_mlp_gateup_up_workspace.get()};
        for (int layer=0;layer<kLayers;++layer) {
            if (impl->full_layers[layer])
                impl->full_layers[layer]->prepare_eager_mlp_gateup_concurrency(
                    view);
            else if (impl->gdn_layers[layer])
                impl->gdn_layers[layer]->prepare_eager_mlp_gateup_concurrency(
                    view);
            else
                throw std::runtime_error(
                    "eager MLP gate/up layer missing");
            ++impl->eager_mlp_gateup_layer_count;
        }
        require(impl->eager_mlp_gateup_layer_count==kLayers,
                "eager MLP gate/up concurrency requires all 64 layers");
    }
    if(impl->prefill_qkv_concurrent) {
        const auto expected_qkv_persistent=
            Exl3LinearWorkspaceRequirements::append_owned_bytes(
                impl->persistent_bytes,prefill_qkv_bytes);
        impl->prefill_qkv_k_workspace=impl->make_base_linear(
            kHidden,kKVProjection,impl->prefill_capacity,false,false,true,false,true);
        impl->prefill_qkv_v_workspace=impl->make_base_linear(
            kHidden,kKVProjection,impl->prefill_capacity,false,false,true,false,true);
        impl->prefill_qkv_private_workspace_bytes=
            Exl3LinearWorkspaceRequirements::append_owned_bytes(
                impl->prefill_qkv_k_workspace->workspace_bytes(),
                impl->prefill_qkv_v_workspace->workspace_bytes());
        require(impl->prefill_qkv_private_workspace_bytes==prefill_qkv_bytes,
            "wide-prefill QKV allocation requirement mismatch");
        impl->persistent_bytes=expected_qkv_persistent;
        cuda_check(cudaStreamCreateWithFlags(&impl->prefill_qkv_k_stream,
            cudaStreamNonBlocking),"create wide-prefill K projection stream");
        cuda_check(cudaStreamCreateWithFlags(&impl->prefill_qkv_v_stream,
            cudaStreamNonBlocking),"create wide-prefill V projection stream");
        nvtxNameCudaStreamA(impl->prefill_qkv_k_stream,"voidinfer.prefill.k_aux");
        nvtxNameCudaStreamA(impl->prefill_qkv_v_stream,"voidinfer.prefill.v_aux");
        cuda_check(cudaEventCreateWithFlags(&impl->prefill_qkv_fork,cudaEventDisableTiming),
            "create wide-prefill QKV fork event");
        cuda_check(cudaEventCreateWithFlags(&impl->prefill_qkv_k_done,cudaEventDisableTiming),
            "create wide-prefill K completion event");
        cuda_check(cudaEventCreateWithFlags(&impl->prefill_qkv_v_done,cudaEventDisableTiming),
            "create wide-prefill V completion event");
        const Exl3PrefillQkvConcurrencyView view{
            impl->prefill_qkv_k_stream,impl->prefill_qkv_v_stream,
            impl->prefill_qkv_fork,impl->prefill_qkv_k_done,
            impl->prefill_qkv_v_done,impl->prefill_qkv_k_workspace.get(),
            impl->prefill_qkv_v_workspace.get()};
        for(int layer=0;layer<kLayers;++layer)if(impl->full_layers[layer]) {
            impl->full_layers[layer]->prepare_prefill_qkv_concurrency(view);
            ++impl->prefill_qkv_layer_count;
        }
        require(impl->prefill_qkv_layer_count==16,
            "wide-prefill QKV concurrency requires all 16 attention layers");
    }
    if(media_enabled){
        impl->media_features=impl->make_private_allocation(media_feature_bytes,"allocate bounded media FP32 input");
        impl->media_positions=impl->make_private_allocation(media_position_bytes,"allocate bounded media positions");
        impl->persistent_bytes+=impl->media_features->bytes+impl->media_positions->bytes;
    }
    require(impl->persistent_bytes==expected_with_optional,"target optional workspace requirement mismatch");
    if(cache_rows){
        impl->device_prefix=Exl3DevicePrefixCache::create(cache_rows);
        if(impl->device_prefix->admitted()){
            require(impl->device_prefix->allocation_bytes()==prefix_requirement,"device prefix allocation requirement mismatch");
            impl->persistent_bytes=expected_with_prefix;
            impl->host_kv.device_prefix_bytes=impl->device_prefix->allocation_bytes();
        }else ++impl->host_kv.device_prefix_fallbacks;
    }
    materialize_transfer();
    if(startup_fault==6 || startup_fault==7) {
        require(impl->host_kv_copy_stream,"context unwind fault requires HostKV transfer stream");
        impl->host_kv_retirement_failure_for_test=startup_fault==7;
        throw std::runtime_error("injected context post-transfer construction failure");
    }
    auto result = std::unique_ptr<Exl3TextContext>(new Exl3TextContext(std::move(impl)));
    struct WrappedStartupRetirement {
        std::unique_ptr<Exl3TextContext>& owner;
        Exl3VeriCacheServingCoordinator* authority;
        int exceptions=std::uncaught_exceptions();
        ~WrappedStartupRetirement() noexcept {
            if(std::uncaught_exceptions()<=exceptions || !owner)return;
            const auto before=Exl3TextContext::retirement_quarantine_witness();
            owner.reset();
            if(authority && Exl3TextContext::retirement_quarantine_witness()!=before)
                authority->seal_failed_startup_retirement();
        }
    } wrapped_retirement{result,authority};
    result->logits_ = result->impl_->logits;
    result->persistent_bytes_ = result->impl_->persistent_bytes;
    if(greedy_enabled) result->prepare_greedy_packet();
    require(result->allocation_owner_metadata_bytes()==generic_metadata_requirement.units[
        static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)],
        "target generic owner metadata requirement mismatch");
    const auto unused_prefix=prefix_requirement && (!result->impl_->device_prefix ||
        !result->impl_->device_prefix->admitted())?prefix_requirement:0;
    require(result->persistent_bytes_==context_requirement.units[
        static_cast<unsigned>(Exl3ResourceInventory::Domain::device)]-unused_prefix,
        "target complete startup device requirement mismatch");
    result->impl_->capture_host_kv_gdn_segment_graphs();
    result->impl_->capture_host_kv_full_layer_graphs();
    result->impl_->capture_ordinary_full_layer_graphs();
    result->impl_->capture_host_kv_mlp_tail_graphs();
    result->reset();
    if(startup_fault==8 || startup_fault==9) {
        require(result->impl_->host_kv_forward_used,"wrapped context unwind fault requires HostKV reset work");
        if(startup_fault==9)result->fail_host_kv_compute_retirement_for_test();
        throw std::runtime_error("injected context post-reset construction failure");
    }
    return result;
    };
    if(!authority)return {construct(),{}};
    context_requirement.configuration=0x4354585354415254;
    const auto startup_quarantine_before=Exl3TextContext::retirement_quarantine_witness();
    std::shared_ptr<Exl3TextContext> prepared;
    const auto retire_prepared=[&]() noexcept {
        const auto before=Exl3TextContext::retirement_quarantine_witness();
        prepared.reset();
        if(Exl3TextContext::retirement_quarantine_witness()!=before)
            authority->seal_failed_startup_retirement();
    };
    const auto allocate=[&](std::uint64_t configuration) {
        require(configuration==context_requirement.configuration,"context startup reservation identity");
        if(startup_fault==1)throw std::runtime_error("injected context startup preconstruction failure");
        prepared=std::shared_ptr<Exl3TextContext>(construct());
        try {
        if(startup_fault==10)prepared->fail_host_kv_compute_retirement_for_test();
        if(startup_fault==16) {
            require(bool(prepared->impl_->embedding_trace),"private constructor cleanup fault requires embedding trace");
            prepared->impl_->embedding_trace->cleanup_failure_for_test=true;
        }
        if(startup_fault>=11 && startup_fault<=13) {
            require(prepared->impl_->host_layer_k!=nullptr,"generic cleanup fault requires HostKV K workspace");
            prepared->impl_->host_layer_k->cleanup_failure_for_test=startup_fault==11;
            prepared->impl_->host_layer_k->device_query_failure_for_test=startup_fault==12;
            prepared->impl_->host_layer_k->device_mismatch_for_test=startup_fault==13;
        }
        if(startup_fault==2)throw std::runtime_error("injected context startup precommit failure");
        if(startup_fault==20 || startup_fault==21) {
            const bool selected=startup_fault==20?prepared->fail_attention_buffer_retirement_for_test(0):
                prepared->fail_gdn_buffer_retirement_for_test(0);
            require(selected,"startup layer buffer failure owner missing");
        }
        Exl3ResourceInventory actual;
        using Domain=Exl3ResourceInventory::Domain;
        const std::array<Exl3CudaLinearWorkspace*,4> linear_children{
            prepared->impl_->head_workspace.get(),
            prepared->impl_->gdn_qkvz_z_workspace
                ? prepared->impl_->gdn_qkvz_z_workspace.get()
                : prepared->impl_->eager_mlp_gateup_up_workspace.get(),
            prepared->impl_->prefill_qkv_k_workspace.get(),
            prepared->impl_->prefill_qkv_v_workspace.get()};
        std::uint64_t linear_bytes=0,linear_metadata=0;
        const auto add_linear=[&](Exl3CudaLinearWorkspace* workspace,std::uint64_t slot) {
            if(!workspace)return;
            const auto workspace_bytes=workspace->workspace_bytes();
            // A fully borrowed child still owns metadata, but has no device
            // allocation extent to contribute to the physical inventory.
            if(workspace_bytes)
                linear_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(linear_bytes,workspace_bytes);
            linear_metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(linear_metadata,Exl3CudaLinearWorkspace::metadata_bytes());
            std::shared_ptr<const void> owner(prepared,workspace);
            if(workspace_bytes)actual.add({owner,slot,Domain::device,workspace_bytes,{},nullptr,
                &Exl3CudaLinearWorkspace::attach_owner_device_credit});
            actual.add({owner,slot+1,Domain::host_metadata,Exl3CudaLinearWorkspace::metadata_bytes(),{},
                &Exl3CudaLinearWorkspace::attach_owner_metadata_credit});
        };
        for(unsigned i=0;i<linear_children.size();++i)add_linear(linear_children[i],3+2*i);
        // Fixed disjoint ranges: base slots0..10, then 26 slots per layer
        // (seven attention children followed by six GDN children).
        for(unsigned layer=0;layer<kLayers;++layer) {
            const std::uint64_t first=11+26*layer;
            if(const auto& owner=prepared->impl_->full_layers[layer]) {
                const auto children=owner->linear_workspace_owners();
                for(unsigned i=0;i<children.size();++i)add_linear(children[i],first+2*i);
            }
            if(const auto& owner=prepared->impl_->gdn_layers[layer]) {
                const auto children=owner->linear_workspace_owners();
                for(unsigned i=0;i<children.size();++i)add_linear(children[i],first+14+2*i);
            }
        }
        std::uint64_t generic_bytes=0,generic_metadata=0,generic_slot=11+26*kLayers;
        for(const auto& layer:prepared->impl_->full_layers)if(layer)
            for(auto* child:layer->buffer_retirement_owners())if(child) {
                generic_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_bytes,child->bytes());
                generic_metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_metadata,Exl3LayerBufferRetirement::record_bytes());
                std::shared_ptr<const void> owner(prepared,child);
                actual.add({owner,generic_slot++,Domain::device,child->bytes(),{},nullptr,&Exl3LayerBufferRetirement::attach_device_credit});
                actual.add({owner,generic_slot++,Domain::host_metadata,Exl3LayerBufferRetirement::record_bytes(),{},&Exl3LayerBufferRetirement::attach_metadata_credit});
            }
        for(const auto& layer:prepared->impl_->gdn_layers)if(layer)
            for(auto* child:layer->buffer_retirement_owners())if(child) {
                generic_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_bytes,child->bytes());
                generic_metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_metadata,Exl3LayerBufferRetirement::record_bytes());
                std::shared_ptr<const void> owner(prepared,child);
                actual.add({owner,generic_slot++,Domain::device,child->bytes(),{},nullptr,&Exl3LayerBufferRetirement::attach_device_credit});
                actual.add({owner,generic_slot++,Domain::host_metadata,Exl3LayerBufferRetirement::record_bytes(),{},&Exl3LayerBufferRetirement::attach_metadata_credit});
            }
        const auto add_generic=[&](DeviceAllocation* child) {
            if(!child)return;
            generic_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_bytes,child->bytes);
            generic_metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_metadata,DeviceAllocation::owner_metadata_bytes());
            std::shared_ptr<const void> owner(prepared,child);
            actual.add({owner,generic_slot++,Domain::device,child->bytes,{},nullptr,&DeviceAllocation::attach_device_credit});
            actual.add({owner,generic_slot++,Domain::host_metadata,DeviceAllocation::owner_metadata_bytes(),{},&DeviceAllocation::attach_metadata_credit});
        };
        prepared->impl_->allocations.visit_owners(add_generic);
        for(const auto& child:prepared->impl_->taps)add_generic(child.get());
        add_generic(prepared->impl_->embedding_trace.get());
        add_generic(prepared->impl_->media_features.get());
        add_generic(prepared->impl_->media_positions.get());
        add_generic(prepared->impl_->greedy_rows.get());
        for(const auto& transfer:prepared->impl_->greedy_transfers)if(transfer) {
            const auto bytes=Exl3GreedyPacketTransfer::row_bytes;
            generic_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_bytes,bytes);
            generic_metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(
                generic_metadata,Exl3GreedyPacketTransfer::metadata_bytes_required());
            actual.add({transfer,generic_slot++,Domain::device,bytes,{},nullptr,
                &Exl3GreedyPacketTransfer::attach_device_credit});
            actual.add({transfer,generic_slot++,Domain::cuda_registered_host,bytes,{},nullptr,nullptr,
                &Exl3GreedyPacketTransfer::attach_registration_credit});
            actual.add({transfer,generic_slot++,Domain::host_metadata,
                Exl3GreedyPacketTransfer::metadata_bytes_required(),{},
                &Exl3GreedyPacketTransfer::attach_metadata_credit});
        }
        // Shared KV borrowers retain the allocation's own control block, not
        // the context wrapper. Allocator metadata uses its own final-weak-owner
        // ticket in the fixed-capacity control block.
        std::array<const DeviceAllocation*,2+2*kLayers> shared_seen{};
        std::size_t shared_count=0;
        const auto add_shared_generic=[&](const std::shared_ptr<DeviceAllocation>& child) {
            if(!child)return;
            for(std::size_t i=0;i<shared_count;++i)
                require(shared_seen[i]!=child.get(),"context shared KV allocation appears in multiple physical owner slots");
            shared_seen[shared_count++]=child.get();
            generic_bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_bytes,child->bytes);
            generic_metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_metadata,DeviceAllocation::owner_metadata_bytes());
            actual.add({child,0,Domain::device,child->bytes,{},nullptr,&DeviceAllocation::attach_device_credit});
            actual.add({child,1,Domain::host_metadata,DeviceAllocation::owner_metadata_bytes(),{},&DeviceAllocation::attach_metadata_credit});
            generic_metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(generic_metadata,DeviceAllocation::shared_control_bytes);
            actual.add({child,2,Domain::host_metadata,DeviceAllocation::shared_control_bytes,{},&DeviceAllocation::attach_control_credit});
        };
        add_shared_generic(prepared->impl_->host_layer_k);
        add_shared_generic(prepared->impl_->host_layer_v);
        for(const auto& child:prepared->impl_->cache_k)add_shared_generic(child);
        for(const auto& child:prepared->impl_->cache_v)add_shared_generic(child);
        const auto device_short=(startup_fault==3 || startup_fault==16 ||
            startup_fault==20 || startup_fault==21 ||
            (startup_fault>=10 && startup_fault<=13))?1ULL:0ULL;
        const auto persistent_bytes=prepared->persistent_bytes();
        require(persistent_bytes>=linear_bytes+generic_bytes+device_short,
            "context device inventory child extents exceed persistent allocation");
        const auto direct_device_bytes=persistent_bytes-linear_bytes-generic_bytes-device_short;
        if(direct_device_bytes)
            actual.add({prepared,0,Domain::device,direct_device_bytes});
        const auto host=prepared->host_kv_stats();
        if(host.pinned_staging_bytes)actual.add({prepared,1,Domain::cuda_registered_host,host.pinned_staging_bytes});
        auto metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            0,prepared->allocation_owner_metadata_bytes());
        if(host.batch_metadata_bytes)metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(metadata,host.batch_metadata_bytes);
        metadata=Exl3LinearWorkspaceRequirements::append_owned_bytes(metadata,Exl3TextContext::fixed_owner_metadata_bytes());
        const auto metadata_short=startup_fault==4?1ULL:0ULL;
        require(metadata>=linear_metadata+generic_metadata+metadata_short,
            "context metadata inventory child extents exceed owner allocation");
        const auto direct_metadata_bytes=metadata-linear_metadata-generic_metadata-metadata_short;
        if(direct_metadata_bytes)
            actual.add({prepared,2,Domain::host_metadata,direct_metadata_bytes});
        return actual;
        } catch(...) {
            // Release callback-local inventory before returning the exception to
            // reservation rollback. Preserve failed-drain admission fencing.
            retire_prepared();
            throw;
        }
    };
    if(extend) {
        require(actual_result!=nullptr,"context planning reservation missing inventory destination");
        extend(context_requirement);
        *actual_result=allocate(context_requirement.configuration);
    } else authority->allocate_startup_resources(context_requirement,allocate,retire_prepared,[&]() noexcept {
        if(Exl3TextContext::retirement_quarantine_witness()!=startup_quarantine_before)
            authority->seal_failed_startup_retirement();
    });
    if(!extend)prepared->impl_->finish_private_constructor_credits();
    return {{},std::move(prepared)};
}

Exl3TextContext::Exl3TextContext(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
std::size_t Exl3TextContext::fixed_owner_metadata_bytes() noexcept {
    return sizeof(Exl3TextContext)+sizeof(Impl);
}
std::size_t Exl3TextContext::allocation_record_metadata_bytes() noexcept {
    return DeviceAllocation::owner_metadata_bytes();
}
void Exl3TextContext::reserve_reconstruction_lanes(Exl3VeriCacheServingCoordinator& authority,
    std::span<Exl3TextContext*> contexts,bool allow_budget_fallback) {
    require(ReconstructionBacking::quarantined_allocations.load(std::memory_order_acquire)==0,
        "unresolved reconstruction allocation cleanup; reservation refused");
    require(!contexts.empty() && contexts.size()<=2,"reconstruction reservation lane count");
    using Inventory=Exl3ResourceInventory;
    Inventory::Requirement required;required.configuration=0x524543534c;
    std::array<std::size_t,2> attached_persistent{};
    std::array<std::size_t,2> attached_public_persistent{};
    bool needed=false;
    for(std::size_t i=0;i<contexts.size();++i) {
        auto* context=contexts[i];
        require(context && context->position_==0,"reconstruction reservation requires pristine context");
        for(std::size_t j=0;j<i;++j)require(contexts[j]!=context,"duplicate reconstruction lane");
        const auto& impl=*context->impl_;
        require(!context->transaction_active() && !impl.graph_active && !impl.graph_capture_active,
            "reconstruction reservation requires inactive transaction and graph");
        require(!impl.transaction || !impl.transaction->rollback_required,
            "reconstruction reservation requires completed transaction rollback");
        require(!impl.graph_definition.ready() && !impl.graph_executable.ready(),
            "reconstruction reservation excludes retained target graph");
        if(impl.continuation) {
            require(!impl.continuation->graph_active &&
                !impl.continuation->graph_definition.ready() && !impl.continuation->graph_executable.ready(),
                "reconstruction reservation excludes retained continuation graph");
            for(const auto& graph:impl.continuation->additional_graphs)
                require(!graph.active && !graph.definition.ready() && !graph.executable.ready(),
                    "reconstruction reservation excludes retained additional graph");
        }
        require(!impl.reconstruction_backing,"reconstruction backing already installed");
        if(!impl.deferred_reconstruction_bytes)continue;
        attached_persistent[i]=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            impl.persistent_bytes,impl.deferred_reconstruction_bytes);
        attached_public_persistent[i]=Exl3LinearWorkspaceRequirements::append_owned_bytes(
            context->persistent_bytes_,impl.deferred_reconstruction_bytes);
        needed=true;
        required.add(Inventory::Domain::device,1,impl.deferred_reconstruction_bytes);
        required.add(Inventory::Domain::host_metadata,1,sizeof(ReconstructionBacking)+ReconstructionBacking::control_bytes+sizeof(DeviceAllocation::Retained));
    }
    if(!needed)return;
    std::array<std::shared_ptr<ReconstructionBacking>,2> pending;
    const auto quarantine_before=ReconstructionBacking::quarantined_allocations.load(std::memory_order_acquire);
    bool factory_entered=false;
    try {
    authority.allocate_startup_resources(required,[&](std::uint64_t configuration) {
        factory_entered=true;
        require(configuration==required.configuration,"reconstruction reservation configuration");
        Inventory actual;
        for(std::size_t i=0;i<contexts.size();++i) {
            const auto bytes=contexts[i]->impl_->deferred_reconstruction_bytes;
            if(!bytes)continue;
            if(std::exchange(contexts[i]->impl_->reconstruction_allocation_failure_for_test,false)) {
                if(std::exchange(contexts[i]->impl_->reconstruction_rollback_cleanup_failure_for_test,false))
                    for(auto& owner:pending)if(owner)owner->cleanup_failure_for_test=true;
                throw std::runtime_error("injected reconstruction backing allocation failure");
            }
            pending[i]=ReconstructionBacking::create(bytes);
            pending[i]->stream.bind_model(contexts[i]->impl_->model,bytes,
                contexts[i]->impl_->deferred_reconstruction_k6,
                pending[i]->allocation.ptr,
                contexts[i]->impl_->deferred_reconstruction_k6_gate_up);
            actual.add({pending[i],0,Inventory::Domain::device,bytes});
            actual.add({pending[i],1,Inventory::Domain::host_metadata,sizeof(ReconstructionBacking)+ReconstructionBacking::control_bytes+sizeof(DeviceAllocation::Retained)});
        }
        return actual;
    },[&]() noexcept {
        for(auto& owner:pending)owner.reset();
        if(ReconstructionBacking::quarantined_allocations.load(std::memory_order_acquire)!=quarantine_before)
            authority.seal_failed_startup_retirement();
    });
    } catch(const Exl3ResourceReservationExhausted&) {
        if(!allow_budget_fallback || factory_entered)throw;
        for(auto* context:contexts)if(context->impl_->deferred_reconstruction_bytes) {
            context->impl_->deferred_reconstruction_bytes=0;
            context->impl_->reconstruction_budget_fallback=true;
            context->impl_->reconstruction_allocation_failure_for_test=false;
            context->impl_->reconstruction_rollback_cleanup_failure_for_test=false;
        }
        return; // Existing direct projections remain installed; no slab was allocated.
    }
    // No context sees a partial multi-lane transaction. Setters only copy views.
    for(std::size_t i=0;i<contexts.size();++i)if(pending[i]) {
        auto& impl=*contexts[i]->impl_;
        impl.reconstruction_backing=std::move(pending[i]);
        auto& backing=*impl.reconstruction_backing;
        Exl3ReconstructedExactView view;
        view.data=static_cast<std::uint16_t*>(backing.allocation.ptr);
        view.bytes=backing.allocation.bytes;view.stats=&backing.stats;
        view.allow_k6_down=impl.deferred_reconstruction_k6;
        view.allow_k6_gate_up=impl.deferred_reconstruction_k6_gate_up;
        view.ordered_stream=&backing.stream;view.backing_owner=impl.reconstruction_backing;
        view.model_owner=impl.model;
        for(auto& layer:impl.full_layers)if(layer)layer->set_reconstructed_exact(view);
        for(auto& layer:impl.gdn_layers)if(layer)layer->set_reconstructed_exact(view);
        impl.persistent_bytes=attached_persistent[i];impl.deferred_reconstruction_bytes=0;
        contexts[i]->persistent_bytes_=attached_public_persistent[i];
    }
}
Exl3TextContext::~Exl3TextContext() {
    if(Impl::retain_failed_host_kv_drain(impl_))return;
    if(!impl_ || !impl_->reconstruction_backing || !impl_->reconstruction_backing->stream.used())return;
    // Keep the entire execution bundle alive until all consumers are retired:
    // retaining just decoded weights would release output/partials/model owners.
    // A narrower event-based drain is a separate implementation obligation.
    const auto retired=exl3_retire_reconstruction_device(impl_->reconstruction_backing->device,
        [](int* device) noexcept {return static_cast<int>(cudaGetDevice(device));},
        [](int device) noexcept {return static_cast<int>(cudaSetDevice(device));},
        []() noexcept {return static_cast<int>(cudaDeviceSynchronize());},
        [&]() noexcept {impl_.reset();},impl_->reconstruction_retirement_failure_for_test);
    if(retired.released)return;
    static std::atomic<Impl*> quarantine{nullptr};
    auto* retained=impl_.release();
    retained->reconstruction_backing->stream.fail();
    auto* head=quarantine.load(std::memory_order_relaxed);
    do {retained->reconstruction_quarantine_next=head;}
    while(!quarantine.compare_exchange_weak(head,retained,std::memory_order_release,std::memory_order_relaxed));
    ::ninfer::exl3::reconstruction_quarantined_contexts.fetch_add(1,std::memory_order_release);
}
std::size_t Exl3TextContext::reconstruction_quarantined_contexts() noexcept {
    return ::ninfer::exl3::reconstruction_quarantined_contexts.load(std::memory_order_acquire);
}
std::size_t Exl3TextContext::host_kv_quarantined_contexts() noexcept {
    return hostkv_quarantined_contexts.load(std::memory_order_acquire);
}
std::uint64_t Exl3TextContext::generic_quarantined_allocations() noexcept {
    return DeviceAllocation::quarantined_count.load(std::memory_order_acquire);
}
std::uint64_t Exl3TextContext::live_shared_control_metadata_bytes() noexcept {
    return Exl3SharedControlAccounting::live_bytes.load(std::memory_order_relaxed);
}
std::array<std::uint64_t,8> Exl3TextContext::retirement_quarantine_witness() noexcept {
    return {host_kv_quarantined_contexts(),reconstruction_quarantined_contexts(),
        ReconstructionBacking::quarantined_allocations.load(std::memory_order_acquire),
        Exl3DevicePrefixCache::budget_snapshot()[1],DeviceAllocation::quarantined_count.load(std::memory_order_acquire),
        Exl3CudaLinearWorkspace::quarantined_workspaces(),Exl3LayerBufferRetirement::quarantined(),
        Exl3RecurrentSlab::quarantine_count.load(std::memory_order_acquire)};
}
void Exl3TextContext::exercise_exact_page_extension_for_test() {
    const auto make_prefix=[] {
        auto tail=make_bounded_shared<Exl3ExactKVPage>();tail->rows=1;
        for(int bank=0;bank<16;++bank) {
            tail->k[bank].reserve(64*1024);tail->v[bank].reserve(64*1024);
            tail->k[bank].assign(1024,0x1234);tail->v[bank].assign(1024,0x4321);
        }
        std::vector<std::shared_ptr<const Exl3ExactKVPage>> prefix;prefix.push_back(std::move(tail));return prefix;
    };
    for(unsigned fault:{1U,2U}) {
        auto prefix=make_prefix();const auto* original=prefix.front().get();
        const auto before=*original;
        std::array<const std::uint16_t*,32> addresses{};
        for(int bank=0;bank<16;++bank){addresses[2*bank]=original->k[bank].data();addresses[2*bank+1]=original->v[bank].data();}
        bool refused=false;try{extend_exact_pages(prefix,1,129,true,nullptr,fault);}
        catch(const std::bad_alloc&){refused=true;}
        require(refused && prefix.front().get()==original && original->rows==1 && original->first==0 &&
            original->k==before.k && original->v==before.v,"failed page extension changed original tail");
        for(int bank=0;bank<16;++bank)require(addresses[2*bank]==original->k[bank].data() &&
            addresses[2*bank+1]==original->v[bank].data(),"failed page extension relocated original tail");
        Exl3HostKVStats stats{};auto next=extend_exact_pages(prefix,1,129,true,&stats);
        require(next.all.size()==3 && next.fresh.size()==3 && next.all.front().get()==original &&
            original->rows==64 && stats.page_unique_tail_reuses==1,"page extension retry lost private tail reuse");
    }
    {
        auto unique=make_prefix();const auto* original=unique.front().get();
        unsigned reservations=0;
        const auto refuse=[&](std::uint64_t)->RetainedDescriptorLedger::Ticket {
            ++reservations;throw Exl3ResourceReservationExhausted{};
        };
        bool refused=false;
        try{extend_exact_pages(unique,1,129,true,nullptr,0,refuse);}
        catch(const Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && reservations==1 && unique.front().get()==original && original->rows==1 &&
            original->k[0][0]==0x1234 && original->v[15][0]==0x4321,
            "new-page reservation failure mutated deferred unique tail");
        reservations=0;
        auto reused=extend_exact_pages(unique,1,2,true,nullptr,0,refuse);
        require(reservations==0 && reused.all.front().get()==original && original->rows==2,
            "unique-tail reuse requested new metadata credit or lost identity");
    }
    auto prefix=make_prefix();auto held=prefix.front();
    {
        RetainedDescriptorLedger metadata;unsigned reservations=0;
        bool refused=false;
        try {extend_exact_pages(prefix,1,129,true,nullptr,0,[&](std::uint64_t bytes) {
            if(++reservations==2)throw Exl3ResourceReservationExhausted{};
            return metadata.acquire(bytes);
        });}catch(const Exl3ResourceReservationExhausted&){refused=true;}
        require(refused && reservations==2 && metadata.bytes()==0 && held->rows==1,
            "page reservation failure retained earlier page credit or changed prefix");
    }
    auto next=extend_exact_pages(prefix,1,2,true);
    require(next.all.front()!=held && held->rows==1 && next.all.front()->rows==2,
        "page extension reused externally retained tail");
}

Exl3TextContext::GenericRetirementSnapshot Exl3TextContext::generic_retirement_snapshot_for_test() noexcept {
    const auto* retained=DeviceAllocation::quarantine.load(std::memory_order_acquire);
    return retained?GenericRetirementSnapshot{reinterpret_cast<std::uintptr_t>(retained->pointer),
        retained->bytes,retained->device,retained->error,sizeof(DeviceAllocation::Retained),
        retained->metadata_credit?retained->metadata_credit->bytes():0,
        retained->device_credit?retained->device_credit->bytes():0}:
        GenericRetirementSnapshot{};
}
void Exl3TextContext::exercise_generic_upload_for_test(std::span<const std::byte> source,unsigned fault) {
    require(!source.empty(),"generic upload fixture requires payload");
    DeviceAllocation allocation(source,"allocate generic upload fixture",fault);
}
void Exl3TextContext::exercise_generic_lifetime_credits_for_test() {
    RetainedDeviceLedger device;
    RetainedDescriptorLedger metadata,control;
    constexpr std::uint64_t bytes=128;
    const auto metadata_bytes=DeviceAllocation::owner_metadata_bytes();
    const auto control_bytes=DeviceAllocation::shared_control_bytes;
    const auto quarantine_before=retirement_quarantine_witness();
    const auto controls_before=live_shared_control_metadata_bytes();
    auto owner=DeviceAllocation::create_shared(bytes,"generic lifetime credit fixture");
    require(!DeviceAllocation::attach_device_credit({},device.acquire(bytes)) && device.bytes()==0,
        "generic null device attachment retained rejected ticket");
    require(!DeviceAllocation::attach_metadata_credit({},metadata.acquire(metadata_bytes)) && metadata.bytes()==0,
        "generic null metadata attachment retained rejected ticket");
    require(!DeviceAllocation::attach_control_credit({},control.acquire(control_bytes)) && control.bytes()==0,
        "generic null control attachment retained rejected ticket");
    for(const auto extent:{std::uint64_t(0),bytes-1,bytes+1})
        require(!DeviceAllocation::attach_device_credit(owner,device.acquire(extent)) && device.bytes()==0,
            "generic device extent rejection mutated charge");
    for(const auto extent:{std::size_t(0),metadata_bytes-1,metadata_bytes+1})
        require(!DeviceAllocation::attach_metadata_credit(owner,metadata.acquire(extent)) && metadata.bytes()==0,
            "generic metadata extent rejection mutated charge");
    for(const auto extent:{std::size_t(0),control_bytes-1,control_bytes+1})
        require(!DeviceAllocation::attach_control_credit(owner,control.acquire(extent)) && control.bytes()==0,
            "generic control extent rejection mutated charge");
    require(DeviceAllocation::attach_device_credit(owner,device.acquire(bytes)) &&
        DeviceAllocation::attach_metadata_credit(owner,metadata.acquire(metadata_bytes)) &&
        DeviceAllocation::attach_control_credit(owner,control.acquire(control_bytes)),"generic exact credit attachment refused");
    require(!DeviceAllocation::attach_device_credit(owner,device.acquire(bytes)) && device.bytes()==bytes,
        "generic duplicate device attachment changed prior ticket");
    require(!DeviceAllocation::attach_metadata_credit(owner,metadata.acquire(metadata_bytes)) && metadata.bytes()==metadata_bytes,
        "generic duplicate metadata attachment changed prior ticket");
    require(!DeviceAllocation::attach_control_credit(owner,control.acquire(control_bytes)) && control.bytes()==control_bytes,
        "generic duplicate control attachment changed prior ticket");
    std::weak_ptr<DeviceAllocation> weak=owner;owner.reset();
    require(weak.expired() && device.bytes()==0 && metadata.bytes()==0 && control.bytes()==control_bytes,
        "generic final strong release crossed allocation/control lifetime boundary");
    weak.reset();
    require(control.bytes()==0 && live_shared_control_metadata_bytes()==controls_before &&
        retirement_quarantine_witness()==quarantine_before,"generic credit fixture failed complete normal release");
}
void Exl3TextContext::exercise_generic_constructor_credits_for_test(unsigned fault) {
    require(fault==1 || fault==2,"generic credited constructor fault1..2");
    RetainedDeviceLedger device;RetainedDescriptorLedger metadata;
    const std::array<std::byte,128> payload{};
    const auto before=generic_quarantined_allocations();bool original=false;
    try {
        DeviceAllocation allocation(payload,"credited constructor fixture",fault,
            device.acquire(payload.size()),metadata.acquire(DeviceAllocation::owner_metadata_bytes()));
    }catch(const std::runtime_error& error) {
        original=std::string(error.what())=="injected generic post-upload construction failure";
    }
    require(original,"credited constructor changed original upload failure");
    if(fault==1)require(device.bytes()==0 && metadata.bytes()==0 && generic_quarantined_allocations()==before,
        "successful constructor unwind retained lifetime credit");
    else {
        const auto retained=generic_retirement_snapshot_for_test();
        require(generic_quarantined_allocations()==before+1 && device.bytes()==payload.size() &&
            metadata.bytes()==sizeof(DeviceAllocation::Retained) && retained.device_credit_bytes==device.bytes() &&
            retained.metadata_credit_bytes==metadata.bytes(),"failed constructor unwind lost exact retained credits");
    }
}
void Exl3TextContext::exercise_shared_constructor_credits_for_test(unsigned fault) {
    require(fault==1 || fault==2,"shared credited constructor fault1..2");
    RetainedDeviceLedger device;RetainedDescriptorLedger metadata;
    const auto before=generic_quarantined_allocations(),controls=live_shared_control_metadata_bytes();
    bool refused=false;
    try {
        (void)DeviceAllocation::create_shared(128,"shared credited constructor fixture",fault,
            device.acquire(128),metadata.acquire(DeviceAllocation::owner_metadata_bytes()+DeviceAllocation::shared_control_bytes));
    }catch(const std::bad_alloc&){refused=true;}
    require(refused && live_shared_control_metadata_bytes()==controls,"shared constructor lost control refusal or leaked control backing");
    if(fault==1)require(device.bytes()==0 && metadata.bytes()==0 && generic_quarantined_allocations()==before,
        "shared control refusal with successful cleanup retained credits");
    else {
        const auto retained=generic_retirement_snapshot_for_test();
        require(generic_quarantined_allocations()==before+1 && device.bytes()==128 &&
            metadata.bytes()==sizeof(DeviceAllocation::Retained) && retained.device_credit_bytes==128 &&
            retained.metadata_credit_bytes==metadata.bytes(),"shared control refusal lost failed device/record credits");
    }
}
void Exl3TextContext::fail_host_kv_retirement_for_test(bool device_mismatch) {
    require(impl_->host_kv_copy_stream && !impl_->host_kv_retirement_failure_for_test,
        "HostKV retirement injection requires unarmed transfer stream");
    impl_->host_kv_retirement_failure_for_test=true;
    impl_->host_kv_device_mismatch_for_test=device_mismatch;
}
void Exl3TextContext::fail_next_host_kv_copy_for_test() {
    require(impl_->host_kv_copy_stream && (impl_->host_kv_pinned_chunks || impl_->host_kv_batch_copy) &&
        !impl_->host_kv_copy_failure_for_test && !impl_->host_kv_retirement_failure_for_test,
        "HostKV copy injection requires unarmed pinned or batch transfer stream");
    impl_->host_kv_copy_failure_for_test=true;
}
void Exl3TextContext::fail_host_kv_device_query_for_test() {
    fail_host_kv_retirement_for_test();
    impl_->host_kv_device_query_failure_for_test=true;
}
void Exl3TextContext::fail_host_kv_compute_retirement_for_test() {
    require(impl_->host_kv_forward_used && !impl_->host_kv_compute_retirement_failure_for_test,
        "HostKV compute retirement injection requires unarmed used stream");
    impl_->host_kv_compute_retirement_failure_for_test=true;
}
void Exl3TextContext::retain_execution_stream_owner(cudaStream_t stream,std::shared_ptr<const void> owner) {
    require(stream && owner,"execution stream binding requires stream and owner");
    if(impl_->execution_stream_owner) {
        require(stream==impl_->owned_execution_stream &&
            !owner.owner_before(impl_->execution_stream_owner) && !impl_->execution_stream_owner.owner_before(owner),
            "execution stream owner cannot be replaced");
        return;
    }
    require((!impl_->host_kv_forward_used || !impl_->host_kv_forward_stream) &&
        !impl_->host_kv_failed && !impl_->host_kv_restore_source && position_==0,
        "execution stream owner binding requires pristine context");
    impl_->owned_execution_stream=stream;
    impl_->execution_stream_owner=std::move(owner);
}
void Exl3TextContext::fail_next_host_kv_restore_for_test() {
    require(impl_->host_kv.enabled && !impl_->host_kv_failed &&
        !impl_->host_kv_restore_failure_for_test,"HostKV restore injection requires intact unarmed context");
    impl_->host_kv_restore_failure_for_test=true;
}
void Exl3TextContext::observe_next_exact_restore_layer_for_test(
    std::function<void(int)> observer) {
    require(static_cast<bool>(observer) && !impl_->exact_restore_layer_observer_for_test &&
        !impl_->host_kv_failed,"exact restore layer observer requires intact unarmed context");
    impl_->exact_restore_layer_observer_for_test=std::move(observer);
}
void Exl3TextContext::fail_host_kv_prefill_boundary_for_test(unsigned stage) {
    require(impl_->host_kv.enabled && !impl_->host_kv_failed && position_==0 &&
        !impl_->host_kv_prefill_fault_for_test && (stage==1 || stage==2),
        "HostKV prefill fault requires pristine unarmed stage1..2");
    impl_->host_kv_prefill_fault_for_test=stage;
}
std::weak_ptr<const void> Exl3TextContext::host_kv_workspace_owner_for_test() const {
    require(impl_->host_kv.enabled && impl_->host_layer_k,
        "HostKV owner witness requires owned streamed K workspace");
    return std::weak_ptr<const void>(impl_->host_layer_k);
}
std::array<std::uint64_t,3> Exl3TextContext::shared_kv_credits_for_test(const std::shared_ptr<const void>& owner) noexcept {
    const auto* allocation=static_cast<const DeviceAllocation*>(owner.get());
    if(!allocation || !allocation->retirement)return {};
    const auto& record=*allocation->retirement;
    return {record.device_credit?record.device_credit->bytes():0,
        (record.metadata_credit?record.metadata_credit->bytes():0)+
            (allocation->owner_metadata_credit?allocation->owner_metadata_credit->bytes():0),
        allocation->shared_control_credit && allocation->shared_control_credit->ticket?
            allocation->shared_control_credit->ticket->bytes():0};
}
bool Exl3TextContext::fail_shared_kv_retirement_for_test(const std::shared_ptr<const void>& owner,unsigned fault) noexcept {
    auto* allocation=const_cast<DeviceAllocation*>(static_cast<const DeviceAllocation*>(owner.get()));
    if(!allocation || !allocation->ptr || !allocation->retirement || !allocation->shared_control_credit || fault<1 || fault>3)return false;
    allocation->cleanup_failure_for_test=fault==1;
    allocation->device_query_failure_for_test=fault==2;
    allocation->device_mismatch_for_test=fault==3;
    return true;
}
std::weak_ptr<const void> Exl3TextContext::quarantined_host_kv_workspace_for_test() noexcept {
    const auto* retained=Impl::host_kv_quarantine.load(std::memory_order_acquire);
    return retained?std::weak_ptr<const void>(retained->host_layer_k):std::weak_ptr<const void>{};
}
std::weak_ptr<const Exl3ExactKVPage> Exl3TextContext::host_kv_pending_page_owner_for_test() const noexcept {
    for(const auto& pieces:impl_->host_kv_pinned_scatter)
        for(const auto& piece:pieces)
            if(piece.page)return std::weak_ptr<const Exl3ExactKVPage>(piece.page);
    for(const auto& piece:impl_->host_kv_banked_d2h_scatter)
        if(piece.page)return std::weak_ptr<const Exl3ExactKVPage>(piece.page);
    return {};
}
Exl3TextContext::HostKVRetirementSnapshot Exl3TextContext::host_kv_retirement_snapshot() {
    const auto* retained=Impl::host_kv_quarantine.load(std::memory_order_acquire);
    if(!retained)return {};
    std::size_t pinned=0;
    for(const auto& staging:retained->host_kv_pinned_staging)if(staging)pinned+=staging->size();
    if(retained->host_kv_banked_d2h_staging)
        pinned+=retained->host_kv_banked_d2h_staging->size();
    std::size_t pending=0,pieces=0,failed_final_use=0;
    Exl3ResourceInventory page_metadata;
    std::uint64_t page_payload=0;
    const auto add_page=[&](const auto& page) {
        if(!page)return;
        const auto before=page_metadata.totals();
        page_metadata.add({page,0,Exl3ResourceInventory::Domain::host_metadata,
            sizeof(Exl3ExactKVPage)});
        if(page_metadata.totals()==before)return;
        const auto add_plane=[&](const auto& plane) {
            require(plane.capacity()<=(std::numeric_limits<std::uint64_t>::max()-page_payload)/sizeof(std::uint16_t),
                "retained HostKV page payload capacity overflow");
            page_payload+=plane.capacity()*sizeof(std::uint16_t);
        };
        for(const auto& plane:page->k)add_plane(plane);
        for(const auto& plane:page->v)add_plane(plane);
    };
    for(const auto& page:retained->exact_prefix_pages)add_page(page);
    std::uint64_t restore_payload=0;
    if(retained->host_kv_restore_source) {
        for(const auto& page:retained->host_kv_restore_source->kv_pages_)add_page(page);
        const std::array<std::shared_ptr<const Exl3ExactHostState>,1> source{retained->host_kv_restore_source};
        Exl3ExactHostState::visit_host_allocations(source,[&](const void*,std::size_t bytes) {
            require(bytes<=std::numeric_limits<std::uint64_t>::max()-restore_payload,
                "retained HostKV restore payload overflow");
            restore_payload+=bytes;
        },true);
    }
    for(std::size_t slot=0;slot<retained->host_kv_pinned_in_flight.size();++slot) {
        if(retained->host_kv_pinned_in_flight[slot] ||
           retained->registered_kv_pending[slot])++pending;
        if(slot<retained->host_kv_pinned_scatter.size()) {
            pieces+=retained->host_kv_pinned_scatter[slot].size();
            for(const auto& piece:retained->host_kv_pinned_scatter[slot])
                add_page(piece.page);
        }
        const auto final_use=retained->host_kv_pinned_final_use[slot].snapshot();
        if(final_use.generation && final_use.event && !final_use.ready && final_use.first_error)
            ++failed_final_use;
    }
    if(!retained->host_kv_banked_d2h_scatter.empty())++pending;
    pieces+=retained->host_kv_banked_d2h_scatter.size();
    for(const auto& piece:retained->host_kv_banked_d2h_scatter)add_page(piece.page);
    return {true,retained->host_kv_retirement_error,retained->persistent_bytes,pinned,sizeof(Impl),pending,pieces,failed_final_use,
        retained->host_kv.batch_metadata_bytes,retained->allocation_owner_metadata_bytes(),
        page_metadata.totals()[static_cast<unsigned>(Exl3ResourceInventory::Domain::host_metadata)],page_payload,
        restore_payload,retained->host_kv_restore_source?sizeof(Exl3ExactHostState):0,
        retained->host_kv_restore_source?retained->host_kv_restore_source->owner_metadata_bytes():0,
        retained->host_kv_restore_source?retained->host_kv_restore_source->recurrent_registered_bytes():0,
        retained->host_kv_forward_used,retained->host_kv_copy_stream!=nullptr,
        reinterpret_cast<std::uintptr_t>(retained->host_kv_forward_stream),retained->host_kv_forward_device,
        retained->host_kv_batch_use.pending(),
        {reinterpret_cast<std::uintptr_t>(retained->host_kv_batch_destinations.data()),
         reinterpret_cast<std::uintptr_t>(retained->host_kv_batch_sources.data()),
         reinterpret_cast<std::uintptr_t>(retained->host_kv_batch_sizes.data())},
        {retained->host_kv_batch_destinations.size(),retained->host_kv_batch_sources.size(),retained->host_kv_batch_sizes.size()},
        {retained->host_kv_batch_destinations.capacity(),retained->host_kv_batch_sources.capacity(),retained->host_kv_batch_sizes.capacity()}};
}
std::size_t Exl3TextContext::reconstruction_quarantined_allocations() noexcept {
    return ReconstructionBacking::quarantined_allocations.load(std::memory_order_acquire);
}
std::size_t Exl3TextContext::reconstruction_backing_metadata_bytes() noexcept {
    return sizeof(ReconstructionBacking)+ReconstructionBacking::control_bytes+sizeof(DeviceAllocation::Retained);
}
void Exl3TextContext::fail_reconstruction_retirement_for_test() {
    require(impl_->reconstruction_backing && impl_->reconstruction_backing->stream.used(),
        "reconstruction retirement injection requires an exercised slab");
    require(!impl_->reconstruction_retirement_failure_for_test,"reconstruction retirement injection already armed");
    impl_->reconstruction_retirement_failure_for_test=true;
}
void Exl3TextContext::fail_reconstruction_allocation_for_test(bool fail_prior_cleanup) {
    require(position_==0 && impl_->deferred_reconstruction_bytes && !impl_->reconstruction_backing,
        "reconstruction allocation injection requires deferred pristine context");
    require(!impl_->reconstruction_allocation_failure_for_test,"reconstruction allocation injection already armed");
    impl_->reconstruction_allocation_failure_for_test=true;
    impl_->reconstruction_rollback_cleanup_failure_for_test=fail_prior_cleanup;
}
void Exl3TextContext::fail_reconstruction_cleanup_for_test() {
    require(impl_->reconstruction_backing && !impl_->reconstruction_backing->stream.used(),
        "reconstruction cleanup fixture requires unused backing");
    require(!impl_->reconstruction_backing->cleanup_failure_for_test,"reconstruction cleanup injection already armed");
    impl_->reconstruction_backing->cleanup_failure_for_test=true;
}
bool Exl3TextContext::reconstruction_retirement_uncertain() const noexcept {
    return impl_->reconstruction_retirement_failure_for_test ||
        (impl_->reconstruction_backing && impl_->reconstruction_backing->stream.failed());
}
std::uint64_t Exl3TextContext::stream_reduction_calls(bool extended) const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->stream_reduction_calls(extended);
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->stream_reduction_calls(extended);
    return result;
}
std::uint64_t Exl3TextContext::fast_same_weights_fp16kv_gdn_decode_conv_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->fast_same_weights_fp16kv_gdn_decode_conv_calls();
    return result;
}
std::uint64_t Exl3TextContext::fast_same_weights_fp16kv_gdn_m1_gate_up_pair_submissions() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->fast_same_weights_fp16kv_gdn_m1_gate_up_pair_submissions();
    return result;
}
std::uint64_t Exl3TextContext::fast_wide_prefill_gemm_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->fast_wide_prefill_gemm_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->fast_wide_prefill_gemm_calls();
    return result;
}
std::uint64_t Exl3TextContext::fast_wide_prefill_gemm_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->fast_wide_prefill_gemm_rows();
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->fast_wide_prefill_gemm_rows();
    return result;
}
std::uint64_t Exl3TextContext::exact_attention_gqa_six_softmax_fused_scalar_values_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_fused_scalar_values_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::exact_attention_gqa_six_softmax_fused_scalar_values_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_fused_scalar_values_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::exact_attention_gqa_six_decode_fused_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->exact_attention_gqa_six_decode_fused_calls();
    return result;
}
std::uint64_t Exl3TextContext::k6_gateup_warpgroup_async_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_gateup_warpgroup_async_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_gateup_warpgroup_async_calls();
    return result;
}
std::uint64_t Exl3TextContext::k6_gateup_n32_pair_cta_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_gateup_n32_pair_cta_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_gateup_n32_pair_cta_calls();
    return result;
}
std::uint64_t Exl3TextContext::k6_fast_decode_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_fast_decode_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_fast_decode_calls();
    return result;
}
std::uint64_t Exl3TextContext::k6_fast_decode_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_fast_decode_rows();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_fast_decode_rows();
    return result;
}
std::uint64_t Exl3TextContext::k6_rowpair_n64_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_rowpair_n64_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_rowpair_n64_calls();
    return result;
}
std::uint64_t Exl3TextContext::k6_rowpair_n64_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_rowpair_n64_rows();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_rowpair_n64_rows();
    return result;
}
std::uint64_t Exl3TextContext::k6_down_rowpair_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_down_rowpair_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_down_rowpair_calls();
    return result;
}
std::uint64_t Exl3TextContext::k6_down_rowpair_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_down_rowpair_rows();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_down_rowpair_rows();
    return result;
}
std::uint64_t Exl3TextContext::shape4_n64_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->shape4_n64_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->shape4_n64_calls();
    return result;
}
std::uint64_t Exl3TextContext::shape4_n64_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->shape4_n64_rows();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->shape4_n64_rows();
    return result;
}
std::uint64_t Exl3TextContext::reduce_shfl_min_barrier_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->reduce_shfl_min_barrier_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->reduce_shfl_min_barrier_calls();
    return result;
}
std::uint64_t Exl3TextContext::reduce_shfl_min_barrier_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->reduce_shfl_min_barrier_rows();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->reduce_shfl_min_barrier_rows();
    return result;
}
std::uint64_t Exl3TextContext::k6_gateup_n32_pair_cta_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k6_gateup_n32_pair_cta_rows();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k6_gateup_n32_pair_cta_rows();
    return result;
}
std::uint64_t Exl3TextContext::k7_tiles64_exact_splits_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->k7_tiles64_exact_splits_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->k7_tiles64_exact_splits_calls();
    return result;
}
std::uint64_t Exl3TextContext::target_k8_kv_prefill_async_a_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->target_k8_kv_prefill_async_a_calls();
    return result;
}
std::uint64_t Exl3TextContext::target_k8_kv_prefill_async_a_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->target_k8_kv_prefill_async_a_rows();
    return result;
}
std::uint64_t Exl3TextContext::target_down_k6_async_a_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->target_down_k6_async_a_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->target_down_k6_async_a_calls();
    return result;
}
std::uint64_t Exl3TextContext::target_gateup_k6_n16_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->target_gateup_k6_n16_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->target_gateup_k6_n16_calls();
    return result;
}
std::uint64_t Exl3TextContext::target_k6_small_m_async_a_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->target_k6_small_m_async_a_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->target_k6_small_m_async_a_calls();
    return result;
}
std::uint64_t Exl3TextContext::target_k7_small_m_async_a_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->target_k7_small_m_async_a_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->target_k7_small_m_async_a_calls();
    return result;
}
std::uint64_t Exl3TextContext::target_k5_small_m_batch_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->target_k5_small_m_batch_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->target_k5_small_m_batch_calls();
    return result;
}
std::uint64_t Exl3TextContext::native_k6_critical_path_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)result+=layer->native_k6_critical_path_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)result+=layer->native_k6_critical_path_calls();
    return result;
}

std::uint64_t Exl3TextContext::native_k6_register_pipeline_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)
        if(layer)result+=layer->native_k6_register_pipeline_calls();
    for(const auto& layer:impl_->gdn_layers)
        if(layer)result+=layer->native_k6_register_pipeline_calls();
    return result;
}
std::uint64_t Exl3TextContext::eager_mlp_gateup_concurrent_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->eager_mlp_gateup_concurrent_calls();
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->eager_mlp_gateup_concurrent_calls();
    return result;
}
std::uint64_t Exl3TextContext::prefill_qkv_concurrent_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->prefill_qkv_concurrent_calls();
    return result;
}
std::uint64_t Exl3TextContext::prefill_qkv_concurrent_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->prefill_qkv_concurrent_rows();
    return result;
}
std::uint64_t Exl3TextContext::prefill_projection_graph_captures() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->prefill_projection_graph_captures();
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->prefill_projection_graph_captures();
    return result;
}
std::uint64_t Exl3TextContext::prefill_projection_graph_replays() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->prefill_projection_graph_replays();
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->prefill_projection_graph_replays();
    return result;
}
std::uint64_t Exl3TextContext::prefill_projection_graph_binding_fallbacks() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->prefill_projection_graph_binding_fallbacks();
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->prefill_projection_graph_binding_fallbacks();
    return result;
}

std::uint64_t Exl3TextContext::prefill_projection_chain_graph_captures() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->prefill_projection_chain_graph_stats().captures;
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->prefill_projection_chain_graph_stats().captures;
    return result;
}

std::uint64_t Exl3TextContext::prefill_projection_chain_graph_replays() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->prefill_projection_chain_graph_stats().replays;
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->prefill_projection_chain_graph_stats().replays;
    return result;
}

std::uint64_t Exl3TextContext::prefill_projection_chain_graph_fallbacks() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->prefill_projection_chain_graph_stats().eager_fallbacks;
    for(const auto& layer:impl_->gdn_layers)if(layer)
        result+=layer->prefill_projection_chain_graph_stats().eager_fallbacks;
    return result;
}
std::uint64_t Exl3TextContext::target_prefill_gate_up_pair_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->target_prefill_gate_up_pair_attempts();
    return result;
}
std::uint64_t Exl3TextContext::target_prefill_gate_up_pair_calls() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->target_prefill_gate_up_pair_calls();
    return result;
}
std::uint64_t Exl3TextContext::target_prefill_gate_up_pair_rows() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->target_prefill_gate_up_pair_rows();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_v_tile_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_v_tile_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_v_tile_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_v_tile_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_full_cta_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_full_cta_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_full_cta_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_full_cta_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_threads128_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_threads128_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_threads128_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_threads128_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_score_tile_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_score_tile_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_score_tile_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_score_tile_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_key_pair_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_key_pair_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_key_pair_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_key_pair_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_warp_score_broadcast_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_warp_score_broadcast_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_warp_score_broadcast_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_warp_score_broadcast_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_pair_dimensions_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_pair_dimensions_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_pair_dimensions_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_pair_dimensions_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_six_values_single_load_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_six_values_single_load_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_six_values_single_load_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_six_values_single_load_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_scalar_dim_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_scalar_dim_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_scalar_dim_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_scalar_dim_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_two_query_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_two_query_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_two_query_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_two_query_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_fused_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_fused_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_triple_fused_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_triple_fused_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_tile512_launch_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_tile512_launch_attempts();
    return result;
}
std::uint64_t Exl3TextContext::gqa_six_softmax_tile512_row_attempts() const noexcept {
    std::uint64_t result=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        result+=layer->gqa_six_softmax_tile512_row_attempts();
    return result;
}
std::uint64_t Exl3TextContext::allocation_owner_metadata_bytes() const {
    return impl_->allocation_owner_metadata_bytes();
}
std::uint64_t Exl3TextContext::base_linear_owner_metadata_bytes() const noexcept {
    return (std::uint64_t(bool(impl_->head_workspace))+
            std::uint64_t(bool(impl_->gdn_qkvz_z_workspace))+
            std::uint64_t(bool(impl_->eager_mlp_gateup_up_workspace))+
            std::uint64_t(bool(impl_->prefill_qkv_k_workspace))+
            std::uint64_t(bool(impl_->prefill_qkv_v_workspace)))*
        Exl3CudaLinearWorkspace::metadata_bytes();
}
std::uint64_t Exl3TextContext::private_generic_owner_metadata_bytes() const noexcept {
    std::uint64_t count=impl_->allocations.size();
    for(const auto& child:impl_->taps)if(child)++count;
    count+=bool(impl_->embedding_trace)+bool(impl_->media_features)+bool(impl_->media_positions)+bool(impl_->greedy_rows);
    std::uint64_t bytes=count*DeviceAllocation::owner_metadata_bytes();
    for(const auto& transfer:impl_->greedy_transfers)if(transfer)
        bytes+=Exl3GreedyPacketTransfer::metadata_bytes_required();
    return bytes;
}
std::uint64_t Exl3TextContext::shared_generic_owner_metadata_bytes() const noexcept {
    std::uint64_t count=bool(impl_->host_layer_k)+bool(impl_->host_layer_v);
    for(const auto& child:impl_->cache_k)if(child)++count;
    for(const auto& child:impl_->cache_v)if(child)++count;
    return count*DeviceAllocation::owner_metadata_bytes();
}
std::uint64_t Exl3TextContext::layer_linear_owner_metadata_bytes() const noexcept {
    std::uint64_t count=0;
    for(const auto& layer:impl_->full_layers)if(layer)
        for(const auto* child:layer->linear_workspace_owners())if(child)++count;
    for(const auto& layer:impl_->gdn_layers)if(layer)
        for(const auto* child:layer->linear_workspace_owners())if(child)++count;
    return count*Exl3CudaLinearWorkspace::metadata_bytes();
}
bool Exl3TextContext::fail_linear_retirement_for_test(unsigned slot,bool partial) noexcept {
    Exl3CudaLinearWorkspace* workspace=nullptr;
    if(slot==0)workspace=impl_->head_workspace.get();
    else if(slot==1)workspace=impl_->gdn_qkvz_z_workspace
        ? impl_->gdn_qkvz_z_workspace.get()
        : impl_->eager_mlp_gateup_up_workspace.get();
    else if(slot==2)workspace=impl_->continuation?impl_->continuation->head_workspace.get():nullptr;
    else {
        const auto layer=(slot-3)/13,child=(slot-3)%13;
        if(layer>=kLayers)return false;
        if(child<7 && impl_->full_layers[layer])workspace=impl_->full_layers[layer]->linear_workspace_owners()[child];
        if(child>=7 && impl_->gdn_layers[layer])workspace=impl_->gdn_layers[layer]->linear_workspace_owners()[child-7];
    }
    if(!workspace || !workspace->workspace_bytes())return false;
    workspace->fail_owned_retirement_for_test(partial);return true;
}
bool Exl3TextContext::fail_gdn_buffer_retirement_for_test(unsigned owned_index) noexcept {
    for(const auto& layer:impl_->gdn_layers)if(layer)
        for(auto* child:layer->buffer_retirement_owners())if(child) {
            if(owned_index--==0){child->fail_cleanup_for_test(static_cast<int>(cudaErrorUnknown));return true;}
        }
    return false;
}
std::uint64_t Exl3TextContext::fail_layer_buffer_retirement_for_test(bool gdn,unsigned layer,unsigned slot) noexcept {
    if(layer>=kLayers)return 0;
    Exl3LayerBufferRetirement* child=nullptr;
    if(gdn) {
        if(!impl_->gdn_layers[layer] || slot>=34)return 0;
        child=impl_->gdn_layers[layer]->buffer_retirement_owners()[slot];
    } else {
        if(!impl_->full_layers[layer] || slot>=18)return 0;
        child=impl_->full_layers[layer]->buffer_retirement_owners()[slot];
    }
    if(!child || !child->bytes())return 0;
    child->fail_cleanup_for_test(static_cast<int>(cudaErrorUnknown));return child->bytes();
}
bool Exl3TextContext::fail_attention_buffer_retirement_for_test(unsigned owned_index) noexcept {
    for(const auto& layer:impl_->full_layers)if(layer)
        for(auto* child:layer->buffer_retirement_owners())if(child) {
            if(owned_index--==0){child->fail_cleanup_for_test(static_cast<int>(cudaErrorUnknown));return true;}
        }
    return false;
}
bool Exl3TextContext::fail_continuation_allocation_retirement_for_test(unsigned slot,unsigned fault) noexcept {
    if(!impl_->continuation || slot>1 || fault<1 || fault>3)return false;
    auto* owner=slot?impl_->continuation->logits.get():impl_->continuation->final_norm.get();
    if(!owner || !owner->ptr)return false;
    owner->cleanup_failure_for_test=fault==1;
    owner->device_query_failure_for_test=fault==2;
    owner->device_mismatch_for_test=fault==3;
    return true;
}
std::uint64_t Exl3TextContext::Impl::allocation_owner_metadata_bytes() const {
    const auto* impl_=this;
    Exl3ResourceInventory::Requirement requirement;
    using Domain=Exl3ResourceInventory::Domain;
    if(impl_->allocations.size())requirement.add(Domain::host_metadata,
        impl_->allocations.size(),DeviceAllocation::owner_metadata_bytes());
    if(impl_->device_prefix && !impl_->device_prefix_shared && !impl_->device_prefix_metadata_external)
        requirement.add(Domain::host_metadata,1,impl_->device_prefix->owner_metadata_bytes());
    std::size_t shared_owners=0;
    if(impl_->host_layer_k)++shared_owners;
    if(impl_->host_layer_v)++shared_owners;
    for(const auto& owner:impl_->cache_k)if(owner)++shared_owners;
    for(const auto& owner:impl_->cache_v)if(owner)++shared_owners;
    if(shared_owners)requirement.add(Domain::host_metadata,shared_owners,DeviceAllocation::shared_control_bytes);
    std::size_t separate_owners=0;
    if(impl_->host_layer_k)++separate_owners;
    if(impl_->host_layer_v)++separate_owners;
    for(const auto& owner:impl_->cache_k)if(owner)++separate_owners;
    for(const auto& owner:impl_->cache_v)if(owner)++separate_owners;
    for(const auto& owner:impl_->taps)if(owner)++separate_owners;
    if(impl_->embedding_trace)++separate_owners;
    if(impl_->media_features)++separate_owners;
    if(impl_->media_positions)++separate_owners;
    if(impl_->greedy_rows)++separate_owners;
    if(separate_owners)requirement.add(Domain::host_metadata,separate_owners,DeviceAllocation::owner_metadata_bytes());
    for(const auto& transfer:impl_->greedy_transfers)if(transfer)
        requirement.add(Domain::host_metadata,1,Exl3GreedyPacketTransfer::metadata_bytes_required());
    if(impl_->head_workspace)requirement.add(Domain::host_metadata,1,Exl3CudaLinearWorkspace::metadata_bytes());
    if(impl_->gdn_qkvz_z_workspace)requirement.add(Domain::host_metadata,1,Exl3CudaLinearWorkspace::metadata_bytes());
    if(impl_->eager_mlp_gateup_up_workspace)
        requirement.add(Domain::host_metadata,1,
                        Exl3CudaLinearWorkspace::metadata_bytes());
    if(impl_->prefill_qkv_k_workspace)
        requirement.add(Domain::host_metadata,1,Exl3CudaLinearWorkspace::metadata_bytes());
    if(impl_->prefill_qkv_v_workspace)
        requirement.add(Domain::host_metadata,1,Exl3CudaLinearWorkspace::metadata_bytes());
    if(impl_->numeric_prefill_projection_workspace)
        requirement.add(Domain::host_metadata,1,sizeof(Exl3CudaReconstructGemmWorkspace));
    for(const auto& staging:impl_->host_kv_pinned_staging)if(staging)
        requirement.add(Domain::host_metadata,1,sizeof(PinnedHostBuffer));
    if(impl_->host_kv_banked_d2h_staging)
        requirement.add(Domain::host_metadata,1,sizeof(PinnedHostBuffer));
    for(const auto& layer:impl_->full_layers)if(layer)
        requirement.add(Domain::host_metadata,1,layer->fixed_owner_metadata_bytes());
    for(const auto& layer:impl_->gdn_layers)if(layer)
        requirement.add(Domain::host_metadata,1,layer->fixed_owner_metadata_bytes());
    return requirement.units[static_cast<unsigned>(Domain::host_metadata)];
}
GoptSubmissions Exl3TextContext::gaming_submissions() const noexcept {
    if(!impl_)return {};
    auto result=impl_->gaming_submissions;
    for(const auto& layer:impl_->gdn_layers)if(layer) {
        const auto values=layer->gaming_submissions();
        for(unsigned i=0;i<result.size();++i)result[i]+=values[i];
    }
    return result;
}

Exl3HostKVStats Exl3TextContext::host_kv_stats() const noexcept {
    auto result=impl_->host_kv;
    for(const auto& layer:impl_->full_layers) if(layer) {
        result.query_pair_launch_attempts+=layer->query_pair_launch_attempts();
        result.query_pair_row_attempts+=layer->query_pair_row_attempts();
        result.query_pair_requested_row_attempts+=layer->query_pair_requested_row_attempts();
        result.gqa_six_query_pair_score_launch_attempts+=
            layer->gqa_six_query_pair_score_launch_attempts();
        result.gqa_six_query_pair_score_row_attempts+=
            layer->gqa_six_query_pair_score_row_attempts();
        result.gqa_six_score_k_tile64_launch_attempts+=
            layer->gqa_six_score_k_tile64_launch_attempts();
        result.gqa_six_score_k_tile64_row_attempts+=
            layer->gqa_six_score_k_tile64_row_attempts();
        result.gqa_six_softmax_triple_value_launch_attempts+=
            layer->gqa_six_softmax_triple_value_launch_attempts();
        result.gqa_six_softmax_triple_value_row_attempts+=
            layer->gqa_six_softmax_triple_value_row_attempts();
        result.gqa_six_softmax_triple_pair_dimensions_launch_attempts+=
            layer->gqa_six_softmax_triple_pair_dimensions_launch_attempts();
        result.gqa_six_softmax_triple_pair_dimensions_row_attempts+=
            layer->gqa_six_softmax_triple_pair_dimensions_row_attempts();
        result.gqa_six_softmax_six_values_single_load_launch_attempts+=
            layer->gqa_six_softmax_six_values_single_load_launch_attempts();
        result.gqa_six_softmax_six_values_single_load_row_attempts+=
            layer->gqa_six_softmax_six_values_single_load_row_attempts();
        result.gqa_six_softmax_triple_key_pair_launch_attempts+=
            layer->gqa_six_softmax_triple_key_pair_launch_attempts();
        result.gqa_six_softmax_triple_key_pair_row_attempts+=
            layer->gqa_six_softmax_triple_key_pair_row_attempts();
    }
    return result;
}
Exl3TextContext::HostKVGdnSegmentGraphStats
Exl3TextContext::host_kv_gdn_segment_graph_stats() const noexcept {
    return {impl_->host_kv_gdn_segment_graph_captures,
        impl_->host_kv_gdn_segment_graph_replays,
        impl_->host_kv_gdn_segment_graph_capture_ms,
        impl_->host_kv_gdn_segment_graph_launch_cpu_ns};
}
Exl3TextContext::HostKVFullLayerGraphStats
Exl3TextContext::host_kv_full_layer_graph_stats() const noexcept {
    return {impl_->host_kv_full_layer_graph_captures,
        impl_->host_kv_full_layer_graph_replays,
        impl_->host_kv_full_layer_graph_six_softmax_triple_captures,
        impl_->host_kv_full_layer_graph_k6_stream_reduction_captures,
        impl_->host_kv_full_layer_graph_extended_stream_reduction_captures,
        impl_->host_kv_full_layer_graph_target_down_k6_async_a_captures,
        impl_->host_kv_full_layer_graph_target_k6_small_m_async_a_captures,
        impl_->host_kv_full_layer_graph_target_k7_small_m_async_a_captures,
        impl_->host_kv_full_layer_graph_capture_ms,
        impl_->host_kv_full_layer_graph_launch_cpu_ns};
}
Exl3TextContext::OrdinaryFullLayerGraphStats
Exl3TextContext::ordinary_full_layer_graph_stats() const noexcept {
    return {impl_->ordinary_full_layer_graph_captures,
        impl_->ordinary_full_layer_graph_replays,
        impl_->ordinary_full_layer_graph_capture_ms};
}
Exl3TextContext::OrdinaryGraphProcessStats
Exl3TextContext::ordinary_graph_process_stats_for_test() noexcept {
    const auto& c=ordinary_graph_process_counters;
    return {c.gdn_segment_captures.load(std::memory_order_relaxed),
        c.gdn_segment_replays.load(std::memory_order_relaxed),
        c.full_layer_captures.load(std::memory_order_relaxed),
        c.full_layer_replays.load(std::memory_order_relaxed),
        c.mlp_tail_captures.load(std::memory_order_relaxed),
        c.mlp_tail_replays.load(std::memory_order_relaxed)};
}
Exl3TextContext::HostKVMlpTailGraphStats
Exl3TextContext::host_kv_mlp_tail_graph_stats() const noexcept {
    return {impl_->host_kv_mlp_tail_graph_captures,
        impl_->host_kv_mlp_tail_graph_replays,
        impl_->host_kv_mlp_tail_graph_capture_ms};
}
int Exl3TextContext::rope_offset() const noexcept {return impl_->rope_offset;}
Exl3ReconstructGemmStats Exl3TextContext::numeric_prefill_projection_stats() const noexcept {
    return impl_->numeric_prefill_projection_workspace
        ? impl_->numeric_prefill_projection_workspace->stats()
        : Exl3ReconstructGemmStats{};
}
Exl3TextContext::FastSameWeightsFp16KvPrefillStats
Exl3TextContext::fast_same_weights_fp16kv_prefill_stats() const noexcept {
    return impl_ ? impl_->fast_same_weights_fp16kv_prefill_stats :
        FastSameWeightsFp16KvPrefillStats{};
}
Exl3TextContext::HeadWorkStats Exl3TextContext::head_work_stats() const noexcept {
    return impl_->head_work;
}

std::uint64_t Exl3TextContext::paired_transform_submissions() const noexcept {
    std::uint64_t result=0;
    if(impl_)for(const auto& layer:impl_->gdn_layers)
        if(layer)result+=layer->paired_transform_submissions();
    return result;
}

std::uint64_t Exl3TextContext::fused_gate_up_submissions() const noexcept {
    std::uint64_t result=0;
    if(impl_)for(const auto& layer:impl_->gdn_layers)
        if(layer)result+=layer->fused_gate_up_submissions();
    if(impl_)for(const auto& layer:impl_->full_layers)
        if(layer)result+=layer->fused_gate_up_submissions();
    return result;
}

std::uint64_t Exl3TextContext::fused_residual_norm_submissions() const noexcept {
    std::uint64_t result=0;
    if(impl_)for(const auto& layer:impl_->gdn_layers)
        if(layer)result+=layer->fused_residual_norm_submissions();
    return result;
}

Exl3ReconstructedExactStats Exl3TextContext::reconstructed_exact_stats() const noexcept {
    auto result=impl_->reconstruction_backing ? impl_->reconstruction_backing->stats : Exl3ReconstructedExactStats{};
    result.reservation_fallback=impl_->reconstruction_budget_fallback;
    return result;
}
bool Exl3TextContext::oscar_only_context() const noexcept {return impl_->oscar_only;}
int Exl3TextContext::max_context() const noexcept {return impl_->max_context;}
std::shared_ptr<const void> Exl3TextModel::metadata_owner() const noexcept {
    return std::shared_ptr<const void>(impl_,&impl_->collection);
}
std::shared_ptr<const void> Exl3TextContext::metadata_owner() const noexcept {
    return std::shared_ptr<const void>(impl_->model,&impl_->model->collection);
}
std::shared_ptr<const void> Exl3TextContext::model_identity() const noexcept {return impl_->model->host_state_identity;}
std::size_t Exl3TextContext::cache_staging_bytes() const noexcept {
    return impl_->host_layer_k ? impl_->host_layer_k->bytes+impl_->host_layer_v->bytes : 0;
}

void Exl3TextContext::reset(cudaStream_t stream) {
    reset_impl(stream,false);
}

void Exl3TextContext::reset_impl(cudaStream_t stream,bool preserve_exact_payload) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    require(!impl_->host_kv_failed,"context reset cannot reuse failed HostKV lineage");
    require(!impl_->reconstruction_backing || !impl_->reconstruction_backing->stream.failed(),
        "context reset cannot reuse failed reconstruction slab");
    impl_->drain_continuation_graph_uses();
    impl_->retain_host_kv_forward_stream(stream);
    Impl::HostKVForwardFailureScope failure_scope{*impl_};
    // A captured decode graph is bound to the prior eager/OSCAR lifecycle.
    // Invalidate its admission immediately; the objects are replaced on the
    // next synchronized capture, after reset/reingest work on this stream.
    impl_->graph_active = false;
    impl_->graph_capture_active = false;
    impl_->graph_reason = "not captured after context reset";
    if(!preserve_exact_payload) {
    impl_->resident_exact_state_id=0;
    impl_->exact_prefix_pages.clear();
    if(impl_->device_prefix && !impl_->device_prefix_shared)impl_->device_prefix->invalidate();
    impl_->exact_prefix_position = 0;
    impl_->rope_offset=0;
    impl_->host_kv_failed=false;
    if(impl_->host_kv.enabled || impl_->oscar_only) {
        cuda_check(cudaMemsetAsync(impl_->host_layer_k->ptr,0,impl_->host_layer_k->bytes,stream),"reset streamed K");
        cuda_check(cudaMemsetAsync(impl_->host_layer_v->ptr,0,impl_->host_layer_v->bytes,stream),"reset streamed V");
    }
    }
    if (impl_->transaction) {
        impl_->transaction->active = false;
        impl_->transaction->fresh_snapshot = false;
        impl_->transaction->prefix_available = false;
        impl_->transaction->attempted_rows = 0;
        impl_->transaction->continuation_graph_rows = 0;
        impl_->transaction->continuation_graph_attempt = false;
        impl_->transaction->host_kv_page_count = 0;
        impl_->transaction->host_kv_tail.reset();
    }
    for (auto& layer : impl_->full_layers)
        if (layer) layer->invalidate_retained_prefix();
    if (impl_->continuation) {
        impl_->continuation->rows = 0;
        impl_->continuation->graph_active = false;
        for (auto& graph : impl_->continuation->additional_graphs) {
            graph.active = false;
            const bool graph_cached = graph.definition.ready() &&
                graph.executable.ready() && graph.active_generation != 0 &&
                graph.origin_stream != nullptr && graph.compatibility.valid() &&
                graph.lifecycle.snapshot().phase==Exl3BoundedGraphEntry::Phase::bound;
            if (graph_cached)
                graph.reason =
                    "cached after context reset; request state not yet compatible";
            else {
                graph.split_class = 0;
                graph.active_generation = 0;
                graph.origin_stream = nullptr;
                graph.route_bits = 0;
                graph.compatibility.invalidate();
                graph.lifecycle.invalidate("context reset discarded incomplete graph");
                graph.reason = "not captured after context reset";
            }
        }
        const bool cached=impl_->continuation->graph_definition.ready() &&
            impl_->continuation->graph_executable.ready() &&
            impl_->continuation->graph_active_generation!=0 &&
            impl_->continuation->graph_origin_stream!=nullptr &&
            impl_->continuation->graph_compatibility.valid() &&
            impl_->continuation->graph_lifecycle.snapshot().phase==
                Exl3BoundedGraphEntry::Phase::bound;
        if(cached) {
            // The executable owns only context-stable allocations. Reset
            // revokes launch admission but keeps its exact stream/split key so
            // a later compatible request can reactivate it without mutation.
            impl_->continuation->graph_reason =
                "cached after context reset; request state not yet compatible";
        } else {
            impl_->continuation->graph_split_class = 0;
            impl_->continuation->graph_active_generation = 0;
            impl_->continuation->graph_origin_stream = nullptr;
            impl_->continuation->graph_route_bits = 0;
            impl_->continuation->graph_compatibility.invalidate();
            impl_->continuation->graph_lifecycle.invalidate(
                "context reset discarded incomplete graph");
            impl_->continuation->graph_reason = "not captured after context reset";
        }
    }
    if(!preserve_exact_payload) {
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->gdn_layers[layer]) impl_->gdn_layers[layer]->reset(stream);
        if (impl_->cache_k[layer]) {
            cuda_check(cudaMemsetAsync(impl_->cache_k[layer]->ptr, 0, impl_->cache_k[layer]->bytes, stream), "reset E4A K cache");
            cuda_check(cudaMemsetAsync(impl_->cache_v[layer]->ptr, 0, impl_->cache_v[layer]->bytes, stream), "reset E4A V cache");
        }
    }
    cuda_check(cudaMemsetAsync(impl_->position_device, 0, sizeof(int), stream),
               "reset E4B2 position parameter");
    if (impl_->oscar) impl_->oscar->reset();
    position_ = 0;
    impl_->tap_rows = 0;
    impl_->embedding_rows = 0;
    impl_->last_rows = 0;
    impl_->last_hidden_source = nullptr;
    impl_->last_hidden_source_rows = 0;
    impl_->last_hidden_source_first_row = 0;
    impl_->last_hidden_source_position = 0;
    if (++impl_->last_hidden_generation == 0)
        throw std::overflow_error("native MTP hidden generation exhausted");
    impl_->tap_generation.fetch_add(1,std::memory_order_release);
    }
    // A request/context reset revokes every previously captured hidden range,
    // including a preserved exact HostKV root.  The prepared allocation stays
    // owned for reuse, but no row from the prior request may be exposed.
    impl_->invalidate_native_mtp_hidden_capture();
    last_decode_h2d_ = 0;
    impl_->qkv_trace_valid = false;
    if (impl_->transaction) impl_->transaction->rollback_required = false;
    impl_->fast_prefill_failed = false;
}

std::size_t Exl3TextContext::transaction_bytes_required() const {
    if (impl_->transaction) return 0;
    const bool host_kv = impl_->host_kv.enabled && impl_->oscar == nullptr;
    const bool device_kv = !impl_->host_kv.enabled && impl_->oscar == nullptr &&
        fast_device_kv_transaction_enabled();
    require(host_kv || device_kv ||
            (impl_->oscar != nullptr && impl_->oscar->graph_class() == 0),
            "P2 target transaction requires eager OSCAR, HostKV or guarded device KV");
    std::size_t bytes = 0;
    for (const auto& layer : impl_->gdn_layers) {
        if (!layer) continue;
        bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
            bytes, layer->recurrent_state_bytes());
        bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
            bytes, layer->physical_conv_state_bytes());
    }
    // Preserve the seven-owner transaction inventory. Exact HostKV does not
    // need an OSCAR-cache image because its ordinary K/V rows are already in
    // the authoritative cache and rows beyond the repaired cursor are inert.
    bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
        bytes, (host_kv || device_kv) ? 1 : impl_->oscar->checkpoint_device_bytes());
    bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
        bytes, kVocab * sizeof(std::uint16_t));
    bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
        bytes, kTapLayers.size() * impl_->prefill_capacity * kHidden *
                   sizeof(std::uint16_t));
    bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
        bytes, impl_->prefill_capacity * kHidden * sizeof(std::uint16_t));
    return Exl3LinearWorkspaceRequirements::append_owned_bytes(bytes, sizeof(int));
}

std::size_t Exl3TextContext::transaction_owner_metadata_bytes() noexcept {
    return bounded_shared_allocation_bytes<Impl::Transaction>() +
        7 * DeviceAllocation::owner_metadata_bytes();
}

void Exl3TextContext::prepare_transaction() {
    prepare_transaction_impl(nullptr, 0);
}

void Exl3TextContext::prepare_transaction_reserved(
    Exl3VeriCacheServingCoordinator& authority, unsigned startup_fault_for_test) {
    prepare_transaction_impl(&authority, startup_fault_for_test);
}

void Exl3TextContext::prepare_transaction_impl(
    Exl3VeriCacheServingCoordinator* authority, unsigned startup_fault) {
    impl_->drain_repair();
    require(startup_fault <= 5 && (!startup_fault || authority),
            "P2 target transaction startup fault requires reserved stage1..5");
    if (impl_->transaction) return;
    const bool host_kv = impl_->host_kv.enabled && impl_->oscar == nullptr;
    const bool device_kv = !impl_->host_kv.enabled && impl_->oscar == nullptr &&
        fast_device_kv_transaction_enabled();
    require(host_kv || device_kv ||
            (impl_->oscar != nullptr && impl_->oscar->graph_class() == 0),
            "P2 target transaction requires eager OSCAR, HostKV or guarded device KV");
    require(impl_->capture_taps,
            "P2 target transaction requires captured taps");
    require(!impl_->graph_active && !impl_->graph_capture_active,
            "P2 target transaction setup is eager-only");
    const auto required = transaction_bytes_required();

    const auto construct = [&] {
        auto prepared = make_bounded_shared<Impl::Transaction>();
        prepared->host_kv = host_kv;
        prepared->device_kv = device_kv;
        const auto allocate = [&](std::size_t bytes, const char* label) {
            if (!authority)
                return std::make_unique<DeviceAllocation>(bytes, label);
            auto credits = authority->reserve_constructor_credits(
                bytes, DeviceAllocation::owner_metadata_bytes());
            return std::make_unique<DeviceAllocation>(
                bytes, label, std::move(credits.device),
                std::move(credits.metadata));
        };
        std::size_t recurrent_bytes = 0;
        std::size_t conv_bytes = 0;
        for (const auto& layer : impl_->gdn_layers) {
            if (!layer) continue;
            recurrent_bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
                recurrent_bytes, layer->recurrent_state_bytes());
            conv_bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
                conv_bytes, layer->physical_conv_state_bytes());
        }
        prepared->gdn_recurrent = allocate(
            recurrent_bytes, "allocate P2 target recurrent checkpoint");
        prepared->gdn_conv = allocate(
            conv_bytes, "allocate P2 target convolution checkpoint");
        auto* recurrent = static_cast<std::byte*>(prepared->gdn_recurrent->ptr);
        auto* conv = static_cast<std::byte*>(prepared->gdn_conv->ptr);
        for (int layer = 0; layer < kLayers; ++layer) {
            if (!impl_->gdn_layers[layer]) continue;
            const auto recurrent_size =
                impl_->gdn_layers[layer]->recurrent_state_bytes();
            const auto conv_size =
                impl_->gdn_layers[layer]->physical_conv_state_bytes();
            auto& checkpoint=prepared->gdn_checkpoints[layer];
            const bool recurrent_trace_alias = host_kv &&
                impl_->host_kv_transaction_recurrent_trace_enabled;
            checkpoint={recurrent_trace_alias ?
                    static_cast<void*>(impl_->gdn_layers[layer]->recurrent_state_before_) :
                    static_cast<void*>(recurrent),
                recurrent_size,conv,conv_size};
            checkpoint.recurrent_trace_alias=recurrent_trace_alias;
            checkpoint.model_owner=impl_->model->host_state_identity;
            checkpoint.model_layer=layer;
            checkpoint.recurrent_layer_stride_bytes=Exl3GdnRecurrentLayout::recurrent_bytes;
            checkpoint.convolution_layer_stride_bytes=
                Exl3GdnRecurrentLayout::convolution_storage_bytes;
            recurrent += recurrent_size;
            conv += conv_size;
        }
        prepared->oscar_cache = allocate(
            (host_kv || device_kv) ? 1 : impl_->oscar->checkpoint_device_bytes(),
            (host_kv || device_kv) ? "allocate P2 target KV transaction sentinel" :
                      "allocate P2 target OSCAR checkpoint");
        prepared->oscar_checkpoint.cache_device = prepared->oscar_cache->ptr;
        prepared->oscar_checkpoint.cache_capacity_bytes =
            prepared->oscar_cache->bytes;
        prepared->logits = allocate(
            kVocab * sizeof(std::uint16_t),
            "allocate P2 target logits checkpoint");
        prepared->taps = allocate(
            kTapLayers.size() * impl_->prefill_capacity * kHidden *
                sizeof(std::uint16_t),
            "allocate P2 target taps checkpoint");
        prepared->embedding = allocate(
            impl_->prefill_capacity * kHidden * sizeof(std::uint16_t),
            "allocate P2 target embedding checkpoint");
        prepared->position_device = allocate(
            sizeof(int), "allocate P2 target position checkpoint");
        prepared->bytes = prepared->gdn_recurrent->bytes +
            prepared->gdn_conv->bytes + prepared->oscar_cache->bytes +
            prepared->logits->bytes + prepared->taps->bytes +
            prepared->embedding->bytes + prepared->position_device->bytes;
        require(prepared->bytes == required,
                "P2 target transaction allocation requirement mismatch");
        return prepared;
    };

    std::shared_ptr<Impl::Transaction> prepared;
    if (authority) {
        using Inventory = Exl3ResourceInventory;
        using Domain = Inventory::Domain;
        Inventory::Requirement requirement;
        requirement.configuration = 0x5032545843484B50;
        requirement.add(Domain::device, 1, required);
        requirement.add(Domain::host_metadata, 1,
                        transaction_owner_metadata_bytes());
        const auto quarantine_before = retirement_quarantine_witness();
        authority->allocate_startup_resources(
            requirement,
            [&](std::uint64_t configuration) {
                require(configuration == requirement.configuration,
                        "P2 target transaction reservation identity");
                if (startup_fault == 1)
                    throw std::runtime_error(
                        "injected P2 transaction preconstruction failure");
                prepared = construct();
                if (startup_fault == 5)
                    prepared->gdn_recurrent->cleanup_failure_for_test = true;
                if (startup_fault == 2)
                    throw std::runtime_error(
                        "injected P2 transaction precommit failure");
                Inventory actual;
                actual.add({prepared, 0, Domain::host_metadata,
                    bounded_shared_allocation_bytes<Impl::Transaction>() -
                        (startup_fault == 4 ? 1 : 0), {},
                    &attach_bounded_retirement_credit<Impl::Transaction,
                                                       const void>});
                const std::array<DeviceAllocation*, 7> children{
                    prepared->gdn_recurrent.get(), prepared->gdn_conv.get(),
                    prepared->oscar_cache.get(), prepared->logits.get(),
                    prepared->taps.get(), prepared->embedding.get(),
                    prepared->position_device.get()};
                for (unsigned i = 0; i < children.size(); ++i) {
                    auto* child = children[i];
                    std::shared_ptr<const void> owner(prepared, child);
                    actual.add({owner, 1 + 2 * i, Domain::device,
                        child->bytes - (startup_fault == 3 && i == 0 ? 1 : 0),
                        {}, nullptr, &DeviceAllocation::attach_device_credit});
                    actual.add({owner, 2 + 2 * i, Domain::host_metadata,
                        DeviceAllocation::owner_metadata_bytes(), {},
                        &DeviceAllocation::attach_metadata_credit});
                }
                return actual;
            },
            [&]() noexcept {
                prepared.reset();
                if(retirement_quarantine_witness()!=quarantine_before)
                    authority->seal_failed_startup_retirement();
            },
            [&]() noexcept {
                if(retirement_quarantine_witness()!=quarantine_before)
                    authority->seal_failed_startup_retirement();
            });
        const std::array<DeviceAllocation*, 7> children{
            prepared->gdn_recurrent.get(), prepared->gdn_conv.get(),
            prepared->oscar_cache.get(), prepared->logits.get(),
            prepared->taps.get(), prepared->embedding.get(),
            prepared->position_device.get()};
        for (auto* child : children)
            child->release_constructor_credits_after_commit();
    } else {
        prepared = construct();
    }
    impl_->persistent_bytes = Exl3LinearWorkspaceRequirements::append_owned_bytes(
        impl_->persistent_bytes, prepared->bytes);
    persistent_bytes_ = Exl3LinearWorkspaceRequirements::append_owned_bytes(
        persistent_bytes_, prepared->bytes);
    impl_->transaction = std::move(prepared);
}

void Exl3TextContext::begin_transaction(cudaStream_t stream) {
    impl_->join_repair(stream);
    require(impl_->transaction != nullptr, "P2 target transaction was not prepared");
    auto& transaction = *impl_->transaction;
    require(!transaction.rollback_required,
            "P2 target transaction requires a successful reset after failed rollback");
    require(!transaction.active, "P2 target transaction is already active");
    const bool host_kv = transaction.host_kv && impl_->host_kv.enabled &&
        impl_->oscar == nullptr;
    const bool device_kv=transaction.device_kv && !impl_->host_kv.enabled &&
        impl_->oscar == nullptr;
    require((host_kv || device_kv || (impl_->oscar != nullptr &&
                impl_->oscar->graph_class() == 0)) &&
                !impl_->graph_active && !impl_->graph_capture_active,
            "P2 target transaction begin requires its eager OSCAR or exact HostKV mode");
    require(position_ > 0 && impl_->last_rows > 0,
            "P2 target transaction requires a completed nonempty forward");
    require(impl_->tap_rows > 0 && impl_->tap_rows <= impl_->prefill_capacity &&
                impl_->embedding_rows > 0 && impl_->embedding_rows <= impl_->prefill_capacity &&
                impl_->last_rows <= impl_->prefill_capacity && impl_->tap_rows == impl_->embedding_rows &&
                impl_->tap_rows == impl_->last_rows,
            "P2 target transaction captured rows are invalid or inconsistent");
    impl_->validate_target_projection_execution(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status),
               "query P2 target transaction stream capture state");
    require(capture_status == cudaStreamCaptureStatusNone,
            "P2 target transaction begin is unavailable during external stream capture");
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->gdn_layers[layer])
            impl_->gdn_layers[layer]->validate_checkpoint_storage(
                transaction.gdn_checkpoints[layer]);
    }
    transaction.fresh_snapshot = false;
    transaction.prefix_available = false;
    transaction.rollback_required = false;
    transaction.attempted_rows = 0;
    for (auto& layer : impl_->full_layers)
        if (layer) layer->invalidate_retained_prefix();

    auto* tap_destination = static_cast<std::byte*>(transaction.taps->ptr);
    const std::size_t tap_bytes =
        static_cast<std::size_t>(impl_->tap_rows) * kHidden * sizeof(std::uint16_t);
    const std::size_t embedding_bytes =
        static_cast<std::size_t>(impl_->embedding_rows) * kHidden * sizeof(std::uint16_t);
    const auto submit_non_gdn_checkpoint_copies=[&] {
        cuda_check(cudaMemcpyAsync(transaction.logits->ptr, impl_->logits,
                                   transaction.logits->bytes,
                                   cudaMemcpyDeviceToDevice, stream),
                   "save P2 target logits checkpoint");
        cuda_check(cudaMemcpyAsync(transaction.position_device->ptr,
                                   impl_->position_device,sizeof(int),
                                   cudaMemcpyDeviceToDevice, stream),
                   "save P2 target device position checkpoint");
        for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap)
            cuda_check(cudaMemcpyAsync(
                tap_destination + tap * impl_->prefill_capacity * kHidden *
                    sizeof(std::uint16_t),
                impl_->taps[tap]->ptr,tap_bytes,cudaMemcpyDeviceToDevice,stream),
                "save P2 target hidden tap checkpoint");
        cuda_check(cudaMemcpyAsync(transaction.embedding->ptr,
                                   impl_->embedding_trace->ptr,embedding_bytes,
                                   cudaMemcpyDeviceToDevice,stream),
                   "save P2 target embedding checkpoint");
    };
    // The ordinary device-KV probe has B2..B8 continuation frontiers. Capture
    // the eight-row allocation extent once so the same graph can checkpoint
    // any such frontier; transaction metadata still records the valid rows.
    // HostKV keeps its existing exact one-row graph and copy extent.
    const bool device_checkpoint_graph_eligible=
        impl_->device_transaction_checkpoint_graph_enabled && device_kv &&
        impl_->tap_rows<=8 && impl_->prefill_capacity>=8;
    const bool checkpoint_graph_eligible=
        (impl_->host_kv_transaction_checkpoint_graph_enabled && host_kv &&
         impl_->tap_rows==1) || device_checkpoint_graph_eligible;
    if(checkpoint_graph_eligible) {
        const std::size_t graph_copy_rows=device_checkpoint_graph_eligible?
            8:static_cast<std::size_t>(impl_->tap_rows);
        const std::size_t graph_copy_bytes=
            graph_copy_rows*kHidden*sizeof(std::uint16_t);
        if(!transaction.checkpoint_graph_active) {
            impl_->bind_graph_device();
            const auto started=std::chrono::steady_clock::now();
            cudaStream_t capture_stream=nullptr;
            cuda_check(cudaStreamCreateWithFlags(
                    &capture_stream,cudaStreamNonBlocking),
                "create transaction checkpoint graph capture stream");
            try {
                transaction.checkpoint_graph_definition.capture(
                    capture_stream,[&] {
                    for(int layer=0;layer<kLayers;++layer)
                        if(impl_->gdn_layers[layer])
                            impl_->gdn_layers[layer]->record_checkpoint_copy_nodes(
                                transaction.gdn_checkpoints[layer],capture_stream);
                    cuda_check(cudaMemcpyAsync(transaction.logits->ptr,impl_->logits,
                            transaction.logits->bytes,cudaMemcpyDeviceToDevice,
                            capture_stream),
                        "capture P2 target logits checkpoint");
                    cuda_check(cudaMemcpyAsync(transaction.position_device->ptr,
                            impl_->position_device,sizeof(int),cudaMemcpyDeviceToDevice,
                            capture_stream),
                        "capture P2 target device position checkpoint");
                    for(std::size_t tap=0;tap<kTapLayers.size();++tap)
                        cuda_check(cudaMemcpyAsync(
                                tap_destination+tap*impl_->prefill_capacity*kHidden*
                                    sizeof(std::uint16_t),
                                impl_->taps[tap]->ptr,graph_copy_bytes,
                                cudaMemcpyDeviceToDevice,capture_stream),
                            "capture P2 target hidden tap checkpoint");
                    cuda_check(cudaMemcpyAsync(transaction.embedding->ptr,
                            impl_->embedding_trace->ptr,graph_copy_bytes,
                            cudaMemcpyDeviceToDevice,capture_stream),
                        "capture P2 target embedding checkpoint");
                    });
                transaction.checkpoint_graph_executable.instantiate(
                    transaction.checkpoint_graph_definition);
                transaction.checkpoint_graph_executable.upload(capture_stream);
                cuda_check(cudaStreamSynchronize(capture_stream),
                    "complete transaction checkpoint graph preparation");
            } catch(...) {
                (void)cudaStreamSynchronize(capture_stream);
                (void)cudaStreamDestroy(capture_stream);
                throw;
            }
            cuda_check(cudaStreamDestroy(capture_stream),
                "destroy transaction checkpoint graph capture stream");
            transaction.checkpoint_graph_stream=stream;
            transaction.checkpoint_graph_active=true;
            const double capture_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-started).count();
            if(device_checkpoint_graph_eligible) {
                ++impl_->device_transaction_checkpoint_graph_captures;
                impl_->device_transaction_checkpoint_graph_capture_ms+=capture_ms;
            } else {
                ++impl_->host_kv.transaction_checkpoint_graph_captures;
                impl_->host_kv.transaction_checkpoint_graph_capture_ms+=capture_ms;
            }
        }
        require(transaction.checkpoint_graph_stream==stream &&
                    transaction.checkpoint_graph_definition.ready() &&
                    transaction.checkpoint_graph_executable.ready(),
                "transaction checkpoint graph stream/handle mismatch");
        std::array<std::uint64_t,kLayers> generations{};
        for(int layer=0;layer<kLayers;++layer)
            if(impl_->gdn_layers[layer])
                generations[layer]=impl_->gdn_layers[layer]->
                    begin_checkpoint_graph_replay(
                        transaction.gdn_checkpoints[layer]);
        transaction.checkpoint_graph_executable.launch(stream);
        for(int layer=0;layer<kLayers;++layer)
            if(impl_->gdn_layers[layer])
                impl_->gdn_layers[layer]->publish_checkpoint_graph_replay(
                    transaction.gdn_checkpoints[layer],position_,
                    generations[layer],stream);
        if(device_checkpoint_graph_eligible)
            ++impl_->device_transaction_checkpoint_graph_replays;
        else
            ++impl_->host_kv.transaction_checkpoint_graph_replays;
    } else {
        for (int layer = 0; layer < kLayers; ++layer)
            if (impl_->gdn_layers[layer])
                impl_->gdn_layers[layer]->save_checkpoint(
                    transaction.gdn_checkpoints[layer],position_,stream);
        if (!host_kv && !device_kv)
            impl_->oscar->save_checkpoint(transaction.oscar_checkpoint, stream);
        submit_non_gdn_checkpoint_copies();
    }
    if (host_kv && impl_->host_kv_transaction_recurrent_trace_enabled) {
        for (int layer=0;layer<kLayers;++layer)
            if (impl_->gdn_layers[layer]) {
                ++impl_->host_kv.transaction_recurrent_trace_alias_layers;
                impl_->host_kv.transaction_recurrent_trace_copy_bytes_saved+=
                    impl_->gdn_layers[layer]->recurrent_state_bytes();
            }
    }
    transaction.position = position_;
    transaction.tap_rows = impl_->tap_rows;
    transaction.embedding_rows = impl_->embedding_rows;
    transaction.last_rows = impl_->last_rows;
    transaction.stream = stream;
    transaction.attempt_base_position = transaction.position;
    transaction.continuation_graph_rows = 0;
    transaction.continuation_graph_attempt = false;
    if (host_kv) {
        require(impl_->exact_prefix_position == position_ &&
                    !impl_->exact_prefix_pages.empty(),
                "P2 HostKV transaction requires a complete authoritative prefix");
        transaction.host_kv_page_count = impl_->exact_prefix_pages.size();
        transaction.host_kv_tail = impl_->exact_prefix_pages.back();
    } else {
        transaction.host_kv_page_count = 0;
        transaction.host_kv_tail.reset();
    }
    transaction.fresh_snapshot = true;
    transaction.active = true;
}

void Exl3TextContext::rollback_transaction(cudaStream_t stream) {
    impl_->join_repair(stream);
    require(impl_->transaction != nullptr && impl_->transaction->active,
            "P2 target transaction rollback has no active snapshot");
    auto& transaction = *impl_->transaction;
    for (int layer = 0; layer < kLayers; ++layer)
        if (impl_->gdn_layers[layer])
            impl_->gdn_layers[layer]->validate_saved_checkpoint(
                transaction.gdn_checkpoints[layer],transaction.position);
    transaction.fresh_snapshot = false;
    transaction.prefix_available = false;
    transaction.attempted_rows = 0;
    transaction.continuation_graph_rows = 0;
    transaction.continuation_graph_attempt = false;
    for (auto& layer : impl_->full_layers)
        if (layer) layer->invalidate_retained_prefix();
    const bool host_kv = transaction.host_kv && impl_->host_kv.enabled &&
        impl_->oscar == nullptr;
    const bool device_kv=transaction.device_kv && !impl_->host_kv.enabled &&
        impl_->oscar == nullptr;
    require((host_kv || device_kv || (impl_->oscar != nullptr &&
                impl_->oscar->graph_class() == 0)) &&
                !impl_->graph_active && !impl_->graph_capture_active,
            "P2 target transaction rollback requires its eager OSCAR or exact HostKV mode");
    impl_->validate_target_projection_execution(stream);
    transaction.rollback_required = true;
    if (impl_->continuation) impl_->continuation->rows = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->gdn_layers[layer]) {
            impl_->gdn_layers[layer]->restore_checkpoint(transaction.gdn_checkpoints[layer], stream);
        }
    }
    if (!host_kv && !device_kv)
        impl_->oscar->restore_checkpoint(transaction.oscar_checkpoint, stream);
    cuda_check(cudaMemcpyAsync(impl_->logits, transaction.logits->ptr, transaction.logits->bytes,
                               cudaMemcpyDeviceToDevice, stream),
               "restore P2 target logits checkpoint");
    cuda_check(cudaMemcpyAsync(impl_->position_device, transaction.position_device->ptr,
                               sizeof(int), cudaMemcpyDeviceToDevice, stream),
               "restore P2 target device position checkpoint");
    const auto* tap_source = static_cast<const std::byte*>(transaction.taps->ptr);
    const std::size_t tap_bytes =
        static_cast<std::size_t>(transaction.tap_rows) * kHidden * sizeof(std::uint16_t);
    for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) {
        cuda_check(cudaMemcpyAsync(impl_->taps[tap]->ptr,
                                   tap_source + tap * impl_->prefill_capacity * kHidden * sizeof(std::uint16_t),
                                   tap_bytes, cudaMemcpyDeviceToDevice, stream),
                   "restore P2 target hidden tap checkpoint");
    }
    const std::size_t embedding_bytes =
        static_cast<std::size_t>(transaction.embedding_rows) * kHidden * sizeof(std::uint16_t);
    cuda_check(cudaMemcpyAsync(impl_->embedding_trace->ptr, transaction.embedding->ptr,
                               embedding_bytes, cudaMemcpyDeviceToDevice, stream),
               "restore P2 target embedding checkpoint");
    position_ = transaction.position;
    impl_->tap_rows = transaction.tap_rows;
    impl_->embedding_rows = transaction.embedding_rows;
    impl_->last_rows = transaction.last_rows;
    if (host_kv) {
        require(transaction.host_kv_page_count > 0 && transaction.host_kv_tail &&
                    impl_->exact_prefix_pages.size() >= transaction.host_kv_page_count,
                "P2 HostKV rollback prefix snapshot is incomplete");
        impl_->exact_prefix_pages.resize(transaction.host_kv_page_count);
        impl_->exact_prefix_pages.back() = transaction.host_kv_tail;
        impl_->exact_prefix_position = transaction.position;
        impl_->host_kv_failed = false;
        transaction.host_kv_page_count = 0;
        transaction.host_kv_tail.reset();
    }
    impl_->tap_generation.fetch_add(1,std::memory_order_release);
    impl_->qkv_trace_valid = false;
    impl_->invalidate_native_mtp_hidden_capture();
    transaction.rollback_required = false;
    transaction.active = false;
}

void Exl3TextContext::commit_transaction() {
    require(impl_->transaction != nullptr && impl_->transaction->active,
            "P2 target transaction commit has no active snapshot");
    const auto& transaction=*impl_->transaction;
    const bool host_kv = transaction.host_kv && impl_->host_kv.enabled &&
        impl_->oscar == nullptr;
    const bool device_kv=transaction.device_kv && !impl_->host_kv.enabled &&
        impl_->oscar == nullptr;
    require((host_kv || device_kv || (impl_->oscar != nullptr &&
                impl_->oscar->graph_class() == 0)) &&
                !impl_->graph_active && !impl_->graph_capture_active,
            "P2 target transaction commit requires its eager OSCAR or exact HostKV mode");
    require(!impl_->transaction->rollback_required,
            "P2 target transaction requires rollback after failed prefix retention");
    if (impl_->transaction->continuation_graph_attempt) {
        require(impl_->transaction->continuation_graph_rows ==
                    impl_->transaction->attempted_rows,
                "B8 continuation graph commit row mismatch");
    }
    impl_->transaction->fresh_snapshot = false;
    impl_->transaction->prefix_available = false;
    impl_->transaction->rollback_required = false;
    impl_->transaction->attempted_rows = 0;
    impl_->transaction->continuation_graph_rows = 0;
    impl_->transaction->continuation_graph_attempt = false;
    impl_->transaction->host_kv_page_count = 0;
    impl_->transaction->host_kv_tail.reset();
    for (auto& layer : impl_->full_layers)
        if (layer) layer->invalidate_retained_prefix();
    impl_->transaction->active = false;
}

bool Exl3TextContext::transaction_prepared() const noexcept {
    return impl_->transaction != nullptr;
}

bool Exl3TextContext::transaction_active() const noexcept {
    return impl_->transaction != nullptr && impl_->transaction->active;
}

std::size_t Exl3TextContext::transaction_bytes() const noexcept {
    return impl_->transaction ? impl_->transaction->bytes : 0;
}

bool Exl3TextContext::target_projection_timing_enabled() const noexcept {
    return impl_->target_projection_timing_opt_in;
}

void Exl3TextContext::prepare_target_projection_timing(cudaStream_t stream) {
    require(impl_->target_projection_timing_opt_in,
            "target projection timing was not enabled at context construction");
    require(!impl_->graph_active && !impl_->graph_capture_active,
            "target projection timing setup is unavailable with an internal graph");
    require(!impl_->oscar || impl_->oscar->graph_class() == 0,
            "target projection timing setup requires eager OSCAR");
    require(impl_->target_projection_observer == nullptr,
            "target projection timing is incompatible with projection observation");
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status),
               "query target projection timing setup stream capture");
    require(capture_status == cudaStreamCaptureStatusNone,
            "target projection timing setup is unavailable during stream capture");
    if (impl_->target_projection_timing) return;
    impl_->invalidate_continuation_graphs(
        "projection timing changed captured workspace strategy");
    impl_->target_projection_timing = std::make_unique<Exl3TargetProjectionTiming>();
    auto* timing = impl_->target_projection_timing.get();
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->full_layers[layer])
            impl_->full_layers[layer]->set_projection_timing(timing, layer);
        if (impl_->gdn_layers[layer])
            impl_->gdn_layers[layer]->set_projection_timing(timing, layer);
    }
}

void Exl3TextContext::begin_target_projection_timing_round(int round,
                                                           cudaStream_t stream) {
    require(impl_->target_projection_timing != nullptr,
            "target projection timing was not prepared");
    require(!impl_->graph_active && !impl_->graph_capture_active,
            "target projection timing round is unavailable with an internal graph");
    require(!impl_->oscar || impl_->oscar->graph_class() == 0,
            "target projection timing round requires eager OSCAR");
    impl_->target_projection_timing->begin_round(round, stream);
}

void Exl3TextContext::set_target_projection_timing_phase(
    Exl3TargetProjectionPhase phase) {
    if (impl_->target_projection_timing)
        impl_->target_projection_timing->set_phase(phase);
}

std::vector<Exl3TargetProjectionTimingRecord>
Exl3TextContext::finish_target_projection_timing_round_after_synchronize() {
    require(impl_->target_projection_timing != nullptr,
            "target projection timing was not prepared");
    return impl_->target_projection_timing->finish_after_synchronize();
}

bool Exl3TextContext::attention_staging_uncertain() const noexcept {
    return impl_->attention_stages && impl_->attention_stages->uncertain();
}
void Exl3TextContext::set_attention_staging_enabled_for_test(bool enabled) {
    require(impl_->attention_stage_storage && !attention_staging_uncertain(),
        "attention staging toggle requires idle prepared owners");
    impl_->invalidate_continuation_graphs(
        "attention staging strategy changed captured cache layout");
    impl_->attention_staging_enabled=enabled;
}
void Exl3TextContext::fail_attention_staging_for_test(unsigned stage) {
    // 1..5 select K; 6..10 select the corresponding V boundary.
    require(stage>=1 && stage<=10 && impl_->attention_stage_storage &&
        !impl_->attention_staging_failure_for_test && !attention_staging_uncertain(),
        "attention staging failure requires idle prepared context");
    impl_->attention_staging_failure_for_test=stage;
}
void Exl3TextContext::set_attention_staging(std::shared_ptr<Exl3AttentionStageStorage> storage,
    std::shared_ptr<Exl3AttentionStage> stages,std::shared_ptr<Exl3AttentionStageHistory> history) {
    require(position_==0 && !impl_->graph_active && !impl_->graph_capture_active &&
        impl_->host_kv.enabled && !impl_->oscar_only && !impl_->device_prefix &&
        !impl_->attention_stage_storage && storage && stages && history && storage->capacity()==impl_->max_context,
        "attention staging requires pristine ordinary HostKV and matching reserved owners");
    impl_->invalidate_continuation_graphs(
        "attention staging owners changed captured cache layout");
    impl_->attention_stage_storage=std::move(storage);
    impl_->attention_stages=std::move(stages);
    impl_->attention_stage_history=std::move(history);
}

void Exl3TextContext::set_attention_registration_cache(std::shared_ptr<Exl3KVRegistrationCache> cache) {
    require(position_==0 && !impl_->graph_active && !impl_->graph_capture_active &&
        impl_->attention_stages && impl_->host_kv.enabled && impl_->host_kv_pinned_chunks &&
        !impl_->oscar_only && !impl_->attention_registration_cache && cache,
        "attention registration cache requires pristine prepared staging");
    impl_->invalidate_continuation_graphs(
        "attention registration changed captured physical membership");
    impl_->attention_registration_cache=std::move(cache);
}

void Exl3TextContext::set_shared_page_attention(SharedPageAttention acquire,SharedPageAttentionCompletion complete) {
    require(position_==0 && !impl_->graph_active && !impl_->graph_capture_active &&
        impl_->host_kv.enabled && impl_->host_kv_pinned_chunks && !impl_->oscar_only &&
        !impl_->device_prefix && bool(acquire) && bool(complete),
        "shared page attention requires pristine ordinary pinned context without private prefix");
    impl_->invalidate_continuation_graphs(
        "shared page attention changed captured segmented view");
    impl_->shared_page_attention=std::move(acquire);
    impl_->shared_page_attention_completion=std::move(complete);
}

void Exl3TextContext::set_shared_page_copier(SharedPageCopier copier) {
    require(position_==0 && !impl_->graph_active && !impl_->graph_capture_active &&
        impl_->host_kv.enabled && impl_->host_kv_pinned_chunks && !impl_->oscar_only && bool(copier),
        "shared page copier requires pristine ordinary pinned context");
    impl_->invalidate_continuation_graphs(
        "shared page copier changed captured physical membership");
    impl_->shared_page_copier=std::move(copier);
}

void Exl3TextContext::set_registered_kv_uploader(RegisteredKVUploader uploader,RegisteredKVCompletion completion) {
    require(position_==0 && !impl_->graph_active && !impl_->graph_capture_active &&
        impl_->host_kv.enabled && impl_->host_kv_pinned_chunks && !impl_->oscar_only,
        "registered KV uploader requires pristine ordinary pinned context");
    require(bool(uploader) && bool(completion),"registered KV callbacks must be complete");
    impl_->invalidate_continuation_graphs(
        "registered KV uploader changed captured publication strategy");
    impl_->registered_kv_uploader=std::move(uploader);
    impl_->registered_kv_completion=std::move(completion);
}

void Exl3TextContext::set_target_q_executor(Exl3TargetQExecutor executor,bool kv,bool output,bool gateup,bool down,bool head) {
    require(!impl_->graph_active && !impl_->graph_capture_active && !impl_->oscar &&
        (!impl_->transaction || !impl_->transaction->active) && impl_->host_kv.enabled,
        "target Q executor requires idle ordinary host-KV context");
    impl_->invalidate_continuation_graphs(
        "target projection executor changed captured physical membership");
    impl_->shared_head_executor={};
    if(head && executor)impl_->shared_head_executor=[this,executor](const Exl3TargetQContinuation& projection) {
        auto request=projection;request.position=position_;return executor(request);
    };
    for(int layer=0;layer<64;++layer) {
        if(impl_->full_layers[layer])impl_->full_layers[layer]->set_target_q_executor(executor,layer,kv,output,gateup,down);
        if(impl_->gdn_layers[layer]) {
            Exl3TargetQExecutor positioned;
            if((gateup || down) && executor)positioned=[this,executor](const Exl3TargetQContinuation& projection) {
                auto request=projection;request.position=position_;
                return executor(request);
            };
            impl_->gdn_layers[layer]->set_shared_gateup_executor(std::move(positioned),layer,gateup,down);
        }
    }
}

bool Exl3TextContext::prepare_device_prefix_reserved(Exl3VeriCacheServingCoordinator& authority,
    int capacity,bool segmented,unsigned startup_fault_for_test) {
    require(!impl_->host_kv_failed,"reserved prefix requires intact HostKV lineage");
    const auto bytes=Exl3DevicePrefixCache::allocation_bytes_required(capacity);
    require(position_==0 && !impl_->device_prefix && impl_->host_kv.enabled &&
        !impl_->graph_active && !impl_->graph_capture_active &&
        (!impl_->continuation || !impl_->continuation->graph_active) &&
        impl_->host_kv_pinned_chunks && !impl_->oscar_only && !impl_->continuation_graph_b8_enabled &&
        max_context()>=capacity && (!impl_->transaction || !impl_->transaction->active),
        "reserved prefix requires pristine ordinary eager pinned HostKV context");
    require(!impl_->device_prefix_forward_publish || (capacity==16384 && !segmented),
        "reserved forward-published prefix requires unsegmented 16K storage");
    if(impl_->continuation)for(const auto& graph:impl_->continuation->additional_graphs)
        require(!graph.active,"reserved prefix cannot replace active continuation graph storage");
    const auto internal=Exl3LinearWorkspaceRequirements::append_owned_bytes(impl_->persistent_bytes,bytes);
    const auto published=Exl3LinearWorkspaceRequirements::append_owned_bytes(persistent_bytes_,bytes);
    auto prepared=Exl3DevicePrefixCache::create_reserved(authority,capacity,startup_fault_for_test);
    if(!prepared){++impl_->host_kv.device_prefix_fallbacks;return false;}
    impl_->invalidate_continuation_graphs(
        "device prefix changed captured cache layout");
    impl_->device_prefix=std::move(prepared);
    impl_->device_prefix_metadata_external=true;
    impl_->segmented_device_prefix=segmented;
    impl_->persistent_bytes=internal;persistent_bytes_=published;
    impl_->host_kv.device_prefix_bytes=bytes;
    return true;
}

std::shared_ptr<const void> Exl3TextContext::share_device_prefix_with(Exl3TextContext& other,
    Exl3ResourceInventory& shared_resources) {
    require(!impl_->device_prefix_metadata_external && !other.impl_->device_prefix_metadata_external,
        "legacy prefix sharing cannot replace reserved owners");
    require(this!=&other && position_==0 && other.position_==0 && impl_->model==other.impl_->model &&
        !impl_->host_kv_failed && !other.impl_->host_kv_failed &&
        !impl_->graph_active && !other.impl_->graph_active &&
        !impl_->graph_capture_active && !other.impl_->graph_capture_active &&
        (!impl_->continuation || !impl_->continuation->graph_active) &&
        (!other.impl_->continuation || !other.impl_->continuation->graph_active) &&
        impl_->host_kv.enabled && other.impl_->host_kv.enabled &&
        impl_->host_kv_pinned_chunks && other.impl_->host_kv_pinned_chunks &&
        !impl_->oscar && !other.impl_->oscar && !impl_->device_prefix_shared && !other.impl_->device_prefix_shared &&
        (!impl_->transaction || !impl_->transaction->active) &&
        (!other.impl_->transaction || !other.impl_->transaction->active) &&
        impl_->device_prefix && other.impl_->device_prefix && impl_->device_prefix->admitted() &&
        impl_->device_prefix->capacity_tokens()==other.impl_->device_prefix->capacity_tokens(),
        "shared represented prefix requires two pristine compatible owners");
    for(const auto* context:{this,&other})if(context->impl_->continuation)
        for(const auto& graph:context->impl_->continuation->additional_graphs)
            require(!graph.active,"legacy prefix sharing cannot replace active continuation graph storage");
    require(impl_->device_prefix->reusable() && other.impl_->device_prefix->reusable(),
        "legacy prefix sharing requires reusable caches");
    const auto first_bytes=impl_->device_prefix->allocation_bytes();
    const auto other_bytes=other.impl_->device_prefix->allocation_bytes();
    require(first_bytes<=impl_->persistent_bytes && other_bytes<=other.impl_->persistent_bytes &&
        first_bytes<=persistent_bytes_ && other_bytes<=other.persistent_bytes_,
        "shared represented prefix accounting extent");
    auto next=shared_resources;
    next.append(Exl3DevicePrefixCache::resources(impl_->device_prefix));
    require(other.impl_->device_prefix.use_count()==1,
        "legacy prefix replacement requires sole destination cache ownership");
    impl_->invalidate_continuation_graphs(
        "shared prefix changed captured physical membership");
    other.impl_->invalidate_continuation_graphs(
        "shared prefix changed captured physical membership");
    require(other.impl_->device_prefix->retire_pristine(),
        "legacy prefix destination retirement failed");
    shared_resources=std::move(next);
    impl_->persistent_bytes-=first_bytes;other.impl_->persistent_bytes-=other_bytes;
    persistent_bytes_-=first_bytes;other.persistent_bytes_-=other_bytes;
    other.impl_->device_prefix=impl_->device_prefix;
    impl_->device_prefix_shared=true;other.impl_->device_prefix_shared=true;
    other.impl_->host_kv.device_prefix_bytes=first_bytes;
    return impl_->device_prefix;
}
std::shared_ptr<const void> Exl3TextContext::share_device_prefix_with_empty(Exl3TextContext& other) {
    require(!impl_->host_kv_failed && !other.impl_->host_kv_failed,
        "shared reserved prefix requires intact HostKV lineages");
    require(this!=&other && position_==0 && other.position_==0 && impl_->model==other.impl_->model &&
        !impl_->graph_active && !other.impl_->graph_active &&
        !impl_->graph_capture_active && !other.impl_->graph_capture_active &&
        (!impl_->continuation || !impl_->continuation->graph_active) &&
        (!other.impl_->continuation || !other.impl_->continuation->graph_active) &&
        impl_->host_kv.enabled && other.impl_->host_kv.enabled &&
        impl_->host_kv_pinned_chunks && other.impl_->host_kv_pinned_chunks &&
        !impl_->oscar_only && !other.impl_->oscar_only &&
        !impl_->continuation_graph_b8_enabled && !other.impl_->continuation_graph_b8_enabled &&
        !impl_->device_prefix_shared && !other.impl_->device_prefix_shared &&
        (!impl_->transaction || !impl_->transaction->active) &&
        (!other.impl_->transaction || !other.impl_->transaction->active) &&
        impl_->device_prefix && impl_->device_prefix->reusable() && !other.impl_->device_prefix &&
        other.max_context()>=impl_->device_prefix->capacity_tokens(),
        "shared reserved prefix requires pristine compatible empty peer");
    for(const auto* context:{this,&other})if(context->impl_->continuation)
        for(const auto& graph:context->impl_->continuation->additional_graphs)
            require(!graph.active,"shared reserved prefix cannot replace active continuation graph storage");
    const auto bytes=impl_->device_prefix->allocation_bytes();
    require(bytes<=impl_->persistent_bytes && bytes<=persistent_bytes_,"shared reserved prefix accounting extent");
    impl_->invalidate_continuation_graphs(
        "shared reserved prefix changed captured physical membership");
    other.impl_->invalidate_continuation_graphs(
        "shared reserved prefix changed captured physical membership");
    // The independent startup inventory retains the one allocation. Neither
    // private lane reports these shared bytes and no credited owner is replaced.
    other.impl_->device_prefix=impl_->device_prefix;
    other.impl_->segmented_device_prefix=impl_->segmented_device_prefix;
    impl_->persistent_bytes-=bytes;persistent_bytes_-=bytes;
    impl_->device_prefix_shared=true;other.impl_->device_prefix_shared=true;
    other.impl_->host_kv.device_prefix_bytes=bytes;
    return impl_->device_prefix;
}
void Exl3TextContext::retire_device_prefix_after_drain() noexcept {
    if(impl_->device_prefix)impl_->device_prefix->retire_after_device_drain();
}
std::weak_ptr<const void> Exl3TextContext::device_prefix_owner_for_test() const noexcept {
    return impl_->device_prefix;
}
void Exl3TextContext::fail_device_prefix_cleanup_for_test() {
    require(impl_->device_prefix!=nullptr,"prefix cleanup injection requires cache owner");
    impl_->device_prefix->fail_cleanup_for_test();
}

void Exl3TextContext::set_target_projection_observer_for_test(
    Exl3TargetProjectionObserver observer, void* user, cudaStream_t stream,
    Exl3TargetProjectionObserverSelection selection) {
    if (observer != nullptr) {
        require(!impl_->graph_active && !impl_->graph_capture_active &&
                    (!impl_->oscar || impl_->oscar->graph_class() == 0),
                "target projection observer requires eager execution");
        require(!impl_->target_projection_timing,
                "target projection observer is incompatible with projection timing");
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture_status),
                   "query target projection observer stream capture");
        require(capture_status == cudaStreamCaptureStatusNone,
                "target projection observer is unavailable during stream capture");
    }
    impl_->invalidate_continuation_graphs(
        "projection observer changed captured callback strategy");
    impl_->target_projection_observer = observer;
    impl_->target_projection_observer_user = observer ? user : nullptr;
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->full_layers[layer])
            impl_->full_layers[layer]->set_projection_observer(
                observer, impl_->target_projection_observer_user, layer,
                selection);
        if (impl_->gdn_layers[layer])
            impl_->gdn_layers[layer]->set_projection_observer(
                observer, impl_->target_projection_observer_user, layer,
                selection);
    }
}

void Exl3TextContext::set_layer_observer_for_test(
    Exl3LayerObserver observer, void* user, cudaStream_t stream) {
    if (observer) {
        require(!impl_->coalesce_attention_input_mlp && impl_->host_kv.enabled && !impl_->graph_active &&
                !impl_->graph_capture_active && !impl_->transaction,
                "layer observer requires ordinary eager exact-host context");
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &status), "query layer observer capture");
        require(status == cudaStreamCaptureStatusNone, "layer observer rejects capture");
    }
    impl_->invalidate_continuation_graphs(
        "layer observer changed captured callback strategy");
    impl_->layer_observer = observer;
    impl_->layer_observer_user = observer ? user : nullptr;
}

void Exl3TextContext::prefill(std::span<const std::int64_t> token_ids, cudaStream_t stream) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    require(!impl_->fast_prefill_failed,"failed layer-major prefill requires context reset");
    require(!impl_->host_kv_failed,"HostKV prefill requires intact transfer lineage");
    require(!impl_->deferred_reconstruction_bytes,"reconstruction startup reservation not installed");
    require(!impl_->transaction || !impl_->transaction->rollback_required,
            "P2 target transaction requires rollback after failed prefix retention");
    require(!token_ids.empty() && token_ids.size() <= static_cast<std::size_t>(impl_->max_context),
            "E4A prefill token count is outside context capacity");
    require(position_ == 0, "E4A prefill requires a reset context");
    require(token_ids.size() <= 16, "initial prefill supports at most16 rows; append wider suffix explicitly");
    impl_->validate_target_projection_execution(stream);
    impl_->retain_host_kv_forward_stream(stream);
    Impl::HostKVForwardFailureScope failure_scope{*impl_};
    if (impl_->continuation) impl_->continuation->rows = 0;
    cuda_check(cudaMemcpyAsync(impl_->token_ids, token_ids.data(), token_ids.size_bytes(),
                               cudaMemcpyHostToDevice, stream), "upload E4A prefill token IDs");
    cuda_check(cudaMemsetAsync(impl_->position_device, 0, sizeof(int), stream),
               "set E4B2 prefill position parameter");
    if(impl_->host_kv_prefill_fault_for_test==1) {
        impl_->host_kv_prefill_fault_for_test=0;
        throw std::runtime_error("injected HostKV pre-layer prefill failure");
    }
    impl_->process_rows(impl_->token_ids, static_cast<int>(token_ids.size()), 0, stream,
                        nullptr, true, false, true);
    if(impl_->host_kv_prefill_fault_for_test==2) {
        impl_->host_kv_prefill_fault_for_test=0;
        throw std::runtime_error("injected HostKV post-forward prefill failure");
    }
    position_ = Exl3NativeContextExtent::checked_input_end(token_ids.size(),impl_->max_context);
}

void Exl3TextContext::decode(std::int64_t token_id, cudaStream_t stream) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    require(!impl_->fast_prefill_failed,"failed layer-major prefill requires context reset");
    require(!impl_->host_kv_failed,"HostKV decode requires intact transfer lineage");
    require(!impl_->deferred_reconstruction_bytes,"reconstruction startup reservation not installed");
    require(!impl_->transaction || !impl_->transaction->rollback_required,
            "P2 target transaction requires rollback after failed prefix retention");
    require(position_ < impl_->max_context, "E4A context capacity exhausted");
    impl_->validate_target_projection_execution(stream);
    impl_->retain_host_kv_forward_stream(stream);
    Impl::HostKVForwardFailureScope failure_scope{*impl_};
    if (impl_->continuation) impl_->continuation->rows = 0;
    cuda_check(cudaMemcpyAsync(impl_->token_ids, &token_id, sizeof(token_id),
                               cudaMemcpyHostToDevice, stream), "upload E4A decode token ID");
    cuda_check(cudaMemcpyAsync(impl_->position_device, &position_, sizeof(position_),
                               cudaMemcpyHostToDevice, stream), "upload E4B2 decode position");
    ++last_decode_h2d_;
    impl_->process_rows(impl_->token_ids, 1, position_, stream,
                        nullptr, true, false, true);
    ++position_;
}

void Exl3TextContext::append_prefill(std::span<const std::int64_t> token_ids,
                                    cudaStream_t stream) {
    append_prefill_impl(token_ids, stream, false);
}

void Exl3TextContext::append_prefill_wide(std::span<const std::int64_t> token_ids,
                                        cudaStream_t stream) {
    require(impl_->wide_prefill_enabled, "wide prefill requires construction-latched opt-in");
    append_prefill_impl(token_ids, stream, true);
}

void Exl3TextContext::append_prefill_impl(std::span<const std::int64_t> token_ids,
                                        cudaStream_t stream, bool wide, bool exact) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    require(!impl_->fast_prefill_failed,"failed layer-major prefill requires context reset");
    require(!impl_->host_kv_failed,"HostKV append requires intact transfer lineage");
    require(!impl_->deferred_reconstruction_bytes,"reconstruction startup reservation not installed");
    require(!impl_->transaction ||
                (!impl_->transaction->active && !impl_->transaction->rollback_required),
            "chunked prefill requires no active or failed transaction");
    require(token_ids.size() >= 1 && token_ids.size() <= (wide ? static_cast<std::size_t>(impl_->prefill_capacity) : 8u),
            wide ? "wide prefill exceeds latched row capacity" : "chunked prefill accepts 1..8 rows");
    require(std::all_of(token_ids.begin(), token_ids.end(), [](std::int64_t token) {
                return token >= 0 && token < kVocab;
            }), "chunked prefill token outside vocabulary");
    const int rows = static_cast<int>(token_ids.size());
    require(position_ > 0 && impl_->last_rows > 0 &&
                position_ <= impl_->max_context - rows,
            "chunked prefill requires a nonempty prefix and available capacity");
    // Ordinary FP16 device-KV is a third eager identity alongside exact
    // HostKV and OSCAR.  It owns the same wide-row process_rows contract but
    // does not have a HostKV transaction or OSCAR graph class to identify it.
    // Keep the admission explicit: this is a separate backend route and must
    // never silently change an exact-host or OSCAR context.
    const bool ordinary_device_kv = !exact && !impl_->host_kv.enabled && !impl_->oscar;
    require(!impl_->graph_active && !impl_->graph_capture_active &&
                (exact ? (!impl_->oscar && impl_->host_kv.enabled) :
                    (ordinary_device_kv ||
                     (impl_->oscar && impl_->oscar->graph_class() == 0))),
            "chunked prefill requires its declared eager cache identity");
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status),
               "query chunked prefill stream capture");
    require(capture_status == cudaStreamCaptureStatusNone,
            "chunked prefill rejects external graph capture");
    impl_->validate_target_projection_execution(stream);

    // All admission is complete before changing output metadata or enqueuing.
    impl_->retain_host_kv_forward_stream(stream);
    Impl::HostKVForwardFailureScope failure_scope{*impl_};
    if (impl_->continuation) impl_->continuation->rows = 0;
    const int base_position = position_;
    cuda_check(cudaMemcpyAsync(impl_->token_ids, token_ids.data(), token_ids.size_bytes(),
                               cudaMemcpyHostToDevice, stream), "upload chunked prefill IDs");
    cuda_check(cudaMemcpyAsync(impl_->position_device, &base_position, sizeof(base_position),
                               cudaMemcpyHostToDevice, stream), "upload chunked prefill position");
    impl_->process_rows(impl_->token_ids, rows, base_position, stream,
                        nullptr, true, rows > 1, true, true, wide);
    const int final_position = base_position + rows - 1;
    cuda_check(cudaMemcpyAsync(impl_->position_device, &final_position, sizeof(final_position),
                               cudaMemcpyHostToDevice, stream), "publish chunked prefill position");
    position_ += rows;
    ++last_decode_h2d_;
}

Exl3TextContext::DeviceTransactionCheckpointGraphStats
Exl3TextContext::device_transaction_checkpoint_graph_stats() const noexcept {
    return {impl_->device_transaction_checkpoint_graph_captures,
        impl_->device_transaction_checkpoint_graph_replays,
        impl_->device_transaction_checkpoint_graph_capture_ms};
}

bool Exl3TextContext::layer_major_from_zero() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_EXL3_LAYER_MAJOR_FROM_ZERO");
        if (!value || std::strcmp(value, "1") == 0) return true;
        if (std::strcmp(value, "0") == 0) return false;
        throw std::invalid_argument("NINFER_EXL3_LAYER_MAJOR_FROM_ZERO must be 0 or 1");
    }();
    return enabled;
}

void Exl3TextContext::append_prefill_layer_major(
    std::span<const std::int64_t> token_ids, cudaStream_t stream,
    const RetainedTapTail* retained_taps) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    require(!impl_->fast_prefill_failed,"failed layer-major prefill requires context reset");
    require(impl_->fast_same_weights_fp16kv_prefill_enabled &&
            impl_->numeric_prefill_projection_workspace &&
            impl_->prefill_capacity == 1024 && !impl_->host_kv.enabled &&
            !impl_->oscar && !impl_->oscar_only && !impl_->device_prefix &&
            !impl_->graph_active && !impl_->graph_capture_active &&
            !impl_->layer_observer && !impl_->target_projection_timing &&
            !impl_->target_projection_observer &&
            !impl_->native_mtp_hidden_capture_active() &&
            (!impl_->transaction ||
             (!impl_->transaction->active && !impl_->transaction->rollback_required)),
            "layer-major prefill requires an idle Fast90 ordinary device-KV context");
    const bool fresh_prompt = position_ == 0 && layer_major_from_zero() &&
        !impl_->host_kv_failed && !impl_->deferred_reconstruction_bytes;
    require((fresh_prompt || (position_ > 0 && impl_->last_rows > 0)) &&
            token_ids.size() > 1024 &&
            token_ids.size() <= static_cast<std::size_t>(impl_->max_context-position_) &&
            std::all_of(token_ids.begin(),token_ids.end(),[](std::int64_t token) {
                return token >= 0 && token < kVocab;
            }), "layer-major prefill suffix extent and token contract");
    if (retained_taps) {
        require(impl_->capture_taps && retained_taps->device &&
                retained_taps->rows >= 1 && retained_taps->rows <= 2063 &&
                retained_taps->first_abs >= position_ &&
                static_cast<long long>(retained_taps->first_abs) + retained_taps->rows ==
                    static_cast<long long>(position_) + token_ids.size() &&
                retained_taps->bytes == static_cast<std::size_t>(retained_taps->rows) *
                    5 * kHidden * sizeof(std::uint16_t),
                "layer-major retained tap tail extent");
        cudaPointerAttributes attributes{};
        cuda_check(cudaPointerGetAttributes(&attributes, retained_taps->device),
                   "layer-major retained tap arena attributes");
        int device = -1;
        cuda_check(cudaGetDevice(&device), "layer-major retained tap device");
        require(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
                "layer-major retained tap arena must be local device memory");
    }
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream,&capture_status),
               "query layer-major prefill stream capture");
    require(capture_status == cudaStreamCaptureStatusNone,
            "layer-major prefill rejects external graph capture");
    impl_->validate_target_projection_execution(stream);
    impl_->retain_host_kv_forward_stream(stream);
    std::size_t free_bytes=0,total_bytes=0;
    cuda_check(cudaMemGetInfo(&free_bytes,&total_bytes),
               "budget layer-major projection cache");
    // The guarded device transaction has already retained its physical target
    // and draft owners. Keep a smaller, still explicit free-space reserve for
    // its bounded projection reuse; ordinary routes retain their 2 GiB floor.
    const std::size_t reserve_bytes=fast_device_kv_transaction_enabled()?
        1536ull*1024*1024:2ull*1024*1024*1024;
    constexpr std::size_t cache_limit=1536ull*1024*1024;
    constexpr std::size_t minimum_cache=256ull*1024*1024;
    auto& workspace=*impl_->numeric_prefill_projection_workspace;
    const auto existing_cache=workspace.stats().cached_weight_capacity_bytes;
    if(free_bytes<reserve_bytes ||
       existing_cache>cache_limit ||
       existing_cache+free_bytes-reserve_bytes<minimum_cache)
        throw std::runtime_error(
            "layer-major projection cache lacks bounded device headroom: free_mib="+
            std::to_string(free_bytes/(1024*1024))+
            " retained_cache_mib="+
            std::to_string(existing_cache/(1024*1024))+
            " free_reserve_mib="+
            std::to_string(reserve_bytes/(1024*1024)));
    const std::size_t cache_budget=std::min(cache_limit,
        existing_cache+free_bytes-reserve_bytes);
    require(workspace.stats().cached_weight_capacity_bytes<=cache_budget,
            "retained layer-major cache exceeds current device budget");
    const char* injected_layer=std::getenv("NINFER_EXL3_TEST_LAYER_MAJOR_FAIL_AFTER_LAYER");
    const int fail_after_layer=injected_layer ? std::atoi(injected_layer) : -1;
    require(!injected_layer ||
            (fail_after_layer>=0 && fail_after_layer<kLayers &&
             std::to_string(fail_after_layer)==injected_layer),
            "layer-major test fault index must be a model layer");
    struct LayerProfileEvents {
        std::vector<cudaEvent_t> events;
        ~LayerProfileEvents() {
            for (auto event : events) if (event) cudaEventDestroy(event);
        }
    } layer_profile;
    const char* profile_option=std::getenv("NINFER_EXL3_TEST_LAYER_MAJOR_PROFILE");
    if (profile_option && std::strcmp(profile_option,"1")==0) {
        layer_profile.events.resize(kLayers+1,nullptr);
        for (auto& event : layer_profile.events)
            cuda_check(cudaEventCreate(&event),"create layer-major profile event");
    }
    // From this point, a failure may leave different layers at different
    // frontiers. The context is invalid until reset, including if a CUDA
    // submission fails after some of the layer work was enqueued.
    try {
    if (impl_->continuation) impl_->continuation->rows = 0;

    const int base_position = position_;
    const int total = static_cast<int>(token_ids.size());
    // Causal chunk partition: 1024-row chunks with the remainder last. A
    // remainder below 256 rows is merged with the preceding chunk and split
    // evenly, so no chunk falls to the small-M decode route or below the bulk
    // admission width (a fresh prompt at 1024k+r would otherwise end in r rows).
    std::vector<int> chunk_offsets;
    std::vector<int> chunk_sizes;
    {
        const int full = (total+1023)/1024;
        for (int chunk=0;chunk<full;++chunk) {
            chunk_offsets.push_back(chunk*1024);
            chunk_sizes.push_back(std::min(1024,total-chunk*1024));
        }
        if (full>1 && chunk_sizes.back()<256) {
            const int merged=1024+chunk_sizes.back();
            chunk_sizes[full-2]=merged/2;
            chunk_sizes[full-1]=merged-merged/2;
            chunk_offsets[full-1]=chunk_offsets[full-2]+chunk_sizes[full-2];
        }
    }
    const int chunks = static_cast<int>(chunk_offsets.size());
    const int final_offset = chunk_offsets.back();
    const int final_rows = chunk_sizes.back();
    cuda_check(cudaMemcpyAsync(impl_->token_ids,token_ids.data(),
                               token_ids.size_bytes(),cudaMemcpyHostToDevice,stream),
               "upload layer-major prefill IDs");
    for (int offset=0;offset<total;offset+=1024) {
        const int rows=std::min(1024,total-offset);
        embedding_lookup_kernel<<<(rows*kHidden+255)/256,256,0,stream>>>(
            impl_->token_ids+offset,impl_->model->embedding,
            impl_->hidden_a+static_cast<std::size_t>(offset)*kHidden,rows);
        cuda_check(cudaGetLastError(),"embed layer-major prefill rows");
    }
    if (impl_->capture_taps) {
        cuda_check(cudaMemcpyAsync(impl_->embedding_trace->ptr,
            impl_->hidden_a+static_cast<std::size_t>(final_offset)*kHidden,
            static_cast<std::size_t>(final_rows)*kHidden*sizeof(std::uint16_t),
            cudaMemcpyDeviceToDevice,stream),
            "capture final layer-major embedding rows");
        impl_->embedding_rows=final_rows;
        impl_->tap_rows=final_rows;
    }

    const auto before=workspace.stats();
    std::uint64_t gdn_bulk_calls=0,gdn_bulk_rows=0;
    std::uint64_t gdn_bulk_mlp_blocks=0,gdn_bulk_mlp_rows=0;
    if (!layer_profile.events.empty())
        cuda_check(cudaEventRecord(layer_profile.events[0],stream),
                   "record layer-major profile start");
    for (int layer=0;layer<kLayers;++layer) {
        workspace.begin_layer_reuse(cache_budget,
            base_position+total>=8192 ||
            impl_->gdn_bulk_mlp_short_k5_enabled);
        workspace.set_prefill_layer(layer);
        const auto execute_chunk=[&](int offset,int rows,
                const Exl3GdnLayer::BulkPrefillBuffers* prepared) {
                const int logical_position=base_position+offset;
                cuda_check(cudaMemcpyAsync(impl_->position_device,&logical_position,
                    sizeof(logical_position),cudaMemcpyHostToDevice,stream),
                    "publish layer-major causal position");
                auto* input=(layer%2==0 ? impl_->hidden_a : impl_->hidden_b)+
                    static_cast<std::size_t>(offset)*kHidden;
                auto* output=(layer%2==0 ? impl_->hidden_b : impl_->hidden_a)+
                    static_cast<std::size_t>(offset)*kHidden;
                impl_->process_rows(nullptr,rows,logical_position,stream,
                    nullptr,false,rows>1,true,true,true,nullptr,
                    layer,layer+1,input,output,true,prepared);
        };
        const bool bulk_mlp=impl_->gdn_bulk_mlp_enabled &&
            impl_->gdn_layers[layer];
        const bool bulk=impl_->gdn_bulk_prefill_enabled &&
            impl_->gdn_layers[layer] &&
            impl_->gdn_layers[layer]->supports_bulk_prefill();
        // Bulk blocks group whole consecutive chunks up to the arena capacity.
        const auto for_each_block=[&](int capacity,const auto& body) {
            for (int first=0;first<chunks;) {
                int last=first,count=0;
                while (last<chunks && count+chunk_sizes[last]<=capacity)
                    count+=chunk_sizes[last++];
                require(last>first,"layer-major chunk exceeds bulk capacity");
                body(first,last,chunk_offsets[first],count);
                first=last;
            }
        };
        if (bulk_mlp) {
            const int batch_rows=impl_->gdn_bulk_capacity;
            for_each_block(batch_rows,[&](int first_chunk,int end_chunk,int block,int count) {
                if(count<256 ||
                   !impl_->gdn_layers[layer]->supports_bulk_mlp(count)) {
                    for(int chunk=first_chunk;chunk<end_chunk;++chunk)
                        execute_chunk(chunk_offsets[chunk],chunk_sizes[chunk],nullptr);
                    return;
                }
                require(count<=impl_->gdn_bulk_capacity,
                        "GDN bulk MLP arena capacity");
                auto* input_base=(layer%2==0?impl_->hidden_a:impl_->hidden_b)+
                    static_cast<std::size_t>(block)*kHidden;
                auto* output_base=(layer%2==0?impl_->hidden_b:impl_->hidden_a)+
                    static_cast<std::size_t>(block)*kHidden;
                for(int chunk=first_chunk;chunk<end_chunk;++chunk) {
                    const int relative=chunk_offsets[chunk]-block;
                    const int rows=chunk_sizes[chunk];
                    const int logical_position=base_position+block+relative;
                    cuda_check(cudaMemcpyAsync(impl_->position_device,
                        &logical_position,sizeof(logical_position),
                        cudaMemcpyHostToDevice,stream),
                        "publish GDN bulk MLP causal position");
                    Exl3GdnLayer::DeferredMlpBuffers slice{
                        impl_->gdn_mlp_post+
                            static_cast<std::size_t>(relative)*kHidden,
                        impl_->gdn_mlp_input+
                            static_cast<std::size_t>(relative)*kHidden,rows};
                    impl_->gdn_layers[layer]->forward_before_bulk_mlp(
                        input_base+static_cast<std::size_t>(relative)*kHidden,
                        slice,stream);
                }
                Exl3GdnLayer::DeferredMlpBuffers ready{
                    impl_->gdn_mlp_post,impl_->gdn_mlp_input,count};
                impl_->gdn_layers[layer]->finish_bulk_mlp(ready,
                    impl_->gdn_mlp_gate,impl_->gdn_mlp_up,
                    impl_->gdn_mlp_gate,output_base,
                    output_base,stream,block+count==total);
                ++gdn_bulk_mlp_blocks;
                gdn_bulk_mlp_rows+=static_cast<std::uint64_t>(count);
                const int tap=impl_->tap_index(layer);
                if(impl_->capture_taps && tap>=0 && block+count==total)
                    cuda_check(cudaMemcpyAsync(impl_->taps[tap]->ptr,
                        output_base+static_cast<std::size_t>(count-final_rows)*kHidden,
                        static_cast<std::size_t>(final_rows)*kHidden*
                            sizeof(std::uint16_t),cudaMemcpyDeviceToDevice,stream),
                        "capture final GDN bulk MLP hidden tap");
            });
        } else if (bulk) {
            constexpr int bulk_rows=4096;
            for_each_block(bulk_rows,[&](int first_chunk,int end_chunk,int block,int count) {
                require(count<=impl_->gdn_bulk_capacity,
                        "GDN bulk arena capacity");
                if (count<256) {
                    for(int chunk=first_chunk;chunk<end_chunk;++chunk)
                        execute_chunk(chunk_offsets[chunk],chunk_sizes[chunk],nullptr);
                    return;
                }
                auto* input=(layer%2==0?impl_->hidden_a:impl_->hidden_b)+
                    static_cast<std::size_t>(block)*kHidden;
                Exl3GdnLayer::BulkPrefillBuffers buffers{
                    impl_->gdn_bulk_h,impl_->gdn_bulk_qkv,impl_->gdn_bulk_z,count};
                impl_->gdn_layers[layer]->prepare_bulk_prefill(input,buffers,stream);
                ++gdn_bulk_calls;
                gdn_bulk_rows+=static_cast<std::uint64_t>(count);
                for(int chunk=first_chunk;chunk<end_chunk;++chunk) {
                    const int offset=chunk_offsets[chunk];
                    const int relative=offset-block;
                    Exl3GdnLayer::BulkPrefillBuffers slice{
                        buffers.h+static_cast<std::size_t>(relative)*kHidden,
                        buffers.qkv+static_cast<std::size_t>(relative)*10240,
                        buffers.z+static_cast<std::size_t>(relative)*6144,
                        chunk_sizes[chunk]};
                    execute_chunk(offset,chunk_sizes[chunk],&slice);
                }
            });
        } else {
            for(int chunk=0;chunk<chunks;++chunk)
                execute_chunk(chunk_offsets[chunk],chunk_sizes[chunk],nullptr);
        }
        const int tap=impl_->tap_index(layer);
        if (retained_taps && tap>=0) {
            const auto relative=static_cast<std::size_t>(retained_taps->first_abs-base_position);
            const auto* completed=(layer%2==0?impl_->hidden_b:impl_->hidden_a)+
                relative*kHidden;
            auto* plane=retained_taps->device+
                static_cast<std::size_t>(tap)*retained_taps->rows*kHidden;
            cuda_check(cudaMemcpyAsync(plane,completed,
                static_cast<std::size_t>(retained_taps->rows)*kHidden*sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice,stream),
                "retain completed layer-major tap tail");
        }
        workspace.end_layer_reuse();
        if (!layer_profile.events.empty())
            cuda_check(cudaEventRecord(layer_profile.events[layer+1],stream),
                       "record layer-major profile layer end");
        if (layer==fail_after_layer)
            throw std::runtime_error("injected layer-major partial-layer failure");
    }
    const auto after=workspace.stats();
    if (impl_->gdn_bulk_prefill_enabled)
        std::fprintf(stderr,"GDN_BULK_PREFILL prepares=%llu rows=%llu arena_bytes=%llu\n",
            static_cast<unsigned long long>(gdn_bulk_calls),
            static_cast<unsigned long long>(gdn_bulk_rows),
            static_cast<unsigned long long>(
                static_cast<std::size_t>(impl_->gdn_bulk_capacity)*
                (kHidden+10240+6144)*sizeof(std::uint16_t)));
    if (impl_->gdn_bulk_mlp_enabled)
        std::fprintf(stderr,"GDN_BULK_MLP blocks=%llu rows=%llu arena_bytes=%llu "
            "fused_down_calls=%llu prefetch_submissions=%llu prefetch_hits=%llu "
            "fused_residual_calls=%llu fused_residual_rows=%llu "
            "packed_k5_gate_up_calls=%llu packed_k5_down_calls=%llu "
            "packed_k5_rows=%llu\n",
            static_cast<unsigned long long>(gdn_bulk_mlp_blocks),
            static_cast<unsigned long long>(gdn_bulk_mlp_rows),
            static_cast<unsigned long long>(
                static_cast<std::size_t>(impl_->gdn_bulk_capacity)*
                (2*kHidden+2*17408)*sizeof(std::uint16_t)),
            static_cast<unsigned long long>(
                after.fused_gate_up_down_calls-before.fused_gate_up_down_calls),
            static_cast<unsigned long long>(
                after.prefetched_weight_submissions-
                before.prefetched_weight_submissions),
            static_cast<unsigned long long>(
                after.prefetched_weight_hits-before.prefetched_weight_hits),
            static_cast<unsigned long long>(
                after.fused_down_residual_calls-before.fused_down_residual_calls),
            static_cast<unsigned long long>(
                after.fused_down_residual_rows-before.fused_down_residual_rows),
            static_cast<unsigned long long>(
                after.packed_direct_k5_gate_up_calls-
                before.packed_direct_k5_gate_up_calls),
            static_cast<unsigned long long>(
                after.packed_direct_k5_down_calls-
                before.packed_direct_k5_down_calls),
            static_cast<unsigned long long>(
                after.packed_direct_k5_rows-before.packed_direct_k5_rows));
    impl_->fast_same_weights_fp16kv_prefill_stats.wide_prefill_forwards+=chunks;
    impl_->fast_same_weights_fp16kv_prefill_stats.wide_prefill_rows+=total;
    impl_->fast_same_weights_fp16kv_prefill_stats.numeric_dispatch_calls+=
        after.calls-before.calls;
    impl_->fast_same_weights_fp16kv_prefill_stats.numeric_dispatch_rows+=
        after.rows-before.rows;

    const int final_position=base_position+final_offset;
    cuda_check(cudaMemcpyAsync(impl_->position_device,&final_position,
                               sizeof(final_position),cudaMemcpyHostToDevice,stream),
               "publish final layer-major head position");
    // The even layer count leaves the completed residual in hidden_a. The
    // existing final norm/head and tap-generation code publishes the same
    // final chunk that sequential append_prefill_wide would expose.
    impl_->process_rows(nullptr,final_rows,final_position,stream,
        nullptr,false,final_rows>1,true,true,false,nullptr,
        kLayers,kLayers,
        impl_->hidden_a+static_cast<std::size_t>(final_offset)*kHidden,
        nullptr,false);
    const int last_position=base_position+total-1;
    cuda_check(cudaMemcpyAsync(impl_->position_device,&last_position,
                               sizeof(last_position),cudaMemcpyHostToDevice,stream),
               "publish completed layer-major position");
    cuda_check(cudaStreamSynchronize(stream),
               "complete layer-major layers before public position commit");
    if (!layer_profile.events.empty()) {
        double gdn_gpu_ms=0.0,attention_gpu_ms=0.0;
        for (int layer=0;layer<kLayers;++layer) {
            float elapsed_ms=0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms,
                layer_profile.events[layer],layer_profile.events[layer+1]),
                "resolve layer-major GPU owner timing");
            if (impl_->model->layers[layer].full_attention)
                attention_gpu_ms+=elapsed_ms;
            else gdn_gpu_ms+=elapsed_ms;
        }
        std::fprintf(stderr,
            "LAYER_MAJOR_PROFILE gdn_gpu_ms=%.3f full_attention_gpu_ms=%.3f "
            "gpu_stack_ms=%.3f rows=%d\n",
            gdn_gpu_ms,attention_gpu_ms,gdn_gpu_ms+attention_gpu_ms,total);
    }
    position_+=total;
    last_decode_h2d_+=chunks;
    } catch (...) {
        workspace.end_layer_reuse();
        impl_->fast_prefill_failed=true;
        impl_->last_rows=0;
        impl_->tap_rows=0;
        impl_->embedding_rows=0;
        impl_->qkv_trace_valid=false;
        impl_->last_hidden_source=nullptr;
        impl_->graph_active=false;
        throw;
    }
}

void Exl3TextContext::append_media_embeddings_numeric(std::span<const float> embeddings,
    std::span<const std::int32_t> positions_xyz,cudaStream_t stream){
    Impl::require_host_kv_retirement_admission();
    require(impl_->media_features&&impl_->media_positions&&impl_->host_kv.enabled&&impl_->capture_taps&&
        !impl_->oscar&&!impl_->graph_active&&!impl_->graph_capture_active&&!impl_->host_kv_failed&&
        (!impl_->transaction||(!impl_->transaction->active&&!impl_->transaction->rollback_required)),"media append requires intact research eager L2 owner");
    require(!embeddings.empty()&&embeddings.size()%kHidden==0&&embeddings.size()<=8ULL*kHidden,"media feature rows1..8 of5120");
    const int rows=static_cast<int>(embeddings.size()/kHidden),base=position_;
    require(positions_xyz.size()==std::size_t(rows)*3&&base<=impl_->max_context-rows,"media coordinate/remaining capacity extent");
    require(std::all_of(embeddings.begin(),embeddings.end(),[](float v){return std::isfinite(v)&&std::abs(v)<=65504.f;}),"media features must be finite FP16-representable");
    require(std::all_of(positions_xyz.begin(),positions_xyz.end(),[&](auto p){return p>=0&&p<impl_->max_context;}),"media rotary coordinates outside bounded context");
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;cuda_check(cudaStreamIsCapturing(stream,&capture),"media stream capture query");
    require(capture==cudaStreamCaptureStatusNone,"media append rejects external graph capture");
    impl_->validate_target_projection_execution(stream);
    const int maximum=std::max(base+impl_->rope_offset-1,*std::max_element(positions_xyz.begin(),positions_xyz.end()));
    const int next_offset=maximum+1-(base+rows);
    impl_->retain_host_kv_forward_stream(stream);
    if(impl_->continuation)impl_->continuation->rows=0;
    impl_->resident_exact_state_id=0;
    try {
        cuda_check(cudaMemcpyAsync(impl_->media_features->ptr,embeddings.data(),embeddings.size_bytes(),cudaMemcpyHostToDevice,stream),"upload bounded media features");
        cuda_check(cudaMemcpyAsync(impl_->media_positions->ptr,positions_xyz.data(),positions_xyz.size_bytes(),cudaMemcpyHostToDevice,stream),"upload bounded media positions");
        media_embedding_cast_kernel<<<(rows*kHidden+255)/256,256,0,stream>>>(static_cast<const float*>(impl_->media_features->ptr),impl_->hidden_a,rows*kHidden);
        cuda_check(cudaGetLastError(),"cast media features to ordinary FP16 L2 input");
        cuda_check(cudaMemcpyAsync(impl_->embedding_trace->ptr,impl_->hidden_a,std::size_t(rows)*kHidden*2,cudaMemcpyDeviceToDevice,stream),"capture injected media embeddings");
        impl_->embedding_rows=rows;impl_->tap_rows=rows;
        cuda_check(cudaMemcpyAsync(impl_->position_device,&base,sizeof(base),cudaMemcpyHostToDevice,stream),"media logical KV position");
        impl_->process_rows(nullptr,rows,base,stream,nullptr,false,rows>1,true,true,false,static_cast<const int*>(impl_->media_positions->ptr));
        const int final_position=base+rows-1;
        cuda_check(cudaMemcpyAsync(impl_->position_device,&final_position,sizeof(final_position),cudaMemcpyHostToDevice,stream),"media final logical position");
        cuda_check(cudaStreamSynchronize(stream),"media append full completion");
        position_=base+rows;impl_->rope_offset=next_offset;++last_decode_h2d_;
    }catch(...){impl_->host_kv_failed=true;cudaStreamSynchronize(stream);throw;}
}

std::size_t Exl3TextContext::continuation_bytes_required(int capacity) const {
    const bool native16 = capacity == 16 && impl_->native_continuation16_enabled &&
        impl_->host_kv.enabled && !impl_->oscar && !impl_->transaction;
    require((capacity >= 2 && capacity <= 8) || native16,
            "P2 target continuation capacity must be 2..8 or explicit exact-host16");
    require(impl_->capture_taps, "P2 target continuation requires captured taps");
    require(!impl_->graph_active && !impl_->graph_capture_active &&
                (!impl_->oscar || impl_->oscar->graph_class() == 0),
            "P2 target continuation setup requires eager target execution");
    if (impl_->continuation) {
        require(impl_->continuation->capacity == capacity,
                "P2 target continuation was prepared with a different capacity");
        return 0;
    }
    require(!impl_->transaction || !impl_->transaction->active,
            "P2 target continuation setup cannot allocate during an active transaction");
    auto bytes=Exl3LinearWorkspaceRequirements::derive(kHidden,kVocab,capacity).owned_bytes;
    bytes=Exl3LinearWorkspaceRequirements::append_owned_bytes(bytes,
        static_cast<std::size_t>(capacity)*kHidden*sizeof(std::uint16_t)+
            Impl::continuation_graph_role_bytes(capacity));
    return Exl3LinearWorkspaceRequirements::append_owned_bytes(bytes,
        static_cast<std::size_t>(capacity)*kVocab*sizeof(std::uint16_t));
}
void Exl3TextContext::prepare_continuation(int capacity) {
    prepare_continuation_impl(capacity,nullptr);
}
void Exl3TextContext::prepare_continuation_reserved(Exl3VeriCacheServingCoordinator& authority,int capacity,unsigned startup_fault_for_test) {
    prepare_continuation_impl(capacity,&authority,startup_fault_for_test);
}
std::size_t Exl3TextContext::continuation_owner_metadata_bytes() noexcept {
    return bounded_shared_allocation_bytes<Impl::Continuation>()+
        2*DeviceAllocation::owner_metadata_bytes()+Exl3CudaLinearWorkspace::metadata_bytes();
}
std::size_t Exl3TextContext::continuation_owner_blocks_for_test() noexcept {
    return bounded_shared_live_blocks_for_test<Impl::Continuation>();
}
void Exl3TextContext::exercise_continuation_owner_metadata_for_test() {
    RetainedDescriptorLedger ledger;
    const auto before=continuation_owner_blocks_for_test();
    const auto bytes=bounded_shared_allocation_bytes<Impl::Continuation>();
    require(bytes>sizeof(Impl::Continuation),"continuation metadata omits control storage");
    auto owner=make_bounded_shared<Impl::Continuation>();
    require(attach_bounded_retirement_credit<Impl::Continuation>(owner,ledger.acquire(bytes)),
        "continuation control credit attachment failed");
    std::weak_ptr<Impl::Continuation> weak=owner;
    owner.reset();
    require(weak.expired() && ledger.bytes()==bytes && continuation_owner_blocks_for_test()==before+1,
        "continuation final strong release dropped live weak control credit");
    weak.reset();
    require(ledger.bytes()==0 && continuation_owner_blocks_for_test()==before,
        "continuation final weak release retained control allocation or credit");
}
void Exl3TextContext::exercise_graph_retirement_poison_for_test() {
    const auto before=hostkv_quarantined_contexts.load(std::memory_order_acquire);
    require(before==0,"graph retirement poison fixture requires isolated process");
    auto owner=std::make_unique<Impl>();
    auto* raw=owner.get();
    owner->poison_graph_retirement(cudaSuccess);
    require(!owner->graph_retirement_error && hostkv_quarantined_contexts.load()==before,
        "successful graph retirement poisoned admission");
    owner->poison_graph_retirement(cudaErrorUnknown);
    owner->poison_graph_retirement(cudaErrorInvalidDevice);
    require(owner->graph_retirement_error==static_cast<int>(cudaErrorUnknown) &&
        hostkv_quarantined_contexts.load()==before+1,"graph retirement lost first error or double-counted owner");
    bool refused=false;
    try{Impl::require_host_kv_retirement_admission();}catch(const std::runtime_error&){refused=true;}
    require(refused,"graph retirement allowed execution before owner destruction");
    require(Impl::retain_failed_host_kv_drain(owner) && !owner &&
        Impl::host_kv_quarantine.load()==raw && raw->host_kv_retirement_error==static_cast<int>(cudaErrorUnknown) &&
        hostkv_quarantined_contexts.load()==before+1,
        "graph retirement failed to retain whole owner without a second charge");
    // Permanent retention is the expected isolated failure outcome.
}
void Exl3TextContext::exercise_context_graph_retirement_for_test(unsigned failure) {
    require(failure<=5,"context graph retirement failure menu");
    require(hostkv_quarantined_contexts.load()==0,"context graph retirement fixture requires isolated process");
    static int graph_storage=0,exec_storage=0;
    static unsigned graph_destroys=0,exec_destroys=0;
    static unsigned fault=0;
    graph_destroys=exec_destroys=0;fault=failure;
    RetainedDescriptorLedger ledger;
    const auto blocks=continuation_owner_blocks_for_test();
    const auto bytes=bounded_shared_allocation_bytes<Impl::Continuation>();
    auto owner=std::make_unique<Impl>();
    owner->graph_owner_device=3;
    owner->continuation=make_bounded_shared<Impl::Continuation>();
    require(attach_bounded_retirement_credit<Impl::Continuation>(owner->continuation,ledger.acquire(bytes)),
        "context graph continuation credit attachment");
    auto& continuation=*owner->continuation;
    continuation.graph_definition=DecodeGraphDefinition(DecodeGraphDefinition::Provider{
        +[](cudaStream_t){return cudaSuccess;},
        +[](cudaStream_t,cudaGraph_t* graph){*graph=reinterpret_cast<cudaGraph_t>(&graph_storage);return cudaSuccess;},
        +[](cudaGraph_t){++graph_destroys;return fault==2?cudaErrorUnknown:cudaSuccess;}});
    continuation.graph_executable=DecodeGraphExecutable(DecodeGraphExecutable::Provider{
        +[](cudaGraphExec_t* exec,cudaGraph_t){*exec=reinterpret_cast<cudaGraphExec_t>(&exec_storage);return cudaSuccess;},
        +[](cudaGraphExec_t){++exec_destroys;return fault==1?cudaErrorUnknown:cudaSuccess;}});
    continuation.graph_definition.capture(nullptr,[]{});
    continuation.graph_executable.instantiate(continuation.graph_definition);
    std::weak_ptr<Impl::Continuation> weak=owner->continuation;
    unsigned queries=0,drains=0;
    const auto query=[&](int* device) noexcept {++queries;*device=failure==4?4:3;return failure==3?cudaErrorInitializationError:cudaSuccess;};
    const auto drain=[&]() noexcept {++drains;return failure==5?cudaErrorUnknown:cudaSuccess;};
    const auto expected=failure==0?cudaSuccess:failure==3?cudaErrorInitializationError:
        failure==4?cudaErrorInvalidDevice:cudaErrorUnknown;
    const unsigned expected_drains=failure==3 || failure==4?0:1;
    const unsigned expected_exec=failure<=2?1:0;
    const unsigned expected_graph=failure==0 || failure==2?1:0;
    require(owner->retire_graph_handles_with(query,drain)==expected && queries==1 && drains==expected_drains &&
        exec_destroys==expected_exec && graph_destroys==expected_graph,
        "context graph retirement lost first-failure ordering");
    require(owner->retire_graph_handles_with(query,drain)==expected && queries==1 && drains==expected_drains &&
        exec_destroys==expected_exec && graph_destroys==expected_graph,
        "poisoned context retried graph retirement provider");
    const auto accounting=owner->continuation_graph_stats_snapshot();
    const auto entered_retirement=failure<=2?1ULL:0ULL;
    require(accounting.destruction_attempts==entered_retirement &&
        accounting.destruction_successes==(failure==0?1ULL:0ULL) &&
        accounting.destruction_failures==((failure==1 || failure==2)?1ULL:0ULL) &&
        accounting.live_definitions==(failure==0?0ULL:1ULL) &&
        accounting.live_executables==((failure==0 || failure==2)?0ULL:1ULL) &&
        !accounting.driver_memory_bytes_known && accounting.driver_memory_bytes==0,
        "context graph phase accounting fabricated or lost retirement state");
    if(!failure) {
        Impl::require_host_kv_retirement_admission();
        require(!owner->has_graph_handles() && !Impl::retain_failed_host_kv_drain(owner),
            "clean graph retirement poisoned or retained context");
        owner.reset();
        require(weak.expired() && ledger.bytes()==bytes && continuation_owner_blocks_for_test()==blocks+1,
            "clean context retirement lost weak continuation metadata");
        weak.reset();
        require(ledger.bytes()==0 && continuation_owner_blocks_for_test()==blocks,
            "clean context final weak retirement leaked metadata");
        return;
    }
    bool refused=false;try{Impl::require_host_kv_retirement_admission();}
    catch(const std::runtime_error&){refused=true;}
    auto* raw=owner.get();
    require(refused && Impl::retain_failed_host_kv_drain(owner) && !owner && !weak.expired() &&
        Impl::host_kv_quarantine.load()==raw && hostkv_quarantined_contexts.load()==1 &&
        continuation.graph_definition.ready() && continuation.graph_executable.ready()==(failure!=2) &&
        exec_destroys==expected_exec && graph_destroys==expected_graph && ledger.bytes()==bytes &&
        continuation_owner_blocks_for_test()==blocks+1,
        "failed context graph retirement lost continuation or retried cleanup");
}
void Exl3TextContext::exercise_graph_invalidation_for_test() {
    auto owner=std::make_unique<Impl>();
    owner->continuation=make_bounded_shared<Impl::Continuation>();
    auto& continuation=*owner->continuation;
    continuation.capacity=8;
    continuation.additional_graphs[0].rows=4;
    continuation.additional_graphs[1].rows=6;

    int context_address=0,model_address=0,scratch_address=0;
    std::array<int,3> buffer_addresses{};
    int stream_address=0,retired_resource_address=0;
    const auto stream=reinterpret_cast<cudaStream_t>(&stream_address);
    auto context_owner=std::make_shared<int>(1);
    auto model_owner=std::make_shared<int>(2);
    auto scratch_owner=std::make_shared<int>(3);
    auto retired_resource=std::make_shared<int>(4);
    std::shared_ptr<const void> context_identity(context_owner,&context_address);
    std::shared_ptr<const void> model_identity(model_owner,&model_address);
    std::shared_ptr<const void> scratch_identity(scratch_owner,&scratch_address);
    std::shared_ptr<const void> retained_identity(
        retired_resource,&retired_resource_address);
    std::weak_ptr<int> weak_retired=retired_resource;
    const auto options=Exl3ProjectionGraphOptions::from_values({"2","4","1","0"});
    Exl3GraphCaptureExtent reservation;
    reservation.known=true;
    reservation.retained[static_cast<unsigned>(
        Exl3ResourceInventory::Domain::graph_count)]=2;
    reservation.temporary[static_cast<unsigned>(
        Exl3ResourceInventory::Domain::graph_count)]=1;
    std::array<Exl3GraphCompatibilityFingerprint,3> candidates{};
    const auto bind=[&](Exl3ProjectionGraphBinding& projection,
        Exl3GraphCompatibilityFingerprint& compatibility,
        Exl3BoundedGraphEntry& lifecycle,unsigned rows,std::uint64_t generation,
        std::size_t slot) {
        projection.bind(model_identity,scratch_identity,options);
        std::array<Exl3GraphBufferIdentity,1> buffers{{
            {&buffer_addresses[slot],sizeof(buffer_addresses[slot])}}};
        candidates[slot].bind(context_identity,model_identity,scratch_identity,
            buffers,rows,8,8,5120,1,
            Exl3ContinuationGraphRouteVariableRows,
            Exl3GraphPrecision::oscar_int2_fp16,
            Exl3GraphPositionPolicy::oscar_split_class,generation,stream,options);
        require(candidates[slot].valid() &&
            projection.matches(model_identity,scratch_identity,options),
            "graph invalidation fixture binding");
        compatibility=candidates[slot];
        std::array<Exl3GraphBoundResource,1> resources{{
            {retained_identity,&buffer_addresses[slot],
                sizeof(buffer_addresses[slot]),rows}}};
        lifecycle.bind(candidates[slot],resources,generation,100+rows,reservation);
        require(lifecycle.admits(candidates[slot],generation),
            "graph invalidation fixture lifecycle admission");
    };
    bind(continuation.additional_graphs[0].projection_binding,
        continuation.additional_graphs[0].compatibility,
        continuation.additional_graphs[0].lifecycle,4,41,0);
    bind(continuation.additional_graphs[1].projection_binding,
        continuation.additional_graphs[1].compatibility,
        continuation.additional_graphs[1].lifecycle,6,61,1);
    bind(continuation.projection_binding,continuation.graph_compatibility,
        continuation.graph_lifecycle,8,81,2);
    continuation.graph_active=true;
    continuation.graph_reason="captured";
    for(auto& graph:continuation.additional_graphs) {
        graph.active=true;graph.reason="captured";
    }
    const auto replay=continuation.graph_lifecycle.begin_replay(81,stream);
    retired_resource.reset();retained_identity.reset();

    owner->invalidate_continuation_graphs(
        "reload changed physical membership");
    require(!continuation.graph_active &&
        !continuation.additional_graphs[0].active &&
        !continuation.additional_graphs[1].active &&
        continuation.graph_lifecycle.snapshot().phase==
            Exl3BoundedGraphEntry::Phase::invalidated &&
        continuation.additional_graphs[0].lifecycle.snapshot().phase==
            Exl3BoundedGraphEntry::Phase::invalidated &&
        continuation.additional_graphs[1].lifecycle.snapshot().phase==
            Exl3BoundedGraphEntry::Phase::invalidated &&
        continuation.graph_lifecycle.pending() &&
        !continuation.projection_binding.matches(
            model_identity,scratch_identity,options) &&
        !continuation.graph_compatibility.matches(candidates[2]) &&
        !continuation.graph_lifecycle.admits(candidates[2],81) &&
        !weak_retired.expired(),
        "graph invalidation restored admission or released an in-flight owner");
    bool failed_update=false;
    try {throw std::runtime_error("replacement allocation failed");}
    catch(const std::runtime_error&) {failed_update=true;}
    require(failed_update && continuation.graph_lifecycle.pending() &&
        continuation.graph_lifecycle.snapshot().phase==
            Exl3BoundedGraphEntry::Phase::invalidated &&
        !weak_retired.expired(),
        "failed graph update restored eligibility or released old ownership");
    require(!continuation.graph_lifecycle.release_after_destroy() &&
        continuation.graph_lifecycle.complete_after_drain(stream,replay,0) &&
        continuation.graph_lifecycle.release_after_destroy() &&
        continuation.additional_graphs[0].lifecycle.release_after_destroy() &&
        continuation.additional_graphs[1].lifecycle.release_after_destroy() &&
        weak_retired.expired(),
        "graph invalidation did not retain through final use and release after destruction");
}
void Exl3TextContext::prepare_continuation_impl(int capacity,Exl3VeriCacheServingCoordinator* authority,unsigned startup_fault) {
    require(startup_fault<=6 && (!startup_fault || authority),"continuation fault requires reserved stage1..6");
    const auto required=continuation_bytes_required(capacity);
    if(!required)return;
    const auto expected_impl=Exl3LinearWorkspaceRequirements::append_owned_bytes(impl_->persistent_bytes,required);
    const auto expected_public=Exl3LinearWorkspaceRequirements::append_owned_bytes(persistent_bytes_,required);
    const bool native16=capacity==16;
    const auto construct=[&] {
    auto prepared = make_bounded_shared<Impl::Continuation>();
    prepared->capacity = capacity;
    prepared->additional_graphs[0].rows = 4;
    prepared->additional_graphs[1].rows = 6;
    if(capacity==8) {
        constexpr std::array<unsigned,3> rows_menu{4,6,8};
        for(std::size_t i=0;i<rows_menu.size();++i) {
            prepared->graph_menu[i].rows=rows_menu[i];
            prepared->graph_menu[i].extent.known=true;
            prepared->graph_menu[i].extent.retained[static_cast<unsigned>(
                Exl3ResourceInventory::Domain::graph_count)]=2;
            // One bounded construction slot remains charged for the lifetime
            // of the menu, so later capture never depends on new availability.
            prepared->graph_menu[i].extent.temporary[static_cast<unsigned>(
                Exl3ResourceInventory::Domain::graph_count)]=1;
        }
        prepared->additional_graphs[0].roles.configure(4,kHidden);
        prepared->additional_graphs[1].roles.configure(6,kHidden);
        prepared->graph_roles.configure(8,kHidden);
    }
    const auto allocate_generic=[&](std::size_t bytes,const char* label) {
        if(!authority)return std::make_unique<DeviceAllocation>(bytes,label);
        auto credits=authority->reserve_constructor_credits(bytes,DeviceAllocation::owner_metadata_bytes());
        return std::make_unique<DeviceAllocation>(bytes,label,std::move(credits.device),std::move(credits.metadata));
    };
    prepared->final_norm=allocate_generic(
        static_cast<std::size_t>(capacity)*kHidden*sizeof(std::uint16_t)+
            Impl::continuation_graph_role_bytes(capacity),
        "allocate P2 continuation normalized rows");
    prepared->logits=allocate_generic(static_cast<std::size_t>(capacity)*kVocab*sizeof(std::uint16_t),
        "allocate P2 continuation logits");
    if(authority) {
        const auto plan=Exl3LinearWorkspaceRequirements::derive(kHidden,kVocab,capacity);
        auto credits=authority->reserve_constructor_credits(plan.owned_bytes,Exl3CudaLinearWorkspace::metadata_bytes());
        if(startup_fault==6)Exl3CudaLinearWorkspace::fail_constructor_cleanup_for_test(true);
        prepared->head_workspace=std::make_unique<Exl3CudaLinearWorkspace>(kHidden,kVocab,capacity,
            false,false,false,false,false,false,false,Exl3CudaAccumulationView{},Exl3CudaTransformView{},
            false,false,false,std::move(credits.device),std::move(credits.metadata));
    } else prepared->head_workspace=std::make_unique<Exl3CudaLinearWorkspace>(kHidden,kVocab,capacity);
    prepared->head_workspace->set_native_continuation16(native16);
    prepared->bytes = prepared->final_norm->bytes + prepared->logits->bytes +
        prepared->head_workspace->workspace_bytes();
    require(prepared->bytes==required,"target continuation allocation requirement mismatch");
    require(!prepared->graph_definition.ready() && !prepared->graph_executable.ready(),
        "continuation startup unexpectedly allocated graph resources");
    for(const auto& graph:prepared->additional_graphs)
        require(!graph.definition.ready() && !graph.executable.ready(),
            "continuation startup unexpectedly allocated additional graph resources");
    return prepared;
    };
    std::shared_ptr<Impl::Continuation> prepared;
    if(authority) {
        const auto quarantine_before=retirement_quarantine_witness();
        Exl3ResourceInventory::Requirement requirement;requirement.configuration=0x434F4E54494E5545;
        requirement.add(Exl3ResourceInventory::Domain::device,1,required);
        requirement.add(Exl3ResourceInventory::Domain::host_metadata,1,continuation_owner_metadata_bytes());
        authority->allocate_startup_resources(requirement,[&](std::uint64_t configuration) {
            require(configuration==requirement.configuration,"continuation startup reservation identity");
            if(startup_fault==1)throw std::runtime_error("injected continuation preconstruction failure");
            prepared=construct();
            if(startup_fault==5)prepared->final_norm->cleanup_failure_for_test=true;
            if(startup_fault==2)throw std::runtime_error("injected continuation precommit failure");
            Exl3ResourceInventory actual;
            const auto linear_bytes=prepared->head_workspace->workspace_bytes();
            actual.add({prepared,1,Exl3ResourceInventory::Domain::host_metadata,
                bounded_shared_allocation_bytes<Impl::Continuation>()-(startup_fault==4?1:0),{},
                &attach_bounded_retirement_credit<Impl::Continuation,const void>});
            const std::array<DeviceAllocation*,2> generic_children{prepared->final_norm.get(),prepared->logits.get()};
            for(unsigned i=0;i<generic_children.size();++i) {
                auto* child=generic_children[i];
                std::shared_ptr<const void> owner(prepared,child);
                actual.add({owner,4+2*i,Exl3ResourceInventory::Domain::device,
                    child->bytes-(!i && (startup_fault==3 || startup_fault==5)?1:0),{},nullptr,
                    &DeviceAllocation::attach_device_credit});
                actual.add({owner,5+2*i,Exl3ResourceInventory::Domain::host_metadata,
                    DeviceAllocation::owner_metadata_bytes(),{},&DeviceAllocation::attach_metadata_credit});
            }
            // Alias the actual child while retaining its enclosing continuation.
            // Distinct slots preserve disjoint extents under one control block.
            std::shared_ptr<const void> linear_owner(prepared,prepared->head_workspace.get());
            actual.add({linear_owner,2,Exl3ResourceInventory::Domain::device,linear_bytes,{},nullptr,
                +[](const std::shared_ptr<const void>& owner,RetainedDeviceLedger::Ticket credit) noexcept {
                    auto* workspace=const_cast<Exl3CudaLinearWorkspace*>(
                        static_cast<const Exl3CudaLinearWorkspace*>(owner.get()));
                    return workspace && workspace->attach_device_credit(std::move(credit));
                }});
            actual.add({linear_owner,3,Exl3ResourceInventory::Domain::host_metadata,
                Exl3CudaLinearWorkspace::metadata_bytes(),{},
                +[](const std::shared_ptr<const void>& owner,RetainedDescriptorLedger::Ticket credit) noexcept {
                    auto* workspace=const_cast<Exl3CudaLinearWorkspace*>(
                        static_cast<const Exl3CudaLinearWorkspace*>(owner.get()));
                    return workspace && workspace->attach_metadata_credit(std::move(credit));
                }});
            return actual;
        },[&]() noexcept {
            // Startup has no captured graphs or submitted continuation work.
            // Drop the external workspace owner before reopening admission.
            prepared.reset();
            if(retirement_quarantine_witness()!=quarantine_before)
                authority->seal_failed_startup_retirement();
        },[&]() noexcept {
            // The coordinator's inventory copies may release the final owner
            // only after rollback returns. Observe that second destruction edge.
            if(retirement_quarantine_witness()!=quarantine_before)
                authority->seal_failed_startup_retirement();
        });
        prepared->final_norm->release_constructor_credits_after_commit();
        prepared->logits->release_constructor_credits_after_commit();
        prepared->head_workspace->release_constructor_credits_after_commit();
    } else prepared=construct();
    if (native16) {
        // These kernels already own a physical M16 tile; all sixteen
        // rows share each packed weight load and original K-split.
        // Other projections retain their qualified per-row fallback.
        for (int layer=0; layer<kLayers; ++layer) {
            if (impl_->full_layers[layer]) impl_->full_layers[layer]->set_native_continuation16(true);
            if (impl_->gdn_layers[layer]) impl_->gdn_layers[layer]->set_native_continuation16(true);
        }
    }
    impl_->continuation = std::move(prepared);
    impl_->persistent_bytes=expected_impl;persistent_bytes_=expected_public;
    if(authority && capacity==8 && impl_->continuation_graph_b8_enabled) {
        auto& scratch=*impl_->continuation;
        scratch.graph_menu_required=true;
        scratch.graph_menu_configuration=0x475241504D454E55ULL;
        try {
            scratch.graph_menu_reservation=exl3_reserve_graph_menu_startup(
                *authority,scratch.graph_menu_configuration,scratch.graph_menu,
                impl_->continuation,200);
            scratch.graph_menu_reserved=true;
            scratch.graph_menu_reason="reserved B4/B6/B8 handles and capture slot";
        } catch(const Exl3ResourceReservationExhausted&) {
            scratch.graph_menu_reason=
                "graph menu peak unavailable; eager continuation selected before capture";
        }
    }
}

void Exl3TextContext::continue_rows(std::span<const std::int64_t> token_ids,
                                    cudaStream_t stream) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    require(!impl_->host_kv_failed,"HostKV continuation requires intact transfer lineage");
    require(!impl_->transaction || !impl_->transaction->rollback_required,
            "P2 target transaction requires rollback after failed prefix retention");
    require(impl_->continuation != nullptr,
            "P2 target continuation was not prepared");
    require(token_ids.size() >= 2 &&
                token_ids.size() <= static_cast<std::size_t>(impl_->continuation->capacity),
            "P2 target continuation row count is outside prepared capacity");
    require(std::all_of(token_ids.begin(), token_ids.end(), [](std::int64_t token) {
                return token >= 0 && token < kVocab;
            }),
            "P2 target continuation token ID is outside the vocabulary");
    const int rows = static_cast<int>(token_ids.size());
    require(position_ > 0 && impl_->last_rows > 0,
            "P2 target continuation requires a completed nonempty prefix");
    require(position_ <= impl_->max_context - rows,
            "P2 target continuation exceeds context capacity");
    require(impl_->capture_taps && !impl_->graph_active &&
                !impl_->graph_capture_active && (!impl_->oscar || impl_->oscar->graph_class() == 0),
            "P2 target continuation requires eager target execution");
    if (impl_->target_projection_timing &&
        impl_->target_projection_timing->active()) {
        impl_->validate_target_projection_execution(stream);
    } else {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        cuda_check(cudaStreamIsCapturing(stream, &capture_status),
                   "query P2 target continuation stream capture state");
        require(capture_status == cudaStreamCaptureStatusNone,
                "P2 target continuation is unavailable during external stream capture");
    }

    const bool arm_retained_prefix = impl_->transaction &&
        impl_->transaction->active && impl_->transaction->fresh_snapshot &&
        impl_->transaction->stream == stream;

    // Validation above is mutation-free. From this point any failure leaves
    // derived all-row output invalid rather than exposing a partial result.
    impl_->retain_host_kv_forward_stream(stream);
    Impl::HostKVForwardFailureScope failure_scope{*impl_};
    impl_->continuation->rows = 0;
    cuda_check(cudaMemcpyAsync(impl_->token_ids, token_ids.data(), token_ids.size_bytes(),
                               cudaMemcpyHostToDevice, stream),
               "upload P2 target continuation token IDs");
    const int base_position = position_;
    cuda_check(cudaMemcpyAsync(impl_->position_device, &base_position, sizeof(base_position),
                               cudaMemcpyHostToDevice, stream),
               "set P2 target continuation base position");
    impl_->process_rows(impl_->token_ids, rows, base_position, stream,
                        nullptr, true, true, true);
    const int final_device_position = base_position + rows - 1;
    cuda_check(cudaMemcpyAsync(impl_->position_device, &final_device_position,
                               sizeof(final_device_position), cudaMemcpyHostToDevice, stream),
               "set P2 target continuation final device position");
    position_ += rows;
    impl_->continuation->rows = rows;
    if (arm_retained_prefix) {
        auto& transaction = *impl_->transaction;
        transaction.attempt_base_position = base_position;
        transaction.attempted_rows = rows;
        transaction.prefix_available = true;
    }
    ++last_decode_h2d_;
}

void Exl3TextContext::bind_request_compatibility(std::string contract) {
    if(contract.empty()) throw std::invalid_argument("request compatibility contract missing");
    if(!impl_->request_compatibility_contract.empty()) {
        if(impl_->request_compatibility_contract!=contract)
            throw std::invalid_argument("request compatibility contract already bound");
        return;
    }
    require(position_==0 && (!impl_->transaction || !impl_->transaction->active) &&
        !impl_->graph_active && !impl_->graph_capture_active,
        "request compatibility binding requires a pristine context");
    impl_->request_compatibility_contract=std::move(contract);
}

Exl3RequestResetStats Exl3TextContext::reset_for_request(std::string_view contract) {
    return reset_for_request_impl(contract,nullptr,0);
}
Exl3RequestResetStats Exl3TextContext::reset_for_request_preserving(std::string_view contract,
    const Exl3ExactHostState& root,std::uint64_t expected_generation) {
    return reset_for_request_impl(contract,&root,expected_generation);
}
Exl3RequestResetStats Exl3TextContext::reset_for_request_impl(std::string_view contract,
    const Exl3ExactHostState* root,std::uint64_t expected_generation) {
    impl_->drain_repair();
    Impl::require_host_kv_retirement_admission();
    require(!impl_->host_kv_failed,"request reset cannot reuse failed HostKV lineage");
    require(!impl_->reconstruction_backing || !impl_->reconstruction_backing->stream.failed(),
        "request reset cannot reuse failed reconstruction slab");
    require(std::none_of(impl_->registered_kv_pending.begin(),
            impl_->registered_kv_pending.end(),[](std::uint64_t pending) {
                return pending!=0;
            }),
        "request reset cannot abandon pending registered KV owners");
    if(contract.empty() || impl_->request_compatibility_contract!=contract)
        throw std::invalid_argument("incompatible request context reuse");
    if(impl_->request_generation==std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("request context generation exhausted");
    if(root && expected_generation!=impl_->request_generation)
        throw std::invalid_argument("stale request context generation");
    // Include both prior-work drains and final reset completion. A failed
    // synchronization cannot leave a preserved residency witness reusable merely
    // because reset_impl itself returned successfully.
    Impl::HostKVForwardFailureScope request_failure_scope{*impl_};
    const auto start=std::chrono::steady_clock::now();
    // Request reuse composes default-stream target work, optional copy streams
    // and graph/explicit execution streams. T011 narrows only its proven graph
    // mutation subset; this whole-context fallback remains device-wide.
    cuda_check(cudaDeviceSynchronize(),"drain prior request work before context reuse");
    if(impl_->host_kv_copy_stream) {
        const auto finish_copies=[&] {
            cuda_check(cudaStreamSynchronize(impl_->host_kv_copy_stream),
                "drain exact host copy work before context reuse");
        };
        if(!impl_->host_kv_batch_use.retire_before_reuse(finish_copies))finish_copies();
    }
    impl_->host_kv_batch_destinations.clear();
    impl_->host_kv_batch_sources.clear();
    impl_->host_kv_batch_sizes.clear();
    const auto drained=std::chrono::steady_clock::now();
    // A failed/cancelled deferred publication may retain descriptors for pages
    // that will never become authoritative. Work is complete above, so release
    // those owners without scattering abandoned bytes.
    for(std::size_t slot=0;slot<impl_->host_kv_pinned_in_flight.size();++slot) {
        impl_->host_kv_pinned_in_flight[slot]=false;
        impl_->host_kv_pinned_final_use[slot]={};
        impl_->registered_kv_pending[slot]=0;
        impl_->registered_kv_pending_bytes[slot]=0;
    }
    for(std::size_t slot=0;slot<impl_->host_kv_pinned_scatter.size();++slot) {
        impl_->host_kv_pinned_scatter[slot].clear();
    }
    // Recheck after draining all prior work. Same-position/foreign/changed roots
    // conservatively take full reset; no payload comparison or copy is needed.
    const bool preserve=root && root->rope_offset_==0 && exact_host_state_resident(*root);
    reset_impl(nullptr,preserve);
    if(std::exchange(impl_->fail_request_reset_completion_for_test,false))
        throw std::runtime_error("injected request reset completion failure");
    cuda_check(cudaDeviceSynchronize(),"complete request context reset");
    const auto complete=std::chrono::steady_clock::now();
    ++impl_->request_generation;
    return Exl3RequestResetStats{
        impl_->request_generation,
        std::chrono::duration<double,std::milli>(drained-start).count(),
        std::chrono::duration<double,std::milli>(complete-drained).count(),
        persistent_bytes_,preserve};
}

std::uint64_t Exl3TextContext::request_generation() const noexcept {
    return impl_->request_generation;
}
void Exl3TextContext::exhaust_request_generation_for_test() {
    impl_->request_generation=std::numeric_limits<std::uint64_t>::max();
}
void Exl3TextContext::fail_next_request_reset_completion_for_test() {
    require(!impl_->host_kv_failed,"cannot arm reset completion on failed HostKV lineage");
    if(impl_->fail_request_reset_completion_for_test)
        throw std::logic_error("request reset completion failure already armed");
    impl_->fail_request_reset_completion_for_test=true;
}

void Exl3TextContext::append_exact_prefill_wide(std::span<const std::int64_t> token_ids,cudaStream_t stream) {
    require(impl_->wide_prefill_enabled && impl_->host_kv.enabled,"exact wide prefill requires latched wide host context");
    require(token_ids.size()==16 || token_ids.size()==32 || token_ids.size()==128 || token_ids.size()==1024,
        "exact wide prefill requires a qualified16/32/128/1024-row shape");
    append_prefill_impl(token_ids,stream,true,true);
}

void Exl3TextContext::poison_prefill_tail_hidden_for_test(int rows,int byte,cudaStream_t stream) {
    require(rows>=17 && rows<=31 && byte>=0 && byte<=255 && impl_->host_kv.enabled &&
        impl_->prefill_capacity>=32 && !impl_->oscar && !impl_->host_kv_failed &&
        !impl_->graph_active && !impl_->graph_capture_active &&
        (!impl_->transaction || !impl_->transaction->active),
        "hidden tail poison requires idle exact-host reserved32");
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream,&capture),"hidden tail poison capture query");
    require(capture==cudaStreamCaptureStatusNone,"hidden tail poison rejects capture");
    const auto offset=static_cast<std::size_t>(rows)*kHidden;
    const auto bytes=static_cast<std::size_t>(32-rows)*kHidden*sizeof(std::uint16_t);
    cuda_check(cudaMemsetAsync(impl_->hidden_a+offset,byte,bytes,stream),"poison hidden A row tail");
    cuda_check(cudaMemsetAsync(impl_->hidden_b+offset,byte,bytes,stream),"poison hidden B row tail");
}

void Exl3TextContext::append_exact_prefill_tail(std::span<const std::int64_t> token_ids,cudaStream_t stream) {
    require(impl_->wide_prefill_enabled && impl_->host_kv.enabled,
        "actual-row tail requires latched wide exact-host context");
    require(token_ids.size()>=17 && token_ids.size()<=31,
        "actual-row tail supports only17..31 represented rows");
    // The shared admission checks enforce remaining position and reserved row
    // capacity before uploads. No padded token is passed to recurrent/KV work.
    append_prefill_impl(token_ids,stream,true,true);
}

void Exl3TextContext::finish_exact_continuation(cudaStream_t stream) {
    impl_->join_repair(stream);
    const bool transaction_ready=!impl_->transaction ||
        (!impl_->transaction->active && !impl_->transaction->rollback_required);
    require(transaction_ready && !impl_->graph_active && !impl_->graph_capture_active &&
            impl_->continuation && impl_->continuation->rows > 0,
            "finish exact continuation requires ordinary eager completed rows");
    const int rows=impl_->continuation->rows;
    require(impl_->tap_rows==rows && impl_->embedding_rows==rows,
            "finish exact continuation payload extent");
    canonicalize_exact_rows(stream);
    impl_->continuation->rows=0;
}

void Exl3TextContext::finish_exact_prefill(cudaStream_t stream) {
    require(!impl_->continuation || impl_->continuation->rows==0,"exact prefill still has continuation output");
    canonicalize_exact_rows(stream);
}

void Exl3TextContext::canonicalize_exact_rows(cudaStream_t stream) {
    const bool transaction_ready=!impl_->transaction ||
        (!impl_->transaction->active && !impl_->transaction->rollback_required);
    require(!impl_->oscar_only && transaction_ready && !impl_->graph_active && !impl_->graph_capture_active &&
        !impl_->host_kv_failed && impl_->capture_taps && impl_->tap_rows>0 &&
        impl_->tap_rows==impl_->embedding_rows && impl_->tap_rows==impl_->last_rows,
        "exact final-row canonicalization requires intact ordinary output");
    impl_->validate_target_projection_execution(stream);
    const int rows=impl_->tap_rows;
    if(rows>1) {
        for(const auto& tap:impl_->taps)
            cuda_check(cudaMemcpyAsync(tap->ptr,
                static_cast<const std::uint16_t*>(tap->ptr)+(rows-1)*kHidden,
                kHidden*sizeof(std::uint16_t),cudaMemcpyDeviceToDevice,stream),
                "retain exact continuation final tap");
        cuda_check(cudaMemcpyAsync(impl_->embedding_trace->ptr,
            static_cast<const std::uint16_t*>(impl_->embedding_trace->ptr)+(rows-1)*kHidden,
            kHidden*sizeof(std::uint16_t),cudaMemcpyDeviceToDevice,stream),
            "retain exact continuation final embedding");
    }
    impl_->tap_rows=1; impl_->embedding_rows=1; impl_->last_rows=1;
    impl_->tap_generation.fetch_add(1,std::memory_order_release);
}

void Exl3TextContext::retain_transaction_prefix(int retained_rows,
                                                cudaStream_t stream) {
    retain_transaction_prefix_impl(retained_rows, -1, stream);
}

void Exl3TextContext::retain_transaction_prefix_for_test(
    int retained_rows, int fail_after_model_layer, cudaStream_t stream) {
    require(fail_after_model_layer >= 0 && fail_after_model_layer < kLayers,
            "P2 retained-prefix test failure layer must be 0..63");
    retain_transaction_prefix_impl(
        retained_rows, fail_after_model_layer, stream);
}

void Exl3TextContext::retain_transaction_prefix_impl(
    int retained_rows, int fail_after_model_layer, cudaStream_t stream) {
    impl_->join_repair(stream);
    require(impl_->transaction != nullptr && impl_->transaction->active,
            "P2 retained prefix requires an active target transaction");
    auto& transaction = *impl_->transaction;
    require(!transaction.rollback_required,
            "P2 target transaction requires rollback after failed prefix retention");
    require(transaction.prefix_available && transaction.attempted_rows >= 2,
            "P2 retained prefix has no immediate continuation source");
    require(retained_rows > 0 && retained_rows <= transaction.attempted_rows,
            "P2 retained prefix row count is invalid");
    require(stream == transaction.stream,
            "P2 retained prefix stream differs from its transaction");
    require(transaction.attempt_base_position == transaction.position &&
                position_ == transaction.position + transaction.attempted_rows,
            "P2 retained prefix position provenance mismatch");
    for (int layer = 0; layer < kLayers; ++layer)
        if (impl_->gdn_layers[layer])
            impl_->gdn_layers[layer]->validate_saved_checkpoint(
                transaction.gdn_checkpoints[layer],transaction.position);
    require(impl_->continuation != nullptr &&
                impl_->continuation->rows == transaction.attempted_rows &&
                impl_->last_rows == transaction.attempted_rows &&
                impl_->tap_rows == transaction.attempted_rows &&
                impl_->embedding_rows == transaction.attempted_rows,
            "P2 retained prefix source rows are inconsistent");
    const bool host_kv = transaction.host_kv && impl_->host_kv.enabled &&
        impl_->oscar == nullptr;
    const bool device_kv=transaction.device_kv && !impl_->host_kv.enabled &&
        impl_->oscar == nullptr;
    require(impl_->capture_taps &&
                (host_kv || device_kv || (impl_->oscar != nullptr &&
                    impl_->oscar->graph_class() == 0)) &&
                !impl_->graph_active && !impl_->graph_capture_active,
            "P2 retained prefix requires its eager OSCAR or exact HostKV mode");
    impl_->validate_target_projection_execution(stream);
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status),
               "query P2 retained prefix stream capture state");
    require(capture_status == cudaStreamCaptureStatusNone,
            "P2 retained prefix is unavailable during external stream capture");

    const int attempted_rows = transaction.attempted_rows;
    const int base_position = transaction.position;
    if (retained_rows < attempted_rows) {
        // Validate every full-attention scratch capability before the first
        // state mutation. GDN capabilities derive from the same fresh begin +
        // immediate continuation protocol and validate again on consumption.
        for (const auto& layer : impl_->full_layers) {
            if (layer) layer->validate_retained_prefix_reappend(
                retained_rows, attempted_rows, base_position, stream);
        }
    }

    // Consume before mutation. Any later exception leaves the original
    // transaction active and rollback remains the only permitted recovery.
    transaction.prefix_available = false;
    transaction.fresh_snapshot = false;
    if (retained_rows == attempted_rows) {
        for (auto& layer : impl_->full_layers)
            if (layer) layer->invalidate_retained_prefix();
        return;
    }

    transaction.continuation_graph_attempt = false;
    transaction.continuation_graph_rows = 0;

    transaction.rollback_required = true;
    impl_->continuation->rows = 0;
    impl_->tap_rows = 0;
    impl_->embedding_rows = 0;
    impl_->last_rows = 0;
    impl_->qkv_trace_valid = false;

    if (!host_kv && !device_kv)
        impl_->oscar->restore_checkpoint(transaction.oscar_checkpoint, stream);
    static const bool repair_graph=[] {
        const char* value=std::getenv("NINFER_EXL3_GDN_REPAIR_GRAPH");
        if(!value)return true;  // measured default; "0" keeps eager per-layer repair
        if(std::strcmp(value,"0")==0)return false;
        if(std::strcmp(value,"1")==0)return true;
        throw std::invalid_argument("NINFER_EXL3_GDN_REPAIR_GRAPH must be 0 or 1");
    }();
    if (repair_graph && device_kv && fail_after_model_layer < 0 &&
        attempted_rows >= 2 && attempted_rows <= 8) {
        // Every per-layer host check and state transition runs first; the
        // device work of all GDN layers then replays as one captured graph.
        for (int layer = 0; layer < kLayers; ++layer) {
            if (impl_->full_layers[layer]) {
                impl_->full_layers[layer]->reappend_retained_prefix(
                    retained_rows, attempted_rows, base_position, stream);
            } else if (impl_->gdn_layers[layer]) {
                const int layer_attempted=impl_->gdn_layers[layer]->
                    reconstruct_retained_prefix_host(
                        transaction.gdn_checkpoints[layer], retained_rows, stream);
                require(layer_attempted==attempted_rows,
                    "P2 GDN repair graph attempted-row mismatch");
            }
        }
        auto& graph=impl_->gdn_repair_graphs[
            static_cast<std::size_t>(retained_rows-1)*8+(attempted_rows-1)];
        std::array<const void*,kLayers> checkpoints{};
        for (int layer = 0; layer < kLayers; ++layer)
            if (impl_->gdn_layers[layer])
                checkpoints[layer]=transaction.gdn_checkpoints[layer].recurrent_state_device;
        if (graph.ready && graph.recorded_checkpoints!=checkpoints) {
            // Waits for any in-flight replay before the executable is replaced.
            cuda_check(cudaStreamSynchronize(stream),"drain stale GDN repair graph");
            if(impl_->repair_stream)
                cuda_check(cudaStreamSynchronize(impl_->repair_stream),
                    "drain stale overlapped GDN repair graph");
            graph.ready=false;
        }
        if (!graph.ready) {
            graph.recorded_checkpoints=checkpoints;
            impl_->bind_graph_device();
            cudaStream_t capture_stream=nullptr;
            cuda_check(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking),
                "create GDN repair graph capture stream");
            try {
                graph.definition.capture(capture_stream,[&] {
                    for (int layer = 0; layer < kLayers; ++layer)
                        if (impl_->gdn_layers[layer])
                            impl_->gdn_layers[layer]->enqueue_retained_prefix_reconstruct(
                                transaction.gdn_checkpoints[layer], retained_rows,
                                attempted_rows, capture_stream);
                });
                graph.executable.instantiate(graph.definition);
                graph.executable.upload(capture_stream);
                cuda_check(cudaStreamSynchronize(capture_stream),
                    "complete GDN repair graph preparation");
            } catch(...) {
                (void)cudaStreamSynchronize(capture_stream);
                (void)cudaStreamDestroy(capture_stream);
                throw;
            }
            cuda_check(cudaStreamDestroy(capture_stream),
                "destroy GDN repair graph capture stream");
            graph.ready=true;
        }
        static const bool overlap=[] {
            const char* value=std::getenv("NINFER_EXL3_GDN_REPAIR_OVERLAP");
            return !value || std::strcmp(value,"0")!=0;
        }();
        if(overlap) {
            if(!impl_->repair_stream) {
                cuda_check(cudaStreamCreateWithFlags(&impl_->repair_stream,
                    cudaStreamNonBlocking),"create overlapped GDN repair stream");
                cuda_check(cudaEventCreateWithFlags(&impl_->repair_fork,
                    cudaEventDisableTiming),"create GDN repair fork event");
                cuda_check(cudaEventCreateWithFlags(&impl_->repair_join,
                    cudaEventDisableTiming),"create GDN repair join event");
            }
            cuda_check(cudaEventRecord(impl_->repair_fork,stream),"fork GDN repair");
            cuda_check(cudaStreamWaitEvent(impl_->repair_stream,impl_->repair_fork,0),
                "order GDN repair after verification");
            graph.executable.launch(impl_->repair_stream);
            cuda_check(cudaEventRecord(impl_->repair_join,impl_->repair_stream),
                "record GDN repair completion");
            impl_->repair_pending=true;
        } else
        graph.executable.launch(stream);
    } else
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->full_layers[layer]) {
            impl_->full_layers[layer]->reappend_retained_prefix(
                retained_rows, attempted_rows, base_position, stream);
        } else if (impl_->gdn_layers[layer]) {
            impl_->gdn_layers[layer]->reconstruct_retained_prefix(
                transaction.gdn_checkpoints[layer], retained_rows, stream);
        }
        if (layer == fail_after_model_layer) {
            throw std::runtime_error(
                "P2 injected retained-prefix failure after model layer " +
                std::to_string(layer));
        }
    }
    const auto* all_logits = static_cast<const std::uint16_t*>(
        impl_->continuation->logits->ptr);
    cuda_check(cudaMemcpyAsync(
        impl_->logits,
        all_logits + static_cast<std::size_t>(retained_rows - 1) * kVocab,
        kVocab * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice, stream),
        "select P2 retained-prefix logits");
    const int device_position = base_position + retained_rows - 1;
    cuda_check(cudaMemcpyAsync(impl_->position_device, &device_position,
                               sizeof(device_position), cudaMemcpyHostToDevice,
                               stream),
               "set P2 retained-prefix device position");

    if (host_kv) {
        const int retained_position = base_position + retained_rows;
        const auto retained_pages = static_cast<std::size_t>(
            (retained_position + Exl3ExactKVPage::token_capacity - 1) /
            Exl3ExactKVPage::token_capacity);
        require(retained_pages > 0 && retained_pages <= impl_->exact_prefix_pages.size(),
                "P2 HostKV retained-prefix page extent mismatch");
        impl_->exact_prefix_pages.resize(retained_pages);
        const int tail_rows = retained_position -
            static_cast<int>(retained_pages - 1) * Exl3ExactKVPage::token_capacity;
        auto& tail_ref = impl_->exact_prefix_pages.back();
        require(tail_ref.use_count() == 1 && tail_rows > 0 &&
                    tail_rows <= tail_ref->rows,
                "P2 HostKV retained-prefix tail is not privately truncatable");
        auto tail = std::const_pointer_cast<Exl3ExactKVPage>(tail_ref);
        if (tail_rows < tail->rows) {
            tail->rows = tail_rows;
            for (int bank = 0; bank < 16; ++bank) {
                tail->k[bank].resize(static_cast<std::size_t>(tail_rows) * 1024);
                tail->v[bank].resize(static_cast<std::size_t>(tail_rows) * 1024);
            }
        }
        impl_->exact_prefix_position = retained_position;
    }

    // Attempt taps and embedding already contain the exact prefix. Publish
    // their shorter logical extent only after all repair work and final device
    // metadata copies were enqueued successfully.
    position_ = base_position + retained_rows;
    impl_->tap_rows = retained_rows;
    impl_->embedding_rows = retained_rows;
    impl_->last_rows = retained_rows;
    impl_->continuation->rows = retained_rows;
    impl_->qkv_trace_valid = false;
    impl_->invalidate_native_mtp_hidden_capture();
    transaction.rollback_required = false;
}

int Exl3TextContext::continuation_capacity() const noexcept {
    return impl_->continuation ? impl_->continuation->capacity : 0;
}

int Exl3TextContext::continuation_rows() const noexcept {
    return impl_->continuation ? impl_->continuation->rows : 0;
}

std::size_t Exl3TextContext::continuation_bytes() const noexcept {
    return impl_->continuation ? impl_->continuation->bytes : 0;
}

std::string Exl3TextContext::continuation_head_dispatch() const {
    require(impl_->continuation != nullptr,
            "P2 target continuation was not prepared");
    const char* route = impl_->continuation->head_workspace->dispatch_name(
        impl_->model->lm_head_metadata, impl_->continuation->capacity);
    return std::string(route) == "h6_small_m_single_split"
        ? route : "h6_single_split_per_row";
}

const std::uint16_t* Exl3TextContext::continuation_logits_device() const noexcept {
    return impl_->continuation && impl_->continuation->rows > 0
        ? static_cast<const std::uint16_t*>(impl_->continuation->logits->ptr) : nullptr;
}

std::vector<std::uint16_t> Exl3TextContext::continuation_logits_bits_host(
    cudaStream_t stream) const {
    require(impl_->continuation != nullptr && impl_->continuation->rows > 0,
            "P2 target continuation logits are invalid");
    const std::size_t count = static_cast<std::size_t>(impl_->continuation->rows) * kVocab;
    std::vector<std::uint16_t> raw(count);
    cuda_check(cudaMemcpyAsync(raw.data(), impl_->continuation->logits->ptr,
                               count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, stream),
               "download P2 target continuation logits");
    cuda_check(cudaStreamSynchronize(stream), "synchronize P2 target continuation logits");
    const auto rows=impl_->continuation->rows;
    const auto* graph_entry=rows==8?&impl_->continuation->graph_lifecycle:
        (rows==4?&impl_->continuation->additional_graphs[0].lifecycle:
            (rows==6?&impl_->continuation->additional_graphs[1].lifecycle:nullptr));
    const auto* graph_roles=impl_->continuation_graph_roles(
        static_cast<unsigned>(rows));
    const bool graph_use_pending=graph_entry &&
        (graph_entry->pending() || (graph_roles && graph_roles->pending_replay()));
    if(graph_use_pending &&
       !impl_->complete_continuation_graph_use(rows,stream)) {
        const auto& entry=*graph_entry;
        const auto snapshot=entry.snapshot();
        throw std::runtime_error(
            "continuation graph logits completion lost final-use ownership rows="+
            std::to_string(rows)+" phase="+
            std::to_string(static_cast<unsigned>(snapshot.phase))+" pending="+
            std::to_string(snapshot.pending)+" replay="+
            std::to_string(snapshot.replay_serial)+" pending_stream="+
            std::to_string(snapshot.pending_stream)+" completion_stream="+
            std::to_string(reinterpret_cast<std::uintptr_t>(stream))+
            " roles="+std::to_string(graph_roles!=nullptr)+" role_pending="+
            std::to_string(graph_roles?graph_roles->pending_replay():0)+
            " role_generation="+
            std::to_string(graph_roles?graph_roles->generation():0)+" role_uploaded="+
            std::to_string(graph_roles?graph_roles->uploaded_generation():0));
    }
    impl_->greedy_readback.full_score_bytes+=count*sizeof(std::uint16_t);
    return raw;
}

std::vector<float> Exl3TextContext::continuation_logits_host(cudaStream_t stream) const {
    const auto raw = continuation_logits_bits_host(stream);
    std::vector<float> result(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) result[i] = half_to_float(raw[i]);
    return result;
}

std::size_t Exl3TextContext::greedy_packet_bytes_required() noexcept {
    return Exl3GreedyPacketTransfer::row_bytes*(1+greedy_transfer_pool_capacity());
}
std::size_t Exl3TextContext::greedy_transfer_pool_registered_bytes_required() noexcept {
    return Exl3GreedyPacketTransfer::row_bytes*greedy_transfer_pool_capacity();
}
std::size_t Exl3TextContext::greedy_transfer_pool_metadata_bytes_required() noexcept {
    return Exl3GreedyPacketTransfer::metadata_bytes_required()*greedy_transfer_pool_capacity();
}
void Exl3TextContext::prepare_greedy_packet() {
    if(!impl_->greedy_rows) {
        const auto required=greedy_packet_bytes_required();
        const auto next_impl=Exl3LinearWorkspaceRequirements::append_owned_bytes(impl_->persistent_bytes,required);
        const auto next_public=Exl3LinearWorkspaceRequirements::append_owned_bytes(persistent_bytes_,required);
        auto allocation=std::make_unique<DeviceAllocation>(
            Exl3GreedyPacketTransfer::row_bytes,"greedy packet");
        std::array<std::shared_ptr<Exl3GreedyPacketTransfer>,2> transfers;
        for(auto& transfer:transfers)
            transfer=make_bounded_shared<Exl3GreedyPacketTransfer>();
        require(allocation->bytes+transfers.size()*Exl3GreedyPacketTransfer::row_bytes==required,
            "greedy packet pool allocation requirement mismatch");
        impl_->greedy_rows=std::move(allocation);
        impl_->greedy_transfers=std::move(transfers);
        impl_->persistent_bytes=next_impl;persistent_bytes_=next_public;
    }
}
Exl3GreedyPacket Exl3TextContext::greedy_packet(bool continuation, cudaStream_t stream) {
    const auto* source=continuation ? continuation_logits_device() : logits_device();
    const int rows=continuation ? continuation_rows() : 1;
    require(source && rows>0 && rows<=16 && impl_->greedy_rows && !impl_->greedy_readback_failed,
        "greedy packet unprepared/failed/score extent");
    require(impl_->greedy_serial!=std::numeric_limits<std::uint64_t>::max(),
            "greedy packet serial exhausted");
    Exl3GreedyPacket result;
    result.rows=rows;result.position=position_;result.generation=impl_->request_generation;
    result.serial=++impl_->greedy_serial;
    auto* output=static_cast<Exl3GreedyRow*>(impl_->greedy_rows->ptr);
    try {
        exl3_launch_greedy_packet(impl_->gaming[Gopt::GreedyWarp],rows,stream,source,kVocab,kVocab,result.serial,output);
        cuda_check(cudaGetLastError(),"greedy packet reduction");
        if(impl_->gaming[Gopt::GreedyWarp])gopt_record(impl_->gaming_submissions,Gopt::GreedyWarp);
        // Context-owned destination survives exceptions/partial transfers. No
        // stack packet escapes or is reused after a failed completion fence.
        cuda_check(cudaMemcpyAsync(impl_->greedy_host_rows.data(),output,rows*sizeof(Exl3GreedyRow),
            cudaMemcpyDeviceToHost,stream),"greedy packet download");
        cuda_check(cudaStreamSynchronize(stream),"greedy packet readiness");
        if(continuation)
            require(impl_->complete_continuation_graph_use(rows,stream),
                "continuation graph greedy completion lost final-use ownership");
    } catch(...) {impl_->greedy_readback_failed=true;throw;}
    std::copy_n(impl_->greedy_host_rows.begin(),rows,result.decisions.begin());
    impl_->greedy_readback.packet_bytes+=rows*sizeof(Exl3GreedyRow);
    impl_->greedy_readback.packet_rows+=rows;++impl_->greedy_readback.packet_calls;
    result.ready=true;
    result.validate(rows,impl_->request_generation,position_,impl_->greedy_serial);
    return result;
}

Exl3PendingGreedyPacket Exl3TextContext::submit_greedy_packet(
    bool continuation,std::uint64_t acquisition,std::uint64_t execution,
    cudaStream_t stream,bool defer_host_readback) {
    const auto* source=continuation?continuation_logits_device():logits_device();
    const int rows=continuation?continuation_rows():1;
    require(source && rows>0 && rows<=16 && acquisition && execution &&
                !impl_->greedy_readback_failed &&
                impl_->greedy_serial!=std::numeric_limits<std::uint64_t>::max() &&
                (!stream || (impl_->execution_stream_owner &&
                    stream==impl_->owned_execution_stream)) &&
                !Exl3GreedyPacketTransfer::quarantined.load(std::memory_order_acquire),
            "pending greedy packet owner/scope unavailable");
    Exl3PendingGreedyPacket pending;
    std::shared_ptr<Exl3GreedyPacketTransfer> transfer;
    for(const auto& candidate:impl_->greedy_transfers) {
        // The context's pool reference must be the only remaining owner. This
        // excludes the Pending handle and every retained seed/batch diagnostic.
        if(!candidate || candidate.use_count()!=1)continue;
        const auto error=candidate->recycle();
        if(error) {
            impl_->greedy_readback_failed=true;
            throw std::runtime_error("pending greedy packet pool recycle failed");
        }
        transfer=candidate;break;
    }
    if(!transfer)throw Exl3ResourceReservationExhausted{};
    pending.acquisition_=acquisition;pending.execution_=execution;
    pending.generation_=impl_->request_generation;pending.position_=position_;
    pending.rows_=rows;pending.serial_=++impl_->greedy_serial;
    pending.continuation_=continuation;
    transfer->stream=stream;transfer->submitted=true;
    transfer->model_owner=metadata_owner();
    transfer->stream_owner=stream?impl_->execution_stream_owner:
        std::shared_ptr<const void>{};
    transfer->completion_generation=transfer->final_use.begin(
        acquisition,execution,reinterpret_cast<std::uintptr_t>(
            defer_host_readback?transfer->consumer_event:transfer->event));
    try {
        exl3_launch_greedy_packet(impl_->gaming[Gopt::GreedyWarp],rows,stream,source,kVocab,kVocab,
            pending.serial_,static_cast<Exl3GreedyRow*>(transfer->device->ptr));
        cuda_check(cudaGetLastError(),"pending greedy packet reduction");
        if(impl_->gaming[Gopt::GreedyWarp])gopt_record(impl_->gaming_submissions,Gopt::GreedyWarp);
        cuda_check(cudaEventRecord(transfer->device_event,stream),
            "pending greedy packet device readiness event");
        if(!defer_host_readback) {
            transfer->host_readback_submitted=true;
            cuda_check(cudaMemcpyAsync(transfer->host,transfer->device->ptr,
                rows*sizeof(Exl3GreedyRow),cudaMemcpyDeviceToHost,stream),
                "pending greedy packet download");
            cuda_check(cudaEventRecord(transfer->event,stream),
                "pending greedy packet readiness event");
            transfer->event_recorded=true;
        }
    } catch(...) {
        impl_->greedy_readback_failed=true;
        throw;
    }
    pending.owner_=std::move(transfer);
    return pending;
}

Exl3DeviceGreedySeed Exl3TextContext::device_greedy_seed(
    const Exl3PendingGreedyPacket& pending,std::uint64_t acquisition,
    std::uint64_t execution) const {
    require(pending.owner_ && pending.rows_==1 && !pending.continuation_ &&
                acquisition && execution &&
                pending.acquisition_==acquisition && pending.execution_==execution &&
                pending.generation_==impl_->request_generation &&
                pending.position_==position_ && pending.serial_==impl_->greedy_serial &&
                !pending.owner_->completed && pending.owner_->host_readback_submitted &&
                pending.owner_->event_recorded,
            "device greedy seed stale/incomplete scope");
    Exl3DeviceGreedySeed seed;
    seed.owner=pending.owner_;
    seed.model_owner=pending.owner_->model_owner;
    seed.model_identity=model_identity();
    seed.device_row=static_cast<const Exl3GreedyRow*>(pending.owner_->device->ptr);
    seed.target_embedding_bf16=impl_->model->embedding;
    seed.target_head=impl_->model->lm_head;
    seed.target_head_metadata=impl_->model->lm_head_metadata;
    seed.producer_ready=pending.owner_->device_event;
    seed.consumer_done=pending.owner_->consumer_event;
    seed.consumer_claimed=&pending.owner_->consumer_claimed;
    seed.consumer_recorded=&pending.owner_->consumer_recorded;
    seed.consumer_stream=&pending.owner_->consumer_stream;
    seed.acquisition=acquisition;seed.execution=execution;
    seed.generation=pending.generation_;seed.serial=pending.serial_;
    seed.position=pending.position_;
    return seed;
}

Exl3GreedyPacket Exl3TextContext::finish_greedy_packet(
    Exl3PendingGreedyPacket&& pending,std::uint64_t acquisition,
    std::uint64_t execution) {
    auto owner=std::move(pending.owner_);
    require(owner && owner->host_readback_submitted && owner->event_recorded &&
                acquisition && execution &&
                pending.acquisition_==acquisition &&
                pending.execution_==execution && pending.rows_>0 &&
                pending.rows_<=16 && pending.generation_==impl_->request_generation &&
                pending.position_==position_ && pending.serial_==impl_->greedy_serial,
            "stale pending greedy packet");
    const auto error=owner->wait();
    if(error) {
        if(pending.continuation_)
            (void)impl_->complete_continuation_graph_use(
                pending.rows_,owner->stream,error);
        impl_->greedy_readback_failed=true;
        for(auto& slot:impl_->greedy_transfers)if(slot==owner){slot.reset();break;}
        throw std::runtime_error("pending greedy packet completion failed");
    }
    if(pending.continuation_)
        require(impl_->complete_continuation_graph_use(
            pending.rows_,owner->stream),
            "pending continuation graph completion lost final-use ownership");
    Exl3GreedyPacket result;
    result.rows=pending.rows_;result.position=pending.position_;
    result.generation=pending.generation_;result.serial=pending.serial_;
    std::copy_n(owner->host,pending.rows_,result.decisions.begin());
    result.ready=true;
    result.validate(pending.rows_,pending.generation_,pending.position_,pending.serial_);
    impl_->greedy_readback.packet_bytes+=pending.rows_*sizeof(Exl3GreedyRow);
    impl_->greedy_readback.packet_rows+=pending.rows_;
    ++impl_->greedy_readback.packet_calls;
    return result;
}

Exl3GreedyBatchSource Exl3TextContext::greedy_batch_source(
    const Exl3PendingGreedyPacket& pending,std::uint64_t acquisition,
    std::uint64_t execution) const {
    require(pending.owner_ && pending.rows_>0 && pending.rows_<=16 &&
                !pending.owner_->host_readback_submitted &&
                !pending.owner_->event_recorded && !pending.owner_->completed &&
                pending.owner_->device_event && acquisition && execution &&
                pending.acquisition_==acquisition && pending.execution_==execution &&
                pending.generation_==impl_->request_generation &&
                pending.position_==position_ && pending.serial_==impl_->greedy_serial,
            "batched greedy source stale/incompatible scope");
    Exl3GreedyBatchSource source;
    source.owner=pending.owner_;source.model_owner=pending.owner_->model_owner;
    source.model_identity=model_identity();
    source.stream_owner=pending.owner_->stream_owner;
    source.producer_stream=pending.owner_->stream;
    source.device_row=static_cast<const Exl3GreedyRow*>(pending.owner_->device->ptr);
    source.producer_ready=pending.owner_->device_event;
    source.consumer_done=pending.owner_->consumer_event;
    source.consumer_claimed=&pending.owner_->consumer_claimed;
    source.consumer_recorded=&pending.owner_->consumer_recorded;
    source.consumer_stream=&pending.owner_->consumer_stream;
    source.acquisition=acquisition;source.execution=execution;
    source.generation=pending.generation_;source.serial=pending.serial_;
    source.final_use_generation=pending.owner_->completion_generation;
    source.final_use_event=reinterpret_cast<std::uintptr_t>(
        pending.owner_->consumer_event);
    source.device_bytes=Exl3GreedyPacketTransfer::row_bytes;
    source.registered_host_bytes=Exl3GreedyPacketTransfer::row_bytes;
    source.metadata_bytes=Exl3GreedyPacketTransfer::metadata_bytes_required();
    source.position=pending.position_;source.rows=pending.rows_;
    require(source.owns_pending_execution(),
        "batched greedy source final-use ownership incomplete");
    return source;
}

Exl3GreedyPacket Exl3TextContext::finish_batched_greedy_packet(
    Exl3PendingGreedyPacket&& pending,std::span<const Exl3GreedyRow> rows,
    std::uint64_t acquisition,std::uint64_t execution) {
    auto owner=std::move(pending.owner_);
    require(owner && !owner->host_readback_submitted && !owner->event_recorded &&
                owner->consumer_claimed.load(std::memory_order_acquire) &&
                owner->consumer_recorded.load(std::memory_order_acquire) &&
                acquisition && execution && pending.acquisition_==acquisition &&
                pending.execution_==execution && pending.rows_>0 && pending.rows_<=16 &&
                rows.size()==static_cast<std::size_t>(pending.rows_) &&
                pending.generation_==impl_->request_generation && pending.position_==position_ &&
                pending.serial_==impl_->greedy_serial,
            "stale/incomplete batched greedy result");
    const auto error=cudaEventSynchronize(owner->consumer_event);
    require(owner->final_use.finish(owner->completion_generation,
                static_cast<int>(error)),
        "batched greedy packet final-use identity mismatch");
    owner->completed=true;
    if(error){
        if(pending.continuation_)
            (void)impl_->complete_continuation_graph_use(
                pending.rows_,owner->stream,static_cast<int>(error));
        impl_->greedy_readback_failed=true;
        for(auto& slot:impl_->greedy_transfers)if(slot==owner){slot.reset();break;}
        throw std::runtime_error("batched greedy packet completion failed");
    }
    if(pending.continuation_)
        require(impl_->complete_continuation_graph_use(
            pending.rows_,owner->stream),
            "batched continuation graph completion lost final-use ownership");
    Exl3GreedyPacket result;result.rows=pending.rows_;result.position=pending.position_;
    result.generation=pending.generation_;result.serial=pending.serial_;
    std::copy(rows.begin(),rows.end(),result.decisions.begin());result.ready=true;
    result.validate(pending.rows_,pending.generation_,pending.position_,pending.serial_);
    impl_->greedy_readback.packet_bytes+=pending.rows_*sizeof(Exl3GreedyRow);
    impl_->greedy_readback.packet_rows+=pending.rows_;++impl_->greedy_readback.packet_calls;
    return result;
}

std::uint64_t Exl3TextContext::quarantined_greedy_packet_transfers() noexcept {
    return Exl3GreedyPacketTransfer::quarantined.load(std::memory_order_acquire);
}

std::size_t Exl3TextContext::greedy_transfer_pool_available_for_test() const noexcept {
    return std::count_if(impl_->greedy_transfers.begin(),impl_->greedy_transfers.end(),
        [](const auto& owner){return owner && owner.use_count()==1;});
}
std::uintptr_t Exl3TextContext::greedy_transfer_identity_for_test(
    const Exl3PendingGreedyPacket& pending) noexcept {
    return reinterpret_cast<std::uintptr_t>(pending.owner_.get());
}

void Exl3TextContext::fail_pending_greedy_packet_completion_for_test(
    Exl3PendingGreedyPacket& pending) {
    require(pending.owner_ && !pending.owner_->completed,
            "pending greedy packet failure injection requires live owner");
    pending.owner_->completion_failure_for_test=true;
}

Exl3GreedyPacket Exl3TextContext::greedy_packet_from_scores_for_test(
    std::span<const std::uint16_t> scores,int rows,int vocabulary,int stride,
    cudaStream_t stream) {
    require(rows>0 && rows<=16 && vocabulary>0 && stride>=vocabulary &&
                scores.size()==static_cast<std::size_t>(rows)*stride &&
                impl_->greedy_rows && !impl_->greedy_readback_failed,
            "greedy packet score fixture extent");
    require(impl_->greedy_serial!=std::numeric_limits<std::uint64_t>::max(),
            "greedy packet serial exhausted");
    DeviceAllocation represented(std::as_bytes(scores),
        "allocate greedy packet represented score fixture");
    Exl3GreedyPacket result;
    result.rows=rows;result.position=position_;result.generation=impl_->request_generation;
    result.serial=++impl_->greedy_serial;
    auto* output=static_cast<Exl3GreedyRow*>(impl_->greedy_rows->ptr);
    try {
        exl3_launch_greedy_packet(impl_->gaming[Gopt::GreedyWarp],rows,stream,
            static_cast<const std::uint16_t*>(represented.ptr),vocabulary,stride,
            result.serial,output);
        cuda_check(cudaGetLastError(),"greedy packet fixture reduction");
        if(impl_->gaming[Gopt::GreedyWarp])gopt_record(impl_->gaming_submissions,Gopt::GreedyWarp);
        cuda_check(cudaMemcpyAsync(impl_->greedy_host_rows.data(),output,
            rows*sizeof(Exl3GreedyRow),cudaMemcpyDeviceToHost,stream),
            "greedy packet fixture download");
        cuda_check(cudaStreamSynchronize(stream),"greedy packet fixture readiness");
    } catch(...) {impl_->greedy_readback_failed=true;throw;}
    std::copy_n(impl_->greedy_host_rows.begin(),rows,result.decisions.begin());
    result.ready=true;
    result.validate(rows,impl_->request_generation,position_,impl_->greedy_serial,
                    vocabulary);
    return result;
}

Exl3TextContext::GreedyReadbackStats Exl3TextContext::greedy_readback_stats() const noexcept {
    return impl_->greedy_readback;
}

bool Exl3TextContext::capture_continuation_graph(cudaStream_t stream) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    if(impl_->continuation)
        ++impl_->continuation->graph_stats.preparation_attempts;
    auto fail = [&](const std::string& reason) {
        if (impl_->continuation) {
            ++impl_->continuation->graph_stats.eager_fallbacks;
            impl_->continuation->graph_active = std::any_of(
                impl_->continuation->additional_graphs.begin(),
                impl_->continuation->additional_graphs.end(),
                [](const auto& graph) { return graph.active; });
            impl_->continuation->graph_split_class = 0;
            impl_->continuation->graph_active_generation = 0;
            impl_->continuation->graph_origin_stream = nullptr;
            impl_->continuation->graph_route_bits = 0;
            impl_->continuation->graph_compatibility.invalidate();
            impl_->continuation->graph_lifecycle.invalidate(reason);
            impl_->continuation->graph_reason = reason;
        }
        return false;
    };
    if (!impl_->continuation || impl_->continuation->capacity != 8)
        return fail("fixed-B8 continuation was not prepared");
    if (!impl_->continuation_graph_b8_enabled)
        return fail("fixed-B8 continuation graph flag is disabled");
    if(!impl_->continuation_graph_menu_admits(8))
        return fail(impl_->continuation->graph_menu_reason);
    if (!impl_->transaction || impl_->transaction->active ||
        impl_->transaction->rollback_required)
        return fail("fixed-B8 continuation graph requires an idle prepared transaction");
    if (!impl_->capture_taps || !impl_->oscar || impl_->host_kv.enabled ||
        impl_->graph_active || impl_->graph_capture_active ||
        impl_->oscar->graph_class() != 0)
        return fail("fixed-B8 continuation graph requires eager canonical OSCAR");
    if (impl_->target_projection_observer || impl_->target_projection_timing)
        return fail("fixed-B8 continuation graph rejects projection instrumentation");
    const auto numerical_boundary=impl_->continuation_numerical_boundary(8,stream);
    const auto numerical_assessment=numerical_boundary.assess();
    if(!numerical_assessment.eligible())
        return fail(std::string(Exl3GraphNumericalBoundary::reason(
            numerical_assessment.reason)));
    if (position_ <= 0 || position_ > impl_->max_context - 8)
        return fail("fixed-B8 continuation graph position extent");
    const int split_class = ninfer::ops::detail::
        oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(position_ + 1);
    for (int row = 1; row < 8; ++row) {
        if (ninfer::ops::detail::
                oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
                    position_ + row + 1) != split_class)
            return fail("fixed-B8 continuation crosses an OSCAR split-class boundary");
    }
    auto& continuation = *impl_->continuation;
    const auto projection_options=Exl3ProjectionGraphOptions::current();
    if(!projection_options.valid)return fail("projection graph options exceed identity capacity");
    const auto route_bits=impl_->continuation_graph_route(true);
    const auto compatibility=impl_->continuation_graph_fingerprint(8,split_class,
        route_bits,continuation.graph_active_generation,stream,projection_options);
    const bool projection_binding_matches=continuation.projection_binding.matches(
        impl_->model,impl_->reconstruction_backing,projection_options);
    const bool compatibility_matches=continuation.graph_compatibility.matches(compatibility) &&
        continuation.graph_lifecycle.admits(compatibility,
            continuation.graph_active_generation);
    if (projection_binding_matches && compatibility_matches && continuation.graph_active && continuation.graph_split_class == split_class &&
        continuation.graph_origin_stream == stream) {
        ++continuation.graph_stats.compatible_active_hits;
        return true;
    }
    if (projection_binding_matches && compatibility_matches && !continuation.graph_active && continuation.graph_definition.ready() &&
        continuation.graph_executable.ready() &&
        continuation.graph_active_generation!=0 &&
        continuation.graph_split_class==split_class &&
        continuation.graph_origin_stream==stream) {
        // reset_for_request has already drained every stream. A direct reset
        // followed by same-stream work is ordered by that stream. No graph
        // object or captured pointer is changed on this reactivation path.
        continuation.graph_active=true;
        ++continuation.graph_compatible_reuse_count;
        ++continuation.graph_stats.compatible_reactivations;
        continuation.graph_reason=
            "active: compatible fixed-B8 graph reused after context reset";
        return true;
    }
    const int saved_tap_rows = impl_->tap_rows;
    const int saved_embedding_rows = impl_->embedding_rows;
    const int saved_last_rows = impl_->last_rows;
    const bool saved_qkv_trace_valid = impl_->qkv_trace_valid;
    const auto clear_layer_capture_state = [&] {
        for (auto& layer : impl_->full_layers)
            if (layer) layer->set_capture_active(false);
        for (auto& layer : impl_->gdn_layers)
            if (layer) layer->set_capture_active(false);
    };
    try {
        // An incompatible cached key requires object replacement. The prior
        // executable may have been launched on another stream, so fence all
        // device work before capture() destroys its definition and
        // instantiate() destroys its executable.
        if(continuation.graph_definition.ready() ||
            continuation.graph_executable.ready()) {
            cuda_check(cudaDeviceSynchronize(),
                "drain incompatible fixed-B8 graph before recapture");
            if(continuation.graph_lifecycle.pending())
                require(impl_->complete_continuation_graph_use(
                    8,continuation.graph_origin_stream),
                    "fixed-B8 graph final-use drain identity mismatch");
            continuation.graph_lifecycle.invalidate(
                "fixed-B8 compatibility changed");
        }
        cuda_check(cudaStreamSynchronize(stream),
                   "synchronize before fixed-B8 continuation capture");
        impl_->bind_graph_device();
        const auto retired=ninfer::detail::retire_decode_graph_pair(
            continuation.graph_executable,continuation.graph_definition);
        if(retired!=cudaSuccess) {
            continuation.graph_active=false;
            continuation.projection_binding.invalidate();
            continuation.graph_compatibility.invalidate();
            continuation.graph_lifecycle.quarantine(
                "fixed-B8 graph replacement destruction failed",retired);
            impl_->poison_graph_retirement(retired);
            cuda_check(retired,"retire incompatible fixed-B8 graph");
        }
        if(continuation.graph_lifecycle.snapshot().phase==
                Exl3BoundedGraphEntry::Phase::invalidated)
            require(continuation.graph_lifecycle.release_after_destroy(),
                "fixed-B8 graph released resources before final handle destruction");
        impl_->publish_continuation_graph_roles(8,stream,true);
        impl_->oscar->set_graph_class(split_class);
        impl_->graph_capture_active = true;
        continuation.graph_definition.capture(stream, [&] {
            graph_role_embedding_kernel<<<(8*kHidden+255)/256,256,0,stream>>>(
                impl_->continuation_graph_role_record(8),
                impl_->model->embedding,8);
            cuda_check(cudaGetLastError(),
                "capture fixed-B8 role-table embedding");
            impl_->process_rows(nullptr,8,position_,stream,nullptr,false,true,true);
        });
        require(projection_options.matches(Exl3ProjectionGraphOptions::current()),
            "projection options changed during fixed-B8 capture");
        continuation.graph_executable.instantiate(continuation.graph_definition);
        continuation.graph_executable.upload(stream);
        cuda_check(cudaStreamSynchronize(stream),
                   "complete fixed-B8 continuation graph upload");
        impl_->graph_capture_active = false;
        clear_layer_capture_state();
        impl_->oscar->set_graph_class(0);
        impl_->tap_rows = saved_tap_rows;
        impl_->embedding_rows = saved_embedding_rows;
        impl_->last_rows = saved_last_rows;
        impl_->qkv_trace_valid = saved_qkv_trace_valid;
        continuation.rows = 0;
        continuation.graph_split_class = split_class;
        continuation.graph_origin_stream = stream;
        continuation.projection_binding.bind(impl_->model,impl_->reconstruction_backing,projection_options);
        continuation.graph_route_bits = route_bits;
        continuation.graph_active_generation =
            ++continuation.graph_generation_counter;
        continuation.graph_compatibility=impl_->continuation_graph_fingerprint(
            8,split_class,route_bits,continuation.graph_active_generation,
            stream,projection_options);
        require(continuation.graph_compatibility.valid(),
            "fixed-B8 graph compatibility fingerprint is incomplete");
        impl_->bind_continuation_graph_lifecycle(continuation.graph_lifecycle,
            continuation.graph_compatibility,8,
            continuation.graph_active_generation);
        ++continuation.graph_capture_count;
        ++continuation.graph_stats.preparation_successes;
        continuation.graph_active = true;
        continuation.graph_reason = impl_->oscar->continuation_cohort_b8_enabled()
            ? "active: fixed-B8 chronological OSCAR cohort + layer stack + taps + final norm + H6 head"
            : "active: fixed-B8 sequential OSCAR + layer stack + taps + final norm + H6 head";
        if (impl_->continuation_graph_gdn_qkvz_concurrent)
            continuation.graph_reason +=
                " + sibling GDN QKV/Z projections; "
                "gdn_qkvz_private_workspace_bytes=" +
                std::to_string(impl_->gdn_qkvz_private_workspace_bytes) +
                "; gdn_qkvz_layers=" +
                std::to_string(impl_->gdn_qkvz_layer_count);
        else
            continuation.graph_reason += " + sequential GDN QKV/Z";
        return true;
    } catch (const std::exception& error) {
        impl_->graph_capture_active = false;
        clear_layer_capture_state();
        impl_->oscar->set_graph_class(0);
        impl_->tap_rows = saved_tap_rows;
        impl_->embedding_rows = saved_embedding_rows;
        impl_->last_rows = saved_last_rows;
        impl_->qkv_trace_valid = saved_qkv_trace_valid;
        continuation.rows = 0;
        if(impl_->graph_retirement_error)throw; // Never retry a failed native destroy.
        const auto retirement=cudaDeviceSynchronize();
        if(retirement!=cudaSuccess) {
            continuation.graph_active=false;
            continuation.projection_binding.invalidate();
            continuation.graph_compatibility.invalidate();
            continuation.graph_lifecycle.quarantine(
                "fixed-B8 graph drain failed",retirement);
            impl_->poison_graph_retirement(retirement);
            throw; // Preserve the capture error; eager fallback is unsafe.
        }
        if(continuation.graph_lifecycle.pending())
            (void)impl_->complete_continuation_graph_use(
                8,continuation.graph_origin_stream);
        continuation.graph_lifecycle.invalidate("fixed-B8 capture failed");
        const auto destroyed=ninfer::detail::retire_decode_graph_pair(
            continuation.graph_executable,continuation.graph_definition);
        if(destroyed!=cudaSuccess) {
            continuation.graph_active=false;
            continuation.projection_binding.invalidate();
            continuation.graph_compatibility.invalidate();
            continuation.graph_lifecycle.quarantine(
                "fixed-B8 graph destruction failed",destroyed);
            impl_->poison_graph_retirement(destroyed);
            throw;
        }
        if(continuation.graph_lifecycle.snapshot().phase==
                Exl3BoundedGraphEntry::Phase::invalidated)
            require(continuation.graph_lifecycle.release_after_destroy(),
                "fixed-B8 failed capture retained released graph resources");
        return fail(error.what());
    }
}

bool Exl3TextContext::capture_continuation_graph_rows(
    int rows,cudaStream_t stream) {
    Impl::require_host_kv_retirement_admission();
    if (rows == 8) return capture_continuation_graph(stream);
    if (!impl_->continuation || (rows != 4 && rows != 6)) return false;
    auto& continuation = *impl_->continuation;
    ++continuation.graph_stats.preparation_attempts;
    auto& graph = continuation.additional_graphs[rows == 4 ? 0 : 1];
    auto fail = [&](const std::string& reason) {
        ++continuation.graph_stats.eager_fallbacks;
        graph.active = false;
        graph.split_class = 0;
        graph.active_generation = 0;
        graph.origin_stream = nullptr;
        graph.route_bits = 0;
        graph.compatibility.invalidate();
        graph.lifecycle.invalidate(reason);
        graph.reason = reason;
        continuation.graph_active = continuation.graph_active ||
            std::any_of(continuation.additional_graphs.begin(),
                        continuation.additional_graphs.end(),
                        [](const auto& candidate) { return candidate.active; });
        return false;
    };
    if (continuation.capacity != 8)
        return fail("multi-depth continuation requires prepared B8 capacity");
    if (!impl_->continuation_graph_b8_enabled)
        return fail("multi-depth continuation graph flag is disabled");
    if(!impl_->continuation_graph_menu_admits(static_cast<unsigned>(rows)))
        return fail(continuation.graph_menu_reason);
    if (!impl_->transaction || impl_->transaction->active ||
        impl_->transaction->rollback_required)
        return fail("multi-depth continuation graph requires an idle prepared transaction");
    if (!impl_->capture_taps || !impl_->oscar || impl_->host_kv.enabled ||
        impl_->graph_active || impl_->graph_capture_active ||
        impl_->oscar->graph_class() != 0)
        return fail("multi-depth continuation graph requires eager canonical OSCAR");
    if (impl_->target_projection_observer || impl_->target_projection_timing)
        return fail("multi-depth continuation graph rejects projection instrumentation");
    const auto numerical_boundary=impl_->continuation_numerical_boundary(
        static_cast<unsigned>(rows),stream);
    const auto numerical_assessment=numerical_boundary.assess();
    if(!numerical_assessment.eligible())
        return fail(std::string(Exl3GraphNumericalBoundary::reason(
            numerical_assessment.reason)));
    if (position_ <= 0 || position_ > impl_->max_context - rows)
        return fail("multi-depth continuation graph position extent");
    const int split_class = ninfer::ops::detail::
        oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(position_ + 1);
    for (int row = 1; row < rows; ++row) {
        if (ninfer::ops::detail::
                oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
                    position_ + row + 1) != split_class)
            return fail("multi-depth continuation crosses an OSCAR split-class boundary");
    }
    const auto projection_options=Exl3ProjectionGraphOptions::current();
    if(!projection_options.valid)return fail("projection graph options exceed identity capacity");
    const auto route_bits=impl_->continuation_graph_route(false);
    const auto compatibility=impl_->continuation_graph_fingerprint(
        static_cast<unsigned>(rows),split_class,route_bits,
        graph.active_generation,stream,projection_options);
    const bool projection_binding_matches=graph.projection_binding.matches(
        impl_->model,impl_->reconstruction_backing,projection_options);
    const bool compatibility_matches=graph.compatibility.matches(compatibility) &&
        graph.lifecycle.admits(compatibility,graph.active_generation);
    if (projection_binding_matches && compatibility_matches && graph.active && graph.split_class == split_class &&
        graph.origin_stream == stream) {
        ++continuation.graph_stats.compatible_active_hits;
        continuation.graph_active = true;
        return true;
    }
    if (projection_binding_matches && compatibility_matches && !graph.active && graph.definition.ready() && graph.executable.ready() &&
        graph.active_generation != 0 && graph.split_class == split_class &&
        graph.origin_stream == stream) {
        graph.active = true;
        ++graph.compatible_reuse_count;
        ++continuation.graph_stats.compatible_reactivations;
        graph.reason = "active: compatible multi-depth graph reused after context reset";
        continuation.graph_active = true;
        return true;
    }
    const int saved_tap_rows = impl_->tap_rows;
    const int saved_embedding_rows = impl_->embedding_rows;
    const int saved_last_rows = impl_->last_rows;
    const bool saved_qkv_trace_valid = impl_->qkv_trace_valid;
    const auto clear_layer_capture_state = [&] {
        for (auto& layer : impl_->full_layers)
            if (layer) layer->set_capture_active(false);
        for (auto& layer : impl_->gdn_layers)
            if (layer) layer->set_capture_active(false);
    };
    try {
        if (graph.definition.ready() || graph.executable.ready()) {
            cuda_check(cudaDeviceSynchronize(),
                       "drain incompatible multi-depth graph before recapture");
            if(graph.lifecycle.pending())
                require(impl_->complete_continuation_graph_use(
                    rows,graph.origin_stream),
                    "multi-depth graph final-use drain identity mismatch");
            graph.lifecycle.invalidate("multi-depth compatibility changed");
        }
        cuda_check(cudaStreamSynchronize(stream),
                   "synchronize before multi-depth continuation capture");
        impl_->bind_graph_device();
        const auto retired=ninfer::detail::retire_decode_graph_pair(graph.executable,graph.definition);
        if(retired!=cudaSuccess) {
            graph.active=false;
            continuation.graph_active=false;
            graph.projection_binding.invalidate();
            graph.compatibility.invalidate();
            graph.lifecycle.quarantine(
                "multi-depth graph replacement destruction failed",retired);
            impl_->poison_graph_retirement(retired);
            cuda_check(retired,"retire incompatible multi-depth graph");
        }
        if(graph.lifecycle.snapshot().phase==
                Exl3BoundedGraphEntry::Phase::invalidated)
            require(graph.lifecycle.release_after_destroy(),
                "multi-depth graph released resources before final handle destruction");
        impl_->publish_continuation_graph_roles(
            static_cast<unsigned>(rows),stream,true);
        impl_->oscar->set_graph_class(split_class);
        impl_->graph_capture_active = true;
        graph.definition.capture(stream, [&] {
            graph_role_embedding_kernel<<<(rows*kHidden+255)/256,256,0,stream>>>(
                impl_->continuation_graph_role_record(
                    static_cast<unsigned>(rows)),impl_->model->embedding,rows);
            cuda_check(cudaGetLastError(),
                "capture multi-depth role-table embedding");
            impl_->process_rows(nullptr,rows,position_,stream,nullptr,false,true,true);
        });
        require(projection_options.matches(Exl3ProjectionGraphOptions::current()),
            "projection options changed during multi-depth capture");
        graph.executable.instantiate(graph.definition);
        graph.executable.upload(stream);
        cuda_check(cudaStreamSynchronize(stream),
                   "complete multi-depth continuation graph upload");
        impl_->graph_capture_active = false;
        clear_layer_capture_state();
        impl_->oscar->set_graph_class(0);
        impl_->tap_rows = saved_tap_rows;
        impl_->embedding_rows = saved_embedding_rows;
        impl_->last_rows = saved_last_rows;
        impl_->qkv_trace_valid = saved_qkv_trace_valid;
        continuation.rows = 0;
        graph.split_class = split_class;
        graph.origin_stream = stream;
        graph.projection_binding.bind(impl_->model,impl_->reconstruction_backing,projection_options);
        graph.route_bits = route_bits;
        graph.active_generation = ++continuation.graph_generation_counter;
        graph.compatibility=impl_->continuation_graph_fingerprint(
            static_cast<unsigned>(rows),split_class,route_bits,
            graph.active_generation,stream,projection_options);
        require(graph.compatibility.valid(),
            "multi-depth graph compatibility fingerprint is incomplete");
        impl_->bind_continuation_graph_lifecycle(graph.lifecycle,
            graph.compatibility,static_cast<unsigned>(rows),
            graph.active_generation);
        ++graph.capture_count;
        ++continuation.graph_stats.preparation_successes;
        graph.active = true;
        graph.reason = "active: fixed-B" + std::to_string(rows) +
            (impl_->oscar->continuation_cohort_b8_enabled()
                ? " chronological OSCAR cohort"
                : " sequential OSCAR") +
            " + layer stack + taps + final norm + H6 head";
        if (impl_->continuation_graph_gdn_qkvz_concurrent)
            graph.reason += " + sibling GDN QKV/Z projections";
        else
            graph.reason += " + sequential GDN QKV/Z";
        continuation.graph_active = true;
        return true;
    } catch (const std::exception& error) {
        impl_->graph_capture_active = false;
        clear_layer_capture_state();
        impl_->oscar->set_graph_class(0);
        impl_->tap_rows = saved_tap_rows;
        impl_->embedding_rows = saved_embedding_rows;
        impl_->last_rows = saved_last_rows;
        impl_->qkv_trace_valid = saved_qkv_trace_valid;
        continuation.rows = 0;
        if(impl_->graph_retirement_error)throw; // Preserve uncertain handles without retry.
        const auto retirement=cudaDeviceSynchronize();
        if(retirement!=cudaSuccess) {
            graph.active=false;
            continuation.graph_active=false;
            graph.projection_binding.invalidate();
            graph.compatibility.invalidate();
            graph.lifecycle.quarantine("multi-depth graph drain failed",retirement);
            impl_->poison_graph_retirement(retirement);
            throw;
        }
        if(graph.lifecycle.pending())
            (void)impl_->complete_continuation_graph_use(rows,graph.origin_stream);
        graph.lifecycle.invalidate("multi-depth capture failed");
        const auto destroyed=ninfer::detail::retire_decode_graph_pair(graph.executable,graph.definition);
        if(destroyed!=cudaSuccess) {
            graph.active=false;
            continuation.graph_active=false;
            graph.projection_binding.invalidate();
            graph.compatibility.invalidate();
            graph.lifecycle.quarantine("multi-depth graph destruction failed",destroyed);
            impl_->poison_graph_retirement(destroyed);
            throw;
        }
        if(graph.lifecycle.snapshot().phase==Exl3BoundedGraphEntry::Phase::invalidated)
            require(graph.lifecycle.release_after_destroy(),
                "multi-depth failed capture retained released graph resources");
        return fail(error.what());
    }
}

void Exl3TextContext::continue_rows_graph(
    std::span<const std::int64_t> token_ids,cudaStream_t stream) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    const int rows = static_cast<int>(token_ids.size());
    require(impl_->continuation && impl_->continuation->graph_active &&
                impl_->continuation->capacity == 8 &&
                (rows == 4 || rows == 6 || rows == 8),
            "continuation graph row variant is not active");
    auto& continuation = *impl_->continuation;
    auto* additional = rows == 8 ? nullptr :
        &continuation.additional_graphs[rows == 4 ? 0 : 1];
    const bool selected_active = rows == 8
        ? continuation.graph_definition.ready() &&
            continuation.graph_executable.ready()
        : additional->active && additional->definition.ready() &&
            additional->executable.ready();
    const cudaStream_t selected_stream = rows == 8
        ? continuation.graph_origin_stream : additional->origin_stream;
    const int selected_split_class = rows == 8
        ? continuation.graph_split_class : additional->split_class;
    const std::uint32_t selected_route_bits = rows == 8
        ? continuation.graph_route_bits : additional->route_bits;
    const std::uint64_t selected_generation = rows == 8
        ? continuation.graph_active_generation : additional->active_generation;
    require(selected_active && selected_stream == stream,
            "selected continuation graph row variant is not active on this stream");
    require((rows==8?continuation.projection_binding:additional->projection_binding).matches(
                impl_->model,impl_->reconstruction_backing),
            "continuation graph projection backing changed; recapture required");
    require(impl_->transaction && impl_->transaction->active &&
                impl_->transaction->fresh_snapshot &&
                !impl_->transaction->rollback_required &&
                impl_->transaction->stream == stream &&
                selected_stream == stream,
            "continuation graph requires a fresh same-stream transaction");
    require(position_ == impl_->transaction->position &&
                position_ <= impl_->max_context - rows && impl_->oscar &&
                impl_->continuation_graph_b8_enabled &&
                impl_->oscar->graph_class() == 0 && !impl_->graph_active &&
                !impl_->graph_capture_active,
            "continuation graph state mismatch");
    require(std::all_of(token_ids.begin(),token_ids.end(),[](std::int64_t token) {
                return token >= 0 && token < kVocab;
            }),"continuation graph token extent");
    const int split_class = ninfer::ops::detail::
        oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(position_ + 1);
    for (int row = 1; row < rows; ++row)
        require(ninfer::ops::detail::
                    oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
                        position_ + row + 1) == split_class,
                "continuation graph replay crosses OSCAR split class");
    require(split_class == selected_split_class,
            "continuation graph requires recapture for OSCAR split class");
    const auto current_route_bits=impl_->continuation_graph_route(rows==8);
    const auto compatibility=impl_->continuation_graph_fingerprint(
        static_cast<unsigned>(rows),split_class,current_route_bits,
        selected_generation,stream);
    auto& selected_lifecycle=rows==8
        ? continuation.graph_lifecycle : additional->lifecycle;
    require(selected_route_bits==current_route_bits &&
            (rows==8?continuation.graph_compatibility:additional->compatibility).
                matches(compatibility) &&
            selected_lifecycle.admits(compatibility,selected_generation),
            "continuation graph compatibility fingerprint changed; recapture required");

    const int base_position = position_;
    for (int layer = 0; layer < kLayers; ++layer)
        if (impl_->gdn_layers[layer])
            impl_->gdn_layers[layer]->validate_saved_checkpoint(
                impl_->transaction->gdn_checkpoints[layer],base_position);
    for (auto& layer : impl_->full_layers)
        if (layer) layer->invalidate_retained_prefix();
    cuda_check(cudaMemcpyAsync(impl_->token_ids,token_ids.data(),token_ids.size_bytes(),
                               cudaMemcpyHostToDevice,stream),
               "upload fixed-B8 continuation token IDs");
    cuda_check(cudaMemcpyAsync(impl_->position_device,&base_position,sizeof(base_position),
                               cudaMemcpyHostToDevice,stream),
               "upload fixed-B8 continuation base position");
    impl_->publish_continuation_graph_roles(
        static_cast<unsigned>(rows),stream,false);
    const auto replay_serial=selected_lifecycle.begin_replay(
        selected_generation,stream);
    auto* selected_roles=impl_->continuation_graph_roles(
        static_cast<unsigned>(rows));
    try {
        require(selected_roles,"selected continuation graph roles disappeared");
        selected_roles->begin_use(selected_roles->generation(),replay_serial);
        if (rows == 8) continuation.graph_executable.launch(stream);
        else additional->executable.launch(stream);
        ++continuation.graph_stats.replay_submissions;
    } catch(...) {
        selected_lifecycle.quarantine("continuation graph launch failed",
            static_cast<int>(cudaErrorUnknown));
        impl_->poison_graph_retirement(cudaErrorUnknown);
        throw;
    }
    impl_->oscar->note_graph_execution(rows,split_class);
    // Keep the canonical host cursor coherent with the device cache throughout
    // the active transaction. Rollback and partial retention restore the saved
    // checkpoint before any eager reappend; full commit must not advance twice.
    impl_->oscar->note_graph_replay(rows);
    const int final_device_position = base_position + rows - 1;
    cuda_check(cudaMemcpyAsync(impl_->position_device,&final_device_position,
                               sizeof(final_device_position),cudaMemcpyHostToDevice,stream),
               "publish fixed-B8 continuation final device position");

    auto& transaction = *impl_->transaction;
    transaction.fresh_snapshot = false;
    transaction.prefix_available = true;
    transaction.attempt_base_position = base_position;
    transaction.attempted_rows = rows;
    transaction.continuation_graph_rows = rows;
    transaction.continuation_graph_attempt = true;
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->full_layers[layer])
            impl_->full_layers[layer]->arm_captured_retained_prefix(
                rows,base_position,stream);
        else if (impl_->gdn_layers[layer])
            impl_->gdn_layers[layer]->arm_captured_retained_prefix(
                transaction.gdn_checkpoints[layer],rows,stream);
    }
    position_ += rows;
    impl_->tap_rows = rows;
    impl_->embedding_rows = rows;
    impl_->last_rows = rows;
    impl_->continuation->rows = rows;
    impl_->qkv_trace_valid = true;
    impl_->resident_exact_state_id = 0;
    ++last_decode_h2d_;
}

bool Exl3TextContext::continuation_graph_active() const noexcept {
    return impl_->continuation && impl_->continuation->graph_active;
}

Exl3ContinuationGraphAdmission
Exl3TextContext::continuation_graph_admission() const noexcept {
    Exl3ContinuationGraphAdmission result;
    result.position = position_;
    result.remaining_position_extent = std::max(0, impl_->max_context - position_);
    if (!impl_->continuation) return result;
    const auto& continuation = *impl_->continuation;
    result.capacity = continuation.capacity;
    result.active = continuation.graph_active;
    result.route_generation = continuation.graph_active_generation;
    result.capture_count = continuation.graph_capture_count;
    result.compatible_reuse_count = continuation.graph_compatible_reuse_count;
    result.oscar_split_class = continuation.graph_split_class;
    result.graph_origin_stream_identity = reinterpret_cast<std::uintptr_t>(
        continuation.graph_origin_stream);
    result.qualified_route_bits = continuation.graph_route_bits;
    // A reset-compatible cached graph exposes its immutable key for lifecycle
    // auditing while active/ready remain false until request state is rebuilt.
    if (!result.active) return result;
    if (result.capacity != 8 || position_ <= 0 ||
        position_ > impl_->max_context - result.capacity)
        return result;
    for (int row = 0; row < result.capacity; ++row) {
        if (ninfer::ops::detail::
                oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(
                    position_ + row + 1) != result.oscar_split_class)
            return result;
    }
    result.position_ready = true;
    const auto* transaction = impl_->transaction.get();
    const auto current_route_bits=impl_->continuation_graph_route(true);
    const auto compatibility=impl_->continuation_graph_fingerprint(
        8,continuation.graph_split_class,current_route_bits,
        continuation.graph_active_generation,continuation.graph_origin_stream);
    result.ready = continuation.projection_binding.matches(impl_->model,impl_->reconstruction_backing) &&
        continuation.graph_route_bits==current_route_bits &&
        continuation.graph_compatibility.matches(compatibility) &&
        continuation.graph_lifecycle.admits(compatibility,
            continuation.graph_active_generation) &&
        continuation.graph_definition.ready() &&
        continuation.graph_executable.ready() && impl_->continuation_graph_b8_enabled &&
        impl_->capture_taps && impl_->oscar && !impl_->host_kv.enabled &&
        !impl_->graph_active && !impl_->graph_capture_active &&
        impl_->oscar->graph_class() == 0 && !impl_->target_projection_timing &&
        !impl_->target_projection_observer &&
        transaction && transaction->active && transaction->fresh_snapshot &&
        !transaction->prefix_available && !transaction->rollback_required &&
        transaction->attempted_rows == 0 && transaction->continuation_graph_rows == 0 &&
        !transaction->continuation_graph_attempt && transaction->position == position_ &&
        transaction->attempt_base_position == position_ &&
        transaction->stream == continuation.graph_origin_stream;
    return result;
}

Exl3ContinuationGraphStats
Exl3TextContext::continuation_graph_stats() const noexcept {
    return impl_->continuation_graph_stats_snapshot();
}

std::string Exl3TextContext::continuation_graph_status() const {
    return impl_->continuation ? impl_->continuation->graph_reason :
        "fixed-B8 continuation was not prepared";
}

std::string Exl3TextContext::continuation_graph_rows_status(int rows) const {
    if (!impl_->continuation) return "continuation was not prepared";
    if (rows == 8) return impl_->continuation->graph_reason;
    if (rows != 4 && rows != 6) return "unsupported continuation graph row extent";
    return impl_->continuation->additional_graphs[rows == 4 ? 0 : 1].reason;
}

void Exl3TextContext::invalidate_projection_graph_binding_for_test() {
    require(impl_->continuation && !impl_->graph_capture_active &&
        (!impl_->transaction || !impl_->transaction->active),
        "projection binding invalidation requires idle continuation");
    impl_->invalidate_continuation_graphs(
        "projection graph binding invalidated for test");
}

void Exl3TextContext::discard_continuation_graph_for_test() {
    Impl::require_host_kv_retirement_admission();
    require(impl_->continuation && !impl_->continuation->graph_active &&
        !impl_->graph_active && !impl_->graph_capture_active &&
        (!impl_->transaction || !impl_->transaction->active),
        "test graph discard requires inactive reset context");
    auto& continuation=*impl_->continuation;
    cuda_check(impl_->retire_graph_handles(),"retire discarded continuation graphs");
    for (auto& graph : continuation.additional_graphs) {
        graph.executable.reset();
        graph.definition.reset();
        graph.split_class=0;
        graph.projection_binding={};
        graph.compatibility={};
        graph.active_generation=0;
        graph.origin_stream=nullptr;
        graph.route_bits=0;
        graph.active=false;
        graph.reason="discarded by matched recapture control";
    }
    continuation.graph_executable.reset();
    continuation.graph_definition.reset();
    continuation.graph_split_class=0;
    continuation.projection_binding={};
    continuation.graph_compatibility={};
    continuation.graph_active_generation=0;
    continuation.graph_origin_stream=nullptr;
    continuation.graph_route_bits=0;
    continuation.graph_reason="discarded by matched recapture control";
}

bool Exl3TextContext::capture_decode_graph(cudaStream_t stream) {
    Impl::require_host_kv_retirement_admission();
    // Keep graph-admission failures self-identifying.  This path is exercised
    // by the ordinary device-KV parity lane and must not report the generic
    // post-reset sentinel after a rejection or a capture exception.
    impl_->graph_reason = "decode graph capture entered";
    if(impl_->oscar_only) {
        impl_->graph_reason="OSCAR-only allocation is eager qualified only";
        return false;
    }
    if(impl_->host_kv.enabled) {
        impl_->graph_reason="host KV streaming is eager only";
        return false;
    }
    if (impl_->transaction && impl_->transaction->rollback_required) {
        impl_->graph_reason =
            "P2 target transaction requires rollback after failed prefix retention";
        return false;
    }
    if (impl_->target_projection_observer) {
        impl_->graph_reason = "target projection observer is installed";
        return false;
    }
    if (impl_->target_projection_timing) {
        impl_->graph_reason = "target projection timing is prepared";
        return false;
    }
    if (impl_->transaction && impl_->transaction->active) {
        impl_->graph_reason = "P2 target transaction is active";
        return false;
    }
    if (impl_->graph_active) return true;
    if (impl_->continuation) impl_->continuation->rows = 0;
    if (impl_->oscar && impl_->oscar->graph_class() == 0) {
        impl_->graph_active = false;
        impl_->graph_reason = "E4C1 OSCAR graph needs set_graph_class before capture";
        return false;
    }
    if (position_ >= impl_->max_context) {
        impl_->graph_reason = "context capacity exhausted";
        return false;
    }
    try {
        cuda_check(cudaStreamSynchronize(stream), "synchronize before E4B2 graph capture");
        impl_->bind_graph_device();
        cuda_check(impl_->retire_graph_handles(),"retire incompatible E4B2 graphs");
        impl_->graph_capture_active = true;
        impl_->graph_definition.capture(stream, [&] {
            // The token upload and embedding remain outside the graph.  The
            // graph starts at hidden_a, whose address is stable for the life
            // of this context.
            impl_->process_rows(nullptr, 1, position_, stream, nullptr, false);
        });
        impl_->graph_executable.instantiate(impl_->graph_definition);
        impl_->graph_executable.upload(stream);
        impl_->graph_capture_active = false;
        impl_->graph_active = true;
        impl_->graph_reason = "active: layer stack + final norm + H6 head";
        return true;
    } catch (const std::exception& error) {
        impl_->graph_active = false;
        impl_->graph_capture_active = false;
        if(impl_->graph_retirement_error)throw;
        if(impl_->retire_graph_handles()!=cudaSuccess)throw;
        impl_->graph_reason = std::string("decode graph capture exception: ") + error.what();
        return false;
    }
}

void Exl3TextContext::decode_graph(std::int64_t token_id, cudaStream_t stream) {
    impl_->join_repair(stream);
    require(!impl_->fast_prefill_failed,"failed layer-major prefill requires context reset");
    Impl::require_host_kv_retirement_admission();
    require(!impl_->transaction || !impl_->transaction->rollback_required,
            "P2 target transaction requires rollback after failed prefix retention");
    require(!impl_->target_projection_observer,
            "graph decode is unavailable while target projection observer is installed");
    require(!impl_->target_projection_timing,
            "graph decode is unavailable while target projection timing is prepared");
    require(!impl_->transaction || !impl_->transaction->active,
            "P2 graph decode is unavailable during a target transaction");
    require(impl_->graph_active, "E4B2 graph decode requested before successful capture");
    require(position_ < impl_->max_context, "E4B2 graph context capacity exhausted");
    if (impl_->continuation) impl_->continuation->rows = 0;
    if (impl_->oscar && impl_->oscar->graph_class() != 0) {
        const int need = ninfer::ops::detail::
            oscar_int2_g128_mixed_attention_decode_split_count_for_tokens(position_ + 1);
        if (need != impl_->oscar->graph_class()) {
            throw std::invalid_argument("E4C1 OSCAR split-class transition: recapture the graph");
        }
    }
    cuda_check(cudaMemcpyAsync(impl_->token_ids, &token_id, sizeof(token_id),
                               cudaMemcpyHostToDevice, stream), "upload E4B2 graph token ID");
    cuda_check(cudaMemcpyAsync(impl_->position_device, &position_, sizeof(position_),
                               cudaMemcpyHostToDevice, stream), "upload E4B2 graph position");
    embedding_lookup_kernel<<<(kHidden + 255) / 256, 256, 0, stream>>>(
        impl_->token_ids, impl_->model->embedding, impl_->hidden_a, 1);
    cuda_check(cudaGetLastError(), "launch E4B2 graph embedding lookup");
    impl_->graph_executable.launch(stream);
    impl_->last_rows = 1;
    impl_->qkv_trace_valid = true;
    ++last_decode_h2d_;
    ++position_;
    for (auto& layer : impl_->full_layers)
        if (layer) layer->note_captured_replay(1);
    if (impl_->oscar && impl_->oscar->graph_class() != 0) {
        impl_->oscar->note_graph_execution();
        impl_->oscar->note_graph_replay();
    }
}

bool Exl3TextContext::graph_active() const noexcept { return impl_->graph_active; }

std::string Exl3TextContext::graph_status() const { return impl_->graph_reason; }

Exl3TextDecodeAttribution Exl3TextContext::profile_decode(std::int64_t token_id, cudaStream_t stream) {
    require(!impl_->transaction || !impl_->transaction->rollback_required,
            "P2 target transaction requires rollback after failed prefix retention");
    require(position_ < impl_->max_context, "E4B1 context capacity exhausted");
    impl_->validate_target_projection_execution(stream);
    if (impl_->continuation) impl_->continuation->rows = 0;
    Impl::ProfileEvents events;
    NvtxRange model_range("exl3.text_model.decode");
    try {
        events.create();
        Impl::record(events.total_start, stream, "record E4B1 total start");
        cuda_check(cudaMemcpyAsync(impl_->token_ids, &token_id, sizeof(token_id),
                                   cudaMemcpyHostToDevice, stream), "upload E4B1 profile token ID");
        Impl::record(events.token_end, stream, "record E4B1 token end");
        impl_->process_rows(impl_->token_ids, 1, position_, stream,
                            &events, true, false, true);
        Impl::record(events.total_end, stream, "record E4B1 total end");
        cuda_check(cudaEventSynchronize(events.total_end), "synchronize E4B1 profile total");

        auto elapsed = [](cudaEvent_t start, cudaEvent_t end, const char* label) {
            float milliseconds = 0.0f;
            cuda_check(cudaEventElapsedTime(&milliseconds, start, end), label);
            return static_cast<double>(milliseconds) * 1000.0;
        };
        Exl3TextDecodeAttribution result;
        result.total_microseconds = elapsed(events.total_start, events.total_end, "read E4B1 total");
        result.token_id_h2d_microseconds = elapsed(events.total_start, events.token_end, "read E4B1 token time");
        result.embedding_microseconds = elapsed(events.embedding_start, events.embedding_end, "read E4B1 embedding time");
        result.layer_stack_microseconds = elapsed(events.layer_stack_start, events.layer_stack_end, "read E4B1 stack time");
        result.final_norm_microseconds = elapsed(events.final_norm_start, events.final_norm_end, "read E4B1 final norm time");
        result.lm_head_microseconds = elapsed(events.lm_head_start, events.lm_head_end, "read E4B1 LM head time");
        for (int i = 0; i < kLayers; ++i) {
            result.layer_microseconds[i] = elapsed(events.layer_start[i], events.layer_end[i], "read E4B1 layer time");
        }
        result.other_microseconds = result.total_microseconds - result.token_id_h2d_microseconds -
            result.embedding_microseconds - result.layer_stack_microseconds -
            result.final_norm_microseconds - result.lm_head_microseconds;
        ++last_decode_h2d_;
        ++position_;
        events.destroy();
        return result;
    } catch (...) {
        events.destroy();
        throw;
    }
}

const std::uint16_t* Exl3TextContext::logits_device() const noexcept {
    return impl_->transaction && impl_->transaction->rollback_required
        ? nullptr : logits_;
}

std::vector<float> Exl3TextContext::logits_host(cudaStream_t stream) const {
    require(!impl_->transaction || !impl_->transaction->rollback_required,
            "P2 target logits are invalid until transaction rollback");
    std::vector<std::uint16_t> raw(kVocab);
    cuda_check(cudaMemcpyAsync(raw.data(), logits_, raw.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream), "download E4A logits");
    cuda_check(cudaStreamSynchronize(stream), "synchronize E4A logits");
    impl_->greedy_readback.full_score_bytes+=raw.size()*sizeof(std::uint16_t);
    std::vector<float> result;
    result.reserve(raw.size());
    for (const auto value : raw) result.push_back(half_to_float(value));
    return result;
}

std::vector<float> Exl3TextContext::hidden_host(int layer, cudaStream_t stream) const {
    const int tap = impl_->tap_index(layer);
    require(tap >= 0, "E4A hidden layer is not one of the retained tap locations");
    require(impl_->capture_taps && impl_->tap_rows > 0, "E4A hidden taps were not captured");
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(impl_->tap_rows) * kHidden);
    cuda_check(cudaMemcpyAsync(raw.data(), impl_->taps[tap]->ptr, raw.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream), "download E4A hidden tap");
    cuda_check(cudaStreamSynchronize(stream), "synchronize E4A hidden tap");
    std::vector<float> result;
    result.reserve(raw.size());
    for (const auto value : raw) result.push_back(half_to_float(value));
    return result;
}

std::size_t Exl3ExactHostState::payload_bytes() const noexcept {
    std::size_t bytes = logits_.size() * 2 + embedding_.size() * 2;
    for(const auto& page:kv_pages_) {
        for(const auto& x:page->k) bytes+=x.size()*2;
        for(const auto& x:page->v) bytes+=x.size()*2;
    }
    for (std::size_t i=0;i<48;++i) bytes += recurrent_plane(i).size_bytes();
    for (const auto& x : convolution_) bytes += x.size() * 2;
    for (const auto& x : taps_) bytes += x.size() * 2;
    return bytes;
}

void Exl3ExactHostState::visit_kv_for_test(const std::function<void(
    int, int, int, std::span<const std::uint16_t>,
    std::span<const std::uint16_t>)>& visitor) const {
    require(static_cast<bool>(visitor), "KV visitor missing");
    int first = 0;
    for (const auto& page : kv_pages_) {
        require(page && page->first == first && page->rows > 0 &&
                page->rows <= Exl3ExactKVPage::token_capacity &&
                page->first + page->rows <= position_, "KV visitor page extent");
        const std::size_t elements = static_cast<std::size_t>(page->rows) * 1024;
        for (int bank = 0; bank < 16; ++bank) {
            require(page->k[bank].size() == elements && page->v[bank].size() == elements,
                    "KV visitor plane extent");
            visitor(bank * 4 + 3, first, page->rows, page->k[bank], page->v[bank]);
        }
        first += page->rows;
    }
    require(first == position_, "KV visitor incomplete state");
}

bool Exl3ExactHostState::native_extent_valid(const int maximum_position) const noexcept {
    if(maximum_position<=0 || position_<=0 || position_>maximum_position ||
       device_position_<0 || device_position_>=position_)return false;
    int first=0;
    for(const auto& page:kv_pages_) {
        if(!page || page->first!=first || page->rows<=0 ||
           page->rows>Exl3ExactKVPage::token_capacity || page->rows>position_-first)return false;
        const auto elements=static_cast<std::size_t>(page->rows)*1024;
        for(int bank=0;bank<16;++bank)
            if(page->k[bank].size()!=elements || page->v[bank].size()!=elements)return false;
        first+=page->rows;
    }
    return first==position_;
}

std::shared_ptr<const Exl3ExactHostState> Exl3ExactHostState::detached_payload_for_test() const {
    auto copy=create();
    copy->position_=position_;copy->device_position_=device_position_;
    copy->rope_offset_=rope_offset_;copy->tap_rows_=tap_rows_;
    copy->embedding_rows_=embedding_rows_;copy->last_rows_=last_rows_;
    copy->convolution_=convolution_;copy->logits_=logits_;
    copy->embedding_=embedding_;copy->taps_=taps_;
    copy->kv_pages_.reserve(kv_pages_.size());
    for(const auto& page:kv_pages_)
        copy->kv_pages_.push_back(make_bounded_shared<Exl3ExactKVPage>(*page));
    for(std::size_t i=0;i<recurrent_.size();++i) {
        const auto plane=recurrent_plane(i);
        copy->recurrent_[i].assign(plane.begin(),plane.end());
    }
    return copy;
}
bool Exl3ExactHostState::same_payload(const Exl3ExactHostState& x) const {
    return model_identity_==x.model_identity_ && same_represented_payload_for_test(x);
}
bool Exl3ExactHostState::same_represented_payload_for_test(const Exl3ExactHostState& x) const {
    if (position_ != x.position_ || rope_offset_ != x.rope_offset_ || device_position_ != x.device_position_ ||
        tap_rows_ != x.tap_rows_ || embedding_rows_ != x.embedding_rows_ || last_rows_ != x.last_rows_ ||
        kv_pages_.size()!=x.kv_pages_.size() || convolution_ != x.convolution_ || logits_ != x.logits_ ||
        embedding_ != x.embedding_ || taps_ != x.taps_) return false;
    for(std::size_t i=0;i<kv_pages_.size();++i) {
        if(kv_pages_[i]==x.kv_pages_[i]) continue;
        const auto& a=*kv_pages_[i]; const auto& b=*x.kv_pages_[i];
        if(a.first!=b.first || a.rows!=b.rows || a.k!=b.k || a.v!=b.v) return false;
    }
    for (std::size_t i = 0; i < recurrent_.size(); ++i)
        if (recurrent_plane(i).size() != x.recurrent_plane(i).size() ||
            std::memcmp(recurrent_plane(i).data(), x.recurrent_plane(i).data(), recurrent_plane(i).size_bytes()) != 0)
            return false;
    return true;
}
std::string Exl3ExactHostState::represented_first_difference_for_test(
    const Exl3ExactHostState& x) const {
    const auto scalar=[&](const char* name,auto a,auto b)->std::string {
        if(a==b)return {};
        return std::string(name)+" expected="+std::to_string(a)+
            " actual="+std::to_string(b);
    };
    if(auto d=scalar("position",position_,x.position_);!d.empty())return d;
    if(auto d=scalar("device_position",device_position_,x.device_position_);!d.empty())return d;
    if(auto d=scalar("rope_offset",rope_offset_,x.rope_offset_);!d.empty())return d;
    if(auto d=scalar("tap_rows",tap_rows_,x.tap_rows_);!d.empty())return d;
    if(auto d=scalar("embedding_rows",embedding_rows_,x.embedding_rows_);!d.empty())return d;
    if(auto d=scalar("last_rows",last_rows_,x.last_rows_);!d.empty())return d;
    const auto vector_difference=[&](const std::string& name,const auto& a,
                                     const auto& b)->std::string {
        if(a.size()!=b.size())return name+" size expected="+
            std::to_string(a.size())+" actual="+std::to_string(b.size());
        for(std::size_t i=0;i<a.size();++i)
            if(std::memcmp(&a[i],&b[i],sizeof(a[i]))!=0)
                return name+" index="+std::to_string(i)+" expected="+
                    std::to_string(a[i])+" actual="+std::to_string(b[i]);
        return {};
    };
    if(auto d=scalar("kv_page_count",kv_pages_.size(),x.kv_pages_.size());!d.empty())return d;
    for(std::size_t i=0;i<kv_pages_.size();++i) {
        const auto& a=*kv_pages_[i];const auto& b=*x.kv_pages_[i];
        const auto name="kv_page="+std::to_string(i);
        if(auto d=scalar((name+" first").c_str(),a.first,b.first);!d.empty())return d;
        if(auto d=scalar((name+" rows").c_str(),a.rows,b.rows);!d.empty())return d;
        for(std::size_t bank=0;bank<a.k.size();++bank) {
            if(auto d=vector_difference(name+" bank="+std::to_string(bank)+" k",
                    a.k[bank],b.k[bank]);!d.empty())return d;
            if(auto d=vector_difference(name+" bank="+std::to_string(bank)+" v",
                    a.v[bank],b.v[bank]);!d.empty())return d;
        }
    }
    for(std::size_t i=0;i<convolution_.size();++i)
        if(auto d=vector_difference("convolution layer="+std::to_string(i),
                convolution_[i],x.convolution_[i]);!d.empty())return d;
    if(auto d=vector_difference("logits",logits_,x.logits_);!d.empty())return d;
    if(auto d=vector_difference("embedding",embedding_,x.embedding_);!d.empty())return d;
    for(std::size_t i=0;i<taps_.size();++i)
        if(auto d=vector_difference("tap="+std::to_string(i),taps_[i],x.taps_[i]);!d.empty())return d;
    for(std::size_t i=0;i<recurrent_.size();++i) {
        const auto a=recurrent_plane(i),b=x.recurrent_plane(i);
        if(auto d=vector_difference("recurrent layer="+std::to_string(i),a,b);!d.empty())return d;
    }
    return "equal";
}
std::uint64_t Exl3ExactHostState::represented_payload_hash_for_test() const noexcept {
    std::uint64_t hash=1469598103934665603ull;
    const auto bytes=[&](const void* data,std::size_t size) {
        const auto* p=static_cast<const std::uint8_t*>(data);
        for(std::size_t i=0;i<size;++i)hash=(hash^p[i])*1099511628211ull;
    };
    const auto value=[&](const auto& v){bytes(&v,sizeof(v));};
    const auto vector=[&](const auto& v){
        const std::uint64_t size=static_cast<std::uint64_t>(v.size());
        value(size);if(!v.empty())bytes(v.data(),v.size()*sizeof(v[0]));
    };
    value(position_);value(device_position_);value(rope_offset_);
    value(tap_rows_);value(embedding_rows_);value(last_rows_);
    const std::uint64_t page_count=static_cast<std::uint64_t>(kv_pages_.size());
    value(page_count);
    for(const auto& page:kv_pages_) {
        const bool present=static_cast<bool>(page);value(present);
        if(!page)continue;
        value(page->first);value(page->rows);
        for(std::size_t bank=0;bank<page->k.size();++bank) {
            vector(page->k[bank]);vector(page->v[bank]);
        }
    }
    for(const auto& plane:convolution_)vector(plane);
    vector(logits_);vector(embedding_);
    for(const auto& tap:taps_)vector(tap);
    for(std::size_t i=0;i<recurrent_.size();++i) {
        const auto plane=recurrent_plane(i);
        const std::uint64_t size=static_cast<std::uint64_t>(plane.size());
        value(size);if(!plane.empty())bytes(plane.data(),plane.size_bytes());
    }
    return hash;
}
std::uint64_t Exl3ExactHostState::owner_metadata_bytes() const {
    Exl3ResourceInventory::Requirement metadata;
    using Domain=Exl3ResourceInventory::Domain;
    metadata.add(Domain::host_metadata,1,sizeof(Exl3ExactHostState));
    if(kv_pages_.capacity())metadata.add(Domain::host_metadata,kv_pages_.capacity(),
        sizeof(std::shared_ptr<const Exl3ExactKVPage>));
    if(recurrent_slab_)metadata.add(Domain::host_metadata,1,sizeof(Exl3RecurrentSlab));
    return metadata.units[static_cast<unsigned>(Domain::host_metadata)];
}
std::shared_ptr<Exl3ExactHostState> Exl3ExactHostState::create() {
    return make_bounded_shared<Exl3ExactHostState>(ConstructionKey{});
}
std::uint64_t Exl3ExactHostState::snapshot_metadata_bytes() const {
    Exl3ResourceInventory::Requirement required;
    using Domain=Exl3ResourceInventory::Domain;
    required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3ExactHostState>());
    if(kv_pages_.capacity())required.add(Domain::host_metadata,kv_pages_.capacity(),sizeof(kv_pages_[0]));
    return required.units[static_cast<unsigned>(Domain::host_metadata)];
}
bool Exl3ExactHostState::attach_snapshot_metadata_credit(const std::shared_ptr<const void>& owner,
    RetainedDescriptorLedger::Ticket credit) noexcept {
    if(!owner || !owner.use_count())return false;
    const auto* state=static_cast<const Exl3ExactHostState*>(owner.get());
    if(state->kv_pages_.capacity()>std::numeric_limits<std::size_t>::max()/sizeof(state->kv_pages_[0]))return false;
    return attach_bounded_split_retirement_credit<Exl3ExactHostState>(owner,std::move(credit),
        state->kv_pages_.capacity()*sizeof(state->kv_pages_[0]));
}
bool Exl3ExactHostState::snapshot_metadata_credit_belongs_to(const std::shared_ptr<const void>& owner,
    const RetainedDescriptorLedger& ledger) noexcept {
    return bounded_split_credit_belongs_to<Exl3ExactHostState>(owner,ledger);
}
std::size_t Exl3ExactHostState::recurrent_registered_bytes() const noexcept {
    return recurrent_slab_ && recurrent_slab_->registered?recurrent_slab_->bytes:0;
}

std::shared_ptr<const Exl3ExactKVPage> Exl3ExactHostState::shared_kv_page(int first) const noexcept {
    if(first<0 || first>position_ || position_-first<Exl3ExactKVPage::token_capacity)return {};
    const auto found=std::lower_bound(kv_pages_.begin(),kv_pages_.end(),first,
        [](const auto& page,int offset){return page->first<offset;});
    if(found==kv_pages_.end() || (*found)->first!=first ||
       (*found)->rows!=Exl3ExactKVPage::token_capacity)return {};
    return *found;
}

bool Exl3ExactHostState::is_prefix_of(const Exl3ExactHostState& x) const {
    if(model_identity_!=x.model_identity_ || rope_offset_!=x.rope_offset_ || position_<=0 || position_>x.position_ ||
       kv_pages_.size()>x.kv_pages_.size()) return false;
    for(std::size_t page_index=0;page_index<kv_pages_.size();++page_index) {
        const auto& a=*kv_pages_[page_index];const auto& b=*x.kv_pages_[page_index];
        if(a.first!=b.first || a.first>=position_) return false;
        const int rows=std::min(a.rows,position_-a.first);
        if(rows<=0 || b.rows<rows) return false;
        if(kv_pages_[page_index]==x.kv_pages_[page_index]) continue;
        const std::size_t elements=static_cast<std::size_t>(rows)*1024;
        for(int bank=0;bank<16;++bank) {
            if(a.k[bank].size()<elements || b.k[bank].size()<elements ||
               a.v[bank].size()<elements || b.v[bank].size()<elements ||
               !std::equal(a.k[bank].begin(),a.k[bank].begin()+elements,b.k[bank].begin()) ||
               !std::equal(a.v[bank].begin(),a.v[bank].begin()+elements,b.v[bank].begin())) return false;
        }
    }
    return true;
}

void Exl3ExactHostState::exercise_recurrent_storage_accounting_for_test() {
    {
        const auto blocks=bounded_shared_live_blocks_for_test<Exl3ExactHostState>();
        RetainedDescriptorLedger ledger;
        auto snapshot=create();snapshot->kv_pages_.reserve(8);
        RetainedDescriptorLedger foreign;
        require(!snapshot_metadata_credit_belongs_to(snapshot,ledger),"uncredited snapshot claimed ledger identity");
        const auto required=snapshot->snapshot_metadata_bytes();
        require(required==bounded_shared_allocation_bytes<Exl3ExactHostState>()+
            snapshot->kv_pages_.capacity()*sizeof(snapshot->kv_pages_[0]),"snapshot metadata extent");
        require(!attach_snapshot_metadata_credit(snapshot,ledger.acquire(required-1)) &&
            attach_snapshot_metadata_credit(snapshot,ledger.acquire(required)) &&
            !attach_snapshot_metadata_credit(snapshot,ledger.acquire(required)),
            "snapshot metadata short/exact/duplicate credit");
        require(snapshot_metadata_credit_belongs_to(snapshot,ledger) &&
            !snapshot_metadata_credit_belongs_to(snapshot,foreign),"snapshot credit accepted foreign registry");
        std::weak_ptr<Exl3ExactHostState> weak=snapshot;snapshot.reset();
        require(weak.expired() && ledger.bytes()==bounded_shared_allocation_bytes<Exl3ExactHostState>() &&
            bounded_shared_live_blocks_for_test<Exl3ExactHostState>()==blocks+1,
            "snapshot page table or containing block retired at wrong lifetime");
        weak.reset();
        require(ledger.bytes()==0 && bounded_shared_live_blocks_for_test<Exl3ExactHostState>()==blocks,
            "snapshot final weak release retained metadata block");
    }
    std::array<float,8> backing_a{},backing_b{};
    const auto slab=[](float* data) {
        auto owner=std::shared_ptr<Exl3RecurrentSlab>(new Exl3RecurrentSlab,[](Exl3RecurrentSlab* value) noexcept {
            // Fixture backing is stack-owned, with no native allocation/event.
            value->data=nullptr;Exl3RecurrentSlab::retire(value);
        });
        owner->data=data;owner->bytes=8*sizeof(float);owner->offsets.fill(8);owner->offsets[0]=0;return owner;
    };
    auto shared=slab(backing_a.data()),independent=slab(backing_b.data());
    auto first=create();
    auto alias=create();
    auto other=create();
    auto pageable=create();
    first->recurrent_slab_=shared;alias->recurrent_slab_=shared;other->recurrent_slab_=independent;
    pageable->recurrent_[0].reserve(16);pageable->recurrent_[0].resize(8);
    const std::array<std::shared_ptr<const Exl3ExactHostState>,5> roots{first,first,alias,other,pageable};
    const std::array<std::shared_ptr<const Exl3ExactHostState>,2> invalid_roots{first,{}};
    unsigned partial_visits=0;bool invalid_refused=false;
    try{visit_host_allocations(invalid_roots,[&](const void*,std::size_t){++partial_visits;});}
    catch(const std::exception&){invalid_refused=true;}
    require(invalid_refused && partial_visits==0,"invalid later image emitted partial allocation inventory");
    require(first->payload_bytes()==32 && alias->payload_bytes()==32,"shared recurrent logical payload changed");
    const auto storage=storage_stats(roots);
    require(storage.unique_images==4 && storage.unique_other_payload_bytes==96,"shared recurrent storage counted per image instead of allocation");
    const auto used=allocation_domain_stats(roots,false),capacity=allocation_domain_stats(roots,true);
    require(used.recurrent==96 && capacity.recurrent==64+pageable->recurrent_[0].capacity()*sizeof(float),
        "shared recurrent allocation domains lost slab deduplication or pageable capacity");
    for(bool full:{false,true}) {
        std::uint64_t visited=0;unsigned shared_visits=0,independent_visits=0;
        visit_host_allocations(roots,[&](const void* address,std::size_t bytes) {
            visited+=bytes;
            if(address==backing_a.data())++shared_visits;
            if(address==backing_b.data())++independent_visits;
        },full);
        require(shared_visits==1 && independent_visits==1 && visited==(full?capacity.recurrent:used.recurrent),
            "recurrent visitor duplicated aliased slab or lost pageable extent");
    }
}
Exl3ExactStorageStats Exl3ExactHostState::storage_stats(
    std::span<const std::shared_ptr<const Exl3ExactHostState>> requests) {
    Exl3ExactStorageStats result;
    std::unordered_set<const Exl3ExactHostState*> images;
    std::unordered_set<const Exl3ExactKVPage*> pages;
    std::unordered_set<const Exl3RecurrentSlab*> recurrent_slabs;
    for(const auto& state:requests) {
        require(state!=nullptr,"storage accounting null request");
        result.logical_context_tokens+=state->position_;
        result.longest_context=std::max(result.longest_context,static_cast<std::uint64_t>(state->position_));
        if(!images.insert(state.get()).second) continue;
        ++result.unique_images;
        result.unique_other_payload_bytes+=state->payload_bytes()-static_cast<std::size_t>(state->position_)*65536;
        if(state->recurrent_slab_ && !recurrent_slabs.insert(state->recurrent_slab_.get()).second)
            result.unique_other_payload_bytes-=state->recurrent_slab_->bytes;
        for(const auto& page:state->kv_pages_) if(pages.insert(page.get()).second) {
            ++result.unique_kv_pages;result.materialized_kv_token_rows+=page->rows;
            for(const auto& plane:page->k) {result.materialized_kv_bytes+=plane.size()*2;result.allocated_kv_bytes+=plane.capacity()*2;}
            for(const auto& plane:page->v) {result.materialized_kv_bytes+=plane.size()*2;result.allocated_kv_bytes+=plane.capacity()*2;}
        }
    }
    result.logical_kv_bytes=result.logical_context_tokens*65536;
    return result;
}

Exl3ExactAllocationDomainStats Exl3ExactHostState::allocation_domain_stats(
    std::span<const std::shared_ptr<const Exl3ExactHostState>> requests,
    bool include_unused_capacity) {
    Exl3ExactAllocationDomainStats result;
    std::unordered_set<const Exl3ExactHostState*> images;
    std::unordered_set<const Exl3ExactKVPage*> pages;
    std::unordered_set<const Exl3RecurrentSlab*> recurrent_slabs;
    const auto bytes = [&](const auto& plane) -> std::uint64_t {
        const auto elements = include_unused_capacity ? plane.capacity() : plane.size();
        return static_cast<std::uint64_t>(elements) * sizeof(plane[0]);
    };
    for (const auto& state : requests) {
        require(state != nullptr, "allocation domain null request");
        if (!images.insert(state.get()).second) continue;
        for (const auto& page : state->kv_pages_) if (pages.insert(page.get()).second) {
            for (const auto& plane : page->k) result.kv += bytes(plane);
            for (const auto& plane : page->v) result.kv += bytes(plane);
        }
        if(state->recurrent_slab_) {
            if(recurrent_slabs.insert(state->recurrent_slab_.get()).second)result.recurrent+=state->recurrent_slab_->bytes;
        }
        else for (const auto& plane : state->recurrent_) result.recurrent += bytes(plane);
        for (const auto& plane : state->convolution_) result.convolution += bytes(plane);
        for (const auto& plane : state->taps_) result.state_taps += bytes(plane);
        result.logits += bytes(state->logits_);
        result.embedding += bytes(state->embedding_);
    }
    return result;
}

void Exl3ExactHostState::visit_host_allocations(
    std::span<const std::shared_ptr<const Exl3ExactHostState>> requests,
    const std::function<void(const void*,std::size_t)>& visitor,bool include_unused_capacity) {
    require(static_cast<bool>(visitor),"host allocation visitor missing");
    for(const auto& state:requests)
        require(state!=nullptr,"host allocation null request");
    std::unordered_set<const Exl3ExactHostState*> images;
    std::unordered_set<const Exl3ExactKVPage*> pages;
    std::unordered_set<const Exl3RecurrentSlab*> recurrent_slabs;
    const auto visit=[&](const auto& plane) {
        const auto elements=include_unused_capacity?plane.capacity():plane.size();
        if(elements) visitor(plane.data(),elements*sizeof(plane[0]));
    };
    for(const auto& state:requests) {
        if(!images.insert(state.get()).second) continue;
        for(const auto& page:state->kv_pages_) if(pages.insert(page.get()).second) {
            for(const auto& plane:page->k) visit(plane);
            for(const auto& plane:page->v) visit(plane);
        }
        if(state->recurrent_slab_) {
            if(recurrent_slabs.insert(state->recurrent_slab_.get()).second)
                visitor(state->recurrent_slab_->data,state->recurrent_slab_->bytes);
        } else for(const auto& plane:state->recurrent_) visit(plane);
        for(const auto& plane:state->convolution_) visit(plane);
        for(const auto& plane:state->taps_) visit(plane);
        visit(state->logits_);visit(state->embedding_);
    }
}

Exl3RecurrentExportStats Exl3TextContext::recurrent_export_stats() const {return impl_->recurrent_export.snapshot();}
void Exl3TextContext::set_recurrent_growth_admission(Exl3RecurrentExportPool::GrowthAdmission admission) {
    impl_->recurrent_export.set_growth_admission(std::move(admission));
}
void Exl3TextContext::set_snapshot_metadata_reservation(SnapshotMetadataReservation reservation) {
    impl_->snapshot_metadata_reservation=std::move(reservation);
}
void Exl3TextContext::set_request_metadata_reservation(SnapshotMetadataReservation reservation) {
    impl_->request_metadata_reservation=std::move(reservation);
}
void Exl3TextContext::set_request_host_payload_reservation(RequestHostPayloadReservation reservation) {
    impl_->request_host_payload_reservation=std::move(reservation);
}
const Exl3TextContext::RequestHostPayloadReservation& Exl3TextContext::request_host_payload_reservation() const noexcept {
    return impl_->request_host_payload_reservation;
}
void Exl3TextContext::set_request_host_payload_observer(RequestHostPayloadObserver observer) {
    impl_->request_host_payload_observer=std::move(observer);
}
const Exl3TextContext::RequestHostPayloadObserver& Exl3TextContext::request_host_payload_observer() const noexcept {
    return impl_->request_host_payload_observer;
}
const Exl3TextContext::SnapshotMetadataReservation& Exl3TextContext::request_metadata_reservation() const noexcept {
    return impl_->request_metadata_reservation;
}
void Exl3TextContext::set_recurrent_borrow_admission(Exl3RecurrentExportPool::BorrowAdmission admission) {
    impl_->recurrent_export.set_borrow_admission(std::move(admission));
}
void Exl3TextContext::fail_recurrent_constructor_for_test(unsigned fault) {
    if(!impl_->pinned_recurrent_export)
        throw std::logic_error("recurrent constructor fixture requires pinned recurrent export");
    impl_->recurrent_export.fail_constructor_for_test(fault);
}
void Exl3TextContext::fail_export_copy_for_test(unsigned submission) {
    require(submission>=1 && submission<=96 && !impl_->export_copy_fault_for_test,
        "export copy fixture requires one fresh submission1..96");
    impl_->export_copy_fault_for_test=submission;
}

std::shared_ptr<const Exl3ExactHostState> Exl3TextContext::export_exact_host_state(cudaStream_t stream,bool share_prefix,
    bool fail_after_recurrent_plan_for_test) const {
    impl_->join_repair(stream);
    const bool transaction_ready = !impl_->transaction ||
        ((impl_->transaction->host_kv || impl_->transaction->device_kv) &&
         !impl_->transaction->active &&
         !impl_->transaction->rollback_required);
    require(!impl_->oscar_only && !impl_->oscar && !impl_->graph_active && !impl_->graph_capture_active &&
            transaction_ready && impl_->capture_taps && position_ > 0 &&
            (!impl_->continuation || impl_->continuation->rows == 0),
            "exact host export requires ordinary eager target state");
    require(!impl_->host_kv.enabled || (share_prefix && !impl_->host_kv_failed &&
        impl_->exact_prefix_position==position_),"host KV export requires completed authoritative host pages");
    using ExportClock=std::chrono::steady_clock;
    auto& export_stats=impl_->recurrent_export.stats;
    const auto export_copy_fault=std::exchange(
        impl_->export_copy_fault_for_test,0U);
    unsigned export_copy_submission=0;
    struct Timing {double& sum;ExportClock::time_point start=ExportClock::now();~Timing(){sum+=std::chrono::duration<double,std::milli>(ExportClock::now()-start).count();}} timing{export_stats.total_ms};
    ++export_stats.exports;
    require(impl_->exact_prefix_position<=position_,"exact export lineage moved backwards");
    // Promise precedes every snapshot allocation and outlives partial storage
    // on unwind. Publication recognizes the same owner-bound ledger afterward.
    std::optional<RetainedDescriptorLedger::Ticket> snapshot_credit;
    if(impl_->snapshot_metadata_reservation) {
        Exl3ResourceInventory::Requirement required;
        using Domain=Exl3ResourceInventory::Domain;
        required.add(Domain::host_metadata,1,bounded_shared_allocation_bytes<Exl3ExactHostState>());
        required.add(Domain::host_metadata,(static_cast<std::uint64_t>(position_)+63)/64,
            sizeof(std::shared_ptr<const Exl3ExactKVPage>));
        snapshot_credit.emplace(impl_->snapshot_metadata_reservation(required.units[static_cast<unsigned>(Domain::host_metadata)]));
        require(snapshot_credit->bytes()==required.units[static_cast<unsigned>(Domain::host_metadata)],
            "snapshot metadata reservation extent changed");
    }
    // Establish the bounded recurrent destination before allocating snapshot
    // metadata, extending page tables or downloading any authoritative state.
    // On subsequent preparation failure, the borrower releases its flight and
    // the admitted physical slab remains reusable in the existing pool.
    std::array<std::size_t,48> recurrent_sizes{};
    std::shared_ptr<Exl3RecurrentSlab> planned_recurrent;
    if(impl_->pinned_recurrent_export) {
        Timing recurrent_timing{export_stats.recurrent_ms};
        std::size_t index=0;
        for(const auto& layer:impl_->gdn_layers)if(layer) {
            require(index<recurrent_sizes.size(),"recurrent export plane count");
            recurrent_sizes[index++]=layer->recurrent_state_bytes()/4;
        }
        require(index==recurrent_sizes.size(),"recurrent export plane count");
        planned_recurrent=impl_->recurrent_export.acquire(recurrent_sizes);
    }
    if(fail_after_recurrent_plan_for_test) {
        require(bool(planned_recurrent),"export preparation fault requires planned recurrent destination");
        throw std::runtime_error("injected exact export preparation failure after recurrent plan");
    }
    auto state = Exl3ExactHostState::create();
    state->residency_id_=next_exact_residency_id.fetch_add(1,std::memory_order_relaxed);
    if(state->residency_id_==0)
        state->residency_id_=next_exact_residency_id.fetch_add(1,std::memory_order_relaxed);
    state->model_identity_ = impl_->model->host_state_identity;
    state->position_ = position_;
    state->rope_offset_=impl_->rope_offset;
    state->device_position_ = device_position_host(stream);
    state->tap_rows_ = impl_->tap_rows;
    state->embedding_rows_ = impl_->embedding_rows;
    state->last_rows_ = impl_->last_rows;
    const int prefix_position=share_prefix?impl_->exact_prefix_position:0;
    const std::vector<std::shared_ptr<const Exl3ExactKVPage>> empty_prefix;
    const auto& prefix_pages=share_prefix?impl_->exact_prefix_pages:empty_prefix;
    auto extension=extend_exact_pages(prefix_pages,prefix_position,position_,false,nullptr,0,impl_->request_metadata_reservation);
    state->kv_pages_=std::move(extension.all);
    if(snapshot_credit) {
        require(snapshot_credit->bytes()==state->snapshot_metadata_bytes(),"snapshot metadata prepared capacity changed");
        require(Exl3ExactHostState::attach_snapshot_metadata_credit(state,std::move(*snapshot_credit)),
            "snapshot metadata construction credit refused");
        snapshot_credit.reset();
    }
    const auto& fresh_pages=extension.fresh;
    const auto* batched_kv_option=std::getenv("NINFER_EXL3_EXACT_KV_EXPORT_BATCHED_FENCE");
    if(batched_kv_option && std::string_view(batched_kv_option)!="0" &&
       std::string_view(batched_kv_option)!="1")
        throw std::invalid_argument("exact KV export batched fence must be0 or1");
    constexpr std::size_t batched_kv_capacity=256;
    std::size_t full_layer_count=0;
    for(const auto& layer:impl_->full_layers)full_layer_count+=bool(layer);
    const bool batched_kv_requested=batched_kv_option &&
        std::string_view(batched_kv_option)=="1";
    const bool batched_kv_export=batched_kv_requested && !fresh_pages.empty() &&
        full_layer_count && fresh_pages.size()<=
            batched_kv_capacity/(2*full_layer_count);
    const auto download = [&](auto& out, const void* src, std::size_t elements) {
        out.resize(elements);
        cuda_check(cudaMemcpyAsync(out.data(), src, elements * sizeof(out[0]),
                                  cudaMemcpyDeviceToHost, stream), "export exact host plane");
        // Pageable vector lifetime is bounded by this blocking API, including failure.
        cuda_check(cudaStreamSynchronize(stream), "complete exact host plane");
    };
    const auto add_counter=[](std::uint64_t& value,std::uint64_t increment) noexcept {
        value=increment>UINT64_MAX-value?UINT64_MAX:value+increment;
    };
    // cudaMemcpyBatchAsync may consume descriptor storage until the batch has
    // completed. Keep these arrays alive through the recurrent export fence;
    // stack-local arrays inside submit_export_plan would retire too early.
    std::array<void*,48> recurrent_batch_destinations{};
    std::array<const void*,48> recurrent_batch_sources{};
    std::array<std::size_t,48> recurrent_batch_sizes{};
    const auto submit_export_plan=[&](const auto& plan,Exl3ExportPrecision precision,
                                       const char* label) {
        if(impl_->batched_recurrent_export &&
           precision==Exl3ExportPrecision::fp32 && stream &&
           !export_copy_fault && plan.size()>1) {
            require(plan.size()<=48,"batched recurrent export range capacity");
            for(std::size_t index=0;index<plan.size();++index) {
                recurrent_batch_destinations[index]=plan[index].destination;
                recurrent_batch_sources[index]=plan[index].source;
                recurrent_batch_sizes[index]=plan[index].bytes;
            }
            cudaMemcpyAttributes attributes{};
            attributes.srcAccessOrder=cudaMemcpySrcAccessOrderStream;
            std::size_t first_attribute=0;
            cuda_check(cudaMemcpyBatchAsync(recurrent_batch_destinations.data(),
                recurrent_batch_sources.data(),recurrent_batch_sizes.data(),plan.size(),
                &attributes,&first_attribute,1,stream),label);
            add_counter(export_stats.planned_copy_ranges,plan.logical_ranges());
            add_counter(export_stats.coalesced_copy_ranges,plan.coalesced_ranges());
            add_counter(export_stats.copy_submissions,1);
            add_counter(export_stats.fp32_copy_submissions,1);
            add_counter(export_stats.batched_copy_calls,1);
            add_counter(export_stats.batched_copy_ranges,plan.size());
            return;
        }
        const auto submitted=plan.submit([&](const auto& segment) noexcept {
            if(++export_copy_submission==export_copy_fault)
                return static_cast<int>(cudaErrorUnknown);
            return static_cast<int>(cudaMemcpyAsync(segment.destination,
                segment.source,segment.bytes,cudaMemcpyDeviceToHost,stream));
        });
        add_counter(export_stats.planned_copy_ranges,plan.logical_ranges());
        add_counter(export_stats.coalesced_copy_ranges,plan.coalesced_ranges());
        add_counter(export_stats.copy_submissions,submitted.attempted);
        add_counter(precision==Exl3ExportPrecision::fp16?
            export_stats.fp16_copy_submissions:export_stats.fp32_copy_submissions,
            submitted.attempted);
        if(!submitted)cuda_check(static_cast<cudaError_t>(submitted.status),label);
    };
    if(impl_->pinned_recurrent_export){
        Timing recurrent_timing{export_stats.recurrent_ms};
        auto slab=std::move(planned_recurrent);
        if(slab){
            try {
                {Timing submit_timing{export_stats.submit_ms};std::size_t index=0;
                    Exl3ExportCopyPlan<48> plan;
                    for(const auto& layer:impl_->gdn_layers)if(layer){
                        require(plan.add(slab->data+slab->offsets[index],
                            layer->recurrent_state_device(),recurrent_sizes[index]*4,
                            Exl3ExportPrecision::fp32),
                            "recurrent export copy plan capacity");
                        ++index;
                    }
                    submit_export_plan(plan,Exl3ExportPrecision::fp32,
                        "direct recurrent slab copy");
                    cuda_check(cudaEventRecord(slab->completed,stream),"recurrent export completion event");slab->recorded=true;
                }
                {Timing fence_timing{export_stats.fence_ms};cuda_check(cudaEventSynchronize(slab->completed),"recurrent export immutable fence");}
                state->recurrent_slab_=slab;++export_stats.pinned_exports;
            }catch(...){slab->poisoned=true;++export_stats.poisons;cudaStreamSynchronize(stream);throw;}
        }
    }
    int full = 0, gdn = 0;
    Exl3ExportCopyPlan<48> pageable_recurrent_plan;
    Exl3ExportCopyPlan<48> convolution_plan;
    Exl3ExportCopyPlan<batched_kv_capacity> batched_kv_plan;
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->full_layers[layer]) {
            for(const auto& page:fresh_pages) {
                const int skip=std::clamp(prefix_position-page->first,0,page->rows);
                const std::size_t elements=static_cast<std::size_t>(page->rows-skip)*1024;
                const auto copy=[&](auto& output,const void* source) {
                    auto* destination=output.data()+skip*1024;
                    const auto* source_first=static_cast<const std::uint16_t*>(source)+
                        static_cast<std::size_t>(page->first+skip)*1024;
                    if(batched_kv_export) {
                        require(batched_kv_plan.add(destination,source_first,elements*2,
                            Exl3ExportPrecision::fp16),
                            "exact KV export batched plan capacity");
                    } else {
                        cuda_check(cudaMemcpyAsync(destination,source_first,elements*2,
                            cudaMemcpyDeviceToHost,stream),"export exact KV page suffix");
                        cuda_check(cudaStreamSynchronize(stream),"complete exact KV page suffix");
                    }
                    state->kv_export_bytes_+=elements*2;
                };
                copy(page->k[full],impl_->cache_k[layer]->ptr);
                copy(page->v[full],impl_->cache_v[layer]->ptr);
            }
            ++full;
        } else {
            const auto& l = impl_->gdn_layers[layer];
            export_stats.recurrent_bytes+=l->recurrent_state_bytes();
            if(!state->recurrent_slab_) {
                auto& output=state->recurrent_[gdn];
                output.resize(l->recurrent_state_bytes()/sizeof(output[0]));
                require(pageable_recurrent_plan.add(output.data(),
                    l->recurrent_state_device(),l->recurrent_state_bytes(),
                    Exl3ExportPrecision::fp32),
                    "pageable recurrent export copy plan capacity");
            }
            auto& convolution=state->convolution_[gdn];
            convolution.resize(l->physical_conv_state_bytes()/sizeof(convolution[0]));
            require(convolution_plan.add(convolution.data(),
                l->physical_conv_state_device(),l->physical_conv_state_bytes(),
                Exl3ExportPrecision::fp16),
                "convolution export copy plan capacity");
            ++gdn;
        }
    }
    if(batched_kv_plan.size()) {
        const auto submitted=batched_kv_plan.submit([&](const auto& segment) noexcept {
            return static_cast<int>(cudaMemcpyAsync(segment.destination,segment.source,
                segment.bytes,cudaMemcpyDeviceToHost,stream));
        });
        if(!submitted)cuda_check(static_cast<cudaError_t>(submitted.status),
            "submit exact KV export batched plan");
        cuda_check(cudaStreamSynchronize(stream),"complete exact KV export batched plan");
        ++impl_->host_kv.exact_kv_export_batched_calls;
        impl_->host_kv.exact_kv_export_batched_ranges+=batched_kv_plan.logical_ranges();
        impl_->host_kv.exact_kv_export_batched_bytes+=batched_kv_plan.logical_bytes();
        ++impl_->host_kv.exact_kv_export_batched_fences;
        impl_->host_kv.exact_kv_export_saved_fences+=
            batched_kv_plan.logical_ranges()-1;
    }
    if(pageable_recurrent_plan.size()) {
        Timing recurrent_timing{export_stats.recurrent_ms};
        submit_export_plan(pageable_recurrent_plan,Exl3ExportPrecision::fp32,
            "pageable recurrent export copy");
        cuda_check(cudaStreamSynchronize(stream),
            "complete pageable recurrent export copy plan");
    }
    if(convolution_plan.size()) {
        submit_export_plan(convolution_plan,Exl3ExportPrecision::fp16,
            "convolution export copy");
        cuda_check(cudaStreamSynchronize(stream),
            "complete convolution export copy plan");
    }
    download(state->logits_, impl_->logits, kVocab);
    download(state->embedding_, impl_->embedding_trace->ptr,
             static_cast<std::size_t>(impl_->embedding_rows) * kHidden);
    for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap)
        download(state->taps_[tap], impl_->taps[tap]->ptr,
                 static_cast<std::size_t>(impl_->tap_rows) * kHidden);
    if(share_prefix) {
        impl_->exact_prefix_pages=state->kv_pages_;
        impl_->exact_prefix_position=position_;
    }
    impl_->resident_exact_state_id=state->residency_id_;
    return state;
}

void Exl3TextContext::restore_exact_host_state(const Exl3ExactHostState& state, cudaStream_t stream) {
    Impl::require_host_kv_retirement_admission();
    require(!impl_->host_kv_failed,"exact restore cannot reuse failed HostKV lineage");
    require(state.rope_offset_==0||impl_->media_features,"nonzero media offset requires research context");
    restore_host_state_impl(state, nullptr, false, stream);
    impl_->resident_exact_state_id=state.residency_id_;
}

bool Exl3TextContext::restore_exact_host_state_if_needed(const Exl3ExactHostState& state,cudaStream_t stream) {
    Impl::require_host_kv_retirement_admission();
    require(!impl_->host_kv_failed,"exact restore cannot reuse failed HostKV lineage");
    require(state.rope_offset_==0||impl_->media_features,"nonzero media offset requires research context");
    if(exact_host_state_resident(state))return false;
    restore_exact_host_state(state,stream);return true;
}

bool Exl3TextContext::exact_host_state_resident(const Exl3ExactHostState& state) const noexcept {
    return !impl_->host_kv_failed && !impl_->oscar && !impl_->oscar_only && state.residency_id_!=0 &&
        impl_->resident_exact_state_id==state.residency_id_ &&
        state.model_identity_==impl_->model->host_state_identity && position_==state.position_ && impl_->rope_offset==state.rope_offset_;
}

std::size_t Exl3TurboAngleWarmPages::payload_bytes() const noexcept {
    std::size_t bytes=0;
    for(const auto& x:k_) bytes+=x.size();
    for(const auto& x:v_) bytes+=x.size();
    return bytes;
}

std::shared_ptr<const Exl3TurboAngleWarmPages> Exl3TextContext::make_turboangle_warm_pages(
    std::shared_ptr<const Exl3ExactHostState> source, int rows) {
    require(source && rows>0 && rows<=64 && rows<=source->position_, "TurboAngle bounded warm rows");
    return make_turboangle_pages(std::move(source),rows);
}

std::shared_ptr<const Exl3TurboAngleWarmPages> Exl3TextContext::make_turboangle_l1_pages(
    std::shared_ptr<const Exl3ExactHostState> source) {
    require(source && source->position_>0,"TurboAngle L1 source extent");
    const int rows=source->position_;
    return make_turboangle_pages(std::move(source),rows);
}

std::shared_ptr<const Exl3TurboAngleWarmPages> Exl3TextContext::extend_turboangle_l1_pages(
    std::shared_ptr<const Exl3ExactHostState> source,
    std::shared_ptr<const Exl3TurboAngleWarmPages> prefix) {
    require(source && prefix && prefix->full_history() && prefix->source_->is_prefix_of(*source) &&
        source->position_>prefix->source_->position_ && source->position_-prefix->source_->position_<=32,
        "TurboAngle L1 prefix extension lineage/extent");
    auto next=std::shared_ptr<Exl3TurboAngleWarmPages>(new Exl3TurboAngleWarmPages);
    next->source_=std::move(source);next->first_=0;next->rows_=next->source_->position_;
    const int first=prefix->source_->position_;
    for(int bank=0;bank<16;++bank) {
        const auto extend=[&](auto& output,const auto& prior,bool key) {
            output=prior;
            output.reserve(static_cast<std::size_t>(next->rows_)*4*TurboAngle256::bytes(key));
            for(int row=first;row<next->rows_;++row) for(int head=0;head<4;++head) {
                std::array<float,256> values;
                const auto& page=*next->source_->kv_pages_[row/Exl3ExactKVPage::token_capacity];
                const auto& input=key?page.k[bank]:page.v[bank];
                for(int d=0;d<256;++d) values[d]=half_to_float(input[((row-page.first)*4+head)*256+d]);
                const auto record=TurboAngle256::encode(values,key);
                output.insert(output.end(),record.begin(),record.end());
            }
        };
        extend(next->k_[bank],prefix->k_[bank],true);
        extend(next->v_[bank],prefix->v_[bank],false);
    }
    return next;
}

std::shared_ptr<const Exl3TurboAngleWarmPages> Exl3TextContext::make_turboangle_pages(
    std::shared_ptr<const Exl3ExactHostState> source,int rows) {
    require(source && rows>0 && rows<=source->position_,"TurboAngle page extent");
    auto warm=std::shared_ptr<Exl3TurboAngleWarmPages>(new Exl3TurboAngleWarmPages);
    warm->source_=std::move(source); warm->rows_=rows; warm->first_=warm->source_->position_-rows;
    for(int bank=0;bank<16;++bank) {
        const auto encode=[&](auto& output,bool key) {
            output.reserve(static_cast<std::size_t>(rows)*4*TurboAngle256::bytes(key));
            for(int row=warm->first_;row<warm->source_->position_;++row)
                for(int head=0;head<4;++head) {
                    std::array<float,256> values;
                    const auto& page=*warm->source_->kv_pages_[row/Exl3ExactKVPage::token_capacity];
                    const auto& input=key?page.k[bank]:page.v[bank];
                    for(int d=0;d<256;++d) values[d]=half_to_float(input[((row-page.first)*4+head)*256+d]);
                    const auto record=TurboAngle256::encode(values,key);
                    output.insert(output.end(),record.begin(),record.end());
                }
        };
        encode(warm->k_[bank],true);
        encode(warm->v_[bank],false);
    }
    return warm;
}

void Exl3TextContext::restore_oscar_host_state(const Exl3ExactHostState& state,
    const Exl3TurboAngleWarmPages* warm, cudaStream_t stream) {
    impl_->resident_exact_state_id=0;
    restore_host_state_impl(state,warm,true,stream);
}

void Exl3TextContext::rebase_oscar_host_state_delta(const Exl3ExactHostState& old_root,
    const Exl3ExactHostState& new_root,const Exl3TurboAngleWarmPages& pages,cudaStream_t stream) {
    require(impl_->oscar_only && impl_->oscar && impl_->capture_taps && !impl_->graph_active &&
        !impl_->graph_capture_active && (!impl_->transaction || !impl_->transaction->active) &&
        pages.source_.get()==&new_root && pages.full_history() && old_root.is_prefix_of(new_root) &&
        position_==old_root.position_ && new_root.position_>old_root.position_ &&
        new_root.position_-old_root.position_<=32 && new_root.model_identity_==impl_->model->host_state_identity,
        "TurboAngle L1 delta rebase identity/lineage/state");
    const int first=old_root.position_,rows=new_root.position_-first;
    const auto upload=[&](void* destination,const auto& source,const char* label) {
        cuda_check(cudaMemcpyAsync(destination,source.data(),source.size()*sizeof(source[0]),
            cudaMemcpyHostToDevice,stream),label);
        cuda_check(cudaStreamSynchronize(stream),"complete TurboAngle L1 delta upload");
    };
    int full=0,gdn=0;
    for(int layer=0;layer<kLayers;++layer) {
        if(impl_->full_layers[layer]) {
            std::vector<std::uint16_t> k(static_cast<std::size_t>(rows)*1024),v(k.size());
            const auto decode=[&](auto& output,bool key) {
                const auto& packed=key?pages.k_[full]:pages.v_[full];
                const auto record_bytes=TurboAngle256::bytes(key);
                for(int absolute=first;absolute<new_root.position_;++absolute) for(int head=0;head<4;++head) {
                    const auto values=TurboAngle256::decode(std::span<const std::uint8_t>(
                        packed.data()+(static_cast<std::size_t>(absolute)*4+head)*record_bytes,record_bytes),key);
                    for(int d=0;d<256;++d) {
                        require(std::isfinite(values[d]) && std::abs(values[d])<=65504.0f,
                            "TurboAngle L1 delta FP16 range");
                        output[((absolute-first)*4+head)*256+d]=float_to_half(values[d]);
                    }
                }
            };
            decode(k,true);decode(v,false);
            upload(impl_->host_layer_k->ptr,k,"upload TurboAngle L1 delta K");
            upload(impl_->host_layer_v->ptr,v,"upload TurboAngle L1 delta V");
            impl_->oscar->append_kv_layer(layer,
                static_cast<const std::uint16_t*>(impl_->host_layer_k->ptr),
                static_cast<const std::uint16_t*>(impl_->host_layer_v->ptr),rows,first,stream);
            cuda_check(cudaStreamSynchronize(stream),"complete TurboAngle L1 delta append");
            ++full;
        } else {
            impl_->gdn_layers[layer]->restore_host_state(new_root.recurrent_plane(gdn),new_root.convolution_[gdn],stream);
            ++gdn;
        }
    }
    upload(impl_->logits,new_root.logits_,"upload TurboAngle L1 delta logits");
    upload(impl_->embedding_trace->ptr,new_root.embedding_,"upload TurboAngle L1 delta embedding");
    for(std::size_t tap=0;tap<kTapLayers.size();++tap)
        upload(impl_->taps[tap]->ptr,new_root.taps_[tap],"upload TurboAngle L1 delta tap");
    cuda_check(cudaMemcpyAsync(impl_->position_device,&new_root.device_position_,sizeof(int),
        cudaMemcpyHostToDevice,stream),"upload TurboAngle L1 delta position");
    cuda_check(cudaStreamSynchronize(stream),"complete TurboAngle L1 delta rebase");
    position_=new_root.position_;impl_->tap_rows=new_root.tap_rows_;
    impl_->embedding_rows=new_root.embedding_rows_;impl_->last_rows=new_root.last_rows_;
    impl_->last_hidden_source=nullptr;
    impl_->last_hidden_source_rows=0;
    impl_->last_hidden_source_first_row=0;
    impl_->last_hidden_source_position=0;
    if(++impl_->last_hidden_generation==0)
        throw std::overflow_error("native MTP hidden generation exhausted");
    impl_->invalidate_native_mtp_hidden_capture();
    impl_->qkv_trace_valid=false;if(impl_->continuation) impl_->continuation->rows=0;
}

void Exl3TextContext::restore_host_state_impl(const Exl3ExactHostState& state,
    const Exl3TurboAngleWarmPages* warm, bool oscar, cudaStream_t stream) {
    impl_->join_repair(stream);
    Impl::require_host_kv_retirement_admission();
    require(!impl_->host_kv_failed,"host restore cannot reuse failed HostKV lineage");
    require(!impl_->graph_active && !impl_->graph_capture_active &&
            (oscar ? impl_->oscar!=nullptr : impl_->oscar==nullptr) &&
            (state.rope_offset_==0||(!oscar&&impl_->media_features)) &&
            (!impl_->oscar_only || oscar) &&
            (!impl_->transaction ||
                (!impl_->transaction->active && !impl_->transaction->rollback_required &&
                 (oscar ? (!impl_->transaction->host_kv &&
                           !impl_->transaction->device_kv) :
                          (impl_->transaction->host_kv ||
                           impl_->transaction->device_kv)))) &&
            (!warm || (oscar && warm->source_.get()==&state)) &&
            impl_->capture_taps && state.model_identity_ == impl_->model->host_state_identity &&
            state.position_ > 0 && state.position_ <= impl_->max_context &&
            state.native_extent_valid(impl_->max_context) &&
            state.tap_rows_ > 0 && state.tap_rows_ <= impl_->prefill_capacity &&
            state.embedding_rows_ > 0 && state.embedding_rows_ <= impl_->prefill_capacity,
            "exact host restore identity/extent/execution mismatch");
    auto restore_layer_observer=std::move(impl_->exact_restore_layer_observer_for_test);
    impl_->drain_continuation_graph_uses();
    impl_->invalidate_continuation_graphs("host restore changed graph state");
    impl_->validate_target_projection_execution(stream);
    // Admission is complete. Any failure during mutation keeps HostKV poisoned;
    // only complete restoration below may publish intact lineage again.
    if(impl_->host_kv.enabled) {
        auto source=state.shared_from_this();
        require(!impl_->execution_stream_owner || !stream || stream==impl_->owned_execution_stream,
            "HostKV restore stream differs from retained stream owner");
        impl_->retain_host_kv_forward_stream(stream);
        int device=-1;
        cuda_check(cudaGetDevice(&device),"HostKV restore allocation device");
        impl_->host_kv_restore_source=std::move(source);
        impl_->host_kv_restore_stream=stream;
        impl_->host_kv_restore_device=device;
        impl_->host_kv_failed=true;
    }
    impl_->resident_exact_state_id=0;
    impl_->exact_prefix_pages.clear();
    impl_->exact_prefix_position=0;
    if(oscar) impl_->oscar->reset();
    // State is immutable and constructible only by export; no untrusted plane shapes.
    const auto upload = [&](void* dst, const auto& src) {
        cuda_check(cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(src[0]),
                                  cudaMemcpyHostToDevice, stream), "restore exact host plane");
        cuda_check(cudaStreamSynchronize(stream), "complete exact host plane restore");
        if(std::exchange(impl_->host_kv_restore_failure_for_test,false))
            throw std::runtime_error("injected HostKV failure after restore upload");
    };
    int full = 0, gdn = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->full_layers[layer]) {
            impl_->full_layers[layer]->invalidate_retained_prefix();
            const auto restore_kv=[&](void* destination,bool key,int first,int rows) {
                for(const auto& page:state.kv_pages_) {
                    if(page->first<first || page->first>=first+rows) continue;
                    const auto& original=key?page->k[full]:page->v[full];
                    auto* dst=static_cast<std::uint16_t*>(destination)+static_cast<std::size_t>(page->first-first)*1024;
                    if(!warm || page->first+page->rows<=warm->first_) {upload(dst,original);continue;}
                    auto decoded=original;
                    const auto& packed=key ? warm->k_[full] : warm->v_[full];
                    const auto record_bytes=TurboAngle256::bytes(key);
                    for(int absolute=std::max(page->first,warm->first_);absolute<page->first+page->rows;++absolute)
                        for(int head=0;head<4;++head) {
                            const int row=absolute-warm->first_;
                            const auto values=TurboAngle256::decode(
                                std::span<const std::uint8_t>(packed.data()+(row*4+head)*record_bytes,record_bytes),key);
                            for(int d=0;d<256;++d) {
                                require(std::isfinite(values[d]) && std::abs(values[d])<=65504.0f,
                                        "TurboAngle reconstructed FP16 range");
                                decoded[((absolute-page->first)*4+head)*256+d]=float_to_half(values[d]);
                            }
                        }
                    upload(dst,decoded);
                }
            };
            if(impl_->oscar_only) {
                // Preserve append_kv_layer's original256-row partition. Host
                //64-token pages are gathered without changing rotation/aging.
                for(int first=0;first<state.position_;first+=256) {
                    const int rows=std::min(256,state.position_-first);
                    restore_kv(impl_->host_layer_k->ptr,true,first,rows);
                    restore_kv(impl_->host_layer_v->ptr,false,first,rows);
                    impl_->oscar->append_kv_layer(layer,
                        static_cast<const std::uint16_t*>(impl_->host_layer_k->ptr),
                        static_cast<const std::uint16_t*>(impl_->host_layer_v->ptr),rows,first,stream);
                    cuda_check(cudaStreamSynchronize(stream),"complete bounded OSCAR restore chunk");
                }
            } else if(!impl_->host_kv.enabled) {
                restore_kv(impl_->cache_k[layer]->ptr,true,0,state.position_);
                restore_kv(impl_->cache_v[layer]->ptr,false,0,state.position_);
                if(oscar) impl_->oscar->append_kv_layer(layer,
                    static_cast<const std::uint16_t*>(impl_->cache_k[layer]->ptr),
                    static_cast<const std::uint16_t*>(impl_->cache_v[layer]->ptr),state.position_,0,stream);
            }
            ++full;
        } else {
            impl_->gdn_layers[layer]->restore_host_state(state.recurrent_plane(gdn), state.convolution_[gdn], stream);
            ++gdn;
        }
        if(restore_layer_observer)restore_layer_observer(layer);
    }
    upload(impl_->logits, state.logits_);
    upload(impl_->embedding_trace->ptr, state.embedding_);
    for (std::size_t tap = 0; tap < kTapLayers.size(); ++tap) upload(impl_->taps[tap]->ptr, state.taps_[tap]);
    cuda_check(cudaMemcpyAsync(impl_->position_device, &state.device_position_, sizeof(int),
                              cudaMemcpyHostToDevice, stream), "restore exact host position");
    cuda_check(cudaStreamSynchronize(stream), "complete exact host restore");
    position_ = state.position_;
    impl_->tap_rows = state.tap_rows_;
    impl_->embedding_rows = state.embedding_rows_;
    impl_->last_rows = state.last_rows_;
    impl_->last_hidden_source = nullptr;
    impl_->last_hidden_source_rows = 0;
    impl_->last_hidden_source_first_row = 0;
    impl_->last_hidden_source_position = 0;
    if (++impl_->last_hidden_generation == 0)
        throw std::overflow_error("native MTP hidden generation exhausted");
    impl_->invalidate_native_mtp_hidden_capture();
    impl_->tap_generation.fetch_add(1,std::memory_order_release);
    impl_->rope_offset=state.rope_offset_;
    impl_->qkv_trace_valid = false;
    if (impl_->continuation) {
        impl_->continuation->rows = 0;
    }
    if(!oscar) {
        impl_->exact_prefix_pages=state.kv_pages_;
        impl_->exact_prefix_position=state.position_;
        impl_->host_kv_failed=false;
        impl_->host_kv_restore_source.reset();
    }
}

std::vector<float> Exl3TextContext::gdn_state_host(int layer, cudaStream_t stream) const {
    impl_->join_repair(stream);
    require(layer >= 0 && layer < kLayers && impl_->gdn_layers[layer] != nullptr,
            "E4B2 requested state from a non-GDN layer");
    const auto* gdn = impl_->gdn_layers[layer].get();
    std::vector<float> result(gdn->recurrent_state_bytes() / sizeof(float));
    cuda_check(cudaMemcpyAsync(result.data(), gdn->recurrent_state_device(),
                               gdn->recurrent_state_bytes(), cudaMemcpyDeviceToHost, stream),
               "download E4B2 GDN state");
    cuda_check(cudaStreamSynchronize(stream), "synchronize E4B2 GDN state");
    return result;
}

Exl3GdnContinuationHistoryView Exl3TextContext::gdn_continuation_history(
    int layer,int first_row,int rows) const {
    require(layer>=0 && layer<kLayers && impl_->gdn_layers[layer],
            "continuation history requested from a non-GDN layer");
    require(impl_->continuation && impl_->continuation->rows>0 &&
                impl_->last_rows==impl_->continuation->rows &&
                position_>=impl_->last_rows,
            "GDN continuation history has no current native rows");
    auto* gdn=impl_->gdn_layers[layer].get();
    std::shared_ptr<const void> storage_owner=impl_->gdn_layers[layer];
    return gdn->continuation_history_view(
        std::move(storage_owner),position_-impl_->last_rows,first_row,rows);
}

std::vector<std::uint16_t> Exl3TextContext::gdn_layer0_serial_for_test(
    std::span<const std::int64_t> token_ids, cudaStream_t stream) {
    require(token_ids.size() == 8 &&
                std::all_of(token_ids.begin(), token_ids.end(), [](std::int64_t token) {
                    return token >= 0 && token < kVocab;
                }),
            "staged GDN T0a requires exactly eight valid token IDs");
    require(impl_->gdn_layers[0] && impl_->model && impl_->model->embedding,
            "staged GDN T0a requires model layer 0 GDN");
    require(impl_->transaction && impl_->transaction->active &&
                impl_->transaction->fresh_snapshot &&
                !impl_->transaction->rollback_required &&
                impl_->transaction->stream == stream,
            "staged GDN T0a requires a fresh same-stream transaction checkpoint");
    require(!impl_->graph_active && !impl_->graph_capture_active &&
                (!impl_->oscar || impl_->oscar->graph_class() == 0),
            "staged GDN T0a requires eager execution");
    require(!impl_->target_projection_timing && !impl_->target_projection_observer,
            "staged GDN T0a rejects projection instrumentation");

    // The snapshot remains rollback-able, but this direct layer mutation must
    // not remain eligible as a fresh whole-context continuation snapshot.
    impl_->transaction->fresh_snapshot = false;
    impl_->transaction->prefix_available = false;
    impl_->transaction->rollback_required = true;
    impl_->transaction->attempted_rows = 0;
    cuda_check(cudaMemcpyAsync(impl_->token_ids, token_ids.data(), token_ids.size_bytes(),
                               cudaMemcpyHostToDevice, stream),
               "upload staged GDN T0a token IDs");
    embedding_lookup_kernel<<<(8 * kHidden + 255) / 256, 256, 0, stream>>>(
        impl_->token_ids, impl_->model->embedding, impl_->hidden_a, 8);
    cuda_check(cudaGetLastError(), "launch staged GDN T0a embedding");
    impl_->gdn_layers[0]->forward(impl_->hidden_a, impl_->hidden_b, 8, stream,
                                  false, true, false);
    std::vector<std::uint16_t> output(8ULL * kHidden);
    cuda_check(cudaMemcpyAsync(output.data(), impl_->hidden_b,
                               output.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
               "download serial GDN T0a output");
    cuda_check(cudaStreamSynchronize(stream), "synchronize serial GDN T0a output");
    return output;
}

std::array<std::vector<std::uint16_t>, 2>
Exl3TextContext::gdn_layer0_pair_staged_serial_for_test(
    Exl3TextContext& peer, std::span<const std::int64_t> token_ids,
    std::span<const std::int64_t> peer_token_ids, bool peer_first,
    cudaStream_t stream, int fail_after_published_lane,
    Exl3GdnStageOracleTelemetry* telemetry) {
    require(this != &peer && impl_->model == peer.impl_->model,
            "staged GDN T0a pair requires distinct contexts from one model");
    const auto valid_ids = [](std::span<const std::int64_t> ids) {
        return ids.size() == 8 &&
            std::all_of(ids.begin(), ids.end(), [](std::int64_t token) {
                return token >= 0 && token < kVocab;
            });
    };
    require(valid_ids(token_ids) && valid_ids(peer_token_ids),
            "staged GDN T0a pair requires two valid B8 token sets");
    const auto valid_context = [stream](const Exl3TextContext& context) {
        const auto& impl = context.impl_;
        return impl->gdn_layers[0] && impl->model && impl->model->embedding &&
            impl->transaction && impl->transaction->active &&
            impl->transaction->fresh_snapshot &&
            !impl->transaction->rollback_required &&
            impl->transaction->stream == stream &&
            !impl->graph_active && !impl->graph_capture_active &&
            (!impl->oscar || impl->oscar->graph_class() == 0) &&
            !impl->target_projection_timing && !impl->target_projection_observer;
    };
    require(valid_context(*this) && valid_context(peer),
            "staged GDN T0a pair requires fresh eager same-stream checkpoints");

    // Admission for both owners is complete. Consume freshness together before
    // either context's device state is touched; rollback remains valid even at
    // either deterministic publication failpoint.
    impl_->transaction->fresh_snapshot = false;
    impl_->transaction->prefix_available = false;
    impl_->transaction->rollback_required = true;
    impl_->transaction->attempted_rows = 0;
    peer.impl_->transaction->fresh_snapshot = false;
    peer.impl_->transaction->prefix_available = false;
    peer.impl_->transaction->rollback_required = true;
    peer.impl_->transaction->attempted_rows = 0;
    cuda_check(cudaMemcpyAsync(impl_->token_ids, token_ids.data(), token_ids.size_bytes(),
                               cudaMemcpyHostToDevice, stream),
               "upload staged GDN T0a first token IDs");
    cuda_check(cudaMemcpyAsync(peer.impl_->token_ids, peer_token_ids.data(),
                               peer_token_ids.size_bytes(), cudaMemcpyHostToDevice, stream),
               "upload staged GDN T0a peer token IDs");
    embedding_lookup_kernel<<<(8 * kHidden + 255) / 256, 256, 0, stream>>>(
        impl_->token_ids, impl_->model->embedding, impl_->hidden_a, 8);
    cuda_check(cudaGetLastError(), "launch staged GDN T0a first embedding");
    embedding_lookup_kernel<<<(8 * kHidden + 255) / 256, 256, 0, stream>>>(
        peer.impl_->token_ids, peer.impl_->model->embedding, peer.impl_->hidden_a, 8);
    cuda_check(cudaGetLastError(), "launch staged GDN T0a peer embedding");

    if (peer_first) {
        Exl3GdnLayer::forward_pair_staged_serial_for_test(
            *peer.impl_->gdn_layers[0], peer.impl_->hidden_a, peer.impl_->hidden_b,
            *impl_->gdn_layers[0], impl_->hidden_a, impl_->hidden_b, stream,
            fail_after_published_lane, telemetry);
    } else {
        Exl3GdnLayer::forward_pair_staged_serial_for_test(
            *impl_->gdn_layers[0], impl_->hidden_a, impl_->hidden_b,
            *peer.impl_->gdn_layers[0], peer.impl_->hidden_a, peer.impl_->hidden_b, stream,
            fail_after_published_lane, telemetry);
    }
    std::array<std::vector<std::uint16_t>, 2> outputs;
    outputs[0].resize(8ULL * kHidden);
    outputs[1].resize(8ULL * kHidden);
    cuda_check(cudaMemcpyAsync(outputs[0].data(), impl_->hidden_b,
                               outputs[0].size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
               "download staged GDN T0a first output");
    cuda_check(cudaMemcpyAsync(outputs[1].data(), peer.impl_->hidden_b,
                               outputs[1].size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
               "download staged GDN T0a peer output");
    cuda_check(cudaStreamSynchronize(stream), "synchronize staged GDN T0a outputs");
    return outputs;
}

std::vector<std::uint16_t> Exl3TextContext::full_attention_k_host(
    int layer, int position, cudaStream_t stream) const {
    require(layer >= 0 && layer < kLayers && impl_->cache_k[layer] != nullptr,
            "E4B2 requested KV from a non-full-attention layer");
    require(position >= 0 && position < impl_->max_context, "E4B2 KV position outside context");
    constexpr std::size_t kKvValues = static_cast<std::size_t>(kKVHeads) * kHeadDim;
    std::vector<std::uint16_t> result(kKvValues);
    const auto* source = static_cast<const std::uint16_t*>(impl_->cache_k[layer]->ptr) +
        static_cast<std::size_t>(position) * kKvValues;
    cuda_check(cudaMemcpyAsync(result.data(), source, result.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream), "download E4B2 KV entry");
    cuda_check(cudaStreamSynchronize(stream), "synchronize E4B2 KV entry");
    return result;
}

bool Exl3TextContext::try_enable_oscar_from_environment() {
    const char* flag = std::getenv("NINFER_OSCAR_EXL3");
    if (flag == nullptr || std::string(flag) != "1") return false;
    require(!impl_->host_kv.enabled,"authoritative host KV context cannot enable approximate OSCAR");
    require(!impl_->oscar,"OSCAR context strategy is already enabled");
    require(!impl_->transaction || !impl_->transaction->active,
            "P2 cannot replace OSCAR during an active target transaction");
    if (impl_->continuation) impl_->continuation->rows = 0;
    const char* asset = std::getenv("NINFER_OSCAR_ROTATION_ASSET_DIR");
    if (asset == nullptr || asset[0] == 0) {
        throw std::invalid_argument("E4C1 OSCAR enabled without NINFER_OSCAR_ROTATION_ASSET_DIR");
    }
    const char* manifest = std::getenv("NINFER_OSCAR_COMPAT_MANIFEST");
    const std::string manifest_path = (manifest != nullptr && manifest[0] != 0)
                                        ? manifest
                                        : "results/oscar/E4C1_EXL3_COMPATIBILITY_MANIFEST.json";
    Exl3OscarRotations rotations = exl3_oscar_load_rotations(asset, manifest_path);
    auto prepared=Exl3OscarContext::create(
        rotations,impl_->max_context,&impl_->oscar_telemetry);
    prepared->ensure_device_attributes();
    impl_->invalidate_continuation_graphs(
        "OSCAR strategy changed captured context membership");
    impl_->oscar=std::move(prepared);
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->full_layers[layer]) {
            impl_->full_layers[layer]->set_oscar(impl_->oscar.get(), layer);
        }
    }
    return true;
}

void Exl3TextContext::oscar_set_graph_class(int splits) {
    require(impl_->oscar != nullptr, "E4C1 OSCAR graph class without OSCAR");
    require(splits == 0 || splits == 16 || splits == 32 || splits == 64,
            "E4C1 OSCAR graph class must be 0/16/32/64");
    require(splits == 0 || !impl_->transaction || !impl_->transaction->active,
            "P2 OSCAR graph mode is unavailable during a target transaction");
    require(splits == 0 || !impl_->target_projection_timing,
            "OSCAR graph mode is unavailable while target projection timing is prepared");
    require(splits == 0 || !impl_->target_projection_observer,
            "OSCAR graph mode is unavailable while target projection observer is installed");
    if(impl_->oscar->graph_class()!=splits)
        impl_->invalidate_continuation_graphs(
            "OSCAR graph class changed continuation eligibility");
    if (impl_->continuation) impl_->continuation->rows = 0;
    impl_->oscar->set_graph_class(splits);
}

void Exl3TextContext::oscar_sync_device_state(cudaStream_t stream) {
    require(impl_->oscar != nullptr, "E4C1 OSCAR state sync without OSCAR");
    impl_->oscar->sync_device_state(stream);
}

void Exl3TextContext::oscar_routing_counts(std::uint64_t& oscar_full,
                                               std::uint64_t& ordinary_full) const {
    oscar_full = 0;
    ordinary_full = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        if (impl_->full_layers[layer]) {
            std::uint64_t oscar = 0, ordinary = 0;
            impl_->full_layers[layer]->dispatch_counts(oscar, ordinary);
            oscar_full += oscar;
            ordinary_full += ordinary;
        }
    }
}

bool Exl3TextContext::oscar_enabled() const noexcept {
    return impl_->oscar != nullptr;
}

const Exl3OscarTelemetry& Exl3TextContext::oscar_telemetry() const noexcept {
    return impl_->oscar_telemetry;
}

Exl3FullAttentionQKVHost Exl3TextContext::full_attention_qkv_host(
    int layer, cudaStream_t stream) const {
    require(layer >= 0 && layer < kLayers && impl_->full_layers[layer] != nullptr,
            "E4C1 requested QKV from a non-full-attention layer");
    require(impl_->qkv_trace_valid && impl_->last_rows > 0,
            "E4C1 native EXL3 QKV trace is invalid until a fresh forward completes");
    const auto trace = impl_->full_layers[layer]->trace();
    require(trace.q_rope != nullptr && trace.k_rope != nullptr && trace.v_projection != nullptr,
            "E4C1 native EXL3 QKV trace is incomplete");
    const std::size_t q_values = static_cast<std::size_t>(impl_->last_rows) * kQHeads * kHeadDim;
    const std::size_t kv_values = static_cast<std::size_t>(impl_->last_rows) * kKVHeads * kHeadDim;
    Exl3FullAttentionQKVHost result;
    result.layer = layer;
    result.rows = impl_->last_rows;
    result.q_rope.resize(q_values);
    result.k_rope.resize(kv_values);
    result.v_projection.resize(kv_values);
    cuda_check(cudaMemcpyAsync(result.q_rope.data(), trace.q_rope,
                               q_values * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, stream),
               "download E4C1 EXL3 Q trace");
    cuda_check(cudaMemcpyAsync(result.k_rope.data(), trace.k_rope,
                               kv_values * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, stream),
               "download E4C1 EXL3 K trace");
    cuda_check(cudaMemcpyAsync(result.v_projection.data(), trace.v_projection,
                               kv_values * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, stream),
               "download E4C1 EXL3 V trace");
    cuda_check(cudaStreamSynchronize(stream), "synchronize E4C1 EXL3 QKV capture");
    return result;
}

void Exl3TextContext::copy_tap_row_to_device(int layer, std::uint16_t* dst,
                                                 cudaStream_t stream) const {
    const int tap = impl_->tap_index(layer);
    require(tap >= 0, "E5A2 requested a tap row from a non-tap layer");
    require(impl_->capture_taps && impl_->tap_rows > 0 && impl_->taps[tap] != nullptr,
            "E5A2 tap rows were not captured");
    require(dst != nullptr, "E5A2 tap copy needs a device destination");
    const auto* source = static_cast<const std::uint16_t*>(impl_->taps[tap]->ptr) +
        static_cast<std::size_t>(impl_->tap_rows - 1) * kHidden;
    cuda_check(cudaMemcpyAsync(dst, source, kHidden * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream),
               "E5A2 copy tap row");
}

int Exl3TextContext::captured_tap_rows() const noexcept {
    return impl_->capture_taps ? impl_->tap_rows : 0;
}

int Exl3TextContext::captured_embedding_rows() const noexcept {
    return impl_->capture_taps ? impl_->embedding_rows : 0;
}

int Exl3TextContext::last_forward_rows() const noexcept {
    return impl_->last_rows;
}

int Exl3TextContext::device_position_host(cudaStream_t stream) const {
    int result = -1;
    cuda_check(cudaMemcpyAsync(&result, impl_->position_device, sizeof(result),
                               cudaMemcpyDeviceToHost, stream),
               "download P2 target device position");
    cuda_check(cudaStreamSynchronize(stream), "synchronize P2 target device position");
    return result;
}

std::vector<std::uint16_t> Exl3TextContext::gdn_physical_conv_host(
    int layer, cudaStream_t stream) const {
    require(layer >= 0 && layer < kLayers && impl_->gdn_layers[layer] != nullptr,
            "P2 requested physical convolution state from a non-GDN layer");
    const auto* gdn = impl_->gdn_layers[layer].get();
    std::vector<std::uint16_t> result(
        gdn->physical_conv_state_bytes() / sizeof(std::uint16_t));
    cuda_check(cudaMemcpyAsync(result.data(), gdn->physical_conv_state_device(),
                               gdn->physical_conv_state_bytes(),
                               cudaMemcpyDeviceToHost, stream),
               "download P2 GDN physical convolution state");
    cuda_check(cudaStreamSynchronize(stream),
               "synchronize P2 GDN physical convolution state");
    return result;
}

std::shared_ptr<const void> Exl3TextContext::gdn_coefficient_owner_for_test(
    int layer) const {
    require(layer>=0 && layer<kLayers && impl_->gdn_layers[layer],
            "requested immutable coefficients from a non-GDN layer");
    return impl_->gdn_layers[layer]->immutable_coefficient_owner();
}

const void* Exl3TextContext::gdn_recurrent_state_identity_for_test(int layer) const {
    require(layer>=0 && layer<kLayers && impl_->gdn_layers[layer],
            "requested recurrent identity from a non-GDN layer");
    return impl_->gdn_layers[layer]->recurrent_state_device();
}

const void* Exl3TextContext::gdn_convolution_state_identity_for_test(int layer) const {
    require(layer>=0 && layer<kLayers && impl_->gdn_layers[layer],
            "requested convolution identity from a non-GDN layer");
    return impl_->gdn_layers[layer]->physical_conv_state_device();
}

std::vector<std::byte> Exl3TextContext::oscar_live_state_host_for_test(
    cudaStream_t stream) const {
    require(impl_->oscar != nullptr,
            "P2 requested live OSCAR state without OSCAR");
    return impl_->oscar->live_state_host_for_test(stream);
}

Exl3OscarGraphStateObservation
Exl3TextContext::oscar_graph_state_host_for_test(cudaStream_t stream) const {
    require(impl_->oscar != nullptr,
            "P2 requested graph live OSCAR state without OSCAR");
    require(impl_->graph_active,
            "P2 requested graph live OSCAR state before active graph replay");
    return impl_->oscar->graph_live_state_host_for_test(stream);
}

std::array<std::vector<std::uint16_t>,5> Exl3TextContext::exact_tap_rows_host(cudaStream_t stream) const {
    const bool transaction_taps=!impl_->transaction || !impl_->transaction->rollback_required;
    require(transaction_taps && !impl_->graph_active && !impl_->graph_capture_active &&
                impl_->capture_taps && impl_->tap_rows > 0,
            "authoritative tap export requires ordinary eager captured state");
    require(!impl_->host_kv_failed,"authoritative tap export after failed host KV forward");
    std::array<std::vector<std::uint16_t>,5> result;
    for(int tap=0;tap<5;++tap) {
        result[tap].resize(static_cast<std::size_t>(impl_->tap_rows)*kHidden);
        cuda_check(cudaMemcpyAsync(result[tap].data(),impl_->taps[tap]->ptr,
            result[tap].size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost,stream),
            "export authoritative tap rows");
    }
    cuda_check(cudaStreamSynchronize(stream),"complete authoritative tap rows");
    return result;
}

void Exl3CommittedTapSegment::validate() const {
    const auto sum=[](int first,int count) {
        if(first<0 || count<1 || first>std::numeric_limits<int>::max()-count)
            throw std::invalid_argument("committed tap segment range");
        return first+count;
    };
    if(!current() || !acquisition || !execution || root_position<0 ||
       source_position<root_position || logical_first<root_position ||
       rows<1 || rows>16 || destination_first<0 || destination_first>16-rows ||
       source_first<0 || source_first>16-rows || attempt_rows<1 ||
       source_partition_rows<1 || (correction && !repair))
        throw std::invalid_argument("committed tap segment stale/incomplete owner");
    for(auto plane:planes)if(!plane)
        throw std::invalid_argument("committed tap segment missing plane");
    const auto logical_end=sum(logical_first,rows);
    const auto attempt_end=sum(attempt_first,attempt_rows);
    const auto partition_end=sum(source_partition_first,source_partition_rows);
    if(logical_first<attempt_first || logical_end>attempt_end ||
       logical_first<source_partition_first || logical_end>partition_end ||
       logical_first!=source_partition_first+source_first)
        throw std::invalid_argument("committed tap segment row/partition mismatch");
}

Exl3CommittedTapSegment Exl3TextContext::committed_tap_segment(
    const Exl3CommittedTapBinding& binding,int source_first,int rows,
    int logical_first,int destination_first,int attempt_first,int attempt_rows,
    int source_partition_first,int source_partition_rows,
    bool repair,bool correction) const {
    const bool transaction_taps=!impl_->transaction || !impl_->transaction->rollback_required;
    require(binding && binding.context_owner.get()==static_cast<const void*>(this) &&
                transaction_taps && !impl_->graph_active && !impl_->graph_capture_active &&
                !impl_->host_kv_failed && impl_->capture_taps &&
                impl_->tap_rows>0 && source_first>=0 && rows>0 &&
                source_first<=impl_->tap_rows-rows,
            "committed tap segment source owner/range");
    Exl3CommittedTapSegment result;
    result.owner=binding.context_owner;result.model_identity=model_identity();
    result.root_revision=binding.root_revision;result.root_position=binding.root_position;
    for(std::size_t plane=0;plane<result.planes.size();++plane)
        result.planes[plane]=static_cast<const std::uint16_t*>(impl_->taps[plane]->ptr);
    result.current_generation=&impl_->tap_generation;
    result.generation=impl_->tap_generation.load(std::memory_order_acquire);
    result.acquisition=binding.acquisition;result.execution=binding.execution;
    result.source_position=position_-impl_->tap_rows;
    result.source_first=source_first;result.rows=rows;
    result.logical_first=logical_first;result.destination_first=destination_first;
    result.attempt_first=attempt_first;result.attempt_rows=attempt_rows;
    result.source_partition_first=source_partition_first;
    result.source_partition_rows=source_partition_rows;
    result.repair=repair;result.correction=correction;
    result.validate();return result;
}

void Exl3TextContext::copy_tap_rows_to_device(int layer, int first_row, std::uint16_t* dst,
                                                 int rows, cudaStream_t stream) const {
    const int tap = impl_->tap_index(layer);
    require(tap >= 0, "E5A2 requested tap rows from a non-tap layer");
    require(rows > 0 && first_row >= 0 && dst != nullptr,
            "E5A2 tap range copy needs a valid range and destination");
    require(impl_->capture_taps && impl_->taps[tap] != nullptr &&
            impl_->tap_rows >= first_row + rows,
            "E5A2 tap range was not captured");
    const auto* source = static_cast<const std::uint16_t*>(impl_->taps[tap]->ptr) +
        static_cast<std::size_t>(first_row) * kHidden;
    cuda_check(cudaMemcpyAsync(dst, source,
                               static_cast<std::size_t>(rows) * kHidden * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream),
               "E5A2 copy tap rows");
}

void Exl3TextContext::embed_token_to_device(std::int64_t token_id, std::uint16_t* dst,
                                                 cudaStream_t stream) const {
    require(token_id >= 0 && token_id < kVocab, "E5A2 token id outside vocabulary");
    require(dst != nullptr && impl_->draft_token_id != nullptr,
            "E5A2 embed needs a device destination");
    require(impl_->model != nullptr && impl_->model->embedding != nullptr,
            "E5A2 target embeddings are unavailable");
    cuda_check(cudaMemcpyAsync(impl_->draft_token_id, &token_id, sizeof(token_id),
                               cudaMemcpyHostToDevice, stream),
               "E5A2 upload draft token id");
    embedding_lookup_kernel<<<(kHidden + 255) / 256, 256, 0, stream>>>(
        impl_->draft_token_id, impl_->model->embedding,
                                              dst, 1);
    cuda_check(cudaGetLastError(), "launch E5A2 token embedding");
}

void Exl3TextContext::embed_tokens_to_device(const std::int64_t* ids, std::uint16_t* dst,
                                                 int rows, cudaStream_t stream) const {
    require(rows > 0 && dst != nullptr && ids != nullptr,
            "E5A2 embed needs device ids and a device destination");
    require(impl_->model != nullptr && impl_->model->embedding != nullptr,
            "E5A2 target embeddings are unavailable");
    embedding_lookup_kernel<<<(rows * kHidden + 255) / 256, 256, 0, stream>>>(
        ids, impl_->model->embedding, dst, rows);
    cuda_check(cudaGetLastError(), "launch E5A2 token embeddings");
}

const Exl3CudaLinearWeights& Exl3TextContext::target_lm_head_weights() const noexcept {
    return impl_->model->lm_head;
}

const Exl3CudaLinearMetadata& Exl3TextContext::target_lm_head_metadata() const noexcept {
    return impl_->model->lm_head_metadata;
}

const std::uint16_t* Exl3TextContext::target_embedding() const noexcept {
    return impl_->model == nullptr ? nullptr : impl_->model->embedding;
}

void Exl3TextContext::prepare_native_mtp_hidden_capture(int capacity) {
    require(impl_->model != nullptr && impl_->model->final_norm != nullptr,
        "native MTP hidden capture needs a loaded target model");
    require(capacity > 0 && capacity <= impl_->max_context,
        "native MTP hidden capture capacity is outside the target context");
    require(position_ == 0 && impl_->last_rows == 0,
        "native MTP hidden capture must be prepared before target inference");
    require(!impl_->graph_active && !impl_->graph_capture_active &&
                (!impl_->transaction || !impl_->transaction->active),
        "native MTP hidden capture setup requires a pristine eager context");
    if (impl_->native_mtp_hidden_capture) {
        require(impl_->native_mtp_hidden_capture_capacity == capacity,
            "native MTP hidden capture capacity cannot change after preparation");
        require(impl_->native_mtp_hidden_capture_valid_rows == 0,
            "native MTP hidden capture already contains rows");
        return;
    }
    const std::size_t bytes = static_cast<std::size_t>(capacity) * kHidden *
        sizeof(std::uint16_t);
    const auto next_impl = Exl3LinearWorkspaceRequirements::append_owned_bytes(
        impl_->persistent_bytes, bytes);
    const auto next_public = Exl3LinearWorkspaceRequirements::append_owned_bytes(
        persistent_bytes_, bytes);
    auto capture = impl_->make_private_allocation(bytes,
        "allocate native MTP target hidden capture");
    require(capture && capture->ptr && capture->bytes == bytes,
        "native MTP hidden capture allocation extent mismatch");
    impl_->native_mtp_hidden_capture = std::move(capture);
    impl_->native_mtp_hidden_capture_model_identity =
        impl_->model->host_state_identity;
    impl_->native_mtp_hidden_capture_capacity = capacity;
    impl_->native_mtp_hidden_capture_first_position = -1;
    impl_->native_mtp_hidden_capture_valid_rows = 0;
    if (++impl_->native_mtp_hidden_capture_generation == 0)
        throw std::overflow_error("native MTP hidden capture generation exhausted");
    impl_->persistent_bytes = next_impl;
    persistent_bytes_ = next_public;
}

bool Exl3TextContext::native_mtp_hidden_capture_prepared() const noexcept {
    return impl_->native_mtp_hidden_capture_active();
}

int Exl3TextContext::native_mtp_hidden_capture_capacity() const noexcept {
    return impl_->native_mtp_hidden_capture_active()
        ? impl_->native_mtp_hidden_capture_capacity : 0;
}

int Exl3TextContext::native_mtp_hidden_capture_rows() const noexcept {
    return impl_->native_mtp_hidden_capture_active()
        ? impl_->native_mtp_hidden_capture_valid_rows : 0;
}

int Exl3TextContext::native_mtp_hidden_capture_first_position() const noexcept {
    return impl_->native_mtp_hidden_capture_active()
        ? impl_->native_mtp_hidden_capture_first_position : -1;
}

std::uint64_t Exl3TextContext::native_mtp_hidden_capture_generation() const noexcept {
    return impl_->native_mtp_hidden_capture_active()
        ? impl_->native_mtp_hidden_capture_generation : 0;
}

Exl3MtpTargetHiddenHandoff Exl3TextContext::target_hidden_handoff(
    int row, cudaStream_t stream) const {
    const bool use_capture = impl_->native_mtp_hidden_capture_active() &&
        row >= 0 && row < impl_->native_mtp_hidden_capture_valid_rows;
    const std::uint16_t* source = nullptr;
    int source_position = 0;
    std::uint64_t source_generation = 0;
    std::shared_ptr<const void> source_identity;
    if (use_capture) {
        source = static_cast<const std::uint16_t*>(
            impl_->native_mtp_hidden_capture->ptr) +
            static_cast<std::size_t>(row) * kHidden;
        source_position = impl_->native_mtp_hidden_capture_first_position + row;
        source_generation = impl_->native_mtp_hidden_capture_generation;
        source_identity = impl_->native_mtp_hidden_capture_model_identity;
    } else {
        require(impl_->model != nullptr && impl_->last_hidden_source != nullptr &&
                    impl_->last_hidden_source_rows > 0 &&
                    row >= 0 && row < impl_->last_hidden_source_rows &&
                    impl_->last_hidden_generation != 0,
                "native MTP target hidden is unavailable until a target forward completes");
        require(impl_->last_hidden_source_position >= 0 &&
                    impl_->last_hidden_source_position + impl_->last_hidden_source_first_row + row >= 0 &&
                    impl_->last_hidden_source_position + impl_->last_hidden_source_first_row + row <
                        impl_->last_hidden_source_position + impl_->last_rows,
                "native MTP target hidden position is invalid");
        source = impl_->last_hidden_source + static_cast<std::size_t>(row) * kHidden;
        source_position = impl_->last_hidden_source_position +
            impl_->last_hidden_source_first_row + row;
        source_generation = impl_->last_hidden_generation;
        source_identity = model_identity();
    }

    // Do not expose hidden_a/hidden_b directly: the next target forward is
    // allowed to reuse either ping-pong buffer. The copied allocation is
    // independently owned until every stream-ordered MTP consumer releases
    // its handoff owner.
    auto* mutable_impl = const_cast<Impl*>(impl_.get());
    const std::size_t bytes = static_cast<std::size_t>(kHidden) *
                              sizeof(std::uint16_t);
    auto allocation = std::shared_ptr<DeviceAllocation>(
        mutable_impl->make_private_allocation(bytes,
                                              "allocate native MTP target hidden")
            .release(),
        [](DeviceAllocation* value) noexcept { delete value; });
    cuda_check(cudaMemcpyAsync(
                   allocation->ptr,
                   source,
                   bytes, cudaMemcpyDeviceToDevice, stream),
               "copy native MTP target hidden handoff");

    Exl3MtpTargetHiddenHandoff result;
    result.owner = std::shared_ptr<const void>(allocation,
                                               static_cast<const void*>(allocation.get()));
    result.model_identity = std::move(source_identity);
    result.hidden = {reinterpret_cast<std::uintptr_t>(allocation->ptr), bytes,
                     kExl3MtpHiddenWidth, 1};
    result.hidden_generation = source_generation;
    result.position = static_cast<std::int64_t>(source_position);
    return result;
}

std::vector<Exl3MtpTargetHiddenHandoff> Exl3TextContext::target_hidden_handoffs(
    int first_row, int rows, cudaStream_t stream) const {
    const bool use_capture = impl_->native_mtp_hidden_capture_active() &&
        rows > 0 && first_row >= 0 &&
        first_row <= impl_->native_mtp_hidden_capture_valid_rows - rows;
    const std::uint16_t* source = nullptr;
    int source_position = 0;
    std::uint64_t source_generation = 0;
    std::shared_ptr<const void> source_identity;
    if (use_capture) {
        source = static_cast<const std::uint16_t*>(
            impl_->native_mtp_hidden_capture->ptr) +
            static_cast<std::size_t>(first_row) * kHidden;
        source_position = impl_->native_mtp_hidden_capture_first_position + first_row;
        source_generation = impl_->native_mtp_hidden_capture_generation;
        source_identity = impl_->native_mtp_hidden_capture_model_identity;
    } else {
        require(impl_->model != nullptr && impl_->last_hidden_source != nullptr &&
                    impl_->last_hidden_source_rows > 0 && rows > 0 && first_row >= 0 &&
                    first_row + rows <= impl_->last_hidden_source_rows &&
                    impl_->last_hidden_generation != 0,
                "native MTP target hidden range is unavailable until a target forward completes");
        require(impl_->last_hidden_source_position >= 0 &&
                    impl_->last_hidden_source_position + impl_->last_hidden_source_first_row +
                            first_row >= 0 &&
                    impl_->last_hidden_source_position + impl_->last_hidden_source_first_row +
                            first_row + rows <= impl_->last_hidden_source_position + impl_->last_rows,
                "native MTP target hidden range position is invalid");
        source = impl_->last_hidden_source + static_cast<std::size_t>(first_row) * kHidden;
        source_position = impl_->last_hidden_source_position +
            impl_->last_hidden_source_first_row + first_row;
        source_generation = impl_->last_hidden_generation;
        source_identity = model_identity();
    }

    auto* mutable_impl = const_cast<Impl*>(impl_.get());
    const std::size_t row_bytes = static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t);
    const std::size_t bytes = static_cast<std::size_t>(rows) * row_bytes;
    auto allocation = std::shared_ptr<DeviceAllocation>(
        mutable_impl->make_private_allocation(bytes, "allocate native MTP target hidden rows")
            .release(),
        [](DeviceAllocation* value) noexcept { delete value; });
    cuda_check(cudaMemcpyAsync(
                   allocation->ptr,
                   source,
                   bytes, cudaMemcpyDeviceToDevice, stream),
               "copy native MTP target hidden rows");
    const auto owner = std::shared_ptr<const void>(
        allocation, static_cast<const void*>(allocation.get()));
    const auto target_identity = std::move(source_identity);
    std::vector<Exl3MtpTargetHiddenHandoff> result;
    result.reserve(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        auto* data = static_cast<std::uint16_t*>(allocation->ptr) +
                     static_cast<std::size_t>(row) * kHidden;
        result.push_back({
            owner,
            target_identity,
            {reinterpret_cast<std::uintptr_t>(data), row_bytes,
             kExl3MtpHiddenWidth, 1},
            source_generation,
            static_cast<std::int64_t>(source_position + row)});
    }
    return result;
}

std::shared_ptr<Exl3MtpPrefixState> Exl3TextContext::create_native_mtp_prefix_state(
    std::uint32_t kv_capacity) const {
    require(impl_->model != nullptr && impl_->model->embedding != nullptr &&
                impl_->model->lm_head.trellis != nullptr,
            "native MTP target shared weights are unavailable");
    const auto target_owner = model_identity();
    require(target_owner != nullptr, "native MTP target model identity is unavailable");
    const auto model_owner = std::shared_ptr<const void>(
        impl_->model, static_cast<const void*>(impl_->model.get()));
    Exl3MtpSharedTargetBinding shared_target;
    shared_target.token_embedding = {
        reinterpret_cast<std::uintptr_t>(impl_->model->embedding),
        static_cast<std::size_t>(kVocab) * static_cast<std::size_t>(kHidden) *
            sizeof(std::uint16_t),
        kVocab, kHidden};
    // The descriptor is a lifetime witness; the actual H6 payload is bound by
    // lm_head_weights below because EXL3 stores trellis/scales in four planes.
    shared_target.lm_head = {
        reinterpret_cast<std::uintptr_t>(impl_->model->lm_head.trellis), sizeof(std::uint16_t),
        1, 1};
    shared_target.owner = model_owner;
    shared_target.model_identity = target_owner;
    shared_target.lm_head_weights = impl_->model->lm_head;
    shared_target.lm_head_metadata = impl_->model->lm_head_metadata;
    const auto numeric_identity = static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(target_owner.get()));
    return Exl3MtpPrefixState::materialize(impl_->model->collection,
                                            std::move(shared_target), numeric_identity,
                                            kv_capacity);
}

std::vector<float> Exl3TextContext::embedding_host(cudaStream_t stream) const {
    require(impl_->capture_taps && impl_->embedding_rows > 0, "E4A embedding trace was not captured");
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(impl_->embedding_rows) * kHidden);
    cuda_check(cudaMemcpyAsync(raw.data(), impl_->embedding_trace->ptr,
                               raw.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream), "download E4A embedding output");
    cuda_check(cudaStreamSynchronize(stream), "synchronize E4A embedding output");
    std::vector<float> result;
    result.reserve(raw.size());
    for (const auto value : raw) result.push_back(half_to_float(value));
    return result;
}

std::vector<std::uint16_t> Exl3TextContext::embedding_bits_host_for_test(
    cudaStream_t stream) const {
    require(impl_->capture_taps && impl_->embedding_rows > 0,
            "P2 embedding bits were not captured");
    std::vector<std::uint16_t> result(
        static_cast<std::size_t>(impl_->embedding_rows) * kHidden);
    cuda_check(cudaMemcpyAsync(result.data(), impl_->embedding_trace->ptr,
                               result.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost, stream),
               "download P2 embedding bits");
    cuda_check(cudaStreamSynchronize(stream),
               "synchronize P2 embedding bits");
    return result;
}


} // namespace ninfer::exl3
